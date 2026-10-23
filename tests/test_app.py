# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_app.py - API endpoint tests for FastAPI backend."""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app import app, STATE
from engine.clean import DeflickerMetrics


@pytest.fixture
def client():
    return TestClient(app)


def test_api_doctor(client):
    res = client.get("/api/doctor")
    assert res.status_code == 200
    data = res.json()
    assert "all_ok" in data
    assert "items" in data
    assert "comfy_flavor" in data


def test_api_serve_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "MatteDesk" in res.text


def test_api_project_lifecycle(client, tmp_path):
    work_dir = tmp_path / "shot_test"
    work_dir.mkdir()
    plates_dir = work_dir / "plates"
    plates_dir.mkdir()

    # Create 3 dummy frames
    for f in range(1, 4):
        img_path = plates_dir / f"test_{f:04d}.png"
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

    manifest_data = {
        "shot": "TEST_LIFECYCLE",
        "fps": 24,
        "first": 1,
        "last": 3,
        "plate_w": 200,
        "plate_h": 100,
        "colorspace": "sRGB",
        "source_type": "img_seq",
        "bit_depth": 8,
        "plates": {
            "beauty": str(plates_dir / "test_####.png")
        },
        "work": str(work_dir)
    }

    manifest_file = work_dir / "shot_manifest_TEST_LIFECYCLE.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Open Project
    res = client.post("/api/project/open", json={"manifest_path": str(manifest_file)})
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["success"] is True
    assert res_data["manifest"]["shot"] == "TEST_LIFECYCLE"

    # 2. Get Current Project
    res_curr = client.get("/api/project/current")
    assert res_curr.status_code == 200
    curr_data = res_curr.json()
    assert curr_data["loaded"] is True
    assert curr_data["manifest"]["shot"] == "TEST_LIFECYCLE"

    # 3. Generate Proxies
    res_prox = client.post("/api/project/generate_proxies?plate=beauty")
    assert res_prox.status_code == 200
    assert res_prox.json()["rendered_count"] == 3

    # 4. Fetch Frame
    res_frame = client.get("/api/frame/2")
    assert res_frame.status_code == 200
    assert res_frame.headers["content-type"] == "image/jpeg"


def test_api_plate_align_lifecycle(client, tmp_path):
    work_dir = tmp_path / "shot_align"
    work_dir.mkdir()
    plates_dir = work_dir / "plates"
    plates_dir.mkdir()

    # Create dummy beauty and backlit frames
    for f in range(1, 4):
        b_img = np.zeros((100, 100, 3), dtype=np.float32)
        b_img[30:70, 30:70] = 1.0
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.exr"), b_img)

        # Aux with 2px shift
        a_img = np.zeros((100, 100, 3), dtype=np.float32)
        a_img[32:72, 30:70] = 1.0
        cv2.imwrite(str(plates_dir / f"x2_{f:04d}.exr"), a_img)

    manifest_data = {
        "shot": "TEST_ALIGN",
        "fps": 24,
        "first": 1,
        "last": 3,
        "plate_w": 100,
        "plate_h": 100,
        "colorspace": "ACES2065-1",
        "plates": {
            "beauty": str(plates_dir / "b_####.exr"),
            "backlit": str(plates_dir / "x2_####.exr")
        },
        "work": str(work_dir)
    }

    manifest_file = work_dir / "shot_manifest_TEST_ALIGN.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    manifest_before = manifest_file.read_bytes()

    # Open project
    client.post("/api/project/open", json={"manifest_path": str(manifest_file)})

    # Measure alignment
    res_m = client.post("/api/plate/align/measure", json={"plate": "backlit", "aux_plate_type": "backlit"})
    assert res_m.status_code == 200
    m_data = res_m.json()
    assert "residual_median" in m_data
    assert "recommendation" in m_data

    # Export Nuke script
    res_nk = client.post("/api/plate/align/export_nk", json={"plate": "backlit"})
    assert res_nk.status_code == 200
    nk_data = res_nk.json()
    assert nk_data["success"] is True
    assert "Group {" in nk_data["script"]

    # Check report endpoint
    res_rep = client.get("/api/plate/align/report?plate=backlit")
    assert res_rep.status_code == 200
    assert res_rep.json()["exists"] is True

    # Test apply (requires confirmed=True)
    res_fail = client.post("/api/plate/align/apply", json={"plate": "backlit", "confirmed": False})
    assert res_fail.status_code == 400

    res_apply = client.post("/api/plate/align/apply", json={"plate": "backlit", "confirmed": True})
    assert res_apply.status_code == 200
    assert res_apply.json()["success"] is True
    assert res_apply.json()["aligned_count"] == 3
    assert res_apply.json()["aligned_dir"].replace("\\", "/").endswith("/plates_align")

    # Aligned plate registered in project.json, never in the user's manifest
    assert manifest_file.read_bytes() == manifest_before
    pj = json.loads((work_dir / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    assert "backlit_align" in pj["plates_added"]
    assert "backlit_align" in client.get("/api/project/current").json()["manifest"]["plates"]


def test_api_plate_management_lifecycle(client, tmp_path):
    work_dir = tmp_path / "shot_plates_mgmt"
    work_dir.mkdir()
    plates_dir = work_dir / "plates"
    plates_dir.mkdir()

    for f in range(1, 4):
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.png"), np.zeros((50, 50, 3), dtype=np.uint8))

    manifest_data = {
        "shot": "TEST_MGMT",
        "fps": 24,
        "first": 1,
        "last": 3,
        "plate_w": 50,
        "plate_h": 50,
        "colorspace": "sRGB",
        "source_type": "img_seq",
        "plates": {
            "beauty": str(plates_dir / "b_####.png")
        },
        "work": str(work_dir)
    }

    manifest_file = work_dir / "shot_manifest_TEST_MGMT.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Open project
    client.post("/api/project/open", json={"manifest_path": str(manifest_file)})

    # 2. Get plates list
    res_list = client.get("/api/project/plates")
    assert res_list.status_code == 200
    plates = res_list.json()["plates"]
    assert len(plates) >= 1
    assert any(p["name"] == "beauty" for p in plates)

    # 3. Create dummy cleanplate files and inspect path
    cp_dir = work_dir / "cleanplate_files"
    cp_dir.mkdir()
    for f in range(1, 4):
        cv2.imwrite(str(cp_dir / f"clean_{f:04d}.png"), np.zeros((50, 50, 3), dtype=np.uint8))

    res_insp = client.post("/api/project/plate/inspect", json={"path": str(cp_dir)})
    assert res_insp.status_code == 200
    info = res_insp.json()["info"]
    assert info["exists"] is True
    assert info["frame_count"] == 3
    assert "####" in info["pattern"]

    # 4. Add cleanplate to project
    res_update = client.post("/api/project/plate/update", json={
        "plate_name": "cleanplate",
        "pattern": str(cp_dir),
        "generate_proxies": True
    })
    assert res_update.status_code == 200
    up_data = res_update.json()
    assert up_data["success"] is True
    assert "cleanplate" in up_data["manifest"]["plates"]
    assert up_data["rendered_proxies"] == 3

    # Check that plates list now reflects cleanplate
    res_list2 = client.get("/api/project/plates")
    plates2 = res_list2.json()["plates"]
    cp_entry = next(p for p in plates2 if p["name"] == "cleanplate")
    assert cp_entry["exists"] is True
    assert cp_entry["proxy_count"] == 3

    # 5. Add custom plate
    res_custom = client.post("/api/project/plate/update", json={
        "plate_name": "fg_elements",
        "pattern": "D:/dummy/fg_####.exr",
        "generate_proxies": False
    })
    assert res_custom.status_code == 200
    assert "fg_elements" in res_custom.json()["manifest"]["plates"]

    # 6. Delete custom plate
    res_del = client.delete("/api/project/plate/fg_elements")
    assert res_del.status_code == 200
    assert "fg_elements" not in res_del.json()["manifest"]["plates"]

    # 7. Attempting to delete beauty must fail
    res_del_beauty = client.delete("/api/project/plate/beauty")
    assert res_del_beauty.status_code == 400

    # None of the plate edits touched the user's manifest
    assert json.loads(manifest_file.read_text(encoding="utf-8")) == manifest_data


def test_api_inspect_working_dir_and_open_or_create(tmp_path, monkeypatch):
    client = TestClient(app)

    work_dir = tmp_path / "SHOT_INSPECT_TEST"
    work_dir.mkdir()

    # 1. Inspect clean directory (no manifest yet)
    res_insp1 = client.post("/api/project/inspect_working_dir", json={"dir_path": str(work_dir)})
    assert res_insp1.status_code == 200
    info1 = res_insp1.json()["info"]
    assert info1["exists"] is True
    assert info1["manifest_found"] is False

    # 2. Put beauty frames in a subfolder
    beauty_dir = work_dir / "plate_beauty"
    beauty_dir.mkdir()
    for f in range(1, 4):
        cv2.imwrite(str(beauty_dir / f"SHOT_010_beauty_{f:04d}.png"), np.zeros((64, 64, 3), dtype=np.uint8))

    # Inspect again -> auto detects beauty plate
    res_insp2 = client.post("/api/project/inspect_working_dir", json={"dir_path": str(work_dir)})
    assert res_insp2.status_code == 200
    info2 = res_insp2.json()["info"]
    assert "beauty" in info2["detected_plates"]

    # 3. Create project via open_or_create
    res_open = client.post("/api/project/open_or_create", json={
        "work_dir": str(work_dir),
        "plates": {
            "beauty": str(beauty_dir),
            "cleanplate": "D:/dummy/clean_####.exr"
        },
        "shot_name": "SHOT_010"
    })
    assert res_open.status_code == 200
    data = res_open.json()
    assert data["success"] is True
    assert data["manifest"]["shot"] == "SHOT_010"
    assert "beauty" in data["manifest"]["plates"]
    assert "cleanplate" in data["manifest"]["plates"]

    # 4. Verify English labels in /api/project/plates
    res_plates = client.get("/api/project/plates")
    assert res_plates.status_code == 200
    plates_list = res_plates.json()["plates"]
    beauty_entry = next(p for p in plates_list if p["name"] == "beauty")
    assert beauty_entry["label"] == "Beauty Plate"
    clean_entry = next(p for p in plates_list if p["name"] == "cleanplate")
    assert clean_entry["label"] == "Clean Plate"

    # 5. Inspecting the directory now finds the manifest
    res_insp3 = client.post("/api/project/inspect_working_dir", json={"dir_path": str(work_dir)})
    assert res_insp3.status_code == 200
    info3 = res_insp3.json()["info"]
    assert info3["manifest_found"] is True
    assert info3["shot"] == "SHOT_010"
    assert "beauty" in info3["plates"]

    # 5b. Re-opening the existing shot with a different plate: manifest stays read-only,
    #     the override lands in project.json
    manifest_file = work_dir / "shot_manifest_SHOT_010.json"
    before = manifest_file.read_bytes()
    res_reopen = client.post("/api/project/open_or_create", json={
        "work_dir": str(work_dir),
        "plates": {"cleanplate": "D:/dummy/other_clean_####.exr"},
    })
    assert res_reopen.status_code == 200
    assert res_reopen.json()["manifest"]["plates"]["cleanplate"] == "D:/dummy/other_clean_####.exr"
    assert manifest_file.read_bytes() == before

    # 6. Test dialog pick_folder endpoint with mock (tkinter may be absent, e.g. in the cloud)
    import types
    import app as app_module
    fake_root = types.SimpleNamespace(withdraw=lambda: None, attributes=lambda *a: None, destroy=lambda: None)
    monkeypatch.setattr(app_module, "tk", types.SimpleNamespace(Tk=lambda: fake_root))
    monkeypatch.setattr(app_module, "filedialog", types.SimpleNamespace(askdirectory=lambda **kwargs: str(work_dir)))
    res_dialog = client.post("/api/dialog/pick_folder", json={"title": "Test Dialog"})
    assert res_dialog.status_code == 200
    assert res_dialog.json()["folder"] == str(work_dir).replace("\\", "/")


def test_api_plate_clean_lifecycle(tmp_path):
    client = TestClient(app)

    work_dir = tmp_path / "SHOT_CLEAN_TEST"
    work_dir.mkdir()
    plates_dir = work_dir / "plates_beauty"
    plates_dir.mkdir()

    # Generate 5 test frames with intentional flicker
    for f in range(1, 6):
        factor = 1.0 + (0.2 if f % 2 == 0 else -0.1)
        img = (np.ones((64, 64, 3), dtype=np.float32) * 0.5 * factor).astype(np.float32)
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.exr"), img)

    manifest_data = {
        "shot": "SHOT_CLEAN",
        "first": 1,
        "last": 5,
        "plate_w": 64,
        "plate_h": 64,
        "colorspace": "ACES2065-1",
        "source_type": "exr_seq",
        "plates": {
            "beauty": str(plates_dir / "b_####.exr")
        },
        "work": str(work_dir)
    }

    manifest_file = work_dir / "shot_manifest_SHOT_CLEAN.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Open project
    res_open = client.post("/api/project/open", json={"manifest_path": str(manifest_file)})
    assert res_open.status_code == 200

    # 2. Measure flicker
    res_m = client.post("/api/plate/clean/measure", json={"plate": "beauty", "max_frames": 5})
    assert res_m.status_code == 200
    m_data = res_m.json()
    assert m_data["success"] is True
    assert "flicker" in m_data

    # 3. Run deflicker in Nuke Group mode
    res_run = client.post("/api/plate/clean/run", json={
        "plate": "beauty",
        "export_mode": "nuke_group",
        "enable_stage1": True,
        "enable_stage2": False,
        "enable_stage4": True,
        "enable_denoise": False
    })
    assert res_run.status_code == 200
    r_data = res_run.json()
    assert r_data["success"] is True
    assert "nk_script_path" in r_data
    assert Path(r_data["nk_script_path"]).exists()

    # 4. Fetch clean report
    res_rep = client.get("/api/plate/clean/report?plate=beauty")
    assert res_rep.status_code == 200
    assert res_rep.json()["exists"] is True
    assert "report" in res_rep.json()


def _open_clean_project(client, tmp_path, name, ext, make_frame, n=5):
    work_dir = tmp_path / name
    plates_dir = work_dir / "plates_beauty"
    plates_dir.mkdir(parents=True)
    for f in range(1, n + 1):
        assert cv2.imwrite(str(plates_dir / f"b_{f:04d}.{ext}"), make_frame(f))
    manifest_file = work_dir / f"shot_manifest_{name}.json"
    manifest_file.write_text(json.dumps({
        "shot": name, "first": 1, "last": n, "plate_w": 64, "plate_h": 64, "colorspace": "ACES2065-1",
        "source_type": "exr_seq", "plates": {"beauty": str(plates_dir / f"b_####.{ext}")}, "work": str(work_dir)}))
    assert client.post("/api/project/open", json={"manifest_path": str(manifest_file)}).status_code == 200


@pytest.mark.parametrize("channels", [1, 4])
def test_api_plate_clean_non_rgb_plate(tmp_path, channels):
    """Deflicker measure + run on grayscale / RGBA plates (measure used to crash with an IndexError -> HTTP 500)."""
    def frame(f):
        v = np.full((64, 64), int(100 + 30 * (f % 2)), np.uint8)
        return v if channels == 1 else np.dstack([v, v, v, np.full_like(v, 255)])

    client = TestClient(app)
    _open_clean_project(client, tmp_path, f"CLEAN_CH{channels}", "png", frame)
    res_m = client.post("/api/plate/clean/measure", json={"plate": "beauty", "max_frames": 5})
    assert res_m.status_code == 200, res_m.text
    assert res_m.json()["success"] is True
    res_r = client.post("/api/plate/clean/run", json={"plate": "beauty", "export_mode": "nuke_group"})
    assert res_r.status_code == 200, res_r.text


def test_unhandled_error_is_json_detail(tmp_path, monkeypatch):
    """An unhandled exception must come back as JSON {"detail": ...} (the UI shows it), not plain-text 'Internal Server Error'."""
    client = TestClient(app, raise_server_exceptions=False)
    _open_clean_project(client, tmp_path, "CLEAN_BOOM", "png", lambda f: np.full((64, 64, 3), 100 + f, np.uint8))

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(DeflickerMetrics, "measure_flow_aligned", staticmethod(boom))
    res = client.post("/api/plate/clean/measure", json={"plate": "beauty", "max_frames": 5})
    assert res.status_code == 500
    assert res.headers["content-type"].startswith("application/json")
    assert "RuntimeError: synthetic failure" in res.json()["detail"]


def test_api_camera_lifecycle(tmp_path):
    client = TestClient(app)

    work_dir = tmp_path / "SHOT_CAMERA_TEST"
    work_dir.mkdir()
    plates_dir = work_dir / "plates_beauty"
    plates_dir.mkdir()

    # Generate 6 test frames with small shift
    for f in range(1, 7):
        img = np.zeros((80, 100, 3), dtype=np.float32)
        cv2.circle(img, (30 + f * 2, 40), 10, (1.0, 1.0, 1.0), -1)
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.exr"), img)

    manifest_data = {
        "shot": "SHOT_CAMERA",
        "first": 1,
        "last": 6,
        "plate_w": 100,
        "plate_h": 80,
        "colorspace": "ACES2065-1",
        "source_type": "exr_seq",
        "plates": {
            "beauty": str(plates_dir / "b_####.exr")
        },
        "work": str(work_dir)
    }

    manifest_file = work_dir / "shot_manifest_SHOT_CAMERA.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 1. Open project
    res_open = client.post("/api/project/open", json={"manifest_path": str(manifest_file)})
    assert res_open.status_code == 200

    # 2. Check camera report before tracking
    res_before = client.get("/api/camera/report")
    assert res_before.status_code == 200
    assert res_before.json()["exists"] is False

    # 3. Track camera motion
    res_track = client.post("/api/camera/track", json={
        "eval_w": 100,
        "anchor_interval": 3,
        "export_nk": True
    })
    assert res_track.status_code == 200
    track_data = res_track.json()
    assert track_data["success"] is True
    assert "camera2d" in track_data
    assert track_data["camera2d"]["shot_name"] == "SHOT_CAMERA"
    assert len(track_data["camera2d"]["frames"]) == 6
    assert Path(track_data["nk_script_path"]).exists()

    # 4. Check camera report after tracking
    res_after = client.get("/api/camera/report")
    assert res_after.status_code == 200
    assert res_after.json()["exists"] is True
    assert res_after.json()["nk_exists"] is True

    # 5. Export Nuke script again
    res_export = client.post("/api/camera/export_nk")
    assert res_export.status_code == 200
    assert res_export.json()["success"] is True
    assert Path(res_export.json()["nk_path"]).exists()




def test_plate_size_comes_from_manifest_not_4k_fallback(client, tmp_path):
    """Audit fix 2: plate size = manifest plate_w/plate_h (here 1280x720), never 4096x2731."""
    work_dir = tmp_path / "shot_hd"
    plates_dir = work_dir / "plates"
    plates_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random((720, 1280, 3)).astype(np.float32), (0, 0), 3)
    for f in range(1, 4):
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.exr"), base)
        cv2.imwrite(str(plates_dir / f"x2_{f:04d}.exr"), np.roll(base, 2, axis=1))
    manifest_file = work_dir / "shot_manifest_HD.json"
    manifest_file.write_text(json.dumps({
        "shot": "HD", "fps": 25, "first": 1, "last": 3, "plate_w": 1280, "plate_h": 720,
        "colorspace": "ACES2065-1", "work": str(work_dir),
        "plates": {"beauty": str(plates_dir / "b_####.exr"), "backlit": str(plates_dir / "x2_####.exr")},
    }), encoding="utf-8")

    res = client.post("/api/project/open", json={"manifest_path": str(manifest_file)})
    assert res.status_code == 200
    assert (res.json()["project"]["plate_w"], res.json()["project"]["plate_h"]) == (1280, 720)

    assert client.post("/api/plate/align/measure", json={"plate": "backlit", "aux_plate_type": "backlit"}).status_code == 200
    nk = client.post("/api/plate/align/export_nk", json={"plate": "backlit"}).json()["script"]
    assert "4096" not in nk and "2731" not in nk


def test_plate_size_from_header_when_manifest_lacks_it(tmp_path):
    from engine.project import resolve_plate_size
    plates = tmp_path / "p"
    plates.mkdir()
    for f in (1, 2):
        cv2.imwrite(str(plates / f"b_{f:04d}.png"), np.zeros((540, 960, 3), np.uint8))
    assert resolve_plate_size({"plates": {"beauty": str(plates / "b_####.png")}}) == (960, 540)
    with pytest.raises(ValueError):
        resolve_plate_size({"plates": {}})


def test_camera_homography_reported_in_plate_pixels_from_proxies():
    """Proxies (half size) in, homography out in full-plate pixels (manifest plate_w/plate_h)."""
    from engine.camera import Camera2DTracker
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((400, 600, 3)) * 255).astype(np.uint8), (0, 0), 1.5)
    frames = [tex[50:306, 50:434].copy(), tex[50:306, 53:437].copy()]  # 3 px pan in proxy space
    cam = Camera2DTracker.track_sequence(frames, frame_numbers=[1, 2], ref_frame=1, eval_w=384,
                                         plate_w=768, plate_h=512)
    assert (cam.plate_w, cam.plate_h) == (768, 512)
    assert abs(abs(cam.frames[1].dx) - 6.0) < 0.5


def test_api_project_new_wizard_and_refuses_overwrite(client, tmp_path):
    work = tmp_path / "NEW_010"
    req = {"shot_name": "NEW_010", "work_dir": str(work), "beauty_pattern": str(work / "b_####.png"),
           "first_frame": 1, "last_frame": 5, "plate_w": 1920, "plate_h": 1080}
    res = client.post("/api/project/new", json=req)
    assert res.status_code == 200, res.text
    assert res.json()["project"]["work_dir"].endswith("/matte_desk")
    assert (work / "matte_desk" / "project.json").exists()
    before = (work / "shot_manifest_NEW_010.json").read_bytes()
    again = client.post("/api/project/new", json={**req, "plate_w": 10})
    assert again.status_code == 409
    assert (work / "shot_manifest_NEW_010.json").read_bytes() == before


def test_last_project_reopens_on_startup(tmp_path, monkeypatch):
    """Z0: an opened shot folder is remembered in config.local.json (last + recent, max 5, no duplicates)
    and reopened on server startup (lifespan) without touching the shot manifest."""
    import engine.config as ec
    local = tmp_path / "config.local.json"
    local.write_text(json.dumps({"comfy_port": 8001}), encoding="utf-8")
    monkeypatch.setattr(ec, "LOCAL", local)

    work_dir = tmp_path / "SHOT_LAST"
    beauty_dir = work_dir / "plate_beauty"
    beauty_dir.mkdir(parents=True)
    for f in range(1, 4):
        cv2.imwrite(str(beauty_dir / f"SHOT_LAST_beauty_{f:04d}.png"), np.zeros((48, 80, 3), dtype=np.uint8))

    saved = dict(STATE)
    try:
        res = TestClient(app).post("/api/project/open_or_create", json={
            "work_dir": str(work_dir), "plates": {"beauty": str(beauty_dir)}, "shot_name": "SHOT_LAST"})
        assert res.status_code == 200, res.text
        cfg = json.loads(local.read_text(encoding="utf-8"))
        wd = str(work_dir).replace("\\", "/")
        assert cfg["last_work_dir"] == wd
        assert cfg["recent_work_dirs"] == [wd]
        assert cfg["comfy_port"] == 8001  # other machine settings kept
        mf = next(work_dir.glob("shot_manifest_*.json"))
        before = mf.read_bytes()

        # server restart: nothing open -> lifespan reopens the last shot
        for k in ("project", "manifest", "manifest_path", "work_dir", "shot_project"):
            STATE[k] = None
        with TestClient(app) as c:
            cur = c.get("/api/project/current").json()
            assert cur["loaded"] is True
            assert cur["project"]["shot"] == "SHOT_LAST"
            assert cur["project"]["plate_w"] == 80 and cur["project"]["plate_h"] == 48
            rec = c.get("/api/project/recent").json()
            assert rec["last_work_dir"] == wd
            assert rec["recent"] == [{"work_dir": wd, "exists": True}]
        assert mf.read_bytes() == before

        # recent list: newest first, no duplicates, at most 5
        for i in range(6):
            ec.remember_work_dir(str(tmp_path / f"other_{i}"))
        ec.remember_work_dir(wd + "/")
        cfg = json.loads(local.read_text(encoding="utf-8"))
        assert cfg["recent_work_dirs"][0] == wd and len(cfg["recent_work_dirs"]) == 5
        assert len(set(cfg["recent_work_dirs"])) == 5

        # last folder gone -> startup does not fail, nothing opened
        ec.write_local({"last_work_dir": str(tmp_path / "missing_shot")})
        for k in ("project", "manifest", "manifest_path", "work_dir", "shot_project"):
            STATE[k] = None
        with TestClient(app) as c:
            assert c.get("/api/project/current").json() == {"loaded": False}
        assert mf.read_bytes() == before
    finally:
        STATE.clear()
        STATE.update(saved)


def _tiny_video(path, n=5, size=(64, 32)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, size)
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), 40 * i, np.uint8))
    vw.release()


def test_video_plate_is_decoded_to_sequence(client, tmp_path):
    """A .mp4/.mov given as a plate (Manage plates, or an old project that stored the video path) becomes a PNG
    sequence in <work>/source/<stem>/ before proxies are made (2026-10-18: proxies of an .mp4 plate -> HTTP 500)."""
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    work_dir = tmp_path / "shot_vid"
    work_dir.mkdir()
    for f in range(1, 6):
        cv2.imwrite(str(work_dir / f"seq_{f:04d}.png"), np.zeros((32, 64, 3), np.uint8))
    manifest = {"shot": "VID", "fps": 24, "first": 1, "last": 5, "plate_w": 64, "plate_h": 32, "colorspace": "sRGB",
                "plates": {"beauty": str(work_dir / "seq_####.png")}, "work": str(work_dir)}
    mf = work_dir / "shot_manifest_VID.json"
    mf.write_text(json.dumps(manifest), encoding="utf-8")
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    vid = work_dir / "clip.mp4"
    _tiny_video(vid)

    # Manage plates -> the stored plate is the decoded sequence, never the video
    res = client.post("/api/project/plate/update", json={"plate_name": "beauty", "pattern": str(vid)})
    assert res.status_code == 200, res.text
    pat = res.json()["pattern"]
    assert pat.endswith("clip_####.png") and "/source/clip/" in pat.replace("\\", "/")
    assert client.post("/api/project/generate_proxies?plate=beauty").json()["rendered_count"] == 5

    # an old project that kept the video path itself: generate_proxies decodes it (reusing the sequence)
    STATE["shot_project"].set_plate("beauty", str(vid).replace("\\", "/"))
    from app import _refresh_manifest_view
    _refresh_manifest_view()
    res = client.post("/api/project/generate_proxies?plate=beauty")
    assert res.status_code == 200, res.text
    assert res.json()["rendered_count"] == 5
    assert client.get("/api/project/current").json()["manifest"]["plates"]["beauty"].endswith("clip_####.png")


def test_wizard_video_plates_new_shot(client, tmp_path):
    """Project wizard: beauty and backlit given as video files -> both decoded into <work>/matte_desk/source/."""
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")
    work_dir = tmp_path / "shot_wiz"
    work_dir.mkdir()
    _tiny_video(work_dir / "beauty.mp4")
    _tiny_video(work_dir / "backlit.mov")
    res = client.post("/api/project/open_or_create", json={
        "work_dir": str(work_dir), "shot_name": "WIZ",
        "plates": {"beauty": str(work_dir / "beauty.mp4"), "backlit": str(work_dir / "backlit.mov")}})
    assert res.status_code == 200, res.text
    plates = client.get("/api/project/current").json()["manifest"]["plates"]
    assert plates["beauty"].endswith("source/beauty/beauty_####.png")
    assert plates["backlit"].endswith("source/backlit/backlit_####.png")
    assert len(list((work_dir / "matte_desk" / "source" / "backlit").glob("backlit_*.png"))) == 5


def _png_seq(d: Path, name: str, w: int, h: int, n: int = 3) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in range(1, n + 1):
        cv2.imwrite(str(d / f"{name}_{f:04d}.png"), np.zeros((h, w, 3), dtype=np.uint8))
    return d


def test_open_rejects_aux_plate_from_another_shot(tmp_path):
    """2026-10-23: form fields left from SH010 were saved into sh060's project.json as added plates."""
    client = TestClient(app)
    saved = dict(STATE)
    try:
        work = tmp_path / "SHOT_A"
        beauty = _png_seq(work / "plate_beauty", "SHOT_A_beauty", 80, 48)
        foreign = _png_seq(tmp_path / "other" / "backlit", "OTHER_backlit", 64, 32)
        same = _png_seq(work / "plate_backlit", "SHOT_A_backlit", 80, 48)

        # new shot (wizard): a foreign backlit is refused, nothing created
        r = client.post("/api/project/open_or_create", json={
            "work_dir": str(work), "shot_name": "SHOT_A",
            "plates": {"beauty": str(beauty), "backlit": str(foreign)}})
        assert r.status_code == 400 and "innego ujęcia" in r.text
        assert not list(work.glob("shot_manifest_*.json"))

        r = client.post("/api/project/open_or_create", json={
            "work_dir": str(work), "shot_name": "SHOT_A", "plates": {"beauty": str(beauty)}})
        assert r.status_code == 200, r.text
        pj = work / "matte_desk" / "project.json"

        # existing shot: a leftover foreign plate is refused and not saved
        for k in ("backlit", "cleanplate"):
            r = client.post("/api/project/open_or_create", json={
                "work_dir": str(work), "plates": {"beauty": str(beauty), k: str(foreign)}})
            assert r.status_code == 400 and k in r.text
            assert k not in json.loads(pj.read_text(encoding="utf-8"))["plates_added"]

        # a plate of the same resolution is still accepted
        r = client.post("/api/project/open_or_create", json={
            "work_dir": str(work), "plates": {"beauty": str(beauty), "backlit": str(same)}})
        assert r.status_code == 200, r.text
        assert "backlit" in json.loads(pj.read_text(encoding="utf-8"))["plates_added"]
    finally:
        STATE.clear()
        STATE.update(saved)
