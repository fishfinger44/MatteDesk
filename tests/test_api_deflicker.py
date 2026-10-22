# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""api_deflicker.py (docs/PLAN_DEFLICKER_AUTO.md § 2–5): analyze (measurement + automat on ±radius frames), preview (deflicker at
eval_w with the palette's settings, files for the before / after loop), preview files; the run endpoint takes the palette knobs."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api_deflicker import settings_from_dict
from app import app


def _project(tmp_path, n=12, w=96, h=64):
    work = tmp_path / "DFL"
    plates = work / "plates_beauty"
    plates.mkdir(parents=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.35 + 0.4 * xx / w + 0.1 * np.sin(yy / 5.0))[..., None] * np.array([0.9, 1.0, 1.1], np.float32)
    for f in range(1, n + 1):
        img = np.clip(base * (1 + 0.08 * rng.normal()) * 255, 0, 255).astype(np.uint8)   # random level flicker
        assert cv2.imwrite(str(plates / f"b_{f:04d}.png"), img)
    mf = work / "shot_manifest_DFL.json"
    mf.write_text(json.dumps({"shot": "DFL", "first": 1, "last": n, "plate_w": w, "plate_h": h, "colorspace": "ACES2065-1",
                              "source_type": "exr_seq", "plates": {"beauty": str(plates / "b_####.png")}, "work": str(work)}))
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    return client


def test_settings_from_dict_casts_and_clamps():
    s = settings_from_dict({"strength": "7", "window": 99, "grid": 0, "texture": True, "texture_radius": 0, "unknown": 1})
    assert (s.strength, s.window, s.grid, s.texture, s.texture_radius) == (7, 30, False, True, 1)


def test_analyze_then_preview_then_files(tmp_path):
    client = _project(tmp_path)
    r = client.post("/api/plate/clean/analyze", json={"plate": "beauty", "frame": 6, "radius": 10, "eval_w": 96})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["success"] and d["window"] == [1, 12] and d["n_frames"] == 12 and "all" in d["measure"]["regions"]
    s = d["settings"]
    assert 1 <= s["strength"] <= 8 and 3 <= s["window"] <= 30 and isinstance(s["notes"], list) and s["notes"]
    assert "flicker" in d and "needs_deflicker" in d and d["measure"]["regions"]["all"]["metric_pct"] > 0

    p = client.post("/api/plate/clean/preview", json={"plate": "beauty", "frame": 6, "radius": 10, "eval_w": 96,
                                                        "settings": s, "export_mode": "nuke_group"})
    assert p.status_code == 200, p.text
    pv = p.json()
    assert pv["success"] and pv["window"] == [1, 12] and len(pv["files"]["frames"]) == 12 and pv["exr"] is False
    reg = pv["regions"]["all"]
    assert reg["after"] < reg["before"] and reg["floor"] >= 0 and -100 <= reg["removed_pct"] <= 100
    assert "stage1_local_mc" in pv["stages_applied"]
    img = client.get(pv["files"]["map_before"])
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")
    fr = client.get(pv["files"]["frames"][3][1])
    assert fr.status_code == 200 and fr.headers["content-type"].startswith("image/")
    assert client.get("/api/plate/clean/preview/file?plate=beauty&name=../x.png").status_code == 400
    assert client.get("/api/plate/clean/preview/file?plate=beauty&name=nope.png").status_code == 404


def test_analyze_window_is_clamped_and_needs_frames(tmp_path):
    client = _project(tmp_path, n=5)
    d = client.post("/api/plate/clean/analyze", json={"plate": "beauty", "frame": 1, "radius": 2, "eval_w": 96}).json()
    assert d["window"] == [1, 3]
    assert client.post("/api/plate/clean/analyze", json={"plate": "nope"}).status_code == 400


def test_run_accepts_palette_knobs(tmp_path):
    client = _project(tmp_path, n=8)
    r = client.post("/api/plate/clean/run", json={"plate": "beauty", "export_mode": "nuke_group", "stage1_iterations": 1,
                                                    "stage1_window": 5, "stage1_sigma": 4, "stage4_luma": True, "target_reduction": 2.0})
    assert r.status_code == 200, r.text
    assert r.json()["success"] and "stage1_local_mc" in r.json()["report"]["stages_applied"]


def test_preview_uses_the_measured_window(tmp_path):
    client = _project(tmp_path, n=12)
    p = client.post("/api/plate/clean/preview", json={"plate": "beauty", "frame": 11, "radius": 10, "window": [2, 7], "eval_w": 96,
                                                        "settings": {"strength": 2, "window": 5}, "export_mode": "nuke_group"})
    assert p.status_code == 200, p.text
    assert p.json()["window"] == [2, 7] and len(p.json()["files"]["frames"]) == 6


def test_exr_export_goes_to_the_chosen_folder_and_never_into_the_plate_folder(tmp_path):
    client = _project(tmp_path, n=6)
    d = client.get("/api/plate/clean/exr_dir?plate=beauty&denoise=false").json()
    assert d["path"].endswith("/plates_beauty_deflicker")
    assert client.get("/api/plate/clean/exr_dir?plate=beauty&denoise=true").json()["path"].endswith("/plates_beauty_deflicker_dn")
    body = {"plate": "beauty", "export_mode": "exr_deflicker", "confirmed": True, "stage1_iterations": 1, "target_reduction": 2.0}
    plate_dir = tmp_path / "DFL" / "plates_beauty"
    r = client.post("/api/plate/clean/run", json={**body, "exr_dir": str(plate_dir)})
    assert r.status_code == 400 and "nadpisałby" in r.json()["detail"]
    out = tmp_path / "wybrany" / "dfl"
    r = client.post("/api/plate/clean/run", json={**body, "exr_dir": str(out)})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["exr_written"] == 6 and j["export_dir"] == str(out).replace("\\", "/") and len(list(out.glob("b_*.png"))) == 6
    assert client.post("/api/plate/clean/run", json={**body, "exr_dir": "wzgledny/katalog"}).status_code == 400
