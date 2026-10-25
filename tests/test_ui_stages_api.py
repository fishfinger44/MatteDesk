# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""UI v3 backend pieces: points -> key (stage 1), aligned proxy preview (stage 0), project stage status.

- POST /api/object/{o}/points/{f}: the points of a frame ARE its key (SAM3_Detect from the points -> key `points`);
  no points -> the points key is deleted; an empty SAM3 result removes a stale points key and says so.
- GET /api/frame/{f}?plate=X&aligned=1: proxy warped by Align_<plate>.json (preview of "Skoryguj plate"), nothing
  written to the plate folders, the user's manifest untouched.
- GET /api/project/stages: plates / align / clean / camera / depth summary for the stage bar.
- GET /api/object/{o}/stages carries `has` (results on disk, possibly stale) for the view tabs.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.segment import iou
from mock_comfy import MockComfyClient, MockOutputs
from synth import display, make_shot, sam3_handlers


def _open(tmp_path, monkeypatch, n=8):
    import app as app_module
    shot = make_shot(tmp_path, n=n)
    outs = MockOutputs(tmp_path / "comfy_out")
    client = MockComfyClient(handlers=sam3_handlers(outs))
    client.output_root = outs.root
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    assert api.post("/api/object", json={"name": "puppet", "prompts": ["felt puppet"]}).status_code == 200
    return api, shot, mf


def test_points_are_the_key(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch)
    f = shot.first + 2
    ys, xs = np.where(shot.masks[f] > 0.5)
    px, py = float(np.median(xs)), float(np.median(ys))
    before = mf.read_bytes()

    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[px, py]]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["key"]["kind"] == "points" and not d.get("empty")
    keys = api.get("/api/object/puppet/keys").json()["keys"]
    k = next(x for x in keys if x["frame"] == f)
    assert k["kind"] == "points" and k["author"] == "user" and k["positive"] == [[px, py]]
    m = cv2.imread(str(tmp_path / "work" / "matte_desk" / "objects" / "puppet" / "keys" / f"k_{f:04d}_mask.png"),
                   cv2.IMREAD_GRAYSCALE) > 127
    assert iou(m, shot.masks[f] > 0.5) > 0.85
    st = api.get("/api/object/puppet/stages").json()
    assert st["stages"][0]["state"] == "done" and st["has"] == {"core": False, "final": False, "core_fix": False}
    # the preview endpoint serves the mask of the points for the stage-1 overlay
    assert api.get(f"/api/object/puppet/frame/{f}?view=key").status_code == 200

    # with the object's prompt the prompt is the base of the mask (user decision 2026-10-15): a point on the background
    # adds nothing, the puppet from the prompt stays
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[2.0, 2.0]]})
    assert r.status_code == 200 and r.json()["mode"] == "prompt+points" and r.json()["key"]["kind"] == "points"
    # points only (no prompt): a point far from the puppet -> SAM3 finds nothing -> the stale points key is removed
    assert api.put("/api/object/puppet/settings", json={"prompts": []}).status_code == 200
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[2.0, 2.0]]})
    assert r.status_code == 200 and r.json()["empty"] and r.json()["deleted"]
    assert all(x["frame"] != f for x in api.get("/api/object/puppet/keys").json()["keys"])

    # points again, then none -> key deleted; undo brings it back (history.json)
    assert api.post(f"/api/object/puppet/points/{f}", json={"positive": [[px, py]]}).json()["key"]["kind"] == "points"
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [], "negative": []})
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert all(x["frame"] != f for x in api.get("/api/object/puppet/keys").json()["keys"])
    assert api.post("/api/object/puppet/undo").json()["success"]
    assert any(x["frame"] == f for x in api.get("/api/object/puppet/keys").json()["keys"])

    # a brush key is never overwritten by points
    r = api.post(f"/api/object/puppet/keys/{f}", json={"source": "preview", "strokes": [{"mode": "add", "radius_px": 4, "p": [[px, py]]}]})
    assert r.status_code == 200 and r.json()["key"]["kind"] == "brush"
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[px, py]]})
    assert r.status_code == 200 and r.json()["kept"] == "brush"
    assert next(x for x in api.get("/api/object/puppet/keys").json()["keys"] if x["frame"] == f)["kind"] == "brush"

    # the prompt is editable without touching keys
    assert api.put("/api/object/puppet/settings", json={"prompts": ["blue puppet"]}).status_code == 200
    assert api.get("/api/objects").json()["objects"][0]["detect"]["prompts"] == ["blue puppet"]
    assert mf.read_bytes() == before


def test_aligned_proxy_preview_and_project_stages(tmp_path, monkeypatch):
    import app as app_module
    api = TestClient(app_module.app)
    work_dir = tmp_path / "shot_align"
    plates_dir = work_dir / "plates"
    plates_dir.mkdir(parents=True)
    for f in range(1, 4):
        b = np.zeros((120, 160, 3), np.float32); b[40:80, 50:110] = 1.0; b[20:25, 10:150] = 0.5
        cv2.imwrite(str(plates_dir / f"b_{f:04d}.exr"), b)
        a = np.zeros((120, 160, 3), np.float32); a[44:84, 50:110] = 1.0; a[24:29, 10:150] = 0.5   # 4 px down
        cv2.imwrite(str(plates_dir / f"cp_{f:04d}.exr"), a)
    mf = work_dir / "shot_manifest_TEST_AL.json"
    mf.write_text(json.dumps({"shot": "TEST_AL", "fps": 24, "first": 1, "last": 3, "plate_w": 160, "plate_h": 120,
                              "colorspace": "ACES2065-1",
                              "plates": {"beauty": str(plates_dir / "b_####.exr"), "cleanplate": str(plates_dir / "cp_####.exr")},
                              "work": str(work_dir)}), encoding="utf-8")
    before = mf.read_bytes()
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200

    st = api.get("/api/project/stages").json()
    assert {p["name"] for p in st["plates"]} == {"beauty", "cleanplate"}
    assert st["align"]["cleanplate"] == {"measured": False, "applied": False}
    assert "backlit" not in st["align"] and st["camera"]["done"] is False and st["objects"] == 0

    # no measurement yet -> no aligned preview
    assert api.get("/api/frame/1?plate=cleanplate&aligned=1").status_code == 404
    assert api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate"}).status_code == 200
    st = api.get("/api/project/stages").json()
    assert st["align"]["cleanplate"]["measured"] and st["align"]["cleanplate"]["nk"] and "residual_median" in st["align"]["cleanplate"]
    assert st["align"]["cleanplate"]["device"] == "cpu"          # tests pin MD_ALIGN_DEVICE=cpu; UI shows where it ran
    assert st["align"]["cleanplate"]["stale"] == "" and api.get("/api/plate/align/report?plate=cleanplate").json()["stale"] == ""
    # a report from before 2026-10-15 (local grid without its global model) -> stage 0 says: measure again
    rp = work_dir / "matte_desk" / "align" / "Align_cleanplate.json"
    cur = rp.read_text(encoding="utf-8")
    old = json.loads(cur)
    old["frames"][0].update({"method": "local_stmap", "grid_rel_path": "grid_0001.exr"})
    old["frames"][0].pop("global_method", None)
    rp.write_text(json.dumps(old), encoding="utf-8")
    assert "Zmierz ponownie" in api.get("/api/project/stages").json()["align"]["cleanplate"]["stale"]
    assert "Zmierz ponownie" in api.get("/api/plate/align/report?plate=cleanplate").json()["stale"]
    rp.write_text(cur, encoding="utf-8")
    assert api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate",
                                                      "device": "cpu"}).status_code == 200
    assert api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate",
                                                      "device": "tpu"}).status_code == 400

    # a known report (translation measured in PLATE px) -> the preview rescales it to the proxy:
    # plate declared 2x the proxy size => dy = -8 plate px must become -4 proxy px (= exactly beauty)
    rep = json.loads((work_dir / "matte_desk" / "align" / "Align_cleanplate.json").read_text(encoding="utf-8"))
    rep["method"] = "translation"
    for fr in rep["frames"]:
        fr.update({"method": "translation", "dx": 0.0, "dy": -8.0, "transform_matrix": None, "grid_rel_path": None})
    (work_dir / "matte_desk" / "align" / "Align_cleanplate.json").write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setattr(app_module, "_plate_size", lambda: (320, 240))
    r = api.get("/api/frame/2?plate=cleanplate&aligned=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    plain = cv2.imdecode(np.frombuffer(api.get("/api/frame/2?plate=cleanplate").content, np.uint8), cv2.IMREAD_GRAYSCALE)
    al = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    beauty = cv2.imdecode(np.frombuffer(api.get("/api/frame/2?plate=beauty").content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.abs(plain.astype(float) - beauty).mean() > 5
    assert np.abs(al.astype(float)[8:, :] - beauty[8:, :]).mean() < 2.0   # JPEG noise only (top rows: border fill)
    # cached next to the proxies; nothing written into the plate folder; manifest untouched
    assert (work_dir / "matte_desk" / "proxy" / "cleanplate_aligned_preview" / "p_0002.jpg").exists()
    assert sorted(p.name for p in plates_dir.iterdir()) == sorted([f"b_{f:04d}.exr" for f in range(1, 4)] + [f"cp_{f:04d}.exr" for f in range(1, 4)])
    assert not (plates_dir.parent / "plates_align").exists()
    assert mf.read_bytes() == before
    # frame outside the measured range -> 404 with a reason, not a 500
    assert api.get("/api/frame/9?plate=cleanplate&aligned=1").status_code == 404


def test_points_outside_the_plate_are_refused(tmp_path, monkeypatch):
    """Z11: SH010 F20 had a point at [2189, -422] -> wrong SAM3 mask -> empty core. The server refuses it (400)."""
    api, shot, mf = _open(tmp_path, monkeypatch)
    f = shot.first + 1
    before = mf.read_bytes()
    ys, xs = np.where(shot.masks[f] > 0.5)
    px, py = float(np.median(xs)), float(np.median(ys))
    for bad in ([[px, py]], [[px, -422.0]]), ([[shot.w + 0.0, py]], []), ([[px, py]], [[px, shot.h + 5.0]]), ([[-0.5, py]], []):
        r = api.post(f"/api/object/puppet/points/{f}", json={"positive": bad[0], "negative": bad[1]})
        assert r.status_code == 400 and "poza kadrem" in r.json()["detail"], r.text
    assert all(x["frame"] != f for x in api.get("/api/object/puppet/keys").json()["keys"])   # nothing saved
    # edges of the plate are inside
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[px, py]], "negative": [[0.0, 0.0], [shot.w - 0.5, shot.h - 0.5]]})
    assert r.status_code == 200, r.text
    assert mf.read_bytes() == before


def test_small_fixes_z17(tmp_path, monkeypatch):
    """Z17: /qa of an object without a version -> 200 {} (no 404 in the console); favicon; stage 0 knows the frame count."""
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    r = api.get("/api/object/puppet/qa")
    assert r.status_code == 200 and r.json() == {}
    r = api.get("/favicon.ico")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml") and b"<svg" in r.content
    st = api.get("/api/project/stages").json()
    assert st["n_frames"] == shot.last - shot.first + 1
    assert mf.read_bytes() == before


def test_runs_and_active_run(tmp_path, monkeypatch):
    """Z2: GET /runs lists versions with run.json + qa.json summary; PUT /active_run switches the version used by export."""
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    store = app_module._objects()
    rd = store.dir("puppet") / "runs"
    for i, (gate, fl) in enumerate(((True, {}), (False, {"3": [["red", "dziury"]], "4": [], "5": [["yellow", "x"]]}))):
        d = rd / f"v{i + 1:03d}"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"run": d.name, "edge_mode": "core", "seconds": 10.0 + i,
                                                "created_at": f"2026-10-12T2{i}:00:00", "smoothing": "normal", "band_px": 6.0}))
        (d / "qa.json").write_text(json.dumps({"gate_ok": gate, "flicker_median": 0.002 + i / 1000, "flags": fl}))
    (rd / "v003").mkdir()                                                            # unfinished (no run.json) -> skipped
    store.update("puppet", tool={"active_run": "v002"})
    d = api.get("/api/object/puppet/runs").json()
    assert [r["run"] for r in d["runs"]] == ["v001", "v002"] and d["active_run"] == "v002"
    v1, v2 = d["runs"]
    assert v1["gate_ok"] is True and v1["n_flags"] == 0 and not v1["active"]
    assert v2["gate_ok"] is False and v2["n_flags"] == 2 and v2["active"] and v2["seconds"] == 11.0
    r = api.put("/api/object/puppet/active_run", json={"run": "v001"})
    assert r.status_code == 200 and r.json()["final_valid"] is True
    tool = api.get("/api/objects").json()["objects"][0]["tool"]
    assert tool["active_run"] == "v001" and tool["final_valid"] is True and tool["status"] == "QA OK"
    assert api.put("/api/object/puppet/active_run", json={"run": "v003"}).status_code == 404
    st = api.get("/api/object/puppet/stages").json()
    assert "stale_intervals" in st
    assert mf.read_bytes() == before


def test_depth_order_reorder(tmp_path, monkeypatch):
    """Z3: PUT /api/objects/order – depth_order = index, no gaps; wrong list -> 400; a part before its parent -> 400;
    objects whose front set changed and that have a result -> 'do przeliczenia'."""
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    for n in ("tree", "chair"):
        assert api.post("/api/object", json={"name": n, "prompts": [n]}).status_code == 200
    store = app_module._objects()
    store.update("chair", tool={"active_run": "v001", "final_valid": True})
    store.update("tree", tool={"active_run": "v001", "final_valid": True})
    order = lambda: [(o["name"], o["depth_order"]) for o in api.get("/api/objects").json()["objects"]]
    assert order() == [("puppet", 0), ("tree", 1), ("chair", 2)]
    r = api.put("/api/objects/order", json={"names": ["chair", "puppet", "tree"]})
    assert r.status_code == 200, r.text
    assert order() == [("chair", 0), ("puppet", 1), ("tree", 2)]
    # chair lost its fronts, tree gained chair in front -> both recompute; puppet (no result) only renumbered
    assert sorted(r.json()["to_recompute"]) == ["chair", "tree"]
    tools = {o["name"]: o["tool"] for o in api.get("/api/objects").json()["objects"]}
    assert tools["tree"]["final_valid"] is False and tools["tree"]["status"] == "do przeliczenia"
    for bad in (["chair", "puppet"], ["chair", "puppet", "tree", "tree"], ["chair", "puppet", "nope"]):
        assert api.put("/api/objects/order", json={"names": bad}).status_code == 400
    # part before its parent -> 400
    assert api.put("/api/object/tree/settings", json={"part_of": "puppet"}).status_code == 200
    r = api.put("/api/objects/order", json={"names": ["tree", "chair", "puppet"]})
    assert r.status_code == 400 and "rodzic" in r.json()["detail"]
    assert order() == [("chair", 0), ("puppet", 1), ("tree", 2)]
    assert mf.read_bytes() == before


def test_objects_list_says_which_results_exist(tmp_path, monkeypatch):
    """/api/objects `has` (core / final of the active version) – the UI draws the OTHER computed objects as extra layers."""
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    assert api.post("/api/object", json={"name": "tree", "prompts": ["tree"]}).status_code == 200
    has = lambda: {o["name"]: o["has"] for o in api.get("/api/objects").json()["objects"]}
    assert has() == {"puppet": {"core": False, "final": False}, "tree": {"core": False, "final": False}}
    store = app_module._objects()
    od = store.dir("tree")
    (od / "core").mkdir(parents=True, exist_ok=True)
    (od / "core" / "core.json").write_text("{}", encoding="utf-8")
    assert has()["tree"] == {"core": True, "final": False}
    store.update("tree", tool={"active_run": "v001"})
    assert has()["tree"] == {"core": True, "final": False}          # version named, no run.json yet
    (od / "runs" / "v001").mkdir(parents=True, exist_ok=True)
    (od / "runs" / "v001" / "run.json").write_text("{}", encoding="utf-8")
    assert has()["tree"] == {"core": True, "final": True} and has()["puppet"] == {"core": False, "final": False}


def test_delete_object_dialog_info_and_optional_exports(tmp_path, monkeypatch):
    """Delete from the objects list (user request 2026-10-15): delete_info for the dialog; the folder always goes to the
    project trash, the exported masks only with exports=1 (to trash/exports/, recoverable); other objects untouched."""
    import app as app_module
    from engine.outputs import load_outputs, update_outputs
    api, shot, mf = _open(tmp_path, monkeypatch)
    for n in ("tree", "rock"):
        assert api.post("/api/object", json={"name": n, "prompts": [n]}).status_code == 200
    work = Path(app_module.STATE["work_dir"])
    entries = []
    for n in ("tree", "rock"):
        d = work / "export" / "mattes" / n
        d.mkdir(parents=True, exist_ok=True)
        for f in (1, 2, 3):
            (d / f"{n}_{f:04d}.exr").write_bytes(b"x" * 1000)
        (d / "notes.txt").write_text("keep", encoding="utf-8")          # not part of the sequence
        entries.append({"name": n, "matte": str(d / f"{n}_####.exr"), "core": None, "version": "v001"})
    update_outputs(work, "objects", entries)

    info = api.get("/api/object/tree/delete_info").json()
    assert info["exports"]["files"] == 3 and info["folder"]["files"] >= 1 and info["parts"] == []
    assert api.get("/api/object/nope/delete_info").status_code == 404

    # without exports: folder -> trash, masks stay, entry gone
    r = api.delete("/api/object/rock").json()
    assert r["exports_moved"] == 0 and Path(r["trash"]).is_dir()
    assert sorted(p.name for p in (work / "export" / "mattes" / "rock").iterdir()) == ["notes.txt", "rock_0001.exr", "rock_0002.exr", "rock_0003.exr"]
    # with exports: the sequence moves to trash/exports, other files stay
    r = api.delete("/api/object/tree?exports=1").json()
    assert r["exports_moved"] == 3
    moved = sorted(p.name for p in Path(r["exports_trash"]).rglob("*.exr"))
    assert moved == ["tree_0001.exr", "tree_0002.exr", "tree_0003.exr"]
    assert sorted(p.name for p in (work / "export" / "mattes" / "tree").iterdir()) == ["notes.txt"]
    assert [e["name"] for e in load_outputs(work)["objects"]] == []
    assert [o["name"] for o in api.get("/api/objects").json()["objects"]] == ["puppet"]


def test_przelicz_klatke_with_the_prompt_alone(tmp_path, monkeypatch):
    """Bug 2026-10-15: „Wyczyść punkty” + a new prompt + „Przelicz klatkę” did nothing (no points = the key was only
    deleted). Now the prompt alone gives the key; the field's prompt is sent with the request and saved in the object;
    an empty field and no points = the points key is deleted."""
    api, shot, mf = _open(tmp_path, monkeypatch)
    f = shot.first + 2
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [], "negative": [], "prompt": "felt puppet"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["mode"] == "prompt" and d["key"]["kind"] == "points" and d["prompt"] == "felt puppet"
    k = next(x for x in api.get("/api/object/puppet/keys").json()["keys"] if x["frame"] == f)
    assert k["prompt"] == "felt puppet"
    # a different prompt in the field is saved in the object right away (no race with the blur save)
    api.post(f"/api/object/puppet/points/{f}", json={"positive": [], "negative": [], "prompt": "orange felt puppet"})
    obj = next(o for o in api.get("/api/objects").json()["objects"] if o["name"] == "puppet")
    assert obj["detect"]["prompts"] == ["orange felt puppet"]
    # empty field + no points -> the key goes
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [], "negative": [], "prompt": ""})
    assert r.json()["deleted"] is True
    assert all(x["frame"] != f for x in api.get("/api/object/puppet/keys").json()["keys"])


def test_przelicz_klatke_ignores_the_object_crops(tmp_path, monkeypatch):
    """Bug 2026-10-15 (CLIP2_A): with crops from „Przelicz nieaktualne”, a + point on the wall outside the object's crop gave
    the hair (SAM3 got the point outside its image). User decision: in stage Wykryj „Przelicz klatkę” never uses the
    object's crops (they come from „Przelicz nieaktualne”) – a square around the points, or the whole frame for a prompt."""
    import app as app_module
    from engine.crop import CropSegment, save_crops
    api, shot, mf = _open(tmp_path, monkeypatch)
    f = shot.first + 2
    od = app_module._objects().dir("puppet")
    save_crops(od / "crops.json", [CropSegment("s00", shot.first, shot.last, 0, 0, [0, 0, 64, 64], [0, 0, 0, 0], 15.75)])
    px, py = shot.w - 20, shot.h - 20                              # far outside the 64 px crop
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [[px, py]], "prompt": ""})
    assert r.status_code == 200, r.text
    cx, cy, side = r.json()["crop"][:3]
    assert cx <= px < cx + side and cy <= py < cy + side and [cx, cy, side] != [0, 0, 64]
    # a prompt: the whole frame (the square covering the plate), not the object's crop either
    r = api.post(f"/api/object/puppet/points/{f}", json={"positive": [], "prompt": "felt puppet"})
    cx, cy, side = r.json()["crop"][:3]
    assert cx <= 0 and cy <= 0 and cx + side >= shot.w and cy + side >= shot.h
