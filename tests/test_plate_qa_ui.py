# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z26 UI backend – plate QA decisions, the stage-1 gate (like Z18), before / after preview, the write plan with
size + ETA, the `_fix` switch (change of the repair from the source, restore) and objects "do przeliczenia"."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.plate_qa import gate_state

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_gate_state_transitions():
    rep = {"pattern": "D:/s/X2/x_####.exr", "flagged": [5, 9], "decisions": {}}
    g = gate_state(rep, "D:/s/X2/x_####.exr", None)
    assert g["state"] == "undecided" and g["blocking"] and g["undecided"] == [5, 9]
    rep["decisions"] = {"5": "mean"}
    assert gate_state(rep, "D:/s/X2/x_####.exr", None)["undecided"] == [9]
    rep["decisions"] = {"5": "mean", "9": "keep"}
    g = gate_state(rep, "D:/s/X2/x_####.exr", None)
    assert g["state"] == "pending" and g["blocking"] and g["repairs"] == {"5": "mean"}
    fix = {"source": "D:/s/X2/x_####.exr", "pattern": "D:/s/X2_fix/x_####.exr", "decisions": {"5": "mean", "9": "keep"}}
    g = gate_state(rep, "d:\\s\\X2_fix\\x_####.exr", fix)                      # Windows spelling of the same path
    assert g["state"] == "fixed" and not g["blocking"]
    rep["decisions"] = {"5": "flow", "9": "keep"}                             # repair changed after the write
    assert gate_state(rep, "D:/s/X2_fix/x_####.exr", fix)["state"] == "pending"
    rep["decisions"] = {"5": "keep", "9": "keep"}                             # everything kept now -> back to the source
    assert gate_state(rep, "D:/s/X2_fix/x_####.exr", fix)["state"] == "pending"
    assert gate_state(rep, "D:/s/X2/x_####.exr", None)["state"] == "kept"
    assert gate_state({"pattern": "a", "flagged": []}, "a", None)["state"] == "clean"
    assert gate_state(rep, "D:/other/y_####.exr", None)["state"] == "other"       # plate replaced: not blocking
    assert not gate_state(rep, "D:/other/y_####.exr", None)["blocking"]
    assert gate_state(None, "a", None) == {"state": "none", "blocking": False}


def test_api_plate_qa_ui_flow(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    from engine.plate_fix import fixed_dir_for
    from engine.plate_qa import frame_path
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    work = Path(app_module.STATE["work_dir"])
    pattern = app_module.STATE["manifest"]["plates"]["beauty"]
    assert api.get("/api/project/stages").json()["plate_qa_block"] is None       # not checked = not blocking
    # an object computed before the repair (its core older than the fix)
    core_json = pipe.store.dir("obj") / "core" / "core.json"
    had_core = core_json.exists()
    victim = frames[len(frames) // 2]
    p = frame_path(pattern, victim)
    bak = p.read_bytes()
    p.unlink()
    try:
        r = api.post("/api/plate/qa", json={"plate": "beauty"})
        assert r.status_code == 200 and victim in r.json()["flagged"], r.text
        other = next(f for f in frames if f not in r.json()["flagged"])
        # gate: stage 1 locked, detect / crops refused with the reason
        st = api.get("/api/project/stages").json()
        assert st["plate_qa"]["beauty"]["state"] == "undecided" and "kontrola klatek" in st["plate_qa_block"]
        ob = api.get("/api/object/obj/stages").json()
        assert ob["plate_block"] and ob["steps"][0]["state"] == "locked"
        assert all(s["state"] in ("locked", "done") for s in ob["stages"])
        r = api.post("/api/object/obj/detect", json={"frame": start})
        assert r.status_code == 409 and "kontrola klatek" in r.json()["detail"]
        assert api.post("/api/object/obj/crops").status_code == 409
        assert api.post("/api/object/obj/run").status_code == 409
        # decisions: only flagged frames, valid methods
        assert api.put("/api/plate/qa/beauty/decisions", json={"decisions": {str(other): "mean"}}).status_code == 400
        assert api.put("/api/plate/qa/beauty/decisions", json={"decisions": {str(victim): "zzz"}}).status_code == 400
        assert api.put("/api/plate/qa/nope/decisions", json={"decisions": {}}).status_code == 404
        r = api.put("/api/plate/qa/beauty/decisions", json={"decisions": {str(victim): "mean"}})
        assert r.status_code == 200 and r.json()["gate"]["state"] == "pending" and "zapisz naprawę" in r.json()["block"]
        rep = api.get("/api/plate/qa/beauty").json()
        assert rep["decisions"] == {str(victim): "mean"} and rep["gate"]["state"] == "pending"
        # before / after preview: the missing source 404, the repair = mean of the neighbours (JPEG)
        assert api.get(f"/api/plate/qa/beauty/preview/{victim}?method=orig").status_code == 404
        assert api.get(f"/api/plate/qa/beauty/preview/{victim}?method=zzz").status_code == 400
        r = api.get(f"/api/plate/qa/beauty/preview/{victim}?method=mean&side=128")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) == 128
        assert api.get(f"/api/plate/qa/beauty/preview/{other}?method=orig").status_code == 200
        # plan: sizes + ETA, nothing written
        new_dir, new_pattern = fixed_dir_for(pattern)
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "mean"}})
        plan = r.json()["plan"]
        assert r.json()["success"] is False and plan["eta_s"] >= 0 and plan["exists"] is False and not new_dir.exists()
        # proxy of the damaged frame rendered before the repair -> dropped by the write
        px = work / "proxy" / "beauty" / f"p_{victim:04d}.jpg"
        px.parent.mkdir(parents=True, exist_ok=True)
        px.write_bytes(b"old")
        time.sleep(0.05)
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "mean"}, "confirmed": True})
        assert r.status_code == 200 and r.json()["success"], r.text
        assert not px.exists()
        assert had_core
        if had_core:
            assert r.json()["stale_objects"] == ["obj"]
            ob = api.get("/api/object/obj/stages").json()
            core = next(s for s in ob["stages"] if s["id"] == "core")
            assert core["state"] == "stale" and "naprawiony" in core["why"] and str(victim) in core["why"]
            assert ob["stale_intervals"] and any(a <= victim <= b for a, b in ob["stale_intervals"])
        pj = json.loads((work / "project.json").read_text(encoding="utf-8"))
        assert Path(pj["plate_fix"]["beauty"]["source"]) == Path(pattern)
        assert pj["plate_changed"]["beauty"]["frames"] == [victim]
        st = api.get("/api/project/stages").json()
        assert st["plate_qa"]["beauty"]["state"] == "fixed" and st["plate_qa_block"] is None
        assert "plate_block" not in api.get("/api/object/obj/stages").json()
        # change of the repair: re-check the source, decisions start from the written ones; the write goes to the
        # same _fix (never _fix_fix)
        r = api.post("/api/plate/qa", json={"plate": "beauty", "source": True})
        assert r.status_code == 200 and victim in r.json()["flagged"]
        rep = api.get("/api/plate/qa/beauty").json()
        assert Path(rep["pattern"]) == Path(pattern) and rep["gate"]["state"] == "fixed"
        api.put("/api/plate/qa/beauty/decisions", json={"decisions": {str(victim): "flow"}})
        assert api.get("/api/project/stages").json()["plate_qa"]["beauty"]["state"] == "pending"
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "flow"}, "confirmed": True})
        assert r.status_code == 200 and Path(r.json()["dir"]) == new_dir
        assert not Path(str(new_dir) + "_fix").exists()
        assert api.get("/api/project/stages").json()["plate_qa"]["beauty"]["state"] == "fixed"
        # everything kept: plan = restore, confirmed -> the project reads the source again
        api.put("/api/plate/qa/beauty/decisions", json={"decisions": {str(victim): "keep"}})
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "keep"}})
        assert r.json()["success"] is False and Path(r.json()["plan"]["restore"]) == Path(pattern)
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "keep"}, "confirmed": True})
        assert r.status_code == 200 and r.json()["restored"]
        assert Path(app_module.STATE["manifest"]["plates"]["beauty"]) == Path(pattern)
        assert api.get("/api/project/stages").json()["plate_qa"]["beauty"]["state"] == "kept"
        # the manifest of the shot is never written
        mf = json.loads((tmp_path / "work" / "shot_manifest_SYN.json").read_text(encoding="utf-8"))
        assert Path(mf["plates"]["beauty"]) == Path(pattern)
    finally:
        p.write_bytes(bak)


def test_fix_refuses_foreign_fix_folder(tmp_path):
    """A `<plate>_fix` folder not written by MatteDesk (e.g. made by hand – SH010 X2_fix) is never overwritten."""
    from engine.plate_fix import fixed_dir_for, write_fixed_plate
    d = tmp_path / "X2"
    d.mkdir()
    img = np.full((16, 24, 3), 128, np.uint8)
    for f in range(1, 6):
        cv2.imwrite(str(d / f"x_{f:04d}.png"), img)
    pattern = str(d / "x_####.png")
    new_dir, _ = fixed_dir_for(pattern)
    new_dir.mkdir()
    (new_dir / "x_0003.png").write_bytes(b"hand-made")
    with pytest.raises(FileExistsError):
        write_fixed_plate(pattern, list(range(1, 6)), {3: "mean"}, confirmed=False)
    assert (new_dir / "x_0003.png").read_bytes() == b"hand-made"
