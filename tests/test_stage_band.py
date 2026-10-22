# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Band of a stage (VideoMaMa) with control points – the same band system as the stage-5 zone (2026-10-22)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)       # no CoTracker3: nearest key
    return _api_project(tmp_path, monkeypatch)


def test_stage_band_points_lines_copy_and_delete(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.put("/api/object/obj/videomama", json={"reach_px": 12, "inset_px": 4, "feather_px": 2,
                                                       "outer_feather_px": 5}).status_code == 200
    st = api.get(f"/api/object/obj/stage_band/videomama/{start}").json()
    assert st["points"] == [] and st["defaults"] == {"out": 12, "in": 4}
    lines0 = api.get(f"/api/object/obj/stage_band/videomama/{start}/lines").json()
    assert lines0["band_contours"] and lines0["inset_feather_contours"] and lines0["feather_contours"]
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    right = (float(xs.max()), float(np.median(ys[xs == xs.max()])))
    r = api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"x": right[0], "y": right[1], "out": 40})
    assert r.status_code == 200, r.text
    assert r.json()["band"]["points"][0]["out"] == 40
    obj = api.get("/api/object/obj/videomama").json()["settings"]
    assert obj["band_points"] and obj["band_ref_out"] == 12
    lines1 = api.get(f"/api/object/obj/stage_band/videomama/{start}/lines").json()
    xmax = lambda L: max(q[0] for c in L["band_contours"] for q in c)
    assert xmax(lines1) > xmax(lines0) + 15                                   # the point widened the band on its side
    # copy from a stage-5 zone: its points replace the stage's (only into a detached stage – 2026-10-22)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    assert api.put(f"/api/object/obj/stage_band/zone:{zid}/mode", json={"mode": "own"}).json()["changed"]
    zst = api.get(f"/api/object/obj/zones/{zid}/band/{start}").json()
    assert [p["id"] for p in zst["points"]] == ["p1"]                         # the detached zone got the shared copy
    api.delete(f"/api/object/obj/zones/{zid}/band")
    api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": right[0], "y": right[1], "out": 9})
    api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.min()), "y": right[1], "out": 3})
    c = api.post("/api/object/obj/stage_band/videomama/copy", json={"source": f"zone:{zid}"})
    assert c.status_code == 409                                               # VideoMaMa reads the shared band
    assert api.put("/api/object/obj/stage_band/videomama/mode", json={"mode": "own"}).status_code == 200
    c = api.post("/api/object/obj/stage_band/videomama/copy", json={"source": f"zone:{zid}"})
    assert c.status_code == 200 and c.json()["points"] == 2, c.text
    st = api.get(f"/api/object/obj/stage_band/videomama/{start}").json()
    assert sorted(p["out"] for p in st["points"]) == [3, 9]
    # keys of the frame, then nothing left = the constant band
    assert api.delete(f"/api/object/obj/stage_band/videomama?frame={start}").status_code == 200
    obj = api.get("/api/object/obj/videomama").json()["settings"]
    assert obj["band_points"] == [] and obj["band_ref_out"] is None
    assert api.put("/api/object/obj/stage_band/nope/1", json={"x": 1, "y": 1}).status_code == 400


def test_matanyone_band_points_need_px_mode_and_undo_restores(tmp_path, monkeypatch):
    """Stage 3: band points shape the px mode of the edge band; Ctrl+Z (params PUT) restores points + reference."""
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.get(f"/api/object/obj/stage_band/matanyone/{start}").status_code == 409        # no px mode yet
    assert api.put("/api/object/obj/settings", json={"band_out_px": 10, "band_in_px": 4, "band_in_feather_px": 2,
                                                     "band_out_feather_px": 5}).status_code == 200
    st = api.get(f"/api/object/obj/stage_band/matanyone/{start}").json()
    assert st["points"] == [] and st["defaults"] == {"out": 10, "in": 4}
    l0 = api.get(f"/api/object/obj/stage_band/matanyone/{start}/lines").json()
    assert l0["band_contours"] and l0["inset_feather_contours"] and l0["feather_contours"]
    before = api.get("/api/object/obj/stage_band/matanyone/params").json()
    assert before == {"band_points": [], "band_ref_out": None, "band_ref_in": None}       # the (empty) shared band
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    right = (float(xs.max()), float(np.median(ys[xs == xs.max()])))
    r = api.put(f"/api/object/obj/stage_band/matanyone/{start}", json={"x": right[0], "y": right[1], "out": 40})
    assert r.status_code == 200, r.text
    eb = api.get("/api/object/obj/stage_band/matanyone/params").json()
    assert eb["band_points"] and eb["band_ref_out"] == 10 and pipe.store.get("obj")["band"]["band_points"]   # shared
    l1 = api.get(f"/api/object/obj/stage_band/matanyone/{start}/lines").json()
    xmax = lambda L: max(q[0] for c in L["band_contours"] for q in c)
    assert xmax(l1) > xmax(l0) + 15
    assert api.put("/api/object/obj/stage_band/matanyone/params",
                   json={"band_points": [], "band_ref_out": None, "band_ref_in": None}).status_code == 200
    assert not pipe.store.get("obj")["band"]["band_points"] and pipe.store.get("obj").get("edge_band") is None


def test_ltx_band_mode_points_and_core_blend(tmp_path, monkeypatch):
    """Stage 7 LTX-2.5, result mode „pas”: the matte blends into the core (1 inside, 0 outside) through the band; band
    points need that mode (409 otherwise); the former modes keep their signature (band keys out of it)."""
    from engine.ltx_alpha import finish, ltx_settings, settings_sig
    core = np.zeros((120, 120), np.float32); core[30:90, 30:90] = 1.0
    s = ltx_settings({"ltx": {"combine": "band", "region_px": 10, "core_erode_px": 4, "feather_px": 0, "region_soft_px": 0}})
    a = finish(np.full(core.shape, 0.5, np.float32), core, s)
    assert a[60, 60] == 1.0 and a[60, 5] == 0.0 and a[60, 95] == 0.5 and a[60, 88] == 0.5   # core / outside / band
    old = ltx_settings({"ltx": {"combine": "region"}})
    assert settings_sig(old) == settings_sig({**old, "feather_px": 3.0})                     # band keys not in old sigs
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.get(f"/api/object/obj/stage_band/ltx/{start}").status_code == 409
    assert api.put("/api/object/obj/ltx", json={"combine": "band", "region_px": 12, "core_erode_px": 4,
                                                "feather_px": 2, "region_soft_px": 5}).status_code == 200
    st = api.get(f"/api/object/obj/stage_band/ltx/{start}").json()
    assert st["defaults"] == {"out": 12, "in": 4}
    l0 = api.get(f"/api/object/obj/stage_band/ltx/{start}/lines").json()
    assert l0["band_contours"] and l0["inset_feather_contours"] and l0["feather_contours"]
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    r = api.put(f"/api/object/obj/stage_band/ltx/{start}", json={"x": float(xs.max()), "y": float(np.median(ys)), "out": 30})
    assert r.status_code == 200, r.text
    assert api.get("/api/object/obj/stage_band/ltx/params").json()["band_ref_out"] == 12


def test_stage3_lines_read_the_mask12_written_after_the_core(tmp_path, monkeypatch):
    """2026-10-22: stage 1 writes the mask 1+2 (stabilised core, no brush corrections) to core_fix/; the stage-3 band
    preview reads it while it is current instead of stabilising the core again; stale / brush zones = computed."""
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.put("/api/object/obj/settings", json={"band_out_px": 10, "band_in_px": 4}).status_code == 200
    assert pipe._mask12_disk("obj", start) is None                          # nothing on disk yet
    info = pipe.write_mask12("obj", frames)
    assert info and (pipe.store.dir("obj") / "core_fix" / f"c_{start:04d}.png").exists()
    _, pi = pipe.edge_preview("obj", start)
    assert pi["source"].startswith("mask12 (core_fix)")
    assert api.get(f"/api/object/obj/stage_band/matanyone/{start}/lines").json()["band_contours"]
    # a brush correction: stage 2 („Zastosuj poprawki”) owns the mask 1+2 – nothing written automatically, and the
    # mask on disk is stale (its signature misses the zone) -> computed again
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    xm = int(np.median(xs)); top = int(ys[xs == xm].min())
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": start, "polygon": [],
                                                     "strokes": [{"p": [[xm, top + 3], [xm, top - 20]], "radius_px": 4}],
                                                     "params": {"source": "brush", "track": "rigid"}})
    assert r.status_code == 200, r.text
    assert pipe.write_mask12("obj", frames) is None
    assert pipe._mask12_disk("obj", start) is None
