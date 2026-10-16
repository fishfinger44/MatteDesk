# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""2026-10-16 (user report CLIP2_A F58): group crops cover the parts' crops; band widths smoothed along the contour."""
import numpy as np

from engine import edge
from engine.crop import CropSegment, group_segments, square_crop


def _seg(cx, cy, need, w, h, first=1, last=50, sid="s00"):
    crop, pad = square_crop(cx, cy, need, w, h)
    return CropSegment(seg_id=sid, first=first, last=last, overlap_in=0, overlap_out=0, crop=crop, pad=pad,
                       scale_to_1008=1.0)


def _covers(seg, b):
    x, y, s, _ = seg.crop
    return x <= b.x and y <= b.y and x + s >= b.x + b.w and y + s >= b.y + b.h


def test_group_crop_grows_to_the_parts_and_stays_in_the_plate():
    w, h = 1280, 720
    parent = _seg(800, 400, 500, w, h)
    part = _seg(400, 300, 600, w, h)
    g = group_segments([parent], [[part]], w, h)[0]
    assert _covers(g, parent.box()) and _covers(g, part.box())
    assert g.seg_id == parent.seg_id and (g.first, g.last) == (parent.first, parent.last)
    assert g.side % 16 == 0 and "kadr grupy" in g.flags
    # padding only where the square leaves the plate
    l, t, r, b = g.pad
    x, y, s, _ = g.crop
    assert l == max(0, -x) and t == max(0, -y) and r == max(0, x + s - w) and b == max(0, y + s - h)


def test_group_crop_unchanged_when_covered_or_not_overlapping_in_time():
    w, h = 1280, 720
    parent = _seg(640, 360, 600, w, h)
    inside = _seg(640, 360, 100, w, h)
    later = _seg(100, 100, 150, w, h, first=60, last=90)
    assert group_segments([parent], [[inside, later]], w, h)[0] is parent


def test_band_widths_smoothed_along_each_contour():
    rng = np.random.default_rng(0)
    n = 400
    wid = np.clip(rng.normal(7, 3, n), 3, 24).astype(np.float32)
    pieces = [(0, i // 32) for i in range(250)] + [(1, i // 32) for i in range(150)]
    sm = edge._smooth_along_contours(wid, pieces, edge.BAND_CONTOUR_SIGMA)
    assert np.abs(np.diff(sm[:250])).max() < 0.5 * np.abs(np.diff(wid[:250])).max()
    assert np.allclose(edge._smooth_along_contours(np.full(n, 5, np.float32), pieces, 6.0), 5.0, atol=1e-5)
    # contours do not mix: a constant second contour stays constant
    w2 = wid.copy(); w2[250:] = 9.0
    assert np.allclose(edge._smooth_along_contours(w2, pieces, 6.0)[250:], 9.0, atol=1e-5)
    assert np.array_equal(edge._smooth_along_contours(wid, pieces, 0), wid)


def test_band_has_no_tabs_on_a_noisy_edge():
    """A disc on a plate with a noisy rim: the width map along the contour changes smoothly (no square tabs)."""
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    core = (xx - 128) ** 2 + (yy - 128) ** 2 < 80 ** 2
    rng = np.random.default_rng(1)
    g = np.where(core, 0.8, 0.2).astype(np.float32) + rng.normal(0, 0.08, (h, w)).astype(np.float32)
    meas = edge.measure_edge(core.astype(np.float32), g)
    def jumps(sig):
        wm = edge.adaptive_band_width(core, g, None, 180.0, 1.0, contour_sigma=sig)
        p = np.round(meas.points).astype(int)
        v = wm[np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)]
        return np.abs(np.diff(v)).max()
    assert jumps(edge.BAND_CONTOUR_SIGMA) <= jumps(0)


def test_group_zone_sure_fg_per_member_and_whole_core(tmp_path, monkeypatch):
    """2026-10-16 (CLIP2_A: parent „Hands”, part „Head” with the hair): params.sure_fg picks the sure members of a group
    (parent included); absent = the old rule (parts sure); the zone previews work on the whole of the group."""
    import json
    import cv2
    from engine.pipeline import ObjectPipeline
    from engine.zones import Zone
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    store = pipe.store
    store.create("head", ["head"], depth_order=1)
    store.update("head", part_of="obj")
    cd = store.dir("head") / "core"; cd.mkdir(parents=True)
    m = np.zeros((shot.h, shot.w), np.uint8); m[5:25, 5:25] = 255
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    (cd / "core.json").write_text(json.dumps({"coverage": [[frames[0], frames[-1]]]}))
    z = Zone(zone_id="z_01", mode="add", frame=start, polygon=[[0, 0], [10, 0], [10, 10]], criterion="matte", matting="vitmatte")
    fg, names = pipe.zone_fg("obj", z, [start])
    assert names == ["obj"]                    # default: the member in front (depth 0) is sure, the head (1) is matted
    store.update("head", depth_order=0)
    assert pipe.zone_fg("obj", z, [start])[1] == ["head"]              # same depth: the old rule, the parts are sure
    store.update("head", depth_order=1)
    z.params = {"sure_fg": ["obj"]}
    fg, names = pipe.zone_fg("obj", z, [start])
    assert names == ["obj"] and np.array_equal(fg(start), pipe.load_core("obj", [start])[start])   # own core, not a group version
    z.params = {"sure_fg": []}
    assert pipe.zone_fg("obj", z, [start]) == (None, [])
    whole = pipe.preview_core("obj", [start])[start]
    own = pipe.load_core("obj", [start])[start]
    assert whole[10, 10] and (whole | own).sum() == whole.sum() and whole.sum() > own.sum()
    assert np.array_equal(pipe.preview_core("head", [start])[start], m > 127)   # a part alone: its own core


def test_band_px_mode_minimum_out_and_in_no_shift():
    """2026-10-16: band_out_px / band_in_px = at least this far out of / into the mask edge, adaptive wins where wider;
    objects without the fields keep their settings signature (no stage goes stale) and the old ×scale / shift."""
    from engine.segment import sdf
    from engine.stages import settings_sig
    h = w = 200
    yy, xx = np.mgrid[:h, :w]
    core = (xx - 100) ** 2 + (yy - 100) ** 2 < 50 ** 2
    wmap = np.full((h, w), 3.0, np.float32)
    d = sdf(core)
    b = edge.band_px(core, wmap, 20.0, 4.0)
    assert -d[b].min() <= 20.01 and -d[b].min() >= 19 and d[b].max() <= 4.01 and d[b].max() >= 3.5
    b0 = edge.band_px(core, wmap, 0.0, 0.0)
    assert np.array_equal(b0, edge.adaptive_band(core, wmap))          # 0 / 0 = the adaptive band alone
    wide = np.full((h, w), 12.0, np.float32)
    assert d[edge.band_px(core, wide, 5.0, 2.0)].max() >= 11.5          # adaptive wider than the px -> adaptive
    assert edge.band_px_limits({"band_scale": 2}) is None
    assert edge.band_px_limits({"band_out_px": 10}) == (10.0, 0.0)
    old = {"band_scale": 1.0, "shift_px": 0.0, "temporal_smoothing": "normal"}
    assert settings_sig(old) == settings_sig({**old, "band_out_px": None, "band_in_px": None})
    assert settings_sig(old) != settings_sig({**old, "band_out_px": 24.0})


def test_experiments_analysis_of_a_chosen_base(tmp_path, monkeypatch):
    """2026-10-16: stage Eksperymenty – QA + edge proposals computed on the CHOSEN base version (measured once into it),
    thresholds recount the proposals from the saved measurements; a version the fix cannot start from says why."""
    import json
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    ma = r.json()["run"]
    d = api.post("/api/object/obj/analyze", json={"run": "matanyone", "thr_px": 0.5, "min_frames": 5})
    assert d.status_code == 200, d.text
    assert d.json()["run"] == ma and d.json()["fix_why"] == ""
    loose = api.get(f"/api/object/obj/edge_recount?v={ma}&thr_px=0.05&min_frames=1").json()
    strict = api.get(f"/api/object/obj/edge_recount?v={ma}&thr_px=5&min_frames=20").json()
    assert loose["n_fix"] >= strict["n_fix"] and strict["n_fix"] == 0
    d = api.post("/api/object/obj/analyze", json={"run": ma, "thr_px": 0.05, "min_frames": 1}).json()
    ej = json.loads((pipe.store.dir("obj") / "runs" / ma / "edge" / "edge.json").read_text(encoding="utf-8"))
    assert len(ej["fix"]) == loose["n_fix"] == d["edge"]["pieces_to_fix"] and ej["thr_px"] == 0.05
    # the mask 1+2 as a version (base of ViTMatte on 1+2): no QA yet -> measured now, the fix is not for it
    core_run = pipe.core_base_run("obj", frames)
    rd = pipe.store.dir("obj") / "runs" / core_run
    assert not (rd / "qa.json").exists()
    d = api.post("/api/object/obj/analyze", json={"run": core_run}).json()
    assert (rd / "qa.json").exists() and any((rd / "edge").glob("e_*.npz"))
    assert d["fix_why"] == ""                                     # any base can be shifted and recomputed (2026-10-16)
    qa = api.get(f"/api/object/obj/qa?v={core_run}").json()
    assert qa["run"] == core_run and qa["colours"]
    assert api.post("/api/object/obj/analyze", json={"run": "ltx"}).status_code == 404     # no LTX result
    # „Przelicz z tej bazy”: the mask 1+2 version through MatAnyone, with the edge correction of the base's pieces
    pieces = [p["id"] for p in api.post("/api/object/obj/analyze", json={"run": core_run, "thr_px": 0.05,
                                                                         "min_frames": 1}).json()["edge"]["pieces"]]
    r = api.post("/api/object/obj/experiment", json={"base": core_run, "mode": "matanyone", "fix": pieces[:2]})
    assert r.status_code == 200, r.text
    new = r.json()["run"]
    nrj = json.loads((pipe.store.dir("obj") / "runs" / new / "run.json").read_text(encoding="utf-8"))
    assert nrj["stage"] == "experiment" and nrj["base_run"] == core_run and nrj["base_stage"] == "core"
    assert nrj["base_edge_fix"] == pieces[:2]
    assert (pipe.store.dir("obj") / "runs" / new / "qa.json").exists()
    runs = {x["run"]: x for x in api.get("/api/object/obj/runs").json()["runs"]}
    assert runs[new]["base_run"] == core_run and runs[new]["active"]
    # hard engine: the base mask after the zones / fix, no soft edge
    r = api.post("/api/object/obj/experiment", json={"base": new, "mode": "core"})
    assert r.status_code == 200, r.text
    assert api.post("/api/object/obj/experiment", json={"base": new, "mode": "zzz"}).status_code == 409
    # stage LTX may take a version of the Experiments as its mask (not any other version)
    assert api.put("/api/object/obj/ltx", json={"mask_source": new}).status_code == 200
    assert api.get(f"/api/object/obj/ltx_preview/{start}?what=band").status_code == 200
    assert api.put("/api/object/obj/ltx", json={"mask_source": core_run}).status_code == 200
    rb = api.get(f"/api/object/obj/ltx_preview/{start}?what=band")
    assert rb.status_code == 404 and "Eksperyment" in rb.json()["detail"]
