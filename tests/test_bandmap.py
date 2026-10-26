# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Variable band width from control points on the outline (engine/bandmap.py, 2026-10-21)."""

import cv2
import numpy as np

from engine import bandmap as B
from engine.zones import outline_band


def _disk(h=200, w=200, r=60, c=(100, 100)):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def _band_width_at(band, core, x, y, step=(1, 0)):
    """Band pixels counted from the contour point (x, y) outwards along `step` (outside part only)."""
    n = 0
    while 0 <= x < band.shape[1] and 0 <= y < band.shape[0]:
        if not core[y, x]:
            if not band[y, x]:
                break
            n += 1
        x, y = x + step[0], y + step[1]
    return n


def test_no_points_defaults_and_gap():
    core = _disk()
    o, i = B.width_maps(core, [], 10, 4)
    assert np.allclose(o, 10) and np.allclose(i, 4)
    band = B.variable_band(core, o, i)
    ref = outline_band(core, 10, 4)
    assert abs(int(band.sum()) - int(ref.sum())) / ref.sum() < 0.05      # same band as the disk kernels (L2 vs disk)
    zero = B.variable_band(core, np.zeros_like(o), np.zeros_like(i))
    assert not zero.any()                                               # width 0 = no band at all


def test_two_points_interpolate_along_arc():
    core = _disk()
    pts = [(160.0, 100.0, 30.0, 4.0), (40.0, 100.0, 2.0, 4.0)]          # right: 30 px out, left: 2 px out
    o, i = B.width_maps(core, pts, 10, 4)
    band = B.variable_band(core, o, i)
    right = _band_width_at(band, core, 160, 100, (1, 0))
    left = _band_width_at(band, core, 39, 100, (-1, 0))
    top = _band_width_at(band, core, 100, 39, (0, -1))
    assert 28 <= right <= 31 and 1 <= left <= 3
    assert 12 <= top <= 20                                              # half way along the arc: ~16 px


def test_gap_point_cuts_the_band():
    core = _disk()
    o, i = B.width_maps(core, [(160.0, 100.0, 0.0, 0.0)], 10, 4)       # one point with 0 = the whole contour 0
    assert not B.variable_band(core, o, i).any()


def test_contour_without_points_keeps_defaults():
    core = _disk(h=200, w=400, r=50, c=(100, 100)) | _disk(h=200, w=400, r=50, c=(300, 100))
    o, _ = B.width_maps(core, [(150.0, 100.0, 25.0, 4.0)], 8, 4)        # point only on the left disk
    assert abs(o[100, 165] - 25) < 1e-3 and abs(o[100, 360] - 8) < 1e-3


def test_widths_animate_between_keys_and_hold():
    p = {"id": "p1", "keys": {"10": {"x": 0, "y": 0, "out": 10, "in": 2}, "20": {"x": 0, "y": 0, "out": 30, "in": 6}}}
    assert B.widths_at(p, 5) == (10, 2) and B.widths_at(p, 25) == (30, 6)
    assert B.widths_at(p, 15) == (20, 4)


def test_occluded_point_does_not_shape_the_band():
    pts = [{"id": "a", "keys": {"0": {"x": 160, "y": 100, "out": 30, "in": 4}}},
           {"id": "b", "keys": {"0": {"x": 40, "y": 100, "out": 2, "in": 4}}}]
    at = B.points_at(pts, 0, {"a": (160, 100, True), "b": (40, 100, False)})
    assert at == [(160, 100, 30.0, 4.0)]
    pos = B.static_positions(pts, [0, 5])
    assert pos[5]["a"] == (160.0, 100.0, True)


def test_api_band_points_shape_the_outline_zone(tmp_path, monkeypatch):
    """Band control points on a "whole outline" zone (2026-10-21): a point with a wide outer whisker widens the band on
    its side; width 0 cuts it; removing the last point = the constant band again (bit-identical region)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)          # no CoTracker: nearest key
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    reg0 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert api.get(f"/api/object/obj/zones/{zid}/band/{start}").json()["points"] == []
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    right = (float(xs.max()), float(np.median(ys[xs == xs.max()])))
    r = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": right[0], "y": right[1], "out": 40})
    assert r.status_code == 200, r.text
    pid = r.json()["point"]
    st = r.json()["band"]["points"][0]
    assert st["id"] == pid and st["key"] and st["out"] == 40 and st["visible"]
    reg1 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert reg1["area_px"] > reg0["area_px"] and reg1["bbox"][2] > reg0["bbox"][2] + 20     # wider, on the right
    other = next(f for f in frames if f != start)
    r = api.put(f"/api/object/obj/zones/{zid}/band/{other}", json={"id": pid, "out": 0, "in": 0})   # key: gap
    assert r.status_code == 200 and r.json()["band"]["points"][0]["keys"] == sorted([start, other])
    # the gap: the band of the point's contour is gone; the preview now builds the band on the stabilised mask 1+2 (as
    # the runs, 2026-10-24) – a speck of it may form a contour of its own without points (zone defaults), so: ~nothing left
    assert api.get(f"/api/object/obj/zones/{zid}/region/{other}").json()["area_px"] < 0.05 * reg0["area_px"]
    assert api.delete(f"/api/object/obj/zones/{zid}/band/{pid}?frame={other}").status_code == 200
    assert api.delete(f"/api/object/obj/zones/{zid}/band/{pid}").status_code == 200
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert "band_points" not in z["params"]
    assert api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()["area_px"] == reg0["area_px"]


def test_sliders_size_the_band_points_shape_it():
    """„Pas na zewnątrz” / „Wcięcie w rdzeń” scale the point widths by slider / slider-at-first-point (2026-10-21)."""
    from engine.zones import band_refs, band_scales
    assert band_scales({"outline_out_px": 12}) == (1.0, 1.0)                    # no reference = no scaling
    p = {"outline_out_px": 6, "outline_in_px": 4}
    p.update(band_refs(p))
    p["outline_out_px"], p["outline_in_px"] = 12, 2
    assert band_scales(p) == (2.0, 0.5)
    pts = [{"id": "a", "keys": {"0": {"x": 160, "y": 100, "out": 30, "in": 4}}}]
    assert B.points_at(pts, 0, {"a": (160, 100, True)}, *band_scales(p)) == [(160, 100, 60.0, 2.0)]


def test_api_slider_scales_the_point_band(tmp_path, monkeypatch):
    """The outer slider sizes the whole band the points shape: doubling it doubles the whiskers (state and region);
    a width typed while scaled is stored unscaled; removing the last point drops the reference."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    right = (float(xs.max()), float(np.median(ys[xs == xs.max()])))
    pid = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": right[0], "y": right[1], "out": 10}).json()["point"]
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert z["params"]["band_ref_out"] == 6
    reg1 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    r = api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**z["params"], "outline_out_px": 12}})
    assert r.status_code == 200, r.text
    st = api.get(f"/api/object/obj/zones/{zid}/band/{start}").json()
    assert st["scale"]["out"] == 2.0 and st["points"][0]["out"] == 20
    reg2 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert reg2["area_px"] > reg1["area_px"]
    r = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"id": pid, "out": 30})      # shown 30 = stored 15
    assert r.json()["band"]["points"][0]["out"] == 30
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert z["params"]["band_points"][0]["keys"][str(start)]["out"] == 15
    assert api.delete(f"/api/object/obj/zones/{zid}/band/{pid}").status_code == 200
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert "band_ref_out" not in z["params"] and "band_ref_in" not in z["params"]


def test_api_delete_band_keys_of_a_frame_and_all_points(tmp_path, monkeypatch):
    """🔑✕ / Delete / RMB on the triangle (2026-10-21): every point's key in that frame goes, a point left without keys
    goes too; DELETE without a frame removes all points (the constant band again, no slider reference)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    other = next(f for f in frames if f != start)
    a = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.max()), "y": float(ys.mean()), "out": 20}).json()["point"]
    b = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.min()), "y": float(ys.mean()), "out": 3}).json()["point"]
    api.put(f"/api/object/obj/zones/{zid}/band/{other}", json={"id": a, "out": 30})
    assert api.delete(f"/api/object/obj/zones/{zid}/band?frame={other + 1000}").status_code == 404
    r = api.delete(f"/api/object/obj/zones/{zid}/band?frame={start}").json()
    assert sorted(r["removed"]) == sorted([a, b]) and r["points"] == [a]          # b had only this key: gone
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert list(z["params"]["band_points"][0]["keys"]) == [str(other)]
    r = api.delete(f"/api/object/obj/zones/{zid}/band").json()
    assert r["removed"] == [a] and r["points"] == []
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert "band_points" not in z["params"] and "band_ref_out" not in z["params"]


def test_api_undo_restores_band_points_and_reference(tmp_path, monkeypatch):
    """Ctrl+Z in stage 5 (2026-10-21) PATCHes the zone params back: band points as they were, slider reference None when
    none were left – the next first point takes the sliders as they are then (scale 1)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    # a detached zone keeps its own points in its params (2026-10-22: the shared object band undoes through /band/params)
    assert api.put(f"/api/object/obj/stage_band/zone:{zid}/mode", json={"mode": "own"}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.max()), "y": float(ys.mean()), "out": 20})
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    params = {**z["params"], "band_ref_out": None, "band_ref_in": None}
    params.pop("band_points")
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**params, "outline_out_px": 12}}).status_code == 200
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert "band_points" not in z["params"]
    r = api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.max()), "y": float(ys.mean()), "out": 30})
    assert r.json()["band"]["scale"]["out"] == 1.0 and r.json()["band"]["points"][0]["out"] == 30
    z = api.get("/api/object/obj/zones").json()["zones"][0]
    assert z["params"]["band_ref_out"] == 12


def test_points_are_tracked_from_inside_the_object(tmp_path):
    """Station 2026-10-21 (fast pan): a point on the hair/sky edge followed the background. The tracker gets a point
    moved inwards along the normal (inset_queries); the band snaps it back to the outline in every frame."""
    from types import SimpleNamespace
    from engine import bandtrack as T
    core = _disk()                                         # centre (100, 100), r 60
    seen = {}

    class Tr:
        def track(self, imgs, q):
            seen["q"] = q.copy()
            t = np.repeat(q[None, :, 1:], len(imgs), axis=0)
            return SimpleNamespace(tracks=t, visible=np.ones(t.shape[:2], bool))

    pts = [{"id": "p1", "keys": {"0": {"x": 160.0, "y": 100.0, "out": 30, "in": 4}}}]
    pos = T.track_positions(pts, [0, 1, 2], Tr(), lambda fs: ([np.zeros((4, 4, 3), np.uint8)] * len(fs), 1.0),
                            tmp_path, core_at=lambda k: core)
    qx = float(seen["q"][0, 1])
    d = np.clip(T.TRACK_INSET_FRAC * np.sqrt(core.sum()), *T.TRACK_INSET_PX)
    assert abs(qx - (159 - d)) < 1.5                       # tracked from inside, not from the edge pixel
    assert pos[0]["p1"][:2] == (160.0, 100.0)              # the key frame keeps the key position
    assert abs(pos[2]["p1"][0] - qx) < 1e-3
    o, _ = B.width_maps(core, B.points_at(pts, 2, pos[2]), 10, 4)
    assert abs(o[100, 165] - 30) < 1e-3                    # snapped back: the width still lands on the right side


def test_lost_track_follows_the_object_and_front_object_occludes(tmp_path):
    """A track left behind (the object moved 80 px, the tracker kept the point) is lost, not occluded: the point moves
    with the object's centroid. Visibility = no object in front covers it (user decision 2026-10-18)."""
    from types import SimpleNamespace
    from engine import bandtrack as T
    cores = {0: _disk(w=300, c=(100, 100)), 1: _disk(w=300, c=(180, 100))}

    class Stay:
        def track(self, imgs, q):
            t = np.repeat(q[None, :, 1:], len(imgs), axis=0)
            return SimpleNamespace(tracks=t, visible=np.zeros(t.shape[:2], bool))   # CoTracker3: "occluded"

    pts = [{"id": "p1", "keys": {"0": {"x": 40.0, "y": 100.0, "out": 20, "in": 4}}}]   # left edge
    front = np.zeros((200, 300), bool)
    pos = T.track_positions(pts, [0, 1], Stay(), lambda fs: ([np.zeros((4, 4, 3), np.uint8)] * len(fs), 1.0),
                            tmp_path, core_at=lambda f: cores[f], occluder_at=lambda f: front)
    x, y, vis = pos[1]["p1"]
    assert abs(x - 120) < 2 and abs(y - 100) < 2 and vis        # shifted by +80 with the object, visible
    front[:, 100:140] = True                                    # a hand in front over the left edge
    pos = T.track_positions(pts, [0, 1], Stay(), lambda fs: ([np.zeros((4, 4, 3), np.uint8)] * len(fs), 1.0),
                            tmp_path, core_at=lambda f: cores[f], occluder_at=lambda f: front, resolve=[1])
    assert pos[1]["p1"][2] is False


def test_front_object_read_once_per_frame_not_per_point(tmp_path):
    """Woman_02 behind Woman_01 (2026-10-26): the occluder is a 4K alpha read from disk – one read per frame and pass,
    not one per point (26 points x 148 frames kept VideoMaMa ~15 min in „Śledzenie punktów pasa”)."""
    from types import SimpleNamespace
    from engine import bandtrack as T
    cores = {f: _disk(w=300, c=(100 + 10 * f, 100)) for f in range(4)}

    class Stay:
        def track(self, imgs, q):
            t = np.repeat(q[None, :, 1:], len(imgs), axis=0)
            return SimpleNamespace(tracks=t, visible=np.ones(t.shape[:2], bool))

    pts = [{"id": f"p{i}", "keys": {"0": {"x": 40.0 + i, "y": 100.0, "out": 20, "in": 4}}} for i in range(10)]
    reads = []
    T.track_positions(pts, list(range(4)), Stay(), lambda fs: ([np.zeros((4, 4, 3), np.uint8)] * len(fs), 1.0),
                      tmp_path, core_at=lambda f: cores[f],
                      occluder_at=lambda f: reads.append(f) or np.zeros((200, 300), bool))
    assert len(reads) <= 2 * 4          # the core-motion pass + the tracked pass, each frame once


def test_tracking_once_in_background_and_shared(tmp_path):
    """Station 2026-10-21: every band / region / preview request started its own CoTracker3 worker (7 at once).
    Interactive calls (wait=False) get the nearest keys' positions at once and one background tracking; callers of the
    same point set share it; the full run (wait=True) waits for it."""
    import threading
    import time
    from types import SimpleNamespace
    from engine import bandtrack as T
    calls = []
    gate = threading.Event()

    class Slow:
        def track(self, imgs, q):
            calls.append(1)
            gate.wait(5)
            t = np.repeat(q[None, :, 1:], len(imgs), axis=0) + 3.0
            return SimpleNamespace(tracks=t, visible=np.ones(t.shape[:2], bool))

    pts = [{"id": "p1", "keys": {"0": {"x": 50.0, "y": 60.0, "out": 20, "in": 4}}}]
    frames_fn = lambda fs: ([np.zeros((4, 4, 3), np.uint8)] * len(fs), 1.0)
    info = {}
    pos = T.track_positions(pts, [0, 1, 2], Slow(), frames_fn, tmp_path, wait=False, info=info)
    assert info.get("pending") and pos[2]["p1"][:2] == (50.0, 60.0)          # at once: the key's position
    info2 = {}
    T.track_positions(pts, [0, 1, 2], Slow(), frames_fn, tmp_path, wait=False, info=info2)
    assert info2.get("pending")
    gate.set()
    full = T.track_positions(pts, [0, 1, 2], Slow(), frames_fn, tmp_path)    # waits for the running one
    assert full[2]["p1"][:2] == (53.0, 63.0) and len(calls) == 1
    later = {}
    assert T.track_positions(pts, [0, 1, 2], Slow(), frames_fn, tmp_path, wait=False, info=later)[2]["p1"][0] == 53.0
    assert not later.get("pending") and len(calls) == 1                       # cached: no new tracking


def test_core_motion_positions_and_crop_offsets(tmp_path):
    """User idea 2026-10-21: use the core. Before / without the tracker a point is carried by the core's centroid and
    scale from its key; tracker frames may come from a crop that follows the object – tracks are mapped back."""
    from types import SimpleNamespace
    from engine import bandtrack as T
    cores = {0: _disk(w=400, c=(100, 100), r=50), 1: _disk(w=400, c=(250, 100), r=60)}   # moves +150, grows x1.2
    pts = [{"id": "p1", "keys": {"0": {"x": 150.0, "y": 100.0, "out": 20, "in": 4}}}]    # right edge of the key disk
    pos = T.track_positions(pts, [0, 1], None, None, tmp_path, core_at=lambda f: cores[f])
    x, y, vis = pos[1]["p1"]
    assert abs(x - (250 + 50 * 1.2)) < 2 and abs(y - 100) < 2 and vis    # the right edge of the moved, larger disk

    seen = {}

    class Echo:                                      # the tracker returns the query position in every frame
        def track(self, imgs, q):
            seen["q"] = q.copy()
            return SimpleNamespace(tracks=np.repeat(q[None, :, 1:], len(imgs), axis=0),
                                   visible=np.ones((len(imgs), len(q)), bool))

    def crop_frames(fs):                             # crop origins (plate px) per frame, scale 0.5
        return [np.zeros((50, 50, 3), np.uint8)] * len(fs), 0.5, np.array([[100.0, 40.0], [130.0, 40.0]])
    crop_frames.variant = "test"
    pos = T.track_positions(pts, [0, 1], Echo(), crop_frames, tmp_path)
    assert np.allclose(seen["q"][0, 1:], [(150 - 100) * 0.5, (100 - 40) * 0.5])        # query in crop coordinates
    assert abs(pos[1]["p1"][0] - (150 + 30)) < 1e-3                                    # back to the plate (+ crop move)


def test_region_returns_the_whole_band_ring(tmp_path, monkeypatch):
    """User report 2026-10-21: with band points the band vanished from the preview – GET region gave the outer edge only.
    An outline zone also returns band_contours: outer + inner edges (RETR_CCOMP) for an even-odd fill."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert len(reg["contours"]) == 1 and len(reg["band_contours"]) >= 2      # the ring: outer + inner edge


def test_feathers_meet_inside_a_narrow_band():
    """User rule 2026-10-22: the outer feather (from the outer band edge inwards) never passes the end of the inset
    feather (from the inner band edge outwards) – they touch at most; the inset feather keeps its length."""
    from engine.bandsys import area_weight as feather_weight
    core = _disk(r=50)
    band = (cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= 17) & ~core
    w = feather_weight(band, core, inner_px=11, outer_px=51)
    row = w[100, 150:170]
    assert row.max() > 0.95                                   # (nearly) full zone weight where the feathers touch
    peak = 150 + int(np.argmax(row))
    assert 150 + 11 - 2 <= peak <= 150 + 11 + 2               # the inset feather keeps its 11 px, the outer gets ~6
    big = _disk(h=400, w=400, r=50, c=(200, 200))
    wide = (cv2.distanceTransform((~big).astype(np.uint8), cv2.DIST_L2, 5) <= 80) & ~big
    ww = feather_weight(wide, big, inner_px=11, outer_px=51)  # a wide band: the feathers keep their own length
    assert ww[200, 250 + 11 + 2] > 0.95 and ww[200, 250 + 80 - 51 - 2] > 0.95 and ww[200, 250 + 80 - 10] < 0.3


def test_inset_feather_line_inside_the_core(tmp_path, monkeypatch):
    """User report 2026-10-22: with a deep core inset (band 17 px into the core) the inset-feather line showed the core
    edge – the core hid the feather strip. The line must sit `inset_feather_px` from the inner band edge."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    params = {"region": "outline", "outline_out_px": 20, "fg_erode_px": 13, "force_core_erode_px": 13, "inset_feather_px": 4,
              "outer_feather_px": 6, "temporal": "none"}
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair", "params": params}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    core = pipe.load_core("obj", [start])[start].astype(bool)
    d_in = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    pts = np.array([q for c in reg["inset_feather_contours"] for q in c])
    depth = d_in[np.clip(pts[:, 1], 0, core.shape[0] - 1), np.clip(pts[:, 0], 0, core.shape[1] - 1)]
    assert 10 <= np.median(depth) <= 15, np.median(depth)        # inner edge 17 px deep, feather 4 px -> ~13 px


def test_outer_feather_weight_and_region_line(tmp_path, monkeypatch):
    """User request 2026-10-21: a separate feather at the OUTER band edge (towards the background), shown as a thinner
    dashed line. Weight ramps over outer_px from the background side and over inner_px from the core side."""
    from engine.bandsys import area_weight as feather_weight
    core = _disk(r=50)
    band = (cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) <= 30) & ~core   # 30 px outside the core
    w = feather_weight(band, core, inner_px=4, outer_px=20)
    # along a ray to the right: core edge at x=150, band to x~180, background beyond
    assert w[100, 152] < 0.75                       # next to the core: inset feather (4 px)
    assert w[100, 165] > 0.5 and w[100, 178] < 0.2  # fades over 20 px towards the outer edge
    assert not w[~band].any()
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 20, "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    # 0 = „jak wcięcie”: the outer edge feathers over the inset length, so its line is drawn too (review 2026-10-22)
    reg0 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert reg0["feather_contours"] and reg0["band_contours"]
    api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**body["params"], "outer_feather_px": 10}})
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert reg["feather_contours"] and reg["band_contours"] and reg["feather_contours"] != reg0["feather_contours"]


def test_tilted_whiskers_shift_the_band_along_the_outline_2026_10_26():
    """A whisker pair turned round its point: the band edge goes through the whisker END (cos(a)·L away from the outline,
    sin(a)·L along it); tilt 0 = bit-identical to the 4-tuple points; keys without "ang" read as 0."""
    core = _disk()
    base = [(160.0, 100.0, 30.0, 4.0), (40.0, 100.0, 2.0, 4.0)]
    o0, i0 = B.width_maps(core, base, 10, 4)
    o1, i1 = B.width_maps(core, [base[0] + (0.0,), base[1]], 10, 4)
    assert np.array_equal(o0, o1) and np.array_equal(i0, i1)
    o2, _ = B.width_maps(core, [(160.0, 100.0, 30.0, 4.0, 60.0), base[1]], 10, 4)
    band0, band2 = B.variable_band(core, o0, i0), B.variable_band(core, o2, i0)
    assert _band_width_at(band2, core, 160, 100, (1, 0)) < _band_width_at(band0, core, 160, 100, (1, 0))   # cos 60° = ½
    assert 13 <= _band_width_at(band2, core, 160, 100, (1, 0)) <= 17
    # the outer band edge goes through the whisker END (within ~2 px), also on a curved outline
    sx, sy, nx, ny, tx, ty = B.snap_frame(core, 160.0, 100.0)[:6]
    ex, ey = sx + 30.0 * (nx * np.cos(np.radians(60)) + tx * np.sin(np.radians(60))), sy + 30.0 * (ny * np.cos(np.radians(60)) + ty * np.sin(np.radians(60)))
    edge = band2 & ~core
    ys, xs = np.nonzero(edge & ~cv2.erode(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    assert float(np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2).min()) < 3.0
    assert not np.array_equal(band0, band2)
    p = {"id": "a", "keys": {"5": {"x": 1, "y": 1, "out": 1, "in": 1, "ang": 40}, "15": {"x": 1, "y": 1, "out": 1, "in": 1}}}
    assert B.angle_at(p, 5) == 40 and B.angle_at(p, 10) == 20 and B.angle_at(p, 99) == 0
    assert B.points_at([p], 5, {"a": (1, 1, True)})[0][4] == 40
    sn = B.snap_frame(core, 160.0, 100.0)
    assert B.snap_normal(core, 160.0, 100.0) == sn[:4] and abs(sn[2] * sn[4] + sn[3] * sn[5]) < 1e-6


def test_tilted_whisker_crossing_a_thin_part_makes_no_hole_2026_10_26():
    """An inner whisker longer than a thin limb: its end is past the outline – the band must not pinch to 0 there
    (user report 2026-10-26: holes in the band), and the band edge stays smooth (no jagged steps)."""
    core = np.zeros((200, 300), bool)
    core[80:120, 20:280] = True                                           # a 40 px thick limb
    pts = [(150.0, 80.0, 30.0, 88.0, 29.0)]                               # 88 px inside crosses the limb
    o, i = B.width_maps(core, pts, 10, 4)
    band = B.variable_band(core, o, i)
    assert band[100, 150] and band[100, 120] and band[100, 180]           # the limb under the point is covered, no hole
    top = o[78, 20:280]
    assert float(np.abs(np.diff(top)).max()) < 3.0                       # no jump along the outline


def test_speck_next_to_a_pointed_contour_takes_its_widths_2026_10_26():
    """A split-off speck of the mask has no point of its own: it must not keep the default band (wedges of 62 px on a
    24 px edge, user report 2026-10-26) – it takes the widths of the nearest pointed outline; a far contour keeps the default."""
    core = np.zeros((300, 600), bool)
    core[100:200, 50:250] = True                                   # the object, one point: 10 px out / 4 in
    core[140:150, 262:270] = True                                  # a speck 12 px from the object
    core[140:150, 560:568] = True                                  # a far speck (another object): the default
    o, _ = B.width_maps(core, [(150.0, 100.0, 10.0, 4.0)], 60, 4)
    assert abs(float(o[145, 266]) - 10.0) < 1e-3                  # the near speck = the object's width
    assert abs(float(o[145, 564]) - 60.0) < 1e-3                  # the far one keeps the constant band
