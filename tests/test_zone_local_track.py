# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Painted ViTMatte band, rotobrush-like (user decision 2026-10-15): the band is painted along the hair / background
edge, tracked LOCALLY by CoTracker points (never on a hand that is a part of the object or an object in front), the
part (one matte with the head) is cut out of the band and is sure foreground of the trimap; "Zmaż obszar" strokes."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine.tracking import TrackResult  # noqa: E402
from engine.zones import LocalWarp, Zone, local_warp_mask, propagate_regions, raster_drawing  # noqa: E402


def test_erase_stroke_removes_from_the_drawing():
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    m = raster_drawing(poly, [{"p": [[50, 10], [50, 90]], "radius_px": 5, "erase": True}], (100, 100))
    assert m[50, 20] and m[50, 80] and not m[50, 50]
    m2 = raster_drawing([], [{"p": [[30, 50]], "radius_px": 8}, {"p": [[30, 50]], "radius_px": 3, "erase": True}], (100, 100))
    assert m2[50, 36] and not m2[50, 30]                      # erase after add, whatever the order in the list


def test_local_warp_follows_a_non_rigid_motion():
    H, W = 200, 300
    mask = np.zeros((H, W), bool)
    mask[80:120, 40:80] = True                                # left blob
    mask[80:120, 220:260] = True                              # right blob
    ys, xs = np.mgrid[60:140:10, 20:280:10]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    dst = pts.copy()
    dst[pts[:, 0] < 150, 1] += 12                             # left half moves down, right half up
    dst[pts[:, 0] >= 150, 1] -= 12
    lw = LocalWarp(pts=pts, dst=dst, vis=np.ones(len(pts), bool), fallback=np.eye(2, 3, dtype=np.float32), sigma=30)
    out = local_warp_mask(mask, lw)
    cy = lambda m, x0, x1: float(np.where(m[:, x0:x1])[0].mean())
    assert abs(cy(out, 0, 150) - (cy(mask, 0, 150) + 12)) < 2.0
    assert abs(cy(out, 150, 300) - (cy(mask, 150, 300) - 12)) < 2.0
    # too few visible points: the rigid fallback
    lw2 = LocalWarp(pts=pts[:3], dst=pts[:3], vis=np.ones(3, bool), fallback=np.array([[1, 0, 5], [0, 1, 0]], np.float32))
    assert np.array_equal(local_warp_mask(mask, lw2), np.roll(mask, 5, axis=1))


class _FakeTracker:
    """Moves every point down by 2 px per frame on the left half and up on the right half; records the queries."""
    engine = "cotracker3"

    def __init__(self):
        self.queries = None

    def track(self, frames, q):
        self.queries = q.copy()
        T, N = len(frames), len(q)
        tracks = np.zeros((T, N, 2), np.float32)
        t0 = int(q[0, 0])
        for t in range(T):
            d = (t - t0) * 2.0
            tracks[t, :, 0] = q[:, 1]
            tracks[t, :, 1] = q[:, 2] + np.where(q[:, 1] < 150, d, -d)
        return TrackResult(tracks=tracks, visible=np.ones((T, N), bool), engine=self.engine)


def test_propagate_local_mode_skips_the_excluded_object():
    H, W = 200, 300
    frames = list(range(1, 8))
    core = np.zeros((H, W), bool)
    core[40:160, 20:280] = True
    cores = {f: core for f in frames}
    band = [{"p": [[30, 40], [270, 40]], "radius_px": 10}]          # painted band along the top edge
    hand = np.zeros((H, W), bool)
    hand[40:160, 120:180] = True                                     # a hand over the middle of the head
    tr = _FakeTracker()
    z = Zone("z_01", "add", "matte", 4, polygon=[], strokes=band, matting="vitmatte", params={"track": "local"})
    regions, bgs, T, info = propagate_regions(z, frames, cores, (H, W), lambda sub: ([np.zeros((H, W, 3), np.float32)] * len(sub), 1.0),
                                              tr, point_exclude=lambda k: hand)
    q = tr.queries[:, 1:]
    assert len(q) > 20 and not ((q[:, 0] >= 112) & (q[:, 0] < 188)).any()     # no point on the hand (+8 px)
    assert isinstance(T["4"][1], LocalWarp) and info["tracked"]
    ys_l = np.where(regions[7][:, 30:100])[0].mean(); ys_r = np.where(regions[7][:, 200:270])[0].mean()
    ys0 = np.where(regions[4][:, 30:100])[0].mean()
    assert ys_l > ys0 + 4 and ys_r < ys0 - 4                                   # each side follows its own points


def test_part_of_the_object_is_cut_out_of_the_band_and_is_foreground(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    from test_zone_matte import _api_project
    from test_zone_vitmatte import MockRunner
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    y_top, x_mid = int(ys.min()), int(np.median(xs))
    store = pipe.store
    store.create("hand", ["hand"], depth_order=0)
    store.update("hand", part_of="obj")
    store.update("obj", depth_order=1)     # the hand lies ON the head: in front (2026-10-16: the front members are sure)
    cd = store.dir("hand") / "core"; cd.mkdir(parents=True)
    m = np.zeros((shot.h, shot.w), np.uint8)
    m[max(0, y_top - 40):y_top + 40, x_mid - 30:x_mid + 30] = 255     # the "hand" crosses the top edge of the object
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    band = [{"p": [[int(xs.min()), y_top], [int(xs.max()), y_top]], "radius_px": 20}]
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "strokes": band, "matting": "vitmatte",
            "preset": "hair", "params": {"temporal": "none", "track": "local"}}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=body)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    z = next(x for x in __import__("engine.zones", fromlist=["load_zones"]).load_zones(store.dir("obj")) if x.zone_id == zid)
    regs, _, info = pipe.zone_regions("obj", z, frames, pipe.load_core("obj", frames))
    hand = m > 127
    assert info["fg_objects"] == ["hand"] and "hand" in info.get("point_exclude", [])
    assert regs[start].any() and not (regs[start] & hand).any()                 # the part is out of the band
    # the trimap the model saw: the hand is sure foreground (255) where it meets the band
    tris = [c_ for c_ in runner.calls]
    assert tris, "ViTMatte ran for the preview"
    fg_fn, names = pipe.zone_fg("obj", z, [start])
    assert names == ["hand"] and fg_fn(start)[y_top, x_mid]
    # live finish in the UI (2026-10-16): the preview also writes the alpha before the finish and its area
    for v in ("zone_raw", "zone_area"):
        rv = api.get(f"/api/object/obj/frame/{start}?view={v}")
        assert rv.status_code == 200, (v, rv.text)
    area = cv2.imread(str(store.dir("obj") / "preview" / f"zarea_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert (area == 128).any() and set(np.unique(area)) <= {0, 128, 255}


def test_fg_extra_is_sure_foreground_in_the_trimap(monkeypatch):
    import engine.vitmatte as vm
    from engine.zones import apply_zones
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(3)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    hand = np.zeros_like(core)
    hand[int(ys.min()):int(ys.min()) + 20, int(xs.min()):int(xs.min()) + 30] = True
    counts = {}
    for label, fgz in (("plain", None), ("hand", {"z_01": lambda f: hand})):
        r = MockRunner()
        monkeypatch.setattr(vm, "get_runner", lambda model_id=None, r=r: r)
        z = Zone("z_01", "add", "matte", frames[0], polygon=poly, matting="vitmatte", params={"temporal": "none", "tile_px": 0})
        apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, fg_by_zone=fgz)
        counts[label] = r.calls[0][2]                     # sure-FG pixels of the first trimap
    assert counts["hand"] > counts["plain"]
