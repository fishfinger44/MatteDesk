# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plan docs/PLAN_EDGE_SPEED.md B1 / B2 (2026-10-28), options of the ViTMatte zone smoothing `flow_mean` that change the
result (A/B: local_results/ab_zone_smooth/): B1 `temporal_crop: follow` (zones.follow_crops – one-size crops that follow
the zone) and B2 `temporal_flow_scale` (DIS on shrunk grays, refine._FlowPool scale). Off = bit-identical to before."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_zone_smooth_window import _scene  # noqa: E402

from engine.refine import _FlowPool, temporal_mean_flow  # noqa: E402
from engine.zones import TEMPORAL_CONF_PARAMS, TEMPORAL_CROPS, follow_crops, smooth_zone_alphas  # noqa: E402


def test_follow_crops_one_size_cover_the_region_inside_the_frame():
    H, W = 200, 300
    frames = list(range(10))
    regions = {}
    for t in frames:                     # a region panning right and down, its size breathing
        r = np.zeros((H, W), bool)
        r[20 + 4 * t: 70 + 5 * t, 10 + 18 * t: 60 + 19 * t] = True
        regions[t] = r
    cr = follow_crops(regions, frames)
    sizes = {(c[1] - c[0], c[3] - c[2]) for c in cr.values()}
    assert len(sizes) == 1
    for t in frames:
        y0, y1, x0, x1 = cr[t]
        assert 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W
        ys, xs = np.where(regions[t])
        assert y0 <= ys.min() and ys.max() < y1 and x0 <= xs.min() and xs.max() < x1
    union = np.logical_or.reduce([regions[t] for t in frames])
    ys, xs = np.where(union)
    ch, cw = next(iter(sizes))
    assert ch * cw < 0.5 * (ys.ptp() + 1) * (xs.ptp() + 1)      # far smaller than the union over the pan


def test_flow_scale_one_is_the_old_flow_and_half_measures_the_shift():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur((rng.random((140, 180)) * 255).astype(np.uint8), (0, 0), 1.5)
    a, b = (np.ascontiguousarray(x) for x in (tex[10:130, 10:170], tex[13:133, 6:166]))   # b = a moved by (+4, -3) px
    u8 = lambda i: (a, b)[i]
    full = _FlowPool(u8, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, 1).get(0, 1)
    ref = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)
    assert np.array_equal(full, ref)
    half = _FlowPool(u8, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, 1, scale=0.5).get(0, 1)
    assert half.shape == full.shape and half.dtype == np.float32
    mid = (slice(30, 90), slice(40, 120))
    assert abs(float(np.median(half[mid][..., 0])) - 4.0) < 0.5
    assert abs(float(np.median(half[mid][..., 1])) + 3.0) < 0.5


def test_options_off_are_bit_identical_and_on_stay_close():
    frames, imgs, alphas, regions, force, fg = _scene(9, seed=3)
    bbox = (4, 46, 44, 68)
    fin = {"alpha_black": 0.1, "alpha_white": 0.9}
    run = lambda conf, crops=None: _smooth(frames, imgs, alphas, regions, force, bbox, fin, conf, crops)
    base = run({})
    same = run({"flow_scale": 1.0}, {f: bbox for f in frames})          # crops = the union bbox, scale 1
    for f in frames:
        assert np.array_equal(base[f], same[f])
    for conf, crops in (({"flow_scale": 0.5}, None), ({}, follow_crops(regions, frames)),
                        ({"flow_scale": 0.5}, follow_crops(regions, frames))):
        other = run(conf, crops)
        for f in frames:
            r = regions[f]
            assert np.abs(other[f] - base[f])[r].mean() < 0.05     # a 24 x 42 px noisy toy crop: sanity only
            assert np.array_equal(other[f][~r], base[f][~r])               # outside the region: 0 as before
            assert (other[f][r & force[f]] == 1.0).all()


def _smooth(frames, imgs, alphas, regions, force, bbox, fin, conf, crops):
    a = {f: x.copy() for f, x in alphas.items()}
    smooth_zone_alphas(a, frames, bbox, regions, force, "flow_mean", 2, 2.0, conf, fin, lambda f: imgs[f], crops=crops)
    return a


def test_zone_params():
    assert ("temporal_flow_scale", "flow_scale", float) in TEMPORAL_CONF_PARAMS
    assert TEMPORAL_CROPS == ("union", "follow")
    import inspect
    assert inspect.signature(temporal_mean_flow).parameters["flow_scale"].default == 1.0


def test_follow_falls_back_to_the_union_when_the_zone_stays():
    frames, imgs, alphas, regions, force, fg = _scene(9, seed=3)     # the region does not move
    cr = follow_crops(regions, frames)
    assert set(cr.values()) == {(4, 46, 44, 68)}                       # = the union bbox -> bit-identical to "union"


@pytest.mark.parametrize("params,crop,conf", [
    ({"temporal": "flow_mean", "temporal_scale": 2}, True, {"flow_scale": 0.5}),           # the default (2026-10-28)
    ({"temporal": "flow_mean", "temporal_scale": 1}, True, {}),                            # scale 1: flow at plate res
    ({"temporal": "flow_mean", "temporal_scale": 2, "temporal_crop": "union", "temporal_flow_scale": 1.0}, False,
     {"flow_scale": 1.0}),                                                                 # saved in the zone: kept
    ({"temporal": "median", "temporal_scale": 2}, False, {}),                              # other modes: as before
])
def test_flow_mean_defaults_follow_crop_and_plate_res_flow(tmp_path, monkeypatch, params, crop, conf):
    """User decision 2026-10-28 (A/B RAPORT.md): a `flow_mean` zone without its own values smooths with B1 + B2."""
    import engine.vitmatte as vm
    import engine.zones as Z
    from engine.zonecache import ZoneCache
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(5)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    seen = {}
    monkeypatch.setattr(Z, "smooth_zone_alphas", lambda *a, crops=None, **kw: seen.update(conf=dict(a[8]), crops=crops))
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())],
            [int(xs.min()), int(ys.max())]]
    z = Z.Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params=params)
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    Z.apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr,
                  cache_for=lambda zid: ZoneCache(tmp_path, "z_01"))
    assert (seen["crops"] is not None) == crop
    assert seen["conf"] == conf
