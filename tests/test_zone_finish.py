# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Zone alpha finish (user 2026-10-14): black / white points + blur on the zone alpha, applied after the engine and the
temporal smoothing, inside the region only; the core stays 1 and joins without a step; the engine cache is not
invalidated by the finish parameters."""

import numpy as np
import cv2

from engine.zonecache import ZoneCache
from engine.zones import Zone, apply_zones, finish_params, finish_zone_alpha, zone_mask_at


def _alpha():
    h, w = 120, 160
    region = np.zeros((h, w), bool); region[20:100, 20:140] = True
    core = np.zeros((h, w), bool); core[40:100, 60:140] = True
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.clip(1.0 - (60 - xx) / 40.0, 0, 1).astype(np.float32)      # ramp 0 -> 1 towards the core (x = 20..60)
    a[~region] = 0.0
    a[core] = 1.0
    a[25:30, 25:30] = 0.08                                             # haze / garbage speck
    return a, region, core


def test_levels_remove_haze_keep_core_and_outside():
    a, region, core = _alpha()
    out = finish_zone_alpha(a, region, core, {"alpha_black": 0.15, "alpha_white": 0.8})
    assert out[25:30, 25:30].max() == 0.0                              # garbage below the black point gone
    assert np.array_equal(out[~region], a[~region])                    # outside the region untouched
    assert out[region & core].min() == 1.0                             # core stays 1
    # linear remap: continuous (no step); the clipped white reaches 1 BEFORE the core edge and joins it
    row = out[70, 20:70]
    assert np.all(np.diff(row) >= -1e-6) and row[-1] == 1.0 and (row >= 0.999).sum() > 5
    assert np.abs(np.diff(row)).max() < 0.08                           # no hard cut along the ramp
    # defaults = identity (same object returned)
    assert finish_zone_alpha(a, region, core, {}) is a and finish_zone_alpha(a, region, core, None) is a


def test_blur_softens_inside_region_core_rejoined():
    a, region, core = _alpha()
    a2 = a.copy(); a2[50:52, 30:50] = 1.0                              # a hard 2 px line inside the zone
    out = finish_zone_alpha(a2, region, core, {"alpha_blur_px": 2.0})
    assert out[51, 40] < 0.95 and out[51, 40] > 0.3                    # blurred
    assert out[region & core].min() == 1.0                             # the blur never eats into the core
    assert np.array_equal(out[~region], a2[~region])
    # black cut + blur: the cut edge is softened, the core edge stays continuous
    out2 = finish_zone_alpha(a, region, core, {"alpha_black": 0.5, "alpha_blur_px": 3.0})
    row = out2[70, 20:70]
    assert row[-1] == 1.0 and np.abs(np.diff(row)).max() < 0.25


def test_finish_params_split_and_cache_shared(tmp_path, monkeypatch):
    eng, fin = finish_params({"temporal": "flow", "alpha_black": 0.1, "alpha_blur_px": 1.5, "model_px": 504})
    assert eng == {"temporal": "flow", "model_px": 504} and fin == {"alpha_black": 0.1, "alpha_blur_px": 1.5}
    assert finish_params(None) == ({}, {})
    # the finish does not touch the per-frame engine cache (vitmatte mock): second run with sliders = 0 computed
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import engine.vitmatte as vm
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(4)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")
    z0 = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params={"temporal": "none"})
    r0 = apply_zones(frames, cores, lambda f: img, [z0], tr, cache_for=lambda zid: cache)
    z1 = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte",
              params={"temporal": "none", "alpha_black": 0.3, "alpha_white": 0.9})
    r1 = apply_zones(frames, cores, lambda f: img, [z1], tr, cache_for=lambda zid: cache)
    assert r1.stats["z_01"]["computed"] == 0 and r1.stats["z_01"]["cached"] == 4 and len(runner.calls) == 4
    assert r1.stats["z_01"]["finish"] == {"alpha_black": 0.3, "alpha_white": 0.9}
    area = zone_mask_at(z0, core.shape)
    for f in frames:
        a0, _ = r0.soft[f]; a1, _ = r1.soft[f]
        assert a1[~area].max() == 0.0 and a1[area & core].min() == 1.0
        assert (a1 > 0).sum() <= (a0 > 0).sum()                        # the black point only removes
    # per-frame engines (chroma) share the same rule: finish is not part of the hash either
    z2 = Zone("z_02", "add", "matte", 1, polygon=poly, matting="chroma", params={})
    z3 = Zone("z_02", "add", "matte", 1, polygon=poly, matting="chroma", params={"alpha_black": 0.2})
    cache2 = ZoneCache(tmp_path, "z_02")
    r2 = apply_zones(frames, cores, lambda f: img, [z2], tr, cache_for=lambda zid: cache2)
    r3 = apply_zones(frames, cores, lambda f: img, [z3], tr, cache_for=lambda zid: cache2)
    assert r2.stats["z_02"]["computed"] == 4 and r3.stats["z_02"]["computed"] == 0 and r3.stats["z_02"]["finish"] == {"alpha_black": 0.2}
