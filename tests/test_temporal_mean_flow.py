# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""temporal = flow_mean for ViTMatte zones (user decision 2026-10-15): levels (alpha_black / alpha_white) BEFORE the
smoothing, then a motion-compensated, median-anchored weighted mean (refine.temporal_mean_flow) – garbage in one frame
is removed, not smeared onto the neighbours; a real change gives no ghost; per-frame noise drops more than with the
median."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine.refine import temporal_mean_flow, temporal_median_flow  # noqa: E402
from engine.zones import VITMATTE_TEMPORAL, levels_alpha  # noqa: E402


def _texture(n, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.2, 0.8, (h, w)).astype(np.float32)
    return np.stack([g] * n)


def test_single_frame_garbage_removed_not_smeared():
    n = 9
    grays = _texture(n)
    a = np.zeros((n, 48, 64), np.float32)
    a[:, 10:30, 10:30] = 0.5                       # a steady soft area (strands)
    a[4, 35:40, 40:45] = 0.4                       # garbage in frame 4 only
    out = temporal_mean_flow(a, grays, radius=3)
    assert out[4, 35:40, 40:45].max() < 0.02                                 # removed in its frame
    assert max(out[f, 35:40, 40:45].max() for f in range(n) if f != 4) < 0.02   # ... and not smeared onto others
    assert np.allclose(out[:, 10:30, 10:30], 0.5, atol=1e-4)                 # steady soft value kept


def test_noise_drops_more_than_median_and_no_ghost():
    n = 11
    grays = _texture(n, seed=1)
    rng = np.random.default_rng(2)
    a = np.full((n, 48, 64), 0.5, np.float32) + rng.normal(0, 0.04, (n, 48, 64)).astype(np.float32)
    a[:, :, 50:] = 0.0
    a[6:, :, 50:] = 1.0                            # a real change from frame 6 on (right strip)
    a = np.clip(a, 0, 1)
    mean = temporal_mean_flow(a, grays, radius=3)
    med = temporal_median_flow(a, grays, radius=3)
    flick = lambda x: float(np.abs(np.diff(x[:, :, :45], axis=0)).mean())
    assert flick(mean) < 0.8 * flick(med) < flick(a), (flick(mean), flick(med), flick(a))
    assert mean[5, :, 52:].max() < 0.05 and mean[6, :, 52:].min() > 0.95      # no ghost of the change


def test_levels_and_mode_registered():
    a = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 1.0], np.float32)
    assert np.allclose(levels_alpha(a, 0.1, 0.9), [0, 0, 0, 0.5, 1, 1])
    assert levels_alpha(a, None, None) is a
    assert "flow_mean" in VITMATTE_TEMPORAL


def test_zone_flow_mean_shares_the_cache_and_clips_before(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    from engine.zonecache import ZoneCache
    from engine.zones import Zone, apply_zones
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(5)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")

    def run(params):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params=params)
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache)
    r0 = run({"temporal": "none", "alpha_black": 0.3})
    r1 = run({"temporal": "flow_mean", "alpha_black": 0.3})
    assert r1.stats["z_01"]["computed"] == 0 and r1.stats["z_01"]["temporal"] == "flow_mean"   # same raw cache
    for f in frames:            # static scene: smoothing of identical (already clipped) frames changes nothing
        a0, area = r0.soft[f]
        assert np.allclose(r1.soft[f][0], a0, atol=1e-4)
        assert not ((r1.soft[f][0] > 0) & (r1.soft[f][0] < 0.001) & area).any()   # nothing below the cut smeared back


def test_flow_cache_keeps_only_the_window():
    """RAM (user report 2026-10-22: 4K x2 flow_mean filled 60 GB): pairs below the window of the current frame are dropped."""
    from engine.refine import _drop_flows
    cache = {(i, j): i for i in range(10) for j in range(10) if i != j and abs(i - j) <= 2}
    _drop_flows(cache, 5)
    assert cache and all(min(k) >= 5 for k in cache)
