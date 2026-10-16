# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two experiments against the strand flicker of the ViTMatte zone (2026-10-15), both off by default:
(A) flow confidence in refine.temporal_mean_flow (continuous fwd/bwd, photometric, texture weight, 2 flow scales);
(B) hybrid zone alpha: MatAnyone (key from ViTMatte) = the stable base, ViTMatte only adds what is above it (strands)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine.refine import temporal_mean_flow, temporal_median_flow  # noqa: E402
from engine.zones import hybrid_blend, ridge_map  # noqa: E402


def _texture(n, h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.2, 0.8, (h, w)).astype(np.float32)
    return np.stack([g] * n)


def _noisy(n, seed=2, h=48, w=64):
    rng = np.random.default_rng(seed)
    return np.clip(np.full((n, h, w), 0.5, np.float32) + rng.normal(0, 0.04, (n, h, w)).astype(np.float32), 0, 1)


flick = lambda x: float(np.abs(np.diff(x, axis=0)).mean())


def test_confidence_off_is_the_old_path():
    g, a = _texture(7), _noisy(7)
    ref = temporal_mean_flow(a, g, 3)
    assert np.array_equal(temporal_mean_flow(a, g, 3, fb_sigma=0.0, photo_sigma=0.0, tex_k=0.0, multiscale=1), ref)
    with pytest.raises(ValueError):
        temporal_mean_flow(a, g, 3, fb_sigma=1.0, conf_mode="bogus")


@pytest.mark.parametrize("kw", [{"fb_sigma": 1.0}, {"photo_sigma": 0.1}, {"multiscale": 2}, {"multiscale": 3}, {"fb_sigma": 1.0, "conf_mode": "centre"}])
def test_confident_flow_still_smooths_and_removes_garbage(kw):
    n = 11
    g, a = _texture(n, seed=1), _noisy(n)
    a[5, 35:40, 40:45] = 0.95                      # garbage in frame 5 only
    out = temporal_mean_flow(a, g, 3, **kw)
    med = temporal_median_flow(a, g, 3)
    assert flick(out) < 0.8 * flick(med), (kw, flick(out), flick(med))     # static, textured: every neighbour trusted
    assert abs(float(out[5, 35:40, 40:45].mean()) - 0.5) < 0.05             # outlier still rejected by the median anchor


def test_textureless_background_does_not_vote():
    """On a flat gray the flow is a guess: with the texture weight the neighbours drop out and the frame keeps its own
    value (drop mode: only the centre is left); without it the area is smoothed as usual."""
    n = 7
    g = np.full((n, 48, 64), 0.5, np.float32)
    a = _noisy(n)
    out = temporal_mean_flow(a, g, 3, tex_k=1e-3)
    assert np.allclose(out, a, atol=1e-5)
    assert flick(temporal_mean_flow(a, g, 3)) < 0.8 * flick(a)


def test_photometric_mismatch_lowers_the_neighbour():
    """A neighbour whose aligned gray does not match (a flash / wrong flow) and carries an offset alpha still inside the
    median tolerance: the photometric weight keeps it out of the mean."""
    n = 9
    g = _texture(n, seed=3)
    a = np.full((n, 48, 64), 0.5, np.float32)
    g[5, 10:30, 10:30] = np.clip(g[5, 10:30, 10:30] + 0.35, 0, 1)
    a[5, 10:30, 10:30] = 0.68
    plain = temporal_mean_flow(a, g, 3)
    conf = temporal_mean_flow(a, g, 3, photo_sigma=0.05)
    err = lambda x: float(np.abs(x[4, 14:26, 14:26] - 0.5).mean())
    assert err(conf) < 0.5 * err(plain) and err(conf) < 0.005, (err(conf), err(plain))


def test_ridge_map_finds_thin_lines_not_blobs():
    a = np.zeros((64, 64), np.float32)
    a[10, 5:60] = 0.6                               # a 1 px strand
    a[30:60, 20:50] = 1.0                           # a solid blob
    r = ridge_map(a, 1.0)
    assert r[10, 10:55].min() > 0.5
    assert r[40:50, 30:40].max() < 0.05


def test_hybrid_blend_keeps_base_and_adds_strands_only():
    rng = np.random.default_rng(4)
    ma = np.full((40, 40), 0.3, np.float32)
    vm = ma + rng.uniform(-0.04, 0.04, ma.shape).astype(np.float32)      # ViTMatte noise around the base
    vm[20, :] = 0.9                                                       # a strand MatAnyone lost
    vm[30:35, 5:10] = 0.1                                                 # ViTMatte below the base: base wins
    out, w = hybrid_blend(vm, ma, lo=0.05, hi=0.25, blur_px=0.0)
    assert np.array_equal(out[5:15], ma[5:15])                            # noise below lo: exactly the base
    assert np.allclose(out[20, :], 0.9)                                   # the strand comes from ViTMatte
    assert np.array_equal(out[30:35, 5:10], ma[30:35, 5:10])
    lo_b, hi_b = np.minimum(vm, ma), np.maximum(vm, ma)
    out_b, _ = hybrid_blend(vm, ma, ridge_px=1.0, blur_px=1.5)            # ridge + widened weight
    assert (out_b >= lo_b - 1e-6).all() and (out_b <= hi_b + 1e-6).all()   # always between the two


def test_zone_hybrid_needs_matanyone_shares_the_vitmatte_cache(tmp_path, monkeypatch):
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
    ma_calls = []

    def ma_run(seg, sf, core_, keys_, start, scale, tag):
        ma_calls.append(int(keys_[start].sum()))
        return {f: np.full((seg.side, seg.side), 0.2, np.float32) for f in sf}

    def run(params, **kw):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params=params)
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache, **kw)
    r0 = run({"temporal": "flow_mean"})
    with pytest.raises(ValueError):
        run({"temporal": "flow_mean", "hybrid": "matanyone"})                 # no ComfyUI -> clear error
    with pytest.raises(ValueError):
        run({"temporal": "flow_mean", "hybrid": "bogus"}, ma_run=ma_run)
    r1 = run({"temporal": "flow_mean", "hybrid": "matanyone"}, ma_run=ma_run)
    st = r1.stats["z_01"]
    assert st["computed"] == 0 and st["hybrid"] == "matanyone" and st["hybrid_frames"] == len(frames)   # raw VM cache shared
    assert len(ma_calls) == 1 and ma_calls[0] > int(core.sum())             # one MatAnyone pass, key = core ∪ ViTMatte hint
    r2 = run({"temporal": "flow_mean", "hybrid": "matanyone", "temporal_fb_sigma": 1.0}, ma_run=ma_run)
    assert len(ma_calls) == 1 and r2.stats["z_01"]["temporal_conf"] == {"fb_sigma": 1.0}   # MatAnyone from its cache
    for f in frames:
        a0, area = r0.soft[f]
        a1, _ = r1.soft[f]
        m = area & ~core
        assert a1[m].min() >= 0.2 - 1e-3                                     # the MatAnyone base everywhere
        strands = m & (a0 > 0.2 + 0.25 + 0.02)
        assert strands.any() and np.abs(a1[strands] - a0[strands]).max() < 0.02   # strong strands = ViTMatte
        assert a1[area & core].min() == 1.0 and a1[~area].max() == 0.0


def test_black_after_smoothing_cuts_the_haze_the_mean_brings_back(tmp_path, monkeypatch):
    """alpha_black_after (user 2026-10-15, CLIP2_A F18): black-only levels again AFTER the temporal mean, outside the
    forced core; finish key -> the raw ViTMatte cache is shared; absent = as before."""
    import engine.vitmatte as vm
    from engine.zonecache import ZoneCache
    from engine.zones import Zone, apply_zones, levels_alpha
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(5)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")

    def run(params):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params=params)
        return apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, cache_for=lambda zid: cache)
    r0 = run({"temporal": "flow_mean"})
    r1 = run({"temporal": "flow_mean", "alpha_black_after": 0.3})
    assert r1.stats["z_01"]["computed"] == 0 and r1.stats["z_01"]["finish"] == {"alpha_black_after": 0.3}
    for f in frames:
        a0, area = r0.soft[f]
        a1, _ = r1.soft[f]
        m = area & ~core
        assert ((a0 > 0.01) & (a0 < 0.3) & m).any()                        # there was haze to cut
        assert np.allclose(a1[m], levels_alpha(a0, 0.3, None)[m], atol=1e-5)
        assert a1[area & core].min() == 1.0 and a1[~area].max() == 0.0
