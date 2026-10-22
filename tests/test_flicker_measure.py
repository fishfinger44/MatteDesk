# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_measure.py (docs/PLAN_DEFLICKER_AUTO.md § 2): on a synthetic sequence with KNOWN ingredients the measurement must
tell them apart – global level flicker, a slow drift, a fixed lamp footprint with random amplitude inside one object, grain, colour,
regions flickering together vs independently, camera motion."""
import json

import cv2
import numpy as np
import pytest

from engine.flicker_measure import FlickerMeasure, measure_flicker


def synth(n=24, w=192, h=112, level=0.0, drift=0.0, blob=0.0, grain=0.002, colour=0.0, split=False, shift_px=0.0, seed=0,
          memory=False):
    """Scene-linear BGR frames + the injected per-frame log2 series. Left third = dark textured 'tree', the rest = bright 'wall'.
    level: σ of a random per-frame gain (log2) – applied to the whole frame, or (split) independently to tree / wall;
    drift: total log2 change over the window (linear); blob: σ of a random amplitude on a fixed Gaussian footprint on the tree;
    colour: σ of a random per-frame B/G factor; shift_px: whole-frame translation per frame (camera)."""
    rng = np.random.default_rng(seed)
    pad = int(abs(shift_px) * n) + 4
    W, Hh = w + 2 * pad, h + 2 * pad
    yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
    tex = cv2.resize(rng.random((7, 12), dtype=np.float32), (W, Hh), interpolation=cv2.INTER_CUBIC)
    stripes = 0.5 + 0.5 * np.sin(xx / 5.0 + yy / 11.0)
    dots = np.ones((Hh, W), np.float32)                      # corners for the engine's ORB homography (real plates have them)
    for _ in range(350):
        y, x = rng.integers(0, Hh - 2), rng.integers(0, W - 2)
        dots[y:y + 2, x:x + 2] = 0.55
    tex = tex * dots
    tree = xx < (pad + w / 3)
    base = np.where(tree, 0.08 + 0.10 * tex + 0.06 * stripes, 0.75 + 0.1 * tex)[..., None] * np.array([0.95, 1.0, 1.05], np.float32)
    cx, cy = pad + w / 6, pad + h / 2
    foot = np.exp(-(((xx - cx) / (w / 10)) ** 2 + ((yy - cy) / (h / 4)) ** 2)) * tree
    lv_all = rng.normal(0, level, n) if not split else None
    if memory and lv_all is not None:   # slow variation: moving average over 5 frames (a lamp wandering, not pulsing)
        lv_all = np.convolve(lv_all, np.ones(5) / 5, mode="same") * np.sqrt(5)
    lv_tree = rng.normal(0, level, n) if split else None
    lv_wall = rng.normal(0, level, n) if split else None
    amp = rng.normal(0, blob, n)
    col = rng.normal(0, colour, n)
    frames = []
    for t in range(n):
        d = drift * t / (n - 1)
        lg = np.zeros((Hh, W), np.float32) + d
        lg += lv_all[t] if not split else np.where(tree, lv_tree[t], lv_wall[t])
        lg += amp[t] * foot
        img = base * (2.0 ** lg)[..., None]
        img = img * (2.0 ** np.stack([np.full((Hh, W), col[t], np.float32), np.zeros((Hh, W), np.float32), np.zeros((Hh, W), np.float32)], axis=2))
        img = img * (2.0 ** rng.normal(0, grain, (Hh, W, 3)).astype(np.float32))
        sx = int(round(shift_px * t))
        frames.append(np.ascontiguousarray(img[pad:pad + h, pad + sx:pad + sx + w]).astype(np.float32))
    tree_mask = np.zeros((h, w), bool)
    tree_mask[:, : w // 3] = True
    series = {"level": lv_all, "tree": lv_tree, "wall": lv_wall, "amp": amp, "drift": drift}
    return frames, tree_mask, series


def _wall_mask(tree_mask):
    return ~tree_mask


def test_level_and_drift_are_measured_and_told_apart():
    frames, tm, s = synth(level=0.03, drift=0.04)
    m = measure_flicker(frames, {"tree": tm, "wall": _wall_mask(tm)}, eval_w=192)
    a = m.regions["all"]
    n = len(frames)
    injected = s["level"] * 100 + np.linspace(0, 4, n)                 # random level + the 4 % drift
    expected_step = float(np.abs(np.diff(injected)).mean())
    assert a.level_pct == pytest.approx(expected_step, rel=0.25)
    # the trend of what was injected (the random series itself carries a trend of its own: seed 0 -> -4.1 %)
    expected_drift = float(np.polyfit(np.arange(n), injected, 1)[0] * (n - 1))
    assert a.drift_pct == pytest.approx(expected_drift, abs=0.5)
    assert 1.0 < a.drift_sigma_pct < 4.0                                # +-3 % noise on 24 frames -> ~2 % uncertainty
    # without the random part the drift is unambiguous
    q = measure_flicker(synth(level=0.0, drift=0.04)[0], eval_w=192).regions["all"]
    assert q.drift_pct == pytest.approx(4.0, abs=0.3) and q.drift_sigma_pct < 0.3
    # memory: lag-1 autocorrelation of the detrended per-frame level – equals that of the injected series (seed 0: -0.62)
    dev = injected - np.polyval(np.polyfit(np.arange(n), injected, 1), np.arange(n))
    assert a.level_autocorr == pytest.approx(float(np.corrcoef(dev[:-1], dev[1:])[0, 1]), abs=0.15)
    fr_slow, _, s_slow = synth(level=0.03, memory=True)
    slow = measure_flicker(fr_slow, eval_w=192).regions["all"]
    inj_s = s_slow["level"] * 100
    dev_s = inj_s - np.polyval(np.polyfit(np.arange(n), inj_s, 1), np.arange(n))
    assert slow.level_autocorr == pytest.approx(float(np.corrcoef(dev_s[:-1], dev_s[1:])[0, 1]), abs=0.15)
    assert slow.level_autocorr > a.level_autocorr + 0.5         # slow variation has memory, pulsing has not
    assert a.colour_pct["bg"] < 0.3 and a.colour_pct["rg"] < 0.3
    assert a.metric_pct > a.floor_pct > 0
    # a pure global gain leaves no fixed local pattern: low texture share, small local residual above its floor
    assert (a.texture_fixed_share or 0) < 0.45
    assert a.local_pct < a.local_floor_pct + 0.6
    assert m.region_corr is not None and m.region_corr > 0.7   # every tile follows the same lamp
    assert m.global_shift_px < 0.5
    json.dumps(m.to_dict())


def test_fixed_lamp_footprint_on_one_object():
    frames, tm, s = synth(level=0.01, blob=0.06)
    m = measure_flicker(frames, {"tree": tm, "wall": _wall_mask(tm)}, eval_w=192)
    tree, wall = m.regions["tree"], m.regions["wall"]
    assert tree.texture_fixed_share > 0.6 and wall.texture_fixed_share is None   # wall: band at its floor -> nothing fixed
    assert tree.texture_bands_pct["8-40"] > 3 * tree.texture_floor_bands_pct["8-40"]
    assert tree.local_pct > wall.local_pct + 1.0
    assert tree.texture_bands_pct["8-40"] > 2 * wall.texture_bands_pct["8-40"]
    assert tree.metric_pct > wall.metric_pct


def test_regions_independent_vs_together():
    fr_split, tm, _ = synth(level=0.03, split=True, seed=1)
    fr_same, _, _ = synth(level=0.03, split=False, seed=1)
    corr_split = measure_flicker(fr_split, eval_w=192).region_corr
    corr_same = measure_flicker(fr_same, eval_w=192).region_corr
    assert corr_same > 0.7 and corr_split < corr_same - 0.3


def test_camera_motion_is_seen_and_removed_from_flicker():
    still, tm, _ = synth(level=0.02, seed=2)
    moving, _, _ = synth(level=0.02, shift_px=1.0, seed=2)
    ms = measure_flicker(still, {"tree": tm}, eval_w=192)
    mm = measure_flicker(moving, {"tree": tm}, eval_w=192)
    assert mm.global_shift_px > 0.6 and ms.global_shift_px < 0.3
    # the level flicker is the same; alignment keeps the metric close (motion is not read as flicker)
    assert mm.regions["all"].level_pct == pytest.approx(ms.regions["all"].level_pct, rel=0.35)
    assert mm.regions["all"].metric_pct < ms.regions["all"].metric_pct * 1.6
    assert mm.regions["all"].motion_resid_px < 0.5            # the camera (median flow) is taken out
    assert mm.regions["all"].texture_share > 0.2


def test_masks_static_or_per_frame_and_small_inputs():
    frames, tm, _ = synth(n=6, level=0.02)
    per_frame = [tm] * len(frames)
    m1 = measure_flicker(frames, {"tree": tm}, eval_w=192)
    m2 = measure_flicker(frames, {"tree": per_frame}, eval_w=192)
    assert m1.regions["tree"].px == m2.regions["tree"].px and set(m1.regions) == {"all", "inner", "tree"}
    assert m1.regions["inner"].px < m1.regions["all"].px
    # a plain region: no texture -> its mask moves with the camera, no own-motion number
    plain = np.zeros_like(tm)
    plain[10:40, 150:185] = True                                # inside the smooth bright 'wall'
    mp = measure_flicker([f * np.where(np.arange(f.shape[1]) >= 140, 1.0, 1.0)[None, :, None] for f in frames], {"plain": plain}, eval_w=192)
    assert mp.regions["plain"].texture_share < 0.5 and mp.regions["plain"].motion_resid_px is None
    assert isinstance(m1, FlickerMeasure) and len(m1.regions["all"].level_series_pct) == 5
    with pytest.raises(ValueError):
        measure_flicker(frames[:2], eval_w=192)
    # full-resolution mask with a downscaled evaluation
    big = cv2.resize(tm.astype(np.uint8), (384, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    big_frames = [cv2.resize(f, (384, 224), interpolation=cv2.INTER_CUBIC) for f in frames]
    m3 = measure_flicker(big_frames, {"tree": big}, eval_w=192)
    assert m3.eval_w == 192 and m3.regions["tree"].px > 0


def test_level_period_finds_a_lamp_beat_and_not_noise():
    from engine.flicker_measure import level_period
    rng = np.random.default_rng(5)
    n = 40
    beat = 3.0 * np.sin(2 * np.pi * np.arange(n) / 4.0) + rng.normal(0, 0.4, n)     # lamp beat every 4 frames (%)
    per, share, p = level_period(np.diff(beat))
    assert per == pytest.approx(4.0, abs=0.3) and share > 0.5 and p < 0.01
    noise = rng.normal(0, 3.0, n)                                                     # random pulsing (SH010-like)
    per2, share2, p2 = level_period(np.diff(noise))
    assert p2 > 0.01 or share2 < 0.3
    assert level_period([0.1] * 5) == (None, None, None)


def test_measure_reports_the_rhythm_of_a_pulsing_plate():
    frames, tm, _ = synth(n=24, level=0.0, seed=7)
    pulse = 2.0 ** (0.04 * np.sin(2 * np.pi * np.arange(24) / 3.0))
    frames = [(f * pulse[i]).astype(np.float32) for i, f in enumerate(frames)]
    r = measure_flicker(frames, eval_w=192).regions["all"]
    assert r.period_frames == pytest.approx(3.0, abs=0.3) and r.period_p < 0.01

