# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""zones.smooth_zone_alphas (plan docs/PLAN_EDGE_SPEED.md A1, 2026-10-25) = the former in-line stack code of
_vitmatte_sequence, bit for bit: median / flow / flow_mean, scale 1 and 2, parts forced to 1, levels before flow_mean,
region / core forcing after. The reference below is that in-line code frozen (it ran on four full stacks), fed with the
frozen smoothers of tests/_ref_temporal_old.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import _ref_temporal_old as old  # noqa: E402
from engine.zones import levels_alpha, smooth_zone_alphas  # noqa: E402


def _old_smooth(alphas, fl, bbox, regions, force, temporal, t_radius, t_scale, t_conf, fin, images, fg_sure=None):
    """Frozen copy of the stack code removed from _vitmatte_sequence on 2026-10-25."""
    y0, y1, x0, x1 = bbox
    if temporal != "none" and len(fl) >= 3:
        stack = np.stack([alphas[f][y0:y1, x0:x1] for f in fl])
        if fg_sure:
            for i, f in enumerate(fl):
                if f in fg_sure:
                    stack[i][fg_sure[f][y0:y1, x0:x1]] = 1.0
        bh, bw = stack.shape[1:]
        if temporal == "flow_mean" and ("alpha_black" in fin or "alpha_white" in fin):
            stack = levels_alpha(stack, fin.get("alpha_black"), fin.get("alpha_white"))
        stack0 = stack
        if t_scale > 1.0:
            stack = np.stack([cv2.resize(a, (int(round(bw * t_scale)), int(round(bh * t_scale))), interpolation=cv2.INTER_LINEAR)
                              for a in stack])
            up0 = stack.copy()
        if temporal == "median":
            stack = old.temporal_median(stack, t_radius, 0.35)
        else:
            grays = np.stack([cv2.cvtColor(np.clip(images(f)[y0:y1, x0:x1], 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
                              for f in fl])
            if t_scale > 1.0:
                grays = np.stack([cv2.resize(g, (stack.shape[2], stack.shape[1]), interpolation=cv2.INTER_LINEAR) for g in grays])
            stack = (old.temporal_mean_flow(stack, grays, t_radius, **t_conf) if temporal == "flow_mean"
                     else old.temporal_median_flow(stack, grays, t_radius))
        if t_scale > 1.0:
            delta = np.stack([cv2.resize(a - u, (bw, bh), interpolation=cv2.INTER_AREA) for a, u in zip(stack, up0)])
            stack = np.clip(stack0 + delta, 0.0, 1.0).astype(np.float32)
        for i, f in enumerate(fl):
            a = alphas[f].copy()
            a[y0:y1, x0:x1] = stack[i]
            a[~regions[f]] = 0.0
            a[regions[f] & force[f]] = 1.0
            alphas[f] = a


def _scene(n=9, H=48, W=72, seed=0):
    rng = np.random.default_rng(seed)
    bg = rng.uniform(0.2, 0.8, (H + 8, W + 8, 3)).astype(np.float32)
    imgs, alphas, regions, force, fg = {}, {}, {}, {}, {}
    frames = list(range(3, 3 + n))
    for t, f in enumerate(frames):
        dx, dy = (t * 3) % 7, (t * 2) % 5
        imgs[f] = bg[dy:dy + H, dx:dx + W].copy()
        a = np.zeros((H, W), np.float32)
        a[8:40, 10:50] = 1.0                                   # core
        a[8 + t:30 + t, 50:62] = 0.55 + rng.normal(0, 0.05, (22, 12)).astype(np.float32)   # strands
        a[42:46, 60:66] = 0.4 if t == 4 else 0.0               # garbage in one frame
        alphas[f] = np.clip(a, 0, 1)
        r = np.zeros((H, W), bool)
        r[4:46, 44:68] = True
        regions[f] = r
        c = np.zeros((H, W), bool)
        c[8:40, 10:48] = True
        force[f] = c & r
        p = np.zeros((H, W), bool)
        p[30:38, 46:52] = True
        fg[f] = p
    return frames, imgs, alphas, regions, force, fg


@pytest.mark.parametrize("temporal", ["median", "flow", "flow_mean"])
@pytest.mark.parametrize("t_scale", [1.0, 2.0])
@pytest.mark.parametrize("parts", [False, True])
def test_smooth_zone_alphas_identical(temporal, t_scale, parts):
    frames, imgs, alphas, regions, force, fg = _scene()
    bbox = (4, 46, 44, 68)
    fin = {"alpha_black": 0.1, "alpha_white": 0.9}
    fg_sure = fg if parts else None
    a_old = {f: a.copy() for f, a in alphas.items()}
    a_new = {f: a.copy() for f, a in alphas.items()}
    _old_smooth(a_old, frames, bbox, regions, force, temporal, 2, t_scale, {}, fin, lambda f: imgs[f], fg_sure)
    reads = []
    smooth_zone_alphas(a_new, frames, bbox, regions, force, temporal, 2, t_scale, {}, fin,
                       lambda f: (reads.append(f), imgs[f])[1], fg_sure=fg_sure)
    for f in frames:
        assert np.array_equal(a_old[f], a_new[f]), (temporal, t_scale, parts, f)
        assert a_old[f].dtype == a_new[f].dtype == np.float32
    if temporal != "median":
        assert sorted(reads) == frames                   # every plate frame decoded once


def test_conf_params_and_progress_pass_through():
    frames, imgs, alphas, regions, force, _ = _scene(7, seed=2)
    bbox = (4, 46, 44, 68)
    conf = {"fb_sigma": 1.0, "conf_mode": "centre"}
    a_old = {f: a.copy() for f, a in alphas.items()}
    a_new = {f: a.copy() for f, a in alphas.items()}
    _old_smooth(a_old, frames, bbox, regions, force, "flow_mean", 3, 1.0, conf, {}, lambda f: imgs[f])
    seen = []
    smooth_zone_alphas(a_new, frames, bbox, regions, force, "flow_mean", 3, 1.0, conf, {}, lambda f: imgs[f],
                       progress=lambda i, n: seen.append((i, n)))
    assert seen == [(i, 7) for i in range(1, 8)]
    for f in frames:
        assert np.array_equal(a_old[f], a_new[f])


def test_none_and_short_sequences_untouched():
    frames, imgs, alphas, regions, force, _ = _scene(2)
    before = {f: a.copy() for f, a in alphas.items()}
    smooth_zone_alphas(alphas, frames, (4, 46, 44, 68), regions, force, "flow", 2, 2.0, {}, {}, lambda f: imgs[f])
    frames9, imgs9, alphas9, regions9, force9, _ = _scene(9)
    before9 = {f: a.copy() for f, a in alphas9.items()}
    smooth_zone_alphas(alphas9, frames9, (4, 46, 44, 68), regions9, force9, "none", 2, 2.0, {}, {}, lambda f: imgs9[f])
    assert all(np.array_equal(before[f], alphas[f]) for f in frames)
    assert all(np.array_equal(before9[f], alphas9[f]) for f in frames9)
