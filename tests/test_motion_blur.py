# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion blur of the alpha (engine/motionblur.py): a vector blur of the alpha along the object's own motion (plate flow
x shutter / 360 x strength), only where it moves. Used only by Final Touch (the stage-version „Motion blur alfy” was
removed 2026-10-24). Engine on a synthetic moving square."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

H, W = 240, 320


def _square(x: int):
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    tex = cv2.GaussianBlur((np.random.default_rng(1).random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    g, m = bg.copy(), np.zeros((H, W), bool)
    m[80:160, x:x + 80] = True
    g[m] = np.roll(tex, x, axis=1)[m]
    return g, m


def _soft(v):
    return int(((v > 0.02) & (v < 0.98)).sum())


@pytest.mark.parametrize("speed,shutter,strength,ramp", [(12, 180, 1.0, 6), (12, 360, 1.0, 12), (12, 180, 2.0, 12),
                                                         (8, 180, 1.0, 4)])
def test_blur_along_the_motion_only(speed, shutter, strength, ramp):
    from engine.motionblur import motion_blur_alpha
    g0, _ = _square(100 - speed)
    g1, m1 = _square(100)
    g2, _ = _square(100 + speed)
    out, st = motion_blur_alpha(m1.astype(np.float32), g1, g0, g2, shutter, strength)
    row, col = out[120, 60:220], out[50:190, 140]
    # a box blur of length L = speed x shutter / 360 x strength: a linear ramp ~L px on the leading and trailing edge
    assert abs(_soft(row) - 2 * ramp) <= 3, (_soft(row), np.round(row, 2))
    assert _soft(col) == 0                                    # across the motion: sharp
    assert abs(st["max_px"] - ramp) <= 1.0
    assert out[120, 140] == pytest.approx(1.0) and out[20, 20] == 0.0   # inside / far outside unchanged
    assert abs(float(out.sum()) - float(m1.sum())) < 0.02 * m1.sum()    # a blur keeps the coverage


def test_static_object_strength_zero_and_shot_ends_unchanged():
    from engine.motionblur import motion_blur_alpha
    g, m = _square(100)
    a = m.astype(np.float32)
    out, st = motion_blur_alpha(a, g, g, g, 180, 1.0)       # no motion: nothing changes
    assert np.array_equal(out, a) and st["moving_px"] == 0
    g0, _ = _square(88)
    g2, _ = _square(112)
    assert np.array_equal(motion_blur_alpha(a, g, g0, g2, 180, 0.0)[0], a)       # strength 0
    assert np.array_equal(motion_blur_alpha(a, g, g0, g2, 0.0, 1.0)[0], a)       # shutter 0
    one_side, _ = motion_blur_alpha(a, g, None, g2, 180, 1.0)                    # last frame of the shot: one flow
    assert _soft(one_side[120, 60:220]) > 6


def test_background_motion_is_not_used():
    """The camera pans (background moves), the object does not move on screen: no blur (only the object's own flow)."""
    from engine.motionblur import motion_blur_alpha
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur((rng.random((H, W + 40)) * 255).astype(np.uint8), (0, 0), 1.5)
    tex = cv2.GaussianBlur((np.random.default_rng(4).random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    m = np.zeros((H, W), bool)
    m[80:160, 100:180] = True

    def fr(dx):
        g = big[:, 20 + dx:20 + dx + W].copy()
        g[m] = tex[m]
        return g
    out, st = motion_blur_alpha(m.astype(np.float32), fr(0), fr(-10), fr(10), 180, 1.0)
    assert _soft(out[120, 60:220]) <= 2 and st["max_px"] < 1.5


# ------------------------------------------------------------------ GPU (the alignment worker in ComfyUI's Python)
def _gpu_ready():
    from engine import align_gpu
    ok, info = align_gpu.check_worker()
    return ok and info.startswith("cuda")


@pytest.mark.skipif(not _gpu_ready(), reason="ComfyUI's Python with CUDA not available")
@pytest.mark.parametrize("speed,shutter,strength", [(12, 180, 1.0), (30, 360, 1.5), (0, 180, 1.0)])
def test_gpu_equals_cpu(speed, shutter, strength):
    """The vector blur in the GPU worker = the CPU code within the bilinear rounding of cv2.remap (1/32 px)."""
    from engine import align_gpu
    from engine.motionblur import motion_blur_alpha
    g0, _ = _square(100 - speed)
    g1, m1 = _square(100)
    g2, _ = _square(100 + speed)
    a = cv2.GaussianBlur(m1.astype(np.float32), (0, 0), 1.5)     # a soft MatAnyone-like edge
    cpu, sc = motion_blur_alpha(a, g1, g0, g2, shutter, strength)
    gpu = align_gpu.open_aligner()
    try:
        gg, sg = motion_blur_alpha(a, g1, g0, g2, shutter, strength, gpu=gpu)
    finally:
        align_gpu.close_aligner(gpu)
    d = np.abs(gg - cpu)
    assert float(d.max()) < 0.02 and float(d.mean()) < 1e-3, (float(d.max()), float(d.mean()))
    assert sg["samples"] == sc["samples"] and abs(sg["max_px"] - sc["max_px"]) < 0.05
