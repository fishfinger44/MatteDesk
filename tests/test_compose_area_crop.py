# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""bandsys.compose_area_crop (plan docs/PLAN_EDGE_SPEED.md A6b, 2026-10-28): the whole-outline feather composing on
the band's bbox only gives the bits of compose_area on the full frame – any shape, feathers, a band on the frame edge,
an empty band, with and without `outside` / `partner`."""

import cv2
import numpy as np
import pytest

from engine import bandsys as B


def _scene(seed, H=None, W=None, edge=False):
    rng = np.random.default_rng(seed)
    H = H or int(rng.integers(300, 420))      # odd sizes too: the crop end is not always aligned
    W = W or int(rng.integers(400, 530))
    core = np.zeros((H, W), np.uint8)
    for _ in range(3):
        cx = int(rng.integers(0 if edge else 120, W if edge else W - 120))
        cy = int(rng.integers(0 if edge else 100, H if edge else H - 100))
        cv2.ellipse(core, (cx, cy), (int(rng.integers(30, 90)), int(rng.integers(30, 90))), float(rng.integers(0, 180)),
                    0, 360, 1, -1)
    core = core.astype(bool)
    out_px, in_px = int(rng.integers(10, 45)), int(rng.integers(1, 12))
    grow = cv2.dilate(core.astype(np.uint8), np.ones((2 * out_px + 1,) * 2, np.uint8)).astype(bool)
    shrink = cv2.erode(core.astype(np.uint8), np.ones((2 * in_px + 1,) * 2, np.uint8)).astype(bool)
    area = grow & ~shrink
    # ragged band (band points): cut random notches out of it
    for _ in range(6):
        cv2.circle(area.view(np.uint8), (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(5, 25)), 0, -1)
    area = area.astype(bool)
    model = rng.random((H, W)).astype(np.float32)
    outside = np.where(core, 1.0, rng.random((H, W)) * 0.3).astype(np.float32)
    partner = np.maximum(core.astype(np.float32), rng.random((H, W)).astype(np.float32) * 0.5)
    return core, model, area, outside, partner


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("fi,fo", [(2.0, 8.0), (0.0, 14.0), (6.0, 0.0), (0.0, 0.0), (12.5, 30.0)])
def test_crop_equals_full_frame(seed, fi, fo):
    core, model, area, outside, partner = _scene(seed, edge=seed % 2 == 1)
    full = B.compose_area(core, model, area, fi, fo, outside=outside, partner=partner)
    crop = B.compose_area_crop(core, model, area, fi, fo, outside=outside, partner=partner)
    assert crop.dtype == full.dtype == np.float32
    assert np.array_equal(crop, full)
    # defaults: outside / partner = the core
    assert np.array_equal(B.compose_area_crop(core, model, area, fi, fo), B.compose_area(core, model, area, fi, fo))


def test_small_band_far_from_everything_and_empty_band():
    core, model, area, outside, partner = _scene(3)
    tiny = np.zeros_like(area)
    tiny[150:156, 200:206] = True       # a few band pixels: the crop is tiny, distances to far edges do not matter
    for a in (tiny, np.zeros_like(area)):
        assert np.array_equal(B.compose_area_crop(core, model, a, 4.0, 20.0, outside=outside, partner=partner),
                              B.compose_area(core, model, a, 4.0, 20.0, outside=outside, partner=partner))
