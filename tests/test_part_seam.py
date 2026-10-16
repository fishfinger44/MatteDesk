# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole of an object made of parts: seam width `part_seam_px` (parent field) fills the contact gap between parts."""
import numpy as np

from engine.export import SEAM_PX_DEFAULT, seam_px, seamless_union


def _two_parts(gap):
    a = np.zeros((60, 80), np.float32); b = np.zeros_like(a)
    a[10:50, 5:30] = 1
    b[10:50, 30 + gap:70] = 1
    return a, b


def test_seam_px_default_and_clamp():
    assert seam_px({}) == SEAM_PX_DEFAULT == 4
    assert seam_px({"part_seam_px": 12}) == 12
    assert seam_px({"part_seam_px": 999}) == 40 and seam_px({"part_seam_px": -3}) == 0
    assert seam_px({"part_seam_px": "x"}) == SEAM_PX_DEFAULT


def test_wider_seam_fills_wider_gap():
    a, b = _two_parts(10)
    narrow = seamless_union([a, b], [a > 0, b > 0], gap_px=4)
    wide = seamless_union([a, b], [a > 0, b > 0], gap_px=6)
    assert narrow[30, 30:40].max() == 0          # 10 px gap stays open with the default 4 px
    assert wide[30, 30:40].min() == 1            # filled with 6 px (gap <= 2 x seam)
    assert wide[5, 30:40].max() == 0 and wide[55, 30:40].max() == 0   # outer contour untouched


def test_zero_seam_is_plain_max():
    a, b = _two_parts(2)
    u = seamless_union([a, b], [a > 0, b > 0], gap_px=0)
    assert np.array_equal(u, np.maximum(a, b))
