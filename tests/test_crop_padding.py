# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Padding outside the plate must not duplicate the object (mirror padding made SAM3 track the mirrored copy,
SH030 2026-10-14)."""

import numpy as np

from engine.crop import CropSegment, extract_crop, letterbox_square


def test_letterbox_pads_with_black_not_mirror():
    img = np.zeros((108, 192, 3), np.float32)
    img[:20, 80:110] = 1.0          # object touching the top edge
    img[-20:, 70:120] = 1.0         # and the bottom edge
    sq, (ox, oy) = letterbox_square(img)
    assert sq.shape[:2] == (192, 192) and (ox, oy) == (0, 42)
    assert sq[:oy].max() == 0.0 and sq[oy + 108:].max() == 0.0
    np.testing.assert_array_equal(sq[oy:oy + 108], img)


def test_extract_crop_pads_with_black():
    img = np.ones((100, 100), np.float32)
    seg = CropSegment(seg_id="s00", first=1, last=1, overlap_in=0, overlap_out=0,
                      crop=[0, -20, 140, 140], pad=[0, 20, 40, 20], scale_to_1008=7.2)
    patch = extract_crop(img, seg)
    assert patch.shape == (140, 140)
    assert patch[:20].max() == 0.0 and patch[-20:].max() == 0.0 and patch[:, -40:].max() == 0.0
    assert patch[20:120, :100].min() == 1.0
