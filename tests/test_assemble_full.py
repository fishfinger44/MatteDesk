# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 #22: refine.assemble_full composes every crop segment of a frame (a third `manual` one was dropped)."""

import numpy as np

from engine.crop import CropSegment, paste_crop
from engine.refine import assemble_full

W, H, SIDE = 100, 40, 20


def _seg(seg_id, first, last, x, manual=False):
    return CropSegment(seg_id, first, last, 0, 0, [x, 0, SIDE, SIDE], [0, 0, 0, 0], 1.0, manual=manual)


def _per(segs, value=1.0):
    return {s.seg_id: {f: np.full((SIDE, SIDE), value, np.float32) for f in range(s.first, s.last + 1)} for s in segs}


def test_third_segment_reaches_the_result():
    a, b, c = _seg("a", 1, 20, 0), _seg("b", 10, 30, 30), _seg("c", 15, 40, 60, manual=True)   # disjoint crops
    f = 18
    full = assemble_full(_per([a, b, c]), [a, b, c], [f], W, H)[f]
    assert full[5, 65] > 0.0                                 # the third segment's pixels are in the frame
    assert full[5, 5] > 0.0 and full[5, 35] > 0.0            # and the first two stay
    assert full.dtype == np.float32


def test_four_segments_and_order_of_the_list():
    segs = [_seg("d", 8, 12, 80), _seg("b", 5, 15, 30), _seg("c", 7, 15, 60), _seg("a", 1, 12, 0)]
    full = assemble_full(_per(segs), segs, [10], W, H)[10]
    for x in (5, 35, 65, 85):
        assert full[5, x] > 0.0, x


def test_weights_sum_to_one_on_a_shared_crop():
    """All on the same pixels: a constant alpha stays constant through the crossfade, however many segments overlap."""
    segs = [_seg("a", 1, 20, 0), _seg("b", 10, 30, 0), _seg("c", 15, 40, 0)]
    frames = list(range(1, 41))
    full = assemble_full(_per(segs, 0.6), segs, frames, W, H)
    for f in frames:
        assert abs(float(full[f][5, 5]) - 0.6) < 1e-5, f


def test_no_jump_when_a_segment_of_three_ends():
    """A segment leaves the mix with a small weight on its last frame – no pop in the frame after."""
    a, b, c = _seg("a", 1, 20, 0), _seg("b", 10, 30, 30), _seg("c", 15, 40, 60)
    frames = list(range(1, 41))
    full = assemble_full(_per([a, b, c]), [a, b, c], frames, W, H)
    assert full[20][5, 5] < 0.1 and full[21][5, 5] == 0.0     # a ends at 20
    assert full[30][5, 35] < 0.1 and full[31][5, 35] == 0.0   # b ends at 30


def test_one_and_two_segments_keep_the_old_result_bit_for_bit():
    rng = np.random.default_rng(1)
    a, b = _seg("a", 1, 20, 0), _seg("b", 10, 30, 10)
    per = {s.seg_id: {f: rng.random((SIDE, SIDE), dtype=np.float32) for f in range(s.first, s.last + 1)} for s in (a, b)}
    frames = list(range(1, 31))
    full = assemble_full(per, [b, a], frames, W, H)
    for f in frames:
        if f < 10:
            old = paste_crop(per["a"][f], a, W, H)
        elif f > 20:
            old = paste_crop(per["b"][f], b, W, H)
        else:   # the two-segment branch as it was before the fix
            lo, hi = b.first, a.last
            wb = (f - lo + 1) / float(hi - lo + 2)
            old = (1 - wb) * paste_crop(per["a"][f], a, W, H) + wb * paste_crop(per["b"][f], b, W, H)
        assert np.array_equal(full[f], old), f
    assert np.array_equal(assemble_full({}, [a], [1], W, H)[1], np.zeros((H, W), np.float32))
