# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core propagation between two keys when one direction loses the object.

Found on SH010 (Z1, 2026-10-12): backward SAM3 from a bad key went empty after 2 frames; the SDF blend of the
good forward mask with an empty mask (SDF -1e4) wiped the whole interval -> 17 empty core frames between
two keys that both contain the object.
"""

import numpy as np

from engine.crop import CropSegment
from engine.segment import CorePropagator


class _LosingRunner:
    """Forward tracking keeps the init mask; backward tracking (detected by the init mask marker) loses the
    object after its first frame."""

    def __init__(self, lose_marker_px: int):
        self.lose_marker_px = lose_marker_px

    def track(self, imgs, init):
        n = len(imgs)
        m = (init > 0.5).astype(np.float32)
        if int(m.sum()) == self.lose_marker_px:  # the "bad" key: lost after 1 frame
            return [m] + [np.zeros_like(m) for _ in range(n - 1)]
        return [m.copy() for _ in range(n)]


def _seg(w, h, first, last):
    return CropSegment(seg_id="s0", first=first, last=last, overlap_in=0, overlap_out=0,
                       crop=[0, 0, w, w], pad=[0, 0, 0, h - w] if h < w else [0, 0, 0, 0], scale_to_1008=1.0)


def test_one_direction_lost_keeps_the_other():
    w = h = 64
    good = np.zeros((h, w), bool)
    good[20:40, 20:40] = True                      # 400 px
    bad = np.zeros((h, w), bool)
    bad[22:30, 22:30] = True                       # 64 px -> runner loses it
    runner = _LosingRunner(lose_marker_px=int(bad.sum()))
    prop = CorePropagator(runner, [_seg(w, h, 1, 20)], w, h,
                          load_crop_image=lambda f, s: np.zeros((s.side, s.side, 3), np.float32))
    res = prop.propagate(list(range(1, 21)), {1: good, 20: bad})
    empty = [f for f in range(1, 21) if not res.core[f].any()]
    assert empty == [], f"empty core frames between two keys: {empty}"
    # frames where only forward tracks: forward mask kept, flagged uncertain (IoU 0)
    for f in range(3, 20):
        assert np.array_equal(res.core[f], good), f
        assert res.fwd_bwd_iou[f] == 0.0
    assert 10 in res.uncertain and res.suggested_keys


def test_both_directions_present_still_blend():
    w = h = 64
    a = np.zeros((h, w), bool)
    a[10:30, 10:30] = True
    b = np.zeros((h, w), bool)
    b[16:36, 16:36] = True                        # overlaps a (shifted 6 px)
    prop = CorePropagator(_LosingRunner(lose_marker_px=-1), [_seg(w, h, 1, 11)], w, h,
                          load_crop_image=lambda f, s: np.zeros((s.side, s.side, 3), np.float32))
    res = prop.propagate(list(range(1, 12)), {1: a, 11: b})
    mid = res.core[6]
    assert mid.any() and not np.array_equal(mid, a) and not np.array_equal(mid, b)  # SDF blend, not a switch
