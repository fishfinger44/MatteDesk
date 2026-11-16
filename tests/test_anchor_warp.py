# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Linked anchors moved from the main one (engine/anchor_warp.py, user 2026-11-11): the main generation's pattern is
warped onto the linked frame along tracked points, then (warp_qwen) touched up by Qwen-Image-2.1 at low denoise."""
import numpy as np

from engine import anchor_warp as AW
from engine import cleanup as cu
from engine.cleanup3d.regen import qwen_graph
from engine.tracking import TrackResult

W, H = 200, 160


def _disc(cx, cy, r=40):
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.hypot(xx - cx, yy - cy) < r).astype(np.float32)


def _pattern(cx, cy):
    """A checker pattern fixed to the object (moves with it)."""
    yy, xx = np.mgrid[0:H, 0:W]
    c = (((xx - cx) // 8 + (yy - cy) // 8) % 2).astype(np.uint8) * 200 + 30
    return np.dstack([c, c, c]).astype(np.uint8)


class ShiftTracker:
    """Every frame index i moves the object by (dx * i, dy * i) – exact tracks."""
    engine = "fake"

    def __init__(self, dx, dy):
        self.d = np.array([dx, dy], np.float32)

    def track(self, frames, queries):
        t0 = int(queries[0, 0])
        p = queries[:, 1:].astype(np.float32)
        tr = np.stack([p + self.d * (i - t0) for i in range(len(frames))])
        return TrackResult(tr, np.ones(tr.shape[:2], bool), "fake")


def test_query_points_cover_the_patch_and_respect_the_limit():
    m = _disc(100, 80)
    pts = AW.query_points(m, max_n=200)
    assert 50 < len(pts) <= 260
    # 2026-11-11: only ON the patch, a little inside its edge (a ring around it caught the hair)
    assert np.hypot(pts[:, 0] - 100, pts[:, 1] - 80).max() < 40 - AW.INSET_PX + 1
    assert len(AW.query_points(np.zeros((H, W), np.float32))) == 0


def test_warp_moves_the_main_pattern_onto_the_linked_pose():
    main_cx, main_cy, dx, dy = 80, 80, 6, 2               # 5 frames later: +30, +10 px
    main_img = _pattern(main_cx, main_cy)
    plate8 = np.full((H, W, 3), 128, np.uint8)
    frames = [1, 2, 3, 4, 5, 6]
    tgt = 6
    mask_t = _disc(main_cx + 5 * dx, main_cy + 5 * dy)
    pts = AW.query_points(_disc(main_cx, main_cy))
    tr = AW.track(ShiftTracker(dx, dy), frames, lambda f: plate8, 1, [tgt], pts)
    p, v = tr[tgt]
    out = AW.warp_anchor(main_img, plate8, mask_t, pts, p, v)
    want = _pattern(main_cx + 5 * dx, main_cy + 5 * dy)
    inside = mask_t > 0.5
    inner = inside & (_disc(main_cx + 5 * dx, main_cy + 5 * dy, 34) > 0)
    assert (np.abs(out[inner].astype(int) - want[inner].astype(int)) <= 2).mean() > 0.97   # the same pattern, moved
    assert (out[~inside] == 128).all()                                                # outside the patch: the plate


def test_track_runs_one_call_both_ways():
    calls = []

    class T(ShiftTracker):
        def track(self, frames, queries):
            calls.append(len(frames))
            return super().track(frames, queries)
    pts = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 5.0]], np.float32)
    tr = AW.track(T(1, 0), list(range(1, 51)), lambda f: np.zeros((H, W, 3), np.uint8), 31, [1, 7, 41, 50], pts)
    assert calls == [50] and sorted(tr) == [1, 7, 41, 50]
    assert np.allclose(tr[1][0][:, 0], pts[:, 0] - 30) and np.allclose(tr[50][0][:, 0], pts[:, 0] + 19)


def test_settings_and_qwen_denoise():
    s = cu.settings({})
    assert s["link_method"] == "qwen" and s["link_denoise"] == 0.4          # default: Qwen alone (user 2026-11-11)
    assert cu.settings({"link_method": "bad", "link_denoise": 7})["link_method"] == "qwen"
    assert cu.settings({"link_denoise": 7})["link_denoise"] == 1.0
    assert "link_method" in cu.PER_TARGET and "link_denoise" in cu.PER_TARGET
    g = qwen_graph("c", "p", "t", 1, None, 0.35)
    assert g["12"]["inputs"]["denoise"] == 0.35 and qwen_graph("c", "p", "t")["12"]["inputs"]["denoise"] == 1.0
    assert "Keep the pattern" in AW.touchup_prompt("porcelain skin")


def test_holes_where_the_main_patch_does_not_reach_and_where_torn():
    """2026-11-11: the part of the linked patch the main one does not cover (hair over the face in the main frame) is a
    hole – filled from the pattern around it (not stretched pixels) and repainted by Qwen-Image-2.1 in full."""
    main_img = _pattern(100, 80)
    plate8 = np.full((H, W, 3), 128, np.uint8)
    mask = _disc(100, 80)
    main_mask = mask.copy()
    main_mask[:, 115:] = 0                                  # the main frame's patch misses the right side
    pts = AW.query_points(main_mask)
    out, hole = AW.warp_anchor(main_img, plate8, mask, pts, pts.copy(), np.ones(len(pts), bool),
                               main_mask=main_mask, holes=True)
    assert hole[80, 130] and not hole[80, 90] and not hole[80, 30]   # in the patch only
    assert (out[~(mask > 0.5)] == 128).all()
    assert out[80, 130].mean() != 128                                 # filled from the pattern, not the plate
    # a torn warp (a strong local stretch) is a hole too
    mx, my = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx2 = mx.copy()
    mx2[:, 100:] = 100 + (mx[:, 100:] - 100) * 0.1                    # 10 x magnification on the right
    ok = AW.stretch_ok(mx2, my)
    assert ok[80, 50] and not ok[80, 150]
