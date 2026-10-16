# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z22 stage A – SAM3 + CoTracker3 guard metrics (engine/guard.py): a synthetic sequence where the 'SAM3' mask jumps
onto a second object in the middle of the interval must be detected by lost_frac / unsupported_frac."""

import numpy as np
import cv2

from engine.guard import frame_metrics, guard_interval, guard_points, summarize


class _ShiftTracker:
    """Points follow the TRUE object (disc A moving +2 px/frame); frames in `occluded` are invisible."""
    engine = "mock"

    def __init__(self, dx=2.0, occluded=()):
        self.dx = dx
        self.occluded = set(occluded)

    def track(self, frames, queries):
        from engine.tracking import TrackResult
        T, N = len(frames), len(queries)
        tracks = np.zeros((T, N, 2), np.float32)
        visible = np.ones((T, N), bool)
        t0 = int(queries[0, 0])
        for i in range(T):
            tracks[i, :, 0] = queries[:, 1] + self.dx * (i - t0)
            tracks[i, :, 1] = queries[:, 2]
            if i in self.occluded:
                visible[i] = False
        return TrackResult(tracks=tracks, visible=visible, engine=self.engine)


def _masks(n=12, jump_from=7, h=160, w=260):
    """SAM3 stand-in: disc A at x=60+2i (r 30); from `jump_from` the mask is disc B at x=190 (another object)."""
    masks = {}
    for i in range(n):
        m = np.zeros((h, w), np.uint8)
        if i < jump_from:
            cv2.circle(m, (60 + 2 * i, 80), 30, 1, -1)
        else:
            cv2.circle(m, (190, 80), 30, 1, -1)
        masks[i + 1] = m.astype(bool)
    return masks


def test_guard_points_inside_eroded_mask():
    m = np.zeros((160, 260), bool); cv2.circle(m.view(np.uint8), (60, 80), 30, 1, -1)
    pts = guard_points(m, n=60, inset_px=6)
    assert 10 <= len(pts) <= 60
    d = np.hypot(pts[:, 0] - 60, pts[:, 1] - 80)
    assert d.max() <= 30 - 6 + 0.5
    assert len(guard_points(np.zeros((20, 20), bool))) == 0


def test_frame_metrics_lost_and_unsupported():
    m = np.zeros((100, 200), bool); m[20:80, 20:80] = True; m[20:80, 120:180] = True    # two components
    pts = np.array([[50, 50], [55, 45], [150, 50]], np.float32)
    vis = np.array([True, True, True])
    r = frame_metrics(pts, vis, m, radius=10)
    assert r["lost_frac"] == 0.0 and r["unsupported_px"] == 0
    vis2 = np.array([True, True, False])                      # the right component has no visible point -> unsupported
    r2 = frame_metrics(pts, vis2, m, radius=10)
    assert r2["unsupported_px"] == 60 * 60 and abs(r2["unsupported_frac"] - 0.5) < 1e-3 and abs(r2["occluded_frac"] - 1 / 3) < 1e-3
    pts3 = np.array([[50, 50], [100, 50], [150, 50]], np.float32)   # the middle point is outside the mask
    r3 = frame_metrics(pts3, vis, m, radius=10)
    assert abs(r3["lost_frac"] - 1 / 3) < 1e-3
    r4 = frame_metrics(np.zeros((0, 2), np.float32), np.zeros(0, bool), m, 10)
    assert r4["unsupported_frac"] == 1.0 and r4["lost_frac"] == 0.0


def test_guard_interval_detects_the_jump_onto_another_object():
    masks = _masks()
    frames = sorted(masks)
    key = 1
    small = [None] * len(frames)
    met, info = guard_interval(_ShiftTracker(dx=2.0, occluded=[3]), small, 1.0, frames, key, masks[key], masks, n_points=80)
    assert info["points"] >= 10 and set(met) == set(frames)
    s = summarize(met)
    # before the jump: points inside the mask, mask supported; from frame 8 on: every visible point outside, mask unsupported
    assert all(met[f]["lost_frac"] < 0.05 and met[f]["unsupported_frac"] == 0.0 for f in range(1, 8) if f != 4)
    assert all(met[f]["lost_frac"] > 0.9 and met[f]["unsupported_frac"] == 1.0 for f in range(8, 13))
    assert s["lost_frames"] == list(range(8, 13)) and s["unsupported_frames"] == [4] + list(range(8, 13))
    assert met[4]["occluded_frac"] == 1.0 and met[4]["unsupported_frac"] == 1.0     # fully occluded frame flagged, not lost
    # no tracker -> empty metrics without crashing
    met0, info0 = guard_interval(None, small, 1.0, frames, key, masks[key], masks)
    assert info0["points"] > 0 and met0[1]["visible"] == 0
