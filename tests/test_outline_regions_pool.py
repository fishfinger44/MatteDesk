# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""zones.outline_regions with many frames runs the band geometry in a thread pool (plan docs/PLAN_EDGE_SPEED.md A6b,
2026-10-28): every frame must be bit-identical to the same frame computed alone (one frame = no pool), with band points,
a drawing and an excluded object, and the callables are still called in frame order from the caller's thread."""

import threading

import cv2
import numpy as np

from engine.zones import Zone, outline_regions


def _core(f, h=240, w=320):
    c = np.zeros((h, w), np.uint8)
    cv2.ellipse(c, (140 + 3 * f, 120), (70 + f, 50), 10.0 * f, 0, 360, 1, -1)
    cv2.circle(c, (230, 90 + 2 * f), 25, 1, -1)
    return c.astype(bool)


def test_pool_equals_single_frames_and_keeps_order():
    frames = list(range(1, 13))
    cores = {f: _core(f) for f in frames}
    drawn = {f: np.zeros_like(cores[f]) for f in frames}
    for f in frames:
        drawn[f][:, : 40 + f] = True
    hand = np.zeros_like(cores[1]); hand[180:, 150:190] = True
    z = Zone("z_01", "add", "matte", 1, matting="vitmatte",
             params={"region": "outline", "outline_out_px": 18, "outline_in_px": 3, "exclude_grow_px": 4})
    calls = []

    def band_at(f):            # band points on even frames only: both band paths (variable / constant) in one run
        calls.append(("band", f, threading.get_ident()))
        return [(200.0, 120.0, 30.0, 12.0), (90.0, 120.0, 6.0, 2.0)] if f % 2 == 0 else None

    def exclude(f):
        calls.append(("excl", f, threading.get_ident()))
        return hand if f % 3 == 0 else None
    many = outline_regions(z, cores, drawn, exclude, band_at=band_at)
    me = threading.get_ident()
    assert all(t == me for _k, _f, t in calls)
    assert [f for k, f, _t in calls if k == "band"] == frames and [f for k, f, _t in calls if k == "excl"] == frames
    assert list(many) == frames
    for f in frames:
        one = outline_regions(z, {f: cores[f]}, {f: drawn[f]}, exclude, band_at=band_at)[f]
        assert one.dtype == many[f].dtype == bool and np.array_equal(one, many[f]), f
        assert many[f].any()
