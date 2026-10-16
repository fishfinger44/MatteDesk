# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""SPEC 7.2 pkt 7: per-interval propagation engine "points" (contour tracked by point tracks, CoTracker3 / LK fallback)."""

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.segment import CorePropagator, PointPropagator, contour_points, iou
from engine.tracking import LKTracker


def _tex(h, w, seed):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    return (t - t.min()) / (t.max() - t.min())


def _shot(T=9, h=240, w=320):
    bg, fg = _tex(h, w, 1) * 0.5, 0.4 + 0.6 * _tex(h, w, 2)
    yy, xx = np.mgrid[0:h, 0:w]
    imgs, masks = {}, {}
    for t in range(T):
        cx, cy = 110 + 4 * t, 120 + 1.5 * t
        m = ((xx - cx) / 55.0) ** 2 + ((yy - cy) / 40.0) ** 2 <= 1.0
        shift = cv2.warpAffine(fg, np.float32([[1, 0, 4 * t], [0, 1, 1.5 * t]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        img = np.where(m, shift, bg)
        imgs[1001 + t] = np.repeat(img[..., None], 3, 2).astype(np.float32)
        masks[1001 + t] = m
    return imgs, masks


class _NoSam:
    def detect(self, *a, **k):
        raise AssertionError("SAM3 must not run on a 'points' interval")

    track = detect


def test_contour_points_on_object():
    m = np.zeros((100, 100), bool)
    m[20:80, 30:70] = True
    pts = contour_points(m)[0]
    assert len(pts) > 30
    assert m[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)].all()


def test_points_interval_follows_object():
    imgs, masks = _shot()
    h, w = masks[1001].shape
    pp = PointPropagator(LKTracker(), lambda f: imgs[f], w, h)
    assert pp.engine == "lk_fallback"  # honest name of the engine that ran
    seg = CropSegment(seg_id="s0", first=1001, last=1009, overlap_in=0, overlap_out=0, crop=[0, 0, w, w], pad=[0, 0, 0, 0], scale_to_1008=1.0)
    prop = CorePropagator(_NoSam(), [seg], w, h, lambda f, s: None, point_propagator=pp,
                          interval_engines={"1001-1009": "points"})
    res = prop.propagate(list(range(1001, 1010)), {1001: masks[1001], 1009: masks[1009]})
    for f in range(1001, 1010):
        assert iou(res.core[f], masks[f]) >= 0.95, (f, iou(res.core[f], masks[f]))
