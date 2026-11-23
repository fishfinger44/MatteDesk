# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Band preview lines on the band's crop (engine/bandsys.band_lines, user request 2026-11-20: ~0.8 s a 6K frame for a
band covering 1.9 % of it) – the very same lines as on the whole frame, for a constant band, band points, „cały
rdzeń”, zero feathers, a mask at the frame border and several blobs."""

import cv2
import numpy as np
import pytest

from engine.bandsys import band_lines


def _mask(h=900, w=1400, border=False, blobs=1):
    m = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(7)
    for k in range(blobs):
        cx, cy = (60 if border else 420 + 380 * k), 450
        cv2.ellipse(m, (cx, cy), (230, 300), 15 * k, 0, 360, 1, -1)
        for _ in range(40):   # hair-like strands
            a = rng.uniform(0, 2 * np.pi)
            x1, y1 = int(cx + 260 * np.cos(a)), int(cy + 320 * np.sin(a))
            cv2.line(m, (cx, cy), (x1, y1), 1, int(rng.integers(1, 4)))
    return m.astype(bool)


CASES = [
    dict(out_px=40, in_px=10, inner_px=2, outer_px=8),
    dict(out_px=6, in_px=2, inner_px=0, outer_px=0),
    dict(out_px=40, in_px=float("inf"), inner_px=6, outer_px=8),                 # „cały rdzeń”
    dict(out_px=25, in_px=8, inner_px=3, outer_px=12, pts=[(650, 150, 70, 20), (190, 450, 5, 2)]),
    dict(out_px=25, in_px=float("inf"), inner_px=4, outer_px=6, pts=[(650, 150, 70, 20)], unit=8.0),
]


@pytest.mark.parametrize("kw", CASES)
@pytest.mark.parametrize("shape", [dict(), dict(border=True), dict(blobs=2)])
def test_crop_lines_equal_whole_frame(kw, shape):
    m = _mask(**shape)
    a = band_lines(m, crop=False, **kw)
    b = band_lines(m, crop=True, **kw)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k
    assert a["band_contours"]


@pytest.mark.parametrize("shape", [dict(), dict(border=True), dict(blobs=2)])
@pytest.mark.parametrize("op", [(40, 10, False), (6, 2, False), (40, 4, True), (0, 3, False)])
def test_zone_outline_band_crop_equals_whole_frame(shape, op):
    """Stage-5 zone band (engine/zones.outline_band): the morphology on the crop = on the whole frame."""
    from engine.zones import outline_band
    m = _mask(**shape)
    out_px, in_px, whole = op
    assert np.array_equal(outline_band(m, out_px, in_px, whole), outline_band(m, out_px, in_px, whole, crop=False))


@pytest.mark.parametrize("whole", [False, True])
def test_zone_band_with_points_and_lines_crop(whole):
    """A whole-outline zone with band points (variable band on the crop) and its feather lines (area_lines_crop,
    also with a feather MAP) = the whole-frame computation."""
    from engine.bandmap import variable_band, width_maps
    from engine.bandsys import area_lines, area_lines_crop, whole_feather
    from engine.zones import Zone, outline_regions
    m = _mask()
    pts = [(650, 150, 70, 20), (190, 450, 5, 2)]
    z = Zone(zone_id="z_01", criterion="matte", mode="add", polygon=[], frame=1,
             params={"region": "outline", "outline_out_px": 25, "fg_erode_px": 8, "whole_core": whole,
                     "inset_feather_px": 3, "outer_feather_px": 6})
    reg = outline_regions(z, {1: m}, {}, band_at=lambda f: pts)[1]
    om, im = width_maps(m, pts, 25, 0 if whole else 8)
    if whole:
        im = np.full_like(im, np.inf)
    assert np.array_equal(reg, variable_band(m, om, im))
    for fi in (3.0, 0.0, whole_feather(m, 3.0, pts, 8.0)):
        a = area_lines(reg, m, fi, 6.0, whole=whole)
        b = area_lines_crop(reg, m, fi, 6.0, whole=whole)
        assert a == b and a["band_contours"]


def test_empty_mask():
    m = np.zeros((64, 64), bool)
    assert band_lines(m, 10, 2, 2, 8) == band_lines(m, 10, 2, 2, 8, crop=False)
