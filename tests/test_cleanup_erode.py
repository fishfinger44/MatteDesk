# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""C1 „Pas na zewnątrz” = Erode / Dilate (user decision 2026-11-15): grow_px −100…200; below 0 the patch shrinks inside
the removed objects (Euclidean), the soft edge goes outward from that edge, band points rest; the frame edge does not
erode an object that touches it. 0 and above: unchanged."""
from pathlib import Path

import numpy as np

from engine import cleanup as cu
from engine import inpaint_band as IB


def _disc(r=30, size=128, cx=64, cy=64):
    y, x = np.mgrid[:size, :size]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r * r).astype(np.float32)


def test_settings_range():
    assert cu.settings({"grow_px": -40})["grow_px"] == -40.0
    assert cu.settings({"grow_px": -500})["grow_px"] == cu.GROW_MIN == -100.0
    assert cu.settings({"grow_px": 900})["grow_px"] == cu.GROW_MAX == 200.0


def test_erode_shrinks_inside_and_feathers_outward():
    a = _disc(30)
    m = cu.removal_mask([a], -10, 0)
    assert m[64, 64] == 1.0 and m[64, 64 + 19] == 1.0 and m[64, 64 + 21] == 0.0 and m[64, 64 + 29] == 0.0
    s = cu.removal_mask([a], -10, 6)
    assert 0.0 < s[64, 64 + 23] < 1.0 and s[64, 64 + 27] == 0.0   # the ramp lies inside the object
    assert (cu.removal_mask([a], 0, 3) == cu.removal_mask([a], 0.0, 3)).all()
    assert cu.removal_mask([a], -40, 0).sum() == 0                # eroded away completely


def test_erode_keeps_frame_edge():
    a = np.zeros((64, 64), np.float32); a[:, :20] = 1               # touches the left frame edge
    m = cu.removal_mask([a], -5, 0)
    assert m[32, 0] == 1.0 and m[32, 14] == 1.0 and m[32, 15] == 0.0


def test_erode_with_local_points_2026_11_15():
    """A negative band + a band point: away from the point the slider (inside the object), at the point the patch
    reaches the whisker from the outline, smoothly between; the feather from that edge."""
    a = np.zeros((200, 200), np.float32); a[:, :100] = 1               # the object = left half, outline at x = 100
    pt = [(100.0, 40.0, 30.0, 0.0)]                                     # whisker 30 px out at y = 40
    m = cu.removal_mask([a], -20, 0, pts=pt)
    assert m[40, 125] == 1.0 and m[40, 135] == 0.0                     # at the point: out to ~x = 130
    assert m[190, 79] == 1.0 and m[190, 81] == 0.0                     # far away: eroded by 20 (x < 80)
    row = [int((m[y] > 0.5).sum()) for y in (40, 70, 100, 130)]
    assert row[0] > row[1] > row[2] >= row[3]                          # smooth falloff along the outline
    sft = cu.removal_mask([a], -20, 8, pts=pt)
    assert 0.0 < sft[40, 134] < 1.0                                     # feather at the local edge too
    from engine import bandsys
    off = bandsys.local_offsets((200, 200), pt, -20.0)
    assert off[190, 50] == -20.0 and abs(off[40, 100] - 30.0) < 1e-3


def test_positive_band_with_points_unchanged():
    """The point geometry of the positive band (and of the alpha stages) is the old one, bit for bit."""
    a = np.zeros((128, 128), np.float32); a[30:90, 30:90] = 1
    pts = [(90.0, 60.0, 25.0, 0.0), (30.0, 60.0, 5.0, 0.0)]
    from engine import bandsys
    o, _ = bandsys.band_widths(a > 0, 10.0, 0.0, pts)
    import cv2
    hard = (a > 0.02).astype(np.uint8)
    d = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 5)
    want = ((hard > 0) | (d <= np.asarray(o, np.float32))).astype(np.float32)
    assert (cu.removal_mask([a], 10, 0, pts) == want).all()


def test_lines_with_erode():
    L = IB.lines([_disc(30)], -10, 4)
    xs = [p[0] for c in L["band_contours"] for p in c]
    assert xs and max(xs) < 64 + 22                                  # the band edge inside the outline


def test_ui_slider_and_warning():
    js = (Path(__file__).resolve().parents[1] / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert 'id="cuGrowR" min="-100"' in js and 'id="cuGrow" class="form-input edge-num" min="-100"' in js
    assert 'id="cuGrowWarn"' in js and "przy „Usuń”: brzeg obiektu" in js


def test_whisker_turned_into_the_object_cuts_2026_11_15():
    """The C1 whisker turns all round: pointed into the object (180°) it cuts the patch in there (local, through the
    tip), even at a positive band; the stages keep ±80° (bandmap / bandedit defaults)."""
    a = np.zeros((200, 200), np.float32); a[:, :100] = 1               # outline at x = 100, outward normal = +x
    out_pt = [(100.0, 100.0, 30.0, 0.0)]
    in_pt = [(100.0, 100.0, 30.0, 0.0, 180.0)]
    m_out = cu.removal_mask([a], -10, 0, pts=out_pt)
    m_in = cu.removal_mask([a], -10, 0, pts=in_pt)
    assert m_out[100, 125] == 1.0
    assert m_in[100, 75] == 0.0 and m_in[100, 65] == 1.0                # cut to ~x = 70 at the point
    assert m_in[190, 85] == 1.0 and m_in[190, 95] == 0.0                # far away: the slider (-10)
    pos = cu.removal_mask([a], 10, 0, pts=in_pt)                        # positive band + an inward whisker
    assert pos[100, 75] == 0.0 and pos[190, 105] == 1.0
    from engine import bandsys
    t = bandsys.tip_offsets(a > 0, in_pt + [(100.0, 50.0, 20.0, 0.0, 90.0)])
    assert abs(t[0][2] + 30.0) < 1.5 and abs(t[1][2]) < 1.5            # inward = -30, along the outline = ~0


def test_stage_angles_still_80():
    from engine import bandedit, bandmap
    p = {"id": "p1", "keys": {"1": {"x": 0, "y": 0, "out": 5, "in": 0, "ang": 150}}}
    assert bandmap.angle_at(p, 1) == 80.0 and bandmap.angle_at(p, 1, 180.0) == 150.0
    assert bandedit.ang_field(150, 0) == {"ang": 80.0} and bandedit.ang_field(150, 0, 180) == {"ang": 150.0}
    js = (Path(__file__).resolve().parents[1] / "static" / "bandpoints.js").read_text(encoding="utf-8")
    assert "const lim = isC1() ? 180 : 80;" in js
    from engine import inpaint_band as IB
    assert IB.FULL_TURN_DEG == 180.0
