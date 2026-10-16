# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z25 pkt 1 – the backlit wall estimated on the full X2 frame (engine/backlit_wall) vs the mirrored crop: a defocused,
semi-transparent object at the frame edge gets a wrong (dark) wall from the crop -> coverage blob; the full-frame
estimate stays at the true wall. Plus the graph wiring of the optional `wall` input."""

import numpy as np
import cv2

from engine.backlit_wall import coverage_from_wall, luminance, wall_estimate, wall_full_frame
from engine.crop import CropSegment, extract_crop
from engine.refine import backlit_graph

H, W = 600, 900
WALL = 2.0


def _x2_with_edge_object():
    """Bright wall (2.0 with a gentle gradient) and a big defocused semi-transparent object touching the LEFT edge."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall = WALL * (1.0 + 0.1 * (xx / W))
    obj = np.zeros((H, W), np.float32)
    cv2.circle(obj, (120, 300), 260, 1.0, -1)
    obj = cv2.GaussianBlur(obj, (0, 0), 25)             # defocused edge
    trans = 0.3                                          # the object transmits 30 % (leaf / trunk in X2)
    lum = wall * (1.0 - obj * (1.0 - trans))
    x2 = np.dstack([lum, lum, lum]).astype(np.float32)   # gray -> luminance weights sum to 1
    return x2, wall, obj


def test_full_frame_wall_vs_mirrored_crop_at_the_frame_edge():
    x2, wall_true, obj = _x2_with_edge_object()
    lum = luminance(x2)
    assert np.abs(lum - wall_true * (1.0 - obj * 0.7)).max() < 1e-3
    # crop at the left edge, mirror padding (as the Finalizer feeds the node)
    seg = CropSegment("s00", 1, 1, 0, 0, [-150, 60, 480, 480], [150, 0, 0, 0], 1.0)
    crop = extract_crop(x2, seg)
    lum_c = luminance(crop)
    wall_crop, _ = wall_estimate(lum_c, 0.9, 12.0, 200.0, scale=0.5, border="replicate")      # the node's behaviour
    wall_full = extract_crop(wall_full_frame(x2, {"wall_lo": 0.9, "wall_sigma_small": 12.0, "wall_sigma_large": 200.0}, scale=0.25), seg)
    truth = extract_crop(wall_true, seg)
    inside = extract_crop((obj > 0.5).astype(np.float32), seg) > 0.5          # under the object: the wall is hidden
    err_crop = float(np.abs(wall_crop - truth)[inside].mean() / WALL)
    err_full = float(np.abs(wall_full - truth)[inside].mean() / WALL)
    assert err_full < 0.2, err_full                         # full frame: wall behind the object within 20 % (the wall has a 10 % gradient)
    assert err_crop > 1.5 * err_full, (err_crop, err_full)   # the mirrored crop is clearly worse (measured 0.295 vs 0.149)
    # the coverage under the object: ~0.70 (transmission 0.30) with the full-frame wall; the crop under-estimates it
    cov_full = coverage_from_wall(lum_c, wall_full)
    cov_crop = coverage_from_wall(lum_c, wall_crop)
    core = extract_crop(obj, seg) > 0.9
    assert abs(float(cov_full[core].mean()) - 0.70) < 0.08, cov_full[core].mean()
    assert abs(float(cov_crop[core].mean()) - 0.70) > abs(float(cov_full[core].mean()) - 0.70) + 0.03
    # far from any object the full-frame wall equals the true wall (no mirror artefacts at the frame border)
    far = np.zeros((H, W), bool); far[:, 500:] = True
    wall_f = wall_full_frame(x2, {}, scale=0.25)
    assert float(np.abs(wall_f - wall_true)[far].mean() / WALL) < 0.02


def test_backlit_graph_wall_input_wiring():
    g0 = backlit_graph("b_####.exr", 5, "c_####.png", "a_####.exr", {"wall_lo": 0.9})
    assert "7" not in g0 and "wall" not in g0["2"]["inputs"]
    g1 = backlit_graph("b_####.exr", 5, "c_####.png", "a_####.exr", {"wall_lo": 0.9}, wall_pattern="w_####.exr")
    assert g1["7"]["class_type"] == "PlateLoadEXR" and g1["7"]["inputs"]["path_pattern"] == "w_####.exr"
    assert g1["7"]["inputs"]["last_frame"] == 4 and g1["2"]["inputs"]["wall"] == ["7", 0]
    for k in ("1", "3", "4", "5", "6"):
        assert g1[k] == g0[k]
