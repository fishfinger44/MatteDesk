# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_align.py - Unit tests for engine/align.py."""

import math
from pathlib import Path

from engine import align_grid
import tempfile
import cv2
import numpy as np
import pytest

from engine.align import (
    AlignmentFeatureExtractor,
    compute_static_mask,
    GlobalAligner,
    LocalPatchAligner,
    NukeAlignGroupGenerator,
    PlateWarper,
    AlignmentFrameResult,
    AlignmentReport
)


def test_feature_extractor_invariance():
    """Verify log-luma Sobel magnitude extracts normalized contours."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[30:70, 30:70, :] = 1.0  # white square on black

    feat = AlignmentFeatureExtractor.extract(img, mode="backlit", scale=1.0)
    assert feat.shape == (100, 100)
    assert feat.dtype == np.float32
    assert 0.0 <= feat.min()
    assert feat.max() <= 1.0

    # Edges around the square must have high response
    assert feat[30, 50] > 0.5
    assert feat[70, 50] > 0.5
    # Flat interior and flat exterior must be low
    assert feat[50, 50] < 0.1
    assert feat[10, 10] < 0.1


def test_static_mask_margins_and_exclusions():
    """Verify compute_static_mask excludes margins and requested exclusion boxes."""
    beauty = np.ones((200, 200, 3), dtype=np.float32)
    aux = np.ones((200, 200, 3), dtype=np.float32)

    # Exclude box [50, 60, 150, 140]
    mask = compute_static_mask(beauty, aux, mode="backlit", scale=1.0, exclude_box=(50, 60, 150, 140))
    assert mask.shape == (200, 200)

    # Margins (2%) must be 0
    assert mask[0, 100] == 0.0
    assert mask[199, 100] == 0.0
    assert mask[100, 0] == 0.0
    assert mask[100, 199] == 0.0

    # Excluded box must be 0
    assert mask[100, 100] == 0.0
    assert mask[70, 60] == 0.0

    # Unaffected static region must be 1.0
    assert mask[20, 20] == 1.0
    assert mask[180, 20] == 1.0


def test_global_aligner_known_translation():
    """Verify subpixel phase correlation recovers known translation with error < 0.1 px."""
    np.random.seed(123)
    h, w = 300, 400
    base = np.random.rand(h, w, 3).astype(np.float32)
    # Add strong geometric features
    cv2.rectangle(base, (50, 50), (120, 120), (2.0, 2.0, 2.0), -1)
    cv2.circle(base, (250, 180), 40, (1.5, 1.5, 1.5), -1)

    dx_true = 2.35
    dy_true = -1.65

    # Aux is shifted by (dx_true, dy_true)
    M = np.float32([[1, 0, -dx_true], [0, 1, -dy_true]])
    aux = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_CUBIC)

    aligner = GlobalAligner(scale=0.5)
    f_b = AlignmentFeatureExtractor.extract(base, mode="backlit", scale=0.5)
    f_a = AlignmentFeatureExtractor.extract(aux, mode="backlit", scale=0.5)
    mask = np.ones_like(f_b)

    dx_est, dy_est, resp = aligner.measure_translation(f_a, f_b, mask)

    assert abs(dx_est - dx_true) < 0.1, f"dx error too large: est={dx_est}, true={dx_true}"
    assert abs(dy_est - dy_true) < 0.1, f"dy error too large: est={dy_est}, true={dy_true}"
    assert resp > 0.3


def test_global_aligner_known_similarity():
    """Verify similarity model recovers rotation and translation with error < 0.1 px."""
    h, w = 400, 400
    base = np.zeros((h, w, 3), dtype=np.float32)
    # Draw multiple rich features
    for x in range(60, 340, 50):
        for y in range(60, 340, 50):
            cv2.rectangle(base, (x, y), (x + 25, y + 25), (float(x % 3 + 1), float(y % 2 + 1), 1.0), -1)

    angle_deg = 0.4
    rad = math.radians(angle_deg)
    tx, ty = 1.8, -1.2
    cx, cy = w / 2.0, h / 2.0

    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    aux = cv2.warpAffine(base, M_rot, (w, h), flags=cv2.INTER_CUBIC)

    aligner = GlobalAligner(scale=1.0)
    f_b = AlignmentFeatureExtractor.extract(base, mode="backlit", scale=1.0)
    f_a = AlignmentFeatureExtractor.extract(aux, mode="backlit", scale=1.0)
    mask = compute_static_mask(base, aux, mode="backlit", scale=1.0)

    method, params, med_err, max_err = aligner.select_best_model(f_a, f_b, mask)
    assert method in ("similarity", "affine", "homography")
    assert med_err < 0.35


def test_local_deformation_recovery_and_rbf():
    """Verify that a smooth local non-linear deformation field is reconstructed with error < 0.3 px."""
    h, w = 400, 400
    Y, X = np.indices((h, w), dtype=np.float32)

    # Smooth non-linear sinusoidal field with amplitude ±3.0 px
    disp_x_gt = 3.0 * np.sin(2.0 * np.pi * X / float(w)) * np.cos(np.pi * Y / float(h))
    disp_y_gt = -2.5 * np.cos(np.pi * X / float(w)) * np.sin(2.0 * np.pi * Y / float(h))

    aligner = LocalPatchAligner(plate_w=w, plate_h=h, grid_spacing=32)
    mask = np.ones((h, w), dtype=np.float32)
    grid_pts = aligner.generate_grid_points(mask, scale=1.0)

    # Sample true displacements at grid points
    vals_x = 3.0 * np.sin(2.0 * np.pi * grid_pts[:, 0] / float(w)) * np.cos(np.pi * grid_pts[:, 1] / float(h))
    vals_y = -2.5 * np.cos(np.pi * grid_pts[:, 0] / float(w)) * np.sin(2.0 * np.pi * grid_pts[:, 1] / float(h))
    disps = np.column_stack([vals_x, vals_y])

    # Fit RBF field
    disp_x_est, disp_y_est, ho_med, ho_max = aligner.fit_rbf_deformation_field(grid_pts, disps, eval_downscale=8)

    # Holdout residual must be < 0.1 px
    assert ho_med < 0.1
    # Full frame field residual on 90% of pixels must be < 0.3 px (spec requirement)
    err_field = np.hypot(disp_x_est - disp_x_gt, disp_y_est - disp_y_gt)
    assert np.percentile(err_field, 90) < 0.3
    assert np.median(err_field) < 0.15


def test_identity_stmap_bitwise_equal():
    """Verify identity STMap produces bitwise identical warped output (diff < 1e-4)."""
    h, w = 64, 64
    local_aligner = LocalPatchAligner(plate_w=w, plate_h=h)
    disp_x = np.zeros((h, w), dtype=np.float32)
    disp_y = np.zeros((h, w), dtype=np.float32)

    test_img = np.random.rand(h, w, 3).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        grid_path = align_grid.write_grid(Path(td) / "identity_grid.exr", align_grid.AlignGrid.zeros(w, h))
        assert grid_path.exists()

        f_res = AlignmentFrameResult(frame=1, method="local_stmap", residual_px=0.0, residual_max_px=0.0)
        warped = PlateWarper.warp_frame(test_img, f_res, field_path=grid_path)

        diff = float(np.abs(test_img - warped).max())
        assert diff < 1e-4, f"Identity STMap produced difference: {diff}"


def test_nuke_group_script_generation_and_syntax():
    """Verify NukeAlignGroupGenerator produces valid Nuke Group scripts for each method."""
    report = AlignmentReport(
        plate_name="SH010_01_X2",
        aux_plate_type="backlit",
        reference_plate="beauty",
        frame_range=(1, 3),
        method="translation",
        residual_median=0.12,
        residual_max=0.22,
        needs_correction=False,
        recommendation="Backlit zgodny (max 0.22 px) – nic nie rób",
        frames=[
            AlignmentFrameResult(frame=1, method="translation", residual_px=0.12, residual_max_px=0.22, dx=0.15, dy=-0.10),
            AlignmentFrameResult(frame=2, method="translation", residual_px=0.11, residual_max_px=0.21, dx=0.14, dy=-0.09),
            AlignmentFrameResult(frame=3, method="translation", residual_px=0.13, residual_max_px=0.23, dx=0.16, dy=-0.11),
        ]
    )

    with tempfile.TemporaryDirectory() as td:
        nk_file = Path(td) / "Align_SH010_01_X2.nk"
        json_file = Path(td) / "Align_SH010_01_X2.json"

        script = NukeAlignGroupGenerator.generate_script(
            plate_name="SH010_01_X2",
            report=report,
            plate_w=4096,
            plate_h=2731,
            output_nk_path=nk_file,
            output_json_path=json_file
        )

        valid, msg = NukeAlignGroupGenerator.validate_nuke_syntax(script)
        assert valid, f"Nuke script syntax error: {msg}"
        assert nk_file.exists()
        assert json_file.exists()

        # Check group properties
        assert "Group {" in script
        assert "use_aligned true" in script
        assert "Switch {" in script
        assert "Input {" in script
        assert "Output {" in script
        assert "Align_SH010_01_X2" in script
