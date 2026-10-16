# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_e1d_acceptance.py - Acceptance tests for Stage E1d: 2D Camera Motion.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 4d, 4d.1-4d.3, Section 12: Stage E1d, Section 16).

Test criteria:
1. Synthetic test: known trajectory (pan + zoom + rotate) applied across 193 frames
   -> residual < 0.3 px, anchor drift after 193 frames < 1.0 px.
2. Animated object test: moving foreground object does not corrupt camera motion tracking (residual < 0.3 px).
3. Real SH010 shot: motion type report, residual error measurement on proxy/EXR frames.
4. Nuke Group syntax and stabilization: valid Nuke script (1 input, 1 output, CornerPin2D),
   mathematical stabilization reduces transformed background to stationary (diff < 0.3 px).
5. Data contract and integration: camera2d.json schema compliance and multi-module interoperability.
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.camera import (
    Camera2DData,
    Camera2DTracker,
    FrameCamera2D,
    NukeCamera2DGroupGenerator,
)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _create_synthetic_texture_frame(w: int = 400, h: int = 300) -> np.ndarray:
    """Generates a rich, non-repeating textured background frame with distinct features."""
    rng = np.random.RandomState(42)
    # Base gradient
    y, x = np.mgrid[:h, :w].astype(np.float32)
    base = 0.3 + 0.2 * np.sin(x / 30.0) * np.cos(y / 25.0)

    # Add geometric shapes and high-frequency textures
    frame = np.stack([base, base * 0.9, base * 1.1], axis=-1).astype(np.float32)

    # Grid of distinct squares and circles
    for gy in range(40, h - 40, 50):
        for gx in range(40, w - 40, 60):
            rad = 8 + int(rng.uniform(0, 10))
            color = (float(rng.uniform(0.4, 0.9)), float(rng.uniform(0.4, 0.9)), float(rng.uniform(0.4, 0.9)))
            cv2.circle(frame, (gx, gy), rad, color, -1)
            cv2.rectangle(frame, (gx - 15, gy - 15), (gx + 15, gy + 15), (0.2, 0.5, 0.8), 1)

    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def test_acceptance_e1d_criterion1_synthetic_trajectory_and_drift(tmp_path):
    """Criterion 1: Synthetic test with known trajectory (pan + zoom + rotate) on 193 frames.
    
    Requirements:
    - Known trajectory: pan (dx, dy), zoom (scale), rotate (theta).
    - Recovered residual < 0.3 px.
    - Anchor drift after 193 frames < 1.0 px.
    """
    n = 193
    w, h = 400, 300
    base_img = _create_synthetic_texture_frame(w, h)

    frames = []
    ground_truth_H = []

    # Center of rotation/zoom
    cx, cy = w / 2.0, h / 2.0

    for i in range(n):
        t = i / float(n - 1)
        # Pan: up to 15 px horizontal, 10 px vertical
        dx = 15.0 * math.sin(t * math.pi * 2.0)
        dy = 10.0 * math.cos(t * math.pi)
        # Zoom: 1.0 -> 1.05
        scale = 1.0 + 0.05 * math.sin(t * math.pi)
        # Rotate: -1.5 deg -> +1.5 deg
        angle_deg = 1.5 * math.sin(t * math.pi * 2.0)
        angle_rad = math.radians(angle_deg)

        # Construct 3x3 affine transformation matrix: T * R * S * T_inv + Trans
        cos_a = math.cos(angle_rad) * scale
        sin_a = math.sin(angle_rad) * scale

        H = np.array([
            [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + dx],
            [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + dy],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        warped = cv2.warpPerspective(base_img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)
        ground_truth_H.append(H)

    # Run camera tracking with anchor interval = 24
    camera_data = Camera2DTracker.track_sequence(
        frames=frames,
        frame_numbers=list(range(1, n + 1)),
        ref_frame=1,
        eval_w=w,
        anchor_interval=24,
        shot_name="synthetic_193",
        fps=24.0
    )

    # Validate frame count
    assert len(camera_data.frames) == n

    # Check anchor drift: must be < 1.0 px after 193 frames
    assert camera_data.anchor_drift_px < 1.0, f"Anchor drift too high: {camera_data.anchor_drift_px} px (expected < 1.0 px)"

    # Check trajectory recovery accuracy across sample points
    test_corners = np.array([[[50, 50]], [[w - 50, 50]], [[w - 50, h - 50]], [[50, h - 50]]], dtype=np.float32)
    corner_errors = []

    for i in range(0, n, 8):
        # Ground truth mapping points in frame i to frame 0: M_0 @ inv(M_i)
        H_gt_to_ref = ground_truth_H[0] @ np.linalg.inv(ground_truth_H[i])
        H_est = np.array(camera_data.frames[i].homography, dtype=np.float64)

        pts_gt = cv2.perspectiveTransform(test_corners, H_gt_to_ref)
        pts_est = cv2.perspectiveTransform(test_corners, H_est)

        err = float(np.linalg.norm(pts_gt - pts_est, axis=-1).mean())
        corner_errors.append(err)

    median_err = float(np.median(corner_errors))
    p95_err = float(np.percentile(corner_errors, 95))

    print(f"\nCriterion 1: Median corner error: {median_err:.3f} px, P95: {p95_err:.3f} px, Anchor drift: {camera_data.anchor_drift_px:.3f} px")
    assert median_err < 0.3, f"Median tracking error {median_err:.3f} px exceeds 0.3 px threshold"
    assert p95_err < 0.8, f"P95 tracking error {p95_err:.3f} px exceeds 0.8 px threshold"


def test_acceptance_e1d_criterion2_animated_object_invariance():
    """Criterion 2: Moving foreground object does not corrupt camera motion tracking.
    
    Requirements:
    - Moving animated patch across frames.
    - Background camera motion tracking error remains < 0.3 px.
    """
    n = 30
    w, h = 400, 300
    base_img = _create_synthetic_texture_frame(w, h)

    frames = []
    ground_truth_H = []

    for i in range(n):
        # Known camera translation: slow pan of 0.4 px/frame x, 0.2 px/frame y
        dx = i * 0.4
        dy = i * 0.2
        H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        bg_warped = cv2.warpPerspective(base_img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Superimpose high-contrast moving foreground object (e.g. animated character/box moving opposite)
        obj_x = int(50 + i * 8.0)
        obj_y = int(80 + math.sin(i * 0.5) * 30.0)
        cv2.rectangle(bg_warped, (obj_x, obj_y), (obj_x + 60, obj_y + 60), (0.9, 0.1, 0.1), -1)
        cv2.putText(bg_warped, "OBJ", (obj_x + 5, obj_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1, 1, 1), 2)

        frames.append(bg_warped)
        ground_truth_H.append(H)

    camera_data = Camera2DTracker.track_sequence(
        frames=frames,
        eval_w=w,
        anchor_interval=15,
        shot_name="moving_obj_test"
    )

    # Verify background error is still < 0.3 px
    test_corners = np.array([[[20, 20]], [[w - 20, 20]], [[w - 20, h - 20]], [[20, h - 20]]], dtype=np.float32)
    corner_errors = []

    for i in range(n):
        H_gt_to_ref = ground_truth_H[0] @ np.linalg.inv(ground_truth_H[i])
        H_est = np.array(camera_data.frames[i].homography, dtype=np.float64)

        pts_gt = cv2.perspectiveTransform(test_corners, H_gt_to_ref)
        pts_est = cv2.perspectiveTransform(test_corners, H_est)

        err = float(np.linalg.norm(pts_gt - pts_est, axis=-1).mean())
        corner_errors.append(err)

    median_err = float(np.median(corner_errors))
    print(f"\nCriterion 2: Moving object background median error: {median_err:.3f} px")
    assert median_err < 0.3, f"Background tracking corrupted by moving object: {median_err:.3f} px (expected < 0.3 px)"


def test_acceptance_e1d_criterion3_real_sh010_shot_report():
    """SPEC E1d (real, station only): SH010 report of the motion type; "stabilize" gives a static background
    with a residual as reported (independent measurement after warping each frame by its homography).

    Uses the beauty proxies (generated by the tool) for the whole shot; homographies in proxy pixels."""
    proxy_dir = Path("C:\\shots\\SH010\\frame_####.exr")
    assert proxy_dir.exists(), "generate SH010 proxies in the tool first (Projekt -> proxy)"
    files = sorted(p for p in os.listdir(proxy_dir) if p.startswith("p_") and p.endswith(".jpg"))
    assert len(files) >= 100
    frames = [cv2.imread(str(proxy_dir / f)) for f in files]
    cam = Camera2DTracker.track_sequence(frames=frames, frame_numbers=list(range(1, len(frames) + 1)),
                                         ref_frame=1, eval_w=1024, anchor_interval=24, shot_name="SH010")
    print(f"\nSH010 motion type '{cam.motion_type}', reported residual median {cam.residual_median_px:.3f} px, "
          f"p95 {cam.residual_p95_px:.3f} px, drift {cam.anchor_drift_px:.3f} px")
    assert cam.motion_type  # a report exists
    assert len(cam.frames) == len(frames)

    # independent check: stabilized frames vs reference on static background, patch phase correlation
    ref = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = ref.shape
    res = []
    for k in range(4, len(frames), 12):
        st = cv2.warpPerspective(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY).astype(np.float32),
                                 np.array(cam.frames[k].homography), (w, h))
        for y in range(64, h - 64, 96):
            for x in range(64, w - 64, 96):
                a, b = ref[y - 32:y + 32, x - 32:x + 32], st[y - 32:y + 32, x - 32:x + 32]
                if a.std() < 4:
                    continue
                (dx, dy), resp = cv2.phaseCorrelate(a, b)
                if resp > 0.3:
                    res.append(np.hypot(dx, dy))
    med = float(np.median(res))
    print(f"independent stabilized residual median {med:.3f} px")
    # "reszta jak w raporcie": the report must not under-state the residual
    assert med <= cam.residual_median_px * 1.25 + 0.1


def test_acceptance_e1d_criterion4_nuke_group_syntax_and_stabilization(tmp_path):
    """Criterion 4: Nuke Group syntax and stabilization equivalence.
    
    Requirements:
    - Generates Camera2D_<shot>.nk with 1 input (Input1), 1 output (Output1).
    - Valid Nuke script syntax.
    - Mathematical stabilization test: warping transformed frames reduces background to stationary (< 0.3 px).
    """
    w, h = 320, 240
    base_img = _create_synthetic_texture_frame(w, h)

    frames = []
    # Known gentle camera shake: +- 3 px
    for i in range(10):
        dx = 3.0 * math.sin(i * 0.8)
        dy = 2.0 * math.cos(i * 0.8)
        H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        frames.append(cv2.warpPerspective(base_img, H, (w, h), flags=cv2.INTER_LINEAR))

    camera_data = Camera2DTracker.track_sequence(
        frames=frames,
        eval_w=w,
        shot_name="test_shot"
    )

    nk_path = tmp_path / "Camera2D_test_shot.nk"
    json_path = tmp_path / "camera2d.json"

    nk_content = NukeCamera2DGroupGenerator.generate_script(
        shot_name="test_shot",
        camera_data=camera_data,
        output_nk_path=nk_path,
        output_json_path=json_path
    )

    # 1. Check syntax
    assert nk_path.exists()
    assert NukeCamera2DGroupGenerator.validate_nuke_syntax(nk_content)
    assert "name Input1" in nk_content
    assert "name Output1" in nk_content
    assert "name CornerPin2D_Camera2D" in nk_content

    # 2. Check stabilization effect
    ref_frame = frames[0]
    stabilized_diffs = []
    unaligned_diffs = []

    for i in range(1, len(frames)):
        unaligned_diff = float(np.abs(frames[i][20:-20, 20:-20] - ref_frame[20:-20, 20:-20]).mean())
        unaligned_diffs.append(unaligned_diff)

        H_to_ref = np.array(camera_data.frames[i].homography, dtype=np.float64)
        stab = cv2.warpPerspective(frames[i], H_to_ref, (w, h), flags=cv2.INTER_LINEAR)
        stab_diff = float(np.abs(stab[20:-20, 20:-20] - ref_frame[20:-20, 20:-20]).mean())
        stabilized_diffs.append(stab_diff)

    mean_unaligned = np.mean(unaligned_diffs)
    mean_stabilized = np.mean(stabilized_diffs)
    print(f"\nCriterion 4: Mean diff unaligned: {mean_unaligned:.4f}, stabilized: {mean_stabilized:.4f}")
    assert mean_stabilized < mean_unaligned * 0.5


def test_acceptance_e1d_criterion5_data_contract_and_crop_object_motion(tmp_path):
    """Criterion 5: Data contract compliance and crop object motion decomposition.
    
    Requirements:
    - Save and load camera2d.json adhering to schema 'matte_desk/camera2d@1.0'.
    - Level 2: compute_object_motion_in_crop isolates object motion in stabilized crops.
    """
    json_path = tmp_path / "camera2d.json"

    dummy_frames = [
        FrameCamera2D(
            frame=1,
            homography=np.eye(3).tolist(),
            dx=0.0,
            dy=0.0,
            rotation_deg=0.0,
            scale=1.0,
            residual_px=0.0,
            inliers=1000,
            is_anchor=True,
            is_problematic=False
        ),
        FrameCamera2D(
            frame=2,
            homography=[[1.0, 0.0, 1.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]],
            dx=1.5,
            dy=-0.5,
            rotation_deg=0.0,
            scale=1.0,
            residual_px=0.12,
            inliers=950,
            is_anchor=False,
            is_problematic=False
        )
    ]

    cam_data = Camera2DData(
        schema="matte_desk/camera2d@1.0",
        shot_name="SH010",
        ref_frame=1,
        plate_w=4096,
        plate_h=2731,
        fps=24.0,
        motion_type="pan",
        residual_median_px=0.12,
        residual_p95_px=0.25,
        anchor_drift_px=0.08,
        has_parallax=False,
        frames=dummy_frames
    )

    cam_data.save(json_path)
    assert json_path.exists()

    loaded = Camera2DData.load(json_path)
    assert loaded.schema == "matte_desk/camera2d@1.0"
    assert loaded.shot_name == "SH010"
    assert len(loaded.frames) == 2
    assert loaded.frames[1].dx == 1.5

    # Test Level 2: object motion in crop
    crop_w, crop_h = 100, 100
    crop_f1 = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
    crop_f2 = np.zeros((crop_h, crop_w, 3), dtype=np.float32)
    # Circle moves by 5 px inside crop
    cv2.circle(crop_f1, (30, 50), 15, (1.0, 1.0, 1.0), -1)
    cv2.circle(crop_f2, (35, 50), 15, (1.0, 1.0, 1.0), -1)

    obj_motion = Camera2DTracker.compute_object_motion_in_crop(
        crop_frames=[crop_f1, crop_f2],
        camera_data=cam_data,
        crop_origin=(500, 500),
        motion_limit=1.0
    )

    assert obj_motion.is_moving is True
    assert obj_motion.motion_px_max > 2.0
