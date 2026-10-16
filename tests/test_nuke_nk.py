# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit fix 5: .nk generators – valid XY-knob syntax and geometry equal to the engine's warps."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.align import AlignmentFrameResult, AlignmentReport, NukeAlignGroupGenerator
from engine.camera import Camera2DData, Camera2DTracker, FrameCamera2D, NukeCamera2DGroupGenerator
from engine.nuke_nk import (corner_pin_knobs, cv_to_nuke, matrix_cv_to_nuke, similarity_to_transform_knobs,
                            validate_nk, xy_knob)


def test_xy_knob_syntax():
    k = xy_knob("translate", [1, 2], [0.5, 1.5], [3, 4])
    assert k == "translate {{curve x1 0.5000 x2 1.5000} {curve x1 3.0000 x2 4.0000}}"
    bad = "Group {\n name G\n Input {\n  name Input1\n }\n T {\n  translate {{curve x1 1}} {{curve x1 2}}\n }\n Output {\n  name Output1\n }\nend_group\n"
    assert any("XY knob" in p for p in validate_nk(bad))


def test_cv_to_nuke_pixel_centres():
    f = cv_to_nuke(100)
    # OpenCV pixel (0, 0) centre -> Nuke (0.5, 99.5)
    assert np.allclose(f @ [0, 0, 1], [0.5, 99.5, 1])


def _nuke_transform(p, t, rot_deg, scale, c):
    th = math.radians(rot_deg)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * scale
    return np.asarray(c) + r @ (np.asarray(p) - c) + np.asarray(t)


def test_similarity_knobs_reproduce_cv_warp():
    w, h = 640, 480
    th, s = math.radians(2.0), 1.01
    m_cv = np.array([[s * math.cos(th), -s * math.sin(th), 3.2], [s * math.sin(th), s * math.cos(th), -1.7], [0, 0, 1]])
    m_n = matrix_cv_to_nuke(m_cv, h)
    tx, ty, rot, sc = similarity_to_transform_knobs(m_n, (w / 2, h / 2))
    f = cv_to_nuke(h)
    for p_cv in [(0, 0), (639, 0), (320, 240), (17, 400)]:
        q_cv = m_cv @ [*p_cv, 1]
        expected = (f @ q_cv)[:2]
        got = _nuke_transform((f @ [*p_cv, 1])[:2], (tx, ty), rot, sc, (w / 2, h / 2))
        assert np.allclose(got, expected, atol=1e-6)


def test_corner_pin_equals_homography():
    w, h = 800, 600
    H = np.array([[1.01, 0.02, 5], [-0.01, 0.99, -3], [1e-5, -2e-5, 1]])
    m_n = matrix_cv_to_nuke(H, h)
    lines = corner_pin_knobs([1], [m_n], w, h, nd=6)
    to = [list(map(float, ln.split("{")[1].split("}")[0].split())) for ln in lines[:4]]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    Hn = cv2.getPerspectiveTransform(src, np.float32(to))
    assert np.allclose(Hn / Hn[2, 2], m_n / m_n[2, 2], atol=1e-6)


def _frames(method, n=4):
    out = []
    for i in range(n):
        th = math.radians(0.3 * i)
        m = [[math.cos(th), -math.sin(th), 1.5 * i], [math.sin(th), math.cos(th), -0.5 * i]]
        if method == "homography":
            m = m + [[1e-6 * i, 0, 1]]
        out.append(AlignmentFrameResult(frame=1001 + i, method=method, residual_px=0.1, residual_max_px=0.2,
                                        dx=1.5 * i, dy=-0.5 * i, transform_matrix=m if method != "translation" else None))
    return out


@pytest.mark.parametrize("method", ["translation", "similarity", "affine", "homography", "local_stmap"])
def test_align_group_valid_for_every_method(tmp_path, method):
    fr = _frames(method if method != "local_stmap" else "translation")
    for f in fr:
        f.method = method if method != "local_stmap" else "local_stmap"
    rep = AlignmentReport("x2", "backlit", "beauty", (1001, 1004), method, 0.1, 0.2, True, "", fr)
    text = NukeAlignGroupGenerator.generate_script("x2", rep, 1280, 720, tmp_path / "a.nk", tmp_path / "a.json")
    assert validate_nk(text, "Align_x2") == []
    assert "4096" not in text
    # Switch: input 0 = original (Dot1 pushed last = top of stack), input 1 = aligned
    assert text.index("push $N_dot1\n Switch") > 0


def test_camera_group_valid(tmp_path):
    frames = []
    for i in range(3):
        H = np.array([[1, 0, 2.0 * i], [0, 1, -1.0 * i], [0, 0, 1]])
        frames.append(FrameCamera2D(frame=1 + i, homography=H.tolist(), dx=2.0 * i, dy=-1.0 * i, rotation_deg=0.0,
                                    scale=1.0, residual_px=0.1, inliers=100, is_anchor=i == 0))
    cam = Camera2DData(schema="matte_desk/camera2d@1.0", shot_name="S", ref_frame=1, plate_w=1280, plate_h=720,
                       fps=24.0, motion_type="pan", residual_median_px=0.1, residual_p95_px=0.2,
                       anchor_drift_px=0.0, has_parallax=False, frames=frames)
    text = NukeCamera2DGroupGenerator.generate_script("S", cam, tmp_path / "c.nk", tmp_path / "c.json")
    assert validate_nk(text, "Camera2D_S") == []
    assert NukeCamera2DGroupGenerator.validate_nuke_syntax(text)
    assert "to1 {{curve x1 0.000 x2 2.000 x3 4.000} {curve x1 0.000 x2 1.000 x3 2.000}}" in text

    # homography maps frame -> ref, so the CornerPin (from = corners, to = H(corners)) stabilizes as is;
    # Match Move = inverse
    assert "invert {{parent.mode == 1}}" in text


def test_camera_homography_maps_frame_to_ref():
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur((rng.random((300, 420)) * 255).astype(np.uint8), (0, 0), 1.5)
    ref, cur = tex[20:260, 20:380].copy(), tex[20:260, 25:385].copy()  # camera pans: content moves -5 px
    cam = Camera2DTracker.track_sequence([ref, cur], frame_numbers=[1, 2], ref_frame=1, eval_w=360)
    H = np.array(cam.frames[1].homography)
    back = cv2.warpPerspective(cur, H, (360, 240))
    assert np.abs(back[40:200, 40:300].astype(float) - ref[40:200, 40:300]).mean() < 3.0
