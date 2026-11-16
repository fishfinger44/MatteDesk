# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""align_v2 step 0 (2026-11-03): the residual of an aligned auxiliary plate measured on control points
(engine/align_residual.py) must measure the TRUTH – checked on a known shift and on the synthetic alignment shot
(tests/synth.make_align_shot: known similarity + object crossing the frame), where the composed alignment's error
against the true field is known pixel by pixel."""

import numpy as np
import pytest

import cv2

from engine import align_grid
from engine.align import AlignmentFrameResult, AlignmentPipeline, compose_remap
from engine.align_residual import (RESIDUAL_PATCH_PX, frame_residual, residual_map, residual_png, residual_points,
                                   residual_stats, zone_stats)
from engine.tracking import LKTracker
from tests.synth import align_texture, make_align_shot


def _feat(img, mode="cleanplate"):
    from engine.align import AlignmentFeatureExtractor
    return AlignmentFeatureExtractor.extract(img, mode=mode, scale=1.0)


def test_residual_points_measure_a_known_shift():
    h, w = 400, 600
    tex = align_texture(h, w, seed=1)
    aux = cv2.warpAffine(tex, np.float32([[1, 0, 1.3], [0, 1, -0.7]]), (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)
    pts, res, resp = residual_points(_feat(tex), _feat(aux))
    assert len(pts) > 40 and (resp > 0.1).all()
    st = residual_stats(res)
    # same sign as the global model (tests/test_align_gpu: an aux moved by +1.3 measures dx = -1.3)
    assert abs(np.median(res[:, 0]) + 1.3) < 0.12 and abs(np.median(res[:, 1]) - 0.7) < 0.12
    assert abs(st["median"] - np.hypot(1.3, 0.7)) < 0.12 and st["p95"] < 1.8
    # the frame's alignment undoes it: residual ~0
    fr = AlignmentFrameResult(frame=1, method="translation", residual_px=0, residual_max_px=0, dx=-1.3, dy=0.7)
    out = frame_residual(tex, aux, fr, None, mode="cleanplate", static=False)
    assert out["stats"]["n"] > 40 and out["stats"]["median"] < 0.08 and out["stats"]["frac_ok"] >= 0.9
    before = frame_residual(tex, aux, None, None, mode="cleanplate", static=False)
    assert before["stats"]["median"] > 1.3


def test_residual_map_and_png():
    pts = np.array([[10, 10], [70, 10], [10, 70], [500, 300]])
    res = np.array([[1.0, 0.0], [0.0, 2.0], [0.3, 0.4], [0.0, 0.0]])
    m = residual_map(pts, res, 600, 400, step=48)
    assert m.shape == (9, 13) and m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(2.0)
    assert np.isnan(m[5, 5]) and m[6, 10] == pytest.approx(0.0)
    png = residual_png(m, out_w=260, vmax=1.0)
    assert png.shape == (180, 260, 4) and png.dtype == np.uint8
    assert png[100, 100, 3] == 0                      # no measurement = transparent
    assert png[5, 5, 3] > 0 and not np.array_equal(png[30, 5, :3], png[5, 5, :3])   # 0.5 px and 1 px differ in colour
    zs = zone_stats(pts, res, np.zeros((400, 600), bool))
    assert zs["in"]["n"] == 0 and zs["out"]["n"] == 4


def test_residual_matches_truth_on_the_synthetic_shot(tmp_path):
    """The estimator against the known field: per control point the measured residual length equals the composed
    map's error against the truth (within the sub-pixel noise of the patch correlation)."""
    shot = make_align_shot(tmp_path / "shot", n=5, w=640, h=360, shift=(2.6, -1.1), rot_deg=0.2)
    rep = AlignmentPipeline.measure_sequence(beauty_pattern=shot.beauty_pattern, aux_pattern=shot.aux_pattern,
                                             first_frame=shot.first, last_frame=shot.last, plate_name="cleanplate",
                                             aux_plate_type="cleanplate", plate_w=shot.w, plate_h=shot.h,
                                             grid_dir=tmp_path / "grid", tracker=LKTracker(), workers=1, device="cpu")
    f = rep.frames[0]
    assert f.method in ("similarity", "affine", "homography", "local_stmap", "translation")
    beauty = cv2.imread(shot.beauty_pattern.replace("####", f"{f.frame:04d}"), cv2.IMREAD_UNCHANGED)
    aux = cv2.imread(shot.aux_pattern.replace("####", f"{f.frame:04d}"), cv2.IMREAD_UNCHANGED)
    grid = align_grid.read_grid(tmp_path / "grid" / f.grid_rel_path) if f.grid_rel_path else None
    out = frame_residual(beauty, aux, f, grid, mode="cleanplate", exclude=shot.masks[f.frame])
    assert out["stats"]["n"] >= 30, out["stats"]
    mx, my = compose_remap(f, grid, shot.w, shot.h, plate_w=shot.w)
    ux, uy = shot.truth(f.frame)
    Y, X = np.indices((shot.h, shot.w), dtype=np.float32)
    ex, ey = mx - X - ux, my - Y - uy
    p = out["pts"]
    true_len = np.hypot(ex[p[:, 1], p[:, 0]], ey[p[:, 1], p[:, 0]])
    meas_len = np.hypot(out["res"][:, 0], out["res"][:, 1])
    assert np.median(np.abs(meas_len - true_len)) < 0.15, (np.median(meas_len), np.median(true_len))
    # the raw plate ("before") carries the whole shift; the measured median is the known one
    before = frame_residual(beauty, aux, None, None, mode="cleanplate", exclude=shot.masks[f.frame])
    assert abs(before["stats"]["median"] - np.median(np.hypot(ux, uy))) < 0.3
    assert before["stats"]["median"] > out["stats"]["median"]
    assert RESIDUAL_PATCH_PX == 64
