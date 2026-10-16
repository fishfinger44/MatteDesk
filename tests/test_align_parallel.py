# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Alignment on all cores (engine.procpool) + full-res refinement of tracked local offsets."""

import os

import cv2
import numpy as np
import pytest

from engine.align import (AlignmentFeatureExtractor, AlignmentPipeline, GlobalAligner, phase_correlate,
                          refine_local_offsets)
from engine.procpool import run_frames
from engine.tracking import LKTracker, track_many
from tests.test_tracking_local import _field, _sequence, _texture


def _square(x):
    return x * x


def _write_seq(tmp_path, T=5, h=256, w=384):
    beauty, aux, _, _ = _sequence(T, h, w, pan=0.5)
    for i in range(T):
        cv2.imwrite(str(tmp_path / f"b_{i + 1:04d}.exr"), beauty[i].astype(np.float32))
        cv2.imwrite(str(tmp_path / f"a_{i + 1:04d}.exr"), aux[i].astype(np.float32))
    return T, h, w


def test_run_frames_parallel_keeps_order_and_reports_progress():
    seen = []
    out = run_frames(_square, list(range(7)), workers=3, on_done=lambda n, i: seen.append((n, i)))
    assert out == [x * x for x in range(7)]
    assert sorted(n for n, _ in seen) == list(range(1, 8)) and sorted(i for _, i in seen) == list(range(7))


def test_run_frames_env_restored_and_cancel_propagates():
    before = os.environ.get("OPENBLAS_NUM_THREADS")

    class Stop(RuntimeError):
        pass

    def on_done(n, i):
        raise Stop()

    with pytest.raises(Stop):
        run_frames(_square, list(range(20)), workers=2, on_done=on_done)
    assert os.environ.get("OPENBLAS_NUM_THREADS") == before


def test_measure_sequence_parallel_equals_sequential(tmp_path):
    T, h, w = _write_seq(tmp_path)
    kw = dict(beauty_pattern=str(tmp_path / "b_####.exr"), aux_pattern=str(tmp_path / "a_####.exr"),
              first_frame=1, last_frame=T, plate_name="aux", aux_plate_type="cleanplate", plate_w=w, plate_h=h,
              tracker=LKTracker())
    seq = AlignmentPipeline.measure_sequence(grid_dir=tmp_path / "s1", workers=1, **kw)
    par = AlignmentPipeline.measure_sequence(grid_dir=tmp_path / "s2", workers=3, **kw)
    assert [f.method for f in seq.frames] == ["local_stmap"] * T
    for a, b in zip(seq.frames, par.frames):
        assert (a.frame, a.method, a.status, a.grid_rel_path) == (b.frame, b.method, b.status, b.grid_rel_path)
        assert a.residual_px == b.residual_px and a.residual_max_px == b.residual_max_px
        assert (a.dx, a.dy) == (b.dx, b.dy)
    m1 = cv2.imread(str(tmp_path / "s1" / "grid_0003.exr"), cv2.IMREAD_UNCHANGED)
    m2 = cv2.imread(str(tmp_path / "s2" / "grid_0003.exr"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(m1, m2)


def test_measure_sequence_cancel_in_local_stage(tmp_path):
    T, h, w = _write_seq(tmp_path, T=4)
    steps = []

    class Stop(RuntimeError):
        pass

    def step_cb(text):
        steps.append(text)
        if text.startswith("Dopasowanie lokalne: śledzenie"):
            raise Stop()

    with pytest.raises(Stop):
        AlignmentPipeline.measure_sequence(
            beauty_pattern=str(tmp_path / "b_####.exr"), aux_pattern=str(tmp_path / "a_####.exr"),
            first_frame=1, last_frame=T, plate_name="aux", aux_plate_type="cleanplate", plate_w=w, plate_h=h,
            grid_dir=tmp_path / "grid", tracker=LKTracker(), workers=1, step_cb=step_cb)
    assert not (tmp_path / "grid" / "grid_0001.exr").exists()


def test_refine_local_offsets_fixes_tracker_error():
    """Tracked offsets off by up to ~2 px (downscaled tracker) are pulled back to the SPEC 1b gate."""
    h, w = 320, 480
    beauty, aux, ux, uy = _sequence(1, h, w)
    f_b = AlignmentFeatureExtractor.extract(beauty[0], mode="cleanplate", scale=1.0)
    f_a = AlignmentFeatureExtractor.extract(aux[0], mode="cleanplate", scale=1.0)
    gy, gx = np.mgrid[72:h - 72:32, 72:w - 72:32]      # inside the LOCAL_PATCH_PX / 2 = 64 px margin
    pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    truth = -np.stack([ux[gy.ravel(), gx.ravel()], uy[gy.ravel(), gx.ravel()]], axis=1)
    rng = np.random.default_rng(1)
    noisy = (truth + rng.uniform(-2.0, 2.0, truth.shape)).astype(np.float32)

    ref, ok, resp = refine_local_offsets(f_b, f_a, pts, noisy)
    assert np.isfinite(resp).all() and (resp[ok] > 0.1).all()
    e_before = np.linalg.norm(noisy - truth, axis=1)
    e_after = np.linalg.norm(ref - truth, axis=1)
    assert ok.mean() > 0.9
    assert np.median(e_before) > 1.0
    assert np.mean(e_after < 0.3) >= 0.9, (np.median(e_after), np.percentile(e_after, 90))


def test_refine_keeps_tracked_value_on_flat_patch():
    flat = np.zeros((200, 200), np.float32)
    pts = np.array([[100.0, 100.0]], np.float32)
    d = np.array([[1.5, -0.5]], np.float32)
    ref, ok, resp = refine_local_offsets(flat, flat, pts, d)
    assert not ok[0] and np.array_equal(ref, d) and resp[0] <= 0.1


def test_track_many_falls_back_to_track():
    tex = _texture(120, 160)
    frames = [np.roll(tex, k, axis=1) for k in range(3)]
    q = np.array([[0, 60.0, 50.0]], np.float32)
    ra, rb = track_many(LKTracker(), [(frames, q), (frames, q)])
    assert np.allclose(ra.tracks, rb.tracks) and np.allclose(ra.tracks[2, 0], [62, 50], atol=0.1)


def test_phase_correlate_leaves_inputs_untouched():
    """OpenCV multiplies both inputs by the window in place; the wrapper must not (regression: windowed feature
    maps fed ECC / residuals, overlapping patches corrupted each other)."""
    rng = np.random.default_rng(0)
    big = rng.random((200, 200)).astype(np.float32)
    a = rng.random((64, 64)).astype(np.float32)
    win = cv2.createHanningWindow((64, 64), cv2.CV_32F)
    big0, a0 = big.copy(), a.copy()
    phase_correlate(a, big[50:114, 50:114], win)
    assert np.array_equal(big, big0) and np.array_equal(a, a0)

    f_b = rng.random((120, 160)).astype(np.float32)
    f_a = np.roll(f_b, 2, axis=1)
    fb0, fa0 = f_b.copy(), f_a.copy()
    GlobalAligner(scale=1.0).measure_translation(f_a, f_b, np.ones_like(f_b))
    assert np.array_equal(f_b, fb0) and np.array_equal(f_a, fa0)
