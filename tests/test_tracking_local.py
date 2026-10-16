# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit fix 3: local alignment (SPEC 1b) – anchors + point tracks carried in time, honest engine name."""

import numpy as np
import cv2
import pytest

from engine.align import AlignmentPipeline, LocalTemporalAligner
from engine.tracking import CoTracker3Tracker, LKTracker, make_point_tracker


def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
    t = (t - t.min()) / (t.max() - t.min())
    return np.repeat(t[..., None], 3, axis=2)


def _field(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # smooth local deformation, up to ~±3 px (SPEC E1b: "gładkie pole ±3 px")
    ux = 2.5 * np.sin(np.pi * xx / w) * np.cos(0.5 * np.pi * yy / h)
    uy = 2.0 * np.cos(np.pi * yy / h) * np.sin(0.5 * np.pi * xx / w)
    return ux, uy


def _sequence(T=9, h=320, w=480, pan=1.0):
    big = _texture(h + 40, w + 40 + int(pan * T) + 2)
    ux, uy = _field(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    beauty, aux = [], []
    for t in range(T):
        ox = 20 + pan * t
        b = cv2.remap(big, xx + ox, yy + 20, cv2.INTER_CUBIC)
        # aux content is displaced by +u relative to beauty: aux(x) = beauty(x - u(x))
        a = cv2.remap(big, xx + ox - ux, yy + 20 - uy, cv2.INTER_CUBIC)
        beauty.append(b)
        aux.append(a)
    return beauty, aux, ux, uy


def test_lk_tracker_follows_translation():
    tex = _texture(200, 300)
    frames = [np.roll(tex, k, axis=1) for k in range(5)]
    q = np.array([[0, 100.0, 80.0], [2, 150.0, 120.0]], np.float32)
    res = LKTracker().track(frames, q)
    assert res.engine == "lk_fallback"
    assert np.allclose(res.tracks[4, 0], [104, 80], atol=0.1)
    assert np.allclose(res.tracks[0, 1], [148, 120], atol=0.1)
    assert res.visible.all()


def test_cotracker_reports_unavailable_without_weights(tmp_path):
    ok, why = CoTracker3Tracker.available(tmp_path / "missing.pth")
    assert not ok and "CoTracker3" in why
    # factory falls back to LK and says so (engine name is the truth)
    assert make_point_tracker(tmp_path / "missing.pth").engine == "lk_fallback"


def test_local_temporal_aligner_recovers_field_through_camera_move():
    T, h, w = 9, 320, 480
    beauty, aux, ux, uy = _sequence(T, h, w)
    al = LocalTemporalAligner(w, h, grid_spacing=48, anchor_every=8, work_scale=1.0, tracker=LKTracker())
    fields = al.measure(beauty, aux)
    assert al.engine == "lk_fallback"
    errs = []
    for t in (0, 4, 8):  # anchors and the frame between them
        pts, d = fields[t]
        assert len(pts) > 20
        inner = (pts[:, 0] > 40) & (pts[:, 0] < w - 40) & (pts[:, 1] > 40) & (pts[:, 1] < h - 40)
        xi, yi = np.clip(pts[inner].round().astype(int), 0, [w - 1, h - 1]).T
        truth = -np.stack([ux[yi, xi], uy[yi, xi]], axis=1)  # d = beauty point - aux point = -u
        errs.append(np.linalg.norm(d[inner] - truth, axis=1))
    e = np.concatenate(errs)
    # SPEC 1b gate: < 0.3 px on >= 90 % of control points
    assert np.mean(e < 0.3) >= 0.9, (np.median(e), np.percentile(e, 90))


def test_measure_sequence_runs_local_model_and_records_engine(tmp_path):
    T, h, w = 5, 256, 384
    beauty, aux, _, _ = _sequence(T, h, w, pan=0.5)
    for i in range(T):
        cv2.imwrite(str(tmp_path / f"b_{i + 1:04d}.exr"), beauty[i].astype(np.float32))
        cv2.imwrite(str(tmp_path / f"a_{i + 1:04d}.exr"), aux[i].astype(np.float32))

    class NamedLK(LKTracker):
        engine = "mock_tracker"

    rep = AlignmentPipeline.measure_sequence(
        beauty_pattern=str(tmp_path / "b_####.exr"), aux_pattern=str(tmp_path / "a_####.exr"),
        first_frame=1, last_frame=T, plate_name="aux", aux_plate_type="cleanplate",
        plate_w=w, plate_h=h, grid_dir=tmp_path / "grid", tracker=NamedLK())
    assert rep.local_tracker_engine == "mock_tracker"
    assert [f.method for f in rep.frames] == ["local_stmap"] * T, [(f.method, f.status) for f in rep.frames]
    assert (tmp_path / "grid" / "grid_0003.exr").exists()


def test_anchor_segments_match_full_tracking_lk():
    """Each anchor's points are tracked only over [previous anchor, next anchor] (GPU memory); with LK the
    frames inside a segment are the same chain as tracking the whole sequence, so the fields must be identical."""
    T, h, w = 13, 256, 384
    beauty, aux, _, _ = _sequence(T, h, w, pan=0.7)
    calls = []

    class SpyLK(LKTracker):
        def track(self, frames, queries):
            calls.append(len(frames))
            return super().track(frames, queries)

    class FullTracking(LocalTemporalAligner):
        def _track_segments(self, anchors, b_s, a_s, q_b, q_a, owner, cancel_check):
            return self.tracker.track(b_s, q_b), self.tracker.track(a_s, q_a)

    seg = LocalTemporalAligner(w, h, grid_spacing=48, anchor_every=4, work_scale=1.0, tracker=SpyLK())
    full = FullTracking(w, h, grid_spacing=48, anchor_every=4, work_scale=1.0, tracker=LKTracker())
    fs, ff = seg.measure(beauty, aux), full.measure(beauty, aux)
    assert seg.anchors(T) == [0, 4, 8, 12]
    assert sorted(calls) == [5, 5, 5, 5, 9, 9, 9, 9]   # [prev anchor, next anchor], never all T frames
    assert len(calls) == 8
    for t in range(T):
        assert len(fs[t][0]) == len(ff[t][0]) > 0
        assert np.array_equal(fs[t][0], ff[t][0]) and np.array_equal(fs[t][1], ff[t][1]), t
