# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate alignment on the GPU (engine.align_gpu + engine/align_gpu_worker.py in ComfyUI's Python).

Protocol, retry, cancel and CPU fallback run everywhere with MD_ALIGN_GPU_FAKE=1 (worker in this .venv's Python, no
torch). GPU vs CPU tests need ComfyUI's Python with CUDA (skipped otherwise, e.g. in the cloud): the GPU result must
match the CPU code within 0.02 px and pass the SPEC 1b gate (< 0.3 px on >= 90 % of the points)."""

import os
import sys
import time

import cv2
import numpy as np
import pytest

from engine import align_gpu, align_grid
from engine.align import AlignmentPipeline
from engine.procpool import imap_frames
from engine.tracking import LKTracker
from tests.test_align_compose import composed_error, shifted_sequence, write_sequence
from tests.test_tracking_local import _sequence, _texture

FAKE_ENV = {**os.environ, "MD_ALIGN_GPU_FAKE": "1"}
FAKE_CRASH_ENV = {**FAKE_ENV, "MD_ALIGN_GPU_FAKE_CRASH": "1"}


def _gpu_ready():
    ok, info = align_gpu.check_worker()
    return ok and info.startswith("cuda")


needs_gpu = pytest.mark.skipif(not _gpu_ready(), reason="ComfyUI's Python with CUDA not available")


def _square(x):
    return x * x


def _fake(tmp_path):
    return align_gpu.GpuAligner(sys.executable, FAKE_ENV, timeout_s=60)


def _npy(tmp_path, name, arr):
    p = tmp_path / f"{name}.npy"
    np.save(p, arr)
    return str(p)


def _write(tmp_path, beauty, aux):
    for i, (b, a) in enumerate(zip(beauty, aux)):
        cv2.imwrite(str(tmp_path / f"b_{i + 1:04d}.exr"), b.astype(np.float32))
        cv2.imwrite(str(tmp_path / f"a_{i + 1:04d}.exr"), a.astype(np.float32))
    h, w = beauty[0].shape[:2]
    return dict(beauty_pattern=str(tmp_path / "b_####.exr"), aux_pattern=str(tmp_path / "a_####.exr"),
                first_frame=1, last_frame=len(beauty), plate_name="aux", plate_w=w, plate_h=h, workers=1)


# ------------------------------------------------------------------------------------------- protocol (fake)

def test_imap_frames_keeps_order():
    assert [r for _, r in imap_frames(_square, list(range(9)), workers=3, window=2)] == [x * x for x in range(9)]
    assert list(imap_frames(_square, [2, 3], workers=1)) == [(0, 4), (1, 9)]


def test_fake_worker_protocol(tmp_path):
    g = _fake(tmp_path)
    try:
        z = np.zeros((8, 12), np.float32)
        r = g.call({"op": "global", "lf_b": _npy(tmp_path, "b", z), "lf_a": _npy(tmp_path, "a", z)})
        assert r["method"] == "translation" and g.device == "fake"
        pid = g._proc.pid
        out = str(tmp_path / "p.npy")
        g.call({"op": "patches", "xy": _npy(tmp_path, "xy", np.zeros((5, 2), np.int64)), "out": out})
        assert np.load(out).shape == (5, 5)
        out = str(tmp_path / "f.npy")
        r = g.call({"op": "local_frame", "plate_w": 12, "plate_h": 8, "out": out})
        assert np.load(out).shape == (2, 8, 12) and r["ho_frac_ok"] == 1.0
        assert g._proc.pid == pid                       # one long-lived worker
        with pytest.raises(align_gpu.GpuWorkerError, match="unknown op"):
            g.call({"op": "nope"})
    finally:
        g.close()
    assert g._proc is None and not g.tmp.exists()


def test_fake_worker_crash_is_retried_once(tmp_path):
    g = _fake(tmp_path)
    try:
        r = g.call({"op": "release", "fake_crash_once": str(tmp_path / "crashed")})
        assert r["ok"] and g.restarts == 1
        with pytest.raises(align_gpu.GpuWorkerError, match="dwa razy"):
            g.call({"op": "release", "fake_crash": True})
    finally:
        g.close()


def test_cancel_kills_busy_worker(tmp_path):
    g = _fake(tmp_path)

    class Stop(RuntimeError):
        pass

    t0 = time.time()

    def cancel():
        if time.time() - t0 > 0.5:
            raise Stop()

    try:
        g.call({"op": "release"})                       # worker up
        t0 = time.time()
        with pytest.raises(Stop):
            g.call({"op": "release", "fake_sleep": 30}, cancel_check=cancel)
        assert time.time() - t0 < 5 and g._proc is None
    finally:
        g.close()


def test_resolve_device(tmp_path):
    assert align_gpu.resolve_device("cpu") == ("cpu", "")
    missing = str(tmp_path / "python.exe")
    with pytest.raises(align_gpu.GpuWorkerError, match="GPU"):
        align_gpu.resolve_device("gpu", missing)
    dev, note = align_gpu.resolve_device("auto", missing)
    assert dev == "cpu" and "GPU niedostępne" in note
    assert align_gpu.resolve_device("auto", sys.executable, FAKE_ENV, busy=lambda: False)[0] == "gpu"
    dev, note = align_gpu.resolve_device("auto", sys.executable, FAKE_ENV, busy=lambda: True)
    assert dev == "cpu" and "ComfyUI liczy" in note


def test_measure_sequence_fallback_and_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(align_gpu, "comfy_busy", lambda *a, **k: False)
    beauty, aux, _, _ = _sequence(3, 128, 192, pan=0.5)
    kw = _write(tmp_path, beauty, aux)
    cpu = AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="cpu", **kw)
    assert cpu.device == "cpu" and cpu.device_note == ""
    rep = AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="auto", gpu_python=sys.executable,
                                             gpu_env=FAKE_CRASH_ENV, **kw)
    assert rep.device == "cpu" and "GPU zawiodło" in rep.device_note
    assert [(f.method, f.dx, f.dy, f.residual_px) for f in rep.frames] == \
           [(f.method, f.dx, f.dy, f.residual_px) for f in cpu.frames]
    assert not align_gpu._OPEN
    with pytest.raises(align_gpu.GpuWorkerError):
        AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="gpu", gpu_python=sys.executable,
                                           gpu_env=FAKE_CRASH_ENV, **kw)
    assert not align_gpu._OPEN
    rep = AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="gpu", gpu_python=sys.executable,
                                             gpu_env=FAKE_ENV, **kw)
    assert rep.device == "fake" and rep.frames[0].method == "translation"


# --------------------------------------------------------------------------------------------- real GPU

def _shifted(T, h, w, matrix):
    """beauty = static texture, aux(x) = beauty(A^-1 x): aux content moved by the forward map `matrix` (2x3)."""
    tex = _texture(h, w, seed=3)
    aux = cv2.warpAffine(tex, np.float32(matrix), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return [tex] * T, [aux] * T


def _corner_diff(a, b, w, h):
    def mat(f):
        m = np.eye(3)
        if f.transform_matrix is not None:
            t = np.asarray(f.transform_matrix)
            m[:t.shape[0]] = t
        else:
            m[0, 2], m[1, 2] = f.dx, f.dy
        return m
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    pa, pb = mat(a) @ c, mat(b) @ c
    return float(np.abs(pa[:2] / pa[2] - pb[:2] / pb[2]).max())


@needs_gpu
@pytest.mark.parametrize("mode", ["backlit", "cleanplate"])
def test_gpu_matches_cpu_known_shift(tmp_path, mode):
    beauty, aux = _shifted(5, 320, 480, [[1, 0, 1.3], [0, 1, -0.7]])
    kw = _write(tmp_path, beauty, aux)
    cpu = AlignmentPipeline.measure_sequence(aux_plate_type=mode, device="cpu", **kw)
    gpu = AlignmentPipeline.measure_sequence(aux_plate_type=mode, device="gpu", **kw)
    assert gpu.device == "cuda"
    for a, b in zip(cpu.frames, gpu.frames):
        assert a.method == b.method                     # the hierarchy may stop at translation or similarity
        assert _corner_diff(a, b, 480, 320) < 0.02
        if mode == "cleanplate":    # correction aux -> beauty undoes the shift
            # (backlit on this smooth texture: the CPU code itself lands ~0.4 px off with residual 0.26 px < 0.3 –
            # an issue of the method, not of the GPU port; reported in LOCAL_CHECKS)
            assert abs(b.dx + 1.3) < 0.1 and abs(b.dy - 0.7) < 0.1
        assert abs(a.residual_px - b.residual_px) < 0.02


@needs_gpu
def test_gpu_matches_cpu_rotation(tmp_path):
    h, w = 320, 480
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.4, 1.0)
    M[0, 2] += 0.8
    beauty, aux = _shifted(3, h, w, M)
    kw = _write(tmp_path, beauty, aux)
    cpu = AlignmentPipeline.measure_sequence(aux_plate_type="backlit", device="cpu", **kw)
    gpu = AlignmentPipeline.measure_sequence(aux_plate_type="backlit", device="gpu", **kw)
    for a, b in zip(cpu.frames, gpu.frames):
        assert a.method == b.method and a.method in ("similarity", "affine", "homography")
        assert _corner_diff(a, b, w, h) < 0.02
        assert abs(a.residual_px - b.residual_px) < 0.02


@needs_gpu
def test_gpu_matches_cpu_local_field_and_spec_gate(tmp_path):
    T, h, w = 5, 320, 480
    beauty, aux, ux, uy = _sequence(T, h, w, pan=0.5)
    kw = _write(tmp_path, beauty, aux)
    cpu = AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="cpu", tracker=LKTracker(),
                                             grid_dir=tmp_path / "s_cpu", **kw)
    gpu = AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="gpu", tracker=LKTracker(),
                                             grid_dir=tmp_path / "s_gpu", **kw)
    assert gpu.device == "cuda" and [f.method for f in gpu.frames] == ["local_stmap"] * T
    for a, b in zip(cpu.frames, gpu.frames):
        assert (a.method, a.status) == (b.method, b.status) == ("local_stmap", "ok")
        assert abs(a.residual_px - b.residual_px) < 0.02
        g1 = align_grid.read_grid(tmp_path / "s_cpu" / a.grid_rel_path)
        g2 = align_grid.read_grid(tmp_path / "s_gpu" / b.grid_rel_path)
        assert g1.shape == g2.shape == align_grid.grid_shape(w, h)
        assert np.abs(g1.dx - g2.dx).max() < 0.02 and np.abs(g1.dy - g2.dy).max() < 0.02
        assert a.global_method == b.global_method and a.n_points == b.n_points and abs(a.n_rejected - b.n_rejected) <= 1
        # the grid is the residual over the global model: the composed map must hit the true field (SPEC 1b gate)
        err = composed_error(b, g2, ux, uy)
        assert np.mean(err < 0.3) >= 0.9, (np.median(err), np.percentile(err, 90))


@needs_gpu
def test_gpu_matches_cpu_global_shift_plus_local_field(tmp_path):
    """15 px global shift + smooth +-3 px field: global model on both devices, grids (the residual) equal < 0.02 px,
    the same outlier counts, the composed alignment passes the SPEC 1b gate on the GPU."""
    T, h, w = 3, 750, 1200
    beauty, aux, ux, uy = shifted_sequence(T, h, w)
    kw = write_sequence(tmp_path, beauty, aux)
    cpu = AlignmentPipeline.measure_sequence(device="cpu", grid_dir=tmp_path / "s_cpu", **kw)
    gpu = AlignmentPipeline.measure_sequence(device="gpu", grid_dir=tmp_path / "s_gpu", **kw)
    assert gpu.device == "cuda" and [f.method for f in gpu.frames] == ["local_stmap"] * T
    for a, b in zip(cpu.frames, gpu.frames):
        assert a.global_method == b.global_method and _corner_diff(a, b, w, h) < 0.02
        assert abs(a.residual_px - b.residual_px) < 0.02 and abs(a.n_rejected - b.n_rejected) <= 1
        g1 = align_grid.read_grid(tmp_path / "s_cpu" / a.grid_rel_path)
        g2 = align_grid.read_grid(tmp_path / "s_gpu" / b.grid_rel_path)
        assert np.abs(g1.dx - g2.dx).max() < 0.02 and np.abs(g1.dy - g2.dy).max() < 0.02
        assert np.abs(g2.dx).max() < 4.5                                   # residual, not the 15 px
        err = composed_error(b, g2, ux, uy)
        assert np.mean(err < 0.3) >= 0.9 and b.status == "ok", (np.median(err), np.percentile(err, 90))


@needs_gpu
def test_gpu_cancel_in_local_stage(tmp_path):
    beauty, aux, _, _ = _sequence(4, 256, 384, pan=0.5)
    kw = _write(tmp_path, beauty, aux)

    class Stop(RuntimeError):
        pass

    def step_cb(text):
        if text.startswith("Dopasowanie lokalne: śledzenie"):
            raise Stop()

    with pytest.raises(Stop):
        AlignmentPipeline.measure_sequence(aux_plate_type="cleanplate", device="gpu", tracker=LKTracker(),
                                           grid_dir=tmp_path / "grid", step_cb=step_cb, **kw)
    assert not align_gpu._OPEN
    assert not (tmp_path / "grid" / "grid_0001.exr").exists()


def test_doctor_align_gpu_check(monkeypatch):
    from doctor import EnvironmentDoctor
    d = EnvironmentDoctor()
    monkeypatch.setattr(align_gpu, "check_worker", lambda *a, **k: (True, "cuda Test GPU"))
    it = d.check_align_gpu()
    assert it.status == "ok" and "Test GPU" in it.message and it.action is None
    monkeypatch.setattr(align_gpu, "check_worker", lambda *a, **k: (False, "brak torch"))
    it = d.check_align_gpu()
    assert it.status == "warning" and "brak torch" in it.message and "Wymagane teraz: NIE" in it.fix_suggestion
