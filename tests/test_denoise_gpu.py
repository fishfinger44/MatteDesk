# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The plate denoise of stage 0 on the GPU (engine/denoise_gpu.py + `denoise_mc` of engine/align_gpu_worker.py,
user decision 2026-11-10).

Everywhere (MD_ALIGN_GPU_FAKE=1, the worker in this .venv's Python): the worker's numpy twin of the CUDA code (the same
function `denoise_window`, numpy arrays) must equal the CPU code of engine/clean.py within float rounding (< 1e-6
relative) – frames and the noise residual (4c.2) – through the real protocol (shared memory, float16 packing, frames
kept between requests); fallback to the CPU (auto) / error (gpu); no shared memory left behind.
Station only (`real`): the CUDA worker against the CPU code, same tolerance on synthetic frames."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import align_gpu
from engine.clean import DenoiseEngine, PlateCleaner
from engine.denoise_gpu import DenoiseStage, GpuDenoiser, denoise_device

sys.path.insert(0, str(Path(__file__).parent))
from test_clean_stream import _frames  # noqa: E402

FAKE_ENV = {**os.environ, "MD_ALIGN_GPU_FAKE": "1"}
REL_TOL = 1e-6        # max |gpu - cpu| / max |frame| (float32 rounding; measured 2.4e-7)


def _rel(a, b, scale):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) / scale


def _run_gpu(frames, python=sys.executable, env=FAKE_ENV):
    g = align_gpu.GpuAligner(python, env, timeout_s=120)
    try:
        den = GpuDenoiser(g, frames)
        try:
            return [den(i) for i in range(len(frames))], den
        finally:
            den.close()
    finally:
        g.close()


@pytest.mark.parametrize("scale", [1.0, 255.0])       # EXR range and the 8-bit PNG range (float16 transfer exact)
def test_numpy_twin_equals_cpu(scale):
    frames = [f * scale for f in _frames(n=7, w=150, h=90)]
    ref_d, ref_r = DenoiseEngine.denoise_temporal_mc(frames)
    out, den = _run_gpu(frames)
    for (d, r), a, b in zip(out, ref_d, ref_r):
        assert d.dtype == np.float32 and d.shape == a.shape
        assert _rel(d, a, scale) < REL_TOL
        assert _rel(r, b, scale) < REL_TOL
    assert den.calls == len(frames) and not den._up and den._flow_shm is None


def test_frames_sent_once_and_kept_on_the_worker():
    frames = _frames(n=6, w=40, h=24)
    g = align_gpu.GpuAligner(sys.executable, FAKE_ENV, timeout_s=60)
    loaded = []
    orig = g.call

    def spy(req, *a, **k):
        r = orig(req, *a, **k)
        loaded.append((r["loaded"], r["cached"]))
        return r

    g.call = spy
    try:
        den = GpuDenoiser(g, frames)
        for i in range(len(frames)):
            den(i)
        den.close()
    finally:
        g.close()
    assert sum(n for n, _ in loaded) == len(frames)          # every frame uploaded once
    assert max(c for _, c in loaded) <= 5                    # window +-2 only


def test_worker_restart_reloads_window():
    """A worker killed between frames (another work took the card) is restarted; frames come from shared memory again."""
    frames = _frames(n=5, w=40, h=24)
    ref_d, _ = DenoiseEngine.denoise_temporal_mc(frames)
    g = align_gpu.GpuAligner(sys.executable, FAKE_ENV, timeout_s=60)
    try:
        den = GpuDenoiser(g, frames)
        a = den(0)[0]
        g.kill()
        b = den(1)[0]
        den.close()
    finally:
        g.close()
    assert _rel(a, ref_d[0], 1.0) < REL_TOL and _rel(b, ref_d[1], 1.0) < REL_TOL


def test_device_setting(monkeypatch):
    assert denoise_device("cpu") == ("cpu", "")
    monkeypatch.setenv("MD_DENOISE_DEVICE", "cpu")
    assert denoise_device() == ("cpu", "")
    st = DenoiseStage(None, cpu_workers=3)
    assert st.device == "cpu" and st.workers == 3
    st = DenoiseStage("gpu", python=sys.executable, env=FAKE_ENV)
    assert st.device == "gpu" and st.workers == 1           # GPU calls stay on the lease owner's thread
    with pytest.raises(align_gpu.GpuWorkerError):
        DenoiseStage("gpu", python=str(Path(sys.executable).with_name("missing_python.exe")))


def _clean(frames, **kw):
    return PlateCleaner.process_sequence(frames, enable_stage1=False, enable_stage2=False, enable_stage4=False,
                                         enable_denoise=True, **kw)


def test_plate_cleaner_gpu_equals_cpu(monkeypatch):
    monkeypatch.setattr(align_gpu, "worker_python", lambda: sys.executable)
    monkeypatch.setenv("MD_ALIGN_GPU_FAKE", "1")
    frames = _frames(n=8, w=96, h=64)
    c_frames, _g, c_rep, c_res = _clean(frames, denoise_device="cpu")
    g_frames, _g, g_rep, g_res = _clean(frames, denoise_device="gpu")
    assert c_rep.denoise_device == "cpu" and g_rep.denoise_device == "gpu"
    for a, b in zip(c_frames, g_frames):
        assert _rel(a, b, 1.0) < REL_TOL
    for a, b in zip(c_res, g_res):
        assert _rel(a, b, 1.0) < REL_TOL
    assert abs(c_rep.noise_reduction_pct - g_rep.noise_reduction_pct) < 1e-3
    assert g_rep.to_dict()["denoise_device"] == "gpu"


def test_auto_falls_back_to_cpu_when_the_worker_dies(monkeypatch):
    monkeypatch.setattr(align_gpu, "worker_python", lambda: sys.executable)
    monkeypatch.setenv("MD_ALIGN_GPU_FAKE", "1")
    monkeypatch.setenv("MD_ALIGN_GPU_FAKE_CRASH", "1")
    monkeypatch.setattr(align_gpu, "comfy_busy", lambda *a, **k: False)
    frames = _frames(n=6, w=64, h=40)
    ref = DenoiseEngine.denoise_temporal_mc(frames)[0]
    out, _g, rep, _r = _clean(frames, denoise_device="auto")
    assert rep.denoise_device == "cpu" and "GPU" in rep.denoise_note
    for a, b in zip(out, ref):
        assert np.array_equal(a, b)                           # the CPU code itself
    with pytest.raises(align_gpu.GpuWorkerError):
        _clean(frames, denoise_device="gpu")


# --------------------------------------------------------------------------------------------- real GPU

@pytest.fixture(scope="module")
def gpu_worker():
    ok, info = align_gpu.check_worker()
    if not (ok and info.startswith("cuda")):
        pytest.skip("ComfyUI's Python with CUDA not available")


@pytest.mark.real
@pytest.mark.usefixtures("gpu_worker")
@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_cuda_equals_cpu(scale):
    frames = [f * scale for f in _frames(n=7, w=640, h=360, seed=5)]
    ref_d, ref_r = DenoiseEngine.denoise_temporal_mc(frames)
    out, _den = _run_gpu(frames, python=align_gpu.worker_python(), env=None)
    for (d, r), a, b in zip(out, ref_d, ref_r):
        assert _rel(d, a, scale) < REL_TOL
        assert _rel(r, b, scale) < REL_TOL
