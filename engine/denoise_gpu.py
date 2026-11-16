# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/denoise_gpu.py - the plate denoise of stage 0 (engine/clean.py DenoiseEngine, SPEC 4c.2) on the GPU
(user decision 2026-11-10).

Where the time went on the CPU (sh060 3840x2160, one frame, 1 thread, 2026-11-10): photometric weight exp(|cur -
warped|) 49 %, weighted accumulation 25 %, remap 7 %, divide 5 %, DIS flow 4 %, the flow's gray 4 %, flow upscale 3 %,
blur 2 %. So the DIS flow (OpenCV, cheap at <= 1024 px) stays here and is bit for bit the CPU's; everything per pixel
at full resolution runs in the GPU worker of the plate alignment (engine/align_gpu_worker.py `denoise_mc`, ComfyUI's
torch): flow upscale (cv2.resize INTER_LINEAR taps), remap (1/32 px, BORDER_REPLICATE), weights, average, blur.

Frames go through shared memory (a 4K float32 frame is ~100 MB: a .npy file round trip cost ~0.2 s, a memcpy ~0.02 s),
as uint8 / float16 when that is exact (8-bit / half sources); each frame is sent once and kept on the GPU while its
window needs it; the slots are reused, so their pages stay mapped in the worker. The host <-> GPU copies bound the
speed: per frame one input (25 / 50 / 100 MB), the flows (22 MB at 4K) and the float32 result (100 MB). The flows of the next frames are computed on a small thread pool while the GPU works on the current one.
GPU = CPU within float rounding (tests/test_denoise_gpu.py; numbers in PROGRESS.md 2026-11-10).

config `denoise_device` = auto (default: GPU when ComfyUI's Python has CUDA and ComfyUI's queue is idle) | gpu | cpu,
env MD_DENOISE_DEVICE (tests: cpu); the CPU code of engine/clean.py stays unchanged as the fallback: in auto a worker
failure finishes the run on the CPU (logged, in the report), in gpu it is an error. Doctor: "Wyrównanie na GPU".
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("matte_desk.denoise_gpu")

PREFETCH = 3          # output frames whose flows / uploads are prepared ahead of the GPU


def denoise_device(requested: Optional[str] = None) -> Tuple[str, str]:
    """('gpu' | 'cpu', note): `requested` > env MD_DENOISE_DEVICE > config `denoise_device` (auto)."""
    from engine import align_gpu
    requested = requested or os.environ.get("MD_DENOISE_DEVICE") or None
    return align_gpu.resolve_device(requested, key="denoise_device")


def _compact(a: np.ndarray) -> np.ndarray:
    """The smallest exact transfer type of a float32 frame: uint8 (8-bit sources read as 0..255), float16 (half EXR),
    else float32. The worker converts back to float32 on the GPU - bit for bit the same values."""
    if a.min() >= 0 and a.max() <= 255:
        u8 = a.astype(np.uint8)
        if np.array_equal(u8, a):
            return u8
    small = a.astype(np.float16)
    return small if np.array_equal(small.astype(np.float32), a, equal_nan=True) else a


class GpuDenoiser:
    """fn(i) -> (denoised, residual) of DenoiseEngine.denoise_frame_fn computed in the GPU worker `gpu`
    (engine.align_gpu.GpuAligner). Called in order of i from ONE thread (the lease owner); `frames` may be a stream
    view (engine.clean._MemoView): frames not there yet are simply not prefetched."""

    def __init__(self, gpu: Any, frames: Sequence[np.ndarray], radius: int = 2, spatial_sigma: float = 1.0,
                 cancel_check: Optional[Callable[[], None]] = None, threads: Optional[int] = None):
        from engine.clean import DenoiseEngine
        self.gpu = gpu
        self.frames = frames
        self.n = len(frames)
        self.radius = int(radius)
        self.sigma = float(spatial_sigma)
        self.cancel_check = cancel_check
        f0 = frames[0]
        if f0.ndim != 3:
            raise ValueError("GPU denoise: H x W x C frames only")
        self.h, self.w, self.c = f0.shape
        self.gray_fn, self.sx, self.sy, self.lw, self.lh = DenoiseEngine.flow_gray(self.w, self.h)
        self.pool = ThreadPoolExecutor(max_workers=threads or max(2, min(8, os.cpu_count() or 2)),
                                       thread_name_prefix="md-denoise")
        self._lock = threading.Lock()
        self._gray: Dict[int, Future] = {}
        self._flows: Dict[int, Future] = {}
        self._up: Dict[int, Future] = {}            # j -> Future[(SharedMemory, dtype name)]
        self._slot_bytes = self.h * self.w * self.c * 4
        self._free: List[shared_memory.SharedMemory] = []   # frame slots to reuse (their pages stay mapped in the worker)
        self._slots: List[shared_memory.SharedMemory] = []
        self._out = shared_memory.SharedMemory(create=True, size=self._slot_bytes)
        self._flow_shm: Optional[shared_memory.SharedMemory] = None
        self.calls = 0

    # ---------------------------------------------------------------- preparation (threads)
    def _window(self, i: int) -> List[int]:
        return list(range(max(0, i - self.radius), min(self.n, i + self.radius + 1)))

    def _gray_of(self, j: int) -> Future:
        with self._lock:
            fut = self._gray.get(j)
            if fut is None:
                img = self.frames[j]                        # KeyError when not streamed yet: nothing scheduled
                fut = self._gray[j] = self.pool.submit(self.gray_fn, img)
        return fut

    def _upload(self, j: int) -> Future:
        with self._lock:
            fut = self._up.get(j)
            if fut is None:
                img = self.frames[j]
                fut = self._up[j] = self.pool.submit(self._pack, img)
        return fut

    def _pack(self, img: np.ndarray) -> Tuple[shared_memory.SharedMemory, str]:
        a = _compact(np.ascontiguousarray(img, dtype=np.float32))
        with self._lock:
            shm = self._free.pop() if self._free else None
        if shm is None:
            shm = shared_memory.SharedMemory(create=True, size=self._slot_bytes)
            with self._lock:
                self._slots.append(shm)
        np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)[...] = a
        return shm, a.dtype.name

    def _flow_job(self, i: int, nbs: List[int], gi: Future, gj: List[Future]) -> np.ndarray:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)   # as the CPU code: one per frame
        g_i = gi.result()
        out = np.empty((len(nbs), self.lh, self.lw, 2), np.float32)
        for k, fut in enumerate(gj):
            fs = dis.calc(g_i, fut.result(), None)
            out[k, ..., 0] = fs[..., 0] * self.sx          # the CPU code's float32 products
            out[k, ..., 1] = fs[..., 1] * self.sy
        return out

    def _schedule(self, i: int) -> bool:
        """Start the flows and uploads of output frame i; False when its window is not streamed yet."""
        if i >= self.n:
            return False
        if i in self._flows:
            return True
        try:
            win = self._window(i)
            ups = [self._upload(j) for j in win]
            gi = self._gray_of(i)
            nbs = [j for j in win if j != i]
            gj = [self._gray_of(j) for j in nbs]
        except (KeyError, IndexError):
            return False
        del ups
        self._flows[i] = self.pool.submit(self._flow_job, i, nbs, gi, gj)
        return True

    # ---------------------------------------------------------------- one output frame
    def __call__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._schedule(i):
            raise KeyError(f"GPU denoise: window of frame {i} not available")
        for k in range(1, PREFETCH + 1):
            if not self._schedule(i + k):
                break
        win = self._window(i)
        flows = self._flows.pop(i).result()
        if self._flow_shm is None or self._flow_shm.size < flows.nbytes:
            if self._flow_shm is not None:
                self._flow_shm.close()
                self._flow_shm.unlink()
            self._flow_shm = shared_memory.SharedMemory(create=True, size=max(1, flows.nbytes))
        np.ndarray(flows.shape, np.float32, buffer=self._flow_shm.buf)[...] = flows
        frames_req = []
        for j in win:
            shm, dt = self._up[j].result()
            frames_req.append({"j": j, "shm": shm.name, "dtype": dt})
        self.gpu.call({"op": "denoise_mc", "i": i, "h": self.h, "w": self.w, "c": self.c, "frames": frames_req,
                       "flows": {"shm": self._flow_shm.name, "shape": list(flows.shape)},
                       "sigma": self.sigma, "out": self._out.name}, self.cancel_check)
        self.calls += 1
        filtered = np.array(np.ndarray((self.h, self.w, self.c), np.float32, buffer=self._out.buf))
        cur = np.asarray(self.frames[i], dtype=np.float32)
        residual = cur - filtered
        self._drop_below(i + 1 - self.radius)
        return filtered, residual

    def _drop_below(self, j0: int) -> None:
        """Free what no later output frame reads (frames < j0)."""
        with self._lock:
            for j in [j for j in self._up if j < j0]:
                fut = self._up.pop(j)
                try:
                    self._free.append(fut.result()[0])
                except Exception:
                    pass
            for j in [j for j in self._gray if j < j0]:
                self._gray.pop(j)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._drop_below(self.n + self.radius + 1)
        self._flows.clear()
        self._free.clear()
        slots, self._slots = self._slots, []
        for shm in (self._out, self._flow_shm, *slots):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._flow_shm = None


class DenoiseStage:
    """The denoise stage of PlateCleaner.process_sequence on the chosen device: `make(input) -> fn(i)`, `workers`
    for engine.clean.stream_pixel_stages, `close()` at the end; `device` / `note` for the report.
    gpu: one worker thread (the GPU calls stay on the thread that owns the lease), a worker failure in auto mode
    switches the rest of the run to the CPU code."""

    def __init__(self, requested: Optional[str] = None, radius: int = 2, cpu_workers: int = 1,
                 cancel_check: Optional[Callable[[], None]] = None, python: Any = None, env: Optional[Dict[str, str]] = None):
        from engine import align_gpu
        self.radius = radius
        self.cancel_check = cancel_check
        self.requested = (requested or os.environ.get("MD_DENOISE_DEVICE") or "").lower() or None
        self.python, self.env = python, env
        if self.requested is None:
            try:
                from engine.config import load_config
                self.requested = str(load_config().get("denoise_device") or "auto").lower()
            except Exception:
                self.requested = "auto"
        self.device, self.note = align_gpu.resolve_device(self.requested, python, env, key="denoise_device")
        self.workers = 1 if self.device == "gpu" else cpu_workers
        self.gpu = None
        self.den: Optional[GpuDenoiser] = None
        self.gpu_frames = 0
        self.fell_back = False

    def make(self, inp: Sequence[np.ndarray]) -> Callable[[int], Tuple[np.ndarray, np.ndarray]]:
        from engine import align_gpu
        from engine.clean import DenoiseEngine
        cpu_fn: List[Optional[Callable[[int], Any]]] = [None]

        def cpu(i: int):
            if cpu_fn[0] is None:
                cpu_fn[0] = DenoiseEngine.denoise_frame_fn(inp, radius=self.radius)
            return cpu_fn[0](i)

        if self.device != "gpu":
            return cpu
        try:
            self.gpu = align_gpu.open_aligner(self.python, self.env)
            self.den = GpuDenoiser(self.gpu, inp, self.radius, cancel_check=self.cancel_check)
        except (align_gpu.GpuWorkerError, ValueError) as e:
            self._fallback(e)
            return cpu

        def fn(i: int):
            if self.den is None:
                return cpu(i)
            try:
                out = self.den(i)
                self.gpu_frames += 1
                return out
            except align_gpu.GpuWorkerError as e:
                self._fallback(e)
                return cpu(i)

        return fn

    def _fallback(self, e: Exception) -> None:
        if self.requested == "gpu":
            raise e
        log.warning("Denoise: GPU worker failed, the rest on the CPU: %s", e)
        self.note = f"GPU zawiodło ({e}) – reszta klatek na CPU"
        self.fell_back = True
        self.device = "cpu"
        self._close_gpu()

    def _close_gpu(self) -> None:
        from engine import align_gpu
        den, self.den = self.den, None
        if den is not None:
            den.close()
        g, self.gpu = self.gpu, None
        if g is not None:
            align_gpu.close_aligner(g)

    def close(self) -> None:
        self._close_gpu()
