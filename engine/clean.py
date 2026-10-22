# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/clean.py - Plate cleaning module: Deflicker (v10 chain) and Motion-Compensated Denoise.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 4c, 4c.1-4c.5, 12: Stage E1c):
- 4-stage advanced deflicker based on verified SH010 v10 chain:
  1. Local level with motion compensation & guided filter upsampling (deflicker_bg_mc2.py)
  2. Tile-based independently flickering areas (9x6 grid) (deflicker_tiles_mc.py)
  3. Detail / shading band (2-40 px) with DIS optical flow & homography (deflicker_detail_mc.py)
  4. Global scalar per-channel correction (deflicker_global_mc.py)
- Flow-aligned flicker metric (measure_flow.py logic)
- Temporal motion-compensated denoising with noise residual export (4c.2)
- Standalone Nuke Group Deflicker_<plate>.nk generator (4c.4)
- Full preservation of HDR linear values, OpenEXR channels, compression, and metadata (4c.3)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
logger = logging.getLogger("matte_desk.clean")


# -----------------------------------------------------------------------------
# Data Models & Metrics
# -----------------------------------------------------------------------------

@dataclass
class ChannelFlickerMetric:
    r: float
    g: float
    b: float
    mean: float
    p95: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "r": round(self.r, 3),
            "g": round(self.g, 3),
            "b": round(self.b, 3),
            "mean": round(self.mean, 3),
            "p95": round(self.p95, 3)
        }


@dataclass
class CleanReport:
    plate_name: str
    stages_applied: List[str]
    flicker_before: ChannelFlickerMetric
    flicker_after: ChannelFlickerMetric
    reduction_pct: float
    noise_reduction_pct: float = 0.0
    gain_min: float = 1.0
    gain_max: float = 1.0
    frames_count: int = 0
    plate_w: int = 0  # set from manifest plate_w (no guessed default)
    plate_h: int = 0  # set from manifest plate_h
    date: str = ""
    nk_group_path: str = ""
    gain_map_pattern: str = ""
    export_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plate_name": self.plate_name,
            "stages_applied": self.stages_applied,
            "flicker_before": self.flicker_before.to_dict(),
            "flicker_after": self.flicker_after.to_dict(),
            "reduction_pct": round(self.reduction_pct, 2),
            "noise_reduction_pct": round(self.noise_reduction_pct, 2),
            "gain_min": round(self.gain_min, 4),
            "gain_max": round(self.gain_max, 4),
            "frames_count": self.frames_count,
            "plate_w": self.plate_w,
            "plate_h": self.plate_h,
            "date": self.date,
            "nk_group_path": self.nk_group_path,
            "gain_map_pattern": self.gain_map_pattern,
            "export_dir": self.export_dir
        }


# -----------------------------------------------------------------------------
# Guided Filter (He et al.) & Homography Utilities
# -----------------------------------------------------------------------------

def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Fast guided filter for edge-aware upsampling and smoothing.
    
    guide: 2D float32 image (e.g. log-luma)
    src: 2D or 3D float32 image to filter
    r: filter radius
    eps: regularization parameter
    """
    def _box(img: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)

    mean_I = _box(guide)
    mean_II = _box(guide * guide)
    var_I = mean_II - mean_I * mean_I

    if src.ndim == 2:
        mean_p = _box(src)
        mean_Ip = _box(guide * src)
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return _box(a) * guide + _box(b)
    else:
        out = np.empty_like(src)
        for c in range(src.shape[2]):
            sc = src[..., c]
            mean_p = _box(sc)
            mean_Ip = _box(guide * sc)
            cov_Ip = mean_Ip - mean_I * mean_p
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            out[..., c] = _box(a) * guide + _box(b)
        return out


def compute_log_luma_gray(img: np.ndarray, target_w: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Computes downscaled image and normalized 8-bit log-luma image for feature tracking."""
    h, w = img.shape[:2]
    th = max(16, round(h * target_w / w))
    small = np.maximum(cv2.resize(img, (target_w, th), interpolation=cv2.INTER_AREA), 0.0)
    # Luma in BGR:
    luma = small.mean(axis=2) if small.ndim == 3 else small
    log_l = np.log(luma + 0.01)
    gray_u8 = cv2.normalize(log_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return small, gray_u8


def estimate_homography(gray_a: np.ndarray, gray_b: np.ndarray, det: Optional[Any] = None) -> Tuple[np.ndarray, int]:
    """Estimates homography mapping frame b coordinates -> frame a coordinates."""
    if det is None:
        det = cv2.ORB_create(2000)
    ka, da = det.detectAndCompute(gray_a, None)
    kb, db = det.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 15 or len(kb) < 15:
        return np.eye(3, dtype=np.float32), 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(db, da)
    if len(matches) < 15:
        return np.eye(3, dtype=np.float32), len(matches)
    pb = np.float32([kb[m.queryIdx].pt for m in matches])
    pa = np.float32([ka[m.trainIdx].pt for m in matches])
    H, inliers = cv2.findHomography(pb, pa, cv2.RANSAC, 2.5)
    if H is None:
        return np.eye(3, dtype=np.float32), 0
    return H.astype(np.float32), int(inliers.sum())


def smooth_time_keep_trend(L: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass along axis 0 that keeps linear trends at the clip ends.

    Plain `mode="nearest"` flattens a ramp near the first/last frames (intended light change would
    be treated as flicker). Odd (point) reflection about the end samples preserves a linear trend."""
    n = L.shape[0]
    pad = min(n - 1, int(np.ceil(4 * sigma)))
    if pad <= 0:
        return L.copy()
    head = 2 * L[:1] - L[1:pad + 1][::-1]
    tail = 2 * L[-1:] - L[-pad - 1:-1][::-1]
    ext = np.concatenate([head, L, tail], axis=0)
    return gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[pad:pad + n]


def interquartile_mean(stack: np.ndarray) -> np.ndarray:
    """Interquartile mean along axis 0, ignoring NaN values."""
    s = np.sort(stack, axis=0)
    n = np.sum(~np.isnan(stack), axis=0)
    lo_i = (n * 0.25).astype(int)
    hi_i = np.maximum((n * 0.75).astype(int), lo_i + 1)
    idx = np.arange(s.shape[0])[:, None, None, None]
    mask = (idx >= lo_i) & (idx < hi_i)
    denom = np.maximum(mask.sum(axis=0), 1)
    valid_s = np.where(mask & ~np.isnan(s), s, 0.0)
    return np.sum(valid_s, axis=0) / denom


# -----------------------------------------------------------------------------
# Threads for the per-frame loops (OpenCV / numpy release the GIL, so no process pool and no frame copies)
# -----------------------------------------------------------------------------

def _available_ram_bytes() -> Optional[int]:
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            m = _MS()
            m.l = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.avail)
            return None
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def clean_workers(per_worker_bytes: int = 0) -> int:
    """Threads for the per-frame loops. `MD_CLEAN_THREADS` (>= 1) overrides; otherwise the CPU count, capped so that
    `per_worker_bytes` of working memory per thread fits in half of the free RAM (1 thread when unknown / too tight)."""
    env = os.environ.get("MD_CLEAN_THREADS", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    n = max(1, min(os.cpu_count() or 1, 16))
    if per_worker_bytes > 0:
        avail = _available_ram_bytes()
        n = min(n, 2) if avail is None else min(n, max(1, int(avail * 0.5 // per_worker_bytes)))
    return n


def _map_frames(fn: Callable[[int], Any], n: int, workers: int) -> List[Any]:
    """[fn(0), ..., fn(n-1)] in order; fn calls run on `workers` threads (a plain loop for 1 worker / 1 frame).
    Also used on LazyFrames: each access (disk read + gain upscale) then runs on its own thread."""
    workers = max(1, min(workers, n))
    if workers == 1:
        return [fn(i) for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, range(n)))


# -----------------------------------------------------------------------------
# Flicker Metrics (flow-aligned, as in measure_flow.py)
# -----------------------------------------------------------------------------

class LazyFrames(Sequence):
    """Read-only list of images computed on access: `LazyFrames(n, getter)[i] == getter(i)`.

    A 4K RGB float32 frame is ~134 MB, so the cleaner never holds a whole sequence at full resolution: frames and gain
    maps are derived per frame from low-resolution data (stored guided-filter gains) or re-read from disk."""

    def __init__(self, n: int, getter: Callable[[int], np.ndarray]):
        self._n = n
        self._getter = getter

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        return self._getter(i)


class PinnedFrameCache(Sequence):
    """Frames read once and kept in RAM up to a fixed budget; the rest is read again on every access.

    The cleaner sweeps the whole sequence ~10 times (measure, homographies, stages 1 / 2 / 4, combine, export – each
    pass frame 1 .. N), so an LRU / sliding window would evict every frame before the next pass returns to it. The
    first frames that fit the budget are pinned for the whole run instead (hit rate = budget / sequence, never 0).
    Stored as float16 when that is lossless (half EXR plates: always), else as loaded; every access returns a fresh
    float32 array, so the result is bit for bit the one of plain re-reading. Thread-safe (the per-frame loops run on
    threads; two threads loading the same frame at once only waste one read)."""

    def __init__(self, n: int, loader: Callable[[int], np.ndarray], budget_bytes: int):
        self._n = n
        self._loader = loader
        self.budget = max(0, int(budget_bytes))
        self.used = 0
        self._pinned: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self.reads = 0
        self.hits = 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        with self._lock:
            hit = self._pinned.get(i)
            if hit is not None:
                self.hits += 1
        if hit is not None:
            return hit.astype(np.float32)          # astype copies: callers never see the pinned array
        img = np.asarray(self._loader(i), dtype=np.float32)
        with self._lock:
            self.reads += 1
            if i not in self._pinned and self.used < self.budget:
                small = img.astype(np.float16)
                keep = small if np.array_equal(small.astype(np.float32), img, equal_nan=True) else img.copy()
                if self.used + keep.nbytes <= self.budget:
                    self._pinned[i] = keep
                    self.used += keep.nbytes
        return img

    def stats(self) -> Dict[str, Any]:
        return {"frames": self._n, "pinned": len(self._pinned), "pinned_gb": round(self.used / 2 ** 30, 2),
                "budget_gb": round(self.budget / 2 ** 30, 2), "disk_reads": self.reads, "ram_hits": self.hits}


CACHE_RAM_SHARE = 0.4   # of the free RAM when the run starts (the per-frame threads need their working memory too)


def clean_cache_budget() -> int:
    """RAM budget of PinnedFrameCache in bytes: `MD_CLEAN_CACHE_GB` (0 = off) or 40 % of the free RAM (0 when unknown)."""
    env = os.environ.get("MD_CLEAN_CACHE_GB", "").strip()
    if env:
        try:
            return max(0, int(float(env) * 2 ** 30))
        except ValueError:
            pass
    avail = _available_ram_bytes()
    return int(avail * CACHE_RAM_SHARE) if avail else 0


class _MemoView(Sequence):
    """Read-only view of a dict {frame: array} of length n (the input of the next streamed stage)."""

    def __init__(self, n: int, memo: Dict[int, Any]):
        self._n = n
        self._memo = memo

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        return self._memo[i]


def stream_pixel_stages(n: int, stages: List[Tuple[int, Callable[[Sequence], Callable[[int], Any]], int]],
                        emit: Callable[[int, Any], None], block: int,
                        on_value: Optional[List[Optional[Callable[[int, Any], None]]]] = None) -> None:
    """Pixel stages (stage 3, denoise) of a whole sequence with only a few frames in RAM, results bit for bit those of
    the stages run on full lists (every stage is a pure per-frame function of its input window).

    `stages`: [(radius, make_fn, workers)] in order; make_fn(input) -> fn(i) where input is the previous stage's
    frames (stage 0 gets None and must read its own source) and fn(i) reads input[i - radius .. i + radius] only.
    Output frames are produced in blocks of `block` (each block on `workers` threads), passed to emit(i, value) in
    order, and every stage keeps only what its next stage still needs. on_value[k](i, value): called once per new
    value of stage k (e.g. low-res metrics of a stage's output)."""
    m = len(stages)
    memos: List[Dict[int, Any]] = [{} for _ in range(m)]
    fns: List[Optional[Callable[[int], Any]]] = [None] * m
    radii = [r for r, _, _ in stages]

    def ensure(k: int, lo: int, hi: int) -> None:
        lo, hi = max(0, lo), min(n, hi)
        if lo >= hi:
            return
        if k > 0:
            ensure(k - 1, lo - radii[k], hi + radii[k])
        if fns[k] is None:   # built once its input holds frame 0 (fns read frames[0].shape)
            fns[k] = stages[k][1](_MemoView(n, memos[k - 1]) if k > 0 else None)
        todo = [i for i in range(lo, hi) if i not in memos[k]]
        vals = _map_frames(lambda t: fns[k](todo[t]), len(todo), stages[k][2])
        for i, v in zip(todo, vals):
            memos[k][i] = v
            if on_value and on_value[k]:
                on_value[k](i, v)

    for b0 in range(0, n, max(1, block)):
        b1 = min(n, b0 + max(1, block))
        ensure(m - 1, b0, b1)
        for i in range(b0, b1):
            emit(i, memos[m - 1].pop(i))
        need = b1   # frames below `need` of stage k are not read any more
        for k in range(m - 2, -1, -1):
            need -= radii[k + 1]
            for i in [i for i in memos[k] if i < need]:
                del memos[k][i]


def as_bgr(img: np.ndarray) -> np.ndarray:
    """Image as H x W x 3: grayscale is replicated, alpha / extra channels are dropped."""
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img[..., :3]


class DeflickerMetrics:
    """Computes flow-aligned frame-to-frame flicker and temporal variance."""

    @staticmethod
    def measure_flow_aligned(
        frames: List[np.ndarray],
        eval_w: int = 1024,
        roi_slices: Optional[Tuple[slice, slice]] = None,
        flow_ref: Optional[List[np.ndarray]] = None
    ) -> ChannelFlickerMetric:
        """Measures flow-aligned frame-to-frame difference % per channel (BGR/RGB).

        Same metric as tools/sh010_tools/deflicker/measure_flow.py. That script computes the flows on a
        fixed reference render (e.g. v09/v10) so every compared version is aligned identically – pass that
        sequence as `flow_ref` to reproduce it; otherwise flows come from `frames` themselves.
        """
        n = len(frames)
        if n < 2:
            return ChannelFlickerMetric(r=0.0, g=0.0, b=0.0, mean=0.0, p95=0.0)

        h, w = frames[0].shape[:2]
        eval_h = round(h * eval_w / w)
        gx, gy = np.meshgrid(np.arange(eval_w, dtype=np.float32), np.arange(eval_h, dtype=np.float32))

        # Downscale and compute log-luma
        def _small_gray(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # as_bgr: grayscale / RGBA plates, the per-channel metric needs exactly 3 channels
            sm = np.maximum(cv2.resize(as_bgr(f), (eval_w, eval_h), interpolation=cv2.INTER_AREA), 0.0)
            lg = np.log(sm.mean(axis=2) + 0.01) if sm.ndim == 3 else np.log(sm + 0.01)
            return sm, cv2.normalize(lg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        workers = clean_workers(h * w * 3 * 4 * 4)   # a (lazy) full-res frame is alive per thread
        smalls, grays = (list(x) for x in zip(*_map_frames(lambda i: _small_gray(frames[i]), n, workers)))

        if roi_slices is None:
            # Default central ROI: avoid outer 5% border
            ys = slice(int(eval_h * 0.1), int(eval_h * 0.9))
            xs = slice(int(eval_w * 0.05), int(eval_w * 0.95))
        else:
            ys, xs = roi_slices

        if flow_ref is not None:
            grays = [g for _, g in _map_frames(lambda i: _small_gray(flow_ref[i]), len(flow_ref), workers)]

        def _diff(k: int) -> np.ndarray:
            i = k + 1
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)   # one per call: not shared between threads
            flow = dis.calc(grays[i], grays[i - 1], None)
            warped_prev = cv2.remap(
                smalls[i - 1],
                gx + flow[..., 0],
                gy + flow[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            cur_blur = cv2.GaussianBlur(smalls[i], (0, 0), 2.0)[ys, xs]
            prev_blur = cv2.GaussianBlur(warped_prev, (0, 0), 2.0)[ys, xs]
            ratio = np.log2((cur_blur + 0.02) / (prev_blur + 0.02))
            return np.abs(ratio).mean(axis=(0, 1)) * 100.0

        diffs = _map_frames(_diff, n - 1, workers)

        diffs_arr = np.array(diffs)  # shape (N-1, 3) in BGR
        # OpenCV reads as BGR, so indices are 0: B, 1: G, 2: R
        mean_b = float(diffs_arr[:, 0].mean())
        mean_g = float(diffs_arr[:, 1].mean())
        mean_r = float(diffs_arr[:, 2].mean()) if diffs_arr.shape[1] > 2 else mean_b
        all_mean = float(diffs_arr.mean())
        p95 = float(np.percentile(diffs_arr.mean(axis=1), 95))

        return ChannelFlickerMetric(r=mean_r, g=mean_g, b=mean_b, mean=all_mean, p95=p95)

    @staticmethod
    def measure_temporal_variance(
        frames: List[np.ndarray],
        static_mask: Optional[np.ndarray] = None
    ) -> float:
        """Measures rapid frame-to-frame variance of luma in static regions."""
        if len(frames) < 3:
            return 0.0
        n = len(frames)
        h, w = frames[0].shape[:2]
        small_w = min(512, w)
        small_h = round(h * small_w / w)
        
        def _luma(f: np.ndarray) -> np.ndarray:
            sm = cv2.resize(f, (small_w, small_h), interpolation=cv2.INTER_AREA)
            return sm.mean(axis=2) if sm.ndim == 3 else sm

        lumas = _map_frames(lambda i: _luma(frames[i]), n, clean_workers(h * w * 3 * 4 * 4))
        lumas_arr = np.array(lumas, dtype=np.float32)

        # High-pass in time: d2 = y[t+1] - 2*y[t] + y[t-1]
        d2 = lumas_arr[2:] - 2.0 * lumas_arr[1:-1] + lumas_arr[:-2]
        if static_mask is not None:
            sm_res = cv2.resize(static_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0.5
            d2 = d2[:, sm_res]
        
        return float(np.std(d2))


# -----------------------------------------------------------------------------
# Deflicker Algorithm Implementations
# -----------------------------------------------------------------------------

class DeflickerEngine:
    """Implements SH010 v10 deflicker stages with motion compensation."""

    @staticmethod
    def build_homography_chain(frames: List[np.ndarray], track_w: int = 512) -> List[np.ndarray]:
        """Builds cumulative homography chain C[k] mapping frame k -> frame 0."""
        n = len(frames)
        h0, w0 = frames[0].shape[:2]
        workers = clean_workers(h0 * w0 * 3 * 4 * 4)
        grays = _map_frames(lambda i: compute_log_luma_gray(frames[i], track_w)[1], n, workers)

        def _step(k: int) -> np.ndarray:
            return estimate_homography(grays[k - 1], grays[k], cv2.ORB_create(3000))[0]   # one detector per call

        step = [np.eye(3, dtype=np.float32)] + _map_frames(lambda i: _step(i + 1), n - 1, clean_workers())

        C = [np.eye(3, dtype=np.float32)]
        for k in range(1, n):
            C.append(C[-1] @ step[k])
        return C

    @staticmethod
    def _lazy_gain_result(frames: Sequence[np.ndarray], log_gains: List[np.ndarray], w: int, h: int
                          ) -> Tuple[LazyFrames, LazyFrames]:
        """(corrected frames, gain maps) as lazy sequences over the stored low-res log gains: full-res
        exp(cubic upscale) is computed per access instead of holding 2 full-res arrays per frame."""
        gain_maps = LazyFrames(len(log_gains), lambda i: np.exp(cv2.resize(log_gains[i], (w, h), interpolation=cv2.INTER_CUBIC)))
        return LazyFrames(len(log_gains), lambda i: frames[i] * gain_maps[i]), gain_maps

    @staticmethod
    def run_stage1_local_mc(
        frames: List[np.ndarray],
        homography_chain: Optional[List[np.ndarray]] = None,
        win_radius: int = 15,
        lo_w: int = 512,
        gf_w: int = 1024,
        gf_r: int = 16,
        gf_eps: float = 0.1,
        iterations: int = 2,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        sigma: float = 6.0,
        content_check: float = 0.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Stage 1: Motion-compensated local per-channel deflicker (deflicker_bg_mc2.py). `sigma`: blur of the low-res
        (512 px) comparison = size of the smallest brightness patch the stage corrects ("map smoothness" slider).
        `content_check` (0 = off, else the minimum local correlation): a neighbour's sample enters the robust mean only
        where its CONTENT matches the current frame – normalised correlation (9x9 px at 512) of the high-pass log detail
        of the ORIGINAL low-res frames, independent of brightness. A moving object = different content -> left out; a
        lamp footprint on a static surface = same texture, other brightness -> kept (a brightness / sharpness test cannot
        tell the two apart: it removed the SH010 tree correction, 2.88 -> 3.10 %). E1c ghost at 15 / 3 / sigma 4:
        0.0046 (without: 0.0063), SH010 tree 1.96 % (15 / 2 / sigma 6 without: 2.23 %) – PROGRESS 2026-10-22.
        
        Returns:
            (deflickered_frames, gain_maps)
        """
        n = len(frames)
        h, w = frames[0].shape[:2]
        if homography_chain is None:
            homography_chain = DeflickerEngine.build_homography_chain(frames, lo_w)
        C = homography_chain

        lo_h = max(16, round(h * lo_w / w))
        lo_stack = _map_frames(lambda i: np.maximum(cv2.resize(frames[i], (lo_w, lo_h), interpolation=cv2.INTER_AREA), 0.0),
                               n, clean_workers(h * w * 3 * 4 * 4))
        stack = np.stack(lo_stack, axis=0).astype(np.float32)

        sigma = float(sigma)
        eps = 0.01
        min_ok = 10
        fill_sigma = 25.0
        rmin, rmax = 0.25, 4.0

        def _blur(img: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)

        total_gain = np.ones_like(stack)
        cur_stack = stack.copy()
        details: List[np.ndarray] = []
        if content_check > 0:
            def _detail(img: np.ndarray) -> np.ndarray:   # high-pass log luma = content, not brightness
                lg = np.log(np.maximum(img, 0.0) + eps).mean(axis=2).astype(np.float32)
                return lg - cv2.GaussianBlur(lg, (0, 0), 3.0, borderType=cv2.BORDER_REPLICATE)
            details = [_detail(stack[k]) for k in range(n)]

        for it in range(iterations):
            cur_blurred = np.stack([_blur(cur_stack[i]) for i in range(n)], axis=0)
            new_total = np.empty_like(total_gain)

            def _frame(i: int) -> None:
                j_start = max(0, i - win_radius)
                j_end = min(n, i + win_radius + 1)
                Ci_inv = np.linalg.inv(C[i])

                aligned_list = []
                for j in range(j_start, j_end):
                    M = Ci_inv @ C[j]  # maps frame j -> frame i
                    warped = cv2.warpPerspective(
                        cur_blurred[j],
                        M,
                        (lo_w, lo_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(np.nan, np.nan, np.nan)
                    )
                    aligned_list.append(warped)

                A = np.stack(aligned_list, axis=0)
                # Keep intended light changes (SPEC 4c / 13: no flattening): remove the per-pixel linear
                # log-trend of the window before the robust mean, so the target follows the low-pass trend
                # also near the clip ends where the window is one-sided.
                x = (np.arange(j_start, j_end, dtype=np.float32) - i)[:, None, None, None]
                L = np.log(A + eps)
                ok = ~np.isnan(L)
                cnt_t = np.maximum(ok.sum(axis=0), 1)
                xm = np.where(ok, x, 0.0).sum(axis=0) / cnt_t
                Lm = np.where(ok, L, 0.0).sum(axis=0) / cnt_t
                sxx = np.where(ok, (x - xm) ** 2, 0.0).sum(axis=0)
                sxy = np.where(ok, (x - xm) * (L - Lm), 0.0).sum(axis=0)
                slope = np.where(sxx > 1e-6, sxy / np.maximum(sxx, 1e-6), 0.0)
                A = (A + eps) * np.exp(-slope * x) - eps
                if content_check > 0:
                    # a neighbour's sample only where its content correlates with frame i (moving object -> out)
                    di = details[i]
                    box = (9, 9)
                    mi = cv2.boxFilter(di, -1, box)
                    vi = cv2.boxFilter(di * di, -1, box) - mi * mi
                    for q in range(A.shape[0]):
                        j = j_start + q
                        if j == i:
                            continue
                        dj = cv2.warpPerspective(details[j], Ci_inv @ C[j], (lo_w, lo_h), flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                        mj = cv2.boxFilter(dj, -1, box)
                        vj = cv2.boxFilter(dj * dj, -1, box) - mj * mj
                        ncc = (cv2.boxFilter(di * dj, -1, box) - mi * mj) / np.sqrt(np.maximum(vi * vj, 1e-12))
                        bad = (vi > 1e-4) & (vj > 1e-4) & (ncc < content_check)   # flat areas: no evidence, kept
                        if bad.any():
                            bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                            A[q][bad] = np.nan
                ref = interquartile_mean(A)
                cnt = np.sum(~np.isnan(A).any(axis=-1), axis=0).astype(np.float32)
                r = np.clip((ref + eps) / (cur_blurred[i] + eps), rmin, rmax)

                # Normalized convolution for borders with low samples
                w_conf = np.clip((cnt - 2.0) / (min_ok - 2.0), 0.0, 1.0)[..., None]
                # Moving areas excluded from the measurement (SPEC 4c.1): real flicker gains are spatially
                # smooth, so localized structure in the ratio comes from animation/parallax -> no confidence,
                # filled from the surroundings below (otherwise the object imprints a ghost into the gain).
                lr = np.log(r).mean(axis=2)
                hf = np.abs(lr - cv2.GaussianBlur(lr, (0, 0), fill_sigma, borderType=cv2.BORDER_REPLICATE))
                w_conf = w_conf * np.exp(-((hf / 0.02) ** 2))[..., None]
                def _bw(x: np.ndarray) -> np.ndarray:
                    return cv2.GaussianBlur(x, (0, 0), fill_sigma, borderType=cv2.BORDER_REPLICATE)
                wf = _bw(np.repeat(w_conf, 3, axis=2)) + 1e-6
                fill = np.exp(_bw(np.log(r) * w_conf) / wf)
                r_final = w_conf * r + (1.0 - w_conf) * fill
                new_total[i] = total_gain[i] * r_final
            _map_frames(_frame, n, clean_workers((2 * win_radius + 1) * lo_h * lo_w * 3 * 4 * 12))

            total_gain = np.clip(new_total, rmin, rmax)
            cur_stack = stack * total_gain
            if progress_cb:
                progress_cb(it + 1, iterations)

        # Upsample gain maps using Guided Filter to full resolution
        gh = round(h * gf_w / w)
        def _log_gain(i: int) -> np.ndarray:   # guided log-gain at gf_w: the only per-frame data kept; full-res gain is derived on access
            small_a = np.maximum(cv2.resize(frames[i], (gf_w, gh), interpolation=cv2.INTER_AREA), 0.0)
            guide = np.log(small_a.mean(axis=2) + 0.01).astype(np.float32)
            lr = np.log(cv2.resize(total_gain[i], (gf_w, gh), interpolation=cv2.INTER_CUBIC)).astype(np.float32)
            return guided_filter(guide, lr, gf_r, gf_eps)

        log_gains = _map_frames(_log_gain, n, clean_workers(h * w * 3 * 4 * 4))

        return DeflickerEngine._lazy_gain_result(frames, log_gains, w, h)

    @staticmethod
    def run_stage2_tiles_mc(
        frames: List[np.ndarray],
        homography_chain: Optional[List[np.ndarray]] = None,
        gx: int = 9,
        gy: int = 6,
        time_sigma: float = 6.0,
        gmin: float = 0.8,
        gmax: float = 1.25,
        gf_w: int = 1024,
        gf_r: int = 16,
        gf_eps: float = 0.1
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Stage 2: Tile-based motion-compensated per-channel touch-up (deflicker_tiles_mc.py)."""
        n = len(frames)
        h, w = frames[0].shape[:2]
        lw, lh = 512, round(h * 512 / w)

        if homography_chain is None:
            homography_chain = DeflickerEngine.build_homography_chain(frames, lw)
        C = homography_chain

        smalls = _map_frames(lambda i: np.maximum(cv2.resize(frames[i], (lw, lh), interpolation=cv2.INTER_AREA), 0.0),
                             n, clean_workers(h * w * 3 * 4 * 4))

        steps = [np.zeros((gy, gx, 3), dtype=np.float32)]
        for k in range(1, n):
            # Homography mapping k-1 to k
            H_k_prev = np.linalg.inv(C[k]) @ C[k - 1]
            warped_prev = cv2.warpPerspective(
                smalls[k - 1],
                H_k_prev,
                (lw, lh),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(np.nan, np.nan, np.nan)
            )
            s = np.zeros((gy, gx, 3), dtype=np.float32)
            for y in range(gy):
                for x in range(gx):
                    ys = slice(y * lh // gy, (y + 1) * lh // gy)
                    xs = slice(x * lw // gx, (x + 1) * lw // gx)
                    a_tile = smalls[k][ys, xs]
                    b_tile = warped_prev[ys, xs]
                    ok = ~np.isnan(b_tile).any(-1)
                    if ok.mean() > 0.4:
                        s[y, x] = np.log((a_tile[ok].mean(0) + 1e-4) / (b_tile[ok].mean(0) + 1e-4))
            steps.append(s)

        L = np.cumsum(np.array(steps, dtype=np.float32), axis=0)
        # Handle temporal smoothing with fallback when n <= time_sigma
        effective_sigma = min(time_sigma, max(1.0, (n - 1) / 2.0))
        Ls = smooth_time_keep_trend(L, effective_sigma)
        tile_gains = np.clip(np.exp(Ls - L), gmin, gmax).astype(np.float32)

        # Upsample tile gains with guided filter
        gh = round(h * gf_w / w)
        def _log_gain(i: int) -> np.ndarray:
            small_a = np.maximum(cv2.resize(frames[i], (gf_w, gh), interpolation=cv2.INTER_AREA), 0.0)
            guide = np.log(small_a.mean(axis=2) + 0.01).astype(np.float32)
            lg = cv2.resize(np.log(tile_gains[i]), (gf_w, gh), interpolation=cv2.INTER_CUBIC)
            lg = cv2.GaussianBlur(lg, (0, 0), gf_w / gx / 3.0)
            return guided_filter(guide, lg, gf_r, gf_eps)

        log_gains = _map_frames(_log_gain, n, clean_workers(h * w * 3 * 4 * 4))

        return DeflickerEngine._lazy_gain_result(frames, log_gains, w, h)

    @staticmethod
    def run_stage3_detail_mc(
        frames: List[np.ndarray],
        homography_chain: Optional[List[np.ndarray]] = None,
        radius: int = 3,
        fw: int = 1024,
        s_fine: float = 2.0,
        s_base: float = 40.0
    ) -> List[np.ndarray]:
        """Stage 3: Shading/detail deflicker (mid-frequency band 2-40 px) with DIS flow.

        Note: Reconstructs details directly; cannot be represented as low-res gain maps.
        """
        if homography_chain is None:
            homography_chain = DeflickerEngine.build_homography_chain(frames, 512)
        fn = DeflickerEngine.stage3_frame_fn(frames, homography_chain, radius, fw, s_fine, s_base)
        h, w = frames[0].shape[:2]
        return _map_frames(fn, len(frames), clean_workers(h * w * 3 * 4 * 16))

    @staticmethod
    def stage3_frame_fn(
        frames: Sequence[np.ndarray],
        homography_chain: List[np.ndarray],
        radius: int = 3,
        fw: int = 1024,
        s_fine: float = 2.0,
        s_base: float = 40.0
    ) -> Callable[[int], np.ndarray]:
        """Stage 3 of one frame: fn(i) reads only frames[i - radius .. i + radius] (streamed in process_sequence)."""
        n = len(frames)
        h, w = frames[0].shape[:2]
        fh = round(h * fw / w)
        sx = w / fw
        leps = 0.01
        HCH = homography_chain

        def _logimg(img: np.ndarray) -> np.ndarray:
            return np.log(np.maximum(img, 0.0) + leps).astype(np.float32)

        def _luma_u8(sm: np.ndarray) -> np.ndarray:
            g = np.log(sm.mean(axis=2) + 0.01) if sm.ndim == 3 else np.log(sm + 0.01)
            return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        gx, gy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))

        def _frame(i: int) -> np.ndarray:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)   # one per call: not shared between threads
            cur = frames[i]
            Lc = _logimg(cur)
            small_c = cv2.resize(cur, (fw, fh), interpolation=cv2.INTER_AREA)
            gc = _luma_u8(small_c)

            base_c = cv2.resize(
                cv2.GaussianBlur(_logimg(small_c), (0, 0), s_base / sx),
                (w, h),
                interpolation=cv2.INTER_CUBIC
            )
            fine_blur_c = cv2.GaussianBlur(Lc, (0, 0), s_fine)
            fine_c = Lc - fine_blur_c
            mids = [fine_blur_c - base_c]

            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                if j == i:
                    continue
                nb = frames[j]
                M = np.linalg.inv(HCH[i]) @ HCH[j]
                S = np.diag([fw / 512.0, fw / 512.0, 1.0]).astype(np.float32)
                Mf = S @ M @ np.linalg.inv(S)

                small_n = cv2.resize(nb, (fw, fh), interpolation=cv2.INTER_AREA)
                warped_small = cv2.warpPerspective(small_n, Mf, (fw, fh), borderMode=cv2.BORDER_REPLICATE)
                flow = dis.calc(gc, _luma_u8(warped_small), None)

                # Sampling map
                px = gx + flow[..., 0]
                py = gy + flow[..., 1]
                Minv = np.linalg.inv(Mf)
                den = Minv[2, 0] * px + Minv[2, 1] * py + Minv[2, 2]
                mx = (Minv[0, 0] * px + Minv[0, 1] * py + Minv[0, 2]) / den
                my = (Minv[1, 0] * px + Minv[1, 1] * py + Minv[1, 2]) / den

                # small-grid coordinates -> full-size pixel coordinates with the pixel-centre convention
                # ((x + 0.5) * s - 0.5); plain x * s sampled 1.5 px off at 4096 / 1024 and shifted the result
                # ~0.25 px right / 0.2 px down (SH010 export 2026-10-22, also in the v10 script deflicker_detail_mc.py)
                sy = h / fh
                mapx = cv2.resize((mx + 0.5) * sx - 0.5, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                mapy = cv2.resize((my + 0.5) * sy - 0.5, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                Ln = cv2.remap(_logimg(nb), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                mid_n = cv2.GaussianBlur(Ln, (0, 0), s_fine) - cv2.resize(
                    cv2.GaussianBlur(cv2.resize(Ln, (fw, fh), interpolation=cv2.INTER_AREA), (0, 0), s_base / sx),
                    (w, h),
                    interpolation=cv2.INTER_CUBIC
                )
                mids.append(mid_n.astype(np.float32))

            # Block-wise median along axis 0
            mid_median = np.empty_like(mids[0])
            for y0 in range(0, h, 256):
                y1 = min(h, y0 + 256)
                mid_median[y0:y1] = np.median(np.stack([m[y0:y1] for m in mids], axis=0), axis=0)

            out = np.exp(base_c + mid_median + fine_c) - leps
            out = np.where(cur < 0.0, cur, np.maximum(out, 0.0))
            return out.astype(np.float32)

        return _frame

    @staticmethod
    def run_stage4_global_mc(
        frames: List[np.ndarray],
        homography_chain: Optional[List[np.ndarray]] = None,
        channels: str = "BGR",
        time_sigma: float = 6.0,
        luma_only: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Stage 4: Final global per-frame scalar gain per channel (deflicker_global_mc.py). `luma_only`: one gain for
        all channels (from the mean of the channel ratios) – when the lamp's colour is stable, per-channel gains only
        chase colour noise."""
        n = len(frames)
        h, w = frames[0].shape[:2]
        lw, lh = 512, round(h * 512 / w)

        if homography_chain is None:
            homography_chain = DeflickerEngine.build_homography_chain(frames, lw)
        C = homography_chain

        smalls = _map_frames(lambda i: np.maximum(cv2.resize(frames[i], (lw, lh), interpolation=cv2.INTER_AREA), 0.0),
                             n, clean_workers(h * w * 3 * 4 * 4))
        ys = slice(int(lh * 0.1), int(lh * 0.9))
        xs = slice(int(lw * 0.05), int(lw * 0.95))

        logq = [np.zeros(3, dtype=np.float32)]
        for k in range(1, n):
            H_k_prev = np.linalg.inv(C[k]) @ C[k - 1]
            warped_prev = cv2.warpPerspective(
                smalls[k - 1],
                H_k_prev,
                (lw, lh),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(np.nan, np.nan, np.nan)
            )
            a_roi = smalls[k][ys, xs]
            b_roi = warped_prev[ys, xs]
            ok = ~np.isnan(b_roi).any(axis=-1)
            if ok.mean() > 0.2:
                ratio = np.log((a_roi[ok].mean(axis=0) + 1e-4) / (b_roi[ok].mean(axis=0) + 1e-4))
            else:
                ratio = np.zeros(3, dtype=np.float32)
            logq.append(ratio)

        L = np.cumsum(np.array(logq, dtype=np.float32), axis=0)
        if luma_only:
            L = np.repeat(L.mean(axis=1, keepdims=True), 3, axis=1)
        effective_sigma = min(time_sigma, max(1.0, (n - 1) / 2.0))
        Ls = smooth_time_keep_trend(L, effective_sigma)
        gains = np.exp(Ls - L).astype(np.float32)

        idx_map = {"B": 0, "G": 1, "R": 2}
        for c in "BGR":
            if c not in channels:
                gains[:, idx_map[c]] = 1.0

        # Uniform full frame gain, derived on access
        gain_maps = LazyFrames(n, lambda i: np.ones((h, w, 3), dtype=np.float32) * gains[i][None, None, :])
        return LazyFrames(n, lambda i: frames[i] * gain_maps[i]), gain_maps


# -----------------------------------------------------------------------------
# Denoising Engine (Motion-Compensated Temporal Filter)
# -----------------------------------------------------------------------------

class DenoiseEngine:
    """Motion-compensated temporal denoiser with photometric weighting."""

    @staticmethod
    def denoise_temporal_mc(
        frames: List[np.ndarray],
        radius: int = 2,
        spatial_sigma: float = 1.0,
        luminance_weight: float = 0.5
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Applies motion-compensated temporal bilateral/weighted filtering.

        Returns:
            (denoised_frames, noise_residuals)
        """
        fn = DenoiseEngine.denoise_frame_fn(frames, radius, spatial_sigma, luminance_weight)
        out = [fn(i) for i in range(len(frames))]
        return [d for d, _ in out], [r for _, r in out]

    @staticmethod
    def denoise_frame_fn(
        frames: Sequence[np.ndarray],
        radius: int = 2,
        spatial_sigma: float = 1.0,
        luminance_weight: float = 0.5
    ) -> Callable[[int], Tuple[np.ndarray, np.ndarray]]:
        """Denoise of one frame: fn(i) -> (denoised, residual), reads only frames[i - radius .. i + radius]."""
        n = len(frames)
        h, w = frames[0].shape[:2]
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        # Convert to small luma for fast flow
        lw = min(1024, w)
        lh = round(h * lw / w)
        sx, sy = w / lw, h / lh

        def _gray(f: np.ndarray) -> np.ndarray:
            sm = np.maximum(cv2.resize(f, (lw, lh), interpolation=cv2.INTER_AREA), 0.0)
            lg = np.log(sm.mean(axis=2) + 0.01) if sm.ndim == 3 else np.log(sm + 0.01)
            return cv2.normalize(lg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        def _frame(i: int) -> Tuple[np.ndarray, np.ndarray]:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)   # one per call: not shared between threads
            gray_i = _gray(frames[i])
            cur = frames[i].astype(np.float32)
            accum_weight = np.ones((h, w, 1), dtype=np.float32)
            accum_val = cur.copy()

            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                if j == i:
                    continue
                flow_small = dis.calc(gray_i, _gray(frames[j]), None)
                flow_x = cv2.resize(flow_small[..., 0] * sx, (w, h), interpolation=cv2.INTER_LINEAR)
                flow_y = cv2.resize(flow_small[..., 1] * sy, (w, h), interpolation=cv2.INTER_LINEAR)

                warped_nb = cv2.remap(
                    frames[j],
                    gx + flow_x,
                    gy + flow_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )

                # Photometric difference weight (exponential decay)
                diff = np.abs(cur - warped_nb)
                weight = np.exp(-diff / (0.05 + 1e-4)).mean(axis=-1, keepdims=True)
                temporal_dist_weight = 1.0 / (abs(i - j) + 0.5)
                w_total = weight * temporal_dist_weight

                accum_val += warped_nb * w_total
                accum_weight += w_total

            filtered = accum_val / accum_weight

            # Optional subtle spatial filter if requested
            if spatial_sigma > 0:
                filtered = cv2.GaussianBlur(filtered, (0, 0), spatial_sigma)
            return filtered, cur - filtered

        return _frame


# -----------------------------------------------------------------------------
# Nuke Deflicker Group Generator
# -----------------------------------------------------------------------------

class NukeDeflickerGroupGenerator:
    """Generates standard Nuke scripts containing a single Group node 'Deflicker_<plate>'."""

    @staticmethod
    def generate_script(
        plate_name: str,
        gain_pattern: str,
        plate_w: int,
        plate_h: int,
        first_frame: int,
        last_frame: int,
        report: CleanReport,
        output_nk_path: Path
    ) -> str:
        """Generates standard Nuke script with single Group node adhering to Nuke conventions.
        
        Outside: 1 input (Input1), 1 output (Output1).
        Inside: Dot1 -> Read(gain) -> Reformat -> Merge2(multiply) -> Switch1(0=Dot1, 1=Merge2) -> Output1.
        """
        group_name = f"Deflicker_{plate_name}"
        reduction = round(report.reduction_pct, 1)
        methods = "+".join(report.stages_applied)
        g_min = round(report.gain_min, 3)
        g_max = round(report.gain_max, 3)
        gain_pat_str = str(gain_pattern).replace("\\", "/")

        nk_text = f"""Group {{
 name {group_name}
 tile_color 0x2b5b84ff
 label "Deflicker -{reduction}% ({methods})"
 addUserKnob {{20 Deflicker l "Deflicker Control"}}
 addUserKnob {{6 use_deflicker l "Użyj deflickera (aktywny)"}}
 use_deflicker true
 addUserKnob {{26 spacer1 l ""}}
 addUserKnob {{26 info_reduction l "Redukcja migotania:" T "{reduction}%"}}
 addUserKnob {{26 info_method l "Zastosowana metoda:" T "{methods}"}}
 addUserKnob {{26 info_gain l "Zakres wzmocnienia:" T "{g_min} - {g_max}"}}
 addUserKnob {{26 info_date l "Data wygenerowania:" T "{report.date}"}}
 addUserKnob {{26 info_notes l "Uwagi:" T "Mapy wzmocnienia w formacie OpenEXR. Wynik identyczny z eksportem."}}
}}
 Input {{
  inputs 0
  name Input1
  xpos 0
  ypos 0
 }}
 Dot {{
  name Dot1
  xpos 34
  ypos 60
 }}
set N_dot1 [stack 0]
 Read {{
  inputs 0
  file "{gain_pat_str}"
  first {first_frame}
  last {last_frame}
  origfirst {first_frame}
  origlast {last_frame}
  format "{plate_w} {plate_h} 0 0 {plate_w} {plate_h} 1 plate_format"
  name Read_Gain
  xpos 180
  ypos 0
 }}
 Reformat {{
  type "to box"
  box_width {plate_w}
  box_height {plate_h}
  box_fixed true
  resize distort
  filter Cubic
  name Reformat_Gain
  xpos 180
  ypos 80
 }}
push $N_dot1
 Merge2 {{
  inputs 2
  operation multiply
  Achannels rgb
  Bchannels rgb
  output rgb
  name Merge_Gain
  xpos 180
  ypos 140
 }}
push $N_dot1
 Switch {{
  inputs 2
  which {{{{parent.use_deflicker}}}}
  name Switch_Deflicker
  xpos 34
  ypos 200
 }}
 Output {{
  name Output1
  xpos 34
  ypos 260
 }}
end_group
"""
        output_nk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_nk_path, "w", encoding="utf-8") as f:
            f.write(nk_text)
        return nk_text

    @staticmethod
    def validate_nuke_syntax(nk_content: str) -> bool:
        """Structural check (engine.nuke_nk.validate_nk) + use_deflicker switch present."""
        from engine.nuke_nk import validate_nk
        return (not validate_nk(nk_content)) and "which {{parent.use_deflicker}}" in nk_content


# -----------------------------------------------------------------------------
# High-Level Plate Cleaner
# -----------------------------------------------------------------------------

class PlateCleaner:
    """Orchestrates plate deflickering, denoising, metrics, and export."""

    @staticmethod
    def process_sequence(
        frames: List[np.ndarray],
        plate_name: str = "beauty",
        target_reduction: float = 0.70,
        enable_stage1: bool = True,
        enable_stage2: bool = True,
        enable_stage3: bool = False,
        enable_stage4: bool = True,
        enable_denoise: bool = False,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        frame_sink: Optional[Callable[[int, np.ndarray], None]] = None,
        stage1_window: int = 15,
        stage1_iterations: int = 3,
        stage3_radius: int = 4,
        stage1_sigma: float = 4.0,
        stage4_luma: bool = False,
        stage1_content_check: float = 0.5
    ) -> Tuple[Optional[Sequence[np.ndarray]], Sequence[np.ndarray], CleanReport, Optional[List[np.ndarray]]]:
        """Runs the deflicker and optional denoise pipeline on a list of image frames.

        `stage1_window` (+-frames compared), `stage1_iterations`, `stage3_radius` (+-frames of the texture median),
        `stage1_sigma` (map smoothness at 512 px), `stage4_luma` (one gain for all channels instead of R / G / B):
        defaults: window 15, 3 iterations, sigma 4, content check 0.5, texture +-4 (2026-10-22 night grid on the E1c
        synthetic + SH010 X1 F90-109: ghost 0.0046 < 0.005 and better than the old 15 / 2 / sigma 6 on the tree 1.96 vs
        2.23 %, figure 1.57 vs 1.65 %, whole frame 1.75 vs 1.81 %); more iterations remove a bit more but leave traces of
        moving objects again (15 / 5: ghost 0.0066) – reachable through the palette; capped by the sequence length.
        `frame_sink(i, frame)`: every cleaned frame in order (EXR export). With stage 3 / denoise the frames are then
        streamed – computed in small blocks, handed to the sink and dropped (a 4K sequence never sits in RAM) – and
        cleaned_frames / noise_residuals come back as None. Without a sink they are returned as lists (tests, small
        sequences).

        Returns:
            (cleaned_frames, combined_gain_maps, report, noise_residuals)
        """
        n = len(frames)
        h, w = frames[0].shape[:2]

        if progress_cb:
            progress_cb("Pomiar początkowy", 0, 100)
        flicker_init = DeflickerMetrics.measure_flow_aligned(frames)

        cur_frames = frames   # stages 1/2/4 return lazy sequences; nothing is modified in place
        stage_gains = []      # lazy full-res gain sequences of the applied stages (combined per frame below)
        stages_applied = []

        homographies = DeflickerEngine.build_homography_chain(frames, 512)

        # Stage 1: Local level with MC
        if enable_stage1:
            if progress_cb:
                progress_cb("Deflicker: obszary", 10, 100)
            cur_frames, g1 = DeflickerEngine.run_stage1_local_mc(
                cur_frames,
                homography_chain=homographies,
                win_radius=min(int(stage1_window), max(3, n // 2)),
                iterations=int(stage1_iterations),
                sigma=float(stage1_sigma),
                content_check=float(stage1_content_check)
            )
            stages_applied.append("stage1_local_mc")
            stage_gains.append(g1)

            m1 = DeflickerMetrics.measure_flow_aligned(cur_frames)
            red1 = (1.0 - m1.mean / (flicker_init.mean + 1e-6)) * 100.0
            if red1 >= target_reduction * 100.0 and m1.mean <= 4.0:
                enable_stage2 = False
                enable_stage4 = False

        # Stage 2: Tiles MC
        if enable_stage2:
            if progress_cb:
                progress_cb("Deflicker: kratka 9×6", 40, 100)
            cur_frames, g2 = DeflickerEngine.run_stage2_tiles_mc(
                cur_frames,
                homography_chain=homographies,
                gx=9,
                gy=6,
                time_sigma=6.0
            )
            stages_applied.append("stage2_tiles_mc")
            stage_gains.append(g2)

        # Stage 4: Global MC
        if enable_stage4:
            if progress_cb:
                progress_cb("Deflicker: cała klatka", 60, 100)
            cur_frames, g4 = DeflickerEngine.run_stage4_global_mc(
                cur_frames,
                homography_chain=homographies,
                channels="BGR",
                time_sigma=6.0,
                luma_only=bool(stage4_luma)
            )
            stages_applied.append("stage4_global_mc")
            stage_gains.append(g4)

        # Re-apply the gain exactly as the Nuke Group will see it (stored low-res half map, cubic upscale)
        if stages_applied:
            def _low(i: int) -> np.ndarray:
                combined = np.ones((h, w, 3), dtype=np.float32)
                for g in stage_gains:
                    combined *= g[i]
                low, _ = PlateCleaner.quantize_gain(combined)
                return low.astype(np.float16)

            # stored low-res half maps (<= 1024 wide): all that is kept per frame
            lows = _map_frames(_low, n, clean_workers(h * w * 3 * 4 * 6))
            del stage_gains
            combined_gains = LazyFrames(n, lambda i: PlateCleaner.upscale_gain(lows[i], w, h))
            cur_frames = LazyFrames(n, lambda i: (frames[i].astype(np.float32) * combined_gains[i]).astype(np.float32))
        else:
            combined_gains = LazyFrames(n, lambda i: np.ones((h, w, 3), dtype=np.float32))

        # Stage 3 / denoise reconstruct pixels directly (not low-res gains). Each frame only reads its neighbours
        # (stage 3: +-radius, denoise: +-2), so they are streamed in blocks: only a window of frames is in RAM and the
        # metrics are taken from low-res copies made on the way (same numbers as on the full frames).
        residuals = None
        noise_red = 0.0
        flicker_final = None
        if enable_stage3 or enable_denoise:
            r3 = min(int(stage3_radius), max(1, n // 4))
            fb = h * w * 3 * 4   # one full-res float32 frame
            src = cur_frames
            stages: List[Tuple[int, Callable[[Sequence], Callable[[int], Any]], int]] = [
                (0, lambda _inp: (lambda i: np.array(src[i], dtype=np.float32)), clean_workers(fb * 4))]
            if enable_stage3:
                stages.append((r3, lambda inp: DeflickerEngine.stage3_frame_fn(inp, homographies, r3), clean_workers(fb * 16)))
                stages_applied.append("stage3_detail_mc")
            if enable_denoise:
                stages.append((2, lambda inp: DenoiseEngine.denoise_frame_fn(inp, radius=2), clean_workers(fb * 12)))
                stages_applied.append("denoise_mc")
            eval_w = 1024
            eval_size = (eval_w, round(h * eval_w / w))
            var_w = min(512, w)
            var_size = (var_w, round(h * var_w / w))
            final_small: Dict[int, np.ndarray] = {}
            var_out: Dict[int, np.ndarray] = {}
            var_in: Dict[int, np.ndarray] = {}
            out_list: Optional[List[np.ndarray]] = [] if frame_sink is None else None
            res_list: Optional[List[np.ndarray]] = [] if (frame_sink is None and enable_denoise) else None
            on_value: List[Optional[Callable[[int, Any], None]]] = [None] * len(stages)
            if enable_denoise:   # input of the denoise = output of the stage before it
                on_value[-2] = lambda i, f: var_in.__setitem__(i, cv2.resize(f, var_size, interpolation=cv2.INTER_AREA))
            done = [0]

            def emit(i: int, v: Any) -> None:
                frame, res = v if enable_denoise else (v, None)
                final_small[i] = cv2.resize(as_bgr(frame), eval_size, interpolation=cv2.INTER_AREA)
                if enable_denoise:
                    var_out[i] = cv2.resize(frame, var_size, interpolation=cv2.INTER_AREA)
                if frame_sink is not None:
                    frame_sink(i, frame)
                else:
                    out_list.append(frame)
                    if res_list is not None:
                        res_list.append(res)
                done[0] += 1
                if progress_cb:
                    progress_cb("Faktura i cienie + odszumianie" if enable_stage3 and enable_denoise else
                                ("Migotanie faktury i cieni" if enable_stage3 else "Odszumianie w czasie"),
                                75 + int(23 * done[0] / n), 100)

            block = max(st[2] for st in stages)
            stream_pixel_stages(n, stages, emit, block, on_value)
            if enable_denoise:
                var_before = DeflickerMetrics.measure_temporal_variance([var_in[i] for i in range(n)])
                var_after = DeflickerMetrics.measure_temporal_variance([var_out[i] for i in range(n)])
                noise_red = max(0.0, (1.0 - var_after / (var_before + 1e-6)) * 100.0)
            if progress_cb:
                progress_cb("Pomiar końcowy", 98, 100)
            flicker_final = DeflickerMetrics.measure_flow_aligned([final_small[i] for i in range(n)], eval_w=eval_w)
            cur_frames = out_list
            residuals = res_list

        # Final Metrics
        if flicker_final is None:
            if progress_cb:
                progress_cb("Pomiar końcowy", 98, 100)
            flicker_final = DeflickerMetrics.measure_flow_aligned(cur_frames)
            if frame_sink is not None:
                for i in range(n):
                    frame_sink(i, cur_frames[i])
        reduction_pct = max(0.0, (1.0 - flicker_final.mean / (flicker_init.mean + 1e-6)) * 100.0)

        # Gain limits
        all_gains_cat = np.array([g.mean() for g in combined_gains])
        g_min = float(all_gains_cat.min())
        g_max = float(all_gains_cat.max())

        import datetime
        report = CleanReport(
            plate_name=plate_name,
            stages_applied=stages_applied,
            flicker_before=flicker_init,
            flicker_after=flicker_final,
            reduction_pct=reduction_pct,
            noise_reduction_pct=noise_red,
            gain_min=g_min,
            gain_max=g_max,
            frames_count=n,
            plate_w=w,
            plate_h=h,
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        return cur_frames, combined_gains, report, residuals

    @staticmethod
    def gain_map_size(w: int, h: int) -> Tuple[int, int]:
        """Stored gain-map size (width <= 1024), as written by export_gain_maps."""
        gw = min(1024, w)
        return gw, max(1, round(h * gw / w))

    @classmethod
    def quantize_gain(cls, gain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(stored low-res half map as float32, full-res gain the Nuke Group reproduces from it).

        The Group does Read(gain map) -> Reformat (cubic) -> Merge multiply. The engine applies exactly
        this reconstructed gain, so `<plate>_deflicker` == Group output (SPEC 4c.4, < 1e-3)."""
        h, w = gain.shape[:2]
        gw, gh = cls.gain_map_size(w, h)
        low = cv2.resize(gain, (gw, gh), interpolation=cv2.INTER_AREA) if (gw, gh) != (w, h) else gain
        low = low.astype(np.float16).astype(np.float32)
        return low, cls.upscale_gain(low, w, h)

    @staticmethod
    def upscale_gain(low: np.ndarray, w: int, h: int) -> np.ndarray:
        """Full-res gain from a stored low-res map (the Group's Reformat, cubic)."""
        low = low.astype(np.float32)
        full = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) if low.shape[:2] != (h, w) else low
        return full.astype(np.float32)

    @staticmethod
    def export_gain_maps(
        gain_maps: List[np.ndarray],
        first_frame: int,
        output_dir: Path
    ) -> str:
        """Exports gain maps as low-resolution OpenEXR sequence for Nuke Group."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Store at 1024 width for compact file size and fast I/O
        write_flags = [
            cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP
        ]
        for idx, g in enumerate(gain_maps):
            fn = first_frame + idx
            out_file = output_dir / f"gain_{fn:04d}.exr"
            low, _ = PlateCleaner.quantize_gain(g)
            cv2.imwrite(str(out_file), low, write_flags)

        pattern = str(output_dir / "gain_%04d.exr").replace("\\", "/")
        return pattern
