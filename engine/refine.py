# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/refine.py - "Przelicz finalnie" (SPEC 7.3a, 5.4, 5.5) – edge on the whole shot from the finished core.

Pipeline order (7.7 pkt 5): (1) core (segment.py) -> (2) zones -> (3) edge position / shrink-grow
-> (4) edge mode inside the edge band -> (5) temporal smoothing -> (6) alpha hygiene -> QA -> export.
This module does (4)-(5) and the bookkeeping:
- matanyone: one continuous MatAnyone2RunKeyframes pass per crop segment (every key user+auto + brush-correction frames; the node keeps <= 3 permanent,
  start = start key, bidirectional, max_mem_frames 5) -> MatAnyone2CoreCombine (fills false holes) ->
  limited to the edge band (outside: core) -> temporal median with motion_limit;
- core: antialiased SAM 3.1 core (no MatAnyone2 -> no flicker);
- crop-segment overlaps: linear alpha crossfade over the 8 frames;
- ma_scale: object option ma_scale_mode auto | 1.0 | 0.75 | 0.5. auto = 9-frame test batch at 1.0 (crop > 2048 px:
  0.75, > 2600 px: 0.5), VRAM from /system_stats + GPU memory spilled to system RAM (Windows counter); > 11 GB or
  > 0.5 GB spilled -> next scale (logged); a fixed value skips the probe (VRAM still logged per segment);
  Comfy down / OOM on a MatAnyone2 crop -> the error names the crop size and the scale to set;
- MatAnyone2RunKeyframes >= 1.1.0 (installed node checked via /object_info): vram_release_frac (conditional
  empty_cache) and max_mem_frames 0 = auto from defaults.json "matanyone"; an older node gets the old inputs;
- each run -> runs/v###/ (alpha per crop segment rgb=a half ZIP, core PNG, run.json).
Comfy input files (engine/comfy_io.py): "exr" = float32 EXR + PlateLoadEXR / PlateLoadMaskSeq -> PlateSaveEXR
(unchanged); "png8" (8-bit sources, MatAnyone2 only) = 8-bit PNG folders + VHS_LoadImagesPath / ImageToMask ->
SaveImage (8-bit alpha PNG, downloaded). Backlit (scene-linear plate) and LTX-2.5 (user's workflow) stay EXR.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.comfy_io import job_progress
from engine.crop import CropSegment, extract_crop, paste_crop
from engine.depth import frame_path
from engine.segment import sdf

SMOOTHING = {"weak": (1, 0.5), "normal": (2, 0.35), "strong": (4, 0.25)}   # (radius, motion_limit) – SPEC 1.2
# fixed band half-widths per edge mode until the adaptive band (7.8 pkt 2, E5b); x band_scale
BAND_PX = {"matanyone": 6.0, "core": 2.0, "backlit": 40.0, "ltx": 16.0}
VRAM_LIMIT_GB = 11.0
SPILL_LIMIT_GB = 0.5   # GPU memory moved to system RAM during the probe (Windows WDDM) above this = scale too big
TEST_BATCH = 9
PROBE_SCALES = (1.0, 0.75, 0.5)
# auto probe start by crop side (sh060 2026-10-18: a 2784 px crop probed at 1.0 aborted the whole Comfy process)
PROBE_START_BY_SIDE = ((2600, 0.5), (2048, 0.75))
MA_SCALE_MODES = ("auto", "1.0", "0.75", "0.5")
MA_NODE_DEFAULTS = {"vram_release_frac": 0.85, "max_mem_frames": 0,   # defaults.json "matanyone" overrides
                    # node >= 1.2.0 (2026-10-16): soft re-anchoring on the non-start key frames (were as hard as SAM3)
                    "keyframe_anchor": "soft", "keyframe_edge_px": 6}
ZONE_FEATHER_PX = 8.0   # soft-zone override feathered inside its area (PLAN_HAIR_TOOL step 4b)


def ma_node_defaults() -> Dict[str, Any]:
    """defaults.json -> "matanyone": {vram_release_frac, max_mem_frames (0 = auto)}."""
    out = dict(MA_NODE_DEFAULTS)
    try:
        d = json.loads((Path(__file__).resolve().parent.parent / "defaults.json").read_text(encoding="utf-8"))
        out.update({k: v for k, v in (d.get("matanyone") or {}).items() if k in MA_NODE_DEFAULTS})
    except (OSError, ValueError):
        pass
    return out


def node_input_names(info: Any, node_class: str) -> set:
    """Input names of a node from GET /object_info/<class> ({class: {"input": {required, optional}}}) or its inner dict."""
    if not isinstance(info, dict):
        return set()
    node = info.get(node_class, info)
    inp = (node or {}).get("input") or {}
    names = set()
    for sect in ("required", "optional"):
        names |= set((inp.get(sect) or {}).keys())
    return names


def ma_extra_inputs(client: Any, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """New MatAnyone2RunKeyframes inputs, only when the INSTALLED node has them (Comfy rejects unknown inputs and an
    old node has max_mem_frames min 2): vram_release_frac + max_mem_frames (0 = auto). Else {} = old graph."""
    try:
        names = node_input_names(client.get_object_info("MatAnyone2RunKeyframes"), "MatAnyone2RunKeyframes")
    except Exception:
        names = set()
    if "vram_release_frac" not in names:
        return {}
    o = {**ma_node_defaults(), **(opts or {})}
    out = {"vram_release_frac": float(o["vram_release_frac"]), "max_mem_frames": int(o["max_mem_frames"])}
    if "alpha_only" in names:   # node >= 1.1.1: no matte_video / foreground tensors (2 x N x H x W x 3 float32 of RAM)
        out["alpha_only"] = True
    if "keyframe_anchor" in names:   # node >= 1.2.0: soft re-anchoring on the non-start key frames
        out["keyframe_anchor"] = str(o["keyframe_anchor"])
        out["keyframe_edge_px"] = int(o["keyframe_edge_px"])
    return out


def core_soft_supported(client: Any) -> bool:
    """MatAnyone2CoreCombine >= 1.1.2 has `core_mode` (soft start of the band from SAM 3.1 probabilities)."""
    try:
        return "core_mode" in node_input_names(client.get_object_info("MatAnyone2CoreCombine"), "MatAnyone2CoreCombine")
    except Exception:
        return False


def core_soft_start_on(obj: Dict[str, Any]) -> bool:
    """Object option `core_soft_start` (default on): MatAnyone2's core starts from the SAM 3.1 probabilities where maps exist."""
    v = obj.get("core_soft_start")
    return True if v is None else bool(v)


def key_band_same_on(obj: Dict[str, Any]) -> bool:
    """Object option `key_band_same` (default on, user decision 2026-10-15): key frames get the same edge band as the
    other frames instead of the spec's +-3 px (no hard one-frame jump of the edge at every key)."""
    v = obj.get("key_band_same")
    return True if v is None else bool(v)


def probe_start_scale(side: int) -> float:
    """First scale of the auto probe for a crop of `side` px: > 2600 -> 0.5, > 2048 -> 0.75, else 1.0."""
    for limit, scale in PROBE_START_BY_SIDE:
        if side > limit:
            return scale
    return 1.0


def comfy_failure_hint(err: str, side: int, scale: float) -> str:
    """What to change when Comfy died / ran out of memory on a MatAnyone2 crop of `side` px ('' = nothing to add)."""
    if not any(t in err for t in ("nie odpowiada", "pamięci GPU", "pamięci RAM", "zadanie utracone")):
        return ""
    lower = [s for s in PROBE_SCALES if s < scale]
    if not lower:
        return f" – kadr {side} px już przy MatAnyone2 scale {scale}: skróć zakres pracy albo zamknij inne programy na GPU"
    return (f" – kadr {side} px przy MatAnyone2 scale {scale}: ustaw MatAnyone2 scale "
            f"{' / '.join(str(s) for s in lower)} (etap MatAnyone2 → „Skala MatAnyone2”) i policz ponownie")


def ma_scale_mode(obj: Dict[str, Any]) -> str:
    m = str(obj.get("ma_scale_mode") or "auto")
    return m if m in MA_SCALE_MODES else "auto"


def edge_band(core: np.ndarray, width_px: float) -> np.ndarray:
    """|SDF(core)| <= width -> band where the edge mode may act."""
    return np.abs(sdf(core)) <= max(0.5, width_px)


def antialiased_core(core: np.ndarray, soft_px: float = 1.0) -> np.ndarray:
    """`core` edge mode: antialiased core (SDF ramp of +-soft_px)."""
    return np.clip(sdf(core) / (2 * soft_px) + 0.5, 0.0, 1.0).astype(np.float32)


def _median_odd(win: np.ndarray) -> np.ndarray:
    """Element-wise median of an odd number of frames (axis 0) – the middle element of the sorted values, which is
    exactly what np.median returns for an odd count (bit-identical), via min/max sorting networks for 3 and 5
    (no partition along the slow axis: 4x faster on 279 x 704^2); other odd counts -> the middle row of np.sort
    along axis 0 (the same selected value as np.median, ~20 % faster on 11 x 6 Mpx, 2026-10-25)."""
    n = win.shape[0]
    mn, mx = np.minimum, np.maximum
    if n == 3:
        a, b, c = win
        return mx(mn(a, b), mn(mx(a, b), c))
    if n == 5:
        a, b, c, d, e = win
        # 5-element median network (7 comparators), values are only selected -> exact
        f, g = mn(a, b), mx(a, b)
        h, i = mn(c, d), mx(c, d)
        # min(f, h) is the minimum and max(g, i) the maximum of {a, b, c, d}: neither can be the median of five
        m1, m2 = mx(f, h), mn(g, i)          # the two middle candidates of {a, b, c, d}
        p, q = mn(m1, m2), mx(m1, m2)
        return mx(p, mn(q, e))               # median of {p, q, e} with p <= q
    if n % 2 == 1:
        return np.sort(win, axis=0)[n // 2]
    return np.median(win, axis=0)


# ----------------------------------------------------------------------------- temporal smoothing (sliding window)
# The three smoothers below (plan docs/PLAN_EDGE_SPEED.md A1-A3, 2026-10-25) take their frames from any sequence –
# an (n, h, w) stack as before, or a lazy `Sequence` / list-like whose __getitem__ computes a frame on demand – and
# keep in RAM only the window a frame needs (+-radius), so a 4K zone no longer holds four full stacks (41 GB on
# sh060). With `emit(i, frame)` the result is handed out frame by frame instead of a returned stack; `progress(i, n)`
# is called once per frame (the job's cancel check lives there). The per-pixel maths is unchanged: every value is
# computed by the same calls on the same inputs, so the results are bit-identical to the former stack versions
# (tests/test_temporal_window_identity.py against the frozen copy tests/_ref_temporal_old.py).


def flow_workers() -> int:
    """Threads for the DIS flows / warps of the flow smoothers: config `flow_workers`, else min(8, half the CPUs).
    DIS runs on one thread per call and gives identical bits for any thread count (measured 2026-10-25), so the
    pairs of a window are computed side by side; 0 / 1 = sequential (tests, small machines)."""
    try:
        from engine.config import load_config
        n = int(load_config().get("flow_workers") or 0)
    except Exception:
        n = 0
    if n <= 0:
        n = max(1, min(8, (os.cpu_count() or 2) // 2))
    return n


class _FrameCache:
    """Frames of a source computed once and kept until `drop_below` says no later output needs them. The getter is
    called at most once per index (a race between two workers only wastes one computation of the same value)."""

    def __init__(self, get: Callable[[int], np.ndarray]):
        self._get = get
        self._d: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def __call__(self, i: int) -> np.ndarray:
        with self._lock:
            v = self._d.get(i)
        if v is None:
            v = self._get(i)
            with self._lock:
                v = self._d.setdefault(i, v)
        return v

    def drop_below(self, lo: int) -> None:
        with self._lock:
            for k in [k for k in self._d if k < lo]:
                del self._d[k]


class _FlowPool:
    """DIS flows between frame pairs, computed ahead in a thread pool (one DIS instance per thread) and cached until
    `drop_below`: pair (i, j) with a frame below the window of the current centre is never used again. Sequential
    with workers <= 1 (identical values: DIS is deterministic for any thread count)."""

    def __init__(self, u8: Callable[[int], np.ndarray], preset: int, workers: int):
        self.u8 = u8
        self.preset = preset
        self.workers = int(workers)
        self.ex = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self._f: Dict[Tuple[int, int], Any] = {}
        self._local = threading.local()

    def dis(self, shape: Tuple[int, ...]) -> Any:
        """One DIS instance per thread AND per image size: OpenCV's DIS keeps buffers between calls, and a call after
        one at another size returns a different flow (0.27 px on the test scene, 2026-10-25) – per-size instances make
        every flow equal to a fresh instance's, whatever ran before."""
        d = getattr(self._local, "dis", None)
        if d is None:
            d = self._local.dis = {}
        k = tuple(shape[:2])
        if k not in d:
            d[k] = cv2.DISOpticalFlow_create(self.preset)
        return d[k]

    def _calc(self, i: int, j: int) -> np.ndarray:
        a = self.u8(i)
        return self.dis(a.shape).calc(a, self.u8(j), None)

    def request(self, pairs: Sequence[Tuple[int, int]]) -> None:
        if self.ex is None:
            return
        for k in pairs:
            if k not in self._f:
                self._f[k] = self.ex.submit(self._calc, *k)

    def get(self, i: int, j: int) -> np.ndarray:
        v = self._f.get((i, j))
        if v is None:
            v = self._calc(i, j)
        elif not isinstance(v, np.ndarray):
            v = v.result()
        self._f[(i, j)] = v
        return v

    def drop_below(self, lo: int) -> None:
        for k in [k for k in self._f if min(k) < lo]:
            v = self._f.pop(k)
            if not isinstance(v, np.ndarray):
                v.cancel()

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """`fn` over `items`, in the pool when there is one (warps release the GIL), else in order."""
        if self.ex is None:
            return [fn(x) for x in items]
        return list(self.ex.map(fn, items))

    def close(self) -> None:
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None
        self._f.clear()


def _window(i: int, n: int, radius: int) -> List[int]:
    return [min(n - 1, max(0, j)) for j in range(i - radius, i + radius + 1)]


def _pairs(i: int, n: int, radius: int) -> List[Tuple[int, int]]:
    out = []
    for j in _window(i, n, radius):
        if j != i:
            out += [(i, j), (j, i)]
    return out


def _by_rows(fn: Callable[[int, int], np.ndarray], n_rows: int, pool: Optional["_FlowPool"]) -> np.ndarray:
    """`fn(y0, y1)` = the per-pixel maths of one frame on the rows y0..y1 (median, gate, weights: every output pixel
    depends only on its own column of the window), run on row chunks in the pool's threads (numpy releases the GIL
    on large arrays) and concatenated – the same operations in the same per-pixel order, so identical to one call
    over all rows (11 x 13 Mpx: 3.5 s -> 1.1 s, 2026-10-25). Sequential without a pool."""
    k = pool.workers if pool is not None and pool.ex is not None else 1
    if k <= 1 or n_rows < 2 * k:
        return fn(0, n_rows)
    b = np.linspace(0, n_rows, k + 1).astype(int)
    return np.concatenate(pool.map(lambda c: fn(int(b[c]), int(b[c + 1])), range(k)), axis=0)


def _u8_cache(grays: Sequence[np.ndarray]) -> _FrameCache:
    return _FrameCache(lambda i: cv2.normalize(np.asarray(grays[i]).astype(np.float32), None, 0, 255,
                                               cv2.NORM_MINMAX).astype(np.uint8))


def temporal_median(alphas: Sequence[np.ndarray], radius: int, motion_limit: float,
                    band: Optional[Sequence[np.ndarray]] = None,
                    progress: Optional[Callable[[int, int], None]] = None,
                    emit: Optional[Callable[[int, np.ndarray], None]] = None,
                    workers: Optional[int] = None) -> Optional[np.ndarray]:
    """Per-pixel median over +-radius frames, skipped where the alpha range in the window is large (real
    motion -> no ghosts, lesson MBA) – soft gate like CoreCombine's motion_limit. Only inside `band` if given.
    `alphas` / `band`: (n, h, w) stacks or lazy sequences; returns the (n, h, w) result, or None when `emit` takes the
    frames one by one (only the window +-radius is held then). The per-pixel maths runs on row chunks in `workers`
    threads (default flow_workers(); identical values)."""
    n = len(alphas)
    A = _FrameCache(lambda i: np.asarray(alphas[i]))
    P = _FlowPool(lambda i: A(i), 0, flow_workers() if workers is None else int(workers))   # threads only (no flows)
    out = None
    try:
        for i in range(n):
            A.drop_below(i - radius)
            if progress is not None:
                progress(i + 1, n)
            win = np.stack([A(j) for j in _window(i, n, radius)])
            ai = A(i)
            bi = None if band is None else np.asarray(band[i])

            def maths(y0: int, y1: int) -> np.ndarray:
                w_, a_ = win[:, y0:y1], ai[y0:y1]
                med = _median_odd(w_)
                # range over the window incl. frame i (as MatAnyone2CoreCombine): thin parts moving faster than their
                # width must not be mistaken for glitches. Large one-frame jumps are a core problem -> key, not smoothing.
                rng = w_.max(axis=0) - w_.min(axis=0)
                calm = 1.0 - np.clip((rng - motion_limit * 0.6) / (motion_limit * 0.4), 0.0, 1.0)
                sm = a_ + (med - a_) * calm
                return sm if bi is None else np.where(bi[y0:y1], sm, a_)
            res = _by_rows(maths, ai.shape[0], P)
            if emit is not None:
                emit(i, res)
                continue
            if out is None:
                out = np.empty((n,) + res.shape, res.dtype)
            out[i] = res
    finally:
        P.close()
    return out


def _drop_flows(cache: Dict[Tuple[int, int], Any], lo: int) -> None:
    """Frame i of a temporal window needs the pairs (i, j) / (j, i) with |i - j| <= radius; the frames go up, so a pair
    with a frame below i - radius is never used again: dropped (the flows of every pair of a 4K shot do not fit in RAM)."""
    for k in [k for k in cache if min(k) < lo]:
        del cache[k]


def temporal_median_flow(alphas: Sequence[np.ndarray], grays: Sequence[np.ndarray], radius: int = 2,
                         band: Optional[Sequence[np.ndarray]] = None,
                         occl_px: float = 1.0, motion_limit: float = 0.35,
                         preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                         progress: Optional[Callable[[int, int], None]] = None,
                         emit: Optional[Callable[[int, np.ndarray], None]] = None,
                         workers: Optional[int] = None) -> Optional[np.ndarray]:
    """Motion-compensated temporal median (experiment 2026-10-13, ViTMatte zone engine): for frame i the alphas of
    frames i±1..i±radius are warped INTO frame i by DIS optical flow measured on `grays` (same crop as `alphas`,
    float 0..1 or uint8) and the per-pixel median of the ALIGNED window is taken, with the same soft motion gate as
    `temporal_median` (range of the aligned window > motion_limit -> the frame keeps its value: a wrong flow on a
    smooth background, or a real one-frame change, is never "repaired" into a ghost; motion_limit <= 0 = plain
    median). A neighbour whose flow fails the forward/backward check (|f_ij + f_ji(warped)| > occl_px: occlusion /
    unreliable) contributes the centre value instead, so it never votes. Only inside `band` if given. Frames at the
    ends use the available neighbours (window clamped as in temporal_median). Sliding window, flows and warps in a
    thread pool (`workers`, default flow_workers()), `progress` / `emit` as temporal_median."""
    n = len(alphas)
    if n < 3:
        return np.stack([np.asarray(alphas[i]).copy() for i in range(n)]) if emit is None else None
    A = _FrameCache(lambda i: np.asarray(alphas[i]))
    U = _u8_cache(grays)
    F = _FlowPool(U, preset, flow_workers() if workers is None else int(workers))
    out = None
    gx = gy = None
    try:
        for i in range(n):
            lo = i - radius
            A.drop_below(lo)
            U.drop_below(lo)
            F.drop_below(lo)   # pairs no later frame needs (RAM: 4K x2 flows, user report 2026-10-22)
            if progress is not None:
                progress(i + 1, n)
            for j in _window(min(i + 1, n - 1), n, radius):   # grays decoded here (the plate loader is not thread-safe)
                U(j)
            F.request(_pairs(i, n, radius) + (_pairs(i + 1, n, radius) if i + 1 < n else []))
            ai = A(i)
            if gx is None:
                h, w = ai.shape
                gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

            def sample(j: int, i: int = i, ai: np.ndarray = ai) -> np.ndarray:
                if j == i:
                    return ai
                fij, fji = F.get(i, j), F.get(j, i)
                mx, my = gx + fij[..., 0], gy + fij[..., 1]
                warped = cv2.remap(A(j), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                back = cv2.remap(fji, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                occl = np.linalg.norm(fij + back, axis=2) > occl_px
                return np.where(occl, ai, warped).astype(np.float32)
            win = np.stack(F.map(sample, _window(i, n, radius)))
            bi = None if band is None else np.asarray(band[i])

            def maths(y0: int, y1: int) -> np.ndarray:
                w_, a_ = win[:, y0:y1], ai[y0:y1]
                med = _median_odd(w_)
                if motion_limit > 0:
                    rng = w_.max(axis=0) - w_.min(axis=0)
                    calm = 1.0 - np.clip((rng - motion_limit * 0.6) / (motion_limit * 0.4), 0.0, 1.0)
                    med = a_ + (med - a_) * calm
                return med if bi is None else np.where(bi[y0:y1], med, a_)
            res = _by_rows(maths, ai.shape[0], F)
            if emit is not None:
                emit(i, res)
                continue
            if out is None:
                out = np.empty((n,) + res.shape, res.dtype)
            out[i] = res
    finally:
        F.close()
    return out


TEMPORAL_CONF_MODES = ("drop", "centre")


def _min_eig(g: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smaller eigenvalue of the structure tensor of `g` (0..1): ~0 on a smooth area / a single straight edge (the flow
    there is a guess), large on a corner / texture."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    k = int(round(sigma * 3)) * 2 + 1
    jxx = cv2.GaussianBlur(gx * gx, (k, k), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (k, k), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (k, k), sigma)
    return np.maximum(0.5 * (jxx + jyy - np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)), 0.0).astype(np.float32)


def temporal_mean_flow(alphas: Sequence[np.ndarray], grays: Sequence[np.ndarray], radius: int = 3,
                       sigma_t: Optional[float] = None,
                       tol: float = 0.25, occl_px: float = 1.0,
                       preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                       fb_sigma: float = 0.0, photo_sigma: float = 0.0, tex_k: float = 0.0, conf_mode: str = "drop",
                       multiscale: int = 1,
                       progress: Optional[Callable[[int, int], None]] = None,
                       emit: Optional[Callable[[int, np.ndarray], None]] = None,
                       workers: Optional[int] = None) -> Optional[np.ndarray]:
    """Motion-compensated, median-anchored weighted MEAN (user decision 2026-10-15: stronger temporal stabilisation of the
    ViTMatte zone alpha). As temporal_median_flow the frames i±1..i±radius are warped into frame i by DIS flow (a
    neighbour failing the forward/backward check contributes the centre value); then each aligned sample gets the weight
    gauss(d / sigma_t) * clip(1 - |sample - median of the window| / tol, 0, 1): samples near the ROBUST centre average
    out the per-frame noise, an outlier – garbage present in one frame only, the centre included – gets weight ~0 (no
    smearing of garbage onto the neighbours, no ghost of a real change: the median follows it). The caller clips the
    garbage (levels) BEFORE this, so a zeroed speck is never averaged back in. sigma_t default = radius.

    Flow confidence (experiment 2026-10-15, all off by default = bit-identical): with any of them on, a neighbour enters
    the MEAN with its own warped value and an extra weight c = c_fb * c_photo * c_tex (the median anchor keeps the binary
    fwd/bwd substitution): c_fb = gauss(|f_ij + f_ji| / fb_sigma) (fb_sigma 0 -> binary: 0 above occl_px),
    c_photo = gauss(|gray_i - warped gray_j| / photo_sigma), c_tex = t / (t + tex_k) with t = min eigenvalue of the
    structure tensor of gray_i (a smooth background, where DIS only interpolates, does not vote). conf_mode "drop" =
    the lost weight is dropped (the other samples decide), "centre" = it goes to the frame's own value. multiscale 2 =
    the flow is also measured on the half-size grays (3: also quarter ...), per pixel the one with the smaller fwd/bwd
    error is used (also a confidence mode: with nothing else on, c = the binary fwd/bwd test).
    Sliding window, flows and warps in a thread pool, `progress` / `emit` as temporal_median (2026-10-25)."""
    n = len(alphas)
    if n < 3:
        return np.stack([np.asarray(alphas[i]).copy() for i in range(n)]) if emit is None else None
    if conf_mode not in TEMPORAL_CONF_MODES:
        raise ValueError(f"conf_mode musi być jednym z {TEMPORAL_CONF_MODES}, jest {conf_mode!r}")
    conf = fb_sigma > 0 or photo_sigma > 0 or tex_k > 0 or multiscale > 1
    st = float(sigma_t) if sigma_t else max(0.5, float(radius))     # near-flat window: ~1.5x the median's efficiency
    A = _FrameCache(lambda i: np.asarray(alphas[i]))
    U = _u8_cache(grays)
    F = _FlowPool(U, preset, flow_workers() if workers is None else int(workers))
    h, w = np.asarray(alphas[0]).shape
    # half, quarter ... size grays (multiscale), per level a window cache
    pyr = [_FrameCache(lambda i, k=k: cv2.resize(U(i), (max(8, w >> k), max(8, h >> k)), interpolation=cv2.INTER_AREA))
           for k in range(1, max(1, int(multiscale)))]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fbs: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def fb_err(fij: np.ndarray, fji: np.ndarray) -> np.ndarray:
        back = cv2.remap(fji, gx + fij[..., 0], gy + fij[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.linalg.norm(fij + back, axis=2)

    def flow_ms(i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:   # (flow i->j, its fwd/bwd error), best of 2 scales
        k = (i, j)
        if k not in fbs:
            f1, b1 = F.get(i, j), F.get(j, i)
            e1 = fb_err(f1, b1)
            for lv in pyr:
                sc = np.array([w / lv(0).shape[1], h / lv(0).shape[0]], np.float32)
                hs = lambda a, b: cv2.resize(F.dis(lv(a).shape).calc(lv(a), lv(b), None), (w, h),
                                             interpolation=cv2.INTER_LINEAR) * sc
                f2, b2 = hs(i, j), hs(j, i)
                e2 = fb_err(f2, b2)
                pick = e2 < e1
                f1 = np.where(pick[..., None], f2, f1)
                e1 = np.where(pick, e2, e1)
            fbs[k] = (f1, e1)
        return fbs[k]

    out = None
    try:
        for i in range(n):
            lo = i - radius
            A.drop_below(lo)
            U.drop_below(lo)
            F.drop_below(lo)   # pairs no later frame needs (RAM: 4K x2 flows, user report 2026-10-22)
            _drop_flows(fbs, lo)
            for lv in pyr:
                lv.drop_below(lo)
            if progress is not None:
                progress(i + 1, n)
            for j in _window(min(i + 1, n - 1), n, radius):   # grays decoded here (the plate loader is not thread-safe)
                U(j)
            F.request(_pairs(i, n, radius) + (_pairs(i + 1, n, radius) if i + 1 < n else []))
            ai = A(i)
            tw = [float(np.exp(-0.5 * (d / st) ** 2)) for d in range(-radius, radius + 1)]
            tex = None
            if conf and tex_k > 0:
                t = _min_eig(U(i).astype(np.float32) / 255.0)
                tex = t / (t + float(tex_k))
            if conf:
                samples, conf_w, raw = [], [], []
                for j in _window(i, n, radius):
                    if j == i:
                        samples.append(ai)
                        raw.append(ai)
                        conf_w.append(np.ones((h, w), np.float32))
                        continue
                    fij, err = flow_ms(i, j)
                    mx, my = gx + fij[..., 0], gy + fij[..., 1]
                    warped = cv2.remap(A(j), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    occl = err > occl_px
                    c = (np.exp(-0.5 * (err / fb_sigma) ** 2) if fb_sigma > 0 else (~occl).astype(np.float32))
                    if photo_sigma > 0:
                        gw = cv2.remap(U(j).astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        c = c * np.exp(-0.5 * ((gw - U(i).astype(np.float32)) / (255.0 * photo_sigma)) ** 2)
                    if tex is not None:
                        c = c * tex
                    conf_w.append(c.astype(np.float32))
                    raw.append(warped.astype(np.float32))
                    samples.append(np.where(occl, ai, warped).astype(np.float32))
                win = np.stack(samples)
                rw = np.stack(raw)
                cw = np.stack(conf_w)
                twa = np.asarray(tw, np.float32)[:, None, None]

                def maths(y0: int, y1: int) -> np.ndarray:
                    w_, r_, c_ = win[:, y0:y1], rw[:, y0:y1], cw[:, y0:y1]
                    med = _median_odd(w_)
                    close = np.clip(1.0 - np.abs(r_ - med[None]) / max(tol, 1e-3), 0.0, 1.0)
                    wgt = close * twa * c_
                    if conf_mode == "centre":   # the weight a doubtful neighbour loses goes to the frame's own value
                        ci = radius
                        lost = (twa * (1.0 - c_)).sum(axis=0)
                        wgt[ci] = wgt[ci] + lost * close[ci]
                    s_ = wgt.sum(axis=0)
                    mean = np.where(s_ > 1e-6, (wgt * r_).sum(axis=0) / np.maximum(s_, 1e-6), med)
                    return np.clip(mean, 0.0, 1.0).astype(np.float32)
                res = _by_rows(maths, h, F)
            else:
                def sample(j: int, i: int = i, ai: np.ndarray = ai) -> np.ndarray:
                    if j == i:
                        return ai
                    fij, fji = F.get(i, j), F.get(j, i)
                    mx, my = gx + fij[..., 0], gy + fij[..., 1]
                    warped = cv2.remap(A(j), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    back = cv2.remap(fji, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    occl = np.linalg.norm(fij + back, axis=2) > occl_px
                    return np.where(occl, ai, warped).astype(np.float32)
                win = np.stack(F.map(sample, _window(i, n, radius)))
                twa = np.asarray(tw, np.float32)[:, None, None]

                def maths(y0: int, y1: int) -> np.ndarray:
                    w_ = win[:, y0:y1]
                    med = _median_odd(w_)
                    wgt = np.clip(1.0 - np.abs(w_ - med[None]) / max(tol, 1e-3), 0.0, 1.0) * twa
                    s_ = wgt.sum(axis=0)
                    mean = np.where(s_ > 1e-6, (wgt * w_).sum(axis=0) / np.maximum(s_, 1e-6), med)
                    return np.clip(mean, 0.0, 1.0).astype(np.float32)
                res = _by_rows(maths, h, F)
            if emit is not None:
                emit(i, res)
                continue
            if out is None:
                out = np.empty((n,) + res.shape, res.dtype)
            out[i] = res
    finally:
        F.close()
    return out


# ----------------------------------------------------------------------------- Comfy graph
def matanyone_graph(frames_pattern: str, n: int, keymask_pattern: str, key_indices: Sequence[int], start_idx: int,
                    core_pattern: str, out_pattern: str, ma_ckpt: str = "matanyone2.pth",
                    combine: Optional[Dict[str, Any]] = None,
                    ma_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Crop frames 0..n-1 (display), key masks k_0000..k_{K-1} (one per keyframe, in order), core masks
    c_####.png -> MatAnyone2RunKeyframes -> MatAnyone2CoreCombine -> PlateSaveEXR (alpha, crop size).
    `ma_extra`: inputs of MatAnyone2RunKeyframes >= 1.1.0 (ma_extra_inputs; empty for an older installed node)."""
    load = {"path_pattern": frames_pattern, "crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0, "scale": 1.0,
            "bad_frames": ""}
    cc = {"core_erode_px": 3, "core_feather_px": 2, "gate_px": 12, "gate_low": 0.55, "gate_high": 0.8,
          "temporal_sigma": 0.0, "edge_band_px": 6, "motion_limit": 0.35}
    cc.update(combine or {})
    return {
        "1": {"class_type": "LoadMatAnyone2Model", "inputs": {"checkpoint": ma_ckpt}},
        "2": {"class_type": "PlateLoadEXR", "inputs": {**load, "first_frame": 0, "last_frame": n - 1}},
        "3": {"class_type": "PlateLoadEXR", "inputs": {**load, "first_frame": 0, "last_frame": len(key_indices) - 1}},
        "4": {"class_type": "PlateLoadMaskSeq", "inputs": {"path_pattern": keymask_pattern, "plate_info": ["3", 1],
                                                           "frame_offset": 0}},
        "5": {"class_type": "PrimitiveString", "inputs": {"value": ", ".join(str(int(k)) for k in key_indices)}},
        "6": {"class_type": "MatAnyone2RunKeyframes", "inputs": {
            "model_pipe": ["1", 0], "images": ["2", 0], "keyframe_masks": ["4", 0], "keyframes": ["5", 0],
            "start_frame": int(start_idx), "warmup_frames": 10, "erode_kernel": 10, "dilate_kernel": 10,
            "max_size": -1, "max_mem_frames": 5, **(ma_extra or {})}},
        "7": {"class_type": "PlateLoadMaskSeq", "inputs": {"path_pattern": core_pattern, "plate_info": ["2", 1],
                                                           "frame_offset": 0}},
        "8": {"class_type": "MatAnyone2CoreCombine", "inputs": {"alpha": ["6", 0], "core_masks": ["7", 0], **cc}},
        "9": {"class_type": "MaskToImage", "inputs": {"mask": ["8", 0]}},
        "10": {"class_type": "PlateSaveEXR", "inputs": {"rgba": ["9", 0], "plate_info": ["2", 1],
                                                        "out_pattern": out_pattern, "half_float": True, "alpha": ["8", 0]}},
    }


def matanyone_graph_png8(frames_dir: str, keys_dir: str, key_indices: Sequence[int], start_idx: int, core_dir: str,
                         prefix: str, ma_ckpt: str = "matanyone2.pth",
                         combine: Optional[Dict[str, Any]] = None,
                         ma_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """matanyone_graph for 8-bit sources: frames / key masks / core masks = three PNG folders (VHS_LoadImagesPath,
    masks via ImageToMask red); alpha -> SaveImage (8-bit PNG, crop size, frame order)."""
    from engine.comfy_io import image_to_mask, vhs_load
    g = matanyone_graph("", 1, "", key_indices, start_idx, "", "", ma_ckpt, combine, ma_extra)
    g["2"] = vhs_load(frames_dir)
    g["3"] = vhs_load(keys_dir)
    g["4"] = image_to_mask(["3", 0])
    g["11"] = vhs_load(core_dir)
    g["7"] = image_to_mask(["11", 0])
    g["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}}
    return g


def backlit_graph(backlit_pattern: str, n: int, core_pattern: str, out_pattern: str, params: Dict[str, Any],
                  grow_px: int = 40, soft_px: float = 12.0, wall_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Crop frames of the (aligned) backlit plate, scene-linear -> BacklitCoverage -> MaskRegionLimit(core).
    `wall_pattern` (Z25 pkt 1): the wall estimated on the FULL X2 frame (engine/backlit_wall), cut to the crop, as
    gray EXR -> the node's optional `wall` input (its own crop-based estimate is skipped)."""
    load = {"path_pattern": backlit_pattern, "first_frame": 0, "last_frame": n - 1, "crop_x": 0, "crop_y": 0,
            "crop_w": 0, "crop_h": 0, "scale": 1.0, "bad_frames": ""}
    bc = {"wall_lo": params.get("wall_lo", 0.9), "wall_sigma_small": params.get("wall_sigma_small", 12.0),
          "wall_sigma_large": params.get("wall_sigma_large", 200.0), "max_radius": params.get("max_radius", 40),
          "floor": params.get("floor", 0.5), "noise_lo": params.get("noise_lo", 0.05),
          "fill_threshold": params.get("fill_ratio", 0.6), "fill_erode": params.get("fill_radius", 15)}
    g = {
        "1": {"class_type": "PlateLoadEXR", "inputs": load},
        "2": {"class_type": "BacklitCoverage", "inputs": {"backlit": ["1", 0], **bc}},
        "3": {"class_type": "PlateLoadMaskSeq", "inputs": {"path_pattern": core_pattern, "plate_info": ["1", 1],
                                                           "frame_offset": 0}},
        "4": {"class_type": "MaskRegionLimit", "inputs": {"mask": ["2", 0], "region": ["3", 0], "grow_px": grow_px,
                                                          "soft_px": soft_px, "grow_into_tail_only": True}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "PlateSaveEXR", "inputs": {"rgba": ["5", 0], "plate_info": ["1", 1],
                                                       "out_pattern": out_pattern, "half_float": True, "alpha": ["4", 0]}},
    }
    if wall_pattern:
        g["7"] = {"class_type": "PlateLoadEXR", "inputs": {**load, "path_pattern": wall_pattern}}
        g["2"]["inputs"]["wall"] = ["7", 0]
    return g


def ltx_batches(n: int, max_len: int = 25, overlap: int = 8) -> List[Tuple[int, int]]:
    """Index ranges of 8k+1 frames (LTX-2.5), consecutive batches overlapping by `overlap` frames."""
    max_len = max(9, (max_len - 1) // 8 * 8 + 1)
    if n <= max_len:
        L = (n - 1) // 8 * 8 + 1
        if L == n:
            return [(0, n - 1)]
        return [(0, L - 1), (n - L, n - 1)] if L >= overlap + 1 else [(0, n - 1)]
    out, s = [], 0
    while True:
        e = s + max_len - 1
        if e >= n - 1:
            out.append((n - max_len, n - 1))
            break
        out.append((s, e))
        s = e - overlap + 1
    return out


def ltx_side(side: int, max_side: int = 1088, snap: int = 32) -> int:
    """LTX-2.5 input side: <= 1088 (downscale only for LTX-2.5), multiple of 32."""
    return max(snap, min(max_side, side) // snap * snap)


def region_limit(alpha: np.ndarray, core: np.ndarray, grow_px: float = 40.0, soft_px: float = 12.0) -> np.ndarray:
    """numpy MaskRegionLimit: keep the matte only inside soft(dilate(core region))."""
    d = sdf(core)
    reg = np.clip((d + grow_px) / max(soft_px, 1e-3) + 0.5, 0.0, 1.0)
    return (alpha * reg).astype(np.float32)


def pick_keyframes(keys: Sequence[int], start: int, max_keys: Optional[int] = None) -> List[int]:
    """Keyframes of one MatAnyone2RunKeyframes pass: the start key + the nearest others (sorted). Default (2026-10-15,
    user: "set it up to work best"): EVERY key of the segment – the node itself keeps at most 3 of them per pass in
    permanent memory (VRAM, SPEC 5.4/13: the first ones met from the start = the nearest) and re-anchors on the others
    through its working memory, so more keys never cost VRAM. `max_keys` = the old cap (3) when needed."""
    others = sorted((k for k in keys if k != start), key=lambda k: abs(k - start))
    if max_keys is not None:
        others = others[: max(0, max_keys - 1)]
    return sorted([start] + others)


# ----------------------------------------------------------------------------- run bookkeeping
def next_run_dir(obj_dir: Path) -> Path:
    runs = obj_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    from engine.run_trash import last_run_number
    n = last_run_number(runs) + 1      # deleted versions keep their numbers (engine/run_trash.py)
    d = runs / f"v{n:03d}"
    d.mkdir()
    return d


def write_alpha_exr(path: Path, a: np.ndarray) -> None:
    from engine.exr_io import write_gray_exr
    write_gray_exr(path, np.clip(a, 0, 1), half=True, channels=("R", "G", "B"))


def read_alpha(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else m[..., 2]
    return m.astype(np.float32)


@dataclass
class FinalizeReport:
    run: str
    edge_mode: str
    segments: List[str]
    ma_scale: float = 1.0
    ma_scale_mode: str = "auto"
    ma_node: Dict[str, Any] = field(default_factory=dict)   # new node inputs sent (empty = old installed node)
    vram_gb: Optional[float] = None
    vram_log: List[str] = field(default_factory=list)
    smoothing: str = "normal"
    band_px: float = 6.0
    keys_used: List[int] = field(default_factory=list)
    hygiene: Dict[str, int] = field(default_factory=dict)
    soft_start: Dict[str, Any] = field(default_factory=dict)   # SAM3 probabilities as the CoreCombine core (frames, node)
    seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _vram_used_gb(client: Any) -> Optional[float]:
    try:
        st = client.get_system_stats() or {}
        dev = (st.get("devices") or [{}])[0]
        if "torch_vram_total" in dev and "torch_vram_free" in dev:
            return (dev["torch_vram_total"] - dev["torch_vram_free"]) / 2 ** 30
        if "vram_total" in dev and "vram_free" in dev:
            return (dev["vram_total"] - dev["vram_free"]) / 2 ** 30
    except Exception:
        pass
    return None


def _comfy_shared_reader(client: Any) -> Optional[Callable[[], Optional[float]]]:
    """Reader of the Comfy process' shared (spilled) GPU memory – engine.sysinfo.comfy_shared_reader (one reader for the
    MatAnyone2 scale probe and the spill warning of every Comfy prompt, 2026-10-26)."""
    from engine.sysinfo import comfy_shared_reader
    return comfy_shared_reader(client)


def write_run_json(run_dir: Path, rj: Dict[str, Any]) -> None:
    """The one run.json writer of a version: + `gpu_spill_gb` = the running job's peak of GPU memory spilled to RAM
    when it reached comfy_client.SPILL_WARN_GB (user request 2026-10-26: why a run took 10x longer)."""
    from comfy_client import SPILL_WARN_GB
    from engine.jobs import JOBS
    sp = JOBS.spill_gb()
    out = {**rj, **({"gpu_spill_gb": round(sp, 1)} if sp is not None and sp >= SPILL_WARN_GB else {})}
    (Path(run_dir) / "run.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def contiguous_chunks(frames: Sequence[int]) -> List[List[int]]:
    """Frames split where they are not consecutive (gaps = the object outside its life range)."""
    out: List[List[int]] = []
    for f in frames:
        if out and f == out[-1][-1] + 1:
            out[-1].append(f)
        else:
            out.append([f])
    return out or [[]]


class Finalizer:
    """Edge mode over the whole shot. `load_crop_image(frame, seg, scale)` -> display crop image."""

    def __init__(self, client: Any, io_dir: Path, plate_w: int, plate_h: int,
                 load_crop_image: Callable[[int, CropSegment, float], np.ndarray], ma_ckpt: str = "matanyone2.pth",
                 load_backlit_crop: Optional[Callable[[int, CropSegment], np.ndarray]] = None,
                 backlit_params: Optional[Dict[str, Any]] = None, ltx_workflow: Optional[Dict[str, Any]] = None,
                 ltx_max_frames: int = 25, io_mode: str = "exr", ma_node_opts: Optional[Dict[str, Any]] = None,
                 load_backlit_full: Optional[Callable[[int], np.ndarray]] = None):
        self.client = client
        self.ma_node_opts = ma_node_opts
        # extra MatAnyone anchors (frame -> mask), not keys anywhere else (band, hardness): the key frames of the
        # stage-2 brush corrections with the mask 1+2 there (pipeline._brush_anchors, 2026-10-15)
        self.extra_anchors: Dict[int, np.ndarray] = {}
        self._ma_extra: Optional[Dict[str, Any]] = None
        self._soft_ok: Optional[bool] = None
        self.io_mode = io_mode
        self.io = Path(io_dir)
        self.w, self.h = plate_w, plate_h
        self.load = load_crop_image
        self.ma_ckpt = ma_ckpt
        self.load_backlit = load_backlit_crop
        self.load_backlit_full = load_backlit_full     # Z25: full X2 frame (scene-linear BGR) for the wall estimate
        self.backlit_params = backlit_params or {}
        self.ltx_workflow = ltx_workflow
        self.ltx_max_frames = ltx_max_frames
        # GPU memory of the Comfy process spilled to system RAM (GB) – read during the auto probe; None = not measured
        self.shared_gb: Optional[Callable[[], Optional[float]]] = _comfy_shared_reader(client)

    def _backlit_segment(self, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray],
                         grow_px: float) -> Dict[int, np.ndarray]:
        if self.load_backlit is None:
            raise RuntimeError("tryb backlit: brak (wyrównanego) plate'u backlit w projekcie")
        job = _mk(self.io / f"bl_{seg.seg_id}")
        # Z25 pkt 1: the wall estimated on the FULL X2 frame (no mirror padding, nothing outside the frame votes),
        # cut to the crop and handed to the node; "crop" = the old in-node estimate on the mirrored crop
        wall_ctx = str(self.backlit_params.get("wall_context", "full"))
        use_wall = wall_ctx == "full" and self.load_backlit_full is not None
        if use_wall:
            from engine.backlit_wall import wall_full_frame
        for i, f in enumerate(frames):
            cv2.imwrite(str(job / f"b_{i:04d}.exr"), np.ascontiguousarray(self.load_backlit(f, seg), np.float32))
            cv2.imwrite(str(job / f"c_{i:04d}.png"),
                        (extract_crop(core[f].astype(np.float32), seg) > 0.5).astype(np.uint8) * 255)
            if use_wall:
                wall = wall_full_frame(self.load_backlit_full(f), self.backlit_params)
                wc = extract_crop(wall, seg)
                cv2.imwrite(str(job / f"w_{i:04d}.exr"), np.ascontiguousarray(np.repeat(wc[..., None], 3, axis=2), np.float32))
            job_progress("Backlit: zapis klatek do Comfy", i + 1, len(frames))
        out = str(job / "a_####.exr").replace("\\", "/")
        self._run_graph(backlit_graph(str(job / "b_####.exr").replace("\\", "/"), len(frames),
                                      str(job / "c_####.png").replace("\\", "/"), out, self.backlit_params,
                                      grow_px=int(round(grow_px)),
                                      wall_pattern=str(job / "w_####.exr").replace("\\", "/") if use_wall else None))
        return {f: read_alpha(frame_path(out, i)) for i, f in enumerate(frames)}

    def _ltx_segment(self, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """User's LTX-2.5 Alpha Gen workflow (PlateLoadEXR in -> PlateSaveEXR out), batches 8k+1 with 8-frame overlap,
        crop downscaled to <= 1088 px (multiple of 32) for LTX-2.5 only; result region-limited by the core."""
        if not self.ltx_workflow:
            raise RuntimeError("tryb LTX-2.5: brak workflowu LTX-2.5 Alpha Gen (config.json → ltx_alpha_workflow)")
        from engine.comfy_graph import one_node, ui_to_api
        side = seg.side
        ls = ltx_side(side)
        acc = {f: np.zeros((side, side), np.float32) for f in frames}
        wsum = {f: 0.0 for f in frames}
        for bi, (a, b) in enumerate(ltx_batches(len(frames), self.ltx_max_frames)):
            job = _mk(self.io / f"ltx_{seg.seg_id}_{bi:02d}")
            for i, f in enumerate(frames[a:b + 1]):
                img = cv2.resize(self.load(f, seg, 1.0), (ls, ls), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(job / f"f_{i:04d}.exr"), np.ascontiguousarray(img, np.float32))
                job_progress("LTX-2.5: zapis klatek do Comfy", i + 1, b - a + 1)
            g = ui_to_api(self.ltx_workflow)
            _, ld = one_node(g, "PlateLoadEXR")
            _, sv = one_node(g, "PlateSaveEXR")
            ld["inputs"].update({"path_pattern": str(job / "f_####.exr").replace("\\", "/"), "first_frame": 0,
                                 "last_frame": b - a, "crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0})
            sv["inputs"]["out_pattern"] = str(job / "a_####.exr").replace("\\", "/")
            self._run_graph(g)
            n_b = b - a + 1
            for i, f in enumerate(frames[a:b + 1]):
                al = cv2.resize(read_alpha(frame_path(sv["inputs"]["out_pattern"], i)), (side, side),
                                interpolation=cv2.INTER_LINEAR)
                # crossfade weights on batch overlaps (ramps at batch ends)
                wgt = min(1.0, (i + 1) / 8.0, (n_b - i) / 8.0)
                acc[f] += al * wgt
                wsum[f] += wgt
        return {f: region_limit(acc[f] / max(wsum[f], 1e-6), extract_crop(core[f].astype(np.float32), seg) > 0.5)
                for f in frames}

    def ma_extra(self) -> Dict[str, Any]:
        """New MatAnyone2RunKeyframes inputs for the installed node (asked once per Finalizer)."""
        if self._ma_extra is None:
            self._ma_extra = ma_extra_inputs(self.client, self.ma_node_opts)
        return self._ma_extra

    def soft_ok(self) -> bool:
        """Installed CoreCombine accepts core_mode (asked once per Finalizer)."""
        if self._soft_ok is None:
            self._soft_ok = core_soft_supported(self.client)
        return self._soft_ok

    def _run_graph(self, g: Dict[str, Any]) -> Dict[str, Any]:
        busy = self.client.busy_error()
        if busy:
            raise RuntimeError(busy)
        sub = self.client.queue_prompt(g)
        if not sub.get("success"):
            raise RuntimeError(f"Comfy odrzucił graf MatAnyone2: {sub.get('error')}")
        res = self.client.wait_for_prompt(sub["prompt_id"], timeout_sec=7200)
        if not res.get("success"):
            raise RuntimeError(f"MatAnyone2 nie powiódł się: {res.get('error')}")
        return res

    def _release_comfy_cache(self) -> None:
        """After a MatAnyone2 prompt: `POST /free {free_memory}` – Comfy's output cache otherwise keeps every tensor of the
        graph in RAM (frames, core, alpha, matte, fg: 9.5 GB for 279 x 704^2, ~60 GB for 514 x 1440^2) until the next
        prompt evicts it. Models are not unloaded (unload_models False); MatAnyone2's model is reloaded by the next
        prompt (~2 s). Best effort: a failure here never fails the pass."""
        try:
            self.client.free_vram(unload_models=False, free_memory=True)
        except Exception:
            pass

    def _download_masks(self, res: Dict[str, Any], node_id: str) -> List[np.ndarray]:
        """SaveImage outputs of `node_id` (sorted by file name = frame order) -> float 0..1 masks."""
        out = []
        for info in sorted(res.get("outputs", {}).get(node_id, {}).get("images", []), key=lambda i: i["filename"]):
            buf = np.frombuffer(self.client.download_output(info), np.uint8)
            out.append(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
        return out

    def _soft_crop(self, soft: Optional[Dict[int, np.ndarray]], f: int, seg: CropSegment, ss: int) -> Optional[np.ndarray]:
        """SAM 3.1 probability crop of frame f at the MatAnyone2 size (float 0..1), or None (no map / no soft start)."""
        if not soft or f not in soft:
            return None
        p = soft[f]
        p = p.astype(np.float32) / 255.0 if p.dtype == np.uint8 else np.clip(p, 0, 1).astype(np.float32)
        return cv2.resize(extract_crop(p, seg), (ss, ss), interpolation=cv2.INTER_LINEAR)

    def _matanyone_segment(self, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray],
                           keys: Dict[int, np.ndarray], start: int, scale: float, tag: str,
                           soft: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """One MatAnyone2 pass on a crop segment; Comfy down / out of memory -> the error names the crop and the scale to set."""
        try:
            return self._matanyone_segment_run(seg, frames, core, keys, start, scale, tag, soft)
        except RuntimeError as e:
            hint = comfy_failure_hint(str(e), seg.side, scale)
            if not hint:
                raise
            raise RuntimeError(f"{e}{hint}") from e

    def _matanyone_segment_run(self, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray],
                               keys: Dict[int, np.ndarray], start: int, scale: float, tag: str,
                               soft: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """`soft` (frame -> SAM 3.1 probability map, full frame, consistent with `core`): soft start of the band – those
        frames' core files carry the probabilities and CoreCombine runs with core_mode=soft (the caller checks the
        installed node). Without it: the old graph and files bit for bit."""
        job = _mk(self.io / f"ma_{seg.seg_id}_{tag}")
        side = seg.side
        ss = max(16, int(round(side * scale)))
        if self.io_mode == "png8":
            from engine.comfy_io import require_vhs
            require_vhs(self.client)
            return self._matanyone_segment_png8(job, seg, frames, core, keys, start, scale, ss, soft)
        from engine.comfy_io import write_soft_png
        n_soft = 0
        for i, f in enumerate(frames):
            img = self.load(f, seg, scale)
            cv2.imwrite(str(job / f"f_{i:04d}.exr"), np.ascontiguousarray(img, np.float32))
            pc = self._soft_crop(soft, f, seg, ss)
            if pc is not None:
                write_soft_png(job / f"c_{i:04d}.png", pc)
                n_soft += 1
            else:
                c = cv2.resize(extract_crop(core[f].astype(np.float32), seg), (ss, ss), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(job / f"c_{i:04d}.png"), ((c > 0.5) * 255).astype(np.uint8))
            job_progress("MatAnyone2: zapis klatek do Comfy", i + 1, len(frames))
        self._last_soft = n_soft
        keys = {**(self.extra_anchors or {}), **keys}   # brush-correction frames anchor MatAnyone too (keys win)
        seg_keys = [k for k in keys if k in frames]
        if start not in seg_keys:  # segment without the start key: its first frame's core acts as the key
            start = min(seg_keys, key=lambda k: abs(k - start)) if seg_keys else frames[0]
            if start not in keys:
                keys = {**keys, start: core[start]}
                seg_keys.append(start)
        kf = pick_keyframes(seg_keys, start)
        for j, k in enumerate(kf):
            km = cv2.resize(extract_crop(keys[k].astype(np.float32), seg), (ss, ss), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(job / f"k_{j:04d}.png"), ((km > 0.5) * 255).astype(np.uint8))
        idx = [frames.index(k) for k in kf]
        out_pat = str(job / "a_####.exr").replace("\\", "/")
        g = matanyone_graph(str(job / "f_####.exr").replace("\\", "/"), len(frames),
                            str(job / "k_####.png").replace("\\", "/"), idx, frames.index(start),
                            str(job / "c_####.png").replace("\\", "/"), out_pat, self.ma_ckpt,
                            combine={"core_mode": "soft"} if n_soft else None, ma_extra=self.ma_extra())
        self._run_graph(g)
        self._release_comfy_cache()
        res = {}
        for i, f in enumerate(frames):
            a = read_alpha(frame_path(out_pat, i))
            if a.shape[0] != side:  # ma_scale < 1: alpha back up (bilinear); core/band stay full resolution
                a = cv2.resize(a, (side, side), interpolation=cv2.INTER_LINEAR)
            res[f] = a
        self._last_keys = kf
        return res

    def _matanyone_segment_png8(self, job: Path, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray],
                                keys: Dict[int, np.ndarray], start: int, scale: float, ss: int,
                                soft: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """8-bit sources: frames / keys / core in separate PNG folders (VHS reads a whole folder), alpha back as PNG."""
        import shutil
        import uuid
        from engine.comfy_io import write_mask_png, write_png8, write_soft_png
        side = seg.side
        for sub in ("frames", "keys", "core"):   # VHS reads the whole folder: no frames of an earlier job may stay
            shutil.rmtree(job / sub, ignore_errors=True)
        n_soft = 0
        for i, f in enumerate(frames):
            write_png8(job / "frames" / f"f_{i:04d}.png", self.load(f, seg, scale))
            pc = self._soft_crop(soft, f, seg, ss)
            if pc is not None:
                write_soft_png(job / "core" / f"c_{i:04d}.png", pc)
                n_soft += 1
            else:
                c = cv2.resize(extract_crop(core[f].astype(np.float32), seg), (ss, ss), interpolation=cv2.INTER_LINEAR)
                write_mask_png(job / "core" / f"c_{i:04d}.png", c)
            job_progress("MatAnyone2: zapis klatek do Comfy", i + 1, len(frames))
        keys = {**(self.extra_anchors or {}), **keys}   # brush-correction frames anchor MatAnyone too (keys win)
        seg_keys = [k for k in keys if k in frames]
        if start not in seg_keys:  # segment without the start key: its first frame's core acts as the key
            start = min(seg_keys, key=lambda k: abs(k - start)) if seg_keys else frames[0]
            if start not in keys:
                keys = {**keys, start: core[start]}
                seg_keys.append(start)
        kf = pick_keyframes(seg_keys, start)
        for j, k in enumerate(kf):
            km = cv2.resize(extract_crop(keys[k].astype(np.float32), seg), (ss, ss), interpolation=cv2.INTER_LINEAR)
            write_mask_png(job / "keys" / f"k_{j:04d}.png", km)
        idx = [frames.index(k) for k in kf]
        g = matanyone_graph_png8(str(job / "frames").replace("\\", "/"), str(job / "keys").replace("\\", "/"), idx,
                                 frames.index(start), str(job / "core").replace("\\", "/"),
                                 f"matte_desk/{job.name}_{uuid.uuid4().hex[:8]}/a", self.ma_ckpt,
                                 combine={"core_mode": "soft"} if n_soft else None, ma_extra=self.ma_extra())
        self._last_soft = n_soft
        alphas = self._download_masks(self._run_graph(g), "10")
        self._release_comfy_cache()
        if len(alphas) != len(frames):
            raise RuntimeError(f"MatAnyone2 zwrócił {len(alphas)} klatek zamiast {len(frames)}")
        res = {}
        for f, a in zip(frames, alphas):
            if a.shape[0] != side:  # ma_scale < 1: alpha back up (bilinear); core/band stay full resolution
                a = cv2.resize(a, (side, side), interpolation=cv2.INTER_LINEAR)
            res[f] = a
        self._last_keys = kf
        return res

    def probe_scale(self, seg: CropSegment, frames: List[int], core: Dict[int, np.ndarray],
                    keys: Dict[int, np.ndarray], start: int, rep: FinalizeReport) -> float:
        """9-frame test batch 1.0 -> 0.75 -> 0.5 until VRAM <= 11 GB (SPEC 5.4) and <= 0.5 GB spilled to system RAM.
        Large crops start lower (probe_start_scale): the first trial at 1.0 on a 2784 px crop killed Comfy."""
        from engine.sysinfo import SpillSampler
        test = frames[:TEST_BATCH]
        first = probe_start_scale(seg.side)
        if first < 1.0:
            rep.vram_log.append(f"kadr {seg.side} px: próba od ma_scale {first}")
        idle = None
        if self.shared_gb is not None:   # read once: what an earlier trial left in shared memory is not idle
            try:
                idle = self.shared_gb()
            except Exception:
                idle = None
        for scale in [s for s in PROBE_SCALES if s <= first]:
            t0 = time.time()
            sampler = SpillSampler(self.shared_gb, base=idle) if self.shared_gb is not None else None
            if sampler is not None:
                sampler.__enter__()
            try:
                self._matanyone_segment(seg, test, core, {**keys, test[0]: keys.get(test[0], core[test[0]])},
                                        test[0], scale, f"probe{int(scale * 100)}")
            finally:
                if sampler is not None:
                    sampler.__exit__(None, None, None)
            used = _vram_used_gb(self.client)
            spill = sampler.spill_gb if sampler is not None else None
            rep.vram_log.append(f"ma_scale {scale}: VRAM {used if used is None else round(used, 2)} GB"
                                f"{'' if spill is None else f', przelane do RAM {spill:.2f} GB'}, "
                                f"{time.time() - t0:.1f} s / {len(test)} kl.")
            if (used is None or used <= VRAM_LIMIT_GB) and (spill is None or spill <= SPILL_LIMIT_GB):
                rep.vram_gb = used
                return scale
        return PROBE_SCALES[-1]

    def finalize(self, obj: Dict[str, Any], segs: Sequence[CropSegment], frames: Sequence[int],
                 core: Dict[int, np.ndarray], keys: Dict[int, np.ndarray], start: int, run_dir: Path,
                 edge_mode: str = "matanyone", progress_cb: Optional[Callable[[str, int, int], None]] = None,
                 exclude: Optional[Dict[int, np.ndarray]] = None,
                 soft_override: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
                 band_fn: Optional[Callable[[int, CropSegment, np.ndarray, float], np.ndarray]] = None,
                 soft_feather: float = ZONE_FEATHER_PX,
                 soft_weights: Optional[Dict[int, np.ndarray]] = None,
                 keep_holes: Optional[Dict[int, np.ndarray]] = None,
                 core_soft: Optional[Dict[int, np.ndarray]] = None
                 ) -> Tuple[Dict[int, np.ndarray], FinalizeReport]:
        """`exclude`: full-frame pixels removed by Remove zones (+2 px) – out of the edge band, never restored;
        `soft_override`: full-frame (alpha, area) of soft zones – applied last, wins over everything in its area;
        `core_soft`: SAM 3.1 probabilities (uint8, consistent with `core`) of the frames that have them – soft start of the
        band in matanyone mode when the installed CoreCombine has core_mode (else binary as before, noted in the report)."""
        t0 = time.time()
        frames = list(frames)
        smoothing = obj.get("temporal_smoothing", "normal")
        radius, mlimit = SMOOTHING.get(smoothing, SMOOTHING["normal"])
        band_px = BAND_PX.get(edge_mode, 6.0) * float(obj.get("band_scale", 1.0))
        rep = FinalizeReport(run=run_dir.name, edge_mode=edge_mode, segments=[s.seg_id for s in segs],
                             smoothing=smoothing, band_px=band_px)
        # ma_scale_mode (object option): auto = probe (1.0 -> 0.75 -> 0.5 by VRAM and spill to RAM; crops > 2048 px
        # start at 0.75, > 2600 px at 0.5); a fixed value skips the probe.
        # auto starts from probe_start_scale every time (not from the scale stored by the previous run, so going back from a
        # fixed 0.5 to auto is not stuck at 0.5).
        smode = ma_scale_mode(obj)
        rep.ma_scale_mode = smode
        scale = 1.0 if smode == "auto" else float(smode)
        if edge_mode == "matanyone" and segs:
            rep.ma_node = dict(self.ma_extra())
            if smode == "auto":
                first_seg_frames = contiguous_chunks([f for f in frames if segs[0].first <= f <= segs[0].last])[0]
                scale = self.probe_scale(segs[0], first_seg_frames, core, keys, start, rep)
        rep.ma_scale = scale
        soft_in = core_soft if (edge_mode == "matanyone" and core_soft) else None
        if soft_in is not None and not self.soft_ok():
            rep.soft_start = {"frames": 0, "maps": len(soft_in), "node": False,
                              "note": "węzeł MatAnyone2CoreCombine bez core_mode – zaktualizuj węzły w Doktorze"}
            soft_in = None
        elif soft_in is not None:
            rep.soft_start = {"frames": 0, "maps": len(soft_in), "node": True}

        per_seg: Dict[str, Dict[int, np.ndarray]] = {}
        for n, seg in enumerate(segs):
            sf = [f for f in frames if seg.first <= f <= seg.last]
            # object life range: frames with a gap (object invisible) -> one video per contiguous piece (one piece =
            # the segment's frames, as before)
            chunks = contiguous_chunks(sf)
            if edge_mode == "matanyone":
                t_seg = time.time()
                soft = {}
                for ch in chunks:
                    soft.update(self._matanyone_segment(seg, ch, core, keys, start, scale, "final", soft=soft_in))
                    if soft_in is not None:
                        rep.soft_start["frames"] += int(getattr(self, "_last_soft", 0))
                    rep.keys_used = sorted(set(rep.keys_used) | set(self._last_keys))
                if smode != "auto":   # fixed scale: no probe, VRAM still logged (after each segment)
                    used = _vram_used_gb(self.client)
                    rep.vram_log.append(f"ma_scale {scale} (wybrana, bez próby) {seg.seg_id}: VRAM "
                                        f"{used if used is None else round(used, 2)} GB, {time.time() - t_seg:.1f} s / "
                                        f"{len(sf)} kl.")
                    if used is not None:
                        rep.vram_gb = max(rep.vram_gb or 0.0, used)
            elif edge_mode == "backlit":
                soft = {}
                for ch in chunks:
                    soft.update(self._backlit_segment(seg, ch, core, band_px))
            elif edge_mode == "ltx":
                soft = {}
                for ch in chunks:
                    soft.update(self._ltx_segment(seg, ch, core))
            else:  # core: antialiased SAM3 core
                soft = {f: antialiased_core(extract_crop(core[f].astype(np.float32), seg)) for f in sf}
            # band limit: outside the band the result is the core (MatAnyone cannot add/remove regions);
            # key frames: the same band as every frame (object option key_band_same, user decision 2026-10-15: the
            # spec's +-3 px on keys made the edge jump hard for one frame at every key – CLIP2_A F50 had half the soft
            # pixels of its neighbours); key_band_same=False = the spec rule (+-3 px, no adaptive band)
            key_same = key_band_same_on(obj)
            # Band, temporal median, hygiene, soft zones and the EXR write run per frame through a sliding window
            # (docs/PLAN_EDGE_SPEED.md A1, 2026-10-25): the band composite of a frame is made when the median's window
            # first needs it (the engine's crop is freed then), the finished frame is written at once – no per-segment
            # stacks (3 x 55 MB x frames on a 3728 px crop). Same per-frame calls as before and temporal_median streams
            # bit-identically, so the alphas are unchanged.
            from engine.bandsys import compose_area
            from engine.edge import band_feathers, hygiene
            fe, fo = band_feathers(obj)   # unset = the one defaults of every stage (2026-10-24)
            bands: Dict[int, np.ndarray] = {}
            done: Dict[int, np.ndarray] = {}

            def band_of(f: int) -> np.ndarray:
                b = bands.get(f)
                if b is None:
                    c = extract_crop(core[f].astype(np.float32), seg) > 0.5
                    narrow = f in keys and not key_same
                    bw = min(band_px, 3.0) if narrow else band_px
                    # adaptive band (7.8 pkt 2) for matanyone / core; backlit and LTX are limited by the core region
                    adaptive = band_fn is not None and not narrow and edge_mode in ("matanyone", "core")
                    b = band_fn(f, seg, c, bw) if adaptive else edge_band(c, bw)
                    if exclude and f in exclude:
                        b &= ~(extract_crop(exclude[f].astype(np.float32), seg) > 0.5)
                    bands[f] = b
                return b

            def res_i(i: int) -> np.ndarray:
                f = sf[i]
                c = extract_crop(core[f].astype(np.float32), seg) > 0.5
                band = band_of(f)
                sf_ = soft.pop(f)
                if fe > 0 or fo > 0:
                    # one band system (engine/bandsys.py, 2026-10-22 / one implementation 2026-10-24): the inset feather
                    # from the inner band edge blends the core (1) into the edge result (user request 2026-10-17: no
                    # visible line at the inner band edge), the outer feather from the outer edge blends it into the
                    # core (0 there); they never overlap, the inset one keeps its length up to the band width
                    return compose_area(c, sf_, band, fe, fo)
                return np.where(band, sf_, c.astype(np.float32))

            def emit(i: int, a: np.ndarray) -> None:
                f = sf[i]
                job_progress("Higiena i zapis alphy (EXR)", i + 1, len(sf))
                # (6) alpha hygiene (7.8 pkt 4)
                c = extract_crop(core[f].astype(np.float32), seg) > 0.5
                keep = None
                if keep_holes and f in keep_holes:
                    keep = extract_crop(keep_holes[f].astype(np.float32), seg) > 0.5
                a, hrep = hygiene(a, c, bands.pop(f), keep)
                for k, v in hrep.items():
                    rep.hygiene[k] = rep.hygiene.get(k, 0) + v
                # Soft zones LAST (PLAN_HAIR_TOOL 2 / 12, real test SH020 2026-10-13): the base alpha is computed exactly
                # as without zones (edge mode, temporal median, hygiene – bit-identical outside the zone), then the zone's
                # own alpha (already stabilized in time by the zone, strands connected to the core, islands kept) wins in
                # its area. Hygiene of the base cannot eat strands and strands cannot change the base's holes / islands.
                if soft_override and f in soft_override:
                    # the base alpha (no zones) is kept per version: "Zastosuj" (step 4) re-applies changed zones on it.
                    # The override blends on the HALF-precision base (what alpha_base holds), so "Zastosuj" reproduces
                    # "Przelicz finalnie" bit for bit (SH020 2026-10-13: float32 base gave 113 px / frame of rounding).
                    write_alpha_exr(run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", a)
                    a = np.clip(a, 0, 1).astype(np.float16).astype(np.float32)
                    sa, sarea = soft_override[f]
                    area = extract_crop(sarea.astype(np.float32), seg) > 0.5
                    sw = (extract_crop(soft_weights[f], seg) if soft_weights and f in soft_weights else None)
                    a = apply_soft_override(a, extract_crop(sa.astype(np.float32), seg), area, soft_feather, sw)
                done[f] = a
                # write crop-only alpha + core (runs/v###/alpha/<seg>/a_####.exr)
                write_alpha_exr(run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", a)
                cv2.imwrite(str(_mk(run_dir / "core" / seg.seg_id) / f"c_{f:04d}.png"), c.astype(np.uint8) * 255)

            class _Seq:
                def __init__(self, get):
                    self.get = get

                def __len__(self):
                    return len(sf)

                def __getitem__(self, i):
                    return self.get(i)
            temporal_median(_Seq(res_i), radius, mlimit, _Seq(lambda i: band_of(sf[i])),
                            progress=lambda d, t: job_progress("Mediana czasowa", d, t), emit=emit)
            per_seg[seg.seg_id] = {f: done[f] for f in sf}
            if progress_cb:
                progress_cb("Przelicz finalnie", n + 1, len(segs))

        full = assemble_full(per_seg, segs, frames, self.w, self.h)
        rep.seconds = round(time.time() - t0, 1)
        write_run_json(run_dir, {"schema": "matte_desk/run@1.0", **rep.to_dict(),
                                                      "frames": [int(frames[0]), int(frames[-1])],
                                                      "segments_def": [s.to_dict() for s in segs],
                                                      "params": {**{k: obj.get(k) for k in ("edge_mode", "band_scale",
                                                                 "temporal_smoothing", "ma_scale", "ma_scale_mode", "key_band_same",
                                                                 "shift_px")},
                                                                 **{k: obj[k] for k in ("band_out_px", "band_in_px")
                                                                    if obj.get(k) is not None}}})
        return full, rep


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_soft_override(base: np.ndarray, sa: np.ndarray, area: np.ndarray, feather_px: float = ZONE_FEATHER_PX,
                        weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Soft-zone alpha `sa` wins inside `area`, feathered over `feather_px` INSIDE the area (no step at the polygon
    border); pixels outside the area are returned untouched (bit-identical). Used by finalize and by "Zastosuj".
    `weight` (uint8 0..255, zones.ZoneResult.weights): a precomputed blend weight instead – a zone with its own
    outer-edge feather (2026-10-21)."""
    if weight is not None:
        w = weight.astype(np.float32) / 255.0
        return np.where(area, w * sa + (1.0 - w) * base, base).astype(np.float32)
    w = np.clip(cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5) / float(feather_px), 0, 1)
    return np.where(area, w * sa + (1.0 - w) * base, base).astype(np.float32)


def assemble_full(per_seg: Dict[str, Dict[int, np.ndarray]], segs: Sequence[CropSegment], frames: Sequence[int],
                  w: int, h: int) -> Dict[int, np.ndarray]:
    """Full-frame alpha from crop-segment results; overlaps crossfaded linearly over the shared frames."""
    full = {}
    for f in frames:
        cover = [s for s in segs if s.first <= f <= s.last and f in per_seg.get(s.seg_id, {})]
        if not cover:
            full[f] = np.zeros((h, w), np.float32)
            continue
        if len(cover) == 1:
            full[f] = paste_crop(per_seg[cover[0].seg_id][f], cover[0], w, h)
            continue
        a, b = sorted(cover, key=lambda s: s.first)[:2]
        lo, hi = b.first, a.last
        wb = (f - lo + 1) / float(hi - lo + 2)
        full[f] = (1 - wb) * paste_crop(per_seg[a.seg_id][f], a, w, h) + wb * paste_crop(per_seg[b.seg_id][f], b, w, h)
    return full
