# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/segment.py - SAM 3.1 core propagation from keys (SPEC 7.1-7.3, stages E3/E4).

Keys split the shot into intervals [K_i, K_i+1]. Per interval:
  1. forward  SAM3_VideoTrack from K_i (initial_mask = key K_i) to K_i+1,
  2. backward SAM3_VideoTrack from K_i+1 to K_i (reversed frames),
  3. blend forward/backward on signed distance fields with a linear time weight (no "two contours"),
  4. IoU(fwd, bwd) < 0.90 -> uncertain frame; suggested key = middle of the worst run,
  5. before the first key only backward, after the last only forward (no blend, stricter QA later),
  6. result = binary full-shot SAM 3.1 core (the only level where keys act; MatAnyone2 comes later, 7.3a),
  8. an interval crossing crop segments is split at the segment boundary; the mask at the start of the
     overlap (old crop) -> full frame -> cut into the new crop -> initial mask of the continuation; the 8
     overlap frames are computed in both crops and blended on SDF.
Key frames themselves: result = key (hard). Intervals whose inputs (end keys, crops, params, model) did not
change are taken from `cache/<Ki>-<Kj>/` (hash), so a new key only recomputes its two neighbouring intervals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.crop import CropSegment, extract_crop, paste_crop

IOU_UNCERTAIN = 0.90
# Bump whenever the propagation / blending code changes its output: the interval cache (cache/<Ki>-<Kj>/) is keyed by
# it, so results of older code are never reused (Z12; 2 = blend fix 4f74fb4 – one empty direction no longer wipes the
# interval). Changing it makes the next "Licz ujęcie" recompute every interval once.
CORE_ALGO_VERSION = 2


def sdf(m: np.ndarray) -> np.ndarray:
    """Signed distance: > 0 inside, < 0 outside (px)."""
    b = (m > 0.5).astype(np.uint8)
    if not b.any():
        return -np.full(b.shape, 1e4, np.float32)
    if b.all():
        return np.full(b.shape, 1e4, np.float32)
    inside = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    return (inside - outside).astype(np.float32)


def blend_sdf(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    """Binary blend of two masks on SDF: w_b = weight of b (0..1)."""
    if w_b <= 0:
        return a > 0.5
    if w_b >= 1:
        return b > 0.5
    return ((1 - w_b) * sdf(a) + w_b * sdf(b)) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


@dataclass
class CoreResult:
    core: Dict[int, np.ndarray]                       # frame -> full-frame bool
    fwd_bwd_iou: Dict[int, float] = field(default_factory=dict)
    uncertain: List[int] = field(default_factory=list)
    suggested_keys: List[int] = field(default_factory=list)
    recomputed_intervals: List[Tuple[int, int]] = field(default_factory=list)
    cached_intervals: List[Tuple[int, int]] = field(default_factory=list)
    soft: Dict[int, np.ndarray] = field(default_factory=dict)   # frame -> uint8 probabilities (SAM3 soft path)


def contour_points(mask: np.ndarray, step: float = 4.0, inset: float = 1.5, with_offset: bool = False):
    """Outer contours of a key mask sampled every `step` px, pulled `inset` px inside (sits on the object).
    with_offset=True -> [(points, offset)] where points + offset is back on the outline."""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    gy, gx = np.gradient(cv2.GaussianBlur(d, (0, 0), 1.0))
    out = []
    for c in cnts:
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 12:
            continue
        idx = np.arange(0, len(c), max(1, int(step)))
        p = c[idx]
        xi = np.clip(p[:, 0].astype(int), 0, m.shape[1] - 1)
        yi = np.clip(p[:, 1].astype(int), 0, m.shape[0] - 1)
        n = np.stack([gx[yi, xi], gy[yi, xi]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
        out.append((p + n * inset, -n * inset) if with_offset else p + n * inset)
    return out


class PointPropagator:
    """SPEC 7.2 pkt 7: contour propagation by point tracks (CoTracker3; LK only as a named fallback) – for intervals
    where SAM 3.1 loses the object. Works on display frames downscaled to <= 1024 px."""

    def __init__(self, tracker: Any, load_full: Callable[[int], np.ndarray], plate_w: int, plate_h: int):
        self.tracker = tracker
        self.load = load_full
        self.w, self.h = plate_w, plate_h
        self.s = min(1.0, 1024.0 / max(plate_w, plate_h))
        self.engine = getattr(tracker, "engine", "points")

    def run(self, start: int, end: int, key: np.ndarray) -> Dict[int, np.ndarray]:
        d = 1 if end >= start else -1
        frames = list(range(start, end + d, d))
        s = self.s
        imgs = [cv2.resize(self.load(f), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else self.load(f)
                for f in frames]
        po = contour_points(key, with_offset=True)
        polys = [p for p, _ in po]
        if not polys:
            return {f: np.zeros((self.h, self.w), bool) for f in frames}
        q = np.vstack([np.column_stack([np.zeros(len(p)), p * s]) for p in polys]).astype(np.float32)
        res = self.tracker.track(imgs, q)
        out = {}
        for i, f in enumerate(frames):
            m = np.zeros((self.h, self.w), np.uint8)
            o = 0
            for p, off in po:
                pts = res.tracks[i, o:o + len(p)] / s
                vis = res.visible[i, o:o + len(p)]
                o += len(p)
                pts = np.where(vis[:, None], pts, p) + off  # occluded points keep the key position; back to outline
                cv2.fillPoly(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
            out[f] = m.astype(bool)
        return out


class CorePropagator:
    """`load_crop_image(frame, seg)` -> display crop (side x side x 3) of the plate-to-mask (4b / display)."""

    def __init__(self, runner: Any, segments: Sequence[CropSegment], plate_w: int, plate_h: int,
                 load_crop_image: Callable[[int, CropSegment], np.ndarray], cache_dir: Optional[Path] = None,
                 params: Optional[Dict[str, Any]] = None, point_propagator: Optional[PointPropagator] = None,
                 interval_engines: Optional[Dict[str, str]] = None, soft: bool = False,
                 old_core: Optional[Dict[int, np.ndarray]] = None, deltas: Optional[Dict[int, np.ndarray]] = None,
                 region_fn: Optional[Callable[[int, Sequence[int], np.ndarray], Tuple[Dict[int, np.ndarray], Dict[str, Any]]]] = None,
                 region_only: bool = True):
        self.soft = bool(soft)            # also collect SAM3 soft probabilities (runner.track(..., soft=True))
        self._last_soft: Dict[int, np.ndarray] = {}
        # Z21: a recomputed interval whose keys are `correction` keys changes the core only in the tracked region of
        # the correction (old core elsewhere, bit-identical); needs the previous core, the delta masks of the keys and
        # a region tracker `region_fn(key_frame, frames, region) -> ({frame: region}, info)`; off / nothing old = as before
        self.old_core = old_core or {}
        self.deltas = deltas or {}
        self.region_fn = region_fn
        self.region_only = bool(region_only)
        self.region_stats: Dict[str, Any] = {}
        self.runner = runner
        self.segs = sorted(segments, key=lambda s: s.first)
        self.w, self.h = plate_w, plate_h
        self.load = load_crop_image
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.params = params or {}
        self.points = point_propagator
        self.engines = interval_engines or {}

    # ------------------------------------------------------------------ crop segment helpers
    def _seg_for(self, frame: int, direction: int) -> Optional[CropSegment]:
        """Segment containing `frame` with the most room in `direction` (None = object not present)."""
        cands = [s for s in self.segs if s.first <= frame <= s.last]
        if not cands:
            return None
        return max(cands, key=lambda s: (s.last - frame) if direction > 0 else (frame - s.first))

    def _track_run(self, start: int, end: int, init_full: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """Track from `start` (mask known) to `end` inclusive, crossing crop segments.
        Returns frame -> list of full-frame float masks (2 entries on crop overlaps)."""
        d = 1 if end >= start else -1
        out: Dict[int, List[np.ndarray]] = {}
        self._soft_run: Dict[int, List[np.ndarray]] = {}
        f, mask = start, init_full
        while True:
            seg = self._seg_for(f, d)
            if seg is None:
                return out
            stop = min(end, seg.last) if d > 0 else max(end, seg.first)
            frames = list(range(f, stop + d, d))
            imgs = [self.load(x, seg) for x in frames]
            init_crop = extract_crop(mask.astype(np.float32), seg)
            soft_res = None
            if seg.tiles:
                res = self._track_tiled(imgs, init_crop, seg.tiles)
            elif self.soft:
                r = self.runner.track(imgs, init_crop, soft=True)
                res, soft_res = (r if isinstance(r, tuple) else (r, None))
            else:
                res = self.runner.track(imgs, init_crop)
            for i, (x, m) in enumerate(zip(frames, res)):
                out.setdefault(x, []).append(paste_crop(m, seg, self.w, self.h))
                if soft_res is not None:
                    self._soft_run.setdefault(x, []).append(paste_crop(np.asarray(soft_res[i], np.float32), seg, self.w, self.h))
            if stop == end:
                return out
            # hand-off: the next crop starts at the first frame of the overlap, so both crops compute it
            nxt = [s for s in self.segs if s is not seg and (s.first <= stop <= s.last)]
            nxt = [s for s in nxt if (s.last > seg.last if d > 0 else s.first < seg.first)]
            if not nxt:  # object absent beyond this crop (outside its presence range)
                return out
            ns = nxt[0]
            f = ns.first if d > 0 else ns.last
            mask = out[f][0]

    def _track_tiled(self, imgs: List[np.ndarray], init: np.ndarray, tiles: Sequence[Sequence[int]]) -> List[np.ndarray]:
        """SPEC 5.2 pkt 5: SAM 3.1 per 1008-px tile (25 % overlap), composed with a cos^2 window; tiles without the
        object in the initial mask are skipped."""
        side = imgs[0].shape[0]
        acc = [np.zeros((side, side), np.float32) for _ in imgs]
        wsum = np.zeros((side, side), np.float32)
        for x, y, ts, _ in tiles:
            t_init = init[y:y + ts, x:x + ts]
            if not (t_init > 0.5).any():
                continue
            win1 = np.sin(np.linspace(0, np.pi, ts, dtype=np.float32)) ** 2 + 1e-3
            win = np.outer(win1, win1)
            res = self.runner.track([im[y:y + ts, x:x + ts] for im in imgs], t_init)
            for i, m in enumerate(res):
                acc[i][y:y + ts, x:x + ts] += m * win
            wsum[y:y + ts, x:x + ts] += win
        return [np.where(wsum > 0, a / np.maximum(wsum, 1e-6), 0.0) for a in acc]

    @staticmethod
    def _merge_overlaps(run: Dict[int, List[np.ndarray]], segs: Sequence[CropSegment]) -> Dict[int, np.ndarray]:
        """Crop-overlap frames computed twice -> SDF blend with a linear weight across the overlap."""
        merged = {}
        for f, ms in run.items():
            if len(ms) == 1:
                merged[f] = ms[0] > 0.5
                continue
            ov = [s for s in segs if s.first <= f <= s.last]
            ov = sorted(ov, key=lambda s: s.first)
            a, b = ov[0], ov[-1]
            lo, hi = b.first, a.last
            w = (f - lo + 1) / float(hi - lo + 2)
            merged[f] = blend_sdf(ms[0], ms[-1], w)
        return merged

    # ------------------------------------------------------------------ cache
    def _hash(self, ka: Optional[Tuple[int, np.ndarray]], kb: Optional[Tuple[int, np.ndarray]], span: Tuple[int, int]) -> str:
        h = hashlib.sha1()
        for k in (ka, kb):
            if k is None:
                h.update(b"none")
            else:
                h.update(str(k[0]).encode())
                h.update(np.packbits(k[1] > 0.5).tobytes())
        h.update(json.dumps([span, [s.to_dict() for s in self.segs if not (s.last < span[0] or s.first > span[1])],
                             self.params], sort_keys=True, default=str).encode())
        h.update(str(getattr(self.runner, "ckpt", "")).encode())
        h.update(f"core_algo_v{CORE_ALGO_VERSION}".encode())
        # Z21: the result depends on the correction regions (delta masks of the interval's keys) and on the mode
        for k in (ka, kb):
            if k is not None and self.region_only and k[0] in self.deltas and k[0] in self.old_core:
                from engine.correct_region import REGION_ALGO_VERSION
                h.update(f"delta{k[0]}_v{REGION_ALGO_VERSION}".encode())
                h.update(np.packbits(self.deltas[k[0]] > 0.5).tobytes())
        return h.hexdigest()[:16]

    def _cache_get(self, name: str, digest: str, frames: Sequence[int]) -> Optional[Dict[int, np.ndarray]]:
        """Cached interval (binary cores); with `self.soft` the soft maps (p_####.png) must be there too, else a
        miss (an interval cached before the soft path is recomputed once). Soft maps land in self._cached_soft."""
        if not self.cache_dir:
            return None
        d = self.cache_dir / name
        meta = d / "hash.json"
        if not meta.exists() or json.loads(meta.read_text()).get("hash") != digest:
            return None
        out = {}
        self._cached_soft = {}
        for f in frames:
            m = cv2.imread(str(d / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            out[f] = m > 127
            if self.soft:
                p = cv2.imread(str(d / f"p_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if p is None:
                    return None
                self._cached_soft[f] = p
        return out

    def _cache_put(self, name: str, digest: str, data: Dict[int, np.ndarray], extra: Dict[str, Any],
                   soft: Optional[Dict[int, np.ndarray]] = None) -> None:
        if not self.cache_dir:
            return
        d = self.cache_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for f, m in data.items():
            cv2.imwrite(str(d / f"c_{f:04d}.png"), (m > 0.5).astype(np.uint8) * 255)
        for f, p in (soft or {}).items():
            cv2.imwrite(str(d / f"p_{f:04d}.png"), p if p.dtype == np.uint8 else (np.clip(p, 0, 1) * 255 + 0.5).astype(np.uint8))
        (d / "hash.json").write_text(json.dumps({"hash": digest, "core_algo": CORE_ALGO_VERSION, "soft": bool(soft), **extra}, default=str))

    # ------------------------------------------------------------------ main
    def propagate(self, frames: Sequence[int], keys: Dict[int, np.ndarray],
                  progress_cb: Optional[Callable[[str, int, int], None]] = None) -> CoreResult:
        frames = list(frames)
        ks = sorted(k for k in keys if frames[0] <= k <= frames[-1])
        if not ks:
            raise ValueError("brak klucza w zakresie ujęcia")
        res = CoreResult(core={})
        spans: List[Tuple[Optional[int], Optional[int]]] = []
        if ks[0] > frames[0]:
            spans.append((None, ks[0]))
        spans += list(zip(ks[:-1], ks[1:]))
        if ks[-1] < frames[-1]:
            spans.append((ks[-1], None))
        for n_done, (a, b) in enumerate(spans):
            lo = frames[0] if a is None else a
            hi = frames[-1] if b is None else b
            name = f"{'start' if a is None else a}-{'end' if b is None else b}"
            engine = self.engines.get(name, "sam3")
            self._engine = engine if (engine == "points" and self.points is not None) else "sam3"
            self.params = {**self.params, "interval_engine": self._engine}
            digest = self._hash(None if a is None else (a, keys[a]), None if b is None else (b, keys[b]), (lo, hi))
            span_frames = list(range(lo, hi + 1))
            label = f"Propagacja rdzenia: przedział {n_done + 1}/{len(spans)} (kl. {lo}–{hi})"
            # done / total in FRAMES (the job window shows them as frames): with one key the run has 2 intervals and
            # "2/2" read as "only two frames computed" (Woman2, 2026-10-26)
            f_done = lo - frames[0] + (a is not None)   # the start key was counted with the previous interval
            self._step = (lambda d, _l=label, _n=f_done: progress_cb(f"{_l}, {d}", _n, len(frames))) if progress_cb else None
            if progress_cb:
                progress_cb(label, f_done, len(frames))
            cached = self._cache_get(name, digest, span_frames)
            soft_part: Dict[int, np.ndarray] = {}
            if cached is not None:
                res.cached_intervals.append((lo, hi))
                part, ious = cached, {}
                meta = json.loads((self.cache_dir / name / "hash.json").read_text())
                ious = {int(k): v for k, v in meta.get("iou", {}).items()}
                soft_part = dict(getattr(self, "_cached_soft", {}) or {})
            else:
                part, ious = self._interval(a, b, lo, hi, keys)
                part = self._restrict_to_regions(a, b, lo, hi, part, keys)
                res.recomputed_intervals.append((lo, hi))
                if self.soft:
                    # soft maps for every frame of the span: SAM3 probabilities where the tracker gave them, the
                    # binary mask elsewhere (tiles / point engine / lost object)
                    sp = getattr(self, "_soft_part", {}) or {}
                    soft_part = {f: (np.clip(sp[f], 0, 1) * 255 + 0.5).astype(np.uint8) if f in sp
                                 else (part[f] > 0.5).astype(np.uint8) * 255 for f in part}
                self._cache_put(name, digest, part, {"iou": {str(k): v for k, v in ious.items()}},
                                soft=soft_part if self.soft else None)
            for f, m in part.items():
                if f in keys:
                    m = keys[f] > 0.5  # key = hard truth in its frame
                    if self.soft:
                        soft_part[f] = m.astype(np.uint8) * 255
                res.core[f] = m
                if self.soft and f in soft_part:
                    res.soft[f] = soft_part[f]
            res.fwd_bwd_iou.update(ious)
            if progress_cb:
                progress_cb("Propagacja rdzenia SAM 3.1", hi - frames[0] + 1, len(frames))
        res.uncertain = sorted(f for f, v in res.fwd_bwd_iou.items() if v < IOU_UNCERTAIN)
        res.suggested_keys = suggest_keys(res.uncertain)
        return res

    def _restrict_to_regions(self, a: Optional[int], b: Optional[int], lo: int, hi: int,
                             part: Dict[int, np.ndarray], keys: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Z21: when the interval's inputs changed only through correction keys (keys with a delta; the other keys of
        the interval equal the old core in their frames) and the old core covers the interval, the new core replaces
        the old one only inside the tracked correction regions (feathered). A changed key of another kind (points /
        candidate / brush without delta) or a missing old core -> the interval is taken as computed."""
        from engine.correct_region import merge_region, region_from_delta
        if not self.region_only or not self.old_core or self.region_fn is None:
            return part
        span = list(range(lo, hi + 1))
        key_frames = [k for k in (a, b) if k is not None]
        if any(f not in self.old_core for f in span):
            return part
        corr = [k for k in key_frames if k in self.deltas]
        if not corr:
            return part
        for k in key_frames:          # a non-correction key must be unchanged (== old core in its frame)
            if k not in self.deltas and not np.array_equal(keys[k] > 0.5, self.old_core[k] > 0.5):
                return part
        region = {f: np.zeros_like(part[f]) for f in span}
        infos = []
        for k in corr:
            reg_k = region_from_delta(self.deltas[k])
            regs, info = self.region_fn(k, span, reg_k)
            infos.append({"key": k, **info})
            for f in span:
                if f in regs:
                    region[f] |= regs[f]
        out = {}
        changed = 0
        for f in span:
            if f in key_frames:
                out[f] = part[f]
                continue
            out[f] = merge_region(self.old_core[f], part[f], region[f])
            changed += int((out[f] != (self.old_core[f] > 0.5)).sum())
        self.region_stats[f"{a}-{b}"] = {"frames": len(span), "changed_px": changed,
                                         "region_px_mean": round(float(np.mean([region[f].sum() for f in span])), 1),
                                         "tracking": infos}
        return out

    def _run(self, start: int, end: int, key: np.ndarray) -> Dict[int, np.ndarray]:
        self._last_soft = {}
        if getattr(self, "_engine", "sam3") == "points":
            return self.points.run(start, end, key)
        merged = self._merge_overlaps(self._track_run(start, end, key), self.segs)
        if self.soft and self._soft_run:
            self._last_soft = self._merge_soft(self._soft_run, self.segs)
        return merged

    @staticmethod
    def _merge_soft(run: Dict[int, List[np.ndarray]], segs: Sequence[CropSegment]) -> Dict[int, np.ndarray]:
        """Soft probabilities on crop overlaps: linear blend with the same weight as the SDF blend of the masks."""
        merged = {}
        for f, ms in run.items():
            if len(ms) == 1:
                merged[f] = ms[0]
                continue
            ov = sorted([s for s in segs if s.first <= f <= s.last], key=lambda s: s.first)
            a, b = ov[0], ov[-1]
            lo, hi = b.first, a.last
            w = (f - lo + 1) / float(hi - lo + 2)
            merged[f] = ((1.0 - w) * ms[0] + w * ms[-1]).astype(np.float32)
        return merged

    def _interval(self, a: Optional[int], b: Optional[int], lo: int, hi: int, keys: Dict[int, np.ndarray]
                  ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        from engine.comfy_io import progress_prefix
        step = getattr(self, "_step", None) or (lambda d: None)
        self._soft_part: Dict[int, np.ndarray] = {}
        if a is not None and b is not None:
            step(f"w przód {a}→{b}")
            with progress_prefix(f"Rdzeń w przód F{a}→F{b}"):
                fwd = self._run(a, b, keys[a])
            sf = dict(self._last_soft)
            step(f"wstecz {b}→{a}")
            with progress_prefix(f"Rdzeń wstecz F{b}→F{a}"):
                bwd = self._run(b, a, keys[b])
            sb = dict(self._last_soft)
            empty = np.zeros((self.h, self.w), bool)
            for f in range(lo, hi + 1):
                fwd.setdefault(f, empty)
                bwd.setdefault(f, empty)
            out, ious = {}, {}
            for f in range(lo, hi + 1):
                w = (f - a) / float(max(1, b - a))
                has_f, has_b = bool((fwd[f] > 0.5).any()), bool((bwd[f] > 0.5).any())
                if has_f != has_b:
                    # one direction lost the object: its empty SDF (-1e4) would wipe the other side in the
                    # blend -> keep the direction that still tracks (IoU 0 marks the frame uncertain)
                    out[f] = (fwd[f] if has_f else bwd[f]) > 0.5
                    if self.soft and (f in sf if has_f else f in sb):
                        self._soft_part[f] = sf[f] if has_f else sb[f]
                else:
                    out[f] = blend_sdf(fwd[f], bwd[f], w)
                    if self.soft and f in sf and f in sb:      # the same linear weight on the probabilities
                        self._soft_part[f] = ((1.0 - w) * sf[f] + w * sb[f]).astype(np.float32)
                    elif self.soft and (f in sf or f in sb):
                        self._soft_part[f] = sf.get(f, sb.get(f))
                if a < f < b:
                    ious[f] = iou(fwd[f], bwd[f])
            return out, ious
        empty = np.zeros((self.h, self.w), bool)
        if a is None:  # before the first key: backward only
            step(f"wstecz {b}→{lo}")
            with progress_prefix(f"Rdzeń wstecz F{b}→F{lo}"):
                part = self._run(b, lo, keys[b])
        else:
            step(f"w przód {a}→{hi}")
            with progress_prefix(f"Rdzeń w przód F{a}→F{hi}"):
                part = self._run(a, hi, keys[a])
        if self.soft:
            self._soft_part = dict(self._last_soft)
        return {f: part.get(f, empty) for f in range(lo, hi + 1)}, {}


def suggest_keys(uncertain: Sequence[int]) -> List[int]:
    """Middle of each run of uncertain frames, worst (longest) runs first (user's halving technique)."""
    runs, cur = [], []
    for f in sorted(uncertain):
        if cur and f != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)
    return [r[len(r) // 2] for r in sorted(runs, key=len, reverse=True)]
