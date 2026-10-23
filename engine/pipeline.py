# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/pipeline.py - Per-object orchestration: core propagation (7.2) and "Przelicz finalnie" (7.3a).

Glues ObjectStore (keys, crops.json) + plate-to-mask crop loader + SAM 3.1 / MatAnyone2 runners + runs/v###.
Used by the API; tests drive it with the mock Comfy client.
"""

from __future__ import annotations

import json
import os
import threading
import zlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.comfy_io import job_progress

# zone previews (2026-10-16, „live” sliders): mask PNGs and group wholes read again and again for the same frames –
# cached by path + mtime (a rewritten file = a new entry), bounded
_MASK_CACHE: Dict[Any, np.ndarray] = {}
_MASK_CACHE_MAX = 600


def _cache_put(k, v):
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        for kk in list(_MASK_CACHE)[: _MASK_CACHE_MAX // 4]:
            _MASK_CACHE.pop(kk, None)
    _MASK_CACHE[k] = v
    return v


_CORE_LRU: "OrderedDict[Any, np.ndarray]" = OrderedDict()
_CORE_LRU_MAX = 12            # ~8 MB each at 4K
_CORE_LRU_LOCK = threading.Lock()
_EDGE_CORE_MEMO: "OrderedDict[str, Any]" = OrderedDict()   # edge_preview core_key -> (core, source, band ref)


def _read_core_png(fp: Path) -> Optional[np.ndarray]:
    """bool core mask (> 127) of a PNG as read (no resize), None = no file. The last few reads are kept by (path, mtime,
    size) – the band previews read the same key / current frame core several times a request (2026-10-23); always a
    copy, callers may write into it."""
    try:
        st = fp.stat()
    except OSError:
        return None
    k = (str(fp), st.st_mtime_ns, st.st_size)
    with _CORE_LRU_LOCK:
        m = _CORE_LRU.get(k)
        if m is not None:
            _CORE_LRU.move_to_end(k)
    if m is None:
        g = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        m = g > 127
        with _CORE_LRU_LOCK:
            _CORE_LRU[k] = m
            while len(_CORE_LRU) > _CORE_LRU_MAX:
                _CORE_LRU.popitem(last=False)
    return m.copy()


def _read_mask_cached(fp: Path, w: int, h: int) -> Optional[np.ndarray]:
    """bool mask of a PNG at plate size (None = no file), cached by (path, mtime, size)."""
    try:
        st = fp.stat()
    except OSError:
        return None
    k = ("m", str(fp), st.st_mtime_ns, w, h)
    if k in _MASK_CACHE:
        return _MASK_CACHE[k]
    m = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return _cache_put(k, m > 127)
from engine.crop import CropSegment, extract_crop, load_crops, pad_border, paste_crop
from engine.depth import frame_path
from engine.objects import ObjectStore
from engine.refine import Finalizer, contiguous_chunks, core_soft_start_on, next_run_dir
from engine.sam3 import Sam3Runner
from engine.segment import CorePropagator, CoreResult


def default_display(lin: np.ndarray) -> np.ndarray:
    """Scene-linear -> display float 0..1 (PlateToDisplay). Display-referred sources are not transformed."""
    from engine.ingest import plate_to_display_float
    return plate_to_display_float(lin)


# Z30: a display function marked `elementwise` may be applied to a crop instead of the full frame (same pixels);
# `lut_white` = it is PlateToDisplay with that white -> half EXR data goes through the 65 536-entry LUT (bit-identical)
default_display.elementwise = True
default_display.lut_white = 1.0


def identity_display(x: np.ndarray) -> np.ndarray:
    """No transform (backlit X2 crops are used linear)."""
    return x


identity_display.elementwise = True


logger = logging.getLogger(__name__)
_AUX_SIZE: Dict[str, Optional[Tuple[int, int]]] = {}


def plate_size(pattern: str) -> Optional[Tuple[int, int]]:
    """(w, h) of the first existing frame of a #### / %04d sequence (cached per pattern); None = no frame readable."""
    if pattern in _AUX_SIZE:
        return _AUX_SIZE[pattern]
    import glob
    size = None
    files = sorted(glob.glob(pattern.replace("####", "[0-9]" * 4).replace("%04d", "[0-9]" * 4)))
    if files:
        img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
        if img is not None:
            size = (int(img.shape[1]), int(img.shape[0]))
    _AUX_SIZE[pattern] = size
    return size


def aux_plate_mismatch(pattern: Optional[str], w: int, h: int) -> Optional[str]:
    """A backlit / clear plate whose resolution is not the beauty's (2026-10-17: SH010 4096x2731 plates were set in a
    1440x2560 project -> QA of every stage crashed on the shapes) -> the reason it is not used; None = usable."""
    if not pattern:
        return None
    size = plate_size(pattern)
    if size is None or size == (int(w), int(h)):
        return None
    return f"{size[0]}x{size[1]} ≠ beauty {int(w)}x{int(h)}"


class PlateCropLoader:
    """Model-input crops: full-resolution plate frame (x 4c gain maps) -> display -> square crop (mirror pad).
    With `mplate_dir` the crops come from the plate-to-mask cache (4b: + MC temporal denoise, gated)."""

    def __init__(self, plate_pattern: str, display: Callable[[np.ndarray], np.ndarray] = default_display,
                 gain_pattern: Optional[str] = None, mplate_dir: Optional[Path] = None,
                 frames: Optional[Sequence[int]] = None):
        self.pattern = plate_pattern
        self.display = display
        self.gain_pattern = gain_pattern
        self.mplate_dir = Path(mplate_dir) if mplate_dir else None
        self.frames = list(frames) if frames else None
        self._mp: Dict[str, Any] = {}
        self._cache: Dict[int, np.ndarray] = {}
        self._crops: Dict[Any, np.ndarray] = {}
        import threading
        self._lock = threading.Lock()

    def frame(self, f: int) -> np.ndarray:
        if f not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            self._cache[f] = self.read(f)
        return self._cache[f]

    def _gain(self, f: int, w: int, h: int) -> Optional[np.ndarray]:
        if not self.gain_pattern:
            return None
        g = cv2.imread(str(frame_path(self.gain_pattern, f)), cv2.IMREAD_UNCHANGED)
        return None if g is None else cv2.resize(g[..., :3], (w, h), interpolation=cv2.INTER_CUBIC)

    def _rows(self, f: int, y0: Optional[int] = None, y1: Optional[int] = None,
              x0: Optional[int] = None, x1: Optional[int] = None) -> np.ndarray:
        """Display-referred float32 BGR of rows [y0, y1) x columns [x0, x1) of frame f (None = all) – the same
        pixels as `read(f)[y0:y1, x0:x1]` (Z30)."""
        out = self._fast_rows(f, y0, y1, x0, x1)
        return out if out is not None else self.read(f)[y0:y1, x0:x1]

    def _fast_rows(self, f: int, y0=None, y1=None, x0=None, x1=None) -> Optional[np.ndarray]:
        """The display formula is per pixel, so it runs on the region only; half EXR data goes through the
        PlateToDisplay LUT; only the needed scanlines are decompressed. None when not applicable (non-EXR, display
        not marked `elementwise`, layout the fast reader does not mirror) -> the reference path `read`."""
        p = frame_path(self.pattern, f)
        if p.suffix.lower() != ".exr" or not getattr(self.display, "elementwise", False):
            return None
        from engine.exr_io import display_half, read_half_rgb
        try:
            img = read_half_rgb(p, y0, y1)
        except Exception:
            return None
        if img is None:
            return None
        if self.gain_pattern:
            import OpenEXR
            fi = OpenEXR.InputFile(str(p))
            try:
                dw = fi.header()["dataWindow"]
                fw, fh = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
            finally:
                fi.close()
            gain = self._gain(f, fw, fh)
            if gain is not None:   # same deflicker as the comp (4c / 4b pkt 1): float32 product, then the formula
                return self.display(img[:, x0:x1].astype(np.float32) * gain[y0:y1, x0:x1])
        img = img[:, x0:x1]
        lut_white = getattr(self.display, "lut_white", None)
        if lut_white is not None and img.dtype == np.float16:
            return display_half(img, lut_white)
        return self.display(img.astype(np.float32))

    def read(self, f: int) -> np.ndarray:
        """Uncached decode of the whole frame (thread-safe; the QA stage reads frames in parallel)."""
        fast = self._fast_rows(f)
        return fast if fast is not None else self.read_reference(f)

    def read_reference(self, f: int) -> np.ndarray:
        """The pre-Z30 decode (cv2.imread + full-frame formula) – kept for the bit-identity tests / benchmarks."""
        p = frame_path(self.pattern, f)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(p)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dt = img.dtype
        img = img[..., :3].astype(np.float32)
        if p.suffix.lower() != ".exr":   # display-referred (sRGB) source: its own values, no PlateToDisplay curve
            img /= {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(dt, 255.0 if img.max() > 1 else 1.0)
        if self.gain_pattern and p.suffix.lower() == ".exr":
            g = cv2.imread(str(frame_path(self.gain_pattern, f)), cv2.IMREAD_UNCHANGED)
            if g is not None:
                img = img * cv2.resize(g[..., :3], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        return self.display(img) if p.suffix.lower() == ".exr" else img

    def read_crop(self, f: int, seg: CropSegment) -> np.ndarray:
        """= extract_crop(read(f), seg) without decoding / converting the whole frame: only the crop's scanlines
        are read, the colour formula runs on the crop, black padding as in extract_crop (of the in-plate patch)."""
        x, y, s, _ = seg.crop
        l, t, r, b = seg.pad
        patch = self._rows(f, y + t, y + s - b, x + l, x + s - r)
        if l or t or r or b:
            patch = pad_border(patch, t, b, l, r)
        return patch

    def _crop_cached(self, f: int, seg: CropSegment) -> np.ndarray:
        k = (f, tuple(int(v) for v in seg.crop), tuple(int(v) for v in seg.pad))
        with self._lock:
            c = self._crops.get(k)
        if c is None:
            c = self.read_crop(f, seg)
            with self._lock:
                if len(self._crops) > 8:
                    self._crops.pop(next(iter(self._crops)))
                self._crops[k] = c
        return c

    def raw_crop(self, f: int, seg: CropSegment) -> np.ndarray:
        return self._crop_cached(f, seg)

    @staticmethod
    def mp_key(seg: CropSegment) -> str:
        """Plate-to-mask cache key: segment id + crop geometry (ids repeat across objects)."""
        x0, y0, side = (int(v) for v in seg.crop[:3])
        return f"{seg.seg_id}_x{x0}_y{y0}_s{side}"

    def mask_plate(self, seg: CropSegment) -> Optional[Any]:
        """The MaskPlate used for `seg` (after a crop() call), else None."""
        return self._mp.get(self.mp_key(seg))

    def crop(self, f: int, seg: CropSegment, scale: float = 1.0) -> np.ndarray:
        if self.mplate_dir is not None and self.frames and not seg.seg_id.startswith("preview"):
            from engine.mplate import MaskPlate
            # cache key = segment id + crop geometry: every object numbers its segments s00, s01… (shared work/mplate/),
            # so the id alone handed object B the cached crops of object A (SH010: "suknia" tracked on Figure's crop)
            ck = self.mp_key(seg)
            mp = self._mp.get(ck)
            if mp is None or (mp.frames[0], mp.frames[-1]) != (seg.first, seg.last):
                sf = [x for x in self.frames if seg.first <= x <= seg.last]
                mp = self._mp[ck] = MaskPlate(self.mplate_dir, ck, sf, lambda x, s=seg: self.raw_crop(x, s))
            c = mp.crop(f) if f in mp.frames else self.raw_crop(f, seg)
            if c.shape[0] != seg.side or c.shape[1] != seg.side:   # stale / foreign cache: never feed a wrong crop
                c = self.raw_crop(f, seg)
        else:
            c = self._crop_cached(f, seg)
        if scale != 1.0:
            s = max(16, int(round(seg.side * scale)))
            c = cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA)
        return c


class _ReleaseOnUse:
    """A point tracker that asks ComfyUI to free the GPU the first time it is really used (a cached propagation never
    touches it)."""

    def __init__(self, tracker, release):
        self._t, self._release, self._done = tracker, release, False

    def __getattr__(self, k):
        if not self._done:
            self._done = True
            self._release()
        return getattr(self._t, k)

    def __call__(self, *a, **kw):
        if not self._done:
            self._done = True
            self._release()
        return self._t(*a, **kw)


class ObjectPipeline:
    def __init__(self, work_dir: Path, plate_pattern: str, plate_w: int, plate_h: int, client: Any,
                 display: Callable[[np.ndarray], np.ndarray] = default_display, sam3_ckpt: Optional[str] = None,
                 backlit_pattern: Optional[str] = None, backlit_params: Optional[Dict[str, Any]] = None,
                 ltx_workflow: Optional[Dict[str, Any]] = None, use_mplate: bool = True,
                 gain_pattern: Optional[str] = None, frames: Optional[Sequence[int]] = None,
                 shutter_deg: float = 180.0, cleanplate_pattern: Optional[str] = None, io_mode: str = "exr"):
        """io_mode: how images go to Comfy (engine/comfy_io.py) – "exr" (default, unchanged) or "png8" (8-bit sources)."""
        self.work = Path(work_dir)
        self.io_mode = io_mode
        # plate-to-mask denoise (4b) on/off – project.json mask_plate_denoise resolved by the caller; part of the
        # core cache hash and of core.json (stage "core" goes stale when the project setting changes)
        self.use_mplate = bool(use_mplate)
        self.store = ObjectStore(self.work)
        self.w, self.h = plate_w, plate_h
        self.client = client
        self.shutter = shutter_deg
        # backlit / clear plate of another resolution than the beauty = a wrong plate: not used anywhere (QA of every
        # stage, auto edge mode, backlit edge, matte zones' clean plate), the reason in `plates_skipped` -> run.json / UI
        self.plates_skipped: Dict[str, str] = {}
        for kind, pat in (("cleanplate", cleanplate_pattern), ("backlit", backlit_pattern)):
            why = aux_plate_mismatch(pat, plate_w, plate_h)
            if why:
                self.plates_skipped[kind] = why
                logger.warning("%s %s not used: %s", kind, pat, why)
        if "cleanplate" in self.plates_skipped:
            cleanplate_pattern = None
        if "backlit" in self.plates_skipped:
            backlit_pattern = None
        self.clean = PlateCropLoader(cleanplate_pattern, display) if cleanplate_pattern else None
        self.point_tracker = None   # injectable (tests); default make_point_tracker() = CoTracker3 or LK fallback
        self.zone_tracker = None    # injectable (tests); default: CoTracker3 when installed, else core bbox shift
        self.loader = PlateCropLoader(plate_pattern, display, gain_pattern,
                                      self.work / "mplate" if use_mplate else None, frames)
        self.runner = Sam3Runner(client, self.work / "comfy_io", sam3_ckpt, io_mode=io_mode)
        self.backlit = PlateCropLoader(backlit_pattern, identity_display) if backlit_pattern else None
        self.backlit_params = backlit_params or {}
        self.ltx_workflow = ltx_workflow

    def _finalizer(self) -> Finalizer:
        return Finalizer(self.client, self.work / "comfy_io", self.w, self.h,
                         lambda f, s, sc: self.loader.crop(f, s, sc),
                         load_backlit_crop=(lambda f, s: self.backlit.crop(f, s)) if self.backlit else None,
                         backlit_params=self.backlit_params, ltx_workflow=self.ltx_workflow,
                         io_mode=self.io_mode,
                         load_backlit_full=(lambda f: self.backlit.read(f)) if self.backlit else None)

    def auto_edge_mode(self, name: str, frames: Sequence[int]) -> Dict[str, Any]:
        """SPEC 5.5: metrics on ~10 sampled frames; borderline -> test batch in both modes, lower edge flicker wins."""
        from engine.edge_mode import backlit_cover, choose_edge_mode, edge_flicker, routing_metrics
        obj = self.store.get(name)
        core = self.load_core(name, frames)
        sample = list(frames)[:: max(1, len(frames) // 10)]
        segs = self.segments(name)
        imgs, cores, bl = [], [], []
        for f in sample:
            cov = [s for s in segs if s.first <= f <= s.last]
            if not cov:
                continue
            imgs.append(self.loader.crop(f, cov[0]))
            cores.append(extract_crop(core[f].astype(np.float32), cov[0]) > 0.5)
            if self.backlit is not None:
                bl.append(self.backlit.crop(f, cov[0]))
        metrics = routing_metrics(cores, imgs)
        if self.backlit is not None:
            metrics["backlit_cover"] = backlit_cover(cores, bl)
        # LTX is a stage of its own since 2026-10-16 (engine/ltx_alpha.py) – Auto never picks it as an edge mode
        choice = choose_edge_mode(obj, metrics, self.backlit is not None, False)
        if choice.get("borderline") and segs:
            seg = segs[0]
            test = [f for f in frames if seg.first <= f <= seg.last][:9]
            fl = {}
            for mode in (choice["mode"], choice["alternative"]):
                tmp = next_run_dir(self.store.dir(name) / "auto_test")
                full, _ = self._finalizer().finalize(obj, [seg], test, core, self.keys(name), test[0], tmp, mode)
                fl[mode] = edge_flicker([full[f] for f in test], [core[f] for f in test])
            best = min(fl, key=fl.get)
            choice["reason"] += f" (graniczny: test migotania {', '.join(f'{k} {v:.4f}' for k, v in fl.items())} → {best})"
            choice["mode"] = best
        choice["metrics"] = {k: round(v, 4) for k, v in metrics.items()}
        self.store.update(name, edge_mode_auto=choice["mode"], tool={"edge_auto_reason": choice["reason"],
                                                                     "routing_metrics": choice["metrics"]})
        return choice

    def segments(self, name: str) -> List[CropSegment]:
        p = self.store.dir(name) / "crops.json"
        if not p.exists():
            raise FileNotFoundError("brak crops.json – najpierw kadrowanie")
        return load_crops(p)

    def _segments_in(self, name: str, frames: Sequence[int]) -> List[CropSegment]:
        """Crop segments with at least one frame of `frames` (a work area skips the others)."""
        lo, hi = int(frames[0]), int(frames[-1])
        fs = sorted(int(f) for f in frames)
        import bisect
        # a segment with at least one of the frames (frames may have gaps: the object's life range)
        segs = [s for s in self.segments(name)
                if not (s.last < lo or s.first > hi) and bisect.bisect_left(fs, s.first) < bisect.bisect_right(fs, s.last)]
        if not segs:
            raise ValueError(f"brak odcinka kadru w zakresie F{lo}–{hi}")
        return segs

    def keys(self, name: str) -> Dict[int, np.ndarray]:
        """Key masks; keys outside the object's life range are ignored (the object is not there)."""
        from engine.workarea import is_visible, object_life
        life = object_life(self.store.get(name))
        return {f: self.store.key_mask(name, f) for f in self.store.keys(name) if is_visible(f, life)}

    def _visible(self, name: str, frames: Sequence[int]) -> List[int]:
        """`frames` inside the object's life range (all without one); ValueError when none is visible."""
        from engine.workarea import object_life, visible_frames
        frames = list(frames)
        vis = visible_frames(frames, object_life(self.store.get(name)))
        if not vis:
            raise ValueError(f"obiekt niewidoczny w klatkach F{frames[0]}–{frames[-1]} (zakres życia)")
        return vis

    def _core_params(self) -> Dict[str, Any]:
        """Interval cache-hash params. The denoise flag is added only when OFF, so the hash of an ON run (= every core
        computed before the switch existed) is unchanged and existing caches stay valid; ON and OFF never collide."""
        p: Dict[str, Any] = {"engine": "sam3", "plate": self.loader.pattern}
        if not self.use_mplate:
            p["mplate"] = False
        return p

    def run_core(self, name: str, frames: Sequence[int], progress_cb=None) -> CoreResult:
        from engine.stages import keys_signature
        keys_sig = keys_signature(self.store, name)   # inputs of this run (stage "core" goes stale when keys change)
        segs = self.segments(name)
        engines = self.store.get(name)["tool"].get("interval_engines", {})
        pp = None
        if "points" in engines.values():
            from engine.segment import PointPropagator
            from engine.tracking import make_point_tracker
            pp = PointPropagator(self.point_tracker or make_point_tracker(), self.loader.frame, self.w, self.h)
        soft_ok = bool(getattr(self.runner, "soft_available", lambda: False)())   # PlateMatte SAM3_VideoTrackSoft in Comfy
        # Z21: previous core + delta masks of the correction keys + region tracker (CoTracker3, fallback LK) -> a
        # recomputed interval changes the core only in the tracked correction region (object option, default on)
        obj0 = self.store.get(name)
        region_only = bool(obj0.get("correction_region_only", True))
        # object life range (engine/workarea.py): only visible frames are computed, every visible piece on its own
        from engine.workarea import intervals_from_frames, is_visible, mask_stats, object_life, life_hints
        frames = list(frames)
        life = object_life(obj0)
        vis = [f for f in frames if is_visible(f, life)]
        if not vis:
            raise ValueError(f"obiekt niewidoczny w klatkach F{frames[0]}–{frames[-1]} (zakres życia)")
        old_core: Dict[int, np.ndarray] = {}
        deltas: Dict[int, np.ndarray] = {}
        region_fn = None
        if region_only:
            from engine.correct_region import load_deltas
            deltas = load_deltas(self.store.dir(name), self.store.keys(name))
            if deltas:
                cdir0 = self.store.dir(name) / "core"
                for f in vis:
                    m = cv2.imread(str(cdir0 / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        old_core = {}
                        break
                    old_core[f] = m > 127
                if old_core:
                    region_fn = self._region_tracker_fn()
        prop = CorePropagator(self.runner, segs, self.w, self.h, lambda f, s: self.loader.crop(f, s),
                              cache_dir=self.store.dir(name) / "cache",
                              params=self._core_params(),
                              point_propagator=pp, interval_engines=engines, soft=soft_ok,
                              old_core=old_core, deltas=deltas, region_fn=region_fn, region_only=region_only)
        if life is None:
            res = prop.propagate(frames, self.keys(name), progress_cb)
            pieces = [[int(frames[0]), int(frames[-1])]]
        else:
            # no propagation across invisible frames (the tracker would jump onto another object); keys outside the
            # life are ignored
            keys_v = {k: m for k, m in self.keys(name).items() if is_visible(k, life)}
            pieces = intervals_from_frames(vis)
            res = CoreResult(core={})
            for a, b in pieces:
                part = prop.propagate(list(range(a, b + 1)), keys_v, progress_cb)
                res.core.update(part.core)
                res.soft.update(part.soft)
                res.fwd_bwd_iou.update(part.fwd_bwd_iou)
                res.recomputed_intervals += part.recomputed_intervals
                res.cached_intervals += part.cached_intervals
            from engine.segment import IOU_UNCERTAIN, suggest_keys
            res.uncertain = sorted(f for f, v in res.fwd_bwd_iou.items() if v < IOU_UNCERTAIN)
            res.suggested_keys = suggest_keys(res.uncertain)
        res.region_stats = dict(prop.region_stats)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        # lupa on thin parts (5.2 pkt 5b) – part of step (1) of the pipeline
        obj = self.store.get(name)
        lupa_mode = obj.get("thin_lens", "auto")
        start = obj["tool"].get("start_frame") or min(res.core)
        if start not in res.core:      # start frame outside the object's life
            start = min(res.core)
        from engine.lupa import THIN_SCORE_THR, run_lupa, thin_score
        if lupa_mode == "box" or (lupa_mode == "auto" and thin_score(res.core[start]) > THIN_SCORE_THR):
            box = obj["tool"].get("lupa_box") if lupa_mode == "box" else None
            merged, lsegs = {}, []
            for a, b in pieces:            # one piece = the whole run without a life range
                m_, l_ = run_lupa(self.runner, res.core, list(range(a, b + 1)), self.w, self.h,
                                  lambda f, s: self.loader.crop(f, s), manual_box=box)
                merged.update(m_)
                lsegs += l_
            user = {f for f, k in self.store.keys(name).items() if k.get("author") == "user"}
            res.core.update({f: m for f, m in merged.items() if f not in user})  # auto keys may be refined
            (self.store.dir(name) / "lupa.json").write_text(json.dumps([s.to_dict() for s in lsegs], indent=1))
        # objects in front (smaller depth_order): their cores are subtracted (MatteSubtractFront on the core,
        # 5.2 pkt 6); a jump of the tracker onto a front object (IoU > 0.3) is flagged
        res.front_flags = {}
        # front objects without a core in some frames: nothing is subtracted there -> warning for the UI (2026-10-23)
        res.front_missing = []
        me = self.store.get(name)
        for o in self.store.list():
            if o["name"] == name or o.get("depth_order", 0) >= me.get("depth_order", 0):
                continue
            fdir = self.store.dir(o["name"]) / "core"
            missing = []
            for f in list(res.core):
                fm = cv2.imread(str(fdir / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if fm is None:
                    missing.append(int(f))
                    continue
                fm = fm > 127
                from engine.segment import iou as _iou
                if _iou(res.core[f], fm) > 0.3:
                    res.front_flags[f] = f"tracker przeskoczył na obiekt z przodu ({o['name']}) – dodaj klucz"
                if f not in self.store.keys(name) or self.store.keys(name)[f].get("author") != "user":
                    # only the front object's interior (eroded 8 px): the back object stays continuous under the
                    # front edge band, so the export's MatteSubtractFront gives a seamless boundary (9: never max)
                    inner = cv2.erode(fm.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
                    # where the front object occludes this one near its edge, the back core continues under it
                    # (amodal within 8 px) – otherwise the modal SAM hole + the front's soft edge leave a seam
                    # front core + its 2-px soft edge, only where it actually touches this object (<= 3 px)
                    occl = cv2.dilate(fm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    if res.core[f].any():
                        near = cv2.distanceTransform((~res.core[f]).astype(np.uint8), cv2.DIST_L2, 5) <= 3
                        res.core[f] = res.core[f] | (occl & near)
                        if f in res.soft:
                            res.soft[f] = np.where(occl & near, 255, res.soft[f]).astype(np.uint8)
                    res.core[f] = res.core[f] & ~inner
                    if f in res.soft:          # the soft map mirrors the front-object subtraction
                        res.soft[f] = np.where(inner, 0, res.soft[f]).astype(np.uint8)
            if missing:
                missing.sort()
                n = len(missing)
                where = "w żadnej klatce" if n == len(res.core) else f"w {n} kl. (F{missing[0]}–{missing[-1]})"
                res.front_missing.append({
                    "object": o["name"], "frames": n, "first": missing[0], "last": missing[-1],
                    "message": f"Obiekt z przodu „{o['name']}” nie ma policzonego rdzenia {where} – nie został odjęty, "
                               f"maska może go objąć. Policz najpierw „{o['name']}”, potem przelicz ten obiekt."})
        # hint: the core gets empty right after touching the frame border (object leaves / enters) -> life range
        st = {f: mask_stats(m) for f, m in res.core.items()}
        res.life_hints = life_hints({f: v[0] for f, v in st.items()}, {f: v[1] for f, v in st.items()}, life)
        cdir = self.store.dir(name) / "core"
        # work area (engine/workarea.py): the core covers the frames of this run, plus the frames of the previous core
        # when it was computed from the same inputs (keys, crops, denoise) – otherwise only this run's frames
        from engine.workarea import core_coverage, union
        coverage = [list(p) for p in pieces]
        cj0 = cdir / "core.json"
        if cj0.exists():
            try:
                old = json.loads(cj0.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
            crops_p = self.store.dir(name) / "crops.json"
            same = (old.get("keys_sig") == keys_sig and bool(old.get("mask_plate_denoise", True)) == bool(self.use_mplate)
                    and (not crops_p.exists() or crops_p.stat().st_mtime <= cj0.stat().st_mtime)
                    and old.get("life") == life)
            if same:
                coverage = union(coverage, core_coverage(self.store.dir(name), legacy_whole=False))
        cdir.mkdir(parents=True, exist_ok=True)
        for f, m in res.core.items():
            cv2.imwrite(str(cdir / f"c_{f:04d}.png"), (m > 0.5).astype(np.uint8) * 255)
        hidden = [f for f in frames if f not in res.core]       # outside the life: empty core, nothing computed
        if hidden:
            zero = np.zeros((self.h, self.w), np.uint8)
            for f in hidden:
                cv2.imwrite(str(cdir / f"c_{f:04d}.png"), zero)
        # SAM3 soft probabilities (8-bit) next to the binary core – read by stabilized_core (probability mode); a run
        # without the soft node removes stale maps so the stabilisation never mixes old probabilities with a new core
        sdir = self.store.dir(name) / "core_soft"
        if sdir.exists():
            # a run on a work area keeps the maps of the other frames when they stay covered and this run has maps too
            keep_other = bool(res.soft) and coverage != [list(p) for p in pieces]
            fset = set(int(f) for f in frames)
            for p in sdir.glob("p_*.png"):
                if not keep_other or int(p.stem[2:]) in fset:
                    p.unlink()
        if res.soft:
            sdir.mkdir(parents=True, exist_ok=True)
            for f, p in res.soft.items():
                cv2.imwrite(str(sdir / f"p_{f:04d}.png"), p)
        (cdir / "core.json").write_text(json.dumps({
            "fwd_bwd_iou": {str(k): round(v, 4) for k, v in res.fwd_bwd_iou.items()},
            "uncertain": res.uncertain, "suggested_keys": res.suggested_keys,
            "recomputed": res.recomputed_intervals, "cached": res.cached_intervals, "keys_sig": keys_sig,
            "mask_plate_denoise": self.use_mplate, "soft_frames": len(res.soft),
            "correction_regions": getattr(res, "region_stats", {}), "coverage": coverage,
            "frames": [int(frames[0]), int(frames[-1])], "life": life, "life_hints": res.life_hints,
            "front_missing": res.front_missing}, indent=1))
        self.store.update(name, tool={"status": "rdzeń gotowy", "final_valid": False})
        return res

    def load_core(self, name: str, frames: Sequence[int]) -> Dict[int, np.ndarray]:
        cdir = self.store.dir(name) / "core"
        out = {}
        for f in frames:
            m = _read_core_png(cdir / f"c_{f:04d}.png")
            if m is None:
                raise FileNotFoundError(f"brak rdzenia klatki {f} – najpierw policz ujęcie")
            out[f] = m
        return out

    def _region_tracker_fn(self):
        """Z21: `region_fn(key_frame, frames, region) -> ({frame: region}, info)` on CoTracker3 (fallback LK) over the
        display frames downscaled to <= 1024 px; Comfy's models are released first (the tracker runs in this process)."""
        from engine.correct_region import track_region
        tracker = self._zone_tracker()
        if tracker is None:
            try:
                from engine.tracking import LKTracker
                tracker = LKTracker()
            except Exception:
                tracker = None
        s = min(1.0, 1024.0 / max(self.w, self.h))
        released = {"done": False}

        def fn(key_frame: int, frames: Sequence[int], region: np.ndarray):
            if not released["done"]:
                self.release_comfy_gpu("rejon korekty: śledzenie CoTracker3")
                released["done"] = True
            small = [cv2.resize(self.loader.frame(f), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0
                     else self.loader.frame(f) for f in frames]
            return track_region(tracker, small, s, list(frames), key_frame, region)
        return fn

    def release_comfy_gpu(self, reason: str = "") -> bool:
        """Before GPU work outside Comfy (CoTracker3 tracking, ViTMatte): ask ComfyUI to unload its models
        (`POST /free`) – only when its queue is idle (never under someone's running job). Comfy reloads the models on
        the next prompt (~10-30 s). Returns True when a release was requested; silent no-op without a client."""
        client = getattr(self, "client", None)
        if client is None or not hasattr(client, "free_vram"):
            return False
        try:
            q = client.get_queue()
            if q.get("busy"):
                return False
            ok = bool(client.free_vram(unload_models=True, free_memory=True))
            if ok:
                import logging
                logging.getLogger(__name__).info("Comfy: modele zwolnione przed liczeniem na GPU (%s)", reason)
            return ok
        except Exception:
            return False

    def load_core_soft(self, name: str, frames: Sequence[int], core: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """SAM 3.1 soft probabilities written by run_core (`core_soft/p_####.png`, uint8) of the frames that have them."""
        sdir = self.store.dir(name) / "core_soft"
        probs: Dict[int, np.ndarray] = {}
        if sdir.exists():
            for f in frames:
                p = cv2.imread(str(sdir / f"p_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if p is not None and p.shape == core[f].shape:
                    probs[f] = p
        return probs

    def stabilized_core(self, name: str, frames: Sequence[int], core: Dict[int, np.ndarray], keys: Dict[int, np.ndarray],
                        radius: int = 2, band_px: int = 8, probs_out: Optional[Dict[int, np.ndarray]] = None,
                        only: Optional[Sequence[int]] = None):
        """engine/corestab.stabilize_core on the display gray (user keys untouched); deterministic for the same inputs.
        `probs_out`: filled with the stabilised probability maps (soft start of the band). `only`: previews – just
        these frames stabilised (corestab `only`)."""
        from engine.corestab import stabilize_core
        gcache = self.__dict__.setdefault("_stab_gray", {})   # one plate read per frame for every group member (preview)

        def gray(f):
            if f not in gcache:
                if len(gcache) > 16:
                    gcache.clear()
                gcache[f] = cv2.cvtColor((np.clip(self.loader.read(f), 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            return gcache[f]
        user_keys = [f for f, k in keys.items()] if keys else []
        # SAM3 soft probabilities (run_core with the PlateMatte soft node) -> probability mode; missing frames = binary
        probs = self.load_core_soft(name, frames, core)
        stab, stats = stabilize_core(core, gray, radius=radius, band_px=band_px, keys=user_keys, probs=probs or None,
                                     progress=lambda d, t: job_progress("Stabilizacja rdzenia", d, t), probs_out=probs_out,
                                     only=only)
        # the edge against an object in front is the run_core front subtraction, not SAM jitter: it must follow the
        # front object in every frame – the temporal vote dragged it after the moving front object and opened seams
        # in the export union (E6 synthetic: 5 -> 14 px). Within band_px of a front core the frame keeps its own value
        # (core and, for the soft start, its own probability map consistent with it).
        me = self.store.get(name)
        fronts = [o["name"] for o in self.store.list()
                  if o["name"] != name and o.get("depth_order", 0) < me.get("depth_order", 0)]
        if fronts:
            from engine.corestab import consistent_soft
            k = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
            for f in stab:
                near = None
                for o in fronts:
                    fm = cv2.imread(str(self.store.dir(o) / "core" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                    if fm is None or fm.shape != core[f].shape:
                        continue
                    d = cv2.dilate((fm > 127).astype(np.uint8), k).astype(bool)
                    near = d if near is None else (near | d)
                if near is not None:
                    stab[f] = np.where(near, core[f], stab[f])
                    if probs_out is not None and f in probs_out and f in probs:
                        probs_out[f] = np.where(near, consistent_soft(probs[f], core[f]), probs_out[f]).astype(np.uint8)
        return stab, stats

    def core_with_soft(self, name: str, frames: Sequence[int], keys: Dict[int, np.ndarray], stabilize: bool,
                       only: Optional[Sequence[int]] = None):
        """(core, soft): the SAM 3.1 core (stabilised when `stabilize`) and the probability maps of the frames that have one
        and are not keys (stabilised the same way, threshold-consistent with the core; keys stay hard). Soft start of
        the edge band (user decision 2026-10-14): MatAnyone2 CoreCombine core_mode=soft, ViTMatte zone trimap."""
        from engine.corestab import consistent_soft
        core = self.load_core(name, frames)
        soft: Dict[int, np.ndarray] = {}
        if stabilize:
            core, self._corestab_stats = self.stabilized_core(name, frames, core, keys, probs_out=soft, only=only)
        else:
            soft = {f: consistent_soft(p, core[f]) for f, p in self.load_core_soft(name, frames, core).items()}
        kset = set(keys or {})
        return core, {f: p for f, p in soft.items() if f not in kset}

    def _final_core(self, name: str, frames: Sequence[int], obj: Dict[str, Any], keys: Dict[int, np.ndarray],
                    kinds: Optional[Sequence[str]] = None, only: Optional[Sequence[int]] = None):
        """Core fed to the edge mode: zones (2) + shrink / grow slider (3). Deterministic – recomputed by the edge fix.
        `kinds`: zone kinds applied (stage 3 MatAnyone2: only the brush corrections; None = all)."""
        # (1b) temporal stabilisation of the SAM3 core (engine/corestab.py; object option core_stabilize) + the SAM3
        # probabilities stabilised alongside (soft start of the band; consistent with the core, keys hard)
        core, soft = self.core_with_soft(name, frames, keys, core_stabilize_on(obj), only=only)
        self._final_soft = soft
        # (2) zones on the core, before the edge mode (7.7 pkt 5)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=kinds)
        self._band_ref = (name, zres.core if zres else core)
        return self._shifted_core(zres.core if zres else core, obj, keys, zres), zres

    def _keys_with_brush(self, name: str, keys: Dict[int, np.ndarray], core2: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Key masks handed to the edge pass. With stage-2 brush corrections the key frames carry the mask 1+2 (the
        corrected core there), else MatAnyone2 is anchored on the raw SAM 3.1 key and erases thin added parts (antennae:
        alpha 0.05 on the added pixels, synthetic test 2026-10-15). Without brush zones: the keys as they are (bitwise)."""
        from engine.stages import zone_dicts
        if not zone_dicts(self.store.dir(name), ("brush",)):
            return keys
        return {f: (core2[f] if f in core2 else m) for f, m in keys.items()}

    @staticmethod
    def _shifted_core(core: Dict[int, np.ndarray], obj: Dict[str, Any], keys, zres) -> Dict[int, np.ndarray]:
        """(3) shrink / grow slider: SDF shift of the core (keys stay hard); pixels removed by zones never come back."""
        from engine.edge import apply_shift, band_px_limits
        core2 = dict(core)
        shift = 0.0 if band_px_limits(obj) else float(obj.get("shift_px", 0.0) or 0.0)   # px mode: the mask is not shifted
        if abs(shift) > 1e-3:
            core2 = {f: (m if f in keys else apply_shift(m, np.full(m.shape, shift, np.float32))) for f, m in core2.items()}
            if zres:
                core2 = {f: core2[f] & ~zres.removed[f] for f in core2}
        return core2

    def group_members(self, name: str) -> Optional[List[str]]:
        """[parent, parts…] when `name` has parts (a group computed as a whole in stages 3 / 4), else None."""
        from engine.stages import group_parts
        parts = group_parts(self.store, name)
        return [name] + parts if parts else None

    def _mask12(self, name: str, frames: Sequence[int], only: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
        """Mask 1+2 of one object (stabilised SAM 3.1 core + its brush corrections) on `frames`; frames outside its life
        range = empty. ValueError when it has no core there."""
        obj = self.store.get(name)
        keys = self.keys(name)
        fr = self._visible(name, frames)
        empty = np.zeros((self.h, self.w), bool)
        if not fr:
            return {f: empty for f in frames}
        try:
            core, soft = self.core_with_soft(name, fr, keys, core_stabilize_on(obj), only=only)
        except FileNotFoundError as e:
            raise ValueError(f"„{name}” (część grupy): {e} – policz jej rdzeń (etap 1)")
        zres = self.apply_zones(name, fr, core, keys, probs=soft, kinds=("brush",))
        m = zres.core if zres else core
        return {f: m.get(f, empty) for f in frames}

    def _mask12_disk(self, name: str, f: int) -> Optional[np.ndarray]:
        """The mask 1+2 of frame f from core_fix/ (written by stage 1 / „Zastosuj poprawki”) when it is current for
        `name` and every member of its group (a group: joined with the seam as group_core); None = compute it.
        A group whose parent has its own brush zones is recomputed (group_core applies them once more on the whole)."""
        from engine.stages import core_fix_state, zone_dicts
        members = self.group_members(name)
        if members and zone_dicts(self.store.dir(name), ("brush",)):
            return None
        cs = []
        for m in members or [name]:
            try:
                if not core_fix_state(self.store, m)[1]:
                    return None
            except (OSError, ValueError):
                return None
            im = cv2.imread(str(self.store.dir(m) / "core_fix" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape != (self.h, self.w):
                return None
            cs.append(im > 127)
        if not members:
            return cs[0]
        from engine.export import seam_px, seamless_union
        return seamless_union([c.astype(np.float32) for c in cs], cs, gap_px=seam_px(self.store.get(name))) > 0.5

    def write_mask12(self, name: str, frames: Sequence[int], progress_cb=None) -> Optional[Dict[str, Any]]:
        """After the SAM 3.1 core (stage 1, 2026-10-22): the mask 1+2 (stabilised core) written to core_fix/ when the
        object has no brush corrections – with corrections that is „▶▶ Zastosuj poprawki” (stage 2) as before."""
        from engine.stages import zone_dicts
        if zone_dicts(self.store.dir(name), ("brush",)):
            return None
        return self.build_core_fix(name, frames, progress_cb)

    def group_core(self, name: str, frames: Sequence[int], only: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
        """The whole of a group (user decision 2026-10-15): masks 1+2 of the parent and its parts joined with the contact
        seam filled (the parent's part_seam_px, engine/export.seamless_union – the same rule as `<parent>_full`)."""
        from engine.export import seam_px, seamless_union
        members = self.group_members(name) or [name]
        if only is not None and len(members) > 1:   # a preview: the members side by side (cv2 releases the GIL)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(members)) as ex:
                masks = list(ex.map(lambda m: self._mask12(m, frames, only), members))
        else:
            masks = [self._mask12(m, frames, only) for m in members]
        gap = seam_px(self.store.get(name))
        out = {}
        for f in frames:
            cs = [mk[f] for mk in masks]
            out[f] = seamless_union([c.astype(np.float32) for c in cs], cs, gap_px=gap) > 0.5
        return out

    def _stage_core(self, name: str, frames: Sequence[int], obj: Dict[str, Any], keys: Dict[int, np.ndarray]):
        """(core, soft, members) the stages 3 / 4 start from: a group -> its whole (soft maps off: they belong to the
        parent alone); else the (stabilised) SAM 3.1 core of the object. The brush corrections are applied by the caller
        (apply_zones, idempotent on the whole)."""
        members = self.group_members(name)
        if members:
            return self.group_core(name, frames), {}, members
        core, soft = self.core_with_soft(name, frames, keys, core_stabilize_on(obj))
        return core, soft, None

    def _work_segments(self, name: str, frames: Sequence[int]) -> List[CropSegment]:
        """Crop segments stages 3 / 4 compute in: the object's own, for a group grown to cover its parts' crops too
        (engine.crop.group_segments)."""
        segs = self._segments_in(name, frames)
        members = self.group_members(name)
        if not members:
            return segs
        from engine.crop import group_segments
        others = []
        for n in members[1:]:
            try:
                others.append(self.segments(n))
            except FileNotFoundError:
                pass
        return group_segments(segs, others, self.w, self.h)

    def _crop_outside(self, segs, frames: Sequence[int], core: Dict[int, np.ndarray]) -> int:
        """Pixels of `core` outside every crop segment covering the frame (a part of a group reaching out of the
        parent's crops is cut there – reported, the crops come from the parent's keys)."""
        n = 0
        for f in frames:
            cover = np.zeros((self.h, self.w), bool)
            for s in segs:
                if s.first <= f <= s.last:
                    x, y, side, _ = s.crop
                    cover[max(0, y):y + side, max(0, x):x + side] = True
            n += int((core[f] & ~cover).sum())
        return n

    def core_base_run(self, name: str, frames: Sequence[int], progress_cb=None) -> str:
        """The mask 1+2 (+ the shrink / grow slider, exactly the core the edge pass would get) written as a version
        runs/v### (stage "core": alpha = alpha_base = the hard mask, crops of the object's segments) – the base of
        ViTMatte on 1+2 (vitmatte_base = core). Reused while current (tool.stage_runs.core_base); not made active."""
        from engine.refine import write_alpha_exr
        from engine.stages import mask_sig, settings_sig, stage_run
        from engine.workarea import intervals_from_frames, object_life
        cur, ok, _ = stage_run(self.store, name, "core_base")
        if cur and ok:
            return cur
        obj = self.store.get(name)
        frames = self._visible(name, frames)
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft, members = self._stage_core(name, frames, obj, keys)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        core2 = self._shifted_core(zres.core if zres else core, obj, keys, zres)
        segs = self._work_segments(name, frames) if members else self._segments_in(name, frames)
        run_dir = next_run_dir(self.store.dir(name))
        n = 0
        for seg in segs:
            sf = [f for f in frames if seg.first <= f <= seg.last]
            for f in sf:
                n += 1
                job_progress("Maska 1+2 jako wersja", n, len(frames))
                c = (extract_crop(core2[f].astype(np.float32), seg) > 0.5).astype(np.float32)
                write_alpha_exr(run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", c)
                write_alpha_exr(run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", c)
                cd = run_dir / "core" / seg.seg_id
                cd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cd / f"c_{f:04d}.png"), (c > 0.5).astype(np.uint8) * 255)
        rj = {"schema": "matte_desk/run@1.0", "run": run_dir.name, "edge_mode": "core_fix", "stage": "core",
              "frames": [int(frames[0]), int(frames[-1])], "segments_def": [s.to_dict() for s in segs],
              "core_stabilize": core_stabilize_on(obj), "core_fix_sig": mask_sig(self.store, name),
              **({"plates_skipped": self.plates_skipped} if self.plates_skipped else {}),
              "settings_sig": settings_sig(obj), "created_at": datetime.now().isoformat(timespec="seconds")}
        if members:
            rj.update(group=members, group_seam_px=self.store.get(name).get("part_seam_px"),
                      group_outside_crop_px=self._crop_outside(segs, frames, core2))
        if object_life(obj) is not None:
            rj["coverage"] = intervals_from_frames(frames)
        (run_dir / "run.json").write_text(json.dumps(rj, indent=2, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, tool={"stage_runs": {**(self.store.get(name)["tool"].get("stage_runs") or {}),
                                                     "core_base": run_dir.name}})
        if progress_cb:
            progress_cb("Maska 1+2 jako wersja", len(frames), len(frames))
        return run_dir.name

    def ltx_guides(self, name: str, seg: CropSegment, frames: Sequence[int], core2: Dict[int, np.ndarray],
                   others: Dict[int, np.ndarray], guide: str) -> List[np.ndarray]:
        """Guide crops of stage LTX-2.5 (engine/ltx_alpha.py): the plate-to-mask crop as it is ("plain"), with the other
        objects painted out ("isolate") or also the surroundings defocused ("steer")."""
        from engine.ltx_alpha import isolate_guide
        out = []
        for f in frames:
            img = self.loader.crop(f, seg, 1.0)
            if guide != "plain":
                sel = extract_crop(core2[f].astype(np.float32), seg)
                oth = extract_crop(others[f].astype(np.float32), seg) if f in others else None
                img = isolate_guide(img, sel, oth, steer=guide == "steer")
            out.append(img)
        return out

    def ltx_mask(self, name: str, frames: Sequence[int], s: Dict[str, Any], core12: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The mask that limits / guides LTX-2.5 (user request 2026-10-16, setting mask_source): the mask 1+2 (`core12`,
        a group = its whole) or the result of stage 3 / 4 (alpha > 0.5; must exist)."""
        src = s.get("mask_source", "core")
        if src == "core":
            return core12
        from engine.export import run_alpha_loader
        from engine.stages import stage_run
        import re
        if re.fullmatch(r"v\d{3,}", src):   # a version of stage Eksperymenty (later stages may use it; LTX never feeds stage 5 back)
            rjp = self.store.dir(name) / "runs" / src / "run.json"
            if not rjp.exists():
                raise ValueError(f"Maska LTX-2.5 = wersja {src} – nie ma jej (usunięta?); wybierz inną maskę")
            if json.loads(rjp.read_text(encoding="utf-8")).get("stage") != "experiment":
                raise ValueError(f"Maska LTX-2.5 = wersja {src}: tylko wersje z Eksperymentów (wyniki etapów 3 / 4 wybierz po nazwie etapu)")
            run = src
        else:
            run, _, _ = stage_run(self.store, name, src)
        if not run:
            raise ValueError(f"Maska LTX-2.5 = wynik etapu {'MatAnyone2' if src == 'matanyone' else 'ViTMatte'} – brak tego wyniku; "
                             "policz etap albo wybierz maskę 1+2")
        ld = run_alpha_loader(self.store.dir(name), run, self.w, self.h)
        return {f: ld(f) > 0.5 for f in frames}

    def ltx_others(self, name: str, frames: Sequence[int], members: Sequence[str]) -> Dict[int, np.ndarray]:
        """Union of the masks 1+2 of every other object of the project (painted out of the LTX-2.5 guide); objects without
        a core are skipped."""
        acc: Dict[int, np.ndarray] = {}
        for o in self.store.list():
            if o["name"] in members:
                continue
            try:
                m = self._mask12(o["name"], frames)
            except (ValueError, FileNotFoundError, KeyError):
                continue
            for f in frames:
                acc[f] = (acc[f] | m[f]) if f in acc else m[f].copy()
        return acc

    def ltx_run(self, name: str, frames: Sequence[int], settings: Optional[Dict[str, Any]] = None,
                progress_cb=None, models: Optional[Dict[str, str]] = None) -> str:
        """Stage „LTX-2.5 Alpha-Gen” (user decision 2026-10-16, a stage of its own before the export): LTX-2.5 IC-LoRA
        Alpha-Gen on every crop segment of the object, guided by the plate-to-mask crop (other objects painted out),
        limited to the object by its mask 1+2 -> a version runs/v### (stage "ltx", alpha = alpha_base) ->
        tool.stage_runs.ltx. A group (parent + parts) is computed as a whole, as in stages 3 / 4."""
        import time
        from engine.ltx_alpha import AlphaGen, finish, object_settings, settings_sig as ltx_sig
        from engine.refine import write_alpha_exr
        from engine.stages import mask_sig, settings_sig
        from engine.workarea import intervals_from_frames, object_life
        t0 = time.time()
        obj = self.store.get(name)
        s = settings or object_settings(obj)
        frames = self._visible(name, frames)
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft, members = self._stage_core(name, frames, obj, keys)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        # the mask 1+2 as it is (2026-10-16: no MatAnyone shrink / grow slider here – LTX has its own region sliders),
        # or the result of stage 3 / 4 (mask_source); a group = its whole, in the members' crops
        core2 = self.ltx_mask(name, frames, s, zres.core if zres else core)
        others = self.ltx_others(name, frames, members or [name]) if s["guide"] != "plain" else {}
        segs = self._work_segments(name, frames) if members else self._segments_in(name, frames)
        gen = AlphaGen(self.client, self.work / "comfy_io", s, models=models,
                       progress=lambda t, d, n: job_progress(t, d, n))
        pts_at = self.ltx_band_points(name, s, frames)
        crop_pts = lambda f, seg: [(x - seg.crop[0], y - seg.crop[1], o, i) for x, y, o, i in pts_at(f)] if pts_at else None
        run_dir = next_run_dir(self.store.dir(name))
        for seg in segs:
            sf = [f for f in frames if seg.first <= f <= seg.last]
            for chunk in contiguous_chunks(sf):
                if not chunk:
                    continue
                job_progress(f"LTX-2.5: przewodnik {seg.seg_id}", 0, len(chunk))
                guides = self.ltx_guides(name, seg, chunk, core2, others, s["guide"])
                mattes = gen.run(f"{name}_{seg.seg_id}_{chunk[0]}", guides, (seg.side, seg.side))
                for f, m in zip(chunk, mattes):
                    cc = extract_crop(core2[f].astype(np.float32), seg)
                    a = finish(m, cc, s, crop_pts(f, seg))
                    write_alpha_exr(run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", a)
                    write_alpha_exr(run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", a)
                    write_alpha_exr(run_dir / "ltx_raw" / seg.seg_id / f"a_{f:04d}.exr", m)
                    cd = run_dir / "core" / seg.seg_id
                    cd.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cd / f"c_{f:04d}.png"), (cc > 0.5).astype(np.uint8) * 255)
        rj = {"schema": "matte_desk/run@1.0", "run": run_dir.name, "edge_mode": "ltx", "stage": "ltx",
              "frames": [int(frames[0]), int(frames[-1])], "segments_def": [sg.to_dict() for sg in segs],
              "core_stabilize": core_stabilize_on(obj), "core_fix_sig": mask_sig(self.store, name),
              **({"plates_skipped": self.plates_skipped} if self.plates_skipped else {}),
              "settings_sig": settings_sig(obj), "ltx": s, "ltx_sig": ltx_sig(s), "ltx_batches": gen.log,
              "seconds": round(time.time() - t0, 1), "created_at": datetime.now().isoformat(timespec="seconds")}
        if members:
            rj.update(group=members, group_seam_px=obj.get("part_seam_px"),
                      group_outside_crop_px=self._crop_outside(segs, frames, core2))
        if object_life(obj) is not None:
            rj["coverage"] = intervals_from_frames(frames)
        (run_dir / "run.json").write_text(json.dumps(rj, indent=2, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, tool={"stage_runs": {**(self.store.get(name)["tool"].get("stage_runs") or {}),
                                                     "ltx": run_dir.name}})
        if progress_cb:
            progress_cb("LTX-2.5 Alpha-Gen", len(frames), len(frames))
        return run_dir.name

    def run_referenced(self, name: str, run: str) -> bool:
        """Something builds on this version: the active run, another version (applied_from / base_run / refinished_from /
        LTX-2.5 mask), a stage result other than the one being redone, or the object's last export (outputs.json)."""
        obj = self.store.get(name)
        tool = obj.get("tool") or {}
        if tool.get("active_run") == run:
            return True
        od = self.store.dir(name)
        for rj_p in (od / "runs").glob("v*/run.json"):
            if rj_p.parent.name == run:
                continue
            try:
                rj = json.loads(rj_p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if run in (rj.get("applied_from"), rj.get("base_run"), rj.get("refinished_from"),
                       (rj.get("ltx") or {}).get("mask_source")):
                return True
        try:
            out = json.loads((self.work / "outputs.json").read_text(encoding="utf-8"))
            if any(e.get("name") == name and e.get("version") == run for e in out.get("objects", [])):
                return True
        except (OSError, ValueError):
            pass
        return False

    def _refinish_dir(self, name: str, src: str, srj: Dict[str, Any], auto: bool) -> Tuple[Path, str]:
        """(run_dir, refinished_from) for „sliders on the result”. `auto` (sliders applied by themselves, user decision
        2026-10-17): the previous automatic sliders version is overwritten in place – one version per model run, not one
        per slider move – unless something builds on it (then a new version, the old one untouched). The raw model
        matte is copied into the version, so overwriting never loses it."""
        od = self.store.dir(name)
        if auto and srj.get("auto_refinish") and srj.get("refinished_from") and not self.run_referenced(name, src):
            return od / "runs" / src, srj["refinished_from"]
        return next_run_dir(od), src

    def ltx_refinish(self, name: str, frames: Sequence[int], settings: Optional[Dict[str, Any]] = None,
                     progress_cb=None, auto: bool = False) -> str:
        """„Zastosuj suwaki na wynik” of stage LTX-2.5 (2026-10-16): the RAW LTX-2.5 mattes of the current LTX-2.5 version
        (runs/<v>/ltx_raw) limited again with the current region / mask settings -> a new version, no model run."""
        import time
        from engine.export import run_alpha_loader
        from engine.ltx_alpha import finish, object_settings, settings_sig as ltx_sig
        from engine.refine import write_alpha_exr
        from engine.stages import mask_sig, settings_sig
        from engine.workarea import intervals_from_frames, object_life
        t0 = time.time()
        obj = self.store.get(name)
        s = settings or object_settings(obj)
        src = (obj.get("tool") or {}).get("stage_runs", {}).get("ltx")
        od = self.store.dir(name)
        if not src or not (od / "runs" / src / "ltx_raw").exists():
            raise ValueError("brak wyniku LTX-2.5 z surową alfą – najpierw „▶ Policz LTX-2.5”")
        srj = json.loads((od / "runs" / src / "run.json").read_text(encoding="utf-8"))
        segs = [CropSegment.from_dict(d) for d in srj["segments_def"]]
        lo, hi = srj["frames"]
        frames = [f for f in self._visible(name, frames) if lo <= f <= hi]
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft, members = self._stage_core(name, frames, obj, keys)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        mask = self.ltx_mask(name, frames, s, zres.core if zres else core)
        raw = run_alpha_loader(od, src, self.w, self.h, sub="ltx_raw")
        pts_at = self.ltx_band_points(name, s, frames)
        run_dir, from_run = self._refinish_dir(name, src, srj, auto)
        n = 0
        for seg in segs:
            for f in [x for x in frames if seg.first <= x <= seg.last]:
                n += 1
                job_progress("LTX-2.5: suwaki na wynik", n, len(frames))
                m = extract_crop(raw(f).astype(np.float32), seg)
                cc = extract_crop(mask[f].astype(np.float32), seg)
                a = finish(m, cc, s, [(x - seg.crop[0], y - seg.crop[1], o, i) for x, y, o, i in pts_at(f)] if pts_at else None)
                write_alpha_exr(run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", a)
                write_alpha_exr(run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", a)
                write_alpha_exr(run_dir / "ltx_raw" / seg.seg_id / f"a_{f:04d}.exr", m)
                cd = run_dir / "core" / seg.seg_id
                cd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cd / f"c_{f:04d}.png"), (cc > 0.5).astype(np.uint8) * 255)
        rj = {"schema": "matte_desk/run@1.0", "run": run_dir.name, "edge_mode": "ltx", "stage": "ltx",
              "frames": [int(frames[0]), int(frames[-1])], "segments_def": srj["segments_def"],
              "core_stabilize": core_stabilize_on(obj), "core_fix_sig": mask_sig(self.store, name),
              **({"plates_skipped": self.plates_skipped} if self.plates_skipped else {}),
              "settings_sig": settings_sig(obj), "ltx": s, "ltx_sig": ltx_sig(s), "ltx_batches": srj.get("ltx_batches"),
              "refinished_from": from_run, "auto_refinish": bool(auto), "seconds": round(time.time() - t0, 1),
              "created_at": datetime.now().isoformat(timespec="seconds")}
        for k in ("group", "group_seam_px", "group_outside_crop_px", "coverage"):
            if k in srj:
                rj[k] = srj[k]
        if "coverage" not in rj and object_life(obj) is not None:
            rj["coverage"] = intervals_from_frames(frames)
        (run_dir / "run.json").write_text(json.dumps(rj, indent=2, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, tool={"stage_runs": {**(self.store.get(name)["tool"].get("stage_runs") or {}),
                                                     "ltx": run_dir.name}})
        if progress_cb:
            progress_cb("LTX-2.5: suwaki na wynik", len(frames), len(frames))
        return run_dir.name

    def ltx_frame_preview(self, name: str, f: int, s: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """LTX-2.5 palette, live (2026-10-16): (band u8 – 128 kept region / 255 forced 1, the LTX-2.5 alpha of the current LTX-2.5
        version re-limited with `s` or None without one). Mask 1+2 = the members' masks + seam (fast, as the zone
        previews), or the stage 3 / 4 result (mask_source)."""
        from engine.export import run_alpha_loader
        from engine.ltx_alpha import band_u8, finish
        mask = self.ltx_mask(name, [f], s, self.preview_core(name, [f]))[f]
        pf = self.ltx_band_points(name, s, [f], only=[f])
        pts = pf(f) if pf else None
        alpha = None
        src = (self.store.get(name).get("tool") or {}).get("stage_runs", {}).get("ltx")
        od = self.store.dir(name)
        if src and (od / "runs" / src / "ltx_raw").exists():
            try:
                raw = run_alpha_loader(od, src, self.w, self.h, sub="ltx_raw")(f)
                alpha = finish(raw, mask.astype(np.float32), s, pts)
            except (OSError, KeyError, ValueError):
                alpha = None
        return band_u8(mask, s, pts), alpha

    # ------------------------------------------------------------------ stage „VideoMaMa” (engine/videomama.py)
    def vmm_base(self, name: str, frames: Sequence[int], s: Dict[str, Any],
                 core12: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Optional[str]]:
        """Base of stage VideoMaMa (setting `base`): ({frame: full-frame alpha 0..1}, stage-3 run or None). "core" = the
        mask 1+2 (`core12`, a group = its whole); "matanyone" = the current stage-3 version (must be valid)."""
        if s["base"] == "core":
            return {f: core12[f].astype(np.float32) for f in frames}, None
        from engine.export import run_alpha_loader
        from engine.stages import stage_run
        run, ok, why = stage_run(self.store, name, "matanyone")
        if not run or not ok:
            raise ValueError("Baza VideoMaMa = wynik MatAnyone2 (etap 3) – " + (why or "brak tego wyniku") +
                             "; przelicz MatAnyone2 albo wybierz bazę „maska 1+2”")
        ld = run_alpha_loader(self.store.dir(name), run, self.w, self.h)
        return {f: ld(f).astype(np.float32) for f in frames}, run

    def _vmm_write(self, run_dir: Path, seg: CropSegment, f: int, raw: np.ndarray, base_c: np.ndarray,
                   guide_c: np.ndarray, s: Dict[str, Any], pts_at=None) -> None:
        from engine.refine import write_alpha_exr
        from engine.videomama import compose
        x0, y0 = seg.crop[0], seg.crop[1]
        pts = [(x - x0, y - y0, o, i) for x, y, o, i in pts_at(f)] if pts_at else None   # plate px -> crop px
        a = compose(base_c, raw, s["protect"], s["reach_px"], s.get("inset_px"), s.get("feather_px", 0.0),
                    s.get("outer_feather_px", 0.0), pts)
        write_alpha_exr(run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", a)
        write_alpha_exr(run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", a)
        write_alpha_exr(run_dir / "vmm_raw" / seg.seg_id / f"a_{f:04d}.exr", raw)
        cd = run_dir / "core" / seg.seg_id
        cd.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), (guide_c > 0.5).astype(np.uint8) * 255)

    def _vmm_run_json(self, run_dir: Path, name: str, obj: Dict[str, Any], frames: Sequence[int], segs: Sequence[CropSegment],
                      s: Dict[str, Any], base_run: Optional[str], log: Any, t0: float, members: Any,
                      core2: Optional[Dict[int, np.ndarray]] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        import time
        from engine.stages import mask_sig, settings_sig
        from engine.videomama import model_sig, settings_sig as vmm_sig
        from engine.workarea import intervals_from_frames, object_life
        rj = {"schema": "matte_desk/run@1.0", "run": run_dir.name, "edge_mode": "videomama", "stage": "videomama",
              "frames": [int(frames[0]), int(frames[-1])], "segments_def": [sg.to_dict() for sg in segs],
              "core_stabilize": core_stabilize_on(obj), "core_fix_sig": mask_sig(self.store, name),
              **({"plates_skipped": self.plates_skipped} if self.plates_skipped else {}),
              "settings_sig": settings_sig(obj), "videomama": {k: v for k, v in s.items()},
              "vmm_sig": vmm_sig(s), "vmm_model_sig": model_sig(s), "base": s["base"], "applied_from": base_run,
              "vmm_batches": log, "seconds": round(time.time() - t0, 1),
              "created_at": datetime.now().isoformat(timespec="seconds"), **(extra or {})}
        if members:
            rj.update(group=members, group_seam_px=obj.get("part_seam_px"))
            if core2 is not None:
                rj["group_outside_crop_px"] = self._crop_outside(segs, frames, core2)
        if object_life(obj) is not None and "coverage" not in rj:
            rj["coverage"] = intervals_from_frames(frames)
        (run_dir / "run.json").write_text(json.dumps(rj, indent=2, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, tool={"stage_runs": {**(self.store.get(name)["tool"].get("stage_runs") or {}),
                                                     "videomama": run_dir.name}})

    def vmm_run(self, name: str, frames: Sequence[int], models_dir: Optional[Path], settings: Optional[Dict[str, Any]] = None,
                progress_cb=None) -> str:
        """Stage „VideoMaMa” (user decision 2026-10-16, between MatAnyone2 and ViTMatte): VideoMaMa on every crop segment,
        guided by the mask 1+2, put over the base (mask 1+2 or the stage-3 result) -> runs/v### (stage "videomama",
        raw matte in vmm_raw/) -> tool.stage_runs.videomama. A group (parent + parts) is computed as a whole."""
        import time
        from engine.videomama import VideoMaMa, object_settings
        t0 = time.time()
        obj = self.store.get(name)
        s = settings or object_settings(obj)
        frames = self._visible(name, frames)
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft, members = self._stage_core(name, frames, obj, keys)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        core2 = zres.core if zres else core
        base, base_run = self.vmm_base(name, frames, s, core2)
        segs = self._work_segments(name, frames) if members else self._segments_in(name, frames)
        try:   # the VideoMaMa pipeline lives outside Comfy's model management: free what Comfy holds (SAM3, MatAnyone…)
            self.client.free_vram()
        except Exception:
            pass
        gen = VideoMaMa(self.client, self.work / "comfy_io", s, models_dir, progress=lambda t, d, n: job_progress(t, d, n))
        pts_at = self.vmm_band_points(name, s, frames)
        run_dir = next_run_dir(self.store.dir(name))
        for seg in segs:
            sf = [f for f in frames if seg.first <= f <= seg.last]
            for chunk in contiguous_chunks(sf):
                if not chunk:
                    continue
                job_progress(f"VideoMaMa: kadry {seg.seg_id}", 0, len(chunk))
                imgs = [self.loader.crop(f, seg, 1.0) for f in chunk]
                guides = [extract_crop(core2[f].astype(np.float32), seg) for f in chunk]
                mattes = gen.run(f"{name}_{seg.seg_id}_{chunk[0]}", imgs, guides)
                for f, m, gc in zip(chunk, mattes, guides):
                    self._vmm_write(run_dir, seg, f, m, extract_crop(base[f], seg), gc, s, pts_at)
        self._vmm_run_json(run_dir, name, obj, frames, segs, s, base_run, gen.log, t0, members, core2)
        if progress_cb:
            progress_cb("VideoMaMa", len(frames), len(frames))
        return run_dir.name

    def vmm_refinish(self, name: str, frames: Sequence[int], settings: Optional[Dict[str, Any]] = None,
                     progress_cb=None, auto: bool = False) -> str:
        """„Suwaki na wynik” of stage VideoMaMa: the RAW mattes of the current VideoMaMa version (vmm_raw/) put over the
        base again with the current base / reach / protect -> a new version, no model run."""
        import time
        from engine.export import run_alpha_loader
        from engine.videomama import model_sig, object_settings
        t0 = time.time()
        obj = self.store.get(name)
        s = settings or object_settings(obj)
        src = (obj.get("tool") or {}).get("stage_runs", {}).get("videomama")
        od = self.store.dir(name)
        if not src or not (od / "runs" / src / "vmm_raw").exists():
            raise ValueError("brak wyniku VideoMaMa z surową alfą – najpierw „▶ Policz VideoMaMa”")
        srj = json.loads((od / "runs" / src / "run.json").read_text(encoding="utf-8"))
        if srj.get("vmm_model_sig") != model_sig(s):
            raise ValueError("zmienione ustawienia modelu (rozdzielczość / paczki / zakładka) – trzeba „▶ Policz VideoMaMa”")
        segs = [CropSegment.from_dict(d) for d in srj["segments_def"]]
        lo, hi = srj["frames"]
        frames = [f for f in self._visible(name, frames) if lo <= f <= hi]
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft, members = self._stage_core(name, frames, obj, keys)
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        core2 = zres.core if zres else core
        base, base_run = self.vmm_base(name, frames, s, core2)
        raw = run_alpha_loader(od, src, self.w, self.h, sub="vmm_raw")
        pts_at = self.vmm_band_points(name, s, frames)
        run_dir, from_run = self._refinish_dir(name, src, srj, auto)
        n = 0
        for seg in segs:
            for f in [x for x in frames if seg.first <= x <= seg.last]:
                n += 1
                job_progress("VideoMaMa: suwaki na wynik", n, len(frames))
                self._vmm_write(run_dir, seg, f, extract_crop(raw(f).astype(np.float32), seg), extract_crop(base[f], seg),
                                extract_crop(core2[f].astype(np.float32), seg), s, pts_at)
        extra = {k: srj[k] for k in ("coverage",) if k in srj}
        extra["refinished_from"] = from_run
        extra["auto_refinish"] = bool(auto)
        self._vmm_run_json(run_dir, name, obj, frames, segs, s, base_run, srj.get("vmm_batches"), t0,
                           srj.get("group"), None, extra)
        if progress_cb:
            progress_cb("VideoMaMa: suwaki na wynik", len(frames), len(frames))
        return run_dir.name

    def vmm_frame_preview(self, name: str, f: int, s: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """VideoMaMa palette, live: (band u8 – 128 = VideoMaMa adds, 255 = protect band, the current VideoMaMa version's
        raw matte put over the base with `s`, or None without one)."""
        from engine.export import run_alpha_loader
        from engine.videomama import band_u8, compose
        core = self.preview_core(name, [f])
        base = self.vmm_base(name, [f], s, core)[0][f]
        pf = self.vmm_band_points(name, s, [f], only=[f])
        pts = pf(f) if pf else None
        alpha = None
        src = (self.store.get(name).get("tool") or {}).get("stage_runs", {}).get("videomama")
        od = self.store.dir(name)
        if src and (od / "runs" / src / "vmm_raw").exists():
            try:
                raw = run_alpha_loader(od, src, self.w, self.h, sub="vmm_raw")(f)
                alpha = compose(base, raw, s["protect"], s["reach_px"], s.get("inset_px"), s.get("feather_px", 0.0),
                                s.get("outer_feather_px", 0.0), pts)
            except (OSError, KeyError, ValueError):
                alpha = None
        return band_u8(base, s, pts), alpha

    def build_core_fix(self, name: str, frames: Sequence[int], progress_cb=None) -> Dict[str, Any]:
        """Stage 2 „▶▶ Zastosuj poprawki” (user decision 2026-10-15): the mask 1+2 = SAM 3.1 core (stabilised as for the
        edge pass) + the brush corrections (zones params.source = brush, propagated by CoTracker3) on the shot ->
        <obj>/core_fix/c_####.png (full frames, 0 / 255) + core_fix.json {sig}. Exactly the core stage 3 (MatAnyone2)
        starts from (before the shrink / grow slider); the export source "rdzeń (1+2)"."""
        from engine.stages import core_fix_sig, zone_dicts
        from engine.workarea import intervals_from_frames
        obj = self.store.get(name)
        frames = self._visible(name, frames)
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core, soft = self.core_with_soft(name, frames, keys, core_stabilize_on(obj))
        zres = self.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        fixed = zres.core if zres else core
        d = self.store.dir(name) / "core_fix"
        d.mkdir(parents=True, exist_ok=True)
        changed = 0
        for i, f in enumerate(frames):
            job_progress("Zapis maski 1+2", i + 1, len(frames))
            cv2.imwrite(str(d / f"c_{f:04d}.png"), fixed[f].astype(np.uint8) * 255)
            changed += int((fixed[f] != core[f]).sum())
        info = {"schema": "matte_desk/core_fix@1.0", "sig": core_fix_sig(self.store, name),
                "coverage": intervals_from_frames(frames), "changed_px": changed,
                "zones": [z["zone_id"] for z in zone_dicts(self.store.dir(name), ("brush",))],
                "created_at": datetime.now().isoformat(timespec="seconds")}
        (d / "core_fix.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        if progress_cb:
            progress_cb("Zastosuj poprawki", len(frames), len(frames))
        return info

    def finalize(self, name: str, frames: Sequence[int], edge_mode: Optional[str] = None, progress_cb=None,
                 stage: Optional[str] = None, preview_dir: Optional[Path] = None):
        """'Przelicz finalnie': one pass + the edge position report (qa.json -> edge). The edge correction is NOT
        applied automatically (Z27 pkt 1, user decision) – stage 4 shows the pieces and `apply_edge_fix` makes it
        a new version on demand.
        `stage="matanyone"` (stage 3 of the 2026-10-15 layout): edge mode MatAnyone2 on the mask 1+2 (SAM 3.1 core + brush
        corrections), WITHOUT the matte (ViTMatte) and threshold zones -> the base version stage 4 applies its zones on
        (alpha_base for every frame, tool.stage_runs.matanyone).
        `preview_dir` (stage preview, engine/stage_preview.py): the same edge pass written into that scratch folder –
        no version, no QA, nothing in the object state; returns {frame: full-frame alpha}."""
        from engine.stages import mask_sig, settings_sig
        obj = self.store.get(name)
        frames = self._visible(name, frames)      # object life range: only the visible frames are computed
        mode = "matanyone" if stage == "matanyone" else (edge_mode or obj.get("edge_mode", "auto"))
        if mode == "auto":
            mode = self.auto_edge_mode(name, frames)["mode"]
            obj = self.store.get(name)
        members = self.group_members(name) if stage == "matanyone" else None
        segs = self._work_segments(name, frames) if members else self._segments_in(name, frames)
        keys = self.keys(name)
        start = obj["tool"].get("start_frame") or min(keys)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        self._anchor_names = members or [name]
        if members:
            # a group (parent + parts, user decision 2026-10-15): MatAnyone on the whole – the masks 1+2 of every member
            # joined with the seam filled; the key frames carry the whole too
            gcore = self.group_core(name, frames)
            self._final_soft = {}
            zres = self.apply_zones(name, frames, gcore, keys, kinds=("brush",))
            self._band_ref = (name, zres.core if zres else gcore)
            core2 = self._shifted_core(zres.core if zres else gcore, obj, keys, zres)
            keys = {f: core2[f] for f in keys if f in core2}
        else:
            core2, zres = self._final_core(name, frames, obj, keys, kinds=("brush",) if stage == "matanyone" else None)
            keys = self._keys_with_brush(name, keys, core2)   # stage-2 corrections reach the key frames of the edge pass
        attempts0 = int(obj["tool"].get("auto_attempts", 0))
        if preview_dir is not None:
            full, _, _, _ = self._finalize_once(name, obj, segs, frames, core2, keys, start, mode, progress_cb, zres,
                                                run_dir=preview_dir, with_qa=False)
            return full
        full, rep, run_dir, qa = self._finalize_once(name, obj, segs, frames, core2, keys, start, mode, progress_cb, zres)
        if stage == "matanyone":
            import shutil
            # every frame of the base version keeps its alpha_base (frames without a soft override: = alpha), so
            # stage 4 „Zastosuj na ujęcie” can lay the ViTMatte zones on it
            for seg in segs:
                for f in frames:
                    a, b = run_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", run_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr"
                    if a.exists() and not b.exists():
                        b.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(a, b)
                        except OSError:
                            shutil.copyfile(a, b)
            rj = run_dir / "run.json"
            d = json.loads(rj.read_text(encoding="utf-8"))
            d.update(stage="matanyone", core_fix_sig=mask_sig(self.store, name), settings_sig=settings_sig(obj))
            if members:
                d.update(group=members, group_seam_px=obj.get("part_seam_px"),
                         group_outside_crop_px=self._crop_outside(segs, frames, core2))
            rj.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        # 7.8 pkt 1: edge position vs the plate -> report + saved measurements; backlit mode is never corrected
        job_progress("Raport krawędzi", 0, len(frames))
        edge = self.edge_report(segs, frames, full, core2, save_dir=run_dir / "edge")
        fixable = bool(edge["n_fix"]) and mode != "backlit"
        qa.edge = {**qa.edge, "before": edge["median_abs"], "pieces_to_fix": edge["n_fix"], "corrected": False,
                   "fix_available": fixable, "pieces": edge["pieces"]}
        (run_dir / "edge" / "edge.json").write_text(json.dumps(
            {"schema": "matte_desk/edge_fix@1.0", "mode": mode, "fix": [p["id"] for p in edge["pieces"]],
             "attempts": attempts0}, indent=1), encoding="utf-8")
        (run_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        attempts = attempts0 + (0 if qa.gate_ok else 1)
        extra = {}
        if stage == "matanyone":
            extra["stage_runs"] = {**(self.store.get(name)["tool"].get("stage_runs") or {}), "matanyone": run_dir.name}
        self.store.update(name, ma_scale=rep.ma_scale, tool={"active_run": run_dir.name, "final_valid": True,
                                                             "status": "QA OK" if qa.gate_ok else "QA: flagi",
                                                             "auto_attempts": attempts, "stale_from": None, **extra})
        return full, rep, run_dir

    def apply_edge_fix(self, name: str, frames: Sequence[int], base_run: Optional[str] = None,
                       skip: Sequence[str] = (), progress_cb=None):
        """'Zastosuj poprawkę krawędzi' (Z27 pkt 1): the former automatic second round, as a new runs/v###.
        Uses the measurements saved by `finalize` in `<base_run>/edge/`; `skip` = piece ids the user unticked
        (never shifted). Thin parts are not position-corrected (a 1-3 px shift would erase them)."""
        from engine.edge import apply_shift
        from engine.lupa import thin_parts
        obj = self.store.get(name)
        self._anchor_names = [name]
        run = base_run or obj["tool"].get("active_run")
        if not run:
            raise ValueError("brak wersji – najpierw „Przelicz finalnie”")
        rdir = self.store.dir(name) / "runs" / run
        try:
            if json.loads((rdir / "run.json").read_text(encoding="utf-8")).get("group"):
                raise ValueError(f"{run} to wersja grupy (rodzic + części) – poprawka krawędzi działa na wersjach pojedynczego obiektu")
        except OSError:
            pass
        ej = rdir / "edge" / "edge.json"
        if not ej.exists():
            raise ValueError(f"{run}: brak pomiaru krawędzi – policz „Przelicz finalnie” ponownie")
        info = json.loads(ej.read_text(encoding="utf-8"))
        if info.get("mode") == "backlit":
            raise ValueError("tryb backlit: poprawka krawędzi nie jest stosowana")
        skip = {str(x) for x in skip}
        fix = [p for p in info.get("fix", []) if p not in skip]
        if not fix:
            raise ValueError("brak odcinków do poprawy (wszystkie odznaczone albo brak)")
        from engine.workarea import run_frames
        frames = [f for f in frames if f in set(run_frames(self.store.dir(name), run) or frames)]  # work area of the base
        segs = self._segments_in(name, frames)
        keys = self.keys(name)
        start = obj["tool"].get("start_frame") or min(keys)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        core2, zres = self._final_core(name, frames, obj, keys)
        keys = self._keys_with_brush(name, keys, core2)
        disp = self.edge_disp(rdir / "edge", segs, frames, core2, set(fix))
        core3 = {f: (core2[f] if f in keys or f not in disp
                     else apply_shift(core2[f], disp[f]) | thin_parts(core2[f])) for f in frames}
        if zres:  # pixels removed by zones never come back (7.7 pkt 4)
            core3 = {f: core3[f] & ~zres.removed[f] for f in frames}
        full, rep, run_dir, qa = self._finalize_once(name, obj, segs, frames, core3, keys, start, info["mode"],
                                                     progress_cb, zres, attempts=int(info.get("attempts", 0)))
        after = self.edge_report(segs, frames, full, core3, save_dir=run_dir / "edge")
        base_qa = rdir / "qa.json"
        before = json.loads(base_qa.read_text(encoding="utf-8")).get("edge", {}).get("before") if base_qa.exists() else None
        qa.edge = {**qa.edge, "before": before, "pieces_to_fix": len(fix), "corrected": True, "after": after["median_abs"],
                   "base_run": run, "skipped": sorted(skip), "fix_available": False, "pieces": after["pieces"]}
        (run_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        (run_dir / "edge" / "edge.json").write_text(json.dumps(
            {"schema": "matte_desk/edge_fix@1.0", "mode": info["mode"], "fix": [], "attempts": int(info.get("attempts", 0)),
             "applied_from": run, "skipped": sorted(skip)}, indent=1), encoding="utf-8")
        attempts = int(obj["tool"].get("auto_attempts", 0)) + (0 if qa.gate_ok else 1)
        self.store.update(name, ma_scale=rep.ma_scale, tool={"active_run": run_dir.name, "final_valid": True,
                                                             "status": "QA OK" if qa.gate_ok else "QA: flagi",
                                                             "auto_attempts": attempts, "stale_from": None})
        return full, rep, run_dir

    # ------------------------------------------------------------------ motion blur of the alpha (2026-10-16)
    def _mb_base(self, name: str, run: Optional[str]) -> str:
        obj = self.store.get(name)
        run = run or obj["tool"].get("active_run")
        if not run or not (self.store.dir(name) / "runs" / run / "run.json").exists():
            raise ValueError("brak wersji – najpierw „▶ Przelicz MatAnyone2” (etap 3) albo „Przelicz finalnie”")
        return run

    def _mb_gray(self, shot: Sequence[int]):
        """Gray (uint8) of plate frame f for the flow, None outside the shot; a few frames cached (frames go in order)."""
        from engine.motionblur import to_gray_u8
        inside, cache = set(shot), {}

        def gray(f: int):
            if f not in inside:
                return None
            if f not in cache:
                if len(cache) >= 4:
                    cache.pop(next(iter(cache)))
                cache[f] = to_gray_u8(self.loader.read(f))
            return cache[f]
        return gray

    def _mb_shutter(self, shutter_deg: Optional[float]) -> float:
        from engine.motionblur import clamp_shutter
        return float(self.shutter) if shutter_deg is None else clamp_shutter(shutter_deg)

    def motion_blur_frame(self, name: str, f: int, shot: Sequence[int], run: Optional[str] = None,
                          strength: float = 1.0, shutter_deg: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preview of „Motion blur alfy” in one frame: the version's full-frame alpha with the object's motion blur
        (`shutter_deg` None = the manifest's shutter)."""
        from engine.export import run_alpha_loader
        from engine.motionblur import clamp_strength, motion_blur_alpha
        run = self._mb_base(name, run)
        sh = self._mb_shutter(shutter_deg)
        alpha = run_alpha_loader(self.store.dir(name), run, self.w, self.h)(f)
        gray = self._mb_gray(shot)
        out, st = motion_blur_alpha(alpha, gray(f), gray(f - 1), gray(f + 1), sh, clamp_strength(strength))
        return out, {**st, "run": run, "strength": clamp_strength(strength), "shutter_deg": sh}

    def apply_motion_blur(self, name: str, frames: Sequence[int], run: Optional[str] = None, strength: float = 1.0,
                          progress_cb=None, shutter_deg: Optional[float] = None,
                          device: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
        """„Zastosuj motion blur” -> a new runs/v### from the base version: every frame's alpha with the object's motion
        blur (engine/motionblur.py); core of the base kept (hard links). A version without zones (stage 3 MatAnyone2, mask
        1+2) gets the blurred alpha as its alpha_base too, a version with zones keeps the base's alpha_base.
        A stage's result (tool.stage_runs: matanyone, vitmatte@…) stays that stage's result: the new version inherits the
        stage keys and replaces the base in stage_runs (user decision 2026-10-16: ViTMatte on the MatAnyone2 base then
        reads the blurred MatAnyone2 – fine for tests); any other version becomes a plain version (applied_from = base)."""
        import shutil
        from engine.export import run_alpha_loader
        from engine import align_gpu
        from engine.motionblur import MB_DEFAULTS, clamp_strength, mb_device, motion_blur_alpha
        from engine.refine import write_alpha_exr
        from engine.workarea import run_frames
        run = self._mb_base(name, run)
        strength = clamp_strength(strength)
        sh = self._mb_shutter(shutter_deg)
        if strength <= 0 or sh <= 0:
            raise ValueError("siła albo shutter motion blura = 0 – nic do zrobienia")
        od = self.store.dir(name)
        rdir = od / "runs" / run
        rj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        segs = [CropSegment.from_dict(d) for d in rj.get("segments_def", [])]
        shot = list(frames)
        work = [f for f in shot if f in set(run_frames(od, run) or shot) and any(s.first <= f <= s.last for s in segs)]
        if not work:
            raise ValueError(f"{run}: brak klatek wersji")
        load = run_alpha_loader(od, run, self.w, self.h)
        gray = self._mb_gray(shot)
        new_dir = next_run_dir(od)
        full: Dict[int, np.ndarray] = {}
        moving, longest = 0, 0.0
        no_zones = not rj.get("zones")
        dev, dev_note = mb_device(device)          # GPU worker of the plate alignment (auto: when CUDA, Comfy idle)
        gpu = align_gpu.open_aligner() if dev == "gpu" else None
        strict = (device or os.environ.get("MD_MB_DEVICE") or "").lower() == "gpu"

        def link(src: Path, dst: Path):
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)

        try:
            for i, f in enumerate(work):
                job_progress("Motion blur alfy" + (" (GPU)" if gpu else ""), i + 1, len(work))
                a0 = load(f)
                try:
                    out, st = motion_blur_alpha(a0, gray(f), gray(f - 1), gray(f + 1), sh, strength, gpu=gpu)
                except align_gpu.GpuWorkerError as e:
                    if strict:
                        align_gpu.close_aligner(gpu)
                        raise ValueError(f"motion blur na GPU: {e}") from None
                    align_gpu.close_aligner(gpu)          # auto: the rest on the CPU
                    gpu, dev, dev_note = None, "cpu", f"GPU przerwane ({e}) – reszta na CPU"
                    out, st = motion_blur_alpha(a0, gray(f), gray(f - 1), gray(f + 1), sh, strength)
                full[f] = out
                moving += st["moving_px"]
                longest = max(longest, st["max_px"])
                for seg in segs:
                    if not (seg.first <= f <= seg.last) or not (rdir / "alpha" / seg.seg_id / f"a_{f:04d}.exr").exists():
                        continue
                    write_alpha_exr(new_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", extract_crop(out, seg))
                    if no_zones:
                        link(new_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", new_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr")
                    keep = [("core", f"c_{f:04d}.png")] + ([] if no_zones else [("alpha_base", f"a_{f:04d}.exr")])
                    for sub, nm in keep:
                        if (rdir / sub / seg.seg_id / nm).exists():
                            link(rdir / sub / seg.seg_id / nm, new_dir / sub / seg.seg_id / nm)
                if progress_cb:
                    progress_cb("Motion blur alfy", i + 1, len(work))
        finally:   # also on „Przerwij” / an error: never leave the worker running
            align_gpu.close_aligner(gpu)
        mb = {"strength": strength, "shutter_deg": sh, "shutter_manifest": float(self.shutter), "min_px": MB_DEFAULTS["min_px"],
              "max_px": MB_DEFAULTS["max_px"], "base_run": run, "moving_px_mean": int(moving / max(1, len(work))),
              "longest_px": round(longest, 2), "device": dev, "device_note": dev_note}
        tool = self.store.get(name)["tool"]
        sruns = dict(tool.get("stage_runs") or {})
        of_stage = [k for k, v in sruns.items() if v == run]
        rj2 = {**rj, "run": new_dir.name, "motion_blur": mb, "created_at": datetime.now().isoformat(timespec="seconds")}
        if of_stage:   # stays the stage's result (stage keys, applied_from of the stage kept)
            sruns.update({k: new_dir.name for k in of_stage})
        else:
            rj2["applied_from"] = run
            for k in ("stage", "core_fix_sig", "settings_sig", "matte_sig", "base"):
                rj2.pop(k, None)
        (new_dir / "run.json").write_text(json.dumps(rj2, indent=2, ensure_ascii=False), encoding="utf-8")
        qa = self.run_qa(name, work, full, segs, new_dir)
        qa.edge = {**qa.edge, "corrected": False, "fix_available": False, "motion_blur": mb}
        (new_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, tool={"active_run": new_dir.name, "final_valid": True, "stage_runs": sruns,
                                      "status": "QA OK" if qa.gate_ok else "QA: flagi", "stale_from": None})
        mb["stages"] = of_stage
        return new_dir, mb

    def _brush_anchors(self, name: str, frames: Sequence[int], core: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Key frames of the enabled stage-2 brush corrections that carry a drawing (an empty key only ends a correction)
        -> the core passed to the edge mode there (the mask 1+2): extra MatAnyone2 anchors, so its memory keeps what the
        brush added / removed between the SAM 3.1 keys."""
        from engine.stages import zone_kind
        from engine.zones import load_zones, zone_keys
        fs = set(frames)
        out: Dict[int, np.ndarray] = {}
        names = getattr(self, "_anchor_names", None) or [name]   # a group: the brush keys of every member
        zones = [z for n in names for z in load_zones(self.store.dir(n))]
        for z in zones:
            if not z.enabled or zone_kind(z) != "brush":
                continue
            for f, k in zone_keys(z).items():
                drawn = len(k.get("polygon") or []) >= 3 or any(not s.get("erase") for s in (k.get("strokes") or []))
                if drawn and f in fs and f in core:
                    out[int(f)] = core[f]
        return out

    def _finalize_once(self, name, obj, segs, frames, core, keys, start, mode, progress_cb, zres, attempts=None,
                       run_dir: Optional[Path] = None, with_qa: bool = True):
        run_dir = run_dir if run_dir is not None else next_run_dir(self.store.dir(name))
        fin = self._finalizer()
        fin.extra_anchors = self._brush_anchors(name, frames, core) if mode == "matanyone" else {}
        keep = None
        if zres:
            keep = {f: zres.removed[f] for f in frames}
        core_soft = None
        if core_soft_start_on(obj) and getattr(self, "_final_soft", None):
            # the maps follow the core actually passed (zones, shrink / grow, edge fix win with a hard 0 / 1)
            from engine.corestab import consistent_soft
            core_soft = {f: consistent_soft(p, core[f]) for f, p in self._final_soft.items() if f in core and f not in keys}
        from engine.refine import ZONE_FEATHER_PX
        full, rep = fin.finalize(obj, segs, frames, core, keys, start, run_dir, mode, progress_cb,
                                 exclude=zres.removed if zres else None, soft_override=zres.soft if zres else None,
                                 band_fn=self._band_fn(obj, frames), keep_holes=keep, core_soft=core_soft,
                                 soft_feather=(zres.feather_px if zres and zres.feather_px else ZONE_FEATHER_PX),
                                 soft_weights=zres.weights if zres else None)
        if not with_qa:   # stage preview: the alpha only
            return full, rep, run_dir, None
        qa = self.run_qa(name, frames, full, segs, run_dir, removed=zres.removed if zres else None, attempts=attempts,
                         soft_areas={f: a for f, (_, a) in zres.soft.items()} if zres and zres.soft else None)
        qa.hygiene = rep.hygiene
        from engine.workarea import intervals_from_frames, object_life
        life = object_life(obj)
        if life is not None:      # life range: the version computed only the visible pieces (run_coverage reads it)
            rj = run_dir / "run.json"
            d = json.loads(rj.read_text(encoding="utf-8"))
            d["coverage"], d["life"] = intervals_from_frames(frames), life
            rj.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        if (zres and (zres.stats or zres.flags)) or core_stabilize_on(obj):
            rj = run_dir / "run.json"
            d = json.loads(rj.read_text(encoding="utf-8"))
            if zres and (zres.stats or zres.flags):
                d["zones"] = zres.stats
            if core_stabilize_on(obj):
                d["core_stabilize"] = True
                d["core_stabilize_stats"] = getattr(self, "_corestab_stats", None)
            rj.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            for f, fl in (zres.flags.items() if zres else ()):   # zone hints as yellow QA flags (e.g. engines disagree, no holdout)
                if f in qa.flags:
                    qa.flags[f].extend(("yellow", x) for x in fl)
                    if qa.colours.get(f) == "green":
                        qa.colours[f] = "yellow"
        return full, rep, run_dir, qa

    def edge_preview(self, name: str, f: int, band_scale: Optional[float] = None,
                     shift_px: Optional[float] = None, band_out_px: Optional[float] = None,
                     band_in_px: Optional[float] = None, core_key: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preview of the MatAnyone2 edge before computing (stage 3 palette, user request 2026-10-15): full-frame uint8,
        255 = the mask the edge pass keeps (mask 1+2 + „Przesuń środek konturu”), 128 = the adaptive band where MatAnyone2 decides
        (same `_band_fn` as „Przelicz MatAnyone2”, with the slider values not yet saved), 0 = background.
        The mask is built the way `finalize(stage="matanyone")` builds it (2026-10-16; was: raw core_fix / core PNGs):
        stabilised SAM 3.1 core + brush corrections, a group = the whole of its members (group_core) in the group crops
        (_work_segments), then the shrink / grow shift. The stabilisation votes over +-2 frames, so a window of
        f-2..f+2 (inside the object's life range) is computed. `core_key` (app stage_band_lines): a signature of
        everything the shifted core depends on – NOT the band points – so a band point edit reuses the stabilised core
        (~2 s of the ~3.7 s, 2026-10-23); the band itself is always recomputed."""
        from engine.crop import paste_crop
        from engine.edge import clamp_band_scale, clamp_shift_px
        obj = dict(self.store.get(name))
        if band_scale is not None:
            obj["band_scale"] = clamp_band_scale(band_scale)
        if shift_px is not None:
            obj["shift_px"] = clamp_shift_px(shift_px)
        from engine.edge import band_px_limits, clamp_band_px
        if band_out_px is not None or band_in_px is not None:   # px mode values not saved yet (the palette sliders)
            obj["band_out_px"] = clamp_band_px(band_out_px if band_out_px is not None else (obj.get("band_out_px") or 0.0))
            obj["band_in_px"] = clamp_band_px(band_in_px if band_in_px is not None else (obj.get("band_in_px") or 0.0))
        shift = 0.0 if band_px_limits(obj) else float(obj.get("shift_px", 0.0) or 0.0)
        segs = [s for s in self._segments_in(name, [f]) if s.first <= f <= s.last]
        if not segs:
            raise ValueError(f"Klatka {f} poza kadrami obiektu – „▶▶ Przelicz całość” (etap 1)")
        win = [x for x in range(max(segs[0].first, f - 2), min(segs[0].last, f + 2) + 1)]
        try:
            win = self._visible(name, win)
        except ValueError:
            raise ValueError(f"obiekt niewidoczny w klatce {f} (zakres życia)")
        if f not in win:
            raise ValueError(f"obiekt niewidoczny w klatce {f} (zakres życia)")
        hit = _EDGE_CORE_MEMO.get(core_key) if core_key else None
        if hit is not None:
            core, src, ref = hit
            core = core.copy()
            self._band_ref = (name, {k: v.copy() for k, v in ref.items()})
        try:
            if hit is None:
                core, src = self._edge_preview_core(name, f, win, obj)
                if core_key:
                    ref = {k: v.copy() for k, v in (self._band_ref[1] or {}).items() if k == f}
                    with _CORE_LRU_LOCK:
                        _EDGE_CORE_MEMO[core_key] = (core.copy(), src, ref)
                        while len(_EDGE_CORE_MEMO) > 12:
                            _EDGE_CORE_MEMO.popitem(last=False)
        except FileNotFoundError:
            raise ValueError(f"Brak rdzenia w klatce {f} – „▶▶ Przelicz całość” (etap 1)")
        seg = [s for s in self._work_segments(name, [f]) if s.first <= f <= s.last][0]
        c = extract_crop(core.astype(np.float32), seg) > 0.5
        band = self._band_fn(obj)(f, seg, c, None)
        out = np.zeros((self.h, self.w), np.uint8)
        out[core] = 255
        out[paste_crop(band.astype(np.float32), seg, self.w, self.h) > 0.5] = 128
        return out, {"source": src, "band_scale": obj.get("band_scale", 1.0), "shift_px": shift,
                     "band_px_mode": band_px_limits(obj),
                     "band_px": int((out == 128).sum()), "crop": list(seg.crop),
                     "outside_crop_px": self._crop_outside([seg], [f], {f: core}), "core": core, "seg": seg}

    def _edge_preview_core(self, name: str, f: int, win: List[int], obj: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """The shifted mask 1+2 of edge_preview (sets self._band_ref = the edge before the shift)."""
        keys = self.keys(name)
        members = self.group_members(name)
        disk = self._mask12_disk(name, f)
        if disk is not None:   # the mask 1+2 on disk is current: read, not recomputed (2026-10-22, ~0.1 s)
            self._band_ref = (name, {f: disk})
            core = self._shifted_core({f: disk}, obj, keys, None)[f]
            src = "mask12 (core_fix)" + ("+group" if members else "")
        elif members:   # a group: the whole (members' masks 1+2 joined with the seam), as stage 3 computes it
            gcore = self.group_core(name, win, only=[f])   # one frame shown: only it is stabilised
            zres = self.apply_zones(name, win, gcore, keys, kinds=("brush",))
            self._band_ref = (name, zres.core if zres else gcore)
            core = self._shifted_core(zres.core if zres else gcore, obj, keys, zres)[f]
            src = "mask12+group"
        else:
            core = self._final_core(name, win, obj, keys, kinds=("brush",), only=[f])[0][f]
            src = "mask12"
        return core, src

    def _band_fn(self, obj, frames: Optional[Sequence[int]] = None):
        """Adaptive band (7.8 pkt 2) on the plate-to-mask crop: focus blur + motion blur along the normal.
        The widths are measured on the edge BEFORE the shrink / grow shift (`self._band_ref`, set by _final_core /
        the group branch: the real edge on the plate) and laid around the shifted core: measured on the shifted
        contour they grew inside the curls and cancelled the shift (user report CLIP2_A F62, 2026-10-16).
        px mode with band points (one band system, 2026-10-22: object field `edge_band`): the points shape the px widths
        (the sliders size them); `frames` = the run's frames (tracking awaited), None = a preview of one frame."""
        from engine.edge import adaptive_band, adaptive_band_width, band_px, band_px_limits
        lim = band_px_limits(obj)
        from engine.objband import SHARED_OWNER, edge_band, mode
        eb = edge_band(obj) or {}       # the shared object band unless stage 3 is detached (2026-10-22)
        owner = SHARED_OWNER if mode(obj, "matanyone") == "shared" else "stage_matanyone"
        pts_of = None
        if lim and eb.get("band_points"):
            ro, ri = eb.get("band_ref_out"), eb.get("band_ref_in")
            sc = ((lim[0] / float(ro)) if ro else 1.0, (lim[1] / float(ri)) if ri else 1.0)
            pf_all = self.band_points_fn(obj["name"], eb, owner, list(frames), None, sc) if frames else None
            def pts_of(f):
                pf = pf_all or self.band_points_fn(obj["name"], eb, owner, [f], [f], sc)
                return pf(f) if pf else []
        scale = 1.0 if lim else float(obj.get("band_scale", 1.0))
        ref_name, ref = getattr(self, "_band_ref", None) or (None, None)
        if ref_name != obj.get("name"):
            ref = None
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        last = {}   # (frame, seg_id) -> gray of the previous call: frames come in order, so f-1 is usually here
                    # (one plate decode per frame instead of two – the crop LRU is long gone after the MatAnyone pass)

        def gray(f, seg):
            k = (f, seg.seg_id)
            if k in last:
                return last[k]
            g = self.loader.crop(f, seg).mean(axis=2)
            last.clear()
            last[k] = g
            return g

        def fn(f, seg, core_crop, _bw):
            prev = f - 1
            gp = gray(prev, seg) if seg.first <= prev else None
            g = gray(f, seg)
            flow = None
            if gp is not None:
                u8 = lambda x: np.clip(x * 255, 0, 255).astype(np.uint8)
                flow = dis.calc(u8(gp), u8(g), None)
            mc = core_crop
            if ref is not None and f in ref:
                mc = extract_crop(ref[f].astype(np.float32), seg) > 0.5
            wmap = adaptive_band_width(mc, g, flow, self.shutter, scale)
            if lim:
                o, i = lim
                if pts_of:
                    from engine.bandsys import band_widths
                    x0, y0 = seg.crop[0], seg.crop[1]
                    pts = [(x - x0, y - y0, po, pi) for x, y, po, pi in pts_of(f)]   # plate px -> crop px
                    o, i = band_widths(core_crop, lim[0], lim[1], pts)
                return band_px(core_crop, wmap, o, i)
            return adaptive_band(core_crop, wmap)
        return fn

    def edge_report(self, segs, frames, full, core, save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Edge offsets per frame (crop of the first covering segment) -> pieces needing correction (report).
        `save_dir`: per-frame measurements (`e_####.npz`) for `edge_disp` / the stage-4 overlay."""
        from engine.edge import measure_edge, needs_correction
        from concurrent.futures import ThreadPoolExecutor
        from engine.qa import default_workers
        per, meas_by_f, vals = [], {}, []
        cov_of = {f: next((s for s in segs if s.first <= f <= s.last), None) for f in frames}
        # crops read sequentially (plate-to-mask cache may be built on first use), edge measured in parallel;
        # only contour samples inside the plate count (the mirror-padded part of a crop has no real edge)
        inputs = {f: (extract_crop(full[f].astype(np.float32), s), self.loader.crop(f, s).mean(axis=2),
                      crop_plate_mask(s)) for f, s in cov_of.items() if s is not None}
        with ThreadPoolExecutor(max_workers=default_workers()) as pool:
            res = dict(zip(inputs, pool.map(lambda f: measure_edge(inputs[f][0], inputs[f][1], valid=inputs[f][2]),
                                            list(inputs))))
        for f in frames:
            s = cov_of[f]
            if s is None:
                per.append({})
                continue
            m = res[f]
            meas_by_f[f] = (s, m)
            per.append(m.piece_median)
            vals += [abs(v) for v in m.piece_median.values()]
        fix = needs_correction(per)
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for f, (s, m) in meas_by_f.items():
                save_edge_measure(save_dir / f"e_{f:04d}.npz", s, m)
        return {"median_abs": round(float(np.median(vals)) if vals else 0.0, 3),
                "n_fix": int(sum(fix.values())), "pieces": edge_pieces(meas_by_f, fix)}

    # ------------------------------------------------------------------ stage Eksperymenty: analysis of a chosen base
    def analysis_fix_why(self, rj: Dict[str, Any]) -> str:
        """'' when the edge correction can start from this version (2026-10-16: „Przelicz z tej bazy” shifts the BASE's
        edge and computes it with the chosen engine – any version), else why not."""
        return ""   # kept as the hook for future limits; every base can be shifted and recomputed today

    def edge_recount(self, rdir: Path, thr_px: float = 0.5, min_frames: int = 5) -> Dict[str, Any]:
        """Edge-correction proposals of a version from its SAVED measurements (edge/e_####.npz) with these thresholds:
        |median| > thr_px for >= min_frames frames (fast – no measuring, the palette sliders)."""
        from engine.edge import needs_correction
        rj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        segs = {d["seg_id"]: CropSegment.from_dict(d) for d in rj.get("segments_def", [])}
        meas_by_f, per, vals = {}, [], []
        for p in sorted((rdir / "edge").glob("e_*.npz")):
            f = int(p.stem[2:])
            sid, m = load_edge_measure(p)
            if sid not in segs:
                continue
            meas_by_f[f] = (segs[sid], m)
            per.append(m.piece_median)
            vals += [abs(v) for v in m.piece_median.values()]
        fix = needs_correction(per, thr=float(thr_px), run=int(min_frames)) if per else {}
        return {"median_abs": round(float(np.median(vals)) if vals else 0.0, 3), "n_fix": int(sum(fix.values())),
                "pieces": edge_pieces(meas_by_f, fix), "measured_frames": len(meas_by_f)}

    def analyze_run(self, name: str, run: str, thr_px: float = 0.5, min_frames: int = 5, progress_cb=None) -> Dict[str, Any]:
        """Stage Eksperymenty (user request 2026-10-16): QA and edge-correction proposals of the CHOSEN base version
        (MatAnyone2 / ViTMatte / LTX-2.5 / mask 1+2 / any version): measured once into the version's own folder (qa.json,
        flicker_map.png, edge/e_####.npz) when missing, then the proposals counted with the thresholds -> qa.json
        `edge` + edge/edge.json (what „Zastosuj poprawkę krawędzi” reads)."""
        from engine.export import run_alpha_loader
        od = self.store.dir(name)
        rdir = od / "runs" / run
        rj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        segs = [CropSegment.from_dict(d) for d in rj.get("segments_def", [])]
        lo, hi = rj["frames"]
        from engine.workarea import run_frames
        cov = run_frames(od, run)
        frames = [f for f in range(int(lo), int(hi) + 1) if not cov or f in set(cov)]
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        ld = run_alpha_loader(od, run, self.w, self.h)

        class _Lazy(dict):   # full-frame alphas read on demand (a long 4K shot does not fit in RAM at once)
            def __missing__(self, f):
                return ld(f)
        full = _Lazy()
        if not (rdir / "qa.json").exists():
            job_progress("QA wersji", 0, len(frames))
            self.run_qa(name, frames, full, segs, rdir)
        if not any((rdir / "edge").glob("e_*.npz")):
            job_progress("Pomiar krawędzi", 0, len(frames))
            self.edge_report(segs, frames, full, None, save_dir=rdir / "edge")
        rc = self.edge_recount(rdir, thr_px, min_frames)
        why = self.analysis_fix_why(rj)
        qa = json.loads((rdir / "qa.json").read_text(encoding="utf-8"))
        prev = qa.get("edge") or {}
        if not prev.get("corrected"):
            qa["edge"] = {**prev, "before": rc["median_abs"], "pieces_to_fix": rc["n_fix"], "corrected": False,
                          "fix_available": bool(rc["n_fix"]) and not why, "fix_why": why, "pieces": rc["pieces"],
                          "thr_px": float(thr_px), "min_frames": int(min_frames)}
            (rdir / "qa.json").write_text(json.dumps(qa, indent=1, ensure_ascii=False), encoding="utf-8")
            ej = rdir / "edge" / "edge.json"
            old = json.loads(ej.read_text(encoding="utf-8")) if ej.exists() else {}
            (rdir / "edge").mkdir(parents=True, exist_ok=True)
            ej.write_text(json.dumps({"schema": "matte_desk/edge_fix@1.0", "mode": old.get("mode", rj.get("edge_mode")),
                                      "fix": [p["id"] for p in rc["pieces"]], "attempts": old.get("attempts", 0),
                                      "thr_px": float(thr_px), "min_frames": int(min_frames)}, indent=1), encoding="utf-8")
        if progress_cb:
            progress_cb("Analiza wersji", len(frames), len(frames))
        return {"run": run, "frames": [frames[0], frames[-1]], "edge": qa.get("edge"), "fix_why": why,
                "flicker_median": qa.get("flicker_median"), "gate_ok": qa.get("gate_ok")}

    EXPERIMENT_MODES = ("matanyone", "backlit", "core")

    def experiment_from_base(self, name: str, base_run: str, mode: str = "matanyone", fix: Optional[Sequence[str]] = None,
                             progress_cb=None) -> str:
        """Stage Eksperymenty, „Przelicz z tej bazy” (user decision 2026-10-16, replaces „Przelicz finalnie”): the mask of
        the chosen BASE version (alpha > 0.5 – MatAnyone2 / ViTMatte / LTX-2.5 / mask 1+2 / any version; a group version =
        the whole) -> the threshold zones of the object -> optionally the edge correction of the pieces `fix` (measured
        on the base by analyze_run: the BASE's edge is shifted) -> edge mode `mode` -> a new version with QA + edge
        report; run.json records where it came from (`base_run`, `base_stage`). Versions are never rewritten, so the
        origin stays valid; the UI tells when the base is no longer its stage's current result."""
        from engine.edge import apply_shift
        from engine.export import run_alpha_loader
        from engine.lupa import thin_parts
        if mode not in self.EXPERIMENT_MODES:
            raise ValueError(f"silnik krawędzi: {' | '.join(self.EXPERIMENT_MODES)}")
        od = self.store.dir(name)
        rdir = od / "runs" / base_run
        brj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        segs = [CropSegment.from_dict(d) for d in brj.get("segments_def", [])]
        from engine.workarea import run_frames
        cov = run_frames(od, base_run)
        lo, hi = brj["frames"]
        frames = [f for f in self._visible(name, list(range(int(lo), int(hi) + 1))) if not cov or f in set(cov)]
        if not frames:
            raise ValueError(f"{base_run}: brak klatek do policzenia")
        obj = self.store.get(name)
        keys = self.keys(name)
        if self.loader.frames is None:
            self.loader.frames = list(frames)
        ld = run_alpha_loader(od, base_run, self.w, self.h)
        core = {f: ld(f) > 0.5 for f in frames}
        self._final_soft = {}          # the base is a hard mask: no SAM3 soft start
        self._band_ref = (name, core)  # the band widths measured on the base's own edge
        members = brj.get("group")
        self._anchor_names = members or [name]
        zres = self.apply_zones(name, frames, core, keys, kinds=("threshold",))
        core2 = zres.core if zres else core
        fix = [str(x) for x in (fix or [])]
        if fix:
            if not any((rdir / "edge").glob("e_*.npz")):
                raise ValueError(f"{base_run}: brak pomiaru krawędzi – „▶ Analizuj” na tej bazie")
            disp = self.edge_disp(rdir / "edge", segs, frames, core2, set(fix))
            core2 = {f: (core2[f] if f not in disp else apply_shift(core2[f], disp[f]) | thin_parts(core2[f])) for f in frames}
            if zres:
                core2 = {f: core2[f] & ~zres.removed[f] for f in frames}
        keys2 = {f: core2[f] for f in keys if f in core2}   # MatAnyone anchored on the BASE at the key frames
        start = obj["tool"].get("start_frame") or (min(keys) if keys else frames[0])
        if start not in core2:
            start = frames[0]
        full, rep, run_dir, qa = self._finalize_once(name, obj, segs, frames, core2, keys2, start, mode, progress_cb, zres)
        edge = self.edge_report(segs, frames, full, core2, save_dir=run_dir / "edge")
        qa.edge = {**qa.edge, "before": edge["median_abs"], "pieces_to_fix": edge["n_fix"], "corrected": bool(fix),
                   "fix_available": bool(edge["n_fix"]) and mode != "backlit" and not members,
                   "fix_why": "" if not members else "wersja grupy – poprawkę licz „z tej bazy” na nowej wersji",
                   "pieces": edge["pieces"], **({"base_run": base_run} if fix else {})}
        (run_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        (run_dir / "edge" / "edge.json").write_text(json.dumps(
            {"schema": "matte_desk/edge_fix@1.0", "mode": mode, "fix": [p["id"] for p in edge["pieces"]], "attempts": 0},
            indent=1), encoding="utf-8")
        rj = run_dir / "run.json"
        d = json.loads(rj.read_text(encoding="utf-8"))
        d.update(stage="experiment", base_run=base_run, base_stage=brj.get("stage") or "final",
                 base_edge_fix=fix, **({"group": members, "group_seam_px": brj.get("group_seam_px")} if members else {}))
        rj.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        self.store.update(name, ma_scale=rep.ma_scale, tool={"active_run": run_dir.name, "final_valid": True,
                                                             "status": "QA OK" if qa.gate_ok else "QA: flagi",
                                                             "stale_from": None})
        return run_dir.name

    def edge_disp(self, edge_dir: Path, segs, frames, core, fix_ids: set) -> Dict[int, np.ndarray]:
        """Full-frame SDF displacement maps of the pieces `fix_ids` from saved measurements (= the former
        in-memory second round: same medians, same correction_map)."""
        from engine.edge import correction_map
        by_id = {s.seg_id: s for s in segs}
        disp = {}
        for f in frames:
            p = Path(edge_dir) / f"e_{f:04d}.npz"
            if not p.exists():
                continue
            seg_id, m = load_edge_measure(p)
            s = by_id.get(seg_id)
            if s is None:
                continue
            m.piece_median = {k: v for k, v in m.piece_median.items() if piece_id(k) in fix_ids}
            c = extract_crop(core[f].astype(np.float32), s) > 0.5
            disp[f] = paste_crop(correction_map(c, m), s, self.w, self.h)
        return disp

    def front_holdout(self, name: str, frames: Sequence[int], need: bool = True):
        """Masks of the objects in front of `name` (smaller depth_order, not parts of the same object) for `matte`
        zones: 1) final alpha (> 0.5) of the active version, 2) the core; none -> MissingFront (plan section 6; never a
        silent run without holdout). Returns (holdout(f) -> mask | None, [names used]); `need=False` -> no raise."""
        from engine.export import run_alpha_loader
        from engine.zones import MissingFront
        me = self.store.get(name)
        same = {name, me.get("part_of")} - {None}
        fronts = [o for o in self.store.list()
                  if o["name"] != name and o.get("depth_order", 0) < me.get("depth_order", 0)
                  and o["name"] not in same and o.get("part_of") not in same]
        if not fronts:
            return (lambda f: None), []
        loaders, missing, queue = {}, [], []
        for o in fronts:
            od = self.store.dir(o["name"])
            run = o["tool"].get("active_run")
            if run and (od / "runs" / run / "run.json").exists():
                ld = run_alpha_loader(od, run, self.w, self.h)
                loaders[o["name"]] = lambda f, ld=ld: ld(f) > 0.5
            elif (od / "core").exists() and all((od / "core" / f"c_{f:04d}.png").exists() for f in frames):
                cd = od / "core"
                loaders[o["name"]] = lambda f, cd=cd: (cv2.imread(str(cd / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127)
                queue.append({"object": o["name"], "action": "finalize"})
            else:
                missing.append(o["name"])
                queue.append({"object": o["name"], "action": "run"})
                queue.append({"object": o["name"], "action": "finalize"})
        if missing and need:
            raise MissingFront(missing, queue)

        def holdout(f: int):
            if not loaders:
                return None
            m = np.zeros((self.h, self.w), bool)
            for ld in loaders.values():
                m |= ld(f)
            return m
        return holdout, sorted(loaders)

    def clean_source(self):
        """Clean plate for `matte` zones (docs/PLAN_CLEANPLATE.md): (loader f -> (img, conf) | None, label).
        1) the project's clean plate (cleanplate_align / cleanplate, confidence 1), 2) the auto clean plate built from the
        shot (`cleanplate_auto/`), 3) none."""
        if self.clean is not None:
            return (lambda f: (self.clean.frame(f), None)), "projekt"
        from engine.cleanplate_auto import load_cleanplate
        cp = load_cleanplate(self.work / "cleanplate_auto")
        if cp is not None:
            img, conf, _ = cp
            return (lambda f: (img, conf)), "auto"
        return None, "brak"

    def build_auto_cleanplate(self, frames: Sequence[int], grow_px: int = 40, max_samples: int = 60,
                              progress_cb=None) -> Dict[str, Any]:
        """„Policz z ujęcia (auto)” (plan 4b): temporal median of the pixels not covered by ANY object's core (+grow_px),
        holes never uncovered -> LaMa through Comfy (comfyui-inpaint-nodes) or OpenCV; written to <work>/cleanplate_auto/.
        Objects without a core -> error listing them (never a silent clean plate with an object baked in)."""
        from engine.cleanplate_auto import build_cleanplate, lama_available, lama_inpaint_comfy, write_cleanplate
        from engine.zones import MissingFront
        frames = list(frames)
        objs = self.store.list()
        missing = [o["name"] for o in objs
                   if not all((self.store.dir(o["name"]) / "core" / f"c_{f:04d}.png").exists() for f in frames)]
        if missing:
            raise MissingFront(missing, [{"object": n, "action": "run"} for n in missing])
        cores = {o["name"]: self.store.dir(o["name"]) / "core" for o in objs}

        def covered(f: int) -> np.ndarray:
            m = np.zeros((self.h, self.w), bool)
            for cd in cores.values():
                c = cv2.imread(str(cd / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if c is not None:
                    m |= c > 127
            return m
        inpaint = None
        if self.client is not None and lama_available(self.client):
            inpaint = lambda img, mask: lama_inpaint_comfy(self.client, self.work / "comfy_io", img, mask)
        clean, conf, stats = build_cleanplate(frames, self.loader.frame, covered, grow_px=grow_px, max_samples=max_samples,
                                              progress=job_progress, inpaint=inpaint)
        stats["objects"] = sorted(cores)
        pattern = write_cleanplate(self.work / "cleanplate_auto", clean, conf, frames, stats)
        return {**stats, "pattern": pattern}

    # ------------------------------------------------------------------ zones (step 4: keys, propagation, cache, apply)
    def _zone_tracker(self):
        tracker = self.zone_tracker
        if tracker is None:
            try:
                from engine.tracking import make_point_tracker
                t = make_point_tracker()
                tracker = t if t.engine == "cotracker3" else None   # spec: CoTracker3, fallback = core bbox shift
            except Exception:
                tracker = None
        return tracker

    def _tracker_frames_for(self):
        s = min(1.0, 1024.0 / max(self.w, self.h))
        return lambda sub: ([cv2.resize(self.loader.frame(f), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for f in sub], s)

    BAND_TRACK_SIDE = 640         # longer side of the object-stabilised crop the band points are tracked on (px)

    def _band_tracker_frames(self, name: Optional[str] = None):
        """Tracker frames for the band points (engine/bandtrack.py). With `name`: a crop that FOLLOWS the object – the
        core's centroid, smoothed over ±2 frames, a fixed window = the largest bbox + margin, longer side
        BAND_TRACK_SIDE – so the object nearly stands still in it (user idea 2026-10-21: stabilise on the core; the
        beach pan moved ~190 px a frame) and CoTracker3 gets ~3x fewer pixels; returns (imgs, scale, crop origins).
        Images from the beauty proxies (JPEG ~7 ms a frame vs ~70 ms a 4K PNG) when they exist. Without `name` or
        cores: the whole frame at 1024 px, as _tracker_frames_for."""
        proxy = self.store.root.parent / "proxy" / "beauty"
        W, H = self.w, self.h

        def read(f):
            im = cv2.imread(str(proxy / f"p_{f:04d}.jpg"), cv2.IMREAD_COLOR) if proxy.is_dir() else None
            return (im, im.shape[1] / float(W)) if im is not None else (self.loader.frame(f), 1.0)

        def whole(sub):
            s = min(1.0, 1024.0 / max(W, H))
            return [cv2.resize(read(f)[0], (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) for f in sub], s

        def follow(sub):
            core_at = self.band_core_at(name)
            boxes = {}
            for f in sub:
                c = core_at(f)
                if c is not None and c.any():
                    ys, xs = np.nonzero(c)
                    boxes[f] = (xs.min(), ys.min(), xs.max(), ys.max(), xs.mean(), ys.mean())
            if not boxes:
                return whole(sub)
            bw = max(b[2] - b[0] for b in boxes.values()) + 1
            bh = max(b[3] - b[1] for b in boxes.values()) + 1
            m = 0.08 * max(bw, bh)
            cw, ch = int(min(W, bw + 2 * m)), int(min(H, bh + 2 * m))
            known = sorted(boxes)
            cen = np.array([boxes[min(known, key=lambda k: abs(k - f))][4:6] for f in sub], np.float64)
            r = 2                                    # smooth the centre: legs / hair move the centroid frame to frame
            sm = np.array([cen[max(0, i - r):i + r + 1].mean(0) for i in range(len(sub))])
            ox = np.clip(sm[:, 0] - cw / 2.0, 0, W - cw)
            oy = np.clip(sm[:, 1] - ch / 2.0, 0, H - ch)
            s = min(1.0, self.BAND_TRACK_SIDE / float(max(cw, ch)))
            tw, th = max(16, round(cw * s)), max(16, round(ch * s))
            imgs = []
            for i, f in enumerate(sub):
                im, p = read(f)
                x0, y0 = int(round(ox[i] * p)), int(round(oy[i] * p))
                crop = im[y0:y0 + int(round(ch * p)), x0:x0 + int(round(cw * p))]
                imgs.append(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
            return imgs, s, np.stack([ox, oy], axis=1)

        load = follow if name else whole
        load.variant = f"objcrop{self.BAND_TRACK_SIDE}" if name else ""
        return load

    def plate_sig(self, f: int) -> Any:
        """Signature of a plate frame for the zone cache: path + size + mtime (the display transform is fixed)."""
        try:
            p = frame_path(self.loader.pattern, f)
            st = p.stat()
            return [str(p), st.st_size, int(st.st_mtime)]
        except OSError:
            return [str(self.loader.pattern), f]

    def _lazy_core(self, name: str, core: Dict[int, np.ndarray]) -> "_LazyCore":
        """`core` + the core of any other shot frame loaded on first use (zone drawing frames outside the asked frames:
        a preview window, a work area – only the zone's bbox / tracking points are read from them)."""
        members = self.group_members(name)
        return _LazyCore(core, lambda f: (self.group_core(name, [f]) if members else self.load_core(name, [f]))[f])

    def zone_regions(self, name: str, z, frames: Sequence[int], core: Dict[int, np.ndarray],
                     only: Optional[Sequence[int]] = None):
        """Regions + forced-BG masks of a zone in every frame from its keys (zones.propagate_regions); the propagation
        transforms are cached per (keys, core signature) in zones_cache/<zone>/transforms_<hash>.npz."""
        from engine.zonecache import ZoneCache, _sha
        from engine.stages import zone_kind
        from engine.zones import (LocalWarp, outline_without_drawing, propagate_regions, region_mode, track_mode,
                                  zone_keys)
        shape = core[frames[0]].shape
        cache = ZoneCache(self.store.dir(name), z.zone_id)
        matte = z.criterion == "matte"
        brush = zone_kind(z) == "brush"        # stage 2 brush correction (add / remove on the core, 2026-10-15)
        local = (matte or brush) and track_mode(z) == "local"
        # objects the zone works around (2026-10-15): parts of this object (one matte with it, e.g. a hand on the head)
        # and objects in front – never CoTracker points on them; parts are also cut out of a matte region (sure foreground)
        fg_fn, fg_names = self.zone_fg(name, z, frames) if matte else (None, [])
        if brush:
            try:                                # a part without a mask yet: its points are simply not excluded
                fg_fn, fg_names = self.zone_fg(name, z, frames)
            except ValueError:
                fg_fn, fg_names = None, []
        front_fn, front_names = (self.front_holdout(name, frames, need=False) if (matte or brush)
                                 else ((lambda f: None), []))
        excl_names = sorted(set(fg_names) | set(front_names))
        sig = {"keys": zone_keys(z), "static": z.static, "frames": [frames[0], frames[-1]],
               "cores": [int(core[f].sum()) for f in frames[::max(1, len(frames) // 16)]]}
        if local or excl_names:            # absent = the old signature (old caches stay valid)
            sig.update(track="local" if local else "rigid", point_exclude=excl_names)
        tr_sig = _sha(sig)[:16]
        tcache: Dict[str, Dict[int, Any]] = {}
        if outline_without_drawing(z):
            # „whole outline” with nothing painted (user report 2026-10-17): no exclusion to follow, the area is the core
            # outline ± the band in every frame – no CoTracker, nothing to propagate
            regions = {f: np.zeros(shape, bool) for f in frames}
            bgs = {f: np.zeros(shape, bool) for f in frames}
            return self._zone_regions_finish(name, z, frames, core, only, regions, bgs,
                                             {"keys": sorted(zone_keys(z)), "tracked": False}, fg_fn, fg_names, matte)
        tracker = self._zone_tracker()
        if tracker is not None:
            tracker = _ReleaseOnUse(tracker, lambda: self.release_comfy_gpu("obszar strefy: śledzenie CoTracker3"))
        if local:
            for k, d in (cache.get_tracks(tr_sig) or {}).items():
                tcache[str(k)] = {f: LocalWarp(pts=d["pts"], dst=d["dst"][i], vis=d["vis"][i], fallback=d["fallback"][i])
                                  for i, f in enumerate(frames) if i < len(d["dst"])}
        else:
            raw = cache.get_transforms(tr_sig)
            if raw:
                for k, arr in raw.items():       # k = key frame, arr = (N, 2, 3) over `frames`
                    tcache[str(k)] = {f: arr[i] for i, f in enumerate(frames) if i < len(arr)}

        def point_exclude(k: int):
            ms = [m for m in ((fg_fn(k) if fg_fn else None), front_fn(k)) if m is not None]
            return np.logical_or.reduce(ms) if ms else None
        regions, bgs, tr, info = propagate_regions(z, frames, self._lazy_core(name, core), shape, self._tracker_frames_for() if tracker else None,
                                                   tracker, transforms_cache=tcache or None,
                                                   point_exclude=point_exclude if excl_names else None)
        if info.get("tracked"):
            eye = np.eye(2, 3, dtype=np.float32)
            if local:
                cache.put_tracks(tr_sig, {int(k): v for k, v in tr.items()}, frames)
            else:
                cache.put_transforms(tr_sig, {int(k): np.stack([np.asarray(v.get(f, eye), np.float32) for f in frames])
                                              for k, v in tr.items()})
        if excl_names:
            info = {**info, "point_exclude": excl_names}
        return self._zone_regions_finish(name, z, frames, core, only, regions, bgs, info, fg_fn, fg_names, matte)

    def _zone_regions_finish(self, name: str, z, frames, core, only, regions, bgs, info, fg_fn, fg_names, matte):
        """The outline band and the part subtraction on top of the (propagated or empty) drawing."""
        from engine.zones import outline_regions, region_mode
        # `only` (zone previews, 2026-10-16): the propagation needs every frame, the outline band and the part
        # subtraction only the frames shown – the other frames keep the propagated drawing (not used by the caller)
        out_frames = [f for f in frames if f in set(only)] if only is not None else list(frames)
        if z.criterion == "matte" and region_mode(z) == "outline":
            # outline mode (user decision 2026-10-15): band around the whole core edge minus the drawing and the
            # excluded objects – the drawing propagated above is the EXCLUSION
            drawn = regions
            band_core = {f: core[f] for f in out_frames}
            members = self.group_members(name)
            if members and fg_fn is not None:
                # a group (2026-10-16): the band follows the members the zone mattes (CLIP2_A: the head), not the sure
                # ones (the hands) – neither their outer edge nor the head/hand contact is in it (an open outline)
                todo = [m for m in members if m not in fg_names]
                lds = [ld for ld in (self._member_core_loader(m, frames) for m in todo) if ld is not None]
                if lds:
                    band_core = {f: np.logical_or.reduce([ld(f) for ld in lds]) for f in out_frames}
            regions = {**drawn, **outline_regions(z, band_core, drawn, self.zone_exclude(name, z, out_frames),
                                                  band_at=self.zone_band_points(name, z, frames, only=only))}
            if members and fg_fn is not None:
                from engine.export import seam_px
                from engine.zones import _disk_k, outline_in_px
                # the contact: seam gap + how deep the band reaches into the matted member (else a sliver stays there)
                k = _disk_k(max(1, seam_px(self.store.get(name)) + outline_in_px(z.params or {}) + 2))
                for f in out_frames:
                    m = fg_fn(f)
                    if m is not None and m.any():
                        regions[f] = regions[f] & ~cv2.dilate(m.astype(np.uint8), k).astype(bool)
            info = {**info, "region": "outline", "exclude_objects": list((z.params or {}).get("exclude_objects") or []),
                    "drawn": drawn}
        if fg_fn is not None and matte:
            # parts of this object inside the region: not computed by the zone (they are foreground of the whole)
            for f in out_frames:
                m = fg_fn(f)
                if m is not None:
                    regions[f] = regions[f] & ~m
            info = {**info, "fg_objects": fg_names}
        return regions, bgs, info

    def band_points_fn(self, name: str, params: Optional[Dict[str, Any]], owner: str, frames: Sequence[int],
                       only: Optional[Sequence[int]] = None, scales: Tuple[float, float] = (1.0, 1.0)):
        """f -> visible band control points (x, y, out, in, plate px) of a band owner – a whole-outline zone or a stage
        with a band (engine/bandsys.py, 2026-10-22) – None without points. Positions tracked by CoTracker3 from the points'
        keys (engine/bandtrack.py, cached per owner). `only` (previews): the frames shown – positions resolved just there
        and the tracker never waited for (background, nearest keys meanwhile); without it the tracking is awaited."""
        from engine.bandmap import band_points, points_at
        from engine.bandtrack import track_positions
        from engine.zonecache import ZoneCache
        pts = band_points(params)
        if not pts:
            return None
        tracker = self._zone_tracker()
        front_fn, _ = self.front_holdout(name, frames, need=False)
        pos = track_positions(pts, list(frames), tracker, self._band_tracker_frames(name) if tracker else None,
                              ZoneCache(self.store.dir(name), owner).dir, core_at=self.band_core_at(name),
                              occluder_at=front_fn, resolve=only, wait=only is None)
        so, si = scales
        return lambda f: points_at(pts, f, pos.get(f, {}), so, si)

    def zone_band_points(self, name: str, z, frames: Sequence[int], only: Optional[Sequence[int]] = None):
        """band_points_fn of an outline zone (its sliders scale the points' widths)."""
        from engine.objband import SHARED_OWNER, mode, zone_params
        from engine.zones import band_scales
        obj = self.store.get(name)
        root = obj.get("part_of") or name
        if mode(obj, f"zone:{z.zone_id}") != "shared":
            return self.band_points_fn(name, z.params, z.zone_id, frames, only, band_scales(z.params))
        # the shared object band (2026-10-22): tracked once per group on its main object, the zone's sliders scale it
        p = zone_params({"zone_id": z.zone_id, "params": z.params}, obj, self.store.get(root))
        return self.band_points_fn(root, p, SHARED_OWNER, frames, only, band_scales(p))

    def ltx_band_points(self, name: str, s: Dict[str, Any], frames: Sequence[int], only: Optional[Sequence[int]] = None):
        """band_points_fn of stage LTX-2.5 in the band mode: „Pas na zewnątrz” (region) / „Wcięcie w rdzeń” (core erode)."""
        if s.get("combine") != "band" or not s.get("band_points"):
            return None
        ro, ri = s.get("band_ref_out"), s.get("band_ref_in")
        so = float(s["region_px"]) / float(ro) if ro else 1.0
        si = float(s["core_erode_px"]) / float(ri) if ri else 1.0
        return self.band_points_fn(name, {"band_points": s["band_points"]}, s.get("band_owner") or "stage_ltx", frames, only, (so, si))

    def vmm_band_points(self, name: str, s: Dict[str, Any], frames: Sequence[int], only: Optional[Sequence[int]] = None):
        """band_points_fn of stage VideoMaMa: its band = „Pas na zewnątrz” (reach) / „Wcięcie w rdzeń” (inset)."""
        if not s.get("band_points"):
            return None
        ro, ri = s.get("band_ref_out"), s.get("band_ref_in")
        so = float(s["reach_px"]) / float(ro) if ro else 1.0
        si = float(s.get("inset_px") or 0.0) / float(ri) if ri else 1.0
        return self.band_points_fn(name, {"band_points": s["band_points"]}, s.get("band_owner") or "stage_videomama", frames, only,
                                   (so, si))

    def band_core_at(self, name: str):
        """k -> the outline mask band points are tracked from (a group: its members with the seam), None if missing."""
        def at(k: int):
            try:
                return self.preview_core(name, [k])[k]
            except FileNotFoundError:
                return None
        return at

    def zone_fg(self, name: str, z, frames: Sequence[int]):
        """Objects that are one matte with `name` and count as SURE foreground of the zone (out of the ViTMatte region,
        sure FG of the trimap): f -> mask, names. (None, []) when there are none; ValueError when one has no mask yet.
        Default: the parts of `name` (part_of == name) + params.fg_objects. A group may say it per member with
        params.sure_fg (user report 2026-10-16, CLIP2_A: parent „Hands”, part „Head” with the hair – the head must be
        matted, the hands are the sure part): the listed members only, the parent included; their own masks 1+2
        (core_fix / core), never a group version (that is the whole)."""
        members = self.group_members(name) or []
        sure = (z.params or {}).get("sure_fg")
        if sure is None and members:
            sure = self.group_sure_default(name)
        if sure is not None and members:
            names = [n for n in sure if n in members]
        else:
            names = [o["name"] for o in self.store.list() if o.get("part_of") == name and o["name"] != name]
            names += [n for n in ((z.params or {}).get("fg_objects") or []) if n != name and n not in names]
        if not names:
            return None, []
        loaders = []
        for n in names:
            ld = self._member_core_loader(n, frames) if n in members else self._object_mask_loader(n, frames)
            if ld is None:
                raise ValueError(f"Część „{n}” obiektu „{name}” nie ma jeszcze maski (rdzeń albo wynik) – policz ją najpierw")
            loaders.append(ld)

        def fg(f: int):
            m = np.zeros((self.h, self.w), bool)
            for ld in loaders:
                m |= ld(f)
            return m
        return fg, names

    def zone_exclude(self, name: str, z, frames: Sequence[int]):
        """Masks of the objects a zone excludes (params.exclude_objects; e.g. a hand in front, the face): final alpha
        > 0.5 of the active version, else the core. None when the list is empty; ValueError when one has no mask yet."""
        names = [n for n in ((z.params or {}).get("exclude_objects") or []) if n != name]
        if not names:
            return None
        loaders = []
        for n in names:
            ld = self._object_mask_loader(n, frames)
            if ld is None:
                raise ValueError(f"Wykluczany obiekt „{n}” nie ma jeszcze maski (rdzeń albo wynik) – policz go najpierw")
            loaders.append(ld)

        def ex(f: int):
            m = np.zeros((self.h, self.w), bool)
            for ld in loaders:
                m |= ld(f)
            return m
        return ex

    def group_sure_default(self, name: str) -> Optional[List[str]]:
        """Default sure foreground of a group's zone (user report 2026-10-16): the members in FRONT – depth_order lower
        than the backmost member's (CLIP2_A: Hands 0 sure, Head 1 matted); all at the same depth -> None (old rule:
        the parts). Only CLIP2_A had groups when this replaced the parts rule."""
        members = self.group_members(name) or []
        if not members:
            return None
        depth = {m: int(self.store.get(m).get("depth_order", 0) or 0) for m in members}
        back = max(depth.values())
        front = [m for m in members if depth[m] < back]
        return front or None

    def _member_core_loader(self, name: str, frames: Sequence[int]):
        """f -> the own mask 1+2 of a group member (core_fix/ when present in that frame, else the SAM 3.1 core; a frame
        without either – outside the member's life range – = empty); None when the member has no mask in `frames`."""
        od = self.store.dir(name)
        dirs = [od / "core_fix", od / "core"]
        if not any((d / f"c_{f:04d}.png").exists() for d in dirs for f in frames):
            return None

        def ld(f):
            for d in dirs:
                m = _read_mask_cached(d / f"c_{f:04d}.png", self.w, self.h)
                if m is not None:
                    return m
            return np.zeros((self.h, self.w), bool)
        return ld

    def preview_core(self, name: str, frames: Sequence[int]) -> Dict[int, np.ndarray]:
        """Core of the zone previews: the SAM 3.1 core of `name`; a group = the members' masks 1+2 joined with the seam
        (fast stand-in of group_core – no stabilisation; stage 4 „Zastosuj” computes on group_core)."""
        members = self.group_members(name)
        if not members:
            return self.load_core(name, frames)
        from engine.export import seam_px, seamless_union
        lds = [self._member_core_loader(n, frames) for n in members]
        gap = seam_px(self.store.get(name))
        out = {}
        for f in frames:
            cs = [ld(f) for ld in lds if ld is not None]
            if not cs:
                raise FileNotFoundError(f"brak rdzenia klatki {f} – najpierw policz ujęcie")
            k = ("whole", name, f, gap, tuple(zlib.crc32(np.packbits(c)) for c in cs))   # content of the member masks
            w = _MASK_CACHE.get(k)
            out[f] = w if w is not None else _cache_put(k, seamless_union([c.astype(np.float32) for c in cs], cs, gap_px=gap) > 0.5)
        return out

    def _object_mask_loader(self, name: str, frames: Sequence[int]):
        """f -> bool mask of an object: final alpha > 0.5 of the active version, else the core of every frame; None = none."""
        from engine.export import run_alpha_loader
        try:
            o = self.store.get(name)
        except Exception:
            return None
        od = self.store.dir(name)
        run = o["tool"].get("active_run")
        if run and (od / "runs" / run / "run.json").exists():
            ld = run_alpha_loader(od, run, self.w, self.h)
            return lambda f, ld=ld: ld(f) > 0.5
        cd = od / "core"
        if cd.exists() and all((cd / f"c_{f:04d}.png").exists() for f in frames):
            return lambda f, cd=cd: (cv2.imread(str(cd / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127)
        return None

    def zone_region(self, name: str, zone_id: str, frames: Sequence[int], f: int) -> Dict[str, Any]:
        """`GET …/zones/{id}/region/{f}`: the propagated region of the zone in frame f as contours (plate px) + bbox."""
        from engine.zones import load_zones, zone_keys
        z = next((x for x in load_zones(self.store.dir(name)) if x.zone_id == zone_id), None)
        if z is None:
            raise ValueError(f"Brak strefy {zone_id}")
        frames = list(frames)
        from engine.zones import outline_without_drawing
        if outline_without_drawing(z):   # nothing to propagate: only this frame's core (fast, no CoTracker)
            frames = [f]
        core = self.preview_core(name, frames)   # a group: its whole (members + seam), as the zone computes
        # Comfy's models are freed only when CoTracker3 really tracks (zone_regions) – before, every frame step of
        # stage 5 unloaded them (~0.15 s + a 10-30 s reload on the next Comfy run, 2026-10-22)
        # only this frame: the band points resolved here, their tracking never awaited (background; 2026-10-21)
        regions, bgs, info = self.zone_regions(name, z, frames, core, only=[f])
        reg = regions.get(f)
        out: Dict[str, Any] = {"zone_id": zone_id, "frame": f, "keys": info["keys"], "is_key": f in zone_keys(z), "contours": [],
                               "bbox": None, "area_px": int(reg.sum()) if reg is not None else 0}
        if reg is not None and reg.any():
            cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out["contours"] = [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in cnts if len(c) >= 3]
            ys, xs = np.where(reg)
            out["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            out["bg_px"] = int(bgs[f].sum()) if f in bgs else 0
        if info.get("region") == "outline":
            # outline mode: the region is the band; the drawing (propagated) is the exclusion – the UI draws both and a
            # new key starts from the exclusion, never from the band
            out["region_mode"] = "outline"
            if reg is not None and reg.any():
                # the whole band ring – outer AND inner edges (user report 2026-10-21: with band points the band
                # vanished from the preview, only the outer edge came back); the UI fills it even-odd
                cnts, _ = cv2.findContours(reg.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                out["band_contours"] = [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in cnts if len(c) >= 3]
                from engine.zones import feather_maps, zone_feather_px, zone_outer_feather_px
                fo = zone_outer_feather_px(z)
                if f in core:
                    # where the feathers end, both inside the band and never crossing (apply_zones weights, 2026-10-22):
                    # inner = the inset feather from the inner band edge, outer = from the outer band edge (0 = as inset)
                    fi = zone_feather_px(z)
                    d_in, d_out, li, lo = feather_maps(reg, core[f], fi, fo if fo > 0 else fi)
                    c0 = core[f].astype(bool) & ~reg    # the solid side = the core without the band (the band reaches into it)
                    lines = lambda m: [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist()
                                       for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 3]
                    out["inset_feather_contours"] = lines(c0 | (reg & (d_in < li)))
                    # also at 0 = „jak wcięcie”: the result feathers the outer edge over the inset length then, so the
                    # line shows it (it looked like a hard cut, review 2026-10-22)
                    out["feather_contours"] = lines(c0 | (reg & (d_out >= lo)))
            dr = (info.get("drawn") or {}).get(f)
            out["exclude_contours"] = []
            if dr is not None and dr.any():
                cnts, _ = cv2.findContours(dr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out["exclude_contours"] = [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in cnts if len(c) >= 3]
        return out

    def apply_zones(self, name: str, frames: Sequence[int], core: Dict[int, np.ndarray], keys: Dict[int, np.ndarray],
                    probs: Optional[Dict[int, np.ndarray]] = None, base_alpha=None, kinds: Optional[Sequence[str]] = None):
        """`kinds` (engine/stages.zone_kind: brush / matte / threshold; None = all): stage 2 / 3 use only the brush
        corrections, stage 4 brush + matte, stage 5 („Przelicz finalnie”) everything."""
        from engine.stages import zone_kind
        from engine.zonecache import ZoneCache
        from engine.zones import apply_zones, follow_transforms, load_zones
        zones = [z for z in load_zones(self.store.dir(name)) if z.enabled and (kinds is None or zone_kind(z) in kinds)]
        if not zones:
            return None
        # CoTracker3 (zone propagation) and ViTMatte run on the GPU in THIS process: Comfy keeps ~10 GB after a
        # MatAnyone pass and they crash (cuDNN) or thrash – release Comfy's models first (user decision 2026-10-14)
        self.release_comfy_gpu("strefy: śledzenie CoTracker3 / ViTMatte")
        holdout = None
        clean_fn = None
        if any(z.criterion == "matte" for z in zones):
            clean_fn, _ = self.clean_source()
        if any(z.criterion == "matte" and z.holdout for z in zones):
            holdout, _ = self.front_holdout(name, frames)
        # matte zones: regions from their keys (propagation + cache of the transforms); other criteria: as before
        regions_by_zone, bg_by_zone = {}, {}
        for z in zones:
            # brush corrections (stage 2): keys propagated by CoTracker3 like the matte zones (local warp, no points on
            # parts / objects in front); the old threshold zones keep the single drawing moved by follow_transforms
            if z.criterion == "matte" or zone_kind(z) == "brush":
                r, b, _ = self.zone_regions(name, z, list(frames), core)
                regions_by_zone[z.zone_id], bg_by_zone[z.zone_id] = r, b
        tracker = self.zone_tracker
        if tracker is None:
            try:
                from engine.tracking import make_point_tracker
                t = make_point_tracker()
                tracker = t if t.engine == "cotracker3" else None   # spec: CoTracker3, fallback = core bbox shift
            except Exception:
                tracker = None
        s = min(1.0, 1024.0 / max(self.w, self.h))
        frames_small = lambda: ([cv2.resize(self.loader.frame(f), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                                 for f in frames], s)
        def zone_tr(z):
            fr = list(frames)
            if z.static or z.frame in core:
                return follow_transforms(z, fr, core, frames_small if tracker else None, tracker)
            # the zone's drawing frame lies outside these frames (preview window around another frame): follow the zone
            # over the span from its drawing frame to these frames, keep only the asked frames
            ext = list(range(min(fr[0], z.frame), max(fr[-1], z.frame) + 1))
            small = lambda: ([cv2.resize(self.loader.frame(f), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                              for f in ext], s)
            t = follow_transforms(z, ext, self._lazy_core(name, core), small if tracker else None, tracker)
            return {f: t[f] for f in fr}
        tr = {z.zone_id: zone_tr(z) for z in zones if zone_kind(z) != "brush"}   # brush regions: zone_regions above
        user = [f for f, k in self.store.keys(name).items() if k.get("author") == "user"]
        ma_run = None
        # MatAnyone engine, or the ViTMatte hybrid with a MatAnyone base (experiment 2026-10-15)
        if any(z.criterion == "matte" and (z.matting == "matanyone" or (z.matting == "vitmatte" and
               str((z.params or {}).get("hybrid", "") or "") == "matanyone")) for z in zones):
            fin = self._finalizer()
            ma_run = lambda seg, sf, core_, keys_, start, scale, tag: fin._matanyone_segment(seg, sf, core_, keys_, start, scale, tag)
        fg_by_zone = {}
        for z in zones:
            if z.criterion == "matte":
                fg_fn, _ = self.zone_fg(name, z, list(frames))
                if fg_fn is not None:
                    fg_by_zone[z.zone_id] = fg_fn
        return apply_zones(frames, core, self.loader.frame, zones, tr, user_keys=user, holdout=holdout,
                           progress=job_progress, clean=clean_fn, ma_run=ma_run,
                           regions_by_zone=regions_by_zone, bg_by_zone=bg_by_zone,
                           cache_for=lambda zid: ZoneCache(self.store.dir(name), zid), plate_sig=self.plate_sig,
                           probs=probs, base_alpha=base_alpha, fg_by_zone=fg_by_zone or None)

    def apply_zone(self, name: str, frames: Sequence[int], zone_id: Optional[str] = None, progress_cb=None,
                   stage: Optional[str] = None):
        """„Zastosuj” (plan step 4): recompute ONLY the zones (cache: frames with an unchanged hash cost nothing) on the
        base alpha of the active version (runs/v###/alpha_base, written by "Przelicz finalnie" when zones exist) -> a new
        version runs/v### with QA + edge report. The core must be unchanged since that version (else 'Przelicz finalnie').
        `stage="vitmatte"` (stage 4 of the 2026-10-15 layout): the base is the object's `vitmatte_base` – the stage-3
        MatAnyone2 version (tool.stage_runs.matanyone, must be current) or the mask 1+2 written as a version
        (core_base_run) – only the brush + matte zones -> tool.stage_runs["vitmatte@<base>"] (+ "vitmatte")."""
        import shutil
        from engine.refine import ZONE_FEATHER_PX, apply_soft_override, next_run_dir, read_alpha, write_alpha_exr
        from engine.stages import matte_sig, stage_run, vitmatte_base
        from engine.zones import MissingFront  # noqa: F401  (raised by apply_zones -> 409 in the API)
        obj = self.store.get(name)
        run = obj["tool"].get("active_run")
        kinds = None
        vbase = None
        if stage == "vitmatte":
            vbase = vitmatte_base(obj)
            if vbase == "core":
                run = self.core_base_run(name, frames, progress_cb)
            elif vbase == "videomama":
                run, ok, why = stage_run(self.store, name, "videomama")
                if not run or not ok:
                    raise ValueError("najpierw etap VideoMaMa („▶ Policz VideoMaMa”)" + (f" – {why}" if why else "")
                                     + " albo inna baza ViTMatte")
            else:
                run, ok, why = stage_run(self.store, name, "matanyone")
                if not run or not ok:
                    raise ValueError("najpierw etap 3 MatAnyone2 („▶ Przelicz MatAnyone2”)" + (f" – {why}" if why else "")
                                     + " albo baza ViTMatte = maska 1+2")
            kinds = ("brush", "matte")
        if not run:
            raise ValueError("brak wersji – najpierw „Przelicz finalnie”")
        rdir = self.store.dir(name) / "runs" / run
        rj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        segs = [CropSegment.from_dict(d) for d in rj.get("segments_def", [])]
        from engine.workarea import run_frames
        frames = [f for f in frames if f in set(run_frames(self.store.dir(name), run) or frames)]  # work area of the version
        if not (rdir / "alpha_base").exists():
            raise ValueError(f"{run}: brak alphy bazowej (wersja bez stref albo sprzed kroku 4) – „Przelicz finalnie”")
        # the version was computed from the (stabilised) core: compare / reuse the same core and probability maps
        if rj.get("group"):   # a group version (stages 3 / 4 on the whole): the same whole of the masks 1+2
            core, soft = self.group_core(name, frames), {}
        else:
            core, soft = self.core_with_soft(name, frames, self.keys(name), bool(rj.get("core_stabilize")))
        keys = self.keys(name)
        # the version's base alpha (edge mode without zones): ViTMatte zones with trimap_source=base build their trimap
        # from it instead of the SAM3 core (read lazily, only by such zones)
        from engine.export import run_alpha_loader
        base_alpha = run_alpha_loader(self.store.dir(name), run, self.w, self.h, sub="alpha_base")
        zres = self.apply_zones(name, frames, core, keys, probs=soft, base_alpha=base_alpha, kinds=kinds)
        # the core must be the one the version was computed from: SAM3 core + the add / remove zones + shrink / grow
        # (exactly what „Przelicz finalnie” passed to the edge mode and saved in runs/v###/core)
        # (a VideoMaMa base was guided by the mask 1+2 as it is – no MatAnyone shrink / grow)
        cmp = ((zres.core if zres else core) if vbase == "videomama"
               else self._shifted_core(zres.core if zres else core, obj, keys, zres))
        for seg in segs:
            for f in frames[:: max(1, len(frames) // 12)]:
                if not (seg.first <= f <= seg.last):
                    continue
                c0 = cv2.imread(str(rdir / "core" / seg.seg_id / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
                if c0 is None or not np.array_equal(c0 > 127, extract_crop(cmp[f].astype(np.float32), seg) > 0.5):
                    raise ValueError("rdzeń (albo poprawki pędzlem) zmienił się od tej wersji – „Przelicz finalnie” "
                                     "(etap 3: „▶ Przelicz MatAnyone2”) zamiast „Zastosuj”")
        new_dir = next_run_dir(self.store.dir(name))
        per_seg: Dict[str, Dict[int, np.ndarray]] = {}
        for n, seg in enumerate(segs):
            sf = [f for f in frames if seg.first <= f <= seg.last]
            res = {}
            for i, f in enumerate(sf):
                job_progress("Zastosuj: nakładanie stref", i + 1, len(sf))
                base = read_alpha(rdir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr")
                a = base
                if zres and f in zres.soft:
                    sa, sarea = zres.soft[f]
                    area = extract_crop(sarea.astype(np.float32), seg) > 0.5
                    a = apply_soft_override(base, extract_crop(sa.astype(np.float32), seg), area,
                                            zres.feather_px or ZONE_FEATHER_PX,
                                            extract_crop(zres.weights[f], seg) if f in zres.weights else None)
                res[f] = a
                write_alpha_exr(new_dir / "alpha" / seg.seg_id / f"a_{f:04d}.exr", a)
                (new_dir / "alpha_base" / seg.seg_id).mkdir(parents=True, exist_ok=True)
                try:
                    os.link(rdir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", new_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr")
                except OSError:
                    shutil.copyfile(rdir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr", new_dir / "alpha_base" / seg.seg_id / f"a_{f:04d}.exr")
                (new_dir / "core" / seg.seg_id).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(rdir / "core" / seg.seg_id / f"c_{f:04d}.png", new_dir / "core" / seg.seg_id / f"c_{f:04d}.png")
            per_seg[seg.seg_id] = res
            if progress_cb:
                progress_cb("Zastosuj", n + 1, len(segs))
        from engine.refine import assemble_full
        full = assemble_full(per_seg, segs, frames, self.w, self.h)
        rj2 = {**rj, "run": new_dir.name, "applied_from": run, "applied_zone": zone_id,
               "zones": zres.stats if zres else {}, "created_at": datetime.now().isoformat(timespec="seconds")}
        for k in ("stage", "core_fix_sig", "settings_sig"):
            rj2.pop(k, None)
        if stage == "vitmatte":
            rj2.update(stage="vitmatte", base=vbase, matte_sig=matte_sig(self.store.dir(name)))
        (new_dir / "run.json").write_text(json.dumps(rj2, indent=2, ensure_ascii=False), encoding="utf-8")
        qa = self.run_qa(name, frames, full, segs, new_dir, removed=zres.removed if zres else None,
                         soft_areas={f: a for f, (_, a) in zres.soft.items()} if zres and zres.soft else None)
        if zres:
            for f, fl in zres.flags.items():
                if f in qa.flags:
                    qa.flags[f].extend(("yellow", x) for x in fl)
                    if qa.colours.get(f) == "green":
                        qa.colours[f] = "yellow"
        job_progress("Raport krawędzi", 0, len(frames))
        edge = self.edge_report(segs, frames, full, cmp, save_dir=new_dir / "edge")
        mode = rj.get("edge_mode", "matanyone")
        qa.edge = {**qa.edge, "before": edge["median_abs"], "pieces_to_fix": edge["n_fix"], "corrected": False,
                   "fix_available": bool(edge["n_fix"]) and mode != "backlit", "pieces": edge["pieces"]}
        (new_dir / "edge" / "edge.json").write_text(json.dumps({"schema": "matte_desk/edge_fix@1.0", "mode": mode,
                                                                 "fix": [p["id"] for p in edge["pieces"]], "attempts": 0}, indent=1),
                                                    encoding="utf-8")
        (new_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        extra = {}
        if stage == "vitmatte":
            extra["stage_runs"] = {**(self.store.get(name)["tool"].get("stage_runs") or {}), f"vitmatte@{vbase}": new_dir.name,
                                   "vitmatte": new_dir.name}
        self.store.update(name, tool={"active_run": new_dir.name, "final_valid": True,
                                      "status": "QA OK" if qa.gate_ok else "QA: flagi", "stale_from": None, **extra})
        stats = zres.stats if zres else {}
        return full, new_dir, {"computed": sum(int(v.get("computed", 0)) for v in stats.values()),
                               "cached": sum(int(v.get("cached", 0)) for v in stats.values()), "zones": stats}

    def run_qa(self, name: str, frames: Sequence[int], full: Dict[int, np.ndarray], segs: Sequence[CropSegment],
               run_dir: Path, removed: Optional[Dict[int, np.ndarray]] = None, attempts: Optional[int] = None,
               soft_areas: Optional[Dict[int, np.ndarray]] = None):
        """`soft_areas`: per-frame area of soft `matte` zones – strands there are not QA "islands" (plan section 12)."""
        from engine.qa import run_qa
        obj = self.store.get(name)
        if attempts is None:
            attempts = int(obj["tool"].get("auto_attempts", 0))
        cj = self.store.dir(name) / "core" / "core.json"
        fb = {int(k): v for k, v in json.loads(cj.read_text()).get("fwd_bwd_iou", {}).items()} if cj.exists() else {}
        # uint8 gray: _GrayCache keeps it as is and returns float32 (same values, no per-frame integrality check)
        gray = lambda f: cv2.cvtColor((np.clip(self.loader.read(f), 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # holes / islands inside the area of a soft `matte` zone are the zone's topology (gaps between strands), not
        # defects (plan section 12); holes from Remove zones were already allowed
        def _allowed(f):
            m = removed[f] if removed else None
            if soft_areas and f in soft_areas:
                # + 8 px: a pocket of the base's soft edge (alpha ~0.4) closed by the zone's strands lies just outside
                # the area (SH020 2026-10-13: 17-44 px slivers along the zone border) – zone topology, not a defect
                sa = cv2.dilate(soft_areas[f].astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
                m = sa if m is None else (m | sa)
            return m
        allowed = _allowed if (removed or soft_areas) else None
        allowed_isl = (lambda f: soft_areas.get(f)) if soft_areas else None
        cgray = None
        if self.clean is not None:
            cgray = lambda f: cv2.cvtColor((np.clip(self.clean.read(f), 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Display gray of the WHOLE frame for camera shift (ECC), flow and clear-plate agreement – user decision 2026-10-13:
        # the raw-EXR log-luma band (Z27 2b, `qa_gray_rows`) was 2x faster but its ECC compensated the camera worse on
        # SH010 (flicker median +26 % suknia / +7 % figure, clear-plate agreement 0.470 -> 0.429). Quality first: only
        # bit-identical speedups (parallel QA, fast EXR reader) are used here.
        qa = run_qa(frames, lambda f: full[f], gray, fb, seg_boundaries=[s.first for s in segs[1:]],
                    allowed_holes=allowed, attempts=attempts + 1, allowed_islands=allowed_isl,
                    flicker_map_path=run_dir / "flicker_map.png", flow_check=True, load_clean_gray=cgray,
                    gray_source="display_full", progress_cb=job_progress)
        (run_dir / "qa.json").write_text(json.dumps(qa.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        return qa


# ----------------------------------------------------------------------------- edge pieces (Z27 pkt 1)
class _LazyCore(dict):
    """Core masks: the given frames + any other frame with a core on disk, loaded on first access (`in` / [] / get)."""

    def __init__(self, base: Dict[int, np.ndarray], load: Callable[[int], np.ndarray]):
        super().__init__(base)
        self._load = load
        self._absent: set = set()

    def __contains__(self, f) -> bool:
        if dict.__contains__(self, f):
            return True
        try:
            self[f]
            return True
        except KeyError:
            return False

    def __missing__(self, f):
        if f in self._absent:
            raise KeyError(f)
        try:
            m = self._load(int(f))
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            self._absent.add(f)
            raise KeyError(f)
        self[f] = m
        return m

    def get(self, f, default=None):
        try:
            return self[f]
        except KeyError:
            return default


def core_stabilize_on(obj: Dict[str, Any]) -> bool:
    """Object option core_stabilize (user decision 2026-10-14: ON by default, also for objects created before the
    option existed; switched off per object in the settings)."""
    v = obj.get("core_stabilize")
    if v is None:
        try:
            defs = json.loads((Path(__file__).resolve().parents[1] / "defaults.json").read_text(encoding="utf-8"))
            return bool(defs.get("core_stabilize", True))
        except Exception:
            return True
    return bool(v)


def crop_plate_mask(seg: CropSegment) -> np.ndarray:
    """Crop-space bool mask of the pixels that lie inside the plate (False = mirror padding of the square crop)."""
    side = int(seg.side)
    l, t, r, b = (int(v) for v in seg.pad)
    m = np.zeros((side, side), bool)
    m[t:side - b, l:side - r] = True
    return m


def piece_id(k) -> str:
    """Contour piece key (contour index, ~64 px chunk) -> stable string id used by the API / UI."""
    return f"{int(k[0])}_{int(k[1])}"


def save_edge_measure(path: Path, seg: CropSegment, m) -> None:
    keys = list(m.piece_median)
    np.savez(path, seg_id=np.array(seg.seg_id), points=m.points, normals=m.normals, offsets=m.offsets,
             widths=m.widths, pieces=np.array(m.pieces, np.int32).reshape(-1, 2),
             pm_keys=np.array(keys, np.int32).reshape(-1, 2),
             pm_vals=np.array([m.piece_median[k] for k in keys], np.float64))


def load_edge_measure(path: Path):
    from engine.edge import EdgeMeasure
    d = np.load(path, allow_pickle=False)
    pieces = [tuple(int(v) for v in r) for r in d["pieces"]]
    m = EdgeMeasure(d["offsets"], d["points"], d["normals"], pieces, d["widths"])
    m.piece_median = {(int(k[0]), int(k[1])): float(v) for k, v in zip(d["pm_keys"], d["pm_vals"])}
    return str(d["seg_id"]), m


def edge_pieces(meas_by_f: Dict[int, Any], fix: Dict[Any, bool]) -> List[Dict[str, Any]]:
    """Pieces to correct: id, frames, median offset (+ = halo -> shrink, - = cut -> grow), per-frame medians."""
    out = []
    for k in sorted(k for k, v in fix.items() if v):
        series = {f: m.piece_median[k] for f, (_s, m) in meas_by_f.items() if k in m.piece_median}
        if not series:
            continue
        vals = list(series.values())
        med = float(np.median(vals))
        out.append({"id": piece_id(k), "first": min(series), "last": max(series), "n_frames": len(series),
                    "median_px": round(med, 3), "kind": "halo" if med > 0 else "cut",
                    "per_frame": {str(f): round(v, 3) for f, v in sorted(series.items())}})
    return out


def qa_gray_rows(pattern: str, frames: Sequence[int]) -> Optional[Callable[[int, int, int], np.ndarray]]:
    """Row-band log-luminance reader for QA (Z27 2b) when the plate is plain RGB EXR, else None.
    NOT used by `ObjectPipeline.run_qa` (user decision 2026-10-13, see there) – kept for measurements / tools."""
    from engine.exr_io import read_log_luma_rows
    if not frames:
        return None
    p0 = frame_path(pattern, list(frames)[0])
    if p0.suffix.lower() != ".exr":
        return None
    try:
        if read_log_luma_rows(p0, 0, 1) is None:
            return None
    except Exception:
        return None

    def rows(f: int, y0: int, y1: int) -> np.ndarray:
        g = read_log_luma_rows(frame_path(pattern, f), y0, y1)
        if g is None:
            raise FileNotFoundError(frame_path(pattern, f))
        return g
    return rows
