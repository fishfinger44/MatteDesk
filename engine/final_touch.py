# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/final_touch.py - stage Final Touch (PLAN_FINAL_TOUCH.md, step 1): hand corrections of the alpha, frame by
frame, before the export – a bitmap brush over a chosen computed version (the base), no tracking yet.

Storage: `<obj dir>/final_touch/final_touch.json` (never the shot manifest):
  {"base": {"source": "matanyone", "run": null}, "finish": {...}, "frames": {"12": [stroke, ...]}, "updated": <time>}
  finish (user decision 2026-10-22, moved here from the ViTMatte zone sliders) = global finishing applied AFTER the
  strokes (user decision, later that day: the blur / erode also shape the edges of the painted corrections): black /
  white levels, gamma, erode_px (negative = dilate; grey morphology, soft edges stay soft), blur_px (Gaussian). All at
  their defaults = base + strokes bit for bit.
  base.source = an export source (core | matanyone | videomama | vitmatte | ltx | active | version; run = the version for
  "version"); None = the default (the result of the last computed stage, like the export source).
  stroke = {"mode": "add" | "remove" | "erase", "radius_px": r, "hardness": 0..1, "p": [[x, y] | [x, y, r], ...]} in plate
  pixels – strokes are kept as PATHS (plan §3: re-rasterised at any resolution, later tracked), the raster is computed.
Correction layer C in [-1, 1] per frame (strokes in order, opacity m = the soft brush mask):
  add: C += m (1 - C) (towards +1) · remove: C -= m (1 + C) (towards -1) · erase: C *= 1 - m (back to the base)
Result: a = finish(base + C (1 - base) where C > 0, base (1 + C) where C < 0) – untouched pixels keep the base exactly
(before the finish).
Export / preview: the version token `ft:<base run>` (FT_PREFIX) – engine/export.run_alpha_loader applies the layer.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

FT_DIR = "final_touch"
FT_PREFIX = "ft:"          # export version token: "ft:v012" = version v012 + the corrections, "ft:core" = mask 1+2 + them
MODES = ("add", "remove", "erase")
BASE_SOURCES = ("core", "matanyone", "videomama", "vitmatte", "ltx", "active", "version")
# default base = the result of the last computed stage (same order as the UI's default export source)
DEFAULT_BASE_ORDER = ("ltx", "vitmatte", "videomama", "matanyone", "active", "core")
HARDNESS_DEFAULT = 0.6
FINISH_DEFAULT = {"black": 0.0, "white": 1.0, "gamma": 1.0, "erode_px": 0.0, "blur_px": 0.0,
                  # motion blur of the alpha (user decision 2026-10-22: the last finishing step, after the erode, before
                  # the blur): strength x the physical blur of the shutter; shutter None = the manifest's; measured = the
                  # hybrid (strength per edge segment from the plate, engine/mb_map.py), else one strength everywhere
                  "mb_strength": 0.0, "mb_shutter_deg": None, "mb_measured": True}
FINISH_RANGE = {"black": (0.0, 0.9), "white": (0.1, 1.0), "gamma": (0.2, 5.0), "erode_px": (-50.0, 50.0), "blur_px": (0.0, 50.0),
                "mb_strength": (0.0, 10.0), "mb_shutter_deg": (0.0, 720.0)}
FINISH_FLAGS = ("mb_measured",)
# plate context of the motion blur (gray frames, the shot's shutter), given by the app (set_context_provider) – the
# alpha loaders of the export / preview have no plate of their own; None = the motion blur is skipped
_CONTEXT_PROVIDER: Optional[Callable[[], Optional[Dict[str, Any]]]] = None


def set_context_provider(fn: Optional[Callable[[], Optional[Dict[str, Any]]]]) -> None:
    """`fn()` -> {"gray": f -> uint8 gray | None, "shutter_deg": float, "gpu": worker | callable -> worker | None}
    or None. The worker (engine/align_gpu) may be given lazily; a GpuWorkerError falls back to the CPU."""
    global _CONTEXT_PROVIDER
    _CONTEXT_PROVIDER = fn


def context() -> Optional[Dict[str, Any]]:
    try:
        return _CONTEXT_PROVIDER() if _CONTEXT_PROVIDER else None
    except Exception:
        return None


def ft_path(obj_dir: Path) -> Path:
    return Path(obj_dir) / FT_DIR / "final_touch.json"


def load(obj_dir: Path) -> Dict[str, Any]:
    try:
        d = json.loads(ft_path(obj_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d.setdefault("base", {"source": None, "run": None})
    d.setdefault("frames", {})
    d["finish"] = {**FINISH_DEFAULT, **{k: v for k, v in (d.get("finish") or {}).items() if k in FINISH_DEFAULT}}
    d["finish"]["mb_measured"] = bool(d["finish"].get("mb_measured", True))
    return d


def save(obj_dir: Path, d: Dict[str, Any]) -> Dict[str, Any]:
    p = ft_path(obj_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    d["updated"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return d


def clean_stroke(s: Dict[str, Any]) -> Dict[str, Any]:
    """Validated copy of a stroke (ValueError on a bad one)."""
    mode = s.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode: {' | '.join(MODES)}")
    pts = s.get("p") or []
    if not pts or any(len(q) < 2 for q in pts):
        raise ValueError("p: [[x, y] | [x, y, r], ...] – co najmniej jeden punkt")
    r = float(s.get("radius_px") or 8)
    h = float(s.get("hardness", HARDNESS_DEFAULT))
    return {"mode": mode, "radius_px": max(0.5, min(2000.0, r)), "hardness": max(0.0, min(1.0, h)),
            "p": [[round(float(c), 2) for c in q[:3]] for q in pts]}


def set_frame(obj_dir: Path, f: int, strokes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace the strokes of frame f (an empty list removes the frame's corrections)."""
    d = load(obj_dir)
    cl = [clean_stroke(s) for s in strokes]
    if cl:
        d["frames"][str(int(f))] = cl
    else:
        d["frames"].pop(str(int(f)), None)
    return save(obj_dir, d)


def set_base(obj_dir: Path, source: Optional[str], run: Optional[str] = None) -> Dict[str, Any]:
    if source is not None and source not in BASE_SOURCES:
        raise ValueError(f"source: {' | '.join(BASE_SOURCES)}")
    d = load(obj_dir)
    d["base"] = {"source": source, "run": run if source == "version" else None}
    return save(obj_dir, d)


def clean_finish(fin: Dict[str, Any]) -> Dict[str, float]:
    """Validated finish values (unknown keys dropped, values clamped to FINISH_RANGE; flags -> bool; mb_shutter_deg
    None = the manifest's shutter)."""
    out: Dict[str, Any] = {}
    for k, (lo, hi) in FINISH_RANGE.items():
        if k in fin:
            if fin[k] is None:
                if k == "mb_shutter_deg":
                    out[k] = None
                continue
            try:
                out[k] = float(min(hi, max(lo, float(fin[k]))))
            except (TypeError, ValueError):
                raise ValueError(f"finish.{k}: liczba")
    for k in FINISH_FLAGS:
        if k in fin and fin[k] is not None:
            out[k] = bool(fin[k])
    return out


def set_finish(obj_dir: Path, fin: Dict[str, Any]) -> Dict[str, Any]:
    """Merge finish values into the object's Final Touch."""
    d = load(obj_dir)
    d["finish"] = {**d["finish"], **clean_finish(fin)}
    return save(obj_dir, d)


def finish_active(fin: Dict[str, Any]) -> bool:
    f = {**FINISH_DEFAULT, **(fin or {})}
    return any(abs(float(f[k]) - FINISH_DEFAULT[k]) > 1e-9 for k in FINISH_RANGE if k != "mb_shutter_deg")


def finish_alpha(a: np.ndarray, fin: Dict[str, Any], f: Optional[int] = None,
                 ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Global finishing of an alpha (float32 0..1), in order: levels (<= black -> 0, >= white -> 1, linear between),
    gamma, erode (> 0, grey min filter) / dilate (< 0, max filter) by |erode_px|, motion blur (mb_strength > 0; needs
    the frame `f` and the plate context `ctx` – see set_context_provider – else skipped), Gaussian blur of blur_px.
    Defaults = `a` unchanged (same object)."""
    frame = int(f) if f is not None else None
    f = {**FINISH_DEFAULT, **(fin or {})}
    if not finish_active(f):
        return a
    out = np.clip(a, 0.0, 1.0).astype(np.float32)
    b, w = float(f["black"]), float(f["white"])
    if b > 0.0 or w < 1.0:
        out = np.clip((out - b) / (max(w, b + 1e-3) - b), 0.0, 1.0).astype(np.float32)
    g = float(f["gamma"])
    if abs(g - 1.0) > 1e-9:
        out = np.power(out, np.float32(1.0 / g)).astype(np.float32)
    e = float(f["erode_px"])
    if abs(e) >= 0.5:
        r = int(round(abs(e)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = cv2.erode(out, k) if e > 0 else cv2.dilate(out, k)
    mbs = float(f["mb_strength"] or 0.0)
    if mbs > 0.0 and frame is not None:
        c = ctx if ctx is not None else context()
        if c and c.get("gray"):
            from engine.mb_map import MbCache
            from engine.motionblur import motion_blur_measured
            gray = c["gray"]
            sh = float(f["mb_shutter_deg"]) if f.get("mb_shutter_deg") is not None else float(c.get("shutter_deg") or 180.0)
            cache = c.get("cache")
            if cache is None and c.get("cache_dir"):
                cache = c["cache"] = MbCache(c["cache_dir"])
            gpu = c.get("gpu")
            try:
                gpu = gpu() if callable(gpu) else gpu
            except Exception:
                gpu = None
            kw = dict(measured=bool(f.get("mb_measured", True)), cache=cache, frame=frame)
            try:
                out, _st = motion_blur_measured(out, gray(frame), gray(frame - 1), gray(frame + 1), sh, mbs, gpu=gpu, **kw)
            except Exception as e:   # the GPU worker died / refused: the same blur on the CPU (align_gpu.GpuWorkerError)
                if gpu is None:
                    raise
                if c.get("on_gpu_error"):
                    c["on_gpu_error"](e)
                out, _st = motion_blur_measured(out, gray(frame), gray(frame - 1), gray(frame + 1), sh, mbs, gpu=None, **kw)
    bl = float(f["blur_px"])
    if bl > 0.0:
        kk = int(round(bl * 3)) * 2 + 1
        out = cv2.GaussianBlur(out, (kk, kk), bl)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def frames_of(d: Dict[str, Any]) -> List[int]:
    return sorted(int(f) for f, s in (d.get("frames") or {}).items() if s)


def corrections_sig(d: Dict[str, Any]) -> str:
    """Hash of the corrections (all frames' strokes + the finish) – the export records it with the base (plan §3);
    "" = nothing at all (the base alone)."""
    fr = {str(f): d["frames"][str(f)] for f in frames_of(d)}
    fin = d.get("finish") or {}
    if not fr and not finish_active(fin):
        return ""
    return hashlib.sha1(json.dumps({"frames": fr, "finish": {k: (round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                                              for k, v in fin.items()}}, sort_keys=True).encode()).hexdigest()[:12]


def stroke_mask(s: Dict[str, Any], w: int, h: int) -> np.ndarray:
    """Soft opacity of one stroke (float32 h x w, 0..1): the hard stroke (engine/keys.draw_variable_stroke, the same
    capsules the viewer draws) faded inwards from its edge over (1 - hardness) of the radius."""
    from engine.keys import draw_variable_stroke
    r_def = float(s.get("radius_px") or 8)
    pts = s["p"]
    rmax = max([r_def] + [float(q[2]) for q in pts if len(q) >= 3])
    xs, ys = [float(q[0]) for q in pts], [float(q[1]) for q in pts]
    x0, y0 = max(0, int(np.floor(min(xs) - rmax - 2))), max(0, int(np.floor(min(ys) - rmax - 2)))
    x1, y1 = min(w, int(np.ceil(max(xs) + rmax + 3))), min(h, int(np.ceil(max(ys) + rmax + 3)))
    out = np.zeros((h, w), np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
    draw_variable_stroke(roi, [[q[0] - x0, q[1] - y0] + ([q[2]] if len(q) >= 3 else []) for q in pts], r_def, 255)
    feather = rmax * (1.0 - float(s.get("hardness", HARDNESS_DEFAULT)))
    if feather < 0.5:
        m = (roi > 0).astype(np.float32)
    else:
        dist = cv2.distanceTransform((roi > 0).astype(np.uint8), cv2.DIST_L2, 5)
        t = np.clip(dist / feather, 0.0, 1.0)
        m = (t * t * (3.0 - 2.0 * t)).astype(np.float32)      # smoothstep: soft, no visible ring
    out[y0:y1, x0:x1] = m
    return out


def layer(strokes: Sequence[Dict[str, Any]], w: int, h: int) -> np.ndarray:
    """Correction layer C (float32, -1..1) of one frame's strokes, in order."""
    c = np.zeros((h, w), np.float32)
    for s in strokes:
        m = stroke_mask(s, w, h)
        if s["mode"] == "add":
            c += m * (1.0 - c)
        elif s["mode"] == "remove":
            c -= m * (1.0 + c)
        else:
            c *= 1.0 - m
    return c


def compose(base: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Base alpha + correction layer (both h x w float)."""
    b = base.astype(np.float32)
    return np.clip(np.where(c >= 0, b + c * (1.0 - b), b * (1.0 + c)), 0.0, 1.0).astype(np.float32)


def apply(obj_dir: Path, f: int, base: np.ndarray, d: Optional[Dict[str, Any]] = None,
          ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Base alpha of frame f with the object's corrections of that frame, then the finish (the base itself when there
    are none)."""
    d = d if d is not None else load(obj_dir)
    strokes = (d.get("frames") or {}).get(str(int(f))) or []
    if strokes:
        h, w = base.shape[:2]
        base = compose(base, layer(strokes, w, h))
    if ctx is not None and not ctx.get("cache_dir"):
        ctx = {**ctx, "cache_dir": Path(obj_dir) / "mb_cache"}   # per-object cache of the motion blur's velocity / map
    return finish_alpha(base, d.get("finish") or {}, f, ctx)


def is_ft(run: Optional[str]) -> bool:
    return bool(run) and str(run).startswith(FT_PREFIX)


def base_run(run: str) -> str:
    """Version the token stands on ("ft:v012" -> "v012", anything else unchanged)."""
    return run[len(FT_PREFIX):] if is_ft(run) else run


def ft_loader(obj_dir: Path, base_loader: Callable[[int], np.ndarray]) -> Callable[[int], np.ndarray]:
    """Loader of base + corrections (the corrections file read once per export / preview)."""
    d = load(obj_dir)
    ctx = context() if float((d.get("finish") or {}).get("mb_strength") or 0.0) > 0.0 else None

    def load_f(f: int) -> np.ndarray:
        return apply(obj_dir, f, base_loader(f), d, ctx)
    return load_f
