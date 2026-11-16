# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shutter angle measured on the plate (user decision 2026-11-08).

The manifest's `shutter_deg` used to be written by the new-shot wizard as a default (180) – not a measurement; on sh060
the plate's real blur is ~45-90 deg. The motion blur of the alpha (engine/motionblur.py, Final Touch) and the edge band
prediction scale with the shutter, so a wrong default makes every predicted blur too long.

Measurement (plate only, no fill, no object needed):
1. Velocity per frame: dense optical flow (DIS on gray, long side <= FLOW_SIDE), centred like engine/motionblur.py:
   (flow t->t+1 - flow t->t-1) / 2; pixels of the tool's objects can be excluded (`exclude` masks), the rest is robust.
2. Pairs (a, b) of nearby frames where the camera moves clearly faster in b (median speed ratio >= RATIO_MIN, b's speed
   >= SPEED_MIN px): frame a warped onto b (flow b->a) carries the blur of a's speed; blurring it further along b's
   motion by  dl = shutter / 360 x sqrt(|v_b|^2 - |v_a|^2)  (box blurs add in variance) must give b's blur.
3. The angle whose re-blurred a has b's DIRECTIONAL sharpness ratio – gradient energy along the motion / across it, a
   ratio that does not depend on the content's contrast or exposure – summed |log ratio| over all pairs. One value per
   shot. `confident` = enough pairs and a clear minimum; otherwise the caller keeps the manifest / default.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

FLOW_SIDE = 960                 # px, long side of the analysis (flow + blur)
ANGLES = tuple(float(a) for a in range(0, 361, 15))
RATIO_MIN = 1.4                 # b must move this much faster than a (median speed)
SPEED_MIN = 3.0                 # px / frame at FLOW_SIDE: b's blur must be visible
MAX_GAP = 3                     # frames between a and b
MAX_PAIRS = 24
MIN_PAIRS = 3
DEFAULT_DEG = 180.0             # the old assumption, kept as the last fallback


def _small(g: np.ndarray) -> Tuple[np.ndarray, float]:
    s = min(1.0, FLOW_SIDE / float(max(g.shape[:2])))
    if s < 1.0:
        g = cv2.resize(g, (int(round(g.shape[1] * s)), int(round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return g, s


def _gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3].astype(np.float32) if img.dtype != np.uint8 else img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        x = x / max(1e-6, float(np.percentile(x, 99.5)))   # float plates: normalise
        img = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return img


def directional_ratio(g: np.ndarray, ux: np.ndarray, uy: np.ndarray, sel: np.ndarray) -> float:
    """Gradient energy along (ux, uy) / across it, band-passed (contrast-free sharpness anisotropy)."""
    x = g.astype(np.float32)
    x = x - cv2.GaussianBlur(x, (0, 0), 6.0)
    gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    al, ac = gx * ux + gy * uy, -gx * uy + gy * ux
    return float((al[sel] ** 2).mean() / max(1e-6, float((ac[sel] ** 2).mean())))


def path_blur(g: np.ndarray, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """Box average along the per-pixel vector (lx, ly) (centred, bilinear) – the plate's exposure model."""
    ln = float(np.hypot(lx, ly).max()) if lx.size else 0.0
    if ln < 0.5:
        return g.astype(np.float32)
    n = int(min(48, max(2, np.ceil(ln) + 1)))
    h, w = g.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    src = g.astype(np.float32)
    acc = np.zeros((h, w), np.float32)
    for i in range(n):
        t = np.float32((i + 0.5) / n - 0.5)
        acc += cv2.remap(src, gx + t * lx, gy + t * ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def measure(read_gray: Callable[[int], Optional[np.ndarray]], frames: Iterable[int],
            exclude: Optional[Callable[[int], Optional[np.ndarray]]] = None,
            cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """{deg, confident, pairs, errors: {angle: err}, speeds: {frame: px}, why} – `read_gray(f)` = plate frame f (any
    dtype / channels), `exclude(f)` = bool mask of pixels to ignore (the tool's objects), full size or any size."""
    fr = [int(f) for f in frames]
    g: Dict[int, np.ndarray] = {}
    scale = 1.0
    for f in fr:
        im = read_gray(f)
        if im is None:
            continue
        g[f], scale = _small(_gray_u8(im))
    fr = [f for f in fr if f in g]
    out: Dict[str, Any] = {"deg": None, "confident": False, "pairs": 0, "errors": {}, "speeds": {}, "why": ""}
    if len(fr) < 3:
        out["why"] = "za mało klatek"
        return out
    h, w = g[fr[0]].shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def flow(a: int, b: int) -> np.ndarray:          # px of a -> where it is in b
        if (a, b) not in flow_cache:
            flow_cache[(a, b)] = dis.calc(g[a], g[b], None)
        return flow_cache[(a, b)]

    def ignore(f: int) -> np.ndarray:
        m = exclude(f) if exclude is not None else None
        if m is None:
            return np.zeros((h, w), bool)
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

    vel: Dict[int, np.ndarray] = {}
    speed: Dict[int, float] = {}
    for i, f in enumerate(fr):
        if cancel_check:
            cancel_check()
        nxt = fr[i + 1] if i + 1 < len(fr) and fr[i + 1] == f + 1 else None
        prv = fr[i - 1] if i > 0 and fr[i - 1] == f - 1 else None
        if nxt is not None and prv is not None:
            v = (flow(f, nxt) - flow(f, prv)) / 2.0
        elif nxt is not None:
            v = flow(f, nxt)
        elif prv is not None:
            v = -flow(f, prv)
        else:
            continue
        vel[f] = v
        ok = ~ignore(f)
        speed[f] = float(np.median(np.hypot(v[..., 0], v[..., 1])[ok])) if ok.any() else 0.0
    out["speeds"] = {int(f): round(s / scale, 2) for f, s in speed.items()}

    cands: List[Tuple[float, int, int]] = []
    for a in vel:
        for b in vel:
            if 0 < abs(b - a) <= MAX_GAP and speed[b] >= SPEED_MIN and speed[b] >= RATIO_MIN * max(speed[a], 0.25):
                cands.append((speed[b] / max(speed[a], 0.25), a, b))
    cands.sort(reverse=True)
    cands = cands[:MAX_PAIRS]
    if len(cands) < MIN_PAIRS:
        out["why"] = (f"za mało par klatek o różnej prędkości kamery ({len(cands)}, potrzeba {MIN_PAIRS}) – "
                      "ujęcie bez wyraźnego ruchu albo ze stałą prędkością")
        return out
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    err = {ang: 0.0 for ang in ANGLES}
    used = 0
    for _, a, b in cands:
        if cancel_check:
            cancel_check()
        fba = flow(b, a)                                         # px of b -> where it is in a
        mx, my = gx + fba[..., 0], gy + fba[..., 1]
        wa = cv2.remap(g[a].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        va = cv2.remap(np.hypot(vel[a][..., 0], vel[a][..., 1]), mx, my, cv2.INTER_LINEAR)
        vb = vel[b]
        sb = np.hypot(vb[..., 0], vb[..., 1])
        ign_a = cv2.remap(ignore(a).astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=1) > 0
        sel = (wa >= 0) & ~ign_a & ~ignore(b) & (sb > 1.5)
        sel = cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if sel.sum() < 2000:
            continue
        ux, uy = vb[..., 0] / np.maximum(sb, 1e-6), vb[..., 1] / np.maximum(sb, 1e-6)
        target = np.log(directional_ratio(g[b], ux, uy, sel))
        extra = np.sqrt(np.maximum(sb ** 2 - va ** 2, 0.0)).astype(np.float32)
        wa = np.where(wa >= 0, wa, g[b].astype(np.float32))
        for ang in ANGLES:
            k = np.float32(ang / 360.0)
            bl = path_blur(wa, ux * extra * k, uy * extra * k)
            err[ang] += abs(np.log(directional_ratio(bl, ux, uy, sel)) - target)
        used += 1
    out["pairs"] = used
    if used < MIN_PAIRS:
        out["why"] = f"za mało porównywalnych par ({used})"
        return out
    best = min(err, key=err.get)
    vals = np.array([err[a] for a in ANGLES])
    out["deg"] = float(best)
    out["errors"] = {f"{a:g}": round(err[a] / used, 4) for a in ANGLES}
    contrast = (float(np.median(vals)) - float(vals.min())) / max(1e-6, float(np.median(vals)))
    out["confident"] = bool(contrast > 0.1)
    if not out["confident"]:
        out["why"] = "płaskie minimum – pomiar niepewny"
    return out


def resolve(manifest: Dict[str, Any], tool: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    """(shutter deg, source): the plate measurement stored in the tool's project.json (`shutter`, made on the user's
    click – it wins, because older manifests carry the wizard's default 180 that was never measured), else the
    manifest's value, else DEFAULT_DEG. source = measured | manifest | default."""
    rec = (tool or {}).get("shutter") or {}
    if rec.get("deg") is not None and rec.get("confident", True):
        return float(rec["deg"]), "measured"
    v = manifest.get("shutter_deg")
    if v is not None:
        try:
            return float(v), "manifest"
        except (TypeError, ValueError):
            pass
    return DEFAULT_DEG, "default"
