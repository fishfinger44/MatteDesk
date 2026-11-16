# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align_residual.py - how well an auxiliary plate (clear plate / backlit) sits on the beauty AFTER the
alignment (plan align_v2 step 0, 2026-11-03): the residual on control points and a residual map.

The alignment itself (engine/align.py) reports hold-out errors of its own model; those say how consistent the
measurement was, not where the aligned plate still misses. This module measures the miss directly: the aux frame is
warped by the frame's composed alignment (`engine.align.compose_remap` – the same map `_align` and the preview use),
features are extracted at FULL resolution (`AlignmentFeatureExtractor`, mode backlit / cleanplate), and on a dense grid
of control points over the static area the remaining shift aux -> beauty is measured by patch phase correlation + the
Gauss-Newton refinement of the local model (`LocalPatchAligner._refine_translation`). Statistics (median / p95 / max of
the shift length), a coarse map (one cell per grid point) and a PNG heat map come out of it.

The static area = `compute_static_mask` (margins, clear-plate difference) x an exclusion mask (objects' cores dilated,
a user box – what the alignment itself may exclude from 2026-11-03 on) x the coverage of the warped aux. A control
point counts only when its patch has texture (correlation response > RESP_MIN) and the shift is inside the search
range of the patch (|d| <= patch / 4): a point on a moving object or on a black border is dropped, not averaged in.

Used by experiments/exp_align_baseline.py (before / after tables, SH010) and by the stage-0 residual map endpoint
(api/plate_align.py, static/stage0.js – step 3). No GPU: a frame at 4096 x 2731 takes a few seconds on the CPU.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from engine import align_grid
from engine.align import (AlignmentFeatureExtractor, AlignmentFrameResult, LocalPatchAligner, compose_remap,
                          compute_static_mask, phase_correlate)

RESIDUAL_STEP_PX = 48      # control point spacing (plate px) - the local model's grid spacing
RESIDUAL_PATCH_PX = 64     # correlation patch (plate px): half the local model's anchor patch, measured at full res
RESP_MIN = 0.1             # correlation response at or below: no texture under the patch (as engine.align)


def residual_points(feat_b: np.ndarray, feat_a: np.ndarray, mask: Optional[np.ndarray] = None,
                    step: int = RESIDUAL_STEP_PX, patch: int = RESIDUAL_PATCH_PX,
                    max_px: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remaining shift aux -> beauty on a grid of control points of the two FULL-resolution feature maps (the aux one
    already aligned). Returns (pts (N, 2) int plate px, res (N, 2) float px, resp (N,)): only the points whose patch
    lies inside the plate, inside `mask` (> 0.5 at the point; None = everywhere) and has texture (resp > RESP_MIN)
    with a shift inside the search range (|dx|, |dy| <= max_px, default patch / 4).
    Sign as AlignmentFrameResult.dx / dy: the shift that still has to be applied to the aux to land on the beauty
    (an aux whose content sits 1.3 px right of the beauty's measures dx = -1.3, like the global model)."""
    h, w = feat_b.shape[:2]
    half = patch // 2
    lim = float(max_px) if max_px is not None else patch / 4.0
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    ys = np.arange(half + (h - 2 * half) % step // 2, h - half + 1, step)
    xs = np.arange(half + (w - 2 * half) % step // 2, w - half + 1, step)
    pts, res, resp = [], [], []
    for y in ys:
        for x in xs:
            y, x = int(y), int(x)
            if not (half <= x < w - half + 1 and half <= y < h - half + 1):
                continue
            if mask is not None and not (mask[min(y, h - 1), min(x, w - 1)] > 0.5):
                continue
            p_b = feat_b[y - half:y + half, x - half:x + half]
            p_a = feat_a[y - half:y + half, x - half:x + half]
            if p_b.shape != (patch, patch) or float(np.std(p_b)) < 0.01:
                continue
            (dx, dy), r = phase_correlate(p_a, p_b, win)
            if not (r > RESP_MIN) or not (math.isfinite(dx) and math.isfinite(dy)) or abs(dx) > lim or abs(dy) > lim:
                continue
            dx, dy = LocalPatchAligner._refine_translation(feat_a, p_b, x, y, dx, dy, win)
            if not (math.isfinite(dx) and math.isfinite(dy)) or abs(dx) > lim or abs(dy) > lim:
                continue
            pts.append((x, y))
            res.append((dx, dy))
            resp.append(r)
    if not pts:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2), np.float64), np.zeros(0, np.float64)
    return np.array(pts, np.int64), np.array(res, np.float64), np.array(resp, np.float64)


def residual_stats(res: np.ndarray) -> Dict[str, Any]:
    """{n, median, p95, max, frac_ok} of the shift length |res| (px); frac_ok = share under 0.3 px (SPEC 1b gate)."""
    r = np.asarray(res, np.float64).reshape(-1, 2)
    if not len(r):
        return {"n": 0, "median": None, "p95": None, "max": None, "frac_ok": None}
    m = np.hypot(r[:, 0], r[:, 1])
    return {"n": int(len(m)), "median": round(float(np.median(m)), 4), "p95": round(float(np.percentile(m, 95)), 4),
            "max": round(float(m.max()), 4), "frac_ok": round(float(np.mean(m < 0.3)), 4)}


def residual_map(pts: np.ndarray, res: np.ndarray, plate_w: int, plate_h: int,
                 step: int = RESIDUAL_STEP_PX) -> np.ndarray:
    """Coarse map (ceil(h / step), ceil(w / step)) float32 of the shift length per cell: the mean of the control
    points falling into the cell, NaN where none."""
    gw, gh = max(1, math.ceil(plate_w / step)), max(1, math.ceil(plate_h / step))
    acc = np.zeros((gh, gw), np.float64)
    cnt = np.zeros((gh, gw), np.float64)
    p = np.asarray(pts).reshape(-1, 2)
    r = np.asarray(res, np.float64).reshape(-1, 2)
    if len(p):
        m = np.hypot(r[:, 0], r[:, 1])
        cx = np.clip((p[:, 0] * gw) // max(1, plate_w), 0, gw - 1).astype(int)
        cy = np.clip((p[:, 1] * gh) // max(1, plate_h), 0, gh - 1).astype(int)
        np.add.at(acc, (cy, cx), m)
        np.add.at(cnt, (cy, cx), 1.0)
    out = np.full((gh, gw), np.nan, np.float32)
    nz = cnt > 0
    out[nz] = (acc[nz] / cnt[nz]).astype(np.float32)
    return out


def residual_png(rmap: np.ndarray, out_w: int = 1024, vmax: float = 1.0) -> np.ndarray:
    """BGRA uint8 heat map of a coarse residual map scaled to `out_w` px wide (nearest cells): 0 px = blue,
    vmax px and above = red (turbo), no measurement = transparent. The viewer draws it over the plate."""
    gh, gw = rmap.shape
    out_h = max(1, int(round(out_w * gh / float(gw))))
    v = np.nan_to_num(np.clip(rmap / max(vmax, 1e-6), 0.0, 1.0), nan=0.0)
    u8 = (v * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    a = np.where(np.isfinite(rmap), 190, 0).astype(np.uint8)
    img = np.dstack([col, a])
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def frame_residual(beauty: np.ndarray, aux: np.ndarray, frame: Optional[AlignmentFrameResult],
                   grid: Optional[align_grid.AlignGrid], mode: str = "cleanplate",
                   exclude: Optional[np.ndarray] = None, step: int = RESIDUAL_STEP_PX,
                   patch: int = RESIDUAL_PATCH_PX, static: bool = True) -> Dict[str, Any]:
    """Residual of ONE frame: the aux (full plate, float BGR or luma) warped by the composed alignment of `frame`
    (+ `grid`; None = the raw aux: the "before"), features of both at full resolution, control points over the static
    area (`compute_static_mask` when `static`, else only margins) minus `exclude` (bool / float mask, 1 = exclude)
    and minus what the warp left black. Returns {"pts", "res", "resp", "stats", "map", "mask_frac"}."""
    h, w = beauty.shape[:2]
    if frame is not None and not (frame.method == "identity"):
        mx, my = compose_remap(frame, grid, w, h, plate_w=w)
        aligned = cv2.remap(aux, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cover = cv2.remap(np.ones((h, w), np.float32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
    else:
        aligned = aux
        cover = np.ones((h, w), np.float32)
    if static:
        mask = compute_static_mask(beauty, aligned, mode=mode, scale=1.0)
    else:
        mask = np.ones((h, w), np.float32)
        b = max(4, int(round(min(h, w) * 0.02)))
        mask[:b] = mask[-b:] = 0.0
        mask[:, :b] = mask[:, -b:] = 0.0
    if exclude is not None:
        ex = np.asarray(exclude)
        if ex.shape[:2] != (h, w):
            ex = cv2.resize(ex.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask * (1.0 - (ex > 0.5).astype(np.float32))
    k = patch // 2 + 4
    cov = cv2.erode((cover >= 1.0).astype(np.float32), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    mask = mask * cov
    f_b = AlignmentFeatureExtractor.extract(beauty, mode=mode, scale=1.0)
    f_a = AlignmentFeatureExtractor.extract(aligned, mode=mode, scale=1.0)
    pts, res, resp = residual_points(f_b, f_a, mask, step=step, patch=patch)
    return {"pts": pts, "res": res, "resp": resp, "stats": residual_stats(res),
            "map": residual_map(pts, res, w, h, step), "mask_frac": round(float((mask > 0.5).mean()), 4)}


def zone_stats(pts: np.ndarray, res: np.ndarray, zone: np.ndarray) -> Dict[str, Any]:
    """`residual_stats` of the control points inside `zone` (bool mask, plate size) and outside it:
    {"in": {...}, "out": {...}} - e.g. the area an object passed through vs the rest."""
    p = np.asarray(pts).reshape(-1, 2)
    if not len(p):
        return {"in": residual_stats(np.zeros((0, 2))), "out": residual_stats(np.zeros((0, 2)))}
    z = np.asarray(zone, bool)
    inside = z[np.clip(p[:, 1], 0, z.shape[0] - 1), np.clip(p[:, 0], 0, z.shape[1] - 1)]
    r = np.asarray(res, np.float64).reshape(-1, 2)
    return {"in": residual_stats(r[inside]), "out": residual_stats(r[~inside])}
