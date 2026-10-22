# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/mb_map.py - motion blur strength measured on the plate (the "hybrid", docs/PLAN_2026-10-22.md section 2,
user decision 2026-10-22): the DIRECTION and shape of the blur come from the object's velocity (engine/motionblur.py),
its STRENGTH per edge segment from the plate itself – how wide the plate's own 10-90 % transition is across the edge,
against how long the blur predicted from the velocity and the shutter would be there.

    m = width_plate / L_pred        L_pred = 0.8 x |v · n| x shutter / 360   (10-90 % width of a box blur that long)

m = 1 means the plate blur is exactly the predicted one (a sharp hand that moves as the flow says), m < 1 a sharper
plate than predicted (the flow overshoots, e.g. a finger tip pulled from the wrist's flow), m > 1 a wider transition
(rolling shutter, a shutter other than the manifest's, defocus on top). Samples whose predicted blur is shorter than
MIN_PRED_PX cannot measure a ratio (a static edge): they take the median of the measured ones of the same contour
(else 1.0 – the plain vector blur). The ratio is smoothed along every contour, painted into a nearest-sample map and
blurred, like the adaptive band width (engine/edge.adaptive_band_width); the global strength slider multiplies it.
Where FG and BG have the same colour the transition is unmeasurable (contrast below the noise): the sample is skipped
and its neighbours fill in – the fallback to the synthesis from the velocity, not a replacement of it.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

PROFILE_HALF_PX = 24      # measure_edge profile +- this many px across the edge (12-17 px blurs on CLIP2_A, 30 px p90)
MIN_PRED_PX = 1.5         # predicted blur shorter than this: the ratio is not measurable at that sample
M_MAX = 4.0               # ratio clamp (a 4x longer transition than predicted = defocus / another shutter, not more)
CONTOUR_SIGMA = 6.0       # samples (every 2 px) – as the band width smoothing


def strength_map(alpha: np.ndarray, gray: np.ndarray, vx: np.ndarray, vy: np.ndarray, shutter_deg: float,
                 half: int = PROFILE_HALF_PX) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(m map float32 h x w, stats) – the per-pixel strength multiplier of the vector blur, measured on the plate.
    `alpha` = the alpha the blur is applied to (its 0.5 contour is the edge), `gray` = the plate's gray (uint8 or
    float), `vx, vy` = the object's velocity field (px / frame, engine/motionblur.object_velocity)."""
    from engine.edge import _smooth_along_contours, _smooth_widths, measure_edge
    h, w = alpha.shape
    ones = np.ones((h, w), np.float32)
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    meas = measure_edge((alpha > 0.5).astype(np.float32), g, half=half)
    n = len(meas.points)
    st: Dict[str, Any] = {"samples": int(n), "measured": 0, "ratio_median": 1.0, "width_median": 0.0, "pred_median": 0.0}
    if n == 0:
        return ones, st
    xi = np.clip(np.round(meas.points[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(meas.points[:, 1]).astype(int), 0, h - 1)
    v = np.stack([vx[yi, xi], vy[yi, xi]], axis=1)
    # blur length along the normal, px; a box blur of L px shows a 10-90 % transition of 0.8 L – that is what the
    # width measures, so m = 1 for a plate blurred exactly as predicted
    pred = 0.8 * np.abs((v * meas.normals).sum(1)) * (float(shutter_deg) / 360.0)
    width = np.asarray(meas.widths, np.float32)
    ok = np.isfinite(width) & (pred >= MIN_PRED_PX)
    ratio = np.full(n, np.nan, np.float32)
    ratio[ok] = np.clip(width[ok] / np.maximum(pred[ok], 1e-3), 0.0, M_MAX)
    st["measured"] = int(ok.sum())
    if ok.any():
        st["ratio_median"] = round(float(np.median(ratio[ok])), 3)
        st["width_median"] = round(float(np.median(width[ok])), 2)
        st["pred_median"] = round(float(np.median(pred[ok])), 2)
    # unmeasured samples: the median of their own contour, else 1 (the plain vector blur)
    ci = np.asarray([p[0] for p in meas.pieces], np.int64)
    for c in np.unique(ci):
        sel = ci == c
        good = sel & ok
        fill = float(np.median(ratio[good])) if good.any() else 1.0
        ratio[sel & ~ok] = fill
    ratio = _smooth_widths(ratio)
    ratio = _smooth_along_contours(ratio, meas.pieces, CONTOUR_SIGMA)
    m = paint_contour_map(meas.points, ratio, (h, w), fill=1.0)
    return m, st


class MbCache:
    """Per-frame cache of what the motion blur needs and the sliders do not change (user request 2026-10-22: GPU +
    cache): the object's velocity at flow scale and the measured ratio map (flow scale, float16), keyed by the frame,
    the object's outline (mask at flow scale) and the shutter – a slider move then costs only the vector blur.
    Files `<dir>/f_<frame>_<key>.npz` (compressed) + a small memory LRU."""

    def __init__(self, directory: Optional[Path], mem: int = 12):
        self.dir = Path(directory) if directory else None
        self.mem: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.n_mem = mem

    @staticmethod
    def key(frame: int, mask_small: np.ndarray, shutter_deg: float) -> str:
        h = hashlib.md5(np.packbits(mask_small.astype(bool)).tobytes()).hexdigest()[:12]
        return f"{int(frame)}_{h}_{int(round(float(shutter_deg)))}"

    def _path(self, key: str) -> Optional[Path]:
        return (self.dir / f"f_{key}.npz") if self.dir else None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        p = self._path(key)
        if p is None or not p.exists():
            return None
        try:
            with np.load(p) as z:
                d = {"vs": z["vs"].astype(np.float32), "s": float(z["s"]), "m": (z["m"].astype(np.float32) if "m" in z else None)}
        except (OSError, ValueError, KeyError):
            return None
        self._remember(key, d)
        return d

    def put(self, key: str, vs: np.ndarray, s: float, m_small: Optional[np.ndarray]) -> None:
        d = {"vs": vs.astype(np.float32), "s": float(s), "m": None if m_small is None else m_small.astype(np.float32)}
        self._remember(key, d)
        p = self._path(key)
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            arrs = {"vs": vs.astype(np.float16), "s": np.float32(s)}
            if m_small is not None:
                arrs["m"] = m_small.astype(np.float16)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrs)
            tmp.replace(p)
        except OSError:
            pass

    def _remember(self, key: str, d: Dict[str, Any]) -> None:
        self.mem[key] = d
        self.mem.move_to_end(key)
        while len(self.mem) > self.n_mem:
            self.mem.popitem(last=False)


def paint_contour_map(points: np.ndarray, values: np.ndarray, shape: Tuple[int, int], fill: float = 1.0,
                      blur_sigma: Optional[float] = None) -> np.ndarray:
    """Every pixel takes the value of its nearest contour sample (as engine/edge.adaptive_band_width), then a Gaussian
    blur hides the cell steps; no samples = `fill` everywhere."""
    h, w = shape
    if len(points) == 0:
        return np.full((h, w), float(fill), np.float32)
    seeds = np.ones((h, w), np.uint8)
    lab_img = np.zeros((h, w), np.int32)
    for i, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            seeds[y, x] = 0
            lab_img[y, x] = i + 1
    if not (seeds == 0).any():
        return np.full((h, w), float(fill), np.float32)
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.full(int(labels.max()) + 1, float(fill), np.float32)
    ys, xs = np.where(seeds == 0)
    lut[labels[ys, xs]] = np.asarray(values, np.float32)[lab_img[ys, xs] - 1]
    m = lut[labels]
    sig = blur_sigma if blur_sigma is not None else 4.0
    return cv2.GaussianBlur(m, (0, 0), sig) if sig > 0 else m
