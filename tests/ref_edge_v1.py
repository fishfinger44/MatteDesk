# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/edge.py - Edge accuracy (SPEC 7.8): position vs the plate, adaptive band, flow consistency, alpha hygiene.

1. Edge position: along the alpha=0.5 contour every 2 px, a +-6 px profile of the plate-to-mask luma along the normal;
   the true edge = position of the largest gradient. Signed offset (+ = mask too wide / halo, - = too narrow), only on
   sharp, contrasty edges (10-90 % width < 3 px, contrast > 3x noise); median over ~64 px contour pieces, smoothed
   in time (+-4). |median| > 0.5 px for >= 5 consecutive frames -> correction: local SDF shift of the core (smooth
   displacement along the contour, max +-3 px), then "Przelicz finalnie" again; max 1 automatic iteration (Z6).
2. Adaptive band: width = clamp(2 + 1.5 x (focus blur + motion blur), 3, 24) px x band_scale; focus blur = 10-90 %
   transition width on the plate, motion blur = optical flow along the normal x shutter (deg / 360).
3. Flow consistency: DIS flow on the plate-to-mask, alpha(t) warped to t+1, IoU in the band (occlusions excluded by a
   forward/backward check) -> flags / key hints.
4. Hygiene (last step before QA/export): clip to [0, 1]; core outside the band exactly 1; outside the object exactly 0;
   islands < 600 px removed; interior holes filled unless from a Remove zone / user key; counts reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.segment import sdf


# ----------------------------------------------------------------------------- 1. edge position
def _contour_samples(mask: np.ndarray, step: int = 2):
    """Points of the alpha=0.5 contour every `step` px, unit normals pointing OUT of the object, piece index."""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts, nrm, piece = [], [], []
    d_in = sdf(mask)
    for ci, c in enumerate(cnts):
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 16:
            continue
        for i in range(0, len(c), step):
            t = c[(i + 3) % len(c)] - c[i - 3]
            n = np.array([t[1], -t[0]], np.float32)
            ln = float(np.linalg.norm(n))
            if ln == 0:
                continue
            n /= ln
            # orient outward: SDF decreases outward
            p = c[i]
            pa = np.clip(np.round(p + 2 * n).astype(int), 0, [m.shape[1] - 1, m.shape[0] - 1])
            if d_in[pa[1], pa[0]] > 0:
                n = -n
            # boundary pixels lie half a pixel inside the 0.5 iso-line
            pts.append(p + 0.5 * n)
            nrm.append(n)
            piece.append((ci, i // 32))  # ~64 px pieces
    return np.array(pts, np.float32).reshape(-1, 2), np.array(nrm, np.float32).reshape(-1, 2), piece


@dataclass
class EdgeMeasure:
    offsets: np.ndarray            # per sample, px (+ = too wide), NaN where not measurable
    points: np.ndarray
    normals: np.ndarray
    pieces: List[Tuple[int, int]]
    widths: np.ndarray             # 10-90 % transition widths (focus blur), px
    piece_median: Dict[Tuple[int, int], float] = field(default_factory=dict)

    @property
    def median_abs(self) -> float:
        v = self.offsets[~np.isnan(self.offsets)]
        return float(np.median(np.abs(v))) if len(v) else 0.0

    @property
    def median(self) -> float:
        v = self.offsets[~np.isnan(self.offsets)]
        return float(np.median(v)) if len(v) else 0.0


def measure_edge(alpha: np.ndarray, gray: np.ndarray, half: int = 6, noise: Optional[float] = None) -> EdgeMeasure:
    """Offsets of the alpha=0.5 contour vs the strongest plate gradient along each normal (sub-pixel)."""
    pts, nrm, piece = _contour_samples(alpha)
    n = len(pts)
    if n == 0:
        return EdgeMeasure(np.zeros(0), pts, nrm, piece, np.zeros(0))
    ts = np.arange(-2 * half, 2 * half + 1, dtype=np.float32) * 0.5     # +-half px, 0.5 px sampling
    xs = (pts[:, None, 0] + nrm[:, None, 0] * ts[None]).astype(np.float32)
    ys = (pts[:, None, 1] + nrm[:, None, 1] * ts[None]).astype(np.float32)
    g = gray.astype(np.float32)
    prof = cv2.remap(g, xs - 0.0, ys - 0.0, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if noise is None:
        lap = cv2.Laplacian(g, cv2.CV_32F)
        noise = 1.4826 * float(np.median(np.abs(lap - np.median(lap)))) / 2.0 + 1e-6
    grad = np.abs(np.gradient(prof, axis=1)) / 0.5
    k = np.argmax(grad, axis=1)
    off = np.full(n, np.nan, np.float32)
    widths = np.full(n, np.inf, np.float32)
    lo, hi = prof[:, :3].mean(1), prof[:, -3:].mean(1)
    contrast = np.abs(hi - lo)
    for i in range(n):
        if contrast[i] < 3 * noise:
            continue
        q = (prof[i] - lo[i]) / (hi[i] - lo[i])
        if (q >= 0.1).any() and (q >= 0.9).any():   # q rises 0 -> 1 for both edge polarities
            widths[i] = abs(np.argmax(q >= 0.9) - np.argmax(q >= 0.1)) * 0.5
        if widths[i] >= 3.0:
            continue  # soft edge: gradient does not define the true edge (7.8 pkt 1)
        j = int(k[i])
        if 0 < j < len(ts) - 1:  # parabolic sub-sample peak
            a, b, c = grad[i, j - 1], grad[i, j], grad[i, j + 1]
            den = a - 2 * b + c
            dj = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        else:
            dj = 0.0
        pos = ts[0] + (j + dj) * 0.5
        off[i] = -pos  # edge found at +pos along the outward normal -> mask too narrow by pos
    m = EdgeMeasure(off, pts, nrm, piece, widths)
    by: Dict[Tuple[int, int], List[float]] = {}
    for p, o in zip(piece, off):
        if not np.isnan(o):
            by.setdefault(p, []).append(float(o))
    m.piece_median = {p: float(np.median(v)) for p, v in by.items() if len(v) >= 4}
    return m


def correction_map(core: np.ndarray, meas: EdgeMeasure, max_px: float = 3.0, smooth_px: float = 24.0) -> np.ndarray:
    """Smooth signed displacement map (px) from per-piece medians: + = shrink (halo), - = grow."""
    h, w = core.shape
    acc = np.zeros((h, w), np.float32)
    wgt = np.zeros((h, w), np.float32)
    for p, o, piece in zip(meas.points, meas.offsets, meas.pieces):
        v = meas.piece_median.get(piece)
        if v is None:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            acc[y, x] += np.clip(v, -max_px, max_px)
            wgt[y, x] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), smooth_px / 3)
    wgt = cv2.GaussianBlur(wgt, (0, 0), smooth_px / 3)
    return np.where(wgt > 1e-6, acc / np.maximum(wgt, 1e-6), 0.0).astype(np.float32)


def apply_shift(core: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """SDF shift of the core: positive disp shrinks, negative grows (shrink/grow slider = constant disp)."""
    return (sdf(core) - disp) > 0


def needs_correction(per_frame_piece: Sequence[Dict[Tuple[int, int], float]], thr: float = 0.5, run: int = 5
                     ) -> Dict[Tuple[int, int], bool]:
    """Pieces with |median| > thr in >= `run` consecutive frames (after +-4 temporal smoothing)."""
    keys = set().union(*[set(d) for d in per_frame_piece]) if per_frame_piece else set()
    out = {}
    for k in keys:
        series = np.array([d.get(k, np.nan) for d in per_frame_piece], np.float32)
        sm = np.array([np.nanmedian(series[max(0, i - 4):i + 5]) if not np.all(np.isnan(series[max(0, i - 4):i + 5]))
                       else np.nan for i in range(len(series))])
        bad = np.abs(np.nan_to_num(sm)) > thr
        best = cur = 0
        for b in bad:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        out[k] = best >= run
    return out


# ----------------------------------------------------------------------------- 2. adaptive band
def adaptive_band_width(core: np.ndarray, gray: np.ndarray, flow: Optional[np.ndarray] = None,
                        shutter_deg: float = 180.0, band_scale: float = 1.0) -> np.ndarray:
    """Per-pixel band half-width map (px) around the contour: clamp(2 + 1.5 (focus + motion), 3, 24) x band_scale."""
    meas = measure_edge(core.astype(np.float32), gray)
    h, w = core.shape
    if len(meas.points) == 0:
        return np.full((h, w), 3.0 * band_scale, np.float32)
    focus = np.where(np.isfinite(meas.widths), meas.widths, 3.0)
    motion = np.zeros(len(meas.points), np.float32)
    if flow is not None:
        xi = np.clip(np.round(meas.points[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(meas.points[:, 1]).astype(int), 0, h - 1)
        v = flow[yi, xi]
        motion = np.abs((v * meas.normals).sum(1)) * (shutter_deg / 360.0)
    width = np.clip(2 + 1.5 * (focus + motion), 3, 24) * band_scale
    # smooth along the contour (median of neighbours) and paint into a nearest-sample map
    width = np.array([np.median(width[max(0, i - 3):i + 4]) for i in range(len(width))], np.float32)
    seeds = np.ones((h, w), np.uint8)
    lab_img = np.zeros((h, w), np.int32)
    for i, p in enumerate(meas.points):
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            seeds[y, x] = 0
            lab_img[y, x] = i + 1
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # map label ids -> sample index
    lut = np.zeros(labels.max() + 1, np.float32)
    ys, xs = np.where(seeds == 0)
    lut[labels[ys, xs]] = width[lab_img[ys, xs] - 1]
    return lut[labels]


def adaptive_band(core: np.ndarray, width_map: np.ndarray) -> np.ndarray:
    return np.abs(sdf(core)) <= width_map


# ----------------------------------------------------------------------------- 3. flow consistency
def flow_consistency(a0: np.ndarray, a1: np.ndarray, g0: np.ndarray, g1: np.ndarray, band_px: float = 6.0
                     ) -> Tuple[float, np.ndarray]:
    """IoU of alpha(t) warped by DIS flow to t+1 vs alpha(t+1), in the edge band, occlusions excluded."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    u8 = lambda g: cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    f10 = dis.calc(u8(g1), u8(g0), None)   # for each pixel of t+1, where it was in t
    f01 = dis.calc(u8(g0), u8(g1), None)
    h, w = a0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(a0, gx + f10[..., 0], gy + f10[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    back = cv2.remap(f01, gx + f10[..., 0], gy + f10[..., 1], cv2.INTER_LINEAR)
    occl = np.linalg.norm(f10 + back, axis=2) > 1.0
    band = (np.abs(sdf(a1 > 0.5)) <= band_px) & ~occl
    a, b = (warped > 0.5) & band, (a1 > 0.5) & band
    u = (a | b).sum()
    mism = (a ^ b)
    return (1.0 if u == 0 else float((a & b).sum() / u)), mism


# ----------------------------------------------------------------------------- 4. hygiene
def hygiene(alpha: np.ndarray, core: np.ndarray, band: np.ndarray, keep_holes: Optional[np.ndarray] = None,
            island_px: int = 600) -> Tuple[np.ndarray, Dict[str, int]]:
    """Exact 1 inside / 0 outside, islands and unexplained holes cleaned. Returns (alpha, report counts)."""
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    rep = {}
    inside = core & ~band
    rep["interior_set_to_1"] = int((a[inside] < 1.0).sum())
    a[inside] = 1.0
    # noise 0 < a < 0.01 not connected to the edge band -> 0
    faint = (a > 0) & (a < 0.01)
    n, lab = cv2.connectedComponents(faint.astype(np.uint8))
    touch = np.unique(lab[faint & cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)])
    kill = faint & ~np.isin(lab, touch[touch > 0])
    outside = ~band & ~core
    rep["outside_set_to_0"] = int(((a > 0) & outside).sum() + kill.sum())
    a[outside] = 0.0
    a[kill] = 0.0
    fg = (a > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    islands = 0
    if n > 2:
        main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        for i in range(1, n):
            if i != main and st[i, cv2.CC_STAT_AREA] < island_px:
                a[lab == i] = 0.0
                islands += 1
    rep["islands_removed"] = islands
    bg = (a <= 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    holes = 0
    h, w = a.shape
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        reg = lab == i
        if keep_holes is not None and keep_holes[reg].mean() > 0.5:
            continue
        a[reg] = 1.0
        holes += 1
    rep["holes_filled"] = holes
    return a, rep
