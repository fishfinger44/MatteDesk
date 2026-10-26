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
   islands < 600 px removed; interior holes filled unless from a Remove zone / user key or already a hole of the core (mask 1+2); counts reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.segment import sdf


# ----------------------------------------------------------------------------- 1. edge position
def _contour_samples(mask: np.ndarray, step: int = 2, valid: Optional[np.ndarray] = None):
    """Points of the alpha=0.5 contour every `step` px, unit normals pointing OUT of the object, piece index.
    Vectorized per contour (2026-10-13): the same float32 operations as the former per-point loop, bit for bit
    (tests/test_edge_vectorized.py). `valid` (bool, mask shape): samples whose rounded point lies outside it are
    dropped – the mirror-padded part of a crop outside the plate has no real edge (its contour pieces showed up
    outside the frame in the stage-4 overlay)."""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    d_in = sdf(mask)
    hi_xy = [m.shape[1] - 1, m.shape[0] - 1]
    pts, nrm, piece = [], [], []
    for ci, c in enumerate(cnts):
        c = c[:, 0, :].astype(np.float32)
        L = len(c)
        if L < 16:
            continue
        idx = np.arange(0, L, step)
        t = c[(idx + 3) % L] - c[idx - 3]                       # c[i - 3] of the loop: negative index wraps
        n = np.stack([t[:, 1], -t[:, 0]], axis=1).astype(np.float32)
        ln = np.sqrt(n[:, 0] * n[:, 0] + n[:, 1] * n[:, 1])     # = float32 sqrt(dot(n, n)) of np.linalg.norm
        keep = ln != 0
        idx, n, ln = idx[keep], n[keep], ln[keep]
        n = n / ln[:, None]
        p = c[idx]
        # orient outward: SDF decreases outward
        pa = np.clip(np.round(p + 2 * n).astype(int), 0, hi_xy)
        flip = d_in[pa[:, 1], pa[:, 0]] > 0
        n[flip] = -n[flip]
        # boundary pixels lie half a pixel inside the 0.5 iso-line
        pq = p + 0.5 * n
        if valid is not None:
            pi = np.clip(np.round(pq).astype(int), 0, hi_xy)
            inside = valid[pi[:, 1], pi[:, 0]]
            idx, n, pq = idx[inside], n[inside], pq[inside]
        pts.append(pq)
        nrm.append(n)
        piece += [(ci, int(i) // 32) for i in idx]               # ~64 px pieces
    if not pts:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), piece
    return np.concatenate(pts).astype(np.float32), np.concatenate(nrm).astype(np.float32), piece


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


def measure_edge(alpha: np.ndarray, gray: np.ndarray, half: int = 6, noise: Optional[float] = None,
                 valid: Optional[np.ndarray] = None) -> EdgeMeasure:
    """Offsets of the alpha=0.5 contour vs the strongest plate gradient along each normal (sub-pixel).
    Vectorized (2026-10-13) with the exact precision of the former per-sample loop (numpy 1.x scalar promotion:
    float32 where the loop used float32 arrays, float64 where numpy scalars met Python floats) – bit-identical
    offsets, widths and piece medians (tests/test_edge_vectorized.py). `valid`: see `_contour_samples`."""
    pts, nrm, piece = _contour_samples(alpha, valid=valid)
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
    ok = ~(contrast < 3 * noise)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (prof - lo[:, None]) / (hi - lo)[:, None]                  # float32, as prof[i] - lo[i] per row
    q1, q9 = q >= 0.1, q >= 0.9
    has = ok & q1.any(axis=1) & q9.any(axis=1)                            # q rises 0 -> 1 for both edge polarities
    widths[has] = (np.abs(np.argmax(q9, axis=1) - np.argmax(q1, axis=1)) * 0.5)[has]
    sel = ok & (widths < 3.0)              # soft edge (>= 3 px, or unmeasured = inf): gradient does not define it
    j = k.astype(np.int64)
    rows = np.arange(n)
    mid = (j > 0) & (j < len(ts) - 1)      # parabolic sub-sample peak only away from the profile ends
    jm = np.clip(j, 1, len(ts) - 2)
    a, b, c = grad[rows, jm - 1], grad[rows, jm], grad[rows, jm + 1]
    # the loop's numpy-scalar arithmetic: a - 2*b + c and 0.5*(a - c)/den were float64, (a - c) float32
    den = a.astype(np.float64) - 2.0 * b.astype(np.float64) + c.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dj = (0.5 * (a - c).astype(np.float64)) / den
    dj = np.where(mid & (np.abs(den) > 1e-9), dj, 0.0)
    pos = np.float64(ts[0]) + (j.astype(np.float64) + dj) * 0.5
    off[sel] = (-pos).astype(np.float32)[sel]   # edge found at +pos along the outward normal -> mask too narrow by pos
    m = EdgeMeasure(off, pts, nrm, piece, widths)
    m.piece_median = _piece_medians(piece, off)
    return m


def _piece_medians(piece: Sequence[Tuple[int, int]], off: np.ndarray, min_n: int = 4) -> Dict[Tuple[int, int], float]:
    """{piece: median of its measurable offsets} for pieces with >= min_n samples (float64 median of the float32
    offsets, like the former per-sample dict loop; group order does not change a median)."""
    good = ~np.isnan(off)
    if not good.any():
        return {}
    keys = np.asarray(piece, np.int64).reshape(-1, 2)[good]
    vals = off[good].astype(np.float64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, vals = keys[order], vals[order]
    change = np.flatnonzero((keys[1:] != keys[:-1]).any(axis=1)) + 1
    out: Dict[Tuple[int, int], float] = {}
    for s, e in zip(np.r_[0, change], np.r_[change, len(vals)]):
        if e - s >= min_n:
            out[(int(keys[s, 0]), int(keys[s, 1]))] = float(np.median(vals[s:e]))
    return out


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


# edge palette ranges (user request 2026-10-16: the slider is only a shortcut, any value can be typed for experiments,
# e.g. a wide band for motion blur) – only sanity guards here, the sliders' own ranges are much narrower
BAND_SCALE_RANGE = (0.1, 20.0)
SHIFT_PX_RANGE = (-100.0, 100.0)


def clamp_band_scale(v: float) -> float:
    return float(min(BAND_SCALE_RANGE[1], max(BAND_SCALE_RANGE[0], float(v))))


def clamp_shift_px(v: float) -> float:
    return float(min(SHIFT_PX_RANGE[1], max(SHIFT_PX_RANGE[0], float(v))))


# px mode of the edge band (user decision 2026-10-16, like the ViTMatte zone sliders): object fields band_out_px /
# band_in_px = the band reaches at least this far out of / into the mask 1+2 edge; where the adaptive width (blur /
# motion) is larger it wins. The mask is not shifted. Objects without the fields: band_scale / shift_px as before.
BAND_PX_RANGE = (0.0, 200.0)


def clamp_band_px(v: float) -> float:
    return float(min(BAND_PX_RANGE[1], max(BAND_PX_RANGE[0], float(v))))


def edge_settings(obj: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The stage-3 MatAnyone2 band settings – an instance of the one band settings model (engine/objband.BandSettings,
    user request 2026-10-24): `s["band"]` = objband.band_for(obj, "matanyone", overrides) (the shared object band unless
    the stage is detached, unset = engine/bandsys.BAND_DEFAULTS, `overrides` = the palette's values not saved yet,
    canonical or legacy names); the legacy keys {band_out_px, band_in_px, band_in_feather_px, band_out_feather_px,
    band_points, band_ref_out, band_ref_in, band_owner} are its thin mapping for old readers."""
    from engine.objband import band_for
    b = band_for(obj, "matanyone", overrides=overrides)
    return {**b.legacy(), "band": b}


def band_px_limits(obj: Dict[str, Any]) -> Tuple[float, float]:
    """(out_px, in_px) of the stage-3 band (edge_settings)."""
    s = edge_settings(obj)
    return s["band_out_px"], s["band_in_px"]


def band_feathers(obj: Dict[str, Any]) -> Tuple[float, float]:
    """(inset feather, outer feather) of the stage-3 band (edge_settings)."""
    s = edge_settings(obj)
    return s["band_in_feather_px"], s["band_out_feather_px"]


def band_px(core: np.ndarray, width_map: np.ndarray, out_px: float, in_px: float) -> np.ndarray:
    """Asymmetric band of the px mode: max(adaptive width, out_px) outside the edge, max(adaptive, in_px) inside
    (engine/bandsys.band_area of the two width maps – the one band geometry, 2026-10-24)."""
    from engine.bandsys import band_area
    return band_area(sdf(core), np.maximum(width_map, out_px), np.maximum(width_map, in_px))


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
# Gaussian smoothing of the band width along each contour (samples every 2 px -> sigma 6 = ~12 px of contour).
# The 7-sample median alone left neighbouring samples up to ~8 px apart on noisy edges (curly hair): the nearest-sample
# width map then showed square "tabs" sticking out of the band (user report CLIP2_A F58, 2026-10-16).
BAND_CONTOUR_SIGMA = 6.0


def adaptive_band_width(core: np.ndarray, gray: np.ndarray, flow: Optional[np.ndarray] = None,
                        shutter_deg: float = 180.0, band_scale: float = 1.0,
                        contour_sigma: float = BAND_CONTOUR_SIGMA) -> np.ndarray:
    """Per-pixel band half-width map (px) around the contour: clamp(2 + 1.5 (focus + motion), 3, 24) x band_scale.
    `contour_sigma` (samples): circular Gaussian along each contour after the median; 0 = the former widths (v1)."""
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
    width = _smooth_widths(width)
    if contour_sigma > 0:
        width = _smooth_along_contours(width, meas.pieces, contour_sigma)
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
    wmap = lut[labels]
    if contour_sigma > 0:
        # the nearest-sample cells still step where the nearest sample jumps (concave parts, two sides of a neck):
        # with a wide band (x3) the steps showed as square blocks – blur the map itself, sigma ~ half the band
        sig = max(2.0, 0.5 * float(np.median(width)))
        wmap = cv2.GaussianBlur(wmap, (0, 0), sig)
    return wmap


def _smooth_widths(width: np.ndarray) -> np.ndarray:
    """np.median(width[max(0, i - 3):i + 4]) for every i, without 650k Python calls per shot: interior windows
    (7 values, odd) are the 4th of the sorted window = exactly np.median; the 3 truncated windows at each end keep
    the np.median call (even counts: mean of two middles, same code path as before)."""
    n = len(width)
    if n < 7:
        return np.array([np.median(width[max(0, i - 3):i + 4]) for i in range(n)], np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.empty(n, np.float32)
    out[3:n - 3] = np.sort(sliding_window_view(width, 7), axis=1)[:, 3]
    for i in list(range(3)) + list(range(n - 3, n)):
        out[i] = np.median(width[max(0, i - 3):i + 4])
    return out


def _smooth_along_contours(width: np.ndarray, pieces: Sequence[Tuple[int, int]], sigma: float) -> np.ndarray:
    """Circular Gaussian (sigma in samples) of the widths of every contour on its own (contours are closed; samples
    of one contour are consecutive in `width`, `pieces[i][0]` = contour index)."""
    out = width.astype(np.float32).copy()
    if len(width) == 0 or sigma <= 0:
        return out
    ci = np.asarray([p[0] for p in pieces], np.int64)
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float64)
    starts = np.r_[0, np.flatnonzero(ci[1:] != ci[:-1]) + 1, len(ci)]
    for s, e in zip(starts[:-1], starts[1:]):
        seg = width[s:e].astype(np.float64)
        n = len(seg)
        if n < 3:
            continue
        rr = min(r, n - 1)
        kk = k[r - rr:r + rr + 1]
        kk = kk / kk.sum()
        ext = np.r_[seg[-rr:], seg, seg[:rr]] if rr else seg
        out[s:e] = np.convolve(ext, kk, mode="valid").astype(np.float32)
    return out


def adaptive_band(core: np.ndarray, width_map: np.ndarray) -> np.ndarray:
    """|SDF| <= the half-width map = engine/bandsys.band_area with the map on both sides (2026-10-24)."""
    from engine.bandsys import band_area
    return band_area(sdf(core), width_map, width_map)


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
def enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Background of `mask` (> 0.5) enclosed by it: 4-connected components not touching the image border."""
    bg = (np.asarray(mask) <= 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    h, w = bg.shape
    inner = [i for i in range(1, n) if not (st[i, 0] == 0 or st[i, 1] == 0 or st[i, 0] + st[i, 2] >= w
                                            or st[i, 1] + st[i, 3] >= h)]
    return np.isin(lab, inner) if inner else np.zeros((h, w), bool)


def hygiene(alpha: np.ndarray, core: np.ndarray, band: np.ndarray, keep_holes: Optional[np.ndarray] = None,
            island_px: int = 600, keep_islands: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Exact 1 inside / 0 outside, islands and unexplained holes cleaned. Returns (alpha, report counts).
    `keep_islands`: area (soft `matte` zones) where small islands are strands, not noise – never removed."""
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
                reg = lab == i
                if keep_islands is not None and keep_islands[reg].any():
                    continue
                a[reg] = 0.0
                islands += 1
    rep["islands_removed"] = islands
    bg = (a <= 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    holes = 0
    h, w = a.shape
    # holes of the core itself (mask 1+2) are the object's topology, not model defects (sh060 F71 2026-10-26: a gap
    # between arm and body, open in SAM 3.1 / mask 1+2, was filled to 1 whenever the model closed it off from the background)
    core_holes = enclosed_holes(core) if n > 1 else None
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        reg = lab == i
        if keep_holes is not None and keep_holes[reg].mean() > 0.5:
            continue
        if core_holes is not None and core_holes[reg].mean() > 0.5:
            continue
        a[reg] = 1.0
        holes += 1
    rep["holes_filled"] = holes
    return a, rep
