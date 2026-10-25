# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandmap.py - variable band width from control points on the outline (user idea 2026-10-18, plan 2026-10-22
topic 3; started 2026-10-21 for the stage-5 zone in "whole outline" mode).

Control points live in the zone: params["band_points"] = [
    {"id": "p1", "keys": {"50": {"x": 612.0, "y": 88.0, "out": 18, "in": 4}, "70": {...}}}, ...]
Each key = the point's position (full-frame px) and its two "whiskers" in that frame: `out` = band into the background,
`in` = band into the object (px). Between keys the widths are animated linearly; the position comes from tracking
(engine/bandtrack.py; without a tracker: the nearest key). Width 0 on both sides = a gap in the band (no edge model
there: skin, a hand). No points -> the zone's constant band, bit-identical to before (outline_band).

Per frame the points are snapped to the nearest outline pixel; along each closed contour the widths are interpolated
linearly by arc length between its points (a contour without points keeps the zone's defaults), then every pixel takes
the widths of its nearest contour pixel: band = outside within `out` of the contour  |  inside within `in` of it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA = "band_points@1"
MAX_WIDTH_PX = 400

# (x, y, out_px, in_px) of one visible point in one frame; a TILTED point (whiskers turned round it, 2026-10-26) has
# a 5th item: the angle in degrees
PointAt = Tuple[float, ...]
MAX_ANGLE_DEG = 80.0
NEAR_CONTOUR_PX = 120.0         # an unpointed contour this close to a pointed one takes its widths


def band_points(params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The zone's control points (validated shape); [] = none."""
    pts = (params or {}).get("band_points") or []
    return [p for p in pts if isinstance(p, dict) and p.get("keys")]


def _keys_sorted(p: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    return sorted(((int(k), v) for k, v in p["keys"].items()), key=lambda kv: kv[0])


def widths_at(p: Dict[str, Any], f: int) -> Tuple[float, float]:
    """(out, in) of point `p` in frame f: linear between its keys, held before the first / after the last."""
    ks = _keys_sorted(p)
    if f <= ks[0][0]:
        k = ks[0][1]
        return float(k["out"]), float(k["in"])
    if f >= ks[-1][0]:
        k = ks[-1][1]
        return float(k["out"]), float(k["in"])
    for (f0, a), (f1, b) in zip(ks, ks[1:]):
        if f0 <= f <= f1:
            t = (f - f0) / float(f1 - f0)
            return (float(a["out"]) + t * (float(b["out"]) - float(a["out"])),
                    float(a["in"]) + t * (float(b["in"]) - float(a["in"])))
    k = ks[-1][1]
    return float(k["out"]), float(k["in"])


def angle_at(p: Dict[str, Any], f: int) -> float:
    """Tilt of the point's whisker pair in frame f (degrees, key "ang", 0 = along the outline's normal): the pair turns
    rigidly round the point, + = toward the direction of the contour's arc. Linear between keys, held outside."""
    ks = _keys_sorted(p)
    a = lambda k: max(-MAX_ANGLE_DEG, min(MAX_ANGLE_DEG, float(k.get("ang", 0.0) or 0.0)))
    if f <= ks[0][0]:
        return a(ks[0][1])
    if f >= ks[-1][0]:
        return a(ks[-1][1])
    for (f0, k0), (f1, k1) in zip(ks, ks[1:]):
        if f0 <= f <= f1:
            return a(k0) + (f - f0) / float(f1 - f0) * (a(k1) - a(k0))
    return a(ks[-1][1])


def nearest_key(p: Dict[str, Any], f: int) -> int:
    return min((int(k) for k in p["keys"]), key=lambda k: (abs(k - f), k))


def static_positions(points: Sequence[Dict[str, Any]], frames: Sequence[int]) -> Dict[int, Dict[str, Tuple[float, float, bool]]]:
    """Positions without a tracker: the nearest key's position (the snap to the outline does the rest), visible."""
    out: Dict[int, Dict[str, Tuple[float, float, bool]]] = {}
    for f in frames:
        out[f] = {}
        for p in points:
            k = p["keys"][str(nearest_key(p, f))]
            out[f][p["id"]] = (float(k["x"]), float(k["y"]), True)
    return out


def points_at(points: Sequence[Dict[str, Any]], f: int,
              positions: Dict[str, Tuple[float, float, bool]], s_out: float = 1.0, s_in: float = 1.0) -> List[PointAt]:
    """Visible points of frame f with their widths (an occluded point does not shape the band). `s_out` / `s_in`: the
    zone sliders' scale (zones.band_scales) – the points shape the band, the sliders size it."""
    out: List[PointAt] = []
    for p in points:
        pos = positions.get(p["id"])
        if pos is None or not pos[2]:
            continue
        o, i = widths_at(p, f)
        t = (pos[0], pos[1], max(0.0, min(MAX_WIDTH_PX, o * s_out)), max(0.0, min(MAX_WIDTH_PX, i * s_in)))
        a = angle_at(p, f)
        out.append(t + (a,) if a else t)
    return out


def _contours(core: np.ndarray) -> List[np.ndarray]:
    cs, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :].astype(np.float32) for c in cs if len(c) >= 3]


def _interp_closed(n: int, arc: np.ndarray, at: List[Tuple[float, float]], total: float) -> np.ndarray:
    """Values at every contour sample: linear by arc length between (s, value) anchors on a closed curve."""
    if not at:
        return np.full(n, np.nan, np.float32)
    at = sorted(at)
    if len(at) == 1:
        return np.full(n, at[0][1], np.float32)
    s = np.array([a[0] for a in at] + [at[0][0] + total], np.float64)
    v = np.array([a[1] for a in at] + [at[0][1]], np.float64)
    q = np.where(arc < s[0], arc + total, arc)            # samples before the first anchor wrap round
    return np.interp(q, s, v).astype(np.float32)


def _frame_at(core: np.ndarray, c: np.ndarray, j: int) -> Tuple[float, float, float, float, float, float]:
    """(sx, sy, nx, ny, tx, ty) at sample j of contour `c`: the outward normal (+n into the background) and the tangent
    (the direction of the contour's index = of its arc length)."""
    n = len(c)
    t = c[(j + 4) % n] - c[(j - 4) % n]
    ln = float(np.hypot(t[0], t[1])) or 1.0
    nx, ny = float(t[1] / ln), float(-t[0] / ln)
    tx, ty = float(t[0] / ln), float(t[1] / ln)
    sx, sy = float(c[j, 0]), float(c[j, 1])
    h, w = core.shape
    px, py = int(round(sx + 3 * nx)), int(round(sy + 3 * ny))
    if 0 <= px < w and 0 <= py < h and core[py, px]:
        nx, ny = -nx, -ny                    # +n must point into the background
    return sx, sy, nx, ny, tx, ty


def _smooth_closed(v: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Gaussian smoothing of a closed curve's samples (wraps round); NaN (no anchors) stays."""
    if np.isnan(v).any() or len(v) < 8:
        return v
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    ext = np.concatenate([v[-r:], v, v[:r]]) if len(v) > r else np.tile(v, 3)[len(v) - r: 2 * len(v) + r]
    return np.convolve(ext, k, mode="valid").astype(np.float32)[: len(v)]


def snap_normal(core: np.ndarray, x: float, y: float, contours: Optional[List[np.ndarray]] = None
                ) -> Optional[Tuple[float, float, float, float]]:
    r = snap_frame(core, x, y, contours)
    return None if r is None else r[:4]


def snap_frame(core: np.ndarray, x: float, y: float, contours: Optional[List[np.ndarray]] = None
               ) -> Optional[Tuple[float, float, float, float, float, float]]:
    """(sx, sy, nx, ny): the nearest outline pixel and the outward unit normal there (the UI draws the whiskers along
    it: out = +n, in = -n). None without an outline. `contours`: _contours(core) computed once for many points (one
    findContours per frame, not per point – 24 Sapiens2 points took ~2 s a request, 2026-10-23)."""
    if contours is None:
        contours = _contours(core)
    if not contours:
        return None
    best = min(((ci, int(np.argmin((c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2))) for ci, c in enumerate(contours)),
               key=lambda t: float((contours[t[0]][t[1], 0] - x) ** 2 + (contours[t[0]][t[1], 1] - y) ** 2))
    return _frame_at(core, contours[best[0]], best[1])


def width_maps(core: np.ndarray, pts: Sequence[PointAt], default_out: float, default_in: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(out_map, in_map) float32 px: every pixel gets the widths of its nearest outline pixel."""
    h, w = core.shape
    contours = _contours(core)
    if not contours:
        return np.full((h, w), default_out, np.float32), np.full((h, w), default_in, np.float32)
    allpix = np.concatenate(contours)
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(contours)])
    first = np.cumsum([0] + [len(c) for c in contours])
    anchors: Dict[int, List[Tuple[int, float, float]]] = {}
    tilt_o: Dict[int, List[Tuple[int, float, float]]] = {}
    tilt_i: Dict[int, List[Tuple[int, float, float]]] = {}
    for q in pts:                          # snap: nearest outline pixel of all contours
        x, y, o, i = q[:4]
        a = float(q[4]) if len(q) > 4 else 0.0
        j = int(np.argmin((allpix[:, 0] - x) ** 2 + (allpix[:, 1] - y) ** 2))
        if not a:
            anchors.setdefault(int(owner[j]), []).append((j - int(first[owner[j]]), o, i))
            continue
        # tilted pair (2026-10-26): the band edge passes EXACTLY through each whisker end – the end's nearest outline pixel
        # is the anchor and its distance to the outline the width (an end that crosses the outline = 0 on that side)
        ci = int(owner[j]); jl = j - int(first[ci])
        sx, sy, nx, ny, tx, ty = _frame_at(core, contours[ci], jl)
        ca, sa = float(np.cos(np.radians(a))), float(np.sin(np.radians(a)))
        dx, dy = nx * ca + tx * sa, ny * ca + ty * sa
        for sg, wlen, store in ((1.0, o, tilt_o), (-1.0, i, tilt_i)):
            ex, ey = sx + sg * dx * wlen, sy + sg * dy * wlen
            jj = int(np.argmin((allpix[:, 0] - ex) ** 2 + (allpix[:, 1] - ey) ** 2))
            cj = int(owner[jj])
            xi_, yi_ = int(round(min(max(ex, 0), w - 1))), int(round(min(max(ey, 0), h - 1)))
            on_side = (not core[yi_, xi_]) if sg > 0 else bool(core[yi_, xi_])
            if on_side:
                store.setdefault(cj, []).append((jj - int(first[cj]), float(np.hypot(allpix[jj, 0] - ex, allpix[jj, 1] - ey)), 0.0))
            else:   # the end lies on the wrong side (a thin limb, past an edge): no pinch – the straight estimate cos(a)·L, sin(a)·L along
                store.setdefault(ci, []).append((jl, wlen * ca, sg * wlen * sa))
    w_out = np.empty(len(allpix), np.float32)
    w_in = np.empty(len(allpix), np.float32)
    for ci, c in enumerate(contours):
        seg = np.sqrt(((np.roll(c, -1, axis=0) - c) ** 2).sum(1))
        arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        total = float(seg.sum()) or 1.0
        an = anchors.get(ci, [])
        ao = [(float(arc[k]), o) for k, o, _ in an] + [((float(arc[k]) + sh) % total, wv) for k, wv, sh in tilt_o.get(ci, [])]
        ai = [(float(arc[k]), i) for k, _, i in an] + [((float(arc[k]) + sh) % total, wv) for k, wv, sh in tilt_i.get(ci, [])]
        vo = _interp_closed(len(c), arc, ao, total)
        vi = _interp_closed(len(c), arc, ai, total)
        if ci in tilt_o or ci in tilt_i:   # tilted whiskers make steep ramps: a light smoothing along the outline (no jagged band edge)
            vo, vi = _smooth_closed(vo), _smooth_closed(vi)
        sl = slice(int(first[ci]), int(first[ci + 1]))
        w_out[sl] = np.where(np.isnan(vo), default_out, vo)
        w_in[sl] = np.where(np.isnan(vi), default_in, vi)
    # a contour WITHOUT points next to a contour WITH points (a speck / spur of the mask, a thin split-off part) takes the widths
    # of the nearest pointed outline pixel instead of the constant band (user report 2026-10-26: wedges of the full default
    # width – 62 px – on a dress that the points keep at ~24 px); a far contour (another person) keeps the defaults
    pointed = [ci for ci in range(len(contours)) if anchors.get(ci) or tilt_o.get(ci) or tilt_i.get(ci)]
    if pointed and len(pointed) < len(contours):
        ref = np.concatenate([np.arange(first[ci], first[ci + 1]) for ci in pointed])
        for ci in range(len(contours)):
            if ci in pointed:
                continue
            sl = slice(int(first[ci]), int(first[ci + 1]))
            q = allpix[sl]
            nn = np.empty(len(q), np.int64); dd = np.empty(len(q), np.float32)
            for a0 in range(0, len(q), 256):
                blk = q[a0:a0 + 256]
                dist = ((blk[:, None, :] - allpix[ref][None, :, :]) ** 2).sum(2)
                k = dist.argmin(1)
                nn[a0:a0 + 256] = ref[k]; dd[a0:a0 + 256] = np.sqrt(dist[np.arange(len(blk)), k])
            near = dd <= NEAR_CONTOUR_PX
            if near.any():
                w_out[sl] = np.where(near, w_out[nn], w_out[sl])
                w_in[sl] = np.where(near, w_in[nn], w_in[sl])
    seeds = np.ones((h, w), np.uint8)
    idx = np.zeros((h, w), np.int32)
    xi = np.clip(np.round(allpix[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(allpix[:, 1]).astype(int), 0, h - 1)
    seeds[yi, xi] = 0
    idx[yi, xi] = np.arange(len(allpix)) + 1
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(seeds == 0)
    lut_o = np.zeros(int(labels.max()) + 1, np.float32)
    lut_i = np.zeros_like(lut_o)
    lut_o[labels[ys, xs]] = w_out[idx[ys, xs] - 1]
    lut_i[labels[ys, xs]] = w_in[idx[ys, xs] - 1]
    return lut_o[labels], lut_i[labels]


def variable_band(core: np.ndarray, out_map: np.ndarray, in_map: np.ndarray) -> np.ndarray:
    """Outline band with per-pixel widths: background pixels within out_map of the core, core pixels within in_map of
    the background (distance 1 = the first pixel row; width 0 = nothing on that side) – engine/bandsys.band_area of
    the signed distance (its two one-sided distance transforms are the former ones, bit for bit; 2026-10-24)."""
    from engine.bandsys import band_area
    from engine.segment import sdf
    return band_area(sdf(np.asarray(core).astype(bool)), out_map, in_map)
