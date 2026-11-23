# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandsys.py - ONE band system wherever a stage puts a model's matte over its base mask through a band
(user decision 2026-10-22, CLAUDE.md „Pas krawędzi = jeden system”; the pattern = the stage-5 whole-outline zone).
Since 2026-10-24 this module is the only implementation: the band geometry, the feather rule, the composition and
the preview lines of stages 3 MatAnyone2, 4 VideoMaMa, 5 ViTMatte (whole-outline zones) and 7 LTX-2.5 live here –
a change here is a change in every stage (user request 2026-10-24).

The band lies along the edge of the stage's base mask (the "core" of the stage: the mask 1+2, a stage-3 result ...):
`out` px outside the edge .. `in` px inside it. Both widths are maps when band control points shape the band
(engine/bandmap.py width_maps), scalars otherwise (since 2026-10-24 every stage – 3 included – sizes its band with
the same two sliders, no adaptive mode): `band_area` turns the widths into the band mask (one geometry: −out <= d <= in of the signed distance d
to the core edge, > 0 inside). Inside the band the model's matte is blended with the CORE – the rule
(`lengths`, `weight`, `blend`):

  * the inset feather runs from the INNER band edge outwards over `inner_px`,
  * the outer feather runs from the OUTER band edge inwards over `outer_px`,
  * they never overlap – the inset one keeps its length (up to the band width), the outer one gets the rest,
  * weight w = min(ramp_in, ramp_out); alpha = w · model + (1 − w) · core; 0 px = a hard edge on that side;
  * outside the band the core stays (or `outside`, what the caller keeps there).

ONE way to measure the distances to the band edges (2026-10-24): from the band area – d_in / d_out are distance
transforms to the solid core outside the area and to the background outside it (`area_dists`; the former
zones.feather_maps), the solid side is the core without the area. The band may be any mask (a zone band less the
drawing and the excluded objects, the stage-3 band less a holdout); the stages that size it from widths build it with
`band_area` first (`band_weight`, `compose`, `feather_lines` are those thin wrappers).

„Wcięcie: cały rdzeń” (`whole`, user decision 2026-10-29 – hair as an object of its own: SAM 3.1 glues the light gaps
between the strands into the core): the band covers the WHOLE core (inset width = infinity), so the model replaces the
core everywhere; the inner band edge is then the middle of the core – its ridge of depth – and the distance to it is
the depth deficit (`depth_deficit`: the deepest point within reach minus this pixel's depth – 0 on the ridge, growing
1 px per px towards the edge – and never less than the inset feather minus the depth, so the kept core never touches
the edge and fades in parts thinner than the feather). The inset feather keeps that much core around the middle
(0 px = nothing of the core is left), the outer feather is unchanged. (A
medial axis of the distance transform was tried first, 2026-10-29: on a ragged hair outline it branched to every bump
and chamfer plateaus made parallel rungs – the preview turned into a web.)

Preview lines (`preview_lines`): the band edges (CCOMP contours of the band), where the inset feather ends (the
solid side + the ramp of the inset feather) and where the outer feather ends – the drawing is one function too
(static/bandpoints.js strokeLines)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Width = Union[float, np.ndarray]

# ONE set of band defaults for every stage (user decision 2026-10-24): what a stage / a new zone starts with when the
# object has no value of its own – the stage-5 whole-outline zone's values (the pattern of the band system)
BAND_DEFAULTS: Dict[str, float] = {
    "out_px": 40.0,            # the band outside the base mask edge
    "in_px": 2.0,              # the band inside it (the zone's trimap inset)
    # from the inner band edge: the core blends into the model result; <= in_px, else the ramp runs past the core edge
    # (the hard core step shows, the model alpha just outside it is scaled down – test_e3, sh060 2026-10-26; was 8.0)
    "inset_feather_px": 2.0,
    "outer_feather_px": 8.0,   # from the outer band edge: the model result blends into the core (0 = a hard edge)
}


# ----------------------------------------------------------------------------- the band
def band_widths(core: np.ndarray, out_px: float, in_px: float, pts: Optional[Sequence[Tuple[float, float, float, float]]] = None
                ) -> Tuple[Width, Width]:
    """(out, in) of the band: scalars, or per-pixel maps when band points (x, y, out, in in the core's pixel space) shape
    it – each pixel takes the widths of its nearest edge pixel, interpolated along the edge between the points.
    in_px = inf („cały rdzeń”): the inside is the whole core whatever the points say (they shape the outside only)."""
    whole = is_whole(in_px)
    if not pts:
        return float(out_px), (np.inf if whole else float(in_px))
    from engine.bandmap import width_maps
    o, i = width_maps(np.asarray(core) > 0.5, list(pts), float(out_px), 0.0 if whole else float(in_px))
    return o, (np.inf if whole else i)


# ---- slider keys (user decision 2026-11-15, first used by the Cleanup C1 band): the band sliders keyed per frame –
# {"<frame>": {field: value}}, linear between keys, held before the first / after the last key; no keys = the sliders.
# ONE arithmetic for every band owner (the JS twin: MD_BANDPTS.keyedAt in static/bandpoints.js).
def clean_slider_keys(keys: Any, limits: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    """Keys with every field of `limits` present and clamped to its (lo, hi); anything else dropped; sorted by frame."""
    out: Dict[str, Dict[str, float]] = {}
    for k, v in dict(keys or {}).items():
        try:
            f = int(k)
            out[str(f)] = {n: float(min(hi, max(lo, float(v[n])))) for n, (lo, hi) in limits.items()}
        except (TypeError, ValueError, KeyError):
            continue
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def keyed_at(keys: Optional[Dict[str, Dict[str, float]]], f: Optional[int], base: Dict[str, float]) -> Dict[str, float]:
    """The values of the fields of `base` in frame f: the keys interpolated linearly (held outside them); no keys /
    f None = `base` (the sliders)."""
    if not keys or f is None:
        return {n: float(v) for n, v in base.items()}
    fs = sorted(int(k) for k in keys)
    f = int(f)
    if f <= fs[0] or f >= fs[-1]:
        k = keys[str(fs[0] if f <= fs[0] else fs[-1])]
        return {n: float(k.get(n, base[n])) for n in base}
    i = next(j for j in range(1, len(fs)) if fs[j] >= f)
    a, b = fs[i - 1], fs[i]
    t = (f - a) / float(b - a)
    ka, kb = keys[str(a)], keys[str(b)]
    return {n: float(ka.get(n, base[n])) + t * (float(kb.get(n, base[n])) - float(ka.get(n, base[n]))) for n in base}


# ---- local band points over a signed base (user decision 2026-11-15, the C1 band at a NEGATIVE „Pas na zewnątrz”):
# away from the points the edge is the slider (`base`, < 0 = inside the object), at a point the edge is its whisker
# length from the outline (`out`, plate px), blended smoothly over `local_reach` around the point. A NEW function: the
# point geometry of the stages (band_widths / bandmap.width_maps) is not touched.
LOCAL_REACH_MIN = 48.0      # px – the smallest reach of a point's influence
LOCAL_REACH_K = 1.5         # reach = max(LOCAL_REACH_MIN, k × |whisker − base|)


def local_offsets(shape: Tuple[int, int], pts: Sequence[Tuple[float, ...]], base: float) -> np.ndarray:
    """float32 map of the signed edge offset (px from the objects' outline; < 0 = inside): `base` everywhere, rising /
    falling to each point's whisker `out` near it (smoothstep over its reach; the strongest point wins)."""
    h, w = int(shape[0]), int(shape[1])
    off = np.full((h, w), float(base), np.float32)
    if not pts:
        return off
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for p in pts:
        x, y, out = float(p[0]), float(p[1]), float(p[2])       # p[2] = the signed offset (tip_offsets)
        delta = out - float(base)
        reach = max(LOCAL_REACH_MIN, LOCAL_REACH_K * abs(delta))
        t = np.clip(1.0 - np.hypot(xx - x, yy - y) / reach, 0.0, 1.0)
        k = t * t * (3.0 - 2.0 * t)
        cand = float(base) + delta * k
        off = np.maximum(off, cand) if delta > 0 else np.minimum(off, cand)
    return off


def tip_offsets(core: np.ndarray, pts: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    """The points with their whisker turned into a signed offset (2026-11-15, the C1 whisker turns all round): the
    signed distance of the whisker's TIP from the objects' outline (> 0 outside, < 0 inside) – a whisker along the
    normal gives its length, one pointed into the object a cut. Normal = the gradient of the signed distance at the
    point, tangent = the normal turned +90° (static/bandpoints.js wdir). Items: (x, y, offset, in[, ang])."""
    c = np.asarray(core).astype(np.uint8)
    if not pts or not c.any():
        return list(pts or [])
    d_in = cv2.distanceTransform(np.pad(c, 1, mode="edge"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    d_out = cv2.distanceTransform(1 - c, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sd = (d_out - d_in).astype(np.float32)
    sm = cv2.GaussianBlur(sd, (0, 0), 3.0)
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    h, w = c.shape
    out: List[Tuple[float, ...]] = []
    for p in pts:
        x, y, ln = float(p[0]), float(p[1]), float(p[2])
        a = float(p[4]) if len(p) > 4 else 0.0
        xi, yi = int(min(w - 1, max(0, round(x)))), int(min(h - 1, max(0, round(y))))
        nx, ny = float(gx[yi, xi]), float(gy[yi, xi])
        nn = (nx * nx + ny * ny) ** 0.5
        nx, ny = (nx / nn, ny / nn) if nn > 1e-6 else (1.0, 0.0)
        tx, ty = -ny, nx
        r = np.deg2rad(a)
        dx, dy = nx * np.cos(r) + tx * np.sin(r), ny * np.cos(r) + ty * np.sin(r)
        tipx = float(min(w - 1, max(0.0, x + dx * ln))); tipy = float(min(h - 1, max(0.0, y + dy * ln)))
        off = float(cv2.getRectSubPix(sd, (1, 1), (tipx, tipy))[0, 0])
        out.append((x, y, off) + tuple(p[3:]))
    return out


def offset_mask(core: np.ndarray, off: Any) -> np.ndarray:
    """bool: the objects (`core`) moved by the signed offset (scalar or map): inside pixels kept where their depth >
    −off, outside pixels added where their distance <= off. The frame edge does not erode (padded with the core)."""
    c = np.asarray(core).astype(np.uint8)
    if not c.any():
        return c.astype(bool)
    o = np.asarray(off, np.float32)
    if o.ndim == 0:
        d_in = cv2.distanceTransform(np.pad(c, 1, mode="edge"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        d_out = cv2.distanceTransform(1 - c, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return np.where(c > 0, d_in > -float(o), d_out <= float(o))
    # a map: every pixel takes the offset of ITS NEAREST OUTLINE PIXEL (as bandmap.width_maps), so a point's whisker is
    # measured along the normal from the outline, not shortened by the pixel's own distance from the point
    d_in, lab_in = cv2.distanceTransformWithLabels(np.pad(c, 1, mode="edge"), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    d_in, lab_in = d_in[1:-1, 1:-1], lab_in[1:-1, 1:-1]
    d_out, lab_out = cv2.distanceTransformWithLabels(1 - c, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    op = np.pad(o, 1, mode="edge")
    zin = np.argwhere(np.pad(c, 1, mode="edge") == 0)       # label k = the k-th zero pixel in scan order
    zout = np.argwhere(c > 0)                               # the zero pixels of 1 - c = the core
    o_in = op[zin[lab_in - 1, 0], zin[lab_in - 1, 1]] if len(zin) else np.full(c.shape, np.inf, np.float32)
    o_out = o[zout[lab_out - 1, 0], zout[lab_out - 1, 1]]
    return np.where(c > 0, d_in > -o_in, d_out <= o_out)


def is_whole(in_w: Any) -> bool:
    """The inset width means „cały rdzeń” (an infinite scalar)."""
    return np.isscalar(in_w) and bool(np.isinf(float(in_w)))


FAR = np.float32(1e6)   # a distance "never reached" (outside the core in the depth deficit)


def deficit_reach(depth: float) -> int:
    """The window of the deepest point (px): at least twice the inset feather, so the middle is within reach wherever
    the feather keeps core."""
    return max(16, int(np.ceil(2.0 * max(0.0, float(np.max(depth))))) + 2)


def depth_deficit(core: np.ndarray, depth: float = 0.0) -> np.ndarray:
    """„Cały rdzeń”: px from each core pixel to the middle of the core, as a depth deficit – the deepest core point
    within `deficit_reach(depth)` (square window) minus this pixel's depth (exact Euclidean distance to the
    background), and at least `depth` + 1 minus the pixel's depth (the inset feather never reaches the edge, a strand
    thinner than the feather keeps only part of its core); 0 on the ridge of depth (the middle) where it is deeper than
    `depth`, FAR outside the core. Deterministic, OpenCV only."""
    c = np.asarray(core).astype(bool)
    if not c.any():
        return np.full(c.shape, FAR, np.float32)
    d = cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    r = deficit_reach(depth)
    m = cv2.dilate(d, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    dep = float(depth) if np.isscalar(depth) else np.asarray(depth, np.float32)
    return np.where(c, np.maximum(m - d, dep + 1.0 - d), FAR).astype(np.float32)   # edge pixel: depth 1


def whole_feather(core: np.ndarray, inner_px: float, pts=None, in_unit: float = 0.0) -> Width:
    """„Cały rdzeń” (user decision 2026-10-29): the inset feather – how much core is blended into the model result –
    shaped by the band points: each point's inner (blue) whisker over the inset slider (`in_unit`, the unit the points'
    widths are scaled in: in_px) multiplies the slider's feather along the outline (interpolated as the band widths,
    engine/bandmap.width_maps; 0 = the model alone there). No points (or no unit) = the slider's feather (a scalar)."""
    if not pts or float(in_unit) <= 0:
        return float(inner_px)
    from engine.bandmap import width_maps
    _, ratio = width_maps(np.asarray(core) > 0.5, list(pts), 0.0, float(in_unit))
    return (max(0.0, float(inner_px)) * ratio / float(in_unit)).astype(np.float32)


def core_middle(core: np.ndarray, depth: float = 0.0) -> np.ndarray:
    """The middle of the core (the ridge of depth, `depth_deficit` 0) – the inner band edge of „cały rdzeń” and the sure
    foreground seed of a ViTMatte trimap whose band covers the whole core."""
    return depth_deficit(core, depth) <= 0.0


def band_area(d: np.ndarray, out_w: Width, in_w: Width) -> np.ndarray:
    """The band mask from the signed distance `d` to the core edge (> 0 inside, engine/segment.sdf) and the widths
    (scalars or maps): background within `out` of the edge, core within `in` of it. The one band geometry – the
    former edge.adaptive_band (|d| <= map), edge.band_px (max(map, px) per side) and bandmap.variable_band
    (two one-sided distance transforms) are this, bit for bit."""
    d = np.asarray(d, np.float32)
    o = np.asarray(out_w, np.float32)
    i = np.asarray(in_w, np.float32)
    return (d >= -o) & (d <= i)


# ----------------------------------------------------------------------------- distances to the band edges
def area_dists(area: np.ndarray, core: np.ndarray, whole: bool = False, depth: float = 0.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d_in, d_out, solid) of an arbitrary band `area` over `core`: px to the inner band edge = to the solid core
    outside the area, px to the outer band edge = to the background outside the area (float32 distance transforms,
    1 on the first band pixel), and the solid side itself (the core without the area). `whole` („cały rdzeń”): the
    area covers the core, d_in = the depth deficit to the middle of the core (`depth_deficit`, reach from `depth` = the
    inset feather), the solid side = that middle."""
    a = np.asarray(area).astype(bool)
    c = np.asarray(core).astype(bool)
    if whole:
        a = a | c
        d_in = depth_deficit(c, depth)
        solid = d_in <= 0.0
        d_out = cv2.distanceTransform((~(~c & ~a)).astype(np.uint8), cv2.DIST_L2, 5)
        # one feather inside the core (user decision 2026-10-29): the outer feather acts on the outer band only – inside
        # the core it never keeps core at the edge (an outer band of 0 px = no outer feather at all)
        d_out = np.where(c, FAR, d_out).astype(np.float32)
        return d_in, d_out, solid
    solid = c & ~a
    d_in = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~(~c & ~a)).astype(np.uint8), cv2.DIST_L2, 5)
    return d_in, d_out, solid




# ----------------------------------------------------------------------------- the rule
def lengths(width: Width, inner_px: float, outer_px: float) -> Tuple[Width, Width]:
    """(li, lo): the local length of the inset and the outer feather – the inset one keeps its length up to the band
    width, the outer one gets what is left (they never overlap, the inset one has priority; user rule 2026-10-22)."""
    fi = max(0.0, float(inner_px)) if np.isscalar(inner_px) else np.maximum(0.0, np.asarray(inner_px, np.float32))
    fo = max(0.0, float(outer_px))
    li = np.minimum(fi, width)
    lo = np.minimum(fo, np.maximum(0.0, width - li))
    return li, lo


def weight(d_in: np.ndarray, d_out: np.ndarray, li: Width, lo: Width) -> np.ndarray:
    """w = min(ramp over the inset feather from the inner edge, ramp over the outer feather from the outer edge);
    a 0 px feather = 1 on its side (a hard edge). Unmasked – the caller limits it to the band."""
    w_in = np.where(li > 0, np.clip(d_in / np.maximum(li, 1e-6), 0.0, 1.0), 1.0)
    w_out = np.where(lo > 0, np.clip(d_out / np.maximum(lo, 1e-6), 0.0, 1.0), 1.0)
    return np.minimum(w_in, w_out)


def blend(area: np.ndarray, w: np.ndarray, model: np.ndarray, core: np.ndarray,
          outside: Optional[np.ndarray] = None) -> np.ndarray:
    """alpha = w · model + (1 − w) · core inside the band, `outside` (default: the core) elsewhere; float32.
    `model` and `core` are 0..1 (the caller clips when it has to)."""
    a = np.asarray(area).astype(bool)
    c = np.asarray(core, np.float32)
    m = np.asarray(model, np.float32)
    return np.where(a, w * m + (1.0 - w) * c, c if outside is None else outside).astype(np.float32)


def end_masks(area: np.ndarray, solid: np.ndarray, d_in: np.ndarray, d_out: np.ndarray, li: Width, lo: Width,
              inner_px: float, outer_px: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Masks whose outlines are the feather-end lines: the solid side up to where the inset feather ends, the solid
    side up to where the outer feather ends (None when that feather is 0 px – no line)."""
    a = np.asarray(area).astype(bool)
    inner = (solid | (a & (d_in < li))) if float(np.max(inner_px)) > 0 else None
    outer = (solid | (a & (d_out >= lo))) if float(outer_px) > 0 else None
    return inner, outer


# ----------------------------------------------------------------------------- the band from widths (stages 4 / 7)
# One way to measure (user decision 2026-10-24, „ten sam pas wszędzie”): the distances to the band edges are ALWAYS
# measured from the band area (area_dists) – the stages that size their band from widths (4 VideoMaMa, 7 LTX-2.5) build
# the area with band_area and go through the same functions as stage 3 and the stage-5 zones. (Until 2026-10-24 they
# used analytic distances i − d / d + o: the same rule, ~1 px apart on the ramps, so the lines differed between stages.)
def band_weight(d: np.ndarray, out_w: Width, in_w: Width, inner_px: float, outer_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """(area, weight) of the band sized by widths around the core whose signed distance is `d` (> 0 inside)."""
    area = band_area(d, out_w, in_w)
    return area, area_weight(area, np.asarray(d) > 0, inner_px, outer_px, whole=is_whole(in_w))


def compose(core: np.ndarray, model: np.ndarray, d: np.ndarray, out_w: Width, in_w: Width, inner_px: float,
            outer_px: float) -> np.ndarray:
    """The stage alpha: inside the band w · model + (1 − w) · core, outside the core (both 0..1, same shape; `core`
    may be soft – a stage-3 result as the VideoMaMa base – the distances are measured on its edge, d > 0)."""
    c = np.clip(np.asarray(core, np.float32), 0, 1)
    m = np.clip(np.asarray(model, np.float32), 0, 1)
    return compose_area(np.asarray(d) > 0, m, band_area(d, out_w, in_w), inner_px, outer_px, outside=c, partner=c,
                        whole=is_whole(in_w))


def feather_lines(d: np.ndarray, out_w: Width, in_w: Width, inner_px: float, outer_px: float):
    """Masks whose outlines are the preview lines: (band, inset feather end, outer feather end) – the same as
    area_lines of the band area (None when that feather is 0)."""
    area = band_area(d, out_w, in_w)
    d_in, d_out, solid = area_dists(area, np.asarray(d) > 0, whole=is_whole(in_w), depth=inner_px)
    li, lo = lengths(d_in + d_out, inner_px, outer_px)
    inner, outer = end_masks(area, solid, d_in, d_out, li, lo, inner_px, outer_px)
    return area, inner, outer


# ----------------------------------------------------------------------------- band from an area (stage 3, stage-5 zones)
def area_weight(area: np.ndarray, core: np.ndarray, inner_px: float, outer_px: float, whole: bool = False) -> np.ndarray:
    """Blend weight (0..1) of a model over the core inside an arbitrary band `area`: the rule with the distances measured
    from the area (the former zones.feather_weight; user report 2026-10-22: a 51 px outer feather on a 17 px band ran
    across the band into the core – the zone never reached full weight)."""
    d_in, d_out, _ = area_dists(area, core, whole, inner_px)
    li, lo = lengths(d_in + d_out, inner_px, outer_px)
    a = np.asarray(area).astype(bool) | (np.asarray(core).astype(bool) if whole else False)
    return np.where(a, weight(d_in, d_out, li, lo), 0.0).astype(np.float32)


def compose_area(core: np.ndarray, model: np.ndarray, area: np.ndarray, inner_px: float, outer_px: float,
                 outside: Optional[np.ndarray] = None, partner: Optional[np.ndarray] = None, whole: bool = False) -> np.ndarray:
    """The alpha of a stage whose band is an area: w · model + (1 − w) · partner inside the band (`partner` = what the
    feathers blend into, default the core – a whole-outline zone blends into max(core, stage base)), `outside`
    (default the core) elsewhere. The distances are measured against `core` (the hard base) in every case. `whole`
    („cały rdzeń”): the band covers the whole core, the inner edge = its middle (`depth_deficit`)."""
    d_in, d_out, _ = area_dists(area, core, whole, inner_px)
    li, lo = lengths(d_in + d_out, inner_px, outer_px)
    w = weight(d_in, d_out, li, lo)
    if whole:
        area = np.asarray(area).astype(bool) | np.asarray(core).astype(bool)
    return blend(area, w, model, core if partner is None else partner, outside)


CROP_ALIGN = 64      # compose_area_crop: crop start / size multiple (bit-identical distance transforms)


def compose_area_crop(core: np.ndarray, model: np.ndarray, area: np.ndarray, inner_px: float, outer_px: float,
                      outside: Optional[np.ndarray] = None, partner: Optional[np.ndarray] = None, whole: bool = False) -> np.ndarray:
    """`compose_area` computed only on the band's bbox + a margin (plan docs/PLAN_EDGE_SPEED.md A6b, 2026-10-28; the
    band is a few % of a 4K frame) – bit-identical: outside the band the result is `outside` anyway, and inside it a
    distance only matters up to inner + outer feather (beyond, both ramps are 1 and the lengths are the full feathers);
    any edge pixel left out of the crop lies >= the margin away, so the chamfer distances that matter are the same.
    The crop starts on multiples of CROP_ALIGN px and spans multiples of it: OpenCV's distanceTransform (IPP / SIMD)
    gives 1-ulp different distances on a crop starting at an unaligned column (measured 2026-10-28: start % 4 != 0),
    and the feather lengths use the raw distances. Pinned by tests/test_compose_area_crop.py against `compose_area`."""
    a = np.asarray(area).astype(bool)
    if whole:   # the band covers the core: the crop holds all of it (its depth deficit is the same in the crop)
        a = a | np.asarray(core).astype(bool)
    rows, cols = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    if not len(rows):
        return compose_area(core, model, area, inner_px, outer_px, outside=outside, partner=partner, whole=whole)
    H, W = a.shape
    # chamfer DIST_L2 / mask 5 is within ~2 % of the Euclidean distance: 10 % + 8 px of headroom
    m = int(np.ceil(1.1 * (max(0.0, float(np.max(inner_px))) + max(0.0, float(outer_px))))) + 8

    def span(lo: int, hi: int, n: int) -> slice:
        s0 = max(0, lo - m) // CROP_ALIGN * CROP_ALIGN
        s1 = s0 + -(-(min(n, hi + 1 + m) - s0) // CROP_ALIGN) * CROP_ALIGN
        return slice(s0, min(n, s1))
    sl = (span(int(rows[0]), int(rows[-1]), H), span(int(cols[0]), int(cols[-1]), W))
    out = np.array(core if outside is None else outside, dtype=np.float32, copy=True)
    out[sl] = compose_area(np.asarray(core)[sl], np.asarray(model)[sl], a[sl], inner_px, outer_px,
                           outside=None if outside is None else np.asarray(outside)[sl],
                           partner=None if partner is None else np.asarray(partner)[sl], whole=whole)
    return out


def area_lines(area: np.ndarray, core: np.ndarray, inner_px: float, outer_px: float, whole: bool = False) -> Dict[str, Any]:
    """The preview lines of a band that is an area (plate px): band edges + where the feathers end – the same
    distances and lengths as `compose_area`, so the lines show what the composition does."""
    d_in, d_out, solid = area_dists(area, core, whole, inner_px)
    if whole:
        area = np.asarray(area).astype(bool) | np.asarray(core).astype(bool)
    li, lo = lengths(d_in + d_out, inner_px, outer_px)
    inner, outer = end_masks(area, solid, d_in, d_out, li, lo, inner_px, outer_px)
    return preview_lines(area, inner, outer)


def band_lines(mask: np.ndarray, out_px: float, in_px: float, inner_px: float, outer_px: float, pts=None,
               unit: float = 0.0, crop: bool = True) -> Dict[str, Any]:
    """The preview lines of a stage band (GET …/stage_band/{stage}/{f}/lines): band widths from the sliders + points,
    the feather ends – `preview_lines(feather_lines(sdf(mask), …))`. `unit` = the inset slider for „cały rdzeń” (the
    points' inner whiskers shape the inset feather). `crop` (2026-11-20, SD160_020: ~0.8 s a 6K frame for a band
    covering 1.9 % of it): computed on the mask's bbox + the reach of the band and the feathers, starting / spanning
    multiples of CROP_ALIGN (as compose_area_crop) – the same lines; tests/test_band_lines_crop.py pins it."""
    from engine.ltx_alpha import sdf_px
    m = np.asarray(mask) > 0.5
    pts = list(pts or []) or None
    o, i = band_widths(m, out_px, in_px, pts)
    fi: Width = inner_px
    if is_whole(i) and unit:
        fi = whole_feather(m, inner_px, pts, unit)
    # outside the mask's bbox only the outer band and the feathers reach (the inner band is inside the mask); the
    # depth deficit („cały rdzeń”) looks deficit_reach px around; chamfer distances within ~2 %: 10 % + 8 px headroom
    reach = float(np.max(o)) + max(0.0, float(np.max(fi))) + max(0.0, float(outer_px))
    sl = crop_slices(m, int(np.ceil(1.1 * reach)) + deficit_reach(float(np.max(fi))) + 8) if crop else None
    if sl is None:
        return preview_lines(*feather_lines(sdf_px(m.astype(np.float32)), o, i, fi, outer_px))
    cut = (lambda w: w if np.isscalar(w) else np.asarray(w)[sl])
    return shift_lines(preview_lines(*feather_lines(sdf_px(m[sl].astype(np.float32)), cut(o), cut(i), cut(fi), outer_px)), sl)


def crop_slices(mask: np.ndarray, margin: int) -> Optional[Tuple[slice, slice]]:
    """(rows, cols) of the mask's bbox grown by `margin` px, starting on multiples of CROP_ALIGN and spanning multiples
    of it (OpenCV's distance transforms / morphology give 1-ulp different results on unaligned crops – see
    compose_area_crop); None for an empty mask. A band computed on it equals the whole-frame one when nothing it reads
    reaches further than `margin` from the mask."""
    m = np.asarray(mask).astype(bool)
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if not len(rows):
        return None
    H, W = m.shape
    mg = max(0, int(margin))

    def span(lo: int, hi: int, n: int) -> slice:
        s0 = max(0, lo - mg) // CROP_ALIGN * CROP_ALIGN
        s1 = s0 + -(-(min(n, hi + 1 + mg) - s0) // CROP_ALIGN) * CROP_ALIGN
        return slice(s0, min(n, s1))
    return span(int(rows[0]), int(rows[-1]), H), span(int(cols[0]), int(cols[-1]), W)


def shift_lines(lines: Dict[str, Any], sl: Tuple[slice, slice]) -> Dict[str, Any]:
    """Polylines computed on a crop moved back to frame px."""
    x0, y0 = sl[1].start, sl[0].start
    return {k: [[[q[0] + x0, q[1] + y0] for q in c] for c in v] for k, v in lines.items()}


def area_lines_crop(area: np.ndarray, core: np.ndarray, inner_px: Width, outer_px: float, whole: bool = False) -> Dict[str, Any]:
    """`area_lines` (a stage-5 zone's band lines) computed on the bbox of area ∪ core + the feathers' reach (2026-11-20:
    0.46 s on a 6K frame) – the same lines; tests/test_band_lines_crop.py pins it. `inner_px` may be a map (the
    „cały rdzeń” feather shaped by the band points)."""
    a = np.asarray(area).astype(bool)
    c = np.asarray(core).astype(bool)
    fi = max(0.0, float(np.max(inner_px)))
    sl = crop_slices(a | c, int(np.ceil(1.1 * (fi + max(0.0, float(outer_px))))) + deficit_reach(fi) + 8)
    if sl is None:
        return area_lines(a, c, inner_px, outer_px, whole=whole)
    ip = inner_px if np.isscalar(inner_px) else np.asarray(inner_px)[sl]
    return shift_lines(area_lines(a[sl], c[sl], ip, outer_px, whole=whole), sl)


# ----------------------------------------------------------------------------- preview lines
def contours(mask: Optional[np.ndarray], mode: int = cv2.RETR_EXTERNAL) -> List[List[List[int]]]:
    """Polylines (px) of a mask's outlines, simplified to 1.5 px; [] for None / empty."""
    if mask is None:
        return []
    m = np.asarray(mask)
    if not m.any():
        return []
    cnts, _ = cv2.findContours(m.astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in cnts if len(c) >= 3]


def preview_lines(band: np.ndarray, inner_end: Optional[np.ndarray], outer_end: Optional[np.ndarray]) -> Dict[str, Any]:
    """The lines dict of every band preview (GET …/stage_band/{stage}/{f}/lines, a zone's …/region/{f}):
    `band_contours` = the band ring (outer and inner edge, CCOMP), `inset_feather_contours` / `feather_contours` = where
    the inset / the outer feather ends ([] without that feather)."""
    return {"band_contours": contours(band, cv2.RETR_CCOMP),
            "inset_feather_contours": contours(inner_end, cv2.RETR_EXTERNAL),
            "feather_contours": contours(outer_end, cv2.RETR_EXTERNAL)}
