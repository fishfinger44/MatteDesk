# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Linked anchors made from the main one by moving its patch (user decision 2026-11-11): Qwen-Image-2.1 with the main
anchor as Picture 2 copies only the STYLE (another filigree layout on every linked frame); here a grid of points over
the main anchor's patch is tracked by CoTracker3 (LK when it is missing) from the main frame to the linked one, the
main generation is warped to the new pose (piecewise linear over the tracked points, nearest outside their hull) and
pasted inside the linked frame's patch. Then – method "warp_qwen" – Qwen-Image-2.1 only touches it up (low denoise:
light and seams of this frame, the pattern stays)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

METHODS = ("warp_qwen", "warp", "qwen")   # linked anchor: warp + Qwen touch-up / warp only / Qwen with Picture 2
DEFAULT_METHOD = "qwen"   # user 2026-11-11: the warp deformed the neck / edges on a big pose change
DEFAULT_DENOISE = 0.4                     # Qwen touch-up strength (KSampler denoise from the warped image's latent)
GRID_MAX = 900                            # tracked points per anchor (CoTracker3 queries)
REACH_PX = 0                              # points only ON the patch (2026-11-11: a ring around it caught hair moving
                                          # unlike the face – the temples were dragged with it)
INSET_PX = 4                              # ... and this far inside its edge (an edge point sits half on the hair)
STRETCH_MIN, STRETCH_MAX = 0.35, 3.0      # warp area scale outside this = torn / squashed -> repainted (hole)
HOLE_GROW_PX = 6                          # holes grown a little: the repaint covers the seam
MAP_STEP = 4                              # the dense map is interpolated every MAP_STEP px, then scaled up


def method(s: Dict[str, Any]) -> str:
    m = s.get("link_method")
    return m if m in METHODS else DEFAULT_METHOD


def denoise(s: Dict[str, Any]) -> float:
    try:
        v = float(s.get("link_denoise"))
    except (TypeError, ValueError):
        return DEFAULT_DENOISE
    return float(min(1.0, max(0.05, v)))


def query_points(mask: np.ndarray, max_n: int = GRID_MAX, reach_px: int = REACH_PX) -> np.ndarray:
    """(N, 2) x, y of a regular grid over the patch (mask > 0.02) grown by reach_px."""
    m = (np.asarray(mask, np.float32) > 0.02).astype(np.uint8)
    if not m.any():
        return np.zeros((0, 2), np.float32)
    if reach_px <= 0 and INSET_PX > 0:
        e = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INSET_PX + 1, 2 * INSET_PX + 1)))
        if e.sum() >= 0.25 * m.sum():
            m = e
    if reach_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach_px + 1, 2 * reach_px + 1)))
    area = float(m.sum())
    step = max(4, int(np.ceil(np.sqrt(area / float(max_n)))))
    ys, xs = np.mgrid[step // 2:m.shape[0]:step, step // 2:m.shape[1]:step]
    keep = m[ys, xs] > 0
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)


def track(tracker: Any, frames: Sequence[int], image: Callable[[int], np.ndarray], src: int, dst: Sequence[int],
          pts: np.ndarray, cancel_check: Optional[Callable[[], None]] = None
          ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Positions (N, 2) + visibility (N,) of `pts` (set in frame `src`) in every frame of `dst`; ONE tracker call over
    the frames from the earliest to the latest of src + dst (CoTracker3 offline tracks both ways from the query)."""
    want = [int(f) for f in dst]
    lo, hi = min([int(src)] + want), max([int(src)] + want)
    span = [int(f) for f in frames if lo <= int(f) <= hi]
    if int(src) not in span:
        raise ValueError(f"Klatka główna F{src} poza zakresem klatek")
    imgs = [image(f) for f in span]
    t0 = span.index(int(src))
    q = np.concatenate([np.full((len(pts), 1), t0, np.float32), pts.astype(np.float32)], axis=1)
    if cancel_check is not None and getattr(tracker, "engine", "") == "cotracker3":
        res = tracker.track(imgs, q, cancel_check)
    else:
        res = tracker.track(imgs, q)
    out = {}
    for f in want:
        if f in span:
            i = span.index(f)
            out[f] = (np.asarray(res.tracks[i], np.float32), np.asarray(res.visible[i], bool))
    return out


def warp_map(src_pts: np.ndarray, dst_pts: np.ndarray, w: int, h: int, step: int = MAP_STEP
             ) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps (x, y float32, h × w): for every pixel of the linked frame, where it is in the main frame –
    linear over the Delaunay triangles of the tracked points, nearest point's shift outside their hull."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    gx, gy = np.meshgrid(np.arange(0, w + step, step, dtype=np.float32), np.arange(0, h + step, step, dtype=np.float32))
    shift = (src_pts - dst_pts).astype(np.float64)
    if len(dst_pts) >= 3:
        try:
            v = LinearNDInterpolator(dst_pts.astype(np.float64), shift)(gx, gy)
        except Exception:   # degenerate (collinear) points: the nearest shift everywhere
            v = np.full(gx.shape + (2,), np.nan)
    else:
        v = np.full(gx.shape + (2,), np.nan)
    hole = np.isnan(v[..., 0])
    if hole.any():
        v[hole] = NearestNDInterpolator(dst_pts.astype(np.float64), shift)(gx[hole], gy[hole])
    sx = cv2.resize(v[..., 0].astype(np.float32), (gx.shape[1] * step, gx.shape[0] * step), interpolation=cv2.INTER_LINEAR)[:h, :w]
    sy = cv2.resize(v[..., 1].astype(np.float32), (gx.shape[1] * step, gx.shape[0] * step), interpolation=cv2.INTER_LINEAR)[:h, :w]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return xx + sx, yy + sy


def warp_anchor(main_img: np.ndarray, plate8: np.ndarray, mask: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                visible: np.ndarray, main_mask: Optional[np.ndarray] = None, holes: bool = False):
    """uint8 BGR: the linked frame's plate with the main generation, warped to this frame, pasted inside its patch
    (mask > 0.02, hard – like make_anchor). Only points visible in the linked frame steer the warp.
    holes=True (2026-11-11) -> (image, hole): `hole` = the part of the patch the warp cannot fill honestly – not under
    the main patch (main_mask warped along; hair over the face in the main frame, a side turned away) or torn /
    squashed (area scale outside STRETCH_MIN..MAX); there the image holds the surrounding pattern (cv2.inpaint) for
    the repaint, not stretched pixels."""
    h, w = plate8.shape[:2]
    vis = np.asarray(visible, bool)
    if vis.sum() < 3:
        vis = np.ones(len(src_pts), bool)
    if len(src_pts) == 0:
        raise ValueError("Brak punktów do przeniesienia wzoru (pusta łata kotwicy głównej)")
    mx, my = warp_map(src_pts[vis], dst_pts[vis], w, h)
    src = main_img if main_img.shape[:2] == (h, w) else cv2.resize(main_img, (w, h), interpolation=cv2.INTER_AREA)
    warped = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m2 = np.asarray(mask, np.float32) > 0.02
    if not holes:
        return np.where(m2[..., None], warped, plate8).astype(np.uint8)
    hole = np.zeros((h, w), bool)
    if main_mask is not None:
        mm = np.asarray(main_mask, np.float32)
        mm = mm if mm.shape == (h, w) else cv2.resize(mm, (w, h), interpolation=cv2.INTER_LINEAR)
        hole |= cv2.remap((mm > 0.02).astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) < 0.5
    hole |= ~stretch_ok(mx, my)
    hole &= m2
    if hole.any() and HOLE_GROW_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HOLE_GROW_PX + 1, 2 * HOLE_GROW_PX + 1))
        hole = (cv2.dilate(hole.astype(np.uint8), k) > 0) & m2
    out = np.where(m2[..., None], warped, plate8).astype(np.uint8)
    if hole.any():
        out = fill_holes(out, hole, m2)
    return out, hole


def stretch_ok(mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Where the warp keeps the area scale within STRETCH_MIN..MAX (|det| of the map's Jacobian, smoothed)."""
    gxx, gxy = np.gradient(mx, axis=1), np.gradient(mx, axis=0)
    gyx, gyy = np.gradient(my, axis=1), np.gradient(my, axis=0)
    det = cv2.GaussianBlur(np.abs(gxx * gyy - gxy * gyx).astype(np.float32), (0, 0), 3)
    return (det >= STRETCH_MIN) & (det <= STRETCH_MAX)


def fill_holes(img: np.ndarray, hole: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """The holes filled from the warped pattern around them (cv2.inpaint at half size; only the patch's own pixels are
    a source – the plate outside it is the object being replaced)."""
    h, w = hole.shape
    s = 0.5 if max(h, w) > 1200 else 1.0
    sm = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1.0 else img.copy()
    src_bad = (~patch | hole).astype(np.uint8) * 255
    sb = cv2.resize(src_bad, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) if s != 1.0 else src_bad
    if (sb == 0).sum() < 16:
        return img
    filled = cv2.inpaint(sm, sb, 9, cv2.INPAINT_TELEA)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR) if s != 1.0 else filled
    return np.where(hole[..., None], big, img).astype(np.uint8)


def hole_prompt(description: str) -> str:
    """Qwen-Image-2.1 text of the hole repaint (Picture 1 = the warped anchor with patched gaps, Picture 2 = main)."""
    d = (description or "").strip().rstrip(".")
    what = f" ({d})" if d else ""
    return ("Picture 1 shows the design" + what + " moved from Picture 2 onto this pose, with some smeared, stretched "
            "or blank patches. Repaint it so the whole surface shows the same continuous pattern, ornaments and "
            "materials as Picture 2, fitted to the pose and light of Picture 1, with no smears, stretching or seams. "
            "Keep the face, eyes, lips, hair and everything else unchanged.")


def touchup_prompt(description: str) -> str:
    """Qwen-Image-2.1 text of the touch-up (Picture 1 = the warped anchor, Picture 2 = the main anchor)."""
    d = (description or "").strip().rstrip(".")
    what = f" ({d})" if d else ""
    return ("Picture 1 already shows the finished design" + what + ", moved from Picture 2 onto this pose. Keep the "
            "pattern, its layout, ornaments and materials exactly as they are in Picture 1. Only fix the lighting and "
            "shading to match the light of Picture 1, smooth any stretched or torn areas and seams, and keep "
            "everything outside the design unchanged.")


def linked_frames(anchors: Sequence[Dict[str, Any]]) -> List[int]:
    return sorted(int(a["frame"]) for a in anchors if a.get("kind") == "linked")
