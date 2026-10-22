# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/lupa.py - "Lupa" on thin parts (SPEC 5.2 pkt 5b).

When thin_score > 0.15 (or the user adds a lupa frame), a small extra square (256-512 plate px) around the thin
parts lets SAM 3.1 see them 2-4x larger. Prompt = positive points on the core skeleton inside the lupa + negatives
from the background ring; tracking with SAM3_VideoTrack in constant-lupa segments. The result is merged into the
core only inside the lupa area (dilate 16 px) with an SDF crossfade; it may only ADD thin parts connected to the
core (no new islands). Outside the lupa the core is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.crop import CropSegment, extract_crop, paste_crop, plan_segments, smooth_bboxes
from engine.segment import sdf

THIN_SCORE_THR = 0.15


def thin_parts(core: np.ndarray, r: int = 3) -> np.ndarray:
    """Pixels of the core that disappear under an opening with a (2r+1) disc = thin parts."""
    c = core.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return (c & ~cv2.morphologyEx(c, cv2.MORPH_OPEN, k).astype(bool)).astype(bool)


def thin_score(core: np.ndarray) -> float:
    c = core > 0.5
    return float(thin_parts(c).sum() / max(1, c.sum()))


def lupa_segments(cores: Dict[int, np.ndarray], frames: Sequence[int], plate_w: int, plate_h: int,
                  manual_box: Optional[Sequence[float]] = None) -> List[CropSegment]:
    """Constant-lupa segments (256-512 px squares) around the thin parts (or the user's lupa frame)."""
    bbs = []
    for f in frames:
        if manual_box is not None:
            bbs.append(tuple(manual_box))
            continue
        t = thin_parts(cores[f])
        t = cv2.dilate(t.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ys, xs = np.where(t)
        bbs.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) > 10 else None)
    if not any(b is not None for b in bbs):
        return []
    segs = plan_segments(smooth_bboxes(bbs), frames[0], plate_w, plate_h,
                         {"min_fill": 0.35, "pad_ratio": 1.25, "min_margin_px": 32, "snap_multiple": 16})
    out = []
    for s in segs:
        side = s.side
        if side > 512:  # thin parts spread too far for one lupa: keep the square at 512 around the centre
            cx, cy = s.crop[0] + side / 2, s.crop[1] + side / 2
            from engine.crop import square_crop
            s.crop, s.pad = square_crop(cx, cy, 512 / 1.1, plate_w, plate_h, 1.1, 0, 16)
        elif side < 256:
            cx, cy = s.crop[0] + side / 2, s.crop[1] + side / 2
            from engine.crop import square_crop
            s.crop, s.pad = square_crop(cx, cy, 256, plate_w, plate_h, 1.0, 0, 16)
        s.seg_id = "lupa_" + s.seg_id
        s.scale_to_1008 = round(1008 / s.side, 4)
        out.append(s)
    return out


def skeleton_points(mask: np.ndarray, n: int = 6) -> List[Tuple[float, float]]:
    """Up to n points on the skeleton (distance-transform ridge) of the mask."""
    m = mask.astype(np.uint8)
    if not m.any():
        return []
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist > 0)
    ys, xs = np.where(ridge)
    if len(xs) == 0:
        ys, xs = np.where(m)
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in idx]


def merge_lupa(core: np.ndarray, lupa: np.ndarray, area: np.ndarray, feather: float = 16.0) -> np.ndarray:
    """Add thin parts found by the lupa, connected to the core, only inside `area` (crossfade at its border)."""
    from engine.zones import connected_to
    add = (lupa > 0.5) & ~core & area
    add = connected_to(add, core, reach=10 ** 6)  # whole thin parts connected to the core (no reach limit)
    cand = core | add
    # SDF crossfade: full lupa result deep inside the area, plain core at its border
    w = np.clip(sdf(area) / feather, 0.0, 1.0)
    return ((1 - w) * sdf(core) + w * sdf(cand)) > 0


def run_lupa(runner: Any, cores: Dict[int, np.ndarray], frames: Sequence[int], plate_w: int, plate_h: int,
             load_crop: Callable[[int, CropSegment], np.ndarray], manual_box: Optional[Sequence[float]] = None
             ) -> Tuple[Dict[int, np.ndarray], List[CropSegment]]:
    """Track thin parts in lupa segments and merge them into the core."""
    frames = list(frames)
    segs = lupa_segments(cores, frames, plate_w, plate_h, manual_box)
    out = {f: cores[f].copy() for f in frames}
    for s in segs:
        sf = [f for f in frames if s.first <= f <= s.last]
        if not sf:
            continue
        c0 = extract_crop(cores[sf[0]].astype(np.float32), s) > 0.5
        img0 = load_crop(sf[0], s)
        pos = skeleton_points(thin_parts(c0) | c0, 8)
        ring = cv2.dilate(c0.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~cv2.dilate(
            c0.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        neg = skeleton_points(ring, 6)
        det = runner.detect(img0, positive=pos or None, negative=neg or None, individual=False)
        init = det[0] if det else c0.astype(np.float32)
        init = np.maximum(init, c0.astype(np.float32))
        res = runner.track([load_crop(f, s) for f in sf], init)
        area_crop = np.ones((s.side, s.side), np.float32)
        area = paste_crop(area_crop, s, plate_w, plate_h) > 0.5
        area = cv2.erode(area.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        for f, m in zip(sf, res):
            full = paste_crop(m, s, plate_w, plate_h)
            out[f] = merge_lupa(out[f], full, area)
    return out, segs
