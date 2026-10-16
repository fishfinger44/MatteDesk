# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/correct_region.py - Z21: a local correction (stage 3 click / brush = key `correction`) changes the core
ONLY in its region, also in the neighbouring frames of the recomputed intervals (user request 2026-10-12).

- `save_delta` / `load_delta`: the changed region of a correction key accumulates in keys/k_####_delta.png (union of
  every correction made on that frame); the key JSON keeps `corrections[]`.
- `region_from_delta`: region = delta dilated by clamp(0.5 x its radius, 16..48 px) – room for the tracking error.
- `track_region`: control points on the border and inside the region at the key frame -> CoTracker3 (any
  `tracker.track(frames, queries)` of engine/tracking) to every frame of the interval -> per frame the convex hull of
  the visible points + margin; fewer than 3 visible points -> the region shifted by the mean displacement (or static).
- `merge_region`: core = old core outside the region, new core inside, SDF blend over `feather` px at the region border
  (no seam). Pixels outside the (feathered) region are bit-identical to the old core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REGION_ALGO_VERSION = 2     # bump when the region geometry / tracking changes (part of the interval cache hash)
MIN_MARGIN_PX = 16
MARGIN_FACTOR = 0.5       # margin = clamp(factor x radius of the delta, MIN_MARGIN_PX .. MAX_MARGIN_PX)
MAX_MARGIN_PX = 48
FEATHER_PX = 8


def delta_path(obj_dir: Path, frame: int) -> Path:
    return Path(obj_dir) / "keys" / f"k_{int(frame):04d}_delta.png"


def save_delta(obj_dir: Path, frame: int, changed: np.ndarray) -> Path:
    """Union of the changed region with the delta already saved for this key frame."""
    p = delta_path(obj_dir, frame)
    m = changed > 0.5
    old = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    if old is not None and old.shape == m.shape:
        m = m | (old > 127)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), m.astype(np.uint8) * 255)
    return p


def load_delta(obj_dir: Path, frame: int) -> Optional[np.ndarray]:
    p = delta_path(obj_dir, frame)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return None if m is None else (m > 127)


def load_deltas(obj_dir: Path, keys: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Delta masks of every `correction` key that has one (frame -> bool mask)."""
    out = {}
    for f, k in keys.items():
        if k.get("kind") == "correction":
            d = load_delta(obj_dir, f)
            if d is not None and d.any():
                out[int(f)] = d
    return out


def region_from_delta(delta: np.ndarray, min_margin: int = MIN_MARGIN_PX, factor: float = MARGIN_FACTOR) -> np.ndarray:
    m = delta > 0.5
    if not m.any():
        return m
    r = float(np.sqrt(m.sum() / np.pi))
    margin = int(round(min(MAX_MARGIN_PX, max(min_margin, factor * r))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def control_points(region: np.ndarray, n_inside: int = 12, n_border: int = 16) -> np.ndarray:
    """(N, 2) points: distance-transform maxima inside + evenly spaced contour samples (plate px)."""
    from engine.correct import seed_points
    pts: List[List[float]] = seed_points(region, n=n_inside)
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=len).reshape(-1, 2).astype(np.float32)
        step = max(1, len(c) // n_border)
        pts += [[float(x), float(y)] for x, y in c[::step][:n_border]]
    return np.asarray(pts, np.float32).reshape(-1, 2)


def region_from_points(pts: np.ndarray, shape: Tuple[int, int], margin: int) -> np.ndarray:
    hull = cv2.convexHull(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    if margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def track_region(tracker: Any, frames_small: Sequence[np.ndarray], scale: float, frame_ids: Sequence[int], key_frame: int,
                 region: np.ndarray, margin: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Region of the correction in every frame of `frame_ids` (the key frame keeps `region`). `frames_small` =
    the display frames downscaled by `scale` (tracker input), aligned with `frame_ids`."""
    frame_ids = list(frame_ids)
    H, W = region.shape
    out = {key_frame: region.copy()}
    if margin is None:
        r = float(np.sqrt(max(1.0, region.sum()) / np.pi))
        margin = int(round(min(32, max(12, 0.2 * r))))      # room for the tracking error of the hull
    pts = control_points(region)
    info: Dict[str, Any] = {"points": int(len(pts)), "fallback_frames": 0, "hull_frames": 0}
    if len(frame_ids) <= 1 or tracker is None or len(pts) < 3:
        for f in frame_ids:
            out.setdefault(f, region.copy())
        info["fallback_frames"] = len(frame_ids) - 1
        return out, info
    t0 = frame_ids.index(key_frame)
    q = np.column_stack([np.full(len(pts), t0, np.float32), pts * scale]).astype(np.float32)
    res = tracker.track(list(frames_small), q)
    for i, f in enumerate(frame_ids):
        if f == key_frame:
            continue
        vis = res.visible[i]
        if vis.sum() >= 3:
            out[f] = region_from_points(res.tracks[i][vis] / scale, (H, W), margin) | False
            info["hull_frames"] += 1
        else:
            info["fallback_frames"] += 1
            if vis.any():
                d = (res.tracks[i][vis] / scale - pts[vis]).mean(axis=0)
                M = np.array([[1, 0, d[0]], [0, 1, d[1]]], np.float32)
                out[f] = cv2.warpAffine(region.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST).astype(bool)
            else:
                out[f] = region.copy()
    return out, info


def _sdf(m: np.ndarray) -> np.ndarray:
    b = (m > 0.5).astype(np.uint8)
    return cv2.distanceTransform(b, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)


def merge_region(old: np.ndarray, new: np.ndarray, region: np.ndarray, feather: int = FEATHER_PX) -> np.ndarray:
    """old outside the region, new inside; SDF blend over `feather` px INSIDE the region border (outside = old,
    bit-identical)."""
    reg = region > 0.5
    if not reg.any():
        return old > 0.5
    w = np.clip(cv2.distanceTransform(reg.astype(np.uint8), cv2.DIST_L2, 5) / float(max(1, feather)), 0.0, 1.0)
    inner = w >= 1.0
    out = (old > 0.5).copy()
    out[inner] = (new > 0.5)[inner]
    band = reg & ~inner
    if band.any():
        s = w * _sdf(new) + (1.0 - w) * _sdf(old)
        out[band] = s[band] > 0
    return out
