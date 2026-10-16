# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/guard.py - Z22 stage A: SAM3 + CoTracker3 "guard" measurements (experiment, no change of results).

SAM3 gives the shape and the edge; CoTracker3 keeps the identity of the object in time. Points seeded deep inside
the key mask (>= `inset_px` from its edge) are tracked to every frame of the interval; per frame:
- lost_frac       = visible points that fall OUTSIDE the SAM3 mask / visible points  (SAM3 drifted off the object,
                    or the object moved out of the mask),
- unsupported_frac = area of SAM3 mask components with NO visible point within R / mask area  (mask parts that no
                    tracked point supports: a jump onto another object, a leak),
- occluded_frac   = 1 - visible / all points.
R = 2 x the mean point spacing (sqrt(mask area / N)). Pure functions; the tracker is any engine/tracking tracker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def guard_points(mask: np.ndarray, n: int = 120, inset_px: int = 6) -> np.ndarray:
    """(N, 2) points inside `mask` eroded by `inset_px` (distance-transform maxima, spread apart)."""
    from engine.correct import seed_points
    m = (mask > 0.5).astype(np.uint8)
    if inset_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1))
        m = cv2.erode(m, k)
    if not m.any():
        return np.zeros((0, 2), np.float32)
    return np.asarray(seed_points(m, n=n), np.float32).reshape(-1, 2)


def frame_metrics(points: np.ndarray, visible: np.ndarray, mask: np.ndarray, radius: float) -> Dict[str, float]:
    """Metrics of one frame: `points` (N, 2) tracked positions, `visible` (N,) bool, `mask` = SAM3 mask of the frame."""
    h, w = mask.shape
    m = mask > 0.5
    n = int(len(points))
    vis = visible.astype(bool) if n else np.zeros(0, bool)
    nv = int(vis.sum())
    out = {"points": n, "visible": nv, "occluded_frac": round(1.0 - nv / n, 4) if n else 0.0,
           "lost_frac": 0.0, "unsupported_px": 0, "unsupported_frac": 0.0, "mask_px": int(m.sum())}
    if nv == 0:
        out["unsupported_px"] = int(m.sum())
        out["unsupported_frac"] = 1.0 if m.any() else 0.0
        return out
    p = points[vis]
    xi = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
    inside = m[yi, xi]
    out["lost_frac"] = round(float(1.0 - inside.mean()), 4)
    # support: components of the mask without any visible point within `radius`
    support = np.zeros((h, w), np.uint8)
    r = max(1, int(round(radius)))
    for x, y in zip(xi, yi):
        cv2.circle(support, (int(x), int(y)), r, 1, -1)
    ncomp, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    unsupported = 0
    for c in range(1, ncomp):
        comp = lab == c
        if not (support.astype(bool) & comp).any():
            unsupported += int(comp.sum())
    out["unsupported_px"] = unsupported
    out["unsupported_frac"] = round(unsupported / max(1, int(m.sum())), 4)
    return out


def guard_interval(tracker: Any, frames_small: Sequence[np.ndarray], scale: float, frame_ids: Sequence[int],
                   key_frame: int, key_mask: np.ndarray, masks: Dict[int, np.ndarray], n_points: int = 120,
                   inset_px: int = 6) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """Track points from `key_frame` to every frame of `frame_ids` (both directions as given by the order) and
    measure them against `masks[f]` (same geometry as the frames at scale 1 / `frames_small` at `scale`)."""
    frame_ids = list(frame_ids)
    pts = guard_points(key_mask, n_points, inset_px)
    info: Dict[str, Any] = {"key": key_frame, "points": int(len(pts)), "frames": len(frame_ids)}
    if len(pts) == 0 or tracker is None:
        return {f: frame_metrics(np.zeros((0, 2), np.float32), np.zeros(0, bool), masks[f], 1.0) for f in frame_ids}, info
    area = float(max(1, (key_mask > 0.5).sum()))
    radius = 2.0 * float(np.sqrt(area / max(1, len(pts))))
    info["radius_px"] = round(radius, 1)
    t0 = frame_ids.index(key_frame)
    q = np.column_stack([np.full(len(pts), t0, np.float32), pts * scale]).astype(np.float32)
    res = tracker.track(list(frames_small), q)
    out = {}
    for i, f in enumerate(frame_ids):
        out[f] = frame_metrics(res.tracks[i] / scale, res.visible[i], masks[f], radius)
    return out, info


def summarize(metrics: Dict[int, Dict[str, float]], lost_thr: float = 0.10, unsup_thr: float = 0.10) -> Dict[str, Any]:
    fr = sorted(metrics)
    lost = [f for f in fr if metrics[f]["lost_frac"] > lost_thr]
    unsup = [f for f in fr if metrics[f]["unsupported_frac"] > unsup_thr]
    return {"frames": len(fr), "lost_frames": lost, "unsupported_frames": unsup,
            "lost_frac_median": round(float(np.median([metrics[f]["lost_frac"] for f in fr])), 4) if fr else 0.0,
            "lost_frac_max": round(float(max(metrics[f]["lost_frac"] for f in fr)), 4) if fr else 0.0,
            "unsupported_frac_max": round(float(max(metrics[f]["unsupported_frac"] for f in fr)), 4) if fr else 0.0,
            "occluded_frac_max": round(float(max(metrics[f]["occluded_frac"] for f in fr)), 4) if fr else 0.0}
