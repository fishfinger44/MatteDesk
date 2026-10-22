# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/correct.py - local correction of a frame mask (stage 3, user request 2026-10-12).

The user clicks only where the mask is wrong:
  + in a hole / a missed part  -> SAM 3.1 is asked again (seed points from the current mask + the click) and ONLY the
                                  new region connected to the click is added,
  - on a leak                   -> SAM 3.1 with a negative point; ONLY the part of the current mask that SAM 3.1 drops and
                                  that touches the click is removed.
Everything else stays bit-identical to the current mask (key of the frame, else the SAM 3.1 core).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def seed_points(mask: np.ndarray, n: int = 8, exclude: Optional[Sequence[float]] = None,
                exclude_r: float = 0.0) -> List[List[float]]:
    """Up to `n` points deep inside `mask` (distance-transform maxima, spread apart) - they tell SAM 3.1 which object
    it is. Points closer than `exclude_r` to `exclude` are skipped (e.g. around a negative click)."""
    b = (mask > 0.5).astype(np.uint8)
    if not b.any():
        return []
    ys, xs = np.nonzero(b)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sub = b[y0:y1, x0:x1]
    # work on a small copy: seeds need no full-resolution accuracy
    s = min(1.0, 512.0 / max(sub.shape))
    small = cv2.resize(sub, (max(1, round(sub.shape[1] * s)), max(1, round(sub.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(np.pad(small, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    spacing = max(2.0, 0.18 * np.sqrt(float(small.sum())))
    out: List[List[float]] = []
    work = dt.copy()
    for _ in range(n * 4):
        if len(out) >= n:
            break
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[iy, ix] <= 0.5:
            break
        px, py = x0 + (ix + 0.5) / s, y0 + (iy + 0.5) / s
        cv2.circle(work, (int(ix), int(iy)), int(round(spacing)), 0, -1)
        if exclude is not None and np.hypot(px - exclude[0], py - exclude[1]) < exclude_r:
            continue
        out.append([float(px), float(py)])
    return out


def component_at(region: np.ndarray, point: Sequence[float], snap_px: float = 12.0) -> np.ndarray:
    """Connected component of `region` under `point`; if the click is just beside it (<= snap_px), the nearest one."""
    r = (region > 0.5).astype(np.uint8)
    h, w = r.shape
    empty = np.zeros((h, w), bool)
    if not r.any():
        return empty
    n, lab = cv2.connectedComponents(r, connectivity=8)
    x, y = int(round(point[0])), int(round(point[1]))
    if 0 <= x < w and 0 <= y < h and lab[y, x] > 0:
        return lab == lab[y, x]
    s = int(np.ceil(snap_px))
    xa, xb, ya, yb = max(0, x - s), min(w, x + s + 1), max(0, y - s), min(h, y + s + 1)
    if xa >= xb or ya >= yb:
        return empty
    win = lab[ya:yb, xa:xb]
    yy, xx = np.nonzero(win)
    if not len(yy):
        return empty
    d = np.hypot(xx + xa - x, yy + ya - y)
    i = int(np.argmin(d))
    if d[i] > snap_px:
        return empty
    return lab == win[yy[i], xx[i]]


def local_correction(base: np.ndarray, sam: np.ndarray, point: Sequence[float], mode: str,
                     snap_px: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """base = current mask, sam = SAM 3.1 mask for (seeds + click). Returns (new mask, changed region).
    add: only the part of (sam - base) connected to the click is added; remove: only the part of (base - sam)
    under the click is removed. Pixels outside the changed region are exactly `base`."""
    b, m = base > 0.5, sam > 0.5
    if mode == "add":
        comp = component_at(m & ~b, point, snap_px)
        return b | comp, comp
    if mode == "remove":
        comp = component_at(b & ~m, point, snap_px)
        return b & ~comp, comp
    raise ValueError("mode: add | remove")
