# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandseed.py - the automatic band proposal: Sapiens2 body-part classes along the outline -> band control
points of the shared object band (plan docs/PLAN_SAPIENS2.md point 1; user decisions 2026-10-22: proposals go to the
object band, the width is always max(class rule, automatic width), Sapiens2 only in key frames).

Tested on `beatch` Woman_01 (local_results/sapiens2_band_2026-10-22, sapiens2_vitmatte_2026-10-22): with these rules
the band is 26 % smaller than 40 / 8 + automatic and keeps 99.4 % of the soft hair pixels; in ViTMatte +27 % hair
strands and −27 % haze off the hair (user: „the more hair the better”, Final Touch levels clean it up later).

Per outline pixel: the majority class group in a 13 x 13 window of the object-side ring (0..10 px inside the outline),
smoothed along the contour (mode over ±15 px). Runs of one group (≥ 20 px) -> two points at their ends (3 px in), so
the width changes at the class boundary, not halfway to the next point (bandmap interpolates along the arc). Pure
numpy / cv2; the class map comes from engine/sapiens2.segment_object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Sapiens2 body-part classes (docs/SEG.md of facebookresearch/sapiens2) -> group
HAIR = (4,)
CLOTH = (1, 9, 10, 13, 18, 19, 23)            # apparel, shoes, socks, lower / upper clothing
EYE = (2,)                                     # eyeglass
SKIN = (3, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 27, 28)
GROUPS = ("none", "hair", "cloth", "eye", "skin")       # "none" = background / outside the crop on the edge
# (out, in) px of each group at the band reference (the stage sliders when the proposal is made); "none" keeps the
# constant band (the sliders)
RULE: Dict[str, Optional[Tuple[float, float]]] = {"hair": (48.0, 8.0), "cloth": (12.0, 4.0), "eye": (8.0, 3.0),
                                                  "skin": (4.0, 2.0), "none": None}
RING_PX = 10
WIN = 13
SMOOTH = 15
MIN_RUN = 20
END_INSET = 3
MAX_RUNS = 12                    # per contour (two points per run) – the UI stays usable
MIN_CONTOUR = 200
MIN_PERSON = 0.05                # fraction of the outline's inside the model calls a person – below: not a person


def group_index_map(cls: np.ndarray) -> np.ndarray:
    """Class map -> group index map (GROUPS order; 0 = none / background / 255)."""
    lut = np.zeros(256, np.uint8)
    for gi, ids in ((1, HAIR), (2, CLOTH), (3, EYE), (4, SKIN)):
        lut[list(ids)] = gi
    return lut[np.asarray(cls, np.uint8)]


def person_share(core: np.ndarray, cls: np.ndarray) -> float:
    cb = np.asarray(core) > 0.5
    n = int(cb.sum())
    return float((group_index_map(cls)[cb] > 0).sum()) / n if n else 0.0


def _contours(cb: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(cb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :] for c in cnts if len(c) >= MIN_CONTOUR]


def edge_groups(core: np.ndarray, cls: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """[(contour points (N, 2) x / y, group index per point (N,))] of every outer contour of `core`."""
    cb = np.asarray(core) > 0.5
    k = 2 * RING_PX + 1
    ring = cb & ~cv2.erode(cb.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    g = group_index_map(cls)
    votes = np.stack([cv2.boxFilter(((g == gi) & ring).astype(np.float32), -1, (WIN, WIN), normalize=False)
                      for gi in range(1, len(GROUPS))])               # (4, H, W): counts of each group near the pixel
    out = []
    for P in _contours(cb):
        v = votes[:, P[:, 1], P[:, 0]]                                # (4, N)
        lab = np.where(v.max(0) > 0, v.argmax(0) + 1, 0)
        n = len(lab)
        onehot = np.eye(len(GROUPS), dtype=np.float32)[lab]           # (N, 5)
        ker = np.ones(2 * SMOOTH + 1, np.float32)
        pad = np.concatenate([onehot[-SMOOTH:], onehot, onehot[:SMOOTH]]) if n > 2 * SMOOTH else onehot
        sm = np.stack([np.convolve(pad[:, j], ker, "valid") if n > 2 * SMOOTH else pad[:, j] for j in range(len(GROUPS))], 1)
        out.append((P, sm.argmax(1).astype(np.int32)))
    return out


def _runs(lab: np.ndarray) -> List[List[int]]:
    """Circular runs [start, end, group] (end inclusive, may wrap: end < start), short ones merged into the previous."""
    n = len(lab)
    cut = int(np.argmax(lab != lab[0])) if (lab != lab[0]).any() else 0      # start at a boundary: no run wraps
    seq = np.roll(lab, -cut)
    runs, s = [], 0
    for i in range(1, n + 1):
        if i == n or seq[i] != seq[s]:
            runs.append([s, i - 1, int(seq[s])])
            s = i
    merged: List[List[int]] = []
    for r in runs:
        if merged and (r[1] - r[0] + 1 < MIN_RUN or r[2] == merged[-1][2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < MIN_RUN:
        merged[-1][1] = merged[0][1] + n                  # a short first run belongs to the last one
        merged = merged[1:]
    while len(merged) > MAX_RUNS:                        # the shortest run goes to its longer neighbour
        i = min(range(len(merged)), key=lambda j: merged[j][1] - merged[j][0])
        if i == 0:
            merged[1][0] = merged[0][0]
        else:
            merged[i - 1][1] = merged[i][1]
        merged.pop(i)
    return [[(a + cut) % n, (b + cut) % n, gr] for a, b, gr in merged]


def propose(core: np.ndarray, cls: np.ndarray, frame: int, auto_w: Optional[np.ndarray] = None,
            default: Tuple[float, float] = (40.0, 8.0), id_prefix: str = "s") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Band points (band_points@1, one key in `frame`, widths in px at the band reference) + stats.
    `auto_w` = the automatic half-width map (engine/edge.adaptive_band_width): each point's widths = max(rule, auto)
    at its outline pixel – baked in, so stages without the automatic width (4 / 5 / 7) get it too.
    A group without a rule ("none": background on the edge) takes `default` (the sliders = the constant band)."""
    pts: List[Dict[str, Any]] = []
    share = {g: 0 for g in GROUPS}
    total = 0
    for ci, (P, lab) in enumerate(edge_groups(core, cls)):
        n = len(lab)
        total += n
        for gi in range(len(GROUPS)):
            share[GROUPS[gi]] += int((lab == gi).sum())
        runs = _runs(lab)
        if len(runs) == 1 and RULE[GROUPS[runs[0][2]]] is None:
            continue                                     # a contour of no body part: the constant band stays
        for a, b, gi in runs:
            length = (b - a) % n + 1
            o, i = RULE[GROUPS[gi]] or default
            for idx in sorted({(a + min(END_INSET, length // 2)) % n, (b - min(END_INSET, length // 2)) % n}):
                x, y = int(P[idx][0]), int(P[idx][1])
                if auto_w is not None:
                    aw = float(auto_w[y, x])
                    o, i = max(o, aw), max(i, aw)
                pts.append({"id": f"{id_prefix}{len(pts) + 1}",
                            "keys": {str(int(frame)): {"x": float(x), "y": float(y), "out": round(float(o), 2),
                                                       "in": round(float(i), 2)}}})
    stats = {"points": len(pts), "edge_share": {g: round(v / total, 3) if total else 0.0 for g, v in share.items()}}
    return pts, stats
