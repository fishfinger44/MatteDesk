# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/nuke_nk.py - Shared helpers for writing Nuke .nk text (Align / Deflicker / Camera2D / Depth Groups).

Conventions
- OpenCV / numpy pixel coords: pixel centre at integer, y down.
- Nuke: pixel centre at +0.5, origin bottom-left, y up  ->  x_n = x + 0.5,  y_n = H - 0.5 - y.
- Animated knobs:
    single value:  rotate {{curve x1001 0.1 x1002 0.2}}
    XY knob:       translate {{curve x1001 1 x1002 2} {curve x1001 3 x1002 4}}
  (one outer brace pair holding one `{curve ...}` per component – `{{curve}} {{curve}}` is NOT valid.)
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def cv_to_nuke(h: int) -> np.ndarray:
    """3x3 matrix mapping OpenCV pixel coords to Nuke coords for a plate of height h (self-inverse)."""
    return np.array([[1.0, 0.0, 0.5], [0.0, -1.0, h - 0.5], [0.0, 0.0, 1.0]])


def matrix_cv_to_nuke(m: np.ndarray, h: int) -> np.ndarray:
    """Forward transform given in OpenCV coords (2x3 or 3x3) -> same transform in Nuke coords."""
    m3 = np.eye(3)
    m = np.asarray(m, dtype=np.float64)
    m3[: m.shape[0], :] = m
    f = cv_to_nuke(h)
    return f @ m3 @ np.linalg.inv(f)


def similarity_to_transform_knobs(m_nuke: np.ndarray, center: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Nuke Transform (translate x/y, rotate deg CCW, uniform scale) equal to similarity m_nuke about `center`.

    Nuke: p' = c + s R (p - c) + T   ->   T = A c + t - c   (A = sR, t = m[:2, 2]).
    """
    a = m_nuke[:2, :2]
    t = m_nuke[:2, 2]
    c = np.asarray(center, dtype=np.float64)
    tr = a @ c + t - c
    rot = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    scale = math.sqrt(abs(np.linalg.det(a)))
    return float(tr[0]), float(tr[1]), rot, scale


def _fmt(v: float, nd: int) -> str:
    s = f"{v:.{nd}f}"
    return "0" if s in ("-0", "-0." + "0" * nd) else s


def curve(frames: Sequence[int], values: Sequence[float], nd: int = 4) -> str:
    """`{curve x1 v1 x2 v2 ...}` (a constant curve collapses to the bare value)."""
    vals = [float(v) for v in values]
    if vals and max(vals) - min(vals) < 10 ** (-nd):
        return _fmt(vals[0], nd)
    return "{curve " + " ".join(f"x{int(f)} {_fmt(v, nd)}" for f, v in zip(frames, vals)) + "}"


def scalar_knob(name: str, frames: Sequence[int], values: Sequence[float], nd: int = 4) -> str:
    c = curve(frames, values, nd)
    return f"{name} {{{c}}}" if c.startswith("{") else f"{name} {c}"


def xy_knob(name: str, frames: Sequence[int], xs: Sequence[float], ys: Sequence[float], nd: int = 4) -> str:
    return f"{name} {{{curve(frames, xs, nd)} {curve(frames, ys, nd)}}}"


def corner_pin_knobs(frames: Sequence[int], matrices_nuke: Iterable[np.ndarray], w: int, h: int, nd: int = 3) -> List[str]:
    """to1..to4 / from1..from4 for a CornerPin2D whose 'from' = plate corners (Nuke coords, BL BR TR TL)."""
    corners = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    mats = list(matrices_nuke)
    lines = []
    for i, (cx, cy) in enumerate(corners, start=1):
        xs, ys = [], []
        for m in mats:
            p = m @ np.array([cx, cy, 1.0])
            xs.append(p[0] / p[2])
            ys.append(p[1] / p[2])
        lines.append(xy_knob(f"to{i}", frames, xs, ys, nd))
    for i, (cx, cy) in enumerate(corners, start=1):
        lines.append(f"from{i} {{{_fmt(cx, 0)} {_fmt(cy, 0)}}}")
    return lines


_BAD_XY = re.compile(r"\}\}\s+\{\{curve")


def validate_nk(text: str, group_name: str = "") -> List[str]:
    """Structural checks of a generated Group .nk. Returns a list of problems (empty = OK)."""
    problems = []
    depth = 0
    in_str = False
    prev = ""
    for ch in text:
        if ch == '"' and prev != "\\":
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    problems.append("unbalanced '}'")
                    depth = 0
        prev = ch
    if depth != 0:
        problems.append(f"unbalanced braces (depth {depth})")
    if in_str:
        problems.append("unterminated string")
    if _BAD_XY.search(text):
        problems.append("XY knob written as {{curve}} {{curve}} – must be {{curve ..} {curve ..}}")
    if not text.lstrip().startswith("Group {"):
        problems.append("does not start with Group {")
    if not text.rstrip().endswith("end_group"):
        problems.append("does not end with end_group")
    if text.count("name Input1") != 1:
        problems.append("Group must have exactly one Input1")
    if text.count("name Output1") != 1:
        problems.append("Group must have exactly one Output1")
    if group_name and f"name {group_name}\n" not in text:
        problems.append(f"Group name {group_name} missing")
    return problems
