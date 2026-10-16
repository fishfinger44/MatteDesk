# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for the tools/exp_*.py scripts: the repo root (no machine path in the scripts) and plate paths of a shot that
was copied to another machine (the read-only manifest still points at the old drive)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def local_plate(shot_dir, pattern: str) -> str:
    """`pattern` as is when its folder exists; else the same path re-rooted at the shot folder: the part after the shot
    folder's name (e.g. F:/.../work/CLIP2_A/plate/CLIP2_A_####.png -> <shot_dir>/plate/CLIP2_A_####.png)."""
    shot_dir = Path(shot_dir)
    p = Path(pattern)
    if not pattern or p.parent.exists():
        return pattern
    parts = p.parts
    if shot_dir.name in parts:
        i = len(parts) - 1 - parts[::-1].index(shot_dir.name)
        cand = shot_dir.joinpath(*parts[i + 1:])
        if cand.parent.exists():
            return str(cand).replace("\\", "/")
    return pattern
