# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„▶ Przelicz klatkę” with a prompt (user decision 2026-10-15, v2): prompt = base, positive points add, negative points
subtract what they point at: (prompt ∪ plus) − minus (engine.sam3.combine_prompt_points). SAM3_Detect itself, given a
prompt and points at once, returns their union and ignores negative points (measured, local_results/sam3_prompt_points)."""
import numpy as np

from engine.sam3 import combine_prompt_points


def _disc(cx, cy, r=10, shape=(100, 200)):
    yy, xx = np.indices(shape)
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.float32)


def test_plus_adds_to_the_prompt():
    m, c = combine_prompt_points(_disc(30, 50), _disc(100, 50), None)
    assert m[50, 30] and m[50, 100] and c["minus_px"] == 0


def test_minus_subtracts_from_prompt_and_plus():
    base = _disc(30, 50) + _disc(170, 50)
    m, c = combine_prompt_points(base, _disc(100, 50), _disc(170, 50, 12))
    assert m[50, 30] and m[50, 100] and not m[50, 170] and c["minus_px"] > 0


def test_minus_only_and_missing_parts():
    m, _ = combine_prompt_points(_disc(30, 50), None, _disc(30, 50, 4))
    assert m[50, 22] and not m[50, 30]                       # only the clicked part is cut out
    m, c = combine_prompt_points(None, _disc(60, 40), None)
    assert m[40, 60] and c["prompt_px"] == 0
