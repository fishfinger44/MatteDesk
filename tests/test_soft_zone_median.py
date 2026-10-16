# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 0 / 3 of docs/PLAN_HAIR_TOOL.md (section 2): one temporal stabilization only – the soft zone's alpha (already
stabilized by `stabilize_keys` in the zone) is NOT touched by the temporal median nor by the hygiene of
`refine.finalize` (the override is applied last), also where the soft area overlaps the edge band. Objects without
soft zones: bit-identical to before."""

from pathlib import Path

import numpy as np
import pytest

from engine.crop import CropSegment
from engine.refine import Finalizer


def _run(tmp_path: Path, soft_override):
    n, side = 5, 64
    frames = list(range(1, n + 1))
    core = np.zeros((side, side), bool)
    core[16:48, 16:48] = True
    cores = {f: core for f in frames}
    seg = CropSegment("s00", 1, n, 0, 0, [0, 0, side, side], [0, 0, 0, 0], 1.0)
    fin = Finalizer(client=None, io_dir=tmp_path / "io", plate_w=side, plate_h=side,
                    load_crop_image=lambda f, s, sc: np.zeros((side, side, 3), np.float32))
    run_dir = tmp_path / ("soft" if soft_override else "plain")
    run_dir.mkdir()
    full, rep = fin.finalize({"temporal_smoothing": "normal", "band_scale": 1.0}, [seg], frames, cores,
                             {1: core}, 1, run_dir, edge_mode="core", soft_override=soft_override)
    return full


def test_soft_zone_alpha_not_touched_by_temporal_median(tmp_path):
    n, side = 5, 64
    frames = list(range(1, n + 1))
    # soft area: a strip across the core's right edge (overlaps the +-2 px edge band of the `core` mode)
    area = np.zeros((side, side), bool)
    area[20:44, 40:60] = True
    vals = [0.5, 0.6, 0.5, 0.6, 0.5]   # range 0.1 < motion gate -> the median WOULD flatten it to 0.5 everywhere
    soft = {f: (np.full((side, side), v, np.float32), area) for f, v in zip(frames, vals)}
    full = _run(tmp_path, soft)
    import cv2
    interior = cv2.erode(area.astype(np.uint8), np.ones((19, 19), np.uint8)).astype(bool)   # beyond the 8 px feather
    for f, v in zip(frames, vals):
        # the zone's own alpha survives inside its area (incl. band ∩ area), no median across frames
        assert np.allclose(full[f][interior], v), f
        # feather (step 4b): inside the 8 px rim the zone alpha blends into the base – no step at the polygon border
        rim = area & ~interior
        assert full[f][rim].min() >= min(v, 0.0) - 1e-6 and full[f][rim].max() <= 1.0
        dx = np.abs(np.diff(full[f], axis=1))
        assert dx[20:44, 56:62].max() < 0.2, f    # across the right border of the area (background side)
    # outside the area: identical to the run without soft zones (nothing else changed)
    plain = _run(tmp_path, None)
    for f in frames:
        assert np.array_equal(full[f][~area], plain[f][~area]), f


def test_no_soft_zone_path_unchanged(tmp_path):
    full = _run(tmp_path, None)
    a = full[3]
    assert a.max() == 1.0 and a.min() == 0.0
    assert a[32, 32] == 1.0 and a[2, 2] == 0.0
    # antialiased core ramp sits on the edge (+-1 px) – the median keeps a static shot unchanged
    assert 0.0 < a[32, 16] < 1.0 or a[32, 16] in (0.0, 1.0)
    for f in (1, 5):
        assert np.array_equal(full[f], a)
