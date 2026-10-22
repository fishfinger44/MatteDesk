# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_preview.py (docs/PLAN_DEFLICKER_AUTO.md § 5): deflicker on a downscaled window, before / after measured on the
original's flows, heat maps on one scale, display frames for the loop."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from engine.flicker_auto import auto_settings
from engine.flicker_measure import measure_flicker
from engine.flicker_preview import PreviewResult, display_frame, heat_map, preview_deflicker
from test_flicker_measure import synth


def test_preview_reduces_flicker_and_reports_relative_to_floor():
    frames, tm, _ = synth(n=16, level=0.04, seed=4)
    masks = {"tree": tm, "wall": ~tm}
    s = auto_settings(measure_flicker(frames, masks, eval_w=192))
    r = preview_deflicker(frames, s.to_process_kwargs(exr=False), masks, eval_w=192)
    assert isinstance(r, PreviewResult) and r.eval_w == 192 and r.n_frames == 16
    for nm in ("all", "tree", "wall"):
        reg = r.regions[nm]
        assert reg["after"] < reg["before"] and reg["removed_pct"] > 50          # a pure global gain flicker is mostly removable
        assert reg["floor"] <= reg["after"] + 0.5
    assert "stage1_local_mc" in r.stages_applied and r.seconds >= 0
    assert r.map_before.shape == (r.eval_h, 192, 3) and r.map_before.dtype == np.uint8 and r.map_after.shape == r.map_before.shape
    assert len(r.frames_before) == 16 and r.frames_before[0].dtype == np.uint8 and r.frames_after[0].shape == r.frames_before[0].shape
    assert r.vmax_pct > 0 and any("±8" in n or "ograniczone" in n for n in r.notes)   # window 10+ capped by 16 frames
    d = r.to_dict()
    json.dumps(d)
    assert "map_before" not in d and d["before"]["regions"]["all"]["metric_pct"] == r.regions["all"]["before"]


def test_preview_without_images_and_small_input():
    frames, tm, _ = synth(n=8, level=0.02)
    r = preview_deflicker(frames, {"stage1_iterations": 2}, eval_w=192, with_images=False)
    assert r.map_before is None and r.frames_before is None and set(r.regions) == {"all", "inner"}
    with pytest.raises(ValueError):
        preview_deflicker(frames[:2], {}, eval_w=192)


def test_heat_map_and_display_frame_helpers():
    m = np.linspace(0, 4, 32 * 48, dtype=np.float32).reshape(32, 48)
    hm = heat_map(m, vmax=2.0)
    assert hm.shape == (32, 48, 3) and hm.dtype == np.uint8 and hm[0, 0].sum() < hm[-1, -1].sum()   # dark -> bright
    f = display_frame(np.full((8, 8, 3), 0.18, np.float32))
    assert f.shape == (8, 8, 3) and f.dtype == np.uint8 and 30 < int(f.mean()) < 230
