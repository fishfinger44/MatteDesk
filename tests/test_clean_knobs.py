# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deflicker knobs behind the palette sliders (docs/PLAN_DEFLICKER_AUTO.md § 4): stage-1 map smoothness (sigma at 512 px)
and stage-4 luma-only gain. Defaults keep the engine bit for bit as before."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from engine.clean import DeflickerEngine, LazyFrames, PlateCleaner
from engine.flicker_auto import DeflickerSettings
from test_clean_defaults_v10 import _frames


def test_stage4_luma_only_gives_one_gain_for_all_channels():
    frames = _frames(n=10)
    _, g_rgb = DeflickerEngine.run_stage4_global_mc([f.copy() for f in frames])
    _, g_luma = DeflickerEngine.run_stage4_global_mc([f.copy() for f in frames], luma_only=True)
    per = np.array([g_rgb[i][0, 0] for i in range(10)])          # (n, 3) gains
    lum = np.array([g_luma[i][0, 0] for i in range(10)])
    assert np.allclose(lum[:, 0], lum[:, 1]) and np.allclose(lum[:, 1], lum[:, 2])   # equal across channels
    assert not np.allclose(per[:, 0], per[:, 2])                                      # the test frames have a colour tilt
    assert np.allclose(np.log(lum[:, 0]), np.log(per).mean(axis=1), atol=0.02)        # luma gain ~ mean of the channel gains


def test_stage1_sigma_changes_the_map_and_defaults_are_unchanged():
    frames = _frames(n=10)
    a, ga, _, _ = PlateCleaner.process_sequence(LazyFrames(10, lambda i: frames[i].copy()), target_reduction=2.0)
    b, gb, _, _ = PlateCleaner.process_sequence(LazyFrames(10, lambda i: frames[i].copy()), target_reduction=2.0, stage1_sigma=4.0, stage4_luma=False)
    assert all(np.array_equal(x, y) for x, y in zip(ga, gb))                          # explicit defaults == defaults
    c, gc, _, _ = PlateCleaner.process_sequence(LazyFrames(10, lambda i: frames[i].copy()), target_reduction=2.0, stage1_sigma=6.0)
    assert not all(np.array_equal(x, y) for x, y in zip(ga, gc))                      # the knob steers stage 1


def test_settings_carry_the_knobs():
    s = DeflickerSettings(strength=5, window=20, grid=True, per_channel=False, smoothness_px=4)
    kw = s.to_process_kwargs(exr=False)
    assert kw["stage1_sigma"] == 4.0 and kw["stage4_luma"] is True and kw["enable_stage3"] is False
