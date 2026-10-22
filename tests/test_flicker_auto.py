# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_auto.py (docs/PLAN_DEFLICKER_AUTO.md § 3): measurement -> slider values + notes. Rules on constructed
measurements, then end to end on the synthetic sequence of test_flicker_measure (lamp footprint on one object)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from engine import flicker_auto as fa
from engine.flicker_auto import DeflickerSettings, auto_settings, strength_for, texture_radius_for, window_for
from engine.flicker_measure import FlickerMeasure, RegionMeasure, measure_flicker
from test_flicker_measure import synth


def region(name, level=1.0, ac=0.0, drift=0.0, dsig=1.0, rg=0.1, bg=0.1, band=0.1, band_floor=0.1, share=None, motion=0.2,
           metric=2.0, floor=0.8, grain=0.3, grain_floor=0.3):
    return RegionMeasure(name=name, px=1000, metric_pct=metric, floor_pct=floor, level_pct=level, level_series_pct=[level] * 9,
                         level_autocorr=ac, drift_pct=drift, drift_sigma_pct=dsig, colour_pct={"rg": rg, "bg": bg},
                         local_pct=1.0, local_floor_pct=0.5,
                         texture_bands_pct={"<2": grain, "2-8": 0.2, "8-40": band, ">40": 0.1},
                         texture_floor_bands_pct={"<2": grain_floor, "2-8": 0.1, "8-40": band_floor, ">40": 0.05},
                         texture_fixed_share=share, motion_resid_px=motion, texture_share=0.8)


def measure(regions, corr=0.3, shift=0.3, n=20):
    return FlickerMeasure(n_frames=n, eval_w=1024, eval_h=683, global_shift_px=shift, region_corr=corr,
                          tiles_level_pct=[[1.0] * 9] * 6, regions={r.name: r for r in regions})


def test_strength_window_radius_rules():
    assert [strength_for(x) for x in (0.1, 0.5, 2.0, 4.0, 9.0)] == [2, 3, 3, 3, 3]          # capped at 3 (ghosts)
    assert window_for(None) == (10, "unknown") and window_for(-0.4) == (10, "random") and window_for(0.1) == (10, "random")
    w8, k8 = window_for(0.8)
    assert k8 == "memory" and 10 <= w8 <= 20 and window_for(0.95)[0] == 20
    assert texture_radius_for(-0.3) == 4 and texture_radius_for(0.5) == 3 and texture_radius_for(0.9) == 2


def test_sh010_like_measurement_gives_v10_settings_and_texture():
    # numbers of SH010 X1 F90-109 (tree: pulsing lamp with a fixed footprint; wall: plain, slow variation)
    tree = region("tree", level=1.16, ac=-0.38, drift=-4.1, dsig=0.83, rg=0.22, bg=0.26, band=2.24, band_floor=0.28, share=0.88,
                  motion=1.0, metric=4.17, floor=1.32)
    wall = region("wall", level=1.35, ac=0.79, drift=-3.1, dsig=3.9, rg=0.14, bg=0.52, band=0.5, band_floor=0.21, share=None,
                  motion=None, metric=1.55, floor=0.45)
    allr = region("all", level=1.17, ac=0.57, rg=0.13, bg=0.35, band=1.34, band_floor=0.69, share=0.63, motion=0.9, metric=2.1, floor=0.77)
    s = auto_settings(measure([allr, tree, wall], corr=0.35, shift=0.41))
    assert s.strength == 3 and s.driver_region == "wall"            # capped at 3 iterations (E1c ghost limit)
    assert s.grid is True                                            # tiles flicker apart (0.35 < 0.5)
    assert s.per_channel is True                                     # wall B/G 0.52 % > 0.3
    assert s.texture is True and s.texture_region == "tree" and s.texture_radius == 4   # fixed footprint, pulsing
    assert s.denoise_hint is False
    assert any("Powolna zmiana" in n and "tree" in n for n in s.notes)           # tree drift -4.1 > 2 sigma (0.83)
    assert not any("Powolna zmiana" in n and "wall" in n for n in s.notes)       # wall drift -3.1 < 2 sigma (3.9)
    assert any("rusza się niezależnie" in w and "tree" not in w for w in s.warnings) is False   # 1.0 px is not > 1.0
    assert s.achievable["tree"] == {"now": 4.17, "floor": 1.32}
    kw = s.to_process_kwargs(exr=True)
    assert kw["enable_stage3"] is True and kw["stage1_window"] == s.window and kw["stage1_iterations"] == 3 and kw["target_reduction"] == 2.0
    assert s.to_process_kwargs(exr=False)["enable_stage3"] is False  # Nuke Group: stage 3 never
    assert "window" in s.to_dict() and all(isinstance(n, str) for n in s.notes)


def test_quiet_plate_and_grid_off_and_motion_warning():
    quiet = region("all", level=0.2, ac=0.05, rg=0.05, bg=0.05, band=0.1, band_floor=0.1, share=None, motion=0.1)
    s = auto_settings(measure([quiet], corr=0.9))
    assert s.strength == 2 and s.window == 10 and s.grid is False and s.per_channel is False and s.texture is False
    assert any("kratka 9×6 wyłączona" in n for n in s.notes)
    mover = region("head", level=2.5, ac=0.85, motion=2.4)
    s2 = auto_settings(measure([region("all", level=1.0), mover], corr=None))
    assert s2.driver_region == "head" and s2.strength == 3 and s2.window > 10 and s2.grid is True
    assert any("head" in w and "rusza się" in w for w in s2.warnings)
    assert any("bezpiecznie" in n for n in s2.notes)
    s3 = auto_settings(measure([region("all", level=6.0, grain=1.2, grain_floor=0.3)]))
    assert s3.strength == 3 and s3.denoise_hint is True
    # camera motion (SH010 F1-16): the grid is off even though the tiles do not agree, and the border warning is given
    s4 = auto_settings(measure([region("all", level=1.5), region("inner", level=1.4)], corr=0.2, shift=5.1))
    assert s4.grid is False and any("kratka 9×6 wyłączona" in n for n in s4.notes) and any("brzeg kadru" in w for w in s4.warnings)
    assert s4.driver_region == "all"                                   # 'inner' never drives the strength on its own
    with pytest.raises(ValueError):
        auto_settings(measure([]))


def test_end_to_end_on_synthetic_lamp_footprint():
    frames, tm, _ = synth(level=0.01, blob=0.06)
    m = measure_flicker(frames, {"tree": tm, "wall": ~tm}, eval_w=192)
    s = auto_settings(m)
    assert isinstance(s, DeflickerSettings)
    assert s.texture is True and s.texture_region == "tree"           # the fixed footprint is found on the right object
    assert s.strength >= 3                                            # ±1 % random gain + the blob
    frames2, _, _ = synth(level=0.03, drift=0.0, seed=3)
    s2 = auto_settings(measure_flicker(frames2, eval_w=192))
    assert s2.texture is False and s2.grid is False                   # one lamp over the whole frame: tiles agree, no fixed pattern


def test_rhythm_sets_the_window_to_whole_periods():
    r = region("all", level=1.5, ac=-0.5)
    r.period_frames, r.period_share, r.period_p = 3.0, 0.7, 0.0
    s = auto_settings(measure([r], corr=0.9))
    assert s.window == 12 and any("rytm" in n for n in s.notes)                   # 4 periods of 3 frames >= 10
    r2 = region("all", level=1.5, ac=-0.5)
    r2.period_frames, r2.period_share, r2.period_p = 3.0, 0.7, 0.2               # not significant -> as before
    assert auto_settings(measure([r2], corr=0.9)).window == 10

