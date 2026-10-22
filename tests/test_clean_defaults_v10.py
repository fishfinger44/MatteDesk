# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deflicker defaults (2026-10-22 night): stage 1 window 15, 3 iterations, sigma 4, content check 0.5; texture +-4. With the
content check off and the legacy knobs (15 / 2 / sigma 6) the engine equals the frozen reference bit for bit."""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ref_clean_v1 as old
from engine.clean import LazyFrames, PlateCleaner

DEFAULTS = dict(stage1_window=15, stage1_iterations=3, stage1_sigma=4.0, stage1_content_check=0.5, stage3_radius=4)
LEGACY = dict(stage1_window=15, stage1_iterations=2, stage1_sigma=6.0, stage1_content_check=0.0, stage3_radius=4)


def _frames(n=12, w=160, h=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.2 + 0.5 * xx / w)[..., None] * np.array([1.0, 0.9, 0.8], np.float32) + 0.05 * np.sin(yy / 9.0)[..., None]
    return [(base * (1 + 0.10 * np.sin(i * 1.7)) * (1 + 0.05 * (xx / w)[..., None] * (i % 3 - 1))).astype(np.float32) for i in range(n)]


def test_defaults():
    sig = inspect.signature(PlateCleaner.process_sequence).parameters
    assert {k: sig[k].default for k in DEFAULTS} == DEFAULTS


def test_legacy_knobs_equal_reference_and_defaults_differ():
    frames = _frames()
    lazy = lambda: LazyFrames(len(frames), lambda i: frames[i].copy())
    o_frames, o_gains, o_rep, _ = old.PlateCleaner.process_sequence([f.copy() for f in frames], target_reduction=2.0, **LEGACY)
    n_frames, n_gains, n_rep, _ = PlateCleaner.process_sequence(lazy(), target_reduction=2.0, **LEGACY)
    assert n_rep.stages_applied == o_rep.stages_applied
    for i in range(len(frames)):
        assert np.array_equal(n_frames[i], o_frames[i]) and np.array_equal(n_gains[i], o_gains[i]), i
    d_frames, _, _, _ = PlateCleaner.process_sequence(lazy(), target_reduction=2.0)
    e_frames, _, _, _ = PlateCleaner.process_sequence(lazy(), target_reduction=2.0, **DEFAULTS)
    assert all(np.array_equal(a, b) for a, b in zip(d_frames, e_frames))          # defaults == the explicit values
    assert not all(np.array_equal(a, b) for a, b in zip(d_frames, n_frames))      # and they differ from the legacy run


def test_content_check_keeps_a_static_lamp_footprint_and_drops_a_moving_object():
    """A lamp footprint on static texture (same content, other brightness) is still corrected with the check on; the check
    removes the moving object from the neighbours' samples (synthetic E1c ghost metric covers the latter)."""
    rng = np.random.default_rng(3)
    h, w, n = 96, 160, 12
    import cv2
    tex = cv2.resize(rng.random((24, 40)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)[..., None] * np.ones(3, np.float32) * 0.4 + 0.3
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    foot = np.exp(-(((xx - 50) / 18) ** 2 + ((yy - 48) / 18) ** 2))[..., None]
    amp = rng.normal(0, 0.08, n)
    frames = [(tex * np.exp(amp[i] * foot)).astype(np.float32) for i in range(n)]
    lazy = lambda: LazyFrames(n, lambda i: frames[i].copy())
    on, _, _, _ = PlateCleaner.process_sequence(lazy(), target_reduction=2.0, enable_stage2=False, enable_stage4=False)
    off, _, _, _ = PlateCleaner.process_sequence(lazy(), target_reduction=2.0, enable_stage2=False, enable_stage4=False,
                                                 stage1_content_check=0.0)
    spot = foot[..., 0] > 0.5

    def flick(seq):
        return float(np.mean([np.abs(np.log(np.asarray(seq[i])[spot].mean() / np.asarray(seq[i - 1])[spot].mean())) for i in range(1, n)]))
    before = flick(frames)
    assert flick(on) < 0.5 * before and flick(on) <= flick(off) * 1.15         # the footprint is corrected like without the check
