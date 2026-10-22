# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deflicker frame cache (2026-10-21): frames pinned in RAM up to a fixed budget, the rest re-read; the result is bit
for bit the one of plain re-reading (the cleaner sweeps the whole sequence ~10x, so a sliding window would never hit)."""
import numpy as np
import pytest

from engine import clean
from engine.clean import LazyFrames, PinnedFrameCache, PlateCleaner, clean_cache_budget


def _half_frames(n=8, w=160, h=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.2 + 0.5 * xx / w)[..., None] * np.array([1.0, 0.9, 0.8], np.float32) + 0.05 * np.sin(yy / 9.0)[..., None]
    # values representable in half (like a half EXR read as float32)
    return [(base * (1 + 0.1 * np.sin(i * 1.7))).astype(np.float16).astype(np.float32) for i in range(n)]


def test_pinned_cache_budget_hits_and_copies():
    frames = _half_frames(6)
    reads = []
    per = frames[0].astype(np.float16).nbytes
    c = PinnedFrameCache(6, lambda i: (reads.append(i), frames[i])[1], budget_bytes=3 * per)
    for _ in range(3):                                   # three sweeps over the sequence, like the cleaner
        for i in range(6):
            assert np.array_equal(c[i], frames[i])
    st = c.stats()
    assert st["pinned"] == 3 and st["disk_reads"] == 3 + 3 * 3 and st["ram_hits"] == 3 * 2
    assert reads.count(0) == 1 and reads.count(5) == 3   # first frames pinned, the rest re-read every sweep
    a = c[0]
    a[:] = 7.0                                           # a caller writing into its frame never changes the cache
    assert np.array_equal(c[0], frames[0]) and c[0].dtype == np.float32 and c[-1].shape == frames[5].shape


def test_pinned_cache_keeps_float32_when_half_is_lossy():
    f = np.full((4, 4, 3), 0.1234567, np.float32)        # not representable in half
    c = PinnedFrameCache(1, lambda i: f, budget_bytes=10 ** 6)
    c[0]
    assert c.stats()["pinned_gb"] * 2 ** 30 == pytest.approx(f.nbytes, abs=2 ** 30 * 0.006)
    assert np.array_equal(c[0], f)


def test_zero_budget_is_plain_rereading():
    frames = _half_frames(3)
    c = PinnedFrameCache(3, lambda i: frames[i], 0)
    for i in range(3):
        c[i]
        c[i]
    assert c.stats()["pinned"] == 0 and c.stats()["disk_reads"] == 6


@pytest.mark.parametrize("threads", ["1", "4"])
def test_cleaner_bitwise_same_with_cache(monkeypatch, threads):
    monkeypatch.setenv("MD_CLEAN_THREADS", threads)
    frames = _half_frames(8)
    kw = dict(target_reduction=2.0)                      # all stages
    o_frames, o_gains, o_rep, _ = PlateCleaner.process_sequence(LazyFrames(8, lambda i: frames[i].copy()), **kw)
    per = frames[0].astype(np.float16).nbytes
    for budget in (0, 3 * per, 100 * per):               # none / part / all pinned
        c = PinnedFrameCache(8, lambda i: frames[i].copy(), budget)
        n_frames, n_gains, n_rep, _ = PlateCleaner.process_sequence(c, **kw)
        for i in range(8):
            assert np.array_equal(n_frames[i], o_frames[i]) and np.array_equal(n_gains[i], o_gains[i])
        assert n_rep.reduction_pct == o_rep.reduction_pct and n_rep.stages_applied == o_rep.stages_applied
        if budget >= 8 * per:
            assert c.stats()["disk_reads"] == 8              # every frame decoded exactly once


def test_cache_budget_env_and_free_ram(monkeypatch):
    monkeypatch.setenv("MD_CLEAN_CACHE_GB", "2")
    assert clean_cache_budget() == 2 * 2 ** 30
    monkeypatch.setenv("MD_CLEAN_CACHE_GB", "0")
    assert clean_cache_budget() == 0
    monkeypatch.delenv("MD_CLEAN_CACHE_GB")
    monkeypatch.setattr(clean, "_available_ram_bytes", lambda: 10 * 2 ** 30)
    assert clean_cache_budget() == int(10 * 2 ** 30 * clean.CACHE_RAM_SHARE)
    monkeypatch.setattr(clean, "_available_ram_bytes", lambda: None)
    assert clean_cache_budget() == 0
