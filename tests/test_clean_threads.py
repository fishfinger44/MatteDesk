# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate cleaning (engine/clean.py): the threaded per-frame loops give exactly the single-threaded result."""

import cv2
import numpy as np
import pytest

from engine import clean
from engine.clean import PlateCleaner, _map_frames, clean_workers


def _frames(n=6, w=160, h=107):
    rng = np.random.default_rng(3)
    base = cv2.resize(rng.random((16, 24, 3), dtype=np.float32), (w + 20, h + 20), interpolation=cv2.INTER_CUBIC) * 0.6 + 0.2
    return [(base[5:5 + h, 5 + i % 3:5 + i % 3 + w] * (1 + 0.08 * np.sin(i * 1.7))).astype(np.float32) for i in range(n)]


@pytest.mark.parametrize("kw", [{}, {"enable_stage3": True}])
def test_threads_match_single_thread(monkeypatch, kw):
    frames = _frames()
    monkeypatch.setenv("MD_CLEAN_THREADS", "1")
    a_frames, a_gains, a_rep, _ = PlateCleaner.process_sequence(frames, **kw)
    monkeypatch.setenv("MD_CLEAN_THREADS", "4")
    b_frames, b_gains, b_rep, _ = PlateCleaner.process_sequence(frames, **kw)
    assert a_rep.stages_applied == b_rep.stages_applied
    for x, y in zip(a_frames, b_frames):
        assert np.array_equal(x, y)
    for x, y in zip(a_gains, b_gains):
        assert np.array_equal(x, y)


def test_map_frames_keeps_order_and_propagates_errors():
    assert _map_frames(lambda i: i * i, 20, 4) == [i * i for i in range(20)]
    assert _map_frames(lambda i: i, 0, 4) == []

    def boom(i):
        if i == 3:
            raise ValueError("frame 3")
        return i
    with pytest.raises(ValueError):
        _map_frames(boom, 10, 4)


def test_clean_workers_env_and_ram_cap(monkeypatch):
    monkeypatch.setenv("MD_CLEAN_THREADS", "3")
    assert clean_workers(10 ** 12) == 3
    monkeypatch.delenv("MD_CLEAN_THREADS")
    monkeypatch.setattr(clean, "_available_ram_bytes", lambda: 4 * 10 ** 9)
    assert clean_workers(10 ** 9) <= 2                       # half of 4 GB / 1 GB per thread
    assert clean_workers(10 ** 12) == 1                      # never below one thread
    monkeypatch.setattr(clean, "_available_ram_bytes", lambda: None)
    assert clean_workers(10 ** 9) <= 2                       # unknown RAM -> conservative
