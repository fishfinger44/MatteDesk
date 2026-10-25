# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Station only (`pytest -m real`): the served CoTracker3 worker (engine/cotracker_worker.py --serve) returns the
same tracks as the one-shot process-per-call path it replaced (2026-10-25), on real plate frames of the tutorial
shot, and the served worker is reproducible run to run."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import tracking as tk

SHOT = Path(__file__).resolve().parents[1] / "_tutorial" / "work" / "sh060" / "matte_desk" / "source" / "sh060"

pytestmark = pytest.mark.real


def _frames(n=12, first=30):
    out = []
    for f in range(first, first + n):
        im = cv2.imread(str(SHOT / f"sh060_{f:04d}.png"), cv2.IMREAD_COLOR)
        if im is None:
            pytest.skip("brak klatek sh060")
        out.append(cv2.resize(im[..., ::-1], (1024, 576), interpolation=cv2.INTER_AREA))
    return out


def test_served_worker_bit_identical_to_one_shot():
    ckpt = tk.default_cotracker_checkpoint()
    if not ckpt or not ckpt.exists():
        pytest.skip("brak wag CoTracker3")
    frames = _frames()
    rng = np.random.default_rng(1)
    q = np.column_stack([np.zeros(10), rng.uniform(300, 700, 10), rng.uniform(100, 500, 10)]).astype(np.float32)
    q[5:, 0] = 3
    t = tk.CoTracker3Tracker(ckpt, persistent=False)
    one_shot = t.track(frames, q)
    t.persistent = True
    served = t.track(frames, q)
    served2 = t.track(frames, q)
    tk.close_workers()
    assert np.array_equal(one_shot.tracks, served.tracks) and np.array_equal(one_shot.visible, served.visible)
    assert np.array_equal(served.tracks, served2.tracks)
    assert served.tracks.shape == (12, 10, 2) and np.isfinite(served.tracks).all()
