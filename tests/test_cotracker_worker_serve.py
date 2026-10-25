# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The long-lived CoTracker3 worker (engine/cotracker_worker.py --serve, 2026-10-25): one process per interpreter,
the model loaded once, every tracking = one request; the one-shot path (a process per call) stays for the real
bit-identity check (tests/test_cotracker_real.py). Protocol tests with MD_COTRACKER_FAKE=1 in this .venv's Python."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import tracking as tk

FAKE_ENV = {**os.environ, "MD_COTRACKER_FAKE": "1"}


@pytest.fixture
def fake_tracker(tmp_path):
    tk._CHECK_CACHE.clear()
    ckpt = tmp_path / "scaled_offline.pth"
    ckpt.write_bytes(b"fake")
    t = tk.CoTracker3Tracker(ckpt, python=sys.executable, env=FAKE_ENV, max_side=64, timeout_s=60)
    yield t
    tk.close_workers()
    tk._CHECK_CACHE.clear()


def _video(T=6, h=40, w=48, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), np.uint8) for _ in range(T)]


def test_served_worker_equals_one_shot_and_is_reused(fake_tracker):
    t = fake_tracker
    frames = _video()
    q = np.array([[0, 10.0, 12.0], [2, 30.0, 20.0]], np.float32)
    served = t.track(frames, q)
    t.persistent = False
    one_shot = t.track(frames, q)
    t.persistent = True
    assert served.engine == "cotracker3" and served.tracks.shape == (6, 2, 2)
    assert np.array_equal(served.tracks, one_shot.tracks) and np.array_equal(served.visible, one_shot.visible)
    w = tk.cotracker_worker(t.python, t.env)
    pid = w._proc.pid
    t.track(frames, q)
    assert w._proc.pid == pid and w.device == "fake"          # one long-lived process, no reload
    # several videos in one request, frames shared through the pool, scaled back to the input pixels
    res = t.track_many([(frames, q), (frames[2:], np.array([[0, 5.0, 5.0]], np.float32))])
    assert len(res) == 2 and res[1].tracks.shape == (4, 1, 2) and np.allclose(res[1].tracks[:, 0], [5.0, 5.0])


def test_dead_worker_restarts(fake_tracker):
    t = fake_tracker
    frames = _video()
    q = np.array([[0, 10.0, 12.0]], np.float32)
    t.track(frames, q)
    w = tk.cotracker_worker(t.python, t.env)
    pid = w._proc.pid
    w._proc.kill()
    w._proc.wait()
    assert t.track(frames, q).tracks.shape == (6, 1, 2)       # dead worker -> a fresh one
    assert w._proc.pid != pid

    src, dst = Path(w._dir.name) / "x.npz", Path(w._dir.name) / "y.npz"
    np.savez(src, n_videos=np.array(0), checkpoint=np.array("x"))
    w._proc.kill()
    w._proc.wait()
    assert w.request(src, dst, None, timeout_s=30)["ok"]      # a dead worker is replaced on the next request
    assert t.track(frames, q).tracks.shape == (6, 1, 2)       # and the next call works again


def test_worker_error_does_not_hang(fake_tracker):
    t = fake_tracker
    w = tk.cotracker_worker(t.python, t.env)
    with pytest.raises(RuntimeError, match="CoTracker3"):
        w.request(Path("Z:/does/not/exist.npz"), Path(os.environ.get("TEMP", ".")) / "md_never.npz", None, 60)
    frames = _video()
    assert t.track(frames, np.array([[0, 1.0, 1.0]], np.float32)).tracks.shape == (6, 1, 2)   # worker survives
