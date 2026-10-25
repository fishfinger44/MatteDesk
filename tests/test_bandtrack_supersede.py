# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Background band trackings supersede each other per (cache dir, frame span) (engine/bandtrack.py, 2026-10-25):
an edit that changes the query set makes the previous set's tracking worthless – skipped if queued, cancelled if
running; an awaited tracking (a run, a preview) is never superseded; the cache holds only the latest set."""

import threading
import time

import numpy as np

from engine import bandtrack as bt
from engine.tracking import TrackResult


class SlowTracker:
    """A tracker whose run blocks until `release` is set; polls cancel_check like CoTracker3Tracker."""

    engine = "cotracker3"

    def __init__(self):
        self.release = threading.Event()
        self.started = []
        self.cancelled = []

    def track(self, frames, queries, cancel_check=None):
        self.started.append(len(queries))
        while not self.release.wait(0.02):
            if cancel_check:
                try:
                    cancel_check()
                except Exception:
                    self.cancelled.append(len(queries))
                    raise
        T = len(frames)
        return TrackResult(tracks=np.repeat(np.asarray(queries, np.float32)[None, :, 1:], T, 0),
                           visible=np.ones((T, len(queries)), bool), engine=self.engine)


def _points(n):
    return [{"id": f"p{i}", "keys": {"3": {"x": 10.0 + i, "y": 20.0, "out": 5, "in": 2}}} for i in range(n)]


def _frames_fn(frames):
    return [np.zeros((20, 30, 3), np.uint8) for _ in frames], 1.0


def test_newer_edit_supersedes_queued_and_running_background_tracking(tmp_path):
    bt._LATEST.clear()
    tr = SlowTracker()
    frames = list(range(1, 9))
    # edit 1: 2 points -> background tracking starts and blocks in the tracker
    out1 = bt.track_positions(_points(2), frames, tr, _frames_fn, tmp_path, resolve=[5], wait=False)
    assert out1[5]["p0"] == (10.0, 20.0, True)                # nearest key's position meanwhile
    deadline = time.time() + 5
    while not tr.started and time.time() < deadline:
        time.sleep(0.01)
    assert tr.started == [2]
    # edit 2 and 3 follow before it finishes: 3 and 4 points
    bt.track_positions(_points(3), frames, tr, _frames_fn, tmp_path, resolve=[5], wait=False)
    bt.track_positions(_points(4), frames, tr, _frames_fn, tmp_path, resolve=[5], wait=False)
    deadline = time.time() + 5
    while not tr.cancelled and time.time() < deadline:
        time.sleep(0.01)
    assert tr.cancelled == [2]                                 # the running one was cancelled
    tr.release.set()
    deadline = time.time() + 5
    while len(list(tmp_path.glob("bandtracks_*.npz"))) < 1 and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    # only the latest set (4 points) was tracked and cached; the 3-point set never started
    assert tr.started == [2, 4]
    assert len(list(tmp_path.glob("bandtracks_*.npz"))) == 1
    pos = bt.track_positions(_points(4), frames, tr, _frames_fn, tmp_path, resolve=[5], wait=False)
    assert pos[5]["p3"] == (13.0, 20.0, True) and tr.started == [2, 4]      # from the cache, no new run


def test_awaited_tracking_is_never_superseded_and_spans_are_independent(tmp_path):
    bt._LATEST.clear()
    tr = SlowTracker()
    tr.release.set()
    frames = list(range(1, 9))
    pos = bt.track_positions(_points(2), frames, tr, _frames_fn, tmp_path, wait=True)
    assert pos[8]["p1"] == (11.0, 20.0, True) and tr.started == [2]
    # a background edit on another span (a 3-frame window) does not touch the full span's latest
    bt.track_positions(_points(3), [3, 4, 5], tr, _frames_fn, tmp_path, resolve=[4], wait=False)
    time.sleep(0.3)
    assert bt._LATEST[(str(tmp_path), 1, 8, 8)] != bt._LATEST[(str(tmp_path), 3, 5, 3)]
    assert len(list(tmp_path.glob("bandtracks_*.npz"))) == 2
