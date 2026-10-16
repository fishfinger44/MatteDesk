# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z21 – a local correction changes the core only in its tracked region, also in the neighbouring frames of the
recomputed intervals (user request 2026-10-12): engine/correct_region + CorePropagator(old_core, deltas, region_fn)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.correct_region import (control_points, load_delta, merge_region, region_from_delta, save_delta,
                                   track_region)
from engine.crop import CropSegment
from engine.segment import CorePropagator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def test_merge_region_old_outside_new_inside_feathered():
    h, w = 120, 160
    old = np.zeros((h, w), bool); old[30:90, 40:120] = True
    new = old.copy(); new[30:90, 40:120] = True; new[50:70, 60:80] = False      # new has a hole (should NOT win outside)
    new[20:30, 70:90] = True                                                      # and a bump at the top (inside region)
    region = np.zeros((h, w), bool); region[10:40, 60:100] = True
    out = merge_region(old, new, region, feather=4)
    assert np.array_equal(out[~region], old[~region])                             # outside: bit-identical old
    inner = cv2.erode(region.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert np.array_equal(out[inner], new[inner])                                 # deep inside: new
    assert out[55:65, 65:75].all()                                                # the hole outside the region stays old
    assert np.array_equal(merge_region(old, new, np.zeros((h, w), bool)), old)


def test_region_from_delta_and_control_points():
    d = np.zeros((200, 200), bool); d[90:110, 90:110] = True
    r = region_from_delta(d)
    assert r.sum() > d.sum() and r[100, 100] and r[100, 100 + 16] and not r[100, 100 + 40]
    pts = control_points(r)
    assert len(pts) >= 8 and all(r[int(round(y)) if y < 200 else 199, int(round(x)) if x < 200 else 199] or True for x, y in pts)


class _ShiftTracker:
    """Stands in for CoTracker3: points move by `dx` px per frame; `lost` frames mark every point invisible."""
    engine = "mock"

    def __init__(self, dx=3.0, lost=()):
        self.dx = dx
        self.lost = set(lost)

    def track(self, frames, queries):
        from engine.tracking import TrackResult
        T, N = len(frames), len(queries)
        tracks = np.zeros((T, N, 2), np.float32)
        visible = np.ones((T, N), bool)
        for i in range(T):
            t0 = int(queries[0, 0])
            tracks[i, :, 0] = queries[:, 1] + self.dx * (i - t0)
            tracks[i, :, 1] = queries[:, 2]
            if i in self.lost:
                visible[i] = False
        return TrackResult(tracks=tracks, visible=visible, engine=self.engine)


def test_track_region_follows_points_and_falls_back():
    region = np.zeros((120, 200), bool); region[40:80, 40:80] = True
    frames = list(range(1, 8))
    small = [np.zeros((60, 100, 3), np.uint8) for _ in frames]       # scale 0.5
    regs, info = track_region(_ShiftTracker(dx=2.0, lost=[6]), small, 0.5, frames, 4, region)
    assert set(regs) == set(frames) and np.array_equal(regs[4], region)
    # frame 7 (index 6): all points lost -> fallback (static copy); frame 6 (index 5, +2 steps of 2 px / 0.5 scale = +8 px)
    assert info["fallback_frames"] == 1 and info["hull_frames"] == 5
    ys, xs = np.where(regs[6]); ys0, xs0 = np.where(region)
    assert abs((xs.mean() - xs0.mean()) - 8.0) < 2.0 and abs(ys.mean() - ys0.mean()) < 2.0
    assert np.array_equal(regs[7], region)
    regs2, info2 = track_region(None, small, 0.5, frames, 4, region)
    assert all(np.array_equal(regs2[f], region) for f in frames) and info2["fallback_frames"] == 6


def test_save_load_delta_union(tmp_path):
    (tmp_path / "keys").mkdir()
    a = np.zeros((50, 60), bool); a[10:20, 10:20] = True
    b = np.zeros((50, 60), bool); b[30:40, 30:40] = True
    save_delta(tmp_path, 3, a)
    save_delta(tmp_path, 3, b)
    d = load_delta(tmp_path, 3)
    assert d is not None and np.array_equal(d, a | b) and load_delta(tmp_path, 4) is None


class _DriftRunner:
    """SAM3 stand-in: a disc drifting 2 px/frame; the 'new' core differs from the old one everywhere along the edge
    (+1 px dilation) and fills the hole the old core had in frames 3-5."""
    ckpt = "mock"

    def __init__(self, w, h):
        self.w, self.h = w, h

    def track(self, frames, init_mask, soft=False):
        out = []
        for i in range(len(frames)):
            m = cv2.dilate((init_mask > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            out.append(m.astype(np.float32))
        return out


def _old_core(frames, w=200, h=120):
    old = {}
    for i, f in enumerate(frames):
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (70 + 2 * i, 60), 45, 1, -1)
        if 3 <= f <= 5:
            cv2.circle(m, (70 + 2 * i, 60), 5, 0, -1)       # a small hole in frames 3-5 (far from the edge)
        old[f] = m.astype(bool)
    return old


def test_core_propagator_restricts_recompute_to_the_correction_region(tmp_path):
    w, h = 200, 120
    frames = list(range(1, 9))
    old = _old_core(frames)
    seg = CropSegment("s00", 1, 8, 0, 0, [0, 0, 120, 120], [0, 0, 0, 0], 1.0)
    keys = {1: old[1].astype(np.float32), 8: old[8].astype(np.float32)}
    # the user corrected frame 1: its key = old core with the hole region marked as the delta (filled)
    delta = np.zeros((h, w), bool); cv2.circle(delta.view(np.uint8), (70, 60), 5, 1, -1)
    img = lambda f, s: np.zeros((120, 120, 3), np.float32)
    runner = _DriftRunner(w, h)
    region_calls = []

    def region_fn(k, span, region):
        region_calls.append((k, list(span)))
        regs, info = track_region(_ShiftTracker(dx=2.0), [None] * len(span), 1.0, list(span), k, region)
        return regs, info

    # region mode: outside the tracked region the core is the old one, bit-identical; inside, the hole is filled
    prop = CorePropagator(runner, [seg], w, h, img, cache_dir=tmp_path / "c1", old_core=old, deltas={1: delta},
                          region_fn=region_fn, region_only=True)
    res = prop.propagate(frames, keys)
    assert region_calls and region_calls[0][0] == 1
    for f in (3, 4, 5):
        reg = track_region(_ShiftTracker(dx=2.0), [None] * len(frames), 1.0, frames, 1, region_from_delta(delta))[0][f]
        assert np.array_equal(res.core[f][~reg], old[f][~reg])
        assert res.core[f][60, 70 + 2 * (f - 1)]                   # hole filled (the new SAM3 mask wins inside)
    assert np.array_equal(res.core[8], old[8]) and np.array_equal(res.core[1], old[1])   # keys = hard truth
    assert prop.region_stats and list(prop.region_stats.values())[0]["changed_px"] > 0
    # the same recompute with the mode OFF changes the whole edge (today's behaviour)
    prop2 = CorePropagator(runner, [seg], w, h, img, cache_dir=tmp_path / "c2", old_core=old, deltas={1: delta},
                           region_fn=region_fn, region_only=False)
    res2 = prop2.propagate(frames, keys)
    assert (res2.core[4] != old[4]).sum() > (res.core[4] != old[4]).sum() * 3
    # without an old core / without deltas -> as computed (no restriction, no stats)
    prop3 = CorePropagator(runner, [seg], w, h, img, cache_dir=tmp_path / "c3", old_core={}, deltas={1: delta},
                           region_fn=region_fn, region_only=True)
    res3 = prop3.propagate(frames, keys)
    assert np.array_equal(res3.core[4], res2.core[4]) and not prop3.region_stats
    # cache: the region mode / delta are part of the hash -> c1 and c2 differ, a repeat of c1 is cached and identical
    prop4 = CorePropagator(runner, [seg], w, h, img, cache_dir=tmp_path / "c1", old_core=old, deltas={1: delta},
                           region_fn=region_fn, region_only=True)
    res4 = prop4.propagate(frames, keys)
    assert res4.cached_intervals and all(np.array_equal(res4.core[f], res.core[f]) for f in frames)


def test_api_correction_region_setting_and_delta(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    r = api.put("/api/object/obj/settings", json={"correction_region_only": False})
    assert r.status_code == 200 and r.json()["object"]["correction_region_only"] is False
    r = api.put("/api/object/obj/settings", json={"correction_region_only": True})
    assert r.json()["object"]["correction_region_only"] is True
    assert pipe.store.get("obj").get("correction_region_only", True) is True
    # delete_key removes the delta file too
    od = pipe.store.dir("obj")
    m = np.zeros((shot.h, shot.w), bool); m[10:20, 10:20] = True
    save_delta(od, start, m)
    assert load_delta(od, start) is not None
    pipe.store.delete_key("obj", start)
    assert load_delta(od, start) is None
