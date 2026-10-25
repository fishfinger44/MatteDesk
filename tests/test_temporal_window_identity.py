# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract of the sliding-window temporal smoothers (docs/PLAN_EDGE_SPEED.md A1-A3, 2026-10-25): bit-identical to
the frozen stack versions in tests/_ref_temporal_old.py – for every radius, for the plain / flow / mean smoothers,
with and without a band, with the flow-confidence options, sequential and with a thread pool, from a stack and from a
lazy sequence, returned as a stack and emitted frame by frame. Speed is never a reason to change a value
(user rule: quality over speed)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from engine import refine  # noqa: E402
import _ref_temporal_old as old  # noqa: E402

_REAL_DIS_CREATE = cv2.DISOpticalFlow_create


def _scene(n=9, h=40, w=56, seed=0):
    """Moving textured strands over a drifting background: the flows matter and differ between pairs."""
    rng = np.random.default_rng(seed)
    bg = rng.uniform(0.2, 0.8, (h + 8, w + 8)).astype(np.float32)
    grays, alphas = [], []
    for t in range(n):
        dx, dy = (t * 3) % 7, (t * 2) % 5
        g = bg[dy:dy + h, dx:dx + w].copy()
        a = np.zeros((h, w), np.float32)
        a[8 + t:20 + t, 10:30] = 0.6
        a[5:35, 36 + (t % 3)] = 0.9
        a += rng.normal(0, 0.03, (h, w)).astype(np.float32)
        if t == 4:
            a[30:36, 44:50] = 0.5                     # garbage in one frame
        grays.append(g)
        alphas.append(np.clip(a, 0, 1))
    return np.stack(alphas), np.stack(grays)


class _Lazy:
    """Frames computed on demand (what the zone pipeline hands over instead of a stack); counts the getter calls."""

    def __init__(self, stack):
        self.stack = stack
        self.calls = 0

    def __len__(self):
        return len(self.stack)

    def __getitem__(self, i):
        self.calls += 1
        return self.stack[i].copy()


def _collect(fn, *a, **kw):
    got = {}
    r = fn(*a, emit=lambda i, f: got.__setitem__(i, f.copy()), **kw)
    assert r is None
    return np.stack([got[i] for i in range(len(got))])


@pytest.mark.parametrize("radius", [1, 2, 3, 5])
@pytest.mark.parametrize("band", [False, True])
def test_temporal_median_identical(radius, band):
    a, _ = _scene()
    b = (a > 0.3) if band else None
    ref = old.temporal_median(a, radius, 0.35, b)
    assert np.array_equal(ref, refine.temporal_median(a, radius, 0.35, b))
    assert np.array_equal(ref, refine.temporal_median(_Lazy(a), radius, 0.35, _Lazy(b) if band else None))
    assert np.array_equal(ref, _collect(refine.temporal_median, a, radius, 0.35, b))


@pytest.mark.parametrize("radius", [1, 2, 5])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("band", [False, True])
def test_temporal_median_flow_identical(radius, workers, band):
    a, g = _scene(11)
    b = (a > 0.3) if band else None
    ref = old.temporal_median_flow(a, g, radius, b)
    assert np.array_equal(ref, refine.temporal_median_flow(a, g, radius, b, workers=workers))
    lazy_a, lazy_g = _Lazy(a), _Lazy(g)
    assert np.array_equal(ref, refine.temporal_median_flow(lazy_a, lazy_g, radius, b, workers=workers))
    assert lazy_a.calls == len(a) and lazy_g.calls == len(g)        # every frame fetched once (window cache)
    assert np.array_equal(ref, _collect(refine.temporal_median_flow, a, g, radius, b, workers=workers))


@pytest.mark.parametrize("radius", [2, 3, 5])
@pytest.mark.parametrize("workers", [1, 4])
def test_temporal_mean_flow_identical(radius, workers):
    a, g = _scene(11, seed=3)
    ref = old.temporal_mean_flow(a, g, radius)
    assert np.array_equal(ref, refine.temporal_mean_flow(a, g, radius, workers=workers))
    assert np.array_equal(ref, refine.temporal_mean_flow(_Lazy(a), _Lazy(g), radius, workers=workers))
    assert np.array_equal(ref, _collect(refine.temporal_mean_flow, a, g, radius, workers=workers))


class _PerSizeDIS:
    """cv2.DISOpticalFlow_create stand-in: one instance per image size. OpenCV's DIS keeps buffers between calls and a
    flow computed after one at another size differs (0.27 px on the test scene, 2026-10-25); the old `multiscale`
    option ran full- and half-size flows on ONE instance, so its result depended on the call order. The new code uses
    per-size instances; the reference for that option is the old code with the same separation."""

    def __init__(self, preset):
        self.preset, self.d = preset, {}

    def calc(self, a, b, flow):
        d = self.d.get(a.shape)
        if d is None:
            d = self.d[a.shape] = _REAL_DIS_CREATE(self.preset)
        return d.calc(a, b, flow)


def test_dis_is_stateful_across_sizes_so_instances_are_per_size():
    """Documents the OpenCV behaviour the per-size instances work around (if this ever passes, they can go)."""
    _, g = _scene(3, seed=5)
    u = [cv2.normalize(x.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for x in g]
    half = [cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA) for x in u]
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.calc(half[0], half[1], None)
    after_half = d.calc(u[0], u[1], None)
    fresh = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u[0], u[1], None)
    assert not np.array_equal(after_half, fresh)
    pool = refine._FlowPool(lambda i: u[i], cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, 1)
    pool.dis(half[0].shape).calc(half[0], half[1], None)
    assert np.array_equal(pool.get(0, 1), fresh)      # per-size instance: equal to a fresh one whatever ran before


@pytest.mark.parametrize("kw", [dict(fb_sigma=1.0), dict(photo_sigma=0.1), dict(tex_k=0.01),
                                dict(multiscale=2), dict(fb_sigma=0.5, conf_mode="centre"),
                                dict(fb_sigma=1.0, photo_sigma=0.2, tex_k=0.02, multiscale=2),
                                dict(sigma_t=1.0, tol=0.1, occl_px=0.5)])
def test_temporal_mean_flow_confidence_identical(kw, monkeypatch):
    a, g = _scene(9, seed=5)
    if kw.get("multiscale", 1) > 1:
        monkeypatch.setattr(old.cv2, "DISOpticalFlow_create", _PerSizeDIS)
    ref = old.temporal_mean_flow(a, g, 3, **kw)
    monkeypatch.undo()
    assert np.array_equal(ref, refine.temporal_mean_flow(a, g, 3, workers=4, **kw))
    assert np.array_equal(ref, _collect(refine.temporal_mean_flow, _Lazy(a), _Lazy(g), 3, workers=1, **kw))


def test_short_sequences_and_dtype_as_before():
    a, g = _scene(2)
    assert np.array_equal(old.temporal_mean_flow(a, g, 2), refine.temporal_mean_flow(a, g, 2))
    assert np.array_equal(old.temporal_median_flow(a, g, 2), refine.temporal_median_flow(a, g, 2))
    a64 = _scene(5)[0].astype(np.float64)
    r = refine.temporal_median(a64, 1, 0.35)
    assert r.dtype == np.float64 and np.array_equal(r, old.temporal_median(a64, 1, 0.35))


def test_progress_called_per_frame_and_cancel_stops_early():
    a, g = _scene(7)
    seen = []
    refine.temporal_mean_flow(a, g, 2, progress=lambda i, n: seen.append((i, n)), workers=2)
    assert seen == [(i, 7) for i in range(1, 8)]

    class Stop(RuntimeError):
        pass

    def cancel(i, n):
        if i == 3:
            raise Stop()
    emitted = []
    with pytest.raises(Stop):
        refine.temporal_median_flow(a, g, 2, progress=cancel, emit=lambda i, f: emitted.append(i), workers=2)
    assert emitted == [0, 1]                       # frames before the cancel were handed out, nothing after


def test_window_is_the_only_thing_kept():
    """The lazy source is asked for frames i-r..i+r (+ the next frame's window), never kept beyond the window."""
    a, g = _scene(13)
    la = _Lazy(a)
    held = []
    cache_holder = {}
    orig = refine._FrameCache.__call__

    def spy(self, i):
        v = orig(self, i)
        cache_holder[id(self)] = self
        held.append(max(len(c._d) for c in cache_holder.values()))
        return v
    try:
        refine._FrameCache.__call__ = spy
        refine.temporal_mean_flow(la, _Lazy(g), 2, workers=1)
    finally:
        refine._FrameCache.__call__ = orig
    assert max(held) <= 2 * 2 + 2                   # window of 5 (+ the prefetched next frame)


def test_median_odd_other_counts_exact():
    rng = np.random.default_rng(0)
    for n in (7, 9, 11, 13):
        win = rng.random((n, 20, 30), np.float32)
        assert np.array_equal(refine._median_odd(win), np.median(win, axis=0))
