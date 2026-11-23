"""The C1 patch band preview reads the removal masks at the lines grid from a cache (user 2026-11-23: 4-8 s per frame
on a 6K group with MiniMax-H3's token of ~9 masks), and the lines are the same as from the full-resolution masks."""
import time

import numpy as np

import api.inpaint_band as AB
from engine import inpaint_band as IB


class FakeHold:
    """A removal loader with a stamp like engine/cleanup_hold.Hold."""

    def __init__(self, w, h, key="k1"):
        self.w, self.h, self.key, self.calls = w, h, key, []

    def cached(self):
        return None

    def __call__(self, f):
        self.calls.append(int(f))
        a = np.zeros((self.h, self.w), np.float32)
        r = 40 + 3 * int(f)
        yy, xx = np.ogrid[:self.h, :self.w]
        a[(xx - self.w // 2) ** 2 + (yy - self.h // 2) ** 2 < r * r] = 1.0
        return a


def setup_function():
    AB._SMALL.clear()


def test_small_alpha_is_read_once_per_frame_and_stamp():
    fn = FakeHold(640, 400)
    size = (320, 200)
    a = AB.small_alpha(fn, 3, size)
    b = AB.small_alpha(fn, 3, size)
    assert fn.calls == [3] and a.shape == (200, 320) and np.array_equal(a, b)
    fn.key = "k2"                                            # the mask changed (new stamp): read again
    AB.small_alpha(fn, 3, size)
    assert fn.calls == [3, 3]


def test_no_stamp_no_cache():
    fn = FakeHold(64, 40, key=None)
    AB.small_alpha(fn, 1, (32, 20))
    AB.small_alpha(fn, 1, (32, 20))
    assert fn.calls == [1, 1]


def test_small_keeps_the_mask_threshold():
    a = np.full((8, 8), 0.021, np.float32)
    out = AB.small_alpha(type("L", (), {"key": "t", "cached": lambda self: None, "__call__": lambda self, f: a})(), 0, (8, 8))
    assert (out > 0.02).all()                                # ceil: a pixel just over 0.02 stays in the mask


def test_lines_from_the_cached_grid_match_the_full_resolution():
    fn = FakeHold(1200, 800)
    full = [fn(5)]
    lim = [fn(k) for k in (4, 5, 6)]
    scale = 0.5
    size = (600, 400)
    ref = IB.lines(full, 20.0, 10.0, None, scale, 30.0, limit_alphas=lim)
    got = IB.lines([AB.small_alpha(fn, 5, size)], 20.0, 10.0, None, scale, 30.0,
                   limit_alphas=[AB.small_alpha(fn, k, size) for k in (4, 5, 6)], full_size=(1200, 800))

    def pts(L, key):
        return np.array([p for c in L[key] for p in c], np.float32)
    for key in ("band_contours", "feather_contours", "limit_contours"):
        a, b = pts(ref, key), pts(got, key)
        assert len(a) and len(b)
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
        assert d.max() <= 2.5, key                           # plate px (one grid px = 2 plate px here)
    assert got["cut"] == ref["cut"]


def test_warm_reads_the_rest_of_the_range_in_the_background():
    fn = FakeHold(64, 40)
    AB._warm([fn], [1, 2, 3, 4], (32, 20))
    for _ in range(200):
        if len(AB._SMALL) >= 4:
            break
        time.sleep(0.01)
    assert sorted(set(fn.calls)) == [1, 2, 3, 4]
    n = len(fn.calls)
    AB.small_alpha(fn, 2, (32, 20))
    assert len(fn.calls) == n                                # from the cache


def test_endpoint_uses_the_cache():
    src = (AB.__file__ and open(AB.__file__, encoding="utf-8").read())
    assert "sal = (lambda k: [small_alpha(fn, int(k), size) for fn in rm])" in src
    assert "_warm(rm, wf, size)" in src
