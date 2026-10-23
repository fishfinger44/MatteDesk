# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Band point edits (2026-10-23): the outline is read / traced once per frame, not once per point – same result."""
import cv2
import numpy as np

from engine.bandmap import _contours, snap_normal
from engine.bandtrack import inset_queries
from engine.pipeline import _read_core_png


def _disc():
    m = np.zeros((200, 240), np.uint8)
    cv2.circle(m, (120, 100), 60, 1, -1)
    return m.astype(bool)


def test_snap_normal_with_precomputed_contours_is_identical():
    c = _disc()
    cs = _contours(c)
    for x, y in [(60, 100), (180, 100), (120, 30), (100, 170), (5, 5)]:
        assert snap_normal(c, x, y, cs) == snap_normal(c, x, y)


def test_inset_queries_reads_each_key_frame_once():
    c = _disc()
    calls = []

    def core_at(k):
        calls.append(k)
        return c
    qs = [(f"p{i}", 1, 120 + 60 * np.cos(a), 100 + 60 * np.sin(a)) for i, a in enumerate(np.linspace(0, 6, 24))]
    qs += [("q", 5, 60.0, 100.0)]
    out = inset_queries(qs, core_at)
    assert sorted(calls) == [1, 5]
    # the same queries one point at a time (the old per-point read) give the same insets
    assert out == [inset_queries([q], lambda k: c)[0] for q in qs]


def test_read_core_png_cached_copy(tmp_path):
    p = tmp_path / "c_0001.png"
    cv2.imwrite(str(p), (_disc() * 255).astype(np.uint8))
    a = _read_core_png(p)
    a[:] = False                       # a caller writing into its mask never changes the cached one
    b = _read_core_png(p)
    assert b.sum() == _disc().sum()
    assert _read_core_png(tmp_path / "none.png") is None
