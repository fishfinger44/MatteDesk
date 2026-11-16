# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inpaint / Cleanup: the frames of a batch are prepared by several CPU threads and written by two (user 2026-11-14);
the files are bit-identical to the one-thread run, in every output folder."""
from pathlib import Path

import cv2
import numpy as np

from engine import cleanup as cu

W, H = 160, 120


class _Fill:
    """A video model's frames: noise seeded by the frame, a small level error (levelling and grain on)."""
    label = "MiniMax-H3"
    tone = True
    tone_gap_px = 4.0

    def __call__(self, f):
        rng = np.random.default_rng(f)
        img = (0.4 + 0.1 * rng.random((H, W, 3))).astype(np.float32) * 1.03
        return img, np.ones((H, W), np.float32)


def _run(out: Path, workers: int):
    yy, xx = np.mgrid[0:H, 0:W]

    def plate(f):
        rng = np.random.default_rng(100 + f)
        return (0.4 + 0.12 * rng.random((H, W, 3))).astype(np.float32)

    def obj(f):
        return (np.hypot(xx - 60 - f, yy - 60) < 20).astype(np.float32)
    keep = ((xx > 110) & (xx < 140)).astype(np.float32)
    c = cu.Cleaner({"grow_px": 4, "soft_px": 3, "grain": "on", "tone": True, "batch": 5}, W, H, False, plate=plate,
                   remove_alphas=lambda f: [obj(f)], keep_alphas=lambda f: [keep], key_fill=_Fill())
    c.workers = workers
    rj = c.run(list(range(1, 13)), out, "png16")
    return rj


def test_parallel_frames_are_bit_identical(tmp_path):
    a = _run(tmp_path / "one", 1)
    b = _run(tmp_path / "many", 4)
    assert a["counts"] == b["counts"] and a.get("grain", {}).get("added") == b.get("grain", {}).get("added")
    for sub in ("comp", "patch", "source"):
        fa = sorted((tmp_path / "one" / sub).iterdir())
        fb = sorted((tmp_path / "many" / sub).iterdir())
        assert [p.name for p in fa] == [p.name for p in fb] and len(fa) == 12
        for x, y in zip(fa, fb):
            assert np.array_equal(cv2.imread(str(x), -1), cv2.imread(str(y), -1)), (sub, x.name)


def test_worker_counts():
    assert 1 <= cu.FRAME_WORKERS <= 4 and cu.WRITE_WORKERS == 2
