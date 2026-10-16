# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z12 – the interval cache of the core depends on the algorithm version (CORE_ALGO_VERSION).

After the blend fix (4f74fb4) an old, empty result in cache/1-20 would have been reused (same keys, crops, params,
checkpoint). Bumping the constant must force the interval to be recomputed instead of read from cache.
"""

import json

import numpy as np

import engine.segment as segment
from engine.crop import CropSegment
from engine.segment import CorePropagator


class _CountingRunner:
    ckpt = "sam3.st"

    def __init__(self):
        self.calls = 0

    def track(self, imgs, init):
        self.calls += 1
        return [(init > 0.5).astype(np.float32) for _ in imgs]


def _prop(runner, tmp_path, w=64):
    seg = CropSegment(seg_id="s0", first=1, last=10, overlap_in=0, overlap_out=0, crop=[0, 0, w, w], pad=[0, 0, 0, 0],
                      scale_to_1008=1.0)
    return CorePropagator(runner, [seg], w, w, load_crop_image=lambda f, s: np.zeros((s.side, s.side, 3), np.float32),
                          cache_dir=tmp_path / "cache", params={"engine": "sam3", "plate": "p_####.exr"})


def test_algo_version_invalidates_interval_cache(tmp_path, monkeypatch):
    k = np.zeros((64, 64), bool)
    k[20:40, 20:40] = True
    keys = {1: k, 10: k}
    frames = list(range(1, 11))
    r = _CountingRunner()
    res = _prop(r, tmp_path).propagate(frames, keys)
    assert res.recomputed_intervals == [(1, 10)] and r.calls > 0
    meta = json.loads((tmp_path / "cache" / "1-10" / "hash.json").read_text())
    assert meta["core_algo"] == segment.CORE_ALGO_VERSION
    # same inputs, same version -> from cache, SAM3 not called
    r2 = _CountingRunner()
    res2 = _prop(r2, tmp_path).propagate(frames, keys)
    assert res2.cached_intervals == [(1, 10)] and res2.recomputed_intervals == [] and r2.calls == 0
    # new algorithm version -> recomputed, not read from cache
    monkeypatch.setattr(segment, "CORE_ALGO_VERSION", segment.CORE_ALGO_VERSION + 1)
    r3 = _CountingRunner()
    res3 = _prop(r3, tmp_path).propagate(frames, keys)
    assert res3.recomputed_intervals == [(1, 10)] and res3.cached_intervals == [] and r3.calls > 0
    assert json.loads((tmp_path / "cache" / "1-10" / "hash.json").read_text())["core_algo"] == segment.CORE_ALGO_VERSION
