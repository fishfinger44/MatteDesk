# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""All ViTMatte tiles of a frame in ONE worker request (docs/PLAN_EDGE_SPEED.md A5, 2026-10-25): the alphas are the
ones the per-tile `run` gives (the worker runs the same forward per tile), tiles of different sizes and workers
without stacks (BiRefNet) fall back to one by one, and vitmatte_zone sums the tiles in the same order."""

import os
import sys

import numpy as np

from engine import birefnet, vitmatte as vm

FAKE_ENV = {**os.environ, "MD_VITMATTE_FAKE": "1"}


def _tiles(n=5, side=64, seed=0):
    rng = np.random.default_rng(seed)
    imgs, tris = [], []
    for k in range(n):
        imgs.append(rng.random((side, side, 3)).astype(np.float32))
        t = np.full((side, side), 128, np.uint8)
        t[: 10 + k] = 255
        t[-8:] = 0
        tris.append(t)
    return imgs, tris


def test_run_many_equals_run_per_tile_and_one_request():
    vm._CHECK_CACHE.clear()
    r = vm.ComfyVitMatteRunner("hustvl/fake", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    imgs, tris = _tiles()
    single = [r.run(i, t, 0.7) for i, t in zip(imgs, tris)]
    n0 = r._n
    many = r.run_many(imgs, tris, 0.7)
    assert r._n == n0 + 1                                # one request for the whole frame
    assert len(many) == len(single) and all(np.array_equal(a, b) for a, b in zip(single, many))
    # different tile sizes -> one by one, same values
    imgs2, tris2 = imgs[:2] + [imgs[2][:32, :32]], tris[:2] + [tris[2][:32, :32]]
    out = r.run_many(imgs2, tris2, 0.7)
    assert all(np.array_equal(a, r.run(i, t, 0.7)) for a, i, t in zip(out, imgs2, tris2))
    assert r.run_many([], [], 0.7) == []
    r.close()


def test_in_process_runner_and_birefnet_loop(monkeypatch):
    calls = []

    class R(vm.VitMatteRunner):
        def run(self, img, tri, unknown_value=0.5):
            calls.append(unknown_value)
            return np.full(tri.shape, unknown_value, np.float32)
    imgs, tris = _tiles(3)
    out = R("x").run_many(imgs, tris, 0.6)
    assert len(out) == 3 and calls == [0.6] * 3
    assert birefnet.ComfyBirefnetRunner.batched is False
    monkeypatch.setattr(birefnet.ComfyBirefnetRunner, "run", lambda self, i, t, u=0.5: np.zeros(t.shape, np.float32))
    b = birefnet.ComfyBirefnetRunner("ZhengPeng7/fake", python=sys.executable, env=FAKE_ENV)
    assert len(b.run_many(imgs, tris)) == 3


def test_vitmatte_zone_tiles_use_run_many_in_grid_order(monkeypatch):
    """vitmatte_zone (outline tiles) hands the frame's tiles to run_many and sums them as before (bit-identical)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_zone_vitmatte import MockRunner, _frames_scene
    from engine.zones import Zone, vitmatte_zone, zone_mask_at
    img, core, hair, region, frames = _frames_scene(3)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params={"region": "outline"})
    cores = {f: core for f in frames}
    regions = {f: zone_mask_at(z, core.shape) for f in frames}
    per_tile = MockRunner()
    a1, _, s1 = vitmatte_zone(z, frames, cores, regions, lambda f: img, per_tile, tile_px=48, model_px=48)

    class Many(MockRunner):
        requests = 0

        def run_many(self, imgs, tris, unknown_value=0.5):
            Many.requests += 1
            return [self.run(i, t) for i, t in zip(imgs, tris)]
    a2, _, s2 = vitmatte_zone(z, frames, cores, regions, lambda f: img, Many(), tile_px=48, model_px=48)
    assert s1["tiles"] == s2["tiles"] > 3 and Many.requests == len(frames)     # one request per frame
    assert all(np.array_equal(a1[f], a2[f]) for f in frames)


def test_phase_timings_a6():
    """Plan A6 (2026-10-28): the worker reports its own phases, the runner splits a request into prepare / npy / worker
    / ipc_wait / finish, and vitmatte_zone puts the frame-loop phases + the runner's delta into its stats."""
    vm._CHECK_CACHE.clear()
    r = vm.ComfyVitMatteRunner("hustvl/fake", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    imgs, tris = _tiles()
    before = vm.prof_snapshot(r)
    r.run_many(imgs, tris)
    d = vm.prof_delta(r, before)
    for k in ("prepare", "npy_write", "worker_read", "worker_forward", "worker_load", "worker_write", "ipc_wait",
              "npy_read", "finish"):
        assert k in d and d[k] >= 0.0 or k == "ipc_wait", (k, d)
    r.close()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_zone_vitmatte import MockRunner, _frames_scene
    from engine.zones import Zone, vitmatte_zone, zone_mask_at
    img, core, hair, region, frames = _frames_scene(3)
    z = Zone("z_01", "add", "matte", 1, polygon=[[0, 0], [10, 0], [10, 10]], matting="vitmatte")
    regions = {f: region for f in frames}
    for tile in (0, 48):
        _, _, st = vitmatte_zone(z, frames, {f: core for f in frames}, regions, lambda f: img, MockRunner(),
                                 tile_px=tile, model_px=48)
        ph = st["phases"]
        assert {"masks", "decode", "trimap", "tile_image", "model", "assemble", "trimap_full", "other"} <= set(ph), ph
        assert st["runner_phases"] == {}                      # mock without `prof`
        assert abs(sum(ph.values()) - st["wall_seconds"]) < 0.2
