# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Soft start of the edge band from the SAM3 probability maps (user decision 2026-10-14):
(1) MatAnyone – CoreCombine (>= 1.1.2, core_mode=soft) gets the stabilised probabilities instead of the 0/1 core;
(2) ViTMatte zone – trimap from SAM3's uncertainty (FG p >= 0.95, unknown 0.05 < p < 0.95 + zone + core rim).
Frames without a map / key frames = as before; no maps at all = the old graphs, files and alphas bit for bit."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.corestab import consistent_soft, stabilize_core
from engine.localmatte import build_trimap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_core_soft import _pan_probs, _propagate  # noqa: E402
from test_zone_vitmatte import MockRunner, _frames_scene  # noqa: E402

CC_NEW = {"MatAnyone2CoreCombine": {"input": {"required": {}, "optional": {"core_mode": [["binary", "soft"], {}]}}}}
CC_OLD = {"MatAnyone2CoreCombine": {"input": {"required": {}}}}


# ------------------------------------------------------------------------------------------- corestab
def test_consistent_soft_follows_the_core():
    p = np.array([[0, 100, 127, 128, 200, 255]], np.uint8)
    c = np.array([[1, 1, 0, 0, 1, 0]], bool)
    q = consistent_soft(p, c)
    assert q.tolist() == [[255, 255, 127, 0, 200, 0]] and ((q >= 128) == c).all()
    assert p.tolist() == [[0, 100, 127, 128, 200, 255]]          # input untouched


def test_stabilize_core_probs_out_consistent_keys_hard_and_optional():
    cores, probs, truth, grays = _pan_probs()
    po = {}
    out, st = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, probs=probs, keys=[3], probs_out=po)
    out0, _ = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, probs=probs, keys=[3])
    assert all(np.array_equal(out[f], out0[f]) for f in cores)    # probs_out changes nothing in the core
    assert sorted(po) == [f for f in sorted(cores) if f != 3]      # key frame: no soft map (stays hard)
    for f, p in po.items():
        assert p.dtype == np.uint8 and np.array_equal(p >= 128, out[f])
        assert ((p > 12) & (p < 243)).sum() > 100                    # a real soft band survives
    # jittered frame 5: the stabilised probabilities moved the edge back towards the truth
    d_raw = np.abs(probs[5].astype(np.float32) / 255 - truth[5]).mean()
    d_st = np.abs(po[5].astype(np.float32) / 255 - truth[5]).mean()
    assert d_st < d_raw, (d_st, d_raw)
    # binary mode: nothing is produced
    po2 = {}
    stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, probs_out=po2)
    assert po2 == {}


# ------------------------------------------------------------------------------------------- MatAnyone (pipeline)
def _ma_dirs(pipe):
    return sorted((pipe.work / "comfy_io").glob("ma_*_final")) if hasattr(pipe, "work") else []


def _ma_graphs(client):
    return [g for g in client.sent if any(n.get("class_type") == "MatAnyone2CoreCombine" for n in g.values())]


def _cc(g):
    return next(n for n in g.values() if n.get("class_type") == "MatAnyone2CoreCombine")["inputs"]


def _core_files(g):
    pat = next(n for n in g.values() if n.get("class_type") == "PlateLoadMaskSeq"
               and "c_" in str(n["inputs"].get("path_pattern", "")))["inputs"]["path_pattern"]
    d = Path(pat).parent
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(d.glob("c_*.png"))]


def _prep(tmp_path, soft, info, **opts):
    shot, pipe, frames, start = _propagate(tmp_path, soft)
    pipe.client._object_info = {**pipe.client._object_info, **info}
    pipe.run_core("obj", frames)
    if opts:
        pipe.store.update("obj", **opts)
    return pipe, frames, start


def test_finalize_matanyone_soft_start(tmp_path):
    pipe, frames, start = _prep(tmp_path / "a", True, CC_NEW)
    full, rep, run_dir = pipe.finalize("obj", frames, edge_mode="matanyone")
    gs = _ma_graphs(pipe.client)
    final = [g for g in gs if _cc(g).get("core_mode") == "soft"]
    assert final, "final pass without core_mode=soft"
    files = _core_files(final[-1])
    soft_files = [c for c in files if ((c > 2) & (c < 253)).any()]
    assert soft_files and len(soft_files) < len(files) + 1
    # every file is either strictly binary (key frame / no map) or a probability map with values >= 2
    for c in files:
        assert not ((c > 0) & (c < 2)).any()
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ss = rj["soft_start"]
    assert ss["node"] is True and ss["maps"] == len(frames) - 1 and ss["frames"] >= len(frames) - 1   # key = hard
    assert rj.get("core_stabilize") is True


def test_finalize_matanyone_old_node_or_option_off_stays_binary(tmp_path):
    pipe, frames, _ = _prep(tmp_path / "old", True, CC_OLD)
    _, _, run_dir = pipe.finalize("obj", frames, edge_mode="matanyone")
    assert all("core_mode" not in _cc(g) for g in _ma_graphs(pipe.client))
    ss = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["soft_start"]
    assert ss["node"] is False and ss["frames"] == 0 and "Doktor" in ss["note"]
    for g in _ma_graphs(pipe.client):
        assert all(set(np.unique(c)) <= {0, 255} for c in _core_files(g))
    pipe2, frames2, _ = _prep(tmp_path / "off", True, CC_NEW, core_soft_start=False)
    _, _, run_dir2 = pipe2.finalize("obj", frames2, edge_mode="matanyone")
    assert all("core_mode" not in _cc(g) for g in _ma_graphs(pipe2.client))
    assert json.loads((run_dir2 / "run.json").read_text(encoding="utf-8"))["soft_start"] == {}


def test_no_maps_bit_identical_to_maps_with_soft_start_off(tmp_path):
    """Without soft maps the pass is the old one: same graphs (no core_mode), same core files, same alpha as a shot
    WITH maps but soft start and probability stabilisation off (the binary core is identical with / without maps)."""
    pa, frames, _ = _prep(tmp_path / "plain", False, CC_NEW, core_stabilize=False)
    pb, _, _ = _prep(tmp_path / "maps_off", True, CC_NEW, core_stabilize=False, core_soft_start=False)
    fa, _, ra = pa.finalize("obj", frames, edge_mode="matanyone")
    fb, _, rb = pb.finalize("obj", frames, edge_mode="matanyone")
    for f in frames:
        assert np.array_equal(fa[f], fb[f])
    ga, gb = _ma_graphs(pa.client), _ma_graphs(pb.client)
    assert len(ga) == len(gb) and all("core_mode" not in _cc(g) for g in ga + gb)
    for x, y in zip(ga, gb):
        assert all(np.array_equal(c1, c2) for c1, c2 in zip(_core_files(x), _core_files(y)))
    assert json.loads((ra / "run.json").read_text(encoding="utf-8"))["soft_start"] == {}


def test_soft_start_follows_shrink_grow(tmp_path):
    """The maps passed to MatAnyone always agree with the core actually used (shift_px wins with a hard 0 / 1)."""
    pipe, frames, start = _prep(tmp_path / "shift", True, CC_NEW, shift_px=-2.0)
    pipe.finalize("obj", frames, edge_mode="matanyone")
    g = [g for g in _ma_graphs(pipe.client) if _cc(g).get("core_mode") == "soft"][-1]
    files = _core_files(g)
    rdir = pipe.store.dir("obj") / "runs"
    run = sorted(rdir.iterdir())[-1]
    seg = sorted((run / "core").iterdir())[0]
    cores = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127 for p in sorted(seg.glob("c_*.png"))]
    for c, core in zip(files, cores):
        if c.shape == core.shape:
            assert np.array_equal(c >= 128, core)


# ------------------------------------------------------------------------------------------- ViTMatte zone trimap
def test_build_trimap_from_probabilities():
    core = np.zeros((40, 40), bool)
    core[10:30, 10:30] = True
    yy, xx = np.mgrid[0:40, 0:40]
    d = 10 - np.maximum(np.abs(yy - 19.5), np.abs(xx - 19.5))
    p = (1 / (1 + np.exp(-d / 1.0))).astype(np.float32)
    region = np.zeros_like(core)
    region[0:40, 25:40] = True
    forced = np.zeros_like(core)
    forced[0:5, 0:5] = True
    old = build_trimap(core, region, forced, 2)
    new = build_trimap(core, region, forced, 2, prob=p)
    assert np.array_equal(new.fg, core & (p >= 0.95))
    assert np.array_equal(new.unknown, (region | core | ((p > 0.05) & (p < 0.95))) & ~new.fg & ~new.forced_bg)
    assert new.unknown[(p > 0.05) & (p < 0.95) & ~core & ~region].all()      # SAM3's uncertain outer band
    assert not np.array_equal(old.fg, new.fg)
    assert np.array_equal(build_trimap(core, region, forced, 2, prob=None).fg, old.fg)


def test_vitmatte_zone_soft_trimap_cache_and_param(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    from engine.zonecache import ZoneCache
    from engine.zones import Zone, apply_zones
    img, core, hair, region, frames = _frames_scene(4)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")
    # soft maps = blurred binary core (as the SAM3 node on the mock), consistent with the core
    blur = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.0)
    p8 = consistent_soft((blur * 255 + 0.5).astype(np.uint8), core)
    probs = {f: p8 for f in frames[1:]}                  # first frame: no map (e.g. a key) -> old trimap

    def run(params=None, pr=None):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params={"temporal": "none", **(params or {})})
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache, probs=pr)
    r_old = run()
    calls_old = list(runner.calls)
    assert r_old.stats["z_01"]["computed"] == 4 and r_old.stats["z_01"]["soft_trimap_frames"] == 0
    r_soft = run(pr=probs)
    st = r_soft.stats["z_01"]
    # frames with a map: new hash -> recomputed with the soft trimap; the frame without one: from the cache
    assert st["computed"] == 3 and st["cached"] == 1 and st["soft_trimap_frames"] == 3
    new_calls = runner.calls[len(calls_old):]
    assert all(c[2] != calls_old[0][2] for c in new_calls)      # FG area differs (p >= 0.95 instead of the erosion)
    assert np.array_equal(r_soft.soft[frames[0]][0], r_old.soft[frames[0]][0])
    # second soft pass: all cached; soft_trimap=False -> the old hashes (cached old results), no model call
    n = len(runner.calls)
    assert run(pr=probs).stats["z_01"]["computed"] == 0 and len(runner.calls) == n
    r_off = run({"soft_trimap": False}, pr=probs)            # explicit param -> its own signature, old trimaps
    assert r_off.stats["z_01"]["computed"] == 4 and r_off.stats["z_01"]["soft_trimap_frames"] == 0
    for f in frames:
        assert np.array_equal(r_off.soft[f][0], r_old.soft[f][0])
    r_none = run(pr=None)
    assert r_none.stats["z_01"]["computed"] == 0
    for f in frames:
        assert np.array_equal(r_none.soft[f][0], r_old.soft[f][0])
