# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""`vitmatte` zone engine (experiment 2026-10-13, user decision: the CLAUDE.md ban is lifted for this comparison):
ViTMatte per frame on the up-scaled union crop of the zone (as `matanyone`), trimap from the core / region / forced BG,
result only inside the zone, per-frame cache shared by the temporal options none / median / flow, DIS-flow median."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engine.localmatte import LocalMatteResult, local_matte  # noqa: F401
from engine.refine import temporal_median, temporal_median_flow
from engine.zonecache import ZoneCache
from engine.zones import Zone, apply_zones, vitmatte_zone, zone_mask_at

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


class MockRunner:
    """Stands in for engine.vitmatte.VitMatteRunner: FG / BG of the trimap forced, unknown = warmth of the pixel
    (R - B, the synthetic hair is warm on a bluish background). Records every call."""
    model_id = "mock"
    device = "cpu"
    load_seconds = 0.0

    def __init__(self):
        self.calls = []

    def run(self, img, tri, unknown_value=0.5):
        self.calls.append((img.shape, tri.shape, int((tri == 255).sum()), int((tri == 128).sum())))
        a = np.clip(4.0 * (img[..., 2] - img[..., 0]), 0, 1).astype(np.float32)
        a[tri == 255] = 1.0
        a[tri == 0] = 0.0
        return a


def _frames_scene(n=5):
    from tests.test_localmatte import _motley, _scene
    img, core, hair, line, region = _scene(_motley(192, 256))
    return img, core.astype(bool), hair.astype(bool), region.astype(bool), list(range(1, n + 1))


def test_vitmatte_zone_crop_trimap_and_masks():
    img, core, hair, region, frames = _frames_scene(3)
    H, W = core.shape
    cores = {f: core for f in frames}
    regions = {f: region for f in frames}
    ys, xs = np.where(region & ~core)
    hold = np.zeros((H, W), bool); hold[ys[:40], xs[:40]] = True          # 40 region pixels = a front object
    bgm = np.zeros((H, W), bool); bgm[ys[-40:], xs[-40:]] = True          # 40 region pixels = "-" strokes
    ys, xs = np.where(region)
    z = Zone("z_01", "add", "matte", 2, polygon=[[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())],
                                                [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]], matting="vitmatte")
    runner = MockRunner()
    out, tris, st = vitmatte_zone(z, frames, cores, regions, lambda f: img, runner, holdout=lambda f: hold, bg=lambda f: bgm)
    assert len(runner.calls) == 3 and st["engine"] == "vitmatte" and st["frames"] == 3
    assert 1.0 < st["scale"] <= 3.0 and st["model_px"] == runner.calls[0][0][0] == runner.calls[0][1][0]   # upscaled crop
    assert runner.calls[0][2] > 0 and runner.calls[0][3] > 0                                              # FG and unknown present
    for f in frames:
        a = out[f]
        assert a.shape == (H, W) and a[~region].max() == 0.0
        assert a[region & core].min() == 1.0
        assert a[hold & region & ~core].max() == 0.0 and a[bgm & region & ~core].max() == 0.0
        assert a[hair & region & ~core & ~hold & ~bgm].mean() > 0.6                                     # warm strands kept
        t = tris[f]
        assert t.dtype == np.uint8 and set(np.unique(t)) <= {0, 128, 255}
        assert (t[core] == 255).mean() > 0.9 and (t[region & ~core] == 128).mean() > 0.9
        assert t[~region & ~core].max() in (0, 128)       # outside: BG, except the core rim (unknown)
    # `only` computes a subset
    out2, _, st2 = vitmatte_zone(z, frames, cores, regions, lambda f: img, runner, only=[2])
    assert list(out2) == [2] and st2["frames"] == 1


def test_apply_zones_vitmatte_cache_shared_by_temporal_options(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    img, core, hair, region, frames = _frames_scene(5)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")

    def run(temporal):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params={"temporal": temporal})
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache)
    r0 = run("none")
    s0 = r0.stats["z_01"]
    assert s0["engine"] == "vitmatte" and s0["temporal"] == "none" and s0["computed"] == 5 and s0["cached"] == 0
    assert len(runner.calls) == 5
    area = zone_mask_at(Zone("z_01", "add", "matte", 1, polygon=poly), core.shape)
    for f in frames:
        a, ar = r0.soft[f]
        assert np.array_equal(ar, area) and a[~area].max() == 0.0 and a[area & core].min() == 1.0
        assert a[hair & area & ~core].mean() > 0.6
    # the same frames come from the cache (bit-identical) – with any temporal option
    r1 = run("none")
    assert r1.stats["z_01"]["computed"] == 0 and r1.stats["z_01"]["cached"] == 5 and len(runner.calls) == 5
    for f in frames:
        assert np.array_equal(r1.soft[f][0], r0.soft[f][0])
    r2 = run("median")
    assert r2.stats["z_01"]["computed"] == 0 and r2.stats["z_01"]["temporal"] == "median" and len(runner.calls) == 5
    r3 = run("flow")
    assert r3.stats["z_01"]["computed"] == 0 and r3.stats["z_01"]["temporal"] == "flow" and len(runner.calls) == 5
    for r in (r2, r3):   # static scene: smoothing changes nothing outside the area, core stays 1
        for f in frames:
            a, _ = r.soft[f]
            assert a[~area].max() == 0.0 and a[area & core].min() == 1.0
            assert np.allclose(a, r0.soft[f][0], atol=1e-6)          # identical frames -> median == frame
    with pytest.raises(ValueError):
        run("gauss")
    # a different model id -> different hash -> recomputed
    runner.model_id = "mock-2"
    r4 = run("none")
    assert r4.stats["z_01"]["computed"] == 5


def test_local_matte_rejects_vitmatte_per_frame():
    img, core, hair, region, frames = _frames_scene(1)
    with pytest.raises(ValueError):
        local_matte(img, core, region, engine="vitmatte")


def _pan_sequence(n=7, step=2, h=128, w=192, smooth=False, seed=0, width=4):
    """Background panning `step` px/frame (fine texture, or a smooth one where DIS flow is unreliable) with a `width` px
    strand moving with it; alpha of the strand = 0.65 + per-frame flicker (±0.1, inside the motion gate). Returns (alphas, grays, positions,
    clean alphas)."""
    import cv2
    rng = np.random.default_rng(seed)
    big = rng.uniform(0.2, 0.8, (h, w + step * n)).astype(np.float32)
    if smooth:
        big = cv2.resize(rng.uniform(0.2, 0.8, (h // 4 + 2, (w + step * n) // 4 + 2)).astype(np.float32),
                         (w + step * n, h), interpolation=cv2.INTER_CUBIC)
    else:
        big = cv2.GaussianBlur(big, (0, 0), 0.7)
    alphas, clean, grays, pos = [], [], [], []
    flick = rng.uniform(-0.1, 0.1, n)
    for i in range(n):
        g = big[:, step * i:step * i + w].copy()
        a = np.zeros((h, w), np.float32)
        sx = 20 + step * i
        a[10:h - 14, sx:sx + width] = 0.65
        clean.append(a.copy())
        a[10:h - 14, sx:sx + width] += flick[i]
        g[10:h - 14, sx:sx + width] = 1.0
        alphas.append(a); grays.append(g); pos.append(sx)
    return np.stack(alphas), np.stack(grays), pos, np.stack(clean)


def _strand_flicker(stack, pos, h=128, width=4):
    on = [stack[i][10:h - 14, pos[i]:pos[i] + width].mean() for i in range(len(pos))]
    return float(np.mean(np.abs(np.diff(on)))), on


def test_temporal_median_flow_smooths_moving_strand_without_ghosts():
    alphas, grays, pos, clean = _pan_sequence()
    plain = temporal_median(alphas, 2, 0.35)
    flow = temporal_median_flow(alphas, grays, 2)
    f_in, _ = _strand_flicker(alphas, pos)
    f_plain, _ = _strand_flicker(plain, pos)
    f_flow, on = _strand_flicker(flow, pos)
    # the strand moves 2 px/frame: the unaligned gated median sees "motion" and leaves the flicker untouched; the aligned
    # one smooths it (>= 20 % less on a 4 px strand; sub-pixel flow error limits the gain on thinner strands – measured
    # 2026-10-13: 2 px none, 4 px -25 %, 8 px -43 %) while every frame keeps its strand and nothing appears next to it
    assert f_plain > 0.95 * f_in and f_flow < 0.8 * f_in, (f_in, f_plain, f_flow)
    assert min(on) > 0.4
    for i in range(len(pos)):
        off = flow[i].copy(); off[10:118, max(0, pos[i] - 2):pos[i] + 6] = 0
        assert off.max() < 0.05
    # band: outside it the input is returned untouched; short stacks pass through
    band = np.zeros_like(alphas, dtype=bool); band[:, :30] = True
    fb = temporal_median_flow(alphas, grays, 2, band=band)
    assert np.array_equal(fb[:, 30:], alphas[:, 30:])
    assert np.array_equal(temporal_median_flow(alphas[:2], grays[:2], 2), alphas[:2])


def test_temporal_median_flow_keeps_strands_when_flow_is_wrong():
    # smooth background: DIS cannot see the pan (aperture) -> the aligned window is not aligned -> the motion gate
    # keeps every frame's strand (a plain aligned median would erase it)
    alphas, grays, pos, clean = _pan_sequence(smooth=True, width=2)
    flow = temporal_median_flow(alphas, grays, 2)
    _, on = _strand_flicker(flow, pos, width=2)
    assert min(on) > 0.4, on
    plain_aligned = temporal_median_flow(alphas, grays, 2, motion_limit=0.0)
    _, on0 = _strand_flicker(plain_aligned, pos, width=2)
    assert min(on0) < 0.2, on0            # the hazard the gate protects from


def test_api_vitmatte_zone_preview_and_finalize(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    from test_zone_matte import _api_project
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "vitmatte", "preset": "hair",
         "params": {"temporal": "flow"}}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    pv = r.json()["preview"]
    assert pv["stats"]["engine"] == "vitmatte" and pv["stats"]["model"] == "mock" and "ViTMatte" in pv["note"]
    assert len(runner.calls) == 1
    zid = r.json()["zone"]["zone_id"]
    # knobs tuned on the key frame through the preview (user 2026-10-14): model scale, fine tune, model, erosion
    r2 = api.post(f"/api/object/obj/zones/{zid}?preview_frame={start}",
                  json={**z, "params": {"temporal": "flow", "model_px": 504, "unknown_value": 0.3, "fg_erode_px": 4}})
    assert r2.status_code == 200, r2.text
    st2 = r2.json()["preview"]["stats"]
    assert st2["model_px"] < pv["stats"]["model_px"] and st2["unknown_value"] == 0.3 and len(runner.calls) == 2
    assert r2.json()["zone"]["params"]["model_px"] == 504
    # preset default (user decision 2026-10-25): hair without an explicit engine -> vitmatte at 1/2 (504 px), fine tune 0.5,
    # NO temporal smoothing (flow_mean r3 x2 stays the setting when switched on), black / white; fibres -> matanyone
    r3 = api.post(f"/api/object/obj/zones/new", json={k: v for k, v in z.items() if k not in ("matting", "params")})
    assert r3.status_code == 200 and r3.json()["zone"]["matting"] == "vitmatte"
    assert r3.json()["zone"]["params"] == {"temporal": "none", "temporal_radius": 3, "temporal_scale": 2,
                                          "model_px": 504, "unknown_value": 0.5,
                                          "alpha_black": 0.1, "alpha_white": 0.9, "soft_trimap": False}   # a drawn zone: no band sliders
    r4 = api.post(f"/api/object/obj/zones/new", json={**{k: v for k, v in z.items() if k not in ("matting", "params")}, "preset": "fibres"})
    assert r4.status_code == 200 and r4.json()["zone"]["matting"] == "matanyone"
    for zz in (r3.json()["zone"]["zone_id"], r4.json()["zone"]["zone_id"]):
        api.delete(f"/api/object/obj/zones/{zz}")
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "matanyone"})
    assert fz.status_code == 200, fz.text
    run = fz.json()["run"]
    rj = json.loads((pipe.store.dir("obj") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    zs = rj["zones"][zid]
    assert zs["engine"] == "vitmatte" and zs["temporal"] == "flow" and zs["frames"] == len(frames)
    assert zs["computed"] == len(frames) and zs["cached"] == 0 and len(runner.calls) == 2 + len(frames)
    # "Zastosuj" again: every frame from the cache, no model call
    fa = api.post(f"/api/object/obj/zones/{zid}/apply", json={})
    assert fa.status_code == 200, fa.text
    rj2 = json.loads((pipe.store.dir("obj") / "runs" / fa.json()["run"] / "run.json").read_text(encoding="utf-8"))
    assert rj2["zones"][zid]["computed"] == 0 and rj2["zones"][zid]["cached"] == len(frames)
    assert len(runner.calls) == 2 + len(frames)


def test_vitmatte_temporal_radius_and_scale_params(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    img, core, hair, region, frames = _frames_scene(7)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")
    area = zone_mask_at(Zone("z_01", "add", "matte", 1, polygon=poly), core.shape)

    def run(params):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params=params)
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache)
    r0 = run({"temporal": "none"})
    r1 = run({"temporal": "flow", "temporal_radius": 3, "temporal_scale": 2})
    st = r1.stats["z_01"]
    assert st["temporal"] == "flow" and st["temporal_radius"] == 3 and st["temporal_scale"] == 2.0
    assert st["computed"] == 0 and st["cached"] == 7          # radius / scale do not touch the per-frame cache
    for f in frames:
        a, ar = r1.soft[f]
        assert np.array_equal(ar, area) and a[~area].max() == 0.0 and a[area & core].min() == 1.0
        assert np.abs(a - r0.soft[f][0]).max() < 0.05           # static scene: the 2x smoothing changes (almost) nothing
    r2 = run({"temporal": "median", "temporal_radius": 1})
    assert r2.stats["z_01"]["temporal_radius"] == 1 and r2.stats["z_01"]["temporal_scale"] == 1.0
