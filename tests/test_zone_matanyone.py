# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""`matanyone` zone engine (docs/PLAN_HAIR_TOOL.md section 3, pulled forward 2026-10-13): one MatAnyone2 pass on the
up-scaled crop of the zone, key = core ∪ hint, result only inside the zone, outside bitwise the run without the zone."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engine.crop import CropSegment
from engine.zones import Zone, apply_zones, matanyone_zone, save_zone, zone_crop_segment, zone_mask_at

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def test_zone_crop_segment_and_scale():
    H, W = 400, 600
    regions = {f: np.zeros((H, W), bool) for f in range(1, 4)}
    for f in regions:
        regions[f][100:220, 200 + 10 * f:260 + 10 * f] = True
    seg = zone_crop_segment(regions, [1, 2, 3], "z_01")
    assert seg is not None and seg.side % 16 == 0 and seg.crop[2] == seg.side
    x, y, s, _ = seg.crop
    assert x <= 200 and y <= 100 and x + s >= 290 and y + s >= 220       # union + 48 px inside the square
    assert zone_crop_segment({1: np.zeros((H, W), bool)}, [1], "z_02") is None


def test_matanyone_zone_pastes_only_inside_region_core_one_bg_zero():
    H, W = 240, 320
    core = np.zeros((H, W), bool); core[120:200, 100:220] = True
    frames = [1, 2, 3]
    cores = {f: core for f in frames}
    region = np.zeros((H, W), bool); region[60:140, 90:150] = True
    regions = {f: region for f in frames}
    seen = {}

    def ma_run(seg, sf, core_, keys_, start, scale, tag):
        seen.update(seg=seg, frames=list(sf), keys=keys_, start=start, scale=scale, tag=tag)
        out = {}
        for f in sf:
            a = np.zeros((seg.side, seg.side), np.float32)
            a[:, :] = 0.7                                     # the "model" says 0.7 everywhere in the crop
            out[f] = a
        return out
    init = np.zeros((H, W), np.float32); init[70:80, 95:105] = 0.9     # hint strands
    z = Zone("z_01", "add", "matte", 2, polygon=[[90, 60], [150, 60], [150, 140], [90, 140]], matting="matanyone")
    hold = np.zeros((H, W), bool); hold[60:70, 140:150] = True
    bgm = np.zeros((H, W), bool); bgm[130:140, 90:100] = True
    out, st = matanyone_zone(z, frames, cores, regions, init, ma_run, holdout=lambda f: hold, bg=lambda f: bgm)
    assert seen["start"] == 2 and seen["tag"] == "zone_z_01" and seen["frames"] == frames
    assert 1.0 < seen["scale"] <= 3.0                           # upscaled towards 1008 px
    key = seen["keys"][2]
    assert key[150, 150] and key[75, 100] and not key[65, 95]   # core ∪ hint, not the plain region
    for f in frames:
        a = out[f]
        assert a[~region].max() == 0.0
        assert np.isclose(a[region & ~core & ~hold & ~bgm].mean(), 0.7)
        assert a[region & core].min() == 1.0
        assert a[hold & region].max() == 0.0 and a[bgm & region].max() == 0.0
    assert st["engine"] == "matanyone" and st["key_frame"] == 2 and st["hint_px"] == 100


def test_apply_zones_matanyone_requires_runner_and_uses_init_hint():
    from tests.test_localmatte import _motley, _scene
    img, core, hair, line, region = _scene(_motley(192, 256))
    frames = [1, 2]
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="matanyone")
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    with pytest.raises(ValueError):
        apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr)
    calls = []

    def ma_run(seg, sf, core_, keys_, start, scale, tag):
        calls.append((seg.side, scale, int(keys_[start].sum())))
        return {f: np.ones((seg.side, seg.side), np.float32) * 0.5 for f in sf}
    res = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, ma_run=ma_run)
    assert len(calls) == 1 and calls[0][2] > int(core.sum())          # key = core ∪ hint from the chroma/clean init
    area = zone_mask_at(z, core.shape)
    for f in frames:
        a, ar = res.soft[f]
        assert np.array_equal(ar, area) and a[~area].max() == 0.0 and a[area & core].min() == 1.0
    assert res.stats["z_01"]["engine"] == "matanyone"


def test_api_matanyone_zone_preview_shows_hint_and_finalize_runs(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "matanyone", "preset": "hair"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    assert "MatAnyone2" in r.json()["preview"]["note"] and r.json()["preview"]["stats"]["engine"] in ("chroma", "clean")
    # finalize with the mock Comfy (synthetic MatAnyone handlers): a version with the zone stats
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "matanyone"})
    assert fz.status_code == 200, fz.text
    run = fz.json()["run"]
    rj = json.loads((pipe.store.dir("obj") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    zs = rj["zones"][r.json()["zone"]["zone_id"]]
    assert zs["engine"] == "matanyone" and zs["frames"] == len(frames) and zs["scale"] > 1.0
