# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ViTMatte zone trimap from the version's base alpha (zone param trimap_source=base, user idea 2026-10-15): SAM3 makes the
key, MatAnyone the whole object, ViTMatte refines the hair with a trimap built from MatAnyone's (temporally coherent)
alpha instead of the SAM3 core. Default "core" = unchanged; "base" without a base alpha falls back to the core."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_zone_vitmatte import MockRunner, _frames_scene  # noqa: E402

from engine.zonecache import ZoneCache  # noqa: E402
from engine.zones import BASE_TRIMAP_FG, Zone, apply_zones  # noqa: E402


def _setup(tmp_path, monkeypatch, n=4):
    import engine.vitmatte as vm
    img, core, hair, region, frames = _frames_scene(n)
    runner = MockRunner()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache = ZoneCache(tmp_path, "z_01")

    def run(params=None, base_alpha=None):
        z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="vitmatte", params={"temporal": "none", **(params or {})})
        return apply_zones(frames, cores, lambda f: img, [z], tr, cache_for=lambda zid: cache, base_alpha=base_alpha)
    return run, runner, core, region, frames


def test_trimap_from_base_alpha(tmp_path, monkeypatch):
    run, runner, core, region, frames = _setup(tmp_path, monkeypatch)
    r_core = run()
    assert r_core.stats["z_01"]["trimap_source"] == "core" and r_core.stats["z_01"]["computed"] == 4
    fg_core = runner.calls[0][2]
    # base alpha = a jittery SAM3-like core grown by 3 px with a soft rim: its solid part is NOT the SAM3 core
    import cv2
    grown = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    base = np.clip(cv2.GaussianBlur(grown, (0, 0), 1.0) * 1.05, 0, 1).astype(np.float32)
    calls = []

    def base_alpha(f):
        calls.append(f)
        return base
    n0 = len(runner.calls)
    r_base = run({"trimap_source": "base"}, base_alpha)
    st = r_base.stats["z_01"]
    assert st["trimap_source"] == "base" and st["computed"] == 4 and sorted(set(calls)) == frames   # new hash, recomputed
    assert all(c[2] != fg_core for c in runner.calls[n0:])                   # FG of the trimap follows the base alpha
    solid = base >= BASE_TRIMAP_FG
    for f in frames:
        a, area = r_base.soft[f]
        assert a[area & solid].min() == 1.0                                   # forced to 1 where the base is solid
        assert not np.array_equal(a, r_core.soft[f][0])
    # second pass: cached
    assert run({"trimap_source": "base"}, base_alpha).stats["z_01"]["computed"] == 0


def test_trimap_base_without_base_alpha_falls_back_and_bad_value(tmp_path, monkeypatch):
    run, runner, core, region, frames = _setup(tmp_path, monkeypatch)
    r_core = run()
    r_fb = run({"trimap_source": "base"}, None)
    assert r_fb.stats["z_01"]["trimap_source"].startswith("core (brak alphy bazowej")
    for f in frames:                                                          # same trimap as the core source
        assert np.array_equal(r_fb.soft[f][0], r_core.soft[f][0])
    with pytest.raises(ValueError):
        run({"trimap_source": "sam"}, None)


def test_run_alpha_loader_reads_alpha_base(tmp_path):
    import json
    from engine.crop import CropSegment
    from engine.export import run_alpha_loader
    from engine.refine import write_alpha_exr
    seg = CropSegment("s00", 1, 2, 0, 0, [0, 0, 32, 32], [0, 0, 0, 0], 1.0)
    rd = tmp_path / "runs" / "v001"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"segments_def": [seg.to_dict()]}), encoding="utf-8")
    write_alpha_exr(rd / "alpha" / "s00" / "a_0001.exr", np.full((32, 32), 0.25, np.float32))
    write_alpha_exr(rd / "alpha_base" / "s00" / "a_0001.exr", np.full((32, 32), 0.75, np.float32))
    assert np.allclose(run_alpha_loader(tmp_path, "v001", 32, 32)(1), 0.25)
    assert np.allclose(run_alpha_loader(tmp_path, "v001", 32, 32, sub="alpha_base")(1), 0.75)


def test_force_core_erode_leaves_the_rim_to_the_model(tmp_path, monkeypatch):
    """force_core_erode_px: the result is forced to 1 only on the core eroded by N px (CLIP2_A F50: a hard step on the SAM3
    core edge came from forcing the whole core back to 1 after ViTMatte). 0 = as before, bit for bit."""
    import cv2
    run, runner, core, region, frames = _setup(tmp_path, monkeypatch)
    r0 = run()
    r0b = run({"force_core_erode_px": 0})
    for f in frames:
        assert np.array_equal(r0b.soft[f][0], r0.soft[f][0])
    assert r0.stats["z_01"]["force_core_erode_px"] == 0
    r6 = run({"force_core_erode_px": 6, "fg_erode_px": 6})
    assert r6.stats["z_01"]["force_core_erode_px"] == 6
    inner = cv2.erode(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)
    rim = core & ~inner
    for f in frames:
        a, area = r6.soft[f]
        assert a[area & inner].min() == 1.0                      # eroded core still solid
        assert a[area & rim].min() < 1.0                         # the rim is the model's (MockRunner: warmth of the pixel)
        assert r0.soft[f][0][area & rim].min() == 1.0            # before: forced
