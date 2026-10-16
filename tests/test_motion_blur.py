# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„Motion blur alfy” (user decision 2026-10-16, engine/motionblur.py): a vector blur of the alpha along the object's own
motion (plate flow x shutter / 360 x strength), only where it moves; applied on a version -> a new version, a stage's
result stays that stage's result. Engine on a synthetic moving square, API on the synthetic shot of test_stage_layout."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

H, W = 240, 320


def _square(x: int):
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    tex = cv2.GaussianBlur((np.random.default_rng(1).random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    g, m = bg.copy(), np.zeros((H, W), bool)
    m[80:160, x:x + 80] = True
    g[m] = np.roll(tex, x, axis=1)[m]
    return g, m


def _soft(v):
    return int(((v > 0.02) & (v < 0.98)).sum())


@pytest.mark.parametrize("speed,shutter,strength,ramp", [(12, 180, 1.0, 6), (12, 360, 1.0, 12), (12, 180, 2.0, 12),
                                                         (8, 180, 1.0, 4)])
def test_blur_along_the_motion_only(speed, shutter, strength, ramp):
    from engine.motionblur import motion_blur_alpha
    g0, _ = _square(100 - speed)
    g1, m1 = _square(100)
    g2, _ = _square(100 + speed)
    out, st = motion_blur_alpha(m1.astype(np.float32), g1, g0, g2, shutter, strength)
    row, col = out[120, 60:220], out[50:190, 140]
    # a box blur of length L = speed x shutter / 360 x strength: a linear ramp ~L px on the leading and trailing edge
    assert abs(_soft(row) - 2 * ramp) <= 3, (_soft(row), np.round(row, 2))
    assert _soft(col) == 0                                    # across the motion: sharp
    assert abs(st["max_px"] - ramp) <= 1.0
    assert out[120, 140] == pytest.approx(1.0) and out[20, 20] == 0.0   # inside / far outside unchanged
    assert abs(float(out.sum()) - float(m1.sum())) < 0.02 * m1.sum()    # a blur keeps the coverage


def test_static_object_strength_zero_and_shot_ends_unchanged():
    from engine.motionblur import motion_blur_alpha
    g, m = _square(100)
    a = m.astype(np.float32)
    out, st = motion_blur_alpha(a, g, g, g, 180, 1.0)       # no motion: nothing changes
    assert np.array_equal(out, a) and st["moving_px"] == 0
    g0, _ = _square(88)
    g2, _ = _square(112)
    assert np.array_equal(motion_blur_alpha(a, g, g0, g2, 180, 0.0)[0], a)       # strength 0
    assert np.array_equal(motion_blur_alpha(a, g, g0, g2, 0.0, 1.0)[0], a)       # shutter 0
    one_side, _ = motion_blur_alpha(a, g, None, g2, 180, 1.0)                    # last frame of the shot: one flow
    assert _soft(one_side[120, 60:220]) > 6


def test_background_motion_is_not_used():
    """The camera pans (background moves), the object does not move on screen: no blur (only the object's own flow)."""
    from engine.motionblur import motion_blur_alpha
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur((rng.random((H, W + 40)) * 255).astype(np.uint8), (0, 0), 1.5)
    tex = cv2.GaussianBlur((np.random.default_rng(4).random((H, W)) * 255).astype(np.uint8), (0, 0), 1.5)
    m = np.zeros((H, W), bool)
    m[80:160, 100:180] = True

    def fr(dx):
        g = big[:, 20 + dx:20 + dx + W].copy()
        g[m] = tex[m]
        return g
    out, st = motion_blur_alpha(m.astype(np.float32), fr(0), fr(-10), fr(10), 180, 1.0)
    assert _soft(out[120, 60:220]) <= 2 and st["max_px"] < 1.5


@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_api_preview_and_apply_on_the_stage_3_result(proj):
    app_module, api, shot, pipe, frames, start = proj
    od = pipe.store.dir("obj")
    assert api.post("/api/object/obj/motion_blur", json={"strength": 1}).status_code == 409   # no version yet
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    base = r.json()["run"]
    f = frames[len(frames) // 2]
    pv = api.get(f"/api/object/obj/motion_blur/{f}?v={base}&strength=1&shutter=360")
    assert pv.status_code == 200 and pv.headers["X-MD-MB-Run"] == base
    assert float(pv.headers["X-MD-MB-Max-Px"]) > 1.0        # the synthetic object moves
    assert api.post("/api/object/obj/motion_blur", json={"run": base, "strength": 0}).status_code == 409

    r = api.post("/api/object/obj/motion_blur", json={"run": base, "strength": 1.5, "shutter_deg": 270})
    assert r.status_code == 200, r.text
    new, mb = r.json()["run"], r.json()["motion_blur"]
    assert mb["strength"] == 1.5 and mb["shutter_deg"] == 270.0 and mb["base_run"] == base and mb["stages"] == ["matanyone"]
    rj = json.loads((od / "runs" / new / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "matanyone" and rj["motion_blur"]["strength"] == 1.5     # still the stage-3 result
    obj = pipe.store.get("obj")
    assert obj["tool"]["stage_runs"]["matanyone"] == new and obj["tool"]["active_run"] == new
    d = api.get("/api/object/obj/stages").json()
    assert {s["id"]: s["state"] for s in d["steps"]}["matanyone"] == "done"         # valid: ViTMatte can use it as base
    runs = {x["run"]: x for x in api.get("/api/object/obj/runs").json()["runs"]}
    assert runs[new]["motion_blur"]["base_run"] == base and runs[base]["motion_blur"] is None

    from engine.export import run_alpha_loader
    a0 = run_alpha_loader(od, base, shot.w, shot.h)(f)
    a1 = run_alpha_loader(od, new, shot.w, shot.h)(f)
    ab = run_alpha_loader(od, new, shot.w, shot.h, sub="alpha_base")(f)
    assert _soft(a1) > _soft(a0)                              # more soft edge where it moves
    assert np.array_equal(a1, ab)                             # no zones: the blurred alpha is the base too
    assert abs(float(a1.sum()) - float(a0.sum())) < 0.05 * a0.sum()


def test_apply_on_a_plain_version_is_a_plain_version(proj):
    app_module, api, shot, pipe, frames, start = proj
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    base = r.json()["run"]
    pipe.store.update("obj", tool={"stage_runs": {}})         # not a stage's result any more
    r = api.post("/api/object/obj/motion_blur", json={"run": base})
    assert r.status_code == 200, r.text
    rj = json.loads((pipe.store.dir("obj") / "runs" / r.json()["run"] / "run.json").read_text(encoding="utf-8"))
    assert rj["applied_from"] == base and "stage" not in rj and r.json()["motion_blur"]["stages"] == []


# ------------------------------------------------------------------ GPU (the alignment worker in ComfyUI's Python)
def _gpu_ready():
    from engine import align_gpu
    ok, info = align_gpu.check_worker()
    return ok and info.startswith("cuda")


@pytest.mark.skipif(not _gpu_ready(), reason="ComfyUI's Python with CUDA not available")
@pytest.mark.parametrize("speed,shutter,strength", [(12, 180, 1.0), (30, 360, 1.5), (0, 180, 1.0)])
def test_gpu_equals_cpu(speed, shutter, strength):
    """The vector blur in the GPU worker = the CPU code within the bilinear rounding of cv2.remap (1/32 px)."""
    from engine import align_gpu
    from engine.motionblur import motion_blur_alpha
    g0, _ = _square(100 - speed)
    g1, m1 = _square(100)
    g2, _ = _square(100 + speed)
    a = cv2.GaussianBlur(m1.astype(np.float32), (0, 0), 1.5)     # a soft MatAnyone-like edge
    cpu, sc = motion_blur_alpha(a, g1, g0, g2, shutter, strength)
    gpu = align_gpu.open_aligner()
    try:
        gg, sg = motion_blur_alpha(a, g1, g0, g2, shutter, strength, gpu=gpu)
    finally:
        align_gpu.close_aligner(gpu)
    d = np.abs(gg - cpu)
    assert float(d.max()) < 0.02 and float(d.mean()) < 1e-3, (float(d.max()), float(d.mean()))
    assert sg["samples"] == sc["samples"] and abs(sg["max_px"] - sc["max_px"]) < 0.05


def test_apply_falls_back_to_the_cpu_when_the_gpu_worker_dies(proj, monkeypatch):
    """auto: a GPU worker that dies -> the rest on the CPU (fake worker in this .venv's Python, no torch)."""
    import os
    from engine import align_gpu, motionblur
    app_module, api, shot, pipe, frames, start = proj
    base = api.post("/api/object/obj/finalize", json={"stage": "matanyone"}).json()["run"]
    env = {**os.environ, "MD_ALIGN_GPU_FAKE": "1", "MD_ALIGN_GPU_FAKE_CRASH": "1"}
    monkeypatch.setattr(motionblur, "mb_device", lambda requested=None: ("gpu", "fake"))
    monkeypatch.setattr(align_gpu, "open_aligner", lambda *a, **k: align_gpu.GpuAligner(sys.executable, env, timeout_s=60))
    new_dir, mb = pipe.apply_motion_blur("obj", frames, base, 1.0)
    assert mb["device"] == "cpu" and "GPU przerwane" in mb["device_note"]
    assert (new_dir / "run.json").exists()
