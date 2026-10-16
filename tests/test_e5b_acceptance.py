# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E5b – edge accuracy (7.8), plate-to-mask (4b), lupa (5.2 pkt 5b), comp preview.

Acceptance (spec 12): mask widened on purpose by 1.5 px on a sharp edge -> measured +1.5 +- 0.2 px and the
correction brings it to <= 0.5 px; plate-to-mask: static edges moved < 0.1 px, noise down >= 40 %, Figure mask
flicker down vs the display plate of E3 (A/B); lupa: antenna / horn tips present in >= 90 % of frames; hygiene:
interior = 1.0 and background = 0 exactly. Cloud: synthetic shots, SAM mock that sees 1008 px like the real one.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import choose_instance, run_crop_engine, save_crops
from engine.edge import apply_shift, correction_map, measure_edge
from engine.mplate import MaskPlate, noise_level, static_edge_shift
from engine.pipeline import ObjectPipeline
from engine.qa import run_qa
from engine.segment import iou
from mock_comfy import MockComfyClient, MockOutputs
from synth import (PUPPET_BGR, SynthShot, display, make_shot, matanyone_handlers, puppet_mask, sam3_handlers)


class StatsClient(MockComfyClient):
    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}


def _ellipse(w, h, da, ss=8):
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.ellipse(big, (w // 2 * ss, h // 2 * ss), (int((w * 0.3 + da) * ss), int((h * 0.3 + da) * ss)), 15, 0, 360, 255, -1)
    return cv2.resize(big.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)


def test_edge_offset_measure_and_correction():
    w, h = 400, 320
    gt = _ellipse(w, h, 0)
    gray = 0.3 + 0.4 * gt + np.random.default_rng(0).normal(0, 0.005, gt.shape)
    wide = _ellipse(w, h, 1.5)
    m = measure_edge(wide, gray)
    assert abs(m.median - 1.5) <= 0.2, m.median
    fixed = apply_shift(wide > 0.5, correction_map(wide > 0.5, m))
    assert measure_edge(fixed.astype(np.float32), gray).median_abs <= 0.5
    # soft (blurred) edges are not measured
    soft = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3.0)
    assert np.isnan(measure_edge(wide, soft).offsets).mean() > 0.9


def _pipe(tmp_path, shot, use_mplate=True, **sam_kw):
    outs = MockOutputs(tmp_path / "comfy_out")
    client = StatsClient(handlers={**sam3_handlers(outs, **sam_kw), **matanyone_handlers()})
    client.output_root = outs.root
    frames = list(range(shot.first, shot.last + 1))
    pipe = ObjectPipeline(tmp_path / "work" / "matte_desk", shot.plate_pattern, shot.w, shot.h, client,
                          display=display, sam3_ckpt="sam3.st", use_mplate=use_mplate, frames=frames)
    pipe.store.create("obj", ["felt puppet"])
    start = frames[len(frames) // 2]
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="felt puppet")
    pipe.store.save_key("obj", start, cands[0] > 0.5, author="auto")
    pipe.store.update("obj", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("obj") / "crops.json", res.segments)
    return pipe, frames


def _rigid(root, n=10, w=640, h=480, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0) * 0.3 + 0.2
    a = _ellipse(w, h, 0)
    pdir = Path(root) / "plates"
    pdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1) if noise else img
        cv2.imwrite(str(pdir / f"r_{1001 + i:04d}.exr"), ((img ** 2.2) * 1.5).astype(np.float32))
    return SynthShot(str(pdir / "r_####.exr"), 1001, 1000 + n, w, h, {1001 + i: a for i in range(n)},
                     np.zeros((h, w), bool), {})


def test_pipeline_corrects_halo_and_hygiene(tmp_path):
    shot = _rigid(tmp_path)
    pipe, frames = _pipe(tmp_path, shot, use_mplate=False, grow_px=1.5)   # SAM core 1.5 px too wide (halo)
    pipe.store.update("obj", flags=["rigid"])
    pipe.run_core("obj", frames)
    full, rep, run_dir = pipe.finalize("obj", frames, edge_mode="core")
    qa = json.loads((run_dir / "qa.json").read_text())
    # the binary SAM halo of +1.5 px ends up ~+1 px on the antialiased core edge: detected ...
    assert qa["edge"]["before"] >= 0.75 and qa["edge"]["fix_available"], qa["edge"]
    # ... and corrected by "Zastosuj poprawkę krawędzi" (Z27 pkt 1: user decision, no automatic second round)
    full, rep, run_dir = pipe.apply_edge_fix("obj", frames)
    qa = json.loads((run_dir / "qa.json").read_text())
    assert qa["edge"]["corrected"], qa["edge"]
    assert qa["edge"]["after"] <= 0.5
    f = frames[3]
    gt = shot.masks[f]
    assert iou(full[f] > 0.5, gt > 0.5) >= 0.99
    # hygiene: exact 1 inside, exact 0 outside
    core = gt > 0.99
    inner = cv2.erode(core.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    outer = ~cv2.dilate((gt > 0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    assert (full[f][inner] == 1.0).all() and (full[f][outer] == 0.0).all()
    assert "interior_set_to_1" in rep.hygiene


def test_plate_to_mask_gate_and_mask_flicker_ab(tmp_path):
    shot = _rigid(tmp_path, n=12, noise=0.06)
    # A: display plate (E3), B: plate-to-mask (4b); the colour-key SAM reacts to noise like a real model
    res = {}
    for name, use in (("display", False), ("mplate", True)):
        root = tmp_path / name
        root.mkdir()
        pipe, frames = _pipe(root, shot, use_mplate=use, tol=0.15)
        pipe.run_core("obj", frames)
        core = pipe.load_core("obj", frames)
        res[name] = run_qa(frames, lambda f: core[f].astype(np.float32), None).flicker_median
        if use:
            seg = pipe.segments("obj")[0]
            mp_rep = pipe.loader.mask_plate(seg).report
            raw = pipe.loader.raw_crop(frames[4], seg)
            clean = pipe.loader.crop(frames[4], seg)
            assert static_edge_shift(raw, clean) < 0.1
            assert noise_level(clean) <= 0.6 * noise_level(raw)
            assert mp_rep.strength >= 0
    assert res["mplate"] < res["display"], res


def test_lupa_keeps_thin_tips(tmp_path):
    """Big object (crop ~1500 px -> SAM sees it at 1008: 2-px antennae melt); the lupa (<= 512 px) recovers them."""
    w, h, n = 1800, 1400, 6
    rng = np.random.default_rng(6)
    tex = cv2.GaussianBlur(rng.random((h // 4, w // 4, 3)).astype(np.float32), (0, 0), 2.0) * 0.35 + 0.25
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)
    pdir = tmp_path / "plates"
    pdir.mkdir()
    masks, centers = {}, {}
    for i in range(n):
        f = 1001 + i
        cx, cy, s = 900 + 4 * i, 860, 4.5
        a = puppet_mask(w, h, cx, cy, s, ss=2)
        # antennae stay 2 px wide regardless of the scale
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        cv2.imwrite(str(pdir / f"b_{f:04d}.exr"), ((img ** 2.2) * 1.5).astype(np.float32))
        masks[f], centers[f] = a, (cx, cy, s)
    shot = SynthShot(str(pdir / "b_####.exr"), 1001, 1000 + n, w, h, masks, np.zeros((h, w), bool), centers)
    pipe, frames = _pipe(tmp_path, shot, use_mplate=False, simulate_res=True)
    cx, cy, s = centers[1003]
    # SAM at 1008 lost the 2-px antennae completely, so the user adds a lupa frame ("dodaj lupę") over the head
    pipe.store.update("obj", thin_lens="box", tool={"lupa_box": [cx - 260, cy - 75 * s - 340, cx + 260, cy - 75 * s + 60]})
    pipe.run_core("obj", frames)
    core = pipe.load_core("obj", frames)
    assert (pipe.store.dir("obj") / "lupa.json").exists()

    def tips_present(m, f):
        cx, cy, s = centers[f]
        head = (cx, cy - 75 * s)
        tips = [(head[0] + dx * 45 * s * 0.92, head[1] - 70 * s * 0.92) for dx in (-1, 1)]
        return all(m[int(ty) - 2:int(ty) + 3, int(tx) - 2:int(tx) + 3].any() for tx, ty in tips)
    present = sum(tips_present(core[f], f) for f in frames)
    assert present >= 0.9 * len(frames), present



def test_auto_lupa_triggers_on_thin_score():
    from engine.lupa import lupa_segments, thin_score
    m = puppet_mask(640, 480, 320, 300, 1.5) > 0.5
    assert thin_score(m) > 0.0
    body = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    assert thin_score(body) < 0.01
    segs = lupa_segments({1: m, 2: m}, [1, 2], 640, 480)
    assert segs and 256 <= segs[0].side <= 512


def test_api_object_settings(tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    shot = make_shot(tmp_path, n=3)
    work = tmp_path / "w"
    work.mkdir()
    mf = work / "shot_manifest_S.json"
    mf.write_text(json.dumps({"shot": "S", "first": shot.first, "last": shot.last, "plate_w": shot.w, "plate_h": shot.h,
                              "work": str(work), "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    api.post("/api/object", json={"name": "o", "prompts": ["x"]})
    r = api.put("/api/object/o/settings", json={"band_scale": 5, "shift_px": -9, "temporal_smoothing": "strong",
                                                "thin_lens": "box", "lupa_box": [1, 2, 3, 4]})
    o = r.json()["object"]
    # any typed value is kept (user request 2026-10-16: experiments, e.g. a wide band for motion blur) ...
    assert o["band_scale"] == 5.0 and o["shift_px"] == -9.0 and o["temporal_smoothing"] == "strong"
    o2 = api.put("/api/object/o/settings", json={"band_scale": 500, "shift_px": -500}).json()["object"]
    assert o2["band_scale"] == 20.0 and o2["shift_px"] == -100.0   # ... within sanity guards only
    assert o["tool"]["lupa_box"] == [1, 2, 3, 4]
    assert api.put("/api/object/o/settings", json={"temporal_smoothing": "x"}).status_code == 400
