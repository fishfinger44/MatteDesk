# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E2 – crop engine (SPEC 5.2). Cloud: synthetic puppet shot + colour-key SAM3 behind the mock Comfy.

Acceptance (spec 12): bbox covers the object in every frame (GT mask inside the crop, 0 frames cut);
crops square, side a multiple of 16. Plus: fill >= min_fill where geometry allows, 8-frame overlap,
mirror padding / shift when the object leaves the frame, tracker gets no text.
Real SH010 / Figure v5: test_e2_real_figure_v5 (station).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import (CropSegment, choose_instance, extract_crop, frame_segments, letterbox_square,
                         load_crops, paste_crop, plan_segments, run_crop_engine, save_crops, smooth_bboxes)
from engine.sam3 import Sam3Runner
from mock_comfy import MockComfyClient, MockOutputs
from synth import display, make_shot, sam3_handlers


def _runner(tmp_path):
    outs = MockOutputs(tmp_path / "comfy_out")
    client = MockComfyClient(handlers=sam3_handlers(outs))
    client.output_root = outs.root
    return Sam3Runner(client, tmp_path / "scratch", ckpt="sam3.safetensors"), client


def _load_proxy(shot, scale):
    def f(fr):
        img = display(cv2.imread(frame_path(shot.plate_pattern, fr), cv2.IMREAD_UNCHANGED))
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
    return f


def frame_path(p, f):
    return p.replace("####", f"{f:04d}")


def test_e2_acceptance_crop_covers_object_every_frame(tmp_path):
    shot = make_shot(tmp_path)
    runner, client = _runner(tmp_path)
    scale = 0.5  # proxy half size
    lp = _load_proxy(shot, scale)
    frames = list(range(shot.first, shot.last + 1))

    # 0. instance choice from the text prompt on the letterboxed proxy
    start = shot.first + 20
    cands = choose_instance(runner, lp(start), prompt="felt puppet with two long antennae")
    assert len(cands) == 1
    res = run_crop_engine(runner, lp, frames, start, cands[0], scale, shot.w, shot.h)
    segs = res.segments
    assert segs and segs[0].first == shot.first and segs[-1].last == shot.last

    for g in client.sent:
        if any(v["class_type"] == "SAM3_VideoTrack" for v in g.values()):
            assert not any(v["class_type"] == "CLIPTextEncode" for v in g.values())

    cut_frames = []
    for f in frames:
        gt = shot.masks[f] > 0.01
        for s in frame_segments(segs, f):
            assert s.crop[2] == s.crop[3] and s.side % 16 == 0
            inside = paste_crop(np.ones((s.side, s.side), np.float32), s, shot.w, shot.h) > 0.5
            if (gt & ~inside).any():
                cut_frames.append((f, s.seg_id))
    assert cut_frames == [], f"object cut in {cut_frames}"

    # overlap of 8 frames between consecutive segments; fill rule
    for a, b in zip(segs, segs[1:]):
        assert a.last - b.first + 1 == 8 and a.overlap_out == 8 and b.overlap_in == 8
    for s in segs:
        assert s.fill_min >= 0.35 or s.flags, (s.seg_id, s.fill_min)
        x, y, side, _ = s.crop
        l, t, r, b = s.pad
        assert x + l >= 0 and y + t >= 0 and x + side - r <= shot.w and y + side - b <= shot.h

    # crops.json round trip + SAM px per plate px reported
    save_crops(tmp_path / "crops.json", segs)
    assert [s.to_dict() for s in load_crops(tmp_path / "crops.json")] == [s.to_dict() for s in segs]
    assert all(abs(s.scale_to_1008 - 1008 / s.side) < 1e-3 for s in segs)


def test_mirror_padding_when_plate_smaller_than_crop():
    bbs = [(10, 10, 300, 200)] * 5
    segs = plan_segments(bbs, 1, 320, 240)
    s = segs[0]
    assert s.side >= 300 + 64 and s.side % 16 == 0
    assert sum(s.pad) > 0  # square larger than the plate: mirror padding recorded
    img = np.random.default_rng(0).random((240, 320)).astype(np.float32)
    crop = extract_crop(img, s)
    assert crop.shape == (s.side, s.side)
    back = paste_crop(crop, s, 320, 240)
    assert np.allclose(back, img, atol=1e-6)


def test_letterbox_is_square_and_isotropic():
    img = np.zeros((90, 160, 3), np.float32)
    sq, (ox, oy) = letterbox_square(img)
    assert sq.shape[:2] == (160, 160) and (ox, oy) == (0, 35)


def test_smoothing_never_cuts_the_raw_bbox():
    rng = np.random.default_rng(0)
    raw = [(100 + rng.normal(0, 3), 50 + rng.normal(0, 3), 200 + rng.normal(0, 3), 160 + rng.normal(0, 3)) for _ in range(30)]
    sm = smooth_bboxes(raw)
    for r, s in zip(raw, sm):
        assert s[0] <= r[0] and s[1] <= r[1] and s[2] >= r[2] and s[3] >= r[3]


REAL_FIGURE_V5 = Path("C:\\shots\\SH010\\frame_####.exr")


def test_e2_real_figure_v5_masks_inside_crops():
    """Station: crop planning from the accepted Figure v5 mattes (bbox per frame) -> v5 ⊂ crop, 0 cut frames.
    (The SAM3 coarse pass itself is checked from the UI, LOCAL_CHECKS E2.)"""
    assert REAL_FIGURE_V5.exists(), "MD_REAL_PROJECT Figure v5 missing"
    files = sorted(p for p in os.listdir(REAL_FIGURE_V5) if p.lower().endswith(".exr"))
    from engine.exr_io import read_exr
    bbs, masks = [], []
    for p in files:
        ch, _ = read_exr(REAL_FIGURE_V5 / p)
        a = ch.get("A", ch.get("R"))
        masks.append(a > 0.01)
        ys, xs = np.where(a > 0.01)
        bbs.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None)
    h, w = masks[0].shape
    segs = plan_segments(smooth_bboxes(bbs), 1, w, h)
    cut = 0
    for i, m in enumerate(masks):
        for s in frame_segments(segs, i + 1):
            inside = paste_crop(np.ones((s.side, s.side), np.float32), s, w, h) > 0.5
            cut += int((m & ~inside).any())
            assert s.side % 16 == 0
    print([(s.seg_id, s.first, s.last, s.side, s.fill_min) for s in segs])
    assert cut == 0


def test_api_object_detect_choose_crops(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module

    shot = make_shot(tmp_path, n=24)
    work = tmp_path / "shot"
    work.mkdir()
    mf = work / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "colorspace": "ACES2065-1", "work": str(work),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    runner, client = _runner(tmp_path)
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    # proxies in the test = our display transform (ProxyGenerator tonemap differs from the synthetic encoding)
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": display(
        cv2.imread(frame_path(shot.plate_pattern, f), cv2.IMREAD_UNCHANGED)))

    assert api.post("/api/object", json={"name": "puppet", "prompts": ["felt puppet"]}).status_code == 200
    d = api.post("/api/object/puppet/detect", json={"frame": shot.first + 10}).json()
    assert d["auto_choice"] == 0 and len(d["candidates"]) == 1
    assert api.post("/api/object/puppet/choose", json={"frame": shot.first + 10, "candidate": 0}).status_code == 200
    r = api.post("/api/object/puppet/crops")
    assert r.status_code == 200, r.text
    segs = r.json()["segments"]
    assert segs[0]["first"] == shot.first and segs[-1]["last"] == shot.last
    s0 = segs[0]
    r2 = api.put(f"/api/object/puppet/crops/{s0['seg_id']}", json={"crop": [s0["crop"][0] - 10, s0["crop"][1], s0["crop"][2] + 20]})
    s0b = r2.json()["segments"][0]
    assert s0b["manual"] is True and s0b["crop"][2] % 16 == 0
    keys = json.loads((work / "matte_desk" / "objects" / "puppet" / "keys" / f"k_{shot.first + 10:04d}.json").read_text())
    assert keys["author"] == "auto"


def test_crops_cover_keys_rule():
    """„Przelicz nieaktualne” (2026-10-15) redoes the crops only when a user key no longer fits its segment's crop."""
    from engine.crop import CropSegment, crops_cover_keys
    seg = CropSegment("s00", 1, 10, 0, 0, [100, 100, 400, 400], [0, 0, 0, 0], 2.52)
    assert crops_cover_keys([seg], {5: (150, 150, 300, 300)}) is None
    assert "poza kadr" in crops_cover_keys([seg], {5: (150, 150, 520, 300)})        # mask leaves the crop
    assert "poza odcinkami" in crops_cover_keys([seg], {12: (150, 150, 300, 300)})  # frame without a segment
    assert crops_cover_keys([seg], {}) is None


def test_api_compute_all_crops_then_core_then_cache(tmp_path, monkeypatch):
    """„▶▶ Przelicz nieaktualne”: crops computed automatically (none yet), core on the whole shot; the second call keeps the
    crops and takes every interval from the cache."""
    from fastapi.testclient import TestClient
    import app as app_module

    shot = make_shot(tmp_path, n=24)
    work = tmp_path / "shot"
    work.mkdir()
    mf = work / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "colorspace": "ACES2065-1", "work": str(work),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    runner, client = _runner(tmp_path)
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": display(
        cv2.imread(frame_path(shot.plate_pattern, f), cv2.IMREAD_UNCHANGED)))
    monkeypatch.setattr(app_module, "_plate_display", display)
    assert api.post("/api/object", json={"name": "puppet", "prompts": ["felt puppet"]}).status_code == 200
    assert api.post("/api/object/puppet/compute_all").status_code == 409          # no key yet
    assert api.post("/api/object/puppet/detect", json={"frame": shot.first + 10}).status_code == 200
    assert api.post("/api/object/puppet/choose", json={"frame": shot.first + 10, "candidate": 0}).status_code == 200
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["crops_recomputed"] and d["crops_reason"] == "brak kadrów" and d["crop_segments"] >= 1
    assert d["recomputed"] and (work / "matte_desk" / "objects" / "puppet" / "core" / "core.json").exists()
    d2 = api.post("/api/object/puppet/compute_all").json()
    assert d2["crops_recomputed"] is False and d2["recomputed"] == [] and d2["cached"]
