# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E7 – generalization (SPEC 1.0): (a) a second shot without backlit (EXR, other resolution), (b) an 8-bit
MP4 video with a person through the wizard WITHOUT a manifest. Both go through the whole flow with no code
changes (only manifest / wizard); 8-bit -> PNG export. User acceptance of the masks: LOCAL_CHECKS.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.segment import iou
from mock_comfy import MockComfyClient, MockOutputs
from synth import PUPPET_BGR, matanyone_handlers, puppet_mask, sam3_handlers


class StatsClient(MockComfyClient):
    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}


def _client(tmp_path):
    outs = MockOutputs(tmp_path / "comfy_out")
    c = StatsClient(handlers={**sam3_handlers(outs, tol=0.16), **matanyone_handlers(outputs=outs)})
    c.output_root = outs.root
    return c


def _full_flow(api, frames, prompt, start):
    assert api.post("/api/object", json={"name": "hero", "prompts": [prompt]}).status_code == 200
    rd = api.post("/api/object/hero/detect", json={"frame": start})
    assert rd.status_code == 200, rd.text
    d = rd.json()
    assert d["auto_choice"] == 0, d
    assert api.post("/api/object/hero/choose", json={"frame": start, "candidate": 0}).status_code == 200
    r = api.post("/api/object/hero/crops")
    assert r.status_code == 200, r.text
    assert api.post("/api/object/hero/run").status_code == 200
    fz = api.post("/api/object/hero/finalize", json={})
    assert fz.status_code == 200, fz.text
    ex = api.post("/api/export", json={"multichannel": True})
    assert ex.status_code == 200, ex.text
    return fz.json(), ex.json()


def _make_video(root: Path, n=16, w=640, h=360):
    rng = np.random.default_rng(11)
    tex = (cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0) * 0.35 + 0.25)
    fdir = root / "src_frames"
    fdir.mkdir(parents=True)
    gts = []
    for i in range(n):
        a = puppet_mask(w, h, 150 + 18 * i, 220, 0.9)
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        cv2.imwrite(str(fdir / f"f_{i:04d}.png"), (np.clip(img, 0, 1) * 255).astype(np.uint8))
        gts.append(a)
    mp4 = root / "MBA_like_person.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", str(fdir / "f_%04d.png"),
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(mp4)], check=True)
    return mp4, gts


def test_e7b_mp4_person_through_wizard(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    mp4, gts = _make_video(tmp_path)
    work = tmp_path / "MBA_S1"
    api = TestClient(app_module.app)
    r = api.post("/api/project/open_or_create", json={"work_dir": str(work), "shot_name": "MBA_S1",
                                                      "plates": {"beauty": str(mp4)}})
    assert r.status_code == 200, r.text
    monkeypatch.setitem(app_module.STATE, "comfy_client", _client(tmp_path))
    m = r.json()["manifest"]
    assert m["source_type"] == "video" and m["bit_depth"] == 8 and m["fps"] == 25.0
    assert (m["plate_w"], m["plate_h"]) == (640, 360) and m["plates"]["beauty"].endswith("_####.png")
    assert "matte_desk/source/" in m["plates"]["beauty"]
    frames = list(range(m["first"], m["last"] + 1))
    assert len(frames) == len(gts)
    fz, ex = _full_flow(api, frames, "woman in red dress", frames[len(frames) // 2])
    assert fz["report"]["edge_mode"] == "matanyone"  # a person -> MatAnyone (5.5)
    obj = ex["outputs"]["objects"][0]
    assert obj["matte"].endswith(".png") and "mattes_all" not in ex["outputs"]  # 8-bit source -> PNG export
    ious = []
    for i, f in enumerate(frames):
        a = cv2.imread(obj["matte"].replace("####", f"{f:04d}"), cv2.IMREAD_GRAYSCALE)
        assert a.shape == (360, 640)
        ious.append(iou(a > 127, gts[i] > 0.5))
    assert min(ious) >= 0.9, ious


def test_e7a_second_exr_shot_without_backlit(tmp_path, monkeypatch):
    """Another resolution (2048 x 858 scope), other frame numbers, no backlit / clear plate."""
    from fastapi.testclient import TestClient
    import app as app_module
    w, h, n, f0 = 1024, 429, 10, 2001   # half of 2048x858 to keep the test fast
    rng = np.random.default_rng(5)
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0) * 0.35 + 0.25
    pdir = tmp_path / "plates" / "SH040"
    pdir.mkdir(parents=True)
    gts = {}
    for i in range(n):
        a = puppet_mask(w, h, 300 + 20 * i, 260, 0.8)
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        cv2.imwrite(str(pdir / f"SH040.{f0 + i}.exr"), ((img ** 2.2) * 1.5).astype(np.float32))
        gts[f0 + i] = a
    work = tmp_path / "work" / "SH040"
    work.mkdir(parents=True)
    mf = work / "shot_manifest_SH040.json"
    mf.write_text(json.dumps({"shot": "SH040", "project": "SHOW", "fps": 25, "first": f0, "last": f0 + n - 1,
                              "plate_w": w, "plate_h": h, "colorspace": "ACEScg", "source_type": "exr_seq", "bit_depth": 16,
                              "plates": {"beauty": str(pdir / "SH040.####.exr")}, "work": str(work), "objects": []}),
                  encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", _client(tmp_path))
    monkeypatch.setattr(app_module, "_plate_display", lambda x: np.clip(x / 1.5, 0, None) ** (1 / 2.2))
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": np.clip(
        cv2.imread(str(pdir / f"SH040.{f}.exr"), cv2.IMREAD_UNCHANGED) / 1.5, 0, None) ** (1 / 2.2))
    frames = list(range(f0, f0 + n))
    fz, ex = _full_flow(api, frames, "felt puppet", f0 + 5)
    assert fz["report"]["edge_mode"] in ("matanyone", "core")   # no backlit in this shot
    obj = ex["outputs"]["objects"][0]
    assert obj["matte"].endswith(".exr") and ex["outputs"]["frame_range"] == [f0, f0 + n - 1]
    from engine.exr_io import read_exr
    for f in frames:
        ch, _ = read_exr(obj["matte"].replace("####", f"{f:04d}"))
        assert ch["R"].shape == (h, w)
        assert iou(ch["R"] > 0.5, gts[f] > 0.5) >= 0.9
