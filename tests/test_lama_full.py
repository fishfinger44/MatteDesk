# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""LaMa at full resolution (engine/lama_full.py + engine/lama_worker.py, user 2026-11-10): the worker protocol with its
numpy fake, the size cap, and the regeneration engine "lama" using it with the ComfyUI node as the fallback."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import lama_full  # noqa: E402

FAKE_ENV = {**os.environ, "MD_LAMA_FAKE": "1"}


@pytest.fixture
def cfg(tmp_path):
    md = tmp_path / "models"
    (md / "inpaint").mkdir(parents=True)
    (md / "inpaint" / "big-lama.pt").write_bytes(b"fake")
    return {"models_dir": str(md)}


def _scene(h=90, w=120):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([0.3 + 0.2 * xx / w, 0.5 + 0.1 * yy / h, np.full((h, w), 0.6, np.float32)]).astype(np.float32)
    m = np.zeros((h, w), np.float32)
    m[30:60, 40:80] = 1
    bad = img.copy()
    bad[m > 0] = (1.0, 0.0, 1.0)
    return img, bad, m


def test_worker_fills_the_hole_and_keeps_the_rest(cfg):
    img, bad, m = _scene()
    w = lama_full.LamaWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        out = lama_full.inpaint(bad, m, cfg, worker=w)
        assert w.device == "fake"
        assert out.shape == bad.shape and out.dtype == np.float32
        assert np.array_equal(out[m == 0], bad[m == 0])                     # outside the mask untouched
        assert np.abs(out[m > 0] - img[m > 0]).mean() < 0.05                 # the hole got its surroundings
        assert (out[m > 0][:, 1] > 0.3).all()                                 # no magenta left
    finally:
        w.close()


def test_big_crop_runs_at_max_side_and_back(cfg, monkeypatch):
    img, bad, m = _scene(180, 240)
    monkeypatch.setattr(lama_full, "MAX_SIDE", 120)
    w = lama_full.LamaWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        out = lama_full.inpaint(bad, m, cfg, worker=w)
        assert out.shape == bad.shape and np.array_equal(out[m == 0], bad[m == 0])
        assert np.abs(out[m > 0] - img[m > 0]).mean() < 0.06
    finally:
        w.close()


def test_scale_keeps_small_holes_at_full_resolution_and_caps_big_ones():
    m = np.zeros((1344, 1024), np.float32)
    m[300:500, 300:400] = 1                                                   # 200 px hole: full resolution
    assert lama_full.scale_for(m) == 1.0
    m[214:1169, 332:692] = 1                                                  # 955 px hole (sh060): down to HOLE_MAX
    assert abs(lama_full.scale_for(m) - lama_full.HOLE_MAX / 955.0) < 1e-6
    assert abs(lama_full.scale_for(np.zeros((4096, 1000), np.float32)) - lama_full.MAX_SIDE / 4096.0) < 1e-6


def test_missing_model_is_refused(tmp_path):
    with pytest.raises(lama_full.LamaFullError, match="big-lama.pt"):
        lama_full.inpaint(np.zeros((8, 8, 3), np.float32), np.ones((8, 8), np.float32), {"models_dir": str(tmp_path)})
    assert lama_full.model_path({"models_dir": str(tmp_path)}) is None


def test_regen_lama_uses_full_resolution_then_falls_back_to_the_node(tmp_path, monkeypatch):
    """The engine "lama" sends the crop to lama_full at its own size; when the worker fails, the ComfyUI node path runs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_cleanup3d_regen as TR
    from engine.cleanup3d import regen as RG
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    TR._make_key(work)
    seen = []

    def full(crop, m, cfg=None, **kw):
        seen.append(crop.shape)
        out = np.asarray(crop, np.float32).copy()
        out[np.asarray(m) > 0] = (0.25, 0.75, 0.25)
        return out
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: True)
    monkeypatch.setattr(lama_full, "inpaint", full)
    fake = TR.Fake(tmp_path / "comfy")
    r = TR._run(work, fake, "lama", tmp_path)
    assert seen and not fake.client.sent                                     # no ComfyUI prompt
    img = TR._u8(work, r["version"]).astype(np.float32) / 255
    area = TR._rect(TR.AREA) > 0
    inner = cv2.erode(area.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(img[inner][:, 1] - 0.75).max() < 0.05                     # the full-resolution result pasted

    def broken(*a, **k):
        raise lama_full.LamaFullError("worker LaMa: no CUDA")
    monkeypatch.setattr(lama_full, "inpaint", broken)
    fake2 = TR.Fake(tmp_path / "comfy2")
    TR._run(work, fake2, "lama", tmp_path)
    assert fake2.client.sent                                                  # fell back to the node


@pytest.mark.real
def test_real_lama_full_resolution_cuda(tmp_path):
    """On the station: big-lama.pt in ComfyUI's Python at 1024 x 1344 (the node would work at 256 x 256)."""
    from engine.config import load_config
    cfg = load_config()
    if not lama_full.available(cfg):
        pytest.skip("big-lama.pt or ComfyUI's Python not found")
    p = Path(r"D:/MD_REAL_PROJECT/Documents\shots\shots\matte_desk\source\sh060\sh060_0001.png")
    if not p.is_file():
        pytest.skip("sh060 plate not on this machine")
    img = cv2.imread(str(p)).astype(np.float32)[816:2160, 1798:2822] / 255
    m = np.zeros(img.shape[:2], np.float32)
    m[214:1169, 332:692] = 1
    out = lama_full.inpaint(img, m, cfg)
    try:
        assert out.shape == img.shape and np.array_equal(out[m == 0], img[m == 0])
        assert np.abs(out[m > 0] - img[m > 0]).mean() > 0.01
        # a 955 px hole runs at HOLE_MAX (at full size big-lama paints a grey blot there): the hole takes the colour
        # of its surroundings
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & (m == 0)
        assert np.abs(out[m > 0].mean(0) - img[ring].mean(0)).max() < 0.03
    finally:
        lama_full.close()
