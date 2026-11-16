# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Qwen-Image-2.1's whole picture as the video model's anchor (user test 2026-11-11, setting anchor_raw): the raw
picture is saved next to each generation, its crop blended into the plate at the crop's inner edges."""
import numpy as np

from engine import cleanup as cu
from engine.cleanup3d import regen as R


def test_crop_ramp_soft_inside_whole_at_the_image_border():
    r = R.crop_ramp((10, 0, 110, 50), 200, 50, 20)   # touches the top and bottom image borders
    assert r.shape == (50, 100)
    assert r[25, 0] < 0.05 and r[25, 99] < 0.05 and r[25, 50] == 1.0   # left / right edges inside the image
    assert r[0, 50] == 1.0 and r[49, 50] == 1.0                          # top / bottom = image border
    assert (R.crop_ramp((0, 0, 30, 30), 100, 100, 0) == 1).all()


def test_run_qwen_feather_keeps_the_region_and_softens_the_crop_edge(monkeypatch):
    real = np.zeros((800, 800, 3), np.float32)
    region = np.zeros((800, 800), bool)
    region[380:420, 380:420] = True
    monkeypatch.setattr("engine.comfy_io.require_vhs", lambda c: None)
    monkeypatch.setattr("engine.comfy_io.write_png8", lambda p, im: None)
    monkeypatch.setattr(R, "_comfy_image", lambda c, g, n, name, mw, mh: np.ones((mh, mw, 3), np.float32))
    hard, _w, _n = R._run_qwen(None, __import__("pathlib").Path("."), real, region, "p", None, "replace")
    soft, _w, _n = R._run_qwen(None, __import__("pathlib").Path("."), real, region, "p", None, "replace", feather_px=40)
    assert np.allclose(soft[region], 1, atol=1e-4) and np.allclose(hard[region], 1, atol=1e-4)              # the region: Qwen's, whole
    x0 = int(np.where(hard[400, :, 0] > 0.5)[0].min())                            # the crop's left edge
    assert x0 > 0 and hard[400, x0, 0] > 0.999 and soft[400, x0, 0] < 0.1            # hard step -> a ramp


def test_setting():
    assert cu.settings({})["anchor_raw"] is True                       # default on (user 2026-11-11)
    assert cu.settings({"anchor_raw": False})["anchor_raw"] is False
    assert "anchor_raw" in cu.PER_TARGET
