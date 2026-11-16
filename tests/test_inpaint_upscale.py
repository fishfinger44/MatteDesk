# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„⤴ Upscale (SeedVR2)” of an Inpaint version (user decision 2026-11-11): one crop around what the version filled,
taken at the model's scale, SeedVR2 back to the plate's scale, laid back as a key fill only inside the crop.
Synthetic: SeedVR2 and ffmpeg are stand-ins."""
from pathlib import Path

import cv2
import numpy as np

from engine import inpaint_upscale as IU


def _version(tmp_path, frames=(1, 2, 3)):
    v = tmp_path / "v007"
    (v / "comp").mkdir(parents=True)
    (v / "source").mkdir()
    for f in frames:
        cv2.imwrite(str(v / "comp" / f"c_{f:04d}.png"), np.full((200, 120, 3), 90, np.uint8))
        s = np.zeros((200, 120), np.uint8)
        s[80:120, 40:70] = 6
        cv2.imwrite(str(v / "source" / f"s_{f:04d}.png"), s)
    return v


def test_crop_box_and_scale(tmp_path):
    v = _version(tmp_path)
    r = IU.filled_region(v, 1, (200, 120))
    assert r.sum() == 40 * 30
    box = IU.crop_box([r], (200, 120), margin=10)
    x0, y0, cw, ch = box
    assert x0 <= 30 and y0 <= 70 and x0 + cw >= 80 and y0 + ch >= 130 and cw % 16 == 0 and ch % 16 == 0
    assert IU.crop_box([np.zeros((5, 5), np.uint8)], (5, 5)) is None
    assert abs(IU.model_scale("h3", {}, 1080, 1920) - 1344 / 1920) < 1e-6
    assert abs(IU.model_scale("vace", {"vace_res": 480}, 1080, 1920) - 848 / 1920) < 1e-6
    assert IU.version_frames(v) == [1, 2, 3]


def test_upscale_fills_only_the_crop(tmp_path, monkeypatch):
    from engine.cleanup3d import void_engine as VE
    v = _version(tmp_path)
    seen = {}
    monkeypatch.setattr(VE, "encode_video", lambda ff, pat, dst, lossless, **k: seen.update(clip=Path(dst)))

    class Runner:
        def run(self, clip, out, res, batch, n, cancel=None):
            seen.update(res=res, n=n)
            Path(out).mkdir(parents=True)
            for i in range(n):
                cv2.imwrite(str(Path(out) / f"{i:06d}.png"), np.full((10, 10, 3), 250, np.uint8))
            return out
    fill = IU.upscale(v, [1, 2, 3], "h3", {}, False, Runner(), "ffmpeg", tmp_path / "tmp")
    x0, y0, cw, ch = fill.box
    assert seen["n"] == 3 and seen["res"] == min(cw, ch)
    img, wgt = fill(2)
    assert img.shape == (200, 120, 3) and wgt[y0 + 1, x0 + 1] == 1.0 and wgt.sum() == cw * ch
    assert abs(float(img[y0 + 1, x0 + 1, 0]) - 250 / 255) < 1e-3 and abs(float(img[0, 0, 0]) - 90 / 255) < 1e-3
    assert fill(9) is None


def test_ui_and_api_wiring():
    root = Path(__file__).resolve().parent.parent
    js = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "data-cu-up=" in js and "/upscale`" in js
    api = (root / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert '"/api/cleanup/versions/{version}/upscale"' in api and 'cl.report["upscaled_from"] = version' in api
