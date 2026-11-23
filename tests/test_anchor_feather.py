"""The anchor's soft edge (user 2026-11-23): a hard latent mask made Qwen-Image-2.1 draw a sharp outline along the
model's mask (SD160 F27). The anchor now fades OUTWARD from the model's mask over the C1 patch band x
ANCHOR_FEATHER_MULT, Qwen gets that soft mask (DifferentialDiffusion) and the paste uses the same weight."""
from pathlib import Path

import numpy as np

from engine import cleanup as cu
from engine import h3_fill as H
from engine.cleanup3d import regen as R


def test_feather_is_outward_only_and_smooth():
    hard = np.zeros((64, 64), np.float32)
    hard[16:48, 16:48] = 1.0
    w = H.anchor_feather(hard, 10.0)
    assert (w[16:48, 16:48] == 1.0).all()                         # never inward: the object's area stays whole
    assert 0.0 < w[32, 50] < 1.0 and w[32, 49] > w[32, 52]         # falls off outside
    assert w[32, 59] == 0.0 and w[0, 0] == 0.0                     # gone beyond the width
    assert (H.anchor_feather(hard, 0.0) == hard).all()             # 0 = the old hard anchor


def test_width_comes_from_the_patch_band():
    s = cu.settings({"grow_px": 8, "soft_px": 3})
    assert H.anchor_feather_px(s, 27) == (8 + 3) * H.ANCHOR_FEATHER_MULT
    s = cu.settings({"grow_px": 8, "soft_px": 3, "band_keys": {"10": {"grow_px": 20, "soft_px": 5}}})
    assert H.anchor_feather_px(s, 10) == 25 * H.ANCHOR_FEATHER_MULT   # keyed band at the anchor's frame
    assert H.anchor_feather_px({}, 1) == 0.0                           # no band: hard


def test_qwen_gets_the_soft_mask_and_the_paste_is_soft(monkeypatch):
    seen = {}

    def fake(client, io, disp, region, prompt, sem, mode, **k):
        seen.update(k, region=region)
        return np.ones_like(disp), None, 1
    monkeypatch.setattr(R, "_run_qwen", fake)
    monkeypatch.setattr("engine.stage_models.comfy_gate", lambda *a, **k: None)
    plate = np.zeros((64, 64, 3), np.float32)
    m = np.zeros((64, 64), np.float32)
    m[16:48, 16:48] = 1.0
    out = H.make_anchor(None, Path("."), plate, m, "p", "replace", feather_px=10.0)
    wt = seen["mask_weight"]
    assert wt is not None and 0.0 < wt[32, 50] < 1.0
    assert seen["region"][32, 50]                                  # the crop covers the ramp
    assert out[32, 30, 0] == 255 and 0 < out[32, 50, 0] < 255 and out[32, 60, 0] == 0
    H.make_anchor(None, Path("."), plate, m, "p", "replace")       # no feather: hard, no soft mask
    assert seen["mask_weight"] is None
    hole = np.zeros((64, 64), np.float32)
    hole[20:30, 20:30] = 1.0
    H.make_anchor(None, Path("."), plate, m, "p", "replace", region=hole, feather_px=10.0)
    assert seen["mask_weight"] is None                             # the warp's holes stay hard


def test_graph_adds_differential_diffusion_only_for_a_soft_mask(tmp_path):
    g = R.qwen_graph(tmp_path, "p", "x", mask_dir=tmp_path / "m", soft_mask=True)
    assert g["24"]["class_type"] == "DifferentialDiffusion" and g["12"]["inputs"]["model"] == ["24", 0]
    hard = R.qwen_graph(tmp_path, "p", "x", mask_dir=tmp_path / "m")
    assert "24" not in hard and hard["12"]["inputs"]["model"] == ["1", 0]
    assert "24" not in R.qwen_graph(tmp_path, "p", "x", soft_mask=True)   # no mask: nothing to soften


def test_api_passes_the_band_width():
    src = (Path(__file__).resolve().parents[1] / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert 'feather = 0.0 if how == "warp" else h3_fill.anchor_feather_px(s, a["frame"])' in src
    assert "feather_px=feather)" in src
