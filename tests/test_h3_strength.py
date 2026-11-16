"""MiniMax-H3 „Siła zmiany” (user 2026-11-13): Replace may start from the noised plate under the mask instead of a hole."""
import numpy as np

from engine import cleanup as cu
from engine import h3_fill

FILES = {"unet": "u.safetensors", "clip": "c.safetensors", "vae": "v.safetensors", "turbo": "t.safetensors"}


def test_full_strength_matches_basic_scheduler_beta():
    # strength 1.0 = H3's own 8-step beta schedule with shift 12 (BasicScheduler, up to its 1000-step rounding)
    sig = h3_fill.strength_sigmas(1.0, 8)
    ref = [1.0, 0.995, 0.983, 0.961, 0.923, 0.855, 0.719, 0.42, 0.0]
    assert len(sig) == 9
    assert np.allclose(sig, ref, atol=0.01)


def test_lower_strength_starts_there_and_keeps_steps():
    for s in (0.9, 0.7, 0.5, 0.3):
        sig = h3_fill.strength_sigmas(s, 8)
        assert len(sig) == 9 and abs(sig[0] - s) < 1e-6 and sig[-1] == 0.0
        assert all(a > b for a, b in zip(sig, sig[1:]))


def test_remove_always_a_hole():
    assert h3_fill.run_strength(0.5, "remove") == 1.0
    assert h3_fill.run_strength(0.5, "replace") == 0.5
    assert h3_fill.run_strength(0.01, "replace") == h3_fill.STRENGTH_MIN
    assert h3_fill.run_strength("bad", "replace") == 1.0


def test_graph_uses_manual_sigmas_only_below_one(tmp_path):
    g = h3_fill.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, FILES)
    assert g["24"]["class_type"] == "BasicScheduler"
    g = h3_fill.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, FILES, strength=0.7)
    assert g["24"]["class_type"] == "ManualSigmas"
    vals = [float(v) for v in g["24"]["inputs"]["sigmas"].split(",")]
    assert abs(vals[0] - 0.7) < 1e-5 and len(vals) == h3_fill.STEPS + 1
    assert g["27"]["inputs"]["sigmas"] == ["24", 0]


def test_setting_default_clamp_and_per_target():
    assert cu.settings({})["h3_strength"] == 1.0
    assert cu.settings({"h3_strength": 0.05})["h3_strength"] == h3_fill.STRENGTH_MIN
    assert cu.settings({"h3_strength": 3})["h3_strength"] == 1.0
    assert "h3_strength" in cu.PER_TARGET
