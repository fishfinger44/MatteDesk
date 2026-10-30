# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/merge_matte.py – the Python port of the MergeMatte Nuke group (PLAN_FINAL_TOUCH.md §11).

Synthetic: every node function against its Nuke formula, the group with everything off = plain over, knob plumbing.
`real` (station only): the port vs the renders of the real MergeMatte group from the live Nuke session
(tools/readme_compare/mm_nuke_match.py; inputs/outputs in _readme_compare/c5_nuke_match, user renders the Writes)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from engine import merge_matte as fuse

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / "_readme_compare/c5_nuke_match"


def _rgba(h=24, w=32, seed=0):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.random((h, w), np.float32) * 1.4 - 0.2, 0, 1)
    rgb = rng.random((h, w, 3), np.float32) * 0.8
    return np.dstack([rgb * a[..., None], a]).astype(np.float32)


def test_unpremult_premult_round_trip_and_zero_alpha_untouched():
    img = _rgba()
    img[0, 0] = (0.3, 0.2, 0.1, 0.0)
    u = fuse.unpremult(img)
    assert np.allclose(u[0, 0], (0.3, 0.2, 0.1, 0.0))            # alpha 0 → rgb left as is (Nuke)
    nz = img[..., 3] > 0
    assert np.allclose(fuse.premult(u)[nz], img[nz], atol=1e-6)


def test_acescct_round_trip_and_known_points():
    x = np.array([0.0, 0.0078125, 0.18, 1.0, 16.0], np.float32)
    y = fuse.lin_to_acescct(x)
    assert np.allclose(fuse.acescct_to_lin(y), x, rtol=1e-5, atol=1e-7)
    assert abs(float(y[2]) - 0.4135884) < 1e-5                   # ACEScct of 0.18
    assert abs(float(y[3]) - 0.5547945) < 1e-5                   # ACEScct of 1.0
    assert np.all(np.diff(fuse.lin_to_acescct(np.linspace(-0.1, 2, 500))) > 0)


def test_srgb_acescg_round_trip():
    c = np.random.default_rng(1).random((5, 7, 3)).astype(np.float32)
    assert np.allclose(fuse.acescg_to_srgb(fuse.srgb_to_acescg(c)), c, atol=2e-4)


def test_merge_formulas():
    a, b = _rgba(seed=2), _rgba(seed=3)
    aa, ba = a[..., 3:4], b[..., 3:4]
    assert np.allclose(fuse.merge_over(a, b), a + b * (1 - aa))
    assert np.allclose(fuse.merge_xor(a, b), a * (1 - ba) + b * (1 - aa))
    assert np.allclose(fuse.merge_mask(a, b), b * aa)
    assert np.allclose(fuse.merge_min(a, b), np.minimum(a, b))
    d = fuse.merge_divide(a, b)
    nz = b != 0
    assert np.allclose(d[nz], (a / np.where(nz, b, 1))[nz]) and np.all(d[~nz] == 0)


def test_keymix_endpoints_and_channels():
    b, a = _rgba(seed=4), _rgba(seed=5)
    m = np.zeros(b.shape[:2], np.float32)
    assert np.allclose(fuse.keymix(b, a, m), b)
    m[:] = 1
    assert np.allclose(fuse.keymix(b, a, m), a)
    half = fuse.keymix(b, a, m, 0.5, "rgb")
    assert np.allclose(half[..., :3], (a[..., :3] + b[..., :3]) / 2) and np.allclose(half[..., 3], b[..., 3])


def test_blur_keeps_energy_and_pads_black():
    img = np.zeros((65, 65, 4), np.float32)
    img[32, 32] = 1
    out = fuse.blur(img, 5.0)
    assert abs(float(out[..., 0].sum()) - 1) < 1e-3 and float(out[32, 32, 0]) < 1
    assert fuse.blur(img, 0).tolist() == img.tolist()
    edge = np.ones((33, 33, 4), np.float32)
    assert float(fuse.blur(edge, 10)[0, 0, 0]) < 0.5            # black outside the image, as Nuke's bbox padding


def test_blur_masked_mixes_by_mask():
    img = _rgba()
    m = np.zeros(img.shape[:2], np.float32)
    assert np.allclose(fuse.blur_masked(img, 3, m), img)
    m[:] = 1
    assert np.allclose(fuse.blur_masked(img, 3, m), fuse.blur(img, 3))


def test_saturation_zero_is_rec709_luminance():
    rgb = _rgba(seed=6)[..., :3]
    lum = (rgb * fuse.REC709).sum(-1)
    s0 = fuse.saturation(rgb, 0)
    assert np.allclose(s0[..., 0], lum) and np.allclose(s0[..., 2], lum)
    assert np.allclose(fuse.saturation(rgb, 1), rgb)


def test_keyer_luminance_ramp():
    ramp = np.linspace(0, 1, 101, dtype=np.float32)[None, :, None].repeat(3, 2)
    k = fuse.keyer_luminance(ramp, 0.2, 0.4, 0.6, 0.8)[0]
    assert k[10] == 0 and abs(k[30] - 0.5) < 1e-5 and k[50] == 1 and abs(k[70] - 0.5) < 1e-5 and k[90] == 0


def test_filter_erode_shrinks_and_expands_a_disc():
    yy, xx = np.mgrid[0:101, 0:101]
    disc = (np.hypot(xx - 50, yy - 50) < 30).astype(np.float32)
    assert 0.6 < fuse.filter_erode(disc, 6).sum() / disc.sum() < 0.7
    assert 1.3 < fuse.filter_erode(disc, -6).sum() / disc.sum() < 1.5
    assert fuse.filter_erode(disc, 0).tolist() == disc.tolist()


def _all_off(**over):
    """Every section off."""
    k = {n: 0 for n in fuse.KNOB_DEFAULTS}
    k.update(mix=1, log=0)
    k.update(over)
    return k


def test_everything_off_is_a_plain_over_linear_and_log():
    fg, bg = _rgba(seed=7), _rgba(seed=8)
    bg[..., 3] = 0
    out = fuse.merge_matte(fg, bg, _all_off())
    assert np.allclose(out[..., :3], fuse.merge_over(fg, bg)[..., :3], atol=1e-6)
    out_log = fuse.merge_matte(fg, bg, _all_off(log=1))
    assert np.abs(out_log[..., :3] - out[..., :3]).max() > 1e-3   # log over differs from linear over in the edge…
    solid = fg[..., 3] > 0.999
    if solid.any():                                                # …but not where the FG is opaque
        assert np.allclose(out_log[solid, :3], out[solid, :3], atol=1e-4)


def test_bypass_and_mix():
    fg, bg = _rgba(seed=9), _rgba(seed=10)
    bg[..., 3] = 0
    k = fuse.knobs({"bypass": 1, "log": 0, "choke": 0, "soften": 0})   # the bypass sits after Alpha / Log
    assert np.allclose(fuse.merge_matte(fg, bg, k)[..., :3], fuse.merge_over(fg, bg)[..., :3], atol=1e-6)
    assert np.allclose(fuse.merge_matte(fg, bg, fuse.knobs({"mix": 0}))[..., :3], bg[..., :3], atol=1e-6)


def test_choke_only_changes_alpha_and_interior_colour_stays():
    fg, bg = _rgba(seed=11), _rgba(seed=12)
    fg[..., 3] = 1
    fg[:, :8, 3] = 0
    fg[..., :3] *= fg[..., 3:4]
    bg[..., 3] = 0
    out = fuse.merge_matte(fg, bg, _all_off(choke=3, soften=1))
    assert np.allclose(out[4:-4, 24:-4, :3], fg[4:-4, 24:-4, :3], atol=1e-6)   # far from the edge (and the image border, padded black like Nuke's bbox) nothing happens
    assert float(out[4:-4, 9:12, 3].mean()) < 0.5                     # the edge moved inward


def test_knobs_only_accept_mergematte_names_and_ranges_cover_them():
    k = fuse.knobs({"choke": 2, "nonsense": 5})
    assert k["choke"] == 2 and "nonsense" not in k
    assert set(fuse.KNOB_RANGES) == set(fuse.KNOB_DEFAULTS) - {"log", "bypass"}


def test_comp_srgb_interior_is_the_plate():
    rng = np.random.default_rng(13)
    plate = rng.random((200, 200, 3), np.float32) * 0.6 + 0.2
    a = np.zeros((200, 200), np.float32)
    a[40:160, 40:160] = 1
    out = fuse.comp_srgb(plate, a, (0.1, 0.16, 0.29), fuse.PRESET_SH010)   # SH010 preset: choke 6 + darken band 23 px reach ~40 px inward
    assert np.abs(out[90:110, 90:110] - plate[90:110, 90:110]).max() < 2 / 255
    assert np.abs(out[0, 0] - (0.1, 0.16, 0.29)).max() < 2 / 255


# ---------------------------------------------------------------- real: against the live Nuke renders
def _outs():
    return sorted((MATCH / "out").glob("*.exr")) if (MATCH / "out").exists() else []


@pytest.mark.real
def test_port_matches_nuke_merge_matte_v01():
    """Max |Δ| ≤ 1/255 and mean ≤ 0.1/255 (8-bit sRGB) on the 3 frames + the alt knob set (plan §11, step 1)."""
    if not any(p.name.startswith("nuke_") for p in _outs()):
        pytest.skip("no Nuke renders in _readme_compare/c5_nuke_match/out – build + render first (mm_nuke_match.py)")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "tools/readme_compare/mm_nuke_match.py"), "compare"], check=True,
                   cwd=ROOT, env={**os.environ, "OPENCV_IO_ENABLE_OPENEXR": "1"})
    rep = json.loads((MATCH / "metrics.json").read_text("utf-8"))
    assert rep["comp"], rep
    for name, m in rep["comp"].items():
        assert m["max_8bit"] <= 1.0, (name, m)
        assert m["mean_8bit"] <= 0.1, (name, m)


@pytest.mark.real
def test_blur_size_to_sigma_real():
    """The Blur `size` → sigma constant in the port equals the one measured on Nuke's impulse responses (±2 %)."""
    if not (MATCH / "out/blur_5.exr").exists():
        pytest.skip("no calibration renders")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "tools/readme_compare/mm_nuke_match.py"), "compare"], check=True, cwd=ROOT)
    rep = json.loads((MATCH / "metrics.json").read_text("utf-8"))
    ratios = [m["sigma_per_size"] for s, m in rep["blur"].items() if float(s) >= 5]
    assert ratios and abs(np.mean(ratios) - fuse.BLUR_SIGMA_PER_SIZE) / fuse.BLUR_SIGMA_PER_SIZE < 0.02, (ratios, fuse.BLUR_SIGMA_PER_SIZE)


def test_blur_sigma_is_the_value_measured_in_nuke():
    assert abs(fuse.BLUR_SIGMA_PER_SIZE - 0.4236) < 1e-4
