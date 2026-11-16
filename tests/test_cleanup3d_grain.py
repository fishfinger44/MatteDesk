# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""P17 grain match of the Cleanup 3D patch (engine/cleanup3d/grain.py): measurement of plate / patch noise on flat
areas, the auto decision, synthesis with the plate's band spectrum, new noise every frame (deterministic per frame),
nothing outside the patch – unit tests here, the render integration in tests/test_cleanup3d_render.py."""

import numpy as np
import cv2

from engine.cleanup3d import grain as GR

H, W = 240, 320


def _bg():
    """A smooth background with a little structure (gradient + a slow wave)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 0.45 + 0.15 * np.sin(xx / 40.0) * np.cos(yy / 55.0) + 0.1 * xx / W
    return np.stack([g, g * 0.9 + 0.05, g * 0.8 + 0.1], -1).astype(np.float32)


def _noise(sigma, seed, blur=0.0):
    n = np.random.default_rng(seed).standard_normal((H, W, 3)).astype(np.float32)
    if blur > 0:                                   # coloured (clumped) grain, renormalised to sigma
        n = cv2.GaussianBlur(n, (0, 0), blur)
        n /= n.std()
    return n * sigma


def _hole():
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - 160) / 40.0) ** 2 + ((yy - 120) / 60.0) ** 2) < 1.0


def test_measure_white_noise_levels():
    bg = _bg()
    hole = _hole()
    m = GR.measure(bg + _noise(0.03, 1), bg, ~hole)
    assert m is not None and m["px"] >= GR.MIN_PX
    assert np.all(np.abs(m["plate_sigma"] - 0.03) < 0.15 * 0.03), m["plate_sigma"]
    assert np.all(m["patch_sigma"] < 0.003)
    # a different level per channel is measured per channel
    n = _noise(1.0, 2) * np.array([0.01, 0.02, 0.04], np.float32)
    m = GR.measure(bg + n, bg + _noise(0.005, 3), ~hole)
    assert np.all(np.abs(m["plate_sigma"] - [0.01, 0.02, 0.04]) < 0.15 * np.array([0.01, 0.02, 0.04])), m["plate_sigma"]
    assert np.all(np.abs(m["patch_sigma"] - 0.005) < 0.15 * 0.005)


def test_measure_ignores_the_hole_and_needs_pixels():
    bg = _bg()
    plate = bg + _noise(0.02, 4)
    hole = _hole()
    wild = plate.copy()
    wild[hole] = np.random.default_rng(5).random((int(hole.sum()), 3))       # the object: anything
    a, b = GR.measure(plate, bg, ~hole), GR.measure(wild, bg, ~hole)
    assert np.allclose(a["plate_sigma"], b["plate_sigma"], rtol=0.01)
    tiny = np.zeros((H, W), bool)
    tiny[10:30, 10:30] = True
    assert GR.measure(plate, bg, tiny) is None


def test_decide_auto_off_on():
    bg = _bg()
    region = ~_hole()
    noisy = GR.measure(bg + _noise(0.03, 6), bg, region)
    assert GR.decide(noisy, "auto") and GR.decide(noisy, "on") and not GR.decide(noisy, "off")
    # denoised plate (as clean as the patch): auto adds nothing
    same = GR.measure(bg + _noise(0.004, 7), bg + _noise(0.004, 8), region)
    assert not GR.decide(same, "auto")
    # a little noisier but below the threshold: auto no, "on" yes
    near = GR.measure(bg + _noise(0.005, 9), bg + _noise(0.0045, 10), region)
    assert not GR.decide(near, "auto")
    assert not GR.decide(None, "on")


def _apply(plate, fill, frame, **s):
    hole = _hole()
    patch = np.zeros((H, W), np.float32)
    patch[cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = 0.5
    patch[hole] = 1.0
    st = {}
    out = GR.apply(fill, plate, hole, np.ones((H, W), bool), patch, frame, {"grain": "auto", "grain_strength": 1.0, **s},
                   stats=st)
    return out, st["grain"] if st else None, hole, patch


def test_apply_adds_the_missing_grain_only_in_the_patch():
    bg = _bg()
    plate = bg + _noise(0.03, 11)
    out, rep, hole, patch = _apply(plate, bg, 3)
    assert rep["added"] and abs(np.mean(rep["plate"]) - 0.03) < 0.0045 and np.mean(rep["patch"]) < 0.003
    d = out - bg
    assert np.all(np.abs(d[hole].std(0) - 0.03) < 0.15 * 0.03), d[hole].std(0)
    assert np.abs(d[patch == 0]).max() == 0.0                              # untouched outside the patch
    # strength scales; off = nothing
    half, _, _, _ = _apply(plate, bg, 3, grain_strength=0.5)
    assert abs((half - bg)[hole].std() - 0.015) < 0.15 * 0.015
    off, rep_off, _, _ = _apply(plate, bg, 3, grain="off")
    assert off is bg and rep_off is None
    # a patch already noisy: only what is missing (sqrt(0.03^2 - 0.02^2) ~ 0.0224)
    fill = bg + _noise(0.02, 12)
    out2, rep2, _, _ = _apply(plate, fill, 3)
    assert abs((out2 - fill)[hole].std() - np.sqrt(0.03 ** 2 - 0.02 ** 2)) < 0.15 * 0.0224


def test_clean_plate_adds_nothing():
    bg = _bg()
    out, rep, _, _ = _apply(bg.copy(), bg, 3)
    assert not rep["added"] and np.array_equal(out, bg)
    # denoised plate vs clean patch below MIN_DIFF: nothing
    out, rep, _, _ = _apply(bg + _noise(0.0012, 13), bg, 3)
    assert not rep["added"] and np.array_equal(out, bg)


def test_grain_varies_per_frame_and_is_deterministic():
    bg = _bg()
    plate = bg + _noise(0.03, 14)
    a, _, hole, _ = _apply(plate, bg, 3)
    b, _, _, _ = _apply(plate, bg, 3)
    c, _, _, _ = _apply(plate, bg, 4)
    assert np.array_equal(a, b)
    da, dc = (a - bg)[hole].ravel(), (c - bg)[hole].ravel()
    assert abs(np.corrcoef(da, dc)[0, 1]) < 0.1                            # not a fixed pattern


def test_spectrum_follows_the_plate():
    """Clumped grain (blurred noise) on the plate: the added grain has the same band energies (not white)."""
    bg = _bg()
    plate = bg + _noise(0.03, 15, blur=2.0)
    out, rep, hole, _ = _apply(plate, bg, 5)
    want = GR.measure(plate, bg, ~hole)["plate"]
    got = GR.band_levels(out - bg, hole)                                    # grain is added in the whole patch
    assert np.all(np.abs(got - want) < 0.15 * want), (got, want)
    assert want[0].mean() < 0.6 * want[1].mean()                           # the test grain is really clumped


def test_brightness_dependent_noise():
    """Noise that grows with brightness (as in a real sensor): darker pixels of the patch get less grain."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 0.15 + 0.7 * xx / W
    bg = np.stack([g, g, g], -1).astype(np.float32)
    plate = bg + _noise(1.0, 16) * (0.01 + 0.04 * (g - 0.15) / 0.7)[..., None]
    region = np.ones((H, W), bool)
    m = GR.measure(plate, bg, region)
    assert m["lum"] is not None
    cen, lp, _ = m["lum"]
    assert lp[0] < lp[-1] * 0.6
    gain = GR.lum_gain(m, np.array([0.2, 0.8], np.float32))
    assert gain[0] < gain[1]


def test_settings_cleaning_and_summary():
    assert GR.clean_mode("x") == "auto" and GR.clean_mode("on") == "on"
    assert GR.clean_strength(5) == 2.0 and GR.clean_strength(-1) == 0.0 and GR.clean_strength(None) == 1.0
    s = GR.summary({3: {"grain": {"plate": [0.03, 0.03, 0.03], "patch": [0.001] * 3, "added": True, "mode": "auto"}},
                    4: {"grain": {"plate": None, "patch": None, "added": False, "mode": "auto"}}, 5: {}})
    assert s == {"plate": 0.03, "patch": 0.001, "added": 1, "measured": 1, "frames": 2, "mode": "auto"}
    assert GR.summary({}) is None
