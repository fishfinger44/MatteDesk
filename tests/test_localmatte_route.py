# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 2 of docs/PLAN_HAIR_TOOL.md (section 4, synthetic): engine routing per fragment and the seamless blend.

Two halves of one zone – uniform navy on the left, motley on the right – the weight map separates them (chroma left,
knn right), no alpha jump in the transition band (< 0.05 on background pixels), hair alpha > 0.6 on both sides; a preset
with a single engine gives bitwise the forced engine's result; presets / defaults.json are read.
"""

import cv2
import numpy as np

from engine.localmatte import local_matte
from engine.localmatte_route import (AVAILABLE, calibrate, feather_weights, localmatte_defaults, seam_qa)
from tests.test_localmatte import FG, NAVY, _motley


def _two_halves(h=256, w=384, seed=3):
    bg = _motley(h, w, seed)
    bg[:, : w // 2] = NAVY
    rng = np.random.default_rng(seed)
    bg = np.clip(bg + rng.normal(0, 0.01, bg.shape).astype(np.float32), 0, 1)
    img = bg.copy()
    core = np.zeros((h, w), np.uint8)
    cv2.rectangle(core, (40, h - 70), (w - 40, h - 1), 1, -1)       # a block at the bottom = the object
    hair = np.zeros((h, w), np.uint8)
    for x in range(48, w - 48, 9):                                  # vertical strands growing up from the block
        cv2.line(hair, (x, h - 70), (x + int(rng.integers(-6, 7)), h - 70 - int(rng.integers(30, 60))), 1, 2)
    img[core > 0] = FG
    img[hair > 0] = FG
    region = np.zeros((h, w), np.uint8)
    region[h - 140: h - 60, 30: w - 30] = 1
    return img, core.astype(bool), hair.astype(bool) & ~core.astype(bool), region.astype(bool)


def test_two_halves_weights_and_no_seam():
    img, core, hair, region = _two_halves()
    res = local_matte(img, core, region, engine="auto", preset="hair")
    w = res.weights
    assert set(w) == {"chroma", "knn"}
    h, W = img.shape[:2]
    unk = region & ~core
    left, right = unk & (np.arange(W)[None, :] < W // 2 - 40), unk & (np.arange(W)[None, :] > W // 2 + 40)
    assert w["chroma"][left].mean() > 0.7, w["chroma"][left].mean()      # uniform -> chroma
    assert w["knn"][right].mean() > 0.7, w["knn"][right].mean()          # motley -> knn
    a = res.alpha
    assert a[hair & left].mean() > 0.6 and a[hair & right].mean() > 0.6
    bgpx = unk & ~hair
    assert a[bgpx & left].mean() < 0.1 and a[bgpx & right].mean() < 0.15
    # no jump across the transition band: horizontal alpha steps on background pixels < 0.05
    band = unk & (np.abs(np.arange(W)[None, :] - W // 2) <= 24)
    dil_hair = cv2.dilate(hair.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bb = band & ~dil_hair
    dx = np.abs(np.diff(a, axis=1))
    sel = bb[:, 1:] & bb[:, :-1]
    assert dx[sel].max() < 0.05, dx[sel].max()
    st = res.stats
    assert st["engine_share"]["chroma"] > 0.3 and st["engine_share"]["knn"] > 0.3
    assert "calibration" in st and all(v["ref"] in ("chroma", "knn") for v in st["calibration"].values())
    assert st.get("seam_ratio") is None or st["seam_ratio"] <= 1.5


def test_single_engine_preset_is_bitwise_the_forced_engine():
    img, core, hair, region = _two_halves(seed=4)
    forced = local_matte(img, core, region, engine="knn")
    auto = local_matte(img, core, region, engine="auto", preset="hair", params={"engines": {"chroma": 0.0, "knn": 1.0}})
    assert np.array_equal(forced.alpha, auto.alpha)
    assert list(auto.weights) == ["knn"] and auto.stats["engine_share"] == {"knn": 1.0}
    forced_c = local_matte(img, core, region, engine="chroma")
    auto_c = local_matte(img, core, region, engine="auto", preset="hair", params={"engines": {"chroma": 1.0, "knn": 0.0}})
    assert np.array_equal(forced_c.alpha, auto_c.alpha)


def test_hints_shift_weights():
    img, core, hair, region = _two_halves(seed=5)
    base = local_matte(img, core, region, engine="auto", preset="hair")
    hinted = local_matte(img, core, region, engine="auto", preset="hair", hints={"chroma": 0.2})  # "plamy kolorowe"
    unk = region & ~core
    assert hinted.weights["chroma"][unk].mean() < base.weights["chroma"][unk].mean() - 0.1


def test_defaults_and_presets_loaded():
    d = localmatte_defaults()
    for k in ("hair", "fibres", "smoke", "motion", "thin"):
        assert k in d["presets"] and "engines" in d["presets"][k]
    assert d["presets"]["thin"].get("keep_islands") is True
    assert d["feather_px"] >= 16 and d["patch_px"] == 32
    assert set(AVAILABLE) == {"clean", "chroma", "knn"}


def test_feather_and_calibration_helpers():
    unk = np.zeros((64, 64), bool)
    unk[8:56, 8:56] = True
    wa = np.zeros((64, 64), np.float32); wa[:, :32] = 1.0; wa *= unk
    wb = np.zeros((64, 64), np.float32); wb[:, 32:] = 1.0; wb *= unk
    f = feather_weights({"a": wa, "b": wb}, unk, 16)
    tot = f["a"] + f["b"]
    assert np.allclose(tot[unk], 1.0, atol=1e-4) and tot[~unk].max() == 0.0
    assert 0.2 < f["a"][30, 32] < 0.8                       # soft transition, >= 16 px wide
    assert f["a"][30, 10] > 0.95 and f["b"][30, 54] > 0.95
    ref = np.random.default_rng(1).uniform(0.1, 0.9, (64, 64)).astype(np.float32)
    other = np.clip((ref - 0.1) / 1.25, 0, 1).astype(np.float32)      # ref = 1.25 * other + 0.1
    g, o, n = calibrate(ref, other, unk)
    assert n > 50 and abs(g - 1.25) < 0.02 and abs(o - 0.1) < 0.02
    a = np.where(unk, ref, 0.0).astype(np.float32)
    assert seam_qa(a, f, unk)["boundary_px"] > 0
