# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The model's mask covers the C1 composite band, band points included (user 2026-11-23, SD160_150 v004 F27: a band
point's whisker reached the frame corner, the model's mask = object + 24 px did not; the band was cut at what the model
drew and a prop of the plate stayed in the corner). New runs: `gen_band` True. A version saved before has no key – its
recomposite keeps the old rule; a recomposite with another band keeps the SOURCE's drawn area (`gen_band_src`).
Synthetic."""
import cv2
import numpy as np

from engine import cleanup as cu
from engine import inpaint_recompose as IR

W, H = 240, 180


def _obj():
    """An object touching the bottom edge, ending 50 px short of the bottom-right corner (as rig_01 on SD160_150)."""
    m = np.zeros((H, W), np.float32)
    m[140:, 60:190] = 1.0
    return m


PTS = [{"id": "p1", "keys": {"1": {"x": 189.0, "y": 170.0, "out": 80.0, "in": 0.0}}}]   # whisker into the corner


def _s(**kw):
    return cu.settings({"remove": ["rig"], "model": "vace", "grow_px": 6, "soft_px": 4, "gen_margin_px": 8,
                        "band_points": PTS, "band_ref_out": 6, **kw})


def test_model_mask_covers_the_band_points_to_the_frame_corner():
    obj = _obj()
    s = _s()
    assert s["gen_band"] is True and cu.settings()["gen_band"] is True        # new runs
    band = cu.mask_fn(s, [lambda f: obj], [])(1)
    assert band[H - 1, W - 12:].min() > 0                                     # the whisker's band reaches the corner
    g = cu.gen_mask_fn(s, [lambda f: obj], [])(1)
    assert set(np.unique(g)) <= {0.0, 1.0}                                    # still hard
    assert (g[band > 0] == 1).all()                                           # the model draws the whole band
    comp = cu.comp_mask_fn(s, [lambda f: obj], [], 0.0)(1)
    assert np.array_equal(comp, band)                                         # nothing of the band is cut
    old = cu.gen_mask_fn({**s, "gen_band": False}, [lambda f: obj], [])(1)    # the 2026-11-13 rule
    assert old[H - 1, W - 1] == 0
    assert cu.comp_mask_fn({**s, "gen_band": False}, [lambda f: obj], [], 0.0)(1)[H - 1, W - 1] == 0


def test_margin_counts_from_the_band_edge():
    obj = _obj()
    s = _s(band_points=[], grow_px=20, soft_px=10, gen_margin_px=15)
    band = cu.mask_fn(s, [lambda f: obj], [])(1) > 0
    g = cu.gen_mask_fn(s, [lambda f: obj], [])(1)
    d = cv2.distanceTransform((~band).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    assert (g[d <= 14.5] == 1).all() and (g[d > 15.5] == 0).all()             # band + 15 px, not object + 15 px
    assert (cu.gen_mask_fn({**s, "gen_margin_px": 0.0}, [lambda f: obj], [])(1) == band).all()   # 0 = the band itself


def test_kept_object_does_not_cut_the_model_band():
    obj = _obj()
    keep = np.zeros_like(obj)
    keep[150:, 200:] = 1.0
    s = _s(holdout=True)
    g = cu.gen_mask_fn(s, [lambda f: obj], [lambda f: keep])(1)
    band = cu.mask_fn({**s, "holdout": False}, [lambda f: obj], [])(1)
    assert (g[band > 0] == 1).all()                                           # drawn under the kept one too


def test_current_settings_never_carry_a_frozen_band(tmp_path):
    work = tmp_path / "md"
    work.mkdir()
    cu.save_settings(work, _s(gen_band=False, gen_band_src={"grow_px": 1.0}))
    s = cu.load_settings(work)
    assert s["gen_band"] is True and s["gen_band_src"] is None


def test_recompose_keeps_what_the_source_model_drew():
    old_vs = {k: v for k, v in _s().items() if k not in ("gen_band", "gen_band_src")}   # a version before 2026-11-23
    cur = _s(band_points=[])
    assert IR.recompose_settings(cur, old_vs)["gen_band"] is False
    new_vs = _s()                                                                         # a run with gen_band
    r = IR.recompose_settings({**cur, "grow_px": 2.0}, new_vs)
    assert r["gen_band"] is True and r["band_points"] == [] and r["gen_band_src"]["band_points"] == PTS
    obj = _obj()
    g = cu.gen_mask_fn(r, [lambda f: obj], [])(1)
    src_band = cu.mask_fn(new_vs, [lambda f: obj], [])(1)
    assert (g[src_band > 0] == 1).all() and g[H - 1, W - 1] == 1                         # the source's area, not the new band
    rr = IR.recompose_settings(cur, {**r, "band_points": []})                             # a recomposite of a recomposite
    assert rr["gen_band_src"]["band_points"] == PTS


def test_band_limit_says_nothing_is_cut():
    lim = cu.band_limit(_s(grow_px=120), 1920, 1080)
    assert lim["cut"] is False and lim["gen_band"] is True
    assert cu.band_limit(_s(grow_px=120, gen_band=False), 1920, 1080)["cut"] is True
    assert cv2 is not None
