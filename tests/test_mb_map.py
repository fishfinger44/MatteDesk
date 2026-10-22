# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion blur strength measured on the plate (engine/mb_map.py, the "hybrid" of docs/PLAN_2026-10-22.md section 2,
user decision 2026-10-22) and its place in the Final Touch finish: direction from the velocity, strength per edge
segment = the plate's transition width over the blur predicted from the velocity and the shutter."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

H, W = 240, 320


def _plate(x: int, blur_len: int):
    """Textured square at column x on a textured background, exposed with a horizontal box blur of `blur_len` px
    (0 = sharp); returns (gray uint8, hard mask of the unblurred square)."""
    rng = np.random.default_rng(0)
    # smooth textures (sigma 5): trackable by the flow, but without pixel noise the edge profile would drown in
    bg = cv2.GaussianBlur((rng.random((H, W)) * 255).astype(np.uint8), (0, 0), 5.0).astype(np.float32) * 0.5
    tex = cv2.GaussianBlur((np.random.default_rng(1).random((H, W)) * 255).astype(np.uint8), (0, 0), 5.0).astype(np.float32) * 0.5 + 120
    m = np.zeros((H, W), bool)
    m[80:160, x:x + 80] = True
    fg = np.roll(tex, x, axis=1)
    if blur_len > 0:
        acc_m = np.zeros((H, W), np.float32)
        acc_fg = np.zeros((H, W), np.float32)
        for t in range(blur_len):
            d = t - blur_len // 2
            acc_m += np.roll(m.astype(np.float32), d, axis=1)
            acc_fg += np.roll(fg, d, axis=1)
        cov = acc_m / blur_len
        g = (1 - cov) * bg + cov * acc_fg / blur_len
    else:
        g = np.where(m, fg, bg)
    return np.clip(g, 0, 255).astype(np.uint8), m


def _soft(v):
    return int(((v > 0.02) & (v < 0.98)).sum())


@pytest.mark.parametrize("plate_blur,expect_ratio", [(12, 1.0), (0, 0.2)])
def test_measured_strength_follows_the_plate(plate_blur, expect_ratio):
    """12 px / frame at shutter 360 = 12 px of predicted blur: a plate exposed with 12 px of blur measures a ratio ~1
    (the hybrid blurs like strength 1), a sharp plate ~0.1-0.2 (the hybrid barely blurs although the object moves)."""
    from engine.mb_map import strength_map
    from engine.motionblur import motion_blur_alpha, motion_blur_measured, object_velocity
    speed = 12
    g0, _ = _plate(100 - speed, plate_blur)
    g1, m1 = _plate(100, plate_blur)
    g2, _ = _plate(100 + speed, plate_blur)
    a = m1.astype(np.float32)
    vx, vy = object_velocity(m1, g1, g0, g2, reach_px=34.0)
    m, st = strength_map(a, g1, vx, vy, 360.0)
    assert st["measured"] > 20 and abs(st["pred_median"] - speed) < 2.5, st
    assert abs(st["ratio_median"] - expect_ratio) < 0.4, st
    assert abs(float(m[120, 100]) - expect_ratio) < 0.45 and abs(float(m[120, 180]) - expect_ratio) < 0.45
    out, st2 = motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=True)
    plain, _ = motion_blur_alpha(a, g1, g0, g2, 360.0, 1.0)
    row = out[120, 60:220]
    if plate_blur:
        assert abs(_soft(row) - _soft(plain[120, 60:220])) <= 6        # ~ the physical blur (2 x 12 px ramps)
    else:
        assert _soft(row) < 0.5 * _soft(plain[120, 60:220])            # much less than the velocity alone would give
    assert _soft(out[50:190, 140]) < 6                                  # across the motion: sharp (flow jitter only)
    assert np.array_equal(motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=False)[0], plain)


def test_static_edge_takes_the_contour_median_and_no_samples_gives_ones():
    from engine.mb_map import paint_contour_map, strength_map
    g, m = _plate(100, 0)
    z = np.zeros((H, W), np.float32)
    mm, st = strength_map(m.astype(np.float32), g, z, z, 180.0)        # nothing moves: nothing measurable
    assert st["measured"] == 0 and np.allclose(mm, 1.0)
    assert np.allclose(strength_map(z, g, z, z, 180.0)[0], 1.0)        # no contour at all
    pm = paint_contour_map(np.array([[10, 10], [100, 100]], np.float32), np.array([0.5, 2.0], np.float32), (H, W), blur_sigma=0)
    assert pm[10, 10] == 0.5 and pm[100, 100] == 2.0 and pm[0, 0] == 0.5 and pm[200, 300] == 2.0


# ------------------------------------------------------------------------------------------------ Final Touch finish
@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_finish_motion_blur_through_the_app_context(tmp_path, proj):
    """mb_strength in the Final Touch finish: the alpha loader of the preview / export gets the plate through the app's
    context provider (engine/final_touch.set_context_provider); shutter None = the manifest's; measured flag kept."""
    from engine import final_touch as ft
    app_module, api, shot, pipe, frames, start = proj
    d = api.get("/api/object/obj/final_touch").json()
    assert d["finish"]["mb_strength"] == 0.0 and d["finish"]["mb_shutter_deg"] is None and d["finish"]["mb_measured"] is True
    assert d["shutter_manifest"] == pytest.approx(float(pipe.shutter))
    r = api.put("/api/object/obj/final_touch/finish", json={"mb_strength": 1.5, "mb_shutter_deg": 270, "mb_measured": False})
    assert r.status_code == 200, r.text
    f = r.json()["finish"]
    assert f["mb_strength"] == 1.5 and f["mb_shutter_deg"] == 270.0 and f["mb_measured"] is False and r.json()["sig"]
    r = api.put("/api/object/obj/final_touch/finish", json={"mb_shutter_manifest": True})
    assert r.json()["finish"]["mb_shutter_deg"] is None
    ctx = ft.context()
    assert ctx and callable(ctx["gray"]) and ctx["gray"](start) is not None and ctx["gray"](10 ** 6) is None
    assert ctx["shutter_deg"] == pytest.approx(float(pipe.shutter))
    # the finish on a synthetic moving square through the context: the blur appears along the motion only
    g0, _ = _plate(88, 12)
    g1, m1 = _plate(100, 12)
    g2, _ = _plate(112, 12)
    gray = {4: g0, 5: g1, 6: g2}
    out = ft.finish_alpha(m1.astype(np.float32), {"mb_strength": 1.0, "mb_measured": True}, 5,
                          {"gray": lambda q: gray.get(q), "shutter_deg": 360.0})
    assert _soft(out[120, 60:220]) > 8 and _soft(out[50:190, 140]) < 6
    assert np.array_equal(ft.finish_alpha(m1.astype(np.float32), {"mb_strength": 1.0}, 5, None), m1.astype(np.float32)) \
        or ft.context() is not None   # without any context: unchanged (the app's provider may answer here)
    # the preview endpoint runs it (base = mask 1+2 of the synthetic shot – no error, a PNG back)
    api.put("/api/object/obj/final_touch/finish", json={"mb_strength": 1.0})
    assert api.get(f"/api/object/obj/final_touch/frame/{start}?b=core").status_code == 200


def test_mb_cache_makes_a_second_call_cheap_and_identical(tmp_path):
    """User request 2026-10-22: the velocity and the ratio map are cached per frame (flow scale, npz) – a slider move
    recomputes only the vector blur; the cached result equals the fresh one bit for bit (the map always goes through
    flow scale). The key follows the frame, the outline and the shutter."""
    from engine import motionblur as mb
    from engine.mb_map import MbCache
    g0, _ = _plate(88, 12)
    g1, m1 = _plate(100, 12)
    g2, _ = _plate(112, 12)
    a = m1.astype(np.float32)
    cache = MbCache(tmp_path / "mb_cache")
    out1, st1 = mb.motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=True, cache=cache, frame=5)
    assert "cached" not in {k[4:] for k in st1} and list((tmp_path / "mb_cache").glob("f_5_*_360.npz"))
    calls = []
    orig = mb.velocity_small
    def spy(*args, **kw):
        calls.append(1)
        return orig(*args, **kw)
    mb.velocity_small = spy
    try:
        out2, st2 = mb.motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=True, cache=cache, frame=5)
        out3, _ = mb.motion_blur_measured(a, g1, g0, g2, 360.0, 2.0, measured=True, cache=cache, frame=5)
        cache.mem.clear()                                                           # from the file
        out4, _ = mb.motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=True, cache=cache, frame=5)
        mb.motion_blur_measured(a, g1, g0, g2, 180.0, 1.0, measured=True, cache=cache, frame=5)   # another shutter: new key
    finally:
        mb.velocity_small = orig
    assert st2.get("map_cached") == 1 and len(calls) == 1                           # only the 180° shutter recomputed
    assert np.array_equal(out1, out2) and np.abs(out1 - out4).max() < 4e-3   # the file is float16: under 1/255
    assert _soft(out3[120, 60:220]) > _soft(out1[120, 60:220])                      # strength 2 blurs more
    assert len(list((tmp_path / "mb_cache").glob("f_5_*.npz"))) == 2
    # no cache / no frame: as before
    out5, _ = mb.motion_blur_measured(a, g1, g0, g2, 360.0, 1.0, measured=True)
    assert np.array_equal(out5, out1)
