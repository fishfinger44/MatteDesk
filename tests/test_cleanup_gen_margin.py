# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The C1 band split (user decision 2026-11-13): „Zapas generowania” = the mask the video MODEL gets (the removed alpha
grown by it, hard, no band points, minus the holdout); „Pas łaty” = what of the generated material goes into the
picture, cut at what the model drew (margin + the engine's grow), the UI says so; an older project reads margin =
max(24, grow_px + soft_px)."""
import inspect

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as A
from engine import cleanup as cu
from engine import h3_fill

W, H = 240, 180


def _obj():
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.hypot(xx - 120, yy - 90) < 25).astype(np.float32)


def _dist_from(obj):
    return cv2.distanceTransform((obj <= 0.02).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def test_old_settings_read_margin_from_their_band():
    assert cu.settings()["gen_margin_px"] == 24.0
    assert cu.settings({"grow_px": 30, "soft_px": 10})["gen_margin_px"] == 40.0          # older project: its band
    assert cu.settings({"grow_px": 4, "soft_px": 2})["gen_margin_px"] == 24.0
    assert cu.settings({"grow_px": 30, "soft_px": 10, "gen_margin_px": 12})["gen_margin_px"] == 12.0
    assert cu.settings({"gen_margin_px": -5})["gen_margin_px"] == 0.0
    assert cu.settings({"grow_px": 200, "soft_px": 100})["gen_margin_px"] == 300.0      # never cut an old band
    assert "gen_margin_px" in cu.PER_TARGET
    s = cu.settings({"remove": ["a"], "grow_px": 50, "soft_px": 0, "gen_margin_px": 60})
    cu.switch_target(s, "a", "b")
    s = cu.settings(s)
    assert s["gen_margin_px"] == 24.0 and s["per_target"]["a"]["gen_margin_px"] == 60.0
    cu.switch_target(s, "b", "a")
    assert cu.settings(s)["gen_margin_px"] == 60.0


def test_model_mask_is_the_margin_hard_no_band_points_holdout_cut():
    obj = _obj()
    keep = np.zeros_like(obj)
    keep[:, 150:] = 1.0
    s = cu.settings({"grow_px": 40, "soft_px": 10, "gen_margin_px": 12,
                     "band_points": [{"id": "p1", "keys": {"1": {"x": 145, "y": 90, "out": 90, "in": 0}}}],
                     "band_ref_out": 40})
    g = cu.gen_mask_fn(s, [lambda f: obj], [lambda f: keep])(1)
    d = _dist_from(obj)
    assert set(np.unique(g)) <= {0.0, 1.0}                         # hard
    assert (g[(d <= 11.5) & (keep == 0)] == 1).all() and (g[d > 12.5] == 0).all()
    # 2026-11-15: the kept objects do NOT cut the model's mask (the composite protects them); the old rule = gen_holdout
    assert (g[(d <= 11.5) & (keep > 0)] == 1).all()
    old = cu.gen_mask_fn({**s, "gen_holdout": True}, [lambda f: obj], [lambda f: keep])(1)
    assert (old[keep > 0] == 0).all()
    comp = cu.mask_fn(s, [lambda f: obj], [lambda f: keep])(1)
    assert (comp > 0).sum() > (g > 0).sum()                        # the band (40 + 10 px) is wider than the model's mask


class _Fill:
    label = "MiniMax-H3"
    tone = False
    grain = False
    tone_gap_px = 8.0
    drawn_px = 4.0

    def __call__(self, f):
        return np.full((H, W, 3), 0.25, np.float32), np.ones((H, W), np.float32)


def _cleaner(s, gen=True):
    obj = _obj()
    plate = np.full((H, W, 3), 0.6, np.float32)
    s = cu.settings({"tone": False, **s})
    return cu.Cleaner(s, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj], keep_alphas=lambda f: [],
                      key_fill=_Fill(), gen_mask=cu.gen_mask_fn(s, [lambda f: obj], []) if gen else None), obj


def test_composite_band_is_cut_at_what_the_model_drew(tmp_path):
    cl, obj = _cleaner({"grow_px": 40, "soft_px": 10, "gen_margin_px": 10})
    m = cl.mask(1)
    d = _dist_from(obj)
    assert (m[d > 14.6] == 0).all() and (m[(d <= 13.5)] == 1).all()      # margin 10 + engine 4 px
    rj = cl.run([1, 2], tmp_path / "v001", "png16")
    assert rj["band_cut"] == [1, 2] and any("ucięty" in n for n in rj["notes"])
    # a band inside the margin: nothing cut, the mask is exactly the band
    cl2, _ = _cleaner({"grow_px": 6, "soft_px": 2, "gen_margin_px": 24})
    m2 = cl2.mask(1)
    assert np.array_equal(m2, cu.mask_fn(cl2.s, [lambda f: obj], [])(1))
    rj2 = cl2.run([1], tmp_path / "v002", "png16")
    assert "band_cut" not in rj2
    # without the model's mask (Cleanup 3D, older callers): the band as it always was
    cl3, _ = _cleaner({"grow_px": 40, "soft_px": 10, "gen_margin_px": 10}, gen=False)
    assert np.array_equal(cl3.mask(1), cu.mask_fn(cl3.s, [lambda f: obj], [])(1))


def test_band_limit_and_engine_reach():
    s = cu.settings({"model": "h3", "grow_px": 20, "soft_px": 10, "gen_margin_px": 24})
    r = cu.engine_reach_px("h3", s, 1920, 1080)
    assert abs(r - h3_fill.MASK_GROW_PX * 1920 / h3_fill.canvas(1920, 1080)[0]) < 1e-6
    lim = cu.band_limit(s, 1920, 1080)
    assert lim["limit_px"] == round(24 + r, 1) and lim["cut"] is False
    assert cu.band_limit({**s, "grow_px": 60}, 1920, 1080)["cut"] is True
    assert cu.engine_reach_px("lama", s, 1920, 1080) == 0.0
    f = h3_fill.H3Fill({1: np.zeros((768, 1344, 3), np.uint8)}, 1920, 1080, False)
    assert abs(f.drawn_px - r) < 1e-6 and f.tone_gap_px > f.drawn_px


def test_run_and_anchors_give_the_model_the_margin_mask():
    from api import cleanup as AC
    src = inspect.getsource(AC)
    run_src = src[src.index("async def cleanup_run"):src.index("# what api/cleanup3d.py shares")]
    assert "rmask = cu.gen_mask_fn(s, rm, keep)" in run_src and "gen_mask=gen_mask" in run_src
    assert "cu.mask_fn(" not in run_src
    ctx = src[src.index("def _anchor_ctx"):src.index("def _anchors_root")]
    assert "cu.gen_mask_fn(s, rm, keep)" in ctx and "cu.mask_fn(" not in ctx


@pytest.fixture
def c1_app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(tmp_path / "b" / "f_####.png")}})
    monkeypatch.setattr(A, "_plate_size", lambda: (W, H))
    monkeypatch.setattr(A, "_work_frames", lambda: list(range(1, 6)))

    class Objs:
        def list(self, groups=True):
            return [{"name": "obj"}]

        def dir(self, n):
            return tmp_path / n
    import engine.cleanup_masks as CM
    import engine.export as EX
    monkeypatch.setattr(A, "_objects", lambda: Objs())
    monkeypatch.setattr(CM, "resolve", lambda store, o, src: {"run": "r1", "source": "core", "valid": True, "by": "own", "why": ""})
    obj = _obj()
    monkeypatch.setattr(EX, "run_alpha_loader", lambda d, run, w, h: (lambda f: obj))
    cu.save_settings(work, {"remove": ["obj"], "model": "h3", "grow_px": 60, "soft_px": 10, "gen_margin_px": 8})
    return TestClient(A.app), work, obj


def test_mask_preview_and_lines_show_the_limit(c1_app):
    c, work, obj = c1_app
    L = c.get("/api/cleanup/band/2/lines").json()
    assert L["cut"] is True and L["limit_contours"]
    png = c.get("/api/cleanup/mask/2").content
    m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[..., 2]   # R = the patch (G = model area)
    s = cu.load_settings(work)
    lim = s["gen_margin_px"] + cu.engine_reach_px("h3", s, W, H)
    d = _dist_from(obj)
    sc = m.shape[1] / W
    big = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if sc != 1 else m
    assert (big[d > lim + 3] == 0).all()                    # the preview is the cut composite, not the 70 px band
    gch = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[..., 1]   # G = what the model draws (2026-11-15)
    gbig = cv2.resize(gch, (W, H), interpolation=cv2.INTER_NEAREST) if sc != 1 else gch
    assert (gbig[d <= lim - 3] > 0).all() and (gbig[d > lim + 3] == 0).all()
    assert c.put("/api/cleanup", json={"gen_margin_px": 120}).json()["settings"]["gen_margin_px"] == 120
    L2 = c.get("/api/cleanup/band/2/lines").json()
    assert L2["cut"] is False


def test_gen_holdout_default_and_old_versions_2026_11_15():
    from engine import inpaint_recompose as IR
    assert cu.settings({})["gen_holdout"] is False
    assert IR.recompose_settings({}, {"model": "h3"})["gen_holdout"] is True        # a version saved before: the old rule
    assert IR.recompose_settings({}, {"model": "h3", "gen_holdout": False})["gen_holdout"] is False
