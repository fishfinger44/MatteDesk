# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D horizon match (package P15, engine/cleanup3d/horizon.py + the hook in render.KeyFill) on a synthetic shot:
a bright sky over a dark sea plane (haze toward flat colours at the horizon, so the horizon is a clean edge), the solved
camera of tests/test_cleanup3d_render.py, keys rendered with that camera, plates rendered with the camera pitched by a
deliberate error – the projected horizon then sits a few px off the plate's. The correction must remove the step and
leave pixels far from the horizon unchanged."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import curves
from engine.cleanup3d import horizon as HZ
from engine.cleanup3d import keys as KS
from engine.cleanup3d import render as RD
from engine.cleanup3d import store
from engine.cleanup3d.geom import CardGeometry
from engine.cleanup3d.types import Camera3D, Card

W, H = 320, 200
KM = np.array([[260.0, 0, 160.0], [0, 260.0, 100.0], [0, 0, 1]])
FRAMES = list(range(1, 8))
KEY_FRAMES = [1, 7]
SEA_Y = 2.0
PITCH_ERR = 1.0                      # degrees: the plates' camera is pitched by this against the solved one
SHIFT = KM[1, 1] * np.tan(np.radians(PITCH_ERR))      # ~4.5 px: how far the plate's horizon lies from the projection's
HS = {"horizon_match": True, "horizon_search_px": 20.0, "horizon_fade_px": 30.0}
RED = np.array([0.1, 0.1, 0.9], np.float32)


def _rot(yaw, pitch, roll=0.0):
    a, b, c = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def make_camera(err_deg=0.0):
    R, t = {}, {}
    for i, f in enumerate(FRAMES):
        C = np.array([0.25 * i, -0.02 * i, 0.30 * i])
        Rf = _rot(0, err_deg) @ _rot(1.5 * i - 6.0, 9.0 - 0.3 * i, 0.3 * i)
        R[f], t[f] = Rf, -Rf @ C
    return Camera3D(W, H, KM.copy(), np.zeros(4), R, t, "synthetic")


def make_cards():
    return [Card("sea", "morze", "plane", {"n": [0, -1, 0], "d": SEA_Y}, order=0, seam_px=4.0),
            Card("sky", "niebo", "sky", order=1, seam_px=4.0)]


def scene_bg(cam, f):
    """Clean background of frame f (BGR float32): sea dark, sky bright, both flat at the horizon."""
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    r = np.stack([(xx - KM[0, 2]) / KM[0, 0], (yy - KM[1, 2]) / KM[1, 1], np.ones_like(xx)], -1)
    wd = r @ cam.R[f]
    C = cam.centre(f)
    hit = wd[..., 1] > 1e-6
    zp = (SEA_Y - C[1]) / np.where(hit, wd[..., 1], 1.0)
    P = C + zp[..., None] * wd
    X, Z = P[..., 0], P[..., 2]
    tex = 0.06 * np.sin(X * 3.1) * np.cos(Z * 2.3)
    fog = np.exp(-np.maximum(Z, 0.0) / 4.0)[..., None]
    sea = np.array([0.30, 0.26, 0.20]) + (tex[..., None] + np.array([0.05, 0.08, 0.02])) * fog
    el = np.arctan2(-wd[..., 1], np.hypot(wd[..., 0], wd[..., 2]))
    az = np.arctan2(wd[..., 0], wd[..., 2])
    up = (1.0 - np.exp(-np.maximum(el, 0.0) / 0.15))[..., None]
    sky = np.array([0.80, 0.78, 0.74]) + (0.05 * np.sin(az * 9.0) * np.cos(el * 7.0))[..., None] * up
    return np.where(hit[..., None], sea, sky).astype(np.float32)


def horizon_row(cam, f, x):
    """Analytic horizon (vanishing line of the sea) in frame f at column x."""
    R = cam.R[f]
    xn = (x - KM[0, 2]) / KM[0, 0]
    yn = -(R[0, 1] * xn + R[2, 1]) / R[1, 1]
    return KM[1, 2] + KM[1, 1] * yn


def obj_alpha(f):
    yy, xx = np.mgrid[0:H, 0:W]
    cx = 90.0 + 18.0 * (f - 1)
    cy = horizon_row(make_camera(PITCH_ERR), f, cx)
    return ((((xx - cx) / 14.0) ** 2 + ((yy - cy) / 40.0) ** 2) < 1.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------- units
def test_settings_defaults_and_limits():
    s = HZ.settings(None)
    assert s == {"horizon_match": False, "horizon_fade_px": 120.0, "horizon_search_px": 60.0}
    s = HZ.settings({"horizon_match": 1, "horizon_fade_px": 1.0, "horizon_search_px": 9999, "other": 3})
    assert s == {"horizon_match": True, "horizon_fade_px": 8.0, "horizon_search_px": 400.0}
    rs = RD.render_settings({"horizon_match": True, "horizon_fade_px": 50})
    assert rs["horizon_match"] is True and rs["horizon_fade_px"] == 50.0


def test_store_keeps_horizon_settings(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    s = store.load_render_settings(work)
    assert s["horizon_match"] is False and s["horizon_fade_px"] == 120.0
    s = store.save_render_settings(work, {"horizon_match": True, "horizon_fade_px": 5000})
    assert s["horizon_match"] is True and s["horizon_fade_px"] == 800.0
    assert store.load_render_settings(work)["horizon_match"] is True


def test_pick_takes_the_lowest_strong_edge():
    prof = np.zeros(40)
    prof[10], prof[9], prof[11] = 10.0, 6.0, 6.0
    prof[25], prof[24], prof[26] = 8.0, 5.0, 5.0          # the sea top below the sun's band: lower, slightly weaker
    j, s = HZ.pick(prof)
    assert abs(j - 25) < 0.01 and s > 5


def test_predicted_horizon_is_the_vanishing_line():
    cam = make_camera()
    geo = CardGeometry(cam, make_cards())
    hz = HZ.Horizon(geo, lines={}, s=HS)
    for f in (1, 4, 7):
        pred = hz.predicted(f)
        ok = np.isfinite(pred)
        assert ok.mean() > 0.95
        want = horizon_row(cam, f, hz.grid_x[ok])
        assert np.abs(pred[ok] - want).max() < 0.75


def test_warp_moves_the_line_and_leaves_far_rows():
    img = np.zeros((100, 40, 3), np.float32)
    img[:50] = 1.0                                          # edge between rows 49 / 50 (line at 49.5)
    out = HZ.warp_rows(img, np.full(40, 3.0), np.full(40, 46.5), 20.0)
    col = out[:, 20, 0]
    assert col[46] > 0.5 and col[47] < 0.5                  # the edge now at 46.5
    assert np.array_equal(out[:26], img[:26]) and np.array_equal(out[67:], img[67:])


def _frame_pair(f):
    solved, shot = make_camera(), make_camera(PITCH_ERR)
    fill = scene_bg(solved, f)                              # what the keys projected through the solved camera show
    clean = scene_bg(shot, f)                               # the plate's true background
    a = obj_alpha(f)
    plate = np.where(a[..., None] > 0, RED, clean).astype(np.float32)
    return solved, shot, fill, clean, plate, a


def test_match_removes_the_step_and_keeps_far_pixels():
    f = 4
    solved, shot, fill, clean, plate, a = _frame_pair(f)
    hz = HZ.Horizon(CardGeometry(solved, make_cards()), lines={}, s=HS)
    wgt = np.ones((H, W), np.float32)
    out, ow, info = hz.match(f, plate, fill, wgt, a)
    assert info["status"] == "ok" and info["source"] == "auto" and info["warped"]
    assert abs(abs(info["offset_px"]) - SHIFT) < 1.0, info
    assert abs(info["step_px"] - SHIFT) < 1.0 and info["step_after_px"] < 0.6, info
    # far from the horizon (beyond the fade) the fill is untouched, bit for bit
    xs = np.arange(W)
    hrow = horizon_row(shot, f, xs)
    far = np.abs(np.arange(H)[:, None] - hrow[None, :]) > HS["horizon_fade_px"] + 1
    assert np.array_equal(out[far], fill[far]) and np.array_equal(ow[far], wgt[far])
    # at the horizon the corrected fill matches the plate's background much better
    near = np.abs(np.arange(H)[:, None] - hrow[None, :]) < 3
    before = np.abs(fill - clean).max(axis=2)[near].mean()
    after = np.abs(out - clean).max(axis=2)[near].mean()
    assert after < 0.25 * before, (before, after)


def test_match_in_a_crop_and_on_a_preview_grid_report_frame_px():
    f = 3
    solved, shot, fill, clean, plate, a = _frame_pair(f)
    hz = HZ.Horizon(CardGeometry(solved, make_cards()), lines={}, s=HS)
    x0, y0 = 40, 20                                         # a ROI crop: frame px = (x0, y0) + image px
    _o, _w, info = hz.match(f, plate[y0:, x0:], fill[y0:, x0:], np.ones((H - y0, W - x0), np.float32), a[y0:, x0:],
                            (x0, y0, 1.0))
    assert abs(abs(info["offset_px"]) - SHIFT) < 1.0, info
    step = 2                                                # a preview grid: every 2nd px, first sample at 1
    g = (slice(1, None, step), slice(1, None, step))
    _o, _w, info = hz.match(f, plate[g], fill[g], np.ones(a[g].shape, np.float32), a[g], (1.0, 1.0, 2.0), lines=True)
    assert info["status"] == "ok" and abs(abs(info["offset_px"]) - SHIFT) < 1.5, info
    pl = np.array(info["plate"])
    assert np.median(np.abs(pl[:, 1] - horizon_row(shot, f, pl[:, 0]))) < 1.0


def test_no_correction_without_offset_and_without_horizon():
    f = 4
    solved = make_camera()
    fill = scene_bg(solved, f)
    a = obj_alpha(f)
    plate = np.where(a[..., None] > 0, RED, fill).astype(np.float32)
    hz = HZ.Horizon(CardGeometry(solved, make_cards()), lines={}, s=HS)
    out, _w, info = hz.match(f, plate, fill, np.ones((H, W), np.float32), a)
    assert info["status"] == "ok" and not info["warped"] and abs(info["offset_px"]) < 0.25
    assert out is fill
    sky_only = HZ.Horizon(CardGeometry(solved, [make_cards()[1]]), lines={}, s=HS)
    _o, _w, info = sky_only.match(f, plate, fill, np.ones((H, W), np.float32), a)
    assert info["status"] == "no_horizon"


def _line(cam, f, dy=0.0):
    xs = np.linspace(0, W - 1, 6)
    return curves.from_points([[float(x), float(horizon_row(cam, f, x) + dy)] for x in xs])


def test_user_lines_saved_validated_and_interpolated(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    solved = make_camera()
    assert HZ.load_lines(work) == {}
    HZ.save_line(work, 2, _line(solved, 2, dy=2.0))
    lines = HZ.save_line(work, 6, _line(solved, 6, dy=6.0))
    assert sorted(lines) == [2, 6]
    d = json.loads(HZ.lines_file(work).read_text(encoding="utf-8"))
    assert d["schema"] == HZ.SCHEMA and set(d["lines"]) == {"2", "6"}
    with pytest.raises(ValueError):
        HZ.save_line(work, 3, curves.from_points([[0, 0], [10, 0], [5, 5]], closed=True))
    with pytest.raises(ValueError):
        HZ.save_line(work, 3, {"type": "bezier", "closed": False, "pts": [curves.point([1, 1])]})
    hz = HZ.Horizon(CardGeometry(solved, make_cards()), work=work, s=HS)
    pred4 = hz.predicted(4)
    ok = np.isfinite(pred4) & (hz.grid_x > 8) & (hz.grid_x < W - 8)
    u4 = hz.user_rows(4)
    assert np.abs((u4 - pred4)[ok] - 4.0).max() < 0.6     # halfway between +2 and +6
    u2 = hz.user_rows(2)
    assert np.abs(u2[ok] - (horizon_row(solved, 2, hz.grid_x[ok]) + 2.0)).max() < 0.6
    assert hz.user_rows(7) is None                          # outside the drawn frames: the automatic detector
    near = hz.user_rows(7, near=True)
    assert np.abs((near - hz.predicted(7))[ok] - 6.0).max() < 0.6
    lines = HZ.save_line(work, 6, None)
    assert sorted(lines) == [2]


def test_user_line_wins_over_the_detector():
    f = 4
    solved, shot, fill, clean, plate, a = _frame_pair(f)
    # the user says the horizon is where the projection has it (no correction should follow)
    hz = HZ.Horizon(CardGeometry(solved, make_cards()), lines={f: _line(solved, f)}, s=HS)
    out, _w, info = hz.match(f, plate, fill, np.ones((H, W), np.float32), a)
    assert info["source"] == "user" and abs(info["offset_px"]) < 0.5 and not info["warped"]


# ----------------------------------------------------------------------------------------------- the render
class Shot:
    def __init__(self, tmp_path):
        self.work = Path(tmp_path) / "shot" / "matte_desk"
        self.work.mkdir(parents=True)
        self.solved, self.true = make_camera(), make_camera(PITCH_ERR)
        store.save_camera(self.work, self.solved)
        store.save_cards(self.work, make_cards())
        for f in KEY_FRAMES:
            key = scene_bg(self.solved, f)
            for c in ("sea", "sky"):
                KS.add_version(self.work, c, f, key, "test", "exr")

    def plate(self, f):
        bg = scene_bg(self.solved if f in KEY_FRAMES else self.true, f)
        return np.where(obj_alpha(f)[..., None] > 0, RED, bg).astype(np.float32)

    def render(self, frames, **s):
        st = {"key_soft_px": 8.0, "key_grow_px": 0.0, "edge_px": 16.0, "edge_hard_px": 4.0, "key_effects": False,
              "tone": False, "remove": ["obj"], "grow_px": 3.0, "soft_px": 2.0, "motion_blur": "off", **s}
        rj = RD.render(self.work, frames, plate=self.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st)
        out = Path(rj["dir"])
        comp = {f: cv2.imread(str(out / "comp" / f"c_{f:04d}.png")).astype(np.float32) / 255.0 for f in frames}
        return json.loads((out / "run.json").read_text(encoding="utf-8")), comp


def test_render_corrects_the_horizon_and_reports_the_step(tmp_path):
    sh = Shot(tmp_path)
    frames = [3, 4]
    rj_off, comp_off = sh.render(frames)
    rj_on, comp_on = sh.render(frames, **HS)
    assert rj_on["c3d"]["settings"]["horizon_match"] is True
    for f in frames:
        hzr = rj_on["c3d"]["frames"][str(f)]["horizon"]
        assert hzr["status"] == "ok" and hzr["warped"], hzr
        assert hzr["step_px"] > 2.5 and hzr["step_after_px"] < 0.75, hzr
        assert "horizon" not in rj_off["c3d"]["frames"][str(f)]
        hrow = horizon_row(sh.true, f, np.arange(W))
        dist = np.abs(np.arange(H)[:, None] - hrow[None, :])
        # far from the horizon the result is the same with and without the match
        far = dist > HS["horizon_fade_px"] + 2
        assert np.abs(comp_on[f] - comp_off[f]).max(axis=2)[far].max() <= 1.0 / 255 + 1e-6
        # in the patch at the horizon the comp follows the plate's background
        clean = scene_bg(sh.true, f)
        sel = (dist < 3) & (obj_alpha(f) > 0)
        e_off = np.abs(comp_off[f] - clean).max(axis=2)[sel].mean()
        e_on = np.abs(comp_on[f] - clean).max(axis=2)[sel].mean()
        assert e_on < 0.4 * e_off, (f, e_off, e_on)


def test_blend_preview_applies_the_match(tmp_path):
    sh = Shot(tmp_path)
    sc = RD.Scene(sh.work)
    try:
        f = 4
        a = obj_alpha(f)
        fill0, _m, _s = RD.blend_inputs(sc, f, sh.plate(f), a, max_side=W)
        fill1, _m, _s = RD.blend_inputs(sc, f, sh.plate(f), a, max_side=W, settings=HS)
        diff = np.abs(fill1.astype(int) - fill0.astype(int)).max(axis=2)
        hrow = horizon_row(sh.true, f, np.arange(W))
        dist = np.abs(np.arange(H)[:, None] - hrow[None, :])
        assert diff[dist < 3].mean() > 5 and diff[dist > HS["horizon_fade_px"] + 2].max() == 0
        info = HZ.preview(sc, f, sh.plate(f), a, HS, max_side=W)
        assert info["status"] == "ok" and info["drawn"] == [] and info["plate"] and info["predicted"]
        assert abs(abs(info["offset_px"]) - SHIFT) < 1.0
    finally:
        sc.close()


# ----------------------------------------------------------------------------------------------- the routes
def test_horizon_routes(tmp_path, monkeypatch):
    from tests.test_cleanup3d_api import Env
    from tests.test_cleanup3d_render import make_cards as render_cards
    env = Env(tmp_path, monkeypatch)
    store.save_cards(env.work, render_cards())
    store.save_camera(env.work, env.cam)
    for f in (1, 5, 9):
        img = cv2.imread(str(env.shot / "plates" / f"p_{f:04d}.png")).astype(np.float32) / 255.0
        for c in ("ground", "sky"):
            KS.add_version(env.work, c, f, img, "test", "exr")
    c = env.client
    r = c.get("/api/cleanup3d/horizon")
    assert r.status_code == 200 and r.json()["drawn"] == [] and r.json()["settings"]["horizon_match"] is False
    r = c.get("/api/cleanup3d/horizon/3")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["frame"] == 3 and d["status"] in ("ok", "no_plate_line", "no_fill_line") and d["predicted"]
    line = curves.from_points([[0.0, 60.0], [160.0, 58.0], [319.0, 57.0]])
    r = c.put("/api/cleanup3d/horizon/3", json={"curve": line})
    assert r.status_code == 200 and r.json()["drawn"] == [3]
    d = c.get("/api/cleanup3d/horizon/3").json()
    assert d["source"] == "user" and d["user"]["pts"] and d["drawn"] == [3]
    assert c.put("/api/cleanup3d/horizon/4", json={"curve": {**line, "closed": True}}).status_code == 409
    r = c.put("/api/cleanup3d/render_settings", json={"horizon_match": True, "horizon_fade_px": 40})
    assert r.json()["render_settings"]["horizon_match"] is True and r.json()["render_settings"]["horizon_fade_px"] == 40.0
    assert c.get("/api/cleanup3d").json()["render_settings"]["horizon_match"] is True
    assert c.get("/api/cleanup3d/blend/3?part=fill").status_code == 200
    assert c.delete("/api/cleanup3d/horizon/3").json()["drawn"] == []
    assert c.get("/api/cleanup3d/horizon").json()["lines"] == {}


# ----------------------------------------------------------------------------------------------- the browser
def test_horizon_section_in_jsdom():
    """C5 „Horyzont”: settings saved, the line drawn with THE curve editor, saved / removed, status text (jsdom)."""
    from tests.test_cleanup3d_js import _run
    _run("cleanup3d_horizon_harness.mjs", "OK cleanup3d horizon harness")


def test_horizon_ui_reuses_the_curve_editor_and_is_translated():
    root = Path(__file__).resolve().parents[1]
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert html.index('/static/curves.js"') < html.index('/static/cleanup3d_horizon.js"')
    js = (root / "static" / "cleanup3d_horizon.js").read_text(encoding="utf-8")
    assert "new (CV().Editor)(" in js and "CV().fromPoints(" in js
    assert "bezierCurveTo" not in js and "function split(" not in js     # no own curve code (static/curves.js)
    for url in ("/api/cleanup3d/horizon/${f}", "/api/cleanup3d/render_settings"):
        assert url in js
    en = json.loads((root / "static" / "i18n" / "en.json").read_text(encoding="utf-8"))["strings"]
    c3 = (root / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    for s in ("Horyzont", "Dopasuj horyzont", "Zasięg dopasowania (px)", "✎ Narysuj horyzont w tej klatce"):
        assert s in c3 and en.get(s), s
