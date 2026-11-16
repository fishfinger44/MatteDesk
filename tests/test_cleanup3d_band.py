# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""P14 „Pas Czyszczenia” (docs/PLAN_CLEANUP_3D.md): the band of the Cleanup 3D key patch is the app's ONE band system
(engine/objband.cleanup_band + engine/bandsys + engine/bandmap; editor static/bandpoints.js owner „c3d”), outer side
only, and the key patch grows the object by `key_grow_px` (user report sh060 v005 F11: a thin outline of the removed
woman stayed). Synthetic: a point outward widens the patch there and only there; without points the patch is the plain
ramp; the browser's rule (static/cleanup3d.js patchAlpha) agrees with engine/cleanup3d/band.patch_weight; the render
uses the band; a soft fringe ring disappears with key_grow_px 16 and stays with 0; the band endpoints edit the points
with engine/bandedit."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.cleanup3d import band as C3B
from engine.objband import CLEANUP_KEY, cleanup_band

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
H, W = 200, 260


def _disk(cx=130, cy=100, r=30):
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r * r


# ------------------------------------------------------------------------------------------------ the rule
def test_cleanup_band_is_an_instance_of_the_one_model():
    b = cleanup_band({"key_soft_px": 24.0}, {"band_points": [{"id": "p1", "keys": {"3": {"x": 1, "y": 2, "out": 5, "in": 0}}}],
                                             "band_ref_out": 12.0, "band_ref_in": 0.0})
    assert (b.out_px, b.in_px, b.inset_feather_px, b.outer_feather_px) == (24.0, 0.0, 0.0, 24.0)
    assert b.key == CLEANUP_KEY and b.owner_id == CLEANUP_KEY and not b.shared and b.active
    assert b.scales == (2.0, 1.0)                    # the slider scales the points' widths (objband.scales)
    assert cleanup_band({"key_soft_px": 10.0}).band_points == []


def test_without_points_the_patch_is_the_plain_smoothstep_ramp():
    hole = _disk()
    patch, area = C3B.patch_weight(hole, hole, 20.0)
    d = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(1.0 - d / 20.0, 0, 1)
    old = np.where(hole, 1.0, t * t * (3 - 2 * t))      # the patch before P14 (KeyFill._patch 2026-11-10)
    assert (patch[hole] == 1.0).all() and patch[d > 21].max() == 0.0
    assert np.abs(patch - old).max() < 0.12 and np.abs(patch - old).mean() < 0.01
    assert area.sum() > 0 and not (area & hole).any()


def test_a_point_outward_widens_the_patch_there_and_only_there():
    hole = _disk()
    soft = 12.0
    plain, _ = C3B.patch_weight(hole, hole, soft)
    # right side pushed out to 50 px, left side kept at the slider's 12 px
    pts = [(160.0, 100.0, 50.0, 0.0), (100.0, 100.0, 12.0, 0.0)]
    pushed, area = C3B.patch_weight(hole, hole, soft, pts=pts)
    right = (slice(95, 106), slice(175, 190))            # 15–30 px right of the disk: beyond the plain ramp
    left_far = (slice(90, 111), slice(70, 86))           # 15-30 px left of the disk: still no patch
    left_ramp = (slice(98, 103), slice(88, 100))
    assert plain[right].max() == 0.0 and pushed[right].min() > 0.99      # full patch there (feather = the last 12 px)
    assert pushed[left_far].max() == 0.0                                  # the left side not widened
    assert np.abs(pushed[left_ramp] - plain[left_ramp]).max() < 0.05
    assert pushed[100, 207] > 0 and pushed[100, 212] == 0.0             # ends ~50 px out
    L = C3B.lines(hole, hole, soft, pts=pts)
    assert L["band_contours"] and L["feather_contours"] and L["inset_feather_contours"] == []


def test_the_slider_scales_the_points():
    hole = _disk()
    p = {"band_points": [{"id": "p1", "keys": {"0": {"x": 160.0, "y": 100.0, "out": 40.0, "in": 0}}},
                         {"id": "p2", "keys": {"0": {"x": 100.0, "y": 100.0, "out": 10.0, "in": 0}}}],
         "band_ref_out": 10.0, "band_ref_in": 0.0}
    core_at = lambda k: hole
    for soft, reach in ((10.0, 40), (20.0, 80)):
        b = cleanup_band({"key_soft_px": soft}, p)
        pts = C3B.points_fn(b, core_at)(0)
        assert [round(q[2]) for q in pts] == [reach, round(soft)]


def _js_patch_alpha(hole, out_w, fo, tmp):
    src = (ROOT / "static" / "cleanup3d.js").read_text(encoding="utf-8")

    def fn(name):
        i = src.index(f"function {name}(")
        j = src.index("{", i)
        depth = 0
        for k in range(j, len(src)):
            depth += {"{": 1, "}": -1}.get(src[k], 0)
            if depth == 0:
                return src[i:k + 1]
        raise AssertionError(name)
    h, w = hole.shape
    code = (fn("distanceFrom") + "\n" + fn("patchAlpha") + "\n"
            + f"const hole = Uint8Array.from({json.dumps(hole.astype(int).ravel().tolist())});"
            + f"const ow = Float32Array.from({json.dumps(np.round(out_w, 3).ravel().tolist())});"
            + f"process.stdout.write(JSON.stringify(Array.from(patchAlpha(hole, ow, {fo}, {w}, {h}))));")
    f = Path(tmp) / "patch_alpha.js"
    f.write_text(code, encoding="utf-8")
    r = subprocess.run([NODE, str(f)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return np.array(json.loads(r.stdout), np.float32).reshape(h, w)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_browser_rule_agrees_with_the_render_rule(tmp_path):
    """static/cleanup3d.js patchAlpha (chamfer distances) = engine/cleanup3d/band.patch_weight (cv2 L2) on the same hole
    and the same band widths (the map the server sends with part=band)."""
    hole = _disk(r=26)
    hole[100:112, 150:185] = True                         # a trace attached to the object (the hole ≠ the core)
    core = _disk(r=26)
    pts = [(156.0, 100.0, 45.0, 0.0), (104.0, 100.0, 10.0, 0.0)]
    out_w = np.asarray(C3B.out_widths(core, 14.0, pts), np.float32)
    py, _ = C3B.patch_weight(hole, core, 14.0, pts=pts)
    js = _js_patch_alpha(hole, out_w, 14.0, tmp_path)
    assert (js[hole] == 1).all()
    assert np.abs(js - py).mean() < 0.01 and np.abs(js - py).max() < 0.35        # chamfer 3-4 vs L2: single edge pixels
    a, b = js > 0.5, py > 0.5
    assert (a & b).sum() / float((a | b).sum()) > 0.97


def test_preview_hole_follows_the_browser_rule():
    maps = np.zeros((40, 60, 3), np.uint8)
    maps[10:20, 10:20, 1] = 255                            # G object
    maps[..., 2] = 255                                     # R full coverage everywhere
    maps[12:16, 20:30, 0] = 200                            # B difference touching the object = a trace
    maps[30:35, 40:50, 0] = 200                            # a detached difference = not a trace
    h = C3B.preview_hole(maps, True, 0.2, 0.5)
    assert h[14, 25] and not h[32, 45] and h[15, 15]
    assert not C3B.preview_hole(maps, False, 0.2, 0.5)[14, 25]


# ------------------------------------------------------------------------------------------------ the render
def _shot(tmp_path, fringe=False, shadow=False):
    from tests.test_cleanup3d_render import Shot, obj_alpha

    class S(Shot):
        def plate(self, f):
            p = super().plate(f)
            a = obj_alpha(f) > 0
            if fringe:   # a soft fringe ring of 10 px around the object at 50 % darkness (motion blur / hair)
                ring = cv2.dilate(a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
                p = np.where((ring & ~a)[..., None], p * 0.5, p)
            if shadow:   # a dark patch 22–40 px right of the object, not touching it (missed by the traces)
                cx = 70.0 + 22.0 * (f - 1)
                yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]]
                sh = (xx >= cx + 16 + 22) & (xx < cx + 16 + 40) & (yy >= 85) & (yy < 105)
                p = np.where(sh[..., None], p * 0.4, p)
            return p.astype(np.float32)
    return S(tmp_path)


def test_key_grow_removes_a_soft_fringe(tmp_path):
    from tests.test_cleanup3d_render import obj_alpha
    shot = _shot(tmp_path, fringe=True)
    a = obj_alpha(5) > 0
    ring = (cv2.dilate(a.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0) & ~a
    errs = {}
    for g in (0.0, 16.0):
        _rj, comp, _src, _out = shot.render(frames=[5], key_grow_px=g, key_effects=False)
        errs[g] = float(np.abs(comp[5] - shot.bg[5]).max(axis=2)[ring].max())
    assert errs[0.0] > 0.08, errs                       # the outline of the removed object stays
    assert errs[16.0] < 0.03, errs                      # grown: the keys (clean background) replace it


def test_render_band_point_takes_a_missed_shadow_on_its_side_only(tmp_path):
    shot = _shot(tmp_path, shadow=True)
    cx = 70.0 + 22.0 * 4
    yy, xx = np.mgrid[0:200, 0:320]
    sh = (xx >= cx + 16 + 22) & (xx < cx + 16 + 40) & (yy >= 85) & (yy < 105)
    left = (xx >= cx - 16 - 40) & (xx < cx - 16 - 22) & (yy >= 85) & (yy < 105)
    band = {"band_points": [{"id": "p1", "keys": {"5": {"x": cx + 16, "y": 95.0, "out": 50.0, "in": 0}}},
                            {"id": "p2", "keys": {"5": {"x": cx - 16, "y": 95.0, "out": 12.0, "in": 0}}}],
            "band_ref_out": 12.0, "band_ref_in": 0.0}
    _rj, plain, _s, _o = shot.render(frames=[5], key_effects=False)
    _rj, pushed, _s, _o = shot.render(frames=[5], key_effects=False, patch_band=band)
    err = lambda c, m: float(np.abs(c[5] - shot.bg[5]).max(axis=2)[m].max())
    assert err(plain, sh) > 0.1                          # the shadow stays without the point
    assert err(pushed, sh) < 0.03                        # the point pushes the patch over it
    assert np.abs(pushed[5][left] - plain[5][left]).max() < 1e-6     # nothing changes on the other side


# ------------------------------------------------------------------------------------------------ the endpoints
def test_band_store_roundtrip(tmp_path):
    work = tmp_path / "w" / "matte_desk"
    work.mkdir(parents=True)
    assert C3B.load(work) == {"band_points": [], "band_ref_out": None, "band_ref_in": None}
    pts = [{"id": "p1", "keys": {"4": {"x": 1.0, "y": 2.0, "out": 9.0, "in": 0.0}}}]
    C3B.save(work, {"band_points": pts, "band_ref_out": 12.0, "band_ref_in": 0.0})
    assert C3B.load(work) == {"band_points": pts, "band_ref_out": 12.0, "band_ref_in": 0.0}
    C3B.save(work, {"band_points": [], "band_ref_out": 12.0})
    assert C3B.load(work)["band_ref_out"] is None        # no points = the plain ramp, no reference


def test_one_editor_one_edit_implementation():
    """The Cleanup band is an owner of the one editor (static/bandpoints.js) and its routes use engine/bandedit – no
    second band point editor / edit code (CLAUDE.md „Pas krawędzi = jeden system”)."""
    bp = (ROOT / "static" / "bandpoints.js").read_text(encoding="utf-8")
    c3 = (ROOT / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    api = (ROOT / "api" / "cleanup3d.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "'c3d'" in bp and "/api/cleanup3d/band" in bp
    assert "MD_BANDPTS.drawLines" in c3 or "B.drawLines" in c3
    assert "function strokeLines" not in c3 and "setLineDash" not in c3.split("function patchAlpha")[1].split("function hookViewer")[0]
    for src in (api, app):
        assert "from engine.bandedit import" in src
    assert "MAX_WIDTH_PX" not in api and not re.search(r"def _band_put[\s\S]*?raw_o = lambda", app)


def test_band_endpoints_edit_with_the_one_editor_api(tmp_path, monkeypatch):
    """/api/cleanup3d/band/… = the stage_band contract of static/bandpoints.js: params / state / put / delete keys /
    delete point, Ctrl+Z through PUT params; the blend preview sends the band map (part=band) and the lines."""
    from engine.cleanup3d import keys as KS
    from engine.cleanup3d import store
    from tests.test_cleanup3d_api import Env
    from tests.test_cleanup3d_render import KEY_FRAMES, make_cards, true_bg
    env = Env(tmp_path, monkeypatch)
    c = env.client
    store.save_camera(env.work, env.cam)
    store.save_cards(env.work, make_cards())
    for f in KEY_FRAMES:
        for card in ("ground", "sky"):
            KS.add_version(env.work, card, f, true_bg(env.cam, f), "test", "exr")
    c.put("/api/cleanup3d/render_settings", json={"key_soft_px": 10.0, "key_grow_px": 0.0})
    assert c.get("/api/cleanup3d/band/params").json()["band_points"] == []
    assert c.get("/api/cleanup3d/band/5").json() == {"frame": 5, "points": [], "defaults": {"out": 10, "in": 0}}
    cx = 70.0 + 22.0 * 4
    r = c.put("/api/cleanup3d/band/5", json={"x": cx + 20, "y": 95.0, "out": 40})
    assert r.status_code == 200, r.text
    st = r.json()["band"]["points"][0]
    assert st["key"] and st["keys"] == [5] and st["out"] == 40.0 and abs(st["sx"] - (cx + 16 + 4)) < 3   # snapped (grow 4)
    p = c.get("/api/cleanup3d/band/params").json()
    assert p["band_ref_out"] == 10.0 and p["band_points"][0]["keys"]["5"]["out"] == 40.0
    # the slider scales the shown widths
    c.put("/api/cleanup3d/render_settings", json={"key_soft_px": 20.0})
    assert c.get("/api/cleanup3d/band/5").json()["points"][0]["out"] == 80.0
    c.put("/api/cleanup3d/render_settings", json={"key_soft_px": 10.0})
    # a key in another frame, carried by the object's motion in between
    assert c.put("/api/cleanup3d/band/7", json={"id": "p1", "out": 20}).status_code == 200
    s6 = c.get("/api/cleanup3d/band/6").json()["points"][0]
    assert s6["keys"] == [5, 7] and not s6["key"] and abs(s6["out"] - 30.0) < 1e-6 and abs(s6["sx"] - (cx + 22 + 20)) < 4
    # the preview: band map + lines (plate px)
    for part in ("fill", "maps"):
        assert c.get("/api/cleanup3d/blend/5", params={"part": part}).status_code == 200
    rb = c.get("/api/cleanup3d/blend/5", params={"part": "band"})
    assert rb.status_code == 200 and float(rb.headers["X-C3D-Band-Soft"]) == 10.0 and rb.headers["X-C3D-Band-Points"] == "1"
    bm = cv2.imdecode(np.frombuffer(rb.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert bm.max() == 40 and bm.min() == 40              # one point = its width along the whole outline
    L = c.get("/api/cleanup3d/band/5/lines", params={"key_soft_px": 10, "key_effects": 0}).json()
    xs = [q[0] for cnt in L["band_contours"] for q in cnt]
    assert L["band_contours"] and max(xs) > cx + 16 + 4 + 35
    # delete the key of F7, Ctrl+Z restores it, delete the point
    prev = c.get("/api/cleanup3d/band/params").json()
    assert c.delete("/api/cleanup3d/band", params={"frame": 7}).json()["removed"] == ["p1"]
    assert c.get("/api/cleanup3d/band/params").json()["band_points"][0]["keys"].keys() == {"5"}
    assert c.put("/api/cleanup3d/band/params", json=prev).status_code == 200
    assert set(c.get("/api/cleanup3d/band/params").json()["band_points"][0]["keys"]) == {"5", "7"}
    assert c.delete("/api/cleanup3d/band/p9").status_code == 404
    assert c.delete("/api/cleanup3d/band/p1").status_code == 200
    assert c.get("/api/cleanup3d/band/params").json() == {"band_points": [], "band_ref_out": None, "band_ref_in": None}
