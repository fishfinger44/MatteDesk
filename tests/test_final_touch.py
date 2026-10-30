# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage Final Touch (PLAN_FINAL_TOUCH.md step 1, 2026-10-22): hand corrections of the alpha with a bitmap brush over a
chosen base version, stored as stroke paths (engine/final_touch.py), base + corrections as an export source."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


# ------------------------------------------------------------------------------------------------ layer + compose
def test_layer_add_remove_erase_and_compose():
    from engine.final_touch import compose, layer, stroke_mask
    w, h = 120, 80
    hard = {"mode": "add", "radius_px": 10, "hardness": 1.0, "p": [[30, 40], [60, 40]]}
    m = stroke_mask(hard, w, h)
    assert m[40, 45] == 1.0 and m[40, 5] == 0.0 and m[40, 75] == 0.0      # hard: full inside, nothing outside
    soft = {**hard, "hardness": 0.0}
    ms = stroke_mask(soft, w, h)
    assert ms[40, 45] > 0.9 and 0.0 < ms[48, 45] < 0.5 and ms[52, 45] == 0.0   # soft: fades inside the radius only
    base = np.full((h, w), 0.5, np.float32)
    base[:, :10] = 0.2
    # add -> 1, remove -> 0, untouched pixels keep the base exactly (incl. its soft values)
    a = compose(base, layer([hard], w, h))
    assert a[40, 45] == 1.0 and a[10, 5] == pytest.approx(0.2) and a[10, 100] == pytest.approx(0.5)
    rm = {**hard, "mode": "remove"}
    assert compose(base, layer([rm], w, h))[40, 45] == 0.0
    # erase over an add returns to the base; a soft add keeps the soft edge of the brush
    er = {**hard, "mode": "erase", "radius_px": 14}
    assert compose(base, layer([hard, er], w, h))[40, 45] == pytest.approx(0.5)
    half = compose(base, layer([soft], w, h))[48, 45]
    assert 0.5 < half < 1.0
    # strokes with a radius per point (pen pressure) are accepted
    pen = {"mode": "add", "radius_px": 6, "hardness": 1.0, "p": [[20, 20, 2], [40, 20, 9]]}
    mp = stroke_mask(pen, w, h)
    assert mp[20, 40] == 1.0 and mp[26, 40] == 1.0 and mp[24, 20] == 0.0


def test_store_paths_and_signature(tmp_path):
    from engine.final_touch import clean_stroke, corrections_sig, frames_of, load, set_base, set_frame
    od = tmp_path / "obj"
    assert load(od)["frames"] == {} and corrections_sig(load(od)) == ""
    set_frame(od, 7, [{"mode": "add", "radius_px": 5, "p": [[1, 2]]}])
    d = load(od)
    assert frames_of(d) == [7] and d["frames"]["7"][0]["hardness"] == pytest.approx(0.6)   # default hardness
    sig = corrections_sig(d)
    set_frame(od, 9, [{"mode": "remove", "radius_px": 5, "p": [[1, 2]]}])
    assert corrections_sig(load(od)) != sig
    set_frame(od, 9, [])
    assert frames_of(load(od)) == [7] and corrections_sig(load(od)) == sig
    with pytest.raises(ValueError):
        clean_stroke({"mode": "paint", "p": [[1, 1]]})
    with pytest.raises(ValueError):
        set_base(od, "nope")
    set_base(od, "version", "v003")
    assert load(od)["base"] == {"source": "version", "run": "v003"}
    assert (od / "final_touch" / "final_touch.json").exists()


# ------------------------------------------------------------------------------------------------ API + export
@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def other_frame(frames, start):
    return next(f for f in frames if f != start)


def _export_png(api, body):
    r = api.post("/api/export", json={"objects": ["obj"], "format": "png", **body})
    assert r.status_code == 200, r.text
    d = r.json()
    return d, Path(d["outputs"]["export_dir"]) / "mattes" / "obj"


def test_final_touch_api_compose_and_export(proj):
    app_module, api, shot, pipe, frames, start = proj
    od = pipe.store.dir("obj")
    manifest = sorted((pipe.store.root.parent).glob("shot_manifest_*.json"))
    before = [p.read_bytes() for p in manifest]
    st = {s["id"]: s for s in api.get("/api/object/obj/stages").json()["steps"]}
    assert [s for s in st] == ["sam3", "cotracker", "matanyone", "videomama", "vitmatte", "final", "ltx", "final_touch", "export"]
    assert st["final_touch"]["n"] == 8 and st["export"]["n"] == 9 and st["final_touch"]["state"] == "done"
    # default base right after the core: the mask 1+2 (no later stage computed)
    d = api.get("/api/object/obj/final_touch").json()
    assert d["base"]["run"] == "core" and d["base"]["chosen"] is False and d["frames"] == {}
    # stage 3 computed -> the default base is its version
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    ma = r.json()["run"]
    d = api.get("/api/object/obj/final_touch").json()
    assert d["base"] == {"source": "matanyone", "run": ma, "chosen": False, "stale": False, "why": ""}
    # a stage-2 brush layer makes stage 3 stale: the default base STAYS its version, marked stale – it used to jump
    # silently to the next current result (the mask 1+2) and the corrections landed on another alpha (2026-10-30)
    z = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": start, "polygon": [],
                                                     "strokes": [{"p": [[5, 5]], "radius_px": 3}],
                                                     "params": {"source": "brush", "track": "rigid"}})
    assert z.status_code == 200, z.text
    d = api.get("/api/object/obj/final_touch").json()
    assert d["base"]["source"] == "matanyone" and d["base"]["run"] == ma and d["base"]["stale"] is True and d["base"]["why"]
    from engine.groups import ft_run
    assert ft_run(pipe.store, "obj")[:2] == (f"ft:{ma}", False)          # the group's „Do grupy z: Final Touch” the same
    assert api.delete(f"/api/object/obj/zones/{z.json()['zone']['zone_id']}").status_code == 200
    assert api.get("/api/object/obj/final_touch").json()["base"]["stale"] is False

    # a hand correction in the start frame: remove the middle of the object, add a block on the background
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    bx, by = 8, 8
    assert not c[by, bx]
    strokes = [{"mode": "remove", "radius_px": 6, "hardness": 1.0, "p": [[cx, cy]]},
               {"mode": "add", "radius_px": 5, "hardness": 1.0, "p": [[bx, by]]}]
    r = api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": strokes})
    assert r.status_code == 200, r.text
    d = r.json()
    assert list(d["frames"]) == [str(start)] and d["sig"]
    assert json.loads((od / "final_touch" / "final_touch.json").read_text(encoding="utf-8"))["frames"][str(start)][0]["mode"] == "remove"
    info = api.get("/api/object/obj/stages").json()
    assert info["final_touch"]["frames"] == [start] and "1 klatkach" in next(s for s in info["steps"] if s["id"] == "final_touch")["why"]
    # the correction-layer overlay: green / red where the strokes act, nothing where the eraser wiped (2026-10-22)
    lo = api.get(f"/api/object/obj/final_touch/layer/{start}")
    assert lo.status_code == 200 and api.get(f"/api/object/obj/final_touch/layer/{other_frame(frames, start)}").status_code == 404
    li = cv2.imdecode(np.frombuffer(lo.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert li.shape[2] == 4 and li[..., 3].max() > 0
    # preview of base + corrections (proxy PNG) differs from the base exactly at the strokes
    pv = api.get(f"/api/object/obj/final_touch/frame/{start}?b={ma}")
    assert pv.status_code == 200
    base_pv = api.get(f"/api/object/obj/frame/{start}?view=alpha&v={ma}")
    a = cv2.imdecode(np.frombuffer(pv.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    b = cv2.imdecode(np.frombuffer(base_pv.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert a.shape == b.shape and (a != b).any()

    # export source „Final Touch”: base + corrections; outputs.json records the base and the corrections hash
    out, mdir = _export_png(api, {"source": "final_touch"})
    assert out["runs"] == {"obj": f"ft:{ma}"}
    m = cv2.imread(str(mdir / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert m[cy, cx] == 0 and m[by, bx] == 255
    other = next(f for f in frames if f != start)
    ent = next(e for e in out["outputs"]["objects"] if e["name"] == "obj")
    assert ent["version"] == f"ft:{ma}" and ent["final_touch"] == {"base": ma, "corrections_sig": d["sig"]}
    assert api.get("/api/object/obj/stages").json()["steps"][-1]["state"] == "done"
    # the same frame from the base alone: untouched; a frame without corrections = the base exactly
    _o, _ = _export_png(api, {"source": "matanyone"})
    mb = cv2.imread(str(mdir / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert mb[cy, cx] > 127 and mb[by, bx] == 0
    mo = cv2.imread(str(mdir / f"obj_{other:04d}.png"), cv2.IMREAD_GRAYSCALE)
    _export_png(api, {"source": "final_touch"})
    assert np.array_equal(cv2.imread(str(mdir / f"obj_{other:04d}.png"), cv2.IMREAD_GRAYSCALE), mo)
    # a change of the corrections -> the export is stale
    r = api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": strokes[:1]})
    assert r.status_code == 200
    assert api.get("/api/object/obj/stages").json()["steps"][-1]["state"] == "stale"

    # base changed (DECYZJA: any version, changeable): the mask 1+2 – corrections stay and go on the new base
    r = api.put("/api/object/obj/final_touch", json={"source": "core"})
    assert r.status_code == 200 and r.json()["base"] == {"source": "core", "run": "core", "chosen": True, "stale": False, "why": ""}
    out, mdir = _export_png(api, {"source": "final_touch"})
    assert out["runs"] == {"obj": "ft:core"}
    m = cv2.imread(str(mdir / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert m[cy, cx] == 0 and set(np.unique(m)) <= {0, 255}
    assert api.put("/api/object/obj/final_touch", json={"source": "version", "run": "v999"}).status_code == 400
    assert api.put("/api/object/obj/final_touch", json={"source": "bad"}).status_code == 400
    # 🔑✕: the frame's corrections removed
    d = api.delete(f"/api/object/obj/final_touch/frames/{start}").json()
    assert d["frames"] == {} and d["sig"] == ""
    assert api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": [{"mode": "x", "p": [[1, 1]]}]}).status_code == 400
    assert api.put("/api/object/obj/final_touch/frames/99999", json={"strokes": []}).status_code == 400
    # never written: the shot manifest
    assert [p.read_bytes() for p in manifest] == before


# ------------------------------------------------------------------------------------------------ finish (2026-10-22)
def test_finish_levels_gamma_erode_blur_and_order(tmp_path):
    """User decision 2026-10-22: black / white / gamma / erode / blur moved to Final Touch – global, AFTER the strokes
    (the blur / erode shape the painted edges too); defaults = base + strokes bit for bit."""
    from engine.final_touch import FINISH_DEFAULT, apply, corrections_sig, finish_alpha, load, set_finish, set_frame
    a = np.zeros((60, 80), np.float32); a[20:40, 20:60] = 1.0; a[20:40, 60:70] = 0.3; a[10:20, 20:60] = 0.05
    assert finish_alpha(a, {}) is a and finish_alpha(a, dict(FINISH_DEFAULT)) is a
    lv = finish_alpha(a, {"black": 0.1, "white": 0.9})
    assert lv[15, 30] == 0.0 and lv[30, 30] == 1.0 and abs(lv[30, 65] - (0.3 - 0.1) / 0.8) < 1e-5
    assert finish_alpha(a, {"gamma": 2.0})[30, 65] == pytest.approx(0.3 ** 0.5, abs=1e-5)
    er = finish_alpha(a, {"erode_px": 2})
    assert er[21, 21] == 0.0 and er[30, 30] == 1.0 and er[30, 65] == pytest.approx(0.3)   # grey morphology: soft stays
    di = finish_alpha(a, {"erode_px": -2})
    assert di[18, 21] == 1.0 and di[41, 30] == 1.0 and di[43, 30] == 0.0
    bl = finish_alpha(a, {"blur_px": 2.0})
    assert 0.0 < bl[19, 30] < 1.0 and bl[30, 30] == pytest.approx(1.0, abs=1e-3)
    od = tmp_path / "obj"
    set_finish(od, {"black": 0.1, "white": 0.9, "nope": 3, "erode_px": 999})
    d = load(od)
    assert d["finish"]["black"] == 0.1 and d["finish"]["erode_px"] == 50.0 and "nope" not in d["finish"]
    assert d["finish"]["gamma"] == 1.0 and corrections_sig(d) != ""
    set_finish(od, {"black": 0, "white": 1, "erode_px": 0})
    assert corrections_sig(load(od)) == ""
    # order: strokes, then the finish – an add stroke (1) survives the levels, and the erode shapes its edge too
    set_finish(od, {"black": 0.5})
    set_frame(od, 3, [{"mode": "add", "radius_px": 4, "hardness": 1.0, "p": [[30, 15]]}])
    out = apply(od, 3, a)
    assert out[15, 30] == 1.0 and out[15, 50] == 0.0 and out[30, 65] == 0.0 and out[30, 30] == 1.0
    set_finish(od, {"black": 0, "erode_px": 2})
    out = apply(od, 3, a)
    assert out[15, 30] == 1.0 and out[15, 31] == 1.0 and out[15, 34] == pytest.approx(0.05) and out[12, 30] == pytest.approx(0.05)   # the 4 px disc eroded to ~2 px, back to the base (0.05)


def test_finish_api_preview_export_and_stale(proj):
    app_module, api, shot, pipe, frames, start = proj
    r = api.put("/api/object/obj/final_touch/finish", json={"erode_px": 3, "black": 0.2})
    assert r.status_code == 200, r.text
    assert r.json()["finish"]["erode_px"] == 3.0 and r.json()["sig"]
    info = api.get("/api/object/obj/stages").json()
    assert info["final_touch"]["finish"] is True and "wykończenie" in next(s for s in info["steps"] if s["id"] == "final_touch")["why"]
    out, mdir = _export_png(api, {"source": "final_touch"})
    assert out["runs"] == {"obj": "ft:core"}
    m = cv2.imread(str(mdir / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    c = pipe.load_core("obj", [start])[start]
    assert m.sum() < c.sum() and not (m & ~c).any()                        # eroded inside the core
    assert api.get("/api/object/obj/stages").json()["steps"][-1]["state"] == "done"
    assert api.put("/api/object/obj/final_touch/finish", json={"erode_px": 0}).status_code == 200
    assert api.get("/api/object/obj/stages").json()["steps"][-1]["state"] == "stale"
    assert api.put("/api/object/obj/final_touch/finish", json={"black": "x"}).status_code == 422


def test_layer_box_roi_equals_full_frame_arithmetic():
    """layer() works per stroke box; it must equal the full-frame arithmetic bit for bit, and C is 0 outside the box."""
    import numpy as np
    from engine.final_touch import layer_box, stroke_mask
    w, h = 200, 120
    st = [{"mode": "add", "radius_px": 14, "hardness": 0.3, "p": [[20, 20], [60, 40]]},
          {"mode": "remove", "radius_px": 9, "hardness": 1.0, "p": [[50, 30], [190, 110]]},
          {"mode": "erase", "radius_px": 12, "hardness": 0.0, "p": [[30, 25], [40, 30]]},
          {"mode": "add", "radius_px": 6, "hardness": 1.0, "p": [[-30, -30], [-20, -25]]}]
    c = np.zeros((h, w), np.float32)
    for s in st:
        m = stroke_mask(s, w, h)
        c = c + m * (1 - c) if s["mode"] == "add" else (c - m * (1 + c) if s["mode"] == "remove" else c * (1 - m))
    got, box = layer_box(st, w, h)
    assert np.array_equal(got, c)
    x0, y0, x1, y1 = box
    out = got.copy(); out[y0:y1, x0:x1] = 0
    assert not out.any()
    from engine.final_touch import layer_roi   # the box alone (live preview, 2026-10-28): the same values
    rbox, rc = layer_roi(st, w, h)
    assert rbox == box and np.array_equal(rc, got[y0:y1, x0:x1])
    assert layer_roi(st[3:], w, h) == (None, None)


# ------------------------------------------------------------------------------------------------ proxy preview (NEXT_STEPS 7c, 2026-10-28)
@pytest.mark.parametrize("w,h", [(4096, 2731), (4448, 3096), (3840, 2160), (2048, 1152), (6000, 4000), (1000, 700)])
@pytest.mark.parametrize("fmt", ["png", "exr"])
def test_proxy_preview_equals_full_resize(monkeypatch, w, h, fmt):
    """GET final_touch/frame: the cached base proxy + the column strip of the strokes' box (aligned to the INTER_AREA
    blocks) = the old full-resolution compose + resize, bit for bit, at any plate size and export format."""
    import app as app_module
    from engine import final_touch as ft
    rng = np.random.default_rng(w + h)
    base = cv2.GaussianBlur((rng.random((h, w)) > 0.6).astype(np.float32), (0, 0), 2.0) * np.float32(1.02) - np.float32(0.01)
    quant = (lambda a: a.astype(np.float16).astype(np.float32)) if fmt == "exr" else (lambda a: np.clip(a, 0, 1).astype(np.float32))
    od = Path("/nonexistent")
    frames = {"5": [ft.clean_stroke(s) for s in (
        {"mode": "add", "radius_px": 30, "hardness": 0.6, "p": [[w * 0.3, h * 0.4], [w * 0.35, h * 0.5]]},
        {"mode": "remove", "radius_px": 12, "hardness": 1.0, "p": [[w * 0.32, h * 0.45, 20], [w * 0.31, h * 0.6, 4]]},
        {"mode": "erase", "radius_px": 9, "hardness": 0.0, "p": [[3, 3], [w - 1, 7]]},   # touches both borders
    )]}
    ent = {"base": base}
    monkeypatch.setattr(app_module, "_export_quantize", quant)
    monkeypatch.setattr(app_module, "_proxy_scale", lambda: min(1.0, 1920.0 / max(w, h)))
    monkeypatch.setattr(app_module, "_ft_base_entry", lambda *a: ent)
    for strokes in (frames["5"][:1], frames["5"][:2], frames["5"], []):
        d = {"base": {}, "frames": {"5": strokes} if strokes else {}, "finish": dict(ft.FINISH_DEFAULT)}
        monkeypatch.setattr(ft, "load", lambda _od, d=d: d)
        pw, ph = app_module._ft_proxy_size(w, h)
        if strokes:
            full = quant(np.clip(base, 0, 1).astype(np.float32))
            a = quant(ft.compose(base, ft.layer(strokes, w, h)))
            c, box = ft.layer_box(strokes, w, h)
            full[box[1]:box[3], box[0]:box[2]] = a[box[1]:box[3], box[0]:box[2]]
        else:
            full = quant(base)
        want = app_module._alpha_u8(cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA))
        got = app_module._ft_frame_proxy(od, "v001", 5, w, h)
        assert got.shape == want.shape and np.array_equal(got, want), (w, h, fmt, len(strokes), int((got != want).sum()))
    assert np.array_equal(app_module._ft_frame_proxy(od, "v001", 5, w, h, base_only=True),
                          app_module._alpha_u8(cv2.resize(quant(np.clip(base, 0, 1).astype(np.float32)), (pw, ph), interpolation=cv2.INTER_AREA)))


def test_export_and_preview_bit_for_bit_after_the_proxy_cache(proj):
    """NEXT_STEPS 7c: the cheaper preview must not touch the export – the PNG export of source „Final Touch” is still
    u8(compose(base, layer)) at full resolution, the preview its INTER_AREA proxy, the base=1 preview the base alone."""
    app_module, api, shot, pipe, frames, start = proj
    from engine import final_touch as ft
    from engine.export import run_alpha_loader
    od = pipe.store.dir("obj")
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    ma = r.json()["run"]
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    strokes = [{"mode": "remove", "radius_px": 7, "hardness": 0.4, "p": [[cx - 5, cy], [cx + 6, cy + 3]]},
               {"mode": "add", "radius_px": 6, "hardness": 0.7, "p": [[10, 10], [30, 14, 9]]},
               {"mode": "erase", "radius_px": 3, "hardness": 0.0, "p": [[cx, cy]]}]
    assert api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": strokes}).status_code == 200
    w, h = shot.w, shot.h
    base = run_alpha_loader(od, ma, w, h)(start)
    want = (np.clip(ft.compose(base, ft.layer(ft.load(od)["frames"][str(start)], w, h)), 0, 1) * 255 + 0.5).astype(np.uint8)
    _out, mdir = _export_png(api, {"source": "final_touch"})
    assert np.array_equal(cv2.imread(str(mdir / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE), want)
    s = app_module._proxy_scale()
    pw, ph = max(1, round(w * s)), max(1, round(h * s))
    pv = api.get(f"/api/object/obj/final_touch/frame/{start}?b={ma}")
    got = cv2.imdecode(np.frombuffer(pv.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    full = np.clip(ft.compose(base, ft.layer(ft.load(od)["frames"][str(start)], w, h)), 0, 1).astype(np.float32)
    assert np.array_equal(got, app_module._alpha_u8(cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA)))
    bp = api.get(f"/api/object/obj/final_touch/frame/{start}?b={ma}&base=1")
    gb = cv2.imdecode(np.frombuffer(bp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(gb, app_module._alpha_u8(cv2.resize(np.clip(base, 0, 1).astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)))
