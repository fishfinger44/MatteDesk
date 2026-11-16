# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup robustness fixes of 2026-11-10:
1. LaMa at the crop's full resolution in the plain Cleanup (and the Cleanup 3D fallback): `cu.LamaInpainter` runs every
   crop through engine/lama_full.py, the ComfyUI node (256 x 256) for the rest when the worker fails;
2. kept objects that look lost (sh060: „Plaza_A” put on the removed woman's OUTLINE in F10–F14): `cu.check_kept` /
   `cu.kept_guard`, one check for both renders – a warning, the holdout changed only with `holdout_skip_lost`;
3. the Cleanup 3D result goes stale (↻) when the render settings change (store.render_sig);
4. a work range longer than the 3D camera: C5 renders the frames the camera has and says which were skipped.
Synthetic tests run with MD_LAMA_FULL=0 (tests/conftest.py) – lama_full is monkeypatched where it is used."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import cleanup as cu  # noqa: E402
from engine import lama_full  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

W, H = 160, 120


def scene():
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.dstack([xx / W, yy / H, np.full((H, W), 0.3)]).astype(np.float32)
    obj = (np.hypot(xx - 60, yy - 60) < 18).astype(np.float32)
    plate = bg.copy()
    plate[obj > 0] = (0.9, 0.1, 0.1)
    return bg, obj, plate


def _full_fake(seen, colour=(0.25, 0.75, 0.25), fail_after=None):
    def full(crop, m, cfg=None, **kw):
        if fail_after is not None and len(seen) >= fail_after:
            raise lama_full.LamaFullError("worker LaMa: no CUDA")
        seen.append((np.asarray(crop).shape, float(np.asarray(crop).max())))
        out = np.asarray(crop, np.float32).copy()
        out[np.asarray(m) > 0] = colour
        return out
    return full


# ----------------------------------------------------------------------------- 1. LaMa at full resolution
def test_cleaner_lama_full_resolution_per_crop(tmp_path, monkeypatch):
    bg, obj, plate = scene()
    seen = []
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: True)
    monkeypatch.setattr(lama_full, "inpaint", _full_fake(seen))

    def node(*a, **k):
        raise AssertionError("the node must not run while the worker works")
    monkeypatch.setattr(cu, "lama_batch_comfy", node)
    lama = cu.LamaInpainter(None, tmp_path / "io")
    c = cu.Cleaner({"grow_px": 2, "soft_px": 0}, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [], inpaint=lama)
    rj = c.run([1, 2, 3], tmp_path / "v001", "png")
    assert len(seen) == 3 and rj["lama"] == {"full": 3, "node": 0} and rj["backend"] == "lama"
    assert rj["counts"]["LaMa"] > 0 and not rj["notes"]
    comp = cv2.imread(str(tmp_path / "v001" / "comp" / "c_0002.png")).astype(np.float32) / 255
    assert np.abs(comp[60, 60] - (0.25, 0.75, 0.25)).max() < 2 / 255                       # the worker's fill pasted
    out = c.mask(2) == 0
    assert np.abs(comp[out] - plate[out]).max() <= 1 / 255 + 1e-6                          # the plate outside the mask


def test_worker_failure_falls_back_to_the_node_for_the_rest(tmp_path, monkeypatch):
    bg, obj, plate = scene()
    seen, sent = [], []
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: True)
    monkeypatch.setattr(lama_full, "inpaint", _full_fake(seen, fail_after=1))

    def node(client, io, imgs, masks, model_name="big-lama.pt"):
        sent.append(len(imgs))
        return [np.where(np.asarray(m)[..., None] > 0, 0.5, im).astype(np.float32) for im, m in zip(imgs, masks)]
    monkeypatch.setattr(cu, "lama_batch_comfy", node)
    lama = cu.LamaInpainter(object(), tmp_path / "io")
    c = cu.Cleaner({"grow_px": 2, "soft_px": 0, "batch": 2}, W, H, False, plate=lambda f: plate,
                   remove_alphas=lambda f: [obj], keep_alphas=lambda f: [], inpaint=lama)
    rj = c.run([1, 2, 3], tmp_path / "v001", "png")
    # batch 1 (F1, F2): F1 by the worker, it fails on F2 -> the node for F2; batch 2 (F3): the node only
    assert len(seen) == 1 and sent == [1, 1] and rj["lama"] == {"full": 1, "node": 2}
    assert rj["backend"] == "lama" and sum("węzeł ComfyUI (256 × 256)" in n for n in rj["notes"]) == 1


def test_worker_only_without_the_node_and_linear_plates_go_through_acescct(tmp_path, monkeypatch):
    bg, obj, plate = scene()
    lin = plate * 4.0                                       # scene-linear values over 1
    seen = []
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: True)
    monkeypatch.setattr(lama_full, "inpaint", _full_fake(seen, colour=(0.5, 0.5, 0.5)))
    c = cu.Cleaner({"grow_px": 2, "soft_px": 0}, W, H, True, plate=lambda f: lin, remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [], inpaint=cu.LamaInpainter(None, tmp_path / "io"))
    rj = c.run([1], tmp_path / "lin", "exr")
    assert seen and all(mx <= 1.0 for _, mx in seen)       # the model got the ACEScct encoding (0..1)
    comp = cv2.imread(str(tmp_path / "lin" / "comp" / "c_0001.exr"), cv2.IMREAD_UNCHANGED)
    want = cu.from_model(np.full((1, 1, 3), 0.5, np.float32), True)[0, 0]
    assert np.abs(comp[60, 60] - want).max() < 1e-3 and rj["linear"]
    # the worker failing with no node: the Cleaner's OpenCV fallback, said in the notes
    monkeypatch.setattr(lama_full, "inpaint", _full_fake([], fail_after=0))
    c2 = cu.Cleaner({}, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj], keep_alphas=lambda f: [],
                    inpaint=cu.LamaInpainter(None, tmp_path / "io"))
    rj2 = c2.run([1], tmp_path / "b", "png")
    assert rj2["backend"] == "opencv" and rj2["counts"]["OpenCV"] > 0


def test_api_inpainter_picks_full_resolution_with_the_node_as_fallback(tmp_path, monkeypatch):
    from api.cleanup import make
    from engine import cleanplate_auto
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    _, ex = make(STATE={}, objects=None, plate_size=lambda: (W, H), work_frames=lambda: [1], proxy_scale=lambda: 1.0,
                 comfy_client=lambda: object(), long_job=lambda *a, **k: (lambda f: f), refresh_manifest=lambda: {})
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: True)
    monkeypatch.setattr(cleanplate_auto, "lama_available", lambda client: False)
    inp = ex["inpainter"](work, cu.settings())
    assert isinstance(inp, cu.LamaInpainter) and inp.full and inp.client is None
    monkeypatch.setattr(cleanplate_auto, "lama_available", lambda client: True)
    inp = ex["inpainter"](work, cu.settings())
    assert inp.full and inp.client is not None
    monkeypatch.setattr(lama_full, "available", lambda cfg=None: False)
    inp = ex["inpainter"](work, cu.settings())
    assert isinstance(inp, cu.LamaInpainter) and not inp.full and inp.where == "w ComfyUI"
    monkeypatch.setattr(cleanplate_auto, "lama_available", lambda client: False)
    assert ex["inpainter"](work, cu.settings()) is None                                 # OpenCV


# ----------------------------------------------------------------------------- 2. kept objects that look lost
def _ring(a, px=2):
    hard = (np.asarray(a) > 0.5).astype(np.uint8)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return ((cv2.dilate(hard, k) > 0) & ~(cv2.erode(hard, k) > 0)).astype(np.float32)


def _person():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx > 120) & (xx < 132) & (yy > 50) & (yy < 90)).astype(np.float32)   # a small kept person behind


def test_check_kept_finds_the_outline_and_the_area_jump():
    _, obj, _ = scene()
    person = _person()
    big = np.zeros((H, W), np.float32)
    big[:, 100:] = 1                                        # F7: the mask suddenly covers a third of the frame

    def keep(f):
        return [_ring(obj) if f in (3, 4) else big if f == 7 else person]
    rep = cu.check_kept(range(1, 10), lambda f: [obj], keep, ["Plaza_A"])
    assert rep["lost"] == {"Plaza_A": [3, 4, 7]} and rep["reasons"] == {"Plaza_A": "edge"} and rep["idx"] == {0: [3, 4, 7]}
    only_area = cu.check_kept(range(1, 10), lambda f: [obj], lambda f: [big if f == 7 else person], ["P"])
    assert only_area["lost"] == {"P": [7]} and only_area["reasons"] == {"P": "area"}
    assert cu.check_kept(range(1, 10), lambda f: [obj], lambda f: [person], ["P"])["lost"] == {}   # a normal object
    assert cu.frame_ranges([10, 11, 12, 13, 14]) == "F10–F14" and cu.frame_ranges([3, 4, 7], "") == "3–4, 7"
    w = cu.kept_warning("Plaza_A", [10, 11, 12, 13, 14], "edge", False)
    assert w == ("Obiekt chroniony „Plaza_A” w F10–F14 wygląda na zgubiony (maska na krawędzi usuwanego obiektu) – "
                 "popraw go albo wyłącz ochronę")


def test_cleaner_warns_and_changes_nothing_unless_asked(tmp_path):
    bg, obj, plate = scene()
    person = _person()
    ring = _ring(obj)
    keep = cu.Named(lambda f: ring if f == 2 else person, "Plaza_A")
    kept = [keep]

    def run(s, out):
        c = cu.Cleaner({"grow_px": 4, "soft_px": 0, **s}, W, H, False, plate=lambda f: plate,
                       remove_alphas=lambda f: [obj], keep_alphas=lambda f: [fn(f) for fn in kept],
                       inpaint=lambda i, m: [np.where(x[..., None] > 0, 0.5, im).astype(np.float32) for im, x in zip(i, m)],
                       keep_names=cu.names_of(kept))
        return c, c.run([1, 2, 3], tmp_path / out, "png")
    c, rj = run({}, "a")
    want = "Obiekt chroniony „Plaza_A” w F2 wygląda na zgubiony (maska na krawędzi usuwanego obiektu) – popraw go albo wyłącz ochronę"
    assert rj["notes"] == [want] and rj["kept_check"] == {"lost": {"Plaza_A": [2]}, "reasons": {"Plaza_A": "edge"},
                                                          "skipped": False, "warnings": [want]}
    on_ring = (ring > 0) & (obj > 0)
    assert (c.mask(2)[on_ring] == 0).all()                  # default: the holdout still cuts the outline (result unchanged)
    comp = cv2.imread(str(tmp_path / "a" / "comp" / "c_0002.png")).astype(np.float32) / 255
    assert np.abs(comp[on_ring] - plate[on_ring]).max() < 2 / 255                       # the silhouette stays (warned)
    c2, rj2 = run({"holdout_skip_lost": True}, "b")
    assert rj2["kept_check"]["skipped"] and "łata go nie omija" in rj2["notes"][0]
    assert (c2.mask(2)[on_ring] == 1).all()                 # the option: no holdout for Plaza_A in F2
    _, rj3 = run({"holdout": False}, "c")
    assert rj3["kept_check"]["lost"] == {} and not rj3["notes"]                         # no holdout = nothing to check


def test_versions_list_carries_the_warnings(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    bg, obj, plate = scene()
    c = cu.Cleaner({}, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [_ring(obj)], keep_names=["Plaza_A"])
    c.run([1, 2, 3], cu.next_version(work), "png")
    v = cu.versions(work)
    assert v[0]["version"] == "v001" and len(v[0]["warnings"]) == 1 and "Plaza_A" in v[0]["warnings"][0]
    assert cu.settings({"holdout_skip_lost": 1})["holdout_skip_lost"] is True and cu.settings()["holdout_skip_lost"] is False


def test_cleanup3d_render_runs_the_same_check(tmp_path):
    import test_cleanup3d_render as TR
    from engine.cleanup3d import render as RD
    shot = TR.Shot(tmp_path)

    def keep(f):
        return [_ring(TR.obj_alpha(f)) if f in (3, 4) else TR.keep_alpha(f)]

    def render(**s):
        st = {**TR.SMALL, "remove": ["obj"], "grow_px": 4.0, "soft_px": 0.0, "motion_blur": "off", **s}
        rj = RD.render(shot.work, [3, 4, 6], plate=shot.plate, remove_alphas=lambda f: [TR.obj_alpha(f)],
                       keep_alphas=keep, settings=st, keep_names=["Plaza_A"])
        out = Path(rj["dir"])
        return rj, {f: cv2.imread(str(out / "comp" / f"c_{f:04d}.png")).astype(np.float32) / 255.0 for f in (3, 4)}
    rj, comp = render()
    assert rj["kept_check"]["lost"] == {"Plaza_A": [3, 4]} and "F3–F4" in rj["notes"][0]
    ring3 = (_ring(TR.obj_alpha(3)) > 0) & (TR.obj_alpha(3) > 0)
    red = lambda img: np.abs(img - TR.RED).max(-1) < 0.05                               # noqa: E731
    assert red(comp[3])[ring3].mean() > 0.95                # warned only: the outline of the removed object stays
    rj2, comp2 = render(holdout_skip_lost=True)
    assert rj2["kept_check"]["skipped"] and red(comp2[3])[ring3].mean() < 0.05           # the option: patched over


# ----------------------------------------------------------------------------- 3. stale on render settings
def test_render_sig_follows_the_render_settings(tmp_path):
    import test_cleanup3d_render as TR
    from engine.cleanup3d import band as C3B
    from engine.cleanup3d import store
    from engine.cleanup3d import types as T
    shot = TR.Shot(tmp_path)
    work = shot.work
    cu.save_settings(work, {"remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0})
    (T.root_dir(work) / "v001").mkdir(parents=True)
    (T.root_dir(work) / "v001" / "run.json").write_text(json.dumps({"kept_check": {"warnings": ["uwaga"]}}), encoding="utf-8")
    store.save_render_mark(work, "v001")
    st = store.render_state(work)
    assert st["stale"] is False and st["warnings"] == ["uwaga"] and st["skipped"] == []

    def stale():
        return store.render_state(work)["stale"]
    for key, val in (("key_soft_px", 50.0), ("key_grow_px", 3.0), ("key_effects", False), ("key_effects_thr", 0.2),
                     ("clean_first", True)):
        old = store.load_render_settings(work)[key]
        store.save_render_settings(work, {key: val})
        assert stale(), key
        store.save_render_settings(work, {key: old})
        assert not stale(), key
    for key, val in (("grow_px", 9.0), ("soft_px", 1.0), ("holdout", False), ("remove", ["obj", "x"]),
                     ("holdout_skip_lost", True), ("sources", {"obj": "v002"})):
        old = cu.load_settings(work)[key]
        cu.save_settings(work, {**cu.load_settings(work), key: val})
        assert stale(), key
        cu.save_settings(work, {**cu.load_settings(work), key: old})
        assert not stale(), key
    C3B.save(work, {"band_points": [{"id": "p1", "x": 10, "y": 10, "w": 20}], "band_ref_out": 32.0})
    assert stale()                                          # the patch band points
    C3B.save(work, {"band_points": []})
    assert not stale()                                      # an empty band file = no band
    # the request's own soft_px (C5 „Miękkość maski”) is kept with the mark
    store.save_render_mark(work, "v001", overrides={"soft_px": 7.0})
    assert not stale() and store.render_sig(work) != store.render_sig(work, {"soft_px": 7.0})
    # a mark written before the settings were part of the signature: stale once, fresh after the next render
    store._write(store.render_mark_file(work), {"version": "v001", "sig": "0" * 40, "date": "2026-11-09 10:00:00"})
    assert stale()
    store.save_render_mark(work, "v001", skipped=[58, 59, 60])
    assert not stale() and store.render_state(work)["skipped"] == [58, 59, 60]


# ----------------------------------------------------------------------------- 4. work range longer than the camera
def test_render_route_skips_frames_without_the_camera(tmp_path, monkeypatch):
    from test_cleanup3d_api import Env
    from engine.cleanup3d import store
    env = Env(tmp_path, monkeypatch)
    c, work = env.client, env.work
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    for card in ("ground", "sky"):
        assert c.post(f"/api/cleanup3d/keys/{card}", json={"frames": [1, 5]}).status_code == 200
    cam = store.load_camera(work)
    for f in (8, 9):                                       # the camera solved only F1–F7
        cam.R.pop(f)
        cam.t.pop(f)
    store.save_camera(work, cam)
    r = c.post("/api/cleanup3d/render", json={"frames": [6, 7, 8, 9], "motion_blur": "off"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_frames"] == 2 and j["skipped"] == [8, 9] and j["skipped_text"] == "8–9"
    assert j["notes"][0] == "Kamera 3D nie ma klatek 8–9 zakresu pracy – pominięte (2)"
    out = env.shot / "inpaint" / j["version"]
    assert (out / "comp" / "c_0007.png").is_file() and not (out / "comp" / "c_0008.png").exists()
    st = c.get("/api/cleanup3d").json()
    assert st["render"]["skipped"] == [8, 9] and st["stale"] == {"project": False}
    assert st["camera"]["frames"] == [1, 7]
    r = c.post("/api/cleanup3d/render", json={"frames": [8, 9], "motion_blur": "off"})
    assert r.status_code == 409 and "nie ma żadnej klatki zakresu pracy 8–9 (kamera: 1–7)" in r.json()["detail"]


def test_ui_wiring_of_the_fixes():
    root = Path(__file__).resolve().parents[1]
    js1 = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "cuSkipLost" in js1 and "holdout_skip_lost" in js1 and "cuWarn" in js1
    js3 = (root / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    assert "c3RangeHint" in js3 and "c3RenderWarn" in js3 and "zostaną pominięte" in js3 and "skipped_text" in js3
    en = json.loads((root / "static" / "i18n" / "en.json").read_text(encoding="utf-8"))["strings"]
    for k in ("pomiń ochronę zgubionych", "Kamera 3D ma klatki {0}; zakres pracy {1} – klatki {2} zostaną pominięte",
              "Obiekt chroniony „{0}” w {1} wygląda na zgubiony ({2}) – popraw go albo wyłącz ochronę"):
        assert en.get(k), k
