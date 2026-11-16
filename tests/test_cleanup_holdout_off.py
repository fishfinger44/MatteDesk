# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Palette „Złożenie” → Ochrona (user decisions 2026-11-15): the kept objects listed with their masks, „chroń” per
object (`holdout_off`, per cleaned thing) honoured by the ONE keep-loader list (api/cleanup._alpha_loaders) – so an
unticked object cuts neither the composite (Cleaner run / „⟲ Złóż ponownie” / live preview) nor the model's mask –
and the overlay endpoint: the kept object's holdout alpha at preview size + where it cuts the patch. Real case:
_fotage_testy, „Hair” / „Earrings” overlapping the cheek cut the patch and plate skin showed through. Synthetic."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as A
from engine import cleanup as cu
from engine import h3_fill
from engine import inpaint_recompose as IR

W, H = 240, 176
FRAMES = (1, 2, 3)


def _disc(cx, cy, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.hypot(xx - cx, yy - cy) < r).astype(np.float32)


OBJ = _disc(120, 88, 24)                 # the removed object
HAIR = _disc(140, 88, 14)                # kept, overlapping the removed object's right side
EAR = _disc(40, 30, 8)                   # kept, far away (cuts nothing)
OVERLAP = (OBJ > 0) & (HAIR > 0)


@pytest.fixture
def ho_app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    bdir = tmp_path / "b"
    bdir.mkdir()
    rng = np.random.default_rng(3)
    for f in FRAMES:
        p = np.clip(110 + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
        p[OBJ > 0] = (20, 20, 230)
        cv2.imwrite(str(bdir / f"f_{f:04d}.png"), p)
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(bdir / "f_####.png")}})
    monkeypatch.setitem(A.STATE, "shot_project", None)
    monkeypatch.setattr(A, "_plate_size", lambda: (W, H))
    monkeypatch.setattr(A, "_work_frames", lambda: list(FRAMES))
    alphas = {"obj": OBJ, "Hair": HAIR, "Earrings": EAR}
    runs = {"obj": "r1", "Hair": "ft:v008", "Earrings": "core"}

    class Objs:
        def list(self, groups=True):
            return [{"name": n} for n in alphas]

        def dir(self, n):
            return tmp_path / n
    import engine.cleanup_masks as CM
    import engine.export as EX
    monkeypatch.setattr(A, "_objects", lambda: Objs())
    monkeypatch.setattr(CM, "resolve", lambda store, o, src: {
        "run": runs[o["name"]], "source": {"r1": "core", "ft:v008": "final_touch", "core": "core"}[runs[o["name"]]],
        "valid": True, "by": "own", "why": ""})
    monkeypatch.setattr(CM, "available", lambda store, name: [])
    monkeypatch.setattr(EX, "run_alpha_loader", lambda d, run, w, h: (lambda f, a=alphas[d.name]: a))
    vs = {"remove": ["obj"], "model": "h3", "grow_px": 6, "soft_px": 0, "gen_margin_px": 30, "grain": "off",
          "tone": False}
    cu.save_settings(work, vs)
    v = cu.next_version(work)
    (v / "model").mkdir(parents=True)
    cw, ch = h3_fill.canvas(W, H)
    for f in FRAMES:
        cv2.imwrite(str(v / "model" / f"m_{f:04d}.png"), np.full((ch, cw, 3), 200, np.uint8))
    (v / "run.json").write_text(json.dumps({"settings": cu.settings(vs), "frames": [1, 3], "backend": "h3",
                                            "ext": "png", "date": "2026-11-15T10:00:00"}), encoding="utf-8")
    A_HO["alphas"] = alphas
    return TestClient(A.app), work, v


A_HO = {}


def test_setting_normalised_and_per_target():
    s = cu.settings({"holdout_off": ["Hair", "Hair", "", "Earrings"]})
    assert s["holdout_off"] == ["Earrings", "Hair"] and cu.settings({})["holdout_off"] == []
    assert "holdout_off" in cu.PER_TARGET
    s = cu.settings({"remove": ["a"], "holdout_off": ["Hair"]})
    cu.switch_target(s, "a", "b")
    assert s["holdout_off"] == [] and s["per_target"]["a"]["holdout_off"] == ["Hair"]   # b: its own (default)
    cu.switch_target(s, "b", "a")
    assert s["holdout_off"] == ["Hair"]
    # a recomposite takes it from the version's cleaned thing, like the band
    vs = {"remove": ["a"], "model": "h3", "holdout_off": []}
    assert IR.recompose_settings({"remove": ["a"], "holdout_off": ["Hair"]}, vs)["holdout_off"] == ["Hair"]
    assert IR.recompose_settings({"remove": ["b"], "per_target": {"a": {"holdout_off": ["Ear"]}}}, vs)["holdout_off"] == ["Ear"]


def test_get_lists_the_kept_objects_and_put_saves(ho_app):
    c, work, v = ho_app
    kept = c.get("/api/cleanup").json()["kept"]
    assert [(k["name"], k["mask"]["run"], k["on"]) for k in kept] == [("Hair", "ft:v008", True), ("Earrings", "core", True)]
    r = c.put("/api/cleanup", json={"holdout_off": ["Hair"]})
    assert r.status_code == 200 and r.json()["settings"]["holdout_off"] == ["Hair"]
    kept = {k["name"]: k["on"] for k in c.get("/api/cleanup").json()["kept"]}
    assert kept == {"Hair": False, "Earrings": True}


def test_alpha_loaders_honour_holdout_off(ho_app):
    c, work, v = ho_app
    al = A._cleanup_exports["alpha_loaders"]
    s = cu.load_settings(work)
    rm, keep = al(s, W, H)
    assert cu.names_of(keep) == ["Hair", "Earrings"]
    on = cu.mask_fn(s, rm, keep)(2)
    assert on[OVERLAP].max() == 0.0                       # protected: the patch is cut under the hair
    s_off = {**s, "holdout_off": ["Hair"]}
    rm, keep = al(s_off, W, H)
    assert cu.names_of(keep) == ["Earrings"]
    assert cu.mask_fn(s_off, rm, keep)(2)[OVERLAP].min() == 1.0          # unticked: the patch covers it
    assert cu.gen_mask_fn(s_off, rm, keep)(2)[OVERLAP].min() == 1.0      # … the model's mask too


def _patch_alpha(vdir, f):
    return cv2.imread(str(vdir / "patch" / f"p_{f:04d}.png"), cv2.IMREAD_UNCHANGED)[..., 3]


def test_recompose_and_preview_follow_the_tick(ho_app):
    c, work, v = ho_app
    r = c.post(f"/api/cleanup/versions/{v.name}/recompose")
    assert r.status_code == 200, r.text
    a_on = _patch_alpha(cu.cleanup_dir(work) / r.json()["version"], 2)
    assert a_on[OVERLAP].max() == 0
    pv_on = c.get(f"/api/cleanup/versions/{v.name}/preview/2").content
    assert c.put("/api/cleanup", json={"holdout_off": ["Hair"]}).status_code == 200
    r = c.post(f"/api/cleanup/versions/{v.name}/recompose")
    assert r.status_code == 200, r.text
    new = cu.cleanup_dir(work) / r.json()["version"]
    assert _patch_alpha(new, 2)[OVERLAP].min() == 255
    assert json.loads((new / "run.json").read_text(encoding="utf-8"))["settings"]["holdout_off"] == ["Hair"]
    pv_off = c.get(f"/api/cleanup/versions/{v.name}/preview/2").content
    assert pv_on != pv_off                                  # the live preview's cache saw the change


def test_preview_cache_sees_a_new_final_touch(ho_app, tmp_path):
    """User decision 2026-11-15, point 3: a kept object's mask fixed in cutting (a Final Touch stroke) and back in
    Cleanup – the preview cache stamp covers final_touch.json, so the cached alphas are read again."""
    c, work, v = ho_app
    pv = lambda: c.get(f"/api/cleanup/versions/{v.name}/preview/2").content   # noqa: E731
    first = pv()
    A_HO["alphas"]["Hair"] = np.zeros_like(HAIR)            # the mask changed, no file touched: the cache answers
    assert pv() == first
    ft = tmp_path / "Hair" / "final_touch"
    ft.mkdir(parents=True)
    (ft / "final_touch.json").write_text("{}", encoding="utf-8")   # what a Final Touch stroke writes
    assert pv() != first
    r = c.get("/api/cleanup/holdout/Hair/2")                # the overlay shows the new mask too
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)[..., 2].max() == 0


def test_holdout_overlay_alpha_and_cut(ho_app):
    c, work, v = ho_app
    sc = A._proxy_scale()
    size = (round(H * sc), round(W * sc))
    r = c.get("/api/cleanup/holdout/Hair/2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == size
    want_a = cv2.resize((HAIR * 255).astype(np.uint8), size[::-1], interpolation=cv2.INTER_AREA)
    assert np.array_equal(img[..., 2], want_a)             # R = exactly the keep loader's alpha at preview size
    assert img[..., 1].max() > 200 and int(r.headers["X-Holdout-Cut-Px"]) >= int(OVERLAP.sum())
    assert r.headers["X-Holdout-On"] == "1" and r.headers["X-Holdout-Mask"] == "ft:v008"
    # the cut lies inside the hair alpha and inside the patch
    s = cu.load_settings(work)
    rm, _ = A._cleanup_exports["alpha_loaders"]({**s, "holdout": False}, W, H)
    patch = cu.comp_mask_fn({**s, "holdout": False}, rm, [], cu.engine_reach_px("h3", s, W, H))(2)
    cut_full = HAIR * patch
    want_g = cv2.resize((cut_full * 255).astype(np.uint8), size[::-1], interpolation=cv2.INTER_AREA)
    assert np.array_equal(img[..., 1], want_g)
    # the far object cuts nothing; an unticked one is still shown (what it WOULD cut), marked off
    e = c.get("/api/cleanup/holdout/Earrings/2")
    assert e.status_code == 200 and e.headers["X-Holdout-Cut-Px"] == "0"
    c.put("/api/cleanup", json={"holdout_off": ["Hair"]})
    r2 = c.get("/api/cleanup/holdout/Hair/2")
    assert r2.headers["X-Holdout-On"] == "0" and r2.headers["X-Holdout-Cut-Px"] == r.headers["X-Holdout-Cut-Px"]
    # the removed object is not a kept one
    assert c.get("/api/cleanup/holdout/obj/2").status_code == 404
