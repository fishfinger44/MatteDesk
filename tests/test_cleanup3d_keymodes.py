# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D key modes (package P13, engine/cleanup3d/keymodes.py) on the synthetic shot of test_cleanup3d_render.py:
a ground plane + sky, a camera that rotates and moves forward, an object crossing the frame.

- "one" (klucz-suma): the master key of a card covers the background everywhere the moving object was (other frames
  saw it), its canvas is larger than the frame (the camera pans), the key frame's own clean pixels are its plate, a
  static object nobody sees behind becomes the regeneration area `unseen`, and the master key projects back onto
  every frame like any key (the canvas path of the geometry).
- "auto": a static scene needs only the first and the last key; a lighting change in the middle of the shot puts keys
  around it until every frame's projection error is under the threshold; min / max spacing are kept.
"""
import numpy as np
import pytest

from engine.cleanup3d import keymodes as KM
from engine.cleanup3d import render as RD
from engine.cleanup3d import store
from engine.cleanup3d.geom import CardGeometry
from tests.test_cleanup3d_render import FRAMES, H, RED, W, make_camera, make_cards, obj_alpha, true_bg


@pytest.fixture(scope="module")
def scene():
    cam = make_camera()
    cards = make_cards()
    geo = CardGeometry(cam, cards, device="cpu")
    bg = {f: true_bg(cam, f) for f in FRAMES}
    return cam, cards, geo, bg


def _plate(bg, f, gain=1.0, extra=None):
    p = bg[f] * np.float32(gain)
    p = np.where(obj_alpha(f)[..., None] > 0, RED, p)
    if extra is not None:
        p = np.where(extra[..., None] > 0, RED, p)
    return p.astype(np.float32)


def _static_obj():
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx >= 230) & (xx < 262) & (yy >= 150) & (yy < 185)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- one: master key
def test_master_key_covers_where_the_object_was(scene):
    cam, cards, geo, bg = scene
    ground = cards[0]
    res = KM.build_master(geo, ground, FRAMES, lambda f: _plate(bg, f), obj_alpha, key_frame=5)
    cv = res["canvas"]
    x0, y0 = int(round(-cv["x0"])), int(round(-cv["y0"]))
    assert cv["frame"] == 5 and cv["w"] >= W and cv["h"] >= H
    assert cv["w"] > W or cv["h"] > H, "the camera moves: the canvas grows past the frame"
    assert cv["w"] <= int(np.ceil(W * (1 + 2 * KM.OVER))) + 1 and cv["h"] <= int(np.ceil(H * (1 + 2 * KM.OVER))) + 1
    img = res["image"]
    assert img.shape == (cv["h"], cv["w"], 3) and len(res["frames"]) > 1
    inframe = img[y0:y0 + H, x0:x0 + W]
    w5 = np.asarray(geo.card_weight(ground, 5))
    obj = (obj_alpha(5) > 0) & (w5 > 0.999)
    assert obj.sum() > 300
    err_obj = np.abs(inframe[obj] - bg[5][obj]).mean()
    assert err_obj < 0.03, f"behind the object: {err_obj:.4f}"
    # the key frame's own clean ground = its plate (the plate rule: the plate wherever it is available)
    clean = (w5 > 0.999) & (np.asarray(KM._grow(obj_alpha(5), KM.MASTER_MASK_GROW + KM.MASTER_EDGE_PX + 2)) == 0)
    assert np.abs(inframe[clean] - bg[5][clean]).max() < 2e-3
    # the moving object never leaves a hole of the master key where it was in the key frame
    hole_in = res["hole"][y0:y0 + H, x0:x0 + W] > 0
    assert hole_in[obj].mean() < 0.02


def test_master_key_marks_what_no_frame_sees(scene, tmp_path):
    cam, cards, geo, bg = scene
    ground = cards[0]
    st = _static_obj()                         # fixed in the picture: the camera moves, but frame 5's view of the
    res = KM.build_master(geo, ground, FRAMES, lambda f: _plate(bg, f, extra=st),   # ground behind it is never clean
                          lambda f: np.maximum(obj_alpha(f), st), key_frame=5)
    x0, y0 = int(round(-res["canvas"]["x0"])), int(round(-res["canvas"]["y0"]))
    assert res["hole_px"] > 0
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    store.save_camera(work, cam)
    store.save_cards(work, cards)
    key = KM.save_master(work, ground.id, res, "exr")
    assert key.canvas and key.canvas["w"] == res["canvas"]["w"] and key.versions[-1]["source"] == KM.MASTER_SOURCE
    areas = store.load_areas(work, ground.id, 5)
    assert [a.id for a in areas] == [KM.UNSEEN_AREA]
    from engine.cleanup3d import keys as KS
    m = KS.areas_mask(work, ground.id, 5, (res["canvas"]["h"], res["canvas"]["w"]))
    centre = m[y0 + 160:y0 + 175, x0 + 240:x0 + 252]
    assert centre.mean() > 0.9, "the static object's core is never seen clean"
    # a rebuild without a hole drops the area; the same canvas adds a version
    res2 = KM.build_master(geo, ground, FRAMES, lambda f: _plate(bg, f), obj_alpha, key_frame=5,
                           canvas=res["canvas"])
    key2 = KM.save_master(work, ground.id, res2, "exr")
    assert len(key2.versions) == 2 and all(a.id != KM.UNSEEN_AREA for a in store.load_areas(work, ground.id, 5))
    # an other canvas over existing versions is refused (they would be projected through a wrong view)
    res3 = dict(res2, canvas={**res2["canvas"], "w": res2["canvas"]["w"] + 4})
    with pytest.raises(ValueError, match="płótno"):
        KM.save_master(work, ground.id, res3, "exr")


def test_master_key_projects_back_onto_every_frame(scene, tmp_path):
    cam, cards, geo, bg = scene
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    store.save_camera(work, cam)
    store.save_cards(work, cards)
    for c in cards:
        res = KM.build_master(geo, c, FRAMES, lambda f: _plate(bg, f), obj_alpha, key_frame=5)
        KM.save_master(work, c.id, res, "exr")
    sc = RD.Scene(work, device="cpu")
    try:
        assert all(len(sc.keys_by_card[c.id]) == 1 and sc.keys_by_card[c.id][0].canvas for c in cards)
        for f in (1, 9):
            fill, wgt = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, f, sc.cache, settings={"edge_px": 16.0, "edge_hard_px": 4.0})
            obj = obj_alpha(f) > 0
            ok = obj & (wgt > 0.99)
            assert ok.sum() > 0.8 * obj.sum(), f"F{f}: the master keys reach the object"
            err = np.abs(fill[ok] - bg[f][ok]).mean()
            assert err < 0.04, f"F{f}: master keys projected = the clean background ({err:.4f})"
    finally:
        sc.close()


# ---------------------------------------------------------------------------------------------- auto
def test_auto_keys_static_scene_needs_only_the_ends(scene):
    cam, cards, geo, bg = scene
    res = KM.auto_key_frames(geo, cards[0], FRAMES, lambda f: _plate(bg, f), obj_alpha, threshold=0.05, min_gap=1,
                             max_gap=0)
    assert res["frames"][0] == FRAMES[0] and res["frames"][-1] == FRAMES[-1]
    assert res["max_error"] <= 0.05
    assert len(res["frames"]) <= 3, res


def test_auto_keys_follow_a_change_in_the_shot(scene):
    cam, cards, geo, bg = scene
    gain = {f: (1.0 if f <= 5 else 1.45) for f in FRAMES}        # the light changes between F5 and F6
    plate = lambda f: _plate(bg, f, gain[f])                     # noqa: E731
    base = KM.auto_key_frames(geo, cards[0], FRAMES, lambda f: _plate(bg, f), obj_alpha, threshold=0.05, min_gap=1,
                              max_gap=0)
    res = KM.auto_key_frames(geo, cards[0], FRAMES, plate, obj_alpha, threshold=0.05, min_gap=1, max_gap=0)
    ks = res["frames"]
    assert len(ks) > len(base["frames"]), (ks, base["frames"])
    assert any(2 <= k <= 5 for k in ks) and any(6 <= k <= 8 for k in ks), ks
    assert res["max_error"] <= 0.05 and all(res["errors"][k] == 0.0 for k in ks)
    # spacing: min_gap is never broken by an added key, max_gap splits long gaps, max_keys stops the loop
    res2 = KM.auto_key_frames(geo, cards[0], FRAMES, plate, obj_alpha, threshold=0.001, min_gap=3, max_gap=0)
    added = [k for k in res2["frames"] if k not in (FRAMES[0], FRAMES[-1])]
    assert all(min(abs(k - o) for o in res2["frames"] if o != k) >= 3 for k in added), res2["frames"]
    res3 = KM.auto_key_frames(geo, cards[0], FRAMES, lambda f: _plate(bg, f), obj_alpha, threshold=0.5, min_gap=1,
                              max_gap=3)
    assert max(np.diff(res3["frames"])) <= 3, res3["frames"]
    res4 = KM.auto_key_frames(geo, cards[0], FRAMES, plate, obj_alpha, threshold=0.001, min_gap=1, max_gap=0,
                              max_keys=3)
    assert len(res4["frames"]) == 3


def test_auto_settings_clamped():
    s = KM.auto_settings({"threshold": 9, "min_gap": 0, "max_gap": -4, "max_keys": 1.6, "other": 3})
    assert s == {"threshold": 0.5, "min_gap": 1, "max_gap": 0, "max_keys": 2}
