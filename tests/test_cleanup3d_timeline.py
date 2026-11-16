# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P16 „Warstwowa oś czasu kart” (docs/PLAN_CLEANUP_3D.md): the server side of the timeline rows on
the synthetic shot of tests/test_cleanup3d_api.py – coverage numbers (GET /api/cleanup3d/timeline), a card turned off
for good (PUT /api/cleanup3d/cards/{card}/off: out of the render), solo like After Effects (PUT …/cards/{card}/solo:
any card soloed = only the soloed cards that are on in the render, the previews and the coverage) and the key move (POST …/keys/{card}/{frame}/move: a new key
at the target made of the target plate + the source key re-projected through the camera, the old key deleted for good)."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import timeline as TL  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402
from tests.test_cleanup3d_api import _png, env  # noqa: E402,F401 – the fixture
from tests.test_cleanup3d_render import FRAMES, H, W, obj_alpha, true_bg  # noqa: E402


def _setup(env, key_frames=(1, 9)):
    c = env.client
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").json()["added"] == ["ground", "sky"]
    for card in ("ground", "sky"):
        assert c.post(f"/api/cleanup3d/keys/{card}", json={"frames": list(key_frames)}).status_code == 200
    return c


def _clean_key(env, card, f):
    """A fill version of a key: the true clean background of its frame (the red object gone)."""
    KS.add_version(env.work, card, f, true_bg(env.cam, f), "regen:lama", "png")


def test_active_cards_rule_like_after_effects():
    def card(i, off=False, solo=False):
        c = T.Card.from_json({"id": i, "kind": "sky"})
        c.off, c.solo = off, solo
        return c
    ids = lambda cs: [c.id for c in T.active_cards(cs)]                               # noqa: E731
    assert ids([card("a"), card("b", off=True), card("c")]) == ["a", "c"]           # no solo: every card that is on
    assert ids([card("a"), card("b", solo=True), card("c", solo=True)]) == ["b", "c"]
    assert ids([card("a"), card("b", solo=True, off=True)]) == []                   # off wins over solo (AE)
    # JSON: solo only when set (a card without solo keeps its JSON = its render signature), round trip
    c = card("x")
    assert "solo" not in c.to_json() and T.Card.from_json({"id": "x", "kind": "sky"}).solo is False
    c.solo = True
    assert c.to_json()["solo"] is True and T.Card.from_json(c.to_json()).solo is True


def test_coverage_numbers_on_a_synthetic_shot(env):
    c = _setup(env, key_frames=(1, 9))
    r = c.get("/api/cleanup3d/timeline")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["frames"] == FRAMES and j["reason"] == "" and j["band_keys"] == [] and j["off"] == []
    assert set(j["cards"]) == {"ground", "sky"}
    g, s = j["cards"]["ground"], j["cards"]["sky"]
    assert len(g["share"]) == len(FRAMES) == len(g["miss"])
    # the ground and the sky split the frame: shares add up to about the whole frame where the keys cover
    for i, f in enumerate(FRAMES):
        assert 0.0 <= g["share"][i] <= 1.0 and 0.0 <= s["share"][i] <= 1.0
        assert g["share"][i] + s["share"][i] <= 1.0 + 1e-3
    assert g["share"][0] > 0.3 and s["share"][0] > 0.05
    # between the two keys the crossfade covers what the card shows (the keys' borders reach each other); at a key
    # frame the last px at the frame border (edge_hard_px) has no second key that sees them – as the render
    assert max(g["miss"][1:-1]) < 0.05
    assert g["miss"][0] < 0.15 and g["miss"][-1] < 0.15
    # only one key left at F1: the camera moves forward and turns, the far frames show ground the key never saw
    assert c.delete("/api/cleanup3d/keys/ground/9").status_code == 200
    j2 = c.get("/api/cleanup3d/timeline").json()
    g2 = j2["cards"]["ground"]
    assert g2["miss"][-1] > g2["miss"][1] + 0.02                     # the further from the key, the more is missed
    assert all(b >= a - 1e-3 for a, b in zip(g2["miss"][1:], g2["miss"][2:]))
    assert j2["cards"]["sky"] == s                                     # the other card did not change
    # no keys at all: everything the card shows is missed, nothing is filled
    assert c.delete("/api/cleanup3d/keys/ground/1").status_code == 200
    g3 = c.get("/api/cleanup3d/timeline").json()["cards"]["ground"]
    assert all(v == 0.0 for v in g3["share"]) and all(v > 0.99 for v in g3["miss"])
    # the patch band keys come with it
    from engine.cleanup3d import band as C3B
    C3B.save(env.work, {"band_points": [{"id": "p1", "keys": {"4": {"x": 10, "y": 10}, "7": {"x": 12, "y": 10}}}]})
    assert c.get("/api/cleanup3d/timeline").json()["band_keys"] == [4, 7]


def test_timeline_without_camera_says_why(env):
    r = env.client.get("/api/cleanup3d/timeline")
    assert r.status_code == 200 and r.json()["cards"] == {} and "kamery" in r.json()["reason"].lower()


def test_solo_like_after_effects_render_previews_coverage(env):
    """User 2026-11-10: solo works like After Effects – stored on the card, any solo = only the soloed cards (that are
    on) in the render, the previews and the coverage; the render signature changes and comes back."""
    c = _setup(env)
    from engine.cleanup3d import render as RD
    sig = store.render_sig(env.work)
    full = _png(c.get("/api/cleanup3d/view/proj/4"))
    cv_full = _png(c.get("/api/cleanup3d/view/cards/4"))
    b_full = _png(c.get("/api/cleanup3d/blend/4", params={"part": "maps"}))
    assert c.put("/api/cleanup3d/cards/nope/solo", json={"solo": True}).status_code == 404
    r = c.put("/api/cleanup3d/cards/sky/solo", json={"solo": True})
    assert r.status_code == 200 and r.json()["card"]["solo"] is True
    assert store.render_sig(env.work) != sig                          # a rendered result becomes stale (↻)
    assert [x.id for x in RD.Scene(env.work).cards] == ["sky"]        # the render's scene
    st = c.get("/api/cleanup3d").json()
    assert next(x for x in st["cards"] if x["id"] == "sky")["solo"] is True
    assert "solo" not in next(x for x in st["cards"] if x["id"] == "ground")
    tl = c.get("/api/cleanup3d/timeline").json()
    assert tl["solo"] == ["sky"] and tl["off"] == [] and set(tl["cards"]) == {"sky"}
    # the previews follow (no query): the ground part of the frame (bottom rows) = the darkened plate, the sky stays
    sky_only = _png(c.get("/api/cleanup3d/view/proj/4"))
    assert np.abs(sky_only[-20:].astype(int) - full[-20:].astype(int)).mean() > 10
    assert not np.array_equal(_png(c.get("/api/cleanup3d/view/cards/4")), cv_full)
    assert _png(c.get("/api/cleanup3d/blend/4", params={"part": "maps"}))[..., 2].mean() < b_full[..., 2].mean()
    # the old preview query is gone: it does not change anything
    assert np.array_equal(_png(c.get("/api/cleanup3d/view/proj/4", params={"cards": "ground"})), sky_only)
    # both soloed = every card again (same content as no solo, the signature still says solo)
    assert c.put("/api/cleanup3d/cards/ground/solo", json={"solo": True}).status_code == 200
    assert {x.id for x in RD.Scene(env.work).cards} == {"ground", "sky"}
    assert np.array_equal(_png(c.get("/api/cleanup3d/view/proj/4")), full)
    # off wins over solo: the only soloed card that is on turned off -> nothing in the scene
    assert c.put("/api/cleanup3d/cards/ground/solo", json={"solo": False}).status_code == 200
    assert c.put("/api/cleanup3d/cards/sky/off", json={"off": True}).status_code == 200
    r = c.get("/api/cleanup3d/view/proj/4")
    assert r.status_code == 409 and "wyłączone" in r.json()["detail"]
    assert c.get("/api/cleanup3d/timeline").json()["reason"]
    assert c.put("/api/cleanup3d/cards/sky/off", json={"off": False}).status_code == 200
    # a key of a card outside the solo does not move
    r = c.post("/api/cleanup3d/keys/ground/9/move", json={"to": 7})
    assert r.status_code >= 400 and "solo" in r.json()["detail"]
    assert c.put("/api/cleanup3d/cards/sky/solo", json={"solo": False}).status_code == 200
    assert store.render_sig(env.work) == sig                          # solo off: the same content as before
    assert np.array_equal(_png(c.get("/api/cleanup3d/view/proj/4")), full)
    # the flag round-trips through the whole-list PUT of the card palette
    cards = c.get("/api/cleanup3d/cards").json()["cards"]
    cards[0]["solo"] = True
    assert c.put("/api/cleanup3d/cards", json={"cards": cards}).json()["cards"][0]["solo"] is True


def test_card_off_for_good_leaves_the_render(env):
    c = _setup(env)
    from engine.cleanup3d import render as RD
    sig = store.render_sig(env.work)
    assert c.put("/api/cleanup3d/cards/nope/off", json={"off": True}).status_code == 404
    r = c.put("/api/cleanup3d/cards/ground/off", json={"off": True})
    assert r.status_code == 200 and r.json()["card"]["off"] is True
    assert store.render_sig(env.work) != sig                          # a rendered result becomes stale
    st = c.get("/api/cleanup3d").json()
    assert next(x for x in st["cards"] if x["id"] == "ground")["off"] is True
    assert [x.id for x in RD.Scene(env.work).cards] == ["sky"]
    assert c.get("/api/cleanup3d/timeline").json()["off"] == ["ground"]
    assert "ground" not in c.get("/api/cleanup3d/timeline").json()["cards"]
    assert c.put("/api/cleanup3d/cards/sky/off", json={"off": True}).status_code == 200
    r = c.get("/api/cleanup3d/view/proj/4")
    assert r.status_code == 409 and "wyłączone" in r.json()["detail"]
    assert c.put("/api/cleanup3d/cards/sky/off", json={"off": False}).status_code == 200
    assert c.put("/api/cleanup3d/cards/ground/off", json={"off": False}).status_code == 200
    assert store.render_sig(env.work) == sig                          # back on: the same content as before
    # the flag round-trips through the whole-list PUT of the card palette
    cards = c.get("/api/cleanup3d/cards").json()["cards"]
    cards[1]["off"] = True
    assert c.put("/api/cleanup3d/cards", json={"cards": cards}).json()["cards"][1]["off"] is True
    assert T.Card.from_json({"id": "x", "kind": "sky"}).off is False


def test_move_creates_the_target_key_and_deletes_the_source(env):
    c = _setup(env, key_frames=(1, 9))
    _clean_key(env, "ground", 1)
    src_dir = T.key_dir(env.work, "ground", 1)
    assert src_dir.is_dir()
    url = "/api/cleanup3d/keys/ground/1/move"
    assert c.post("/api/cleanup3d/keys/ground/2/move", json={"to": 3}).status_code == 404    # no key at F2
    assert c.post(url, json={"to": 9}).status_code == 409                                    # F9 already has a key
    assert c.post(url, json={"to": 1}).status_code == 409
    assert c.post(url, json={"to": 77}).status_code == 409                                   # no camera frame
    assert src_dir.is_dir()                                                                  # refusals changed nothing
    r = c.post(url, json={"to": 3})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["from"] == 1 and j["to"] == 3 and j["patched"] is True and j["covered"] > 0.5
    k3 = store.load_key(env.work, "ground", 3)
    assert [v["source"] for v in k3.versions] == ["plate", "move:F1:v002"] and k3.active == "v002"
    assert k3.versions[1]["moved_from"] == 1
    assert store.load_key(env.work, "ground", 1) is None and not src_dir.exists()
    assert j["freed_bytes"] > 0 and "trash" not in j and not (T.root_dir(env.work) / "trash").exists()
    # the new key: the plate of F3 outside the removal region, the clean ground where the object stood
    plate3 = cv2.imread(str(env.shot / "plates" / "p_0003.png"))
    v1 = _png(c.get("/api/cleanup3d/keys/ground/3/image", params={"version": "v001"}))
    v2 = _png(c.get("/api/cleanup3d/keys/ground/3/image"))
    assert np.array_equal(v1, plate3)
    obj = obj_alpha(3) > 0
    region = cv2.dilate(obj.astype(np.uint8), np.ones((121, 121), np.uint8)) > 0
    dd = np.abs(v2.astype(int) - plate3.astype(int)).max(-1)       # the plate away from the object (the effects
    assert dd[~region].max() <= 3 and (dd[~region] > 0).mean() < 0.01   # mask may reach a little further, by a hair)
    clean3 = np.clip(np.rint(true_bg(env.cam, 3) * 255), 0, 255).astype(np.uint8)
    ground = obj.copy()
    ground[:125] = False                                               # the part of the object over the ground card
    assert ground.sum() > 100
    err_new = np.abs(v2[ground].astype(int) - clean3[ground].astype(int)).mean()
    err_plate = np.abs(plate3[ground].astype(int) - clean3[ground].astype(int)).mean()
    assert err_new < 0.25 * err_plate, (err_new, err_plate)
    # a key whose active version is the plate itself: moved as the target plate alone
    r = c.post("/api/cleanup3d/keys/sky/1/move", json={"to": 2})
    assert r.status_code == 200 and r.json()["patched"] is False
    assert [v["source"] for v in store.load_key(env.work, "sky", 2).versions] == ["plate"]
    st = c.get("/api/cleanup3d").json()
    assert [k["frame"] for k in next(x for x in st["cards"] if x["id"] == "ground")["keys"]] == [3, 9]


def test_master_key_is_not_moved(env):
    c = _setup(env, key_frames=(5,))
    assert c.post("/api/cleanup3d/keys/ground/master", json={"frame": 5, "replace": True}).status_code == 200
    r = c.post("/api/cleanup3d/keys/ground/5/move", json={"to": 6})
    assert r.status_code == 409 and "główny" in r.json()["detail"]
    assert store.load_key(env.work, "ground", 5) is not None
