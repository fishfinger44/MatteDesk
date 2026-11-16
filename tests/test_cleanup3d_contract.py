# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P0 (docs/PLAN_CLEANUP_3D.md): the contract every other package builds on – types round-trip
through their json, the files validate against engine/contract.py, they live in <work>/inpaint/ (never in matte_desk/,
never the manifest), and the API skeleton answers: reads from disk, 501 with the package's name for what comes later."""
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine import contract
from engine.cleanup3d import store
from engine.cleanup3d import types as T


def _camera():
    K = np.array([[2000.0, 0, 960], [0, 2000.0, 540], [0, 0, 1]])
    cam = T.Camera3D(1920, 1080, K, np.array([-0.05, 0.01, 0, 0]), source="test", report={"reproj_px": 0.9})
    for f in (1, 2, 3):
        a = 0.01 * f
        cam.R[f] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        cam.t[f] = np.array([0.1 * f, 0.0, 0.0])
    return cam


def _cards():
    curve = {"type": "bezier", "closed": True, "pts": [{"p": [10, 10], "in": [0, 0], "out": [0, 0], "cusp": True},
                                                       {"p": [90, 10], "in": [0, 0], "out": [0, 0], "cusp": True},
                                                       {"p": [50, 80], "in": [0, 0], "out": [0, 0], "cusp": True}]}
    return [T.Card("sky", "Niebo", kind="sky", order=2, key_mode="every", key_every=12),
            T.Card("sea", "Woda", plane={"n": [0, -1, 0], "d": 1.5}, order=1,
                   extent=[T.Area("a1", "curve", curve=curve)], extent_frame=2),
            T.Card("sand", "Piasek", plane={"n": [0.1, -0.99, 0], "d": 0.8}, order=0, adjust={"dh": 0.02})]


def test_types_round_trip_and_validate():
    cam = _camera()
    d = json.loads(json.dumps(cam.to_json()))
    assert not contract.validate(d)
    cam2 = T.Camera3D.from_json(d)
    assert cam2.frames == [1, 2, 3] and np.allclose(cam2.R[2], cam.R[2]) and np.allclose(cam2.centre(3), cam.centre(3))
    cj = json.loads(json.dumps(T.cards_to_json(_cards())))
    assert not contract.validate(cj)
    cards = T.cards_from_json(cj)
    assert [c.id for c in cards] == ["sand", "sea", "sky"]                # by order: 0 = in front
    assert cards[1].extent[0].curve["closed"] and cards[1].extent_frame == 2 and cards[2].plane is None
    assert cards[0].adjust == {"dh": 0.02, "pitch": 0.0, "roll": 0.0}
    key = T.Key("sea", 2, versions=[{"v": "v001", "ext": "png", "source": "plate"}], active="v001")
    assert not contract.validate(json.loads(json.dumps(key.to_json())))
    assert T.Key.from_json(key.to_json()).active == "v001"


def test_bad_data_is_refused_with_a_reason():
    with pytest.raises(ValueError):
        T.cards_from_json({"cards": [{"id": "a", "kind": "plane"}]})                      # a plane card without a plane
    with pytest.raises(ValueError):
        T.cards_from_json({"cards": [{"id": "a", "kind": "sky"}, {"id": "a", "kind": "sky"}]})   # duplicate id
    with pytest.raises(ValueError):
        T.Card.from_json({"id": "../x", "kind": "sky"})                                   # a name that would leave the folder
    with pytest.raises(ValueError):
        T.Area.from_json({"id": "a", "kind": "curve", "curve": [[0, 0], [1, 1]]})         # not engine/curves.py JSON
    with pytest.raises(ValueError):
        T.Key.from_json({"card": "sea", "frame": 1, "versions": [], "active": "v003"})
    bad = _camera().to_json()
    bad["K"] = [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        T.Camera3D.from_json(bad)


def test_files_live_next_to_matte_desk(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    store.save_camera(work, _camera())
    store.save_cards(work, _cards())
    store.save_key(work, T.Key("sea", 2, versions=[{"v": "v001", "ext": "png"}], active="v001"))
    store.save_key(work, T.Key("sea", 14))
    store.save_areas(work, "sea", 2, [T.Area("b1", "brush", op="remove", png="a_b1.png")])
    root = tmp_path / "shot" / "inpaint"
    assert (root / "camera3d" / "cams.json").is_file() and (root / "cards.json").is_file()
    assert (root / "keys" / "sea" / "k_0002" / "key.json").is_file() and (root / "keys" / "sea" / "k_0002" / "areas.json").is_file()
    assert not any(work.iterdir())                                         # nothing inside matte_desk/
    assert not contract.validate(json.loads((root / "keys" / "sea" / "k_0002" / "areas.json").read_text(encoding="utf-8")))
    assert [k.frame for k in store.list_keys(work, "sea")] == [2, 14] and store.list_keys(work, "sand") == []
    assert store.load_areas(work, "sea", 2)[0].op == "remove" and store.load_areas(work, "sea", 14) == []
    st = store.state(work)
    assert st["done"] == {"camera": True, "cards": True, "keys": True, "project": False} and st["steps"] == list(T.STEPS)
    assert st["stale"] == {"project": False} and st["render"] is None
    assert [c["id"] for c in st["cards"]] == ["sand", "sea", "sky"] and st["cards"][1]["keys"][0] == {"frame": 2, "active": "v001", "versions": 1, "prompt": "", "master": False, "list": [{"v": "v001"}]}
    (root / "cards.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="cards.json"):
        store.load_cards(work)


def _client(work):
    from api.cleanup import make as make_mvp
    from api.cleanup3d import make
    state = {"work_dir": str(work) if work else None, "manifest": {"plates": {}} if work else None}
    shared = dict(STATE=state, objects=lambda: None, plate_size=lambda: (1920, 1080), work_frames=lambda: [1, 2, 3],
                  proxy_scale=lambda: 1.0, comfy_client=lambda: None, long_job=lambda *a, **k: (lambda f: f),
                  refresh_manifest=lambda: None)
    _mvp, exports = make_mvp(**shared)
    router, _ = make(**shared, cleanup=exports)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_reads_and_refuses_with_a_reason(tmp_path):
    assert _client(None).get("/api/cleanup3d").status_code == 400            # no project
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    c = _client(work)
    st = c.get("/api/cleanup3d").json()
    assert st["done"] == {"camera": False, "cards": False, "keys": False, "project": False}
    assert [e["label"] for e in st["engines"]] == ["LaMa", "Qwen-Image-2.1", "FLUX 3 Image Klein + LoRA Restore"] and st["solve"]["available"] is True   # gate off in tests
    assert c.get("/api/cleanup3d/camera").status_code == 404
    r = c.put("/api/cleanup3d/cards", json={"cards": [x.to_json() for x in _cards()][::-1]})
    assert r.status_code == 200 and [x["id"] for x in r.json()["cards"]] == ["sand", "sea", "sky"]   # list order = card order (the saved "order" fields are ignored)
    assert [x["id"] for x in c.get("/api/cleanup3d/cards").json()["cards"]] == ["sand", "sea", "sky"]
    assert c.put("/api/cleanup3d/cards", json={"cards": [{"id": "x", "kind": "plane"}]}).status_code == 409
    assert c.get("/api/cleanup3d/keys/sea/2/areas").status_code == 404        # no such key yet
    store.save_key(work, T.Key("sea", 2))
    assert c.get("/api/cleanup3d/keys/sea/2/areas").json() == {"areas": []}
    assert c.get("/api/cleanup3d/keys", params={"card": "sea"}).json()["keys"][0]["frame"] == 2
    r = c.post("/api/cleanup3d/keys/sea/2/regen", json={"engine": "klein_restore"})
    assert r.status_code == 409 and "Brak klucza" in r.json()["detail"]      # a key without a version: nothing to regenerate
    assert c.post("/api/cleanup3d/keys/sea/2/regen", json={"engine": "minimax"}).status_code == 400
    assert c.post("/api/cleanup3d/keys/nope/2/regen", json={}).status_code == 404
    r = c.post("/api/cleanup3d/camera/solve", json={})
    assert r.status_code == 409 and "beauty" in r.json()["detail"]           # the manifest of this test has no plate
    assert c.get("/api/cleanup3d/view/bogus/1").status_code == 400
    r = c.get("/api/cleanup3d/view/proj/1")
    assert r.status_code == 409 and "Brak kamery 3D" in r.json()["detail"]   # no route answers 501 any more
