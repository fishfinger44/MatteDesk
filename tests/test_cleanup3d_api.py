# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P8 (docs/PLAN_CLEANUP_3D.md): every route of api/cleanup3d.py on a synthetic shot.

A small app built with `make(...)` of api/cleanup.py + api/cleanup3d.py (not app.py): a plate sequence on disk, one
object with a core mask, a fake camera solver (the real one needs COLMAP), the fake ComfyUI of the regeneration tests.
The flow of the UI: solve -> proposed cards -> extent -> keys -> areas -> mark -> regen -> undo -> export / import ->
render (plate „cleaned”) -> previews -> state done / stale; and the refusals (400 / 404 / 409 with a reason).
The `real` test at the end drives the same routes on the sh060 experiment (tools/c3d_from_experiment.py).
"""
import base64
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from engine import cleanup as cu  # noqa: E402
from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import render as RD  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402
from tests.test_cleanup3d_regen import Fake  # noqa: E402
from tests.test_cleanup3d_render import FRAMES, GROUND_Y, H, W, make_camera, obj_alpha, true_bg  # noqa: E402

RED = np.array([0.1, 0.1, 0.9], np.float32)
EXP = Path(r"D:/MD_REAL_PROJECT/Documents\removers_test")
BEATCH = Path(r"D:/MD_REAL_PROJECT/Documents\shots\shots\matte_desk\source\sh060")


class Objects:
    """What the routers ask of the app's object store: the list and the folders."""

    def __init__(self, root, names):
        self.root, self.names = Path(root), list(names)

    def list(self, groups=False):
        return [{"name": n, "kind": "object", "tool": {}} for n in self.names]

    def dir(self, name):
        return self.root / name


class Project:
    def __init__(self):
        self.plates = {}

    def set_plate(self, name, pattern):
        self.plates[name] = pattern


class Env:
    """The synthetic shot + the app. `client` talks to it; `solved` collects what the fake solver was given."""

    def __init__(self, tmp_path, monkeypatch):
        self.shot = Path(tmp_path) / "shot"
        self.work = self.shot / "matte_desk"
        self.work.mkdir(parents=True)
        self.cam = make_camera()
        (self.shot / "plates").mkdir()
        for f in FRAMES:
            p = np.where(obj_alpha(f)[..., None] > 0, RED, true_bg(self.cam, f))
            cv2.imwrite(str(self.shot / "plates" / f"p_{f:04d}.png"), np.clip(np.rint(p * 255), 0, 255).astype(np.uint8))
            core = self.work / "objects" / "obj" / "core"
            core.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(core / f"c_{f:04d}.png"), (obj_alpha(f) * 255).astype(np.uint8))
        cu.save_settings(self.work, {"remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0})
        self.project = Project()
        beauty = str(self.shot / "plates" / "p_####.png").replace("\\", "/")
        self.state = {"work_dir": str(self.work), "manifest": {"plates": {"beauty": beauty}}, "shot_project": self.project}
        self.fake = Fake(Path(tmp_path) / "comfy_out")
        self.solved = []
        self.cfg = {}
        self.scenes = []
        real_scene = RD.Scene

        def counted(*a, **k):
            self.scenes.append(1)
            return real_scene(*a, **k)
        monkeypatch.setattr(RD, "Scene", counted)
        self.client = self.build(self.state)

    def solver(self, work, frames, plate_path_fn, *, cfg, moving_masks=None, single_focal=True, horizon=True,
               progress=None, cancel_check=None):
        """Stands in for COLMAP + CoTracker3: the test's camera, the report keys the UI reads, sparse ground points."""
        if progress:
            progress("COLMAP: wykrywanie cech", 1, 3)
        assert Path(plate_path_fn(frames[0])).is_file()
        self.solved.append({"frames": list(frames), "mask": None if moving_masks is None else moving_masks(frames[0]),
                            "single_focal": single_focal, "horizon": horizon})
        cam = make_camera()
        cam.source = "fake solver"
        cam.report = {"reproj_px": 0.42, "horizon_px": 1.0, "worst_frames": [{"frame": 7, "err_px": 0.9}]}
        store.save_camera(work, cam)
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-4, 6, 400), np.full(400, GROUND_Y), rng.uniform(2, 14, 400)], 1)
        np.save(T.camera_file(work).parent / "points.npy", pts.astype(np.float32))
        return cam

    def build(self, state):
        from api.cleanup import make as make_mvp
        from api.cleanup3d import make
        shared = dict(STATE=state, objects=lambda: Objects(self.work / "objects", ["obj"]), plate_size=lambda: (W, H),
                      work_frames=lambda: list(FRAMES), proxy_scale=lambda: 1.0, comfy_client=lambda: self.fake.client,
                      long_job=lambda *a, **k: (lambda f: f),
                      refresh_manifest=lambda: {"plates": {**state["manifest"]["plates"], **self.project.plates}})
        mvp, exports = make_mvp(**shared)
        router, _ = make(**shared, cleanup=exports, solve_camera=self.solver, load_cfg=lambda: self.cfg)
        app = FastAPI()
        app.include_router(mvp)
        app.include_router(router)
        return TestClient(app)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def _data_url(mask):
    """A brush layer as the editor sends it: RGBA 8-bit, white = area."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", np.dstack([m, m, m, np.full_like(m, 255)]))
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _rect(x0, y0, x1, y1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def _curve(x0, y0, x1, y1):
    pt = lambda x, y: {"p": [x, y], "in": [0, 0], "out": [0, 0], "cusp": True}   # noqa: E731
    return {"type": "bezier", "closed": True, "pts": [pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)]}


def _brush(aid, mask=None, op="add"):
    a = {"id": aid, "kind": "brush", "op": op, "label": "", "png": f"a_{aid}.png"}
    if mask is not None:
        a.update(png_data=_data_url(mask), width=W, height=H)
    return a


def _png(r):
    assert r.status_code == 200 and r.headers["content-type"] == "image/png", r.text[:200]
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None
    return img


def test_no_project_is_400(tmp_path, monkeypatch):
    e = Env(tmp_path, monkeypatch)
    c = e.build({"work_dir": None, "manifest": None})
    for method, url in (("get", "/api/cleanup3d"), ("post", "/api/cleanup3d/cards/propose"), ("get", "/api/cleanup3d/view/proj/1"),
                        ("get", "/api/cleanup3d/keys/a/1/areas/a1/png")):
        assert getattr(c, method)(url).status_code == 400, url
    assert c.post("/api/cleanup3d/render", json={}).status_code == 400
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 400


def test_whole_flow(env):
    c, work = env.client, env.work
    root = env.shot / "inpaint"
    st = c.get("/api/cleanup3d").json()
    assert st["done"] == {"camera": False, "cards": False, "keys": False, "project": False}
    assert st["solve"] == {"available": True, "reason": ""} and all(e["available"] for e in st["engines"])
    assert c.post("/api/cleanup3d/cards/propose").status_code == 409            # no camera yet

    # ---- C2 camera: the fake solver gets the work range, the plate files and the removed object's alpha as the mask
    assert c.post("/api/cleanup3d/camera/solve", json={"moving": ["nope"]}).status_code == 404
    r = c.post("/api/cleanup3d/camera/solve", json={"moving": None, "single_focal": True, "horizon": False})
    assert r.status_code == 200, r.text
    assert r.json()["report"]["reproj_px"] == 0.42 and r.json()["n_frames"] == len(FRAMES) and r.json()["moving"] == ["obj"]
    got = env.solved[-1]
    assert got["frames"] == list(FRAMES) and got["horizon"] is False
    assert got["mask"].shape == (H, W) and np.array_equal(got["mask"] > 0, obj_alpha(FRAMES[0]) > 0)
    assert c.post("/api/cleanup3d/camera/solve", json={"moving": []}).status_code == 200 and env.solved[-1]["mask"] is None
    st = c.get("/api/cleanup3d").json()
    assert st["done"]["camera"] and st["camera"]["report"]["worst_frames"] == [{"frame": 7, "err_px": 0.9}]
    assert st["camera"]["report"]["horizon_px"] == 1.0

    # ---- C3 cards: proposed from the solve, never over the user's own
    r = c.post("/api/cleanup3d/cards/propose")
    assert r.status_code == 200 and r.json()["added"] == ["ground", "sky"], r.text      # no horizon -> no sea card
    g = r.json()["cards"][0]
    assert g["id"] == "ground" and np.allclose(g["plane"]["n"], [0, -1, 0], atol=1e-3) and abs(g["plane"]["d"] - GROUND_Y) < 1e-3
    cards = r.json()["cards"]
    cards[0]["name"], cards[0]["seam_px"] = "piasek", 4.0
    cards[1]["seam_px"] = 4.0
    assert c.put("/api/cleanup3d/cards", json={"cards": cards}).status_code == 200
    r = c.post("/api/cleanup3d/cards/propose")
    assert r.json()["added"] == [] and r.json()["cards"][0]["name"] == "piasek"

    # ---- C3 extent: a curve + a brush layer with pixels; the layer lives in inpaint/cards/<card>/
    ext = "/api/cleanup3d/cards/ground/extent"
    assert c.put("/api/cleanup3d/cards/nope/extent", json={"areas": [], "frame": None}).status_code == 404
    brush = _rect(40, 120, 200, 190)
    body = {"areas": [{"id": "a1", "kind": "curve", "op": "add", "label": "", "curve": _curve(210, 130, 300, 195)},
                      _brush("a2", brush)], "frame": 5}
    assert c.put(ext, json={**body, "frame": None}).status_code == 409               # an extent needs its frame
    r = c.put(ext, json=body)
    assert r.status_code == 200, r.text
    assert [a["id"] for a in r.json()["areas"]] == ["a1", "a2"] and "png_data" not in r.json()["areas"][1]
    assert r.json()["areas"][1]["png"] == "a_a2.png" and r.json()["frame"] == 5
    layer = root / "cards" / "ground" / "a_a2.png"
    assert layer.is_file() and not (work / "cards").exists()
    assert np.array_equal(_png(c.get(f"{ext}/a2/png")), brush)
    assert c.get(f"{ext}/a1/png").status_code == 404 and c.get(f"{ext}/zz/png").status_code == 404   # a curve has no PNG
    before = layer.read_bytes()
    body["areas"][1] = _brush("a2")                                                  # no png_data = keep the file
    assert c.put(ext, json=body).status_code == 200 and layer.read_bytes() == before
    r = c.put(ext, json={"areas": [_brush("a9")], "frame": 5})
    assert r.status_code == 409 and "a_a9.png" in r.json()["detail"]                 # a new layer without pixels
    st = c.get("/api/cleanup3d").json()
    assert st["cards"][0]["extent_frame"] == 5 and len(st["cards"][0]["extent"]) == 2
    sc = RD.Scene(work)                                                              # the render finds the layer
    e5 = sc.geometry.extent_in_frame(sc.card("ground"), 5)
    assert e5[150, 100] == 1.0 and e5[150, 250] == 1.0 and e5[150, 205] == 0.0 and e5[125, 20] == 0.0
    r = c.put(ext, json={"areas": [], "frame": None})                                # back to the whole card
    assert r.status_code == 200 and r.json()["card"]["extent"] == [] and r.json()["card"]["extent_frame"] is None

    # ---- C4 keys: at frames, every N, first version = the plate
    assert c.post("/api/cleanup3d/keys/nope", json={"frames": [1]}).status_code == 404
    assert c.post("/api/cleanup3d/keys/ground", json={}).status_code == 400
    r = c.post("/api/cleanup3d/keys/ground", json={"frames": [1, 9]})
    assert r.status_code == 200 and r.json()["added"] == [1, 9], r.text
    r = c.post("/api/cleanup3d/keys/ground", json={"every": 4})
    assert [k["frame"] for k in r.json()["keys"]] == [1, 5, 9] and r.json()["added"] == [5]
    assert c.post("/api/cleanup3d/keys/sky", json={"every": 4}).json()["added"] == [1, 5, 9]
    assert c.post("/api/cleanup3d/keys/ground", json={"frames": [99]}).status_code == 409    # no such plate frame
    k5 = store.load_key(work, "ground", 5)
    assert k5.versions[0]["source"] == "plate" and k5.versions[0]["ext"] == "png" and k5.versions[0]["bits"] == 8
    plate5 = cv2.imread(str(env.shot / "plates" / "p_0005.png"))
    assert np.array_equal(_png(c.get("/api/cleanup3d/keys/ground/5/image")), plate5)
    assert _png(c.get("/api/cleanup3d/keys/ground/5/image", params={"max_side": 160})).shape[:2] == (100, 160)
    assert c.get("/api/cleanup3d/keys/ground/4/image").status_code == 404
    assert c.get("/api/cleanup3d/keys/ground/5/image", params={"version": "v009"}).status_code == 409
    st = c.get("/api/cleanup3d").json()
    assert st["done"]["keys"] and [k["frame"] for k in st["cards"][0]["keys"]] == [1, 5, 9]

    # ---- C4 areas of a key: curve + brush with pixels, then without
    ar = "/api/cleanup3d/keys/ground/5/areas"
    hole = (obj_alpha(5) > 0).astype(np.uint8) * 255
    hole[:130] = 0                                                                   # the object's part on the ground
    assert hole.any()
    r = c.put(ar, json={"areas": [{"id": "a1", "kind": "curve", "op": "add", "label": "x", "curve": _curve(20, 150, 50, 180)},
                                  _brush("a2", hole)]})
    assert r.status_code == 200 and [a["kind"] for a in r.json()["areas"]] == ["curve", "brush"], r.text
    got = c.get(ar).json()["areas"]
    assert got[0]["label"] == "x" and got[1] == {"id": "a2", "kind": "brush", "op": "add", "label": "", "png": "a_a2.png"}
    assert np.array_equal(_png(c.get(f"{ar}/a2/png")), hole) and c.get(f"{ar}/a1/png").status_code == 404
    kfile = root / "keys" / "ground" / "k_0005" / "a_a2.png"
    before = kfile.read_bytes()
    assert c.put(ar, json={"areas": [got[0], _brush("a2")]}).status_code == 200 and kfile.read_bytes() == before
    assert c.put(ar, json={"areas": [_brush("a7")]}).status_code == 409
    assert c.put(ar, json={"areas": [{"id": "../x", "kind": "curve", "curve": _curve(0, 0, 5, 5)}]}).status_code == 409
    assert c.put(ar, json={"areas": [got[0], got[0]]}).status_code == 409             # duplicate id
    assert [a["id"] for a in c.get(ar).json()["areas"]] == ["a1", "a2"]              # refused writes changed nothing
    m = KS.areas_mask(work, "ground", 5, (H, W))
    assert m[165, 35] == 1.0 and np.array_equal(m[:130] > 0, hole[:130] > 0)
    assert c.get("/api/cleanup3d/keys/ground/4/areas").status_code == 404

    # ---- mark on a frame that is not a key: carried to the two nearest keys of the card it lies on
    mark = {"areas": [{"id": "a1", "kind": "curve", "op": "add", "label": "", "curve": _curve(100, 150, 150, 185)},
                      _brush("a2", _rect(200, 160, 240, 190))], "cards": None}
    r = c.post("/api/cleanup3d/mark/3", json=mark)
    assert r.status_code == 200, r.text
    assert sorted((k["card"], k["frame"]) for k in r.json()["keys"]) == [("ground", 1), ("ground", 5)]
    a5 = c.get(ar).json()["areas"]
    assert [a["id"] for a in a5] == ["a1", "a2", "m3_a1", "m3_a2"] and all(a["kind"] == "curve" for a in a5[2:])
    a1 = c.get("/api/cleanup3d/keys/ground/1/areas").json()["areas"]
    assert [a["id"] for a in a1] == ["m3_a1", "m3_a2"]
    # the carried curve really is the drawn rectangle seen from the key (the geometry, not a copy of the pixels)
    geo = RD.Scene(work).geometry
    want = geo.points_frame_to_key(np.array([[100, 150]], np.float32), 3, store.load_key(work, "ground", 5),
                                   next(x for x in store.load_cards(work) if x.id == "ground"))[0]
    assert np.allclose(a5[2]["curve"]["pts"][0]["p"], want, atol=1e-3) and abs(want[0] - 100) > 2
    assert c.post("/api/cleanup3d/mark/3", json=mark).json()["keys"][0]["added"] == 2
    assert [a["id"] for a in c.get(ar).json()["areas"]][-2:] == ["m3_a1_2", "m3_a2_2"]       # ids stay unique
    assert c.post("/api/cleanup3d/mark/3", json={**mark, "cards": ["nope"]}).status_code == 404
    assert c.post("/api/cleanup3d/mark/3", json={"areas": [_brush("b1")], "cards": None}).status_code == 409
    assert c.post("/api/cleanup3d/mark/3", json={"areas": [], "cards": None}).status_code == 409
    assert c.post("/api/cleanup3d/mark/77", json=mark).status_code == 409                   # not a frame of the camera
    r = c.post("/api/cleanup3d/mark/5", json={**mark, "cards": ["ground"]})                 # on a key frame: that key only
    assert [(k["card"], k["frame"]) for k in r.json()["keys"]] == [("ground", 5)]
    assert c.put(ar, json={"areas": [got[0], _brush("a2")]}).status_code == 200             # back to the two layers

    # ---- regen: a new active version, nothing outside the area changes
    assert c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "minimax"}).status_code == 400
    assert c.post("/api/cleanup3d/keys/ground/4/regen", json={"engine": "lama"}).status_code == 404
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "lama"})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == "v002" and r.json()["key"]["active"] == "v002" and r.json()["engine"] == "lama"
    assert r.json()["key"]["versions"][-1]["source"] == "regen:lama" and len(env.fake.lama) >= 1
    v2 = _png(c.get("/api/cleanup3d/keys/ground/5/image"))
    changed = np.abs(v2.astype(int) - plate5.astype(int)).max(-1) > 0
    grown = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    assert changed.any() and not (changed & ~grown).any()
    assert v2[(hole > 0) & (m > 0.5)][:, 2].mean() < 0.8 * 255 * RED[2]                    # the red object is gone
    assert np.array_equal(_png(c.get("/api/cleanup3d/keys/ground/5/image", params={"version": "v001"})), plate5)

    # ---- undo / export / import
    r = c.post("/api/cleanup3d/keys/ground/5/undo")
    assert r.status_code == 200 and r.json()["version"] == "v001"
    r = c.post("/api/cleanup3d/keys/ground/5/undo")
    assert r.status_code == 409 and "wcześniejszej" in r.json()["detail"]
    assert c.post("/api/cleanup3d/keys/ground/5/export", json={"format": "tiff"}).status_code == 400
    r = c.post("/api/cleanup3d/keys/ground/5/export", json={"format": "png16"})
    assert r.status_code == 200, r.text
    path = Path(r.json()["path"])
    assert path.is_file() and path.parent == root / "keys_export" and cv2.imread(str(path), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    painted = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    painted[150:180, 20:50] = 0
    cv2.imwrite(str(path), painted)
    r = c.post("/api/cleanup3d/keys/ground/5/import", json={"path": str(path)})
    assert r.status_code == 200 and r.json()["version"] == "v003" and r.json()["key"]["versions"][-1]["source"].startswith("import:")
    small = env.shot / "small.png"
    cv2.imwrite(str(small), np.zeros((H // 2, W // 2, 3), np.uint8))
    r = c.post("/api/cleanup3d/keys/ground/5/import", json={"path": str(small)})
    assert r.status_code == 409 and "rozmiar" in r.json()["detail"] and f"{W}×{H}" in r.json()["detail"]
    assert c.post("/api/cleanup3d/keys/ground/5/import", json={"path": str(env.shot / "nope.png")}).status_code == 409
    assert store.load_key(work, "ground", 5).active == "v003"
    assert c.post("/api/cleanup3d/keys/ground/5/undo").json()["version"] == "v002"          # the regenerated one projects

    # ---- previews: PNG, cached per state (one Scene for all of them)
    n0 = len(env.scenes)
    proj = c.get("/api/cleanup3d/view/proj/4")
    cards_v = c.get("/api/cleanup3d/view/cards/4")
    assert _png(proj).shape == (H, W, 3) and _png(cards_v).shape == (H, W, 3)
    kv = c.get("/api/cleanup3d/view/key/4", params={"card": "ground"})                       # not a key frame: the nearest key
    assert _png(kv).shape == (H, W, 3) and kv.headers["x-c3d-key-frame"] == "5"
    assert np.array_equal(_png(kv), v2)
    assert _png(c.get("/api/cleanup3d/view/proj/4", params={"max_side": 160})).shape[:2] == (100, 160)
    assert c.get("/api/cleanup3d/view/proj/4").content == proj.content and len(env.scenes) == n0 + 1
    assert c.get("/api/cleanup3d/view/key/4").status_code == 400
    assert c.get("/api/cleanup3d/view/key/4", params={"card": "nope"}).status_code == 404
    assert c.get("/api/cleanup3d/view/bogus/4").status_code == 400
    assert c.get("/api/cleanup3d/view/proj/77").status_code == 409
    # the projection shows the keys: far from the removed object it is the plate itself
    p4 = cv2.imread(str(env.shot / "plates" / "p_0004.png")).astype(int)
    far = cv2.dilate((obj_alpha(4) > 0).astype(np.uint8), np.ones((61, 61), np.uint8)) == 0
    assert np.abs(_png(proj).astype(int) - p4)[far].mean() < 6.0

    # ---- C5 render: a version of Cleanup, the plate „cleaned”, the state
    assert c.post("/api/cleanup3d/render", json={"motion_blur": "bogus"}).status_code == 400
    r = c.post("/api/cleanup3d/render", json={"frames": [4, 5, 6], "motion_blur": "off", "feather_px": 2})
    assert r.status_code == 200, r.text
    rj = r.json()
    out = root / rj["version"]
    assert rj["version"] == "v001" and rj["n_frames"] == 3 and rj["plate"] == "cleaned" and rj["counts"]["klucz"] > 0
    assert (out / "comp" / "c_0005.png").is_file() and cu.versions(work)[0]["version"] == "v001"
    assert env.project.plates == {"cleaned": str(out / "comp" / "c_####.png").replace("\\", "/")}
    assert rj["plates"]["cleaned"] == env.project.plates["cleaned"]
    import json
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["settings"]["soft_px"] == 2.0 and run["c3d"]["settings"]["motion_blur"] == "off"
    comp = cv2.imread(str(out / "comp" / "c_0004.png")).astype(np.float32) / 255.0
    inside = cv2.erode((obj_alpha(4) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bg4 = true_bg(env.cam, 4)                # the keys are plate frames (the object still in them further right) –
    assert np.abs(comp - bg4)[inside].mean() < 0.5 * np.abs(RED - bg4)[inside].mean()   # closer to the background than red
    st = c.get("/api/cleanup3d").json()
    assert st["done"]["project"] is True and st["stale"] == {"project": False} and st["render"]["version"] == "v001"

    # ---- stale: a change of the keys / cards after the render (and back = not stale)
    n1 = len(env.scenes)
    assert c.post("/api/cleanup3d/keys/ground/5/undo").status_code == 200
    st = c.get("/api/cleanup3d").json()
    assert st["done"]["project"] is True and st["stale"] == {"project": True} and st["render"]["stale"] is True
    assert c.get("/api/cleanup3d/view/proj/4").content != proj.content and len(env.scenes) == n1 + 1   # a new Scene
    r = c.post("/api/cleanup3d/keys/ground", json={"frames": [3]})
    assert r.json()["added"] == [3]

    # ---- delete: from disk for good (user decision 2026-11-13), nothing in a trash folder
    r = c.delete("/api/cleanup3d/keys/ground/3")
    assert r.status_code == 200 and r.json()["freed_bytes"] > 0 and "trash" not in r.json()
    assert not (root / "keys" / "ground" / "k_0003").exists() and not (root / "trash").exists()
    assert c.delete("/api/cleanup3d/keys/ground/3").status_code == 404
    assert [k["frame"] for k in c.get("/api/cleanup3d/keys", params={"card": "ground"}).json()["keys"]] == [1, 5, 9]
    assert not any(p.name.startswith("shot_manifest") for p in env.shot.rglob("*"))          # nothing like a manifest written


def test_regen_qwen21_edit_through_the_route(env, monkeypatch):
    """Qwen-Image-2.1 (2026-11-10): the bar's description reaches the instruction and stays with the new version."""
    from engine.cleanup3d import regen as RG
    monkeypatch.setattr(RG, "QWEN_PX", 64 * 96)                    # small crops for the fake model
    c = env.client
    assert c.post("/api/cleanup3d/camera/solve", json={"moving": []}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    assert c.post("/api/cleanup3d/keys/ground", json={"frames": [5]}).status_code == 200
    assert c.put("/api/cleanup3d/keys/ground/5/areas", json={"areas": [_brush("a1", _rect(10, 150, 40, 180))]}).status_code == 200
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "qwen21_edit", "prompt": "wet sand"})
    assert r.status_code == 200, r.text
    v = r.json()["key"]["versions"][-1]
    assert r.json()["engine"] == "qwen21_edit" and v["source"] == "regen:qwen21_edit" and v["prompt"] == "wet sand"
    g, crop = env.fake.qwen[-1]
    assert g["8"]["inputs"]["prompt"] == RG.qwen_prompt("wet sand") and crop.shape[0] % 32 == 0 and crop.shape[1] % 32 == 0


def test_key_steps_versions_base_and_whole_patch(env):
    """Key steps (user 2026-11-10): the version list in the state, make active, delete a version for good (not the
    active one, not the last), regen from a chosen base, „cała łata” with the FLUX 3 Image Klein strength."""
    c = env.client
    assert c.post("/api/cleanup3d/camera/solve", json={"moving": []}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    assert c.post("/api/cleanup3d/keys/ground", json={"frames": [5]}).status_code == 200
    key = store.load_key(env.work, "ground", 5)
    plate = KS.read_key_image(env.work, key)
    fill = plate.copy()
    obj = obj_alpha(5) > 0
    fill[obj] = cv2.GaussianBlur(true_bg(env.cam, 5), (0, 0), 3)[obj]     # a model's fill where the object was
    KS.add_version(env.work, "ground", 5, fill, "plate+fill:v001", "png")
    st = c.get("/api/cleanup3d").json()
    k = next(x for x in next(cd for cd in st["cards"] if cd["id"] == "ground")["keys"] if x["frame"] == 5)
    assert [(v["v"], v["source"]) for v in k["list"]] == [("v001", "plate"), ("v002", "plate+fill:v001")]
    assert all(v.get("date") for v in k["list"])

    # „cała łata” from v002 with FLUX 3 Image Klein + LoRA Restore at strength 0.7
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "klein_restore", "area": "patch", "strength": 0.7})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["area"] == "patch" and j["base"] == "v002" and j["strength"] == 0.7 and j["version"] == "v003"
    v = j["key"]["versions"][-1]
    assert v["base"] == "v002" and v["area"] == "patch" and v["strength"] == 0.7
    g, tile = env.fake.tiles[-1]
    assert g["12"]["inputs"]["denoise"] == 0.7 and g[g["12"]["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    k3 = store.load_key(env.work, "ground", 5)
    v2, v3 = KS.read_key_image(env.work, k3, "v002"), KS.read_key_image(env.work, k3, "v003")
    changed = np.abs(v3 - v2).max(axis=2) > 0
    near = cv2.dilate(obj.astype(np.uint8), np.ones((81, 81), np.uint8)) > 0
    assert changed.any() and not changed[~near].any()             # only the patch (+ its band) changed
    assert np.array_equal(v3[~near], plate[~near])                 # the plate outside, bit for bit
    # bad requests
    assert c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "lama", "area": "all"}).status_code == 400
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "klein_restore", "strength": 0.1})
    assert r.status_code == 409 and "Siła" in r.json()["detail"]
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "lama", "area": "patch", "base": "v001"})
    assert r.status_code == 409 and "nie ma łaty" in r.json()["detail"]   # the plate itself has no patch
    r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": "lama", "base": "v077"})
    assert r.status_code == 409 and "v077" in r.json()["detail"]

    # make active / delete a version
    r = c.put("/api/cleanup3d/keys/ground/5/active", json={"version": "v001"})
    assert r.status_code == 200 and r.json()["version"] == "v001"
    assert c.put("/api/cleanup3d/keys/ground/5/active", json={"version": "v099"}).status_code == 409
    assert c.put("/api/cleanup3d/keys/ground/4/active", json={"version": "v001"}).status_code == 404
    r = c.delete("/api/cleanup3d/keys/ground/5/versions/v001")
    assert r.status_code == 409 and "aktywna" in r.json()["detail"]
    r = c.delete("/api/cleanup3d/keys/ground/5/versions/v003")
    assert r.status_code == 200, r.text
    kd = T.key_dir(env.work, "ground", 5)
    assert r.json()["freed_bytes"] > 0 and not list(kd.glob("v003.*")) and not (T.root_dir(env.work) / "trash").exists()
    assert [x["v"] for x in r.json()["key"]["versions"]] == ["v001", "v002"]
    assert c.delete("/api/cleanup3d/keys/ground/5/versions/v003").status_code == 409
    assert c.delete("/api/cleanup3d/keys/ground/5/versions/v002").status_code == 200
    r = c.delete("/api/cleanup3d/keys/ground/5/versions/v001")
    assert r.status_code == 409                                    # active and the last one: delete the key instead
    assert not any(p.name.startswith("shot_manifest") for p in env.shot.rglob("*"))


def test_missing_model_is_409_with_the_gates_reason(env, monkeypatch):
    """Gate on, nothing installed: the state says why, regen and solve refuse before they start anything."""
    from engine import installer
    c = env.client
    assert c.post("/api/cleanup3d/camera/solve", json={"moving": []}).status_code == 200      # gate off (conftest)
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    assert c.post("/api/cleanup3d/keys/ground", json={"frames": [5]}).status_code == 200
    assert c.put("/api/cleanup3d/keys/ground/5/areas", json={"areas": [_brush("a1", _rect(10, 150, 40, 180))]}).status_code == 200
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    monkeypatch.setattr(installer, "COLMAP_DIR", env.shot / "no_colmap")
    env.cfg = {"models_dir": str(env.shot / "models")}
    st = c.get("/api/cleanup3d").json()
    assert st["solve"]["available"] is False and "COLMAP" in st["solve"]["reason"]
    assert [e["available"] for e in st["engines"]] == [False, False, False] and all(e["reason"] for e in st["engines"])
    n = len(env.solved)
    r = c.post("/api/cleanup3d/camera/solve", json={})
    assert r.status_code == 409 and r.json()["detail"] == st["solve"]["reason"] and len(env.solved) == n
    for eng in st["engines"]:
        r = c.post("/api/cleanup3d/keys/ground/5/regen", json={"engine": eng["id"]})
        assert r.status_code == 409 and r.json()["detail"] == eng["reason"], r.text
    assert store.load_key(env.work, "ground", 5).active == "v001" and not env.fake.lama


def test_render_refusals(env):
    c = env.client
    r = c.post("/api/cleanup3d/render", json={"frames": [4]})
    assert r.status_code == 409 and "Brak kamery 3D" in r.json()["detail"]
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    r = c.post("/api/cleanup3d/render", json={"frames": [4]})
    assert r.status_code == 409 and "Brak kluczy" in r.json()["detail"]
    cu.save_settings(env.work, {"remove": []})
    r = c.post("/api/cleanup3d/render", json={"frames": [4]})
    assert r.status_code == 409 and "Wybierz obiekt" in r.json()["detail"]                    # the MVP's own refusal
    assert c.get("/api/cleanup3d").json()["done"]["project"] is False and env.project.plates == {}


def test_key_modes_auto_and_master(env):
    """P13: key mode „automat” (keys where the projection error grows) and „jeden klucz” (the master key)."""
    c, work = env.client, env.work
    assert c.post("/api/cleanup3d/keys/ground/auto", json={}).status_code == 404               # no card yet
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    # automat: the mode and its settings land on the card, the keys are created from the plate frames
    r = c.post("/api/cleanup3d/keys/ground/auto", json={"threshold": 0.05, "min_gap": 2, "max_gap": 4})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["frames"][0] == FRAMES[0] and j["frames"][-1] == FRAMES[-1] and max(np.diff(j["frames"])) <= 4
    assert j["added_keys"] == j["frames"] and set(j["errors"]) == {str(f) for f in FRAMES}
    card = next(x for x in store.load_cards(work) if x.id == "ground")
    assert card.key_mode == "auto" and card.key_auto["max_gap"] == 4 and card.key_auto["threshold"] == 0.05
    assert [k.frame for k in store.list_keys(work, "ground")] == j["frames"]
    r = c.post("/api/cleanup3d/keys/ground/auto", json={"create": False})                     # saved settings reused
    assert r.status_code == 200 and r.json()["settings"]["max_gap"] == 4 and r.json()["added_keys"] == []
    # jeden klucz: refused while the card has other keys, unless they are deleted for good
    r = c.post("/api/cleanup3d/keys/ground/master", json={"frame": 5})
    assert r.status_code == 409 and "trwałe usunięcie" in r.json()["detail"]
    assert c.post("/api/cleanup3d/keys/ground/master", json={"frame": 99}).status_code == 409   # not a camera frame
    r = c.post("/api/cleanup3d/keys/ground/master", json={"frame": 5, "replace": True})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["key_frame"] == 5 and j["canvas"]["w"] >= W and j["canvas"]["h"] >= H and len(j["replaced"]) >= 2
    assert j["seen_share"] > 0.95 and len(j["frames"]) > 1
    keys = store.list_keys(work, "ground")
    assert [k.frame for k in keys] == [5] and keys[0].canvas and keys[0].versions[-1]["source"] == "master"
    assert next(x for x in store.load_cards(work) if x.id == "ground").key_mode == "one"
    assert j["freed_bytes"] > 0 and not (T.root_dir(work) / "trash").exists()
    assert all(not T.key_dir(work, "ground", f).exists() for f in j["replaced"] if f != 5)
    # a rebuild keeps the canvas (a new version), the master key projects like any key
    r = c.post("/api/cleanup3d/keys/ground/master", json={"frame": 5})
    assert r.status_code == 200 and len(store.load_key(work, "ground", 5).versions) == 2 and r.json()["replaced"] == []
    assert c.post("/api/cleanup3d/keys/sky/master", json={}).status_code == 200                # middle of the range
    assert store.load_key(work, "sky", FRAMES[len(FRAMES) // 2]).canvas
    r = c.get("/api/cleanup3d/view/proj/2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    r = c.get("/api/cleanup3d/view/key/2", params={"card": "ground", "max_side": 1000})
    assert r.status_code == 200 and r.headers["X-C3D-Key-Frame"] == "5"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == keys[0].canvas["w"] and img.shape[0] == keys[0].canvas["h"]


def test_routes_are_wired_in_app():
    root = Path(__file__).resolve().parent.parent
    src = (root / "app.py").read_text(encoding="utf-8")
    assert "cleanup=_cleanup_exports" in src and "_cleanup_exports = _x" in src
    api = (root / "api" / "cleanup3d.py").read_text(encoding="utf-8")
    assert "501" not in api and "import app" not in api and "from app " not in api


def test_cli_verbs(env):
    """tools/md.py `c3d …` against the same app (exit codes: 0 ok, 1 error, 2 refused = HTTP 409)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import md

    class Http:
        def request(self, method, path, body=None, timeout=None):
            r = env.client.request(method, path, json=body)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, None

    def run(*argv):
        lines = []
        return md.main(["c3d", *argv], http=Http(), out=lines.append), " | ".join(lines)

    code, txt = run("status")
    assert code == 0 and "○ camera" in txt and "solve kamery: dostępny" in txt
    code, txt = run("solve", "--moving", "", "--no-horizon")
    assert code == 0 and "reprojekcja 0.42 px" in txt and "F7 (0.9 px)" in txt and env.solved[-1]["horizon"] is False
    assert env.solved[-1]["mask"] is None
    code, txt = run("propose")
    assert code == 0 and "dodane: ground, sky" in txt
    code, txt = run("keys", "ground", "--frames", "1,9")
    assert code == 0 and "nowe: F1, F9" in txt
    code, txt = run("keys", "ground")
    assert code == 0 and "F9: aktywna v001" in txt and "(plate)" in txt
    code, txt = run("status")
    assert "✔ camera" in txt and "✔ keys" in txt and "○ project" in txt and "karta ground (plane, cała karta): klucze F1:v001, F9:v001" in txt
    assert run("regen", "ground", "1")[0] == 2                                # no area marked: refused (409)
    assert run("regen", "ground")[0] == 1 and run("keys")[0] == 1             # arguments missing
    code, txt = run("key-export", "ground", "1", "--format", "png16")
    assert code == 0 and Path(txt.split(": ", 1)[1]).is_file()
    code, txt = run("key-import", "ground", "1", txt.split(": ", 1)[1])
    assert code == 0 and "v002" in txt
    assert "v001" in run("key-undo", "ground", "1")[1] and run("key-undo", "ground", "1")[0] == 2
    code, txt = run("render", "--frames", "4-5", "--blur", "off")
    assert code == 0 and "Projekcja: v001 · 2 kl." in txt and "cleaned" in txt
    assert "✔ project" in run("status")[1]
    assert run("key-delete", "ground", "9")[0] == 1 and store.load_key(env.work, "ground", 9) is not None    # needs --yes
    assert run("key-delete", "ground", "9", "--yes")[0] == 0 and store.load_key(env.work, "ground", 9) is None
    assert "↻ project" in run("status")[1]
    assert run("keys", "nope")[0] == 0 and run("keys", "nope", "--every", "4")[0] == 1                      # 404 = error


# ----------------------------------------------------------------------------------------------- real: sh060
@pytest.mark.real
def test_real_sh060_through_the_api(tmp_path):
    """The experiment's camera / cards / keys (tools/c3d_from_experiment.py) in a temp work dir, driven through the
    routes: state, previews, a render of frames 20-22. Prints the seconds."""
    if not ((EXP / "outputs" / "sfm8" / "cams.npz").is_file() and (BEATCH / "sh060_0001.png").is_file()):
        pytest.skip(f"the sh060 experiment is not on this machine ({EXP})")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import c3d_from_experiment as CV
    from api.cleanup import make as make_mvp
    from api.cleanup3d import make
    shot = tmp_path / "sh060"
    work = shot / "matte_desk"
    work.mkdir(parents=True)
    CV.convert(work, EXP, log=lambda *_: None)
    frames = [20, 21, 22]
    w, h = store.load_camera(work).width, store.load_camera(work).height
    core = work / "objects" / "woman" / "core"
    core.mkdir(parents=True)
    for f in frames:
        m = cv2.imread(str(EXP / "outputs" / "gpu_d" / "mask" / f"m_{f:04d}.png"), 0)
        cv2.imwrite(str(core / f"c_{f:04d}.png"), cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
    cu.save_settings(work, {"remove": ["woman"], "grow_px": 40.0, "soft_px": 40.0, "holdout": False, "fill": "lama"})
    proj = Project()
    state = {"work_dir": str(work), "manifest": {"plates": {"beauty": str(BEATCH / "sh060_####.png").replace("\\", "/")}},
             "shot_project": proj}
    shared = dict(STATE=state, objects=lambda: Objects(work / "objects", ["woman"]), plate_size=lambda: (w, h),
                  work_frames=lambda: frames, proxy_scale=lambda: 0.5, comfy_client=lambda: None,
                  long_job=lambda *a, **k: (lambda f: f),
                  refresh_manifest=lambda: {"plates": {**state["manifest"]["plates"], **proj.plates}})
    mvp, exports = make_mvp(**shared)
    router, _ = make(**shared, cleanup=exports)
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    rep = {"size": [w, h]}
    t = time.time()
    st = c.get("/api/cleanup3d").json()
    rep["state_s"] = round(time.time() - t, 2)
    assert st["done"]["camera"] and st["done"]["cards"] and st["done"]["keys"] and not st["done"]["project"]
    assert [x["id"] for x in st["cards"]] == ["sand", "sea", "sky"] and st["camera"]["n_frames"] == 57
    for name, url, params in (("view_proj", "/api/cleanup3d/view/proj/20", {}), ("view_cards", "/api/cleanup3d/view/cards/20", {}),
                              ("view_key", "/api/cleanup3d/view/key/20", {"card": "sand"}),
                              ("view_proj_next_frame", "/api/cleanup3d/view/proj/21", {}),
                              ("view_proj_cached", "/api/cleanup3d/view/proj/20", {})):
        t = time.time()
        img = _png(c.get(url, params=params))
        rep[f"{name}_s"] = round(time.time() - t, 2)
        assert max(img.shape[:2]) == max(w, h) // 2 and img.std() > 5
    t = time.time()
    r = c.post("/api/cleanup3d/render", json={"frames": frames, "motion_blur": "auto"})
    rep["render_s"] = round(time.time() - t, 1)
    assert r.status_code == 200, r.text
    rj = r.json()
    rep.update(render_s_per_frame=round(rep["render_s"] / len(frames), 1), counts=rj["counts"], backend=rj["backend"],
               device=rj["device"], version=rj["version"])
    out = Path(rj["dir"])
    assert all((out / "comp" / f"c_{f:04d}.png").is_file() for f in frames) and "cleaned" in proj.plates
    assert rj["counts"]["klucz"] > 0.97 * sum(rj["counts"].values())
    st = c.get("/api/cleanup3d").json()
    assert st["done"]["project"] and st["stale"] == {"project": False}
    print("\nC3D API REAL " + str(rep))
