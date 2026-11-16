# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D depth card (package P18: engine/cleanup3d/geom_gpu.py `depth_cast`, geom.py CardGeometry, depthcard.py)
on a synthetic scene: a slanted wall with a box in front of it, a camera that slides sideways (parallax), distortion.

Expected values are analytic: a ray of the target frame cast onto the box / the wall (slab test + plane), the hit seen
from the reference frame (nothing nearer on the way) projected with cv2.projectPoints. The depth card gets the
reference frame's depth rendered from the same analytic scene."""

import os
import sys

import cv2
import numpy as np
import pytest

from engine.cleanup3d import depthcard as DC
from engine.cleanup3d import geom_gpu as KER
from engine.cleanup3d import render as RD
from engine.cleanup3d.geom import CardGeometry, GeomGpuWorker
from engine.cleanup3d.types import Camera3D, Card, Key, cards_from_json, cards_to_json, parse_depth

W, H = 320, 200
KM = np.array([[300.0, 0, 160.4], [0, 296.0, 99.6], [0, 0, 1]])
DIST = np.array([-0.08, 0.02, 0.0007, -0.0005])
FRAMES = [1, 2, 3, 4, 5]
REF = 3
WALL_N = np.array([0.35, 0.0, -1.0]) / np.linalg.norm([0.35, 0.0, -1.0])   # toward the cameras
WALL_P = np.array([0.0, 0.0, 22.0])
BOX = (np.array([-1.6, -1.2, 8.0]), np.array([1.4, 2.2, 10.5]))
FAKE_ENV = {**os.environ, "MD_GEOM_GPU_FAKE": "1"}


def _rot(yaw, pitch=0.0):
    a, b = np.radians([yaw, pitch])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def make_camera(dist=DIST):
    R, t = {}, {}
    for i, f in enumerate(FRAMES):
        C = np.array([0.9 * (i - 2), -0.1 * (i - 2), 0.3 * i])     # slides sideways: strong parallax box / wall
        Rf = _rot(-2.5 * (i - 2), 1.0)
        R[f], t[f] = Rf, -Rf @ C
    return Camera3D(W, H, KM.copy(), np.asarray(dist, float), R, t, "synthetic")


def cast(C, dirs):
    """Nearest hit of rays C + s dirs (s > 0) with the box and the wall -> (s, label 1 box / 2 wall / 0 none)."""
    lo, hi = BOX
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - C) / dirs
        t2 = (hi - C) / dirs
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    sbox = np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        swall = (WALL_N @ (WALL_P - C)) / (dirs @ WALL_N)
    swall = np.where(swall > 1e-6, swall, np.inf)
    s = np.minimum(sbox, swall)
    lab = np.where(np.isinf(s), 0, np.where(sbox <= swall, 1, 2))
    return s, lab


def pixel_rays(cam, f, px):
    u = cv2.undistortPointsIter(np.asarray(px, np.float64).reshape(-1, 1, 2), cam.K, cam.dist, None, None,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)).reshape(-1, 2)
    return np.c_[u, np.ones(len(u))] @ cam.R[f]          # camera-z = 1: s along it = camera-z depth


def grid(w=W, h=H):
    ys, xs = np.mgrid[0:h, 0:w]
    return np.c_[xs.ravel(), ys.ravel()].astype(np.float64)


def depth_of(cam, f, conv="z"):
    """The frame's depth map (camera z), or relative disparity 2 / z for conv disparity."""
    s, lab = cast(cam.centre(f), pixel_rays(cam, f, grid()))
    z = np.where(lab > 0, s, 1e3).reshape(H, W)
    return z if conv == "z" else 2.0 / z


def project_world(cam, f, X):
    rv, _ = cv2.Rodrigues(cam.R[f])
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rv, cam.t[f], cam.K, cam.dist)
    return p.reshape(-1, 2)


def depth_card(conv="z", scale=1.0, offset=0.0, **kw):
    return Card("relief", "relief", "depth", depth=parse_depth({"frame": REF, "conv": conv, "scale": scale, "offset": offset, **kw}),
                seam_px=1.0)


def analytic(cam, f, key_frame=REF):
    """Per pixel of frame f: expected key pixel (NaN where the ref does not see the point), and a 'near an edge' mask."""
    C = cam.centre(f)
    s, lab = cast(C, pixel_rays(cam, f, grid()))
    X = C + s[:, None] * pixel_rays(cam, f, grid())
    Cr = cam.centre(key_frame)
    v = X - Cr
    dist = np.linalg.norm(v, axis=1)
    sr, _ = cast(Cr, v / np.where(dist > 0, dist, 1)[:, None])
    vis = (lab > 0) & (np.abs(sr - dist) < 1e-6 * np.maximum(dist, 1))
    exp = np.full((len(X), 2), np.nan)
    p = project_world(cam, key_frame, X[lab > 0])
    exp[lab > 0] = p
    inside = (exp[:, 0] >= 1) & (exp[:, 1] >= 1) & (exp[:, 0] <= W - 2) & (exp[:, 1] <= H - 2)
    border = ~inside & np.isfinite(exp[:, 0]) & (exp[:, 0] > -2) & (exp[:, 1] > -2) & (exp[:, 0] < W + 1) & (exp[:, 1] < H + 1)
    vis &= inside
    exp[~vis] = np.nan
    lab2 = (lab * 2 + vis).reshape(H, W).astype(np.uint8)
    edge = cv2.dilate((cv2.Laplacian(lab2.astype(np.float32), cv2.CV_32F) != 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    near = edge.astype(bool) | border.reshape(H, W)
    return exp.reshape(H, W, 2), near


@pytest.fixture(scope="module")
def scene():
    cam = make_camera()
    return cam, depth_of(cam, REF)


# ------------------------------------------------------------------------------------------- correspondence
@pytest.mark.parametrize("f", [1, 2, 5])
def test_frame_to_key_equals_the_analytic_correspondence(scene, f):
    """Through the depth card a pixel of frame f lands where the reference frame sees the same world point
    (< 0.5 px), and where the reference does not see it (behind the box, outside the frame) it is MISS."""
    cam, depth = scene
    card = depth_card()
    geo = CardGeometry(cam, [card], depths={card.id: depth})
    mx, my = geo.frame_to_key(f, Key(card.id, REF), card)
    exp, near = analytic(cam, f)
    got = mx > KER.MISS / 2
    vis = np.isfinite(exp[..., 0])
    chk = vis & ~near
    assert chk.sum() > 0.5 * W * H
    assert got[chk].mean() > 0.999, f"hits {got[chk].mean():.4f}"
    err = np.hypot(mx - exp[..., 0], my - exp[..., 1])[chk & got]
    assert np.percentile(err, 99.9) < 0.5 and err.max() < 1.0, (np.percentile(err, 99.9), err.max())
    occl = ~vis & ~near
    assert occl.sum() > 200                                    # the box hides part of the wall from the reference
    assert (~got[occl]).mean() > 0.995, f"MISS where unseen {(~got[occl]).mean():.4f}"
    geo.close()


def test_key_frame_itself_is_the_identity(scene):
    cam, depth = scene
    card = depth_card()
    geo = CardGeometry(cam, [card], depths={card.id: depth})
    mx, my = geo.frame_to_key(REF, Key(card.id, REF), card)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ok = mx > KER.MISS / 2
    assert ok.mean() > 0.97                                     # all but the box's one-pixel edge quads
    assert np.abs(mx - gx)[ok].max() < 0.05 and np.abs(my - gy)[ok].max() < 0.05
    geo.close()


def test_disparity_conversion_and_points_api(scene):
    """conv disparity (1/z = scale d + offset) gives the same surface; points_frame_to_key = the maps."""
    cam, _ = scene
    disp = depth_of(cam, REF, "disparity")                       # 2 / z
    card = depth_card("disparity", scale=0.5, offset=0.0)        # 1/z = 0.5 * 2/z
    geo = CardGeometry(cam, [card], depths={card.id: disp})
    mx, my = geo.frame_to_key(1, Key(card.id, REF), card)
    exp, near = analytic(cam, 1)
    chk = np.isfinite(exp[..., 0]) & ~near & (mx > KER.MISS / 2)
    assert np.percentile(np.hypot(mx - exp[..., 0], my - exp[..., 1])[chk], 99) < 0.5
    pts = np.array([[40.0, 60.0], [160.0, 100.0], [300.0, 150.0]])
    pk = geo.points_frame_to_key(pts, 1, Key(card.id, REF), card)
    for (x, y), (a, b) in zip(pts.astype(int), pk):
        if mx[y, x] > KER.MISS / 2:
            assert abs(a - mx[y, x]) < 1e-3 and abs(b - my[y, x]) < 1e-3
        else:
            assert np.isnan(a)
    geo.close()


def test_depth_warp_and_weights(scene):
    cam, depth = scene
    card = depth_card()
    sky = Card("sky", "niebo", "sky", order=1, seam_px=1.0)
    geo = CardGeometry(cam, [card, sky], depths={card.id: depth})
    mx, my, z = geo.depth_warp(1, card)
    k = geo.frame_to_key(1, Key(card.id, REF), card)
    assert np.allclose(mx, k[0]) and np.allclose(my, k[1])
    s, lab = cast(cam.centre(1), pixel_rays(cam, 1, grid()))
    exp, near = analytic(cam, 1)
    ok = ((z > 0) & np.isfinite(exp[..., 0]) & ~near).ravel()
    assert ok.mean() > 0.4
    assert (np.abs(z.ravel()[ok] - s[ok]) / s[ok]).max() < 1e-3             # camera-z depth of the hit
    ws = geo.card_weights(1)
    hit = (mx > KER.MISS / 2)
    assert np.allclose(ws["relief"] + ws["sky"], 1.0, atol=1e-5)           # the sky takes the holes
    assert ws["relief"][hit].mean() > 0.95 and ws["sky"][~hit].mean() > 0.95
    geo.close()


# ------------------------------------------------------------------------------------------- GPU worker (fake)
def test_gpu_fake_worker_equals_cpu(scene):
    cam, depth = scene
    card = depth_card()
    cards = [card, Card("sky", "niebo", "sky", order=1)]
    cpu = CardGeometry(cam, cards, depths={card.id: depth})
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    gpu = CardGeometry(cam, cards, device="gpu", gpu=w, depths={card.id: depth})
    try:
        key = Key(card.id, REF)
        for a, b in zip(cpu.frame_to_key(1, key, card), gpu.frame_to_key(1, key, card)):
            assert np.array_equal(a > KER.MISS / 2, b > KER.MISS / 2)
            ok = a > KER.MISS / 2
            assert np.abs(a[ok] - b[ok]).max() < 1e-3
        for a, b in zip(cpu.key_to_frame(key, card, 5), gpu.key_to_frame(key, card, 5)):
            ok = a > KER.MISS / 2
            assert np.array_equal(ok, b > KER.MISS / 2) and np.abs(a[ok] - b[ok]).max() < 1e-3
        for a, b in zip(cpu.depth_warp(2, card), gpu.depth_warp(2, card)):
            assert np.abs(a - b).max() < 1e-3
        wa, wb = cpu.card_weights(4), gpu.card_weights(4)
        for c in wa:
            assert np.abs(wa[c] - wb[c]).max() < 1e-3
        assert np.abs(cpu.coverage(4, key, card) - gpu.coverage(4, key, card)).max() < 1e-3
        assert w.device == "fake"
    finally:
        gpu.close()
        cpu.close()
        w.close()


@pytest.mark.real
def test_cuda_worker_equals_cpu_and_is_fast(scene):
    """The depth card on the real CUDA worker (user 2026-11-10: depth cards run on the GPU): the same maps as the CPU
    path, and the time of one frame_to_key map at 4K (the CPU path takes ~2 min there)."""
    import time
    from engine.cleanup3d.geom import check_worker
    ok, info = check_worker()
    if not (ok and info.startswith("cuda")):
        pytest.skip(f"ComfyUI's Python with CUDA not available ({info})")
    cam, depth = scene
    card = depth_card()
    cards = [card, Card("sky", "niebo", "sky", order=1)]
    cpu = CardGeometry(cam, cards, depths={card.id: depth})
    w = GeomGpuWorker()
    gpu = CardGeometry(cam, cards, device="gpu", gpu=w, depths={card.id: depth})
    try:
        key = Key(card.id, REF)
        for a, b in zip(cpu.frame_to_key(1, key, card), gpu.frame_to_key(1, key, card)):
            okc, okg = a > KER.MISS / 2, b > KER.MISS / 2
            assert (okc != okg).mean() < 0.002                       # device rounding at occlusion edges only
            both = okc & okg
            assert np.percentile(np.abs(a[both] - b[both]), 99.9) < 0.05
        assert w.device.startswith("cuda")
        # 4K: the same scene scaled (intrinsics and depth map)
        k = 4096 / W
        cam4 = Camera3D(4096, int(round(H * k)), np.diag([k, k, 1.0]) @ cam.K, cam.dist, cam.R, cam.t, "synthetic 4K")
        d4 = cv2.resize(depth, (cam4.width, cam4.height), interpolation=cv2.INTER_LINEAR)
        g4 = CardGeometry(cam4, cards, device="gpu", gpu=w, depths={card.id: d4})
        t0 = time.time()
        mx, my = g4.frame_to_key(1, key, card)
        dt = time.time() - t0
        print(f"depth card frame_to_key 4K on CUDA: {dt:.1f} s, hit {float((mx > KER.MISS / 2).mean()):.3f}")
        assert mx.shape == (cam4.height, cam4.width) and dt < 30.0
        g4.close()
    finally:
        gpu.close()
        cpu.close()
        w.close()


# ------------------------------------------------------------------------------------------- render
def texture(X):
    """BGR of world points: a pattern on every surface (period ~0.5 scene units)."""
    a = np.sin(X[:, 0] * 9.0) * np.cos(X[:, 1] * 7.0) + 0.5 * np.sin(X[:, 2] * 5.0 + X[:, 0] * 2.0)
    return np.stack([0.5 + 0.3 * a, 0.45 + 0.25 * np.sin(X[:, 1] * 11.0), 0.5 - 0.3 * a], 1)


def true_bg(cam, f):
    rays = pixel_rays(cam, f, grid())
    s, lab = cast(cam.centre(f), rays)
    X = cam.centre(f) + np.where(np.isinf(s), 0, s)[:, None] * rays
    img = np.where((lab > 0)[:, None], texture(X), 0.0)
    return img.reshape(H, W, 3).astype(np.float32)


def test_render_with_a_depth_card_beats_a_plane_card(scene):
    """The background behind a moving object, rebuilt from one key: the depth card follows the box's parallax, the
    best plane (the wall) cannot – the error inside the object's area is several times smaller."""
    cam, depth = scene
    key_img = true_bg(cam, REF)
    dc = depth_card()
    n, d = WALL_N, -float(WALL_N @ WALL_P)
    pc = Card("wall", "mur", "plane", {"n": n.tolist(), "d": d}, seam_px=1.0)
    errs = {}
    for card, extra in ((dc, {"depths": {dc.id: depth}}), (pc, {})):
        geo = CardGeometry(cam, [card], **extra)
        keys = {card.id: [Key(card.id, REF, active="v001")]}
        e = []
        for f in (1, 2, 4, 5):
            fill, wgt = RD.project_frame(geo, [card], keys, f, lambda k: key_img)
            obj = np.zeros((H, W), bool)
            cx = 60 + 50 * FRAMES.index(f)                     # a moving object over the box and the wall
            obj[50:170, cx - 30:cx + 30] = True
            ok = obj & (wgt > 0.99)
            e.append(np.abs(fill - true_bg(cam, f))[ok].mean())
        errs[card.kind] = float(np.mean(e))
        geo.close()
    assert errs["depth"] < 0.02 and errs["depth"] * 4 < errs["plane"], errs


def test_render_gpu_fake_equals_cpu_with_a_depth_card(scene):
    cam, depth = scene
    card = depth_card()
    key_img = true_bg(cam, REF)
    keys = {card.id: [Key(card.id, REF, active="v001")]}
    cpu = CardGeometry(cam, [card], depths={card.id: depth})
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    gpu = CardGeometry(cam, [card], device="gpu", gpu=w, depths={card.id: depth})
    try:
        a, wa = RD.project_frame(cpu, [card], keys, 2, RD.KeyCache(lambda k: key_img))
        b, wb = RD.project_frame(gpu, [card], keys, 2, RD.KeyCache(lambda k: key_img))
        assert np.abs(wa - wb).max() < 2e-3
        ok = (wa > 0.5) & (wb > 0.5)
        assert np.percentile(np.abs(a - b)[ok], 99) < 2e-2
    finally:
        gpu.close()
        cpu.close()
        w.close()


# ------------------------------------------------------------------------------------------- fit / contract
def test_fit_to_sparse_points_and_planes(scene):
    cam, _ = scene
    disp = depth_of(cam, REF, "disparity")
    rng = np.random.default_rng(1)
    px = rng.uniform([5, 5], [W - 5, H - 5], (300, 2))
    rays = pixel_rays(cam, REF, px)
    s, lab = cast(cam.centre(REF), rays)
    pts = cam.centre(REF) + s[lab > 0, None] * rays[lab > 0]
    rep = DC.fit_depth(cam, REF, disp, "disparity", points=pts)
    assert rep["method"] == "points" and abs(rep["scale"] - 0.5) < 0.01 and abs(rep["offset"]) < 1e-3, rep
    wall = Card("wall", "mur", "plane", {"n": WALL_N.tolist(), "d": -float(WALL_N @ WALL_P)})
    rep = DC.fit_depth(cam, REF, disp, "disparity", points=None, cards=[wall])
    assert rep["method"] == "planes" and abs(rep["scale"] - 0.5) < 0.02, rep          # the box = outliers
    rep = DC.fit_depth(cam, REF, disp, "disparity")
    assert rep["method"] == "none" and rep["scale"] == 1.0
    dp, plane = DC.make_depth(cam, REF, disp, {"conv": "disparity"}, points=pts)
    assert dp["scale"] == pytest.approx(0.5, abs=0.01) and dp["mul"] == 1.0 and len(plane["n"]) == 3


def test_sliders_scale_and_shift_the_surface(scene):
    cam, depth = scene
    card = depth_card(mul=1.0, shift=0.0)
    geo = CardGeometry(cam, [card], depths={card.id: depth})
    z0 = geo.depth_warp(REF, card)[2]
    card2 = depth_card(mul=2.0, shift=1.0)
    geo2 = CardGeometry(cam, [card2], depths={card2.id: depth})
    z1 = geo2.depth_warp(REF, card2)[2]
    ok = (z0 > 0) & (z1 > 0)
    assert np.allclose(z1[ok], 2.0 * z0[ok] + 1.0, rtol=1e-4)
    geo.close()
    geo2.close()


def test_card_json_round_trip_and_backward_compatibility(tmp_path):
    from engine.contract import validate
    old = {"cards": [{"id": "g", "kind": "plane", "plane": {"n": [0, -1, 0], "d": 2}}, {"id": "s", "kind": "sky"}]}
    cards = cards_from_json(old)
    js = cards_to_json(cards)
    assert all("depth" not in c for c in js["cards"])                   # old cards keep their JSON (render_sig)
    assert validate(js) == []
    c = depth_card("disparity", 0.4, 0.01, mul=1.2, shift=-0.5, source="plate", file="cards/relief/depth_0003.npy",
                   stamp="abc", fit={"method": "points"})
    js = cards_to_json([c] + cards)
    assert validate(js) == []
    back = cards_from_json(js)
    assert back[0].kind == "depth" and back[0].depth == c.depth
    assert back[0].depth["frame"] == REF and back[0].depth["mul"] == 1.2 and back[0].depth["source"] == "plate"
    assert cards_to_json(back) == js
    with pytest.raises(ValueError):
        cards_from_json({"cards": [{"id": "x", "kind": "depth"}]})           # no reference frame
    with pytest.raises(ValueError):
        parse_depth({"frame": 1, "conv": "log"})
    back[0].root = str(tmp_path)
    assert back[0].depth_file() == tmp_path / "cards" / "relief" / "depth_0003.npy"


def test_missing_snapshot_is_said_in_polish(scene):
    cam, _ = scene
    card = depth_card()
    geo = CardGeometry(cam, [card])
    with pytest.raises(ValueError, match="Bieżąca klatka"):
        geo.frame_to_key(1, Key(card.id, REF), card)


# ------------------------------------------------------------------------------------------- API (C3)
def test_api_depth_card(tmp_path, monkeypatch):
    """C3: a card becomes a depth card from the shot's depth plate (16-bit PNG disparity), fitted to the solve's
    points; the card list round-trips through PUT; the cards preview draws it."""
    from tests.test_cleanup3d_api import Env
    from tests.test_cleanup3d_render import GROUND_Y as GY, H as RH, W as RW
    env = Env(tmp_path, monkeypatch)
    c = env.client
    src = c.get("/api/cleanup3d/depth_sources").json()["sources"]
    assert [s["id"] for s in src] == ["computed", "plate"] and not any(s["available"] for s in src)
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    assert c.post("/api/cleanup3d/cards/nope/depth", json={}).status_code == 404
    r = c.post("/api/cleanup3d/cards/ground/depth", json={"frame": 5})
    assert r.status_code == 409 and "Brak mapy głębi" in r.json()["detail"]
    cam = env.cam
    xx, yy = np.meshgrid(np.arange(RW, dtype=np.float64), np.arange(RH, dtype=np.float64))
    x, y = KER.view_rays(KER.NP, {"K": cam.K, "dist": cam.dist, "w": RW, "h": RH}, xx, yy)
    wd = np.stack([x, y, np.ones_like(x)], -1) @ cam.R[5]
    C = cam.centre(5)
    hit = wd[..., 1] > 1e-6
    z = np.where(hit, (GY - C[1]) / np.where(hit, wd[..., 1], 1.0), np.inf)     # ray parameter = camera z
    disp = np.where(np.isfinite(z), 1.0 / z, 0.0)
    (env.shot / "depth").mkdir()
    cv2.imwrite(str(env.shot / "depth" / "d_0005.png"), np.clip(np.rint(disp * 65535), 0, 65535).astype(np.uint16))
    env.state["manifest"]["plates"]["depth"] = str(env.shot / "depth" / "d_####.png").replace("\\", "/")
    assert c.get("/api/cleanup3d/depth_sources").json()["sources"][1]["available"]
    r = c.post("/api/cleanup3d/cards/ground/depth", json={"frame": 5})
    assert r.status_code == 200, r.text
    card, fit = r.json()["card"], r.json()["fit"]
    assert card["kind"] == "depth" and card["depth"]["frame"] == 5 and card["depth"]["source"] == "plate"
    assert fit["method"] == "points" and abs(fit["scale"] - 1.0) < 0.02 and abs(fit["offset"]) < 2e-3, fit
    assert abs(abs(card["plane"]["n"][1]) - 1.0) < 1e-2                     # the approximate plane = the ground
    from engine.cleanup3d import store
    from engine.cleanup3d import types as T
    snap = T.root_dir(env.work) / card["depth"]["file"]
    assert snap.is_file() and np.load(snap).shape == (RH, RW)
    cards = c.get("/api/cleanup3d/cards").json()["cards"]
    cards[0]["depth"] = {**cards[0]["depth"], "mul": 1.1, "shift": 0.2}      # the sliders, as the palette saves them
    r = c.put("/api/cleanup3d/cards", json={"cards": cards})
    assert r.status_code == 200 and r.json()["cards"][0]["depth"]["mul"] == 1.1
    got = store.load_cards(env.work)[0]
    assert got.depth_file() == snap and got.depth["shift"] == 0.2
    r = c.get("/api/cleanup3d/view/cards/5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    r = c.post("/api/cleanup3d/cards/ground/depth", json={"frame": 99})
    assert r.status_code == 409 and "F99" in r.json()["detail"]


def test_a_depth_card_makes_the_scene_use_the_gpu(monkeypatch, tmp_path):
    from engine.cleanup3d import geom as G
    from engine.cleanup3d import render as RD
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    monkeypatch.setattr(G, "worker_python", lambda: py)
    plane = Card("ground", "piasek", "plane", {"n": [0, -1, 0], "d": 1.0})
    deep = depth_card()
    monkeypatch.setenv("MD_C3D_DEPTH_GPU", "1")
    assert RD.depth_needs_gpu([plane, deep]) and not RD.depth_needs_gpu([plane])
    deep.off = True
    assert not RD.depth_needs_gpu([plane, deep])                 # a card turned off with the eye does not count
    deep.off = False
    monkeypatch.setattr(G, "worker_python", lambda: tmp_path / "missing.exe")
    assert not RD.depth_needs_gpu([deep])                        # no ComfyUI Python: the asked device (CPU)
    monkeypatch.setenv("MD_C3D_DEPTH_GPU", "0")
    monkeypatch.setattr(G, "worker_python", lambda: py)
    assert not RD.depth_needs_gpu([deep])
