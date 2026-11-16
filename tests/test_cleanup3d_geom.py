# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D geometry (package P1, engine/cleanup3d/geom.py + geom_gpu.py) on a synthetic scene: a ground plane, a wall
with an extent drawn in one frame, the sky; a camera that rotates and translates; non-zero distortion (k1, k2, p1, p2).

Expected values are derived independently: cv2.projectPoints of world points (distortion included) and the plane
homography K_k (R - t n^T / d) K_t^-1 without distortion. removers_test/cards.py is not imported: it loads its solve's
data files at import (outputs/sfm*/cams.npz, plane.json), which are not part of the repo."""

import os
import sys

import cv2
import numpy as np
import pytest

from engine import curves
from engine.cleanup3d import geom_gpu as KER
from engine.cleanup3d.geom import CardGeometry, GeomGpuWorker, card_plane, check_worker
from engine.cleanup3d.types import Area, Camera3D, Card, Key, cards_from_json, cards_to_json

W, H = 320, 200
KM = np.array([[260.0, 0, 160.3], [0, 255.0, 99.7], [0, 0, 1]])
DIST = np.array([-0.12, 0.03, 0.001, -0.0008])
FRAMES = [1, 2, 3, 4, 5]
GROUND_Y = 2.0          # y down: the ground is 2 below the camera
WALL_Z = 30.0


def _rot(yaw, pitch, roll=0.0):
    a, b, c = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def make_camera(dist=DIST):
    R, t = {}, {}
    for i, f in enumerate(FRAMES):
        C = np.array([0.6 * i, -0.05 * i, 0.8 * i])          # moves right, up a little, forward
        Rf = _rot(4.0 * i - 6.0, 6.0 - 1.5 * i, 0.7 * i)       # yaw, pitch down toward the ground, roll
        R[f], t[f] = Rf, -Rf @ C
    return Camera3D(W, H, KM.copy(), np.asarray(dist, float), R, t, "synthetic")


def rect_curve(x0, y0, x1, y1):
    return curves.spline([curves.point((x0, y0)), curves.point((x1, y0)), curves.point((x1, y1)),
                          curves.point((x0, y1))], closed=True)


WALL_RECT = (40.0, 10.0, 230.0, 70.0)        # in frame 1 (plate px)


def make_cards(wall_extent=True, sky=True, seam=8.0):
    cards = [Card("ground", "piasek", "plane", {"n": [0, -1, 0], "d": GROUND_Y}, order=0, seam_px=seam)]
    wall = Card("wall", "mur", "plane", {"n": [0, 0, 2], "d": -2 * WALL_Z}, order=1, seam_px=seam)   # unnormalised, away from the cameras
    if wall_extent:
        wall.extent = [Area("rect", "curve", "add", rect_curve(*WALL_RECT))]
        wall.extent_frame = 1
    cards.append(wall)
    if sky:
        cards.append(Card("sky", "niebo", "sky", order=2, seam_px=seam))
    return cards


def project_world(cam, f, X, dist=None):
    rv, _ = cv2.Rodrigues(cam.R[f])
    d = cam.dist if dist is None else dist
    p, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rv, cam.t[f], cam.K, d)
    return p.reshape(-1, 2)


def unproject_to_plane(cam, f, px, n, d):
    u = cv2.undistortPointsIter(np.asarray(px, np.float64).reshape(-1, 1, 2), cam.K, cam.dist, None, None,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)).reshape(-1, 2)
    rays = np.c_[u, np.ones(len(u))] @ cam.R[f]                  # R^T r per row
    C = cam.centre(f)
    s = -(n @ C + d) / (rays @ n)
    return C + s[:, None] * rays


@pytest.fixture(scope="module")
def scene():
    cam = make_camera()
    return cam, make_cards()


# ------------------------------------------------------------------------------------------- lens / plane basics
def test_undistort_inverts_distort_like_opencv():
    rng = np.random.default_rng(0)
    px = rng.uniform([-10, -10], [W + 10, H + 10], (500, 2))
    xd, yd = (px[:, 0] - KM[0, 2]) / KM[0, 0], (px[:, 1] - KM[1, 2]) / KM[1, 1]
    x, y = KER.undistort(KER.NP, xd, yd, DIST)
    rx, ry = KER.distort(x, y, DIST)
    assert np.abs(rx - xd).max() < 1e-12 and np.abs(ry - yd).max() < 1e-12
    ref = cv2.undistortPointsIter(px.reshape(-1, 1, 2), KM, DIST, None, None,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-15)).reshape(-1, 2)
    assert np.abs(ref[:, 0] - x).max() * KM[0, 0] < 1e-6


def test_plane_normal_turned_toward_cameras_and_adjust(scene):
    cam, cards = scene
    n, d = card_plane(cam, cards[1])
    assert np.allclose(n, [0, 0, -1]) and abs(d - WALL_Z) < 1e-12        # normalised, flipped toward the cameras
    assert card_plane(cam, cards[2]) is None
    g = Card("g", "g", "plane", {"n": [0, -1, 0], "d": GROUND_Y}, adjust={"dh": 0.5, "pitch": 0.0, "roll": 0.0})
    n, d = card_plane(cam, g)
    assert np.allclose(n, [0, -1, 0]) and abs(d - (GROUND_Y - 0.5)) < 1e-12   # 0.5 toward the cameras
    g.adjust = {"dh": 0.0, "pitch": 5.0, "roll": -3.0}
    n, d = card_plane(cam, g)
    c = cam.centre(FRAMES[len(FRAMES) // 2])
    pivot = c - (np.array([0, -1, 0]) @ c + GROUND_Y) * np.array([0, -1, 0])
    assert abs(n @ pivot + d) < 1e-9                                      # tilts about the point under the middle camera
    assert abs(np.degrees(np.arccos(np.clip(n @ [0, -1, 0], -1, 1))) - np.hypot(5.0, 3.0)) < 0.2


# ------------------------------------------------------------------------------------------- maps
def test_points_match_projected_world_points(scene):
    """A point on a card: its pixel in frame t -> points_frame_to_key = its pixel in the key (cv2.projectPoints)."""
    cam, cards = scene
    g = CardGeometry(cam, cards)
    rng = np.random.default_rng(1)
    Xg = np.c_[rng.uniform(-3, 4, 300), np.full(300, GROUND_Y), rng.uniform(4, 25, 300)]
    Xw = np.c_[rng.uniform(-8, 14, 300), rng.uniform(-12, 1.5, 300), np.full(300, WALL_Z)]
    for card, X in ((cards[0], Xg), (cards[1], Xw)):
        for t, k in ((1, 4), (5, 2), (3, 3)):
            pt, pk = project_world(cam, t, X), project_world(cam, k, X)
            sel = np.all((pt > 0) & (pt < [W - 1, H - 1]), 1) & np.all((pk > -20) & (pk < [W + 20, H + 20]), 1)
            assert sel.sum() > 50
            got = g.points_frame_to_key(pt[sel], t, Key(card.id, k), card)
            assert np.nanmax(np.abs(got - pk[sel])) < 1e-6, (card.id, t, k)


def test_frame_key_frame_identity(scene):
    """frame -> key -> frame is the identity where both see the card (maps composed by bilinear sampling, and exactly
    through the points)."""
    cam, cards = scene
    g = CardGeometry(cam, cards)
    for card in cards:
        key = Key(card.id, 1)
        mx, my = g.frame_to_key(4, key, card)
        bx, by = g.key_to_frame(key, card, 4)
        ok = (mx > 2) & (my > 2) & (mx < W - 3) & (my < H - 3)
        assert ok.mean() > 0.1, card.id
        rx = cv2.remap(bx, mx, my, cv2.INTER_LINEAR)
        ry = cv2.remap(by, mx, my, cv2.INTER_LINEAR)
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        back_ok = ok & (cv2.remap((bx > KER.MISS / 2).astype(np.float32), mx, my, cv2.INTER_LINEAR) > 0.999)
        err = np.hypot(rx - X, ry - Y)[back_ok]
        assert np.percentile(err, 99) < 0.05 and err.max() < 0.1, (card.id, err.max())
        # exact round trip through the points (a key without canvas = the plate frame as shot)
        ys, xs = np.nonzero(ok)
        pts = np.c_[xs, ys][:: max(1, len(xs) // 400)].astype(np.float64)
        kp = g.points_frame_to_key(pts, 4, key, card)
        back = g.points_frame_to_key(kp, 1, Key(card.id, 4), card)
        assert np.nanmax(np.abs(back - pts)) < 1e-6


def test_plane_homography_without_distortion():
    """No distortion: frame_to_key of a plane card = the plane homography (analytic)."""
    cam = make_camera(dist=[0, 0, 0, 0])
    cards = make_cards(wall_extent=False)
    g = CardGeometry(cam, cards)
    t, k = 2, 5
    for card in cards[:2]:
        n, d = card_plane(cam, card)
        Rt, tt, Rk, tk = cam.R[t], cam.t[t], cam.R[k], cam.t[k]
        Rrel = Rk @ Rt.T
        trel = tk - Rrel @ tt
        n_t = Rt @ n                      # the plane in frame-t camera coordinates: n_t . X + d_t = 0
        d_t = d - n_t @ tt
        Hm = KM @ (Rrel - np.outer(trel, n_t) / d_t) @ np.linalg.inv(KM)
        mx, my = g.frame_to_key(t, Key(card.id, k), card)
        X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        p = np.stack([X, Y, np.ones_like(X)], -1) @ Hm.T
        ex, ey = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
        ok = (mx > KER.MISS / 2) & (np.hypot(ex, ey) < 1e4)
        assert ok.mean() > 0.2
        assert np.abs(mx[ok] - ex[ok]).max() < 2e-3 and np.abs(my[ok] - ey[ok]).max() < 2e-3, card.id


def test_sky_is_rotation_only(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    sky = cards[2]
    rng = np.random.default_rng(3)
    pts = rng.uniform([0, 0], [W - 1, H - 1], (200, 2))
    got = g.points_frame_to_key(pts, 5, Key("sky", 1), sky)
    u = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), KM, DIST, None, None,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-15)).reshape(-1, 2)
    dirs = np.c_[u, np.ones(len(u))] @ cam.R[5]          # world directions
    rv, _ = cv2.Rodrigues(cam.R[1])
    exp, _ = cv2.projectPoints(dirs.reshape(-1, 1, 3), rv, np.zeros(3), KM, DIST)
    exp = exp.reshape(-1, 2)
    ok = np.isfinite(got[:, 0]) & np.all(np.abs(exp - [W / 2, H / 2]) < [W, H], 1)
    assert ok.sum() > 100 and np.abs(got[ok] - exp[ok]).max() < 1e-4


def test_key_canvas_is_its_own_pinhole(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    Kc = np.array([[240.0, 0, 250.0], [0, 240.0, 160.0], [0, 0, 1]])
    key = Key("ground", 3, canvas={"w": 500, "h": 320, "K": Kc.tolist(), "frame": 2})
    rng = np.random.default_rng(4)
    X = np.c_[rng.uniform(-3, 4, 200), np.full(200, GROUND_Y), rng.uniform(4, 25, 200)]
    pt = project_world(cam, 5, X)
    pk = project_world(Camera3D(500, 320, Kc, np.zeros(4), cam.R, cam.t), 2, X, dist=np.zeros(4))
    sel = np.all((pt > 0) & (pt < [W - 1, H - 1]), 1)
    got = g.points_frame_to_key(pt[sel], 5, key, cards[0])
    assert np.nanmax(np.abs(got - pk[sel])) < 1e-6
    bx, by = g.key_to_frame(key, cards[0], 5)
    assert bx.shape == (320, 500) and bx.dtype == np.float32
    mx, my = g.frame_to_key(5, key, cards[0])
    assert mx.shape == (H, W)


def test_miss_and_nan_where_the_ray_misses_the_card(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    mx, my = g.frame_to_key(1, Key("ground", 2), cards[0])
    assert (mx[0] == KER.MISS).any() and (mx[-1] > KER.MISS / 2).all()         # top rows look above the horizon
    got = g.points_frame_to_key(np.array([[W / 2, 0.0], [W / 2, H - 1.0]]), 1, Key("ground", 2), cards[0])
    assert np.isnan(got[0]).all() and np.isfinite(got[1]).all()


# ------------------------------------------------------------------------------------------- extent / weights / coverage
def test_extent_stays_fixed_in_the_world(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    wall = cards[1]
    x0, y0, x1, y1 = WALL_RECT
    def world(px):
        return unproject_to_plane(cam, 1, px, np.array([0, 0, -1.0]), WALL_Z)
    corners = world([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    lo, hi = corners.min(0), corners.max(0)
    rng = np.random.default_rng(5)
    span = hi - lo
    inner = np.c_[rng.uniform(lo[0] + 0.08 * span[0], hi[0] - 0.08 * span[0], 400),
                  rng.uniform(lo[1] + 0.08 * span[1], hi[1] - 0.08 * span[1], 400), np.full(400, WALL_Z)]
    outer = np.c_[rng.uniform(lo[0] - span[0], hi[0] + span[0], 2000), rng.uniform(lo[1] - span[1], hi[1] + span[1], 2000),
                  np.full(2000, WALL_Z)]
    pad = 0.08 * span
    outer = outer[(outer[:, 0] < lo[0] - pad[0]) | (outer[:, 0] > hi[0] + pad[0])
                  | (outer[:, 1] < lo[1] - pad[1]) | (outer[:, 1] > hi[1] + pad[1])]
    for f in FRAMES:
        e = g.extent_in_frame(wall, f)
        for X, want in ((inner, 1.0), (outer, 0.0)):
            p = np.round(project_world(cam, f, X)).astype(int)
            sel = np.all((p >= 0) & (p < [W, H]), 1)
            assert sel.sum() > 20, f
            vals = e[p[sel, 1], p[sel, 0]]
            assert np.abs(vals - want).max() < 0.01, (f, want)
    # the extent moves in the picture (the camera moves) – it is not a fixed screen mask
    assert np.abs(g.extent_in_frame(wall, 1) - g.extent_in_frame(wall, 5)).mean() > 0.02


def test_extent_brush_layers_add_and_remove(scene):
    cam, _ = scene
    cards = make_cards()
    wall = cards[1]
    hole = np.zeros((H, W), np.uint8)
    hole[30:50, 100:140] = 255
    wall.extent.append(Area("hole", "brush", "remove", png="a_hole.png"))
    g = CardGeometry(cam, cards, layers={"wall": {"hole": hole}})
    e = g.extent_in_frame(wall, 1)
    assert e[40, 120] == 0.0 and e[20, 60] == 1.0 and e[100, 300] == 0.0
    with pytest.raises(ValueError):
        CardGeometry(cam, cards).extent_in_frame(wall, 1)        # a brush layer the caller did not pass
    only_remove = Card("w2", "w2", "plane", {"n": [0, 0, -1], "d": WALL_Z},
                       extent=[Area("hole", "brush", "remove", png="a_hole.png")], extent_frame=1)
    e2 = CardGeometry(cam, [only_remove], layers={"w2": {"hole": hole}}).extent_in_frame(only_remove, 1)
    assert e2[40, 120] == 0.0 and e2[150, 10] == 1.0              # no "add" layer: the whole card minus the hole


def test_card_weights_partition_the_frame(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    for f in FRAMES:
        ws = g.card_weights(f)
        assert list(ws) == ["ground", "wall", "sky"]
        tot = sum(ws.values())
        assert all(w.dtype == np.float32 and w.shape == (H, W) and w.min() >= 0 and w.max() <= 1 + 1e-6
                   for w in ws.values())
        assert np.abs(tot - 1).max() < 1e-5                   # the sky catches every ray: 1 everywhere
        assert np.array_equal(g.card_weight(cards[0], f), ws["ground"])
    ws = g.card_weights(1)
    assert ws["ground"][-1].min() == 1.0                       # bottom row: only the ground
    assert ws["sky"][0, 300] == 1.0                            # top right, outside the wall's extent: sky
    wall_in = ws["wall"][40, 120]
    assert wall_in == 1.0                                      # inside the extent (drawn in frame 1)
    # soft over seam_px: across the wall's left edge (x = 40 in frame 1) the weight ramps over ~seam_px
    row = ws["wall"][40, 20:60]
    mid = row[(row > 0.05) & (row < 0.95)]
    assert 3 <= len(mid) <= 9
    assert np.all(np.diff(row) >= -1e-6)


def test_weights_zero_where_no_card_is_hit(scene):
    cam, _ = scene
    cards = make_cards(sky=False)
    g = CardGeometry(cam, cards)
    ws = g.card_weights(2)
    tot = ws["ground"] + ws["wall"]
    hit_any = (g.extent_in_frame(cards[0], 2) > 0) | (g.extent_in_frame(cards[1], 2) >= 0.5)
    assert tot.max() <= 1 + 1e-6
    assert np.abs(tot[hit_any] - 1).max() < 1e-5
    assert (~hit_any).sum() > 100 and tot[~hit_any].max() == 0.0


def test_card_order_decides_ownership(scene):
    cam, _ = scene
    cards = make_cards(wall_extent=False, sky=False)
    g1 = CardGeometry(cam, cards)
    cards2 = make_cards(wall_extent=False, sky=False)
    cards2[0].order, cards2[1].order = 1, 0                   # the (infinite) wall in front of the ground
    g2 = CardGeometry(cam, cards2)
    assert g1.card_weights(3)["ground"][-1].min() == 1.0
    assert g2.card_weights(3)["ground"].max() == 0.0          # every forward ray hits the wall first


def test_coverage_fades_to_zero_at_the_key_border(scene):
    cam, cards = scene
    g = CardGeometry(cam, cards)
    key = Key("ground", 3)
    cov = g.coverage(3, key, cards[0], edge_px=20)
    gr = g.extent_in_frame(cards[0], 3) > 0
    assert cov[gr].max() == 1.0
    assert cov[-1, :].max() < 1e-3 and cov[:, 0][gr[:, 0]].max() < 1e-3           # the key's own border = 0
    assert abs(cov[-11, W // 2] - 10 / 20) < 0.02                                  # 10 px inside = half
    assert cov[~gr].max() == 0.0                                                   # no card -> no coverage
    cov2 = g.coverage(1, key, cards[0], edge_px=20)
    mx, my = g.frame_to_key(1, key, cards[0])
    d = np.minimum(np.minimum(mx, W - 1 - mx), np.minimum(my, H - 1 - my))
    exp = np.where(mx > KER.MISS / 2, np.clip(d / 20, 0, 1), 0)
    assert np.abs(cov2 - exp).max() < 1e-5
    assert ((cov2 > 0) & (cov2 < 1)).sum() > 50


# ------------------------------------------------------------------------------------------- kernel pieces
def test_box_blur_and_bilinear():
    a = np.zeros((9, 12))
    a[:, 6:] = 1.0
    b = KER.box_blur(KER.NP, a, 5)
    assert np.allclose(b[:, :4], 0) and np.allclose(b[:, 8:], 1) and np.allclose(b[0, 4:8], [0.2, 0.4, 0.6, 0.8])
    ref = cv2.blur(np.random.default_rng(0).random((20, 30)).astype(np.float64), (7, 7), borderType=cv2.BORDER_REPLICATE)
    got = KER.box_blur(KER.NP, np.random.default_rng(0).random((20, 30)), 7)
    assert np.abs(ref - got).max() < 1e-12
    img = np.arange(20.0).reshape(4, 5)
    v = KER.bilinear(KER.NP, img, np.array([0.0, 1.5, 4.0, -0.1, 4.2]), np.array([0.0, 2.25, 3.0, 1.0, 1.0]), -7)
    assert np.allclose(v, [0, 12.75, 19, -7, -7])


def test_cards_roundtrip_json_keeps_geometry(scene):
    cam, cards = scene
    again = cards_from_json(cards_to_json(cards))
    a = CardGeometry(cam, cards).card_weights(2)
    b = CardGeometry(Camera3D.from_json(cam.to_json()), again).card_weights(2)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------------------------------- GPU path
FAKE_ENV = {**os.environ, "MD_GEOM_GPU_FAKE": "1"}


def _compare_devices(cam, cards, worker, tol_px):
    cpu = CardGeometry(cam, cards)
    gpu = CardGeometry(cam, cards, device="gpu", gpu=worker)
    key = Key("ground", 2)
    for card in cards:
        k = Key(card.id, 2)
        for a, b in zip(cpu.frame_to_key(4, k, card), gpu.frame_to_key(4, k, card)):
            ok = (a > KER.MISS / 2) & (np.abs(a) < 1e4)
            assert np.array_equal(ok, (b > KER.MISS / 2) & (np.abs(b) < 1e4)) or (ok != (b > KER.MISS / 2)).mean() < 1e-4
            both = ok & (b > KER.MISS / 2)
            assert np.abs(a[both] - b[both]).max() < tol_px
        for a, b in zip(cpu.key_to_frame(k, card, 4), gpu.key_to_frame(k, card, 4)):
            both = (a > KER.MISS / 2) & (b > KER.MISS / 2) & (np.abs(a) < 1e4)
            assert np.abs(a[both] - b[both]).max() < tol_px
    pts = np.array([[10.0, 190.0], [160.0, 150.0], [300.0, 120.0], [160.0, 0.0]])
    pa = cpu.points_frame_to_key(pts, 4, key, cards[0])
    pb = gpu.points_frame_to_key(pts, 4, key, cards[0])
    assert np.array_equal(np.isnan(pa), np.isnan(pb)) and np.nanmax(np.abs(pa - pb)) < tol_px
    wa, wb = cpu.card_weights(3), gpu.card_weights(3)
    for c in wa:
        assert np.abs(wa[c] - wb[c]).max() < 1e-3, c
    ca, cb = cpu.coverage(3, key, cards[0]), gpu.coverage(3, key, cards[0])
    assert np.abs(ca - cb).max() < 1e-3


def test_gpu_protocol_fake_worker(scene):
    """The GPU plumbing (requests, .npy files, GPU.use) with the worker's numpy backend in this .venv."""
    cam, cards = scene
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        _compare_devices(cam, cards, w, 1e-3)
        assert w.device == "fake"
    finally:
        w.close()


@pytest.mark.real
def test_gpu_matches_cpu_within_005px(scene):
    ok, info = check_worker()
    if not (ok and info.startswith("cuda")):
        pytest.skip(f"ComfyUI's Python with CUDA not available ({info})")
    cam, cards = scene
    w = GeomGpuWorker()
    try:
        _compare_devices(cam, cards, w, 0.05)
        assert w.device.startswith("cuda")
    finally:
        w.close()
