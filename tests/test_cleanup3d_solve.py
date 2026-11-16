# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D package P10 – the camera solve. Synthetic (no COLMAP, no GPU): the lens model, tracks -> points, the joint
bundle adjustment with / without the horizon, the epipolar test, the horizon of a frame, planes, COLMAP's text model,
the orchestration with a fake COLMAP runner + a fake tracker (stages, masks, cancel, errors, the Camera3D contract).
`real` (workstation): sh060, 57 frames, COLMAP from MD_COLMAP_EXE, compared with the experiment's best solve (sfm8)."""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from engine.cleanup3d import solve as S
from engine.cleanup3d import solve_ba as BA
from engine.cleanup3d import solve_colmap as CM
from engine.cleanup3d import store
from engine.cleanup3d import types as T

UP = np.array([0.02, -1.0, 0.03]) / np.linalg.norm([0.02, -1.0, 0.03])


# ----------------------------------------------------------------------------- a synthetic shot
class Scene:
    """Cameras moving sideways + forward over a ground plane (y down, ground at y = 1.5) with things standing on it."""

    def __init__(self, n_cam=12, n_pts=400, W=1280, H=720, f=1100.0, k1=-0.06, k2=0.015, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.H = W, H
        self.intr = np.array([f, f, (W - 1) / 2, (H - 1) / 2, k1, k2, 0.0, 0.0])
        ng = int(0.7 * n_pts)
        ground = np.stack([rng.uniform(-9, 9, ng), np.full(ng, 1.5), rng.uniform(3, 26, ng)], 1)
        other = np.stack([rng.uniform(-9, 9, n_pts - ng), rng.uniform(-2.5, 1.4, n_pts - ng), rng.uniform(4, 26, n_pts - ng)], 1)
        self.X = np.concatenate([ground, other])
        self.n_ground = ng
        i = np.arange(n_cam)
        self.C = np.stack([0.3 * i - 0.15 * n_cam, 0.05 * np.sin(i), 0.12 * i], 1)
        self.R = Rotation.from_euler("yxz", np.stack([np.radians(-6 + 1.1 * i), np.radians(np.full(n_cam, -7.0) + 0.3 * np.cos(i)),
                                                      np.radians(0.4 * np.sin(0.7 * i))], 1)).as_matrix()
        self.t = -np.einsum("nij,nj->ni", self.R, self.C)

    def observe(self, noise=0.3, seed=1):
        rng = np.random.default_rng(seed)
        oc, op, oxy = [], [], []
        for c in range(len(self.R)):
            z = (self.X @ self.R[c].T + self.t[c])[:, 2]
            uv = BA.project(self.R[c], self.t[c], self.X, self.intr)
            ok = (z > 0.5) & (uv[:, 0] > 0) & (uv[:, 0] < self.W - 1) & (uv[:, 1] > 0) & (uv[:, 1] < self.H - 1)
            oc.append(np.full(int(ok.sum()), c)); op.append(np.nonzero(ok)[0])
            oxy.append(uv[ok] + rng.normal(0, noise, (int(ok.sum()), 2)))
        return np.concatenate(oc), np.concatenate(op), np.concatenate(oxy)

    def horizon(self, noise=0.3, seed=2, per_cam=40):
        """Points of the level plane's line at infinity, as each camera sees them."""
        rng = np.random.default_rng(seed)
        e1, e2 = BA._tangent(UP)
        hc, hx = [], []
        for c in range(len(self.R)):
            a = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            w = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
            xc = w @ self.R[c].T
            lim = 1.5 * np.hypot(self.W, self.H) / 2 / self.intr[0]      # inside the lens model (it folds back far outside)
            ok = (xc[:, 2] > 0.2) & (np.hypot(xc[:, 0], xc[:, 1]) < lim * np.abs(xc[:, 2]))
            xd, yd = BA.distort(xc[ok, 0] / xc[ok, 2], xc[ok, 1] / xc[ok, 2], *self.intr[4:])
            uv = np.stack([xd * self.intr[0] + self.intr[2], yd * self.intr[1] + self.intr[3]], 1)
            uv = uv[(uv[:, 0] > 0) & (uv[:, 0] < self.W - 1) & (uv[:, 1] > 0) & (uv[:, 1] < self.H - 1)]
            uv = uv[np.linspace(0, len(uv) - 1, min(per_cam, len(uv))).astype(int)]
            hc.append(np.full(len(uv), c)); hx.append(uv + rng.normal(0, noise, uv.shape))
        return np.concatenate(hc), np.concatenate(hx)

    def start(self, seed=3, f_scale=1.04, rot_deg=0.3, c_sig=0.02, x_sig=0.05):
        """A COLMAP-like start: noisy poses and points, the focal off, no distortion. Camera 0 is exact (the gauge)."""
        rng = np.random.default_rng(seed)
        n = len(self.R)
        dR = Rotation.from_rotvec(rng.normal(0, np.radians(rot_deg), (n, 3))).as_matrix()
        dR[0] = np.eye(3)
        R0 = np.einsum("nij,njk->nik", dR, self.R)
        dC = rng.normal(0, c_sig, (n, 3))
        dC[0] = 0
        t0 = -np.einsum("nij,nj->ni", R0, self.C + dC)
        intr0 = self.intr.copy()
        intr0[0] *= f_scale
        intr0[1] *= f_scale * 1.01                             # COLMAP's OPENCV model: fx != fy
        intr0[4:6] = 0
        return R0, t0, intr0, self.X + rng.normal(0, x_sig, self.X.shape)


def _pose_errors(R, t, sc):
    """(max rotation error in degrees, max centre error) after the similarity alignment of the two camera sets."""
    s, Ra, ta = BA.align_cameras(R, t, sc.R, sc.t)
    C = BA.centres(R, t)
    rot = [np.degrees(np.linalg.norm(Rotation.from_matrix(R[i] @ Ra.T @ sc.R[i].T).as_rotvec())) for i in range(len(R))]
    return max(rot), float(np.abs((s * C @ Ra.T + ta) - sc.C).max())


# ----------------------------------------------------------------------------- lens model, tracks
def test_lens_model_matches_opencv():
    intr = np.array([1100.0, 1090.0, 640.0, 355.0, -0.08, 0.02, 0.001, -0.002])
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 9, 200)], 1)
    R = Rotation.from_rotvec([0.02, -0.05, 0.01]).as_matrix()
    t = np.array([0.1, -0.2, 0.3])
    uv = BA.project(R, t, X, intr)
    ref = cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, BA.k_matrix(intr), intr[4:])[0].reshape(-1, 2)
    assert np.abs(uv - ref).max() < 1e-6
    und = BA.undistort(uv, intr)
    ref_u = cv2.undistortPoints(uv[:, None], BA.k_matrix(intr), intr[4:]).reshape(-1, 2)
    assert np.abs(und - ref_u).max() < 1e-5
    Xc = X @ R.T + t
    assert np.abs(und - Xc[:, :2] / Xc[:, 2:]).max() < 1e-6


def test_tracks_triangulate_and_moving_ones_drop():
    sc = Scene(n_cam=10, n_pts=120)
    rng = np.random.default_rng(5)
    T_, M = 10, 120
    tr = np.stack([BA.project(sc.R[c], sc.t[c], sc.X, sc.intr) for c in range(T_)]) + rng.normal(0, 0.3, (T_, M, 2))
    vis = np.ones((T_, M), bool)
    tr[:, :15, 1] += 4.0 * np.arange(T_)[:, None]            # 15 tracks move on their own (4 px / frame)
    vis[4:, 15:25] = False                                    # 10 tracks too short
    out = BA.triangulate_tracks(tr, vis, np.arange(T_), sc.R, sc.t, sc.intr, min_vis=6, drop_px=2.0)
    assert out["short"] == 10 and out["total"] == M
    assert not set(out["track"]) & set(range(25)), "moving and short tracks must not be kept"
    assert out["kept"] >= 90
    assert np.median(np.linalg.norm(out["X"] - sc.X[out["track"]], axis=1)) < 0.05
    assert len(out["obs_cam"]) == len(out["obs_pt"]) == len(out["obs_xy"]) and out["obs_pt"].max() == out["kept"] - 1


# ----------------------------------------------------------------------------- bundle adjustment
@pytest.mark.parametrize("with_horizon", [False, True])
def test_bundle_adjustment_recovers_lens_and_poses(with_horizon):
    sc = Scene()
    oc, op, oxy = sc.observe(noise=0.3)
    R0, t0, intr0, X0 = sc.start()
    hc, hx = sc.horizon(noise=0.3) if with_horizon else (None, None)
    e0 = np.median(BA.reproj_px(R0, t0, intr0, X0, oc, op, oxy))
    r = BA.bundle_adjust(R0, t0, intr0, X0, oc, op, oxy, single_focal=True, hz_cam=hc, hz_xy=hx)
    e1 = np.median(BA.reproj_px(r["R"], r["t"], r["intr"], r["X"], oc, op, oxy))
    f, k1, k2 = r["intr"][0], r["intr"][4], r["intr"][5]
    rot, cen = _pose_errors(r["R"], r["t"], sc)
    print(f"horizon={with_horizon}: reproj {e0:.2f} -> {e1:.3f} px, f {f:.2f} (true 1100), k1 {k1:.4f} (-0.06), "
          f"k2 {k2:.4f} (0.015), rot {rot:.4f} deg, centre {cen:.4f}, nfev {r['nfev']}")
    assert e0 > 5 and e1 < 0.6                                # noise 0.3 px per axis -> ~0.37 px median
    assert r["intr"][0] == r["intr"][1]                       # one focal
    assert abs(f / 1100.0 - 1) < 0.005
    assert abs(k1 + 0.06) < 0.008 and abs(k2 - 0.015) < 0.02
    assert rot < 0.05 and cen < 0.02
    assert np.allclose(r["R"][0], R0[0]) and np.allclose(r["t"][0], t0[0])   # the gauge camera does not move
    if with_horizon:
        assert np.degrees(np.arccos(np.clip(r["up"] @ UP, -1, 1))) < 0.05
        assert np.median(np.abs(BA.horizon_residual(r["R"], hc, hx, r["intr"], r["up"]))) < 0.5
    else:
        assert r["up"] is None


def test_horizon_pins_the_distortion_of_a_forward_move():
    """The case the constraint exists for (sh060): a camera moving forward over a plane cannot separate the barrel
    distortion from the focal – with the horizon k1 comes out closer."""
    sc = Scene(n_cam=10, n_pts=300, seed=4)
    i = np.arange(10)
    sc.C = np.stack([0.02 * i, np.zeros(10), 0.25 * i], 1)
    sc.R = Rotation.from_euler("yxz", np.stack([np.radians(0.2 * i), np.radians(np.full(10, -6.0)), np.zeros(10)], 1)).as_matrix()
    sc.t = -np.einsum("nij,nj->ni", sc.R, sc.C)
    sc.X = sc.X[:sc.n_ground]                                 # the ground only
    oc, op, oxy = sc.observe(noise=0.5)
    R0, t0, intr0, X0 = sc.start()
    hc, hx = sc.horizon(noise=0.3)
    a = BA.bundle_adjust(R0, t0, intr0, X0, oc, op, oxy)
    b = BA.bundle_adjust(R0, t0, intr0, X0, oc, op, oxy, hz_cam=hc, hz_xy=hx)
    ea, eb = abs(a["intr"][4] + 0.06), abs(b["intr"][4] + 0.06)
    print(f"k1 error without horizon {ea:.5f}, with {eb:.5f}; f {a['intr'][0]:.1f} / {b['intr'][0]:.1f}")
    assert eb < 0.015 and eb < 0.5 * ea


def test_bundle_adjustment_fixed_lens_and_cancel(monkeypatch):
    sc = Scene(n_cam=6, n_pts=150)
    oc, op, oxy = sc.observe()
    R0, t0, _, X0 = sc.start()
    r = BA.bundle_adjust(R0, t0, sc.intr, X0, oc, op, oxy, single_focal=False)
    assert np.array_equal(r["intr"], sc.intr)                 # single_focal off = the lens stays
    assert np.median(BA.reproj_px(r["R"], r["t"], r["intr"], r["X"], oc, op, oxy)) < 0.6
    old = BA.bundle_adjust(R0, t0, sc.intr, X0, oc, op, oxy, single_focal=False, solver="scipy", max_iter=300)
    assert np.abs(BA.centres(old["R"], old["t"]) - BA.centres(r["R"], r["t"])).max() < 0.01, "both solvers: the same minimum"

    class Stop(RuntimeError):
        pass

    def check():
        raise Stop()

    import types
    ticks = iter(range(10 ** 9))
    monkeypatch.setattr(BA, "time", types.SimpleNamespace(time=lambda: float(next(ticks))))   # every call = 1 s later
    with pytest.raises(Stop):
        BA.bundle_adjust(R0, t0, sc.intr, X0, oc, op, oxy, check=check)


# ----------------------------------------------------------------------------- epipolar test
def test_epipolar_test_marks_what_moves_on_its_own():
    rng = np.random.default_rng(0)
    h, w, f = 240, 320, 300.0
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (h, w + 40)).astype(np.float32), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    blob = cv2.normalize(cv2.GaussianBlur(rng.uniform(0, 255, (40, 40)).astype(np.float32), (0, 0), 1.5), None, 0, 255, cv2.NORM_MINMAX)
    grays, Rs, ts = [], [], []
    for i in range(3):                                        # a fronto-parallel wall 5 away, the camera slides in x
        g = tex[:, 6 * i:6 * i + w].copy()                    # static wall: 6 px / frame along the epipolar lines
        y0 = 80 + 7 * i                                       # a thing sliding DOWN the wall: off the epipolar lines
        g[y0:y0 + 40, 150 - 6 * i:190 - 6 * i] = blob
        grays.append(g.astype(np.uint8))
        Rs.append(np.eye(3)); ts.append(-np.array([6 * i * 5.0 / f, 0, 0]))
    intr = [f, f, (w - 1) / 2, (h - 1) / 2, 0, 0, 0, 0]
    dyn = BA.epipolar_moving(grays, np.array(Rs), np.array(ts), intr, 1.0)
    assert len(dyn) == 3 and dyn[1].shape == (h, w)
    obj = np.zeros((h, w), bool)
    obj[87:127, 144:184] = True
    assert dyn[1][95:119, 152:176].mean() > 0.8, "the moving thing is marked"
    assert dyn[1][~cv2.dilate(obj.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)].mean() < 0.03, "the wall is not"
    assert BA.sample_mask(dyn[1], np.array([[164.0, 107.0], [20.0, 20.0], [-5.0, 3.0]]), 1.0).tolist() == [True, False, False]


# ----------------------------------------------------------------------------- horizon of a frame
def _horizon_image(W=1920, H=1080, a=0.02, b=420.0, seed=0, flip=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    below = 1 / (1 + np.exp(np.clip(-(yy - (a * xx + b)) / 0.8, -50, 50)))      # a soft edge at y = a x + b
    sky, sea = (70.0, 200.0) if flip else (200.0, 70.0)
    img = sky + (sea - sky) * below + cv2.GaussianBlur(rng.normal(0, 25, (H, W)).astype(np.float32), (0, 0), 3)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("flip", [False, True])
def test_detect_horizon_finds_the_edge_subpixel(flip):
    g = _horizon_image(flip=flip)
    p = S.detect_horizon(g)
    assert p is not None and len(p) > 60
    assert np.abs(p[:, 1] - (0.02 * p[:, 0] + 420.0)).max() < 0.6
    assert p[:, 0].min() < 100 and p[:, 0].max() > 1820
    m = np.zeros(g.shape, bool)
    m[300:700, 800:1000] = True                               # a moving thing in front of the horizon
    pm = S.detect_horizon(g, m)
    assert pm is not None and not ((pm[:, 0] > 780) & (pm[:, 0] < 1020)).any()


def test_detect_horizon_none_without_a_straight_edge():
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (720, 1280)).astype(np.float32), (0, 0), 4)
    assert S.detect_horizon(cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) is None
    assert S.detect_horizon(np.full((720, 1280), 128, np.uint8)) is None
    yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)        # a strong CURVED edge (an arch) is not a horizon
    arch = np.where(yy > 300 + 0.0012 * (xx - 640) ** 2, 60, 210).astype(np.uint8)
    assert S.detect_horizon(arch) is None


# ----------------------------------------------------------------------------- planes
def test_fit_plane_ransac_orientation_and_mask():
    rng = np.random.default_rng(0)
    n = np.array([0.1, -0.97, 0.2]) / np.linalg.norm([0.1, -0.97, 0.2])
    e1, e2 = BA._tangent(n)
    on = rng.uniform(-10, 10, (400, 1)) * e1 + rng.uniform(-10, 10, (400, 1)) * e2 + 1.5 * n + rng.normal(0, 0.005, (400, 3))
    off = rng.uniform(-10, 10, (120, 3))
    P = np.concatenate([on, off])
    C = np.array([[0, 0, 0.0], [0.5, 0, 0.2]]) + 4.0 * n       # cameras on the +n side
    pl = S.fit_plane(P, centres=C)
    assert pl["inliers"] >= 400 and 0.7 < pl["share"] <= 1
    assert np.degrees(np.arccos(np.clip(np.array(pl["n"]) @ n, -1, 1))) < 0.2 and abs(pl["d"] + 1.5) < 0.02
    assert (C @ np.array(pl["n"]) + pl["d"] > 0).all(), "normal toward the cameras (Card.plane)"
    flipped = S.fit_plane(P, centres=C - 8.0 * n)
    assert np.array(flipped["n"]) @ n < -0.99
    # mask_fn: fit only a chosen part (here: a second, smaller plane that would lose the vote)
    other = rng.uniform(-3, 3, (80, 1)) * np.array([1.0, 0, 0]) + rng.uniform(-3, 3, (80, 1)) * np.array([0, 1.0, 0]) + np.array([0, 0, 30.0])
    P2 = np.concatenate([P, other])
    pl2 = S.fit_plane(P2, lambda X: X[:, 2] > 20, centres=C)
    assert abs(abs(pl2["n"][2]) - 1) < 1e-6 and pl2["inliers"] == 80
    assert S.fit_plane(P[:2]) is None
    T.Card.from_json({"id": "g", "name": "Ziemia", "plane": {"n": pl["n"], "d": pl["d"]}})   # fits the contract


def test_align_similarity():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (30, 3))
    R = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    B = 2.5 * A @ R.T + np.array([1.0, -2, 3])
    s, Ra, ta = BA.align_similarity(A, B)
    assert abs(s - 2.5) < 1e-9 and np.allclose(Ra, R) and np.allclose(ta, [1, -2, 3])


# ----------------------------------------------------------------------------- COLMAP text model
MODEL_CAMERAS = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 OPENCV 640 360 500.5 501.5 320 180 -0.05 0.01 0.001 -0.002
"""
MODEL_IMAGES = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3, mean observations per image: 1.33
1 1 0 0 0 0.1 0.2 0.3 1 f_000101.jpg
10.5 20.5 7 30.25 40.75 -1 50 60 9
2 0.7071067811865476 0 0.7071067811865476 0 -1 0 2 1 f_000102.jpg

3 0.5 0.5 0.5 0.5 0 0 0 1 f_000103.jpg
100 200 9 300.5 100.5 7
"""
MODEL_POINTS = """# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 2, mean track length: 2
7 1.5 -2.5 3.5 10 20 30 0.75 1 0 3 1
9 -4 5 6 0 0 0 1.25 1 2 3 0
"""


def test_read_colmap_text_model(tmp_path):
    for name, text in (("cameras.txt", MODEL_CAMERAS), ("images.txt", MODEL_IMAGES), ("points3D.txt", MODEL_POINTS)):
        (tmp_path / name).write_text(text, encoding="utf-8")
    m = CM.read_text_model(tmp_path)
    assert (m.model, m.width, m.height) == ("OPENCV", 640, 360)
    assert m.intrinsics().tolist() == [500.5, 501.5, 320, 180, -0.05, 0.01, 0.001, -0.002]
    assert sorted(m.R) == ["f_000101.jpg", "f_000102.jpg", "f_000103.jpg"]
    assert np.allclose(m.R["f_000101.jpg"], np.eye(3)) and m.t["f_000101.jpg"].tolist() == [0.1, 0.2, 0.3]
    assert np.allclose(m.R["f_000102.jpg"], Rotation.from_euler("y", 90, degrees=True).as_matrix(), atol=1e-12)
    assert np.allclose(m.R["f_000103.jpg"], Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix())
    xy, pid = m.obs["f_000101.jpg"]
    assert xy.tolist() == [[10.5, 20.5], [50, 60]] and pid.tolist() == [7, 9]      # the unmatched keypoint (-1) is left out
    assert m.obs["f_000102.jpg"][0].shape == (0, 2), "an image without points has an empty second line"
    assert m.obs["f_000103.jpg"][1].tolist() == [9, 7]
    assert m.point_ids.tolist() == [7, 9] and m.points.tolist() == [[1.5, -2.5, 3.5], [-4, 5, 6]] and m.errors.tolist() == [0.75, 1.25]
    assert CM.frame_of("f_000102.jpg") == 102 and CM.image_name(102) == "f_000102.jpg"
    # COLMAP pixels (centre of the first pixel at 0.5, working image = plate x 0.5) -> plate pixels
    assert CM.to_plate(m.intrinsics(), 0.5)[:4].tolist() == [1001.0, 1003.0, 639.5, 359.5]
    assert CM.xy_to_plate(np.array([[0.25, 0.25], [320.0, 180.0]]), 0.5).tolist() == [[0.0, 0.0], [639.5, 359.5]]
    other = CM.ColmapModel("SIMPLE_RADIAL", 10, 10, np.array([100.0, 5, 5, 0.1]))
    assert other.intrinsics().tolist() == [100, 100, 5, 5, 0.1, 0, 0, 0]
    with pytest.raises(CM.SolveError, match="nieobsługiwany model kamery"):
        CM.ColmapModel("FOV", 10, 10, np.zeros(5)).intrinsics()


# ----------------------------------------------------------------------------- the COLMAP process
def _script(tmp_path, body):
    p = tmp_path / "fake_colmap.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_colmap_runner_runs_logs_fails_and_cancels(tmp_path):
    ok = _script(tmp_path, "import sys\nprint('ARGS', ' '.join(sys.argv[1:]))\nsys.exit(3 if '--fail' in sys.argv else 0)\n")
    log = tmp_path / "colmap.log"
    r = CM.ColmapRunner(Path(sys.executable), log)
    r.run(str(ok), {"database_path": tmp_path / "db", "SiftExtraction.max_num_features": 16000})
    text = log.read_text(encoding="utf-8")
    assert "--database_path" in text and "--SiftExtraction.max_num_features 16000" in text and "ARGS" in text
    with pytest.raises(CM.SolveError, match=r"COLMAP \(.*\) zakończył się błędem 3"):
        r.run(str(ok), {"fail": 1})
    slow = _script(tmp_path, "import time\ntime.sleep(60)\n")

    class Stop(RuntimeError):
        pass

    def cancel():
        raise Stop()

    t0 = time.time()
    with pytest.raises(Stop):
        r.run(str(slow), {}, cancel)
    assert time.time() - t0 < 20, "the process is killed, not waited for"
    assert CM.ColmapRunner(tmp_path / "missing.exe").version() == "COLMAP (wersja nieznana)"


def test_colmap_missing_gives_the_gate_reason(monkeypatch, tmp_path):
    from engine import stage_models
    monkeypatch.setattr(stage_models, "cleanup3d_solve", lambda cfg: {"ok": False, "exe": None, "reason": "Brak programu COLMAP (CUDA) – Doktor → „Zainstaluj COLMAP (CUDA)”"})
    monkeypatch.delenv("MD_COLMAP_EXE", raising=False)
    with pytest.raises(S.SolveError, match="Brak programu COLMAP"):
        CM.find_exe({})
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    with pytest.raises(S.SolveError, match="Doktor"):
        S.solve_camera(work, [1, 2, 3], lambda f: tmp_path / f"{f}.png", cfg={})
    assert not T.camera_file(work).exists()
    exe = tmp_path / "colmap.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("MD_COLMAP_EXE", str(exe))
    assert CM.find_exe({}) == exe                             # the test / external build path
    monkeypatch.setattr(stage_models, "cleanup3d_solve", lambda cfg: {"ok": True, "exe": str(tmp_path / "doc.exe"), "reason": ""})
    assert CM.find_exe({}) == tmp_path / "doc.exe"            # the Doctor's install wins


# ----------------------------------------------------------------------------- orchestration with a fake COLMAP
FRAMES = list(range(101, 111))


class Shot:
    """A synthetic shot on disk: plates (640 x 360), a COLMAP-like model for the fake runner, tracks for the fake tracker."""

    def __init__(self, tmp_path, horizon=False, seed=0):
        self.sc = sc = Scene(n_cam=len(FRAMES), n_pts=320, W=640, H=360, f=560.0, k1=-0.04, k2=0.0, seed=seed)
        self.dir = tmp_path / "plates"
        self.dir.mkdir()
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:sc.H, 0:sc.W]
        rays = np.concatenate([BA.undistort(np.stack([xx.ravel(), yy.ravel()], 1).astype(float), sc.intr), np.ones((sc.W * sc.H, 1))], 1)
        for i, f in enumerate(FRAMES):
            img = cv2.GaussianBlur(rng.uniform(0, 255, (sc.H, sc.W)).astype(np.float32), (0, 0), 1.2)
            img = cv2.normalize(img, None, 90, 170, cv2.NORM_MINMAX)
            if horizon:                                        # sky above the level plane's line at infinity, sea below
                sky = ((rays @ sc.R[i]) @ UP > 0).reshape(sc.H, sc.W)
                img = cv2.GaussianBlur(np.where(sky, 215.0, 60.0).astype(np.float32), (0, 0), 0.8) + (img - 130) * 0.15
            cv2.imwrite(str(self.dir / f"p_{f:04d}.png"), cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        self.work = tmp_path / "shot" / "matte_desk"
        self.work.mkdir(parents=True)

    def path(self, f):
        return self.dir / f"p_{f:04d}.png"


class FakeColmap:
    """Stands in for colmap.exe: records the commands, checks the workspace, writes a noisy text model at the working scale."""

    def __init__(self, shot, scale, drop=()):
        self.shot, self.scale, self.drop = shot, scale, set(drop)
        self.calls, self.seen = [], {}

    def version(self):
        return "COLMAP 9.9 (fake)"

    def run(self, command, args, cancel_check=None):
        self.calls.append(command)
        if cancel_check:
            cancel_check()
        if command == "feature_extractor":
            imgs = sorted(Path(args["image_path"]).glob("*.jpg"))
            self.seen = {"images": [p.name for p in imgs], "size": cv2.imread(str(imgs[0])).shape[:2],
                         "mask_path": args.get("ImageReader.mask_path"), "model": args["ImageReader.camera_model"],
                         "single": args["ImageReader.single_camera"], "gpu": args["FeatureExtraction.use_gpu"]}
            if self.seen["mask_path"]:
                self.seen["masks"] = {p.name: cv2.imread(str(p), 0) for p in sorted(Path(self.seen["mask_path"]).glob("*.png"))}
        elif command == "sequential_matcher":
            self.seen["overlap"] = (args["SequentialMatching.overlap"], args["SequentialMatching.quadratic_overlap"])
        elif command == "mapper":
            (Path(args["output_path"]) / "0").mkdir()
        elif command == "model_converter":
            assert args["output_type"] == "TXT"
            self._write(Path(args["output_path"]))

    def _write(self, out):
        sc, s = self.shot.sc, self.scale
        R0, t0, intr0, X0 = sc.start(f_scale=1.03, rot_deg=0.2, c_sig=0.01, x_sig=0.03)
        oc, op, oxy = sc.observe(noise=0.25)
        intr0[4] = sc.intr[4] + 0.01                           # COLMAP's own estimate of k1: close, not exact
        i = intr0.copy()                                       # plate px -> COLMAP px of the working image
        i[0], i[1], i[2], i[3] = i[0] * s, i[1] * s, (i[2] + 0.5) * s, (i[3] + 0.5) * s
        (out / "cameras.txt").write_text(f"# cameras\n1 OPENCV {round(sc.W * s)} {round(sc.H * s)} " + " ".join(f"{v:.10g}" for v in i) + "\n")
        lines = ["# images"]
        for c, f in enumerate(FRAMES):
            if f in self.drop:
                continue
            q = Rotation.from_matrix(R0[c]).as_quat()
            lines.append(f"{c + 1} {q[3]:.12g} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} " + " ".join(f"{v:.12g}" for v in t0[c]) + f" 1 {CM.image_name(f)}")
            m = oc == c
            lines.append(" ".join(f"{(x + 0.5) * s:.6f} {(y + 0.5) * s:.6f} {p + 1}" for (x, y), p in zip(oxy[m], op[m])) + " 5.0 6.0 -1")
        (out / "images.txt").write_text("\n".join(lines) + "\n")
        (out / "points3D.txt").write_text("# points\n" + "\n".join(
            f"{j + 1} {x[0]:.10g} {x[1]:.10g} {x[2]:.10g} 128 128 128 0.5 1 0" for j, x in enumerate(X0)) + "\n")


class FakeTracker:
    """Stands in for CoTracker3: its own static points seen by the true cameras + a few that move on their own."""

    def __init__(self, shot, n=90, moving=12):
        self.shot, self.n, self.moving, self.calls = shot, n, moving, []

    def track(self, frames, queries, cancel_check=None):
        sc = self.shot.sc
        self.calls.append((len(frames), frames[0].shape, np.asarray(queries).copy()))
        rng = np.random.default_rng(9)
        X = np.stack([rng.uniform(-7, 7, self.n), rng.uniform(-1, 1.5, self.n), rng.uniform(5, 22, self.n)], 1)
        tr = np.stack([BA.project(sc.R[c], sc.t[c], X, sc.intr) for c in range(len(frames))]) + rng.normal(0, 0.2, (len(frames), self.n, 2))
        tr[:, :self.moving, 1] += 5.0 * np.arange(len(frames))[:, None]      # down: across the epipolar lines
        vis = (tr[..., 0] > 0) & (tr[..., 0] < sc.W) & (tr[..., 1] > 0) & (tr[..., 1] < sc.H)
        from engine.tracking import TrackResult
        return TrackResult(tracks=tr.astype(np.float32), visible=vis, engine="fake")


@pytest.mark.parametrize("horizon", [False, True])
def test_solve_camera_end_to_end_with_fake_colmap(tmp_path, horizon):
    shot = Shot(tmp_path, horizon=horizon)
    sc = shot.sc
    runner, tracker = FakeColmap(shot, 0.5), FakeTracker(shot)
    mask = np.zeros((sc.H, sc.W), np.uint8)
    mask[200:300, 100:220] = 255
    stages = []
    cam = S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, moving_masks={f: mask for f in FRAMES[:5]},
                         progress=lambda label, done, total: stages.append(label), runner=runner, tracker=tracker,
                         options={"max_side": 320})
    # stages in order, COLMAP commands in order, the workspace as COLMAP expects it
    order = [s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s]
    assert order[:5] == [S.STAGES[k] for k in ("prepare", "features", "matching", "mapper", "tracks")]
    assert order[5] == S.STAGES["bundle"] and order[-1] == S.STAGES["report"]
    assert runner.calls == ["feature_extractor", "sequential_matcher", "mapper", "model_converter"]
    assert runner.seen["images"] == [CM.image_name(f) for f in FRAMES] and runner.seen["size"] == (180, 320)
    assert (runner.seen["model"], runner.seen["single"], runner.seen["gpu"], runner.seen["overlap"]) == ("OPENCV", 1, 1, (9, 0))
    # every pair of a short shot is matched (quadratic_overlap = only i + 2^k: starved the mapper on sh060)
    assert (CM.default_overlap(57), CM.default_overlap(300), CM.default_overlap(3)) == (56, 20, 2)
    mk = runner.seen["masks"]
    assert sorted(mk) == [CM.image_name(f) + ".png" for f in FRAMES]
    m0, m9 = mk[CM.image_name(101) + ".png"], mk[CM.image_name(110) + ".png"]
    assert m0.shape == (180, 320) and m0[125, 80] == 0 and m0[20, 300] == 255 and (m9 == 255).all()   # 0 = ignore
    assert m0[99, 80] == 0 and m0[125, 49] == 0 and m0[92, 80] == 255, "the mask is dilated (41 px at 3840 wide)"
    assert not list(T.camera_file(shot.work).parent.glob("md_solve_*")), "the temporary workspace is removed"
    # the tracker got small RGB frames and a query grid outside the mask
    nfr, shape, q = tracker.calls[0]
    assert nfr == len(FRAMES) and shape == (360, 640, 3) and len(q) > 200
    q0 = q[q[:, 0] == 0]
    assert not ((q0[:, 1] > 100) & (q0[:, 1] < 220) & (q0[:, 2] > 200) & (q0[:, 2] < 300)).any()
    assert len(np.unique(q[:, 0])) == 7, "queries in 7 frames spread over the shot"
    # the Camera3D contract: plate frame numbers, plate pixels, OpenCV distortion, saved
    assert cam.frames == FRAMES and (cam.width, cam.height) == (640, 360)
    assert cam.K.shape == (3, 3) and cam.dist.shape == (4,) and cam.K[0, 0] == cam.K[1, 1]
    R = np.array([cam.R[f] for f in FRAMES])
    t = np.array([cam.t[f] for f in FRAMES])
    rot, cen = _pose_errors(R, t, sc)
    r = cam.report
    print(f"horizon={horizon}: f {cam.K[0, 0]:.2f} (560), k1 {cam.dist[0]:.4f} (-0.04), rot {rot:.3f} deg, centre {cen:.4f}, "
          f"reproj {r['reproj_px']} / p95 {r['reproj_px_p95']}, tracks {r['tracks']}/{r['tracks_total']}, rounds {r['rounds']}, "
          f"moving {r.get('moving_share')}, horizon {r['horizon_info']}")
    # orchestration test: 10 small frames, the plates are noise unrelated to the points (accuracy: the tests above);
    # without the horizon k1 is weakly determined by so little data
    assert abs(cam.K[0, 0] / 560.0 - 1) < 0.015 and abs(cam.dist[0] + 0.04) < (0.01 if horizon else 0.02)
    assert abs(cam.K[0, 2] - 319.5) < 1e-6 and abs(cam.K[1, 2] - 179.5) < 1e-6        # COLMAP's 0.5 px offset undone
    assert rot < 0.1 and cen < 0.03
    saved = store.load_camera(shot.work)
    assert saved.frames == FRAMES and np.allclose(saved.K, cam.K) and np.allclose(saved.R[105], cam.R[105])
    assert saved.source == "colmap+cotracker3"
    from engine import contract
    assert not contract.validate(json.loads(json.dumps(cam.to_json()))), "cams.json validates against the data contract"
    # report fields
    for key in ("reproj_px", "reproj_px_p95", "focal_px", "k1", "k2", "points", "tracks", "observations", "worst_frames",
                "frame_reproj_px", "seconds", "colmap_version", "registered", "rounds", "horizon_info"):
        assert key in r, key
    assert r["colmap_version"] == "COLMAP 9.9 (fake)" and r["registered"] == 10 and r["interpolated_frames"] == []
    assert 0 < r["reproj_px"] < 0.8 and r["reproj_px"] <= r["reproj_px_p95"] and r["focal_px"] == round(cam.K[0, 0], 2)
    assert r["points"] > 200 and 55 <= r["tracks"] <= 78 and r["tracks_total"] == 90 and r["tracks_dropped"] + r["tracks_short"] >= 12
    assert len(r["worst_frames"]) == 5 and set(r["worst_frames"][0]) == {"frame", "err_px"}
    assert r["worst_frames"][0]["frame"] in FRAMES and r["worst_frames"][0]["err_px"] >= r["worst_frames"][-1]["err_px"]
    assert set(r["seconds"]) >= {"prepare", "colmap_features", "colmap_matching", "colmap_mapper", "cotracker3", "bundle", "epipolar", "total"}
    # what the cards step needs
    pts = S.load_points(shot.work)
    assert pts.shape == (r["points"], 3) and S.load_points(shot.work, tracks=True).shape == (r["points"] + r["tracks"], 3)
    planes = S.propose_planes(shot.work)
    g = planes["ground"]
    Cc = np.array([cam.centre(f) for f in FRAMES])
    assert g is not None and g["share"] > 0.6 and (Cc @ np.array(g["n"]) + g["d"] > 0).all()
    # the ground of the scene is y = 1.5 in the true frame; in the solve's frame the cameras are ~1.5 above it (scale ~1)
    assert abs(np.median(Cc @ np.array(g["n"]) + g["d"]) - 1.5) < 0.1
    if horizon:
        hi = r["horizon_info"]
        assert hi["used"] and hi["frames"] == 10 and r["horizon_px"] == hi["res_px_median"] < 1.0
        _, Ra, _ = BA.align_cameras(R, t, sc.R, sc.t)
        assert np.degrees(np.arccos(np.clip((Ra @ np.array(r["up"])) @ UP, -1, 1))) < 0.5
        assert planes["up"] == r["up"]
        hz = S.load_horizon(shot.work)
        assert sorted(hz) == FRAMES and hz[101].shape[1] == 2
        hu = S.horizon_up(cam, hz)
        assert np.degrees(np.arccos(np.clip(np.array(hu["up"]) @ np.array(r["up"]), -1, 1))) < 0.2 and hu["res_px_median"] < 1.0
        sea = S.sea_plane(cam, pts, r["up"], band_px=(5.0 * 6, 150.0 * 6))      # 640 px plate: the band is given for 3840
        assert sea is not None and sea["n"] == [float(v) for v in np.array(r["up"]) / np.linalg.norm(r["up"])] and sea["points"] >= 5
    else:
        assert not r["horizon_info"]["used"] and "horizon_px" not in r and planes["sea"] is None and planes["up"] is None
        assert S.load_horizon(shot.work) is None


def test_solve_without_tracker_single_round_and_free_lens(tmp_path):
    shot = Shot(tmp_path)

    class NoTracks:
        def track(self, frames, queries, cancel_check=None):
            from engine.tracking import TrackResult
            return TrackResult(np.zeros((len(frames), 0, 2), np.float32), np.zeros((len(frames), 0), bool), "fake")

    cam = S.solve_camera(shot.work, FRAMES, shot.path, cfg={"cleanup3d_solve": {"rounds": 1}}, single_focal=False,
                         horizon=False, runner=FakeColmap(shot, 1.0), tracker=NoTracks(), save=False)
    assert cam.source == "colmap" and cam.report["tracks"] == 0 and cam.report["rounds"] == 1
    assert cam.K[0, 0] != cam.K[1, 1] and cam.report["single_focal"] is False       # the lens as COLMAP gave it
    assert not T.camera_file(shot.work).exists(), "save=False writes nothing"
    assert cam.report["reproj_px"] < 6


def test_solve_too_few_registered_frames_and_interpolated_ones(tmp_path):
    shot = Shot(tmp_path)
    with pytest.raises(S.SolveError, match="COLMAP zarejestrował tylko 3 z 10 klatek"):
        S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, runner=FakeColmap(shot, 1.0, drop=FRAMES[3:]),
                       tracker=FakeTracker(shot), horizon=False)
    assert not T.camera_file(shot.work).exists()
    cam = S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, runner=FakeColmap(shot, 1.0, drop=[104, 110]),
                         tracker=FakeTracker(shot), horizon=False)
    assert cam.frames == FRAMES and cam.report["registered"] == 8 and cam.report["interpolated_frames"] == [104, 110]
    R = np.array([cam.R[f] for f in FRAMES])
    t = np.array([cam.t[f] for f in FRAMES])
    rot, cen = _pose_errors(R, t, shot.sc)
    print(f"interpolated frames: rot {rot:.3f} deg, centre {cen:.4f}")
    assert rot < 0.3 and cen < 0.06, "frames COLMAP skipped are solved from the tracks"
    with pytest.raises(S.SolveError, match="co najmniej 3"):
        S.solve_camera(shot.work, [1, 2], shot.path, cfg={}, runner=FakeColmap(shot, 1.0))
    with pytest.raises(S.SolveError, match="inny rozmiar"):
        bad = shot.dir / "bad.png"
        cv2.imwrite(str(bad), np.zeros((100, 100, 3), np.uint8))
        S.solve_camera(shot.work, FRAMES, lambda f: bad if f == 103 else shot.path(f), cfg={}, runner=FakeColmap(shot, 1.0))


def test_solve_cancel_at_every_stage_leaves_nothing(tmp_path):
    shot = Shot(tmp_path)

    class Stop(RuntimeError):
        pass

    for at in ("prepare", "mapper", "tracks", "bundle", "epipolar"):
        seen = []

        def cancel():
            if seen and seen[-1] == S.STAGES[at]:
                raise Stop()

        with pytest.raises(Stop):
            S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, progress=lambda label, d, t: seen.append(label),
                           cancel_check=cancel, runner=FakeColmap(shot, 1.0), tracker=FakeTracker(shot), horizon=False)
        assert seen[-1] == S.STAGES[at]
        assert not T.camera_file(shot.work).exists() and not list(T.camera_file(shot.work).parent.glob("md_solve_*"))
    with pytest.raises(S.SolveCancelled, match="Solve kamery przerwany"):       # a cancel_check that answers True
        S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, cancel_check=lambda: True, runner=FakeColmap(shot, 1.0))
    assert issubclass(S.SolveCancelled, S.SolveError)


def test_plates_are_only_read(tmp_path):
    shot = Shot(tmp_path)
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in shot.dir.iterdir()}
    S.solve_camera(shot.work, FRAMES, shot.path, cfg={}, runner=FakeColmap(shot, 1.0), tracker=FakeTracker(shot), horizon=False)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) for p in shot.dir.iterdir()}
    assert before == after
    made = sorted(p.name for p in T.camera_file(shot.work).parent.iterdir())
    assert made == ["cams.json", "points.npy", "track_points.npy"]
    assert T.camera_file(shot.work).parent == shot.work.parent / "inpaint" / "camera3d"


def test_read_plate_u8_formats(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (20, 30, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), a)
    assert np.array_equal(S.read_plate_u8(tmp_path / "a.png"), a)
    cv2.imwrite(str(tmp_path / "b.png"), a.astype(np.uint16) * 257)
    assert np.array_equal(S.read_plate_u8(tmp_path / "b.png"), a)
    cv2.imwrite(str(tmp_path / "g.png"), a[..., 0])
    assert S.read_plate_u8(tmp_path / "g.png").shape == (20, 30, 3)
    with pytest.raises(S.SolveError, match="Nie można odczytać klatki"):
        S.read_plate_u8(tmp_path / "missing.png")


def test_colmap_runner_is_a_registered_gpu_lease():
    """The lead adds this line to LEASED_CALLS (engine/gpu_manager.py); here: the call really runs inside GPU.use."""
    import ast
    src = (Path(CM.__file__)).read_text(encoding="utf-8")
    cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "ColmapRunner")
    fn = next(n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert [w for w in ast.walk(fn) if isinstance(w, ast.With) for it in w.items
            if isinstance(it.context_expr, ast.Call) and getattr(it.context_expr.func, "attr", None) == "use"]


# ----------------------------------------------------------------------------- real: sh060 vs the experiment's sfm8
EXP = Path(r"D:/MD_REAL_PROJECT/Documents\removers_test")
BEATCH = Path(r"D:/MD_REAL_PROJECT/Documents\shots\shots\matte_desk\source\sh060")
KEYS = [1, 8, 16, 21, 27, 32, 37, 39, 46, 50, 57]              # the experiment's key frames (outputs/sfm2/cards.json)
N_B, FW, FH, Q = 57, 3840, 2160, 0.25


class _Metrics:
    """The experiment's error measures, computed the same way for any solve (ports of solve_report.py, cards.py
    `card_map` and horizon3d.py; masks gpu_b and the per-frame horizon lines outputs/layers/horizon.json as there)."""

    def __init__(self):
        self.SW, self.SH = int(FW * Q), int(FH * Q)
        self.old = {int(k): v for k, v in json.load(open(EXP / "outputs" / "layers" / "horizon.json")).items()}
        self.gray, self.hz = {}, {}
        for f in range(1, N_B + 1):
            plate = cv2.imread(str(BEATCH / f"sh060_{f:04d}.png"))
            m = cv2.imread(str(EXP / "outputs" / "gpu_b" / "mask" / f"m_{f:04d}.png"), 0)
            g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
            self.gray[f] = (cv2.resize(cv2.GaussianBlur(g, (0, 0), 3), (self.SW, self.SH), interpolation=cv2.INTER_AREA),
                            cv2.resize(cv2.dilate(m, np.ones((61, 61), np.uint8)), (self.SW, self.SH), interpolation=cv2.INTER_NEAREST) > 0)
            self.hz[f] = self._frame_points(g, m > 0, f)

    def _frame_points(self, g, m, f):                         # horizon3d.py frame_points / edges
        a, b = self.old[f]
        md = cv2.dilate(m.astype(np.uint8), np.ones((1, 121), np.uint8)) > 0
        gy = np.abs(cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 2.5), cv2.CV_32F, 0, 1, ksize=5))
        xs = np.arange(20, FW - 20, 40, dtype=float)
        ys, st = [], []
        for x in xs:
            c = a * x + b
            lo, hi = int(max(0, c - 110)), int(min(FH, c + 111))
            prof = gy[lo:hi, max(0, int(x) - 15):int(x) + 16].mean(1)
            j = int(np.argmax(prof))
            ys.append(lo + j); st.append(prof[j] / (np.median(prof) + 1e-3))
        ys, st = np.array(ys, float), np.array(st, float)
        ok = np.array([not md[int(max(0, y - 100)):int(min(FH, y + 100)), int(x)].any() for x, y in zip(xs, ys)]) & (st > 6)
        X_, Y_ = xs[ok], ys[ok]
        keep = np.ones(len(X_), bool)
        for _ in range(4):
            c = np.polyfit(X_[keep], Y_[keep], 2)
            res = np.abs(np.polyval(c, X_) - Y_)
            keep = res < max(4.0, 2.5 * np.median(res[keep]))
        return np.stack([X_[keep], Y_[keep]], 1)

    def horizon(self, K, dist, RS, points):
        """horizon3d.py: (up, per-frame median residual px {frame: r}, h_sea)."""
        Wd = []
        for f, P in self.hz.items():
            und = cv2.undistortPoints(P[:, None].astype(np.float64), K, dist).reshape(-1, 2)
            w = np.concatenate([und, np.ones((len(und), 1))], 1) @ RS[f - 1]
            Wd.append(w / np.linalg.norm(w, axis=1, keepdims=True))
        Wd = np.vstack(Wd)
        up = np.linalg.eigh(Wd.T @ Wd)[1][:, 0]
        if up @ np.mean([RS[i].T @ np.array([0.0, -1.0, 0.0]) for i in range(N_B)], 0) < 0:
            up = -up
        res, quads = {}, {}
        xs = np.linspace(0, FW - 1, 25)
        for f in range(1, N_B + 1):
            n = RS[f - 1] @ up

            def gfun(y):
                u = cv2.undistortPoints(np.stack([xs, y], 1)[:, None].astype(np.float64), K, dist).reshape(-1, 2)
                return u[:, 0] * n[0] + u[:, 1] * n[1] + n[2]
            lo, hi = np.zeros(len(xs)), np.full(len(xs), float(FH - 1))
            glo, ghi = gfun(lo), gfun(hi)
            ok = glo * ghi <= 0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                gm = gfun(mid)
                same = gm * glo > 0
                lo, glo, hi = np.where(same, mid, lo), np.where(same, gm, glo), np.where(same, hi, mid)
            yp = 0.5 * (lo + hi)
            q = np.polyfit(xs[ok], yp[ok], 2)
            P = self.hz[f]
            res[f] = float(np.median(np.abs(np.polyval(q, P[:, 0]) - P[:, 1])))
            quads[f] = q
        i = N_B // 2
        Pc = points @ RS[i].T + self._ts[i]
        okz = Pc[:, 2] > 0.05
        uv = cv2.projectPoints(Pc[okz], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        dy = uv[:, 1] - np.polyval(quads[i + 1], uv[:, 0])
        sea = (dy > 5) & (dy < 150) & (uv[:, 0] > 0) & (uv[:, 0] < FW)
        return up, res, float(np.median((points[okz] @ up)[sea]))

    def key_error(self, K, dist, RS, TS, sand, sea):
        """solve_report.py: mean |key plate projected through the cards - frame plate| outside the masks, per key."""
        SW, SH = self.SW, self.SH
        X, Y = np.meshgrid((np.arange(SW, dtype=np.float32) + 0.5) / Q - 0.5, (np.arange(SH, dtype=np.float32) + 0.5) / Q - 0.5)
        und = cv2.undistortPoints(np.stack([X.ravel(), Y.ravel()], 1)[:, None].astype(np.float64), K, dist).reshape(SH, SW, 2)
        ray = np.concatenate([und, np.ones((SH, SW, 1))], -1).astype(np.float32)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        k1, k2, p1, p2 = dist

        def card_map(t, k):                                    # cards.py card_map: frame-t px -> frame-k px
            Rt, Tt, Rk, Tk = RS[t], TS[t], RS[k], TS[k]
            Ct = -Rt.T @ Tt
            w = ray @ Rt.astype(np.float32)

            def inv_of(nw, dw):
                den = w @ nw.astype(np.float32)
                zp = -(nw @ Ct + dw) / np.where(np.abs(den) < 1e-9, 1e-9, den)
                return np.where(zp > 0, 1.0 / np.maximum(zp, 1e-6), 0.0).astype(np.float32)
            inv = inv_of(*sand)
            inv = np.where(inv > 0, inv, inv_of(*sea))
            p = w @ Rk.T.astype(np.float32) + (Rk @ Ct + Tk).astype(np.float32)[None, None, :] * inv[..., None]
            z = np.maximum(p[..., 2], 1e-4)
            x, y = p[..., 0] / z, p[..., 1] / z
            r2 = x * x + y * y
            rad = 1 + k1 * r2 + k2 * r2 * r2
            xd, yd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            mx, my = (xd * fx + cx).astype(np.float32), (yd * fy + cy).astype(np.float32)
            bad = p[..., 2] <= 1e-4
            mx[bad], my[bad] = -1e4, -1e4
            return mx, my

        def err(a, j):
            ga, ma = self.gray[a]
            gj, mj = self.gray[j]
            mx, my = card_map(j - 1, a - 1)
            mx, my = (mx + 0.5) * Q - 0.5, (my + 0.5) * Q - 0.5
            wa = cv2.remap(ga, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            wm = cv2.remap(ma.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
            sel = (wa >= 0) & ~wm & ~mj
            return float(np.abs(wa - gj)[sel].mean()) if sel.sum() > 1000 else None

        kerr, frame_err = {}, {}
        for i, k in enumerate(KEYS):
            lo = KEYS[i - 1] if i else 1
            hi = KEYS[i + 1] if i + 1 < len(KEYS) else N_B
            es = {f: err(k, f) for f in range(lo, hi + 1) if f != k}
            es = {f: e for f, e in es.items() if e is not None}
            kerr[k] = float(np.mean(list(es.values())))
            for f, e in es.items():
                frame_err[f] = max(frame_err.get(f, 0), e)
        return kerr, frame_err

    def report(self, K, dist, RS, TS, points, all_points, plane=None):
        """{key_err_mean, key_err, horizon_px_median, horizon_px_worst, ...} for one solve. `points`: its sparse SIFT
        points (the ground plane = `fit_plane` on them, as joint_ba.py; `plane` overrides it); `all_points`: SIFT +
        track points (the sea level, as horizon3d.py reads points.npy)."""
        self._ts = TS
        C = np.array([-RS[i].T @ TS[i] for i in range(N_B)])
        pl = plane or S.fit_plane(points, centres=C)
        sn, sd = np.array(pl["n"], np.float64), float(pl["d"])
        if np.mean(C @ sn + sd) < 0:
            sn, sd = -sn, -sd
        up, hres, h_sea = self.horizon(K, dist, RS, all_points)
        kerr, ferr = self.key_error(K, dist, RS, TS, (sn, sd), (up, -h_sea))
        return {"key_err_mean": round(float(np.mean(list(kerr.values()))), 3), "key_err": {k: round(v, 2) for k, v in kerr.items()},
                "key_err_worst": max(kerr.items(), key=lambda kv: kv[1]),
                "worst_frames_key_err": [(f, round(e, 2)) for f, e in sorted(ferr.items(), key=lambda kv: -kv[1])[:5]],
                "horizon_px_median": round(float(np.median(list(hres.values()))), 3),
                "horizon_px_worst": max(hres.items(), key=lambda kv: kv[1]), "up": up.tolist(), "h_sea": h_sea}


@pytest.mark.real
def test_real_sh060_solve_vs_experiment_sfm8(tmp_path):
    """57 frames 3840 x 2160: COLMAP (MD_COLMAP_EXE) + CoTracker3 + joint adjustment, into a temp work dir; compared
    with the experiment's sfm8 (key projection error 4.00, horizon median 1.30 px). Thresholds = the plan's target."""
    exe = os.environ.get("MD_COLMAP_EXE", "")
    if not (exe and Path(exe).is_file() and BEATCH.is_dir() and (EXP / "outputs" / "sfm8" / "cams.npz").is_file()):
        pytest.skip("needs MD_COLMAP_EXE, the sh060 plate and the experiment's outputs/sfm8")
    from engine.config import load_config
    work = tmp_path / "sh060" / "matte_desk"
    work.mkdir(parents=True)
    frames = list(range(1, N_B + 1))
    stages = []

    def progress(label, done, total):
        if not stages or stages[-1][0] != label:
            stages.append((label, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)

    t0 = time.time()
    cam = S.solve_camera(work, frames, lambda f: BEATCH / f"sh060_{f:04d}.png", cfg=load_config(),
                         moving_masks=lambda f: cv2.imread(str(EXP / "outputs" / "gpu_d" / "mask" / f"m_{f:04d}.png"), 0),
                         progress=progress)
    secs = time.time() - t0
    r = cam.report
    print("SOLVE REPORT", json.dumps({k: v for k, v in r.items() if k != "frame_reproj_px"}, ensure_ascii=False), flush=True)
    assert cam.frames == frames and (cam.width, cam.height) == (FW, FH) and r["registered"] == N_B

    ref = np.load(EXP / "outputs" / "sfm8" / "cams.npz")
    Kr, Rr, tr, dr = ref["K"], ref["R"], ref["t"], ref["dist"]
    Rm = np.array([cam.R[f] for f in frames])
    tm = np.array([cam.t[f] for f in frames])
    Cm, Cr = BA.centres(Rm, tm), BA.centres(Rr, tr)
    s, Ra, ta = BA.align_cameras(Rm, tm, Rr, tr)
    cen = np.linalg.norm((s * Cm @ Ra.T + ta) - Cr, axis=1)
    path = np.linalg.norm(np.diff(Cr, axis=0), axis=1).sum()
    rot = np.array([np.degrees(np.linalg.norm(Rotation.from_matrix(Rm[i] @ Ra.T @ Rr[i].T).as_rotvec())) for i in range(N_B)])
    M = _Metrics()
    ref_pts = np.load(EXP / "outputs" / "sfm8" / "points.npy").astype(np.float64)             # 7500 SIFT + tracks
    ref_plane = json.load(open(EXP / "outputs" / "sfm8" / "plane.json"))
    rep_ref = M.report(Kr, dr, Rr, tr, ref_pts[:7500], ref_pts, plane=ref_plane)
    rep_me = M.report(cam.K, cam.dist, Rm, tm, S.load_points(work), S.load_points(work, tracks=True))
    out = {"seconds_total": round(secs, 1), "seconds": r["seconds"], "colmap_version": r["colmap_version"],
           "focal_px": r["focal_px"], "focal_ref": round(float(Kr[0, 0]), 2), "k1": r["k1"], "k2": r["k2"],
           "k1_ref": round(float(dr[0]), 5), "k2_ref": round(float(dr[1]), 5),
           "reproj_px": r["reproj_px"], "reproj_px_p95": r["reproj_px_p95"],
           "sift_reproj_axis_px_median": r["sift_reproj_axis_px_median"], "points": r["points"], "tracks": r["tracks"],
           "tracks_total": r["tracks_total"], "horizon_info": r["horizon_info"], "moving_share": r.get("moving_share"),
           "worst_frames": r["worst_frames"],
           "vs_sfm8_centre_err_over_path_median": round(float(np.median(cen) / path), 5),
           "vs_sfm8_centre_err_over_path_max": round(float(cen.max() / path), 5),
           "vs_sfm8_rot_deg_median": round(float(np.median(rot)), 4), "vs_sfm8_rot_deg_max": round(float(rot.max()), 4),
           "metrics_sfm8": rep_ref, "metrics_new": rep_me}
    print("REAL RESULT", json.dumps(out, ensure_ascii=False, default=str), flush=True)
    dump = os.environ.get("MD_SOLVE_REAL_OUT", "")
    if dump:
        Path(dump).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    planes = S.propose_planes(work)
    assert planes["ground"] is not None and planes["sea"] is not None
    # the port of the experiment's measures reproduces its own report for sfm8 (4.00 / 1.30)
    assert abs(rep_ref["key_err_mean"] - 4.0) < 0.1 and abs(rep_ref["horizon_px_median"] - 1.3) < 0.1
    assert r["tracks"] > 0 and r["horizon_info"]["used"]
    assert abs(r["focal_px"] / float(Kr[0, 0]) - 1) < 0.05
    # the plan's target (docs/PLAN_CLEANUP_3D.md, P10): not worse than sfm8
    assert rep_me["key_err_mean"] <= 4.0, rep_me
    assert rep_me["horizon_px_median"] <= 1.3, rep_me
