# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D projection and compositing (package P6, engine/cleanup3d/render.py + the source „klucz” of
engine/cleanup.py) on a synthetic scene: a textured ground plane with distance haze + a sky, a camera that rotates and
translates, an object crossing the horizon that moves across the frame, keys = the true clean background at three
frames. The truth is rendered independently of the geometry module (own pinhole rays, own plane intersection).

Real test (`-m real`, needs the sh060 experiment in removers_test/): tools/c3d_from_experiment.py -> a temp work dir,
the 57 frames rendered, compared with the experiment's chain v3c (outputs/v3dW3c_b) by the flicker3d.py metric (ported
below) and by the picture difference, near the horizon and away from it."""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import cleanup as cu
from engine.cleanup3d import geom_gpu as KER
from engine.cleanup3d import keys as KS
from engine.cleanup3d import render as RD
from engine.cleanup3d import store
from engine.cleanup3d.geom import CardGeometry, GeomGpuWorker, check_worker
from engine.cleanup3d.types import Camera3D, Card, Key

W, H = 320, 200
KM = np.array([[260.0, 0, 160.0], [0, 260.0, 100.0], [0, 0, 1]])
FRAMES = list(range(1, 10))
KEY_FRAMES = [1, 5, 9]
GROUND_Y = 2.0                       # y down: the ground is 2 below the first camera
HAZE = np.array([0.72, 0.66, 0.60], np.float32)
KEY_SOFT = 12                                              # key_soft_px of the synthetic shots
SMALL = {"key_soft_px": float(KEY_SOFT), "key_grow_px": 0.0, "edge_px": 16.0, "edge_hard_px": 4.0, "keep_grow_px": 4.0, "tone_ring_px": 12.0}   # a 320 px frame
GREEN = np.array([0.1, 0.9, 0.1], np.float32)
RED = np.array([0.1, 0.1, 0.9], np.float32)


def _rot(yaw, pitch, roll=0.0):
    a, b, c = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def make_camera():
    R, t = {}, {}
    for i, f in enumerate(FRAMES):
        C = np.array([0.25 * i, -0.02 * i, 0.30 * i])             # right, up a little, forward
        Rf = _rot(1.5 * i - 6.0, 9.0 - 0.3 * i, 0.3 * i)            # yaw, pitch toward the ground, roll
        R[f], t[f] = Rf, -Rf @ C
    return Camera3D(W, H, KM.copy(), np.zeros(4), R, t, "synthetic")


def make_cards(seam=4.0):
    return [Card("ground", "piasek", "plane", {"n": [0, -1, 0], "d": GROUND_Y}, order=0, seam_px=seam),
            Card("sky", "niebo", "sky", order=1, seam_px=seam)]


def true_bg(cam, f, dy_ground=0.0):
    """The clean background of frame f, BGR float32 – rendered without engine.cleanup3d."""
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    r = np.stack([(xx - KM[0, 2]) / KM[0, 0], (yy - KM[1, 2]) / KM[1, 1], np.ones_like(xx)], -1)
    wd = r @ cam.R[f]                                               # R^T r per pixel
    C = cam.centre(f)
    hit = wd[..., 1] > 1e-6
    zp = (GROUND_Y + dy_ground - C[1]) / np.where(hit, wd[..., 1], 1.0)
    P = C + zp[..., None] * wd
    X, Z = P[..., 0], P[..., 2]
    g = np.stack([0.45 + 0.20 * np.sin(X * 4.1) * np.cos(Z * 3.3) + 0.08 * np.sin(X * 6.0 + Z * 5.0),
                  0.50 + 0.20 * np.cos(X * 2.7 + 1.0) * np.sin(Z * 4.4) + 0.08 * np.cos(X * 5.0 - Z * 6.5),
                  0.40 + 0.20 * np.sin(X * 3.5 + Z * 2.9)], -1)
    fog = np.exp(-np.maximum(Z, 0.0) / 6.0)[..., None]      # haze fixed in the world (not view-dependent): no detail
                                                            # finer than a pixel near the horizon
    ground = HAZE + (g - HAZE) * fog
    az = np.arctan2(wd[..., 0], wd[..., 2])
    el = np.arctan2(-wd[..., 1], np.hypot(wd[..., 0], wd[..., 2]))
    s = np.stack([0.55 + 0.25 * np.sin(az * 9.0) * np.cos(el * 11.0), 0.45 + 0.2 * np.cos(az * 7.0 + el * 13.0),
                  0.35 + 0.2 * np.sin(az * 12.0 - 0.5)], -1)
    up = (1.0 - np.exp(-np.maximum(el, 0.0) / 0.08))[..., None]
    sky = HAZE + (s - HAZE) * up
    return np.where(hit[..., None], ground, sky).astype(np.float32)


def obj_alpha(f):
    """The object to remove: a tall ellipse crossing the horizon, moving right."""
    yy, xx = np.mgrid[0:H, 0:W]
    cx = 70.0 + 22.0 * (f - 1)
    return ((((xx - cx) / 16.0) ** 2 + ((yy - 95.0) / 46.0) ** 2) < 1.0).astype(np.float32)


def keep_alpha(f):
    """An object that stays, in front: a post fixed in the picture (so it slides over the background)."""
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx >= 150) & (xx < 166) & (yy >= 70) & (yy < 170)).astype(np.float32)


class Shot:
    """The synthetic shot written into the contract layout of a work dir."""

    def __init__(self, tmp_path, keep=False, key_gain=0.0, sky_keys=True, seam=4.0):
        self.work = Path(tmp_path) / "shot" / "matte_desk"
        self.work.mkdir(parents=True)
        self.cam, self.cards, self.keep = make_camera(), make_cards(seam), keep
        store.save_camera(self.work, self.cam)
        store.save_cards(self.work, self.cards)
        self.bg = {f: true_bg(self.cam, f) for f in FRAMES}
        for f in KEY_FRAMES:
            key = self.bg[f] + key_gain
            if keep:
                key = np.where(keep_alpha(f)[..., None] > 0, GREEN, key)
            for c in self.cards:
                if c.kind == "sky" and not sky_keys:
                    continue
                KS.add_version(self.work, c.id, f, key.astype(np.float32), "test", "exr")

    def plate(self, f):
        p = np.where(obj_alpha(f)[..., None] > 0, RED, self.bg[f])
        if self.keep:
            p = np.where(keep_alpha(f)[..., None] > 0, GREEN, p)
        return p.astype(np.float32)

    def clean(self, f):
        """The frame as it should come out (the kept object stays)."""
        return (np.where(keep_alpha(f)[..., None] > 0, GREEN, self.bg[f]) if self.keep else self.bg[f]).astype(np.float32)

    def render(self, frames=FRAMES, **s):
        st = {**SMALL, "remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0, "motion_blur": "off", **s}
        rj = RD.render(self.work, frames, plate=self.plate, remove_alphas=lambda f: [obj_alpha(f)],
                       keep_alphas=(lambda f: [keep_alpha(f)]) if self.keep else None, settings=st)
        out = Path(rj["dir"])
        comp = {f: cv2.imread(str(out / "comp" / f"c_{f:04d}.png")).astype(np.float32) / 255.0 for f in frames}
        src = {f: cv2.imread(str(out / "source" / f"s_{f:04d}.png"), 0) for f in frames}
        return rj, comp, src, out


def _shift(a, b, m):
    """Sub-pixel shift (px) of picture a against b inside mask m: global translation by Lucas-Kanade on the gray,
    iterated (a is re-sampled at the current estimate)."""
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]].astype(np.float32)
    d = np.zeros(2)
    for _ in range(8):
        wa = cv2.remap(ga, xx + np.float32(d[0]), yy + np.float32(d[1]), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gx, gy = cv2.Sobel(wa, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(wa, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        G = np.stack([gx[m], gy[m]], 1).astype(np.float64)
        d += np.linalg.solve(G.T @ G + 1e-9 * np.eye(2), G.T @ (gb[m] - wa[m]).astype(np.float64))
    return float(np.hypot(*d))


# ------------------------------------------------------------------------------------------------ crossfade
def test_key_pair_smoothstep_weights():
    ks = [Key("ground", f, [{"v": "v001", "ext": "png"}], "v001") for f in (1, 5, 9)] + [Key("ground", 7)]   # 7: no version
    pair = lambda f: [(k.frame, round(w, 5)) for k, w in RD.key_pair(f, ks)]
    assert pair(1) == [(1, 1.0), (5, 0.0)] and pair(5) == [(5, 1.0), (9, 0.0)] and pair(9) == [(9, 1.0)]
    assert pair(3) == [(1, 0.5), (5, 0.5)]
    assert pair(2) == [(1, 0.84375), (5, 0.15625)]                 # smoothstep(0.25) = 0.15625
    assert pair(7) == [(5, 0.5), (9, 0.5)]                         # the key without a version does not count
    assert pair(0) == [(1, 1.0)] and pair(12) == [(9, 1.0)] and RD.key_pair(3, []) == []


def test_crossfade_of_two_keys_and_coverage_border():
    cam, cards = make_camera(), make_cards()
    geo = CardGeometry(cam, cards)
    A, B = np.array([0.2, 0.4, 0.6], np.float32), np.array([0.8, 0.3, 0.1], np.float32)
    keys = {c.id: [Key(c.id, 1, [{"v": "v001"}], "v001"), Key(c.id, 5, [{"v": "v001"}], "v001")] for c in cards}
    read = lambda k: np.broadcast_to(A if k.frame == 1 else B, (H, W, 3)).copy()
    for f, w in ((1, 0.0), (2, 0.15625), (3, 0.5), (4, 0.84375), (5, 1.0)):
        fill, wgt = RD.project_frame(geo, cards, keys, f, read, settings=SMALL)
        mx, my = geo.frame_to_key(f, keys["ground"][0], cards[0])
        both = np.ones((H, W), bool)
        for k in keys["ground"]:                                   # pixels well inside both keys, away from the horizon seam
            for c in cards:
                mx, my = geo.frame_to_key(f, k, c)
                cw = geo.card_weight(c, f)
                both &= (cw < 1e-3) | ((mx > 20) & (mx < W - 21) & (my > 20) & (my < H - 21))
        assert both.sum() > 5000
        assert np.abs(fill[both] - (A * (1 - w) + B * w)).max() < 2e-3, f
        assert wgt[both].min() > 0.999
    # frame 3: a pixel only key 5 reaches takes key 5 alone (not half of it), with full weight
    fill, wgt = RD.project_frame(geo, cards, keys, 3, read, settings=SMALL)
    c1 = KER.coverage(KER.NP, *[m.astype(np.float64) for m in geo.frame_to_key(3, keys["ground"][0], cards[0])], W, H, 16.0)
    c5 = KER.coverage(KER.NP, *[m.astype(np.float64) for m in geo.frame_to_key(3, keys["ground"][1], cards[0])], W, H, 16.0)
    only5 = (c1 == 0) & (c5 == 1) & (geo.card_weight(cards[0], 3) > 0.999)
    assert only5.sum() > 100 and np.abs(fill[only5] - B).max() < 2e-3 and wgt[only5].min() > 0.999
    # ... and where no key reaches at all the weight is 0
    none = RD.project_frame(geo, cards, {"ground": keys["ground"][:1]}, 9, read, settings=SMALL)[1]
    assert (none == 0).mean() > 0.3 and (none > 0.999).any()


def test_roi_equals_the_full_frame_crop():
    cam, cards = make_camera(), make_cards()
    geo = CardGeometry(cam, cards)
    bg = {f: true_bg(cam, f) for f in KEY_FRAMES}
    keys = {c.id: [Key(c.id, f, [{"v": "v001"}], "v001") for f in KEY_FRAMES] for c in cards}
    full = RD.project_frame(geo, cards, keys, 4, lambda k: bg[k.frame], settings=SMALL)
    roi = (40, 30, 250, 180)
    part = RD.project_frame(geo, cards, keys, 4, lambda k: bg[k.frame], settings=SMALL, roi=roi)
    assert part[0].shape == (150, 210, 3)
    assert np.abs(part[0] - full[0][30:180, 40:250]).max() < 1e-4 and np.abs(part[1] - full[1][30:180, 40:250]).max() < 1e-4


# ------------------------------------------------------------------------------------------------ the scene
def test_render_equals_true_background_and_patch_is_locked(tmp_path):
    shot = Shot(tmp_path)
    rj, comp, src, out = shot.render()
    assert rj["version"] == "v001" and out == cu.cleanup_dir(shot.work) / "v001" and not (out / "_fill").exists()
    assert rj["counts"]["klucz"] > 0 and rj["counts"]["OpenCV"] == 0 and rj["counts"]["LaMa"] == 0
    assert rj["c3d"]["schema"] == RD.SCHEMA and [c["id"] for c in rj["c3d"]["cards"]] == ["ground", "sky"]
    assert [k["frame"] for k in rj["c3d"]["cards"][0]["keys"]] == KEY_FRAMES
    worst, shifts = 0.0, []
    for f in FRAMES:
        m = obj_alpha(f) > 0
        mask = cu.removal_mask([obj_alpha(f)], 4.0, 3.0)
        # the patch = the removal mask + a soft outward ramp of key_soft_px (2026-11-10: no hard edge into the plate)
        ramp = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * KEY_SOFT + 3,) * 2, np.uint8)) > 0
        assert (src[f][mask > 0] == cu.SRC_KEY).all() and (src[f][~ramp] == cu.SRC_PLATE).all()
        err = np.abs(comp[f] - shot.bg[f])
        assert err[~ramp].max() <= 0.5 / 255 + 1e-6                          # the plate untouched outside the patch
        assert err[ramp & (mask == 0)].max() < 0.03                          # the ramp: projected key = background
        worst = max(worst, float(err[m].max()))
        assert err[m].mean() < 0.004, (f, err[m].mean())
        shifts.append(_shift(comp[f], shot.bg[f], m))
    assert worst < 0.03, worst                                              # 8-bit + cubic resampling of the keys
    assert max(shifts) < 0.1, shifts                                        # the patch does not move against the background
    # the patch: premultiplied RGBA, alpha = the mask
    p = cv2.imread(str(out / "patch" / "p_0003.png"), cv2.IMREAD_UNCHANGED)
    a3, m3 = p[..., 3].astype(np.float32) / 255, cu.removal_mask([obj_alpha(3)], 4.0, 3.0)
    far = ~(cv2.dilate((m3 > 0).astype(np.uint8), np.ones((2 * KEY_SOFT + 3,) * 2, np.uint8)) > 0)
    assert p.shape == (H, W, 4) and (a3 >= m3 - 0.01).all() and a3[far].max() == 0.0
    # sensitivity of the check: the ground card 0.25 too low moves the patch against the background
    bad = [Card("ground", "piasek", "plane", {"n": [0, -1, 0], "d": GROUND_Y + 0.25}, order=0, seam_px=4.0), make_cards()[1]]
    sc = RD.Scene(shot.work)
    one = {"ground": sc.keys_by_card["ground"][:1], "sky": sc.keys_by_card["sky"][:1]}      # key 1 alone
    low = (obj_alpha(3) > 0) & (np.mgrid[0:H, 0:W][0] > 120)
    good, _ = RD.project_frame(sc.geometry, sc.cards, one, 3, sc.cache, settings=SMALL)
    fill, _ = RD.project_frame(CardGeometry(shot.cam, bad), bad, one, 3, sc.cache, settings=SMALL)
    assert _shift(good, shot.bg[3], low) < 0.1 and _shift(fill, shot.bg[3], low) > 1.0


def test_fallback_where_no_key_reaches(tmp_path):
    shot = Shot(tmp_path, sky_keys=False)                 # the sky card has no keys
    rj, comp, src, out = shot.render(frames=[3, 4])
    assert rj["counts"]["klucz"] > 0 and rj["counts"]["OpenCV"] > 0
    geo = CardGeometry(shot.cam, shot.cards)
    for f in (3, 4):
        m = cu.removal_mask([obj_alpha(f)], 4.0, 3.0) > 0
        sky = geo.card_weight(shot.cards[1], f)
        assert (src[f][m & (sky > 0.999)] == cu.SRC_CV).all()
        assert (src[f][m & (sky < 0.001)] == cu.SRC_KEY).all()
        g = (obj_alpha(f) > 0) & (sky < 0.001)
        assert np.abs(comp[f] - shot.bg[f])[g].max() < 0.03
    # the fallback model is asked only for the pixels the keys left
    asked = []

    def lama(imgs, masks):
        asked.append(sum(int(m.sum()) for m in masks))
        return [np.full_like(im, 0.5) for im in imgs]
    st = {**SMALL, "grow_px": 4.0, "soft_px": 3.0, "motion_blur": "off"}
    rj2 = RD.render(shot.work, [3], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st, inpaint=lama)
    s2 = cv2.imread(str(Path(rj2["dir"]) / "source" / "s_0003.png"), 0)
    c2 = cv2.imread(str(Path(rj2["dir"]) / "comp" / "c_0003.png")).astype(np.float32) / 255
    assert rj2["version"] == "v002" and rj2["counts"]["LaMa"] > 0 and rj2["backend"] == "lama"
    inner = (s2 == cu.SRC_LAMA) & (obj_alpha(3) > 0) & (geo.card_weight(shot.cards[1], 3) > 0.999)
    assert inner.sum() > 100 and np.abs(c2[inner] - 0.5).max() < 2 / 255
    assert 0 < asked[0] < 1.3 * (cu.removal_mask([obj_alpha(3)], 4.0, 3.0) > 0).sum()
    # no keys at all / no camera: a clear error, nothing written
    with pytest.raises(ValueError, match="Brak kluczy"):
        empty = Path(tmp_path) / "e" / "matte_desk"
        store.save_camera(empty, shot.cam)
        store.save_cards(empty, shot.cards)
        RD.render(empty, [3], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)])
    with pytest.raises(ValueError, match="Brak kamery 3D"):
        RD.render(Path(tmp_path) / "none" / "matte_desk", [3], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)])
    assert not cu.cleanup_dir(Path(tmp_path) / "none" / "matte_desk").exists()


def test_kept_object_is_not_smeared(tmp_path):
    shot = Shot(tmp_path, keep=True)
    sc = RD.Scene(shot.work)
    f = 4                                                  # between the keys 1 and 5: the post of both keys lands elsewhere
    keep_fn = lambda kf: keep_alpha(kf)
    with_cut, w1 = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, f, sc.cache, keep_fn, settings=SMALL, cache=sc.cache)
    smeared, w0 = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, f, sc.cache, None, settings=SMALL, cache=sc.cache)
    green = lambda img: (np.abs(img - GREEN).max(-1) < 0.25)
    assert green(smeared).sum() > 300                       # without the cut the keys' post is projected onto the background
    assert not (green(with_cut) & (w1 > 0.5)).any()
    ok = w1 > 0.999
    assert ok.mean() > 0.7 and np.abs(with_cut - shot.bg[f])[ok].max() < 0.03
    assert (w1 < 0.5).sum() < (w0 < 0.5).sum() + 2500       # the cut is local (one key's post is covered by the other key)
    rj, comp, src, _ = shot.render()
    for f in FRAMES:
        kp = keep_alpha(f) > 0
        m = (obj_alpha(f) > 0) & ~kp
        assert np.abs(comp[f][kp] - GREEN).max() < 1 / 255              # the kept object itself: the plate
        assert (src[f][kp] == cu.SRC_PLATE).all()
        wk = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, f, sc.cache, keep_fn, settings=SMALL, cache=sc.cache)[1]
        e = np.abs(comp[f] - shot.bg[f]).max(-1)
        full = m & (wk >= cu.KEY_FULL)
        assert e[full].max() < 0.03 and (src[f][full] == cu.SRC_KEY).all(), f      # the keys alone: the true background
        # what no key shows (hidden behind the post in both keys, a few px beside it): the Cleaner's fallback
        rest = m & (wk < 0.5)
        far = m & ~(cv2.dilate(kp.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
        assert (src[f][rest] == cu.SRC_CV).all() and not (far & ~full).any(), (f, (far & ~full).sum())
        assert not green(comp[f])[far].any(), f                                    # no post anywhere else in the patch


def test_tone_correction_from_the_ring(tmp_path):
    shot = Shot(tmp_path, key_gain=-0.05)                  # the keys are darker than the plate
    _, off, _, _ = shot.render(frames=[3, 6], tone=False)
    _, on, _, _ = shot.render(frames=[3, 6], tone=True)
    for f in (3, 6):
        m = obj_alpha(f) > 0
        assert abs(np.abs(off[f] - shot.bg[f])[m].mean() - 0.05) < 0.005
        assert np.abs(on[f] - shot.bg[f])[m].mean() < 0.006 and np.abs(on[f] - shot.bg[f])[m].max() < 0.035
    # the correction reads only the ring: a wrong fill value outside it changes nothing
    rng = np.random.default_rng(1)
    plate = rng.random((120, 160, 3)).astype(np.float32)
    obj = np.zeros((120, 160), bool)
    obj[40:80, 60:100] = True
    fill = plate - 0.1
    far = fill.copy()
    far[:10] = 5.0
    us = np.ones_like(obj)
    a, b = RD.tone_match(fill, plate, obj, us, 12.0), RD.tone_match(far, plate, obj, us, 12.0)
    assert np.abs(a[obj] - b[obj]).max() < 1e-5 and abs(float((a - fill)[obj].mean()) - 0.1) < 0.01


# ------------------------------------------------------------------------------------------------ motion blur
def test_motion_field_is_the_camera_motion_through_the_cards():
    cam, cards = make_camera(), make_cards()
    geo = CardGeometry(cam, cards)
    dx, dy, sky = RD.motion_field(geo, 4, step=1)
    px = np.array([[100.0, 160.0], [220.0, 150.0], [60.0, 30.0]])                 # two on the ground, one in the sky
    for (x, y), card in zip(px, (cards[0], cards[0], cards[1])):
        k = lambda f: Key(card.id, f)
        a = geo.points_frame_to_key(np.array([[x, y]]), 4, k(3), card)[0]
        b = geo.points_frame_to_key(np.array([[x, y]]), 4, k(5), card)[0]
        want = (b - a) / 2.0
        assert abs(dx[int(y), int(x)] - want[0]) < 1e-3 and abs(dy[int(y), int(x)] - want[1]) < 1e-3
        assert sky[int(y), int(x)] == (1.0 if card.kind == "sky" else 0.0)
    assert np.hypot(dx, dy)[150:].mean() > 3.0                                     # the scene really moves
    coarse = RD.motion_field(geo, 4, roi=(40, 100, 200, 190))                      # every 4th pixel, scaled up
    err = np.abs(coarse[0] - dx[100:190, 40:200])
    assert coarse[0].shape == (90, 160) and err[4:-4, 4:-4].max() < 0.1 and err.max() < 0.6
    first = RD.motion_field(geo, 1, step=1)                                        # the shot's end: one-sided
    a = geo.points_frame_to_key(np.array([[100.0, 160.0]]), 1, Key("ground", 2), cards[0])[0]
    assert abs(first[0][160, 100] - (a[0] - 100.0)) < 1e-3


def test_blur_length_matched_to_the_plate():
    """A plate blurred along the camera motion by a known length: the estimate finds it (CRIT=hf of mb_match.py),
    a sharp plate gives 0; the sky and the ground get their own lengths."""
    cam, cards = make_camera(), make_cards()
    geo = CardGeometry(cam, cards)
    rng = np.random.default_rng(3)
    mb, sharp = RD.MotionBlur((W, H)), RD.MotionBlur((W, H))
    assert mb.q == 1.0 and abs(mb.k - W / 1920.0) < 1e-9 and mb.ring_px == 17
    for f in FRAMES[1:-1]:
        tex = cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32), (0, 0), 0.8)
        tex = (tex - tex.mean()) * 1.5 + 0.5                              # stays inside 0..1 (the measure clips)
        dx, dy, sky = RD.motion_field(geo, f)
        plate = np.where(sky[..., None] > 0.5, RD.blur_fill(tex, dx, dy, 0.375), RD.blur_fill(tex, dx, dy, 0.125))
        m = obj_alpha(f) > 0
        n = mb.measure(f, tex, plate, m, ~m, dx, dy, sky)
        assert n.get("ground", 0) > 500 and n.get("sky", 0) > 100
        sharp.measure(f, tex, tex, m, ~m, dx, dy, sky)
    got = mb.solve(FRAMES)
    assert mb.shot == {"ground": 0.125, "sky": 0.375}
    assert all(abs(g - 0.125) <= 0.0625 and abs(s - 0.375) <= 0.0625 for g, s in got.values())
    assert got[1] == (0.125, 0.375)                                                # a frame never measured: the shot's value
    assert set(sharp.solve(FRAMES).values()) == {(0.0, 0.0)}
    one = RD.MotionBlur((W, H), per_frame=False)
    one.rows = mb.rows
    assert set(one.solve(FRAMES).values()) == {(0.125, 0.375)}
    assert RD.MotionBlur((3840, 2160)).q == 0.5 and RD.MotionBlur((3840, 2160)).ring_px == 200


def test_render_with_motion_blur(tmp_path):
    shot = Shot(tmp_path)
    rj, comp, _, out = shot.render(motion_blur="auto")                 # sharp plates: the match says "no blur"
    assert rj["c3d"]["motion_blur"]["shot"]["ground"] == 0.0 and not (out / "_fill").exists()
    assert all(v["mb_length"] == [0.0, 0.0] for v in rj["c3d"]["frames"].values())
    for f in FRAMES:
        assert np.abs(comp[f] - shot.bg[f])[obj_alpha(f) > 0].max() < 0.03
    rj2, soft, _, _ = shot.render(frames=[4], motion_blur="manual", mb_length=0.5, tone=False)
    assert rj2["c3d"]["frames"][4]["mb_length"] == [0.5, 0.5] and rj2["c3d"]["frames"][4]["motion_px"] > 3.0
    m = obj_alpha(4) > 0
    hf = lambda img: float(RD.hf_energy_map(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255.0)[m & (np.mgrid[0:H, 0:W][0] > 120)].mean())
    assert hf(soft[4]) < 0.9 * hf(comp[4])                             # blurred along the motion
    geo = CardGeometry(shot.cam, shot.cards)
    dx, dy, sky = RD.motion_field(geo, 4)
    want = RD.blur_fill(shot.bg[4], dx, dy, 0.5)
    inner = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(soft[4] - want)[inner].max() < 0.03


# ------------------------------------------------------------------------------------------------ warps, sizes
def test_key_edge_warp_is_applied_before_projection(tmp_path):
    from engine import curves
    shot = Shot(tmp_path)
    warp = [{"src": [[20.0, 150.0], [300.0, 150.0]], "dst": [[20.0, 158.0], [300.0, 158.0]], "reach": 30}]
    shot.cards[0].warps = {"5": warp}
    store.save_cards(shot.work, shot.cards)
    sc = RD.Scene(shot.work)
    fill, _ = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, 5, sc.cache, settings=SMALL)
    want = curves.apply_warp(shot.bg[5], warp)
    low = np.zeros((H, W), bool)
    low[130:180, 40:280] = True
    assert np.abs(fill - want)[low].max() < 1e-3 and np.abs(fill - shot.bg[5])[low].max() > 0.05
    assert np.abs(fill - shot.bg[5])[5:50, 5:-5].max() < 1e-3          # the sky card's key 5 has no warp
    k8 = RD.view_key(sc, "ground", 5)
    assert k8.dtype == np.uint8 and np.abs(k8 / 255.0 - np.clip(want, 0, 1)).max() < 1 / 255
    assert np.abs(RD.view_key(sc, "ground", 5, warped=False) / 255.0 - shot.bg[5]).max() < 1 / 255


def test_smaller_key_file_is_scaled_to_the_key_view():
    cam, cards = make_camera(), make_cards()
    geo = CardGeometry(cam, cards)
    bg = true_bg(cam, 5)
    keys = {c.id: [Key(c.id, 5, [{"v": "v001"}], "v001")] for c in cards}
    half = cv2.resize(bg, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    fill, wgt = RD.project_frame(geo, cards, keys, 5, lambda k: half, settings=SMALL)
    assert fill.shape == (H, W, 3) and np.abs(fill - bg)[wgt > 0.999].mean() < 0.01


# ------------------------------------------------------------------------------------------------ Cleaner source
def test_cleaner_key_source_blends_partial_weights(tmp_path):
    h, w = 60, 80
    plate = np.full((h, w, 3), 0.2, np.float32)
    obj = np.zeros((h, w), np.float32)
    obj[20:40, 10:70] = 1
    wgt = np.zeros((h, w), np.float32)
    wgt[:, :30], wgt[:, 30:50] = 1.0, 0.25
    key = np.full((h, w, 3), 0.8, np.float32)
    s = {"grow_px": 0, "soft_px": 0, "fill": "lama"}
    c = cu.Cleaner(s, w, h, False, plate=lambda f: plate, remove_alphas=lambda f: [obj], keep_alphas=lambda f: [],
                   inpaint=lambda imgs, masks: [np.full_like(i, 0.4) for i in imgs], key_fill=lambda f: (key, wgt))
    c.report = {"c3d": {"x": 1}}
    rj = c.run([1], tmp_path / "a", "png")
    comp = cv2.imread(str(tmp_path / "a" / "comp" / "c_0001.png")).astype(np.float32) / 255
    src = cv2.imread(str(tmp_path / "a" / "source" / "s_0001.png"), 0)
    assert abs(comp[30, 20] - 0.8).max() < 1 / 255 and src[30, 20] == cu.SRC_KEY
    assert abs(comp[30, 40] - (0.25 * 0.8 + 0.75 * 0.4)).max() < 2 / 255 and src[30, 40] == cu.SRC_LAMA
    assert abs(comp[30, 60] - 0.4).max() < 1 / 255 and src[30, 60] == cu.SRC_LAMA
    assert abs(comp[5, 5] - 0.2).max() < 1 / 255 and src[5, 5] == cu.SRC_PLATE
    assert rj["counts"]["klucz"] == 20 * 20 and rj["c3d"] == {"x": 1} and cu.SRC_KEY == 6 and cu.KEY_NAME == "klucz"
    # "clean plate only" and no clean plate: a partly covered pixel takes the key alone (never mixed with the object)
    c2 = cu.Cleaner({**s, "fill": "cleanplate"}, w, h, False, plate=lambda f: plate, remove_alphas=lambda f: [obj],
                    keep_alphas=lambda f: [], key_fill=lambda f: (key, wgt))
    c2.run([1], tmp_path / "b", "png")
    comp2 = cv2.imread(str(tmp_path / "b" / "comp" / "c_0001.png")).astype(np.float32) / 255
    assert abs(comp2[30, 40] - 0.8).max() < 1 / 255 and abs(comp2[30, 60] - 0.2).max() < 1 / 255
    # a run without key_fill reports what it always did
    c3 = cu.Cleaner(s, w, h, False, plate=lambda f: plate, remove_alphas=lambda f: [obj], keep_alphas=lambda f: [])
    rj3 = c3.run([1], tmp_path / "c", "png")
    assert set(rj3["counts"]) == {"clear plate", "cleanplate_auto", "LaMa", "OpenCV"} and "c3d" not in rj3
    assert cu.SRC_KEY not in cu.SOURCE_NAMES


# ------------------------------------------------------------------------------------------------ previews
def test_previews(tmp_path):
    shot = Shot(tmp_path)
    sc = RD.Scene(shot.work)
    assert sc.size == (W, H) and sc.has_keys() and sc.card("sky").kind == "sky"
    with pytest.raises(ValueError, match="Brak klucza"):
        RD.view_key(sc, "ground", 2)
    with pytest.raises(ValueError, match="Nieznana karta"):
        RD.view_key(sc, "wall", 1)
    assert RD.view_key(sc, "sky", 1, max_side=160).shape == (100, 160, 3)
    pr = RD.view_projection(sc, 3)
    assert pr.dtype == np.uint8 and pr.shape == (H, W, 3)
    full = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, 3, sc.cache, settings=None)[1] > 0.999
    assert np.abs(pr / 255.0 - shot.bg[3])[full].max() < 0.03
    one = RD.Scene(shot.work)
    one.keys_by_card = {"ground": one.keys_by_card["ground"][:1], "sky": []}
    dark = RD.view_projection(one, 9, plate=shot.plate(9))
    sky = one.geometry.card_weight(one.cards[1], 9) > 0.999
    assert np.abs(dark / 255.0 - shot.plate(9) * 0.25)[sky].max() < 1 / 255       # no key there: the plate, darkened
    cards = RD.view_cards(sc, 3, max_side=None)
    plain = RD.view_cards(sc, 3, max_side=None, tint=0.0)
    assert cards.shape == (H, W, 3) and cards.dtype == np.uint8
    gw = sc.geometry.card_weight(sc.cards[0], 3) > 0.999
    lines = np.abs(plain.astype(int) - 31).max(-1) > 40                            # grid lines over the dark base
    assert 0.02 < lines[gw].mean() < 0.6 and 0.005 < lines[~gw].mean() < 0.6
    sc.cards[1].color = "#ff0000"
    red = RD.view_cards(sc, 3, max_side=None)
    assert red[~gw][..., 2].mean() > cards[~gw][..., 2].mean() + 20                # a card's own colour shows
    assert np.array_equal(red[gw], cards[gw])
    small = RD.view_cards(sc, 3, plate=shot.plate(3), max_side=160)
    assert small.shape == (100, 160, 3)


# ------------------------------------------------------------------------------------------------ GPU path
FAKE_ENV = {**os.environ, "MD_GEOM_GPU_FAKE": "1"}


def test_extent_layer_of_the_card_wins_over_a_key_area_of_the_same_name(tmp_path):
    """The API writes an extent's brush layers into inpaint/cards/<card>/ (store.extent_dir); a regeneration area of
    the key at extent_frame may have the same id (the editor numbers both a1, a2, ...) and is NOT the extent."""
    from engine.cleanup3d.types import Area
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    card = Card("ground", "piasek", "plane", {"n": [0, -1, 0], "d": GROUND_Y},
                extent=[Area("a1", "brush", png="a_a1.png")], extent_frame=5)
    kd = cu.cleanup_dir(work) / "keys" / "ground" / "k_0005"
    kd.mkdir(parents=True)
    cv2.imwrite(str(kd / "a_a1.png"), np.full((H, W), 255, np.uint8))        # the key's own area: everything
    assert RD.extent_layers(work, [card])["ground"]["a1"].min() == 255       # old layout: next to the key still read
    ed = store.extent_dir(work, "ground")
    ed.mkdir(parents=True)
    mine = np.zeros((H, W), np.uint8)
    mine[100:, :50] = 255
    cv2.imwrite(str(ed / "a_a1.png"), mine)
    assert np.array_equal(RD.extent_layers(work, [card])["ground"]["a1"], mine)


def _devices_agree(worker):
    cam, cards = make_camera(), make_cards()
    bg = {f: true_bg(cam, f) for f in KEY_FRAMES}
    keys = {c.id: [Key(c.id, f, [{"v": "v001"}], "v001") for f in KEY_FRAMES] for c in cards}
    cpu = CardGeometry(cam, cards)
    gpu = CardGeometry(cam, cards, device="gpu", gpu=worker)
    for f, roi in ((3, None), (7, (30, 20, 260, 190))):
        a = RD.project_frame(cpu, cards, keys, f, lambda k: bg[k.frame], lambda kf: keep_alpha(kf), settings=SMALL, roi=roi)
        b = RD.project_frame(gpu, cards, keys, f, lambda k: bg[k.frame], lambda kf: keep_alpha(kf), settings=SMALL, roi=roi)
        # premultiplied (fill x weight): where the weight is ~0 a fill value is meaningless (cv2.remap's 1/32 px fixed
        # point vs the GPU's exact sampling decide "nothing" vs "almost nothing" there)
        pa, pb = a[0] * a[1][..., None], b[0] * b[1][..., None]
        assert np.abs(pa - pb).max() < 1 / 255 and np.abs(a[1] - b[1]).max() < 2 / 255, f


def test_gpu_path_equals_cpu_fake_worker():
    """The GPU plumbing (maps and card weights from the geometry worker, under GPU.use) with the worker's numpy
    backend in this .venv: the projected fill equals the CPU path within 1 / 255."""
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        _devices_agree(w)
        assert w.device == "fake"
    finally:
        w.close()


def test_gpu_render_equals_cpu_render_fake_worker(tmp_path):
    """The whole render (projection with keys resident in the worker, motion blur, patch) on device "gpu" (the
    worker's numpy backend) equals the CPU render; the blur op equals engine.shutter.path_blur away from the border."""
    from engine.shutter import path_blur
    shot = Shot(tmp_path)
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        st = {**SMALL, "remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0, "motion_blur": "manual", "mb_length": 0.5}
        cpu = RD.render(shot.work, [3, 6], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st)
        sc = RD.Scene(shot.work, device="gpu", gpu=w)
        gpu = RD.render(shot.work, [3, 6], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st,
                        scene=sc, out_dir=shot.work.parent / "gpu_out")
        for f in (3, 6):
            a = cv2.imread(str(Path(cpu["dir"]) / "comp" / f"c_{f:04d}.png")).astype(np.float32)
            b = cv2.imread(str(Path(gpu["dir"]) / "comp" / f"c_{f:04d}.png")).astype(np.float32)
            assert np.abs(a - b).mean() < 0.2 and np.percentile(np.abs(a - b), 99.9) <= 3, f
        assert w.device == "fake"
        rng = np.random.default_rng(5)
        img = cv2.GaussianBlur(rng.random((60, 80, 3)).astype(np.float32), (0, 0), 1.0)
        lx = np.full((60, 80), 6.0, np.float32)
        ly = np.full((60, 80), 2.0, np.float32)
        ref = np.dstack([path_blur(np.ascontiguousarray(img[..., c]), lx, ly) for c in range(3)])
        geo = sc.geometry
        got = RD.blur_fill(img, lx / 2, ly / 2, 2.0, gpu=geo)
        assert np.abs(got - ref)[8:-8, 8:-8].max() < 2e-3
        sc.close()
    finally:
        w.close()


@pytest.mark.real
def test_gpu_path_equals_cpu_cuda():
    ok, info = check_worker()
    if not (ok and info.startswith("cuda")):
        pytest.skip(f"ComfyUI's Python with CUDA not available ({info})")
    w = GeomGpuWorker()
    try:
        _devices_agree(w)
        assert w.device.startswith("cuda")
    finally:
        w.close()


def _keyfill_ops_agree(worker, tmp_path, tol):
    """The key fill's per-frame ops in the geometry worker (resident card weights + projection, motion, mb_measure,
    keyfill) against the CPU path on the same inputs: motion field, blur measure errors, blurred + tone-matched fill.
    `tol` = (motion px, measure error, fill, grain level relative)."""
    from engine.cleanup3d.geom import _jsonable_view
    cam, cards = make_camera(), make_cards()
    bg = {f: true_bg(cam, f) for f in KEY_FRAMES}
    keys = {c.id: [Key(c.id, f, [{"v": "v001"}], "v001") for f in KEY_FRAMES] for c in cards}
    cpu = CardGeometry(cam, cards)
    gpu = CardGeometry(cam, cards, device="gpu", gpu=worker)
    cache = RD.KeyCache(lambda k: bg[k.frame])
    f, roi = 4, (20, 10, 300, 190)
    x0, y0, x1, y1 = roi
    s = RD.render_settings(SMALL)
    fill, wgt = RD.project_frame(gpu, cards, keys, f, cache, settings=SMALL, roi=roi, cache=cache)
    res = {"save": {"fill": str(tmp_path / "f.npy"), "wgt": str(tmp_path / "w.npy")}}
    RD._project_gpu(gpu, cards, keys, f, cache, None, s, roi, {}, resident=res)
    assert res["any"]
    assert np.array_equal(np.load(tmp_path / "f.npy"), fill.astype(np.float16))          # pass 1's files, as the CPU
    assert np.abs(np.load(tmp_path / "w.npy").astype(np.float32) - np.round(wgt * 255)).max() <= 1
    # motion field: the worker's (resident weights) = the CPU's
    mo = str(tmp_path / "m.npy")
    RD._motion_gpu(gpu, f, roi, out=mo)
    dx, dy, sky = RD.motion_field(cpu, f, roi)
    m3 = np.load(mo)
    assert np.abs(m3[0] - dx).max() < tol[0] and np.abs(m3[1] - dy).max() < tol[0] and np.abs(m3[2] - sky).max() < 1e-3
    # blur measure (one request, every candidate) on the resident projection
    rng = np.random.default_rng(11)
    tex = cv2.GaussianBlur(rng.random((y1 - y0, x1 - x0, 3)).astype(np.float32), (0, 0), 0.8)
    tex = (tex - tex.mean()) * 1.5 + 0.5
    plate = RD.blur_fill(tex, dx, dy, 0.25)
    mask = obj_alpha(f)[y0:y1, x0:x1] > 0
    a, b = RD.MotionBlur((W, H)), RD.MotionBlur((W, H))
    na = a.measure(f, fill, plate, mask, (wgt >= cu.KEY_FULL) & ~mask, dx, dy, sky)
    nb = b.measure_gpu(gpu, f, roi, plate, mask, None)
    assert set(na) == set(nb) and na and all(abs(na[r] - nb[r]) <= 0.01 * na[r] for r in na)
    for r in na:
        ea, eb = a.rows[r][f], b.rows[r][f]
        assert set(ea) == set(eb) and max(abs(ea[L] - eb[L]) for L in ea) < tol[1], (r, ea, eb)
    # pass 2: blur along the motion field + tone match from the ring (op keyfill)
    f16 = fill.astype(np.float16)
    hole = RD.cu.removal_mask([obj_alpha(f)], 4.0, 0.0)[y0:y1, x0:x1] > 0
    usable = (wgt >= cu.KEY_FULL)
    sel = np.ones(hole.shape, bool)
    paths = {k: str(tmp_path / f"{k}.npy") for k in ("kf", "pl", "ho", "us", "se", "out")}
    np.save(paths["kf"], f16)
    np.save(paths["pl"], np.clip(plate + 0.04, 0, 1).astype(np.float32))
    np.save(paths["ho"], hole.astype(np.uint8))
    np.save(paths["us"], usable.astype(np.uint8))
    np.save(paths["se"], sel.astype(np.uint8))
    resp = worker.call({"op": "keyfill", "frame": _jsonable_view(gpu.frame_view(f)), "roi": list(roi), "fill": paths["kf"],
                        "lengths": [0.25, 0.5], "sel": paths["se"], "out": paths["out"],
                        "tone": {"plate": paths["pl"], "hole": paths["ho"], "usable": paths["us"], "ring_px": 12.0}})
    want = RD.blur_fill(f16.astype(np.float32), dx, dy, 0.5 * sky + 0.25 * (1.0 - sky))
    want = RD.tone_match(want, np.clip(plate + 0.04, 0, 1).astype(np.float32), hole, usable, 12.0)
    got = np.load(paths["out"])
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.abs(got - want)[6:-6, 6:-6].max() < tol[2]
    assert abs(resp["motion_px"] - round(float(np.hypot(dx, dy).mean()), 2)) < 0.05
    # the grain measurement's levels on the fill / plate keyfill left in the worker (op grain_levels)
    from engine.cleanup3d import grain as GR
    noisy = np.clip(np.load(paths["pl"]) + rng.normal(0, 0.01, plate.shape).astype(np.float32), 0, 1)
    reg = ~hole & usable
    sel = GR.flat_pixels(got, reg) | (rng.random(reg.shape) < 0.05) & reg
    idx = rng.integers(0, GR.LUM_BINS, int(sel.sum())).astype(np.uint8)
    np.save(paths["pl"], noisy)
    want = GR.levels_cpu(noisy, np.clip(got, 0, 1), reg, sel, idx)
    np.save(paths["kf"], got)
    gp = {k: str(tmp_path / f"g{k}.npy") for k in ("r", "s", "i")}
    np.save(gp["r"], reg.astype(np.uint8))
    np.save(gp["s"], sel.astype(np.uint8))
    np.save(gp["i"], idx)
    lv = worker.call({"op": "grain_levels", "frame": _jsonable_view(gpu.frame_view(f)), "roi": list(roi),
                      "fill": paths["kf"], "plate": paths["pl"], "reg": gp["r"], "sel": gp["s"], "idx": gp["i"],
                      "sigmas": list(GR.SIGMAS), "cblock": GR.CBLOCK, "bins": GR.LUM_BINS})
    rel = lambda x, y: float(np.max(np.abs(np.asarray(x, np.float64) - y) / np.maximum(np.abs(y), 1e-6)))
    assert rel(lv["bp"], want[0]) < tol[3] and rel(lv["bf"], want[1]) < tol[3]
    assert all(rel(x, y) < tol[3] for x, y in zip(lv["binp"] + lv["binf"], want[2] + want[3]))


def test_keyfill_ops_equal_cpu_fake_worker(tmp_path):
    """The key fill ops of the geometry worker on its numpy backend (OpenCV primitives, the same rules; grain levels =
    grain.levels_cpu itself) = the CPU path."""
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        _keyfill_ops_agree(w, tmp_path, (1e-3, 1e-6, 2e-3, 1e-6))
    finally:
        w.close()


@pytest.mark.real
def test_keyfill_ops_equal_cpu_cuda(tmp_path):
    ok, info = check_worker()
    if not (ok and info.startswith("cuda")):
        pytest.skip(f"ComfyUI's Python with CUDA not available ({info})")
    w = GeomGpuWorker()
    try:
        _keyfill_ops_agree(w, tmp_path, (2e-3, 5e-3, 3e-3, 2e-3))
        assert w.device.startswith("cuda")
    finally:
        w.close()


def test_gpu_render_auto_blur_equals_cpu_fake_worker(tmp_path):
    """The render with motion blur "auto" on device "gpu" (pass 1 in the worker: project + motion + mb_measure, pass 2:
    keyfill) = the CPU render: the same lengths, the same composite."""
    shot = Shot(tmp_path)
    w = GeomGpuWorker(sys.executable, FAKE_ENV, timeout_s=120)
    try:
        st = {**SMALL, "remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0, "motion_blur": "auto", "mb_scale": 2.0}
        cpu = RD.render(shot.work, [3, 4, 5], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st)
        sc = RD.Scene(shot.work, device="gpu", gpu=w)
        gpu = RD.render(shot.work, [3, 4, 5], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st,
                        scene=sc, out_dir=shot.work.parent / "gpu_auto")
        for f in (3, 4, 5):
            fa, fb = cpu["c3d"]["frames"][f], gpu["c3d"]["frames"][f]
            assert fa["mb_length"] == fb["mb_length"] and fa["mb_ring_px"] == fb["mb_ring_px"], f
            a = cv2.imread(str(Path(cpu["dir"]) / "comp" / f"c_{f:04d}.png")).astype(np.float32)
            b = cv2.imread(str(Path(gpu["dir"]) / "comp" / f"c_{f:04d}.png")).astype(np.float32)
            assert np.abs(a - b).mean() < 0.2 and np.percentile(np.abs(a - b), 99.9) <= 3, f
        assert "blur_tone_gpu" in gpu["c3d"]["seconds"]
        sc.close()
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ the converter
def test_c3d_from_experiment_writes_the_contract(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import c3d_from_experiment as CV
    exp = tmp_path / "exp"
    S = exp / "outputs" / "sfmX"
    S.mkdir(parents=True)
    (exp / "outputs" / "keysX").mkdir()
    cam = make_camera()
    np.savez(S / "cams.npz", K=cam.K, R=np.stack([cam.R[f] for f in FRAMES]), t=np.stack([cam.t[f] for f in FRAMES]),
             dist=np.array([-0.01, 0.002, 0.0, 0.0]), W=W, H=H)
    (S / "plane.json").write_text(json.dumps({"n": [0, 1, 0], "d": -GROUND_Y}))          # normal away from the cameras
    (S / "sea_plane.json").write_text(json.dumps({"up": [0, -1, 0], "h_sea": -3.0}))
    warp = [{"src": [[10, 50], [300, 50]], "dst": [[10, 52], [300, 52]], "reach": 40}]
    (S / "cards.json").write_text(json.dumps({"global": {"sand": {"dh": 0.1, "pitch": 0.0, "roll": 0.0}, "sea": {"dh": 0.0}},
                                              "keys": {"5": {"sand": {"dh": 0.3}}}, "key_frames": [1, 5, 9], "warps": {"5": warp}}))
    for f in (1, 5, 9):
        cv2.imwrite(str(exp / "outputs" / "keysX" / f"g_{f:04d}.png"), np.full((H // 2, W // 2, 3), 20 * f, np.uint8))
    work = tmp_path / "copy" / "matte_desk"
    log = []
    out = CV.convert(work, exp, "sfmX", "outputs/keysX", log=log.append)
    assert out["cards"] == ["sand", "sea", "sky"] and out["key_frames"] == [1, 5, 9] and out["keys_written"] == 9
    assert out["skipped_key_exceptions"] == ["5"] and any("pominięte" in l for l in log)
    sc = RD.Scene(work)
    assert sc.camera.frames == FRAMES and sc.size == (W, H) and sc.camera.dist[0] == -0.01
    assert [(c.id, c.kind, c.order, c.seam_px) for c in sc.cards] == [("sand", "plane", 0, 0.0), ("sea", "plane", 1, 0.0), ("sky", "sky", 2, 0.0)]
    assert sc.cards[0].adjust["dh"] == 0.1 and sc.cards[1].plane == {"n": [0.0, -1.0, 0.0], "d": 3.0}
    assert all(c.warps == {"5": warp} for c in sc.cards)
    assert {c: [k.frame for k in ks] for c, ks in sc.keys_by_card.items()} == {c: [1, 5, 9] for c in ("sand", "sea", "sky")}
    img = sc.cache.image(sc.geometry, sc.cards[0], sc.key("sand", 1))
    assert img.shape == (H, W, 3) and abs(float(img.mean()) - 20 / 255) < 1e-3          # scaled to the camera's size
    fill, wgt = RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, 3, sc.cache, settings=SMALL)
    assert wgt.max() > 0.999 and np.isfinite(fill).all()
    assert CV.convert(work, exp, "sfmX", "outputs/keysX", log=log.append)["keys_written"] == 0     # a re-run adds nothing
    assert CV.main(["--work", str(tmp_path / "shot_manifest_x.json"), "--exp", str(exp), "--sfm", "sfmX", "--keys", "outputs/keysX"]) == 1


# ------------------------------------------------------------------------------------------------ real: sh060, chain v3c
EXP = Path(os.environ.get("MD_C3D_EXPERIMENT", r"D:/MD_REAL_PROJECT/Documents\removers_test"))
BEATCH = Path(os.environ.get("MD_C3D_PLATES", r"D:/MD_REAL_PROJECT/Documents\shots\shots\matte_desk\source\sh060"))
REF = {"sky": 1.17, "sea": 2.08, "sand": 2.03}             # flicker3d.py of the experiment's v3c (FUTURE_MODULES §4)


def flicker3d(geo, comp, mask, frames, grow=80, q=4):
    """removers_test/flicker3d.py on a CardGeometry: comp t-1 projected onto t through the cards, mean |diff| of the
    lightly blurred gray (8-bit) inside the patch (mask grown by `grow`), split by the card a pixel shows (the cards'
    order: the first plane hit; "sky" = none). comp(f) -> BGR uint8, mask(f) -> uint8. Returns {card id: mean}."""
    w, h = int(geo.camera.width), int(geo.camera.height)
    sw, sh = w // q, h // q
    PX, PY = np.meshgrid((np.arange(sw, dtype=np.float64) + 0.5) * q - 0.5, (np.arange(sh, dtype=np.float64) + 0.5) * q - 0.5)
    planes = [c for c in geo.cards if c.kind != "sky"]
    names = [c.id for c in planes] + ["sky"]
    res = {n: [] for n in names}
    load = lambda f: cv2.GaussianBlur(cv2.cvtColor(cv2.resize(comp(f), (sw, sh), interpolation=cv2.INTER_AREA),
                                                   cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    prev = load(frames[0])
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    for t0, t in zip(frames[:-1], frames[1:]):
        cur = load(t)
        m = cv2.resize(cv2.dilate(mask(t), ker), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
        view, src = geo.frame_view(t), geo.frame_view(t0)
        x, y = KER.view_rays(KER.NP, view, PX, PY)
        lab = np.full((sh, sw), len(planes), np.int32)
        mx, my, _ = KER.project(KER.NP, x, y, view, src, {"kind": "sky"})
        for i in reversed(range(len(planes))):                # the first card in order wins
            spec = geo.card_spec(planes[i])
            hit = KER.plane_hit(KER.NP, x, y, view, spec)
            cx, cy, _ = KER.project(KER.NP, x, y, view, src, spec)
            mx, my, lab = np.where(hit, cx, mx), np.where(hit, cy, my), np.where(hit, i, lab)
        inb = (mx > 2) & (mx < w - 3) & (my > 2) & (my < h - 3)
        pw = cv2.remap(prev, ((mx + 0.5) / q - 0.5).astype(np.float32), ((my + 0.5) / q - 0.5).astype(np.float32), cv2.INTER_LINEAR)
        d = np.abs(cur - pw)
        for i, n in enumerate(names):
            sel = m & inb & (lab == i)
            if sel.sum() > 200:
                res[n].append(float(d[sel].mean()))
        prev = cur
    return {n: (float(np.mean(v)) if v else None) for n, v in res.items()}


def _horizon_band(geo, f, px=60):
    """Pixels within `px` of the line where the plane cards end (the horizon) in frame f."""
    hit = np.zeros((int(geo.camera.height), int(geo.camera.width)), np.float32)
    for c in geo.cards:
        if c.kind != "sky":
            hit = np.maximum(hit, (geo.card_weight(c, f) > 0.5).astype(np.float32))
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return (cv2.dilate(hit, k) > 0) & (cv2.erode(hit, k) == 0)


@pytest.mark.real
def test_real_sh060_v3c(tmp_path):
    """The experiment's keys through this package against the experiment's own result. The experiment also ran a
    per-frame horizon snap (horizon_snap.py) between the projection and the blur, which is NOT part of this package:
    the difference is reported near the horizon and away from it."""
    if not ((EXP / "outputs" / "sfm8" / "cams.npz").is_file() and (BEATCH / "sh060_0001.png").is_file()):
        pytest.skip(f"the sh060 experiment is not on this machine ({EXP})")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import c3d_from_experiment as CV
    base = Path(os.environ.get("MD_C3D_REAL_OUT") or tmp_path)
    work = base / "sh060_c3d" / "matte_desk"
    if not (cu.cleanup_dir(work) / "cards.json").is_file():
        CV.convert(work, EXP, log=lambda *_: None)
    frames = list(range(1, 58))
    if os.environ.get("MD_C3D_REAL_FRAMES"):               # a quick look: "20-26"
        a, b = os.environ["MD_C3D_REAL_FRAMES"].split("-")
        frames = list(range(int(a), int(b) + 1))
    plate = lambda f: cv2.imread(str(BEATCH / f"sh060_{f:04d}.png")).astype(np.float32) / 255.0
    mask = lambda f: cv2.imread(str(EXP / "outputs" / "gpu_d" / "mask" / f"m_{f:04d}.png"), 0)
    ref = lambda f: cv2.imread(str(EXP / "outputs" / "v3dW3c_b" / "proj" / f"c_{f:04d}.png"))
    report = {"frames": [frames[0], frames[-1]]}

    # seconds per frame of the projection alone, CPU and GPU (the region around the mask, as the render does it)
    okg, info = check_worker()
    devices = ["cpu"] + (["gpu"] if okg and info.startswith("cuda") else [])
    report["gpu_worker"] = info
    for dev in devices:
        sc = RD.Scene(work, device=dev)
        try:
            ts = []
            for f in frames[1:4]:
                m = cu.removal_mask([mask(f).astype(np.float32) / 255.0], 40.0, 40.0)
                roi = cu.mask_bbox([m], 248, *sc.size)
                t0 = time.time()
                RD.project_frame(sc.geometry, sc.cards, sc.keys_by_card, f, sc.cache, settings=None, roi=roi, cache=sc.cache)
                ts.append(time.time() - t0)
            report[f"project_s_per_frame_{dev}"] = [round(t, 2) for t in ts]
        finally:
            sc.close()

    # the render: void_comp.py GROW=80 FEATHER=40 = the mask grown 40 px + 40 px softened outward
    dev = os.environ.get("MD_C3D_REAL_DEVICE", "cpu")
    st = {"grow_px": 40.0, "soft_px": 40.0, "holdout": False, "fill": "lama", "motion_blur": "auto", "mb_lam": 0.3,
          "device": dev if dev in devices else "cpu"}
    t0 = time.time()
    rj = RD.render(work, frames, plate=plate, remove_alphas=lambda f: [mask(f).astype(np.float32) / 255.0], settings=st,
                   out_dir=cu.cleanup_dir(work) / "v_real")
    report["render_s_per_frame"] = round((time.time() - t0) / len(frames), 2)
    report["render_device"] = st["device"]
    report["seconds"] = {**rj["c3d"]["seconds"], "prepare": rj["c3d"]["seconds_prepare"], "total": rj["seconds"]}
    report["counts"] = rj["counts"]
    report["mb_shot"] = rj["c3d"]["motion_blur"]["shot"]
    report["mb_per_frame"] = {f: v.get("mb_length") for f, v in rj["c3d"]["frames"].items()}
    out = Path(rj["dir"])
    mine = lambda f: cv2.imread(str(out / "comp" / f"c_{f:04d}.png"))

    sc = RD.Scene(work)
    geo = sc.geometry
    report["flicker3d_reference"] = flicker3d(geo, ref, mask, frames)
    report["flicker3d_render"] = flicker3d(geo, mine, mask, frames)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161))
    # "outside": 4 px beyond the patch's nominal 80 px (the Cleaner's soft edge is a 5-mask distance transform,
    # its last alpha steps reach 1-2 px further than the true 80)
    ker_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (169, 169))
    diff = {"patch": [], "horizon": [], "away": [], "outside": []}
    worst = (0.0, None)
    for f in frames:
        d = np.abs(mine(f).astype(np.float32) - ref(f).astype(np.float32)).mean(-1)
        patch = cv2.dilate(mask(f), ker) > 0
        band = _horizon_band(geo, f)
        diff["patch"].append(float(d[patch].mean()))
        outside = ~(cv2.dilate(mask(f), ker_out) > 0)
        diff["outside"].append(float(d[outside].max()) if outside.any() else 0.0)
        if (patch & band).sum() > 500:
            diff["horizon"].append(float(d[patch & band].mean()))
        diff["away"].append(float(d[patch & ~band].mean()))
        if diff["patch"][-1] > worst[0]:
            worst = (diff["patch"][-1], f)
    report["diff_8bit_mean"] = {k: round(float(np.mean(v)), 3) for k, v in diff.items() if k != "outside" and v}
    report["diff_8bit_outside_patch_max"] = round(float(np.max(diff["outside"])), 3)
    report["diff_worst_frame"] = [worst[1], round(worst[0], 3)]
    (base / "c3d_real_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nC3D REAL " + json.dumps(report))

    assert rj["counts"]["klucz"] > 0.97 * sum(rj["counts"].values())          # the keys fill (almost) all of the patch
    assert report["diff_8bit_outside_patch_max"] <= 1.0                       # outside the patch both are the plate
    if len(frames) == 57:
        fr, fm = report["flicker3d_reference"], report["flicker3d_render"]
        for n, name in (("sky", "sky"), ("sea", "sea"), ("sand", "sand")):
            assert abs(fr[n] - REF[name]) < 0.05, (n, fr[n])                  # the ported metric reads the experiment's numbers
            assert fm[n] <= fr[n] * 1.15 + 0.05, (n, fm[n], fr[n])            # not worse than the experiment (15 % + 0.05)


def test_match_patch_blur_finds_the_plate_blur_from_the_generator_output():
    """User 2026-11-10: a sharp generator patch (FLUX 3 Image Klein over VOID) in a motion-blurred plate. The length is
    measured on the generator's own output OUTSIDE the hole and applied to the whole crop."""
    from engine.cleanup3d import render as RD

    class Geo:
        pass
    h, w = 240, 320
    dx = np.full((h, w), 12.0, np.float32)
    dy = np.zeros((h, w), np.float32)
    rng = np.random.default_rng(3)
    sharp = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 0.8)
    plate = RD.blur_fill(sharp, dx, dy, 0.5)                       # the camera's blur at shutter 180
    hole = np.zeros((h, w), bool)
    hole[90:150, 120:200] = True
    orig = RD.motion_field
    RD.motion_field = lambda geo, f, roi=None, step=4: (dx[roi[1]:roi[3], roi[0]:roi[2]], dy[roi[1]:roi[3], roi[0]:roi[2]],
                                                        np.zeros((roi[3] - roi[1], roi[2] - roi[0]), np.float32))
    try:
        out, L = RD.match_patch_blur(Geo(), 5, sharp, plate, hole, margin=60)
        assert abs(L - 0.5) <= 0.0625 + 1e-6
        assert float(np.abs(out[hole] - plate[hole]).mean()) < float(np.abs(sharp[hole] - plate[hole]).mean()) * 0.5
        same, L0 = RD.match_patch_blur(Geo(), 5, plate, plate, hole, margin=60)     # already blurred: nothing to add
        assert L0 == 0.0 and same is plate
        none, Ln = RD.match_patch_blur(Geo(), 5, sharp, plate, np.zeros((h, w), bool))
        assert Ln == 0.0 and none is sharp
    finally:
        RD.motion_field = orig


def test_render_takes_the_objects_shadow_into_the_patch(tmp_path):
    """sh060 2026-11-10: the removed woman's shadow lay outside the removal mask and stayed. The projected keys are the
    clean background of the frame, so a dark area of the plate touching the object = its effect -> patched."""
    shot = Shot(tmp_path)
    plain = shot.plate

    def shadow(f):                                             # a shadow below the object, outside its mask
        yy, xx = np.mgrid[0:H, 0:W]
        cx = 70.0 + 22.0 * (f - 1)
        return (np.abs(xx - cx - 10) < 14) & (yy > 140) & (yy < 175)

    shot.plate = lambda f: np.where(shadow(f)[..., None], plain(f) * 0.45, plain(f)).astype(np.float32)
    rj, comp, src, _ = shot.render(frames=[3, 6])
    for f in (3, 6):
        sh = shadow(f) & (cu.removal_mask([obj_alpha(f)], 4.0, 3.0) == 0)
        assert sh.sum() > 300
        assert (src[f][sh] == cu.SRC_KEY).mean() > 0.95                   # the shadow is patched with the keys
        assert np.abs(comp[f] - shot.bg[f])[sh].mean() < 0.02              # ... and shows the background
    assert all(v.get("effects_px", 0) > 300 for v in rj["c3d"]["frames"].values())
    _, off, src_off, _ = shot.render(frames=[3], key_effects=False)
    sh = shadow(3) & (cu.removal_mask([obj_alpha(3)], 4.0, 3.0) == 0)
    assert np.abs(off[3] - shot.bg[3])[sh].mean() > 0.1                    # without effects the shadow stays


def test_blend_inputs_and_render_settings(tmp_path):
    """C5 live blend preview (2026-11-10): the preview inputs mark the object's shadow as a strong difference next to
    the removal mask; the saved settings are clamped and used by the render."""
    shot = Shot(tmp_path)
    sc = RD.Scene(shot.work)
    try:
        f = 3
        yy, xx = np.mgrid[0:H, 0:W]
        cx = 70.0 + 22.0 * (f - 1)
        sh = (np.abs(xx - cx - 10) < 14) & (yy > 140) & (yy < 175)
        plate = np.where(sh[..., None], shot.plate(f) * 0.45, shot.plate(f)).astype(np.float32)
        fill, maps, scale = RD.blend_inputs(sc, f, plate, obj_alpha(f), grow_px=4.0, settings=SMALL, max_side=W)
        assert fill.shape == (H, W, 3) and maps.shape == (H, W, 3) and abs(scale - 1.0) < 1e-6
        outside = sh & (cu.removal_mask([obj_alpha(f)], 4.0, 0.0) == 0)
        assert (maps[..., 1][obj_alpha(f) > 0] == 255).all()                       # G = the removal mask
        assert maps[..., 0][outside].mean() / 255 * RD.BLEND_DIFF_MAX > 0.15       # B: the shadow differs
        far = (yy < 40) & (xx > 250)
        assert maps[..., 0][far].mean() / 255 * RD.BLEND_DIFF_MAX < 0.03
        small = RD.blend_inputs(sc, f, plate, obj_alpha(f), grow_px=4.0, settings=SMALL, max_side=160)
        assert small[0].shape == (H // 2, W // 2, 3) and abs(small[2] - 0.5) < 1e-6
    finally:
        sc.close()
    assert store.load_render_settings(shot.work) == store.RENDER_SETTINGS
    got = store.save_render_settings(shot.work, {"key_soft_px": 999, "key_effects": 0, "key_effects_thr": 0.2, "x": 1})
    assert got == {"key_soft_px": 400.0, "key_grow_px": 16.0, "key_effects": False, "key_effects_thr": 0.2, "clean_first": False,
                   "horizon_match": False, "horizon_fade_px": 120.0, "horizon_search_px": 60.0,   # + P15 horizon defaults
                   "grain": "auto", "grain_strength": 1.0}
    got = store.save_render_settings(shot.work, {"grain": "nope", "grain_strength": 7})       # P17 limits
    assert got["grain"] == "auto" and got["grain_strength"] == 2.0
    got = store.save_render_settings(shot.work, {"grain": "off", "grain_strength": 0.5})
    assert got["grain"] == "off" and got["grain_strength"] == 0.5
    assert store.load_render_settings(shot.work) == got and store.state(shot.work)["render_settings"] == got


def test_clean_first_takes_the_clear_plate_before_the_keys(tmp_path):
    """Motion control (2026-11-10): with clean_first the aligned clear plate fills the patch where it is usable, the
    keys only the rest; without it the keys win (the default)."""
    shot = Shot(tmp_path)
    blue = np.array([0.8, 0.2, 0.1], np.float32)
    half = np.zeros((H, W), np.float32)
    half[:, : W // 2] = 1.0                                     # the clear plate is usable on the left half only

    def clean(f):
        return np.broadcast_to(blue, (H, W, 3)).astype(np.float32), half

    def run(**kw):
        st = {**SMALL, "remove": ["obj"], "grow_px": 4.0, "soft_px": 3.0, "motion_blur": "off", **kw}
        rj = RD.render(shot.work, [3], plate=shot.plate, remove_alphas=lambda f: [obj_alpha(f)], settings=st, clean=clean)
        return cv2.imread(str(Path(rj["dir"]) / "source" / "s_0003.png"), 0), rj
    m = cu.removal_mask([obj_alpha(3)], 4.0, 3.0) > 0
    left, right = m & (half > 0.5), m & (half < 0.5)
    src, _ = run()
    assert (src[m] == cu.SRC_KEY).all()                          # default: the keys first
    src, rj = run(clean_first=True)
    assert (src[left] == cu.SRC_CLEAN).all() and (src[right] == cu.SRC_KEY).all()
    fr = rj["c3d"]["frames"]
    assert (fr.get(3) or fr.get("3"))["clean_first_px"] > 0


# ------------------------------------------------------------------------------------------------ grain (P17)
GRAIN_SIGMA = 0.03


class NoisyShot(Shot):
    """The synthetic shot with a not-denoised plate: gaussian grain, new every frame; the keys stay clean."""

    def plate(self, f):
        n = np.random.default_rng(100 + f).standard_normal((H, W, 3)).astype(np.float32) * GRAIN_SIGMA
        return (super().plate(f) + n).astype(np.float32)


def _grain(rj, f):
    fr = rj["c3d"]["frames"]
    return (fr.get(f) or fr.get(str(f)) or {}).get("grain")


def test_grain_added_only_for_a_noisy_plate(tmp_path):
    """P17: auto adds grain to the patch when the plate is noisy (measured level within 15 %, the patch then as noisy
    as the plate), nothing for a clean plate; off = nothing; outside the patch the plate is untouched."""
    noisy = NoisyShot(tmp_path / "a")
    rj, comp, src, _ = noisy.render(frames=[3, 6])
    rj_off, off, _, _ = noisy.render(frames=[3, 6], grain="off")
    assert _grain(rj_off, 3) is None
    assert rj["c3d"]["settings"]["grain"] == "auto" and rj["c3d"]["settings"]["grain_strength"] == 1.0
    for f in (3, 6):
        g = _grain(rj, f)
        assert g["added"] and np.all(np.abs(np.array(g["plate"]) - GRAIN_SIGMA) < 0.15 * GRAIN_SIGMA), g
        assert np.mean(g["patch"]) < 0.3 * GRAIN_SIGMA
        m = obj_alpha(f) > 0
        res_on = (comp[f] - noisy.bg[f])[m]
        res_off = (off[f] - noisy.bg[f])[m]
        assert res_off.std() < 0.3 * GRAIN_SIGMA                                    # the keys are clean
        assert abs(float(res_on.std()) - GRAIN_SIGMA) < 0.15 * GRAIN_SIGMA, res_on.std()
        assert abs(float(res_on.mean())) < 0.005                                    # grain, not a tone shift
        mask = cu.removal_mask([obj_alpha(f)], 4.0, 3.0)
        ramp = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * KEY_SOFT + 3,) * 2, np.uint8)) > 0
        plate8 = np.round(np.clip(noisy.plate(f), 0, 1) * 255) / 255
        assert np.abs(comp[f] - plate8)[~ramp].max() <= 0.5 / 255 + 1e-6           # untouched outside the patch
    # the same grain on a re-render (seeded by the frame)
    again = noisy.render(frames=[3])[1]
    assert np.array_equal(again[3], comp[3])
    # a clean plate: measured, nothing added, the picture as with grain off
    clean = Shot(tmp_path / "b")
    rj2, on2, _, _ = clean.render(frames=[3])
    _, off2, _, _ = clean.render(frames=[3], grain="off")
    g = _grain(rj2, 3)
    assert g is not None and not g["added"] and np.array_equal(on2[3], off2[3])
    from engine.cleanup3d import grain as GR
    s = GR.summary(rj["c3d"]["frames"])
    assert s["added"] == 2 and s["measured"] == 2 and abs(s["plate"] - GRAIN_SIGMA) < 0.15 * GRAIN_SIGMA


def test_grain_differs_between_frames_in_the_render(tmp_path):
    """Not a fixed pattern: the grain added in two frames is uncorrelated where the hole of both frames overlaps."""
    noisy = NoisyShot(tmp_path)
    rj, comp, _, _ = noisy.render(frames=[4, 5])
    both = (obj_alpha(4) > 0) & (obj_alpha(5) > 0)
    assert both.sum() > 300
    a, b = (comp[4] - noisy.bg[4])[both].ravel(), (comp[5] - noisy.bg[5])[both].ravel()
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.15
