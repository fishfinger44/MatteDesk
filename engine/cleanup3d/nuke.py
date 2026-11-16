# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – export to Nuke (docs/PLAN_CLEANUP_3D.md, package P11, step C6).

From the contract files of the shot (camera3d/cams.json, cards.json, keys/) a Nuke script that rebuilds the projection:
the shot camera (Camera3, a key per frame) + a `.chan` file, every card as Card3D (a plane card, sized to cover what the
camera sees) or a large Sphere (the sky, at a distance where the camera's travel is negligible), every key as
Read -> Project3D with its OWN static projector camera at the key frame -> ApplyMaterial on the card -> ScanlineRender
with the shot camera; the keys of one card are crossfaded by 2D Dissolves with the same smoothstep weights as the
render (render.key_pair); a card's extent is projected the same way as a mask; cards are merged front over back (the
card order); the lens distortion is applied by STMaps we compute ourselves (exact for our k1 k2 p1 p2 model); finally
Merge over the plate Read through the removal mask (the alpha of the last render's patch) when there is one.
The .nk text is built with the helpers of engine/nuke_nk.py (the one .nk writer; animated XYZ knob = one outer brace
pair, `{{curve ..} {curve ..} {curve ..}}`).

Axis conventions (the core of this module, checked by tests/test_cleanup3d_nuke.py on known poses):
- Our Camera3D (engine/cleanup3d/types.py): OpenCV – X_cam = R @ X_world + t, camera x right, y DOWN, z FORWARD;
  pixels: origin top-left, y down, pixel centre at integer coordinates.
- Nuke camera: local x right, y UP, looking down -Z; pixels: origin bottom-left, pixel centre at +0.5.
  Camera axes: Nuke_local = F @ OpenCV_local, F = diag(1, -1, -1).
- World: Nuke_world = WORLD @ our_world, WORLD = diag(1, -1, -1) (a proper rotation, 180 deg about X), so a solve
  whose world is "y down" like its cameras shows Y up in Nuke's 3D viewer.
- So the Nuke camera's world matrix: rotation Rn = WORLD @ R.T @ F, translation = WORLD @ C, C = -R.T @ t.
- Euler angles: Nuke's default rot_order ZXY = rotate about Z first, then X, then Y -> Rn = Ry @ Rx @ Rz (degrees),
  transform order SRT. Angles are unwrapped frame to frame (no 360 deg jumps in the curves).
- Lens: haperture = 24.576 mm (Nuke's default), focal = fx * haperture / W; win_translate = principal point offset
  in units of half the image WIDTH (Nuke's screen window: x in [-1, 1], y in [-H/W, H/W]); win_scale y = fx / fy.
  `nuke_project` below is exactly this model of Nuke's camera (what the tests compare with our projection).
Not in the script (said in the sticky note): motion blur matched per frame, the tone match of the patch, the key
coverage fallback (the next key filling where the current one does not reach) and the renormalised soft seams – the
render of MatteDesk (step C5) does these; the script is the starting point to finish the projection in Nuke.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from engine import nuke_nk as NK
from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d.geom import card_plane
from engine.cleanup3d.types import Camera3D, Card, Key

SCHEMA = "matte_desk/cleanup3d_nuke@1.0"
F_AXES = np.diag([1.0, -1.0, -1.0])        # OpenCV camera axes -> Nuke camera axes
WORLD = np.diag([1.0, -1.0, -1.0])         # our world -> Nuke world (proper rotation, det +1)
HAPERTURE = 24.576                         # mm, Nuke's default film back width
GRID = 9                                   # rays per image side sampled to size a card
MAX_SAMPLE_FRAMES = 32
FAR_FACTOR = 200.0                         # a card reaches at most this many camera heights from the camera
CARD_MARGIN = 1.2                          # card size = what the camera sees x this
SKY_FACTOR = 2000.0                        # sky sphere radius = this x the scene scale (parallax < 1/SKY_FACTOR rad)
ND = 6                                     # decimals in the curves


# ----------------------------------------------------------------------------------------------- rotations
def rot_zxy(rx: float, ry: float, rz: float) -> np.ndarray:
    """Nuke's rotation for rot_order ZXY (degrees): Ry @ Rx @ Rz."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def euler_zxy(Rm: np.ndarray) -> Tuple[float, float, float]:
    """(rx, ry, rz) degrees with Rm = Ry @ Rx @ Rz (inverse of rot_zxy)."""
    Rm = np.asarray(Rm, np.float64)
    rx = math.asin(max(-1.0, min(1.0, -Rm[1, 2])))
    if abs(math.cos(rx)) > 1e-9:
        ry = math.atan2(Rm[0, 2], Rm[2, 2])
        rz = math.atan2(Rm[1, 0], Rm[1, 1])
    else:                                  # gimbal lock: put everything into ry
        ry = math.atan2(-Rm[2, 0], Rm[0, 0])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def unwrap_deg(prev: Sequence[float], cur: Sequence[float]) -> List[float]:
    """`cur` angles moved by multiples of 360 to be nearest to `prev` (continuous curves)."""
    return [c + 360.0 * round((p - c) / 360.0) for p, c in zip(prev, cur)]


# ----------------------------------------------------------------------------------------------- camera
def nuke_pose(R: np.ndarray, t: np.ndarray, world: np.ndarray = WORLD) -> Tuple[np.ndarray, np.ndarray]:
    """Our pose (X_cam = R X + t, OpenCV axes) -> (Nuke translate (3,), Nuke rotation matrix 3x3) in the Nuke world."""
    R = np.asarray(R, np.float64)
    C = -R.T @ np.asarray(t, np.float64)
    return world @ C, world @ R.T @ F_AXES


def lens_knobs(K: np.ndarray, w: int, h: int, haperture: float = HAPERTURE) -> Dict[str, Any]:
    """Nuke camera lens of our pinhole K for a w x h image (see the module docstring)."""
    K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    half = w / 2.0
    return {"focal": fx * haperture / w, "haperture": haperture, "vaperture": haperture * h / w,
            "win_translate": ((cx + 0.5 - half) / half, (h / 2.0 - 0.5 - cy) / half),
            "win_scale": (1.0, fx / fy)}


def nuke_project(p: np.ndarray, translate: Sequence[float], rotate: Sequence[float], lens: Mapping[str, Any],
                 w: int, h: int) -> Optional[Tuple[float, float]]:
    """A Nuke-world point -> Nuke pixel (origin bottom-left, centre +0.5) through a Nuke camera as we model it.
    None behind the camera."""
    M = rot_zxy(*rotate)
    loc = M.T @ (np.asarray(p, np.float64) - np.asarray(translate, np.float64))
    if loc[2] >= 0:
        return None
    k = 2.0 * float(lens["focal"]) / float(lens["haperture"])
    xs = k * loc[0] / -loc[2] / lens["win_scale"][0] + lens["win_translate"][0]
    ys = k * loc[1] / -loc[2] / lens["win_scale"][1] + lens["win_translate"][1]
    return (xs + 1.0) * w / 2.0, ys * w / 2.0 + h / 2.0


def camera_curves(camera: Camera3D, frames: Optional[Sequence[int]] = None,
                  world: np.ndarray = WORLD) -> Tuple[List[int], List[np.ndarray], List[List[float]]]:
    """(frames, translates, rotates in degrees, unwrapped) of the shot camera in Nuke's convention."""
    frames = [int(f) for f in (frames if frames is not None else camera.frames)]
    tr, rot = [], []
    prev = None
    for f in frames:
        T, Rn = nuke_pose(camera.R[f], camera.t[f], world)
        e = list(euler_zxy(Rn))
        if prev is not None:
            e = unwrap_deg(prev, e)
        prev = e
        tr.append(T)
        rot.append(e)
    return frames, tr, rot


def chan_text(camera: Camera3D, frames: Optional[Sequence[int]] = None, world: np.ndarray = WORLD) -> str:
    """Nuke .chan of the shot camera: `frame tx ty tz rx ry rz vfov` per line (rot_order ZXY, degrees; vfov = the
    vertical field of view in degrees). Import on a Camera: File -> Import chan file."""
    fr, tr, rot = camera_curves(camera, frames, world)
    fy, h = float(np.asarray(camera.K)[1, 1]), int(camera.height)
    vfov = math.degrees(2.0 * math.atan(h / (2.0 * fy)))
    return "".join(f"{f} " + " ".join(f"{v:.{ND}f}" for v in (*T, *r, vfov)) + "\n" for f, T, r in zip(fr, tr, rot))


# ----------------------------------------------------------------------------------------------- lens distortion
def stmaps(K: np.ndarray, dist: Sequence[float], w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """(undistort, distort) STMaps, float32 (h, w, 3) in cv2 BGR order (R = s, G = t, B = 0) for Nuke's STMap node:
    s = (x + 0.5) / w, t = (h - 0.5 - y) / h of the SOURCE pixel (OpenCV coords).
    undistort: for a pinhole pixel, where to read in the distorted image (a key as shot -> pinhole before Project3D);
    distort: for a plate (distorted) pixel, where to read in the pinhole image (ScanlineRender -> the plate's lens)."""
    K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xn, yn = (xs - cx) / fx, (ys - cy) / fy
    xd, yd = K_.distort(xn, yn, dist)
    xu, yu = K_.undistort(K_.NP, xn, yn, dist)

    def st(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        s = (px * fx + cx + 0.5) / w
        t = (h - 0.5 - (py * fy + cy)) / h
        return np.dstack([np.zeros_like(s), t, s]).astype(np.float32)
    return st(xd, yd), st(xu, yu)


def has_dist(dist: Optional[Sequence[float]]) -> bool:
    return dist is not None and any(abs(float(v)) > 1e-12 for v in dist)


def undistort_canvas(canvas: np.ndarray, K: np.ndarray, dist: Sequence[float], ox: int, oy: int) -> np.ndarray:
    """An extent canvas (plate pixels of extent_frame, distorted, with a margin ox / oy) -> the same canvas as a
    pinhole image with K' = K shifted by (ox, oy): what a Project3D with that camera projects."""
    import cv2
    if not has_dist(dist):
        return canvas
    K = np.asarray(K, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = canvas.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xd, yd = K_.distort((xs - ox - cx) / fx, (ys - oy - cy) / fy, dist)
    mx = (xd * fx + cx + ox).astype(np.float32)
    my = (yd * fy + cy + oy).astype(np.float32)
    return cv2.remap(canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------------------------- cards
def _sample_frames(frames: Sequence[int]) -> List[int]:
    step = max(1, int(math.ceil(len(frames) / float(MAX_SAMPLE_FRAMES))))
    out = list(frames[::step])
    if frames and out[-1] != frames[-1]:
        out.append(frames[-1])
    return out


def _rays(camera: Camera3D, frame: int) -> Tuple[np.ndarray, np.ndarray]:
    """(centre (3,), unit directions (n, 3)) in our world of a GRID x GRID of pixels of `frame` (lens undistorted)."""
    K = np.asarray(camera.K, np.float64)
    xs, ys = np.meshgrid(np.linspace(0, camera.width - 1, GRID), np.linspace(0, camera.height - 1, GRID))
    xn, yn = (xs.ravel() - K[0, 2]) / K[0, 0], (ys.ravel() - K[1, 2]) / K[1, 1]
    xn, yn = K_.undistort(K_.NP, xn, yn, camera.dist) if has_dist(camera.dist) else (xn, yn)
    d = np.stack([xn, yn, np.ones_like(xn)], 1) @ camera.R[frame]          # rows: R.T @ [x, y, 1]
    return camera.centre(frame), d / np.linalg.norm(d, axis=1, keepdims=True)


def scene_scale(camera: Camera3D) -> float:
    """A length of the shot: the camera's travel, at least 1e-3 (a locked-off camera)."""
    c = np.array([camera.centre(f) for f in camera.frames])
    return float(max(np.max(np.linalg.norm(c - c.mean(0), axis=1)) * 2.0 if len(c) else 0.0, 1e-3))


def card_geometry(camera: Camera3D, card: Card, world: np.ndarray = WORLD) -> Dict[str, Any]:
    """Where the card goes in Nuke. Plane: {"kind": "plane", "translate", "rotate", "scaling" (sx, sy), "n", "d"} –
    a Card3D (1 x 1, image_aspect off) with local +Z = the plane normal toward the cameras, centred on and sized to
    what the camera sees of the plane (rays of every sampled frame, cut at FAR_FACTOR camera heights) x CARD_MARGIN.
    Sky: {"kind": "sky", "translate", "radius"} – a Sphere around the cameras' mean centre."""
    frames = camera.frames
    if not frames:
        raise ValueError("camera3d: brak klatek")
    centres = np.array([camera.centre(f) for f in frames])
    scale = scene_scale(camera)
    pl = card_plane(camera, card)
    if pl is None:
        r = SKY_FACTOR * scale
        return {"kind": "sky", "translate": world @ centres.mean(0), "radius": r}
    n, d = np.asarray(pl[0], np.float64), float(pl[1])
    heights = np.abs(centres @ n + d)
    far = FAR_FACTOR * max(float(np.median(heights)), 1e-6)
    pts = []
    for f in _sample_frames(frames):
        C, dirs = _rays(camera, f)
        den = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -(C @ n + d) / den
        ok = np.isfinite(s) & (s > 0)
        s = np.where(ok, np.minimum(s, far), far)
        p = C + dirs * s[:, None]
        p = p - (p @ n + d)[:, None] * n                 # cut rays (horizon, sky) land on the plane under them
        pts.append(p)
    pts = np.concatenate(pts)
    fwd = np.mean([camera.R[f].T @ np.array([0, 0, 1.0]) for f in frames], 0)
    u = fwd - (fwd @ n) * n
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    a, b = pts @ u, pts @ v
    mid = (a.max() + a.min()) / 2.0 * u + (b.max() + b.min()) / 2.0 * v - d * n
    size = (max(float(a.max() - a.min()), 1e-3) * CARD_MARGIN, max(float(b.max() - b.min()), 1e-3) * CARD_MARGIN)
    Rc = world @ np.stack([u, v, n], 1)                  # card local x, y, z (= normal) in the Nuke world
    return {"kind": "plane", "translate": world @ mid, "rotate": euler_zxy(Rc), "scaling": size,
            "n": world @ n, "d": d, "far": far}


# ----------------------------------------------------------------------------------------------- keys
def key_projector(camera: Camera3D, key: Key) -> Dict[str, Any]:
    """The projector of a key: pose of its frame (the canvas' frame for a key with its own canvas), K, size and whether
    the image is distorted (a key as shot carries the plate's lens; a canvas is pinhole)."""
    cv = key.canvas
    f = int(cv.get("frame", key.frame)) if cv else int(key.frame)
    if f not in camera.R:
        raise KeyError(f)
    if cv:
        return {"frame": f, "K": np.asarray(cv["K"], np.float64), "w": int(cv["w"]), "h": int(cv["h"]),
                "distorted": False}
    return {"frame": f, "K": np.asarray(camera.K, np.float64), "w": int(camera.width), "h": int(camera.height),
            "distorted": has_dist(camera.dist)}


def smoothstep(x: float) -> float:
    """As engine/cleanup3d/render.smoothstep (the render's crossfade between neighbouring keys)."""
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def dissolve_weights(key_frames: Sequence[int], frames: Sequence[int]) -> List[List[float]]:
    """For keys k0 < k1 < ... the `which` of Dissolve i (i = 1..n-1, chained: D_i = dissolve(D_{i-1}, key_i)) at every
    frame: 0 up to k_{i-1}, smoothstep across (k_{i-1}, k_i), 1 from k_i. The chain gives exactly render.key_pair's
    (1 - w, w) of the two keys around a frame and the nearest key alone outside the keys' range."""
    ks = sorted(int(k) for k in key_frames)
    return [[smoothstep((f - ks[i - 1]) / float(ks[i] - ks[i - 1])) for f in frames] for i in range(1, len(ks))]


# ----------------------------------------------------------------------------------------------- .nk text
def nk_str(s: str) -> str:
    """A quoted .nk string (TCL: escape backslash, quotes, brackets, dollar, braces)."""
    out = str(s)
    for ch in ("\\", '"', "[", "]", "$", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return f'"{out}"'


def node_name(*parts: Any) -> str:
    s = "_".join(str(p) for p in parts)
    s = "".join(ch if ch.isalnum() and ch.isascii() else "_" for ch in s)
    return s if s[:1].isalpha() else "N" + s


def path_nk(p: Any) -> str:
    return str(p).replace("\\", "/")


def _num(v: float, nd: int = ND) -> str:
    return NK._fmt(float(v), nd)


def _vec(name: str, vals: Sequence[float], nd: int = ND) -> str:
    return f"{name} {{" + " ".join(_num(v, nd) for v in vals) + "}"


class NkScript:
    """A tiny writer of a whole Nuke script: nodes are written in order, each takes its inputs by name, then is saved
    with `set N_<name> [stack 0]`. Inputs are pushed LAST INPUT FIRST, so input 0 is the top of the stack (the last
    `push`) – as Nuke writes its own scripts (the main pipe, input 0, runs straight into the node) and as the other
    generators of the repo assume (LOCAL_CHECKS audit 5); `push 0` = not connected."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.names: List[str] = []
        self.classes: Dict[str, str] = {}

    def node(self, cls: str, name: str, knobs: Sequence[str] = (), inputs: Sequence[Optional[str]] = (),
             x: int = 0, y: int = 0) -> str:
        if name in self.classes:
            raise ValueError(f"nk: powtórzona nazwa węzła {name}")
        for i in reversed(list(inputs)):
            if i is not None and i not in self.classes:
                raise ValueError(f"nk: węzeł {name} – nieznane wejście {i}")
            self.lines.append(f"push $N_{i}" if i is not None else "push 0")
        self.lines.append(f"{cls} {{")
        self.lines.append(f" inputs {len(inputs)}")
        self.lines += [" " + k for k in knobs]
        self.lines += [f" name {name}", f" xpos {int(x)}", f" ypos {int(y)}", "}"]
        self.lines.append(f"set N_{name} [stack 0]")
        self.names.append(name)
        self.classes[name] = cls
        return name

    def text(self, header: Sequence[str]) -> str:
        return "\n".join(list(header) + self.lines) + "\n"


def _camera_knobs(lens: Mapping[str, Any], near: float, far: float) -> List[str]:
    return [f"focal {_num(lens['focal'])}", f"haperture {_num(lens['haperture'])}",
            f"vaperture {_num(lens['vaperture'])}", f"near {_num(near)}", f"far {_num(far, 3)}",
            _vec("win_translate", lens["win_translate"]), _vec("win_scale", lens["win_scale"])]


def build_script(camera: Camera3D, cards: Sequence[Card], keys: Sequence[Key], *, key_files: Mapping[Tuple[str, int], str],
                 plate_pattern: str, nk_path: str, frames: Optional[Sequence[int]] = None,
                 patch: Optional[Mapping[str, Any]] = None, stmap_files: Optional[Mapping[str, str]] = None,
                 extents: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 world: np.ndarray = WORLD) -> Tuple[str, Dict[str, Any]]:
    """The .nk text + a report {"counts", "notes", "cards"}. Pure (no file is read or written).
    key_files: (card, frame) -> image of the key's active version; patch: {"pattern", "first", "last"} of the last
    render's patch (its alpha = the removal mask) or None; stmap_files: {"undistort", "distort"} (only with
    distortion); extents: card id -> {"file", "frame", "K" (3x3 of the canvas), "w", "h", "seam_px"} (cards without
    an extent are the whole card)."""
    frames = [int(f) for f in (frames if frames is not None else camera.frames)]
    if not frames:
        raise ValueError("Kamera 3D nie ma klatek")
    missing = [f for f in frames if f not in camera.R]
    if missing:
        raise ValueError(f"Kamera 3D nie ma klatek: {missing[0]}–{missing[-1]}")
    w, h = int(camera.width), int(camera.height)
    first, last = frames[0], frames[-1]
    fmt_name = f"C3D_{w}x{h}"
    fmt = f"{w} {h} 0 0 {w} {h} 1 {fmt_name}"
    notes: List[str] = []
    dist = has_dist(camera.dist) and bool(stmap_files)
    if has_dist(camera.dist) and not stmap_files:
        notes.append("Dystorsja kamery pominięta (brak STMap).")
    scale = scene_scale(camera)
    geo = {c.id: card_geometry(camera, c, world) for c in cards}
    far = 4.0 * max([g["radius"] for g in geo.values() if g["kind"] == "sky"] +
                    [g["far"] * 2.0 for g in geo.values() if g["kind"] == "plane"] + [100.0 * scale])
    near = max(1e-4, 1e-3 * scale)
    lens = lens_knobs(camera.K, w, h)
    S = NkScript()

    # --- shot camera
    fr, tr, rot = camera_curves(camera, frames, world)
    cam = S.node("Camera3", "Camera_shot", [
        NK.vec_knob("translate", fr, [[T[i] for T in tr] for i in range(3)], ND),
        NK.vec_knob("rotate", fr, [[r[i] for r in rot] for i in range(3)], ND),
        "rot_order ZXY", *_camera_knobs(lens, near, far),
        f"label {nk_str('MatteDesk Camera3D: ' + (camera.source or ''))}"], x=0, y=-400)

    # --- STMaps
    und = dis = None
    if dist:
        und = S.node("Read", "Read_stmap_undistort", [f"file {nk_str(path_nk(stmap_files['undistort']))}",
                                                      "first 1", "last 1", "raw true"], x=-300, y=-600)
        dis = S.node("Read", "Read_stmap_distort", [f"file {nk_str(path_nk(stmap_files['distort']))}",
                                                    "first 1", "last 1", "raw true"], x=-150, y=-600)

    by_card: Dict[str, List[Key]] = {}
    for k in keys:
        if k.active and (k.card, int(k.frame)) in key_files:
            by_card.setdefault(k.card, []).append(k)
    card_out: List[Tuple[Card, str]] = []
    counts = {"cards": 0, "keys": 0, "project3d": 0, "scanline": 0, "dissolve": 0, "extents": 0}
    rep_cards = []
    col = 0
    for c in sorted(cards, key=lambda c: c.order):
        ks = sorted(by_card.get(c.id) or [], key=lambda k: k.frame)
        good = []
        for k in ks:
            try:
                good.append((k, key_projector(camera, k)))
            except KeyError:
                notes.append(f"Klucz {c.id} / F{k.frame}: kamera 3D nie ma tej klatki – pominięty.")
        if not good:
            notes.append(f"Karta {c.id}: brak kluczy z aktywną wersją – pominięta.")
            continue
        g = geo[c.id]
        x0 = col * 900
        col += 1
        cid = node_name(c.id)
        if g["kind"] == "sky":
            geo_node = S.node("Sphere", f"Sky_{cid}", [f"radius {_num(g['radius'], 3)}", "rows 64", "columns 64",
                                                       _vec("translate", g["translate"]),
                                                       f"label {nk_str(c.name + ' (niebo)')}"], x=x0, y=0)
        else:
            geo_node = S.node("Card2", f"Card_{cid}", [
                "image_aspect false", "rows 8", "columns 8", "rot_order ZXY",
                _vec("translate", g["translate"]), _vec("rotate", g["rotate"]),
                _vec("scaling", [g["scaling"][0], g["scaling"][1], 1.0]), f"label {nk_str(c.name)}"], x=x0, y=0)
        counts["cards"] += 1
        renders = []
        for i, (k, pj) in enumerate(good):
            xk = x0 + i * 150
            tag = f"{cid}_{int(k.frame):04d}"
            rd = S.node("Read", f"Read_key_{tag}", [f"file {nk_str(path_nk(key_files[(k.card, int(k.frame))]))}",
                                                    "first 1", "last 1", "before hold", "after hold",
                                                    f"label {nk_str(f'F{k.frame} {k.active}')}"], x=xk, y=-300)
            src = S.node("Shuffle", f"Alpha_key_{tag}", ["alpha white"], [rd], x=xk, y=-260)
            if pj["distorted"] and und:
                src = S.node("STMap", f"Undistort_key_{tag}", ["uv rgb"], [src, und], x=xk, y=-220)
            T, Rn = nuke_pose(camera.R[pj["frame"]], camera.t[pj["frame"]], world)
            pcam = S.node("Camera3", f"Proj_{tag}", [_vec("translate", T), _vec("rotate", euler_zxy(Rn)), "rot_order ZXY",
                                                     *_camera_knobs(lens_knobs(pj["K"], pj["w"], pj["h"]), near, far),
                                                     f"label {nk_str('projector F' + str(pj['frame']))}"],
                          x=xk + 60, y=-180)
            pr = S.node("Project3D2", f"Project_{tag}", ["crop true"], [src, pcam], x=xk, y=-120)
            am = S.node("ApplyMaterial", f"Material_{tag}", [], [geo_node, pr], x=xk, y=-60)
            sr = S.node("ScanlineRender", f"Render_{tag}", ["projection_mode perspective"], [None, am, cam], x=xk, y=60)
            renders.append(sr)
            counts["keys"] += 1
            counts["project3d"] += 1
            counts["scanline"] += 1
        out = renders[0]
        wts = dissolve_weights([k.frame for k, _ in good], frames)
        for i in range(1, len(renders)):
            out = S.node("Dissolve", f"Dissolve_{cid}_{i}", [NK.scalar_knob("which", frames, wts[i - 1], ND)],
                         [out, renders[i]], x=x0 + i * 150, y=140 + 40 * i)
            counts["dissolve"] += 1
        ext = (extents or {}).get(c.id)
        if ext:
            xe = x0 + len(renders) * 150 + 100
            rd = S.node("Read", f"Read_extent_{cid}", [f"file {nk_str(path_nk(ext['file']))}", "first 1", "last 1",
                                                       "raw true"], x=xe, y=-300)
            al = S.node("Shuffle", f"Alpha_extent_{cid}", ["alpha red"], [rd], x=xe, y=-260)
            T, Rn = nuke_pose(camera.R[int(ext["frame"])], camera.t[int(ext["frame"])], world)
            ecam = S.node("Camera3", f"Proj_extent_{cid}", [
                _vec("translate", T), _vec("rotate", euler_zxy(Rn)), "rot_order ZXY",
                *_camera_knobs(lens_knobs(ext["K"], int(ext["w"]), int(ext["h"])), near, far)], x=xe + 60, y=-180)
            pr = S.node("Project3D2", f"Project_extent_{cid}", ["crop false"], [al, ecam], x=xe, y=-120)
            am = S.node("ApplyMaterial", f"Material_extent_{cid}", [], [geo_node, pr], x=xe, y=-60)
            sr = S.node("ScanlineRender", f"Render_extent_{cid}", ["projection_mode perspective"], [None, am, cam],
                        x=xe, y=60)
            m = sr
            if float(ext.get("seam_px") or 0) > 0:
                m = S.node("Blur", f"Seam_{cid}", ["channels alpha", f"size {_num(ext['seam_px'], 2)}"], [sr], x=xe, y=120)
            out = S.node("Merge2", f"Extent_{cid}", ["operation mask"], [out, m], x=x0, y=260)
            counts["extents"] += 1
            counts["project3d"] += 1
            counts["scanline"] += 1
        card_out.append((c, out))
        rep_cards.append({"id": c.id, "kind": g["kind"], "keys": [int(k.frame) for k, _ in good], "extent": bool(ext)})
    if not card_out:
        raise ValueError("Brak kart z kluczami – nie ma czego wyeksportować do Nuke'a")

    # --- cards front over back (order 0 = in front, as the render's card_weight)
    acc = card_out[-1][1]
    for i, (c, out) in enumerate(reversed(card_out[:-1])):
        acc = S.node("Merge2", f"Over_{node_name(c.id)}", ["operation over"], [acc, out], x=0, y=340 + 40 * i)
    if dis:
        acc = S.node("STMap", "Distort_lens", ["uv rgb"], [acc, dis], x=0, y=460)
    plate = S.node("Read", "Read_plate", [f"file {nk_str(path_nk(plate_pattern))}", f"format {nk_str(fmt)}",
                                          f"first {first}", f"last {last}", f"origfirst {first}", f"origlast {last}",
                                          "origset true"], x=-400, y=420)
    ins: List[Optional[str]] = [plate, acc]
    knobs = ["operation over"]
    if patch:
        pm = S.node("Read", "Read_removal_mask", [f"file {nk_str(path_nk(patch['pattern']))}",
                                                  f"first {int(patch['first'])}", f"last {int(patch['last'])}",
                                                  f"origfirst {int(patch['first'])}", f"origlast {int(patch['last'])}",
                                                  "origset true", "before black", "after black",
                                                  f"label {nk_str('alpha = maska usuwania ' + str(patch.get('version') or ''))}"],
                    x=300, y=420)
        ins.append(pm)
        knobs.append("maskChannelInput rgba.alpha")
    else:
        notes.append("Brak przebiegu projekcji (C5) – Merge bez maski usuwania; podłącz maskę ręcznie.")
    S.node("Merge2", "Comp_cleanup3d", knobs, ins, x=0, y=540)
    note = ("MatteDesk Cleanup 3D -> Nuke. Camera_shot = solved camera (Nuke axes: Y up, looks down -Z; world = ours "
            "rotated 180 deg about X). Keys: Read -> Project3D (own projector camera at the key frame) -> card; keys of a "
            "card crossfade by Dissolve (smoothstep, as the MatteDesk render). Not here: matched motion blur, tone "
            "match, key coverage fallback, renormalised seams.")
    S.node("StickyNote", "Note_MatteDesk", [f"label {nk_str(note)}", "note_font_size 14"], x=-400, y=-600)
    header = ["#! nuke -nx", "version 14.0 v1", "Root {", " inputs 0", f" name {nk_str(path_nk(nk_path))}",
              f" frame {first}", f" first_frame {first}", f" last_frame {last}", " lock_range true",
              f" format {nk_str(fmt)}", "}"]
    text = S.text(header)
    probs = NK.validate_script(text)
    if probs:
        raise ValueError("nk: " + "; ".join(probs))
    return text, {"counts": {**counts, "nodes": len(S.names)}, "notes": notes, "cards": rep_cards,
                  "frames": [first, last], "classes": dict(S.classes)}


# ----------------------------------------------------------------------------------------------- files
def export(work: Path, *, plate_pattern: str, out_dir: Optional[Path] = None, tag: Optional[str] = None,
           patch: Optional[Mapping[str, Any]] = None, frames: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Write `<out_dir>/c3d_<tag>.nk` + `.chan` (+ STMaps and extent masks next to them) from the contract files of
    `work` (`<work>/matte_desk`). out_dir default `<work>/inpaint/nuke/`, tag default the date. Returns the paths and
    the report. ValueError (Polish) when the camera, the cards or the keys are missing."""
    from engine.cleanup3d import keys as KS
    from engine.cleanup3d import render as RD
    from engine.cleanup3d import store
    from engine.cleanup3d import types as T
    from engine.cleanup3d.geom import extent_canvas
    import cv2

    work = Path(work)
    cam = store.load_camera(work)
    if cam is None:
        raise ValueError("Najpierw policz kamerę 3D (krok C2)")
    cards = T.active_cards(store.load_cards(work))     # P16: off (eye) / not soloed (S) = not exported
    if not cards:
        raise ValueError("Brak kart (albo wszystkie wyłączone / poza solo) – dodaj / włącz karty w kroku C3")
    ids = {c.id for c in cards}
    keys = [k for k in store.list_keys(work) if k.active and k.card in ids]
    if not keys:
        raise ValueError("Brak kluczy z aktywną wersją – dodaj klucze w kroku C4")
    out_dir = Path(out_dir) if out_dir is not None else T.root_dir(work) / "nuke"
    tag = tag or time.strftime("%Y%m%d_%H%M%S")
    base = f"c3d_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    key_files = {}
    for k in keys:
        p = KS.version_path(work, k)
        if p.is_file():
            key_files[(k.card, int(k.frame))] = str(p.resolve())
    w, h = int(cam.width), int(cam.height)
    files: Dict[str, Any] = {}
    stm = None
    if has_dist(cam.dist):
        und, dis = stmaps(cam.K, cam.dist, w, h)
        stm = {"undistort": out_dir / f"{base}_stmap_undistort.exr", "distort": out_dir / f"{base}_stmap_distort.exr"}
        KS.write_image(stm["undistort"], und, "exr")
        KS.write_image(stm["distort"], dis, "exr")
        stm = {k: str(v.resolve()) for k, v in stm.items()}
        files["stmaps"] = list(stm.values())
    layers = RD.extent_layers(work, cards)
    extents: Dict[str, Dict[str, Any]] = {}
    for c in cards:
        if not c.extent:
            continue
        if c.extent_frame is None or int(c.extent_frame) not in cam.R:
            continue
        canvas, ox, oy, _base = extent_canvas(c, w, h, layers.get(c.id))
        img = undistort_canvas(canvas, cam.K, cam.dist, ox, oy)
        p = out_dir / f"{base}_extent_{c.id}.png"
        if not cv2.imwrite(str(p), img):
            raise ValueError(f"Nie udało się zapisać {p.name}")
        Kc = np.asarray(cam.K, np.float64).copy()
        Kc[0, 2] += ox
        Kc[1, 2] += oy
        extents[c.id] = {"file": str(p.resolve()), "frame": int(c.extent_frame), "K": Kc, "w": canvas.shape[1],
                         "h": canvas.shape[0], "seam_px": float(c.seam_px)}
    files["extents"] = [e["file"] for e in extents.values()]
    nk = out_dir / f"{base}.nk"
    text, rep = build_script(cam, cards, keys, key_files=key_files, plate_pattern=plate_pattern,
                             nk_path=str(nk.resolve()), frames=frames, patch=patch, stmap_files=stm, extents=extents)
    nk.write_text(text, encoding="utf-8", newline="\n")
    chan = out_dir / f"{base}.chan"
    chan.write_text(chan_text(cam, frames), encoding="utf-8", newline="\n")
    rep.pop("classes", None)
    lost = [f"{k.card}/F{k.frame}" for k in keys if (k.card, int(k.frame)) not in key_files]
    if lost:
        rep["notes"].append(f"Brak plików aktywnych wersji kluczy: {', '.join(lost)}")
    return {"schema": SCHEMA, "nk": path_nk(nk.resolve()), "chan": path_nk(chan.resolve()),
            "dir": path_nk(out_dir.resolve()),
            **{k: [path_nk(p) for p in v] for k, v in files.items()}, **rep}
