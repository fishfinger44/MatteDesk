# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cleanup3d/geom_gpu.py - the per-pixel maths of the Cleanup 3D geometry (package P1) for BOTH backends, and the
GPU worker that runs it in ComfyUI's Python (its PyTorch + CUDA).

ONE implementation: every function takes a backend `B` (NP = numpy, or a torch backend built in the worker) and is
written with operators / B-methods only, so engine/cleanup3d/geom.py (NumPy, this process) and the worker (torch, float64
on the GPU) compute the same formulas; the GPU path differs only by the device's rounding (tests: < 0.05 px).
Standalone on purpose: imports only numpy (+ torch in the worker) - nothing from MatteDesk (run with `python -I`).

Maths (ported from removers_test/cards.py `card_map` and master_lib.py `project` / `rays`):
- a view = pinhole K, distortion (k1, k2, p1, p2) or none, pose R / t (X_cam = R X + t), size w x h;
- rays of the target view: Newton inversion of the OpenCV distortion (exact to float rounding, not cv2's 5 fixed-point
  steps), direction in the world w = R^T [x, y, 1];
- through a card: plane n.X + d = 0 -> camera-z depth zp = -(n.C + d) / (n.w) (valid when > 0), the point C + zp w;
  sky card = at infinity (rotation only: inv depth 0);
- into the source view: p = R_s w + (R_s C + t_s) / zp, x = p / p_z, distorted, K_s; a point behind the source camera
  or beyond the distortion's monotonic radius is "no data" (map = MISS);
- bilinear sampling (own gather - not cv2.remap's 1/32 px fixed point, so both backends agree bit-close), box blur
  with replicated edges (cumulative sums), coverage ramp at a key's border, partition of a frame between cards.

Worker protocol (engine/worker_proto.py):
    <py> geom_gpu.py --check   -> "ok cuda <gpu name>" / "ok cpu" / "ok fake" (exit 0) or the error (exit 1)
    <py> geom_gpu.py --serve   -> one JSON request per line:
        {"op": "map", "tgt": view, "src": view, "card": card, "out": .npy}  -> (2, h, w) float32 maps (MISS = no data)
        {"op": "points", "pts": .npy (n, 2), "tgt": view, "src": view, "card": card, "out": .npy} -> (n, 2) float64
        {"op": "weights", "frame": view, "cards": [card + {"seam_px", "ext": extent}], "out": .npy}
                                                                             -> (n_cards, h, w) float32 weights
        {"op": "coverage", "tgt": view, "src": view, "card": card, "edge_px": float, "out": .npy} -> (h, w) float32
        {"op": "keys_put", "keys": [{"id": str, "img": .npy (h, w, 3) float, "cut": .npy (h, w) | null}]}
                                                    -> key images kept on the GPU (LRU, KEY_SLOTS) for "project"
        {"op": "project", "frame": view, "roi": [x0, y0, x1, y1], "edge_px", "hard_k",
         "cards": [{"spec": card, "weight": .npy (rh, rw), "keys": [{"id", "view": view, "tw": float}]}],
         "blur": {"lx": .npy, "ly": .npy} | null, "out": .npy}
                                                    -> (rh, rw, 4) float32 = fill BGR + weight (render.project_frame
                                                       on the GPU, then render.blur_fill); {"missing": [ids]} when a
                                                       key is not resident (the caller sends it with keys_put, again)
        {"op": "path_blur", "img": .npy (h, w[, c]), "lx": .npy, "ly": .npy, "out": .npy}  -> engine.shutter.path_blur
      the render's key fill per frame (render.KeyFill on the GPU; the rules = `motion_grid`, `mb_rows`, `tone_rule`):
        {"op": "frame_weights", "frame": view, "cards": [as "weights"], "roi": [x0, y0, x1, y1]}
                                                    -> the frame's card weights kept resident; {"stats": [[any, mean]]}
                                                       per card in roi. "project" then takes a card's weight as
                                                       {"resident": index} and, with "keep": true, keeps its float
                                                       fill / weight; "save": {"fill", "wgt"} writes pass 1's files
                                                       (float16 / uint8 x 255); "out" optional; -> {"any": bool}
        {"op": "motion", "frame": view, "prev": view | null, "next": view | null, "roi", "step",
         "cards": [{"spec", "kind"}], "out": .npy | null}  -> render.motion_field of roi on the resident weights, kept
        {"op": "mb_measure", "frame": view, "roi", "plate": .npy, "obj": .npy u8, "keep": .npy u8 | null, "q", "k",
         "cands", "consts", "key_full"}             -> {"rows": {region: [[length, error]]}, "n": {region: px}}
        {"op": "keyfill", "frame": view, "roi", "fill": .npy float16, "lengths": [ground, sky] | null,
         "sel": .npy u8 | null, "tone": {"plate", "hole", "usable", "ring_px"} | null, "out": .npy}
                                                    -> blurred + tone-matched fill (h, w, 3) float32, {"motion_px"}
                                                       (kept resident with the plate for grain_levels)
        {"op": "grain_levels", "frame": view, "roi", "fill": .npy | null, "plate": .npy | null, "reg": .npy u8,
         "sel": .npy u8, "idx": .npy u8 | null, "sigmas", "cblock", "bins"}
                                                    -> grain.levels_cpu: {"bp", "bf", "binp", "binf"}
        {"op": "depth_warp", "tgt": view, "card": depth card, "out": .npy}
                                                    -> (3, h, w) float32: reference pixel x / y (MISS) + camera-z depth
                                                       of the hit (0 = none) - the dense correspondence of a depth card
        {"op": "release"}  frees cached GPU memory (the resident keys, the depth surfaces);  {"op": "quit"}
    view = {"K": 3x3, "dist": [k1, k2, p1, p2] | null, "R": 3x3, "t": [3], "w": int, "h": int}
    card = {"kind": "plane" | "sky", "n": [3], "d": float}
         | {"kind": "depth", "file": .npy, "stamp", "ref": view, "conv", "scale", "offset", "mul", "shift", "edge",
            "n", "d"}  (P18, see "depth card" below; every op above takes it)
    extent = null (the whole card) | {"canvas": .npy uint8 (H, W), "ox": float, "oy": float, "base": 0 | 1, "view": view}
MD_GEOM_GPU_FAKE=1: the numpy backend inside the worker (protocol tests in MatteDesk's .venv, no torch).
"""

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MISS = -1.0e5          # map value of "no data" (< 0, as the Geometry contract asks; far outside any image)
NEWTON_ITERS = 12      # distortion inversion; quadratic convergence, 12 is far past float64 precision for real lenses


# ------------------------------------------------------------------------------------------------------- backends
class _NumpyBackend:
    name = "numpy"
    f64 = np.float64

    @staticmethod
    def asarray(a, dtype=np.float64):
        return np.asarray(a, dtype)

    @staticmethod
    def arange(n):
        return np.arange(n, dtype=np.float64)

    @staticmethod
    def meshgrid(x, y):
        return np.meshgrid(x, y)

    where = staticmethod(np.where)
    abs = staticmethod(np.abs)
    sqrt = staticmethod(np.sqrt)
    stack = staticmethod(np.stack)
    isfinite = staticmethod(np.isfinite)
    floor = staticmethod(np.floor)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)

    @staticmethod
    def clip(a, lo, hi):
        return np.clip(a, lo, hi)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, np.float64)

    @staticmethod
    def ones(shape):
        return np.ones(shape, np.float64)

    @staticmethod
    def to_index(a):
        return a.astype(np.int64)

    @staticmethod
    def cumsum(a, axis):
        return np.cumsum(a, axis=axis)

    @staticmethod
    def take(a, idx, axis):
        return np.take(a, idx, axis=axis)

    @staticmethod
    def index_clip(lo, hi, n):
        return np.clip(np.arange(lo, hi), 0, n - 1)

    @staticmethod
    def to_numpy(a):
        return np.asarray(a)

    @staticmethod
    def f(a):
        return np.asarray(a, np.float64)

    @staticmethod
    def cat(seq):
        return np.concatenate(list(seq))

    @staticmethod
    def splat_min(idx, z, n):
        """z-buffer: for every cell 0..n-1 the index (into z) of the smallest z that falls into it, -1 = empty."""
        win = np.full(n, -1, np.int64)
        if len(idx):
            order = np.lexsort((z, idx))
            s = idx[order]
            first = np.r_[True, s[1:] != s[:-1]]
            win[s[first]] = order[first]
        return win


NP = _NumpyBackend()


def torch_backend(torch, device):
    """The same interface on torch tensors (float64 on `device`)."""

    class _TorchBackend:
        name = "torch"
        dev = device

        @staticmethod
        def asarray(a, dtype=None):
            if isinstance(a, torch.Tensor):
                return a.to(device=device, dtype=torch.float64)
            return torch.as_tensor(np.asarray(a, np.float64), device=device)

        @staticmethod
        def arange(n):
            return torch.arange(n, dtype=torch.float64, device=device)

        @staticmethod
        def meshgrid(x, y):
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            return xx, yy

        @staticmethod
        def where(c, a, b):
            a = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float64, device=device)
            b = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float64, device=device)
            return torch.where(c, a, b)

        abs = staticmethod(torch.abs)
        sqrt = staticmethod(torch.sqrt)
        isfinite = staticmethod(torch.isfinite)
        floor = staticmethod(torch.floor)

        @staticmethod
        def stack(seq, axis=0):
            return torch.stack(list(seq), dim=axis)

        @staticmethod
        def minimum(a, b):
            if not isinstance(b, torch.Tensor):
                return torch.clamp(a, max=b)
            if not isinstance(a, torch.Tensor):
                return torch.clamp(b, max=a)
            return torch.minimum(a, b)

        @staticmethod
        def maximum(a, b):
            if not isinstance(b, torch.Tensor):
                return torch.clamp(a, min=b)
            if not isinstance(a, torch.Tensor):
                return torch.clamp(b, min=a)
            return torch.maximum(a, b)

        @staticmethod
        def clip(a, lo, hi):
            return torch.clamp(a, lo, hi)

        @staticmethod
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.float64, device=device)

        @staticmethod
        def ones(shape):
            return torch.ones(shape, dtype=torch.float64, device=device)

        @staticmethod
        def to_index(a):
            return a.to(torch.long)

        @staticmethod
        def cumsum(a, axis):
            return torch.cumsum(a, dim=axis)

        @staticmethod
        def take(a, idx, axis):
            return torch.index_select(a, axis, idx)

        @staticmethod
        def index_clip(lo, hi, n):
            return torch.clamp(torch.arange(lo, hi, device=device), 0, n - 1)

        @staticmethod
        def to_numpy(a):
            return a.detach().cpu().numpy()

        @staticmethod
        def f(a):
            return a.to(torch.float64) if isinstance(a, torch.Tensor) else torch.as_tensor(
                np.asarray(a, np.float64), device=device)

        @staticmethod
        def cat(seq):
            return torch.cat(list(seq))

        @staticmethod
        def splat_min(idx, z, n):
            zb = torch.full((n,), float("inf"), dtype=z.dtype, device=device)
            win = torch.full((n,), -1, dtype=torch.long, device=device)
            if idx.numel():
                zb = zb.scatter_reduce(0, idx, z, "amin", include_self=True)
                m = z <= zb[idx]
                ar = torch.arange(z.numel(), device=device)
                win = win.scatter_reduce(0, idx[m], ar[m], "amin", include_self=False)
            return win

    return _TorchBackend()


# ------------------------------------------------------------------------------------------------------- lens
def _has_dist(dist):
    return dist is not None and any(abs(float(v)) > 0 for v in dist)


def distort(x, y, dist):
    """OpenCV (k1, k2, p1, p2) on normalized coordinates (works on numpy arrays and torch tensors)."""
    k1, k2, p1, p2 = (float(v) for v in dist)
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort(B, xd, yd, dist, iters=NEWTON_ITERS):
    """Inverse of distort(): Newton on the 2x2 Jacobian, start = the distorted point."""
    if not _has_dist(dist):
        return xd, yd
    k1, k2, p1, p2 = (float(v) for v in dist)
    x, y = xd * 1.0, yd * 1.0
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        g = 2 * (k1 + 2 * k2 * r2)
        fx = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd
        fy = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd
        a = rad + x * x * g + 2 * p1 * y + 6 * p2 * x
        b = x * y * g + 2 * p1 * x + 2 * p2 * y
        d = rad + y * y * g + 6 * p1 * y + 2 * p2 * x
        det = a * d - b * b
        det = B.where(B.abs(det) < 1e-12, 1e-12, det)
        sx = (d * fx - b * fy) / det
        sy = (a * fy - b * fx) / det
        x = x - sx
        y = y - sy
        if float(B.maximum(B.abs(sx), B.abs(sy)).max()) < 1e-15:     # converged (normalized units: ~1e-12 px)
            break
    return x, y


def r2_limit(view):
    """Largest r^2 (normalized, undistorted) a point may have to be projected into `view`: beyond the distortion's
    monotonic radius the polynomial folds back and far-away points would land inside the image; also capped at 2x
    the image corners' radius (such points are far outside the frame anyway)."""
    K = np.asarray(view["K"], np.float64)
    w, h = int(view["w"]), int(view["h"])
    dist = view.get("dist")
    cx = np.array([-0.5, w - 0.5, -0.5, w - 0.5])
    cy = np.array([-0.5, -0.5, h - 0.5, h - 0.5])
    xd, yd = (cx - K[0, 2]) / K[0, 0], (cy - K[1, 2]) / K[1, 1]
    if not _has_dist(dist):
        return float(4.0 * np.max(xd * xd + yd * yd))
    x, y = undistort(NP, xd, yd, dist)
    lim = 4.0 * float(np.max(x * x + y * y))
    k1, k2 = float(dist[0]), float(dist[1])
    r = np.linspace(0, 20, 200001)
    dr = 1 + 3 * k1 * r * r + 5 * k2 * r ** 4            # d/dr of r (1 + k1 r^2 + k2 r^4)
    bad = np.nonzero(dr <= 0.05)[0]
    if len(bad):
        lim = min(lim, 0.95 * float(r[bad[0]]) ** 2)
    return lim


# ------------------------------------------------------------------------------------------------------- projection
def view_rays(B, view, px=None, py=None):
    """Normalized undistorted camera coordinates (x, y) of the view's pixels (the whole grid, or points px / py)."""
    K = np.asarray(view["K"], np.float64)
    if px is None:
        px, py = B.meshgrid(B.arange(int(view["w"])), B.arange(int(view["h"])))
    xd = (px - K[0, 2]) / K[0, 0]
    yd = (py - K[1, 2]) / K[1, 1]
    return undistort(B, xd, yd, view.get("dist"))


def project(B, x, y, tgt, src, card, r2lim=None):
    """Target-view rays (x, y) -> source-view pixels through `card`. Returns (mx, my, ok)."""
    Rt = np.asarray(tgt["R"], np.float64)
    tt = np.asarray(tgt["t"], np.float64)
    Rs = np.asarray(src["R"], np.float64)
    ts = np.asarray(src["t"], np.float64)
    Ks = np.asarray(src["K"], np.float64)
    C = -Rt.T @ tt
    # world direction w = Rt^T [x, y, 1] (component by component: same code on both backends)
    wx = Rt[0, 0] * x + Rt[1, 0] * y + Rt[2, 0]
    wy = Rt[0, 1] * x + Rt[1, 1] * y + Rt[2, 1]
    wz = Rt[0, 2] * x + Rt[1, 2] * y + Rt[2, 2]
    if card["kind"] == "sky":
        inv = None
        ok = B.isfinite(x)
    elif card["kind"] == "depth":                   # the surface of a depth map (P18): camera-z depth of the hit
        zp, ok = depth_cast(B, x, y, tgt, card)
        inv = B.where(ok, 1.0 / B.maximum(zp, 1e-9), 0.0)
    else:
        n = np.asarray(card["n"], np.float64)
        num = -(float(n @ C) + float(card["d"]))
        den = wx * n[0] + wy * n[1] + wz * n[2]
        den = B.where(B.abs(den) < 1e-12, 1e-12, den)
        zp = num / den
        ok = zp > 1e-9
        inv = B.where(ok, 1.0 / B.maximum(zp, 1e-9), 0.0)
    M = Rs
    off = Rs @ C + ts
    px = M[0, 0] * wx + M[0, 1] * wy + M[0, 2] * wz
    py = M[1, 0] * wx + M[1, 1] * wy + M[1, 2] * wz
    pz = M[2, 0] * wx + M[2, 1] * wy + M[2, 2] * wz
    if inv is not None:
        px = px + off[0] * inv
        py = py + off[1] * inv
        pz = pz + off[2] * inv
    ok = ok & (pz > 1e-12)
    z = B.where(ok, pz, 1.0)
    xs, ys = px / z, py / z
    r2 = xs * xs + ys * ys
    lim = r2_limit(src) if r2lim is None else r2lim
    ok = ok & (r2 < lim)
    if _has_dist(src.get("dist")):
        xs, ys = distort(xs, ys, src["dist"])
    mx = xs * Ks[0, 0] + Ks[0, 2]
    my = ys * Ks[1, 1] + Ks[1, 2]
    mx = B.where(ok, mx, MISS)
    my = B.where(ok, my, MISS)
    return mx, my, ok


def plane_hit(B, x, y, view, card):
    """Where the view's rays hit the card in front of the camera (sky: everywhere; depth: the depth surface)."""
    if card["kind"] == "sky":
        return B.isfinite(x)
    if card["kind"] == "depth":
        return depth_cast(B, x, y, view, card)[1]
    R = np.asarray(view["R"], np.float64)
    t = np.asarray(view["t"], np.float64)
    C = -R.T @ t
    n = np.asarray(card["n"], np.float64)
    nw = R @ n                                   # n . (R^T [x, y, 1]) = (R n) . [x, y, 1]
    den = nw[0] * x + nw[1] * y + nw[2]
    num = -(float(n @ C) + float(card["d"]))
    den = B.where(B.abs(den) < 1e-12, 1e-12, den)
    return (num / den) > 1e-9


# ------------------------------------------------------------------------------------------------------- depth card (P18)
# A depth card = the surface seen by its REFERENCE view (a plate frame) through a depth map: every reference pixel
# unprojected with its metric depth is a world point X(u, v); between pixel centres the surface is the bilinear patch of
# the four points, a pixel quad whose depths jump by more than `edge` (relative) is an occlusion edge (no surface).
# A ray of another view is cast onto it as the classic depth reprojection ("forward warp" inverted):
#   1. the reference grid (every s-th pixel, s = longer side / DEPTH_SPLAT_MAX) is projected into the target view and
#      splatted into a coarse z-buffer on the target's normalized image plane (nearest cell, and a 3x3 footprint that
#      closes the gaps of a magnified surface); the nearest point of a ray's cell = the initial guess (u0, v0);
#   2. Newton on the reference pixel (u, v): project X(u, v) into the target, drive it onto the ray (2x2 Jacobian by
#      finite differences of the bilinear surface), both guesses; the hit = a converged one (residual < DEPTH_TOL_PX)
#      on a quad that is not an edge, the nearer of the two, and not behind the 3x3 footprint's z-buffer by more than
#      2 x `edge` (occlusion: a surface hidden by something whose own quads are edges - a 1-cell MISS band around the
#      silhouettes of near things, s target px);
#   3. holes (what the reference does not see, the rubber sheet across an edge) = no hit -> MISS / coverage 0.
# Limit: a key of another frame than the reference sees the surface from elsewhere - what that key's view hides is not
# tested (the key = the reference frame is the intended use).
# The result is the camera-z depth of the hit along the ray, so `project` carries on exactly as for a plane.
# card = {"kind": "depth", "file": .npy (h, w) relative depth of the reference view, "stamp", "ref": view,
#         "conv": "disparity" | "z", "scale", "offset", "mul", "shift", "edge", "n", "d" (approximate plane, previews)}
DEPTH_EDGE = 0.04        # default relative depth jump inside a pixel quad that splits the surface (occlusion edge)
DEPTH_SPLAT_MAX = 1024   # longer side of the reference grid splatted into the coarse z-buffer
DEPTH_ITERS = 8          # Newton steps (quadratic convergence from a guess within a few px)
DEPTH_TOL_PX = 0.1       # residual (target px) under which a cast is a hit
DEPTH_STEP_MAX = 8.0     # largest Newton step (reference px)
_DEPTH = {}              # reference surfaces (LRU 2)
_GUESS = {}              # coarse z-buffers per target view (LRU 4)
_CAST = {}               # last casts (LRU 2): the render asks the same rays for the weights and for every key


def _lru(cache, key, make, size):
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    v = make()
    cache[key] = v
    while len(cache) > size:
        cache.pop(next(iter(cache)))
    return v


def _view_sig(view):
    parts = [np.asarray(view[k], np.float64).tobytes() for k in ("K", "R", "t")]
    parts.append(b"-" if view.get("dist") is None else np.asarray(view["dist"], np.float64).tobytes())
    return b"|".join(parts) + f"|{int(view['w'])}x{int(view['h'])}".encode()


def depth_params(card):
    """(conv, scale, offset, mul, shift, edge) of a depth card spec (missing = identity)."""
    return (str(card.get("conv") or "disparity"), float(card.get("scale", 1.0)), float(card.get("offset", 0.0)),
            float(card.get("mul", 1.0)), float(card.get("shift", 0.0)), float(card.get("edge", DEPTH_EDGE)))


def depth_metric(d, conv="disparity", scale=1.0, offset=0.0, mul=1.0, shift=0.0):
    """Relative depth d (numpy) -> metric camera-z depth of the reference view (NaN = no surface).
    conv "disparity": 1 / z0 = scale d + offset (DepthCrafter: affine-invariant disparity, nearer = larger);
    conv "z": z0 = scale d + offset. Then the user's sliders: z = mul z0 + shift."""
    d = np.asarray(d, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if conv == "z":
            z0 = scale * d + offset
        else:
            inv = scale * d + offset
            z0 = np.where(inv > 1e-12, 1.0 / np.maximum(inv, 1e-12), np.nan)
        z = mul * z0 + shift
    return np.where(np.isfinite(z) & (z > 0), z, np.nan)


def _depth_ref(B, card):
    """(cache key, the reference surface): world points X (3 arrays (h, w)), ok (h, w) 0/1 flat, qok (h, w) 0/1 =
    the quad with this top-left pixel is surface."""
    ref = card["ref"]
    p = depth_params(card)
    key = (B.name, str(card["file"]), str(card.get("stamp") or ""), p, _view_sig(ref))

    def make():
        d = np.load(card["file"]).astype(np.float64)
        if d.ndim == 3:
            d = d[..., 0]
        w, h = int(ref["w"]), int(ref["h"])
        if d.shape != (h, w):
            raise ValueError(f"mapa głębi {d.shape[1]}x{d.shape[0]} nie pasuje do klatki odniesienia {w}x{h}")
        z = depth_metric(d, *p[:5])
        ok = np.isfinite(z)
        z0 = np.where(ok, z, 0.0)
        if B.name == "torch":                  # the 10 M px lens inversion on the GPU (numpy: ~5 s at 4K)
            xt, yt = view_rays(B, ref)
            x, y = B.to_numpy(xt), B.to_numpy(yt)
        else:
            x, y = view_rays(NP, ref)
        R = np.asarray(ref["R"], np.float64)
        C = -R.T @ np.asarray(ref["t"], np.float64)
        X = [C[i] + z0 * (R[0, i] * x + R[1, i] * y + R[2, i]) for i in range(3)]
        q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
        zz = np.stack([z0[:-1, :-1], z0[1:, :-1], z0[:-1, 1:], z0[1:, 1:]])
        q &= zz.max(0) <= zz.min(0) * (1.0 + p[5])
        qok = np.zeros((h, w))
        qok[:-1, :-1] = q
        if B.name == "numpy":
            return {"w": w, "h": h, "X": X, "okf": ok.astype(np.float64).ravel(), "qok": qok}
        # torch: the surface in float32 – the Newton search runs in float32 on the GPU (_depth_newton); X3 = the
        # three coordinates stacked for one grid_sample per evaluation
        Xt = [B.asarray(a).float() for a in X]
        return {"w": w, "h": h, "X": Xt, "X3": torch.stack(Xt)[None].contiguous(),
                "okf": B.asarray(ok.astype(np.float64).ravel()), "qok": B.asarray(qok)}
    return key, _lru(_DEPTH, key, make, 2)


def _to_cam(B, P, view):
    R = np.asarray(view["R"], np.float64)
    t = np.asarray(view["t"], np.float64)
    c = [R[i, 0] * P[0] + R[i, 1] * P[1] + R[i, 2] * P[2] + t[i] for i in range(3)]
    zs = B.where(B.abs(c[2]) > 1e-12, c[2], 1e-12)
    return c[0] / zs, c[1] / zs, c[2]


def _winners(B, idx, z, src, n):
    """Per cell of a z-buffer of n cells: (the reference pixel (flat index `src`) nearest to the camera, -1 = empty;
    its camera-z depth, 1e300 = empty)."""
    win = B.splat_min(idx, z, n)
    if src.shape[0] == 0:
        return win, B.zeros((n,)) + 1e300
    has = win >= 0
    return (src[win * has] + 1) * has - 1, B.where(has, z[win * has], 1e300)


def _depth_guess(B, rd, tgt, rkey):
    """The coarse z-buffer of the reference surface in the target view: grid on the target's normalized image plane
    (cell = s target px), winners of the nearest-cell splat and of the 3x3 footprint."""
    def make():
        w, h = rd["w"], rd["h"]
        s = max(1, int(math.ceil(max(w, h) / float(DEPTH_SPLAT_MAX))))
        K = np.asarray(tgt["K"], np.float64)
        tw, th = int(tgt["w"]), int(tgt["h"])
        bx = np.r_[np.arange(tw), np.arange(tw), np.zeros(th), np.full(th, tw - 1.0)].astype(np.float64)
        by = np.r_[np.zeros(tw), np.full(tw, th - 1.0), np.arange(th), np.arange(th)].astype(np.float64)
        rx, ry = view_rays(NP, tgt, bx, by)                  # the picture's border, normalized and undistorted
        cx, cy = s / K[0, 0], s / K[1, 1]
        x0, y0 = float(rx.min()) - 3 * cx, float(ry.min()) - 3 * cy
        gw = int(min(8192, math.ceil((rx.max() - rx.min()) / cx) + 7))
        gh = int(min(8192, math.ceil((ry.max() - ry.min()) / cy) + 7))
        vs, us = np.arange(0, h, s), np.arange(0, w, s)
        fi = B.to_index(B.asarray((vs[:, None] * w + us[None, :]).ravel().astype(np.float64)))
        P = [a.reshape(-1)[fi] for a in rd["X"]]
        xn, yn, zt = _to_cam(B, P, tgt)
        good = (rd["okf"][fi] > 0.5) & (zt > 1e-9)
        gx = (xn - x0) / cx
        gy = (yn - y0) / cy
        good = good & (gx > -1.5) & (gx < gw + 0.5) & (gy > -1.5) & (gy < gh + 0.5)
        ix = B.to_index(B.floor(B.clip(gx, -1.0, gw) + 0.5))
        iy = B.to_index(B.floor(B.clip(gy, -1.0, gh) + 0.5))
        ins = good & (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
        near, _zn = _winners(B, (iy * gw + ix)[ins], zt[ins], fi[ins], gw * gh)
        cells, zs, srcs = [], [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jx, jy = ix + dx, iy + dy
                m = good & (jx >= 0) & (jx < gw) & (jy >= 0) & (jy < gh)
                cells.append((jy * gw + jx)[m])
                zs.append(zt[m])
                srcs.append(fi[m])
        wide, zwide = _winners(B, B.cat(cells), B.cat(zs), B.cat(srcs), gw * gh)
        return {"x0": x0, "y0": y0, "cx": cx, "cy": cy, "gw": gw, "gh": gh, "near": near, "zwide": zwide,
                "wide": wide}
    return _lru(_GUESS, (rkey, _view_sig(tgt)), make, 4)


def _depth_newton(B, rd, tgt, x, y, u, v):
    """Newton from (u, v) (reference px) to the point of the surface on the target rays (x, y). Returns (camera-z
    depth in the target, hit)."""
    w1, h1 = rd["w"] - 1.0, rd["h"] - 1.0
    K = np.asarray(tgt["K"], np.float64)
    f = 0.5 * (K[0, 0] + K[1, 1])
    if B.name == "torch":                      # float32 on the GPU (~2x faster; ~3e-4 px at 4K, tolerance 0.1 px)
        x, y, u, v = x.float(), y.float(), u.float(), v.float()

    if B.name == "torch" and "X3" in rd:
        F = torch.nn.functional
        sx, sy = 2.0 / max(w1, 1.0), 2.0 / max(h1, 1.0)

        def cam(uu, vv):                       # bilinear of the 3 coordinates at once (border = clamp, as bilinear)
            g = torch.stack([uu * sx - 1.0, vv * sy - 1.0], -1).reshape(1, 1, -1, 2).float()   # backend where -> f64
            P = F.grid_sample(rd["X3"], g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]
            return _to_cam(B, [P[0].reshape(uu.shape), P[1].reshape(uu.shape), P[2].reshape(uu.shape)], tgt)
    else:
        def cam(uu, vv):
            return _to_cam(B, [bilinear(B, a, uu, vv) for a in rd["X"]], tgt)
    for it in range(DEPTH_ITERS):
        px, py, _zc = cam(u, v)
        rx, ry = px - x, py - y
        if float(B.maximum(B.abs(rx), B.abs(ry)).max()) * f < 1e-4:
            break
        hu = B.where(u + 0.5 <= w1, 0.5, -0.5)
        hv = B.where(v + 0.5 <= h1, 0.5, -0.5)
        ax, ay, _ = cam(u + hu, v)
        bx, by, _ = cam(u, v + hv)
        a, c = (ax - px) / hu, (ay - py) / hu
        b, d = (bx - px) / hv, (by - py) / hv
        det = a * d - b * c
        det = B.where(B.abs(det) < 1e-18, 1e-18, det)
        du = (d * rx - b * ry) / det
        dv = (a * ry - c * rx) / det
        n = B.sqrt(du * du + dv * dv)
        k = B.where(n > DEPTH_STEP_MAX, DEPTH_STEP_MAX / B.maximum(n, 1e-12), 1.0)
        u = B.clip(u - du * k, 0.0, w1)
        v = B.clip(v - dv * k, 0.0, h1)
    px, py, zc = cam(u, v)
    res = B.sqrt((px - x) * (px - x) + (py - y) * (py - y)) * f
    iu = B.to_index(B.minimum(B.floor(u), max(w1 - 1.0, 0.0)))
    iv = B.to_index(B.minimum(B.floor(v), max(h1 - 1.0, 0.0)))
    hit = (res < DEPTH_TOL_PX) & (rd["qok"][iv, iu] > 0.5) & (zc > 1e-9)
    return (zc.double() if B.name == "torch" else zc), hit


def depth_cast(B, x, y, tgt, card):
    """Target-view rays (x, y) (normalized, undistorted, any shape) onto the depth surface -> (camera-z depth zp, hit)."""
    rkey, rd = _depth_ref(B, card)
    shape = tuple(x.shape)
    xf, yf = x.reshape(-1), y.reshape(-1)
    n = int(xf.shape[0])
    step = max(1, n // 61)
    ck = (rkey, _view_sig(tgt), shape, B.to_numpy(xf[::step]).tobytes(), B.to_numpy(yf[::step]).tobytes())
    if ck in _CAST:
        _CAST[ck] = _CAST.pop(ck)
        return _CAST[ck]
    best = B.zeros((n,)) + 1e300
    hit = B.zeros((n,)) > 1.0
    if n:
        g = _depth_guess(B, rd, tgt, rkey)
        gx = (xf - g["x0"]) / g["cx"]
        gy = (yf - g["y0"]) / g["cy"]
        ing = (gx > -0.5) & (gx < g["gw"] - 0.5) & (gy > -0.5) & (gy < g["gh"] - 0.5)
        cell = (B.to_index(B.floor(B.clip(gy, 0.0, g["gh"] - 1.0) + 0.5)) * g["gw"]
                + B.to_index(B.floor(B.clip(gx, 0.0, g["gw"] - 1.0) + 0.5)))
        w_near, w_wide = g["near"][cell], g["wide"][cell]
        zfront = g["zwide"][cell] * (1.0 + 2.0 * depth_params(card)[5])
        for wi, m in ((w_near, ing & (w_near >= 0)),
                      (w_wide, ing & (w_wide >= 0) & (w_wide != w_near))):
            if not bool(m.any()):
                continue
            ws = wi[m]
            zc, ok = _depth_newton(B, rd, tgt, xf[m], yf[m], B.f(ws % rd["w"]), B.f(ws // rd["w"]))
            ok = ok & (zc <= zfront[m])           # behind what the z-buffer shows in this cell = occluded (MISS)
            cur = best[m]
            take = ok & (zc < cur)
            best[m] = B.where(take, zc, cur)
            hit[m] = hit[m] | ok
    zp = B.where(hit, best, 0.0)
    out = (zp.reshape(shape), hit.reshape(shape))
    _CAST[ck] = out
    while len(_CAST) > 2:
        _CAST.pop(next(iter(_CAST)))
    return out


def depth_warp(B, tgt, card):
    """The dense correspondence of a target view with the depth card's reference view: per target pixel the reference
    pixel its ray hits (MISS where none) and the hit's camera-z depth in the target (0 = none)."""
    x, y = view_rays(B, tgt)
    mx, my, ok = project(B, x, y, tgt, card["ref"], card)
    zp, hit = depth_cast(B, x, y, tgt, card)
    return mx, my, B.where(hit & ok, zp, 0.0)


def depth_release():
    _DEPTH.clear()
    _GUESS.clear()
    _CAST.clear()


# ------------------------------------------------------------------------------------------------------- images
def bilinear(B, img, mx, my, border=0.0):
    """img (H, W) sampled at (mx, my) - pixel centres at integers; outside [0, W-1] x [0, H-1] = `border`."""
    H, W = img.shape[-2], img.shape[-1]
    inside = (mx >= 0) & (my >= 0) & (mx <= W - 1) & (my <= H - 1)
    x = B.clip(mx, 0, W - 1)
    y = B.clip(my, 0, H - 1)
    x0f = B.minimum(B.floor(x), max(W - 2, 0))
    y0f = B.minimum(B.floor(y), max(H - 2, 0))
    fx, fy = x - x0f, y - y0f
    x0, y0 = B.to_index(x0f), B.to_index(y0f)
    x1 = B.to_index(B.minimum(x0f + 1, W - 1))
    y1 = B.to_index(B.minimum(y0f + 1, H - 1))
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
         + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return B.where(inside, v, float(border))


def box_blur(B, a, k):
    """Mean over a k x k window (k odd), edges replicated - separable, by cumulative sums."""
    k = int(k)
    if k <= 1:
        return a
    r = k // 2
    for axis in (0, 1):
        n = a.shape[axis]
        p = B.take(a, B.index_clip(-r, n + r, n), axis)
        c = B.cumsum(p, axis)
        z = B.zeros((1, a.shape[1]) if axis == 0 else (a.shape[0], 1))
        c = _cat0(B, z, c, axis)                         # c[j] = sum of p[:j], length n + k
        hi = B.take(c, B.index_clip(k, n + k, n + k), axis)
        lo = B.take(c, B.index_clip(0, n, n + k), axis)
        a = (hi - lo) / k
    return a


def _cat0(B, z, c, axis):
    if B.name == "numpy":
        return np.concatenate([z, c], axis=axis)
    import torch
    return torch.cat([z, c], dim=axis)


def seam_kernel(seam_px):
    """Box size of a soft border of `seam_px` (odd, >= 1): the weight ramps linearly across it."""
    k = int(round(float(seam_px)))
    return max(1, k | 1)


def coverage(B, mx, my, w, h, edge_px):
    """0..1 how far a source image of w x h reaches at its pixels (mx, my): linear 0 -> 1 over `edge_px` from its border
    (= removers_test/vkey3d.py KCOV sampled at the map, analytically); 0 where the map has no data."""
    d = B.minimum(B.minimum(mx, (w - 1) - mx), B.minimum(my, (h - 1) - my))
    cov = B.clip(d / max(float(edge_px), 1e-6), 0.0, 1.0)
    return B.where(mx > MISS / 2, cov, 0.0)


def extent_mask(B, x, y, frame_view, card, ext):
    """0..1 extent of `card` in the frame (rays x, y): the extent canvas (drawn in the extent view) sampled through the
    card. ext None = the whole card."""
    if ext is None:
        return None
    mx, my, ok = project(B, x, y, frame_view, ext["view"], card, ext.get("r2lim"))
    canvas = ext["canvas_arr"]
    v = bilinear(B, canvas, mx + float(ext["ox"]), my + float(ext["oy"]), float(ext["base"]))
    return B.where(ok, v, 0.0)


def weights(B, frame_view, cards, x=None, y=None):
    """Partition of the frame between the cards (in order, 0 = front). cards = [{"kind", "n", "d", "seam_px", "ext"}]
    (ext with "canvas_arr" loaded). Returns a list of (h, w) weights: sum <= 1, = 1 wherever some card is hit."""
    if x is None:
        x, y = view_rays(B, frame_view)
    shape = tuple(x.shape)
    taken = B.zeros(shape)
    owns, hits = [], []
    for c in cards:
        hit = plane_hit(B, x, y, frame_view, c)
        inside = B.where(hit, 1.0, 0.0)
        e = extent_mask(B, x, y, frame_view, c, c.get("ext"))
        if e is not None:
            inside = B.where(e >= 0.5, inside, 0.0)
        own = inside * (1.0 - taken)
        taken = taken + own
        owns.append(own)
        hits.append(hit)
    soft = [B.where(hit, box_blur(B, own, seam_kernel(c.get("seam_px", 16.0))), 0.0)
            for c, own, hit in zip(cards, owns, hits)]
    total = B.zeros(shape)
    for s in soft:
        total = total + s
    safe = B.where(total > 1e-12, total, 1.0)
    return [B.where(total > 1e-12, s / safe, 0.0) * taken for s in soft]


# ------------------------------------------------------------------------------------------------------- render
KEY_SLOTS = 10         # key images resident on the GPU (a 4K key in float16 = 50 MB)
_KEYS = {}             # id -> {"img": (3, h, w) tensor / (h, w, 3) array, "cut": (h, w) or None}  (insertion = LRU)


def _keys_put(B, keys):
    for k in keys:
        img = np.load(k["img"]).astype(np.float32)
        cut = np.load(k["cut"]).astype(np.float32) if k.get("cut") else None
        if B.name == "torch":
            t = torch.as_tensor(img, device=DEV).permute(2, 0, 1).contiguous().to(torch.float16)
            c = torch.as_tensor(cut, device=DEV) if cut is not None else None      # float32: tiny cuts matter
            _KEYS[k["id"]] = {"img": t, "cut": c, "h": img.shape[0], "w": img.shape[1]}
        else:
            _KEYS[k["id"]] = {"img": img, "cut": cut, "h": img.shape[0], "w": img.shape[1]}
        _KEYS[k["id"]] = _KEYS.pop(k["id"])
        while len(_KEYS) > KEY_SLOTS:
            _KEYS.pop(next(iter(_KEYS)))


def _sample(B, entry, name, mx, my, cubic, border):
    """The key's image (3 channels) or cut (1) sampled at (mx, my) - torch: grid_sample (bicubic like cv2.INTER_CUBIC,
    edges replicated / cut: 0 outside), numpy: the own bilinear gather (protocol tests)."""
    a = entry[name]
    h, w = entry["h"], entry["w"]
    if B.name == "torch":
        F = torch.nn.functional
        gx = (mx.to(torch.float32) / max(w - 1, 1)) * 2 - 1
        gy = (my.to(torch.float32) / max(h - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        src = (a if a.dim() == 3 else a[None]).to(torch.float32)[None]
        out = F.grid_sample(src, grid, mode="bicubic" if cubic else "bilinear",
                            padding_mode="border" if border is None else "zeros", align_corners=True)[0]
        return out if a.dim() == 3 else out[0]
    if a.ndim == 3:
        return np.stack([_np_bicubic(a[..., c], mx, my) for c in range(3)]).astype(np.float32)
    return _np_border_bilinear(B, a, mx, my, border).astype(np.float32)


def _np_bicubic(img, mx, my, A=-0.75):
    """cv2.INTER_CUBIC (a = -0.75, the kernel of torch grid_sample "bicubic" too), edges replicated - the numpy twin
    of the GPU sampling for the protocol tests."""
    h, w = img.shape
    x = np.clip(np.asarray(mx, np.float64), 0, w - 1)
    y = np.clip(np.asarray(my, np.float64), 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0

    def k(t):
        t = np.abs(t)
        return np.where(t <= 1, ((A + 2) * t - (A + 3)) * t * t + 1,
                        np.where(t < 2, ((A * t - 5 * A) * t + 8 * A) * t - 4 * A, 0.0))
    out = np.zeros(x.shape, np.float64)
    for j in range(-1, 3):
        wy = k(fy - j)
        yy = np.clip(y0 + j, 0, h - 1)
        for i in range(-1, 3):
            out += img[yy, np.clip(x0 + i, 0, w - 1)] * wy * k(fx - i)
    return out


def _np_border_bilinear(B, img, mx, my, border):
    if border is None:                               # replicate the edge
        h, w = img.shape
        return bilinear(B, img, np.clip(mx, 0, w - 1), np.clip(my, 0, h - 1))
    return bilinear(B, img, mx, my, border=border)


def _f32(B, a):
    return a.to(torch.float32) if B.name == "torch" else np.asarray(a, np.float32)


def _project(B, req):
    """render.project_frame of the region `roi` (the same crossfade, backup keys, coverage, cut), in float32."""
    x0, y0, x1, y1 = [int(v) for v in req["roi"]]
    rh, rw = y1 - y0, x1 - x0
    px, py = B.meshgrid(B.arange(rw) + x0, B.arange(rh) + y0)
    view = req["frame"]
    x, y = view_rays(B, view, px, py)
    edge, hard_k = float(req["edge_px"]), float(req["hard_k"])
    BACKUP = 1e-6
    if B.name == "torch":
        z2 = lambda: torch.zeros((rh, rw), dtype=torch.float32, device=DEV)
        z3 = lambda: torch.zeros((3, rh, rw), dtype=torch.float32, device=DEV)
        mn = torch.minimum
        mxm = torch.maximum
        wh = torch.where
        to = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=DEV)
    else:
        z2 = lambda: np.zeros((rh, rw), np.float32)
        z3 = lambda: np.zeros((3, rh, rw), np.float32)
        mn, mxm, wh = np.minimum, np.maximum, np.where
        to = lambda a: np.asarray(a, np.float32)
    acc, wsum = z3(), z2()
    missing = [k["id"] for c in req["cards"] for k in c["keys"] if k["id"] not in _KEYS]
    if missing:
        return None, sorted(set(missing))
    for c in req["cards"]:
        if isinstance(c["weight"], dict):              # the resident weights of op frame_weights, cropped
            cw = _resident_weights(view)[int(c["weight"]["resident"])][y0:y1, x0:x1]
        else:
            cw = to(np.load(c["weight"]))
        num, den, reach = z3(), z2(), z2()
        col = None
        for k in sorted(c["keys"], key=lambda q: -float(q["tw"])):
            tw = float(k["tw"])
            backup = tw <= 0.0
            if backup:
                if col is None:
                    col = num / mxm(den, BACKUP if B.name != "torch" else torch.tensor(BACKUP, device=DEV))[None]
                    reach = wh(den > BACKUP, reach, 0.0 * reach)
                if not bool(((reach < 0.999) & (cw > 1e-4)).any()):
                    continue
            e = _KEYS[k["id"]]
            _KEYS[k["id"]] = _KEYS.pop(k["id"])              # LRU touch
            mx, my, _ok = project(B, x, y, view, k["view"], c["spec"])
            cov = _f32(B, coverage(B, mx, my, e["w"], e["h"], edge))
            hard = mn(cov * hard_k, 1.0 + 0.0 * cov)
            if e["cut"] is not None:
                ct = _sample(B, e, "cut", mx, my, False, 0.0)
                cov, hard = cov * ct, hard * ct
            pix = _sample(B, e, "img", mx, my, True, None)
            if backup:
                wb = (1.0 - reach) * hard
                col = (col * reach[None] + pix * wb[None]) / mxm(reach + wb, 0.0 * wb + BACKUP)[None]
                reach = reach + wb
                continue
            rel = cov * tw
            num = num + pix * rel[None]
            reach = wh(rel > 0, mxm(reach, hard), reach)
            den = den + rel
        if col is None:
            col = num / mxm(den, 0.0 * den + BACKUP)[None]
            reach = wh(den > BACKUP, reach, 0.0 * reach)
        a = cw * reach
        acc = acc + col * a[None]
        wsum = wsum + a
    fill = acc / mxm(wsum, 0.0 * wsum + 1e-6)[None]
    return (fill, wsum.clip(0.0, 1.0) if B.name == "torch" else np.clip(wsum, 0.0, 1.0)), None


def _path_blur(B, img, lx, ly):
    """engine.shutter.path_blur: box average along (lx, ly) per pixel, centred, bilinear, edges reflected.
    img (c, h, w); lx, ly (h, w)."""
    if B.name == "torch":
        lxt, lyt = torch.as_tensor(lx, device=DEV), torch.as_tensor(ly, device=DEV)
        ln = float(torch.hypot(lxt, lyt).max()) if lxt.numel() else 0.0
    else:
        ln = float(np.hypot(lx, ly).max()) if lx.size else 0.0
    if ln < 0.5:
        return img
    n = int(min(48, max(2, math.ceil(ln) + 1)))
    c, h, w = img.shape
    if B.name == "torch":
        F = torch.nn.functional
        gy, gx = torch.meshgrid(torch.arange(h, device=DEV, dtype=torch.float32),
                                torch.arange(w, device=DEV, dtype=torch.float32), indexing="ij")
        acc = torch.zeros_like(img)
        src = img[None]
        for i in range(n):
            t = (i + 0.5) / n - 0.5
            sx = (gx + t * lxt) / max(w - 1, 1) * 2 - 1
            sy = (gy + t * lyt) / max(h - 1, 1) * 2 - 1
            acc += F.grid_sample(src, torch.stack([sx, sy], -1)[None], mode="bilinear", padding_mode="reflection",
                                 align_corners=True)[0]
        return acc / n
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    acc = np.zeros_like(img)
    for i in range(n):
        t = (i + 0.5) / n - 0.5
        sx = np.clip(gx + t * lx, 0, w - 1)
        sy = np.clip(gy + t * ly, 0, h - 1)
        for ch in range(c):
            acc[ch] += bilinear(B, img[ch].astype(np.float64), sx, sy)
    return acc / n


def _to_np(B, a):
    return a.detach().cpu().numpy() if B.name == "torch" else np.asarray(a)


# ------------------------------------------------------------------------------------------------------- per-frame fill
# The per-frame steps of the render's key fill (render.KeyFill: motion field, motion blur measure, blur, tone) written
# ONCE as rules over a small set of image primitives `O`: CvOps (numpy + OpenCV - the CPU path in MatteDesk and the fake
# worker; exactly the calls render.py made before, so the CPU results are unchanged) and TorchOps (the GPU worker: the
# same primitives on CUDA tensors - separable area / linear / cubic resampling as cv2 defines them, Gaussian kernels of
# cv2.getGaussianKernel with the same border rules, max-pool dilation, the path blur by grid_sample). Image layout:
# CvOps (h, w[, 3]), TorchOps (h, w) / (3, h, w); the rules only use the primitives, arithmetic and comparisons.
def motion_grid(B, view, pview, nview, cards, wsample, roi, step):
    """(dx, dy, sky) on the sample grid of the region `roi` (render.motion_field): the camera's own motion per frame at
    every step-th pixel (half the path from t-1 to t+1 through the cards) and the sky cards' weight. cards = [(spec,
    kind)], wsample(i, iy, ix) -> card i's weight at the grid's pixels (B array (len(iy), len(ix))); pview / nview =
    None when the frame has no neighbour on that side (one-sided, or no motion at all)."""
    x0, y0, x1, y1 = roi
    rh, rw = y1 - y0, x1 - x0
    nx, ny = max(1, -(-rw // step)), max(1, -(-rh // step))
    fx, fy = x0 + (np.arange(nx) + 0.5) * rw / nx - 0.5, y0 + (np.arange(ny) + 0.5) * rh / ny - 0.5
    ix, iy = np.clip(np.rint(fx).astype(int), x0, x1 - 1), np.clip(np.rint(fy).astype(int), y0, y1 - 1)
    PX, PY = B.meshgrid(B.asarray(fx), B.asarray(fy))
    x, y = view_rays(B, view, PX, PY)
    dx, dy, sky = B.zeros(PX.shape), B.zeros(PX.shape), B.zeros(PX.shape)
    ends = [v for v in (pview, nview) if v is not None]
    span = float(len(ends)) if len(ends) == 2 else 1.0
    for i, (spec, kind) in enumerate(cards):
        w = B.f(wsample(i, iy, ix))
        if kind == "sky":
            sky = sky + w
        if not ends or not bool((w > 0).any()):
            continue
        pos = []
        bad = B.zeros(PX.shape) > 1.0
        for v in (pview, nview):
            if v is None:
                pos.append((PX, PY))
                continue
            mx, my, ok = project(B, x, y, view, v, spec)
            pos.append((mx, my))
            bad = bad | ~ok
        dx = dx + w * B.where(bad, 0.0, (pos[1][0] - pos[0][0]) / span)
        dy = dy + w * B.where(bad, 0.0, (pos[1][1] - pos[0][1]) / span)
    return dx, dy, sky


def mb_rows(O, fill, plate, mask, usable, dx, dy, sky, q, k, cands, c):
    """render.MotionBlur.measure of one frame region: ({region: {length: error}}, {region: ring pixels}) - the error =
    |log hf energy of the blurred fill - log hf energy of the plate| on a ring around the mask (c = the constants
    ring_out, ring_in, min_motion, min_pixels of render.py), per region "ground" (no sky) / "sky"."""
    h, w = mask.shape[-2], mask.shape[-1]
    sw, sh = max(1, int(round(w * q))), max(1, int(round(h * q)))
    rs = (lambda a, i: O.resize(a, (sw, sh), i)) if q < 1.0 else (lambda a, i: a)
    gf = rs(O.gray(fill), "area")
    gp = rs(O.gray(plate), "area")
    m = rs(O.u8(mask), "nearest")
    us = rs(O.u8(usable), "nearest") > 0
    dxq, dyq, skyq = rs(dx, "linear") * q, rs(dy, "linear") * q, rs(sky, "linear")
    ro, ri = max(2, int(round(c["ring_out"] * k))), max(1, int(round(c["ring_in"] * k)))
    ring = (O.dilate(m, ro) > 0) & ~(O.dilate(m, ri) > 0) & us
    ring = ring & (O.hypot(dxq, dyq) > c["min_motion"] * k)
    sel = {"ground": ring & (skyq < 0.01), "sky": ring & (skyq > 0.99)}
    need = max(30, int(c["min_pixels"] * k * k))
    n = {r: O.count(v) for r, v in sel.items()}
    sel = {r: v for r, v in sel.items() if n[r] >= need}
    rows = {}
    if sel:
        target = O.hf(gp)
        tg = {r: np.log(max(1e-6, O.mean_at(target, v))) for r, v in sel.items()}
        for L in cands:
            e = O.hf(O.path_blur(gf, dxq, dyq, L) if L > 0 else gf)
            for r, v in sel.items():
                rows.setdefault(r, {})[L] = abs(np.log(max(1e-6, O.mean_at(e, v))) - tg[r])
    return rows, {r: n[r] for r in sel}


def diffuse(O, val, w, sigma, ds=8):
    """`val` known where w > 0 spread everywhere: multi-scale normalised convolution at 1 / ds resolution (render.py /
    void_comp.py `diffuse`; the weight is applied BEFORE the reduction, so pixels outside the ring never leak in)."""
    H, W = w.shape[-2], w.shape[-1]
    sm = (max(1, W // ds), max(1, H // ds))
    v = O.resize(O.mulw(val, w), sm, "area")
    ws = O.resize(w, sm, "area")
    s0 = max(1.0, sigma / ds)
    num, den = None, None
    for sc in (s0, s0 * 2, s0 * 4, s0 * 8, s0 * 16, s0 * 32, s0 * 64):
        gv, gw = O.gauss(v, sc, "constant"), O.gauss(ws, sc, "constant")
        num = gv if num is None else num + gv
        den = gw if den is None else den + gw
    return O.resize(O.divw(num, O.maximum(den, 1e-6)), (W, H), "cubic")


def tone_rule(O, f, p, obj, usable, ring_px):
    """render.tone_match in model space: f + diffuse(p - f) from the ring of `ring_px` outside the object `obj` (bool)
    where `usable`; None when the ring is empty (the fill stays)."""
    r = int(round(ring_px))
    ring = (O.dilate(O.u8(obj), r) > 0) & ~obj & usable
    if not O.count(ring):
        return None
    return f + diffuse(O, p - f, O.f32(ring), ring_px / 2.0)


class CvOps:
    """The primitives on numpy arrays with OpenCV (imported here only: the GPU worker never builds this)."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.interp = {"area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                       "cubic": cv2.INTER_CUBIC}

    def gray(self, img):
        return self.cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), self.cv2.COLOR_BGR2GRAY) * 255.0

    def resize(self, a, size, mode):
        return self.cv2.resize(a, size, interpolation=self.interp[mode])

    @staticmethod
    def u8(m):
        return np.asarray(m).astype(np.uint8)

    @staticmethod
    def f32(m):
        return np.asarray(m).astype(np.float32)

    def dilate(self, m, r):
        return self.cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))

    hypot = staticmethod(np.hypot)
    maximum = staticmethod(np.maximum)

    @staticmethod
    def count(v):
        return int(v.sum())

    @staticmethod
    def mean_at(e, v):
        return float(e[v].mean())

    def gauss(self, a, sigma, border="reflect101"):
        b = self.cv2.BORDER_CONSTANT if border == "constant" else self.cv2.BORDER_DEFAULT
        return self.cv2.GaussianBlur(a, (0, 0), sigma, borderType=b)

    def hf(self, g):
        return np.abs(g - self.cv2.GaussianBlur(g, (0, 0), 2.0))

    def path_blur(self, img, dx, dy, length):
        """engine.shutter.path_blur of `img` along length x (dx, dy) (render.blur_fill on the CPU)."""
        from engine.shutter import path_blur
        lx, ly = (dx * length).astype(np.float32), (dy * length).astype(np.float32)
        if img.ndim == 2:
            return path_blur(img, lx, ly)
        return np.dstack([path_blur(np.ascontiguousarray(img[..., c]), lx, ly) for c in range(img.shape[2])])

    @staticmethod
    def mulw(val, w):
        return val * w[..., None]

    @staticmethod
    def divw(num, den):
        return num / den[..., None]


def _area_matrix(n_src, n_dst):
    """(n_dst, n_src) weights of cv2.INTER_AREA along one axis (downscale: overlap of the destination pixel's span with
    each source pixel / the span; upscale: cv2 then interpolates linearly - not used by the rules)."""
    s = n_src / float(n_dst)
    A = np.zeros((n_dst, n_src), np.float64)
    for d in range(n_dst):
        a, b = d * s, min(n_src, (d + 1) * s)
        i0, i1 = int(math.floor(a)), int(math.ceil(b))
        for i in range(i0, min(i1, n_src)):
            A[d, i] = max(0.0, min(b, i + 1) - max(a, i))
        A[d] /= A[d].sum()
    return A


def _gauss_kernel(sigma):
    """cv2.getGaussianKernel(ksize, sigma) with cv2's ksize for float images (round(sigma * 8 + 1) | 1)."""
    k = int(round(sigma * 8 + 1)) | 1
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    return g / g.sum()


class TorchOps:
    """The same primitives on torch tensors (float32 on the worker's device); images (h, w) or (3, h, w) BGR."""

    def __init__(self, torch_mod, dev):
        self.t, self.dev = torch_mod, dev
        self.F = torch_mod.nn.functional
        self._mats = {}

    def gray(self, img):
        c = self.t.clamp(img, 0, 1)
        return (c[0] * 0.114 + c[1] * 0.587 + c[2] * 0.299) * 255.0

    def _mat(self, n_src, n_dst):
        key = (n_src, n_dst)
        if key not in self._mats:
            self._mats[key] = self.t.as_tensor(_area_matrix(n_src, n_dst), dtype=self.t.float32, device=self.dev)
        return self._mats[key]

    def resize(self, a, size, mode):
        sw, sh = int(size[0]), int(size[1])
        h, w = a.shape[-2], a.shape[-1]
        if (sw, sh) == (w, h):
            return a
        if mode == "area" and sw <= w and sh <= h:
            return self._mat(h, sh) @ a.to(self.t.float32) @ self._mat(w, sw).T
        if mode == "nearest":
            iy = self.t.clamp((self.t.arange(sh, device=self.dev, dtype=self.t.float64) * (h / float(sh))).floor().long(), 0, h - 1)
            ix = self.t.clamp((self.t.arange(sw, device=self.dev, dtype=self.t.float64) * (w / float(sw))).floor().long(), 0, w - 1)
            return a[..., iy, :][..., ix]
        x = a.to(self.t.float32)
        x4 = x[None, None] if x.dim() == 2 else x[None]
        m = {"linear": "bilinear", "cubic": "bicubic", "area": "bilinear"}[mode]
        out = self.F.interpolate(x4, size=(sh, sw), mode=m, align_corners=False)
        return out[0, 0] if x.dim() == 2 else out[0]

    def u8(self, m):
        return m.to(self.t.float32) if m.dtype == self.t.bool else m

    def f32(self, m):
        return m.to(self.t.float32)

    def dilate(self, m, r):
        x = m.to(self.t.float32)[None, None]
        return self.F.max_pool2d(x, 2 * r + 1, stride=1, padding=r)[0, 0]

    def hypot(self, a, b):
        return self.t.hypot(a, b)

    def maximum(self, a, v):
        return self.t.clamp(a, min=v)

    def count(self, v):
        return int(v.sum())

    def mean_at(self, e, v):
        return float(e[v].double().mean()) if bool(v.any()) else float("nan")

    def gauss(self, a, sigma, border="reflect101"):
        g = self.t.as_tensor(_gauss_kernel(sigma), dtype=self.t.float32, device=self.dev)
        k = g.numel()
        r = k // 2
        x = a.to(self.t.float32)
        x4 = x[None, None] if x.dim() == 2 else x[:, None]          # channels as the batch
        for dim in (3, 2):
            n = x4.shape[dim]
            if border == "constant":
                pad = (r, r, 0, 0) if dim == 3 else (0, 0, r, r)
                x4 = self.F.pad(x4, pad)
            elif border == "reflect":                                # cv2.BORDER_REFLECT: the edge pixel repeated
                idx = self.t.arange(-r, n + r, device=self.dev) % (2 * n)
                idx = self.t.where(idx >= n, 2 * n - 1 - idx, idx)
                x4 = x4.index_select(dim, idx)
            else:                                                    # reflect101 = index mirror without the edge
                idx = self.t.arange(-r, n + r, device=self.dev)
                idx = idx.abs()
                if n > 1:
                    per = 2 * (n - 1)
                    idx = idx % per
                    idx = self.t.where(idx >= n, per - idx, idx)
                else:
                    idx = idx * 0
                x4 = x4.index_select(dim, idx)
            kk = g.view(1, 1, 1, k) if dim == 3 else g.view(1, 1, k, 1)
            x4 = self.F.conv2d(x4, kk)
        return x4[0, 0] if x.dim() == 2 else x4[:, 0]

    def hf(self, g):
        return (g - self.gauss(g, 2.0)).abs()

    def path_blur(self, img, dx, dy, length):
        lx, ly = (dx * length).to(self.t.float32), (dy * length).to(self.t.float32)
        two = img.dim() == 2
        out = _path_blur_t(img[None] if two else img, lx, ly)
        return out[0] if two else out

    def mulw(self, val, w):
        return val * w[None]

    def divw(self, num, den):
        return num / den[None]


def grain_levels_t(O, p, f, reg, sel, idx, sigmas, cblock, bins):
    """engine/cleanup3d/grain.levels_cpu on the GPU (TorchOps; p / f (3, h, w) in model units, reg / sel bool (h, w),
    idx = brightness bin per measured pixel or None): the DoG bands as normalised convolutions inside reg (Gaussians with
    cv2.BORDER_REFLECT), minus the median of their cblock x cblock block (exact medians by sorting), robust levels
    1.4826 x median |v|. Returns (bp, bf, binp, binf) as numpy arrays (lists per bin, None for an empty bin)."""
    t = O.t
    m = reg.to(t.float32)
    h, w = sel.shape
    ph, pw = -h % cblock, -w % cblock
    sp = O.F.pad(sel[None, None].to(t.float32), (0, pw, 0, ph))[0, 0] > 0.5
    by, bx = sp.shape[0] // cblock, sp.shape[1] // cblock
    sb = sp.reshape(by, cblock, bx, cblock).permute(0, 2, 1, 3).reshape(by * bx, cblock * cblock)
    used = sb.any(1)
    su = sb[used]                                                    # (u, cblock^2)
    k = su.sum(1)                                                    # values per used block
    ys, xs = t.nonzero(sel, as_tuple=True)
    blk = (ys // cblock) * bx + xs // cblock                         # block of every measured pixel (sel order)
    slot = t.full((by * bx,), -1, dtype=t.long, device=O.dev)
    slot[used] = t.arange(int(used.sum()), device=O.dev)
    pix_slot = slot[blk]

    def median_rows(v, cnt):
        """Median of the first cnt values of each sorted row (np.median: the mean of the two middle ones in float32)."""
        lo = v.gather(1, ((cnt - 1) // 2).clamp(min=0)[:, None])[:, 0]
        hi = v.gather(1, (cnt // 2).clamp(min=0)[:, None])[:, 0]
        return t.where(cnt % 2 == 1, lo, (lo + hi) / 2)

    def centred(b):                                                  # b (3, h, w) -> (n, 3)
        bpad = O.F.pad(b[None], (0, pw, 0, ph))[0]
        bb = bpad.reshape(3, by, cblock, bx, cblock).permute(1, 3, 0, 2, 4).reshape(by * bx, 3, cblock * cblock)[used]
        vals = t.where(su[:, None, :], bb, t.full_like(bb, float("inf")))
        srt = vals.sort(dim=2).values                                # (u, 3, cblock^2), the selected values first
        med = t.stack([median_rows(srt[:, c], k) for c in range(3)], 1)          # (u, 3)
        return b[:, sel].T - med[pix_slot]

    def level(v):                                                    # (n, 3) -> (3,)
        a = v.abs().sort(dim=0).values
        n = a.shape[0]
        lo, hi = a[(n - 1) // 2], a[n // 2]
        return 1.4826 * (lo if n % 2 == 1 else (lo + hi) / 2)

    def bands_centred(img):
        im = img * m[None]
        g = [img] + [O.gauss(im, s_, "reflect") / O.maximum(O.gauss(m, s_, "reflect"), 1e-4)[None] for s_ in sigmas[1:]]
        return [centred(g[i] - g[i + 1]) for i in range(len(sigmas) - 1)]
    cp, cf = bands_centred(p), bands_centred(f)
    bp = t.stack([level(v) for v in cp]).cpu().numpy()
    bf = t.stack([level(v) for v in cf]).cpu().numpy()
    if idx is None:
        return bp, bf, None, None
    it = t.as_tensor(idx, device=O.dev)
    binp, binf = [], []
    for kk in range(bins):
        s_ = it == kk
        if not bool(s_.any()):
            binp.append(None)
            binf.append(None)
            continue
        binp.append(level(cp[0][s_]).cpu().numpy())
        binf.append(level(cf[0][s_]).cpu().numpy())
    return bp, bf, binp, binf


def _path_blur_t(img, lx, ly):
    """engine.shutter.path_blur on the GPU: img (c, h, w), lx / ly (h, w) tensors."""
    F = torch.nn.functional
    ln = float(torch.hypot(lx, ly).max()) if lx.numel() else 0.0
    if ln < 0.5:
        return img.to(torch.float32)
    n = int(min(48, max(2, math.ceil(ln) + 1)))
    c, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32),
                            torch.arange(w, device=img.device, dtype=torch.float32), indexing="ij")
    acc = torch.zeros((c, h, w), dtype=torch.float32, device=img.device)
    src = img.to(torch.float32)[None]
    for i in range(n):
        t = (i + 0.5) / n - 0.5
        sx = (gx + t * lx) / max(w - 1, 1) * 2 - 1
        sy = (gy + t * ly) / max(h - 1, 1) * 2 - 1
        acc += F.grid_sample(src, torch.stack([sx, sy], -1)[None], mode="bilinear", padding_mode="reflection",
                             align_corners=True)[0]
    return acc / n


# ------------------------------------------------------------------------------------------------------- worker
torch = None
DEV = None
FAKE = os.environ.get("MD_GEOM_GPU_FAKE") == "1"


def _backend():
    global torch, DEV
    if FAKE:
        return NP
    if torch is None:
        import torch as _t
        torch = _t
        DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = False      # float32 means float32 (area resampling, Gaussians)
        torch.backends.cudnn.allow_tf32 = False
    return torch_backend(torch, DEV)


def _load_ext(B, ext):
    if not ext:
        return None
    e = dict(ext)
    arr = np.load(e["canvas"]).astype(np.float64) / 255.0
    e["canvas_arr"] = B.asarray(arr) if B.name != "numpy" else arr
    return e


# ------------------------------------------------------------------------------------------------------- fill ops (worker)
_RES = {}              # resident per-frame state: "weights", "motion", "proj" (one frame each)


def _ops():
    """The image primitives of this worker: TorchOps on the device, CvOps in the fake worker (numpy + OpenCV)."""
    if FAKE:
        if "ops" not in _RES:
            root = str(Path(__file__).resolve().parents[2])        # engine.shutter for CvOps.path_blur (tests only)
            if root not in sys.path:
                sys.path.insert(0, root)
            _RES["ops"] = CvOps()
        return _RES["ops"]
    if "ops" not in _RES:
        _RES["ops"] = TorchOps(torch, DEV)
    return _RES["ops"]


def _img_in(a):
    """(h, w, 3) / (h, w) numpy -> the ops' layout ((3, h, w) / (h, w) float32 tensor; numpy as it is)."""
    a = np.ascontiguousarray(a)
    if FAKE:
        return a
    t = torch.as_tensor(a, device=DEV)
    return t.permute(2, 0, 1).contiguous() if t.dim() == 3 else t


def _img_out(a):
    if FAKE:
        return np.asarray(a)
    a = a.detach()
    return (a.permute(1, 2, 0) if a.dim() == 3 else a).contiguous().cpu().numpy()


def _mask_in(path):
    m = np.load(path) > 0
    return m if FAKE else torch.as_tensor(m, device=DEV)


def _weights_key(req):
    import json
    return _view_sig(req["frame"]) + json.dumps(req["cards"], sort_keys=True).encode()


def _frame_weights(B, req):
    """The frame's card weights computed once and kept resident (op "project" / "motion" crop them); per card in `roi`:
    [any weight > 1e-4, mean weight]."""
    key = _weights_key(req)
    got = _RES.get("weights")
    if got is None or got["key"] != key:
        cards = [dict(c, ext=_load_ext(B, c.get("ext"))) for c in req["cards"]]
        ws = weights(B, req["frame"], cards)
        st = B.stack([_f32(B, w) for w in ws], 0)
        got = _RES["weights"] = {"key": key, "frame": _view_sig(req["frame"]), "w": st}
    x0, y0, x1, y1 = [int(v) for v in req["roi"]]
    out = []
    for i in range(int(got["w"].shape[0])):
        cw = got["w"][i, y0:y1, x0:x1]
        n = cw.numel() if B.name == "torch" else cw.size
        out.append([bool((cw > 1e-4).any()), float(cw.mean()) if n else 0.0])
    return out


def _resident_weights(view):
    got = _RES.get("weights")
    if got is None or got["frame"] != _view_sig(view):
        raise ValueError("no resident card weights of this frame (op frame_weights first)")
    return got["w"]


def _motion(B, req):
    """render.motion_field of the region on the resident weights (motion_grid + the scaling up), kept resident."""
    W = _resident_weights(req["frame"])
    x0, y0, x1, y1 = [int(v) for v in req["roi"]]
    rh, rw = y1 - y0, x1 - x0
    cards = [(c["spec"], c["kind"]) for c in req["cards"]]

    def wsample(i, iy, ix):
        if B.name == "torch":
            return W[i][torch.as_tensor(iy, device=DEV)[:, None], torch.as_tensor(ix, device=DEV)[None, :]]
        return W[i][iy[:, None], ix[None, :]]
    dx, dy, sky = motion_grid(B, req["frame"], req.get("prev"), req.get("next"), cards, wsample,
                              (x0, y0, x1, y1), int(req["step"]))
    O = _ops()

    def up(a):
        a = _f32(B, a)
        return O.resize(a, (rw, rh), "linear") if tuple(a.shape) != (rh, rw) else a
    d = (up(dx), up(dy), B.clip(up(sky), 0.0, 1.0))
    _RES["motion"] = {"roi": (x0, y0, x1, y1), "frame": _view_sig(req["frame"]), "d": d}
    if req.get("out"):
        np.save(req["out"], np.stack([_img_out(a) for a in d]).astype(np.float32))
    return {}


def _resident_motion(req):
    got = _RES.get("motion")
    roi = tuple(int(v) for v in req["roi"])
    if got is None or got["roi"] != roi or got["frame"] != _view_sig(req["frame"]):
        raise ValueError("no resident motion field of this frame region (op motion first)")
    return got["d"]


def _resident_proj(req):
    got = _RES.get("proj")
    roi = tuple(int(v) for v in req["roi"])
    if got is None or got["roi"] != roi or got["frame"] != _view_sig(req["frame"]):
        raise ValueError("no resident projection of this frame region (op project with keep first)")
    return got


def _mb_measure(B, req):
    """render.MotionBlur.measure of the resident projection (fill, float weight) against the plate region."""
    O = _ops()
    pr = _resident_proj(req)
    dx, dy, sky = _resident_motion(req)
    fill = pr["fill"] if not FAKE else np.ascontiguousarray(np.asarray(pr["fill"]).transpose(1, 2, 0))
    plate = _img_in(np.load(req["plate"]).astype(np.float32))
    obj = _mask_in(req["obj"])
    us = pr["wgt"] >= float(req["key_full"])
    if req.get("keep"):
        us = us & ~(O.dilate(O.u8(_mask_in(req["keep"])), 4) > 0)
    us = us & ~obj
    rows, n = mb_rows(O, fill, plate, obj, us, dx, dy, sky, float(req["q"]), float(req["k"]),
                      [float(v) for v in req["cands"]], req["consts"])
    return {"rows": {r: [[L, float(e)] for L, e in v.items()] for r, v in rows.items()}, "n": n}


def _keyfill(B, req):
    """The fill of one frame region after the projection (render.KeyFill pass 2): the saved projection (float16, the
    file pass 1 wrote) blurred along the resident motion field with the per-pixel length (sky / ground), tone-matched
    to the plate from the ring around the hole; (h, w, 3) float32 -> out, {"motion_px"}."""
    O = _ops()
    fill = _img_in(np.load(req["fill"]).astype(np.float32))
    res = {}
    ln = req.get("lengths")
    if ln:
        dx, dy, sky = _resident_motion(req)
        lg, ls = float(ln[0]), float(ln[1])
        fill = O.path_blur(fill, dx, dy, sky * ls + (1.0 - sky) * lg)
        if req.get("sel"):
            sel = _mask_in(req["sel"])
            res["motion_px"] = round(O.mean_at(O.hypot(dx, dy), sel), 2) if O.count(sel) else 0.0
    tn = req.get("tone")
    if tn:
        plate = _img_in(np.load(tn["plate"]).astype(np.float32))
        clip = (lambda a: np.clip(a, 0, 1).astype(np.float32)) if FAKE else (lambda a: torch.clamp(a, 0, 1))
        out = tone_rule(O, clip(fill), clip(plate), _mask_in(tn["hole"]), _mask_in(tn["usable"]), float(tn["ring_px"]))
        if out is not None:
            fill = out
        _RES["kf"] = {"roi": tuple(int(v) for v in req["roi"]), "frame": _view_sig(req["frame"]), "fill": fill,
                      "plate": plate}
    else:
        _RES["kf"] = {"roi": tuple(int(v) for v in req["roi"]), "frame": _view_sig(req["frame"]), "fill": fill,
                      "plate": None}
    np.save(req["out"], np.asarray(_img_out(fill), np.float32))
    return res


def _grain_levels(B, req):
    """engine/cleanup3d/grain.levels_cpu of the fill op keyfill returned (resident) or of "fill", against the plate
    (resident from its tone step, or "plate"), on "reg" / "sel" (u8) with "idx" (brightness bins, u8) or null."""
    got = _RES.get("kf")
    roi = tuple(int(v) for v in req["roi"])
    ok = got is not None and got["roi"] == roi and got["frame"] == _view_sig(req["frame"])
    fill = _img_in(np.load(req["fill"]).astype(np.float32)) if req.get("fill") else (got["fill"] if ok else None)
    plate = _img_in(np.load(req["plate"]).astype(np.float32)) if req.get("plate") else (got["plate"] if ok else None)
    if fill is None or plate is None:
        raise ValueError("grain_levels: no fill / plate of this frame region")
    idx = np.load(req["idx"]) if req.get("idx") else None
    if FAKE:
        _ops()
        from engine.cleanup3d.grain import levels_cpu
        clip = lambda a: np.clip(a, 0, 1).astype(np.float32)
        bp, bf, binp, binf = levels_cpu(clip(plate), clip(fill), np.load(req["reg"]) > 0,
                                        np.load(req["sel"]) > 0, idx)
    else:
        bp, bf, binp, binf = grain_levels_t(_ops(), torch.clamp(plate, 0, 1), torch.clamp(fill, 0, 1),
                                            _mask_in(req["reg"]), _mask_in(req["sel"]), idx, req["sigmas"],
                                            int(req["cblock"]), int(req["bins"]))
    lst = lambda v: None if v is None else [None if a is None else np.asarray(a, np.float64).tolist() for a in v]
    return {"bp": np.asarray(bp, np.float64).tolist(), "bf": np.asarray(bf, np.float64).tolist(),
            "binp": lst(binp), "binf": lst(binf)}


def _handle(req, ask=None):
    B = _backend()
    op = req.get("op")
    if op == "release":
        _KEYS.clear()
        _RES.pop("weights", None)
        _RES.pop("motion", None)
        _RES.pop("proj", None)
        _RES.pop("kf", None)
        depth_release()
        if torch is not None and DEV is not None and DEV.type == "cuda":
            torch.cuda.empty_cache()
        return {}
    if op == "keys_put":
        _keys_put(B, req["keys"])
        return {"resident": len(_KEYS)}
    if op == "project":
        got, missing = _project(B, req)
        if missing:
            return {"missing": missing}
        fill, wgt = got
        res = {"any": bool((wgt > 0).any())}
        if req.get("keep"):                            # kept for op mb_measure (float fill and weight)
            _RES["proj"] = {"roi": tuple(int(v) for v in req["roi"]), "frame": _view_sig(req["frame"]),
                            "fill": _f32(B, fill), "wgt": _f32(B, wgt)}
        sv = req.get("save")
        if sv:                                         # pass 1's files, as render.KeyFill.prepare writes them
            np.save(sv["fill"], _to_np(B, fill).transpose(1, 2, 0).astype(np.float16))
            np.save(sv["wgt"], np.round(_to_np(B, wgt) * 255).astype(np.uint8))
        bl = req.get("blur")
        if bl:
            fill = _path_blur(B, fill, np.load(bl["lx"]).astype(np.float32), np.load(bl["ly"]).astype(np.float32))
        if req.get("out"):
            out = np.concatenate([_to_np(B, fill).transpose(1, 2, 0), _to_np(B, wgt)[..., None]], axis=2)
            np.save(req["out"], out.astype(np.float32))
        return res
    if op == "frame_weights":
        return {"stats": _frame_weights(B, req)}
    if op == "motion":
        return _motion(B, req)
    if op == "mb_measure":
        return _mb_measure(B, req)
    if op == "keyfill":
        return _keyfill(B, req)
    if op == "grain_levels":
        return _grain_levels(B, req)
    if op == "path_blur":
        img = np.load(req["img"]).astype(np.float32)
        two = img.ndim == 2
        chw = img[None] if two else img.transpose(2, 0, 1)
        t = torch.as_tensor(np.ascontiguousarray(chw), device=DEV) if B.name == "torch" else chw
        res = _to_np(B, _path_blur(B, t, np.load(req["lx"]).astype(np.float32), np.load(req["ly"]).astype(np.float32)))
        np.save(req["out"], (res[0] if two else res.transpose(1, 2, 0)).astype(np.float32))
        return {}
    if op == "map":
        x, y = view_rays(B, req["tgt"])
        mx, my, _ok = project(B, x, y, req["tgt"], req["src"], req["card"])
        np.save(req["out"], np.stack([B.to_numpy(mx), B.to_numpy(my)]).astype(np.float32))
        return {}
    if op == "points":
        p = np.load(req["pts"]).astype(np.float64)
        x, y = view_rays(B, req["tgt"], B.asarray(p[:, 0]), B.asarray(p[:, 1]))
        mx, my, ok = project(B, x, y, req["tgt"], req["src"], req["card"])
        out = np.stack([B.to_numpy(mx), B.to_numpy(my)], 1)
        out[~B.to_numpy(ok)] = np.nan
        np.save(req["out"], out)
        return {}
    if op == "coverage":
        x, y = view_rays(B, req["tgt"])
        mx, my, _ok = project(B, x, y, req["tgt"], req["src"], req["card"])
        cov = coverage(B, mx, my, int(req["src"]["w"]), int(req["src"]["h"]), float(req["edge_px"]))
        np.save(req["out"], B.to_numpy(cov).astype(np.float32))
        return {}
    if op == "depth_warp":
        mx, my, z = depth_warp(B, req["tgt"], req["card"])
        np.save(req["out"], np.stack([B.to_numpy(mx), B.to_numpy(my), B.to_numpy(z)]).astype(np.float32))
        return {}
    if op == "weights":
        cards = [dict(c, ext=_load_ext(B, c.get("ext"))) for c in req["cards"]]
        ws = weights(B, req["frame"], cards)
        np.save(req["out"], np.stack([B.to_numpy(w) for w in ws]).astype(np.float32))
        return {}
    raise ValueError(f"unknown op {op}")


def _serve():
    spec = importlib.util.spec_from_file_location("worker_proto", Path(__file__).resolve().parents[1] / "worker_proto.py")
    wp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wp)

    def make():
        B = _backend()
        device = "fake" if FAKE else str(DEV)

        def handle(req, ask):
            t0 = time.time()
            out = _handle(req, ask)
            resp = {"device": device, "seconds": round(time.time() - t0, 4), **out}
            if B.name == "torch" and DEV.type == "cuda":
                resp["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            return resp
        return handle
    return wp.serve(make)


def _check():
    if FAKE:
        print("ok fake")
        return 0
    try:
        B = _backend()
        float(B.ones((4, 4)).sum())
        name = torch.cuda.get_device_name(0) if DEV.type == "cuda" else ""
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1
    print("ok", DEV.type, name)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check()
    if len(argv) >= 2 and argv[1] == "--serve":
        return _serve()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
