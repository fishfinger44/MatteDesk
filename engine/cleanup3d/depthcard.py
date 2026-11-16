# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – the depth card (docs/PLAN_CLEANUP_3D.md, package P18, user decision 2026-11-10): a card whose surface is
the DEPTH MAP of one reference plate frame instead of a plane, for background planes cannot describe (parallax of near
things). Ground / sea / sky stay planes.

- Sources: the depth the tool computed (DepthCrafter, stage 0 „Głębia” -> outputs.json `depth`, relative disparity:
  nearer = larger) or the shot's own depth plate (`plates.depth` of the manifest, convention chosen by the user).
- Snapshot: the reference frame's depth, resized to the plate, is stored once as `<work>/inpaint/cards/<card>/
  depth_####.npy` (float32); the card points at it (`depth.file`, relative to `<work>/inpaint/`), so the projection
  never reads the source sequence again and a recomputed depth does not change a finished card silently.
- Fit to the solved camera: relative depth d -> metric camera-z depth of the reference view, `conv` disparity:
  1 / z = scale d + offset, `conv` z: z = scale d + offset (affine, robust: trimmed least squares). Anchors: the solve's
  sparse 3D points seen in the reference frame; else the plane cards (where a plane card owns the reference frame, its
  depth); else no fit (scale 1, offset 0) and the user's sliders. The sliders on top: z = mul z0 + shift
  („Skala głębi”, „Przesunięcie”).
- Approximate plane (least squares through the surface's points): the card's `plane`, used by the previews' grid and
  the Nuke export (a depth card goes to Nuke as that plane – an approximation, noted in the export).
The geometry itself (casting rays onto the surface) is engine/cleanup3d/geom_gpu.py `depth_cast` (CPU and GPU).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d import types as T

FIT_MIN = 12                 # anchors needed for a fit
FIT_GRID = 8                 # px between the samples of a plane card's depth


# ----------------------------------------------------------------------------------------------- sources
def sources(work_dir: Path, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The depth sources of the shot: [{"id", "label", "pattern", "available", "conv"}] (computed first)."""
    pat_c = ""
    try:
        from engine.outputs import load_outputs
        pat_c = str((load_outputs(work_dir).get("depth") or {}).get("path") or "")
    except Exception:  # noqa: BLE001 – a broken outputs.json only hides this source
        pat_c = ""
    pat_p = str(((manifest or {}).get("plates") or {}).get("depth") or "")
    return [{"id": "computed", "label": "Głębia policzona w narzędziu (DepthCrafter)", "pattern": pat_c,
             "available": bool(pat_c), "conv": "disparity"},
            {"id": "plate", "label": "Plate głębi ujęcia", "pattern": pat_p, "available": bool(pat_p),
             "conv": "disparity"}]


def pick_source(srcs: Sequence[Mapping[str, Any]], want: Optional[str]) -> Mapping[str, Any]:
    """The source to use: `want` (computed | plate) or the first available; ValueError (Polish) when there is none."""
    for s in srcs:
        if (want in (None, "", "auto") or s["id"] == want) and s["available"]:
            return s
    if want not in (None, "", "auto"):
        raise ValueError("Wybrane źródło głębi nie istnieje w tym ujęciu – policz głębię (etap 0 „Głębia”) "
                         "albo dodaj plate głębi")
    raise ValueError("Brak mapy głębi ujęcia – policz ją w etapie 0 („Głębia”, DepthCrafter) albo dodaj plate głębi")


def read_depth_file(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """One depth frame -> (h, w) float32 at the plate size `size` = (w, h). EXR: engine/depth.read_depth (depth.disparity,
    R, ...); other formats: the first channel, scaled to 0..1."""
    import cv2
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Brak pliku głębi {path.name}")
    if path.suffix.lower() == ".exr":
        from engine.depth import read_depth
        d = np.asarray(read_depth(path), np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)          # unicode-safe (engine/cv2_unicode.py)
        if img is None:
            raise ValueError(f"Nie da się odczytać pliku głębi {path.name}")
        d = img[..., 0] if img.ndim == 3 else img
        scale = 65535.0 if d.dtype == np.uint16 else 255.0 if d.dtype == np.uint8 else 1.0
        d = d.astype(np.float32) / scale
    w, h = int(size[0]), int(size[1])
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(d, np.float32)


def snapshot(work_dir: Path, card: str, frame: int, depth: np.ndarray) -> Tuple[str, str]:
    """Store the reference depth of a card -> (file relative to <work>/inpaint/, stamp = content hash)."""
    from engine.cleanup3d import store
    folder = store.extent_dir(work_dir, card)
    folder.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(depth, np.float32)
    p = folder / f"depth_{int(frame):04d}.npy"
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(p)
    return p.relative_to(T.root_dir(work_dir)).as_posix(), hashlib.sha1(arr.tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- fit
def _frame_view(camera: T.Camera3D, frame: int) -> Dict[str, Any]:
    if frame not in camera.R:
        raise ValueError(f"Kamera 3D nie ma klatki odniesienia F{frame}")
    return {"K": np.asarray(camera.K, np.float64), "dist": np.asarray(camera.dist, np.float64), "R": camera.R[frame],
            "t": camera.t[frame], "w": int(camera.width), "h": int(camera.height)}


def _robust_affine(d: np.ndarray, t: np.ndarray, iters: int = 6) -> Optional[Tuple[float, float, np.ndarray]]:
    """t ~ a d + b: least median of squares over sample pairs (an object off the anchors' surface may cover half the
    picture), then trimmed least squares (residuals over 2.5 MAD dropped each round). (a, b, inliers) or None."""
    fin = np.isfinite(d) & np.isfinite(t)
    if fin.sum() < FIT_MIN:
        return None
    dd, tt = d[fin], t[fin]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(dd), (2, 400))
    ok = np.abs(dd[i] - dd[j]) > 1e-12 * (np.abs(dd[i]) + 1e-12)
    a = b = 0.0
    if ok.any():
        A = (tt[i[ok]] - tt[j[ok]]) / (dd[i[ok]] - dd[j[ok]])
        Bc = tt[i[ok]] - A * dd[i[ok]]
        sub = rng.choice(len(dd), min(len(dd), 2000), replace=False)
        med = np.median(np.abs(A[:, None] * dd[sub][None, :] + Bc[:, None] - tt[sub][None, :]), axis=1)
        k = int(np.argmin(med))
        a, b = float(A[k]), float(Bc[k])
        r = np.abs(a * d + b - t)
        keep = fin & (r <= 2.5 * 1.4826 * float(med[k]) + 1e-12)
        if keep.sum() < FIT_MIN:
            keep = fin
    else:
        keep = fin
    for _ in range(iters):
        A = np.c_[d[keep], np.ones(int(keep.sum()))]
        sol, *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(a * d + b - t)
        mad = float(np.median(r[keep])) * 1.4826 + 1e-12
        new = np.isfinite(d) & np.isfinite(t) & (r <= 2.5 * mad + 1e-9 * np.abs(t))
        if new.sum() < FIT_MIN or np.array_equal(new, keep):
            break
        keep = new
    return a, b, keep


def _sample(d: np.ndarray, px: np.ndarray) -> np.ndarray:
    return K_.bilinear(K_.NP, d.astype(np.float64), px[:, 0], px[:, 1], border=np.nan)


def anchors_from_points(camera: T.Camera3D, frame: int, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse 3D points seen in the reference frame -> (pixels (n, 2), camera-z depth (n,))."""
    v = _frame_view(camera, frame)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    Xc = P @ np.asarray(v["R"]).T + np.asarray(v["t"])
    z = Xc[:, 2]
    front = z > 1e-6
    if not front.any():
        return np.zeros((0, 2)), np.zeros(0)
    x, y = Xc[front, 0] / z[front], Xc[front, 1] / z[front]
    if K_._has_dist(v["dist"]):
        x, y = K_.distort(x, y, v["dist"])
    K = v["K"]
    px = np.c_[x * K[0, 0] + K[0, 2], y * K[1, 1] + K[1, 2]]
    inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] <= v["w"] - 1) & (px[:, 1] <= v["h"] - 1)
    return px[inside], z[front][inside]


def anchors_from_planes(camera: T.Camera3D, frame: int, cards: Sequence[T.Card],
                        layers: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Where a plane card owns the reference frame (its weight > 0.9 among the plane cards) -> (pixels, its depth)."""
    from engine.cleanup3d.geom import CardGeometry
    planes = [c for c in cards if c.kind == "plane"]
    if not planes:
        return np.zeros((0, 2)), np.zeros(0)
    geo = CardGeometry(camera, planes, layers)
    try:
        ws = geo.card_weights(frame)
        v = geo.frame_view(frame)
        ys, xs = np.mgrid[FIT_GRID // 2:v["h"]:FIT_GRID, FIT_GRID // 2:v["w"]:FIT_GRID]
        px = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
        x, y = K_.view_rays(K_.NP, v, px[:, 0], px[:, 1])
        R = np.asarray(v["R"])
        C = -R.T @ np.asarray(v["t"])
        pts, zs = [], []
        for c in planes:
            w = ws[c.id][ys.ravel(), xs.ravel()]
            spec = geo.card_spec(c)
            n, d0 = np.asarray(spec["n"]), float(spec["d"])
            nw = R @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                z = -(n @ C + d0) / (nw[0] * x + nw[1] * y + nw[2])
            m = (w > 0.9) & np.isfinite(z) & (z > 0)
            pts.append(px[m])
            zs.append(z[m])
        return np.concatenate(pts), np.concatenate(zs)
    finally:
        geo.close()


def fit_depth(camera: T.Camera3D, frame: int, depth: np.ndarray, conv: str = "disparity",
              points: Optional[np.ndarray] = None, cards: Sequence[T.Card] = (),
              layers: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None) -> Dict[str, Any]:
    """Relative depth of the reference frame -> {"scale", "offset", "method": points | planes | none, "anchors",
    "inliers", "err_rel" (median relative depth error of the inliers), "z_median" (metric, whole map)}."""
    if conv not in T.DEPTH_CONVS:
        raise ValueError(f"depth.conv: {conv!r}")
    d = np.asarray(depth, np.float64)
    tries = []
    if points is not None and len(points):
        tries.append(("points", lambda: anchors_from_points(camera, frame, points)))
    tries.append(("planes", lambda: anchors_from_planes(camera, frame, cards, layers)))
    for method, get in tries:
        px, z = get()
        if len(z) < FIT_MIN:
            continue
        dv = _sample(d, px)
        t = 1.0 / z if conv == "disparity" else z
        got = _robust_affine(dv, t)
        if got is None:
            continue
        a, b, keep = got
        if conv == "disparity" and a <= 0:          # nearer must stay nearer
            continue
        zf = K_.depth_metric(dv[keep], conv, a, b)
        err = float(np.nanmedian(np.abs(zf - z[keep]) / z[keep])) if keep.any() else float("nan")
        zm = K_.depth_metric(d, conv, a, b)
        return {"scale": a, "offset": b, "method": method, "anchors": int(len(z)), "inliers": int(keep.sum()),
                "err_rel": round(err, 4), "z_median": float(np.nanmedian(zm)) if np.isfinite(zm).any() else 1.0}
    zm = K_.depth_metric(d, conv, 1.0, 0.0)
    return {"scale": 1.0, "offset": 0.0, "method": "none", "anchors": 0, "inliers": 0, "err_rel": None,
            "z_median": float(np.nanmedian(zm)) if np.isfinite(zm).any() else 1.0}


def approx_plane(camera: T.Camera3D, frame: int, depth: np.ndarray, dp: Mapping[str, Any],
                 step: int = 8) -> Tuple[np.ndarray, float]:
    """Least-squares plane (n, d) through the depth surface's points (every `step` px); n toward the camera."""
    v = _frame_view(camera, frame)
    p = K_.depth_params({**dict(dp), "file": "-", "ref": v})
    d = np.asarray(depth, np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    zs = K_.depth_metric(d[step // 2::step, step // 2::step], *p[:5])
    ys, xs = np.mgrid[step // 2:d.shape[0]:step, step // 2:d.shape[1]:step]
    m = np.isfinite(zs)
    if m.sum() < 3:
        raise ValueError("Mapa głębi bez ważnych punktów – sprawdź „Skala głębi” / „Przesunięcie”")
    x, y = K_.view_rays(K_.NP, v, xs[m].astype(np.float64), ys[m].astype(np.float64))
    R = np.asarray(v["R"])
    C = -R.T @ np.asarray(v["t"])
    P = C + zs[m][:, None] * (np.c_[x, y, np.ones(len(x))] @ R)
    c = P.mean(0)
    _u, _s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    dd = -float(n @ c)
    if n @ C + dd < 0:
        n, dd = -n, -dd
    return n, dd


def make_depth(camera: T.Camera3D, frame: int, depth: np.ndarray, base: Mapping[str, Any],
               points: Optional[np.ndarray] = None, cards: Sequence[T.Card] = (),
               layers: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None, fit: bool = True
               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The `depth` dict + the approximate `plane` of a depth card from its reference depth: fitted to the camera
    (fit=True; the user's sliders reset) or with the scale / offset of `base` kept."""
    dp = T.parse_depth({**dict(base), "frame": int(frame)})
    if fit:
        rep = fit_depth(camera, frame, depth, dp["conv"], points, cards, layers)
        dp.update(scale=rep["scale"], offset=rep["offset"], mul=1.0, shift=0.0, fit=rep)
    n, d = approx_plane(camera, frame, depth, dp)
    return dp, {"n": [float(v) for v in n], "d": float(d)}
