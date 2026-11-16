# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P10 – the numeric part of the camera solve (no COLMAP, no GPU; numpy + scipy + OpenCV).

Ported from the proven experiment (removers_test: joint_ba.py, dyn_mask.py, horizon.py, horizon3d.py, sfm_geo.py):
  * the lens model (OpenCV: k1, k2, p1, p2) – project / undistort;
  * CoTracker3 tracks -> 3D points (DLT on undistorted rays), moving / drifting tracks dropped by their reprojection;
  * the joint bundle adjustment of SIFT points + tracks: Levenberg-Marquardt with the Schur complement over the
    points and the robust soft-L1 loss (`solver="scipy"` = the experiment's `least_squares` trf + lsmr, kept for
    comparison: its inexact steps need several times more iterations), frame 0 fixed (gauge) + a weak prior on the camera centres (scale); `single_focal` = one focal
    fx = fy with k1 / k2 free (cx, cy, p1, p2 stay); the horizon as a constraint: every horizon point is a direction
    on one level plane, residual = its pixel distance from the frame's horizon line / sig_hz (pins the barrel
    distortion the camera motion alone cannot separate from the focal);
  * the epipolar test for pixels that move on their own (dense flow vs the solved cameras, no depth needed);
  * the horizon of a frame (straight strong edge, sub-pixel points) and the world up from it;
  * RANSAC plane fit and the sea-level plane for the cards step.
Pixels everywhere: PLATE pixels, pixel centre at integer coordinates (the DATA_CONTRACT convention).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# intrinsics vector used through the solve: (fx, fy, cx, cy, k1, k2, p1, p2) in plate pixels
FX, FY, CX, CY, K1, K2, P1, P2 = range(8)


# ----------------------------------------------------------------------------- lens model
def distort(x: np.ndarray, y: np.ndarray, k1: float, k2: float, p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def project(R: np.ndarray, t: np.ndarray, X: np.ndarray, intr: Sequence[float]) -> np.ndarray:
    """World points (n, 3) -> plate pixels (n, 2). R (3, 3) / t (3,) of one camera, or (n, 3, 3) / (n, 3) per point."""
    R, t, X = np.asarray(R, np.float64), np.asarray(t, np.float64), np.asarray(X, np.float64)
    Xc = (X @ R.T + t) if R.ndim == 2 else (np.einsum("nij,nj->ni", R, X) + t)
    z = np.maximum(Xc[:, 2], 1e-6)
    xd, yd = distort(Xc[:, 0] / z, Xc[:, 1] / z, intr[K1], intr[K2], intr[P1], intr[P2])
    return np.stack([xd * intr[FX] + intr[CX], yd * intr[FY] + intr[CY]], 1)


def undistort(xy: np.ndarray, intr: Sequence[float], iters: int = 8) -> np.ndarray:
    """Plate pixels (n, 2) -> normalised undistorted coordinates (n, 2) (fixed-point, as cv2.undistortPoints)."""
    xy = np.asarray(xy, np.float64)
    xd, yd = (xy[:, 0] - intr[CX]) / intr[FX], (xy[:, 1] - intr[CY]) / intr[FY]
    k1, k2, p1, p2 = intr[K1], intr[K2], intr[P1], intr[P2]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad, (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    return np.stack([x, y], 1)


def k_matrix(intr: Sequence[float]) -> np.ndarray:
    return np.array([[intr[FX], 0, intr[CX]], [0, intr[FY], intr[CY]], [0, 0, 1.0]])


def reproj_px(Rs: np.ndarray, ts: np.ndarray, intr: Sequence[float], X: np.ndarray, obs_cam: np.ndarray,
              obs_pt: np.ndarray, obs_xy: np.ndarray) -> np.ndarray:
    """Reprojection error (euclidean, px) of every observation."""
    if len(obs_cam) == 0:
        return np.zeros(0)
    return np.linalg.norm(project(Rs[obs_cam], ts[obs_cam], X[obs_pt], intr) - obs_xy, axis=1)


def centres(Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return -np.einsum("nji,nj->ni", np.asarray(Rs), np.asarray(ts))


# ----------------------------------------------------------------------------- tracks -> points
def triangulate_tracks(tracks: np.ndarray, visible: np.ndarray, cam_idx: Sequence[int], Rs: np.ndarray, ts: np.ndarray,
                       intr: Sequence[float], *, min_vis: int = 6, drop_px: float = 6.0, every: int = 2,
                       ) -> Dict[str, Any]:
    """CoTracker3 tracks (T, M, 2 plate px; `visible` T x M; row i = camera `cam_idx[i]`) triangulated with the current
    cameras. A track whose median reprojection error > `drop_px` is a moving thing / a drifting track -> dropped.
    Observations: every `every`-th visible frame (the track is dense in time) with error < 3 x drop_px.
    Returns {"X" (k, 3), "track" (k,) indices into M, "obs_cam", "obs_pt" (0..k-1), "obs_xy", "kept", "dropped", "short"}."""
    tracks, visible = np.asarray(tracks, np.float64), np.asarray(visible, bool)
    cam_idx = np.asarray(cam_idx, int)
    T_, M = visible.shape
    und = undistort(tracks.reshape(-1, 2), intr).reshape(T_, M, 2)
    P = np.concatenate([Rs[cam_idx], ts[cam_idx][:, :, None]], 2)             # T x 3 x 4
    X_out, tr_out, oc, op, oxy = [], [], [], [], []
    dropped = short = 0
    for m in range(M):
        fr = np.nonzero(visible[:, m])[0]
        if len(fr) < min_vis:
            short += 1
            continue
        u, v = und[fr, m, 0], und[fr, m, 1]
        A = np.concatenate([u[:, None] * P[fr, 2] - P[fr, 0], v[:, None] * P[fr, 2] - P[fr, 1]], 0)
        Vt = np.linalg.svd(A)[2]
        if abs(Vt[-1, 3]) < 1e-12:
            dropped += 1
            continue
        X = Vt[-1, :3] / Vt[-1, 3]
        c = cam_idx[fr]
        z = (Rs[c] @ X + ts[c])[:, 2]
        e = np.linalg.norm(project(Rs[c], ts[c], np.repeat(X[None], len(fr), 0), intr) - tracks[fr, m], axis=1)
        if not np.isfinite(e).all() or np.median(e) > drop_px or (z <= 0).mean() > 0.5:
            dropped += 1
            continue
        j = len(X_out)
        X_out.append(X)
        tr_out.append(m)
        sel = np.arange(m % max(1, int(every)), len(fr), max(1, int(every)))     # offset per track: every frame gets tracks
        sel = sel[e[sel] < 3 * drop_px]
        oc.append(c[sel])
        op.append(np.full(len(sel), j))
        oxy.append(tracks[fr[sel], m])
    cat = (lambda a, shape, dt: np.concatenate(a) if a else np.zeros(shape, dt))
    return {"X": np.array(X_out).reshape(-1, 3), "track": np.array(tr_out, int), "obs_cam": cat(oc, (0,), int),
            "obs_pt": cat(op, (0,), int), "obs_xy": cat(oxy, (0, 2), np.float64),
            "kept": len(X_out), "dropped": dropped, "short": short, "total": int(M)}


# ----------------------------------------------------------------------------- horizon as directions
def _tangent(up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def horizon_dirs(Rs: np.ndarray, hz_cam: np.ndarray, hz_xy: np.ndarray, intr: Sequence[float]) -> np.ndarray:
    """Horizon points -> world directions (n, 3), not normalised: w = R^T (undistort(p), 1)."""
    u = undistort(hz_xy, intr)
    return np.einsum("nji,nj->ni", Rs[hz_cam], np.concatenate([u, np.ones((len(u), 1))], 1))


def horizon_residual(Rs: np.ndarray, hz_cam: np.ndarray, hz_xy: np.ndarray, intr: Sequence[float], up: np.ndarray) -> np.ndarray:
    """How far each horizon point is from the horizon line of its frame, in px of the undistorted image:
    f * (n . (x, y, 1)) / |n_xy| with n = R up (the level plane's normal in the camera) and (x, y) the undistorted
    normalised point. The experiment's `(w @ up) / |w| * f` (angle x focal) is NOT used: sin(angle) * f goes to 0
    with f, so it pulls the focal down (synthetic: -0.3 % at sigma 1 px, a runaway at a strong weight)."""
    w = horizon_dirs(Rs, hz_cam, hz_xy, intr)
    nz = np.asarray(Rs)[hz_cam, 2, :] @ up
    return (w @ up) / np.sqrt(np.maximum(1.0 - nz * nz, 1e-12)) * float(np.sqrt(intr[FX] * intr[FY]))


def up_from_horizon(Rs: np.ndarray, hz_cam: np.ndarray, hz_xy: np.ndarray, intr: Sequence[float]) -> np.ndarray:
    """World up = the normal of the plane the horizon directions lie on (smallest eigenvector), toward the cameras' up."""
    w = horizon_dirs(Rs, hz_cam, hz_xy, intr)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    up = np.linalg.eigh(w.T @ w)[1][:, 0]
    cam_up = np.mean(np.asarray(Rs)[:, 1, :], 0) * -1.0                      # mean of R^T (0, -1, 0)
    return -up if up @ cam_up < 0 else up


# ----------------------------------------------------------------------------- joint bundle adjustment
def _lm_schur(resid: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, n_cam: int, n_pt: int, n_glob: int,
              row_cam: np.ndarray, row_pt: np.ndarray, max_iter: int, f_scale: float, verbose: int = 0,
              tol: float = 1e-4) -> Tuple[np.ndarray, int]:
    """Levenberg-Marquardt with the Schur complement over the points (the classic bundle adjustment step, solved
    exactly) and the soft-L1 loss as iteratively reweighted least squares. Parameters: `n_cam` blocks of 6, `n_pt`
    blocks of 3, `n_glob` shared ones (lens, world up). Every residual row depends on at most one camera block
    (`row_cam`, -1 = none), at most one point (`row_pt`) and the shared parameters – so the Jacobian comes from
    6 + 3 + n_glob forward differences of `resid`. Stops when a step lowers the cost by less than `tol` (relative).
    Returns (x, iterations)."""
    from scipy.sparse import bsr_matrix, coo_matrix
    x = np.array(x0, np.float64)
    OC, OI = 6 * n_cam, 6 * n_cam + 3 * n_pt
    A = OC + n_glob
    has_c, has_p = row_cam >= 0, row_pt >= 0
    rc, rp = np.nonzero(has_c)[0], np.nonzero(has_p)[0]
    eps = 1.5e-8
    C2 = float(f_scale) ** 2

    def cost_of(r):
        return float(C2 * (np.sqrt(1 + r * r / C2) - 1).sum())

    r = resid(x)
    cost = cost_of(r)
    lam, it = 1e-4, 0
    for it in range(1, max_iter + 1):
        nres = len(r)
        Jc, Jp, Jg = np.zeros((nres, 6)), np.zeros((nres, 3)), np.zeros((nres, n_glob))
        for q in range(6):
            idx = 6 * np.arange(n_cam) + q
            h = eps * np.maximum(1.0, np.abs(x[idx]))
            xp = x.copy()
            xp[idx] += h
            Jc[rc, q] = (resid(xp) - r)[rc] / h[row_cam[rc]]
        for q in range(3):
            idx = OC + 3 * np.arange(n_pt) + q
            h = eps * np.maximum(1.0, np.abs(x[idx]))
            xp = x.copy()
            xp[idx] += h
            Jp[rp, q] = (resid(xp) - r)[rp] / h[row_pt[rp]]
        for g in range(n_glob):
            h = eps * max(1.0, abs(x[OI + g]))
            xp = x.copy()
            xp[OI + g] += h
            Jg[:, g] = (resid(xp) - r) / h
        sw = (1 + r * r / C2) ** -0.25                        # sqrt of the IRLS weight of soft-L1
        rw = r * sw
        Jc *= sw[:, None]; Jp *= sw[:, None]; Jg *= sw[:, None]
        rows = np.concatenate([np.repeat(rc, 6), np.repeat(np.arange(nres), n_glob)])
        cols = np.concatenate([(6 * row_cam[rc][:, None] + np.arange(6)).ravel(), np.tile(OC + np.arange(n_glob), nres)])
        Ja = coo_matrix((np.concatenate([Jc[rc].ravel(), Jg.ravel()]), (rows, cols)), shape=(nres, A)).tocsr()
        Jb = coo_matrix((Jp[rp].ravel(), (np.repeat(rp, 3), (3 * row_pt[rp][:, None] + np.arange(3)).ravel())),
                        shape=(nres, 3 * n_pt)).tocsr()
        Haa = (Ja.T @ Ja).toarray()
        Hab = (Ja.T @ Jb).tocsr()
        ga, gb = Ja.T @ rw, (Jb.T @ rw).reshape(n_pt, 3)
        Hbb = np.zeros((n_pt, 3, 3))
        np.add.at(Hbb, row_pt[rp], np.einsum("ni,nj->nij", Jp[rp], Jp[rp]))
        da_diag, db_diag = np.maximum(np.diag(Haa), 1e-12), np.maximum(np.einsum("nii->ni", Hbb), 1e-12)
        ok = False
        for _ in range(10):
            V = Hbb.copy()
            V[:, np.arange(3), np.arange(3)] += lam * db_diag
            Vinv = np.linalg.inv(V)
            Tm = Hab @ bsr_matrix((Vinv, np.arange(n_pt), np.arange(n_pt + 1)), shape=(3 * n_pt, 3 * n_pt))
            Sm = Haa + np.diag(lam * da_diag) - (Tm @ Hab.T).toarray()
            rhs = -ga + Tm @ gb.ravel()
            try:
                da = np.linalg.solve(Sm, rhs)
            except np.linalg.LinAlgError:
                da = np.linalg.lstsq(Sm, rhs, rcond=None)[0]
            db = np.einsum("nij,nj->ni", Vinv, -gb - (Hab.T @ da).reshape(n_pt, 3))
            xn = x.copy()
            xn[:OC] += da[:OC]
            xn[OC:OI] += db.ravel()
            xn[OI:] += da[OC:]
            rn = resid(xn)
            cn = cost_of(rn)
            if np.isfinite(cn) and cn < cost:
                ok = True
                break
            lam *= 4.0
        if not ok:
            break
        gain = (cost - cn) / max(cost, 1e-30)
        if verbose:
            print(f"  LM {it}: cost {cost:.6g} -> {cn:.6g} (lambda {lam:.2g})", flush=True)
        x, r, cost = xn, rn, cn
        lam = max(lam / 3.0, 1e-9)
        if gain < tol:
            break
    return x, it


def bundle_adjust(Rs: np.ndarray, ts: np.ndarray, intr: Sequence[float], pts: np.ndarray, obs_cam: np.ndarray,
                  obs_pt: np.ndarray, obs_xy: np.ndarray, obs_w: Optional[np.ndarray] = None, *,
                  single_focal: bool = True, hz_cam: Optional[np.ndarray] = None, hz_xy: Optional[np.ndarray] = None,
                  sig_hz: float = 1.0, prior: float = 0.05, max_iter: int = 30, f_scale: float = 2.0,
                  solver: str = "schur", tol: float = 1e-4,
                  check: Optional[Callable[[], None]] = None, verbose: int = 0) -> Dict[str, Any]:
    """One least squares over all camera poses (camera 0 fixed) and all points; reprojection residuals in plate px
    through the lens model, weights `obs_w` (1 / sigma), soft-L1. `single_focal`: f (fx = fy), k1, k2 are refined with
    everything, otherwise the intrinsics stay. Horizon points (`hz_cam` camera index, `hz_xy` px) add the level-plane
    constraint with a free world up. `check()` is called during the solve (raise to cancel).
    Returns {"R", "t", "X", "intr", "up" | None, "nfev", "res_before", "res_after"} (res = median |axis residual| px)."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix
    from scipy.spatial.transform import Rotation

    Rs, ts, pts = np.asarray(Rs, np.float64), np.asarray(ts, np.float64), np.asarray(pts, np.float64)
    obs_cam, obs_pt, obs_xy = np.asarray(obs_cam, int), np.asarray(obs_pt, int), np.asarray(obs_xy, np.float64)
    obs_w = np.ones(len(obs_cam)) if obs_w is None else np.asarray(obs_w, np.float64)
    intr = np.array(intr, np.float64)
    if single_focal:
        intr[FX] = intr[FY] = float(np.sqrt(intr[FX] * intr[FY]))
    N, NP, nobs = len(Rs), len(pts), len(obs_cam)
    cx, cy, p1, p2 = intr[CX], intr[CY], intr[P1], intr[P2]
    use_hz = hz_cam is not None and len(hz_cam) > 0
    NI, NU = (3 if single_focal else 0), (2 if use_hz else 0)
    rv0 = Rotation.from_matrix(Rs).as_rotvec()
    C0 = centres(Rs, ts)
    up0 = e1 = e2 = None
    if use_hz:
        hz_cam, hz_xy = np.asarray(hz_cam, int), np.asarray(hz_xy, np.float64)
        up0 = up_from_horizon(Rs, hz_cam, hz_xy, intr)
        e1, e2 = _tangent(up0)
    NH = len(hz_cam) if use_hz else 0
    OC, OI = 6 * (N - 1), 6 * (N - 1) + 3 * NP
    last = [time.time()]

    def unpack(x):
        cams = x[:OC].reshape(N - 1, 6)
        rv = np.vstack([rv0[:1], cams[:, :3]])
        tt = np.vstack([ts[:1], cams[:, 3:]])
        X = x[OC:OI].reshape(NP, 3)
        i = intr.copy()
        if NI:
            i[FX] = i[FY] = x[OI]
            i[K1], i[K2] = x[OI + 1], x[OI + 2]
        up = None
        if NU:
            up = up0 + x[OI + NI] * e1 + x[OI + NI + 1] * e2
            up = up / np.linalg.norm(up)
        return rv, tt, X, i, up

    def resid(x):
        if check is not None and time.time() - last[0] > 0.5:
            last[0] = time.time()
            check()
        rv, tt, X, i, up = unpack(x)
        Rm = Rotation.from_rotvec(rv).as_matrix()
        Xc = np.einsum("nij,nj->ni", Rm[obs_cam], X[obs_pt]) + tt[obs_cam]
        z = np.maximum(Xc[:, 2], 1e-6)
        xd, yd = distort(Xc[:, 0] / z, Xc[:, 1] / z, i[K1], i[K2], p1, p2)
        out = [(xd * i[FX] + cx - obs_xy[:, 0]) * obs_w, (yd * i[FY] + cy - obs_xy[:, 1]) * obs_w,
               (centres(Rm, tt)[1:] - C0[1:]).ravel() * prior * 100]
        if NH:
            out.append(horizon_residual(Rm, hz_cam, hz_xy, i, up) / sig_hz)
        return np.concatenate(out)

    # sparsity pattern of the Jacobian (rows: u, v, centre prior, horizon)
    rr, cc = [], []
    ar = np.arange(nobs)
    m = obs_cam > 0
    for k in range(2):
        for q in range(6):
            rr.append(ar[m] + k * nobs); cc.append(6 * (obs_cam[m] - 1) + q)
        for q in range(3):
            rr.append(ar + k * nobs); cc.append(OC + 3 * obs_pt + q)
        for q in range(NI):
            rr.append(ar + k * nobs); cc.append(np.full(nobs, OI + q))
    for i_ in range(1, N):
        for q in range(3):
            rr.append(np.full(6, 2 * nobs + 3 * (i_ - 1) + q)); cc.append(6 * (i_ - 1) + np.arange(6))
    if NH:
        hr = 2 * nobs + 3 * (N - 1) + np.arange(NH)
        mh = hz_cam > 0
        for q in range(6):
            rr.append(hr[mh]); cc.append(6 * (hz_cam[mh] - 1) + q)
        for q in range(NI + NU):
            rr.append(hr); cc.append(np.full(NH, OI + q))
    rr, cc = np.concatenate(rr), np.concatenate(cc)
    nres = 2 * nobs + 3 * (N - 1) + NH
    J = coo_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(nres, OI + NI + NU)).tocsr()

    x0 = np.concatenate([np.hstack([rv0[1:], ts[1:]]).ravel(), pts.ravel(),
                         np.array([intr[FX], intr[K1], intr[K2]])[:NI], np.zeros(NU)])
    ww = np.concatenate([obs_w, obs_w])
    r0 = resid(x0)[:2 * nobs] / ww
    if solver == "scipy":                                     # the experiment's solver (joint_ba.py): inexact steps, slow
        sol = least_squares(resid, x0, jac_sparsity=J, method="trf", loss="soft_l1", f_scale=f_scale, x_scale="jac",
                            max_nfev=int(max_iter), tr_solver="lsmr", verbose=verbose)
        xs, nfev = sol.x, int(sol.nfev)
    else:
        row_cam = np.concatenate([obs_cam - 1, obs_cam - 1, np.repeat(np.arange(N - 1), 3),
                                  (hz_cam - 1) if NH else np.zeros(0, int)])
        row_pt = np.concatenate([obs_pt, obs_pt, np.full(3 * (N - 1) + NH, -1)])
        xs, nfev = _lm_schur(resid, x0, N - 1, NP, NI + NU, row_cam, row_pt, int(max_iter), f_scale, verbose, tol)
    rv, tt, X, i1, up = unpack(xs)
    r1 = resid(xs)[:2 * nobs] / ww
    return {"R": Rotation.from_rotvec(rv).as_matrix(), "t": tt, "X": X.copy(), "intr": i1, "up": up,
            "nfev": nfev, "res_before": float(np.median(np.abs(r0))) if nobs else 0.0,
            "res_after": float(np.median(np.abs(r1))) if nobs else 0.0}


# ----------------------------------------------------------------------------- moving pixels (epipolar test)
def epipolar_moving(grays: Sequence[np.ndarray], Rs: np.ndarray, ts: np.ndarray, intr: Sequence[float], scale: float,
                    *, thr_px: float = 1.5, check: Optional[Callable[[], None]] = None) -> List[np.ndarray]:
    """Per frame a bool mask (analysis resolution) of pixels that move on their own: with known cameras the match of a
    static point lies on its epipolar line in the neighbouring frame whatever its depth. Dense flow (DIS) to the
    previous and the next frame, trusted only where forward ~ -backward; a pixel is moving when it is off the epipolar
    line by more than `thr_px` analysis px on BOTH sides (one side for the first / last frame).
    `grays`: uint8 (h, w) per camera, `scale` = analysis px / plate px."""
    n = len(grays)
    h, w = grays[0].shape[:2]
    K, dist = k_matrix(intr), np.array([intr[K1], intr[K2], intr[P1], intr[P2]])
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    full = np.stack([(gx.ravel() + 0.5) / scale - 0.5, (gy.ravel() + 0.5) / scale - 0.5], 1).astype(np.float64)
    ray = np.concatenate([cv2.undistortPoints(full[:, None], K, dist).reshape(h, w, 2), np.ones((h, w, 1))], -1).astype(np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    f_an = float(np.sqrt(intr[FX] * intr[FY])) * scale
    out = []
    for t in range(n):
        if check is not None:
            check()
        votes = np.zeros((h, w), np.int8)
        for d in (-1, 1):
            j = t + d
            if not 0 <= j < n:
                continue
            fwd = dis.calc(grays[t], grays[j], None)
            bwd = dis.calc(grays[j], grays[t], None)
            ex, ey = gx + fwd[..., 0], gy + fwd[..., 1]
            bx = cv2.remap(bwd[..., 0], ex, ey, cv2.INTER_LINEAR)
            by = cv2.remap(bwd[..., 1], ex, ey, cv2.INTER_LINEAR)
            fb = np.hypot(fwd[..., 0] + bx, fwd[..., 1] + by)
            reliable = fb < np.maximum(1.0, 0.1 * np.linalg.norm(fwd, axis=-1))
            Rrel = Rs[j] @ Rs[t].T
            trel = ts[j] - Rrel @ ts[t]
            tx = np.array([[0, -trel[2], trel[1]], [trel[2], 0, -trel[0]], [-trel[1], trel[0], 0]])
            E = (tx @ Rrel).astype(np.float32)
            efull = np.stack([(ex + 0.5) / scale - 0.5, (ey + 0.5) / scale - 0.5], -1).reshape(-1, 1, 2).astype(np.float64)
            x2 = np.concatenate([cv2.undistortPoints(efull, K, dist).reshape(h, w, 2).astype(np.float32),
                                 np.ones((h, w, 1), np.float32)], -1)
            line = ray @ E.T
            dd = np.abs((x2 * line).sum(-1)) / np.maximum(np.hypot(line[..., 0], line[..., 1]), 1e-9) * f_an
            votes += ((dd > thr_px) & reliable).astype(np.int8)
        dyn = votes >= (1 if t in (0, n - 1) else 2)
        dyn = cv2.morphologyEx(dyn.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out.append(cv2.dilate(dyn, np.ones((9, 9), np.uint8)) > 0)
    return out


def sample_mask(mask: np.ndarray, xy: np.ndarray, scale: float) -> np.ndarray:
    """Is plate pixel `xy` (n, 2) inside the analysis-resolution `mask`? Outside the image = False."""
    h, w = mask.shape[:2]
    x = np.round((np.asarray(xy)[:, 0] + 0.5) * scale - 0.5).astype(int)
    y = np.round((np.asarray(xy)[:, 1] + 0.5) * scale - 0.5).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros(len(x), bool)
    out[ok] = mask[y[ok], x[ok]]
    return out


# ----------------------------------------------------------------------------- horizon of a frame
def detect_horizon(gray: np.ndarray, moving: Optional[np.ndarray] = None, *, min_support: float = 0.35,
                   ) -> Optional[np.ndarray]:
    """Sub-pixel points (n, 2 plate px) on the horizon of one frame, or None when the frame has no straight strong
    edge across it. Step 1 (<= 1280 px wide): per column the strongest vertical luminance edge of one polarity,
    RANSAC line; it must carry >= `min_support` of the columns. Step 2 (full resolution): the edge row around that
    line every ~1 % of the width (parabola peak), columns near `moving` (bool, plate size) skipped, robust quadratic.
    `gray`: uint8 / float (h, w), display values 0..255."""
    g_full = np.asarray(gray, np.float32)
    H, W = g_full.shape[:2]
    s = min(1.0, 1280.0 / W)
    sm = cv2.resize(g_full, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else g_full
    h, w = sm.shape
    gy = cv2.Sobel(cv2.GaussianBlur(sm, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=5)
    mv = None
    if moving is not None and moving.any():
        mv = cv2.resize(moving.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mv = cv2.dilate(mv, np.ones((25, 25), np.uint8)) > 0
    top, bot = int(0.05 * h), int(0.95 * h)
    xs = np.arange(0, w, 4)
    best = None
    rng = np.random.default_rng(0)
    for pol in (-1.0, 1.0):                                   # -1: darker below (sky -> sea), +1: brighter below
        col = pol * gy[top:bot][:, xs]
        if mv is not None:
            col = np.where(mv[top:bot][:, xs], -1e9, col)
        yi = np.argmax(col, 0)
        st = col[yi, np.arange(len(xs))]
        ok = (st > 40) & (st > 4 * np.median(np.abs(col), 0))
        px, py = xs[ok].astype(np.float64), (top + yi[ok]).astype(np.float64)
        if len(px) < 8:
            continue
        cnt, inl = 0, None
        for _ in range(300):
            i, j = rng.choice(len(px), 2, replace=False)
            if abs(px[j] - px[i]) < 0.04 * w:
                continue
            a = (py[j] - py[i]) / (px[j] - px[i])
            m = np.abs(py - (a * px + py[i] - a * px[i])) < 3
            if m.sum() > cnt:
                cnt, inl = int(m.sum()), m
        if inl is not None and (best is None or cnt > best[0]):
            best = (cnt, pol, np.polyfit(px[inl], py[inl], 1))
    if best is None or best[0] < min_support * len(xs):
        return None
    _, pol, (a, b) = best

    def centre(x_full):                                       # the line in plate px
        return ((a * ((x_full + 0.5) * s - 0.5) + b) + 0.5) / s - 0.5

    k = W / 3840.0
    gyf = pol * cv2.Sobel(cv2.GaussianBlur(g_full, (0, 0), max(1.0, 2.0 * k)), cv2.CV_32F, 0, 1, ksize=5)
    r, hw, vr = max(8.0, 110 * k), max(2, round(6 * k)), max(6, round(100 * k))
    md = None
    if moving is not None and moving.any():
        md = cv2.dilate(moving.astype(np.uint8), np.ones((1, 2 * max(3, round(60 * k)) + 1), np.uint8)) > 0
    X_, Y_ = [], []
    for x in np.arange(max(hw, 20 * k), W - max(hw, 20 * k), max(4.0, 40 * k)):
        xi = int(x)
        c = centre(x)
        lo, hi = int(max(0, c - r)), int(min(H, c + r + 1))
        if hi - lo < 5:
            continue
        prof = gyf[lo:hi, xi - hw:xi + hw + 1].mean(1)
        j = int(np.argmax(prof))
        if not 0 < j < len(prof) - 1 or prof[j] / (np.median(np.abs(prof)) + 1e-3) <= 6:
            continue
        y = lo + j + 0.5 * (prof[j - 1] - prof[j + 1]) / (prof[j - 1] - 2 * prof[j] + prof[j + 1] - 1e-9)
        if md is not None and md[int(max(0, y - vr)):int(min(H, y + vr)), xi].any():
            continue
        X_.append(float(xi)); Y_.append(float(y))
    if len(X_) < 8:
        return None
    X_, Y_ = np.array(X_), np.array(Y_)
    keep = np.ones(len(X_), bool)
    for _ in range(4):
        if keep.sum() < 6:
            return None
        q = np.polyfit(X_[keep], Y_[keep], 2)
        e = np.abs(np.polyval(q, X_) - Y_)
        keep = e < max(3.0 * max(k, 0.5), 2.5 * np.median(e[keep]))
    if keep.sum() < 8:
        return None
    return np.stack([X_[keep], Y_[keep]], 1)


# ----------------------------------------------------------------------------- planes for the cards step
def fit_plane(points: np.ndarray, mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, *,
              centres: Optional[np.ndarray] = None, thr_rel: float = 0.02, iters: int = 3000, seed: int = 0,
              ) -> Optional[Dict[str, Any]]:
    """RANSAC plane through sparse points (the ground: most static points of a shot lie on it).
    `mask_fn(points) -> bool (n,)` picks the points to fit (e.g. those that project inside a region the user drew);
    None = all. Inlier distance = `thr_rel` x the median distance of the points from the mean camera centre
    (`centres` (m, 3); without it from the points' own centroid). The normal points toward the cameras (Card.plane).
    Returns {"n": [x, y, z], "d": float, "inliers": int, "share": float} (n . X + d = 0) or None (< 3 points)."""
    P = np.asarray(points, np.float64).reshape(-1, 3)
    if mask_fn is not None:
        P = P[np.asarray(mask_fn(P), bool)]
    P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return None
    Cm = np.asarray(centres, np.float64).reshape(-1, 3).mean(0) if centres is not None else P.mean(0)
    thr = thr_rel * float(np.median(np.linalg.norm(P - Cm, axis=1)))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iters)):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        d = -float(n @ s[0])
        cnt = int((np.abs(P @ n + d) < thr).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None:
        return None
    cnt, n, d = best
    inl = P[np.abs(P @ n + d) < thr]
    if len(inl) >= 3:                                         # least-squares plane of the consensus set
        c0 = inl.mean(0)
        n2 = np.linalg.svd(inl - c0, full_matrices=False)[2][-1]
        n, d = (n2 if n2 @ n > 0 else -n2), -float((n2 if n2 @ n > 0 else -n2) @ c0)
        cnt = int((np.abs(P @ n + d) < thr).sum())
    if centres is not None and float(Cm @ n + d) < 0:
        n, d = -n, -d
    return {"n": [float(v) for v in n], "d": float(d), "inliers": cnt, "share": round(cnt / len(P), 4)}


def align_similarity(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: scale s, rotation R, translation t with B ~ s * R @ A + t (points (n, 3))."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (A0 ** 2).sum() * len(A))
    return s, R, mb - s * R @ ma


def align_cameras(Rs_a: np.ndarray, ts_a: np.ndarray, Rs_b: np.ndarray, ts_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Similarity between two solves of the same cameras: X_b ~ s * R @ X_a + t. The rotation comes from the cameras'
    orientations (a camera path is often a straight line – its centres alone leave the roll about it free), scale and
    translation from the centres."""
    Rs_a, Rs_b = np.asarray(Rs_a, np.float64), np.asarray(Rs_b, np.float64)
    U, _, Vt = np.linalg.svd(np.einsum("nji,njk->ik", Rs_b, Rs_a))          # sum of R_b^T R_a
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    A, B = centres(Rs_a, ts_a) @ R.T, centres(Rs_b, ts_b)
    A0, B0 = A - A.mean(0), B - B.mean(0)
    s = float((A0 * B0).sum() / max((A0 ** 2).sum(), 1e-30))
    return s, R, B.mean(0) - s * A.mean(0)
