# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P10 – the camera solve (docs/PLAN_CLEANUP_3D.md 5b: solved in the program, no import).

`solve_camera(work, frames, plate_path_fn, cfg=...)`:
  1. one pass over the plates (read only): frames exported at the working resolution into a temporary workspace with
     the moving-object masks as COLMAP masks, small frames for CoTracker3 and for the flow, the horizon of each frame;
  2. COLMAP (CUDA, subprocess under the GPU lease – solve_colmap.py): SIFT, sequential matching, mapper;
  3. CoTracker3 tracks on a grid outside the masks (the existing worker, engine/tracking.py);
  4. the joint bundle adjustment (solve_ba.py): one focal, k1 / k2, the horizon as a constraint when the shot has
     one; then the epipolar test marks pixels that move on their own, their observations go out and the adjustment
     runs again (at most `rounds` = 2 adjustments);
  5. the result as `Camera3D` (plate frame numbers, plate pixels, OpenCV distortion) with the report, saved through
     `store.save_camera`; the sparse points next to it (`points.npy`) for the cards step.
For the cards step (P8): `propose_planes(work)` – the ground (RANSAC, `fit_plane`) and the sea level (level plane from
the horizon's world up, `horizon_up` / `sea_plane`).
Errors are `SolveError` with a Polish message (COLMAP missing, too few registered frames, cancelled).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from engine.cleanup3d import solve_ba as BA
from engine.cleanup3d import solve_colmap as CM
from engine.cleanup3d import store
from engine.cleanup3d import types as T
from engine.cleanup3d.solve_ba import align_similarity, detect_horizon, fit_plane  # noqa: F401  (public helpers)
from engine.cleanup3d.solve_colmap import ColmapRunner, SolveCancelled, SolveError  # noqa: F401

log = logging.getLogger(__name__)

# stage key -> the text of the progress window (user-facing, Polish)
STAGES = {
    "prepare": "Przygotowanie klatek",
    "features": "COLMAP: wykrywanie cech",
    "matching": "COLMAP: dopasowanie klatek",
    "mapper": "COLMAP: rekonstrukcja kamery",
    "tracks": "CoTracker3: ścieżki punktów",
    "bundle": "Wspólne wyrównanie kamery (bundle adjustment)",
    "epipolar": "Test epipolarny: ruchome piksele",
    "report": "Raport solve'u",
}

DEFAULTS: Dict[str, Any] = {
    "max_side": 3840,            # working resolution of COLMAP (longer side); larger plates are exported smaller
    "max_features": 16000, "peak_threshold": 0.002,
    "overlap": None,             # frames matched ahead of each frame; None = solve_colmap.default_overlap
    "mask_dilate": 41,           # px at 3840 wide: margin around the moving-object masks
    "analysis_side": 960,        # flow / epipolar test resolution
    "track_side": 1024, "track_step": 16, "track_query_frames": 7, "track_max_frames": 120, "track_min_vis": 6,
    "sig_track": 3.0,            # px at 3840 wide (CoTracker3 runs at 1024); tracks with median error > 2 sigma go out
    "sig_hz": None,              # px: sigma of a horizon point; None = measured (scatter about each frame's own curve)
    "prior": 0.05, "max_iter": 30, "ba_tol": 1e-4, "f_scale": 2.0,
    "rounds": 2,                 # bundle adjustments (epipolar test between them)
    "epipolar": True, "epi_thr_px": 1.5,
    "min_registered": 0.6,       # share of the frames COLMAP must register (the rest is interpolated, then adjusted)
    "hz_min_frames": 0.5,        # share of the frames that must show the horizon
    "hz_accept_px": 8.0,         # px at 3840 wide: the edge must lie on one level plane this well before it is used
    "keep_workspace": False,
}

Masks = Union[None, Dict[int, np.ndarray], Callable[[int], Optional[np.ndarray]]]


# ----------------------------------------------------------------------------- reading plates (never written)
def read_plate_u8(path: Path) -> np.ndarray:
    """A plate frame as 8-bit BGR for the solve (EXR through the PlateToDisplay curve)."""
    path = Path(path)
    if path.suffix.lower() == ".exr":
        from engine.ingest import plate_to_display_float
        from engine.plate_qa import read_frame_linear
        lin = read_frame_linear(path)
        if lin is None:
            raise SolveError(f"Nie można odczytać klatki plate'u: {path.name}")
        return (plate_to_display_float(lin) * 255.0 + 0.5).astype(np.uint8)
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED) if path.is_file() else None
    if img is None:
        raise SolveError(f"Nie można odczytać klatki plate'u: {path.name}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img / 257.0 + 0.5).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(img)


def _mask_of(masks: Masks, frame: int, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    m = None if masks is None else (masks.get(frame) if isinstance(masks, dict) else masks(frame))
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] in (2, 4) else m.max(2)
    m = m > 0
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


def _fit_side(w: int, h: int, side: int) -> Tuple[float, int, int]:
    s = min(1.0, float(side) / max(w, h))
    return s, max(1, round(w * s)), max(1, round(h * s))


# ----------------------------------------------------------------------------- poses of frames COLMAP skipped
def _fill_poses(n: int, known: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Poses for all `n` cameras: the registered ones as they are, the others interpolated between their registered
    neighbours (rotation slerp, centre linear; before the first / after the last = the nearest)."""
    from scipy.spatial.transform import Rotation, Slerp
    idx = sorted(known)
    Rk = Rotation.from_matrix(np.array([known[i][0] for i in idx]))
    Ck = np.array([-known[i][0].T @ known[i][1] for i in idx])
    q = np.clip(np.arange(n), idx[0], idx[-1])
    Rq = (Slerp(idx, Rk)(q) if len(idx) > 1 else Rotation.concatenate([Rk[0]] * n)).as_matrix()
    Cq = np.stack([np.interp(q, idx, Ck[:, k]) for k in range(3)], 1)
    for i in idx:
        Rq[i] = known[i][0]
    ts = -np.einsum("nij,nj->ni", Rq, Cq)
    for i in idx:
        ts[i] = known[i][1]
    return Rq, ts, [i for i in range(n) if i not in known]


# ----------------------------------------------------------------------------- the solve
def solve_camera(work: Path, frames: Sequence[int], plate_path_fn: Callable[[int], Path], *, cfg: Dict[str, Any],
                 moving_masks: Masks = None, single_focal: bool = True, horizon: bool = True,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 cancel_check: Optional[Callable[[], Any]] = None,
                 options: Optional[Dict[str, Any]] = None, runner: Any = None, tracker: Any = None,
                 save: bool = True) -> T.Camera3D:
    """Solve the camera of plate frames `frames` (plate frame numbers; `plate_path_fn(frame)` -> file, read only).
    `work` = the tool's folder of the shot (`<work>/matte_desk`); the result goes to `<work>/inpaint/camera3d/`.
    `moving_masks`: {frame: mask} or `fn(frame) -> mask | None` (plate size, > 0 = a moving thing, excluded).
    `single_focal`: one focal fx = fy with k1 / k2 refined; False = the lens as COLMAP estimated it.
    `horizon`: use the horizon as a constraint when the shot has one (silently skipped when it has none).
    `progress(label, done, total)`; `cancel_check()` raises (or returns True) to stop.
    `options` overrides DEFAULTS (also `cfg["cleanup3d_solve"]`); `runner` / `tracker` replace COLMAP / CoTracker3
    (tests). Returns the saved `Camera3D` (report in `.report`)."""
    opt = {**DEFAULTS, **((cfg or {}).get("cleanup3d_solve") or {}), **(options or {})}
    frames = [int(f) for f in frames]
    n = len(frames)
    if n < 3 or len(set(frames)) != n:
        raise SolveError(f"Solve kamery wymaga co najmniej 3 różnych klatek (podano {n})")
    t_all = time.time()
    secs: Dict[str, float] = {}

    def check() -> None:
        if cancel_check is not None and cancel_check():
            raise SolveCancelled()

    def stage(key: str, done: int = 0, total: int = 0) -> None:
        check()
        if progress is not None:
            progress(STAGES[key], done, total)

    if runner is None:
        runner_exe = CM.find_exe(cfg)                        # SolveError with the gate's reason when missing
    cam_dir = T.camera_file(work).parent
    cam_dir.mkdir(parents=True, exist_ok=True)
    if runner is None:
        (cam_dir / "colmap.log").unlink(missing_ok=True)
        runner = ColmapRunner(runner_exe, cam_dir / "colmap.log")
    ws = Path(tempfile.mkdtemp(prefix="md_solve_", dir=str(cam_dir)))
    try:
        cam, extra = _solve(ws, frames, plate_path_fn, opt, moving_masks, single_focal, horizon, stage, check,
                            runner, tracker, secs)
    finally:
        if not opt["keep_workspace"]:
            shutil.rmtree(ws, ignore_errors=True)
    secs["total"] = round(time.time() - t_all, 1)
    cam.report["seconds"] = secs
    if save:
        store.save_camera(work, cam)
        np.save(cam_dir / "points.npy", extra["points"].astype(np.float32))
        np.save(cam_dir / "track_points.npy", extra["track_points"].astype(np.float32))
        hz = cam_dir / "horizon.npz"
        if extra["hz_xy"] is not None:
            np.savez(hz, frame=extra["hz_frame"], xy=extra["hz_xy"].astype(np.float32))
        else:
            hz.unlink(missing_ok=True)
    return cam


def _solve(ws: Path, frames: List[int], plate_path_fn, opt: Dict[str, Any], moving_masks: Masks, single_focal: bool,
           horizon: bool, stage, check, runner, tracker, secs: Dict[str, float]) -> Tuple[T.Camera3D, Dict[str, Any]]:
    n = len(frames)
    # ---- 1. one pass over the plates
    t0 = time.time()
    (ws / "images").mkdir(parents=True, exist_ok=True)
    (ws / "masks").mkdir(exist_ok=True)
    stride = max(1, -(-n // int(opt["track_max_frames"])))
    tci = np.arange(0, n, stride)                              # cameras CoTracker3 sees
    W = H = 0
    s_w = s_a = s_t = 1.0
    grays: List[np.ndarray] = []
    small: List[np.ndarray] = []
    bad_small: Dict[int, np.ndarray] = {}
    hz_pts: Dict[int, np.ndarray] = {}
    any_mask = False
    for i, f in enumerate(frames):
        stage("prepare", i, n)
        img = read_plate_u8(Path(plate_path_fn(f)))
        if i == 0:
            H, W = img.shape[:2]
            s_w, ww, wh = _fit_side(W, H, int(opt["max_side"]))
            s_a, aw, ah = _fit_side(W, H, int(opt["analysis_side"]))
            s_t, tw, th = _fit_side(W, H, int(opt["track_side"]))
            dil = max(3, round(opt["mask_dilate"] * W / 3840.0 * s_t) | 1)
        elif img.shape[:2] != (H, W):
            raise SolveError(f"Klatka {f} ma inny rozmiar ({img.shape[1]}×{img.shape[0]}) niż pierwsza ({W}×{H})")
        m = _mask_of(moving_masks, f, (H, W))
        wimg = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA) if s_w < 1 else img
        name = CM.image_name(f)
        cv2.imencode(".jpg", wimg, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(ws / "images" / name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA))
        bad = np.zeros((th, tw), bool)
        if m is not None and m.any():
            any_mask = True
            bad = cv2.dilate(cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST),
                             np.ones((dil, dil), np.uint8)) > 0
        keep = cv2.resize((~bad).astype(np.uint8) * 255, (ww, wh), interpolation=cv2.INTER_NEAREST)
        cv2.imencode(".png", keep)[1].tofile(str(ws / "masks" / (name + ".png")))      # COLMAP: 0 = ignore
        if i in tci:
            small.append(cv2.cvtColor(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            bad_small[i] = bad
        if horizon:
            p = detect_horizon(gray, m)
            if p is not None:
                hz_pts[i] = p
        del img, gray
    k = W / 3840.0
    secs["prepare"] = round(time.time() - t0, 1)

    # ---- 2. COLMAP
    tt = {"t": time.time(), "key": None}

    def cstage(key: str) -> None:
        if tt["key"]:
            secs["colmap_" + tt["key"]] = round(time.time() - tt["t"], 1)
        tt["t"], tt["key"] = time.time(), key
        stage(key, 0, 0)

    model = CM.run_sfm(runner, ws, n, masks=any_mask, max_features=opt["max_features"],
                       peak_threshold=opt["peak_threshold"], max_image_size=int(opt["max_side"]),
                       overlap=opt["overlap"], stage=cstage, cancel_check=check)
    secs["colmap_" + tt["key"]] = round(time.time() - tt["t"], 1)
    version = runner.version()
    cam_of = {f: i for i, f in enumerate(frames)}
    known = {cam_of[CM.frame_of(nm)]: (model.R[nm], model.t[nm]) for nm in model.R if CM.frame_of(nm) in cam_of}
    if len(known) < max(3, int(np.ceil(opt["min_registered"] * n))):
        raise SolveError(f"COLMAP zarejestrował tylko {len(known)} z {n} klatek – za mało wspólnych punktów "
                         "(sprawdź maski ruchomych obiektów i ruch kamery)")
    Rs, ts, interpolated = _fill_poses(n, known)
    intr = CM.to_plate(model.intrinsics(), s_w)
    if single_focal:
        intr[BA.FX] = intr[BA.FY] = float(np.sqrt(intr[BA.FX] * intr[BA.FY]))
    pid_row = {int(p): j for j, p in enumerate(model.point_ids)}
    sc, sp, sxy = [], [], []
    for nm, (xy, pid) in model.obs.items():
        if CM.frame_of(nm) not in cam_of or not len(pid):
            continue
        rows = np.array([pid_row.get(int(p), -1) for p in pid])
        ok = rows >= 0
        sc.append(np.full(int(ok.sum()), cam_of[CM.frame_of(nm)])); sp.append(rows[ok])
        sxy.append(CM.xy_to_plate(xy[ok], s_w))
    s_cam, s_pt, s_xy = np.concatenate(sc), np.concatenate(sp), np.concatenate(sxy)
    P_sift = model.points.copy()
    s_keep = np.ones(len(s_cam), bool)

    # ---- 3. CoTracker3 tracks outside the masks
    t0 = time.time()
    stage("tracks", 0, 0)
    tracks = visible = None
    track_note = ""
    if tracker is None:
        tracker, track_note = _default_tracker(int(opt["track_side"]))
    if tracker is not None:
        q = _queries(small, bad_small, tci, int(opt["track_step"]), int(opt["track_query_frames"]))
        if len(q):
            res = tracker.track(small, q, check)
            tracks = (np.asarray(res.tracks, np.float64) + 0.5) / s_t - 0.5            # plate px
            visible = np.asarray(res.visible, bool).copy()
        else:
            track_note = "brak faktury poza maskami"
    secs["cotracker3"] = round(time.time() - t0, 1)

    # ---- horizon: used only when the edge lies on one level plane through the cameras (asked again after a round:
    # a rough COLMAP start may hide a good horizon)
    hz_cam = hz_xy = None
    hz_info: Dict[str, Any] = {"used": False}
    hz_cand: Optional[Tuple[np.ndarray, np.ndarray]] = None
    sig_hz = float(opt["sig_hz"] or 1.0)
    if horizon:
        if len(hz_pts) >= max(3, opt["hz_min_frames"] * n):
            hz_cand = (np.concatenate([np.full(len(p), i) for i, p in sorted(hz_pts.items())]),
                       np.concatenate([p for _, p in sorted(hz_pts.items())]))
            if not opt["sig_hz"]:
                sig_hz = _horizon_sigma(hz_pts)
        else:
            hz_info["reason"] = f"horyzont widoczny w {len(hz_pts)} z {n} klatek"

    def hz_gate() -> bool:
        nonlocal hz_cam, hz_xy, hz_info
        if hz_cand is None or hz_cam is not None:
            return False
        hc, hx = hz_cand
        med = float(np.median(np.abs(BA.horizon_residual(Rs, hc, hx, intr, BA.up_from_horizon(Rs, hc, hx, intr)))))
        if med <= max(2.0, opt["hz_accept_px"] * k):
            hz_cam, hz_xy = hc, hx
            hz_info = {"used": True, "frames": len(hz_pts), "points": int(len(hc)), "sigma_px": round(sig_hz, 3),
                       "res_px_median_before": round(med, 3)}
            return True
        hz_info["reason"] = f"krawędź nie leży na jednej płaszczyźnie poziomu ({med:.1f} px)"
        return False

    hz_gate()
    sig_t = max(0.8, opt["sig_track"] * k)

    # ---- 4. joint bundle adjustment, epipolar test between the rounds
    t_ba = t_epi = 0.0
    rounds = 0
    tri: Dict[str, Any] = {"X": np.zeros((0, 3)), "kept": 0, "dropped": 0, "short": 0, "total": 0,
                           "obs_cam": np.zeros(0, int), "obs_pt": np.zeros(0, int), "obs_xy": np.zeros((0, 2))}
    moving_share: Optional[float] = None
    used = np.zeros(len(P_sift), bool)
    sol: Dict[str, Any] = {}
    for rnd in range(max(1, int(opt["rounds"]))):
        t0 = time.time()
        stage("bundle", rnd, max(1, int(opt["rounds"])))
        cnt = np.bincount(s_pt[s_keep], minlength=len(P_sift))
        used = cnt >= 2
        new = np.cumsum(used) - 1
        om = s_keep & used[s_pt]
        if tracks is not None:
            tri = BA.triangulate_tracks(tracks, visible, tci, Rs, ts, intr, min_vis=int(opt["track_min_vis"]),
                                        drop_px=2 * sig_t)
        ns = int(used.sum())
        if ns + tri["kept"] < 20:
            raise SolveError("Za mało nieruchomych punktów do wyrównania kamery – sprawdź maski ruchomych obiektów")
        pts = np.concatenate([P_sift[used], tri["X"]])
        o_cam = np.concatenate([s_cam[om], tri["obs_cam"]])
        o_pt = np.concatenate([new[s_pt[om]], ns + tri["obs_pt"]])
        o_xy = np.concatenate([s_xy[om], tri["obs_xy"]])
        o_w = np.concatenate([np.ones(int(om.sum())), np.full(len(tri["obs_cam"]), 1.0 / sig_t)])
        sol = BA.bundle_adjust(Rs, ts, intr, pts, o_cam, o_pt, o_xy, o_w, single_focal=single_focal, hz_cam=hz_cam,
                               hz_xy=hz_xy, sig_hz=sig_hz, prior=opt["prior"], max_iter=opt["max_iter"],
                               tol=opt["ba_tol"], f_scale=opt["f_scale"], check=check)
        Rs, ts, intr = sol["R"], sol["t"], sol["intr"]
        P_sift[used] = sol["X"][:ns]
        sol.update({"ns": ns, "o_cam": o_cam, "o_pt": o_pt, "o_xy": o_xy})
        rounds += 1
        t_ba += time.time() - t0
        if rnd + 1 >= int(opt["rounds"]):
            break
        again = hz_gate()                                      # the horizon fits now -> one more round with it
        if not opt["epipolar"]:
            if again:
                continue
            break
        t0 = time.time()
        stage("epipolar", rnd, 1)
        dyn = BA.epipolar_moving(grays, Rs, ts, intr, s_a, thr_px=opt["epi_thr_px"], check=check)
        t_epi += time.time() - t0
        moving_share = float(np.median([d.mean() for d in dyn]))
        if moving_share > 0.5 or not any(d.any() for d in dyn):
            if again:
                continue
            break                                              # nothing found / the test is not usable on this shot
        in_dyn = np.zeros(len(s_cam), bool)
        for i in range(n):
            mi = s_cam == i
            in_dyn[mi] = BA.sample_mask(dyn[i], s_xy[mi], s_a)
        share = np.bincount(s_pt, weights=in_dyn, minlength=len(P_sift)) / np.maximum(np.bincount(s_pt, minlength=len(P_sift)), 1)
        s_keep = ~in_dyn & (share[s_pt] <= 0.5)
        if tracks is not None:
            td = np.stack([BA.sample_mask(dyn[c], tracks[r], s_a) for r, c in enumerate(tci)])
            vis_n = np.maximum(visible.sum(0), 1)
            visible &= ~td & (((td & visible).sum(0) / vis_n) <= 0.5)[None]
    secs["bundle"], secs["epipolar"] = round(t_ba, 1), round(t_epi, 1)

    # ---- 5. Camera3D + report
    stage("report", 0, 0)
    err = BA.reproj_px(Rs, ts, intr, sol["X"], sol["o_cam"], sol["o_pt"], sol["o_xy"])
    ns = sol["ns"]
    is_sift = sol["o_pt"] < ns
    axis = np.abs(BA.project(Rs[sol["o_cam"]], ts[sol["o_cam"]], sol["X"][sol["o_pt"]], intr) - sol["o_xy"])
    per_frame = {frames[i]: float(np.median(err[sol["o_cam"] == i])) for i in range(n) if (sol["o_cam"] == i).any()}
    worst = sorted(per_frame.items(), key=lambda kv: -kv[1])[:5]
    report: Dict[str, Any] = {
        "reproj_px": round(float(np.median(err)), 3), "reproj_px_p95": round(float(np.percentile(err, 95)), 3),
        "sift_reproj_axis_px_median": round(float(np.median(axis[is_sift])), 3) if is_sift.any() else None,
        "focal_px": round(float(np.sqrt(intr[BA.FX] * intr[BA.FY])), 2),
        "fy_over_fx": round(float(intr[BA.FY] / intr[BA.FX]), 5),
        "k1": round(float(intr[BA.K1]), 6), "k2": round(float(intr[BA.K2]), 6),
        "points": ns, "tracks": int(tri["kept"]), "observations": int(len(err)),
        "tracks_total": int(tri["total"]), "tracks_dropped": int(tri["dropped"]), "tracks_short": int(tri["short"]),
        "frames": n, "registered": len(known), "interpolated_frames": [frames[i] for i in interpolated],
        "rounds": rounds, "single_focal": bool(single_focal), "work_scale": round(s_w, 5),
        "worst_frames": [{"frame": int(f), "err_px": round(e, 3)} for f, e in worst],
        "frame_reproj_px": {str(f): round(e, 3) for f, e in per_frame.items()},
        "colmap_version": version,
    }
    if moving_share is not None:
        report["moving_share"] = round(moving_share, 4)
    if track_note:
        report["tracks_note"] = track_note
    if hz_info["used"]:
        hres = np.abs(BA.horizon_residual(Rs, hz_cam, hz_xy, intr, sol["up"]))
        hz_info.update({"res_px_median": round(float(np.median(hres)), 3), "res_px_p95": round(float(np.percentile(hres, 95)), 3),
                        "up": [float(v) for v in sol["up"]]})
        report["horizon_px"] = hz_info["res_px_median"]
        report["up"] = hz_info["up"]
    report["horizon_info"] = hz_info
    cam = T.Camera3D(W, H, BA.k_matrix(intr), np.array([intr[BA.K1], intr[BA.K2], intr[BA.P1], intr[BA.P2]]),
                     {frames[i]: Rs[i].copy() for i in range(n)}, {frames[i]: ts[i].copy() for i in range(n)},
                     "colmap+cotracker3" if tri["kept"] else "colmap", report)
    extra = {"points": sol["X"][:ns], "track_points": sol["X"][ns:], "hz_xy": hz_xy,
             "hz_frame": None if hz_cam is None else np.array([frames[i] for i in hz_cam], np.int32)}
    return cam, extra


def _horizon_sigma(hz_pts: Dict[int, np.ndarray]) -> float:
    """Measured sigma (px) of a horizon point: the robust scatter of each frame's points about that frame's own smooth
    curve (sh060: 0.42 px – sharper than a SIFT point, so the horizon weighs accordingly). Kept within 0.3 .. 2 px."""
    res = []
    for p in hz_pts.values():
        if len(p) >= 6:
            res.append(np.polyval(np.polyfit(p[:, 0], p[:, 1], 2), p[:, 0]) - p[:, 1])
    if not res:
        return 1.0
    r = np.concatenate(res)
    return float(np.clip(1.4826 * np.median(np.abs(r - np.median(r))), 0.3, 2.0))


def _default_tracker(side: int) -> Tuple[Any, str]:
    """CoTracker3 through the existing worker (its calls run under the GPU lease); (None, reason) when unavailable."""
    try:
        from engine.tracking import CoTracker3Tracker, default_cotracker_checkpoint
        ckpt = default_cotracker_checkpoint()
        ok, why = CoTracker3Tracker.available(ckpt)
        if not ok:
            log.warning("solve: CoTracker3 unavailable (%s) – SIFT points only", why)
            return None, why
        return CoTracker3Tracker(ckpt, max_side=side), ""
    except Exception as e:                                     # the solve still works on the SIFT points alone
        log.warning("solve: CoTracker3 unavailable (%s) – SIFT points only", e)
        return None, str(e)


def _queries(small: List[np.ndarray], bad: Dict[int, np.ndarray], tci: np.ndarray, step: int, n_q: int) -> np.ndarray:
    """Query grid [t, x, y] (pixels of the small frames) in `n_q` frames spread over the shot: outside the masks, on
    texture only (flat sky / glare gives drifting tracks)."""
    T_ = len(small)
    rows = sorted(set(int(v) for v in np.round(np.linspace(0, T_ - 1, min(n_q, T_)))))
    q = []
    for r in rows:
        g = cv2.cvtColor(small[r], cv2.COLOR_RGB2GRAY).astype(np.float32)
        tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (0, 0), 3)
        ys, xs = np.mgrid[step // 2:g.shape[0]:step, step // 2:g.shape[1]:step]
        ok = (tex[ys, xs] > 1.0) & ~bad[int(tci[r])][ys, xs]
        q.append(np.stack([np.full(int(ok.sum()), r), xs[ok], ys[ok]], 1))
    return np.concatenate(q).astype(np.float32) if q else np.zeros((0, 3), np.float32)


# ----------------------------------------------------------------------------- for the cards step (P8)
def _arrays(cam: T.Camera3D) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    fr = cam.frames
    K, d = np.asarray(cam.K, np.float64), np.asarray(cam.dist, np.float64)
    return (fr, np.array([cam.R[f] for f in fr]), np.array([cam.t[f] for f in fr]),
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3]]))


def load_points(work: Path, tracks: bool = False) -> Optional[np.ndarray]:
    """The sparse static points of the solve (n, 3) – `<work>/inpaint/camera3d/points.npy` (SIFT points: sharp, mostly
    the ground); `tracks=True` adds the CoTracker3 track points (`track_points.npy`: fewer per area but they also
    cover low-texture regions like the water). None before a solve."""
    d = T.camera_file(work).parent
    if not (d / "points.npy").is_file():
        return None
    pts = np.load(d / "points.npy").astype(np.float64)
    if tracks and (d / "track_points.npy").is_file():
        pts = np.concatenate([pts, np.load(d / "track_points.npy").astype(np.float64)])
    return pts


def load_horizon(work: Path) -> Optional[Dict[int, np.ndarray]]:
    """The horizon points the solve used: {plate frame: (n, 2) px}; None when the shot has no horizon."""
    p = T.camera_file(work).parent / "horizon.npz"
    if not p.is_file():
        return None
    with np.load(p) as d:
        fr, xy = d["frame"], d["xy"].astype(np.float64)
    return {int(f): xy[fr == f] for f in np.unique(fr)}


def horizon_up(cam: T.Camera3D, horizon_points: Dict[int, np.ndarray]) -> Optional[Dict[str, Any]]:
    """World up from the horizon: every horizon point is a direction on the level plane, up = that plane's normal
    (toward the cameras' up). `horizon_points`: {plate frame: (n, 2) px} (`load_horizon`, or `detect_horizon` per
    frame). Returns {"up": [x, y, z], "res_px_median", "res_px_p95", "points"} or None (no points)."""
    fr, Rs, _, intr = _arrays(cam)
    row = {f: i for i, f in enumerate(fr)}
    items = [(row[int(f)], np.asarray(p, np.float64).reshape(-1, 2)) for f, p in horizon_points.items() if int(f) in row]
    items = [(i, p) for i, p in items if len(p)]
    if not items:
        return None
    hc = np.concatenate([np.full(len(p), i) for i, p in items])
    hx = np.concatenate([p for _, p in items])
    up = BA.up_from_horizon(Rs, hc, hx, intr)
    r = np.abs(BA.horizon_residual(Rs, hc, hx, intr, up))
    return {"up": [float(v) for v in up], "res_px_median": round(float(np.median(r)), 3),
            "res_px_p95": round(float(np.percentile(r, 95)), 3), "points": int(len(hc))}


def sea_plane(cam: T.Camera3D, points: np.ndarray, up: Sequence[float], frame: Optional[int] = None,
              band_px: Tuple[float, float] = (5.0, 150.0)) -> Optional[Dict[str, Any]]:
    """The sea-level card: the level plane (normal = `up`) at the median height of the sparse points seen just below
    the horizon in `frame` (default: the middle one) – `band_px` below it, in px of a 3840-wide plate.
    Returns {"n": up, "d": -height, "points": k} (n . X + d = 0) or None (fewer than 5 points in the band)."""
    fr, Rs, ts, intr = _arrays(cam)
    i = fr.index(int(frame)) if frame is not None else len(fr) // 2
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    P = np.asarray(points, np.float64).reshape(-1, 3)
    Pc = P @ Rs[i].T + ts[i]
    uv = BA.project(Rs[i], ts[i], P, intr)
    d = P - (-Rs[i].T @ ts[i])
    below = np.arcsin(np.clip(-(d @ up) / np.maximum(np.linalg.norm(d, axis=1), 1e-12), -1, 1)) * float(np.sqrt(intr[0] * intr[1]))
    k = cam.width / 3840.0
    sel = (Pc[:, 2] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < cam.width) & (below > band_px[0] * k) & (below < band_px[1] * k)
    if sel.sum() < 5:
        return None
    h = float(np.median(P[sel] @ up))
    return {"n": [float(v) for v in up], "d": -h, "points": int(sel.sum())}


STATIC_FOV_DEG = 40.0      # horizontal field of view of the static camera (any value works: nothing moves)
STATIC_DEPTH = 10.0        # the „Plan” card's distance in front of the static camera


def static_camera(frames: Sequence[int], w: int, h: int, fov_deg: float = STATIC_FOV_DEG) -> T.Camera3D:
    """C2 „Kamera nieruchoma” (user 2026-11-11): a locked-off shot needs no solve – one pose for every frame (identity),
    a pinhole without distortion. A card in front of it maps a key 1:1 onto every frame, so a quick removal on a static
    shot goes straight to the keys (C4) without COLMAP."""
    import math
    f = 0.5 * float(w) / math.tan(math.radians(float(fov_deg)) / 2.0)
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    R = {int(fr): np.eye(3) for fr in frames}
    t = {int(fr): np.zeros(3) for fr in frames}
    return T.Camera3D(int(w), int(h), K, np.zeros(4), R, t, source="static",
                      report={"static": True, "fov_deg": float(fov_deg), "frames": len(R)})


def is_static(cam: Optional[T.Camera3D]) -> bool:
    return bool(cam is not None and (cam.source == "static" or (cam.report or {}).get("static")))


def propose_planes(work: Path) -> Dict[str, Any]:
    """What the cards step offers after a solve: {"ground": plane | None, "sea": plane | None, "up": [..] | None};
    a plane = {"n", "d", ...} as `Card.plane` (normal toward the cameras). Reads cams.json + points.npy; the ground
    is fitted to the SIFT points, the sea level uses the track points too (SIFT finds little on water)."""
    cam = store.load_camera(work)
    if is_static(cam):          # no points: one card facing the camera (the whole plate, 1:1 in every frame)
        return {"ground": None, "sea": None, "up": None,
                "plan": {"n": [0.0, 0.0, -1.0], "d": STATIC_DEPTH}}
    pts = load_points(work)
    if cam is None or pts is None or not cam.frames:
        return {"ground": None, "sea": None, "up": None}
    C = np.array([cam.centre(f) for f in cam.frames])
    up = cam.report.get("up")
    return {"ground": fit_plane(pts, centres=C), "up": up,
            "sea": sea_plane(cam, load_points(work, tracks=True), up) if up else None}
