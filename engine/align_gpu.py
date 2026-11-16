# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/align_gpu.py - plate alignment on the GPU: MatteDesk side of engine/align_gpu_worker.py.

MatteDesk's .venv has no PyTorch (v0.11), so the numerics run in one long-lived worker in ComfyUI's Python
(its torch + CUDA). This process and a small pool only decode EXRs into luma (+ the INTER_AREA downscales, exactly
as the CPU code does); the local model's field comes back as the alignment grid (engine.align_grid, a few
hundred KB) written here; arrays go through .npy files in a temporary folder.

config `align_device`: "auto" (default: GPU when the worker runs on CUDA and ComfyUI's queue is idle, else CPU),
"gpu" (GPU or an error), "cpu" (the CPU code of engine/align.py, unchanged). `align_gpu_decode_workers` (default 4):
decoding processes for the GPU path. A worker crash is retried once (unstable machines), then auto falls back to the
CPU for that stage; "Przerwij" kills the worker. Doctor: "Wyrównanie na GPU".
"""

from __future__ import annotations

import atexit
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

from engine.exr_io import imread_unchanged
from engine.worker_proto import CHECK_CACHE, WorkerDied, WorkerProcess, WorkerTimeout, check_worker as wp_check

WORKER = Path(__file__).resolve().parent / "align_gpu_worker.py"   # run with -I (engine/queue.py shadows stdlib)
_CHECK_CACHE = CHECK_CACHE


class GpuWorkerError(RuntimeError):
    """The GPU worker failed (did not start, crashed twice, returned an error)."""


def worker_python() -> Optional[Path]:
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


def check_worker(python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """(ok, "cuda <gpu name>" / "cpu" / the reason): can ComfyUI's Python run the alignment worker? Cached."""
    ok, out = wp_check(python if python is not None else worker_python(), WORKER, env=env)
    return ok, out[3:].strip() if ok else out


def comfy_busy(cfg: Optional[Dict[str, Any]] = None, timeout: float = 2.0) -> bool:
    """True when ComfyUI's queue has a running or pending prompt (do not share the GPU with it)."""
    try:
        if cfg is None:
            from engine.config import load_config
            cfg = load_config()
        url = f"http://{cfg.get('comfy_host') or '127.0.0.1'}:{int(cfg.get('comfy_port') or 8188)}/queue"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            q = json.loads(r.read().decode("utf-8"))
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:
        return False                     # ComfyUI not running -> nothing to share the GPU with


def decode_workers() -> int:
    try:
        from engine.config import load_config
        n = int(load_config().get("align_gpu_decode_workers") or 0)
    except Exception:
        n = 0
    return n if n > 0 else 4


class GpuAligner(WorkerProcess):
    """One worker process in ComfyUI's Python; `call(req)` -> response dict. Restarted once if it dies."""

    worker = WORKER
    label = "worker GPU"
    doctor_item = "Wyrównanie na GPU"
    tmp_prefix = "md_align_gpu_"
    error_cls = GpuWorkerError

    def __init__(self, python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None,
                 timeout_s: float = 600.0):
        super().__init__(python if python is not None else worker_python(), env, timeout_s)
        self.device = "?"
        self.gpu_seconds = 0.0
        self.tmp = self.tmp_dir()

    def call(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None,
             on_sift: Optional[Callable[[Dict[str, str]], Dict[str, Any]]] = None) -> Dict[str, Any]:
        """One request; on_sift(dump paths) -> SIFT models when the worker asks for them. A dead worker is restarted
        and the request repeated once; "Przerwij" (cancel_check raising) kills the worker."""
        from engine.gpu_manager import GPU

        def on_reply(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not resp.get("need_sift"):
                return None
            models = on_sift(resp["dump"]) if on_sift else {}
            return {"op": "sift_models", "models": {k: np.asarray(v).tolist() for k, v in models.items()}}

        with GPU.use(self, "wyrównanie GPU", self.kill, cancel_check):
            try:
                resp = self.request(req, cancel_check, restarts=1, on_reply=on_reply)
            except WorkerDied as e:
                raise GpuWorkerError(f"worker GPU zakończył się dwa razy: {e}") from None
            except WorkerTimeout as e:
                raise GpuWorkerError(str(e)) from None
            if not resp.get("ok"):
                raise GpuWorkerError(f"worker GPU: {resp.get('error')}")
            self.device = resp.get("device") or self.device
            self.gpu_seconds += float(resp.get("seconds") or 0.0)
            return resp


_OPEN: List[GpuAligner] = []


@atexit.register
def _close_all() -> None:
    for g in list(_OPEN):
        g.close()


def resolve_device(requested: Optional[str] = None, python: Optional[Union[str, Path]] = None,
                   env: Optional[Dict[str, str]] = None, busy: Optional[Callable[[], bool]] = None,
                   key: str = "align_device") -> Tuple[str, str]:
    """('gpu' | 'cpu', note). requested None -> config `key` (`align_device`; `mb_device` for the motion blur of the
    alpha) = auto | cpu | gpu. 'gpu' that cannot run raises GpuWorkerError; 'auto' falls back to the CPU with the
    reason in the note."""
    if requested is None:
        try:
            from engine.config import load_config
            requested = str(load_config().get(key) or "auto")
        except Exception:
            requested = "auto"
    requested = requested.lower()
    if requested == "cpu":
        return "cpu", ""
    ok, info = check_worker(python, env)
    if requested == "gpu":
        if not ok:
            raise GpuWorkerError(f"wyrównanie na GPU niedostępne: {info} (Doktor → „Wyrównanie na GPU”)")
        return "gpu", info
    if not ok:
        return "cpu", f"GPU niedostępne: {info}"
    if not (info.startswith("cuda") or info.startswith("fake")):
        return "cpu", f"Python ComfyUI bez CUDA ({info})"
    if (busy or comfy_busy)():
        return "cpu", "ComfyUI liczy (kolejka /queue) – wyrównanie na CPU"
    return "gpu", info


def open_aligner(python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None) -> GpuAligner:
    g = GpuAligner(python, env)
    _OPEN.append(g)
    return g


def close_aligner(g: Optional[GpuAligner]) -> None:
    if g is not None:
        g.close()
        from engine.gpu_manager import GPU
        GPU.forget(g)
        if g in _OPEN:
            _OPEN.remove(g)


# ------------------------------------------------------------------------------------------- pool-side jobs

def _luma(img: np.ndarray) -> np.ndarray:
    """The luma AlignmentFeatureExtractor / compute_static_mask build (mean of the channels, float32)."""
    return img.mean(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def decode_global_job(job: Tuple[int, str, str, float, str]) -> Optional[Dict[str, Any]]:
    """EXR pair -> half-size lumas: `lf` = features' resize (fx = scale), `lm` = static mask's resize (dsize)."""
    f, b_path, a_path, scale, tmp = job
    if not os.path.exists(b_path) or not os.path.exists(a_path):
        return None
    out: Dict[str, Any] = {"frame": f}
    for tag, pth in (("b", b_path), ("a", a_path)):
        luma = _luma(imread_unchanged(pth))
        h, w = luma.shape
        if tag == "b":
            out["size"] = (h, w)       # compute_static_mask sizes everything from the beauty frame
        lf = luma if scale == 1.0 else cv2.resize(luma, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        lm = cv2.resize(luma, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        for kind, arr in (("lf", lf), ("lm", lm)):
            p = os.path.join(tmp, f"g{f}_{kind}_{tag}.npy")
            np.save(p, np.ascontiguousarray(arr, np.float32))
            out[f"{kind}_{tag}"] = p
    return out


def decode_local_job(job: Tuple[int, str, str, str, Any]) -> Dict[str, Any]:
    """EXR pair -> full-resolution lumas (local model refinement); the aux luma warped by the frame's global model
    `m` with the CPU code's own function (engine.align.aux_luma_global) - the same input as _local_frame_job."""
    from engine.align import aux_luma_global
    i, b_path, a_path, tmp, m = job
    out: Dict[str, Any] = {"i": i}
    pb = os.path.join(tmp, f"l{i}_b.npy")
    np.save(pb, _luma(imread_unchanged(b_path)))
    pa = os.path.join(tmp, f"l{i}_a.npy")
    np.save(pa, aux_luma_global(a_path, m))
    out["luma_b"], out["luma_a"] = pb, pa
    return out


def write_grid_npy(grid_npy: str, plate_w: int, plate_h: int, grid_path: str) -> str:
    """(2, gh, gw) grid .npy from the worker -> grid EXR (engine.align_grid); the .npy is removed."""
    from engine import align_grid
    g = np.load(grid_npy)
    align_grid.write_grid(Path(grid_path), align_grid.AlignGrid(g[0], g[1], plate_w, plate_h))
    os.remove(grid_npy)
    return grid_path


def _rm(*paths: Optional[str]) -> None:
    for p in paths:
        if p:
            try:
                os.remove(p)
            except OSError:
                pass


# ------------------------------------------------------------------------------------------- stage drivers

def global_frames(g: GpuAligner, jobs: List[Tuple[Any, ...]], workers: int,
                  on_done: Optional[Callable[[int, int], None]] = None,
                  cancel_check: Optional[Callable[[], None]] = None) -> List[Any]:
    """GPU twin of run_frames(_global_frame_job, jobs): same AlignmentFrameResult per frame (None = missing file).
    A job's 7th element (the frame's exclusion spec, engine.align._job_exclusion) becomes the worker's `exclude`
    mask (uint8 at the work scale, nearest like compute_static_mask)."""
    from engine.align import GlobalAligner, frame_exclusion_of, global_frame_result
    from engine.procpool import imap_frames
    tmp = str(g.tmp)
    dec_jobs = [(j[0], j[1], j[2], j[4], tmp) for j in jobs]
    out: List[Any] = [None] * len(jobs)
    n = 0
    for i, d in imap_frames(decode_global_job, dec_jobs, workers):
        f, _b, _a, mode, scale, box = jobs[i][:6]
        excl = jobs[i][6] if len(jobs[i]) > 6 else None
        if d is not None:
            ex_path = None
            try:
                h0, w0 = d["size"]
                h, w = int(round(h0 * scale)), int(round(w0 * scale))
                req: Dict[str, Any] = {
                    "op": "global", "mode": mode, "scale": scale, "dump_dir": tmp,
                    "lf_b": d["lf_b"], "lf_a": d["lf_a"], "lm_b": d["lm_b"], "lm_a": d["lm_a"],
                    "border": [max(4, int(round(w * 0.02))), max(4, int(round(h * 0.02)))]}
                ex = frame_exclusion_of(excl, f, w0, h0)
                if ex is not None:
                    ex_path = os.path.join(tmp, f"g{f}_ex.npy")
                    np.save(ex_path, cv2.resize(ex.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST))
                    req["exclude"] = ex_path
                if box is not None:
                    bx0, by0, bx1, by1 = box
                    req["box"] = [max(0, int(round(bx0 * scale))), max(0, int(round(by0 * scale))),
                                  min(w, int(round(bx1 * scale))), min(h, int(round(by1 * scale)))]
                if mode == "cleanplate":
                    k = max(5, int(round(40 * scale)))
                    req["kernel"] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(int).tolist()

                def on_sift(dump: Dict[str, str], scale=scale) -> Dict[str, Any]:
                    fa, fb, m = (np.load(dump[k]) for k in ("feat_a", "feat_b", "mask"))
                    _rm(*dump.values())
                    return GlobalAligner(scale=scale).fit_sift_models(fa, fb, m)

                r = g.call(req, cancel_check, on_sift)
            finally:
                _rm(d.get("lf_b"), d.get("lf_a"), d.get("lm_b"), d.get("lm_a"), ex_path)
            out[i] = global_frame_result(f, r["method"], r["params"], r.get("med"), r.get("max"))
        n += 1
        if on_done:
            on_done(n, i)
    return out


def make_patch_measure(g: GpuAligner, cancel_check: Optional[Callable[[], None]] = None) -> Callable[..., Any]:
    """GPU twin of LocalPatchAligner.measure_local_displacements (anchor offsets of LocalTemporalAligner)."""

    def measure(beauty: np.ndarray, aux: np.ndarray, grid_pts: np.ndarray, patch_size: int = 128,
                mode: str = "backlit") -> Tuple[np.ndarray, np.ndarray]:
        h, w = beauty.shape[:2]
        half = patch_size // 2
        xy = [(int(round(p[0])), int(round(p[1]))) for p in grid_pts]
        xy = np.array([q for q in xy if half <= q[0] < w - half and half <= q[1] < h - half], np.int64).reshape(-1, 2)
        tmp = g.tmp
        paths = {k: str(tmp / f"anchor_{k}.npy") for k in ("luma_b", "luma_a", "xy", "d0", "out")}
        np.save(paths["luma_b"], _luma(beauty))
        np.save(paths["luma_a"], _luma(aux))
        np.save(paths["xy"], xy)
        np.save(paths["d0"], np.zeros((len(xy), 2)))
        try:
            g.call({"op": "patches", "mode": mode, "patch": patch_size, **paths}, cancel_check)
            res = np.load(paths["out"]) if len(xy) else np.zeros((0, 5))
        finally:
            _rm(*paths.values())
        valid_pts, disps = [], []
        for (x, y), (rx, ry, resp, gx, gy) in zip(xy, res):
            dx, dy = float(rx), float(ry)
            if not (np.isnan(dx) or np.isnan(dy)) and abs(dx) <= 8 and abs(dy) <= 8:
                dx, dy = float(gx), float(gy)
            if not np.isnan(dx) and not np.isnan(dy) and resp > 0.1:
                valid_pts.append([x, y])
                disps.append([float(np.clip(dx, -8.0, 8.0)), float(np.clip(dy, -8.0, 8.0))])
        if len(valid_pts) < 4:
            return grid_pts, np.zeros_like(grid_pts)
        return np.array(valid_pts, dtype=np.float32), np.array(disps, dtype=np.float32)

    return measure


def make_median(g: GpuAligner, cancel_check: Optional[Callable[[], None]] = None) -> Callable[[np.ndarray], np.ndarray]:
    """GPU twin of the per-point median over the anchors (engine.align.median_anchor_field `median_fn`): (A, N, 3)
    with NaN for an anchor that did not measure the point -> (N, 3) = np.nanmedian(axis=0)."""

    def median(stack: np.ndarray) -> np.ndarray:
        paths = {k: str(g.tmp / f"median_{k}.npy") for k in ("vals", "out")}
        np.save(paths["vals"], np.asarray(stack, np.float64))
        try:
            g.call({"op": "median_disp", **paths}, cancel_check)
            return np.load(paths["out"])
        finally:
            _rm(*paths.values())

    return median


def anchor_refine(g: GpuAligner, paths: List[Tuple[str, str]], fields: List[Tuple[np.ndarray, np.ndarray]], mode: str,
                  mats: List[np.ndarray], workers: int, cancel_check: Optional[Callable[[], None]] = None,
                  patch: int = 128, max_change_px: float = 4.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GPU twin of engine.align._anchor_refine_job for the locked-camera path (align_v2 step 2): per anchor the
    full-res lumas (aux pre-warped by its global model, the decode pool) and the worker's `patches` op started from
    the anchor's offsets (= refine_local_offsets: phase correlation + Gauss-Newton; kept where the response is weak or
    the change exceeds max_change_px). Returns per anchor (disp refined (N, 2) float32, response (N,) float64)."""
    from engine.procpool import imap_frames, open_pool
    tmp = str(g.tmp)
    out: List[Tuple[np.ndarray, np.ndarray]] = [(np.asarray(d, np.float32).reshape(-1, 2), np.full(len(p), np.nan))
                                                for p, d in fields]
    todo = [i for i, (p, _d) in enumerate(fields) if len(p)]
    with open_pool(workers) as pool:
        for k, d in imap_frames(decode_local_job, [(i, *paths[i], tmp, np.asarray(mats[i], np.float64).tolist())
                                                   for i in todo], workers, pool=pool):
            i = todo[k]
            pts, disp = fields[i]
            disp = np.asarray(disp, np.float32).reshape(-1, 2).copy()
            lb = np.load(d["luma_b"], mmap_mode="r")
            h, w = lb.shape
            del lb
            half = patch // 2
            xy = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in pts], np.int64).reshape(-1, 2)
            inside = (xy[:, 0] >= half) & (xy[:, 0] < w - half) & (xy[:, 1] >= half) & (xy[:, 1] < h - half)
            ii = np.nonzero(inside)[0]
            response = np.full(len(disp), np.nan, np.float64)
            fp = {k2: os.path.join(tmp, f"a{i}_{k2}.npy") for k2 in ("xy", "d0", "out")}
            try:
                if len(ii):
                    np.save(fp["xy"], xy[ii])
                    np.save(fp["d0"], disp[ii].astype(np.float64))
                    g.call({"op": "patches", "mode": mode, "patch": int(patch), "luma_b": d["luma_b"],
                            "luma_a": d["luma_a"], "xy": fp["xy"], "d0": fp["d0"], "out": fp["out"]}, cancel_check)
                    res = np.load(fp["out"])
                    for k3, (rx, ry, resp, gx, gy) in zip(ii, res):
                        response[k3] = resp
                        if not (resp > 0.1) or not (np.isfinite(rx) and np.isfinite(ry)):
                            continue
                        d0 = disp[k3]
                        if not (np.isfinite(gx) and np.isfinite(gy)) or float(np.hypot(gx - d0[0], gy - d0[1])) > max_change_px:
                            continue
                        disp[k3] = (np.clip(gx, -8.0, 8.0), np.clip(gy, -8.0, 8.0))
            finally:
                _rm(d["luma_b"], d["luma_a"], *fp.values())
            out[i] = (disp, response)
    return out


def local_frames(g: GpuAligner, paths: List[Tuple[str, str]], fields: List[Tuple[np.ndarray, np.ndarray]],
                 mode: str, plate_w: int, plate_h: int, grid_paths: List[str], workers: int,
                 on_done: Optional[Callable[[int, int], None]] = None,
                 cancel_check: Optional[Callable[[], None]] = None,
                 mats: Optional[List[np.ndarray]] = None, patch: int = 128) -> List[Optional[Dict[str, Any]]]:
    """GPU twin of run_frames(_local_frame_job, ...): field on the GPU, grid EXRs written here (small).
    `mats`: per frame the global model (3x3) the aux is pre-warped by (identity when None); `patch` = LOCAL_PATCH_PX."""
    from engine.procpool import imap_frames, open_pool
    tmp = str(g.tmp)
    if mats is None:
        mats = [np.eye(3)] * len(paths)
    todo = [i for i, (pts, _d) in enumerate(fields) if len(pts) >= 8]
    todo_set = set(todo)
    out: List[Optional[Dict[str, Any]]] = [None] * len(paths)
    n_done = 0
    for i in range(len(paths)):
        if i not in todo_set:
            n_done += 1
            if on_done:
                on_done(n_done, i)
    with open_pool(workers) as pool:
        for k, d in imap_frames(decode_local_job, [(i, *paths[i], tmp, np.asarray(mats[i], np.float64).tolist())
                                                   for i in todo], workers, pool=pool):
            i = todo[k]
            pts, disp = fields[i]
            fp = {"pts": os.path.join(tmp, f"l{i}_pts.npy"), "disp": os.path.join(tmp, f"l{i}_disp.npy"),
                  "out": os.path.join(tmp, f"l{i}_grid.npy")}
            np.save(fp["pts"], np.asarray(pts, np.float32))
            np.save(fp["disp"], np.asarray(disp, np.float32))
            try:
                r = g.call({"op": "local_frame", "mode": mode, "plate_w": plate_w, "plate_h": plate_h, "patch": int(patch),
                            "luma_b": d["luma_b"], "luma_a": d["luma_a"], **fp}, cancel_check)
            finally:
                _rm(d["luma_b"], d["luma_a"], fp["pts"], fp["disp"])
            out[i] = {"ho_med": r["ho_med"], "ho_max": r["ho_max"], "ho_frac_ok": r["ho_frac_ok"],
                      "fitted": bool(r.get("fitted", True)),
                      "refined_frac": r["refined_frac"], "grid": grid_paths[i],
                      "n_points": int(r.get("n_points", len(pts))), "n_rejected": int(r.get("n_rejected", 0)),
                      "reject": r.get("reject") or {}}
            write_grid_npy(fp["out"], plate_w, plate_h, grid_paths[i])
            n_done += 1
            if on_done:
                on_done(n_done, i)
    return out
