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
import queue
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

from engine.exr_io import imread_unchanged

WORKER = Path(__file__).resolve().parent / "align_gpu_worker.py"   # run with -I (engine/queue.py shadows stdlib)
_CHECK_CACHE: Dict[str, Tuple[bool, str]] = {}


class GpuWorkerError(RuntimeError):
    """The GPU worker failed (did not start, crashed twice, returned an error)."""


def worker_python() -> Optional[Path]:
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


def check_worker(python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """(ok, "cuda <gpu name>" / "cpu" / the reason): can ComfyUI's Python run the alignment worker? Cached."""
    python = python if python is not None else worker_python()
    if not python or not Path(python).exists():
        return False, "nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python)"
    key = str(python) + ("|fake" if env and env.get("MD_ALIGN_GPU_FAKE") == "1" else "")
    if key not in _CHECK_CACHE:
        try:
            r = subprocess.run([str(python), "-I", str(WORKER), "--check"], capture_output=True, text=True,
                               timeout=180, env=env)
            out = (r.stdout or r.stderr or "").strip()
            if r.returncode == 0 and out.startswith("ok"):
                _CHECK_CACHE[key] = (True, out[3:].strip())
            else:
                _CHECK_CACHE[key] = (False, out[-300:] or f"kod {r.returncode}")
        except Exception as e:
            _CHECK_CACHE[key] = (False, str(e))
    return _CHECK_CACHE[key]


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


class GpuAligner:
    """One worker process in ComfyUI's Python; `call(req)` -> response dict. Restarted once if it dies."""

    def __init__(self, python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None,
                 timeout_s: float = 600.0):
        py = python if python is not None else worker_python()
        self.python = str(py) if py else ""
        self.env = env
        self.timeout_s = timeout_s
        self.device = "?"
        self.gpu_seconds = 0.0
        self.restarts = 0
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._log = None
        self._dir = tempfile.TemporaryDirectory(prefix="md_align_gpu_")
        self.tmp = Path(self._dir.name)

    def _start(self) -> None:
        if not self.python or not Path(self.python).exists():
            raise GpuWorkerError("nie znaleziono Pythona ComfyUI (Doktor → „Wyrównanie na GPU”)")
        if self._log is None:
            self._log = open(self.tmp / "worker.log", "a+", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen([self.python, "-I", str(WORKER), "--serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, env=self.env)
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lines = lines
        proc = self._proc

        def reader():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def _log_tail(self) -> str:
        try:
            self._log.flush()
            self._log.seek(0)
            return self._log.read()[-600:].strip()
        except Exception:
            return ""

    def _kill(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(timeout=10)
            except Exception:
                pass

    def _send(self, obj: Dict[str, Any]) -> None:
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read(self, cancel_check: Optional[Callable[[], None]]) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            try:
                line = self._lines.get(timeout=0.25)
            except queue.Empty:
                if cancel_check:
                    cancel_check()
                if time.time() - t0 > self.timeout_s:
                    raise GpuWorkerError(f"worker GPU nie odpowiedział w {self.timeout_s:.0f} s")
                continue
            if line is None:
                raise _WorkerDied(self._log_tail())
            return json.loads(line)

    def call(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None,
             on_sift: Optional[Callable[[Dict[str, str]], Dict[str, Any]]] = None) -> Dict[str, Any]:
        """One request; on_sift(dump paths) -> SIFT models when the worker asks for them. A dead worker is restarted
        and the request repeated once; "Przerwij" (cancel_check raising) kills the worker."""
        from engine.gpu_manager import GPU
        with GPU.use(self, "wyrównanie GPU", self._kill, cancel_check):
            for attempt in (0, 1):
                try:
                    if self._proc is None or self._proc.poll() is not None:
                        self._start()
                    self._send(req)
                    resp = self._read(cancel_check)
                    while resp.get("need_sift"):
                        models = on_sift(resp["dump"]) if on_sift else {}
                        self._send({"op": "sift_models", "models": {k: np.asarray(v).tolist() for k, v in models.items()}})
                        resp = self._read(cancel_check)
                except (_WorkerDied, BrokenPipeError, OSError) as e:
                    self._kill()
                    if attempt == 0:
                        self.restarts += 1
                        continue
                    raise GpuWorkerError(f"worker GPU zakończył się dwa razy: {e}") from None
                except BaseException:
                    self._kill()                    # cancelled / timed out: never leave a busy worker behind
                    raise
                if not resp.get("ok"):
                    raise GpuWorkerError(f"worker GPU: {resp.get('error')}")
                self.device = resp.get("device") or self.device
                self.gpu_seconds += float(resp.get("seconds") or 0.0)
                return resp
            raise GpuWorkerError("worker GPU: brak odpowiedzi")

    def close(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                self._send({"op": "quit"})
                p.wait(timeout=10)
            except Exception:
                pass
        self._kill()
        if self._log is not None:
            self._log.close()
            self._log = None
        try:
            self._dir.cleanup()
        except Exception:
            pass


class _WorkerDied(RuntimeError):
    pass


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

def global_frames(g: GpuAligner, jobs: List[Tuple[int, str, str, str, float, Any]], workers: int,
                  on_done: Optional[Callable[[int, int], None]] = None,
                  cancel_check: Optional[Callable[[], None]] = None) -> List[Any]:
    """GPU twin of run_frames(_global_frame_job, jobs): same AlignmentFrameResult per frame (None = missing file)."""
    from engine.align import AlignmentFrameResult, GlobalAligner
    from engine.procpool import imap_frames
    tmp = str(g.tmp)
    dec_jobs = [(f, b, a, scale, tmp) for f, b, a, _mode, scale, _box in jobs]
    out: List[Any] = [None] * len(jobs)
    n = 0
    for i, d in imap_frames(decode_global_job, dec_jobs, workers):
        f, _b, _a, mode, scale, box = jobs[i]
        if d is not None:
            try:
                h0, w0 = d["size"]
                h, w = int(round(h0 * scale)), int(round(w0 * scale))
                req: Dict[str, Any] = {
                    "op": "global", "mode": mode, "scale": scale, "dump_dir": tmp,
                    "lf_b": d["lf_b"], "lf_a": d["lf_a"], "lm_b": d["lm_b"], "lm_a": d["lm_a"],
                    "border": [max(4, int(round(w * 0.02))), max(4, int(round(h * 0.02)))]}
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
                _rm(d.get("lf_b"), d.get("lf_a"), d.get("lm_b"), d.get("lm_a"))
            p = r["params"]
            out[i] = AlignmentFrameResult(
                frame=f, method=r["method"], residual_px=r["med"], residual_max_px=r["max"],
                dx=p.get("dx", 0.0), dy=p.get("dy", 0.0), rotation_deg=p.get("rotation_deg", 0.0),
                scale=p.get("scale", 1.0), transform_matrix=p.get("matrix"))
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
                      "refined_frac": r["refined_frac"], "grid": grid_paths[i],
                      "n_points": int(r.get("n_points", len(pts))), "n_rejected": int(r.get("n_rejected", 0)),
                      "reject": r.get("reject") or {}}
            write_grid_npy(fp["out"], plate_w, plate_h, grid_paths[i])
            n_done += 1
            if on_done:
                on_done(n_done, i)
    return out
