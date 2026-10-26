# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/tracking.py - Long-term point tracking (SPEC 1b, 4d, 7.2 pkt 7, 7.7 pkt 3).

Engines (the name recorded in reports is always the engine that actually ran):
- `cotracker3`  – CoTracker3 offline (`cotracker/scaled_offline.pth` in the Comfy models dir), run in ComfyUI's
                  Python (its PyTorch + GPU) by engine/cotracker_worker.py – MatteDesk's .venv has no PyTorch.
                  Needs the `cotracker` package in ComfyUI's Python (Doctor: "Zainstaluj" at CoTracker3).
- `lk_fallback` – OpenCV pyramidal Lucas-Kanade chained frame-to-frame with forward-backward check.
                  Technical fallback only (user rule: long tracks = CoTracker3, not KLT); reports say so.

All coordinates are pixels of the frames passed in (x right, y down, pixel centre = +0.5 is NOT added:
points are continuous image coordinates as used by OpenCV).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("matte_desk.tracking")

COTRACKER_REL_PATH = "cotracker/scaled_offline.pth"


class TrackerUnavailable(RuntimeError):
    pass


@dataclass
class TrackResult:
    tracks: np.ndarray    # (T, N, 2) float32, positions per frame
    visible: np.ndarray   # (T, N) bool
    engine: str


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.dtype != np.uint8:
        f = img.astype(np.float32)
        hi = float(np.percentile(f, 99.5)) or 1.0
        img = np.clip(f / hi * 255.0, 0, 255).astype(np.uint8)
    return img


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2 or img.shape[2] == 1:
        g = _to_gray_u8(img)
        return np.repeat(g[..., None], 3, axis=2)
    if img.dtype == np.uint8:
        return img[..., :3]
    f = img[..., :3].astype(np.float32)
    hi = float(np.percentile(f, 99.5)) or 1.0
    return np.clip(f / hi * 255.0, 0, 255).astype(np.uint8)


def default_cotracker_checkpoint() -> Optional[Path]:
    """Checkpoint path from the tool config (`models_dir`), None if not configured."""
    from engine.config import load_config
    models_dir = load_config().get("models_dir", "")
    return Path(models_dir) / COTRACKER_REL_PATH if models_dir else None


# run with -I: the worker sits in engine/, where engine/queue.py would shadow the stdlib `queue`
WORKER = Path(__file__).resolve().parent / "cotracker_worker.py"
_CHECK_CACHE: dict = {}


class _CoTrackerWorker:
    """One long-lived `cotracker_worker.py --serve` process per (python, checkpoint): the model is loaded once, every
    `request` is one in.npz -> out.npz round trip over stdin / stdout (2026-10-25; before, every tracking started a
    new process: torch + CUDA + weights, ~10-20 s, for a single band-point edit). Restarted when it died; a cancel
    kills it (the next request starts a fresh one). The maths in the worker is the one-shot code: same tracks."""

    def __init__(self, python: str, env: Optional[Dict[str, str]] = None):
        self.python = python
        self.env = env
        self.device: Optional[str] = None
        self.load_seconds = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._log = None
        self._dir: Optional[tempfile.TemporaryDirectory] = None
        self.lock = threading.Lock()         # one request at a time (the GPU is one; callers queue here)
        self.busy = False

    def _start(self) -> None:
        if self._dir is None:
            # ignore_cleanup_errors: on Windows a worker killed a moment ago can still hold worker.log (its stderr)
            # when the directory goes – that must not raise at interpreter exit (testmon run 2026-10-25)
            self._dir = tempfile.TemporaryDirectory(prefix="md_cotracker_srv_", ignore_cleanup_errors=True)
        if self._log is None:
            self._log = open(Path(self._dir.name) / "worker.log", "a+", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen([self.python, "-I", str(WORKER), "--serve"], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, env=self.env)

    def _log_tail(self) -> str:
        try:
            self._log.flush()
            self._log.seek(0)
            return self._log.read()[-600:].strip()
        except Exception:
            return ""

    def request(self, src: Path, dst: Path, cancel_check: Optional[Callable[[], None]] = None,
                timeout_s: float = 3600.0) -> Dict[str, Any]:
        """Run one in.npz -> out.npz tracking; `cancel_check()` is polled every second and, if it raises, the worker
        is killed (a running CoTracker3 forward cannot be interrupted any other way) and the exception propagates."""
        with self.lock:
            self.busy = True
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._start()
                result: Dict[str, Any] = {}
                done = threading.Event()

                def read():
                    try:
                        result["line"] = self._proc.stdout.readline()
                    finally:
                        done.set()
                self._proc.stdin.write(json.dumps({"in": str(src), "out": str(dst)}) + "\n")
                self._proc.stdin.flush()
                th = threading.Thread(target=read, daemon=True)
                th.start()
                t0 = time.time()
                while not done.wait(1.0):
                    try:
                        if cancel_check:
                            cancel_check()
                        if time.time() - t0 > timeout_s:
                            raise RuntimeError(f"CoTracker3 (Python ComfyUI): przekroczony czas {timeout_s:.0f} s")
                    except BaseException:
                        self.close()
                        raise
                line = result.get("line")
                if not line:
                    alive = self._proc.poll() is None
                    tail = self._log_tail()
                    self.close()
                    raise RuntimeError("CoTracker3 (Python ComfyUI) " + ("nie odpowiedział" if alive else "zakończył się")
                                       + f": {tail}")
                resp = json.loads(line)
                if not resp.get("ok"):
                    raise RuntimeError(f"CoTracker3 (Python ComfyUI) nie policzył: {resp.get('error')}")
                self.device = resp.get("device") or self.device
                self.load_seconds = float(resp.get("load_seconds") or self.load_seconds)
                return resp
            finally:
                self.busy = False

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=5)
            except Exception:
                p.kill()
                p.wait()
        log, self._log = self._log, None
        if log is not None:
            log.close()
        d, self._dir = self._dir, None
        if d is not None:
            try:
                d.cleanup()
            except OSError:
                pass


_WORKERS: Dict[str, _CoTrackerWorker] = {}
_WORKERS_LOCK = threading.Lock()


def cotracker_worker(python: Union[str, Path], env: Optional[Dict[str, str]] = None) -> _CoTrackerWorker:
    """The process-wide CoTracker3 worker of this interpreter (tests pass their own env with MD_COTRACKER_FAKE=1)."""
    key = str(python) + "|" + (json.dumps(sorted((env or {}).items())) if env else "")
    with _WORKERS_LOCK:
        w = _WORKERS.get(key)
        if w is None:
            w = _WORKERS[key] = _CoTrackerWorker(str(python), env)
    return w


def close_workers(idle_only: bool = False) -> None:
    """Stop the served workers (frees their GPU memory; the next request starts a new one). `idle_only`
    (engine/gpu_release before a Comfy stage): a worker answering a request right now is left running."""
    with _WORKERS_LOCK:
        ws = [(k, w) for k, w in _WORKERS.items() if not (idle_only and w.busy)]
        for k, _ in ws:
            del _WORKERS[k]
    for _, w in ws:
        w.close()


atexit.register(close_workers)


def tracker_python() -> Optional[Path]:
    """ComfyUI's Python (config `comfy_python`, else Desktop .venv / portable embedded) – it has PyTorch + CUDA."""
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


def check_tracker_python(python: Optional[Union[str, Path]], env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """(ok, "cuda"/"cpu" or the reason): can ComfyUI's Python import torch + cotracker? Cached per interpreter."""
    if not python or not Path(python).exists():
        return False, "nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python)"
    key = str(python) + ("|fake" if (env or os.environ).get("MD_COTRACKER_FAKE") == "1" else "")
    if key not in _CHECK_CACHE:
        try:
            r = subprocess.run([str(python), "-I", str(WORKER), "--check"], capture_output=True, text=True, timeout=120,
                               env=env)
            out = (r.stdout or r.stderr or "").strip()
            _CHECK_CACHE[key] = (r.returncode == 0, out.split()[-1] if r.returncode == 0 and out else out[-300:])
        except Exception as e:
            _CHECK_CACHE[key] = (False, str(e))
    return _CHECK_CACHE[key]


class CoTracker3Tracker:
    """CoTracker3 offline in ComfyUI's Python (separate process, its GPU). Video is downscaled so the longer side is
    <= max_side (tracks rescaled back); frames go through a temporary .npz."""

    engine = "cotracker3"

    def __init__(self, checkpoint: Union[str, Path], python: Optional[Union[str, Path]] = None, max_side: int = 1024,
                 timeout_s: float = 3600.0, max_queries: int = 2048, env: Optional[Dict[str, str]] = None,
                 persistent: bool = True):
        self.python = Path(python) if python else tracker_python()
        self.env = env
        self.persistent = persistent        # False = one process per call (the pre-2026-10-25 path, kept for A/B)
        ok, why = self.available(checkpoint, self.python, env)
        if not ok:
            raise TrackerUnavailable(why)
        self.checkpoint = str(checkpoint)
        self.device = why            # "cuda" / "cpu" from the check
        self.max_side = max_side
        self.timeout_s = timeout_s
        # queries per model call (SH010 backlit, ~49 fr at 1024 px: 4096 -> 17.2 GiB peak, card full next to the GPU
        # alignment worker); the worker halves a call again on CUDA OOM (each query's track does not depend
        # on the other queries in the call, apart from CoTracker3's joint attention - small effect)
        self.max_queries = max_queries
        self.last_peak_gib: Optional[float] = None   # peak GPU memory of the last track_many (worker report)

    @staticmethod
    def available(checkpoint: Optional[Union[str, Path]], python: Optional[Union[str, Path]] = None,
                  env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        if not checkpoint or not Path(checkpoint).exists():
            return False, f"brak wag CoTracker3 ({checkpoint})"
        ok, why = check_tracker_python(python if python is not None else tracker_python(), env)
        if not ok:
            return False, f"CoTracker3 niedostępny w Pythonie ComfyUI: {why}"
        return True, why

    def track(self, frames: Sequence[np.ndarray], queries: np.ndarray,
              cancel_check: Optional[Callable[[], None]] = None) -> TrackResult:
        """queries: (N, 3) rows [t_index, x, y]."""
        return self.track_many([(frames, queries)], cancel_check)[0]

    def _one_shot(self, src: Path, dst: Path, cancel_check: Optional[Callable[[], None]] = None) -> None:
        """A process per call (the pre-2026-10-25 path): kept for the bit-identity test against the served worker."""
        proc = subprocess.Popen([str(self.python), "-I", str(WORKER), str(src), str(dst)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
        t0 = time.time()
        try:
            while True:
                try:
                    out, err = proc.communicate(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_check:
                        cancel_check()
                    if time.time() - t0 > self.timeout_s:
                        raise RuntimeError(f"CoTracker3 (Python ComfyUI): przekroczony czas {self.timeout_s:.0f} s")
        except BaseException:
            proc.kill()
            proc.communicate()
            raise
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(f"CoTracker3 (Python ComfyUI) nie policzył: {(err or out or '').strip()[-600:]}")

    def track_many(self, videos: Sequence[Tuple[Sequence[np.ndarray], np.ndarray]],
                   cancel_check: Optional[Callable[[], None]] = None) -> List[TrackResult]:
        """Several (frames, queries) videos in one worker process – the model is loaded once.
        Frames shared between videos (the same array objects, e.g. overlapping slices of one list) are downscaled
        and sent once (frame pool). cancel_check() is polled every second while the worker runs; if it raises,
        the worker is killed."""
        data: Dict[str, Any] = {"checkpoint": np.array(self.checkpoint), "n_videos": np.array(len(videos)),
                                "max_queries": np.array(int(self.max_queries))}
        pool: List[np.ndarray] = []
        pool_ids: Dict[Tuple[int, int], int] = {}     # (id(frame), scaled width) -> pool index
        scales = []
        for i, (frames, queries) in enumerate(videos):
            h, w = frames[0].shape[:2]
            s = min(1.0, self.max_side / float(max(h, w)))
            sw, sh = round(w * s), round(h * s)
            idx = []
            for f in frames:
                key = (id(f), sw)
                if key not in pool_ids:
                    pool_ids[key] = len(pool)
                    pool.append(cv2.resize(_to_rgb_u8(f), (sw, sh), interpolation=cv2.INTER_AREA) if s < 1
                                else _to_rgb_u8(f))
                idx.append(pool_ids[key])
            q = np.asarray(queries, dtype=np.float32).copy()
            q[:, 1:] *= s
            data[f"frames_idx_{i}"] = np.array(idx, np.int32)
            data[f"queries_{i}"] = q
            scales.append(s)
        shapes = sorted({f.shape for f in pool})
        for k, shp in enumerate(shapes):    # one stacked pool per frame size
            sel = [j for j, f in enumerate(pool) if f.shape == shp]
            data[f"pool_{k}"] = np.ascontiguousarray(np.stack([pool[j] for j in sel]))
            data[f"pool_members_{k}"] = np.array(sel, np.int32)
        data["n_pools"] = np.array(len(shapes))
        del pool
        with tempfile.TemporaryDirectory(prefix="md_cotracker_") as td:
            src, dst = Path(td) / "in.npz", Path(td) / "out.npz"
            np.savez(src, **data)
            del data
            t0 = time.time()
            if self.persistent:
                cotracker_worker(self.python, self.env).request(src, dst, cancel_check, self.timeout_s)
            else:
                self._one_shot(src, dst, cancel_check)
            if not dst.exists():
                raise RuntimeError("CoTracker3 (Python ComfyUI) nie zapisał wyniku")
            res = []
            with np.load(dst, allow_pickle=False) as d:
                self.last_peak_gib = float(d["peak_gib"]) if "peak_gib" in d.files else None
                for i, s in enumerate(scales):
                    res.append(TrackResult(tracks=d[f"tracks_{i}"].astype(np.float32) / s,
                                           visible=d[f"visible_{i}"].astype(bool), engine=self.engine))
            logger.info("CoTracker3: %d wideo, %.1f s, szczyt GPU %s GiB", len(videos), time.time() - t0,
                        f"{self.last_peak_gib:.2f}" if self.last_peak_gib is not None else "?")
        return res


class LKTracker:
    """Fallback: pyramidal LK chained from each query frame forwards and backwards, fwd-bwd checked."""

    engine = "lk_fallback"

    def __init__(self, win: int = 21, levels: int = 4, fb_max_px: float = 0.5):
        self.win = win
        self.levels = levels
        self.fb_max = fb_max_px

    def _step(self, g0: np.ndarray, g1: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lk = dict(winSize=(self.win, self.win), maxLevel=self.levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.003))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p.reshape(-1, 1, 2), None, **lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
        fb = np.linalg.norm(p0b.reshape(-1, 2) - p.reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_max)
        return p1.reshape(-1, 2), ok

    def track(self, frames: Sequence[np.ndarray], queries: np.ndarray) -> TrackResult:
        grays = [_to_gray_u8(f) for f in frames]
        T, N = len(grays), len(queries)
        tracks = np.zeros((T, N, 2), np.float32)
        visible = np.zeros((T, N), bool)
        q = np.asarray(queries, dtype=np.float32)
        for t0 in np.unique(q[:, 0].astype(int)):
            idx = np.where(q[:, 0].astype(int) == t0)[0]
            start = q[idx, 1:].copy()
            tracks[t0, idx] = start
            visible[t0, idx] = True
            for direction in (1, -1):
                p, alive = start.copy(), np.ones(len(idx), bool)
                t = t0
                while 0 <= t + direction < T and alive.any():
                    p_new, ok = self._step(grays[t], grays[t + direction], p)
                    alive &= ok
                    p = np.where(alive[:, None], p_new, p)
                    t += direction
                    tracks[t, idx] = p
                    visible[t, idx] = alive
        return TrackResult(tracks=tracks, visible=visible, engine=self.engine)


def track_many(tracker: Any, videos: Sequence[Tuple[Sequence[np.ndarray], np.ndarray]],
               cancel_check: Optional[Callable[[], None]] = None) -> List[TrackResult]:
    """tracker.track_many when the engine has it (CoTracker3: one process, model loaded once), else one by one."""
    if hasattr(tracker, "track_many"):
        return tracker.track_many(videos, cancel_check)
    out = []
    for frames, queries in videos:
        if cancel_check:
            cancel_check()
        out.append(tracker.track(frames, queries))
    return out


def make_point_tracker(checkpoint: Optional[Union[str, Path]] = None, prefer: str = "cotracker3"):
    """CoTracker3 when available, otherwise the LK fallback (with a logged reason)."""
    if prefer == "cotracker3":
        ckpt = checkpoint or default_cotracker_checkpoint()
        ok, why = CoTracker3Tracker.available(ckpt)
        if ok:
            return CoTracker3Tracker(ckpt)
        logger.warning("CoTracker3 unavailable (%s) - falling back to LK", why)
    return LKTracker()
