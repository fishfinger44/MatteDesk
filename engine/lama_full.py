# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""LaMa at the crop's full resolution (user 2026-11-10): the client of engine/lama_worker.py, a worker in ComfyUI's
Python holding big-lama.pt on the GPU. Every call runs under the GPU manager's lease (`GPU.use`, LEASED_CALLS).
The ComfyUI node path (engine.cleanup.lama_batch_comfy, 256 x 256) stays as the fallback of the callers.

    from engine import lama_full
    if lama_full.available(cfg): out = lama_full.inpaint(crop_bgr, mask, cfg)     # float32 BGR 0..1
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from engine.worker_proto import WorkerDied, WorkerProcess, WorkerTimeout

WORKER = Path(__file__).resolve().with_name("lama_worker.py")
MAX_SIDE = 2048              # a larger crop goes through the model at this side (2048^2 = ~3.3 GB on the GPU) and back
HOLE_MAX = 480               # big-lama works on holes up to ~480 px long (sh060 2026-11-10: 360 / 480 px clean, 600 px
                             # starts a grid, 950 px = a grey blot) - a crop with a longer hole runs scaled down to it


def scale_for(mask: np.ndarray) -> float:
    """The scale (<= 1, never up) the crop goes through LaMa at: its long side <= MAX_SIDE and the hole's bounding box
    long side <= HOLE_MAX."""
    m = np.asarray(mask) > 0
    h, w = m.shape
    s = min(1.0, MAX_SIDE / float(max(h, w)))
    if m.any():
        ys, xs = np.nonzero(m)
        hole = float(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
        s = min(s, HOLE_MAX / hole)
    return s


class LamaFullError(RuntimeError):
    pass


def model_path(cfg: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    from engine.stage_models import _md
    if cfg is None:
        from engine.config import load_config
        cfg = load_config()
    md = _md(cfg)
    p = Path(md) / "inpaint" / "big-lama.pt" if md else None
    return p if p is not None and p.is_file() else None


def worker_python() -> Optional[Path]:
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


class LamaWorker(WorkerProcess):
    """One worker process (engine/lama_worker.py) in ComfyUI's Python; `call(req)` -> answer dict."""

    worker = WORKER
    label = "worker LaMa"
    doctor_item = "Wyrównanie na GPU"          # same requirement (ComfyUI's Python with torch + CUDA)
    tmp_prefix = "md_lama_full_"
    error_cls = LamaFullError

    def __init__(self, python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None,
                 timeout_s: float = 600.0):
        super().__init__(python if python is not None else worker_python(), env, timeout_s)
        self.device = "?"
        self.tmp = self.tmp_dir()

    def call(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        from engine.gpu_manager import GPU
        with GPU.use(self, "LaMa (pełna rozdzielczość)", self.kill, cancel_check):
            try:
                resp = self.request(req, cancel_check, restarts=1)
            except WorkerDied as e:
                raise LamaFullError(f"worker LaMa zakończył się dwa razy: {e}") from None
            except WorkerTimeout as e:
                raise LamaFullError(str(e)) from None
            if not resp.get("ok"):
                raise LamaFullError(f"worker LaMa: {resp.get('error')}")
            self.device = resp.get("device") or self.device
            return resp


_LOCK = threading.Lock()
_WORKER: Dict[str, Optional[LamaWorker]] = {"w": None}


def _worker() -> LamaWorker:
    if _WORKER["w"] is None:
        _WORKER["w"] = LamaWorker()
    return _WORKER["w"]


def available(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """big-lama.pt on disk and ComfyUI's Python found (a CUDA check happens on the first call)."""
    if os.environ.get("MD_LAMA_FULL", "1") == "0":
        return False
    py = worker_python()
    return model_path(cfg) is not None and py is not None and Path(py).exists()


def inpaint(img_bgr: np.ndarray, mask: np.ndarray, cfg: Optional[Dict[str, Any]] = None,
            worker: Optional[LamaWorker] = None, cancel_check: Optional[Callable[[], None]] = None) -> np.ndarray:
    """LaMa on one crop: img (h, w, 3) BGR float 0..1, mask (h, w) > 0 = fill -> (h, w, 3) float32 BGR 0..1.
    The crop runs at `scale_for(mask)` (full resolution for holes up to HOLE_MAX px) and back; only the mask changes."""
    import cv2
    mp = model_path(cfg)
    if mp is None:
        raise LamaFullError("Brak modelu LaMa: inpaint/big-lama.pt – Doktor → „Zainstaluj” przy „Model: LaMa (big-lama.pt)”")
    img = np.asarray(img_bgr, np.float32)[..., :3]
    m = (np.asarray(mask, np.float32) > 0).astype(np.float32)
    h, w = m.shape
    s = scale_for(m)
    src, sm = img, m
    if s < 1.0:
        sw, sh = max(8, int(round(w * s))), max(8, int(round(h * s)))
        src = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        sm = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
    wk = worker
    with _LOCK if worker is None else _NullLock():
        wk = wk or _worker()
        n = len(os.listdir(wk.tmp))
        ip, kp, op = (str(wk.tmp / f"{t}_{os.getpid()}_{n}.npy") for t in ("img", "mask", "out"))
        try:
            np.save(ip, np.ascontiguousarray(src[..., ::-1]))           # the model is RGB
            np.save(kp, sm)
            wk.call({"op": "inpaint", "model": str(mp), "img": ip, "mask": kp, "out": op}, cancel_check)
            out = np.load(op)[..., ::-1]
        finally:
            for p in (ip, kp, op):
                try:
                    os.remove(p)
                except OSError:
                    pass
    if s < 1.0:
        out = cv2.resize(np.ascontiguousarray(out), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.where(m[..., None] > 0, out, img).astype(np.float32)


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def close() -> None:
    w = _WORKER["w"]
    if w is not None:
        w.close()
        from engine.gpu_manager import GPU
        GPU.forget(w)
        _WORKER["w"] = None
