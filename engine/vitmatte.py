# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/vitmatte.py - ViTMatte (hustvl, HF transformers) as a per-frame matte-zone engine.

ViTMatte is allowed (user decision 2026-10-14; it started as the 2026-10-13 experiment "MatAnyone2 vs ViTMatte on
the zone crop", SH020 `osoba`, zone z_02, when CLAUDE.md still banned it). Weights from the HF cache
(`hustvl/vitmatte-base-composition-1k`, ~390 MB).

Where the model runs (`get_runner` picks, `runner_info()["backend"]` says which):
- `venv`  – in this process, when MatteDesk's Python has torch + transformers (older installs);
- `comfy` – in ComfyUI's Python (its torch + CUDA + transformers) as one long-lived worker process
            (engine/vitmatte_worker.py; model loaded once, frames go through .npy files). MatteDesk's .venv has
            no PyTorch since v0.11, so this is the normal case. Doctor: "Model: ViTMatte".
Pre-processing (normalisation, trimap channel, padding) and the finish (crop, FG / BG forcing) are the same code
for both, only the forward pass differs.

Contract (same as the other zone engines): image crop (display BGR float 0..1) + trimap (0 BG / 128 unknown / 255 FG)
-> alpha (float32, 0..1) of the crop. FP32 on CUDA (0.13 s at 1024^2, ~3 GB); CPU fallback only when CUDA is absent
(reported in `stats.device`, never silent).
"""

from __future__ import annotations

import atexit
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from engine.worker_proto import CHECK_CACHE, WorkerProcess, check_worker

DEFAULT_MODEL = "hustvl/vitmatte-base-composition-1k"
MODELS = {                      # zone params.vitmatte_model (Nuke node: "High detail" = large ~ our base)
    "base": "hustvl/vitmatte-base-composition-1k",
    "small": "hustvl/vitmatte-small-composition-1k",
    "base-distinctions": "hustvl/vitmatte-base-distinctions-646",
    "small-distinctions": "hustvl/vitmatte-small-distinctions-646",
}


def model_id_for(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_MODEL
    return MODELS.get(str(name), str(name))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_MULTIPLE = 32          # ViTMatte (ViT-B/16 + window attention): input side a multiple of 32

# run with -I: the worker sits in engine/, where engine/queue.py would shadow the stdlib `queue`
WORKER = Path(__file__).resolve().parent / "vitmatte_worker.py"
_CHECK_CACHE = CHECK_CACHE


def prepare_input(img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
    """(H', W', 4) float32: ImageNet-normalised RGB + trimap channel, reflect-padded to a multiple of 32.
    `unknown_value` (Nuke ViTMatte node "Fine Tune"): the level the unknown band gets in the trimap channel
    (0.5 = the trained value; higher = a foreground-leaning prior, lower = background-leaning)."""
    h, w = trimap_u8.shape
    ph, pw = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
    rgb = np.clip(img_bgr[..., ::-1], 0, 1).astype(np.float32)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tri = (trimap_u8.astype(np.float32) / 255.0)
    if abs(float(unknown_value) - 0.5) > 1e-6:
        tri = np.where(trimap_u8 == 128, np.float32(unknown_value), tri).astype(np.float32)
    x = np.concatenate([rgb, tri[..., None]], axis=2)
    if ph or pw:
        x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(x, dtype=np.float32)


def finish_alpha(alpha_padded: np.ndarray, trimap_u8: np.ndarray) -> np.ndarray:
    """Crop the padded model output back to the trimap size, clamp, force FG / BG of the trimap."""
    h, w = trimap_u8.shape
    a = np.clip(alpha_padded[:h, :w], 0, 1).astype(np.float32)
    a[trimap_u8 == 255] = 1.0
    a[trimap_u8 == 0] = 0.0
    return a


def prof_add(r: Any, key: str, dt: float) -> None:
    """Accumulate wall seconds of one runner phase in `r.prof` (plan docs/PLAN_EDGE_SPEED.md A6): the zone reads the
    difference before / after its frames (`prof_delta`)."""
    p = getattr(r, "prof", None)
    if p is not None:
        p[key] = p.get(key, 0.0) + float(dt)


def prof_snapshot(r: Any) -> Dict[str, float]:
    return dict(getattr(r, "prof", None) or {})


def prof_delta(r: Any, before: Dict[str, float]) -> Dict[str, float]:
    """Runner phases since `before` (prof_snapshot), seconds rounded; empty for runners without `prof` (mocks)."""
    return {k: round(v - before.get(k, 0.0), 2) for k, v in (getattr(r, "prof", None) or {}).items()}


class VitMatteRunner:
    """In-process model holder (MatteDesk's Python has torch + transformers): `run(img_bgr, trimap_u8)` -> alpha.
    One instance per model id (see `get_runner`)."""

    backend = "venv"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self.load_seconds = 0.0
        self.prof: Dict[str, float] = {}       # prepare / forward (GPU incl. transfers) / finish, A6

    @staticmethod
    def available() -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
            from transformers import VitMatteForImageMatting  # noqa: F401
        except Exception as e:  # pragma: no cover - environment
            return False, f"brak torch / transformers w venv MatteDesk: {e}"
        return True, ""

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                import torch
                from transformers import VitMatteForImageMatting
                t0 = time.time()
                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                from engine.weights_find import resolve   # a folder already in ComfyUI's models, else HF
                m = VitMatteForImageMatting.from_pretrained(resolve(self.model_id)).eval().to(self.device)
                self._model = m
                self.load_seconds = time.time() - t0
        return self._model

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(H', W', 4) prepared input -> (H', W') alpha."""
        import torch
        from engine.gpu_manager import GPU
        with GPU.use(self, "ViTMatte (proces serwera)", self.close):
            model = self._load()
            t0 = time.perf_counter()
            t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]).to(self.device)
            with torch.inference_mode():
                a = model(pixel_values=t).alphas[0, 0].float().clamp_(0, 1).cpu().numpy()
            prof_add(self, "forward", time.perf_counter() - t0)
        return a.astype(np.float32)

    def close(self) -> None:
        """Drop the model (frees its GPU memory; engine/gpu_manager evicts it); the next `run` loads it again."""
        with self._lock:
            m, self._model = self._model, None
        if m is not None and self.device == "cuda":
            del m
            import torch
            torch.cuda.empty_cache()

    def run(self, img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
        """`img_bgr` HxWx3 float32 display (0..1), `trimap_u8` HxW (0 / 128 / 255) -> alpha HxW float32 (0..1)."""
        t0 = time.perf_counter()
        x = prepare_input(img_bgr, trimap_u8, unknown_value)
        prof_add(self, "prepare", time.perf_counter() - t0)
        a = self.forward(x)
        t0 = time.perf_counter()
        out = finish_alpha(a, trimap_u8)
        prof_add(self, "finish", time.perf_counter() - t0)
        return out

    def run_many(self, imgs: Sequence[np.ndarray], tris: Sequence[np.ndarray], unknown_value: float = 0.5) -> List[np.ndarray]:
        """The tiles of one frame, one after the other (each its own forward, batch 1 = `run`)."""
        return [self.run(i, t, unknown_value) for i, t in zip(imgs, tris)]


def worker_python() -> Optional[Path]:
    """ComfyUI's Python (config `comfy_python`, else Desktop .venv / portable embedded) – torch + CUDA + transformers."""
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


def check_worker_python(python: Optional[Union[str, Path]], model_ids: Optional[List[str]] = None,
                        env: Optional[Dict[str, str]] = None,
                        worker: Path = WORKER) -> Tuple[bool, str, List[str]]:
    """(ok, device or the reason, model ids already in the HF cache): can ComfyUI's Python run `worker`
    (ViTMatte by default; engine/birefnet.py passes its own)?"""
    ok, out = check_worker(python, worker, list(model_ids or MODELS.values()), env)
    if not ok:
        return False, out, []
    parts = out.split()
    cached = next((p[len("cached="):] for p in parts if p.startswith("cached=")), "")
    return True, parts[1] if len(parts) > 1 else "?", [c for c in cached.split(",") if c]


def job_cancel_check() -> None:
    """„Przerwij” for a call waiting on a worker: raises JobCancelled when the running job was cancelled. Outside a
    running job nothing: the cancel flag stays set until the next job starts and a pool thread can keep the id of a
    finished job – a background call must not die of either."""
    from engine.jobs import JOBS
    jid = JOBS.current_id()
    if jid and any(j["id"] == jid for j in JOBS.running()):
        JOBS.check()


class ComfyVitMatteRunner(WorkerProcess):
    """ViTMatte in ComfyUI's Python: one long-lived worker per model id (model loaded once), restarted if it dies.
    Same `run()` contract and attributes as `VitMatteRunner`."""

    backend = "comfy"
    label = "ViTMatte"                 # in messages; engine/birefnet.py subclasses with its own worker
    doctor_item = "Model: ViTMatte"
    worker = WORKER
    tmp_prefix = "md_vitmatte_"
    batched = True                     # the worker takes an (N, H, W, 4) stack = the tiles of one frame in ONE request

    def __init__(self, model_id: str = DEFAULT_MODEL, python: Optional[Union[str, Path]] = None,
                 env: Optional[Dict[str, str]] = None, timeout_s: float = 600.0):
        super().__init__(python if python else worker_python(), env, timeout_s)
        self.model_id = model_id
        self.device: Optional[str] = None
        self.load_seconds = 0.0
        self._lock = threading.Lock()
        self._n = 0
        # A6: prepare / npy_write / worker_read / worker_forward / worker_load / worker_write / ipc_wait (request wall
        # minus the worker's own phases: pipe + scheduling) / npy_read / finish; workers without timings -> "request"
        self.prof: Dict[str, float] = {}

    def available(self) -> Tuple[bool, str]:
        ok, why, _ = check_worker_python(self.python, [self.model_id], env=self.env, worker=self.worker)
        return (ok, "" if ok else f"{self.label} niedostępny w Pythonie ComfyUI: {why}")

    def serve_args(self) -> Tuple[str, ...]:
        from engine.weights_find import resolve   # a folder already in ComfyUI's models (2026-11-10), else the HF id
        return ("--serve", resolve(self.model_id))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(H', W', 4) -> (H', W') alpha; an (N, H', W', 4) stack -> (N, H', W') (every tile its own forward)."""
        from engine.gpu_manager import GPU
        with GPU.use(self, self.label, self.close), self._lock:
            self._n += 1
            x_path = self.tmp_dir() / f"x_{self._n}.npy"
            out_path = self.tmp_dir() / f"a_{self._n}.npy"
            t0 = time.perf_counter()
            np.save(x_path, np.ascontiguousarray(x, dtype=np.float32))
            prof_add(self, "npy_write", time.perf_counter() - t0)
            try:
                t0 = time.perf_counter()
                resp = self.request({"x": str(x_path), "out": str(out_path)}, job_cancel_check)
                wall = time.perf_counter() - t0
                if not resp.get("ok"):
                    raise RuntimeError(f"{self.label} (Python ComfyUI): {resp.get('error')}")
                self.device = resp.get("device") or self.device
                self.load_seconds = float(resp.get("load_seconds") or self.load_seconds)
                wt = resp.get("timings")
                if isinstance(wt, dict):
                    for k, v in wt.items():
                        prof_add(self, f"worker_{k}", float(v))
                    prof_add(self, "ipc_wait", wall - sum(float(v) for v in wt.values()))
                else:
                    prof_add(self, "request", wall)
                t0 = time.perf_counter()
                a = np.load(out_path).astype(np.float32)
                prof_add(self, "npy_read", time.perf_counter() - t0)
                return a
            finally:
                for p in (x_path, out_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def run(self, img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
        t0 = time.perf_counter()
        x = prepare_input(img_bgr, trimap_u8, unknown_value)
        prof_add(self, "prepare", time.perf_counter() - t0)
        a = self.forward(x)
        t0 = time.perf_counter()
        out = finish_alpha(a, trimap_u8)
        prof_add(self, "finish", time.perf_counter() - t0)
        return out

    def run_many(self, imgs: Sequence[np.ndarray], tris: Sequence[np.ndarray], unknown_value: float = 0.5) -> List[np.ndarray]:
        """The tiles of one frame in ONE request (plan docs/PLAN_EDGE_SPEED.md A5, 2026-10-25): one .npy with the
        (N, H', W', 4) stack, the worker runs the same forward per tile (batch 1, so the alphas are the ones `run`
        gives) and writes one (N, H', W') file – the per-tile protocol cost (58 ms measured) is paid once per frame.
        Tiles of different sizes, or a worker without stacks (`batched` False), go one by one."""
        t0 = time.perf_counter()
        xs = [prepare_input(i, t, unknown_value) for i, t in zip(imgs, tris)]
        prof_add(self, "prepare", time.perf_counter() - t0)
        if not xs:
            return []
        if not self.batched or len(xs) == 1 or any(x.shape != xs[0].shape for x in xs):
            outs = [self.forward(x) for x in xs]
        else:
            t0 = time.perf_counter()
            stack = np.stack(xs)
            prof_add(self, "prepare", time.perf_counter() - t0)
            outs = self.forward(stack)
        t0 = time.perf_counter()
        res = [finish_alpha(a, t) for a, t in zip(outs, tris)]
        prof_add(self, "finish", time.perf_counter() - t0)
        return res



_RUNNERS: Dict[str, Any] = {}


def get_runner(model_id: str = DEFAULT_MODEL) -> Any:
    """Process-wide runner per model id (tests monkeypatch this to a mock): in this process when the venv has
    torch + transformers, else the ComfyUI-Python worker."""
    r = _RUNNERS.get(model_id)
    if r is None:
        if VitMatteRunner.available()[0]:
            r = VitMatteRunner(model_id)
        else:
            return pooled_comfy_runner(_RUNNERS, ComfyVitMatteRunner, model_id)
        _RUNNERS[model_id] = r
    return r


def pooled_comfy_runner(pool: Dict[str, Any], cls: Any, model_id: str) -> Any:
    """The process-wide worker runner of `cls` per model id; RuntimeError naming the Doctor item when it cannot run."""
    r = pool.get(model_id)
    if r is None:
        r = cls(model_id)
        ok, why = r.available()
        if not ok:
            raise RuntimeError(f"{why} – Doktor → „{r.doctor_item}”")
        pool[model_id] = r
    return r


def close_pool(pool: Dict[str, Any]) -> None:
    for r in list(pool.values()):
        if hasattr(r, "close"):
            r.close()
    pool.clear()


def close_runners() -> None:
    close_pool(_RUNNERS)


atexit.register(close_runners)


def runner_info(r: Any) -> Dict[str, Any]:
    return {"model": getattr(r, "model_id", "mock"), "device": getattr(r, "device", None) or "?",
            "backend": getattr(r, "backend", "mock"),
            "load_seconds": round(float(getattr(r, "load_seconds", 0.0)), 2)}
