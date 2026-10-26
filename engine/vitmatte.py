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
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

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
_CHECK_CACHE: Dict[str, Tuple[bool, str, List[str]]] = {}


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
                m = VitMatteForImageMatting.from_pretrained(self.model_id).eval().to(self.device)
                self._model = m
                self.load_seconds = time.time() - t0
        return self._model

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(H', W', 4) prepared input -> (H', W') alpha."""
        import torch
        model = self._load()
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]).to(self.device)
        with torch.inference_mode():
            a = model(pixel_values=t).alphas[0, 0].float().clamp_(0, 1).cpu().numpy()
        return a.astype(np.float32)

    def close(self) -> None:
        """Drop the model (frees its GPU memory, engine/gpu_release); the next `run` loads it again."""
        with self._lock:
            m, self._model = self._model, None
        if m is not None and self.device == "cuda":
            del m
            import torch
            torch.cuda.empty_cache()

    def run(self, img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
        """`img_bgr` HxWx3 float32 display (0..1), `trimap_u8` HxW (0 / 128 / 255) -> alpha HxW float32 (0..1)."""
        return finish_alpha(self.forward(prepare_input(img_bgr, trimap_u8, unknown_value)), trimap_u8)

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
    (ViTMatte by default; engine/birefnet.py passes its own)? Cached per interpreter (the installer clears `_CHECK_CACHE`)."""
    if not python or not Path(python).exists():
        return False, "nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python)", []
    ids = list(model_ids or MODELS.values())
    key = str(python) + "|" + str(worker) + "|" + ",".join(ids)
    if key not in _CHECK_CACHE:
        try:
            r = subprocess.run([str(python), "-I", str(worker), "--check", *ids], capture_output=True, text=True,
                               timeout=180, env=env)
            out = (r.stdout or r.stderr or "").strip()
            if r.returncode == 0 and out.startswith("ok"):
                parts = out.split()
                cached = next((p[len("cached="):] for p in parts if p.startswith("cached=")), "")
                _CHECK_CACHE[key] = (True, parts[1] if len(parts) > 1 else "?", [c for c in cached.split(",") if c])
            else:
                _CHECK_CACHE[key] = (False, out[-300:] or f"kod {r.returncode}", [])
        except Exception as e:
            _CHECK_CACHE[key] = (False, str(e), [])
    return _CHECK_CACHE[key]


class ComfyVitMatteRunner:
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
        self.model_id = model_id
        self.python = str(python) if python else (str(worker_python()) if worker_python() else "")
        self.env = env
        self.timeout_s = timeout_s
        self.device: Optional[str] = None
        self.load_seconds = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._dir: Optional[tempfile.TemporaryDirectory] = None
        self._log = None
        self._lock = threading.Lock()
        self._n = 0

    def available(self) -> Tuple[bool, str]:
        ok, why, _ = check_worker_python(self.python, [self.model_id], env=self.env, worker=self.worker)
        return (ok, "" if ok else f"{self.label} niedostępny w Pythonie ComfyUI: {why}")

    def _start(self) -> None:
        if self._dir is None:
            self._dir = tempfile.TemporaryDirectory(prefix=self.tmp_prefix)
        if self._log is None:
            self._log = open(Path(self._dir.name) / "worker.log", "a+", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen([self.python, "-I", str(self.worker), "--serve", self.model_id],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True,
                                      bufsize=1, env=self.env)

    def _log_tail(self) -> str:
        try:
            self._log.flush()
            self._log.seek(0)
            return self._log.read()[-600:].strip()
        except Exception:
            return ""

    def _request(self, x_path: Path, out_path: Path) -> Dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            if not self.python or not Path(self.python).exists():
                raise RuntimeError(f"{self.label}: nie znaleziono Pythona ComfyUI (Doktor → „{self.doctor_item}”)")
            self._start()
        result: Dict[str, Any] = {}

        def read():
            result["line"] = self._proc.stdout.readline()

        self._proc.stdin.write(json.dumps({"x": str(x_path), "out": str(out_path)}) + "\n")
        self._proc.stdin.flush()
        th = threading.Thread(target=read, daemon=True)
        th.start()
        th.join(self.timeout_s)
        line = result.get("line")
        if not line:
            alive = self._proc.poll() is None
            tail = self._log_tail()
            self.close()
            raise RuntimeError(f"{self.label} (Python ComfyUI) " +("nie odpowiedział w czasie" if alive else "zakończył się")
                               + f": {tail}")
        return json.loads(line)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(H', W', 4) -> (H', W') alpha; an (N, H', W', 4) stack -> (N, H', W') (every tile its own forward)."""
        with self._lock:
            if self._dir is None:
                self._dir = tempfile.TemporaryDirectory(prefix=self.tmp_prefix)
            self._n += 1
            x_path = Path(self._dir.name) / f"x_{self._n}.npy"
            out_path = Path(self._dir.name) / f"a_{self._n}.npy"
            np.save(x_path, np.ascontiguousarray(x, dtype=np.float32))
            try:
                resp = self._request(x_path, out_path)
                if not resp.get("ok"):
                    raise RuntimeError(f"{self.label} (Python ComfyUI): {resp.get('error')}")
                self.device = resp.get("device") or self.device
                self.load_seconds = float(resp.get("load_seconds") or self.load_seconds)
                return np.load(out_path).astype(np.float32)
            finally:
                for p in (x_path, out_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def run(self, img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
        return finish_alpha(self.forward(prepare_input(img_bgr, trimap_u8, unknown_value)), trimap_u8)

    def run_many(self, imgs: Sequence[np.ndarray], tris: Sequence[np.ndarray], unknown_value: float = 0.5) -> List[np.ndarray]:
        """The tiles of one frame in ONE request (plan docs/PLAN_EDGE_SPEED.md A5, 2026-10-25): one .npy with the
        (N, H', W', 4) stack, the worker runs the same forward per tile (batch 1, so the alphas are the ones `run`
        gives) and writes one (N, H', W') file – the per-tile protocol cost (58 ms measured) is paid once per frame.
        Tiles of different sizes, or a worker without stacks (`batched` False), go one by one."""
        xs = [prepare_input(i, t, unknown_value) for i, t in zip(imgs, tris)]
        if not xs:
            return []
        if not self.batched or len(xs) == 1 or any(x.shape != xs[0].shape for x in xs):
            return [finish_alpha(self.forward(x), t) for x, t in zip(xs, tris)]
        outs = self.forward(np.stack(xs))
        return [finish_alpha(a, t) for a, t in zip(outs, tris)]

    def close(self) -> None:
        """Stop the worker (frees its GPU memory); the next `run` starts a new one."""
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=10)
            except Exception:
                p.kill()
                p.wait()
        log, self._log = self._log, None
        if log is not None:
            log.close()
        d, self._dir = self._dir, None
        if d is not None:
            d.cleanup()


_RUNNERS: Dict[str, Any] = {}


def get_runner(model_id: str = DEFAULT_MODEL) -> Any:
    """Process-wide runner per model id (tests monkeypatch this to a mock): in this process when the venv has
    torch + transformers, else the ComfyUI-Python worker."""
    r = _RUNNERS.get(model_id)
    if r is None:
        if VitMatteRunner.available()[0]:
            r = VitMatteRunner(model_id)
        else:
            r = ComfyVitMatteRunner(model_id)
            ok, why = r.available()
            if not ok:
                raise RuntimeError(f"{why} – Doktor → „Model: ViTMatte” → Zainstaluj")
        _RUNNERS[model_id] = r
    return r


def close_runners() -> None:
    for r in list(_RUNNERS.values()):
        if hasattr(r, "close"):
            r.close()
    _RUNNERS.clear()


atexit.register(close_runners)


def runner_info(r: Any) -> Dict[str, Any]:
    return {"model": getattr(r, "model_id", "mock"), "device": getattr(r, "device", None) or "?",
            "backend": getattr(r, "backend", "mock"),
            "load_seconds": round(float(getattr(r, "load_seconds", 0.0)), 2)}
