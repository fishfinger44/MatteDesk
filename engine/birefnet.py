# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/birefnet.py - BiRefNet_HR-matting / Lucida (MIT weights) as a per-frame matte engine, the candidate
alternative to ViTMatte for commercial work (ViTMatte's composition-1k weights are trained on the Adobe Deep Image
Matting dataset, whose licence limits trained models to non-commercial use – see THIRD_PARTY_LICENSES.md).

Runs only in ComfyUI's Python, as one long-lived worker per model (engine/birefnet_worker.py; same machinery as
`ComfyVitMatteRunner`). Not wired into the zone pipeline yet (plan 2026-10-22, T2: A/B against ViTMatte first).

Contract (same as the other zone engines): image crop (display BGR float 0..1) + trimap (0 BG / 128 unknown / 255 FG)
-> alpha (float32, 0..1) of the crop. The model sees RGB only – the trimap is used just for the finish (FG / BG
forcing, `vitmatte.finish_alpha`), so the alpha in the unknown band is the model's own.
Weights: the user downloads them (Doctor rule): HF cache, `ZhengPeng7/BiRefNet_HR-matting` ~444 MB,
`egeorcun/lucida` ~885 MB.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from engine import vitmatte

WORKER = Path(__file__).resolve().parent / "birefnet_worker.py"
DEFAULT_MODEL = "ZhengPeng7/BiRefNet_HR-matting"
MODELS = {                      # engine name -> HF id
    "birefnet_hr_matting": "ZhengPeng7/BiRefNet_HR-matting",
    "lucida": "egeorcun/lucida",
}
LABELS = {"birefnet_hr_matting": "BiRefNet_HR-matting", "lucida": "Lucida"}


def model_id_for(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_MODEL
    return MODELS.get(str(name), str(name))


def prepare_input(img_bgr: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 RGB 0..1 – resizing to the model's square and normalising happen in the worker (GPU)."""
    return np.ascontiguousarray(np.clip(img_bgr[..., ::-1], 0, 1), dtype=np.float32)


class ComfyBirefnetRunner(vitmatte.ComfyVitMatteRunner):
    """BiRefNet_HR-matting / Lucida in ComfyUI's Python. `run(img_bgr, trimap_u8)` -> alpha, like ViTMatte;
    `unknown_value` is accepted for the common signature and ignored (the model has no trimap input)."""

    label = "BiRefNet"
    doctor_item = "Model: BiRefNet_HR-matting / Lucida"
    worker = WORKER
    tmp_prefix = "md_birefnet_"

    def __init__(self, model_id: str = DEFAULT_MODEL, **kw: Any):
        super().__init__(model_id, **kw)
        self.label = next((LABELS[k] for k, v in MODELS.items() if v == model_id), "BiRefNet")

    def run(self, img_bgr: np.ndarray, trimap_u8: np.ndarray, unknown_value: float = 0.5) -> np.ndarray:
        return vitmatte.finish_alpha(self.forward(prepare_input(img_bgr)), trimap_u8)


_RUNNERS: Dict[str, ComfyBirefnetRunner] = {}


def get_runner(model_id: str = DEFAULT_MODEL) -> ComfyBirefnetRunner:
    """Process-wide runner per model id (tests monkeypatch this to a mock)."""
    r = _RUNNERS.get(model_id)
    if r is None:
        r = ComfyBirefnetRunner(model_id)
        ok, why = r.available()
        if not ok:
            raise RuntimeError(f"{why} – Doktor → „{r.doctor_item}”")
        _RUNNERS[model_id] = r
    return r


def close_runners() -> None:
    for r in list(_RUNNERS.values()):
        r.close()
    _RUNNERS.clear()


atexit.register(close_runners)
