# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/sapiens2.py - Sapiens2 body-part segmentation (Meta, `facebook/sapiens2-seg-0.4b`, Sapiens2 License –
commercial use allowed, no surveillance / biometrics / deepfakes) as the source of the automatic band proposal
(engine/bandseed.py, plan docs/PLAN_SAPIENS2.md point 1, tests local_results/sapiens2_band_2026-10-22 and
sapiens2_vitmatte_2026-10-22).

Runs only in ComfyUI's Python, one long-lived worker (engine/sapiens2_worker.py; same machinery as
`ComfyVitMatteRunner`). Only people: the model knows body parts, nothing else.
Weights: the user downloads them (Doctor rule): HF cache, `facebook/sapiens2-seg-0.4b` ~1.63 GB; 0.17 s / frame and
2.8 GB VRAM in fp16 on an RTX 3080 Ti (2026-10-22).
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from engine import vitmatte

WORKER = Path(__file__).resolve().parent / "sapiens2_worker.py"
DEFAULT_MODEL = "facebook/sapiens2-seg-0.4b"
LABEL = "Sapiens2-seg-0.4B"
INPUT_HW = (1024, 768)          # the model's training aspect (H x W): the person crop is grown to 4:3 portrait
CROP_MARGIN = 0.15              # around the outline bbox (the test 2026-10-22)


class ComfySapiens2Runner(vitmatte.ComfyVitMatteRunner):
    """Sapiens2 segmentation in ComfyUI's Python: `classes(img_bgr)` -> (H, W) uint8 class map 0..28."""

    label = LABEL
    doctor_item = "Model: Sapiens2-seg-0.4B"
    worker = WORKER
    tmp_prefix = "md_sapiens2_"

    def classes(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(np.clip(img_bgr[..., ::-1], 0, 1), dtype=np.float32)
        return np.clip(np.rint(self.forward(rgb)), 0, 28).astype(np.uint8)


def person_crop(core: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    """(x, y, w, h) of the crop the model sees: the outline bbox + 15 %, grown to the model's 3:4 (W:H) aspect; may
    reach outside the frame (padded black). None = empty outline."""
    ys, xs = np.where(np.asarray(core) > 0.5)
    if not len(xs):
        return None
    x0, x1, y0, y1 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
    bw, bh = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - CROP_MARGIN * bw, x1 + CROP_MARGIN * bw, y0 - CROP_MARGIN * bh, y1 + CROP_MARGIN * bh
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    ah, aw = INPUT_HW
    if bw / bh > aw / ah:
        bh = bw * ah / aw
    else:
        bw = bh * aw / ah
    return int(round(cx - bw / 2)), int(round(cy - bh / 2)), max(1, int(round(bw))), max(1, int(round(bh)))


def segment_object(runner: Any, img_bgr: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Full-frame class map (uint8; 255 = outside the crop) of the person whose outline is `core`."""
    h, w = core.shape[:2]
    out = np.full((h, w), 255, np.uint8)
    box = person_crop(core, h, w)
    if box is None:
        return out
    bx, by, bw, bh = box
    pad = cv2.copyMakeBorder(np.asarray(img_bgr, np.float32), max(0, -by), max(0, by + bh - h), max(0, -bx),
                             max(0, bx + bw - w), cv2.BORDER_CONSTANT, value=0)
    ox, oy = max(0, -bx), max(0, -by)
    crop = pad[by + oy: by + oy + bh, bx + ox: bx + ox + bw]
    cls = runner.classes(crop)
    xa, ya, xb, yb = max(0, bx), max(0, by), min(w, bx + bw), min(h, by + bh)
    out[ya:yb, xa:xb] = cls[ya - by: yb - by, xa - bx: xb - bx]
    return out


_RUNNERS: Dict[str, ComfySapiens2Runner] = {}


def get_runner(model_id: str = DEFAULT_MODEL) -> ComfySapiens2Runner:
    """Process-wide runner (tests monkeypatch this to a mock)."""
    r = _RUNNERS.get(model_id)
    if r is None:
        r = ComfySapiens2Runner(model_id)
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
