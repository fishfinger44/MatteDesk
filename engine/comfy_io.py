# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/comfy_io.py - How the tool hands images to ComfyUI, routed by the plate's bit depth (decision 2026-10-13).

- "exr"  (EXR / scene-linear sources, ACES, typically <= 500 frames): float32 EXR + PlateLoadEXR / PlateLoadMaskSeq –
  unchanged, bit-identical to the results before this module (SH010 regression).
- "png8" (8-bit sources: MP4 / PNG / JPG, sRGB): PNG 8-bit (the display values as they are, no colour transform)
  loaded with VideoHelperSuite `VHS_LoadImagesPath` (reads a whole folder, sorted -> every job folder holds only
  the frames of that job; masks go to their own subfolders) + core `ImageToMask` for masks. Frames larger than
  SAM_RES sent to SAM3_VideoTrack are downscaled (INTER_AREA) – SAM 3.1 works at 1008 internally anyway – and the
  masks are resized back (INTER_LINEAR).
"""

from __future__ import annotations

import contextlib
import contextvars

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

EXR = "exr"
PNG8 = "png8"


def io_mode_for(manifest: Optional[Dict[str, Any]]) -> str:
    """'png8' for 8-bit sources (manifest bit_depth <= 8; without bit_depth: source_type 'video'), else 'exr'."""
    m = manifest or {}
    bd = m.get("bit_depth")
    if bd is not None:
        try:
            return PNG8 if int(bd) <= 8 else EXR
        except (TypeError, ValueError):
            pass
    return PNG8 if m.get("source_type") == "video" else EXR


def write_png8(path: Path, img: np.ndarray) -> None:
    """Display float 0..1 (BGR, or one channel) -> 8-bit PNG, values unchanged (no colour transform)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.rint(np.asarray(img, np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), u8):   # Unicode-safe (engine.cv2_unicode wraps cv2.imwrite)
        raise RuntimeError(f"Nie udało się zapisać {path}")


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Hard mask (> 0.5) -> 8-bit PNG 0 / 255."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (np.asarray(mask, np.float32) > 0.5).astype(np.uint8) * 255):
        raise RuntimeError(f"Nie udało się zapisać {path}")


VHS_NODE = "VHS_LoadImagesPath"
VHS_MISSING = ("Brak węzła VHS_LoadImagesPath (ComfyUI-VideoHelperSuite) w ComfyUI – ujęcia 8-bitowe (PNG/JPG) wysyłają "
               "klatki do ComfyUI tym węzłem. Otwórz Doktora → „VideoHelperSuite (ujęcia 8-bit)” → „Zainstaluj”, "
               "potem zrestartuj ComfyUI i uruchom liczenie ponownie.")
_VHS_OK: set = set()   # ComfyUI addresses where the node was found (a node does not vanish without a ComfyUI restart)


class MissingComfyNode(RuntimeError):
    """A node the job needs is not registered in ComfyUI (raised before anything is written or sent)."""


def require_vhs(client: Any) -> None:
    """Comfy I/O "png8" needs VHS_LoadImagesPath: checked via /object_info BEFORE the job writes its frames, so the
    user gets a clear error pointing to the Doctor instead of Comfy's "Node 'VHS_LoadImagesPath' not found" later.
    ComfyUI not answering (None) -> no verdict here: the job's own connection error says that."""
    url = getattr(client, "base_url", None)
    if url and url in _VHS_OK:
        return
    try:
        info = client.get_object_info(VHS_NODE)
    except Exception:
        return
    if info is None:
        return
    if not info:   # ComfyUI answers {} for an unknown node class
        raise MissingComfyNode(VHS_MISSING)
    if url:
        _VHS_OK.add(url)


def write_soft_png(path: Path, p: np.ndarray) -> None:
    """Soft mask (float 0..1) -> 8-bit PNG 0..255. Values below 2 become 0: PlateLoadMaskSeq reads a file whose
    maximum is 1 as a 0..1 mask (a near-empty map would turn into a full core)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p8 = (np.clip(np.asarray(p, np.float32), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    p8[p8 < 2] = 0
    if not cv2.imwrite(str(path), p8):
        raise RuntimeError(f"Nie udało się zapisać {path}")


def vhs_load(directory: Path) -> Dict[str, Any]:
    """VHS_LoadImagesPath node (outputs: 0 IMAGE, 1 MASK, 2 frame_count) for every image of `directory`."""
    return {"class_type": "VHS_LoadImagesPath",
            "inputs": {"directory": str(directory).replace("\\", "/"), "image_load_cap": 0, "skip_first_images": 0,
                       "select_every_nth": 1}}


def image_to_mask(image_ref) -> Dict[str, Any]:
    return {"class_type": "ImageToMask", "inputs": {"image": image_ref, "channel": "red"}}


def fit_sam_res(img: np.ndarray, sam_res: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Downscale so the longer side is <= sam_res (INTER_AREA by default); smaller frames are returned unchanged."""
    h, w = img.shape[:2]
    s = sam_res / float(max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interpolation)


_STEP_PREFIX: contextvars.ContextVar[str] = contextvars.ContextVar("md_step_prefix", default="")


@contextlib.contextmanager
def progress_prefix(label: str):
    """Names the phase in front of the generic per-frame steps (e.g. the crop pass on the whole shot, so its frame
    count is not read as the frames of the core run)."""
    tok = _STEP_PREFIX.set(label)
    try:
        yield
    finally:
        _STEP_PREFIX.reset(tok)


def job_progress(step: str, done: int, total: int) -> None:
    """Progress + 'Przerwij' inside per-frame write loops: JobCancelled after a cancel. No-op outside a job."""
    from engine.jobs import JOBS   # lazy: engine.jobs must not import the engine modules
    if JOBS.running():
        pre = _STEP_PREFIX.get()
        JOBS.progress(f"{pre} – {step}" if pre else step, done, total)


class LazySeq:
    """Read-only sequence whose items are made on access (`fn(index)`); slicing gives another LazySeq. Keeps long
    shots (1000+ frames) from holding every frame / mask in RAM at once (results are the same as with a list)."""

    def __init__(self, fn, idx):
        self.fn = fn
        self.idx = list(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return LazySeq(self.fn, self.idx[k])
        return self.fn(self.idx[k])

    def __iter__(self):
        for i in self.idx:
            yield self.fn(i)
