# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cleanplate_auto.py - automatic clean plate of a shot from the shot itself (docs/PLAN_HAIR_TOOL.md step 4b).

After the cores exist: for every pixel the temporal median over the sampled frames in which it is NOT covered by any
object (union of the cores, grown by `grow_px`). Static camera (the moving-camera variant – Camera2D space + per-frame
warp – is a later step; the confidence map already drops where the background itself changes between frames).
Outputs (`<work>/matte_desk/cleanplate_auto/`): `cp_####.png` (16-bit display values; one file per frame as hard links
of one image, so the plate works everywhere a sequence pattern is expected), `confidence.png` (0..255: 0 = never
uncovered -> inpainted from the surroundings, 255 = >= 5 clean samples and a stable background), `cleanplate_auto.json`.
Used by: the `clean` zone engine (difference key against the background in the same place), QA clear-plate agreement,
the stage-0 compare tools (registered as plate `cleanplate_auto`; a real project clean plate takes precedence).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA = "matte_desk/cleanplate_auto@1.0"
FULL_CONF_SAMPLES = 5        # >= 5 uncovered samples of a stable background = full confidence
STD_REF = 0.06               # background temporal std (display units, mean over BGR) at which confidence drops to 0


def sample_frames(frames: Sequence[int], max_samples: int = 60) -> List[int]:
    frames = list(frames)
    if len(frames) <= max_samples:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_samples).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


CONF_INPAINT = {"comfy_lama": 0.5, "opencv": 0.0, "none": 0.0}   # confidence of filled holes per backend


def opencv_inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fallback without Comfy: Telea on 8-bit (smooth walls OK, structures blur)."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.inpaint(u8, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    out = img.copy()
    out[mask] = u8[mask].astype(np.float32) / 255.0
    return out


def lama_graph(frames_dir: str, mask_dir: str, prefix: str, model_name: str = "big-lama.pt") -> Dict[str, Any]:
    """comfyui-inpaint-nodes: VHS folder (one PNG) + mask PNG -> INPAINT_InpaintWithModel (LaMa) -> SaveImage."""
    from engine.comfy_io import image_to_mask, vhs_load
    return {
        "1": vhs_load(Path(frames_dir)),
        "2": vhs_load(Path(mask_dir)),
        "3": image_to_mask(["2", 0]),
        "4": {"class_type": "INPAINT_LoadInpaintModel", "inputs": {"model_name": model_name}},
        "5": {"class_type": "INPAINT_InpaintWithModel", "inputs": {"inpaint_model": ["4", 0], "image": ["1", 0],
                                                                  "mask": ["3", 0], "seed": 0}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix}},
    }


def lama_available(client: Any) -> bool:
    """LaMa graph = inpaint nodes + VHS_LoadImagesPath (its PNG inputs); without either -> OpenCV fill."""
    try:
        info = client.get_object_info("INPAINT_InpaintWithModel") or {}
        return "INPAINT_InpaintWithModel" in info and bool(client.get_object_info("VHS_LoadImagesPath"))
    except Exception:
        return False


def lama_inpaint_comfy(client: Any, io_dir: Path, img: np.ndarray, mask: np.ndarray,
                       model_name: str = "big-lama.pt") -> np.ndarray:
    """LaMa through the running ComfyUI (8-bit in / out; the mask is grown 4 px so the seam lies in the fill)."""
    import shutil
    import uuid
    from engine.comfy_io import write_mask_png, write_png8
    job = Path(io_dir) / "cleanplate_lama"
    shutil.rmtree(job, ignore_errors=True)
    write_png8(job / "frames" / "f_0000.png", img)
    m = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    write_mask_png(job / "mask" / "m_0000.png", m)
    g = lama_graph(str(job / "frames"), str(job / "mask"), f"matte_desk/cleanplate_lama_{uuid.uuid4().hex[:8]}/cp", model_name)
    sub = client.queue_prompt(g)
    if not sub.get("success"):
        raise RuntimeError(f"Comfy odrzucił graf LaMa: {sub.get('error')}")
    res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=600)
    if not res.get("success"):
        raise RuntimeError(f"LaMa nie powiódł się: {res.get('error')}")
    outs = sorted(res.get("outputs", {}).get("6", {}).get("images", []), key=lambda i: i["filename"])
    if not outs:
        raise RuntimeError("LaMa: brak obrazu wyjściowego")
    buf = np.frombuffer(client.download_output(outs[0]), np.uint8)
    out8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out8 is None or out8.shape[:2] != img.shape[:2]:
        raise RuntimeError("LaMa: zły obraz wyjściowy")
    out = img.copy()
    out[mask] = out8[mask].astype(np.float32) / 255.0
    return out


def fill_holes(clean: np.ndarray, never: np.ndarray, inpaint: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
               ) -> Tuple[np.ndarray, str]:
    """`inpaint(img, mask)` = LaMa via Comfy (preferred) – on failure the OpenCV fallback; returns (image, backend)."""
    if inpaint is not None:
        try:
            return inpaint(clean, never), "comfy_lama"
        except Exception:
            pass
    return opencv_inpaint(clean, never), "opencv"


def build_cleanplate(frames: Sequence[int], load_frame: Callable[[int], np.ndarray],
                     covered: Callable[[int], np.ndarray], grow_px: int = 40, max_samples: int = 60,
                     band_rows: int = 64, progress: Optional[Callable[[str, int, int], None]] = None,
                     inpaint: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
                     ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Returns (clean BGR float32 0..1, confidence float32 0..1, stats). `covered(f)` = bool full-frame mask of the
    objects (cores ∪ front objects) in frame f; it is grown by `grow_px` (soft edges, hair) before use.
    `inpaint(img, mask)` fills the never-uncovered holes (LaMa through Comfy); None / failure -> OpenCV Telea."""
    t0 = time.time()
    sel = sample_frames(frames, max_samples)
    imgs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(grow_px) + 1, 2 * int(grow_px) + 1)) if grow_px > 0 else None
    for i, f in enumerate(sel):
        if progress:
            progress("Clean plate (auto): wczytywanie klatek", i + 1, len(sel))
        img = np.clip(load_frame(f), 0, 1).astype(np.float16)      # half precision: 60 x 1080p = 750 MB, not 1.5 GB
        m = covered(f).astype(np.uint8)
        if k is not None and m.any():
            m = cv2.dilate(m, k)
        imgs.append(img)
        masks.append(m.astype(bool))
    H, W = imgs[0].shape[:2]
    clean = np.zeros((H, W, 3), np.float32)
    count = np.zeros((H, W), np.int32)
    tstd = np.zeros((H, W), np.float32)
    nb = (H + band_rows - 1) // band_rows
    for b in range(nb):
        if progress:
            progress("Clean plate (auto): mediana tła", b + 1, nb)
        y0, y1 = b * band_rows, min(H, (b + 1) * band_rows)
        stack = np.stack([im[y0:y1] for im in imgs]).astype(np.float32)           # N x h x W x 3
        cov = np.stack([mk[y0:y1] for mk in masks])                                 # N x h x W
        stack[cov] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices = never-uncovered pixels (handled below)
            med = np.nanmedian(stack, axis=0)
            sd = np.nanstd(stack, axis=0).mean(axis=2)
        n = (~cov).sum(axis=0)
        med[np.isnan(med)] = 0.0
        sd[np.isnan(sd)] = 0.0
        clean[y0:y1] = med
        count[y0:y1] = n
        tstd[y0:y1] = sd
    never = count == 0
    inpainted_px = int(never.sum())
    backend_used = "none"
    if never.any():   # never uncovered (e.g. right next to a static object): fill from the surroundings
        if progress:
            progress("Clean plate (auto): wypełnianie dziur", 0, 1)
        clean, backend_used = fill_holes(clean, never, inpaint)
    conf = np.clip(count / float(FULL_CONF_SAMPLES), 0, 1) * np.clip(1.0 - tstd / STD_REF, 0, 1)
    conf[never] = CONF_INPAINT.get(backend_used, 0.0)
    stats = {"schema": SCHEMA, "frames_sampled": sel, "n_frames": len(frames), "grow_px": int(grow_px),
             "camera": "static (assumed)", "never_uncovered_px": inpainted_px, "inpaint_backend": backend_used,
             "uncovered_samples_median": float(np.median(count)),
             "bg_temporal_std_median": round(float(np.median(tstd[~never])), 4) if (~never).any() else None,
             "confidence_mean": round(float(conf.mean()), 4), "seconds": round(time.time() - t0, 1),
             "created_at": datetime.now().isoformat(timespec="seconds")}
    return clean.astype(np.float32), conf.astype(np.float32), stats


def write_cleanplate(out_dir: Path, clean: np.ndarray, conf: np.ndarray, frames: Sequence[int],
                     stats: Dict[str, Any]) -> str:
    """cp_####.png (16-bit) for every frame as hard links of the first (copies when linking fails), confidence.png,
    cleanplate_auto.json. Returns the sequence pattern."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    first = out_dir / f"cp_{frames[0]:04d}.png"
    ok = cv2.imwrite(str(first), (np.clip(clean, 0, 1) * 65535.0 + 0.5).astype(np.uint16))
    if not ok:
        raise RuntimeError(f"nie udało się zapisać {first}")
    for f in frames[1:]:
        p = out_dir / f"cp_{f:04d}.png"
        try:
            os.link(first, p)
        except OSError:
            shutil.copyfile(first, p)
    cv2.imwrite(str(out_dir / "confidence.png"), (np.clip(conf, 0, 1) * 255).astype(np.uint8))
    pattern = str(out_dir / "cp_####.png").replace("\\", "/")
    (out_dir / "cleanplate_auto.json").write_text(json.dumps({**stats, "pattern": pattern, "first": frames[0],
                                                              "last": frames[-1]}, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    return pattern


def load_cleanplate(out_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """(clean float32 BGR 0..1, confidence 0..1, stats) or None when not built."""
    out_dir = Path(out_dir)
    j = out_dir / "cleanplate_auto.json"
    if not j.exists():
        return None
    stats = json.loads(j.read_text(encoding="utf-8"))
    p = out_dir / f"cp_{int(stats.get('first', 1)):04d}.png"
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    clean = img[..., :3].astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    c = cv2.imread(str(out_dir / "confidence.png"), cv2.IMREAD_GRAYSCALE)
    conf = (c.astype(np.float32) / 255.0) if c is not None else np.ones(clean.shape[:2], np.float32)
    return clean, conf, stats
