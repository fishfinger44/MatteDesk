# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align_exclude.py - what the plate alignment (engine/align.py) must NOT measure on (align_v2 step 1,
2026-11-03): the objects' masks, a user box, a user mask.

The alignment looks for the camera's misalignment between the beauty and a clear plate / backlit on the STATIC
scene. Until 2026-11-03 the static area of the local model came from the clear-plate difference of the FIRST frame
only: an object walking across the frame dragged grid points with it on later frames, and contact shadows under the
difference threshold were measured as "static". Now the static mask is built from every anchor frame (engine/align.py)
and, through this module, minus the objects already known to the project: the SAM 3.1 cores / masks 1+2
(`<obj>/core_fix/c_####.png` when „Zastosuj poprawki” wrote them, else `<obj>/core/c_####.png`) of the chosen objects,
grown by `dilate_px` (contact shadows, soft edges), on every frame. The report records which objects and their
mask signatures (engine/stages.mask_sig), so a later core changes the alignment's „Zmierz ponownie” reason.

`exclusion_spec` is small (paths only) so it travels to the pool processes and the GPU decode pool; the masks are
read where they are used (`frame_exclusion`, `union_exclusion`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

DEFAULT_DILATE_PX = 32


def core_png(work: Path, name: str, f: int) -> Optional[Path]:
    """The object's mask 1+2 of frame f (core_fix first, then the SAM 3.1 core), None when it has none."""
    od = Path(work) / "objects" / name
    for sub in ("core_fix", "core"):
        p = od / sub / f"c_{f:04d}.png"
        if p.exists():
            return p
    return None


def objects_with_cores(work: Path) -> List[str]:
    """Objects (not group entities) that have at least one core frame - the candidates of „wyklucz obiekty”."""
    from engine.objects import ObjectStore
    out = []
    try:
        objs = ObjectStore(Path(work)).list()
    except Exception:
        return out
    for o in objs:
        od = Path(work) / "objects" / o["name"]
        if any((od / sub).is_dir() and any((od / sub).glob("c_*.png")) for sub in ("core_fix", "core")):
            out.append(o["name"])
    return out


def object_sigs(work: Path, names: Iterable[str]) -> Dict[str, str]:
    """{object: engine.stages.mask_sig} - the signature of the mask the alignment excluded."""
    from engine.objects import ObjectStore
    from engine.stages import mask_sig
    store = ObjectStore(Path(work))
    out = {}
    for n in names:
        try:
            out[n] = mask_sig(store, n)
        except Exception:
            out[n] = ""
    return out


def exclusion_spec(work: Optional[Path], objects: Sequence[str], frames: Sequence[int],
                   dilate_px: int = DEFAULT_DILATE_PX, box: Optional[Sequence[int]] = None,
                   user_mask: Optional[str] = None) -> Dict[str, Any]:
    """What `AlignmentPipeline.measure_sequence(exclusion=...)` takes: objects (+ their mask signatures and the
    per-frame PNG paths), the dilation, a box [x0, y0, x1, y1] (plate px) and a user mask PNG (white = keep)."""
    names = [n for n in objects if n]
    paths: Dict[int, List[str]] = {}
    if work is not None and names:
        for f in frames:
            ps = [str(p) for p in (core_png(Path(work), n, int(f)) for n in names) if p is not None]
            if ps:
                paths[int(f)] = ps
    return {"objects": names, "object_sigs": object_sigs(work, names) if (work is not None and names) else {},
            "dilate_px": int(dilate_px), "box": [int(v) for v in box] if box else None,
            "user_mask": str(user_mask) if user_mask else None, "paths": paths}


def report_fields(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The part of the spec stored in the alignment report (no per-frame paths)."""
    if not spec:
        return {}
    return {k: spec.get(k) for k in ("objects", "object_sigs", "dilate_px", "box", "user_mask")}


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    if px <= 0 or not m.any():
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(px) + 1, 2 * int(px) + 1))
    return cv2.dilate(m, k).astype(bool)


def frame_exclusion(spec: Optional[Dict[str, Any]], f: int, w: int, h: int) -> Optional[np.ndarray]:
    """(h, w) bool, True = excluded, of frame f: the union of the objects' masks dilated + the user mask's black.
    None when the spec excludes nothing on this frame (the box is applied by compute_static_mask itself)."""
    if not spec:
        return None
    out = None
    paths = spec.get("paths") or {}
    ps = paths.get(int(f)) or paths.get(str(int(f))) or []
    if ps:
        u = np.zeros((h, w), np.uint8)
        for p in ps:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            u |= (m > 127).astype(np.uint8)
        out = dilate(u, int(spec.get("dilate_px") or 0))
    um = spec.get("user_mask")
    if um:
        m = cv2.imread(str(um), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            keep = m > 127
            out = (~keep) if out is None else (out | ~keep)
    return out


def union_exclusion(spec: Optional[Dict[str, Any]], frames: Sequence[int], w: int, h: int,
                    cancel_check=None) -> Optional[np.ndarray]:
    """OR of `frame_exclusion` over `frames`: everything an object ever covered (the zone it passed through)."""
    if not spec:
        return None
    out = None
    for f in frames:
        if cancel_check:
            cancel_check()
        m = frame_exclusion(spec, int(f), w, h)
        if m is not None:
            out = m if out is None else (out | m)
    return out
