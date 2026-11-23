# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/core_import.py - the user's external mask as the stage-1 core (user request 2026-11-20).

Instead of SAM 3.1 the core comes from an alpha sequence the user made elsewhere (Nuke roto, another keyer): stage 1
writes exactly what run_core writes, so every later stage reads it unchanged:
  core/c_####.png      binary core (alpha > threshold), empty outside the object's life
  core_soft/p_####.png the alpha as 8-bit "probability" (soft start of the band, probability mode of the stabilisation)
  core/core.json       coverage, life, keys_sig + `source: {kind: import, ...}` (stage 1 done; denoise / plate repair
                       never make it stale – the mask does not come from the plate)
  crops.json           crop segments planned from the imported masks' boxes (engine.crop plan_segments, no SAM)
  keys/                one key per crop segment (its first visible frame with a mask) = MatAnyone2 anchors
The temporal stabilisation of the core is NOT computed here: stages 3 / 4 / 5 / 7 and the export compute it on first
use (pipeline.core_with_soft -> stabilized_core, cached per signature) – the import itself stays a plain file copy.
"Przelicz nieaktualne" (SAM 3.1) refuses an imported core (it would overwrite it): „Wróć do SAM 3.1” removes it first.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from engine.jsonio import write_json

IMPORT_KIND = "import"


def core_source(obj_dir: Path) -> Optional[Dict[str, Any]]:
    """`source` of core/core.json when the core was imported (None = SAM 3.1 core / no core)."""
    try:
        src = json.loads((Path(obj_dir) / "core" / "core.json").read_text(encoding="utf-8")).get("source")
    except (OSError, ValueError):
        return None
    return src if isinstance(src, dict) and src.get("kind") == IMPORT_KIND else None


def resolve(path: str) -> Dict[str, Any]:
    """The sequence behind a folder / a frame file / a pattern (#### or %04d): {pattern, frames: {n: path}}.
    ValueError with the reason (nothing found, a video)."""
    from engine.ingest import MediaInspector, is_video_file
    raw = (path or "").strip().strip('"')
    if not raw:
        raise ValueError("podaj folder, plik klatki albo wzorzec sekwencji maski")
    if is_video_file(raw):
        raise ValueError("wideo nie jest obsługiwane – wczytaj sekwencję klatek (EXR / PNG / TIFF)")
    info = MediaInspector.resolve_sequence_pattern(raw)
    pat = info.get("pattern") or ""
    if not info.get("exists") or "#" not in pat:
        if info.get("exists") and Path(raw).is_file():   # one file without a frame number
            raise ValueError("plik bez numeru klatki – potrzebna sekwencja (np. maska.####.exr)")
        raise ValueError(f"nie znaleziono sekwencji maski: {raw}")
    p = Path(pat)
    prefix, suffix = p.name.split("#")[0], p.name.split("#")[-1]
    frames: Dict[int, str] = {}
    for f in p.parent.iterdir():
        mid = f.name[len(prefix):len(f.name) - len(suffix)] if f.name.startswith(prefix) and f.name.endswith(suffix) else ""
        if mid.isdigit() and f.is_file():
            frames[int(mid)] = str(f)
    if not frames:
        raise ValueError(f"sekwencja bez klatek: {pat}")
    return {"pattern": pat, "frames": frames}


def frame_map(src: Sequence[int], need: Sequence[int], shot_first: int) -> Dict[str, Any]:
    """Shot frame -> source frame. The same numbers when the source has every needed frame; else, a source as long as
    the needed range numbered from elsewhere (1001…, 0…) shifted by its first frame. ValueError with what is missing."""
    s = sorted(set(int(f) for f in src))
    need = sorted(int(f) for f in need)
    if not need:
        raise ValueError("obiekt nie jest widoczny w żadnej klatce ujęcia (zakres życia)")
    have = set(s)
    if all(f in have for f in need):
        return {"offset": 0, "map": {f: f for f in need}}
    off = s[0] - int(shot_first)
    if all(f + off in have for f in need):
        return {"offset": off, "map": {f: f + off for f in need}}
    miss = [f for f in need if f not in have]
    raise ValueError(f"maska nie ma klatek ujęcia: brak {len(miss)} (np. F{miss[0]}"
                     f"{'–' + str(miss[-1]) if len(miss) > 1 else ''}); sekwencja ma F{s[0]}–{s[-1]} ({len(s)} kl.)")


def _norm(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def read_mask(path: str) -> np.ndarray:
    """Alpha of one mask frame as float 0..1: the A channel when it carries the mask, else R (Nuke writes a matte
    as rgb with a constant alpha just as often), a single channel as it is."""
    a = None
    if path.lower().endswith(".exr"):
        try:
            from engine.refine import _read_alpha_exr
            a = _read_alpha_exr(Path(path))
        except Exception:
            a = None
        if a is not None and float(a.max() - a.min()) > 1e-6:
            return np.clip(a, 0.0, 1.0)
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        if a is not None:
            return np.clip(a, 0.0, 1.0)
        raise ValueError(f"nie da się odczytać klatki maski: {path}")
    m = _norm(m)
    if m.ndim == 2:
        return np.clip(m, 0.0, 1.0)
    if m.shape[2] == 4 and float(m[..., 3].max() - m[..., 3].min()) > 1e-6:
        return np.clip(m[..., 3], 0.0, 1.0)
    return np.clip(m[..., 2], 0.0, 1.0)      # BGR(A) -> R


def fit(a: np.ndarray, w: int, h: int, path: str = "") -> np.ndarray:
    """The mask at plate size: the same aspect (±1 %) is scaled (a proxy-size roto), another aspect is refused."""
    mh, mw = a.shape[:2]
    if (mw, mh) == (w, h):
        return a
    if abs(mw / mh - w / h) > 0.01 * (w / h):
        raise ValueError(f"maska {mw}×{mh} ma inne proporcje niż plate {w}×{h}" + (f" ({Path(path).name})" if path else ""))
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if mw > w else cv2.INTER_LINEAR)


def import_core(store: Any, name: str, path: str, shot: Sequence[int], w: int, h: int,
                threshold: float = 0.5, crop_cfg: Optional[Dict[str, Any]] = None, thin_parts: bool = False,
                for_ltx: bool = False, key_history: Any = None,
                progress_cb: Optional[Callable[[str, int, int], None]] = None,
                cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """Write the external mask `path` as the core of object `name` (see the module doc). `key_history` = a
    KeyHistory of the object (keys replaced undoably), None = the store directly. Returns a summary for the UI."""
    from engine.crop import mask_bbox, plan_segments, save_crops, smooth_bboxes
    from engine.stages import keys_signature
    from engine.workarea import intervals_from_frames, is_visible, object_life
    shot = [int(f) for f in shot]
    obj = store.get(name)
    life = object_life(obj)
    vis = [f for f in shot if is_visible(f, life)]
    seq = resolve(path)
    fm = frame_map(list(seq["frames"]), vis, shot[0])
    od = store.dir(name)
    thr = float(min(0.99, max(0.01, threshold)))

    # 1. read every visible frame (alpha -> soft 8-bit + binary core) – nothing on disk changes until all are read
    soft: Dict[int, np.ndarray] = {}
    boxes: List[Optional[tuple]] = []
    empty: List[int] = []
    for i, f in enumerate(shot):
        if cancel_check:
            cancel_check()
        if f not in fm["map"]:
            boxes.append(None)
            continue
        src = seq["frames"][fm["map"][f]]
        a = fit(read_mask(src), w, h, src)
        p8 = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
        soft[f] = p8
        b = mask_bbox(a, thr)
        boxes.append(b)
        if b is None:
            empty.append(f)
        if progress_cb:
            progress_cb("Wczytuję maskę", i + 1, len(shot))
    if len(empty) == len(vis):
        raise ValueError(f"maska jest pusta we wszystkich klatkach (próg {thr:g}) – zły kanał albo zły plik?")

    # 2. crops from the mask boxes (the same smoothing + segment plan as the SAM 3.1 crop engine)
    segs = plan_segments(smooth_bboxes(boxes), shot[0], w, h, crop_cfg, thin_parts=thin_parts, for_ltx=for_ltx)
    if not segs:
        raise ValueError("nie da się wyznaczyć kadrów z maski")
    save_crops(od / "crops.json", segs, {"flags": {}, "bboxes": {str(f): b for f, b in zip(shot, boxes)},
                                         "source": IMPORT_KIND})

    # 3. keys: the object's old keys go (undoable with the key history), one key per crop segment
    kh = key_history
    for f in list(store.keys(name)):
        (kh.delete_key(f) if kh is not None else store.delete_key(name, f))
    key_frames: List[int] = []
    for sg in segs:
        f = next((f for f in range(sg.first, sg.last + 1) if f in soft and f not in empty), None)
        if f is not None and f not in key_frames:
            key_frames.append(f)
    for f in key_frames:
        m = soft[f] > round(thr * 255)
        (kh.save_key(f, m, author="user", kind="mask", source=IMPORT_KIND) if kh is not None
         else store.save_key(name, f, m, author="user", kind="mask", source=IMPORT_KIND))

    # 4. core + soft maps (stale SAM 3.1 probabilities removed), core.json last = stage 1 done
    cdir, sdir = od / "core", od / "core_soft"
    cdir.mkdir(parents=True, exist_ok=True)
    if sdir.exists():
        for p in sdir.glob("p_*.png"):
            p.unlink()
    sdir.mkdir(parents=True, exist_ok=True)
    zero = np.zeros((h, w), np.uint8)
    t8 = round(thr * 255)
    for i, f in enumerate(shot):
        if f in soft:
            cv2.imwrite(str(cdir / f"c_{f:04d}.png"), (soft[f] > t8).astype(np.uint8) * 255)
            cv2.imwrite(str(sdir / f"p_{f:04d}.png"), soft[f])
        else:
            cv2.imwrite(str(cdir / f"c_{f:04d}.png"), zero)
        if progress_cb:
            progress_cb("Zapisuję rdzeń", i + 1, len(shot))
    src_info = {"kind": IMPORT_KIND, "path": str(path), "pattern": seq["pattern"], "offset": fm["offset"],
                "threshold": thr, "t": datetime.now().isoformat(timespec="seconds"), "empty_frames": len(empty)}
    write_json(cdir / "core.json", {
        "fwd_bwd_iou": {}, "uncertain": [], "suggested_keys": [], "recomputed": [], "cached": [],
        "keys_sig": keys_signature(store, name), "soft_frames": len(soft), "correction_regions": {},
        "coverage": [[shot[0], shot[-1]]], "frames": [shot[0], shot[-1]], "life": life, "life_hints": [],
        "front_missing": [], "source": src_info}, indent=1)
    store.update(name, tool={"status": "rdzeń z maski zewnętrznej", "final_valid": False,
                             "start_frame": key_frames[0] if key_frames else None})
    return {"frames": [shot[0], shot[-1]], "visible": len(vis), "empty": intervals_from_frames(empty),
            "segments": len(segs), "keys": key_frames, "pattern": seq["pattern"], "offset": fm["offset"],
            "threshold": thr}


def remove_import(store: Any, name: str) -> int:
    """„Wróć do SAM 3.1”: the imported core (core/, core_soft/) removed from disk for good; the keys stay (SAM 3.1
    propagates from them). Returns the bytes freed. ValueError when the core is not imported."""
    from engine.deletion import remove
    od = store.dir(name)
    if core_source(od) is None:
        raise ValueError("rdzeń obiektu nie pochodzi z maski zewnętrznej")
    freed = remove(od / "core", od / "core_soft", inside=od)
    store.update(name, tool={"status": "rdzeń usunięty (maska zewnętrzna)", "final_valid": False})
    return freed
