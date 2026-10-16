# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/plate_fix.py - Z26: repair of damaged frames and the written `<plate>_fix` sequence.

Methods (per frame, chosen by the user from the plate QA report):
- `mean`  : average of t-1 and t+1 (a missing / damaged frame with little motion – how SH010 X2 F162 was fixed),
- `flow`  : motion-compensated interpolation: DIS flow between t-1 and t+1, both neighbours warped half-way, averaged
            (large motion); falls back to `mean` when a neighbour is missing,
- `gain`  : the frame itself with a per-channel gain so its channel means match the mean of the neighbours (exposure jump),
- `keep`  : no change (decided consciously).
`repair_frame` returns the repaired BGR float frame (same values / encoding family as the source: linear EXR or 0..1
display); `write_fixed_plate` writes `<dir>_fix/` next to the source: repaired frames in the SOURCE encoding (OpenEXR:
same header / channels / compression, other channels of the source frame copied; PNG/TIFF: same bit depth), every
other frame an NTFS hard link (copy when linking fails), README.txt with the list, and returns the new pattern.
Nothing is written without `confirmed=True` (the caller asks the user; EXR = GBs).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.plate_qa import frame_path, read_frame_linear

METHODS = ("mean", "flow", "gain", "keep")


def _neighbours(f: int, frames: Sequence[int], ok: Callable[[int], bool]) -> Tuple[Optional[int], Optional[int]]:
    fr = list(frames)
    i = fr.index(f)
    prev_ = next((fr[j] for j in range(i - 1, -1, -1) if ok(fr[j])), None)
    next_ = next((fr[j] for j in range(i + 1, len(fr)) if ok(fr[j])), None)
    return prev_, next_


def flow_interpolate(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Frame between a (t=0) and b (t=1): DIS flow both ways on gray, each neighbour warped to t, averaged."""
    ga = cv2.cvtColor(np.clip(a, 0, None).astype(np.float32), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(np.clip(b, 0, None).astype(np.float32), cv2.COLOR_BGR2GRAY)
    u8 = lambda g: cv2.normalize(np.log1p(g * 4.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f_ab = dis.calc(u8(ga), u8(gb), None)      # for each pixel of a: where it is in b
    f_ba = dis.calc(u8(gb), u8(ga), None)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # linear motion: the middle-frame pixel p was at p - t*v in a and lands at p + (1-t)*v in b; v approximated by
    # the flow evaluated at p (backward sampling of each side)
    wa = cv2.remap(a, gx - t * f_ab[..., 0], gy - t * f_ab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx - (1 - t) * f_ba[..., 0], gy - (1 - t) * f_ba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ((1 - t) * wa + t * wb).astype(np.float32)


def repair_frame(f: int, method: str, frames: Sequence[int], load: Callable[[int], Optional[np.ndarray]],
                 ok: Optional[Callable[[int], bool]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Repaired BGR float frame (None for `keep`) + info. `ok(g)` = frame g is usable as a neighbour."""
    if method not in METHODS:
        raise ValueError(f"metoda naprawy: {'|'.join(METHODS)}")
    if method == "keep":
        return None, {"method": "keep"}
    ok = ok or (lambda g: True)
    p, q = _neighbours(f, frames, lambda g: g != f and ok(g))
    a = load(p) if p is not None else None
    b = load(q) if q is not None else None
    if method == "gain":
        img = load(f)
        if img is None:
            raise ValueError(f"klatka {f}: brak pliku – wzmocnienie niemożliwe, użyj mean / flow")
        refs = [x for x in (a, b) if x is not None]
        if not refs:
            raise ValueError(f"klatka {f}: brak sąsiadów do wzmocnienia")
        target = np.mean([x.reshape(-1, 3).mean(axis=0) for x in refs], axis=0)
        own = img.reshape(-1, 3).mean(axis=0)
        gain = target / np.maximum(own, 1e-6)
        return (img * gain[None, None, :]).astype(np.float32), {"method": "gain", "gain": [round(float(g), 4) for g in gain],
                                                                  "neighbours": [p, q]}
    if a is None and b is None:
        raise ValueError(f"klatka {f}: brak sąsiadów do interpolacji")
    if a is None or b is None:
        return (a if a is not None else b).astype(np.float32), {"method": "copy", "neighbours": [p, q]}
    if method == "flow":
        t = 0.5 if (q - f) == (f - p) else (f - p) / float(q - p)
        return flow_interpolate(a, b, t), {"method": "flow", "neighbours": [p, q], "t": round(t, 3)}
    w = (f - p) / float(q - p)
    return ((1 - w) * a + w * b).astype(np.float32), {"method": "mean", "neighbours": [p, q], "t": round(w, 3)}


def fixed_dir_for(pattern: str) -> Tuple[Path, str]:
    """`<dir>_fix/` next to the source directory + the new pattern (same file names)."""
    src = Path(pattern)
    d = src.parent
    new_dir = d.with_name(d.name + "_fix")
    return new_dir, str(new_dir / src.name).replace("\\", "/")


def write_fixed_plate(pattern: str, frames: Sequence[int], decisions: Dict[int, str], confirmed: bool,
                      progress: Optional[Callable[[str, int, int], None]] = None,
                      ok_frames: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Write `<dir>_fix/`: repaired frames (decisions {frame: method}) in the source encoding, the rest hard-linked.
    Refuses without `confirmed` (returns the plan with the size estimate)."""
    frames = list(frames)
    new_dir, new_pattern = fixed_dir_for(pattern)
    to_fix = {int(f): m for f, m in decisions.items() if m in METHODS and m != "keep"}
    bad = set(to_fix)
    okset = set(ok_frames) if ok_frames is not None else set(frames) - bad
    src0 = frame_path(pattern, frames[0])
    size0 = src0.stat().st_size if src0.exists() else 0
    exists = new_dir.exists() and any(new_dir.iterdir())
    ours = (new_dir / "plate_fix.json").exists()
    plan = {"dir": str(new_dir), "pattern": new_pattern, "repair": to_fix, "hardlinks": len(frames) - len(to_fix),
            "bytes_written_estimate": size0 * len(to_fix), "confirmed": bool(confirmed),
            "exists": exists, "ours": ours, "eta_s": estimate_seconds(pattern, frames, to_fix)}
    if exists and not ours:
        # a `_fix` folder made outside MatteDesk (e.g. by hand in Nuke) is never overwritten
        raise FileExistsError(f"folder {new_dir} już istnieje i nie pochodzi z MatteDesk (brak plate_fix.json) – "
                              "nie nadpisuję; przenieś go albo wskaż ten plate w oknie plate'ów")
    if not confirmed:
        return {"success": False, "plan": plan, "detail": "zapis plate'u _fix wymaga potwierdzenia użytkownika"}
    load = lambda g: read_frame_linear(frame_path(pattern, g))
    new_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done: Dict[int, Dict[str, Any]] = {}
    links = copies = 0
    for i, f in enumerate(frames):
        if progress:
            progress("Zapis plate'u _fix", i + 1, len(frames))
        src = frame_path(pattern, f)
        dst = new_dir / src.name
        if dst.exists():
            dst.unlink()
        if f in to_fix:
            img, info = repair_frame(f, to_fix[f], frames, load, ok=lambda g: g in okset)
            if img is None:
                continue
            _write_like_source(src, dst, img, pattern, frames, f)
            done[f] = info
            continue
        try:
            os.link(src, dst)
            links += 1
        except OSError:
            shutil.copyfile(src, dst)
            copies += 1
    readme = [f"MatteDesk plate fix – {datetime.now().isoformat(timespec='seconds')}", f"source: {pattern}",
              f"frames: {frames[0]}–{frames[-1]} ({len(frames)}); repaired {len(done)}; hard links {links}; copies {copies}", ""]
    for f, info in sorted(done.items()):
        readme.append(f"frame {f}: {json.dumps(info)}")
    (new_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (new_dir / "plate_fix.json").write_text(json.dumps({"source": pattern, "pattern": new_pattern, "repaired": {str(k): v for k, v in done.items()},
                                                         "hardlinks": links, "copies": copies, "t": datetime.now().isoformat(timespec="seconds")},
                                                        indent=1), encoding="utf-8")
    return {"success": True, "pattern": new_pattern, "dir": str(new_dir), "repaired": done, "hardlinks": links, "copies": copies,
            "seconds": round(time.time() - t0, 1), "plan": plan}


def estimate_seconds(pattern: str, frames: Sequence[int], to_fix: Dict[int, str]) -> float:
    """Rough ETA of write_fixed_plate: one source frame is read and timed; a repaired frame costs ~4 reads (two
    neighbours, the frame / the reference header, the write), `flow` adds DIS on the full frame; a hard link ~2 ms."""
    t_read, px = 0.0, 0
    for g in frames:
        p = frame_path(pattern, g)
        if p.exists():
            t0 = time.time()
            img = read_frame_linear(p)
            t_read = time.time() - t0
            px = int(img.shape[0] * img.shape[1]) if img is not None else 0
            break
    flow = sum(1 for m in to_fix.values() if m == "flow")
    return round(len(to_fix) * 4 * t_read + flow * 4.0 * px / 11.2e6 + 0.002 * (len(frames) - len(to_fix)), 1)


def preview_repair(pattern: str, frames: Sequence[int], f: int, method: str, side: int = 1280,
                   flagged: Sequence[int] = ()) -> Optional[np.ndarray]:
    """Preview of the repair of frame `f` (BGR float, longest side <= `side`; `orig` = the source frame, None when
    missing). Computed on the downscaled neighbours – the written frame is computed at full resolution. Other flagged
    frames are not used as neighbours (same as the write)."""
    from engine.plate_qa import small_of
    cache: Dict[int, Optional[np.ndarray]] = {}

    def load(g: int) -> Optional[np.ndarray]:
        if g not in cache:
            img = read_frame_linear(frame_path(pattern, g))
            cache[g] = None if img is None else small_of(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0), side)
        return cache[g]
    if method == "orig" or method == "keep":
        return load(f)
    bad = set(int(x) for x in flagged) - {f}
    img, _ = repair_frame(f, method, frames, load, ok=lambda g: g not in bad and frame_path(pattern, g).exists())
    return img


def _write_like_source(src: Path, dst: Path, img_bgr: np.ndarray, pattern: str, frames: Sequence[int], f: int) -> None:
    """Repaired frame in the source encoding. EXR: header / channels of the source frame (if it exists, else of the
    nearest existing frame), R/G/B replaced, other channels copied. 8 / 16-bit: same bit depth."""
    ext = src.suffix.lower()
    if ext == ".exr":
        from engine.exr_io import read_exr, rgb_channel_names, write_exr_like
        ref = src if src.exists() else next((frame_path(pattern, g) for g in sorted(frames, key=lambda g: abs(g - f))
                                              if frame_path(pattern, g).exists()), None)
        if ref is None:
            raise FileNotFoundError("brak klatki wzorcowej EXR")
        ch, hdr = read_exr(ref)
        r, g, b = rgb_channel_names(ch)
        ch[b], ch[g], ch[r] = img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]
        write_exr_like(dst, ch, hdr)
        return
    ref_img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED) if src.exists() else None
    depth16 = ref_img is not None and ref_img.dtype == np.uint16
    if ref_img is None:   # missing source: pick the bit depth of the first existing frame
        for g in sorted(frames, key=lambda g: abs(g - f)):
            p = frame_path(pattern, g)
            if p.exists():
                ri = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                depth16 = ri is not None and ri.dtype == np.uint16
                break
    if depth16:
        cv2.imwrite(str(dst), np.clip(img_bgr * 65535.0 + 0.5, 0, 65535).astype(np.uint16))
    else:
        cv2.imwrite(str(dst), np.clip(img_bgr * 255.0 + 0.5, 0, 255).astype(np.uint8))
