# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Target background plate ("bg") for the quick comp preview: the cut-out (beauty RGB x object alpha) over the
background the shot will be comped on. Preview only – nothing is computed from it and it is never exported.

The source may be a single image (EXR / PNG / JPG …), an image sequence or a video file (MP4 / MOV …). Its frames
are mapped onto the shot: a single image = every frame; a sequence = the same frame number when the sequence covers
it, else counted from the sequence start (shot first frame = sequence first frame), clamped to the last one; a video
= frame index from the shot start, clamped. Every frame is fitted to the beauty proxy size (cover + centre crop).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from engine.ingest import VIDEO_EXT, MediaInspector, ProxyGenerator

BG_PLATE = "bg"


def source_info(pattern: str) -> Dict[str, Any]:
    """{kind: single|seq|video, pattern, first, last, count} of a background source (no image decoding)."""
    pattern = (pattern or "").strip().replace("\\", "/")
    if not pattern:
        return {"kind": "none", "pattern": "", "count": 0}
    if Path(pattern).suffix.lower() in VIDEO_EXT:
        meta = MediaInspector.read_metadata(pattern) if Path(pattern).is_file() else {}
        n = int(meta.get("frame_count") or 0)
        return {"kind": "video", "pattern": pattern, "first": 0, "last": max(0, n - 1), "count": n}
    if "#" in pattern or "%0" in pattern:
        info = MediaInspector.resolve_sequence_pattern(pattern)
        first, last = info.get("first_frame"), info.get("last_frame")
        if first is not None and last is not None:
            return {"kind": "seq", "pattern": pattern, "first": int(first), "last": int(last),
                    "count": int(info.get("frame_count") or 0)}
    return {"kind": "single", "pattern": pattern, "first": 0, "last": 0, "count": 1 if Path(pattern).is_file() else 0}


def source_frame(src: Dict[str, Any], frame: int, shot_first: int) -> int:
    """Frame number (seq) / index (video) of the background for shot frame `frame`; 0 for a single image."""
    kind = src.get("kind")
    if kind == "seq":
        first, last = int(src["first"]), int(src["last"])
        n = frame if first <= frame <= last else first + (frame - shot_first)
        return int(min(max(n, first), last))
    if kind == "video":
        return int(min(max(frame - shot_first, 0), max(0, int(src.get("count") or 1) - 1)))
    return 0


def fit_cover(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale to cover w x h (aspect kept) and crop the centre."""
    sh, sw = rgb.shape[:2]
    s = max(w / float(sw), h / float(sh))
    nw, nh = max(w, int(round(sw * s))), max(h, int(round(sh * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    r = cv2.resize(rgb, (nw, nh), interpolation=interp) if (nw, nh) != (sw, sh) else rgb
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])


def load_source_rgb(src: Dict[str, Any], n: int, colorspace: str) -> np.ndarray:
    """Display RGB uint8 of background frame `n` (same conversion as the plate proxies)."""
    kind = src.get("kind")
    if kind == "video":
        cap = cv2.VideoCapture(src["pattern"])
        try:
            if n:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
            ok, bgr = cap.read()
        finally:
            cap.release()
        if not ok or bgr is None:
            raise FileNotFoundError(f"Brak klatki {n} w {src['pattern']}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    path = ProxyGenerator.resolve_frame_path(src["pattern"], n) if kind == "seq" else Path(src["pattern"])
    if not Path(path).exists():
        raise FileNotFoundError(f"Brak pliku tła: {path}")
    # a single non-EXR image may be in any colour space; EXR follows the shot colour space like the plates
    return ProxyGenerator.load_frame_rgb(path, colorspace=colorspace)


def cache_dir(work_dir: Path, pattern: str, size: Tuple[int, int]) -> Path:
    """Proxy cache of one background source at one size (a new source / size = a new folder, no stale frames)."""
    h = hashlib.sha1(f"{pattern}|{size[0]}x{size[1]}".encode("utf-8")).hexdigest()[:10]
    return Path(work_dir) / "proxy" / BG_PLATE / h


def proxy_path(work_dir: Path, pattern: str, frame: int, shot_first: int, size: Tuple[int, int],
               colorspace: str, src: Optional[Dict[str, Any]] = None) -> Path:
    """JPEG of the background for shot frame `frame`, fitted to `size` (w, h) – rendered on demand, then cached."""
    src = src or source_info(pattern)
    if not src.get("count"):
        raise FileNotFoundError(f"Tło niedostępne: {pattern}")
    n = source_frame(src, frame, shot_first)
    out = cache_dir(work_dir, pattern, size) / f"bg_{n:06d}.jpg"
    if out.exists():
        return out
    rgb = fit_cover(load_source_rgb(src, n, colorspace), int(size[0]), int(size[1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    tmp.replace(out)
    return out
