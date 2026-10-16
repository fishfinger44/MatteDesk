# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/exr_io.py - OpenEXR read/write that keeps the source encoding (SPEC 4c.3, 9).

"Zapis w kodowaniu źródła": same channels, pixel types (half/float), compression, metadata
(chromaticities, custom attributes) and data window as the source frame. Uses the legacy
OpenEXR Python API (InputFile/OutputFile), available in OpenEXR 3.x.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import Imath
import OpenEXR

_PT = {
    "HALF": (Imath.PixelType(Imath.PixelType.HALF), np.float16),
    "FLOAT": (Imath.PixelType(Imath.PixelType.FLOAT), np.float32),
    "UINT": (Imath.PixelType(Imath.PixelType.UINT), np.uint32),
}


def _ptype_name(ch: "Imath.Channel") -> str:
    return str(ch.type)


def read_exr(path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], dict]:
    """Returns ({channel: float32 HxW array}, header). Header is the OpenEXR header dict (reusable for writing)."""
    f = OpenEXR.InputFile(str(path))
    try:
        hdr = f.header()
        dw = hdr["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        out = {}
        for name, ch in hdr["channels"].items():
            pt, dt = _PT[_ptype_name(ch)]
            out[name] = np.frombuffer(f.channel(name, pt), dtype=dt).reshape(h, w).astype(np.float32)
        return out, hdr
    finally:
        f.close()


def rgb_channel_names(channels: Dict[str, np.ndarray]) -> List[str]:
    """Names of the R, G, B channels (plain or layered like `rgba.R`)."""
    names = []
    for c in ("R", "G", "B"):
        cand = [n for n in channels if n == c or n.endswith("." + c)]
        if not cand:
            raise KeyError(f"no {c} channel in {sorted(channels)}")
        names.append(sorted(cand, key=len)[0])
    return names


def read_rgb(path: Union[str, Path]) -> np.ndarray:
    """HxWx3 float32 in **BGR** order (same as cv2.imread) for interop with the rest of the engine."""
    ch, _ = read_exr(path)
    r, g, b = rgb_channel_names(ch)
    return np.dstack([ch[b], ch[g], ch[r]])


def write_exr_like(path: Union[str, Path], channels: Dict[str, np.ndarray], src_header: dict,
                   extra_attrs: Optional[dict] = None) -> None:
    """Write `channels` using the source header (pixel types, compression, metadata) unchanged."""
    hdr = dict(src_header)
    if extra_attrs:
        hdr.update(extra_attrs)
    data = {}
    for name, ch in hdr["channels"].items():
        _, dt = _PT[_ptype_name(ch)]
        data[name] = np.ascontiguousarray(channels[name]).astype(dt).tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o = OpenEXR.OutputFile(str(path), hdr)
    try:
        o.writePixels(data)
    finally:
        o.close()


def write_rgb_like(path: Union[str, Path], bgr: np.ndarray, src_path: Union[str, Path],
                   extra_attrs: Optional[dict] = None) -> None:
    """Replace R, G, B of the source frame by `bgr` (cv2 order); every other channel (alpha, layers) is copied."""
    ch, hdr = read_exr(src_path)
    r, g, b = rgb_channel_names(ch)
    ch[b], ch[g], ch[r] = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    write_exr_like(path, ch, hdr, extra_attrs)


def write_gray_exr(path: Union[str, Path], data: np.ndarray, half: bool = True,
                   channels: Tuple[str, ...] = ("R", "G", "B"), compression: str = "ZIP_COMPRESSION",
                   extra_attrs: Optional[dict] = None) -> None:
    """Single-plane image written into the given channels (e.g. alpha as rgb=a, or depth.disparity)."""
    h, w = data.shape[:2]
    hdr = OpenEXR.Header(w, h)
    pt = Imath.PixelType(Imath.PixelType.HALF if half else Imath.PixelType.FLOAT)
    hdr["channels"] = {c: Imath.Channel(pt) for c in channels}
    hdr["compression"] = Imath.Compression(getattr(Imath.Compression, compression))
    if extra_attrs:
        hdr.update(extra_attrs)
    buf = np.ascontiguousarray(data).astype(np.float16 if half else np.float32).tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o = OpenEXR.OutputFile(str(path), hdr)
    try:
        o.writePixels({c: buf for c in channels})
    finally:
        o.close()


# ----------------------------------------------------------------------------- fast reading (Z30)
# One OpenEXR `channels()` call decodes every requested channel in a single pass (ZIP/ZIPS decompress whole
# scanline blocks, so separate `channel()` calls – or cv2.imread + float conversion – decode the file several
# times). Optional scanline range: only the blocks covering rows y0..y1 are decompressed. Half data stays half
# (exact; float16 -> float32 is lossless), so `display_lut` can map it bit-identically to the float formula.

def _simple_layout(hdr: dict) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, w, h) of the data window when it equals the display window (else None -> caller falls back)."""
    dw, dsp = hdr["dataWindow"], hdr.get("displayWindow", hdr["dataWindow"])
    if (dw.min.x, dw.min.y, dw.max.x, dw.max.y) != (dsp.min.x, dsp.min.y, dsp.max.x, dsp.max.y):
        return None
    return dw.min.x, dw.min.y, dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1


def read_half_rgb(path: Union[str, Path], y0: Optional[int] = None, y1: Optional[int] = None,
                  alpha: bool = False) -> Optional[np.ndarray]:
    """BGR(A) H x W x 3|4 array like `cv2.imread(path, IMREAD_UNCHANGED)` for plain R/G/B(/A) EXR files, read with
    one OpenEXR call. dtype float16 when all read channels are HALF (lossless), else float32. Rows [y0, y1)
    (image coordinates) only, when given. `alpha=True` adds the A channel when present (cv2 order BGRA).
    Returns None for layouts it does not mirror exactly (no plain R/G/B, data window != display window,
    non-float pixel types) – the caller then uses cv2.imread."""
    f = OpenEXR.InputFile(str(path))
    try:
        hdr = f.header()
        lay = _simple_layout(hdr)
        chs = hdr["channels"]
        if lay is None or not all(c in chs for c in ("R", "G", "B")):
            return None
        names = ["B", "G", "R"] + (["A"] if alpha and "A" in chs else [])
        types = {_ptype_name(chs[c]) for c in names}
        if not types <= {"HALF", "FLOAT"}:
            return None
        half = types == {"HALF"}
        pt, dt = _PT["HALF" if half else "FLOAT"]
        x0, top, w, h = lay
        a = 0 if y0 is None else max(0, int(y0))
        b = h if y1 is None else min(h, int(y1))
        if b <= a:
            return np.zeros((0, w, len(names)), dt)
        bufs = f.channels(names, pt, top + a, top + b - 1)
        return np.dstack([np.frombuffer(x, dtype=dt).reshape(b - a, w) for x in bufs])
    finally:
        f.close()


def imread_unchanged(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Drop-in for `cv2.imread(path, cv2.IMREAD_UNCHANGED)` with the fast EXR reader (float32, same channels)."""
    import cv2
    p = Path(path)
    if p.suffix.lower() == ".exr" and p.exists():
        try:
            img = read_half_rgb(p, alpha=True)
        except Exception:
            img = None
        if img is not None:
            return img.astype(np.float32)
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


_LUTS: Dict[float, np.ndarray] = {}


def display_lut(white: float = 1.0) -> np.ndarray:
    """PlateToDisplay (`engine.ingest.plate_to_display_float`) for every one of the 65 536 half values, float32.
    `display(x_half) = lut[x_half.view(uint16)]` – identical to the formula on float32(x_half)."""
    lut = _LUTS.get(float(white))
    if lut is None:
        from engine.ingest import plate_to_display_float
        allh = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):   # NaN / Inf entries behave as in the formula
            lut = _LUTS[float(white)] = plate_to_display_float(allh, white)
    return lut


def display_half(x: np.ndarray, white: float = 1.0) -> np.ndarray:
    """PlateToDisplay of a float16 array through the LUT (float32 result)."""
    return display_lut(white)[np.ascontiguousarray(x, np.float16).view(np.uint16)]


_LUTS_U8: Dict[float, np.ndarray] = {}


def display_half_u8_rgb(bgr: np.ndarray, white: float = 1.0) -> np.ndarray:
    """`(display_half(bgr) * 255.0).astype(uint8)[..., ::-1]` (contiguous RGB) in one gather: the same
    elementwise float32 ops applied once to all 65 536 half values, channel flip done by the index view."""
    lut8 = _LUTS_U8.get(float(white))
    if lut8 is None:
        with np.errstate(invalid="ignore"):
            lut8 = _LUTS_U8[float(white)] = (display_lut(white) * 255.0).astype(np.uint8)
    return lut8[np.ascontiguousarray(bgr, np.float16).view(np.uint16)[..., ::-1]]


# ----------------------------------------------------------------------------- QA gray from the raw EXR (Z27 2b)
LOG_LUMA_RANGE = (-12.0, 4.0)   # log2 scene luminance mapped to 0..255 (16 stops, 1/16 EV per code value)


def log_luma(bgr: np.ndarray, lo: float = LOG_LUMA_RANGE[0], hi: float = LOG_LUMA_RANGE[1]) -> np.ndarray:
    """Scene-linear BGR -> log2 luminance (Rec.709 weights over the three channels – AP0 channels are strongly
    correlated, the mix is more robust to noise than one channel) on a FIXED scale 0..255, float16 (spacing
    <= 0.125 code = 0.008 EV; 2 B / px like the former uint8 display gray x 2). Fixed (not per frame), so the camera
    shift / flow / clear-plate measurements compare like with like between frames and plates."""
    import cv2
    x = np.ascontiguousarray(bgr, dtype=np.float32)
    lum = cv2.transform(x, np.array([[0.0722, 0.7152, 0.2126]], np.float32))   # one pass, ~15x faster than numpy
    lum = lum.reshape(x.shape[:2])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        np.fmax(lum, np.float32(2.0 ** lo), out=lum)        # fmax: NaN -> floor; +Inf stays (clipped to 255 below)
        np.log2(lum, out=lum)
        lum -= np.float32(lo)
        lum *= np.float32(255.0 / (hi - lo))
    np.clip(lum, 0, 255, out=lum)
    return lum.astype(np.float16)


def read_log_luma_rows(path: Union[str, Path], y0: Optional[int] = None, y1: Optional[int] = None) -> Optional[np.ndarray]:
    """float16 log-luminance (0..255 scale) of rows [y0, y1) of a plain RGB EXR (one OpenEXR call, only those
    scanlines decoded); None when the layout is not supported (caller falls back)."""
    img = read_half_rgb(path, y0, y1)
    return None if img is None else log_luma(img)
