# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/dpx.py - SMPTE 268M DPX reader in numpy (OpenCV has no DPX codec: cv2.imread returns None for .dpx).

`imread(path, flags)` answers like cv2.imread (BGR / BGRA / gray, 8-bit for IMREAD_COLOR / GRAYSCALE, the file's
depth for IMREAD_UNCHANGED / IMREAD_ANYDEPTH: 10 / 12 / 16-bit -> uint16 scaled to the full 16-bit range like a PNG16,
8-bit -> uint8) and is hooked into cv2.imread by engine/cv2_unicode.py for every path ending in .dpx, so the whole app
reads DPX plates through its usual calls. `info(path)` gives the header fields that matter for colour: the transfer
characteristic (printing density / logarithmic = Cineon-like log, linear, video curves), bit depth, packing, size.
No colour conversion happens here: the code values come as they are in the file (the project's colour space decides).

Supported: element 0 of the image, descriptors RGB (50), RGBA (51), ABGR (52), luma (6), alpha (4); bit depths 8, 10,
12, 16; 10-bit "filled" packing method A (padding in the 2 low bits – the common one) and method B (padding in the 2
high bits) and the contiguous "packed" bit stream; 12-bit filled (A / B) and packed; big ("SDPX") and little ("XPDS")
endian. Not supported (ValueError in Polish): run-length encoding, 1-bit / 32 / 64-bit float, other descriptors.
"""
from __future__ import annotations

import os
import struct
from typing import Any, Dict, Tuple

import numpy as np

TRANSFER = {0: "użytkownika", 1: "gęstość kopii (log)", 2: "liniowa", 3: "logarytmiczna", 4: "nieokreślona",
            5: "SMPTE 274M", 6: "ITU-R BT.709", 7: "ITU-R BT.601 (625)", 8: "ITU-R BT.601 (525)",
            9: "NTSC", 10: "PAL", 11: "Z liniowa", 12: "Z homogeniczna"}
LOG_TRANSFERS = (1, 3)                       # printing density, logarithmic = Cineon-like log code values
CHANNELS = {50: 3, 51: 4, 52: 4, 6: 1, 4: 1}


def _header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < 2048:
        raise ValueError("DPX: plik za krótki na nagłówek")
    magic = raw[:4]
    if magic == b"SDPX":
        e = ">"
    elif magic == b"XPDS":
        e = "<"
    else:
        raise ValueError("DPX: brak sygnatury SDPX / XPDS – to nie jest plik DPX")
    u32 = lambda o: struct.unpack_from(e + "I", raw, o)[0]
    u16 = lambda o: struct.unpack_from(e + "H", raw, o)[0]
    h = {"endian": "big" if e == ">" else "little", "offset": u32(4), "orientation": u16(768), "elements": u16(770),
         "width": u32(772), "height": u32(776), "descriptor": raw[800], "transfer": raw[801], "colorimetric": raw[802],
         "bits": raw[803], "packing": u16(804), "encoding": u16(806), "data_offset": u32(808), "eol_pad": u32(812)}
    if h["data_offset"] in (0, 0xFFFFFFFF):
        h["data_offset"] = h["offset"]
    if h["eol_pad"] == 0xFFFFFFFF:
        h["eol_pad"] = 0
    h["channels"] = CHANNELS.get(h["descriptor"])
    h["transfer_name"] = TRANSFER.get(h["transfer"], f"kod {h['transfer']}")
    h["log"] = h["transfer"] in LOG_TRANSFERS
    return h


def info(path) -> Dict[str, Any]:
    """Header of a DPX file: size, bits, packing, descriptor, transfer (+ `log` True for log code values)."""
    with open(os.fspath(path), "rb") as fh:
        return _header(fh.read(2048))


def _check(h: Dict[str, Any]) -> None:
    if h["encoding"] != 0:
        raise ValueError("DPX: kodowanie RLE nie jest obsługiwane")
    if h["channels"] is None:
        raise ValueError(f"DPX: deskryptor {h['descriptor']} nie jest obsługiwany (RGB / RGBA / ABGR / luma / alfa)")
    if h["bits"] not in (8, 10, 12, 16):
        raise ValueError(f"DPX: głębia {h['bits']} bit nie jest obsługiwana (8 / 10 / 12 / 16)")
    if h["width"] <= 0 or h["height"] <= 0:
        raise ValueError("DPX: zerowy rozmiar obrazu")


def _bitstream(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    """n values of `bits` bits from a contiguous stream in 32-bit words, MSB first within each word."""
    w = words.astype(np.uint64)
    pos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    wi = (pos // 32).astype(np.int64)
    off = (pos % 32).astype(np.uint64)
    pair = (w[wi] << np.uint64(32)) | np.where(wi + 1 < w.size, w[np.minimum(wi + 1, w.size - 1)], 0).astype(np.uint64)
    return ((pair >> (np.uint64(64) - off - np.uint64(bits))) & np.uint64((1 << bits) - 1)).astype(np.uint16)


def read(path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(image (h, w, c) in the file's channel order R G B (A) / A B G R / single, uint8 or uint16 at the file's
    bit depth – NOT rescaled, header). Element 0 only."""
    with open(os.fspath(path), "rb") as fh:
        raw = fh.read()
    h = _header(raw)
    _check(h)
    w, hh, c, bits = h["width"], h["height"], h["channels"], h["bits"]
    e = ">" if h["endian"] == "big" else "<"
    n = w * c                                             # components per line
    data = memoryview(raw)[h["data_offset"]:]
    if bits == 8:
        line = n + (-n) % 4 + h["eol_pad"]
        a = np.frombuffer(data, np.uint8, count=line * hh).reshape(hh, line)[:, :n]
    elif bits == 16:
        line = 2 * n + (-2 * n) % 4 + h["eol_pad"]
        a = np.frombuffer(data, np.uint8, count=line * hh).reshape(hh, line)[:, :2 * n]
        a = np.ascontiguousarray(a).view(e + "u2").astype(np.uint16)
    elif bits == 10 and h["packing"] in (1, 2):
        per = (n + 2) // 3                                # three 10-bit components per 32-bit word
        line = per * 4 + h["eol_pad"]
        wd = np.frombuffer(data, np.uint8, count=line * hh).reshape(hh, line)[:, :per * 4]
        words = np.ascontiguousarray(wd).view(e + "u4").astype(np.uint32)
        shifts = (22, 12, 2) if h["packing"] == 1 else (20, 10, 0)
        comp = np.stack([(words >> s) & 0x3FF for s in shifts], axis=-1).reshape(hh, per * 3)
        a = comp[:, :n].astype(np.uint16)
    elif bits == 12 and h["packing"] in (1, 2):
        line = 2 * n + (-2 * n) % 4 + h["eol_pad"]
        wd = np.frombuffer(data, np.uint8, count=line * hh).reshape(hh, line)[:, :2 * n]
        v = np.ascontiguousarray(wd).view(e + "u2").astype(np.uint16)
        a = (v >> 4) if h["packing"] == 1 else (v & 0x0FFF)
    elif bits in (10, 12) and h["packing"] == 0:
        per_line_words = (n * bits + 31) // 32
        line = per_line_words * 4 + h["eol_pad"]
        rows = []
        for y in range(hh):
            wd = np.frombuffer(data, np.uint8, count=per_line_words * 4, offset=y * line)
            rows.append(_bitstream(np.ascontiguousarray(wd).view(e + "u4"), bits, n))
        a = np.stack(rows)
    else:
        raise ValueError(f"DPX: pakowanie {h['packing']} dla {bits} bit nie jest obsługiwane")
    return a.reshape(hh, w, c), h


def _encode_lines(a: np.ndarray, h: Dict[str, Any]) -> np.ndarray:
    """(rows, bytes) uint8: the image data of every line without its end-of-line padding, in the file's layout –
    the inverse of `read` for the same header (code values `a` (h, w, c) in the file's channel order)."""
    hh, w, c = a.shape
    bits, packing = h["bits"], h["packing"]
    e = ">" if h["endian"] == "big" else "<"
    n = w * c
    v = a.reshape(hh, n).astype(np.uint32)

    def pad4(b: np.ndarray) -> np.ndarray:
        extra = (-b.shape[1]) % 4
        return np.concatenate([b, np.zeros((hh, extra), np.uint8)], axis=1) if extra else b
    if bits == 8:
        return pad4(v.astype(np.uint8))
    if bits == 16:
        return pad4(np.ascontiguousarray(v.astype(e + "u2")).view(np.uint8).reshape(hh, 2 * n))
    if bits == 10 and packing in (1, 2):
        per = (n + 2) // 3
        vv = np.concatenate([v, np.zeros((hh, per * 3 - n), np.uint32)], axis=1).reshape(hh, per, 3)
        sh = (22, 12, 2) if packing == 1 else (20, 10, 0)
        words = (vv[..., 0] << sh[0]) | (vv[..., 1] << sh[1]) | (vv[..., 2] << sh[2])
        return np.ascontiguousarray(words.astype(e + "u4")).view(np.uint8).reshape(hh, per * 4)
    if bits == 12 and packing in (1, 2):
        vv = (v << 4) if packing == 1 else v
        return pad4(np.ascontiguousarray(vv.astype(e + "u2")).view(np.uint8).reshape(hh, 2 * n))
    if bits in (10, 12) and packing == 0:                 # contiguous bit stream, MSB first in 32-bit words
        per_line_words = (n * bits + 31) // 32
        shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
        stream = ((v[..., None] >> shifts) & 1).astype(np.uint8).reshape(hh, n * bits)
        stream = np.concatenate([stream, np.zeros((hh, per_line_words * 32 - n * bits), np.uint8)], axis=1)
        be = np.packbits(stream, axis=1)                  # big-endian words
        if e == "<":
            be = np.ascontiguousarray(be.reshape(hh, per_line_words, 4)[..., ::-1]).reshape(hh, per_line_words * 4)
        return be
    raise ValueError(f"DPX: pakowanie {packing} dla {bits} bit nie jest obsługiwane")


def write_like(path, codes: np.ndarray, src_path) -> None:
    """Write code values `codes` (h, w, c, the file's channel order, as `read` gives them) into a copy of the DPX
    `src_path`: the same header (every field, user data, timecode), packing, endianness and end-of-line padding – only
    the image data changes, so a frame written back with the codes `read` gave is byte-identical to the source."""
    with open(os.fspath(src_path), "rb") as fh:
        raw = bytearray(fh.read())
    h = _header(bytes(raw[:2048]))
    _check(h)
    if codes.shape != (h["height"], h["width"], h["channels"]):
        raise ValueError(f"DPX: rozmiar {codes.shape} ≠ plik źródłowy {(h['height'], h['width'], h['channels'])}")
    enc = _encode_lines(np.asarray(codes), h)
    line = enc.shape[1] + h["eol_pad"]
    o, hh = h["data_offset"], h["height"]
    block = np.frombuffer(bytes(raw[o:o + line * hh]), np.uint8).reshape(hh, line).copy()
    block[:, :enc.shape[1]] = enc
    raw[o:o + line * hh] = block.tobytes()
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(os.fspath(path), "wb") as fh:
        fh.write(bytes(raw))


def _to16(a: np.ndarray, bits: int) -> np.ndarray:
    """Code values of `bits` bits -> the full uint16 range (bit replication, as a PNG16 of the same picture)."""
    a = a.astype(np.uint32)
    if bits == 16:
        return a.astype(np.uint16)
    return ((a << (16 - bits)) | (a >> (2 * bits - 16) if 2 * bits >= 16 else 0)).astype(np.uint16)


def imread(path, flags: int = 1):
    """cv2.imread for DPX (None when the file cannot be read, like OpenCV). flags: cv2.IMREAD_* values."""
    try:
        a, h = read(path)
    except (OSError, ValueError):
        return None
    if h["descriptor"] == 52:                             # ABGR -> RGBA
        a = a[..., ::-1]
    unchanged = flags == -1
    anydepth = unchanged or bool(flags & 2)
    if h["bits"] == 8:
        out = a.astype(np.uint8)
    else:
        out = _to16(a, h["bits"])
        if not anydepth:
            out = (out >> 8).astype(np.uint8)
    c = out.shape[2]
    if c >= 3:                                            # RGB(A) -> BGR(A)
        out = np.concatenate([out[..., 2::-1], out[..., 3:]], axis=2)
    if unchanged or (flags & 4 and not flags & 1):        # UNCHANGED / ANYCOLOR: the file's channels
        return out[..., 0] if c == 1 else np.ascontiguousarray(out if unchanged else out[..., :3])
    if not flags & 1:                                     # GRAYSCALE (+ ANYDEPTH)
        if c == 1:
            return out[..., 0]
        b, g, r = (out[..., i].astype(np.float32) for i in range(3))
        return np.clip(np.rint(0.114 * b + 0.587 * g + 0.299 * r), 0, np.iinfo(out.dtype).max).astype(out.dtype)
    if c == 1:
        return np.ascontiguousarray(np.repeat(out, 3, axis=2))
    return np.ascontiguousarray(out[..., :3])
