# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""DPX plates (engine/dpx.py hooked into cv2.imread by engine/cv2_unicode.py; user 2026-11-10): every packing and
depth round-trips through a minimal writer here, and files made by ffmpeg read the same as ffmpeg decodes them."""
import shutil
import struct
import subprocess

import cv2
import numpy as np
import pytest

import engine  # noqa: F401  (installs the cv2.imread hook)
from engine import dpx


def write_dpx(path, rgb, bits, packing=1, endian=">", descriptor=50, transfer=1, eol_pad=0):
    """A minimal DPX writer for the tests: rgb (h, w, c) integer code values of `bits` bits."""
    h, w, c = rgb.shape
    n = w * c
    flat = rgb.reshape(h, n).astype(np.uint32)
    rows = []
    for y in range(h):
        v = flat[y]
        if bits == 8:
            b = v.astype(np.uint8).tobytes()
            b += b"\0" * ((-len(b)) % 4)
        elif bits == 16 or (bits == 12 and packing in (1, 2)):
            if bits == 12:
                v = (v << 4) if packing == 1 else v
            b = v.astype(endian + "u2").tobytes()
            b += b"\0" * ((-len(b)) % 4)
        elif bits == 10 and packing in (1, 2):
            pad = (-n) % 3
            vv = np.concatenate([v, np.zeros(pad, np.uint32)]).reshape(-1, 3)
            sh = (22, 12, 2) if packing == 1 else (20, 10, 0)
            words = (vv[:, 0] << sh[0]) | (vv[:, 1] << sh[1]) | (vv[:, 2] << sh[2])
            b = words.astype(endian + "u4").tobytes()
        else:                                             # packed bit stream, MSB first in 32-bit words
            bitsstr = "".join(format(int(x), f"0{bits}b") for x in v)
            bitsstr += "0" * ((-len(bitsstr)) % 32)
            words = np.array([int(bitsstr[i:i + 32], 2) for i in range(0, len(bitsstr), 32)], np.uint64)
            b = words.astype(endian + "u4").tobytes()
        rows.append(b + b"\0" * eol_pad)
    data = b"".join(rows)
    hdr = bytearray(2048)
    hdr[0:4] = b"SDPX" if endian == ">" else b"XPDS"
    struct.pack_into(endian + "I", hdr, 4, 2048)
    struct.pack_into(endian + "H", hdr, 768, 0)
    struct.pack_into(endian + "H", hdr, 770, 1)
    struct.pack_into(endian + "I", hdr, 772, w)
    struct.pack_into(endian + "I", hdr, 776, h)
    hdr[800], hdr[801], hdr[802], hdr[803] = descriptor, transfer, 1, bits
    struct.pack_into(endian + "H", hdr, 804, packing)
    struct.pack_into(endian + "H", hdr, 806, 0)
    struct.pack_into(endian + "I", hdr, 808, 2048)
    struct.pack_into(endian + "I", hdr, 812, eol_pad)
    with open(path, "wb") as fh:
        fh.write(bytes(hdr) + data)


def _img(h, w, c, bits, seed=0):
    return np.random.default_rng(seed).integers(0, 1 << bits, size=(h, w, c), dtype=np.uint32)


CASES = [(8, 0), (10, 1), (10, 2), (10, 0), (12, 1), (12, 2), (12, 0), (16, 0)]


@pytest.mark.parametrize("bits,packing", CASES)
@pytest.mark.parametrize("endian", [">", "<"])
def test_round_trip_every_packing(tmp_path, bits, packing, endian):
    rgb = _img(7, 13, 3, bits)                           # odd sizes: line padding and partial words
    p = tmp_path / "a.dpx"
    write_dpx(p, rgb, bits, packing, endian)
    a, h = dpx.read(p)
    assert np.array_equal(a.astype(np.uint32), rgb), (bits, packing, endian)
    assert h["width"] == 13 and h["height"] == 7 and h["bits"] == bits and h["log"] is True
    full = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)       # through the cv2 hook: BGR, full 16-bit range
    want16 = rgb[..., ::-1].astype(np.float64) / ((1 << bits) - 1) * 65535
    if bits == 8:
        assert full.dtype == np.uint8 and np.array_equal(full, rgb[..., ::-1].astype(np.uint8))
    else:
        assert full.dtype == np.uint16 and np.abs(full - want16).max() <= 64
    col = cv2.imread(str(p))                               # IMREAD_COLOR: 8-bit BGR
    assert col.dtype == np.uint8 and col.shape == (7, 13, 3)
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert gray.shape == (7, 13) and gray.dtype == np.uint8


def test_rgba_abgr_luma_and_eol_padding(tmp_path):
    rgba = _img(5, 6, 4, 10, 1)
    write_dpx(tmp_path / "rgba.dpx", rgba, 10, 1, descriptor=51, eol_pad=8)
    u = cv2.imread(str(tmp_path / "rgba.dpx"), cv2.IMREAD_UNCHANGED)
    assert u.shape == (5, 6, 4) and np.array_equal(u[..., 3] >> 6, rgba[..., 3].astype(np.uint16))
    assert np.array_equal(u[..., 0] >> 6, rgba[..., 2].astype(np.uint16))       # BGR order
    abgr = rgba[..., ::-1]
    write_dpx(tmp_path / "abgr.dpx", abgr, 10, 1, descriptor=52)
    assert np.array_equal(cv2.imread(str(tmp_path / "abgr.dpx"), cv2.IMREAD_UNCHANGED), u)
    luma = _img(4, 9, 1, 12, 2)
    write_dpx(tmp_path / "y.dpx", luma, 12, 1, descriptor=6)
    y = cv2.imread(str(tmp_path / "y.dpx"), cv2.IMREAD_UNCHANGED)
    assert y.shape == (4, 9) and np.array_equal(y >> 4, luma[..., 0].astype(np.uint16))
    assert cv2.imread(str(tmp_path / "y.dpx")).shape == (4, 9, 3)


def test_unsupported_and_broken_files(tmp_path):
    p = tmp_path / "x.dpx"
    p.write_bytes(b"nope" * 600)
    assert cv2.imread(str(p)) is None
    with pytest.raises(ValueError, match="SDPX"):
        dpx.read(p)
    write_dpx(p, _img(2, 2, 3, 10), 10, 1)
    raw = bytearray(p.read_bytes())
    struct.pack_into(">H", raw, 806, 1)                   # RLE
    p.write_bytes(bytes(raw))
    assert cv2.imread(str(p)) is None
    with pytest.raises(ValueError, match="RLE"):
        dpx.read(p)


def test_unicode_path_and_transfer_info(tmp_path):
    d = tmp_path / "październik"
    d.mkdir()
    p = d / "plate_0001.dpx"
    write_dpx(p, _img(3, 4, 3, 10), 10, 1, transfer=2)
    assert cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape == (3, 4, 3)
    i = dpx.info(p)
    assert i["transfer_name"] == "liniowa" and i["log"] is False


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize("pix", ["gbrp10le", "rgb48le", "rgb24", "gbrp12le"])
def test_files_made_by_ffmpeg_read_like_ffmpeg_decodes_them(tmp_path, pix):
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), (np.random.default_rng(3).random((36, 50, 3)) * 65535).astype(np.uint16))
    d = tmp_path / "a.dpx"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-pix_fmt", pix, str(d)], check=True)
    ref = tmp_path / "ref.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d), "-pix_fmt", "rgb48be", str(ref)], check=True)
    want = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED).astype(np.int64)
    got = cv2.imread(str(d), cv2.IMREAD_UNCHANGED).astype(np.int64)
    if pix == "rgb24":                                    # an 8-bit DPX reads as uint8 (like an 8-bit PNG); swscale's
        want = np.rint(want / 257.0).astype(np.int64)     # 8 -> 16 bit is not exactly x257: compare in 8 bit
    assert got.shape == want.shape
    tol = 0 if pix in ("rgb48le", "rgb24") else 2 ** (16 - int(pix[4:6])) if pix.startswith("gbrp") else 0
    assert np.abs(got - want).max() <= max(tol, 1) * 2, (pix, np.abs(got - want).max())


def test_ingest_reads_dpx_metadata(tmp_path):
    from engine.ingest import MediaInspector
    p = tmp_path / "plate_0001.dpx"
    write_dpx(p, _img(6, 8, 3, 10), 10, 1, transfer=1)
    m = MediaInspector.read_metadata(str(p))
    assert (m["width"], m["height"], m["bit_depth"], m["source_type"]) == (8, 6, 10, "img_seq")
    assert m["dpx_log"] is True and "log" in m["note"]
    write_dpx(p, _img(6, 8, 3, 16), 16, 0, transfer=2)
    assert MediaInspector.read_metadata(str(p))["dpx_log"] is False

