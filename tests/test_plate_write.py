# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup's full frame in the plate's own format (engine/plate_write.py, user decision 2026-11-12): EXR keeps its
header (half, compression, attributes, extra layers), DPX its header and packing, TIFF / PNG their depth and alpha;
outside the patch every value is the source file's own (bit for bit); JPG becomes a lossless PNG; MODELS.md lists the
models behind the pixels."""
import json

import cv2
import Imath
import numpy as np
import OpenEXR
import pytest

import engine  # noqa: F401  (cv2.imread hook for DPX)
from engine import cleanup as cu
from engine import dpx
from engine import plate_write as PW
from tests.test_dpx import write_dpx

W, H = 64, 48


def _mask():
    m = np.zeros((H, W), np.float32)
    m[10:30, 20:44] = 1.0
    m[9, 20:44] = 0.5                       # a soft edge
    return m


def _exr(path, rng):
    hdr = OpenEXR.Header(W, H)
    half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    hdr["channels"] = {"R": half, "G": half, "B": half, "A": half, "depth.Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    hdr["compression"] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    hdr["owner"] = b"plate-owner"
    planes = {c: (rng.random((H, W)) * 4).astype(np.float16).tobytes() for c in "RGBA"}
    planes["depth.Z"] = rng.random((H, W)).astype(np.float32).tobytes()
    o = OpenEXR.OutputFile(str(path), hdr)
    o.writePixels(planes)
    o.close()


def _read_all(path):
    f = OpenEXR.InputFile(str(path))
    hdr = f.header()
    out = {}
    for n, ch in hdr["channels"].items():
        dt = np.float16 if str(ch.type) == "HALF" else np.float32
        out[n] = np.frombuffer(f.channel(n, ch.type), dt).reshape(H, W)
    f.close()
    return out, hdr


def test_exr_keeps_header_layers_and_source_values(tmp_path):
    src = tmp_path / "plate.0001.exr"
    _exr(src, np.random.default_rng(1))
    fmt = PW.plate_format(src, W, H)
    assert fmt.native and fmt.comp_ext == "exr" and fmt.describe()["pixel_type"] == "half"
    m = _mask()
    comp = np.full((H, W, 3), 0.25, np.float32)
    fmt.write_comp(tmp_path / "c.exr", src, comp, m)
    a, ha = _read_all(src)
    b, hb = _read_all(tmp_path / "c.exr")
    assert set(b) == set(a) and str(hb["compression"]) == str(ha["compression"]) and hb["owner"] == b"plate-owner"
    assert all(str(hb["channels"][n].type) == str(ha["channels"][n].type) for n in a)
    out = m == 0
    for n in a:                                            # outside the patch: the source bits
        assert np.array_equal(a[n].view(np.uint16 if a[n].dtype == np.float16 else np.uint32)[out],
                              b[n].view(np.uint16 if b[n].dtype == np.float16 else np.uint32)[out])
    assert np.array_equal(a["A"], b["A"]) and np.array_equal(a["depth.Z"], b["depth.Z"])   # other channels whole
    assert np.allclose(b["R"][m > 0].astype(np.float32), 0.25)
    fmt.write_patch_exr(tmp_path / "p.exr", np.dstack([comp * m[..., None], m]))
    p, hp = _read_all(tmp_path / "p.exr")
    assert sorted(p) == ["A", "B", "G", "R"] and str(hp["channels"]["R"].type) == "HALF"
    assert str(hp["compression"]) == str(ha["compression"]) and hp["owner"] == b"plate-owner"


@pytest.mark.parametrize("bits,packing", [(8, 0), (10, 1), (10, 2), (10, 0), (12, 1), (12, 0), (16, 0)])
@pytest.mark.parametrize("endian", [">", "<"])
def test_dpx_write_like_round_trip_and_patch(tmp_path, bits, packing, endian):
    rng = np.random.default_rng(bits * 10 + packing)
    codes = rng.integers(0, 1 << bits, size=(H, W, 3), dtype=np.uint32)
    src = tmp_path / "plate.0001.dpx"
    write_dpx(src, codes, bits, packing=packing, endian=endian, eol_pad=4 if bits == 10 else 0)
    a, _ = dpx.read(src)
    dpx.write_like(tmp_path / "same.dpx", a, src)             # unchanged codes = the same bytes
    assert (tmp_path / "same.dpx").read_bytes() == src.read_bytes()
    fmt = PW.plate_format(src, W, H)
    assert fmt.native and fmt.comp_ext == "dpx"
    m = _mask()
    comp = np.zeros((H, W, 3), np.float32)
    comp[..., 2] = 1.0                                          # pure red (BGR)
    fmt.write_comp(tmp_path / "c.dpx", src, comp, m)
    b, hb = dpx.read(tmp_path / "c.dpx")
    assert hb["bits"] == bits and hb["packing"] == packing and hb["endian"] == ("big" if endian == ">" else "little")
    out = m == 0
    assert np.array_equal(a[out], b[out])
    top = (1 << bits) - 1
    inside = m == 1
    assert (b[..., 0][inside] == top).all() and (b[..., 1][inside] == 0).all() and (b[..., 2][inside] == 0).all()


def test_dpx_abgr_keeps_alpha(tmp_path):
    codes = np.random.default_rng(3).integers(0, 1024, size=(H, W, 4), dtype=np.uint32)
    src = tmp_path / "p.0001.dpx"
    write_dpx(src, codes, 10, packing=1, descriptor=52)
    fmt = PW.plate_format(src, W, H)
    comp = np.zeros((H, W, 3), np.float32)
    comp[..., 0] = 1.0                                          # blue
    fmt.write_comp(tmp_path / "c.dpx", src, comp, _mask())
    a, _ = dpx.read(src)
    b, _ = dpx.read(tmp_path / "c.dpx")
    assert np.array_equal(a[..., 0], b[..., 0])                 # A untouched
    assert (b[..., 1][_mask() == 1] == 1023).all()              # B (file order A B G R)


@pytest.mark.parametrize("ext,dt", [("tif", np.uint16), ("tif", np.uint8), ("png", np.uint16), ("png", np.uint8)])
def test_tiff_png_keep_depth_and_alpha(tmp_path, ext, dt):
    top = np.iinfo(dt).max
    raw = np.random.default_rng(4).integers(0, top + 1, size=(H, W, 4)).astype(dt)
    src = tmp_path / f"p.0001.{ext}"
    cv2.imwrite(str(src), raw)
    fmt = PW.plate_format(src, W, H)
    assert fmt.native and fmt.comp_ext == ext
    m = _mask()
    fmt.write_comp(tmp_path / f"c.{ext}", src, np.full((H, W, 3), 0.5, np.float32), m)
    b = cv2.imread(str(tmp_path / f"c.{ext}"), cv2.IMREAD_UNCHANGED)
    assert b.dtype == dt and b.shape == raw.shape
    assert np.array_equal(b[m == 0], raw[m == 0]) and np.array_equal(b[..., 3], raw[..., 3])


def test_jpg_plate_becomes_lossless_png(tmp_path):
    src = tmp_path / "p.0001.jpg"
    cv2.imwrite(str(src), np.random.default_rng(5).integers(0, 256, size=(H, W, 3)).astype(np.uint8))
    fmt = PW.plate_format(src, W, H)
    assert fmt.comp_ext == "png" and any("PNG" in n for n in fmt.notes)
    m = _mask()
    fmt.write_comp(tmp_path / "c.png", src, np.zeros((H, W, 3), np.float32), m)
    assert np.array_equal(cv2.imread(str(tmp_path / "c.png"))[m == 0], cv2.imread(str(src))[m == 0])


def test_cleaner_writes_plate_format_run_json_and_models_md(tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    rng = np.random.default_rng(6)
    files = {}
    for f in (1, 2):
        p = src / f"sh.{f:04d}.tif"
        cv2.imwrite(str(p), rng.integers(0, 65536, size=(H, W, 3)).astype(np.uint16))
        files[f] = p
    obj = np.zeros((H, W), np.float32)
    obj[15:25, 25:35] = 1

    def plate(f):
        return cv2.imread(str(files[f]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0

    def lama(imgs, masks):
        return [np.where(mk[..., None] > 0, 0.5, im).astype(np.float32) for im, mk in zip(imgs, masks)]
    c = cu.Cleaner({"grow_px": 2, "soft_px": 2}, W, H, False, plate=plate, remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [], inpaint=lama, plate_file=lambda f: str(files[f]))
    rj = c.run([1, 2], tmp_path / "v001", "png16")
    assert rj["comp_ext"] == "tif" and rj["format"]["native"] and rj["rgb_source"].endswith("sh.0001.tif")
    for f in (1, 2):
        out = cv2.imread(str(tmp_path / "v001" / "comp" / f"c_{f:04d}.tif"), cv2.IMREAD_UNCHANGED)
        raw = cv2.imread(str(files[f]), cv2.IMREAD_UNCHANGED)
        m = c.mask(f)
        assert out.dtype == np.uint16 and np.array_equal(out[m == 0], raw[m == 0])
        assert (tmp_path / "v001" / "patch" / f"p_{f:04d}.png").exists()
    md = (tmp_path / "v001" / "MODELS.md").read_text(encoding="utf-8")
    assert "LaMa" in md and "Apache-2.0" in md
    assert json.loads((tmp_path / "v001" / "run.json").read_text(encoding="utf-8"))["comp_ext"] == "tif"


def test_cleaner_without_plate_file_keeps_old_format(tmp_path):
    obj = np.zeros((H, W), np.float32)
    obj[15:25, 25:35] = 1
    plate = np.full((H, W, 3), 0.3, np.float32)
    c = cu.Cleaner({}, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj], keep_alphas=lambda f: [])
    rj = c.run([1], tmp_path / "v001", "png")
    assert rj["comp_ext"] == "png" and "format" not in rj
    assert (tmp_path / "v001" / "comp" / "c_0001.png").exists() and (tmp_path / "v001" / "MODELS.md").exists()
