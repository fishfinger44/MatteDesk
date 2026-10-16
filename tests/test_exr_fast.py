# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z30 – fast EXR reading everywhere, bit-identical to the reference paths (condition of the task).

One OpenEXR `channels()` call (half kept), optional scanline band, PlateToDisplay through a 65 536-entry LUT,
model crops converted after cropping. Every comparison here is exact (max |Δ| = 0 / identical bytes).
"""

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import Imath  # noqa: E402
import numpy as np  # noqa: E402
import OpenEXR  # noqa: E402
import pytest  # noqa: E402

from engine.crop import CropSegment, extract_crop  # noqa: E402
from engine.exr_io import display_half, display_half_u8_rgb, display_lut, imread_unchanged, read_half_rgb  # noqa: E402
from engine.ingest import ProxyGenerator, plate_to_display_float, tonemap_plate_to_display  # noqa: E402
from engine.pipeline import PlateCropLoader, default_display, identity_display  # noqa: E402

W, H = 1000, 700   # deliberately != 4096 x 2731


def _write(path, bgr, half=True, comp="ZIPS_COMPRESSION", alpha=None):
    hdr = OpenEXR.Header(bgr.shape[1], bgr.shape[0])
    pt = Imath.PixelType(Imath.PixelType.HALF if half else Imath.PixelType.FLOAT)
    ch = {"B": bgr[..., 0], "G": bgr[..., 1], "R": bgr[..., 2]}
    if alpha is not None:
        ch["A"] = alpha
    hdr["channels"] = {c: Imath.Channel(pt) for c in ch}
    hdr["compression"] = Imath.Compression(getattr(Imath.Compression, comp))
    o = OpenEXR.OutputFile(str(path), hdr)
    o.writePixels({c: np.ascontiguousarray(v.astype(np.float16 if half else np.float32)).tobytes() for c, v in ch.items()})
    o.close()


def _plate(seed=0, w=W, h=H):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 4) * 3.0   # compressible, HDR range
    img = base + rng.normal(0, 0.02, base.shape).astype(np.float32) - 0.05                  # some negatives
    img[0, 0, 0], img[0, 1, 1], img[0, 2, 2], img[1, 0, 0] = np.nan, np.inf, -np.inf, 70000.0
    return img


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    assert np.array_equal(a.view(np.uint8), b.view(np.uint8))   # bitwise (NaN payloads included)


@pytest.mark.parametrize("half,comp", [(True, "ZIPS_COMPRESSION"), (True, "ZIP_COMPRESSION"), (True, "PIZ_COMPRESSION"),
                                       (True, "NO_COMPRESSION"), (False, "ZIPS_COMPRESSION")])
def test_reader_equals_cv2(tmp_path, half, comp):
    p = tmp_path / "a.exr"
    _write(p, _plate(), half, comp)
    ref = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    got = read_half_rgb(p)
    assert got.dtype == (np.float16 if half else np.float32)
    _same(got.astype(np.float32), ref)
    _same(read_half_rgb(p, 123, 457), got[123:457])            # scanline band
    _same(read_half_rgb(p, 650, 10_000), got[650:])            # clipped to the image
    _same(imread_unchanged(p), ref)


def test_reader_with_alpha_equals_cv2(tmp_path):
    p = tmp_path / "rgba.exr"
    a = np.random.default_rng(3).random((H, W)).astype(np.float32)
    _write(p, _plate(1), alpha=a)
    ref = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert ref.shape[2] == 4
    _same(imread_unchanged(p), ref)
    assert read_half_rgb(p).shape[2] == 3                       # colour only by default


def test_lut_equals_formula_for_all_half_values():
    allh = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = plate_to_display_float(allh.astype(np.float32))
    _same(display_lut(1.0), ref)
    # same values reached through an image-shaped array (other memory layout / SIMD tails)
    rng = np.random.default_rng(5)
    img = rng.permutation(np.tile(allh, 11))[: 701 * 333 * 3].reshape(701, 333, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        _same(display_half(img), plate_to_display_float(img.astype(np.float32)))
        _same(display_half(img, 2.5), plate_to_display_float(img.astype(np.float32), 2.5))
    assert np.isnan(display_lut()[np.float16(np.nan).view(np.uint16)])


def test_u8_rgb_lut_equals_float_path_for_all_half_values():
    """Proxy fast path: uint8 LUT + channel flip in the gather == (display_half * 255).astype(uint8)[..., ::-1]."""
    allh = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    rng = np.random.default_rng(7)
    img = rng.permutation(np.tile(allh, 11))[: 701 * 333 * 3].reshape(701, 333, 3)
    with np.errstate(invalid="ignore"):
        ref = np.ascontiguousarray((display_half(img) * 255.0).astype(np.uint8)[..., ::-1])
    out = display_half_u8_rgb(img)
    assert out.flags["C_CONTIGUOUS"] and out.dtype == np.uint8
    assert np.array_equal(out, ref)


def _old_load_frame_rgb(p):
    """The pre-Z30 proxy decode: three FLOAT channel() calls + tonemap of the RGB stack."""
    f = OpenEXR.InputFile(str(p))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb = np.stack([np.frombuffer(f.channel(c, pt), np.float32).reshape(h, w) for c in "RGB"], axis=-1)
    f.close()
    with np.errstate(invalid="ignore"):
        return tonemap_plate_to_display(rgb)


@pytest.mark.parametrize("half", [True, False])
def test_proxy_identical(tmp_path, half):
    p = tmp_path / "plate_0001.exr"
    _write(p, _plate(2), half)
    with np.errstate(invalid="ignore"):
        new = ProxyGenerator.load_frame_rgb(p)
    _same(new, _old_load_frame_rgb(p))
    # whole proxy file: same bytes as the old pipeline (resize after the colour conversion, JPEG q92)
    dest = tmp_path / "p.jpg"
    with np.errstate(invalid="ignore"):
        ProxyGenerator.render_proxy_file(p, dest, max_side=640)
    old = _old_load_frame_rgb(p)
    s = min(640 / W, 640 / H)
    rs = cv2.resize(old, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rs, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert dest.read_bytes() == buf.tobytes()


def test_generate_proxies_parallel_in_order(tmp_path):
    for f in range(1, 7):
        _write(tmp_path / f"pl_{f:04d}.exr", _plate(10 + f, 320, 200))
    seen = []
    with np.errstate(invalid="ignore"):
        out = ProxyGenerator.generate_plate_proxies(str(tmp_path / "pl_####.exr"), 1, 7, tmp_path / "px",
                                                    max_side=160, progress_cb=lambda d, t: seen.append((d, t)))
    assert [p.name for p in out] == [f"p_{f:04d}.jpg" for f in range(1, 7)]   # frame 7 missing -> skipped
    assert seen == [(i, 7) for i in range(1, 8)]


def _segs():
    inner = CropSegment("s00", 1, 1, 0, 0, [300, 200, 256, 256], [0, 0, 0, 0], 1.0)
    edge = CropSegment("s01", 1, 1, 0, 0, [-40, 560, 288, 288], [40, 0, 0, 148], 1.0)   # mirror left + bottom
    corner = CropSegment("s02", 1, 1, 0, 0, [880, -30, 160, 160], [0, 30, 40, 0], 1.0)  # mirror top + right
    return [inner, edge, corner]


@pytest.mark.parametrize("display", [default_display, identity_display])
@pytest.mark.parametrize("gain", [False, True])
def test_model_crops_identical(tmp_path, display, gain):
    _write(tmp_path / "b_0001.exr", _plate(4))
    gp = None
    if gain:
        g = (0.8 + 0.4 * np.random.default_rng(9).random((64, 96, 3))).astype(np.float32)   # deflicker gain map
        cv2.imwrite(str(tmp_path / "g_0001.exr"), g)
        gp = str(tmp_path / "g_####.exr")
    ld = PlateCropLoader(str(tmp_path / "b_####.exr"), display, gain_pattern=gp)
    with np.errstate(invalid="ignore"):
        ref = ld.read_reference(1)
        _same(ld.read(1), ref)
        for seg in _segs():
            _same(ld.read_crop(1, seg), extract_crop(ref, seg))
            _same(ld.crop(1, seg), extract_crop(ref, seg))
            _same(ld.crop(1, seg, 0.5), cv2.resize(extract_crop(ref, seg), (max(16, round(seg.side * 0.5)),) * 2,
                                                   interpolation=cv2.INTER_AREA))


def test_unmarked_display_uses_reference_path(tmp_path):
    _write(tmp_path / "b_0001.exr", _plate(6))
    nonlocal_display = lambda x: x / max(1e-6, float(np.nanmax(np.where(np.isfinite(x), x, 0))))   # not per pixel
    ld = PlateCropLoader(str(tmp_path / "b_####.exr"), nonlocal_display)
    seg = _segs()[0]
    with np.errstate(invalid="ignore"):
        _same(ld.read_crop(1, seg), extract_crop(ld.read_reference(1), seg))
