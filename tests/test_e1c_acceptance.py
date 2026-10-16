# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_e1c_acceptance.py - Acceptance tests for Stage E1c: Plate Cleaning (Deflicker & Denoise).

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 4c, 4c.1-4c.5, Section 12: Stage E1c):
1. Synthetic test: artificial flicker (global +-10% + local corner + color shift) applied to 33 frames
   -> flicker reduced by >= 70%, intended brightness ramp preserved, HDR values preserved.
2. Group in Nuke mathematical equivalence: output of Nuke Group (gain map multiplication)
   equals the `_deflicker` export (max diff < 1e-3), EXR keeps the source encoding.
3. Real SH010_04 panel test (station only): flow-aligned flicker per channel <= v10 (2.3 / 2.0 / 4.8 %)
   or not worse by > 20 %, compared with render/SH010_04_X1_dfl_v10; log2 map without ghosts.
4. Nuke Group syntax and interface validation: 1 input (Input1), 1 output (Output1), use_deflicker switch.
5. Denoise and noise residual test: noise variance reduced >= 40%, denoised + residual == plate (lossless).
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.clean import (
    CleanReport,
    DeflickerEngine,
    DeflickerMetrics,
    DenoiseEngine,
    NukeDeflickerGroupGenerator,
    PlateCleaner,
)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _synthetic_flicker_sequence(n=33, h=144, w=216, seed=42):
    """Textured static set + intended slow light ramp + HDR highlight + moving object, then flicker:
    global +-10 %, local corner, colour (B channel). Returns (flickering, clean_reference, object_masks)."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 1.5) * 0.5 + 0.25
    obj_tex = cv2.GaussianBlur(rng.random((40, 40, 3)).astype(np.float32), (0, 0), 1.0) * 0.6 + 0.2
    ramp = np.linspace(0.8, 1.2, n)  # intended light change – must survive
    yy, xx = np.mgrid[0:h, 0:w]
    hl = ((xx - 40) ** 2 + (yy - 40) ** 2) < 12 ** 2
    clean, flick, masks = [], [], []
    corner = np.zeros((h, w), np.float32)
    corner[: h // 3, 2 * w // 3:] = 1.0
    corner = cv2.GaussianBlur(corner, (0, 0), 5.0)[..., None]
    for i in range(n):
        f = tex * ramp[i]
        f[hl] = 3.0 * ramp[i]  # HDR highlight
        ox = 30 + int(4 * i)  # object moves 4 px / frame
        m = np.zeros((h, w), bool)
        m[80:120, ox:ox + 40] = True
        f[80:120, ox:ox + 40] = obj_tex * ramp[i]
        clean.append(f.astype(np.float32))
        masks.append(m)
        g = f * (1.0 + 0.10 * np.sin(i * 1.5))
        g = g * (1.0 - corner) + g * (1.0 + 0.25 * np.cos(i * 2.0)) * corner
        g[..., 0] *= 1.0 + 0.15 * np.sin(i * 3.0)
        flick.append(g.astype(np.float32))
    return flick, clean, masks


def test_acceptance_e1c_criterion1_synthetic_flicker_removal(tmp_path):
    """SPEC E1c: artificial flicker (global +-10 % + local corner + colour) on 33 frames -> removed >= 70 %,
    intended ramp preserved, no ghosts on the moving object, HDR values preserved."""
    flick, clean, masks = _synthetic_flicker_sequence()
    n = len(flick)

    cleaned, gains, report, _ = PlateCleaner.process_sequence(
        flick, plate_name="synth", target_reduction=0.70,
        enable_stage1=True, enable_stage2=True, enable_stage3=False, enable_stage4=True)

    assert report.flicker_before.mean > 5.0
    assert report.reduction_pct >= 70.0, (report.flicker_before.to_dict(), report.flicker_after.to_dict())

    # intended light change: low-pass trend of the input == low-pass trend of the result (4c.5)
    from scipy.ndimage import gaussian_filter1d
    static = ~np.any(masks, axis=0)
    m_in = np.array([f[static].mean() for f in flick])
    m_out = np.array([c[static].mean() for c in cleaned])
    t_in, t_out = gaussian_filter1d(m_in, 6, mode="nearest"), gaussian_filter1d(m_out, 6, mode="nearest")
    assert np.max(np.abs(t_out / t_in - 1.0)) < 0.03, "intended ramp not preserved"
    assert m_out[-1] / m_out[0] > 1.3  # ramp 0.8 -> 1.2 still there

    # no ghosts of the moving object: where the object was in neighbouring frames (but is not now) the
    # correction must stay a smooth multiplier – a ghost = the object's imprint in the gain (lesson v07)
    for i in range(n):
        lr = np.log((cleaned[i] + 1e-3) / (flick[i] + 1e-3)).mean(axis=2)
        hf = np.abs(lr - cv2.GaussianBlur(lr, (0, 0), 8.0))
        trail = np.zeros_like(masks[i])
        for j in range(max(0, i - 6), min(n, i + 7)):
            trail |= masks[j]
        cur = cv2.dilate(masks[i].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        trail &= ~cur
        if trail.any():
            assert float(hf[trail].mean()) < 0.005, f"ghost of the moving object in frame {i}: {hf[trail].mean():.4f}"

    # HDR preserved (no clipping at 1.0)
    hl_vals = [c.max() for c in cleaned]
    assert min(hl_vals) > 2.0


def test_acceptance_e1c_criterion2_nuke_group_mathematical_equivalence(tmp_path):
    """SPEC 4c.4 / E1c: Group (Read gain map -> Reformat -> multiply) == `_deflicker` export, max diff < 1e-3.

    Exercised at a plate size where the stored gain map is downscaled (w > 1024)."""
    flick, _, _ = _synthetic_flicker_sequence(n=12, h=720, w=1280)
    cleaned, gains, report, _ = PlateCleaner.process_sequence(
        flick, plate_name="beauty", enable_stage1=True, enable_stage2=True, enable_stage3=False, enable_stage4=True)
    h, w = flick[0].shape[:2]

    gain_dir = tmp_path / "gain_maps"
    PlateCleaner.export_gain_maps(gains, first_frame=1, output_dir=gain_dir)
    nk_file = tmp_path / "Deflicker_beauty.nk"
    NukeDeflickerGroupGenerator.generate_script(
        plate_name="beauty", gain_pattern=str(gain_dir / "gain_%04d.exr"), plate_w=w, plate_h=h,
        first_frame=1, last_frame=len(flick), report=report, output_nk_path=nk_file)
    assert "Reformat" in nk_file.read_text() and "operation multiply" in nk_file.read_text()

    for i in range(len(flick)):
        g = cv2.imread(str(gain_dir / f"gain_{i + 1:04d}.exr"), cv2.IMREAD_UNCHANGED)
        assert g.shape[1] == 1024  # compact map, not full resolution
        # Group: Reformat (cubic) to plate size, Merge multiply
        nuke = flick[i] * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        diff = float(np.abs(nuke - cleaned[i]).max())
        assert diff < 1e-3, f"frame {i + 1}: max |Group - _deflicker| = {diff}"
        # the gain is smooth enough that a different reconstruction filter stays within tolerance too
        lin = flick[i] * cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        assert float(np.abs(lin - cleaned[i]).max()) < 1e-2


def test_acceptance_e1c_exr_keeps_source_encoding(tmp_path):
    """SPEC E1c: EXR of the result has the same channels / compression / metadata as the source; HDR kept."""
    import Imath
    import OpenEXR
    from engine.exr_io import read_exr, read_rgb, write_rgb_like

    h, w = 32, 48
    src = tmp_path / "src_0001.exr"
    hdr = OpenEXR.Header(w, h)
    hdr["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)) for c in ("R", "G", "B", "A")}
    hdr["compression"] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    hdr["owner"] = b"MD_REAL_PROJECT test"
    rng = np.random.default_rng(0)
    planes = {c: (rng.random((h, w)) * 4.0).astype(np.float16).tobytes() for c in ("R", "G", "B")}
    planes["A"] = np.full((h, w), 0.5, np.float16).tobytes()
    o = OpenEXR.OutputFile(str(src), hdr)
    o.writePixels(planes)
    o.close()

    bgr = read_rgb(src)
    out = tmp_path / "out" / "src_0001.exr"
    write_rgb_like(out, bgr * 1.1, src, extra_attrs={"matteDesk": b"deflicker test"})
    ch, h2 = read_exr(out)
    _, h1 = read_exr(src)
    assert sorted(h2["channels"]) == sorted(h1["channels"])
    assert all(str(h2["channels"][c].type) == "HALF" for c in h2["channels"])
    assert str(h2["compression"]) == str(h1["compression"])
    assert h2["owner"] == b"MD_REAL_PROJECT test"
    assert np.allclose(ch["A"], 0.5)  # other channels copied
    assert ch["R"].max() > 4.0  # HDR preserved


def test_acceptance_e1c_criterion3_nuke_group_syntax_and_interface(tmp_path):
    """Criterion 3: Nuke Group syntax and interface validation.
    
    Must have exactly 1 input (Input1), 1 output (Output1), use_deflicker switch, and valid Nuke syntax.
    """
    report = CleanReport(
        plate_name="test_plate",
        stages_applied=["stage1_local_mc", "stage4_global_mc"],
        flicker_before=DeflickerMetrics.measure_flow_aligned([]),
        flicker_after=DeflickerMetrics.measure_flow_aligned([]),
        reduction_pct=78.5,
        gain_min=0.85,
        gain_max=1.15,
        frames_count=24,
        plate_w=4096,
        plate_h=2731,
        date="2026-10-11 12:00:00"
    )

    nk_path = tmp_path / "Deflicker_test_plate.nk"
    content = NukeDeflickerGroupGenerator.generate_script(
        plate_name="test_plate",
        gain_pattern="D:/test/gain_%04d.exr",
        plate_w=4096,
        plate_h=2731,
        first_frame=1,
        last_frame=24,
        report=report,
        output_nk_path=nk_path
    )

    assert NukeDeflickerGroupGenerator.validate_nuke_syntax(content) is True

    # Check strict interface: exactly 1 Input, 1 Output
    assert content.count("Input {") == 1
    assert content.count("name Input1") == 1
    assert content.count("Output {") == 1
    assert content.count("name Output1") == 1
    assert "use_deflicker" in content
    assert "Merge2 {" in content
    assert "operation multiply" in content


REAL_SH010_PANEL04 = Path(os.environ.get("MATTE_DESK_SH010_04", "C:\\shots\\SH010\\frame_####.exr"))
REAL_SH010_V10 = Path("C:\\shots\\SH010\\frame_####.exr")
V10_FLICKER_RGB = (2.3, 2.0, 4.8)  # SPEC 4c.1, whole clip, measure_flow.py


def test_acceptance_e1c_criterion4_real_sh010_panel_deflicker():
    """SPEC E1c (real, station only): panel 04 – flow-aligned flicker (measure_flow.py method: flows from the
    v10 render, panel ROI) <= v10 or not worse than v10 by more than 20 % per channel; log2(result/original)
    without ghosts. Compared on the same frame range against render/SH010_04_X1_dfl_v10.

    Frame count: MATTE_DESK_E1C_FRAMES (default 48 – RAM; 193 = whole clip as in the spec numbers)."""
    assert REAL_SH010_PANEL04.exists() and REAL_SH010_V10.exists(), "MD_REAL_PROJECT SH010 panel 04 / v10 render missing"
    n = int(os.environ.get("MATTE_DESK_E1C_FRAMES", "48"))
    files = sorted(p for p in os.listdir(REAL_SH010_PANEL04) if p.lower().endswith(".exr"))[:n]
    v10_files = sorted(p for p in os.listdir(REAL_SH010_V10) if p.lower().endswith(".exr"))[:n]
    assert len(files) == len(v10_files) == n

    from engine.exr_io import read_rgb
    frames = [read_rgb(REAL_SH010_PANEL04 / f) for f in files]
    v10 = [read_rgb(REAL_SH010_V10 / f) for f in v10_files]

    cleaned, _, report, _ = PlateCleaner.process_sequence(
        frames, plate_name="SH010_04", target_reduction=0.70,
        enable_stage1=True, enable_stage2=True, enable_stage3=True, enable_stage4=True)

    W = 1024
    Hh = round(frames[0].shape[0] * W / frames[0].shape[1])
    roi = (slice(int(Hh * .27), int(Hh * .8)), slice(20, W - 20))  # panel area as in measure_flow.py
    m_ours = DeflickerMetrics.measure_flow_aligned(cleaned, eval_w=W, roi_slices=roi, flow_ref=v10)
    m_v10 = DeflickerMetrics.measure_flow_aligned(v10, eval_w=W, roi_slices=roi, flow_ref=v10)
    print("ours", m_ours.to_dict(), "v10 (same frames)", m_v10.to_dict(), "spec v10 whole clip", V10_FLICKER_RGB)
    for ours, ref, ch in ((m_ours.r, m_v10.r, "R"), (m_ours.g, m_v10.g, "G"), (m_ours.b, m_v10.b, "B")):
        assert ours <= ref * 1.2, f"{ch}: {ours:.2f} % vs v10 {ref:.2f} % (> 20 % worse)"

    # ghosts: log2(result / original) must not carry structure of neighbouring frames
    for i in (0, n // 2, n - 1):
        lr = np.log2((cleaned[i] + 0.01) / (frames[i] + 0.01)).mean(axis=2)
        hf = np.abs(lr - cv2.GaussianBlur(lr, (0, 0), 16.0))
        lr_v10 = np.log2((v10[i] + 0.01) / (frames[i] + 0.01)).mean(axis=2)
        hf_v10 = np.abs(lr_v10 - cv2.GaussianBlur(lr_v10, (0, 0), 16.0))
        assert float(hf.mean()) <= float(hf_v10.mean()) * 1.2 + 1e-3


def test_acceptance_e1c_criterion5_denoise_and_noise_residual():
    """Criterion 5: Denoising and noise residual test.
    
    - Noise variance reduced >= 40% on flat areas.
    - Denoised + noise_residual == original (exact lossless reconstruction).
    """
    n = 7
    h, w = 100, 100
    np.random.seed(123)

    # Base clean flat area + additive Gaussian noise
    clean_base = np.ones((h, w, 3), dtype=np.float32) * 0.5
    noisy_frames = []
    for _ in range(n):
        noise = np.random.normal(0, 0.05, (h, w, 3)).astype(np.float32)
        noisy_frames.append(clean_base + noise)

    var_before = DeflickerMetrics.measure_temporal_variance(noisy_frames)

    denoised, residuals = DenoiseEngine.denoise_temporal_mc(
        noisy_frames,
        radius=2,
        spatial_sigma=0.5
    )

    var_after = DeflickerMetrics.measure_temporal_variance(denoised)
    reduction = (1.0 - var_after / var_before) * 100.0

    assert reduction >= 40.0, f"Expected noise reduction >= 40%, got {reduction:.1f}%"

    # Lossless reconstruction: denoised + residual == noisy_frames
    for i in range(n):
        reconstructed = denoised[i] + residuals[i]
        diff = np.abs(reconstructed - noisy_frames[i])
        assert float(diff.max()) < 1e-5, f"Reconstruction failed for frame {i}"
