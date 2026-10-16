# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorized edge measurement / band smoothing / temporal median (2026-10-13) must be bit-identical to the former
per-sample loops (tests/ref_edge_v1.py = engine/edge.py before the change; temporal_median reference inline).
Synthetic shapes here; on the workstation also real crops of SH020 (alpha of a finished version + plate gray) and SH010."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import ref_edge_v1 as ref
from engine import edge
from engine.refine import temporal_median


def _ref_temporal_median(alphas, radius, motion_limit, band=None):
    n = alphas.shape[0]
    out = alphas.copy()
    for i in range(n):
        idx = [min(n - 1, max(0, j)) for j in range(i - radius, i + radius + 1)]
        win = alphas[idx]
        med = np.median(win, axis=0)
        rng = win.max(axis=0) - win.min(axis=0)
        calm = 1.0 - np.clip((rng - motion_limit * 0.6) / (motion_limit * 0.4), 0.0, 1.0)
        sm = alphas[i] + (med - alphas[i]) * calm
        out[i] = sm if band is None else np.where(band[i], sm, alphas[i])
    return out


def _shapes(seed=0, n=6, h=240, w=320):
    """Soft blobs + thin parts + a noisy plate with a real edge slightly off the alpha edge."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        a = np.zeros((h, w), np.float32)
        cv2.ellipse(a, (w // 2 + 10 * k, h // 2), (70 + 5 * k, 50), 15.0 * k, 0, 360, 1.0, -1)
        cv2.rectangle(a, (40, 30 + 3 * k), (60, 200), 1.0, -1)
        cv2.line(a, (200, 20), (300, 220), 1.0, 2)
        a = cv2.GaussianBlur(a, (0, 0), 0.8 + 0.3 * (k % 3))
        g = np.where(cv2.GaussianBlur(a, (0, 0), 1.2 + 0.2 * k) > 0.5 - 0.05 * k, 0.8, 0.25).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.6 + 0.4 * (k % 2)) + rng.normal(0, 0.01, (h, w)).astype(np.float32)
        out.append((a, g))
    return out


def _same_measure(m_new, m_old):
    assert np.array_equal(m_new.points, m_old.points) and m_new.points.dtype == m_old.points.dtype
    assert np.array_equal(m_new.normals, m_old.normals) and m_new.normals.dtype == m_old.normals.dtype
    assert list(m_new.pieces) == list(m_old.pieces)
    assert np.array_equal(m_new.offsets, m_old.offsets, equal_nan=True) and m_new.offsets.dtype == m_old.offsets.dtype
    assert np.array_equal(m_new.widths, m_old.widths) and m_new.widths.dtype == m_old.widths.dtype
    assert m_new.piece_median == m_old.piece_median


def test_contour_samples_and_measure_edge_bit_identical_synthetic():
    n_samples = 0
    for a, g in _shapes():
        p_new, n_new, pc_new = edge._contour_samples(a)
        p_old, n_old, pc_old = ref._contour_samples(a)
        assert np.array_equal(p_new, p_old) and np.array_equal(n_new, n_old) and pc_new == pc_old
        _same_measure(edge.measure_edge(a, g), ref.measure_edge(a, g))
        _same_measure(edge.measure_edge(a, g, half=4, noise=0.02), ref.measure_edge(a, g, half=4, noise=0.02))
        n_samples += len(p_new)
    assert n_samples > 2000


def test_measure_edge_empty_and_tiny():
    z = np.zeros((50, 50), np.float32)
    m = edge.measure_edge(z, z)
    assert len(m.points) == 0 and m.piece_median == {}
    z[20:24, 20:24] = 1.0                     # contour of 12 px: skipped (< 16) in both
    _same_measure(edge.measure_edge(z, z), ref.measure_edge(z, z))


def test_adaptive_band_width_bit_identical():
    a, g = _shapes(n=1)[0]
    rng = np.random.default_rng(3)
    flow = rng.normal(0, 2, (a.shape[0], a.shape[1], 2)).astype(np.float32)
    core = a > 0.5
    for fl in (None, flow):
        wn = edge.adaptive_band_width(core, g, fl, 180.0, 1.0, contour_sigma=0)
        wo = ref.adaptive_band_width(core, g, fl, 180.0, 1.0)
        assert np.array_equal(wn, wo) and wn.dtype == wo.dtype


def test_smooth_widths_matches_per_index_median():
    rng = np.random.default_rng(1)
    for n in (1, 3, 6, 7, 8, 20, 333):
        w = rng.uniform(3, 24, n).astype(np.float32)
        r = np.array([np.median(w[max(0, i - 3):i + 4]) for i in range(n)], np.float32)
        assert np.array_equal(edge._smooth_widths(w), r)


def test_valid_mask_drops_samples_outside_the_plate_only():
    a, g = _shapes(n=1)[0]
    valid = np.zeros(a.shape, bool)
    valid[:, :200] = True                      # right part = "mirror padding"
    m_all = edge.measure_edge(a, g)
    m_in = edge.measure_edge(a, g, valid=valid)
    assert 0 < len(m_in.points) < len(m_all.points)
    assert (np.round(m_in.points[:, 0]) < 200).all()
    # the kept samples are exactly the subset of the unfiltered ones, with the same offsets
    keep = np.round(m_all.points[:, 0]) < 200
    assert np.array_equal(m_in.points, m_all.points[keep])
    assert np.array_equal(m_in.offsets, m_all.offsets[keep], equal_nan=True)


def test_temporal_median_bit_identical():
    rng = np.random.default_rng(5)
    for n, radius in ((9, 1), (12, 2), (11, 4), (3, 2), (1, 2)):
        a = np.clip(rng.random((n, 16, 20)).astype(np.float32) + rng.normal(0, 0.1, (n, 16, 20)).astype(np.float32), 0, 1)
        a[:, 4:8, 4:8] = 1.0
        band = rng.random((n, 16, 20)) > 0.3
        for b in (None, band):
            new = temporal_median(a, radius, 0.35, b)
            old = _ref_temporal_median(a, radius, 0.35, b)
            assert np.array_equal(new, old) and new.dtype == old.dtype


def test_median_odd_networks_exact():
    rng = np.random.default_rng(7)
    for n in (3, 5, 7):
        w = rng.random((n, 64, 64)).astype(np.float32)
        w[:, :8] = np.round(w[:, :8] * 3) / 3            # ties
        assert np.array_equal(edge_median(w), np.median(w, axis=0))


def edge_median(w):
    from engine.refine import _median_odd
    return _median_odd(w)


# ----------------------------------------------------------------------------- real crops (workstation)
# local workstation shots (skipped elsewhere); paths from the environment so no shot paths live in the code
R1F2 = Path(os.environ.get("MATTE_DESK_REAL_8BIT", "__no_8bit_shot__"))
SH010 = Path(os.environ.get("MATTE_DESK_REAL_EXR", "__no_exr_shot__"))


def _real_cases():
    """(alpha crop, gray crop) of every 7th frame of a finished version – SH020 (8-bit) and SH010 (EXR) when present."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import json
    from engine.crop import CropSegment
    from engine.pipeline import PlateCropLoader, default_display
    from engine.refine import read_alpha
    cases = []
    for shot, obj, ver in ((R1F2, "osoba", "v004"), (SH010, "suknia", "v004"), (SH010, "figure", "v003")):
        rd = shot / "matte_desk" / "objects" / obj / "runs" / ver
        if not rd.exists():
            continue
        rj = json.loads((rd / "run.json").read_text(encoding="utf-8"))
        mf = next(shot.glob("shot_manifest_*.json"))
        man = json.loads(mf.read_text(encoding="utf-8"))
        disp = default_display
        disp.elementwise = True
        disp.lut_white = 1.0
        loader = PlateCropLoader(man["plates"]["beauty"], disp)
        for sd in rj["segments_def"]:
            seg = CropSegment.from_dict(sd)
            for f in range(int(seg.first), int(seg.last) + 1, 7):
                p = rd / "alpha" / seg.seg_id / f"a_{f:04d}.exr"
                if p.exists():
                    cases.append((f"{shot.name}/{obj}/{seg.seg_id}/{f}", read_alpha(p), loader.crop(f, seg).mean(axis=2)))
    return cases


@pytest.mark.real
def test_measure_edge_bit_identical_on_real_crops():
    cases = _real_cases()
    if not cases:
        pytest.skip("no finished versions of SH020 / SH010 (D:/MD_REAL_PROJECT) on this machine")
    n = 0
    for name, a, g in cases:
        m_new, m_old = edge.measure_edge(a, g), ref.measure_edge(a, g)
        _same_measure(m_new, m_old)
        n += len(m_new.points)
    assert n > 50000, n
