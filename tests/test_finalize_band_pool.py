# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3 `Finalizer.finalize` speed-ups (plan docs/PLAN_EDGE_SPEED.md section 11, 2026-10-29) are bit-identical: the
band geometry of the frames computed ahead in a thread pool, the feathers composed on the band's bbox
(bandsys.compose_area_crop) and hygiene + soft zones + writes of the finished frames in the pool give the alphas (returned and written EXR), alpha_base and core PNGs of the former
sequential code, kept in-line here as the reference (band -> compose_area on the whole crop -> temporal median of the
stacks -> hygiene -> soft override -> EXR)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import refine as R
from engine.bandsys import compose_area
from engine.crop import CropSegment, extract_crop
from engine import edge as E
from engine.edge import hygiene


def _scene(n=12, side=96, seed=0):
    rng = np.random.default_rng(seed)
    frames = list(range(1, n + 1))
    cores, excl, holes = {}, {}, {}
    for i, f in enumerate(frames):
        c = np.zeros((side, side), np.uint8)
        cv2.ellipse(c, (40 + i, 48 + (i % 3)), (22 + i % 4, 16), 10.0 * i, 0, 360, 1, -1)
        cv2.circle(c, (40 + i, 48), 4, 0, -1)                      # a hole hygiene must keep
        cores[f] = c.astype(bool)
        e = np.zeros((side, side), bool)
        e[10:18, 60 + i % 5:70] = True
        excl[f] = e
        holes[f] = ~c.astype(bool) & (cv2.dilate(c, np.ones((9, 9), np.uint8)) > 0) & (rng.random((side, side)) > 0.8)
    soft = {f: (rng.random((side, side)).astype(np.float32), np.pad(np.ones((20, 20), bool), ((50, 26), (50, 26))))
            for f in frames[::3]}
    return frames, cores, excl, holes, soft


def _band_fn(f, seg, c, bw):
    # a frame-dependent band (as the band points make it): wider on even frames, a notch on every third
    b = R.edge_band(c, bw + (3.0 if f % 2 == 0 else 1.0))
    if f % 3 == 0:
        b[:, :20] = False
    return b


def _reference(obj, seg, frames, cores, keys, excl, holes, soft, band_fn, run_dir):
    radius, mlimit = R.SMOOTHING[obj.get("temporal_smoothing", "normal")]
    band_px = R.BAND_PX["core"] * float(obj.get("band_scale", 1.0))
    fe, fo = E.band_feathers(obj)
    out, bands, res = {}, [], []
    for f in frames:
        c = extract_crop(cores[f].astype(np.float32), seg) > 0.5
        narrow = f in keys and not R.key_band_same_on(obj)
        bw = min(band_px, 3.0) if narrow else band_px
        b = band_fn(f, seg, c, bw) if (band_fn is not None and not narrow) else R.edge_band(c, bw)
        b &= ~(extract_crop(excl[f].astype(np.float32), seg) > 0.5)
        s_ = R.antialiased_core(c)
        a = compose_area(c, s_, b, fe, fo) if (fe > 0 or fo > 0) else np.where(b, s_, c.astype(np.float32))
        bands.append(b)
        res.append(a)
    med = R.temporal_median(np.stack(res), radius, mlimit, np.stack(bands))
    for i, f in enumerate(frames):
        c = extract_crop(cores[f].astype(np.float32), seg) > 0.5
        keep = extract_crop(holes[f].astype(np.float32), seg) > 0.5
        a, _ = hygiene(med[i], c, bands[i], keep)
        if f in soft:
            R.write_alpha_exr(run_dir / "alpha_base" / f"a_{f:04d}.exr", a)
            a = np.clip(a, 0, 1).astype(np.float16).astype(np.float32)
            sa, sarea = soft[f]
            a = R.apply_soft_override(a, extract_crop(sa, seg), extract_crop(sarea.astype(np.float32), seg) > 0.5,
                                      R.ZONE_FEATHER_PX, None)
        out[f] = a
    return out


@pytest.mark.parametrize("ahead", [1, 3, 64])
@pytest.mark.parametrize("feathers", [None, (0.0, 0.0), (6.0, 14.0)])
def test_finalize_matches_sequential_reference(tmp_path: Path, monkeypatch, ahead, feathers):
    monkeypatch.setattr(R, "BAND_AHEAD", ahead, raising=False)
    side = 96
    frames, cores, excl, holes, soft = _scene(side=side)
    seg = CropSegment("s00", frames[0], frames[-1], 0, 0, [0, 0, side, side], [0, 0, 0, 0], 1.0)
    obj = {"temporal_smoothing": "normal", "band_scale": 1.0}
    if feathers is not None:      # None = the defaults of every stage (bandsys.BAND_DEFAULTS)
        monkeypatch.setattr(E, "band_feathers", lambda o: feathers)
    keys = {frames[0]: cores[frames[0]]}
    fin = R.Finalizer(client=None, io_dir=tmp_path / "io", plate_w=side, plate_h=side,
                      load_crop_image=lambda f, s, sc: np.zeros((side, side, 3), np.float32))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tm = {}
    full, _ = fin.finalize(obj, [seg], frames, cores, keys, frames[0], run_dir, edge_mode="core",
                           exclude=excl, soft_override=soft, band_fn=_band_fn, keep_holes=holes, timings=tm)
    ref_dir = tmp_path / "ref"
    (ref_dir / "alpha_base").mkdir(parents=True)
    ref = _reference(obj, seg, frames, cores, keys, excl, holes, soft, _band_fn, ref_dir)
    for f in frames:
        assert np.array_equal(full[f], ref[f]), f
        got = R.read_alpha(run_dir / "alpha" / "s00" / f"a_{f:04d}.exr")
        assert np.array_equal(got, np.clip(ref[f], 0, 1).astype(np.float16).astype(np.float32)), f
        c = cv2.imread(str(run_dir / "core" / "s00" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(c > 127, cores[f]), f
        if f in soft:
            assert np.array_equal(R.read_alpha(run_dir / "alpha_base" / "s00" / f"a_{f:04d}.exr"),
                                  R.read_alpha(ref_dir / "alpha_base" / f"a_{f:04d}.exr")), f
    for k in ("model", "band_geometry", "feather_compose", "median", "post_frames", "hygiene_threads",
              "write_exr_threads", "write_core_png_threads"):
        assert k in tm, k
