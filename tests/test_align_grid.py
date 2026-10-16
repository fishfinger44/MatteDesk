# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""PLAN Z18 step 1: the local alignment field stored as a grid (engine.align_grid) instead of full STMaps.

Nodes at explicit positions (cell centres), one upsampling for everything, small files, Nuke group from the grid."""

import numpy as np
import cv2
import pytest

from engine import align_grid
from engine.align import (AlignmentFrameResult, AlignmentReport, LocalPatchAligner, NukeAlignGroupGenerator,
                          PlateWarper)
from engine.nuke_nk import validate_nk


def _field(w, h):
    """Smooth +-3 px field (SPEC 1b) with a gradient everywhere, so a half-cell shift of the nodes shows up."""
    Y, X = np.indices((h, w), dtype=np.float64)
    ux = 3.0 * np.sin(2 * np.pi * X / w) * np.cos(np.pi * Y / h) + 0.4 * X / w
    uy = -2.5 * np.cos(np.pi * X / w) * np.sin(2 * np.pi * Y / h) - 0.3 * Y / h
    return ux, uy


def _grid_from(w, h, step, fn):
    nodes, (gh, gw) = align_grid.node_points(w, h, step)
    vx, vy = fn(nodes[:, 0], nodes[:, 1])
    return align_grid.AlignGrid(vx.reshape(gh, gw).astype(np.float32), vy.reshape(gh, gw).astype(np.float32), w, h)


def test_nodes_are_cell_centres_and_upsample_hits_the_field():
    # plate size not a multiple of the step (the old code stretched 171 x 16 = 2736 rows onto 2731)
    w, h, step = 1000, 613, 16
    gw, gh = align_grid.grid_shape(w, h, step)
    assert (gw, gh) == (63, 39)
    xs, ys = align_grid.node_axes(w, h, step)
    assert len(xs) == gw + 2 and len(ys) == gh + 2          # one guard node beyond each edge
    assert xs[1] == pytest.approx(0.5 * w / gw - 0.5) and xs[-2] == pytest.approx(w - 0.5 * w / gw - 0.5)
    assert xs[0] < 0 and xs[-1] > w - 1 and ys[-2] < h - 1 and np.allclose(np.diff(xs), w / gw)
    ux, uy = _field(w, h)
    Y, X = np.indices((h, w), dtype=np.float64)

    def fn(x, y):
        return (3.0 * np.sin(2 * np.pi * x / w) * np.cos(np.pi * y / h) + 0.4 * x / w,
                -2.5 * np.cos(np.pi * x / w) * np.sin(2 * np.pi * y / h) - 0.3 * y / h)

    dx, dy = align_grid.upsample(_grid_from(w, h, step, fn))
    assert dx.shape == (h, w) and dx.dtype == np.float32
    err = np.hypot(dx - ux, dy - uy)
    # a half-cell (8 px) shift of the nodes would give ~0.15 px here (slope up to 0.02 px/px); the bicubic
    # (a = -0.75, step 16 px, field period 500 px) stays well under the 0.3 px gate, edges included
    assert err.max() < 0.05 and np.percentile(err, 99) < 0.03, (err.max(), np.percentile(err, 99))
    assert err[:8].max() < 0.05 and err[:, -8:].max() < 0.05        # plate edges: guard nodes, no extrapolation
    # a grid shifted by half a cell (the old evaluation at 0, 16, 32 px) is clearly worse
    nodes_old, (gh2, gw2) = align_grid.node_points(w, h, step)
    vx, vy = fn(nodes_old[:, 0] - 0.5 * w / gw, nodes_old[:, 1] - 0.5 * h / gh)
    dx_o, dy_o = align_grid.upsample(align_grid.AlignGrid(vx.reshape(gh2, gw2).astype(np.float32),
                                                          vy.reshape(gh2, gw2).astype(np.float32), w, h))
    assert np.percentile(np.hypot(dx_o - ux, dy_o - uy), 99) > 0.1


def test_upsample_to_proxy_size_matches_full_field():
    w, h = 800, 450
    g = _grid_from(w, h, 16, lambda x, y: _field_at(x, y, w, h))
    full_x, full_y = align_grid.upsample(g)
    px, py = align_grid.upsample(g, w // 4, h // 4, scale=0.25)
    ref_x = cv2.resize(full_x, (w // 4, h // 4), interpolation=cv2.INTER_AREA) * 0.25
    assert np.abs(px - ref_x).max() < 0.02
    assert px.shape == (h // 4, w // 4)


def _field_at(x, y, w, h):
    return (3.0 * np.sin(2 * np.pi * x / w) * np.cos(np.pi * y / h), -2.5 * np.cos(np.pi * x / w) * np.sin(2 * np.pi * y / h))


def test_grid_exr_roundtrip_header_and_size(tmp_path):
    w, h = 4096, 2731
    rng = np.random.default_rng(0)
    gw, gh = align_grid.grid_shape(w, h)
    g = align_grid.AlignGrid(rng.normal(0, 2, (gh + 2, gw + 2)).astype(np.float32), rng.normal(0, 2, (gh + 2, gw + 2)).astype(np.float32), w, h)
    p = align_grid.write_grid(tmp_path / "grid_0001.exr", g)
    r = align_grid.read_grid(p)
    assert (r.plate_w, r.plate_h) == (w, h) and r.shape == (gw, gh) == (256, 171) and r.dx.shape == (173, 258)
    assert np.array_equal(r.dx, g.dx) and np.array_equal(r.dy, g.dy)
    assert r.step == pytest.approx((16.0, 2731 / 171))
    assert p.stat().st_size < 1_000_000           # was ~57 MB per frame as a full STMap
    with pytest.raises(ValueError):
        cv2.imwrite(str(tmp_path / "plain.exr"), np.zeros((4, 4, 3), np.float32))
        align_grid.read_grid(tmp_path / "plain.exr")


def test_fit_rbf_grid_matches_dense_field_and_warp(tmp_path):
    w, h = 640, 400
    ux, uy = _field(w, h)
    al = LocalPatchAligner(w, h, grid_spacing=32)
    pts = al.generate_grid_points(np.ones((h, w), np.float32), scale=1.0)
    xi, yi = pts[:, 0].round().astype(int), pts[:, 1].round().astype(int)
    disps = np.column_stack([ux[yi, xi], uy[yi, xi]]).astype(np.float32)
    grid, ho_med, _ = al.fit_rbf_grid(pts, disps, step=8)
    assert ho_med < 0.1 and grid.shape == align_grid.grid_shape(w, h, 8)
    dx, dy = align_grid.upsample(grid)
    err = np.hypot(dx - ux, dy - uy)[24:-24, 24:-24]
    assert np.percentile(err, 90) < 0.3 and np.median(err) < 0.15
    # fit_rbf_deformation_field is the same grid upsampled
    dx2, dy2, _, _ = al.fit_rbf_deformation_field(pts, disps, eval_downscale=8)
    assert np.array_equal(dx, dx2) and np.array_equal(dy, dy2)
    # the warp through the grid undoes the deformation (aux = beauty sampled at x - u)
    rng = np.random.default_rng(1)
    beauty = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 2.0)
    Y, X = np.indices((h, w), dtype=np.float32)
    aux = cv2.remap(beauty, (X - ux).astype(np.float32), (Y - uy).astype(np.float32), cv2.INTER_CUBIC)
    gp = align_grid.write_grid(tmp_path / "grid_0001.exr", grid)
    fr = AlignmentFrameResult(frame=1, method="local_stmap", residual_px=0.0, residual_max_px=0.0, grid_rel_path=gp.name)
    back = PlateWarper.warp_frame(aux, fr, field_path=gp)
    d = np.abs(back - beauty)[24:-24, 24:-24]
    assert np.median(d) < 0.05            # two cubic resamplings of a sigma-2 texture (E1b criterion 2 threshold)
    # a proxy (quarter size) is warped with the same grid, displacements scaled
    small = cv2.resize(aux, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    back_s = PlateWarper.warp_frame(small, fr, field_path=gp)
    ref_s = cv2.resize(beauty, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    assert np.median(np.abs(back_s - ref_s)[6:-6, 6:-6]) < 0.02


def test_stmap_from_grid_equals_generate_stmap(tmp_path):
    w, h = 320, 200
    g = _grid_from(w, h, 16, lambda x, y: _field_at(x, y, w, h))
    align_grid.stmap_from_grid(g, tmp_path / "stmap.exr")
    dx, dy = align_grid.upsample(g)
    LocalPatchAligner(w, h).generate_stmap(dx, dy, tmp_path / "ref.exr")
    a = cv2.imread(str(tmp_path / "stmap.exr"), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(tmp_path / "ref.exr"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(a, b)


def test_nuke_group_from_grid_reads_grid_and_reformats_to_plate(tmp_path):
    fr = [AlignmentFrameResult(frame=1001 + i, method="local_stmap", residual_px=0.1, residual_max_px=0.2,
                               grid_rel_path=f"grid_{1001 + i:04d}.exr") for i in range(3)]
    rep = AlignmentReport("x2", "backlit", "beauty", (1001, 1003), "local_stmap", 0.1, 0.2, True, "", fr)
    text = NukeAlignGroupGenerator.generate_script("x2", rep, 1000, 613, tmp_path / "a.nk", tmp_path / "a.json")
    assert validate_nk(text, "Align_x2") == []
    assert "grid_x2/grid_%04d.exr" in text and "stmap_" not in text
    assert 'format "63 39 0 0 63 39 1 grid_format"' in text and "raw true" in text
    assert 'format "1000 613 0 0 1000 613 1 plate_format"' in text and "resize distort" in text
    assert f"filter {align_grid.NUKE_FILTER}" in text
    assert '"(x - r + 0.5) / width"' in text and '"(y + g + 0.5) / height"' in text
    assert text.index("Read_Grid") < text.index("Reformat_Grid") < text.index("Grid_to_ST") < text.index("STMap1")


def test_old_report_with_stmap_path_loads_without_it():
    d = {"plate_name": "x", "aux_plate_type": "backlit", "reference_plate": "b", "frame_range": [1, 1], "method": "local_stmap",
         "residual_median": 0.1, "residual_max": 0.2, "needs_correction": True, "recommendation": "",
         "frames": [{"frame": 1, "method": "local_stmap", "residual_px": 0.1, "residual_max_px": 0.2, "stmap_rel_path": "stmap_0001.exr"}]}
    rep = AlignmentReport.from_dict(d)
    assert rep.frames[0].grid_rel_path is None
