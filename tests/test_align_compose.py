# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local alignment model measured over the global model (2026-10-15): the grid is the residual, `compose_remap`
puts both together (global model first, then the grid - the order of the Nuke group), outliers are rejected before
the thin-plate fit. Plate sizes here are deliberately not 4096 x 2731. Thresholds = SPEC section 12 (1b):
residual < 0.3 px on >= 90 % of the control points."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import align_grid
from engine.align import (AlignmentFrameResult, AlignmentPipeline, AlignmentReport, LocalPatchAligner,
                          NukeAlignGroupGenerator, PlateWarper, _frame_matrix_cv, compose_remap, reject_outliers,
                          stale_report_reason, stmap_for_frame, warp_global)
from engine.nuke_nk import validate_nk
from engine.tracking import LKTracker
from tests.test_tracking_local import _texture

WORKER = Path(__file__).resolve().parents[1] / "engine" / "align_gpu_worker.py"


def _field(h, w):
    """Smooth local deformation up to +-3 px (SPEC E1b) with a full sine period across the plate - a homography
    cannot absorb it (a half period over a small plate it nearly can)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ux = 3.0 * np.sin(2 * np.pi * xx / w) * np.cos(np.pi * yy / h)
    uy = -2.5 * np.cos(np.pi * xx / w) * np.sin(2 * np.pi * yy / h)
    return ux, uy


def shifted_sequence(T=4, h=750, w=1200, shift=(15.0, -4.0), pan=0.5):
    """beauty = texture window moving by `pan` px/frame; aux(x) = beauty(x - u(x)) with u = global `shift` + the
    smooth local field: the global model must catch the shift, the grid the field.
    Returns (beauty, aux, u_x, u_y) with u the TOTAL displacement (plate px)."""
    sx, sy = shift
    m = 40 + int(abs(sx) + abs(sy)) + int(pan * T) + 4
    big = _texture(h + 2 * m, w + 2 * m, seed=7)
    ux, uy = _field(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    beauty, aux = [], []
    for t in range(T):
        ox = m + pan * t
        beauty.append(cv2.remap(big, xx + ox, yy + m, cv2.INTER_CUBIC))
        aux.append(cv2.remap(big, xx + ox - ux - sx, yy + m - uy - sy, cv2.INTER_CUBIC))
    return beauty, aux, ux + sx, uy + sy


def write_sequence(tmp_path, beauty, aux):
    for i, (b, a) in enumerate(zip(beauty, aux)):
        cv2.imwrite(str(tmp_path / f"b_{i + 1:04d}.exr"), b.astype(np.float32))
        cv2.imwrite(str(tmp_path / f"a_{i + 1:04d}.exr"), a.astype(np.float32))
    h, w = beauty[0].shape[:2]
    return dict(beauty_pattern=str(tmp_path / "b_####.exr"), aux_pattern=str(tmp_path / "a_####.exr"),
                first_frame=1, last_frame=len(beauty), plate_name="aux", aux_plate_type="cleanplate",
                plate_w=w, plate_h=h, workers=1, tracker=LKTracker())


def composed_error(frame, grid, ux, uy, margin=48):
    """Error of the composed map against the true total displacement on the interior (plate px)."""
    h, w = ux.shape
    mx, my = compose_remap(frame, grid, w, h)
    Y, X = np.indices((h, w), dtype=np.float32)
    err = np.hypot(mx - X - ux, my - Y - uy)
    return err[margin:-margin, margin:-margin]


def test_global_shift_is_caught_by_global_model_grid_holds_the_rest(tmp_path):
    T, h, w = 4, 750, 1200
    beauty, aux, ux, uy = shifted_sequence(T, h, w)
    kw = write_sequence(tmp_path, beauty, aux)
    rep = AlignmentPipeline.measure_sequence(grid_dir=tmp_path / "grid", **kw)
    assert [f.method for f in rep.frames] == ["local_stmap"] * T
    for f in rep.frames:
        assert f.global_method in ("translation", "similarity", "affine", "homography"), f.global_method
        assert f.n_points >= 80 and f.n_rejected <= 0.1 * f.n_points, (f.n_points, f.n_rejected)
        # the global model carries the 15 px: content at the plate centre moved by ~(-15, +4) (aux -> beauty);
        # it may absorb part of the +-3 px local field (the grid holds the rest)
        m = _frame_matrix_cv(f)
        c = m @ np.array([w / 2, h / 2, 1.0])
        assert abs(c[0] / c[2] - w / 2 + 15.0) < 3.5 and abs(c[1] / c[2] - h / 2 - 4.0) < 3.5, c
        grid = align_grid.read_grid(tmp_path / "grid" / f.grid_rel_path)
        assert np.abs(grid.dx).max() < 4.5 and np.abs(grid.dy).max() < 4.5       # residual only, not 15 px
        # SPEC 1b gate on the composed alignment, the hold-out gate of the frame, and the warped image
        err = composed_error(f, grid, ux, uy)
        assert np.mean(err < 0.3) >= 0.9, (f.frame, np.median(err), np.percentile(err, 90))
        assert f.status == "ok" and f.holdout_frac_ok >= 0.9
    k = T // 2
    back = PlateWarper.warp_frame(aux[k], rep.frames[k], field_path=tmp_path / "grid" / rep.frames[k].grid_rel_path)
    d = np.abs(back - beauty[k])[48:-48, 48:-48]
    assert np.median(d) < 0.05, np.median(d)
    # without the grid the global model alone leaves the local field (the grid is not a no-op)
    err_g = composed_error(rep.frames[k], None, ux, uy)
    assert np.percentile(err_g, 90) > 1.0


def test_outliers_are_rejected_and_do_not_spoil_the_field():
    w, h = 640, 400
    ux, uy = _field(h, w)
    al = LocalPatchAligner(w, h, grid_spacing=32)
    pts = al.generate_grid_points(np.ones((h, w), np.float32), scale=1.0)
    xi, yi = pts[:, 0].round().astype(int), pts[:, 1].round().astype(int)
    truth = np.column_stack([ux[yi, xi], uy[yi, xi]]).astype(np.float64)
    rng = np.random.default_rng(5)
    n = len(pts)
    bad = rng.choice(n, n // 10, replace=False)
    noisy = truth.copy()
    noisy[bad] += rng.uniform(2.0, 6.0, (len(bad), 2)) * rng.choice([-1, 1], (len(bad), 2))
    keep, info = reject_outliers(pts, noisy)
    assert info["n"] == n and info["nbr"] >= 0.9 * len(bad)
    assert (~keep[bad]).mean() >= 0.9 and keep[np.setdiff1d(np.arange(n), bad)].mean() >= 0.97
    g_rej, _, _ = al.fit_rbf_grid(pts, noisy)
    assert al.last_n_rejected == int((~keep).sum()) and al.last_reject == info
    dx, dy = align_grid.upsample(g_rej)
    err = np.hypot(dx - ux, dy - uy)[32:-32, 32:-32]
    assert np.percentile(err, 90) < 0.3 and np.median(err) < 0.15, (np.median(err), np.percentile(err, 90))
    g_raw, _, _ = al.fit_rbf_grid(pts, noisy, reject=False)
    dx, dy = align_grid.upsample(g_raw)
    assert np.percentile(np.hypot(dx - ux, dy - uy)[32:-32, 32:-32], 90) > 0.3   # the outliers did spoil it
    # rule 1: weak correlation response (NaN = not measured -> kept); rule 2: values sitting on the +-8 px clamp
    resp = np.full(n, np.nan)
    resp[:5] = 0.05
    clipped = truth.copy()
    clipped[10:13, 0] = 8.0
    clipped[13, 1] = -8.0
    keep2, info2 = reject_outliers(pts, clipped, resp)
    assert info2["resp"] == 5 and info2["clip"] == 4 and not keep2[:5].any() and not keep2[10:14].any()
    assert keep2[20:].all()


def test_reject_outliers_cpu_equals_gpu_worker_copy():
    spec = importlib.util.spec_from_file_location("md_align_gpu_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    rng = np.random.default_rng(11)
    pts = rng.uniform(0, 1000, (700, 2))
    d = rng.normal(0, 0.1, (700, 2)) + 0.004 * pts
    d[::9] += 3.0
    d[5:9] = 8.0
    resp = rng.uniform(0, 1, 700)
    resp[::17] = np.nan
    k1, i1 = reject_outliers(pts, d, resp)
    k2, i2 = worker.reject_outliers(pts, d, resp)
    assert np.array_equal(k1, k2) and i1 == i2 and i1["resp"] > 0 and i1["clip"] > 0 and i1["nbr"] > 0


def _frame_with_homography(w, h):
    th = np.radians(0.3)
    m = np.array([[np.cos(th), -np.sin(th), 12.5], [np.sin(th), np.cos(th), -6.5], [1e-6, -2e-6, 1.0]])
    return AlignmentFrameResult(frame=1, method="local_stmap", residual_px=0.1, residual_max_px=0.2,
                                transform_matrix=m.tolist(), global_method="homography"), m


def _grid_from_field(w, h):
    ux, uy = _field(h, w)
    al = LocalPatchAligner(w, h, grid_spacing=32)
    pts = al.generate_grid_points(np.ones((h, w), np.float32), scale=1.0)
    xi, yi = pts[:, 0].round().astype(int), pts[:, 1].round().astype(int)
    grid, _, _ = al.fit_rbf_grid(pts, np.column_stack([-ux[yi, xi], -uy[yi, xi]]))
    return grid


def test_compose_remap_is_global_then_grid_like_the_nuke_chain(tmp_path):
    w, h = 480, 320
    fr, m = _frame_with_homography(w, h)
    grid = _grid_from_field(w, h)
    mx, my = compose_remap(fr, grid, w, h)
    # map level: M^-1 (x - d(x)) at every pixel
    dx, dy = align_grid.upsample(grid)
    Y, X = np.indices((h, w), dtype=np.float64)
    mi = np.linalg.inv(m)
    px, py = X - dx, Y - dy
    wq = mi[2, 0] * px + mi[2, 1] * py + mi[2, 2]
    ex = (mi[0, 0] * px + mi[0, 1] * py + mi[0, 2]) / wq
    ey = (mi[1, 0] * px + mi[1, 1] * py + mi[1, 2]) / wq
    assert np.abs(mx - ex).max() < 1e-3 and np.abs(my - ey).max() < 1e-3
    # the grid lives in beauty coordinates (measured on the pre-warped aux): sampling it in aux coordinates
    # instead (d(M^-1 x)) is a different map - the order is not a free choice
    wq0 = mi[2, 0] * X + mi[2, 1] * Y + mi[2, 2]
    gx0 = ((mi[0, 0] * X + mi[0, 1] * Y + mi[0, 2]) / wq0).astype(np.float32)
    gy0 = ((mi[1, 0] * X + mi[1, 1] * Y + mi[1, 2]) / wq0).astype(np.float32)
    wrong_x = gx0 - cv2.remap(dx, gx0, gy0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wrong_y = gy0 - cv2.remap(dy, gx0, gy0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    assert np.hypot(mx - wrong_x, my - wrong_y)[24:-24, 24:-24].max() > 0.1
    # image level: the Nuke chain (CornerPin = warpPerspective(M), then STMap of the grid) gives the same picture
    # as the single composed resampling
    tex = _texture(h, w, seed=2)
    aux = cv2.remap(tex, (X + 3).astype(np.float32), (Y - 2).astype(np.float32), cv2.INTER_CUBIC)   # any content
    one = cv2.remap(aux, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    chain = cv2.remap(warp_global(aux, m), (X - dx).astype(np.float32), (Y - dy).astype(np.float32), cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inner = (slice(24, -24), slice(24, -24))
    assert np.median(np.abs(one - chain)[inner]) < 0.01 and np.abs(one - chain)[inner].max() < 0.1
    assert np.array_equal(PlateWarper.warp_frame(aux, fr, field_path=align_grid.write_grid(tmp_path / "g.exr", grid)), one)
    # the Nuke group: CornerPin2D (global) feeds the STMap (grid); the Switch keeps the raw input as input 0
    rep = AlignmentReport("x2", "backlit", "beauty", (1, 1), "local_stmap", 0.1, 0.2, True, "", [fr])
    text = NukeAlignGroupGenerator.generate_script("x2", rep, w, h, tmp_path / "a.nk", tmp_path / "a.json")
    assert validate_nk(text, "Align_x2") == []
    assert text.index("CornerPin2D_1") < text.index("set N_global [stack 0]") < text.index("Read_Grid") \
        < text.index("push $N_global\n STMap") < text.index("push $N_dot1\n Switch")
    # STMap on request = the composed maps
    stmap_for_frame(fr, grid, tmp_path / "st.exr")
    st = cv2.imread(str(tmp_path / "st.exr"), cv2.IMREAD_UNCHANGED)
    u, v = st[..., 2], st[..., 1]           # OpenCV reads EXR as BGR: R = u in [..., 2], G = v in [..., 1]
    assert np.abs(u * w - 0.5 - mx)[inner].max() < 1e-2 and np.abs((h - 1) - (v * h - 0.5) - my)[inner].max() < 1e-2


def test_proxy_gets_both_models_scaled(tmp_path):
    w, h = 480, 320
    fr, m = _frame_with_homography(w, h)
    grid = _grid_from_field(w, h)
    gp = align_grid.write_grid(tmp_path / "g.exr", grid)
    tex = _texture(h, w, seed=4)
    full = PlateWarper.warp_frame(tex, fr, field_path=gp)
    small = cv2.resize(tex, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    back_s = PlateWarper.warp_frame(small, fr, field_path=gp)
    ref_s = cv2.resize(full, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    assert back_s.shape == ref_s.shape and np.median(np.abs(back_s - ref_s)[8:-8, 8:-8]) < 0.02
    # no grid (old report): the global model alone, still scaled to the proxy
    fr2 = AlignmentFrameResult(frame=1, method="local_stmap", residual_px=0.1, residual_max_px=0.2,
                               transform_matrix=m.tolist(), global_method="homography")
    g_only = PlateWarper.warp_frame(small, fr2, plate_w=w)
    ref_g = cv2.resize(warp_global(tex, m), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    assert np.median(np.abs(g_only - ref_g)[8:-8, 8:-8]) < 0.02


def test_old_report_grid_measured_on_raw_aux_is_dropped():
    d = {"plate_name": "x", "aux_plate_type": "cleanplate", "reference_plate": "b", "frame_range": [1, 2],
         "method": "local_stmap", "residual_median": 0.1, "residual_max": 0.2, "needs_correction": True,
         "recommendation": "",
         "frames": [{"frame": 1, "method": "local_stmap", "residual_px": 0.1, "residual_max_px": 0.2,
                     "grid_rel_path": "grid_0001.exr", "transform_matrix": [[1, 0, 2.0], [0, 1, -1.0], [0, 0, 1]]},
                    {"frame": 2, "method": "local_stmap", "residual_px": 0.1, "residual_max_px": 0.2,
                     "grid_rel_path": "grid_0002.exr", "dx": 1.5, "dy": 0.5}]}
    assert "Zmierz ponownie" in stale_report_reason(d)        # UI: badge + reason in stage 0
    rep = AlignmentReport.from_dict(d)
    assert [f.grid_rel_path for f in rep.frames] == [None, None]
    assert [f.global_method for f in rep.frames] == ["homography", "translation"]
    Y, X = np.indices((40, 60), dtype=np.float32)
    img = np.exp(-((X - 30) ** 2 + (Y - 20) ** 2) / 8.0)
    out = PlateWarper.warp_frame(img, rep.frames[1])          # global model alone: content moved by (1.5, 0.5)
    cx, cy = (out * X).sum() / out.sum(), (out * Y).sum() / out.sum()
    assert abs(cx - 31.5) < 0.05 and abs(cy - 20.5) < 0.05, (cx, cy)
    # a new report keeps its grid; round trip keeps the new fields
    f = AlignmentFrameResult(frame=1, method="local_stmap", residual_px=0.1, residual_max_px=0.2, dx=1.0,
                             global_method="translation", grid_rel_path="grid_0001.exr", n_points=100, n_rejected=3,
                             holdout_frac_ok=0.95)
    r2 = AlignmentReport.from_dict(AlignmentReport("x", "cleanplate", "b", (1, 1), "local_stmap", 0.1, 0.2, True, "", [f]).to_dict())
    assert r2.frames[0] == f
    assert stale_report_reason(AlignmentReport("x", "cleanplate", "b", (1, 1), "local_stmap", 0.1, 0.2, True, "", [f]).to_dict()) == ""
    old_stmap = {"frames": [{"frame": 1, "method": "local_stmap", "stmap_rel_path": "stmap_0001.exr", "global_method": "identity"}]}
    assert "STMapy" in stale_report_reason(old_stmap)
    assert stale_report_reason({"frames": [{"frame": 1, "method": "translation", "dx": 1.0}]}) == ""   # global only: nothing dropped


def test_degenerate_points_give_zero_grid_not_a_crash():
    """SH010 cleanplate 2026-10-15: after rejection a frame was left with collinear points and torch.linalg.solve
    raised 'singular' in the GPU worker, killing the whole 193-frame run. Both paths now return a zero grid."""
    w, h = 640, 400
    al = LocalPatchAligner(w, h, grid_spacing=32)
    line = np.column_stack([np.linspace(50, 600, 12), np.full(12, 200.0)])          # collinear
    d = np.column_stack([np.linspace(0, 1, 12), np.zeros(12)])
    grid, med, p95 = al.fit_rbf_grid(line, d, reject=False)
    assert al.last_holdout_frac_ok == 0.0 and not grid.dx.any() and not grid.dy.any()
    few = np.array([[100.0, 100.0], [300.0, 120.0], [200.0, 300.0], [400.0, 250.0], [150.0, 220.0]])
    grid, _, _ = al.fit_rbf_grid(few, np.zeros((5, 2)) + 0.5, reject=False)          # < MIN_FIT_POINTS
    assert not grid.dx.any()
    # mostly noise: 90 % of the points random, the 10 % survivors are not a field (SH010 clear plate)
    rng = np.random.default_rng(3)
    pts = rng.uniform(40, 600, (300, 2))
    noise = rng.uniform(-7.5, 7.5, (300, 2))
    noise[:30] = 0.3
    grid, _, _ = al.fit_rbf_grid(pts, noise)
    assert al.last_n_rejected > 200 and not grid.dx.any() and al.last_holdout_frac_ok == 0.0
    spec = importlib.util.spec_from_file_location("md_align_gpu_worker2", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    try:
        worker._init_torch()
    except ImportError:
        pytest.skip("no torch in this Python")
    g, med, p95, frac, n_rej, info = worker.rbf_field(line, d, w, h, reject=False)
    assert frac == 0.0 and not bool(g.abs().any())
