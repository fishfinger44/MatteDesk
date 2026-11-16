# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""align_v2 step 2 (2026-11-03): „kamera zablokowana” (stop motion: beauty and clear plate shot with the same locked
camera). Contract on the synthetic shot (tests/synth.make_align_shot, known truth): auto detects the lock from the
per-frame global models (non-translational shake under the threshold), one global model + one local field from the
median of the anchor measurements beat the per-frame fields (noise ~ 1 / sqrt(N)); a per-frame rotation shake is
NOT locked by auto but is when forced; the report shows the mode and the gain."""

import numpy as np
import pytest

from engine import align_grid
from engine.align import LOCK_SHAKE_PX, AlignmentPipeline, AlignmentReport, detect_camera_lock
from engine.tracking import LKTracker
from tests.synth import make_align_shot
from tests.test_align_compose import composed_error


def _measure(shot, grid_dir, **kw):
    return AlignmentPipeline.measure_sequence(beauty_pattern=shot.beauty_pattern, aux_pattern=shot.aux_pattern,
                                             first_frame=shot.first, last_frame=shot.last, plate_name="cleanplate",
                                             aux_plate_type="cleanplate", plate_w=shot.w, plate_h=shot.h,
                                             grid_dir=grid_dir, tracker=LKTracker(), workers=1, device="cpu", **kw)


def _truth_error(shot, rep, grid_dir, frames):
    errs = []
    for fr in rep.frames:
        if fr.frame not in frames:
            continue
        grid = align_grid.read_grid(grid_dir / fr.grid_rel_path) if fr.grid_rel_path else None
        ux, uy = shot.truth(fr.frame)
        e = composed_error(fr, grid, ux, uy, margin=40)
        z = shot.zone[40:-40, 40:-40]
        errs.append(float(np.median(e[~z])))
    return float(np.median(errs))


def test_detect_camera_lock_from_frame_models():
    from engine.align import AlignmentFrameResult as R
    w, h = 1000, 600
    fr = [R(frame=i, method="translation", residual_px=0.1, residual_max_px=0.2, dx=1.0 + 0.1 * (i % 3), dy=-0.5)
          for i in range(10)]
    d = detect_camera_lock(fr, w, h, "auto")
    assert d["locked"] and d["shake_p95_px"] < 0.05 and d["jitter_p95_px"] < 0.3 and d["global_method"] == "translation"
    # one frame with a 0.5° rotation = shake well above the threshold -> auto says no, "on" forces it
    import cv2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.5, 1.0)
    fr[4] = R(frame=4, method="similarity", residual_px=0.1, residual_max_px=0.2, transform_matrix=M.tolist())
    d = detect_camera_lock(fr, w, h, "auto")
    assert not d["locked"] and d["shake_p95_px"] > LOCK_SHAKE_PX and "drganie" in d["reason"]
    d = detect_camera_lock(fr, w, h, "on")
    assert d["locked"] and d["mode"] == "on"
    assert not detect_camera_lock(fr, w, h, "off")["locked"]


def test_locked_camera_median_field_beats_per_frame_fields(tmp_path):
    shot = make_align_shot(tmp_path / "shot", n=25, w=800, h=450, shift=(2.4, -1.2), rot_deg=0.15, local_amp=1.5,
                           noise=0.1, with_object=True)
    frames = list(range(shot.first, shot.last + 1))
    off = _measure(shot, tmp_path / "g_off", camera_lock="off", anchor_every=4)
    on = _measure(shot, tmp_path / "g_on", camera_lock="auto", anchor_every=4)
    assert not off.camera_lock["locked"] and off.camera_lock["mode"] == "off"
    cl = on.camera_lock
    assert cl["locked"] and cl["mode"] == "auto" and cl["anchors"] == 7, cl
    assert [f.method for f in on.frames] == ["local_stmap"] * len(frames)
    assert len({f.grid_rel_path for f in on.frames}) == 1 and (tmp_path / "g_on" / on.frames[0].grid_rel_path).exists()
    assert cl["local_after_px"] < cl["local_before_px"], cl
    e_off = _truth_error(shot, off, tmp_path / "g_off", frames[::6])
    e_on = _truth_error(shot, on, tmp_path / "g_on", frames[::6])
    # the median over 7 anchors removes the noise part of the error (the correlation bias of a patch is the same on
    # every frame of a locked camera and stays): better against the truth, not just equal
    assert e_on < e_off, (e_on, e_off)
    assert "zablokowana" in on.recommendation


def test_median_of_anchors_follows_sqrt_n():
    """The median step alone (engine.align.median_anchor_field): N anchors = truth + independent noise -> the error
    of the median against the truth shrinks ~ 1 / sqrt(N) (median of 7 Gaussians: ~0.45 sigma), and the fair
    before / after of the report shows the gain."""
    from engine.align import median_anchor_field
    rng = np.random.default_rng(1)
    w, h = 800, 450
    xs, ys = np.meshgrid(np.arange(64, w - 64, 48), np.arange(64, h - 64, 48))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    truth = np.column_stack([1.5 * np.sin(2 * np.pi * pts[:, 0] / w), -1.0 * np.cos(np.pi * pts[:, 1] / h)]).astype(np.float32)
    sigma = 0.3
    per, ref = [], []
    for _ in range(7):
        d = truth + rng.normal(0, sigma, truth.shape).astype(np.float32)
        per.append((pts, d))
        ref.append((d, np.full(len(pts), 0.9)))
    pm, dm, rm, before, after = median_anchor_field(per, ref, w, h)
    assert len(pm) == len(pts) and np.isfinite(rm).all()
    e1 = np.median(np.hypot(*(ref[0][0] - truth).T))
    em = np.median(np.hypot(*(dm - truth).T))
    assert em < 0.6 * e1, (em, e1)
    assert after < before, (before, after)
    # one anchor only: nothing to average, before == after (its own hold-out)
    pm1, dm1, _r, b1, a1 = median_anchor_field(per[:1], ref[:1], w, h)
    assert np.allclose(dm1, per[0][1]) and b1 == a1


def test_rotation_shake_is_not_locked_by_auto(tmp_path):
    shot = make_align_shot(tmp_path / "shot", n=9, w=640, h=360, shift=(1.5, -0.8), rot_deg=0.1, rot_jitter_deg=0.4,
                           with_object=False)
    rep = _measure(shot, tmp_path / "g", camera_lock="auto", anchor_every=4)
    assert not rep.camera_lock["locked"] and rep.camera_lock["shake_p95_px"] > LOCK_SHAKE_PX
    forced = _measure(shot, tmp_path / "g2", camera_lock="on", anchor_every=4)
    assert forced.camera_lock["locked"] and forced.camera_lock["mode"] == "on"
    with pytest.raises(ValueError):
        _measure(shot, tmp_path / "g3", camera_lock="maybe")


def test_fullres_check_is_honest_and_triggers_the_local_model(tmp_path):
    """Step 6a (option): the full-res residual at the anchors measures the composed alignment's real miss (within
    0.1 px of the truth, where the 21 px evaluator's residual_px underestimates), decides the local model and is
    reported before / after it."""
    from engine.align_residual import frame_residual
    shot = make_align_shot(tmp_path / "shot", n=9, w=800, h=450, shift=(2.4, -1.2), rot_deg=0.15, local_amp=1.5,
                           with_object=False)
    frames = list(range(shot.first, shot.last + 1))
    rep = _measure(shot, tmp_path / "g", camera_lock="off", anchor_every=4, fullres_check=True)
    g, l = rep.checks["global"], rep.checks["local"]
    assert sorted(g["frames"]) == [1, 5, 9] and g["median"] is not None
    assert g["median"] >= 0.5 and [f.method for f in rep.frames] == ["local_stmap"] * len(frames)
    assert l["median"] < g["median"] and "punktach kontrolnych" in rep.recommendation
    # honesty: the global check equals the truth error of the global model on its control points
    fr = rep.frames[0]
    from engine.align import _frame_matrix_cv, AlignmentFrameResult
    g0 = AlignmentFrameResult(frame=fr.frame, method=fr.global_method, residual_px=0, residual_max_px=0, dx=fr.dx, dy=fr.dy,
                              transform_matrix=fr.transform_matrix)
    e = composed_error(g0, None, *shot.truth(fr.frame), margin=48)
    assert abs(g["frames"][fr.frame]["median"] - float(np.median(e))) < 0.15, (g["frames"][fr.frame], np.median(e))
    # off by default: no checks
    rep0 = _measure(shot, tmp_path / "g0", camera_lock="off", anchor_every=4)
    assert rep0.checks == {}


def test_global_only_never_warps(tmp_path):
    """User 2026-11-10, "tylko transform": a motion-control clear plate with a constant camera offset gets only the
    global Transform / CornerPin - no local grid even where the global model leaves a residual; the report says so."""
    shot = make_align_shot(tmp_path / "shot", n=9, w=800, h=450, shift=(2.4, -1.2), rot_deg=0.15, local_amp=1.5,
                           with_object=False)
    rep = _measure(shot, tmp_path / "g", camera_lock="off", anchor_every=4, global_only=True)
    assert rep.global_only and all(f.method != "local_stmap" and not f.grid_rel_path for f in rep.frames)
    assert not list((tmp_path / "g").glob("*")) and "tylko transform" in rep.recommendation
    assert AlignmentReport.from_dict(rep.to_dict()).global_only
