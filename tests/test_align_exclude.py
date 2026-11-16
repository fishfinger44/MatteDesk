# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""align_v2 step 1 (2026-11-03): exclusions of the plate alignment. Contract: on the synthetic shot with an object
walking across the whole frame (tests/synth.make_align_shot) the residual of the aligned clear plate on the control
points inside the zone the object passed through does not grow above the residual outside it when the objects'
cores are excluded; the static mask of the local model is built from every anchor frame; the report records what
was excluded (objects + mask signatures, dilation, box); the API builds the spec from the project's objects."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import align_grid
from engine.align import AlignmentPipeline, compute_static_mask
from engine.align_exclude import (exclusion_spec, frame_exclusion, objects_with_cores, report_fields,
                                  union_exclusion)
from engine.align_residual import frame_residual, zone_stats
from engine.tracking import LKTracker
from tests.synth import make_align_shot


def _measure(shot, grid_dir, **kw):
    return AlignmentPipeline.measure_sequence(beauty_pattern=shot.beauty_pattern, aux_pattern=shot.aux_pattern,
                                             first_frame=shot.first, last_frame=shot.last, plate_name="cleanplate",
                                             aux_plate_type="cleanplate", plate_w=shot.w, plate_h=shot.h,
                                             grid_dir=grid_dir, tracker=LKTracker(), workers=1, device="cpu", **kw)


def test_static_mask_takes_an_exclusion_mask():
    beauty = np.ones((200, 200, 3), np.float32)
    ex = np.zeros((200, 200), bool)
    ex[40:90, 50:120] = True
    m = compute_static_mask(beauty, beauty, mode="backlit", scale=0.5, exclude_mask=ex)
    assert m.shape == (100, 100) and m[30, 40] == 0.0 and m[70, 40] == 1.0
    m1 = compute_static_mask(beauty, beauty, mode="backlit", scale=0.5)
    assert m1[30, 40] == 1.0


def test_exclusion_spec_and_masks(tmp_path):
    shot = make_align_shot(tmp_path / "shot", n=4, w=320, h=180, with_object=True)
    work = tmp_path / "work"
    shot.write_object_cores(work, "puppet")
    assert objects_with_cores(work) == ["puppet"]
    spec = exclusion_spec(work, ["puppet"], range(shot.first, shot.last + 1), dilate_px=8, box=[1, 2, 3, 4])
    assert spec["objects"] == ["puppet"] and len(spec["paths"]) == 4 and spec["box"] == [1, 2, 3, 4]
    assert set(spec["object_sigs"]) == {"puppet"} and spec["object_sigs"]["puppet"]
    f = shot.first
    ex = frame_exclusion(spec, f, shot.w, shot.h)
    assert ex.dtype == bool and ex.sum() > shot.masks[f].sum() and ex[shot.masks[f]].all()
    u = union_exclusion(spec, range(shot.first, shot.last + 1), shot.w, shot.h)
    assert u.sum() >= ex.sum() and u[shot.masks[shot.last]].all()
    assert report_fields(spec) == {"objects": ["puppet"], "object_sigs": spec["object_sigs"], "dilate_px": 8,
                                   "box": [1, 2, 3, 4], "user_mask": None}
    assert frame_exclusion(None, f, 10, 10) is None and exclusion_spec(None, [], [1])["paths"] == {}


def test_object_crossing_the_frame_does_not_spoil_the_zone_it_passed(tmp_path):
    """The old first-frame static mask put grid points where the object only arrives later; with the object's cores
    excluded (and the mask from every anchor) the residual inside that zone is as good as outside it."""
    shot = make_align_shot(tmp_path / "shot", n=9, w=800, h=450, shift=(2.4, -1.2), rot_deg=0.2, local_amp=1.5)
    work = tmp_path / "work"
    shot.write_object_cores(work, "puppet")
    frames = list(range(shot.first, shot.last + 1))
    zone = shot.zone
    spec = exclusion_spec(work, ["puppet"], frames, dilate_px=32)   # = the zone's growth (AlignSynth.zone)
    # the static mask itself: the old first-frame difference mask kept most of the zone (the object is not there yet),
    # the all-anchor mask minus the objects' cores keeps none of it
    from engine.align import _frame_matrix_cv, AlignmentFrameResult
    b0 = cv2.imread(shot.beauty_pattern.replace("####", f"{shot.first:04d}"), cv2.IMREAD_UNCHANGED)
    a0 = cv2.imread(shot.aux_pattern.replace("####", f"{shot.first:04d}"), cv2.IMREAD_UNCHANGED)
    old = cv2.resize(compute_static_mask(b0, a0, mode="cleanplate", scale=0.5), (shot.w, shot.h), interpolation=cv2.INTER_NEAREST)
    assert old[zone].mean() > 0.3, old[zone].mean()
    paths = [(shot.beauty_pattern.replace("####", f"{f:04d}"), shot.aux_pattern.replace("####", f"{f:04d}")) for f in frames]
    mats = [np.eye(3)] * len(frames)
    new, per = AlignmentPipeline.local_static_mask(paths, mats, [0, len(frames) - 1], frames, "cleanplate", shot.w, shot.h,
                                                   None, spec)
    assert len(per) == 2 and new.mean() > 0.2
    # each anchor excludes the object where it stands on THAT frame (dilated); the grid (OR) still covers the zone
    # elsewhere, so the field there is measured, not interpolated
    half = lambda m: cv2.resize(m.astype(np.uint8), (per[0].shape[1], per[0].shape[0]), interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    assert not per[0][half(shot.masks[shot.first])].any() and not per[1][half(shot.masks[shot.last])].any()
    assert per[1][half(shot.masks[shot.first])].mean() > 0.5 and new[zone].mean() > 0.3
    rep = _measure(shot, tmp_path / "g_ex", exclusion=spec)
    assert rep.exclusions["objects"] == ["puppet"] and rep.exclusions["dilate_px"] == 32
    assert rep.exclusions.get("static_mask") == "per_anchor" and 0.2 < rep.exclusions["static_frac"] < 1.0
    assert [f.method for f in rep.frames] == ["local_stmap"] * len(frames)
    worst_in, worst_out = 0.0, 0.0
    for fr in rep.frames[::4]:
        f = fr.frame
        b = cv2.imread(shot.beauty_pattern.replace("####", f"{f:04d}"), cv2.IMREAD_UNCHANGED)
        a = cv2.imread(shot.aux_pattern.replace("####", f"{f:04d}"), cv2.IMREAD_UNCHANGED)
        grid = align_grid.read_grid(tmp_path / "g_ex" / fr.grid_rel_path)
        out = frame_residual(b, a, fr, grid, mode="cleanplate", exclude=shot.masks[f])
        zs = zone_stats(out["pts"], out["res"], zone)
        assert zs["in"]["n"] >= 10 and zs["out"]["n"] >= 10, zs
        worst_in = max(worst_in, zs["in"]["median"])
        worst_out = max(worst_out, zs["out"]["median"])
    assert worst_in < 0.3 and worst_in <= worst_out + 0.1, (worst_in, worst_out)


def test_api_measure_builds_the_exclusion_from_the_project(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_zone_matte import _api_project
    from engine.align import AlignmentReport
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    work = Path(app_module.STATE["work_dir"])
    # a clear plate = the beauty itself (the measurement is mocked), two objects with cores, one group
    app_module.STATE["shot_project"].set_plate("cleanplate", shot.plate_pattern)
    app_module._refresh_manifest_view()
    for n in ("obj", "hand"):
        cd = work / "objects" / n / "core"
        cd.mkdir(parents=True, exist_ok=True)
        for f in frames:
            cv2.imwrite(str(cd / f"c_{f:04d}.png"), np.zeros((shot.h, shot.w), np.uint8))
    pipe.store.create("hand", ["hand"])
    seen = {}

    def fake_measure(**kw):
        seen.update(kw)
        return AlignmentReport("cleanplate", "cleanplate", "beauty", (kw["first_frame"], kw["last_frame"]), "translation",
                               0.1, 0.2, False, "ok", frames=[], exclusions=kw.get("exclusion") and
                               {k: kw["exclusion"][k] for k in ("objects", "object_sigs", "dilate_px", "box", "user_mask")})
    import engine.align as al
    monkeypatch.setattr(al.AlignmentPipeline, "measure_sequence", staticmethod(fake_measure))
    st = api.get("/api/project/stages").json()
    assert sorted(o["name"] for o in st["align_objects"]) == ["hand", "obj"]
    r = api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate",
                                                   "exclude_objects": ["obj"], "exclude_dilate_px": 20,
                                                   "exclude_box": [10, 20, 100, 200]})
    assert r.status_code == 200, r.text
    assert seen["camera_lock"] == "auto"
    ex = seen["exclusion"]
    assert ex["objects"] == ["obj"] and ex["dilate_px"] == 20 and ex["box"] == [10, 20, 100, 200]
    assert len(ex["paths"]) == len(frames) and all(p.endswith(".png") for ps in ex["paths"].values() for p in ps)
    assert r.json()["exclusions"]["objects"] == ["obj"] and r.json()["exclusions"]["object_sigs"]["obj"]
    # default: every object with a core; [] = none
    r = api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate"})
    assert r.status_code == 200 and sorted(seen["exclusion"]["objects"]) == ["hand", "obj"]
    r = api.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate", "exclude_objects": [],
                                                   "camera_lock": "on"})
    assert r.status_code == 200 and (seen["exclusion"] is None or seen["exclusion"]["objects"] == [])
    assert seen["camera_lock"] == "on"
    assert api.post("/api/plate/align/measure", json={"plate": "cleanplate", "camera_lock": "x"}).status_code == 400
    st = api.get("/api/project/stages").json()
    assert st["align"]["cleanplate"]["exclusions"]["objects"] == []
    with pytest.raises(Exception):
        api.post("/api/plate/align/measure", json={"plate": "cleanplate", "exclude_box": [1, 2, 3]}).raise_for_status()


def test_no_texture_in_the_static_area_is_not_a_zero_residual(tmp_path):
    """2026-11-03 (_fotage_testy): with every textured object excluded the global model used to report "translation 0,
    residual 0.1 px" from a feature-difference fallback. Now such a frame is NO_TEXTURE (identity, residual -1), the
    report says so, and the measured frames carry the statistics."""
    from engine.align import NO_TEXTURE, GlobalAligner, AlignmentFeatureExtractor
    shot = make_align_shot(tmp_path / "shot", n=4, w=640, h=360, shift=(2.0, -1.0), rot_deg=0.0, with_object=False)
    beauty = cv2.imread(shot.beauty_pattern.replace("####", f"{shot.first:04d}"), cv2.IMREAD_UNCHANGED)
    aux = cv2.imread(shot.aux_pattern.replace("####", f"{shot.first:04d}"), cv2.IMREAD_UNCHANGED)
    fb = AlignmentFeatureExtractor.extract(beauty, mode="cleanplate", scale=0.5)
    fa = AlignmentFeatureExtractor.extract(aux, mode="cleanplate", scale=0.5)
    empty = np.zeros(fb.shape, np.float32)
    med, mx = GlobalAligner(scale=0.5).evaluate_model_residual(fa, fb, empty, "translation", {"dx": 0.0, "dy": 0.0})
    assert np.isnan(med) and np.isnan(mx)
    # a box covering the whole plate = nothing to measure on any frame
    spec = exclusion_spec(None, [], [], box=[0, 0, shot.w, shot.h])
    with pytest.raises(ValueError, match="tekstury"):
        _measure(shot, tmp_path / "g0", exclusion=spec)
    # a box leaving a 60 px strip: still measurable, no NO_TEXTURE
    spec3 = exclusion_spec(None, [], [], box=[0, 0, shot.w, shot.h - 60])
    rep = _measure(shot, tmp_path / "g1", exclusion=spec3, camera_lock="off")
    assert all(f.status != NO_TEXTURE for f in rep.frames)
    assert rep.residual_median >= 0 and "bez tekstury" not in rep.recommendation
    rep0 = _measure(shot, tmp_path / "g2", camera_lock="off")
    # the whole-frame phase correlation at scale 0.5 lands within ~0.5 px (its own residual says 0.2 px: the
    # evaluator's weakness, measured 2026-11-03 - see NEXT_STEPS step 6)
    assert all(abs(f.dx + 2.0) < 0.5 and abs(f.dy - 1.0) < 0.5 for f in rep0.frames)
