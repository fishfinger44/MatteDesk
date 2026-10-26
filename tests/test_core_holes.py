# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Holes of the core (mask 1+2) survive alpha hygiene and are not QA defects (sh060 F71 2026-10-26: the gap between
arm and body, open in SAM 3.1 / mask 1+2, was filled to 1 whenever MatAnyone2 closed it off from the background)."""

import types

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.edge import enclosed_holes, hygiene


def _ring(n=96, outer=(16, 80), hole=(36, 60)):
    core = np.zeros((n, n), bool)
    core[outer[0]:outer[1], outer[0]:outer[1]] = True
    core[hole[0]:hole[1], hole[0]:hole[1]] = False
    return core


def test_enclosed_holes():
    core = _ring()
    h = enclosed_holes(core)
    assert h[40:56, 40:56].all() and not h[:10].any() and h.sum() == 24 * 24
    # a gap open to the frame border is not a hole
    open_ = core.copy()
    open_[44:52, 0:40] = False
    assert not enclosed_holes(open_).any()
    assert not enclosed_holes(np.zeros((8, 8), bool)).any()


def test_hygiene_keeps_core_hole_fills_false_hole():
    core = _ring()
    a = core.astype(np.float32)
    # the band (±4 px around every core contour) covers the hole's rim; the hole's middle is beyond the band (0)
    band = cv2.dilate(core.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & \
        ~cv2.erode(core.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    a[band & ~core] = 0.0
    # a false hole punched by the model in the solid core, inside the band of the outer contour
    a[18:22, 30:34] = 0.0
    out, rep = hygiene(a.copy(), core, band)
    assert out[38:58, 38:58].max() == 0.0            # the real gap stays open
    assert out[18:22, 30:34].min() == 1.0            # the false hole is filled
    assert rep["holes_filled"] == 1


def test_hygiene_core_hole_partly_closed_by_model():
    """The model shrinks the gap (alpha > 0.5 on its rim inside the band): what is left of it is still the core's hole."""
    core = _ring()
    band = cv2.dilate(core.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & \
        ~cv2.erode(core.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    a = core.astype(np.float32)
    a[36:60, 36:60] = 0.0
    a[36:40, 36:60] = 0.8                             # model alpha over the hole's rim
    out, rep = hygiene(a.copy(), core, band)
    assert out[44:58, 40:56].max() == 0.0 and rep["holes_filled"] == 0


def test_run_qa_allows_core_holes(tmp_path, monkeypatch):
    """Pipeline QA: holes of the run's core (core/<seg>/c_####.png, pasted to the frame) are allowed holes."""
    from engine import pipeline, qa
    n, w, h = 96, 160, 120
    seg = CropSegment.from_dict({"seg_id": "s00", "first": 0, "last": 0, "crop": [20, 10, n, n], "pad": [0, 0, 0, 0],
                                 "overlap_in": 0, "overlap_out": 0, "scale_to_1008": 1.0})
    run_dir = tmp_path / "v001"
    (run_dir / "core" / "s00").mkdir(parents=True)
    cv2.imwrite(str(run_dir / "core" / "s00" / "c_0000.png"), _ring().astype(np.uint8) * 255)
    got = {}

    def fake_run_qa(frames, get, gray, fb, allowed_holes=None, **kw):
        got["allowed"] = allowed_holes
        return types.SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(qa, "run_qa", fake_run_qa)
    obj_dir = tmp_path / "obj"
    obj_dir.mkdir()
    fake = types.SimpleNamespace(
        store=types.SimpleNamespace(get=lambda name: {"tool": {}}, dir=lambda name: obj_dir),
        loader=None, clean=None, w=w, h=h)
    pipeline.ObjectPipeline.run_qa(fake, "obj", [0], {0: np.zeros((h, w), np.float32)}, [seg], run_dir)
    al = got["allowed"](0)
    assert al.shape == (h, w)
    assert al[10 + 40:10 + 56, 20 + 40:20 + 56].all() and al.sum() == 24 * 24
    # the QA hole count with that mask: the core's own gap is not a defect, a new one is
    alpha = np.zeros((h, w), np.float32)
    alpha[10:10 + n, 20:20 + n] = _ring()
    assert qa.interior_holes(alpha, allowed=al) == 0
    alpha[10 + 20:10 + 26, 20 + 20:20 + 26] = 0.0
    assert qa.interior_holes(alpha, allowed=al) == 1
