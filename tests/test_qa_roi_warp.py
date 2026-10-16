# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""QA flicker warp on the object's box only (2026-10-13, measured on SH010: QA 174 -> 138 s, qa.json + flicker_map bitwise
the same). Here: synthetic shot, camera shifts fractional / both signs / near and over the 40 px box margin (-> full-frame
fallback); the report and the flicker map must equal the full-frame reference byte for byte."""

import json

import cv2
import numpy as np
import pytest

import engine.qa as qa_mod
from engine.qa import run_qa

W, H = 900, 640   # != 4096 x 2731


def _shot(n=10):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    al, gray = {}, {}
    for i in range(n):
        cx, cy = 450 + 3.3 * i, 300 - 1.7 * i
        r = np.hypot(xx - cx, yy - cy)
        a = np.clip((120 - r) / 4 + 0.5, 0, 1).astype(np.float32)
        if i == 4:
            a[290:300, 440:452] = 0                       # a hole
        al[1001 + i] = a
        gray[1001 + i] = (rng.random((H, W)) * 255).astype(np.uint8)
    return al, gray


SHIFTS = [(2.25, -1.5), (-38.9, 0.3), (39.0, -2.0), (0.7, 45.0), (-12.125, 7.75), (0.0, 0.0), (5.5, -38.5),
          (-0.33, 0.66), (20.0, -20.0)]   # fractional, both signs, at / over the margin (38.9 -> box; 39.0 / 45 -> full frame)


@pytest.mark.parametrize("shifts", [None, "big"])
def test_roi_warp_is_bitwise_the_full_frame_warp(tmp_path, monkeypatch, shifts):
    al, gray = _shot()
    frames = sorted(al)
    if shifts == "big":   # pairs run in threads: the shift is chosen by the frame (its index is stored in gray[0, 0])
        for i, f in enumerate(frames):
            gray[f][0, 0] = i
        monkeypatch.setattr(qa_mod, "camera_shift", lambda g0, g1: np.array(
            [[1, 0, SHIFTS[int(g1[0, 0]) - 1][0]], [0, 1, SHIFTS[int(g1[0, 0]) - 1][1]]], np.float32))
    out = {}
    for roi in (True, False):
        monkeypatch.setattr(qa_mod, "ROI_WARP", roi)
        fm = tmp_path / f"fm_{roi}.png"
        rep = run_qa(frames, lambda f: al[f], lambda f: gray[f], {}, seg_boundaries=[1006], attempts=1,
                     flicker_map_path=fm, flow_check=True, load_clean_gray=lambda f: gray[f], workers=4)
        out[roi] = (json.dumps(rep.to_dict(), sort_keys=True), fm.read_bytes())
    assert out[True][0] == out[False][0]
    assert out[True][1] == out[False][1]
    assert json.loads(out[True][0])["holes"]["1005"] == 1


def test_running_max_bbox_equals_stack_max():
    al, _ = _shot(6)
    ref = np.max(np.stack(list(al.values())), 0) > 0.05
    u = next(iter(al.values())).copy()
    for a in list(al.values())[1:]:
        np.maximum(u, a, out=u)
    assert np.array_equal(u > 0.05, ref)
