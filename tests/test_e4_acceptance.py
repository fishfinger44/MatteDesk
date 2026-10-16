# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E4 – user/auto keys, bidirectional core propagation, shot edges, hand-off between crop segments,
cache, undo/redo (SPEC 7.1-7.3).

Acceptance (spec 12): spoil a frame on purpose (remove an antenna in a key) -> the SAM3 core changes only in the
neighbouring intervals, elsewhere bit-identical; time <= 30 s; after "Przelicz finalnie" differences outside the
interval only inside the edge band. Cloud: synthetic puppet + mock SAM3 / MatAnyone (timing on the station:
LOCAL_CHECKS).
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.keys import KeyHistory, apply_strokes
from engine.refine import edge_band
from engine.segment import iou, suggest_keys
from test_e3_acceptance import _setup


def _antenna_stroke(shot, f):
    cx, cy, s = shot.centers[f]
    head = (cx, cy - 75 * s)
    tip = (head[0] + 45 * s, head[1] - 70 * s)
    mid = ((head[0] + tip[0]) / 2, (head[1] + tip[1]) / 2)
    return [{"mode": "remove", "radius_px": 6, "p": [list(mid), list(tip)]}]


def test_e4_acceptance_incremental_core(tmp_path):
    shot, pipe, client, frames = _setup(tmp_path, n=30)
    start = pipe.store.get("puppet")["tool"]["start_frame"]
    hist = KeyHistory(pipe.store, "puppet")
    # second user key near the end, so there are intervals on both sides of the spoiled frame
    k_end = frames[-4]
    base = pipe.run_core("puppet", frames)
    hist.save_key(k_end, base.core[k_end])
    before = pipe.run_core("puppet", frames)
    full_before, _, _ = pipe.finalize("puppet", frames, edge_mode="matanyone")

    # spoil: new key between start and k_end with one antenna removed
    k = start + 5
    spoiled = apply_strokes(before.core[k], _antenna_stroke(shot, k))
    assert spoiled.sum() < before.core[k].sum()
    hist.save_key(k, spoiled)
    t0 = time.time()
    after = pipe.run_core("puppet", frames)
    dt = time.time() - t0

    assert sorted(after.recomputed_intervals) == [(start, k), (k, k_end)]
    assert (frames[0], start) in after.cached_intervals and (k_end, frames[-1]) in after.cached_intervals
    assert np.array_equal(after.core[k], spoiled)  # key = hard truth in its frame
    for f in frames:
        if f < start or f > k_end:
            assert np.array_equal(after.core[f], before.core[f]), f"core changed outside the neighbours: {f}"
    assert dt <= 30.0

    # after 'Przelicz finalnie': outside the interval differences only inside the edge band
    full_after, rep, _ = pipe.finalize("puppet", frames, edge_mode="matanyone")
    for f in frames:
        if start <= f <= k_end:
            continue
        diff = np.abs(full_after[f] - full_before[f]) > 1e-3
        band = edge_band(after.core[f], rep.band_px + 1)
        assert not (diff & ~band).any(), f"final changed outside the edge band in frame {f}"


def test_undo_redo_keys(tmp_path):
    shot, pipe, client, frames = _setup(tmp_path, n=8)
    hist = KeyHistory(pipe.store, "puppet")
    f = frames[2]
    m1 = shot.masks[f] > 0.5
    hist.save_key(f, m1)
    m2 = apply_strokes(m1, [{"mode": "add", "radius_px": 20, "p": [[100, 100]]}])
    hist.save_key(f, m2)
    assert np.array_equal(pipe.store.key_mask("puppet", f), m2)
    hist.undo()
    assert np.array_equal(pipe.store.key_mask("puppet", f), m1)
    hist.undo()
    assert f not in pipe.store.keys("puppet")
    hist.redo()
    hist.redo()
    assert np.array_equal(pipe.store.key_mask("puppet", f), m2)
    hist.delete_key(f)
    assert f not in pipe.store.keys("puppet")
    hist.undo()
    assert np.array_equal(pipe.store.key_mask("puppet", f), m2)


def test_auto_key_never_overwrites_user(tmp_path):
    shot, pipe, client, frames = _setup(tmp_path, n=6)
    f = frames[1]
    pipe.store.save_key("puppet", f, shot.masks[f] > 0.5, author="user")
    pipe.store.save_key("puppet", f, np.zeros_like(shot.masks[f], bool), author="auto")
    assert pipe.store.keys("puppet")[f]["author"] == "user" and pipe.store.key_mask("puppet", f).any()


def test_suggest_keys_halving():
    assert suggest_keys([3, 4, 5, 6, 7, 20, 21]) == [5, 21]


def test_api_keys_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    from synth import display
    shot, pipe, client, frames = _setup(tmp_path, n=10)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    assert api.post("/api/object/puppet/run").status_code == 200
    f = frames[3]
    r = api.post(f"/api/object/puppet/keys/{f}", json={"source": "core", "strokes": _antenna_stroke(shot, f)})
    assert r.status_code == 200 and r.json()["key"]["author"] == "user" and r.json()["key"]["kind"] == "brush"
    assert api.delete(f"/api/object/puppet/keys/{f}").status_code == 400  # needs confirmation
    rr = api.post(f"/api/object/puppet/rerun_from/{f}").json()
    assert rr["recomputed"] and rr["cached"] is not None
    assert api.post("/api/object/puppet/undo").json()["result"]["op"] == "save_key"
    assert f not in [k["frame"] for k in api.get("/api/object/puppet/keys").json()["keys"]]
    assert api.post("/api/object/puppet/redo").json()["success"]
    assert api.delete(f"/api/object/puppet/keys/{f}?confirm=true").status_code == 200
