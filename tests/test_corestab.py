# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/corestab.stabilize_core – temporal stabilisation of the binary SAM3 core (user decision 2026-10-13)."""

import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.corestab import stabilize_core


def _pan(n=9, step=2, h=160, w=240, r=40, seed=0):
    """Textured background panning `step` px/frame with a disc (the core) moving with it; frame 4 has a bite (jitter)
    of the edge, frame 6 a bump. Returns (cores, true cores, grays)."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.uniform(0.2, 0.8, (h, w + step * n)).astype(np.float32), (0, 0), 0.7)
    cores, truth, grays = {}, {}, {}
    for i in range(n):
        g = big[:, step * i:step * i + w].copy()
        c = np.zeros((h, w), np.uint8)
        cx, cy = 80 + step * i, h // 2
        cv2.circle(c, (cx, cy), r, 1, -1)
        g[c > 0] = g[c > 0] * 0.5 + 0.5          # the object is brighter (texture kept)
        t = c.copy()
        if i == 4:                                 # jitter: a 6 px bite out of the edge
            cv2.circle(c, (cx + r, cy), 7, 0, -1)
        if i == 6:                                 # jitter: a 6 px bump
            cv2.circle(c, (cx - r, cy + 10), 6, 1, -1)
        cores[i + 1] = c.astype(bool)
        truth[i + 1] = t.astype(bool)
        grays[i + 1] = g
    return cores, truth, grays


def test_stabilize_core_removes_one_frame_jitter_keeps_interior_and_keys():
    cores, truth, grays = _pan()
    out, st = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8)
    assert st["frames"] == 9 and st["changed_px_per_frame"] > 0
    for f in cores:
        # interior (> 8 px inside the frame's own core) untouched, far background untouched
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        inner = cv2.erode(cores[f].astype(np.uint8), k).astype(bool)
        outer = ~cv2.dilate(cores[f].astype(np.uint8), k).astype(bool)
        assert np.array_equal(out[f][inner], cores[f][inner]) and np.array_equal(out[f][outer], cores[f][outer])
    # the bite (frame 5) and the bump (frame 7) are mostly repaired; clean frames stay close to themselves
    bite = cores[5] != truth[5]
    bump = cores[7] != truth[7]
    assert (out[5] == truth[5])[bite].mean() > 0.7, (out[5] == truth[5])[bite].mean()
    assert (out[7] == truth[7])[bump].mean() > 0.7, (out[7] == truth[7])[bump].mean()
    for f in (2, 3, 8):
        assert (out[f] != cores[f]).sum() < 0.15 * (cores[f] != truth[f]).sum() + 60
    # keys stay hard: the jittered frame listed as a key is returned bit-identical
    out2, _ = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, keys=[5])
    assert np.array_equal(out2[5], cores[5])


def test_stabilize_core_static_scene_is_bitwise_and_short_stacks_pass_through():
    h, w = 120, 160
    c = np.zeros((h, w), np.uint8); cv2.circle(c, (80, 60), 30, 1, -1)
    rng = np.random.default_rng(1)
    g = cv2.GaussianBlur(rng.uniform(0.2, 0.8, (h, w)).astype(np.float32), (0, 0), 0.7)
    cores = {f: c.astype(bool) for f in range(1, 6)}
    out, st = stabilize_core(cores, lambda f: g, radius=2)
    for f in cores:
        assert np.array_equal(out[f], cores[f])
    assert st["changed_px_per_frame"] == 0.0
    out2, st2 = stabilize_core({1: cores[1], 2: cores[2]}, lambda f: g, radius=2)
    assert np.array_equal(out2[1], cores[1]) and st2["changed_px_per_frame"] == 0.0


def test_api_core_stabilize_option_finalize_and_apply(tmp_path, monkeypatch):
    """Object option core_stabilize -> finalize records it in run.json; 'Zastosuj' on that version compares against the
    same stabilised core (no false 'core changed'); option off = the old path (no key in run.json)."""
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    r = api.put("/api/object/obj/settings", json={"core_stabilize": True})
    assert r.status_code == 200 and r.json()["object"]["core_stabilize"] is True
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "chroma", "preset": "hair"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "matanyone"})
    assert fz.status_code == 200, fz.text
    rj = json.loads((pipe.store.dir("obj") / "runs" / fz.json()["run"] / "run.json").read_text(encoding="utf-8"))
    assert rj.get("core_stabilize") is True and rj["core_stabilize_stats"]["frames"] == len(frames)
    fa = api.post(f"/api/object/obj/zones/{zid}/apply", json={})
    assert fa.status_code == 200, fa.text
    r = api.put("/api/object/obj/settings", json={"core_stabilize": False})
    assert r.status_code == 200
    fz2 = api.post("/api/object/obj/finalize", json={"edge_mode": "matanyone"})
    rj2 = json.loads((pipe.store.dir("obj") / "runs" / fz2.json()["run"] / "run.json").read_text(encoding="utf-8"))
    assert "core_stabilize" not in rj2


def test_stabilize_core_only_one_frame_is_bitwise_the_full_window():
    """Previews (2026-10-22): `only` stabilises one frame of the window (its flows on several cores) – the same result
    for it as the whole window, the other frames come back as they are; with soft probabilities too."""
    cores, truth, grays = _pan()
    win = {f: cores[f] for f in range(3, 8)}
    probs = {f: (cv2.GaussianBlur(cores[f].astype(np.float32), (0, 0), 1.5) * 255).astype(np.uint8) for f in win}
    for pr in (None, probs):
        po_full, po_one = {}, {}
        full, _ = stabilize_core(win, lambda f: grays[f], radius=2, band_px=8, probs=pr, probs_out=po_full)
        one, _ = stabilize_core(win, lambda f: grays[f], radius=2, band_px=8, probs=pr, probs_out=po_one, only=[5])
        assert np.array_equal(one[5], full[5]) and (one[5] != win[5]).any()
        assert all(np.array_equal(one[f], win[f]) for f in win if f != 5)
        if pr is not None:
            assert np.array_equal(po_one[5], po_full[5])
