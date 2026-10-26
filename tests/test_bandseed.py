# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic band proposal (2026-10-22, docs/PLAN_SAPIENS2.md point 1): Sapiens2 classes along the outline -> band
points of the shared object band (engine/bandseed.py), the worker protocol with the fake model
(engine/sapiens2_worker.py, MD_SAPIENS2_FAKE=1) and POST /api/object/{o}/band/propose with a mocked segmentation."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import bandseed, sapiens2, vitmatte as vm

sys.path.insert(0, str(Path(__file__).resolve().parent))
FAKE_ENV = {**os.environ, "MD_SAPIENS2_FAKE": "1"}


def _disc(h=300, w=300, r=100):
    yy, xx = np.mgrid[:h, :w]
    return (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r * r


def _halves(core, top=4, bottom=3):
    """Class map: upper half `top`, lower half `bottom` (inside the object), background 0."""
    cls = np.zeros(core.shape, np.uint8)
    h = core.shape[0]
    cls[: h // 2][core[: h // 2]] = top
    cls[h // 2:][core[h // 2:]] = bottom
    return cls


def test_hair_wide_skin_thin_points_ramp_across_the_class_boundary():
    core = _disc()
    pts, st = bandseed.propose(core, _halves(core), 7)
    assert st["edge_share"]["hair"] == pytest.approx(0.5, abs=0.05) and st["edge_share"]["skin"] == pytest.approx(0.5, abs=0.05)
    assert len(pts) == 4                                            # two runs, two ends each
    for p in pts:
        k = p["keys"]["7"]
        assert set(p["keys"]) == {"7"}
        assert (k["out"], k["in"]) in ((48.0, 8.0), (4.0, 2.0))
        hair = k["y"] < 150
        assert (k["out"] == 48.0) == hair                           # hair above, skin below
        # hair keeps its width up to the hair / skin boundary (y 150); the skin point is set back by the ramp
        # 2 x |48 - 4| = 88 px along the outline (~77 px in y on this disc), so the width ramps smoothly on the skin
        # side instead of a step between two points 6 px apart (user report 2026-10-23)
        if hair:
            assert abs(k["y"] - 150) < 12
        else:
            assert 60 < k["y"] - 150 < 95
    assert len({p["id"] for p in pts}) == 4
    xy = np.array([[p["keys"]["7"]["x"], p["keys"]["7"]["y"]] for p in pts])
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) + np.eye(len(xy)) * 1e9
    assert d.min() > 2 * bandseed.RAMP_HALF[0]                      # no two points next to each other


def test_one_group_contour_one_point_and_short_run_one_point():
    core = _disc()
    cls = np.where(core, 4, 0).astype(np.uint8)                    # all hair: one point (was two, 6 px apart)
    pts, _ = bandseed.propose(core, cls, 2)
    assert len(pts) == 1 and pts[0]["keys"]["2"]["out"] == 48.0
    cls = _halves(core)
    ys, xs = np.where(core)
    cls[ys[(np.abs(ys - 150) < 16) & (xs > 240)], xs[(np.abs(ys - 150) < 16) & (xs > 240)]] = 1   # ~32 px of clothing
    pts, _ = bandseed.propose(core, cls, 2)
    assert sum(p["keys"]["2"]["out"] == 12.0 for p in pts) == 1   # too short for two points apart: one in the middle


def test_auto_width_is_a_floor():
    core = _disc()
    auto = np.full(core.shape, 20.0, np.float32)
    pts, _ = bandseed.propose(core, _halves(core), 3, auto_w=auto)
    outs = sorted({p["keys"]["3"]["out"] for p in pts})
    ins = sorted({p["keys"]["3"]["in"] for p in pts})
    assert outs == [20.0, 48.0] and ins == [20.0]                   # skin 4 -> 20, hair 48 stays; in = max(rule, auto)


def test_short_runs_merged_and_runs_capped():
    core = _disc()
    cls = _halves(core)
    ys, xs = np.where(core)
    speck = (np.abs(ys - 150) < 3) & (xs > 290 - 100)               # a 5 px stripe of clothing on the right edge
    cls[ys[speck], xs[speck]] = 1
    pts, _ = bandseed.propose(core, cls, 1)
    assert {p["keys"]["1"]["out"] for p in pts} == {48.0, 4.0}      # the speck is below MIN_RUN: no clothing point
    # many alternating runs: at most MAX_RUNS runs (2 points each) per contour
    ang = np.arctan2(*(np.mgrid[:300, :300] - 150))
    cls2 = np.where(core, np.where((np.floor((ang + np.pi) / (2 * np.pi) * 40) % 2) == 0, 4, 1), 0).astype(np.uint8)
    pts2, _ = bandseed.propose(core, cls2, 1)
    assert len(pts2) <= 2 * bandseed.MAX_RUNS


def test_no_person_and_background_only():
    core = _disc()
    cls = np.zeros(core.shape, np.uint8)
    assert bandseed.person_share(core, cls) == 0.0
    pts, st = bandseed.propose(core, cls, 1)
    assert pts == [] and st["edge_share"]["none"] == 1.0           # background on the whole outline: constant band
    assert bandseed.person_share(core, _halves(core)) > 0.99


def test_person_crop_aspect_and_paste_back():
    core = np.zeros((200, 400), bool)
    core[50:150, 100:300] = True                                    # wide object -> crop grown in height to 3:4 (W:H)
    x, y, w, h = sapiens2.person_crop(core, 200, 400)
    assert abs(w / h - 0.75) < 0.01 and w >= 230
    class Fake:
        def classes(self, img):
            return np.full(img.shape[:2], 4, np.uint8)
    cls = sapiens2.segment_object(Fake(), np.zeros((200, 400, 3), np.float32), core)
    assert cls.shape == (200, 400) and cls[100, 200] == 4 and cls[100, 0] == 255     # outside the crop = 255


def test_worker_protocol_with_fake_model():
    vm._CHECK_CACHE.clear()
    r = sapiens2.ComfySapiens2Runner(sapiens2.DEFAULT_MODEL, python=sys.executable, env=FAKE_ENV, timeout_s=60)
    assert r.available() == (True, "")
    img = np.zeros((40, 30, 3), np.float32)
    img[:10] = 0.1
    img[10:20] = 0.4
    img[20:] = 0.9
    cls = r.classes(img)
    assert cls.dtype == np.uint8 and cls.shape == (40, 30)
    assert (cls[:10] == 4).all() and (cls[10:20] == 3).all() and (cls[20:] == 1).all()
    r.close()
    vm._CHECK_CACHE.clear()


def test_propose_endpoint_writes_the_shared_band(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)

    class Seg:
        def classes(self, img):                                      # upper half of the crop = hair, lower = skin
            c = np.full(img.shape[:2], 3, np.uint8)
            c[: img.shape[0] // 2] = 4
            return c
    monkeypatch.setattr(sapiens2, "get_runner", lambda *a, **k: Seg())
    assert api.put("/api/object/obj/videomama", json={"reach_px": 12, "inset_px": 4}).status_code == 200
    r = api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["points"] >= 2 and d["keys"] == [start] and d["replaced"] == 0
    band = pipe.store.get("obj")["band"]
    assert len(band["band_points"]) == d["points"] and band["band_ref_out"] == 12 and band["band_ref_in"] == 4
    outs = {p["keys"][str(start)]["out"] for p in band["band_points"]}
    assert 48.0 in outs and min(outs) < 48.0
    st = api.get(f"/api/object/obj/stage_band/videomama/{start}").json()
    assert len(st["points"]) == d["points"]                          # the stage reads it (shared)
    # the band has points now: 409 asks, replace=True replaces
    r2 = api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama"})
    assert r2.status_code == 409 and "zastąpić" in r2.json()["detail"]
    r3 = api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama", "replace": True})
    assert r3.status_code == 200 and r3.json()["replaced"] == d["points"]

    class NoPerson:
        def classes(self, img):
            return np.zeros(img.shape[:2], np.uint8)
    monkeypatch.setattr(sapiens2, "get_runner", lambda *a, **k: NoPerson())
    r4 = api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama", "replace": True})
    assert r4.status_code == 422


def test_propose_with_inset_slider_at_zero_has_no_inset_and_px_whiskers(tmp_path, monkeypatch):
    """User decisions 2026-10-23: „Wcięcie w rdzeń” 0 means 0 – the proposed points cut 10–14 px into the core; with the
    slider at 0 a blue whisker dragged on one point is plain px there (scale 1, as hand-set points made at 0)."""
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)

    class Seg:
        def classes(self, img):
            c = np.full(img.shape[:2], 3, np.uint8)
            c[: img.shape[0] // 2] = 4
            return c
    monkeypatch.setattr(sapiens2, "get_runner", lambda *a, **k: Seg())
    assert api.put("/api/object/obj/videomama", json={"reach_px": 12, "inset_px": 0}).status_code == 200
    assert api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama"}).status_code == 200
    band = pipe.store.get("obj")["band"]
    assert all(p["keys"][str(start)]["in"] == 0 for p in band["band_points"])
    assert max(p["keys"][str(start)]["out"] for p in band["band_points"]) == 48.0     # outer widths as proposed
    st = api.get(f"/api/object/obj/stage_band/videomama/{start}").json()
    assert st["scale"]["in"] == 1 and all(p["in"] == 0 for p in st["points"])
    pid = st["points"][0]["id"]
    r = api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"id": pid, "in": 6})
    assert r.status_code == 200, r.text
    st = api.get(f"/api/object/obj/stage_band/videomama/{start}").json()
    by = {p["id"]: p["in"] for p in st["points"]}
    assert by[pid] == 6 and all(v == 0 for k, v in by.items() if k != pid)       # one point cuts in, the rest not


def test_proposals_never_cross_the_core_edge_2026_10_26():
    """User decision 2026-10-26 (test_e3 + sh060: inset 2 / inset feather 8 put the feather ramp ~6 px OUTSIDE the core –
    the hard core step showed and the model alpha just outside it was scaled down, 1.0 -> 0.3): the sliders stay the
    artist's, but what MatteDesk sets by itself must not cross – the band defaults (inset feather <= inset) and the
    Sapiens2 proposal (every point's inset >= the current inset feather; the inset slider at 0 still means 0)."""
    from engine.bandsys import BAND_DEFAULTS
    assert BAND_DEFAULTS["inset_feather_px"] <= BAND_DEFAULTS["in_px"]
    core = _disc()
    pts, _ = bandseed.propose(core, _halves(core), 7, min_in=11.0)
    ins = [p["keys"]["7"]["in"] for p in pts]
    assert ins and min(ins) >= 11.0
    pts0, _ = bandseed.propose(core, _halves(core), 7, inset=False, min_in=11.0)
    assert all(p["keys"]["7"]["in"] == 0 for p in pts0)


def test_propose_endpoint_takes_the_inset_feather_as_the_inset_floor(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)

    class Seg:
        def classes(self, img):                                      # upper half hair (in 8), lower skin (in 2)
            c = np.full(img.shape[:2], 3, np.uint8)
            c[: img.shape[0] // 2] = 4
            return c
    monkeypatch.setattr(sapiens2, "get_runner", lambda *a, **k: Seg())
    assert api.put("/api/object/obj/videomama", json={"reach_px": 12, "inset_px": 4, "feather_px": 11}).status_code == 200
    assert api.post("/api/object/obj/band/propose", json={"frame": start, "stage": "videomama"}).status_code == 200
    band = pipe.store.get("obj")["band"]
    assert min(p["keys"][str(start)]["in"] for p in band["band_points"]) >= 11.0
    assert band["inset_feather_px"] == 11                            # the artist's slider untouched
