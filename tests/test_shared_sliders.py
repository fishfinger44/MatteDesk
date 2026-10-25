# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The band SLIDERS are a feature of the object (user decision 2026-10-24): one set of out / in / inset feather / outer
feather in the object band, read by every stage on the shared band, moved from any stage; one „odłącz” for the sliders
and the points; the migration detaches every shared owner whose own sliders differ (results unchanged)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import objband  # noqa: E402
from engine.bandsys import BAND_DEFAULTS  # noqa: E402


def _proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return app_module, api, shot, pipe, frames, start


def _obj_json(pipe, name="obj"):
    return json.loads((pipe.store.dir(name) / "object.json").read_text(encoding="utf-8"))


def test_sliders_shared_by_every_stage_and_moved_from_any(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    o = _obj_json(pipe)
    assert "out_px" in o["band"]                                   # migrated on the first read
    from engine.edge import band_feathers, band_px_limits
    from engine.ltx_alpha import object_settings as ltx_s
    from engine.videomama import object_settings as vmm_s
    # unset everywhere = the one defaults
    assert band_px_limits(o) == (BAND_DEFAULTS["out_px"], BAND_DEFAULTS["in_px"])
    assert vmm_s(o)["reach_px"] == BAND_DEFAULTS["out_px"] and ltx_s(o)["region_px"] == BAND_DEFAULTS["out_px"]
    # a slider moved in stage 4 moves the band of stages 3 / 7 (and the stage-5 zone)
    assert api.put("/api/object/obj/videomama", json={"reach_px": 33, "inset_px": 5, "feather_px": 2, "outer_feather_px": 7}).status_code == 200
    o = pipe.store.get("obj")
    assert o["band"]["out_px"] == 33 and o["band"]["in_px"] == 5 and (o.get("videomama") or {}).get("reach_px") is None
    assert band_px_limits(o) == (33.0, 5.0) and band_feathers(o) == (2.0, 7.0)
    assert ltx_s(o)["region_px"] == 33 and ltx_s(o)["core_erode_px"] == 5 and ltx_s(o)["feather_px"] == 2 and ltx_s(o)["region_soft_px"] == 7
    assert vmm_s(o)["inset_px"] == 5 and vmm_s(o)["outer_feather_px"] == 7
    # stage 3's palette and its settings endpoint go through the same band
    assert api.get("/api/object/obj/edge_modes").json()["band_px"] == [33, 5]
    assert api.put("/api/object/obj/settings", json={"band_out_px": 44}).status_code == 200
    assert pipe.store.get("obj")["band"]["out_px"] == 44 and pipe.store.get("obj").get("band_out_px") is None
    assert api.get("/api/object/obj/ltx").json()["settings"]["region_px"] == 44
    # a whole-outline zone reads the band too, and its slider moves the band
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "temporal": "none"}}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    from engine.zones import load_zones, outline_in_px, zone_feather_px, zone_outer_feather_px
    z = next(x for x in load_zones(pipe.store.dir("obj")) if x.zone_id == zid)
    assert z.params["outline_out_px"] == 44 and outline_in_px(z.params) == 5 and zone_feather_px(z) == 2 and zone_outer_feather_px(z) == 7
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**z.params, "outline_out_px": 50, "fg_erode_px": 6, "force_core_erode_px": 6}}).status_code == 200
    o = pipe.store.get("obj")
    assert o["band"]["out_px"] == 50 and o["band"]["in_px"] == 6 and band_px_limits(o) == (50.0, 6.0)
    zf = json.loads((pipe.store.dir("obj") / "zones" / f"{zid}.json").read_text(encoding="utf-8"))
    assert "outline_out_px" not in zf["params"] and "fg_erode_px" not in zf["params"]   # no slider of its own while shared
    # the stage-3 signature follows the shared slider (↻ in every shared stage)
    from engine.stages import settings_sig
    s1 = settings_sig(pipe.store.get("obj"))
    assert api.put("/api/object/obj/ltx", json={"region_soft_px": 3}).status_code == 200
    assert settings_sig(pipe.store.get("obj")) != s1 and band_feathers(pipe.store.get("obj")) == (2.0, 3.0)
    assert api.get("/api/object/obj/band").json()["sliders"] == {"out_px": 50.0, "in_px": 6.0, "inset_feather_px": 2.0, "outer_feather_px": 3.0}


def test_one_detach_for_sliders_and_points(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.put("/api/object/obj/videomama", json={"reach_px": 30, "inset_px": 4}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    r = api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"x": float(xs.max()), "y": float(np.median(ys[xs == xs.max()])), "out": 40})
    assert r.status_code == 200, r.text
    # detach stage 7: it keeps the sliders and the points it shows; the shared band moves on without it
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "own"}).status_code == 200
    o = pipe.store.get("obj")
    assert o["band_modes"]["ltx"] == "own" and o["ltx"]["region_px"] == 30 and o["ltx"]["core_erode_px"] == 4 and len(o["ltx"]["band_points"]) == 1
    assert api.put("/api/object/obj/videomama", json={"reach_px": 60}).status_code == 200
    from engine.ltx_alpha import object_settings as ltx_s
    from engine.videomama import object_settings as vmm_s
    o = pipe.store.get("obj")
    assert ltx_s(o)["region_px"] == 30 and vmm_s(o)["reach_px"] == 60
    from engine.edge import band_px_limits
    assert band_px_limits(o) == (60.0, 4.0)
    # a detached stage's slider stays its own
    assert api.put("/api/object/obj/ltx", json={"region_px": 12}).status_code == 200
    o = pipe.store.get("obj")
    assert ltx_s(o)["region_px"] == 12 and o["band"]["out_px"] == 60
    # back to the shared band: the shared sliders and points again
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "shared"}).status_code == 200
    assert ltx_s(pipe.store.get("obj"))["region_px"] == 60


def test_migration_detaches_the_owners_whose_sliders_differ(tmp_path):
    """Objects from before 2026-10-24: every shared owner keeps the result it had – the band takes the first shared
    owner's sliders (a whole-outline zone first), the others with different sliders are detached with a copy of the
    points; equal ones stay shared; owners already detached stay so."""
    root = tmp_path / "objects"
    od = root / "obj"
    (od / "zones").mkdir(parents=True)
    pts = {"band_points": [{"id": "p1", "keys": {"1": {"x": 10.0, "y": 10.0, "out": 20, "in": 4}}}], "band_ref_out": 52.0, "band_ref_in": 0.0}
    obj = {"name": "obj", "band": pts, "band_modes": {"ltx": "own"},
           "band_out_px": 57, "band_in_px": 3, "band_in_feather_px": 5, "band_out_feather_px": 2,          # stage 3: differs
           "videomama": {"reach_px": 52.0, "inset_px": 0.0, "feather_px": 8.0, "outer_feather_px": 15.0},  # stage 4: = the zone
           "ltx": {"region_px": 35, "core_erode_px": 5, "band_points": [], "band_ref_out": None, "band_ref_in": None},
           "tool": {}}
    (od / "object.json").write_text(json.dumps(obj), encoding="utf-8")
    zone = {"zone_id": "z_01", "mode": "add", "criterion": "matte", "frame": 1, "polygon": [], "matting": "vitmatte",
            "params": {"region": "outline", "outline_out_px": 52, "fg_erode_px": 0, "force_core_erode_px": 0, "outer_feather_px": 15}}
    (od / "zones" / "z_01.json").write_text(json.dumps(zone), encoding="utf-8")
    objband.migrate_sliders(root, "obj")
    o = json.loads((od / "object.json").read_text(encoding="utf-8"))
    assert o["band"]["out_px"] == 52 and o["band"]["in_px"] == 0 and o["band"]["inset_feather_px"] == 8 and o["band"]["outer_feather_px"] == 15
    assert o["band_modes"] == {"ltx": "own", "matanyone": "own"}            # stage 3 differs -> detached; stage 4 = the zone -> shared
    assert o["edge_band"]["band_points"] == pts["band_points"]              # ... with a copy of the shared points
    assert o["band_out_px"] == 57                                           # ... and its own sliders as they were
    assert objband.sliders(o, "matanyone") == (57.0, 3.0, 5.0, 2.0)
    assert objband.sliders(o, "videomama") == (52.0, 0.0, 8.0, 15.0)
    assert objband.sliders(o, "ltx") == (35.0, 5.0, BAND_DEFAULTS["inset_feather_px"], BAND_DEFAULTS["outer_feather_px"])
    z = json.loads((od / "zones" / "z_01.json").read_text(encoding="utf-8"))
    assert "band_points" not in z["params"] and objband.mode(o, "zone:z_01") == "shared"
    objband.migrate_sliders(root, "obj")                                    # idempotent
    assert json.loads((od / "object.json").read_text(encoding="utf-8")) == o


def test_sliders_fall_back_to_the_defaults_without_a_band():
    o = {"name": "x"}   # not migrated at all: own fields, defaults
    assert objband.sliders(o, "videomama") == (BAND_DEFAULTS["out_px"], BAND_DEFAULTS["in_px"], BAND_DEFAULTS["inset_feather_px"], BAND_DEFAULTS["outer_feather_px"])
    o = {"name": "x", "band": {"band_points": [], "band_ref_out": None, "band_ref_in": None}}   # points migrated, sliders not
    assert objband.sliders({**o, "videomama": {"reach_px": 12}}, "videomama")[0] == 12.0
    zone = {"params": {"region": "outline", "inset_feather_px": 3}}
    assert objband.sliders(o, "zone:z_01", zone=zone)[2:] == (3.0, 3.0)   # a zone's outer feather unset = its inset feather


def test_same_sliders_give_the_same_lines_in_every_stage_from_one_cache(tmp_path, monkeypatch):
    """The contract of the one band (user, 2026-10-24): with the same sliders and the shared points, stages 3 / 4 / 7
    (all on the mask 1+2) show the very same band lines – and they come from ONE cache entry (the band is computed
    once per frame, not per stage)."""
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.put("/api/object/obj/ltx", json={"combine": "band"}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    r = api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"x": float(xs.max()), "y": float(np.median(ys[xs == xs.max()])), "out": 30})
    assert r.status_code == 200, r.text
    app_module._LINES_CACHE.clear()
    import time
    api.get(f"/api/object/obj/stage_band/videomama/{start}/lines")          # starts the work-area mask batch
    t0 = time.time()
    while not api.get("/api/object/obj/mask12_status").json()["ready"] and time.time() - t0 < 120:
        time.sleep(0.5)                                                       # lines on the stand-in mask are never cached
    assert api.get("/api/object/obj/mask12_status").json()["ready"]
    app_module._LINES_CACHE.clear()
    q = "reach_px=24&inset_px=5&feather_px=3&outer_feather_px=6"
    lines = {st: api.get(f"/api/object/obj/stage_band/{st}/{start}/lines?{q}").json() for st in ("videomama", "matanyone", "ltx")}
    keys = ("band_contours", "inset_feather_contours", "feather_contours")
    assert all(lines[st][k] == lines["videomama"][k] for st in lines for k in keys)
    assert lines["videomama"]["band_contours"] and lines["videomama"]["inset_feather_contours"]
    assert len(app_module._LINES_CACHE) == 1                      # one entry served the three stages
    # the lines follow the shared sliders without query values too
    assert api.put("/api/object/obj/settings", json={"band_out_px": 24, "band_in_px": 5, "band_in_feather_px": 3, "band_out_feather_px": 6}).status_code == 200
    plain = {st: api.get(f"/api/object/obj/stage_band/{st}/{start}/lines").json() for st in ("videomama", "matanyone", "ltx")}
    assert all(plain[st][k] == lines["videomama"][k] for st in plain for k in keys)
