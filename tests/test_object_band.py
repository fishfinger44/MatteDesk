# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The shared band of an object (user decision 2026-10-22, docs/PLAN_WSPOLNY_PAS.md): one set of band points read by
stages 3 / 4 / 5 / 7 with their own slider scale; a detached stage („odłącz → własny pas”) keeps its own copy;
migration of objects saved before."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ZONE = {"mode": "add", "criterion": "matte", "polygon": [], "matting": "vitmatte", "preset": "hair",
        "params": {"region": "outline", "outline_out_px": 6, "temporal": "none"}}


def _proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)       # no CoTracker3: nearest key
    return _api_project(tmp_path, monkeypatch)


def _sigs(store, name="obj"):
    from engine import ltx_alpha, videomama
    from engine.stages import matte_sig, settings_sig
    obj = store.get(name)
    return {"matanyone": settings_sig(obj), "videomama": videomama.settings_sig(videomama.object_settings(obj)),
            "ltx": ltx_alpha.settings_sig(ltx_alpha.object_settings(obj)), "zone": matte_sig(store.dir(name))}


def _setup_all(api, start):
    """Every stage with a band set up: px mode of stage 3, VideoMaMa sliders, LTX-2.5 band mode, a whole-outline zone."""
    assert api.put("/api/object/obj/settings", json={"band_out_px": 10, "band_in_px": 4}).status_code == 200
    assert api.put("/api/object/obj/videomama", json={"reach_px": 20, "inset_px": 4}).status_code == 200
    assert api.put("/api/object/obj/ltx", json={"combine": "band", "region_px": 30, "core_erode_px": 4}).status_code == 200
    return api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]


def test_shared_band_is_read_by_every_stage_with_its_own_scale(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = _setup_all(api, start)
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    r = api.put(f"/api/object/obj/stage_band/videomama/{start}",
                json={"x": float(xs.max()), "y": float(np.median(ys)), "out": 40})
    assert r.status_code == 200, r.text
    ob = api.get("/api/object/obj/band").json()
    assert ob["points"] == 1 and ob["keys"] == [start] and ob["band_ref_out"] == 20
    assert {o["key"]: o["mode"] for o in ob["owners"]} == {"matanyone": "shared", "videomama": "shared", "ltx": "shared",
                                                          f"zone:{zid}": "shared"}
    # the same point everywhere, widths scaled by each stage's slider / the reference (20 px)
    outs = {}
    for st in ("matanyone", "videomama", "ltx", f"zone:{zid}"):
        url = (f"/api/object/obj/zones/{zid}/band/{start}" if st.startswith("zone:")
               else f"/api/object/obj/stage_band/{st}/{start}")
        pts = api.get(url).json()["points"]
        assert [p["id"] for p in pts] == ["p1"], st
        outs[st] = pts[0]["out"]
    assert outs == {"videomama": 40, "matanyone": 20, "ltx": 60, f"zone:{zid}": 12}
    # an edit from another stage (stage 5 zone) changes the shared band – seen by stage 4
    api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"id": "p1", "out": 24})
    assert api.get(f"/api/object/obj/stage_band/videomama/{start}").json()["points"][0]["out"] == 80


def test_shared_band_change_invalidates_every_shared_stage_but_not_a_detached_one(tmp_path, monkeypatch):
    from engine.stages import stage_run
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = _setup_all(api, start)
    store = pipe.store
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    pt = {"x": float(xs.max()), "y": float(np.median(ys)), "out": 30}
    assert api.put(f"/api/object/obj/stage_band/videomama/{start}", json=pt).status_code == 200
    s0 = _sigs(store)
    # a computed VideoMaMa version, current now
    from engine.stages import mask_sig
    from engine.videomama import object_settings, settings_sig
    rd = store.dir("obj") / "runs" / "v900"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"core_fix_sig": mask_sig(store, "obj"),
                                             "vmm_sig": settings_sig(object_settings(store.get("obj")))}), encoding="utf-8")
    store.update("obj", tool={"stage_runs": {"videomama": "v900"}})
    assert stage_run(store, "obj", "videomama")[1]
    api.put(f"/api/object/obj/stage_band/ltx/{start}", json={"id": "p1", "out": 10})   # edited in stage 7
    s1 = _sigs(store)
    assert all(s1[k] != s0[k] for k in s0), (s0, s1)                     # ↻ in 3 / 4 / 5 / 7
    assert not stage_run(store, "obj", "videomama")[1]
    # detach VideoMaMa: its own copy = the shared band (no change in its result), then shared edits pass it by
    assert api.put("/api/object/obj/stage_band/videomama/mode", json={"mode": "own"}).json()["changed"]
    assert _sigs(store)["videomama"] == s1["videomama"]
    assert store.get("obj")["band_modes"] == {"videomama": "own"} and store.get("obj")["videomama"]["band_points"]
    api.put(f"/api/object/obj/stage_band/matanyone/{start}", json={"id": "p1", "out": 5})
    s2 = _sigs(store)
    assert s2["videomama"] == s1["videomama"] and s2["matanyone"] != s1["matanyone"] and s2["zone"] != s1["zone"]
    assert s2["ltx"] != s1["ltx"]
    # the detached stage's own edits stay its own
    api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"id": "p1", "out": 77})
    assert api.get(f"/api/object/obj/stage_band/matanyone/{start}").json()["points"][0]["out"] != 77
    assert _sigs(store)["matanyone"] == s2["matanyone"]
    # „ustaw jako wspólny” from the detached stage: its points become the shared band, it is shared again
    r = api.post("/api/object/obj/band/from", json={"source": "videomama"})
    assert r.status_code == 200, r.text
    assert store.get("obj")["band_modes"] == {} and not store.get("obj")["videomama"]["band_points"]
    assert api.get(f"/api/object/obj/stage_band/matanyone/{start}").json()["points"][0]["out"] == round(77 * 10 / 20, 2)
    # back to shared from own drops the own copy
    api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "own"})
    api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "shared"})
    assert store.get("obj")["band_modes"] == {} and not store.get("obj")["ltx"]["band_points"]
    assert api.post("/api/object/obj/band/from", json={"source": "ltx"}).status_code == 409    # already shared
    # delete the whole shared band = the constant band everywhere; Ctrl+Z (PUT /band/params) brings it back
    before = api.get("/api/object/obj/band/params").json()
    assert api.delete("/api/object/obj/band").status_code == 200
    assert api.get("/api/object/obj/band").json()["points"] == 0
    assert api.put("/api/object/obj/band/params", json=before).status_code == 200
    assert api.get("/api/object/obj/band/params").json() == before
    # a zone card PATCH never writes the shared band (the card sends its params as shown)
    z = next(z for z in api.get("/api/object/obj/zones").json()["zones"] if z["zone_id"] == zid)
    assert z["band_mode"] == "shared" and z["params"]["band_points"] == before["band_points"]
    stale = {**z["params"], "band_points": [], "outline_out_px": 8}
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": stale}).status_code == 200
    assert api.get("/api/object/obj/band/params").json() == before
    assert "band_points" not in json.loads((store.dir("obj") / "zones" / f"{zid}.json").read_text(encoding="utf-8"))["params"]
    assert {o["key"]: o["active"] for o in api.get("/api/object/obj/band").json()["owners"]}["ltx"] is True


def _legacy(store, name, **fields):
    """An object as saved before 2026-10-22 (no `band`): written straight to object.json."""
    p = store.dir(name) / "object.json"
    o = json.loads(p.read_text(encoding="utf-8"))
    o.pop("band", None)
    o.pop("band_modes", None)
    o.update(fields)
    p.write_text(json.dumps(o), encoding="utf-8")


PTS = [{"id": "p1", "keys": {"10": {"x": 5.0, "y": 6.0, "out": 12.0, "in": 3.0}}}]


def test_migration_one_owner_becomes_the_object_band_and_results_stay_current(tmp_path):
    from engine import videomama
    from engine.objects import ObjectStore
    from engine.stages import settings_sig
    store = ObjectStore(tmp_path)
    store.create("a")
    vmm = {"reach_px": 20, "inset_px": 4, "band_points": PTS, "band_ref_out": 20, "band_ref_in": 4}
    _legacy(store, "a", videomama=vmm)
    raw = json.loads((store.dir("a") / "object.json").read_text(encoding="utf-8"))
    sig_before = videomama.settings_sig(videomama.object_settings(raw))      # not migrated = its own points
    mig_before = settings_sig(raw)
    o = store.get("a")                                                      # the first read migrates
    assert o["band"] == {"band_points": PTS, "band_ref_out": 20, "band_ref_in": 4} and o["band_modes"] == {}
    assert o["videomama"]["band_points"] == []
    assert videomama.settings_sig(videomama.object_settings(o)) == sig_before   # the computed result stays current
    assert settings_sig(o) == mig_before            # stage 3 (adaptive mode) does not read the points
    assert store.get("a") == o                      # idempotent


def test_migration_several_different_owners_stay_own(tmp_path):
    from engine.objects import ObjectStore
    from engine.stages import matte_sig
    store = ObjectStore(tmp_path)
    store.create("a")
    store.create("b")
    store.update("b", part_of="a")
    other = [{"id": "p1", "keys": {"10": {"x": 9.0, "y": 6.0, "out": 4.0, "in": 3.0}}}]
    _legacy(store, "a", ltx={"combine": "band", "band_points": PTS, "band_ref_out": 24, "band_ref_in": 4})
    _legacy(store, "b")
    zd = store.dir("b") / "zones"
    zd.mkdir(parents=True, exist_ok=True)
    zone = {"zone_id": "z_01", "criterion": "matte", "enabled": True,
            "params": {"region": "outline", "band_points": other, "band_ref_out": 6, "band_ref_in": 4}}
    (zd / "z_01.json").write_text(json.dumps(zone), encoding="utf-8")
    sig_b = matte_sig(store.dir("b"))
    a = store.get("a")
    b = json.loads((store.dir("b") / "object.json").read_text(encoding="utf-8"))
    assert a["band"]["band_points"] == [] and a["band_modes"] == {"ltx": "own"}
    assert b["band"] == {} and b["band_modes"] == {"zone:z_01": "own"}
    assert a["ltx"]["band_points"] == PTS and json.loads((zd / "z_01.json").read_text(encoding="utf-8")) == zone
    assert matte_sig(store.dir("b")) == sig_b       # nothing changed for the detached zone


def test_migration_of_a_group_with_the_same_points_and_new_objects(tmp_path):
    from engine.objects import ObjectStore
    from engine.stages import matte_sig
    store = ObjectStore(tmp_path)
    store.create("a")
    _legacy(store, "a", edge_band={"band_points": PTS, "band_ref_out": 20, "band_ref_in": 4}, band_out_px=10,
            band_in_px=4)
    zd = store.dir("a") / "zones"
    zone = {"zone_id": "z_02", "criterion": "matte", "enabled": True,
            "params": {"region": "outline", "band_points": PTS, "band_ref_out": 20, "band_ref_in": 4}}
    (zd / "z_02.json").write_text(json.dumps(zone), encoding="utf-8")
    sig = matte_sig(store.dir("a"))
    a = store.get("a")
    assert a["band"]["band_points"] == PTS and a["band_modes"] == {} and a["edge_band"] is None
    assert "band_points" not in json.loads((zd / "z_02.json").read_text(encoding="utf-8"))["params"]
    assert matte_sig(store.dir("a")) == sig         # the shared zone carries the same band: still current
    assert store.create("c")["band"] == {"band_points": [], "band_ref_out": None, "band_ref_in": None}
