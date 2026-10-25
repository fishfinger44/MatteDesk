# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ONE band settings model (user request 2026-10-24, „jeden wzór, reszta instancje”): engine/objband.BandSettings +
band_for is THE resolver of the band every owner works with – stages 3 MatAnyone2 / 4 VideoMaMa / 7 LTX-2.5 and the
whole-outline zones of stage 5 are instances of it. The contract:
  * an object on the shared band: band_for of every key gives equal sliders / points / reference,
  * a detached key gives its own,
  * a slider saved through ANY endpoint lands in the one place (the object band; a detached owner: its own fields),
  * pipeline.band_points_fn of every key gives the same positions / widths,
  * the stage settings functions embed the same BandSettings as `s["band"]` (their legacy keys = its thin mapping)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import objband  # noqa: E402
from engine.bandsys import BAND_DEFAULTS  # noqa: E402
from engine.objband import BandSettings, band_for, canon_sliders, legacy_fields, scales  # noqa: E402

ZONE = {"mode": "add", "criterion": "matte", "polygon": [], "matting": "vitmatte", "preset": "hair",
        "params": {"region": "outline", "temporal": "none"}}


def _proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)       # no CoTracker3: nearest key
    return _api_project(tmp_path, monkeypatch)


def _zone(pipe, zid, name="obj"):
    from engine.zones import load_zones
    return next(z for z in load_zones(pipe.store.dir(name)) if z.zone_id == zid)


def _keys(zid):
    return ("matanyone", "videomama", "ltx", f"zone:{zid}")


def _band_of(pipe, key, zid, name="obj"):
    obj = pipe.store.get(name)
    return band_for(obj, key, zone=_zone(pipe, zid) if key.startswith("zone:") else None)


def _stage_settings_bands(obj):
    from engine.edge import edge_settings
    from engine.ltx_alpha import object_settings as ltx_s
    from engine.videomama import object_settings as vmm_s
    return {"matanyone": edge_settings(obj)["band"], "videomama": vmm_s(obj)["band"], "ltx": ltx_s(obj)["band"]}


# ----------------------------------------------------------------------------- the model alone
def test_band_settings_is_a_dict_with_attributes_and_one_scale_arithmetic():
    b = BandSettings(20, 4, 3, 5, band_points=[{"id": "p1", "keys": {"3": {"x": 1, "y": 2, "out": 10, "in": 2}}}],
                     band_ref_out=10, band_ref_in=2, owner_id="band", shared=True, key="videomama")
    assert b.sliders == (20.0, 4.0, 3.0, 5.0) and b["out_px"] == 20.0 and b.scales == (2.0, 2.0) == scales(b)
    assert json.loads(json.dumps(b))["band_ref_out"] == 10                      # JSON (run.json embeds it)
    assert scales(BandSettings(20, 4, 0, 0)) == (1.0, 1.0)                       # no reference = 1
    b.active = False
    assert b["active"] is False
    with pytest.raises(AttributeError):
        _ = b.nope


def test_legacy_fields_are_the_thin_mapping_of_every_owner():
    b = BandSettings(20.4, 4.6, 3, 5, band_ref_out=10, band_ref_in=2, owner_id="stage_ltx", shared=False)
    assert legacy_fields("matanyone", b) == {"band_out_px": 20.4, "band_in_px": 4.6, "band_in_feather_px": 3.0,
                                             "band_out_feather_px": 5.0, "band_points": [], "band_ref_out": 10,
                                             "band_ref_in": 2, "band_owner": "stage_ltx"}
    v = legacy_fields("videomama", b)
    assert (v["reach_px"], v["inset_px"], v["feather_px"], v["outer_feather_px"]) == (20.4, 4.6, 3.0, 5.0)
    lx = legacy_fields("ltx", b)
    assert (lx["region_px"], lx["core_erode_px"], lx["feather_px"], lx["region_soft_px"]) == (20.4, 5, 3.0, 5.0)   # int inset
    z = legacy_fields("zone:z_01", b)
    assert (z["outline_out_px"], z["fg_erode_px"], z["force_core_erode_px"], z["inset_feather_px"], z["outer_feather_px"]) \
        == (20.4, 5, 5, 3.0, 5.0) and "band_owner" not in z
    # canonical and legacy names both read, the zone's inset = max(fg, force) unless outline_in_px is explicit
    assert canon_sliders("ltx", {"region_px": 1, "in_px": 2, "feather_px": None}) == {"out_px": 1.0, "in_px": 2.0,
                                                                                      "inset_feather_px": None, "outer_feather_px": None}
    assert canon_sliders("zone:z", {"fg_erode_px": 2, "force_core_erode_px": 5})["in_px"] == 5.0
    assert canon_sliders("zone:z", {"fg_erode_px": 2, "outline_in_px": 7})["in_px"] == 7.0


def test_band_for_defaults_own_fields_and_overrides():
    assert band_for({}, "videomama").sliders == tuple(BAND_DEFAULTS[k] for k in objband.SLIDERS)
    assert band_for({}, "videomama").shared is False and band_for({}, "videomama").owner_id == "stage_videomama"
    o = {"videomama": {"reach_px": 12}, "ltx": {"core_erode_px": 3}, "band_out_feather_px": 1.5}
    assert band_for(o, "videomama").out_px == 12.0 and band_for(o, "ltx").in_px == 3.0 and band_for(o, "matanyone").outer_feather_px == 1.5
    assert band_for(o, "videomama", overrides={"out_px": 50, "inset_px": 9}).sliders[:2] == (50.0, 9.0)   # both name sets
    assert band_for(o, "videomama", own={"reach_px": 77}).out_px == 77.0                                 # the stage's validated dict
    zone = {"zone_id": "z_01", "params": {"outline_out_px": 9, "inset_feather_px": 3, "band_points": [
        {"id": "a", "keys": {"0": {"x": 1, "y": 1, "out": 9, "in": 2}}}], "band_ref_out": 9, "band_ref_in": 2}}
    zb = band_for({}, "zone:z_01", zone=zone)
    assert zb.sliders == (9.0, 2.0, 3.0, 3.0) and zb.owner_id == "z_01" and len(zb.band_points) == 1 and zb.scales == (1.0, 1.0)


# ----------------------------------------------------------------------------- the contract on a project
def test_shared_band_every_key_equal_detached_key_its_own(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]
    assert api.put("/api/object/obj/videomama", json={"reach_px": 24, "inset_px": 3, "feather_px": 2, "outer_feather_px": 5}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    r = api.put(f"/api/object/obj/stage_band/ltx/{start}", json={"x": float(xs.max()), "y": float(np.median(ys[xs == xs.max()])), "out": 30})
    assert r.status_code == 200, r.text
    bands = {k: _band_of(pipe, k, zid) for k in _keys(zid)}
    ref = bands["videomama"]
    for k, b in bands.items():
        assert b.shared and b.owner_id == objband.SHARED_OWNER and b.key == k, k
        assert b.sliders == ref.sliders == (24.0, 3.0, 2.0, 5.0), k
        assert b.band_points == ref.band_points and (b.band_ref_out, b.band_ref_in) == (24.0, 3.0), k
        assert b.scales == (1.0, 1.0)
    # the stage settings functions embed the very same BandSettings (their legacy keys are its mapping)
    for k, b in _stage_settings_bands(pipe.store.get("obj")).items():
        assert b == bands[k], k
    s = pipe.store.get("obj")
    from engine.ltx_alpha import object_settings as ltx_s
    from engine.videomama import object_settings as vmm_s
    assert vmm_s(s)["reach_px"] == 24.0 and ltx_s(s)["core_erode_px"] == 3 and ltx_s(s)["band_owner"] == "band"
    # a zone's effective params (load_zones) are the mapping of the same band
    z = _zone(pipe, zid)
    assert z.params["outline_out_px"] == 24.0 and z.params["fg_erode_px"] == 3 and z.params["force_core_erode_px"] == 3
    assert z.params["inset_feather_px"] == 2.0 and z.params["outer_feather_px"] == 5.0 and len(z.params["band_points"]) == 1
    # detach stage 7: its own copy, the others keep sharing; a shared slider moves on without it
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "own"}).status_code == 200
    assert api.put("/api/object/obj/settings", json={"band_out_px": 48}).status_code == 200
    own = _band_of(pipe, "ltx", zid)
    assert not own.shared and own.owner_id == "stage_ltx" and own.sliders == (24.0, 3.0, 2.0, 5.0) and len(own.band_points) == 1
    assert own.scales == (1.0, 1.0)
    for k in ("matanyone", "videomama", f"zone:{zid}"):
        b = _band_of(pipe, k, zid)
        assert b.shared and b.out_px == 48.0 and b.scales == (2.0, 1.0), k
    # back: the shared band again
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "shared"}).status_code == 200
    assert {**_band_of(pipe, "ltx", zid), "key": ""} == {**_band_of(pipe, "videomama", zid), "key": ""}


def test_a_slider_saved_through_any_endpoint_lands_in_the_one_place(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]
    store = pipe.store

    def band():
        return {k: store.get("obj")["band"].get(k) for k in objband.SLIDERS}
    assert api.put("/api/object/obj/settings", json={"band_out_px": 11}).status_code == 200            # stage 3
    assert api.put("/api/object/obj/videomama", json={"inset_px": 5}).status_code == 200               # stage 4
    assert api.put("/api/object/obj/ltx", json={"feather_px": 1.5}).status_code == 200                 # stage 7
    z = _zone(pipe, zid)
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**z.params, "outer_feather_px": 9}}).status_code == 200   # the zone card
    assert band() == {"out_px": 11.0, "in_px": 5.0, "inset_feather_px": 1.5, "outer_feather_px": 9.0}
    o = store.get("obj")
    assert o.get("band_out_px") is None and (o.get("videomama") or {}).get("inset_px") is None and (o.get("ltx") or {}).get("feather_px") is None
    raw = json.loads((store.dir("obj") / "zones" / f"{zid}.json").read_text(encoding="utf-8"))["params"]
    assert not any(k in raw for k in objband.ZONE_SLIDER_KEYS)                   # the zone file keeps no slider of its own
    # the zone saved whole (save_zone_ep, the params as the card shows them now) with sliders: the same place
    z = _zone(pipe, zid)
    body = {**ZONE, "frame": start, "params": {**z.params, "outline_out_px": 33, "fg_erode_px": 7, "force_core_erode_px": 7}}
    assert api.post(f"/api/object/obj/zones/{zid}", json=body).status_code == 200
    assert band()["out_px"] == 33.0 and band()["in_px"] == 7.0
    for k in _keys(zid):
        assert _band_of(pipe, k, zid).sliders == (33.0, 7.0, 1.5, 9.0), k
    # objband.save_sliders itself: the one function behind every endpoint, canonical or legacy names
    assert objband.save_sliders(store, "obj", "ltx", {"region_soft_px": 4}) == "shared" and band()["outer_feather_px"] == 4.0
    assert objband.save_sliders(store, "obj", "ltx", {}) == ""
    # a detached owner: its own fields, the band untouched
    assert api.put("/api/object/obj/stage_band/videomama/mode", json={"mode": "own"}).status_code == 200
    assert api.put("/api/object/obj/videomama", json={"reach_px": 60}).status_code == 200
    assert store.get("obj")["videomama"]["reach_px"] == 60.0 and band()["out_px"] == 33.0
    assert objband.save_sliders(store, "obj", "videomama", {"out_px": 61}) == "own" and store.get("obj")["videomama"]["reach_px"] == 61.0
    assert _band_of(pipe, "videomama", zid).out_px == 61.0 and _band_of(pipe, "ltx", zid).out_px == 33.0
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "own"}).status_code == 200
    assert api.put("/api/object/obj/ltx", json={"core_erode_px": 2}).status_code == 200
    assert store.get("obj")["ltx"]["core_erode_px"] == 2 and band()["in_px"] == 7.0
    assert api.put("/api/object/obj/stage_band/matanyone/mode", json={"mode": "own"}).status_code == 200
    assert api.put("/api/object/obj/settings", json={"band_in_feather_px": 0.5}).status_code == 200
    assert store.get("obj")["band_in_feather_px"] == 0.5 and band()["inset_feather_px"] == 1.5
    assert api.put(f"/api/object/obj/stage_band/zone:{zid}/mode", json={"mode": "own"}).status_code == 200
    z = _zone(pipe, zid)
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**z.params, "outline_out_px": 5}}).status_code == 200
    raw = json.loads((store.dir("obj") / "zones" / f"{zid}.json").read_text(encoding="utf-8"))["params"]
    assert raw["outline_out_px"] == 5 and band()["out_px"] == 33.0 and _band_of(pipe, f"zone:{zid}", zid).out_px == 5.0


def test_detach_copies_the_shared_sliders_over_stale_own_values(tmp_path, monkeypatch):
    """User report 2026-10-24 (sh060): a zone file keeps old own slider values from before it joined the shared band;
    „odłącz” must copy the SHARED values into it (the zone shows them), and „ustaw jako wspólny” then carries them back.
    The same for a stage with stale own fields."""
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]
    store = pipe.store
    assert api.put("/api/object/obj/videomama", json={"reach_px": 70, "inset_px": 8, "feather_px": 27, "outer_feather_px": 12}).status_code == 200
    # stale leftovers in the zone file and in the stage-3 / LTX own fields (as a migrated project may have)
    zf = store.dir("obj") / "zones" / f"{zid}.json"
    raw = json.loads(zf.read_text(encoding="utf-8"))
    raw["params"].update(outline_out_px=61, fg_erode_px=4, force_core_erode_px=4, outer_feather_px=9)
    zf.write_text(json.dumps(raw), encoding="utf-8")
    store.update("obj", band_out_px=33.0, ltx={**(store.get("obj").get("ltx") or {}), "region_soft_px": 1.0})
    for key in (f"zone:{zid}", "matanyone", "ltx"):
        assert api.put(f"/api/object/obj/stage_band/{key}/mode", json={"mode": "own"}).status_code == 200
        assert _band_of(pipe, key, zid).sliders == (70.0, 8.0, 27.0, 12.0), key
    raw = json.loads(zf.read_text(encoding="utf-8"))["params"]
    assert (raw["outline_out_px"], raw["fg_erode_px"], raw["force_core_erode_px"], raw["inset_feather_px"], raw["outer_feather_px"]) == (70.0, 8, 8, 27.0, 12.0)
    assert store.get("obj")["band_out_px"] == 70.0 and store.get("obj")["ltx"]["region_soft_px"] == 12.0
    # the detached zone's slider is its own; „ustaw jako wspólny” carries the zone's values (and its points) back
    z = _zone(pipe, zid)
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**z.params, "outline_out_px": 61}}).status_code == 200
    assert _band_of(pipe, f"zone:{zid}", zid).out_px == 61.0 and store.get("obj")["band"]["out_px"] == 70.0
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    assert api.put(f"/api/object/obj/zones/{zid}/band/{start}", json={"x": float(xs.max()), "y": float(np.median(ys[xs == xs.max()])), "out": 20}).status_code == 200
    assert api.post("/api/object/obj/band/from", json={"source": f"zone:{zid}"}).status_code == 200
    assert store.get("obj")["band"]["out_px"] == 61.0 and len(store.get("obj")["band"]["band_points"]) == 1
    for key in ("videomama", f"zone:{zid}"):
        assert _band_of(pipe, key, zid).sliders == (61.0, 8.0, 27.0, 12.0), key
    assert api.get("/api/object/obj/band").json()["sliders"]["out_px"] == 61.0   # /band = the object band itself


def test_band_points_fn_of_every_key_gives_the_same_positions_and_widths(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    zid = api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]
    assert api.put("/api/object/obj/ltx", json={"combine": "band", "region_px": 20, "core_erode_px": 4}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    x, y = float(xs.max()), float(np.median(ys[xs == xs.max()]))
    assert api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"x": x, "y": y, "out": 30, "in": 2}).status_code == 200
    assert api.put(f"/api/object/obj/stage_band/videomama/{start}", json={"x": x - 40, "y": y - 10, "out": 10, "in": 4}).status_code == 200
    fns = {k: pipe.band_points_fn("obj", _band_of(pipe, k, zid), frames) for k in _keys(zid)}
    ref = fns["videomama"]
    assert ref is not None
    for k, fn in fns.items():
        assert fn is not None, k
        for f in frames:
            assert fn(f) == ref(f), (k, f)
    pts = ref(start)
    assert len(pts) == 2 and {(round(o, 3), round(i, 3)) for _x, _y, o, i in pts} == {(30.0, 2.0), (10.0, 4.0)}
    # a slider doubled in stage 3 doubles the widths everywhere (reference 20 -> 40), one frame at a time too
    assert api.put("/api/object/obj/settings", json={"band_out_px": 40}).status_code == 200
    for k in _keys(zid):
        fn = pipe.band_points_fn("obj", _band_of(pipe, k, zid), None)     # lazy: one frame at a time (a preview)
        assert {round(o, 3) for _x, _y, o, _i in fn(start)} == {60.0, 20.0}, k
    # the stage settings go through the same function (the stage-3 band / the previews): equal to the owner's call
    from engine.edge import edge_settings
    s3 = pipe.band_points_fn("obj", edge_settings(pipe.store.get("obj"))["band"], [start], only=[start])
    assert s3(start) == pipe.band_points_fn("obj", _band_of(pipe, "ltx", zid), [start], only=[start])(start)
    # LTX-2.5 „surowy” = no band: no points
    assert api.put("/api/object/obj/ltx", json={"combine": "raw"}).status_code == 200
    from engine.ltx_alpha import object_settings as ltx_s
    assert ltx_s(pipe.store.get("obj"))["band"].active is False
    assert pipe.band_points_fn("obj", ltx_s(pipe.store.get("obj"))["band"], frames) is None
    assert api.get(f"/api/object/obj/stage_band/ltx/{start}").status_code == 409


def test_same_lines_in_every_stage_through_band_for(tmp_path, monkeypatch):
    """stage_band_lines of stages 3 / 4 / 7 on the shared band: the same BandSettings -> the same lines (also with
    the palette values given, canonical in band_for – the former per-stage branches are gone)."""
    app_module, api, shot, pipe, frames, start = _proj(tmp_path, monkeypatch)
    assert api.put("/api/object/obj/ltx", json={"combine": "band"}).status_code == 200
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    assert api.put(f"/api/object/obj/stage_band/matanyone/{start}",
                   json={"x": float(xs.max()), "y": float(np.median(ys[xs == xs.max()])), "out": 14, "in": 3}).status_code == 200
    q = "reach_px=18&inset_px=5&feather_px=2&outer_feather_px=6"
    lines = {st: api.get(f"/api/object/obj/stage_band/{st}/{start}/lines?{q}").json() for st in ("matanyone", "videomama", "ltx")}
    keys = ("band_contours", "inset_feather_contours", "feather_contours")
    assert lines["matanyone"]["band_contours"]
    for st in ("videomama", "ltx"):
        assert all(lines[st][k] == lines["matanyone"][k] for k in keys), st
    # the band the lines were computed with = band_for with those overrides
    b = band_for(pipe.store.get("obj"), "ltx", overrides={"out_px": 18, "in_px": 5, "inset_feather_px": 2, "outer_feather_px": 6})
    assert b.sliders == (18.0, 5.0, 2.0, 6.0) and b.scales == (18.0 / BAND_DEFAULTS["out_px"], 5.0 / BAND_DEFAULTS["in_px"])


def test_no_band_resolution_of_its_own_in_any_stage_module():
    """Done = the stage modules contain no slider / point resolution: no band_ref arithmetic outside objband.scales and
    no reading of the shared band / the modes outside objband."""
    import re
    root = Path(__file__).resolve().parent.parent
    for rel in ("engine/edge.py", "engine/videomama.py", "engine/ltx_alpha.py", "engine/zones.py", "engine/pipeline.py",
                "engine/stages.py", "app.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert not re.search(r"[\]\)\w]\s*/\s*float\(\s*(ro|ri|band_ref|\w+\.get\(\"band_ref)", src), rel   # x / float(ref)
        assert not re.search(r"[\]\)\w]\s*/\s*(ro|ri|band_ref_out|band_ref_in)\b", src), rel           # x / ref
    for rel in ("engine/edge.py", "engine/videomama.py", "engine/ltx_alpha.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "band_modes" not in src and "shared(" not in src and "own_params(" not in src, rel
        assert "band_for(" in src, rel
