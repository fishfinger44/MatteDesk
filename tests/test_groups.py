# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Data model of the group entity (docs/PLAN_GROUP_COMPOSE.md step 1, variant A accepted 2026-10-29):
objects/<group>/object.json with kind "group", members with `group` + `group_source`."""

from pathlib import Path

import pytest

from engine import groups
from engine.objects import ObjectStore


@pytest.fixture
def store(tmp_path):
    s = ObjectStore(tmp_path / "matte_desk")
    for n in ("head", "torso", "legs", "tree"):
        s.create(n, [n])
    return s


def test_create_group_entity_and_members(store):
    g = groups.create_group(store, "body", ["torso", "head"], seam_px=6)
    assert g["kind"] == "group" and g["seam_px"] == 6 and g["members"] == ["torso", "head"]
    assert (store.dir("body") / "object.json").exists() and (store.dir("body") / "runs").is_dir()
    assert groups.members(store, "body") == ["head", "torso"]                       # depth order
    for m in ("head", "torso"):
        o = store.get(m)
        assert o["group"] == "body" and o["group_source"] == {"source": "core", "run": None}   # default = Core
    assert groups.group_depth(store, "body") == 0
    assert [o["name"] for o in store.list()] == ["head", "torso", "legs", "tree"]   # objects only
    assert [o["name"] for o in groups.list_groups(store)] == ["body"]
    assert store.get("body")["tool"]["stage_runs"] == {} and "out_px" in store.get("body")["band"]


def test_member_rules(store):
    groups.create_group(store, "body", ["head"])
    groups.create_group(store, "other", [])
    with pytest.raises(ValueError):
        groups.add_member(store, "other", "head")          # one group per object
    with pytest.raises(ValueError):
        groups.add_member(store, "other", "body")          # no groups in groups
    with pytest.raises(ValueError):
        groups.add_member(store, "body", "body")
    with pytest.raises(FileNotFoundError):
        groups.add_member(store, "body", "nope")
    with pytest.raises(FileExistsError):
        groups.create_group(store, "tree", [])
    groups.add_member(store, "body", "head")                # idempotent
    assert store.get("body")["members"] == ["head"]


def test_member_source_and_seam(store):
    groups.create_group(store, "body", ["head", "torso"])
    assert groups.set_member_source(store, "body", "torso", {"source": "vitmatte", "run": "v009"}) == \
        {"source": "vitmatte", "run": None}                  # run only for "version"
    with pytest.raises(ValueError):
        groups.set_member_source(store, "body", "torso", {"source": "active"})
    with pytest.raises(ValueError):
        groups.set_member_source(store, "body", "torso", {"source": "version", "run": "v001"})   # no such version
    (store.dir("torso") / "runs" / "v001").mkdir()
    (store.dir("torso") / "runs" / "v001" / "run.json").write_text("{}")
    assert groups.set_member_source(store, "body", "torso", {"source": "version", "run": "v001"})["run"] == "v001"
    with pytest.raises(ValueError):
        groups.set_member_source(store, "body", "legs", {"source": "core"})        # not a member
    assert groups.set_seam(store, "body", 999) == 40 and groups.seam(store, "body") == 40
    assert groups.set_seam(store, "body", -3) == 0
    assert store.get("torso")["tool"]["active_run"] == ""   # own export choice untouched


def test_detach_delete_and_rename(store):
    groups.create_group(store, "body", ["head", "torso", "legs"])
    groups.set_member_source(store, "body", "legs", {"source": "matanyone"})
    assert groups.remove_member(store, "body", "legs")["success"]
    assert store.get("legs").get("group") is None and store.get("legs").get("group_source") is None
    groups.add_member(store, "body", "legs")
    assert store.get("legs")["group_source"] == {"source": "core", "run": None}   # re-joined: Core again
    store.remove("torso")                                                        # deleted member leaves the group
    assert store.get("body")["members"] == ["head", "legs"]
    store.rename("head", "glowa")
    assert store.get("body")["members"] == ["glowa", "legs"] and store.get("glowa")["group"] == "body"
    store.rename("body", "cialo")
    assert store.get("glowa")["group"] == "cialo" and groups.members(store, "cialo") == ["glowa", "legs"]
    r = groups.ungroup(store, "cialo")
    assert r["success"] and Path(r["trash"]).exists() and groups.list_groups(store) == []
    assert all(store.get(n).get("group") is None for n in ("glowa", "legs"))
    assert [o["depth_order"] for o in store.list()] == [0, 1, 2]


def test_members_compute_alone(store):
    """Step 2 (PLAN_GROUP_COMPOSE): `part_of` / a group no longer redirects anything – a member's mask signature, stage
    results and band are its own; the group entity's signature covers its members and the seam."""
    from engine.objband import band_for, save_sliders
    from engine.stages import core_fix_sig, group_members, mask_sig, stage_run
    store.update("torso", part_of="head")                                   # legacy part: computes alone too
    assert mask_sig(store, "head") == core_fix_sig(store, "head")
    assert group_members(store, "head") is None
    (store.dir("head") / "runs" / "v001").mkdir()
    (store.dir("head") / "runs" / "v001" / "run.json").write_text('{"group": ["head", "torso"]}')
    store.update("head", tool={"stage_runs": {"matanyone": "v001"}})
    assert stage_run(store, "torso", "matanyone")[0] is None               # never the parent's result
    assert save_sliders(store, "torso", "matanyone", {"out_px": 33}) == "shared"
    assert store.get("torso")["band"]["out_px"] == 33.0 and store.get("head")["band"].get("out_px") != 33.0
    assert band_for(store.get("torso"), "matanyone", root=store.get("torso")).out_px == 33.0
    store.update("torso", part_of=None)
    groups.create_group(store, "body", ["head", "torso"], seam_px=5)
    assert group_members(store, "body") == ["head", "torso"]
    sig = mask_sig(store, "body")
    groups.set_seam(store, "body", 7)
    assert mask_sig(store, "body") != sig
    sig = mask_sig(store, "body")
    groups.add_member(store, "body", "legs")
    assert mask_sig(store, "body") != sig


def test_front_holdout_includes_group_members(tmp_path):
    """A member computes on its own: the other members in front of it are holdouts like any object."""
    import numpy as np
    import cv2
    from engine.pipeline import ObjectPipeline
    s = ObjectStore(tmp_path / "matte_desk")
    s.create("hand", ["hand"], depth_order=0)
    s.create("face", ["face"], depth_order=1)
    (s.dir("hand") / "core").mkdir(exist_ok=True)
    cv2.imwrite(str(s.dir("hand") / "core" / "c_0001.png"), np.full((8, 8), 255, np.uint8))
    groups.create_group(s, "body", ["hand", "face"])
    pipe = ObjectPipeline.__new__(ObjectPipeline)
    pipe.store, pipe.w, pipe.h = s, 8, 8
    _, used = pipe.front_holdout("face", [1], need=False)
    assert used == ["hand"]


def test_group_api_palette(tmp_path, monkeypatch):
    """Step 5 (palette Obiekty): create a group, drop objects in, „Do grupy z:”, seam, export selection, detach with
    the warning when the group has a whole result, ungroup."""
    import json
    from fastapi.testclient import TestClient
    import app as app_module
    work = tmp_path / "matte_desk"
    s = ObjectStore(work)
    for n in ("head", "torso", "tree"):
        s.create(n, [n])
    monkeypatch.setitem(app_module.STATE, "work_dir", str(work))
    api = TestClient(app_module.app)
    r = api.post("/api/groups", json={"name": "body", "members": ["head"]})
    assert r.status_code == 200, r.text
    assert api.post("/api/groups", json={"name": "body"}).status_code == 409
    assert api.post("/api/group/body/members", json={"name": "torso"}).status_code == 200
    assert api.post("/api/group/body/members", json={"name": "body"}).status_code == 400
    d = api.get("/api/objects").json()
    assert [o["name"] for o in d["objects"]] == ["body", "head", "torso", "tree"]       # the group above its members
    assert d["objects"][0]["kind"] == "group" and d["objects"][0]["group_info"]["members"][0]["member"] == "head"
    g = d["groups"][0]
    assert g["name"] == "body" and [m["member"] for m in g["members"]] == ["head", "torso"]
    assert g["members"][0]["source"] == "core" and g["members"][0]["available"][0]["source"] == "core"
    assert all(m["on"] for m in g["members"])
    # the palette checkbox of a member: switched off = out of the composition, still a member (2026-10-29)
    g = api.put("/api/group/body/members/head/on", json={"on": False}).json()["group"]
    assert [(m["member"], m["on"]) for m in g["members"]] == [("head", False), ("torso", True)]
    assert api.put("/api/group/body/members/tree/on", json={"on": False}).status_code == 400   # not a member
    assert api.put("/api/group/body/members/head/on", json={"on": True}).json()["group"]["members"][0]["on"]
    assert api.put("/api/group/body/members/torso/source", json={"source": "matanyone"}).status_code == 200
    assert api.put("/api/group/body/members/torso/source", json={"source": "zzz"}).status_code == 400
    assert api.put("/api/group/body", json={"seam_px": 9}).json()["group"]["seam_px"] == 9
    # a group has no stages 1 + 2 of its own: SAM 3.1 / key / core actions refused on it
    r = api.post("/api/object/body/points/1", json={"points": []})
    assert r.status_code == 409 and "grupa" in r.json()["detail"]
    assert api.post("/api/object/body/core_fix").status_code == 409
    # a whole result: detaching warns (409, nothing changed), ?confirm=1 trashes it
    rd = s.dir("body") / "runs" / "v001"
    rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"stage": "matanyone", "segments_def": []}))
    s.update("body", tool={"stage_runs": {"matanyone": "v001"}})
    r = api.delete("/api/group/body/members/torso")
    assert r.status_code == 409 and r.json()["detail"]["needs_confirm"] and "matanyone" in r.json()["detail"]["whole_stages"]
    assert s.get("torso")["group"] == "body"
    r = api.delete("/api/group/body/members/torso?confirm=true")
    assert r.status_code == 200 and r.json()["trash"] and s.get("torso").get("group") is None
    assert api.delete("/api/group/body").status_code == 200
    assert api.get("/api/objects").json()["groups"] == [] and s.get("head").get("group") is None


def test_composition_cache_follows_every_change_of_the_input(tmp_path):
    """The composition is cached per frame (previews ask again, 2026-10-29): a change of a member's source, on / off or
    the seam is a new entry – never the old composition."""
    import numpy as np
    from engine import groups
    from test_group_compose import SEAM, W, H, _masks, _object
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(store, n, ms[n], i)
    groups.create_group(store, "body", ["head", "torso", "legs"], seam_px=SEAM)
    a = groups.group_alpha_loader(store, "body", W, H)(1)
    assert np.array_equal(groups.group_alpha_loader(store, "body", W, H)(1), a)       # cached: the same
    groups.set_member_on(store, "body", "legs", False)
    b = groups.group_alpha_loader(store, "body", W, H)(1)
    assert b.sum() < a.sum()
    groups.set_member_on(store, "body", "legs", True)
    groups.set_seam(store, "body", 0)
    c = groups.group_alpha_loader(store, "body", W, H)(1)
    assert not np.array_equal(c, a)
