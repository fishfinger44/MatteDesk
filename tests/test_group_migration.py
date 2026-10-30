# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Migration of the legacy parts (part_of) to a group entity on project open (docs/PLAN_GROUP_COMPOSE.md step 6)."""

import json

import cv2
import numpy as np

from engine import groups
from engine.objects import ObjectStore
from engine.stages import settings_sig, stage_run


def _legacy(tmp_path, with_whole=True):
    work = tmp_path / "matte_desk"
    s = ObjectStore(work)
    for i, n in enumerate(("figure", "suknia", "rogi", "drzewo")):
        s.create(n, [n], depth_order=i)
        (s.dir(n) / "core").mkdir(exist_ok=True)
        cv2.imwrite(str(s.dir(n) / "core" / "c_0001.png"), np.full((8, 8), 255, np.uint8))
    s.update("suknia", part_of="figure")
    s.update("rogi", part_of="figure")
    s.update("figure", part_seam_px=7, shift_px=1.5, band={**s.get("figure")["band"], "out_px": 33.0})
    if with_whole:   # zones before the versions: a brush zone is part of the mask 1+2 signature
        (s.dir("figure") / "zones" / "z_01.json").write_text(json.dumps({"zone_id": "z_01", "criterion": "matte", "mode": "add"}))
        (s.dir("figure") / "zones" / "z_02.json").write_text(json.dumps({"zone_id": "z_02", "criterion": "all", "mode": "add",
                                                                         "params": {"source": "brush"}}))
    # an own (non-group) version of the parent and a whole computed on it (run.json "group"), current
    for r, grp in (("v001", None), ("v002", ["figure", "suknia", "rogi"])):
        rd = s.dir("figure") / "runs" / r
        rd.mkdir(parents=True)
        rj = {"stage": "matanyone", "segments_def": [], "settings_sig": settings_sig(s.get("figure"))}
        if grp:
            rj.update(group=grp, core_fix_sig=groups._legacy_mask_sig(s, "figure", ["suknia", "rogi"]))
        (rd / "run.json").write_text(json.dumps(rj))
    if with_whole:
        s.update("figure", tool={"stage_runs": {"matanyone": "v002", "core_base": "v001"}, "active_run": "v002",
                                "final_valid": True})
        (s.dir("figure") / "final_touch").mkdir()
        (s.dir("figure") / "final_touch" / "final_touch.json").write_text(json.dumps({"base": {"source": "matanyone"}, "frames": {}}))
    return work, s


def test_parts_become_a_group_with_the_whole(tmp_path):
    work, s = _legacy(tmp_path)
    done = groups.migrate_part_of(s)
    assert done == [{"group": "figure_grp", "members": ["figure", "suknia", "rogi"], "runs": ["v002"]}]
    g = s.get("figure_grp")
    assert g["kind"] == "group" and g["seam_px"] == 7 and groups.members(s, "figure_grp") == ["figure", "suknia", "rogi"]
    assert all(s.get(n).get("part_of") is None and s.get(n)["group"] == "figure_grp" for n in ("figure", "suknia", "rogi"))
    assert all(s.get(n)["group_source"] == {"source": "core", "run": None} for n in ("figure", "suknia", "rogi"))
    assert s.get("drzewo").get("group") is None
    # the whole moved with its stage result and stays current (same input, now the group's signature)
    assert (s.dir("figure_grp") / "runs" / "v002" / "run.json").exists() and not (s.dir("figure") / "runs" / "v002").exists()
    assert stage_run(s, "figure_grp", "matanyone")[:2] == ("v002", True)
    assert g["tool"]["active_run"] == "v002" and g["tool"]["final_valid"]
    # the parent keeps its own version and settings; its band / settings are copied to the group
    pt = s.get("figure")["tool"]
    assert pt["stage_runs"] == {"core_base": "v001"} and pt["active_run"] == "" and (s.dir("figure") / "runs" / "v001").exists()
    assert g["shift_px"] == 1.5 and g["band"]["out_px"] == 33.0 and s.get("figure")["band"]["out_px"] == 33.0
    # Final Touch of the whole + the ViTMatte zones go to the group (brush zones stay with the parent)
    assert (s.dir("figure_grp") / "final_touch" / "final_touch.json").exists() and not (s.dir("figure") / "final_touch").exists()
    assert (s.dir("figure_grp") / "zones" / "z_01.json").exists() and not (s.dir("figure_grp") / "zones" / "z_02.json").exists()
    assert (s.dir("figure") / "zones" / "z_01.json").exists()
    assert groups.migrate_part_of(s) == []                                       # idempotent


def test_parts_without_a_whole_and_a_name_clash(tmp_path):
    work, s = _legacy(tmp_path, with_whole=False)
    s.create("figure_grp", ["x"])                                                 # the name is taken
    s.create("sierota", ["x"])
    s.update("sierota", part_of="nie_ma")                                        # a dangling relation
    done = groups.migrate_part_of(s)
    assert [d["group"] for d in done] == ["figure_grp2"] and done[0]["runs"] == ["v002"]   # every whole version moves
    assert "matanyone" not in (s.get("figure")["tool"].get("stage_runs") or {})
    assert s.get("sierota").get("part_of") is None and s.get("sierota").get("group") is None


def test_project_open_migrates(tmp_path, monkeypatch):
    """The migration runs when a project is opened (app._activate_project)."""
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    store = pipe.store
    store.create("hand", ["hand"], depth_order=1)
    store.update("hand", part_of="obj")
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    assert store.get("hand").get("part_of") is None and store.get("hand")["group"] == "obj_grp"
    assert [g["name"] for g in groups.list_groups(store)] == ["obj_grp"]
