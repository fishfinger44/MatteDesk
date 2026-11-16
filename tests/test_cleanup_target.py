# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup C1 „Co usuwamy” (user decisions 2026-11-10): ONE thing is cleaned – a whole group or one object outside any
group, picked in the shared Objects palette; a member means its group. The mask: a group = its composition (members'
„Do grupy z:”), an object without a group = its own choice; no mask = cannot be picked (the reason is given). Versions
can be shown as the plate „cleaned” and deleted from disk for good (user decision 2026-11-13)."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine import groups
from engine.cleanup_masks import available, resolve
from engine.export import CORE_RUN
from engine.objects import ObjectStore
from test_group_compose import H, W, _masks, _object

import api.cleanup as AC


@pytest.fixture
def store(tmp_path):
    s = ObjectStore(tmp_path / "shot" / "matte_desk")
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(s, n, ms[n], i)
    s.create("empty", ["empty"], depth_order=3)            # nothing computed
    groups.create_group(s, "body", ["head", "torso"])
    return s


def _objs(s):
    return s.list(groups=True)


def test_one_target_member_means_its_group(store):
    objs = _objs(store)
    assert AC.cleanup_target(objs, ["head"]) == ["body"]
    assert AC.cleanup_target(objs, ["legs", "body"]) == ["body"]          # the last one wins, one thing only
    assert AC.cleanup_target(objs, ["body", "legs"]) == ["legs"]
    assert AC.cleanup_target(objs, ["nope"]) == [] and AC.cleanup_target(objs, []) == []


def test_mask_of_group_member_and_object(store):
    by = {o["name"]: o for o in _objs(store)}
    g = resolve(store, by["body"])
    assert g["run"] == CORE_RUN and g["by"] == "group"                     # the composition of the group entity
    m = resolve(store, by["head"])
    assert m["by"] == "member" and m["source"] == "core"                   # its „Do grupy z:” (default Core)
    groups.set_member_source(store, "body", "head", {"source": "version", "run": "v001"})
    m = resolve(store, store.get("head"))
    assert m["source"] == "version" and m["run"] == "v001"                 # follows the palette's „Do grupy z:”
    own = resolve(store, by["legs"], {"legs": "v001"})
    assert own["run"] == "v001" and own["by"] == "own"
    assert resolve(store, by["legs"], {"legs": "core"})["run"] == CORE_RUN
    none = resolve(store, by["empty"])
    assert none["run"] is None and "etap 1" in none["why"]                 # no mask: cannot be cleaned
    assert {a["source"] for a in available(store, "legs")} >= {"core", "v001"}
    assert available(store, "empty") == []


@pytest.fixture
def api(store, tmp_path):
    work = Path(store.root)
    STATE = {"work_dir": str(work), "manifest": {"plates": {"beauty": str(tmp_path / "b_####.png")}}}
    added = {}

    class Proj:
        def set_plate(self, n, p):
            added[n] = p

        def remove_plate(self, n):
            added.pop(n, None)
    STATE["shot_project"] = Proj()

    def refresh():
        STATE["manifest"]["plates"] = {"beauty": STATE["manifest"]["plates"]["beauty"], **added}
        return STATE["manifest"]
    router, _ = AC.make(STATE=STATE, objects=lambda: store, plate_size=lambda: (W, H), work_frames=lambda: [1],
                        proxy_scale=lambda: 1.0, comfy_client=lambda: None, long_job=lambda t: (lambda f: f),
                        refresh_manifest=refresh)
    app = FastAPI()
    app.include_router(router)
    inp = work.parent / "inpaint"
    for v in ("v001", "v002"):
        (inp / v / "comp").mkdir(parents=True)
        (inp / v / "run.json").write_text(json.dumps({"frames": [1, 1], "ext": "png", "settings": {"remove": ["body"]}}))
    added["cleaned"] = str(inp / "v002" / "comp" / "c_####.png").replace("\\", "/")
    refresh()
    return TestClient(app), work, inp, STATE


def test_api_pick_shows_and_deletes_versions(api):
    c, work, inp, STATE = api
    r = c.put("/api/cleanup", json={"remove": ["torso"]}).json()
    assert r["settings"]["remove"] == ["body"]                              # member -> its group
    d = c.get("/api/cleanup").json()
    by = {o["name"]: o for o in d["objects"]}
    assert by["body"]["mask"]["run"] == "core" and by["empty"]["mask"]["run"] is None
    assert by["legs"]["available"] and by["head"]["available"] == []         # own pick only outside a group
    assert [v["version"] for v in d["versions"] if v["shown"]] == ["v002"]
    assert c.post("/api/cleanup/versions/v001/show").status_code == 200
    assert "/v001/comp/" in STATE["manifest"]["plates"]["cleaned"]
    r = c.delete("/api/cleanup/versions/v001").json()                      # shown one deleted -> the newest left
    assert not (inp / "v001").exists() and r["freed_bytes"] > 0 and "trash" not in r
    assert not (work / "trash").exists()                                   # nothing went to a trash folder
    assert "/v002/comp/" in STATE["manifest"]["plates"]["cleaned"]
    c.delete("/api/cleanup/versions/v002")
    assert "cleaned" not in STATE["manifest"]["plates"]                    # nothing left: plate removed
    assert c.post("/api/cleanup/versions/v009/show").status_code == 404
    assert c.post("/api/cleanup/versions/..%2Fx/show").status_code in (400, 404)


def test_ui_picks_in_the_palette_one_thing():
    root = Path(__file__).resolve().parent.parent
    obj = (root / "static" / "objects.js").read_text(encoding="utf-8")
    assert "new CustomEvent('md-object-pick'" in obj and "function cleanMark(o)" in obj
    # the palette's highlight follows the cleaned thing in Cleanup (user 2026-11-10), with a badge
    assert "const cur = cleanTgt === undefined ? o.name === S.obj : false;" in obj and "${cuOn}" in obj
    assert ".other-layers .ol-row.cu-on" in (root / "static" / "index.css").read_text(encoding="utf-8")
    sel = obj[obj.index("async function selectObject(name) {"):]
    sel = sel[:sel.index("await loadObject();")]
    # Cleanup: the pick goes out, then nothing of the cutting stages is loaded
    assert sel.index("md-object-pick") < sel.index("if (document.body.classList.contains('mode-clean')) return;")
    cu = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "window.addEventListener('md-object-pick'" in cu and "data-cu-rm" not in cu        # no checkbox list
    assert "/api/cleanup/versions/" in cu and "data-cu-del" in cu and "data-cu-show" in cu


def test_api_pick_shows_the_things_newest_version_or_the_plate(api):
    """User 2026-11-15: another thing picked = the plate „cleaned” shows ITS newest version; none = the plate goes
    (the viewer shows the original plate)."""
    c, work, inp, STATE = api
    c.post("/api/cleanup/versions/v001/show")
    r = c.put("/api/cleanup", json={"remove": ["legs"]}).json()
    assert r["version"] is None and "cleaned" not in r["plates"] and "cleaned" not in STATE["manifest"]["plates"]
    r = c.put("/api/cleanup", json={"remove": ["torso"]}).json()             # member -> its group „body”
    assert r["version"] == "v002" and "/v002/comp/" in r["plates"]["cleaned"]
    r = c.put("/api/cleanup", json={"prompt": "x"}).json()                   # no pick: the plate list untouched
    assert "plates" not in r and "/v002/comp/" in STATE["manifest"]["plates"]["cleaned"]


def test_show_version_keeps_the_composite_band_2026_11_15(api):
    """Clicking a version brings back its model / description / anchors, NOT the band (user report: band keys deleted in
    v031 came back after v030 -> v031)."""
    c, work, inp, STATE = api
    c.put("/api/cleanup", json={"remove": ["body"]})
    rj = json.loads((inp / "v001" / "run.json").read_text())
    rj["settings"].update({"prompt": "old prompt", "grow_px": 33, "soft_px": 7, "band_keys": {"5": {"grow_px": 1, "soft_px": 1}},
                           "holdout_off": ["x"]})
    (inp / "v001" / "run.json").write_text(json.dumps(rj))
    c.put("/api/cleanup", json={"grow_px": -10, "soft_px": 20, "prompt": "new"})
    s = c.post("/api/cleanup/versions/v001/show").json()["settings"]
    assert s["prompt"] == "old prompt"                                     # the version's own
    assert s["grow_px"] == -10 and s["soft_px"] == 20 and s["band_keys"] == {} and s["holdout_off"] == []
