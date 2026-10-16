# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Object stages in order (detect -> crops -> core -> final -> export): locked until the previous one is done;
changing keys after the core makes it stale; the API refuses a locked action with the reason."""

import json
import time

import numpy as np
import pytest

from engine.objects import ObjectStore
from engine.stages import keys_signature, object_stages, require


def _st(store, work):
    return {s["id"]: s["state"] for s in object_stages(store, "obj", work)["stages"]}


def test_stage_progression_and_staleness(tmp_path):
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    store.create("obj", ["puppet"])
    assert _st(store, work) == {"detect": "todo", "crops": "locked", "core": "locked", "cotracker": "locked",
                                "matanyone": "locked", "videomama": "locked", "vitmatte": "locked", "final": "locked", "ltx": "locked",
                                "export": "locked"}
    with pytest.raises(ValueError, match="Wykryj"):
        require(store, "obj", work, "core")
    m = np.zeros((40, 40), bool); m[10:20, 10:20] = True
    store.save_key("obj", 5, m, author="auto")
    assert _st(store, work)["crops"] == "todo" and object_stages(store, "obj", work)["next"] == "crops"
    (store.dir("obj") / "crops.json").write_text("{}")
    assert _st(store, work)["core"] == "todo"
    with pytest.raises(ValueError, match="Rdzeń SAM3"):
        require(store, "obj", work, "final")
    cd = store.dir("obj") / "core"; cd.mkdir()
    (cd / "core.json").write_text(json.dumps({"keys_sig": keys_signature(store, "obj")}))
    st = _st(store, work)
    assert st["core"] == "done" and st["final"] == "todo"
    # 2026-10-15: export right after the core; CoTracker without brush zones = done (optional); MatAnyone todo,
    # ViTMatte waits for MatAnyone
    assert st["export"] == "todo" and st["cotracker"] == "done" and st["matanyone"] == "todo" and st["vitmatte"] == "locked"
    # a new user key after the core -> core stale, final/export locked again
    time.sleep(0.01)
    store.save_key("obj", 9, m, author="user")
    st = _st(store, work)
    assert st["core"] == "stale" and st["final"] == "locked"
    (cd / "core.json").write_text(json.dumps({"keys_sig": keys_signature(store, "obj")}))
    rd = store.dir("obj") / "runs" / "v001"; rd.mkdir(parents=True)
    time.sleep(0.01)
    (rd / "run.json").write_text("{}")
    store.update("obj", tool={"active_run": "v001", "final_valid": True})
    assert _st(store, work)["final"] == "done" and _st(store, work)["export"] == "todo"
    (work / "outputs.json").write_text(json.dumps({"objects": [{"name": "obj", "version": "v001"}]}))
    st = _st(store, work)
    assert st["final"] == "done" and st["export"] == "done" and st["matanyone"] == "todo"
    # settings change (final_valid False, e.g. a front object changed: 'do przeliczenia') -> final stale
    store.update("obj", tool={"final_valid": False, "status": "do przeliczenia"})
    st = _st(store, work)
    assert st["final"] == "stale" and st["export"] != "done"


def test_delete_object_moves_to_trash_and_marks_dependents(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    from engine.outputs import load_outputs, update_outputs
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    for n in ("front", "mid", "back"):
        store.create(n, [n])
    store.update("back", tool={"active_run": "v001", "final_valid": True})
    update_outputs(work, "objects", [{"name": "front", "version": "v001"}, {"name": "back", "version": "v001"}])
    monkeypatch.setitem(app_module.STATE, "work_dir", str(work))
    api = TestClient(app_module.app)
    r = api.delete("/api/object/front")
    assert r.status_code == 200, r.text
    d = r.json()
    assert "trash" in d["trash"] and (tmp_path / "matte_desk" / "trash" / "objects").exists()
    assert not (work / "objects" / "front").exists()
    assert d["marked_for_recompute"] == ["back"]
    assert [(o["name"], o["depth_order"]) for o in store.list()] == [("mid", 0), ("back", 1)]
    assert store.get("back")["tool"]["status"] == "do przeliczenia" and not store.get("back")["tool"]["final_valid"]
    assert [e["name"] for e in load_outputs(work)["objects"]] == ["back"]
    assert api.delete("/api/object/front").status_code == 404
    store.create("new", ["x"])
    assert store.get("new")["depth_order"] == 2   # no collision after renumbering


def test_stale_intervals(tmp_path):
    """Z2: intervals the next core run will recompute = around keys added / removed / changed since the last run."""
    from engine.stages import stale_intervals
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    store.create("obj", ["puppet"])
    m = np.zeros((40, 40), bool); m[10:20, 10:20] = True
    for f in (1, 50, 120):
        store.save_key("obj", f, m, author="user")
    (store.dir("obj") / "crops.json").write_text("{}")
    assert stale_intervals(store, "obj", (1, 193)) == [[1, 193]]                     # no core.json yet
    cd = store.dir("obj") / "core"; cd.mkdir()
    time.sleep(0.01)
    (cd / "core.json").write_text(json.dumps({"keys_sig": keys_signature(store, "obj")}))
    assert stale_intervals(store, "obj", (1, 193)) == []
    assert object_stages(store, "obj", work, (1, 193))["stale_intervals"] == []
    time.sleep(0.01)
    store.save_key("obj", 80, m, author="user")                                     # added between 50 and 120
    assert stale_intervals(store, "obj", (1, 193)) == [[50, 120]]
    store.save_key("obj", 120, ~m, author="user")                                   # changed: 80..193 (after the last key)
    assert stale_intervals(store, "obj", (1, 193)) == [[50, 193]]
    (cd / "core.json").write_text(json.dumps({"keys_sig": keys_signature(store, "obj")}))
    store.delete_key("obj", 50)                                                     # removed: from key 1 to key 80
    assert stale_intervals(store, "obj", (1, 193)) == [[1, 80]]
    assert object_stages(store, "obj", work, (1, 193))["stale_intervals"] == [[1, 80]]
    time.sleep(0.01)
    (store.dir("obj") / "crops.json").write_text("{}")                              # crops newer than the core
    assert stale_intervals(store, "obj", (1, 193)) == [[1, 193]]
