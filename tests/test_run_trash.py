# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Delete a version (user request 2026-10-25, stage 5 "Version comparison"): to the project trash, never one in use,
numbers never reused."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _run(od: Path, run: str, **extra) -> None:
    d = od / "runs" / run
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"stage": "core", "created_at": "2026-10-25T10:00:00", **extra}), encoding="utf-8")
    (d / "a_0001.png").write_bytes(b"x" * 1000)


def test_delete_run_to_trash_refs_and_numbering(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    from engine.refine import next_run_dir
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=3)
    store = pipe.store
    od = store.dir("obj")
    for r in ("v001", "v002", "v003", "v004"):
        _run(od, r, **({"base_run": "v002"} if r == "v003" else {}))
    store.update("obj", tool={"active_run": "v004", "stage_runs": {"core_base": "v003"}})
    # dry run: size + where it is used
    d = api.delete("/api/object/obj/runs/v001?dry=1").json()
    assert d["files"] == 2 and d["bytes"] > 1000 and d["refs"] == []
    assert (od / "runs" / "v001").is_dir()                                   # dry = nothing moved
    # in use: active, a stage's current result, the base of another version -> 409, folder stays
    for r, where in (("v004", "active_run"), ("v003", "core_base"), ("v002", "base_run")):
        x = api.delete(f"/api/object/obj/runs/{r}")
        assert x.status_code == 409 and where in x.json()["detail"], x.text
        assert (od / "runs" / r).is_dir()
    assert api.delete("/api/object/obj/runs/v009").status_code == 404
    assert api.delete("/api/object/obj/runs/..%2Fobject.json").status_code == 404
    # free: moved to <work>/trash/runs, gone from the list
    ok = api.delete("/api/object/obj/runs/v001").json()
    assert ok["success"] and not (od / "runs" / "v001").exists()
    trash = Path(ok["trash"])
    assert trash.parent == store.root.parent / "trash" / "runs" and (trash / "run.json").exists()
    assert "v001" not in [r["run"] for r in api.get("/api/object/obj/runs").json()["runs"]]
    # the highest number is never given again, even after deleting it
    store.update("obj", tool={"active_run": "v002", "stage_runs": {}})
    (od / "runs" / "v003" / "run.json").write_text(json.dumps({"stage": "core"}), encoding="utf-8")   # no base_run
    assert api.delete("/api/object/obj/runs/v004").status_code == 200
    assert next_run_dir(od).name == "v005"
