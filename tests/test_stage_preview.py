# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage preview (user request 2026-10-17): a few frames of stage 3 MatAnyone / 5 ViTMatte before the long run.
User condition: a preview is never a stage result – no version, no stage_runs / active_run / core_base, no base unlocked.
Synthetic shot, mock Comfy (same fixture as test_stage_layout). VideoMaMa needs its models – its path is covered by
the same code as the full run (vmm_run) and checked on the station (LOCAL_CHECKS)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_stage_layout import _steps, proj  # noqa: E402,F401


def _state(pipe):
    obj = pipe.store.get("obj")
    runs = sorted(p.name for p in (pipe.store.dir("obj") / "runs").glob("v*")) if (pipe.store.dir("obj") / "runs").exists() else []
    return runs, dict(obj["tool"].get("stage_runs") or {}), obj["tool"].get("active_run")


def test_window():
    from engine.stage_preview import window
    fr = list(range(1, 21))
    assert window(fr, 10, 4) == list(range(6, 15))
    assert window(fr, 1, 4) == list(range(1, 10))          # shifted inwards at the start
    assert window(fr, 20, 2) == [16, 17, 18, 19, 20]
    assert window([3, 4], 10, 4) == [3, 4]
    assert window([], 5, 2) == []


def test_matanyone_preview_changes_nothing(proj):
    app_module, api, shot, pipe, frames, start = proj
    before = _state(pipe)
    st0, _ = _steps(api)
    r = api.post(f"/api/object/obj/stage_preview/{start}?stage=matanyone")
    assert r.status_code == 200, r.text
    d = r.json()
    assert start in d["frames"] and len(d["frames"]) <= 9 and d["stale"] is False
    assert _state(pipe) == before                            # no version, no stage_runs, active run untouched
    st1, info = _steps(api)
    assert st1 == st0 and st1["matanyone"] == "todo"           # stage not done, nothing unlocked
    assert not info["stage_runs"].get("matanyone")
    # frames of the window are served, others are not
    assert api.get(f"/api/object/obj/stage_preview/matanyone/{start}").status_code == 200
    outside = next((f for f in frames if f not in d["frames"]), max(frames) + 100)
    assert api.get(f"/api/object/obj/stage_preview/matanyone/{outside}").status_code == 404
    m = api.get("/api/object/obj/stage_preview/matanyone").json()
    assert m["exists"] and m["stale"] is False and m["center"] == start
    # other settings -> the preview is reported stale
    assert api.put("/api/object/obj/settings", json={"band_out_px": 31}).status_code == 200
    assert api.get("/api/object/obj/stage_preview/matanyone").json()["stale"] is True
    assert api.get(f"/api/object/obj/stage_preview/matanyone/{start}").headers["X-MD-Stale"] == "1"


def test_vitmatte_preview_on_mask_1_2_makes_no_core_base(proj):
    app_module, api, shot, pipe, frames, start = proj
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "core"}).status_code == 200
    before = _state(pipe)
    r = api.post(f"/api/object/obj/stage_preview/{start}?stage=vitmatte")
    assert r.status_code == 200, r.text
    runs, sr, active = _state(pipe)
    assert (runs, sr, active) == before and "core_base" not in sr and "vitmatte" not in sr
    assert _steps(api)[0]["vitmatte"] == "todo"
    # the MatAnyone base without a stage-3 result is refused with the reason (no preview of a missing base)
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "matanyone"}).status_code == 200
    r = api.post(f"/api/object/obj/stage_preview/{start}?stage=vitmatte")
    assert r.status_code == 409 and "MatAnyone" in r.json()["detail"]


def test_bad_stage(proj):
    app_module, api, shot, pipe, frames, start = proj
    assert api.post(f"/api/object/obj/stage_preview/{start}?stage=ltx").status_code == 400
    assert api.get("/api/object/obj/stage_preview/ltx").status_code == 400
    assert api.get("/api/object/obj/stage_preview/vitmatte").json()["exists"] is False


def test_auto_sliders_version_reused_unless_referenced(proj):
    """User decision 2026-10-17: sliders apply themselves; the automatic sliders version is overwritten in place – unless
    something builds on it (then a new version, the old one untouched)."""
    import json
    app_module, api, shot, pipe, frames, start = proj
    od = pipe.store.dir("obj")
    runs = od / "runs"
    (runs / "v101").mkdir(parents=True)                       # a model run
    (runs / "v101" / "run.json").write_text(json.dumps({"stage": "videomama"}), encoding="utf-8")
    (runs / "v102").mkdir()                                   # its automatic sliders version
    auto = {"stage": "videomama", "auto_refinish": True, "refinished_from": "v101"}
    (runs / "v102" / "run.json").write_text(json.dumps(auto), encoding="utf-8")
    d, src = pipe._refinish_dir("obj", "v102", auto, auto=True)
    assert d.name == "v102" and src == "v101"                 # overwritten in place, still made from the model run
    d, src = pipe._refinish_dir("obj", "v102", auto, auto=False)
    assert d.name != "v102" and src == "v102"                 # the old button / API without auto: always a new version
    d.rmdir()
    d, src = pipe._refinish_dir("obj", "v101", {"stage": "videomama"}, auto=True)
    assert d.name not in ("v101", "v102")                     # a model run is never overwritten
    d.rmdir()
    # something builds on v102 (a ViTMatte version made on it) -> a new version
    (runs / "v103").mkdir()
    (runs / "v103" / "run.json").write_text(json.dumps({"stage": "vitmatte", "applied_from": "v102"}), encoding="utf-8")
    assert pipe.run_referenced("obj", "v102")
    d, _ = pipe._refinish_dir("obj", "v102", auto, auto=True)
    assert d.name != "v102"
