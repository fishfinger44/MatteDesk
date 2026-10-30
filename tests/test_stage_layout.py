# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage layout of the user decision 2026-10-15: 1 SAM3 · 2 CoTracker (brush corrections) · 3 MatAnyone · 4 ViTMatte ·
5 Eksperymenty · 6 Eksport (engine/stages.py, app.py). Synthetic shot, mock Comfy (SAM3 + MatAnyone stand-ins), point
tracker off (core bbox fallback) – the real CoTracker3 / MatAnyone run in the browser check (LOCAL_CHECKS)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _steps(api):
    d = api.get("/api/object/obj/stages").json()
    return {s["id"]: s["state"] for s in d["steps"]}, d


@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)   # bbox fallback, no GPU in the test
    return _api_project(tmp_path, monkeypatch)


def test_stage_layout_brush_matanyone_export(proj):
    app_module, api, shot, pipe, frames, start = proj
    store = pipe.store
    od = store.dir("obj")
    st, d = _steps(api)
    # right after the core: export open, CoTracker optional (done), MatAnyone to do, ViTMatte open on the mask 1+2
    # 4 = VideoMaMa, 7 = LTX Alpha-Gen (2026-10-16), 8 = Final Touch (2026-10-22, optional, no model)
    assert [s["n"] for s in d["steps"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert st == {"sam3": "done", "cotracker": "done", "matanyone": "todo", "videomama": "todo", "vitmatte": "todo", "final": "todo",
                  "ltx": "todo", "final_touch": "done", "export": "todo"}
    assert "etap opcjonalny" in next(s for s in d["steps"] if s["id"] == "cotracker")["why"]

    # a brush correction (stage 2): "+" just above the object in the start frame, an empty key (end) in the last frame
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    xm = int(np.median(xs))
    top = int(np.where(c[:, xm])[0].min())                                     # top of the mask in this column
    stroke = {"p": [[xm, top + 3], [xm, top - 20]], "radius_px": 4}            # starts inside: connected to the mask
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": start, "polygon": [],
                                                     "strokes": [stroke], "params": {"source": "brush", "track": "rigid"}})
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    r = api.put(f"/api/object/obj/zones/{zid}/keys/{frames[-1]}", json={"polygon": [], "strokes": [], "bg_strokes": []})
    assert r.status_code == 200, r.text                                       # empty key allowed for a brush zone
    st, d = _steps(api)
    assert st["cotracker"] == "todo" and d["brush_zones"] == 1

    # „▶▶ Zastosuj poprawki”: the mask 1+2 on the shot
    r = api.post("/api/object/obj/core_fix")
    assert r.status_code == 200, r.text
    fixed = {f: cv2.imread(str(od / "core_fix" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for f in (start, frames[-1])}
    assert fixed[start][top - 10, xm] and not c[top - 10, xm]                # the stroke joined the mask (connected to it)
    assert not fixed[frames[-1]].sum() > pipe.load_core("obj", [frames[-1]])[frames[-1]].sum() + 50   # the empty key ends it
    st, d = _steps(api)
    assert st["cotracker"] == "done" and d["has"]["core_fix"]
    assert api.get(f"/api/object/obj/frame/{start}?view=core_fix").status_code == 200

    # stage 3: MatAnyone on the mask 1+2 -> the base version with alpha_base in every frame
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    rj = json.loads((od / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "matanyone" and rj["edge_mode"] == "matanyone" and rj["core_fix_sig"] and rj["settings_sig"]
    assert len(list((od / "runs" / run / "alpha_base").rglob("a_*.exr"))) >= len(frames)
    st, d = _steps(api)
    assert st["matanyone"] == "done" and st["vitmatte"] == "todo" and d["stage_runs"]["matanyone"] == run
    runs = api.get("/api/object/obj/runs").json()["runs"]
    assert next(x for x in runs if x["run"] == run)["stage"] == "matanyone"

    # export sources: the mask 1+2 exactly, the stage-3 version; no ViTMatte result -> 409 with the reason
    r = api.post("/api/export", json={"objects": ["obj"], "source": "core", "format": "png"})
    assert r.status_code == 200, r.text
    assert r.json()["runs"] == {"obj": "core"}
    out = cv2.imread(str(Path(r.json()["outputs"]["export_dir"]) / "mattes" / "obj" / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(out > 127, fixed[start])
    assert _steps(api)[0]["export"] == "done"
    r = api.post("/api/export", json={"objects": ["obj"], "source": "matanyone", "format": "png"})
    assert r.status_code == 200 and r.json()["runs"] == {"obj": run}
    r = api.post("/api/export", json={"objects": ["obj"], "source": "vitmatte", "format": "png"})
    assert r.status_code == 409 and "ViTMatte" in r.json()["detail"]
    r = api.post("/api/export", json={"objects": ["obj"], "source": "version", "run": run, "format": "png"})
    assert r.status_code == 200 and r.json()["runs"] == {"obj": run}

    # a change of the brush correction: the mask 1+2 and the MatAnyone result are stale; ViTMatte stays open (2026-10-17)
    # but its MatAnyone base is not current – it says so
    # (2026-10-30: the only layer switched off = core_fix/ still holds its fix – stage 2 is stale, so ViTMatte, which
    # waits for stage 2 (needs_of), says so; the UI rebuilds the mask 1+2 at once when no layer is left on)
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"enabled": False}).status_code == 200
    st, d = _steps(api)
    assert st["matanyone"] == "stale" and st["cotracker"] == "stale" and st["vitmatte"] == "locked" and st["export"] == "stale"
    assert "CoTracker3" in next(s for s in d["steps"] if s["id"] == "vitmatte")["why"]
    r = api.post("/api/export", json={"objects": ["obj"], "source": "matanyone", "format": "png"})
    assert r.status_code == 409

    # every brush layer deleted: the old core_fix/ on disk is not offered to the views (user report 2026-10-30: the view
    # kept the deleted fixes) – the mask 1+2 is the SAM 3.1 core again
    assert api.delete(f"/api/object/obj/zones/{zid}").status_code == 200
    st, d = _steps(api)
    assert st["cotracker"] == "stale" and not d["has"]["core_fix"]          # the group / export would read it: stale
    r = api.post("/api/object/obj/core_fix")                                 # rebuilt = the (stabilised) SAM 3.1 core
    assert r.status_code == 200, r.text
    st, d = _steps(api)
    assert st["cotracker"] == "done" and d["has"]["core_fix"]
    m12 = cv2.imread(str(od / "core_fix" / f"c_{start:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert not m12[top - 10, xm]                                             # the deleted stroke is gone from the mask 1+2


def test_vitmatte_requires_a_current_matanyone_base(proj):
    """Bug 2026-10-17: the stage is open after stages 1 + 2 (the base selector lives inside it); the MatAnyone base
    without a stage-3 result is refused when computing, and the stage says why."""
    from engine.stages import require
    app_module, api, shot, pipe, frames, start = proj
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "matanyone"}).status_code == 200
    require(pipe.store, "obj", pipe.store.root.parent, "vitmatte")              # open
    st, d = _steps(api)
    assert st["vitmatte"] == "todo" and "MatAnyone" in next(s for s in d["steps"] if s["id"] == "vitmatte")["why"]
    with pytest.raises(ValueError, match="etap 3 MatAnyone"):
        pipe.apply_zone("obj", frames, None, stage="vitmatte")
    r = api.post("/api/object/obj/finalize", json={"stage": "other"})
    assert r.status_code == 400


def test_matanyone_gets_every_key_and_the_brush_frames(proj):
    """User 2026-10-15 („ustaw, żeby działała najlepiej”): every key of the segment goes to MatAnyone2RunKeyframes (the node
    keeps <= 3 permanent per pass and re-anchors on the rest through its working memory) + the frames of the brush keys."""
    from engine.refine import pick_keyframes
    assert pick_keyframes([1, 5, 9, 20, 30], 9) == [1, 5, 9, 20, 30]
    assert pick_keyframes([1, 5, 9, 20, 30], 9, max_keys=3) == [1, 5, 9]
    app_module, api, shot, pipe, frames, start = proj
    c = pipe.load_core("obj", [start])[start]
    xm = int(np.median(np.where(c)[1])); top = int(np.where(c[:, xm])[0].min())
    far = frames[0] if frames[0] != start else frames[-1]
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": far, "polygon": [],
                                                     "strokes": [{"p": [[xm, top + 3], [xm, top - 10]], "radius_px": 4}],
                                                     "params": {"source": "brush", "track": "rigid"}})
    assert r.status_code == 200, r.text
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    assert far in r.json()["report"]["keys_used"] and start in r.json()["report"]["keys_used"]


def test_vitmatte_apply_computes_only_the_work_area(proj):
    """User report 2026-10-22: stage 5 „Zastosuj na ujęcie” computed the whole shot, not the work area (stages 3 / 4 do)."""
    app_module, api, shot, pipe, frames, start = proj
    api.put("/api/object/obj/settings", json={"vitmatte_base": "core"})
    wa = [frames[1], frames[-2]]
    assert api.put("/api/project/work_area", json={"range": wa}).status_code == 200
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": start, "polygon": [[1, 1], [9, 1], [9, 9]],
                                                     "strokes": [], "params": {"source": "brush", "track": "rigid"}})
    zid = r.json()["zone"]["zone_id"]
    assert api.post("/api/object/obj/core_fix").status_code == 200
    r = api.post(f"/api/object/obj/zones/{zid}/apply?stage=vitmatte")
    assert r.status_code == 200, r.text
    vm = r.json()["run"]
    runs = {x["run"]: x for x in api.get("/api/object/obj/runs").json()["runs"]}
    assert runs[vm]["coverage"] == [wa], runs[vm]["coverage"]
    assert runs[vm]["edge_engine"] == "vitmatte"                   # HUD ⇄: which engine computed the version


def test_vitmatte_base_mask_1_2_and_the_comparison(proj):
    """Stage 4 on the mask 1+2 (vitmatte_base = core): open right after stages 1 + 2, base = the mask written as a version,
    a result per base (vitmatte@core / vitmatte@matanyone) and the |A − B| view for the comparison."""
    from engine.stages import require
    app_module, api, shot, pipe, frames, start = proj
    st, d = _steps(api)
    assert st["vitmatte"] == "todo" and d["vitmatte_base"] == "core"           # no MatAnyone result: default = mask 1+2
    r = api.put("/api/object/obj/settings", json={"vitmatte_base": "core"})
    assert r.status_code == 200 and r.json()["object"]["vitmatte_base"] == "core"
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "x"}).status_code == 400
    st, d = _steps(api)
    assert st["vitmatte"] == "todo" and d["vitmatte_base"] == "core"           # no MatAnyone needed
    require(pipe.store, "obj", pipe.store.root.parent, "vitmatte")
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "all", "frame": start, "polygon": [[1, 1], [9, 1], [9, 9]],
                                                     "strokes": [], "params": {"source": "brush", "track": "rigid"}})
    zid = r.json()["zone"]["zone_id"]
    assert _steps(api)[0]["vitmatte"] == "locked"                                # brush zone not applied: stage 2 first
    assert api.post("/api/object/obj/core_fix").status_code == 200
    r = api.post(f"/api/object/obj/zones/{zid}/apply?stage=vitmatte")
    assert r.status_code == 200, r.text
    vm = r.json()["run"]
    st, d = _steps(api)
    base = d["stage_runs"]
    assert st["vitmatte"] == "done" and base["vitmatte@core"] == vm and base["vitmatte"] == vm and base["vitmatte@matanyone"] is None
    runs = {x["run"]: x for x in api.get("/api/object/obj/runs").json()["runs"]}
    core_run = next(k for k, x in runs.items() if x["stage"] == "core")
    assert runs[vm]["stage"] == "vitmatte" and runs[vm]["base"] == "core"
    # a second Zastosuj reuses the base version (still current)
    r = api.post(f"/api/object/obj/zones/{zid}/apply?stage=vitmatte")
    assert r.status_code == 200
    vm = r.json()["run"]
    assert sum(1 for x in api.get("/api/object/obj/runs").json()["runs"] if x["stage"] == "core") == 1
    # comparison view |A − B| of two versions
    assert api.get(f"/api/object/obj/frame/{start}?view=alpha_diff&v={core_run}&v2={vm}").status_code == 200
    assert api.get(f"/api/object/obj/frame/{start}?view=alpha_diff&v={core_run}").status_code == 400
    # back to the MatAnyone base: open but "todo" until stage 3, the result on the mask 1+2 stays listed for the comparison
    api.put("/api/object/obj/settings", json={"vitmatte_base": "matanyone"})
    st, d = _steps(api)
    assert st["vitmatte"] == "todo" and d["stage_runs"]["vitmatte@core"] == vm


def test_group_is_computed_as_a_whole_with_its_seam(proj):
    """PLAN_GROUP_COMPOSE step 4 (user decisions 2026-10-15 / 2026-10-29): a group entity is computed in stage 3 as ONE
    whole – its input is the composition of the members („Do grupy z:”, here Core) with the seam; the band preview of
    the palette (edge_preview) works before computing; the members keep their own results (none is redirected)."""
    from engine import groups
    from engine.crop import CropSegment, extract_crop
    from engine.stages import stage_run
    app_module, api, shot, pipe, frames, start = proj
    store = pipe.store
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    store.create("hand", ["hand"], depth_order=1)
    cd = store.dir("hand") / "core"; cd.mkdir(parents=True)
    m = np.zeros((shot.h, shot.w), np.uint8)
    y0, x0 = int(ys.max()) - 4, int(np.median(xs)) - 10          # a "hand" under the object, a 3 px gap at the contact
    m[y0 + 7:y0 + 37, x0:x0 + 20] = 255
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    (cd / "core.json").write_text(json.dumps({"coverage": [[frames[0], frames[-1]]]}))
    groups.create_group(store, "grp", ["obj", "hand"])
    st = {s["id"]: s["state"] for s in api.get("/api/object/grp/stages").json()["stages"]}
    assert st["core"] == "done" and st["matanyone"] == "todo"           # stages 1 + 2 of a group = its composition
    # the band preview (palette „Krawędź”) of the group: 255 = mask kept, 128 = band, on the composition
    r = api.get(f"/api/object/grp/edge_preview/{start}?band_scale=1.5&shift_px=1")
    assert r.status_code == 200, r.text
    assert r.headers["X-MD-Source"].startswith("mask12") and "group" in r.headers["X-MD-Source"]
    assert int(r.headers["X-MD-Band-Px"]) > 0
    pv = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert (pv == 128).any() and (pv == 255).any()
    wide = int(api.get(f"/api/object/grp/edge_preview/{start}?band_out_px=30&band_in_px=8").headers["X-MD-Band-Px"])
    narrow = int(api.get(f"/api/object/grp/edge_preview/{start}?band_out_px=6&band_in_px=2").headers["X-MD-Band-Px"])
    assert wide > narrow
    # stage 3 on the group entity: the whole
    r = api.post("/api/object/grp/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["object"] == "grp" and d["group"] == ["obj", "hand"]
    run = d["run"]
    rj = json.loads((store.dir("grp") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["group"] == ["obj", "hand"] and rj["stage"] == "matanyone"
    assert stage_run(store, "grp", "matanyone")[:2] == (run, True)
    cr = cv2.imread(str(next((store.dir("grp") / "runs" / run / "core").rglob(f"c_{start:04d}.png"))), cv2.IMREAD_GRAYSCALE)
    # the preview's mask = the core stage 3 handed to MatAnyone2 = the composition (seam filled), in the group crop
    _, info = pipe.edge_preview("grp", start)
    rseg = next(CropSegment.from_dict(x) for x in rj["segments_def"] if x["first"] <= start <= x["last"])
    assert list(info["seg"].crop) == list(rseg.crop)
    assert np.array_equal(extract_crop(info["core"].astype(np.float32), rseg) > 0.5, cr > 127)
    comp = groups.group_alpha_loader(store, "grp", shot.w, shot.h)(start)
    assert comp[y0 + 5, x0 + 10] == 1.0                                # the 3 px gap is filled (seam) in the input
    # the members are not redirected: no stage-3 result of their own
    assert stage_run(store, "hand", "matanyone")[0] is None and stage_run(store, "obj", "matanyone")[0] is None
    # a change of the composition (a member's source) -> the whole is stale (↻)
    groups.set_seam(store, "grp", 9)
    assert stage_run(store, "grp", "matanyone")[:2] == (run, False)
    # detaching a member with a whole result: warning first, then the whole goes to the trash
    r = groups.remove_member(store, "grp", "hand")
    assert r.get("needs_confirm") and "matanyone" in r["whole_stages"]
    assert groups.remove_member(store, "grp", "hand", confirm=True)["trash"]
    assert stage_run(store, "grp", "matanyone")[0] is None


def test_experiments_core_base_made_on_pick(proj):
    """Stage 6 offers the computed mask 1+2 as a base before any MatAnyone2 / ViTMatte run: POST core_base writes it as a
    version (reused while current) and the analysis accepts its id."""
    app_module, api, shot, pipe, frames, start = proj
    r = api.post("/api/object/obj/core_base")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert api.post("/api/object/obj/core_base").json()["run"] == run          # reused, not duplicated
    runs = {x["run"]: x for x in api.get("/api/object/obj/runs").json()["runs"]}
    assert runs[run]["stage"] == "core"
    assert api.get("/api/object/obj/stages").json()["stage_runs"]["core_base"] == run
