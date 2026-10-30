# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stages 4 (VideoMaMa) and 7 (LTX-2.5) recomputed on a group entity (docs/PLAN_GROUP_COMPOSE.md step 4): the input is
the group's composition, the result lives in the group's runs/, members keep their own results."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import groups
from engine.stages import stage_run


@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    store = pipe.store
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    store.create("hand", ["hand"], depth_order=1)
    cd = store.dir("hand") / "core"
    cd.mkdir(parents=True)
    m = np.zeros((shot.h, shot.w), np.uint8)
    y0, x0 = int(ys.max()) - 4, int(np.median(xs)) - 10        # under the object, a 3 px gap at the contact
    m[y0 + 7:y0 + 37, x0:x0 + 20] = 255
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    (cd / "core.json").write_text(json.dumps({"coverage": [[frames[0], frames[-1]]]}))
    groups.create_group(store, "grp", ["obj", "hand"])
    return app_module, api, shot, pipe, frames, start, (y0, x0)


def _ready(monkeypatch, app_module, key):
    monkeypatch.setattr(app_module, key, lambda: {"missing_models": [], "missing_gb": 0, "missing_nodes": [],
                                                  "models": [], "ready": True})


def test_group_videomama_on_the_composition(proj, tmp_path, monkeypatch):
    from test_videomama import _vmm_handler
    app_module, api, shot, pipe, frames, start, (y0, x0) = proj
    rec = []
    pipe.client.handlers["SaveImageAdvanced"] = _vmm_handler(tmp_path / "vmm_out", rec)
    pipe.client.output_root = tmp_path / "vmm_out"
    _ready(monkeypatch, app_module, "_vmm_status")
    assert api.put("/api/object/grp/videomama", json={"base": "core", "batch": 4, "overlap": 2}).status_code == 200
    r = api.post("/api/object/grp/videomama")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    od = pipe.store.dir("grp")
    rj = json.loads((od / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "videomama" and rj["group"] == ["obj", "hand"]
    assert stage_run(pipe.store, "grp", "videomama")[:2] == (run, True)
    assert stage_run(pipe.store, "obj", "videomama")[0] is None and stage_run(pipe.store, "hand", "videomama")[0] is None
    from engine.export import run_alpha_loader
    a = run_alpha_loader(od, run, shot.w, shot.h)(start)
    assert a[y0 + 20, x0 + 10] > 0.5 and a[y0 + 5, x0 + 10] > 0.5        # the member and the filled seam
    groups.set_member_source(pipe.store, "grp", "hand", {"source": "core"})   # same source: still current
    assert stage_run(pipe.store, "grp", "videomama")[1]
    groups.set_seam(pipe.store, "grp", 0)
    assert stage_run(pipe.store, "grp", "videomama")[:2] == (run, False)


def test_group_ltx_on_the_composition(proj, tmp_path, monkeypatch):
    from test_ltx_alpha import _alphagen_handler
    app_module, api, shot, pipe, frames, start, (y0, x0) = proj
    rec = []
    pipe.client.handlers["SaveImageAdvanced"] = _alphagen_handler(tmp_path / "ltx_out", rec)
    pipe.client.output_root = tmp_path / "ltx_out"
    _ready(monkeypatch, app_module, "_ltx_status")
    assert api.put("/api/object/grp/ltx", json={"guide": "plain", "max_frames": 9, "overlap": 2}).status_code == 200
    r = api.post("/api/object/grp/ltx")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    rj = json.loads((pipe.store.dir("grp") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "ltx" and rj["group"] == ["obj", "hand"]
    assert stage_run(pipe.store, "grp", "ltx")[:2] == (run, True)
    st = {s["id"]: s["state"] for s in api.get("/api/object/grp/stages").json()["stages"]}
    assert st["ltx"] == "done"
    r = groups.ungroup(pipe.store, "grp")
    assert r.get("needs_confirm") and "ltx" in r["whole_stages"]
    assert groups.ungroup(pipe.store, "grp", confirm=True)["success"]
    assert pipe.store.get("hand").get("group") is None


def test_group_vitmatte_zone_applied_on_the_composition(proj, monkeypatch):
    import engine.vitmatte as vm
    from test_zone_vitmatte import MockRunner
    app_module, api, shot, pipe, frames, start, (y0, x0) = proj
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    assert api.put("/api/object/grp/settings", json={"vitmatte_base": "core"}).status_code == 200
    ys, xs = np.where(pipe.group_core("grp", [start])[start])
    band = [{"p": [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())]], "radius_px": 20}]
    r = api.post("/api/object/grp/zones/new", json={"mode": "add", "criterion": "matte", "frame": start, "polygon": [],
                                                    "strokes": band, "matting": "vitmatte", "preset": "hair",
                                                    "params": {"temporal": "none"}})
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    r = api.post(f"/api/object/grp/zones/{zid}/apply?stage=vitmatte")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert stage_run(pipe.store, "grp", "vitmatte")[:2] == (run, True)
    rj = json.loads((pipe.store.dir("grp") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj.get("base") == "core"
    from engine.export import run_alpha_loader
    a = run_alpha_loader(pipe.store.dir("grp"), run, shot.w, shot.h)(start)
    assert a[y0 + 20, x0 + 10] > 0.5                                     # the member below is in the group's result


def test_group_export_whole_composition_and_final_touch(proj):
    """Step 4 (user decisions 2026-10-29): the group is exported as ONE layer – its whole result of the chosen stage,
    its Final Touch, or the composition when nothing was computed on the whole; members are exported with their own
    choice; the group stands at its front-most member (never subtracts / is subtracted by its own members)."""
    from engine.export import frame_alphas, run_alpha_loader
    app_module, api, shot, pipe, frames, start, (y0, x0) = proj
    store = pipe.store
    comp = groups.group_alpha_loader(store, "grp", shot.w, shot.h)(start)
    # nothing computed on the whole: the composition (source matanyone falls back to it, reported)
    r = api.post("/api/export", json={"objects": ["grp"], "source": "matanyone", "format": "png"})
    assert r.status_code == 200, r.text
    assert r.json()["runs"]["grp"] == "core"
    ex = Path(r.json()["outputs"]["export_dir"]) / "mattes"
    png = cv2.imread(str(ex / "grp" / f"grp_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(png, (np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8))
    entry = next(e for e in r.json()["outputs"]["objects"] if e["name"] == "grp")
    assert entry["members"] == ["obj", "hand"] and entry["depth_order"] == 0
    assert png[y0 + 5, x0 + 10] == 255                                   # the seam of the composition
    # a member of the group exported in the same context keeps its own alpha (not cut by its group)
    objs = [{**store.get("obj"), "tool": {**store.get("obj")["tool"], "active_run": "core"}},
            {**store.get("hand"), "tool": {**store.get("hand")["tool"], "active_run": "core"}},
            {**store.get("grp"), "tool": {**store.get("grp")["tool"], "active_run": "core"}}]
    al = frame_alphas(store.root.parent, objs, start, shot.w, shot.h)
    assert np.array_equal(al["grp"], comp) and al["hand"][y0 + 20, x0 + 10] == 1.0
    # a whole result exists: strict source -> that stage's version
    r = api.post("/api/object/grp/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    r = api.post("/api/export", json={"objects": ["grp"], "source": "matanyone", "format": "png"})
    assert r.status_code == 200 and r.json()["runs"]["grp"] == run
    assert api.post("/api/export", json={"objects": ["grp"], "source": "videomama"}).status_code == 409   # no silent fallback
    # Final Touch of the group: a white dot on the composition base, exported as the group's matte
    assert api.put("/api/object/grp/final_touch", json={"source": "core"}).status_code == 200
    dot = {"mode": "add", "radius_px": 6, "hardness": 1.0, "p": [[20, 20]]}
    assert api.put(f"/api/object/grp/final_touch/frames/{start}", json={"strokes": [dot]}).status_code == 200
    r = api.post("/api/export", json={"objects": ["grp"], "source": "final_touch", "format": "png"})
    assert r.status_code == 200, r.text
    assert r.json()["runs"]["grp"] == "ft:core"
    png = cv2.imread(str(ex / "grp" / f"grp_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert png[20, 20] == 255 and comp[20, 20] == 0.0
    ft = run_alpha_loader(store.dir("grp"), "ft:core", shot.w, shot.h)(start)
    assert np.array_equal(png, (np.clip(ft, 0, 1) * 255 + 0.5).astype(np.uint8))
    # Final Touch counts as a whole result: detaching a member warns
    assert "final_touch" in groups.whole_results(store, "grp")


def test_member_brings_its_final_touch(proj):
    """„Do grupy z: Final Touch” (user decision 2026-10-29): the member brings its Final Touch – the base chosen on
    stage 8 + its hand corrections; a change of the corrections = ↻ on the group's whole recompute."""
    app_module, api, shot, pipe, frames, start, _ = proj
    store = pipe.store
    pal = lambda: {m["member"]: m for m in app_module._group_payload("grp")["members"]}   # the palette entry
    assert "final_touch" in [a["source"] for a in pal()["obj"]["available"]]
    assert api.put("/api/group/grp/members/obj/source", json={"source": "final_touch"}).status_code == 200
    mi = pal()["obj"]
    assert (mi["source"], mi["run"], mi["valid"]) == ("final_touch", "ft:core", True)   # default base: the mask 1+2
    sig0 = groups.group_sig(store, "grp")
    before = groups.group_alpha_loader(store, "grp", shot.w, shot.h)(start)
    dot = {"mode": "add", "radius_px": 6, "hardness": 1.0, "p": [[20, 20]]}
    assert api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": [dot]}).status_code == 200
    after = groups.group_alpha_loader(store, "grp", shot.w, shot.h)(start)
    assert before[20, 20] == 0.0 and after[20, 20] == 1.0
    assert groups.group_sig(store, "grp") != sig0
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 200, r.text
    assert api.put("/api/object/obj/final_touch", json={"source": "matanyone"}).status_code == 200
    assert pal()["obj"]["run"] == "ft:" + r.json()["run"]                 # the base chosen on stage 8
