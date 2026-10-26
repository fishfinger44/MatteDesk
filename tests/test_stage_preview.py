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


def test_videomama_preview_composes_with_band_points(proj, monkeypatch):
    """User report 2026-10-26 (sh060): the VideoMaMa „Preview frames” composed with the sliders alone – the band
    points (CoTracker3) were ignored, so the inset feather crossed the core edge while the preview lines (with points)
    showed it deep inside. The preview must give compose the same points as the full run (_vmm_write): plate -> crop px."""
    import numpy as np
    import engine.videomama as vmm
    from engine.stage_preview import _videomama, window
    app_module, api, shot, pipe, frames, start = proj
    win = window(pipe._visible("obj", frames), start, 2)
    plate_pts = [(40.0, 30.0, 12.0, 9.0), (50.0, 35.0, 10.0, 8.0, 25.0)]   # the 2nd one tilted (5 fields, 2026-10-26)
    monkeypatch.setattr(pipe, "band_points_fn", lambda name, s, fr, only=None: (lambda f: list(plate_pts)))
    monkeypatch.setattr(vmm.VideoMaMa, "run", lambda self, tag, imgs, masks: [np.full(m.shape[:2], 0.5, np.float32)
                                                                               for m in masks])
    seen = []
    real = vmm.compose
    monkeypatch.setattr(vmm, "compose", lambda *a, **k: (seen.append(a[6] if len(a) > 6 else k.get("pts")), real(*a, **k))[1])
    _videomama(pipe, "obj", win, None)
    segs = pipe._segments_in("obj", win)
    x0, y0 = segs[0].crop[0], segs[0].crop[1]
    assert seen and all(p == [(40.0 - x0, 30.0 - y0, 12.0, 9.0), (50.0 - x0, 35.0 - y0, 10.0, 8.0, 25.0)] for p in seen)


def test_matanyone_band_takes_a_tilted_point(proj, monkeypatch):
    """User report 2026-10-26: the MatAnyone2 „Preview frames” failed with „too many values to unpack (expected 4)” –
    the stage-3 band (pipeline._band_fn) unpacked 4 fields of a band point; a tilted one has 5 (angle)."""
    import numpy as np
    from types import SimpleNamespace
    app_module, api, shot, pipe, frames, start = proj
    seen = []
    monkeypatch.setattr(pipe, "band_points_fn",
                        lambda name, s, fr, only=None: (lambda f: seen.append(f) or [(60.0, 40.0, 10.0, 6.0, 20.0)]))
    core = np.zeros((80, 120), np.float32)
    core[20:60, 30:90] = 1
    band = pipe._band_fn(pipe.store.get("obj"))(start, SimpleNamespace(crop=(10, 5, 120, 80)), core, None)
    assert seen == [start] and band.shape == core.shape and band.any()


def test_crop_points_keep_every_field():
    """The one plate -> crop conversion of band points (VideoMaMa, LTX-2.5): the tilt angle survives, none -> None."""
    from types import SimpleNamespace
    from engine.bandmap import crop_points
    seg = SimpleNamespace(crop=(100, 50, 400, 300))
    pts = [(140.0, 80.0, 12.0, 9.0), (150.0, 85.0, 10.0, 8.0, -30.0)]
    assert crop_points(lambda f: pts, 7, seg) == [(40.0, 30.0, 12.0, 9.0), (50.0, 35.0, 10.0, 8.0, -30.0)]
    assert crop_points(None, 7, seg) is None


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


def test_zone_drawn_outside_the_frames_is_followed_to_them(proj):
    """User report 2026-10-22 (CLIP_A): ViTMatte preview around F21 with the zone drawn in F24 → KeyError: 24 (the zone
    transforms needed the core of the drawing frame, outside the preview window). The zone is now followed from its
    drawing frame to the asked frames."""
    app_module, api, shot, pipe, frames, start = proj
    far = frames[-1] if frames[-1] != start else frames[0]
    cx, cy, s = shot.centers[far]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    r = api.post("/api/object/obj/zones/new", json={"mode": "add", "criterion": "matte", "frame": far, "polygon": poly,
                                                     "matting": "chroma", "preset": "hair"})
    assert r.status_code == 200, r.text
    sub = [f for f in frames if abs(f - far) >= 2][:2]                 # frames that do not contain the drawing frame
    assert sub and far not in sub
    core = pipe.load_core("obj", sub)
    res = pipe.apply_zones("obj", sub, core, pipe.keys("obj"), kinds=("matte",))
    assert res is not None and set(res.core) >= set(sub)


def test_vitmatte_preview_is_kept_per_engine(proj):
    """User request 2026-10-22: switching the stage-5 engine and computing the preview again must not throw away the
    other engine's preview – both are kept (slot "vitmatte" / "vitmatte@birefnet"), both stay valid although the engine
    switch rewrites `vitmatte_model` in the zones, and the HUD ⇄ can show either."""
    from engine.stage_preview import slot
    assert slot("vitmatte") == "vitmatte" and slot("vitmatte", "vitmatte") == "vitmatte"
    assert slot("vitmatte", "birefnet") == "vitmatte@birefnet" and slot("matanyone", "birefnet") == "matanyone"
    app_module, api, shot, pipe, frames, start = proj
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "core"}).status_code == 200
    r = api.post(f"/api/object/obj/stage_preview/{start}?stage=vitmatte")
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "vitmatte" and r.json()["slot"] == "vitmatte"
    assert (pipe.store.dir("obj") / "stage_preview" / "vitmatte" / "meta.json").exists()
    assert api.get("/api/object/obj/stage_preview/vitmatte?engine=vitmatte").json()["exists"] is True
    assert api.get("/api/object/obj/stage_preview/vitmatte?engine=birefnet").json()["exists"] is False
    assert api.get("/api/object/obj/stage_preview/vitmatte?engine=nope").status_code == 400
    assert api.get(f"/api/object/obj/stage_preview/vitmatte/{start}?engine=vitmatte").status_code == 200
    assert api.get(f"/api/object/obj/stage_preview/vitmatte/{start}?engine=birefnet").status_code == 404
    # the stage engine switched to BiRefNet: the default slot is now the (missing) BiRefNet preview, the ViTMatte one
    # is still there and still current (the zones' model change is not a settings change of that preview)
    r = api.put("/api/object/obj/edge_engine", json={"engine": "birefnet"})
    assert r.status_code == 200, r.text
    d = api.get("/api/object/obj/stage_preview/vitmatte").json()
    assert d["exists"] is False and d["engine"] == "birefnet" and d["slot"] == "vitmatte@birefnet"
    d = api.get("/api/object/obj/stage_preview/vitmatte?engine=vitmatte").json()
    assert d["exists"] is True and d["stale"] is False
