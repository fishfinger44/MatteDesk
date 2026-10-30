# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 3 of docs/PLAN_HAIR_TOOL.md (synthetic): the `matte` zone criterion in engine/zones.py, holdout of the objects in
front (pipeline.front_holdout / MissingFront -> 409), the soft override in refine.finalize (alpha outside the zone is
bitwise the run without the zone), hygiene keeps strands in the zone, API preview views, CLI `zone-matte` / `zones` /
`queue-all` and the "Policz wszystko" queue preset."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import CropSegment
from engine.edge import hygiene
from engine.refine import Finalizer
from engine.zones import MissingFront, Zone, apply_zones, load_zones, save_zone, zone_bg_mask_at, zone_mask_at
from tests.test_localmatte import _motley, _scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def _poly_of(mask: np.ndarray):
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _frames_scene(n=5):
    img, core, hair, line, region = _scene(_motley(256, 320))
    frames = list(range(1, n + 1))
    return img, core, hair, line, region, frames


def test_apply_zones_matte_soft_override_also_in_user_key_frames():
    img, core, hair, line, region, frames = _frames_scene()
    ys, xs = np.where(line & region)
    bg_stroke = {"p": [[int(x), int(y)] for x, y in zip(xs[::6], ys[::6])], "radius_px": 3}
    z = Zone("z_01", "add", "matte", frames[0], polygon=_poly_of(region), bg_strokes=[bg_stroke], matting="knn")
    # the polygon = bbox of the ring region: use the real ring as the drawing via a second zone-less check
    cores = {f: core for f in frames}
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    res = apply_zones(frames, cores, lambda f: img, [z], tr, user_keys=[frames[2]])
    assert set(res.soft) == set(frames)                       # user-key frame 3 NOT skipped (plan section 12)
    area = zone_mask_at(z, core.shape)
    bg = zone_bg_mask_at(z, core.shape)
    assert bg.any() and (bg & line).sum() > 0
    for f in frames:
        a, ar = res.soft[f]
        assert np.array_equal(ar, area)
        assert a[hair & area].mean() > 0.6
        assert a[bg & area & ~core].max() == 0.0               # "-" strokes force background
        assert a[~area].max() == 0.0
    assert np.array_equal(res.core[frames[0]], core)           # the core itself untouched
    # "-" strokes are static by default (a line on the wall does not move with the head); the zone itself moves
    shift = np.array([[1, 0, 40], [0, 1, 0]], np.float32)
    moved = apply_zones(frames[:1], {1: core}, lambda f: img, [z], {"z_01": {1: shift}})
    a_m, ar_m = moved.soft[1]
    assert np.array_equal(ar_m, zone_mask_at(z, core.shape, shift)) and not np.array_equal(ar_m, area)
    assert a_m[bg & ar_m & ~core].max() == 0.0                 # the stroke stayed where the line is
    z_follow = Zone("z_09", "add", "matte", 1, polygon=z.polygon, bg_strokes=z.bg_strokes, matting="knn", bg_static=False)
    moved2 = apply_zones(frames[:1], {1: core}, lambda f: img, [z_follow], {"z_09": {1: shift}})
    bg_shifted = zone_bg_mask_at(z_follow, core.shape, shift)
    assert moved2.soft[1][0][bg_shifted & ar_m & ~core].max() == 0.0
    assert res.stats["z_01"]["frames"] == len(frames) and res.stats["z_01"]["engine"] == "knn"


def test_apply_zones_matte_holdout_and_remove_mode_refused():
    img, core, hair, line, region, frames = _frames_scene(3)
    z = Zone("z_01", "add", "matte", 1, polygon=_poly_of(region), matting="chroma")
    hold = np.zeros_like(core)
    hold[:, : core.shape[1] // 2] = True
    res = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z],
                      {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}, holdout=lambda f: hold)
    a, ar = res.soft[1]
    assert a[hold & ar & ~core].max() == 0.0                   # 0 under the object in front
    assert a[~hold & hair & ar].mean() > 0.5
    # "Licz mimo to bez holdoutu" -> yellow hint in every frame
    z2 = Zone("z_02", "add", "matte", 1, polygon=_poly_of(region), matting="chroma", holdout=False)
    res2 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z2],
                       {"z_02": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}, holdout=lambda f: hold)
    assert all("bez holdoutu" in " ".join(res2.flags[f]) for f in frames)
    assert res2.soft[1][0][hold & hair & region].mean() > 0.5   # holdout not applied
    with pytest.raises(ValueError):
        apply_zones(frames, {f: core for f in frames}, lambda f: img,
                    [Zone("z_03", "remove", "matte", 1, polygon=_poly_of(region))],
                    {"z_03": {f: np.eye(2, 3, dtype=np.float32) for f in frames}})


def test_zone_schema_roundtrip_and_old_files(tmp_path):
    d = tmp_path / "obj"
    z = Zone("z_01", "add", "matte", 7, polygon=[[0, 0], [10, 0], [10, 10]], preset="smoke", matting="hybrid",
             prompt="wytnij dym", bg_strokes=[{"p": [[1, 1], [2, 2]], "radius_px": 2}], engine_hints={"chroma": 1.5},
             params={"chroma_lo": 3.0})
    save_zone(d, z)
    back = load_zones(d)[0]
    # zone@1.2 (2026-10-24): saved with the current schema; params as saved
    assert back.schema == "matte_desk/zone@1.2" and back.preset == "smoke" and back.params == {"chroma_lo": 3.0}
    assert back.bg_strokes == z.bg_strokes and back.engine_hints == {"chroma": 1.5} and back.holdout is True
    # a zone@1.0 file (no new fields) still loads with the defaults
    (d / "zones" / "z_02.json").write_text(json.dumps({"zone_id": "z_02", "mode": "remove", "criterion": "all", "frame": 1,
                                                      "polygon": [], "strokes": [], "param": 0.15, "soft": False,
                                                      "softness": 0.05, "color_lab": None, "static": False, "grow_px": 200,
                                                      "enabled": True, "schema": "matte_desk/zone@1.0"}), encoding="utf-8")
    old = [x for x in load_zones(d) if x.zone_id == "z_02"][0]
    assert old.criterion == "all" and old.preset == "hair" and old.matting == "matanyone" and old.keys == {}


def test_finalize_soft_override_bitwise_outside_zone_and_keeps_strands(tmp_path):
    img, core, hair, line, region, frames = _frames_scene(5)
    h, w = core.shape
    side = 256
    seg = CropSegment("s00", 1, 5, 0, 0, [32, 0, side, side], [0, 0, 0, 0], 1.0)
    cores = {f: core for f in frames}
    z = Zone("z_01", "add", "matte", 1, polygon=_poly_of(region), matting="knn")
    zres = apply_zones(frames, cores, lambda f: img, [z], {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}})

    def run(soft, name):
        fin = Finalizer(client=None, io_dir=tmp_path / "io", plate_w=w, plate_h=h,
                        load_crop_image=lambda f, s, sc: np.zeros((side, side, 3), np.float32))
        rd = tmp_path / name
        rd.mkdir()
        full, rep = fin.finalize({"temporal_smoothing": "normal", "band_scale": 1.0}, [seg], frames, cores, {1: core}, 1,
                                 rd, edge_mode="core", soft_override=soft)
        return full, rep
    plain, _ = run(None, "plain")
    withz, rep = run(zres.soft, "zone")
    area = zone_mask_at(z, core.shape)
    for f in frames:
        assert np.array_equal(withz[f][~area], plain[f][~area]), f      # outside the zone: bitwise the old result
        assert withz[f][hair & area].mean() > 0.6, f                     # strands inside the zone
        assert withz[f][core].min() == 1.0
    # strands disconnected at alpha > 0.5 (8-connectivity) inside the zone survive the hygiene (not "islands")
    assert rep.hygiene.get("islands_removed", 0) == 0


def test_qa_islands_allowed_in_zone_area():
    from engine.qa import run_qa, small_islands
    a = np.zeros((96, 96), np.float32)
    a[30:70, 30:70] = 1.0
    a[10:13, 40:43] = 0.9        # a strand drawn by a matte zone
    a[80:83, 80:83] = 0.9        # a real island elsewhere
    zone = np.zeros((96, 96), bool)
    zone[5:20, 30:60] = True
    assert small_islands(a) == 2 and small_islands(a, allowed=zone) == 1
    frames = list(range(4))
    rep = run_qa(frames, lambda f: a, None, allowed_islands=lambda f: zone)
    assert all(rep.islands[f] == 1 for f in frames)          # the island outside the zone still counts (red)
    rep2 = run_qa(frames, lambda f: a, None, allowed_islands=lambda f: zone | (np.arange(96)[:, None] > 75))
    assert all(rep2.islands[f] == 0 for f in frames) and rep2.gate_ok
    # a gap between strands inside the zone area is not an interior hole either (pipeline passes the area as allowed)
    b = a.copy()
    b[14:26, 44:48] = 1.0                 # strands enclosing...
    b[16:24, 45:47] = 0.0                 # ...a gap, all inside the zone area
    zone2 = zone.copy(); zone2[5:30, 30:60] = True
    rep3 = run_qa(frames, lambda f: b, None, allowed_islands=lambda f: zone2 | (np.arange(96)[:, None] > 75))
    assert all(rep3.holes[f] >= 1 for f in frames)
    rep4 = run_qa(frames, lambda f: b, None, allowed_holes=lambda f: zone2,
                  allowed_islands=lambda f: zone2 | (np.arange(96)[:, None] > 75))
    assert all(rep4.holes[f] == 0 for f in frames) and rep4.gate_ok


def test_hygiene_keep_islands():
    a = np.zeros((64, 64), np.float32)
    core = np.zeros((64, 64), bool)
    core[20:44, 20:44] = True
    a[core] = 1.0
    band = np.zeros((64, 64), bool)
    band[16:48, 16:48] = True
    a[50:53, 50:53] = 0.9      # island outside the core, inside the band
    band[48:56, 48:56] = True
    a2, rep = hygiene(a.copy(), core, band)
    assert rep["islands_removed"] == 1 and a2[50:53, 50:53].max() == 0.0
    keep = np.zeros((64, 64), bool)
    keep[48:56, 48:56] = True
    a3, rep3 = hygiene(a.copy(), core, band, keep_islands=keep)
    assert rep3["islands_removed"] == 0 and np.isclose(a3[50:53, 50:53].min(), 0.9)


# ----------------------------------------------------------------------------- pipeline holdout + API + CLI
def _api_project(tmp_path, monkeypatch, n=6):
    from fastapi.testclient import TestClient
    import app as app_module
    from test_e5_acceptance import _setup
    from synth import display
    shot, pipe, frames, start = _setup(tmp_path, n=n)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    monkeypatch.setitem(app_module.STATE, "comfy_client", pipe.client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    assert api.put("/api/project/mask_plate_denoise", json={"setting": "off"}).status_code == 200
    return app_module, api, shot, pipe, frames, start


def test_pipeline_front_holdout_sources_and_missing(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    store = pipe.store
    # a "hand" in front (depth_order 0) without any mask; obj moves behind it
    store.create("hand", ["hand"], depth_order=0)
    store.update("obj", depth_order=1)
    with pytest.raises(MissingFront) as ei:
        pipe.front_holdout("obj", frames)
    assert ei.value.missing == ["hand"] and {q["action"] for q in ei.value.queue} == {"run", "finalize"}
    fn, used = pipe.front_holdout("obj", frames, need=False)
    assert used == [] and fn(start) is None
    # the hand gets a core -> holdout from the core; the queue proposes its finalize
    cd = store.dir("hand") / "core"
    cd.mkdir(parents=True)
    m = np.zeros((shot.h, shot.w), np.uint8)
    m[100:300, 100:300] = 255
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    fn, used = pipe.front_holdout("obj", frames)
    assert used == ["hand"] and fn(start)[150, 150] and not fn(start)[50, 50]
    # a group member computes on its own (PLAN_GROUP_COMPOSE 2026-10-29): a legacy part in front is a holdout too
    store.update("hand", part_of="obj")
    fn, used = pipe.front_holdout("obj", frames)
    assert used == ["hand"]


def test_api_matte_zone_preview_views_409_and_cli(tmp_path, monkeypatch):
    import md
    from test_md_cli import TCHttp, _run
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "chroma", "preset": "hair"}
    # validation
    assert api.post(f"/api/object/obj/zones/new?preview_frame={start}", json={**z, "mode": "remove"}).status_code == 400
    assert api.post(f"/api/object/obj/zones/new?preview_frame={start}", json={**z, "matting": "nope"}).status_code == 400   # vitmatte allowed since 2026-10-13 (experiment)
    assert api.post(f"/api/object/obj/zones/new?preview_frame={start}", json={**z, "polygon": []}).status_code == 400
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    pv = r.json()["preview"]
    assert "stats" in pv and pv["stats"]["engine"] == "chroma" and pv["holdout_objects"] == []
    for view in ("zone_add", "zone_remove", "zone_alpha", "zone_trimap", "zone_engines"):
        pr = api.get(f"/api/object/obj/frame/{start}?view={view}")
        assert pr.status_code == 200 and pr.headers["content-type"] == "image/png", view
    assert api.get(f"/api/object/obj/frame/{start + 1}?view=zone_alpha").status_code == 404
    assert api.get("/api/object/obj/zones").json()["zones"][0]["criterion"] == "matte"
    # CLI
    draw = tmp_path / "zone.json"
    draw.write_text(json.dumps({"polygon": poly}), encoding="utf-8")
    code, out = _run(["zone-matte", "obj", "--frame", str(start), "--draw", str(draw), "--matting", "chroma", "--zone", zid], TCHttp(api))
    assert code == 0, out
    assert "zapisana; klucz F%d" % start in out and "podgląd F%d" % start in out and "view=zone_alpha" in out
    code, out = _run(["zones", "obj"], TCHttp(api))
    assert code == 0 and "add / matte" in out and "preset hair" in out
    # objects in front without masks -> 409 with the list, CLI exit 2 and the queue hint
    pipe.store.create("hand", ["hand"], depth_order=0)
    pipe.store.update("obj", depth_order=1)
    r = api.post(f"/api/object/obj/zones/{zid}?preview_frame={start}", json=z)
    assert r.status_code == 409 and r.json()["detail"]["missing_front"] == ["hand"]
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 409 and fz.json()["detail"]["missing_front"] == ["hand"], fz.text
    code, out = _run(["zone-matte", "obj", "--frame", str(start), "--draw", str(draw), "--zone", zid], TCHttp(api))
    assert code == 2 and "brak maski: hand" in out and "queue-all" in out
    # "Policz wszystko": the hand (no result) and obj (final invalid after the zone) are queued front to back
    code, out = _run(["queue-all"], TCHttp(api))
    assert code == 0 and "hand" in out and "obj" in out
    q = api.get("/api/queue").json()
    assert [j["object"] for j in q["jobs"]][-2:] == ["hand", "obj"]
    # "Licz mimo to bez holdoutu" -> saved, preview ok
    r = api.post(f"/api/object/obj/zones/{zid}?preview_frame={start}", json={**z, "holdout": False})
    assert r.status_code == 200 and r.json()["zone"]["holdout"] is False
    # finalize (core mode, no Comfy) with the zone and without holdout: a version with the zone stats, yellow hint
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    run = fz.json()["run"]
    rj = json.loads((pipe.store.dir("obj") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert zid in rj["zones"] and rj["zones"][zid]["frames"] == len(frames)
    qa = json.loads((pipe.store.dir("obj") / "runs" / run / "qa.json").read_text(encoding="utf-8"))
    assert any("bez holdoutu" in x[1] for x in qa["flags"][str(start)])
