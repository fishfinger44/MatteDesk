# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plan step 7 (docs/PLAN_HAIR_TOOL.md, sections 8 pkt 6 and 12 / Z8), synthetic, mock Comfy:
- faithful preview: `GET frame/{f}?view=alpha` (INTER_AREA to the proxy, export rounding) == the exported matte of that
  frame processed the same way (hash of the preview = hash of the export); `export=1` = the matte as exported (front objects
  subtracted);
- Z8 tiles: `GET /api/frame/{f}/tile` = crop of the plate at full resolution with the proxy's colour conversion (PNG exact);
  `GET /api/object/{o}/frame/{f}/tile?view=alpha|core|key|zone_alpha` = crops of the masks at plate resolution, the alpha tile
  equal to the exported file (EXR half / PNG 8-bit);
- correction loop: an iteration note stored in the zone key (PUT keys/{f} prompt -> prompt_history, key_zone prompt), CLI
  `zone-keys --prompt`."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from engine.exr_io import read_exr  # noqa: E402
from engine.ingest import ProxyGenerator  # noqa: E402


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _api_project(tmp_path, monkeypatch, n=4, **shot_kw):
    """Like test_zone_matte._api_project, with a 2560x1440 plate so the proxy (1920 px) is smaller than the plate."""
    from fastapi.testclient import TestClient
    import app as app_module
    from test_e5_acceptance import _setup
    from synth import display
    shot, pipe, frames, start = _setup(tmp_path, n=n, **shot_kw)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", pipe.client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    assert api.put("/api/project/mask_plate_denoise", json={"setting": "off"}).status_code == 200
    return app_module, api, shot, pipe, frames, start


def _png(r):
    assert r.status_code == 200, r.text
    m = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert m is not None
    return m


def _u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def _export_file(api, name, f, objects=None, fmt=None):
    body = {"objects": objects if objects is not None else [name]}
    if fmt:
        body["format"] = fmt
    r = api.post("/api/export", json=body)
    assert r.status_code == 200, r.text
    entry = next(e for e in r.json()["outputs"]["objects"] if e["name"] == name)
    return Path(entry["matte"].replace("####", f"{f:04d}"))


def _read_matte(p):
    if p.suffix == ".png":
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0, True
    ch, _ = read_exr(p)
    return ch["R"], False


# ---------------------------------------------------------------------------------------------- Z8: plate tile
def test_plate_tile_is_the_full_resolution_plate_with_the_proxy_colours(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=3, w=2560, h=1440)
    src = ProxyGenerator.resolve_frame_path(shot.plate_pattern, start)
    full = ProxyGenerator.load_frame_rgb(src, colorspace="ACES2065-1")[..., ::-1]     # BGR uint8, plate resolution
    assert full.shape[:2] == (1440, 2560)
    # lossless tile = exactly the crop of the display image the proxy is made from
    t = _png(api.get(f"/api/frame/{start}/tile?x=1000&y=300&w=640&h=400&fmt=png"))
    assert t.shape == (400, 640, 3)
    assert np.array_equal(t, full[300:700, 1000:1640])
    # default JPEG q95: same size, visually the same (mean |diff| < 1.5 / 255)
    r = api.get(f"/api/frame/{start}/tile?x=1000&y=300&w=640&h=400")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    j = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert j.shape == (400, 640, 3) and np.abs(j.astype(np.float32) - full[300:700, 1000:1640]).mean() < 1.5
    # the proxy is the INTER_AREA downscale of the same image (so proxy -> tile does not change colours)
    assert api.get(f"/api/frame/{start}").status_code == 200
    proxy = cv2.imread(str(Path(app_module.STATE["work_dir"]) / "proxy" / "beauty" / f"p_{start:04d}.jpg"), cv2.IMREAD_COLOR)
    assert proxy is not None and proxy.shape[1] == 1920
    small = cv2.resize(full, (proxy.shape[1], proxy.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(small.astype(np.float32) - proxy).mean() < 2.0
    # edge tile (last pixel column / row) and the max size
    t2 = _png(api.get(f"/api/frame/{start}/tile?x=2496&y=1376&w=64&h=64&fmt=png"))
    assert np.array_equal(t2, full[1376:1440, 2496:2560])
    assert api.get(f"/api/frame/{start}/tile?x=0&y=0&w=2048&h=1440&fmt=png").status_code == 200
    # validation: too big / outside the plate / unknown plate or frame
    assert api.get(f"/api/frame/{start}/tile?x=0&y=0&w=2049&h=64").status_code == 400
    assert api.get(f"/api/frame/{start}/tile?x=2500&y=0&w=128&h=64").status_code == 400
    assert api.get(f"/api/frame/{start}/tile?x=0&y=0&w=0&h=64").status_code == 400
    assert api.get(f"/api/frame/{start}/tile?x=0&y=0&w=64&h=64&plate=backlit").status_code == 404
    assert api.get(f"/api/frame/{start + 900}/tile?x=0&y=0&w=64&h=64").status_code == 404
    # LRU of full frames in memory: never more than DISPLAY_CACHE_FRAMES
    for f in frames:
        assert api.get(f"/api/frame/{f}/tile?x=0&y=0&w=64&h=64").status_code == 200
    assert len(app_module.STATE["display_cache"]) <= app_module.DISPLAY_CACHE_FRAMES


# ---------------------------------------------------------------------------------------------- faithful preview + mask tiles
def test_alpha_preview_and_tiles_equal_the_export(tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=4)
    monkeypatch.setattr(app_module, "_proxy_scale", lambda: 0.5)      # a proxy smaller than the plate (as on a 4K shot)
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    f = frames[1]
    # EXR export (16-bit synthetic plates): the preview = INTER_AREA of the exported half alpha, rounded like the export
    p = _export_file(api, "obj", f)
    assert p.suffix == ".exr" and p.exists()
    exr, _ = _read_matte(p)
    pw, ph = 640, 360
    pv = _png(api.get(f"/api/object/obj/frame/{f}?view=alpha"))
    assert pv.shape == (ph, pw)
    ref = _u8(cv2.resize(exr, (pw, ph), interpolation=cv2.INTER_AREA))
    assert hashlib.sha1(pv.tobytes()).hexdigest() == hashlib.sha1(ref.tobytes()).hexdigest()
    assert pv.max() == 255 and (pv > 0).sum() > 1000                      # a real matte, not an empty frame
    # the full-resolution alpha tile == the exported file crop (bitwise after the export's 8-bit rounding)
    ys, xs = np.where(exr > 0.5)
    x, y = int(xs.min()) // 64 * 64, int(ys.min()) // 64 * 64
    w, h = min(512, shot.w - x), min(512, shot.h - y)
    t = _png(api.get(f"/api/object/obj/frame/{f}/tile?view=alpha&x={x}&y={y}&w={w}&h={h}"))
    assert t.shape == (h, w) and np.array_equal(t, _u8(exr[y:y + h, x:x + w])) and t.max() == 255
    # core / key tiles = crops of the stored masks; zone_alpha 404 until a zone preview exists
    core = cv2.imread(str(pipe.store.dir("obj") / "core" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(_png(api.get(f"/api/object/obj/frame/{f}/tile?view=core&x={x}&y={y}&w={w}&h={h}")), core[y:y + h, x:x + w])
    km = cv2.imread(str(pipe.store.dir("obj") / "keys" / f"k_{start:04d}_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(_png(api.get(f"/api/object/obj/frame/{start}/tile?view=key&x={x}&y={y}&w={w}&h={h}")), km[y:y + h, x:x + w])
    assert api.get(f"/api/object/obj/frame/{f}/tile?view=key&x={x}&y={y}&w={w}&h={h}").status_code == 404
    assert api.get(f"/api/object/obj/frame/{f}/tile?view=zone_alpha&x={x}&y={y}&w={w}&h={h}").status_code == 404
    assert api.get(f"/api/object/obj/frame/{f}/tile?view=nope&x={x}&y={y}&w={w}&h={h}").status_code == 400
    assert api.get(f"/api/object/obj/frame/{f}/tile?view=core&x={x}&y={y}&w=4096&h=64").status_code == 400
    cx, cy, s = shot.centers[f]
    z = {"mode": "add", "criterion": "matte", "frame": f, "polygon": _rect(cx - 60, cy - 260, cx + 60, cy - 120), "matting": "chroma"}
    assert api.post(f"/api/object/obj/zones/new?preview_frame={f}", json=z).status_code == 200
    za = cv2.imread(str(pipe.store.dir("obj") / "preview" / f"zalpha_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(_png(api.get(f"/api/object/obj/frame/{f}/tile?view=zone_alpha&x={x}&y={y}&w={w}&h={h}")), za[y:y + h, x:x + w])
    # PNG export: the alpha tile == the exported 8-bit PNG crop, bitwise (the zone invalidated the result -> finalize again)
    assert api.post("/api/object/obj/finalize", json={"edge_mode": "core"}).status_code == 200
    assert api.put("/api/project/export_format", json={"format": "png"}).status_code == 200
    p8 = _export_file(api, "obj", f)
    assert p8.suffix == ".png"
    png = cv2.imread(str(p8), cv2.IMREAD_GRAYSCALE)
    t8 = _png(api.get(f"/api/object/obj/frame/{f}/tile?view=alpha&x={x}&y={y}&w={w}&h={h}"))
    assert np.array_equal(t8, png[y:y + h, x:x + w])
    # a version without a result in the frame / unknown version
    assert api.get(f"/api/object/obj/frame/{f}/tile?view=alpha&x={x}&y={y}&w={w}&h={h}&v=v099").status_code == 404


def test_export_context_subtracts_the_front_object(tmp_path, monkeypatch):
    """`export=1`: the alpha the viewer shows in stage 5 is the exported matte – objects in front (with a final result)
    subtracted. The mock SAM3 keys the same puppet for `hand`, so `obj` behind it exports as (almost) nothing."""
    from engine.crop import choose_instance, run_crop_engine, save_crops
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(app_module, "_proxy_scale", lambda: 0.5)
    assert api.post("/api/object/obj/finalize", json={"edge_mode": "core"}).status_code == 200
    f = frames[0]
    raw = _png(api.get(f"/api/object/obj/frame/{f}?view=alpha"))
    assert np.array_equal(raw, _png(api.get(f"/api/object/obj/frame/{f}?view=alpha&export=1")))   # alone: export == own alpha
    # a front object with a final result
    pipe.store.create("hand", ["hand"], depth_order=0)
    pipe.store.update("obj", depth_order=1)
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="hand")
    pipe.store.save_key("hand", start, cands[0] > 0.5, author="auto")
    pipe.store.update("hand", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("hand") / "crops.json", res.segments)
    pipe.run_core("hand", frames)
    assert api.post("/api/object/hand/finalize", json={"edge_mode": "core"}).status_code == 200
    # the new front object marks obj "do przeliczenia" (7.3): until it is recomputed, obj cannot be exported -> export=1 == own
    assert pipe.store.get("obj")["tool"]["status"] == "do przeliczenia"
    assert np.array_equal(_png(api.get(f"/api/object/obj/frame/{f}?view=alpha&export=1")), raw)
    assert api.post("/api/object/obj/finalize", json={"edge_mode": "core"}).status_code == 200
    own = _png(api.get(f"/api/object/obj/frame/{f}?view=alpha"))
    ex = _png(api.get(f"/api/object/obj/frame/{f}?view=alpha&export=1"))
    assert own.max() == 255 and ex.sum() < own.sum() * 0.05, (own.sum(), ex.sum())
    # == the exported file of obj (export of both objects: hand in front)
    p = _export_file(api, "obj", f, objects=["hand", "obj"])
    exr, _ = _read_matte(p)
    assert np.array_equal(ex, _u8(cv2.resize(exr, (640, 360), interpolation=cv2.INTER_AREA)))
    ys, xs = np.where(cv2.imread(str(pipe.store.dir("obj") / "core" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127)
    x, y = min(int(xs.min()) // 64 * 64, shot.w - 256), min(int(ys.min()) // 64 * 64, shot.h - 256)
    t = _png(api.get(f"/api/object/obj/frame/{f}/tile?view=alpha&export=1&x={x}&y={y}&w=256&h=256"))
    assert np.array_equal(t, _u8(exr[y:y + 256, x:x + 256]))
    # another version than the active one is shown as is (no export context)
    runs = api.get("/api/object/obj/runs").json()["runs"]
    v = runs[0]["run"]
    assert v != pipe.store.get("obj")["tool"]["active_run"]
    # compared with its own alpha, not with the active one: v002 was finalised with `hand` in front, so the core
    # stabilisation keeps the frame's own values near it (pipeline.stabilized_core) and v002 may differ from v001
    old = _png(api.get(f"/api/object/obj/frame/{f}?view=alpha&v={v}"))
    assert old.max() == 255
    assert np.array_equal(_png(api.get(f"/api/object/obj/frame/{f}?view=alpha&export=1&v={v}")), old)


# ---------------------------------------------------------------------------------------------- correction loop: iteration note
def test_zone_key_iteration_note_api_and_cli(tmp_path, monkeypatch):
    import md
    from test_md_cli import TCHttp, _run
    from test_zone_matte import _api_project as _small_project
    app_module, api, shot, pipe, frames, start = _small_project(tmp_path, monkeypatch, n=4)
    cx, cy, s = shot.centers[start]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": _rect(cx - 60, cy - 260, cx + 60, cy - 120), "matting": "chroma",
         "prompt": "włosy"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    f2 = frames[-1]
    # a key with an iteration note: stored in the key, appended to prompt_history, returned
    k = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx, cy - 260, cx + 120, cy - 120), "prompt": " mniej tła przy uchu "})
    assert k.status_code == 200, k.text
    assert k.json()["key"]["prompt"] == "mniej tła przy uchu" and k.json()["zone"]["keys"][str(f2)]["prompt"] == "mniej tła przy uchu"
    assert k.json()["zone"]["prompt_history"] == ["mniej tła przy uchu"] and k.json()["zone"]["prompt"] == "włosy"
    g = api.get(f"/api/object/obj/zones/{zid}/keys").json()
    assert g["keys"][str(f2)]["prompt"] == "mniej tła przy uchu" and g["keys"][str(start)]["prompt"] == "" and g["prompt_history"] == ["mniej tła przy uchu"]
    # without `prompt` the note is kept; the same note again is not duplicated in the history; "" clears it
    k = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx, cy - 260, cx + 130, cy - 120)})
    assert k.json()["key"]["prompt"] == "mniej tła przy uchu" and k.json()["zone"]["prompt_history"] == ["mniej tła przy uchu"]
    k = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx, cy - 260, cx + 130, cy - 120), "prompt": "mniej tła przy uchu"})
    assert k.json()["zone"]["prompt_history"] == ["mniej tła przy uchu"]
    k = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx, cy - 260, cx + 130, cy - 120), "prompt": ""})
    assert k.json()["key"]["prompt"] == "" and "prompt" not in k.json()["zone"]["keys"][str(f2)]
    # the preview in that frame still follows the key (the note changes nothing in the drawing)
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{f2}").json()["preview"]["is_key"] is True
    # engine: key_zone carries the note in `prompt` (for the prompt rules of step 9); zone_keys stays drawing-only (cache hashes)
    from engine.zones import Zone, key_zone, set_zone_key, zone_key_prompt, zone_keys
    zz = Zone("z_09", "add", "matte", 10, polygon=_rect(0.0, 0.0, 10.0, 10.0), prompt="włosy")
    before = json.dumps(zone_keys(zz), sort_keys=True)
    set_zone_key(zz, 10, _rect(0, 0, 10, 10), prompt="bez linii")
    assert json.dumps(zone_keys(zz), sort_keys=True) == before and zone_key_prompt(zz, 10) == "bez linii"
    assert key_zone(zz, 10).prompt == "włosy; bez linii" and zz.prompt_history == ["bez linii"]
    set_zone_key(zz, 20, _rect(5, 5, 15, 15), prompt="tylko kosmyki")
    assert zone_key_prompt(zz, 20) == "tylko kosmyki" and key_zone(zz, 20).prompt == "włosy; tylko kosmyki" and zz.prompt_history == ["bez linii", "tylko kosmyki"]
    zz.prompt = ""
    assert key_zone(zz, 20).prompt == "tylko kosmyki"
    # CLI: zone-keys --set F --draw key.json --prompt "…"
    draw = tmp_path / "key.json"
    draw.write_text(json.dumps({"polygon": _rect(cx - 20, cy - 260, cx + 100, cy - 120)}), encoding="utf-8")
    code, out = _run(["zone-keys", "obj", zid, "--set", str(f2), "--draw", str(draw), "--prompt", "więcej włosów z lewej"], TCHttp(api))
    assert code == 0 and "więcej włosów z lewej" in out, out
    assert api.get(f"/api/object/obj/zones/{zid}/keys").json()["keys"][str(f2)]["prompt"] == "więcej włosów z lewej"
    # CLI `zones` shows the card status (state, keys, version stats) – paczka B
    code, out = _run(["zones", "obj"], TCHttp(api))
    assert code == 0 and "do przeliczenia" in out and "klucze 2" in out and f"F{start}, F{f2}" in out, out
