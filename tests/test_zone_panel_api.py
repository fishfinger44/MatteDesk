# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 5 of docs/PLAN_HAIR_TOOL.md (synthetic, mock Comfy): the API the stage-4 zone panel runs on – zone creation
with preview, card status (GET /zones -> status), PATCH (prompt / preset / engine / enabled / holdout), the preview
endpoint following the zone KEY of the frame, region in a frame without a key, "Zastosuj" -> version with zone
stats in GET /runs, 409 missing_front -> "Licz mimo to bez holdoutu" (PATCH holdout=false) -> apply."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _mean_x(api, frame, view="zone_alpha"):
    r = api.get(f"/api/object/obj/frame/{frame}?view={view}")
    assert r.status_code == 200, r.text
    m = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 5)
    return float(xs.mean()) if len(xs) else None


def test_zone_panel_flow(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=8)
    cx, cy, s = shot.centers[start]
    poly = _rect(cx - 60, cy - 260, cx + 60, cy - 120)
    # 1. first stroke of the panel = a new zone with the current preset, preview of that frame at once
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "chroma", "preset": "hair"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    pv = r.json()["preview"]
    assert pv["is_key"] is True and pv["frame"] == start and pv["engine_used"] == "chroma" and "note" not in pv
    # 2. card status: one key, name = id (no prompt), "do przeliczenia" (no version yet)
    d = api.get("/api/object/obj/zones").json()
    st = d["status"][zid]
    assert st["state"] == "do przeliczenia" and st["keys"] == [start] and st["n_keys"] == 1 and st["name"] == zid and st["run_stats"] is None
    # 3. PATCH from the card / panel: prompt (history), preset, engine; validation
    p = api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy bez złotej linii", "preset": "fibres"})
    assert p.status_code == 200 and p.json()["zone"]["preset"] == "fibres" and p.json()["changed"] == ["preset", "prompt"], p.text
    p = api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy"})
    assert p.json()["zone"]["prompt_history"] == ["włosy bez złotej linii"] and p.json()["status"]["name"] == "włosy"
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"preset": "vitmatte"}).status_code == 400
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"matting": "nope"}).status_code == 400
    assert api.patch(f"/api/object/obj/zones/{zid}", json={}).status_code == 400
    assert api.patch("/api/object/obj/zones/z_99", json={"enabled": False}).status_code == 404
    # 4. default engine `matanyone`: the automatic preview is the key hint (note), nothing written to the zone
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"matting": "matanyone"}).status_code == 200
    pv = api.post(f"/api/object/obj/zones/{zid}/preview/{start}")
    assert pv.status_code == 200 and "note" in pv.json()["preview"] and pv.json()["preview"]["engine_used"] == "chroma", pv.text
    assert api.get("/api/object/obj/zones").json()["zones"][0]["matting"] == "matanyone"
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"matting": "chroma"}).status_code == 200
    # a frame without a core / an unknown zone
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start + 500}").status_code == 404
    assert api.post(f"/api/object/obj/zones/z_99/preview/{start}").status_code == 404
    # 5. painting in another frame = a key there; the preview of that frame follows ITS key (region shifted right)
    f2 = frames[-1]
    x_base = _mean_x(api, start)
    k = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx + 100, cy - 260, cx + 220, cy - 120)})
    assert k.status_code == 200 and k.json()["keys"] == sorted({start, f2}), k.text
    pv2 = api.post(f"/api/object/obj/zones/{zid}/preview/{f2}").json()["preview"]
    assert pv2["is_key"] is True
    x_key = _mean_x(api, f2, "zone_trimap")
    assert x_key is not None and x_base is not None and x_key > x_base + 20, (x_key, x_base)   # proxy px, key +160 plate px
    # a frame between the keys has no key: the preview says so (region = base drawing there, plan step 5 note)
    mid = next(f for f in frames if f not in (start, f2))
    pv3 = api.post(f"/api/object/obj/zones/{zid}/preview/{mid}").json()["preview"]
    assert pv3["is_key"] is False
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{mid}").json()
    assert reg["is_key"] is False and reg["area_px"] > 0 and reg["bbox"] is not None and len(reg["contours"]) >= 1
    assert api.get("/api/object/obj/zones").json()["status"][zid]["keys"] == sorted({start, f2})
    # 6. "Zastosuj": finalize once (version with alpha_base), apply -> new version, GET /runs carries the zone stats
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    v1 = fz.json()["run"]
    assert api.get("/api/object/obj/zones").json()["status"][zid]["state"] == "policzona"
    api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy, kosmyki"})     # a change -> do przeliczenia
    assert api.get("/api/object/obj/zones").json()["status"][zid]["state"] == "do przeliczenia"
    ap = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert ap.status_code == 200, ap.text
    v2 = ap.json()["run"]
    runs = api.get("/api/object/obj/runs").json()
    r2 = next(r for r in runs["runs"] if r["run"] == v2)
    assert r2["applied_from"] == v1 and r2["applied_zone"] == zid and zid in r2["zones"]
    zs = r2["zones"][zid]
    assert zs["frames"] == len(frames) and zs["engine"] == "chroma" and zs["s_per_frame"] is not None and "engine_share" in zs
    st = api.get("/api/object/obj/zones").json()["status"][zid]
    assert st["state"] == "policzona" and st["run"] == v2 and st["run_stats"]["frames"] == len(frames)
    # enable / disable from the card
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"enabled": False}).json()["status"]["state"] == "wyłączona"
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"enabled": True}).json()["status"]["state"] == "do przeliczenia"
    # 7. an object in front without a mask -> apply 409 missing_front -> "Licz mimo to bez holdoutu" -> apply ok
    pipe.store.create("hand", ["hand"], depth_order=0)
    pipe.store.update("obj", depth_order=1)
    r = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert r.status_code == 409 and r.json()["detail"]["missing_front"] == ["hand"], r.text
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 409
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"holdout": False}).json()["zone"]["holdout"] is False
    r = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert r.status_code == 200, r.text
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 200
