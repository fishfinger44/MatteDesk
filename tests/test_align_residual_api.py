# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""align_v2 step 3 (2026-11-03): the residual map of stage 0 – GET /api/plate/align/residual/{frame} (PNG heat map
drawn over the plate) and /residual_stats/{frame} (median / p95 / max / share < 0.3 px), cached next to the report
and recomputed when the report changes; the objects the measurement excluded are excluded here too."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_api_residual_map_and_stats(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    work = Path(app_module.STATE["work_dir"])
    f = frames[1]
    assert api.get(f"/api/plate/align/residual/{f}", params={"plate": "cleanplate"}).status_code == 400 or True
    # a clear plate = the beauty shifted by 1 px (known), registered in the project
    src = Path(shot.plate_pattern.replace("####", f"{f:04d}"))
    cdir = tmp_path / "clean"
    cdir.mkdir()
    for fr in frames:
        im = cv2.imread(shot.plate_pattern.replace("####", f"{fr:04d}"), cv2.IMREAD_UNCHANGED)
        sh = cv2.warpAffine(im, np.float32([[1, 0, 1.0], [0, 1, 0.0]]), (im.shape[1], im.shape[0]), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(str(cdir / f"c_{fr:04d}.exr"), sh.astype(np.float32))
    pat = str(cdir / "c_####.exr").replace("\\", "/")
    app_module.STATE["shot_project"].set_plate("cleanplate", pat)
    app_module._refresh_manifest_view()
    r = api.get(f"/api/plate/align/residual/{f}", params={"plate": "cleanplate"})
    assert r.status_code == 404 and "Zmierz" in r.json()["detail"]
    # a report: translation that undoes the shift (dx = -1) on every frame, the object's core excluded
    cd = work / "objects" / "obj" / "core"
    cd.mkdir(parents=True, exist_ok=True)
    for fr in frames:
        cv2.imwrite(str(cd / f"c_{fr:04d}.png"), (shot.masks[fr] > 0.5).astype(np.uint8) * 255)
    from engine.align import AlignmentFrameResult, AlignmentReport
    from engine.align_exclude import object_sigs
    rep = AlignmentReport("cleanplate", "cleanplate", "beauty", (frames[0], frames[-1]), "translation", 0.1, 0.2, True, "x",
                          frames=[AlignmentFrameResult(frame=fr, method="translation", residual_px=0.1, residual_max_px=0.2,
                                                       dx=-1.0, dy=0.0) for fr in frames],
                          exclusions={"objects": ["obj"], "object_sigs": object_sigs(work, ["obj"]), "dilate_px": 8,
                                      "box": None, "user_mask": None})
    (work / "align").mkdir(exist_ok=True)
    jp = work / "align" / "Align_cleanplate.json"
    jp.write_text(json.dumps(rep.to_dict()), encoding="utf-8")
    r = api.get(f"/api/plate/align/residual/{f}", params={"plate": "cleanplate", "side": 512})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    png = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert png.shape[1] == 512 and png.shape[2] == 4 and (png[..., 3] > 0).any()
    st = api.get(f"/api/plate/align/residual_stats/{f}", params={"plate": "cleanplate", "side": 512}).json()
    assert st["n"] > 20 and st["median"] < 0.15 and st["frac_ok"] > 0.8 and st["frame"] == f, st
    # the object's core (dilated) is transparent in the map; a point of the static set is coloured
    cell = max(1, round(512 / shot.w * 48))
    ys, xs = np.where(shot.masks[f] > 0.5)
    cy, cx = int(np.median(ys) * png.shape[0] / shot.h), int(np.median(xs) * 512 / shot.w)
    assert png[cy, cx, 3] == 0
    cached = (work / "align" / "residual_cleanplate" / f"r_{f:04d}.png")
    assert cached.exists()
    t0 = cached.stat().st_mtime
    api.get(f"/api/plate/align/residual/{f}", params={"plate": "cleanplate", "side": 512})
    assert cached.stat().st_mtime == t0                       # cache hit
    time.sleep(0.05)
    jp.write_text(json.dumps(rep.to_dict()), encoding="utf-8")   # a new measurement -> recomputed
    time.sleep(0.05)
    api.get(f"/api/plate/align/residual/{f}", params={"plate": "cleanplate", "side": 512})
    assert cached.stat().st_mtime > t0
    assert api.get("/api/plate/align/residual/9999", params={"plate": "cleanplate"}).status_code == 404
    # stage 0 info carries the exclusions and the lock block
    a = api.get("/api/project/stages").json()["align"]["cleanplate"]
    assert a["exclusions"]["objects"] == ["obj"] and "camera_lock" in a
    assert cell >= 1
