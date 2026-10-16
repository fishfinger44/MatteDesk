# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z9 (control views) + Z10 (SAM3 candidates on the canvas), synthetic, mock Comfy:
- `GET frame/{f}?view=flicker` = the version's flicker map (runs/v###/flicker_map.png) placed at the QA bbox, proxy size,
  BGRA with alpha = brightness; 404 without a version;
- `GET frame/{f}?view=cand&cand=i` = mask of candidate i (proxy size); `GET candidate/{i}` = BGRA thumbnail (<= 160 px)
  of the proxy crop with the mask as alpha; 404 for an unknown candidate / before detect."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _png(r, flags=cv2.IMREAD_UNCHANGED):
    assert r.status_code == 200, r.text
    m = cv2.imdecode(np.frombuffer(r.content, np.uint8), flags)
    assert m is not None
    return m


def test_candidates_and_flicker_views(tmp_path, monkeypatch):
    from test_zone_matte import _api_project
    from synth import display
    from engine.ingest import ProxyGenerator
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=4)
    # the mock SAM3 keys the puppet colour on the synthetic display (the proxy tonemap differs) – as in test_e2_acceptance
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": display(
        cv2.imread(str(ProxyGenerator.resolve_frame_path(shot.plate_pattern, f)), cv2.IMREAD_UNCHANGED)))
    pw, ph = shot.w, shot.h   # proxy == plate (1280 px)
    # Z10 – before detect: nothing
    assert api.get("/api/object/obj/candidate/0").status_code == 404
    assert api.get(f"/api/object/obj/frame/{start}?view=cand&cand=0").status_code == 404
    d = api.post("/api/object/obj/detect", json={"frame": start}).json()
    assert len(d["candidates"]) >= 1
    b = d["candidates"][0]
    m = _png(api.get(f"/api/object/obj/frame/{start}?view=cand&cand=0"))
    assert m.shape == (ph, pw) and m.ndim == 2 and (m > 127).sum() > 500
    ys, xs = np.where(m > 127)
    assert abs(xs.min() - b[0]) <= 1 and abs(ys.min() - b[1]) <= 1 and abs(xs.max() + 1 - b[2]) <= 1
    t = _png(api.get("/api/object/obj/candidate/0"))
    assert t.ndim == 3 and t.shape[2] == 4 and max(t.shape[:2]) <= 160 and t.shape[0] > 8 and t.shape[1] > 8
    assert (t[..., 3] == 255).any() and (t[..., 3] == 70).any()      # mask opaque, the rest of the crop dimmed
    big = _png(api.get("/api/object/obj/candidate/0?side=400"))
    assert max(big.shape[:2]) <= 400 and max(big.shape[:2]) > 160
    assert api.get("/api/object/obj/candidate/9").status_code == 404
    assert api.get(f"/api/object/obj/frame/{start}?view=cand&cand=9").status_code == 404
    # Z9 – flicker map: 404 before a version, BGRA proxy-size overlay after finalize, alpha only where the map is bright
    assert api.get(f"/api/object/obj/frame/{start}?view=flicker").status_code == 404
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    run = fz.json()["run"]
    rd = pipe.store.dir("obj") / "runs" / run
    assert (rd / "flicker_map.png").exists()
    qa = json.loads((rd / "qa.json").read_text(encoding="utf-8"))
    x0, y0, x1, y1 = qa["bbox"]
    fm = cv2.imread(str(rd / "flicker_map.png"), cv2.IMREAD_COLOR)
    assert fm.shape[:2] == (y1 - y0, x1 - x0)          # the map covers the QA bbox (plan: "sprawdź najpierw")
    fl = _png(api.get(f"/api/object/obj/frame/{start}?view=flicker"))
    assert fl.shape == (ph, pw, 4)
    a = fl[..., 3]
    assert a[y0:y1, x0:x1].max() > 0                   # inside the bbox: the map
    outside = np.ones((ph, pw), bool); outside[y0:y1, x0:x1] = False
    assert a[outside].max() == 0                       # outside: transparent
    full = np.zeros((ph, pw, 3), np.uint8); full[y0:y1, x0:x1] = fm
    assert np.array_equal(fl[..., :3], full) and np.array_equal(a, full.max(axis=2))
    assert api.get(f"/api/object/obj/frame/{start}?view=flicker&v={run}").status_code == 200
    assert api.get(f"/api/object/obj/frame/{start}?view=flicker&v=v099").status_code == 404
