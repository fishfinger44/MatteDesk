# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E5 – QA (SPEC 8), timeline flags, attempts counter, manual-path banner, zones Add / Remove (7.7) incl.
soft mode.

Acceptance (spec 12): Figure v5 metrics reproduced (flicker ~0.0034) – station; seams = 0; zone `Add + dark` on the
nest bottom reproduces v5 inside the zone (IoU >= 0.97); a `Remove` zone at the edge does not come back after
"Przelicz finalnie" (0 px alpha > 0.02 in the removed area); soft `bright` zone on a synthetic gradient gives the
alpha ramp of the setting (error < 0.02), no temporal flicker. Cloud: synthetic shots + mocks.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import choose_instance, run_crop_engine, save_crops
from engine.pipeline import ObjectPipeline
from engine.qa import interior_holes, run_qa, seam_count, small_islands
from engine.segment import iou
from engine.zones import Zone, apply_zones, criterion_key, save_zone
from mock_comfy import MockComfyClient, MockOutputs
from synth import dark_part_mask, display, make_shot, matanyone_handlers, sam3_handlers


class StatsClient(MockComfyClient):
    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}


def _setup(tmp_path, n=16, **shot_kw):
    shot = make_shot(tmp_path, n=n, **shot_kw)
    outs = MockOutputs(tmp_path / "comfy_out")
    client = StatsClient(handlers={**sam3_handlers(outs), **matanyone_handlers()})
    client.output_root = outs.root
    pipe = ObjectPipeline(tmp_path / "work" / "matte_desk", shot.plate_pattern, shot.w, shot.h, client,
                          display=display, sam3_ckpt="sam3.st", use_mplate=False)
    pipe.store.create("obj", ["felt puppet"])
    frames = list(range(shot.first, shot.last + 1))
    start = frames[len(frames) // 2]
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="felt puppet")
    pipe.store.save_key("obj", start, cands[0] > 0.5, author="auto")
    pipe.store.update("obj", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("obj") / "crops.json", res.segments)
    pipe.run_core("obj", frames)
    return shot, pipe, frames, start


def _zone_tracker(pipe, kind):
    """None = core bbox fallback (cloud default); "points" = point-tracker path (CoTracker3 on the station, LK here)."""
    if kind == "points":
        from engine.tracking import LKTracker
        pipe.zone_tracker = LKTracker()


@pytest.mark.parametrize("tracker", [None, "points"])
def test_zone_add_dark_recovers_nest_bottom(tmp_path, tracker):
    shot, pipe, frames, start = _setup(tmp_path, dark_part=True)
    _zone_tracker(pipe, tracker)
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60 * s, cy + 35 * s], [cx + 60 * s, cy + 35 * s], [cx + 60 * s, cy + 110 * s], [cx - 60 * s, cy + 110 * s]]
    save_zone(pipe.store.dir("obj"), Zone("z_01", "add", "dark", start, polygon=poly, param=0.15))
    full, rep, run_dir = pipe.finalize("obj", frames, edge_mode="matanyone")
    for f in frames:
        cx, cy, s = shot.centers[f]
        zone = dark_part_mask(shot.w, shot.h, cx, cy, s) > 0.01
        zone = cv2.dilate(zone.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        assert iou((full[f] > 0.5) & zone, (shot.masks[f] > 0.5) & zone) >= 0.97, f
    # floating islands never added: background dark pixels elsewhere are not connected
    qa = json.loads((run_dir / "qa.json").read_text())
    assert all(v == 0 for v in qa["islands"].values())


@pytest.mark.parametrize("tracker", [None, "points"])
def test_zone_remove_never_comes_back(tmp_path, tracker):
    shot, pipe, frames, start = _setup(tmp_path)
    _zone_tracker(pipe, tracker)
    cx, cy, s = shot.centers[start]
    head = (cx, cy - 75 * s)
    # remove the right antenna (zone follows the object)
    poly = [[head[0] + 8 * s, head[1] - 80 * s], [head[0] + 60 * s, head[1] - 80 * s],
            [head[0] + 60 * s, head[1] - 20 * s], [head[0] + 8 * s, head[1] - 20 * s]]
    save_zone(pipe.store.dir("obj"), Zone("z_01", "remove", "all", start, polygon=poly))
    full, rep, run_dir = pipe.finalize("obj", frames, edge_mode="matanyone")
    # the same core the finalize used (core_stabilize is on by default since 2026-10-14)
    from engine.pipeline import core_stabilize_on
    core = pipe.load_core("obj", frames)
    if core_stabilize_on(pipe.store.get("obj")):
        core, _ = pipe.stabilized_core("obj", frames, core, pipe.keys("obj"))
    zres = pipe.apply_zones("obj", frames, core, pipe.keys("obj"))
    for f in frames:
        removed = zres.removed[f]
        assert removed.any()
        assert int((full[f][removed] > 0.02).sum()) == 0, f


def test_soft_bright_zone_ramp_on_gradient():
    h, w, n = 60, 200, 8
    grad = np.tile(np.linspace(0.2, 0.8, w, dtype=np.float32), (h, 1))
    img = np.dstack([grad, grad, grad])
    core = {f: np.zeros((h, w), bool) for f in range(n)}
    for f in core:
        core[f][:, :] = True  # object everywhere -> soft remove of the bright part = smoke-like transparency
    z = Zone("z_01", "remove", "bright", 0, polygon=[[0, 0], [w, 0], [w, h], [0, h]], param=0.5, soft=True,
             softness=0.1, static=True)
    res = apply_zones(range(n), core, lambda f: img, [z], {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in range(n)}})
    expected = 1.0 - np.clip((grad - 0.4) / 0.2, 0, 1)   # remove bright: alpha falls from 1 to 0 across 0.4..0.6
    for f in range(n):
        a, area = res.soft[f]
        assert np.abs(a[area] - expected[area]).max() < 0.02
    stack = np.stack([res.soft[f][0] for f in range(n)])
    assert stack.std(axis=0).max() < 1e-6  # no flicker in time


def test_criteria_hard():
    img = np.zeros((4, 4, 3), np.float32)
    img[:2] = 0.05
    img[2:] = 0.9
    assert criterion_key(img, Zone("z", "add", "dark", 0, param=0.15))[:2].all()
    assert criterion_key(img, Zone("z", "add", "bright", 0, param=0.5))[2:].all()


def test_qa_metrics_and_banner():
    h, w = 100, 120
    a = np.zeros((h, w), np.float32)
    a[20:80, 30:90] = 1
    holed = a.copy()
    holed[45:55, 55:65] = 0
    assert interior_holes(holed) == 1 and interior_holes(a) == 0
    isl = a.copy()
    isl[5:10, 5:10] = 1
    assert small_islands(isl) == 1
    # seams: two objects split with a thin gap of 0.8 inside the union
    u = np.zeros((20, 20), np.float32)
    u[:, :] = 1.0
    u[5:15, 10] = 0.8
    assert seam_count(u) > 0
    u2 = np.ones((20, 20), np.float32)
    assert seam_count(u2) == 0
    # flicker: stable sequence -> 0; banner after 2 failed attempts
    al = {f: (a if f % 2 else holed) for f in range(10)}
    rep = run_qa(list(range(10)), lambda f: al[f], None, attempts=2)
    assert not rep.gate_ok and rep.banner.startswith("Automat nie daje rady") and rep.suggested_keys
    assert any(c == "red" for c in rep.colours.values())
    stable = run_qa(list(range(10)), lambda f: a, None)
    assert stable.flicker_median == 0.0 and stable.gate_ok


def test_api_zones_and_qa(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    shot, pipe, frames, start = _setup(tmp_path, n=10)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    monkeypatch.setitem(app_module.STATE, "comfy_client", pipe.client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    # the core above was computed with use_mplate=False: the project setting must match (else the core is stale)
    assert api.put("/api/project/mask_plate_denoise", json={"setting": "off"}).status_code == 200
    cx, cy, s = shot.centers[start]
    z = {"mode": "remove", "criterion": "all", "frame": start,
         "polygon": [[cx + 20, cy - 200], [cx + 120, cy - 200], [cx + 120, cy - 100], [cx + 20, cy - 100]]}
    r = api.post("/api/object/obj/zones/new?preview_frame=%d" % start, json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    assert r.json()["preview"]["removed_px"] >= 0
    # the preview is drawable in the stage-4 viewer (user 2026-10-13: it existed only as pixel counts)
    for view in ("zone_add", "zone_remove"):
        pr = api.get(f"/api/object/obj/frame/{start}?view={view}")
        assert pr.status_code == 200 and pr.headers["content-type"] == "image/png", view
    assert api.get(f"/api/object/obj/frame/{start + 1}?view=zone_add").status_code == 404   # no preview there
    assert [x["zone_id"] for x in api.get("/api/object/obj/zones").json()["zones"]] == [zid]
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "matanyone"})
    assert fz.status_code == 200, fz.text
    qa = api.get("/api/object/obj/qa").json()
    assert "flicker_median" in qa and len(qa["colours"]) == len(frames)
    assert api.delete(f"/api/object/obj/zones/{zid}").status_code == 200


REAL_V5 = Path("C:\\shots\\SH010\\frame_####.exr")
REAL_X1 = Path("C:\\shots\\SH010\\frame_####.exr")


def test_e5_real_v5_flicker_metric():
    """Station (D:/MD_REAL_PROJECT): QA on the accepted Figure v5 reproduces its flicker metric (~0.0034, gate <= 0.004)."""
    assert REAL_V5.exists() and REAL_X1.exists()
    from engine.exr_io import read_exr, read_rgb
    from engine.ingest import plate_to_display_float
    a_files = sorted(p for p in os.listdir(REAL_V5) if p.lower().endswith(".exr"))
    x_files = sorted(p for p in os.listdir(REAL_X1) if p.lower().endswith(".exr"))
    frames = list(range(len(a_files)))
    la = lambda i: (lambda ch: ch.get("A", ch.get("R")))(read_exr(REAL_V5 / a_files[i])[0])
    lg = lambda i: (plate_to_display_float(read_rgb(REAL_X1 / x_files[i])).mean(axis=2) * 255).astype(np.float32)
    rep = run_qa(frames, la, lg)
    print("v5 flicker median", rep.flicker_median, "holes", sum(rep.holes.values()), "islands", sum(rep.islands.values()))
    assert abs(rep.flicker_median - 0.0034) < 0.0008 and rep.flicker_median <= 0.004


def test_qa_flow_consistency_and_cleanplate_yardstick():
    h, w = 240, 320
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2) * 0.4 + 0.1
    obj_tex = cv2.GaussianBlur(rng.random((80, 80)).astype(np.float32), (0, 0), 2) * 0.3 + 0.65
    frames = list(range(6))
    al, gray = {}, {}
    for f in frames:
        a = np.zeros((h, w), np.float32)
        a[80:160, 60 + 4 * f:140 + 4 * f] = 1
        g = bg.copy()
        g[80:160, 60 + 4 * f:140 + 4 * f] = obj_tex
        al[f], gray[f] = a, g * 255
    ok = run_qa(frames, lambda f: al[f], lambda f: gray[f], flow_check=True, load_clean_gray=lambda f: bg * 255)
    assert ok.edge["flow_iou_min"] > 0.95 and ok.edge["cleanplate_agreement_median"] > 0.8
    bad = dict(al)
    bad[3] = np.roll(al[3], 20, axis=0)  # matte jumps while the plate does not
    rep = run_qa(frames, lambda f: bad[f], lambda f: gray[f], flow_check=True, load_clean_gray=lambda f: bg * 255)
    assert any("przepływie" in r for _, r in rep.flags[3]) and any("clear plate" in r for _, r in rep.flags[3])
