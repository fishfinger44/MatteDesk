# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E3 – objects, SAM3 core on crop segments (tracker without text), "Przelicz finalnie" = MatAnyone2 in
the edge band (SPEC 5.4, 7.3a). Cloud: synthetic puppet + colour-key SAM3 / soft-key MatAnyone in the mock.

Acceptance (spec 12): IoU with the reference >= 0.95 (Figure v5 outside the nest-bottom zone – station test);
one-frame preview (click -> result) < 3 s; VRAM and ma_scale in the log.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import choose_instance, run_crop_engine, save_crops
from engine.pipeline import ObjectPipeline
from engine.refine import temporal_median
from engine.segment import iou
from mock_comfy import MockComfyClient, MockOutputs
from synth import display, make_shot, matanyone_handlers, sam3_handlers


def fp(p, f):
    return p.replace("####", f"{f:04d}")


class StatsClient(MockComfyClient):
    def __init__(self, vram_gb, **kw):
        super().__init__(**kw)
        self.vram_gb = vram_gb

    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": int((12 - self.vram_gb) * 2 ** 30)}]}


def _setup(tmp_path, n=30, vram_gb=6.0):
    shot = make_shot(tmp_path, n=n)
    outs = MockOutputs(tmp_path / "comfy_out")
    client = StatsClient(vram_gb, handlers={**sam3_handlers(outs), **matanyone_handlers()})
    client.output_root = outs.root
    work = tmp_path / "work" / "matte_desk"
    pipe = ObjectPipeline(work, shot.plate_pattern, shot.w, shot.h, client, display=display, sam3_ckpt="sam3.st", use_mplate=False)
    pipe.store.create("puppet", ["felt puppet with two long antennae"], flags=["thin_parts"])
    frames = list(range(shot.first, shot.last + 1))
    start = shot.first + n // 2
    full = lambda f: pipe.loader.frame(f)
    cands = choose_instance(pipe.runner, full(start), prompt="felt puppet")
    pipe.store.save_key("puppet", start, cands[0] > 0.5, author="auto")
    pipe.store.update("puppet", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, full, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("puppet") / "crops.json", res.segments)
    return shot, pipe, client, frames


def test_e3_acceptance_core_and_final(tmp_path):
    shot, pipe, client, frames = _setup(tmp_path)
    core = pipe.run_core("puppet", frames)
    full, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")

    ious_core = [iou(core.core[f], shot.masks[f] > 0.5) for f in frames]
    ious = [iou(full[f] > 0.5, shot.masks[f] > 0.5) for f in frames]
    print("core IoU min", min(ious_core), "final IoU min", min(ious), "mean", np.mean(ious))
    assert min(ious) >= 0.95, ious

    # tracker graphs never carry text; MatAnyone graph contract checked in the mock (<= 3 keys, mem 5)
    for g in client.sent:
        if any(v["class_type"] == "SAM3_VideoTrack" for v in g.values()):
            assert not any(v["class_type"] == "CLIPTextEncode" for v in g.values())

    # outside the edge band the result is exactly the core (MatAnyone cannot add/remove regions)
    from engine.refine import edge_band
    f = frames[5]
    band = edge_band(core.core[f], rep.band_px + 1)
    assert np.array_equal(full[f][~band] > 0.5, core.core[f][~band])

    # ma_scale + VRAM in the log, run stored as a version
    run = json.loads((run_dir / "run.json").read_text())
    assert run["ma_scale"] == 1.0 and run["vram_log"] and "VRAM" in run["vram_log"][0]
    # v001, or v002 when the automatic edge-position correction (7.8 pkt 1, E5b) re-ran it
    assert (run_dir / "alpha").is_dir() and run_dir.name in ("v001", "v002")
    assert pipe.store.get("puppet")["tool"]["active_run"] == run_dir.name


def test_ma_scale_steps_down_when_vram_high(tmp_path):
    shot, pipe, client, frames = _setup(tmp_path, n=12, vram_gb=11.5)
    pipe.run_core("puppet", frames)
    full, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")
    assert rep.ma_scale == 0.5 and len(rep.vram_log) == 3
    assert min(iou(full[f] > 0.5, shot.masks[f] > 0.5) for f in frames) >= 0.9


def test_temporal_median_keeps_motion_but_kills_flicker():
    a = np.zeros((9, 1, 3), np.float32)
    a[:, 0, 0] = [0.5, 0.56, 0.44, 0.58, 0.45, 0.55, 0.43, 0.5, 0.5]  # edge flicker (small)
    a[:, 0, 1] = [0, 0, 0, 0, 1, 1, 1, 1, 1]                          # real motion edge
    a[:, 0, 2] = [0, 0, 0, 0, 1, 0, 0, 0, 0]                          # thin part passing in one frame
    out = temporal_median(a, 2, 0.35)
    assert np.std(out[2:7, 0, 0]) < 0.6 * np.std(a[2:7, 0, 0])
    assert np.allclose(out[:, 0, 1], a[:, 0, 1])
    assert out[4, 0, 2] == 1.0  # not erased (no ghosts / no lost antennae)


def test_preview_frame_api_under_3s(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module

    shot, pipe, client, frames = _setup(tmp_path, n=12)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    t0 = time.time()
    r = api.post("/api/object/puppet/preview_frame", json={"frame": shot.first + 3, "prompt": "felt puppet"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert time.time() - t0 < 3.0 and d["seconds"] < 3.0
    m = cv2.imread(str(Path(d["mask_path"])), cv2.IMREAD_GRAYSCALE) > 127
    assert iou(m, shot.masks[shot.first + 3] > 0.5) > 0.85


def test_api_run_finalize_and_frame_views(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module

    shot, pipe, client, frames = _setup(tmp_path, n=14)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": display(
        cv2.imread(fp(shot.plate_pattern, f), cv2.IMREAD_UNCHANGED)))
    r = api.post("/api/object/puppet/run")
    assert r.status_code == 200, r.text
    r2 = api.post("/api/object/puppet/run")  # nothing changed -> everything from cache
    assert r2.json()["recomputed"] == [] and r2.json()["cached"]
    fz = api.post("/api/object/puppet/finalize", json={"edge_mode": "matanyone"})
    assert fz.status_code == 200, fz.text
    for view in ("core", "alpha", "overlay"):
        im = api.get(f"/api/object/puppet/frame/{shot.first + 4}?view={view}")
        assert im.status_code == 200 and im.headers["content-type"] == "image/png"


REAL_WORK = Path("C:\\shots\\SH010\\frame_####.exr")


@pytest.mark.skip(reason="E3 wyrzucony 2026-10-13 (decyzja uzytkownika): stary silnik; kryterium SH010 = test realny strefy matting lokalny, docs/PLAN_HAIR_TOOL.md sekcja 10")
def test_e3_real_figure_iou_vs_v5():
    """Station (D:/MD_REAL_PROJECT): after "Licz ujęcie" + "Przelicz finalnie" of object `figure` in the UI (plate display input),
    IoU with render/comfy_mattes/figure_alpha_v5 >= 0.95 outside the nest-bottom zone (PNG mask
    matte_desk/qa/nest_zone.png, white = excluded; the zone arrives in E5)."""
    # v5 = Figure + nest. User decision 2026-10-12: two objects (figure in front, nest behind) -> compare their union.
    from engine.exr_io import read_exr
    from engine.export import run_alpha_loader
    # default: figure + objects that are its parts (part_of = figure, e.g. the dress / "nest")
    objs_dir = REAL_WORK / "matte_desk" / "objects"
    parts = [d.name for d in objs_dir.iterdir() if (d / "object.json").exists()
             and json.loads((d / "object.json").read_text(encoding="utf-8")).get("part_of") == "figure"] if objs_dir.exists() else []
    names = os.environ.get("MATTE_DESK_E3_OBJECTS", ",".join(["figure"] + parts)).split(",")
    v5 = REAL_WORK / "render" / "comfy_mattes" / "figure_alpha_v5"
    proj = json.loads((REAL_WORK / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    pw, ph = int(proj["plate_w"]), int(proj["plate_h"])
    loaders, vram = [], []
    for n in names:
        od = REAL_WORK / "matte_desk" / "objects" / n
        assert od.exists() and v5.exists(), f"compute object '{n}' in the UI first (objects: {names})"
        obj = json.loads((od / "object.json").read_text(encoding="utf-8"))
        run = obj["tool"]["active_run"]
        assert run, f"'{n}': no final result yet ('Przelicz finalnie')"
        vram += json.loads((od / "runs" / run / "run.json").read_text(encoding="utf-8")).get("vram_log", [])
        loaders.append(run_alpha_loader(od, run, pw, ph))
    zone_p = REAL_WORK / "matte_desk" / "qa" / "nest_zone.png"
    zone = cv2.imread(str(zone_p), cv2.IMREAD_GRAYSCALE) > 127 if zone_p.exists() else None
    files = sorted(p for p in os.listdir(v5) if p.lower().endswith(".exr"))
    ious = []
    for p in files:
        f = int(p.rsplit("_", 1)[-1].split(".")[0])
        ch, _ = read_exr(v5 / p)
        ref = ch.get("A", ch.get("R")) > 0.5
        from engine.export import seamless_union
        cores = []
        for n in names:
            cp = REAL_WORK / "matte_desk" / "objects" / n / "core" / f"c_{f:04d}.png"
            cores.append(cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE) > 127 if cp.exists() else None)
        ours = seamless_union([load(f) for load in loaders], cores) > 0.5   # same whole as the export
        if zone is not None:
            ref, ours = ref & ~zone, ours & ~zone
        ious.append(iou(ours, ref))
    print("objects", names, "IoU min", min(ious), "mean", np.mean(ious), "nest zone", zone is not None, "VRAM", vram)
    assert zone is not None, "paint matte_desk/qa/nest_zone.png (nest bottom) to finish the E3 criterion"
    assert min(ious) >= 0.95
