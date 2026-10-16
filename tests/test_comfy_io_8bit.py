# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comfy input routing by source bit depth (decision 2026-10-13, engine/comfy_io.py).

- 8-bit sources (MP4 / PNG / JPG, sRGB): every image to Comfy = 8-bit PNG (own values, no colour transform) loaded with
  VHS_LoadImagesPath (+ ImageToMask for masks), one folder per job; SAM3 track frames > SAM_RES downscaled, masks back.
- EXR sources: float32 EXR + PlateLoadEXR / PlateLoadMaskSeq exactly as before.
- 'Przerwij' works inside the per-frame write loops (JobCancelled, nothing sent to Comfy).
- Mask export format: project.json export_format auto / png / exr.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import engine  # noqa: F401  (Unicode-safe cv2)
from engine.comfy_graph import find_nodes
from engine.comfy_io import LazySeq, fit_sam_res, io_mode_for
from engine.jobs import JOBS, JobCancelled
from engine.sam3 import SAM_RES, Sam3Runner, track_graph
from mock_comfy import MockComfyClient, MockOutputs
from synth import display, load_node_images, load_node_masks, make_shot, matanyone_handlers, sam3_handlers


def _classes(g):
    return sorted(v["class_type"] for v in g.values())


def _track_runner(tmp_path, io_mode, seen):
    outs = MockOutputs(tmp_path / "out")

    def track(graph):
        (_, tr), = find_nodes(graph, "SAM3_VideoTrack")
        imgs = load_node_images(graph, tr["inputs"]["images"][0])
        init = load_node_masks(graph, tr["inputs"]["initial_mask"][0])
        seen.update(graph=graph, shape=imgs[0].shape, n=len(imgs), init_shape=init[0].shape)
        return outs.save_masks("8", graph["8"]["inputs"]["filename_prefix"],
                               [(im[..., 0] > 0.5).astype(np.float32) for im in imgs])

    client = MockComfyClient({"SAM3_VideoTrack": track})
    client.output_root = outs.root
    return Sam3Runner(client, tmp_path / "comfy_io", ckpt="x", io_mode=io_mode), client


def _square_frames(side, n=3):
    frames = []
    for i in range(n):
        f = np.zeros((side, side, 3), np.float32)
        f[int(side * 0.2):int(side * 0.78), int(side * 0.31) + 10 * i:int(side * 0.68)] = 1.0
        frames.append(f)
    return frames


def test_io_mode_from_manifest():
    assert io_mode_for({"bit_depth": 8, "source_type": "img_seq"}) == "png8"
    assert io_mode_for({"bit_depth": 8, "source_type": "video"}) == "png8"
    assert io_mode_for({"bit_depth": 16, "source_type": "exr_seq"}) == "exr"
    assert io_mode_for({"bit_depth": 32}) == "exr"
    assert io_mode_for({"source_type": "video"}) == "png8"
    assert io_mode_for({}) == "exr" and io_mode_for(None) == "exr"
    with pytest.raises(ValueError):
        Sam3Runner(MockComfyClient(), Path("."), ckpt="x", io_mode="half")


def test_png8_track_writes_png_uses_vhs_downscales_and_restores(tmp_path):
    seen = {}
    frames = _square_frames(1920)
    init = (frames[0][..., 0] > 0.5).astype(np.float32)
    runner, client = _track_runner(tmp_path, "png8", seen)
    masks = runner.track(frames, init)
    g = seen["graph"]
    assert "PlateLoadEXR" not in _classes(g) and "PlateLoadMaskSeq" not in _classes(g)
    assert _classes(g).count("VHS_LoadImagesPath") == 2 and "ImageToMask" in _classes(g)
    assert g["5"]["inputs"]["images"] == ["2", 0] and g["5"]["inputs"]["initial_mask"] == ["4", 0]
    assert g["4"]["inputs"]["channel"] == "red" and g["5"]["inputs"]["max_objects"] == 1
    # one folder per job: frames only (the loader reads the whole folder), the key mask in its own folder
    fdir, kdir = Path(g["2"]["inputs"]["directory"]), Path(g["3"]["inputs"]["directory"])
    assert fdir != kdir and sorted(p.name for p in fdir.iterdir()) == ["f_0000.png", "f_0001.png", "f_0002.png"]
    assert [p.name for p in kdir.iterdir()] == ["k_0000.png"]
    assert not list((tmp_path / "comfy_io").rglob("*.exr"))
    f0 = cv2.imread(str(fdir / "f_0000.png"), cv2.IMREAD_UNCHANGED)
    assert f0.dtype == np.uint8 and f0.shape == (SAM_RES, SAM_RES, 3)            # downscaled to SAM_RES
    assert seen["init_shape"] == (SAM_RES, SAM_RES) and seen["n"] == 3
    # masks back at the input size, object where it was
    assert len(masks) == 3 and all(m.shape == (1920, 1920) for m in masks)
    ys, xs = np.nonzero(masks[2] > 0.5)
    assert abs(ys.min() - 384) <= 3 and abs(ys.max() - 1496) <= 3
    assert abs(xs.min() - (595 + 20)) <= 3 and abs(xs.max() - 1304) <= 3


def test_png8_values_unchanged_and_small_frames_not_resized(tmp_path):
    seen = {}
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, (600, 640, 3), dtype=np.uint8)
    f = u8.astype(np.float32) / 255.0                                            # what the loaders give (8-bit / 255)
    runner, _ = _track_runner(tmp_path, "png8", seen)
    masks = runner.track([f, f], (f[..., 0] > 0.5).astype(np.float32))
    fdir = Path(seen["graph"]["2"]["inputs"]["directory"])
    back = cv2.imread(str(fdir / "f_0001.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(back, u8)                                              # sRGB values as they are, no transform
    assert seen["shape"][:2] == (600, 640) and masks[0].shape == (600, 640)


def test_exr_track_graph_and_files_unchanged(tmp_path):
    """EXR sources: float32 EXR frames at full size + PlateLoadEXR / PlateLoadMaskSeq, the pre-2026-10-13 graph."""
    seen = {}
    frames = _square_frames(1920, n=2)
    runner, _ = _track_runner(tmp_path, "exr", seen)
    assert Sam3Runner(MockComfyClient(), tmp_path, ckpt="x").io_mode == "exr"   # default
    masks = runner.track(frames, (frames[0][..., 0] > 0.5).astype(np.float32))
    g = seen["graph"]
    job = Path(g["2"]["inputs"]["path_pattern"]).parent
    assert g == track_graph(str(job / "f_####.exr").replace("\\", "/"), 2, str(job / "k_####.png").replace("\\", "/"),
                            g["8"]["inputs"]["filename_prefix"], "x")
    assert "VHS_LoadImagesPath" not in _classes(g)
    f0 = cv2.imread(str(job / "f_0000.exr"), cv2.IMREAD_UNCHANGED)
    assert f0.dtype == np.float32 and f0.shape == (1920, 1920, 3) and np.array_equal(f0, frames[0])
    assert masks[0].shape == (1920, 1920)


def test_png8_detect_uses_vhs(tmp_path):
    outs = MockOutputs(tmp_path / "out")
    seen = {}

    def detect(graph):
        (_, sam), = find_nodes(graph, "SAM3_Detect")
        img = load_node_images(graph, sam["inputs"]["image"][0])[0]
        seen.update(graph=graph, shape=img.shape)
        return outs.save_masks("8", graph["8"]["inputs"]["filename_prefix"], [(img[..., 2] > 0.5).astype(np.float32)])

    client = MockComfyClient({"SAM3_Detect": detect})
    client.output_root = outs.root
    img = np.zeros((1920, 1920, 3), np.float32)
    img[500:900, 600:1000, 2] = 1.0
    masks = Sam3Runner(client, tmp_path / "io", ckpt="x", io_mode="png8").detect(img, prompt="box")
    g = seen["graph"]
    assert g["2"]["class_type"] == "VHS_LoadImagesPath" and "PlateLoadEXR" not in _classes(g)
    assert seen["shape"][:2] == (1920, 1920)                                     # detect: full size (points are in px)
    assert masks[0].shape == (1920, 1920) and masks[0][700, 800] == 1.0 and masks[0][100, 100] == 0.0


@pytest.mark.parametrize("mode", ["png8", "exr"])
def test_cancel_inside_the_write_loop(tmp_path, mode):
    """'Przerwij' during the frame writes: JobCancelled at the next frame, nothing is sent to Comfy."""
    seen = {}
    runner, client = _track_runner(tmp_path, mode, seen)
    jid = JOBS.start("t", "Kadrowanie: test")
    JOBS.bind(jid)
    written = []

    def frame(i):
        written.append(i)
        if i == 3:
            JOBS.cancel()
        return np.zeros((64, 64, 3), np.float32)

    try:
        with pytest.raises(JobCancelled):
            runner.track(LazySeq(frame, range(50)), np.zeros((64, 64), np.float32))
        assert written == [0, 1, 2, 3] and client.sent == []
    finally:
        JOBS.finish(jid, "cancelled")
    # progress of the write loop is visible on the job (not 0/0)
    jid = JOBS.start("t", "Kadrowanie: test2")
    JOBS.bind(jid)
    try:
        runner.track([np.zeros((64, 64, 3), np.float32)] * 4, np.zeros((64, 64), np.float32))
        j = next(x for x in JOBS.snapshot() if x["id"] == jid)
        assert (j["step"], j["done"], j["total"]) == ("SAM3: zapis klatek do Comfy", 4, 4)
    finally:
        JOBS.finish(jid)


def test_fit_sam_res():
    a = np.zeros((1080, 1920, 3), np.float32)
    assert fit_sam_res(a, SAM_RES).shape == (567, 1008, 3)
    b = np.zeros((500, 400), np.float32)
    assert fit_sam_res(b, SAM_RES) is b


# ------------------------------------------------------------------------------------------- 8-bit shot end to end
def _png8_shot(tmp_path, n=16):
    """The synthetic shot as an 8-bit PNG sequence (display values) – like a decoded MP4."""
    shot = make_shot(tmp_path, n=n)
    d = tmp_path / "png8"
    d.mkdir()
    for f in range(shot.first, shot.last + 1):
        lin = cv2.imread(shot.plate_pattern.replace("####", f"{f:04d}"), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(d / f"shot_{f:04d}.png"), np.clip(np.rint(display(lin) * 255), 0, 255).astype(np.uint8))
    return shot, str(d / "shot_####.png").replace("\\", "/")


def test_png8_shot_end_to_end_core_and_matanyone(tmp_path):
    from engine.crop import choose_instance, run_crop_engine, save_crops
    from engine.pipeline import ObjectPipeline
    from engine.segment import iou
    shot, pattern = _png8_shot(tmp_path)
    outs = MockOutputs(tmp_path / "comfy_out")
    client = MockComfyClient(handlers={**sam3_handlers(outs), **matanyone_handlers(outputs=outs)})
    client.output_root = outs.root
    client.get_system_stats = lambda: {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}
    work = tmp_path / "work" / "matte_desk"
    frames = list(range(shot.first, shot.last + 1))
    pipe = ObjectPipeline(work, pattern, shot.w, shot.h, client, sam3_ckpt="sam3.st", use_mplate=False,
                          frames=frames, io_mode="png8")
    pipe.store.create("puppet", ["felt puppet"])
    start = frames[len(frames) // 2]
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="felt puppet")
    pipe.store.save_key("puppet", start, cands[0] > 0.5, author="auto")
    pipe.store.update("puppet", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("puppet") / "crops.json", res.segments)
    pipe.run_core("puppet", frames)
    full, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")
    ious = [iou(full[f] > 0.5, shot.masks[f] > 0.5) for f in frames]
    assert min(ious) >= 0.95, ious
    # every graph read PNG folders (no EXR in or out), MatAnyone alpha came back via SaveImage
    for g in client.sent:
        cls = _classes(g)
        assert not {"PlateLoadEXR", "PlateLoadMaskSeq", "PlateSaveEXR"} & set(cls), cls
        assert "VHS_LoadImagesPath" in cls
    assert any("MatAnyone2RunKeyframes" in _classes(g) for g in client.sent)
    assert not list((work / "comfy_io").rglob("*.exr"))
    for d in (work / "comfy_io").glob("ma_*"):
        assert sorted(p.name for p in d.iterdir()) == ["core", "frames", "keys"]  # frames / masks never share a folder


def test_png8_matanyone_job_folder_cleared_between_runs(tmp_path):
    """A shorter second run in the same ma_<seg>_<tag> folder must not see frames of the first one (VHS reads all)."""
    from engine.crop import CropSegment
    from engine.refine import Finalizer
    outs = MockOutputs(tmp_path / "out")
    client = MockComfyClient(handlers=matanyone_handlers(outputs=outs))
    client.output_root = outs.root
    seg = CropSegment("s00", 1, 10, 0, 0, [0, 0, 64, 64], [0, 0, 0, 0], 1.0)
    img = np.full((64, 64, 3), 0.5, np.float32)
    fin = Finalizer(client, tmp_path / "io", 64, 64, lambda f, s, sc: img, io_mode="png8")
    core = {f: np.zeros((64, 64), bool) for f in range(1, 11)}
    for f in core:
        core[f][20:40, 20:40] = True
    a = fin._matanyone_segment(seg, list(range(1, 11)), core, {1: core[1]}, 1, 1.0, "final")
    b = fin._matanyone_segment(seg, list(range(1, 5)), core, {1: core[1]}, 1, 1.0, "final")
    assert len(a) == 10 and len(b) == 4
    assert len(list((tmp_path / "io" / "ma_s00_final" / "frames").iterdir())) == 4


# ------------------------------------------------------------------------------------------- sRGB display
def test_srgb_plates_reach_the_models_as_their_own_values(tmp_path):
    """8-bit / 16-bit display-referred plates: no PlateToDisplay curve (only EXR is treated as scene-linear)."""
    from engine.ingest import ProxyGenerator
    from engine.pipeline import PlateCropLoader, default_display
    rng = np.random.default_rng(1)
    u8 = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    u8[0, 0] = 1                                                                  # a dark frame must not escape / 255
    p8 = tmp_path / "a_0001.png"
    cv2.imwrite(str(p8), u8)
    ld = PlateCropLoader(str(tmp_path / "a_####.png"), default_display)
    assert np.array_equal(ld.frame(1), u8.astype(np.float32) / 255.0)
    dark = np.zeros((8, 8, 3), np.uint8)
    dark[0, 0] = 1
    cv2.imwrite(str(tmp_path / "d_0001.png"), dark)
    assert PlateCropLoader(str(tmp_path / "d_####.png"), default_display).frame(1).max() == pytest.approx(1 / 255)
    u16 = rng.integers(0, 65536, (40, 50, 3), dtype=np.uint16)
    cv2.imwrite(str(tmp_path / "b_0001.png"), u16)
    v16 = PlateCropLoader(str(tmp_path / "b_####.png"), default_display).frame(1)
    assert v16.max() <= 1.0 and np.allclose(v16, u16.astype(np.float32) / 65535.0)
    # proxy (UI + Kadrowanie input) of an sRGB PNG: the pixel values, untouched
    rgb = ProxyGenerator.load_frame_rgb(p8, colorspace="sRGB")
    assert np.array_equal(rgb, u8[..., ::-1])


# ------------------------------------------------------------------------------------------- export format
def test_resolve_export_format():
    from engine.export import resolve_export_format
    assert resolve_export_format("auto", 8) == "png" and resolve_export_format(None, 8) == "png"
    assert resolve_export_format("auto", 16) == "exr" and resolve_export_format("auto", 32) == "exr"
    assert resolve_export_format("png", 16) == "png" and resolve_export_format("EXR", 8) == "exr"
    with pytest.raises(ValueError):
        resolve_export_format("tiff", 8)


def test_export_format_png_and_exr_files_and_outputs(tmp_path):
    from engine.contract import validate
    from engine.export import export_objects
    from test_e6_acceptance import _two_objects
    shot, pipe, frames, work = _two_objects(tmp_path, n=6)
    objs = pipe.store.list()
    out = export_objects(work, "SYN", objs, frames, shot.w, shot.h, bit_depth=16, multichannel=True)   # auto -> exr
    assert out["export_format"] == "exr" and out["mattes_all"].endswith(".exr") and validate(out) == []
    assert all(o["matte"].endswith("_####.exr") for o in out["objects"])
    out = export_objects(work, "SYN", objs, frames, shot.w, shot.h, bit_depth=16, multichannel=True, export_format="png")
    assert out["export_format"] == "png" and validate(out) == []
    assert all(o["matte"].endswith("_####.png") for o in out["objects"])
    assert "mattes_all" not in out and "mattes_all_channels" not in out        # no multi-EXR for PNG (stale entry dropped)
    p = Path(out["objects"][0]["matte"].replace("####", f"{frames[1]:04d}"))
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert m.dtype == np.uint8 and m.shape == (shot.h, shot.w)
    out = export_objects(work, "SYN", objs, frames, shot.w, shot.h, bit_depth=8, multichannel=True, export_format="auto")
    assert out["export_format"] == "png"
    out = export_objects(work, "SYN", objs, frames, shot.w, shot.h, bit_depth=8, multichannel=True, export_format="exr")
    assert out["export_format"] == "exr" and out["mattes_all"].endswith(".exr")


def test_export_format_api_in_project_json(tmp_path, monkeypatch):
    import app as app_module
    from test_ui_stages_api import _open
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    pj = tmp_path / "work" / "matte_desk" / "project.json"
    d = api.get("/api/project/export_format").json()
    assert d["format"] == "auto" and d["resolved"] == "exr" and d["auto_resolves_to"] == "exr"
    d = api.put("/api/project/export_format", json={"format": "png"}).json()
    assert d["format"] == "png" and d["resolved"] == "png"
    assert json.loads(pj.read_text(encoding="utf-8"))["export_format"] == "png"
    assert api.put("/api/project/export_format", json={"format": "tiff"}).status_code == 400
    monkeypatch.setitem(app_module.STATE["manifest"], "bit_depth", 8)
    d = api.put("/api/project/export_format", json={"format": "auto"}).json()
    assert d["format"] == "auto" and d["resolved"] == "png" and d["bit_depth"] == 8
    assert "export_format" not in json.loads(pj.read_text(encoding="utf-8"))
    r = api.post("/api/export", json={"format": "gif"})
    assert r.status_code in (400, 409)                                           # bad format (or nothing to export)
    assert mf.read_bytes() == before                                             # the shot manifest is never written


def test_md_cli_export_format_flag():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import md
    calls = []

    class H:
        def request(self, method, path, body=None, timeout=None):
            calls.append((method, path, body))
            if path == "/api/jobs":
                return 200, {"jobs": []}
            return 200, {"success": True, "outputs": {"objects": []}}

    md.main(["export", "--all", "--yes", "--format", "png"], http=H(), out=lambda s: None)
    md.main(["export", "puppet", "--yes"], http=H(), out=lambda s: None)
    posts = [c for c in calls if c[0] == "POST"]
    assert posts[0][2] == {"objects": [], "format": "png"} and posts[1][2] == {"objects": ["puppet"]}
