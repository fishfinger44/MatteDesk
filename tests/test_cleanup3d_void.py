# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P12 (docs/PLAN_CLEANUP_3D.md): VOID + SeedVR2 as key engines (engine/cleanup3d/void_engine.py).

Synthetic, no real tools: a fake VOID that "removes" the object by painting the quadmask's non-keep area a constant
colour, a fake SeedVR2 that upscales + sharpens the crop deterministically. Checked: the VOID input (quadmask values,
frame count padded to 4k + 1, the 1344 x 768 size, prompt.json), the run record, keys_from_void keeps the plate bit for
bit outside the patch (and records source / base / run), SeedVR2 gets ONE clip of ONE crop for all keys and the new
versions record their base, the runners' commands (the experiment's flags), cancel kills the process, the gates /
Doctor / licences, the API routes. `real` tests at the end run the configured tools on sh060 (not run in the cloud).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import void_engine as VE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PW, PH = 320, 180
FILL = (40, 160, 90)                      # what the fake VOID paints over the removed object (BGR uint8)


def plate8(f):
    """Plate F: a horizontal gradient + noise-free texture, a red disc (the object) moving right."""
    yy, xx = np.mgrid[0:PH, 0:PW]
    img = np.dstack([(xx * 0.6 + 30) % 255, (yy * 1.1 + 20) % 255, ((xx + yy) * 0.3 + 60) % 255]).astype(np.uint8)
    m = obj(f) > 0.5
    img[m] = (30, 30, 220)
    return img


def obj(f):
    yy, xx = np.mgrid[0:PH, 0:PW]
    cx, cy = 80 + 12 * (f - 1), 90
    return (((xx - cx) / 18.0) ** 2 + ((yy - cy) / 30.0) ** 2 <= 1.0).astype(np.float32)


def plate_fn(f):
    return plate8(f).astype(np.float32) / 255.0


class FakeVoid:
    """Stands in for VoidRunner: reads the PNG frames the input video / quadmask were made of, fills qm < 255."""

    def __init__(self):
        self.calls = []

    def run(self, data_root, seq, save_path, size, length, window, cancel_check=None):
        d = Path(data_root) / seq
        rgb = sorted((d / "_rgb").glob("*.png"))
        qm = sorted((d / "_qm").glob("*.png"))
        self.calls.append({"size": tuple(size), "length": length, "window": window, "n_rgb": len(rgb), "n_qm": len(qm),
                           "prompt": json.loads((d / "prompt.json").read_text(encoding="utf-8"))["bg"],
                           "qm_values": sorted(set(np.unique(np.concatenate([cv2.imread(str(p), 0).ravel() for p in qm])).tolist()))})
        out = Path(save_path) / "frames"
        out.mkdir(parents=True)
        for i, (a, b) in enumerate(zip(rgb, qm)):
            im, q = cv2.imread(str(a)), cv2.imread(str(b), 0)
            assert im.shape[:2] == q.shape == (size[1], size[0])
            im[q < 255] = FILL
            cv2.imwrite(str(out / f"o_{i:06d}.png"), im)
        return out


class FakeSV:
    """Stands in for SeedVR2Runner: the clip's PNG frames (next to the clip file) upscaled to `resolution` (short side)
    and sharpened by a fixed unsharp mask."""

    def __init__(self):
        self.calls = []

    def run(self, src, out_dir, resolution, batch, frames, cancel_check=None):
        files = sorted((Path(src).parent / "clip").glob("*.png"))
        ims = [cv2.imread(str(p)) for p in files]
        self.calls.append({"n": len(ims), "shapes": {im.shape for im in ims}, "resolution": resolution, "batch": batch,
                           "frames": frames})
        o = Path(out_dir) / "clip"
        o.mkdir(parents=True)
        for i, im in enumerate(ims):
            h, w = im.shape[:2]
            k = resolution / float(min(h, w))
            up = cv2.resize(im, (round(w * k), round(h * k)), interpolation=cv2.INTER_CUBIC)
            sh = cv2.addWeighted(up, 1.5, cv2.GaussianBlur(up, (0, 0), 1.5), -0.5, 0)
            cv2.imwrite(str(o / f"clip_{i:06d}.png"), sh)
        return Path(out_dir)


def no_encode(pattern, dst, lossless):
    Path(dst).write_bytes(b"")


# ----------------------------------------------------------------------------- input preparation
def test_quadmask_values():
    o = np.zeros((40, 60), np.float32)
    o[5:30, 20:30] = 1
    a = VE.grown(o, 6)
    keep = np.zeros_like(o)
    keep[:, 50:] = 1
    q = VE.quadmask(o, a, keep)
    assert set(np.unique(q)) == {0, 63, 127, 255}
    assert (q[o > 0][:5] == 0).all() and q[2, 25] == 127 and q[0, 0] == 255
    assert (q[:, 50:] != 127).all()                      # a kept object is never "affected"
    ys = np.nonzero(q == 63)[0]
    assert ys.min() > 3 + 0.8 * (31 - 3)                 # feet: only the lowest 20 % of the object (grown 2 px)


def test_void_size_and_length():
    assert VE.void_size(3840, 2160) == (1344, 768) and VE.void_size(1080, 1920) == (768, 1344)
    assert [VE.video_length(n) for n in (1, 2, 5, 6, 57, 58)] == [1, 5, 5, 9, 57, 61]


def test_void_fill_input_and_run(tmp_path):
    work = tmp_path / "matte_desk"
    frames = list(range(1, 7))                            # 6 frames -> padded to 9
    fake = FakeVoid()
    r = VE.void_fill(work, frames, plate_fn, obj, fake, encode=no_encode, affect_px=8, prompt="empty street")
    c = fake.calls[0]
    assert c["size"] == (1344, 768) and c["length"] == 9 and c["n_rgb"] == c["n_qm"] == 9 and c["window"] == 9
    assert c["prompt"] == "empty street"
    assert {0, 127, 255} <= set(c["qm_values"]) <= {0, 63, 127, 255}
    data = Path(r["dir"]) / "input" / VE.VOID_SEQ
    assert (data / "input_video.mp4").exists() and (data / "quadmask_0.mp4").exists()
    assert r["status"] == "done" and r["frames"] == frames and r["size"] == [1344, 768]
    for f in frames:
        im = cv2.imread(r["paths"][f])
        assert im.shape[:2] == (768, 1344)
    assert VE.last_run(work)["id"] == r["id"]
    st = VE.load_runs(work)
    assert st["last"] == r["id"] and st["runs"][-1]["qm_values"] == r["qm_values"]


def test_void_fill_failure_is_recorded(tmp_path):
    class Boom(FakeVoid):
        def run(self, *a, **k):
            raise VE.ToolError("VOID zakończył się błędem 1: CUDA out of memory")
    work = tmp_path / "matte_desk"
    with pytest.raises(VE.ToolError):
        VE.void_fill(work, [1, 2], plate_fn, obj, Boom(), encode=no_encode)
    d = VE.load_runs(work)
    assert d["last"] is None and d["runs"][-1]["status"] == "failed" and "CUDA" in d["runs"][-1]["error"]
    assert VE.last_run(work) is None


def test_encode_video_lossless_quadmask(tmp_path):
    """With a real ffmpeg (when on PATH): the four quadmask values survive the lossless encode."""
    import shutil
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "qm"
    d.mkdir()
    q = np.full((64, 96), 255, np.uint8)
    q[10:30, 10:40], q[30:40, 10:40], q[10:40, 40:60] = 0, 63, 127
    for i in range(5):
        cv2.imwrite(str(d / f"{i:06d}.png"), q)
    VE.encode_video(ff, d / "%06d.png", tmp_path / "q.mp4", True)
    fr = VE.read_frames(tmp_path / "q.mp4")
    assert len(fr) == 5 and set(np.unique(cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY))) == {0, 63, 127, 255}


# ----------------------------------------------------------------------------- keys from VOID
def far_from_patch(f):
    """Pixels beyond the reach of any patch of frame f: the object grown (40) + the traces' edge (12) + the soft ramp
    (24) + 2 px – there a key must be the plate bit for bit (the fake fill changes only the object + 6 px)."""
    far = cv2.distanceTransform((obj(f) == 0).astype(np.uint8), cv2.DIST_L2, 5) > \
        KS.REBASE_GROW_PX + KS.EFFECT_EDGE_PX + KS.REBASE_SOFT_PX + 2
    assert far.sum() > 5000
    return far


def _write_cards(work, ids=("ground",)):
    from engine.cleanup3d import types as T
    store.save_cards(work, [T.Card(i, i, plane={"n": [0.0, -1.0, 0.0], "d": 2.0}) for i in ids])


def test_keys_from_void_plate_outside_patch(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work, ("ground", "sky"))
    run = VE.void_fill(work, list(range(1, 10)), plate_fn, obj, FakeVoid(), encode=no_encode, affect_px=6)
    KS.create_keys(work, "ground", [1, 5, 9], plate_fn, "png")
    res = VE.keys_from_void(work, ["ground", "sky"], {"sky": [5]}, plate_fn=plate_fn, mask_fn=obj, ext="png",
                            effects=True)
    assert res["run"] == run["id"]
    assert sorted((k["card"], k["frame"]) for k in res["keys"]) == [("ground", 1), ("ground", 5), ("ground", 9), ("sky", 5)]
    for card, f in (("ground", 5), ("sky", 5)):
        k = store.load_key(work, card, f)
        assert [e["source"] for e in k.versions] == ["plate", "void"]          # a new key starts from the plate
        e = k.versions[-1]
        assert e["base"] == "v001" and e["void_run"] == run["id"] and k.active == e["v"]
        img = np.rint(KS.read_key_image(work, k) * 255).astype(np.uint8)
        pl = plate8(f)
        far = far_from_patch(f)
        assert np.array_equal(img[far], pl[far])                               # the plate bit for bit outside
        inside = obj(f) > 0.5
        assert np.abs(img[inside].astype(int) - np.array(FILL)).mean() < 40    # the VOID fill in the hole
    a = np.rint(KS.read_key_image(work, store.load_key(work, "ground", 5)) * 255)
    b = np.rint(KS.read_key_image(work, store.load_key(work, "sky", 5)) * 255)
    assert np.array_equal(a, b)                                                 # one patch per frame for every card


def test_keys_from_void_refusals(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work)
    with pytest.raises(ValueError, match="VOID"):
        VE.keys_from_void(work, ["ground"], None, plate_fn=plate_fn, mask_fn=obj)
    VE.void_fill(work, [1, 2, 3], plate_fn, obj, FakeVoid(), encode=no_encode)
    with pytest.raises(ValueError, match="Brak kluczy"):
        VE.keys_from_void(work, ["ground"], None, plate_fn=plate_fn, mask_fn=obj)
    with pytest.raises(ValueError, match="F7"):
        VE.keys_from_void(work, ["ground"], {"ground": [2, 7]}, plate_fn=plate_fn, mask_fn=obj)


def test_keys_from_void_blur_hook_called(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work)
    VE.void_fill(work, [1, 2, 3], plate_fn, obj, FakeVoid(), encode=no_encode)
    seen = []

    def blur_fn(f):
        def b(fill, pl, hole):
            seen.append((f, bool(hole.any())))
            return fill
        return b
    VE.keys_from_void(work, ["ground"], {"ground": [2]}, plate_fn=plate_fn, mask_fn=obj, blur_fn=blur_fn)
    assert seen == [(2, True)]


# ----------------------------------------------------------------------------- SeedVR2 on the keys of a card
def test_patch_crop_one_box():
    ms = [obj(1), obj(5), obj(9)]
    x0, y0, w, h = VE.patch_crop(ms, (PH, PW), grow_px=8, margin=10)
    assert w % 16 == 0 and h % 16 == 0 and 0 <= x0 and x0 + w <= PW and 0 <= y0 and y0 + h <= PH
    for m in ms:
        ys, xs = np.nonzero(m)
        assert x0 <= xs.min() and xs.max() < x0 + w and y0 <= ys.min() and ys.max() < y0 + h
    with pytest.raises(ValueError):
        VE.patch_crop([np.zeros((PH, PW))], (PH, PW))


def test_seedvr2_one_clip_and_base(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work, ("ground",))
    VE.void_fill(work, list(range(1, 10)), plate_fn, obj, FakeVoid(), encode=no_encode, affect_px=6)
    KS.create_keys(work, "ground", [1, 3, 5, 7, 9, 2], plate_fn, "png")
    VE.keys_from_void(work, ["ground"], {"ground": [1, 3, 5, 7, 9]}, plate_fn=plate_fn, mask_fn=obj, ext="png")
    sv = FakeSV()
    res = VE.seedvr2_keys(work, "ground", plate_fn=plate_fn, mask_fn=obj, runner=sv, encode=no_encode)
    c = sv.calls
    assert len(c) == 1 and c[0]["n"] == 5 and len(c[0]["shapes"]) == 1        # ONE clip, ONE crop for all keys
    assert c[0]["batch"] == 5 and c[0]["frames"] == 5 and c[0]["resolution"] == min(res["crop"][2:])
    assert sorted(k["frame"] for k in res["keys"]) == [1, 3, 5, 7, 9]          # F2 is still the plate: skipped
    x0, y0, cw, ch = res["crop"]
    # in_scale default = VOID long side / plate long side (capped at 1) -> 1 here (the plate is smaller than VOID)
    assert res["clip_size"][0] >= min(cw, VE.SEEDVR2_MIN_SIDE) or res["clip_size"] == [cw, ch]
    for f in (1, 5, 9):
        k = store.load_key(work, "ground", f)
        e = k.versions[-1]
        assert e["source"] == "seedvr2:v002" and e["base"] == "v002" and e["crop"] == res["crop"]
        img = np.rint(KS.read_key_image(work, k) * 255).astype(np.uint8)
        pl = plate8(f)
        far = far_from_patch(f)
        assert np.array_equal(img[far], pl[far])
    assert [e["source"] for e in store.load_key(work, "ground", 2).versions] == ["plate"]
    # again on the new versions: the chain goes on (base = v003)
    VE.seedvr2_keys(work, "ground", plate_fn=plate_fn, mask_fn=obj, runner=FakeSV(), encode=no_encode, frames=[5])
    e = store.load_key(work, "ground", 5).versions[-1]
    assert e["source"] == "seedvr2:v003" and e["base"] == "v003"


def test_seedvr2_scales_the_clip_down_by_the_void_ratio(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work)
    VE.void_fill(work, [1, 2, 3], plate_fn, obj, FakeVoid(), encode=no_encode, size=(160, 90))
    KS.create_keys(work, "ground", [1, 2, 3], plate_fn, "png")
    VE.keys_from_void(work, ["ground"], None, plate_fn=plate_fn, mask_fn=obj, ext="png")
    sv = FakeSV()
    res = VE.seedvr2_keys(work, "ground", plate_fn=plate_fn, mask_fn=obj, runner=sv, encode=no_encode,
                          in_scale=0.5)
    cw, ch = res["crop"][2:]
    assert res["clip_size"] == [max(2, round(cw * 0.5 / 2) * 2), max(2, round(ch * 0.5 / 2) * 2)] or \
        min(res["clip_size"]) >= min(VE.SEEDVR2_MIN_SIDE, min(cw, ch))


def test_seedvr2_refuses_plate_only_keys(tmp_path):
    work = tmp_path / "matte_desk"
    _write_cards(work)
    KS.create_keys(work, "ground", [1, 2], plate_fn, "png")
    with pytest.raises(ValueError, match="plate"):
        VE.seedvr2_keys(work, "ground", plate_fn=plate_fn, mask_fn=obj, runner=FakeSV(), encode=no_encode)


def test_batch_sizes():
    assert [VE._batch(n) for n in (1, 2, 5, 6, 15, 40)] == [1, 5, 5, 9, 17, 21]
    assert VE._batch(40, 13) == 13 and VE._batch(3, 1) == 1


# ----------------------------------------------------------------------------- runners
def test_runner_commands_follow_the_experiment(tmp_path):
    v = VE.VoidRunner(tmp_path / "py.exe", tmp_path / "void-model", tmp_path / "models")
    cmd = v.command(tmp_path / "in", "shot", tmp_path / "out", (1344, 768), 57, 57)
    assert cmd[:3] == [VE.VOID_SCRIPT, "--config", VE.VOID_CONFIG]
    assert "--config.data.sample_size=768x1344" in cmd and "--config.video_model.temporal_window_size=57" in cmd
    assert "--config.experiment.run_seqs=shot" in cmd and "--config.data.max_video_length=57" in cmd
    assert any(c.endswith("CogVideoX-Fun-V1.5-5b-InP") for c in cmd) and any(c.endswith("void_pass1.safetensors") for c in cmd)
    s = VE.SeedVR2Runner(tmp_path / "py.exe", tmp_path / "seedvr2", tmp_path / "models")
    cmd = s.command(tmp_path / "clip.mkv", tmp_path / "out", 1024, 9, 7)
    for a in ("--vae_encode_tiled", "--vae_decode_tiled", "--uniform_batch_size"):
        assert a in cmd
    assert cmd[cmd.index("--dit_model") + 1] == "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
    assert cmd[cmd.index("--color_correction") + 1] == "lab" and cmd[cmd.index("--resolution") + 1] == "1024"
    assert cmd[cmd.index("--output_format") + 1] == "png" and "--temporal_overlap" not in cmd
    assert "--temporal_overlap" in s.command(tmp_path / "c.mkv", tmp_path / "o", 512, 5, 12)


def test_void_runner_runs_a_process_under_the_lease(tmp_path):
    """A stand-in predict_v2v.py in a fake repo: the runner starts it with the tool's Python, returns the result video
    (not the `_tuple` one) and logs the command."""
    repo = tmp_path / "void-model"
    (repo / "inference" / "cogvideox_fun").mkdir(parents=True)
    (repo / "inference" / "cogvideox_fun" / "predict_v2v.py").write_text(
        "import sys, pathlib\n"
        "save = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--config.experiment.save_path=')][0]\n"
        "p = pathlib.Path(save); p.mkdir(parents=True, exist_ok=True)\n"
        "(p / 'shot-fg=-1-0001.mp4').write_bytes(b'x'); (p / 'shot-fg=-1-0001_tuple.mp4').write_bytes(b'x')\n"
        "print('done')\n", encoding="utf-8")
    log = tmp_path / "void.log"
    v = VE.VoidRunner(sys.executable, repo, tmp_path / "models", log=log)
    out = v.run(tmp_path / "in", "shot", tmp_path / "out", (1344, 768), 9, 9)
    assert out.name == "shot-fg=-1-0001.mp4"
    assert "predict_v2v.py" in log.read_text(encoding="utf-8") and "done" in log.read_text(encoding="utf-8")


def test_runner_cancel_kills_the_process(tmp_path):
    from engine.jobs import JobCancelled
    r = VE._ProcRunner(sys.executable, tmp_path)
    t0 = time.time()

    def cancel():
        if time.time() - t0 > 0.6:
            raise JobCancelled()
    with pytest.raises(JobCancelled):
        r._run(["-c", "import time; time.sleep(60)"], cancel)
    assert time.time() - t0 < 20


def test_runner_error_has_the_log_tail(tmp_path):
    r = VE._ProcRunner(sys.executable, tmp_path, log=tmp_path / "x.log")
    with pytest.raises(VE.ToolError, match="CUDA out of memory"):
        r._run(["-c", "import sys; print('CUDA out of memory'); sys.exit(3)"], None)


def test_leased_and_stage_calls_registered():
    from engine.gpu_manager import LEASED_CALLS, STAGE_CALLS
    assert ("engine/cleanup3d/void_engine.py", "VoidRunner", "run") in LEASED_CALLS
    assert ("engine/cleanup3d/void_engine.py", "SeedVR2Runner", "run") in LEASED_CALLS
    assert ("engine/cleanup3d/void_engine.py", "void_fill", "comfy") in STAGE_CALLS
    assert ("engine/cleanup3d/void_engine.py", "seedvr2_keys", "comfy") in STAGE_CALLS


# ----------------------------------------------------------------------------- gate, Doctor, licences
def _tool_tree(tmp, tool):
    """Every file `tool` needs, under tmp/<tool>; returns its config."""
    if tool == "void":   # natively in ComfyUI (2026-11-11): the Comfy-Org files in models_dir
        cfg = {"models_dir": str(tmp / "void_models")}
    else:
        cfg = {"seedvr2_python": str(tmp / "sv" / ".venv" / "python.exe"), "seedvr2_repo": str(tmp / "sv"),
               "seedvr2_models": str(tmp / "sv" / "models" / "SEEDVR2")}   # explicit: VOID's models_dir must not move it
    for _d, p in VE.required_files(cfg, tool):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return cfg


def test_gate_unset_missing_ok(tmp_path, monkeypatch):
    from engine import stage_models as G
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    monkeypatch.setattr(VE, "VOID_HOME", tmp_path / "bin_void")         # never this machine's real install
    g = G.cleanup3d_tools({})
    assert not g["void"]["ok"] and not g["seedvr2"]["ok"]
    # VOID lives in ComfyUI's models now, SeedVR2 in ComfyUI -> both unknown without a configured ComfyUI
    assert "Zainstaluj" in g["void"]["reason"]
    assert "seedvr2_python" in g["seedvr2"]["reason"] and "Doktor" in g["seedvr2"]["reason"]
    cfg = _tool_tree(tmp_path, "void")
    cfg.update(_tool_tree(tmp_path, "seedvr2"))
    g = G.cleanup3d_tools(cfg)
    assert g["void"]["ok"] and g["seedvr2"]["ok"]                     # SeedVR2 models default = <repo>/models/SEEDVR2
    (Path(cfg["models_dir"]) / "diffusion_models" / VE.VOID_NATIVE_UNET).unlink()
    g = G.cleanup3d_tools(cfg)
    assert not g["void"]["ok"] and "Brak plików" in g["void"]["reason"] and VE.VOID_NATIVE_UNET in g["void"]["reason"]
    with pytest.raises(VE.ToolError, match="Brak plików"):
        VE.void_runner(cfg)
    assert isinstance(VE.seedvr2_runner(cfg), VE.SeedVR2Runner)
    monkeypatch.setenv("MD_MODEL_GATE", "0")
    assert all(v["ok"] for v in G.cleanup3d_tools({}).values())


def test_doctor_items_and_licences(tmp_path, monkeypatch):
    import doctor
    from doctor import EnvironmentDoctor
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    monkeypatch.setattr(VE, "VOID_HOME", tmp_path / "bin_void")
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    for k in ("void_python", "void_repo", "void_models", "seedvr2_python", "seedvr2_repo", "seedvr2_models"):
        doc.config[k] = ""
    doc.config["comfy_dir"] = str(tmp_path / "comfy")                   # no SeedVR2 node there
    doc.config["models_dir"] = str(tmp_path / "comfy" / "models")
    v, s = doc.check_cleanup3d_tools()
    assert (v.name, s.name) == (doctor.VOID_ITEM, doctor.SEEDVR2_ITEM)
    # nothing installed = info + the Doctor's own install button (2026-11-10)
    assert (v.status, s.status) == ("info", "info") and v.links and s.links
    assert (v.action, s.action) == ("install_void", "install_seedvr2")
    assert "Wymagane teraz: NIE" in v.fix_suggestion and "Comfy-Org/" in v.fix_suggestion and "bin/void" not in v.fix_suggestion
    assert "custom_nodes" in s.fix_suggestion
    doc.config.update(_tool_tree(tmp_path, "void"))
    doc.config["seedvr2_repo"] = str(tmp_path / "nowhere")
    doc.config["seedvr2_python"] = str(tmp_path / "nowhere" / "python.exe")
    v, s = doc.check_cleanup3d_tools()
    assert v.status == "ok" and v.fix_suggestion is None and v.action is None and s.status == "info"
    sv = _tool_tree(tmp_path, "seedvr2")
    (Path(sv["seedvr2_repo"]) / "models" / "SEEDVR2" / VE.SEEDVR2_VAE).unlink()
    doc.config.update(sv, models_dir="")                                 # models = <repo>/models/SEEDVR2 as built
    s = doc.check_cleanup3d_tools()[1]
    assert s.status == "warning" and s.action == "install_seedvr2"        # partly there: warning, install completes it
    assert doctor.LICENSES["void"][1] is False and "void" in doctor.LICENSE_LIMITED
    assert "CogVideoX License" in doctor.LICENSES["void"][0] and doctor.LICENSES["seedvr2"][0].startswith("Apache-2.0")
    rows = {r["name"]: r for r in doctor.licence_summary()}
    assert rows["SeedVR2"]["use"] == "commercial"
    assert rows["VOID (CogVideoX-Fun-V1.5-5b-InP + void_pass1)"]["use"] == "limited"
    tp = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    rd = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**VOID**" in tp and "**SeedVR2**" in tp and "CogVideoX License" in tp
    assert "**VOID**" in rd and "**SeedVR2**" in rd and "`bin/void`" in rd and "ComfyUI's Python" in rd
    ex = json.loads((ROOT / "config.local.example.json").read_text(encoding="utf-8"))
    assert all(ex[k] == "" for k in ("void_python", "void_repo", "void_models", "seedvr2_python", "seedvr2_repo",
                                     "seedvr2_models"))


def test_config_resolves_tool_paths():
    from engine.config import PATH_KEYS
    for keys in VE.TOOL_KEYS.values():
        assert set(keys) <= set(PATH_KEYS)


# ----------------------------------------------------------------------------- API
@pytest.fixture
def api_env(tmp_path, monkeypatch):
    from tests.test_cleanup3d_api import Env
    e = Env(tmp_path, monkeypatch)
    e.void, e.sv = FakeVoid(), FakeSV()
    monkeypatch.setattr(VE, "void_runner", lambda cfg, log=None: e.void)
    monkeypatch.setattr(VE, "seedvr2_runner", lambda cfg, log=None: e.sv)
    monkeypatch.setattr(VE, "encode_video", lambda ff, pat, dst, lossless, **k: Path(dst).write_bytes(b""))
    return e


def test_api_void_keys_and_seedvr2(api_env):
    e = api_env
    c = e.client
    st = c.get("/api/cleanup3d").json()
    assert st["void"]["available"] and st["void"]["last"] is None and st["seedvr2"]["available"]
    assert c.post("/api/cleanup3d/void/keys", json={}).status_code == 409          # no VOID run yet
    assert c.post("/api/cleanup3d/camera/solve", json={}).status_code == 200
    assert c.post("/api/cleanup3d/cards/propose").status_code == 200
    card = c.get("/api/cleanup3d/cards").json()["cards"][0]["id"]
    assert c.post(f"/api/cleanup3d/keys/{card}", json={"frames": [1, 5, 9]}).status_code == 200
    r = c.post("/api/cleanup3d/void", json={"prompt": "empty beach", "affect_px": 6})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["run"]["status"] == "done" and j["run"]["frames"] == list(range(1, 10))
    assert e.void.calls[0]["prompt"] == "empty beach" and e.void.calls[0]["length"] == 9
    assert sorted(k["frame"] for k in j["keys"] if k["card"] == card) == [1, 5, 9]
    k = store.load_key(e.work, card, 5)
    assert k.versions[-1]["source"] == "void" and k.versions[-1]["base"] == "v001"
    st = c.get("/api/cleanup3d").json()
    assert st["void"]["last"]["id"] == j["run"]["id"] and st["void"]["last"]["frames"] == 9
    r = c.post("/api/cleanup3d/void/keys", json={"cards": [card], "frames": [5]})
    assert r.status_code == 200 and [x["frame"] for x in r.json()["keys"]] == [5]
    assert store.load_key(e.work, card, 5).versions[-1]["base"] == "v002"
    r = c.post(f"/api/cleanup3d/keys/{card}/seedvr2", json={})
    assert r.status_code == 200, r.text
    assert len(e.sv.calls) == 1 and e.sv.calls[0]["n"] == 3
    assert store.load_key(e.work, card, 5).versions[-1]["source"] == "seedvr2:v003"
    assert c.post("/api/cleanup3d/keys/nope/seedvr2", json={}).status_code == 404


def test_api_refuses_without_tools(api_env, monkeypatch):
    def gate(cfg, log=None):
        raise VE.ToolError("VOID (usuwanie obiektu z wideo) nie jest skonfigurowany: void_python w config.local.json")
    monkeypatch.setattr(VE, "void_runner", gate)
    r = api_env.client.post("/api/cleanup3d/void", json={})
    assert r.status_code == 409 and "config.local.json" in r.json()["detail"]


# ----------------------------------------------------------------------------- UI (static/cleanup3d.js, own block)
def test_ui_block_in_jsdom():
    """The VOID + SeedVR2 block: mounted under „Tło kluczy z plate'u”, disabled with the gate's reason, the last run,
    the three requests (tests/js/cleanup3d_void_harness.mjs; skipped without Node / jsdom = NOT checked)."""
    from tests.test_cleanup3d_js import _run
    _run("cleanup3d_void_harness.mjs", "OK cleanup3d void harness")


def test_ui_block_wired_to_the_routes_and_translated():
    js = (ROOT / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    api = (ROOT / "api" / "cleanup3d.py").read_text(encoding="utf-8")
    for url, route in (("'/api/cleanup3d/void'", '"/api/cleanup3d/void"'), ("'/api/cleanup3d/void/keys'", '"/api/cleanup3d/void/keys"'),
                       ("/seedvr2`", '"/api/cleanup3d/keys/{card}/seedvr2"')):
        assert url in js and route in api, url
    en = json.loads((ROOT / "static" / "i18n" / "en.json").read_text(encoding="utf-8"))["strings"]
    for s in ("VOID – wypełnienie ujęcia", "▶ Policz VOID", "⟲ Klucze z VOID", "⇧ SeedVR2: podbij klucze karty",
              "VOID niedostępny: {0}", "SeedVR2 niedostępny: {0}", "Wybierz obiekty do usunięcia (krok C1 „Inpaint”)"):
        assert en.get(s), s
    # the block is its own: genBarHtml / renderKeys untouched beyond the one refresh call in syncButtons
    assert js.count("renderVoid();") == 1 and js.count("mountVoid();") == 1


# ----------------------------------------------------------------------------- real tools on sh060 (workstation only)
SHOT = Path(r"D:/MD_REAL_PROJECT/Documents\shots\shots")
BEATCH = SHOT / "matte_desk" / "source" / "sh060"


def _real_cfg():
    from engine.config import load_config
    return load_config()


@pytest.mark.real
def test_real_void_on_sh060(tmp_path):
    """VOID (configured in config.local.json) over sh060 F1-9 with the Women object, into a scratch work dir: the run
    finishes and gives 9 frames at 1344 x 768. Needs the GPU (~24 GB) and minutes; not run in the cloud."""
    from engine.export import run_alpha_loader
    cfg = _real_cfg()
    if not VE.state(tmp_path, cfg)["void"]["available"] or not BEATCH.is_dir():
        pytest.skip("VOID not configured / sh060 missing")
    al = run_alpha_loader(SHOT / "matte_desk/objects/Women", "v005", 3840, 2160)
    plate = lambda f: cv2.imread(str(BEATCH / f"sh060_{f:04d}.png")).astype(np.float32) / 255  # noqa: E731
    r = VE.void_fill(tmp_path / "matte_desk", list(range(1, 10)), plate, lambda f: np.asarray(al(f), np.float32),
                     VE.void_runner(cfg, log=tmp_path / "void.log"), ffmpeg=VE._ffmpeg(cfg))
    assert r["status"] == "done" and len(r["paths"]) == 9


@pytest.mark.real
def test_real_seedvr2_on_sh060_keys(tmp_path):
    """SeedVR2 (configured) on 3 keys of a scratch copy built from the VOID experiment frames (removers_test
    outputs/voidhr3 when present): one clip, new versions with their base."""
    cfg = _real_cfg()
    src = Path(r"D:/MD_REAL_PROJECT/Documents\removers_test\outputs\voidhr3")
    if not VE.state(tmp_path, cfg)["seedvr2"]["available"] or not src.is_dir() or not BEATCH.is_dir():
        pytest.skip("SeedVR2 not configured / experiment frames missing")
    from engine.export import run_alpha_loader
    work = tmp_path / "matte_desk"
    _write_cards(work)
    al = run_alpha_loader(SHOT / "matte_desk/objects/Women", "v005", 3840, 2160)
    mask = lambda f: np.asarray(al(f), np.float32)  # noqa: E731
    plate = lambda f: cv2.imread(str(BEATCH / f"sh060_{f:04d}.png")).astype(np.float32) / 255  # noqa: E731
    for f in (5, 9, 13):
        g = cv2.imread(str(src / f"g_{f:04d}.png")).astype(np.float32) / 255
        KS.create_keys(work, "ground", [f], plate, "png")
        KS.add_version(work, "ground", f, KS.patch_on_plate(plate(f), g, mask(f), effects=True), "void", "png")
    res = VE.seedvr2_keys(work, "ground", plate_fn=plate, mask_fn=mask,
                          runner=VE.seedvr2_runner(cfg, log=tmp_path / "sv.log"), ffmpeg=VE._ffmpeg(cfg))
    assert len(res["keys"]) == 3 and all(k["base"] == "v002" for k in res["keys"])


def test_seedvr2_variant_by_card(tmp_path, monkeypatch):
    """User 2026-11-11: the 7B was „unstable” on 12 GB – the 3B fp8 under 16 GB, the 7B sharp fp8 from there; a variant
    already on disk is used when the card's one is not."""
    from engine import installer
    monkeypatch.delenv("MD_SEEDVR2_VARIANT", raising=False)
    assert VE.seedvr2_variant(12.0) == "3b" and VE.seedvr2_variant(24.0) == "7b" and VE.seedvr2_variant(16.0) == "7b"
    three = VE.SEEDVR2_DITS["3b"][0]
    assert installer.seedvr2_files(12.0)[0] == VE.SEEDVR2_DITS["3b"] and installer.seedvr2_files(24.0)[0][0] == VE.SEEDVR2_DIT
    assert VE.seedvr2_dit(tmp_path, 12.0) == three                       # nothing yet: what the Doctor downloads
    (tmp_path / VE.SEEDVR2_DIT).write_bytes(b"x")
    assert VE.seedvr2_dit(tmp_path, 12.0) == VE.SEEDVR2_DIT               # the 7B installed earlier is used
    (tmp_path / three).write_bytes(b"x")
    assert VE.seedvr2_dit(tmp_path, 12.0) == three and VE.seedvr2_dit(tmp_path, 24.0) == VE.SEEDVR2_DIT



def test_void_native_in_comfyui(tmp_path):
    """VOID natively in ComfyUI (user 2026-11-11): the graph of the user's workflow (pass 1), lengths with an even latent
    count, the files in models_dir (any t5xxl), the native runner unless the old repo + venv setup is complete."""
    assert [VE.void_native_length(n) for n in (1, 45, 49, 50, 53)] == [5, 45, 53, 53, 53]
    g = VE.void_native_graph(tmp_path, tmp_path, "an empty beach", 1344, 768, 53,
                             {"unet": "u", "clip": "t5", "vae": "v"}, seed=3)
    kinds = {n["class_type"] for n in g.values()}
    assert {"VOIDQuadmaskPreprocess", "VOIDInpaintConditioning", "VOIDSampler", "CFGGuider", "SamplerCustomAdvanced"} <= kinds
    assert g["2"]["inputs"]["type"] == "cogvideox" and g["21"]["inputs"]["cfg"] == VE.VOID_NATIVE_CFG
    assert g["20"]["inputs"]["length"] == 53 and g["25"]["inputs"]["latent_image"] == ["20", 2]
    md = tmp_path / "models"
    cfg = {"models_dir": str(md)}
    assert not any(VE.void_native_files(cfg).values())
    for folder, name in (("diffusion_models", VE.VOID_NATIVE_UNET), ("vae", VE.VOID_NATIVE_VAE),
                         ("text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors")):
        (md / folder).mkdir(parents=True, exist_ok=True)
        (md / folder / name).write_bytes(b"x")
    f = VE.void_native_files(cfg)
    assert all(f.values()) and f["clip"].name == "t5xxl_fp8_e4m3fn_scaled.safetensors"
    assert [p for _d, p in VE.required_files(cfg, "void")] == [f["unet"], f["vae"], f["clip"]]
    assert not VE.void_legacy_complete(cfg)


def test_install_void_native_downloads_once(tmp_path):
    from engine import installer
    got = []

    def dl(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    md = tmp_path / "models"
    (md / "text_encoders").mkdir(parents=True)
    (md / "text_encoders" / "t5xxl_fp16.safetensors").write_bytes(b"x")       # T5 already there (shared)
    res = installer.install_void({"models_dir": str(md)}, download=dl)
    assert res["success"] and len(got) == 2 and all("Comfy-Org/void-model" in u for u in got)
    assert any(u.endswith("diffusion_models/void_pass1.safetensors") for u in got)
    assert (md / "vae" / VE.VOID_NATIVE_VAE).exists() and not res["restart_comfy"]


def test_static_camera_plan_card_maps_a_key_1_to_1(tmp_path):
    """C2 „Kamera nieruchoma” (user 2026-11-11): one identity pose for every frame, no COLMAP; the cards step offers the
    „Plan” card facing it, which projects every frame onto the same pixels."""
    from engine.cleanup3d import solve as S
    from engine.cleanup3d import store
    cam = S.static_camera([1, 2, 3], 1080, 1920)
    assert cam.frames == [1, 2, 3] and S.is_static(cam) and np.allclose(cam.R[2], np.eye(3))
    work = tmp_path / "work"
    work.mkdir()
    store.save_camera(work, cam)
    p = S.propose_planes(work)
    assert p["ground"] is None and p["plan"] == {"n": [0.0, 0.0, -1.0], "d": S.STATIC_DEPTH}
    # a pixel → the plane → back to the same pixel in another frame
    K = cam.K
    uv = np.array([300.0, 700.0, 1.0])
    ray = np.linalg.inv(K) @ uv
    X = ray * (S.STATIC_DEPTH / ray[2])                     # n·X + d = 0 with n = (0, 0, -1)
    back = K @ (cam.R[3] @ X + cam.t[3])
    assert np.allclose(back[:2] / back[2], uv[:2])
