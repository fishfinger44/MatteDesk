# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""SAM3 soft-probability path (PlateMatte node SAM3_VideoTrackSoft, 2026-10-14): runner -> propagation / cache ->
core_soft/ -> corestab probability mode. The binary core is bit-identical with and without the soft path."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from engine.corestab import stabilize_core
from engine.crop import run_crop_engine, save_crops
from engine.pipeline import ObjectPipeline
from engine.sam3 import SOFT_NODE, Sam3Runner, track_graph, with_soft_output
from engine.segment import CorePropagator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from mock_comfy import MockComfyClient, MockOutputs  # noqa: E402
from synth import display, make_shot, matanyone_handlers, sam3_handlers  # noqa: E402
from test_e5_acceptance import StatsClient  # noqa: E402


def _client(tmp_path, soft: bool):
    outs = MockOutputs(tmp_path / "comfy_out")
    info = {SOFT_NODE: {"input": {}}} if soft else {}
    client = StatsClient(handlers={**sam3_handlers(outs), **matanyone_handlers()}, object_info=info)
    client.output_root = outs.root
    return client


def test_with_soft_output_graph_and_runner_track(tmp_path):
    g = track_graph("f_####.exr", 4, "k_####.png", "p/t", "ck")
    g2 = with_soft_output(g, "p/t")
    assert g2["5"]["class_type"] == SOFT_NODE and g2["5"]["inputs"] == g["5"]["inputs"]
    assert g2["9"]["inputs"]["mask"] == ["5", 1] and g2["10"]["inputs"]["filename_prefix"] == "p/t_soft"
    assert g2["6"] == g["6"] and g2["8"] == g["8"]            # binary path untouched
    shot = make_shot(tmp_path, n=4)
    c_soft, c_plain = _client(tmp_path, True), _client(tmp_path, False)
    r_soft = Sam3Runner(c_soft, tmp_path / "io_s", ckpt="sam3.st")
    r_plain = Sam3Runner(c_plain, tmp_path / "io_p", ckpt="sam3.st")
    assert r_soft.soft_available() and not r_plain.soft_available()
    from engine.pipeline import PlateCropLoader
    loader = PlateCropLoader(shot.plate_pattern, display)
    frames = [loader.frame(f) for f in range(shot.first, shot.first + 4)]
    from synth import PUPPET_BGR, colour_key
    init = colour_key(frames[0], PUPPET_BGR, 0.12) > 0.5
    masks, softs = r_soft.track(frames, init.astype(np.float32), soft=True)
    plain = r_plain.track(frames, init.astype(np.float32))
    assert len(masks) == len(softs) == len(plain) == 4
    for m, s, p in zip(masks, softs, plain):
        assert np.array_equal(m > 0.5, p > 0.5)                 # bit-identical binary masks
        assert 0.0 <= s.min() and s.max() <= 1.0 and s.shape == m.shape
        assert ((s > 0.5) == (m > 0.5)).mean() > 0.97           # soft ~ blurred binary
        assert ((s > 0.02) & (s < 0.98)).sum() > 0               # a real soft band
    assert isinstance(r_plain.track(frames, init.astype(np.float32), soft=True), list)   # no node -> plain list


def _propagate(tmp_path, soft, n=8):
    shot = make_shot(tmp_path, n=n)
    client = _client(tmp_path, soft)
    pipe = ObjectPipeline(tmp_path / "work" / ("s" if soft else "p") / "matte_desk", shot.plate_pattern, shot.w, shot.h,
                          client, display=display, sam3_ckpt="sam3.st", use_mplate=False)
    pipe.store.create("obj", ["felt puppet"])
    frames = list(range(shot.first, shot.last + 1))
    start = frames[len(frames) // 2]
    from engine.crop import choose_instance
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="felt puppet")
    pipe.store.save_key("obj", start, cands[0] > 0.5, author="auto")
    pipe.store.update("obj", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("obj") / "crops.json", res.segments)
    return shot, pipe, frames, start


def test_run_core_soft_maps_cache_and_bitwise_core(tmp_path):
    shot_s, pipe_s, frames, start = _propagate(tmp_path, True)
    shot_p, pipe_p, _, _ = _propagate(tmp_path, False)
    rs = pipe_s.run_core("obj", frames)
    rp = pipe_p.run_core("obj", frames)
    for f in frames:
        assert np.array_equal(rs.core[f], rp.core[f])            # binary core bit-identical with / without soft
    sdir = pipe_s.store.dir("obj") / "core_soft"
    assert len(rs.soft) == len(frames) and all((sdir / f"p_{f:04d}.png").exists() for f in frames)
    assert not (pipe_p.store.dir("obj") / "core_soft").exists()
    cj = json.loads((pipe_s.store.dir("obj") / "core" / "core.json").read_text())
    assert cj["soft_frames"] == len(frames)
    p = cv2.imread(str(sdir / f"p_{frames[0]:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert p.dtype == np.uint8 and p.shape == rs.core[frames[0]].shape
    assert ((p > 127) == rs.core[frames[0]]).mean() > 0.97 and ((p > 5) & (p < 250)).sum() > 0
    assert rs.soft[start].max() == 255 and set(np.unique(rs.soft[start])) <= {0, 255}   # key frame = hard truth
    # second run: every interval from the cache, soft maps included (p_####.png in cache/<interval>/)
    n0 = pipe_s.client.stats.get("SAM3_VideoTrackSoft", 0) if hasattr(pipe_s.client, "stats") else None
    rs2 = pipe_s.run_core("obj", frames)
    assert rs2.cached_intervals and not rs2.recomputed_intervals and len(rs2.soft) == len(frames)
    for f in frames:
        assert np.array_equal(rs2.soft[f], rs.soft[f]) and np.array_equal(rs2.core[f], rs.core[f])
    # probability mode in the stabilisation
    core = pipe_s.load_core("obj", frames)
    out, st = pipe_s.stabilized_core("obj", frames, core, pipe_s.keys("obj"))
    assert st["mode"] == "prob" and st["prob_frames"] == len(frames)
    assert np.array_equal(out[start], core[start])             # key untouched
    out_p, st_p = pipe_p.stabilized_core("obj", frames, core, pipe_p.keys("obj"))
    assert st_p["mode"] == "binary"


def _pan_probs(n=9, step=2, h=160, w=240, r=40, seed=0):
    """Textured pan with a disc; probabilities = sigmoid of the signed distance (soft edge 2 px); frame 5 jitters:
    its probability edge is shifted 3 px outwards on the right half (the binary core follows)."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.uniform(0.2, 0.8, (h, w + step * n)).astype(np.float32), (0, 0), 0.7)
    cores, probs, truth, grays = {}, {}, {}, {}
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        g = big[:, step * i:step * i + w].copy()
        cx, cy = 80 + step * i, h // 2
        d = r - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)          # signed distance, > 0 inside
        if i == 4:
            d = np.where(xx > cx, d + 3.0, d)
        p = 1.0 / (1.0 + np.exp(-d / 1.5))
        c = p > 0.5
        t = (r - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)) > 0
        g[t] = g[t] * 0.5 + 0.5
        cores[i + 1], probs[i + 1], truth[i + 1], grays[i + 1] = c, (p * 255).astype(np.uint8), t, g
    return cores, probs, truth, grays


def test_stabilize_core_probability_mode_repairs_jitter_and_keeps_interior():
    cores, probs, truth, grays = _pan_probs()
    out, st = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, probs=probs)
    assert st["mode"] == "prob" and st["prob_frames"] == 9
    bad = cores[5] != truth[5]
    assert bad.sum() > 100
    assert (out[5] == truth[5])[bad].mean() > 0.6, (out[5] == truth[5])[bad].mean()
    for f in cores:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        inner = cv2.erode(cores[f].astype(np.uint8), k).astype(bool)
        outer = ~cv2.dilate(cores[f].astype(np.uint8), k).astype(bool)
        assert np.array_equal(out[f][inner], cores[f][inner]) and np.array_equal(out[f][outer], cores[f][outer])
    # binary fallback for frames without a map; keys untouched
    out2, st2 = stabilize_core(cores, lambda f: grays[f], radius=2, band_px=8, probs={5: probs[5]}, keys=[5])
    assert st2["prob_frames"] == 1 and np.array_equal(out2[5], cores[5])
