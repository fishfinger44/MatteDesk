# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3 / denoise streamed in blocks (2026-10-21): each frame only reads its neighbours, so only a window of frames
is in RAM; results (frames, residuals, report numbers) must be bit for bit those of the original full-list code
(frozen in tests/ref_clean_v1.py), with and without a frame sink, on 1 and 4 threads, for any block size."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ref_clean_v1 as old
from engine import clean
from engine.clean import DenoiseEngine, LazyFrames, PlateCleaner, stream_pixel_stages

OLD_PARAMS = dict(stage1_window=15, stage1_iterations=2, stage3_radius=3, stage1_sigma=6.0, stage1_content_check=0.0)   # the first port's values: these tests check memory / threads / cache, not the defaults


def _frames(n=9, w=96, h=64, seed=3):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.random((12, 20, 3), dtype=np.float32), (w + 20, h + 20), interpolation=cv2.INTER_CUBIC) * 0.6 + 0.2
    out = []
    for i in range(n):
        f = base[5:5 + h, 5 + i % 3:5 + i % 3 + w] * (1 + 0.08 * np.sin(i * 1.7))
        out.append((f + rng.normal(0, 0.01, f.shape)).astype(np.float32))
    return out


def test_stream_driver_order_window_and_eviction():
    n, seen, alive = 23, [], []
    src = list(range(100, 100 + n))
    stages = [(0, lambda _: (lambda i: src[i]), 2),
              (2, lambda inp: (lambda i: sum(inp[j] for j in range(max(0, i - 2), min(n, i + 3)))), 3),
              (1, lambda inp: (lambda i: (inp[i], max(inp[j] for j in range(max(0, i - 1), min(n, i + 2))))), 2)]
    on_value = [lambda i, v: alive.append(i), None, None]
    stream_pixel_stages(n, stages, lambda i, v: seen.append((i, v)), block=4, on_value=on_value)
    assert [i for i, _ in seen] == list(range(n))
    s1 = [sum(src[j] for j in range(max(0, i - 2), min(n, i + 3))) for i in range(n)]
    assert [v for _, v in seen] == [(s1[i], max(s1[max(0, i - 1):min(n, i + 2)])) for i in range(n)]
    assert sorted(alive) == list(range(n))                       # every source frame computed exactly once


@pytest.mark.parametrize("threads,s3,dn", [("4", True, False), ("4", False, True), ("1", True, True)])
def test_streamed_pixel_stages_match_original(monkeypatch, threads, s3, dn):
    monkeypatch.setenv("MD_CLEAN_THREADS", threads)
    frames = _frames()
    kw = dict(enable_stage3=s3, enable_denoise=dn, target_reduction=2.0, **OLD_PARAMS)
    o_frames, o_gains, o_rep, o_res = old.PlateCleaner.process_sequence([f.copy() for f in frames], **kw)
    n_frames, n_gains, n_rep, n_res = PlateCleaner.process_sequence(LazyFrames(len(frames), lambda i: frames[i].copy()), **kw)
    assert n_rep.stages_applied == o_rep.stages_applied
    for i in range(len(frames)):
        assert np.array_equal(n_frames[i], o_frames[i]), i
        if dn:
            assert np.array_equal(n_res[i], o_res[i]), i
    assert n_rep.reduction_pct == o_rep.reduction_pct and n_rep.noise_reduction_pct == o_rep.noise_reduction_pct
    assert n_rep.flicker_after.to_dict() == o_rep.flicker_after.to_dict()

    got = {}                                                     # with a sink: same frames, in order, nothing returned
    s_frames, _, s_rep, s_res = PlateCleaner.process_sequence(LazyFrames(len(frames), lambda i: frames[i].copy()),
                                                             frame_sink=lambda i, f: got.__setitem__(i, f), **kw)
    assert s_frames is None and s_res is None and list(got) == list(range(len(frames)))
    assert all(np.array_equal(got[i], o_frames[i]) for i in range(len(frames)))
    assert s_rep.reduction_pct == o_rep.reduction_pct


def test_small_blocks_same_result(monkeypatch):
    """Block size 1 (heaviest eviction) gives the same frames as one big block."""
    frames = _frames(n=9)
    kw = dict(enable_stage3=True, enable_denoise=True, target_reduction=2.0)
    monkeypatch.setenv("MD_CLEAN_THREADS", "1")                  # workers 1 -> block 1
    a, _, ra, _ = PlateCleaner.process_sequence(LazyFrames(9, lambda i: frames[i].copy()), **kw)
    monkeypatch.setenv("MD_CLEAN_THREADS", "16")
    b, _, rb, _ = PlateCleaner.process_sequence(LazyFrames(9, lambda i: frames[i].copy()), **kw)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and ra.reduction_pct == rb.reduction_pct


def test_gain_only_run_feeds_the_sink():
    frames = _frames(n=6)
    got = []
    c, g, rep, _ = PlateCleaner.process_sequence(LazyFrames(6, lambda i: frames[i].copy()),
                                                 frame_sink=lambda i, f: got.append((i, f)))
    assert [i for i, _ in got] == list(range(6)) and all(np.array_equal(f, c[i]) for i, f in got)


def test_denoise_engine_matches_original():
    frames = _frames(n=7)
    a, ra = DenoiseEngine.denoise_temporal_mc([f.copy() for f in frames], radius=2)
    b, rb = old.DenoiseEngine.denoise_temporal_mc([f.copy() for f in frames], radius=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and all(np.array_equal(x, y) for x, y in zip(ra, rb))


def _exr_project(client, tmp_path, frames):
    import json
    work = tmp_path / "STR"
    plates = work / "plates_beauty"
    plates.mkdir(parents=True)
    for i, f in enumerate(frames, start=1):
        assert cv2.imwrite(str(plates / f"b_{i:04d}.exr"), f)
    h, w = frames[0].shape[:2]
    mf = work / "shot_manifest_STR.json"
    mf.write_text(json.dumps({"shot": "STR", "first": 1, "last": len(frames), "plate_w": w, "plate_h": h,
                              "colorspace": "ACES2065-1", "source_type": "exr_seq",
                              "plates": {"beauty": str(plates / "b_####.exr")}, "work": str(work)}))
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    return plates


def test_api_exr_export_streams_pixel_stages(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    from engine.exr_io import read_rgb
    frames = _frames(n=6)
    client = TestClient(app)
    plates = _exr_project(client, tmp_path, frames)
    src = [read_rgb(plates / f"b_{i:04d}.exr") for i in range(1, 7)]
    ref, _, rep, _ = PlateCleaner.process_sequence([f.copy() for f in src], enable_stage3=True, enable_denoise=True)   # engine defaults, like the endpoint
    r = client.post("/api/plate/clean/run", json={"plate": "beauty", "export_mode": "exr_deflicker", "confirmed": True,
                                                   "enable_stage3": True, "enable_denoise": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["notes"] == [] and "stage3_detail_mc" in d["report"]["stages_applied"] and "denoise_mc" in d["report"]["stages_applied"]
    out = Path(d["export_dir"])
    for i in range(1, 7):
        assert np.array_equal(read_rgb(out / f"b_{i:04d}.exr"), ref[i - 1])     # float EXR source -> exact


def test_api_nuke_group_skips_pixel_stages(tmp_path):
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    _exr_project(client, tmp_path, _frames(n=5))
    r = client.post("/api/plate/clean/run", json={"plate": "beauty", "export_mode": "nuke_group",
                                                   "enable_stage3": True, "enable_denoise": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["export_dir"] == "" and len(d["notes"]) == 1 and "tylko przy zapisie EXR" in d["notes"][0]
    assert not {"stage3_detail_mc", "denoise_mc"} & set(d["report"]["stages_applied"])


def test_stage3_does_not_shift_the_image():
    """2026-10-22: the stage-3 coordinate map lacked the pixel-centre offset -> neighbours sampled 1.5 px off at 4x and the
    export shifted ~0.25 px right / 0.2 px down. Identical frames with mid-scale texture must come out unshifted."""
    from engine.clean import DeflickerEngine
    h, w = 512, 768                                                     # > 1024 is not needed: fw = 1024 > w -> use fw 256
    rng = np.random.default_rng(1)
    base = cv2.resize(rng.random((h // 16, w // 16, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * 0.5 + 0.3
    frames = [base.copy() for _ in range(7)]
    eye = [np.eye(3, dtype=np.float32)] * 7
    out = DeflickerEngine.stage3_frame_fn(frames, eye, radius=3, fw=192)(3)   # 4x like 4096 / 1024
    ga = np.log(base.mean(axis=2)).astype(np.float32)
    gb = np.log(np.maximum(out, 1e-4).mean(axis=2)).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, cv2.createHanningWindow((w, h), cv2.CV_32F))
    assert abs(dx) < 0.03 and abs(dy) < 0.03, (dx, dy)
    assert float(np.abs(out - base).max()) < 0.02                       # identical frames: the median changes ~nothing

