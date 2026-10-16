# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z26 – plate QA (engine/plate_qa) and repair / `_fix` writing (engine/plate_fix): a synthetic textured pan with a slow
light change plus a missing frame, a +1.5 EV frame, a 30 px shifted frame and a duplicate -> all detected, nothing else."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.plate_fix import fixed_dir_for, repair_frame, write_fixed_plate
from engine.plate_qa import analyze_frames, analyze_plate, load_report, save_report, suggest_method

H, W = 120, 180
N = 24
MISSING, EV, SHIFT, DUP = 7, 11, 15, 18


def _seq(seed=0, step=2):
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.uniform(0.1, 0.9, (H, W + step * N, 3)).astype(np.float32), (0, 0), 1.2)
    frames = {}
    for i in range(N):
        f = i + 1
        img = big[:, step * i:step * i + W].copy() * (1.0 + 0.01 * i)      # slow pan + slow light change (+1 % / frame)
        if f == EV:
            img = img * 2.0 ** 1.5                                         # +1.5 EV
        if f == SHIFT:
            img = np.roll(img, 30, axis=1)                                 # 30 px image jump
        if f == DUP:
            img = frames[f - 1].copy()                                     # duplicate of the previous frame
        frames[f] = np.clip(img, 0, 4).astype(np.float32)
    return frames


def test_analyze_frames_detects_every_defect_without_false_alarms():
    frames = _seq()
    load = lambda f: None if f == MISSING else frames[f]
    rep = analyze_frames(list(range(1, N + 1)), load)
    flags = {f: rep["frames"][f]["flags"] for f in rep["frames"]}
    assert "missing" in flags[MISSING]
    assert "exposure_jump" in flags[EV]
    assert "image_jump" in flags[SHIFT]
    assert "duplicate" in flags[DUP]
    assert sorted(rep["flagged"]) == [MISSING, EV, SHIFT, DUP], rep["flagged"]
    assert suggest_method(rep["frames"][MISSING]) in ("mean", "flow")
    assert suggest_method(rep["frames"][EV]) == "gain"
    assert suggest_method(rep["frames"][SHIFT]) == "flow" and suggest_method(rep["frames"][DUP]) in ("mean", "flow")
    assert suggest_method(rep["frames"][3]) == "keep"
    # a plain smooth sequence: nothing flagged
    clean = _seq_clean()
    rep2 = analyze_frames(list(range(1, N + 1)), lambda f: clean[f])
    assert rep2["flagged"] == [], rep2["flagged"]


def _seq_clean(seed=0, step=2):
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.uniform(0.1, 0.9, (H, W + step * N, 3)).astype(np.float32), (0, 0), 1.2)
    return {i + 1: np.clip(big[:, step * i:step * i + W] * (1.0 + 0.01 * i), 0, 4).astype(np.float32) for i in range(N)}


def test_nan_and_size_flags():
    frames = _seq_clean()
    frames[5] = frames[5].copy(); frames[5][10:20, 10:20] = np.nan
    frames[9] = cv2.resize(frames[9], (W // 2, H // 2))
    rep = analyze_frames(list(range(1, N + 1)), lambda f: frames[f])
    assert "nan" in rep["frames"][5]["flags"] and rep["frames"][5]["nan_px"] == 100
    assert "size" in rep["frames"][9]["flags"]


def test_repair_methods():
    frames = _seq_clean()
    fr = list(range(1, N + 1))
    load = lambda f: frames.get(f)
    img, info = repair_frame(10, "mean", fr, load)
    assert info["method"] == "mean" and np.allclose(img, 0.5 * (frames[9] + frames[11]))
    img, info = repair_frame(10, "flow", fr, load)
    assert info["method"] == "flow" and img.shape == frames[10].shape
    # flow interpolation on a pure pan is closer to the truth than the plain mean (the mean ghosts the texture)
    assert np.abs(img - frames[10]).mean() < np.abs(0.5 * (frames[9] + frames[11]) - frames[10]).mean()
    bright = {**frames, 10: frames[10] * 2.0 ** 1.5}
    img, info = repair_frame(10, "gain", fr, lambda f: bright.get(f))
    assert info["method"] == "gain" and all(0.3 < g < 0.4 for g in info["gain"])
    assert np.abs(img.mean() - frames[10].mean()) < 0.02 * frames[10].mean()
    assert repair_frame(10, "keep", fr, load) == (None, {"method": "keep"})
    with pytest.raises(ValueError):
        repair_frame(10, "nope", fr, load)
    # missing neighbour on one side -> copy of the other
    img, info = repair_frame(1, "mean", fr, load)
    assert info["method"] == "copy" and np.array_equal(img, frames[2])


def _write_png16(d: Path, frames, skip=()):
    d.mkdir(parents=True, exist_ok=True)
    for f, img in frames.items():
        if f in skip:
            continue
        cv2.imwrite(str(d / f"p_{f:04d}.png"), np.clip(img / 4.0 * 65535 + 0.5, 0, 65535).astype(np.uint16))
    return str(d / "p_####.png")


def test_write_fixed_plate_png_hardlinks_and_report(tmp_path):
    frames = _seq()
    pattern = _write_png16(tmp_path / "plate", frames, skip=(MISSING,))
    fr = list(range(1, N + 1))
    rep = analyze_plate(pattern, fr)
    assert MISSING in rep["flagged"] and EV in rep["flagged"]
    p = save_report(tmp_path / "work", "beauty", rep)
    assert load_report(tmp_path / "work", "beauty")["flagged"] == rep["flagged"]
    decisions = {f: suggest_method(rep["frames"][f]) for f in rep["flagged"]}
    plan = write_fixed_plate(pattern, fr, decisions, confirmed=False)
    assert plan["success"] is False and plan["plan"]["hardlinks"] == N - len(decisions)
    res = write_fixed_plate(pattern, fr, decisions, confirmed=True)
    assert res["success"] and set(res["repaired"]) == set(decisions)
    new_dir, new_pattern = fixed_dir_for(pattern)
    assert Path(res["pattern"]) == Path(new_pattern) and (new_dir / "README.txt").exists() and (new_dir / "plate_fix.json").exists()
    assert len(list(new_dir.glob("p_*.png"))) == N                       # the missing frame exists now
    assert res["hardlinks"] + res["copies"] == N - len(decisions)
    # untouched frames: the same file (hard link) or byte-identical copy
    src, dst = Path(pattern.replace("####", "0003")), new_dir / "p_0003.png"
    assert dst.read_bytes() == src.read_bytes()
    if res["hardlinks"]:
        assert os.stat(src).st_ino == os.stat(dst).st_ino
    # repaired frames = the in-memory repair quantised to 16 bit (bitwise)
    img, _ = repair_frame(MISSING, decisions[MISSING], fr, lambda g: None if g == MISSING else
                          cv2.imread(pattern.replace("####", f"{g:04d}"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0)
    written = cv2.imread(str(new_dir / f"p_{MISSING:04d}.png"), cv2.IMREAD_UNCHANGED)
    assert written.dtype == np.uint16 and np.array_equal(written, np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16))
    # the fixed plate passes the QA
    rep2 = analyze_plate(new_pattern, fr)
    assert rep2["flagged"] == [], rep2["flagged"]


def test_write_fixed_plate_exr_keeps_header(tmp_path):
    pytest.importorskip("OpenEXR")
    from engine.exr_io import read_exr, write_exr_like
    import Imath, OpenEXR
    frames = _seq_clean()
    d = tmp_path / "exr"; d.mkdir()
    hdr = OpenEXR.Header(W, H)
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    hdr["channels"] = {"R": half, "G": half, "B": half, "A": half}
    for f, img in frames.items():
        if f == 5:
            continue
        ch = {"R": img[..., 2], "G": img[..., 1], "B": img[..., 0], "A": np.full((H, W), 0.5, np.float32)}
        write_exr_like(d / f"e_{f:04d}.exr", ch, hdr)
    pattern = str(d / "e_####.exr")
    fr = list(range(1, N + 1))
    res = write_fixed_plate(pattern, fr, {5: "mean"}, confirmed=True)
    assert res["success"] and 5 in res["repaired"]
    ch, h2 = read_exr(Path(res["dir"]) / "e_0005.exr")
    assert set(ch) == {"R", "G", "B", "A"} and np.allclose(ch["A"], 0.5)       # other channels copied from the reference frame
    assert h2["compression"].v == Imath.Compression.ZIP_COMPRESSION
    exp = 0.5 * (frames[4] + frames[6])
    assert np.abs(ch["R"] - exp[..., 2]).max() < 2e-3                           # half precision


def test_api_plate_qa_and_fix(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    # damage one frame of the synthetic plate: remove it (missing)
    import os
    from engine.plate_qa import frame_path
    pattern = app_module.STATE["manifest"]["plates"]["beauty"]
    victim = frames[len(frames) // 2]
    p = frame_path(pattern, victim)
    bak = p.read_bytes()
    p.unlink()
    try:
        r = api.post("/api/plate/qa", json={"plate": "beauty"})
        assert r.status_code == 200, r.text
        assert victim in r.json()["flagged"] and r.json()["suggested"][str(victim)] in ("mean", "flow")
        rep = api.get("/api/plate/qa/beauty").json()
        assert rep["exists"] and "missing" in rep["frames"][str(victim)]["flags"]
        # plan without confirmation: nothing written
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "mean"}})
        assert r.status_code == 200 and r.json()["success"] is False and r.json()["plan"]["hardlinks"] == len(frames) - 1
        from engine.plate_fix import fixed_dir_for
        new_dir, new_pattern = fixed_dir_for(pattern)
        assert not new_dir.exists()
        r = api.post("/api/plate/fix", json={"plate": "beauty", "decisions": {str(victim): "mean"}, "confirmed": True})
        assert r.status_code == 200 and r.json()["success"], r.text
        assert (new_dir / "README.txt").exists() and frame_path(new_pattern, victim).exists()
        # the project now reads the fixed plate; the manifest file is untouched
        assert Path(app_module.STATE["manifest"]["plates"]["beauty"]) == Path(new_pattern)
        mf = json.loads((tmp_path / "work" / "shot_manifest_SYN.json").read_text(encoding="utf-8"))
        assert Path(mf["plates"]["beauty"]) == Path(pattern)
        r = api.post("/api/plate/qa", json={"plate": "beauty"})
        assert r.status_code == 200 and r.json()["flagged"] == []
    finally:
        p.write_bytes(bak)
