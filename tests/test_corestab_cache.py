# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stabilised SAM 3.1 core of a whole run on disk (2026-10-25): stage 1 writes it, stages 3 / 4 / 5 / 7 read it –
bit-identical to recomputing (core, soft maps, stats, the finalize alpha), recomputed when any input changes."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_soft_start import CC_NEW, _prep  # noqa: E402

import engine.corestab as corestab  # noqa: E402


def _count_calls(monkeypatch):
    calls = []
    real = corestab.stabilize_core

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(corestab, "stabilize_core", spy)
    return calls


def _clear(pipe):
    import shutil
    shutil.rmtree(pipe._corestab_dir("obj"), ignore_errors=True)


def test_cache_hit_is_bit_identical_to_recompute(tmp_path, monkeypatch):
    pipe, frames, _ = _prep(tmp_path / "a", True, CC_NEW)
    keys = pipe.keys("obj")
    calls = _count_calls(monkeypatch)
    _clear(pipe)
    c1, s1 = pipe.core_with_soft("obj", frames, keys, True)
    st1 = json.loads(json.dumps(pipe._corestab_stats))
    assert len(calls) == 1 and s1, "soft maps expected on this shot"
    c2, s2 = pipe.core_with_soft("obj", frames, keys, True)
    assert len(calls) == 1, "second run must come from disk"
    assert pipe._corestab_stats == st1
    assert sorted(c1) == sorted(c2) and sorted(s1) == sorted(s2)
    for f in c1:
        assert c2[f].dtype == c1[f].dtype and np.array_equal(c1[f], c2[f])
    for f in s1:
        assert s2[f].dtype == s1[f].dtype and np.array_equal(s1[f], s2[f])


def test_finalize_from_cache_equals_finalize_without(tmp_path, monkeypatch):
    pa, frames, _ = _prep(tmp_path / "a", True, CC_NEW)
    _clear(pa)
    fa, _, _ = pa.finalize("obj", frames, edge_mode="matanyone")        # computes + writes
    calls = _count_calls(monkeypatch)
    fb, _, rb = pa.finalize("obj", frames, edge_mode="matanyone")       # from disk
    assert not calls
    for f in frames:
        assert np.array_equal(fa[f], fb[f])
    assert json.loads((rb / "run.json").read_text(encoding="utf-8"))["core_stabilize_stats"]["frames"] == len(frames)


def test_stage1_writes_cache_then_stage_run_hits(tmp_path, monkeypatch):
    pipe, frames, _ = _prep(tmp_path / "a", True, CC_NEW)
    _clear(pipe)
    pipe.write_mask12("obj", frames)
    assert any(pipe._corestab_dir("obj").glob("*/corestab.json"))
    calls = _count_calls(monkeypatch)
    pipe.core_with_soft("obj", pipe._visible("obj", frames), pipe.keys("obj"), True)
    assert not calls


def test_cache_invalidated_by_frames_keys_and_core(tmp_path, monkeypatch):
    pipe, frames, _ = _prep(tmp_path / "a", True, CC_NEW)
    keys = pipe.keys("obj")
    _clear(pipe)
    pipe.core_with_soft("obj", frames, keys, True)
    calls = _count_calls(monkeypatch)
    # another frame list (window ends / crop box differ) -> recomputed
    pipe.core_with_soft("obj", frames[1:], keys, True)
    assert len(calls) == 1
    # a changed key mask -> recomputed
    k2 = {f: m.copy() for f, m in keys.items()}
    f0 = next(iter(k2))
    k2[f0][0, 0] = not k2[f0][0, 0]
    pipe.core_with_soft("obj", frames, k2, True)
    assert len(calls) == 2
    # a rewritten core file -> recomputed, and the result equals a fresh computation
    import cv2
    p = pipe.store.dir("obj") / "core" / f"c_{frames[2]:04d}.png"
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    im[:4, :4] = 255 - im[:4, :4]
    cv2.imwrite(str(p), im)
    c3, s3 = pipe.core_with_soft("obj", frames, keys, True)
    assert len(calls) == 3
    _clear(pipe)
    c4, s4 = pipe.core_with_soft("obj", frames, keys, True)
    assert all(np.array_equal(c3[f], c4[f]) for f in frames) and all(np.array_equal(s3[f], s4[f]) for f in s3)
    # at most _CORESTAB_KEEP signatures stay on disk
    assert len([d for d in pipe._corestab_dir("obj").iterdir() if d.is_dir()]) <= pipe._CORESTAB_KEEP


def test_preview_only_is_never_cached(tmp_path, monkeypatch):
    pipe, frames, _ = _prep(tmp_path / "a", True, CC_NEW)
    _clear(pipe)
    pipe.core_with_soft("obj", frames, pipe.keys("obj"), True, only=[frames[2]])
    assert not pipe._corestab_dir("obj").exists() or not any(pipe._corestab_dir("obj").glob("*/corestab.json"))
