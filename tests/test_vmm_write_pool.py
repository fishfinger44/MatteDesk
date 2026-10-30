# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 4 (VideoMaMa) writes, plan docs/PLAN_EDGE_SPEED.md section 11 (2026-10-29) – bit-identical to the former
code: alpha_base is a link of the alpha file (it used to be a second write of the same array), also when an automatic
„sliders on the result” rewrites the version in place (old separate files, or links of the previous pass)."""

from pathlib import Path

import numpy as np

from engine.crop import CropSegment
from engine.pipeline import ObjectPipeline as Pipeline
from engine.refine import read_alpha, write_alpha_exr
from engine.videomama import compose

S = {"reach_px": 12.0, "inset_px": 3.0, "feather_px": 2.0, "outer_feather_px": 6.0}


def _frame(seed, side=80):
    rng = np.random.default_rng(seed)
    base = np.zeros((side, side), np.float32)
    base[20 + seed % 5:60, 18:58 + seed % 3] = 1.0
    raw = rng.random((side, side)).astype(np.float32)
    return raw, base, (base > 0.5).astype(np.float32)


def _seg(side=80):
    return CropSegment("s00", 1, 9, 0, 0, [0, 0, side, side], [0, 0, 0, 0], 1.0)


def test_alpha_base_is_the_alpha_bits(tmp_path: Path):
    seg = _seg()
    for f in (1, 2, 3):
        raw, base, guide = _frame(f)
        Pipeline._vmm_write(None, tmp_path, seg, f, raw, base, guide, S)
        a = compose(base, raw, S["reach_px"], S["inset_px"], S["feather_px"], S["outer_feather_px"], None)
        ref = tmp_path / "ref" / f"a_{f:04d}.exr"
        write_alpha_exr(ref, a)        # the former second write of the same array
        al, ab = (tmp_path / sub / "s00" / f"a_{f:04d}.exr" for sub in ("alpha", "alpha_base"))
        assert al.read_bytes() == ref.read_bytes() == ab.read_bytes()
        assert np.array_equal(read_alpha(ab), read_alpha(ref))
        assert np.array_equal(read_alpha(tmp_path / "vmm_raw" / "s00" / f"a_{f:04d}.exr"),
                              np.clip(raw, 0, 1).astype(np.float16).astype(np.float32))


def test_rewrite_in_place_keeps_alpha_base_current(tmp_path: Path):
    """Automatic sliders overwrite the version: an old separate alpha_base file and a link from the previous pass both
    end up with the new alpha."""
    seg = _seg()
    raw, base, guide = _frame(1)
    old = tmp_path / "alpha_base" / "s00" / "a_0001.exr"
    write_alpha_exr(old, np.zeros((80, 80), np.float32))           # a version written before the link
    for reach in (12.0, 30.0, 5.0):                                    # three passes in place
        s = {**S, "reach_px": reach}
        Pipeline._vmm_write(None, tmp_path, seg, 1, raw, base, guide, s)
        a = compose(base, raw, reach, S["inset_px"], S["feather_px"], S["outer_feather_px"], None)
        want = np.clip(a, 0, 1).astype(np.float16).astype(np.float32)
        assert np.array_equal(read_alpha(tmp_path / "alpha" / "s00" / "a_0001.exr"), want)
        assert np.array_equal(read_alpha(old), want)


def test_pool_frames_equal_sequential_frames(tmp_path: Path):
    """_vmm_frame in a thread pool (vmm_run / vmm_refinish) writes the files of the sequential _vmm_write, byte for
    byte, for every frame."""
    from concurrent.futures import ThreadPoolExecutor
    seg = _seg()
    data = {f: _frame(f) for f in range(1, 9)}
    seq, par = tmp_path / "seq", tmp_path / "par"
    for f, (raw, base, guide) in data.items():
        Pipeline._vmm_write(None, seq, seg, f, raw, base, guide, S)
    with ThreadPoolExecutor(max_workers=4) as pool:
        sts = list(pool.map(lambda f: object.__new__(Pipeline)._vmm_frame(par, seg, f, lambda: data[f][0], lambda: data[f][1],
                                                          lambda: data[f][2], S), data))
    assert all({"compose", "write_alpha", "write_alpha_base", "write_vmm_raw"} <= set(st) for st in sts)
    files = sorted(p.relative_to(seq) for p in seq.rglob("*") if p.is_file())
    assert len(files) == 4 * len(data)
    assert files == sorted(p.relative_to(par) for p in par.rglob("*") if p.is_file())
    for rel in files:
        assert (seq / rel).read_bytes() == (par / rel).read_bytes(), rel

def test_vmm_base_core_made_per_frame_same_values():
    """vmm_base "core": the frames are made on access (no dict of every frame), with the former values."""
    core12 = {f: _frame(f)[1] > 0.5 for f in (3, 4, 5)}
    base, run = Pipeline.vmm_base(None, "Woman", [3, 4, 5], {"base": "core"}, core12)
    assert run is None and list(base) == [3, 4, 5] and len(base) == 3 and 4 in base and 6 not in base
    for f in (3, 4, 5):
        assert base[f].dtype == np.float32 and np.array_equal(base[f], core12[f].astype(np.float32))