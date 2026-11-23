# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export speed-up (user request 2026-11-20, SD160_020: ~2.2 s a frame on a 6K plate):
- read_alpha decodes the one channel it takes with OpenEXR (0.78 -> 0.05 s on a 4624 px crop) – the same values as
  cv2.imread + the channel pick, for RGB / RGBA / 1-channel files, half and float; other files still go through cv2;
- export_objects writes the frames in a process pool – every file equal to the one-process export."""

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.exr_io import write_gray_exr  # noqa: E402
from engine.refine import read_alpha  # noqa: E402


def _cv2_alpha(p):
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else m[..., 2]
    return m.astype(np.float32)


def test_read_alpha_equals_cv2(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.random((37, 53)).astype(np.float32)
    cases = {
        "rgb_half.exr": dict(half=True, channels=("R", "G", "B")),
        "rgb_float.exr": dict(half=False, channels=("R", "G", "B")),
        "y_half.exr": dict(half=True, channels=("Y",)),
    }
    for name, kw in cases.items():
        write_gray_exr(tmp_path / name, a, **kw)
    # RGBA with a different A (A must be picked, not R)
    write_gray_exr(tmp_path / "rgba.exr", a, half=True, channels=("R", "G", "B"))
    from engine.exr_io import read_exr, write_exr_like
    ch, hdr = read_exr(tmp_path / "rgba.exr")
    import Imath
    hdr["channels"]["A"] = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    ch["A"] = 1.0 - a
    write_exr_like(tmp_path / "rgba.exr", ch, hdr)
    for name in [*cases, "rgba.exr"]:
        p = tmp_path / name
        got, ref = read_alpha(p), _cv2_alpha(p)
        assert got.dtype == np.float32 and got.shape == ref.shape, name
        assert np.array_equal(got, ref), name
    # PNG: the cv2 path as before
    cv2.imwrite(str(tmp_path / "a.png"), (a * 255).astype(np.uint8))
    assert np.array_equal(read_alpha(tmp_path / "a.png"), _cv2_alpha(tmp_path / "a.png"))


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _pixels(p):
    if p.suffix == ".exr":
        f = OpenEXR.InputFile(str(p))
        hdr = f.header()
        import Imath
        out = {c: f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)) for c in sorted(hdr["channels"])}
        f.close()
        return out
    return p.read_bytes()


def test_parallel_export_equals_one_process(tmp_path, monkeypatch):
    from engine.export import export_objects
    from test_e6_acceptance import _two_objects
    shot, pipe, frames, work = _two_objects(tmp_path, n=6)
    objs = pipe.store.list()
    outs = {}
    for n in ("1", "2"):
        monkeypatch.setenv("MD_EXPORT_WORKERS", n)
        d = tmp_path / f"export_{n}"
        prog = []
        outs[n] = export_objects(work, "SYN", objs, frames, shot.w, shot.h, bit_depth=16, multichannel=True,
                                 with_core=True, export_dir=d, progress_cb=lambda s, a, b: prog.append((a, b)))
        assert sorted(prog) == [(i, len(frames)) for i in range(1, len(frames) + 1)]
    a, b = tmp_path / "export_1", tmp_path / "export_2"
    assert _files(a) == _files(b) and _files(a)
    for rel in _files(a):
        if rel.endswith(".nk"):
            continue                       # the script names its own export dir
        assert _pixels(a / rel) == _pixels(b / rel), rel
    strip = lambda o: json.loads(json.dumps({k: v for k, v in o.items() if k not in ("export_dir", "mattes_nk", "updated_at")}).replace("export_2", "export_1"))
    da, db = strip(outs["1"]), strip(outs["2"])
    for e in da.get("objects", []) + db.get("objects", []):
        e.pop("date", None)
    assert da == db
    assert outs["2"]["mattes_all_channels"] == ["puppet", "tree", "union"]
