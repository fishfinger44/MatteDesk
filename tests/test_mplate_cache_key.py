# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate-to-mask cache must not mix objects.

Found on SH010 (2026-10-12): every object numbers its crop segments s00, s01… and the cache lived in
work/mplate/<seg_id>/, so object "suknia" (crop 1312 px) was tracked by SAM3 on the cached crops of "figure"
(crop 928 px) -> its core turned into Figure's silhouette.
"""

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.pipeline import PlateCropLoader


def _seg(x, y, side):
    return CropSegment(seg_id="s00", first=1, last=5, overlap_in=0, overlap_out=0,
                       crop=[x, y, side, side], pad=[0, 0, 0, 0], scale_to_1008=1.0)


def test_same_segment_id_different_crops_do_not_share_cache(tmp_path):
    plates = tmp_path / "plate"
    plates.mkdir()
    yy, xx = np.mgrid[0:240, 0:320]
    for f in range(1, 6):
        img = np.dstack([xx / 320.0, yy / 240.0, np.full_like(xx, f / 10.0, dtype=float)]).astype(np.float32)
        cv2.imwrite(str(plates / f"p_{f:04d}.png"), (img * 255).astype(np.uint8))
    pattern = str(plates / "p_####.png")
    mdir = tmp_path / "mplate"

    a, b = _seg(10, 10, 64), _seg(150, 100, 96)          # object A, object B: both "s00"
    la = PlateCropLoader(pattern, mplate_dir=mdir, frames=range(1, 6))
    ca = la.crop(3, a)
    lb = PlateCropLoader(pattern, mplate_dir=mdir, frames=range(1, 6))   # new run, same work folder
    cb = lb.crop(3, b)

    assert ca.shape[:2] == (64, 64)
    assert cb.shape[:2] == (96, 96), "object B got object A's cached crops"
    raw_b = lb.raw_crop(3, b)
    assert np.abs(cb - raw_b).mean() < 0.05            # B's crop is B's region (denoise may change it slightly)
    assert len([d for d in mdir.iterdir() if d.is_dir()]) == 2


def test_wrong_size_cache_falls_back_to_raw_crop(tmp_path):
    plates = tmp_path / "plate"
    plates.mkdir()
    for f in range(1, 6):
        cv2.imwrite(str(plates / f"p_{f:04d}.png"), np.full((120, 160, 3), 40 * f, np.uint8))
    pattern = str(plates / "p_####.png")
    seg = _seg(0, 0, 48)
    loader = PlateCropLoader(pattern, mplate_dir=tmp_path / "mplate", frames=range(1, 6))
    loader.crop(2, seg)                                   # builds the cache
    cdir = next(d for d in (tmp_path / "mplate").iterdir() if d.is_dir())
    cv2.imwrite(str(cdir / "m_0002.png"), np.zeros((32, 32, 3), np.uint16))   # corrupt: wrong size
    fresh = PlateCropLoader(pattern, mplate_dir=tmp_path / "mplate", frames=range(1, 6))
    assert fresh.crop(2, seg).shape[:2] == (48, 48)
