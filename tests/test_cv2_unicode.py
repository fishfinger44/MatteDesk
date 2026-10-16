# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""cv2.imread / imwrite through engine.cv2_unicode handle non-ASCII paths (Windows ANSI API bug)."""
import numpy as np

import engine  # noqa: F401  (installs the wrapper)
import cv2


def test_png_roundtrip_polish_path(tmp_path):
    d = tmp_path / "październik żółć"
    d.mkdir()
    img = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    p = d / "R1_0001.png"
    assert cv2.imwrite(str(p), img)
    back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert back is not None and np.array_equal(back, img)
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert g.shape == (48, 64)


def test_png16_and_params(tmp_path):
    p = tmp_path / "ą" / "m.png"
    p.parent.mkdir()
    img = (np.arange(32 * 32) * 40).astype(np.uint16).reshape(32, 32)
    assert cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    assert np.array_equal(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), img)


def test_missing_file_returns_none(tmp_path):
    assert cv2.imread(str(tmp_path / "ł" / "nie_ma.png")) is None


def test_ascii_path_uses_native(tmp_path):
    p = tmp_path / "a.png"
    assert cv2.imwrite(str(p), np.zeros((4, 4), np.uint8))
    assert cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape == (4, 4)


def test_hash_pattern_gives_frame_range(tmp_path):
    from engine.ingest import MediaInspector
    d = tmp_path / "płyta"
    d.mkdir()
    for n in (5, 6, 7, 1059):
        cv2.imwrite(str(d / f"R1_{n:04d}.png"), np.zeros((8, 16, 3), np.uint8))
    r = MediaInspector.resolve_sequence_pattern(str(d / "R1_####.png"))
    assert (r["first_frame"], r["last_frame"], r["frame_count"]) == (5, 1059, 4)
    assert (r["width"], r["height"]) == (16, 8)
