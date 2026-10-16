# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Target background plate "bg" (comp preview): source kinds, frame mapping onto the shot, cover-fit to the proxy."""
import cv2
import numpy as np

from engine import bgplate


def _img(path, val, w=64, h=32):
    im = np.full((h, w, 3), val, np.uint8)
    cv2.imwrite(str(path), im)


def test_single_image_every_frame(tmp_path):
    p = tmp_path / "sky.png"
    _img(p, 100)
    src = bgplate.source_info(str(p))
    assert src["kind"] == "single" and src["count"] == 1
    assert bgplate.source_frame(src, 1, 1) == 0 and bgplate.source_frame(src, 500, 1) == 0
    out = bgplate.proxy_path(tmp_path / "work", str(p), 7, 1, (40, 30), "sRGB", src)
    im = cv2.imread(str(out))
    assert im.shape == (30, 40, 3) and abs(int(im.mean()) - 100) <= 2


def test_sequence_mapping(tmp_path):
    for n in range(1001, 1006):
        _img(tmp_path / f"bg_{n:04d}.png", n - 1000)
    src = bgplate.source_info(str(tmp_path / "bg_####.png"))
    assert src["kind"] == "seq" and (src["first"], src["last"]) == (1001, 1005)
    assert bgplate.source_frame(src, 1003, 1) == 1003          # numbers overlap: same frame number
    assert bgplate.source_frame(src, 1, 1) == 1001             # shot F1 = sequence start
    assert bgplate.source_frame(src, 3, 1) == 1003
    assert bgplate.source_frame(src, 50, 1) == 1005            # clamped to the last frame


def test_video_mapping(tmp_path):
    p = tmp_path / "bg.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 32))
    for i in range(6):
        vw.write(np.full((32, 64, 3), 40 * i, np.uint8))
    vw.release()
    src = bgplate.source_info(str(p))
    assert src["kind"] == "video" and src["count"] == 6
    assert bgplate.source_frame(src, 101, 101) == 0 and bgplate.source_frame(src, 103, 101) == 2
    assert bgplate.source_frame(src, 999, 101) == 5
    out = bgplate.proxy_path(tmp_path / "work", str(p), 104, 101, (32, 16), "sRGB", src)
    im = cv2.imread(str(out))
    assert im.shape == (16, 32, 3) and abs(int(im.mean()) - 120) <= 12


def test_fit_cover_keeps_aspect():
    rgb = np.zeros((100, 400, 3), np.uint8)
    rgb[:, 150:250] = 255                       # centre band
    out = bgplate.fit_cover(rgb, 100, 100)      # 4:1 into 1:1 -> centre crop
    assert out.shape == (100, 100, 3) and out[50, 50, 0] == 255


def test_cache_dir_per_source_and_size(tmp_path):
    a = bgplate.cache_dir(tmp_path, "a.png", (10, 10))
    assert a != bgplate.cache_dir(tmp_path, "b.png", (10, 10))
    assert a != bgplate.cache_dir(tmp_path, "a.png", (20, 10))
