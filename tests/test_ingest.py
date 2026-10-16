# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_ingest.py - Unit tests for engine/ingest.py."""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from engine.ingest import MediaInspector, ProjectCreator, ProxyGenerator, tonemap_plate_to_display
from engine.project import ShotProject


def test_tonemap_plate_to_display():
    # Test linear float array tonemapping
    linear_img = np.array([[[0.0, 0.18, 1.0, 5.0]]], dtype=np.float32)
    srgb = tonemap_plate_to_display(linear_img, white=1.0)
    assert srgb.dtype == np.uint8
    assert srgb.shape == linear_img.shape
    # Zero stays zero
    assert srgb[0, 0, 0] == 0
    # 0.18 mid grey maps to non-zero reasonable value
    assert 50 < srgb[0, 0, 1] < 150
    # Over 1.0 clamped and compressed
    assert srgb[0, 0, 2] > srgb[0, 0, 1]
    assert srgb[0, 0, 3] == 255


def test_media_inspector_sequence_detection(tmp_path):
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()

    # Create dummy image sequence
    for f in range(101, 111):
        file_path = seq_dir / f"test_shot_{f:04d}.png"
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(file_path), img)

    seqs = MediaInspector.scan_directory(seq_dir)
    assert len(seqs) == 1
    info = seqs[0]
    assert info["first_frame"] == 101
    assert info["last_frame"] == 110
    assert info["frame_count"] == 10
    assert info["extension"] == "png"
    assert "####" in info["pattern"]

    meta = MediaInspector.read_metadata(info["sample_file"])
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["bit_depth"] == 8


def test_project_creator_isolation(tmp_path):
    work_dir = tmp_path / "shot_work"
    work_dir.mkdir()

    creator = ProjectCreator()
    manifest = creator.create_manifest(
        shot_name="TEST_010",
        work_dir=str(work_dir),
        beauty_pattern=str(work_dir / "plates" / "b_####.png"),
        first_frame=1,
        last_frame=50,
        plate_w=1920,
        plate_h=1080
    )

    manifest_file = work_dir / "shot_manifest_TEST_010.json"
    assert manifest_file.exists()

    before = manifest_file.read_bytes()
    proj_info = creator.init_project_from_manifest(manifest_file)
    work = Path(proj_info["work_dir"])
    assert work.name == "matte_desk"
    assert (work / "project.json").exists()
    assert (work / "proxy").is_dir()
    assert (work / "objects").is_dir()
    assert (work / "export").is_dir()
    # opening a shot never writes to its manifest
    assert manifest_file.read_bytes() == before

    # the wizard refuses to overwrite an existing manifest
    with pytest.raises(FileExistsError):
        creator.create_manifest(shot_name="TEST_010", work_dir=str(work_dir), beauty_pattern="x_####.png",
                                first_frame=1, last_frame=2, plate_w=10, plate_h=10)
    assert manifest_file.read_bytes() == before


def test_proxy_generator(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    proxy_dir = tmp_path / "proxy"

    # Create a 4K dummy frame (4096 x 2731)
    sample_src = frames_dir / "sample_0001.png"
    test_img = np.ones((2731, 4096, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(sample_src), test_img)

    dest_proxy = proxy_dir / "p_0001.jpg"
    ProxyGenerator.render_proxy_file(
        source_frame_path=sample_src,
        dest_proxy_path=dest_proxy,
        colorspace="sRGB",
        max_side=1920
    )

    assert dest_proxy.exists()
    res = cv2.imread(str(dest_proxy))
    h, w = res.shape[:2]
    assert w == 1920
    # Aspect ratio preserved ~ 1920 / 4096 * 2731 ≈ 1280
    assert abs(h - 1280) <= 2


def test_plate_edits_go_to_project_json_not_manifest(tmp_path):
    work_dir = tmp_path / "shot_work"
    work_dir.mkdir()

    creator = ProjectCreator()
    creator.create_manifest(
        shot_name="TEST_PLATES",
        work_dir=str(work_dir),
        beauty_pattern=str(work_dir / "plates" / "b_####.exr"),
        first_frame=1,
        last_frame=10,
        plate_w=1920,
        plate_h=1080,
        backlit_pattern="D:/x/backlit_####.exr",
    )
    manifest_file = work_dir / "shot_manifest_TEST_PLATES.json"
    before = manifest_file.read_bytes()

    proj = ShotProject.open(manifest_file)

    cp_dir = tmp_path / "cleanplate_seq"
    cp_dir.mkdir()
    for f in range(1, 11):
        cv2.imwrite(str(cp_dir / f"cp_{f:04d}.png"), np.zeros((10, 10, 3), dtype=np.uint8))
    info = MediaInspector.resolve_sequence_pattern(str(cp_dir))
    assert info["frame_count"] == 10
    proj.set_plate("cleanplate", info["pattern"])
    proj.set_plate("custom_depth", "D:/test/depth_####.exr")
    proj.remove_plate("custom_depth")
    proj.remove_plate("backlit")  # a manifest plate: hidden in project.json

    eff = proj.effective_manifest()
    assert "####" in eff["plates"]["cleanplate"]
    assert "custom_depth" not in eff["plates"]
    assert "backlit" not in eff["plates"]
    assert proj.plate_origin("cleanplate") == "project"

    # manifest untouched, state persisted in project.json and reloaded
    assert manifest_file.read_bytes() == before
    again = ShotProject.open(manifest_file)
    assert again.effective_manifest()["plates"] == eff["plates"]
    pj = json.loads((work_dir / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    assert pj["schema"] == "matte_desk/project@1.0"
    assert pj["plates_hidden"] == ["backlit"]


def test_model_display_is_float_0_1():
    """SAM3 / MatAnyone / zones get display values 0..1 (float), not 8-bit."""
    from engine.pipeline import default_display
    lin = np.array([[[0.0, 0.18, 4.0]]], np.float32)
    d = default_display(lin)
    assert d.dtype == np.float32 and 0.0 <= d.min() and d.max() <= 1.0 and d[0, 0, 2] > 0.9
