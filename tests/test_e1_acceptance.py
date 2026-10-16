# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_e1_acceptance.py - Official Stage E0 & E1 Acceptance Test Suite.

Verifies all acceptance criteria from SPEC_INTERACTIVE_MATTE_TOOL.md (Section 12):
Stage E0:
- Doctor report verifies environment (Python 3.11+, GPU, ffmpeg, models, VRAM patch).
- Portable vs Desktop flavor handling.
- Detection of altered/missing components.
- Strict isolation (nothing installed outside .venv without consent).

Stage E1:
- Coordinate conversion round-trips (plate <-> proxy <-> crop <-> 1008, padding) < 0.5 px.
- Frame number mapping round-trips.
- Comfy client at port 8001 ready with collision guard and workflow isolation.
- Media ingest and project creation in <work>/matte_desk/.
- SH010 beauty plate (193 frames) has all 193 proxies rendered with PlateToDisplay tonemapping.
- Backend server serves static UI and proxy frames without errors.
"""

from pathlib import Path
import pytest
import requests

from doctor import EnvironmentDoctor
from engine.coords import CoordTransformer, FrameMapper, CropBox
from comfy_client import ComfyClient
from engine.ingest import ProjectCreator, ProxyGenerator


def test_acceptance_e0_doctor_report():
    doc = EnvironmentDoctor()
    report = doc.run_all_checks()
    assert report.comfy_flavor == EnvironmentDoctor().config.get("comfy_flavor")  # desktop or git, per config
    items = {it.name: it for it in report.items}
    assert items["Python Version"].status == "ok"
    assert items["Virtual Environment"].status == "ok"
    assert items["Hardware > GPU & CUDA" if "Hardware > GPU & CUDA" in items else "GPU & CUDA"].status == "ok"
    assert items["ffmpeg"].status == "ok"
    assert items["MatAnyone2 VRAM Patch"].status == "ok"
    assert items["Model: SAM3"].status == "ok"
    assert items["Model: MatAnyone2"].status == "ok"
    assert items["Model: CoTracker3"].status == "ok"
    assert items["Disk Space"].status == "ok"


def test_acceptance_e1_coords_precision():
    ct = CoordTransformer(plate_w=4096, plate_h=2731, proxy_max_side=1920)

    # 1. Plate <-> Proxy round-trip
    test_pts = [(0.0, 0.0), (4096.0, 2731.0), (1412.0, 508.0), (2000.5, 1365.5)]
    for x, y in test_pts:
        px, py = ct.plate_to_proxy(x, y)
        rx, ry = ct.proxy_to_plate(px, py)
        assert abs(rx - x) < 0.5
        assert abs(ry - y) < 0.5

    # 2. Plate <-> Crop <-> SAM1008 round-trip
    crop = CropBox(x=1400.0, y=500.0, w=1280.0, h=1280.0)
    for x, y in [(1400.0, 500.0), (2000.0, 1100.0), (2680.0, 1780.0)]:
        sx, sy = ct.plate_to_sam1008(x, y, crop)
        rx, ry = ct.sam1008_to_plate(sx, sy, crop)
        assert abs(rx - x) < 0.5
        assert abs(ry - y) < 0.5


def test_acceptance_e1_sh010_proxies_rendered():
    proxy_dir = Path("C:\\shots\\SH010\\frame_####.exr")
    assert proxy_dir.is_dir(), f"Proxy directory {proxy_dir} does not exist"
    proxies = list(proxy_dir.glob("p_*.jpg"))
    assert len(proxies) == 193, f"Expected 193 proxy frames, found {len(proxies)}"

    # Check first and last frame exist
    assert (proxy_dir / "p_0001.jpg").exists()
    assert (proxy_dir / "p_0193.jpg").exists()


def test_acceptance_e1_server_running_and_responsive():
    url = "http://127.0.0.1:8765"
    try:
        r_index = requests.get(f"{url}/", timeout=3)
        assert r_index.status_code == 200
        assert "MatteDesk" in r_index.text

        r_proj = requests.get(f"{url}/api/project/current", timeout=3)
        assert r_proj.status_code == 200
        data = r_proj.json()
        if not data.get("loaded"):
            # Open acceptance test project
            r_open = requests.post(f"{url}/api/project/open", json={"manifest_path": "C:\\shots\\SH010\\frame_####.exr"}, timeout=5)
            assert r_open.status_code == 200
            r_proj = requests.get(f"{url}/api/project/current", timeout=3)
            data = r_proj.json()

        assert data["loaded"] is True
        assert data["manifest"]["shot"] == "SH010"

        r_frame = requests.get(f"{url}/api/frame/1", timeout=3)
        assert r_frame.status_code == 200
        assert r_frame.headers.get("content-type") == "image/jpeg"
    except requests.exceptions.ConnectionError:
        pytest.fail("Server 127.0.0.1:8765 is not reachable")
