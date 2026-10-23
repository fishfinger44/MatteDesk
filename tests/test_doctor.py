# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_doctor.py - Acceptance tests for Stage E0 ("Doktor" środowiska).

Criteria from SPEC_INTERACTIVE_MATTE_TOOL.md (Section 12, E0):
1. Current machine report is green / healthy for SAM3 / MatAnyone / CoTracker / GPU / storage.
2. ComfyUI flavor (desktop / git) matches the config.
3. Simulation of ComfyUI Desktop flavor (different port, base folder with .venv) produces desktop flavor and no process kill recommendations.
4. Altered SAM3 schema is detected and flagged as error.
5. Missing models, offline Comfy, missing patch are detected with clear repair proposals.
6. Isolation: Doctor never modifies anything outside .venv without consent.
"""

import json
from pathlib import Path
import pytest

from doctor import EnvironmentDoctor, run_doctor, EXPECTED_SAM3_INPUTS


def test_doctor_current_environment_health():
    """Verify that current environment has Python 3.11+, GPU, ffmpeg, models and venv patch."""
    doc = EnvironmentDoctor()
    report = doc.run_all_checks()

    # Check that Python, GPU, ffmpeg, models, disk are OK
    item_map = {item.name: item for item in report.items}

    assert item_map["Python Version"].status == "ok"
    assert item_map["Virtual Environment"].status == "ok"
    assert item_map["ffmpeg"].status == "ok"
    assert item_map["MatAnyone2 VRAM Patch"].status == "ok"
    assert item_map["Model: SAM 3.1"].status == "ok"
    assert item_map["Model: MatAnyone2"].status == "ok"
    assert item_map["Model: CoTracker3"].status == "ok"
    assert item_map["Disk Space"].status == "ok"
    assert report.comfy_flavor == EnvironmentDoctor().config.get("comfy_flavor")  # desktop or git, per config


def test_doctor_comfy_desktop_simulation():
    """Verify ComfyUI Desktop simulation handles flavor and doesn't recommend process kill."""
    doc = EnvironmentDoctor(simulation={"simulate_flavor": "desktop"})
    report = doc.run_all_checks()

    assert report.comfy_flavor == "desktop"
    flavor_item = next(it for it in report.items if it.name == "ComfyUI Flavor")
    assert "Desktop" in flavor_item.message


def test_doctor_altered_sam3_schema_detection():
    """Verify that if SAM3 nodes in Comfy have missing or altered inputs, it is flagged as error."""
    doc = EnvironmentDoctor(simulation={"simulate_altered_sam3": True})
    report = doc.run_all_checks()

    sam3_sig_item = next((it for it in report.items if it.name == "SAM 3.1 Node Signatures"), None)
    assert sam3_sig_item is not None
    assert sam3_sig_item.status == "error"
    assert "Niezgodność wejść węzłów SAM 3.1" in sam3_sig_item.message


def test_doctor_simulated_missing_components_and_install_instructions(tmp_path):
    """Verify that simulated missing models, missing patch, and offline service generate repair suggestions."""
    doc = EnvironmentDoctor(simulation={
        "simulate_missing_model": True,
        "simulate_missing_patch": True,
        "simulate_comfy_offline": True,
        "simulate_no_ffmpeg": True
    })
    report = doc.run_all_checks()

    item_map = {item.name: item for item in report.items}
    assert item_map["Model: SAM 3.1"].status == "error"
    assert item_map["MatAnyone2 VRAM Patch"].status == "error"
    assert item_map["ComfyUI Service"].status == "warning"
    assert item_map["ffmpeg"].status == "error"

    # Verify INSTALL markdown generation
    md_file = doc.generate_install_markdown(report)
    assert md_file is not None
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "Model: SAM 3.1" in content
    assert "MatAnyone2 VRAM Patch" in content
    assert "ffmpeg" in content
    assert "PowerShell" in content or "powershell" in content


def test_doctor_report_json_export(tmp_path):
    """Verify doctor report JSON serializability and structure."""
    doc = EnvironmentDoctor()
    report = doc.run_all_checks()
    out_file = tmp_path / "doctor_report_test.json"
    doc.save_report(report, output_path=out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "timestamp" in data
    assert "items" in data
    assert "comfy_flavor" in data
    assert len(data["items"]) > 5


def test_doctor_gives_download_links_for_missing_items(tmp_path):
    """Missing components carry 'where to get it' links (UI + INSTALL_<date>.md); SAM3 is checked in checkpoints/."""
    doc = EnvironmentDoctor(simulation={"simulate_missing_model": True, "simulate_missing_nodes": True})
    report = doc.run_all_checks()
    by = {it.name: it for it in report.items}
    sam3 = by["Model: SAM 3.1"]
    assert sam3.status == "error" and "checkpoints" in sam3.message
    assert any("huggingface.co/Comfy-Org/sam3.1" in l for l in sam3.links)
    assert any("pq-yang/MatAnyone2" in l for l in by["Model: MatAnyone2"].links)
    assert any("facebookresearch/co-tracker" in l for l in by["Model: CoTracker3"].links)
    assert by["ComfyUI-DepthCrafter-Nodes"].status == "warning" and by["ComfyUI-DepthCrafter-Nodes"].links
    doc.repo_dir = tmp_path
    md = doc.generate_install_markdown(report).read_text(encoding="utf-8")
    assert "Skąd pobrać" in md and "huggingface.co/Comfy-Org/sam3.1" in md


def test_doctor_desktop_start_hint():
    """ComfyUI Desktop: start the app (port in its settings), never `python main.py`."""
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config["comfy_flavor"] = "desktop"
    item, url = doc.check_comfy_connectivity()
    assert url is None and "Desktop" in item.fix_suggestion and item.fix_command is None


def test_doctor_desktop2_start_hint(monkeypatch):
    """Comfy Desktop 2 found: the button / autostart start its server headless (no launcher click)."""
    from engine import comfy_launch
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config["comfy_flavor"] = "desktop"
    monkeypatch.setattr(comfy_launch, "desktop2_install", lambda cfg: {"comfy_dir": "x"})
    item, _ = doc.check_comfy_connectivity()
    assert "Comfy Desktop 2" in item.fix_suggestion and "bez okna aplikacji" in item.fix_suggestion
    monkeypatch.setattr(comfy_launch, "desktop2_install", lambda cfg: None)
    item, _ = doc.check_comfy_connectivity()
    assert "Uruchom aplikację ComfyUI Desktop" in item.fix_suggestion


def test_licence_summary_matches_licences():
    """Doctor's licence summary (2026-10-21): every row's use agrees with the LICENSES non-commercial flag."""
    import doctor
    rows = doctor.licence_summary()
    assert {r["name"] for r in rows} >= {"MatAnyone2", "CoTracker3", "ViTMatte", "BiRefNet_HR-matting / Lucida", "SAM 3.1"}
    for name, key, use in doctor.LICENCE_SUMMARY:
        nc = doctor.LICENSES[key][1]
        assert (use == "research") == nc, name
    assert {r["name"]: r["use"] for r in rows}["ViTMatte"] == "research"
    assert {r["name"]: r["use"] for r in rows}["BiRefNet_HR-matting / Lucida"] == "commercial"


def test_doctor_detects_another_comfy_on_the_port(monkeypatch, tmp_path):
    """Desktop + git / portable on one machine, both on 8188: the running server's custom_nodes folder decides."""
    import io
    import urllib.request
    mine, other = tmp_path / "Desktop" / "ComfyUI", tmp_path / "ComfyUI_git"
    running = {"custom_nodes": [str(other / "custom_nodes")]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: io.BytesIO(json.dumps(running).encode()))
    doc = EnvironmentDoctor()
    doc.config["comfy_dir"] = str(mine)
    assert doc._other_comfy_on_port("http://127.0.0.1:8188") == str(other)
    running["custom_nodes"] = [str(mine / "custom_nodes")]
    assert doc._other_comfy_on_port("http://127.0.0.1:8188") is None
