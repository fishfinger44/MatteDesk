# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/conftest.py - Pytest global configuration.

Markers:
- `real`: needs the local workstation (SH010 plates on D:, ComfyUI on 8001, GPU, running server).
  Auto-applied to tests whose source references MD_REAL_PROJECT / REAL_SH010 or that probe the local
  environment. Skipped automatically when D:/MD_REAL_PROJECT is not present (e.g. cloud sessions).
  Run them locally with:  pytest -m real
"""

import inspect
import os
from pathlib import Path

import pytest

# Enable OpenEXR support in OpenCV before any other module imports cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Plate alignment: the CPU code unless a test asks for the GPU (measure_sequence(device="gpu"/"auto"));
# config `align_device` = auto would otherwise send these tests to the GPU worker on a workstation.
os.environ.setdefault("MD_ALIGN_DEVICE", "cpu")
os.environ.setdefault("MD_MODEL_GATE", "0")      # model gate of the stages (engine/stage_models.py): off – synthetic tests have no models
os.environ.setdefault("MD_MB_DEVICE", "cpu")      # motion blur of the alpha: the same (tests/test_motion_blur.py asks for gpu)
os.environ.setdefault("MD_SHARED_PROBE", "0")   # spilled-GPU-memory counter in the MatAnyone2 probe (engine/refine.py): off, no PowerShell in tests

REAL_DATA_ROOT = Path(os.environ.get("MATTE_DESK_REAL_ROOT", "C:\\shots\\SH010\\frame_####.exr"))

# Tests that need the local environment but do not mention MD_REAL_PROJECT in their source.
_LOCAL_ENV_TESTS = {
    "test_doctor_current_environment_health",
    "test_acceptance_e0_doctor_report",
    "test_acceptance_e1_server_running_and_responsive",
}


def _needs_real(item) -> bool:
    if item.name in _LOCAL_ENV_TESTS:
        return True
    try:
        src = inspect.getsource(item.function)
    except (OSError, TypeError):
        return False
    return "MD_REAL_PROJECT" in src or "REAL_SH010" in src or "X1_PATH" in src or "X2_PATH" in src


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs local plates / ComfyUI / GPU (skipped in the cloud)")


def pytest_collection_modifyitems(config, items):
    have_real = REAL_DATA_ROOT.exists()
    skip_real = pytest.mark.skip(reason=f"real data not available ({REAL_DATA_ROOT} missing) – run locally: pytest -m real")
    for item in items:
        if _needs_real(item):
            item.add_marker(pytest.mark.real)
            if not have_real:
                item.add_marker(skip_real)


@pytest.fixture(scope="session")
def _local_config_copy(tmp_path_factory):
    """Copy of this machine's config.local.json (comfy port, paths) for the test session."""
    import engine.config as ec
    dst = tmp_path_factory.mktemp("cfg") / "config.local.json"
    if ec.LOCAL.exists():
        dst.write_bytes(ec.LOCAL.read_bytes())
    return dst


@pytest.fixture(autouse=True)
def _isolate_local_config(_local_config_copy, monkeypatch):
    """Tests read the machine config but never write the real config.local.json
    (opening a project records last_work_dir / recent_work_dirs there)."""
    import engine.config as ec
    monkeypatch.setattr(ec, "LOCAL", _local_config_copy)


@pytest.fixture(autouse=True)
def _no_comfy_autostart(monkeypatch):
    """The server's lifespan (`with TestClient(app)`) must never start a real ComfyUI: a test config without a flavor
    = Desktop, and test_last_project_reopens_on_startup opened the ComfyUI Desktop app on the workstation (2026-10-15)."""
    from engine import comfy_launch as cl
    orig = cl.autostart
    monkeypatch.setattr(cl, "autostart", lambda cfg: orig({**cfg, "comfy_autostart": False}))
