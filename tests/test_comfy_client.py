# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_comfy_client.py - Unit tests for comfy_client.py."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from comfy_client import ComfyClient


def test_comfy_client_isolation_workflow_naming(tmp_path):
    comfy_dir = tmp_path / "ComfyUI"
    client = ComfyClient(comfy_dir=str(comfy_dir))

    wf_data = {"nodes": [], "links": []}
    # Test saving without prefix
    path1 = client.save_workflow_ui(wf_data, "test_flow.json")
    assert path1 is not None
    assert path1.name == "matte_desk_test_flow.json"
    assert path1.exists()

    # Test saving with prefix already present
    path2 = client.save_workflow_ui(wf_data, "matte_desk_stage1")
    assert path2 is not None
    assert path2.name == "matte_desk_stage1.json"
    assert path2.exists()


def test_comfy_client_queue_collision_guard():
    client = ComfyClient()

    # Mock get_queue returning busy
    with patch.object(client, "get_queue", return_value={"running": ["job1"], "pending": [], "busy": True}):
        res = client.queue_prompt({"1": {}})
        assert res["success"] is False
        assert res.get("busy") is True
        assert "zajęte" in res.get("error", "").lower() or "zajete" in res.get("error", "").lower()


def test_comfy_client_free_vram_request():
    client = ComfyClient()

    with patch.object(client, "_request", return_value=(200, {"success": True})) as mock_req:
        ok = client.free_vram()
        assert ok is True
        mock_req.assert_called_once_with(
            "/free",
            method="POST",
            data={"unload_models": True, "free_memory": True}
        )
