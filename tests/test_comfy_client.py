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


def test_busy_error_names_an_interrupted_prompt_still_running():
    """2026-10-17: after "Przerwij" an older MatAnyone node kept computing its pass (no interrupt check); every new run
    failed with a bare "ComfyUI zajęte" – the message must say the cancelled job is still finishing."""
    import comfy_client as cc
    client = cc.ComfyClient("http://127.0.0.1:1")
    q = {"running": [[1, "pid-old", {}]], "pending": [], "busy": True}
    with patch.object(client, "get_queue", return_value=q):
        assert client.busy_error() == "ComfyUI zajęte – poczekaj, aż skończy inne zadanie"
        cc.INTERRUPTED_PROMPTS.add("pid-old")
        try:
            assert "przerwane" in client.busy_error()
        finally:
            cc.INTERRUPTED_PROMPTS.discard("pid-old")
    with patch.object(client, "get_queue", return_value={"running": [], "pending": [], "busy": False}):
        assert client.busy_error() is None


def test_matanyone_node_checks_interrupt_every_frame():
    src = open("handoff/comfy/custom_nodes/ComfyUI-MatAnyone2-Keyframes/__init__.py", encoding="utf-8").read()
    loop = src.split("for ti in range(len(order) + warmup_frames):", 1)[1][:200]
    assert "throw_exception_if_processing_interrupted" in loop
