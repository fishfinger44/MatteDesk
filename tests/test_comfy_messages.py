# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the user reads when a graph does not get into ComfyUI (MatteDesk-dev issues 11 and 12)."""

import pytest

import comfy_client as cc


def _client(monkeypatch, responses):
    c = cc.ComfyClient.__new__(cc.ComfyClient)
    c.base_url, c.client_id, c.our_prompts, c._sent = "http://127.0.0.1:8000", "t", set(), {}
    it = iter(responses)
    monkeypatch.setattr(c, "_request", lambda *a, **k: next(it))
    return c


def test_comfy_down_is_not_a_rejected_graph(monkeypatch):
    err = "<urlopen error [WinError 10061] Nie można nawiązać połączenia>"
    c = _client(monkeypatch, [(0, err), (0, err)])   # /queue, then /prompt
    with pytest.raises(cc.ComfyOffline) as e:
        c.queue_prompt({"1": {}})
    assert "ComfyUI nie odpowiada (http://127.0.0.1:8000)" in str(e.value) and "Uruchom ComfyUI" in str(e.value)


def test_missing_node_names_what_to_install(monkeypatch):
    resp = {"error": {"type": "missing_node_type", "message": "Node 'LoadMatAnyone2Model' not found.",
                      "extra_info": {"node_id": "1", "class_type": "LoadMatAnyone2Model"}}, "node_errors": {}}
    c = _client(monkeypatch, [(200, {"queue_running": [], "queue_pending": []}), (400, resp)])
    sub = c.queue_prompt({"1": {}})
    assert not sub["success"]
    assert sub["error"].startswith("W ComfyUI brakuje węzła LoadMatAnyone2Model – Doktor → „Zainstaluj”")


def test_other_rejections_unchanged(monkeypatch):
    resp = {"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": {"3": {}}}
    c = _client(monkeypatch, [(200, {"queue_running": [], "queue_pending": []}), (400, resp)])
    assert c.queue_prompt({"1": {}})["error"] == resp
