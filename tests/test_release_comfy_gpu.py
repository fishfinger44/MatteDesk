# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ObjectPipeline.release_comfy_gpu – Comfy's models are unloaded (POST /free) before GPU work outside Comfy
(CoTracker3, ViTMatte), only when Comfy's queue is idle; no client / busy queue / errors -> no-op."""

from types import SimpleNamespace

from engine.pipeline import ObjectPipeline


class _Client:
    def __init__(self, busy=False, fail=False):
        self.busy = busy
        self.fail = fail
        self.calls = []

    def get_queue(self):
        if self.fail:
            raise RuntimeError("comfy down")
        return {"busy": self.busy, "running": [], "pending": []}

    def free_vram(self, unload_models=True, free_memory=True):
        self.calls.append((unload_models, free_memory))
        return True


def _pipe(client):
    return SimpleNamespace(client=client, release_comfy_gpu=lambda reason="": ObjectPipeline.release_comfy_gpu(SimpleNamespace(client=client), reason))


def test_release_when_idle_only():
    c = _Client()
    assert _pipe(c).release_comfy_gpu("test") is True and c.calls == [(True, True)]
    busy = _Client(busy=True)
    assert _pipe(busy).release_comfy_gpu("test") is False and busy.calls == []
    down = _Client(fail=True)
    assert _pipe(down).release_comfy_gpu("test") is False and down.calls == []
    assert _pipe(None).release_comfy_gpu("test") is False
    assert _pipe(object()).release_comfy_gpu("test") is False      # a client without free_vram (tests' stubs)
