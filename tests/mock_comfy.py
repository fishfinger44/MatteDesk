# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/mock_comfy.py - In-process stand-in for ComfyClient (cloud tests: no ComfyUI, no GPU).

Same public methods as comfy_client.ComfyClient. A graph sent with queue_prompt is "executed" by a handler
registered per node class (e.g. "DepthCrafter", "SAM3_VideoTrack"); the handler receives the whole API graph
and writes whatever files the real nodes would write. Sent graphs are kept in `.sent` for assertions.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional


class MockComfyClient:
    def __init__(self, handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
                 busy: bool = False, object_info: Optional[Dict[str, Any]] = None):
        self.handlers = handlers or {}
        self.busy = busy
        self.sent: List[Dict[str, Any]] = []
        self.freed = 0
        self._results: Dict[str, Dict[str, Any]] = {}
        self._object_info = object_info or {}

    def is_alive(self, timeout: float = 2.0) -> bool:
        return True

    def get_object_info(self, node_class: Optional[str] = None):
        return self._object_info if node_class is None else self._object_info.get(node_class)

    def get_queue(self) -> Dict[str, Any]:
        return {"running": [], "pending": [], "busy": self.busy}

    def busy_error(self) -> Optional[str]:
        return "ComfyUI zajęte – poczekaj, aż skończy inne zadanie" if self.busy else None

    def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        self.freed += 1
        return True

    def queue_prompt(self, workflow_api: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.busy:
            return {"success": False, "error": "ComfyUI zajęte – kolejka zawiera inne zadania.", "busy": True}
        graph = copy.deepcopy(workflow_api)
        self.sent.append(graph)
        pid = uuid.uuid4().hex
        outputs: Dict[str, Any] = {}
        try:
            for node in graph.values():
                h = self.handlers.get(node.get("class_type"))
                if h:
                    outputs.update(h(graph) or {})
            self._results[pid] = {"success": True, "outputs": outputs}
        except Exception as e:  # surfaces like a Comfy execution_error
            self._results[pid] = {"success": False, "error": str(e)}
        return {"success": True, "prompt_id": pid}

    def wait_for_prompt(self, prompt_id: str, timeout_sec: float = 300.0, poll_interval: float = 1.0,
                        progress_callback: Optional[Any] = None) -> Dict[str, Any]:
        return self._results.get(prompt_id, {"success": False, "error": "unknown prompt"})


class MockOutputs:
    """Helper for handlers that emulate SaveImage: writes PNGs to a temp 'output' dir and returns history
    outputs in the Comfy format; MockComfyClient.download_output reads them back."""

    def __init__(self, root):
        from pathlib import Path
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def save_masks(self, node_id: str, prefix: str, masks) -> Dict[str, Any]:
        import cv2
        import numpy as np
        sub = self.root / prefix.rsplit("/", 1)[0] if "/" in prefix else self.root
        sub.mkdir(parents=True, exist_ok=True)
        stem = prefix.rsplit("/", 1)[-1]
        images = []
        for m in masks:
            self.counter += 1
            name = f"{stem}_{self.counter:05d}_.png"
            cv2.imwrite(str(sub / name), (np.asarray(m, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8))
            images.append({"filename": name, "subfolder": prefix.rsplit("/", 1)[0] if "/" in prefix else "",
                           "type": "output"})
        return {node_id: {"images": images}}


def _download(self, file_info):
    root = getattr(self, "output_root", None)
    from pathlib import Path
    return (Path(root) / file_info.get("subfolder", "") / file_info["filename"]).read_bytes()


MockComfyClient.download_output = _download


class FakeComfyServer:
    """Scripted ComfyUI HTTP API for the real `comfy_client.ComfyClient` (Z19 failure detection): replaces
    `client._request`, and drives a fake clock (`client._clock` / `client._sleep`) so 10-s thresholds take no time.

    State: `alive` (False -> every request fails like a refused connection), `running` / `pending` prompt ids,
    `history` {prompt_id: entry}, `stats` (/system_stats). `at` = {time_s: callable(server)} events on the clock."""

    def __init__(self, client, at: Optional[Dict[float, Callable[["FakeComfyServer"], None]]] = None):
        self.t = 0.0
        self.alive = True
        self.busy = False           # True -> every request times out (Comfy alive but its HTTP server blocked)
        self.running: List[str] = []
        self.pending: List[str] = []
        self.history: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {"system": {"ram_total": 64 * 2 ** 30, "ram_free": 40 * 2 ** 30},
                                      "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3080 Ti", "vram_total": 12 * 2 ** 30,
                                                   "vram_free": 5 * 2 ** 30, "torch_vram_total": 9 * 2 ** 30,
                                                   "torch_vram_free": 2 ** 30}]}
        self.at = dict(at or {})
        self.requests: List[str] = []
        client._request = self.request
        client._clock = lambda: self.t
        client._sleep = self.sleep

    def sleep(self, s: float) -> None:
        t0, self.t = self.t, self.t + s
        for k in sorted(self.at):
            if t0 < k <= self.t:
                self.at.pop(k)(self)

    def request(self, endpoint: str, method: str = "GET", data=None, timeout: float = 10.0):
        self.requests.append(f"{method} {endpoint}")
        if not self.alive:
            return 0, "[WinError 10061] connection refused"
        if self.busy:
            self.sleep(min(timeout, 10.0))   # the request itself takes its timeout (scheduled events fire)
            return 0, "<urlopen error timed out>"
        if endpoint == "/system_stats":
            return 200, self.stats
        if endpoint == "/queue" and method == "GET":
            return 200, {"queue_running": [[0, p, {}, {}, []] for p in self.running],
                         "queue_pending": [[i + 1, p, {}, {}, []] for i, p in enumerate(self.pending)]}
        if endpoint.startswith("/history/"):
            pid = endpoint.split("/", 2)[2]
            return 200, ({pid: self.history[pid]} if pid in self.history else {})
        if endpoint == "/prompt" and method == "POST":
            pid = f"ours{len(self.requests)}"
            self.pending.append(pid)
            return 200, {"prompt_id": pid}
        return 200, {}
