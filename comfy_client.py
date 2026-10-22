# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""comfy_client.py - Client for ComfyUI API (port 8001).

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 3, 10, 11):
- Talks to ComfyUI over HTTP API:
    /system_stats, /object_info, /queue, /prompt, /history, /view, /free
- Respects other user jobs: checks /queue before submitting; warns if busy.
- VRAM management: calls POST /free {"unload_models": true, "free_memory": true}.
- Process control: start / restart for ComfyUI from git; non-intrusive notification for Desktop flavor.
- Saves API/UI workflows with prefix matte_desk_*.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# prompts interrupted by "Przerwij" (process-wide): a node without an interrupt check keeps running them
INTERRUPTED_PROMPTS: set = set()

logger = logging.getLogger("matte_desk.comfy")


class ComfyClient:
    OUR_PROMPTS: set = set()   # prompt ids sent by MatteDesk in this process (clients are created per job)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        comfy_dir: Optional[str] = None,
        flavor: str = "desktop",
        client_id: str = "matte_desk"
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.comfy_dir = Path(comfy_dir) if comfy_dir else None
        self.flavor = flavor.lower()
        self.client_id = client_id
        self.our_prompts: set = ComfyClient.OUR_PROMPTS   # shared by all clients of this process (anything else = foreign)
        # failure detection in wait_for_prompt (Z19): Comfy down / prompt lost for this long -> error (not 30 min of silence)
        self.down_after_s = 10.0
        self.lost_after_s = 10.0
        # Comfy that ACCEPTS the connection but does not answer (request timed out) is busy, not down: loading a
        # 96-frame batch of 2688 px float EXR froze its HTTP server for > 10 s (SH010 2026-10-21) and the run was
        # aborted as "ComfyUI nie odpowiada" although the process lived. Busy is tolerated this long.
        self.busy_after_s = 900.0
        self._clock = time.monotonic       # injectable (tests: fake clock)
        self._sleep = time.sleep

    @staticmethod
    def is_timeout(err: Any) -> bool:
        """The `_request` failure text of a request that timed out (connection accepted, no answer)."""
        t = str(err).lower()
        return "timed out" in t or "timeout" in t

    @classmethod
    def from_config(cls) -> "ComfyClient":
        """Host / port / folder from config.json + config.local.json (per machine)."""
        from engine.config import load_config
        c = load_config()
        return cls(host=c["comfy_host"], port=int(c["comfy_port"]), comfy_dir=c.get("comfy_dir") or None,
                   flavor=c.get("comfy_flavor", "desktop"))

    # -------------------------------------------------------------------------
    # Low-level HTTP requests
    # -------------------------------------------------------------------------
    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> Tuple[int, Any]:
        url = f"{self.base_url}{endpoint}"
        req_data = None
        headers = {"User-Agent": "MatteDeskClient"}
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                raw = resp.read()
                try:
                    return status, json.loads(raw.decode("utf-8"))
                except Exception:
                    return status, raw
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                return e.code, json.loads(body)
            except Exception:
                return e.code, str(e)
        except Exception as e:
            return 0, str(e)

    # -------------------------------------------------------------------------
    # API endpoints
    # -------------------------------------------------------------------------
    def is_alive(self, timeout: float = 2.0) -> bool:
        """Check if ComfyUI service is responding."""
        status, _ = self._request("/system_stats", timeout=timeout)
        return status == 200

    def get_system_stats(self) -> Optional[Dict[str, Any]]:
        status, data = self._request("/system_stats")
        return data if status == 200 and isinstance(data, dict) else None

    def get_object_info(self, node_class: Optional[str] = None) -> Optional[Dict[str, Any]]:
        endpoint = f"/object_info/{node_class}" if node_class else "/object_info"
        status, data = self._request(endpoint, timeout=15.0)
        return data if status == 200 and isinstance(data, dict) else None

    def busy_error(self) -> Optional[str]:
        """None when Comfy's queue is empty, else the message for the user. A queue still running a prompt we interrupted
        ("Przerwij") = an older node that does not check the interrupt finishes its pass first."""
        q = self.get_queue() or {}
        if not q.get("busy"):
            return None
        running = {p[1] for p in q.get("running", []) if isinstance(p, (list, tuple)) and len(p) > 1}
        if running & INTERRUPTED_PROMPTS:
            return ("ComfyUI kończy jeszcze przerwane wcześniej zadanie (węzeł nie zatrzymuje się w trakcie przebiegu) – "
                    "poczekaj, aż skończy, albo zrestartuj ComfyUI; Doktor → zaktualizuj nasze węzły (przerywanie od razu)")
        return "ComfyUI zajęte – poczekaj, aż skończy inne zadanie"

    def get_queue(self) -> Dict[str, Any]:
        """Returns {'running': [...], 'pending': [...]}."""
        status, data = self._request("/queue")
        if status == 200 and isinstance(data, dict):
            running = data.get("queue_running", [])
            pending = data.get("queue_pending", [])
            return {"running": running, "pending": pending, "busy": len(running) > 0 or len(pending) > 0}
        return {"running": [], "pending": [], "busy": False, "error": data}

    def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """Free VRAM by asking ComfyUI to unload models and empty torch cache."""
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        status, _ = self._request("/free", method="POST", data=payload)
        return status == 200

    def queue_prompt(self, workflow_api: Dict[str, Any], extra_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit a prompt workflow. Checks queue collision first."""
        queue_info = self.get_queue()
        if queue_info.get("busy"):
            return {
                "success": False,
                "error": "ComfyUI zajęte – kolejka zawiera inne zadania.",
                "busy": True
            }

        payload: Dict[str, Any] = {
            "prompt": workflow_api,
            "client_id": self.client_id
        }
        if extra_data:
            payload["extra_data"] = extra_data

        status, resp = self._request("/prompt", method="POST", data=payload)
        if status == 200 and isinstance(resp, dict):
            prompt_id = resp.get("prompt_id")
            if prompt_id:
                self.our_prompts.add(prompt_id)
            return {"success": True, "prompt_id": prompt_id, "data": resp}
        return {"success": False, "error": resp}

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        status, data = self._request(f"/history/{prompt_id}")
        if status == 200 and isinstance(data, dict):
            return data.get(prompt_id, data)
        return None

    def wait_for_prompt(
        self,
        prompt_id: str,
        timeout_sec: float = 300.0,
        poll_interval: float = 1.0,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Poll until prompt finishes or fails. "Przerwij" (engine.jobs) -> our prompt is removed from the queue or
        interrupted in Comfy and JobCancelled is raised. Failures are detected instead of waiting for the timeout (Z19):
        Comfy not answering for `down_after_s` -> kind "comfy_down"; our prompt neither queued nor in the history for
        `lost_after_s` (Comfy restarted) -> kind "lost"; execution_error -> readable message (GPU / RAM out of memory)
        with the raw text in `raw_error`. A request that TIMES OUT (the connection is accepted, no answer) means
        Comfy is busy, not down: waited for up to `busy_after_s`, then kind "comfy_hung"."""
        from engine.jobs import JOBS, JobCancelled
        clock = self._clock
        start = clock()
        down_since: Optional[float] = None
        busy_since: Optional[float] = None
        lost_since: Optional[float] = None
        while clock() - start < timeout_sec:
            if JOBS.cancelled():
                self.cancel_prompt(prompt_id)
                raise JobCancelled()
            elapsed = clock() - start
            st, data = self._request(f"/history/{prompt_id}")
            if st == 0 and self.is_timeout(data):   # accepted, no answer in time: busy (e.g. loading inputs)
                down_since = None
                busy_since = busy_since if busy_since is not None else clock()
                waited = clock() - busy_since
                JOBS.update(comfy=f"ComfyUI zajęty – nie odpowiada od {waited:.0f}s (wczytuje dane?)",
                            comfy_info={"state": "busy", "prompt": prompt_id})
                if waited >= self.busy_after_s:
                    msg = (f"ComfyUI nie odpowiada od {waited / 60:.0f} min, choć działa (zawieszony?) – "
                           "zrestartuj go i policz ponownie")
                    JOBS.update(comfy=msg, comfy_info={"state": "hung", "prompt": prompt_id, "error": msg})
                    return {"success": False, "error": msg, "kind": "comfy_hung"}
            elif st == 0:   # connection refused: Comfy closed or crashed
                busy_since = None
                down_since = down_since if down_since is not None else clock()
                JOBS.update(comfy=f"ComfyUI nie odpowiada · {clock() - down_since:.0f}s",
                            comfy_info={"state": "down", "prompt": prompt_id})
                if clock() - down_since >= self.down_after_s:
                    msg = "ComfyUI nie odpowiada (zamknięty albo padł) – uruchom go i policz ponownie"
                    JOBS.update(comfy=msg, comfy_info={"state": "down", "prompt": prompt_id, "error": msg})
                    return {"success": False, "error": msg, "kind": "comfy_down"}
            else:
                down_since = busy_since = None
                hist = data.get(prompt_id, data) if st == 200 and isinstance(data, dict) else None
                if hist:
                    status_info = hist.get("status", {}) if isinstance(hist, dict) else {}
                    for msg in status_info.get("messages", []) or []:
                        if msg and msg[0] == "execution_error":
                            readable, kind = self.describe_error(msg[1] if len(msg) > 1 else {})
                            JOBS.update(comfy=readable, comfy_info={"state": "error", "prompt": prompt_id, "error": readable,
                                                                    "raw_error": msg[1] if len(msg) > 1 else None})
                            return {"success": False, "error": readable, "raw_error": msg[1] if len(msg) > 1 else None,
                                    "kind": kind, "history": hist}
                    if status_info.get("completed", False):
                        return {"success": True, "outputs": hist.get("outputs", {}), "history": hist}
                text, info = self._queue_info(prompt_id, elapsed)
                JOBS.update(comfy=text, comfy_info=info)
                if not hist and info.get("state") == "missing":
                    lost_since = lost_since if lost_since is not None else clock()
                    if clock() - lost_since >= self.lost_after_s:
                        msg = "ComfyUI zrestartowany – zadanie utracone, uruchom ponownie"
                        JOBS.update(comfy=msg, comfy_info={**info, "state": "lost", "error": msg})
                        return {"success": False, "error": msg, "kind": "lost"}
                else:
                    lost_since = None

            if progress_callback:
                progress_callback(clock() - start)

            self._sleep(poll_interval)

        return {"success": False, "error": f"Timeout after {timeout_sec}s", "kind": "timeout"}

    @staticmethod
    def describe_error(err: Any) -> Tuple[str, str]:
        """Readable Polish message for a Comfy execution_error (+ kind: oom_gpu / oom_ram / error)."""
        e = err if isinstance(err, dict) else {"exception_message": str(err)}
        node = e.get("node_type") or e.get("class_type") or "?"
        text = f"{e.get('exception_type', '')} {e.get('exception_message', '')}"
        low = text.lower()
        if any(k in low for k in ("outofmemory", "out of memory", "allocation on device")):
            return (f"Brak pamięci GPU w węźle {node}. Zamknij inne programy na GPU albo zmniejsz paczkę/kadr "
                    f"(szczegóły: {e.get('exception_message', '').strip()[:200]})", "oom_gpu")
        if any(k in low for k in ("memoryerror", "bad_alloc", "std::bad_alloc")):
            return (f"Brak pamięci RAM w węźle {node}. Zamknij inne programy albo zmniejsz paczkę/kadr "
                    f"(szczegóły: {e.get('exception_message', '').strip()[:200]})", "oom_ram")
        return (f"Błąd w węźle {node}: {e.get('exception_message', text).strip()[:300]}", "error")

    def _queue_info(self, prompt_id: str, elapsed: float) -> Tuple[str, Dict[str, Any]]:
        """Status text + structured state: ours running / waiting behind a foreign prompt / missing from the queue."""
        try:
            q = self.get_queue()
            if q.get("error") is not None:
                raise RuntimeError(q["error"])
            running = [p[1] for p in q.get("running", []) if len(p) > 1]
            pending = [p[1] for p in q.get("pending", []) if len(p) > 1]
        except Exception:
            return f"Comfy: {elapsed:.0f}s", {"state": "unknown", "prompt": prompt_id}
        foreign = [r for r in running if r != prompt_id and r not in self.our_prompts]
        info: Dict[str, Any] = {"prompt": prompt_id, "running": running, "pending": pending, "foreign_running": foreign}
        if prompt_id in running:
            info["state"] = "ours"
            return f"Comfy liczy · {elapsed:.0f}s", info
        if prompt_id in pending:
            ahead = pending.index(prompt_id) + len(running)
            info.update(state="foreign" if foreign else "queued", ahead=ahead)
            if foreign:
                return (f"Comfy liczy cudze zadanie (prompt {foreign[0][:8]}…), nasze czeka: {ahead} przed nami · "
                        f"{elapsed:.0f}s"), info
            return f"Comfy: w kolejce (przed nami {ahead}) · {elapsed:.0f}s", info
        info["state"] = "missing"
        return f"Comfy · {elapsed:.0f}s", info

    def _queue_state(self, prompt_id: str, elapsed: float) -> str:
        """Short text for the status bar: is our prompt running in Comfy or waiting behind other jobs?"""
        return self._queue_info(prompt_id, elapsed)[0]

    def cancel_prompt(self, prompt_id: str) -> None:
        """Remove our prompt from Comfy's queue; if it is already running, interrupt it (only ours)."""
        try:
            q = self.get_queue()
            running = [p[1] for p in q.get("running", []) if len(p) > 1]
            self._request("/queue", method="POST", data={"delete": [prompt_id]})
            if prompt_id in running:
                self._request("/interrupt", method="POST", data={})
                INTERRUPTED_PROMPTS.add(prompt_id)
        except Exception as e:  # Comfy gone: nothing to stop there
            logger.warning("cancel_prompt %s: %s", prompt_id, e)

    def download_output(self, file_info: Dict[str, Any]) -> bytes:
        """Bytes of an output file listed in history outputs ({filename, subfolder, type}) via /view."""
        q = urllib.parse.urlencode({"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                                    "type": file_info.get("type", "output")})
        with urllib.request.urlopen(f"{self.base_url}/view?{q}", timeout=60) as resp:
            return resp.read()

    # -------------------------------------------------------------------------
    # Process management (ComfyUI from git)
    # -------------------------------------------------------------------------
    def start_git_comfy(self, python_exe: Optional[str] = None) -> bool:
        """Start ComfyUI from git as a background process on self.port."""
        if self.flavor not in ("git", "portable"):   # "portable" = old name of the git variant
            logger.warning("Auto-start is only supported for ComfyUI from git.")
            return False

        if not self.comfy_dir or not (self.comfy_dir / "main.py").exists():
            logger.error(f"Cannot find main.py in {self.comfy_dir}")
            return False

        py_cmd = python_exe or "python"
        cmd = [py_cmd, "main.py", "--port", str(self.port)]
        logger.info(f"Starting ComfyUI: {' '.join(cmd)} in {self.comfy_dir}")

        try:
            subprocess.Popen(
                cmd,
                cwd=str(self.comfy_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            )
            # Wait up to 15 seconds for startup
            for _ in range(15):
                time.sleep(1.0)
                if self.is_alive():
                    return True
        except Exception as e:
            logger.error(f"Failed to start ComfyUI: {e}")

        return self.is_alive()

    def restart_git_comfy(self, python_exe: Optional[str] = None) -> bool:
        """Restart ComfyUI from git by killing process on port and starting new instance.
        
        Strict check: ONLY when queue is empty!
        """
        if self.flavor not in ("git", "portable"):   # "portable" = old name of the git variant
            logger.info("Restart in Comfy Desktop must be done by the user in the desktop application.")
            return False

        queue = self.get_queue()
        if queue.get("busy"):
            logger.warning("Refusing to restart ComfyUI while queue is busy.")
            return False

        # On Windows, kill process listening on port
        if os.name == "nt":
            try:
                # Find PID on port
                netstat_out = subprocess.check_output(
                    f"netstat -ano | findstr :{self.port}",
                    shell=True,
                    text=True,
                    errors="ignore"
                )
                pids = set()
                for line in netstat_out.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and "LISTENING" in line:
                        pids.add(parts[-1])
                for pid in pids:
                    logger.info(f"Terminating ComfyUI PID {pid}")
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        time.sleep(2.0)
        return self.start_git_comfy(python_exe=python_exe)

    # -------------------------------------------------------------------------
    # Workflow saving
    # -------------------------------------------------------------------------
    def save_workflow_ui(self, workflow_data: Dict[str, Any], filename: str) -> Optional[Path]:
        """Save workflow as UI JSON into ComfyUI workflows directory under prefix matte_desk_*.
        
        Complies with isolation rules: workflows named matte_desk_*.json.
        """
        if not filename.startswith("matte_desk_"):
            filename = f"matte_desk_{filename}"
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        if not self.comfy_dir:
            return None

        wf_dir = self.comfy_dir / "user" / "default" / "workflows" / "MatAnyone 2"
        wf_dir.mkdir(parents=True, exist_ok=True)
        target = wf_dir / filename
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            return target
        except Exception as e:
            logger.error(f"Failed to save workflow {target}: {e}")
            return None
