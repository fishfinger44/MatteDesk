# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/sysinfo.py - "Szczegóły" of the running job (Z19): GPU / VRAM / RAM, what Comfy is computing, our step.

Sources: `nvidia-smi` (GPU load %, temperature, memory – Comfy does not report load); Comfy `/system_stats` (RAM, VRAM per
device, torch VRAM) and `/queue` (running / pending prompt ids – a prompt not sent by this tool = foreign); the job
registry (`engine.jobs`: step, frames, Comfy state from `comfy_client.wait_for_prompt`). Missing `nvidia-smi` or a dead
Comfy -> the fields are null, never an exception. The result is cached for <= 1 s (the UI polls every second).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

NVSMI_QUERY = ["--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]


def _num(v: str) -> Optional[float]:
    try:
        return float(v.strip())
    except (TypeError, ValueError):
        return None


def gpu_info(run: Optional[Callable[..., Any]] = None) -> Optional[List[Dict[str, Any]]]:
    """[{name, util_pct, mem_used_mb, mem_total_mb, temp_c}] per GPU from nvidia-smi; None when it is not available."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = (run or subprocess.run)([exe] + NVSMI_QUERY, capture_output=True, text=True, timeout=2.0)
        if getattr(out, "returncode", 1) != 0:
            return None
        gpus = []
        for line in (out.stdout or "").strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            gpus.append({"name": p[0], "util_pct": _num(p[1]), "mem_used_mb": _num(p[2]), "mem_total_mb": _num(p[3]),
                         "temp_c": _num(p[4])})
        return gpus or None
    except Exception:
        return None


def comfy_info(client: Any) -> Dict[str, Any]:
    """RAM / VRAM from /system_stats and the queue with foreign prompts marked; alive False when Comfy is down."""
    out: Dict[str, Any] = {"alive": False, "ram": None, "devices": [], "queue": None}
    if client is None:
        return out
    try:   # short timeout: the UI polls every second, a dead Comfy must not block it
        if hasattr(client, "_request"):
            code, st = client._request("/system_stats", timeout=1.0)
            st = st if code == 200 else None
        else:
            st = client.get_system_stats() if hasattr(client, "get_system_stats") else None
    except Exception:
        st = None
    if not isinstance(st, dict):
        return out
    out["alive"] = True
    sysd = st.get("system", {}) or {}
    if "ram_total" in sysd:
        out["ram"] = {"total": sysd.get("ram_total"), "free": sysd.get("ram_free")}
    for d in st.get("devices", []) or []:
        out["devices"].append({k: d.get(k) for k in ("name", "vram_total", "vram_free", "torch_vram_total", "torch_vram_free")})
    try:
        if hasattr(client, "_request"):
            code, raw = client._request("/queue", timeout=1.0)
            raw = raw if code == 200 and isinstance(raw, dict) else {}
            q = {"running": raw.get("queue_running", []), "pending": raw.get("queue_pending", [])}
        else:
            q = client.get_queue()
        ours = set(getattr(client, "our_prompts", set()) or set())
        running = [p[1] for p in q.get("running", []) if len(p) > 1]
        pending = [p[1] for p in q.get("pending", []) if len(p) > 1]
        out["queue"] = {"running": running, "pending": pending,
                        "foreign_running": [r for r in running if r not in ours],
                        "foreign_pending": [r for r in pending if r not in ours]}
    except Exception:
        pass
    return out


class DetailsCache:
    def __init__(self, ttl_s: float = 1.0):
        self.ttl = ttl_s
        self._lock = threading.Lock()
        self._t = -1e9
        self._val: Optional[Dict[str, Any]] = None

    def get(self, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._val is not None and time.monotonic() - self._t <= self.ttl:
                return self._val
        val = build()
        with self._lock:
            self._val, self._t = val, time.monotonic()
        return val


def job_details(client: Any, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    run = [j for j in jobs if j.get("state") == "running"]
    j = run[-1] if run else None
    job = None
    if j:
        eta = None
        if j.get("total") and j.get("done"):
            eta = round(j["elapsed_s"] / j["done"] * (j["total"] - j["done"]), 1)
        job = {k: j.get(k) for k in ("id", "label", "step", "done", "total", "elapsed_s", "comfy", "comfy_info", "quiet")}
        job["eta_s"] = eta
    return {"gpu": gpu_info(), "comfy": comfy_info(client), "job": job, "t": time.time()}
