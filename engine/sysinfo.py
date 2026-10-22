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
        if j.get("total") and j.get("done") and j.get("unit", "frames") == "frames":   # % milestones: no ETA
            eta = round(j["elapsed_s"] / j["done"] * (j["total"] - j["done"]), 1)
        job = {k: j.get(k) for k in ("id", "label", "step", "done", "total", "elapsed_s", "comfy", "comfy_info", "quiet", "unit")}
        job["eta_s"] = eta
    return {"gpu": gpu_info(), "comfy": comfy_info(client), "job": job, "t": time.time()}


# ---------------------------------------------------------------------------- GPU memory spilled to system RAM (Windows)
# The NVIDIA driver on Windows (WDDM) moves what does not fit in the card to shared system memory: no OOM, the job just
# gets many times slower – or the CUDA allocator aborts the whole Comfy process (beatch 2026-10-18, crop 2784 px).
# /system_stats never shows it (VRAM stays <= the card), the per-process counter "GPU Process Memory\Shared Usage" does.

def pid_on_port(port: int, run: Optional[Callable[..., Any]] = None) -> Optional[int]:
    """PID of the process listening on localhost `port` (Windows, PowerShell); None elsewhere / unknown."""
    import os
    if run is None and os.name != "nt":
        return None
    ps = (f"$c=Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select -First 1;"
          "if($c){$c.OwningProcess}")
    try:
        out = (run or subprocess.run)(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                      timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        v = (out.stdout or "").strip()
        return int(v) if v.isdigit() else None
    except Exception:
        return None


def gpu_shared_gb(pid: int, run: Optional[Callable[..., Any]] = None) -> Optional[float]:
    """GPU memory of process `pid` held in shared system RAM (GB, summed over adapters); None when the counter is not
    available. Get-Counter takes ~1-2 s – call it from a background thread."""
    import os
    if run is None and os.name != "nt":
        return None
    ps = (f"$s=(Get-Counter '\\GPU Process Memory(pid_{int(pid)}_*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples;"
          "if($s){($s | Measure-Object CookedValue -Sum).Sum}")
    try:
        out = (run or subprocess.run)(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                      timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        v = _num((out.stdout or "").strip().replace(",", "."))
        return None if v is None else v / 2 ** 30
    except Exception:
        return None


class SpillSampler:
    """Peak of `read()` (shared GB) sampled every `interval_s` in a background thread while the `with` block runs.
    `spill_gb` = peak - `base` (Comfy holds a little shared memory at idle; default = the value read when the block
    starts – pass the value read before a series of runs so memory kept from an earlier run is not taken as idle);
    None when the counter gave nothing. Never raises."""

    def __init__(self, read: Callable[[], Optional[float]], interval_s: float = 3.0, base: Optional[float] = None):
        self.read = read
        self.interval = interval_s
        self.base: Optional[float] = base
        self._fixed_base = base is not None
        self.peak: Optional[float] = None
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def _sample(self) -> None:
        try:
            v = self.read()
        except Exception:
            v = None
        if v is not None:
            self.peak = v if self.peak is None else max(self.peak, v)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "SpillSampler":
        if not self._fixed_base:
            try:
                self.base = self.read()
            except Exception:
                self.base = None
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=self.interval + 25)
        self._sample()   # one reading right after the prompt (short probes may end between two samples)

    @property
    def spill_gb(self) -> Optional[float]:
        if self.peak is None:
            return None
        return max(0.0, self.peak - (self.base or 0.0))
