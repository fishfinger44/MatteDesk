# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/jobs.py - Running jobs: what is being computed now (status bar) + cooperative cancel ("Przerwij").

Long operations run in worker threads. They call `JOBS.check()` (directly or through their progress callbacks and
the Comfy wait loop); after `JOBS.cancel()` the next check raises `JobCancelled`, so the operation stops at a safe
point (finished frames / run folders stay as they are; an unfinished run version is not activated).
Comfy: the wait loop also sends /interrupt and removes our queued prompt (see comfy_client.wait_for_prompt).
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Dict, List, Optional


class JobCancelled(RuntimeError):
    """Raised inside a job after the user pressed "Przerwij"."""

    def __init__(self, msg: str = "Przerwane przez użytkownika"):
        super().__init__(msg)


class JobBusy(RuntimeError):
    """Another heavy job is running (one heavy job at a time: GPU / Comfy / shared files)."""

    def __init__(self, job: Dict[str, Any]):
        self.job = job
        super().__init__(f"Trwa: {job['label']} – poczekaj na koniec albo kliknij „Przerwij” na pasku na dole")


class Jobs:
    def __init__(self):
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._jobs: Dict[int, Dict[str, Any]] = {}
        self._cancel = threading.Event()
        self._local = threading.local()

    # ------------------------------------------------------------------ lifecycle
    def try_start(self, kind: str, label: str, obj: Optional[str] = None, quiet: bool = False) -> int:
        """Start a heavy job only if nothing else runs (atomic); otherwise JobBusy.
        quiet = short job (a few seconds, e.g. SAM 3.1 on one frame): the UI shows it on the bar only, no modal."""
        with self._lock:
            run = [j for j in self._jobs.values() if j["state"] == "running"]
            if run:
                raise JobBusy(dict(run[-1]))
        return self.start(kind, label, obj, quiet=quiet)

    def busy_object(self, name: str) -> Optional[Dict[str, Any]]:
        """The running job that works on object `name` (or on all objects, e.g. export / queue), else None."""
        with self._lock:
            for j in self._jobs.values():
                if j["state"] == "running" and (j.get("obj") == name or j.get("obj") == "*"):
                    return dict(j)
        return None

    def running(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(j) for j in self._jobs.values() if j["state"] == "running"]

    def start(self, kind: str, label: str, obj: Optional[str] = None, quiet: bool = False) -> int:
        with self._lock:
            if not any(j["state"] == "running" for j in self._jobs.values()):
                self._cancel.clear()  # a new job after a cancel starts clean
            jid = next(self._ids)
            self._jobs[jid] = {"id": jid, "kind": kind, "label": label, "state": "running", "step": "",
                               "done": 0, "total": 0, "started": time.time(), "updated": time.time(), "comfy": None,
                               "comfy_info": None, "obj": obj, "quiet": bool(quiet), "unit": "frames",
                               "gpu_spill_gb": None}
            return jid

    def bind(self, jid: int) -> None:
        """Mark the current (worker) thread as running job `jid` (progress without an explicit id)."""
        self._local.jid = jid

    def finish(self, jid: int, state: str = "done", error: str = "") -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j:
                j.update(state=state, error=error, ended=time.time())
            # keep only the last few finished jobs
            done = sorted((k for k, v in self._jobs.items() if v["state"] != "running"))
            for k in done[:-5]:
                self._jobs.pop(k, None)

    # ------------------------------------------------------------------ progress / cancel
    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def progress(self, *args) -> None:
        """Progress-callback adapter for any of the tool's callback shapes: (done, total[, frame]) or
        (step, done, total). Checks for cancel first."""
        self.check()
        if args and isinstance(args[0], str):
            self.update(step=args[0], done=args[1] if len(args) > 1 else None, total=args[2] if len(args) > 2 else None)
        elif len(args) >= 2:
            self.update(done=args[0], total=args[1])

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def update(self, step: Optional[str] = None, done: Optional[int] = None, total: Optional[int] = None,
               jid: Optional[int] = None, comfy: Optional[str] = None,
               comfy_info: Optional[Dict[str, Any]] = None, unit: Optional[str] = None,
               gpu_spill_gb: Optional[float] = None) -> None:
        """`unit`: what done / total count – "frames" (default) or "pct" (a job that only reports stage milestones in %:
        the UI shows "postęp N %" and no ETA). `gpu_spill_gb`: GPU memory of Comfy spilled to RAM – the job keeps the peak
        (comfy_client.wait_for_prompt, run.json)."""
        jid = jid or getattr(self._local, "jid", None) or self._latest_running()
        with self._lock:
            j = self._jobs.get(jid) if jid else None
            if j and j["state"] == "running":
                if step is not None:
                    j["step"] = step
                if done is not None:
                    j["done"] = int(done)
                if total is not None:
                    j["total"] = int(total)
                if comfy is not None:
                    j["comfy"] = comfy
                if comfy_info is not None:
                    j["comfy_info"] = comfy_info
                if unit is not None:
                    j["unit"] = unit
                if gpu_spill_gb is not None:
                    j["gpu_spill_gb"] = max(float(gpu_spill_gb), j.get("gpu_spill_gb") or 0.0)
                j["updated"] = time.time()

    def spill_gb(self, jid: Optional[int] = None) -> Optional[float]:
        """Peak GPU memory spilled to RAM in job `jid` (default: this thread's job / the latest running); None = none seen."""
        jid = jid or getattr(self._local, "jid", None) or self._latest_running()
        with self._lock:
            j = self._jobs.get(jid) if jid else None
            return j.get("gpu_spill_gb") if j else None

    def cancel(self) -> bool:
        with self._lock:
            running = any(j["state"] == "running" for j in self._jobs.values())
            if running:
                self._cancel.set()
                for j in self._jobs.values():
                    if j["state"] == "running":
                        j["step"] = "przerywanie…"
            return running

    def snapshot(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            out = []
            for j in sorted(self._jobs.values(), key=lambda x: x["id"]):
                d = dict(j)
                d["elapsed_s"] = round((j.get("ended") or now) - j["started"], 1)
                out.append(d)
            return out

    def _latest_running(self) -> Optional[int]:
        run = [k for k, v in self._jobs.items() if v["state"] == "running"]
        return max(run) if run else None


JOBS = Jobs()
