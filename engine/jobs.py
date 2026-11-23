# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/jobs.py - Running jobs: what is being computed now (status bar) + cooperative cancel ("Przerwij").

Long operations run in worker threads. They call `JOBS.check()` (directly or through their progress callbacks and
the Comfy wait loop); after `JOBS.cancel()` the next check raises `JobCancelled`, so the operation stops at a safe
point (finished frames / run folders stay as they are; an unfinished run version is not activated).
Comfy: the wait loop also sends /interrupt and removes our queued prompt (see comfy_client.wait_for_prompt).
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import time
from typing import Any, Dict, List, Optional


# the job of the current request (app.long_job sets it; executor threads run in a copy of the request's context, so
# they see it; plain threads started from a job do not – they are other work, e.g. a background band tracking)
CURRENT_JOB: contextvars.ContextVar = contextvars.ContextVar("matte_desk_in_job", default=None)


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
        self._thumbs: Dict[int, Dict[str, Any]] = {}   # job id -> last finished frame's mask (small uint8) + crop

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
                               "gpu_spill_gb": None, "gpu_wait": None}
            return jid

    def current_id(self) -> Optional[int]:
        """The job this thread works for: bound (`bind`) or the request's (CURRENT_JOB); None = not a job."""
        return getattr(self._local, "jid", None) or CURRENT_JOB.get()

    def set_gpu_wait(self, jid: int, holder: Optional[str]) -> None:
        """Job `jid` waits for the GPU held by `holder` (engine/gpu_manager); None = no longer waiting."""
        with self._lock:
            j = self._jobs.get(jid)
            if j and j["state"] == "running":
                j["gpu_wait"] = holder
                j["updated"] = time.time()

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
            self._thumbs.pop(jid, None)

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

    THUMB_SIDE = 320       # px, longer side of the stored mask
    THUMB_EVERY_S = 0.4    # at most this often per job (the UI polls once a second)

    def thumb(self, frame: int, mask: Any, crop: Optional[Any] = None, jid: Optional[int] = None) -> None:
        """The progress window's thumbnail (user request 2026-11-23): the mask of the frame the job has just finished.
        `mask` = 2-D array (bool / uint8 0..255 / float 0..1) of the full frame or of `crop` = [x, y, w, h] in plate px
        (x, y may be < 0: a padded crop square). Kept small, in memory; the job dict gets {"frame", "seq"} only."""
        jid = jid or getattr(self._local, "jid", None) or self._latest_running()
        now = time.time()
        with self._lock:
            j = self._jobs.get(jid) if jid else None
            old = self._thumbs.get(jid, {})
            if not j or j["state"] != "running" or now - old.get("t", 0.0) < self.THUMB_EVERY_S:
                return
            self._thumbs[jid] = {**old, "t": now}   # claimed: a parallel writer within the interval skips
        import cv2
        import numpy as np
        m = np.asarray(mask)
        if m.ndim != 2 or not m.size:
            return
        m = (m.astype(np.uint8) * 255 if m.dtype == bool else
             m if m.dtype == np.uint8 else (np.clip(m.astype(np.float32), 0, 1) * 255.0 + 0.5).astype(np.uint8))
        s = min(1.0, self.THUMB_SIDE / float(max(m.shape)))
        if s < 1.0:
            m = cv2.resize(m, (max(1, round(m.shape[1] * s)), max(1, round(m.shape[0] * s))), interpolation=cv2.INTER_AREA)
        with self._lock:
            j = self._jobs.get(jid)
            if not j or j["state"] != "running":
                return
            seq = int((j.get("thumb") or {}).get("seq", 0)) + 1
            self._thumbs[jid] = {"t": now, "frame": int(frame), "mask": m,
                                 "crop": [int(v) for v in crop] if crop is not None else None}
            j["thumb"] = {"frame": int(frame), "seq": seq}

    def thumb_of(self, jid: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """{"frame", "mask", "crop"} of the job's latest thumbnail (default: the latest running job), None = none yet."""
        with self._lock:
            jid = jid or self._latest_running()
            t = self._thumbs.get(jid) if jid else None
            return dict(t) if t and "mask" in t else None

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
