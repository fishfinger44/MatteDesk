# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/gpu_manager.py - THE coordinator of everything that computes on the (one) GPU (user decision 2026-10-25:
a GPU manager instead of rewriting the engine; NEXT_STEPS 2026-10-28/29 point 5). It replaces the former stage-release module.

Case it removes (2026-10-25 11:45-12:00): a zone preview "matting 0/5" stood for 17 min – ViTMatte in this process
shared a full card with a CoTracker3 band-point tracking running in the background.

GPU users of one card:
  * ComfyUI (SAM 3.1, MatAnyone2, VideoMaMa, LTX-2.5, DepthCrafter, LaMa) – its memory is released by `POST /free`
    (unload + free), only when its queue is idle (never under someone else's prompt), then waited for (`wait_comfy_freed`);
  * our long-lived workers (CoTracker3, ViTMatte / BiRefNet / Sapiens2 in ComfyUI's Python, the alignment worker) and
    ViTMatte in this process (`VitMatteRunner`, when the venv sees a system torch) – each keeps its model until closed.

The lease: every model call runs inside `GPU.use(key, label, evict)` (the runners' forward / request / call, Comfy's
queue + wait through `comfy_lease`). The OWNER of a lease is a work: the job (engine/jobs, also in executor threads that
carry its context) or else the thread (background band tracking). One work holds the card at a time:
  * another work WAITS until the lease is free (its job shows "czeka na GPU: <who>"; "Przerwij" / `cancel_check` end
    the wait – no deadlock; an exception in the holder releases the lease);
  * a work that gets the card first unloads the models OTHER works left on it (their `evict`); models of the same work
    stay (band tracking + ViTMatte of one zone "Zastosuj" keep both loaded, as before);
  * a resident worker unused for `gpu_idle_release_s` (config, 180 s) is closed by the reaper (Sapiens2 kept 2.8 GB idle).
    Comfy is not reaped (its own memory management).
Stage entries still call `before_stage("comfy" | "local", client, reason)` (STAGE_CALLS, tests/test_gpu_manager.py):
"comfy" unloads all our resident models + Comfy's, "local" Comfy's – both as a lease, so never under another work.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)

# the stage entries that must call before_stage (module, function, kind) – checked by the test
STAGE_CALLS = (
    ("engine/pipeline.py", "run_core", "comfy"),            # 1 SAM 3.1
    ("engine/pipeline.py", "finalize", "comfy"),            # 3 MatAnyone2 (+ experiments' passes)
    ("engine/pipeline.py", "vmm_run", "comfy"),             # 4 VideoMaMa
    ("engine/stage_preview.py", "_videomama", "comfy"),     # 4 VideoMaMa stage preview
    ("engine/pipeline.py", "ltx_run", "comfy"),             # 7 LTX-2.5
    ("engine/depth.py", "run_depth", "comfy"),              # depth (DepthCrafter)
    ("engine/pipeline.py", "release_comfy_gpu", "local"),   # 2 CoTracker3, 5 ViTMatte / BiRefNet, Sapiens2
)

# the model calls that must run under a lease (module, class, method) – checked by the test (AST)
LEASED_CALLS = (
    ("engine/vitmatte.py", "VitMatteRunner", "forward"),            # ViTMatte in this process
    ("engine/vitmatte.py", "ComfyVitMatteRunner", "forward"),       # ViTMatte / BiRefNet / Sapiens2 workers
    ("engine/tracking.py", "_CoTrackerWorker", "request"),          # CoTracker3 (served)
    ("engine/tracking.py", "CoTracker3Tracker", "_one_shot"),       # CoTracker3 (process per call)
    ("engine/align_gpu.py", "GpuAligner", "call"),                  # alignment / motion blur worker
)

COMFY = "comfy"               # the lease key of ComfyUI
IDLE_RELEASE_S = 180.0        # default of config `gpu_idle_release_s`
WAIT_POLL_S = 0.25            # a waiting work checks "Przerwij" / its cancel_check this often


def _idle_s_from_config() -> float:
    try:
        from engine.config import load_config
        return float(load_config().get("gpu_idle_release_s", IDLE_RELEASE_S))
    except Exception:
        return IDLE_RELEASE_S


class GpuManager:
    """One per process (`GPU`); tests make their own with a fake clock and no reaper thread."""

    def __init__(self, idle_s: Optional[float] = None, clock: Callable[[], float] = time.monotonic,
                 reaper: bool = True, jobs: Any = None):
        self._cond = threading.Condition()
        self._holder: Optional[Hashable] = None       # owner of the lease
        self._holder_desc = ""
        self._holder_model = ""
        self._depth = 0
        self._waiting: Dict[int, Dict[str, Any]] = {}  # thread ident -> {owner, label, for, since}
        self._res: Dict[Hashable, Dict[str, Any]] = {}  # resident models: key -> {label, evict, owner, last, busy}
        self._idle_s = idle_s
        self._clock = clock
        self._jobs = jobs
        self._reaper_on = reaper
        self._reaper: Optional[threading.Thread] = None
        self._local = threading.local()               # work label of this thread (GPU.work)

    # ------------------------------------------------------------------ owners / labels
    def _jobs_api(self):
        if self._jobs is not None:
            return self._jobs
        from engine.jobs import JOBS
        return JOBS

    def _owner(self) -> Tuple[Hashable, Optional[int]]:
        jid = self._jobs_api().current_id()
        return (("job", jid) if jid else ("thread", threading.get_ident())), jid

    def _desc(self, jid: Optional[int], label: str) -> str:
        work = getattr(self._local, "work", None)
        if work:
            return f"{work} ({label})"
        if jid:
            j = next((x for x in self._jobs_api().running() if x.get("id") == jid), None)
            if j:
                return f"{j['label']} ({label})"
        return label

    @contextlib.contextmanager
    def work(self, label: str) -> Iterator[None]:
        """Name the work of this thread for the labels ("śledzenie punktów pasa" in a background thread)."""
        prev = getattr(self._local, "work", None)
        self._local.work = label
        try:
            yield
        finally:
            self._local.work = prev

    @property
    def idle_s(self) -> float:
        return float(self._idle_s if self._idle_s is not None else _idle_s_from_config())

    # ------------------------------------------------------------------ the lease
    def _acquire(self, owner: Hashable, jid: Optional[int], desc: str, model: str,
                 cancel_check: Optional[Callable[[], None]]) -> None:
        me = threading.get_ident()
        with self._cond:
            try:
                while self._holder is not None and self._holder != owner:
                    if me not in self._waiting:
                        self._waiting[me] = {"owner": owner, "label": desc, "for": self._holder_desc,
                                             "since": self._clock(), "job": jid}
                        if jid:
                            self._jobs_api().set_gpu_wait(jid, self._holder_desc)
                        log.info("GPU: %s czeka na %s", desc, self._holder_desc)
                    self._waiting[me]["for"] = self._holder_desc
                    self._cond.wait(WAIT_POLL_S)
                    if jid:
                        self._jobs_api().check()          # "Przerwij"
                    if cancel_check:
                        cancel_check()
            finally:
                if self._waiting.pop(me, None) is not None and jid:
                    self._jobs_api().set_gpu_wait(jid, None)
            if self._holder is None:
                self._holder, self._holder_desc, self._holder_model = owner, desc, model
            self._depth += 1

    def _release(self) -> None:
        with self._cond:
            self._depth -= 1
            if self._depth <= 0:
                self._holder, self._holder_desc, self._holder_model, self._depth = None, "", "", 0
                self._cond.notify_all()

    def _evict(self, keys: List[Hashable], why: str) -> List[str]:
        """Unload the residents `keys` (the caller holds the lease). An evict returning False (Comfy busy) keeps it."""
        done = []
        for k in keys:
            with self._cond:
                r = self._res.get(k)
            if r is None or r["evict"] is None:
                continue
            try:
                ok = r["evict"]()
            except Exception as e:      # a worker that will not quit: never fails the work that needs the card
                log.warning("GPU: %s nie zwolniony (%s)", r["label"], e)
                ok = None
            if ok is False:
                continue
            with self._cond:
                self._res.pop(k, None)
            done.append(r["label"])
        if done:
            log.info("GPU: zwolnione %s (%s)", ", ".join(done), why)
        return done

    @contextlib.contextmanager
    def use(self, key: Hashable, label: str, evict: Optional[Callable[[], Any]] = None,
            cancel_check: Optional[Callable[[], None]] = None) -> Iterator[None]:
        """Run a model call on the card: wait for the lease (another work holds it), unload what other works left
        on the card, mark `key` resident (with `evict` – how to unload it) until it is evicted or reaped.
        `evict` None = the call leaves nothing on the card (a process per call): not kept as resident."""
        owner, jid = self._owner()
        desc = self._desc(jid, label)
        self._acquire(owner, jid, desc, label, cancel_check)
        try:
            with self._cond:
                other = [k for k, r in self._res.items() if k != key and r["owner"] != owner and not r["busy"]]
            if other:
                self._evict(other, f"dla: {desc}")
            with self._cond:
                r = self._res.get(key) or {}
                self._res[key] = {"label": label, "evict": evict if evict is not None else r.get("evict"),
                                  "owner": owner, "last": self._clock(), "busy": r.get("busy", 0) + 1}
            self._start_reaper()
            yield
        finally:
            with self._cond:
                r = self._res.get(key)
                if r is not None:
                    r["busy"] = max(0, r["busy"] - 1)
                    r["last"] = self._clock()
                    if r["evict"] is None and not r["busy"]:
                        self._res.pop(key, None)
            self._release()

    @contextlib.contextmanager
    def exclusive(self, label: str, cancel_check: Optional[Callable[[], None]] = None) -> Iterator[None]:
        """The lease without a model (a stage entry releasing memory before it starts)."""
        owner, jid = self._owner()
        self._acquire(owner, jid, self._desc(jid, label), label, cancel_check)
        try:
            yield
        finally:
            self._release()

    def forget(self, key: Hashable) -> None:
        """A model that unloaded itself (worker closed by its owner, e.g. the alignment worker at the end of a job)."""
        with self._cond:
            self._res.pop(key, None)

    # ------------------------------------------------------------------ release
    def release_all(self, keep: Tuple[Hashable, ...] = (), why: str = "") -> List[str]:
        """Unload every resident except `keep` (caller holds the lease)."""
        with self._cond:
            keys = [k for k, r in self._res.items() if k not in keep and not r["busy"]]
        return self._evict(keys, why)

    def reap_idle(self) -> List[str]:
        """Close our workers unused for `idle_s` (not Comfy) – only when nobody holds the card."""
        idle = self.idle_s
        if idle <= 0:
            return []
        with self._cond:
            if self._holder is not None:
                return []
            now = self._clock()
            keys = [k for k, r in self._res.items() if k != COMFY and not r["busy"] and now - r["last"] >= idle]
            if not keys:
                return []
            self._holder, self._holder_desc, self._holder_model, self._depth = ("reaper",), "zwalnianie bezczynnych", "", 1
        try:
            return self._evict(keys, f"bezczynne ≥ {idle:.0f} s")
        finally:
            self._release()

    def _start_reaper(self) -> None:
        if not self._reaper_on or (self._reaper is not None and self._reaper.is_alive()):
            return

        def loop():
            while True:
                time.sleep(5.0)
                try:
                    self.reap_idle()
                except Exception as e:     # noqa: BLE001 – the reaper must never die
                    log.debug("GPU reaper: %s", e)
        self._reaper = threading.Thread(target=loop, name="gpu-reaper", daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------ status (GET /api/jobs -> gpu)
    def snapshot(self) -> Dict[str, Any]:
        now = self._clock()
        with self._cond:
            return {
                "holder": self._holder_desc or None,
                "model": self._holder_model or None,
                "waiting": [{"label": w["label"], "for": w["for"], "job": w["job"], "waited_s": round(now - w["since"], 1)}
                            for w in self._waiting.values()],
                "resident": [{"label": r["label"], "busy": bool(r["busy"]), "idle_s": round(now - r["last"], 1)}
                             for r in self._res.values()],
            }


GPU = GpuManager()


# ---------------------------------------------------------------------------------------------- ComfyUI

def release_comfy(client: Any, reason: str = "") -> bool:
    """`POST /free` (unload models + free memory) when Comfy's queue is idle; True when requested."""
    if client is None or not hasattr(client, "free_vram"):
        return False
    try:
        q = client.get_queue() if hasattr(client, "get_queue") else {}
        if (q or {}).get("busy"):
            return False
        return bool(client.free_vram(unload_models=True, free_memory=True))
    except Exception:
        return False


# Comfy applies /free in its prompt worker loop (main.py: the flags are read AFTER a queued prompt runs), so a prompt
# queued right after /free ran first on the old models (user request 2026-10-26, LTX-2.5 spill check). After /free
# the free VRAM is polled until it has risen and settled, or stays put (nothing was loaded) – then the stage queues.
FREE_WAIT_S = 5.0
FREE_POLL_S = 0.2
FREE_STILL_S = 0.6          # unchanged this long without any rise = nothing to unload
FREE_STEP_B = 64 << 20      # changes below this are noise


def _vram_free(client: Any) -> Optional[int]:
    try:
        st = client.get_system_stats() or {}
        return int(st["devices"][0]["vram_free"])
    except Exception:
        return None


def wait_comfy_freed(client: Any, before: Optional[int], clock=None, sleep=None) -> Optional[float]:
    """Wait (max FREE_WAIT_S) until Comfy has applied /free: GB freed, None when its stats are unavailable."""
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    if before is None or not hasattr(client, "get_system_stats"):
        return None
    t0 = clock()
    last, last_change, rose = before, t0, False
    while clock() - t0 < FREE_WAIT_S:
        sleep(FREE_POLL_S)
        cur = _vram_free(client)
        if cur is None:
            return None
        if abs(cur - last) > FREE_STEP_B:
            rose = rose or cur > before + FREE_STEP_B
            last, last_change = cur, clock()
        elif clock() - last_change >= (FREE_POLL_S * 2 if rose else FREE_STILL_S):
            break                                   # risen and settled, or nothing was loaded
    return round(max(0, last - before) / (1 << 30), 2)


def free_comfy(client: Any, reason: str = "") -> Tuple[bool, Optional[float]]:
    """/free + wait until it took effect: (requested, GB freed). False = Comfy busy / absent (stays resident)."""
    before = _vram_free(client) if client is not None and hasattr(client, "get_system_stats") else None
    ok = release_comfy(client, reason)
    return ok, (wait_comfy_freed(client, before) if ok else None)


def comfy_lease(client: Any, label: str, cancel_check: Optional[Callable[[], None]] = None):
    """The lease of one Comfy prompt (queue_prompt + wait_for_prompt): other works' models leave the card first, and
    another work that needs the card later unloads Comfy's models (`/free`, idle queue only)."""
    return GPU.use(COMFY, f"ComfyUI: {label}", evict=lambda: free_comfy(client, label)[0], cancel_check=cancel_check)


def before_stage(kind: str, client: Any, reason: str = "") -> dict:
    """THE release before a heavy stage: kind "comfy" = our resident models + Comfy, "local" = Comfy only. Runs as a
    lease (waits for another work on the card) and returns once Comfy has really freed its memory."""
    if kind not in ("comfy", "local"):
        raise ValueError(f"before_stage: nieznany rodzaj {kind!r}")
    with GPU.exclusive(reason or kind):
        workers = GPU.release_all(keep=(COMFY,), why=f"przed: {reason or kind}") if kind == "comfy" else []
        comfy, freed = free_comfy(client, reason)
        if comfy:
            GPU.forget(COMFY)
    log.info("GPU zwolnione przed: %s (Comfy: %s, zwolniono %s GB, workery: %s)", reason or kind, comfy,
             freed if freed is not None else "?", ", ".join(workers) or "-")
    return {"comfy": comfy, "workers": workers, "freed_gb": freed}
