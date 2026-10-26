# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/gpu_release.py - THE GPU memory release before every heavy stage (user decision 2026-10-26, after LTX-2.5
spilled 10.1 GB of GPU memory into RAM on Woman_01: Comfy still held SAM 3.1 / MatAnyone2 and our own workers
Sapiens2 + CoTracker3 kept their models on the card).

Two kinds of GPU users share one card:
  * ComfyUI (SAM 3.1, MatAnyone2, VideoMaMa, LTX-2.5, DepthCrafter) – released by `POST /free` (unload + free memory),
    only when its queue is idle (never under someone else's running prompt);
  * our long-lived worker processes (CoTracker3, Sapiens2, ViTMatte, BiRefNet) – each keeps its
    model + CUDA context until it is closed; the next request starts a fresh one (a few seconds).

    before_stage("comfy", client, reason)  – a stage computed in Comfy: our workers closed + Comfy's models unloaded
    before_stage("local", client, reason)  – a stage computed by our workers: Comfy's models unloaded (the workers
                                             of that stage stay – it is about to use them)

Every stage entry calls it (tests/test_gpu_release.py checks the list in STAGE_CALLS). A busy CoTracker3 worker (a
band-point edit running beside the job) is left alone. Best effort: a failure never fails the stage.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)

# (label, module, closer) – the long-lived GPU workers of this process (CoTracker3: idle ones only)
WORKERS = (
    ("CoTracker3", "engine.tracking", "close_workers"),
    ("Sapiens2", "engine.sapiens2", "close_runners"),
    ("ViTMatte", "engine.vitmatte", "close_runners"),
    ("BiRefNet", "engine.birefnet", "close_runners"),
)   # the alignment worker is opened / closed per job (engine/align_gpu.open_aligner / close_aligner)

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


def release_workers() -> List[str]:
    """Close our long-lived GPU workers (the idle ones); returns the labels that were asked to close."""
    done = []
    for label, mod, fn in WORKERS:
        try:
            m = importlib.import_module(mod)
            closer = getattr(m, fn)
            closer(idle_only=True) if mod == "engine.tracking" else closer()
            done.append(label)
        except Exception as e:   # a module without its optional deps, a worker that will not quit
            log.debug("GPU: %s nie zamknięty (%s)", label, e)
    return done


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
    import time
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


def before_stage(kind: str, client: Any, reason: str = "") -> dict:
    """THE release before a heavy stage: kind "comfy" = our workers + Comfy, "local" = Comfy only. Returns once Comfy
    has really freed its memory (wait_comfy_freed), not just accepted the request."""
    if kind not in ("comfy", "local"):
        raise ValueError(f"before_stage: nieznany rodzaj {kind!r}")
    workers = release_workers() if kind == "comfy" else []
    before = _vram_free(client) if client is not None and hasattr(client, "get_system_stats") else None
    comfy = release_comfy(client, reason)
    freed = wait_comfy_freed(client, before) if comfy else None
    log.info("GPU zwolnione przed: %s (Comfy: %s, zwolniono %s GB, workery: %s)", reason or kind, comfy,
             freed if freed is not None else "?", ", ".join(workers) or "-")
    return {"comfy": comfy, "workers": workers, "freed_gb": freed}
