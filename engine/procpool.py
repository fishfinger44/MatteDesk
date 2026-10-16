# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/procpool.py - Per-frame work on all CPU cores (process pool, one frame per process).

Processes, not threads: alignment's residual checks and patch refinement are Python loops that hold the GIL.
Each worker runs OpenCV and BLAS single-threaded so N processes use N cores instead of fighting over them.
config.json / config.local.json -> `align_workers` (0 or missing = auto: cores - 4, at most 16).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence

_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_env_lock = threading.Lock()


def default_workers() -> int:
    try:
        from engine.config import load_config
        n = int(load_config().get("align_workers") or 0)
    except Exception:
        n = 0
    return n if n > 0 else max(1, min(16, (os.cpu_count() or 4) - 4))


def _worker_init() -> None:
    import cv2
    cv2.setNumThreads(1)


@contextmanager
def _single_thread_children() -> Iterator[None]:
    """BLAS reads its thread count when numpy is imported, i.e. in the child before any initializer could set it,
    so the variables are set in this process while the children are spawned and restored right after."""
    with _env_lock:
        old = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
        os.environ.update({k: "1" for k in _SINGLE_THREAD_ENV})
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


@contextmanager
def open_pool(workers: int) -> Iterator[Optional[ProcessPoolExecutor]]:
    """A pool for `imap_frames` / `submit` (None when workers <= 1: the work runs in this process).
    On an exception (e.g. "Przerwij") the jobs not started yet are cancelled."""
    if int(workers) <= 1:
        yield None
        return
    pool = ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init)
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def submit(pool: ProcessPoolExecutor, fn: Callable[[Any], Any], job: Any) -> Any:
    """pool.submit with single-threaded BLAS in any child it spawns (processes start on demand)."""
    with _single_thread_children():
        return pool.submit(fn, job)


def imap_frames(fn: Callable[[Any], Any], jobs: Sequence[Any], workers: int,
                pool: Optional[ProcessPoolExecutor] = None, window: Optional[int] = None) -> Iterator[Any]:
    """Yields (job_index, fn(job)) in job order with at most `window` (default 2 x workers) jobs in flight – a
    producer for one consumer such as the GPU worker (engine.align_gpu). workers <= 1 without a pool: in this
    process."""
    if pool is None and int(workers) > 1:
        with open_pool(workers) as own:
            yield from imap_frames(fn, jobs, workers, pool=own, window=window)
        return
    if pool is None:
        for i, job in enumerate(jobs):
            yield i, fn(job)
        return
    window = window or 2 * max(1, int(workers))
    futs: List[Any] = []
    nxt = 0
    try:
        for i in range(len(jobs)):
            while nxt < len(jobs) and len(futs) < window:
                futs.append(submit(pool, fn, jobs[nxt]))
                nxt += 1
            yield i, futs.pop(0).result()
    finally:
        for f in futs:
            f.cancel()


def run_frames(fn: Callable[[Any], Any], jobs: Sequence[Any], workers: Optional[int] = None,
               on_done: Optional[Callable[[int, int], None]] = None) -> List[Any]:
    """fn(job) for every job; results in job order. on_done(n_done, job_index) after each finished job – an
    exception raised there (e.g. JobCancelled from "Przerwij") cancels the jobs not started yet and propagates.
    workers <= 1 runs in this process (same results, used by tests and short sequences)."""
    n_w = default_workers() if workers is None else int(workers)
    if n_w <= 1 or len(jobs) <= 1:
        out = []
        for i, job in enumerate(jobs):
            out.append(fn(job))
            if on_done:
                on_done(i + 1, i)
        return out
    out: List[Any] = [None] * len(jobs)
    pool = ProcessPoolExecutor(max_workers=min(n_w, len(jobs)), initializer=_worker_init)
    try:
        with _single_thread_children():
            futs = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            out[i] = fut.result()
            if on_done:
                on_done(n, i)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return out
