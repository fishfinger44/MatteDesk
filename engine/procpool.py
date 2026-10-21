# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/procpool.py - Per-frame work on all CPU cores (process pool, one frame per process).

Processes, not threads: alignment's residual checks and patch refinement are Python loops that hold the GIL.
Each worker runs OpenCV and BLAS single-threaded so N processes use N cores instead of fighting over them.
config.json / config.local.json -> `align_workers` (0 or missing = auto: cores - 4, at most 16).
The worker count is also capped by the free RAM (`ram_workers`, `per_worker_bytes` of the caller) and a process that
dies (the OS kills it when memory runs out) becomes a readable `PoolMemoryError` instead of a bare BrokenProcessPool.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence

_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_env_lock = threading.Lock()


class RamLimitError(MemoryError):
    """Not enough RAM for the job; the message is meant for the user (the API answers HTTP 507 with it)."""


class PoolMemoryError(RamLimitError):
    """A pool process died mid-job - almost always the OS killing it for lack of memory."""


def available_ram_bytes() -> Optional[int]:
    """Free physical RAM in bytes, None when it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ave", ctypes.c_ulonglong)]
            m = _MS()
            m.l = ctypes.sizeof(_MS)
            return int(m.avail) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else None
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


RAM_SHARE = 0.6   # of the free RAM the workers may plan to use (the rest: this process, the OS, other programs)


def ram_workers(workers: int, per_worker_bytes: int) -> int:
    """`workers` capped so that `per_worker_bytes` per process fits in RAM_SHARE of the free RAM (never below 1;
    unknown RAM or per_worker_bytes <= 0: unchanged)."""
    if per_worker_bytes <= 0 or workers <= 1:
        return workers
    avail = available_ram_bytes()
    if avail is None:
        return workers
    return max(1, min(int(workers), int(avail * RAM_SHARE // per_worker_bytes)))


def _died() -> PoolMemoryError:
    return PoolMemoryError("Proces roboczy zakończył się nagle – najpewniej zabrakło pamięci RAM. "
                           "Skróć zakres klatek albo ustaw mniejszą liczbę procesów (align_workers).")


def default_workers(per_worker_bytes: int = 0) -> int:
    try:
        from engine.config import load_config
        n = int(load_config().get("align_workers") or 0)
    except Exception:
        n = 0
    return ram_workers(n if n > 0 else max(1, min(16, (os.cpu_count() or 4) - 4)), per_worker_bytes)


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
            try:
                res = futs.pop(0).result()
            except BrokenProcessPool:
                raise _died() from None
            yield i, res
    finally:
        for f in futs:
            f.cancel()


def run_frames(fn: Callable[[Any], Any], jobs: Sequence[Any], workers: Optional[int] = None,
               on_done: Optional[Callable[[int, int], None]] = None, per_worker_bytes: int = 0) -> List[Any]:
    """fn(job) for every job; results in job order. on_done(n_done, job_index) after each finished job – an
    exception raised there (e.g. JobCancelled from "Przerwij") cancels the jobs not started yet and propagates.
    workers <= 1 runs in this process (same results, used by tests and short sequences).
    `per_worker_bytes`: the job's peak RAM per process - the worker count is capped by the free RAM (ram_workers); a
    pool process that dies raises PoolMemoryError."""
    n_w = default_workers(per_worker_bytes) if workers is None else ram_workers(int(workers), per_worker_bytes)
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
    except BrokenProcessPool:
        pool.shutdown(wait=True, cancel_futures=True)
        raise _died() from None
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return out
