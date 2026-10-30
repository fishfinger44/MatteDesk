# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/gpu_manager – THE coordinator of the GPU (user decision 2026-10-25, NEXT_STEPS 2026-10-28/29 point 5).

Contract: one work on the card at a time – a second work WAITS (with a label) instead of sharing the card; the models
another work left are unloaded first; an idle worker is closed after `idle_s`; Comfy never gets /free with a busy
queue; an exception / "Przerwij" / a cancel_check never leaves the lease taken (no deadlock); every model call runs
under the lease. Plus the stage entries' release (before_stage, from the former engine/gpu_release)."""

import ast
import threading
import time
from pathlib import Path

import pytest

from engine import gpu_manager
from engine.gpu_manager import COMFY, LEASED_CALLS, STAGE_CALLS, GpuManager, before_stage, comfy_lease
from engine.jobs import Jobs, JobCancelled

ROOT = Path(__file__).resolve().parents[1]
T = 5.0     # max seconds any thread step may take (a hang fails instead of blocking the run)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Client:
    def __init__(self, busy=False):
        self.busy = busy
        self.calls = []

    def get_queue(self):
        return {"busy": self.busy}

    def free_vram(self, unload_models=True, free_memory=True):
        self.calls.append((unload_models, free_memory))
        return True


def _mgr(**kw):
    return GpuManager(idle_s=kw.pop("idle_s", 60.0), clock=kw.pop("clock", time.monotonic), reaper=False,
                      jobs=kw.pop("jobs", Jobs()), **kw)


def _until(cond, what):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < T, f"timeout: {what}"
        time.sleep(0.01)


def _hold(g, key, label, evict, started, release, jobs=None, jid=None, errors=None):
    """A thread holding the card until `release` is set (a job thread when `jid`)."""
    def run():
        if jid:
            jobs.bind(jid)
        try:
            with g.use(key, label, evict):
                started.set()
                assert release.wait(T)
        except BaseException as e:   # noqa: BLE001
            if errors is not None:
                errors.append(e)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


# --------------------------------------------------------------------------------------------- the queue

def test_second_work_waits_with_a_label_then_unloads_the_first():
    jobs = Jobs()
    g = _mgr(jobs=jobs)
    evicted = []
    started, release = threading.Event(), threading.Event()
    bg = _hold(g, "cot", "CoTracker3", lambda: evicted.append("cot"), started, release)   # background band tracking
    assert started.wait(T)

    jid = jobs.start("zone_preview", "Podgląd strefy: Woman", "Woman", quiet=True)
    order = []

    def preview():
        jobs.bind(jid)
        with g.use("vit", "ViTMatte", lambda: evicted.append("vit")):
            order.append("vit")
    pv = threading.Thread(target=preview, daemon=True)
    pv.start()
    _until(lambda: g.snapshot()["waiting"], "preview waits")
    snap = g.snapshot()
    assert snap["holder"] == "CoTracker3" and snap["waiting"][0]["for"] == "CoTracker3"
    assert "Podgląd strefy: Woman" in snap["waiting"][0]["label"]
    assert jobs.snapshot()[0]["gpu_wait"] == "CoTracker3"                 # "czeka na GPU: CoTracker3" in the UI
    assert order == []                                                      # never on the card together
    release.set()
    bg.join(T)
    pv.join(T)
    assert order == ["vit"] and evicted == ["cot"]                          # the other work's model left the card first
    assert jobs.snapshot()[0]["gpu_wait"] is None and g.snapshot()["holder"] is None
    assert [r["label"] for r in g.snapshot()["resident"]] == ["ViTMatte"]


def test_same_work_keeps_its_models_and_reenters():
    g = _mgr()
    evicted = []
    with g.use("cot", "CoTracker3", lambda: evicted.append("cot")):
        with g.use("vit", "ViTMatte", lambda: evicted.append("vit")):      # nested in the same work: no wait
            pass
    with g.use("vit", "ViTMatte"):
        pass
    assert evicted == [] and sorted(r["label"] for r in g.snapshot()["resident"]) == ["CoTracker3", "ViTMatte"]


def test_job_threads_share_the_job_lease():
    """Executor threads of one request see its job (CURRENT_JOB) – the same work, they do not queue behind each other."""
    jobs = Jobs()
    g = _mgr(jobs=jobs)
    jid = jobs.start("apply", "Zastosuj", "Woman")
    started, release = threading.Event(), threading.Event()
    th = _hold(g, "vit", "ViTMatte", None, started, release, jobs, jid)
    assert started.wait(T)
    jobs.bind(jid)
    try:
        with g.use("cot", "CoTracker3"):       # same job, another thread: enters at once
            pass
    finally:
        jobs.bind(None)
        release.set()
        th.join(T)
    assert g.snapshot()["holder"] is None


# --------------------------------------------------------------------------------------------- idle release

def test_idle_worker_released_after_idle_s():
    clk = _Clock()
    g = _mgr(clock=clk, idle_s=180.0)
    evicted = []
    with g.use("s2", "Sapiens2-seg-0.4B", lambda: evicted.append("s2")):
        pass
    with g.use(COMFY, "ComfyUI: SAM 3.1", lambda: evicted.append("comfy")):
        pass
    clk.t = 179.0
    assert g.reap_idle() == [] and evicted == []
    clk.t = 181.0
    assert g.reap_idle() == ["Sapiens2-seg-0.4B"] and evicted == ["s2"]    # Comfy is never reaped
    assert [r["label"] for r in g.snapshot()["resident"]] == ["ComfyUI: SAM 3.1"]


def test_reaper_never_unloads_under_a_running_work():
    clk = _Clock()
    g = _mgr(clock=clk, idle_s=10.0)
    evicted = []
    with g.use("cot", "CoTracker3", lambda: evicted.append("cot")):
        pass
    clk.t = 100.0
    with g.use("vit", "ViTMatte", lambda: evicted.append("vit")):           # same thread = same work, card held
        assert g.reap_idle() == []
    assert evicted == []
    assert g.reap_idle() == ["CoTracker3"]                                 # idle 100 s; ViTMatte just used
    clk.t = 200.0
    assert g.reap_idle() == ["ViTMatte"] and evicted == ["cot", "vit"]


# --------------------------------------------------------------------------------------------- Comfy

def test_comfy_busy_queue_is_never_freed():
    jobs = Jobs()
    g = _mgr(jobs=jobs)
    c = _Client(busy=True)
    with g.use(COMFY, "ComfyUI: SAM 3.1", lambda: gpu_manager.free_comfy(c)[0]):
        pass
    done = threading.Event()

    def other():           # another work (thread) needs the card: Comfy's queue is busy -> no /free, Comfy stays known
        with g.use("vit", "ViTMatte"):
            done.set()
    threading.Thread(target=other, daemon=True).start()
    assert done.wait(T)
    assert c.calls == [] and "ComfyUI: SAM 3.1" in [r["label"] for r in g.snapshot()["resident"]]
    c.busy = False
    done.clear()
    threading.Thread(target=lambda: (g.use(COMFY, "x").__enter__(), done.set()), daemon=True).start()
    assert done.wait(T)


def test_comfy_unloaded_when_idle_for_another_work():
    g = _mgr()
    c = _Client()
    with g.use(COMFY, "ComfyUI: MatAnyone2", lambda: gpu_manager.free_comfy(c)[0]):
        pass
    done = threading.Event()

    def other():
        with g.use("cot", "CoTracker3"):
            done.set()
    threading.Thread(target=other, daemon=True).start()
    assert done.wait(T) and c.calls == [(True, True)]


# --------------------------------------------------------------------------------------------- no deadlocks

def test_exception_releases_the_lease():
    g = _mgr()
    with pytest.raises(RuntimeError):
        with g.use("vit", "ViTMatte"):
            raise RuntimeError("CUDA out of memory")
    done = threading.Event()
    threading.Thread(target=lambda: (g.use("cot", "CoTracker3").__enter__(), done.set()), daemon=True).start()
    assert done.wait(T)


def test_cancel_ends_the_wait():
    jobs = Jobs()
    g = _mgr(jobs=jobs)
    started, release = threading.Event(), threading.Event()
    bg = _hold(g, "cot", "CoTracker3", None, started, release)
    assert started.wait(T)
    jid = jobs.start("zone_preview", "Podgląd strefy", "Woman", quiet=True)
    errs = []

    def preview():
        jobs.bind(jid)
        try:
            with g.use("vit", "ViTMatte"):
                errs.append("ran")
        except JobCancelled as e:
            errs.append(e)
    pv = threading.Thread(target=preview, daemon=True)
    pv.start()
    _until(lambda: g.snapshot()["waiting"], "preview waits")
    jobs.cancel()                                                    # "Przerwij"
    pv.join(T)
    assert not pv.is_alive() and isinstance(errs[0], JobCancelled)
    assert g.snapshot()["waiting"] == [] and jobs.snapshot()[0]["gpu_wait"] is None
    release.set()
    bg.join(T)
    assert g.snapshot()["holder"] is None


def test_cancel_check_ends_a_background_wait():
    """A superseded background band tracking stops waiting (bandtrack.Superseded through its cancel_check)."""
    g = _mgr()
    started, release = threading.Event(), threading.Event()
    th = _hold(g, "vit", "ViTMatte", None, started, release)
    assert started.wait(T)
    stop = threading.Event()
    errs = []

    def check():
        if stop.is_set():
            raise RuntimeError("superseded")

    def tracking():
        try:
            with g.use("cot", "CoTracker3", cancel_check=check):
                errs.append("ran")
        except RuntimeError as e:
            errs.append(str(e))
    tr = threading.Thread(target=tracking, daemon=True)
    tr.start()
    _until(lambda: g.snapshot()["waiting"], "tracking waits")
    stop.set()
    tr.join(T)
    assert errs == ["superseded"] and g.snapshot()["waiting"] == []
    release.set()
    th.join(T)


def test_failing_evict_never_fails_the_work():
    g = _mgr()

    def boom():
        raise RuntimeError("worker hung")
    with g.use("s2", "Sapiens2", boom):
        pass
    done = threading.Event()

    def other():
        with g.use("vit", "ViTMatte"):
            done.set()
    threading.Thread(target=other, daemon=True).start()
    assert done.wait(T)


# --------------------------------------------------------------------------------------------- stage entries

@pytest.fixture
def fresh_gpu(monkeypatch):
    g = _mgr()
    monkeypatch.setattr(gpu_manager, "GPU", g)
    return g


def test_comfy_stage_unloads_our_models_and_comfy(fresh_gpu):
    evicted = []
    with fresh_gpu.use("cot", "CoTracker3", lambda: evicted.append("cot")):
        pass
    with fresh_gpu.use("s2", "Sapiens2", lambda: evicted.append("s2")):
        pass
    c = _Client()
    r = before_stage("comfy", c, "LTX-2.5")
    assert c.calls == [(True, True)] and r["comfy"] is True
    assert sorted(evicted) == ["cot", "s2"] and sorted(r["workers"]) == ["CoTracker3", "Sapiens2"]
    assert fresh_gpu.snapshot()["resident"] == []


def test_local_stage_unloads_comfy_only(fresh_gpu):
    evicted = []
    with fresh_gpu.use("vit", "ViTMatte", lambda: evicted.append("vit")):
        pass
    c = _Client()
    r = before_stage("local", c, "ViTMatte")
    assert c.calls == [(True, True)] and evicted == [] and r["workers"] == []


def test_busy_comfy_queue_stage(fresh_gpu):
    c = _Client(busy=True)
    assert before_stage("comfy", c)["comfy"] is False and c.calls == []
    assert before_stage("comfy", None)["comfy"] is False
    with pytest.raises(ValueError):
        before_stage("gpu", c)


def test_comfy_lease_marks_comfy_resident(fresh_gpu):
    c = _Client()
    with comfy_lease(c, "SAM 3.1"):
        pass
    assert [r["label"] for r in fresh_gpu.snapshot()["resident"]] == ["ComfyUI: SAM 3.1"]


def _func(path, name, cls=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    scope = tree
    if cls:
        scope = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    for n in ast.walk(scope):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{path}: no function {cls + '.' if cls else ''}{name}")


@pytest.mark.parametrize("path,name,kind", STAGE_CALLS)
def test_every_stage_entry_calls_before_stage(path, name, kind):
    fn = _func(path, name)
    kinds = [c.args[0].value for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "before_stage" and c.args and isinstance(c.args[0], ast.Constant)]
    assert kinds == [kind], f"{path}:{name} must call before_stage({kind!r}) once, found {kinds}"


@pytest.mark.parametrize("path,cls,name", LEASED_CALLS)
def test_every_model_call_runs_under_the_lease(path, cls, name):
    fn = _func(path, name, cls)
    uses = [w for w in ast.walk(fn) if isinstance(w, ast.With) for it in w.items
            if isinstance(it.context_expr, ast.Call) and getattr(it.context_expr.func, "attr", None) == "use"]
    assert uses, f"{path}:{cls}.{name} must run inside GPU.use(...)"


def test_every_comfy_prompt_runs_under_the_lease():
    """queue_prompt + wait_for_prompt of every Comfy graph inside `with comfy_lease(...)`."""
    bad = []
    for p in list((ROOT / "engine").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        leased = set()
        for w in ast.walk(tree):
            if isinstance(w, ast.With) and any(isinstance(it.context_expr, ast.Call)
                                               and getattr(it.context_expr.func, "id", None) == "comfy_lease"
                                               for it in w.items):
                leased |= {id(n) for n in ast.walk(w)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in ("queue_prompt", "wait_for_prompt") \
                    and id(n) not in leased:
                bad.append(f"{p.name}:{n.lineno}")
    assert not bad, f"Comfy prompts outside comfy_lease: {bad}"


def test_heavy_stages_release_after_band_tracking():
    """VideoMaMa / LTX-2.5: the band points (CoTracker3, awaited) are tracked BEFORE the release – otherwise the
    tracker would load again next to the model."""
    for path, name in (("engine/pipeline.py", "vmm_run"), ("engine/pipeline.py", "ltx_run"),
                       ("engine/stage_preview.py", "_videomama")):
        src = ast.get_source_segment((ROOT / path).read_text(encoding="utf-8"), _func(path, name))
        assert src.index("band_points_fn(") < src.index("before_stage("), f"{path}:{name}"


def test_nobody_unloads_comfy_on_its_own():
    """Unloading Comfy's models (free_vram with unload) only through gpu_manager; refine's after-prompt cache release
    (unload_models=False) is not a stage release."""
    bad = []
    for p in list((ROOT / "engine").glob("*.py")) + [ROOT / "app.py"]:
        if p.name in ("gpu_manager.py",) or p.name.endswith("_worker.py"):
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "free_vram":
                kw = {k.arg: getattr(k.value, "value", None) for k in n.keywords}
                if kw.get("unload_models") is not False:
                    bad.append(f"{p.name}:{n.lineno}")
    assert not bad, f"free_vram outside engine/gpu_manager: {bad}"


def test_no_second_release_layer():
    """engine/gpu_release is gone and nothing closes our workers around the manager (close_runners / close_workers
    only at exit or inside the runners)."""
    assert not (ROOT / "engine" / "gpu_release.py").exists()
    for p in list((ROOT / "engine").glob("*.py")) + [ROOT / "app.py"]:
        assert "gpu_release" not in p.read_text(encoding="utf-8"), p.name


class _Stats:
    """Comfy whose free VRAM follows `seq` (bytes) one poll at a time; `t` = the fake clock."""
    def __init__(self, seq):
        self.seq, self.t, self.calls = list(seq), 0.0, []

    def get_queue(self):
        return {"busy": False}

    def free_vram(self, unload_models=True, free_memory=True):
        self.calls.append("free")
        return True

    def get_system_stats(self):
        v = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
        return {"devices": [{"vram_free": v}]}

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


GB = 1 << 30


def test_waits_until_comfy_really_freed_then_settles():
    """Comfy applies /free in its worker loop – a prompt queued at once ran on the old models (2026-10-26): wait for the
    free VRAM to rise and settle."""
    c = _Stats([2 * GB, 2 * GB, 9 * GB, 14 * GB, 14 * GB, 14 * GB, 14 * GB])
    before = gpu_manager._vram_free(c)
    freed = gpu_manager.wait_comfy_freed(c, before, c.clock, c.sleep)
    assert freed == 12.0 and c.t < 1.5


def test_nothing_to_free_returns_quickly():
    c = _Stats([5 * GB])
    assert gpu_manager.wait_comfy_freed(c, gpu_manager._vram_free(c), c.clock, c.sleep) == 0.0
    assert c.t <= gpu_manager.FREE_STILL_S + gpu_manager.FREE_POLL_S + 1e-9


def test_wait_is_bounded_and_skipped_without_stats(fresh_gpu):
    c = _Stats([GB * i for i in range(1, 200)])             # keeps changing: stop at FREE_WAIT_S
    gpu_manager.wait_comfy_freed(c, 0, c.clock, c.sleep)
    assert c.t <= gpu_manager.FREE_WAIT_S + gpu_manager.FREE_POLL_S + 1e-9
    r = before_stage("comfy", _Client(), "x")                  # a client without stats: no wait, freed unknown
    assert r["comfy"] is True and r["freed_gb"] is None


def test_executor_work_of_a_job_knows_its_job():
    """Endpoints call loop.run_in_executor(None, …) directly: the work must still be the job's (CURRENT_JOB), else the
    GPU manager takes it for background work and the job never shows "czeka na GPU"."""
    import asyncio
    import app as app_mod
    from engine.jobs import JOBS

    seen = {}

    @app_mod.long_job("Test GPU: {name}", quiet=True)
    async def ep(name: str):
        seen["id"] = await asyncio.get_running_loop().run_in_executor(None, JOBS.current_id)
        return seen["id"]

    async def main():
        return await ep(name="Woman")
    jid = asyncio.run(main())
    assert jid and seen["id"] == jid
    assert next(j for j in JOBS.snapshot() if j["id"] == jid)["label"] == "Test GPU: Woman"
