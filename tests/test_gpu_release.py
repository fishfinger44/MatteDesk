# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/gpu_release – THE GPU release before every heavy stage (user decision 2026-10-26, LTX-2.5 spilled 10.1 GB
into RAM): a Comfy stage closes our GPU workers + unloads Comfy's models, a local stage unloads Comfy only; every
stage entry calls it, nobody frees the GPU on its own."""

import ast
import importlib
from pathlib import Path

import pytest

from engine import gpu_release
from engine.gpu_release import STAGE_CALLS, WORKERS, before_stage

ROOT = Path(__file__).resolve().parents[1]


class _Client:
    def __init__(self, busy=False):
        self.busy = busy
        self.calls = []

    def get_queue(self):
        return {"busy": self.busy}

    def free_vram(self, unload_models=True, free_memory=True):
        self.calls.append((unload_models, free_memory))
        return True


@pytest.fixture
def closed(monkeypatch):
    """Every worker closer replaced by a recorder."""
    log = []
    for label, mod, fn in WORKERS:
        m = importlib.import_module(mod)
        monkeypatch.setattr(m, fn, lambda idle_only=False, _l=label: log.append((_l, idle_only)))
    return log


def test_comfy_stage_closes_workers_and_unloads_comfy(closed):
    c = _Client()
    r = before_stage("comfy", c, "LTX-2.5")
    assert c.calls == [(True, True)] and r["comfy"] is True
    assert [l for l, _ in closed] == [l for l, _, _ in WORKERS] == r["workers"]
    assert ("CoTracker3", True) in closed            # a CoTracker3 worker busy with a band-point edit is left running


def test_local_stage_unloads_comfy_only(closed):
    c = _Client()
    r = before_stage("local", c, "ViTMatte")
    assert c.calls == [(True, True)] and closed == [] and r["workers"] == []


def test_busy_comfy_queue_is_never_freed(closed):
    c = _Client(busy=True)
    assert before_stage("comfy", c)["comfy"] is False and c.calls == []
    assert len(closed) == len(WORKERS)               # our own workers still closed
    assert before_stage("comfy", None)["comfy"] is False
    with pytest.raises(ValueError):
        before_stage("gpu", c)


def test_failing_closer_never_fails_the_stage(monkeypatch):
    import engine.sapiens2 as s2

    def boom():
        raise RuntimeError("worker hung")
    monkeypatch.setattr(s2, "close_runners", boom)
    r = before_stage("comfy", _Client())
    assert "Sapiens2" not in r["workers"] and r["comfy"] is True


def test_cotracker_close_idle_only():
    from engine import tracking

    class W:
        def __init__(self, busy):
            self.busy, self.closed = busy, False

        def close(self):
            self.closed = True
    idle, busy = W(False), W(True)
    saved = dict(tracking._WORKERS)
    try:
        tracking._WORKERS.clear()
        tracking._WORKERS.update(a=idle, b=busy)
        tracking.close_workers(idle_only=True)
        assert idle.closed and not busy.closed and list(tracking._WORKERS) == ["b"]
        tracking.close_workers()
        assert busy.closed and not tracking._WORKERS
    finally:
        tracking._WORKERS.clear()
        tracking._WORKERS.update(saved)


def _func(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{path}: no function {name}")


@pytest.mark.parametrize("path,name,kind", STAGE_CALLS)
def test_every_stage_entry_calls_before_stage(path, name, kind):
    fn = _func(path, name)
    kinds = [c.args[0].value for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "id", None) == "before_stage" and c.args and isinstance(c.args[0], ast.Constant)]
    assert kinds == [kind], f"{path}:{name} must call before_stage({kind!r}) once, found {kinds}"


def test_heavy_stages_release_after_band_tracking():
    """VideoMaMa / LTX-2.5: the band points (CoTracker3, awaited) are tracked BEFORE the release – otherwise the
    tracker would load again next to the model."""
    for path, name in (("engine/pipeline.py", "vmm_run"), ("engine/pipeline.py", "ltx_run"),
                       ("engine/stage_preview.py", "_videomama")):
        src = ast.get_source_segment((ROOT / path).read_text(encoding="utf-8"), _func(path, name))
        assert src.index("band_points_fn(") < src.index("before_stage("), f"{path}:{name}"


def test_nobody_unloads_comfy_on_its_own():
    """Unloading Comfy's models (free_vram with unload) only through gpu_release; refine's after-prompt cache release
    (unload_models=False) is not a stage release."""
    bad = []
    for p in list((ROOT / "engine").glob("*.py")) + [ROOT / "app.py"]:
        if p.name in ("gpu_release.py",) or p.name.endswith("_worker.py"):
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "free_vram":
                kw = {k.arg: getattr(k.value, "value", None) for k in n.keywords}
                if kw.get("unload_models") is not False:
                    bad.append(f"{p.name}:{n.lineno}")
    assert not bad, f"free_vram outside engine/gpu_release: {bad}"


def test_module_lists_are_real():
    for _, mod, fn in WORKERS:
        assert callable(getattr(importlib.import_module(mod), fn))
    assert gpu_release.release_comfy(object()) is False


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
    before = gpu_release._vram_free(c)
    freed = gpu_release.wait_comfy_freed(c, before, c.clock, c.sleep)
    assert freed == 12.0 and c.t < 1.5


def test_nothing_to_free_returns_quickly():
    c = _Stats([5 * GB])
    assert gpu_release.wait_comfy_freed(c, gpu_release._vram_free(c), c.clock, c.sleep) == 0.0
    assert c.t <= gpu_release.FREE_STILL_S + gpu_release.FREE_POLL_S + 1e-9


def test_wait_is_bounded_and_skipped_without_stats(closed):
    c = _Stats([GB * i for i in range(1, 200)])             # keeps changing: stop at FREE_WAIT_S
    gpu_release.wait_comfy_freed(c, 0, c.clock, c.sleep)
    assert c.t <= gpu_release.FREE_WAIT_S + gpu_release.FREE_POLL_S + 1e-9
    r = before_stage("comfy", _Client(), "x")                  # a client without stats: no wait, freed unknown
    assert r["comfy"] is True and r["freed_gb"] is None
