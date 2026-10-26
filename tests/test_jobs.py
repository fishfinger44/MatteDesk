# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Status bar + "Przerwij": running jobs are listed, cancel stops them at the next progress point / Comfy poll."""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from comfy_client import ComfyClient
from engine.jobs import JOBS, JobCancelled


def test_progress_adapter_and_cancel():
    jid = JOBS.start("t", "Test")
    JOBS.bind(jid)
    JOBS.progress("SAM3", 3, 10)
    JOBS.progress(4, 10, 1004)
    j = next(x for x in JOBS.snapshot() if x["id"] == jid)
    assert (j["step"], j["done"], j["total"]) == ("SAM3", 4, 10)
    assert JOBS.cancel()
    with pytest.raises(JobCancelled):
        JOBS.progress("SAM3", 5, 10)
    JOBS.finish(jid, "cancelled")
    assert JOBS.start("t", "next") and not JOBS.cancelled()  # a new job starts clean
    JOBS.finish(JOBS.snapshot()[-1]["id"])



def test_percent_jobs_have_no_frame_count_eta():
    """Deflicker reports stage milestones in % (0 / 10 / 40 ... of 100), not frames: unit "pct" -> no ETA from the jumps
    (2026-10-21: the window showed "klatki 10/100 · ETA 48:39" on a 193-frame shot)."""
    from engine.sysinfo import job_details
    jid = JOBS.start("t", "Deflicker")
    JOBS.bind(jid)
    assert next(x for x in JOBS.snapshot() if x["id"] == jid)["unit"] == "frames"
    JOBS.update(unit="pct")
    JOBS.progress("Etap 1", 10, 100)
    job = job_details(None, JOBS.snapshot())["job"]
    assert job["unit"] == "pct" and job["eta_s"] is None and job["done"] == 10
    JOBS.finish(jid)
    jid = JOBS.start("t", "Frames")
    JOBS.bind(jid)
    JOBS.progress("MatAnyone2", 5, 20)
    time.sleep(0.01)
    job = job_details(None, JOBS.snapshot())["job"]
    assert job["unit"] == "frames" and job["eta_s"] is not None
    JOBS.finish(jid)

def test_api_cancel_long_object_run(monkeypatch):
    seen = {}

    class SlowPipe:
        def run_core(self, name, frames, cb):
            for i in range(200):
                cb("SAM3 śledzenie", i, 200)
                time.sleep(0.02)
            seen["finished"] = True

    monkeypatch.setattr(app_module, "_pipeline", lambda: SlowPipe())
    monkeypatch.setattr(app_module, "_require_stage", lambda name, stage: None)
    monkeypatch.setattr(app_module, "_objects", lambda: type("S", (), {"get": lambda self, n: {}})())
    monkeypatch.setattr(app_module, "_shot_frames", lambda: list(range(1, 201)))
    api = TestClient(app_module.app)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", api.post("/api/object/figure/run")))
    t.start()
    for _ in range(100):
        run = [j for j in api.get("/api/jobs").json()["jobs"] if j["state"] == "running"]
        if run and run[-1]["done"] > 2:
            break
        time.sleep(0.05)
    assert run and run[-1]["label"] == "Licz ujęcie: figure" and run[-1]["total"] == 200
    assert api.post("/api/jobs/cancel").json()["cancelled"]
    t.join(10)
    assert out["r"].status_code == 409 and "przerwane" in out["r"].json()["detail"]
    assert "finished" not in seen
    last = [j for j in api.get("/api/jobs").json()["jobs"] if j["label"] == "Licz ujęcie: figure"][-1]
    assert last["state"] == "cancelled"


def test_comfy_wait_cancels_our_prompt(monkeypatch):
    c = ComfyClient()
    calls = []

    def fake_request(endpoint, method="GET", data=None, timeout=10.0):
        calls.append((endpoint, method, data))
        if endpoint == "/queue" and method == "GET":
            return 200, {"queue_running": [[0, "p1"]], "queue_pending": []}
        if endpoint.startswith("/history"):
            return 200, {}
        return 200, {}

    monkeypatch.setattr(c, "_request", fake_request)
    jid = JOBS.start("t", "Comfy")
    threading.Timer(0.3, JOBS.cancel).start()
    with pytest.raises(JobCancelled):
        c.wait_for_prompt("p1", timeout_sec=10, poll_interval=0.05)
    JOBS.finish(jid, "cancelled")
    assert ("/queue", "POST", {"delete": ["p1"]}) in calls and ("/interrupt", "POST", {}) in calls


def test_busy_rules_while_a_job_runs(monkeypatch):
    """User clicks other things while 'Licz ujęcie: figure' runs: a second heavy job and edits of figure are
    refused with a clear message; project switch is refused; other objects can still be edited / added."""
    gate = threading.Event()

    class SlowPipe:
        def run_core(self, name, frames, cb):
            while not gate.is_set():
                cb("SAM3 śledzenie", 1, 10)
                time.sleep(0.02)
            return type("R", (), {"uncertain": [], "suggested_keys": [], "recomputed_intervals": [], "cached_intervals": []})()

    class Store:
        def get(self, n):
            return {"name": n, "tool": {}}

        def update(self, n, **k):
            return None

    monkeypatch.setattr(app_module, "_pipeline", lambda: SlowPipe())
    monkeypatch.setattr(app_module, "_require_stage", lambda name, stage: None)
    monkeypatch.setattr(app_module, "_objects", lambda: Store())
    monkeypatch.setattr(app_module, "_shot_frames", lambda: list(range(1, 11)))
    api = TestClient(app_module.app)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", api.post("/api/object/figure/run")))
    t.start()
    for _ in range(100):
        if any(j["state"] == "running" for j in api.get("/api/jobs").json()["jobs"]):
            break
        time.sleep(0.02)
    try:
        r = api.post("/api/object/tree/run")                       # second heavy job
        assert r.status_code == 409 and "Trwa: Licz ujęcie: figure" in r.json()["detail"]
        r = api.post("/api/object/figure/finalize", json={})          # double click on the same object
        assert r.status_code == 409
        r = api.put("/api/object/figure/edge_mode", json={"mode": "core"})   # edit of the object being computed
        assert r.status_code == 409 and "jest teraz liczony" in r.json()["detail"]
        r = api.put("/api/object/tree/edge_mode", json={"mode": "core"})    # another object: not blocked by the job
        assert r.status_code != 409
        r = api.post("/api/project/open", json={"manifest_path": "x.json"})
        assert r.status_code == 409 and "Otwarcie projektu" in r.json()["detail"]
    finally:
        gate.set()
        t.join(10)
    assert out["r"].status_code == 200
    assert api.post("/api/object/tree/run").status_code == 200       # free again


# ----------------------------------------------------------------------------- Z19: failure detection + details
from mock_comfy import FakeComfyServer  # noqa: E402


def _wait(c, pid, **kw):
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    try:
        return c.wait_for_prompt(pid, timeout_sec=kw.pop("timeout_sec", 1800), poll_interval=1.0, **kw), JOBS.snapshot()
    finally:
        JOBS.finish(jid, "done")


def test_comfy_crash_detected_within_15s():
    c = ComfyClient()
    srv = FakeComfyServer(c, at={5.0: lambda s: setattr(s, "alive", False)})   # Comfy dies 5 s into the job
    srv.running = ["p1"]
    c.our_prompts.add("p1")
    res, _ = _wait(c, "p1")
    assert not res["success"] and res["kind"] == "comfy_down" and "nie odpowiada" in res["error"]
    assert srv.t <= 15.0 + 1.0, srv.t                       # SAM3 timeout is 1800 s – not waited for


def test_prompt_lost_after_restart():
    c = ComfyClient()
    srv = FakeComfyServer(c, at={3.0: lambda s: (s.running.clear(), s.pending.clear())})   # restart: queue empty, no history
    srv.running = ["p1"]
    res, _ = _wait(c, "p1")
    assert not res["success"] and res["kind"] == "lost" and "zrestartowany" in res["error"]
    assert srv.t <= 3.0 + c.lost_after_s + 1.0


def test_oom_execution_error_is_readable():
    c = ComfyClient()
    srv = FakeComfyServer(c)
    srv.history["p1"] = {"status": {"completed": False, "messages": [["execution_error", {
        "node_type": "SAM3_VideoTrack", "exception_type": "torch.OutOfMemoryError",
        "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"}]]}}
    res, _ = _wait(c, "p1")
    assert res["kind"] == "oom_gpu" and res["error"].startswith("Brak pamięci GPU w węźle SAM3_VideoTrack")
    assert "CUDA out of memory" in str(res["raw_error"])
    assert ComfyClient.describe_error({"node_type": "X", "exception_type": "MemoryError", "exception_message": ""})[1] == "oom_ram"
    assert ComfyClient.describe_error({"node_type": "X", "exception_message": "boom"}) == ("Błąd w węźle X: boom", "error")


def test_foreign_prompt_status():
    c = ComfyClient()
    srv = FakeComfyServer(c, at={4.0: lambda s: (s.running.__setitem__(0, "p1"), s.pending.clear()),
                                 6.0: lambda s: s.history.__setitem__("p1", {"status": {"completed": True}, "outputs": {"8": {}}})})
    srv.running, srv.pending = ["someone_elses"], ["p1"]
    c.our_prompts.add("p1")
    seen = []
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    try:
        res = c.wait_for_prompt("p1", timeout_sec=60, poll_interval=1.0,
                                progress_callback=lambda e: seen.append(next(j for j in JOBS.snapshot() if j["id"] == jid)))
    finally:
        JOBS.finish(jid, "done")
    assert res["success"]
    first = seen[0]
    assert first["comfy_info"]["state"] == "foreign" and first["comfy_info"]["ahead"] == 1
    assert "cudze zadanie" in first["comfy"] and "nasze czeka: 1" in first["comfy"]
    assert any(x["comfy_info"]["state"] == "ours" for x in seen)       # then ours runs


def test_core_progress_per_interval():
    from engine.crop import CropSegment
    from engine.segment import CorePropagator
    w = 32
    seg = CropSegment(seg_id="s0", first=1, last=12, overlap_in=0, overlap_out=0, crop=[0, 0, w, w], pad=[0, 0, 0, 0],
                      scale_to_1008=1.0)

    class R:
        def track(self, imgs, init):
            return [(init > 0.5).astype(np.float32) for _ in imgs]
    k = np.zeros((w, w), bool)
    k[8:20, 8:20] = True
    steps = []
    CorePropagator(R(), [seg], w, w, load_crop_image=lambda f, s: np.zeros((w, w, 3), np.float32)).propagate(
        list(range(1, 13)), {3: k, 9: k}, lambda st, d, t: steps.append((st, d, t)))
    texts = [s for s, _, _ in steps]
    assert any("przedział 2/3" in t and "w przód 3→9" in t for t in texts)
    assert any("przedział 2/3" in t and "wstecz 9→3" in t for t in texts)
    assert any("przedział 1/3" in t and "wstecz 3→1" in t for t in texts)
    done = [d for _, d, _ in steps]   # done / total count FRAMES (the job window shows them as frames)
    assert done == sorted(done) and done[-1] == 12 and {t for _, _, t in steps} == {12}
    # one key (Woman2, 2026-10-26): 2 intervals must not read as "2/2 frames"
    steps.clear()
    CorePropagator(R(), [seg], w, w, load_crop_image=lambda f, s: np.zeros((w, w, 3), np.float32)).propagate(
        list(range(1, 13)), {2: k}, lambda st, d, t: steps.append((st, d, t)))
    assert steps[-1][1:] == (12, 12) and (2, 12) in [x[1:] for x in steps]


def test_jobs_details_without_nvidia_smi(monkeypatch):
    import shutil
    import engine.sysinfo as si
    c = ComfyClient()
    srv = FakeComfyServer(c)
    srv.running, srv.pending = ["foreign1"], ["ours_x"]
    c.our_prompts.add("ours_x")
    monkeypatch.setattr(shutil, "which", lambda name: None)                 # no nvidia-smi on this machine
    monkeypatch.setitem(app_module.STATE, "comfy_client", c)
    monkeypatch.setattr(app_module, "_DETAILS", None)
    api = TestClient(app_module.app)
    jid = JOBS.start("t", "Przelicz finalnie: x")
    JOBS.update(step="QA: migotanie", done=50, total=193, jid=jid)
    try:
        d = api.get("/api/jobs/details").json()
    finally:
        JOBS.finish(jid, "done")
    assert d["gpu"] is None
    assert d["comfy"]["alive"] and d["comfy"]["ram"]["total"] == 64 * 2 ** 30
    assert d["comfy"]["devices"][0]["vram_total"] == 12 * 2 ** 30
    assert d["comfy"]["queue"]["foreign_running"] == ["foreign1"] and d["comfy"]["queue"]["foreign_pending"] == []
    assert d["job"]["label"] == "Przelicz finalnie: x" and d["job"]["done"] == 50 and d["job"]["eta_s"] is not None
    # Comfy down -> alive False, still 200
    srv.alive = False
    monkeypatch.setattr(app_module, "_DETAILS", None)
    d = api.get("/api/jobs/details").json()
    assert d["comfy"]["alive"] is False and d["job"] is None


def test_gpu_info_parses_nvidia_smi(monkeypatch):
    import shutil
    import engine.sysinfo as si

    class Out:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3080 Ti, 87, 10240, 12288, 71\n"
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    g = si.gpu_info(run=lambda *a, **k: Out())
    assert g == [{"name": "NVIDIA GeForce RTX 3080 Ti", "util_pct": 87.0, "mem_used_mb": 10240.0, "mem_total_mb": 12288.0, "temp_c": 71.0}]

    def boom(*a, **k):
        raise OSError("no device")
    assert si.gpu_info(run=boom) is None


def test_comfy_busy_is_not_down_and_hung_after_the_limit():
    """2026-10-21 SH010: Comfy blocked > 10 s while loading 96 frames of 2688 px float EXR – requests time out, the process
    lives. That must not abort the run as "nie odpowiada"; only a refused connection is "down"."""
    c = ComfyClient()
    srv = FakeComfyServer(c, at={5.0: lambda s: setattr(s, "busy", True),
                                 125.0: lambda s: (setattr(s, "busy", False), s.history.__setitem__("p2", {
                                     "status": {"completed": True, "messages": []}, "outputs": {"8": {}}}))})
    srv.running = ["p2"]
    c.our_prompts.add("p2")
    res, _ = _wait(c, "p2")
    assert res["success"], res                                  # waited through 2 minutes of timeouts
    assert srv.t >= 120.0

    c2 = ComfyClient()
    c2.busy_after_s = 60.0
    srv2 = FakeComfyServer(c2, at={1.0: lambda s: setattr(s, "busy", True)})
    srv2.running = ["p3"]
    c2.our_prompts.add("p3")
    res2, _ = _wait(c2, "p3")
    assert not res2["success"] and res2["kind"] == "comfy_hung" and "zrestartuj" in res2["error"]
    assert 60.0 <= srv2.t <= 90.0
    assert ComfyClient.is_timeout("<urlopen error timed out>") and not ComfyClient.is_timeout("[WinError 10061] connection refused")



class _FakeSpill:
    """Stand-in for the spilled-GPU-memory sampler (engine.sysinfo.SpillSampler): a fixed peak, no PowerShell."""

    def __init__(self, gb):
        self.peak, self.base = gb, 0.0
        self.entered = self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    @property
    def spill_gb(self):
        return self.peak


def test_spill_to_ram_warns_while_comfy_computes():
    """User request 2026-10-26 (sh060: MatAnyone2 ran 90+ min with 7.7 GB of GPU memory in RAM, nothing said so): every
    Comfy prompt is watched; >= SPILL_WARN_GB spilled -> a warning in the job's Comfy text + comfy_info.spill_gb, and the
    job keeps the peak (gpu_spill_gb) for run.json."""
    import comfy_client as cc
    c = ComfyClient()
    spill = _FakeSpill(7.7)
    c.spill_watch_factory = lambda: spill
    srv = FakeComfyServer(c, at={4.0: lambda s: s.history.__setitem__("p1", {"status": {"completed": True}, "outputs": {}})})
    srv.running = ["p1"]
    c.our_prompts.add("p1")
    seen = []
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    try:
        res = c.wait_for_prompt("p1", timeout_sec=60, poll_interval=1.0,
                                progress_callback=lambda e: seen.append(next(j for j in JOBS.snapshot() if j["id"] == jid)))
        assert JOBS.spill_gb() == pytest.approx(7.7)
    finally:
        JOBS.finish(jid, "done")
    assert res["success"] and spill.entered and spill.exited
    last = seen[-1]
    assert last["comfy_info"]["spill_gb"] == pytest.approx(7.7) and "do RAM" in last["comfy"]
    assert last["gpu_spill_gb"] == pytest.approx(7.7)
    assert cc.SPILL_WARN_GB == pytest.approx(1.0)


def test_small_spill_says_nothing_and_no_reader_is_fine():
    c = ComfyClient()
    c.spill_watch_factory = lambda: _FakeSpill(0.4)
    srv = FakeComfyServer(c, at={3.0: lambda s: s.history.__setitem__("p1", {"status": {"completed": True}, "outputs": {}})})
    srv.running = ["p1"]
    c.our_prompts.add("p1")
    res, snap = _wait(c, "p1")
    j = snap[-1]
    assert res["success"] and "spill_gb" not in (j["comfy_info"] or {}) and "RAM" not in (j["comfy"] or "")
    assert not j.get("gpu_spill_gb")
    c2 = ComfyClient()
    c2.spill_watch_factory = lambda: None              # no counter (Linux / tests): waits as before
    srv2 = FakeComfyServer(c2, at={2.0: lambda s: s.history.__setitem__("p2", {"status": {"completed": True}, "outputs": {}})})
    srv2.running = ["p2"]
    assert _wait(c2, "p2")[0]["success"]


def test_run_json_records_the_spill(tmp_path):
    """The one run.json writer (engine/refine.write_run_json) adds the job's spill peak when it was >= SPILL_WARN_GB."""
    import json
    from engine.refine import write_run_json
    jid = JOBS.start("t", "run")
    JOBS.bind(jid)
    try:
        write_run_json(tmp_path, {"stage": "x"})
        assert "gpu_spill_gb" not in json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        JOBS.update(gpu_spill_gb=3.26)
        JOBS.update(gpu_spill_gb=1.5)                   # the peak stays
        write_run_json(tmp_path, {"stage": "x"})
        rj = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "x" and rj["gpu_spill_gb"] == pytest.approx(3.3)
    finally:
        JOBS.finish(jid, "done")
