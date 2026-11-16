# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""One retry after a Comfy prompt ran out of GPU memory (engine.gpu_manager.retry_after_oom, wired into
comfy_client.ComfyClient.wait_for_prompt = the path of every graph).

Contract: GPU out-of-memory -> /free once, wait until the memory is back, the SAME graph queued once more; a second
out-of-memory -> the failure with "dwa razy" (exactly 2 attempts); any other failure is never retried; "Przerwij" /
cancel_check during the wait -> cancelled, nothing queued again. Synthetic: no ComfyUI, no GPU."""

import pytest

from comfy_client import ComfyClient
from engine import gpu_manager
from engine.gpu_manager import OOM_RETRY_TEXT, OOM_TWICE_TEXT, retry_after_oom
from engine.jobs import JOBS, JobCancelled
from mock_comfy import FakeComfyServer

GB = 1 << 30
OOM = {"success": False, "kind": "oom_gpu", "error": "Brak pamięci GPU w węźle X"}
OK = {"success": True, "outputs": {"8": {}}}
BOOM = {"success": False, "kind": "error", "error": "Błąd w węźle X: boom"}


class _Comfy:
    """A client whose prompts end with the scripted `results`, on a fake clock; free VRAM follows `vram`."""

    def __init__(self, results, vram=(2 * GB, 2 * GB, 9 * GB, 9 * GB), busy=False):
        self.results, self.vram, self.busy = [dict(r) for r in results], list(vram), busy
        self.t, self.sent, self.waited, self.freed, self.events = 0.0, [], [], 0, []

    def get_queue(self):
        return {"busy": self.busy}

    def free_vram(self, unload_models=True, free_memory=True):
        assert unload_models and free_memory
        self.freed += 1
        self.events.append("free")
        return True

    def get_system_stats(self):
        v = self.vram.pop(0) if len(self.vram) > 1 else self.vram[0]
        return {"devices": [{"vram_free": v}]}

    def queue_prompt(self, graph, extra_data=None):
        self.sent.append(graph)
        self.events.append("queue")
        return {"success": True, "prompt_id": f"p{len(self.sent)}"}

    def wait(self, pid):
        self.waited.append(pid)
        return self.results.pop(0)

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.events.append("sleep")


def _run(c, graph, **kw):
    sub = c.queue_prompt(graph)
    res = c.wait(sub["prompt_id"])
    return retry_after_oom(c, res, requeue=lambda: c.queue_prompt(graph), wait=c.wait, clock=c.clock, sleep=c.sleep, **kw)


def test_oom_then_success_is_one_retry_with_the_same_graph(caplog):
    c, g = _Comfy([OOM, OK]), {"1": {"class_type": "X"}}
    with caplog.at_level("WARNING", logger="engine.gpu_manager"):
        res = _run(c, g)
    assert res["success"] and res["outputs"] == {"8": {}} and res["oom_retried"] is True
    assert c.freed == 1 and c.waited == ["p1", "p2"]
    assert len(c.sent) == 2 and c.sent[0] is c.sent[1] is g
    # /free first, then the wait for the memory (polls), only then the second queue
    assert c.events[0] == "queue" and c.events[1] == "free" and c.events[-1] == "queue" and "sleep" in c.events[2:-1]
    assert OOM_RETRY_TEXT in caplog.text and OOM_RETRY_TEXT == "Brak pamięci GPU – zwalniam i ponawiam (1/1)"


def test_two_ooms_fail_after_exactly_two_attempts():
    c = _Comfy([OOM, OOM, OK])
    res = _run(c, {"1": {}})
    assert not res["success"] and res["kind"] == "oom_gpu"
    assert res["error"].startswith(OOM_TWICE_TEXT) and "dwa razy" in res["error"] and OOM["error"] in res["error"]
    assert len(c.sent) == 2 and c.waited == ["p1", "p2"] and c.freed == 1


@pytest.mark.parametrize("first", [BOOM, {"success": False, "kind": "oom_ram", "error": "Brak pamięci RAM"},
                                   {"success": False, "kind": "comfy_down", "error": "ComfyUI nie odpowiada"}, OK])
def test_anything_else_is_never_retried(first):
    c = _Comfy([first, OK])
    res = _run(c, {"1": {}})
    assert res == first and "oom_retried" not in res
    assert len(c.sent) == 1 and c.freed == 0 and c.t == 0.0


def test_cancel_during_the_wait_ends_it():
    c = _Comfy([OOM, OK], vram=[2 * GB, 3 * GB, 4 * GB, 5 * GB, 6 * GB, 7 * GB, 8 * GB, 9 * GB])   # still rising

    def cancel_check():
        if c.t >= 0.4:
            raise JobCancelled()

    with pytest.raises(JobCancelled):
        _run(c, {"1": {}}, cancel_check=cancel_check)
    assert c.freed == 1 and len(c.sent) == 1 and c.waited == ["p1"]      # nothing queued again


def test_przerwij_of_the_job_ends_it():
    c = _Comfy([OOM, OK])
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    try:
        JOBS.cancel()
        with pytest.raises(JobCancelled):
            _run(c, {"1": {}})
    finally:
        JOBS.finish(jid, "cancelled")
        JOBS._cancel.clear()        # the flag is cleared by the next job start only
    assert c.freed == 0 and len(c.sent) == 1


def test_busy_queue_means_no_free_and_no_retry():
    """Comfy never gets /free under a (foreign) prompt: after OOM_FREE_WAIT_S the first failure is returned."""
    c = _Comfy([OOM, OK], busy=True)
    res = _run(c, {"1": {}})
    assert res == OOM and c.freed == 0 and len(c.sent) == 1
    assert gpu_manager.OOM_FREE_WAIT_S <= c.t < gpu_manager.OOM_FREE_WAIT_S + 1.0


def test_rejected_requeue_keeps_the_first_failure():
    c = _Comfy([OOM])
    c.queue_prompt({"1": {}})
    res = retry_after_oom(c, c.wait("p1"), requeue=lambda: {"success": False, "error": "ComfyUI zajęte"},
                          wait=c.wait, clock=c.clock, sleep=c.sleep)
    assert not res["success"] and res["kind"] == "oom_gpu" and "ponowienie" in res["error"] and "zajęte" in res["error"]


# ----------------------------------------------------------------------------- the real client (scripted HTTP)
def _err(msg, etype="RuntimeError"):
    return {"status": {"completed": False, "messages": [["execution_error", {
        "node_type": "MatAnyone2", "exception_type": etype, "exception_message": msg}]]}}


DONE = {"status": {"completed": True}, "outputs": {"8": {"images": []}}}
CUDA = _err("CUDA out of memory. Tried to allocate 2.00 GiB", "torch.OutOfMemoryError")


def _server(c, script):
    """FakeComfyServer whose queued prompts finish at once with the next entry of `script`; counts /prompt, /free."""
    srv = FakeComfyServer(c)
    srv.posted, srv.frees = [], 0
    base = srv.request

    def request(endpoint, method="GET", data=None, timeout=10.0):
        if endpoint == "/prompt" and method == "POST":
            pid = f"ours{len(srv.posted) + 1}"
            srv.posted.append(data["prompt"])
            srv.history[pid] = script.pop(0)
            return 200, {"prompt_id": pid}
        if endpoint == "/free" and method == "POST":
            assert data == {"unload_models": True, "free_memory": True}
            srv.frees += 1
        return base(endpoint, method, data, timeout)

    c._request = request
    return srv


def _client_run(script, graph=None):
    c = ComfyClient()
    c.spill_watch_factory = lambda: None
    srv = _server(c, list(script))
    graph = graph or {"1": {"class_type": "MatAnyone2"}}
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    try:
        sub = c.queue_prompt(graph)
        assert sub["success"]
        res = c.wait_for_prompt(sub["prompt_id"], timeout_sec=600, poll_interval=1.0)
        job = next(j for j in JOBS.snapshot() if j["id"] == jid)
    finally:
        JOBS.finish(jid, "done")
    return res, srv, job, graph


def test_client_retries_its_prompt_once_after_cuda_oom():
    res, srv, _job, g = _client_run([CUDA, DONE])
    assert res["success"] and res["oom_retried"] and res["outputs"] == DONE["outputs"]
    assert srv.frees == 1 and srv.posted == [g, g]


@pytest.mark.parametrize("msg,etype", [("CUDA out of memory. Tried to allocate 2.00 GiB", "torch.OutOfMemoryError"),
                                       ("Allocation on device", "torch.OutOfMemoryError"),
                                       ("Allocation on device 0 would exceed allowed memory", "RuntimeError"),
                                       ("CUDA error: out of memory", "RuntimeError")])
def test_client_two_ooms_two_attempts_polish_message(msg, etype):
    res, srv, job, _g = _client_run([_err(msg, etype), _err(msg, etype), DONE])
    assert not res["success"] and res["kind"] == "oom_gpu" and len(srv.posted) == 2 and srv.frees == 1
    assert res["error"].startswith(OOM_TWICE_TEXT + ": Brak pamięci GPU w węźle MatAnyone2")
    assert job["comfy"] == res["error"]                       # what the progress window shows


def test_client_does_not_retry_other_errors():
    res, srv, _job, _g = _client_run([_err("boom"), DONE])
    assert not res["success"] and res["kind"] == "error" and len(srv.posted) == 1 and srv.frees == 0
    res, srv, _job, _g = _client_run([_err("std::bad_alloc", "MemoryError"), DONE])
    assert res["kind"] == "oom_ram" and len(srv.posted) == 1 and srv.frees == 0


def test_client_shows_the_retry_in_the_job():
    seen = []
    c = ComfyClient()
    c.spill_watch_factory = lambda: None
    _server(c, [CUDA, DONE])
    jid = JOBS.start("t", "Comfy test")
    JOBS.bind(jid)
    orig = JOBS.update

    def update(*a, **kw):
        if kw.get("comfy"):
            seen.append(kw["comfy"])
        return orig(*a, **kw)

    JOBS.update = update
    try:
        sub = c.queue_prompt({"1": {}})
        assert c.wait_for_prompt(sub["prompt_id"], timeout_sec=600, poll_interval=1.0)["success"]
    finally:
        del JOBS.update
        JOBS.finish(jid, "done")
    assert OOM_RETRY_TEXT in seen
