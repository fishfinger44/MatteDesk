"""Audit 2026-11-02 step 6 (#11 + the drift between the workers): one serve loop and one WorkerProcess. The wire protocol
to the workers in ComfyUI's Python stays what it was: test_wire_* run on the old loops too."""
import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from engine import birefnet as br
from engine import sapiens2 as sp
from engine import tracking as tk
from engine import vitmatte as vm
from engine import worker_proto as wp
from engine.jobs import JOBS, JobCancelled

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
WORKERS = {"vitmatte": "MD_VITMATTE_FAKE", "birefnet": "MD_BIREFNET_FAKE", "sapiens2": "MD_SAPIENS2_FAKE",
           "cotracker": "MD_COTRACKER_FAKE", "align_gpu": "MD_ALIGN_GPU_FAKE"}


def _talk(worker, env_key, lines, args=("--serve",)):
    env = {**os.environ, env_key: "1"}
    p = subprocess.run([sys.executable, "-I", str(ENGINE / f"{worker}_worker.py"), *args], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, timeout=120, env=env)
    return p.returncode, [json.loads(x) for x in p.stdout.splitlines()]


def _npy(tmp_path, shape):
    x, out = tmp_path / "x.npy", tmp_path / "out.npy"
    np.save(x, np.zeros(shape, np.float32))
    return str(x), str(out)


@pytest.mark.parametrize("worker,shape,keys", [
    ("vitmatte", (8, 8, 4), {"ok", "device", "load_seconds", "timings"}),
    ("birefnet", (8, 8, 3), {"ok", "device", "dtype", "size", "load_seconds"}),
    ("sapiens2", (8, 8, 3), {"ok", "device", "dtype", "load_seconds"}),
])
def test_wire_model_workers(tmp_path, worker, shape, keys):
    x, out = _npy(tmp_path, shape)
    args = ("--serve", "fake/model")
    code, resp = _talk(worker, WORKERS[worker], ["", json.dumps({"x": x, "out": out}), "   ",
                                                 json.dumps({"x": "Z:/nope.npy", "out": out}), json.dumps({"op": "quit"}),
                                                 json.dumps({"x": x, "out": out})], args)
    assert code == 0 and len(resp) == 2            # blank lines skipped, nothing after quit
    assert set(resp[0]) == keys and resp[0]["ok"] is True and Path(out).exists()
    assert resp[1]["ok"] is False and set(resp[1]) == {"ok", "error"}
    assert "Error" in resp[1]["error"]


def test_wire_cotracker(tmp_path):
    src, dst = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(src, n_videos=np.array(0), checkpoint=np.array("x"))
    code, resp = _talk("cotracker", WORKERS["cotracker"], [json.dumps({"in": str(src), "out": str(dst)}),
                                                           json.dumps({"in": "Z:/nope.npz", "out": str(dst)}),
                                                           json.dumps({"op": "quit"})])
    assert code == 0 and resp[0]["ok"] is True and {"ok", "device", "peak_gib", "load_seconds"} == set(resp[0])
    assert resp[1]["ok"] is False and set(resp[1]) == {"ok", "error"}


def test_wire_align(tmp_path):
    code, resp = _talk("align_gpu", WORKERS["align_gpu"], [json.dumps({"op": "release"}), json.dumps({"op": "nope"}),
                                                           json.dumps({"op": "quit"})])
    assert code == 0 and set(resp[0]) == {"ok", "device", "seconds"} and resp[0]["device"] == "fake"
    assert resp[1] == {"ok": False, "error": "ValueError: unknown op nope"}


# ------------------------------------------------------------------------------------------- the common code

def _script(tmp_path):
    """A worker in the shape of the real ones (run with -I, the protocol module loaded by its path)."""
    p = tmp_path / "w_worker.py"
    p.write_text(f"""
import importlib.util, os, sys, time
_s = importlib.util.spec_from_file_location("worker_proto", {str(ENGINE / "worker_proto.py")!r})
_wp = importlib.util.module_from_spec(_s); _s.loader.exec_module(_wp)
def make():
    print("noise from a library at load")
    def handle(req, ask):
        print("noise in a forward")
        op = req.get("op")
        if op == "sleep": time.sleep(float(req["s"]))
        if op == "die": sys.stderr.write("boom in the log\\n"); sys.stderr.flush(); os._exit(3)
        if op == "die_once" and not os.path.exists(req["flag"]):
            open(req["flag"], "w").close(); os._exit(3)
        if op == "fail": raise ValueError("no such thing")
        if op == "ask": return {{"got": ask({{"ok": True, "need": 1}})}}
        return {{"pid": os.getpid()}}
    return handle
if __name__ == "__main__":
    if sys.argv[1] == "--check":
        print("ok fake cached=a,b") if os.environ.get("W_OK", "1") == "1" else print("no torch")
        sys.exit(0 if os.environ.get("W_OK", "1") == "1" else 1)
    sys.exit(_wp.serve(make))
""", encoding="utf-8")
    return p


class W(wp.WorkerProcess):
    label = "Test"
    doctor_item = "Test item"
    tmp_prefix = "md_test_"


@pytest.fixture
def w(tmp_path):
    x = W(sys.executable, env=dict(os.environ), timeout_s=30)
    x.worker = _script(tmp_path)
    yield x
    x.close()


def test_one_long_lived_process_and_prints_do_not_break_the_protocol(w):
    a = w.request({"op": "x"})
    b = w.request({"op": "y"})
    assert a["ok"] and a["pid"] == b["pid"]


def test_error_answer_keeps_the_worker(w):
    pid = w.request({})["pid"]
    r = w.request({"op": "fail"})
    assert r == {"ok": False, "error": "ValueError: no such thing"}
    assert w.request({})["pid"] == pid


def test_cancel_kills_within_a_poll_and_the_next_request_is_fresh(w):
    pid = w.request({})["pid"]

    class Stop(RuntimeError):
        pass
    t0 = time.time()

    def cancel():
        if time.time() - t0 > 0.3:
            raise Stop()
    with pytest.raises(Stop):
        w.request({"op": "sleep", "s": 30}, cancel_check=cancel)
    assert time.time() - t0 < 5 and w._proc is None
    assert w.request({})["pid"] != pid


def test_timeout_kills_and_says_so(w):
    t0 = time.time()
    with pytest.raises(wp.WorkerTimeout, match="Test.*nie odpowiedział"):
        w.request({"op": "sleep", "s": 30}, timeout_s=0.5)
    assert time.time() - t0 < 5 and w._proc is None


def test_death_reports_the_log_tail(w):
    with pytest.raises(wp.WorkerDied, match="boom in the log"):
        w.request({"op": "die"})
    assert w._proc is None


def test_restart_repeats_the_request_once(w, tmp_path):
    r = w.request({"op": "die_once", "flag": str(tmp_path / "f")}, restarts=1)
    assert r["ok"] and w.restarts == 1
    with pytest.raises(wp.WorkerDied):
        w.request({"op": "die"}, restarts=1)
    assert w.restarts == 2


def test_worker_may_ask_during_a_request(w):
    r = w.request({"op": "ask"}, on_reply=lambda resp: {"answer": 42} if resp.get("need") else None)
    assert r["got"] == {"answer": 42} and r["ok"]


def test_close_quits_politely_and_cleans_up(w):
    w.request({})
    p, d = w._proc, Path(w._dir.name)
    w.close()
    assert p.returncode == 0 and w._proc is None and not d.exists()
    w.close()                  # twice is fine
    assert w.request({})["ok"]  # a fresh process afterwards


def test_kill_of_a_dead_process_is_fine(w):
    w.request({})
    w._proc.kill()
    w._proc.wait()
    assert w.request({})["ok"]       # dead -> a fresh one on the next request


def test_missing_python_names_the_doctor_item(tmp_path):
    x = W("Z:/no/python.exe")
    x.worker = _script(tmp_path)
    with pytest.raises(RuntimeError, match="Test item"):
        x.request({})


def test_check_worker_ok_fail_and_cache(tmp_path):
    s = _script(tmp_path)
    wp.CHECK_CACHE.clear()
    assert wp.check_worker(sys.executable, s) == (True, "ok fake cached=a,b")
    assert any(str(s) in k for k in wp.CHECK_CACHE)
    env = {**os.environ, "W_OK": "1"}
    assert wp.check_worker(sys.executable, s, env=env) == (True, "ok fake cached=a,b")      # cached by python + worker + args
    wp.CHECK_CACHE.clear()
    ok, why = wp.check_worker(sys.executable, s, env={**os.environ, "W_OK": "0"})
    assert not ok and "no torch" in why
    assert wp.check_worker(None, s)[0] is False
    assert "nie znaleziono" in wp.check_worker("Z:/no.exe", s)[1]
    wp.CHECK_CACHE.clear()


# ------------------------------------------------------------------------------------------- #11: „Przerwij” in forward

def _slow_runner(tmp_path, cls=vm.ComfyVitMatteRunner):
    r = cls("fake/model", python=sys.executable, env=dict(os.environ), timeout_s=60)
    r.worker = _script(tmp_path)
    r.serve_args = lambda: ("--serve",)
    return r


class _Sleepy:
    def __init__(self, r):
        self.r = r
        orig = r.send

        def send(obj):
            orig({"op": "sleep", "s": 30, **obj})
        r.send = send


@pytest.mark.parametrize("cls", [vm.ComfyVitMatteRunner, br.ComfyBirefnetRunner, sp.ComfySapiens2Runner])
def test_forward_stops_on_cancel_of_the_job(tmp_path, monkeypatch, cls):
    """It used to wait up to timeout_s (600 s) after „Przerwij”."""
    r = _slow_runner(tmp_path, cls)
    _Sleepy(r)
    monkeypatch.setattr(JOBS, "current_id", lambda: 7)
    monkeypatch.setattr(JOBS, "running", lambda: [{"id": 7, "label": "test"}])
    t0 = time.time()

    def check():
        if time.time() - t0 > 0.5:
            raise JobCancelled()
    monkeypatch.setattr(JOBS, "check", check)
    try:
        with pytest.raises(JobCancelled):
            r.forward(np.zeros((8, 8, 4), np.float32))
        assert time.time() - t0 < 5 and r._proc is None
    finally:
        r.close()


def test_forward_outside_a_job_ignores_an_old_cancel(tmp_path, monkeypatch):
    """The cancel flag stays set until the next job: a background forward (no job) must not die of it."""
    r = _slow_runner(tmp_path)
    monkeypatch.setattr(JOBS, "current_id", lambda: None)

    def check():
        raise JobCancelled()
    monkeypatch.setattr(JOBS, "check", check)
    assert vm.job_cancel_check() is None
    monkeypatch.setattr(JOBS, "current_id", lambda: 7)         # a pool thread still bound to a finished job
    monkeypatch.setattr(JOBS, "running", lambda: [])
    assert vm.job_cancel_check() is None
    r.close()


# ------------------------------------------------------------------------------------------- one implementation

@pytest.mark.parametrize("name", list(WORKERS))
def test_workers_use_the_one_loop(name):
    src = (ENGINE / f"{name}_worker.py").read_text(encoding="utf-8")
    assert "for line in sys.stdin" not in src
    assert "_wp.serve(" in src


@pytest.mark.parametrize("path,count", [("vitmatte.py", 0), ("tracking.py", 1), ("align_gpu.py", 0)])
def test_launchers_have_no_process_code_of_their_own(path, count):
    """tracking.py: the one-shot path (a process per call, kept for the bit-identity test) stays."""
    src = (ENGINE / path).read_text(encoding="utf-8")
    assert src.count("subprocess.Popen(") == count
    assert "stdin.write" not in src and "--check" not in src


def test_worker_proto_is_stdlib_only():
    """It is loaded by ComfyUI's Python too, which has no engine/ on its path."""
    tree = ast.parse((ENGINE / "worker_proto.py").read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    top |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert top <= (set(sys.stdlib_module_names) | {"__future__"}), top


def test_same_close_for_all():
    for cls in (vm.ComfyVitMatteRunner, br.ComfyBirefnetRunner, sp.ComfySapiens2Runner, tk._CoTrackerWorker):
        assert issubclass(cls, wp.WorkerProcess)
        assert "close" not in cls.__dict__ and "_start" not in cls.__dict__
    from engine.align_gpu import GpuAligner
    assert issubclass(GpuAligner, wp.WorkerProcess)


def test_threads_can_cancel_a_waiting_request(w):
    """The cancel is polled while the reply is awaited, not only between requests."""
    seen = threading.Event()

    def cancel():
        seen.set()
    assert w.request({"op": "sleep", "s": 0.6}, cancel_check=cancel)["ok"] and seen.is_set()
