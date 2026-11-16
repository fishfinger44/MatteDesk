# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/worker_proto.py - The protocol of the long-lived GPU workers run in ComfyUI's Python (audit 2026-11-02, step 6).

Worker side, `serve(make_handler)`: one JSON request per line on stdin, one JSON answer per line on stdout, {"op": "quit"}
ends it; an exception of the handler = {"ok": false, "error": "<type>: <message>"} and the worker lives on.
This process, `WorkerProcess`: start / request with cancel_check, timeout, restart / polite close / log tail.
`check_worker`: the `--check` call, one cache.

Stdlib only: the workers load this file by its path. They run with -I, because engine/queue.py would shadow the stdlib
`queue` if engine/ were on the path of ComfyUI's Python (and that Python has no MatteDesk packages).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

CLOSE_WAIT_S = 10.0     # how long a worker gets to finish after {"op": "quit"} before it is killed
POLL_S = 0.25           # how often a waiting request checks cancel_check / the timeout


# ------------------------------------------------------------------ the worker's side
def serve(make_handler: Callable[[], Callable[..., Dict[str, Any]]]) -> int:
    """The request loop. `make_handler()` runs after stdout is taken over (library prints go to stderr, they must not
    break the protocol) and gives `handle(req, ask) -> fields of the answer`; `ask(msg)` sends `msg` and returns the
    next line as JSON (a question to the caller in the middle of a request: on_reply of WorkerProcess.request)."""
    proto = sys.stdout
    sys.stdout = sys.stderr

    def send(obj: Dict[str, Any]) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    def ask(msg: Dict[str, Any]) -> Dict[str, Any]:
        send(msg)
        line = sys.stdin.readline()
        return json.loads(line) if line.strip() else {}

    handle = make_handler()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "quit":
                break
            resp = {"ok": True, **handle(req, ask)}
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        send(resp)
    return 0


# ------------------------------------------------------------------ this process' side
class WorkerDied(RuntimeError):
    """The worker exited during a request; the message holds the tail of its log."""


class WorkerTimeout(RuntimeError):
    """No answer in time; the worker was killed."""


class WorkerProcess:
    """One long-lived worker: started on the first request, replaced when it died, killed by a cancel or a timeout (a
    running forward cannot be interrupted any other way), so the next request starts a fresh one. A subclass sets
    `worker`, `label`, `doctor_item`, `tmp_prefix` and calls __init__. Not thread-safe: callers hold their own lock."""

    worker: Optional[Path] = None
    label = "worker"
    doctor_item = ""
    tmp_prefix = "md_worker_"
    error_cls = RuntimeError          # raised when the interpreter is missing

    def __init__(self, python: Optional[Union[str, Path]], env: Optional[Dict[str, str]] = None, timeout_s: float = 600.0):
        self.python = str(python) if python else ""
        self.env = env
        self.timeout_s = timeout_s
        self.restarts = 0
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._log = None
        self._dir: Optional[tempfile.TemporaryDirectory] = None

    def serve_args(self) -> Sequence[str]:
        return ("--serve",)

    def tmp_dir(self) -> Path:
        """The folder of the request files and the log; goes with close()."""
        if self._dir is None:
            # ignore_cleanup_errors: on Windows a worker killed a moment ago can still hold worker.log (its stderr)
            # when the directory goes – that must not raise at interpreter exit (testmon run 2026-10-25)
            self._dir = tempfile.TemporaryDirectory(prefix=self.tmp_prefix, ignore_cleanup_errors=True)
        return Path(self._dir.name)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if not self.python or not Path(self.python).exists():
            raise self.error_cls(f"{self.label}: nie znaleziono Pythona ComfyUI (Doktor → „{self.doctor_item}”)")
        if self._log is None:
            self._log = open(self.tmp_dir() / "worker.log", "a+", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen([self.python, "-I", str(self.worker), *self.serve_args()], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, env=self.env)
        lines: "queue.Queue[Optional[str]]" = queue.Queue()     # of this process only: nothing left over from a killed one
        self._lines = lines
        proc = self._proc

        def reader() -> None:
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def log_tail(self) -> str:
        try:
            self._log.flush()
            self._log.seek(0)
            return self._log.read()[-600:].strip()
        except Exception:
            return ""

    def send(self, obj: Dict[str, Any]) -> None:
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def read(self, cancel_check: Optional[Callable[[], None]] = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """The next answer. `cancel_check()` is polled every POLL_S and may raise."""
        limit = self.timeout_s if timeout_s is None else timeout_s
        t0 = time.time()
        while True:
            try:
                line = self._lines.get(timeout=POLL_S)
            except queue.Empty:
                if cancel_check:
                    cancel_check()
                if time.time() - t0 > limit:
                    raise WorkerTimeout(f"{self.label} (Python ComfyUI) nie odpowiedział w {limit:.0f} s: {self.log_tail()}") from None
                continue
            if line is None:
                raise WorkerDied(f"{self.label} (Python ComfyUI) zakończył się: {self.log_tail()}")
            return json.loads(line)

    def request(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None,
                timeout_s: Optional[float] = None, restarts: int = 0,
                on_reply: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """One request -> its answer (`ok` false is an answer: the caller decides). `restarts`: a worker that dies is
        replaced and the request repeated this many times. `on_reply(answer)` -> a message to send when the worker
        asked something (serve's `ask`), None = this is the answer. Anything else raised – a cancel, a timeout – kills
        the worker: never a busy one left behind."""
        attempt = 0
        while True:
            try:
                if not self.running:
                    self.start()
                self.send(req)
                resp = self.read(cancel_check, timeout_s)
                while on_reply is not None:
                    msg = on_reply(resp)
                    if msg is None:
                        break
                    self.send(msg)
                    resp = self.read(cancel_check, timeout_s)
                return resp
            except (WorkerDied, BrokenPipeError, OSError) as e:
                tail = self.log_tail()
                self.kill()
                if attempt < restarts:
                    attempt += 1
                    self.restarts += 1
                    continue
                if isinstance(e, WorkerDied):
                    raise
                raise WorkerDied(f"{self.label} (Python ComfyUI) zakończył się: {tail or e}") from e
            except BaseException:
                self.kill()
                raise

    def kill(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(timeout=CLOSE_WAIT_S)
            except Exception:
                pass

    def close(self) -> None:
        """Stop the worker (frees its GPU memory), its log and folder; the next request starts a new one."""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                self.send({"op": "quit"})
                p.wait(timeout=CLOSE_WAIT_S)
            except Exception:
                pass
        self.kill()
        log, self._log = self._log, None
        if log is not None:
            log.close()
        d, self._dir = self._dir, None
        if d is not None:
            d.cleanup()


CHECK_CACHE: Dict[str, Tuple[bool, str]] = {}   # the installer clears it after installing a package


def check_worker(python: Optional[Union[str, Path]], worker: Path, args: Sequence[str] = (),
                 env: Optional[Dict[str, str]] = None, timeout_s: float = 180.0) -> Tuple[bool, str]:
    """(ok, the worker's `--check` output or the reason): can this interpreter run `worker`? Cached per interpreter,
    worker, arguments and the fake flags of the tests' workers."""
    if not python or not Path(python).exists():
        return False, "nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python)"
    e = env if env is not None else os.environ
    fake = ",".join(sorted(k for k, v in e.items() if k.startswith("MD_") and k.endswith("_FAKE") and v == "1"))
    key = "|".join((str(python), str(worker), ",".join(args), fake))
    if key not in CHECK_CACHE:
        try:
            r = subprocess.run([str(python), "-I", str(worker), "--check", *args], capture_output=True, text=True,
                               timeout=timeout_s, env=env)
            out = (r.stdout or r.stderr or "").strip()
            ok = r.returncode == 0 and out.startswith("ok")
            CHECK_CACHE[key] = (ok, out if ok else (out[-300:] or f"kod {r.returncode}"))
        except Exception as ex:
            CHECK_CACHE[key] = (False, str(ex))
    return CHECK_CACHE[key]
