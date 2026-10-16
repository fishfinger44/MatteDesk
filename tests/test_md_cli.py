# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z28 – tools/md.py (CLI for Claude Code) against the app through TestClient; docs/AGENT_API.md lists every route."""

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import md  # noqa: E402
from engine.jobs import JOBS  # noqa: E402


class TCHttp:
    """md.Http over a FastAPI TestClient (same request() contract)."""

    def __init__(self, client):
        self.c = client

    def request(self, method, path, body=None, timeout=None):
        r = self.c.request(method, path, json=body)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None


def _run(argv, http):
    lines = []
    code = md.main(argv, http=http, out=lines.append)
    return code, "\n".join(lines)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    from test_ui_stages_api import _open
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    return app_module, api, shot, mf


def test_status_keys_jobs(project):
    app_module, api, shot, mf = project
    before = mf.read_bytes()
    http = TCHttp(api)
    code, out = _run(["status"], http)
    assert code == 0 and "SYN" in out and "puppet" in out and "Zadanie: brak" in out
    code, out = _run(["status", "--json"], http)
    assert code == 0 and '"objects"' in out
    code, out = _run(["keys", "puppet"], http)
    assert code == 0 and "puppet: 0 kluczy" in out
    code, out = _run(["jobs"], http)
    assert code == 0
    assert mf.read_bytes() == before


def test_run_wait_prints_progress_and_busy_is_refused(project, monkeypatch):
    app_module, api, shot, mf = project

    class FakeRes:
        uncertain, suggested_keys, recomputed_intervals, cached_intervals = [], [], [(1, 8)], []

    class FakePipe:
        def run_core(self, name, frames, cb=None):
            for i in range(3):
                cb("Propagacja rdzenia: przedział 1/1, w przód", i, 3)
                time.sleep(0.6)
            return FakeRes()
    monkeypatch.setattr(app_module, "_pipeline", lambda: FakePipe())
    monkeypatch.setattr(app_module, "_require_stage", lambda name, stage: None)
    code, out = _run(["run", "puppet", "--wait"], TCHttp(api))
    assert code == 0, out
    assert "Licz ujęcie puppet: gotowe" in out and "przeliczone [[1, 8]]" in out
    assert "Propagacja rdzenia" in out                       # progress printed while waiting

    # another heavy job is running -> the CLI does not start / interrupt it: exit 2 with a message
    jid = JOBS.start("x", "Przelicz finalnie: inny")
    try:
        code, out = _run(["finalize", "puppet", "--wait"], TCHttp(api))
    finally:
        JOBS.finish(jid, "done")
    assert code == 2 and "Serwer liczy: Przelicz finalnie: inny" in out and "nie przerywam" in out


def test_409_gives_exit_code_2(project, monkeypatch):
    app_module, api, shot, mf = project
    code, out = _run(["finalize", "puppet", "--wait"], TCHttp(api))   # stage "final" locked -> 409 from the server
    assert code == 2 and "ODMOWA (409)" in out


def test_qa_json_and_text(project):
    app_module, api, shot, mf = project
    store = app_module._objects()
    run = store.dir("puppet") / "runs" / "v001"
    run.mkdir(parents=True)
    qa = {"frames": [1, 2, 3], "flicker_median": 0.0031, "gate_ok": True, "colours": {"1": "green", "2": "red", "3": "yellow"},
          "flags": {"2": [["red", "dziury wewnętrzne: 1"]]}, "edge": {"before": 1.1, "pieces_to_fix": 1, "corrected": False,
          "pieces": [{"id": "0_3", "first": 1, "last": 3, "median_px": 1.2, "kind": "halo"}]}}
    (run / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    store.update("puppet", tool={"active_run": "v001"})
    code, out = _run(["qa", "puppet", "--json"], TCHttp(api))
    assert code == 0 and json.loads(out)["flicker_median"] == 0.0031 and json.loads(out)["run"] == "v001"
    code, out = _run(["qa", "puppet"], TCHttp(api))
    assert code == 0 and "migotanie (mediana) 0.0031" in out and "czerwone 1 [2]" in out and "0_3: F1–3 +1.2 px (halo)" in out


def test_export_needs_yes(project):
    app_module, api, shot, mf = project
    code, out = _run(["export", "puppet"], TCHttp(api))
    assert code == 1 and "--yes" in out


def test_agent_api_doc_lists_every_route():
    import app as app_module
    doc = (ROOT / "docs" / "AGENT_API.md").read_text(encoding="utf-8")
    missing = []
    for r in app_module.app.routes:
        if isinstance(r, APIRoute) and r.include_in_schema:
            for m in r.methods - {"HEAD"}:
                if f"| {m} | `{r.path}` |" not in doc:
                    missing.append(f"{m} {r.path}")
    assert not missing, f"docs/AGENT_API.md: brak tras {missing} – python tools/md.py api-doc --write"
    if (ROOT / "CLAUDE.md").exists():   # agent instructions: only in the development repo, not in the public copy
        assert "docs/AGENT_API.md" in (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
