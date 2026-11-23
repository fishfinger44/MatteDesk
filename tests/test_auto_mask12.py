# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 1's mask 1+2 write (app._auto_mask12). 2026-11-20, SD160_020 „tlo”: the progress callback was built inside
the executor lambda (no event loop in that thread) – every stage-1 write failed at once, silently, and the export
stabilised all 148 frames again. Now: it writes (callback from the thread works), a failure is returned (endpoint
`mask12_error`, a toast in the UI), „Przerwij” is not swallowed."""

import asyncio
from pathlib import Path

import pytest


class _Pipe:
    def __init__(self, exc=None):
        self.exc, self.calls = exc, []

    def write_mask12(self, name, frames, cb=None):
        self.calls.append((name, list(frames)))
        if cb:
            cb("Zapis maski 1+2", 1, len(frames))   # called from the executor thread, as the real write
        if self.exc:
            raise self.exc
        return {"coverage": [[frames[0], frames[-1]]]}


def _patch(monkeypatch, tmp_path):
    import app as app_module
    import engine.workarea as wa

    class _Store:
        def dir(self, n):
            return Path(tmp_path) / n
    monkeypatch.setattr(app_module, "_objects", lambda: _Store())
    monkeypatch.setattr(app_module, "_shot_frames", lambda: list(range(1, 11)))
    monkeypatch.setattr(wa, "core_coverage", lambda d: [[1, 10]])
    return app_module


def test_auto_mask12_reports_errors_and_keeps_cancel(tmp_path, monkeypatch):
    from engine.jobs import JobCancelled
    app_module = _patch(monkeypatch, tmp_path)
    ok = _Pipe()
    assert asyncio.run(app_module._auto_mask12(ok, "o")) is None and ok.calls == [("o", list(range(1, 11)))]
    bad = _Pipe(MemoryError("brak RAM"))
    assert asyncio.run(app_module._auto_mask12(bad, "o")) == "brak RAM"
    with pytest.raises(JobCancelled):
        asyncio.run(app_module._auto_mask12(_Pipe(JobCancelled("przerwane")), "o"))
