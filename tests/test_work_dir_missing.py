# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_work_dir_missing.py - the open project's work folder vanished (user report 2026-10-28: drive unplugged /
folder moved): a clear 409 `work_dir_missing`, never a 500 mkdir traceback, and nothing recreated on disk."""

import asyncio
import json

from fastapi.testclient import TestClient
from starlette.requests import Request

import app as app_mod
from app import app, STATE


def test_object_endpoints_give_409_and_create_nothing(tmp_path, monkeypatch):
    gone = tmp_path / "unplugged" / "shot" / "matte_desk"
    monkeypatch.setitem(STATE, "work_dir", str(gone))
    c = TestClient(app, raise_server_exceptions=False)
    for url in ("/api/object/obj1/life", "/api/object/obj1/stage_preview/matanyone", "/api/object/obj1/stage_preview/vitmatte"):
        r = c.get(url)
        assert r.status_code == 409, (url, r.status_code, r.text)
        det = r.json()["detail"]
        assert det["code"] == "work_dir_missing" and str(gone) in det["detail"]
    assert not (tmp_path / "unplugged").exists()   # GETs must not rebuild the vanished path


def test_unhandled_oserror_with_missing_work_dir_is_409(tmp_path, monkeypatch):
    monkeypatch.setitem(STATE, "work_dir", str(tmp_path / "gone"))
    req = Request({"type": "http", "method": "GET", "path": "/api/x", "headers": [], "query_string": b""})
    r = asyncio.run(app_mod._unhandled_error_as_json(req, FileNotFoundError("x")))
    assert r.status_code == 409 and json.loads(r.body)["detail"]["code"] == "work_dir_missing"
    # folder present -> still the plain 500 with the real reason
    monkeypatch.setitem(STATE, "work_dir", str(tmp_path))
    r = asyncio.run(app_mod._unhandled_error_as_json(req, FileNotFoundError("x")))
    assert r.status_code == 500
