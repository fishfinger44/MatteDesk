# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""A request for an object the open project does not have is a 404, not a 500 with a traceback in the log: e.g. an
overlay request of the previous project that reaches the server after a project switch (2026-11-03)."""

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from engine.objects import ObjectNotFound, ObjectStore


@pytest.mark.parametrize("url", ["/api/object/nope/frame/1?view=core", "/api/object/nope/life"])
def test_missing_object_is_404(tmp_path, monkeypatch, url):
    monkeypatch.setattr(app_module, "_objects", lambda: ObjectStore(tmp_path))
    monkeypatch.setitem(app_module.STATE, "work_dir", str(tmp_path))
    r = TestClient(app, raise_server_exceptions=False).get(url)
    assert r.status_code == 404
    assert "brak obiektu nope" in r.json()["detail"]


def test_store_error_is_still_a_file_not_found(tmp_path):
    """The endpoints that catch FileNotFoundError themselves keep working."""
    with pytest.raises(FileNotFoundError):
        ObjectStore(tmp_path).get("nope")
    assert issubclass(ObjectNotFound, FileNotFoundError)
