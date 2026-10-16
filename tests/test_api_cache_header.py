# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""/api/* responses carry Cache-Control: no-cache (frame URLs are shared between projects -> stale previews).
Since fc6e3ab the page (/) and /static/* revalidate too (a stale cached index.html / *.js broke the page init)."""
from fastapi.testclient import TestClient

import app as md_app


def test_api_responses_revalidate():
    c = TestClient(md_app.app)
    r = c.get("/api/jobs")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-cache"


def test_page_and_static_revalidate():
    c = TestClient(md_app.app)
    for path in ("/", "/static/app.js"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-cache", path


def test_other_paths_untouched():
    c = TestClient(md_app.app)
    r = c.get("/favicon.ico")
    assert r.status_code == 200
    assert "no-cache" not in (r.headers.get("cache-control") or "")
