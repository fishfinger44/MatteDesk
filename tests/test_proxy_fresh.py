# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""User report 2026-11-10: „czyszczenie się policzyło, ale nie widać efektu” – the plate „cleaned” moved to a new
Cleanup version, but /api/frame kept serving the proxies of the first run. A proxy is rendered again when its source
frame is newer; the response tells the browser to revalidate."""
import os
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app as A


def _frame(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((64, 48, 3), value, np.uint8))


def test_proxy_follows_a_new_cleanup_version(tmp_path, monkeypatch):
    work = tmp_path / "work"
    v1, v2 = tmp_path / "v001" / "c_####.png", tmp_path / "v003" / "c_####.png"
    _frame(tmp_path / "v001" / "c_0010.png", 30)
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"cleaned": str(v1)}, "colorspace": "sRGB"})
    monkeypatch.setattr(A, "_plate_name_or_400", lambda p: None)
    first = A._proxy_path("cleaned", 10)
    assert abs(int(cv2.imread(str(first))[..., 0].mean()) - 30) < 4
    time.sleep(0.05)
    _frame(tmp_path / "v003" / "c_0010.png", 200)                          # the new run, written after the proxy
    A.STATE["manifest"]["plates"]["cleaned"] = str(v2)
    again = A._proxy_path("cleaned", 10)
    assert again == first and abs(int(cv2.imread(str(again))[..., 0].mean()) - 200) < 4
    # unchanged source: the cached proxy is reused (not rendered again)
    m = os.stat(again).st_mtime_ns
    A._proxy_path("cleaned", 10)
    assert os.stat(again).st_mtime_ns == m
    # the full-resolution tile cache follows the source too
    assert int(A._full_display_u8("cleaned", 10)[..., 0].mean()) > 190


def test_frame_response_is_revalidated(tmp_path, monkeypatch):
    _frame(tmp_path / "b" / "f_0001.png", 90)
    monkeypatch.setitem(A.STATE, "work_dir", str(tmp_path / "work"))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(tmp_path / "b" / "f_####.png")}, "colorspace": "sRGB"})
    monkeypatch.setattr(A, "_plate_name_or_400", lambda p: None)
    r = TestClient(A.app).get("/api/frame/1?plate=beauty")
    assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache"


def test_proxy_follows_a_switch_back_to_an_older_version(tmp_path, monkeypatch):
    """2026-11-11: „👁” on v002 after v004 – the older source is not newer than the proxy, the view kept v004."""
    work = tmp_path / "work"
    old, new = tmp_path / "v002" / "c_####.png", tmp_path / "v004" / "c_####.png"
    _frame(tmp_path / "v002" / "c_0005.png", 40)
    time.sleep(0.05)
    _frame(tmp_path / "v004" / "c_0005.png", 210)
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"cleaned": str(new)}, "colorspace": "sRGB"})
    monkeypatch.setattr(A, "_plate_name_or_400", lambda p: None)
    assert int(cv2.imread(str(A._proxy_path("cleaned", 5)))[..., 0].mean()) > 200
    A.STATE["manifest"]["plates"]["cleaned"] = str(old)
    assert abs(int(cv2.imread(str(A._proxy_path("cleaned", 5)))[..., 0].mean()) - 40) < 4


def test_stale_code_check_sees_the_routers(monkeypatch, tmp_path):
    """2026-11-11: a new route in api/cleanup.py did not light the „restart” bar – the static JS called a route the
    running server did not have."""
    monkeypatch.setattr(A, "_STARTED_AT", 0.0)
    assert A._code_changed_since_start()
    import inspect
    src = inspect.getsource(A._code_changed_since_start)
    assert '"api").glob("*.py")' in src and '"engine").rglob("*.py")' in src
