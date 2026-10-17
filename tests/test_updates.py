# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Update check at start (engine/updates.py, user decision 2026-10-17): version order, once a day, skip, kinds."""
import urllib.error
from email.message import Message
from pathlib import Path

from engine import updates as up

REL = {"tag_name": "v0.4", "html_url": "https://github.com/fishfinger44/MatteDesk/releases/tag/v0.4",
       "body": "- New A\n- Fix B\n\n" + "\n".join(f"- line {i}" for i in range(20)),
       "assets": [{"name": "MatteDesk-0.4-setup.exe", "browser_download_url": "https://x/setup.exe", "size": 130_000_000,
                   "digest": "sha256:" + "e" * 64}]}


def installer_dir(tmp: Path) -> Path:
    (tmp / "python").mkdir(parents=True)
    (tmp / "python" / "python.exe").write_bytes(b"")
    return tmp


def test_version_order():
    p = up.parse_version
    assert p("0.1") < p("0.11") < p("0.12") < p("0.2") < p("0.25") < p("0.3") < p("v0.31") < p("0.4") < p("1.0")
    assert p("garbage") < p("0.1")


def test_check_newer_installer_copy(tmp_path):
    saved = {}
    u = up.check("0.3", {}, installer_dir(tmp_path), now=1000.0, api_get=lambda url: REL, save=saved.update)
    assert u["available"] and u["version"] == "0.4" and u["kind"] == "installer" and u["can_install"]
    assert u["notes"].startswith("- New A\n- Fix B") and u["notes"].endswith("…") and u["notes"].count("\n") == up.NOTES_LINES
    assert u["setup"]["sha256"] == "e" * 64 and u["changelog"].endswith("CHANGELOG.md")
    assert saved["update_last"]["t"] == 1000.0


def test_once_a_day_and_skip(tmp_path):
    calls = []
    api = lambda url: calls.append(url) or REL
    cfg = {"update_last": {"t": 1000.0, "release": up.latest_release(lambda u: REL)}}
    u = up.check("0.3", cfg, tmp_path, now=1000.0 + 3600, api_get=api)
    assert calls == [] and u["available"]                         # cached result, no request
    up.check("0.3", cfg, tmp_path, now=1000.0 + up.DAY + 1, api_get=api)
    assert len(calls) == 1
    u = up.check("0.3", {**cfg, "update_skip": "0.4"}, tmp_path, now=1001.0, api_get=api)
    assert u["skipped"]
    assert not up.check("0.4", cfg, tmp_path, now=1001.0, api_get=api)["available"]


def test_git_copy_no_installer_and_disabled(tmp_path):
    (tmp_path / ".git").mkdir()
    installer_dir(tmp_path)
    u = up.check("0.3", {}, tmp_path, now=1.0, api_get=lambda url: REL)
    assert u["kind"] == "git" and u["available"] and not u["can_install"]
    called = []
    u = up.check("0.3", {"update_check": False}, tmp_path, now=1.0, api_get=lambda url: called.append(1) or REL)
    assert not u["available"] and not called and not u["enabled"]


def test_no_release_yet_and_offline(tmp_path):
    def e404(url):
        raise urllib.error.HTTPError(url, 404, "nf", Message(), None)
    assert not up.check("0.3", {}, tmp_path, now=1.0, api_get=e404)["available"]

    def offline(url):
        raise urllib.error.URLError("no network")
    u = up.check("0.3", {}, tmp_path, now=1.0, api_get=offline)
    assert not u["available"] and "error" in u


def test_download_and_run(tmp_path):
    seen = {}

    def fake_dl(url, dst, sha256=None, progress=None):
        seen["dl"] = (url, sha256)
        progress(5, 10)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"exe")
    exe = up.download_and_run({"url": "u", "sha256": "f" * 64, "size": 10, "name": "MatteDesk-0.4-setup.exe"},
                              tmp_path, download=fake_dl, popen=lambda argv, **k: seen.update(argv=argv))
    assert seen["dl"] == ("u", "f" * 64) and seen["argv"] == [str(exe), "/SP-"] and up.STATE["state"] == "started"


def test_api_update_endpoints(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod
    import engine.config as ec
    monkeypatch.setattr(ec, "LOCAL", tmp_path / "config.local.json")
    monkeypatch.setattr(up, "_api", lambda url: REL)
    monkeypatch.setattr(up.check, "__defaults__", (False, None, lambda url: REL, None))
    c = TestClient(app_mod.app)
    r = c.get("/api/update?force=1").json()
    assert r["version"] == "0.4" and "available" in r
    assert c.post("/api/update/skip", json={"version": "0.4"}).json()["ok"]
    assert c.post("/api/update/settings", json={"enabled": False}).json()["enabled"] is False
    assert ec._read(ec.LOCAL)["update_skip"] == "0.4" and ec._read(ec.LOCAL)["update_check"] is False
