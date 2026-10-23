# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Version consistency: VERSION == start_v<ver>.bat == top CHANGELOG entry == what the server reports."""
import re

from engine.config import REPO_DIR


def _version():
    return (REPO_DIR / "VERSION").read_text(encoding="utf-8").strip()


def test_version_format_and_bat_name():
    v = _version()
    assert re.fullmatch(r"\d+\.\d\d?", v), v
    assert [p.name for p in REPO_DIR.glob("start*.bat")] == [f"start_v{v}.bat"]


def test_changelog_top_entry_matches_version():
    m = re.search(r"^## (\S+) ", (REPO_DIR / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert m and m.group(1) == _version()


def test_server_reports_version():
    import app
    assert app.APP_VERSION == _version() and app.app.version == _version()


def test_bump_scheme():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bump", REPO_DIR / "tools" / "bump_version.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert [b.next_version(v, False) for v in ("0.1", "0.11", "0.18")] == ["0.11", "0.12", "0.19"]
    assert [b.next_version(v, True) for v in ("0.1", "0.13", "0.9", "0.95")] == ["0.20", "0.20", "1.0", "1.0"]   # two digits: GitHub lists 0.4 below 0.35 (2026-10-24)
