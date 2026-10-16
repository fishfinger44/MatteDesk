# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""UI translation (static/i18n.js + static/i18n/en.json): catalog shape, placeholders kept, loaded before app.js."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = ROOT / "static" / "i18n" / "en.json"
PH = re.compile(r"\{\d+\}")


def test_catalog_valid_and_placeholders_kept():
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    strings = cat["strings"]
    assert len(strings) > 1500
    bad = [k for k, v in strings.items() if v and sorted(PH.findall(k)) != sorted(PH.findall(v))]
    assert not bad, f"placeholders changed: {bad[:5]}"
    pl = [k for k, v in strings.items() if v and re.search(r"[ąćęłńśźż]", v)]
    assert not pl, f"Polish left in translations: {pl[:5]}"


def test_i18n_loaded_first_and_switch_present():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert '<script src="/static/i18n.js"></script>' in html
    assert html.index("/static/i18n.js") < html.index("/static/app.js")
    assert 'id="langToggle"' in html
