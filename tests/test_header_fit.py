# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Top bar always fits the window (static/header_fit.js; user report 2026-10-15: the Doctor button fell off the right edge
at smaller resolutions). Layout itself was checked in the browser at 1024 / 1100 / 1280 / 1366 / 1920 px; here: the wiring
and the folding rule (node, no browser)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "index.css").read_text(encoding="utf-8")
JS = (STATIC / "header_fit.js").read_text(encoding="utf-8")


def _fold_order():
    m = re.search(r"const FOLD_ORDER = \[([^\]]*)\]", JS)
    assert m
    return re.findall(r"'(\w+)'", m.group(1))


def test_script_and_menu_are_wired():
    assert '<script src="/static/header_fit.js"></script>' in HTML
    for i in ("headerMoreWrap", "headerMoreBtn", "headerMoreMenu"):
        assert f'id="{i}"' in HTML
    # the menu lives in the right group, next to the Doctor
    right = HTML.split('<div class="header-right">', 1)[1].split("</header>", 1)[0]
    assert 'id="headerMoreWrap"' in right and 'id="openDoctorBtn"' in right


def test_fold_order_never_hides_the_doctor_project_or_view_tabs():
    order = _fold_order()
    assert order, "pusta lista FOLD_ORDER"
    for i in order:
        assert f'id="{i}"' in HTML, f"{i} nie istnieje w index.html"
    for keep in ("openDoctorBtn", "openProjectModalBtn", "viewModeTabs", "appVersion"):
        assert keep not in order


def test_version_tag_is_never_hidden_by_css():
    # version = top-left corner of the UI (CLAUDE.md, versioning rule)
    for block in re.findall(r"@media[^{]*\{(.*?)\}\s*\}", CSS, re.S):
        assert ".version-tag" not in block.split("{")[0], block


def test_fit_rule_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – reguła składania sprawdzona tylko tekstowo")
    script = r"""
const { fitItems, FOLD_ORDER } = require(process.argv[1]);
function run(avail, widths) {
  const shown = new Set(Object.keys(widths));
  const total = () => [...shown].reduce((s, k) => s + widths[k], 0) + (shown.size < Object.keys(widths).length ? 40 : 0);
  const folded = fitItems(FOLD_ORDER.filter(k => k in widths), () => total() > avail,
                          k => shown.delete(k), () => Object.keys(widths).forEach(k => shown.add(k)), k => shown.add(k));
  return { folded, fits: total() <= avail };
}
// 2026-10-22 (evening): the shot info, the ComfyUI status and the language left the top bar for the compare row
const w = { reportBtn: 130, overlayStyle: 280, plateSelectorPill: 330, rest: 600 };
const w2 = { reportBtn: 130, overlayStyle: 180, plateSelectorPill: 330, rest: 600 };
console.log(JSON.stringify([run(5000, w), run(1300, w), run(1000, w), run(680, w), run(300, w), run(1400, w2)]));
"""
    r = subprocess.run([node, "-e", script, str(STATIC / "header_fit.js")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    wide, mid, narrow, tiny, impossible, overlay = json.loads(r.stdout)
    assert overlay == {"folded": [], "fits": True}                            # everything fits
    assert wide == {"folded": [], "fits": True}
    # least important first, stops once it fits
    assert mid["folded"] == ["reportBtn"] and mid["fits"]
    assert narrow["folded"] == ["reportBtn", "overlayStyle"] and narrow["fits"]
    assert tiny["folded"] == _fold_order() and tiny["fits"]
    assert impossible["folded"] == _fold_order() and not impossible["fits"]  # never loops, never folds the rest
