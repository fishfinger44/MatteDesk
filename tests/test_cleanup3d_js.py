# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D steps C1–C6 in the browser (static/cleanup3d.js, docs/PLAN_CLEANUP_3D.md §2, package P4) and the area /
extent editor (static/cleanup3d_areas.js, package P7): headless runs in jsdom with a fake API – tests/js/cleanup3d_*.mjs –
plus structural checks of the files.

The harness needs Node and jsdom: `cd tests/js && npm ci` (node_modules outside git). Skipped with that reason when either
is missing – a skip here means the Cleanup steps were NOT checked."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
JS = ROOT / "tests" / "js"


def _run(harness: str, ok: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – harness jsdom kroków Czyszczenia pominięty")
    probe = subprocess.run([node, "-e", "require('jsdom')"], capture_output=True, text=True, cwd=str(JS))
    if probe.returncode != 0:
        pytest.skip("brak pakietu jsdom – zainstaluj: cd tests/js && npm ci; kroki Czyszczenia 3D NIE sprawdzone")
    r = subprocess.run([node, str(JS / harness)], capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), timeout=180)
    assert r.returncode == 0, f"{harness}:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    assert ok in r.stdout


def test_cleanup3d_steps_in_jsdom():
    _run("cleanup3d_harness.mjs", "OK cleanup3d harness")


def test_cleanup3d_area_and_extent_editor_in_jsdom():
    """P7: brush stroke / closed curve -> the areas payload, layers, undo, extent with its frame, mark route, 501."""
    _run("cleanup3d_areas_harness.mjs", "OK cleanup3d areas harness")


def test_cleanup3d_reload_keeps_one_pane_and_descriptions_are_english():
    """Defects of the wave-1 browser check: (a) reload in step C3 shows only the Cleanup pane, (b) all six step
    descriptions translate (real static/i18n.js + catalog)."""
    _run("cleanup3d_reload_harness.mjs", "OK cleanup3d reload harness")


def test_cleanup3d_loaded_after_cleanup_and_wired_to_the_contract():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index('/static/cleanup.js"') < html.index('/static/cleanup3d.js"')
    js = (STATIC / "cleanup3d.js").read_text(encoding="utf-8")
    api = (ROOT / "api" / "cleanup3d.py").read_text(encoding="utf-8")
    # every URL the UI calls is a route of the P0 contract
    for url in ("/api/cleanup3d", "/api/cleanup3d/camera/solve", "/api/cleanup3d/cards", "/api/cleanup3d/render", "/api/cleanup3d/view/"):
        assert url in js and url.rstrip("/") in api, url
    for tail in ("/export", "/import", "/undo", "/regen"):
        assert f"keyUrl('{tail}')" in js and f'/api/cleanup3d/keys/{{card}}/{{frame}}{tail}"' in api, tail
    # the steps are the contract's steps, in order
    steps = re.findall(r"\{ id: '(\w+)', n: 'C\d'", js)
    types = (ROOT / "engine" / "cleanup3d" / "types.py").read_text(encoding="utf-8")
    contract = re.search(r'STEPS: Tuple\[str, \.\.\.\] = \(([^)]*)\)', types).group(1)
    assert steps == re.findall(r'"(\w+)"', contract)
    # the hook object is documented at the top of the file
    head = js[:js.index("(function ()")]
    for name in ("window.MD_CLEANUP3D", ".step", ".refresh()", ".areaMount()", ".toolMount()", ".onChange(fn)", ".onViewImage"):
        assert name in head, name
    # the object stage bar is only hidden (never rewritten) by Cleanup 3D
    assert "$('stageList').innerHTML" not in js and "stageList').innerHTML" not in js
    # actions answer 501 until their package lands: the server's text goes to a toast, never swallowed
    assert "toast(e.message || String(e), 8000)" in js


def test_cleanup3d_css_rules_prefixed_and_texts_translated():
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    block = css[css.index("/* Cleanup 3D steps C1-C6"):]
    for line in block.splitlines()[1:]:
        if line.strip():
            assert line.startswith(".c3-"), line
    en = json.loads((STATIC / "i18n" / "en.json").read_text(encoding="utf-8"))["strings"]
    js = (STATIC / "cleanup3d.js").read_text(encoding="utf-8")
    for s in ("Inpaint", "Kamera 3D", "Karty", "Klucze", "Projekcja", "Eksport", "Klucz karty", "▶ Policz kamerę",
              "✦ Generuj", "↶ Poprzednia wersja klucza", "⬇ Eksportuj klucz", "⬆ Wczytaj klucz", "＋ karta",
              "Obszary do dogenerowania"):
        assert s in js and en.get(s), s
    # „Co robi / Najlepsze do”: label and sentence are separate text nodes, each an EXACT catalog entry (as one node the
    # text went through the placeholder patterns and C3 lost to a generic one – it stayed Polish in the English UI)
    assert "<span>Co robi:</span> <span>${does}</span>" in js and "<span>Najlepsze do:</span> <span>${best}</span>" in js
    assert "Co robi: ${" not in js and "Najlepsze do: ${" not in js
    assert en.get("Co robi:") and en.get("Najlepsze do:")
    pairs = re.findall(r"\$\{what\('((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'\)\}", js)
    assert len(pairs) == 5
    for does, best in pairs:
        for s in (does, best):
            assert en.get(s.replace("\\'", "'")), s


def test_cleanup_mode_hides_stage_panes_with_one_css_rule():
    """Defect (a): objects.js rewrites the inline display of its panes (setStage) – Cleanup hides them with a stylesheet
    rule and never touches that inline style (nothing stale to restore in the cutting mode)."""
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    rule = "body.mode-clean #stagePanel > :not(.c3-pane):not(#cleanupPanel) { display: none !important; }"
    assert rule in css and css.index(rule) < css.index("/* Cleanup 3D steps C1-C6")
    cu = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    assert "classList.toggle('mode-clean', clean)" in cu and "cuHidden" not in cu and "panel.children" not in cu


def test_area_editor_reuses_the_curve_editor_and_the_brush_core():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    tags = [html.index(f'<script src="/static/{n}"></script>') for n in
            ("brushcore.js", "brush.js", "cleanup3d.js", "curves.js", "cleanup3d_areas.js", "tips.js")]
    assert tags == sorted(tags)
    ar = (STATIC / "cleanup3d_areas.js").read_text(encoding="utf-8")
    # ONE Bezier implementation (static/curves.js) and one stroke geometry (static/brushcore.js)
    assert "new (CV().Editor)(" in ar and "CV().fromPoints(" in ar and "CV().segments(" in ar
    assert "bezierCurveTo" in ar and "function bez(" not in ar and "function split(" not in ar and "de Casteljau" not in ar
    assert "new (BC().StrokeInput)(" in ar and "BC().strokePath(" in ar and "lineCap" not in ar
    brush = (STATIC / "brush.js").read_text(encoding="utf-8")
    core = (STATIC / "brushcore.js").read_text(encoding="utf-8")
    assert "window.MD_BRUSHCORE.regionCanvas(" in brush and "window.MD_BRUSHCORE.paintRegion(" in brush
    assert "lineCap" not in brush and "destination-out" not in brush     # the geometry lives in the core only
    assert "function strokePath(" in core and "function paintRegion(" in core and "class StrokeInput" in core
    # the editor talks to the routes of the P0 contract
    api = (ROOT / "api" / "cleanup3d.py").read_text(encoding="utf-8")
    for js_url, route in (("/api/cleanup3d/keys/${encodeURIComponent(T.card)}/${T.frame}/areas", '/api/cleanup3d/keys/{card}/{frame}/areas"'),
                          ("/api/cleanup3d/cards/${encodeURIComponent(T.card)}/extent", '/api/cleanup3d/cards/{card}/extent"'),
                          ("/api/cleanup3d/mark/${T.frame}", '/api/cleanup3d/mark/{frame}"')):
        assert js_url in ar and route in api, js_url
    head = (STATIC / "cleanup3d.js").read_text(encoding="utf-8")
    head = head[:head.index("(function ()")]
    for name in (".extentMount()", ".setCardExtent(id, areas, frame)"):
        assert name in head, name
