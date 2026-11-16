# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless run of the ViTMatte zone panel (static/zones.js) and the brush fixes (static/brush.js) in jsdom with a fake
API – tests/js/zones_harness.mjs.

Needs Node and jsdom: `cd tests/js && npm ci` (versions in tests/js/package-lock.json, node_modules outside git). Skipped
with that reason when either is missing – a skip here means the panels were NOT checked."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "zones_harness.mjs"


def test_zone_panel_flow_in_jsdom():
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – harness jsdom panelu Strefy pominięty")
    probe = subprocess.run([node, "-e", "require('jsdom')"], capture_output=True, text=True, cwd=str(HARNESS.parent))
    if probe.returncode != 0:
        pytest.skip("brak pakietu jsdom – zainstaluj: cd tests/js && npm ci; panele Strefy / Pędzel NIE sprawdzone")
    r = subprocess.run([node, str(HARNESS)], capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, f"harness zones.js:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    assert "OK zones harness" in r.stdout
