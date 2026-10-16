# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless run of the stage-4 zone panel (static/zones.js) in jsdom with a fake API – tests/js/zones_harness.mjs.

Needs Node and the `jsdom` package: `MD_NODE_PATH=<dir>/node_modules` (or NODE_PATH) pointing at an install with
jsdom, e.g. `npm install jsdom@24` in any folder. Skipped (with the reason) when either is missing, so the suite
stays green on the station without Node tooling; in the cloud session it ran green (2026-10-13)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "zones_harness.mjs"


def _node_env():
    env = dict(os.environ)
    np_ = env.get("MD_NODE_PATH") or env.get("NODE_PATH")
    if np_:
        env["NODE_PATH"] = np_
    return env


def test_zone_panel_flow_in_jsdom():
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – harness jsdom panelu Strefy pominięty")
    env = _node_env()
    probe = subprocess.run([node, "-e", "require('jsdom')"], capture_output=True, text=True, env=env, cwd=str(ROOT))
    if probe.returncode != 0:
        pytest.skip("brak pakietu jsdom (ustaw MD_NODE_PATH na node_modules z jsdom) – harness pominięty")
    r = subprocess.run([node, str(HARNESS)], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, f"harness zones.js:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    assert "OK zones harness" in r.stdout
