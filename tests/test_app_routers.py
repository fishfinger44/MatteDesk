# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 step 8: app.py's sections move into api/<module>.py routers. Contract: the route table of the app
(method, path, endpoint name – in registration order, which decides which route wins an ambiguous URL) is exactly what
it was before the split (tests/data/app_routes_2026_11_02.json, taken from main @ 214b362); every router module builds
with explicit dependencies and imports nothing from app.py; no route is registered after the `__main__` guard (the
plate-QA routes used to be – `python app.py` started uvicorn before they existed)."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute

import app as app_module

ROOT = Path(app_module.__file__).resolve().parent
SNAPSHOT = ROOT / "tests" / "data" / "app_routes_2026_11_02.json"
API_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "WS"}


def _table(routes=None, rows=None):
    """Routes in matching order; an included router (FastAPI 0.141 `_IncludedRouter`) is flattened in place."""
    rows = [] if rows is None else rows
    for r in (app_module.app.routes if routes is None else routes):
        if isinstance(r, APIRoute):
            rows.append([sorted(r.methods), r.path, r.endpoint.__name__])
        elif isinstance(r, APIWebSocketRoute):
            rows.append([["WS"], r.path, r.endpoint.__name__])
        elif getattr(r, "original_router", None) is not None:
            _table(r.original_router.routes, rows)
    return rows


def test_route_table_unchanged_by_the_split():
    want = [r for r in json.loads(SNAPSHOT.read_text(encoding="utf-8")) if r[0][0] in API_METHODS]
    got = _table()
    assert len(got) == len(want) == 303           # 195 in app.py + the deflicker, MergeMatte and colour-profile routers + 2 align residual (2026-11-03) + VRAM forecast, MergeMatte edge_bg, 4 cleanup (2026-11-06), plate content action (2026-11-07), 3 shutter (2026-11-08), 19 cleanup3d (2026-11-09, P0 skeleton) + 3 cleanup3d (2026-11-10, P8: cards/propose, two brush-layer PNG routes) + cleanup3d rebase_keys + render_settings + blend (2026-11-10) + cleanup3d keys auto / master (2026-11-10, P13) + cleanup3d export_nuke (2026-11-10) + 7 cleanup3d band (2026-11-10, P14) + cleanup3d key active / delete version (2026-11-10, key steps) + 4 horizon (P15) + 3 VOID / SeedVR2 (P12) + 3 timeline (P16) + 2 depth card (P18) + timeline solo (P16, 2026-11-10) + cleanup versions show / delete (2026-11-10) + weights folder get / set (2026-11-10) + H3 anchors list / add / patch / delete / generate / generation image + delete + import + generate linked + version upscale + static camera (2026-11-11) + 7 C1 band (2026-11-11, api/inpaint_band.py) + C1 prompt check / assistant state / assistant (2026-11-12) + H3 anchors from the card keys (2026-11-12) + cleanup3d card delete (2026-11-13) + C1 version recompose / live preview (2026-11-13) + C1 holdout overlay (2026-11-15) + 3 C1 band slider keys (2026-11-15) + C1 version model frame (2026-11-15) + object mask source (2026-11-15) + Final Touch nuke mode / export nk_text + section nk (2026-11-16)
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(want, got)) if a != b]
    assert not diff, diff[:5]


def test_router_modules_are_self_contained():
    mods = sorted(p for p in (ROOT / "api").glob("*.py") if p.name != "__init__.py")
    assert mods, "api/ has no routers"
    for p in mods:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + \
                   [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m == "app" or m.startswith("app.") for m in imported), (p.name, "imports app.py")
        make = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make"), None)
        assert make is not None, (p.name, "no make()")
        assert not make.args.args and make.args.kwonlyargs, (p.name, "make() takes keyword-only dependencies")
        # the section's request models stay at module level (FastAPI resolves annotations through module globals)
        assert not any(isinstance(n, ast.ClassDef) for n in ast.walk(make)), (p.name, "class inside make()")


def test_no_route_after_the_main_guard():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    tail = src.split('if __name__ == "__main__":', 1)[1]
    assert "@app." not in tail and "include_router" not in tail


@pytest.mark.skipif(shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff.exe").exists(),
                    reason="ruff not installed")
def test_router_modules_have_no_undefined_names():
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff.exe")
    r = subprocess.run([ruff, "check", str(ROOT / "api"), "--select", "F821,F401,F811", "--output-format", "concise"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
