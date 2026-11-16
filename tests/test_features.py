# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parts switched off in a release (engine/features.py, user decision 2026-11-16): `feature_cleanup3d` off = the Cleanup
3D routes answer 404 with the reason (GET /api/cleanup3d stays for the C1 state), /api/version tells the UI, the
stage bar shows only C1, the Doctor hides the C2–C6-only tools, the public copy writes the switch off."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from engine import features

ROOT = Path(__file__).resolve().parents[1]


def test_enabled_reads_the_config(tmp_path):
    from engine.config import load_config
    sh, lo = tmp_path / "config.json", tmp_path / "config.local.json"
    sh.write_text(json.dumps({"feature_cleanup3d": False}), encoding="utf-8")
    cfg = load_config(sh, lo, appdata="")
    assert features.enabled("cleanup3d", cfg) is False and features.all_features(cfg) == {"cleanup3d": False}
    lo.write_text(json.dumps({"feature_cleanup3d": True}), encoding="utf-8")       # this machine turns it back on
    assert features.enabled("cleanup3d", load_config(sh, lo, appdata="")) is True
    sh.write_text("{}", encoding="utf-8"); lo.write_text("{}", encoding="utf-8")
    assert features.enabled("cleanup3d", load_config(sh, lo, appdata="")) is True    # default on (development)


def test_cleanup3d_routes_gated(monkeypatch):
    c = TestClient(app_module.app)
    assert c.get("/api/version").json()["features"]["cleanup3d"] is True
    monkeypatch.setattr(features, "enabled", lambda name, cfg=None: False)
    v = c.get("/api/version").json()
    assert v["features"] == {"cleanup3d": False}
    r = c.get("/api/cleanup3d/cards")
    assert r.status_code == 404 and "feature_cleanup3d" in r.json()["detail"]
    assert c.post("/api/cleanup3d/camera/solve").status_code == 404
    # the mode's state stays reachable (the C1 step reads it); without a project it is the usual 400, not the gate
    assert c.get("/api/cleanup3d").status_code != 404 or "feature_cleanup3d" not in c.get("/api/cleanup3d").text
    assert c.get("/api/cleanup/state").status_code != 404 or "feature_cleanup3d" not in c.get("/api/cleanup/state").text


def test_ui_and_doctor_and_public_copy_follow_the_switch():
    js = (ROOT / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    assert "features.cleanup3d === false" in js and "ALL_STEPS.slice(0, 1)" in js and "C3.c3on === false" in js
    doc = (ROOT / "doctor.py").read_text(encoding="utf-8")
    assert '_feature_on("cleanup3d", self.config)' in doc and "SEEDVR2_ITEM}" in doc and "VOID_ITEM" not in doc.split("c3_only = ")[1].split("\n")[1]
    pub = (ROOT / "tools" / "export_public.py").read_text(encoding="utf-8")
    assert 'cfg["feature_cleanup3d"] = False' in pub
