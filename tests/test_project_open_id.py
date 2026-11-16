# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 #15: one open of a project has one `open_id` – in the POST response, in the WS `project_opened` and
in /api/project/current – so the page applies it once (static/app.js `applyOpened`). Every open gets a new one, also
the same shot opened again, and across a server restart the ids do not repeat."""

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app


@pytest.fixture
def shot(tmp_path):
    plates = tmp_path / "plates"
    plates.mkdir()
    for f in range(1, 4):
        cv2.imwrite(str(plates / f"t_{f:04d}.png"), np.zeros((20, 40, 3), np.uint8))
    mf = tmp_path / "shot_manifest_OPENID.json"
    mf.write_text(json.dumps({"shot": "OPENID", "fps": 24, "first": 1, "last": 3, "plate_w": 40, "plate_h": 20,
                              "colorspace": "sRGB", "source_type": "img_seq", "bit_depth": 8,
                              "plates": {"beauty": str(plates / "t_####.png")}, "work": str(tmp_path)}), encoding="utf-8")
    return tmp_path, mf


@pytest.mark.parametrize("route", ["open", "open_or_create"])
def test_one_open_one_id_everywhere(shot, monkeypatch, route):
    work, mf = shot
    sent = []

    async def broadcast(msg):
        sent.append(msg)
    monkeypatch.setattr(app_module.ws_manager, "broadcast", broadcast)
    client = TestClient(app)
    url = f"/api/project/{route}"
    body = {"manifest_path": str(mf)} if route == "open" else {"work_dir": str(work), "plates": {}}
    first = client.post(url, json=body).json()
    assert first["open_id"]
    assert client.get("/api/project/current").json()["open_id"] == first["open_id"]
    assert [m["open_id"] for m in sent if m["type"] == "project_opened"] == [first["open_id"]]
    again = client.post(url, json=body).json()      # the same shot opened again is a new open
    assert again["open_id"] != first["open_id"]
    assert client.get("/api/project/current").json()["open_id"] == again["open_id"]
