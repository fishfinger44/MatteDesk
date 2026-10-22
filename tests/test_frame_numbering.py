# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_frame_numbering.py - new shot from a 1001-numbered sequence: keep the file range or start at N (mocha-like)."""

import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from engine.ingest import MediaInspector, ProxyGenerator, renumber_sequence


def _seq(folder, name, first, last, value=0):
    folder.mkdir(parents=True, exist_ok=True)
    for f in range(first, last + 1):
        img = np.full((32, 48, 3), (value + f) % 256, dtype=np.uint8)
        cv2.imwrite(str(folder / f"{name}.{f:04d}.png"), img)
    return folder


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "_remember_work_dir", lambda wd: None)   # never touch config.local.json
    return TestClient(app_module.app)


def test_printf_pattern_resolves(tmp_path):
    _seq(tmp_path / "plate", "sh010_beauty", 1001, 1004)
    info = MediaInspector.resolve_sequence_pattern(str(tmp_path / "plate" / "sh010_beauty.%04d.png"))
    assert (info["first_frame"], info["last_frame"], info["frame_count"]) == (1001, 1004, 4)
    assert info["pattern"].endswith("sh010_beauty.####.png")


def test_renumber_sequence_links_same_pixels(tmp_path):
    src = _seq(tmp_path / "plate", "sh010_beauty", 1001, 1003)
    res = renumber_sequence(str(src / "sh010_beauty.####.png"), 1001, 1003, 1, tmp_path / "out")
    assert (res["first_frame"], res["last_frame"]) == (1, 3)
    for k in range(3):
        a = ProxyGenerator.resolve_frame_path(res["pattern"], 1 + k)
        b = src / f"sh010_beauty.{1001 + k:04d}.png"
        assert a.read_bytes() == b.read_bytes()
    # reuse: a second call changes nothing
    assert renumber_sequence(str(src / "sh010_beauty.####.png"), 1001, 1003, 1, tmp_path / "out")["pattern"] == res["pattern"]


def test_new_shot_keeps_file_range_by_default(client, tmp_path):
    src = _seq(tmp_path / "plates" / "beauty", "sh010_beauty", 1001, 1005)
    r = client.post("/api/project/open_or_create", json={
        "work_dir": str(tmp_path / "work"), "shot_name": "SH010", "plates": {"beauty": str(src)}})
    assert r.status_code == 200, r.text
    m = r.json()["manifest"]
    assert (m["first"], m["last"]) == (1001, 1005)
    assert "source_first" not in m
    assert m["plates"]["beauty"].endswith("plates/beauty/sh010_beauty.####.png")


def test_new_shot_start_frame_renumbers_beauty_and_aux(client, tmp_path):
    beauty = _seq(tmp_path / "plates" / "beauty", "sh010_beauty", 1001, 1005)
    backlit = _seq(tmp_path / "plates" / "backlit", "sh010_backlit", 1001, 1005, value=100)
    work = tmp_path / "work"
    r = client.post("/api/project/open_or_create", json={
        "work_dir": str(work), "shot_name": "SH010", "first_frame": 1,
        "plates": {"beauty": str(beauty / "sh010_beauty.1001.png"), "backlit": str(backlit)}})
    assert r.status_code == 200, r.text
    m = r.json()["manifest"]
    assert (m["first"], m["last"], m["source_first"]) == (1, 5, 1001)
    for name, src, stem in (("beauty", beauty, "sh010_beauty"), ("backlit", backlit, "sh010_backlit")):
        pat = m["plates"][name]
        assert "/source/" in pat and pat.endswith(".####.png")
        assert ProxyGenerator.resolve_frame_path(pat, 1).read_bytes() == (src / f"{stem}.1001.png").read_bytes()
        assert ProxyGenerator.resolve_frame_path(pat, 5).read_bytes() == (src / f"{stem}.1005.png").read_bytes()
    # source plates untouched, manifest on disk has the new range
    assert len(os.listdir(beauty)) == 5
    saved = json.loads((work / "shot_manifest_SH010.json").read_text(encoding="utf-8"))
    assert (saved["first"], saved["last"]) == (1, 5)
