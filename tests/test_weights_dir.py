# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights outside ComfyUI's folders (Hugging Face cache) – user decision 2026-11-10: the Doctor's choice > the HF_HOME
variable > <models_dir>/huggingface; the setting moves what is already downloaded (a move, nothing deleted) and
applies to this process (and so to its workers)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine import config as C


def test_priority(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "USER_HF_HOME", "")
    md = str(tmp_path / "models")
    assert C.weights_dir({"models_dir": md}) == (f"{md}/huggingface".replace("\\", "/"), "folder modeli ComfyUI")
    monkeypatch.setattr(C, "USER_HF_HOME", "E:/hf")
    assert C.weights_dir({"models_dir": md})[0] == "E:/hf"
    assert C.weights_dir({"models_dir": md, "hf_home": "G:/w"}) == ("G:/w", "config.local.json")
    monkeypatch.setattr(C, "USER_HF_HOME", "")
    assert C.weights_dir({})[1] == "domyślny Hugging Face"
    assert "hf_home" in C.PATH_KEYS


def test_set_moves_and_applies(monkeypatch, tmp_path):
    import app as A
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "hub" / "models--a--b" / "snapshots" / "x").mkdir(parents=True)
    (old / "hub" / "models--a--b" / "snapshots" / "x" / "w.bin").write_bytes(b"1234")
    (new / "hub" / "models--c--d").mkdir(parents=True)                       # already there: stays
    (old / "hub" / "models--c--d").mkdir(parents=True)
    local = tmp_path / "config.local.json"
    monkeypatch.setattr(C, "LOCAL", local)
    monkeypatch.setattr(C, "USER_HF_HOME", str(old))
    monkeypatch.setattr(A, "load_config", lambda: C.load_config(local=local))
    monkeypatch.delenv("HF_HOME", raising=False)
    c = TestClient(A.app)
    r = c.post("/api/config/weights", json={"path": str(new), "move": True}).json()
    assert r["path"] == str(new).replace("\\", "/") and r["source"] == "config.local.json" and r["active"]
    assert (new / "hub" / "models--a--b" / "snapshots" / "x" / "w.bin").exists() and not (old / "hub" / "models--a--b").exists()
    assert r["moved"] == ["models--a--b"] and len(r["kept_in_old"]) == 1 and (old / "hub" / "models--c--d").exists()
    import os
    assert os.environ["HF_HOME"].replace("\\", "/") == str(new).replace("\\", "/")
    g = c.get("/api/config/weights").json()
    assert g["used_gb"] >= 0 and g["default"].replace("\\", "/") == str(old).replace("\\", "/")
    back = c.post("/api/config/weights", json={"path": "", "move": False}).json()     # empty = the default again
    assert back["source"] == "zmienna HF_HOME"
    assert c.post("/api/config/weights", json={"path": "relative/dir", "move": False}).status_code == 400


def test_finds_a_model_already_in_comfy_models(tmp_path):
    """A ComfyUI node's plain folder (models/vitmatte/vitmatte-base-composition-1k on the workstation) counts as
    installed and is what from_pretrained loads; an incomplete folder does not count."""
    from engine import weights_find as W
    from engine.stage_models import hf_cached
    W._find.cache_clear()
    md = tmp_path / "models"
    d = md / "vitmatte" / "vitmatte-base-composition-1k"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert W.local_dir("hustvl/vitmatte-base-composition-1k", md) is None          # no weights yet
    W._find.cache_clear()
    (d / "pytorch_model.bin").write_bytes(b"0" * 1_100_000)
    assert W.local_dir("hustvl/vitmatte-base-composition-1k", md) == d
    assert hf_cached("hustvl/vitmatte-base-composition-1k", md) and not hf_cached("x/other", md)
    assert not hf_cached("hustvl/vitmatte-base-composition-1k")                   # no models_dir: HF cache only
    W._find.cache_clear()


def test_doctor_ui_field():
    js = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text(encoding="utf-8")
    assert "async weightsDirHtml(inp)" in js and "this.wireWeightsDir();" in js and "'/api/config/weights'" in js
