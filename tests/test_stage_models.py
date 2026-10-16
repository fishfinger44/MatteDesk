# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Model gate of the stages (user rule 2026-10-16): a stage whose model is not downloaded / installed is unavailable –
locked in the stage bar with the reason (the Doctor's button); a done result stays usable. File checks only."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine import stage_models as G   # noqa: E402


def _touch(p: Path, size: int = 1):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(b"0" * size)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    import engine.videomama as V
    import engine.ltx_alpha as L
    monkeypatch.setattr(V, "VMM_FILES", [(s, r, f, 0.0) for s, r, f, _g in V.VMM_FILES])
    monkeypatch.setattr(L, "LTX_MODELS", {k: (a, b, c, 0.0) for k, (a, b, c, _g) in L.LTX_MODELS.items()})
    cfg = {"models_dir": str(tmp_path / "models"), "comfy_dir": str(tmp_path / "comfy")}
    (tmp_path / "comfy" / "custom_nodes").mkdir(parents=True)
    return tmp_path, cfg


def test_gate_per_stage(env):
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    g = G.model_gate(cfg)
    assert set(g) == {"detect", "crops", "core", "cotracker", "matanyone", "videomama", "vitmatte", "ltx"}
    assert "Doktor" in g["videomama"]["why"] and "ComfyUI-VideoMaMa" in g["videomama"]["why"]
    _touch(md / "checkpoints" / "sam3.1_multiplex_fp16.safetensors")
    _touch(md / "cotracker" / "scaled_offline.pth")
    _touch(md / "matanyone2" / "matanyone2.pth")
    _touch(tmp / "hf" / "hub" / "models--hustvl--vitmatte-base-composition-1k" / "snapshots" / "abc" / "config.json")
    g = G.model_gate(cfg)
    assert set(g) == {"videomama", "ltx"}
    import engine.videomama as V
    import engine.ltx_alpha as L
    for sub, _r, fn, _g in V.VMM_FILES:
        _touch(md / "videomama" / sub / fn)
    assert "videomama" in G.model_gate(cfg)                     # weights alone are not enough: the node too
    (tmp / "comfy" / "custom_nodes" / V.NODE_PACK).mkdir()
    for _k, (folder, fn, _repo, _gb) in L.LTX_MODELS.items():
        _touch(md / folder / fn)
    assert G.model_gate(cfg) == {}


def test_gate_off_and_no_models_dir(env, monkeypatch):
    _tmp, cfg = env
    g = G.model_gate({"comfy_dir": cfg["comfy_dir"]})
    assert "models_dir" in g["matanyone"]["why"]
    monkeypatch.setenv("MD_MODEL_GATE", "0")
    assert G.model_gate(cfg) == {}


def test_apply_gate_locks_step_keeps_done_result():
    info = {"stages": [{"id": "matanyone", "state": "done", "why": ""}, {"id": "videomama", "state": "todo", "why": ""},
                       {"id": "vitmatte", "state": "locked", "why": "x"}],
            "steps": [{"id": "matanyone", "state": "done", "why": ""}, {"id": "videomama", "state": "todo", "why": ""}]}
    gate = {"videomama": {"missing": ["a"], "why": "Brak modelu VideoMaMa – Doktor"},
            "matanyone": {"missing": ["b"], "why": "Brak modelu MatAnyone2 – Doktor"}}
    G.apply_gate(info, gate, {"matanyone": ["matanyone"], "videomama": ["videomama"]})
    st = {s["id"]: s for s in info["stages"]}
    assert st["matanyone"]["state"] == "done"                     # an existing result stays usable (export, bases)
    assert st["videomama"]["state"] == "locked" and "Doktor" in st["videomama"]["why"]
    steps = {s["id"]: s for s in info["steps"]}
    assert steps["matanyone"]["state"] == "locked" and steps["videomama"]["state"] == "locked"   # cannot be opened / run
    assert set(info["models_missing"]) == {"videomama", "matanyone"}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_api_stages_and_require(proj, env, monkeypatch):
    app_module, api, shot, pipe, frames, start = proj
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    for p in ("checkpoints/sam3.1_multiplex_fp16.safetensors", "cotracker/scaled_offline.pth"):
        _touch(md / p)
    _touch(tmp / "hf" / "hub" / "models--hustvl--vitmatte-base-composition-1k" / "snapshots" / "abc" / "config.json")
    real = app_module.load_config
    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: {**real(*a, **k), **cfg})
    d = api.get("/api/object/obj/stages").json()
    steps = {s["id"]: s for s in d["steps"]}
    for sid in ("matanyone", "videomama", "ltx"):
        assert steps[sid]["state"] == "locked" and "Doktor" in steps[sid]["why"], sid
    assert steps["sam3"]["state"] != "locked" and steps["export"]["state"] != "locked"
    assert set(d["models_missing"]) >= {"matanyone", "videomama", "ltx"}
    r = api.post("/api/object/obj/finalize", json={"stage": "matanyone"})
    assert r.status_code == 409 and "MatAnyone2" in r.json()["detail"]
