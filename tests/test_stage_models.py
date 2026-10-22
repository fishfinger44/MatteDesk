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


def test_vitmatte_stage_open_with_birefnet_only(env):
    """Stage 5 needs one edge model: BiRefNet_HR-matting / Lucida alone (no ViTMatte, commercial work) unlock it too."""
    tmp, cfg = env
    assert "vitmatte" in G.model_gate(cfg)
    _touch(tmp / "hf" / "hub" / "models--egeorcun--lucida" / "snapshots" / "abc" / "model.safetensors")
    assert "vitmatte" not in G.model_gate(cfg)


def test_edge_engine_menu_and_switch(proj, env, monkeypatch):
    """Stage 5 engine (user decision 2026-10-21): ViTMatte by default; BiRefNet when only it is installed; switching
    (menu shown only with both installed) moves the object's ViTMatte zones to the BiRefNet default model and back."""
    import shutil
    from engine.zones import Zone, load_zones, save_zone
    app_module, api, shot, pipe, frames, start = proj
    tmp, cfg = env
    real = app_module.load_config
    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: {**real(*a, **k), **cfg})
    d = app_module._objects().dir("obj")
    save_zone(d, Zone("z_01", "add", "matte", start, matting="vitmatte", params={"region": "outline", "model_px": 504}))
    save_zone(d, Zone("z_02", "add", "matte", start, matting="vitmatte", params={"region": "outline", "vitmatte_model": "small"}))
    vm_cache = tmp / "hf" / "hub" / "models--hustvl--vitmatte-base-composition-1k"
    _touch(vm_cache / "snapshots" / "abc" / "config.json")
    s = api.get("/api/object/obj/stages").json()
    assert s["edge_engines"] == {"vitmatte": True, "birefnet": False} and s["edge_engine"] == "vitmatte"
    assert api.put("/api/object/obj/edge_engine", json={"engine": "birefnet"}).status_code == 409   # not installed
    _touch(tmp / "hf" / "hub" / "models--ZhengPeng7--BiRefNet_HR-matting" / "snapshots" / "abc" / "model.safetensors")
    s = api.get("/api/object/obj/stages").json()
    assert s["edge_engines"] == {"vitmatte": True, "birefnet": True} and s["edge_engine"] == "vitmatte"   # both: ViTMatte
    r = api.put("/api/object/obj/edge_engine", json={"engine": "birefnet"}).json()
    assert r["edge_engine"] == "birefnet" and sorted(r["zones_changed"]) == ["z_01", "z_02"]
    zs = {z.zone_id: z.params for z in load_zones(d)}
    assert zs["z_01"]["vitmatte_model"] == "birefnet_hr_matting" and "model_px" not in zs["z_01"]
    assert api.get("/api/object/obj/stages").json()["edge_engine"] == "birefnet"
    assert app_module._edge_engine_default_model("obj") == "birefnet_hr_matting"   # new zones follow the stage engine
    r = api.put("/api/object/obj/edge_engine", json={"engine": "vitmatte"}).json()
    assert all("vitmatte_model" not in z.params for z in load_zones(d))
    shutil.rmtree(vm_cache)                                   # only BiRefNet left: the stage is BiRefNet, no menu
    s = api.get("/api/object/obj/stages").json()
    assert s["edge_engines"] == {"vitmatte": False, "birefnet": True} and s["edge_engine"] == "birefnet"


def test_birefnet_stage_no_matanyone_zones(proj, env, monkeypatch):
    """BiRefNet stage (user report 2026-10-21): the zone engine list offers no ViTMatte / MatAnyone2 – a MatAnyone2 zone
    moves to BiRefNet on the switch, a zone switched to the per-frame engine gets the BiRefNet model, MatAnyone2 = 409."""
    from engine.zones import Zone, load_zones, save_zone
    app_module, api, shot, pipe, frames, start = proj
    tmp, cfg = env
    real = app_module.load_config
    monkeypatch.setattr(app_module, "load_config", lambda *a, **k: {**real(*a, **k), **cfg})
    d = app_module._objects().dir("obj")
    save_zone(d, Zone("z_01", "add", "matte", start, matting="matanyone", params={"region": "outline"}))
    save_zone(d, Zone("z_02", "add", "matte", start, matting="chroma", params={"region": "outline"}))
    _touch(tmp / "hf" / "hub" / "models--hustvl--vitmatte-base-composition-1k" / "snapshots" / "abc" / "config.json")
    _touch(tmp / "hf" / "hub" / "models--egeorcun--lucida" / "snapshots" / "abc" / "model.safetensors")
    r = api.put("/api/object/obj/edge_engine", json={"engine": "birefnet"}).json()
    assert r["zones_changed"] == ["z_01"]
    z1 = {z.zone_id: z for z in load_zones(d)}["z_01"]
    assert z1.matting == "vitmatte" and z1.params["vitmatte_model"] == "birefnet_hr_matting"
    assert api.patch("/api/object/obj/zones/z_02", json={"matting": "matanyone"}).status_code == 409
    assert api.patch("/api/object/obj/zones/z_02", json={"matting": "vitmatte"}).status_code == 200
    assert {z.zone_id: z for z in load_zones(d)}["z_02"].params["vitmatte_model"] == "birefnet_hr_matting"
    api.put("/api/object/obj/edge_engine", json={"engine": "vitmatte"})
    assert api.patch("/api/object/obj/zones/z_02", json={"matting": "matanyone"}).status_code == 200   # ViTMatte stage: allowed
