# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor's 🗑 (engine/model_trash.py): only weights of optional models, only inside models_dir / the HF cache."""
from pathlib import Path

import pytest

from engine import model_trash


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    md = tmp_path / "models"
    return tmp_path, {"models_dir": str(md)}


def _put(p: Path, n: int = 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    return p


def test_listing_and_remove_only_the_model_files(env):
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    from engine.ltx_alpha import LTX_MODELS
    folder, fn, _r, _g = LTX_MODELS["video_vae"]
    ltx = _put(md / folder / fn, 100)
    other = _put(md / folder / "someone_elses_vae.safetensors")       # shared vae/ folder: must stay
    sam3 = _put(md / "checkpoints" / "sam3.1_multiplex_fp16.safetensors")   # required: never listed
    mat = _put(md / "matanyone2" / "matanyone2.pth", 30)                    # optional since 2026-10-23
    _put(tmp / "hf" / "hub" / "models--ZhengPeng7--BiRefNet_HR-matting" / "snapshots" / "a" / "m.safetensors")  # required
    sap = _put(tmp / "hf" / "hub" / "models--facebook--sapiens2-seg-0.4b" / "snapshots" / "a" / "model.safetensors", 50)
    got = {m["key"]: m for m in model_trash.listing(cfg)}
    assert set(got) == {"ltx", "sapiens2", "matanyone2"}
    assert got["ltx"]["bytes"] == 100 and got["sapiens2"]["bytes"] == 50
    r = model_trash.remove(cfg, "ltx")
    assert r["freed_bytes"] == 100 and not ltx.exists() and other.exists() and sam3.exists()
    model_trash.remove(cfg, "sapiens2")
    assert not sap.exists() and not (tmp / "hf" / "hub" / "models--facebook--sapiens2-seg-0.4b").exists()
    model_trash.remove(cfg, "matanyone2")
    assert not mat.exists() and model_trash.listing(cfg) == []


def test_required_models_cannot_be_removed(env):
    _tmp, cfg = env
    for key in ("sam3", "cotracker", "birefnet", "../x"):   # BiRefNet = the required stage-5 engine (2026-10-23)
        with pytest.raises(ValueError):
            model_trash.remove(cfg, key)


def test_paths_outside_the_model_folders_are_skipped(env, monkeypatch):
    tmp, cfg = env
    outside = _put(tmp / "elsewhere" / "big-lama.pt")
    monkeypatch.setitem(model_trash.MODELS, "lama", ("LaMa", "-", lambda c: [outside]))
    assert model_trash.listing(cfg) == []
    r = model_trash.remove(cfg, "lama")
    assert outside.exists() and r["removed"] == [] and r["errors"]
