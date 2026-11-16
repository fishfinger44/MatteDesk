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


def test_klein_and_lora_list_exactly_their_files(env):
    from engine.stage_models import KLEIN_FILES, KLEIN_LORA
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    klein = [_put(md / folder / fn, 20) for folder, fn, *_ in KLEIN_FILES.values()]
    part = _put(md / "diffusion_models" / "flux3-image-klein-bf16.safetensors.part", 5)   # interrupted download too
    lora = _put(md / "loras" / KLEIN_LORA["lora"][1], 7)
    other = [_put(md / "vae" / "flux1-vae.safetensors"), _put(md / "loras" / "my_style.safetensors"),
             _put(md / "diffusion_models" / "flux-2-klein-base-4b-fp8.safetensors")]
    # the int8 variant (2026-11-10, chosen by the card) is FLUX 3 Image Klein too
    int8 = _put(md / "diffusion_models" / "flux3-image-klein-int8-convrot.safetensors", 3)
    got = {m["key"]: m for m in model_trash.listing(cfg)}
    assert {p["path"] for p in got["flux3_klein"]["paths"]} == {str(p).replace("\\", "/") for p in klein + [part, int8]}
    assert got["flux3_klein"]["bytes"] == 68 and got["klein_restore_lora"]["bytes"] == 7
    assert [p["path"] for p in got["klein_restore_lora"]["paths"]] == [str(lora).replace("\\", "/")]
    assert "LoRA Restore" in got["flux3_klein"]["affects"] and "LaMa" in got["flux3_klein"]["affects"]
    r = model_trash.remove(cfg, "flux3_klein")
    assert r["freed_bytes"] == 68 and not r["errors"]
    assert not any(p.exists() for p in klein + [part, int8]) and lora.exists() and all(p.exists() for p in other)
    model_trash.remove(cfg, "klein_restore_lora")
    assert not lora.exists() and all(p.exists() for p in other)


def test_qwen21_lists_exactly_its_files(env):
    from engine.stage_models import QWEN21_FILES
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    mine = [_put(md / folder / fn, 10) for folder, fn, *_ in QWEN21_FILES.values()]
    other = [_put(md / "vae" / "qwen_image_vae.safetensors"),
             _put(md / "diffusion_models" / "qwen_image_edit_2511_fp8mixed.safetensors"),
             _put(md / "text_encoders" / "qwen3vl_4b_bf16.safetensors")]       # FLUX 3 Image Klein's encoder stays
    got = {m["key"]: m for m in model_trash.listing(cfg)}
    assert got["qwen21"]["label"] == "Qwen-Image-2.1" and got["qwen21"]["bytes"] == 30
    assert "Qwen-Image-2.1" in got["qwen21"]["affects"] and "LaMa" in got["qwen21"]["affects"]
    r = model_trash.remove(cfg, "qwen21")
    assert r["freed_bytes"] == 30 and not r["errors"]
    assert not any(p.exists() for p in mine) and all(p.exists() for p in other)


def test_new_cleanup_models_list_only_their_own_files(env):
    """2026-11-11: Wan 2.2 Fun VACE, VOID (native) and SeedVR2 – the shared encoders / VAEs stay."""
    from engine import vace_fill as VF
    from engine.cleanup3d import void_engine as VE
    tmp, cfg = env
    md = Path(cfg["models_dir"])
    vace = [_put(md / "diffusion_models" / VF.HIGH), _put(md / "diffusion_models" / VF.LOW)]
    void = [_put(md / "diffusion_models" / VE.VOID_NATIVE_UNET), _put(md / "vae" / VE.VOID_NATIVE_VAE)]
    sv = [_put(md / "SEEDVR2" / VE.SEEDVR2_DITS["3b"][0]), _put(md / "SEEDVR2" / VE.SEEDVR2_VAE)]
    shared = [_put(md / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"), _put(md / "vae" / "wan_2.1_vae.safetensors"),
              _put(md / "text_encoders" / "t5xxl_fp16.safetensors")]
    got = {m["key"]: m for m in model_trash.listing(cfg)}
    assert got["vace"]["bytes"] == 20 and got["void"]["bytes"] == 20 and got["seedvr2"]["bytes"] == 20
    assert "MiniMax-H3" in got["vace"]["affects"] and "Upscale" in got["seedvr2"]["affects"]
    for key in ("vace", "void", "seedvr2"):
        assert not model_trash.remove(cfg, key)["errors"]
    assert not any(p.exists() for p in vace + void + sv) and all(p.exists() for p in shared)
