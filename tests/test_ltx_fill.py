"""LTX-2.5 as an Inpaint (C1) engine (2026-11-13): Lightricks' IC-LoRA inpaint on native nodes + the C1 anchors."""
from pathlib import Path

import numpy as np
import pytest

from engine import cleanup as cu
from engine import ltx_fill as LF

FILES = {"unet": "u.safetensors", "text_encoder": "t.safetensors", "video_vae": "v.safetensors",
         "audio_vae": "a.safetensors", "lora": LF.LORA[1]}


def test_canvas_and_length():
    assert LF.canvas(3840, 2160) == (1024, 576)
    assert LF.canvas(1440, 2560) == (576, 1024)
    cw, ch = LF.canvas(4096, 2731)
    assert cw % 64 == 0 and ch % 64 == 0 and max(cw, ch) == 1024
    assert LF.canvas(640, 360) == (640, 384)           # never up-scaled past the plate (sides snapped to 64)
    assert [LF.clip_length(n) for n in (1, 9, 10, 50, 57)] == [9, 9, 17, 57, 57]


def test_green_key_like_lightricks_preprocess():
    im = np.full((4, 4, 3), 10, np.uint8)
    m = np.zeros((4, 4), np.uint8)
    m[:2] = 255
    out = LF.green_frames([im], [m])[0]
    assert tuple(out[0, 0]) == LF.GREEN_BGR == (0, 255, 102)   # #66FF00 in BGR
    assert tuple(out[3, 3]) == (10, 10, 10)


def test_graph_iclora_guide_then_anchor_keyframes(tmp_path):
    g = LF.graph(tmp_path, "p", "a porcelain face", 1024, 576, 57, FILES, seed=7,
                 anchors=[(4, tmp_path), (30, tmp_path)])
    assert g["31"]["inputs"]["lora_name"] == LF.LORA[1]
    assert g["39"]["inputs"]["iclora_parameters"] == ["51", 0] and g["39"]["inputs"]["frame_idx"] == 0
    a1, a2 = g["61"], g["63"]
    assert a1["class_type"] == a2["class_type"] == "LTXVAddGuide"
    assert a1["inputs"]["latent"] == ["39", 2] and a2["inputs"]["latent"] == ["61", 2]
    assert (a1["inputs"]["frame_idx"], a2["inputs"]["frame_idx"]) == (4, 30)
    assert "iclora_parameters" not in a1["inputs"]                # keyframes are plain guides
    assert g["42"]["inputs"]["video_latent"] == ["63", 2]
    assert g["44"]["inputs"]["positive"] == ["63", 0] and g["49"]["inputs"]["positive"] == ["63", 0]
    assert g["46"]["inputs"]["sigmas"] == LF.SIGMAS and g["43"]["inputs"]["noise_seed"] == 7
    assert g["35"]["inputs"]["text"] == "a porcelain face"
    assert LF.graph(tmp_path, "p", "", 64, 64, 9, FILES)["35"]["inputs"]["text"] == " "   # "" crashes the tokenizer


def test_prompt_is_a_scene_description():
    assert LF.prompt_for("", "remove") == ""
    assert LF.prompt_for("plain wall.", "remove").startswith("plain wall.")
    assert "first frame to the last" in LF.prompt_for("a porcelain face", "replace")


def test_status_without_lora(tmp_path):
    from engine.ltx_alpha import LTX_MODELS
    for k in LF.BASE_KEYS:
        folder, fn = LTX_MODELS[k][0], LTX_MODELS[k][1]
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / fn).write_bytes(b"x")
    st = LF.status({"models_dir": str(tmp_path)})
    assert not st["ok"] and "IC-LoRA" in st["reason"]
    (tmp_path / "loras").mkdir(exist_ok=True)
    (tmp_path / "loras" / LF.LORA[1]).write_bytes(b"x")
    assert LF.status({"models_dir": str(tmp_path)})["ok"]


def test_registered_as_c1_anchor_engine():
    for lst in (cu.MODELS, cu.PROMPT_MODELS, cu.ANCHOR_MODELS, cu.VIDEO_MODELS, cu.C1_MODELS):
        assert "ltx" in lst
    assert cu.MODEL_LABEL["ltx"] == "LTX-2.5" and cu.c1_model("ltx") == "ltx"
    from engine.prompt_rules import final_prompts
    fp = final_prompts({"model": "ltx", "prompt": "a porcelain face", "prompt_mode": "replace"})
    assert any(i["model"] == "LTX-2.5" for i in fp["items"])


def test_installer_needs_hf_login(tmp_path, monkeypatch):
    from engine import installer
    monkeypatch.setattr(installer, "hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="gated"):
        installer.install_ltx_inpaint({"models_dir": str(tmp_path)})
    got = []
    r = installer.install_ltx_inpaint({"models_dir": str(tmp_path)}, download=lambda u, d: got.append((u, d)))
    assert r["success"] and got[0][0].endswith(f"{LF.LORA[2]}/resolve/main/{LF.LORA[1]}")
    assert Path(got[0][1]) == tmp_path / "loras" / LF.LORA[1]
