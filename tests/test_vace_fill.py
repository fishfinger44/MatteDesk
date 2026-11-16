from pathlib import Path
# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wan 2.2 Fun VACE as the second Inpaint (C1) engine (user decision 2026-11-11): canvas / length, the native graph
(two GGUF experts, high noise then low noise), anchors as given control frames, the main anchor as the reference,
status from the files. Synthetic: no ComfyUI."""
import numpy as np

from engine import cleanup as cu
from engine import vace_fill as V


def test_canvas_length_prompt():
    assert V.canvas(1080, 1920, 720) == (720, 1280) and V.canvas(1920, 1080, 480) == (848, 480)
    assert [V.clip_length(n) for n in (1, 5, 6, 50)] == [5, 5, 9, 53]
    assert "replaced: a porcelain mask" in V.prompt_for("a porcelain mask", "replace")
    assert V.prompt_for("", "replace") == V.REMOVE_PROMPT
    assert "turns" in V.prompt_for("her face slowly turns to porcelain", "replace", transition=True)
    assert "vace" in cu.MODELS and "vace" in cu.ANCHOR_MODELS and "vace" in cu.VIDEO_MODELS
    assert cu.settings({"vace_res": 480})["vace_res"] == 480 and cu.settings({"vace_res": 9})["vace_res"] == 720


def test_graph_two_experts_and_reference(tmp_path):
    f = {"high": "h.gguf", "low": "l.gguf", "clip": "c.safetensors", "vae": "v.safetensors"}
    g = V.graph(tmp_path, tmp_path, tmp_path / "r", "p", "x", 720, 1280, 53, f, seed=7)
    assert g["1"]["class_type"] == "UnetLoaderGGUF" and g["1"]["inputs"]["unet_name"] == "h.gguf"
    hi, lo = g["21"]["inputs"], g["22"]["inputs"]
    assert (hi["start_at_step"], hi["end_at_step"], hi["add_noise"]) == (0, V.SPLIT, "enable")
    assert (lo["start_at_step"], lo["add_noise"], lo["latent_image"]) == (V.SPLIT, "disable", ["21", 0])
    assert hi["model"] == ["7", 0] and lo["model"] == ["8", 0] and hi["noise_seed"] == 7
    assert g["20"]["inputs"]["reference_image"] == ["13", 0] and g["23"]["inputs"]["trim_amount"] == ["20", 3]
    assert "reference_image" not in V.graph(tmp_path, tmp_path, None, "p", "x", 720, 1280, 53, f)["20"]["inputs"]


def test_anchor_frames_are_given():
    masks = [np.full((4, 6), 255, np.uint8)] * 3
    anchor = np.full((8, 12, 3), 200, np.uint8)
    ctrl, msk = V.control_frames([1, 2, 3], lambda f: np.full((8, 12, 3), 50, np.uint8), masks, {2: anchor}, 6, 4)
    assert ctrl[0].max() == 127 and msk[0].max() == 255                  # grey inside the mask, generated
    assert ctrl[1].min() == 200 and msk[1].max() == 0                    # the anchor frame: given, nothing generated


def test_status_from_files(tmp_path):
    st = V.status({"models_dir": str(tmp_path)})
    assert not st["ok"] and V.HIGH in st["reason"] and "Doktor" in st["reason"]
    for folder, name in (("diffusion_models", V.HIGH), ("diffusion_models", V.LOW), ("text_encoders", V.CLIPS[0]),
                         ("vae", V.VAES[0])):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"x")
    assert V.status({"models_dir": str(tmp_path)})["ok"]

    class C:
        base_url = "x"

        def get_object_info(self, n):
            return {} if n == "UnetLoaderGGUF" else {n: {}}
    st = V.status({"models_dir": str(tmp_path)}, C())
    assert not st["ok"] and "ComfyUI-GGUF" in st["reason"]


def test_lightning_fast_graph(tmp_path):
    """„Szybko” (user 2026-11-11): the lightx2v 4-step LoRAs on both experts, 4 steps at CFG 1; without the LoRAs (or
    „dokładnie”) the 20-step graph."""
    f = {"high": "h.gguf", "low": "l.gguf", "clip": "c", "vae": "v", "lora_high": "lh.safetensors", "lora_low": "ll.safetensors"}
    g = V.graph(tmp_path, tmp_path, None, "p", "x", 720, 1280, 53, f, seed=1, fast=True)
    assert g["31"]["inputs"]["lora_name"] == "lh.safetensors" and g["7"]["inputs"]["model"] == ["31", 0]
    assert g["8"]["inputs"]["model"] == ["32", 0] and g["7"]["inputs"]["shift"] == V.FAST_SHIFT
    hi, lo = g["21"]["inputs"], g["22"]["inputs"]
    assert (hi["steps"], hi["cfg"], hi["end_at_step"], lo["start_at_step"]) == (4, 1.0, 2, 2)
    slow = V.graph(tmp_path, tmp_path, None, "p", "x", 720, 1280, 53, {**f, "lora_low": None}, fast=True)
    assert "31" not in slow and slow["21"]["inputs"]["steps"] == V.STEPS and slow["21"]["inputs"]["cfg"] == V.CFG
    assert cu.settings({})["vace_speed"] == "fast" and cu.settings({"vace_speed": "quality"})["vace_speed"] == "quality"


def test_install_lightning_downloads_both_once(tmp_path):
    from engine import installer
    got = []

    def dl(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x" * 10)
    res = installer.install_vace_lightning({"models_dir": str(tmp_path)}, download=dl)
    assert res["success"] and len(got) == 2 and all("Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/" in u for u in got)
    assert V.has_lightning(V.files({"models_dir": str(tmp_path)})) and "vace_lightning" in installer.INSTALL_ACTIONS


def test_negative_names_the_removed_object_2026_11_15(tmp_path):
    f = {"high": "h", "low": "l", "vae": "v", "clip": "c"}
    g = V.graph(tmp_path, tmp_path, None, "p", "x", 720, 1280, 53, f, negative="earrings")
    assert g["6"]["inputs"]["text"].startswith("earrings, ") and V.NEGATIVE in g["6"]["inputs"]["text"]
    assert V.graph(tmp_path, tmp_path, None, "p", "x", 720, 1280, 53, f)["6"]["inputs"]["text"] == V.NEGATIVE
    src = (Path(__file__).resolve().parents[1] / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "negative=_removed_noun(s))" in src
