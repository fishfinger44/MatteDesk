# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step C1 fill models (user 2026-11-10): LaMa, Qwen-Image-2.1 (prompt, frame by frame) and VOID (prompt, the whole
shot; its frames are the key fill, LaMa the rest). Run.json names what filled; the panel lists the models with their
install state, VRAM verdict and the prompt field. Synthetic: the models are stand-ins."""
from pathlib import Path

import cv2
import numpy as np

from engine import cleanup as cu
from test_cleanup import H, W, scene

ROOT = Path(__file__).resolve().parent.parent


def test_settings_models_and_prompt():
    s = cu.settings({"model": "h3", "prompt": "plain wall", "anchor": "first_last"})
    assert s["model"] == "h3" and s["prompt"] == "plain wall" and s["anchor"] == "frame"   # 2026-11-11: one anchor
    assert s["anchor_frame"] is None and cu.settings({"anchor_frame": 12})["anchor_frame"] == 12
    assert cu.settings({"anchor_frame": -1})["anchor_frame"] is None
    assert cu.settings({"model": "minimax"})["model"] == "lama" and cu.settings({})["prompt"] == ""
    # 2026-11-10: Qwen-Image-2.1 is the H3 anchor, not a C1 fill (an image model flickers frame by frame)
    assert cu.settings({"model": "qwen21"})["model"] == "lama" and "qwen21" not in cu.MODELS
    assert set(cu.PROMPT_MODELS) == {"void", "h3", "vace", "ltx"} and "klein" not in " ".join(cu.MODELS)
    assert cu.settings({"anchor": "x"})["anchor"] == "none"


def test_h3_anchors_pinned_in_the_graph(tmp_path):
    """Qwen-Image-2.1 look frame(s) pinned with MiniMaxH3AddGuide at their clip index; cache key follows the frame,
    description and mask (user 2026-11-10)."""
    from engine import h3_fill as Hf
    # ONE anchor (user 2026-11-11): the chosen frame in the range, else the frame where the object is largest and
    # least cut by the frame edge
    def mask(f):
        m = np.zeros((60, 100), np.float32)
        if f == 1:
            m[0:40, 0:40] = 1          # larger but cut by two edges
        elif f == 2:
            m[10:40, 30:60] = 1        # whole, a bit smaller -> the suggestion
        elif f == 3:
            m[20:30, 40:50] = 1        # whole, small
        return m
    assert Hf.suggest_anchor([1, 2, 3, 4], mask) == 2
    assert Hf.anchor_frame([1, 2, 3], "frame", 3, mask) == 3 and Hf.anchor_frame([1, 2, 3], "frame", 9, mask) == 2
    assert Hf.anchor_frame([1, 2, 3], "none", 3, mask) is None
    files = {"unet": "u", "clip": "c", "vae": "v", "turbo": "t"}
    g = Hf.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, files, guides=[(0, tmp_path / "a"), (49, tmp_path / "b")])
    adds = sorted((n["inputs"]["frame_idx"], n["inputs"]["positive"]) for n in g.values() if n["class_type"] == "MiniMaxH3AddGuide")
    assert [a[0] for a in adds] == [0, 49]
    assert g["23"]["inputs"]["conditioning"] == ["63", 0] and g["61"]["inputs"]["positive"] == ["20", 0]   # chained
    assert Hf.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, files)["23"]["inputs"]["conditioning"] == ["20", 0]
    # 2026-11-11: the anchor also conditions the whole clip as <Picture 1> (AddGuide alone held only nearby frames)
    assert g["20"]["class_type"] == "MiniMaxH3ReferenceToVideo" and g["20"]["inputs"]["ref_images.ref_image_0"] == ["60", 0]
    assert "<Picture 1>" in g["20"]["inputs"]["prompt"] and g["20"]["inputs"]["prompt"].endswith("x")
    assert Hf.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, files)["20"]["class_type"] == "MiniMaxH3ImageToVideo"
    # derived anchors at the range's ends (user test v004: H3 reached the main anchor only next to it)
    # the start one on the 5th frame: on F1 (a latent token of its own) it came out as Qwen's frame (test v005)
    assert Hf.derived_frames(list(range(1, 51)), 31) == [5, 50] and Hf.derived_frames(list(range(1, 51)), 3) == [50]
    assert Hf.derived_frames(list(range(1, 51)), 46) == [5] and Hf.derived_frames([1, 2], None) == []
    assert "Picture 2" in Hf.derived_prompt("a porcelain face", "replace")
    from engine.cleanup3d.regen import qwen_graph
    qg = qwen_graph(tmp_path, "p", "x", 7, tmp_path / "r")
    assert qg["8"]["inputs"]["images.image_2"] == ["5", 0] and qg["12"]["inputs"]["seed"] == 7
    assert "5" not in qwen_graph(tmp_path, "p", "x")
    assert cu.settings({"seed": 5})["seed"] == 5 and cu.settings({"seed": "x"})["seed"] == 42
    m = np.zeros((40, 40), np.float32)
    m[10:20, 10:20] = 1
    assert Hf.anchor_key(1, "a", "replace", m, 1) != Hf.anchor_key(1, "a", "replace", m, 2)
    assert Hf.anchor_key(1, "a", "replace", m, 1, "a_0031_x") != Hf.anchor_key(1, "a", "replace", m, 1)
    k = Hf.anchor_key(1, "a mask", "replace", m)
    assert k == Hf.anchor_key(1, "a mask", "replace", m) and k != Hf.anchor_key(1, "a mask", "remove", m)
    m2 = m.copy()
    m2[25:35, 25:35] = 1
    assert k != Hf.anchor_key(1, "a mask", "replace", m2) and k != Hf.anchor_key(2, "a mask", "replace", m)


def test_void_frames_fill_the_mask_lama_the_rest(tmp_path):
    bg, obj, keep, plate = scene()

    class FakeVoid:
        label = "VOID"

        def __call__(self, f):
            return (None if f == 2 else (bg.copy(), np.ones((H, W), np.float32)))   # F2: no VOID frame -> LaMa
    lama_calls = []

    def lama(imgs, masks):
        lama_calls.append(len(imgs))
        return [np.where(m[..., None] > 0, 0.5, im).astype(np.float32) for im, m in zip(imgs, masks)]
    c = cu.Cleaner({"model": "void", "fill": "lama", "grow_px": 2, "soft_px": 0}, W, H, False, plate=lambda f: plate,
                   remove_alphas=lambda f: [obj], keep_alphas=lambda f: [keep], inpaint=lama, key_fill=FakeVoid())
    rj = c.run([1, 2], tmp_path / "v001", "png")
    assert rj["backend"] == "void" and rj["counts"]["VOID"] > 0 and rj["counts"]["LaMa"] > 0 and lama_calls == [1]
    comp = cv2.imread(str(tmp_path / "v001" / "comp" / "c_0001.png")).astype(np.float32) / 255
    assert abs(comp[60, 60] - bg[60, 60]).max() < 2 / 255                       # the VOID frame inside the mask
    assert np.abs(comp[5, 5] - plate[5, 5]).max() <= 1 / 255 + 1e-6            # the plate outside


def test_void_key_fill_reads_the_run_frames(tmp_path, monkeypatch):
    from engine.cleanup3d import void_engine as VE
    img = np.full((8, 12, 3), 128, np.uint8)
    monkeypatch.setattr(VE, "void_frame", lambda work, run, f, size: cv2.resize(img, size) if f == 1 else (_ for _ in ()).throw(ValueError("x")))
    kf = cu.VoidKeyFill(tmp_path, {"id": "run_x"}, W, H, False)
    got = kf(1)
    assert got[0].shape == (H, W, 3) and abs(float(got[0][0, 0, 0]) - 128 / 255) < 1e-3 and got[1].min() == 1.0
    assert kf(2) is None


def test_remove_or_replace_instruction():
    """„Usuń / Zastąp” (user 2026-11-10): only the instruction around the description changes."""
    from engine.cleanup3d.regen import QWEN_PROMPT, qwen_prompt
    assert qwen_prompt("", "remove") == QWEN_PROMPT == qwen_prompt("", "replace")          # nothing to replace with
    rm = qwen_prompt("a white wall", "remove")
    assert rm.startswith(QWEN_PROMPT) and "In its place there should be: a white wall." in rm
    rp = qwen_prompt("the woman's face with a porcelain mask.", "replace")
    assert rp.startswith("Replace the woman's face with a porcelain mask.") and "background" not in rp
    assert cu.settings({"prompt_mode": "replace"})["prompt_mode"] == "replace"
    assert cu.settings({"prompt_mode": "x"})["prompt_mode"] == "remove"
    seen = {}
    import engine.cleanup3d.regen as R
    orig = R._run_qwen
    try:
        R._run_qwen = lambda client, io, real, region, prompt, progress, mode="remove": (seen.update(p=prompt, m=mode) or (real, None, 1))
        q = cu.QwenInpainter(None, Path("."), "a mask", "replace")
        q([np.zeros((4, 4, 3), np.float32)], [np.ones((4, 4), np.uint8)])
    finally:
        R._run_qwen = orig
    assert seen == {"p": "a mask", "m": "replace"}
    js = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert 'id="cuPromptMode"' in js and "save({ prompt_mode: b.dataset.pm })" in js
    # 2026-11-11: Inpaint offers only MiniMax-H3 / Wan 2.2 Fun VACE (LaMa / VOID are in the projection steps)
    api = (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "usuwanie LaMą / " in api and cu.C1_MODELS == ("h3", "vace", "ltx") and cu.c1_model("lama") == "h3"


def test_h3_clip_canvas_prompt_and_graph(tmp_path):
    """MiniMax-H3 (user 2026-11-10): the model's canvas and 17k+5 length, the mask grown in space and time, the
    native masked V2V graph (no custom node), remove / replace prompt, frames back as the key fill."""
    from engine import h3_fill as H
    assert H.canvas(1080, 1920) == (768, 1344) and H.canvas(1920, 1080) == (1344, 768)
    assert [H.clip_length(n) for n in (1, 5, 6, 39, 40, 50)] == [5, 5, 22, 39, 56, 56]
    m = np.zeros((64, 64), np.float32)
    m[30, 30] = 1
    g = H.grow_masks([np.zeros((64, 64)), m, np.zeros((64, 64))], px=4, radius=1)
    assert g[0][30, 34] == 255 and g[2][30, 26] == 255 and g[1][30, 36] == 0     # grown 4 px, spread to +-1 frame
    files = {"unet": "u.safetensors", "clip": "c.safetensors", "vae": "v.safetensors", "turbo": "t.safetensors"}
    gr = H.graph(tmp_path / "f", tmp_path / "m", "p", "x", 768, 1344, 56, files)
    kinds = {n["class_type"] for n in gr.values()}
    assert {"SetLatentNoiseMask", "LTXVConcatAVLatent", "MiniMaxH3ImageToVideo", "SamplerCustomAdvanced"} <= kinds
    assert gr["2"]["inputs"]["type"] == "minimax" and gr["24"]["inputs"]["steps"] == H.STEPS
    assert gr["17"]["inputs"]["video_latent"] == ["14", 0]                       # the masked plate latent is sampled
    assert H.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, {**files, "turbo": None})["24"]["inputs"]["steps"] == H.STEPS_NO_TURBO
    assert "replaced: a porcelain mask" in H.h3_prompt("a porcelain mask", "replace")
    assert H.h3_prompt("", "replace") == H.REMOVE_PROMPT
    fill = H.H3Fill({3: np.full((1344, 768, 3), 128, np.uint8)}, 1080, 1920, False)
    img, wgt = fill(3)
    assert img.shape == (1920, 1080, 3) and wgt.min() == 1.0 and fill(4) is None
    st = H.status({"models_dir": str(tmp_path)}, None)
    assert not st["ok"] and "Comfy-Org/MiniMax-H3" in st["reason"]


def test_model_gate_checks_comfy_nodes(tmp_path):
    """Files on disk are not enough (2026-11-10: TextEncodeQwenImage21 missing in a ComfyUI 0.33 git install)."""
    from engine.stage_models import comfy_gate

    class C:
        def __init__(self, have):
            self.have = have

        def get_object_info(self, n):
            return {n: {}} if n in self.have else {}
    assert "TextEncodeQwenImage21" in comfy_gate(C(set()), "qwen21_edit") and "zaktualizuj ComfyUI" in comfy_gate(C(set()), "qwen21_edit")
    assert comfy_gate(C({"TextEncodeQwenImage21"}), "qwen21_edit") == ""
    assert comfy_gate(None, "qwen21_edit") == ""                                 # ComfyUI not answering: no verdict


def test_ui_model_picker_card_and_prompt():
    js = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    for s in ('id="cuModel"', 'id="cuPrompt"', "data-cu-install=", "v.installWithProgress(m.install)", "m && m.prompt",
              'id="cuAnchors"', "MD_CU_ANCHORS", "installModel: (m) => installModel(m)",
              'id="cuSeedRnd"', "v.bumpPlate('cleaned')"):
        assert s in js
    # the anchors on / off and the Qwen-Image-2.1 card live in the head of the anchor palette (2026-11-11)
    anc = (ROOT / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert 'id="cuAnchor"' in anc and "window.MD_CLEANUP.installModel(info.anchor_model)" in anc
    api = (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert '"anchor_model"' in api and "cu.VoidKeyFill(" in api and "anchors=anchors" in api


def test_holdout_keeps_the_kept_objects_soft_edge():
    """User report 2026-11-11: the soft hair edge (Final Touch blur) came out as a hard cut of the patch – the holdout
    thresholded the kept alpha at 0.5. The patch now goes under a kept object by its transparency."""
    hair = np.zeros((10, 10), np.float32)
    hair[:, 5:] = np.linspace(0.0, 1.0, 5)
    face = np.ones((10, 10), np.float32)
    m = cu.final_mask([face], [hair], cu.settings({"grow_px": 0, "soft_px": 0, "holdout": True}))
    assert np.allclose(m[0, 5:], 1.0 - np.linspace(0.0, 1.0, 5), atol=1e-6) and m[0, 0] == 1.0
