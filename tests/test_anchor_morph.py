# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„Przejście między kotwicami” (user 2026-11-15, F31 -> F32 jumped at a pinned linked anchor): with `anchor_morph` and
linked anchors the clip is a transition – no single <Picture 1> look, the clip prompt asks for a gradual change from
keyframe to keyframe (MiniMax-H3 / Wan 2.2 Fun VACE)."""
from pathlib import Path

from engine import cleanup as cu
from engine import h3_anchors as HA
from engine import h3_fill as H
from engine import vace_fill as V

LST = [{"kind": "main", "frame": 42}, {"kind": "linked", "frame": 12}]


def test_setting_and_rule():
    assert cu.settings({})["anchor_morph"] is False and "anchor_morph" in cu.PER_TARGET
    assert HA.morphs({"anchor_morph": True}, LST) and not HA.morphs({}, LST)
    assert not HA.morphs({"anchor_morph": True}, LST[:1])                    # one anchor: nothing to morph
    assert HA.transition_of({"anchor_morph": True}, LST) and not HA.transition_of({}, LST)


def test_prompts_ask_for_the_morph():
    p = H.h3_prompt("a painted face", "replace", True, True)
    assert H.MORPH_SENTENCE in p and "nothing transforms" not in p
    assert H.MORPH_SENTENCE not in H.h3_prompt("a painted face", "replace", True)
    assert H.MORPH_SENTENCE in V.prompt_for("a painted face", "replace", True, True)


def test_run_and_ui_wired():
    root = Path(__file__).resolve().parents[1]
    src = (root / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "transition = HA.transition_of(s, lst)" in src and "h3_fill.h3_prompt(text, pm, transition, morph)" in src
    assert "reference=not transition" in src
    js = (root / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert 'id="cuAnchorMorph"' in js and "setLink({ anchor_morph: e.target.checked })" in js


def test_prompt_preview_no_clip_lint_for_a_morph():
    from engine import prompt_rules as PR
    s = cu.settings({"model": "h3", "anchor": "frame", "anchor_morph": True, "prompt": "a painted face", "prompt_mode": "replace",
                     "anchors": HA.defaults(42, [12])})
    fin = PR.final_prompts(s) if hasattr(PR, "final_prompts") else None
    if fin is not None:
        assert fin["transition"] and fin["morph"]
    src = (Path(__file__).resolve().parents[1] / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert 'if fin["transition"] and not fin.get("morph") else []' in src


def test_ltx_gets_the_morph_too():
    from engine import ltx_fill as L
    assert "never jumping" in L.prompt_for("a painted face", "replace", True, True)
    assert "never jumping" not in L.prompt_for("a painted face", "replace", True)
    assert "stays the same" not in L.prompt_for("a painted face", "replace", True, True)


def test_anchor_strength_2026_11_15(monkeypatch):
    """„Siła kotwicy”: main / own anchors start from the plate below 1 (Qwen-Image-2.1 denoise); 1 = drawn anew."""
    import numpy as np
    from engine.cleanup3d import regen
    assert cu.settings({})["anchor_denoise"] == 1.0 and "anchor_denoise" in cu.PER_TARGET
    assert cu.settings({"anchor_denoise": 0.1})["anchor_denoise"] == 0.3
    seen = {}

    def fake(client, io, disp, m, prompt, sem, mode, **k):
        seen.update(k)
        return disp, None, None
    monkeypatch.setattr(regen, "_run_qwen", fake)
    monkeypatch.setattr("engine.stage_models.comfy_gate", lambda *a, **k: None)
    plate = np.zeros((16, 16, 3), np.float32); mask = np.ones((16, 16), np.float32)
    H.make_anchor(None, Path("."), plate, mask, "p", "replace", plate_denoise=0.8)
    assert seen["denoise"] == 0.8
    H.make_anchor(None, Path("."), plate, mask, "p", "replace", plate_denoise=1.0)
    assert seen["denoise"] == 1.0
    src = (Path(__file__).resolve().parents[1] / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert 'own_dn = float(s.get("anchor_denoise", 1.0)) if a["kind"] in ("main", "own") else None' in src
    js = (Path(__file__).resolve().parents[1] / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert 'id="cuAnchorDn"' in js and "setLink({ anchor_denoise: +e.target.value })" in js
    assert "!['main', 'own'].includes(sk) ? 'none' : ''" in js     # hidden for a linked anchor


def test_anchor_inpaints_under_the_mask_2026_11_15(tmp_path):
    """Qwen-Image-2.1 anchors: the latent is noised only under the mask (native ImageToMask / GrowMask /
    SetLatentNoiseMask) – the model knows WHERE; without a mask dir the graph is the old edit graph."""
    from engine.cleanup3d import regen as R
    g = R.qwen_graph(tmp_path, "p", "x", mask_dir=tmp_path / "m")
    assert g["23"]["class_type"] == "SetLatentNoiseMask" and g["12"]["inputs"]["latent_image"] == ["23", 0]
    assert g["22"]["class_type"] == "GrowMask" and g["21"]["inputs"]["channel"] == "red"
    old = R.qwen_graph(tmp_path, "p", "x")
    assert "23" not in old and old["12"]["inputs"]["latent_image"] == ["10", 0]
    assert H.ANCHOR_LATENT_MASK is True
