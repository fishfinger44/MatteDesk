# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„✨ Popraw opis” (engine/prompt_assist.py + engine/prompt_worker.py, user 2026-11-12): the worker protocol (fake model
in this Python), the answer parsing, the outline of the mask, the lease and the Doctor / trash / installer wiring."""
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import prompt_assist as PA

ROOT = Path(__file__).resolve().parent.parent


def test_parse_tolerates_fences_and_keeps_fields():
    d = PA.parse('```json\n{"description": "white porcelain face", "clip": "", "notes": "ok"}\n```')
    assert d == {"description": "white porcelain face", "clip": "", "notes": ["ok"]}
    with pytest.raises(PA.PromptAssistError):
        PA.parse("no json here")
    with pytest.raises(PA.PromptAssistError):
        PA.parse('{"description": ""}')


def test_outline_draws_red_and_scales_down():
    img = np.zeros((2000, 1000, 3), np.uint8)
    m = np.zeros((2000, 1000), np.float32)
    m[800:1200, 300:700] = 1
    out = PA.outlined(img, m)
    assert max(out.shape[:2]) == PA.IMAGE_SIDE
    red = (out[..., 2] > 200) & (out[..., 1] < 50)
    assert red.any() and not red[:300].any()


def test_worker_protocol_fake(monkeypatch, tmp_path):
    monkeypatch.setenv("MD_PROMPT_FAKE", "1")
    w = PA.PromptWorker(python=sys.executable, env={"MD_PROMPT_FAKE": "1"})
    try:
        img = np.zeros((64, 64, 3), np.uint8)
        d = PA.assist("biała porcelanowa twarz", "replace", False, img, np.ones((64, 64), np.float32), worker=w)
    finally:
        w.close()
    assert d["description"].startswith("FAKE:") and "biała porcelanowa twarz" in d["description"]
    assert d["clip"] == "" and any(x["code"] == "polish" for x in d["lint"])   # the lint runs on the answer too


def test_user_text_names_mode_and_transition():
    t = PA.user_text("abc", "replace", True, True)
    assert "REPLACE" in t and "TRANSITION" in t and "outlined in red" in t and t.endswith("abc")
    assert "REMOVE" in PA.user_text("", "remove", False, False) and "propose a description" in PA.user_text("", "remove", False, False)


def test_wiring_doctor_trash_installer_lease():
    from engine import installer, model_trash
    from engine.gpu_manager import LEASED_CALLS
    import doctor
    assert ("engine/prompt_assist.py", "PromptWorker", "call") in LEASED_CALLS
    assert installer.INSTALL_ACTIONS["prompt_assist"] is installer.install_prompt_assist
    assert "prompt_assist" in model_trash.MODELS
    assert doctor.MODULE_OF[doctor.PROMPT_ASSIST_ITEM] == "cleanup"
    assert "prompt_assist" in doctor.LICENSES
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "install_prompt_assist: ['prompt_assist'" in app_js and "'comfy_rollback', 'prompt_assist']" in app_js
    assert "Qwen3-VL-4B-Instruct" in (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    assert "Qwen3-VL-4B" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_install_downloads_the_snapshot(tmp_path):
    from engine import installer
    got = []

    def dl(url, dst):
        got.append(url)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"x")
    r = installer.install_prompt_assist({"models_dir": str(tmp_path)}, download=dl)
    assert r["success"] and r["restart_comfy"] is False and len(got) == len(PA.FILES)
    assert all(u.startswith("https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/main/") for u in got)


def test_ui_wiring():
    js = (ROOT / "static" / "cleanup_prompt.js").read_text(encoding="utf-8")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert '<script src="/static/cleanup_prompt.js"></script>' in html
    assert "if (!st.available) { await installPopup(st); return; }" in js           # no model = the install popup
    assert "/api/cleanup/prompt/check" in js and "/api/cleanup/prompt/assist" in js
    cl = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert 'id="cuAssist"' in cl and 'id="cuPromptLint"' in cl and 'id="cuFinal"' in cl
