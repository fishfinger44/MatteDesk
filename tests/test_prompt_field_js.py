# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bug 2026-10-15: changing the SAM3 prompt „did nothing” – Enter in the prompt field only saved it (the stage-1 Enter
shortcut skips text fields), the button did not show the frame as outdated, and the blur save reloaded the object
(dropping unsaved points). Now: Enter in the field = „▶ Przelicz klatkę”, typing marks the frame, the save does not reload."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "static" / "objects.js").read_text(encoding="utf-8")


def test_prompt_field_enter_recomputes_and_save_does_not_reload():
    i = SRC.index("$('st1Prompt').addEventListener('change'")
    block = SRC[i:SRC.index("$('st1PtsClear').addEventListener", i)]
    assert "refreshList" not in block and "run('st1Prompt'" not in block
    assert "$('st1Prompt').addEventListener('keydown'" in block and "computePoints(V().currentFrame)" in block
    assert "$('st1Prompt').addEventListener('input'" in block and "S.pointsDirty[V().currentFrame] = true" in block
