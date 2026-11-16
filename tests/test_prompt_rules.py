# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Inpaint (C1) description rules (engine/prompt_rules.py, user 2026-11-12): the lint and the final texts each model
gets – built by the run's own functions, so the preview never drifts from what is sent."""
from engine import cleanup as cu
from engine import h3_fill
from engine import prompt_rules as PR
from engine.cleanup3d import regen

GALAXY = ("Close-up portrait of the exact same woman. Her skin smoothly transforms into a deep cosmic galaxy texture "
          "[cite: 1]. She keeps a slight blink and subtle head movement, static camera, 4k.")


def codes(text, mode="replace", transition=False):
    return {w["code"] for w in PR.lint(text, mode, transition)}


def test_lint_the_user_galaxy_prompt():
    c = codes(GALAXY)
    assert {"event", "motion", "cite", "filler"} <= c and "polish" not in c
    ev = next(w for w in PR.lint(GALAXY, "replace") if w["code"] == "event")
    assert "transforms" in ev["words"] and ev["level"] == "warn"


def test_lint_transition_allows_change_words():
    assert "event" not in codes("her natural skin slowly turns into porcelain", transition=True)


def test_lint_polish_empty_and_remove_wording():
    assert "polish" in codes("Biała porcelanowa twarz ze złotymi zdobieniami")
    assert "polish" not in codes("I want a white porcelain face with gold filigree")       # English "I" is not Polish
    assert codes("", "replace") == {"empty_replace"} and codes("", "remove") == set()
    assert "remove_object" in codes("remove the rig and the stand", "remove")
    assert codes("plain white wall, soft daylight", "remove") == set()


def test_final_prompts_use_the_run_builders():
    base = cu.settings({"model": "h3", "prompt": "a porcelain face", "prompt_mode": "replace", "anchor": "frame",
                        "anchors": [{"id": "a111111", "frame": 31, "kind": "main"}, {"id": "a222222", "frame": 5, "kind": "linked"}]})
    f = PR.final_prompts(base)
    by = {i["key"]: i for i in f["items"]}
    assert not f["transition"] and set(by) == {"anchor_main", "anchor_linked", "video"}
    assert by["anchor_main"]["text"] == regen.qwen_prompt("a porcelain face", "replace")
    assert by["anchor_linked"]["text"] == h3_fill.derived_prompt("a porcelain face", "replace")
    assert by["video"]["text"] == h3_fill.ref_prompt(h3_fill.h3_prompt("a porcelain face", "replace", False))
    # a transition (an original anchor): the clip prompt goes to the video model as is, no <Picture 1> reference
    tr = dict(base, h3_prompt="her skin slowly turns into porcelain",
              anchors=base["anchors"] + [{"id": "a333333", "frame": 50, "kind": "original"}])
    f = PR.final_prompts(tr)
    v = next(i for i in f["items"] if i["key"] == "video")
    assert f["transition"] and v["text"] == h3_fill.h3_prompt("her skin slowly turns into porcelain", "replace", True)
    # no anchors: only the clip; Wan 2.2 Fun VACE its own wrapper
    from engine import vace_fill
    f = PR.final_prompts(dict(base, anchor="none", model="vace"))
    assert [i["key"] for i in f["items"]] == ["video"]
    assert f["items"][0]["text"] == vace_fill.prompt_for("a porcelain face", "replace", False)


def test_assistant_rules_carry_the_lint_rules():
    r = PR.ASSIST_RULES
    for w in ("transforms", "blink", "[cite: 1]", "English", "JSON", "REMOVE", "REPLACE"):
        assert w in r, w
