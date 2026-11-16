# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""REMOVE never names the removed object (2026-11-12; lesson of AInVFX „Paint a plate”: an erased object comes back
when its name reaches the prompt): the instruction of the Qwen3-VL-4B assistant per mode, the object's label passed only
as the thing to ignore, the deterministic guard on the assistant's answer (a fake model), and what is NOT touched –
REPLACE, the user's own text, the wording of the run's prompt builders."""
import json
from pathlib import Path

from engine import h3_fill, vace_fill
from engine import prompt_assist as PA
from engine import prompt_rules as PR
from engine.cleanup3d import regen

ROOT = Path(__file__).resolve().parent.parent


class FakeWorker:
    """Answers with a fixed description; keeps the requests (system / user texts the model would get)."""

    def __init__(self, tmp, description, clip=""):
        self.tmp, self.description, self.clip, self.reqs = Path(tmp), description, clip, []

    def call(self, req, cancel_check=None):
        self.reqs.append(req)
        return {"ok": True, "load_s": 0, "gen_s": 0, "tokens": 0,
                "text": json.dumps({"description": self.description, "clip": self.clip, "notes": []})}


LEAK = ("An empty sandy beach under soft overcast light. The kobieta in the white dress is gone from the shore. "
        "Wet sand with gentle ripples continues to the waterline.")


def run(tmp_path, monkeypatch, description, mode="remove", names=("kobieta",), text="", transition=False, clip=""):
    monkeypatch.setenv("MD_PROMPT_FAKE", "1")
    w = FakeWorker(tmp_path, description, clip)
    return PA.assist(text, mode, transition, worker=w, names=names), w.reqs[0]


def test_remove_instruction_forbids_naming_and_differs_from_replace():
    rm, rp = PR.assist_rules("remove"), PR.assist_rules("replace")
    assert rm != rp and PR.NO_OBJECT_RULE in rm and PR.NO_OBJECT_RULE not in rp
    assert "BEHIND" in rm and "background" in rm and "This job is REMOVE" in rm and "This job is REPLACE" not in rm
    assert "build the description around their words" in rp and "This job is REMOVE" not in rp
    assert PR.assist_rules() == rm and PR.ASSIST_RULES == rp                 # the default mode of the tool = remove
    for common in ("Always English", "transforms", "[cite: 1]", "JSON"):     # the shared rules stay in both
        assert common in rm and common in rp


def test_request_carries_the_mode_rules_and_never_the_label(tmp_path, monkeypatch):
    _, req = run(tmp_path, monkeypatch, "plain wall", names=("kobieta", "suknia"), text="pusta plaża")
    assert req["system"] == PR.assist_rules("remove")
    u = req["user"]
    assert "Do not name or describe the object" in u and u.endswith("pusta plaża")
    assert "kobieta" not in u and "suknia" not in u       # the labels only feed the guard, the model never sees them
    # replace: the replace rules, the label is not sent at all
    _, req = run(tmp_path, monkeypatch, "a porcelain mask", mode="replace", text="porcelain")
    assert req["system"] == PR.assist_rules("replace")
    assert "kobieta" not in req["user"] and "IGNORE" not in req["user"] and "Do not name" not in req["user"]
    # no label known: no label line, the rule of the mode stays
    u = PA.user_text("", "remove", False, True)
    assert "IGNORE" not in u and "Do not name or describe the object" in u


def test_guard_drops_the_sentence_with_the_object_name(tmp_path, monkeypatch):
    d, _ = run(tmp_path, monkeypatch, LEAK)
    assert d["description"] == ("An empty sandy beach under soft overcast light. "
                                "Wet sand with gentle ripples continues to the waterline.")
    assert d["guard"]["words"] == ["kobieta"] and len(d["guard"]["dropped"]) == 1
    assert any("kobieta" in n and "Pominięto" in n for n in d["notes"])
    assert not PR.named(d["description"], ["kobieta"])
    # case, the label with a number / separators, a group with its members
    d, _ = run(tmp_path, monkeypatch, "Grey wall. Kobieta stands there. The Left Wall is bare. Floor of oak boards.",
               names=("kobieta_2", "left_wall"))
    assert d["description"] == "Grey wall. Floor of oak boards." and set(d["guard"]["words"]) == {"kobieta", "left wall"}


def test_guard_whole_words_only_and_clean_answers_untouched(tmp_path, monkeypatch):
    clean = "Plain white wall, soft daylight from the left. Fine plaster texture continues behind."
    d, _ = run(tmp_path, monkeypatch, clean, names=("wal", "all", "x1", "o"))
    assert d["description"] == clean and d["guard"] == {"dropped": [], "words": []} and d["notes"] == []
    assert PR.name_terms(["x1", "o", "", None]) == []                        # too short to be a word of the text
    assert PR.name_terms(["Kobieta_2"]) == ["kobieta_2", "kobieta 2", "kobieta"]
    assert PR.strip_named("The rigging is steel.", ["rig"])["text"] == "The rigging is steel."


def test_guard_everything_dropped_gives_an_empty_proposal_with_a_note(tmp_path, monkeypatch):
    d, _ = run(tmp_path, monkeypatch, "The scene without the kobieta.")
    assert d["description"] == "" and d["lint"] == []
    assert any("pusty opis" in n for n in d["notes"])
    assert h3_fill.h3_prompt(d["description"], "remove") == h3_fill.REMOVE_PROMPT   # = the program's instruction alone


def test_guard_covers_the_clip_text_of_a_transition(tmp_path, monkeypatch):
    d, _ = run(tmp_path, monkeypatch, "Plain wall.", transition=True, clip="The kobieta fades. Wall stays lit.")
    assert d["clip"] == "Wall stays lit." and d["description"] == "Plain wall."


def test_replace_mode_untouched(tmp_path, monkeypatch):
    text = "The kobieta's face replaced with a white porcelain mask, gold filigree. Hair stays unchanged."
    d, _ = run(tmp_path, monkeypatch, text, mode="replace")
    assert d["description"] == text and d["guard"] == {"dropped": [], "words": []} and d["notes"] == []


def test_user_text_is_never_changed():
    typed = "usuń kobieta z plaży, zostaje mokry piasek"
    u = PA.user_text(typed, "remove", False, True, ["kobieta"])
    assert u.endswith("The user's text (any language):\n" + typed)            # sent to the assistant as typed
    # the lint only warns (the check endpoint passes the labels); the description field is not rewritten anywhere
    w = [x for x in PR.lint("the kobieta on wet sand", "remove", names=["kobieta"]) if x["code"] == "object_named"]
    assert len(w) == 1 and w[0]["level"] == "warn" and w[0]["words"] == "kobieta"
    assert not [x for x in PR.lint("the kobieta on wet sand", "replace", names=["kobieta"]) if x["code"] == "object_named"]
    assert not [x for x in PR.lint("wet sand", "remove", names=["kobieta"]) if x["code"] == "object_named"]
    assert not [x for x in PR.lint("the kobieta on wet sand", "remove") if x["code"] == "object_named"]
    # the run's builders pass the typed description through as it is
    for built in (h3_fill.h3_prompt(typed, "remove"), vace_fill.prompt_for(typed, "remove"), regen.qwen_prompt(typed, "remove")):
        assert typed in built


def test_run_builders_never_name_the_object_themselves():
    """Nothing but the description reaches a model prompt: the builders take no object name and their fixed REMOVE
    texts say only „the masked / marked object”."""
    import inspect
    for fn in (h3_fill.h3_prompt, vace_fill.prompt_for, regen.qwen_prompt, h3_fill.derived_prompt, h3_fill.ref_prompt):
        assert not {"name", "names", "obj", "object", "label"} & set(inspect.signature(fn).parameters), fn.__name__
    assert h3_fill.h3_prompt("", "remove") == h3_fill.REMOVE_PROMPT and "masked object removed" in h3_fill.REMOVE_PROMPT
    assert vace_fill.prompt_for("", "remove") == vace_fill.REMOVE_PROMPT
    assert regen.qwen_prompt("", "remove") == regen.QWEN_PROMPT and "marked object" in regen.QWEN_PROMPT
    assert h3_fill.derived_prompt("anything typed", "remove").count("anything typed") == 0   # remove: no description


def test_api_passes_the_labels_of_the_cleaned_thing():
    src = (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "names=_removed_names(s)))" in src                                  # the assistant
    assert 'PR.lint(s.get("prompt"), pm, fin["transition"], names=_removed_names(s))' in src
    js = (ROOT / "static" / "cleanup_prompt.js").read_text(encoding="utf-8")
    assert "d.description ||" in js                                            # an empty proposal is shown as such


def test_replace_keeps_the_users_look_and_no_example_text_2026_11_13():
    """"psychodelick woman face" came back as a porcelain face: the 4B model copied the porcelain example of rule 7.
    The rules have no concrete look any more, REPLACE builds on the user's words, and a dropped word is flagged."""
    for mode in ("replace", "remove"):
        r = PR.assist_rules(mode).lower()
        assert "porcelain" not in r and "never copy any example" in r
    assert "the user's idea" in PR.assist_rules("replace").lower() and "never describe how the object looks now" in PR.assist_rules("replace").lower()
    bad = ("A woman with voluminous, curly brown hair. Her face is replaced with a smooth, pale, porcelain-like "
           "complexion. Her dark top and curly hair remain unchanged.")
    assert PR.missing_terms("psychodelick woman face", bad) == ["psychodelick"]
    good = "Her face is covered in vivid psychedelic swirls of neon magenta and acid green. Hair and earrings unchanged."
    assert PR.missing_terms("psychodelick woman face", good) == []
    assert PR.missing_terms("psychodeliczna twarz", bad) == []        # Polish: translated, never flagged (no false alarm)
    assert PR.missing_terms("rdzawy samochód", "a rusty red car") == []
    # any object, not only faces: the user's look words are checked the same way
    assert PR.missing_terms("rusty vintage car with chrome", "A shiny new blue car. The street stays unchanged.") == ["rusty", "vintage", "chrome"]
    assert PR.missing_terms("rusty vintage car with chrome", "A rusty vintage car, flaking paint, chrome bumpers.") == []
    u = PA.user_text("psychodelick woman face", "replace", False, True)
    assert "build the description around it" in u and u.endswith("psychodelick woman face")


def test_remove_stays_inside_the_outline_any_object_2026_11_15():
    """User 2026-11-15 (earrings -> hair: the proposal said "hair frames the face"): REMOVE describes only what fills the
    outline and continues at its edge, never things fully outside it, short – for any object."""
    rm, rp = PR.assist_rules("remove"), PR.assist_rules("replace")
    assert "Stay INSIDE the outline" in rm and "frames the face" in rm and "Stay INSIDE the outline" not in rp
    assert "15–60 words" in rm and "40–110 words" in rp and "{words}" not in rm + rp


def test_qwen_anchor_names_the_removed_object_2026_11_15():
    """An earring stayed in every anchor: Qwen-Image-2.1 gets the real image with nothing marked – its instruction now
    names the removed object (the SAM 3.1 prompt); the video prompt still never names it."""
    from engine import h3_fill
    from engine.cleanup3d import regen
    p = regen.qwen_prompt("curly hair around the ear", "remove", "earrings")
    assert p.startswith("Remove the earrings (the marked object)") and "In its place there should be: curly hair" in p
    assert regen.qwen_prompt("x", "remove") .startswith("Remove the marked object")          # no name: as before
    assert "Replace" in regen.qwen_prompt("a red hat", "replace", "earrings") and "earrings" not in regen.qwen_prompt("a red hat", "replace", "earrings")
    assert "the earrings (the masked object) removed" in h3_fill.derived_prompt("hair", "remove", "earrings")
    assert "earrings" not in h3_fill.h3_prompt("curly hair around the ear", "remove")       # the video model: never
    src = (Path(__file__).resolve().parents[1] / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "def _removed_noun(s" in src and "PR.final_prompts(s, _removed_noun(s))" in src
    assert "what=_removed_noun(s) if init is None else None" in src
