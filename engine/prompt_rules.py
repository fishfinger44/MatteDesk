# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""How to write the Inpaint (C1) description – ONE source of the rules (user 2026-11-12):

* `lint(text, mode, transition)` – checks without any model (Polish words, „transforms…” read by a video model as an event,
  motion / camera words the plate already gives, citation junk pasted from a chat, an empty Replace);
* `final_prompts(settings)` – the exact texts each model gets, built by the SAME functions the run uses
  (regen.qwen_prompt, h3_fill.derived_prompt / h3_prompt / ref_prompt, anchor_warp.touchup_prompt, vace_fill.prompt_for);
* `assist_rules(mode)` – the same rules as the instruction of the local prompt assistant (engine/prompt_assist.py);
  REMOVE: only the background behind the object, the removed object is never named (`strip_named` guards the answer).

A new rule goes here once: the lint, the assistant's instruction and the tests read it from this module.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# the words below are matched as whole words, case-insensitively
EVENT_WORDS = ("transform", "transforms", "transforming", "transformed", "turns into", "turning into", "turn into",
               "slowly", "gradually", "morph", "morphs", "morphing", "becomes", "becoming", "changes into",
               "fades in", "fades out", "fading", "evolves", "appears", "dissolves")
MOTION_WORDS = ("blink", "blinks", "blinking", "head movement", "moves her head", "moves his head", "static camera",
                "camera pan", "camera move", "camera movement", "dolly", "zoom in", "zoom out", "handheld",
                "slow motion", "subtle movement", "subtle head movement", "moving")
FILLER_WORDS = ("8k", "4k", "masterpiece", "best quality", "ultra detailed", "highly detailed", "trending on artstation",
                "photorealistic 8k")
CITE_RE = re.compile(r"\[\s*cite\s*:[^\]]*\]|\[\d+\]", re.I)
PL_RE = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]|\b(oraz|się|jest|twarz|włosy|tło|skóra|kobieta|mężczyzna|usuń|zamień|"
                   r"na|z|w|bez|który|która|jak|ma|być|biał\w*|złot\w*|czarn\w*)\b", re.I)


def _words(text: str, words) -> List[str]:
    t = text.lower()
    return [w for w in words if re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", t)]


def lint(text: Optional[str], mode: str = "remove", transition: bool = False, names=()) -> List[Dict[str, str]]:
    """Warnings for one description: [{code, level (warn | info), msg (Polish UI text), words?}]. `names` = the labels
    of the cleaned thing: in REMOVE a description that carries one gets a warning (the text itself is never changed)."""
    t = (text or "").strip()
    out: List[Dict[str, str]] = []
    if not t:
        if mode == "replace":
            out.append({"code": "empty_replace", "level": "warn",
                        "msg": "Zastąp bez opisu = model tylko usunie obiekt. Napisz, co ma się pojawić."})
        return out
    if CITE_RE.search(t):
        out.append({"code": "cite", "level": "warn",
                    "msg": "Znaczniki „[cite: …]” (wklejone z czatu) – model weźmie je za tekst. Usuń je."})
    if len(PL_RE.findall(t)) >= 2:
        out.append({"code": "polish", "level": "warn",
                    "msg": "Opis wygląda na polski – modele rozumieją angielski. „✨ Popraw opis” przetłumaczy."})
    ev = _words(t, EVENT_WORDS)
    if ev and not transition:
        out.append({"code": "event", "level": "warn", "words": ", ".join(ev),
                    "msg": f"„{', '.join(ev)}” – model wideo zrobi z tego przemianę w połowie klipu. Opisz gotowy "
                           "wygląd (stan od pierwszej klatki). Przemiana = kotwica-oryginał + prompt klipu."})
    mv = _words(t, MOTION_WORDS)
    if mv:
        out.append({"code": "motion", "level": "info", "words": ", ".join(mv),
                    "msg": f"„{', '.join(mv)}” – ruch i kamerę daje plate; Qwen-Image-2.1 (kotwice) tego nie użyje. "
                           "Można usunąć."})
    fl = _words(t, FILLER_WORDS)
    if fl:
        out.append({"code": "filler", "level": "info", "words": ", ".join(fl),
                    "msg": f"„{', '.join(fl)}” – nic nie wnosi do edycji (rozdzielczość = plate). Można usunąć."})
    if mode == "remove" and re.match(r"\s*(remove|delete|erase)\b", t, re.I):
        out.append({"code": "remove_object", "level": "info",
                    "msg": "Usuń: opisz tło, które ma zostać (np. „plain white wall”), nie obiekt – polecenie "
                           "usunięcia dodaje program."})
    nm = named(t, names) if mode == "remove" and not transition else []
    if nm:
        out.append({"code": "object_named", "level": "warn", "words": ", ".join(nm),
                    "msg": f"„{', '.join(nm)}” to nazwa usuwanego obiektu – model wideo maluje to, co jest w opisie, więc "
                           "obiekt może wrócić. Opisz tylko tło, które ma zostać."})
    if len(t) > 900:
        out.append({"code": "long", "level": "info",
                    "msg": f"Opis ma {len(t)} znaków – modele biorą zwykle pierwsze kilkaset; ważne rzeczy na początek."})
    return out


def final_prompts(s: Dict[str, Any], what: Optional[str] = None) -> Dict[str, Any]:
    """The texts each model of the chosen C1 engine gets for the settings `s` (cleanup settings of the cleaned thing):
    {engine, transition, items: [{key, model, title, text}]}. Same builders as the run (api/cleanup.py)."""
    from engine import h3_anchors as HA
    from engine import h3_fill
    from engine.cleanup3d import regen
    model = s.get("model") or ""
    desc = (s.get("prompt") or "").strip()
    pm = s.get("prompt_mode") or "remove"
    anchors = s.get("anchors") or []
    anchors_on = s.get("anchor") == "frame" and model in ("h3", "vace", "ltx")
    transition = bool(anchors_on and HA.transition_of(s, anchors))
    morph = bool(anchors_on and HA.morphs(s, anchors) and not HA.is_transition(anchors))
    items: List[Dict[str, str]] = []
    if anchors_on and any(a.get("kind") == "main" for a in anchors):   # projection-only anchors: no Qwen-Image-2.1
        items.append({"key": "anchor_main", "model": "Qwen-Image-2.1", "title": "Kotwica główna (Qwen-Image-2.1)",
                      "text": regen.qwen_prompt(desc, pm, what)})   # `what` = the removed object's noun (anchors only)
        if any(a.get("kind") == "linked" for a in anchors):
            if s.get("link_method") == "warp_qwen":
                from engine.anchor_warp import touchup_prompt
                lt = touchup_prompt(desc)
            else:
                lt = h3_fill.derived_prompt(desc, pm, what)
            items.append({"key": "anchor_linked", "model": "Qwen-Image-2.1",
                          "title": "Kotwice podłączone (Qwen-Image-2.1, wzór z głównej)", "text": lt})
        for a in anchors:
            if a.get("kind") == "own" and (a.get("prompt") or "").strip():
                items.append({"key": f"anchor_own_{a.get('id')}", "model": "Qwen-Image-2.1",
                              "title": f"Kotwica własna F{a.get('frame')} (Qwen-Image-2.1)",
                              "text": regen.qwen_prompt(a["prompt"], pm)})
    text = (s.get("h3_prompt") or desc) if transition else desc
    if model == "h3":
        p = h3_fill.h3_prompt(text, pm, transition, morph)
        if anchors_on and not transition and anchors:
            p = h3_fill.ref_prompt(p)   # the main anchor rides as <Picture 1> of the whole clip
        items.append({"key": "video", "model": "MiniMax-H3", "title": "Klip wideo (MiniMax-H3)", "text": p})
    elif model == "vace":
        from engine import vace_fill
        items.append({"key": "video", "model": "Wan 2.2 Fun VACE", "title": "Klip wideo (Wan 2.2 Fun VACE)",
                      "text": vace_fill.prompt_for(text, pm, transition, morph)})
    elif model == "ltx":
        from engine import ltx_fill
        items.append({"key": "video", "model": "LTX-2.5", "title": "Klip wideo (LTX-2.5, IC-LoRA inpaint)",
                      "text": ltx_fill.prompt_for(text, pm, transition, morph) or "(pusty – wypełnienie dopasowane do otoczenia)"})
    elif model == "void":
        from engine.cleanup3d import void_engine as VE
        items.append({"key": "video", "model": "VOID", "title": "Ujęcie (VOID)", "text": desc or VE.VOID_PROMPT})
    return {"engine": model, "transition": transition, "morph": morph, "items": items}


# the instruction of the local assistant (engine/prompt_assist.py) – the rules above in words a language model follows.
# Rule 5 depends on the mode (2026-11-12, lesson of AInVFX „Paint a plate”: an erased object comes back when its name
# reaches the prompt – REMOVE describes the scene as it stays, never the erased object).
_ASSIST_HEAD = """You write the description for a video inpainting tool. The user marks an object in a shot (the mask) and
either REMOVES it (the background continues) or REPLACES it (a new look inside the mask). The same description goes to
Qwen-Image-2.1 (edits one still frame = the anchor) and to a video model (MiniMax-H3 or Wan 2.2 Fun VACE) that repaints
the masked area in the whole clip, pinned to the anchor.

Rules for the description:
1. Always English, whatever language the user wrote in. Keep every concrete idea of the user.
2. Describe the LOOK as a finished state, present in every frame. Never words of change over time: no "transforms",
   "turns into", "slowly", "gradually", "becomes", "morphs", "appears", "fades". (A video model makes such words an event
   halfway through the clip.)
3. No motion or camera words (blink, head movement, static camera, pan, zoom) – the shot already gives the motion.
4. No filler (8k, masterpiece, best quality, trending), no citation markers like [cite: 1].
"""
ASSIST_RULE_REPLACE = """5. This job is REPLACE: the NEW look is the USER's idea – build the description around their words, translated to
   English (correct their spelling), and keep every one of them: a style, a material, a colour or a mood they name
   must be in your text. Then make that look concrete – materials, colours, patterns, surface, light response – in the
   spirit of their words. Never invent a different look and never describe how the object looks NOW in the image (the
   image only shows WHERE the object is and what stays around it). Last, say what stays unchanged around it (what you
   SEE outside the mask: hair, jewellery, clothes, background).
"""
ASSIST_RULE_REMOVE = """5. This job is REMOVE: the object outlined in red is erased. Describe ONLY what is BEHIND it – the background that
   continues through the outlined area once the object is gone (surfaces, materials, colours, light), taken from what
   you SEE around the outline. Do not name the removed object and do not describe it: not its kind, colour, shape,
   pose, clothes, shadow or reflection, not with a synonym, not as "the absence of ...", "without ...", "no ...",
   "empty of ..." or "where ... was". Write no "remove", "erase" or "delete" either. (A video model paints every noun
   of the text: an erased object comes back when it is named.) If the user's text or the tool's label names the
   object, that only tells you which thing goes – leave it out and keep only the user's ideas about the background.
   Stay INSIDE the outline (user 2026-11-15, any object – jewellery, a rig, a sign, a person): describe only the
   surfaces that fill it and continue right at its edge (the hair, wall, sky, cloth or ground the outline cuts into),
   matching what surrounds it. Do not describe things fully outside the outline (a face, a whole person, the rest of the
   room) and no relations to them like "frames the face" or "around the person" – the model would repaint them too.
"""
_ASSIST_TAIL = """6. One paragraph, {words} words, concrete nouns and adjectives first, the most important thing at the start.
7. TRANSITION (only when asked): a second short text "clip" describing what happens over time in the clip, in the
   form "<what is there at the start> slowly turns into <the user's new look>"; there change words are allowed.
   (The words in angle brackets are placeholders – never copy any example text of these rules into your answer.)

Answer ONLY with JSON: {"description": "...", "clip": "..." or "", "notes": ["short note in Polish", ...]}
"""
NO_OBJECT_RULE = "Do not name the removed object"     # the sentence of ASSIST_RULE_REMOVE the tests pin


_STOP = {"the", "and", "with", "into", "from", "that", "this", "face", "head", "woman", "man", "person", "make", "look",
         "like", "very", "more", "some", "replace", "replaced", "change", "turn", "want", "please", "should", "have",
         # the Polish words of the same kind (the object itself and verbs of asking): translated, never flagged
         "twarz", "twarzy", "kobieta", "kobiety", "osoba", "osoby", "głowa", "glowa", "skóra", "skora", "zamień", "zamien",
         "zrób", "zrob", "niech", "żeby", "zeby", "taka", "taki", "takie", "jako", "bardzo"}


_PL_HINT = {"twarz", "twarzy", "kobieta", "kobiety", "osoba", "glowa", "skora", "zamien", "zrob", "niech", "zeby", "jako",
            "bardzo", "na", "w", "z", "i", "jest", "ma", "do", "ze", "jej", "jego", "taka", "taki", "takie", "albo", "lub"}


def user_terms(text: Optional[str]) -> List[str]:
    """The content words of what the user typed (Latin letters, 4+ letters, not a common word) – what a REPLACE
    proposal must keep (`missing_terms`, which checks only an English text)."""
    words = re.findall(r"[A-Za-zÀ-ž]{4,}", text or "")
    out: List[str] = []
    for w in words:
        lw = w.lower()
        if lw not in _STOP and lw not in out:
            out.append(lw)
    return out


def missing_terms(user_text: Optional[str], description: Optional[str]) -> List[str]:
    """The user's content words a REPLACE proposal dropped – any object, any look (a stem of 5 letters must appear:
    "psychodelick" ~ "psychedelic" share "psych"; "rusty" in "rusty red car"). Only an English text is checked: a
    Polish one is translated and its stems need not survive, so it is never flagged (no dictionary here)."""
    d = (description or "").lower()
    t = (user_text or "").lower()
    # a Polish text is translated – its stems need not appear in English ("rdzawy" -> "rusty"): no check, no false alarm
    if re.search(r"[ąćęłńóśźż]", t) or set(re.findall(r"[a-z]+", t)) & _PL_HINT:
        return []
    miss = []
    for w in user_terms(user_text):
        stem = w[:5]
        if stem not in d:
            miss.append(w)
    return miss


def assist_rules(mode: str = "remove") -> str:
    """The assistant's instruction for one job: rule 5 = REPLACE (the new content) or REMOVE (only the background)."""
    # REMOVE is short: only what fills the outline (2026-11-15); REPLACE describes a whole new look
    words = "40–110" if mode == "replace" else "15–60"
    return _ASSIST_HEAD + (ASSIST_RULE_REPLACE if mode == "replace" else ASSIST_RULE_REMOVE) + _ASSIST_TAIL.replace("{words}", words)


ASSIST_RULES = assist_rules("replace")                # the old name (the replace wording)


# --- the guard of the assistant's REMOVE answer: the removed object's name never reaches a model prompt ---------------
def name_terms(names) -> List[str]:
    """What counts as "the object's name" for the labels of the cleaned thing: the label itself, the label with
    separators as spaces ("left_wall" -> "left wall") and without a trailing number ("kobieta_2" -> "kobieta").
    Terms with fewer than 3 letters are skipped (a label "a" / "o1" would match ordinary text)."""
    out: List[str] = []
    for n in names or ():
        n = str(n or "").strip().lower()
        if not n:
            continue
        spaced = re.sub(r"[\s_\-.]+", " ", n).strip()
        base = re.sub(r"[\s_\-.]*\d+$", "", spaced).strip()
        for c in (n, spaced, base):
            if len(re.sub(r"[\W\d_]", "", c)) >= 3 and c not in out:
                out.append(c)
    return out


def named(text: Optional[str], names) -> List[str]:
    """The terms of `names` (name_terms) present in `text` as whole words, case-insensitively."""
    return _words(text or "", name_terms(names))


def strip_named(text: Optional[str], names) -> Dict[str, Any]:
    """The assistant's answer without the sentences that carry the removed object's name:
    {text, dropped: [sentence], words: [term]}. Deterministic and model-free; a text with no hit comes back unchanged."""
    t = (text or "").strip()
    if not named(t, names):
        return {"text": t, "dropped": [], "words": []}
    keep, dropped, words = [], [], []
    for sent in re.split(r"(?<=[.!?])\s+", t):
        hit = named(sent, names)
        if hit:
            dropped.append(sent)
            words += [w for w in hit if w not in words]
        elif sent.strip():
            keep.append(sent.strip())
    return {"text": " ".join(keep), "dropped": dropped, "words": words}
