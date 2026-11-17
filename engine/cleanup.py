# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate cleaning – remove objects from the plate (MVP of the „Czyszczenie” mode, user decision 2026-11-06,
docs/FUTURE_MODULES.md §2): the alpha of the chosen objects (any computed version) → hard + grown + softened outward
removal mask, minus a holdout (the other objects – the patch never covers what stays) → fill:
  1. clean plate (the project's clear plate, aligned `cleanplate_align` when measured, used only when its content check
     says „same scene”; else the auto clean plate of a static shot `cleanplate_auto` where its confidence is high),
  2. the inpaint model for the rest, in a fixed crop around the mask: LaMa (big-lama through ComfyUI,
     comfyui-inpaint-nodes, frame by frame; MiniMax-Remover was tried 2026-11-07 and dropped 2026-11-09 – user
     decision after the sh060 test); since 2026-11-10 at the crop's full resolution (`LamaInpainter`:
     engine/lama_full.py, the ComfyUI node 256 x 256 as the fallback); scene-linear EXR plates go through ACEScct for
     the model and back,
  3. OpenCV Telea when ComfyUI / LaMa is missing (said in the run's report).
  0. (Cleanup 3D, engine/cleanup3d/render.py) the source „klucz”: a per-frame fill handed in by the caller (`key_fill`,
     the cards' keys projected through the 3D camera) with a weight 0..1 – taken first; where its weight is 0 the
     sources above fill in, a partial weight blends the two.
Output = its own versions in `<work>/inpaint/v###/` (NEVER in matte_desk/, never the manifest):
  Format = the plate's depth (user 2026-11-10; was always 8-bit PNG for non-linear plates): EXR float for linear
  plates, 16-bit PNG for 10 / 12 / 16-bit plates (DPX, PNG16, TIFF16), 8-bit PNG for 8-bit plates (`output_ext`).
  patch/p_####   RGBA premultiplied: the new pixels × mask, alpha = the mask – `Merge over` the original Read in Nuke
                 (an EXR plate: the plate's pixel type, compression and colour attributes)
  comp/c_####    the full frame with the patch. With the plate's frame files (`plate_file`, user decision 2026-11-12):
                 in the PLATE'S OWN format – EXR with its header, DPX with its header and packing, TIFF / PNG at its
                 depth – and outside the mask the source values themselves (engine/plate_write.py); run.json
                 `comp_ext` / `format` say what was written
  MODELS.md      the models behind the version's pixels with their licences (as Paint a plate's export)
  source/s_####.png  where each pixel came from: 0 plate, 1 clean plate, 2 auto clean plate, 3 LaMa, 4 OpenCV
                     (5 = MiniMax-Remover in runs before 2026-11-09), 6 klucz (Cleanup 3D)
  run.json       settings, sources per object, counts, seconds, `kept_check` (kept objects that look lost, check_kept).
RGB from the original plate everywhere outside the mask (the MatteDesk rule): comp = plate · (1 − m) + fill · m.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA = "matte_desk/cleanup@1.0"
FILLS = ("auto", "cleanplate", "lama")
# the fill model for what no clean plate covers (an old "minimax" setting reads as lama). User 2026-11-10 (step C1): LaMa
# (no prompt, frame by frame), Qwen-Image-2.1 edit (prompt, frame by frame – may flicker) and VOID (prompt, the whole
# shot at once; its frames go in as the "key" fill, LaMa covers what it leaves). FLUX 3 Image Klein is not offered: it
# draws shapes into a hole (engine/cleanup3d/regen.py).
# 2026-11-10 (user): Qwen-Image-2.1 left the list – an image model flickers frame by frame; it makes the H3 anchor
# instead (`anchor`, engine/h3_fill.make_anchor). An older "qwen21" setting reads as lama. QwenInpainter stays for tests
# / a later use.
MODELS = ("lama", "h3", "vace", "ltx", "void")
PROMPT_MODELS = ("void", "h3", "vace", "ltx")
ANCHOR_MODELS = ("h3", "vace", "ltx")   # take the Qwen-Image-2.1 anchors (engine/h3_anchors.py)
ANCHORS = ("none", "frame")   # 2026-11-11: ONE anchor at a chosen frame (old first / first_last read as frame)
VIDEO_MODELS = ("void", "h3", "vace", "ltx")
# Inpaint (C1) offers only the video models that take the anchors (user decision 2026-11-11): LaMa / VOID remove objects
# in the projection steps C2–C6 (card keys); the Cleaner still uses LaMa for what a video model leaves, and older
# versions made with LaMa / VOID stay readable. A setting naming another model reads as the first of these.
C1_MODELS = ("h3", "vace", "ltx")   # ltx = LTX-2.5 IC-LoRA inpaint (2026-11-13)


def active_anchors(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The anchors a run may pin – NONE unless the anchors are on (`anchor` == "frame"). The saved list (`anchors`)
    stays in the settings when the user switches them off, so switching back brings them again, but with „brak” no
    model gets a single anchor (user decision 2026-11-13: a remove run pinned four face anchors of an earlier
    session). Every place that hands anchors to a model reads them through here."""
    if s.get("anchor") != "frame":
        return []
    lst = list(s.get("anchors") or [])
    if s.get("anchor_ref_only"):   # „tylko referencja”: the main anchor alone (it is the reference, nothing is pinned)
        lst = [a for a in lst if a.get("kind") == "main"][:1]
    return lst


REF_ONLY_MODELS = ("h3", "vace")   # engines with a whole-clip reference input („tylko referencja”)


def c1_model(model: Optional[str]) -> str:
    return model if model in C1_MODELS else C1_MODELS[0]   # whole-clip models: their frames are the key fill, LaMa covers what they leave
SRC_PLATE, SRC_CLEAN, SRC_AUTO, SRC_LAMA, SRC_CV = 0, 1, 2, 3, 4
SOURCE_NAMES = {SRC_PLATE: "plate", SRC_CLEAN: "clear plate", SRC_AUTO: "cleanplate_auto", SRC_LAMA: "LaMa", SRC_CV: "OpenCV"}
# Cleanup 3D (engine/cleanup3d/render.py): the keys of the cards projected onto the frame. Not in SOURCE_NAMES – a run
# without `key_fill` reports exactly the sources it always did; code 5 stays MiniMax-Remover of the old runs.
SRC_KEY = 6
KEY_NAME = "klucz"
KEY_FULL = 0.999           # key weight from which a pixel is the key alone (below: blended with the other sources)
SRC_QWEN = 7
MODEL_LABEL = {"lama": "LaMa", "qwen21": "Qwen-Image-2.1", "void": "VOID", "h3": "MiniMax-H3", "vace": "Wan 2.2 Fun VACE", "ltx": "LTX-2.5"}
MODEL_SRC = {"lama": SRC_LAMA, "qwen21": SRC_QWEN}
DEFAULTS: Dict[str, Any] = {
    "remove": [],          # object / group names to remove
    # „Wejście” (user 2026-11-15): the plate the clean-up works on – "" = the original Beauty, "v###" = an earlier
    # Inpaint version's composite (clean on top of a replaced face …); frames it lacks come from Beauty
    "input_plate": "",
    "sources": {},         # name -> version ("" / None = the object's active version; "core" = the mask 1+2)
    # „Zapas generowania” (user decision 2026-11-13): how far around the removed object the video model is asked to
    # draw – the MODEL's mask = the removed alpha grown by this, hard (no band points, no feather), minus the holdout;
    # the engines grow it again and snap it to their latent grid. None = an older project: max(GEN_MARGIN_PX, grow_px +
    # soft_px), so its results keep their meaning (`settings`)
    "gen_margin_px": None,
    # the kept objects cut the MODEL's mask too? (user decision 2026-11-15: no – the removed object's mask already
    # lacks them, the composite brings them back with their soft alpha; cut out, the model never redrew the skin under a
    # see-through hair fringe and the old skin showed through). False for new runs; a version saved before has no key
    # and reads as True (engine/inpaint_recompose: what that model really drew)
    "gen_holdout": False,
    # the composite band („Pas łaty”): what of the generated material goes into the picture – clamped to what the model
    # drew (gen_margin_px + the engine's grow, `drawn_area`)
    "grow_px": 8.0,        # the hard mask grown by (motion blur / soft edge / a halo of the object's light)
    "soft_px": 3.0,        # then softened outward over (the patch's edge)
    # band points of the C1 band (engine/objband.inpaint_band, user 2026-11-11): grow_px = „Pas na zewnątrz”, soft_px =
    # „Wtopienie zewnętrzne”, the points widen / narrow the band locally (set against band_ref_out = grow_px then)
    "band_points": [],
    "band_ref_out": None,
    "band_ref_in": None,
    # keys of the two band sliders (user decision 2026-11-15, like the band of the alpha stages): {"<frame>": {grow_px,
    # soft_px}} - a slider change at a frame = a key there, linear between keys, held before the first / after the last;
    # none = grow_px / soft_px everywhere (`band_at`)
    "band_keys": {},
    "holdout": True,       # the other objects are never covered by the patch
    "holdout_skip_lost": False,   # frames where a kept object looks lost (check_kept): no holdout for it there
    # kept objects NOT protected (user decision 2026-11-15, palette „Złożenie” → Ochrona): their names are left out of
    # the holdout – of the composite AND of the model's mask / the kept-object check (api/cleanup._alpha_loaders), so
    # the patch may cover them. Per cleaned thing (PER_TARGET). Real case _fotage_testy: „Hair” / „Earrings” overlap
    # the cheek, the holdout cut the patch there and plate skin showed through.
    "holdout_off": [],
    "fill": "auto",       # auto = clean plate where usable, the model the rest | cleanplate only | lama = the model only
    "model": "lama",       # the fill model: lama | qwen21 | void (MODELS)
    "prompt": "",          # what should be in the area (Qwen-Image-2.1 / VOID, PROMPT_MODELS; in English)
    "prompt_mode": "remove",   # remove | replace (the description = what replaces the object)
    "anchor": "none",      # MiniMax-H3: none | frame – ONE Qwen-Image-2.1 look frame pinned in the clip
    "anchor_frame": None,  # the anchor's plate frame; None / outside the work range = auto (h3_fill.suggest_anchor)
    "seed": 42,            # MiniMax-H3 + its Qwen-Image-2.1 anchors (🎲 in the panel = a new random one)
    "vace_res": 720,       # Wan 2.2 Fun VACE canvas short edge: 720 (best) | 480 (faster, less memory)
    "vace_speed": "fast",  # fast = Lightning LoRA, 4 steps (when downloaded) | quality = 20 steps, CFG 3.5
    "anchor_morph": False,   # linked anchors: the video model morphs from keyframe to keyframe (h3_anchors.morphs)
    "link_method": "qwen",   # linked anchors: qwen (default) | warp_qwen | warp (engine/anchor_warp.py, 2026-11-11)
    "link_denoise": 0.4,   # Qwen-Image-2.1 touch-up strength of a warped linked anchor
    # „Siła kotwicy” (user 2026-11-15): Qwen-Image-2.1 strength of the main / own anchors – 1 = drawn anew in the mask,
    # < 1 = starts from the plate (the face's shape, light, expression stay); linked anchors keep their own setting
    "anchor_denoise": 1.0,
    # „tylko referencja” (user decision 2026-11-13): the main anchor goes to the video model ONLY as the whole-clip
    # reference image (Wan 2.2 Fun VACE `reference_image`, MiniMax-H3 `MiniMaxH3ReferenceToVideo`) - no frame of the
    # clip is pinned to it, so the look guides every frame without a jump at the anchor's frame. LTX-2.5 has no
    # reference input (refused there with a message).
    "anchor_ref_only": False,
    "anchor_raw": True,    # the video model gets Qwen-Image-2.1's whole picture (g_###_raw.png), not the one pasted
                           # into the patch – the patch is cut after the run anyway. Default on (user test
                           # 2026-11-11: with the pasted anchor MiniMax-H3 left narrow skin strips at the patch edge)
    "h3_strength": 1.0,    # MiniMax-H3 Replace: the sigma the masked area starts at from the plate (1.0 = a hole;
                           # lower = keeps the face's expression / pose, re-draws the look). Remove ignores it
                           # (engine/h3_fill.run_strength, user 2026-11-13)
    "anchors": [],         # MiniMax-H3 anchors (engine/h3_anchors.py: id, frame, kind, prompt, gen)
    "h3_prompt": "",       # MiniMax-H3 clip prompt of a TRANSITION (own / original anchors); "" = the shared description
    "per_target": {},      # the settings below of the OTHER cleaned things (name -> PER_TARGET fields; user 2026-11-11)
    # C1 „wyrównanie poziomu łaty” (user decision 2026-11-12; rule rebuilt 2026-11-13 after the real test): a video
    # model's frames (MiniMax-H3, Wan 2.2 Fun VACE, VOID, SeedVR2 upscale) get ONE gain per channel for the whole shot
    # before they go into the patch (Paint a plate's HDR Match) – plate / model over every pixel the model did not
    # draw (beyond its own grown mask, without the kept objects), blurred (detail is not level), median over the
    # pixels, then over the frames (`shot_gain`). The first rule (plate − model on a ring, diffused into the patch, as
    # Cleanup 3D's keys) measured content: VACE re-draws the hair around the mask, the "correction" averaged ~14
    # levels of 255 and jumped 9 levels frame to frame, while the real error of Wan 2.2 Fun VACE on _fotage_testy is
    # a stable ~1-2 levels (gain 0.990 / 0.992 / 0.995, std over frames 0.001). Cleanup 3D's keys stay levelled there.
    # Real test of this rule (v021 vs v019, same seed, identical model frames): gain 0.989 / 0.991 / 0.995 measured
    # on 10 frames (spread 0.0006), patch level 134.0 vs 135.1 (plate 133.1), frame-to-frame jumps 2.10 vs 2.12 –
    # right and harmless, so ON by default again.
    "tone": True,
    # C1 grain match (user decision 2026-11-13): the video models' fill is cleaner than the plate (VAE / canvas scale);
    # grain of the plate's spectrum is added inside the patch, after the levelling – engine/cleanup3d/grain.py (the
    # module of Cleanup 3D's patch, P17), measured where plate and model show the same background (`Cleaner._grain`)
    "grain": "auto",       # auto = only when the plate is clearly noisier | off | on
    "grain_strength": 1.0,  # multiplies the missing grain (0..2)
    "pad_px": 96,          # context around the mask the inpaint model sees
    "batch": 12,           # frames per ComfyUI prompt
}
AUTO_CONF_MIN = 0.5        # cleanplate_auto confidence (0..1) from which its pixels are trusted
LAMA_MAX_SIDE = 2048       # a larger crop is scaled down for LaMa (the result scaled back – softer)


# Per cleaned thing (user 2026-11-11: picking another object / group in the palette must show ITS model, description,
# anchors): the top-level fields are the current target's; switching stores them under the old name and brings back
# the new one's (switch_target).
PER_TARGET = ("input_plate", "model", "prompt", "prompt_mode", "anchor", "anchor_frame", "anchors", "h3_prompt", "seed", "vace_res", "vace_speed",
              "link_method", "link_denoise", "anchor_morph", "anchor_denoise", "anchor_raw", "anchor_ref_only", "h3_strength",
              # the C1 band is the cleaned thing's own (user 2026-11-11): its sliders and points
              "grow_px", "soft_px", "band_points", "band_ref_out", "band_ref_in", "band_keys",
              "gen_margin_px",   # the generation margin too (2026-11-13)
              "holdout_off")     # which kept objects are not protected (2026-11-15)


def switch_target(s: Dict[str, Any], old: Optional[str], new: Optional[str]) -> Dict[str, Any]:
    """`s` with the PER_TARGET fields of `new` (its own when it was cleaned before, else the defaults – the model
    stays); the current ones kept under `old`."""
    if old == new:
        return s
    per = dict(s.get("per_target") or {})
    if old:
        per[old] = {k: s.get(k) for k in PER_TARGET}
    mine = per.pop(new, None) if new else None
    for k in PER_TARGET:
        if mine is not None and k in mine:
            s[k] = mine[k]
        elif k != "model":
            s[k] = list(DEFAULTS[k]) if isinstance(DEFAULTS[k], list) else DEFAULTS[k]
    s["per_target"] = per
    return s


def settings(d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = dict(DEFAULTS)
    if d:
        s.update({k: v for k, v in d.items() if k in DEFAULTS and v is not None})
    s["remove"] = [str(x) for x in (s["remove"] or [])]
    s["sources"] = {str(k): (str(v) if v else "") for k, v in dict(s["sources"] or {}).items()}
    s["grow_px"] = float(min(GROW_MAX, max(GROW_MIN, float(s["grow_px"]))))
    s["soft_px"] = float(min(100.0, max(0.0, float(s["soft_px"]))))
    try:
        gm = float(s["gen_margin_px"]) if s["gen_margin_px"] is not None else None
    except (TypeError, ValueError):
        gm = None
    # an older project (no key): the margin its model mask had, so a recomposite / rerun means what it meant
    s["gen_margin_px"] = float(min(GEN_MARGIN_MAX, max(0.0, gm if gm is not None else
                                                         max(GEN_MARGIN_PX, s["grow_px"] + s["soft_px"]))))
    s["band_points"] = [dict(p) for p in (s["band_points"] or []) if isinstance(p, dict)]
    for k in ("band_ref_out", "band_ref_in"):
        try:
            s[k] = float(s[k]) if s[k] is not None and s["band_points"] else None
        except (TypeError, ValueError):
            s[k] = None
    s["band_keys"] = clean_band_keys(s["band_keys"])
    s["holdout"] = bool(s["holdout"])
    s["tone"] = bool(s["tone"])
    from engine.cleanup3d import grain as _gr
    s["grain"], s["grain_strength"] = _gr.clean_mode(s["grain"]), _gr.clean_strength(s["grain_strength"])
    # „tylko referencja” is off for good (user 2026-11-15, tests v025–v028 on `_fotage_testy`: with no pinned frame
    # MiniMax-H3 brought the look in only at ~F20–F30). The checkbox left the UI; a True saved before is read as False.
    # The engine paths (active_anchors, h3_fill / vace_fill ref_only) stay for a possible return.
    s["anchor_ref_only"] = False
    s["holdout_skip_lost"] = bool(s["holdout_skip_lost"])
    s["gen_holdout"] = bool(s["gen_holdout"])
    import re as _re
    s["input_plate"] = str(s.get("input_plate") or "") if _re.fullmatch(r"v\d{3}", str(s.get("input_plate") or "")) else ""
    s["anchor_morph"] = bool(s["anchor_morph"])
    try:
        s["anchor_denoise"] = float(min(1.0, max(0.3, float(s["anchor_denoise"]))))
    except (TypeError, ValueError):
        s["anchor_denoise"] = 1.0
    s["holdout_off"] = sorted({str(x) for x in (s["holdout_off"] or []) if str(x)})
    s["fill"] = s["fill"] if s["fill"] in FILLS else "auto"
    s["model"] = s["model"] if s["model"] in MODELS else "lama"
    s["prompt"] = str(s["prompt"] or "")[:1000]
    s["prompt_mode"] = s["prompt_mode"] if s["prompt_mode"] in ("remove", "replace") else "remove"
    s["anchor"] = "frame" if s["anchor"] in ("first", "first_last") else (s["anchor"] if s["anchor"] in ANCHORS else "none")
    try:
        af = int(s["anchor_frame"]) if s["anchor_frame"] is not None else None
    except (TypeError, ValueError):
        af = None
    s["anchor_frame"] = af if af is not None and af >= 0 else None   # -1 from the panel = back to auto
    from engine.h3_anchors import normalize as _norm_anchors
    s["anchors"] = _norm_anchors(s["anchors"])
    s["h3_prompt"] = str(s["h3_prompt"] or "")[:2000]
    s["per_target"] = {str(k): {f: v[f] for f in PER_TARGET if f in v}
                       for k, v in dict(s["per_target"] or {}).items() if isinstance(v, dict)}
    s["vace_res"] = 480 if str(s["vace_res"]) == "480" else 720
    s["vace_speed"] = "quality" if s["vace_speed"] == "quality" else "fast"
    from engine import anchor_warp as _aw
    s["link_method"] = _aw.method(s)
    s["link_denoise"] = _aw.denoise(s)
    s["anchor_raw"] = s.get("anchor_raw") is not False
    from engine.h3_fill import clamp_strength as _cs
    s["h3_strength"] = _cs(s.get("h3_strength"))
    try:
        s["seed"] = int(s["seed"]) % (2 ** 31)
    except (TypeError, ValueError):
        s["seed"] = 42
    s["pad_px"] = int(min(512, max(16, int(s["pad_px"]))))
    s["batch"] = int(min(48, max(1, int(s["batch"]))))
    return s


def cleanup_dir(work_dir: Path) -> Path:
    """`<work>/inpaint/` – next to the tool's matte_desk/ folder, not inside it (FUTURE_MODULES §2)."""
    return Path(work_dir).parent / "inpaint"


def load_settings(work_dir: Path) -> Dict[str, Any]:
    try:
        return settings(json.loads((cleanup_dir(work_dir) / "cleanup.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return settings()


def save_settings(work_dir: Path, s: Dict[str, Any]) -> Dict[str, Any]:
    from engine.jsonio import write_json
    s = settings(s)
    d = cleanup_dir(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "cleanup.json", {"schema": SCHEMA, **s}, indent=1, ensure_ascii=False)
    return s


def versions(work_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for d in sorted(cleanup_dir(work_dir).glob("v[0-9][0-9][0-9]")):
        try:
            rj = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        st = rj.get("settings", {}) or {}
        out.append({"version": d.name, "frames": rj.get("frames"), "remove": st.get("remove"),
                    "counts": rj.get("counts"), "date": rj.get("date"), "ext": rj.get("ext"),
                    "comp_ext": rj.get("comp_ext") or file_ext(rj.get("ext") or "png"),
                    "warnings": list((rj.get("kept_check") or {}).get("warnings") or [])
                    + list((rj.get("held") or {}).get("warnings") or []),   # removed objects held (cleanup_hold)
                    # what the list shows (user 2026-11-11, stage-5 style): model, description, anchors pinned
                    "model": st.get("model"), "prompt": st.get("prompt") or "", "prompt_mode": st.get("prompt_mode"),
                    "anchors": list(rj.get("anchors") or []), "seed": st.get("seed"),
                    "upscaled_from": rj.get("upscaled_from"), "grain": rj.get("grain"),
                    # „⟲ Złóż ponownie” (2026-11-13): the model's frames kept next to the version / where it came from
                    "has_model": (d / "model").is_dir(), "recomposed_from": rj.get("recomposed_from"),
                    "input_plate": st.get("input_plate") or ""})   # made on top of this version („Wejście”, 2026-11-15)
    return out


def input_frame_fn(work_dir: Path, s: Dict[str, Any], beauty_frame: Callable[[int], str]) -> Callable[[int], str]:
    """f -> the frame file the clean-up works on („Wejście”, user 2026-11-15): the chosen version's composite frame
    (v###/comp/c_####.<its ext>) when it has that frame, else the Beauty frame (`beauty_frame`)."""
    v = str(s.get("input_plate") or "")
    if not v:
        return beauty_frame
    d = cleanup_dir(work_dir) / v
    try:
        rj = json.loads((d / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return beauty_frame
    ext = rj.get("comp_ext") or file_ext(rj.get("ext") or "png")

    def at(f: int) -> str:
        p = d / "comp" / f"c_{int(f):04d}.{ext}"
        return str(p) if p.exists() else beauty_frame(int(f))
    return at


def input_missing(work_dir: Path, s: Dict[str, Any], frames: Sequence[int]) -> List[int]:
    """The frames of `frames` the chosen input version does not have (they come from Beauty)."""
    v = str(s.get("input_plate") or "")
    if not v:
        return []
    fn = input_frame_fn(work_dir, s, lambda f: "")
    return [int(f) for f in frames if not fn(int(f))]


LAST_VERSION = ".last_version"   # <work>/inpaint/.last_version: the highest version number ever given


def last_version_number(work_dir: Path) -> int:
    """The highest Cleanup version number ever given: the folders there and `.last_version` (written by every new
    version and before a deletion), so a deleted version's number is never given again (user decision 2026-11-13:
    deleting is permanent – nothing in a trash keeps the number taken; browser caches and notes name versions by it)."""
    d = cleanup_dir(work_dir)
    n = [int(p.name[1:]) for p in d.glob("v[0-9][0-9][0-9]")]
    try:
        n.append(int((d / LAST_VERSION).read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        pass
    return max(n) if n else 0


def remember_version(work_dir: Path) -> None:
    d = cleanup_dir(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / LAST_VERSION).write_text(str(last_version_number(work_dir)), encoding="utf-8")


def next_version(work_dir: Path) -> Path:
    d = cleanup_dir(work_dir)
    out = d / f"v{last_version_number(work_dir) + 1:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / LAST_VERSION).write_text(str(int(out.name[1:])), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- masks
OUT_EXTS = ("exr", "png16", "png")


def output_ext(linear: bool, sample: Optional[np.ndarray]) -> str:
    """Render format by the plate (one rule for both renders): linear -> "exr", an 8-bit plate frame -> "png",
    anything deeper (uint16: 10 / 12 / 16-bit DPX, PNG16, TIFF; or an unreadable sample) -> "png16"."""
    if linear:
        return "exr"
    return "png" if sample is not None and np.asarray(sample).dtype == np.uint8 else "png16"


def file_ext(ext: str) -> str:
    """File suffix of a render format ("png16" is written as .png)."""
    return "png" if ext == "png16" else ext


# „Pas na zewnątrz” = Erode / Dilate (user decision 2026-11-15): below 0 the patch shrinks INSIDE the removed objects
# (Replace keeps the object's own edge, e.g. hair over the face edge; Remove leaves the object's rim – the palette warns)
GROW_MIN, GROW_MAX = -100.0, 200.0


def clean_band_keys(keys: Any) -> Dict[str, Dict[str, float]]:
    """{"<frame>": {grow_px, soft_px}} with both values in range; anything else dropped (engine/bandsys)."""
    from engine import bandsys
    return bandsys.clean_slider_keys(keys, {"grow_px": (GROW_MIN, GROW_MAX), "soft_px": (0.0, 100.0)})


def band_at(s: Dict[str, Any], f: Optional[int]) -> Tuple[float, float]:
    """(grow_px, soft_px) of frame f: the slider keys interpolated linearly (held outside them), no keys / f None =
    the sliders' values. The ONE place the C1 band's per-frame widths come from."""
    from engine import bandsys
    v = bandsys.keyed_at(s.get("band_keys"), f, {"grow_px": float(s["grow_px"]), "soft_px": float(s["soft_px"])})
    return v["grow_px"], v["soft_px"]


def band_reach_max(s: Dict[str, Any]) -> float:
    """The widest grow + soft over the keys (the sliders without keys) - the band limit check."""
    keys = s.get("band_keys") or {}
    vals = [float(k["grow_px"]) + float(k["soft_px"]) for k in keys.values()] or [float(s["grow_px"]) + float(s["soft_px"])]
    return max(vals)


def at_override(s: Dict[str, Any], grow_px: Optional[float], soft_px: Optional[float], f: int) -> Dict[str, Any]:
    """A one-frame endpoint with the palette's unsaved values: those win for that frame (keys dropped); the value not
    given = the keyed one at f."""
    if grow_px is None and soft_px is None:
        return s
    g, sf = band_at(s, f)
    out = dict(s)
    out["grow_px"] = float(min(GROW_MAX, max(GROW_MIN, float(grow_px)))) if grow_px is not None else g
    out["soft_px"] = float(min(100.0, max(0.0, float(soft_px)))) if soft_px is not None else sf
    out["band_keys"] = {}
    return out


def removal_mask(alphas: Sequence[np.ndarray], grow_px: float, soft_px: float,
                 pts: Optional[Sequence[Tuple[float, ...]]] = None) -> np.ndarray:
    """max of the alphas > 0.02 → grown by grow_px (elliptic) → 1 inside, falling to 0 over soft_px outward.
    `pts` = the C1 band points of this frame (x, y, out, in[, ang], plate px, widths already scaled – engine/inpaint_band
    .points_fn): the band (grow) is then a per-pixel width along the objects' outline (engine/bandsys.band_widths, outer
    side) – the patch reaches further / less far there; the soft edge stays `soft_px` outside it. Without points the
    result is what it always was (bit for bit, tests/test_inpaint_band.py). grow_px < 0 = erode (2026-11-15): the
    objects shrunk by |grow_px| (Euclidean), the soft edge `soft_px` outward from that edge; band points then act
    locally – at a point the patch reaches its whisker (raw px from the outline), the slider away from the points
    (engine/bandsys.local_offsets)."""
    if not alphas:
        raise ValueError("brak obiektów do usunięcia")
    hard = (np.max(np.stack([np.asarray(a, np.float32) for a in alphas]), axis=0) > 0.02).astype(np.uint8)
    g = int(round(grow_px))
    inward = bool(pts) and any(len(p) > 4 and abs(float(p[4])) > 90.0 for p in pts)
    if (g < 0 or inward) and pts and hard.any():
        # erode, or a whisker turned into the object (2026-11-15): the points act LOCALLY – the patch edge goes through
        # each whisker's tip (inside = a cut), the slider away from them (engine/bandsys.tip_offsets / local_offsets)
        from engine import bandsys
        off = bandsys.local_offsets(hard.shape, bandsys.tip_offsets(hard, list(pts)), float(grow_px))
        hard = bandsys.offset_mask(hard, off).astype(np.uint8)
    elif g < 0 and hard.any():
        d = cv2.distanceTransform(np.pad(hard, 1, mode="edge"), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        hard = (d > float(-g)).astype(np.uint8)          # the frame edge does not eat the object (padded with itself)
    elif pts and hard.any():
        from engine import bandsys
        o, _ = bandsys.band_widths(hard > 0, float(grow_px), 0.0, list(pts))
        d = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 5)
        hard = ((hard > 0) | (d <= np.asarray(o, np.float32))).astype(np.uint8)
    elif g > 0 and hard.any():
        hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    if soft_px <= 0 or not hard.any():
        return hard.astype(np.float32)
    d = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - d / float(soft_px), 0.0, 1.0).astype(np.float32)


GEN_MARGIN_PX = 24.0       # default „Zapas generowania” (plate px)
GEN_MARGIN_MAX = 300.0     # grow_px 200 + soft_px 100: an older project's derived margin is never cut


def grow_hard(alphas: Sequence[np.ndarray], px: float) -> np.ndarray:
    """float 0 / 1: the alphas (max, > 0.02) grown by `px` (Euclidean distance – constant time in px)."""
    hard = np.max(np.stack([np.asarray(a, np.float32) for a in alphas]), axis=0) > 0.02
    if px > 0 and hard.any():
        d = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        hard |= d <= float(px)
    return hard.astype(np.float32)


def gen_mask(remove: Sequence[np.ndarray], keep: Sequence[np.ndarray], s: Dict[str, Any]) -> np.ndarray:
    """The mask the video MODEL gets (user decision 2026-11-13): the removed objects grown by „Zapas generowania”
    (`gen_margin_px`), hard – no band points, no feather. Since 2026-11-15 it reaches over the kept objects too (the
    model draws under them, the composite protects them); `gen_holdout` True = the old rule (minus the kept objects)
    for versions computed that way."""
    if not remove:
        raise ValueError("brak obiektów do usunięcia")
    m = grow_hard(remove, s["gen_margin_px"])
    h = holdout_mask(keep) if s["holdout"] and s.get("gen_holdout") else None
    return m if h is None else m * (1.0 - h)


def gen_mask_fn(s: Dict[str, Any], remove: Sequence[Callable[[int], np.ndarray]],
                keep: Sequence[Callable[[int], np.ndarray]]) -> Callable[[int], np.ndarray]:
    """f -> the model's mask of frame f – ONE function for the run (MiniMax-H3 / Wan 2.2 Fun VACE / LTX-2.5), the
    anchors and the recomposite (what the model drew)."""
    return lambda f: gen_mask([fn(int(f)) for fn in remove], [fn(int(f)) for fn in keep], s)


def engine_reach_px(model: Optional[str], s: Dict[str, Any], w: int, h: int) -> float:
    """Plate px the engine grows the model's mask by on its canvas (h3_fill / vace_fill / ltx_fill MASK_GROW_PX, the
    latent grid snap not counted – it reaches further only on some sides); 0 for engines without a canvas mask."""
    if model == "h3":
        from engine import h3_fill as E
        cw, _ = E.canvas(w, h)
    elif model == "vace":
        from engine import vace_fill as E
        cw, _ = E.canvas(w, h, int(s.get("vace_res") or 720))
    elif model == "ltx":
        from engine import ltx_fill as E
        cw, _ = E.canvas(w, h)
    else:
        return 0.0
    return float(E.MASK_GROW_PX) * float(w) / float(max(1, cw))


def model_frames(model: Optional[str], f: int, frames: Sequence[int]) -> List[int]:
    """The plate frames whose masks the engine combines into ITS mask of frame `f` (the frame mask preview, user
    2026-11-16 „zależne od modelu”): every canvas engine takes the max over +-MASK_TIME_RADIUS frames (grow_masks);
    MiniMax-H3 then one mask per latent token (h3_fill.token_masks) = the union over the token's frames, padded clip
    frames = the last one. `frames` = the work range (sorted); a frame outside it / another model = [f]."""
    frames = [int(x) for x in frames]
    if model not in ("h3", "vace", "ltx") or int(f) not in frames:
        return [int(f)]
    if model == "h3":
        from engine import h3_fill as E
        from engine.h3_anchors import h3_token, token_spans
        i = frames.index(int(f))
        lo, hi = token_spans(E.clip_length(len(frames)))[h3_token(i)]
    elif model == "vace":
        from engine import vace_fill as E
        lo = hi = frames.index(int(f))
    else:
        from engine import ltx_fill as E
        lo = hi = frames.index(int(f))
    r = int(E.MASK_TIME_RADIUS)
    return frames[max(0, lo - r):min(len(frames), hi + r + 1)]


def drawn_area(gen: np.ndarray, reach_px: float) -> np.ndarray:
    """bool: what the model drew – its mask (> 0) grown by the engine's reach (`engine_reach_px`)."""
    g = np.asarray(gen, np.float32) > 0
    if reach_px <= 0 or not g.any():
        return g
    d = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return g | (d <= float(reach_px))


def comp_mask_fn(s: Dict[str, Any], remove: Sequence[Callable[[int], np.ndarray]],
                 keep: Sequence[Callable[[int], np.ndarray]], reach_px: float) -> Callable[[int], np.ndarray]:
    """f -> the composite mask of a video model's run as the Cleaner makes it: the band (`mask_fn`) cut at what the
    model drew (`drawn_area` of `gen_mask_fn`) – the mask preview and the live preview."""
    cm, gm = mask_fn(s, remove, keep), gen_mask_fn(s, remove, keep)
    return lambda f: cm(f) * drawn_area(gm(f), reach_px)


def band_limit(s: Dict[str, Any], w: int, h: int) -> Dict[str, Any]:
    """The composite band against what the model draws, for the palette: generation margin + the engine's reach =
    `limit_px`; the band's plain reach (grow_px + soft_px, without band points) beyond it = `cut`."""
    eng = engine_reach_px(s.get("model"), s, w, h)
    lim = float(s["gen_margin_px"]) + eng
    reach = band_reach_max(s)
    return {"gen_margin_px": s["gen_margin_px"], "engine_px": round(eng, 1), "limit_px": round(lim, 1),
            "reach_px": reach, "cut": reach > lim + 0.5}


def holdout_mask(keep: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """The objects that stay: their SOFT alpha (max), the patch is cut by it – the patch goes under a kept object as
    much as the object is transparent. Until 2026-11-11 this was alpha > 0.5, which turned the soft hair edge of Final
    Touch (blur 6 px) into a hard cut: the MiniMax-H3 patch ended on a sharp line at the hairline (user report)."""
    if not keep:
        return None
    return np.clip(np.max(np.stack([np.asarray(a, np.float32) for a in keep]), axis=0), 0.0, 1.0)


def final_mask(remove: Sequence[np.ndarray], keep: Sequence[np.ndarray], s: Dict[str, Any],
               pts: Optional[Sequence[Tuple[float, ...]]] = None, f: Optional[int] = None) -> np.ndarray:
    g, sf = band_at(s, f)                  # the slider keys of frame f (2026-11-15)
    m = removal_mask(remove, g, sf, pts)
    h = holdout_mask(keep) if s["holdout"] else None
    return m if h is None else m * (1.0 - h)


def mask_fn(s: Dict[str, Any], remove: Sequence[Callable[[int], np.ndarray]],
            keep: Sequence[Callable[[int], np.ndarray]]) -> Callable[[int], np.ndarray]:
    """f -> the C1 mask of frame f WITH its band points – ONE function for the preview (GET /api/cleanup/mask), the
    anchors, MiniMax-H3 and the run (Cleaner.mask), so the run is what the preview shows."""
    from engine import inpaint_band as IB
    pf = IB.points_fn(s, lambda k: [fn(int(k)) for fn in remove])

    def at(f: int) -> np.ndarray:
        f = int(f)
        return final_mask([fn(f) for fn in remove], [fn(f) for fn in keep], s, pf(f) if pf else None, f)
    return at


# ----------------------------------------------------------------------------- kept objects that lost their track
# sh060 2026-11-10: the kept „Plaza_A” (a small person behind) lost its track while the removed woman passed in front
# and SAM 3.1 put its mask on the woman's OUTLINE in F10–F14; the holdout cut that outline out of the patch and the
# woman's silhouette stayed. Both renders (Cleaner here, engine/cleanup3d/render.render) run `kept_guard` first.
LOST_AREA_JUMP = 10.0      # a kept alpha over this x its median area (over the range) = lost
LOST_EDGE_SHARE = 0.6      # ... or more than this share of its pixels in the band around the removed objects' outline
LOST_BAND_REL = 0.004      # half-width of that band, x the plate's long side (4K: 16 px), at least LOST_BAND_MIN
LOST_BAND_MIN = 3
LOST_MIN_PX = 16           # fewer pixels say nothing (edge test)
LOST_MIN_FRAMES = 3        # frames with the object needed for its median (area test)


class Named:
    """An alpha loader with the object's name (api/cleanup.py: the holdout loaders, for the kept-object warnings)."""

    def __init__(self, fn: Callable[[int], np.ndarray], name: str):
        self.fn, self.name = fn, str(name)

    def __call__(self, f: int) -> np.ndarray:
        return self.fn(f)


def names_of(loaders: Sequence[Any]) -> List[str]:
    return [str(getattr(fn, "name", "") or f"obiekt {i + 1}") for i, fn in enumerate(loaders)]


def frame_ranges(frames: Sequence[int], prefix: str = "F") -> str:
    """[10, 11, 12, 20] -> "F10–F12, F20" (prefix "" -> "10–12, 20")."""
    fs = sorted({int(f) for f in frames})
    out: List[str] = []
    i = 0
    while i < len(fs):
        j = i
        while j + 1 < len(fs) and fs[j + 1] == fs[j] + 1:
            j += 1
        out.append(f"{prefix}{fs[i]}" if i == j else f"{prefix}{fs[i]}–{prefix}{fs[j]}")
        i = j + 1
    return ", ".join(out)


def outline_band(remove: Sequence[np.ndarray], band_px: int) -> Optional[np.ndarray]:
    """Pixels within `band_px` of the outline of the removed objects (alpha > 0.5), both sides; None = nothing removed."""
    if not remove:
        return None
    hard = (np.max(np.stack([np.asarray(a, np.float32) for a in remove]), axis=0) > 0.5).astype(np.uint8)
    if not hard.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    return (cv2.dilate(hard, k) > 0) & ~(cv2.erode(hard, k) > 0)


def check_kept(frames: Sequence[int], remove_alphas: Callable[[int], List[np.ndarray]],
               keep_alphas: Callable[[int], List[np.ndarray]], names: Optional[Sequence[str]] = None,
               band_px: Optional[int] = None, progress: Optional[Callable[[str, int, int], None]] = None
               ) -> Dict[str, Any]:
    """Kept objects (the holdout) that look lost in some frames: their alpha's area jumps over LOST_AREA_JUMP x its
    median over `frames`, or more than LOST_EDGE_SHARE of its pixels lie in a thin band around the removed objects'
    outline (SAM 3.1 sticking to the passing removed object). keep_alphas(f) -> the kept alphas in a fixed order
    (`names` of the same order). Changes nothing; returns {"lost": {name: [frames]}, "reasons": {name: "edge" | "area"},
    "idx": {i: [frames]}, "warnings": [Polish text]}."""
    frames = [int(f) for f in frames]
    stats: Dict[int, Dict[int, Tuple[int, float]]] = {}
    for n, f in enumerate(frames):
        if progress:
            progress(f"Czyszczenie: sprawdzam obiekty chronione F{f}", n, len(frames))
        keep = keep_alphas(f) or []
        if not keep:
            continue
        rm = remove_alphas(f) or []
        if band_px is None:
            hh, ww = np.asarray(keep[0]).shape[:2]
            band_px = max(LOST_BAND_MIN, int(round(LOST_BAND_REL * max(hh, ww))))
        band = outline_band(rm, band_px)
        for i, a in enumerate(keep):
            k = np.asarray(a, np.float32) > 0.5
            px = int(k.sum())
            share = float((k & band).sum()) / px if px and band is not None else 0.0
            stats.setdefault(i, {})[f] = (px, share)
    names = list(names) if names else []
    lost: Dict[str, List[int]] = {}
    reasons: Dict[str, str] = {}
    idx: Dict[int, List[int]] = {}
    for i, per in sorted(stats.items()):
        nz = [px for px, _ in per.values() if px > 0]
        med = float(np.median(nz)) if len(nz) >= LOST_MIN_FRAMES else 0.0
        edge, area = [], []
        for f, (px, share) in sorted(per.items()):
            if px >= LOST_MIN_PX and share > LOST_EDGE_SHARE:
                edge.append(f)
            elif med > 0 and px > LOST_AREA_JUMP * med:
                area.append(f)
        if edge or area:
            name = names[i] if i < len(names) and names[i] else f"obiekt {i + 1}"
            idx[i] = sorted(edge + area)
            lost[name] = idx[i]
            reasons[name] = "edge" if edge else "area"
    return {"lost": lost, "reasons": reasons, "idx": idx, "warnings": []}


def kept_warning(name: str, frames: Sequence[int], reason: str, skipped: bool) -> str:
    why = "maska na krawędzi usuwanego obiektu" if reason == "edge" else "maska nagle ponad 10× większa niż zwykle"
    if skipped:
        return (f"Obiekt chroniony „{name}” w {frame_ranges(frames)} wygląda na zgubiony ({why}) – w tych klatkach łata "
                f"go nie omija (ustawienie „pomiń ochronę zgubionych”)")
    return f"Obiekt chroniony „{name}” w {frame_ranges(frames)} wygląda na zgubiony ({why}) – popraw go albo wyłącz ochronę"


def kept_guard(frames: Sequence[int], remove_alphas: Callable[[int], List[np.ndarray]],
               keep_alphas: Callable[[int], List[np.ndarray]], s: Dict[str, Any],
               names: Optional[Sequence[str]] = None, progress: Optional[Callable[[str, int, int], None]] = None
               ) -> Tuple[Callable[[int], List[np.ndarray]], Dict[str, Any]]:
    """The one kept-object check of both renders: (keep_alphas to use, report). Warns only; the holdout changes only
    with the setting `holdout_skip_lost` (default off): then a lost object's alpha is left out in its lost frames."""
    rep: Dict[str, Any] = {"lost": {}, "reasons": {}, "idx": {}, "warnings": [], "skipped": False}
    if not s.get("holdout", True):
        return keep_alphas, rep
    rep.update(check_kept(frames, remove_alphas, keep_alphas, names, progress=progress))
    skip = bool(s.get("holdout_skip_lost")) and bool(rep["idx"])
    rep["skipped"] = skip
    rep["warnings"] = [kept_warning(n, fr, rep["reasons"][n], skip) for n, fr in rep["lost"].items()]
    if not skip:
        return keep_alphas, rep
    drop = {i: set(fr) for i, fr in rep["idx"].items()}

    def keep(f: int) -> List[np.ndarray]:
        return [a for i, a in enumerate(keep_alphas(f) or []) if int(f) not in drop.get(i, ())]
    return keep, rep


def kept_public(rep: Dict[str, Any]) -> Dict[str, Any]:
    """The report as run.json keeps it (no indices)."""
    return {"lost": rep.get("lost") or {}, "reasons": rep.get("reasons") or {}, "skipped": bool(rep.get("skipped")),
            "warnings": list(rep.get("warnings") or [])}


def mask_bbox(masks: Sequence[np.ndarray], pad: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) around every mask > 0 of the batch, padded and clipped; None = empty."""
    ys, xs = [], []
    for m in masks:
        nz = np.argwhere(m > 0)
        if len(nz):
            ys += [nz[:, 0].min(), nz[:, 0].max()]
            xs += [nz[:, 1].min(), nz[:, 1].max()]
    if not ys:
        return None
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x1, y1 = min(w, max(xs) + pad + 1), min(h, max(ys) + pad + 1)
    return int(x0), int(y0), int(x1), int(y1)


# ----------------------------------------------------------------------------- encodings for the model
def to_model(img: np.ndarray, linear: bool) -> np.ndarray:
    """Plate BGR → 0..1 for an 8-bit model: 8-bit sources as they are, scene-linear through ACEScct."""
    if not linear:
        return np.clip(img, 0, 1).astype(np.float32)
    from engine.log_curves import get
    c = get("acescct")
    return np.clip(c.lin_to_log(np.maximum(img, 0).astype(np.float32)), 0, 1).astype(np.float32)


def from_model(img: np.ndarray, linear: bool) -> np.ndarray:
    if not linear:
        return img.astype(np.float32)
    from engine.log_curves import get
    return get("acescct").log_to_lin(img.astype(np.float32)).astype(np.float32)


# ----------------------------------------------------------------------------- fill
def lama_batch_comfy(client: Any, io_dir: Path, imgs: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                     model_name: str = "big-lama.pt") -> List[np.ndarray]:
    """LaMa on a batch of same-size crops (8-bit in / out) through ComfyUI – one prompt (cleanplate_auto.lama_graph)."""
    from engine.cleanplate_auto import lama_graph
    from engine.comfy_io import write_mask_png, write_png8
    from engine.gpu_manager import comfy_lease
    job = Path(io_dir) / f"cleanup_lama_{uuid.uuid4().hex[:8]}"
    try:
        for i, (im, m) in enumerate(zip(imgs, masks)):
            write_png8(job / "frames" / f"f_{i:04d}.png", im)
            write_mask_png(job / "mask" / f"m_{i:04d}.png", m)
        g = lama_graph(str(job / "frames"), str(job / "mask"), f"matte_desk/cleanup_{job.name}/c", model_name)
        with comfy_lease(client, "LaMa (czyszczenie)"):
            sub = client.queue_prompt(g)
            if not sub.get("success"):
                raise RuntimeError(f"Comfy odrzucił graf LaMa: {sub.get('error')}")
            res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=3600)
        if not res.get("success"):
            raise RuntimeError(f"LaMa nie powiódł się: {res.get('error')}")
        outs = sorted(res.get("outputs", {}).get("6", {}).get("images", []), key=lambda i: i["filename"])
        if len(outs) != len(imgs):
            raise RuntimeError(f"LaMa zwrócił {len(outs)} obrazów zamiast {len(imgs)}")
        out = []
        for info, im in zip(outs, imgs):
            o8 = cv2.imdecode(np.frombuffer(client.download_output(info), np.uint8), cv2.IMREAD_COLOR)
            if o8 is None or o8.shape[:2] != im.shape[:2]:
                raise RuntimeError("LaMa: zły obraz wyjściowy")
            out.append(o8.astype(np.float32) / 255.0)
        return out
    finally:
        shutil.rmtree(job, ignore_errors=True)


class LamaInpainter:
    """The Cleaner's LaMa (crops in the model's 0..1 encoding in, filled crops out): every crop at its FULL resolution
    through engine/lama_full.py (a worker in ComfyUI's Python; holes over lama_full.HOLE_MAX px run scaled to it) when
    that is available; when the worker fails (LamaFullError), the ComfyUI node (256 x 256, `lama_batch_comfy`) for that
    and every later crop – the pattern of engine/cleanup3d/regen._run_lama. `client` None = no node (the worker only).
    `notes`: what the Cleaner adds to run.json; `used`: crops per path."""

    def __init__(self, client: Any, io_dir: Path, cfg: Optional[Dict[str, Any]] = None, full: Optional[bool] = None,
                 model_name: str = "big-lama.pt"):
        from engine import lama_full
        self.client, self.io_dir, self.cfg, self.model_name = client, Path(io_dir), cfg, model_name
        self.full = lama_full.available(cfg) if full is None else bool(full)
        self.notes: List[str] = []
        self.used = {"full": 0, "node": 0}

    @property
    def where(self) -> str:
        return "w pełnej rozdzielczości" if self.full else "w ComfyUI"

    def __call__(self, imgs: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
        from engine import lama_full
        out: List[Optional[np.ndarray]] = [None] * len(imgs)
        if self.full:
            for i, (im, m) in enumerate(zip(imgs, masks)):
                try:
                    out[i] = np.asarray(lama_full.inpaint(im, m, self.cfg), np.float32)
                except lama_full.LamaFullError as e:
                    self.full = False                  # the node for this and the remaining crops
                    self.notes.append(f"LaMa w pełnej rozdzielczości niedostępna ({str(e)[:160]}) – węzeł ComfyUI (256 × 256)")
                    break
        rest = [i for i, o in enumerate(out) if o is None]
        if rest:
            if self.client is None:
                raise RuntimeError("węzeł LaMa w ComfyUI niedostępny")
            got = lama_batch_comfy(self.client, self.io_dir, [imgs[i] for i in rest], [masks[i] for i in rest],
                                   self.model_name)
            for i, o in zip(rest, got):
                out[i] = o
        self.used["node"] += len(rest)
        self.used["full"] += len(imgs) - len(rest)
        return [o for o in out]   # type: ignore[misc]


class QwenInpainter:
    """Qwen-Image-2.1 edit as the Cleaner's model (step C1, user 2026-11-10): every crop (the object still visible – an
    edit model removes what the instruction names) goes through engine/cleanup3d/regen._run_qwen with the prompt; the
    Cleaner pastes the result only in the hole. Frame by frame, so the fill may flicker between frames."""

    where = "klatka po klatce w ComfyUI"

    def __init__(self, client: Any, io_dir: Path, prompt: str = "", mode: str = "remove"):
        self.client, self.io_dir, self.prompt, self.mode = client, Path(io_dir), prompt or "", mode
        self.notes: List[str] = []
        self.used = {"qwen21": 0}

    def __call__(self, imgs: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
        from engine.cleanup3d.regen import _run_qwen    # every ComfyUI prompt under its own comfy_lease there
        out = []
        for im, m in zip(imgs, masks):
            gen, _w, _n = _run_qwen(self.client, self.io_dir, np.asarray(im, np.float32), np.asarray(m) > 0,
                                    self.prompt, None, self.mode)
            out.append(np.asarray(gen, np.float32))
            self.used["qwen21"] += 1
        return out


class VoidKeyFill:
    """VOID frames as the Cleaner's key fill (step C1): f -> (the VOID frame scaled to the plate, in the plate's
    encoding; weight 1) – the Cleaner pastes it only inside its mask. None when the run has no such frame (LaMa then)."""
    parallel = True            # read-only frames: safe for the Cleaner's frame threads

    label = "VOID"
    tone = True                # levelled to the plate by the Cleaner (Cleaner._tone)
    tone_gap_px = 0.0

    def __init__(self, work: Path, run: Dict[str, Any], w: int, h: int, linear: bool):
        self.work, self.run, self.w, self.h, self.linear = Path(work), run, int(w), int(h), bool(linear)

    def __call__(self, f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        from engine.cleanup3d import void_engine as VE
        try:
            img = VE.from_display(VE.void_frame(self.work, self.run, int(f), (self.w, self.h)), self.linear)
        except ValueError:
            return None
        return np.asarray(img, np.float32), np.ones((self.h, self.w), np.float32)


# the models a version's pixels can come from -> doctor.LICENSES key (MODELS.md, user decision 2026-11-12)
MODEL_LICENSE = {"LaMa": "inpaint_lama", "VOID": "void", "MiniMax-H3": "h3", "Wan 2.2 Fun VACE": "vace", "LTX-2.5": "ltx",
                 "Qwen-Image-2.1": "qwen21", "SeedVR2": "seedvr2", "FLUX 3 Image Klein": "flux3_klein"}


def models_md(rj: Dict[str, Any]) -> str:
    """MODELS.md of a version: the models whose output is in its pixels (sources that filled > 0 px, the anchors'
    Qwen-Image-2.1, SeedVR2 of an upscale), each with the licence Doctor states; the masks' own models (SAM 3.1 and the
    matte stages) are in the objects' versions."""
    used = [k for k, v in (rj.get("counts") or {}).items() if v]
    if rj.get("anchors"):
        used.append("Qwen-Image-2.1")
    if rj.get("upscaled_from"):
        used.append("SeedVR2")
    rec = rj.get("recomposed_from")
    try:
        from doctor import LICENSES
    except Exception:            # doctor.py not importable (a packaged copy without it): names only
        LICENSES = {}
    lines = ["# Modele w tej wersji", "",
             f"Wersja z {rj.get('date') or '?'}, klatki {rj.get('frames')}, silnik `{rj.get('backend')}`.", ""]
    seen = set()
    for name in used:
        if name in seen:
            continue
        seen.add(name)
        key = MODEL_LICENSE.get(name)
        lic = LICENSES.get(key) if key else None
        if lic:
            lines.append(f"- **{name}** – {lic[0]}" + (" (NIEKOMERCYJNIE)" if lic[1] else ""))
        elif name in ("plate", "clear plate", "cleanplate_auto", "OpenCV") or name in (rj.get("fill_no_model") or []):
            lines.append(f"- {name} – bez modelu generatywnego")
        else:
            lines.append(f"- **{name}** – licencja: sprawdź w Doktorze")
    if not seen:
        lines.append("- brak wypełnienia (maska pusta)")
    if rec:   # „⟲ Złóż ponownie”: the model's pixels are the source version's, nothing was generated again
        src = rj.get("source_model") or {}
        lines += ["", f"Złożona ponownie z wersji {rec} bez ponownego liczenia modelu – obraz modelu "
                      f"**{src.get('label') or '?'}** z {rec} (model/), nowy pas łaty."]
    lines += ["", "Maski obiektów: SAM 3.1 i etapy mattów – licencje w Doktorze („Licencje”).",
              "Poza łatką piksele są wartościami oryginalnego plate'u (" + str(rj.get("rgb_source") or "plate") + ").", ""]
    return "\n".join(lines)


GAIN_FRAMES = 12          # frames the shot's gain is measured on (evenly spread over the run)
GAIN_BLUR_PX = 8.0        # both pictures blurred first: the model re-draws detail (hair), only the level is measured
GAIN_MIN_PX = 2000        # usable pixels a frame needs to count
GAIN_LIMIT = (0.85, 1.15)  # a gain outside this = the two pictures differ in content, not level: not applied
GAIN_DEADBAND = 0.002     # |gain - 1| below this on every channel: nothing to fix (under half a level of 255)
GAIN_DARK = 0.02          # pixels darker than this (in the model space) carry no reliable ratio
GRAIN_FRAMES = 6          # frames of the run the shot's grain is measured on (grain.combine), evenly spread
GRAIN_PAD_PX = 96         # plate px of ring around the model's area the grain is measured on (grain.measure's crop)


def shot_gain(samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], linear: bool = False) -> Optional[Dict[str, Any]]:
    """ONE gain per channel for a whole shot (C1 „wyrównanie poziomu”, 2026-11-13, Paint a plate's HDR Match): from
    [(plate, model frame, usable mask)] of a few frames – both blurred by GAIN_BLUR_PX, the median of plate / model over
    the usable pixels (not dark) per frame and channel, then the median over the frames. Values stay in the plate's
    encoding (display 0..1 for 8 / 16-bit plates, scene-linear for EXR: a gain is exact there). Returns {"bgr": [b, g,
    r], "frames": n, "spread": largest std over the frames, "applied": bool, "why": Polish reason when not}; None when
    no frame had enough usable pixels."""
    per = []
    for plate, model, use in samples:
        use = np.asarray(use, bool)
        if int(use.sum()) < GAIN_MIN_PX:
            continue
        lp = cv2.GaussianBlur(np.asarray(plate, np.float32), (0, 0), GAIN_BLUR_PX)
        lm = cv2.GaussianBlur(np.asarray(model, np.float32), (0, 0), GAIN_BLUR_PX)
        dark = to_model(lm, linear).max(axis=-1) if linear else lm.max(axis=-1)
        ok = use & (dark > GAIN_DARK)
        if int(ok.sum()) < GAIN_MIN_PX:
            continue
        per.append(np.median(lp[ok] / np.maximum(lm[ok], 1e-6), axis=0))
    if not per:
        return None
    arr = np.array(per, np.float64)
    g = np.median(arr, axis=0)
    out = {"bgr": [round(float(x), 5) for x in g], "frames": len(per),
           "spread": round(float(arr.std(axis=0).max()), 5), "applied": True, "why": ""}
    if not all(GAIN_LIMIT[0] <= x <= GAIN_LIMIT[1] for x in g):
        out.update(applied=False, why=f"wzmocnienie {', '.join(f'{x:.3f}' for x in g)} poza {GAIN_LIMIT} – obraz "
                                      "modelu różni się od plate'u treścią, nie poziomem")
    elif all(abs(x - 1.0) < GAIN_DEADBAND for x in g):
        out.update(applied=False, why="")          # nothing to fix: no note
    return out


def opencv_batch(imgs: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    from engine.cleanplate_auto import opencv_inpaint
    return [opencv_inpaint(im, m > 0.5) for im, m in zip(imgs, masks)]


# CPU workers of a run (2026-11-14, user: „równoległe liczenie klatek na CPU”): the frames of a batch are prepared
# (plate, masks, sources, levelling, grain) by this many threads and written by WRITE_WORKERS – OpenCV and numpy release
# the GIL in the heavy parts (imread / imwrite, blur, dilate, distance transforms). Every frame is computed by the same
# code with the same inputs, so the files are bit-identical to the one-thread run (tests/test_cleanup_parallel.py).
FRAME_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
WRITE_WORKERS = 2


class _Writer:
    """The frames' composite + PNG / EXR writing in background threads (WRITE_WORKERS), overlapped with the next
    frames' fill (cv2.imwrite releases the GIL). At most `depth` frames wait (memory: a 4K frame's arrays ~250 MB); the
    arrays handed over are not touched by the caller afterwards, so the files are bit-identical to writing in line.
    An error of the writer is raised on the next `put` / on `close`."""

    def __init__(self, fn: Callable[..., None], depth: int = 2 * WRITE_WORKERS, workers: int = WRITE_WORKERS):
        from concurrent.futures import ThreadPoolExecutor
        self.fn, self.depth, self.empty = fn, max(1, int(depth)), 0
        self._ex = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="cleanup_write")
        self._pending: List[Any] = []

    def put(self, *args: Any) -> None:
        while len(self._pending) >= self.depth:
            self._pending.pop(0).result()
        self._pending.append(self._ex.submit(self.fn, *args))

    def close(self) -> None:
        try:
            while self._pending:
                self._pending.pop(0).result()
        finally:
            self.abort()

    def abort(self) -> None:
        """Drop what has not started (an error / „Przerwij” of the fill), wait for the running write."""
        for p in self._pending:
            p.cancel()
        self._pending = []
        self._ex.shutdown(wait=True)


class Cleaner:
    """One cleaning run. Loaders are passed in (tests: synthetic; app: the project's plates and objects):
    plate(f) -> BGR float (display 0..1 for 8-bit, scene-linear for EXR); remove_alphas(f) / keep_alphas(f) -> lists of
    full-frame alphas; clean(f) -> (BGR, ok mask) or None; auto_clean(f) -> (BGR, confidence 0..1) or None;
    inpaint(imgs, masks) -> filled imgs (None = OpenCV); key_fill(f) -> (BGR, weight 0..1) or None – the source
    „klucz” (full frame, same encoding as the plate), taken before every other source. `report`: extra fields of
    run.json (what the caller's fill was made from). `keep_names`: the kept objects' names in keep_alphas' order (the
    kept-object warnings, `kept_guard`); `kept_check`: a check the caller already ran (engine/cleanup3d/render.render:
    its keep_alphas are then already the guarded ones) – else `run` runs it."""

    def __init__(self, s: Dict[str, Any], w: int, h: int, linear: bool,
                 plate: Callable[[int], np.ndarray], remove_alphas: Callable[[int], List[np.ndarray]],
                 keep_alphas: Callable[[int], List[np.ndarray]],
                 clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
                 auto_clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
                 inpaint: Optional[Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], List[np.ndarray]]] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 key_fill: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
                 keep_names: Optional[Sequence[str]] = None, kept_check: Optional[Dict[str, Any]] = None,
                 plate_file: Optional[Callable[[int], str]] = None,
                 gen_mask: Optional[Callable[[int], np.ndarray]] = None):
        self.s, self.w, self.h, self.linear = settings(s), w, h, linear
        # f -> the mask the video model got (`gen_mask_fn`, user decision 2026-11-13): the composite band is cut at what
        # the model drew (that mask + the fill's `drawn_px`), the levelling / grain measured beyond it. None = the
        # composite mask was the model's mask (Cleanup 3D, older callers): nothing cut, measured beyond `mask`
        self.gen_mask = gen_mask
        self._gen_cache: Dict[int, np.ndarray] = {}
        self._cut: List[int] = []                       # frames where the band reached beyond the model's drawing
        self.plate_file = plate_file          # f -> the plate's frame file: the full frame in the plate's own format
        # threads preparing the frames of a batch (1 = in line): only with a key fill that can be read from several
        # threads at once (`parallel = True`: frames read from disk / memory); Cleanup 3D's KeyFill drives one
        # geometry worker per region and stays in line
        self.workers = FRAME_WORKERS if key_fill is None or getattr(key_fill, "parallel", False) else 1
        self.fmt = None
        self.keep_names = list(keep_names) if keep_names else None
        self.kept = kept_check
        self.plate, self.remove_alphas, self.keep_alphas = plate, remove_alphas, keep_alphas
        self.clean, self.auto_clean, self.inpaint = clean, auto_clean, inpaint
        self.key_fill = key_fill
        self.report: Dict[str, Any] = {}
        self._key_part: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._gain: Optional[Dict[str, Any]] = None     # the shot's gain per channel (setting `tone`, `shot_gain`)
        self._lock = threading.Lock()                    # the per-frame caches below, filled by FRAME_WORKERS threads
        self._grain_shot: Optional[Dict[str, Any]] = None   # {"meas": combined measurement or None, "added": bool}
        self._grain_stats: Dict[int, Dict[str, Any]] = {}   # frame -> {"grain": grain.apply's report} (setting `grain`)
        self.progress = progress or (lambda *a: None)
        self.model = self.s["model"]
        # VOID comes in as the key fill (whole frames); what it leaves is inpainted with LaMa
        self.inpaint_model = "lama" if self.model in VIDEO_MODELS else self.model
        self.backend = self.model if inpaint is not None or (self.model in VIDEO_MODELS and key_fill is not None) else "opencv"
        self.notes: List[str] = []

    def mask(self, f: int) -> np.ndarray:
        if not hasattr(self, "_band_pts"):     # the C1 band points (engine/inpaint_band), carried by the removed objects
            from engine import inpaint_band as IB
            self._band_pts = IB.points_fn(self.s, self.remove_alphas)
        m = final_mask(self.remove_alphas(f), self.keep_alphas(f), self.s,
                       self._band_pts(int(f)) if self._band_pts else None, int(f))
        lim = self._drawn(f)
        if lim is not None:                   # the band keeps only what the model drew (cut there, said in run.json)
            if (m[~lim] > 0).any() and int(f) not in self._cut:
                self._cut.append(int(f))
            m = m * lim
        pm = getattr(self.key_fill, "patch_mask", None) if self.key_fill is not None else None
        if pm is not None:                    # Cleanup 3D: the keys' own patch (what they changed + a soft edge)
            km = pm(f)
            if km is not None:
                m = np.maximum(m, np.asarray(km, np.float32))
        return m

    def _gen(self, f: int) -> Optional[np.ndarray]:
        """The model's mask of frame f (`gen_mask`, the last few kept: the mask, the levelling and the grain ask)."""
        if self.gen_mask is None:
            return None
        f = int(f)
        with self._lock:
            got = self._gen_cache.get(f)
        if got is not None:
            return got
        got = np.asarray(self.gen_mask(f), np.float32)
        with self._lock:
            while len(self._gen_cache) >= 2 * FRAME_WORKERS + 2:
                self._gen_cache.pop(next(iter(self._gen_cache)))
            self._gen_cache[f] = got
        return got

    def _drawn(self, f: int) -> Optional[np.ndarray]:
        """bool: what the model drew (its mask + the fill's `drawn_px`) – the composite's limit; None = no limit."""
        reach = getattr(self.key_fill, "drawn_px", None) if self.key_fill is not None else None
        g = self._gen(f) if reach is not None else None
        return None if g is None else drawn_area(g, float(reach))

    def _model_area(self, f: int) -> np.ndarray:
        """bool: where the video model drew its own picture – its mask grown by `tone_gap_px` (the engine's grow + the
        latent grid, plate px). Outside it plate and model frame show the same background: the levelling and the grain
        are measured there. Its mask = `gen_mask` when given, else the composite mask (what the model got before
        2026-11-13)."""
        gap = int(np.ceil(float(getattr(self.key_fill, "tone_gap_px", 0.0) or 0.0)))
        g = self._gen(f)
        base = (g if g is not None else self.mask(f)) > 0
        return cv2.dilate(base.astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0

    def _free(self, f: int, wgt: np.ndarray, drawn: np.ndarray) -> np.ndarray:
        """bool: the same background in plate and model frame – outside the model's area, no kept object, where the
        model's frame exists (weight 1)."""
        free = ~drawn
        for a in self.keep_alphas(f) or []:
            free &= np.asarray(a, np.float32) <= 0.02
        free &= np.asarray(wgt, np.float32) >= KEY_FULL
        return free

    def _measure_gain(self, frames: Sequence[int]) -> Optional[Dict[str, Any]]:
        """The shot's gain per channel for a video model's frames (`shot_gain` over up to GAIN_FRAMES evenly spread
        frames): the plate / model ratio where the model did not draw. None = nothing to measure."""
        pick = list(frames)[::max(1, int(np.ceil(len(frames) / GAIN_FRAMES)))] if frames else []
        samples = []
        for f in pick:
            got = self.key_fill(f)
            if got is None:
                continue
            free = self._free(f, got[1], self._model_area(f))
            samples.append((self.plate(f).astype(np.float32), np.asarray(got[0], np.float32), free))
        return shot_gain(samples, self.linear)

    def _grain_on(self) -> bool:
        """Grain match of a video model's fill (setting `grain` not off): the key fills marked `grain` (default: their
        `tone` mark – MiniMax-H3, Wan 2.2 Fun VACE, LTX-2.5, VOID; SeedVR2's upscale sets it itself). Cleanup 3D's
        keys are grained inside their own render (engine/cleanup3d/render.KeyFill), never here."""
        from engine.cleanup3d import grain as GR
        kf = self.key_fill
        if kf is None or GR.clean_mode(self.s.get("grain")) == "off":
            return False
        return bool(getattr(kf, "grain", getattr(kf, "tone", False)))

    def measure_grain(self, frames: Sequence[int]) -> Dict[str, Any]:
        """The shot's grain measurement (2026-11-14, speed: one measurement for the run instead of one per frame):
        grain.measure on up to GRAIN_FRAMES evenly spread frames – where plate and model show the same background, as
        `_grain` – combined (grain.combine); `added` = grain.decide for the setting. Stored in `_grain_shot`."""
        from engine.cleanup3d import grain as GR
        fr = list(frames)
        pick = fr[::max(1, int(np.ceil(len(fr) / GRAIN_FRAMES)))] if fr else []
        ms = []
        for f in pick:
            got = self.key_fill(f) if self.key_fill is not None else None
            if got is None:
                continue
            img, wgt = got
            drawn = self._model_area(f)
            box = mask_bbox([drawn], GRAIN_PAD_PX, self.w, self.h)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            if self.s["tone"] and getattr(self.key_fill, "tone", False):
                img = self._tone(img)
            free = self._free(f, wgt, drawn)
            plate = self.plate(f).astype(np.float32)
            ms.append(GR.measure(plate[y0:y1, x0:x1], np.asarray(img[y0:y1, x0:x1], np.float32),
                                 (free & ~drawn)[y0:y1, x0:x1], self.linear))
        meas = GR.combine(ms)
        self._grain_shot = {"meas": meas, "added": GR.decide(meas, GR.clean_mode(self.s.get("grain")))}
        return self._grain_shot

    def _grain(self, f: int, img: np.ndarray, wgt: np.ndarray, need: np.ndarray, plate: np.ndarray) -> np.ndarray:
        """The video model's frame with grain inside the patch (engine/cleanup3d/grain.apply – the module of Cleanup 3D,
        not a copy): measured in a crop around the model's area, on the ring where plate and model show the same
        background (`_free`); the noise seeded by the frame (grain.synth), so a re-run is identical. Returns a new
        array (the fill's own frame is never changed); the report per frame in `_grain_stats`."""
        from engine.cleanup3d import grain as GR
        if not need.any():
            return img
        shot = self._grain_shot
        if shot is not None and not shot["added"]:      # the shot needs no grain: nothing to measure or add here
            self._grain_stats[int(f)] = {"grain": {"added": False, "mode": GR.clean_mode(self.s.get("grain")),
                                                   "shot": True}}
            return img
        drawn = self._model_area(f)
        box = mask_bbox([drawn], GRAIN_PAD_PX, self.w, self.h)
        if box is None:
            return img
        x0, y0, x1, y1 = box
        free = self._free(f, wgt, drawn)
        st: Dict[str, Any] = {}
        g = GR.apply(np.asarray(img[y0:y1, x0:x1], np.float32), np.asarray(plate[y0:y1, x0:x1], np.float32),
                     drawn[y0:y1, x0:x1], free[y0:y1, x0:x1], need[y0:y1, x0:x1].astype(np.float32), int(f), self.s,
                     self.linear, st, meas=(shot or {}).get("meas"))
        self._grain_stats[int(f)] = st
        if not (st.get("grain") or {}).get("added"):
            return img
        out = np.array(img, np.float32, copy=True)
        out[y0:y1, x0:x1] = g
        return out

    def _tone(self, img: np.ndarray) -> np.ndarray:
        """A video model's frame with the shot's gain (`_gain`, measured once in `run`)."""
        g = self._gain
        if not g or not g.get("applied"):
            return img
        return np.asarray(img, np.float32) * np.asarray(g["bgr"], np.float32)

    def _sources(self, f: int, need: np.ndarray, out: np.ndarray, src: np.ndarray,
                 plate: Optional[np.ndarray] = None) -> np.ndarray:
        """The key fill (when given), then the clean plate (as `fill` allows); returns the pixels still to inpaint.
        `plate` = the frame's plate (the grain measurement; None = read it)."""
        if self.key_fill is not None:
            got = self.key_fill(f)
            if got is not None:
                img, wgt = got
                if self.s["tone"] and getattr(self.key_fill, "tone", False):
                    img = self._tone(img)                     # the shot's gain per channel (measured in `run`)
                if self._grain_on():                          # after the levelling, before the composite
                    img = self._grain(f, img, wgt, need, self.plate(f).astype(np.float32) if plate is None else plate)
                full = need & (wgt >= KEY_FULL)
                out[full] = img[full]
                src[full] = SRC_KEY
                part = need & (wgt > 0) & ~full       # stays "to fill" for the other sources, blended in _key_blend
                if part.any():
                    self._key_part[f] = (part, img[part].astype(np.float32), wgt[part].astype(np.float32))
        if self.s["fill"] in ("auto", "cleanplate"):
            for fn, code in ((self.clean, SRC_CLEAN), (self.auto_clean, SRC_AUTO)):
                if fn is None:
                    continue
                got = fn(f)
                if got is None:
                    continue
                img, ok = got
                ok = (ok >= AUTO_CONF_MIN) if code == SRC_AUTO else (ok > 0.5)
                use = need & ok & (src == SRC_PLATE)
                out[use] = img[use]
                src[use] = code
        return need & (src == SRC_PLATE)

    def _key_blend(self, f: int, out: np.ndarray, src: np.ndarray) -> None:
        """Pixels the key covers only partly: key x weight + what the other sources gave x (1 - weight); the key alone
        where nothing else filled the pixel (it would otherwise be mixed with the removed object)."""
        kp = self._key_part.pop(f, None)
        if kp is None:
            return
        part, img, wgt = kp
        w = np.where(src[part] == SRC_PLATE, 1.0, wgt).astype(np.float32)
        out[part] = img * w[:, None] + out[part] * (1.0 - w[:, None])
        code = src[part]
        code[w >= 0.5] = SRC_KEY
        src[part] = code

    def run(self, frames: Sequence[int], out_dir: Path, ext: str) -> Dict[str, Any]:
        t0 = time.time()
        out_dir = Path(out_dir)
        for sub in ("patch", "comp", "source"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        frames = list(frames)
        if self.plate_file is not None and frames:
            from engine.plate_write import plate_format
            self.fmt = plate_format(self.plate_file(frames[0]), self.w, self.h)
            if self.fmt is not None:
                self.notes.extend(n for n in self.fmt.notes if n not in self.notes)
                if not self.fmt.native:
                    self.fmt = None
        if self.kept is None:      # the kept objects that look lost: warned, the holdout changed only by the setting
            self.keep_alphas, self.kept = kept_guard(frames, self.remove_alphas, self.keep_alphas, self.s,
                                                     self.keep_names, self.progress)
        self.notes.extend(w for w in self.kept.get("warnings") or [] if w not in self.notes)
        # a caller may hand in the gain / grain measured before (a recomposite: the same model frames over the same
        # model area – measuring again would read every frame once more for the same numbers)
        if self.s["tone"] and self.key_fill is not None and getattr(self.key_fill, "tone", False) and frames:
            if self._gain is None:
                self.progress("Czyszczenie: wyrównanie poziomu – pomiar wzmocnienia ujęcia", 0, len(frames))
                self._gain = self._measure_gain(frames)
            if self._gain and not self._gain["applied"] and self._gain.get("why"):
                self.notes.append(f"Wyrównanie poziomu pominięte: {self._gain['why']}")
        if self._grain_on() and frames and self._grain_shot is None:
            self.progress("Czyszczenie: ziarno – pomiar ujęcia", 0, len(frames))
            self.measure_grain(frames)
        names = dict(SOURCE_NAMES)
        if self.key_fill is not None:
            names[SRC_KEY] = getattr(self.key_fill, "label", None) or KEY_NAME   # e.g. "VOID" (step C1)
        if self.inpaint_model not in ("lama",):
            names[MODEL_SRC[self.inpaint_model]] = MODEL_LABEL[self.inpaint_model]   # Qwen-Image-2.1
        counts = {names[k]: 0 for k in names if k != SRC_PLATE}
        empty = 0
        bs = self.s["batch"]
        writer = _Writer(self._write)
        try:
            self._frames(frames, out_dir, ext, bs, names, counts, writer)
        except BaseException:
            writer.abort()
            raise
        writer.close()
        empty = writer.empty
        if self._cut:
            self.notes.append(f"Pas łaty sięga dalej niż obszar narysowany przez model (zapas generowania "
                              f"{self.s['gen_margin_px']:.0f} px + rozrost silnika) – ucięty na tej granicy w "
                              f"{frame_ranges(self._cut)}")
        rj = {"schema": SCHEMA, "settings": self.s, "frames": [frames[0], frames[-1]] if frames else None,
              "n_frames": len(frames), "ext": file_ext(ext), "bits": {"exr": 32, "png16": 16}.get(ext, 8), "linear": self.linear, "backend": self.backend,
              "counts": counts, "empty_frames": empty, "notes": self.notes,
              "seconds": round(time.time() - t0, 1), "date": datetime.now().isoformat(timespec="seconds"),
              "kept_check": kept_public(self.kept), **self.report}
        if self._cut:            # frames where the composite band was cut at what the model drew
            rj["band_cut"] = sorted(self._cut)
        if self._gain:           # the shot's gain per channel (B, G, R), how it was measured, applied or why not
            rj["tone"] = dict(self._gain)
        if self._grain_stats:    # grain match: plate / patch noise (equivalent sigma, model units), frames grained
            from engine.cleanup3d import grain as GR
            rj["grain"] = GR.summary(self._grain_stats)
            sm = (self._grain_shot or {}).get("meas")
            if rj["grain"] is not None and sm is not None:    # the shot's one measurement (measure_grain)
                rj["grain"].update(plate=[round(float(v), 5) for v in sm["plate_sigma"]],
                                   patch=[round(float(v), 5) for v in sm["patch_sigma"]],
                                   measured=int(sm.get("frames") or 0), shot=True)
        rj["comp_ext"] = self.fmt.comp_ext if self.fmt is not None else file_ext(ext)
        if self.fmt is not None:
            rj["format"] = self.fmt.describe()
        if self.plate_file is not None and frames:
            rj["rgb_source"] = str(self.plate_file(frames[0])).replace("\\", "/")
        used = getattr(self.inpaint, "used", None)
        if isinstance(used, dict):
            rj["lama"] = dict(used)
        from engine.jsonio import write_json
        write_json(out_dir / "run.json", rj, indent=1, ensure_ascii=False)
        (out_dir / "MODELS.md").write_text(models_md(rj), encoding="utf-8")
        return rj

    def preview(self, f: int) -> Tuple[np.ndarray, np.ndarray]:
        """(comp, mask) of ONE frame by the run's own pieces (mask, key fill, levelling `_gain` as set by the caller,
        grain, key blend) – the live preview of a recomposite (user decision 2026-11-13). No inpaint model for what the
        key fill leaves (a video model's frame covers the whole mask) and no kept-object check: those stay the run's."""
        plate = self.plate(f).astype(np.float32)
        m = self.mask(f)
        need = m > 0
        out = plate.copy()
        src = np.zeros(m.shape, np.uint8)
        if need.any():
            self._sources(f, need, out, src, plate)
            self._key_blend(f, out, src)
        m3 = m[..., None]
        return plate * (1.0 - m3) + out * m3, m

    def _frames(self, frames, out_dir: Path, ext: str, bs: int, names, counts, writer: "_Writer") -> None:
        from concurrent.futures import ThreadPoolExecutor

        def prep(i_f):
            i, f = i_f
            self.progress(f"Czyszczenie: maska i clear plate F{f}", i, len(frames))
            plate = self.plate(f).astype(np.float32)
            m = self.mask(f)
            need = m > 0
            out = plate.copy()
            src = np.zeros(m.shape, np.uint8)
            rest = self._sources(f, need, out, src, plate) if need.any() else need
            return f, plate, m, out, src, rest
        if frames and not hasattr(self, "_band_pts"):     # the band points built once, before the threads ask
            self.mask(frames[0])
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="cleanup_frame") if self.workers > 1 else None
        try:
            self._frames_loop(frames, out_dir, ext, bs, names, counts, writer, prep, pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

    def _frames_loop(self, frames, out_dir, ext, bs, names, counts, writer, prep, pool) -> None:
        for b0 in range(0, len(frames), bs):
            chunk = list(enumerate(frames[b0:b0 + bs], start=b0))
            items = list(pool.map(prep, chunk)) if pool is not None else [prep(x) for x in chunk]   # frame order kept
            todo = [it for it in items if it[5].any()] if self.s["fill"] != "cleanplate" else []
            if self.s["fill"] == "cleanplate" and any(it[5].any() for it in items):
                self.notes.append("tylko clear plate: część maski bez źródła została z oryginału")
            if todo:
                self._inpaint(todo, b0, len(frames))
            for f, plate, m, out, src, rest in items:
                self._key_blend(f, out, src)
                if not (m > 0).any():
                    writer.empty += 1
                for k in counts:
                    code = next(c for c, n in names.items() if n == k)
                    counts[k] += int((src == code).sum())
                writer.put(out_dir, f, plate, m, out, src, ext)

    def _inpaint(self, todo, done0: int, total: int) -> None:
        box = mask_bbox([it[5] for it in todo], self.s["pad_px"], self.w, self.h)
        x0, y0, x1, y1 = box
        cw, ch = x1 - x0, y1 - y0
        side, mult = LAMA_MAX_SIDE, 8
        sc = min(1.0, side / float(max(cw, ch)))
        sz = (max(mult, int(round(cw * sc)) // mult * mult), max(mult, int(round(ch * sc)) // mult * mult))
        label = MODEL_LABEL[self.inpaint_model]
        imgs, masks = [], []
        for f, plate, m, out, src, rest in todo:
            im = to_model(out[y0:y1, x0:x1], self.linear)
            mk = rest[y0:y1, x0:x1].astype(np.float32)
            imgs.append(cv2.resize(im, sz, interpolation=cv2.INTER_AREA))
            masks.append(cv2.dilate((cv2.resize(mk, sz, interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)))
        where = getattr(self.inpaint, "where", "w ComfyUI")
        self.progress(f"Czyszczenie: {label + ' ' + where if self.inpaint else 'OpenCV'} ({len(todo)} kl., {sz[0]}×{sz[1]} px)", done0, total)
        filled = None
        if self.inpaint is not None:
            try:
                filled = self.inpaint(imgs, masks)
            except Exception as e:   # Comfy gone / the model missing: the OpenCV fill, said in run.json
                from engine.jobs import JobCancelled
                if isinstance(e, JobCancelled):
                    raise
                if self.inpaint_model != "lama":
                    # a chosen generative model (Qwen-Image-2.1) failing must stop the run – an OpenCV smear in its
                    # place is not what was asked for (user 2026-11-10: the run "finished" with no Qwen result)
                    raise RuntimeError(f"{label}: {str(e)[:300]}") from e
                self.notes.append(f"{label} niedostępna ({str(e)[:160]}) – wypełnienie OpenCV")
                self.backend = "opencv"
            said = getattr(self.inpaint, "notes", None)     # LamaInpainter: the full-resolution worker fell back
            if said:
                self.notes.extend(n for n in said if n not in self.notes)
                said.clear()
        code = MODEL_SRC[self.inpaint_model] if filled is not None else SRC_CV
        if filled is None:
            filled = opencv_batch(imgs, masks)
        for (f, plate, m, out, src, rest), fi in zip(todo, filled):
            back = from_model(cv2.resize(fi, (cw, ch), interpolation=cv2.INTER_CUBIC), self.linear)
            r = rest[y0:y1, x0:x1]
            reg = out[y0:y1, x0:x1]
            reg[r] = back[r]
            src[y0:y1, x0:x1][r] = code

    def _write(self, out_dir: Path, f: int, plate: np.ndarray, m: np.ndarray, out: np.ndarray, src: np.ndarray,
               ext: str) -> None:
        m3 = m[..., None]
        comp = plate * (1.0 - m3) + out * m3
        patch = np.dstack([out * m3, m])
        fmt = self.fmt
        if fmt is not None:      # the plate's own format; outside the mask the source file's values (plate_write)
            fmt.write_comp(out_dir / "comp" / fmt.comp_name(f), self.plate_file(f), comp, m)
            if fmt.patch_exr():
                fmt.write_patch_exr(out_dir / "patch" / f"p_{f:04d}.exr", patch)
            else:
                hi, dt = (65535, np.uint16) if ext == "png16" else (255, np.uint8)
                cv2.imwrite(str(out_dir / "patch" / f"p_{f:04d}.png"), np.clip(np.rint(patch * hi), 0, hi).astype(dt))
            cv2.imwrite(str(out_dir / "source" / f"s_{f:04d}.png"), src)
            return
        if ext == "exr":
            cv2.imwrite(str(out_dir / "comp" / f"c_{f:04d}.exr"), comp.astype(np.float32))
            cv2.imwrite(str(out_dir / "patch" / f"p_{f:04d}.exr"), patch.astype(np.float32))
        else:                                   # png16 for deeper plates (the full frame keeps the plate's precision)
            hi, dt = (65535, np.uint16) if ext == "png16" else (255, np.uint8)
            cv2.imwrite(str(out_dir / "comp" / f"c_{f:04d}.png"), np.clip(np.rint(comp * hi), 0, hi).astype(dt))
            cv2.imwrite(str(out_dir / "patch" / f"p_{f:04d}.png"), np.clip(np.rint(patch * hi), 0, hi).astype(dt))
        cv2.imwrite(str(out_dir / "source" / f"s_{f:04d}.png"), src)
