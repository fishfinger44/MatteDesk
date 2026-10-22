# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prompt rules of the `matte` zone without a VLM (docs/PLAN_HAIR_TOOL.md section 7, step 9).

`interpret_prompt(text)` reads the zone description (PL / EN keywords) into a `PromptPlan`:
- preset (hair / fibres / smoke / motion / thin) from the subject words,
- `fg` = the subject phrase (what to cut) and `fg_en` – the same in English for SAM3_Detect on the zone crop,
- `distractors` = phrases after "bez / without / no / minus / oprócz / pomiń / remove / ignore …" (what to get rid of in
  the background); line-like distractors set `bg_line` (Hough line detection in the background ring -> proposed "-" strokes),
- `engine_hints` = multipliers of the routing scores (`localmatte_route.engine_weights`): patterned / colourful background
  -> chroma down, plain / sky / wall -> chroma up, "same colour as the background" -> chroma off, knn up + warning; an
  engine named in the text is favoured ("tylko <silnik>" / "only <engine>" = the others off),
- `warnings` for the UI (nothing here changes a zone by itself – the UI / API apply the plan, strokes are confirmed).

`detect_bg_lines(img, region, core)` proposes "-" strokes along straight lines in the background of the zone (HoughLinesP
on Canny edges outside the dilated core). `propose_zones(core, fg_mask, soft)` turns "SAM 3.1 foreground outside the core"
(+ soft edge pixels) into zone proposals (polygons) to accept in the UI. Pure functions, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.localmatte import ENGINES, PRESETS

# ---------------------------------------------------------------------------------------------------- vocabulary
# token prefixes (lower case, no diacritics) -> preset
_PRESET_STEMS: Dict[str, Sequence[str]] = {
    "hair": ("wlos", "kosmyk", "futr", "siersc", "grzyw", "brod", "wasy", "wasik", "rzes", "hair", "fur", "strand", "curl",
             "beard", "mane", "whisker", "eyelash", "lash", "ponytail", "warkocz", "fryzur"),
    "fibres": ("wlokn", "filc", "nitk", "tkanin", "weln", "wlocz", "fibre", "fiber", "felt", "thread", "fabric", "wool", "yarn", "lint"),
    "smoke": ("dym", "mgl", "woalk", "oddech", "smoke", "steam", "fog", "mist", "haze", "veil", "vapor", "vapour"),
    "motion": ("rozmy", "nieostr", "poruszon", "defocus", "blur", "motion", "smuz"),
    "thin": ("traw", "galaz", "galez", "anten", "cienk", "drucik", "sznur", "lisci", "grass", "branch", "twig", "antenna",
             "thin", "string", "rope", "leaves", "reed", "trzcin"),
}
_WORD_PRESET: Dict[str, str] = {"clip": "hair", "lok": "hair", "lokow": "hair", "para": "smoke", "pary": "smoke", "parze": "smoke",
                                "ruch": "motion", "ruchu": "motion", "fast": "motion"}
_LINE_STEMS = ("lini", "slup", "krawedz", "kabel", "drut", "listw", "belk", "pret", "rur", "line", "pole", "edge", "cable",
               "rail", "wire", "bar", "rod", "pipe", "beam", "frame", "ramk")
_DISTRACTOR_MARKERS = ("bez", "oprocz", "pomin", "pominac", "usun", "minus", "without", "no", "not", "remove", "ignore", "except", "minus")
_BUSY_BG = ("pstr", "plam", "wzorz", "wzork", "kolorow", "barwn", "zabrudz", "chaotyczn", "patterned", "busy", "colourful",
            "colorful", "cluttered", "textured", "noisy", "spotty")
_PLAIN_BG = ("jednolit", "gladk", "niebo", "nieb", "jasne", "ciemne", "scian", "biale", "czarne", "szare", "plain", "uniform", "sky",
             "wall", "white", "black", "grey", "gray", "flat", "solid", "greenscreen", "bluescreen", "ekran")
_SAME_PHRASES = ("w kolorze tla", "ten sam kolor", "tym samym kolorze", "podobny kolor", "podobnym kolorze", "zlewa", "zlewaj",
                 "same colour", "same color", "similar colour", "similar color", "blends", "blend into", "camouflag")
_ONLY = ("tylko", "wylacznie", "jedynie", "only", "just")
_ENGINE_WORDS: Dict[str, str] = {"chroma": "chroma", "chrom": "chroma", "kluczem": "chroma", "knn": "knn", "cf": "cf", "closed": "cf",
                                 "clean": "clean", "cleanplate": "clean", "roznic": "clean", "matanyone": "matanyone", "matanyone2": "matanyone",
                                 "vitmatte": "vitmatte", "backlit": "backlit", "backlight": "backlit", "podswietl": "backlit"}
# PL -> EN for the SAM3 phrase (first word form wins; inflected forms reduced by prefix)
_PL_EN: Sequence[Tuple[str, str]] = (
    ("wlos", "hair"), ("kosmyk", "hair strands"), ("futr", "fur"), ("siersc", "fur"), ("grzyw", "mane"), ("brod", "beard"),
    ("warkocz", "braid"), ("fryzur", "hairstyle"), ("wlokn", "fibres"), ("filc", "felt"), ("nitk", "threads"), ("dym", "smoke"),
    ("par", "steam"), ("mgl", "fog"), ("traw", "grass"), ("galaz", "branches"), ("galez", "branches"), ("anten", "antennae"),
    ("lisc", "leaves"), ("drzew", "tree"), ("sukn", "dress"), ("sukien", "dress"), ("spodnic", "skirt"), ("szal", "scarf"),
    ("pies", "dog"), ("psa", "dog"), ("psy", "dogs"), ("kot", "cat"), ("rek", "hand"), ("rac", "hands"), ("dlon", "hand"),
    ("glow", "head"), ("twarz", "face"), ("osob", "person"), ("kobiet", "woman"), ("mezczyzn", "man"), ("dzieck", "child"),
    ("czlowiek", "person"), ("postac", "figure"), ("lalk", "puppet"), ("kukiel", "puppet"), ("ptak", "bird"), ("pior", "feathers"),
    ("skrzyd", "wings"), ("kwiat", "flower"), ("lisci", "leaves"), ("chmur", "cloud"), ("wod", "water"), ("fal", "waves"),
)
_STOP = {"i", "oraz", "a", "the", "and", "of", "z", "ze", "na", "w", "we", "do", "u", "o", "po", "przy", "od", "dla", "to", "in", "on", "at", "with"}


def _fold(t: str) -> str:
    """Lower case without Polish diacritics (so stems match any inflection / spelling)."""
    tr = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return t.translate(tr).lower()


def _tokens(t: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", _fold(t)) if w]


@dataclass
class PromptPlan:
    text: str = ""
    preset: Optional[str] = None                 # None = the prompt does not name a kind of detail
    fg: str = ""                                 # subject phrase as written
    fg_en: str = ""                              # subject phrase for SAM3_Detect (English, best effort)
    distractors: List[str] = field(default_factory=list)
    bg_line: bool = False                        # a line-like distractor -> Hough "-" strokes to confirm
    same_color_bg: bool = False
    engine_hints: Dict[str, float] = field(default_factory=dict)
    forced_engine: Optional[str] = None          # "tylko knn" / "only chroma"
    warnings: List[str] = field(default_factory=list)
    matched: List[str] = field(default_factory=list)   # rule ids that fired (for tests / the UI line)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)) for k, v in self.__dict__.items()}


def _preset_of(tokens: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    hits: List[Tuple[int, str, str]] = []
    for i, w in enumerate(tokens):
        if w in _WORD_PRESET:
            hits.append((i, _WORD_PRESET[w], w))
            continue
        for preset, stems in _PRESET_STEMS.items():
            if any(w.startswith(s) for s in stems):
                hits.append((i, preset, w))
                break
    if not hits:
        return None, None
    hits.sort()
    return hits[0][1], hits[0][2]


_NOT_SUBJECT = ("kolor", "tla", "tle", "tlo", "background", "bg", "behind", "ruch", "szybk", "same", "similar", "podobn")


def _translate(phrase: str) -> str:
    out = []
    for w in _tokens(phrase):
        if w in _STOP or w in _ONLY or w in _ENGINE_WORDS or any(w.startswith(s) for s in _NOT_SUBJECT):
            continue
        if any(w.startswith(s) for s in _BUSY_BG + _PLAIN_BG) or w.startswith("rozmy") or w.startswith("nieostr") or w.startswith("podobn"):
            continue
        en = next((e for s, e in _PL_EN if w.startswith(s)), None)
        out.append(en or w)
    # drop duplicates keeping order
    seen, res = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            res.append(w)
    return " ".join(res)


def interpret_prompt(text: Optional[str]) -> PromptPlan:
    plan = PromptPlan(text=(text or "").strip())
    if not plan.text:
        return plan
    folded = _fold(plan.text)
    # 1. split into the subject part and distractor parts
    parts = [p.strip() for p in re.split(r"[,;.\n]+|\s[-–—]\s", plan.text) if p.strip()]
    subject: List[str] = []       # the first part (before any distractor marker) = what to cut; later parts = context
    for pi, part in enumerate(parts):
        words = re.findall(r"\S+", part)
        folded_w = [_tokens(w)[0] if _tokens(w) else "" for w in words]
        cut = next((i for i, w in enumerate(folded_w) if w in _DISTRACTOR_MARKERS), None)
        if cut is None:
            if pi == 0:
                subject.append(part)
            continue
        if cut > 0 and pi == 0:
            subject.append(" ".join(words[:cut]))
        rest = [w for w in _tokens(" ".join(words[cut + 1:])) if w not in _STOP]
        if rest:
            plan.distractors.append(" ".join(rest))
            plan.matched.append("distractor")
    plan.fg = " ".join(s for s in subject if s).strip()
    toks_fg = _tokens(plan.fg)
    plan.fg_en = _translate(plan.fg)
    # 2. preset from the subject (fall back to the whole text)
    preset, word = _preset_of(toks_fg)
    if not preset:
        preset, word = _preset_of(_tokens(plan.text))   # the kind of detail may sit in a later part ("tylko knn, clip")
    if preset:
        plan.preset = preset
        plan.matched.append(f"preset:{word}")
    # 3. background character
    tokens = _tokens(plan.text)
    if any(w.startswith(s) for w in tokens for s in _BUSY_BG):
        plan.engine_hints["chroma"] = min(plan.engine_hints.get("chroma", 1.0), 0.3)
        plan.matched.append("bg:busy")
    if any(w.startswith(s) for w in tokens for s in _PLAIN_BG) and "bg:busy" not in plan.matched:
        plan.engine_hints["chroma"] = max(plan.engine_hints.get("chroma", 1.0), 1.5)
        plan.matched.append("bg:plain")
    if any(ph in folded for ph in _SAME_PHRASES):
        plan.same_color_bg = True
        plan.engine_hints["chroma"] = 0.0
        plan.engine_hints["knn"] = max(plan.engine_hints.get("knn", 1.0), 1.3)
        plan.warnings.append("Włosy w kolorze tła: chroma wyłączona, lepiej knn / MatAnyone2 i klucz w trudnej klatce")
        plan.matched.append("bg:same")
    # 4. line-like distractors -> Hough "-" strokes
    for d in plan.distractors:
        if any(w.startswith(s) for w in _tokens(d) for s in _LINE_STEMS):
            plan.bg_line = True
            plan.matched.append("bg:line")
            break
    # 5. engines named in the text
    named = []
    for i, w in enumerate(tokens):
        eng = next((e for k, e in _ENGINE_WORDS.items() if w == k or (len(k) >= 5 and w.startswith(k))), None)
        if eng and eng in ENGINES + ("backlit",) and eng not in named:
            named.append(eng)
            if i > 0 and tokens[i - 1] in _ONLY:
                plan.forced_engine = eng
    for eng in named:
        plan.engine_hints[eng] = max(plan.engine_hints.get(eng, 1.0), 1.5)
        plan.matched.append(f"engine:{eng}")
    if plan.forced_engine:
        for eng in ("chroma", "knn", "cf", "clean", "backlit"):
            if eng != plan.forced_engine:
                plan.engine_hints[eng] = 0.0
    if plan.preset and plan.preset not in PRESETS:   # defensive – the tables above only name known presets
        plan.preset = None
    return plan


def merge_hints(explicit: Optional[Dict[str, float]], plan: Optional[PromptPlan]) -> Dict[str, float]:
    """Effective routing multipliers: prompt-derived x explicit `engine_hints` of the zone (explicit always applies)."""
    out: Dict[str, float] = dict(plan.engine_hints) if plan else {}
    for k, v in (explicit or {}).items():
        out[k] = out.get(k, 1.0) * float(v)
    return out


# ---------------------------------------------------------------------------------------------------- lines in the background
def detect_bg_lines(img: np.ndarray, region: np.ndarray, core: np.ndarray, max_lines: int = 3, min_len_frac: float = 0.45,
                    radius_px: int = 3, margin_px: int = 24) -> List[Dict[str, Any]]:
    """Straight lines in the background around the zone (plan: "linia / słup / krawędź w tle -> kreska '-' do
    potwierdzenia"). Canny on the display image in bbox(region) + margin, edges inside the dilated core discarded,
    HoughLinesP; a line must be at least `min_len_frac` of the longer bbox side and must not cross the core.
    Returns "-" strokes `[{p: [[x1, y1], [x2, y2]], radius_px, length_px, source: "hough"}]` (plate px), longest first."""
    h, w = core.shape
    reg = region.astype(bool)
    if not reg.any():
        return []
    ys, xs = np.where(reg)
    x0, x1 = max(0, int(xs.min()) - margin_px), min(w, int(xs.max()) + 1 + margin_px)
    y0, y1 = max(0, int(ys.min()) - margin_px), min(h, int(ys.max()) + 1 + margin_px)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(np.clip(crop, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    g8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(g8, 50, 150)
    core_d = cv2.dilate(core[y0:y1, x0:x1].astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    edges[core_d] = 0
    min_len = max(20, int(min_len_frac * max(x1 - x0, y1 - y0)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return []
    out: List[Dict[str, Any]] = []
    for lx1, ly1, lx2, ly2 in lines.reshape(-1, 4):
        n = max(8, int(np.hypot(lx2 - lx1, ly2 - ly1) / 4))
        sx = np.linspace(lx1, lx2, n).round().astype(int).clip(0, x1 - x0 - 1)
        sy = np.linspace(ly1, ly2, n).round().astype(int).clip(0, y1 - y0 - 1)
        if core_d[sy, sx].mean() > 0.1:
            continue
        length = float(np.hypot(lx2 - lx1, ly2 - ly1))
        ang = float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))) % 180.0
        p = [[float(lx1 + x0), float(ly1 + y0)], [float(lx2 + x0), float(ly2 + y0)]]
        dup = False
        for o in out:   # near-parallel and near-collinear -> keep the longer one
            da = abs(o["angle"] - ang)
            da = min(da, 180 - da)
            mid = np.array([(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2])
            a_, b_ = np.array(o["p"][0]), np.array(o["p"][1])
            d_ = abs(np.cross(b_ - a_, mid - a_)) / max(1e-6, np.linalg.norm(b_ - a_))
            if da < 4 and d_ < 6:
                dup = True
                break
        if not dup:
            out.append({"p": p, "radius_px": int(radius_px), "length_px": round(length, 1), "angle": round(ang, 1), "source": "hough"})
    out.sort(key=lambda o: -o["length_px"])
    return out[:max_lines]


# ---------------------------------------------------------------------------------------------------- zone proposals
def soft_edge_candidates(img: np.ndarray, core: np.ndarray, ring_px: int = 24, bg_px: int = 48) -> np.ndarray:
    """Pixels just outside the core whose colour is closer to the object than to the local background (mixed /
    thin detail the core missed: strands, fibres). Lab distances; the local background = ring `ring_px`..`bg_px`."""
    from engine.localmatte import lab_of
    cb = core.astype(bool)
    if not cb.any():
        return np.zeros_like(cb)
    lab = lab_of(np.clip(img, 0, 1).astype(np.float32))
    k_in = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    k_out = np.ones((2 * bg_px + 1, 2 * bg_px + 1), np.uint8)
    d_in = cv2.dilate(cb.astype(np.uint8), k_in).astype(bool)
    d_out = cv2.dilate(cb.astype(np.uint8), k_out).astype(bool)
    ring = d_in & ~cb
    bgring = d_out & ~d_in
    if not ring.any() or not bgring.any():
        return np.zeros_like(cb)
    inner = cv2.erode(cb.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    fg_col = np.median(lab[inner if inner.any() else cb], axis=0)
    bg_col = np.median(lab[bgring], axis=0)
    d_fg = np.linalg.norm(lab - fg_col, axis=2)
    d_bg = np.linalg.norm(lab - bg_col, axis=2)
    sep = float(np.linalg.norm(fg_col - bg_col))
    if sep < 6.0:   # object and background alike – no colour evidence
        return np.zeros_like(cb)
    return ring & (d_fg < 0.6 * d_bg)


def propose_zones(core: np.ndarray, fg_mask: Optional[np.ndarray], soft: Optional[np.ndarray] = None, min_area: int = 400,
                  grow_px: int = 6, max_props: int = 6) -> List[Dict[str, Any]]:
    """Zone proposals = connected pieces of (SAM 3.1 foreground ∪ soft edge pixels) outside the core, grown by `grow_px`
    (the zone region should hold a margin around the strands). Each: polygon (plate px), bbox, area_px, where it lies
    relative to the core (lewo / prawo / góra / dół), source (sam3 / soft / both)."""
    cb = core.astype(bool)
    h, w = cb.shape
    cand = np.zeros((h, w), bool)
    src_sam = np.zeros((h, w), bool)
    if fg_mask is not None:
        src_sam = (np.asarray(fg_mask) > 0.5) & ~cb
        cand |= src_sam
    if soft is not None:
        cand |= np.asarray(soft).astype(bool) & ~cb
    if not cand.any():
        return []
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    ys, xs = np.where(cb)
    cy, cx = (float(ys.mean()), float(xs.mean())) if len(xs) else (h / 2, w / 2)
    out: List[Dict[str, Any]] = []
    for i in range(1, n):
        comp = lab == i
        area = int(comp.sum())
        if area < min_area:
            continue
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)).astype(bool)
        cnts, _ = cv2.findContours(grown.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        py, px = np.where(comp)
        mx, my = float(px.mean()), float(py.mean())
        dx, dy = mx - cx, my - cy
        side = ("prawo" if dx > 0 else "lewo") if abs(dx) >= abs(dy) else ("dół" if dy > 0 else "góra")
        in_sam = bool((comp & src_sam).any())
        in_soft = soft is not None and bool((comp & np.asarray(soft).astype(bool)).any())
        out.append({"polygon": poly.astype(float).tolist(), "area_px": area,
                    "bbox": [int(px.min()), int(py.min()), int(px.max()) + 1, int(py.max()) + 1], "side": side,
                    "source": "both" if in_sam and in_soft else ("sam3" if in_sam else "soft")})
    out.sort(key=lambda o: -o["area_px"])
    for i, o in enumerate(out[:max_props]):
        o["id"] = i
    return out[:max_props]
