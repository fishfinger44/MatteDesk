# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Will a model fit THIS machine's GPU? – the Doctor's warning before an install (user request 2026-11-10: "warn at the
install in the Doctor what will spill or run unstably on this machine").

One verdict per component (key = doctor.LINKS key) against the card's total memory:
  ok       – fits the card, no spill;
  spill    – works, but part of the model / activations spill to system RAM (Windows WDDM shared memory or ComfyUI's
             partial loading): slower, needs free RAM;
  unstable – may run out of memory mid-run or depends on settings MatteDesk does not set (e.g. no block swap);
  no       – does not fit: the run ends with an out-of-memory error (no offload path in MatteDesk).
Where a model has measured points in engine/vram_table.json (VideoMaMa) the verdict comes from engine/vram_forecast
(the largest crop that fits); the others are tiers from measurements written in PROGRESS / LOCAL_CHECKS or – marked
`basis: estimate` – from the size of the weights. A new heavy model = a row in NEEDS (tests/test_vram_fit.py checks it).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from engine import vram_forecast as vf

LEVELS = ("ok", "spill", "unstable", "no")
DESKTOP_GB = 1.5        # what Windows + the browser usually hold on the card next to the model (base_gb of the measurements)
TOL_GB = 0.3            # a "12 GB" card reports 11.7-12.0 GB (Comfy /system_stats vs nvidia-smi): tiers compare with this slack

# key -> {"basis": "measured" | "estimate", "why": source of the numbers, "tiers": [(min card GB, level, text)]}
# tiers from the largest; the first with card >= min applies. Texts are Polish (UI), {gb} = the card's memory.
NEEDS: Dict[str, Dict[str, Any]] = {
    "ltx": {
        "basis": "measured", "why": "szczyt 22,5 GB na RTX 4090 przy 1280×736 / 97 kl. (ładowanie dynamiczne)",
        "tiers": [(23.0, "spill", "mieści się z dynamicznym ładowaniem wag (część z 22B w RAM-ie) – potrzebne ≥ 64 GB RAM"),
                  (0.0, "no", "przy {gb} GB VRAM nie zmieści się: szczyt 22,5 GB nawet z dynamicznym ładowaniem; "
                              "etap 7 wymaga karty 24 GB")]},
    "void": {
        "basis": "measured", "why": "szczyt 16,4 GB w 1344×768, okno 57 kl. (stary tryb repo + venv, RTX 4090, 2026-11-08); "
                                  "natywnie w ComfyUI (od 2026-11-11) bez pomiaru",
        # natively in ComfyUI (2026-11-11) the model (void_pass1 11.1 GB) is loaded partly from RAM on smaller cards by
        # ComfyUI's own memory management – not measured yet (LOCAL_CHECKS)
        "tiers": [(19.0, "ok", "mieści się (szczyt ok. 16,4 GB)"),
                  (10.0, "spill", "przy {gb} GB VRAM ComfyUI ładuje model 11,1 GB częściowo z RAM-u – działa wolniej, "
                                  "niesprawdzone; potrzebne ≥ 32 GB RAM"),
                  (0.0, "unstable", "przy {gb} GB VRAM możliwy brak pamięci (szczyt 16,4 GB przy pełnym trybie)")]},
    "seedvr2": {
        "basis": "estimate", "why": "wariant wg karty: 3B fp8 3,4 GB poniżej 16 GB, 7B sharp fp8 8,2 GB od 16 GB; VAE 0,5 GB kafelkowany (bez pomiaru)",
        "tiers": [(16.0, "ok", "powinien się zmieścić (wagi 7B fp8 8,2 GB + VAE 0,5 GB)"),
                  (8.0, "ok", "mieści się: dla tej karty wariant 3B fp8 (3,4 GB) + VAE 0,5 GB"),
                  (6.0, "unstable", "przy {gb} GB VRAM na granicy: wagi 3B fp8 3,4 GB + aktywacje klipu; MatteDesk "
                                     "nie ustawia „block swap”, więc dłuższy klip może skończyć się brakiem pamięci"),
                  (0.0, "no", "przy {gb} GB VRAM nie zmieści się (wagi 3B fp8 3,4 GB)")]},
    "flux3_klein": {
        "basis": "estimate", "why": "model bf16 11,1 GB; koder 8,9 GB ładowany osobno (bez pomiaru)",
        "tiers": [(16.0, "ok", "mieści się (model 11,1 GB, koder ładowany osobno)"),
                  (10.0, "spill", "przy {gb} GB VRAM ComfyUI ładuje model 11,1 GB częściowo (reszta w RAM-ie) – "
                                  "działa, ale wolniej"),
                  (0.0, "unstable", "przy {gb} GB VRAM większość modelu 11,1 GB zostaje w RAM-ie – bardzo wolno, "
                                    "możliwy brak pamięci")]},
    "qwen21": {
        "basis": "estimate", "why": "model int8 6,8 GB i koder int8 8,7 GB ładowane po kolei (bez pomiaru)",
        "tiers": [(12.0, "ok", "mieści się: model 6,8 GB i koder 8,7 GB ładowane po kolei"),
                  (8.0, "spill", "przy {gb} GB VRAM koder 8,7 GB ładuje się częściowo (reszta w RAM-ie) – wolniej"),
                  (0.0, "unstable", "przy {gb} GB VRAM model i koder nie mieszczą się – bardzo wolno albo brak pamięci")]},
    "h3": {
        "basis": "estimate", "why": "model fl2va pruned int8 19,5 GB + koder Qwen3-VL 32B 14,6–24,6 GB (bez pomiaru)",
        "tiers": [(32.0, "ok", "powinien się zmieścić (model 19,5 GB, koder ładowany osobno)"),
                  (0.0, "spill", "przy {gb} GB VRAM model 19,5 GB i koder Qwen3-VL 32B ładują się częściowo z RAM-u – "
                                 "działa, ale wolno; potrzebne ≥ 64 GB RAM")]},
    "vace": {
        "basis": "estimate", "why": "eksperty Wan 2.2 Fun VACE Q4_K_S po 10,6 GB ładowane po kolei + koder umt5 fp8 6,7 GB osobno (bez pomiaru)",
        "tiers": [(16.0, "ok", "powinien się zmieścić (ekspert 10,6 GB, koder ładowany osobno)"),
                  (12.0, "spill", "przy {gb} GB VRAM ekspert 10,6 GB zostawia mało miejsca na klatki – część wag w RAM-ie, "
                                  "wolniej; przy 720p dłuższe klipy mogą się nie zmieścić (wtedy 480p)"),
                  (0.0, "unstable", "przy {gb} GB VRAM ekspert 10,6 GB nie mieści się – bardzo wolno albo brak pamięci")]},
    "ltx_fill": {
        "basis": "estimate", "why": "LTX-2.5 22B distilled int8 21,5 GB + koder Gemma 4 12B int8 15,4 GB ładowany osobno; szczyt etapu 7 ok. 22,5 GB (pomiar LTX-2.5 Alpha-Gen)",
        "tiers": [(24.0, "ok", "powinien się zmieścić (szczyt ok. 22,5 GB jak w etapie 7)"),
                  (0.0, "spill", "przy {gb} GB VRAM transformer 21,5 GB ładuje się częściowo z RAM-u – działa, ale wolno")]},
    "prompt_assist": {
        "basis": "estimate", "why": "Qwen3-VL-4B-Instruct bf16: wagi 8,9 GB + obraz 1024 px i ok. 500 tokenów (bez pomiaru)",
        "tiers": [(11.0, "ok", "mieści się (wagi 8,9 GB bf16, model wyładowany po pracy)"),
                  (8.0, "spill", "przy {gb} GB VRAM część wag 8,9 GB zostaje w RAM-ie – wolniej"),
                  (0.0, "unstable", "przy {gb} GB VRAM wagi 8,9 GB się nie mieszczą – bardzo wolno albo brak pamięci")]},
    "cotracker3": {
        "basis": "measured", "why": "szczyt 9,3 GiB przy paczce 2048 (SH010, LOCAL_CHECKS)",
        "tiers": [(11.0, "ok", "mieści się (szczyt ok. 9,3 GB)"),
                  (0.0, "spill", "przy {gb} GB VRAM szczyt ok. 9,3 GB przy dużym kadrze przeleje się do RAM-u – wolniej")]},
}


def _videomama(card: float) -> Tuple[str, str]:
    """VideoMaMa from the measured line: the largest square crop that fits, without / with the node's CPU offload."""
    budget = card - vf.MARGIN_GB - DESKTOP_GB
    side = vf.side_for("videomama", 24, budget) or 0
    side_off = vf.side_for("videomama", 24, budget, "offload") or 0
    off = f" Z „offload” do ok. {side_off} px." if side_off > side else ""
    if side >= 1280:
        return "ok", "mieści się do kadru 1280 px (pomiar RTX 4090)"
    if side >= 512:
        return "spill", (f"przy {card:.0f} GB VRAM bez przelewania do kadru ok. {side} px; większe kadry przelewają "
                         "się do RAM-u albo kończą brakiem pamięci (MatteDesk ostrzega przed startem)." + off)
    if side_off >= 512:
        return "unstable", (f"przy {card:.0f} GB VRAM tylko z „offload” (do ok. {side_off} px); bez niego brak pamięci")
    return "no", f"przy {card:.0f} GB VRAM nie zmieści się nawet kadr 512 px"


def assess(key: str, card_gb: Optional[float]) -> Optional[Dict[str, Any]]:
    """{level, text, card_gb, basis, why} of component `key` on a card of `card_gb`; None = not rated (light model,
    or the card is unknown)."""
    if card_gb is None or card_gb <= 0:
        return None
    card = float(card_gb)
    if key == "videomama":
        if vf.fit(vf.points("videomama")) is None:
            return None
        level, text = _videomama(card)
        return {"level": level, "text": text, "card_gb": round(card, 1), "basis": "measured",
                "why": "pomiary engine/vram_table.json (RTX 4090, 2026-11-06)"}
    need = NEEDS.get(key)
    if not need:
        return None
    for min_gb, level, text in need["tiers"]:
        if card + TOL_GB >= min_gb:
            return {"level": level, "text": text.format(gb=f"{card:.0f}"), "card_gb": round(card, 1),
                    "basis": need["basis"], "why": need["why"]}
    return None


def rated_keys() -> List[str]:
    return ["videomama"] + list(NEEDS)


def card_gb(comfy_vram_gb: Optional[float] = None) -> Optional[float]:
    """Total GPU memory: ComfyUI's /system_stats (what the Doctor read), else nvidia-smi; None = unknown."""
    if comfy_vram_gb:
        return float(comfy_vram_gb)
    now = time.monotonic()
    if _CARD["t"] and now - _CARD["t"] < CARD_TTL_S:     # nvidia-smi once a minute (the Cleanup 3D state is polled)
        return _CARD["gb"]
    try:
        from engine.sysinfo import gpu_info
        gb = vf.card_gb(None, gpu_info() or [])
    except Exception:
        gb = None
    _CARD.update(t=now, gb=gb)
    return gb


CARD_TTL_S = 60.0
_CARD: Dict[str, Any] = {"t": 0.0, "gb": None}
