# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/stage_models.py - model gate of the object stages (user rule 2026-10-16: a stage whose model is not downloaded
/ installed is unavailable – locked in the stage bar with the reason and the Doctor's button).

Only cheap file checks (no subprocess, no Comfy call): the files the Doctor checks on disk. Deeper checks (the package in
ComfyUI's Python, node signatures) stay in the Doctor. Disabled with MD_MODEL_GATE=0 (synthetic tests run without models).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# fine-grained stage -> (Doctor's install action label, checker(cfg) -> missing descriptions)
DOCTOR = {"core": "„Zainstaluj” przy „Model: SAM 3.1”", "cotracker": "„Zainstaluj” przy „Model: CoTracker3”",
          "matanyone": "„Zainstaluj” przy „Model: MatAnyone2”", "videomama": "„Zainstaluj VideoMaMa”",
          "vitmatte": "„Zainstaluj” przy „Model: ViTMatte” albo „Model: BiRefNet_HR-matting / Lucida”", "ltx": "„Zainstaluj LTX-2.5”"}
LABEL = {"core": "SAM 3.1", "cotracker": "CoTracker3", "matanyone": "MatAnyone2", "videomama": "VideoMaMa",
         "vitmatte": "ViTMatte", "ltx": "LTX-2.5 Alpha-Gen"}
# the SAM3 model gates the whole stage 1 (detect / crops / core)
ALSO = {"core": ("detect", "crops")}


def _md(cfg: Dict[str, Any]) -> Optional[Path]:
    m = cfg.get("models_dir")
    return Path(m) if m else None


def _sam3(cfg: Dict[str, Any]) -> List[str]:
    md, name = _md(cfg), cfg.get("sam3_checkpoint") or "sam3.1_multiplex_fp16.safetensors"
    return [] if md and (md / "checkpoints" / name).exists() else [f"checkpoints/{name}"]


def _cotracker(cfg: Dict[str, Any]) -> List[str]:
    md = _md(cfg)
    return [] if md and (md / "cotracker" / "scaled_offline.pth").exists() else ["cotracker/scaled_offline.pth"]


def _matanyone(cfg: Dict[str, Any]) -> List[str]:
    md = _md(cfg)
    return [] if md and (md / "matanyone2" / "matanyone2.pth").exists() else ["matanyone2/matanyone2.pth"]


def hf_cached(model_id: str) -> bool:
    """A model in the Hugging Face cache (HF_HOME / HF_HUB_CACHE or ~/.cache/huggingface) with a snapshot."""
    roots = [os.environ.get("HF_HUB_CACHE"), (Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             Path.home() / ".cache" / "huggingface" / "hub"]
    folder = "models--" + model_id.replace("/", "--")
    for r in roots:
        if r and (Path(r) / folder / "snapshots").is_dir() and any((Path(r) / folder / "snapshots").iterdir()):
            return True
    return False


def _vitmatte(cfg: Dict[str, Any]) -> List[str]:
    """Stage 5 needs one edge model of the zone engine: ViTMatte or (since 2026-10-21) BiRefNet_HR-matting / Lucida."""
    from engine.birefnet import MODELS as BIREFNET
    from engine.vitmatte import DEFAULT_MODEL
    ids = [DEFAULT_MODEL, *BIREFNET.values()]
    return [] if any(hf_cached(m) for m in ids) else [f"{' / '.join(ids)} (cache Hugging Face)"]


EDGE_ENGINES = ("vitmatte", "birefnet")   # stage 5 engine (object field `edge_engine`, user decision 2026-10-21)
EDGE_ENGINE_LABEL = {"vitmatte": "ViTMatte", "birefnet": "BiRefNet"}


def edge_engines(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Which stage-5 edge engines have weights on disk: ViTMatte (any hustvl model) / BiRefNet (BiRefNet_HR-matting or
    Lucida). The UI shows the stage menu only when both are there. Gate off (tests): both."""
    if not enabled():
        return {e: True for e in EDGE_ENGINES}
    from engine.birefnet import MODELS as BIREFNET
    from engine.vitmatte import MODELS as VITMATTE
    return {"vitmatte": any(hf_cached(m) for m in VITMATTE.values()),
            "birefnet": any(hf_cached(m) for m in BIREFNET.values())}


def effective_edge_engine(choice: Optional[str], installed: Dict[str, bool]) -> str:
    """The object's choice when installed; else the one that is installed; default ViTMatte (both or none)."""
    c = choice if choice in EDGE_ENGINES else "vitmatte"
    if installed.get(c):
        return c
    other = next((e for e in EDGE_ENGINES if e != c and installed.get(e)), None)
    return other or c


def _videomama(cfg: Dict[str, Any]) -> List[str]:
    from engine.videomama import NODE_PACK, models_status
    miss = [f"models/videomama/{m['dir']}/{m['file']}" for m in models_status(_md(cfg)) if not m["present"]]
    cn = Path(cfg.get("comfy_dir") or "") / "custom_nodes" / NODE_PACK
    if not (cfg.get("comfy_dir") and cn.is_dir()):
        miss.insert(0, f"węzeł {NODE_PACK}")
    return miss


def _ltx(cfg: Dict[str, Any]) -> List[str]:
    from engine.ltx_alpha import models_status
    return [f"{m['folder']}/{m['file']}" for m in models_status(_md(cfg)) if not m["present"]]


CHECKS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "core": _sam3, "cotracker": _cotracker, "matanyone": _matanyone, "videomama": _videomama, "vitmatte": _vitmatte,
    "ltx": _ltx}


def enabled() -> bool:
    return os.environ.get("MD_MODEL_GATE", "1") != "0"


def model_gate(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{fine-grained stage: {"missing": [...], "why": reason}} for every stage whose model is not on disk."""
    if not enabled():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for stage, check in CHECKS.items():
        try:
            miss = check(cfg)
        except Exception as e:   # a broken check never blocks silently: report it as missing
            miss = [f"sprawdzenie nie powiodło się: {e}"]
        if miss:
            more = f" (+{len(miss) - 2})" if len(miss) > 2 else ""
            why = (f"Brak modelu {LABEL[stage]}: {', '.join(miss[:2])}{more} – Doktor → {DOCTOR[stage]}"
                   if _md(cfg) or stage == "vitmatte" else
                   f"Brak models_dir w konfiguracji – Doktor (model {LABEL[stage]} niedostępny)")
            for s in (stage,) + ALSO.get(stage, ()):
                out[s] = {"missing": miss, "why": why, "model": LABEL[stage]}
    return out


def apply_gate(info: Dict[str, Any], gate: Dict[str, Dict[str, Any]], steps_members: Dict[str, List[str]]) -> Dict[str, Any]:
    """Lock the stages / bar steps of `info` (engine.stages.object_stages) whose model is missing: a fine-grained stage
    not done -> locked (a done result stays usable: export, bases); its bar step -> locked always (the stage cannot be
    opened or run). `steps_members`: step id -> fine-grained stages."""
    if not gate:
        return info
    for x in info.get("stages", []):
        g = gate.get(x["id"])
        if g and x["state"] != "done":
            x["state"], x["why"] = "locked", g["why"]
    for s in info.get("steps", []):
        g = next((gate[m] for m in steps_members.get(s["id"], []) if m in gate), None)
        if g:
            s["state"], s["why"] = "locked", g["why"]
            s["models_missing"] = True
    info["models_missing"] = {k: v["why"] for k, v in gate.items()}
    return info
