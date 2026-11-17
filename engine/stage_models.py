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
          "matanyone": "„Zainstaluj” przy „Model: MatAnyone2”, „ComfyUI-MatAnyone2” i „Węzły MatteDesk (pliki)”",
          "videomama": "„Zainstaluj VideoMaMa”",
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


# node packs the MatAnyone2 graph needs (LoadMatAnyone2Model / MatAnyone2RunKeyframes): weights alone are not enough
# (MatteDesk-dev issue 12: the weights were there, the nodes were not – the run failed in ComfyUI)
MATANYONE_NODE_PACKS = ("ComfyUI-MatAnyone2", "ComfyUI-MatAnyone2-Keyframes")


def _pack_key(name: str) -> str:
    return name.lower().replace("_", "-")


def _missing_packs(cfg: Dict[str, Any], packs) -> List[str]:
    """Node packs of `packs` not in custom_nodes (name compared case- and -/_-insensitively, as ComfyUI-Manager may
    name the folder); [] when comfy_dir is unknown (no verdict – ComfyUI's own rejection then names the node)."""
    from engine.config import safe_is_dir
    cn = Path(cfg.get("comfy_dir") or "") / "custom_nodes"
    if not cfg.get("comfy_dir") or not safe_is_dir(cn):
        return []
    try:
        have = {_pack_key(p.name) for p in cn.iterdir() if p.is_dir()}
    except OSError:
        return []
    return [p for p in packs if _pack_key(p) not in have]


def _matanyone(cfg: Dict[str, Any]) -> List[str]:
    md = _md(cfg)
    miss = [] if md and (md / "matanyone2" / "matanyone2.pth").exists() else ["matanyone2/matanyone2.pth"]
    return [f"węzeł {p}" for p in _missing_packs(cfg, MATANYONE_NODE_PACKS)] + miss


def hf_cached(model_id: str, models_dir: Any = None) -> bool:
    """A model in the Hugging Face cache (HF_HOME / HF_HUB_CACHE or ~/.cache/huggingface) with a snapshot – or, given
    ComfyUI's `models_dir`, a plain folder of it a ComfyUI node already downloaded there (engine/weights_find.py)."""
    roots = [os.environ.get("HF_HUB_CACHE"), (Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             Path.home() / ".cache" / "huggingface" / "hub"]
    folder = "models--" + model_id.replace("/", "--")
    for r in roots:
        if r and (Path(r) / folder / "snapshots").is_dir() and any((Path(r) / folder / "snapshots").iterdir()):
            return True
    if not models_dir or not Path(models_dir).is_dir():
        return False
    # a plain folder of the model already downloaded by a ComfyUI node (engine/weights_find.py, 2026-11-10)
    from engine.weights_find import local_dir
    return local_dir(model_id, Path(models_dir)) is not None


def _vitmatte(cfg: Dict[str, Any]) -> List[str]:
    """Stage 5 needs one edge model of the zone engine: ViTMatte or (since 2026-10-21) BiRefNet_HR-matting / Lucida."""
    from engine.birefnet import MODELS as BIREFNET
    from engine.vitmatte import DEFAULT_MODEL
    ids = [DEFAULT_MODEL, *BIREFNET.values()]
    md = (cfg or {}).get("models_dir")
    return [] if any(hf_cached(m, md) for m in ids) else [f"{' / '.join(ids)} (cache Hugging Face / folder modeli ComfyUI)"]


EDGE_ENGINES = ("vitmatte", "birefnet")   # stage 5 engine (object field `edge_engine`, user decision 2026-10-21)
EDGE_ENGINE_LABEL = {"vitmatte": "ViTMatte", "birefnet": "BiRefNet"}


def edge_engines(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Which stage-5 edge engines have weights on disk: ViTMatte (any hustvl model) / BiRefNet (BiRefNet_HR-matting or
    Lucida). The UI shows the stage menu only when both are there. Gate off (tests): both."""
    if not enabled():
        return {e: True for e in EDGE_ENGINES}
    from engine.birefnet import MODELS as BIREFNET
    from engine.vitmatte import MODELS as VITMATTE
    md = (cfg or {}).get("models_dir")
    return {"vitmatte": any(hf_cached(m, md) for m in VITMATTE.values()),
            "birefnet": any(hf_cached(m, md) for m in BIREFNET.values())}


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
    from engine.ltx_alpha import missing_models
    return [f"{m['folder']}/{m['file']}" for m in missing_models(_md(cfg))]


CHECKS: Dict[str, Callable[[Dict[str, Any]], List[str]]] = {
    "core": _sam3, "cotracker": _cotracker, "matanyone": _matanyone, "videomama": _videomama, "vitmatte": _vitmatte,
    "ltx": _ltx}


def enabled() -> bool:
    return os.environ.get("MD_MODEL_GATE", "1") != "0"


# --- Cleanup 3D (docs/PLAN_CLEANUP_3D.md, package P3): fill engines of a key ------------------------------------------
# FLUX 3 Image Klein for ComfyUI (gated repo, measured 2026-11-09 on ComfyUI Desktop 0.44.2) + the Restore LoRA (public).
# key -> (folder in models_dir, file, HF repo, path inside the repo, bytes). Small files first (like LTX-2.5).
KLEIN_REPO = "black-forest-labs/flux-3-image-klein-comfy"
KLEIN_PAGE = f"https://huggingface.co/{KLEIN_REPO}"
KLEIN_LORA_REPO = "ausboss/FLUX-3-Image-Klein-Restore-LoRA"
KLEIN_FILES: Dict[str, tuple] = {
    "vae": ("vae", "flux2-vae.safetensors", KLEIN_REPO, "vae/", 336211292),
    "text_encoder": ("text_encoders", "qwen3vl_4b_bf16.safetensors", KLEIN_REPO, "text_encoders/", 8875719408),
    "unet": ("diffusion_models", "flux3-image-klein-bf16.safetensors", KLEIN_REPO, "diffusion_models/", 11097404512),
}
# int8-convrot of the same repo (user decision 2026-11-10: a card under KLEIN_BF16_MIN_GB gets int8 – model 5.8 GB +
# encoder 4.9 GB fit a 12 GB card; bf16 above). Sizes read at the source 2026-11-10. The VAE is shared.
KLEIN_FILES_INT8: Dict[str, tuple] = {
    "vae": KLEIN_FILES["vae"],
    "text_encoder": ("text_encoders", "qwen3vl_4b_int8_convrot.safetensors", KLEIN_REPO, "text_encoders/", 4858559240),
    "unet": ("diffusion_models", "flux3-image-klein-int8-convrot.safetensors", KLEIN_REPO, "diffusion_models/", 5827874779),
}
KLEIN_BF16_MIN_GB = 16.0
KLEIN_VARIANT_LABEL = {"bf16": "bf16 (pełna precyzja)", "int8": "int8 (mniej VRAM)"}
KLEIN_LORA: Dict[str, tuple] = {
    "lora": ("loras", "flux-3-image-klein-restore.safetensors", KLEIN_LORA_REPO, "", 160697064),
}
KLEIN_MIN_COMFY = (0, 41)   # ComfyUI added FLUX 3 Image Klein (CLIPLoader type flux3, ReferenceMultiLatent) in 0.41
KLEIN_LABEL = "FLUX 3 Image Klein"
KLEIN_LORA_LABEL = "LoRA Restore (FLUX 3 Image Klein)"
KLEIN_DOCTOR = "„Zainstaluj FLUX 3 Image Klein”"
KLEIN_LORA_DOCTOR = "„Zainstaluj LoRA Restore (FLUX 3 Image Klein)”"

# Qwen-Image-2.1 (edit mode) for ComfyUI: the files of the test 2026-11-10 (removers_test/remove_test.py, mode q21; sizes
# checked at the source 2026-11-10 = the files on the workstation). Repo Comfy-Org/Qwen-Image-2.1, Qwen Research License
# (non-commercial). Same row format as KLEIN_FILES.
QWEN21_REPO = "Comfy-Org/Qwen-Image-2.1"
QWEN21_PAGE = f"https://huggingface.co/{QWEN21_REPO}"
QWEN21_FILES: Dict[str, tuple] = {
    "vae": ("vae", "qwen_image_2.1_vae_bf16.safetensors", QWEN21_REPO, "vae/", 675509688),
    "text_encoder": ("text_encoders", "qwen3vl_8b_int8_convrot.safetensors", QWEN21_REPO, "text_encoders/", 9350798360),
    "unet": ("diffusion_models", "qwen_image_2.1_int8_convrot.safetensors", QWEN21_REPO, "diffusion_models/", 7256783064),
}
QWEN21_LABEL = "Qwen-Image-2.1"
QWEN21_DOCTOR = "„Zainstaluj Qwen-Image-2.1”"


def _status(table: Dict[str, tuple], models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    out = []
    for key, (folder, fn, repo, sub, size) in table.items():
        p = Path(models_dir) / folder / fn if models_dir else None
        out.append({"key": key, "folder": folder, "file": fn, "repo": repo, "bytes": size, "gb": size / 1e9,
                    "url": f"https://huggingface.co/{repo}/resolve/main/{sub}{fn}",
                    "present": bool(p and p.is_file() and p.stat().st_size > 0.9 * size)})
    return out


# ComfyUI node classes an engine needs besides its files (user report 2026-11-10: Qwen-Image-2.1 on a ComfyUI 0.33 git
# install – the files were there, TextEncodeQwenImage21 was not; the run failed in ComfyUI). Checked through /object_info
# when ComfyUI answers (comfy_gate); the file gates above stay disk-only.
COMFY_NODES: Dict[str, List[str]] = {
    "qwen21_edit": ["TextEncodeQwenImage21"],
    "h3": ["MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift", "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
           "EmptyMiniMaxH3LatentAV", "SetLatentNoiseMask", "SamplerCustomAdvanced"],
}
COMFY_UPDATE_HINT = "zaktualizuj ComfyUI (Desktop: aktualizacja w aplikacji; z gita: git pull + restart)"


def comfy_gate(client: Any, engine: str) -> str:
    """'' = the engine's ComfyUI nodes are there (or ComfyUI does not answer: no verdict); else the Polish reason."""
    names = COMFY_NODES.get(engine) or []
    if not names:
        return ""
    from engine.comfy_io import missing_nodes
    miss = missing_nodes(client, names)
    if not miss:
        return ""
    return f"Twój ComfyUI nie ma węzła {', '.join(miss)} – {COMFY_UPDATE_HINT}"


def klein_tables() -> Dict[str, Dict[str, tuple]]:
    return {"bf16": KLEIN_FILES, "int8": KLEIN_FILES_INT8}   # read at call time (tests replace the tables)


def klein_preferred(card_gb: Optional[float] = None) -> str:
    """The variant for this machine: int8 on a card under KLEIN_BF16_MIN_GB, else bf16 (unknown card = bf16).
    MD_KLEIN_VARIANT forces one (tests)."""
    forced = os.environ.get("MD_KLEIN_VARIANT", "")
    if forced in ("bf16", "int8"):
        return forced
    if card_gb is None:
        try:
            from engine.vram_fit import card_gb as _card
            card_gb = _card()
        except Exception:  # noqa: BLE001
            card_gb = None
    from engine.vram_fit import TOL_GB
    return "int8" if card_gb and card_gb + TOL_GB < KLEIN_BF16_MIN_GB else "bf16"


def klein_variant(models_dir: Optional[Path], card_gb: Optional[float] = None) -> str:
    """The variant used: the preferred one when complete on disk, else the other one when complete (a project moved
    between a 12 and a 24 GB machine keeps working), else the preferred one (what the install downloads)."""
    pref = klein_preferred(card_gb)
    tables = klein_tables()
    done = [v for v in (pref, "int8" if pref == "bf16" else "bf16") if all(m["present"] for m in _status(tables[v], models_dir))]
    return done[0] if done else pref


def klein_files(models_dir: Optional[Path], card_gb: Optional[float] = None) -> Dict[str, tuple]:
    return klein_tables()[klein_variant(models_dir, card_gb)]


def klein_status(models_dir: Optional[Path], card_gb: Optional[float] = None) -> List[Dict[str, Any]]:
    """The three FLUX 3 Image Klein files of the variant used here (klein_variant): {key, folder, file, repo, bytes,
    gb, url, present}."""
    return _status(klein_files(models_dir, card_gb), models_dir)


def klein_lora_status(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The LoRA Restore file (same fields as klein_status)."""
    return _status(KLEIN_LORA, models_dir)


def qwen21_status(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The three Qwen-Image-2.1 files (same fields as klein_status)."""
    return _status(QWEN21_FILES, models_dir)

def _lama_cleanup(cfg: Dict[str, Any]) -> List[str]:
    md = _md(cfg)
    return [] if md and (md / "inpaint" / "big-lama.pt").exists() else ["inpaint/big-lama.pt"]


def _cleanup_lama(cfg: Dict[str, Any]) -> Dict[str, Any]:
    miss = _lama_cleanup(cfg)
    if not miss:
        return {"ok": True, "reason": "", "missing": []}
    if not _md(cfg):
        return {"ok": False, "missing": miss, "reason": "Brak models_dir w konfiguracji – Doktor (model LaMa niedostępny)"}
    return {"ok": False, "missing": miss,
            "reason": "Brak modelu LaMa: inpaint/big-lama.pt – Doktor → „Zainstaluj” przy „Model: LaMa (big-lama.pt)”"}


def _cleanup_klein(cfg: Dict[str, Any]) -> Dict[str, Any]:
    md = _md(cfg)
    base = [f"{m['folder']}/{m['file']}" for m in klein_status(md) if not m["present"]]
    lora = [f"{m['folder']}/{m['file']}" for m in klein_lora_status(md) if not m["present"]]
    miss = base + lora
    if not miss:
        return {"ok": True, "reason": "", "missing": []}
    if not md:
        return {"ok": False, "missing": miss,
                "reason": f"Brak models_dir w konfiguracji – Doktor (model {KLEIN_LABEL} + LoRA Restore niedostępny)"}
    more = f" (+{len(miss) - 2})" if len(miss) > 2 else ""
    what = (f"{KLEIN_LABEL} + LoRA Restore" if base and lora else KLEIN_LABEL if base else KLEIN_LORA_LABEL)
    buttons = " i ".join(([KLEIN_DOCTOR] if base else []) + ([KLEIN_LORA_DOCTOR] if lora else []))
    return {"ok": False, "missing": miss, "reason": f"Brak modelu {what}: {', '.join(miss[:2])}{more} – Doktor → {buttons}"}


def _cleanup_qwen21(cfg: Dict[str, Any]) -> Dict[str, Any]:
    md = _md(cfg)
    miss = [f"{m['folder']}/{m['file']}" for m in qwen21_status(md) if not m["present"]]
    if not miss:
        return {"ok": True, "reason": "", "missing": []}
    if not md:
        return {"ok": False, "missing": miss,
                "reason": f"Brak models_dir w konfiguracji – Doktor (model {QWEN21_LABEL} niedostępny)"}
    more = f" (+{len(miss) - 2})" if len(miss) > 2 else ""
    return {"ok": False, "missing": miss,
            "reason": f"Brak modelu {QWEN21_LABEL}: {', '.join(miss[:2])}{more} – Doktor → {QWEN21_DOCTOR}"}

CLEANUP3D_CHECKS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "lama": _cleanup_lama, "qwen21_edit": _cleanup_qwen21, "klein_restore": _cleanup_klein}


def cleanup3d_engines(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fill engines of Cleanup 3D (engine.cleanup3d.types.ENGINES) -> {engine: {"ok", "reason", "missing"}}. Disk only
    (no ComfyUI call); the ComfyUI version (>= 0.41 for FLUX 3 Image Klein) is the Doctor's item. Gate off: all ok."""
    out: Dict[str, Dict[str, Any]] = {}
    for eng, check in CLEANUP3D_CHECKS.items():
        if not enabled():
            out[eng] = {"ok": True, "reason": "", "missing": []}
            continue
        try:
            out[eng] = check(cfg)
        except Exception as e:   # a broken check never blocks silently
            out[eng] = {"ok": False, "missing": [], "reason": f"sprawdzenie nie powiodło się: {e}"}
    return out


def cleanup3d_solve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Camera solve of Cleanup 3D (package P10): COLMAP (CUDA) on disk -> {"ok", "reason", "exe"}."""
    if not enabled():
        return {"ok": True, "reason": "", "exe": None}
    from engine.installer import colmap_exe
    exe = colmap_exe(cfg)
    if exe:
        return {"ok": True, "reason": "", "exe": str(exe)}
    return {"ok": False, "exe": None, "reason": "Brak programu COLMAP (CUDA) – Doktor → „Zainstaluj COLMAP (CUDA)”"}


# Cleanup 3D package P12: VOID (own venv on ComfyUI's Python, bin/void) and SeedVR2 (ComfyUI node, ComfyUI's Python),
# both installed by the Doctor (2026-11-10); places = void_engine.tool_paths (config.local.json keys override).
TOOL_LABEL = {"void": "VOID (usuwanie obiektu z wideo)", "seedvr2": "SeedVR2 (podbicie kluczy)"}
TOOL_DOCTOR = {"void": "Doktor → „VOID (usuwanie obiektu z wideo)”", "seedvr2": "Doktor → „SeedVR2 (podbicie kluczy)”"}


def _cleanup3d_tool(cfg: Dict[str, Any], tool: str) -> Dict[str, Any]:
    from engine.cleanup3d.void_engine import TOOL_KEYS, required_files
    rows = required_files(cfg, tool)
    unset = [d for d, p in rows if p is None]
    if unset:
        keys = ", ".join(k for k in TOOL_KEYS[tool] if not cfg.get(k))
        return {"ok": False, "missing": unset,   # SeedVR2 without a configured ComfyUI (its default place is ComfyUI)
                "reason": f"{TOOL_LABEL[tool]} nie jest skonfigurowany ({keys} w config.local.json; najpierw ComfyUI) – "
                          f"{TOOL_DOCTOR[tool]} → Zainstaluj"}
    miss = [f"{d} ({str(p)})" for d, p in rows if not p.exists()]
    if miss:
        more = f" (+{len(miss) - 2})" if len(miss) > 2 else ""
        return {"ok": False, "missing": miss,
                "reason": f"Brak plików {TOOL_LABEL[tool]}: {', '.join(miss[:2])}{more} – {TOOL_DOCTOR[tool]} → Zainstaluj"}
    return {"ok": True, "reason": "", "missing": []}


def cleanup3d_tools(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """VOID / SeedVR2 of Cleanup 3D (package P12) -> {tool: {"ok", "reason", "missing"}}: the configured Python, the
    tool's script and its model files on disk (no subprocess). Gate off (MD_MODEL_GATE=0): all ok."""
    out: Dict[str, Dict[str, Any]] = {}
    for tool in TOOL_LABEL:
        if not enabled():
            out[tool] = {"ok": True, "reason": "", "missing": []}
            continue
        try:
            out[tool] = _cleanup3d_tool(cfg or {}, tool)
        except Exception as e:   # a broken check never blocks silently
            out[tool] = {"ok": False, "missing": [], "reason": f"sprawdzenie nie powiodło się: {e}"}
    return out


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
