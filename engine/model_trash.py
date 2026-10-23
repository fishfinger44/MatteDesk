# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/model_trash.py - Doctor's 🗑: remove the WEIGHTS of an optional model (user decision 2026-10-23).

Only weights – node packs in custom_nodes and pip packages stay. SAM 3.1, CoTracker3 and BiRefNet_HR-matting / Lucida (the
stage-5 edge engine kept when ViTMatte is removed) have no entry here; without MatAnyone2 / ViTMatte their stages are
simply locked by engine/stage_models.py (user decision 2026-10-23). Every path is checked to lie inside
models_dir or the Hugging Face hub cache before anything is removed; the UI lists the files with their size and asks
first. Shared model folders (vae/, loras/ …) are never removed whole – only the files the model downloaded (+ `.part`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def hf_roots() -> List[Path]:
    """The Hugging Face hub caches, as engine.stage_models.hf_cached looks them up."""
    roots = [os.environ.get("HF_HUB_CACHE"), (Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
             Path.home() / ".cache" / "huggingface" / "hub"]
    out: List[Path] = []
    for r in roots:
        if r and Path(r) not in out:
            out.append(Path(r))
    return out


def _hf(model_ids: List[str]) -> List[Path]:
    out = []
    for mid in model_ids:
        folder = "models--" + mid.replace("/", "--")
        out += [r / folder for r in hf_roots() if (r / folder).is_dir()]
    return out


def _md(cfg: Dict[str, Any]) -> Optional[Path]:
    m = cfg.get("models_dir")
    return Path(m) if m else None


def _files(cfg: Dict[str, Any], rels: List[str]) -> List[Path]:
    md = _md(cfg)
    if not md:
        return []
    out = []
    for rel in rels:
        for p in (md / rel, md / (rel + ".part")):
            if p.exists():
                out.append(p)
    return out


def _ltx(cfg):
    from engine.ltx_alpha import LTX_MODELS
    return _files(cfg, [f"{folder}/{fn}" for folder, fn, _repo, _gb in LTX_MODELS.values()])


def _videomama(cfg):
    from engine.videomama import models_root
    root = models_root(_md(cfg))
    return [root] if root and root.is_dir() else []


def _matanyone2(cfg):
    return _files(cfg, ["matanyone2/matanyone2.pth"])


def _vitmatte(cfg):
    from engine.vitmatte import MODELS
    return _hf(list(MODELS.values()))


def _sapiens2(cfg):
    from engine.sapiens2 import DEFAULT_MODEL
    return _hf([DEFAULT_MODEL])


def _minimax(cfg):
    from engine.installer import HF_FOLDERS
    md = _md(cfg)
    p = md / HF_FOLDERS["minimax_weights"][2] if md else None
    return [p] if p and p.is_dir() else []


def _lama(cfg):
    return _files(cfg, ["inpaint/big-lama.pt"])


# key -> (label in the Doctor, what stops working, paths(cfg)); the order = the Doctor's list
MODELS: Dict[str, tuple] = {
    "ltx": ("LTX-2.5 Alpha-Gen", "etap 7 „LTX-2.5”", _ltx),
    "videomama": ("VideoMaMa", "etap 4 „VideoMaMa”", _videomama),
    "matanyone2": ("MatAnyone2", "etap 3 „MatAnyone2” (i silnik MatAnyone2 w Eksperymentach)", _matanyone2),
    "vitmatte": ("ViTMatte", "silnik ViTMatte w etapie 5 (zostaje BiRefNet_HR-matting / Lucida)", _vitmatte),
    "sapiens2": ("Sapiens2-seg-0.4B", "przycisk „Zaproponuj pas (Sapiens2)”", _sapiens2),
    "minimax_weights": ("MiniMax-Remover", "nic (Inpainting jeszcze nieużywany)", _minimax),
    "lama": ("LaMa (big-lama)", "nic (Inpainting jeszcze nieużywany)", _lama),
}


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _allowed(cfg: Dict[str, Any]) -> List[Path]:
    md = _md(cfg)
    return ([md.resolve()] if md else []) + [r.resolve() for r in hf_roots()]


def _inside(p: Path, roots: List[Path]) -> bool:
    rp = p.resolve()
    return any(rp != r and rp.is_relative_to(r) for r in roots)


def listing(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Optional models whose weights are on disk: [{key, label, affects, paths: [{path, bytes}], bytes}]."""
    out = []
    roots = _allowed(cfg)
    for key, (label, affects, paths) in MODELS.items():
        try:
            ps = [p for p in paths(cfg) if _inside(p, roots)]
        except Exception:
            ps = []
        if not ps:
            continue
        items = [{"path": str(p).replace("\\", "/"), "bytes": _size(p)} for p in ps]
        out.append({"key": key, "label": label, "affects": affects, "paths": items, "bytes": sum(i["bytes"] for i in items)})
    return out


def remove(cfg: Dict[str, Any], key: str, rmtree: Callable[[Path], None] = shutil.rmtree) -> Dict[str, Any]:
    """Delete the weights of one optional model. ValueError for an unknown / required key."""
    if key not in MODELS:
        raise ValueError(f"Tego modelu nie można usunąć w Doktorze: {key}")
    label, _affects, paths = MODELS[key]
    roots = _allowed(cfg)
    removed, freed, errors = [], 0, []
    for p in paths(cfg):
        if not _inside(p, roots):
            errors.append(f"pominięte (poza folderem modeli): {p}")
            continue
        size = _size(p)
        try:
            if p.is_dir():
                rmtree(p)
            else:
                p.unlink()
            removed.append(str(p).replace("\\", "/"))
            freed += size
        except OSError as e:
            errors.append(f"{p}: {e}")
    try:   # the Doctor caches "is the model there" per interpreter
        from engine import vitmatte
        vitmatte._CHECK_CACHE.clear()
    except Exception:
        pass
    return {"key": key, "label": label, "removed": removed, "freed_bytes": freed, "errors": errors}
