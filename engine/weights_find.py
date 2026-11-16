# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights already downloaded by ComfyUI nodes, found automatically (user request 2026-11-10: „żeby szukał
automatycznie te, które są już pobrane w Comfy”).

Some ComfyUI nodes keep Hugging Face models as plain folders in ComfyUI's models directory – on the workstation
models/vitmatte/vitmatte-base-composition-1k (config.json + pytorch_model.bin) – instead of the HF cache MatteDesk
reads. `local_dir(model_id)` finds such a folder (models_dir/<anything>/<model name>, also one level deeper) with a
config.json and a weights file; `resolve(model_id)` gives the path to load (transformers' from_pretrained takes a
folder) or the id itself (then the HF cache / a download). The stage gate (stage_models.hf_cached) counts these too.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

WEIGHT_FILES = ("*.safetensors", "*.bin", "*.pth", "*.pt")


def _models_dir() -> Optional[Path]:
    try:
        from engine.config import load_config
        md = load_config().get("models_dir")
    except Exception:  # noqa: BLE001
        return None
    return Path(md) if md and Path(md).is_dir() else None


def _complete(d: Path) -> bool:
    return (d / "config.json").is_file() and any(f.stat().st_size > 1_000_000 for pat in WEIGHT_FILES for f in d.glob(pat))


@lru_cache(maxsize=64)
def _find(models_dir: str, name: str) -> Optional[str]:
    root = Path(models_dir)
    for pat in (f"*/{name}", f"*/*/{name}"):
        for d in sorted(root.glob(pat)):
            if d.is_dir() and _complete(d):
                return str(d).replace("\\", "/")
    return None


def local_dir(model_id: str, models_dir: Optional[Path] = None) -> Optional[Path]:
    """A complete plain folder of `model_id` (e.g. hustvl/vitmatte-base-composition-1k) in ComfyUI's models; None."""
    md = models_dir or _models_dir()
    if not md or not model_id:
        return None
    hit = _find(str(md), model_id.rstrip("/").split("/")[-1])
    if hit and not _complete(Path(hit)):      # removed since the last look
        _find.cache_clear()
        hit = _find(str(md), model_id.rstrip("/").split("/")[-1])
    return Path(hit) if hit else None


def resolve(model_id: str) -> str:
    """What to give from_pretrained: a local folder found in ComfyUI's models, else the id (HF cache / download)."""
    d = local_dir(model_id)
    return str(d) if d else model_id
