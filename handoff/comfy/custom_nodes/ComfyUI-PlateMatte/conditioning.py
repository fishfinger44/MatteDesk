# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""A positive / negative CONDITIONING pair saved to one .safetensors file and loaded back (2026-11-06).

LTX-2.5 Alpha-Gen always runs on the empty prompt ("" / " "), so the text encoder (Gemma int8, 15.4 GB) only has to
run once: the pair is saved, later graphs load the file instead of CLIPLoader + CLIPTextEncode. The tensors are written
exactly as the encoder returned them (dtype and values unchanged); non-tensor entries of the conditioning dicts
(e.g. `unprocessed_ltxav_embeds: True`) go to the file's metadata as JSON. No pickle: safetensors only.
"""
import json
import os

import torch
from safetensors.torch import load_file, save_file

FORMAT = "platematte.conditioning_pair.v1"


def _pack(cond, prefix, tensors, meta):
    entries = []
    for i, (t, extra) in enumerate(cond):
        tensors[f"{prefix}.{i}.cond"] = t.detach().to("cpu").contiguous()
        plain, tkeys, none_keys = {}, [], []
        for k, v in (extra or {}).items():
            if isinstance(v, torch.Tensor):
                tensors[f"{prefix}.{i}.{k}"] = v.detach().to("cpu").contiguous()
                tkeys.append(k)
            elif v is None:
                none_keys.append(k)
            else:
                try:
                    json.dumps(v)
                except TypeError:
                    raise ValueError(f"conditioning entry {k!r} ({type(v).__name__}) cannot be saved")
                plain[k] = v
        entries.append({"tensors": tkeys, "none": none_keys, "plain": plain})
    meta[prefix] = entries


def _unpack(prefix, tensors, meta):
    out = []
    for i, e in enumerate(meta[prefix]):
        extra = dict(e["plain"])
        extra.update({k: None for k in e["none"]})
        extra.update({k: tensors[f"{prefix}.{i}.{k}"] for k in e["tensors"]})
        out.append([tensors[f"{prefix}.{i}.cond"], extra])
    return out


def save_pair(path, positive, negative):
    tensors, meta = {}, {}
    _pack(positive, "positive", tensors, meta)
    _pack(negative, "negative", tensors, meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={"format": FORMAT, "entries": json.dumps(meta)})
    os.replace(tmp, path)


def load_pair(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
    if md.get("format") != FORMAT:
        raise ValueError(f"{path}: not a PlateMatte conditioning pair (format {md.get('format')!r})")
    meta = json.loads(md["entries"])
    tensors = load_file(path)
    return _unpack("positive", tensors, meta), _unpack("negative", tensors, meta)


class ConditioningPairSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "path": ("STRING", {"default": "C:/models/ltx_conditioning/empty_prompt.safetensors"}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "PlateMatte"
    DESCRIPTION = "Saves a positive / negative conditioning pair (e.g. an encoded empty prompt) to one .safetensors file."

    def save(self, positive, negative, path):
        save_pair(path, positive, negative)
        return {}


class ConditioningPairLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "C:/models/ltx_conditioning/empty_prompt.safetensors"})}}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "load"
    CATEGORY = "PlateMatte"
    DESCRIPTION = "Loads a conditioning pair saved by Conditioning Pair Save – no text encoder needed."

    @classmethod
    def IS_CHANGED(cls, path):
        try:
            st = os.stat(path)
            return f"{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return float("nan")

    def load(self, path):
        return load_pair(path)
