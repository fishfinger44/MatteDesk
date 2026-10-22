# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/ltx_alpha.py - stage "LTX-2.5 Alpha-Gen" (user decision 2026-10-16: its own stage before the export, taken out of
the experiments' edge modes).

LTX-2.5 22B + IC-LoRA Alpha-Gen (Lightricks) turns a plain RGB video into a grey alpha matte. The model takes NO mask
and no prompt – it picks the foreground itself. The masks of the tool (SAM 3.1 core + brush = mask 1+2) steer it the way
dughogan/nuke-alphagen-ofx (MIT) does ("isolate then matte"):
  guide "plain"   – the crop of the plate-to-mask as it is;
  guide "isolate" – the OTHER objects of the project painted out of the crop (fill from the plate around them, solved
                    without the target and the others, so the fill never looks like either of them);
  guide "steer"   – isolate + everything outside the (grown) object defocused – for objects that are not the
                    foreground of the crop (the model mattes the sharp, near thing).
The grey matte comes back as 16-bit PNG (SaveImageAdvanced, downloaded over /view – also works with a remote Comfy),
is limited to the object's region (mask 1+2 grown by `region_px`, soft edge `region_soft_px`) and optionally joined
with the eroded mask 1+2 (`combine: max_core` – holes of the generated matte inside the object filled).

Graph = the OFX `addAlphaGen` recipe on core nodes (Comfy >= 0.39): UNETLoader (distilled int8) -> LoraLoaderModelOnly
(Alpha-Gen 1.0) -> GetICLoRAParameters; CLIPLoader ltxv, prompt "" / negative " "; LTXVConditioning 24 fps;
EmptyLTXVLatentVideo; LTXVAddGuide (the guide frames, frame_idx 0, strength 1); empty audio latent; CFGGuider cfg 1;
euler_ancestral with the 8 distilled sigmas; LTXVCropGuides; VAEDecodeTiled. Batches of 8k+1 frames (<= 145, model
card), crossfaded over the overlap; the guide scaled down only (long side <= max_side, short <= 1088 – model limit
1920 x 1088) and mirror-padded to multiples of 32.
Measured on SH050 (RTX 4090 24 GB, 128 GB RAM, Comfy 0.40, int8 transformer + int8 Gemma, 2026-10-16): 97 frames x
1280 x 736 in one batch = 1.14 s / frame, 22.5 GB VRAM peak (dynamic loading), ~33 GB RAM; first batch +~45 s (load).
Quality (local_results/ltx_alpha/RAPORT.md): best on real hair / motion blur / defocus with a clear foreground (CLIP2_A,
CLIP3_A – finer strands than MatAnyone2, less flicker); on SH010 (puppet in front of a textured set) plain picks the whole
set – "steer" is needed and the edge stays soft; "isolate" fails for occluders lying ON the target.
"""

from __future__ import annotations

import hashlib
import shutil
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import re

import cv2
import numpy as np

# model files (Lightricks/LTX-2.5 and Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen, both gated on Hugging Face: the user
# accepts the LTX-2.x Community License with their account; the Doctor downloads with their HF login)
LTX_MODELS = {      # small files first (an interrupted install still leaves the quick ones)
    "lora": ("loras", "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors", "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen", 1.31),
    "video_vae": ("vae", "ltx-2.5-video-vae-bf16.safetensors", "Lightricks/LTX-2.5", 1.47),
    "audio_vae": ("vae", "ltx-2.5-audio-vae-bf16.safetensors", "Lightricks/LTX-2.5", 0.36),
    "text_encoder": ("text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "Lightricks/LTX-2.5", 15.4),
    "unet": ("diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "Lightricks/LTX-2.5", 21.5),
}
# path of a file inside its HF repo (the base repo keeps the files in folders named like Comfy's)
HF_PATH = {"unet": "diffusion_models/", "text_encoder": "text_encoders/", "video_vae": "vae/", "audio_vae": "vae/", "lora": ""}
LTX_NODES = ("UNETLoader", "LoraLoaderModelOnly", "GetICLoRAParameters", "CLIPLoader", "LTXVConditioning",
             "EmptyLTXVLatentVideo", "LTXVAddGuide", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "CFGGuider",
             "ManualSigmas", "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVCropGuides", "VAEDecodeTiled",
             "SaveImageAdvanced", "PlateLoadEXR")
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
GUIDES = ("plain", "isolate", "steer")
COMBINES = ("region", "max_core", "raw", "band")   # band = the one band system (engine/bandsys.py, 2026-10-22)
# which mask limits / guides LTX (user request 2026-10-16): the mask 1+2 (default), or the result of stage 3 / 4 (> 0.5)
MASK_SOURCES = ("core", "matanyone", "vitmatte")
LTX_DEFAULTS: Dict[str, Any] = {
    "guide": "plain",         # plain | isolate | steer (isolate fails for occluders ON the target: the painted blob is matted as foreground – CLIP2_A 2026-10-16)
    "max_side": 1280,         # long side of the LTX input (<= 1920, short side <= 1088); scaled DOWN only, then padded to %32
    "max_frames": 97,         # batch length (8k+1, <= 145 – model card); 97 x 1280x736 = 22.5 GB VRAM peak on a 4090 (dynamic loading)
    "overlap": 16,            # batch overlap, linear crossfade (8 left a visible seam, CLIP2_A 2026-10-16)
    "seed": 1234,
    "region_px": 24,          # the matte is kept inside the mask 1+2 grown by this (plate px) ...
    "region_soft_px": 12,     # ... with this soft edge
    "combine": "region",      # region | max_core (+ eroded mask 1+2: holes filled) | raw (no limit – diagnostics)
    "core_erode_px": 4,
    "mask_source": "core",    # core = mask 1+2 | matanyone = stage 3 result | vitmatte = stage 4 result (alpha > 0.5)
    # combine "band" (one band system, user decision 2026-10-22): the band region_px outside .. core_erode_px inside the
    # mask edge; the matte blends into the core over feather_px from the inner edge and region_soft_px from the outer
    # one; band points shape it (the sliders' reference band_ref_out / _in)
    "feather_px": 0.0,
    "band_points": [],
    "band_ref_out": None,
    "band_ref_in": None,
}
SETTING_KEYS = tuple(LTX_DEFAULTS)
BAND_KEYS = ("feather_px", "band_points", "band_ref_out", "band_ref_in")   # in the signature only in the band mode


def ltx_settings(obj: Optional[Dict[str, Any]] = None, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """LTX_DEFAULTS <- defaults.json "ltx" <- object field `ltx` (only known keys, validated)."""
    s = dict(LTX_DEFAULTS)
    for src in (defaults or {}, (obj or {}).get("ltx") or {}):
        s.update({k: v for k, v in src.items() if k in LTX_DEFAULTS and v is not None})
    s["guide"] = s["guide"] if s["guide"] in GUIDES else "plain"
    s["combine"] = s["combine"] if s["combine"] in COMBINES else "region"
    ms = str(s["mask_source"])
    # a stage name, or a version of stage Eksperymenty ("v###", 2026-10-16 – origin kept in run.json)
    s["mask_source"] = ms if (ms in MASK_SOURCES or re.fullmatch(r"v\d{3,}", ms)) else "core"
    s["max_side"] = int(min(1920, max(256, int(s["max_side"]) // 32 * 32)))   # long side; short side <= 1088 (work_size)
    s["max_frames"] = int(min(145, max(9, (int(s["max_frames"]) - 1) // 8 * 8 + 1)))
    s["overlap"] = int(max(0, min(int(s["overlap"]), s["max_frames"] // 2)))
    for k in ("seed", "core_erode_px"):
        s[k] = int(s[k])
    for k in ("region_px", "region_soft_px"):
        s[k] = float(s[k])
    s["feather_px"] = float(min(200.0, max(0.0, float(s.get("feather_px") or 0.0))))
    from engine.bandmap import band_points
    s["band_points"] = band_points({"band_points": s.get("band_points")}) if s.get("band_points") else []
    from engine.objband import apply   # the shared object band unless the stage is detached (2026-10-22)
    return apply(obj, "ltx", s)


def object_settings(obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """ltx_settings with defaults.json "ltx" (the project-wide defaults) under the object's own `ltx`."""
    try:
        d = json.loads((Path(__file__).resolve().parent.parent / "defaults.json").read_text(encoding="utf-8")).get("ltx")
    except (OSError, ValueError):
        d = None
    return ltx_settings(obj, d or {})


def settings_sig(s: Dict[str, Any]) -> str:
    d = {k: s.get(k) for k in SETTING_KEYS if k not in BAND_KEYS or s.get("combine") == "band"}
    if d.get("mask_source", "core") == "core":   # the default: absent from the signature (older runs stay current)
        d.pop("mask_source", None)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def band_u8(mask: np.ndarray, s: Dict[str, Any], pts=None) -> np.ndarray:
    """Band preview of the LTX-2.5 palette: 128 = where the LTX-2.5 matte is kept (the mask grown by region_px), 255 = forced 1
    (combine max_core: the mask eroded by core_erode_px); raw = the whole frame 128."""
    if s["combine"] == "raw":
        return np.full(mask.shape, 128, np.uint8)
    d = sdf_px(mask)
    out = np.zeros(mask.shape, np.uint8)
    if s["combine"] == "band":
        from engine.bandsys import band_widths
        o, i = band_widths(mask, s["region_px"], s["core_erode_px"], pts)
        out[(d >= -o) & (d <= i)] = 128
        out[d > i] = 255
        return out
    out[d >= -s["region_px"]] = 128
    if s["combine"] == "max_core":
        out[d >= s["core_erode_px"]] = 255
    return out


def models_status(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Every LTX-2.5 model file: {key, folder, file, repo, gb, present}."""
    out = []
    for key, (folder, fn, repo, gb) in LTX_MODELS.items():
        p = Path(models_dir) / folder / fn if models_dir else None
        out.append({"key": key, "folder": folder, "file": fn, "repo": repo, "gb": gb,
                    "present": bool(p and p.exists() and p.stat().st_size > 0.9 * gb * 1e9)})
    return out


def missing_nodes(object_info: Optional[Dict[str, Any]]) -> List[str]:
    return [n for n in LTX_NODES if not object_info or n not in object_info]


# ----------------------------------------------------------------------------- batching
def batch_length(n: int) -> int:
    """Frames LTX-2.5 gets for a batch of n: the next 8k+1 (the last frame repeated)."""
    return max(9, ((max(1, n) - 1 + 7) // 8) * 8 + 1)


def batches(n: int, max_len: int = 49, overlap: int = 8) -> List[Tuple[int, int]]:
    """Index ranges [a, b] covering 0..n-1, each <= max_len frames, consecutive ones overlapping by `overlap`.
    Lengths need not be 8k+1 – `batch_length` pads each to it."""
    if n <= 0:
        return []
    if n <= max_len:
        return [(0, n - 1)]
    k = -(-(n - overlap) // (max_len - overlap))           # batches needed
    total = n + (k - 1) * overlap                          # even lengths (differ by <= 1, no short tail batch)
    out, a = [], 0
    for i in range(k):
        ln = total // k + (1 if i < total % k else 0)
        out.append((a, a + ln - 1))
        a += ln - overlap
    return out


def crossfade_weight(i: int, n: int, first: bool, last: bool, overlap: int) -> float:
    """Weight of frame i of an n-frame batch: linear ramp over `overlap` frames at a batch end that meets another
    batch (never at the shot's own ends)."""
    if overlap <= 0:
        return 1.0
    w = 1.0
    if not first:
        w = min(w, (i + 1) / (overlap + 1))
    if not last:
        w = min(w, (n - i) / (overlap + 1))
    return max(w, 1e-3)


def work_size(w: int, h: int, max_side: int = 1024) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """((sw, sh), (pw, ph)): the guide scaled DOWN only (long side <= max_side, short side <= 1088 – model limit
    1920 x 1088), then padded (mirror, model card: padding, not stretching) up to multiples of 32."""
    sc = min(1.0, float(max_side) / max(w, h), 1088.0 / min(w, h))
    sw, sh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    return (sw, sh), (-(-sw // 32) * 32, -(-sh // 32) * 32)


# ----------------------------------------------------------------------------- guide (isolate / steer)
def _grow(m: np.ndarray, px: int) -> np.ndarray:
    m = m.astype(np.float32)
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def isolate_guide(img: np.ndarray, sel: np.ndarray, others: Optional[np.ndarray], steer: bool = False) -> np.ndarray:
    """numpy port of the OFX `isolatedGuide`: the other objects (grown a little, never over the target) painted with a
    fill solved at 1/4 resolution from the plate with the target AND the others removed; `steer` also defocuses
    everything outside the grown target. img: float32 HxWx3 0..1; sel / others: masks HxW (0..1)."""
    H, W = img.shape[:2]
    sel = np.clip(sel.astype(np.float32), 0, 1)
    oth = np.zeros_like(sel) if others is None else np.clip(others.astype(np.float32), 0, 1)
    paint = np.clip(_grow(oth, max(2, W // 320)) * (1 - sel), 0, 1)
    if paint.max() <= 0 and not steer:
        return img.astype(np.float32)
    k = 4
    w, h = max(1, W // k), max(1, H // k)
    small = cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    paint_s = cv2.resize(_grow(paint, 2), (w, h), interpolation=cv2.INTER_AREA)
    sel_s = cv2.resize(_grow(sel, 2), (w, h), interpolation=cv2.INTER_AREA)
    unknown = ((paint_s > 0.02) | (sel_s > 0.02)).astype(np.uint8) * 255
    if unknown.all():
        fill = np.repeat(small.reshape(-1, 3).mean(0)[None, None], h, 0).repeat(w, 1).astype(np.float32)
    else:
        src = np.clip(small * 65535.0, 0, 65535).astype(np.uint16)
        # cv2.inpaint takes 8-bit 3-channel only: inpaint each channel as 16-bit 1-channel (Telea accepts CV_16UC1)
        fill = np.dstack([cv2.inpaint(src[..., c], unknown, 5, cv2.INPAINT_TELEA) for c in range(3)]).astype(np.float32) / 65535.0
    fill = _blur(fill, 1.5)
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    hm = (_blur(paint, 1.0) * (1 - sel))[..., None]
    out = img.astype(np.float32) * (1 - hm) + up(fill) * hm
    if steer:
        base = np.where((paint_s > 0.02)[..., None], fill, small)
        sel_g = cv2.resize(_grow(sel, 6), (w, h), interpolation=cv2.INTER_AREA)
        den = 1.0 - np.minimum(1.0, sel_g * 2.0)
        num = _blur(base * den[..., None], 10.0)
        den_b = _blur(den, 10.0)
        defocus = np.where(den_b[..., None] > 1e-3, num / np.maximum(den_b[..., None], 1e-3), base)
        om = (1.0 - _blur(_grow(sel, 10), 4.0))[..., None]
        out = out * (1 - om) + up(defocus.astype(np.float32)) * om
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- result -> object alpha
def sdf_px(core: np.ndarray) -> np.ndarray:
    """Signed distance (px), > 0 inside the mask."""
    m = (core > 0.5).astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return (inside - outside).astype(np.float32)


def finish(alpha: np.ndarray, core: np.ndarray, s: Dict[str, Any], pts=None) -> np.ndarray:
    """LTX-2.5 matte (crop) -> the object's alpha: limited to the grown mask 1+2 (`region`), optionally joined with the
    eroded mask 1+2 (`max_core`); `raw` = untouched; `band` = the one band system (engine/bandsys.py; band points (x, y,
    out, in) in the crop's pixels shape it)."""
    a = np.clip(alpha.astype(np.float32), 0, 1)
    if s["combine"] == "raw":
        return a
    d = sdf_px(core)
    if s["combine"] == "band":
        from engine.bandsys import band_widths, compose
        c = (np.asarray(core) > 0.5).astype(np.float32)
        o, i = band_widths(c, s["region_px"], s["core_erode_px"], pts)
        return compose(c, a, d, o, i, s.get("feather_px", 0.0), s["region_soft_px"])
    reg = np.clip((d + s["region_px"]) / max(s["region_soft_px"], 1e-3) + 0.5, 0.0, 1.0)
    a = a * reg
    if s["combine"] == "max_core":
        a = np.maximum(a, (d >= s["core_erode_px"]).astype(np.float32))
    return a.astype(np.float32)


# ----------------------------------------------------------------------------- Comfy graph
def alphagen_graph(load_pattern: str, n_frames: int, side_w: int, side_h: int, out_prefix: str, seed: int = 1234,
                   models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """API graph: PlateLoadEXR (guide frames 0..n-1, display 0..1) -> Alpha-Gen (OFX recipe) -> 16-bit PNG."""
    m = {k: v[1] for k, v in LTX_MODELS.items()}
    m.update(models or {})
    L = lambda nid, slot=0: [str(nid), slot]
    g: Dict[str, Any] = {
        "2": {"class_type": "PlateLoadEXR", "inputs": {"path_pattern": load_pattern, "first_frame": 0,
                                                        "last_frame": n_frames - 1, "crop_x": 0, "crop_y": 0,
                                                        "crop_w": 0, "crop_h": 0}},
        "30": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "31": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": L(30), "lora_name": m["lora"], "strength_model": 1.0}},
        "51": {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": L(31)}},
        "32": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "ltxv", "device": "default"}},
        "33": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "34": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "35": {"class_type": "CLIPTextEncode", "inputs": {"clip": L(32), "text": ""}},
        "36": {"class_type": "CLIPTextEncode", "inputs": {"clip": L(32), "text": " "}},   # "" negative crashes the tokenizer
        "37": {"class_type": "LTXVConditioning", "inputs": {"positive": L(35), "negative": L(36), "frame_rate": 24.0}},
        "38": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": int(side_w), "height": int(side_h),
                                                                "length": int(n_frames), "batch_size": 1}},
        "39": {"class_type": "LTXVAddGuide", "inputs": {"positive": L(37, 0), "negative": L(37, 1), "vae": L(33),
                                                        "latent": L(38), "image": L(2, 0), "frame_idx": 0, "strength": 1.0,
                                                        "iclora_parameters": L(51)}},
        "40": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": int(n_frames), "frame_rate": 24,
                                                                "batch_size": 1, "audio_vae": L(34)}},
        "41": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": L(39, 2), "audio_latent": L(40)}},
        "42": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "43": {"class_type": "CFGGuider", "inputs": {"model": L(31), "positive": L(39, 0), "negative": L(39, 1), "cfg": 1.0}},
        "44": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "45": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "46": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": L(42), "guider": L(43), "sampler": L(44),
                                                                 "sigmas": L(45), "latent_image": L(41)}},
        "47": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": L(46)}},
        "48": {"class_type": "LTXVCropGuides", "inputs": {"positive": L(39, 0), "negative": L(39, 1), "latent": L(47, 0)}},
        "49": {"class_type": "VAEDecodeTiled", "inputs": {"samples": L(48, 2), "vae": L(33), "tile_size": 768,
                                                          "overlap": 128, "temporal_size": 128, "temporal_overlap": 16}},
        "50": {"class_type": "SaveImageAdvanced", "inputs": {"images": L(49), "filename_prefix": out_prefix,
                                                             "format": "png", "format.bit_depth": "16-bit",
                                                             "format.input_color_space": "sRGB"}},
    }
    return g


def decode_matte(png_bytes: bytes) -> np.ndarray:
    """16-bit (or 8-bit) grey / RGB PNG -> float32 0..1 (mean of the channels: the model's matte is grey)."""
    a = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError("LTX-2.5: nieczytelny PNG wyniku")
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    a = a.astype(np.float32) / scale
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


class AlphaGen:
    """Runs Alpha-Gen on one crop segment. `client` = ComfyClient (queue_prompt / wait_for_prompt / download_output)."""

    def __init__(self, client: Any, io_dir: Path, settings: Dict[str, Any], models: Optional[Dict[str, str]] = None,
                 progress: Optional[Callable[[str, int, int], None]] = None, cancelled: Optional[Callable[[], bool]] = None):
        self.client = client
        self.io = Path(io_dir)
        self.s = settings
        self.models = models
        self.progress = progress or (lambda *a: None)
        self.log: List[Dict[str, Any]] = []

    def _run(self, g: Dict[str, Any]) -> Dict[str, Any]:
        busy = self.client.busy_error()
        if busy:
            raise RuntimeError(busy)
        sub = self.client.queue_prompt(g)
        if not sub.get("success"):
            raise RuntimeError(f"Comfy odrzucił graf LTX-2.5 Alpha-Gen: {sub.get('error')}")
        res = self.client.wait_for_prompt(sub["prompt_id"], timeout_sec=4 * 3600)
        if not res.get("success"):
            raise RuntimeError(f"LTX-2.5 Alpha-Gen nie powiódł się: {res.get('error')}")
        return res

    def run(self, tag: str, guides: Sequence[np.ndarray], out_size: Tuple[int, int]) -> List[np.ndarray]:
        """guides: display crops (float32 HxWx3 BGR 0..1, one per frame) -> mattes at `out_size` (w, h), same order."""
        import time
        n = len(guides)
        H, W = guides[0].shape[:2]
        (sw, sh), (side_w, side_h) = work_size(W, H, self.s["max_side"])
        acc = [np.zeros((out_size[1], out_size[0]), np.float32) for _ in range(n)]
        wsum = [0.0] * n
        bl = batches(n, self.s["max_frames"], self.s["overlap"])
        for bi, (a, b) in enumerate(bl):
            nb = b - a + 1
            length = batch_length(nb)
            job = (self.io / f"ltx_{tag}_{bi:02d}").resolve()   # Comfy reads it: absolute path (other cwd)
            job.mkdir(parents=True, exist_ok=True)
            for i in range(length):
                img = guides[a + min(i, nb - 1)]
                if (sw, sh) != (W, H):
                    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                img = cv2.copyMakeBorder(img, 0, side_h - sh, 0, side_w - sw, cv2.BORDER_REFLECT_101)
                cv2.imwrite(str(job / f"g_{i:04d}.exr"), np.ascontiguousarray(img, np.float32))
            self.progress(f"LTX-2.5: paczka {bi + 1}/{len(bl)} ({nb} kl., {side_w}×{side_h})", bi, len(bl))
            t0 = time.time()
            prefix = f"matte_desk_ltx/{tag}_{bi:02d}_{int(t0)}"
            res = self._run(alphagen_graph(str(job / "g_####.exr").replace("\\", "/"), length, side_w, side_h,
                                           prefix, self.s["seed"], self.models))
            files = sorted(res.get("outputs", {}).get("50", {}).get("images", []), key=lambda i: i["filename"])
            if len(files) < nb:
                raise RuntimeError(f"LTX-2.5: paczka {bi + 1} zwróciła {len(files)} klatek z {nb}")
            self.log.append({"batch": bi, "frames": [a, b], "length": length, "side": [side_w, side_h],
                             "seconds": round(time.time() - t0, 1)})
            for i in range(nb):
                m = decode_matte(self.client.download_output(files[i]))[:sh, :sw]   # padding off
                m = cv2.resize(m, out_size, interpolation=cv2.INTER_LINEAR)
                wgt = crossfade_weight(i, nb, bi == 0, bi == len(bl) - 1, self.s["overlap"])
                acc[a + i] += m * wgt
                wsum[a + i] += wgt
            # the batch's guide EXRs (up to ~1 GB at 97 x 1280 x 736 float32) are only Comfy's input: removed after a
            # successful batch (a failed one keeps them for diagnosis)
            shutil.rmtree(job, ignore_errors=True)
        return [acc[i] / max(wsum[i], 1e-6) for i in range(n)]
