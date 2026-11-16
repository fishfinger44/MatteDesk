# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""LTX-2.5 as the third Inpaint (C1) engine, next to MiniMax-H3 and Wan 2.2 Fun VACE, with the same Qwen-Image-2.1
anchors (user request 2026-11-12 / 2026-11-13).

Recipe = Lightricks' `LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json` (ComfyUI-LTXVideo, read 2026-11-13), first stage
only and on NATIVE ComfyUI nodes (the LTX-2.5 files of stage 7 + one LoRA, no ComfyUI-LTXVideo pack):
- the in-/outpainting IC-LoRA (`ltx-2.3-22b-ic-lora-in-outpainting-0.9`, trained on 2.3, Lightricks run it on 2.5)
  loaded with LoraLoaderModelOnly + GetICLoRAParameters (reads its reference_downscale_factor), as Alpha-Gen;
- the IC-LoRA guide = the plate clip with the grown mask painted #66FF00 – what their `LTXVInpaintPreprocess` does
  (`images * (1 - mask) + #66FF00 * mask`, always SDR green: the key the LoRA was trained on), done here before upload;
  their `LTXVDilateVideoMask` = our grow_masks;
- the whole frame is generated from an empty latent (8 distilled steps, the sigmas of their stage 1 = Alpha-Gen's);
  their Laplacian-pyramid blend with the plate = the Cleaner pasting the frames only inside its mask (H3Fill);
- anchors (not in Lightricks' graph): each one an ordinary LTXVAddGuide keyframe at its clip frame (a single image
  takes any frame_idx – only guide clips of 9+ frames snap to 8), chained after the IC-LoRA guide; LTXVCropGuides drops
  all guide frames before decoding. Untested by Lightricks: whether the inpaint IC-LoRA works well with keyframes.
Prompt: their note – "it's not an edit tool", a T2V-style description of what is in the area; no prompt = the area is
filled to match the surroundings. Their second stage (latent upscale x2 + 2 steps) is not built: on 12 GB the first
already spills (transformer int8 21.5 GB)."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.h3_fill import H3Fill, grow_masks

LABEL = "LTX-2.5"
LORA = ("loras", "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors", "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
        1_308_778_338)
BASE_KEYS = ("unet", "text_encoder", "video_vae", "audio_vae")   # engine/ltx_alpha.LTX_MODELS (stage 7's files)
NODES = ["UNETLoader", "LoraLoaderModelOnly", "GetICLoRAParameters", "CLIPLoader", "LTXVConditioning",
         "EmptyLTXVLatentVideo", "LTXVAddGuide", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "CFGGuider",
         "ManualSigmas", "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "LTXVCropGuides", "VAEDecodeTiled"]
GREEN_BGR = (0, 255, 102)        # #66FF00 (Lightricks _BG_COLOR_RGB 102, 255, 0)
LONG_EDGE = 1024                 # their stage-2 size (longer side 1024, sides multiples of 64); stage 1 alone here
CANVAS_MULTIPLE = 64
MASK_GROW_PX = 16
MASK_GRID_PX = 32                # the LTX VAE cell (32 px)
MASK_TIME_RADIUS = 2
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"   # their stage 1 (distilled 8 steps)
SAMPLER = "euler_ancestral"
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"            # their negative prompt
SEED = 42
SAVE_NODE = "40"
FPS = 24.0


def _base_files(models_dir: Optional[Path]) -> Dict[str, Optional[str]]:
    from engine.ltx_alpha import LTX_MODELS
    out: Dict[str, Optional[str]] = {}
    for k in BASE_KEYS:
        folder, fn = LTX_MODELS[k][0], LTX_MODELS[k][1]
        out[k] = fn if models_dir and (Path(models_dir) / folder / fn).is_file() else None
    return out


def files(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    md = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    f = _base_files(md)
    f["lora"] = LORA[1] if md and (md / LORA[0] / LORA[1]).is_file() else None
    return f


def status(cfg: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    """{ok, reason, files}: the four LTX-2.5 files of stage 7 + the in-/outpainting IC-LoRA on disk and the native LTX-2.5
    nodes in this ComfyUI (when it answers)."""
    f = files(cfg)
    if not cfg.get("models_dir"):
        return {"ok": False, "reason": "Brak models_dir w konfiguracji – Doktor", "files": f}
    if not all(f[k] for k in BASE_KEYS):
        return {"ok": False, "files": f, "reason": f"Brak plików {LABEL} (etap 7) – Doktor → Zainstaluj LTX-2.5"}
    if not f["lora"]:
        return {"ok": False, "files": f, "reason": f"Brak IC-LoRA inpaintu {LORA[1]} (1,31 GB, Hugging Face {LORA[2]}, "
                                                   "gated) – Doktor → Zainstaluj LoRA inpaintu LTX-2.5"}
    if client is not None:
        from engine.comfy_io import missing_nodes
        m = missing_nodes(client, NODES)
        if m:
            return {"ok": False, "files": f, "reason": f"Twój ComfyUI nie ma węzła {', '.join(m)} – zaktualizuj ComfyUI"}
    return {"ok": True, "reason": "", "files": f}


def canvas(w: int, h: int, long_edge: int = LONG_EDGE) -> Tuple[int, int]:
    """(cw, ch): the longer side = `long_edge` (never up-scaled past the plate), sides multiples of 64."""
    s = min(1.0, float(long_edge) / max(w, h))
    m = CANVAS_MULTIPLE
    return max(m, int(round(w * s / m)) * m), max(m, int(round(h * s / m)) * m)


def clip_length(n: int) -> int:
    """LTX-2.5 clips are 8k + 1 frames (at least 9)."""
    n = max(9, int(n))
    while n % 8 != 1:
        n += 1
    return n


def prompt_for(description: Optional[str], mode: str = "remove", transition: bool = False, morph: bool = False) -> str:
    """A T2V-style description of the area (Lightricks: "prompts describe the scene, not the edit"); "" = fill to match
    the surroundings."""
    d = (description or "").strip().rstrip(".")
    if not d:
        return ""
    if transition:   # morph (2026-11-15): the keyframe guides differ – a scene-style line about the gradual change
        mo = " The pattern slowly and continuously shifts from one keyframe's design to the next, never jumping." if morph else ""
        return f"{d}.{mo} Same light, perspective, colours, grain and camera motion as the rest of the shot."
    if mode == "replace":
        return (f"{d}. This look is there from the first frame to the last and stays the same through the whole shot, "
                "matching the light, perspective, colours, grain and camera motion.")
    return f"{d}. Same light, colours, grain and camera motion; nothing else in the shot."


def green_frames(frames: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    """LTXVInpaintPreprocess: uint8 BGR frames with #66FF00 where the (0 / 255) mask is set."""
    out = []
    g = np.array(GREEN_BGR, np.float32)
    for im, m in zip(frames, masks):
        a = (np.asarray(m, np.float32) / 255.0)[..., None]
        out.append(np.clip(np.rint(np.asarray(im, np.float32) * (1.0 - a) + g * a), 0, 255).astype(np.uint8))
    return out


def graph(guide_dir: Path, prefix: str, prompt: str, w: int, h: int, length: int, f: Dict[str, Optional[str]],
          seed: int = SEED, anchors: Optional[Sequence[Tuple[int, Path]]] = None) -> Dict[str, Any]:
    """The inpaint graph (API format) – see the module docstring. `anchors` = [(clip frame index, folder with ONE image
    at the canvas size)]."""
    from engine.comfy_io import vhs_load
    L = lambda nid, slot=0: [str(nid), slot]
    g: Dict[str, Any] = {
        "10": vhs_load(Path(guide_dir)),
        "30": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "31": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": L(30), "lora_name": f["lora"], "strength_model": 1.0}},
        "51": {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": L(31)}},
        "32": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["text_encoder"], "type": "ltxv", "device": "default"}},
        "33": {"class_type": "VAELoader", "inputs": {"vae_name": f["video_vae"]}},
        "34": {"class_type": "VAELoader", "inputs": {"vae_name": f["audio_vae"]}},
        "35": {"class_type": "CLIPTextEncode", "inputs": {"clip": L(32), "text": prompt or " "}},   # "" crashes the tokenizer
        "36": {"class_type": "CLIPTextEncode", "inputs": {"clip": L(32), "text": NEGATIVE}},
        "37": {"class_type": "LTXVConditioning", "inputs": {"positive": L(35), "negative": L(36), "frame_rate": FPS}},
        "38": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": int(w), "height": int(h), "length": int(length),
                                                                "batch_size": 1}},
        "39": {"class_type": "LTXVAddGuide", "inputs": {"positive": L(37, 0), "negative": L(37, 1), "vae": L(33),
                                                        "latent": L(38), "image": L(10, 0), "frame_idx": 0,
                                                        "strength": 1.0, "iclora_parameters": L(51)}},
    }
    last = "39"
    for i, (idx, d) in enumerate(anchors or []):   # anchors: plain keyframe guides chained after the IC-LoRA guide
        lid, gid = str(60 + 2 * i), str(61 + 2 * i)
        g[lid] = vhs_load(Path(d))
        g[gid] = {"class_type": "LTXVAddGuide", "inputs": {"positive": L(last, 0), "negative": L(last, 1), "vae": L(33),
                                                           "latent": L(last, 2), "image": L(lid, 0),
                                                           "frame_idx": int(idx), "strength": 1.0}}
        last = gid
    g.update({
        "41": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": int(length), "frame_rate": int(FPS),
                                                                "batch_size": 1, "audio_vae": L(34)}},
        "42": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": L(last, 2), "audio_latent": L(41)}},
        "43": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "44": {"class_type": "CFGGuider", "inputs": {"model": L(31), "positive": L(last, 0), "negative": L(last, 1), "cfg": 1.0}},
        "45": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "46": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "47": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": L(43), "guider": L(44), "sampler": L(45),
                                                                 "sigmas": L(46), "latent_image": L(42)}},
        "48": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": L(47)}},
        "49": {"class_type": "LTXVCropGuides", "inputs": {"positive": L(last, 0), "negative": L(last, 1), "latent": L(48, 0)}},
        "50": {"class_type": "VAEDecodeTiled", "inputs": {"samples": L(49, 2), "vae": L(33), "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 128, "temporal_overlap": 32}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": L(50), "filename_prefix": prefix}},
    })
    return g


def run(client: Any, io_dir: Path, cfg: Dict[str, Any], frames: Sequence[int], plate_fn: Callable[[int], np.ndarray],
        mask_fn: Callable[[int], np.ndarray], prompt: str, linear: bool = False,
        progress: Optional[Callable[[str, int, int], None]] = None, seed: int = SEED,
        anchors: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
    """One LTX-2.5 pass over `frames` -> {frame: uint8 BGR at the canvas}. `anchors` = {plate frame: uint8 BGR full
    frame} pinned at that frame of the clip."""
    from engine.cleanup3d.void_engine import to_display8
    from engine.comfy_io import require_vhs
    from engine.gpu_manager import before_stage, comfy_lease
    st = status(cfg, client)
    if not st["ok"]:
        raise RuntimeError(st["reason"])
    require_vhs(client)
    before_stage("comfy", client, LABEL)   # Comfy's cached models out first (as MiniMax-H3 / Wan 2.2 Fun VACE)
    frames = [int(f) for f in frames]
    if not frames:
        raise ValueError(f"{LABEL}: pusty zakres klatek")
    h0, w0 = plate_fn(frames[0]).shape[:2]
    cw, ch = canvas(w0, h0)
    n = len(frames)
    length = clip_length(n)
    job = Path(io_dir) / f"ltxfill_{uuid.uuid4().hex[:8]}"
    gd = job / "guide"
    gd.mkdir(parents=True)
    try:
        small, masks = [], []
        for i, f in enumerate(frames):
            if progress:
                progress(f"{LABEL}: klatki i maska F{f}", i, n)
            small.append(cv2.resize(to_display8(plate_fn(f), linear), (cw, ch), interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(np.asarray(mask_fn(f), np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR))
        guide = green_frames(small, grow_masks(masks, MASK_GROW_PX, MASK_TIME_RADIUS, MASK_GRID_PX))
        for i in range(length):   # padded to 8k + 1: the last frame repeated
            cv2.imwrite(str(gd / f"g_{i:05d}.png"), guide[min(i, n - 1)])
        guides = []
        for f, img in (anchors or {}).items():
            if int(f) not in frames:
                continue
            ad = job / f"anchor_{int(f):05d}"
            ad.mkdir(parents=True)
            cv2.imwrite(str(ad / "a.png"), cv2.resize(np.asarray(img), (cw, ch), interpolation=cv2.INTER_AREA))
            guides.append((frames.index(int(f)), ad))
        g = graph(gd, f"matte_desk/{job.name}/ltx", prompt, cw, ch, length, st["files"], seed, guides)
        if progress:
            progress(f"{LABEL}: liczenie w ComfyUI ({length} kl., {cw}×{ch})", 0, 1)
        with comfy_lease(client, f"{LABEL} (inpaint)"):
            sub = client.queue_prompt(g)
            if not sub.get("success"):
                raise RuntimeError(f"ComfyUI odrzucił graf {LABEL}: {sub.get('error')}")
            res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=6 * 3600)
        if not res.get("success"):
            raise RuntimeError(f"{LABEL} nie powiódł się: {res.get('error')}")
        outs = res.get("outputs", {}).get(SAVE_NODE, {}).get("images", [])
        if len(outs) < n:
            raise RuntimeError(f"{LABEL} zwrócił {len(outs)} klatek zamiast {n}")
        got: Dict[int, np.ndarray] = {}
        for f, info in zip(frames, outs[:n]):
            im = cv2.imdecode(np.frombuffer(client.download_output(info), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"{LABEL}: zły obraz wyjściowy F{f}")
            got[f] = im
        return got
    finally:
        shutil.rmtree(job, ignore_errors=True)


class LtxFill(H3Fill):
    label = LABEL
    grow_px, grid_px = MASK_GROW_PX, MASK_GRID_PX
