# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wan 2.2 Fun VACE (A14B, GGUF Q4_K_S) as the second Inpaint (C1) engine – the 12 GB alternative to MiniMax-H3
(user decision 2026-11-11). Native ComfyUI graph (`WanVaceToVideo` + two `KSamplerAdvanced` passes: the high-noise
expert, then the low-noise one; the GGUF experts load through ComfyUI-GGUF's `UnetLoaderGGUF`).

Inpaint: the control video is the plate with the masked area grey (0.5) and the control masks say where to generate.
Anchors (the same Qwen-Image-2.1 anchors as MiniMax-H3, engine/h3_anchors.py): at an anchor's frame the control frame
IS the anchor image and its mask is empty – the frame is given, the rest of the clip is generated around it; the main
anchor also goes in as the reference image. Frames come back at the canvas size as the key fill (as H3)."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.h3_fill import H3Fill, grow_masks

LABEL = "Wan 2.2 Fun VACE"
REPO = "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF"
HIGH = "Wan2.2-VACE-Fun-A14B-high-noise-Q4_K_S.gguf"
LOW = "Wan2.2-VACE-Fun-A14B-low-noise-Q4_K_S.gguf"
HF_FILES = [("HighNoise/" + HIGH, "diffusion_models", HIGH, 10_552_539_040,
             "35d99d193779be7d04c29453591d8dc266f110f7cbc2cc090126c172c514e551"),
            ("LowNoise/" + LOW, "diffusion_models", LOW, 10_552_539_040,
             "bde561ed44958776a47cebfa35a5202c2b706987aebf76d46c1298cc6286fa3b")]
# shared with the Wan 2.1 / 2.2 repacks (often already in models/): the text encoder and the video VAE
WAN_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
CLIPS = ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5_xxl_fp16.safetensors")
VAES = ("wan_2.1_vae.safetensors",)
SHARED_FILES = [("split_files/text_encoders/" + CLIPS[0], "text_encoders", CLIPS[0], 6_735_906_897),
                ("split_files/vae/" + VAES[0], "vae", VAES[0], 253_815_318)]
NODES = ["UnetLoaderGGUF", "WanVaceToVideo", "KSamplerAdvanced", "TrimVideoLatent", "ModelSamplingSD3"]
# „Szybko” (user 2026-11-11: MiniMax-H3 with its turbo LoRA was much faster): the Wan 2.2 lightx2v 4-step distill
# LoRAs (T2V A14B, the base of VACE Fun) on both experts – 4 steps, CFG 1 = ~4 model passes instead of 40
LIGHTNING_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
LIGHTNING = {"high": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
             "low": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"}
LIGHTNING_FILES = [("split_files/loras/" + LIGHTNING["high"], "loras", LIGHTNING["high"], 1_226_977_424),
                   ("split_files/loras/" + LIGHTNING["low"], "loras", LIGHTNING["low"], 1_226_977_424)]
FAST_STEPS, FAST_SPLIT, FAST_CFG, FAST_SHIFT = 4, 2, 1.0, 5.0
SPEEDS = ("fast", "quality")
CANVAS_MULTIPLE = 16
SHORT_EDGES = (480, 720)     # 720 = the model's best; 480 = faster / less memory
STEPS, SPLIT, CFG, SHIFT = 20, 10, 3.5, 8.0
SAMPLER, SCHEDULER = "euler", "simple"
SEED = 42
SAVE_NODE = "40"
MASK_GROW_PX = 8
MASK_GRID_PX = 8             # the Wan VAE's latent cell (8 px): the mask covers whole cells (h3_fill.snap_to_grid)
MASK_TIME_RADIUS = 2
NEGATIVE = ("blurry, low quality, distorted, deformed, flicker, jitter, artifacts, watermark, text, extra limbs, "
            "bad anatomy, overexposed, static frame")
REMOVE_PROMPT = ("The same shot with the masked object removed: the background continues naturally behind it, with the "
                 "same light, colours, grain and camera motion.")


def _find(models_dir: Optional[Path], folder: str, names: Sequence[str]) -> Optional[str]:
    if not models_dir:
        return None
    for n in names:
        if (Path(models_dir) / folder / n).is_file():
            return n
    return None


def files(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    md = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    return {"high": _find(md, "diffusion_models", (HIGH,)), "low": _find(md, "diffusion_models", (LOW,)),
            "clip": _find(md, "text_encoders", CLIPS), "vae": _find(md, "vae", VAES),
            "lora_high": _find(md, "loras", (LIGHTNING["high"],)), "lora_low": _find(md, "loras", (LIGHTNING["low"],))}


def has_lightning(f: Dict[str, Optional[str]]) -> bool:
    return bool(f.get("lora_high") and f.get("lora_low"))


def status(cfg: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    """{ok, reason, files}: both GGUF experts + the umt5 text encoder + the Wan VAE on disk, and (ComfyUI answering)
    the nodes – ComfyUI-GGUF's UnetLoaderGGUF and the native VACE ones."""
    f = files(cfg)
    if not cfg.get("models_dir"):
        return {"ok": False, "reason": "Brak models_dir w konfiguracji – Doktor", "files": f}
    miss = [n for k, n in (("high", HIGH), ("low", LOW), ("clip", CLIPS[0]), ("vae", VAES[0])) if not f[k]]   # LoRAs optional
    if miss:
        return {"ok": False, "reason": f"Brak plików {LABEL}: {', '.join(miss)} – Doktor → Zainstaluj {LABEL}", "files": f}
    if client is not None:
        from engine.comfy_io import missing_nodes
        m = missing_nodes(client, NODES)
        if m:
            gg = "UnetLoaderGGUF" in m
            return {"ok": False, "files": f, "reason": f"Twój ComfyUI nie ma węzła {', '.join(m)} – " +
                    ("Doktor → Zainstaluj Wan 2.2 Fun VACE (węzeł ComfyUI-GGUF), potem restart ComfyUI" if gg
                     else "zaktualizuj ComfyUI")}
    return {"ok": True, "reason": "", "files": f}


def canvas(w: int, h: int, short: int = 720) -> Tuple[int, int]:
    """(cw, ch): the shorter side = `short`, the other by the aspect, both multiples of 16."""
    s = float(short) / min(w, h)
    m = CANVAS_MULTIPLE
    return max(m, int(round(w * s / m)) * m), max(m, int(round(h * s / m)) * m)


def clip_length(n: int) -> int:
    """Wan clips are 4k + 1 frames (at least 5)."""
    n = max(5, int(n))
    while n % 4 != 1:
        n += 1
    return n


def prompt_for(description: Optional[str], mode: str = "remove", transition: bool = False, morph: bool = False) -> str:
    d = (description or "").strip().rstrip(".")
    if not d:
        return REMOVE_PROMPT
    if transition:
        from engine.h3_fill import MORPH_SENTENCE
        return (f"The same shot; inside the masked area: {d}. " + (MORPH_SENTENCE + " " if morph else "") +
                "Same light, perspective, colours, grain and camera motion.")
    if mode == "replace":
        return (f"The same shot, from the first frame to the last the masked object is replaced: {d}. The look stays the "
                "same through the whole shot, matching the light, perspective, colours, grain and camera motion.")
    return f"{REMOVE_PROMPT} In its place: {d}."


def graph(ctrl_dir: Path, masks_dir: Path, ref_dir: Optional[Path], prefix: str, prompt: str, w: int, h: int,
          length: int, f: Dict[str, Optional[str]], seed: int = SEED, fast: bool = False,
          negative: Optional[str] = None) -> Dict[str, Any]:
    """API-format graph: VACE inpaint conditioning on the control video / masks (+ the main anchor as the reference),
    the high-noise expert for the first SPLIT steps, the low-noise one for the rest, trimmed and decoded. `fast` = the
    Lightning LoRAs on both experts, 4 steps, CFG 1 (needs f["lora_high"] / f["lora_low"])."""
    fast = bool(fast and has_lightning(f))
    steps, split, cfg, shift = (FAST_STEPS, FAST_SPLIT, FAST_CFG, FAST_SHIFT) if fast else (STEPS, SPLIT, CFG, SHIFT)
    from engine.comfy_io import image_to_mask, vhs_load
    vace = {"positive": ["5", 0], "negative": ["6", 0], "vae": ["3", 0], "width": int(w), "height": int(h),
            "length": int(length), "batch_size": 1, "strength": 1.0, "control_video": ["10", 0],
            "control_masks": ["12", 0]}
    g: Dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": f["high"]}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": f["low"]}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["clip"], "type": "wan", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        # `negative` (user 2026-11-15): the removed object's noun in front of the usual negative – the one place a video
        # prompt may name it; acts only with CFG > 1 („jakość”), the Lightning „szybko” runs CFG 1
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": f"{negative.strip()}, {NEGATIVE}" if negative else NEGATIVE}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["31", 0] if fast else ["1", 0], "shift": shift}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["32", 0] if fast else ["2", 0], "shift": shift}},
        "10": vhs_load(Path(ctrl_dir)),
        "11": vhs_load(Path(masks_dir)),
        "12": image_to_mask(["11", 0]),
        "20": {"class_type": "WanVaceToVideo", "inputs": vace},
        "21": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "add_noise": "enable", "noise_seed": int(seed), "steps": steps, "cfg": cfg,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "positive": ["20", 0], "negative": ["20", 1],
            "latent_image": ["20", 2], "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}},
        "22": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["8", 0], "add_noise": "disable", "noise_seed": int(seed), "steps": steps, "cfg": cfg,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "positive": ["20", 0], "negative": ["20", 1],
            "latent_image": ["21", 0], "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "23": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["22", 0], "trim_amount": ["20", 3]}},
        "29": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["3", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["29", 0], "filename_prefix": prefix}},
    }
    if fast:
        g["31"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": f["lora_high"],
                                                                   "strength_model": 1.0}}
        g["32"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": f["lora_low"],
                                                                   "strength_model": 1.0}}
    if ref_dir is not None:
        g["13"] = vhs_load(Path(ref_dir))
        vace["reference_image"] = ["13", 0]
    return g


def control_frames(frames: Sequence[int], plate8: Callable[[int], np.ndarray], masks: Sequence[np.ndarray],
                   anchors: Dict[int, np.ndarray], cw: int, ch: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(control images uint8 BGR, control masks uint8 0/255) at the canvas: grey inside the grown mask; an anchor's
    frame = the anchor image with an empty mask (the frame is given)."""
    ctrl, msk = [], []
    for i, f in enumerate(frames):
        if int(f) in anchors:
            ctrl.append(cv2.resize(np.asarray(anchors[int(f)]), (cw, ch), interpolation=cv2.INTER_AREA))
            msk.append(np.zeros((ch, cw), np.uint8))
            continue
        img = cv2.resize(plate8(int(f)), (cw, ch), interpolation=cv2.INTER_AREA)
        m = masks[i]
        img = np.where(m[..., None] > 127, np.uint8(127), img).astype(np.uint8)
        ctrl.append(img)
        msk.append(m)
    return ctrl, msk


def run(client: Any, io_dir: Path, cfg: Dict[str, Any], frames: Sequence[int], plate_fn: Callable[[int], np.ndarray],
        mask_fn: Callable[[int], np.ndarray], prompt: str, linear: bool = False,
        progress: Optional[Callable[[str, int, int], None]] = None, seed: int = SEED,
        anchors: Optional[Dict[int, np.ndarray]] = None, short_edge: int = 720,
        reference: bool = True, fast: bool = False, ref_only: bool = False,
        negative: Optional[str] = None) -> Dict[int, np.ndarray]:
    """One VACE pass over `frames` -> {frame: uint8 BGR at the canvas}. `anchors` = {plate frame: uint8 BGR full frame},
    the first one (the main) also the reference image unless `reference` is False (a transition). `ref_only`
    („tylko referencja”, 2026-11-13) = the main anchor is the reference image only: no control frame is replaced."""
    from engine.cleanup3d.void_engine import to_display8
    from engine.comfy_io import require_vhs
    from engine.gpu_manager import comfy_lease
    st = status(cfg, client)
    if not st["ok"]:
        raise RuntimeError(st["reason"])
    require_vhs(client)
    from engine.gpu_manager import before_stage   # Comfy's cached models out first (as MiniMax-H3, 2026-11-11)
    before_stage("comfy", client, LABEL)
    frames = [int(f) for f in frames]
    if not frames:
        raise ValueError(f"{LABEL}: pusty zakres klatek")
    h0, w0 = plate_fn(frames[0]).shape[:2]
    cw, ch = canvas(w0, h0, short_edge if short_edge in SHORT_EDGES else 720)
    n = len(frames)
    length = clip_length(n)
    anchors = {int(k): v for k, v in (anchors or {}).items() if int(k) in frames}
    job = Path(io_dir) / f"vace_{uuid.uuid4().hex[:8]}"
    cd, md = job / "ctrl", job / "masks"
    cd.mkdir(parents=True)
    md.mkdir(parents=True)
    try:
        small = []
        for i, f in enumerate(frames):
            if progress:
                progress(f"{LABEL}: maska F{f}", i, n)
            small.append(cv2.resize(np.asarray(mask_fn(f), np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR))
        grown = grow_masks(small, MASK_GROW_PX, MASK_TIME_RADIUS, MASK_GRID_PX)
        ctrl, msk = control_frames(frames, lambda f: to_display8(plate_fn(f), linear), grown,
                                   {} if ref_only else anchors, cw, ch)
        for i in range(length):   # padded to 4k + 1: the last frame repeated
            j = min(i, n - 1)
            cv2.imwrite(str(cd / f"c_{i:05d}.png"), ctrl[j])
            cv2.imwrite(str(md / f"m_{i:05d}.png"), msk[j])
        rd = None
        if anchors and (reference or ref_only):
            rd = job / "ref"
            rd.mkdir()
            first = next(iter(anchors.values()))
            cv2.imwrite(str(rd / "r.png"), cv2.resize(np.asarray(first), (cw, ch), interpolation=cv2.INTER_AREA))
        g = graph(cd, md, rd, f"matte_desk/{job.name}/vace", prompt, cw, ch, length, st["files"], seed, fast,
                  **({"negative": negative} if negative else {}))
        if progress:
            fs = "4 kroki, Lightning" if fast and has_lightning(st["files"]) else f"{STEPS} kroków"
            progress(f"{LABEL}: liczenie w ComfyUI ({length} kl., {cw}×{ch}, {fs})", 0, 1)
        with comfy_lease(client, f"{LABEL} (inpaint)"):
            sub = client.queue_prompt(g)
            if not sub.get("success"):
                raise RuntimeError(f"ComfyUI odrzucił graf {LABEL}: {sub.get('error')}")
            res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=8 * 3600)
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


class VaceFill(H3Fill):
    label = LABEL
    grow_px, grid_px = MASK_GROW_PX, MASK_GRID_PX
