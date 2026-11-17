# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MiniMax-H3 masked video-to-video as a Cleanup fill model (step C1, user decision 2026-11-10: "wepnijmy w MatteDeska").

H3 is a video (+ audio) model; its DiT takes a per-token denoise mask (comfy/ldm/minimax/model.py `mask_row_values`),
so the plate clip is VAE-encoded, the removal mask set as the latent noise mask and only the masked tokens are
generated – the rest stays the plate. Native ComfyUI nodes only (no custom node):
    frames -> VAEEncode -> SetLatentNoiseMask(mask) ─┐
    EmptyMiniMaxH3LatentAV -> LTXVSeparateAVLatent ──┴> LTXVConcatAVLatent (video + empty audio) -> SamplerCustomAdvanced
    MiniMaxH3ImageToVideo(prompt) -> BasicGuider;  UNETLoader -> turbo LoRA -> MiniMaxH3SigmaShift(12 / 3)
    -> LTXVSeparateAVLatent -> VAEDecode -> SaveImage
Settings of the user's working graph (video_minimax_h3_F1.json): fl2va pruned int8, CLIP type "minimax", turbo LoRA,
BasicScheduler beta 8 steps, euler. The clip goes in at the model's canvas (768 short edge, area <= 768 x 1344) and its
length snapped up to 17k + 5 frames (the last frame repeated); the output frames are scaled back to the plate and
pasted only inside the Cleaner's mask (H3Fill = the Cleaner's key fill, like VOID). The mask is grown a little in space
and time first: the model reads it at 1/16 resolution (2 x 2 token rows) and blends it over time.
Remove = the prompt describes the shot without the object; Replace = the description of what appears in its place.
"""
from __future__ import annotations

import math
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LABEL = "MiniMax-H3"
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
MASK_GROW_PX = 16            # canvas px (one latent cell): a thin mask must cover whole token rows
MASK_GRID_PX = 16            # canvas px: the grown mask is snapped outward to whole latent cells (user 2026-11-12, after
                             # Paint a plate's erase pass: the model never gets a cell that is half hole, half picture)
MASK_TIME_RADIUS = 2         # frames: the mask is the max over +-this (the latent packs ~3-4 frames per token)
STEPS, SAMPLER, SCHEDULER = 8, "euler", "beta"
STEPS_NO_TURBO = 30
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0
# „Siła zmiany” (user 2026-11-13): Replace may start from the plate under the mask instead of a hole, so MiniMax-H3 keeps
# its structure (a face's expression, pose) and only re-draws the look. strength = the sigma the masked tokens start at
# (1.0 = pure noise = a hole, as before); the noised plate latent is the start. BasicScheduler's `denoise` cannot do it:
# with shift 12 even denoise 0.5 starts at sigma 0.92 (checked 2026-11-13) – so the schedule is built here
# (strength_sigmas, ManualSigmas): the same beta steps over the base time grid cut at the strength. Remove always = 1.0
# (the model must not see the object it removes).
STRENGTH_MIN = 0.3
SEED = 42
SAVE_NODE = "40"
# first file found wins (folder in models_dir, candidates) – the pruned int8 of the user's graph first
UNETS = ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
         "minimax_h3_fl2va_pruned_w6a8.safetensors", "minimax_h3_fl2va_int8_convrot.safetensors",
         "minimax_h3_fl2va_pruned_bf16.safetensors")
CLIPS = ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
         "qwen3vl_32b_minimax_h3_bf16.safetensors")
VAES = ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors")
TURBO = ("minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
         "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")
# MiniMax-H3 Fun ControlNet Union (alibaba-pai, repacked by Comfy-Org; user 2026-11-13): Canny / Depth / HED / MLSD / Pose
# control video + masked inpaint via MiniMaxH3FunControlNetApply – optional, not used by a stage yet (structure under the
# mask for Replace); the 2.0 pruned int8 first (matches the pruned int8 UNET)
CONTROLNETS = ("minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
               "minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors",
               "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",
               "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors")
FOLDERS = {"unet": ("diffusion_models", UNETS), "clip": ("text_encoders", CLIPS), "vae": ("vae", VAES),
           "turbo": ("loras", TURBO), "control": ("model_patches", CONTROLNETS)}
REMOVE_PROMPT = ("The same shot with the masked object removed: the background continues naturally behind it, with the "
                 "same light, colours, grain and camera motion. Nothing else changes.")


def _find(models_dir: Optional[Path], key: str) -> Optional[str]:
    folder, names = FOLDERS[key]
    if not models_dir:
        return None
    for n in names:
        if (Path(models_dir) / folder / n).is_file():
            return n
    return None


def files(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    md = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    return {k: _find(md, k) for k in FOLDERS}


def status(cfg: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    """{ok, reason, files}: model + text encoder + video VAE on disk (the turbo LoRA is optional: 30 steps without it)
    and the native H3 nodes in this ComfyUI (when it answers)."""
    f = files(cfg)
    miss = [FOLDERS[k][0] + "/" + FOLDERS[k][1][0] for k in ("unet", "clip", "vae") if not f[k]]
    if not cfg.get("models_dir"):
        return {"ok": False, "reason": "Brak models_dir w konfiguracji – Doktor", "files": f}
    if miss:
        return {"ok": False, "reason": f"Brak plików {LABEL}: {', '.join(miss)} (Comfy-Org/MiniMax-H3) – pobierz je do "
                                       "models ComfyUI", "files": f}
    from engine.stage_models import comfy_gate
    why = comfy_gate(client, "h3")
    return {"ok": not why, "reason": why, "files": f}


def canvas(w: int, h: int) -> Tuple[int, int]:
    """(cw, ch) = H3's adapt_canvas: 768 short edge, area capped at 768 x 1344, sides multiples of 32."""
    ratio = w / float(h)
    nw, nh = (BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE) if ratio >= 1.0 else (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio)
    if nw * nh > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nw * nh))
        nw, nh = nw * s, nh * s
    return (max(CANVAS_MULTIPLE, round(nw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def clip_length(n: int) -> int:
    """Frames the model takes for n plate frames: snapped up to 17k + 5 (at least 5)."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def clamp_strength(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 1.0
    return float(min(1.0, max(STRENGTH_MIN, x)))


def run_strength(strength: Any, mode: str) -> float:
    """The strength a run uses: Replace = the setting; Remove = always 1.0 (a hole)."""
    return clamp_strength(strength) if mode == "replace" else 1.0


def strength_sigmas(strength: float, steps: int, shift: float = SHIFT_VIDEO) -> List[float]:
    """`steps` sigmas from `strength` down to 0 on H3's schedule: ComfyUI's beta scheduler (alpha = beta = 0.6) over the
    base time grid [0, t0], t0 = the base time of sigma `strength` under the flow shift, then shifted. strength 1.0 =
    the full schedule (what BasicScheduler beta gives, up to its 1000-step rounding)."""
    import scipy.stats
    s = clamp_strength(strength)
    t0 = s / (shift + s * (1.0 - shift))             # invert sigma = shift*t / (1 + (shift-1)*t)
    q = 1.0 - np.linspace(0.0, 1.0, int(steps), endpoint=False)
    ts = scipy.stats.beta.ppf(q, 0.6, 0.6) * t0
    sig = [float(shift * t / (1.0 + (shift - 1.0) * t)) for t in ts]
    return sig + [0.0]


MORPH_SENTENCE = ("The design inside the masked area changes gradually and continuously from each pinned keyframe to the "
                  "next: a little in every frame, flowing smoothly between them, never jumping.")


def h3_prompt(description: Optional[str], mode: str = "remove", transition: bool = False, morph: bool = False) -> str:
    d = (description or "").strip()
    if transition and d:   # the anchors differ on purpose (a morph / from the original): the user's clip prompt as is
        return (f"The same shot; inside the masked area: {d.rstrip('.')}. " + (MORPH_SENTENCE + " " if morph else "") +
                "It matches the light, perspective, colours, "
                "grain and camera motion of the shot. Nothing outside the masked area changes.")
    if mode == "replace" and d:
        # 2026-11-11 (user test, „Transform the facial skin into …”): a video model reads a change as an EVENT – the
        # face stayed itself until ~F25 and then turned. The look is a STATE from the first frame to the last.
        return (f"The same shot, but from the very first frame to the last the masked object is already replaced: "
                f"{d.rstrip('.')}. This look is there in every frame and stays the same through the whole shot – "
                "nothing transforms, appears or fades in over time. It matches the light, perspective, colours, grain "
                "and camera motion of the shot. Nothing else changes.")
    if d:
        return f"{REMOVE_PROMPT} In its place: {d.rstrip('.')}."
    return REMOVE_PROMPT


def ref_prompt(prompt: str) -> str:
    """The clip prompt when the anchor rides as <Picture 1>: the masked area looks like it in EVERY frame."""
    return ("From the first frame to the last the masked area looks exactly like in <Picture 1> – the same design, "
            f"colours, materials and details in every frame, following the motion of the shot. {prompt}")


def snap_to_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """A binary mask (uint8 0 / 255) grown outward to whole `grid` x `grid` cells counted from the top left corner (the
    latent's cells): a cell with any masked pixel is masked whole. grid <= 1 = unchanged."""
    g = int(grid)
    if g <= 1:
        return mask
    h, w = mask.shape[:2]
    ph, pw = (-h) % g, (-w) % g
    m = np.pad(mask, ((0, ph), (0, pw))) if ph or pw else mask
    cells = m.reshape(m.shape[0] // g, g, m.shape[1] // g, g).max(axis=(1, 3))
    return np.ascontiguousarray(np.repeat(np.repeat(cells, g, axis=0), g, axis=1)[:h, :w])


def grow_masks(masks: Sequence[np.ndarray], px: int = MASK_GROW_PX, radius: int = MASK_TIME_RADIUS,
               grid: int = 0) -> List[np.ndarray]:
    """Binary masks grown by `px` (ellipse), max over +-`radius` frames, then snapped outward to whole `grid` cells
    (`snap_to_grid`; 0 = no snap) – uint8 0 / 255."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)) if px > 0 else None
    hard = [((np.asarray(m, np.float32) > 0.02).astype(np.uint8) * 255) for m in masks]
    if k is not None:
        hard = [cv2.dilate(m, k) for m in hard]
    out = []
    for i in range(len(hard)):
        lo, hi = max(0, i - radius), min(len(hard), i + radius + 1)
        out.append(snap_to_grid(np.max(np.stack(hard[lo:hi]), axis=0), grid))
    return out


def token_masks(masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Per-frame masks of the whole clip (17k + 5) -> one mask per latent token = the max over the token's frames
    (h3_anchors.token_spans: 1, 4, 4, 4, 4 …). ComfyUI fits the noise mask to the latent with a TRILINEAR resize
    (comfy.utils.reshape_mask): from per-frame masks the token of frames 1-4 sampled the mask near frame 4.4, so where
    the object moves fast its pixels in the token's first frames stayed UNMASKED and the plate (the object) was kept
    (user report 2026-11-16, a test shot, v005: at F2 only 11 % of the earring's pixels fully masked, 51 % not at
    all – the earring came back whatever the anchor / seed). With exactly one mask per token the resize is identity."""
    from engine.h3_anchors import token_spans
    return [np.max(np.stack(masks[a:b + 1]), axis=0) for a, b in token_spans(len(masks))]


def graph(frames_dir: Path, masks_dir: Path, prefix: str, prompt: str, w: int, h: int, length: int,
          f: Dict[str, Optional[str]], seed: int = SEED,
          guides: Optional[Sequence[Tuple[int, Path]]] = None, reference: bool = True,
          strength: float = 1.0, ref_only: bool = False) -> Dict[str, Any]:
    """The masked V2V graph (API format) – see the module docstring. `guides` = [(clip frame index, folder with ONE
    image at the canvas size)] pinned with MiniMaxH3AddGuide (the anchors). `strength` < 1 = the masked area starts
    from the noised plate at that sigma (ManualSigmas, strength_sigmas) instead of a hole. `ref_only` („tylko
    referencja”, 2026-11-13) = the first guide's image is the whole-clip reference ONLY: no frame is pinned."""
    from engine.comfy_io import image_to_mask, vhs_load
    turbo = f.get("turbo")
    g: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "10": vhs_load(Path(frames_dir)),
        "11": vhs_load(Path(masks_dir)),
        "12": image_to_mask(["11", 0]),
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["12", 0]}},
        "15": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": int(w), "height": int(h), "length": int(length)}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["14", 0], "audio_latent": ["16", 1]}},
        "20": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                                                                 "width": int(w), "height": int(h), "length": int(length)}},
        "22": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["21", 0] if turbo else ["1", 0],
                                                               "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO}},
        "23": {"class_type": "BasicGuider", "inputs": {"model": ["22", 0], "conditioning": ["20", 0]}},
        "24": {"class_type": "BasicScheduler", "inputs": {"model": ["22", 0], "scheduler": SCHEDULER,
                                                          "steps": STEPS if turbo else STEPS_NO_TURBO, "denoise": 1.0}},
        "25": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "26": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "27": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["26", 0], "guider": ["23", 0],
                                                                 "sampler": ["25", 0], "sigmas": ["24", 0],
                                                                 "latent_image": ["17", 0]}},
        "28": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["27", 0]}},
        "29": {"class_type": "VAEDecode", "inputs": {"samples": ["28", 0], "vae": ["3", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["29", 0], "filename_prefix": prefix}},
    }
    if clamp_strength(strength) < 1.0:
        sig = strength_sigmas(strength, STEPS if turbo else STEPS_NO_TURBO)
        g["24"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(f"{x:.6f}" for x in sig)}}
    if turbo:
        g["21"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": turbo,
                                                                   "strength_model": 1.0}}
    pos: List[Any] = ["20", 0]
    if guides and reference:   # 2026-11-11: the anchor is ALSO a whole-clip reference (<Picture 1>): AddGuide alone pins only the
        # frames near its index (user test: forward fine, a few frames back the model drew a plain face again)
        g["20"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": ref_prompt(prompt), "width": int(w), "height": int(h),
            "length": int(length), "ref_image_size": "match", "ref_images.ref_image_0": ["60", 0]}}
    if guides and ref_only:   # the reference image without any AddGuide (node "60" as with pinned guides)
        g["60"] = vhs_load(Path(guides[0][1]))
        g["20"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": ref_prompt(prompt), "width": int(w), "height": int(h),
            "length": int(length), "ref_image_size": "match", "ref_images.ref_image_0": ["60", 0]}}
        guides = []
    for i, (idx, d) in enumerate(guides or []):   # anchors: chained on the positive conditioning
        lid, gid = str(60 + 2 * i), str(61 + 2 * i)
        g[lid] = vhs_load(Path(d))
        g[gid] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": pos, "vae": ["3", 0], "latent": ["17", 0],
                                                                "image": [lid, 0], "frame_idx": int(idx)}}
        pos = [gid, 0]
    g["23"]["inputs"]["conditioning"] = pos
    return g


def run(client: Any, io_dir: Path, cfg: Dict[str, Any], frames: Sequence[int], plate_fn: Callable[[int], np.ndarray],
        mask_fn: Callable[[int], np.ndarray], prompt: str, linear: bool = False,
        progress: Optional[Callable[[str, int, int], None]] = None, seed: int = SEED,
        anchors: Optional[Dict[int, np.ndarray]] = None, reference: bool = True,
        strength: float = 1.0, ref_only: bool = False) -> Dict[int, np.ndarray]:
    """One H3 pass over `frames` -> {frame: uint8 BGR at the model's canvas}. Plate in display sRGB (as VOID).
    `anchors` = {plate frame: uint8 BGR full frame} pinned at that frame of the clip (make_anchor).
    `strength` < 1 = start from the plate under the mask (Replace keeping the structure; run_strength)."""
    from engine.cleanup3d.void_engine import to_display8
    from engine.comfy_io import require_vhs
    from engine.gpu_manager import comfy_lease
    st = status(cfg, client)
    if not st["ok"]:
        raise RuntimeError(st["reason"])
    require_vhs(client)
    # the whole card AND Comfy's model cache first (user report 2026-11-11: after a Wan 2.2 Fun VACE run ComfyUI still
    # held its experts in RAM – 52 GB private – and MiniMax-H3 spilled 25 GB with the RAM nearly full)
    from engine.gpu_manager import before_stage
    before_stage("comfy", client, LABEL)
    frames = [int(f) for f in frames]
    if not frames:
        raise ValueError(f"{LABEL}: pusty zakres klatek")
    first = plate_fn(frames[0])
    h0, w0 = first.shape[:2]
    cw, ch = canvas(w0, h0)
    n = len(frames)
    length = clip_length(n)
    job = Path(io_dir) / f"h3_{uuid.uuid4().hex[:8]}"
    fd, md = job / "frames", job / "masks"
    fd.mkdir(parents=True)
    md.mkdir(parents=True)
    try:
        masks = []
        for i, f in enumerate(frames):
            if progress:
                progress(f"{LABEL}: klatki i maska F{f}", i, n)
            pl = plate_fn(f)
            cv2.imwrite(str(fd / f"f_{i:05d}.png"), cv2.resize(to_display8(pl, linear), (cw, ch), interpolation=cv2.INTER_AREA))
            masks.append(cv2.resize(np.asarray(mask_fn(f), np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR))
        grown = grow_masks(masks, grid=MASK_GRID_PX)
        grown += [grown[-1]] * (length - n)   # pad to 17k + 5: the last mask repeated
        for i, m in enumerate(token_masks(grown)):   # ONE mask per latent token (see token_masks)
            cv2.imwrite(str(md / f"m_{i:05d}.png"), m)
        for i in range(n, length):   # pad to 17k + 5: the last frame repeated
            shutil.copy(fd / f"f_{n - 1:05d}.png", fd / f"f_{i:05d}.png")
        guides = []
        for f, img in (anchors or {}).items():   # insertion order: the main anchor first (= the <Picture 1> reference)
            if int(f) not in frames:
                continue
            gd = job / f"guide_{int(f):05d}"
            gd.mkdir(parents=True)
            cv2.imwrite(str(gd / "g.png"), cv2.resize(np.asarray(img), (cw, ch), interpolation=cv2.INTER_AREA))
            guides.append((frames.index(int(f)), gd))
        g = graph(fd, md, f"matte_desk/{job.name}/h3", prompt, cw, ch, length, st["files"], seed, guides, reference,
                  strength, ref_only)
        if progress:
            progress(f"{LABEL}: liczenie w ComfyUI ({length} kl., {cw}×{ch})", 0, 1)
        with comfy_lease(client, f"{LABEL} (czyszczenie)"):
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


# ----------------------------------------------------------------------------- anchors (user decision 2026-11-10)
# Qwen-Image-2.1 (an image model: frame by frame it flickers, so it is not a C1 fill model) makes ONE look frame – the
# plate with the object replaced / removed inside the mask – and H3 gets it pinned at that frame of the clip
# (MiniMaxH3AddGuide: re-injected every step, never denoised); the rest of the clip follows it instead of drifting
# back to the original towards the end (test 2026-11-10, F44 of 50).
# 2026-11-11 (user): ONE anchor at a chosen frame – two independent Qwen-Image-2.1 frames (first + last) can disagree
# (the very flicker that took it off the fill list) and H3 would be pulled between them; a frame where the whole object
# is visible (often mid-clip) holds both ends. Auto = suggest_anchor.
ANCHOR_MODES = ("none", "frame")


def suggest_anchor(frames: Sequence[int], mask_fn: Callable[[int], np.ndarray], samples: int = 24) -> Optional[int]:
    """The frame where the object is largest and least cut by the frame edge: score = mask area x (1 - the share of
    its outline lying on the image border). `samples` frames evenly over the range (the mask is costly)."""
    frames = [int(f) for f in frames]
    if not frames:
        return None
    step = max(1, len(frames) // max(1, samples))
    best, best_s = frames[0], -1.0
    for f in frames[::step]:
        m = np.asarray(mask_fn(f), np.float32)
        if m.size == 0:
            continue
        sc = 256.0 / max(m.shape[:2])
        b = (cv2.resize(m, (max(1, round(m.shape[1] * sc)), max(1, round(m.shape[0] * sc))), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
        area = float(b.sum())
        if area <= 0:
            continue
        outline = float((b - cv2.erode(b, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)).sum())
        edge = float(b[0].sum() + b[-1].sum() + b[1:-1, 0].sum() + b[1:-1, -1].sum())
        s = area * (1.0 - min(1.0, edge / max(1.0, outline)))
        if s > best_s:
            best, best_s = f, s
    return best


def anchor_frame(frames: Sequence[int], mode: str, chosen: Optional[int], mask_fn: Callable[[int], np.ndarray]) -> Optional[int]:
    """The anchor's frame: the chosen one when it lies in the work range, else the suggestion; None = no anchor."""
    frames = [int(f) for f in frames]
    if mode != "frame" or not frames:
        return None
    return int(chosen) if chosen is not None and int(chosen) in frames else suggest_anchor(frames, mask_fn)


# 2026-11-11 (user test v004, main anchor F31): H3 reads a keyframe as the moment to REACH – the clip started from the
# look of the shot (the plain face) and turned at F29-31. So the ends of the range get anchors too, DERIVED from the main
# one: Qwen-Image-2.1 gets that frame + the main anchor as Picture 2 (the same design, only fitted to the pose / light).
DERIVED_MIN_GAP = 8        # an end closer than this to the main anchor needs no anchor of its own
# 2026-11-11 (test v005): an anchor ON the first frame came out as Qwen-Image-2.1's own frame (sharper, its own detail)
# – the first frame is a latent token of its own (FRAME_PER_TOKEN 1, 4, 4, …), the pinned image passes nearly as is.
# The start anchor sits on the 5th frame (first 4-frame token), so F1 is drawn by H3 like the rest.
DERIVED_START = 4


def derived_frames(frames: Sequence[int], main: Optional[int]) -> List[int]:
    """Near the range's start (its 5th frame, DERIVED_START) / its last frame, when more than DERIVED_MIN_GAP frames
    from the main anchor."""
    frames = [int(f) for f in frames]
    if main is None or not frames:
        return []
    i = frames.index(int(main)) if int(main) in frames else 0
    out = []
    if i >= DERIVED_MIN_GAP:
        out.append(frames[min(DERIVED_START, i - 1)])
    if len(frames) - 1 - i >= DERIVED_MIN_GAP:
        out.append(frames[-1])
    return out


def derived_prompt(description: str, mode: str, removed: Optional[str] = None) -> str:
    d = (description or "").strip().rstrip(".")
    # REMOVE names the object for the edit model (`removed`, 2026-11-15): nothing is marked on its image
    what = (f"the masked object replaced with {d}" if mode == "replace" and d
            else f"the {removed.strip()} (the masked object) removed" if removed else "the masked object removed")
    return ("Edit Picture 1 so it shows exactly the same as Picture 2: " + what + " – the same design, pattern, "
            "colours and materials as in Picture 2, fitted to the pose, perspective and light of Picture 1. "
            "Everything else in Picture 1 stays unchanged.")


def anchor_key(frame: int, prompt: str, mode: str, mask: np.ndarray, seed: int = 0, ref: str = "") -> str:
    """Cache key of an anchor: the frame, the instruction, the mask, the seed and (a derived anchor) the main anchor's
    key – any of them changed = a new anchor."""
    import hashlib
    h = hashlib.sha1()
    h.update(f"{int(frame)}|{mode}|{prompt}".encode("utf-8"))
    if seed or ref:   # old keys (seed 0, no ref) stay valid
        h.update(f"|{int(seed)}|{ref}".encode("utf-8"))
    small = cv2.resize((np.asarray(mask, np.float32) > 0.02).astype(np.uint8) * 255, (128, 128), interpolation=cv2.INTER_AREA)
    h.update(small.tobytes())
    return h.hexdigest()[:16]


ANCHOR_LATENT_MASK = True   # anchors: Qwen-Image-2.1 inpaints under the mask (2026-11-15); False = the whole crop edited


def make_anchor(client: Any, io_dir: Path, plate: np.ndarray, mask: np.ndarray, prompt: str, mode: str,
                linear: bool = False, seed: Optional[int] = None, ref: Optional[np.ndarray] = None,
                init: Optional[np.ndarray] = None, denoise: float = 1.0, region: Optional[np.ndarray] = None,
                text: Optional[str] = None, raw: Optional[list] = None,
                plate_denoise: Optional[float] = None, what: Optional[str] = None) -> np.ndarray:
    """uint8 BGR plate frame with Qwen-Image-2.1's edit pasted inside the mask (soft by the mask's own values).
    `plate_denoise` < 1 (no `init`, „Siła kotwicy”, user 2026-11-15): Qwen-Image-2.1 starts from the plate itself –
    the face's shape, light and expression stay, the edit goes over them; 1 / None = drawn anew in the mask.
    `init` (uint8 BGR, 2026-11-11) = the main anchor warped onto this frame (engine/anchor_warp): Qwen-Image-2.1 starts
    from it with `denoise` < 1 and the touch-up text – it fixes light / seams, the pattern stays."""
    from engine.cleanup3d.regen import _run_qwen
    from engine.cleanup3d.void_engine import to_display8
    from engine.stage_models import comfy_gate
    why = comfy_gate(client, "qwen21_edit")
    if why:
        raise RuntimeError(f"Kotwica Qwen-Image-2.1: {why}")
    disp = to_display8(plate, linear).astype(np.float32) / 255.0
    if init is not None:
        disp = np.asarray(init, np.float32) / 255.0
    # the anchor = the edit over the WHOLE area the video model repaints (mask > 0.02), not blended by the patch's soft
    # edge / the soft holdout (user 2026-11-11): that blend is the Cleaner's, at the end; a half-plate ring in the anchor
    # was a pattern MiniMax-H3 had to copy
    m = (np.asarray(mask, np.float32) > 0.02).astype(np.float32)
    if region is not None:   # only this part is edited and pasted (the warp's holes, engine/anchor_warp)
        m = m * (np.asarray(region, np.float32) > 0.5)
    from engine.cleanup3d.regen import QWEN_FEATHER_PX, QWEN_SEED
    r = None if ref is None else cv2.resize(np.asarray(ref), (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # the anchors are inpainted (latent noise mask, 2026-11-15): Qwen-Image-2.1 redraws only the mask
    gen, _w, _n = _run_qwen(client, Path(io_dir), disp, m > 0.02, prompt, None, mode, what=what, latent_mask=ANCHOR_LATENT_MASK,
                            seed=QWEN_SEED if seed is None else int(seed), ref=r,
                            text=text or (_touchup(prompt) if init is not None else
                                  derived_prompt(prompt, mode, what) if r is not None else None),
                            denoise=float(denoise) if init is not None else
                            (float(plate_denoise) if plate_denoise is not None and float(plate_denoise) < 1.0 else 1.0),
                            feather_px=QWEN_FEATHER_PX)   # the whole picture is an anchor too (anchor_raw)
    if raw is not None:   # Qwen-Image-2.1's own picture before the paste (user 2026-11-11: kept next to the anchor)
        raw.append(np.clip(np.rint(np.asarray(gen, np.float32) * 255.0), 0, 255).astype(np.uint8))
    out = disp * (1.0 - m[..., None]) + np.asarray(gen, np.float32) * m[..., None]
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _touchup(prompt: str) -> str:
    from engine.anchor_warp import touchup_prompt
    return touchup_prompt(prompt)


class H3Fill:
    """The H3 frames as the Cleaner's key fill: f -> (frame scaled to the plate in the plate's encoding, weight 1)."""
    parallel = True            # frames from a dict / disk, read only: safe for the Cleaner's frame threads

    label = LABEL
    tone = True                   # the Cleaner levels these frames to the plate (engine/cleanup.Cleaner._tone)
    grow_px, grid_px = MASK_GROW_PX, MASK_GRID_PX      # canvas px the model's mask reaches beyond the Cleaner's

    def __init__(self, frames: Dict[int, np.ndarray], w: int, h: int, linear: bool):
        self.frames, self.w, self.h, self.linear = frames, int(w), int(h), bool(linear)

    @property
    def tone_gap_px(self) -> float:
        """Plate px around the Cleaner's mask where the model drew its own picture (its mask is grown and snapped to
        the grid at the canvas' size): the tone ring starts beyond it."""
        cw = next((im.shape[1] for im in self.frames.values() if im is not None), self.w)
        return (self.grow_px + self.grid_px) * self.w / float(max(1, cw))

    @property
    def drawn_px(self) -> float:
        """Plate px the model surely drew beyond the mask it was given (its grow on the canvas; the grid snap reaches
        further only on some sides): the Cleaner cuts the composite band there (engine/cleanup.drawn_area)."""
        cw = next((im.shape[1] for im in self.frames.values() if im is not None), self.w)
        return self.grow_px * self.w / float(max(1, cw))

    def __call__(self, f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        from engine.cleanup3d.void_engine import from_display
        im = self.frames.get(int(f))
        if im is None:
            return None
        if (im.shape[1], im.shape[0]) != (self.w, self.h):
            im = cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        return np.asarray(from_display(im, self.linear), np.float32), np.ones((self.h, self.w), np.float32)
