# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MatAnyone2 with keyframe re-anchoring.

Separate package so updates of ComfyUI-MatAnyone2 don't overwrite it.
Reuses the MATANYONE2_MODEL pipe from "Load MatAnyone2 Model".
"""
__version__ = "1.2.2"   # 1.2.2 (2026-10-17): stops between frames on ComfyUI interrupt ("Przerwij" in MatteDesk); 1.2.1 (2026-10-17): comments only (no shot names);  1.1.0 (2026-10-13): conditional VRAM release (vram_release_frac), max_mem_frames 0 = auto
                        # 1.1.1 (2026-10-13): alpha_only – matte_video / foreground outputs skipped (RAM: 2 x N x H x W x 3 float32)
                        # 1.1.2 (2026-10-15): CoreCombine core_mode = soft (SAM3 probabilities as the core, soft start of the band)
                        # 1.2.0 (2026-10-16): keyframe_anchor = soft – non-start key frames no longer as hard as the key mask

import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_management as mm
import comfy.utils

from .vram import VramReleaser, resolve_max_mem_frames


def _parse_keyframes(text, n_frames):
    idx = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 0:
            i += n_frames
        if not 0 <= i < n_frames:
            raise ValueError(f"Keyframe {part} is outside the clip (0..{n_frames - 1})")
        idx.append(i)
    idx = sorted(set(idx))
    if not idx:
        raise ValueError("Give at least one keyframe")
    return idx


class MatAnyone2SelectKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "keyframes": ("STRING", {"default": "0", "multiline": False,
                                         "tooltip": "Frame numbers separated by commas, e.g. 0, 120, 200. Negative = from the end (-1 = last frame)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("keyframe_images", "keyframes")
    FUNCTION = "select"
    CATEGORY = "video/matting"

    def select(self, images, keyframes):
        idx = _parse_keyframes(keyframes, images.shape[0])
        return (images[idx], ", ".join(str(i) for i in idx))


class MatAnyone2RunKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_pipe": ("MATANYONE2_MODEL",),
                "images": ("IMAGE",),
                "keyframe_masks": ("MASK", {"tooltip": "One mask per keyframe, in the same order (e.g. SAM3 Detect on the selected keyframes)."}),
                "keyframes": ("STRING", {"forceInput": True}),
                "start_frame": ("INT", {"default": 0, "min": -100000, "max": 100000,
                                        "tooltip": "Keyframe to start from (must be one of the keyframes). Pick the frame where the person is largest/sharpest. -1 = last frame. The clip is processed forward AND backward from here."}),
                "warmup_frames": ("INT", {"default": 10, "min": 0, "max": 50}),
                "erode_kernel": ("INT", {"default": 10, "min": 0, "max": 50}),
                "dilate_kernel": ("INT", {"default": 10, "min": 0, "max": 50}),
                "max_size": ("INT", {"default": -1, "min": -1, "max": 4096,
                                     "tooltip": "Downscale for processing if the short side is larger (-1 = full resolution)."}),
                "max_mem_frames": ("INT", {"default": 5, "min": 0, "max": 20,
                                           "tooltip": "Working memory frames. 5 = official value; 10 does not fit 12 GB at 1080p. "
                                                      "0 = auto from free VRAM and resolution (5..15; 12 GB at 1440 px -> 5)."}),
            },
            "optional": {
                "keyframe_anchor": (["soft", "hard"], {"default": "soft",
                                    "tooltip": "Re-anchoring on a key frame other than the start one. hard (old): the key mask is the "
                                               "frame's alpha and the last mask – the edge there is as hard as SAM3. soft: the model "
                                               "predicts the frame from memory first; its alpha is kept in a thin band around the key's "
                                               "edge, the key everywhere else, and the memory is anchored on that soft key."}),
                "keyframe_edge_px": ("INT", {"default": 6, "min": 0, "max": 64,
                                     "tooltip": "soft anchor: half-width (px of the processed image) of the band around the key's "
                                                "edge where the model's own alpha is kept"}),
                "vram_release_frac": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Empty the CUDA cache only when reserved VRAM exceeds this fraction of the "
                                                           "card (Windows: no expandable_segments). 0 = after every frame (old, slow)."}),
                "alpha_only": ("BOOLEAN", {"default": False,
                                           "tooltip": "Return only alpha_mattes; matte_video and foreground become 1x1 dummies. "
                                                      "Saves 2 x N x H x W x 3 float32 of RAM (514 frames at 1440 px: 25 GB) "
                                                      "and their compute. The alpha is unchanged."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("alpha_mattes", "matte_video", "foreground")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    @torch.inference_mode()
    def run(self, model_pipe, images, keyframe_masks, keyframes, start_frame, warmup_frames, erode_kernel,
            dilate_kernel, max_size, max_mem_frames, vram_release_frac=0.85, alpha_only=False, keyframe_anchor="soft",
            keyframe_edge_px=6):
        from omegaconf import open_dict
        from matanyone2.inference.inference_core import InferenceCore
        from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

        model, cfg, device = model_pipe["model"], model_pipe["cfg"], model_pipe["device"]
        n_frames, H, W = images.shape[0], images.shape[1], images.shape[2]
        kf = _parse_keyframes(keyframes, n_frames)
        if keyframe_masks.shape[0] != len(kf):
            raise ValueError(f"Got {keyframe_masks.shape[0]} keyframe masks for {len(kf)} keyframes ({keyframes})")
        anchor = start_frame + n_frames if start_frame < 0 else start_frame
        if anchor not in kf:
            raise ValueError(f"start_frame {start_frame} (= frame {anchor}) must be one of the keyframes: {keyframes}")

        nh, nw = H, W
        if max_size > 0 and min(H, W) > max_size:
            s = max_size / min(H, W)
            nh, nw = int(H * s), int(W * s)

        use_amp = device.type == "cuda"
        mem_frames = resolve_max_mem_frames(max_mem_frames, torch, nh, nw, use_amp)
        if max_mem_frames <= 0:
            print(f"[MatAnyone2RunKeyframes] max_mem_frames auto -> {mem_frames} ({nw}x{nh})")
        with open_dict(cfg):
            cfg.max_mem_frames = mem_frames

        def prep_mask(m):
            m = m.cpu().float()
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0]
            m_np = (m.numpy() > 0.5).astype(np.uint8) * 255
            if dilate_kernel > 0:
                m_np = gen_dilate(m_np, dilate_kernel, dilate_kernel)
            if erode_kernel > 0:
                m_np = gen_erosion(m_np, erode_kernel, erode_kernel)
            t = torch.from_numpy(m_np).float().to(device)
            if (nh, nw) != (H, W):
                t = F.interpolate(t[None, None], size=(nh, nw), mode="nearest")[0, 0]
            return t

        kf_masks = {i: prep_mask(keyframe_masks[k]) for k, i in enumerate(kf)}

        def frame(i):
            img = images[i].permute(2, 0, 1).float().to(device)
            if (nh, nw) != (H, W):
                img = F.interpolate(img[None], size=(nh, nw), mode="area")[0]
            return img

        forward = list(range(anchor, n_frames))
        backward = list(range(anchor, -1, -1)) if anchor > 0 else []
        pbar = comfy.utils.ProgressBar(len(forward) + len(backward) + 2 * warmup_frames)
        alphas = [None] * n_frames

        def propagate(order):
            processor = InferenceCore(model, cfg=cfg, device=device)
            # Windows: no expandable_segments, reserved VRAM balloons otherwise -> release the cache, but only
            # above vram_release_frac of the card (every frame stalled the GPU at ~20 %); 0 = every frame
            release = VramReleaser(torch, vram_release_frac, enabled=use_amp)
            out = {}
            perm = [0]   # permanent keyframes in this pass (each costs VRAM for the rest of the pass)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                for ti in range(len(order) + warmup_frames):
                    mm.throw_exception_if_processing_interrupted()   # Cancel / /interrupt: stop now, not after the pass
                    fi = order[max(0, ti - warmup_frames)]
                    img = frame(fi)
                    soft_out = None
                    if ti == 0:
                        processor.step(img, kf_masks[fi], objects=[1])
                        prob = processor.step(img, first_frame_pred=True)
                    elif ti <= warmup_frames:
                        prob = processor.step(img, first_frame_pred=True)
                    elif fi in kf_masks and keyframe_anchor == "soft":
                        # soft re-anchor (1.2.0): with the hard key as the output / last mask every non-start key frame
                        # was as hard as SAM3 (test clip, keys F20 / F80: 13.5k / 16.5k soft px vs ~19k / ~21k around them).
                        # Predict from memory first (soft edge), keep that alpha in a thin band around the key's edge and
                        # the key everywhere else, anchor the memory on this soft key and output it
                        pred = processor.step(img)
                        pa = processor.output_prob_to_mask(pred).float()
                        k = kf_masks[fi].float() / 255.0
                        r = int(keyframe_edge_px)
                        if r > 0 and pa.shape == k.shape:
                            kk = k[None, None]
                            dil = F.max_pool2d(kk, 2 * r + 1, 1, r)[0, 0]
                            ero = -F.max_pool2d(-kk, 2 * r + 1, 1, r)[0, 0]
                            sk = torch.where((dil - ero) > 0.5, pa.to(k.dtype), k)
                        else:
                            sk = k
                        processor.step(img, sk * 255.0, objects=[1], force_permanent=perm[0] < 3)
                        perm[0] += 1
                        soft_out = sk
                    elif fi in kf_masks:
                        # re-anchor: write the keyframe mask into permanent memory, then predict
                        # alpha for this frame from memory (keeps soft edges, no reset / no jump)
                        # only the first few keyframes go to PERMANENT memory (VRAM grows with each one; 12 GB OOM'd
                        # with dense keys); further keys re-anchor through the normal working memory
                        processor.step(img, kf_masks[fi], objects=[1], force_permanent=perm[0] < 3)
                        perm[0] += 1
                        prob = processor.step(img)
                    else:
                        prob = processor.step(img)
                    release.step()
                    if ti >= warmup_frames:
                        a = soft_out if soft_out is not None else processor.output_prob_to_mask(prob).float()
                        if (nh, nw) != (H, W):
                            a = F.interpolate(a[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                        out[fi] = a.clamp(0, 1).cpu()
                    pbar.update(1)
            return out

        for fi, a in propagate(forward).items():
            alphas[fi] = a
        if backward:
            for fi, a in propagate(backward).items():
                if alphas[fi] is None:
                    alphas[fi] = a

        if use_amp:
            # once at the end: hand the cached pool back (other nodes, and MatteDesk's ma_scale probe reads VRAM
            # from /system_stats after the prompt – it must not see our cache as "used")
            torch.cuda.empty_cache()
        alpha = torch.stack(alphas, 0)
        if alpha_only:
            dummy = torch.zeros((1, 1, 1, 3), dtype=images.dtype)
            return (alpha, dummy, dummy)
        matte = alpha.unsqueeze(-1).repeat(1, 1, 1, 3)
        fg = images * alpha.unsqueeze(-1)
        return (alpha, matte, fg)


from .vitmatte_refine import ViTMatteLoad, ViTMatteRefineAlpha
from .guided_filter import AlphaGuidedFilter
from .core_combine import MatAnyone2CoreCombine
from .foreground_decontam import ForegroundDecontaminate
from .edge_cleanup import EdgeCleanup
from .save_auto import SaveImageAuto

NODE_CLASS_MAPPINGS = {
    "AlphaGuidedFilter": AlphaGuidedFilter,
    "MatAnyone2CoreCombine": MatAnyone2CoreCombine,
    "ForegroundDecontaminate": ForegroundDecontaminate,
    "EdgeCleanup": EdgeCleanup,
    "SaveImageAuto": SaveImageAuto,
    "ViTMatteLoad": ViTMatteLoad,
    "ViTMatteRefineAlpha": ViTMatteRefineAlpha,
    "MatAnyone2SelectKeyframes": MatAnyone2SelectKeyframes,
    "MatAnyone2RunKeyframes": MatAnyone2RunKeyframes,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaGuidedFilter": "Alpha Guided Filter Edges (video)",
    "MatAnyone2CoreCombine": "MatAnyone2 + SAM3 Core (fill holes, stabilize edges)",
    "ForegroundDecontaminate": "Foreground Decontaminate (edge colours, pymatting)",
    "EdgeCleanup": "Edge Cleanup (bright/dark specks, despill, region)",
    "SaveImageAuto": "Save Image Auto (PNG for 8-bit source, EXR 32f for EXR source)",
    "ViTMatteLoad": "Load ViTMatte",
    "ViTMatteRefineAlpha": "ViTMatte Refine Alpha Edges (video)",
    "MatAnyone2SelectKeyframes": "MatAnyone2 Select Keyframes",
    "MatAnyone2RunKeyframes": "MatAnyone2 Video Matting (Keyframes)",
}
