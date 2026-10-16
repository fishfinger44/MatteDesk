# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ViTMatte edge refinement on top of a video alpha (e.g. MatAnyone2).

Builds a narrow trimap band around the existing alpha edge, runs ViTMatte only inside the
band's bounding box, and replaces alpha only inside the band. Everything else keeps the
temporally stable MatAnyone2 alpha.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import comfy.model_management as mm
import comfy.utils

VITMATTE_DIR = os.path.join(folder_paths.models_dir, "vitmatte")
os.makedirs(VITMATTE_DIR, exist_ok=True)
REPOS = {
    "vitmatte-base-composition-1k": "hustvl/vitmatte-base-composition-1k",
    "vitmatte-small-composition-1k": "hustvl/vitmatte-small-composition-1k",
}


def _dilate(m, px):
    if px <= 0:
        return m
    k = 2 * px + 1
    return F.max_pool2d(m[None, None].float(), k, stride=1, padding=px)[0, 0] > 0.5


class ViTMatteLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (list(REPOS.keys()), {"default": "vitmatte-base-composition-1k"})}}

    RETURN_TYPES = ("VITMATTE_MODEL",)
    RETURN_NAMES = ("vitmatte",)
    FUNCTION = "load"
    CATEGORY = "video/matting"

    def load(self, model_name):
        from transformers import VitMatteForImageMatting
        from huggingface_hub import snapshot_download
        local = os.path.join(VITMATTE_DIR, model_name)
        if not os.path.exists(os.path.join(local, "config.json")):
            snapshot_download(REPOS[model_name], local_dir=local)
        device = mm.get_torch_device()
        model = VitMatteForImageMatting.from_pretrained(local).to(device).eval()
        return ({"model": model, "device": device},)


class ViTMatteRefineAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vitmatte": ("VITMATTE_MODEL",),
                "images": ("IMAGE",),
                "alpha": ("MASK",),
                "band_px": ("INT", {"default": 8, "min": 1, "max": 64,
                                    "tooltip": "Width of the 'unknown' band around the edge that ViTMatte recomputes (px at clip resolution)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Mix inside the band: 1 = ViTMatte, 0 = original alpha."}),
                "temporal_smooth": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                                              "tooltip": "Blend refined band with previous frame's refined band to fight per-frame flicker. 0 = off. High values lag on fast motion."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("alpha_refined", "matte_video", "trimap_band")
    FUNCTION = "refine"
    CATEGORY = "video/matting"

    @torch.inference_mode()
    def refine(self, vitmatte, images, alpha, band_px, strength, temporal_smooth):
        model, device = vitmatte["model"], vitmatte["device"]
        N, H, W = alpha.shape
        out, bands = [], []
        prev = None
        pbar = comfy.utils.ProgressBar(N)
        for i in range(N):
            a = alpha[i].to(device).float()
            edge = (a > 0.02) & (a < 0.98)
            hard = a > 0.5
            unknown = _dilate(edge | (_dilate(hard, 1) ^ hard), band_px)  # soft edge + 1px contour, widened
            if not unknown.any():
                out.append(a.cpu()); bands.append(unknown.float().cpu()); prev = None; pbar.update(1); continue

            ys, xs = torch.where(unknown)
            m = 64
            y0, y1 = max(0, int(ys.min()) - m), min(H, int(ys.max()) + m + 1)
            x0, x1 = max(0, int(xs.min()) - m), min(W, int(xs.max()) + m + 1)

            img = images[i, y0:y1, x0:x1, :3].to(device).permute(2, 0, 1).float()
            tri = torch.where(unknown, torch.full_like(a, 0.5), (a > 0.5).float())[y0:y1, x0:x1]
            h, w = img.shape[1:]
            ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
            x = torch.cat([(img - 0.5) / 0.5, tri[None]], 0)[None]
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(pixel_values=x).alphas[0, 0, :h, :w].float().clamp(0, 1)

            band = unknown[y0:y1, x0:x1]
            new_band = pred
            if temporal_smooth > 0 and prev is not None:
                new_band = (1 - temporal_smooth) * pred + temporal_smooth * prev[y0:y1, x0:x1]
            full_pred = a.clone()
            crop = full_pred[y0:y1, x0:x1]
            mixed = (1 - strength) * crop + strength * new_band
            crop[band] = mixed[band]
            full_pred[y0:y1, x0:x1] = crop
            prev = full_pred
            out.append(full_pred.cpu())
            bands.append(unknown.float().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(1)

        alpha_r = torch.stack(out, 0)
        return (alpha_r, alpha_r.unsqueeze(-1).repeat(1, 1, 1, 3), torch.stack(bands, 0))
