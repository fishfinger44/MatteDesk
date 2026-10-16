# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guided-filter edge refinement for a video alpha (He et al. 2010).

Snaps the soft alpha edge to the image's own edges. Deterministic per frame and cheap,
applied only in a narrow band around the existing edge so the interior stays untouched.
"""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils


def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def guided_filter(guide, src, r, eps):
    """guide: [1,C,H,W] (C=1 or 3), src: [1,1,H,W]."""
    if guide.shape[1] == 3:
        guide = (0.299 * guide[:, 0:1] + 0.587 * guide[:, 1:2] + 0.114 * guide[:, 2:3])
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    var_i = _box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


class AlphaGuidedFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha": ("MASK",),
                "radius": ("INT", {"default": 4, "min": 1, "max": 32,
                                   "tooltip": "Filter radius in px. Bigger = follows larger image structures, softer."}),
                "eps": ("FLOAT", {"default": 0.001, "min": 0.00001, "max": 0.1, "step": 0.0001,
                                  "tooltip": "Edge sensitivity. Smaller = snaps harder to image edges; larger = smoother."}),
                "band_px": ("INT", {"default": 6, "min": 0, "max": 64,
                                    "tooltip": "Only change alpha within this many px of the soft edge (0 = whole frame)."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha_refined", "matte_video")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    @torch.inference_mode()
    def run(self, images, alpha, radius, eps, band_px):
        device = mm.get_torch_device()
        out = []
        pbar = comfy.utils.ProgressBar(alpha.shape[0])
        for i in range(alpha.shape[0]):
            a = alpha[i].to(device).float()[None, None]
            g = images[i, ..., :3].to(device).float().permute(2, 0, 1)[None]
            r = guided_filter(g, a, radius, eps).clamp(0, 1)
            if band_px > 0:
                edge = ((a > 0.01) & (a < 0.99)).float()
                band = F.max_pool2d(edge, 2 * band_px + 1, stride=1, padding=band_px) > 0.5
                r = torch.where(band, r, a)
            out.append(r[0, 0].cpu())
            pbar.update(1)
        res = torch.stack(out, 0)
        return (res, res.unsqueeze(-1).repeat(1, 1, 1, 3))
