# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Foreground colour decontamination (pymatting "Fast Multi-Level Foreground Estimation").

Semi-transparent pixels (hair strands) in a real shot contain a mix of the person and the old
background (e.g. a bright atrium). Composited over a new background they show a whitish fringe.
This recovers the true foreground colour F for every pixel given image I and alpha a (I = aF + (1-a)B),
so the edge takes the colour of the hair, not of the old background. Deterministic per frame.
"""
import numpy as np
import torch
import comfy.utils


class ForegroundDecontaminate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha": ("MASK",),
                "regularization": ("FLOAT", {"default": 1e-5, "min": 1e-7, "max": 1e-2, "step": 1e-6,
                                             "tooltip": "Smoothness of the estimated foreground (pymatting default 1e-5)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "foreground_rgba")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    def run(self, images, alpha, regularization):
        from pymatting import estimate_foreground_ml
        N = images.shape[0]
        fg = torch.empty_like(images[..., :3])
        pbar = comfy.utils.ProgressBar(N)
        for i in range(N):
            img = images[i, ..., :3].cpu().numpy().astype(np.float64)
            a = alpha[min(i, alpha.shape[0] - 1)].cpu().numpy().astype(np.float64)
            f = estimate_foreground_ml(img, a, regularization=regularization)
            fg[i] = torch.from_numpy(np.clip(f, 0, 1).astype(np.float32))
            pbar.update(1)
        rgba = torch.cat([fg, alpha[:N, ..., None].to(fg.dtype)], dim=-1)
        return (fg, rgba)
