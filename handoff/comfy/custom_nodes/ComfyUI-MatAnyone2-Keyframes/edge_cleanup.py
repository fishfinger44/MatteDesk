# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edge Cleanup: fix bright/dark specks and colour spill on a matted person, optionally only inside a region.

Typical use: hair against a bright background leaves whitish strands (old background seen between strands).
Give it the foreground (e.g. after Foreground Decontaminate), the alpha, and a region mask (e.g. SAM3 "hair").
Each pixel is compared to the local colour of its "normal" neighbours (radius ref_radius):
  - bright: noticeably lighter and less saturated than the neighbourhood (white streaks)
  - dark:   noticeably darker (dark fringes)
  - despill: remove blue or green cast
Action "recolor" paints them with the local colour; "cut" makes them transparent (alpha down).
Everything is computed from the image (not from the matte), so it is stable over time.
"""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils


def _box(m, r):
    return m if r <= 0 else F.avg_pool2d(m, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def _maxp(m, r):
    return m if r <= 0 else F.max_pool2d(m, 2 * r + 1, stride=1, padding=r)


def _ss(x, a, b):
    t = ((x - a) / max(b - a, 1e-4)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class EdgeCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Foreground frames (e.g. output of Foreground Decontaminate)."}),
                "alpha": ("MASK",),
                "action": (["recolor", "cut"], {"default": "recolor",
                           "tooltip": "recolor = paint with local hair colour, cut = make transparent."}),
                "bright_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bright_threshold": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                                               "tooltip": "How much lighter than the neighbourhood (luma) before it counts."}),
                "dark_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dark_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "despill": (["off", "blue", "green"], {"default": "off"}),
                "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "edge_zone_px": ("INT", {"default": 30, "min": 0, "max": 300,
                                         "tooltip": "Only within this distance from the alpha edge (0 = whole region/person)."}),
                "ref_radius": ("INT", {"default": 12, "min": 2, "max": 60,
                                       "tooltip": "Radius of the neighbourhood whose colour is used as reference."}),
                "protect_large_px": ("INT", {"default": 0, "min": 0, "max": 40,
                                             "tooltip": "Ignore bright/dark blobs wider than ~2x this (protects e.g. a tablet). 0 = off (use with a region mask)."}),
                "region_grow_px": ("INT", {"default": 4, "min": 0, "max": 50}),
            },
            "optional": {
                "region": ("MASK", {"tooltip": "Work only inside this mask (e.g. SAM3 'hair'). Empty = whole person."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("images", "alpha", "fix_map")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    @torch.inference_mode()
    def run(self, images, alpha, action, bright_strength, bright_threshold, dark_strength, dark_threshold,
            despill, despill_strength, edge_zone_px, ref_radius, protect_large_px, region_grow_px, region=None):
        dev = mm.get_torch_device()
        N, H, W = alpha.shape
        LW = torch.tensor([0.299, 0.587, 0.114], device=dev)
        out_img = torch.empty_like(images[..., :3]); out_a = torch.empty_like(alpha); fmap = torch.empty_like(alpha)
        pbar = comfy.utils.ProgressBar(N)
        B = 8
        for i in range(0, N, B):
            rgb = images[i:i + B, ..., :3].to(dev).float()                 # b,H,W,3
            a = alpha[i:i + B].to(dev).float()[:, None]                   # b,1,H,W
            b = rgb.shape[0]
            lum = (rgb @ LW)[:, None]
            mx = rgb.max(-1).values[:, None]; mn = rgb.min(-1).values[:, None]
            sat = (mx - mn) / (mx + 1e-4)
            # obszar dzialania
            area = (a > 0.6).float()
            if edge_zone_px > 0:
                area = area * _maxp((a < 0.5).float(), edge_zone_px)
            if region is not None:
                reg = region[min(i, region.shape[0] - 1):min(i, region.shape[0] - 1) + b].to(dev).float()
                if reg.shape[0] < b:
                    reg = reg[:1].repeat(b, 1, 1)
                if reg.shape[-2:] != (H, W):
                    reg = F.interpolate(reg[:, None], size=(H, W), mode="bilinear")[:, 0]
                area = area * _maxp((reg > 0.5).float()[:, None], region_grow_px)
            # kolor odniesienia: "zwykle" piksele sasiedztwa (bez odstajacych jasnych/ciemnych)
            loc = _box(lum, ref_radius)
            normal = (((lum - loc).abs() < 0.08) & (a > 0.9)).float()
            wsum = _box(normal, ref_radius) + 1e-4
            ref = torch.cat([_box(rgb[..., c][:, None] * normal, ref_radius) for c in range(3)], 1) / wsum   # b,3,H,W
            reflum = (ref * LW.view(1, 3, 1, 1)).sum(1, keepdim=True)
            w = torch.zeros_like(lum)
            if bright_strength > 0:
                wb = _ss(lum - reflum, bright_threshold, bright_threshold + 0.14) * (1 - _ss(sat, 0.35, 0.5))
                if protect_large_px > 0:
                    blob = (lum - reflum > bright_threshold).float()
                    wb = wb * (1 - _maxp(-_maxp(-blob, protect_large_px), protect_large_px + 4))
                w = torch.maximum(w, wb * bright_strength)
            if dark_strength > 0:
                wd = _ss(reflum - lum, dark_threshold, dark_threshold + 0.14)
                if protect_large_px > 0:
                    blob = (reflum - lum > dark_threshold).float()
                    wd = wd * (1 - _maxp(-_maxp(-blob, protect_large_px), protect_large_px + 4))
                w = torch.maximum(w, wd * dark_strength)
            w = _box(w * area * (wsum > 0.05).float(), 1)
            rgbc = rgb.permute(0, 3, 1, 2)
            if action == "recolor":
                rgbc = rgbc * (1 - w) + ref * w
                a_out = a
            else:
                a_out = a * (1 - w)
            if despill != "off" and despill_strength > 0:
                r_, g_, b_ = rgbc[:, 0:1], rgbc[:, 1:2], rgbc[:, 2:3]
                if despill == "blue":
                    nb = torch.minimum(b_, torch.maximum(r_, g_)); rgbc = torch.cat([r_, g_, b_ + (nb - b_) * despill_strength * area], 1)
                else:
                    ng = torch.minimum(g_, (r_ + b_) / 2); rgbc = torch.cat([r_, g_ + (ng - g_) * despill_strength * area, b_], 1)
            out_img[i:i + b] = rgbc.permute(0, 2, 3, 1).clamp(0, 1).cpu()
            out_a[i:i + b] = a_out[:, 0].clamp(0, 1).cpu()
            fmap[i:i + b] = w[:, 0].cpu()
            pbar.update(b)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return (out_img, out_a, fmap)
