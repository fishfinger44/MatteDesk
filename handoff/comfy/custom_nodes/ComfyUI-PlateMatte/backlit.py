# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Matte from a backlit second exposure (X2): wall bright, objects block the light.

coverage = 1 - lum(X2) / lum(wall behind), with the wall estimated from the bright X2 pixels by a
normalised blur at two scales (fills holes where objects cover the wall, also at the frame edge).
Defocused near objects are not black in X2 (leaves transmit, trunks are partly front-lit), so the
coverage is divided by its local maximum, and the interior is filled to 1 (eroded so soft edges stay).
Physically correct defocus softness, no thresholds on the edge.
Ported 1:1 from the Nuke branch BD_FG_DEFOCUS (SH010), where it was tuned.
"""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils

LUM = (0.2722, 0.6741, 0.0537)


def _gauss1d(sigma, dev):
    r = max(1, int(round(sigma * 3)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), r


def gblur(x, sigma):
    """x: N,1,H,W. Separable gaussian; large sigmas computed at reduced resolution."""
    if sigma <= 0:
        return x
    H, W = x.shape[-2:]
    s = max(1, int(sigma // 6))
    y = F.interpolate(x, scale_factor=1 / s, mode="area") if s > 1 else x
    k, r = _gauss1d(sigma / s, x.device)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    if s > 1:
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return y


def dilate(x, px):
    if px <= 0:
        return x
    s = max(1, px // 10)
    H, W = x.shape[-2:]
    y = F.interpolate(x, scale_factor=1 / s, mode="area") if s > 1 else x
    rr = max(1, px // s)
    y = F.max_pool2d(y, 2 * rr + 1, stride=1, padding=rr)
    return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False) if s > 1 else y


def erode(x, px):
    return -dilate(-x, px) if px > 0 else x


class BacklitCoverage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backlit": ("IMAGE", {"tooltip": "Backlit exposure (X2), scene-linear, e.g. from Plate Load EXR."}),
                "wall_lo": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.05,
                                      "tooltip": "X2 luminance above which a pixel counts as wall (fully from wall_lo+0.4)."}),
                "wall_sigma_small": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 500.0}),
                "wall_sigma_large": ("FLOAT", {"default": 200.0, "min": 1.0, "max": 2000.0}),
                "max_radius": ("INT", {"default": 40, "min": 0, "max": 300,
                                       "tooltip": "Local 'solid' level = max coverage within this radius (leaf transmits ~25% in X2)."}),
                "floor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                                    "tooltip": "Never divide by less than this (keeps faint tails faint)."}),
                "noise_lo": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.3, "step": 0.005}),
                "fill_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Interior (alpha above this, eroded) is forced to 1."}),
                "fill_erode": ("INT", {"default": 15, "min": 0, "max": 200}),
            },
            "optional": {
                "wall": ("IMAGE", {"tooltip": "Wall luminance estimated outside (MatteDesk: on the FULL X2 frame, no mirror "
                                              "padding) – same crop as `backlit`; replaces the in-node wall estimate."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("alpha", "raw_coverage", "wall_debug")
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    @torch.inference_mode()
    def run(self, backlit, wall_lo, wall_sigma_small, wall_sigma_large, max_radius, floor, noise_lo, fill_threshold, fill_erode,
            wall=None):
        dev = mm.get_torch_device()
        N, H, W, _ = backlit.shape
        if wall is not None and tuple(wall.shape[1:3]) != (H, W):
            raise ValueError(f"BacklitCoverage: wall {tuple(wall.shape[1:3])} != backlit {(H, W)}")
        outs = [torch.empty((N, H, W)) for _ in range(3)]
        lw = torch.tensor(LUM, device=dev).view(1, 3, 1, 1)
        pbar = comfy.utils.ProgressBar(N)
        B = 4
        for i in range(0, N, B):
            x = backlit[i:i + B].to(dev).permute(0, 3, 1, 2).float()
            lum = (x * lw).sum(1, keepdim=True)
            m = ((lum - wall_lo) / 0.4).clamp(0, 1)
            a1, b1 = gblur(lum * m, wall_sigma_small), gblur(m, wall_sigma_small)
            a2, b2 = gblur(lum * m, wall_sigma_large), gblur(m, wall_sigma_large)
            if wall is None:
                wall_i = (a1 + 0.02 * a2) / (b1 + 0.02 * b2).clamp(min=1e-5)
            else:   # external wall (full-frame estimate): luminance of the given image, the node's blur is skipped
                wall_i = (wall[i:i + B].to(dev).permute(0, 3, 1, 2).float() * lw).sum(1, keepdim=True)
            known = ((b1 + 0.02 * b2) * 50).clamp(0, 1)
            wall_use = wall_i
            cov = torch.where(known > 0.001, (1 - lum / wall_use.clamp(min=0.3)).clamp(0, 1), torch.ones_like(lum))
            cmax = gblur(dilate(cov, max_radius), max_radius * 0.25)
            norm = ((cov - noise_lo) / torch.maximum(cmax - noise_lo, torch.full_like(cmax, floor - noise_lo))).clamp(0, 1)
            fill = gblur(erode((norm > fill_threshold).float(), fill_erode), 3)
            alpha = torch.maximum(norm, fill)
            for o, v in zip(outs, (alpha, cov, (wall_use / 4).clamp(0, 1))):
                o[i:i + B] = v[:, 0].cpu()
            pbar.update(min(B, N - i))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return tuple(outs)


class MaskRegionLimit:
    """Keep a matte only inside one object's region (selective mattes): mask * soft(dilate(region))."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "region": ("MASK", {"tooltip": "Object silhouette (e.g. SAM3 track), any resolution."}),
                "grow_px": ("INT", {"default": 40, "min": 0, "max": 400, "tooltip": "Cover the soft defocus tail."}),
                "soft_px": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 200.0}),
                "grow_into_tail_only": ("BOOLEAN", {"default": True,
                                                    "tooltip": "Grow the region only into the soft edge tail (mask < ~1), never into neighbouring OPAQUE areas (floor band, other objects). There the region stays at the silhouette edge."}),
            },
            "optional": {
                "subtract": ("MASK", {"tooltip": "Other objects in front (their core is removed)."}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "region")
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    @torch.inference_mode()
    def run(self, mask, region, grow_px, soft_px, grow_into_tail_only=True, subtract=None):
        dev = mm.get_torch_device()
        N, H, W = mask.shape
        out = torch.empty_like(mask)
        reg_out = torch.empty_like(mask)
        for i in range(N):
            r = region[min(i, region.shape[0] - 1)].to(dev).float()[None, None]
            if r.shape[-2:] != (H, W):
                r = F.interpolate(r, size=(H, W), mode="bilinear", align_corners=False)
            rb = (r > 0.5).float()
            grown = gblur(dilate(rb, grow_px), soft_px)
            if grow_into_tail_only:
                m = mask[i].to(dev).float()[None, None]
                tail = ((0.995 - m) / 0.03).clamp(0, 1)          # 1 in the soft tail, 0 where opaque
                near = gblur(dilate(rb, 10), 3.0)                   # region masks are a few px off -> always include a 10 px band
                r = torch.maximum(near, grown * tail)
            else:
                r = grown
            if subtract is not None:
                s = subtract[min(i, subtract.shape[0] - 1)].to(dev).float()[None, None]
                if s.shape[-2:] != (H, W):
                    s = F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)
                r = r * (1 - (s > 0.5).float())
            reg_out[i] = r[0, 0].cpu()
            out[i] = (mask[i].to(dev) * r[0, 0]).cpu()
        return (out, reg_out)
