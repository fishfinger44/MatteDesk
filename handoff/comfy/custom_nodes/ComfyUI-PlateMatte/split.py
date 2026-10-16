# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Split one physical matte between several objects WITHOUT seams (partition of unity).

Every matte pixel gets one owner: inside the (slightly grown) object regions the FRONTMOST object owns it
(depth order = input order), outside all regions the NEAREST region owns it (soft tails, gaps between
masks). Ownership is softened over a few px and renormalised, so the per-object mattes always add up
exactly to the input matte -> no grey seam where objects touch, no object stealing a neighbour's edge.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils

from .backlit import gblur, dilate

MAXR = 8


def geodesic_grow(seed, inside, px):
    """Grow seed by up to px, only through pixels where inside > 0 (connected to the seed). Half res for speed."""
    if px <= 0:
        return seed
    H, W = seed.shape[-2:]
    s = F.interpolate(seed, scale_factor=0.5, mode="area")
    ins = (F.interpolate(inside, scale_factor=0.5, mode="area") > 0.2).float()   # low: thin twigs survive the half-res step
    for _ in range(max(1, px // 2)):
        s = torch.maximum(s, F.max_pool2d(s, 3, stride=1, padding=1) * ins)
    return F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)


class MatteSplitByRegions:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"region_{i}": ("MASK",) for i in range(2, MAXR + 1)}
        return {
            "required": {
                "matte": ("MASK", {"tooltip": "The physical matte to split (e.g. Backlit Coverage)."}),
                "region_1": ("MASK", {"tooltip": "FRONTMOST object region. Then region_2, ... further back."}),
                "near_px": ("INT", {"default": 10, "min": 0, "max": 100, "tooltip": "Regions are a few px off: always own this band."}),
                "grow_px": ("INT", {"default": 40, "min": 0, "max": 2000, "tooltip": "Geodesic growth through the matte: object parts/tails up to this far from a region (e.g. a long side branch) belong to the nearest object."}),
                "soft_px": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "tooltip": "Softness of the ownership boundary between objects."}),
                "limit_soft_px": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 60.0}),
                "min_island_px": ("INT", {"default": 600, "min": 0, "max": 1_000_000,
                                          "tooltip": "Remove isolated matte specks smaller than this (full-res px), e.g. wall noise blobs."}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("MASK",) * (MAXR + 1)
    RETURN_NAMES = ("union",) + tuple(f"object_{i}" for i in range(1, MAXR + 1))
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    @torch.inference_mode()
    def run(self, matte, region_1, near_px, grow_px, soft_px, limit_soft_px, min_island_px=600, **kw):
        dev = mm.get_torch_device()
        regions = [region_1] + [kw.get(f"region_{i}") for i in range(2, MAXR + 1)]
        regions = [r for r in regions if r is not None]
        K = len(regions)
        N, H, W = matte.shape
        outs = [torch.zeros((N, H, W)) for _ in range(MAXR + 1)]
        ds = 4   # nearest-owner search at 1/4 res
        pbar = comfy.utils.ProgressBar(N)
        for i in range(N):
            m = matte[i].to(dev).float()[None, None]
            R = []
            for r in regions:
                x = r[min(i, r.shape[0] - 1)].to(dev).float()[None, None]
                if x.shape[-2:] != (H, W):
                    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
                R.append((x > 0.5).float())
            P = [dilate(x, near_px) > 0.5 for x in R]
            # owner inside regions: frontmost wins
            owner = torch.full((H, W), -1, dtype=torch.long, device=dev)
            for k in range(K - 1, -1, -1):          # back to front, front overwrites
                owner[P[k][0, 0]] = k
            # owner outside: nearest region (distance transform at 1/4 res)
            small = [cv2.resize((x[0, 0].cpu().numpy() > 0.5).astype(np.uint8), (W // ds, H // ds), interpolation=cv2.INTER_NEAREST) for x in R]
            dist = np.stack([cv2.distanceTransform(1 - s, cv2.DIST_L2, 3) if s.any() else np.full(s.shape, 1e9, np.float32) for s in small])
            near = torch.from_numpy(dist.argmin(0)).to(dev)[None, None].float()
            near = F.interpolate(near, size=(H, W), mode="nearest")[0, 0].long()
            owner = torch.where(owner < 0, near, owner)
            # allowed area: near band always + geodesic growth through the matte (connected object pixels,
            # opaque or soft) up to grow_px: catches object parts outside an undersized region mask
            anyr = torch.clamp(sum(R), 0, 1)
            allowed = torch.maximum(dilate(anyr, near_px), geodesic_grow(anyr, (m > 0.03).float(), grow_px))
            allowed = gblur(allowed, limit_soft_px).clamp(0, 1)
            # soft ownership weights, renormalised (partition of unity)
            Wk = [gblur((owner == k).float()[None, None], soft_px) for k in range(K)]
            S = torch.clamp(sum(Wk), min=1e-6)
            base = m * allowed
            if min_island_px > 0:   # drop isolated specks (connected components at 1/4 res)
                b = cv2.resize((base[0, 0] > 0.05).byte().cpu().numpy(), (W // ds, H // ds), interpolation=cv2.INTER_NEAREST)
                n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
                small_ids = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] * ds * ds < min_island_px]
                if small_ids:
                    kill = cv2.resize(np.isin(lab, small_ids).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    kill = cv2.dilate(kill, np.ones((2 * ds + 1, 2 * ds + 1), np.uint8))
                    base = base * (1 - torch.from_numpy(kill).to(dev).float())[None, None]
            outs[0][i] = base[0, 0].cpu()
            for k in range(K):
                outs[k + 1][i] = (base * Wk[k] / S)[0, 0].cpu()
            pbar.update(1)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return tuple(outs)
