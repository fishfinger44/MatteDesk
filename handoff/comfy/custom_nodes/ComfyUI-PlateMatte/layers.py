# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-object layers: load region mask sequences (any resolution, full frame) and depth-order subtraction."""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .plate_io import _frame_path, read_exr


def _exr_alpha(p):
    import OpenEXR, Imath
    f = OpenEXR.InputFile(p)
    h = f.header()
    dw, dp = h["dataWindow"], h["displayWindow"]
    W, H = dp.max.x - dp.min.x + 1, dp.max.y - dp.min.y + 1
    w, hh = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    c = "A" if "A" in h["channels"] else "R"
    a = np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(hh, w)
    full = np.zeros((H, W), np.float32)
    y0, x0 = dw.min.y - dp.min.y, dw.min.x - dp.min.x
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(H, y0 + hh), min(W, x0 + w)
    full[ys:ye, xs:xe] = a[ys - y0:ye - y0, xs - x0:xe - x0]
    return full


class PlateLoadMaskSeq:
    """Full-frame mask sequence (PNG/JPG/EXR, any size) -> MASK at the plate crop (resized to the plate size first)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_pattern": ("STRING", {"default": "C:/shots/SHOT/masks/mask_####.png"}),
                "plate_info": ("PLATE_INFO",),
                "frame_offset": ("INT", {"default": 0, "min": -100000, "max": 100000,
                                         "tooltip": "mask frame = plate frame + offset"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load"
    CATEGORY = "PlateMatte"

    def load(self, path_pattern, plate_info, frame_offset):
        W, H = plate_info["full_w"], plate_info["full_h"]
        cx, cy, cw, ch = plate_info["crop"]
        sc = plate_info.get("scale", 1.0)
        ow, oh = (cw, ch) if sc >= 1.0 else (max(1, round(cw * sc)), max(1, round(ch * sc)))
        frames = plate_info["frames"]
        out = np.zeros((len(frames), oh, ow), np.float32)

        def job(i):
            p = _frame_path(path_pattern, frames[i] + frame_offset)
            if not os.path.exists(p):
                return
            if p.lower().endswith(".exr"):
                m = _exr_alpha(p)                 # A channel (premult RGBA or matte EXRs)
            else:
                m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if m.ndim == 3:
                    m = m[..., -1] if m.shape[2] == 4 else m[..., 0]
                m = m.astype(np.float32) / (255.0 if m.max() > 1 else 1.0)
            if m.shape[:2] == (ch, cw) and (ch, cw) != (H, W):
                pass                                   # already crop-sized (e.g. per-crop guide keys)
            else:
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                m = m[cy:cy + ch, cx:cx + cw]
            out[i] = m if (ow, oh) == (cw, ch) else cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)

        with ThreadPoolExecutor(6) as ex:
            list(ex.map(job, range(len(frames))))
        return (torch.from_numpy(out),)


class MatteSubtractFront:
    """Visible part of an object behind others: clamp(mask - max(front mattes)).
    With precise front mattes the boundary in the overlap is as precise as the front object's edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"mask": ("MASK",)},
            "optional": {"front_1": ("MASK",), "front_2": ("MASK",), "front_3": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask, front_1=None, front_2=None, front_3=None):
        out = mask.clone()
        for fr in (front_1, front_2, front_3):
            if fr is not None:
                out = (out - fr[:out.shape[0]]).clamp(min=0)
        return (out,)


class MaskTemporalMedian:
    """Per-pixel median over +-radius frames: removes 1-2 frame segmentation glitches (an extra blob,
    a missing hand) from region masks. Use on REGIONS/cores, not on final soft edges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "radius": ("INT", {"default": 2, "min": 1, "max": 6})},
                "optional": {"region": ("MASK", {"tooltip": "Apply only inside this region (soft), e.g. the static nest; elsewhere the mask is untouched."})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask, radius, region=None):
        N = mask.shape[0]
        out = torch.empty_like(mask)
        for i in range(N):
            idx = [min(N - 1, max(0, j)) for j in range(i - radius, i + radius + 1)]
            med = mask[idx].float().median(dim=0).values
            if region is not None:
                r = region[min(i, region.shape[0] - 1)].float()
                if r.shape != med.shape:
                    r = F.interpolate(r[None, None], size=med.shape, mode="bilinear", align_corners=False)[0, 0]
                med = mask[i].float() * (1 - r) + med * r
            out[i] = med
        return (out,)


class MaskTemporalMedianMC:
    """Motion-compensated temporal median for (near-)RIGID objects (e.g. a nest glued to a trunk): every
    neighbour frame's mask is aligned to the current frame (ECC affine on the plate, inside the region bbox),
    then the per-pixel median of +-radius aligned masks. Kills edge jumps lasting up to ~radius frames while
    following the camera move. Applied only inside 'region' (soft); elsewhere the mask is untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "images": ("IMAGE", {"tooltip": "Display-range plate for alignment (same size as mask)."}),
                             "region": ("MASK",), "radius": ("INT", {"default": 12, "min": 1, "max": 30})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask, images, region, radius):
        N, H, W = mask.shape
        m = mask.cpu().numpy().astype(np.float32)
        g = (images[..., :3].mean(-1).cpu().numpy() * 255).astype(np.float32)
        reg = region.cpu().numpy().astype(np.float32)
        if reg.shape[-2:] != (H, W):
            reg = np.stack([cv2.resize(r, (W, H)) for r in reg])
        ys, xs = np.nonzero(reg.max(0) > 0.1)
        pad = 80
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)
        s = 0.5
        small = [cv2.resize(gi[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for gi in g]
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        out = m.copy()
        S = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        Si = np.linalg.inv(S)
        Ds, Dsi = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
        for i in range(N):
            stack = []
            for j in range(max(0, i - radius), min(N, i + radius + 1)):
                if j == i:
                    stack.append(m[i]); continue
                warp = np.eye(2, 3, dtype=np.float32)
                try:
                    _, warp = cv2.findTransformECC(small[i], small[j], warp, cv2.MOTION_AFFINE, crit, None, 5)
                except cv2.error:
                    continue
                A = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
                Afull = Si @ Dsi @ A @ Ds @ S               # maps frame-i coords -> frame-j coords (full res)
                stack.append(cv2.warpAffine(m[j], Afull[:2].astype(np.float32), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
            med = np.median(np.stack(stack), 0)
            r = reg[min(i, len(reg) - 1)]
            out[i] = m[i] * (1 - r) + med * r
        return (torch.from_numpy(out),)


class MaskRecoverFromGuide:
    """Recover thin parts the matte loses (e.g. horn tips) from a plate-based GUIDE (clean-plate difference key):
    guide pixels outside the matte are added only if they are CONNECTED to the matte (geodesic growth through the
    guide, up to grow_px) and inside the zone (top zone_px rows of the object, or an optional zone mask) - shadows,
    grain and background noise are not connected to the object and stay out. Added alpha = guide value (soft)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "guide": ("MASK",),
                             "guide_thr": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
                             "grow_px": ("INT", {"default": 60, "min": 1, "max": 400}),
                             "zone_top_px": ("INT", {"default": 420, "min": 0, "max": 5000, "tooltip": "Only the top N px of the object (0 = everywhere)."}),
                             "zone_bottom_px": ("INT", {"default": 0, "min": 0, "max": 5000, "tooltip": "Only the bottom N px of the object (+ grow below it), e.g. a hole under a nest. 0 = off."}),
                             "min_area_px": ("INT", {"default": 40, "min": 0, "max": 100000})},
                "optional": {"zone": ("MASK",)}}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "added")
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask, guide, guide_thr, grow_px, zone_top_px, min_area_px, zone_bottom_px=0, zone=None):
        N, H, W = mask.shape
        m_all = mask.cpu().numpy(); g_all = guide.cpu().numpy()
        out = m_all.copy(); added = np.zeros_like(m_all)
        k3 = np.ones((3, 3), np.uint8)
        for i in range(N):
            m = m_all[i]; g = g_all[min(i, len(g_all) - 1)]
            if g.shape != m.shape:
                g = cv2.resize(g, (W, H))
            mb = (m > 0.5).astype(np.uint8)
            if not mb.any():
                continue
            z = np.ones((H, W), np.uint8)
            if zone is not None:
                zz = zone[min(i, zone.shape[0] - 1)].cpu().numpy()
                z = (cv2.resize(zz, (W, H)) > 0.5).astype(np.uint8) if zz.shape != m.shape else (zz > 0.5).astype(np.uint8)
            elif zone_bottom_px > 0:
                bot = int(np.nonzero(mb.any(1))[0].max()); z[:] = 0; z[max(0, bot - zone_bottom_px):bot + grow_px + 1] = 1
            elif zone_top_px > 0:
                top = int(np.nonzero(mb.any(1))[0].min()); z[:] = 0; z[max(0, top - grow_px):top + zone_top_px] = 1
            cand = ((g > guide_thr) & (mb == 0) & (z > 0)).astype(np.uint8)
            grown = mb.copy()
            for _ in range(grow_px):                         # geodesic growth through the guide
                nxt = cv2.dilate(grown, k3) & (cand | mb)
                if (nxt == grown).all():
                    break
                grown = nxt
            add = (grown & (1 - mb)).astype(np.uint8)
            if min_area_px > 0 and add.any():
                n, lab, st, _ = cv2.connectedComponentsWithStats(add, connectivity=8)
                add = np.isin(lab, [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= min_area_px]).astype(np.uint8)
            # seamless join: blur the UNION (old matte + added) so the old soft edge between them is filled,
            # applied only near the added area (elsewhere the old matte is untouched)
            union = cv2.GaussianBlur((add | mb).astype(np.float32), (0, 0), 1.0)
            near = cv2.GaussianBlur(cv2.dilate(add, np.ones((17, 17), np.uint8)).astype(np.float32), (0, 0), 2.0)
            out[i] = np.maximum(m, union * near)
            added[i] = cv2.GaussianBlur(add.astype(np.float32), (0, 0), 1.0)
        return (torch.from_numpy(out), torch.from_numpy(added))


class MaskMaxMerge:
    """Union of up to 6 masks (max)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask_1": ("MASK",)},
                "optional": {f"mask_{i}": ("MASK",) for i in range(2, 7)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask_1, **kw):
        out = mask_1.clone()
        for m in kw.values():
            if m is not None:
                if m.shape[-2:] != out.shape[-2:]:
                    m = F.interpolate(m[:, None].float(), size=out.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                out = torch.maximum(out, m[:out.shape[0]])
        return (out,)
