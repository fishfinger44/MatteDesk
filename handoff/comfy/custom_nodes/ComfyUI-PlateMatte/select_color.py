# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Object selection by frame-edge connectivity + scene-linear foreground colour (decontamination)."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import comfy.utils


class MaskEdgeComponents:
    """Keep the parts of a matte that touch a frame edge (e.g. a foreground tree entering from the left).
    A band at the bottom (table/floor) can be ignored, because it connects everything to everything."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "plate_info": ("PLATE_INFO", {"tooltip": "So 'touches the edge' means the real frame edge, not the crop edge."}),
                "edge": (["left", "right", "top", "bottom"], {"default": "left"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "ignore_bottom_px": ("INT", {"default": 300, "min": 0, "max": 4000,
                                             "tooltip": "Rows from the bottom of the FULL frame ignored for connectivity (table / floor)."}),
                "min_area_px": ("INT", {"default": 500, "min": 0, "max": 10_000_000}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("region",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, mask, plate_info, edge, threshold, ignore_bottom_px, min_area_px):
        cx, cy, cw, ch = plate_info["crop"]
        W, H = plate_info["full_w"], plate_info["full_h"]
        N, h, w = mask.shape
        m_all = mask.cpu().numpy()
        out = np.zeros_like(m_all)
        cut = max(0, min(h, H - ignore_bottom_px - cy))
        touches = {"left": cx == 0, "right": cx + cw >= W, "top": cy == 0, "bottom": cy + ch >= H and ignore_bottom_px == 0}
        if not touches[edge]:
            raise ValueError(f"The crop does not reach the {edge} frame edge.")

        def job(i):
            b = (m_all[i] > threshold).astype(np.uint8)
            b[cut:] = 0
            n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
            border = {"left": lab[:, 0], "right": lab[:, -1], "top": lab[0, :], "bottom": lab[cut - 1, :] if cut else lab[-1, :]}[edge]
            ids = [k for k in np.unique(border) if k != 0 and stats[k, cv2.CC_STAT_AREA] >= min_area_px]
            out[i] = np.isin(lab, ids).astype(np.float32)

        with ThreadPoolExecutor(6) as ex:
            list(ex.map(job, range(N)))
        return (torch.from_numpy(out),)


class PlateForegroundHDR:
    """True foreground colour for semi-transparent pixels (pymatting ML foreground estimation),
    scene-linear: no clamp to 1, so bright highlights survive. Output premultiplied RGBA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Beauty plate (X1), scene-linear."}),
                "alpha": ("MASK",),
                "regularization": ("FLOAT", {"default": 1e-5, "min": 1e-7, "max": 1e-2, "step": 1e-6}),
                "alpha_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.1, "step": 0.001,
                                        "tooltip": "Below this alpha the output is 0 (no colour smear outside the matte)."}),
                "colour_inset_px": ("INT", {"default": 8, "min": 0, "max": 60,
                                            "tooltip": "Bright background blooms a few px INTO the object in the beauty plate. The colour is estimated with the matte shrunk by this much, so the edge band takes the object's interior colour (the output alpha is unchanged)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_premult",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, images, alpha, regularization, alpha_min, colour_inset_px=8):
        from pymatting import estimate_foreground_ml
        N = images.shape[0]
        imgs = images.cpu().numpy()
        al = alpha.cpu().numpy()
        out = np.zeros(imgs.shape[:3] + (4,), np.float32)
        pbar = comfy.utils.ProgressBar(N)

        def job(i):
            a = al[min(i, al.shape[0] - 1)].astype(np.float64)
            ys, xs = np.nonzero(a > alpha_min)
            if len(ys) == 0:
                pbar.update(1)
                return
            # work only on the bounding box of the matte (+margin) - much faster on a mostly empty crop
            m = 64
            y0, y1 = max(0, ys.min() - m), min(a.shape[0], ys.max() + m + 1)
            x0, x1 = max(0, xs.min() - m), min(a.shape[1], xs.max() + m + 1)
            img = np.clip(imgs[i, y0:y1, x0:x1].astype(np.float64), 0, None)
            aa = a[y0:y1, x0:x1]
            # pymatting clips F to [0, 1] -> scale HDR into range and back (linear problem, exact)
            s = max(1.0, float(img.max()))
            fg = estimate_foreground_ml(img / s, aa, regularization=regularization) * s
            fg = np.clip(fg, 0, None).astype(np.float32)
            if colour_inset_px > 0:
                # bloom band: replace the colour near the edge by the interior colour spread outward
                # (normalised blur of the solid core, two scales so far tails are filled too)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * colour_inset_px + 1, 2 * colour_inset_px + 1))
                core = cv2.erode((aa > 0.98).astype(np.float32), k)
                spread = np.zeros_like(fg)
                den_acc = np.zeros(core.shape, np.float32)
                for s, wgt in ((colour_inset_px * 1.5, 1.0), (colour_inset_px * 6.0, 0.02)):
                    den = cv2.GaussianBlur(core, (0, 0), s)
                    num = cv2.GaussianBlur(fg * core[..., None], (0, 0), s)
                    spread += num * wgt
                    den_acc += den * wgt
                spread = spread / np.maximum(den_acc, 1e-6)[..., None]
                keep = cv2.GaussianBlur(core, (0, 0), 1.5)[..., None]
                fg = fg * keep + spread * (1 - keep)
            ab = aa.astype(np.float32) * (aa > alpha_min)
            out[i, y0:y1, x0:x1, :3] = fg * ab[..., None]
            out[i, y0:y1, x0:x1, 3] = ab
            pbar.update(1)

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(job, range(N)))
        return (torch.from_numpy(out),)
