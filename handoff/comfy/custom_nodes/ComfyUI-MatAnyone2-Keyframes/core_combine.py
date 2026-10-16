# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Combine a MatAnyone2 alpha with a solid "core" mask from SAM3 video tracking (GPU, batched).

MatAnyone2 gives great soft edges but can punch false holes inside the person
(e.g. bright hair highlights read as background). SAM3 tracking gives a solid
silhouette per frame. We erode the SAM3 mask so only the interior remains, feather it,
and take max(alpha, core): interior holes are filled, true edges/hair stay from MatAnyone2.
Optionally smooth the alpha over time, only in the soft edge band (kills slow edge "swimming").

core_mode "soft" (1.1.2, MatteDesk soft start): core_masks may carry SAM3 probabilities. A frame with intermediate
values uses them directly as the core (no threshold / erode / feather); a strictly binary frame keeps the old path.
"""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils


def _erode(m, px):
    return m if px <= 0 else -F.max_pool2d(-m, 2 * px + 1, stride=1, padding=px)


def _dilate(m, px):
    return m if px <= 0 else F.max_pool2d(m, 2 * px + 1, stride=1, padding=px)


def _blur(m, px):
    if px <= 0:
        return m
    k = 2 * px + 1
    for _ in range(2):
        m = F.avg_pool2d(m, k, stride=1, padding=px, count_include_pad=False)
    return m


class MatAnyone2CoreCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha": ("MASK", {"tooltip": "MatAnyone2 alpha (soft edges)."}),
                "core_masks": ("MASK", {"tooltip": "Per-frame solid person masks, e.g. SAM3 Video Track -> Track to Mask."}),
                "core_erode_px": ("INT", {"default": 3, "min": 0, "max": 100,
                                          "tooltip": "Shrink the SAM3 mask a little (its edge is blocky)."}),
                "core_feather_px": ("INT", {"default": 2, "min": 0, "max": 50}),
                "gate_px": ("INT", {"default": 12, "min": 0, "max": 100,
                                    "tooltip": "Fill only where the neighbourhood (this radius) is mostly person = holes. True soft edges (background next to them) stay from MatAnyone2."}),
                "gate_low": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "gate_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temporal_sigma": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 8.0, "step": 0.5,
                                             "tooltip": "Temporal smoothing of the edge band, in frames (0 = off). Keep low for fast motion."}),
                "edge_band_px": ("INT", {"default": 6, "min": 0, "max": 50,
                                         "tooltip": "Width around the soft edge where temporal smoothing is applied."}),
                "motion_limit": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
                                           "tooltip": "No smoothing where alpha changes more than this within the window = real motion (prevents ghosting). Flicker is small, motion is big."}),
            },
            "optional": {
                "core_mode": (["binary", "soft"], {"default": "binary",
                                                   "tooltip": "soft: frames whose core mask has intermediate values (SAM3 probabilities) use it as is; binary frames as before."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "matte_video")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    @torch.inference_mode()
    def run(self, alpha, core_masks, core_erode_px, core_feather_px, gate_px, gate_low, gate_high, temporal_sigma, edge_band_px, motion_limit=0.35,
            core_mode="binary"):
        dev = mm.get_torch_device()
        N, H, W = alpha.shape
        c = core_masks
        if c.shape[-2:] != (H, W):
            c = F.interpolate(c.float()[:, None], size=(H, W), mode="nearest")[:, 0]
        if c.shape[0] != N:
            c = c[:1].repeat(N, 1, 1) if c.shape[0] == 1 else c[:N]
        pbar = comfy.utils.ProgressBar(3)
        B = 16   # klatek na raz (VRAM)

        # 1) rdzen z SAM3 -> max z alfa (wsadowo na GPU)
        out = torch.empty((N, H, W), dtype=torch.float16)
        band = torch.empty((N, H, W), dtype=torch.bool)
        for i in range(0, N, B):
            a = alpha[i:i + B].float().to(dev)[:, None]
            cb = c[i:i + B].to(dev).float()[:, None]
            core = _blur(_erode((cb > 0.5).float(), core_erode_px), core_feather_px)
            if core_mode == "soft":   # frames with intermediate values = probabilities: used directly
                soft = ((cb > 0.002) & (cb < 0.998)).flatten(1).any(1).view(-1, 1, 1, 1)
                core = torch.where(soft, cb.clamp(0, 1), core)
            if gate_px > 0:   # bramka: wypelniaj tylko tam, gdzie wokol przewaza postac (dziury), nie na prawdziwej krawedzi
                g = ((_blur(a, gate_px) - gate_low) / max(gate_high - gate_low, 1e-3)).clamp(0, 1)
                core = core * g * g * (3 - 2 * g)
            o = torch.maximum(a, core)
            out[i:i + B] = o[:, 0].half().cpu()
            band[i:i + B] = (_dilate(((o > 0.02) & (o < 0.98)).float(), edge_band_px) > 0)[:, 0].cpu()
        pbar.update(1)

        # 2) wygladzenie w czasie tylko w pasie krawedzi (conv1d po osi czasu, blokami wierszy)
        if temporal_sigma > 0:
            r = int(round(temporal_sigma * 3))
            t = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
            k = torch.exp(-0.5 * (t / temporal_sigma) ** 2); k = (k / k.sum()).view(1, 1, -1)
            RB = 64   # wierszy na blok
            for y in range(0, H, RB):
                blk = out[:, y:y + RB].to(dev).float()                        # N,h,W
                h = blk.shape[1]
                series = blk.permute(1, 2, 0).reshape(-1, 1, N)                # h*W,1,N
                sm = F.conv1d(F.pad(series, (r, r), mode="replicate"), k)      # h*W,1,N
                sm = sm.reshape(h, W, N).permute(2, 0, 1)
                bd = band[:, y:y + RB].to(dev).float()
                bd = F.max_pool1d(bd.permute(1, 2, 0).reshape(-1, 1, N), 2 * r + 1, stride=1, padding=r).reshape(h, W, N).permute(2, 0, 1) > 0
                # ruch: zakres alfy w oknie czasowym; duzy = prawdziwy ruch -> nie wygladzac (bez duchow)
                sp = F.pad(series, (r, r), mode="replicate")
                rng = (F.max_pool1d(sp, 2 * r + 1, stride=1) + F.max_pool1d(-sp, 2 * r + 1, stride=1)).reshape(h, W, N).permute(2, 0, 1)
                calm = (1 - ((rng - motion_limit * 0.6) / (motion_limit * 0.4)).clamp(0, 1))
                sm = blk + (sm - blk) * calm
                out[:, y:y + RB] = torch.where(bd, sm, blk).half().cpu()
            pbar.update(1)

        res = out.float().clamp(0, 1)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        pbar.update(1)
        return (res, res.unsqueeze(-1).repeat(1, 1, 1, 3))
