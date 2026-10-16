# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""EXR plate sequence I/O for scene-linear (ACES) footage.

Values are kept exactly as stored (no clamp, no colour conversion), so a matte/colour computed here
lines up 1:1 with the same plates read in Nuke. A crop window keeps 4K float sequences in RAM budget;
PLATE_INFO remembers where the crop sits so the saver can write full-frame EXRs again.
"""
import os
import re
import glob
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import OpenEXR
import Imath
import comfy.utils

_FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
_HALF = Imath.PixelType(Imath.PixelType.HALF)


def _frame_path(pattern, frame):
    """pattern uses #### (any count of #) or %04d for the frame number."""
    if "#" in pattern:
        m = re.search(r"#+", pattern)
        return pattern[:m.start()] + str(frame).zfill(len(m.group(0))) + pattern[m.end():]
    return pattern % frame


def read_exr(path, crop=None):
    """Return float32 HxWx3 RGB in display-window coordinates (missing data = 0) and full (W, H)."""
    f = OpenEXR.InputFile(path)
    h = f.header()
    dw, dp = h["dataWindow"], h["displayWindow"]
    W, H = dp.max.x - dp.min.x + 1, dp.max.y - dp.min.y + 1
    w, hh = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    chans = h["channels"]
    names = []
    for c in "RGB":
        names.append(c if c in chans else next((k for k in chans if k.endswith("." + c)), None))
    full = np.zeros((H, W, 3), np.float32)
    y0, x0 = dw.min.y - dp.min.y, dw.min.x - dp.min.x
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(H, y0 + hh), min(W, x0 + w)
    for i, n in enumerate(names):
        if n is None:
            continue
        a = np.frombuffer(f.channel(n, _FLOAT), np.float32).reshape(hh, w)
        full[ys:ye, xs:xe, i] = a[ys - y0:ye - y0, xs - x0:xe - x0]
    f.close()
    if crop:
        cx, cy, cw, ch = crop
        full = full[cy:cy + ch, cx:cx + cw]
    return full, (W, H)


def write_exr(path, rgba, half=True):
    """rgba: HxWx4 float32 (premultiplied). Writes R,G,B,A, ZIP compression."""
    H, W = rgba.shape[:2]
    hdr = OpenEXR.Header(W, H)
    pt = _HALF if half else _FLOAT
    hdr["channels"] = {c: Imath.Channel(pt) for c in "RGBA"}
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    out = OpenEXR.OutputFile(path, hdr)
    dt = np.float16 if half else np.float32
    out.writePixels({c: np.ascontiguousarray(rgba[..., i]).astype(dt).tobytes() for i, c in enumerate("RGBA")})
    out.close()


class PlateLoadEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_pattern": ("STRING", {"default": "C:/shots/SHOT/plates/plate_####.exr",
                                            "tooltip": "Frame number as #### (or %04d)."}),
                "first_frame": ("INT", {"default": 1, "min": 0, "max": 100000}),
                "last_frame": ("INT", {"default": 1, "min": 0, "max": 100000}),
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "From the TOP of the frame."}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 = full width"}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 = full height"}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                                    "tooltip": "Downscale on load (RAM). The saver scales results back up to the crop size."}),
                "bad_frames": ("STRING", {"default": "", "tooltip": "Broken plate frames, e.g. '162' or '40, 162': replaced by the mean of the previous and next frame."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PLATE_INFO")
    RETURN_NAMES = ("images", "plate_info")
    FUNCTION = "load"
    CATEGORY = "PlateMatte"
    DESCRIPTION = "Loads a scene-linear EXR sequence untouched (no clamp / colour conversion), optionally cropped."

    def load(self, path_pattern, first_frame, last_frame, crop_x, crop_y, crop_w, crop_h, scale=1.0, bad_frames=""):
        bad = {int(v) for v in bad_frames.replace(";", ",").split(",") if v.strip()}
        frames = list(range(first_frame, last_frame + 1))
        _, (W, H) = read_exr(_frame_path(path_pattern, frames[0]), crop=(0, 0, 1, 1))
        cw = crop_w or (W - crop_x)
        ch = crop_h or (H - crop_y)
        crop = (crop_x, crop_y, min(cw, W - crop_x), min(ch, H - crop_y))
        ow, oh = (crop[2], crop[3]) if scale >= 1.0 else (max(1, round(crop[2] * scale)), max(1, round(crop[3] * scale)))
        arr = np.empty((len(frames), oh, ow, 3), dtype=np.float32)
        pbar = comfy.utils.ProgressBar(len(frames))

        def job(i_f):   # numpy only: worker threads are outside torch inference mode
            i, f = i_f
            if f in bad:   # broken plate frame -> mean of the neighbours
                img = 0.5 * (read_exr(_frame_path(path_pattern, f - 1), crop=crop)[0] + read_exr(_frame_path(path_pattern, f + 1), crop=crop)[0])
            else:
                img, _ = read_exr(_frame_path(path_pattern, f), crop=crop)
            arr[i] = img if scale >= 1.0 else cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            pbar.update(1)

        with ThreadPoolExecutor(6) as ex:
            list(ex.map(job, enumerate(frames)))
        out = torch.from_numpy(arr)
        info = {"frames": frames, "full_w": W, "full_h": H, "crop": crop, "pattern": path_pattern, "scale": min(scale, 1.0)}
        return (out, info)


class PlateToDisplay:
    """Scene-linear -> 0..1 preview (soft tonemap) for SAM / previews. Not for the final output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "white": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                                    "tooltip": "Linear value that maps to ~mid-high grey. Raise for bright plates."}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Downscale factor."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "PlateMatte"

    def run(self, images, white, scale):
        x = images.clamp(min=0) / white
        x = (x / (x + 0.6) * 1.6).clamp(0, 1) ** (1 / 1.6)
        if scale < 1.0:
            x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), scale_factor=scale, mode="area").permute(0, 2, 3, 1)
        return (x.contiguous(),)


class PlateSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba": ("IMAGE", {"tooltip": "Premultiplied RGBA at crop resolution (4 channels), or RGB + alpha input."}),
                "plate_info": ("PLATE_INFO",),
                "out_pattern": ("STRING", {"default": "C:/shots/SHOT/mattes/obj_####.exr"}),
                "half_float": ("BOOLEAN", {"default": True}),
            },
            "optional": {"alpha": ("MASK",)},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "PlateMatte"
    DESCRIPTION = "Writes full-frame EXRs (crop pasted back at its position, zero elsewhere), frame numbers = plate numbers."

    def save(self, rgba, plate_info, out_pattern, half_float, alpha=None):
        os.makedirs(os.path.dirname(out_pattern), exist_ok=True)
        cx, cy, cw, ch = plate_info["crop"]
        W, H = plate_info["full_w"], plate_info["full_h"]
        frames = plate_info["frames"]
        pbar = comfy.utils.ProgressBar(len(frames))

        def job(i):
            px = rgba[i].cpu().numpy()
            if px.shape[-1] == 3:
                a = alpha[min(i, alpha.shape[0] - 1)].cpu().numpy() if alpha is not None else np.ones(px.shape[:2], np.float32)
                px = np.concatenate([px, a[..., None]], -1)
            if px.shape[:2] != (ch, cw):   # loaded downscaled -> back to crop size
                px = cv2.resize(px, (cw, ch), interpolation=cv2.INTER_LINEAR)
            full = np.zeros((H, W, 4), np.float32)
            full[cy:cy + px.shape[0], cx:cx + px.shape[1]] = px
            write_exr(_frame_path(out_pattern, frames[i]), full, half=half_float)
            pbar.update(1)

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(job, range(len(frames))))
        return {}
