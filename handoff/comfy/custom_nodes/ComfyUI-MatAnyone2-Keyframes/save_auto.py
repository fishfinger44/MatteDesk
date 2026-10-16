# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Save images as PNG or EXR depending on the source precision.

format "auto": if the frames look like 8-bit data (every value is a multiple of 1/255 and within 0..1)
the source was an 8-bit video/PNG -> save 8-bit PNG (EXR would only waste disk).
Otherwise (EXR/16-bit/float source, or values outside 0..1) -> save 32-bit float EXR, no compression,
values written 1:1 (no colour conversion).
Note: processing nodes (decontamination etc.) produce non-8-bit values even from an 8-bit source,
so the check is done on the SOURCE input when connected, otherwise on the images themselves.
"""
import os
import numpy as np
import torch
import folder_paths
from comfy_extras.nodes_images import _encode_image


def _looks_8bit(t: torch.Tensor) -> bool:
    s = t[: min(3, t.shape[0])].float().flatten()
    if s.numel() > 2_000_000:
        idx = torch.randint(0, s.numel(), (2_000_000,))
        s = s[idx]
    if s.min() < 0 or s.max() > 1:
        return False
    return bool(((s * 255 - (s * 255).round()).abs() < 1e-3).all())


class SaveImageAuto:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "MatAnyone2/out"}),
                "format": (["auto", "png", "exr"], {"default": "auto",
                           "tooltip": "auto = PNG for 8-bit sources (mp4/png), EXR 32-bit float (uncompressed) for EXR/16-bit sources."}),
            },
            "optional": {
                "source_video": ("VIDEO", {"tooltip": "Load Video output. 'auto' reads its bit depth from the file header (no decoding): 8-bit -> PNG, >8-bit -> EXR."}),
                "source": ("IMAGE", {"tooltip": "Original frames from an image sequence (EXR/PNG). Used by 'auto' when no video is connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "saved_as")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "video/matting"

    def save(self, images, filename_prefix, format, source_video=None, source=None):
        fmt = format
        if fmt == "auto":
            if source_video is not None:
                try:
                    fmt = "png" if int(source_video.get_bit_depth()) <= 8 else "exr"
                except Exception:
                    fmt = "png"
            else:
                fmt = "png" if _looks_8bit(source if source is not None else images) else "exr"
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
        results = []
        for i in range(images.shape[0]):
            img = images[i]
            if fmt == "png":
                data = _encode_image(img.clamp(0, 1), "png", "8-bit", "sRGB")
            else:
                data = _encode_image(img, "exr", "32-bit float", "linear")
            file = f"{filename}_{counter:05}.{fmt}"
            with open(os.path.join(out_dir, file), "wb") as f:
                f.write(data)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        print(f"[SaveImageAuto] {len(results)} frames -> {fmt.upper()}")
        return {"ui": {"images": results if fmt == "png" else []}, "result": (images, fmt)}
