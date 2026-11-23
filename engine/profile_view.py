# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/profile_view.py - the plate seen through a colour profile curve (user request 2026-11-20).

VIEW ONLY: the viewer can show a plate frame encoded with a lin → log curve of engine/log_curves.py (the curve of
the stage-0 colour profile, or any other) – what MergeMatte's „Log” operations and the .nk export work on. The models
never see this: they keep the fixed PlateToDisplay image (engine/ingest.plate_to_display_float), so no stage result
and no cache depends on it.

    view_rgb(path, colorspace, curve_id) -> uint8 RGB   the frame through the curve (0..1 clipped, like a log viewer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from engine import log_curves as lc


def is_linear(colorspace: str) -> bool:
    """The same rule as ProxyGenerator.load_frame_rgb: ACES / linear / raw EXR data is scene-linear."""
    return any(t in str(colorspace or "").upper() for t in ("ACES", "LINEAR", "RAW"))


def linear_rgb(path: Union[str, Path], colorspace: str) -> np.ndarray:
    """Scene-linear float32 RGB of a plate frame: linear EXR as stored, display-referred sources (8-bit images, a
    non-linear EXR) decoded with the sRGB EOTF (MatteDesk's 8-bit plates are display sRGB)."""
    p = Path(path)
    if p.suffix.lower() == ".exr":
        from engine.exr_io import read_half_rgb
        bgr = read_half_rgb(p)
        if bgr is None:
            bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise FileNotFoundError(f"Nie można odczytać klatki {p}")
            bgr = bgr[..., :3]
        rgb = np.ascontiguousarray(bgr[..., ::-1]).astype(np.float32)
        return rgb if is_linear(colorspace) else lc.LOG_CURVES["video"].log_to_lin(np.clip(rgb, 0.0, 1.0))
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie można odczytać klatki {p}")
    return lc.LOG_CURVES["video"].log_to_lin(bgr[..., ::-1].astype(np.float32) / 255.0)


def view_rgb(path: Union[str, Path], colorspace: str, curve_id: str) -> np.ndarray:
    """uint8 RGB of the frame encoded with the curve `curve_id` (lin → log, clipped to 0..1)."""
    curve = lc.LOG_CURVES.get(curve_id)
    if curve is None:
        raise ValueError(f"Nieznana krzywa podglądu: {curve_id!r} (jedna z: {', '.join(lc.LOG_CURVES)})")
    enc = curve.lin_to_log(linear_rgb(path, colorspace))
    return (np.clip(np.nan_to_num(enc, nan=0.0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
