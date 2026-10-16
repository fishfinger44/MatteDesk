# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/backlit_wall.py - Z25 pkt 1: the backlit (X2) wall estimated on the FULL frame, not on the mirrored crop.

`BacklitCoverage` (PlateMatte) estimates the wall behind an object by a normalised two-scale blur of the bright X2
pixels of the image it gets. The Finalizer feeds it the square crop of the object with mirror padding at the frame
edges (engine/crop.extract_crop), so for a defocused object at the frame edge (SH010: tree LL from F146, monstera
leaf F158-193) the mirrored object pixels and the small crop leave the wall without data -> a hard, flickering blob.

Here the wall is estimated once per frame on the whole X2 frame (downscaled, sigmas scaled), with NO data outside the
frame (zero-padded normalised convolution: pixels outside the frame do not vote), then cut to the crop and handed to
the node (optional `wall` input; the coverage / normalisation / fill stay in the node, 1:1 as before). The formula is
the node's: m = clamp((lum - wall_lo) / 0.4), wall = (B_s(lum m) + 0.02 B_l(lum m)) / (B_s(m) + 0.02 B_l(m)).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

LUM = (0.2722, 0.6741, 0.0537)       # the node's luminance weights (R, G, B)


def luminance(bgr: np.ndarray) -> np.ndarray:
    return (LUM[0] * bgr[..., 2] + LUM[1] * bgr[..., 1] + LUM[2] * bgr[..., 0]).astype(np.float32)


def _blur0(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with ZERO outside the image (normalised-convolution numerator / denominator)."""
    if sigma <= 0:
        return x
    k = int(round(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_CONSTANT)


def _blur_rep(x: np.ndarray, sigma: float) -> np.ndarray:
    """The node's behaviour: replicate border (used only for the crop-based reference / tests)."""
    if sigma <= 0:
        return x
    k = int(round(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)


def wall_estimate(lum: np.ndarray, wall_lo: float = 0.9, sigma_small: float = 12.0, sigma_large: float = 200.0,
                  scale: float = 0.25, border: str = "zero") -> Tuple[np.ndarray, np.ndarray]:
    """Wall luminance (full resolution of `lum`) and `known` (0..1 wall-data density), estimated at `scale`.
    border="zero": nothing outside the image votes (full-frame use); "replicate": the node's crop behaviour."""
    h, w = lum.shape
    s = min(1.0, float(scale))
    small = cv2.resize(lum, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else lum
    m = np.clip((small - wall_lo) / 0.4, 0.0, 1.0).astype(np.float32)
    blur = _blur0 if border == "zero" else _blur_rep
    a1, b1 = blur(small * m, sigma_small * s), blur(m, sigma_small * s)
    a2, b2 = blur(small * m, sigma_large * s), blur(m, sigma_large * s)
    den = np.maximum(b1 + 0.02 * b2, 1e-5)
    wall_s = (a1 + 0.02 * a2) / den
    known_s = np.clip((b1 + 0.02 * b2) * 50.0, 0.0, 1.0)
    if s < 1.0:
        wall = cv2.resize(wall_s, (w, h), interpolation=cv2.INTER_LINEAR)
        known = cv2.resize(known_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        wall, known = wall_s, known_s
    return wall.astype(np.float32), known.astype(np.float32)


def wall_full_frame(x2_bgr: np.ndarray, params: Optional[Dict[str, Any]] = None, scale: float = 0.25) -> np.ndarray:
    p = params or {}
    wall, _ = wall_estimate(luminance(x2_bgr), float(p.get("wall_lo", 0.9)), float(p.get("wall_sigma_small", 12.0)),
                            float(p.get("wall_sigma_large", 200.0)), scale=scale, border="zero")
    return wall


def coverage_from_wall(lum: np.ndarray, wall: np.ndarray, known: Optional[np.ndarray] = None) -> np.ndarray:
    """The node's raw coverage: 1 - lum / max(wall, 0.3) where wall data exist, 1 elsewhere."""
    cov = np.clip(1.0 - lum / np.maximum(wall, 0.3), 0.0, 1.0)
    if known is not None:
        cov = np.where(known > 0.001, cov, 1.0)
    return cov.astype(np.float32)
