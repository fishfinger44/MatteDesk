# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers of the README comparisons: comp over navy (RGB always from the plate, only the matte differs),
labelled side-by-side panels, edge metrics measured against the plate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAVY_BGR = (0x4a / 255, 0x2a / 255, 0x1b / 255)   # #1b2a4a
ACCENT = (232, 163, 61)                           # MatteDesk amber, RGB


def decontaminate(img: np.ndarray, alpha: np.ndarray, size: float, amount: float = 1.0) -> np.ndarray:
    """SD_Fuse "Edge Decontaminate" (D:/MD_REAL_PROJECT/assets/nuke_scripts/SD_Fuse_v01.nk): premult -> Blur(size) -> unpremult
    = interior colour spread outward, keymixed into the plate RGB by (1 - alpha) × amount. Alpha unchanged.
    Nuke Blur `size` ~ gaussian sigma = size / 2.4 (to be matched against Nuke before export uses it)."""
    a = np.clip(alpha, 0, 1).astype(np.float32)
    s = max(size / 2.4, 0.1)
    num = cv2.GaussianBlur(img * a[..., None], (0, 0), s)
    den = cv2.GaussianBlur(a, (0, 0), s)[..., None]
    spread = np.where(den > 1e-4, num / np.maximum(den, 1e-4), img)
    m = ((1 - a) * amount)[..., None]
    return img * (1 - m) + spread * m


def unmix(img: np.ndarray, alpha: np.ndarray, size: float, min_a: float = 0.08) -> np.ndarray:
    """Background unmix: B = background colour next to the edge (plate weighted by 1 - alpha, blurred), then
    F = (I - (1 - a) B) / a, kept inside the interior's colour range; below min_a the interior spread (decontaminate)."""
    a = np.clip(alpha, 0, 1).astype(np.float32)
    s = max(size / 2.4, 0.1)
    bw = (1 - a) * (a < 0.98)
    b = cv2.GaussianBlur(img * bw[..., None], (0, 0), s) / np.maximum(cv2.GaussianBlur(bw, (0, 0), s), 1e-4)[..., None]
    f = (img - (1 - a)[..., None] * b) / np.maximum(a, min_a)[..., None]
    solid = a > 0.98
    lo = np.percentile(img[solid], 1, axis=0) if solid.any() else 0.0
    hi = np.percentile(img[solid], 99, axis=0) if solid.any() else 1.0
    f = np.clip(f, lo, hi)
    spread = decontaminate(img, a, size)
    w = np.clip((a - min_a) / min_a, 0, 1)[..., None]
    out = f * w + spread * (1 - w)
    return np.where(solid[..., None], img, out)


def navy_comp(img: np.ndarray, alpha: np.ndarray, decon_px: float = 0.0, mode: str = "decon") -> np.ndarray:
    """img BGR 0..1, alpha 0..1 -> BGR 0..1 over navy; decon_px > 0 = edge colour fix first
    (mode "decon" = SD_Fuse Edge Decontaminate, "unmix" = background unmix)."""
    a = np.clip(alpha, 0, 1)
    if decon_px > 0:
        img = (unmix if mode == "unmix" else decontaminate)(img, a, decon_px)
    a = a[..., None]
    return img * a + np.array(NAVY_BGR, np.float32) * (1 - a)


def _font(px: int):
    for f in ("IBMPlexSans-Medium.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            pass
    return ImageFont.load_default()


def panel_row(imgs: Sequence[np.ndarray], labels: Sequence[str], path: Path, zoom: Optional[Tuple[int, int, int, int]] = None,
              width: int = 2400, gap: int = 6, label_h: int = 56) -> None:
    """Images (BGR 0..1, same size) side by side, a label bar over each, optional zoom box (x, y, w, h) in image px."""
    n = len(imgs)
    cw = (width - gap * (n - 1)) // n
    tiles = []
    for im in imgs:
        if zoom:
            x, y, w, h = zoom
            im = im[y:y + h, x:x + w]
        ch = int(round(im.shape[0] * cw / im.shape[1]))
        interp = cv2.INTER_AREA if cw < im.shape[1] else cv2.INTER_LANCZOS4
        tiles.append(np.clip(cv2.resize(im, (cw, ch), interpolation=interp), 0, 1))
    ch = tiles[0].shape[0]
    canvas = Image.new("RGB", (width, ch + label_h), (24, 24, 26))
    draw = ImageDraw.Draw(canvas)
    font = _font(int(label_h * 0.5))
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x0 = i * (cw + gap)
        rgb = (t[..., ::-1] * 255 + 0.5).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb), (x0, label_h))
        draw.text((x0 + 18, label_h / 2), lab, fill=ACCENT if i == n - 1 else (225, 225, 225), font=font, anchor="lm")
    canvas.save(str(path), quality=92)


def edge_metrics(img: np.ndarray, alpha: np.ndarray) -> dict:
    """How well the matte outline sits on the real image edge (plate luminance):
    - grad_on_outline: mean Sobel magnitude of the plate under the outline (higher = on a real edge),
    - holes: enclosed background islands > 20 px inside the object, soft_px: pixels 0.02..0.98."""
    m = (alpha > 0.5).astype(np.uint8)
    grey = cv2.GaussianBlur(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), (3, 3), 0)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    outline = (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    holes = sum(1 for i in range(1, n) if st[i, 4] > 20 and st[i, 0] > 0 and st[i, 1] > 0
                and st[i, 0] + st[i, 2] < w and st[i, 1] + st[i, 3] < h)
    return {"area_px": int(m.sum()), "outline_px": int(outline.sum()),
            "grad_on_outline": round(float(mag[outline].mean()), 2), "holes": int(holes), "soft_px": int(((alpha > 0.02) & (alpha < 0.98)).sum())}
