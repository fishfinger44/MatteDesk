# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""PLAN_FINAL_TOUCH.md §11, step 0 / 5: does MergeMatte remove the white halos over navy? Test frames = hair at native
resolution (Hair 4820794 F36 + F26, 1080×1920, ViTMatte alpha) + CLIP_A F100 (fur, 2560×1440) as a control.
Panels "bez / MergeMatte (SH010 preset) / MergeMatte (defaults)" + a halo number per column.

    python tools/readme_compare/mm_measure.py [out_dir]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from engine import merge_matte as mm   # noqa: E402
from panels import NAVY_BGR, panel_row   # noqa: E402

NAVY_RGB = np.array(NAVY_BGR[::-1], np.float32)
HAIR_PLATE = Path("F:/claude_code/_fotage_testy/matte_desk/source/4820794-hd_1080_1920_25fps/4820794-hd_1080_1920_25fps_{f:04d}.png")
FRAMES = [
    {"name": "hair_f0036", "plate": Path(str(HAIR_PLATE).format(f=36)),
     "alpha": ROOT / "_readme_compare/c2_hair_f0036/alpha_4.png", "zoom": (268, 343, 605, 605)},          # ViTMatte column of c2
    {"name": "hair_f0026", "plate": Path(str(HAIR_PLATE).format(f=26)),
     "alpha": Path("F:/claude_code/_fotage_testy/matte_desk/objects/Hair/stage_preview/vitmatte/a_0026.png"), "zoom": (420, 150, 660, 660)},
    {"name": "clip_dog_f0100", "plate": Path("F:/claude_code/_fotage_testy/work/CLIP_A/plate/CLIP_A_0100.png"),
     "alpha": ROOT / "_readme_compare/c1_clip_dog_f0100/mask_crop.png", "zoom": (396, 847, 398, 398)},
]


def load(p: Path, color: bool) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    im = im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)
    return im[..., ::-1] if color else im


def halo(comp_rgb: np.ndarray, plate_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> dict:
    """Halo in the edge band (0.05 < α < 0.95): how much brighter the comp is than the ideal blend of the local
    interior colour with the BG (sRGB luminance, mean of the positive excess, 0..255) and the share of band pixels
    brighter than both the interior and the BG."""
    lum = lambda x: (x * mm.REC709).sum(-1)   # noqa: E731
    a = np.clip(alpha, 0, 1)
    band = (a > 0.05) & (a < 0.95)
    inside = cv2.GaussianBlur(plate_rgb * (a > 0.98)[..., None], (0, 0), 12) / \
        np.maximum(cv2.GaussianBlur((a > 0.98).astype(np.float32), (0, 0), 12), 1e-4)[..., None]
    ideal = lum(inside) * a + lum(bg_rgb) * (1 - a)
    excess = np.maximum(lum(comp_rgb) - ideal, 0)
    brighter = (lum(comp_rgb) > np.maximum(lum(inside), lum(bg_rgb)) + 0.02)
    return {"band_px": int(band.sum()), "halo_excess_mean_8bit": round(float(excess[band].mean() * 255), 2),
            "halo_px_share": round(float(brighter[band].mean()), 3)}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_readme_compare/c4_mergematte"
    out.mkdir(parents=True, exist_ok=True)
    report = {"preset_sh010": mm.PRESET_SH010, "defaults": mm.KNOB_DEFAULTS, "blur_sigma_per_size": mm.BLUR_SIGMA_PER_SIZE, "frames": {}}
    for fr in FRAMES:
        t0 = time.time()
        plate = load(fr["plate"], True)
        alpha = load(fr["alpha"], False)
        if alpha.shape != plate.shape[:2]:
            alpha = cv2.resize(alpha, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
        plain_lin = mm.acescg_to_srgb(mm.srgb_to_acescg(plate) * alpha[..., None]
                                      + mm.srgb_to_acescg(NAVY_RGB.reshape(1, 1, 3)) * (1 - alpha[..., None]))
        sh010 = mm.comp_srgb(plate, alpha, NAVY_RGB, mm.PRESET_SH010)
        default = mm.comp_srgb(plate, alpha, NAVY_RGB)
        cols = [("bez (over, linear)", plain_lin), ("MergeMatte – preset SH010", sh010), ("MergeMatte – domyślne", default)]
        bgr = [c[1][..., ::-1] for c in cols]
        panel_row(bgr, [c[0] for c in cols], out / f"{fr['name']}_zoom.jpg", zoom=fr["zoom"])
        panel_row(bgr, [c[0] for c in cols], out / f"{fr['name']}_full.jpg")
        m = {lab: halo(img, plate, alpha, NAVY_RGB) for lab, img in cols}
        m["seconds"] = round(time.time() - t0, 1)
        report["frames"][fr["name"]] = m
        print(fr["name"], json.dumps(m, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
