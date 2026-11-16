# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""README comparison 1: SAM 3.1 on the whole frame vs on MatteDesk's tight square crop – one frame, same model,
same 1008 px input, same prompt. Writes masks, metrics.json and the navy comp panels.

    python tools/readme_compare/crop_vs_full.py <plate.png> <out_dir> [--prompt woman] [--port 8001]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyClient          # noqa: E402
from engine.crop import square_crop           # noqa: E402
from engine.sam3 import SAM_RES, Sam3Runner   # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from panels import edge_metrics, navy_comp, panel_row   # noqa: E402


def pick(masks, ref=None):
    """Largest mask, or the one overlapping `ref` most."""
    if not masks:
        raise RuntimeError("SAM 3.1 found nothing")
    if ref is None:
        return max(masks, key=lambda m: float((m > 0.5).sum()))
    return max(masks, key=lambda m: float(((m > 0.5) & ref).sum()))


def run_full(sam, img, prompt):
    """Whole frame letterboxed to a square, down to SAM_RES, mask back to plate size."""
    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), np.float32)
    oy, ox = (s - h) // 2, (s - w) // 2
    sq[oy:oy + h, ox:ox + w] = img
    small = cv2.resize(sq, (SAM_RES, SAM_RES), interpolation=cv2.INTER_CUBIC)
    m = pick(sam.detect(small, prompt=prompt))
    big = cv2.resize(m, (s, s), interpolation=cv2.INTER_LINEAR)
    return big[oy:oy + h, ox:ox + w], {"input_side_px": s, "scale_to_sam": SAM_RES / s}


def run_crop(sam, img, prompt, ref):
    """MatteDesk crop: tight square around the object (engine.crop.square_crop), down to SAM_RES, back to the plate."""
    h, w = img.shape[:2]
    ys, xs = np.where(ref)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    box, pad = square_crop((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0), w, h)
    x, y, side, _ = box
    crop = np.zeros((side, side, 3), np.float32)
    sx0, sy0, sx1, sy1 = max(x, 0), max(y, 0), min(x + side, w), min(y + side, h)
    crop[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = img[sy0:sy1, sx0:sx1]
    small = cv2.resize(crop, (SAM_RES, SAM_RES), interpolation=cv2.INTER_CUBIC)
    ref_small = cv2.resize(np.pad(ref.astype(np.float32), ((max(-y, 0), 0), (max(-x, 0), 0)))[
        max(y, 0):max(y, 0) + side, max(x, 0):max(x, 0) + side], (SAM_RES, SAM_RES)) > 0.5
    m = pick(sam.detect(small, prompt=prompt), ref_small)
    back = cv2.resize(m, (side, side), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((h, w), np.float32)
    full[sy0:sy1, sx0:sx1] = back[sy0 - y:sy1 - y, sx0 - x:sx1 - x]
    return full, {"input_side_px": side, "scale_to_sam": SAM_RES / side, "crop": box}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plate")
    ap.add_argument("out")
    ap.add_argument("--prompt", default="woman")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--zoom", default="", help="x,y,w,h zoom box in plate px (default: top of the object)")
    a = ap.parse_args()
    out = Path(a.out).resolve()   # Comfy reads the scratch frames: absolute paths
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(a.plate, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    sam = Sam3Runner(ComfyClient("127.0.0.1", a.port), out / "scratch", io_mode="png8")

    full, info_f = run_full(sam, img, a.prompt)
    crop, info_c = run_crop(sam, img, a.prompt, full > 0.5)
    for n, m in (("full", full), ("crop", crop)):
        cv2.imwrite(str(out / f"mask_{n}.png"), (np.clip(m, 0, 1) * 255).astype(np.uint8))

    ys, xs = np.where(crop > 0.5)
    if a.zoom:
        zx, zy, zw, zh = (int(v) for v in a.zoom.split(","))
    else:   # the head: top quarter of the object, square
        zw = zh = int((ys.max() - ys.min()) * 0.28)
        zx, zy = int(xs[ys == ys.min()].mean() - zw / 2), int(ys.min() - zh * 0.12)
    zoom = (max(zx, 0), max(zy, 0), zw, zh)

    metrics = {"plate": Path(a.plate).name, "plate_wh": [img.shape[1], img.shape[0]], "prompt": a.prompt,
               "zoom": zoom, "full": {**info_f, **edge_metrics(img, full)}, "crop": {**info_c, **edge_metrics(img, crop)}}
    obj_h = int(ys.max() - ys.min())
    for k, inf in (("full", info_f), ("crop", info_c)):
        metrics[k]["object_height_in_sam_px"] = round(obj_h * inf["scale_to_sam"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

    comps = [navy_comp(img, m) for m in (full, crop)]
    panel_row(comps, ["SAM 3.1 on the whole frame", "SAM 3.1 on the MatteDesk crop"], out / "panel_full.jpg", width=2400)
    panel_row([img] + comps, ["Plate", "Whole frame", "MatteDesk crop"], out / "panel_zoom.jpg", zoom=zoom, width=2400)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
