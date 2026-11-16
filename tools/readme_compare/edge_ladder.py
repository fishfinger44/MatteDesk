# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""README comparison 2: one object, one frame, every edge stage – read-only from a computed project.
Sources: `run:<version>` (full-frame alpha of a version), `core` (mask 1+2), `preview:<stage>` (stage preview PNG).

    python tools/readme_compare/edge_ladder.py <work_dir> <object> <frame> <out_dir>
        --src "SAM 3.1 core=run:v001" --src "MatAnyone2=preview:matanyone" ... [--zoom x,y,w,h]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from engine.export import run_alpha_loader   # noqa: E402
from panels import edge_metrics, navy_comp, panel_row   # noqa: E402


def load_alpha(obj_dir: Path, src: str, f: int, w: int, h: int) -> np.ndarray:
    kind, _, arg = src.partition(":")
    if kind == "run":
        return np.clip(run_alpha_loader(obj_dir, arg, w, h)(f), 0, 1).astype(np.float32)
    if kind in ("preview", "file"):   # file:<folder with a_####.png> (results of a local_results test)
        d = obj_dir / "stage_preview" / arg if kind == "preview" else Path(arg)
        a = cv2.imread(str(d / f"a_{f:04d}.png"), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(f"no {arg} preview of frame {f}")
        a = a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) if a.shape[:2] != (h, w) else a
    raise ValueError(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("object")
    ap.add_argument("frame", type=int)
    ap.add_argument("out")
    ap.add_argument("--src", action="append", required=True, help='"Label=run:v001" | "Label=preview:matanyone"')
    ap.add_argument("--zoom", default="")
    ap.add_argument("--plate", default="", help="plate frame (default: the manifest beauty)")
    ap.add_argument("--decon", type=float, default=0.0, help="SD_Fuse Edge Decontaminate spread in px (0 = off)")
    ap.add_argument("--decon-mode", default="decon", choices=("decon", "unmix"))
    a = ap.parse_args()
    work, f = Path(a.work), a.frame
    man = json.loads(next(work.glob("shot_manifest*.json")).read_text("utf-8")) if list(work.glob("shot_manifest*.json")) \
        else json.loads(next(work.parent.glob("shot_manifest*.json")).read_text("utf-8"))
    plate = a.plate or man["plates"]["beauty"].replace("####", f"{f:04d}")
    img = cv2.imread(plate, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    obj_dir = work / "matte_desk" / "objects" / a.object if (work / "matte_desk").exists() else work / "objects" / a.object
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, alphas = [], []
    for s in a.src:
        lab, _, src = s.partition("=")
        labels.append(lab)
        alphas.append(load_alpha(obj_dir, src, f, w, h))
        cv2.imwrite(str(out / f"alpha_{len(alphas)}.png"), (alphas[-1] * 255 + 0.5).astype(np.uint8))

    if a.zoom:
        zoom = tuple(int(v) for v in a.zoom.split(","))
    else:   # square around the soft edge of the last source
        ys, xs = np.where((alphas[-1] > 0.02) & (alphas[-1] < 0.98))
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.45)
        cx, cy = int(np.median(xs)), int(np.median(ys))
        zoom = (max(cx - side // 2, 0), max(cy - side // 2, 0), side, side)

    metrics = {"plate": Path(plate).name, "frame": f, "object": a.object, "zoom": zoom,
               "sources": {lab: {"src": s.partition("=")[2], **edge_metrics(img, al)} for lab, s, al in zip(labels, a.src, alphas)}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    comps = [navy_comp(img, al, a.decon, a.decon_mode) for al in alphas]
    panel_row(comps, labels, out / "panel_full.jpg", width=3000)
    panel_row([img] + comps, ["Plate"] + labels, out / "panel_zoom.jpg", zoom=zoom, width=3000)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
