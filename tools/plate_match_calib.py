# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibration of engine/plate_match.py thresholds on real plates (read-only; nothing of the shots is written).

    python tools/plate_match_calib.py <beauty pattern> <plate pattern> [<plate pattern> ...] [--other <pattern of another shot>]

Prints per pair: score / shift for the plate as it is, for the plate shifted by 12 px (synthetic), for the beauty
mirrored (same statistics, other layout = a negative) and for another shot scaled to the beauty's size.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import plate_match as pm   # noqa: E402


def pair(la, lb):
    ga, k = pm.grad_image(la)
    gb, _ = pm.grad_image(lb)
    (dx, dy), s = pm.compare(ga, gb)
    return {"score": round(s, 3), "dx": round(dx * k, 1), "dy": round(dy * k, 1)}


def main() -> int:
    args = sys.argv[1:]
    other = None
    if "--other" in args:
        i = args.index("--other")
        other = args[i + 1]
        args = args[:i] + args[i + 2:]
    beauty, plates = args[0], args[1:]
    bf = pm.frames_of(beauty)
    rep = {}
    for p in plates:
        pf = pm.frames_of(p)
        rows = []
        for f in pm.pick_frames(sorted(set(bf) & set(pf)), 3):
            la, lb = pm._lum(bf[f]), pm._lum(pf[f])
            sh = np.roll(np.roll(lb, 12, axis=1), 7, axis=0)
            rows.append({"frame": f, "as_is": pair(la, lb), "shift_12_7": pair(la, sh), "mirror": pair(la, la[:, ::-1].copy())})
        rep[Path(p).parent.name] = rows
    if other:
        of = sorted(pm.frames_of(other).items())
        la = pm._lum(bf[sorted(bf)[len(bf) // 2]])
        rows = []
        for _, path in of[:: max(1, len(of) // 3)][:3]:
            lo = pm._lum(path)
            lo = cv2.resize(lo, (la.shape[1], la.shape[0]), interpolation=cv2.INTER_AREA)
            rows.append(pair(la, lo))
        rep["other_shot"] = rows
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
