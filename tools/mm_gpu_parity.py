# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU MergeMatte (static/mm_gpu.js) parity with the Python port: writes the port's expected uint8 outputs of a synthetic
scene to a JSON; in the browser the same scene (same integer formulas) goes through mdMmGpu.render and is compared.
2026-10-30: neutral / hair preset / every section / choke + mix -> max 1 / 255, no value off by more.
    python tools/mm_gpu_parity.py static/_mm_parity.json   (temporary, delete after the check)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import merge_matte as mm, merge_matte_nk as nk   # noqa: E402

W, H = 96, 64
y, x = np.mgrid[0:H, 0:W]
plate = np.dstack([(x * 5 + y * 3) % 256, (x * 2 + y * 7) % 256, (x * 11 + 40) % 256]).astype(np.uint8)
alpha = np.clip(900 - ((x - 48) ** 2 + (y - 32) ** 2), 0, 255).astype(np.uint8)
bg = np.array([118, 118, 118]) / 255.0

SETS = {
    "neutral": {},
    "hair": dict(nk.PRESET_SH010),
    "mix": {"choke": -3, "soften": 2, "decon": 0.7, "decon_spread": 12, "bleed": 0.6, "bleed_size": 2, "wrap": 0.5,
            "wrap_size": 3, "flare": 0.4, "flare_threshold": 0.3, "flare_softness": 0.2, "match": 0.6,
            "match_exposure": -1, "match_saturation": 0.5, "darken": 0.8, "darken_width": 6, "edge_blur": 2, "log": 1},
    "choke": {"choke": 4, "mix": 0.8},
}
out = {}
for name, k in SETS.items():
    kk = mm.knobs(k)
    img, s = mm.preview_region(plate / 255.0, alpha / 255.0, bg, kk)
    out[name] = {"knobs": kk, "rgb": img.reshape(-1).tolist()}
json.dump(out, open(sys.argv[1], "w"))
print({n: len(v["rgb"]) for n, v in out.items()})
