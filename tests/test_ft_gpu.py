# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract of the Final Touch GPU preview (NEXT_STEPS 7a/b, 2026-10-28): static/ft_gpu.js (its JS reference cpu() =
the maths of the GLSL, checked against the GLSL in the browser by mdFtGpu.selfTest()) against engine/final_touch.py –
the server's preview of the same strokes (full-resolution layer + compose, INTER_AREA to the proxy, u8).

The server rasterises a stroke with cv2 and fades it with a cv2 distance transform (DIST_L2, mask 5) of that raster;
the shader uses the analytic distance to the stroke's shape. Hard strokes and the inside of soft ones agree to 1/255;
on the soft ramp the transform's pixel grid shows as steps of a few 1/255 (measured 2026-10-28, see LIMITS) – the
preview is replaced by the server's image as soon as it arrives, so the limits below only guard the maths (a wrong
mode / radius / feather / footprint gives mean errors of 30+ / 255)."""

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "tests" / "js" / "ft_gpu_cpu.cjs"

W, H = 4096, 2731
S = min(1.0, 1920.0 / max(W, H))
PW, PH = max(1, round(W * S)), max(1, round(H * S))

CASES = {
    "hard_line": [{"mode": "add", "radius_px": 30, "hardness": 1.0, "p": [[1000, 800], [1600, 900]]}],
    "soft_line": [{"mode": "add", "radius_px": 30, "hardness": 0.6, "p": [[1000, 800], [1600, 900]]}],
    "soft_dot_remove": [{"mode": "remove", "radius_px": 60, "hardness": 0.0, "p": [[2000, 1400]]}],
    "small": [{"mode": "add", "radius_px": 4, "hardness": 0.6, "p": [[1500, 1500], [1520, 1600]]}],
    "curve": [{"mode": "add", "radius_px": 25, "hardness": 0.6,
               "p": [[1500 + 300 * np.cos(t), 1400 + 300 * np.sin(t)] for t in np.linspace(0, 3, 40)]}],
    "pen": [{"mode": "add", "radius_px": 20, "hardness": 0.6, "p": [[1000, 2000, 5], [1200, 2050, 25], [1400, 2000, 40], [1600, 2100, 10]]}],
    "mix": [{"mode": "add", "radius_px": 40, "hardness": 0.5, "p": [[1700, 1100], [2300, 1700]]},
            {"mode": "remove", "radius_px": 30, "hardness": 0.8, "p": [[1700, 1700], [2300, 1100]]},
            {"mode": "erase", "radius_px": 20, "hardness": 0.3, "p": [[2000, 1000], [2000, 1800]]}],
    "border": [{"mode": "add", "radius_px": 50, "hardness": 1.0, "p": [[-20, 30], [300, -10]]}],
}
# measured 2026-10-28 (GROW / OFF of ft_gpu.js), with margin: mean |Δ| over the touched pixels, 99th percentile, max (u8)
LIMITS = {"hard_line": (0.5, 1, 120), "border": (0.5, 1, 120), "soft_line": (1.5, 8, 16), "soft_dot_remove": (2.5, 8, 10),
          "small": (10.0, 64, 90), "curve": (2.0, 16, 26), "pen": (1.5, 8, 20), "mix": (4.5, 30, 42)}


def _node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – kontrakt ft_gpu.js pominięty")
    return node


def _base():
    b = np.zeros((H, W), np.float32)
    cv2.circle(b, (2000, 1400), 500, 1.0, -1)
    return cv2.GaussianBlur(b, (0, 0), 6)


def _server(base, strokes, finish=None):
    """The server's preview (app._ft_frame_proxy == this, tests/test_final_touch.py) in u8."""
    from engine import final_touch as ft
    a = ft.compose(base, ft.layer(strokes, W, H)) if strokes else base
    if finish:
        a = ft.finish_alpha(a, finish)
    return (np.clip(cv2.resize(np.clip(a, 0, 1).astype(np.float32), (PW, PH), interpolation=cv2.INTER_AREA), 0, 1) * 255 + 0.5).astype(np.uint8)


def _gpu_ref(jobs):
    r = subprocess.run([_node(), str(SCRIPT)], input=json.dumps(jobs), capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return [np.frombuffer(base64.b64decode(x["out"]), np.uint8).reshape(PH, PW) for x in json.loads(r.stdout)]


def _job(base_p, strokes, live=None, finish=None):
    return {"W": W, "H": H, "PW": PW, "PH": PH, "base": base64.b64encode(base_p.tobytes()).decode(),
            "strokes": strokes, "live": live, "finish": finish or {}}


def _stats(sv, gp, b8):
    d = np.abs(sv.astype(int) - gp.astype(int))
    t = (sv != b8) | (gp != b8)
    return float(d[t].mean()), float(np.percentile(d[t], 99)), int(d.max()), int(t.sum())


def test_ft_gpu_reference_matches_final_touch():
    from engine import final_touch as ft
    base = _base()
    b8 = _server(base, [])
    cases = {k: [ft.clean_stroke(s) for s in v] for k, v in CASES.items()}
    outs = _gpu_ref([_job(b8, st) for st in cases.values()])
    rep = {}
    for (name, st), gp in zip(cases.items(), outs):
        rep[name] = _stats(_server(base, st), gp, b8)
    print(json.dumps(rep))
    for name, (mean, p99, mx, n) in rep.items():
        lm, lp, lx = LIMITS[name]
        assert n > 0 and mean <= lm and p99 <= lp and mx <= lx, (name, rep[name])


def test_ft_gpu_live_stroke_levels_and_untouched_pixels():
    """The stroke being drawn (`live`) = the same as a committed one; black / white / gamma after the strokes like
    final_touch.finish_alpha; pixels away from every stroke = the base proxy exactly; a wrong mode is caught."""
    from engine import final_touch as ft
    base = _base()
    b8 = _server(base, [])
    st = [ft.clean_stroke(s) for s in CASES["mix"]]
    fin = {"black": 0.1, "white": 0.9, "gamma": 1.4}
    a, b, c, d = _gpu_ref([_job(b8, st), _job(b8, st[:2], live=st[2]), _job(b8, st, finish=fin),
                           _job(b8, [{**st[0], "mode": "remove"}] + st[1:])])
    assert np.array_equal(a, b)
    far = np.ones((PH, PW), bool)
    far[int(900 * S):int(1900 * S), int(1600 * S):int(2400 * S)] = False
    assert np.array_equal(a[far], b8[far])
    sv = _server(base, st)
    # levels on the proxy vs on the full frame: the base's soft edge differs a little, the rest is the same maths
    svf = _server(base, st, {**ft.FINISH_DEFAULT, **fin})
    mean, p99, _mx, _n = _stats(svf, c, _server(base, [], {**ft.FINISH_DEFAULT, **fin}))
    assert mean <= 6.0 and p99 <= 48, (mean, p99)
    bad = _stats(sv, d, b8)
    assert bad[0] > 20, bad   # what the browser's comparison (finaltouch.js) flags


def test_ft_gpu_js_is_wired():
    """index.html loads ft_gpu.js before finaltouch.js; finaltouch.js draws through it and compares with the server."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/ft_gpu.js") < html.index("/static/finaltouch.js")
    js = (ROOT / "static" / "finaltouch.js").read_text(encoding="utf-8")
    for s in ("mdFtGpu.render", "liveMatte", "compareWithServer", "base=1"):
        assert s in js, s
