# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU MergeMatte preview (static/mm_gpu.js) vs the Python port, per log curve (2026-11-03) – the same contract as the
existing operations (2026-10-30: max 1 / 255). The shader needs a browser with WebGL2, so the comparison runs in the
app's browser: `python tools/mm_gpu_parity.py static/_mm_parity.json`, open MatteDesk, run BROWSER_JS in its console
(or through the desktop app's browser pane) – it prints {set: max |Δ| in 1/255} and `ok`. Here: the parity table builds
for every curve and the JS snippet stays in step with the tool's scene (same integer formulas)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from engine import log_curves as lc

ROOT = Path(__file__).resolve().parents[1]

BROWSER_JS = r"""
(async () => {
  const sets = await (await fetch('/static/_mm_parity.json?' + Date.now())).json();
  const W = 96, H = 64, out = {};
  const mk = (fill) => { const c = document.createElement('canvas'); c.width = W; c.height = H;
    const ctx = c.getContext('2d'), im = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 4; const v = fill(x, y); im.data.set([v[0], v[1], v[2], 255], i); }
    ctx.putImageData(im, 0, 0); return c; };
  const plate = mk((x, y) => [(x * 5 + y * 3) % 256, (x * 2 + y * 7) % 256, (x * 11 + 40) % 256]);
  const alpha = mk((x, y) => { const a = Math.min(255, Math.max(0, 900 - ((x - 48) ** 2 + (y - 32) ** 2))); return [a, a, a]; });
  for (const [name, s] of Object.entries(sets)) {
    const c = window.mdMmGpu.render({ plate, alpha, bg: { mode: 'color', color: [118 / 255, 118 / 255, 118 / 255] }, knobs: s.knobs, W, H });
    const d = c.getContext('2d').getImageData(0, 0, W, H).data;
    let mx = 0, n = 0;
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) for (let k = 0; k < 3; k++) { const e = Math.abs(d[i + k] - s.rgb[j + k]); if (e > mx) mx = e; if (e > 1) n++; }
    out[name] = { max: mx, over1: n };
  }
  console.log(JSON.stringify(out));
  return out;
})()
"""


def _tool():
    spec = importlib.util.spec_from_file_location("mm_gpu_parity", ROOT / "tools/mm_gpu_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parity_table_has_a_set_per_curve_with_its_index():
    t = _tool()
    out = t.build()
    for cid in lc.LOG_CURVES:
        s = out[f"curve_{cid}"]
        assert s["knobs"]["log_curve"] == cid and s["knobs"]["curve"] == lc.curve_index(cid)
        assert len(s["rgb"]) == t.W * t.H * 3
    assert out["neutral"]["knobs"]["curve"] == 0 and out["hair"]["knobs"]["log_curve"] == "acescct"   # project -> ACEScct
    # the log sets differ from each other in the edge (a wrong curve index would make two of them identical)
    imgs = {cid: np.asarray(out[f"curve_{cid}"]["rgb"]) for cid in lc.LOG_CURVES}
    ids = list(imgs)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert np.abs(imgs[a] - imgs[b]).max() >= 2, (a, b)


def test_browser_snippet_matches_the_tool_scene():
    src = (ROOT / "tools/mm_gpu_parity.py").read_text(encoding="utf-8")
    assert "W, H = 96, 64" in src and "(x * 5 + y * 3) % 256" in src and "900 - ((x - 48) ** 2 + (y - 32) ** 2)" in src
    assert "const W = 96, H = 64" in BROWSER_JS and "(x * 5 + y * 3) % 256" in BROWSER_JS and "900 - ((x - 48) ** 2 + (y - 32) ** 2)" in BROWSER_JS
    assert "118 / 255" in BROWSER_JS and "np.array([118, 118, 118]) / 255.0" in src
