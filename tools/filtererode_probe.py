# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""FilterErode (Nuke) response probe – the data to port MergeMatte's `choke` 1:1 (NEXT_STEPS „choke 1:1 w podglądzie”).

    python tools/filtererode_probe.py prepare   # calibration EXRs -> local_results/filtererode/in
    python tools/filtererode_probe.py build     # live Nuke through the bridge (nuke-mcp, port 54321): Read -> FilterErode
                                                # (gaussian + box, many sizes) -> Write, rendered in that session
    python tools/filtererode_probe.py compare   # port (engine/merge_matte.filter_erode) vs Nuke per profile / size

Profiles are 1-D along x (rows of a band = the same profile), so the response of each size is a curve: a hard step,
ramps 2–40 px wide, a blurred edge, thin lines 1–5 px, steps to and from 0.5; plus a 2-D sheet (discs, a square
corner, a 1 px diagonal). Nothing of the user's shots is read. Nuke: Write raw 32-bit float, no colour management.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "local_results" / "filtererode"
BRIDGE = Path(os.environ.get("NUKE_MCP_DIR", "C:/claude_code/nuke-mcp"))
SIZES = [-20, -10, -6, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 6, 10, 20]
FILTERS = ["gaussian", "box"]
W, ROWS = 512, 16
X0 = 256


def profiles():
    """name -> 1-D alpha along x (length W)."""
    x = np.arange(W, dtype=np.float32) - X0
    p = {"step": (x < 0).astype(np.float32)}
    for w in (2, 5, 10, 20, 40):
        p[f"ramp{w}"] = np.clip(0.5 - x / w, 0, 1)
    g = np.exp(-0.5 * (np.arange(-15, 16) / 3.0) ** 2)
    p["blur3"] = np.convolve(p["step"], g / g.sum(), mode="same").astype(np.float32)
    for w in (1, 2, 3, 5):
        p[f"line{w}"] = ((x >= 0) & (x < w)).astype(np.float32)
    p["step_1_05"] = np.where(x < 0, 1.0, 0.5).astype(np.float32)
    p["step_05_0"] = np.where(x < 0, 0.5, 0.0).astype(np.float32)
    return p


def sheet_1d():
    pr = profiles()
    names = list(pr)
    im = np.zeros((ROWS * len(names), W), np.float32)
    for i, n in enumerate(names):
        im[i * ROWS:(i + 1) * ROWS] = pr[n]
    return names, im


def sheet_2d():
    n = 512
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    a = np.zeros((n, n), np.float32)
    a[np.hypot(xx - 128, yy - 128) < 60] = 1                 # disc r 60
    a[np.hypot(xx - 384, yy - 128) < 6] = 1                  # disc r 6
    a[300:450, 50:200] = 1                                   # square (corners)
    for i in range(300, 460):                                # 1 px diagonal
        a[i, i - 20] = 1
    return a


def write_exr(path: Path, a: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack([a, a, a, a]).astype(np.float32)
    cv2.imwrite(str(path), rgba[..., [2, 1, 0, 3]])


def read_alpha(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return im[..., 3] if im.ndim == 3 and im.shape[2] == 4 else im[..., 0]


def prepare():
    names, im = sheet_1d()
    write_exr(OUT / "in" / "profiles.exr", im)
    write_exr(OUT / "in" / "sheet2d.exr", sheet_2d())
    (OUT / "in" / "profiles.json").write_text(json.dumps({"names": names, "rows": ROWS, "x0": X0}), "utf-8")
    print("inputs in", OUT / "in")


BUILD = r'''
import nuke
for n in nuke.allNodes():
    if n.name().startswith(('FEP_',)):
        nuke.delete(n)
IN = {in_dir!r}
OUTD = {out_dir!r}
made = []
x = 0
for src in ('profiles', 'sheet2d'):
    r = nuke.nodes.Read(file=IN + '/' + src + '.exr', raw=True, name='FEP_Read_' + src)
    r.setXYpos(x, 0)
    y = 120
    for flt in {filters!r}:
        for s in {sizes!r}:
            tag = '%s_%s_%g' % (src, flt, s)
            e = nuke.nodes.FilterErode(channels='rgba', size=s, filter=flt, name='FEP_E_' + tag.replace('-', 'm').replace('.', 'p'))
            e.setInput(0, r)
            e.setXYpos(x, y)
            w = nuke.nodes.Write(file=OUTD + '/' + tag + '.exr', file_type='exr', raw=True, name='FEP_W_' + tag.replace('-', 'm').replace('.', 'p'))
            w['datatype'].setValue('32 bit float')
            w['compression'].setValue('none')
            w.setInput(0, e)
            w.setXYpos(x + 150, y)
            nuke.execute(w, 1, 1)
            made.append(tag)
            y += 60
    x += 400
result = {{'made': len(made), 'version': nuke.NUKE_VERSION_STRING}}
'''


def build():
    sys.path.insert(0, str(BRIDGE))
    from nk import call   # type: ignore
    (OUT / "out").mkdir(parents=True, exist_ok=True)
    code = BUILD.format(in_dir=str(OUT / "in").replace("\\", "/"), out_dir=str(OUT / "out").replace("\\", "/"),
                        filters=FILTERS, sizes=SIZES)
    print(json.dumps(call("execute_python", code=code), indent=1))


def script(path: str = None):
    """The same graph as a script for a batch Nuke: `Nuke11.1.exe --safe -t <path>` (the bridge is not needed;
    --safe skips ~/.nuke startup scripts). Writes local_results/filtererode/build_nuke.py."""
    (OUT / "out").mkdir(parents=True, exist_ok=True)
    code = BUILD.format(in_dir=str(OUT / "in").replace("\\", "/"), out_dir=str(OUT / "out").replace("\\", "/"),
                        filters=FILTERS, sizes=SIZES) + "\nprint(result)\n"
    p = Path(path) if path else OUT / "build_nuke.py"
    p.write_text(code, "utf-8")
    print(p)


def compare():
    from engine import merge_matte as mm
    meta = json.loads((OUT / "in" / "profiles.json").read_text("utf-8"))
    names, im = sheet_1d()
    rep = {}
    for s in SIZES:
        p = OUT / "out" / f"profiles_gaussian_{s:g}.exr"
        if not p.exists():
            continue
        nk = read_alpha(p)
        port = mm.filter_erode(im, s)
        row = {}
        for i, n in enumerate(names):
            r = i * meta["rows"] + meta["rows"] // 2
            row[n] = round(float(np.abs(nk[r] - port[r]).max()), 4)
        rep[str(s)] = row
    (OUT / "compare.json").write_text(json.dumps(rep, indent=1), "utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "build": build, "script": script, "compare": compare}[sys.argv[1]]()
