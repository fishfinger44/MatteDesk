# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 1 of PLAN_FINAL_TOUCH.md §11: engine/merge_matte.py vs the real MergeMatte group in the live Nuke session.

    python tools/readme_compare/mm_nuke_match.py prepare   # input EXRs (3 frames + calibration images) → c5_nuke_match/in
    python tools/readme_compare/mm_nuke_match.py build     # Nuke graph through the NukeMCP bridge (port 54321); user renders
    python tools/readme_compare/mm_nuke_match.py compare   # Nuke renders (c5_nuke_match/out) vs the port → metrics.json

Everything is raw (Read raw=True, Write raw=True, float32 EXR) so the Nuke colour management is out of the comparison;
the group's own OCIOColorSpace nodes (ACEScg ↔ ACEScct) need the aces_1.2 OCIO config in the Nuke root (build sets it).
"""

from __future__ import annotations

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

from engine import merge_matte as mm   # noqa: E402
from engine import merge_matte_nk as mmnk   # noqa: E402
from mm_measure import FRAMES, NAVY_RGB, load   # noqa: E402

OUT = ROOT / "_readme_compare/c5_nuke_match"
BLUR_SIZES = [2.5, 3.0, 5.0, 23.2, 100.0]
ERODE_SIZES = [6.0, -6.0, 20.0]
# knob sets rendered per frame: the SH010 preset and the node defaults; plus "alt" on the dog over a bright BG with
# every section on (bleed, flare, match) to check the dormant paths too
SETS = {"sh010": mmnk.PRESET_SH010, "default": mmnk.KNOB_DEFAULTS}
ALT_KNOBS = {**mmnk.PRESET_SH010, "bleed": 0.5, "flare": 0.5, "flare_threshold": 0.1, "flare_softness": 0.2, "match": 0.5,
             "match_exposure": -1.0, "match_saturation": 0.5, "wrap": 0.5}
ALT_BG_RGB = np.array([0.85, 0.80, 0.70], np.float32)


def write_exr(p: Path, rgba: np.ndarray) -> None:
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]].astype(np.float32))
    assert cv2.imwrite(str(p), bgra, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]), p


def read_exr(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 2:
        return im.astype(np.float32)
    if im.shape[2] == 3:
        im = np.dstack([im, np.ones(im.shape[:2], np.float32)])
    return np.ascontiguousarray(im[..., [2, 1, 0, 3]].astype(np.float32))


def inputs(fr: dict):
    plate = load(fr["plate"], True)
    alpha = load(fr["alpha"], False)
    if alpha.shape != plate.shape[:2]:
        alpha = cv2.resize(alpha, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
    lin = mm.srgb_to_acescg(plate)
    a = np.clip(alpha, 0, 1).astype(np.float32)
    return np.dstack([lin * a[..., None], a]), a


def bg_const(h: int, w: int, rgb_srgb) -> np.ndarray:
    bg = np.zeros((h, w, 4), np.float32)
    bg[..., :3] = mm.srgb_to_acescg(np.asarray(rgb_srgb, np.float32).reshape(1, 1, 3))
    return bg


def calib_images():
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - n / 2, yy - n / 2)
    im = np.zeros((n, n, 4), np.float32)
    im[n // 2, n // 2, 0] = 1.0                                  # R: impulse
    im[..., 1] = (r < 200).astype(np.float32)                   # G: hard disc
    im[..., 2] = np.clip((xx - 312) / 400, 0, 1)                # B: linear ramp 0..1 over 400 px
    im[..., 3] = np.clip(1 - (r - 180) / 40, 0, 1)              # A: disc with a 40 px linear edge
    ramp = np.zeros((n, n, 4), np.float32)
    ramp[..., :3] = np.clip(xx / (n - 1) * 1.2, 0, 1.2)[..., None]   # rgb ramp 0..1.2 for the Keyer
    ramp[..., 3] = 1
    return im, ramp


def prepare():
    (OUT / "in").mkdir(parents=True, exist_ok=True)
    (OUT / "out").mkdir(parents=True, exist_ok=True)
    for fr in FRAMES:
        fg, a = inputs(fr)
        write_exr(OUT / "in" / f"fg_{fr['name']}.exr", fg)
        write_exr(OUT / "in" / f"bg_{fr['name']}.exr", bg_const(*a.shape, NAVY_RGB))
        print("wrote", fr["name"], fg.shape)
    fg, a = inputs(FRAMES[2])
    write_exr(OUT / "in" / "bg_alt.exr", bg_const(*a.shape, ALT_BG_RGB))
    im, ramp = calib_images()
    write_exr(OUT / "in" / "calib.exr", im)
    write_exr(OUT / "in" / "calib_ramp.exr", ramp)
    print("inputs in", OUT / "in")


# Bridge quirk: code runs without closures (nested defs do not see top-level names) → helpers take them as default args.
NUKE_BUILD = r'''
import nuke
root = nuke.root()
root['colorManagement'].setValue('OCIO')
root['OCIO_config'].setValue('aces_1.2')
for n in nuke.selectedNodes():
    n.setSelected(False)
# previous comparison graphs (SD_Fuse / MergeMatte test branches, calibration) go away
for n in nuke.allNodes():
    nm = n.name()
    if nm.startswith(('Read_', 'Write_', 'SD_Fuse', 'MergeMatte', 'MM_', 'Blur_', 'FilterErode_', 'Keyer_')):
        nuke.delete(n)
IN = {in_dir!r}
OUTD = {out_dir!r}
NK = {nk_dir!r}
def rd(name, x, y, nuke=nuke, IN=IN):
    n = nuke.nodes.Read(file=IN + '/' + name, raw=True, name='Read_' + name.replace('.exr', ''))
    n.setXYpos(x, y)
    return n
def wr(src, name, x, y, nuke=nuke, OUTD=OUTD):
    w = nuke.nodes.Write(file=OUTD + '/' + name, file_type='exr', raw=True, name='Write_' + name.replace('.exr', ''))
    w['datatype'].setValue('32 bit float')
    w['compression'].setValue('none')
    w.setInput(0, src)
    w.setXYpos(x, y)
    return w
def mm_group(tag, x, y, nuke=nuke, NK=NK):
    for n in nuke.selectedNodes():
        n.setSelected(False)
    nuke.nodePaste(NK + '/MM_' + tag + '.nk')
    g = nuke.selectedNodes()[0]
    g.setSelected(False)
    g.setName('MM_' + tag)
    g.setXYpos(x, y)
    return g
x = 0
made = []
for name in {frames!r}:
    bg = rd('bg_' + name + '.exr', x, 0)
    fg = rd('fg_' + name + '.exr', x + 150, 0)
    for i, s in enumerate({sets!r}):
        g = mm_group(name + '_' + s, x + i * 180, 200)
        g.setInput(0, bg)
        g.setInput(1, fg)
        made.append(wr(g, 'nuke_' + name + '_' + s + '.exr', x + i * 180, 350).name())
    x += 500
bg = rd('bg_alt.exr', x, 0)
fg = rd('fg_' + {frames!r}[2] + '.exr', x + 150, 0)
g = mm_group('alt', x, 200)
g.setInput(0, bg)
g.setInput(1, fg)
made.append(wr(g, 'nuke_alt.exr', x, 350).name())
x += 500
cal = rd('calib.exr', x, 0)
y = 150
for s in {blurs!r}:
    b = nuke.nodes.Blur(channels='rgba', size=s, name='Blur_%g' % s)
    b.setInput(0, cal)
    b.setXYpos(x, y)
    made.append(wr(b, 'blur_%g.exr' % s, x + 150, y).name())
    y += 80
for s in {erodes!r}:
    e = nuke.nodes.FilterErode(channels='rgba', size=s, filter='gaussian', name='FilterErode_%g' % s)
    e.setInput(0, cal)
    e.setXYpos(x, y)
    made.append(wr(e, 'erode_%g.exr' % s, x + 150, y).name())
    y += 80
x += 400
ramp = rd('calib_ramp.exr', x, 0)
k = nuke.nodes.Keyer(operation='luminance key', name='Keyer_lowhigh')
k['range'].setValue([1.0, 0.6, 1.0, 1.0])
k.setInput(0, ramp)
k.setXYpos(x, 150)
made.append(wr(k, 'keyer_lowhigh.exr', x + 150, 150).name())
bad = []
for n in nuke.allNodes('Read'):
    if n.hasError():
        bad.append(n.name())
result = {{'writes': made, 'ocio': root['OCIO_config'].value(), 'read_errors': bad}}
'''


def build():
    sys.path.insert(0, "F:/claude_code/nuke-mcp")
    from nk import call   # type: ignore
    nk_dir = OUT / "nk"
    nk_dir.mkdir(exist_ok=True)
    for fr in FRAMES:
        for s, k in SETS.items():
            mmnk.write_group(nk_dir / f"MM_{fr['name']}_{s}.nk", k)
    mmnk.write_group(nk_dir / "MM_alt.nk", ALT_KNOBS)
    code = NUKE_BUILD.format(in_dir=str(OUT / "in").replace("\\", "/"), out_dir=str(OUT / "out").replace("\\", "/"),
                             nk_dir=str(nk_dir).replace("\\", "/"), frames=[f["name"] for f in FRAMES], sets=list(SETS),
                             blurs=BLUR_SIZES, erodes=ERODE_SIZES)
    print(json.dumps(call("execute_python", code=code), indent=1))


def fit_sigma(img: np.ndarray) -> float:
    """Sigma of a gaussian impulse response (R channel of the calib render), from its second moment."""
    r = np.maximum(img[..., 0], 0)
    n = r.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    m = r.sum()
    cx, cy = (r * xx).sum() / m, (r * yy).sum() / m
    var = (r * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / m / 2
    return float(np.sqrt(var))


def compare():
    rep = {"blur": {}, "erode": {}, "comp": {}}
    for s in BLUR_SIZES:
        p = OUT / "out" / f"blur_{s:g}.exr"
        if p.exists():
            im = read_exr(p)
            sig = fit_sigma(im)
            rep["blur"][str(s)] = {"sigma": round(sig, 4), "sigma_per_size": round(sig / s, 4),
                                   "peak": float(im[..., 0].max()), "sum": float(im[..., 0].sum())}
    calib, _ = calib_images()
    for s in ERODE_SIZES:
        p = OUT / "out" / f"erode_{s:g}.exr"
        if p.exists():
            im = read_exr(p)
            port = mm.filter_erode(calib[..., 3], s)
            d = np.abs(im[..., 3] - port)
            row = im[512, :, 1]
            xs = np.where(row > 0.5)[0]
            rep["erode"][str(s)] = {"disc_radius_05": float((xs.max() - xs.min()) / 2) if len(xs) else None,
                                    "port_vs_nuke_alpha_max": round(float(d.max()), 4), "mean": round(float(d.mean()), 5)}
    p = OUT / "out" / "keyer_lowhigh.exr"
    if p.exists():
        a = read_exr(p)[..., 3][512]
        rep["keyer_lowhigh"] = {"alpha_at_x": {str(x): round(float(a[x]), 4) for x in (0, 400, 512, 700, 852, 853, 900, 1023)}}
    jobs = [(f"{fr['name']}_{s}", fr, k, NAVY_RGB) for fr in FRAMES for s, k in SETS.items()] + [("alt", FRAMES[2], ALT_KNOBS, ALT_BG_RGB)]
    for name, fr, k, bgc in jobs:
        p = OUT / "out" / f"nuke_{name}.exr"
        if not p.exists():
            continue
        nk = read_exr(p)
        fg, a = inputs(fr)
        port = mm.merge_matte(fg, bg_const(*a.shape, bgc), k)
        d8 = np.abs(mm.acescg_to_srgb(nk[..., :3]) - mm.acescg_to_srgb(port[..., :3])) * 255
        band = (a > 0.02) & (a < 0.98)
        rep["comp"][name] = {"max_8bit": round(float(d8.max()), 3), "mean_8bit": round(float(d8.mean()), 4),
                             "p999_8bit": round(float(np.percentile(d8, 99.9)), 3),
                             "band_mean_8bit": round(float(d8[band].mean()), 4) if band.any() else None,
                             "lin_max": round(float(np.abs(nk[..., :3] - port[..., :3]).max()), 5)}
        vis = np.clip(d8.max(-1) / 4, 0, 1)
        cv2.imwrite(str(OUT / f"diff_{name}.png"), (vis * 255).astype(np.uint8))
    (OUT / "metrics.json").write_text(json.dumps(rep, indent=2), "utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "build": build, "compare": compare}[sys.argv[1]]()
