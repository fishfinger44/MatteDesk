# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MergeMatte group without OCIO (user decisions 2026-11-03): the knob `log_curve` (none / ACEScct / Cineon / ARRI LogC3 /
ARRI LogC4 / Sony S-Log3 / video sRGB) and `Expression` nodes with the explicit formulas of engine/log_curves.py replace
the OCIOColorSpace nodes `fg_to_log` / `bg_to_log` / `from_log`, so the group works in Nuke's default OCIO config as in
ACES 1.x / 2.x. Old settings: `log: true` = acescct, `false` = none.

`real` (station, live Nuke through the bridge, no Write renders – pixels are sampled): the new group under the
nuke-default config vs the Python port (≤ 1/255 in display sRGB 8-bit) for ACEScct and for a curve switched on the knob
in Nuke (the enumeration's index drives the expressions), and vs the old OCIO group under aces_1.2 when its .nk is
still in _readme_compare/c5_nuke_match/nk (the ACES 1.x result stays)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from engine import log_curves as lc
from engine import merge_matte as mm
from engine import merge_matte_nk as mmnk

ROOT = Path(__file__).resolve().parents[1]
MATCH = ROOT / "_readme_compare/c5_nuke_match"


def _group(**knobs):
    return mmnk.group_nk({**mmnk.KNOB_DEFAULTS, **knobs})


def test_no_ocio_nodes_three_expression_nodes():
    t = _group(log_curve="acescct")
    assert "OCIOColorSpace" not in t
    names = [ln.split()[1] for ln in t.splitlines() if ln.startswith(" name ")]
    assert names.count("fg_to_log") == 1 and "bg_to_log" in names and "from_log" in names
    assert t.count("Expression {") == 3
    assert t.count(" disable {{parent.log_curve==0}}") == 5      # the 3 Expression nodes + Unpremult / Premult around the FG one
    assert "parent.log" not in t.replace("parent.log_curve", "")  # the old bool is gone


def test_enum_knob_menu_and_values():
    t = _group(log_curve="logc3")
    assert ' addUserKnob {4 log_curve l "Apply operations in Log"' in t
    assert 'M {none ACEScct Cineon "ARRI LogC3" "ARRI LogC4" "Sony S-Log3" "video (sRGB)"}' in t
    assert ' log_curve "ARRI LogC3"' in t
    assert " log_curve none" in _group(log_curve="none") and " log_curve ACEScct" in _group(log_curve="acescct")
    assert " log_curve Cineon" in _group(log_curve="cineon") and ' log_curve "video (sRGB)"' in _group(log_curve="video")


def test_old_bool_and_project_are_resolved_in_the_script():
    assert " log_curve ACEScct" in mmnk.group_nk({"log": 1}) and " log_curve none" in mmnk.group_nk({"log": 0})
    assert " log_curve ACEScct" in mmnk.group_nk({"log_curve": "project"})                      # no profile known
    assert ' log_curve "Sony S-Log3"' in mmnk.group_nk({"log_curve": "project"}, project_curve="slog3")
    assert " log_curve none" in mmnk.group_nk({"log_curve": "nonsense"})
    assert mmnk.normalize_knobs({"log": 1, "log_curve": "cineon"})["log_curve"] == "cineon"   # the new knob wins
    assert mmnk.PRESET_SH010["log_curve"] == lc.PROJECT and mmnk.KNOB_DEFAULTS["log_curve"] == lc.NONE


def test_expressions_carry_every_curve_by_knob_index():
    e = mmnk.log_expression("lin_to_log", "g")
    for cid, c in lc.LOG_CURVES.items():
        assert f"parent.log_curve=={lc.curve_index(cid)} ? (" in e
        assert re.sub(r"\bx\b", "g", c.nuke_lin_to_log) in e
    assert e.endswith(" : g") and "mar" not in e and "max(g" in e        # the variable substitution leaves `max` alone
    d = mmnk.log_expression("log_to_lin", "b")
    assert re.sub(r"\bx\b", "b", lc.LOG_CURVES["video"].nuke_log_to_lin) in d
    t = _group()
    assert f" expr0 \"{mmnk.log_expression('lin_to_log', 'r')}\"" in t
    assert f" expr2 \"{mmnk.log_expression('log_to_lin', 'b')}\"" in t
    assert t.count(" expr1 ") == 3


# ---------------------------------------------------------------- real
NUKE_SAMPLE = r'''
import nuke, json
root = nuke.root()
prev = {'cm': root['colorManagement'].value(), 'cfg': root['OCIO_config'].value()}
for n in nuke.allNodes():
    if n.name().startswith('MMT_'):
        nuke.delete(n)
def rd(path, name, nuke=nuke):
    return nuke.nodes.Read(file=path, raw=True, name=name)
def paste(path, name, nuke=nuke):
    for n in nuke.selectedNodes():
        n.setSelected(False)
    nuke.nodePaste(path)
    g = nuke.selectedNodes()[0]
    g.setSelected(False)
    g.setName(name)
    return g
def grid(node, pts, H, nuke=nuke):
    out = []
    for x, y in pts:
        out.append([node.sample(ch, x + 0.5, H - 1 - y + 0.5, 1, 1) for ch in ('red', 'green', 'blue')])
    return out
P = json.loads({params!r})
bg = rd(P['bg'], 'MMT_bg')
fg = rd(P['fg'], 'MMT_fg')
res = {{'read_errors': [n.name() for n in (bg, fg) if n.hasError()]}}
root['colorManagement'].setValue('OCIO')
root['OCIO_config'].setValue('nuke-default')
g = paste(P['new_nk'], 'MMT_new')
g.setInput(0, bg)
g.setInput(1, fg)
res['new_cfg'] = root['OCIO_config'].value()
res['new'] = grid(g, P['pts'], P['H'])
g['log_curve'].setValue(P['switch_menu'])
res['switched_knob'] = g['log_curve'].value()
res['switched'] = grid(g, P['pts'], P['H'])
nuke.delete(g)
if P.get('old_nk'):
    root['OCIO_config'].setValue('aces_1.2')
    g = paste(P['old_nk'], 'MMT_old')
    g.setInput(0, bg)
    g.setInput(1, fg)
    res['old_cfg'] = root['OCIO_config'].value()
    res['old'] = grid(g, P['pts'], P['H'])
    nuke.delete(g)
nuke.delete(bg)
nuke.delete(fg)
root['colorManagement'].setValue(prev['cm'])
root['OCIO_config'].setValue(prev['cfg'])
result = res
'''


def _read_exr(p: Path) -> np.ndarray:
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert im is not None, p
    if im.shape[2] == 3:
        im = np.dstack([im, np.ones(im.shape[:2], np.float32)])
    return np.ascontiguousarray(im[..., [2, 1, 0, 3]].astype(np.float32))


@pytest.mark.real
def test_group_in_nuke_default_config_matches_port_and_old_aces_group(tmp_path):
    from tests import nuke_bridge as nb
    fgp, bgp = MATCH / "in/fg_hair_f0026.exr", MATCH / "in/bg_hair_f0026.exr"
    if not (fgp.exists() and bgp.exists()):
        pytest.skip("no MergeMatte test inputs in _readme_compare/c5_nuke_match/in (mm_nuke_match.py prepare)")
    if not nb.available():
        pytest.skip("Nuke bridge (port 54321) not available – slot held by another client or no Nuke")
    fg, bg = _read_exr(fgp), _read_exr(bgp)
    H, W = fg.shape[:2]
    a = fg[..., 3]
    rng = np.random.default_rng(0)
    ys, xs = np.where((a > 0.02) & (a < 0.98))
    pick = rng.choice(len(xs), min(1200, len(xs)), replace=False)
    pts = [(int(x), int(y)) for x, y in zip(xs[pick], ys[pick])]
    pts += [(int(x), int(y)) for x, y in zip(rng.integers(0, W, 150), rng.integers(0, H, 150))]
    knobs = {**mmnk.PRESET_SH010, "log_curve": "acescct"}
    new_nk = tmp_path / "MMT_new.nk"
    mmnk.write_group(new_nk, knobs)
    old_nk = MATCH / "nk/MM_hair_f0026_sh010.nk"   # written by the OCIO version of the generator (before 2026-11-03)
    old_ok = old_nk.exists() and "OCIOColorSpace" in old_nk.read_text("utf-8")
    switch = "logc3"
    params = {"fg": str(fgp).replace("\\", "/"), "bg": str(bgp).replace("\\", "/"), "new_nk": str(new_nk).replace("\\", "/"),
              "old_nk": str(old_nk).replace("\\", "/") if old_ok else "", "pts": pts, "H": H,
              "switch_menu": lc.LOG_CURVES[switch].menu}
    res = nb.result(nb.execute(NUKE_SAMPLE.format(params=json.dumps(params)), timeout=1800))
    assert res.get("new") and not res.get("read_errors"), {k: v for k, v in res.items() if k not in ("new", "old", "switched")}
    assert res["new_cfg"] == "nuke-default" and res["switched_knob"] == params["switch_menu"]
    idx = (np.array([p[1] for p in pts]), np.array([p[0] for p in pts]))

    def srgb8(rgb):
        return mm.acescg_to_srgb(np.asarray(rgb, np.float32).reshape(-1, 1, 3))[:, 0] * 255

    def port(k):
        return srgb8(mm.merge_matte(fg, bg, k)[..., :3][idx])

    table = {}
    e = np.abs(srgb8(res["new"]) - port(knobs))
    table["acescct: nuke-default vs port"] = {"max_8bit": round(float(e.max()), 3), "mean_8bit": round(float(e.mean()), 4)}
    e2 = np.abs(srgb8(res["switched"]) - port({**knobs, "log_curve": switch}))
    table[f"{switch} (knob switched in Nuke) vs port"] = {"max_8bit": round(float(e2.max()), 3), "mean_8bit": round(float(e2.mean()), 4)}
    e3 = np.abs(srgb8(res["switched"]) - srgb8(res["new"]))
    table["switching the knob changes the comp"] = {"max_8bit": round(float(e3.max()), 3)}
    if old_ok and res.get("old"):
        e4 = np.abs(srgb8(res["old"]) - srgb8(res["new"]))
        table["old OCIO group (aces_1.2) vs new (nuke-default)"] = {"max_8bit": round(float(e4.max()), 3), "mean_8bit": round(float(e4.mean()), 4)}
        assert res["old_cfg"] == "aces_1.2" and e4.max() <= 1.0, table
    print("\nMergeMatte group vs port, sampled pixels:", json.dumps(table, indent=1))
    assert e.max() <= 1.0 and e2.max() <= 1.0 and e3.max() > 1.0, table
