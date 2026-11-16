# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/log_curves.py – the lin ↔ log curves of MergeMatte (user decisions 2026-11-03: a curve only, explicit formulas,
no OCIO names). Synthetic: round trip on [0, 16] within 1e-5, monotonic, published mid-grey values, the table is the one
source of ids / menu order. `real` (station, live Nuke through the bridge): for every curve a 1D ramp through an
`Expression` node built from the table's Nuke text vs the port – max |Δ| ≤ 1/255 both ways and in a Nuke round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine import log_curves as lc

N = 1024


def _ramp(lo, hi):
    return np.linspace(lo, hi, N, dtype=np.float32)


@pytest.mark.parametrize("cid", list(lc.LOG_CURVES))
def test_round_trip_0_16(cid):
    c = lc.LOG_CURVES[cid]
    x = np.linspace(0, 16, 100_001, dtype=np.float32)
    y = c.lin_to_log(x)
    assert y.dtype == np.float32 and np.all(np.isfinite(y))
    assert np.abs(c.log_to_lin(y) - x).max() <= 1e-5 * 1.0 + 1e-5 * 16   # 1e-5 at the top of the range (float32)
    assert np.abs(c.log_to_lin(y) - x).max() <= 2e-5
    assert np.all(np.diff(y) > 0), cid


@pytest.mark.parametrize("cid, grey, white", [
    ("acescct", 0.4135884, 0.5547945), ("cineon", 0.4573196, 0.6695992), ("logc3", 0.3910068, 0.5706316),
    ("logc4", 0.2783958, 0.4275194), ("slog3", 0.4105572, 0.5960273), ("video", 0.4613561, 1.0)])
def test_published_mid_grey_and_white(cid, grey, white):
    c = lc.LOG_CURVES[cid]
    assert abs(float(c.lin_to_log(np.float32(0.18))) - grey) < 2e-6
    assert abs(float(c.lin_to_log(np.float32(1.0))) - white) < 2e-6


def test_table_is_the_one_source_of_ids_and_menu():
    assert lc.CURVE_IDS[0] == lc.NONE and lc.CURVE_IDS[1:] == list(lc.LOG_CURVES)
    assert lc.CURVE_IDS == ["none", "acescct", "cineon", "logc3", "logc4", "slog3", "video"]
    assert lc.MENU[0] == "none" and len(lc.MENU) == len(lc.CURVE_IDS)
    assert [lc.curve_index(i) for i in lc.CURVE_IDS] == list(range(7)) and lc.curve_index("nonsense") == 0
    assert lc.get("none") is None and lc.get(None) is None and lc.get("acescct").id == "acescct"
    with pytest.raises(ValueError):
        lc.get(lc.PROJECT)
    assert lc.legacy_curve(True) == "acescct" and lc.legacy_curve(1) == "acescct" and lc.legacy_curve(0) == "none"


def test_nuke_text_uses_only_the_expression_library():
    """`?:`, log, log10, pow, max, min – no log2 / exp2 / numpy names (the Nuke Expression node has none)."""
    import re
    for c in lc.LOG_CURVES.values():
        for t in (c.nuke_lin_to_log, c.nuke_log_to_lin):
            names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", t))
            assert names <= {"x", "log", "log10", "pow", "max", "min", "e"}, (c.id, names - {"x"})
            assert " ? " in t or "?" not in t


# ---------------------------------------------------------------- real: Nuke Expression vs the port
NUKE_RAMP = r'''
import nuke, json, re
for n in nuke.allNodes():
    if n.name().startswith('LCT_'):
        nuke.delete(n)
fmt = None
for f in nuke.formats():
    if f.width() == {n} and f.height() == 1:
        fmt = f
if fmt is None:
    fmt = nuke.addFormat('{n} 1 0 0 {n} 1 1 LCT_ramp')
def ramp(expr, name, nuke=nuke, fmt=fmt):
    r = nuke.nodes.Expression(expr0=expr, expr1=expr, expr2=expr, name=name, inputs=[])
    r['format'].setValue(fmt)   # an Expression without input takes its format from the knob
    return r
def thru(src, expr, name, nuke=nuke, re=re):
    sub = lambda v: re.sub(r'\bx\b', v, expr)
    e = nuke.nodes.Expression(expr0=sub('r'), expr1=sub('g'), expr2=sub('b'), name=name)
    e.setInput(0, src)
    return e
def row(node, n={n}):
    return [node.sample('red', i + 0.5, 0.5) for i in range(n)]
out = {{}}
curves = json.loads({curves!r})
lin = ramp('x/({n}-1)*16', 'LCT_lin')
logr = ramp('x/({n}-1)', 'LCT_log')
for cid, c in curves.items():
    enc = thru(lin, c['enc'], 'LCT_enc_' + cid)
    dec = thru(logr, c['dec'], 'LCT_dec_' + cid)
    rt = thru(enc, c['dec'], 'LCT_rt_' + cid)
    out[cid] = {{'enc': row(enc), 'dec': row(dec), 'rt': row(rt)}}
for n in nuke.allNodes():
    if n.name().startswith('LCT_'):
        nuke.delete(n)
result = {{'rows': out, 'x_lin': row(lin) if False else None}}
'''


@pytest.mark.real
def test_nuke_expression_matches_port_per_curve():
    from tests import nuke_bridge as nb
    if not nb.available():
        pytest.skip("Nuke bridge (port 54321) not available – slot held by another client or no Nuke")
    curves = {cid: {"enc": c.nuke_lin_to_log, "dec": c.nuke_log_to_lin} for cid, c in lc.LOG_CURVES.items()}
    rep = nb.result(nb.execute(NUKE_RAMP.format(n=N, curves=json.dumps(curves))))
    rows = rep.get("rows") or {}
    assert set(rows) == set(lc.LOG_CURVES), rep
    x_lin, y_log = _ramp(0, 16), _ramp(0, 1)
    table = {}
    for cid, c in lc.LOG_CURVES.items():
        enc = np.asarray(rows[cid]["enc"], np.float32)
        dec = np.asarray(rows[cid]["dec"], np.float32)
        rt = np.asarray(rows[cid]["rt"], np.float32)
        e_enc = float(np.abs(enc - c.lin_to_log(x_lin)).max())
        e_dec = float(np.abs(dec - c.log_to_lin(y_log)).max())
        e_rt = float(np.abs(rt - x_lin).max())
        table[cid] = {"lin_to_log": e_enc, "log_to_lin": e_dec, "round_trip": e_rt}
        assert e_enc <= 1 / 255 and e_dec <= 1 / 255 and e_rt <= 1 / 255, (cid, table[cid])
    print("\nmax |Nuke Expression − port| per curve:", json.dumps(table, indent=1))
