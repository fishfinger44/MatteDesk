# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""lin ↔ log curves of MergeMatte (user decisions 2026-11-03): the „Apply operations in Log” mode picks a CURVE only
(the edge operations are per channel, so the gamut plays no role – no full OCIO profile, no gamut conversion), and the
Nuke group must not depend on OCIO colourspace names: every curve is an explicit formula, written into the group as an
`Expression` node (engine/merge_matte_nk.py) and computed here for the Python port (engine/merge_matte.py) and mirrored
in GLSL (static/mm_gpu.js). One table, `LOG_CURVES`: id → label, lin→log, log→lin, and the same formula as Nuke expression
text (variable `x`). tests/test_log_curves.py: round trip on [0, 16] (1e-5) and, `real`, the Nuke Expression vs the
port on a ramp (≤ 1/255).

Formulas: ACEScct (ACES 1.x / 2.x, same constants as OCIO `ACES - ACEScct`); Cineon as Nuke's Log2Lin (black 95,
white 685, gamma 0.6); ARRI LogC3 (EI 800, ALEXA SUP 3.x); ARRI LogC4 (ALEXA 35); Sony S-Log3 (reflectance input);
„video” = the sRGB OETF (8-bit sources: PNG / MP4 plates of MatteDesk are display sRGB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

NONE = "none"          # linear: no conversion (the knob's first entry, index 0)
PROJECT = "project"    # palette / object setting only: the curve of the project's plate colour profile (resolved before use)


@dataclass(frozen=True)
class LogCurve:
    id: str
    label: str                    # UI label (palette, stage 0 profile list)
    menu: str                     # entry of the Nuke enumeration knob
    lin_to_log: Callable[[np.ndarray], np.ndarray]
    log_to_lin: Callable[[np.ndarray], np.ndarray]
    nuke_lin_to_log: str          # Nuke Expression text, variable `x` (one channel)
    nuke_log_to_lin: str
    note: str = ""


def _f32(fn):
    def run(x):
        return np.asarray(fn(np.asarray(x, np.float64)), np.float32)
    return run


# ---------------------------------------------------------------- ACEScct
_CCT_A, _CCT_B, _CCT_X0, _CCT_Y0 = 10.5402377416545, 0.0729055341958355, 0.0078125, 0.155251141552511


def _acescct_enc(x):
    hi = (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52
    return np.where(x <= _CCT_X0, _CCT_A * x + _CCT_B, hi)


def _acescct_dec(y):
    hi = np.exp2(np.minimum(y, 10.0) * 17.52 - 9.72)
    return np.where(y <= _CCT_Y0, (y - _CCT_B) / _CCT_A, hi)


# ---------------------------------------------------------------- Cineon (Nuke Log2Lin defaults)
_CIN_BLACK, _CIN_WHITE, _CIN_GAMMA = 95.0, 685.0, 0.6
_CIN_OFFSET = 10.0 ** ((_CIN_BLACK - _CIN_WHITE) * 0.002 / _CIN_GAMMA)
_CIN_GAIN = 1.0 / (1.0 - _CIN_OFFSET)
_CIN_K = 0.002 / _CIN_GAMMA   # density per code value


def _cineon_enc(x):
    return (np.log10(np.maximum(x / _CIN_GAIN + _CIN_OFFSET, 1e-10)) / _CIN_K + _CIN_WHITE) / 1023.0


def _cineon_dec(y):
    return _CIN_GAIN * (10.0 ** ((1023.0 * y - _CIN_WHITE) * _CIN_K) - _CIN_OFFSET)


# ---------------------------------------------------------------- ARRI LogC3 (EI 800)
_C3 = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)
_C3_YCUT = _C3["e"] * _C3["cut"] + _C3["f"]


def _logc3_enc(x):
    c = _C3
    return np.where(x > c["cut"], c["c"] * np.log10(np.maximum(c["a"] * x + c["b"], 1e-10)) + c["d"], c["e"] * x + c["f"])


def _logc3_dec(y):
    c = _C3
    return np.where(y > _C3_YCUT, (10.0 ** ((y - c["d"]) / c["c"]) - c["b"]) / c["a"], (y - c["f"]) / c["e"])


# ---------------------------------------------------------------- ARRI LogC4
_C4_A = (2.0 ** 18 - 16.0) / 117.45
_C4_B = (1023.0 - 95.0) / 1023.0
_C4_C = 95.0 / 1023.0
_C4_S = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * _C4_C / _C4_B)) / (_C4_A * _C4_B)
_C4_T = (2.0 ** (14.0 * (-_C4_C / _C4_B) + 6.0) - 64.0) / _C4_A


def _logc4_enc(x):
    return np.where(x < _C4_T, (x - _C4_T) / _C4_S,
                    (np.log2(np.maximum(_C4_A * x + 64.0, 1e-10)) - 6.0) / 14.0 * _C4_B + _C4_C)


def _logc4_dec(y):
    return np.where(y < 0.0, y * _C4_S + _C4_T, (2.0 ** (14.0 * (y - _C4_C) / _C4_B + 6.0) - 64.0) / _C4_A)


# ---------------------------------------------------------------- Sony S-Log3
_SL3_CUT, _SL3_YCUT = 0.01125000, 171.2102946929


def _slog3_enc(x):
    return np.where(x >= _SL3_CUT, (420.0 + np.log10(np.maximum((x + 0.01) / 0.19, 1e-10)) * 261.5) / 1023.0,
                    (x * (_SL3_YCUT - 95.0) / _SL3_CUT + 95.0) / 1023.0)


def _slog3_dec(y):
    return np.where(y >= _SL3_YCUT / 1023.0, 10.0 ** ((y * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01,
                    (y * 1023.0 - 95.0) * _SL3_CUT / (_SL3_YCUT - 95.0))


# ---------------------------------------------------------------- video (sRGB OETF)
def _srgb_enc(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055)


def _srgb_dec(y):
    return np.where(y <= 0.04045, y / 12.92, ((np.maximum(y, 0) + 0.055) / 1.055) ** 2.4)


def _r(v: float) -> str:
    return repr(float(v))


# Nuke expression text: `?:` of the Expression node, log / log10 / pow / max of its library (no log2 / exp2 there)
_LN2 = math.log(2.0)
LOG_CURVES: Dict[str, LogCurve] = {c.id: c for c in [
    LogCurve("acescct", "ACEScct", "ACEScct", _f32(_acescct_enc), _f32(_acescct_dec),
             f"x <= {_CCT_X0} ? {_CCT_A}*x + {_CCT_B} : (log(max(x, 1e-10))/{_r(_LN2)} + 9.72)/17.52",
             f"x <= {_CCT_Y0} ? (x - {_CCT_B})/{_CCT_A} : pow(2, min(x, 10)*17.52 - 9.72)",
             "ACES 1.x / 2.x (the same constants as OCIO „ACES - ACEScct”)"),
    LogCurve("cineon", "Cineon", "Cineon", _f32(_cineon_enc), _f32(_cineon_dec),
             f"(log10(max(x/{_r(_CIN_GAIN)} + {_r(_CIN_OFFSET)}, 1e-10))/{_r(_CIN_K)} + {_CIN_WHITE})/1023",
             f"{_r(_CIN_GAIN)}*(pow(10, (1023*x - {_CIN_WHITE})*{_r(_CIN_K)}) - {_r(_CIN_OFFSET)})",
             "as Nuke's Log2Lin: black 95, white 685, gamma 0.6"),
    LogCurve("logc3", "ARRI LogC3 (EI 800)", "ARRI LogC3", _f32(_logc3_enc), _f32(_logc3_dec),
             f"x > {_C3['cut']} ? {_C3['c']}*log10(max({_C3['a']}*x + {_C3['b']}, 1e-10)) + {_C3['d']} : {_C3['e']}*x + {_C3['f']}",
             f"x > {_r(_C3_YCUT)} ? (pow(10, (x - {_C3['d']})/{_C3['c']}) - {_C3['b']})/{_C3['a']} : (x - {_C3['f']})/{_C3['e']}",
             "ALEXA (SUP 3.x) at EI 800"),
    LogCurve("logc4", "ARRI LogC4", "ARRI LogC4", _f32(_logc4_enc), _f32(_logc4_dec),
             f"x < {_r(_C4_T)} ? (x - {_r(_C4_T)})/{_r(_C4_S)} : (log(max({_r(_C4_A)}*x + 64, 1e-10))/{_r(_LN2)} - 6)/14*{_r(_C4_B)} + {_r(_C4_C)}",
             f"x < 0 ? x*{_r(_C4_S)} + {_r(_C4_T)} : (pow(2, 14*(x - {_r(_C4_C)})/{_r(_C4_B)} + 6) - 64)/{_r(_C4_A)}",
             "ALEXA 35"),
    LogCurve("slog3", "Sony S-Log3", "Sony S-Log3", _f32(_slog3_enc), _f32(_slog3_dec),
             f"x >= {_SL3_CUT} ? (420 + log10(max((x + 0.01)/0.19, 1e-10))*261.5)/1023 : (x*{_r(_SL3_YCUT - 95.0)}/{_SL3_CUT} + 95)/1023",
             f"x >= {_r(_SL3_YCUT / 1023.0)} ? pow(10, (x*1023 - 420)/261.5)*0.19 - 0.01 : (x*1023 - 95)*{_SL3_CUT}/{_r(_SL3_YCUT - 95.0)}",
             "reflectance input (0.18 → 0.4106)"),
    LogCurve("video", "video (sRGB)", "video (sRGB)", _f32(_srgb_enc), _f32(_srgb_dec),
             "x <= 0.0031308 ? 12.92*x : 1.055*pow(max(x, 0), 1/2.4) - 0.055",
             "x <= 0.04045 ? x/12.92 : pow((max(x, 0) + 0.055)/1.055, 2.4)",
             "sRGB OETF – 8-bit sources (PNG / MP4 plates)"),
]}
CURVE_IDS: List[str] = [NONE] + list(LOG_CURVES)          # knob order: index 0 = none
MENU: List[str] = [NONE] + [c.menu for c in LOG_CURVES.values()]


def curve_index(cid: Optional[str]) -> int:
    """Index of the Nuke enumeration knob / the GLSL `curve` uniform (0 = none)."""
    return CURVE_IDS.index(cid) if cid in CURVE_IDS else 0


def get(cid: Optional[str]) -> Optional[LogCurve]:
    """The curve of an id; None for `none` / unknown. `project` must be resolved by the caller (ValueError)."""
    if cid == PROJECT:
        raise ValueError("log_curve 'project' – resolve it with the project's colour profile first")
    return LOG_CURVES.get(cid or NONE)


def legacy_curve(log: object) -> str:
    """The old bool knob `log` (MergeMatte v01 until 2026-11-03): true = ACEScct (the only curve then), false = none."""
    return "acescct" if log else NONE
