# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate colour profile of a shot (stage 0; user decision 2026-11-03): which lin ↔ log CURVE (engine/log_curves.py) the
MergeMatte operations and the .nk export use by default. Proposed automatically from what the plate tells – the source
type and bit depth, the manifest's `colorspace`, the EXR header (ACES container flag, camera metadata of ARRI / Sony /
RED) – with a reason; the user can override it in stage 0. Stored ONLY in project.json as `color_profile`
(`auto` | a curve id), never in the shot manifest. No OCIO lists, no gamut conversion: a curve only.

    proposal(manifest, header)  → {"id", "reason"}            the automatic choice
    resolve(setting, manifest, header) → {"setting", "id", "label", "reason", "auto": {...}}   what the project uses
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from engine import log_curves as lc

KEY = "color_profile"
AUTO = "auto"
CHOICES = [AUTO] + list(lc.LOG_CURVES)

# header hints (EXR attribute names / string values, lower case) → curve; first match in this order wins
_CAMERA_HINTS = [
    (("logc4", "alexa 35", "alexa35"), "logc4", "metadane ARRI (ALEXA 35 / LogC4) w nagłówku EXR"),
    (("arri", "alexa", "logc"), "logc3", "metadane ARRI w nagłówku EXR"),
    (("s-log3", "slog3", "sony", "venice"), "slog3", "metadane Sony w nagłówku EXR"),
    (("redlog", "red digital", "red_", "redcode", "r3d"), "cineon", "metadane RED w nagłówku EXR (brak krzywej RED – Cineon jako najbliższa)"),
]
_COLORSPACE_HINTS = [
    (("logc4",), "logc4", "colorspace manifestu wskazuje ARRI LogC4"),
    (("logc", "arri", "alexa"), "logc3", "colorspace manifestu wskazuje ARRI LogC3"),
    (("s-log3", "slog3", "sony"), "slog3", "colorspace manifestu wskazuje Sony S-Log3"),
    (("cineon", "adx", "log"), "cineon", "colorspace manifestu wskazuje Cineon / log"),
    (("srgb", "rec709", "rec.709", "bt709", "video"), "video", "colorspace manifestu to sRGB / Rec.709 (wideo)"),
    (("aces",), "acescct", "colorspace manifestu to ACES"),
]


def header_hints(header: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The EXR header reduced to lower-case text per attribute (strings / bytes / numbers; images skipped)."""
    out: Dict[str, str] = {}
    for k, v in (header or {}).items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if isinstance(v, (str, int, float)):
            out[str(k).lower()] = str(v).lower()
    return out


def read_header(sample_file: Optional[str]) -> Dict[str, Any]:
    """The OpenEXR header of a plate frame ({} for a non-EXR / unreadable file or without the OpenEXR module)."""
    if not sample_file or not str(sample_file).lower().endswith(".exr") or not Path(sample_file).exists():
        return {}
    try:
        import OpenEXR
        f = OpenEXR.InputFile(str(sample_file))
        try:
            return dict(f.header())
        finally:
            f.close()
    except Exception:
        return {}


def proposal(manifest: Optional[Mapping[str, Any]], header: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """The automatic curve: EXR camera metadata > ACES container flag / ACES colorspace > manifest colorspace naming a
    curve > 8-bit / video source → video > Cineon."""
    m = manifest or {}
    hints = header_hints(header)
    text = " ".join(f"{k}={v}" for k, v in hints.items())
    for needles, cid, why in _CAMERA_HINTS:
        if any(n in text for n in needles):
            return {"id": cid, "reason": why}
    if any("acesimagecontainerflag" in k for k in hints) or hints.get("acesimagecontainerflag") in ("1", "true"):
        return {"id": "acescct", "reason": "flaga kontenera ACES w nagłówku EXR"}
    cs = str(m.get("colorspace") or "").lower()
    for needles, cid, why in _COLORSPACE_HINTS:
        if cs and any(n in cs for n in needles):
            return {"id": cid, "reason": f"{why} („{m.get('colorspace')}”)"}
    st = str(m.get("source_type") or "").lower()
    depth = m.get("bit_depth")
    pattern = str((m.get("plates") or {}).get("beauty") or "")
    ext = pattern.rsplit(".", 1)[-1].lower() if "." in pattern else ""
    if st == "video" or (depth is not None and int(depth) <= 8) or ext in ("png", "jpg", "jpeg", "mp4", "mov", "mkv", "avi"):
        return {"id": "video", "reason": "źródło 8-bit / wideo (sRGB)"}
    return {"id": "cineon", "reason": "brak wskazówek w metadanych – Cineon jako ogólna krzywa log"}


def resolve(setting: Optional[str], manifest: Optional[Mapping[str, Any]], header: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """project.json `color_profile` (`auto` / absent / a curve id) → the curve in use, with the automatic proposal."""
    auto = proposal(manifest, header)
    s = str(setting or AUTO)
    if s not in CHOICES:
        s = AUTO
    cid = auto["id"] if s == AUTO else s
    return {"setting": s, "id": cid, "label": lc.LOG_CURVES[cid].label,
            "reason": auto["reason"] if s == AUTO else "ustawiony ręcznie w etapie 0", "auto": auto}


def project_setting(project_data: Optional[Mapping[str, Any]]) -> str:
    return str((project_data or {}).get(KEY) or AUTO)


def validate(setting: str) -> str:
    s = str(setting or AUTO).lower()
    if s not in CHOICES:
        raise ValueError(f"Profil koloru musi być jednym z: {', '.join(CHOICES)}")
    return s


def sample_file(manifest: Optional[Mapping[str, Any]]) -> Optional[str]:
    """First frame of the beauty plate (for the EXR header), None when the pattern has no file on disk."""
    pattern = str(((manifest or {}).get("plates") or {}).get("beauty") or "")
    if not pattern:
        return None
    p = Path(pattern)
    m = re.search(r"#+|%0?(\d*)d", p.name)
    if not m:
        return pattern if p.exists() else None
    stem = re.escape(p.name[:m.start()]) + r"\d+" + re.escape(p.name[m.end():])
    try:
        files = sorted(f for f in p.parent.iterdir() if re.fullmatch(stem, f.name))
    except OSError:
        return None
    return str(files[0]) if files else None
