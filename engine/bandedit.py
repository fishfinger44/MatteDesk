# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandedit.py - THE edit of band control points (one band system, CLAUDE.md „Pas krawędzi = jeden system”):
what a band point editor (static/bandpoints.js) asks a server for, independent of WHO owns the band. Moved out of app.py
2026-11-10 (P14 „Pas Czyszczenia”) so the object bands (stages 3 / 4 / 7, whole-outline zones – app.py `_BandOwner`) and
the Cleanup 3D patch band (api/cleanup3d.py, owner „c3d”) edit points with the same code:

  state_points(pts, frame, positions, core, so, si)   the points of a frame as the editor shows them (snapped, whiskers)
  put_point(pts, frame, current, defaults, so, si, …)  add a point / set its key in a frame
  delete_point(pts, point_id, frame)                   remove a point or its key in a frame
  delete_keys(pts, frame)                              the keys of every point in a frame, or every point

Errors are `BandEditError` with the HTTP status the routes answer (400 / 404 / 409) and a Polish message."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BandEditError(ValueError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = int(status)
        self.detail = detail


def ang_field(new: Optional[float], cur: Any, max_deg: Optional[float] = None) -> Dict[str, float]:
    """The key's tilt: the request's, else the one shown now; 0 is not stored (keys of old projects stay as they are).
    `max_deg`: None = the stages' ±80; the Cleanup C1 band passes 180 (a whisker all round)."""
    from engine.bandmap import MAX_ANGLE_DEG
    m = MAX_ANGLE_DEG if max_deg is None else float(max_deg)
    a = float(max(-m, min(m, float(new if new is not None else (cur or 0.0)))))
    return {"ang": round(a, 2)} if a else {}


def state_points(pts: Sequence[Dict[str, Any]], frame: int, positions: Dict[str, Tuple[Any, Any, bool]],
                 core: Optional[np.ndarray], so: float, si: float, max_angle: Optional[float] = None) -> List[Dict[str, Any]]:
    """The points of `frame` as the editor draws them: position (tracked / nearest key), visibility, the widths scaled by
    the sliders (so / si – objband.scales), the tilt, whether `frame` is a key, all keys and – with an outline `core` –
    the snapped outline pixel with its normal / tangent (sx, sy, nx, ny, tx, ty)."""
    from engine.bandmap import _contours, angle_at, snap_frame, widths_at
    cs = _contours(core) if core is not None else None     # one outline trace for all the points of the frame
    out = []
    for q in pts:
        x, y, vis = positions.get(q["id"], (None, None, False))
        o, i = widths_at(q, frame)
        o, i = o * so, i * si
        sn = snap_frame(core, x, y, cs) if core is not None and x is not None else None
        out.append({"id": q["id"], "x": x, "y": y, "visible": vis, "out": round(o, 2), "in": round(i, 2),
                    "ang": round(angle_at(q, frame) if max_angle is None else angle_at(q, frame, max_angle), 2),
                    "key": str(frame) in q["keys"], "keys": sorted(int(k) for k in q["keys"]),
                    **({"sx": round(sn[0], 2), "sy": round(sn[1], 2), "nx": round(sn[2], 4), "ny": round(sn[3], 4),
                        "tx": round(sn[4], 4), "ty": round(sn[5], 4)} if sn else {})})
    return out


def put_point(pts: Sequence[Dict[str, Any]], frame: int, current: Dict[str, Dict[str, Any]], defaults: Dict[str, Any],
              so: float, si: float, *, id: Optional[str] = None, x: Optional[float] = None, y: Optional[float] = None,
              out: Optional[float] = None, in_: Optional[float] = None, ang: Optional[float] = None,
              max_angle: Optional[float] = None) -> Tuple[List[Dict[str, Any]], str]:
    """(points, point id) after adding a point (`id` None: x, y needed; widths default to `defaults` out / in) or setting
    the key of point `id` in `frame` (absent fields = what is shown now, `current` = state_points by id; the position
    snapped to the outline). Widths come in shown px and are stored unscaled (/ so, / si)."""
    from engine.bandmap import MAX_WIDTH_PX
    pts = [dict(p) for p in pts]
    clamp = lambda v: float(min(MAX_WIDTH_PX, max(0.0, v)))
    raw_o = lambda v: clamp(v) / so if so > 1e-6 else clamp(v)     # shown (scaled) px -> stored px
    raw_i = lambda v: clamp(v) / si if si > 1e-6 else clamp(v)
    if id is None:
        if x is None or y is None:
            raise BandEditError(400, "Nowy punkt pasa: podaj x i y")
        n = 1 + max([int(p["id"][1:]) for p in pts if str(p["id"]).startswith("p") and str(p["id"])[1:].isdigit()] or [0])
        pts.append({"id": f"p{n}", "keys": {str(frame): {"x": float(x), "y": float(y),
                                                         "out": raw_o(out if out is not None else defaults["out"]),
                                                         "in": raw_i(in_ if in_ is not None else defaults["in"]),
                                                         **ang_field(ang, 0.0, max_angle)}}})
        return pts, f"p{n}"
    p = next((q for q in pts if q["id"] == id), None)
    if p is None:
        raise BandEditError(404, f"Brak punktu pasa {id}")
    c = current.get(id, {})
    px = x if x is not None else c.get("sx", c.get("x"))     # snapped to the outline (station 2026-10-21:
    py = y if y is not None else c.get("sy", c.get("y"))     # a raw track had drifted 500 px into the sky)
    if px is None or py is None:
        raise BandEditError(409, "Punkt jest w tej klatce niewidoczny – podaj jego położenie (x, y)")
    p["keys"] = {**p["keys"], str(frame): {"x": float(px), "y": float(py),
                                           "out": raw_o(out if out is not None else c.get("out", 0)),
                                           "in": raw_i(in_ if in_ is not None else c.get("in", 0)),
                                           **ang_field(ang, c.get("ang", 0.0), max_angle)}}
    return pts, id


def delete_point(pts: Sequence[Dict[str, Any]], point_id: str, frame: Optional[int]) -> List[Dict[str, Any]]:
    """Points without `point_id` (frame None) or without its key in `frame` (the last key removes the point)."""
    pts = [dict(p) for p in pts]
    p = next((q for q in pts if q["id"] == point_id), None)
    if p is None:
        raise BandEditError(404, f"Brak punktu pasa {point_id}")
    if frame is not None:
        if str(frame) not in p["keys"]:
            raise BandEditError(404, f"Punkt {point_id} nie ma klucza w klatce {frame}")
        p["keys"] = {k: v for k, v in p["keys"].items() if k != str(frame)}
    if frame is None or not p["keys"]:
        pts = [q for q in pts if q["id"] != point_id]
    return pts


def delete_keys(pts: Sequence[Dict[str, Any]], frame: Optional[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(points, removed ids): every point (frame None = the constant band again) or the keys of all points in `frame`
    (a point left without keys goes too)."""
    pts = [dict(p) for p in pts]
    if frame is None:
        return [], [p["id"] for p in pts]
    removed = [p["id"] for p in pts if str(frame) in p["keys"]]
    if not removed:
        raise BandEditError(404, f"Brak kluczy pasa w klatce {frame}")
    for p in pts:
        p["keys"] = {k: v for k, v in p["keys"].items() if k != str(frame)}
    return [p for p in pts if p["keys"]], removed
