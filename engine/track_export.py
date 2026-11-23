# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/track_export.py - the band's tracker for Nuke (Tracker4) and After Effects (nulls) (user request 2026-11-23:
„jeżeli maska się rusza, to też mamy jej trackera”; next to the band shapes of engine/band_export in the same „⇪ Export”).

Tracks:
  * every band point (followed by CoTracker3, the positions the editor shows) = one track „pkt_<n>” (n = the order of
    the points); frames where the point is hidden (occluded / outside its lifetime) are left out of the track;
  * „srodek_maski” = the area centroid of the band's outer edge in every frame (engine/band_export.split → „outer”):
    the motion of the masked thing even without band points. It moves with the shape too (a hand going out = a shift).

Nuke (`nk_text`, format checked on Nuke 15.0v1 2026-11-23: nodePaste + `tracks.getValueAt` gives the written values):
one `Tracker4` (no input) with a track per row, track_x / track_y as curves `x<f> v`; points plate px -> Nuke
(x + 0.5, H − y − 0.5) like the Roto. Use: Transform / CornerPin from the tracker, or „match-move” in its Transform tab.

After Effects (`jsx_text`): a script adding one Null per track to the ACTIVE comp, Position keyed on the track's frames
(plate frame `first` = comp time 0; another comp size = positions scaled).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

CENTRE = "srodek_maski"

# the Tracker4 table columns as Nuke 15 writes them (31 per track; track_x / track_y = columns 2 / 3)
_COLS = [("5 1 20", "enable", "e"), ("3 1 75", "name", "name"), ("2 1 58", "track_x", "track_x"),
         ("2 1 58", "track_y", "track_y"), ("2 1 63", "offset_x", "offset_x"), ("2 1 63", "offset_y", "offset_y"),
         ("4 1 27", "T", "T"), ("4 1 27", "R", "R"), ("4 1 27", "S", "S"), ("2 0 45", "error", "error")] + \
        [("1 1 0", n, n) for n in ("error_min", "error_max", "pattern_x", "pattern_y", "pattern_r", "pattern_t",
                                   "search_x", "search_y", "search_r", "search_t")] + \
        [("2 1 0", n, n) for n in ("key_track", "key_search_x", "key_search_y", "key_search_r", "key_search_t",
                                   "key_track_x", "key_track_y", "key_track_r", "key_track_t", "key_centre_offset_x",
                                   "key_centre_offset_y")]
assert len(_COLS) == 31


def centroid(polys: Sequence[np.ndarray]):
    """Area centroid of the outer polygons (x, y plate px) or None."""
    A = cx = cy = 0.0
    for p in polys:
        p = np.asarray(p, np.float64).reshape(-1, 2)
        if len(p) < 3:
            continue
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        a = 0.5 * float(c.sum())
        if abs(a) < 1e-9:
            continue
        A += a
        cx += float(((x + x1) * c).sum()) / 6.0
        cy += float(((y + y1) * c).sum()) / 6.0
    if abs(A) < 1e-9:
        return None
    return cx / A, cy / A


def tracks(points_by_frame: Dict[int, List[Dict[str, Any]]], lines_by_frame: Dict[int, Dict[str, Any]]
           ) -> List[Dict[str, Any]]:
    """[{name, keys: {frame: (x, y)}}]: the band points (order of first appearance) + the mask centre."""
    from engine.band_export import split
    order: List[str] = []
    keys: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for f in sorted(points_by_frame):
        for q in points_by_frame[f] or []:
            pid = str(q.get("id"))
            if pid not in keys:
                order.append(pid)
                keys[pid] = {}
            if q.get("visible", True) is False or q.get("x") is None or q.get("y") is None:
                continue
            keys[pid][int(f)] = (float(q["x"]), float(q["y"]))
    out = [{"name": f"pkt_{i + 1}", "id": pid, "keys": keys[pid]} for i, pid in enumerate(order) if keys[pid]]
    cen: Dict[int, Tuple[float, float]] = {}
    for f in sorted(lines_by_frame):
        c = centroid(split(lines_by_frame[f])["outer"])
        if c is not None:
            cen[int(f)] = c
    if cen:
        out.append({"name": CENTRE, "id": None, "keys": cen})
    return out


def _curve(keys: Dict[int, float]) -> str:
    return "{curve " + " ".join(f"x{f} {v:.3f}" for f, v in sorted(keys.items())) + "}"


def nk_text(trk: Sequence[Dict[str, Any]], w: int, h: int, node: str, label: str) -> str:
    """Nuke: one Tracker4 with a track per entry (clipboard-ready, like engine/band_export.nk_text)."""
    cols = " \n".join("{ %s %s %s 1 }" % c for c in _COLS)
    rows = []
    for t in trk:
        xs = {f: x + 0.5 for f, (x, _y) in t["keys"].items()}
        ys = {f: h - y - 0.5 for f, (_x, y) in t["keys"].items()}
        name = re.sub(r'["\\{}]', "_", t["name"])
        rows.append('{ {curve K x1 1} "%s" %s %s {} {} 1 0 0 {} 1 0 -32 -32 32 32 -22 -22 22 22 '
                    '{} {} {} {} {} {} {} {} {} {} {} }' % (name, _curve(xs), _curve(ys)))
    lab = label.replace("\\", "/").replace('"', "'").replace("{", "(").replace("}", ")")
    return ("set cut_paste_input [stack 0]\nversion 15.0 v1\nTracker4 {\n inputs 0\n"
            " tracks { { 1 31 %d } \n{ %s } \n{ \n %s \n} \n}\n name %s\n label \"%s\"\n}\n"
            % (len(rows), cols, " \n ".join(rows), node, lab))


def jsx_text(trk: Sequence[Dict[str, Any]], w: int, h: int, first: int, prefix: str, label: str) -> str:
    """After Effects script: a Null per track in the active comp, Position keyed (see the module doc)."""
    data = [{"n": t["name"], "k": [[int(f), round(x + 0.5, 2), round(y + 0.5, 2)] for f, (x, y) in sorted(t["keys"].items())]}
            for t in trk]
    return """// MatteDesk – %(label)s
// After Effects: File → Scripts → Run Script File… with the plate's comp open (active).
// One Null per track (band points + the mask centre); plate frame FIRST = comp time 0, another comp size = scaled.
(function () {
  var W = %(w)d, H = %(h)d, FIRST = %(first)d, PREFIX = %(prefix)s;
  var T = %(data)s;
  var comp = app.project.activeItem;
  if (!(comp && comp instanceof CompItem)) { alert("MatteDesk: otwórz kompozycję z plate'em (aktywną) i uruchom skrypt ponownie."); return; }
  var sx = comp.width / W, sy = comp.height / H, dt = comp.frameDuration;
  app.beginUndoGroup("MatteDesk: " + PREFIX);
  for (var i = 0; i < T.length; i++) {
    var t = T[i], L = comp.layers.addNull(comp.duration);
    L.name = PREFIX + " " + t.n; L.comment = %(label_js)s;
    L.property("ADBE Transform Group").property("ADBE Anchor Point").setValue([50, 50]);
    var pos = L.property("ADBE Transform Group").property("ADBE Position"), times = [], vals = [];
    for (var k = 0; k < t.k.length; k++) { times.push((t.k[k][0] - FIRST) * dt); vals.push([t.k[k][1] * sx, t.k[k][2] * sy]); }
    if (vals.length == 1) pos.setValue(vals[0]); else pos.setValuesAtTimes(times, vals);
  }
  app.endUndoGroup();
})();
""" % {"label": label.replace("\n", " "), "w": w, "h": h, "first": int(first), "prefix": json.dumps(prefix),
       "data": json.dumps(data, separators=(",", ":")), "label_js": json.dumps(label)}


def summary(trk: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {"tracks": len(trk), "points": sum(1 for t in trk if t["id"] is not None),
            "centre": int(any(t["id"] is None for t in trk))}
