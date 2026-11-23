# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/band_export.py - a stage band as editable shapes for Nuke (Roto) and After Effects (masks) (user request
2026-11-23, like the „⇪ Export” of Final Touch).

Input = the band preview lines of every frame (engine/bandsys.preview_lines via GET …/stage_band/{stage}/{f}/lines):
`band_contours` (the ring: outer edges + holes = the inner edge, RETR_CCOMP), `inset_feather_contours` (where the inset
feather ends), `feather_contours` (where the outer feather ends).

Shapes: every contour is followed across frames (`track`: bbox overlap / centroid with the previous frame; a contour that
vanishes ends its shape, one that appears starts a new one), resampled to ONE point count per shape (uniform arc
length, `spacing_px`, cyclic start matched to the previous frame), so a shape is an animated curve with a key on every
frame and a lifetime = its frames – editable in Nuke / AE (not one shape per frame).

Nuke (`nk_text`, checked on Nuke 15.0v1 batch renders 2026-11-23): one `Roto` (no input, plate format) – layer „pas”
= outer edges (over) + holes (blending mode 12 = out: cut what is under them; the first shape of a list renders on top,
so the holes are written first), layers „koniec_wtopienia_wciecia” / „koniec_wtopienia_zewn” hidden (`vis 0`) – the
Roto's alpha = the band ring, the feather ends are there to switch on. Points: plate px -> Nuke (x + 0.5, H − y − 0.5);
an animated point = `{{{f x} …} {{f y} …}}` (Nuke fills the slopes), a curvegroup needs its feather `cc` block.
Shapes nested in a hole (an island inside the ring's hole) are cut by the hole too – rare, the outline is still there.

After Effects (`jsx_text`): a script (File → Scripts → Run Script File…) adding to the ACTIVE comp a solid with one mask
per shape: outer edges Add, holes Subtract, feather ends None; Mask Path keys on every frame, Mask Opacity 0 outside
the shape's lifetime (hold keys). Plate frame `first` = comp time 0; a comp of another size gets the shapes scaled.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROLES = ("outer", "inner", "inset_end", "outer_end")
ROLE_NAME = {"outer": "pas_zewn", "inner": "pas_wewn", "inset_end": "koniec_wciecia", "outer_end": "koniec_wtopienia"}
ROLE_LABEL = {"outer": "zewnętrzna krawędź pasa", "inner": "wewnętrzna krawędź pasa",
              "inset_end": "koniec wtopienia wcięcia", "outer_end": "koniec wtopienia zewnętrznego"}
SPACING_PX = 4.0          # one point per ~4 px of the longest outline of a shape
MIN_PTS, MAX_PTS = 8, 400
HOLE_BM = 12              # Nuke Roto blending mode „out” (measured 2026-11-23: B·(1 − a))
# the Root layer's planar-track attributes, as Nuke 15 writes them for a fresh Roto (same as engine/ft_nuke)
from engine.ft_nuke import LAYER_ATTRS  # noqa: E402


def _area(poly: np.ndarray) -> float:
    """Signed area (image coords, y down): OpenCV outer contours < 0, holes > 0."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def split(lines: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    """{role: [poly (n, 2) float]} of one frame's lines dict (holes of the band ring by orientation)."""
    out: Dict[str, List[np.ndarray]] = {r: [] for r in ROLES}
    for c in lines.get("band_contours") or []:
        p = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        if len(p) >= 3:
            out["inner" if _area(p) > 0 else "outer"].append(p)
    for key, role in (("inset_feather_contours", "inset_end"), ("feather_contours", "outer_end")):
        for c in lines.get(key) or []:
            p = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            if len(p) >= 3:
                out[role].append(p)
    return out


def _bbox(p: np.ndarray) -> Tuple[float, float, float, float]:
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def _match_score(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two outlines of neighbouring frames are: bbox IoU, or (no overlap) a small score by centroid distance
    relative to the size; 0 = not the same outline."""
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    ua = (ax1 - ax0 + 1) * (ay1 - ay0 + 1) + (bx1 - bx0 + 1) * (by1 - by0 + 1)
    if iw > 0 and ih > 0:
        inter = (iw + 1) * (ih + 1)
        return inter / max(1.0, ua - inter)
    size = max(ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0, 8.0)
    d = math.hypot((ax0 + ax1 - bx0 - bx1) / 2, (ay0 + ay1 - by0 - by1) / 2)
    return 0.05 * max(0.0, 1.0 - d / (0.5 * size))


def track(frames: Sequence[int], polys: Dict[int, List[np.ndarray]]) -> List[Dict[int, np.ndarray]]:
    """Outlines followed across consecutive frames: [{frame: poly}] – greedy best match with the previous frame's
    live shapes; a frame without a match ends a shape (a gap in `frames` ends them all)."""
    tracks: List[Dict[int, np.ndarray]] = []
    live: List[int] = []
    prev_f: Optional[int] = None
    for f in frames:
        cur = polys.get(f) or []
        if prev_f is None or f != prev_f + 1:
            live = []
        pairs = sorted(((_match_score(tracks[t][prev_f], p), t, i) for t in live for i, p in enumerate(cur)),
                       reverse=True) if live else []
        used_t, used_i, nxt = set(), set(), []
        for s, t, i in pairs:
            if s <= 0 or t in used_t or i in used_i:
                continue
            tracks[t][f] = cur[i]
            used_t.add(t); used_i.add(i); nxt.append(t)
        for i, p in enumerate(cur):
            if i not in used_i:
                tracks.append({f: p})
                nxt.append(len(tracks) - 1)
        live, prev_f = nxt, f
    return tracks


def _perimeter(p: np.ndarray) -> float:
    return float(np.sum(np.hypot(*(np.roll(p, -1, axis=0) - p).T)))


def resample(p: np.ndarray, n: int) -> np.ndarray:
    """n points at equal arc length along the closed polyline p, starting at p[0]."""
    q = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(q, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(p[:1], n, axis=0).astype(np.float64)
    t = np.arange(n) * (cum[-1] / n)
    return np.stack([np.interp(t, cum, q[:, 0]), np.interp(t, cum, q[:, 1])], axis=1)


def align(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """`cur` rolled so its points sit nearest to `prev` (same count) – no point slides around the outline between
    frames. The orientation is kept (OpenCV gives every outline of a role the same direction)."""
    n = len(cur)
    d = np.stack([np.sum((np.roll(cur, -k, axis=0) - prev) ** 2) for k in range(n)]) if n <= 64 else None
    if d is None:   # long outlines: candidate starts near prev[0], then the exact sum on those
        near = np.argsort(np.sum((cur - prev[0]) ** 2, axis=1))[:24]
        best = min(near, key=lambda k: float(np.sum((np.roll(cur, -int(k), axis=0) - prev) ** 2)))
        return np.roll(cur, -int(best), axis=0)
    return np.roll(cur, -int(np.argmin(d)), axis=0)


def shapes(lines_by_frame: Dict[int, Dict[str, Any]], spacing_px: float = SPACING_PX,
           max_pts: int = MAX_PTS) -> List[Dict[str, Any]]:
    """[{name, role, first, last, pts (frames, n, 2) plate px}] of every band outline over the frames given."""
    frames = sorted(int(f) for f in lines_by_frame)
    per = {f: split(lines_by_frame[f]) for f in frames}
    out: List[Dict[str, Any]] = []
    for role in ROLES:
        tr = track(frames, {f: per[f][role] for f in frames})
        for k, t in enumerate(tr, 1):
            fr = sorted(t)
            n = int(min(max_pts, max(MIN_PTS, round(max(_perimeter(t[f]) for f in fr) / max(0.5, spacing_px)))))
            pts, prev = [], None
            for f in fr:
                r = resample(t[f], n)
                if prev is not None:
                    r = align(r, prev)
                pts.append(r)
                prev = r
            out.append({"name": f"{ROLE_NAME[role]}_{k}", "role": role, "first": fr[0], "last": fr[-1],
                        "pts": np.stack(pts)})
    return out


def _f(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "/").replace('"', '\\"') + '"'


def _curvegroup(s: Dict[str, Any], h: int) -> str:
    fr = list(range(int(s["first"]), int(s["last"]) + 1))
    pts = s["pts"]
    cps = []
    for j in range(pts.shape[1]):
        if len(fr) == 1:
            x, y = pts[0, j]
            cps.append("{} {%s %s 1} {}" % (_f(x + 0.5), _f(h - y - 0.5)))
            continue
        xs = " ".join("{%d %s}" % (f, _f(pts[i, j, 0] + 0.5)) for i, f in enumerate(fr))
        ys = " ".join("{%d %s}" % (f, _f(h - pts[i, j, 1] - 0.5)) for i, f in enumerate(fr))
        cps.append("{} {{%s} {%s}} {}" % (xs, ys))
    feather = "{cc {f 8192} {p %s}}" % " ".join(["{}"] * (3 * pts.shape[1]))
    attrs = (["bm %d" % HOLE_BM] if s["role"] == "inner" else []) + \
        [f"ltn {int(s['first'])}", f"ltm {int(s['last'])}", "ltt 3", "str 1"]
    return "{curvegroup %s 512 bezier {{cc {f 8192} {p %s}} %s} {tx 0 0} {a %s}}" % (
        s["name"], " ".join(cps), feather, " ".join(attrs))


def nk_text(shp: Sequence[Dict[str, Any]], w: int, h: int, node: str, label: str) -> str:
    """Clipboard / .nk text: one Roto with the band (see the module doc). Ctrl+V in Nuke's node graph."""
    by = {r: [s for s in shp if s["role"] == r] for r in ROLES}
    band = [_curvegroup(s, h) for s in by["inner"]] + [_curvegroup(s, h) for s in by["outer"]]   # holes on top
    layers = []
    for role, lname in (("inset_end", "koniec_wtopienia_wciecia"), ("outer_end", "koniec_wtopienia_zewn")):
        if by[role]:
            layers.append("{layer %s {f 0} {t 0} {a vis 0} %s}" % (lname, " ".join(_curvegroup(s, h) for s in by[role])))
    if band:
        layers.append("{layer pas {f 0} {t 0} {a vis 1} %s}" % " ".join(band))
    curves = "{{{v x3f99999a} {f 0} {n {layer Root {f 0} {t %s %s} {a %s} %s}}}}" % (
        _f(w / 2), _f(h / 2), LAYER_ATTRS, " ".join(layers))
    fmt = f'"{w} {h} 0 0 {w} {h} 1 MatteDesk_{w}x{h}"'
    lines = ["Roto {", " inputs 0", f" format {fmt}", " output alpha", f" curves {curves}", f" name {node}",
             f" label {_q(label)}", "}"]
    return "set cut_paste_input [stack 0]\nversion 15.0 v1\n" + "\n".join(lines) + "\n"


AE_MODE = {"outer": "ADD", "inner": "SUBTRACT", "inset_end": "NONE", "outer_end": "NONE"}


def jsx_text(shp: Sequence[Dict[str, Any]], w: int, h: int, first: int, layer: str, label: str) -> str:
    """After Effects script: a solid in the active comp with the band as animated masks (see the module doc)."""
    order = [s for r in ("outer", "inner", "inset_end", "outer_end") for s in shp if s["role"] == r]
    data = []
    for s in order:
        fr = [[[round(float(x) + 0.5, 1), round(float(y) + 0.5, 1)] for x, y in frame] for frame in s["pts"]]
        data.append({"n": s["name"], "m": AE_MODE[s["role"]], "f": int(s["first"]), "p": fr})
    return """// MatteDesk – %(label)s
// After Effects: File → Scripts → Run Script File… with the plate's comp open (active).
// Plate frame FIRST = comp time 0; a comp of another size than the plate gets the shapes scaled.
(function () {
  var W = %(w)d, H = %(h)d, FIRST = %(first)d, NAME = %(name)s;
  var S = %(data)s;
  var comp = app.project.activeItem;
  if (!(comp && comp instanceof CompItem)) { alert("MatteDesk: otwórz kompozycję z plate'em (aktywną) i uruchom skrypt ponownie."); return; }
  var sx = comp.width / W, sy = comp.height / H, dt = comp.frameDuration;
  app.beginUndoGroup("MatteDesk: " + NAME);
  var L = comp.layers.addSolid([1, 1, 1], NAME, comp.width, comp.height, comp.pixelAspect, comp.duration);
  L.comment = %(label_js)s;
  var MODES = {ADD: MaskMode.ADD, SUBTRACT: MaskMode.SUBTRACT, NONE: MaskMode.NONE};
  for (var i = 0; i < S.length; i++) {
    var s = S[i], m = L.Masks.addProperty("ADBE Mask Atom");
    m.name = s.n; m.maskMode = MODES[s.m];
    var times = [], vals = [];
    for (var k = 0; k < s.p.length; k++) {
      var sh = new Shape(), v = [];
      for (var j = 0; j < s.p[k].length; j++) v.push([s.p[k][j][0] * sx, s.p[k][j][1] * sy]);
      sh.vertices = v; sh.closed = true;
      times.push((s.f + k - FIRST) * dt); vals.push(sh);
    }
    var path = m.property("ADBE Mask Shape");
    if (vals.length == 1) path.setValue(vals[0]); else path.setValuesAtTimes(times, vals);
    var op = m.property("ADBE Mask Opacity"), t0 = times[0], t1 = times[times.length - 1];
    if (t0 > 0) { op.setValueAtTime(0, 0); op.setValueAtTime(t0, 100); }
    if (t1 + dt < comp.duration) { op.setValueAtTime(t0, 100); op.setValueAtTime(t1 + dt, 0); }
    for (var q = 1; q <= op.numKeys; q++) op.setInterpolationTypeAtKey(q, KeyframeInterpolationType.HOLD);
  }
  app.endUndoGroup();
})();
""" % {"label": label.replace("\n", " "), "w": w, "h": h, "first": int(first), "name": json.dumps(layer),
       "data": json.dumps(data, separators=(",", ":")), "label_js": json.dumps(label)}


def summary(shp: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {r: sum(1 for s in shp if s["role"] == r) for r in ROLES}
