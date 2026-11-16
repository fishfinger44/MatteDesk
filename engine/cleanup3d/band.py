# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cleanup3d/band.py - the band of the Cleanup 3D key patch (P14 „Pas Czyszczenia”, docs/PLAN_CLEANUP_3D.md;
user decision: the patch blend works like the app's ONE band system). Glue only – the band itself is
engine/objband.cleanup_band (the settings model), engine/bandsys (geometry, rule, preview lines), engine/bandmap (points,
widths along the outline), engine/bandtrack (positions between keys), engine/bandedit (the point edits) and
static/bandpoints.js (the editor, owner „c3d”).

The patch of a frame (`patch_weight`, used by render.KeyFill._patch and – at preview size – by the server side of the
live preview; static/cleanup3d.js `patchAlpha` is the same rule on the browser's live hole):
  hole  = the removal mask U the object grown by `key_grow_px` U the object's traces (alpha 1, the base mask of the band)
  band  = OUTSIDE the hole, within out(x) of it (bandsys.band_area, inset 0); out = „Wtopienie łaty” (`key_soft_px`),
          shaped along the object's outline by the band points (bandsys.band_widths; the points snap to the OBJECT
          part of the hole – removal mask U grown object –, not to the traces, so the browser's live threshold never
          moves them)
  patch = 1 in the hole, smoothstep(bandsys weight) in the band: the outer feather (= key_soft_px) runs from the outer
          band edge inwards – without points out = feather, the ramp spans the band (the patch before P14); a point
          pushing out keeps the patch at 1 up to out − feather and ramps over the last `key_soft_px`.
Only the outer side applies (the inset / blue whisker means nothing for a patch).

Positions between keys: the nearest key carried by the object's own motion (bandtrack.track_positions with `core_at` =
the removal objects – centroid / scale), then snapped to the outline. CoTracker3 tracking of these points is not wired
(the object bands' tracker works on an object's crop frames); the CPU path is what the render and the editor use.

Storage: <work>/inpaint/c3d_band.json {schema, band_points, band_ref_out, band_ref_in} (per project, beside
c3d_settings.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine import bandsys
from engine.cleanup3d import types as T

SCHEMA = "matte_desk/c3d_band@1.0"
KEYS = ("band_points", "band_ref_out", "band_ref_in")


# ----------------------------------------------------------------------------- storage
def band_file(work_dir: Path) -> Path:
    return T.root_dir(work_dir) / "c3d_band.json"


def load(work_dir: Path) -> Dict[str, Any]:
    """{band_points, band_ref_out, band_ref_in} of the project (empty band when there is no file / a broken one)."""
    import json
    p = band_file(work_dir)
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        d = {}
    return {"band_points": list(d.get("band_points") or []), "band_ref_out": d.get("band_ref_out"),
            "band_ref_in": d.get("band_ref_in")}


def save(work_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    from engine.jsonio import write_json
    pts = list(params.get("band_points") or [])
    d = {"band_points": pts, "band_ref_out": params.get("band_ref_out") if pts else None,
         "band_ref_in": params.get("band_ref_in") if pts else None}
    p = band_file(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {"schema": SCHEMA, **d})
    return d


# ----------------------------------------------------------------------------- masks
def object_mask(alphas: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """The removal objects of a frame (alpha > 0.02, bool), None without any – what the points are carried with."""
    al = [np.asarray(a, np.float32) for a in alphas]
    return None if not al else np.max(np.stack(al), axis=0) > 0.02


def grow(mask: np.ndarray, px: float) -> np.ndarray:
    """`mask` dilated by `px` (elliptic, bool) – the patch's object part grows by `key_grow_px` (render and preview)."""
    g = int(round(float(px)))
    m = np.asarray(mask).astype(np.uint8)
    if g <= 0 or not m.any():
        return m > 0
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0


def preview_object(remove_alpha: np.ndarray, size: Tuple[int, int], scale: float, grow_px: float,
                   key_grow_px: float) -> np.ndarray:
    """The object part of the patch hole on a preview grid (`size` = (w, h), `scale` = preview px per frame px): the
    objects (alpha > 0.02) grown by max(grow_px, key_grow_px) – the blend maps' G channel and the core the preview's
    band points snap to (render.blend_inputs, api/cleanup3d.py part=band / lines)."""
    obj = cv2.resize(np.asarray(remove_alpha, np.float32), tuple(int(v) for v in size), interpolation=cv2.INTER_AREA) > 0.02
    return grow(obj, max(float(grow_px), float(key_grow_px)) * float(scale))


# ----------------------------------------------------------------------------- points
def points_fn(b: Any, core_at: Callable[[int], Optional[np.ndarray]],
              max_angle: Optional[float] = None) -> Optional[Callable[[int], List[Tuple[float, ...]]]]:
    """f -> the visible band points of frame f (x, y, out, in[, ang], plate px; widths scaled by the slider –
    objband.scales); None without points. Positions: bandtrack.track_positions without a tracker (the nearest key carried
    by the object's motion, `core_at(k)` = the removal objects of frame k)."""
    from engine.bandmap import points_at
    from engine.bandtrack import track_positions
    pts = b.band_points
    if not pts or not b.active:
        return None
    so, si = b.scales
    memo: Dict[int, List[Tuple[float, ...]]] = {}

    def at(f: int):
        f = int(f)
        if f not in memo:
            pos = track_positions(pts, [f], None, None, None, core_at=core_at).get(f, {})
            memo[f] = points_at(pts, f, pos, so, si) if max_angle is None else points_at(pts, f, pos, so, si, max_angle)
        return memo[f]
    return at


def positions(b: Any, f: int, core_at: Callable[[int], Optional[np.ndarray]]) -> Dict[str, Tuple[Any, Any, bool]]:
    """{point id: (x, y, visible)} of frame f (the editor's state – the same positions as `points_fn`)."""
    from engine.bandtrack import track_positions
    if not b.band_points:
        return {}
    return track_positions(b.band_points, [int(f)], None, None, None, core_at=core_at).get(int(f), {})


def shift_points(pts: Optional[Sequence[Tuple[float, ...]]], dx: float, dy: float, scale: float = 1.0
                 ) -> Optional[List[Tuple[float, ...]]]:
    """Points moved into a crop (x - dx, y - dy) and / or scaled (positions AND widths × scale – a preview grid)."""
    if not pts:
        return None
    return [((p[0] - dx) * scale, (p[1] - dy) * scale, p[2] * scale, p[3] * scale) + tuple(p[4:]) for p in pts]


# ----------------------------------------------------------------------------- the rule
def out_widths(core: np.ndarray, soft: float, pts: Optional[Sequence[Tuple[float, ...]]]):
    """The band's outer width: `soft` (scalar) or, with points, a per-pixel map from the outline of `core` (the object
    part of the hole) – bandsys.band_widths, outer side only."""
    o, _ = bandsys.band_widths(np.asarray(core) > 0, float(soft), 0.0, list(pts) if pts else None)
    return o


def area(hole: np.ndarray, out_w: Any) -> np.ndarray:
    """The band: outside the hole within out_w of it (bandsys.band_area with the signed distance, inset 0)."""
    h = np.asarray(hole).astype(bool)
    d = cv2.distanceTransform((~h).astype(np.uint8), cv2.DIST_L2, 5)
    return bandsys.band_area(np.where(h, 1.0, -d).astype(np.float32), out_w, 0.0)


def patch_weight(hole: np.ndarray, core: np.ndarray, soft: float, feather: Optional[float] = None,
                 pts: Optional[Sequence[Tuple[float, ...]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(patch 0..1 float32, band area) of the key patch: 1 in `hole`, smoothstep of the band weight (bandsys.area_weight,
    inset 0, outer feather `feather` = default `soft`) in the band sized by `soft` and shaped by the points (`core` = the
    object part of the hole the points snap to; all in one pixel grid)."""
    h = np.asarray(hole).astype(bool)
    fo = float(soft if feather is None else feather)
    if float(soft) <= 0 and not pts:
        return h.astype(np.float32), np.zeros(h.shape, bool)
    a = area(h, out_widths(core, soft, pts))
    w = bandsys.area_weight(a, h, 0.0, fo)
    patch = np.where(h, 1.0, np.where(a, w * w * (3.0 - 2.0 * w), 0.0)).astype(np.float32)
    return patch, a


def lines(hole: np.ndarray, core: np.ndarray, soft: float, feather: Optional[float] = None,
          pts: Optional[Sequence[Tuple[float, ...]]] = None) -> Dict[str, Any]:
    """The preview lines of the patch band (bandsys.area_lines – band edges + where the outer feather ends)."""
    h = np.asarray(hole).astype(bool)
    fo = float(soft if feather is None else feather)
    if float(soft) <= 0 and not pts:
        return bandsys.preview_lines(np.zeros(h.shape, bool), None, None)
    a = area(h, out_widths(core, soft, pts))
    return bandsys.area_lines(a, h, 0.0, fo)


def preview_hole(maps: np.ndarray, effects: bool, thr: float, diff_max: float) -> np.ndarray:
    """The hole of the live preview from the blend maps (render.blend_inputs: B difference, G object part, R full key
    coverage) – static/cleanup3d.js composeBlend's rule, for the server-side band lines: G, plus (with `effects`) the
    pixels where the keys fully cover inside the 4-connected components of (G | difference > thr) that touch G."""
    g = maps[..., 1] > 127
    if not effects:
        return g
    t = max(0.0, min(255.0, float(thr) / float(diff_max) * 255.0))
    cand = (g | (maps[..., 0] > t)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    touch = np.zeros(n, bool)
    touch[np.unique(lab[g])] = True
    touch[0] = False
    return g | (touch[lab] & (maps[..., 2] > 127))
