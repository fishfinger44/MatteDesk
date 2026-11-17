# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/inpaint_band.py - the band of Cleanup step C1 „Inpaint” (user decision 2026-11-11: the cleanup mask is the
app's ONE band system with its own settings). Glue only – the settings model is engine/objband.inpaint_band, the
geometry engine/bandsys (band_widths, preview_lines), the points engine/bandmap / engine/bandtrack / engine/bandedit,
the editor static/bandpoints.js (owner „c1”), the mask itself engine/cleanup.removal_mask.

The C1 patch of a frame:
  base  = the removed objects (alpha > 0.02)
  band  = outside the base within out(x) of it; out = „Pas na zewnątrz” (`grow_px`), shaped along the outline by the
          band points (yellow whisker); the patch is 1 there – it must cover the whole object, so there is NO inset
  edge  = „Wtopienie zewnętrzne” (`soft_px`): the patch falls to 0 over that many px OUTSIDE the band
Without points the mask is bit for bit the one before the band (elliptic dilation by grow_px + the outward ramp).

Positions between keys: as the Cleanup 3D patch band (engine/cleanup3d/band.py) – the nearest key carried by the
removed objects' motion (bandtrack.track_positions, CPU), snapped to the outline. Storage: the cleaned thing's own
fields in <work>/inpaint/cleanup.json (`band_points`, `band_ref_out`, `band_ref_in`; engine/cleanup.PER_TARGET)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine import bandsys

Alphas = Callable[[int], Sequence[np.ndarray]]
# the C1 whisker turns all round (user decision 2026-11-15): pointed into the object it cuts the patch in there
# (engine/bandsys.tip_offsets – the patch edge goes through the whisker's tip); the stages keep ±80 (bandmap)
FULL_TURN_DEG = 180.0


def band(s: Optional[Dict[str, Any]]):
    """The C1 band settings (engine/objband.inpaint_band) of the cleanup settings `s`."""
    from engine.objband import inpaint_band
    return inpaint_band(s)


def core_at(remove_alphas: Alphas) -> Callable[[int], Optional[np.ndarray]]:
    """k -> the removed objects of frame k (bool), what the points are carried with and snap to."""
    from engine.cleanup3d.band import object_mask
    return lambda k: object_mask(list(remove_alphas(int(k))))


def points_fn(s: Optional[Dict[str, Any]], remove_alphas: Alphas) -> Optional[Callable[[int], List[Tuple[float, ...]]]]:
    """f -> the visible C1 band points of frame f (x, y, out, in[, ang], plate px; widths scaled by „Pas na zewnątrz”
    over its value when they were set); None without points."""
    from engine.cleanup3d.band import points_fn as _pf
    b = band(s)
    if not b.band_points:
        return None
    keyed = bool((s or {}).get("band_keys"))
    if not keyed and b.out_px >= 0:
        return _pf(b, core_at(remove_alphas), FULL_TURN_DEG)
    if not keyed:                                      # a negative band: the raw whiskers (local points, 2026-11-15)
        return _pf(band({**s, "band_ref_out": None}), core_at(remove_alphas), FULL_TURN_DEG)
    if not b.band_ref_out:
        raw = _pf(b if b.out_px >= 0 else band({**s, "band_ref_out": None}), core_at(remove_alphas), FULL_TURN_DEG)
        return raw
    # slider keys (2026-11-15): the points' widths scaled by "Pas na zewnatrz" OF THEIR FRAME over the reference
    from engine import cleanup as cu
    ref = float(b.band_ref_out)
    raw = _pf(band({**s, "grow_px": ref}), core_at(remove_alphas), FULL_TURN_DEG)     # scale 1

    def at(f: int):
        g = cu.band_at(s, int(f))[0]
        k = g / ref if g >= 0 else 1.0                 # a negative band in that frame: the raw whiskers
        return [(p[0], p[1], p[2] * k, p[3] * k) + tuple(p[4:]) for p in raw(int(f))]
    return at


def state(s: Dict[str, Any], frame: int, remove_alphas: Alphas, b=None) -> Dict[str, Any]:
    """The editor's state of `frame` (GET /api/cleanup/band/{f}): the points as shown (position, whiskers, keys)."""
    from engine.bandedit import state_points
    from engine.cleanup3d.band import positions
    b = b if b is not None else band(s)
    defaults = {"out": max(0, int(round(b.out_px))), "in": 0}
    if not b.band_points:
        return {"frame": int(frame), "points": [], "defaults": defaults}
    ca = core_at(remove_alphas)
    core = ca(int(frame))
    so, si = b.scales
    if b.out_px < 0:                                   # a negative band: the whiskers as set (local points, 2026-11-15)
        so = 1.0
    pos = positions(b, int(frame), ca)
    return {"frame": int(frame), "points": state_points(b.band_points, int(frame), pos, core, so, si, FULL_TURN_DEG),
            "defaults": defaults, "tracking": False, "scale": {"out": round(so, 4), "in": round(si, 4)}}


def lines(alphas: Sequence[np.ndarray], grow_px: float, soft_px: float,
          pts: Optional[Sequence[Tuple[float, ...]]] = None, scale: float = 1.0,
          limit_px: Optional[float] = None, limit_alphas: Optional[Sequence[np.ndarray]] = None) -> Dict[str, Any]:
    """The preview lines of the C1 band of one frame (plate px): `band_contours` = the band ring (the objects' outline and
    the band's outer edge, where the solid patch ends), `feather_contours` = where the soft edge ends (the patch reaches
    0) – the same geometry as engine/cleanup.removal_mask, measured on a grid `scale` × the plate (lines scaled back).
    `limit_px` (user decision 2026-11-13) = generation margin + the engine's grow: `limit_contours` = the edge of what the
    model draws (the objects grown by it, before the holdout) and `cut` = the band reaches beyond it (cut there)."""
    from engine import cleanup as cu
    if not alphas:
        return bandsys.preview_lines(np.zeros((1, 1), bool), None, None)
    al = [np.asarray(a, np.float32) for a in alphas]
    if scale != 1.0:
        h, w = al[0].shape[:2]
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        al = [cv2.resize(a, size, interpolation=cv2.INTER_AREA) for a in al]
        pts = [((p[0]) * scale, (p[1]) * scale, p[2] * scale, p[3] * scale) + tuple(p[4:]) for p in pts] if pts else None
    hard = np.max(np.stack(al), axis=0) > 0.02
    if grow_px < 0:   # erode (2026-11-15): removal_mask's own geometry (local band points too)
        solid = cu.removal_mask(al, grow_px * scale, 0.0, pts) > 0.5
        full = cu.removal_mask(al, grow_px * scale, soft_px * scale, pts) > 0.0 if soft_px > 0 else None
        L = bandsys.preview_lines((hard & ~solid) | (solid & ~hard), None, full)
        L["band_contours"] = bandsys.contours(solid, cv2.RETR_CCOMP)
    elif pts:   # band points: removal_mask's per-pixel widths (already a distance transform)
        solid = cu.removal_mask(al, grow_px * scale, 0.0, pts) > 0.5
        full = cu.removal_mask(al, grow_px * scale, soft_px * scale, pts) > 0.0 if soft_px > 0 else None
    else:     # the same band from ONE distance transform – constant time; removal_mask's elliptic dilate grows with the
        # band² (user 2026-11-11: „pasy odświeżają się wolno” – 1.1 s at 120 px); the lines differ by < 1 px
        g, sf = float(grow_px) * scale, float(soft_px) * scale
        d = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if hard.any() else None
        solid = hard if d is None or g < 0.5 else (hard | (d <= round(g) + 0.5))
        full = None if d is None or soft_px <= 0 else (d < round(g) + sf)
    if grow_px >= 0:
        L = bandsys.preview_lines(solid & ~hard, None, full)
    # the band's outer edge only (user 2026-11-11: the object's own outline, the ring's inner side, looked like a second
    # band line – C1 has no inset); no band (Pas 0, no points): the outline itself
    if grow_px >= 0:
        L["band_contours"] = bandsys.contours(solid if solid.any() and not (solid == hard).all() else hard, cv2.RETR_CCOMP)
    cut = None
    if limit_px is not None and hard.any():
        lh = hard   # `limit_alphas` = the masks the engine combines for this frame (cleanup.model_frames), else this one
        if limit_alphas:
            la = [np.asarray(a, np.float32) for a in limit_alphas]
            if scale != 1.0:
                la = [cv2.resize(a, (hard.shape[1], hard.shape[0]), interpolation=cv2.INTER_AREA) for a in la]
            lh = hard | (np.max(np.stack(la), axis=0) > 0.02)
        dl = cv2.distanceTransform((~lh).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        lim = dl <= float(limit_px) * scale + 0.5
        L["limit_contours"] = bandsys.contours(lim, cv2.RETR_CCOMP)
        if lh is not hard:   # `cut` = the composite's own limit: this frame's area (cleanup.comp_mask_fn), not the token's
            dl = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            lim = dl <= float(limit_px) * scale + 0.5
        cut = bool(((full if full is not None else solid) & ~lim).any())
    k = 1.0 / max(scale, 1e-9)
    out: Dict[str, Any] = {key: [[[round(x * k, 1), round(y * k, 1)] for x, y in c] for c in v] for key, v in L.items()}
    if cut is not None:
        out["cut"] = cut
    return out
