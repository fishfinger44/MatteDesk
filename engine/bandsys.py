# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandsys.py - one band system wherever a stage puts a model's matte over its base mask through a band
(user decision 2026-10-22, CLAUDE.md „Pas krawędzi = jeden system”; the pattern = the stage-5 whole-outline zone).

The band lies along the edge of the stage's base mask (the "core" of the stage: the mask 1+2, a stage-3 result ...):
`out` px outside the edge .. `in` px inside it. Both widths are maps when band control points shape the band
(engine/bandmap.py width_maps), scalars otherwise. Inside the band the model's matte is blended with the CORE:

  * the inset feather runs from the INNER band edge outwards over `inner_px`,
  * the outer feather runs from the OUTER band edge inwards over `outer_px`,
  * they never overlap – the inset one keeps its length (up to the band width), the outer one gets the rest,
  * weight w = min(ramp_in, ramp_out); alpha = w · model + (1 − w) · core; 0 px = a hard edge on that side;
  * outside the band the core stays.

Distances are measured from the edge of the core analytically (signed distance, > 0 inside), so a constant band and
its feathers are exact (the former VideoMaMa composition with the unified sliders is the case without points and
without an outer feather – bit-identical)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

Width = Union[float, np.ndarray]


def band_widths(core: np.ndarray, out_px: float, in_px: float, pts: Optional[Sequence[Tuple[float, float, float, float]]] = None
                ) -> Tuple[Width, Width]:
    """(out, in) of the band: scalars, or per-pixel maps when band points (x, y, out, in in the core's pixel space) shape
    it – each pixel takes the widths of its nearest edge pixel, interpolated along the edge between the points."""
    if not pts:
        return float(out_px), float(in_px)
    from engine.bandmap import width_maps
    return width_maps(np.asarray(core) > 0.5, list(pts), float(out_px), float(in_px))


def band_weight(d: np.ndarray, out_w: Width, in_w: Width, inner_px: float, outer_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """(area, weight) from the signed distance `d` to the core edge (> 0 inside): area = −out <= d <= in; the weight
    ramps over the inset feather from the inner edge and over the outer feather from the outer edge (never overlapping)."""
    d = d.astype(np.float32)
    o = np.asarray(out_w, np.float32)
    i = np.asarray(in_w, np.float32)
    area = (d >= -o) & (d <= i)
    d_in = i - d                     # px from the inner band edge
    d_out = d + o                    # px from the outer band edge
    width = o + i
    fi, fo = max(0.0, float(inner_px)), max(0.0, float(outer_px))
    li = np.minimum(fi, width)
    lo = np.minimum(fo, np.maximum(0.0, width - li))
    w_in = np.where(li > 0, np.clip(d_in / np.maximum(li, 1e-6), 0.0, 1.0), 1.0)
    w_out = np.where(lo > 0, np.clip(d_out / np.maximum(lo, 1e-6), 0.0, 1.0), 1.0)
    return area, np.where(area, np.minimum(w_in, w_out), 0.0).astype(np.float32)


def compose(core: np.ndarray, model: np.ndarray, d: np.ndarray, out_w: Width, in_w: Width, inner_px: float,
            outer_px: float) -> np.ndarray:
    """The stage alpha: inside the band w · model + (1 − w) · core, outside the core (both 0..1, same shape)."""
    c = np.clip(core.astype(np.float32), 0, 1)
    m = np.clip(model.astype(np.float32), 0, 1)
    area, w = band_weight(d, out_w, in_w, inner_px, outer_px)
    return np.where(area, w * m + (1.0 - w) * c, c).astype(np.float32)


def feather_lines(d: np.ndarray, out_w: Width, in_w: Width, inner_px: float, outer_px: float):
    """Masks whose outlines are the preview lines: (band, inset feather end, outer feather end) – the end masks are the
    solid side up to where each feather ends (None when that feather is 0)."""
    d = d.astype(np.float32)
    o = np.asarray(out_w, np.float32)
    i = np.asarray(in_w, np.float32)
    band = (d >= -o) & (d <= i)
    width = o + i
    li = np.minimum(max(0.0, float(inner_px)), width)
    lo = np.minimum(max(0.0, float(outer_px)), np.maximum(0.0, width - li))
    inner_end = (d > i) | (band & (i - d < li)) if float(inner_px) > 0 else None
    outer_end = (d > i) | (band & (d + o >= lo)) if float(outer_px) > 0 else None
    return band, inner_end, outer_end
