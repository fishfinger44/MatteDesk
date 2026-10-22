# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_preview.py – „👁 Podgląd klatek” of the deflicker (docs/PLAN_DEFLICKER_AUTO.md § 5).

Runs the deflicker with the palette's settings on a short window of frames downscaled to `eval_w` px (1024: the scale the
gain maps are computed at anyway – only the full-resolution guided upsampling and the export are skipped, so the result is
not bit for bit the full run, but the flicker numbers and the heat map are representative; the user decided 1024 px is
enough to see whether it worked). Before / after are measured with `engine.flicker_measure` on the ORIGINAL's flows, so the
two numbers sit on identical alignment; "reduction" is given relative to what is removable at all (the flow floor).
Images: heat maps (inferno, common scale) and display frames (tone-mapped like the proxies) for the before / after loop.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from engine.clean import PlateCleaner, as_bgr
from engine.flicker_measure import FlickerMeasure, MaskSpec, measure_flicker

try:   # the proxies' display transform (ACES -> sRGB-ish); a plain curve when unavailable
    from engine.ingest import tonemap_plate_to_display as _tonemap
except Exception:   # pragma: no cover
    _tonemap = None


@dataclass
class PreviewResult:
    n_frames: int
    eval_w: int
    eval_h: int
    stages_applied: List[str]
    report_reduction_pct: float
    regions: Dict[str, Dict[str, float]]          # name -> before / after / floor / removed_pct (of the removable part)
    before: Dict[str, Any]                        # FlickerMeasure.to_dict()
    after: Dict[str, Any]
    seconds: float
    notes: List[str] = field(default_factory=list)
    vmax_pct: float = 0.0                         # heat-map scale (98th percentile of the before map)
    # images – not in to_dict
    map_before: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    map_after: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    frames_before: Optional[List[np.ndarray]] = field(default=None, repr=False, compare=False)
    frames_after: Optional[List[np.ndarray]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("map_before", "map_after", "frames_before", "frames_after"):
            d.pop(k, None)
        return d


def heat_map(map_pct: np.ndarray, vmax: float) -> np.ndarray:
    """Flicker map (%) -> uint8 BGR inferno on a 0..vmax scale (one scale for before / after)."""
    u8 = np.clip(map_pct / max(vmax, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.GaussianBlur(u8, (0, 0), 2.0), cv2.COLORMAP_INFERNO)


def display_frame(linear_bgr: np.ndarray) -> np.ndarray:
    """Scene-linear BGR float -> uint8 BGR for the browser (same look as the proxies when the tonemap is available)."""
    x = np.maximum(np.nan_to_num(as_bgr(np.asarray(linear_bgr, np.float32)), nan=0.0), 0.0)
    if _tonemap is not None:
        try:
            out = _tonemap(x)
            if out is not None and out.dtype == np.uint8:
                return out
        except Exception:
            pass
    return np.clip((x / (x + 1.0)) ** (1 / 1.8) * 255.0 * 1.5, 0, 255).astype(np.uint8)


def preview_deflicker(frames: Sequence[np.ndarray], settings_kwargs: Dict[str, Any], masks: Optional[Dict[str, MaskSpec]] = None,
                      eval_w: int = 1024, progress_cb: Optional[Callable[[str, int, int], None]] = None,
                      with_images: bool = True) -> PreviewResult:
    """`frames`: the window (any size, scene-linear BGR); `settings_kwargs`: DeflickerSettings.to_process_kwargs(...)."""
    t0 = time.time()
    n = len(frames)
    if n < 3:
        raise ValueError("podgląd deflickera potrzebuje co najmniej 3 klatek")
    f0 = as_bgr(np.asarray(frames[0], np.float32))
    H0, W0 = f0.shape[:2]
    ew = min(int(eval_w), W0)
    eh = max(8, round(H0 * ew / W0))

    def small(a: np.ndarray) -> np.ndarray:
        a = as_bgr(np.asarray(a, np.float32))
        a = cv2.resize(a, (ew, eh), interpolation=cv2.INTER_AREA) if a.shape[:2] != (eh, ew) else a.copy()
        return np.nan_to_num(a, nan=0.0)

    if progress_cb:
        progress_cb("Podgląd deflickera: klatki", 0, 100)
    src = [small(frames[i]) for i in range(n)]
    kw = dict(settings_kwargs)
    kw.setdefault("target_reduction", 2.0)
    if progress_cb:
        progress_cb("Podgląd deflickera: liczenie", 15, 100)
    cleaned, _, rep, _ = PlateCleaner.process_sequence([f.copy() for f in src], **kw)
    cleaned = [np.asarray(f, np.float32) for f in cleaned]
    if progress_cb:
        progress_cb("Podgląd deflickera: pomiar przed / po", 70, 100)
    before = measure_flicker(src, masks, eval_w=ew, return_maps=True)
    after = measure_flicker(cleaned, masks, eval_w=ew, flow_from=src, return_maps=True)
    regions: Dict[str, Dict[str, float]] = {}
    for nm, rb in before.regions.items():
        ra = after.regions.get(nm)
        if ra is None:
            continue
        removable = max(rb.metric_pct - rb.floor_pct, 1e-6)
        removed = (rb.metric_pct - ra.metric_pct) / removable * 100.0
        regions[nm] = {"before": rb.metric_pct, "after": ra.metric_pct, "floor": rb.floor_pct,
                       "removed_pct": round(float(np.clip(removed, -100.0, 100.0)), 1)}
    notes: List[str] = []
    win = int(kw.get("stage1_window", 20))
    if win > max(3, n // 2):
        notes.append(f"Okno czasowe ograniczone do ±{max(3, n // 2)} kl. przez długość podglądu ({n} kl.) – pełne liczenie użyje ±{win}")
    if kw.get("enable_stage3"):
        notes.append("Podgląd liczy „fakturę i cienie” w 1024 px – pełny wynik tylko przy zapisie EXR")
    res = PreviewResult(n_frames=n, eval_w=ew, eval_h=eh, stages_applied=list(rep.stages_applied),
                        report_reduction_pct=float(rep.reduction_pct), regions=regions,
                        before=before.to_dict(), after=after.to_dict(), seconds=round(time.time() - t0, 1), notes=notes)
    if with_images and before.maps is not None and after.maps is not None:
        mb, ma = before.maps["metric"], after.maps["metric"]
        res.vmax_pct = float(np.percentile(mb, 98))
        res.map_before, res.map_after = heat_map(mb, res.vmax_pct), heat_map(ma, res.vmax_pct)
        res.frames_before = [display_frame(f) for f in src]
        res.frames_after = [display_frame(f) for f in cleaned]
    if progress_cb:
        progress_cb("Podgląd deflickera: gotowe", 100, 100)
    return res
