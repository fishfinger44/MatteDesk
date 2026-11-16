# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU memory forecast BEFORE a Comfy stage starts (user request 2026-10-26, NEXT_STEPS „prognoza VRAM”).

Why: VideoMaMa at 2048 px on a 12 GB card first spilled ~6 GB of GPU memory into shared RAM (Windows WDDM) and failed
with an out-of-memory error only after ~9 minutes (sh060 2026-10-26). The spill warning DURING the run exists
(comfy_client.SpillSampler); this module says it before the start.

Model: the peak GPU memory of one model pass grows linearly with the megapixels of ONE work frame
    peak_gb ≈ a + b · x,   x = work width · work height / 1e6
(measured 2026-11-06 on the RTX 4090: VideoMaMa 1280 px = 23.3 GB with 24 frames, 23.1 GB with 12; with offload 19.4 /
19.2 GB – the frame count of the batch barely matters, the frame size does) fitted per model and variant (VideoMaMa: "default" / "offload" = the node's CPU offload) on the points measured with
`tools/measure_vram.py` (peak of the whole card by nvidia-smi, Comfy freed before every pass) and stored in
`engine/vram_table.json`. The peak is of the whole card, so it includes what the desktop holds at the time of measuring
(`base_gb` of the point) – the forecast removes it and adds what this machine holds now when it knows it.
Not forecast: MatAnyone2 (its own auto-scale probe measures the spill on the real crop, engine/refine.py) and LTX-2.5
(Comfy loads its 22B weights dynamically – the peak follows the card, not the input).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TABLE_PATH = Path(__file__).with_name("vram_table.json")
MARGIN_GB = 0.8          # kept free on the card: CUDA context of other programs, fragmentation
MIN_POINTS = 2           # a line needs two distinct x


# ----------------------------------------------------------------------------- table
def load_table(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else TABLE_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"models": {}}


def points(model: str, variant: str = "default", table: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    t = table if table is not None else load_table()
    return list(((t.get("models") or {}).get(model) or {}).get(variant) or [])


def mpx_frames(w: int, h: int, frames: int) -> float:
    """Megapixel-frames of a pass (kept in the table for the record; the fit uses `point_x`)."""
    return float(w) * float(h) * float(frames) / 1e6


def mpx(w: int, h: int) -> float:
    return float(w) * float(h) / 1e6


def point_x(p: Dict[str, Any]) -> float:
    """x of a measured point = megapixels of one work frame (side × height or side²)."""
    return mpx(p["side"], p.get("height") or p["side"])


def fit(pts: Sequence[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """(a, b) of peak_gb - base_gb ≈ a + b·x by least squares; b >= 0; None with fewer than MIN_POINTS distinct x."""
    xs = [point_x(p) for p in pts]
    ys = [float(p["peak_gb"]) - float(p.get("base_gb") or 0.0) for p in pts]
    if len(set(round(x, 6) for x in xs)) < MIN_POINTS:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = max(0.0, sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx)
    return my - b * mx, b


def predict(model: str, x: float, variant: str = "default", table: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """The model's own peak (GB, without the desktop's share) for a work frame of `x` megapixels; None = not measured."""
    ab = fit(points(model, variant, table))
    if ab is None:
        return None
    return ab[0] + ab[1] * x


def measured_range(model: str, variant: str = "default", table: Optional[Dict[str, Any]] = None) -> Optional[Tuple[float, float]]:
    xs = [point_x(p) for p in points(model, variant, table)]
    return (min(xs), max(xs)) if xs else None


# ----------------------------------------------------------------------------- VideoMaMa
def videomama_passes(sides: Iterable[int], chunk_lengths: Iterable[int], s: Dict[str, Any]) -> List[Tuple[int, int]]:
    """(work side, frames) of the biggest pass per crop segment: the node gets side × side (engine/videomama.work_side)
    and at most `batch` frames at once. `sides` / `chunk_lengths` pair up per segment chunk."""
    from engine.videomama import work_side
    out = []
    for side, n in zip(sides, chunk_lengths):
        if n <= 0:
            continue
        out.append((work_side(int(side), int(side), int(s["max_side"])), int(min(n, s["batch"]))))
    return out


def side_for(model: str, frames: int, budget_gb: float, variant: str = "default", table: Optional[Dict[str, Any]] = None,
             step: int = 64, lo: int = 256, hi: int = 2048) -> Optional[int]:
    """The largest square work side (multiple of `step`, lo..hi) whose forecast fits `budget_gb`; None = not measured
    or not even `lo` fits. (`frames` kept for the callers: the peak does not depend on it.)"""
    ab = fit(points(model, variant, table))
    if ab is None:
        return None
    a, b = ab
    if b <= 0:
        return hi if a <= budget_gb else None
    x = (budget_gb - a) / b
    if x <= 0:
        return None
    side = int(math.sqrt(x * 1e6)) // step * step
    side = min(hi, side)
    return side if side >= lo else None


# ----------------------------------------------------------------------------- the check
def card_gb(system_stats: Optional[Dict[str, Any]] = None, gpus: Optional[List[Dict[str, Any]]] = None) -> Optional[float]:
    """Total memory of the Comfy GPU (Comfy /system_stats), else of the first card nvidia-smi lists."""
    try:
        v = float(((system_stats or {}).get("devices") or [{}])[0].get("vram_total") or 0)
        if v > 0:
            return v / 2 ** 30
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    for g in gpus or []:
        mb = g.get("mem_total_mb") or g.get("memory_total_mb")
        if mb:
            return float(mb) / 1024
    return None


def check(model: str, passes: Sequence[Tuple[int, int]], card: Optional[float], variant: str = "default",
          others_gb: float = 0.0, table: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Forecast of the worst pass against the card. Returns {state: ok | over | unknown, peak_gb, card_gb, budget_gb,
    side, frames, suggest_side, extrapolated, why}. `others_gb` = memory other programs hold now (not Comfy)."""
    out: Dict[str, Any] = {"model": model, "variant": variant, "card_gb": None if card is None else round(card, 1)}
    if not passes:
        return {**out, "state": "unknown", "why": "brak kadrów do policzenia"}
    side, frames = max(passes, key=lambda p: (p[0], p[1]))
    x = mpx(side, side)
    peak = predict(model, x, variant, table)
    out.update(side=side, frames=frames, x=round(x, 2))
    if peak is None:
        return {**out, "state": "unknown", "why": f"brak pomiarów pamięci dla {model} ({variant})"}
    rng = measured_range(model, variant, table)
    out.update(peak_gb=round(peak, 1), extrapolated=bool(rng and (x > rng[1] * 1.15 or x < rng[0] * 0.85)))
    if card is None:
        return {**out, "state": "unknown", "why": "nie znam pamięci karty (ComfyUI nie odpowiada)"}
    budget = card - MARGIN_GB - max(0.0, others_gb)
    out["budget_gb"] = round(budget, 1)
    if peak <= budget:
        return {**out, "state": "ok"}
    out["suggest_side"] = side_for(model, frames, budget, variant, table)
    return {**out, "state": "over"}
