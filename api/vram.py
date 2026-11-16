# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU memory forecast before a stage starts (user request 2026-10-26; engine/vram_forecast.py)

    GET /api/object/{name}/vram_forecast?stage=videomama[&frame=F]
        the biggest model pass of the run (work area) or of the frame preview (frame given) against the Comfy card:
        {state: ok | over | unknown, peak_gb, card_gb, budget_gb, side, frames, suggest_side, extrapolated, why, text}

Built by `make(*, ...)` like the other api/ routers (nothing imported from app.py; functions passed as lambdas).
The UI (static/vramcheck.js) asks before „▶ Policz VideoMaMa” / „👁 Podgląd klatek” and shows a window when the
forecast is over the card; „unknown” never stops anything.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

STAGES = ("videomama",)


def forecast_text(r: Dict[str, Any], label: str) -> List[str]:
    """Polish sentences for the window, one per line (each translated on its own: static/i18n/en.json patterns)."""
    if r.get("state") != "over":
        return []
    t = [f"{label}: prognoza pamięci GPU {r['peak_gb']:.1f} GB przy kadrze {r['side']} px × {r['frames']} kl. – "
         f"karta ma {r['card_gb']:.1f} GB (wolne na model ok. {r['budget_gb']:.1f} GB).",
         "Nadmiar przeleje się do RAM: liczenie kilka razy wolniejsze, może skończyć się brakiem pamięci po wielu minutach."]
    if r.get("suggest_side"):
        t.append(f"Zmieści się „Rozdzielczość” ≤ {r['suggest_side']} px.")
    if r.get("extrapolated"):
        t.append("Prognoza poza zakresem pomiarów – przybliżona.")
    return t


def make(*, objects, pipeline, work_frames, shot_frames, comfy_client):
    """Build the router. Keyword-only, functions (late binding: tests swap the app's names). Returns (router, exports)."""
    router = APIRouter()

    def _others_gb(stats: Optional[Dict[str, Any]]) -> float:
        """GPU memory other programs hold now (nvidia-smi used minus what Comfy's torch reserved – the stage frees
        Comfy's models before it starts, engine/gpu_manager.before_stage)."""
        from engine.sysinfo import gpu_info
        g = gpu_info() or []
        if not g or g[0].get("mem_used_mb") is None:
            return 0.0
        used = float(g[0]["mem_used_mb"]) / 1024
        try:
            comfy = float(((stats or {}).get("devices") or [{}])[0].get("torch_vram_total") or 0) / 2 ** 30
        except (TypeError, ValueError, IndexError):
            comfy = 0.0
        return max(0.0, used - comfy - 0.5)   # 0.5 GB: Comfy's own CUDA context

    @router.get("/api/object/{name}/vram_forecast")
    def vram_forecast(name: str, stage: str = "videomama", frame: Optional[int] = None):
        """Forecast of the stage's peak GPU memory against the Comfy card (engine/vram_forecast.py)."""
        from engine import vram_forecast as vf
        from engine.refine import contiguous_chunks
        if stage not in STAGES:
            raise HTTPException(status_code=400, detail="stage: " + " | ".join(STAGES))
        from engine.videomama import object_settings, offload_on
        store = objects()
        try:
            obj = store.get(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"brak obiektu {name}")
        s = object_settings(obj)
        pipe = pipeline()
        try:
            if frame is not None:
                from engine import stage_preview as sp
                frames = sp.window(pipe._visible(name, shot_frames()), int(frame), sp.WINDOW[stage])
            else:
                frames = pipe._visible(name, work_frames())
            segs = pipe._work_segments(name, frames) if frames else []
        except (FileNotFoundError, ValueError) as e:
            return {"state": "unknown", "why": str(e), "text": []}
        sides, lens = [], []
        for seg in segs:
            for chunk in contiguous_chunks([f for f in frames if seg.first <= f <= seg.last]):
                sides.append(seg.side)
                lens.append(len(chunk))
        try:
            stats = comfy_client().get_system_stats()
        except Exception:
            stats = None
        variant = "offload" if offload_on(s, stats) else "default"
        r = vf.check("videomama", vf.videomama_passes(sides, lens, s), vf.card_gb(stats), variant,
                     others_gb=_others_gb(stats))
        r["max_side"] = s["max_side"]
        r["text"] = forecast_text(r, "VideoMaMa")
        return r

    return router, {}
