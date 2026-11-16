# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shutter angle of the shot measured on the plate (user decision 2026-11-08, engine/shutter.py)

    GET    /api/project/shutter          -> {deg, source: measured | manifest | default, measured: {...} | null}
    POST   /api/project/shutter/measure  -> measures on the beauty plate (heavy job), stores it in project.json `shutter`
    DELETE /api/project/shutter          -> forgets the measurement (back to the manifest / default)

The measurement lives in the tool's project.json, never in the user's manifest. Built by `make(*, ...)` like the other
api/ routers (nothing imported from app.py; functions passed as lambdas).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException


def make(*, STATE, pipeline, shot_frames, long_job):
    """Build the router. Returns (router, exports) – exports: `shutter_info`."""
    router = APIRouter()

    def _proj():
        proj = STATE.get("shot_project")
        if proj is None:
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        return proj

    def shutter_info() -> Dict[str, Any]:
        from engine.shutter import resolve
        proj = _proj()
        deg, src = resolve(proj.effective_manifest(), proj.data)
        return {"deg": deg, "source": src, "measured": proj.data.get("shutter")}

    @router.get("/api/project/shutter")
    def get_shutter():
        return shutter_info()

    @router.post("/api/project/shutter/measure")
    @long_job("Pomiar migawki z plate'u")
    async def measure_shutter():
        from engine.jobs import JOBS
        from engine.shutter import measure
        proj = _proj()
        pipe = pipeline()
        frames = list(shot_frames())
        gray = pipe._mb_gray(frames)
        r = await asyncio.get_running_loop().run_in_executor(
            None, lambda: measure(gray, frames, cancel_check=JOBS.check))
        if r.get("deg") is None:
            raise HTTPException(status_code=409, detail="Nie da się zmierzyć migawki: " + (r.get("why") or "?"))
        proj.data["shutter"] = {"deg": r["deg"], "confident": r["confident"], "pairs": r["pairs"], "why": r["why"],
                                "errors": r["errors"], "frames": [frames[0], frames[-1]],
                                "at": time.strftime("%Y-%m-%d %H:%M")}
        proj.save()
        return shutter_info()

    @router.delete("/api/project/shutter")
    def forget_shutter():
        proj = _proj()
        proj.data.pop("shutter", None)
        proj.save()
        return shutter_info()

    return router, {"shutter_info": shutter_info}
