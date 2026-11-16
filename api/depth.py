# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth map endpoints (stage E1e, SPEC 4e)

Routes of app.py's section „Depth map endpoints (stage E1e, SPEC 4e)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi import HTTPException
import numpy as np
from pydantic import BaseModel

from comfy_client import ComfyClient
from engine.camera import Camera2DData
from engine.config import load_config
from engine.jobs import JOBS
from engine.project import ShotProject


# -----------------------------------------------------------------------------
# Depth map Endpoints (Stage E1e, SPEC Section 4e, 10)
# -----------------------------------------------------------------------------

class DepthComputeRequest(BaseModel):
    plate: str = "beauty"
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    workflow_path: Optional[str] = None


def make(*, STATE, long_job, ws_manager):
    """Build the router. Keyword-only: exactly what the section takes from app.py (3 names).
    Returns (router, exports) – exports: none."""
    router = APIRouter()



    def _depth_workflow_path(explicit: Optional[str]) -> Path:
        """The user's DepthCrafter workflow (Comfy UI JSON). Config `depth_workflow`, else Comfy's workflows dir."""
        if explicit:
            return Path(explicit)
        cfg = load_config()
        if cfg.get("depth_workflow"):
            return Path(cfg["depth_workflow"])
        return Path(cfg.get("comfy_dir", "")) / "user" / "default" / "workflows" / "workflow_comfyui_depthcrafter.json"


    @router.post("/api/depth/compute")
    @long_job("Głębia (DepthCrafter)")
    async def compute_depth(req: DepthComputeRequest):
        if not STATE.get("shot_project"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        from engine.depth import run_depth

        proj: ShotProject = STATE["shot_project"]
        manifest = STATE["manifest"]
        plates = manifest.get("plates", {})
        work_dir = Path(STATE["work_dir"])
        wf = _depth_workflow_path(req.workflow_path)
        if not wf.exists():
            raise HTTPException(status_code=400, detail=f"Nie znaleziono workflowu DepthCrafter: {wf}")

        # Input after deflicker (4e.1 pkt 1): the `_deflicker` EXR if exported, otherwise plate x gain maps
        plate_pat = plates.get(f"{req.plate}_clean") or plates.get(req.plate)
        if not plate_pat:
            raise HTTPException(status_code=400, detail=f"Brak plate'u '{req.plate}'")
        gain_dir = work_dir / "deflicker" / f"gain_{req.plate}"
        gain_pat = str(gain_dir / "gain_%04d.exr") if not plates.get(f"{req.plate}_clean") and gain_dir.exists() else None

        first_f = req.first_frame or int(manifest.get("first", 1))
        last_f = req.last_frame or int(manifest.get("last", 1))
        frames = list(range(first_f, last_f + 1))
        w, h = proj.plate_size

        homs = None
        cam_json = work_dir / "camera" / "camera2d.json"
        if cam_json.exists():
            cam = Camera2DData.load(cam_json)
            by_f = {fr.frame: np.array(fr.homography, dtype=np.float64) for fr in cam.frames}
            if all(f in by_f for f in frames):
                sx, sy = w / float(cam.plate_w), h / float(cam.plate_h)
                S = np.diag([sx, sy, 1.0])
                homs = [S @ by_f[f] @ np.linalg.inv(S) for f in frames]

        client = STATE.get("comfy_client") or ComfyClient.from_config()
        loop = asyncio.get_running_loop()

        def on_progress(step: str, done: int, total: int):
            JOBS.progress(step, done, total)
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({"type": "depth_progress", "step": step, "done": done, "total": total}), loop)

        try:
            rep = await loop.run_in_executor(None, lambda: run_depth(
                client, wf, manifest.get("shot", "shot"), plate_pat, frames, w, h, work_dir, work_dir / "export",
                gain_pattern=gain_pat, homographies=homs, progress_cb=on_progress))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"success": True, "report": rep.to_dict()}


    @router.get("/api/depth/report")
    def get_depth_report():
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        p = Path(STATE["work_dir"]) / "depth" / "depth_report.json"
        if not p.exists():
            return {"exists": False}
        return {"exists": True, "report": json.loads(p.read_text(encoding="utf-8"))}


    @router.post("/api/depth/export_nk")
    def export_depth_nk():
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        from engine.depth import DepthReport, NukeDepthGroupGenerator

        p = Path(STATE["work_dir"]) / "depth" / "depth_report.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="Najpierw policz głębię")
        rep = DepthReport(**json.loads(p.read_text(encoding="utf-8")))
        nk = Path(STATE["work_dir"]) / "depth" / f"Depth_{rep.shot}.nk"
        text = NukeDepthGroupGenerator.generate_script(rep.shot, rep, nk)
        return {"success": True, "nk_path": str(nk).replace("\\", "/"), "script": text}

    return router, {}
