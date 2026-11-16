# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""2D camera motion endpoints (stage E1d, SPEC 4d)

Routes of app.py's section „2D camera motion endpoints (stage E1d, SPEC 4d)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from engine.camera import Camera2DData, Camera2DTracker, NukeCamera2DGroupGenerator
from engine.ingest import ProxyGenerator
from engine.jobs import JOBS

import logging

logger = logging.getLogger("matte_desk.api.camera")


# -----------------------------------------------------------------------------
# 2D Camera Motion Endpoints (Stage E1d, SPEC Section 4d, 12, 16)
# -----------------------------------------------------------------------------

class CameraTrackRequest(BaseModel):
    ref_frame: Optional[int] = None
    eval_w: int = 1024
    anchor_interval: int = 24
    export_nk: bool = True


def make(*, STATE, _plate_size, long_job, ws_manager):
    """Build the router. Keyword-only: exactly what the section takes from app.py (4 names).
    Returns (router, exports) – exports: none."""
    router = APIRouter()



    @router.post("/api/camera/track")
    @long_job("Kamera 2D")
    async def track_camera_motion(req: CameraTrackRequest):
        # the whole tracking off the event loop (frame decoding, reports): the loop stays for /api/jobs, „Przerwij”, the
        # WebSocket; async only for the loop on_progress broadcasts through
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _track_camera_motion(req, loop))


    def _track_camera_motion(req: CameraTrackRequest, loop: asyncio.AbstractEventLoop):
        if not STATE.get("manifest") or not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        manifest = STATE["manifest"]
        shot_name = manifest.get("shot", "shot")
        fps = float(manifest.get("fps", 24.0))
        first_f = int(manifest.get("first", 1))
        last_f = int(manifest.get("last", 1))
        ref_frame = req.ref_frame if req.ref_frame is not None else first_f

        work_dir = Path(STATE["work_dir"])
        cam_dir = work_dir / "camera"
        cam_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gather frames: check proxy frames first for lightning fast processing, fallback to source
        proxy_dir = work_dir / "proxy" / "beauty"
        frames: List[np.ndarray] = []
        frame_nums: List[int] = []

        if proxy_dir.exists() and len(list(proxy_dir.glob("p_*.jpg"))) >= (last_f - first_f + 1):
            for fn in range(first_f, last_f + 1):
                JOBS.check()   # „Przerwij” between frames
                pf = proxy_dir / f"p_{fn:04d}.jpg"
                if pf.exists():
                    img = cv2.imread(str(pf))
                    if img is not None:
                        frames.append(img)
                        frame_nums.append(fn)
        else:
            # Load from plate pattern
            plates = manifest.get("plates", {})
            pat = plates.get("beauty")
            if not pat:
                raise HTTPException(status_code=400, detail="Brak plate'u beauty w manifeście")
            for fn in range(first_f, last_f + 1):
                JOBS.check()   # „Przerwij” between frames
                src_file = ProxyGenerator.resolve_frame_path(pat, fn)
                if src_file.exists():
                    img = cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        frames.append(img)
                        frame_nums.append(fn)

        if not frames:
            raise HTTPException(status_code=404, detail="Nie znaleziono klatek do analizy ruchu kamery")

        def on_progress(step_name: str, done: int, total: int):
            JOBS.progress(step_name, done, total)
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "camera_progress",
                    "step": step_name,
                    "done": done,
                    "total": total
                }),
                loop
            )

        cam_data: Camera2DData = Camera2DTracker.track_sequence(
            frames=frames,
            frame_numbers=frame_nums,
            ref_frame=ref_frame,
            eval_w=req.eval_w,
            anchor_interval=req.anchor_interval,
            shot_name=shot_name,
            fps=fps,
            progress_cb=on_progress,
            plate_w=_plate_size()[0],
            plate_h=_plate_size()[1],
        )

        json_path = cam_dir / "camera2d.json"
        cam_data.save(json_path)

        nk_path_str = ""
        if req.export_nk:
            nk_file = cam_dir / f"Camera2D_{shot_name}.nk"
            NukeCamera2DGroupGenerator.generate_script(
                shot_name=shot_name,
                camera_data=cam_data,
                output_nk_path=nk_file,
                output_json_path=json_path
            )
            nk_path_str = str(nk_file).replace("\\", "/")

        from engine.outputs import update_outputs
        update_outputs(work_dir, "camera2d", {
            "json_path": str(json_path).replace("\\", "/"),
            "nk_path": nk_path_str,
            "motion_type": cam_data.motion_type,
            "residual_median_px": cam_data.residual_median_px,
            "anchor_drift_px": cam_data.anchor_drift_px
        })

        return {
            "success": True,
            "camera2d": cam_data.to_dict(),
            "nk_script_path": nk_path_str,
            "json_path": str(json_path).replace("\\", "/")
        }


    @router.get("/api/camera/report")
    def get_camera_report():
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "camera" / "camera2d.json"
        if not json_path.exists():
            return {"exists": False}

        try:
            cam_data = Camera2DData.load(json_path)
            nk_file = work_dir / "camera" / f"Camera2D_{cam_data.shot_name}.nk"
            return {
                "exists": True,
                "camera2d": cam_data.to_dict(),
                "nk_exists": nk_file.exists(),
                "nk_path": str(nk_file).replace("\\", "/") if nk_file.exists() else ""
            }
        except Exception as e:
            logger.error(f"Error loading camera2d.json: {e}")
            return {"exists": False, "error": str(e)}


    @router.post("/api/camera/export_nk")
    def export_camera_nuke_group():
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "camera" / "camera2d.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Brak pliku camera2d.json. Najpierw przeprowadź analizę ruchu.")

        cam_data = Camera2DData.load(json_path)
        shot_name = cam_data.shot_name
        nk_file = work_dir / "camera" / f"Camera2D_{shot_name}.nk"

        nk_content = NukeCamera2DGroupGenerator.generate_script(
            shot_name=shot_name,
            camera_data=cam_data,
            output_nk_path=nk_file,
            output_json_path=json_path
        )

        return {
            "success": True,
            "nk_path": str(nk_file).replace("\\", "/"),
            "script": nk_content
        }

    return router, {}
