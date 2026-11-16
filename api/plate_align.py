# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate alignment endpoints (stage E1b, SPEC 4a)

Routes of app.py's section „Plate alignment endpoints (stage E1b, SPEC 4a)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.jobs import JOBS


# -----------------------------------------------------------------------------
# Plate Alignment Endpoints (Stage E1b, SPEC Section 4a, 10, 12)
# -----------------------------------------------------------------------------

class AlignMeasureRequest(BaseModel):
    plate: str
    aux_plate_type: str = "backlit"
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    device: Optional[str] = None   # auto | gpu | cpu; None = config `align_device` (UI: Auto / GPU / CPU)
    # align_v2 step 1 (2026-11-03): the static area minus the objects' masks 1+2 (None = every object with a core,
    # [] = none), grown by exclude_dilate_px, minus a box [x0, y0, x1, y1] in plate px
    exclude_objects: Optional[List[str]] = None
    exclude_dilate_px: int = 32
    exclude_box: Optional[List[int]] = None
    camera_lock: str = "auto"   # step 2: auto (detect from the frames' global models) | on | off
    global_only: bool = False  # user 2026-11-10: "tylko transform" - Transform / CornerPin only, no local warp
    fullres_check: bool = False  # step 6a (option): full-res residual at the anchors decides the local model + in the report


class AlignExportNkRequest(BaseModel):
    plate: str


class AlignApplyRequest(BaseModel):
    plate: str
    confirmed: bool = False


def make(*, STATE, _plate_size, _refresh_manifest_view, long_job, ws_manager):
    """Build the router. Keyword-only: exactly what the section takes from app.py (5 names).
    Returns (router, exports) – exports: none."""
    router = APIRouter()



    def _align_oom_message(e: BaseException, w: int, h: int, n: int) -> str:
        """HTTP 507 text of the alignment endpoints: our own readable RamLimitError, else a generic hint."""
        from engine.procpool import RamLimitError
        if isinstance(e, RamLimitError):
            return str(e)
        return f"Brak pamięci RAM przy {w}x{h}, {n} klatek. Skróć zakres klatek albo ustaw mniejszą liczbę procesów (align_workers)."


    @router.post("/api/plate/align/measure")
    @long_job("Pomiar wyrównania")
    async def measure_plate_alignment(req: AlignMeasureRequest):
        if not STATE.get("manifest") or not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        device = (req.device or "").strip().lower() or None
        if device not in (None, "auto", "gpu", "cpu"):
            raise HTTPException(status_code=400, detail="device: auto | gpu | cpu")

        manifest = STATE["manifest"]
        plates = manifest.get("plates", {})
        beauty_pat = plates.get("beauty")
        if not beauty_pat:
            raise HTTPException(status_code=400, detail="Brak zdefiniowanego plate'u beauty w manifeście")

        aux_pat = plates.get(req.plate)
        if not aux_pat:
            # Check if plate key matches a plate path directly
            aux_pat = plates.get("backlit" if req.aux_plate_type == "backlit" else "cleanplate")
            if not aux_pat:
                raise HTTPException(status_code=400, detail=f"Nie znaleziono ścieżki dla plate'u '{req.plate}' w manifeście")

        first_f = req.first_frame or manifest.get("first", 1)
        last_f = req.last_frame or manifest.get("last", 100)
        w, h = _plate_size()

        work_dir = Path(STATE["work_dir"])
        align_dir = work_dir / "align"
        align_dir.mkdir(parents=True, exist_ok=True)

        # exclusions (engine/align_exclude.py): objects' cores (default: every object that has one), dilation, box
        from engine.align_exclude import exclusion_spec, objects_with_cores
        if req.exclude_box is not None and (len(req.exclude_box) != 4 or req.exclude_box[2] <= req.exclude_box[0]
                                            or req.exclude_box[3] <= req.exclude_box[1]):
            raise HTTPException(status_code=400, detail="exclude_box: [x0, y0, x1, y1] w px plate'u, x1 > x0, y1 > y0")
        if req.exclude_dilate_px < 0 or req.exclude_dilate_px > 400:
            raise HTTPException(status_code=400, detail="exclude_dilate_px: 0–400")
        if (req.camera_lock or "auto").lower() not in ("auto", "on", "off"):
            raise HTTPException(status_code=400, detail="camera_lock: auto | on | off")
        names = objects_with_cores(work_dir) if req.exclude_objects is None else list(req.exclude_objects)
        exclusion = exclusion_spec(work_dir, names, range(first_f, last_f + 1), dilate_px=req.exclude_dilate_px,
                                   box=req.exclude_box)

        loop = asyncio.get_running_loop()

        def on_progress(done: int, total: int, f_num: int):
            JOBS.progress(done, total, f_num)
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "align_progress",
                    "plate": req.plate,
                    "done": done,
                    "total": total,
                    "frame": f_num
                }),
                loop
            )

        from engine.align import AlignmentPipeline, NukeAlignGroupGenerator

        try:
            report = await loop.run_in_executor(
                None,
                lambda: AlignmentPipeline.measure_sequence(
                    beauty_pattern=beauty_pat,
                    aux_pattern=aux_pat,
                    first_frame=first_f,
                    last_frame=last_f,
                    plate_name=req.plate,
                    aux_plate_type=req.aux_plate_type,
                    plate_w=w,
                    plate_h=h,
                    progress_cb=on_progress,
                    grid_dir=align_dir / f"grid_{req.plate}",
                    step_cb=JOBS.progress,
                    cancel_check=JOBS.check,
                    device=device,
                    exclusion=exclusion,
                    camera_lock=(req.camera_lock or "auto").lower(),
                    fullres_check=bool(req.fullres_check),
                    global_only=bool(req.global_only),
                )
            )
        except MemoryError as e:
            raise HTTPException(status_code=507, detail=_align_oom_message(e, w, h, last_f - first_f + 1))

        # Save report JSON
        json_path = align_dir / f"Align_{req.plate}.json"
        nk_path = align_dir / f"Align_{req.plate}.nk"
        NukeAlignGroupGenerator.generate_script(
            plate_name=req.plate,
            report=report,
            plate_w=w,
            plate_h=h,
            output_nk_path=nk_path,
            output_json_path=json_path
        )

        report.nk_script_path = str(nk_path)
        report.json_path = str(json_path)

        await ws_manager.broadcast({
            "type": "align_complete",
            "plate": req.plate,
            "recommendation": report.recommendation,
            "residual_median": report.residual_median,
            "residual_max": report.residual_max,
            "method": report.method,
            "needs_correction": report.needs_correction
        })

        return report.to_dict()


    @router.post("/api/plate/align/export_nk")
    def export_align_nk(req: AlignExportNkRequest):
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "align" / f"Align_{req.plate}.json"
        nk_path = work_dir / "align" / f"Align_{req.plate}.nk"

        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Brak zapisanego pomiaru wyrównania dla '{req.plate}'. Najpierw uruchom pomiar.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        from engine.align import AlignmentReport, NukeAlignGroupGenerator
        report = AlignmentReport.from_dict(data)

        manifest = STATE["manifest"]
        w, h = _plate_size()

        script = NukeAlignGroupGenerator.generate_script(
            plate_name=req.plate,
            report=report,
            plate_w=w,
            plate_h=h,
            output_nk_path=nk_path,
            output_json_path=json_path
        )

        valid, msg = NukeAlignGroupGenerator.validate_nuke_syntax(script)
        if not valid:
            raise HTTPException(status_code=500, detail=f"Błąd składni skryptu Nuke: {msg}")

        return {
            "success": True,
            "nk_path": str(nk_path),
            "json_path": str(json_path),
            "script": script
        }


    @router.post("/api/plate/align/apply")
    @long_job("Zapis wyrównania")
    async def apply_plate_alignment(req: AlignApplyRequest):
        if not req.confirmed:
            raise HTTPException(status_code=400, detail="Operacja wymaga potwierdzenia użytkownika ('confirmed': true)")

        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "align" / f"Align_{req.plate}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Brak pomiaru wyrównania dla '{req.plate}'")

        with open(json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        from engine.align import AlignmentReport, PlateWarper
        report = AlignmentReport.from_dict(report_data)

        manifest = STATE["manifest"]
        plates = manifest.get("plates", {})
        aux_pat = plates.get(req.plate)
        if not aux_pat:
            aux_pat = plates.get("backlit" if report.aux_plate_type == "backlit" else "cleanplate")
            if not aux_pat:
                raise HTTPException(status_code=400, detail=f"Brak ścieżki plate'u '{req.plate}'")

        # Output directory alongside the original plate folder: <plate folder>_align (SPEC 4a pkt 3)
        sample_path = Path(aux_pat.replace("####", "0001").replace("%04d", "0001"))
        src_parent = sample_path.parent
        align_dest_dir = src_parent.parent / f"{src_parent.name}_align"

        grid_dir = work_dir / "align" / f"grid_{req.plate}"

        loop = asyncio.get_running_loop()

        def on_progress(done: int, total: int, f_num: int):
            JOBS.progress(done, total, f_num)
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "align_apply_progress",
                    "plate": req.plate,
                    "done": done,
                    "total": total,
                    "frame": f_num
                }),
                loop
            )

        try:
            written = await loop.run_in_executor(
                None,
                lambda: PlateWarper.warp_sequence(
                    aux_pattern=aux_pat,
                    report=report,
                    dest_dir=align_dest_dir,
                    grid_dir=grid_dir if grid_dir.exists() else None,
                    progress_cb=on_progress
                )
            )
        except MemoryError as e:
            raise HTTPException(status_code=507, detail=_align_oom_message(e, *_plate_size(), len(report.frames)))

        # Register the aligned plate in project.json (manifest is read-only)
        align_pat = str(align_dest_dir / sample_path.name.replace("0001", "####"))
        STATE["shot_project"].set_plate(f"{req.plate}_align", align_pat)
        _refresh_manifest_view()

        return {
            "success": True,
            "aligned_count": len(written),
            "aligned_dir": str(align_dest_dir),
            "pattern": align_pat
        }


    @router.get("/api/plate/align/report")
    def get_align_report(plate: str):
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "align" / f"Align_{plate}.json"
        if not json_path.exists():
            return {"exists": False}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from engine.align import stale_report_reason
        return {"exists": True, "report": data, "stale": stale_report_reason(data, work_dir)}

    # ---- align_v2 step 3 (2026-11-03): residual map of the aligned plate (engine/align_residual.py) for stage 0
    def _residual_paths(plate: str, frame: int):
        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "align" / f"Align_{plate}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Brak pomiaru wyrównania dla '{plate}' – najpierw „Zmierz dopasowanie”")
        out_dir = work_dir / "align" / f"residual_{plate}"
        return work_dir, json_path, out_dir / f"r_{frame:04d}.png", out_dir / f"r_{frame:04d}.json"

    def _residual_compute(plate: str, frame: int, side: int):
        """PNG (BGRA heat map, `side` px wide) + stats JSON of the frame's residual; cached next to the report,
        recomputed when the report is newer."""
        import cv2
        import numpy as np
        work_dir, json_path, png, js = _residual_paths(plate, frame)
        if png.exists() and js.exists() and png.stat().st_mtime >= json_path.stat().st_mtime:
            try:
                st = json.loads(js.read_text(encoding="utf-8"))
                if int(st.get("side", 0)) == int(side):
                    return png, st
            except (OSError, ValueError):
                pass
        from engine import align_grid
        from engine.align import AlignmentReport
        from engine.align_exclude import exclusion_spec, frame_exclusion
        from engine.align_residual import frame_residual, residual_png
        from engine.exr_io import imread_unchanged
        data = json.loads(json_path.read_text(encoding="utf-8"))
        report = AlignmentReport.from_dict(data)
        fr = next((f for f in report.frames if f.frame == frame), None)
        if fr is None:
            raise HTTPException(status_code=404, detail=f"Klatka {frame} poza zakresem pomiaru (F{report.frame_range[0]}–F{report.frame_range[1]})")
        plates = (STATE["manifest"] or {}).get("plates") or {}
        beauty_pat, aux_pat = plates.get("beauty"), plates.get(plate)
        if not beauty_pat or not aux_pat:
            raise HTTPException(status_code=400, detail="Brak plate'u beauty albo pomocniczego w projekcie")
        fp = lambda pat: pat.replace("####", f"{frame:04d}").replace("%04d", f"{frame:04d}")  # noqa: E731
        beauty, aux = imread_unchanged(fp(beauty_pat)), imread_unchanged(fp(aux_pat))
        if beauty is None or aux is None:
            raise HTTPException(status_code=404, detail=f"Brak klatki {frame} plate'u")
        gp = work_dir / "align" / f"grid_{plate}" / (fr.grid_rel_path or f"grid_{frame:04d}.exr")
        grid = align_grid.read_grid(gp) if fr.method == "local_stmap" and gp.exists() else None
        ex_info = data.get("exclusions") or {}
        spec = exclusion_spec(work_dir, ex_info.get("objects") or [], [frame], dilate_px=int(ex_info.get("dilate_px") or 0),
                              box=ex_info.get("box"))
        ex = frame_exclusion(spec, frame, beauty.shape[1], beauty.shape[0])
        if spec.get("box"):
            x0, y0, x1, y1 = spec["box"]
            e2 = np.zeros(beauty.shape[:2], bool) if ex is None else ex
            e2[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            ex = e2
        res = frame_residual(beauty, aux, fr, grid, mode=report.aux_plate_type, exclude=ex)
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), residual_png(res["map"], out_w=int(side), vmax=1.0))
        st = {**res["stats"], "frame": frame, "plate": plate, "side": int(side), "vmax_px": 1.0,
              "mask_frac": res["mask_frac"], "method": fr.method, "created_at": data.get("created_at")}
        js.write_text(json.dumps(st), encoding="utf-8")
        return png, st

    @router.get("/api/plate/align/residual/{frame}")
    def align_residual_png(frame: int, plate: str, side: int = 1024):
        """Heat map PNG (BGRA, 0 px = blue … 1 px and above = red, no measurement = transparent) of the residual of
        the aligned plate on frame `frame` – drawn over the plate in stage 0 („Mapa reszty”)."""
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        png, _st = _residual_compute(plate, frame, max(128, min(4096, int(side))))
        return FileResponse(png, media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.get("/api/plate/align/residual_stats/{frame}")
    def align_residual_stats(frame: int, plate: str, side: int = 1024):
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        _png, st = _residual_compute(plate, frame, max(128, min(4096, int(side))))
        return st

    return router, {}
