# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate cleaning endpoints: deflicker and denoise (stage E1c, SPEC 4c)

Routes of app.py's section „Plate cleaning endpoints: deflicker and denoise (stage E1c, SPEC 4c)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from engine.clean import DeflickerMetrics, NukeDeflickerGroupGenerator, PinnedFrameCache, PlateCleaner, clean_cache_budget
from engine.ingest import ProxyGenerator
from engine.jobs import JOBS

import logging

logger = logging.getLogger("matte_desk.api.plate_clean")


# -----------------------------------------------------------------------------
# Plate Cleaning Endpoints: Deflicker & Denoise (Stage E1c, SPEC Section 4c)
# -----------------------------------------------------------------------------

class CleanMeasureRequest(BaseModel):
    plate: str = "beauty"
    max_frames: int = 16


class CleanRunRequest(BaseModel):
    plate: str = "beauty"
    target_reduction: float = 0.70
    enable_stage1: bool = True
    enable_stage2: bool = True
    enable_stage3: bool = False
    enable_stage4: bool = True
    enable_denoise: bool = False
    export_mode: str = "nuke_group"  # "nuke_group" | "exr_deflicker" | "exr_deflicker_dn"
    confirmed: bool = False
    # palette knobs (docs/PLAN_DEFLICKER_AUTO.md § 4); None = engine defaults
    stage1_window: Optional[int] = None
    stage1_iterations: Optional[int] = None
    stage3_radius: Optional[int] = None
    stage1_sigma: Optional[float] = None
    stage4_luma: Optional[bool] = None
    exr_dir: Optional[str] = None   # EXR export folder; None / "" = default <plate folder>_deflicker[_dn] next to the plate
    denoise_device: Optional[str] = None   # auto | gpu | cpu; None = config `denoise_device` (engine/denoise_gpu.py)


class CleanExportNkRequest(BaseModel):
    plate: str = "beauty"


def make(*, STATE, _plate_size, _refresh_manifest_view, long_job, ws_manager):
    """Build the router. Keyword-only: exactly what the section takes from app.py (5 names).
    Returns (router, exports) – exports: none."""
    router = APIRouter()



    def _clean_default_exr_dir(pat: str, denoise: bool) -> Path:
        """<plate folder>_deflicker / _deflicker_dn next to the original (SPEC 4c.3)."""
        source_p = Path(pat)
        return source_p.parent.parent / f"{source_p.parent.name}_{'deflicker_dn' if denoise else 'deflicker'}"


    def _clean_exr_dir(pat: str, denoise: bool, chosen: Optional[str]) -> Path:
        """Folder for the deflickered EXRs: the user's choice or the default; never the plate's own folder (same file names ->
    the original frames would be overwritten)."""
        dest = Path(chosen).expanduser() if chosen and chosen.strip() else _clean_default_exr_dir(pat, denoise)
        if not dest.is_absolute():
            raise HTTPException(status_code=400, detail=f"Katalog EXR musi być pełną ścieżką: {chosen}")
        try:
            same = dest.resolve() == Path(pat).parent.resolve()
        except OSError:
            same = False
        if same:
            raise HTTPException(status_code=400, detail="Katalog EXR = katalog oryginalnego plate'u – zapis nadpisałby oryginalne klatki. Wybierz inny katalog.")
        return dest


    @router.get("/api/plate/clean/exr_dir")
    def get_clean_exr_dir(plate: str = "beauty", denoise: bool = False):
        """Default EXR export folder of a plate (the palette shows it before the run)."""
        if not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        pat = STATE["manifest"].get("plates", {}).get(plate)
        if not pat:
            raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{plate}' w manifeście")
        return {"path": str(_clean_default_exr_dir(pat, denoise)).replace("\\", "/")}


    def _clean_oom_message(w: int, h: int, n: int) -> str:
        return f"Brak pamięci RAM przy {w}x{h}, {n} klatek. Skróć zakres klatek."


    @router.post("/api/plate/clean/measure")
    @long_job("Pomiar migotania")
    def measure_plate_clean(req: CleanMeasureRequest):   # plain def: long_job runs it in a worker thread
        if not STATE.get("manifest") or not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        manifest = STATE["manifest"]
        plates = manifest.get("plates", {})
        pat = plates.get(req.plate)
        if not pat:
            raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{req.plate}' w manifeście")

        first_f = manifest.get("first", 1)
        last_f = manifest.get("last", 100)
        total_frames = last_f - first_f + 1
        sample_count = min(req.max_frames, total_frames) if total_frames > 24 else total_frames

        frames = []
        for f_idx in range(sample_count):
            JOBS.check()   # „Przerwij” between frames (a 4K EXR takes a while)
            fn = first_f + f_idx
            f_path = Path(pat.replace("####", f"{fn:04d}"))
            if not f_path.exists():
                break
            img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img)

        if len(frames) < 2:
            raise HTTPException(status_code=400, detail=f"Nie udało się wczytać klatek z {pat}")

        try:
            flicker = DeflickerMetrics.measure_flow_aligned(frames)
            variance = DeflickerMetrics.measure_temporal_variance(frames)
            JOBS.check()   # a „Przerwij” during the metrics is a cancel, not a finished measure
        except MemoryError:
            h0, w0 = frames[0].shape[:2]
            raise HTTPException(status_code=507, detail=f"Brak pamięci RAM przy {w0}x{h0}, {len(frames)} klatek. Zmniejsz liczbę próbkowanych klatek.")

        return {
            "success": True,
            "flicker": flicker.to_dict(),
            "temporal_variance": round(variance, 4),
            "needs_deflicker": flicker.mean > 4.0 or flicker.p95 > 6.0
        }


    @router.post("/api/plate/clean/run")
    @long_job("Deflicker")
    async def run_plate_clean(req: CleanRunRequest):
        # the whole run off the event loop (EXR reads / writes, gain maps): the loop stays for /api/jobs, „Przerwij”, the
        # WebSocket; async only for the loop on_progress broadcasts through
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _run_plate_clean(req, loop))


    def _run_plate_clean(req: CleanRunRequest, loop: asyncio.AbstractEventLoop):
        if not STATE.get("manifest") or not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        manifest = STATE["manifest"]
        plates = manifest.get("plates", {})
        pat = plates.get(req.plate)
        if not pat:
            raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{req.plate}' w manifeście")

        if req.export_mode in ("exr_deflicker", "exr_deflicker_dn") and not req.confirmed:
            raise HTTPException(
                status_code=400,
                detail="Wymagane jest potwierdzenie zapisu pełnych klatek EXR na dysku (zasada IZOLACJI)."
            )

        first_f = manifest.get("first", 1)
        last_f = manifest.get("last", 100)
        work_dir = Path(STATE["work_dir"])
        plate_w, plate_h = _plate_size()

        # Frames are read from disk on demand (RGB only, BGR order; alpha / other layers are copied untouched on export):
        # a 4K float32 frame is ~134 MB, so the whole sequence is never held in RAM (the cleaner passes over it a few times)
        from engine.exr_io import read_rgb
        file_paths = []
        for fn in range(first_f, last_f + 1):
            f_path = ProxyGenerator.resolve_frame_path(pat, fn)
            if not f_path.exists():
                break
            file_paths.append(f_path)

        if len(file_paths) < 2:
            raise HTTPException(status_code=400, detail=f"Brak wystarczającej liczby klatek ({len(file_paths)}) dla plate'u {req.plate}")

        def load_frame(i: int) -> np.ndarray:
            f_path = file_paths[i]
            img = read_rgb(f_path) if f_path.suffix.lower() == ".exr" else cv2.imread(str(f_path), cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail=f"Nie udało się wczytać klatki {f_path}")
            return img.astype(np.float32)

        # frames read once are pinned in RAM up to a budget (the cleaner sweeps the sequence ~10x; the rest is re-read)
        frames = PinnedFrameCache(len(file_paths), load_frame, clean_cache_budget())
        frames[0]   # unreadable first frame -> clear error before the long run

        # Stage 3 and denoise rebuild pixels: only an EXR export carries them (the Nuke Group applies gain maps only) –
        # without one they are skipped, so the report measures what the result really contains
        write_exr = req.export_mode in ("exr_deflicker", "exr_deflicker_dn")
        notes: List[str] = []
        enable_stage3, enable_denoise = req.enable_stage3, req.enable_denoise
        if not write_exr and (enable_stage3 or enable_denoise):
            enable_stage3 = enable_denoise = False
            notes.append("Etap 3 / odszumianie pominięte – działają tylko przy zapisie EXR (Group w Nuke'u nakłada tylko mapy jasności).")

        # EXR export: each cleaned frame is written as soon as it is ready (stage 3 / denoise never hold the sequence)
        sink = None
        dest_exr_dir = None
        written = [0]
        if write_exr:
            from engine.exr_io import write_rgb_like
            # chosen folder or <plate folder>_deflicker / _deflicker_dn next to the original (SPEC 4c.3)
            source_p = Path(pat)
            dest_exr_dir = _clean_exr_dir(pat, req.export_mode == "exr_deflicker_dn", req.exr_dir)
            dest_exr_dir.mkdir(parents=True, exist_ok=True)
            tag = "+".join(n for n, on in (("stage1", req.enable_stage1), ("stage2", req.enable_stage2),
                                           ("stage4", req.enable_stage4), ("stage3", enable_stage3),
                                           ("denoise", enable_denoise)) if on)

            def sink(idx: int, cl_frame: np.ndarray) -> None:
                # Same encoding as the source (channels, half/float, compression, metadata) – SPEC 4c.3
                src_file = file_paths[idx]
                dest_file = dest_exr_dir / src_file.name
                if src_file.suffix.lower() == ".exr":
                    write_rgb_like(dest_file, cl_frame, src_file, extra_attrs={"matteDesk": f"deflicker {tag}".encode()})
                else:
                    cv2.imwrite(str(dest_file), cl_frame)
                written[0] += 1

        JOBS.update(unit="pct")   # the cleaner reports stage milestones in % (0 / 10 / 40 / ... / 100), not frames
        knobs = {k: getattr(req, k) for k in ("stage1_window", "stage1_iterations", "stage3_radius", "stage1_sigma", "stage4_luma")
                 if getattr(req, k) is not None}

        def on_progress(step_name: str, done: int, total: int):
            JOBS.progress(step_name, done, total)
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast({
                    "type": "clean_progress",
                    "step": step_name,
                    "done": done,
                    "total": total,
                    "plate": req.plate
                }),
                loop
            )

        try:
            cleaned, gains, report, residuals = PlateCleaner.process_sequence(
                frames=frames,
                plate_name=req.plate,
                target_reduction=req.target_reduction,
                enable_stage1=req.enable_stage1,
                enable_stage2=req.enable_stage2,
                enable_stage3=enable_stage3,
                enable_stage4=req.enable_stage4,
                enable_denoise=enable_denoise,
                progress_cb=on_progress,
                frame_sink=sink,
                denoise_device=req.denoise_device,
                cancel_check=JOBS.check,
                **knobs
            )
        except MemoryError:
            raise HTTPException(status_code=507, detail=_clean_oom_message(plate_w, plate_h, len(file_paths)))

        logger.info(f"Deflicker {req.plate}: frame cache {frames.stats()}")
        if report.denoise_device:
            logger.info(f"Deflicker {req.plate}: denoise on {report.denoise_device} {report.denoise_note}")
        if report.denoise_fell_back:
            notes.append("Odszumianie: worker GPU zawiódł w trakcie liczenia – reszta klatek policzona na CPU (szczegóły w logu).")
        clean_dir = work_dir / "deflicker"
        clean_dir.mkdir(parents=True, exist_ok=True)
        gain_dir = clean_dir / f"gain_{req.plate}"
        gain_pattern = PlateCleaner.export_gain_maps(gains, first_frame=first_f, output_dir=gain_dir)
        report.gain_map_pattern = gain_pattern

        # Generate Nuke Group script
        nk_file = clean_dir / f"Deflicker_{req.plate}.nk"
        NukeDeflickerGroupGenerator.generate_script(
            plate_name=req.plate,
            gain_pattern=gain_pattern,
            plate_w=plate_w,
            plate_h=plate_h,
            first_frame=first_f,
            last_frame=last_f,
            report=report,
            output_nk_path=nk_file
        )
        report.nk_group_path = str(nk_file).replace("\\", "/")

        # EXRs were written by the sink while cleaning
        export_dir_str = ""
        if write_exr:
            export_dir_str = str(dest_exr_dir).replace("\\", "/")
            report.export_dir = export_dir_str

            # Update manifest to point beauty or plate to cleaned version if beauty
            new_pat = str(dest_exr_dir / source_p.name).replace("\\", "/")
            STATE["shot_project"].set_plate(f"{req.plate}_clean", new_pat)
            _refresh_manifest_view()

        # Save JSON report
        json_path = clean_dir / f"Deflicker_{req.plate}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "report": report.to_dict(),
            "nk_script_path": report.nk_group_path,
            "export_dir": export_dir_str,
            "exr_written": written[0],
            "notes": notes
        }


    @router.get("/api/plate/clean/report")
    def get_clean_report(plate: str):
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

        work_dir = Path(STATE["work_dir"])
        json_path = work_dir / "deflicker" / f"Deflicker_{plate}.json"
        if not json_path.exists():
            return {"exists": False}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "report": data}

    return router, {}
