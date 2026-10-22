# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""api_deflicker.py – deflicker endpoints: measurement → automatic settings → 1024 px preview (docs/PLAN_DEFLICKER_AUTO.md § 2–5).

The routes are built by `make_router(ns)` with app.py's namespace (long_job, STATE, JOBS, _objects, _pipeline, _work_area,
ProxyGenerator) so this module needs no import of app.py; app.py does `app.include_router(make_router(globals()))`.

- POST /api/plate/clean/analyze   – window of ±radius frames around `frame` (default: middle of the work area), downscaled to
  eval_w; regions = whole frame + SAM 3.1 cores of the objects that have them; returns the measurement, the automatic
  settings with their notes, and the legacy flicker numbers of the old modal.
- POST /api/plate/clean/preview   – the deflicker with the palette's settings on the same window at eval_w; before / after
  numbers per region, heat maps and the before / after frames as files under <work>/deflicker/preview_<plate>/.
- GET  /api/plate/clean/preview/file?plate=&name= – one of those files.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.clean import DeflickerMetrics, as_bgr
from engine.flicker_auto import DeflickerSettings, auto_settings
from engine.flicker_measure import measure_flicker
from engine.flicker_preview import preview_deflicker


class CleanAnalyzeRequest(BaseModel):
    plate: str = "beauty"
    frame: Optional[int] = None     # centre of the window; default = middle of the work area / shot
    radius: int = 10                # ± frames
    objects: bool = True            # SAM 3.1 cores of the computed objects as regions
    eval_w: int = 1024
    window: Optional[List[int]] = None   # [first, last] explicitly (the preview uses the measured window)


class CleanPreviewRequest(CleanAnalyzeRequest):
    settings: Dict[str, Any] = {}   # DeflickerSettings fields from the palette
    export_mode: str = "nuke_group"


SETTING_KEYS = ("strength", "window", "grid", "whole_frame", "per_channel", "texture", "texture_radius", "smoothness_px")


def settings_from_dict(d: Dict[str, Any]) -> DeflickerSettings:
    """Palette values -> DeflickerSettings (unknown keys ignored, types cast, missing = engine defaults)."""
    s = DeflickerSettings(strength=5, window=20, grid=True)
    for k in SETTING_KEYS:
        if k in d and d[k] is not None:
            cur = getattr(s, k)
            setattr(s, k, bool(d[k]) if isinstance(cur, bool) else type(cur)(d[k]))
    s.strength = int(min(max(s.strength, 1), 8))
    s.window = int(min(max(s.window, 3), 30))
    s.texture_radius = int(min(max(s.texture_radius, 1), 6))
    s.smoothness_px = int(min(max(s.smoothness_px, 2), 16))
    return s


def make_router(ns: Dict[str, Any]) -> APIRouter:
    router = APIRouter()
    long_job, STATE, JOBS = ns["long_job"], ns["STATE"], ns["JOBS"]
    resolve = ns["ProxyGenerator"].resolve_frame_path

    def _project() -> Dict[str, Any]:
        if not STATE.get("manifest") or not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        return STATE["manifest"]

    def _window(req: CleanAnalyzeRequest) -> Tuple[int, int, List[Tuple[int, Path]]]:
        manifest = _project()
        pat = (manifest.get("plates") or {}).get(req.plate)
        if not pat:
            raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{req.plate}' w manifeście")
        first, last = int(manifest.get("first", 1)), int(manifest.get("last", 100))
        wa = None
        try:
            wa = ns["_work_area"]() if "_work_area" in ns else None
        except Exception:
            wa = None
        lo, hi = (int(wa[0]), int(wa[1])) if wa else (first, last)
        if req.window and len(req.window) == 2:
            f0, f1 = max(first, int(req.window[0])), min(last, int(req.window[1]))
        else:
            centre = int(req.frame) if req.frame is not None else (lo + hi) // 2
            centre = min(max(centre, lo), hi)
            r = max(1, int(req.radius))
            f0, f1 = max(lo, centre - r), min(hi, centre + r)
        paths = [(fn, resolve(pat, fn)) for fn in range(f0, f1 + 1)]
        paths = [(fn, p) for fn, p in paths if p.exists()]
        if len(paths) < 3:
            raise HTTPException(status_code=400, detail=f"Za mało klatek plate'u {req.plate} w oknie F{f0}–F{f1} ({len(paths)})")
        return paths[0][0], paths[-1][0], paths

    def _load_small(paths: List[Tuple[int, Path]], eval_w: int, step_cb=None) -> List[np.ndarray]:
        """Frames of the window downscaled to eval_w (the whole window never sits in RAM at full size); 8-bit -> 0..1."""
        from engine.exr_io import read_rgb
        out = []
        for i, (fn, p) in enumerate(paths):
            if p.suffix.lower() == ".exr":
                img = read_rgb(p)
            else:
                raw = cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
                if raw is None:
                    raise HTTPException(status_code=400, detail=f"Nie udało się wczytać klatki {p}")
                img = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0 if raw.dtype == np.uint8 else 1.0)
            img = as_bgr(np.asarray(img, np.float32))
            h, w = img.shape[:2]
            ew = min(int(eval_w), w)
            eh = max(8, round(h * ew / w))
            out.append(cv2.resize(img, (ew, eh), interpolation=cv2.INTER_AREA) if (eh, ew) != (h, w) else img)
            if step_cb:
                step_cb(i + 1, len(paths))
        return out

    def _masks(frame_nums: List[int], eval_size: Tuple[int, int], want: bool) -> Optional[Dict[str, List[np.ndarray]]]:
        """SAM 3.1 cores of every object that has them for all frames of the window, at eval size."""
        if not want:
            return None
        try:
            store, pipe = ns["_objects"](), ns["_pipeline"]()
        except Exception:
            return None
        out: Dict[str, List[np.ndarray]] = {}
        for o in store.list():
            name = o.get("name")
            try:
                cores = pipe.load_core(name, frame_nums)
            except Exception:
                continue
            out[name] = [cv2.resize(cores[fn].astype(np.uint8), eval_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                         for fn in frame_nums]
        return out or None

    @router.post("/api/plate/clean/analyze")
    @long_job("Pomiar migotania: {plate}")
    def analyze(req: CleanAnalyzeRequest):
        f0, f1, paths = _window(req)
        JOBS.update(unit="pct")
        frames = _load_small(paths, req.eval_w, lambda d, t: JOBS.progress("Pomiar migotania: wczytywanie klatek", int(10 * d / t), 100))
        eh, ew = frames[0].shape[:2]
        masks = _masks([fn for fn, _ in paths], (ew, eh), req.objects)
        m = measure_flicker(frames, masks, eval_w=ew,
                            progress_cb=lambda d, t: JOBS.progress("Pomiar migotania: pary klatek", 10 + int(80 * d / t), 100))
        s = auto_settings(m)
        JOBS.progress("Pomiar migotania: podsumowanie", 95, 100)
        legacy = DeflickerMetrics.measure_flow_aligned(frames)
        STATE["clean_analysis"] = {"plate": req.plate, "window": [f0, f1], "settings": s.to_dict(), "t": time.time()}
        return {"success": True, "window": [f0, f1], "n_frames": len(frames), "eval_w": m.eval_w, "eval_h": m.eval_h,
                "regions": list(m.regions), "measure": m.to_dict(), "settings": s.to_dict(),
                "flicker": legacy.to_dict(), "needs_deflicker": legacy.mean > 4.0 or legacy.p95 > 6.0}

    @router.post("/api/plate/clean/preview")
    @long_job("Podgląd deflickera: {plate}")
    def preview(req: CleanPreviewRequest):
        f0, f1, paths = _window(req)
        JOBS.update(unit="pct")
        frames = _load_small(paths, req.eval_w, lambda d, t: JOBS.progress("Podgląd deflickera: wczytywanie klatek", int(10 * d / t), 100))
        eh, ew = frames[0].shape[:2]
        masks = _masks([fn for fn, _ in paths], (ew, eh), req.objects)
        s = settings_from_dict(req.settings)
        exr = req.export_mode != "nuke_group"
        r = preview_deflicker(frames, s.to_process_kwargs(exr), masks, eval_w=ew,
                              progress_cb=lambda step, d, t: JOBS.progress(step, 10 + int(85 * d / t), 100))
        out_dir = Path(STATE["work_dir"]) / "deflicker" / f"preview_{req.plate}"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "map_before.png"), r.map_before)
        cv2.imwrite(str(out_dir / "map_after.png"), r.map_after)
        for i, (b, a) in enumerate(zip(r.frames_before, r.frames_after)):
            cv2.imwrite(str(out_dir / f"b_{i:03d}.jpg"), b, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            cv2.imwrite(str(out_dir / f"a_{i:03d}.jpg"), a, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        tok = int(time.time() * 1000) % 10 ** 9

        def url(name: str) -> str:
            return f"/api/plate/clean/preview/file?plate={req.plate}&name={name}&t={tok}"

        d = r.to_dict()
        d.update({"success": True, "window": [f0, f1], "frames": [fn for fn, _ in paths], "settings": s.to_dict(),
                  "exr": exr, "files": {"map_before": url("map_before.png"), "map_after": url("map_after.png"),
                                        "frames": [[url(f"b_{i:03d}.jpg"), url(f"a_{i:03d}.jpg")] for i in range(len(frames))]}})
        return d

    @router.get("/api/plate/clean/preview/file")
    def preview_file(plate: str, name: str, t: Optional[int] = None):
        _project()
        if any(ch in name for ch in ("/", "\\")) or ".." in name or any(ch in plate for ch in ("/", "\\", ".")):
            raise HTTPException(status_code=400, detail="Zła nazwa pliku")
        p = Path(STATE["work_dir"]) / "deflicker" / f"preview_{plate}" / name
        if not p.exists():
            raise HTTPException(status_code=404, detail="Brak pliku podglądu – policz podgląd ponownie")
        return FileResponse(str(p), headers={"Cache-Control": "no-store"})

    return router
