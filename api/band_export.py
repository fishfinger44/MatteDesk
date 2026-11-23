# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Band export endpoint (user request 2026-11-23): a stage band as a Nuke Roto / After Effects masks.

    POST /api/object/{name}/stage_band/{stage}/export?fmt=nk|jsx&what=band|tracker

`stage` = matanyone | videomama | ltx | zone:<id> (the same keys as …/stage_band/{stage}/{f}/lines). The lines of every
frame of the work area where the band exists (the computed core) go through engine/band_export (shapes followed across
frames, one point count per shape) -> objects/<o>/band_export/PAS_<stage>_<o>.nk and .jsx (both written on every call);
the answer carries the text of the asked format (`text`, the .nk is clipboard-ready) and both paths.
`what=tracker` (2026-11-23): the band points (CoTracker3) + the mask centre as a Nuke Tracker4 / After Effects nulls
(engine/track_export) -> TRK_<stage>_<o>.nk / .jsx.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, HTTPException

from engine.jobs import JOBS

STAGE_LABEL = {"matanyone": "MatAnyone2", "videomama": "VideoMaMa", "ltx": "LTX-2.5"}

# frames of an export computed side by side (user 2026-11-23: a 6K group band took ~3.7 s a frame, 49 frames ~3 min):
# every frame is independent (composition, band geometry, points), OpenCV / numpy release the GIL. Few workers – each
# holds a few 6K float frames (~0.2 GB)
EXPORT_WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))


def per_frame(frames: List[int], fn: Callable, label: str, workers: int = EXPORT_WORKERS) -> Dict[int, Any]:
    """{frame: fn(frame)} over `frames` in a thread pool; None results left out. Progress and „Przerwij” from the
    calling (job) thread; the first error stops the rest and is raised."""
    out: Dict[int, Any] = {}
    if not frames:
        return out
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(frames))), thread_name_prefix="band-export") as ex:
        futs = {ex.submit(fn, f): f for f in frames}
        try:
            for i, fu in enumerate(as_completed(futs)):
                if JOBS.cancelled():
                    raise HTTPException(status_code=409, detail="przerwane")
                r = fu.result()
                if r is not None:
                    out[futs[fu]] = r
                JOBS.progress(f"{label} {i + 1}/{len(frames)}", i + 1, len(frames))
        except BaseException:
            for fu in futs:
                fu.cancel()
            raise
    return out


def make(*, STATE, long_job, objects: Callable, lines_fn: Callable, band_frames: Callable, work_frames: Callable,
         plate_size: Callable, points_fn: Callable = None):
    """Build the router. Keyword-only: `lines_fn(name, stage, frame)` = app.stage_band_lines (the preview lines),
    `band_frames(name, frame)` = frames the band exists in, `work_frames()` = the work area, `points_fn(name, stage,
    frame)` = the band points' state in a frame (app.get_stage_band). Returns (router, {})."""
    router = APIRouter()

    @router.post("/api/object/{name}/stage_band/{stage}/export")
    @long_job("Eksport pasa: {name}")
    def export_band(name: str, stage: str, fmt: str = "nk", what: str = "band") -> Dict[str, Any]:
        from engine import band_export as BE
        if fmt not in ("nk", "jsx"):
            raise HTTPException(status_code=400, detail="fmt: nk | jsx")
        if what not in ("band", "tracker"):
            raise HTTPException(status_code=400, detail="what: band | tracker")
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        if stage not in STAGE_LABEL and not stage.startswith("zone:"):
            raise HTTPException(status_code=400, detail="stage: matanyone | videomama | ltx | zone:<id>")
        wf = work_frames()
        have = band_frames(name, wf[0])
        # a group has no core/ of its own: band_frames falls back to [frame] – then the whole work area, frames without
        # a mask skipped (the preview lines of a group exist wherever its members do; user report 2026-11-23: F1 only)
        frames: List[int] = list(wf) if list(have) == [wf[0]] else [f for f in wf if f in set(have)]
        if not frames:
            raise HTTPException(status_code=409, detail="Brak klatek z pasem w obszarze roboczym (rdzeń SAM 3.1 niepoliczony?)")

        def one(f: int):
            try:
                d = lines_fn(name, stage, f)
            except HTTPException as e:
                if e.status_code == 404:          # no base mask in this frame
                    return None
                raise
            if d.get("approx"):
                raise HTTPException(status_code=409, detail="Maska 1+2 jeszcze się liczy (podgląd przybliżony) – spróbuj za chwilę")
            p = None
            if what == "tracker" and points_fn is not None:
                st_ = points_fn(name, stage, f)
                if st_.get("tracking"):
                    raise HTTPException(status_code=409, detail="CoTracker3 jeszcze śledzi punkty pasa – spróbuj za chwilę")
                p = st_.get("points") or []
            return (None if d.get("outside") else d, p)

        # the first frame alone: it starts the mask 1+2 warm-up / point tracking if needed (`approx` / `tracking` 409
        # then), the shared caches are filled before the pool reads them
        res = per_frame(frames[:1], one, "linie pasa", workers=1)
        res.update(per_frame(frames[1:], one, "linie pasa"))
        lines: Dict[int, Dict[str, Any]] = {f: r[0] for f, r in res.items() if r[0] is not None}
        pts: Dict[int, List[Dict[str, Any]]] = {f: r[1] for f, r in res.items() if r[1] is not None}
        JOBS.progress("kształty", len(frames), len(frames))
        frames = sorted(lines) or frames          # the frames that really have a band (label, AE time 0)
        if what == "tracker":
            return _tracker(name, stage, frames, lines, pts, fmt, wf[0])
        shp = BE.shapes(lines)
        if not shp:
            raise HTTPException(status_code=409, detail="Pas jest pusty w tych klatkach")
        w, h = plate_size()
        st = STAGE_LABEL.get(stage, f"strefa {stage[5:]}")
        safe = (lambda s: re.sub(r"\W", "_", s) or "x")
        node = f"PAS_{safe(stage.replace('zone:', 'strefa_'))}_{safe(name)}"
        label = f"Pas {st} · {name} · F{frames[0]}–{frames[-1]}"
        out = objects().dir(name) / "band_export"
        out.mkdir(parents=True, exist_ok=True)
        nk, jsx = out / f"{node}.nk", out / f"{node}.jsx"
        nk_text = BE.nk_text(shp, w, h, node, label)
        jsx_text = BE.jsx_text(shp, w, h, wf[0], node, label)
        nk.write_text(nk_text, "utf-8", newline="\n")
        jsx.write_text(jsx_text, "utf-8", newline="\n")
        return {"text": nk_text if fmt == "nk" else jsx_text, "path": str(nk if fmt == "nk" else jsx).replace("\\", "/"),
                "nk": str(nk).replace("\\", "/"), "jsx": str(jsx).replace("\\", "/"), "node": node,
                "frames": [frames[0], frames[-1]], "shapes": BE.summary(shp)}

    def _tracker(name, stage, frames, lines, pts, fmt, first):
        from engine import track_export as TE
        trk = TE.tracks(pts, lines)
        if not trk:
            raise HTTPException(status_code=409, detail="Nic do śledzenia – pas pusty w tych klatkach")
        w, h = plate_size()
        st = STAGE_LABEL.get(stage, f"strefa {stage[5:]}")
        safe = (lambda s: re.sub(r"\W", "_", s) or "x")
        node = f"TRK_{safe(stage.replace('zone:', 'strefa_'))}_{safe(name)}"
        label = f"Tracker pasa {st} · {name} · F{frames[0]}–{frames[-1]}"
        out = objects().dir(name) / "band_export"
        out.mkdir(parents=True, exist_ok=True)
        return write_tracker(out, node, label, trk, w, h, first, fmt)

    return router, {}


def write_tracker(out, node: str, label: str, trk, w: int, h: int, first: int, fmt: str) -> Dict[str, Any]:
    """Write <node>.nk (Tracker4) + <node>.jsx (nulls) into `out`; the answer like the band export's."""
    from engine import track_export as TE
    nk, jsx = out / f"{node}.nk", out / f"{node}.jsx"
    nk_text = TE.nk_text(trk, int(w), int(h), node, label)
    jsx_text = TE.jsx_text(trk, int(w), int(h), int(first), node, label)
    nk.write_text(nk_text, "utf-8", newline="\n")
    jsx.write_text(jsx_text, "utf-8", newline="\n")
    fr = sorted({f for t in trk for f in t["keys"]})
    return {"text": nk_text if fmt == "nk" else jsx_text, "path": str(nk if fmt == "nk" else jsx).replace("\\", "/"),
            "nk": str(nk).replace("\\", "/"), "jsx": str(jsx).replace("\\", "/"), "node": node,
            "frames": [fr[0], fr[-1]] if fr else [], "tracks": TE.summary(trk)}
