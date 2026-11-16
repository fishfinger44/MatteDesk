# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate QA + fix (stage 0, Z26)

Routes of app.py's section „Plate QA + fix (stage 0, Z26)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from engine.project import load_defaults


# ----------------------------------------------------------------------------- Z26: plate QA + fix (stage 0)
class PlateQARequest(BaseModel):
    plate: str = "beauty"
    frames: Optional[List[int]] = None     # [first, last]
    source: bool = False                   # plate switched to a MatteDesk `_fix`: check its source again (change the repair)


class PlateDecisionsRequest(BaseModel):
    decisions: Dict[str, str] = {}         # frame -> mean | flow | gain | keep ("" removes the decision)


class PlateFixRequest(BaseModel):
    plate: str = "beauty"
    decisions: Dict[str, str] = {}         # frame -> mean | flow | gain | keep
    confirmed: bool = False                # EXR / GBs are written only after the user confirmed (plan with sizes first)


def make(*, STATE, _objects, _shot_frames, long_job):
    """Build the router. Keyword-only: exactly what the section takes from app.py (4 names).
    Returns (router, exports) – exports: _plate_gate, _require_plates_ok."""
    router = APIRouter()



    def _plate_fix_rec(plate: str) -> Optional[Dict[str, Any]]:
        proj = STATE.get("shot_project")
        return ((proj.data.get("plate_fix") or {}).get(plate)) if proj else None


    def _plate_source(plate: str) -> Optional[str]:
        """Pattern a repair / a re-check of `plate` starts from: the source of the MatteDesk `_fix` in use, else the plate."""
        from engine.plate_qa import _norm
        cur = ((STATE.get("manifest") or {}).get("plates") or {}).get(plate)
        rec = _plate_fix_rec(plate)
        if rec and cur and _norm(rec.get("pattern")) == _norm(cur):
            return rec.get("source")
        return cur


    def _plate_gate() -> Dict[str, Any]:
        """Z26 gate of stage 1 (like Z18): per plate of the project the QA state; `block` = reason while some flagged
    frame has no decision or decided repairs are not written yet (None = stage 1 open)."""
        from engine.plate_qa import gate_state, load_report
        plates = {k: v for k, v in ((STATE.get("manifest") or {}).get("plates") or {}).items() if v and k != "bg"}
        work = Path(STATE["work_dir"]) if STATE.get("work_dir") else None
        out: Dict[str, Any] = {}
        for name, pattern in plates.items():
            out[name] = gate_state(load_report(work, name) if work else None, pattern, _plate_fix_rec(name))
        bad = [(n, g) for n, g in out.items() if g["blocking"]]
        block = None
        if bad:
            block = "najpierw etap 0 – kontrola klatek: " + "; ".join(
                (f"„{n}”: {len(g['undecided'])} kl. bez decyzji" if g["state"] == "undecided" else f"„{n}”: zapisz naprawę (_fix)")
                for n, g in bad)
        return {"plates": out, "block": block}


    def _require_plates_ok() -> None:
        if not STATE.get("work_dir"):
            return
        block = _plate_gate()["block"]
        if block:
            raise HTTPException(status_code=409, detail=block)


    @router.post("/api/plate/qa")
    @long_job("Kontrola klatek: {plate}", quiet=True)
    async def plate_qa_run(req: PlateQARequest):
        """Z26: control of damaged frames of a plate (missing / unreadable / NaN / size / exposure jump / image jump /
    duplicate) -> matte_desk/plate_qa/<plate>.json with per-frame measures, flagged frames and a suggested repair."""
        from engine.comfy_io import job_progress
        from engine.plate_qa import analyze_plate, save_report, suggest_method
        m = STATE.get("manifest") or {}
        pattern = (m.get("plates") or {}).get(req.plate)
        if not pattern:
            raise HTTPException(status_code=400, detail=f"Brak plate'u {req.plate} w projekcie")
        if req.source:
            pattern = _plate_source(req.plate) or pattern
        frames = _shot_frames()
        if req.frames and len(req.frames) == 2:
            frames = [f for f in frames if req.frames[0] <= f <= req.frames[1]]
        thr = load_defaults().get("plate_qa") or {}
        cb = lambda d, t: job_progress(f"Kontrola klatek: {req.plate}", d, t)
        rep = await asyncio.get_running_loop().run_in_executor(None, lambda: analyze_plate(pattern, frames, thr, cb))
        rep["suggested"] = {str(f): suggest_method(rep["frames"][f]) for f in rep["flagged"]}
        # re-check of the source of the `_fix` in use: the written decisions are the starting point (unchanged -> "fixed")
        rec = _plate_fix_rec(req.plate)
        if req.source and rec and rec.get("source") == pattern:
            rep["decisions"] = {str(k): v for k, v in (rec.get("decisions") or {}).items() if int(k) in rep["flagged"]}
        save_report(Path(STATE["work_dir"]), req.plate, rep)
        return {"success": True, "plate": req.plate, "flagged": rep["flagged"], "suggested": rep["suggested"],
                "seconds": rep["seconds"], "n": rep.get("n")}


    @router.get("/api/plate/qa/{plate}")
    def plate_qa_report(plate: str):
        from engine.plate_qa import load_report
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        rep = load_report(Path(STATE["work_dir"]), plate)
        if rep is None:
            return {"exists": False, "plate": plate}
        rep["frames"] = {str(k): v for k, v in rep["frames"].items()}
        return {"exists": True, **rep, "gate": _plate_gate()["plates"].get(plate), "fix": _plate_fix_rec(plate)}


    @router.put("/api/plate/qa/{plate}/decisions")
    def plate_qa_decisions(plate: str, req: PlateDecisionsRequest):
        """Z26: the user's decision per flagged frame (mean / flow / gain / keep) – stored in the report; the gate of
    stage 1 opens once every flag has one and the repairs are written."""
        from engine.plate_qa import save_decisions
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        try:
            rp = save_decisions(Path(STATE["work_dir"]), plate, req.decisions)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        gate = _plate_gate()
        return {"decisions": rp.get("decisions") or {}, "gate": gate["plates"].get(plate), "block": gate["block"]}


    @router.get("/api/plate/qa/{plate}/preview/{frame}")
    def plate_qa_preview(plate: str, frame: int, method: str = "orig", side: int = 960):
        """Z26: before / after of a repair (JPEG, display conversion like the proxy, longest side <= `side`): method
    orig = the source frame (404 when missing), mean / flow / gain = the repair computed on downscaled neighbours."""
        from fastapi.responses import Response
        from engine.ingest import tonemap_plate_to_display
        from engine.plate_fix import preview_repair
        from engine.plate_qa import load_report
        if not STATE.get("work_dir"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        if method not in ("orig", "mean", "flow", "gain", "keep"):
            raise HTTPException(status_code=400, detail="method: orig | mean | flow | gain | keep")
        rp = load_report(Path(STATE["work_dir"]), plate)
        pattern = (rp or {}).get("pattern") or _plate_source(plate)
        if not pattern:
            raise HTTPException(status_code=404, detail=f"Brak plate'u {plate}")
        dec = {int(k): v for k, v in ((rp or {}).get("decisions") or {}).items()}
        bad = [f for f in ((rp or {}).get("flagged") or []) if dec.get(int(f)) != "keep"]
        try:
            img = preview_repair(pattern, _shot_frames(), frame, method, side=max(64, min(int(side), 2048)), flagged=bad)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        if img is None:
            raise HTTPException(status_code=404, detail=f"klatka {frame}: brak pliku")
        cs = str((STATE.get("manifest") or {}).get("colorspace", "ACES2065-1")).upper()
        linear = pattern.lower().endswith(".exr") and any(t in cs for t in ("ACES", "LINEAR", "RAW"))
        u8 = tonemap_plate_to_display(img) if linear else np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(u8), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


    @router.post("/api/plate/fix")
    @long_job("Naprawa plate'u: {plate}")
    async def plate_fix_run(req: PlateFixRequest):
        """Z26: write `<plate dir>_fix/` (repaired frames in the source encoding, the rest hard links, README) and switch
    the plate in project.json to it (manifest untouched). Without `confirmed` only the plan (sizes) is returned."""
        from engine.comfy_io import job_progress
        from engine.plate_fix import write_fixed_plate
        import time as _time
        from engine.plate_qa import _norm, load_report, save_decisions
        m = STATE.get("manifest") or {}
        current = (m.get("plates") or {}).get(req.plate)
        if not current:
            raise HTTPException(status_code=400, detail=f"Brak plate'u {req.plate} w projekcie")
        pattern = _plate_source(req.plate) or current      # a change of the repair starts from the source, never `_fix_fix`
        frames = _shot_frames()
        decisions = {int(k): v for k, v in req.decisions.items()}
        bad = [v for v in decisions.values() if v not in ("mean", "flow", "gain", "keep")]
        if bad:
            raise HTTPException(status_code=400, detail="metoda naprawy: mean | flow | gain | keep")
        repairs = {f: v for f, v in decisions.items() if v != "keep"}
        if not repairs and _norm(pattern) == _norm(current):
            raise HTTPException(status_code=400, detail="brak klatek do naprawy (same „zostaw”) – nic do zapisu")
        if not repairs:
            # every flag kept now: back to the source plate (nothing written; the `_fix` folder stays on disk)
            if not req.confirmed:
                return {"success": False, "detail": "powrót do oryginalnego plate'u wymaga potwierdzenia",
                        "plan": {"restore": pattern, "repair": {}, "hardlinks": 0, "bytes_written_estimate": 0, "eta_s": 0,
                                 "dir": str(Path(pattern).parent), "exists": True, "ours": True}}
            res: Dict[str, Any] = {"success": True, "pattern": pattern, "dir": str(Path(pattern).parent), "repaired": {},
                                   "hardlinks": 0, "copies": 0, "seconds": 0.0, "restored": True}
        else:
            cb = lambda label, d, t: job_progress(label, d, t)
            try:
                res = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: write_fixed_plate(pattern, frames, decisions, req.confirmed, cb))
            except (ValueError, FileNotFoundError, FileExistsError, OSError) as e:
                raise HTTPException(status_code=409, detail=str(e))
            if not res.get("success"):
                return {"success": False, "plan": res["plan"], "detail": res["detail"]}
        stale: List[str] = []
        proj = STATE.get("shot_project")
        if proj is not None:
            proj.set_plate(req.plate, res["pattern"])
            fx = proj.data.setdefault("plate_fix", {})
            if res.get("restored"):
                fx.pop(req.plate, None)
            else:
                fx[req.plate] = {"source": pattern, "pattern": res["pattern"], "at": _time.time(),
                                 "decisions": {str(k): v for k, v in decisions.items()}}
            # objects computed from this plate before the change are "do przeliczenia" (engine/stages.py, plate_changed)
            proj.data.setdefault("plate_changed", {})[req.plate] = {"at": _time.time(),
                                                                   "frames": sorted(int(f) for f in (repairs or decisions))}
            proj.save()
            STATE["manifest"] = proj.effective_manifest()
            STATE["project"] = proj.to_dict()
            # proxies / tiles of the changed frames were rendered from the old frame -> rendered again on demand
            for f in proj.data["plate_changed"][req.plate]["frames"]:
                for d in (req.plate, f"{req.plate}_aligned_preview"):
                    (Path(STATE["work_dir"]) / "proxy" / d / f"p_{int(f):04d}.jpg").unlink(missing_ok=True)
            STATE.pop("display_cache", None)
            from engine.stages import plate_stale_objects
            stale = plate_stale_objects(_objects(), Path(STATE["work_dir"]), req.plate)
        # the decisions sent (CLI / UI) become the report's decisions of that source
        rp = load_report(Path(STATE["work_dir"]), req.plate)
        if rp is not None and _norm(rp.get("pattern")) == _norm(pattern):
            flagged = {int(f) for f in rp.get("flagged") or []}
            save_decisions(Path(STATE["work_dir"]), req.plate, {str(f): v for f, v in decisions.items() if f in flagged})
        return {"success": True, "plate": req.plate, **{k: v for k, v in res.items() if k != "repaired"},
                "repaired": {str(k): v for k, v in res["repaired"].items()}, "stale_objects": stale}

    return router, {"_plate_gate": _plate_gate, "_require_plates_ok": _require_plates_ok}
