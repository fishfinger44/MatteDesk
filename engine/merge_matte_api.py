# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP routes of the Final Touch palette „MergeMatte” (PLAN_FINAL_TOUCH.md §11): the object's MergeMatte settings and
a preview of one frame region on the Comp view background. The computation is engine/merge_matte.py (the port of the
Nuke group); app.py only provides its helpers through `make_router(...)` (no import of app here).

    GET  /api/object/{name}/merge_matte                  settings + knob table + presets
    PUT  /api/object/{name}/merge_matte                  {on?, knobs?, reset?: "default" | "sh010"}
    GET  /api/object/{name}/merge_matte/frame/{frame}    PNG of plate region x, y, w, h over bg (= "#rrggbb" | "checker" |
                                                         "bg" = the target background plate); b = Final Touch base version
    GET  /api/object/{name}/merge_matte/nk               the object's MergeMatte group as Nuke clipboard text (Ctrl+V in the
                                                         node graph), also written to objects/<o>/final_touch/MergeMatte_<o>.nk
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from engine import final_touch as ft
from engine import merge_matte as mm
from engine import merge_matte_nk as mmnk


class MergeMatteRequest(BaseModel):
    on: Optional[bool] = None
    knobs: Optional[Dict[str, float]] = None
    reset: Optional[str] = None


def specs() -> list:
    """The knob table for the palette (same order, labels and ranges as the Nuke group)."""
    return [{"name": n, "label": lab, "range": (list(r) if isinstance(r, tuple) else r), "default": v, "tip": tip,
             **({"field": list(mmnk.KNOB_FIELD_RANGES[n])} if n in mmnk.KNOB_FIELD_RANGES else {})}
            for n, lab, r, v, tip in mmnk.KNOB_SPECS]


def make_router(objects: Callable[[], Any], require_stage: Callable[[str, str], None], ft_base: Callable[[str], Dict[str, Any]],
                ft_alpha: Callable[..., np.ndarray], plate_size: Callable[[], tuple], display_u8: Callable[[str, int], np.ndarray],
                has_project: Callable[[], bool]) -> APIRouter:
    """objects() = the object store; ft_alpha(obj_dir, base_run, frame, w, h) = full-res Final Touch alpha (what the export
    writes); display_u8(plate, frame) = full-res display BGR uint8 of a plate frame."""
    r = APIRouter()

    def _need_project():
        if not has_project():
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    @r.get("/api/object/{name}/merge_matte")
    def get_merge_matte(name: str):
        _need_project()
        d = ft.load(objects().dir(name))
        return {**ft.merge_matte_settings(d), "specs": specs(), "presets": {"default": mmnk.KNOB_DEFAULTS, "sh010": mmnk.PRESET_SH010},
                "approx": ["choke"]}

    @r.put("/api/object/{name}/merge_matte")
    def put_merge_matte(name: str, req: MergeMatteRequest):
        _need_project()
        require_stage(name, "final_touch")
        try:
            return ft.set_merge_matte(objects().dir(name), on=req.on, knobs=req.knobs, reset=req.reset)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @r.get("/api/object/{name}/merge_matte/nk")
    def merge_matte_nk(name: str):
        """The palette's „Nuke” button (user request 2026-10-30): the group with the object's knobs, pasteable in Nuke."""
        _need_project()
        od = objects().dir(name)
        knobs = ft.merge_matte_settings(ft.load(od))["knobs"]
        node = "MergeMatte_" + (re.sub(r"\W", "_", name) or "obj")   # a valid Nuke node name
        text = mmnk.paste_text(knobs, name=node)
        path = od / "final_touch" / f"{node}.nk"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8", newline="\n")
        return {"text": text, "path": str(path).replace("\\", "/"), "node": node}

    @r.get("/api/object/{name}/merge_matte/frame/{frame}")
    def merge_matte_frame(name: str, frame: int, x: int = 0, y: int = 0, w: int = 0, h: int = 0, bg: str = "#1b2a4a",
                          b: Optional[str] = None, max_px: int = 2_500_000, k: Optional[str] = None):
        """k = the settings hash of the client (cache buster only); the knobs are read from final_touch.json."""
        _need_project()
        od = objects().dir(name)
        if b is None:
            b = ft_base(name)["run"]
        if not b:
            raise HTTPException(status_code=404, detail="Brak bazy Final Touch – najpierw policzony etap")
        W, H = plate_size()
        alpha = ft_alpha(od, b, frame, W, H)
        plate = display_u8("beauty", frame)[..., ::-1].astype(np.float32) / 255.0
        if plate.shape[:2] != alpha.shape[:2]:
            raise HTTPException(status_code=409, detail="Rozmiar plate'u ≠ rozmiar alfy")
        if bg == "checker":
            bgv = mm.checker_srgb(H, W)
        elif bg == "bg":
            bgv = display_u8("bg", frame)[..., ::-1].astype(np.float32) / 255.0
        else:
            try:
                bgv = mm.hex_srgb(bg)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Złe tło: {bg}")
        rect = (x, y, w or W, h or H)
        knobs = ft.merge_matte_settings(ft.load(od))["knobs"]
        img, s = mm.preview_region(plate, alpha, bgv, knobs, rect, max_px=max(100_000, min(int(max_px), 9_000_000)))
        ok, buf = cv2.imencode(".png", img[..., ::-1])
        return Response(content=buf.tobytes(), media_type="image/png", headers={"X-MM-Scale": f"{s:.6f}", "Cache-Control": "no-store"})

    return r
