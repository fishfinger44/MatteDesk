# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate colour profile (stage 0, user decision 2026-11-03)

    GET /api/project/color_profile    setting (auto | curve id), the effective curve with its reason, the proposal, options
    PUT /api/project/color_profile    {"color_profile": "auto" | curve id} → project.json only (the manifest is never written)

Built by `make(*, STATE)` like the other api/ routers (nothing imported from app.py); `exports["_project_curve"]()` =
{"id", "label", "reason"} of the curve in use (None without a project) – what MergeMatte's `log_curve` = `project` and
the .nk export resolve to (engine/color_profile.py, engine/log_curves.py).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine import color_profile as cp


class ColorProfileRequest(BaseModel):
    color_profile: str = "auto"


def make(*, STATE):
    """Build the router. Keyword-only: the app state (1 name). Returns (router, exports) – exports: _project_curve."""
    router = APIRouter()

    def _state() -> Optional[Dict[str, Any]]:
        proj = STATE.get("shot_project")
        if not proj:
            return None
        manifest = STATE.get("manifest") or {}
        header = cp.read_header(cp.sample_file(manifest))
        r = cp.resolve(cp.project_setting(proj.data), manifest, header)
        return r

    def _project_curve() -> Optional[Dict[str, Any]]:
        r = _state()
        return {"id": r["id"], "label": r["label"], "reason": r["reason"]} if r else None

    def _payload(r: Dict[str, Any]) -> Dict[str, Any]:
        from engine import log_curves as lc
        auto = r["auto"]
        options = [{"id": cp.AUTO, "label": f"auto (→ {lc.LOG_CURVES[auto['id']].label}, {auto['reason']})"}] + \
                  [{"id": c.id, "label": c.label, "note": c.note} for c in lc.LOG_CURVES.values()]
        return {"setting": r["setting"], "effective": {"id": r["id"], "label": r["label"], "reason": r["reason"]},
                "proposal": auto, "options": options}

    @router.get("/api/project/color_profile")
    def get_color_profile():
        """The plate colour profile of the shot: project.json `color_profile` (auto | curve id), the curve in use with its reason."""
        r = _state()
        if not r:
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        return _payload(r)

    @router.put("/api/project/color_profile")
    def put_color_profile(req: ColorProfileRequest):
        """Stored in project.json -> color_profile (never in the shot manifest); `auto` removes the override."""
        proj = STATE.get("shot_project")
        if not proj:
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        try:
            v = cp.validate(req.color_profile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if v == cp.AUTO:
            proj.data.pop(cp.KEY, None)
        else:
            proj.data[cp.KEY] = v
        proj.save()
        return _payload(_state())

    return router, {"_project_curve": _project_curve}
