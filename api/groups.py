# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Groups (PLAN_GROUP_COMPOSE, 2026-10-29)

Routes of app.py's section „Groups (PLAN_GROUP_COMPOSE, 2026-10-29)” (audit 2026-11-02 step 8): the same functions, built by `make(...)` with the
app's shared pieces passed in explicitly (nothing is imported from app.py – no cycle); app.py includes the router
where the section used to be, so the route order is unchanged. `exports` = the section's helpers app.py still calls.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
import cv2
from fastapi import HTTPException
from pydantic import BaseModel


class GroupCreateRequest(BaseModel):
    name: str
    members: List[str] = []
    seam_px: Optional[int] = None


class GroupMemberRequest(BaseModel):
    name: str


class GroupSourceRequest(BaseModel):
    source: str
    run: Optional[str] = None


class GroupSettingsRequest(BaseModel):
    seam_px: Optional[int] = None


class GroupMemberOnRequest(BaseModel):
    on: bool


class GroupMemberSeamRequest(BaseModel):
    seam_px: Optional[int] = None    # None = the group's seam


def make(*, _alpha_u8, _objects, _plate_size, _png_response, _proxy_scale, _run_edge_engine, when_idle):
    """Build the router. Keyword-only: exactly what the section takes from app.py (7 names).
    Returns (router, exports) – exports: _group_payload."""
    router = APIRouter()

    # ----------------------------------------------------------------------------- groups (PLAN_GROUP_COMPOSE, 2026-10-29)
    def _group_or_404(group: str) -> Dict[str, Any]:
        from engine.groups import get_group
        try:
            return get_group(_objects(), group)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))


    def _mask_info(store, m: str) -> Dict[str, Any]:
        """What object `m` brings (groups.member_run: source, run, valid, why) + what it can pick (only computed results,
        ↻ when stale) – a group member's „Do grupy z:” and an object's own mask outside a group (2026-11-15)."""
        from engine import groups
        from engine.stages import _run_json, stage_run
        avail = [{"source": "core", "run": "core", "valid": True}]
        for src in ("matanyone", "videomama", "vitmatte", "ltx"):
            r, ok, _ = stage_run(store, m, src)
            if r:
                e = {"source": src, "run": r, "valid": ok}
                if src == "vitmatte":   # stage 5 = ViTMatte or BiRefNet: the engine of THIS result names it in the picker
                    e["engine"] = _run_edge_engine(_run_json(store.dir(m), r)) or "vitmatte"
                avail.append(e)
        r, ok, _ = groups.ft_run(store, m)   # Final Touch: its base is chosen on stage 8 (user decision 2026-10-29)
        if r:
            avail.append({"source": "final_touch", "run": r, "valid": ok})
        return {**groups.member_run(store, m), "available": avail, "own": groups.has_own_source(store.get(m))}

    def _group_payload(group: str) -> Dict[str, Any]:
        """The group as the palette shows it: members (depth order) with what each brings, the sources it can pick
    („Do grupy z:” – only computed ones, ↻ when stale) and whether it is switched on, seam, depth place, whole results."""
        from engine import groups
        store = _objects()
        g = groups.get_group(store, group)
        ms = []
        for m in groups.members(store, group):
            o = store.get(m)
            ms.append({**_mask_info(store, m), "depth_order": o.get("depth_order", 0),
                       "on": groups.is_on(o), "seam_px": groups.member_seam(store, group, m),
                       "seam_own": o.get("group_seam_px") is not None})
        return {"name": group, "kind": "group", "members": ms, "seam_px": groups.seam(store, group),
                "depth_order": groups.group_depth(store, group), "whole_results": groups.whole_results(store, group),
                "stage_runs": (g.get("tool") or {}).get("stage_runs") or {}}


    @router.post("/api/groups")
    @when_idle("Nowa grupa")
    def create_group_ep(req: GroupCreateRequest):
        """New group entity (palette Obiekty): optionally with its first members (each joins with „Do grupy z:” = Core)."""
        from engine.export import SEAM_PX_DEFAULT
        from engine.groups import create_group
        try:
            create_group(_objects(), req.name, req.members, seam_px=SEAM_PX_DEFAULT if req.seam_px is None else req.seam_px)
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "group": _group_payload(req.name)}


    @router.post("/api/group/{group}/members")
    @when_idle("Obiekt do grupy")
    def add_group_member(group: str, req: GroupMemberRequest):
        """Drop an object into the group (palette): it joins with Core; the whole recompute becomes stale (↻)."""
        from engine.groups import add_member
        _group_or_404(group)
        try:
            add_member(_objects(), group, req.name)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "group": _group_payload(group)}


    @router.delete("/api/group/{group}/members/{member}")
    @when_idle("Obiekt z grupy")
    def remove_group_member(group: str, member: str, confirm: bool = False):
        """Detach a member. With a whole result (stages on the whole / Final Touch) and without ?confirm=1: 409 with the
    warning (needs_confirm, whole_stages, bytes) – nothing changed; with it the whole results are deleted for good."""
        from engine.groups import remove_member
        _group_or_404(group)
        try:
            r = remove_member(_objects(), group, member, confirm=confirm)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if r.get("needs_confirm"):
            raise HTTPException(status_code=409, detail=r)
        return {**r, "group": _group_payload(group)}


    @router.put("/api/group/{group}/members/{member}/source")
    @when_idle("Źródło członka grupy")
    def set_group_member_source(group: str, member: str, req: GroupSourceRequest):
        """„Do grupy z:” of a member (core | matanyone | videomama | vitmatte | ltx | version | final_touch)."""
        from engine.groups import set_member_source
        _group_or_404(group)
        try:
            set_member_source(_objects(), group, member, {"source": req.source, "run": req.run})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "group": _group_payload(group)}


    @router.put("/api/object/{name}/mask_source")
    @when_idle("Maska obiektu")
    def set_object_mask_source(name: str, req: GroupSourceRequest):
        """The mask of an object outside a group (user 2026-11-15) – the same choice as „Do grupy z:”: the default when
        it joins a group and in Cleanup (removed / protected)."""
        from engine.groups import set_object_source
        try:
            set_object_source(_objects(), name, {"source": req.source, "run": req.run})
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "mask_info": _mask_info(_objects(), name)}

    @router.put("/api/group/{group}")
    @when_idle("Ustawienia grupy")
    def put_group(group: str, req: GroupSettingsRequest):
        """Seam of the group (↻ on the whole recompute)."""
        from engine.groups import set_seam
        _group_or_404(group)
        if req.seam_px is not None:
            set_seam(_objects(), group, req.seam_px)
        return {"success": True, "group": _group_payload(group)}


    @router.put("/api/group/{group}/members/{member}/seam")
    @when_idle("Szew członka grupy")
    def set_group_member_seam(group: str, member: str, req: GroupMemberSeamRequest):
        """A member's own seam reach (user decision 2026-10-29: hair small, a dress large); null = the group's seam.
    Part of the group signature – ↻ on the whole recompute."""
        from engine.groups import set_member_seam
        _group_or_404(group)
        try:
            set_member_seam(_objects(), group, member, req.seam_px)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "group": _group_payload(group)}


    @router.put("/api/group/{group}/members/{member}/on")
    @when_idle("Członek grupy wł. / wył.")
    def set_group_member_on(group: str, member: str, req: GroupMemberOnRequest):
        """Palette checkbox of a member (user decision 2026-10-29): switched off = out of the group's composition (alpha,
    preview, whole recompute input – ↻), still a member."""
        from engine.groups import set_member_on
        _group_or_404(group)
        try:
            set_member_on(_objects(), group, member, req.on)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"success": True, "group": _group_payload(group)}


    @router.delete("/api/group/{group}")
    @when_idle("Rozgrupowanie")
    def delete_group(group: str, confirm: bool = False):
        """Ungroup: members become standalone (results untouched), the group's folder is deleted for good; with whole
    results and without ?confirm=1 -> 409 with the warning."""
        from engine.groups import ungroup
        _group_or_404(group)
        r = ungroup(_objects(), group, confirm=confirm)
        if r.get("needs_confirm"):
            raise HTTPException(status_code=409, detail=r)
        return r


    @router.get("/api/group/{group}")
    def group_info(group: str):
        """The group entity: members in depth order with what each brings („Do grupy z:” – source, version, current or ↻,
    the reason when there is nothing), the seam and the group's place in the depth order (its front-most member)."""
        from engine import groups
        g = _group_or_404(group)
        store = _objects()
        ms = [{**groups.member_run(store, m), "depth_order": store.get(m).get("depth_order", 0)}
              for m in groups.members(store, group)]
        return {"name": group, "members": ms, "seam_px": groups.seam(store, group),
                "depth_order": groups.group_depth(store, group), "stage_runs": (g.get("tool") or {}).get("stage_runs") or {}}


    @router.get("/api/group/{group}/frame/{frame}")
    def group_frame(group: str, frame: int):
        """Proxy PNG of the basic group matte (the composition of the members' chosen alphas with the seams, nothing
    computed; objects in front not subtracted). 409 + `missing` when a member has nothing to bring."""
        from engine.groups import GroupSourceMissing, group_alpha_loader
        _group_or_404(group)
        w, h = _plate_size()
        sc = _proxy_scale()
        pw, ph = max(1, round(w * sc)), max(1, round(h * sc))
        try:
            a = group_alpha_loader(_objects(), group, w, h)(int(frame))
        except GroupSourceMissing as e:
            raise HTTPException(status_code=409, detail={"message": str(e), "missing": e.missing})
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return _png_response(_alpha_u8(cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)))

    return router, {"_group_payload": _group_payload, "_mask_info": _mask_info}
