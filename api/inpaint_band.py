# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The band of Cleanup step C1 „Inpaint” (user decision 2026-11-11; engine/inpaint_band.py, engine/objband.inpaint_band).
The editor is static/bandpoints.js (owner „c1”) – the same endpoints as the Cleanup 3D patch band (/api/cleanup3d/band):

    GET    /api/cleanup/band/params    PUT the same    the band points of the cleaned thing (Ctrl+Z restores them)
    GET    /api/cleanup/band/{frame}   PUT the same    the points of a frame / add a point, set its key
    GET    /api/cleanup/band/{frame}/lines?grow_px&soft_px   preview lines (plate px), the palette's unsaved values;
                                       + limit_contours / cut: the edge of what the model draws (2026-11-13)
    DELETE /api/cleanup/band/{point}?frame=N   DELETE /api/cleanup/band?frame=N
    PUT    /api/cleanup/band_keys/{frame} {grow_px?, soft_px?}   a key of the two sliders in that frame (2026-11-15)
    DELETE /api/cleanup/band_keys/{frame}                        its key gone (the last one: its values stay as sliders)
    PUT    /api/cleanup/band_keys {band_keys}                    the whole key set back (Ctrl+Z)

The band is the cleaned thing's own (engine/cleanup.PER_TARGET): stored in <work>/inpaint/cleanup.json beside its model
and description. Built by `make(*, ...)` like the other api/ routers (nothing imported from app.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine import cleanup as cu
from engine import inpaint_band as IB

LINES_SIDE = 1600      # the preview lines are measured on a grid with this long side (plate px back in the answer)


class BandParamsRequest(BaseModel):              # PUT /api/cleanup/band/params (Ctrl+Z of a band edit)
    band_points: List[Dict[str, Any]] = []
    band_ref_out: Optional[float] = None
    band_ref_in: Optional[float] = None


class BandPointRequest(BaseModel):               # PUT /api/cleanup/band/{frame} (one band point, as the stages')
    id: Optional[str] = None                     # absent = a new point
    x: Optional[float] = None                    # position in this frame (full-frame px); absent = where it is now
    y: Optional[float] = None
    out: Optional[float] = None                  # yellow whisker (px): how far the patch reaches out of the object there
    in_: Optional[float] = Field(None, alias="in")   # ignored (outer side only), kept for the editor
    ang: Optional[float] = None                  # tilt of the whisker, degrees


class BandKeyRequest(BaseModel):                 # PUT /api/cleanup/band_keys/{frame}
    grow_px: Optional[float] = None              # absent = the keyed value in that frame
    soft_px: Optional[float] = None


class BandKeysRequest(BaseModel):                # PUT /api/cleanup/band_keys (Ctrl+Z)
    band_keys: Dict[str, Dict[str, float]] = {}


def make(*, STATE, plate_size, cleanup):
    """`cleanup` = the exports of api/cleanup.make (need, alpha_loaders)."""
    router = APIRouter()

    def _ctx():
        work, _manifest = cleanup["need"]()
        s = cu.load_settings(work)
        if not s.get("remove"):
            raise HTTPException(status_code=409, detail="Nic nie wybrano do czyszczenia – kliknij obiekt w palecie Obiekty")
        w, h = plate_size()
        rm, _keep = cleanup["alpha_loaders"](s, w, h)
        return work, s, (lambda k: [fn(int(k)) for fn in rm])

    def _save(work, s, params):
        pts = [dict(p) for p in (params.get("band_points") or [])]
        s.update(band_points=pts, band_ref_out=params.get("band_ref_out") if pts else None,
                 band_ref_in=params.get("band_ref_in") if pts else None)
        cu.save_settings(work, s)

    def _edit(fn):
        from engine.bandedit import BandEditError
        try:
            return fn()
        except BandEditError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    # ---- keys of the two band sliders (user decision 2026-11-15: a slider change = a key of the current frame, linear
    # between keys – like the band keys of the alpha stages). engine/cleanup.band_at reads them everywhere.
    def _keys_view(s):
        return {"band_keys": s.get("band_keys") or {}, "grow_px": s["grow_px"], "soft_px": s["soft_px"]}

    @router.put("/api/cleanup/band_keys/{frame}")
    def band_key_set(frame: int, req: BandKeyRequest):
        work, _m = cleanup["need"]()
        s = cu.load_settings(work)
        g, sf = cu.band_at(s, int(frame))
        keys = dict(s.get("band_keys") or {})
        keys[str(int(frame))] = {"grow_px": g if req.grow_px is None else float(req.grow_px),
                                 "soft_px": sf if req.soft_px is None else float(req.soft_px)}
        s["band_keys"] = keys
        return _keys_view(cu.save_settings(work, s))

    @router.delete("/api/cleanup/band_keys/{frame}")
    def band_key_delete(frame: int):
        work, _m = cleanup["need"]()
        s = cu.load_settings(work)
        keys = dict(s.get("band_keys") or {})
        k = keys.pop(str(int(frame)), None)
        if k is None:
            raise HTTPException(status_code=404, detail=f"F{frame}: brak klucza pasa")
        if not keys:                              # the last key: the band stays as it was there (no jump)
            s["grow_px"], s["soft_px"] = float(k["grow_px"]), float(k["soft_px"])
        s["band_keys"] = keys
        return _keys_view(cu.save_settings(work, s))

    @router.put("/api/cleanup/band_keys")
    def band_keys_put(req: BandKeysRequest):
        work, _m = cleanup["need"]()
        s = cu.load_settings(work)
        s["band_keys"] = req.band_keys
        return _keys_view(cu.save_settings(work, s))

    @router.get("/api/cleanup/band/params")
    def band_params():
        """The band points as stored (+ the slider's reference) – what Ctrl+Z restores."""
        s = cu.load_settings(cleanup["need"]()[0])
        return {k: s.get(k) for k in ("band_points", "band_ref_out", "band_ref_in")}

    @router.put("/api/cleanup/band/params")
    def band_params_put(req: BandParamsRequest):
        work = cleanup["need"]()[0]
        _save(work, cu.load_settings(work), {"band_points": req.band_points, "band_ref_out": req.band_ref_out,
                                             "band_ref_in": req.band_ref_in})
        return {"success": True}

    @router.get("/api/cleanup/band/{frame}")
    def band_get(frame: int):
        """The band points in `frame` as the editor shows them (carried by the removed objects, snapped to their outline,
        widths scaled by „Pas na zewnątrz”)."""
        _work, s, ra = _ctx()
        return _edit(lambda: IB.state(s, int(frame), ra))

    @router.put("/api/cleanup/band/{frame}")
    def band_put(frame: int, req: BandPointRequest):
        """Add a band point in `frame` or set its key there (position / outer width); the mask and the run use it."""
        from engine.bandedit import put_point
        from engine.bandmap import band_points
        work, s, ra = _ctx()
        params = {"band_points": s.get("band_points") or [], "band_ref_out": s.get("band_ref_out"),
                  "band_ref_in": s.get("band_ref_in")}
        if not params.get("band_ref_out"):       # the slider as it is now = scale 1 for these points
            params.update(band_ref_out=float(s["grow_px"]) if s["grow_px"] > 0 else None, band_ref_in=None)
        b = IB.band({**s, **params})
        st = _edit(lambda: IB.state(s, int(frame), ra, b))
        so, si = b.scales
        pts, pid = _edit(lambda: put_point(band_points(params), int(frame), {q["id"]: q for q in st["points"]},
                                           st["defaults"], so, si, id=req.id, x=req.x, y=req.y, out=req.out,
                                           in_=req.in_, ang=req.ang, max_angle=IB.FULL_TURN_DEG))
        _save(work, s, {**params, "band_points": pts})
        s = cu.load_settings(work)
        return {"success": True, "point": pid, "band": _edit(lambda: IB.state(s, int(frame), ra))}

    @router.get("/api/cleanup/band/{frame}/lines")
    def band_lines(frame: int, grow_px: Optional[float] = None, soft_px: Optional[float] = None):
        """Preview lines of the C1 band in `frame` (plate px): the band ring (objects' outline + where the solid patch
        ends) and where the soft edge ends – the geometry of the mask itself (engine/cleanup.removal_mask)."""
        _work, s, ra = _ctx()
        s = cu.at_override(s, grow_px, soft_px, int(frame))      # unsaved values win in this frame; else its keys
        g, sf = cu.band_at(s, int(frame))
        pf = IB.points_fn(s, ra)
        al = ra(int(frame))
        if not al:
            raise HTTPException(status_code=409, detail=f"F{frame}: brak maski czyszczonego obiektu")
        h, w = al[0].shape[:2]
        scale = min(1.0, LINES_SIDE / float(max(w, h)))
        # the edge of what the model draws (generation margin + the engine's grow, user 2026-11-13): limit_contours / cut
        lim = float(s["gen_margin_px"]) + cu.engine_reach_px(s.get("model"), s, w, h) if s.get("model") in cu.C1_MODELS else None
        return _edit(lambda: IB.lines(al, g, sf, pf(int(frame)) if pf else None, scale, lim))

    @router.delete("/api/cleanup/band/{point_id}")
    def band_delete_point(point_id: str, frame: Optional[int] = None):
        """Remove a band point, or only its key in `frame` (?frame=N; the last key removes the point)."""
        from engine.bandedit import delete_point
        work = cleanup["need"]()[0]
        s = cu.load_settings(work)
        pts = _edit(lambda: delete_point(s.get("band_points") or [], point_id, frame))
        _save(work, s, {"band_points": pts, "band_ref_out": s.get("band_ref_out"), "band_ref_in": s.get("band_ref_in")})
        return {"success": True}

    @router.delete("/api/cleanup/band")
    def band_delete_keys(frame: Optional[int] = None):
        """The band keys of ALL points in `frame` (?frame=N), or every point (no frame = the plain band again)."""
        from engine.bandedit import delete_keys
        work = cleanup["need"]()[0]
        s = cu.load_settings(work)
        pts, removed = _edit(lambda: delete_keys(s.get("band_points") or [], frame))
        _save(work, s, {"band_points": pts, "band_ref_out": s.get("band_ref_out"), "band_ref_in": s.get("band_ref_in")})
        return {"success": True, "removed": removed, "points": [p["id"] for p in pts]}

    return router, {}
