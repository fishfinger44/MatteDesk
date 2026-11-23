# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The band of Cleanup step C1 „Inpaint” (user decision 2026-11-11; engine/inpaint_band.py, engine/objband.inpaint_band).
The editor is static/bandpoints.js (owner „c1”) – the same endpoints as the Cleanup 3D patch band (/api/cleanup3d/band):

    GET    /api/cleanup/band/params    PUT the same    the band points of the cleaned thing (Ctrl+Z restores them)
    GET    /api/cleanup/band/{frame}   PUT the same    the points of a frame / add a point, set its key
    GET    /api/cleanup/band/{frame}/lines?grow_px&soft_px   preview lines (plate px), the palette's unsaved values;
                                       + limit_contours / cut: the edge of what the model draws (2026-11-13)
    DELETE /api/cleanup/band/{point}?frame=N   DELETE /api/cleanup/band?frame=N
    POST   /api/cleanup/band/export?fmt=nk|jsx&what=band|tracker  the band over the work range as a Nuke Roto / After
                                       Effects masks, or its tracker (band points + mask centre: Tracker4 / nulls)
                                       (engine/band_export, user request 2026-11-23) -> <work>/inpaint/band_export/
    PUT    /api/cleanup/band_keys/{frame} {grow_px?, soft_px?}   a key of the two sliders in that frame (2026-11-15)
    DELETE /api/cleanup/band_keys/{frame}                        its key gone (the last one: its values stay as sliders)
    PUT    /api/cleanup/band_keys {band_keys}                    the whole key set back (Ctrl+Z)

The band is the cleaned thing's own (engine/cleanup.PER_TARGET): stored in <work>/inpaint/cleanup.json beside its model
and description. Built by `make(*, ...)` like the other api/ routers (nothing imported from app.py).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine import cleanup as cu
from engine import inpaint_band as IB
from engine.lru import LockedLRU

LINES_SIDE = 1600      # the preview lines are measured on a grid with this long side (plate px back in the answer)

# The removal masks at the lines grid (user 2026-11-23: the patch band preview took 4-8 s on SD160). A MiniMax-H3 frame
# needs its whole latent token +-2 frames (~9 masks); one 6K group frame = 0.9 s (two EXR + seamless_union) and the
# group cache (engine/groups._CACHE, 16 full frames) was evicted by the next frame. Kept here as uint8 (ceil, so
# > 0.02 still holds) at ~1.7 MB a frame, keyed by the loader's stamp (cleanup_hold.Hold.key) + the held source;
# the rest of the work range is read in the background after the first request (`_warm`).
_SMALL = LockedLRU(240)
_WARMING: set = set()
_WARM_LOCK = threading.Lock()


def _small_size(w: int, h: int) -> Tuple[int, int]:
    sc = min(1.0, LINES_SIDE / float(max(w, h)))
    return max(1, int(round(w * sc))), max(1, int(round(h * sc)))


def _small_key(fn: Any, k: int, size: Tuple[int, int]) -> Optional[tuple]:
    key = getattr(fn, "key", None)
    if key is None:
        return None
    try:
        p = fn.cached()
    except Exception:  # noqa: BLE001 - no plan: the loader's own alpha
        p = None
    return (key, int(k), (p or {}).get("held", {}).get(int(k)), size)


def small_alpha(fn: Any, k: int, size: Tuple[int, int]) -> np.ndarray:
    """float32 0..1 at `size`: the removal alpha of frame k, from `_SMALL` when its loader has a stamp."""
    ck = _small_key(fn, k, size)
    hit = _SMALL.get(ck) if ck is not None else None
    if hit is None:
        a = np.asarray(fn(int(k)), np.float32)
        if (a.shape[1], a.shape[0]) != size:
            a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        hit = np.ceil(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        if ck is not None:
            _SMALL.put(ck, hit)
    return hit.astype(np.float32) / 255.0


def _warm(fns: List[Any], frames: List[int], size: Tuple[int, int]) -> None:
    """Read the rest of the work range into `_SMALL` in one background thread per mask stamp; it stops while a
    heavy job runs (that job owns the disk and the CPU)."""
    from engine.jobs import JOBS
    todo = [fn for fn in fns if getattr(fn, "key", None) is not None]
    if not todo:
        return
    tag = tuple(fn.key for fn in todo) + (size,)
    with _WARM_LOCK:
        if tag in _WARMING:
            return
        _WARMING.add(tag)

    def run():
        try:
            for f in frames:
                if JOBS.running():
                    return
                for fn in todo:
                    ck = _small_key(fn, f, size)
                    if ck is not None and _SMALL.get(ck) is None:
                        small_alpha(fn, f, size)
        except Exception:  # noqa: BLE001 - a warm-up never fails the preview
            pass
        finally:
            with _WARM_LOCK:
                _WARMING.discard(tag)
    threading.Thread(target=run, name="c1-band-warm", daemon=True).start()


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


def make(*, STATE, plate_size, cleanup, long_job=None):
    """`cleanup` = the exports of api/cleanup.make (need, alpha_loaders); `long_job` = app.long_job (band export)."""
    router = APIRouter()

    def _loaders():
        work, _manifest = cleanup["need"]()
        s = cu.load_settings(work)
        if not s.get("remove"):
            raise HTTPException(status_code=409, detail="Nic nie wybrano do czyszczenia – kliknij obiekt w palecie Obiekty")
        w, h = plate_size()
        rm, _keep = cleanup["alpha_loaders"](s, w, h)
        return work, s, rm

    def _ctx():
        work, s, rm = _loaders()
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
        _work, s, rm = _loaders()
        ra = (lambda k: [fn(int(k)) for fn in rm])
        s = cu.at_override(s, grow_px, soft_px, int(frame))      # unsaved values win in this frame; else its keys
        g, sf = cu.band_at(s, int(frame))
        pf = IB.points_fn(s, ra)
        if not rm:
            raise HTTPException(status_code=409, detail=f"F{frame}: brak maski czyszczonego obiektu")
        w, h = (int(v) for v in plate_size())
        size = _small_size(w, h)
        scale = size[0] / float(w)
        sal = (lambda k: [small_alpha(fn, int(k), size) for fn in rm])   # the lines grid, cached (2026-11-23)
        al = sal(int(frame))
        # the edge of what the model draws (generation margin + the engine's grow, user 2026-11-13): limit_contours / cut
        lim_m = float(s["gen_margin_px"]) + cu.engine_reach_px(s.get("model"), s, w, h) if s.get("model") in cu.C1_MODELS else None
        # 2026-11-23 (`gen_band`): the model's mask covers the band itself – no limit line, nothing cut (the mask
        # preview's light-blue area = what the model draws, band included)
        lim = None if s.get("gen_band") else lim_m
        # the engine's mask of this frame (user 2026-11-16): ±2 frames, MiniMax-H3 its whole latent token
        wf = list(cleanup["work_frames"]())
        mf = cu.model_frames(s.get("model"), int(frame), wf) if lim_m is not None else [int(frame)]
        lal = [a for k in mf for a in sal(k)] if mf != [int(frame)] else None
        L = _edit(lambda: IB.lines(al, g, sf, pf(int(frame)) if pf else None, scale, lim, limit_alphas=lal,
                                   full_size=(w, h)))
        _warm(rm, wf, size)   # the next frames of the range come from the cache
        if lim_m is not None:
            L["model_frames"] = [mf[0], mf[-1]]
            if s.get("model") == "h3":
                from engine.h3_anchors import h3_token, h3_token_span
                wf = cleanup["work_frames"]()
                if int(frame) in wf:
                    L["h3_token"] = h3_token_span(h3_token(wf.index(int(frame))), wf)
        return L

    if long_job is not None:
        @router.post("/api/cleanup/band/export")
        @long_job("Eksport pasa łaty")
        def band_export(fmt: str = "nk", what: str = "band") -> Dict[str, Any]:
            """The C1 patch band (band ring + where the soft edge ends) of every frame of the work range as a Nuke Roto
            (clipboard / .nk) or After Effects masks (.jsx) – the same lines as the preview, engine/band_export."""
            import re
            from engine import band_export as BE
            from engine.jobs import JOBS
            if fmt not in ("nk", "jsx"):
                raise HTTPException(status_code=400, detail="fmt: nk | jsx")
            if what not in ("band", "tracker"):
                raise HTTPException(status_code=400, detail="what: band | tracker")
            work, s, _rm = _loaders()
            _w, s_, ra_ = _ctx() if what == "tracker" else (None, None, None)
            pts: Dict[int, List[Dict[str, Any]]] = {}
            frames = list(cleanup["work_frames"]())
            from api.band_export import per_frame

            def one(f: int):                          # frames side by side (api/band_export.per_frame, 2026-11-23)
                try:
                    d = band_lines(int(f))
                except HTTPException:
                    return None                       # no mask of the cleaned thing in this frame
                p = _edit(lambda: IB.state(s_, int(f), ra_)).get("points") or [] if what == "tracker" else None
                return (d if d.get("band_contours") else None, p)

            res = per_frame(frames, one, "linie pasa łaty")
            lines: Dict[int, Dict[str, Any]] = {f: r[0] for f, r in res.items() if r[0] is not None}
            pts.update({f: r[1] for f, r in res.items() if r[1] is not None})
            JOBS.progress("kształty", len(frames), len(frames))
            tgt = re.sub(r"\W", "_", "_".join(str(x) for x in (s.get("remove") or []))) or "x"
            if what == "tracker":
                from api.band_export import write_tracker
                from engine import track_export as TE
                trk = TE.tracks(pts, lines)
                if not trk:
                    raise HTTPException(status_code=409, detail="Nic do śledzenia – pas łaty pusty w obszarze roboczym")
                out = work / "inpaint" / "band_export"
                out.mkdir(parents=True, exist_ok=True)
                w, h = (int(v) for v in plate_size())
                fr = sorted(lines) or frames
                return write_tracker(out, f"TRK_laty_{tgt}"[:80],
                                     f"Tracker pasa łaty · {', '.join(str(x) for x in (s.get('remove') or []))} · F{fr[0]}–{fr[-1]}",
                                     trk, w, h, frames[0], fmt)
            shp = BE.shapes(lines) if lines else []
            if not shp:
                raise HTTPException(status_code=409, detail="Pas łaty jest pusty w obszarze roboczym")
            w, h = (int(v) for v in plate_size())
            node = f"PAS_laty_{tgt}"[:80]
            fr = sorted(lines)
            label = f"Pas łaty · {', '.join(str(x) for x in (s.get('remove') or []))} · F{fr[0]}–{fr[-1]}"
            out = work / "inpaint" / "band_export"
            out.mkdir(parents=True, exist_ok=True)
            nk, jsx = out / f"{node}.nk", out / f"{node}.jsx"
            nk_text = BE.nk_text(shp, w, h, node, label)
            jsx_text = BE.jsx_text(shp, w, h, frames[0], node, label)
            nk.write_text(nk_text, "utf-8", newline="\n")
            jsx.write_text(jsx_text, "utf-8", newline="\n")
            return {"text": nk_text if fmt == "nk" else jsx_text, "path": str(nk if fmt == "nk" else jsx).replace("\\", "/"),
                    "nk": str(nk).replace("\\", "/"), "jsx": str(jsx).replace("\\", "/"), "node": node,
                    "frames": [fr[0], fr[-1]], "shapes": BE.summary(shp)}

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
