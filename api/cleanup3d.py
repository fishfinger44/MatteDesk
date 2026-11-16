# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D endpoints (docs/PLAN_CLEANUP_3D.md; package P0 = the contract, package P8 = the bodies).

Steps of the Cleanup mode: C1 remove (the MVP's /api/cleanup routes) · C2 camera · C3 cards · C4 keys · C5 project · C6 export.

    GET    /api/cleanup3d                                  state of the mode: done / stale per step, camera report, cards
                                                           with their keys, engines and the solve with availability
    GET    /api/cleanup3d/camera                           the solved camera (cams.json)
    POST   /api/cleanup3d/camera/solve                     heavy job: COLMAP + CoTracker3 + bundle adjustment -> camera
    POST   /api/cleanup3d/cards/propose                    cards proposed from the solve (ground, sea, sky), merged
    GET    /api/cleanup3d/cards     PUT /api/cleanup3d/cards
    PUT    /api/cleanup3d/cards/{card}/extent              areas (curves + brush layers) drawn in one frame
    GET    /api/cleanup3d/cards/{card}/extent/{area}/png   a brush layer of the extent (8-bit, white = area)
    GET    /api/cleanup3d/depth_sources                    depth maps for a depth card (DepthCrafter result, depth plate)
    POST   /api/cleanup3d/cards/{card}/depth               P18: the card becomes a depth card (snapshot of the reference
                                                           frame's depth, fitted to the camera)
    GET    /api/cleanup3d/keys      POST /api/cleanup3d/keys/{card}      DELETE /api/cleanup3d/keys/{card}/{frame}
    POST   /api/cleanup3d/keys/{card}/auto                 heavy job (P13): key frames where the projection error grows
    POST   /api/cleanup3d/keys/{card}/master               heavy job (P13): ONE master key accumulated from all frames
    POST   /api/cleanup3d/keys/{card}/seedvr2              heavy job (P12): SeedVR2 on the card's keys as one clip
    GET    /api/cleanup3d/keys/{card}/{frame}/image        PNG preview of a version
    POST   /api/cleanup3d/keys/{card}/{frame}/export | import | undo
    PUT    /api/cleanup3d/keys/{card}/{frame}/active       make a version active (the C4 version list)
    DELETE /api/cleanup3d/keys/{card}/{frame}/versions/{version}   one version deleted for good (not the active one)
    GET    /api/cleanup3d/keys/{card}/{frame}/areas        PUT the same (whole list)
    GET    /api/cleanup3d/keys/{card}/{frame}/areas/{area}/png
    POST   /api/cleanup3d/keys/{card}/{frame}/regen        heavy job: LaMa / Qwen-Image-2.1 / FLUX 3 Image Klein + LoRA Restore
                                                           from a base version, on the marked area or the whole patch
    POST   /api/cleanup3d/void                             heavy job (P12): VOID once over the work range (+ keys)
    POST   /api/cleanup3d/void/keys                        heavy job (P12): key versions from the last VOID run
    POST   /api/cleanup3d/mark/{frame}                     areas drawn on any frame -> the keys that frame is made of
    POST   /api/cleanup3d/render                           heavy job: projection + comp -> <work>/inpaint/v###/, plate „cleaned”
    GET    /api/cleanup3d/view/{kind}/{frame}              PNG preview: key (needs card) | proj | cards
    POST   /api/cleanup3d/export_nuke                      C6: Nuke script (.nk + .chan + STMaps) -> <work>/inpaint/nuke/
    GET    /api/cleanup3d/blend/{frame}?part=fill|maps|band  inputs of the C5 live patch blend (band = the band's widths)
    GET    /api/cleanup3d/band/params   PUT the same       patch band points (P14, owner „c3d” of static/bandpoints.js)
    GET    /api/cleanup3d/band/{frame}  PUT the same       the points of a frame / add a point, set its key
    GET    /api/cleanup3d/band/{frame}/lines               preview lines of the patch band (plate px)
    DELETE /api/cleanup3d/band/{point}?frame=N   DELETE /api/cleanup3d/band?frame=N
    GET    /api/cleanup3d/horizon                          the user's horizon lines (P15) + the horizon settings
    GET    /api/cleanup3d/horizon/{frame}                  the horizon of a frame measured on the preview (plate /
                                                           projection lines, offset, step before / after)
    PUT    /api/cleanup3d/horizon/{frame}   DELETE the same   the user's horizon line of a frame (curve)

Built by `make(...)` with the app's shared pieces passed in explicitly (nothing is imported from app.py); `cleanup` =
the exports of api/cleanup.py (the MVP's plate / alpha / fill-source helpers – one implementation for both renders).
Long actions go through `long_job` (engine/jobs.py) with progress and „Przerwij”. Errors: ValueError of the engine ->
409 with its Polish text, unknown card / key -> 404, no project -> 400, a missing model -> 409 with the gate's reason.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from engine.cleanup3d import store
from engine.cleanup3d import types as T

VIEW_CACHE = 96                                # preview PNGs kept in memory (scrubbing back and forth)
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]


class SolveRequest(BaseModel):                 # C2 – POST /api/cleanup3d/camera/solve
    moving: Optional[List[str]] = None         # objects left out of the solve (moving things); None = the objects chosen
                                               # for removal in the Cleanup settings, [] = nothing moves
    single_focal: bool = True
    horizon: bool = True                       # use the sea / ground horizon as a constraint when one is found


class CardsRequest(BaseModel):                 # C3 – PUT /api/cleanup3d/cards (the whole list: order = list order)
    cards: List[Dict[str, Any]]


class DepthCardRequest(BaseModel):             # C3 – POST /api/cleanup3d/cards/{card}/depth (P18 depth card)
    frame: Optional[int] = None                # reference frame (None = the card's, else the camera's middle frame)
    source: Optional[str] = None               # computed | plate | None / "auto" = the first available
    conv: Optional[str] = None                 # disparity | z (None = the source's default)
    fit: bool = True                           # fit scale / offset to the camera (resets the sliders)


class AreasRequest(BaseModel):                 # extent of a card / regeneration areas of a key
    areas: List[Dict[str, Any]]                # brush layers carry `png_data` (data URL) only when their pixels changed
    frame: Optional[int] = None                # extent only: the key frame the areas were drawn in


class KeysRequest(BaseModel):                  # C4 – POST /api/cleanup3d/keys/{card}
    frames: Optional[List[int]] = None         # add keys at these plate frames (key_mode manual)
    every: Optional[int] = None                # or one key every N frames of the work range (key_mode every)
    source: str = "plate"                      # what the first version of a new key is: plate (the plate frame itself)


class AutoKeysRequest(BaseModel):              # C4 – POST /api/cleanup3d/keys/{card}/auto (key_mode auto, P13)
    threshold: Optional[float] = None          # projection error (0..1 per channel) above which a frame gets a key
    min_gap: Optional[int] = None              # frames at least between an added key and the others
    max_gap: Optional[int] = None              # no two keys further apart (0 = no limit)
    max_keys: Optional[int] = None
    create: bool = True                        # add the keys (first version = the plate frame); False = only measure


class MasterKeyRequest(BaseModel):             # C4 – POST /api/cleanup3d/keys/{card}/master (key_mode one, P13)
    frame: Optional[int] = None                # key frame = the view of the canvas; None = the middle of the work range
    replace: bool = False                      # the card's other keys are deleted for good (asked in the UI)


class RebaseRequest(BaseModel):                # C4 – POST /api/cleanup3d/rebase_keys
    card: Optional[str] = None                 # None = the keys of every card
    frames: Optional[List[int]] = None         # None = every key frame
    grow_px: Optional[float] = None            # how far past the removed objects' mattes the fill is kept (default 40)
    soft_px: Optional[float] = None            # soft edge of that region (default 24)
    effects: bool = True                       # also the objects' shadow / kicked sand the model removed with them
    again: bool = False                        # rebuild keys that already are such a patch (from their model frame)


class VoidRequest(BaseModel):                  # C4 – POST /api/cleanup3d/void (P12)
    frames: Optional[List[int]] = None         # None = the work range (VOID runs once over the whole shot)
    prompt: Optional[str] = None               # what the shot shows AFTER the removal (VOID's background prompt)
    affect_px: Optional[float] = None          # traces area = the removed objects grown by this (plate px, default 40)
    keys: bool = True                          # then build key versions from the run (existing keys of every card)


class VoidKeysRequest(BaseModel):              # C4 – POST /api/cleanup3d/void/keys (P12)
    cards: Optional[List[str]] = None          # None = every card
    frames: Optional[List[int]] = None         # None = the existing keys of each card


class SeedVR2Request(BaseModel):               # C4 – POST /api/cleanup3d/keys/{card}/seedvr2 (P12)
    frames: Optional[List[int]] = None         # None = every key of the card with a fill (active version not the plate)
    in_scale: Optional[float] = None           # the crop goes to SeedVR2 scaled by this (None = plate / VOID ratio)


class ImportRequest(BaseModel):                # C4 – POST .../import
    path: str                                  # a file on this machine (PNG 8 / 16-bit, EXR), same size as the key


class ExportRequest(BaseModel):                # C4 – POST .../export
    format: str = "png"                        # png | png16 | exr
    dir: Optional[str] = None                  # default: <work>/inpaint/keys_export/


class MarkRequest(BaseModel):                  # C4 – POST /api/cleanup3d/mark/{frame}: areas drawn on ANY frame -> its keys
    areas: List[Dict[str, Any]]
    cards: Optional[List[str]] = None          # None = every card the areas touch


class RegenRequest(BaseModel):                 # C4 – POST .../regen
    engine: str = "lama"                       # lama | qwen21_edit | klein_restore (types.ENGINES)
    soften_seams: bool = False                 # klein_restore: blur the reference in the area + a second pass
    prompt: Optional[str] = None               # klein_restore / qwen21_edit: what should be in the area
    base: Optional[str] = None                 # the version the step starts from (None = the active one)
    area: str = "marked"                       # marked = the key's areas | patch = the whole patch of the base version
    strength: Optional[float] = None           # klein_restore: denoise 0.3–1.0 (None = 1.0 marked, 0.9 patch)


class ActiveRequest(BaseModel):                # C4 – PUT .../active
    version: str


class RenderSettingsRequest(BaseModel):        # C5 – PUT /api/cleanup3d/render_settings (patch blend)
    key_soft_px: Optional[float] = None        # soft outward ramp of the key patch (px)
    key_grow_px: Optional[float] = None        # the object part of the key patch grown by this (px)
    key_effects: Optional[bool] = None         # the object's shadow / kicked sand join the patch
    key_effects_thr: Optional[float] = None    # plate vs keys difference above which a pixel is an effect
    clean_first: Optional[bool] = None         # the clear plate before the keys (motion control)
    horizon_match: Optional[bool] = None       # P15: the projection moved onto the plate's horizon
    horizon_fade_px: Optional[float] = None    # P15: how far above / below the horizon the correction fades out (px)
    horizon_search_px: Optional[float] = None  # P15: band around the cards' horizon where the edges are searched (px)
    grain: Optional[str] = None                # P17 grain match of the patch: auto | off | on
    grain_strength: Optional[float] = None     # multiplies the missing grain (0..2)


class KeyMoveRequest(BaseModel):               # C4 timeline – POST /api/cleanup3d/keys/{card}/{frame}/move (P16)
    to: int                                    # the target frame


class CardOffRequest(BaseModel):               # timeline – PUT /api/cleanup3d/cards/{card}/off (P16)
    off: bool                                  # True = the card is left out of the render, previews and Nuke export


class CardSoloRequest(BaseModel):              # timeline – PUT /api/cleanup3d/cards/{card}/solo (P16, like After Effects)
    solo: bool                                 # any card soloed = only the soloed cards go into render / previews / Nuke


class HorizonLineRequest(BaseModel):           # C5 – PUT /api/cleanup3d/horizon/{frame} (P15)
    curve: Dict[str, Any]                      # engine/curves.py JSON (open Bezier curve, plate px)


class C3BandParamsRequest(BaseModel):          # C5 – PUT /api/cleanup3d/band/params (Ctrl+Z of a patch band edit)
    band_points: List[Dict[str, Any]] = []
    band_ref_out: Optional[float] = None
    band_ref_in: Optional[float] = None


class C3BandPointRequest(BaseModel):           # C5 – PUT /api/cleanup3d/band/{frame} (one band point, as the stages')
    id: Optional[str] = None                   # absent = a new point
    x: Optional[float] = None                  # position in this frame (full-frame px); absent = where it is now
    y: Optional[float] = None
    out: Optional[float] = None                # outer whisker (px): how far the patch reaches out of its hole there
    in_: Optional[float] = Field(None, alias="in")   # ignored by the patch (outer side only), kept for the editor
    ang: Optional[float] = None                # tilt of the whisker pair, degrees

class RenderRequest(BaseModel):                # C5 – POST /api/cleanup3d/render
    frames: Optional[List[int]] = None         # None = the work range
    motion_blur: str = "auto"                  # auto (matched per frame) | off
    feather_px: Optional[float] = None         # soft edge of the patch in the plate (the Cleanup setting soft_px)


class NukeExportRequest(BaseModel):            # C6 – POST /api/cleanup3d/export_nuke
    frames: Optional[List[int]] = None         # None = every frame of the solved camera


def decode_mask(data: str) -> np.ndarray:
    """A brush layer sent by the editor (`data:image/png;base64,…`, RGBA 8-bit, white = area) -> uint8 (h, w)."""
    try:
        raw = base64.b64decode(str(data).split(",", 1)[-1], validate=False)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (ValueError, cv2.error):
        img = None
    if img is None:
        raise ValueError("Nie można odczytać warstwy pędzla (png_data)")
    if img.ndim == 3:
        img = img[..., 0]                      # any colour channel (the alpha is not the mask)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return np.ascontiguousarray(img.astype(np.uint8))


def make(*, STATE, objects, plate_size, work_frames, proxy_scale, comfy_client, long_job, refresh_manifest, cleanup,
         solve_camera: Optional[Callable[..., Any]] = None, load_cfg: Optional[Callable[[], Dict[str, Any]]] = None):
    """Build the router. Keyword-only: the app state + functions (late binding); `cleanup` = the exports of
    api/cleanup.py; `solve_camera` / `load_cfg` replace engine.cleanup3d.solve.solve_camera / engine.config.load_config
    (tests). Returns (router, exports)."""
    router = APIRouter()
    lock = threading.RLock()                   # the Scene and its caches are not thread-safe: one preview at a time
    cache: Dict[str, Any] = {"stamp": None, "scene": None}
    views: "OrderedDict[Tuple, bytes]" = OrderedDict()

    # ------------------------------------------------------------------ shared pieces
    def _need() -> Tuple[Path, Dict[str, Any]]:
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        return Path(STATE["work_dir"]), STATE["manifest"]

    def _work() -> Path:
        return _need()[0]

    def _cfg() -> Dict[str, Any]:
        if load_cfg is not None:
            return load_cfg()
        from engine.config import load_config
        return load_config()

    def _guard(fn):
        try:
            return fn()
        except ValueError as e:                # a broken state file, a bad name, an engine refusal: said in Polish
            raise HTTPException(status_code=409, detail=str(e))

    def _card(work: Path, card: str) -> T.Card:
        c = next((c for c in _guard(lambda: store.load_cards(work)) if c.id == card), None)
        if c is None:
            raise HTTPException(status_code=404, detail=f"Brak karty {card}")
        return c

    def _key(work: Path, card: str, frame: int) -> T.Key:
        _card(work, card)
        k = _guard(lambda: store.load_key(work, card, frame))
        if k is None:
            raise HTTPException(status_code=404, detail=f"Brak klucza karty {card} w klatce {frame}")
        return k

    def _beauty(manifest: Dict[str, Any]) -> str:
        pat = (manifest.get("plates") or {}).get("beauty") or ""
        if not pat:
            raise HTTPException(status_code=409, detail="Brak plate'u beauty w projekcie")
        return pat

    def _linear(manifest: Dict[str, Any]) -> bool:
        return ((manifest.get("plates") or {}).get("beauty") or "").lower().endswith(".exr")

    def _plate_fn(beauty: str) -> Callable[[int], np.ndarray]:
        def plate(f: int) -> np.ndarray:
            img = cleanup["read_plate"](cleanup["frame_file"](beauty, f))
            if img is None:
                raise RuntimeError(f"Brak klatki beauty F{f}")
            return img
        return plate

    def _png(img: np.ndarray, **headers: str) -> Response:
        ok, buf = cv2.imencode(".png", img, PNG_FAST)
        if not ok:
            raise HTTPException(status_code=500, detail="Nie udało się zapisać podglądu PNG")
        return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store", **headers})

    def _file_png(path: Path, what: str) -> Response:
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Brak warstwy pędzla {what}")
        return Response(content=path.read_bytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    async def _bg(fn):
        """Run the blocking part of a long job in the executor; the engine's refusals become 409 with their text."""
        from engine.cleanup3d.solve_colmap import SolveCancelled, SolveError
        from engine.jobs import JobCancelled
        try:
            return await asyncio.get_running_loop().run_in_executor(None, fn)
        except SolveCancelled:
            raise JobCancelled()
        except (SolveError, ValueError, RuntimeError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))

    def _progress(step: str, done: int, total: int) -> None:
        from engine.comfy_io import job_progress
        job_progress(step, done, total)

    def _cancel_check() -> None:
        from engine.jobs import JOBS
        if JOBS.running():
            JOBS.check()

    # ------------------------------------------------------------------ the Scene of the open shot (cached)
    def _drop_scene() -> None:
        with lock:
            if cache["scene"] is not None:
                try:
                    cache["scene"].close()
                except Exception:  # noqa: BLE001 – a dead worker must not block the next Scene
                    pass
            cache.update(stamp=None, scene=None)
            views.clear()

    def _scene(work: Path):
        """render.Scene of the shot, rebuilt when the camera, the cards, an extent layer or a key changed
        (store.scene_stamp). Call with `lock` held. ValueError (Polish) when the camera or the cards are missing."""
        from engine.cleanup3d import render as RD
        stamp = (str(work), store.scene_stamp(work))
        if cache["stamp"] != stamp:
            _drop_scene()
            cache.update(scene=RD.Scene(work, device="cpu"), stamp=stamp)
        return cache["scene"]

    # ------------------------------------------------------------------ areas sent by the editor
    def _parse_areas(raw: List[Dict[str, Any]]) -> Tuple[List[T.Area], Dict[str, np.ndarray]]:
        """The editor's list -> (areas, {area id: mask of the brush layers that came with pixels}). The file name of a
        brush layer is always `a_<id>.png` (whatever the client says). ValueError (Polish) on bad data."""
        from engine.cleanup3d import keys as KS
        areas: List[T.Area] = []
        pixels: Dict[str, np.ndarray] = {}
        for d in raw:
            if not isinstance(d, dict):
                raise ValueError("area: element listy obszarów musi być obiektem")
            aid = T.check_name(str(d.get("id")))
            if any(a.id == aid for a in areas):
                raise ValueError(f"Powtórzone id obszaru: {aid}")
            base = {"id": aid, "kind": d.get("kind"), "op": d.get("op", "add"), "label": d.get("label") or ""}
            if d.get("kind") == "brush":
                if d.get("png_data"):
                    pixels[aid] = decode_mask(d["png_data"])
                areas.append(T.Area.from_json({**base, "png": KS.brush_name(aid)}))
            else:
                areas.append(T.Area.from_json({**base, "curve": d.get("curve")}))
        return areas, pixels

    def _write_layers(folder: Path, areas: List[T.Area], pixels: Dict[str, np.ndarray], size: Tuple[int, int]) -> None:
        """Brush layers with new pixels -> `folder/a_<id>.png` (at `size` = (w, h)); a layer without pixels must have
        its file already. Everything is checked before the first file is written."""
        w, h = int(size[0]), int(size[1])
        for a in areas:
            if a.kind == "brush" and a.id not in pixels and not (folder / str(a.png)).is_file():
                raise ValueError(f"Brak warstwy pędzla {a.png} – wyślij jej piksele (png_data)")
        folder.mkdir(parents=True, exist_ok=True)
        for a in areas:
            if a.kind != "brush" or a.id not in pixels:
                continue
            m = pixels[a.id]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(folder / str(a.png)), m):
                raise ValueError(f"Nie udało się zapisać {a.png}")

    def _key_size(key: T.Key) -> Tuple[int, int]:
        if key.canvas and key.canvas.get("w") and key.canvas.get("h"):
            return int(key.canvas["w"]), int(key.canvas["h"])
        return tuple(int(v) for v in plate_size())

    # ------------------------------------------------------------------ state of the whole mode
    def _engines() -> Dict[str, Dict[str, Any]]:
        """Which regeneration engines have their model files (engine/stage_models.py, disk only)."""
        from engine.stage_models import cleanup3d_engines, comfy_gate
        out = cleanup3d_engines(_cfg())
        g = out.get("qwen21_edit")
        if g and g.get("ok"):   # the files are there – does this ComfyUI have the node? (2026-11-10)
            try:
                why = comfy_gate(comfy_client(), "qwen21_edit")
            except Exception:  # noqa: BLE001
                why = ""
            if why:
                out["qwen21_edit"] = {**g, "ok": False, "reason": why}
        return out

    def _solve_gate() -> Dict[str, Any]:
        from engine.stage_models import cleanup3d_solve
        try:
            return cleanup3d_solve(_cfg())
        except Exception as e:  # noqa: BLE001 – a broken check never blocks silently
            return {"ok": False, "reason": f"sprawdzenie nie powiodło się: {e}", "exe": None}

    @router.get("/api/cleanup3d")
    def c3_state():
        """State of the Cleanup 3D mode: steps done / stale, camera report, cards with keys, engines and solve availability."""
        work = _work()
        st = _guard(lambda: store.state(work))
        gate = _engines()
        for e in st["engines"]:
            g = gate.get(e["id"]) or {}
            e["available"], e["reason"] = bool(g.get("ok", True)), str(g.get("reason") or "")
        sg = _solve_gate()
        st["solve"] = {"available": bool(sg.get("ok")), "reason": str(sg.get("reason") or "")}
        st["out_dir"] = str(T.root_dir(work)).replace("\\", "/")
        st.update(_p12_state(work))            # P12: VOID / SeedVR2 availability + the last VOID run
        return st

    # ------------------------------------------------------------------ C2 camera
    @router.get("/api/cleanup3d/camera")
    def c3_camera():
        """The solved 3D camera (cams.json); 404 before the first solve."""
        cam = _guard(lambda: store.load_camera(_work()))
        if cam is None:
            raise HTTPException(status_code=404, detail="Kamera 3D nie jest jeszcze policzona")
        return cam.to_json()

    @router.post("/api/cleanup3d/camera/solve")
    @long_job("Czyszczenie 3D: kamera 3D")
    async def c3_camera_solve(req: SolveRequest):
        """Solve the 3D camera of the work range (COLMAP + CoTracker3 + bundle adjustment); moving objects are masked out."""
        work, manifest = _need()
        sg = _solve_gate()
        if not sg.get("ok"):
            raise HTTPException(status_code=409, detail=sg.get("reason") or "Brak programu COLMAP (CUDA) – Doktor")
        beauty = _beauty(manifest)
        frames = [int(f) for f in work_frames()]
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        w, h = plate_size()
        from engine import cleanup as cu
        s = cu.load_settings(work)
        names = list(s["remove"]) if req.moving is None else [str(n) for n in req.moving]
        masks = None
        if names:
            known = {o["name"] for o in objects().list(groups=True)}
            missing = [n for n in names if n not in known]
            if missing:
                raise HTTPException(status_code=404, detail=f"Brak obiektu {missing[0]}")
            loaders, _ = cleanup["alpha_loaders"]({**s, "remove": names}, w, h)

            def masks(f: int) -> Optional[np.ndarray]:
                al = [np.asarray(fn(f), np.float32) for fn in loaders]
                return (np.max(np.stack(al), axis=0) > 0.02).astype(np.uint8) * 255 if al else None
        solver = solve_camera
        if solver is None:
            from engine.cleanup3d.solve import solve_camera as solver
        cfg = _cfg()
        cam = await _bg(lambda: solver(work, frames, lambda f: Path(cleanup["frame_file"](beauty, f)), cfg=cfg,
                                       moving_masks=masks, single_focal=bool(req.single_focal),
                                       horizon=bool(req.horizon), progress=_progress, cancel_check=_cancel_check))
        _drop_scene()
        return {"success": True, "frames": [cam.frames[0], cam.frames[-1]] if cam.frames else None,
                "n_frames": len(cam.frames), "source": cam.source, "moving": names, "report": cam.report}

    @router.post("/api/cleanup3d/camera/static")
    def c3_camera_static():
        """C2 „Kamera nieruchoma” (user 2026-11-11): a locked-off shot – no COLMAP; one identity pose for every frame of
        the work range (engine/cleanup3d/solve.static_camera). The cards step then offers the „Plan” card facing it."""
        work, _ = _need()
        frames = [int(f) for f in work_frames()]
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        w, h = plate_size()
        from engine.cleanup3d.solve import static_camera

        def run():
            cam = static_camera(frames, w, h)
            store.save_camera(work, cam)
            return cam
        cam = _guard(run)
        _drop_scene()
        return {"success": True, "frames": [cam.frames[0], cam.frames[-1]], "n_frames": len(cam.frames),
                "source": cam.source, "report": cam.report}

    # ------------------------------------------------------------------ C3 cards
    @router.get("/api/cleanup3d/cards")
    def c3_cards():
        """The cards (order = front to back)."""
        return T.cards_to_json(_guard(lambda: store.load_cards(_work())))

    @router.put("/api/cleanup3d/cards")
    def c3_cards_put(req: CardsRequest):
        """Replace the card list (list order = card order)."""
        work = _work()

        def run():
            cards = T.cards_from_json({"cards": [{**c, "order": i} for i, c in enumerate(req.cards)]})
            store.save_cards(work, cards)
            return T.cards_to_json(cards)
        return _guard(run)

    @router.delete("/api/cleanup3d/cards/{card}")
    def c3_card_delete(card: str):
        """Remove a card: its entry, keys and extent are deleted from disk for good (user decision 2026-11-13)."""
        work = _work()
        freed = _guard(lambda: store.delete_card(work, card))
        return {"success": True, "freed_bytes": freed,
                "cards": T.cards_to_json(store.load_cards(work))["cards"]}

    @router.post("/api/cleanup3d/cards/propose")
    def c3_cards_propose():
        """Cards the solve suggests: the ground plane, the sea level (when the shot has a horizon) and the sky. A card
        the user already has (same id; any sky card for the sky) is left exactly as it is."""
        work = _work()
        if _guard(lambda: store.load_camera(work)) is None:
            raise HTTPException(status_code=409, detail="Najpierw policz kamerę 3D (krok „Kamera 3D”)")
        from engine.cleanup3d.solve import propose_planes

        def run():
            p = propose_planes(work)
            cards = store.load_cards(work)
            have = {c.id for c in cards}
            new: List[T.Card] = []
            for cid, name, color, plane in (("plan", "Plan (kamera nieruchoma)", "#9ad06a", p.get("plan")),
                                            ("ground", "Ziemia", "#ffb84a", p.get("ground")),
                                            ("sea", "Morze", "#4ae0ff", p.get("sea"))):
                if plane and cid not in have:
                    new.append(T.Card(cid, name, "plane", {"n": [float(v) for v in plane["n"]], "d": float(plane["d"])},
                                      color=color))
            if p.get("plan") is None and "sky" not in have and not any(c.kind == "sky" for c in cards):
                new.append(T.Card("sky", "Niebo", "sky", color="#b48cff"))
            for i, c in enumerate(new):
                c.order = len(cards) + i
            if new:
                store.save_cards(work, cards + new)
            return {**T.cards_to_json(cards + new), "added": [c.id for c in new],
                    "proposed": {k: p.get(k) for k in ("ground", "sea", "up", "plan")}}
        return _guard(run)

    @router.put("/api/cleanup3d/cards/{card}/extent")
    def c3_card_extent(card: str, req: AreasRequest):
        """Set a card's extent: areas (curves + brush layers with png_data) drawn in `frame`; empty list = the whole card."""
        work = _work()
        _card(work, card)

        def run():
            areas, pixels = _parse_areas(req.areas)
            if areas and req.frame is None:
                raise ValueError("Zasięg karty wymaga klatki, w której został narysowany (frame)")
            for a in areas:
                if a.kind == "curve" and not a.curve.get("closed", True):
                    raise ValueError(f"zasięg karty {card}: krzywa {a.id} nie jest zamknięta")
            _write_layers(store.extent_dir(work, card), areas, pixels, plate_size())
            cards = store.load_cards(work)
            c = next(x for x in cards if x.id == card)
            c.extent, c.extent_frame = areas, (int(req.frame) if areas else None)
            store.save_cards(work, cards)
            return {"card": c.to_json(), "areas": [a.to_json() for a in areas], "frame": c.extent_frame}
        return _guard(run)

    @router.get("/api/cleanup3d/cards/{card}/extent/{area_id}/png")
    def c3_card_extent_png(card: str, area_id: str):
        """A brush layer of a card's extent as 8-bit PNG (white = area)."""
        work = _work()
        c = _card(work, card)
        a = next((a for a in c.extent if a.id == area_id and a.kind == "brush"), None)
        if a is None:
            raise HTTPException(status_code=404, detail=f"Brak warstwy pędzla {area_id}")
        return _file_png(store.extent_dir(work, card) / str(a.png), str(a.png))

    # ------------------------------------------------------------------ C3 depth cards (P18)
    def _depth_sources(work: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        from engine.cleanup3d import depthcard as DC
        return DC.sources(work, manifest)

    @router.get("/api/cleanup3d/depth_sources")
    def c3_depth_sources():
        """Depth maps a depth card can use: the tool's DepthCrafter result (outputs.json) and the shot's depth plate."""
        work, manifest = _need()
        return {"sources": _depth_sources(work, manifest), "convs": list(T.DEPTH_CONVS)}

    @router.post("/api/cleanup3d/cards/{card}/depth")
    def c3_card_depth(card: str, req: DepthCardRequest):
        """Make `card` a depth card: the depth of reference frame `frame` from `source` stored as the card's snapshot,
        fitted to the camera (the solve's points, else the plane cards; fit=false keeps scale / offset)."""
        work, manifest = _need()
        _card(work, card)
        cam = _guard(lambda: store.load_camera(work))
        if cam is None:
            raise HTTPException(status_code=409, detail="Najpierw policz kamerę 3D (krok „Kamera 3D”)")
        from engine.cleanup3d import depthcard as DC

        def run():
            cards = store.load_cards(work)
            c = next(x for x in cards if x.id == card)
            old = dict(c.depth or {}) if c.kind == "depth" else {}
            frame = int(req.frame if req.frame is not None else old.get("frame", cam.frames[len(cam.frames) // 2]))
            if frame not in cam.R:
                raise ValueError(f"Kamera 3D nie ma klatki F{frame} – klatka odniesienia musi być w zakresie kamery")
            src = DC.pick_source(_depth_sources(work, manifest), req.source or old.get("source"))
            conv = req.conv or (old.get("conv") if old.get("source") == src["id"] else None) or src["conv"]
            if conv not in T.DEPTH_CONVS:
                raise ValueError(f"Nieznana konwencja głębi {conv!r}")
            d = DC.read_depth_file(Path(cleanup["frame_file"](src["pattern"], frame)), plate_size())
            rel, stamp = DC.snapshot(work, card, frame, d)
            from engine.cleanup3d.solve import load_points
            from engine.cleanup3d.render import extent_layers
            others = [x for x in cards if x.id != card]
            base = {**old, "frame": frame, "source": src["id"], "conv": conv, "file": rel, "stamp": stamp}
            if not req.fit:
                base.setdefault("fit", {"method": "none"})
            dp, plane = DC.make_depth(cam, frame, d, base, points=load_points(work), cards=others,
                                      layers=extent_layers(work, others), fit=bool(req.fit))
            c.kind, c.depth, c.plane = "depth", dp, plane
            store.save_cards(work, cards)
            _drop_scene()
            return {"card": c.to_json(), "fit": dp.get("fit"), "source": {k: src[k] for k in ("id", "label")}}
        return _guard(run)

    # ------------------------------------------------------------------ C4 keys
    @router.get("/api/cleanup3d/keys")
    def c3_keys(card: Optional[str] = None):
        """Keys of every card (or of `card`) with their versions."""
        return {"keys": [k.to_json() for k in _guard(lambda: store.list_keys(_work(), card))]}

    def _key_ext(manifest: Dict[str, Any], beauty: str, frame: int) -> str:
        """Format of keys made from plate frames: EXR for linear plates, 8-bit plates keep 8-bit, deeper = 16-bit PNG."""
        if _linear(manifest):
            return "exr"
        probe = cv2.imread(cleanup["frame_file"](beauty, int(frame)), cv2.IMREAD_UNCHANGED)
        return "png" if probe is not None and probe.dtype == np.uint8 else "png16"

    @router.post("/api/cleanup3d/keys/{card}")
    def c3_keys_add(card: str, req: KeysRequest):
        """Add keys of a card at `frames` or every N frames of the work range; the first version is the plate frame."""
        work, manifest = _need()
        _card(work, card)
        if req.source != "plate":
            raise HTTPException(status_code=400, detail=f"Nieznane źródło klucza: {req.source}")
        from engine.cleanup3d import keys as KS
        if req.frames:
            frames = sorted({int(f) for f in req.frames})
        elif req.every:
            frames = KS.frames_every(work_frames(), int(req.every))
        else:
            raise HTTPException(status_code=400, detail="Podaj klatki kluczy (frames) albo odstęp (every)")
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        beauty = _beauty(manifest)
        ext = _key_ext(manifest, beauty, frames[0])
        had ={k.frame for k in _guard(lambda: store.list_keys(work, card)) if k.versions}
        try:
            keys = KS.create_keys(work, card, frames, _plate_fn(beauty), ext)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"keys": [k.to_json() for k in keys], "added": [k.frame for k in keys if k.frame not in had]}

    # ------------------------------------------------------------------ C4 key modes of stage III (P13)
    def _remove_mask_fn(work: Path) -> Callable[[int], Optional[np.ndarray]]:
        """f -> the union (0..1, plate size) of the objects chosen for removal in step C1, None when there are none."""
        from engine import cleanup as cu
        w, h = plate_size()
        rm, _keep = cleanup["alpha_loaders"](cu.load_settings(work), w, h)

        def mask(f: int) -> Optional[np.ndarray]:
            al = [np.asarray(fn(int(f)), np.float32) for fn in rm]
            return np.max(np.stack(al), axis=0) if al else None
        return mask

    def _set_mode(work: Path, card: str, mode: str, **fields: Any) -> None:
        cards = store.load_cards(work)
        c = next(x for x in cards if x.id == card)
        c.key_mode = mode
        for k, v in fields.items():
            setattr(c, k, v)
        store.save_cards(work, cards)

    def _key_frames(work: Path, manifest: Dict[str, Any]) -> List[int]:
        if _guard(lambda: store.load_camera(work)) is None:
            raise HTTPException(status_code=409, detail="Najpierw policz kamerę 3D (krok „Kamera 3D”)")
        frames = sorted({int(f) for f in work_frames()})
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        _beauty(manifest)
        return frames

    @router.post("/api/cleanup3d/keys/{card}/auto")
    @long_job("Czyszczenie 3D: klucze automatycznie {card}")
    async def c3_keys_auto(card: str, req: AutoKeysRequest):
        """Key mode „automat”: key frames where the projection error of the card grows (keymodes.auto_key_frames);
        the settings and the mode are saved on the card, the keys are added (first version = the plate frame)."""
        work, manifest = _need()
        c = _card(work, card)
        frames = _key_frames(work, manifest)
        from engine.cleanup3d import keymodes as KM
        from engine.cleanup3d import keys as KS
        from engine.cleanup3d import render as RD
        s = KM.auto_settings({**c.key_auto, **req.model_dump(exclude_none=True, exclude={"create"})})
        beauty = _beauty(manifest)
        plate, rmask, linear = _plate_fn(beauty), _remove_mask_fn(work), _linear(manifest)

        def run():
            sc = RD.Scene(work, device="cpu")  # own Scene: the job runs long, the previews keep theirs
            try:
                res = KM.auto_key_frames(sc.geometry, sc.card(card), frames, plate, rmask, linear=linear,
                                         progress=_progress, cancel_check=_cancel_check, **s)
            finally:
                sc.close()
            _set_mode(work, card, "auto", key_auto={k: float(s[k]) for k in KM.AUTO_DEFAULTS})
            added: List[int] = []
            if req.create:
                had = {k.frame for k in store.list_keys(work, card) if k.versions}
                ext = _key_ext(manifest, beauty, res["frames"][0])
                keys = KS.create_keys(work, card, res["frames"], plate, ext)
                added = [k.frame for k in keys if k.frame not in had]
            return {**res, "errors": {str(f): e for f, e in res["errors"].items()}, "added_keys": added,
                    "settings": s}
        res = await _bg(run)
        return {"success": True, **res}

    @router.post("/api/cleanup3d/keys/{card}/master")
    @long_job("Czyszczenie 3D: klucz-suma {card}")
    async def c3_keys_master(card: str, req: MasterKeyRequest):
        """Key mode „jeden klucz”: ONE master key of the card accumulated from the clean pixels of every frame
        (keymodes.build_master) on its own canvas larger than the frame; what no frame sees becomes the key's area
        `unseen` (for regeneration). The card's other keys must go first: 409 unless `replace` (deleted for good)."""
        work, manifest = _need()
        _card(work, card)
        frames = _key_frames(work, manifest)
        from engine.cleanup3d import keymodes as KM
        from engine.cleanup3d import keys as KS
        from engine.cleanup3d import render as RD
        cam = _guard(lambda: store.load_camera(work))
        in_cam = [f for f in frames if f in cam.R]
        if not in_cam:
            raise HTTPException(status_code=409, detail="Kamera 3D nie ma żadnej klatki z zakresu pracy")
        M = int(req.frame) if req.frame is not None else in_cam[len(in_cam) // 2]
        if M not in cam.R:
            raise HTTPException(status_code=409, detail=f"Kamera 3D nie ma klatki {M}")
        keys = [k for k in _guard(lambda: store.list_keys(work, card)) if k.versions]
        cur = next((k for k in keys if k.frame == M and k.canvas), None)
        others = [k for k in keys if k is not cur]
        if others and not req.replace:
            raise HTTPException(status_code=409, detail=(
                f"Karta {card} ma już {len(others)} kluczy (F{', F'.join(str(k.frame) for k in others[:8])}"
                f"{' …' if len(others) > 8 else ''}) – klucz-suma je zastępuje: potwierdź ich trwałe usunięcie"))
        beauty = _beauty(manifest)
        plate, rmask = _plate_fn(beauty), _remove_mask_fn(work)
        ext = "exr" if _linear(manifest) else "png16"   # accumulated + colour-normalised pixels: never 8-bit

        def run():
            sc = RD.Scene(work, device="cpu")
            try:
                res = KM.build_master(sc.geometry, sc.card(card), frames, plate, rmask, key_frame=M,
                                      canvas=None if cur is None else dict(cur.canvas),
                                      progress=_progress, cancel_check=_cancel_check)
            finally:
                sc.close()
            freed = 0
            for k in others:                   # only after the build succeeded (a cancel keeps the old keys)
                freed += KS.delete_key(work, card, k.frame) or 0
            key = KM.save_master(work, card, res, ext)
            _set_mode(work, card, "one")
            return {"key": key.to_json(), "key_frame": M, "canvas": key.canvas, "frames": res["frames"],
                    "seen_share": res["seen_share"], "hole_px": res["hole_px"],
                    "replaced": [k.frame for k in others], "freed_bytes": freed,
                    "unseen_area": KM.UNSEEN_AREA if res["hole_px"] else None}
        res = await _bg(run)
        return {"success": True, **res}

    # ------------------------------------------------------------------ C4 VOID + SeedVR2 as key engines (P12)
    def _p12_state(work: Path) -> Dict[str, Any]:
        from engine.cleanup3d import void_engine as VE
        return VE.state(work, _cfg())

    async def _p12_bg(fn):
        """Like _bg, but a „Przerwij” stays a cancel (JobCancelled is a RuntimeError) and a tool's refusal is 409."""
        from engine.cleanup3d.void_engine import ToolCancelled, ToolError
        from engine.jobs import JobCancelled
        try:
            return await asyncio.get_running_loop().run_in_executor(None, fn)
        except JobCancelled:
            raise
        except ToolCancelled:
            raise JobCancelled()
        except (ToolError, ValueError, RuntimeError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))

    def _p12_masks(work: Path):
        """(removed objects, kept objects): f -> union 0..1 at plate size or None (the objects chosen in step C1)."""
        from engine import cleanup as cu
        w, h = plate_size()
        rm, keep = cleanup["alpha_loaders"](cu.load_settings(work), w, h)

        def union(fns):
            def m(f: int) -> Optional[np.ndarray]:
                al = [np.asarray(fn(int(f)), np.float32) for fn in fns]
                return np.max(np.stack(al), axis=0) if al else None
            return m
        return (union(rm) if rm else None), (union(keep) if keep else None)

    def _p12_blur(work: Path):
        """frame -> patch_on_plate's blur hook (render.match_patch_blur on the frame's own plate) or None (no camera)."""
        if store.load_camera(work) is None:
            return None
        from engine.cleanup3d import render as RD

        def of(frame: int):
            with lock:
                sc = _scene(work)
                geo = sc.geometry if int(frame) in sc.camera.R else None
            if geo is None:
                return None

            def blur(fill, pl, hole):
                with lock:
                    return RD.match_patch_blur(geo, int(frame), fill, pl, hole)[0]
            return blur
        return of

    def _p12_log(work: Path, name: str) -> Path:
        from engine.cleanup3d import void_engine as VE
        d = VE.void_root(work)
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    @router.post("/api/cleanup3d/keys/{card}/seedvr2")
    @long_job("Czyszczenie 3D: SeedVR2 klucze {card}")
    async def c3_keys_seedvr2(card: str, req: SeedVR2Request):
        """SeedVR2 on the card's keys (active versions with a fill) as ONE clip of one crop around the patches, pasted
        back over the plate (keys.patch_on_plate): a new version per key, source „seedvr2:<base>”, base recorded."""
        work, manifest = _need()
        _card(work, card)
        from engine.cleanup3d import void_engine as VE
        cfg = _cfg()
        runner = _guard(lambda: _p12_runner(VE.seedvr2_runner, cfg, _p12_log(work, "seedvr2.log")))
        rmask, _keep = _p12_masks(work)
        if rmask is None:
            raise HTTPException(status_code=409, detail="Wybierz obiekty do usunięcia (krok C1 „Inpaint”)")
        plate, linear = _plate_fn(_beauty(manifest)), _linear(manifest)

        def run():
            return VE.seedvr2_keys(work, card, plate_fn=plate, mask_fn=rmask, runner=runner, frames=req.frames,
                                   in_scale=req.in_scale, blur_fn=_p12_blur(work), linear=linear,
                                   ffmpeg=VE._ffmpeg(cfg), progress=_progress, cancel=_cancel_check,
                                   client=comfy_client())
        res = await _p12_bg(run)
        return {"success": True, **res}

    def _p12_runner(make_runner, cfg, log):
        from engine.cleanup3d.void_engine import ToolError
        try:
            return make_runner(cfg, log=log)
        except ToolError as e:                 # the gate's Polish reason (not configured / files missing)
            raise ValueError(str(e))

    @router.delete("/api/cleanup3d/keys/{card}/{frame}")
    def c3_key_delete(card: str, frame: int):
        """Delete a key (all its versions) from disk for good (user decision 2026-11-13)."""
        work = _work()
        _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        return {"success": True, "freed_bytes": KS.delete_key(work, card, frame) or 0}

    @router.get("/api/cleanup3d/keys/{card}/{frame}/image")
    def c3_key_image(card: str, frame: int, version: Optional[str] = None, max_side: Optional[int] = None):
        """PNG preview of the active (or given) version of a key."""
        work = _work()
        key = _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        from engine.cleanup3d import render as RD

        def run():
            path = KS.version_path(work, key, version)
            return RD._to8(KS.read_key_image(work, key, version), path.suffix.lower() == ".exr",
                           int(max_side) if max_side else None)
        return _png(_guard(run))

    @router.post("/api/cleanup3d/keys/{card}/{frame}/export")
    def c3_key_export(card: str, frame: int, req: ExportRequest):
        """Write the key's active version for painting outside (png / png16 / exr) and return the path."""
        work = _work()
        _key(work, card, frame)
        if req.format not in T.KEY_FORMATS:
            raise HTTPException(status_code=400, detail=f"Nieznany format klucza: {req.format}")
        from engine.cleanup3d import keys as KS
        out_dir = Path(req.dir) if req.dir else T.root_dir(work) / "keys_export"
        try:
            path = KS.export_key(work, card, frame, req.format, out_dir)
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"success": True, "path": str(path).replace("\\", "/"), "format": req.format}

    @router.post("/api/cleanup3d/keys/{card}/{frame}/import")
    def c3_key_import(card: str, frame: int, req: ImportRequest):
        """A painted file of this machine as a NEW version of the key (409: other size, unreadable, not RGB)."""
        work = _work()
        _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        key = _guard(lambda: KS.import_key(work, card, frame, req.path.strip().strip('"')))
        return {"success": True, "version": key.active, "key": key.to_json()}

    @router.post("/api/cleanup3d/keys/{card}/{frame}/undo")
    def c3_key_undo(card: str, frame: int):
        """Make the previous version of the key active (files stay)."""
        work = _work()
        _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        key = _guard(lambda: KS.undo(work, card, frame))
        return {"success": True, "version": key.active, "key": key.to_json()}

    @router.put("/api/cleanup3d/keys/{card}/{frame}/active")
    def c3_key_set_active(card: str, frame: int, req: ActiveRequest):
        """Make a version of the key active (the C4 version list; files stay)."""
        work = _work()
        _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        key = _guard(lambda: KS.set_active(work, card, frame, req.version))
        return {"success": True, "version": key.active, "key": key.to_json()}

    @router.delete("/api/cleanup3d/keys/{card}/{frame}/versions/{version}")
    def c3_key_version_delete(card: str, frame: int, version: str):
        """Delete one version of the key from disk for good (its number is never given again). 409: the active
        version (switch to another one first), the last version (delete the key instead), no such version."""
        work = _work()
        _key(work, card, frame)
        from engine.cleanup3d import keys as KS
        freed = _guard(lambda: KS.delete_version(work, card, frame, version))
        k = store.load_key(work, card, frame)
        return {"success": True, "freed_bytes": freed, "key": None if k is None else k.to_json()}

    @router.get("/api/cleanup3d/keys/{card}/{frame}/areas")
    def c3_key_areas(card: str, frame: int):
        """The regeneration areas of a key (layers in order)."""
        work = _work()
        _key(work, card, frame)
        return {"areas": [a.to_json() for a in _guard(lambda: store.load_areas(work, card, frame))]}

    @router.put("/api/cleanup3d/keys/{card}/{frame}/areas")
    def c3_key_areas_put(card: str, frame: int, req: AreasRequest):
        """Replace the regeneration areas of a key; a brush layer without png_data keeps its file."""
        work = _work()
        key = _key(work, card, frame)

        def run():
            areas, pixels = _parse_areas(req.areas)
            _write_layers(T.key_dir(work, card, frame), areas, pixels, _key_size(key))
            store.save_areas(work, card, frame, areas)
            return {"areas": [a.to_json() for a in areas]}
        return _guard(run)

    @router.get("/api/cleanup3d/keys/{card}/{frame}/areas/{area_id}/png")
    def c3_key_area_png(card: str, frame: int, area_id: str):
        """A brush layer of a key's areas as 8-bit PNG (white = area)."""
        work = _work()
        _key(work, card, frame)
        a = next((a for a in _guard(lambda: store.load_areas(work, card, frame))
                  if a.id == area_id and a.kind == "brush"), None)
        if a is None:
            raise HTTPException(status_code=404, detail=f"Brak warstwy pędzla {area_id}")
        return _file_png(T.key_dir(work, card, frame) / str(a.png), str(a.png))

    @router.post("/api/cleanup3d/keys/{card}/{frame}/regen")
    @long_job("Czyszczenie 3D: dogenerowanie klucza {card} F{frame}")
    async def c3_key_regen(card: str, frame: int, req: RegenRequest):
        """Regenerate the marked areas of a key (LaMa, Qwen-Image-2.1 or FLUX 3 Image Klein + LoRA Restore) into a new
        active version."""
        work = _work()
        key = _key(work, card, frame)
        if req.engine not in T.ENGINES:
            raise HTTPException(status_code=400, detail=f"Nieznany silnik dogenerowania: {req.engine}")
        from engine.cleanup3d import regen as RG
        if req.area not in RG.AREA_MODES:
            raise HTTPException(status_code=400, detail=f"Nieznany obszar dogenerowania: {req.area}")
        cfg = _cfg()
        _guard(lambda: RG.check_engine(req.engine, cfg))          # 409 with the gate's reason before anything runs
        if req.engine == "klein_restore" and req.strength is not None:
            _guard(lambda: RG.check_strength(req.strength))
        from engine.cleanup3d import keys as KS
        if key.versions:                                         # (no version at all: the engine says „Brak klucza”)
            _guard(lambda: KS._version(key, req.base))           # 409: no such base version
        if req.area == "patch" and key.canvas is not None:
            raise HTTPException(status_code=409, detail="„Cała łata” działa na kluczach z klatki plate'u – klucz-suma ma "
                                                        "własne płótno, zaznacz obszar")
        manifest = _need()[1]
        label = T.ENGINE_LABEL[req.engine]

        def patch_inputs():
            """(patch weight, plate of the key frame) of the „cała łata” mode: where the base version differs from the
            plate + the patch band (render settings key_soft_px, the project's band points of the key frame)."""
            from engine.cleanup3d import band as C3B
            plate = KS._bgr3(KS._to_float(_plate_fn(_beauty(manifest))(int(frame))))
            base = KS.read_key_image(work, key, req.base)
            if plate.shape[:2] != base.shape[:2]:
                raise ValueError(f"Plate klatki {int(frame)} ma inny rozmiar niż klucz")
            b, rm, _s, _rs = _band_ctx(work)
            pts = None
            try:                               # the band points of the key frame, carried by the removal objects
                fn = C3B.points_fn(b, _band_core_at(rm)) if rm else None
                pts = fn(int(frame)) if fn else None
            except Exception:  # noqa: BLE001 – no objects in this frame: the band without its points
                pts = None
            wgt = RG.patch_weight(base, plate, b.out_px, b.outer_feather_px, pts, linear=_linear(manifest))
            return wgt, plate

        def run():
            extent = None
            if key.canvas is None and store.load_camera(work) is not None:
                with lock:                     # nothing outside the card's weight in the key frame may change
                    sc = _scene(work)
                    if int(frame) in sc.camera.R:
                        extent = np.array(sc.geometry.card_weight(sc.card(card), int(frame)), np.float32)
            blur = None
            if key.canvas is None and store.load_camera(work) is not None:
                from engine.cleanup3d import render as RD
                with lock:
                    sc = _scene(work)
                    geo = sc.geometry if int(frame) in sc.camera.R else None
                if geo is not None:
                    def blur(gen, real, hole, known):
                        with lock:
                            return RD.match_patch_blur(geo, int(frame), gen, real, hole, known)[0]
            patch, plate = patch_inputs() if req.area == "patch" else (None, None)
            return RG.regenerate(work, card, frame, req.engine, client=comfy_client(), io_dir=work / "comfy_io",
                                 blur=blur,
                                 soften_seams=bool(req.soften_seams), prompt=(req.prompt or None), extent=extent,
                                 progress=lambda d, n: _progress(f"Dogenerowanie klucza: {label}", d, n), cfg=cfg,
                                 base=req.base, area_mode=req.area, patch=patch, plate=plate, strength=req.strength)
        res = await _bg(run)
        k = store.load_key(work, card, frame)
        return {"success": True, **res, "key": None if k is None else k.to_json()}

    @router.post("/api/cleanup3d/rebase_keys")
    @long_job("Czyszczenie 3D: tło kluczy z plate'u")
    async def c3_rebase_keys(req: RebaseRequest):
        """Keys that are whole frames of a model (a video inpainter repaints everything) -> the PLATE + that fill only
        where the removed objects of step C1 are (grown, soft edge, tone-matched). A new version per key; keys whose
        active version already is the plate or such a patch are left."""
        work, manifest = _need()
        from engine import cleanup as cu
        from engine.cleanup3d import keys as KS
        s = cu.load_settings(work)
        w, h = plate_size()
        rm, _keep = cleanup["alpha_loaders"](s, w, h)
        plate = _plate_fn(_beauty(manifest))
        keys = [k for k in _guard(lambda: store.list_keys(work, req.card)) if k.versions
                and (not req.frames or k.frame in {int(f) for f in req.frames})]
        if not keys:
            raise HTTPException(status_code=409, detail="Brak kluczy do przebudowania")
        keys.sort(key=lambda k: (k.frame, k.card))
        grow = KS.REBASE_GROW_PX if req.grow_px is None else float(req.grow_px)
        soft = KS.REBASE_SOFT_PX if req.soft_px is None else float(req.soft_px)

        from engine.cleanup3d import render as RD
        lengths: Dict[str, float] = {}

        def blur_of(frame: int, tag: str):
            """The generated patch gets the motion blur of its key frame (measured on the model's own output)."""
            with lock:
                sc = _scene(work)
                geo = sc.geometry if int(frame) in sc.camera.R else None
            if geo is None:
                return None

            def blur(fill, pl, hole):
                with lock:
                    out, L = RD.match_patch_blur(geo, int(frame), fill, pl, hole)
                lengths[tag] = L
                return out
            return blur

        def run():
            done, left = [], []
            by_frame: Dict[int, Any] = {}
            for i, k in enumerate(keys):
                _cancel_check()
                _progress("Tło kluczy z plate'u", i, len(keys))
                if k.frame not in by_frame:    # the same plate and mask for every card's key of this frame
                    al = [np.asarray(fn(k.frame), np.float32) for fn in rm]
                    by_frame = {k.frame: (plate(k.frame), np.max(np.stack(al), axis=0))}
                pl, mask = by_frame[k.frame]
                nk = KS.rebase_on_plate(work, k.card, k.frame, pl, mask, grow, soft, effects=bool(req.effects),
                                        again=bool(req.again), blur=blur_of(k.frame, f"{k.card}/{k.frame}"))
                (done if nk is not None else left).append({"card": k.card, "frame": k.frame})
            _progress("Tło kluczy z plate'u", len(keys), len(keys))
            return {"rebased": done, "unchanged": left, "grow_px": grow, "soft_px": soft, "blur_lengths": lengths}
        res = await _bg(run)
        return {"success": True, **res}

    @router.post("/api/cleanup3d/void")
    @long_job("Czyszczenie 3D: VOID – wypełnienie ujęcia")
    async def c3_void(req: VoidRequest):
        """VOID once over the work range (input video + quadmask from the objects of step C1), the frames kept in
        <work>/inpaint/void/<run>/; with `keys` the cards' existing keys get a new version = the plate + the VOID fill
        only where the objects and their traces are."""
        work, manifest = _need()
        from engine.cleanup3d import void_engine as VE
        cfg = _cfg()
        runner = _guard(lambda: _p12_runner(VE.void_runner, cfg, _p12_log(work, "void.log")))
        frames = sorted({int(f) for f in (req.frames or work_frames())})
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        rmask, keep = _p12_masks(work)
        if rmask is None:
            raise HTTPException(status_code=409, detail="Wybierz obiekty do usunięcia (krok C1 „Inpaint”)")
        beauty = _beauty(manifest)
        plate, linear = _plate_fn(beauty), _linear(manifest)
        ext = _key_ext(manifest, beauty, frames[0])
        affect = VE.AFFECT_PX if req.affect_px is None else max(0.0, float(req.affect_px))

        def run():
            r = VE.void_fill(work, frames, plate, rmask, runner, progress=_progress, cancel=_cancel_check,
                             keep_fn=keep, affect_px=affect, linear=linear, prompt=(req.prompt or "").strip()
                             or VE.VOID_PROMPT, ffmpeg=VE._ffmpeg(cfg), client=comfy_client())
            r.pop("paths", None)
            out = {"run": r, "keys": []}
            cards = [c.id for c in store.load_cards(work)]
            if req.keys and any(k.versions for c in cards for k in store.list_keys(work, c)):
                out["keys"] = VE.keys_from_void(work, cards, None, plate_fn=plate, mask_fn=rmask, run=r,
                                                blur_fn=_p12_blur(work), ext=ext, linear=linear,
                                                progress=_progress, cancel=_cancel_check)["keys"]
            return out
        res = await _p12_bg(run)
        return {"success": True, **res}

    @router.post("/api/cleanup3d/void/keys")
    @long_job("Czyszczenie 3D: klucze z VOID")
    async def c3_void_keys(req: VoidKeysRequest):
        """Key versions from the last VOID run (no new VOID pass): the plate + the VOID fill (upscaled to the plate)
        only where the objects of step C1 and their traces are; source „void”, base = the version before."""
        work, manifest = _need()
        from engine.cleanup3d import void_engine as VE
        if VE.last_run(work) is None:
            raise HTTPException(status_code=409, detail="Brak policzonego wypełnienia VOID – najpierw „VOID – wypełnienie ujęcia”")
        cards = req.cards or [c.id for c in _guard(lambda: store.load_cards(work))]
        for c in cards:
            _card(work, c)
        rmask, _keep = _p12_masks(work)
        if rmask is None:
            raise HTTPException(status_code=409, detail="Wybierz obiekty do usunięcia (krok C1 „Inpaint”)")
        beauty = _beauty(manifest)
        plate, linear = _plate_fn(beauty), _linear(manifest)
        frames = None if not req.frames else {c: [int(f) for f in req.frames] for c in cards}
        ext = _key_ext(manifest, beauty, (req.frames or work_frames() or [1])[0])

        def run():
            return VE.keys_from_void(work, cards, frames, plate_fn=plate, mask_fn=rmask, blur_fn=_p12_blur(work),
                                     ext=ext, linear=linear, progress=_progress, cancel=_cancel_check)
        res = await _p12_bg(run)
        return {"success": True, **res}

    @router.post("/api/cleanup3d/mark/{frame}")
    def c3_mark(frame: int, req: MarkRequest):
        """Areas drawn in plate frame `frame` -> appended (as closed curves in the key's pixels) to the areas of the
        keys that frame is made of: the frame's own key, else the two nearest keys of each card."""
        work = _work()
        from engine.cleanup3d import keys as KS

        def run():
            areas, pixels = _parse_areas(req.areas)
            lost = [a.id for a in areas if a.kind == "brush" and a.id not in pixels]
            if lost:
                raise ValueError(f"Zaznaczenie pędzlem {lost[0]} bez pikseli (png_data)")
            if not areas:
                raise ValueError("Puste zaznaczenie – narysuj obszar")
            with lock:
                sc = _scene(work)
                if int(frame) not in sc.camera.R:
                    raise ValueError(f"Kamera 3D nie ma klatki {int(frame)}")
                w, h = sc.size
                if req.cards is None:
                    # where the drawing lies, whatever the layers' op: the cards whose weight it touches
                    touch = KS.render_areas([T.Area(a.id, a.kind, "add", a.curve, a.png, a.label) for a in areas],
                                            (h, w), brush=pixels) > 0.5
                    wts = sc.geometry.card_weights(int(frame))
                    cards = [c for c in sc.cards if touch.any() and (wts[c.id][touch] > 0.5).any()]
                else:
                    cards = [c for c in sc.cards if c.id in set(req.cards)]
                out = []
                for c in cards:
                    ks = sc.keys_by_card.get(c.id) or []
                    if not ks:
                        continue
                    for kf, new in KS.carry_areas(sc.geometry, areas, int(frame), ks, c, brush=pixels).items():
                        if not new:
                            continue
                        cur = store.load_areas(work, c.id, kf)
                        ids = {a.id for a in cur}
                        for a in new:
                            base, n = f"m{int(frame)}_{a.id}"[:56], 1
                            aid = base
                            while aid in ids:
                                n += 1
                                aid = f"{base}_{n}"
                            a.id = T.check_name(aid)
                            ids.add(a.id)
                        store.save_areas(work, c.id, kf, cur + new)
                        out.append({"card": c.id, "frame": int(kf), "added": len(new)})
            if not out:
                raise ValueError("Zaznaczenie nie trafia w żadną kartę z kluczami – dodaj klucz karty albo wybierz kartę")
            return {"success": True, "keys": out}
        if req.cards is not None:
            for cid in req.cards:
                _card(work, cid)
        return _guard(run)

    # ------------------------------------------------------------------ C5 projection + previews
    @router.post("/api/cleanup3d/render")
    @long_job("Czyszczenie 3D: projekcja kluczy")
    async def c3_render(req: RenderRequest):
        """Project the keys onto the frames and compose them into the plate: a new Cleanup version, shown as plate „cleaned”."""
        work, manifest = _need()
        from engine import cleanup as cu
        from engine.cleanup3d import render as RD
        if req.motion_blur not in RD.BLUR_MODES:
            raise HTTPException(status_code=400, detail=f"Nieznany tryb rozmycia ruchu: {req.motion_blur}")
        s = cu.load_settings(work)
        w, h = plate_size()
        beauty = _beauty(manifest)
        linear = _linear(manifest)
        rm, keep = cleanup["alpha_loaders"](s, w, h)
        clean, auto, info = cleanup["fill_sources"](work, manifest, w, h, linear)
        frames = [int(f) for f in (req.frames if req.frames else work_frames())]
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        # a work range longer than the solved camera: render the frames it has, say which were skipped (2026-11-10)
        cam = _guard(lambda: store.load_camera(work))
        skipped: List[int] = []
        if cam is not None and cam.R:
            skipped = [f for f in frames if f not in cam.R]
            if len(skipped) == len(frames):
                cf = cam.frames
                raise HTTPException(status_code=409, detail=(
                    f"Kamera 3D nie ma żadnej klatki zakresu pracy {cu.frame_ranges([frames[0], frames[-1]], '')} "
                    f"(kamera: {cf[0]}–{cf[-1]})"))
            frames = [f for f in frames if f in cam.R]
        from engine.cleanup3d import band as C3B
        st: Dict[str, Any] = {**s, **store.load_render_settings(work), "patch_band": C3B.load(work),
                              "motion_blur": req.motion_blur,
                              "device": str(_cfg().get("cleanup3d_device") or "cpu")}
        if req.feather_px is not None:
            st["soft_px"] = float(req.feather_px)
        ov = {"soft_px": float(req.feather_px)} if req.feather_px is not None else None
        sig = _guard(lambda: store.render_sig(work, ov))   # what this render is made of (a change meanwhile = stale)
        inpaint = cleanup["inpainter"](work, s)
        # removed objects lost for a few frames: scanned BEFORE the render, so the render's loaders hold their mask
        held = await _bg(lambda: cleanup["held"](rm, frames, progress=_progress))
        rj = await _bg(lambda: RD.render(
            work, frames, plate=_plate_fn(beauty), remove_alphas=lambda f: [fn(f) for fn in rm],
            keep_alphas=lambda f: [fn(f) for fn in keep], settings=st, progress=_progress, clean=clean,
            auto_clean=auto, inpaint=inpaint, linear=linear, cancel_check=_cancel_check,
            keep_names=cu.names_of(keep), plate_file=lambda f: cleanup["frame_file"](beauty, int(f)),
            ext=cu.output_ext(linear, cv2.imread(cleanup["frame_file"](beauty, int(frames[0])), cv2.IMREAD_UNCHANGED))))
        out = Path(rj["dir"])
        store.save_render_mark(work, out.name, overrides=ov, sig=sig, skipped=skipped)
        plates = cleanup["set_cleaned"](out, rj.get("comp_ext") or rj.get("ext") or ("exr" if linear else "png"))
        c3 = rj.get("c3d") or {}
        skip_txt = cu.frame_ranges(skipped, "") if skipped else ""
        notes = list(rj.get("notes") or [])
        if skipped:
            notes.insert(0, f"Kamera 3D nie ma klatek {skip_txt} zakresu pracy – pominięte ({len(skipped)})")
        notes.extend(held["warnings"])
        return {"success": True, "version": out.name, "dir": rj["dir"], "plate": cleanup["plate"], "plates": plates,
                "held": held if held["objects"] else None,
                "sources": info, **{k: rj.get(k) for k in ("counts", "backend", "seconds", "n_frames")},
                "notes": notes, "skipped": skipped, "skipped_text": skip_txt,
                "warnings": list((rj.get("kept_check") or {}).get("warnings") or []),
                "device": c3.get("device"), "motion_blur": c3.get("motion_blur"),
                "grain": RD.GR.summary(c3.get("frames"))}

    @router.get("/api/cleanup3d/view/{kind}/{frame}")
    def c3_view(kind: str, frame: int, card: Optional[str] = None, max_side: Optional[int] = None):
        """PNG preview of plate frame `frame`: `key` = the key of `card` shown at that frame (its own key, else the
        nearest one – header X-C3D-Key-Frame), `proj` = the keys projected onto the frame, `cards` = the cards' extents
        and grids over the plate – the cards of the scene (P16: without the cards turned off or outside the solo, as the
        render). Cached per (kind, frame, card, size, state of camera / cards / keys)."""
        work, manifest = _need()
        if kind not in T.VIEWS:
            raise HTTPException(status_code=400, detail=f"Nieznany podgląd: {kind}")
        if kind == "key":
            if not card:
                raise HTTPException(status_code=400, detail="Podgląd klucza wymaga karty (card)")
            _card(work, card)
        from engine.cleanup3d import render as RD
        w, h = plate_size()
        side = int(max_side) if max_side else max(256, int(round(max(w, h) * float(proxy_scale() or 1.0))))
        linear = _linear(manifest)
        ck = (str(work), store.scene_stamp(work), kind, int(frame), card if kind == "key" else None, side)
        with lock:
            got = views.get(ck)
            if got is not None:
                views.move_to_end(ck)
                return Response(content=got[0], media_type="image/png", headers=got[1])

            def run():
                sc = _scene(work)
                head = {"Cache-Control": "no-store"}
                if kind == "key":
                    ks = sc.keys_by_card.get(card) or []
                    if not ks:
                        raise HTTPException(status_code=404, detail=f"Karta {card} nie ma jeszcze kluczy")
                    kf = min(ks, key=lambda k: (abs(k.frame - int(frame)), k.frame)).frame
                    head["X-C3D-Key-Frame"] = str(kf)
                    return RD.view_key(sc, card, kf, linear=linear, max_side=side), head
                if int(frame) not in sc.camera.R:
                    raise ValueError(f"Kamera 3D nie ma klatki {int(frame)}")
                pat = (manifest.get("plates") or {}).get("beauty") or ""
                plate = cleanup["read_plate"](cleanup["frame_file"](pat, int(frame))) if pat else None
                if plate is not None and plate.shape[:2] != (sc.size[1], sc.size[0]):
                    plate = None
                if kind == "proj":
                    return RD.view_projection(sc, int(frame), plate=plate, linear=linear, max_side=side), head
                return RD.view_cards(sc, int(frame), plate=plate, linear=linear, max_side=side), head
            img, head = _guard(run)
            ok, buf = cv2.imencode(".png", img, PNG_FAST)
            if not ok:
                raise HTTPException(status_code=500, detail="Nie udało się zapisać podglądu PNG")
            views[ck] = (buf.tobytes(), head)
            while len(views) > VIEW_CACHE:
                views.popitem(last=False)
            return Response(content=views[ck][0], media_type="image/png", headers=head)

    @router.put("/api/cleanup3d/render_settings")
    def c3_render_settings(req: RenderSettingsRequest):
        """Save the patch blend of C5 (used by the next render; the live preview applies it in the browser)."""
        work = _work()
        return {"render_settings": _guard(lambda: store.save_render_settings(work, req.model_dump(exclude_none=True)))}

    blend_arrays: OrderedDict = OrderedDict()   # maps + scale for the band lines

    def _blend_key(work: Path, frame: int, side: int, s: Dict[str, Any]) -> Tuple:
        from engine import cleanup as cu
        rs = store.load_render_settings(work)
        from engine.cleanup3d import horizon as HZ
        return (str(work), store.scene_stamp(work), "blend", int(frame), side, float(cu.band_at(s, frame)[0]),
                float(rs["key_grow_px"]), tuple(s["remove"]),
                tuple(rs[k] for k in HZ.DEFAULTS), store._stat(HZ.lines_file(work)))   # P15: horizon match

    def _blend_compute(work: Path, manifest: Dict[str, Any], frame: int, side: int):
        """(fill, maps, scale) of render.blend_inputs for frame `frame` – PNGs cached in `views`, the maps array in
        `blend_arrays` (the band lines read it). Call under `lock`."""
        from engine import cleanup as cu
        from engine.cleanup3d import render as RD
        w, h = plate_size()
        s = cu.load_settings(work)
        rs = store.load_render_settings(work)
        rm, _keep = cleanup["alpha_loaders"](s, w, h)

        def run():
            sc = _scene(work)
            if int(frame) not in sc.camera.R:
                raise ValueError(f"Kamera 3D nie ma klatki {int(frame)}")
            plate = _plate_fn(_beauty(manifest))(int(frame))
            al = [np.asarray(fn(int(frame)), np.float32) for fn in rm]
            return RD.blend_inputs(sc, int(frame), plate, np.max(np.stack(al), axis=0), grow_px=float(cu.band_at(s, frame)[0]),
                                   key_grow_px=float(rs["key_grow_px"]), linear=_linear(manifest), max_side=side,
                                   settings=rs)
        fill, maps, scale = _guard(run)
        ck = _blend_key(work, frame, side, s)
        head = {"Cache-Control": "no-store", "X-C3D-Scale": f"{scale:.6f}", "Access-Control-Expose-Headers": "X-C3D-Scale"}
        for name, img in (("fill", fill), ("maps", maps)):
            ok, buf = cv2.imencode(".png", img, PNG_FAST)
            if not ok:
                raise HTTPException(status_code=500, detail="Nie udało się zapisać podglądu PNG")
            views[ck + (name,)] = (buf.tobytes(), head)
        while len(views) > VIEW_CACHE:
            views.popitem(last=False)
        blend_arrays[ck] = (maps, float(scale))
        while len(blend_arrays) > 8:
            blend_arrays.popitem(last=False)
        return fill, maps, scale

    def _blend_maps(work: Path, manifest: Dict[str, Any], frame: int, side: int) -> Tuple[np.ndarray, float]:
        from engine import cleanup as cu
        ck = _blend_key(work, frame, side, cu.load_settings(work))
        got = blend_arrays.get(ck)
        if got is not None:
            return got
        _f, maps, scale = _blend_compute(work, manifest, frame, side)
        return maps, float(scale)

    @router.get("/api/cleanup3d/blend/{frame}")
    def c3_blend(frame: int, part: str = "fill", max_side: Optional[int] = None):
        """Inputs of the live blend preview of C5 at preview size (render.blend_inputs): `part=fill` = the keys
        projected onto the frame (plate where they do not cover), `part=maps` = B difference plate vs keys, G the object
        part of the patch hole (grown by max(grow_px, key_grow_px)), R full coverage; `part=band` = the patch band's
        outer width per pixel (8-bit, preview px, engine/cleanup3d/band.py – the band points' effect; header
        X-C3D-Band-Soft = the „Wtopienie łaty” value it was computed with, the browser scales it by the slider now).
        Header X-C3D-Scale = preview px per frame px. The cards of the scene (P16: without the cards turned off or
        outside the solo, as the render). Cached like the previews."""
        work, manifest = _need()
        if part not in ("fill", "maps", "band"):
            raise HTTPException(status_code=400, detail=f"Nieznana część podglądu wtopienia: {part}")
        from engine import cleanup as cu
        side = int(max_side) if max_side else 1600
        s = cu.load_settings(work)
        with lock:
            if part == "band":
                img, head = _band_map(work, manifest, int(frame), side)
                ok, buf = cv2.imencode(".png", img, PNG_FAST)
                if not ok:
                    raise HTTPException(status_code=500, detail="Nie udało się zapisać podglądu PNG")
                return Response(content=buf.tobytes(), media_type="image/png", headers=head)
            ck = _blend_key(work, frame, side, s) + (part,)
            got = views.get(ck)
            if got is None:
                _blend_compute(work, manifest, int(frame), side)
                got = views[ck]
            views.move_to_end(ck)
            return Response(content=got[0], media_type="image/png", headers=got[1])

    # ------------------------------------------------------------------ C5 patch band (P14, one band system)
    # The band points of the key patch: the editor is static/bandpoints.js (owner „c3d”), the edits engine/bandedit.py,
    # the settings engine/objband.cleanup_band, the rule engine/cleanup3d/band.py. Stored per project (c3d_band.json).
    def _band_ctx(work: Path, overrides: Optional[Dict[str, Any]] = None):
        """(band settings, removal-object loaders, cleanup settings) of the project."""
        from engine import cleanup as cu
        from engine.cleanup3d import band as C3B
        from engine.objband import cleanup_band
        w, h = plate_size()
        s = cu.load_settings(work)
        rs = {**store.load_render_settings(work), **{k: v for k, v in (overrides or {}).items() if v is not None}}
        rm, _keep = cleanup["alpha_loaders"](s, w, h)
        return cleanup_band(rs, C3B.load(work)), rm, s, rs

    def _band_core_at(rm):
        from engine.cleanup3d import band as C3B
        return lambda k: C3B.object_mask([fn(int(k)) for fn in rm])

    def _band_map(work: Path, manifest: Dict[str, Any], frame: int, side: int):
        from engine.cleanup3d import band as C3B
        maps, scale = _blend_maps(work, manifest, frame, side)
        b, rm, s, rs = _band_ctx(work)
        gh, gw = maps.shape[:2]
        fn = C3B.points_fn(b, _band_core_at(rm))
        pts = C3B.shift_points(fn(frame) if fn else None, 0.0, 0.0, scale)
        if pts:
            core = maps[..., 1] > 127
            o = np.asarray(C3B.out_widths(core, b.out_px * scale, pts), np.float32)
        else:
            o = np.full((gh, gw), b.out_px * scale, np.float32)
        head = {"Cache-Control": "no-store", "X-C3D-Scale": f"{scale:.6f}", "X-C3D-Band-Soft": f"{b.out_px:.4f}",
                "X-C3D-Band-Points": str(len(pts or [])),
                "Access-Control-Expose-Headers": "X-C3D-Scale, X-C3D-Band-Soft, X-C3D-Band-Points"}
        return np.clip(np.rint(o), 0, 255).astype(np.uint8), head

    def _band_state(work: Path, frame: int, b=None) -> Dict[str, Any]:
        from engine import cleanup as cu
        from engine.bandedit import state_points
        from engine.cleanup3d import band as C3B
        b0, rm, s, rs = _band_ctx(work)
        b = b if b is not None else b0
        defaults = {"out": int(round(b.out_px)), "in": 0}
        if not b.band_points:
            return {"frame": frame, "points": [], "defaults": defaults}
        core_at = _band_core_at(rm)
        obj = core_at(frame)
        core = C3B.grow(obj, max(float(cu.band_at(s, frame)[0]), float(rs["key_grow_px"]))) if obj is not None else None
        so, si = b.scales
        pos = C3B.positions(b, frame, core_at)
        return {"frame": frame, "points": state_points(b.band_points, frame, pos, core, so, si), "defaults": defaults,
                "tracking": False, "scale": {"out": round(so, 4), "in": round(si, 4)}}

    def _band_edit(fn):
        from engine.bandedit import BandEditError
        try:
            return fn()
        except BandEditError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)

    @router.get("/api/cleanup3d/band/params")
    def c3_band_params():
        """The patch band points as stored (+ the slider's reference) – what Ctrl+Z restores."""
        from engine.cleanup3d import band as C3B
        return C3B.load(_work())

    @router.put("/api/cleanup3d/band/params")
    def c3_band_params_put(req: C3BandParamsRequest):
        """Ctrl+Z of a patch band edit (static/bandpoints.js): the points and their reference exactly as given."""
        from engine.cleanup3d import band as C3B
        work = _work()
        C3B.save(work, {"band_points": [dict(p) for p in req.band_points], "band_ref_out": req.band_ref_out,
                        "band_ref_in": req.band_ref_in})
        return {"success": True}

    @router.get("/api/cleanup3d/band/{frame}")
    def c3_band_get(frame: int):
        """The patch band points in `frame` as the editor shows them (position carried by the removal objects' motion,
        snapped to the object part of the patch hole; widths scaled by „Wtopienie łaty”)."""
        work = _work()
        return _guard(lambda: _band_state(work, int(frame)))

    @router.put("/api/cleanup3d/band/{frame}")
    def c3_band_put(frame: int, req: C3BandPointRequest):
        """Add a patch band point in `frame` or set its key there (position / outer width); the next render uses it."""
        from engine.bandedit import put_point
        from engine.bandmap import band_points
        from engine.cleanup3d import band as C3B
        from engine.objband import cleanup_band
        work = _work()
        params = C3B.load(work)
        b, _rm, _s, rs = _band_ctx(work)
        if not params.get("band_ref_out"):     # the slider as it is now = scale 1 for these points
            params = {**params, "band_ref_out": b.out_px, "band_ref_in": 0.0}
        b = cleanup_band(rs, params)
        state = _guard(lambda: _band_state(work, int(frame), b))
        so, si = b.scales
        pts, pid = _band_edit(lambda: put_point(band_points(params), int(frame), {q["id"]: q for q in state["points"]},
                                                state["defaults"], so, si, id=req.id, x=req.x, y=req.y, out=req.out,
                                                in_=req.in_, ang=req.ang))
        C3B.save(work, {**params, "band_points": pts})
        return {"success": True, "point": pid, "band": _guard(lambda: _band_state(work, int(frame)))}

    @router.get("/api/cleanup3d/band/{frame}/lines")
    def c3_band_lines(frame: int, key_soft_px: Optional[float] = None, key_effects: Optional[bool] = None,
                      key_effects_thr: Optional[float] = None, max_side: Optional[int] = None):
        """The preview lines of the patch band in `frame` (engine/bandsys.area_lines via band.lines, plate px): the band
        edges and where the outer feather ends, on the preview's hole (G + the traces by the browser's rule) – the
        palette's values not saved yet come as the query (static/bandpoints.js drawLines)."""
        from engine.cleanup3d import band as C3B
        from engine.cleanup3d import render as RD
        work, manifest = _need()
        side = int(max_side) if max_side else 1600
        with lock:
            maps, scale = _blend_maps(work, manifest, int(frame), side)
            b, rm, _s, rs = _band_ctx(work, {"key_soft_px": key_soft_px})
            eff = rs["key_effects"] if key_effects is None else bool(key_effects)
            thr = rs["key_effects_thr"] if key_effects_thr is None else float(key_effects_thr)
            hole = C3B.preview_hole(maps, eff, thr, RD.BLEND_DIFF_MAX)
            fn = C3B.points_fn(b, _band_core_at(rm))
            pts = C3B.shift_points(fn(int(frame)) if fn else None, 0.0, 0.0, scale)
            L = C3B.lines(hole, maps[..., 1] > 127, b.out_px * scale, b.outer_feather_px * scale, pts)
        k = 1.0 / max(scale, 1e-9)
        return {key: [[[round(x * k, 1), round(y * k, 1)] for x, y in c] for c in v] for key, v in L.items()}

    @router.delete("/api/cleanup3d/band/{point_id}")
    def c3_band_delete_point(point_id: str, frame: Optional[int] = None):
        """Remove a patch band point, or only its key in `frame` (?frame=N; the last key removes the point)."""
        from engine.bandedit import delete_point
        from engine.cleanup3d import band as C3B
        work = _work()
        params = C3B.load(work)
        pts = _band_edit(lambda: delete_point(params["band_points"], point_id, frame))
        C3B.save(work, {**params, "band_points": pts})
        return {"success": True}

    @router.delete("/api/cleanup3d/band")
    def c3_band_delete_keys(frame: Optional[int] = None):
        """The patch band keys of ALL points in `frame` (?frame=N), or every point (no frame = the plain ramp again)."""
        from engine.bandedit import delete_keys
        from engine.cleanup3d import band as C3B
        work = _work()
        params = C3B.load(work)
        pts, removed = _band_edit(lambda: delete_keys(params["band_points"], frame))
        C3B.save(work, {**params, "band_points": pts})
        return {"success": True, "removed": removed, "points": [p["id"] for p in pts]}

    # ------------------------------------------------------------------ C5 horizon match (P15)
    # The plate's horizon is detected automatically (engine/cleanup3d/horizon.py) or drawn by the user with the one
    # curve editor (static/curves.js) in a frame; lines stored per project (<work>/inpaint/c3d_horizon.json).
    @router.get("/api/cleanup3d/horizon")
    def c3_horizon():
        """The user's horizon lines {frame: curve} and the horizon settings of the render."""
        from engine.cleanup3d import horizon as HZ
        work = _work()
        lines = _guard(lambda: HZ.load_lines(work))
        rs = store.load_render_settings(work)
        return {"lines": {str(f): c for f, c in sorted(lines.items())}, "drawn": sorted(lines),
                "settings": HZ.settings(rs)}

    @router.get("/api/cleanup3d/horizon/{frame}")
    def c3_horizon_frame(frame: int, max_side: Optional[int] = None):
        """The horizon of plate frame `frame` measured on the live preview's grid: the cards' predicted horizon, the
        plate's line (the user's / automatic), the projection's line (plate px), the offset and the step before / after
        the correction (px) – measured whether or not „Dopasuj horyzont” is on."""
        from engine import cleanup as cu
        from engine.cleanup3d import horizon as HZ
        work, manifest = _need()
        side = int(max_side) if max_side else 1600
        w, h = plate_size()
        s = cu.load_settings(work)
        rs = store.load_render_settings(work)
        rm, _keep = cleanup["alpha_loaders"](s, w, h)
        with lock:
            def run():
                sc = _scene(work)
                if int(frame) not in sc.camera.R:
                    raise ValueError(f"Kamera 3D nie ma klatki {int(frame)}")
                plate = _plate_fn(_beauty(manifest))(int(frame))
                al = [np.asarray(fn(int(frame)), np.float32) for fn in rm]
                obj = np.max(np.stack(al), axis=0) if al else None
                return HZ.preview(sc, int(frame), plate, obj, rs, linear=_linear(manifest), max_side=side)
            return _guard(run)

    @router.put("/api/cleanup3d/horizon/{frame}")
    def c3_horizon_put(frame: int, req: HorizonLineRequest):
        """Save the user's horizon line of `frame` (open curve, plate px); between drawn frames the render interpolates."""
        from engine.cleanup3d import horizon as HZ
        work = _work()
        lines = _guard(lambda: HZ.save_line(work, int(frame), req.curve))
        return {"success": True, "drawn": sorted(lines)}

    @router.delete("/api/cleanup3d/horizon/{frame}")
    def c3_horizon_delete(frame: int):
        """Remove the user's horizon line of `frame` (the automatic detector takes the frame again)."""
        from engine.cleanup3d import horizon as HZ
        work = _work()
        lines = _guard(lambda: HZ.save_line(work, int(frame), None))
        return {"success": True, "drawn": sorted(lines)}

    # ------------------------------------------------------------------ layered card timeline (P16)
    # One row per card under the frame ruler (static/cleanup3d.js): keys as diamonds (drag = move), eye (Card.off) /
    # solo (Card.solo) stored on the card like After Effects (types.active_cards: the render, the previews, the coverage
    # and the Nuke export), a coverage bar per card, the patch band keys (P14).
    timeline_cache: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()

    @router.get("/api/cleanup3d/timeline")
    def c3_timeline(side: Optional[int] = None):
        """Numbers of the timeline rows for the work range: per card per frame `share` (mean weight the card fills in
        the frame) and `miss` (part of what the card shows that its keys do not fully cover – the render falls back to
        other sources there), computed on a coarse grid (engine/cleanup3d/timeline.py, cached by the scene stamp);
        `band_keys` = frames of the patch band keys; `off` = cards turned off for good; `solo` = soloed cards (any =
        only they, when not off, are in the scene). Without a camera or cards: empty coverage with `reason`."""
        from engine.cleanup3d import timeline as TL
        work = _work()
        cards = _guard(lambda: store.load_cards(work))
        off = [c.id for c in cards if c.off]
        base = {"band_keys": TL.band_keys(work), "off": off, "solo": [c.id for c in cards if c.solo]}
        frames = [int(f) for f in work_frames()]
        sd = int(side) if side else TL.COVER_SIDE
        rs = store.load_render_settings(work)
        ck = (str(work), store.scene_stamp(work), tuple(frames), sd, json.dumps(rs, sort_keys=True, default=str))
        with lock:
            got = timeline_cache.get(ck)
            if got is None:
                try:
                    sc = _scene(work)
                except ValueError as e:                  # no camera / no cards / every card off / out of solo
                    return {**base, "frames": [], "cards": {}, "step": None, "reason": str(e)}
                got = TL.coverage(sc, frames, side=sd, settings=rs)
                timeline_cache[ck] = got
                while len(timeline_cache) > 4:
                    timeline_cache.popitem(last=False)
            timeline_cache.move_to_end(ck)
        return {**base, **got, "reason": ""}

    @router.post("/api/cleanup3d/keys/{card}/{frame}/move")
    async def c3_key_move(card: str, frame: int, req: KeyMoveRequest):
        """Move a key to frame `to` (timeline drag): the new key = the target plate frame + the source key's active
        version projected through the 3D camera onto the target, only in the target's removal mask (timeline.move_key);
        the old key is deleted for good. 409: a master key, a key already at the target, no camera frame."""
        work, manifest = _need()
        _key(work, card, frame)
        from engine.cleanup3d import timeline as TL
        plate = _plate_fn(_beauty(manifest))
        mask = _remove_mask_fn(work)

        def run():
            with lock:
                res = TL.move_key(_scene(work), work, card, int(frame), int(req.to), plate=plate, remove_mask=mask)
                _drop_scene()                            # the keys changed (the stamp would say so too)
            return res
        res = await _bg(run)
        k = res["key"]
        return {"success": True, "key": None if k is None else k.to_json(), "from": int(frame), "to": int(req.to),
                "patched": res["patched"], "covered": res["covered"],
                "freed_bytes": res["freed_bytes"]}

    @router.put("/api/cleanup3d/cards/{card}/off")
    def c3_card_off(card: str, req: CardOffRequest):
        """Turn a card off for good (left out of the render, the previews and the Nuke export – the result becomes
        stale) or back on. Only this field of the card changes."""
        work = _work()
        _card(work, card)

        def run():
            cards = store.load_cards(work)
            c = next(x for x in cards if x.id == card)
            c.off = bool(req.off)
            store.save_cards(work, cards)
            return c.to_json()
        return {"success": True, "card": _guard(run)}

    @router.put("/api/cleanup3d/cards/{card}/solo")
    def c3_card_solo(card: str, req: CardSoloRequest):
        """Solo a card or take its solo back (P16, like After Effects): when any card is soloed, only the soloed cards
        that are not off go into the render, the previews and the Nuke export (the result becomes stale). Only this
        field of the card changes; turning every solo off gives the old render signature."""
        work = _work()
        _card(work, card)

        def run():
            cards = store.load_cards(work)
            c = next(x for x in cards if x.id == card)
            c.solo = bool(req.solo)
            store.save_cards(work, cards)
            return c.to_json()
        return {"success": True, "card": _guard(run)}

    # ------------------------------------------------------------------ C6 export to Nuke (package P11)
    def _last_patch(work: Path, rs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The patch sequence of the last projection (its alpha = the removal mask), None without a projection."""
        if not rs:
            return None
        pdir = T.root_dir(work) / str(rs["version"]) / "patch"
        files = sorted(pdir.glob("p_[0-9][0-9][0-9][0-9].*"))
        if not files:
            return None
        ext = files[0].suffix
        fr = [int(p.stem[2:]) for p in files if p.suffix == ext]
        return {"pattern": str(pdir / f"p_####{ext}"), "first": min(fr), "last": max(fr), "version": rs["version"]}

    @router.post("/api/cleanup3d/export_nuke")
    def c3_export_nuke(req: Optional[NukeExportRequest] = None):
        """Write a Nuke script rebuilding the projection (Camera3 + .chan, cards as Card3D / sky Sphere, keys through
        Project3D, lens STMaps, Merge over the plate through the removal mask) -> <work>/inpaint/nuke/c3d_<v### or date>.nk."""
        work, manifest = _need()
        from engine.cleanup3d import nuke as NU
        beauty = _beauty(manifest)
        rs = store.render_state(work)
        frames = [int(f) for f in req.frames] if req and req.frames else None
        tag = rs["version"] if rs and not rs["stale"] else None      # a stale projection: the date, not its version
        out = _guard(lambda: NU.export(work, plate_pattern=beauty, tag=tag, patch=_last_patch(work, rs), frames=frames))
        if rs and rs["stale"]:
            out["notes"].append(f"Maska usuwania z nieaktualnej projekcji {rs['version']} – policz C5 ponownie.")
        return {"success": True, **out, "render": rs}

    return router, {"drop_scene": _drop_scene}
