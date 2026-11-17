# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate cleaning – the „Czyszczenie” mode (MVP, user decision 2026-11-06; engine/cleanup.py)

    GET  /api/cleanup               settings, the objects with their versions, what can fill (clear plate / auto / LaMa),
                                    the versions made so far
    PUT  /api/cleanup               {remove?, sources?, grow_px?, soft_px?, holdout?, holdout_skip_lost?, holdout_off?, fill?}
                                    -> <work>/inpaint/cleanup.json
    GET  /api/cleanup/mask/{f}      proxy PNG: the removal mask of frame f (255 = removed, 0 = plate) – the preview
    POST /api/cleanup/run           heavy job: the work area -> <work>/inpaint/v###/ + plate „cleaned” = its comp
                                    (A/B, wipe, difference against the beauty in the viewer)
    POST /api/cleanup/versions/{v}/recompose     „⟲ Złóż ponownie”: a new version from v's stored model frames with
                                    the current composite band (no model run; engine/inpaint_recompose.py)
    GET  /api/cleanup/versions/{v}/preview/{f}?grow_px&soft_px   live preview PNG of that recomposite (preview size)
    GET  /api/cleanup/holdout/{obj}/{f}?grow_px&soft_px   palette „Złożenie”: a kept object's holdout alpha (R) and
                                    where it cuts the patch (G), PNG at preview size (2026-11-15)

Built by `make(*, ...)` like the other api/ routers (functions passed as lambdas, nothing imported from app.py).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from engine import cleanup as cu

CLEANED_PLATE = "cleaned"


class PromptCheckRequest(BaseModel):   # typed in the palette, not saved yet (None = the saved value)
    prompt: Optional[str] = None
    prompt_mode: Optional[str] = None
    h3_prompt: Optional[str] = None


class PromptAssistRequest(BaseModel):
    text: str = ""                        # what the user wrote (any language)
    mode: Optional[str] = None            # remove | replace (None = the saved one)
    transition: bool = False              # also a clip prompt of the change over time
    frame: Optional[int] = None           # the frame the assistant looks at (None = the main anchor's / the current)


class CleanupRequest(BaseModel):
    remove: Optional[List[str]] = None
    sources: Optional[Dict[str, str]] = None
    grow_px: Optional[float] = None
    soft_px: Optional[float] = None
    holdout: Optional[bool] = None
    holdout_skip_lost: Optional[bool] = None   # no holdout in the frames where a kept object looks lost (default off)
    holdout_off: Optional[List[str]] = None    # kept objects NOT protected (palette „Złożenie”, 2026-11-15)
    fill: Optional[str] = None
    model: Optional[str] = None          # lama | qwen21 | void (engine/cleanup.MODELS; MiniMax-Remover dropped 2026-11-09)
    prompt: Optional[str] = None         # Qwen-Image-2.1 / VOID: what should be in the area
    prompt_mode: Optional[str] = None    # remove | replace
    anchor: Optional[str] = None         # MiniMax-H3: none | frame (one Qwen-Image-2.1 look frame)
    anchor_frame: Optional[int] = None   # the anchor's plate frame; -1 = back to auto
    seed: Optional[int] = None           # MiniMax-H3 + its anchors
    h3_prompt: Optional[str] = None      # MiniMax-H3 clip prompt of a transition
    vace_res: Optional[int] = None       # Wan 2.2 Fun VACE canvas short edge: 720 | 480
    vace_speed: Optional[str] = None     # fast (Lightning LoRA, 4 steps) | quality (20 steps)
    link_method: Optional[str] = None    # linked anchors: warp_qwen | warp | qwen (engine/anchor_warp.py)
    input_plate: Optional[str] = None    # „Wejście”: "" = Beauty, "v###" = an earlier Inpaint version (2026-11-15)
    anchor_morph: Optional[bool] = None  # linked anchors: morph keyframe to keyframe (engine/h3_anchors.morphs, 2026-11-15)
    link_denoise: Optional[float] = None  # Qwen-Image-2.1 touch-up strength of a warped linked anchor
    anchor_denoise: Optional[float] = None  # „Siła kotwicy”: main / own anchors from the plate (< 1) or anew (1)
    anchor_raw: Optional[bool] = None    # the video model gets Qwen-Image-2.1's whole picture, not the pasted one
    anchor_ref_only: Optional[bool] = None   # the main anchor only as the whole-clip reference, no pinned frame
    tone: Optional[bool] = None          # C1 „wyrównanie poziomu łaty” (off by default since the 2026-11-13 test)
    h3_strength: Optional[float] = None  # MiniMax-H3 Replace: start sigma from the plate (1.0 = a hole)
    gen_margin_px: Optional[float] = None   # „Zapas generowania”: the model's mask = the removed alpha grown by this
    grain: Optional[str] = None          # C1 grain match of the video model's fill: auto | off | on (2026-11-13)
    grain_strength: Optional[float] = None   # multiplies the missing grain (0..2)


class AnchorAdd(BaseModel):
    frame: int
    kind: str = "linked"                 # linked | original | own | projection (the C5 render of the frame)


class AnchorProjection(BaseModel):
    step: int = 0                        # 0 = only the frames of the card keys; N = also every N-th frame of the range


class AnchorPatch(BaseModel):
    frame: Optional[int] = None
    kind: Optional[str] = None           # main | linked | own (detach) | original
    prompt: Optional[str] = None         # an own anchor's description
    gen: Optional[str] = None            # the chosen generation (g_###)


class AnchorImport(BaseModel):
    data: str                            # data:image/...;base64,… (PNG / JPEG / WebP)
    name: Optional[str] = None           # the file's name (kept in the generation's note)


class AnchorGen(BaseModel):
    prompt: Optional[str] = None         # the description shown in the panel (main: the shared one; own: its own)
    seed: Optional[int] = None           # None = a new random seed


def _read_plate(path: str) -> Optional[np.ndarray]:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    raw = raw[..., :3]
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    return raw.astype(np.float32)


def cleanup_target(objs: List[Dict[str, Any]], names: List[str]) -> List[str]:
    """ONE thing is cleaned at a time (user decision 2026-11-10): a whole group or one object outside any group. A member
    of a group means its group (to clean a member alone, take it out of the group). The last valid name wins."""
    by = {o["name"]: o for o in objs}
    for n in reversed(list(names or [])):
        o = by.get(n)
        if o is None:
            continue
        g = o.get("group") if o.get("kind") != "group" else None
        return [g] if g and g in by else [n]
    return []


def frame_file(pattern: str, f: int) -> str:
    return pattern.replace("####", f"{f:04d}").replace("%04d", f"{f:04d}")


def make(*, STATE, objects, plate_size, work_frames, proxy_scale, comfy_client, long_job, refresh_manifest):
    """Build the router. Keyword-only: the app state + functions (late binding). Returns (router, exports)."""
    router = APIRouter()

    def _need():
        if not STATE.get("work_dir") or not STATE.get("manifest"):
            raise HTTPException(status_code=400, detail="Brak otwartego projektu")
        return Path(STATE["work_dir"]), STATE["manifest"]

    def _objs() -> List[Dict[str, Any]]:
        return objects().list(groups=True)

    def _mask_of(o: Dict[str, Any], s: Dict[str, Any]) -> Dict[str, Any]:
        """The mask removed for `o` (engine/cleanup_masks.resolve): a group = its composition, a member = its „Do
        grupy z:”, an object without a group = its own choice in the palette."""
        from engine.cleanup_masks import resolve
        try:
            return resolve(objects(), o, {})   # the object's own mask (palette Obiekty, 2026-11-15) – no C1 override
        except Exception as e:  # noqa: BLE001 – a broken object is „no mask”, never a crash of the panel
            return {"run": None, "source": "", "valid": False, "by": "own", "why": f"sprawdzenie nie powiodło się: {e}"}

    def _related(objs: List[Dict[str, Any]], rem: List[str]) -> set:
        """The cleaned thing + its group / members – never in the holdout."""
        by = {o["name"]: o for o in objs}
        related = set(rem)
        for n in rem:
            o = by[n]
            related |= set(o.get("members") or []) if o.get("kind") == "group" else ({o["group"]} if o.get("group") else set())
        return related

    def _kept(s: Dict[str, Any], objs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """The kept objects of the cleaned thing – ONE list for the holdout loaders, the palette „Złożenie” (Ochrona)
        and the overlay (GET /api/cleanup/holdout): every object outside the cleaned thing / its group, groups
        themselves not (their members are listed one by one). [{name, group, mask (cleanup_masks.resolve), on}] –
        on = protected (the holdout on and the name not in `holdout_off`, user decision 2026-11-15)."""
        objs = _objs() if objs is None else objs
        rem = cleanup_target(objs, s.get("remove") or [])
        related = _related(objs, rem)
        off = set(s.get("holdout_off") or [])
        out = []
        for o in objs:
            if o["name"] in related or o.get("kind") == "group":
                continue
            m = _mask_of(o, s)
            out.append({"name": o["name"], "group": o.get("group"), "mask": m,
                        "on": bool(s.get("holdout", True)) and o["name"] not in off and bool(m["run"])})
        return out

    def _alpha_loaders(s: Dict[str, Any], w: int, h: int):
        """(remove loaders, keep loaders) – a removed group takes its members out of the holdout and vice versa. The keep
        loaders carry the object's name (`cu.Named`, `cu.names_of`) for the kept-object warnings; a kept object the
        user unticked (`holdout_off`, palette „Złożenie”) is not among them – the patch may cover it, and neither the
        model's mask nor the kept-object check sees it. The remove loaders hold the nearest mask where the object is
        lost for a few frames of the work range (engine/cleanup_hold.py)."""
        from engine import cleanup_hold as CH
        from engine.export import run_alpha_loader
        objs = _objs()
        by = {o["name"]: o for o in objs}
        rem = cleanup_target(objs, s["remove"])   # an older setting may list several / a member: one target
        if not rem:
            raise HTTPException(status_code=409, detail="Wybierz obiekt do usunięcia (klik w palecie Obiekty)")
        load_rm, load_keep = [], []
        for n in rem:
            m = _mask_of(by[n], s)
            if not m["run"]:
                raise HTTPException(status_code=409, detail=f"{n}: brak maski – {m['why']}")
            st = CH.stamp(objects(), by[n], m["run"])
            load_rm.append(CH.Hold(run_alpha_loader(objects().dir(n), m["run"], w, h), n,
                                   lambda: list(work_frames()), key=st and (st, m["run"], int(w), int(h)),
                                   alive=CH.life_fn(objects(), by[n])))
        off = set(s.get("holdout_off") or [])
        for k in _kept(s, objs):
            if k["mask"]["run"] and k["name"] not in off:
                load_keep.append(cu.Named(run_alpha_loader(objects().dir(k["name"]), k["mask"]["run"], w, h), k["name"]))
        return load_rm, load_keep

    def _held(rm, frames, cl=None, progress=None) -> Dict[str, Any]:
        """The removed objects' held frames (engine/cleanup_hold.held_report) of a run over `frames`; with a Cleaner:
        into its run.json (`held`) and its notes, next to the kept-object warnings."""
        from engine import cleanup_hold as CH
        rep = CH.held_report(rm, frames, progress)
        if cl is not None and rep["objects"]:
            cl.report["held"] = rep
            cl.notes.extend(x for x in rep["warnings"] if x not in cl.notes)
        return rep

    def _fill_sources(work: Path, manifest: Dict[str, Any], w: int, h: int, linear: bool):
        """(clean(f) or None, auto_clean(f) or None, info) – the clear plate only when its content check says the
        same scene (engine/plate_match.py) and the aligned version when measured."""
        from engine import plate_match as pm
        plates = manifest.get("plates") or {}
        beauty = plates.get("beauty") or ""
        info: Dict[str, Any] = {"clean": None, "auto": None}
        clean = None
        for name in ("cleanplate", "clean_plate"):
            pat = plates.get(name)
            if not pat:
                continue
            r = pm.cached(work, beauty, pat)
            if r and r.get("state") == "other":
                info["clean"] = f"{name}: inne ujęcie – pomijam"
                continue
            use = plates.get(f"{name}_align") or pat
            info["clean"] = f"{name}{' (wyrównany)' if use != pat else ''}"

            def _clean(f, use=use):
                img = _read_plate(frame_file(use, f))
                if img is None or img.shape[:2] != (h, w):
                    return None
                return img, (img.max(axis=2) > 0).astype(np.float32)   # the aligned plate's black border = no source
            clean = _clean
            break
        auto = None
        if not linear:   # cleanplate_auto is display-referred: only for 8-bit plates in the MVP
            from engine.cleanplate_auto import load_cleanplate
            got = load_cleanplate(work / "cleanplate_auto")
            if got is not None and got[0].shape[:2] == (h, w):
                img, conf, _ = got
                info["auto"] = "cleanplate_auto"
                auto = lambda f: (img, conf)   # noqa: E731
        return clean, auto, info

    def _lama_paths():
        """(ComfyUI client or None, LaMa node available, LaMa at full resolution available)."""
        from engine import lama_full
        from engine.cleanplate_auto import lama_available
        full = lama_full.available()
        try:
            client = comfy_client()
            node = bool(lama_available(client))
        except Exception:
            client, node = None, False
        return client, node, full

    # step C1 fill models (user 2026-11-10): what each brings, whether it is installed (the same gates as the Doctor /
    # Cleanup 3D), the GPU verdict on this card (engine/vram_fit) and the Doctor's install action
    MODEL_INFO = {
        "lama": {"label": "LaMa", "prompt": False, "download": "0,2 GB", "install": "lama",
                 "what": "szybkie wypełnienie klatka po klatce, bez opisu – małe i średnie obszary"},
        "qwen21": {"label": "Qwen-Image-2.1", "prompt": True, "download": "17,3 GB", "install": "qwen21",
                   "what": "edycja z opisem klatka po klatce – duże obszary; między klatkami może migotać (do ujęć z ruchem "
                           "lepsze klucze w C4)", "license": "Qwen Research License – tylko niekomercyjnie"},
        "h3": {"label": "MiniMax-H3", "prompt": True, "anchors": True, "download": "ok. 55 GB", "install": "",
               "what": "model wideo: regeneruje tylko obszar maski w całym klipie (1344×768 / 768×1344, 24 kl./s) – "
                       "spójne w czasie; Usuń albo Zastąp z opisem",
               "license": "MiniMax-H3 – licencja „other”, sprawdź przed pracą komercyjną"},
        "vace": {"label": "Wan 2.2 Fun VACE", "prompt": True, "anchors": True, "download": "23,6 GB z LoRA Lightning (+7 GB koder / VAE, gdy ich brak)",
                 "install": "vace",
                 "what": "model wideo Wan 2.2 Fun VACE A14B (GGUF Q4_K_S): inpaint maski w całym klipie (720p albo 480p), "
                         "Usuń albo Zastąp z opisem, kotwice Qwen-Image-2.1 jako klatki zadane – alternatywa dla MiniMax-H3 "
                         "na kartach 12 GB", "license": "Apache-2.0"},
        "ltx": {"label": "LTX-2.5", "prompt": True, "anchors": True,
                "download": "1,31 GB (IC-LoRA inpaintu; LTX-2.5 z etapu 7 ok. 40 GB, gdy go brak)", "install": "ltx_inpaint",
                "what": "model wideo LTX-2.5 22B z IC-LoRA in-/outpainting (Lightricks): cały klip naraz (dłuższy bok 1024), "
                        "zielony klucz pod maską, Usuń albo Zastąp z opisem sceny (nie polecenia edycji), kotwice "
                        "Qwen-Image-2.1 jako klatki kluczowe", "license": "LTX-2.x Community – darmowo do 10 mln USD przychodu/rok"},
        "void": {"label": "VOID", "prompt": True, "download": "11,6 GB (+9,8 GB T5, gdy go brak)", "install": "void",
                 "what": "usuwa obiekt z całego ujęcia naraz (1344×768) z opisem tła – spójne w czasie; wynik wklejany "
                         "tylko w masce", "license": "kod Apache-2.0; CogVideoX License – komercyjnie po bezpłatnej rejestracji"},
    }

    def _comfy_old_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """A model on disk whose node this RUNNING ComfyUI lacks: ComfyUI updated but not restarted yet (the update is
        still waiting for a restart – installer.pending_restart) -> „Zrestartuj ComfyUI”; else (a git install) the
        ComfyUI update; Desktop: no button (it updates in its app). User 2026-11-10: updated, the card still asked."""
        from engine import installer
        from engine.comfy_update import comfy_git_dir
        try:
            client = comfy_client()
            waiting = "comfy_update" in installer.pending_restart(getattr(client, "base_url", None))
        except Exception:  # noqa: BLE001
            waiting = False
        if waiting:
            return {"state": "comfy_restart", "install": "restart_comfy"}
        return {"state": "comfy_old", "install": "comfy_update" if comfy_git_dir(cfg) else ""}

    def _models_info() -> List[Dict[str, Any]]:
        from engine import vram_fit
        from engine.config import load_config
        from engine.stage_models import cleanup3d_engines, cleanup3d_tools
        cfg = load_config()
        try:
            card = vram_fit.card_gb()
        except Exception:  # noqa: BLE001
            card = None
        out = []
        for mid in (*cu.MODELS, "qwen21"):   # qwen21 = the H3 anchor model (not a C1 fill since 2026-11-10)
            info = dict(MODEL_INFO[mid], id=mid)
            try:
                if mid == "lama":
                    _, node, full = _lama_paths()
                    ok, why = bool(node or full), "Brak LaMa (big-lama.pt / węzły Inpaint) – Doktor → Zainstaluj"
                elif mid == "qwen21":
                    from engine.stage_models import comfy_gate
                    g = cleanup3d_engines(cfg).get("qwen21_edit") or {}
                    ok, why = bool(g.get("ok")), str(g.get("reason") or "")
                    if ok:   # the files are there – and the node in this ComfyUI? (2026-11-10)
                        why = comfy_gate(comfy_client(), "qwen21_edit")
                        ok = not why
                        if why:   # installed, ComfyUI too old: the card offers the ComfyUI update, not the download
                            info.update(_comfy_old_state(cfg))
                elif mid == "h3":
                    from engine import h3_fill
                    st = h3_fill.status(cfg, comfy_client())
                    ok, why = st["ok"], st["reason"]
                    if not ok and all(st["files"].get(k) for k in ("unet", "clip", "vae")):
                        info.update(_comfy_old_state(cfg))   # files there, nodes missing
                elif mid == "vace":
                    from engine import vace_fill
                    st = vace_fill.status(cfg, comfy_client())
                    ok, why = st["ok"], st["reason"]
                    info["lightning"] = vace_fill.has_lightning(st["files"])
                    if not ok and all(st["files"].values()):
                        gguf_on_disk = bool(cfg.get("comfy_dir")) and any(
                            (Path(str(cfg["comfy_dir"])) / "custom_nodes" / n).is_dir() for n in ("ComfyUI-GGUF", "comfyui-gguf"))
                        if "ComfyUI-GGUF" in why and gguf_on_disk:
                            # installed, ComfyUI not restarted yet (user 2026-11-11: the Doctor said ok, the panel
                            # „not installed”): the card offers the restart, not the download
                            info.update(state="comfy_restart", install="restart_comfy")
                            why = "ComfyUI-GGUF zainstalowany – zrestartuj ComfyUI, żeby wczytał węzeł UnetLoaderGGUF"
                        elif "ComfyUI-GGUF" not in why:
                            info.update(_comfy_old_state(cfg))   # files and GGUF node there, a native node missing
                elif mid == "ltx":
                    from engine import ltx_fill
                    st = ltx_fill.status(cfg, comfy_client())
                    ok, why = st["ok"], st["reason"]
                    if not ok and all(st["files"].values()):
                        info.update(_comfy_old_state(cfg))   # files there, a native LTX node missing
                    elif not ok and not all(st["files"].get(k) for k in ltx_fill.BASE_KEYS):
                        info["install"] = "ltx"   # the base LTX-2.5 files first (stage 7's install)
                else:
                    g = cleanup3d_tools(cfg).get("void") or {}
                    ok, why = bool(g.get("ok")), str(g.get("reason") or "")
            except Exception as e:  # noqa: BLE001 – a broken check = not available, never a crash of the panel
                ok, why = False, f"sprawdzenie nie powiodło się: {e}"
            info.update(available=ok, reason="" if ok else why,
                        vram=vram_fit.assess("ltx_fill" if mid == "ltx" else mid, card) if mid != "lama" else None)
            if mid == "h3" and not ok and info.get("state") not in ("comfy_old", "comfy_restart"):
                info["install"] = ""   # no Doctor install (about 55 GB, the user downloads it): the card shows the reason
            out.append(info)
        return out

    def _upscale_state() -> Dict[str, Any]:
        """SeedVR2 for „⤴ Upscale” of a version (installed in the Doctor for Cleanup 3D, the same tool)."""
        try:
            from engine.config import load_config
            from engine.stage_models import cleanup3d_tools
            g = cleanup3d_tools(load_config()).get("seedvr2") or {}
            return {"available": bool(g.get("ok")), "reason": str(g.get("reason") or "")}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": str(e)}

    def _inpainter(work: Path, s: Dict[str, Any]):
        """The chosen model's fill (None = not available -> OpenCV, said in run.json): LaMa at the crop's full resolution
        (engine/lama_full.py) with the ComfyUI node (256 x 256) as the fallback – `cu.LamaInpainter`; Qwen-Image-2.1 edit
        with the prompt – `cu.QwenInpainter` (VOID: LaMa here, for what its frames leave)."""
        if s.get("model") == "qwen21":
            return cu.QwenInpainter(comfy_client(), work / "comfy_io", s.get("prompt") or "", s.get("prompt_mode") or "remove")
        client, node, full = _lama_paths()
        if not (node or full):
            return None
        return cu.LamaInpainter(client if node else None, work / "comfy_io", full=full)

    def _set_cleaned(out: Path, ext: str):
        """The comp of version `out` as plate „cleaned” (project.json, never the manifest): the viewer's A/B / wipe /
        difference work on it. Returns the refreshed plate list (None without a project file)."""
        proj = STATE.get("shot_project")
        plates = None
        if proj is not None:
            proj.set_plate(CLEANED_PLATE, str(out / "comp" / f"c_####.{ext}").replace("\\", "/"))
            plates = (refresh_manifest() or {}).get("plates")
        return plates

    @router.get("/api/cleanup")
    def get_cleanup():
        work, manifest = _need()
        s = cu.load_settings(work)
        if s["model"] not in cu.C1_MODELS:   # LaMa / VOID left Inpaint (2026-11-11): the object goes on with MiniMax-H3
            s["model"] = cu.c1_model(s["model"])
            s = cu.save_settings(work, s)
        w, h = plate_size()
        beauty = (manifest.get("plates") or {}).get("beauty") or ""
        linear = beauty.lower().endswith(".exr")
        _, _, info = _fill_sources(work, manifest, w, h, linear)
        from engine.cleanup_masks import available
        objs = []
        for o in _objs():
            runs_dir = objects().dir(o["name"]) / "runs"
            runs = sorted(p.name for p in runs_dir.glob("v[0-9][0-9][0-9]")) if runs_dir.is_dir() else []
            standalone = o.get("kind") != "group" and not o.get("group")
            objs.append({"name": o["name"], "kind": o.get("kind") or "object", "group": o.get("group"),
                         "active_run": (o.get("tool") or {}).get("active_run"), "runs": runs,
                         "has_core": (objects().dir(o["name"]) / "core").is_dir(),
                         # the mask removed for it + (an object without a group) what it can pick in the palette
                         "mask": _mask_of(o, s), "available": available(objects(), o["name"]) if standalone else []})
        try:
            _, node, full = _lama_paths()
        except Exception:
            node, full = False, False
        lama = node or full
        shown = str((manifest.get("plates") or {}).get(CLEANED_PLATE) or "").replace("\\", "/")
        vers = cu.versions(work)
        for v in vers:   # the version the plate „cleaned” shows now
            v["shown"] = f"/{v['version']}/comp/" in shown
        # the composite band against what the model draws (user 2026-11-13): generation margin + the engine's grow per
        # engine (plate px) – the palette says when the band reaches beyond and is cut there
        reach = {mid: round(cu.engine_reach_px(mid, s, w, h), 1) for mid in cu.C1_MODELS}
        return {"settings": s, "objects": objs, "versions": vers, "fills": list(cu.FILLS),
                "kept": _kept(s),   # palette „Złożenie” → Ochrona: the kept objects, their masks, protected or not
                "engine_reach": reach, "band_limit": cu.band_limit(s, w, h),
                "models": list(cu.C1_MODELS),
                **(lambda mi: {"model_info": [m for m in mi if m["id"] in cu.C1_MODELS],
                               "anchor_model": next((m for m in mi if m["id"] == "qwen21"), None),
                               "upscale": _upscale_state()})(_models_info()),
                "sources": {**info, "lama": lama, "lama_full": full},
                "out_dir": str(cu.cleanup_dir(work)).replace("\\", "/"),
                "plate": CLEANED_PLATE}

    @router.put("/api/cleanup")
    def put_cleanup(req: CleanupRequest):
        work, _ = _need()
        cur = cu.load_settings(work)
        old = (cur.get("remove") or [None])[0]
        body = {k: v for k, v in req.model_dump().items() if v is not None}
        if "model" in body and body["model"] not in cu.C1_MODELS:
            raise HTTPException(status_code=400, detail="Silnik Inpaint: MiniMax-H3 albo Wan 2.2 Fun VACE")
        new_remove = body.pop("remove", None)
        cur.update(body)                                    # fields of the thing shown when they were edited
        shown = None
        if new_remove is not None:
            cur["remove"] = cleanup_target(_objs(), new_remove)
            new = (cur["remove"] or [None])[0]
            cu.switch_target(cur, old, new)   # its own model / description / anchors
            if new != old:
                shown = _show_target(work, new)
        else:
            cur["remove"] = cleanup_target(_objs(), cur["remove"])
        out = {"settings": cu.save_settings(work, cur)}
        if shown is not None:
            out.update(shown)
        return out

    def _show_target(work: Path, name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Another thing picked (user 2026-11-15): the plate „cleaned” = ITS newest version (settings untouched – the
        thing's own are restored by switch_target); none computed = the plate „cleaned” goes, the viewer shows the
        original plate. Returns {plates, version} (version None = no version), None without a project file."""
        if STATE.get("shot_project") is None:
            return None
        mine = [v for v in cu.versions(work) if name and name in (v.get("remove") or [])]
        if mine:
            v = mine[-1]
            plates = _set_cleaned(cu.cleanup_dir(work) / v["version"], v.get("comp_ext") or cu.file_ext(v.get("ext") or "png"))
            return {"plates": plates, "version": v["version"]}
        STATE["shot_project"].remove_plate(CLEANED_PLATE)
        return {"plates": (refresh_manifest() or {}).get("plates"), "version": None}

    def _removed_noun(s: Dict[str, Any]) -> Optional[str]:
        """The removed thing as a noun for Qwen-Image-2.1's anchor instruction (user 2026-11-15: an earring stayed –
        the edit model gets the real image, nothing marked, and removes only what it is told): the objects' SAM 3.1
        text prompts (a group: its members'), else their labels; None for Replace or nothing picked."""
        if (s.get("prompt_mode") or "remove") == "replace":
            return None
        names = _removed_names(s)
        out: List[str] = []
        for n in names:
            try:
                o = objects().get(n)
            except Exception:  # noqa: BLE001 – a missing object: its label
                o = {}
            if o.get("kind") == "group":
                continue
            ps = [str(p).strip() for p in ((o.get("detect") or {}).get("prompts") or []) if str(p).strip()]
            for t in ps or [n.replace("_", " ").strip().lower()]:
                if t and t not in out:
                    out.append(t)
        return " and ".join(out) or None

    def _removed_names(s: Dict[str, Any]) -> List[str]:
        """The labels of the cleaned thing (a group: the group + its members) – what a REMOVE description must not
        carry (engine/prompt_rules.strip_named / lint `object_named`)."""
        out = [str(n) for n in (s.get("remove") or [])]
        try:
            out += [o["name"] for o in _objs() if o.get("group") in out and o["name"] not in out]
        except Exception:  # noqa: BLE001 – no object list: the saved names alone
            pass
        return out

    @router.post("/api/cleanup/prompt/check")
    def cleanup_prompt_check(req: PromptCheckRequest):
        """The description's warnings + the exact texts each model gets (engine/prompt_rules; user 2026-11-12). The
        fields of the request = what is typed and not saved yet (None = the saved value)."""
        from engine import prompt_rules as PR
        work, _ = _need()
        s = dict(cu.load_settings(work))
        s.update({k: v for k, v in req.model_dump().items() if v is not None})
        fin = PR.final_prompts(s, _removed_noun(s))
        pm = s.get("prompt_mode") or "remove"
        return {"lint": PR.lint(s.get("prompt"), pm, fin["transition"], names=_removed_names(s)),
                "clip_lint": PR.lint(s.get("h3_prompt"), pm, True) if fin["transition"] and not fin.get("morph") else [],   # a morph uses the description
                "final": fin}

    @router.get("/api/cleanup/prompt/assist")
    def cleanup_prompt_assist_state():
        """„✨ Popraw opis”: is the local assistant there (what the install popup shows when not)."""
        from engine import prompt_assist as PA
        from engine import vram_fit
        try:
            vr = vram_fit.assess("prompt_assist", vram_fit.card_gb())
        except Exception:  # noqa: BLE001
            vr = None
        return {"available": PA.available(), "label": PA.LABEL, "install": "prompt_assist",
                "download": f"{PA.DOWNLOAD_GB:.1f} GB", "license": PA.LICENSE, "page": PA.PAGE, "vram": vr,
                "what": "lokalny asystent opisu: tłumaczy na angielski i przepisuje opis według zasad MiniMax-H3 / "
                        "Wan 2.2 Fun VACE / Qwen-Image-2.1, patrząc na klatkę z zaznaczonym obiektem"}

    @router.post("/api/cleanup/prompt/assist")
    @long_job("Asystent opisu (Qwen3-VL-4B)")
    async def cleanup_prompt_assist(req: PromptAssistRequest):
        """The proposal of the local assistant for the typed text (nothing saved – the palette shows it, „Wstaw” saves)."""
        from engine import prompt_assist as PA
        from engine.cleanup3d.void_engine import to_display8
        from engine.jobs import JOBS
        work, manifest = _need()
        s = cu.load_settings(work)
        if not PA.available():
            raise HTTPException(status_code=409, detail=f"Brak modelu {PA.LABEL} – Doktor → Zainstaluj")
        mode = req.mode or s.get("prompt_mode") or "remove"
        img = msk = None
        try:   # the frame it looks at: the asked one, else the main anchor's, else the first of the work range
            frames, plate, rmask, linear = _anchor_ctx(work, manifest, s)
            from engine import h3_anchors as HA
            main = HA.main_of(s.get("anchors") or [])
            f = req.frame if req.frame in frames else (main["frame"] if main and main["frame"] in frames else frames[0])
            img, msk = to_display8(plate(f), linear), rmask(f)
        except Exception:  # noqa: BLE001 – no mask / frame: the assistant works from the text alone
            img = msk = None

        def cancel_check() -> None:
            if JOBS.running():
                JOBS.check()
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, lambda: PA.assist(req.text, mode, bool(req.transition), img, msk, cancel_check=cancel_check,
                                        names=_removed_names(s)))
        except PA.PromptAssistError as e:
            raise HTTPException(status_code=409, detail=str(e))

    # ------------------------------------------------------------------ MiniMax-H3 anchors (engine/h3_anchors.py)
    def _anchor_ctx(work: Path, manifest: Dict[str, Any], s: Dict[str, Any]):
        """(frames, plate, rmask, linear) of the work range – what the anchors read."""
        w, h = plate_size()
        beauty = (manifest.get("plates") or {}).get("beauty") or ""
        linear = beauty.lower().endswith(".exr")
        rm, keep = _alpha_loaders(s, w, h)
        frames = list(work_frames())
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")

        src = cu.input_frame_fn(work, s, lambda f: frame_file(beauty, f))   # „Wejście” (2026-11-15)

        def plate(f):
            img = _read_plate(src(f))
            if img is None:
                raise RuntimeError(f"Brak klatki wejścia F{f}")
            return img

        rmask = cu.gen_mask_fn(s, rm, keep)   # the MODEL's mask („Zapas generowania”, 2026-11-13) – what the anchors fill
        return frames, plate, rmask, linear

    def _anchors_root(work: Path) -> Path:
        return cu.cleanup_dir(work) / "anchors"

    def _ensure_anchors(work: Path, s: Dict[str, Any], frames, rmask) -> List[Dict[str, Any]]:
        """The anchor list; when the anchors are on and the list is empty: the main one (chosen / suggested frame) +
        the linked ones near the range's ends (h3_fill.derived_frames). Saved."""
        from engine import h3_anchors as HA
        from engine import h3_fill
        if s.get("anchor") != "frame":
            return cu.active_anchors(s)          # anchors off = none, whatever the saved list holds
        if not s.get("anchors"):
            mf = h3_fill.anchor_frame(frames, "frame", s.get("anchor_frame"), rmask)
            s["anchors"] = HA.defaults(mf, [] if s.get("anchor_ref_only") else h3_fill.derived_frames(frames, mf))
            s.update(cu.save_settings(work, s))
        return cu.active_anchors(s)               # „tylko referencja”: the main anchor alone

    def _gen_image(work: Path, a: Dict[str, Any]) -> Optional[np.ndarray]:
        from engine import h3_anchors as HA
        if not a.get("gen"):
            return None
        f = HA.gen_file(_anchors_root(work), a["id"], a["gen"])
        return cv2.imread(str(f), cv2.IMREAD_COLOR) if f.exists() else None

    def _warp_linked(work: Path, s: Dict[str, Any], m: Dict[str, Any], targets, frames, plate, rmask, linear,
                     cancel_check=None) -> Dict[int, np.ndarray]:
        """The main anchor's chosen generation moved onto every frame of `targets` (engine/anchor_warp: CoTracker3 –
        LK when missing – over a grid of the main patch, ONE tracker call for all of them). frame -> uint8 BGR."""
        from engine import anchor_warp as AW
        from engine.cleanup3d.void_engine import to_display8
        from engine.tracking import make_point_tracker
        main_img = _gen_image(work, m)
        if main_img is None:
            raise RuntimeError("Brak generacji kotwicy głównej")
        pts = AW.query_points(rmask(m["frame"]))
        cache: Dict[int, np.ndarray] = {}

        def img(f):
            if f not in cache:
                cache[f] = to_display8(plate(f), linear)
            return cache[f]
        tr = AW.track(make_point_tracker(), frames, img, m["frame"], targets, pts, cancel_check)
        mm = rmask(m["frame"])
        return {f: AW.warp_anchor(main_img, img(f), rmask(f), pts, p, v, main_mask=mm, holes=True) for f, (p, v) in tr.items()}

    def _generate(work: Path, s: Dict[str, Any], a: Dict[str, Any], anchors, plate, rmask, linear,
                  prompt: Optional[str] = None, seed: Optional[int] = None, frames=None,
                  warped: Optional[Dict[int, np.ndarray]] = None) -> str:
        """A new Qwen-Image-2.1 generation of anchor `a` (chosen at once). main: the shared description; linked: the
        main anchor's chosen generation as Picture 2 (made first when missing); own: its own description."""
        from engine import h3_anchors as HA
        from engine import h3_fill
        root = _anchors_root(work)
        mode = s.get("prompt_mode") or "remove"
        ref_img, ref, init, how, hole = None, None, None, None, None
        if a["kind"] == "linked":
            m = HA.main_of(anchors)
            if m is None:
                raise RuntimeError("Brak kotwicy głównej")
            if not (m.get("gen") and HA.gen_file(root, m["id"], m["gen"]).exists()):
                _generate(work, s, m, anchors, plate, rmask, linear)
            ref_img, ref = _gen_image(work, m), f"{m['id']}/{m['gen']}"
            text = s.get("prompt") or ""
            from engine import anchor_warp as AW
            how = AW.method(s)
            if how != "qwen":   # the main anchor's pattern moved onto this frame (user 2026-11-11)
                if warped is None or a["frame"] not in warped:
                    warped = _warp_linked(work, s, m, [a["frame"]], frames or [a["frame"], m["frame"]], plate, rmask, linear)
                init, hole = warped[a["frame"]]
        elif a["kind"] == "own":
            text = (a.get("prompt") or "") if prompt is None else prompt
            a["prompt"] = text
        else:
            text = (s.get("prompt") or "") if prompt is None else prompt
            s["prompt"] = text
        sd = int(s.get("seed") or 42) if seed is None else int(seed)
        from engine import anchor_warp as AW
        raws: list = []   # Qwen-Image-2.1's pictures before the paste: saved as g_###_raw.png (+ _hole_raw for the repaint)
        if how == "warp":
            img = init   # the warp alone: no Qwen-Image-2.1 pass (the holes hold the surrounding pattern)
        else:
            if init is not None and hole is not None and hole.mean() > 0.0005:
                # the warp's holes (not under the main patch / torn) repainted in full first (user 2026-11-11: the
                # warp stretched and deformed the temples and cheeks), then the light touch-up of the whole patch
                init = h3_fill.make_anchor(comfy_client(), work / "comfy_io", plate(a["frame"]), rmask(a["frame"]), text,
                                           mode, linear, seed=sd, ref=ref_img, init=init, denoise=1.0, region=hole,
                                           text=AW.hole_prompt(text), raw=raws)
            own_dn = float(s.get("anchor_denoise", 1.0)) if a["kind"] in ("main", "own") else None   # „Siła kotwicy”
            img = h3_fill.make_anchor(comfy_client(), work / "comfy_io", plate(a["frame"]), rmask(a["frame"]), text, mode,
                                      linear, seed=sd, ref=ref_img, init=init, denoise=AW.denoise(s), raw=raws,
                                      plate_denoise=own_dn, what=_removed_noun(s) if init is None else None)
        ok, buf = cv2.imencode(".png", img)
        meta = {"prompt": text, "mode": mode, "seed": sd, "ref": ref, "frame": a["frame"], "kind": a["kind"]}
        if a["kind"] in ("main", "own") and how != "warp" and float(s.get("anchor_denoise", 1.0)) < 1.0:
            meta["denoise"] = float(s["anchor_denoise"])     # made from the plate at this strength
        if how:
            meta["link"] = how
            if hole is not None:
                meta["hole"] = round(float(hole.mean()), 5)
            if how == "warp_qwen":
                meta["denoise"] = AW.denoise(s)
        name = HA.add_gen(root, a["id"], buf.tobytes(), meta)
        gdir = HA.gen_file(root, a["id"], name).parent
        for i, r in enumerate(raws):   # the last one = the picture this generation was pasted from
            tag = "raw" if i == len(raws) - 1 else "hole_raw"
            cv2.imwrite(str(gdir / f"{name}_{tag}.png"), r)
        if init is not None:
            cv2.imwrite(str(gdir / f"{name}_warp.png"), init)   # what Qwen-Image-2.1 started from
        a["gen"] = name
        return name

    def _anchor_image(work: Path, s: Dict[str, Any], a: Dict[str, Any], anchors, plate, rmask, linear,
                      frames=None) -> np.ndarray:
        """The image the run pins: the original frame, the chosen generation when still valid, else a new one."""
        from engine import h3_anchors as HA
        from engine.cleanup3d.void_engine import to_display8
        if a["kind"] == "original":
            return to_display8(plate(a["frame"]), linear)
        if a["kind"] == "projection":   # the copied C5 render of its frame (none yet = copy it now)
            if HA.valid_gen(_anchors_root(work), a, None) is None:
                HA.snapshot_projection(work, _anchors_root(work), a, linear)
            return _gen_image(work, a)
        if HA.valid_gen(_anchors_root(work), a, HA.main_of(anchors)) is None:
            _generate(work, s, a, anchors, plate, rmask, linear, frames=frames)
        if s.get("anchor_raw"):   # Qwen-Image-2.1's whole picture (user test 2026-11-11); none saved = the pasted one
            raw = _raw_image(work, a)
            if raw is not None:
                return raw
        return _gen_image(work, a)

    def _raw_image(work: Path, a: Dict[str, Any]) -> Optional[np.ndarray]:
        from engine import h3_anchors as HA
        if not a.get("gen"):
            return None
        f = HA.gen_file(_anchors_root(work), a["id"], a["gen"])
        r = f.with_name(f"{a['gen']}_raw.png")
        return cv2.imread(str(r), cv2.IMREAD_COLOR) if r.exists() else None

    def _anchors_view(work: Path, s: Dict[str, Any]) -> Dict[str, Any]:
        from engine import h3_anchors as HA
        from engine import h3_fill
        root = _anchors_root(work)
        anchors = s.get("anchors") or []
        main = HA.main_of(anchors)
        raw_on = bool(s.get("anchor_raw"))
        out = []
        for a in anchors:
            gl = HA.gens(root, a["id"])
            vg = HA.valid_gen(root, a, main)
            out.append({**a, "valid": a["kind"] == "original" or vg is not None,
                        "gens": [_gen_view(root, a["id"], g, raw_on) for g in reversed(gl)],
                        "derived_prompt": h3_fill.derived_prompt(s.get("prompt") or "", s.get("prompt_mode") or "remove", _removed_noun(s))
                        if a["kind"] == "linked" else ""})
        try:   # the C5 render the projection anchors copy (the button needs one)
            from engine.cleanup3d import store as c3d_store
            rs = c3d_store.render_state(work)
        except Exception:   # noqa: BLE001 - the anchors work without Cleanup 3D
            rs = None
        proj = [a for a in out if a["kind"] == "projection"]

        def copied_from(a):
            return next((g.get("projection") for g in a["gens"] if g["name"] == a.get("gen")), None)
        clashes = []
        if s.get("model") == "h3":   # two anchors in one MiniMax-H3 token: only one holds (HA.token_clashes)
            try:
                clashes = [{**c, "note": HA.clash_note(c)} for c in HA.token_clashes(anchors, list(work_frames()))]
            except Exception:  # noqa: BLE001 – no work range: no warning
                clashes = []
        return {"anchors": out, "main": main["id"] if main else None, "transition": HA.is_transition(anchors),
                "token_clash": clashes,
                "h3_prompt": s.get("h3_prompt") or "", "on": s.get("anchor") == "frame",
                "projection": {"render": rs["version"] if rs else None, "stale": bool(rs and rs["stale"]),
                               # copied from another render than the last one = press the button again
                               "outdated": [a["id"] for a in proj if rs and copied_from(a) != rs["version"]]}}

    def _gen_view(root: Path, aid: str, g: Dict[str, Any], raw_on: bool) -> Dict[str, Any]:
        """A generation for the UI: with anchor_raw the preview / export shows what the video model gets – Qwen-Image-2.1's
        whole picture g_###_raw.png (user 2026-11-12: the pasted one was shown while the raw one was pinned)."""
        from engine import h3_anchors as HA
        url = f"/api/cleanup/anchors/{aid}/gens/{g['name']}"
        has_raw = HA.gen_file(root, aid, g["name"]).with_name(f"{g['name']}_raw.png").exists()
        shown = "raw" if raw_on and has_raw else "pasted"
        return {**g, "url": url + ("?raw=1" if shown == "raw" else ""), "pasted_url": url, "has_raw": has_raw,
                "shown": shown}

    def _import_from_versions(work: Path, s: Dict[str, Any]) -> bool:
        """Anchors without any generation take the image the NEWEST version pinned at their frame (v###/anchors/
        anchor_####.png – runs before the generation history, 2026-11-11), so a good result is not lost. A linked one
        imported from the same version as the main one counts as made from it."""
        from engine import h3_anchors as HA
        root = _anchors_root(work)
        vers = sorted((d for d in cu.cleanup_dir(work).glob("v[0-9][0-9][0-9]") if (d / "anchors").is_dir()), reverse=True)
        changed = False
        anchors = s.get("anchors") or []
        main = HA.main_of(anchors)
        order = sorted(anchors, key=lambda a: a["kind"] != "main")   # the main first: the linked ones reference it
        src_of = {}
        for a in order:
            if a["kind"] in ("original", "projection") or HA.gens(root, a["id"]):
                continue
            v = next((d for d in vers if (d / "anchors" / f"anchor_{a['frame']:04d}.png").exists()), None)
            if v is None:
                continue
            ref = None
            if a["kind"] == "linked" and main and main.get("gen") and src_of.get(main["id"]) == v.name:
                ref = f"{main['id']}/{main['gen']}"
            a["gen"] = HA.add_gen(root, a["id"], (v / "anchors" / f"anchor_{a['frame']:04d}.png").read_bytes(),
                                  {"prompt": s.get("prompt") or "", "mode": s.get("prompt_mode") or "remove",
                                   "seed": s.get("seed"), "ref": ref, "frame": a["frame"], "kind": a["kind"],
                                   "imported": v.name})
            src_of[a["id"]] = v.name
            changed = True
        return changed

    @router.get("/api/cleanup/anchors")
    async def cleanup_anchors():
        work, manifest = _need()
        s = cu.load_settings(work)
        if s.get("anchor") == "frame" and not s.get("anchors"):
            frames, _p, rmask, _l = _anchor_ctx(work, manifest, s)
            await asyncio.get_running_loop().run_in_executor(None, lambda: _ensure_anchors(work, s, frames, rmask))
        if _import_from_versions(work, s):
            s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    @router.post("/api/cleanup/anchors")
    def cleanup_anchor_add(req: AnchorAdd):
        """A new anchor at a frame: linked (follows the main one), original (the plate frame itself) or own."""
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        if req.kind not in ("linked", "original", "own", "projection"):
            raise HTTPException(status_code=400, detail="Rodzaj kotwicy: linked | original | own | projection")
        if any(a["frame"] == int(req.frame) for a in s["anchors"]):
            raise HTTPException(status_code=409, detail=f"W F{req.frame} już jest kotwica")
        new = {"id": HA.new_id(), "frame": int(req.frame), "kind": req.kind,   # no main yet: normalize makes the first linked one main
               "prompt": s.get("prompt") or ""}
        if req.kind == "projection":
            _snapshot(work, new)
        s["anchor"] = "frame"
        s["anchors"] = HA.normalize(s["anchors"] + [new])
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    def _plate_linear() -> bool:
        _w, manifest = _need()
        return ((manifest.get("plates") or {}).get("beauty") or "").lower().endswith(".exr")

    def _snapshot(work: Path, a: Dict[str, Any]) -> None:
        """Copy the last C5 render of the anchor's frame in as its new generation (409 without a render)."""
        from engine import h3_anchors as HA
        try:
            HA.snapshot_projection(work, _anchors_root(work), a, _plate_linear())
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @router.post("/api/cleanup/anchors/projection")
    def cleanup_anchor_projection(req: AnchorProjection):
        """Anchors from the card keys (user request 2026-11-12): projection anchors at every frame where a card key
        brings a new image (+ every `step`-th frame), each = the last Cleanup 3D render (C5) of that frame. An anchor
        already there: a projection one gets the current render as a new generation, another kind is turned into a
        projection one (its generations stay in the history). Not a transition: the shared prompt and the whole-clip
        reference stay."""
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        try:
            HA.projection_version(work)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        frames = HA.projection_frames(work, list(work_frames()), max(0, int(req.step or 0)))
        if not frames:
            raise HTTPException(status_code=409, detail="W obszarze roboczym nie ma kluczy kart (Czyszczenie 3D, C4)")
        root = _anchors_root(work)
        # the automatic main / linked anchors that never got an image (turning the anchors on makes them) would make
        # Qwen-Image-2.1 generate at the run – dropped; anchors with any generation stay
        s["anchors"] = [a for a in s["anchors"] if a["kind"] not in ("main", "linked") or HA.gens(root, a["id"])
                        or a["frame"] in frames]
        by_frame = {a["frame"]: a for a in s["anchors"]}
        for f in frames:
            a = by_frame.get(f)
            if a is None:
                a = {"id": HA.new_id(), "frame": f, "kind": "projection", "prompt": ""}
                s["anchors"].append(a)
            a["kind"] = "projection"
            _snapshot(work, a)
        s["anchor"] = "frame"
        s["anchors"] = HA.normalize(s["anchors"])
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    def _find(s: Dict[str, Any], aid: str) -> Dict[str, Any]:
        a = next((x for x in s["anchors"] if x["id"] == aid), None)
        if a is None:
            raise HTTPException(status_code=404, detail="Brak kotwicy")
        return a

    @router.patch("/api/cleanup/anchors/{aid}")
    def cleanup_anchor_patch(aid: str, req: AnchorPatch):
        """Chosen generation, own description, or kind: main (the old main becomes linked), linked (attach again),
        own (DETACH: keeps the generation it showed), original. The frame is fixed (an anchor never moves)."""
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        a = _find(s, aid)
        root = _anchors_root(work)
        if req.frame is not None and int(req.frame) != a["frame"]:
            # anchors are locked to their frame (user 2026-11-11): a generation fits only the frame it was made at –
            # another frame = a new anchor there (delete this one with its generations)
            raise HTTPException(status_code=409, detail=f"Kotwica jest przypięta do F{a['frame']} – dodaj nową kotwicę w innej klatce")
        if req.gen is not None:
            try:
                ok = HA.gen_file(root, aid, req.gen).exists()
            except ValueError:
                ok = False
            if not ok:
                raise HTTPException(status_code=404, detail="Brak generacji")
            a["gen"] = req.gen
        if req.prompt is not None:
            a["prompt"] = req.prompt[:1000]
        if req.kind is not None and req.kind != a["kind"]:
            if req.kind not in HA.KINDS:
                raise HTTPException(status_code=400, detail="Zły rodzaj kotwicy")
            if req.kind == "main":
                for x in s["anchors"]:
                    if x["kind"] == "main":
                        x["kind"] = "linked"
            if req.kind == "projection":   # the C5 render of its frame becomes the chosen generation
                _snapshot(work, a)
            if req.kind == "own" and a["kind"] == "linked":   # detach: the same image, now its own (a morph step)
                a["prompt"] = a.get("prompt") or s.get("prompt") or ""
                if a.get("gen") and HA.gen_file(root, aid, a["gen"]).exists():
                    src = HA.gen_file(root, aid, a["gen"])
                    meta = next((g for g in HA.gens(root, aid) if g["name"] == a["gen"]), {})
                    keep = {k: v for k, v in meta.items() if k not in ("name", "time")}
                    a["gen"] = HA.add_gen(root, aid, src.read_bytes(),
                                          {**keep, "kind": "own", "ref": None, "detached_from": meta.get("ref")})
            a["kind"] = req.kind
        s["anchors"] = HA.normalize(s["anchors"])
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    @router.delete("/api/cleanup/anchors/{aid}")
    def cleanup_anchor_delete(aid: str):
        """The anchor and its generations are deleted from disk for good (user decision 2026-11-13). The main one takes
        its LINKED anchors with it (they only copy its look – user 2026-11-11); own / original anchors stay and the first
        of them becomes main. The view carries `freed_bytes`."""
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        a = _find(s, aid)
        gone = {aid} | ({x["id"] for x in s["anchors"] if x["kind"] == "linked"} if a["kind"] == "main" else set())
        freed = sum(HA.delete(HA.anchor_dir(_anchors_root(work), x)) for x in gone)
        s["anchors"] = HA.normalize([x for x in s["anchors"] if x["id"] not in gone])
        if not s["anchors"]:
            s["anchor"] = "none"
        s = cu.save_settings(work, s)
        return {**_anchors_view(work, s), "freed_bytes": freed}

    @router.post("/api/cleanup/anchors/linked/generate")
    @long_job("Kotwice podłączone (Qwen-Image-2.1)")
    async def cleanup_anchor_generate_linked(all: bool = False):
        """Every linked anchor not made from the main one's chosen image yet (all=1: every linked one again, e.g.
        after another link method): the main pattern moved onto it (engine/anchor_warp) + Qwen-Image-2.1 touch-up, or
        Qwen-Image-2.1 with that image as Picture 2 – the main one itself is not generated again (user 2026-11-11)."""
        from engine import h3_anchors as HA
        from engine.comfy_io import job_progress
        work, manifest = _need()
        s = cu.load_settings(work)
        root = _anchors_root(work)
        m = HA.main_of(s["anchors"])
        if m is None or not (m.get("gen") and HA.gen_file(root, m["id"], m["gen"]).exists()):
            raise HTTPException(status_code=409, detail="Najpierw wygeneruj (albo wybierz) generację kotwicy głównej")
        todo = [a for a in s["anchors"] if a["kind"] == "linked" and (all or HA.valid_gen(root, a, m) is None)]
        frames, plate, rmask, linear = _anchor_ctx(work, manifest, s)

        def go():
            from engine import anchor_warp as AW
            warped = None
            if todo and AW.method(s) != "qwen":   # one tracker call for all of them
                job_progress(f"CoTracker3: wzór F{m['frame']} → {len(todo)} kotwic", 0, len(todo) + 1)
                warped = _warp_linked(work, s, m, [a["frame"] for a in todo], frames, plate, rmask, linear)
            for i, a in enumerate(todo):
                job_progress(f"Kotwica podłączona F{a['frame']} (wzór F{m['frame']})", i + 1, len(todo) + 1)
                _generate(work, s, a, s["anchors"], plate, rmask, linear, frames=frames, warped=warped)
        try:
            await asyncio.get_running_loop().run_in_executor(None, go)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    @router.post("/api/cleanup/anchors/{aid}/generate")
    @long_job("Kotwica Qwen-Image-2.1")
    async def cleanup_anchor_generate(aid: str, req: AnchorGen):
        """A new generation of the anchor (a new random seed unless given; the description shown in the panel)."""
        import random
        work, manifest = _need()
        s = cu.load_settings(work)
        a = _find(s, aid)
        if a["kind"] == "original":
            raise HTTPException(status_code=409, detail="Kotwica-oryginał to klatka plate'u – nie generuje się")
        frames, plate, rmask, linear = _anchor_ctx(work, manifest, s)
        seed = req.seed if req.seed is not None else random.randrange(2 ** 31)
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _generate(work, s, a, s["anchors"], plate, rmask, linear, req.prompt, seed, frames=frames))
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    @router.post("/api/cleanup/anchors/{aid}/import")
    def cleanup_anchor_import(aid: str, req: AnchorImport):
        """The user's own image as a new generation of the anchor (e.g. an exported anchor touched up in Photoshop, or
        a reference): scaled to the plate frame and chosen. A linked anchor becomes own – the image is not made
        from the main one; on the main one the linked anchors follow it."""
        import base64
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        a = _find(s, aid)
        try:
            raw = base64.b64decode(str(req.data).split(",", 1)[-1], validate=False)
        except ValueError:
            raw = b""
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if img is None:
            raise HTTPException(status_code=400, detail="Nie da się odczytać obrazu (PNG / JPEG / WebP)")
        w, h = plate_size()
        if (img.shape[1], img.shape[0]) != (int(w), int(h)):
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_CUBIC)
        if a["kind"] in ("linked", "original"):
            a["kind"] = "own"
        ok, buf = cv2.imencode(".png", img)
        a["gen"] = HA.add_gen(_anchors_root(work), aid, buf.tobytes(),
                              {"prompt": a.get("prompt") or s.get("prompt") or "", "seed": None, "ref": None,
                               "frame": a["frame"], "kind": a["kind"], "imported": (req.name or "plik")[:120]})
        s["anchors"] = HA.normalize(s["anchors"])
        s = cu.save_settings(work, s)
        return _anchors_view(work, s)

    @router.get("/api/cleanup/anchors/{aid}/gens/{name}")
    def cleanup_anchor_gen_image(aid: str, name: str, raw: str = ""):   # str: an old cached UI glues "?raw=1?t=…"
        """raw=1: Qwen-Image-2.1's whole picture saved next to the generation (the pasted one when there is none)."""
        from engine import h3_anchors as HA
        work, _ = _need()
        try:
            f = HA.gen_file(_anchors_root(work), aid, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if str(raw).startswith("1") and f.with_name(f"{name}_raw.png").exists():
            f = f.with_name(f"{name}_raw.png")
        if not f.exists():
            raise HTTPException(status_code=404, detail="Brak generacji")
        return Response(content=f.read_bytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.delete("/api/cleanup/anchors/{aid}/gens/{name}")
    def cleanup_anchor_gen_delete(aid: str, name: str):
        """A generation deleted from disk for good (user decision 2026-11-13); the chosen one deleted = the newest
        remaining is chosen. The view carries `freed_bytes`."""
        from engine import h3_anchors as HA
        work, _ = _need()
        s = cu.load_settings(work)
        a = _find(s, aid)
        root = _anchors_root(work)
        try:
            f = HA.gen_file(root, aid, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        freed = HA.delete(f)
        for side in sorted(f.parent.glob(f"{name}_*.png")):   # its _warp / _raw / _hole_raw pictures go with it
            freed += HA.delete(side)
        if a.get("gen") == name:
            rest = HA.gens(root, aid)
            a["gen"] = rest[-1]["name"] if rest else None
        s = cu.save_settings(work, s)
        return {**_anchors_view(work, s), "freed_bytes": freed}

    def _version_dir(work: Path, version: str) -> Path:
        import re
        if not re.fullmatch(r"v\d{3}", version or ""):
            raise HTTPException(status_code=400, detail="Wersja: v###")
        d = cu.cleanup_dir(work) / version
        if not (d / "run.json").exists():
            raise HTTPException(status_code=404, detail=f"Brak wersji czyszczenia {version}")
        return d

    @router.post("/api/cleanup/versions/{version}/show")
    def cleanup_show_version(version: str):
        """Plate „cleaned” = the comp of this version (nothing recomputed) AND its settings come back for its object:
        model, description, seed and its anchors (user 2026-11-11) – from run.json, or rebuilt from the images the
        version pinned (v###/anchors) for runs made before the anchor list."""
        work, _ = _need()
        d = _version_dir(work, version)
        v = next((x for x in cu.versions(work) if x["version"] == version), {})
        plates = _set_cleaned(d, v.get("comp_ext") or cu.file_ext(v.get("ext") or "png"))
        s = _restore_version(work, d)
        return {"version": version, "plate": CLEANED_PLATE, "plates": plates, "settings": s}

    def _version_anchors(work: Path, vdir: Path, vs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The anchor list of a version: its saved list when every chosen generation is still there; else one anchor
        per image it pinned (v###/anchors/anchor_####.png, imported as a generation – the same ids every time, so a
        second click reuses them); the main one = the version's main / anchor_frame, else the middle one."""
        import hashlib
        from engine import h3_anchors as HA
        root = _anchors_root(work)
        saved = HA.normalize(vs.get("anchors") or [])
        if saved and all(a["kind"] == "original" or (a.get("gen") and HA.gen_file(root, a["id"], a["gen"]).exists())
                         for a in saved):
            return saved
        files = sorted((vdir / "anchors").glob("anchor_[0-9][0-9][0-9][0-9].png")) if (vdir / "anchors").is_dir() else []
        if not files:
            return saved
        frames = [int(f.stem.split("_")[1]) for f in files]
        old_main = next((a["frame"] for a in saved if a["kind"] == "main"), None)
        mf = old_main if old_main in frames else (vs.get("anchor_frame") if vs.get("anchor_frame") in frames else frames[len(frames) // 2])
        out, main = [], None
        for f, png in sorted(zip(frames, files), key=lambda x: x[0] != mf):   # the main first: the linked reference it
            aid = "a" + hashlib.sha1(f"{vdir.name}|{f}".encode()).hexdigest()[:6]
            kind = "main" if f == mf else "linked"
            g = next((x for x in HA.gens(root, aid) if x.get("imported") == vdir.name), None)
            if g:
                name = g["name"]
            else:
                ref = f"{main['id']}/{main['gen']}" if kind == "linked" and main else None
                name = HA.add_gen(root, aid, png.read_bytes(), {"prompt": vs.get("prompt") or "", "mode": vs.get("prompt_mode"),
                                                               "seed": vs.get("seed"), "ref": ref, "frame": f, "kind": kind,
                                                               "imported": vdir.name})
            a = {"id": aid, "frame": f, "kind": kind, "prompt": "", "gen": name}
            if kind == "main":
                main = a
            out.append(a)
        return HA.normalize(out)

    def _restore_version(work: Path, vdir: Path) -> Dict[str, Any]:
        import json as _json
        try:
            vs = (_json.loads((vdir / "run.json").read_text(encoding="utf-8")).get("settings") or {})
        except (OSError, ValueError):
            return cu.load_settings(work)
        s = cu.load_settings(work)
        # the composite (band sliders / keys / points, unprotected objects) is NOT the version's: it stays as the user set
        # it (user report 2026-11-15: keys deleted in v031 came back after v030 -> v031); a version brings back what
        # made its model picture – model, description, seed, anchors, generation margin
        from engine import inpaint_recompose as IR
        fields = {k: vs[k] for k in cu.PER_TARGET if k in vs and k not in IR.BAND_FIELDS + IR.TARGET_FIELDS}
        fields["anchors"] = _version_anchors(work, vdir, vs)
        fields["anchor"] = "frame" if fields["anchors"] else (vs.get("anchor") if vs.get("anchor") in ("none", "frame") else "none")
        who = (vs.get("remove") or [None])[0]
        cur = (s.get("remove") or [None])[0]
        if who is None or who == cur:
            s.update(fields)
        else:   # a version of another object: its settings wait for it (the panel lists only the current one's)
            s.setdefault("per_target", {})[who] = {**(s.get("per_target", {}).get(who) or {}), **fields}
        return cu.save_settings(work, s)

    @router.delete("/api/cleanup/versions/{version}")
    def cleanup_delete_version(version: str):
        """A version deleted from disk for good (user decision 2026-11-13: no project trash any more).
        When the plate „cleaned” showed it, it moves to the newest remaining version (none left = plate removed)."""
        from engine import deletion as DEL
        work, manifest = _need()
        d = _version_dir(work, version)
        shown = f"/{version}/comp/" in str((manifest.get("plates") or {}).get(CLEANED_PLATE) or "").replace("\\", "/")
        cu.remember_version(work)                     # its number stays taken (cu.next_version)
        freed = DEL.remove(d, inside=cu.cleanup_dir(work))
        plates = None
        if shown:
            rest = cu.versions(work)
            if rest:
                plates = _set_cleaned(cu.cleanup_dir(work) / rest[-1]["version"], rest[-1].get("comp_ext") or cu.file_ext(rest[-1].get("ext") or "png"))
            elif STATE.get("shot_project") is not None:
                STATE["shot_project"].remove_plate(CLEANED_PLATE)
                plates = (refresh_manifest() or {}).get("plates")
        return {"version": version, "freed_bytes": freed, "plates": plates,
                "versions": cu.versions(work)}

    @router.post("/api/cleanup/versions/{version}/upscale")
    @long_job("Inpaint: SeedVR2 wersji {version}")
    async def cleanup_upscale_version(version: str):
        """„⤴ Upscale (SeedVR2)” (user 2026-11-11): the version's fill restored at the plate's scale by SeedVR2 as one
        clip (engine/inpaint_upscale.py) and laid back through the Cleaner with the version's own settings – only the
        fill inside the mask changes; a NEW version (`upscaled_from`), the plate „cleaned” shows it."""
        import json as _json
        from engine import inpaint_upscale as IU
        from engine.cleanup3d import void_engine as VE
        from engine.comfy_io import job_progress
        from engine.config import load_config
        from engine.jobs import JOBS
        work, manifest = _need()
        vdir = _version_dir(work, version)
        rj = _json.loads((vdir / "run.json").read_text(encoding="utf-8"))
        vs = cu.settings(rj.get("settings") or {})
        frames = IU.version_frames(vdir)
        if not frames:
            raise HTTPException(status_code=409, detail=f"Wersja {version} nie ma klatek")
        cfg = load_config()
        try:
            runner = VE.seedvr2_runner(cfg, log=VE.void_root(work) / "seedvr2_inpaint.log")
        except VE.ToolError as e:
            raise HTTPException(status_code=409, detail=f"SeedVR2 niedostępny: {e} – Doktor → Zainstaluj SeedVR2")
        w, h = plate_size()
        beauty = (manifest.get("plates") or {}).get("beauty") or ""
        linear = beauty.lower().endswith(".exr")
        rm, keep = _alpha_loaders(vs, w, h)
        src = cu.input_frame_fn(work, vs, lambda f: frame_file(beauty, f))   # the version's own input (2026-11-15)

        def plate(f):
            img = _read_plate(src(f))
            if img is None:
                raise RuntimeError(f"Brak klatki wejścia F{f}")
            return img

        def cancel_check() -> None:
            if JOBS.running():
                JOBS.check()
        out = cu.next_version(work)

        def run():
            fill = IU.upscale(vdir, frames, vs.get("model") or "h3", vs, linear, runner, VE._ffmpeg(cfg),
                              VE.void_root(work) / "seedvr2_inpaint", progress=lambda t, d, n: job_progress(t, d, n),
                              cancel=cancel_check, client=comfy_client())
            cl = cu.Cleaner(vs, w, h, linear, plate=plate, remove_alphas=lambda f: [fn(f) for fn in rm],
                            keep_alphas=lambda f: [fn(f) for fn in keep], clean=None, auto_clean=None,
                            inpaint=_inpainter(work, vs), progress=lambda t, d, n: job_progress(t, d, n),
                            keep_names=cu.names_of(keep), key_fill=fill,
                            plate_file=src)
            cl.report["upscaled_from"] = version
            _held(rm, frames, cl)
            if rj.get("anchors"):
                cl.report["anchors"] = rj["anchors"]
                if (vdir / "anchors").is_dir():
                    import shutil
                    shutil.copytree(vdir / "anchors", out / "anchors", dirs_exist_ok=True)
            return cl.run(frames, out, rj.get("ext") or "png")
        try:
            res = await asyncio.get_running_loop().run_in_executor(None, run)
        except (RuntimeError, ValueError, VE.ToolError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        plates = _set_cleaned(out, res.get("comp_ext") or cu.file_ext(res.get("ext") or "png"))
        return {"version": out.name, "from": version, "plate": CLEANED_PLATE, "plates": plates,
                "n_frames": res.get("n_frames"), "seconds": res.get("seconds")}

    # ------------------------------------------------------------------ „⟲ Złóż ponownie” (user decision 2026-11-13)
    # per-frame inputs of a recomposite kept between requests (2026-11-14, speed: a live preview re-read the plate and
    # every object's alpha – EXR – for each slider move, ~0.7 s of a 2 s preview): the plate, the removed / kept
    # alphas and the shot's grain measurement of ONE version at a time, a few frames each; a change of what is
    # removed / kept (or of their masks' files) starts a new cache
    _PV: Dict[str, Any] = {"key": None, "frames": {}, "grain": None}
    import threading as _threading
    _PV_LOCK = _threading.Lock()                 # the Cleaner prepares frames in several threads
    PV_FRAMES = 6

    def _pv_cache(key: Any) -> Dict[str, Any]:
        if _PV["key"] != key:
            _PV.update(key=key, frames={}, grain=None)
        return _PV

    def _pv_wrap(cache: Dict[str, Any], tag: str, fn):
        """fn(f) memoised in the shared preview cache under (tag, f), the newest PV_FRAMES frames kept."""
        def at(f):
            k = (tag, int(f))
            fr = cache["frames"]
            with _PV_LOCK:
                if k in fr:
                    v = fr.pop(k)
                    fr[k] = v
                    return v
            v = fn(f)
            with _PV_LOCK:
                fr[k] = v
                while len([x for x in fr if x[0] == tag]) > PV_FRAMES:
                    fr.pop(next(x for x in fr if x[0] == tag))
            return v
        return at

    def _masks_stamp(s: Dict[str, Any]) -> tuple:
        """What the removed / kept alphas are made of, by their files' times: every object's object.json, core_fix,
        runs and final_touch folders (an added version, a changed active one) + the resolved mask's own files
        (engine/cleanup_hold.stamp: the run's run.json, core.json, core_fix.json, final_touch.json) – a Final Touch
        stroke or a new mask 1+2 made in cutting changes it, so the preview / overlay caches start again (user
        decision 2026-11-15, point 3)."""
        from engine import cleanup_hold as CH
        out: List[Any] = []
        for o in _objs():
            od = objects().dir(o["name"])
            for p in (od / "object.json", od / "core_fix", od / "runs", od / "final_touch"):
                try:
                    out.append(p.stat().st_mtime_ns)
                except OSError:
                    out.append(0)
            if o.get("kind") != "group":
                m = _mask_of(o, s)
                out.append(CH.stamp(objects(), o, m["run"]) if m["run"] else None)
        return tuple(out)

    def _recompose_parts(version: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """What a recomposite of `version` needs (engine/inpaint_recompose.py): its stored model frames over its own
        frames, the settings (the source's model / masks / generation margin + the current composite band, `overrides`
        = unsaved slider values) and a Cleaner factory – the run's Cleaner, with the stored frames as the key fill of
        the engine that made them and the source's model mask (what it drew)."""
        import json as _json
        from engine import inpaint_recompose as IR
        work, manifest = _need()
        vdir = _version_dir(work, version)
        rj = _json.loads((vdir / "run.json").read_text(encoding="utf-8"))
        if not IR.model_dir(vdir).is_dir():
            raise HTTPException(status_code=409, detail=f"Wersja {version} nie ma zapisanych klatek modelu (model/) – starsze "
                                                        "wersje nie dają się złożyć ponownie; policz ją od nowa")
        rng = rj.get("frames") or []
        frames = list(range(int(rng[0]), int(rng[1]) + 1)) if len(rng) == 2 else []
        miss = IR.missing_frames(vdir, frames) if frames else []
        if not frames or miss:
            raise HTTPException(status_code=409, detail=f"Wersja {version}: klatki modelu (model/) nie pokrywają jej zakresu"
                                                        + (f" – brak {cu.frame_ranges(miss)}" if miss else ""))
        s = IR.recompose_settings(cu.load_settings(work), rj.get("settings") or {}, overrides)
        if s["model"] not in cu.ANCHOR_MODELS:
            raise HTTPException(status_code=409, detail=f"Wersja {version}: silnik {s['model']} nie zapisuje klatek modelu")
        w, h = plate_size()
        beauty = (manifest.get("plates") or {}).get("beauty") or ""
        linear = beauty.lower().endswith(".exr")
        rm, keep = _alpha_loaders(s, w, h)
        fill = IR.stored_fill(vdir, s["model"], frames, w, h, linear)

        src = cu.input_frame_fn(work, s, lambda f: frame_file(beauty, f))   # the version's input (2026-11-15)

        def plate0(f):
            img = _read_plate(src(f))
            if img is None:
                raise RuntimeError(f"Brak klatki wejścia F{f}")
            return img
        # what the inputs are made of: the version, what is removed / kept and from which masks (their files' times)
        ck = _pv_cache((str(vdir), (vdir / "run.json").stat().st_mtime_ns, tuple(s.get("remove") or []),
                        _json.dumps(s.get("sources") or {}, sort_keys=True), bool(s.get("holdout")),
                        tuple(s.get("holdout_off") or []), _masks_stamp(s)))
        plate = _pv_wrap(ck, "plate", plate0)
        rm_hold = rm   # the cleanup_hold loaders themselves: `_held` scans them, the wrappers below use their plan
        rm = [cu.Named(_pv_wrap(ck, f"rm{i}", fn), getattr(fn, "name", "")) if hasattr(fn, "name") else _pv_wrap(ck, f"rm{i}", fn)
              for i, fn in enumerate(rm)]
        keep = [cu.Named(_pv_wrap(ck, f"keep{i}", fn), fn.name) if hasattr(fn, "name") else _pv_wrap(ck, f"keep{i}", fn)
                for i, fn in enumerate(keep)]

        def cleaner(progress=None):
            return cu.Cleaner(s, w, h, linear, plate=plate, remove_alphas=lambda f: [fn(f) for fn in rm],
                              keep_alphas=lambda f: [fn(f) for fn in keep], clean=None, auto_clean=None, inpaint=None,
                              progress=progress, keep_names=cu.names_of(keep), key_fill=fill,
                              plate_file=src, gen_mask=cu.gen_mask_fn(s, rm, keep))
        return {"work": work, "vdir": vdir, "rj": rj, "s": s, "frames": frames, "rm": rm_hold, "beauty": beauty,
                "linear": linear, "w": w, "h": h, "cleaner": cleaner, "cache": ck}

    @router.post("/api/cleanup/versions/{version}/recompose")
    @long_job("Inpaint: złóż ponownie {version}")
    async def cleanup_recompose_version(version: str):
        """„⟲ Złóż ponownie”: a NEW version from the stored model frames of `version` with the CURRENT composite band
        (grow_px, soft_px, band points / keys), holdout, levelling and grain – no model run. model/ and anchors/ are
        copied in (it can be recomposed again); run.json `recomposed_from`, `source_model`; the plate „cleaned” shows it."""
        import shutil
        from engine.comfy_io import job_progress
        P = _recompose_parts(version)
        rj, vdir, s, frames = P["rj"], P["vdir"], P["s"], P["frames"]
        out = cu.next_version(P["work"])

        def run():
            shutil.copytree(vdir / "model", out / "model", dirs_exist_ok=True)
            if (vdir / "anchors").is_dir():
                shutil.copytree(vdir / "anchors", out / "anchors", dirs_exist_ok=True)
            cl = P["cleaner"](lambda t, d, n: job_progress(t, d, n))
            # the same model frames over the same model area as the source: its levelling gain and its kept-object
            # check stand (2026-11-14, speed – each was one more read of every frame); the grain is measured again
            # (the setting may have changed)
            if s["tone"] and isinstance(rj.get("tone"), dict) and rj["tone"].get("bgr"):
                cl._gain = dict(rj["tone"])
            if not s.get("holdout_skip_lost") and isinstance(rj.get("kept_check"), dict):
                cl.kept = {"lost": {}, "reasons": {}, "skipped": False, "warnings": [], **rj["kept_check"]}
            vs = rj.get("settings") or {}
            cl.report["recomposed_from"] = version
            cl.report["source_model"] = {"model": vs.get("model"), "label": cu.MODEL_LABEL.get(vs.get("model"), vs.get("model")),
                                         "backend": rj.get("backend"), "frames": rj.get("frames"),
                                         "gen_margin_px": s["gen_margin_px"], "date": rj.get("date")}
            if rj.get("anchors"):
                cl.report["anchors"] = rj["anchors"]
            _held(P["rm"], frames, cl)
            ext = cu.output_ext(P["linear"], cv2.imread(frame_file(P["beauty"], frames[0]), cv2.IMREAD_UNCHANGED))
            return cl.run(frames, out, ext)
        try:
            res = await asyncio.get_running_loop().run_in_executor(None, run)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        plates = _set_cleaned(out, res.get("comp_ext") or cu.file_ext(res.get("ext") or "png"))
        return {"version": out.name, "from": version, "plate": CLEANED_PLATE, "plates": plates,
                "n_frames": res.get("n_frames"), "seconds": res.get("seconds"), "notes": res.get("notes"),
                "band_cut": res.get("band_cut")}

    @router.get("/api/cleanup/versions/{version}/model/{frame}")
    def cleanup_version_model_frame(version: str, frame: int):
        """The model's own picture of `frame` (model/m_####.png – what the video model drew, before the composite) at
        the viewer's preview size, stretched to the plate as the key fill does (h3_fill.H3Fill) – JPEG. The palette
        „Złożenie” shows it over the frame (user 2026-11-15: is the skin strip in the model or in the composite?)."""
        from engine import inpaint_recompose as IR
        work, _ = _need()
        d = _version_dir(work, version)
        f = IR.model_dir(d) / f"m_{int(frame):04d}.png"
        im = cv2.imread(str(f), cv2.IMREAD_COLOR) if f.exists() else None
        if im is None:
            raise HTTPException(status_code=404, detail=f"{version}: brak obrazu modelu F{frame}")
        w, h = plate_size()
        sc = proxy_scale()
        im = cv2.resize(im, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @router.get("/api/cleanup/versions/{version}/preview/{frame}")
    def cleanup_recompose_preview(version: str, frame: int, grow_px: Optional[float] = None,
                                  soft_px: Optional[float] = None):
        """Live preview of a recomposite: frame `frame` of `version` composited with the current band (+ the palette's
        unsaved grow_px / soft_px), the source version's levelling gain, grain – PNG at the viewer's preview size."""
        from engine.cleanup3d.void_engine import to_display8
        ov = {"grow_px": None if grow_px is None else max(cu.GROW_MIN, min(cu.GROW_MAX, float(grow_px))),
              "soft_px": None if soft_px is None else max(0.0, min(100.0, float(soft_px)))}
        P = _recompose_parts(version, ov)
        if int(frame) not in P["frames"]:
            raise HTTPException(status_code=409, detail=f"F{frame} poza zakresem wersji {version}")
        cl = P["cleaner"]()
        tone = P["rj"].get("tone")
        if P["s"]["tone"] and tone:      # the version's measured gain (re-measured by „Zastosuj”: the same region)
            cl._gain = dict(tone)
        try:
            if cl._grain_on():           # the grain measurement of the shown frame, once per version (cache)
                if P["cache"]["grain"] is None:
                    P["cache"]["grain"] = cl.measure_grain([int(frame)])
                cl._grain_shot = P["cache"]["grain"]
            comp, _m = cl.preview(int(frame))
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        sc = proxy_scale()
        w, h = P["w"], P["h"]
        img = cv2.resize(to_display8(comp, P["linear"]), (max(1, round(w * sc)), max(1, round(h * sc))),
                         interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])   # ~10x smaller than PNG, a preview
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ palette „Złożenie” → Ochrona (2026-11-15)
    # the overlay of one kept object: EXACTLY the alpha its holdout loader gives at the frame + where it cuts the patch.
    # The patch before the holdout (the composite band with its points, cut at what the model draws – cu.comp_mask_fn
    # with holdout off) is cached per frame / band / masks, so several overlays at once compute it once.
    _HO: Dict[str, Any] = {"key": None, "patch": None}
    HO_CUT_MIN = 0.02            # cut below this (0..1) is not counted

    @router.get("/api/cleanup/holdout/{name}/{frame}")
    def cleanup_holdout(name: str, frame: int, grow_px: Optional[float] = None, soft_px: Optional[float] = None):
        """PNG at the viewer's preview size: R = the kept object's holdout alpha (its keep loader – the mask
        cleanup_masks.resolve picks), G = where it cuts the current patch (alpha x the patch mask WITHOUT the holdout),
        B = 0. Headers: X-Holdout-On (1 = protected now: the cut is real; 0 = unticked / holdout off: the cut is what it
        WOULD cut), X-Holdout-Cut-Px (plate px cut > HO_CUT_MIN), X-Holdout-Mask (the run token)."""
        import json as _json
        from engine.export import run_alpha_loader
        work, _ = _need()
        s = cu.at_override(cu.load_settings(work), grow_px, soft_px, int(frame))   # unsaved values win in this frame
        objs = _objs()
        k = next((x for x in _kept(s, objs) if x["name"] == name), None)
        if k is None:
            raise HTTPException(status_code=404, detail=f"„{name}” nie jest obiektem chronionym tego czyszczenia")
        if not k["mask"]["run"]:
            raise HTTPException(status_code=409, detail=f"{name}: brak maski – {k['mask']['why']}")
        w, h = plate_size()
        a = np.clip(np.asarray(run_alpha_loader(objects().dir(name), k["mask"]["run"], w, h)(int(frame)), np.float32), 0, 1)
        so = {**s, "holdout": False}
        key = (int(frame), s["grow_px"], s["soft_px"], _json.dumps(s.get("band_points") or [], sort_keys=True),
               _json.dumps(s.get("band_keys") or {}, sort_keys=True),
               s.get("band_ref_out"), s["gen_margin_px"], s.get("model"), tuple(s.get("remove") or []),
               _json.dumps(s.get("sources") or {}, sort_keys=True), _masks_stamp(s))
        if _HO["key"] != key:
            rm, _keep = _alpha_loaders(so, w, h)
            _HO.update(key=key, patch=cu.comp_mask_fn(so, rm, [], cu.engine_reach_px(s.get("model"), s, w, h))(int(frame)))
        cut = a * np.asarray(_HO["patch"], np.float32)
        sc = proxy_scale()
        size = (max(1, round(w * sc)), max(1, round(h * sc)))
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[..., 2] = cv2.resize((a * 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)     # R (BGR order)
        img[..., 1] = cv2.resize((cut * 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)   # G
        ok, buf = cv2.imencode(".png", img)
        return Response(content=buf.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "no-store", "X-Holdout-On": "1" if k["on"] else "0",
                                 "X-Holdout-Cut-Px": str(int((cut > HO_CUT_MIN).sum())),
                                 "X-Holdout-Mask": str(k["mask"]["run"])})

    @router.get("/api/cleanup/mask/{frame}")
    def cleanup_mask(frame: int):
        work, _ = _need()
        s = cu.load_settings(work)
        w, h = plate_size()
        rm, keep = _alpha_loaders(s, w, h)
        # R = the run's composite mask, band points included, cut at what the model draws (cu.comp_mask_fn);
        # G = what the model draws: its mask (generation margin) grown by the engine (user 2026-11-15: „maska klatki
        # niech uwzględnia pas generowania”) – the same area the composite is cut at
        reach = cu.engine_reach_px(s.get("model"), s, w, h)
        m = cu.comp_mask_fn(s, rm, keep, reach)(frame)
        if s.get("model") in cu.C1_MODELS:   # the engine's own mask of this frame (MiniMax-H3: its latent token, ±2 fr.)
            gfn = cu.gen_mask_fn(s, rm, keep)
            g = cu.drawn_area(np.max(np.stack([np.asarray(gfn(k), np.float32)
                                               for k in cu.model_frames(s.get("model"), frame, work_frames())]), axis=0), reach)
        else:
            g = np.zeros_like(m, bool)
        sc = proxy_scale()
        size = (max(1, round(w * sc)), max(1, round(h * sc)))
        r8 = cv2.resize((m * 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
        g8 = cv2.resize(np.asarray(g, np.uint8) * 255, size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", cv2.merge([np.zeros_like(r8), g8, r8]))
        return Response(content=buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @router.post("/api/cleanup/run")
    @long_job("Czyszczenie plate'u")
    async def cleanup_run():
        work, manifest = _need()
        s = cu.load_settings(work)
        w, h = plate_size()
        beauty = (manifest.get("plates") or {}).get("beauty") or ""
        linear = beauty.lower().endswith(".exr")
        rm, keep = _alpha_loaders(s, w, h)
        clean, auto, info = _fill_sources(work, manifest, w, h, linear)
        frames = list(work_frames())
        if not frames:
            raise HTTPException(status_code=409, detail="Pusty obszar roboczy")
        from engine.comfy_io import job_progress
        if s["model"] not in cu.C1_MODELS:
            raise HTTPException(status_code=409, detail="Inpaint liczy MiniMax-H3 albo Wan 2.2 Fun VACE – usuwanie LaMą / "
                                                        "VOID jest w krokach projekcji (C2–C6)")
        infos = _models_info() if s["model"] != "lama" else []
        if s["model"] != "lama":   # a model that is not installed: the panel's install card says how (no silent LaMa)
            mi = next((m for m in infos if m["id"] == s["model"]), {})
            if not mi.get("available"):
                raise HTTPException(status_code=409, detail=f"{mi.get('label', s['model'])} niedostępny: {mi.get('reason') or 'brak modelu'}")
        if s["model"] in cu.ANCHOR_MODELS and s.get("anchor", "none") != "none":
            qi = next((m for m in infos if m["id"] == "qwen21"), {})
            if not qi.get("available"):
                raise HTTPException(status_code=409, detail=f"Kotwica Qwen-Image-2.1 niedostępna: {qi.get('reason') or 'brak modelu'} "
                                                            "– wyłącz kotwicę albo zainstaluj / uruchom ponownie ComfyUI")

        src = cu.input_frame_fn(work, s, lambda f: frame_file(beauty, f))   # „Wejście” (2026-11-15)
        in_miss = cu.input_missing(work, s, frames)

        def plate(f):
            img = _read_plate(src(f))
            if img is None:
                raise RuntimeError(f"Brak klatki wejścia F{f}")
            return img
        out = cu.next_version(work)
        ext = cu.output_ext(linear, cv2.imread(frame_file(beauty, frames[0]), cv2.IMREAD_UNCHANGED))

        def run():
            key_fill = gen_mask = None
            _held(rm, frames, progress=lambda t, d, n: job_progress(t, d, n))   # scans the removed masks once, up front
            if s["model"] == "void":   # VOID once over the work range, then its frames fill the mask (LaMa the rest)
                from engine.cleanup3d import void_engine as VE
                from engine.config import load_config
                from engine.jobs import JOBS

                def cancel_check() -> None:      # „Przerwij” kills VOID (as api/cleanup3d._cancel_check)
                    if JOBS.running():
                        JOBS.check()
                cfg = load_config()
                log = VE.void_root(work)
                log.mkdir(parents=True, exist_ok=True)
                runner = VE.void_runner(cfg, log=log / "void.log")

                def union(fns):
                    return (lambda f: np.max(np.stack([np.asarray(fn(int(f)), np.float32) for fn in fns]), axis=0)) if fns else None
                vr = VE.void_fill(work, frames, plate, union(rm), runner, progress=lambda t, d, n: job_progress(t, d, n),
                                  cancel=cancel_check, keep_fn=union(keep), linear=linear,
                                  prompt=(s.get("prompt") or "").strip() or VE.VOID_PROMPT, ffmpeg=VE._ffmpeg(cfg),
                                  client=comfy_client())
                key_fill = cu.VoidKeyFill(work, vr, w, h, linear)
            elif s["model"] in cu.ANCHOR_MODELS:   # MiniMax-H3 / Wan 2.2 Fun VACE over the work range
                from engine import h3_fill
                from engine.config import load_config

                # the MODEL's mask = „Zapas generowania” (user decision 2026-11-13); the composite band (cu.mask_fn)
                # only decides what of its picture goes in (the Cleaner, gen_mask=)
                rmask = cu.gen_mask_fn(s, rm, keep)
                gen_mask = rmask
                from engine import h3_anchors as HA
                anchors = {}
                ref_only = bool(s.get("anchor_ref_only")) and s.get("anchor") == "frame"
                if ref_only and s["model"] not in cu.REF_ONLY_MODELS:
                    raise RuntimeError("LTX-2.5 nie ma wejścia referencji całego klipu – wyłącz „tylko referencja” "
                                       "(kotwice przypięte w klatkach działają) albo wybierz MiniMax-H3 / Wan 2.2 Fun VACE")
                lst = [a for a in _ensure_anchors(work, s, frames, rmask) if a["frame"] in frames]
                lst.sort(key=lambda a: a["kind"] != "main")   # the main one first (= the <Picture 1> reference)
                transition = HA.transition_of(s, lst)
                morph = HA.morphs(s, lst) and not HA.is_transition(lst)   # a morph of linked anchors: the description
                for i, a in enumerate(lst):
                    job_progress(f"Kotwica F{a['frame']} ({a['kind']})", i, max(1, len(lst)))
                    anchors[a["frame"]] = _anchor_image(work, s, a, s["anchors"], plate, rmask, linear, frames)
                if lst:
                    cu.save_settings(work, s)                 # generations made here are now the chosen ones
                text = (s.get("h3_prompt") or s.get("prompt")) if transition and not morph else s.get("prompt")
                pm = s.get("prompt_mode") or "remove"
                if s["model"] == "vace":   # engine/vace_fill.py: the anchors are given frames of its control video
                    from engine import vace_fill
                    got = vace_fill.run(comfy_client(), work / "comfy_io", load_config(), frames, plate, rmask,
                                        vace_fill.prompt_for(text, pm, transition, morph), linear=linear,
                                        progress=lambda t, d, n: job_progress(t, d, n), anchors=anchors, ref_only=ref_only,
                                        seed=int(s.get("seed") or 42), short_edge=int(s.get("vace_res") or 720),
                                        reference=not transition, fast=s.get("vace_speed") != "quality",
                                        negative=_removed_noun(s))   # the removed object, negative prompt only
                    key_fill = vace_fill.VaceFill(got, w, h, linear)
                elif s["model"] == "ltx":   # engine/ltx_fill.py: IC-LoRA inpaint, the anchors as keyframe guides
                    from engine import ltx_fill
                    got = ltx_fill.run(comfy_client(), work / "comfy_io", load_config(), frames, plate, rmask,
                                       ltx_fill.prompt_for(text, pm, transition, morph), linear=linear,
                                       progress=lambda t, d, n: job_progress(t, d, n), anchors=anchors,
                                       seed=int(s.get("seed") or 42))
                    key_fill = ltx_fill.LtxFill(got, w, h, linear)
                else:
                    clash_notes = [HA.clash_note(c) for c in HA.token_clashes(lst, frames)]
                    got = h3_fill.run(comfy_client(), work / "comfy_io", load_config(), frames, plate, rmask,
                                      h3_fill.h3_prompt(text, pm, transition, morph), linear=linear,
                                      progress=lambda t, d, n: job_progress(t, d, n), anchors=anchors, ref_only=ref_only,
                                      seed=int(s.get("seed") or 42), reference=not transition,
                                      strength=h3_fill.run_strength(s.get("h3_strength"), pm))
                    key_fill = h3_fill.H3Fill(got, w, h, linear)
                    key_fill.clash_notes = clash_notes
                # the model's own frames next to the version (its canvas size): „⤴ Upscale” works from them
                (out / "model").mkdir(parents=True, exist_ok=True)
                for mf, img in got.items():
                    cv2.imwrite(str(out / "model" / f"m_{int(mf):04d}.png"), img)
                if anchors:   # the anchors the clip started from, next to the version (what H3 was pinned to)
                    (out / "anchors").mkdir(parents=True, exist_ok=True)
                    for af, img in anchors.items():
                        cv2.imwrite(str(out / "anchors" / f"anchor_{af:04d}.png"), img)
                    key_fill.anchors = sorted(anchors)
            cl = cu.Cleaner(s, w, h, linear, plate=plate, remove_alphas=lambda f: [fn(f) for fn in rm],
                            keep_alphas=lambda f: [fn(f) for fn in keep], clean=clean, auto_clean=auto,
                            inpaint=_inpainter(work, s), progress=lambda t, d, n: job_progress(t, d, n),
                            keep_names=cu.names_of(keep), key_fill=key_fill,
                            plate_file=src, gen_mask=gen_mask)
            if isinstance(key_fill, cu.VoidKeyFill):
                cl.report["void_run"] = key_fill.run.get("id")
            if getattr(key_fill, "anchors", None):
                cl.report["anchors"] = list(key_fill.anchors)
            cl.notes.extend(getattr(key_fill, "clash_notes", None) or [])   # anchors sharing a MiniMax-H3 token
            if s.get("input_plate"):   # „Wejście” (2026-11-15): made on top of an earlier version
                cl.report["input_plate"] = s["input_plate"]
                if in_miss:
                    cl.notes.append(f"Wejście {s['input_plate']} nie ma {cu.frame_ranges(in_miss)} – te klatki z Beauty")
            _held(rm, frames, cl)
            return cl.run(frames, out, ext)
        try:
            rj = await asyncio.get_running_loop().run_in_executor(None, run)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        plates = _set_cleaned(out, rj.get("comp_ext") or cu.file_ext(ext))
        return {"success": True, "version": out.name, "dir": str(out).replace("\\", "/"), "plate": CLEANED_PLATE,
                "plates": plates,
                "sources": info, **{k: rj[k] for k in ("counts", "backend", "notes", "seconds", "n_frames")},
                "kept_check": rj.get("kept_check"), "held": rj.get("held")}

    # what api/cleanup3d.py shares with this router (the same plates, alphas and fill sources for both renders)
    return router, {"need": _need, "alpha_loaders": _alpha_loaders, "held": _held, "fill_sources": _fill_sources,
                    "inpainter": _inpainter, "set_cleaned": _set_cleaned, "read_plate": _read_plate,
                    "frame_file": frame_file, "plate": CLEANED_PLATE, "work_frames": lambda: list(work_frames())}
