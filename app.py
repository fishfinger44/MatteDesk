# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""app.py - FastAPI backend server for MatteDesk.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 3, 10, 11, 12: Stage E1):
- Runs locally at 127.0.0.1:8765
- Project lifecycle endpoints (/api/project/open, /api/project/current, /api/project/generate_proxies)
- Frame proxy serving (/api/frame/{frame_num})
- Doctor environment endpoint (/api/doctor)
- WebSocket event stream (/ws/events)
- Serves interactive browser frontend
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # must precede the first `import cv2` (EXR alpha / plates)
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfy_client import ComfyClient
import engine.config as engine_config
from engine.config import load_config
from engine.jobs import JOBS, JobBusy, JobCancelled
from doctor import EnvironmentDoctor
from engine.coords import CoordTransformer, FrameMapper
from engine.clean import CleanReport, DeflickerMetrics, NukeDeflickerGroupGenerator, PlateCleaner
from engine.camera import Camera2DData, Camera2DTracker, NukeCamera2DGroupGenerator
from engine.ingest import VIDEO_EXT, MediaInspector, ProjectCreator, ProxyGenerator, decode_video
from engine.project import ShotProject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matte_desk.app")



@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _reopen_last_project()
    try:   # ComfyUI chosen in the Doctor, started in the background when it is not running (comfy_autostart)
        from engine import comfy_launch
        comfy_launch.autostart(load_config())
    except Exception as e:  # never block the MatteDesk start
        logging.getLogger("matte_desk").warning("autostart ComfyUI: %s", e)
    yield


def _read_version() -> str:
    try:
        return (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip() or "0"
    except OSError:
        return "0"


APP_VERSION = _read_version()   # file VERSION = single source (bump: tools/bump_version.py)
app = FastAPI(title="MatteDesk", version=APP_VERSION, lifespan=_lifespan)


_STARTED_AT = __import__("time").time()


def _code_changed_since_start() -> bool:
    """True when the Python code on disk is newer than this server process (git pull / edits without a restart)."""
    root = Path(__file__).resolve().parent
    files = [root / "app.py", root / "doctor.py", root / "comfy_client.py", root / "VERSION", *(root / "engine").glob("*.py")]
    try:
        return max(f.stat().st_mtime for f in files if f.exists()) > _STARTED_AT + 1
    except ValueError:
        return False


@app.get("/api/version")
def get_version():
    """Running version + whether the server runs older code than on disk (the UI then asks for a restart)."""
    return {"version": APP_VERSION, "on_disk": _read_version(), "stale": _code_changed_since_start()}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _api_no_heuristic_cache(request, call_next):
    """/api/* URLs (frames, object PNGs, reports) are the same across projects, and without Cache-Control the browser
    treats an old FileResponse (Last-Modified days ago) as fresh and shows the PREVIOUS project's frame. no-cache =
    always revalidate (ETag / Last-Modified), unchanged images still come from the cache via 304."""
    response = await call_next(request)
    path = request.url.path
    # also the page and its scripts: an old cached index.html / *.js mixed with new files broke the page init after an
    # update (buttons dead, "v…" in the corner) – always revalidate, unchanged files still come back as 304
    if (path.startswith("/api/") or path.startswith("/static/") or path == "/") and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response

REPO_DIR = Path(__file__).resolve().parent
STATIC_DIR = REPO_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Application state
STATE: Dict[str, Any] = {
    "project": None,
    "manifest": None,
    "manifest_path": None,
    "work_dir": None,
    "shot_project": None,
    "frame_mapper": None,
    "coord_transformer": None,
    "comfy_client": None,
    "active_plate": "beauty"
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()


import contextvars
import functools
import inspect

_IN_JOB: contextvars.ContextVar = contextvars.ContextVar("matte_desk_in_job", default=None)
# objects that every object job touches as a whole (export reads all active runs)
_ALL_OBJECTS_JOBS = {"export_mattes", "recompute_dependents"}


def long_job(label: str, quiet: bool = False):
    """Heavy endpoint (GPU / Comfy / long CPU): one at a time. Registers the job (status bar, GET /api/jobs);
    another heavy request meanwhile -> HTTP 409 "Trwa: …"; a user cancel ("Przerwij") -> HTTP 409 "przerwane".
    Nested calls (e.g. 'Przelicz zależne' -> run + finalize) run inside the outer job."""

    def deco(fn):
        is_async = inspect.iscoroutinefunction(fn)

        async def call(*args, **kwargs):
            if is_async:
                return await fn(*args, **kwargs)
            ctx = contextvars.copy_context()
            return await asyncio.get_running_loop().run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if _IN_JOB.get() is not None:
                return await call(*args, **kwargs)
            try:   # fields of a request model (e.g. {plate} of PlateQARequest) fill the label too
                fmt = {k: v for m in kwargs.values() if isinstance(m, BaseModel) for k, v in m.__dict__.items()}
                text = label.format(**{**fmt, **kwargs})
            except (KeyError, IndexError, AttributeError):
                text = label
            obj = "*" if fn.__name__ in _ALL_OBJECTS_JOBS else kwargs.get("name")
            try:
                jid = JOBS.try_start(fn.__name__, text, obj, quiet=quiet)
            except JobBusy as e:
                raise HTTPException(status_code=409, detail=str(e))
            token = _IN_JOB.set(jid)
            try:
                res = await call(*args, **kwargs)
            except JobCancelled:
                JOBS.finish(jid, "cancelled")
                raise HTTPException(status_code=409, detail=f"{text}: przerwane przez użytkownika")
            except HTTPException as e:
                cancelled = JOBS.cancelled()
                JOBS.finish(jid, "cancelled" if cancelled else "error",
                            str(e.detail.get("detail", e.detail)) if isinstance(e.detail, dict) else str(e.detail))
                if cancelled:
                    raise HTTPException(status_code=409, detail=f"{text}: przerwane przez użytkownika")
                raise
            except Exception as e:
                JOBS.finish(jid, "error", str(e))
                raise
            finally:
                _IN_JOB.reset(token)
            JOBS.finish(jid, "done")
            return res
        wrapper.md_long_job = {"label": label, "quiet": quiet}   # docs/AGENT_API.md (tools/md.py api-doc)
        return wrapper
    return deco


def object_edit(fn):
    """Edits of an object (keys, zones, settings, crops…) are refused while that object is being computed –
    the running job read its inputs at the start and would overwrite / mix them."""
    def check(kwargs):
        j = JOBS.busy_object(kwargs.get("name", ""))
        if j:
            raise HTTPException(status_code=409, detail=f"Obiekt „{kwargs.get('name')}” jest teraz liczony ({j['label']}) – "
                                                         "zmiany możliwe po zakończeniu albo po „Przerwij”")
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*args, **kwargs):
            check(kwargs)
            return await fn(*args, **kwargs)
        awrapper.md_object_edit = True
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        check(kwargs)
        return fn(*args, **kwargs)
    wrapper.md_object_edit = True
    return wrapper


def when_idle(what: str):
    """Project-level changes (switch project, plates, installs) only when nothing is being computed."""
    def deco(fn):
        def check():
            run = JOBS.running()
            if run:
                raise HTTPException(status_code=409, detail=f"{what}: niemożliwe w trakcie liczenia ({run[-1]['label']}) – "
                                                             "poczekaj albo kliknij „Przerwij”")
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                check()
                return await fn(*args, **kwargs)
            awrapper.md_when_idle = what
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            check()
            return fn(*args, **kwargs)
        wrapper.md_when_idle = what
        return wrapper
    return deco


@app.get("/api/jobs")
def list_jobs():
    """What is being computed now (status bar)."""
    return {"jobs": JOBS.snapshot()}


_DETAILS = None


@app.get("/api/jobs/details")
def jobs_details():
    """'Szczegóły' of the running job (Z19): GPU (nvidia-smi; null without it), Comfy RAM / VRAM / queue (foreign prompts),
    our step with frames and ETA, Comfy state of our prompt. Cached <= 1 s."""
    global _DETAILS
    from engine.sysinfo import DetailsCache, job_details
    if _DETAILS is None:
        _DETAILS = DetailsCache(1.0)

    def build():
        try:
            client = STATE.get("comfy_client") or ComfyClient.from_config()
        except Exception:
            client = None
        return job_details(client, JOBS.snapshot())
    return _DETAILS.get(build)


@app.post("/api/jobs/cancel")
def cancel_jobs():
    """'Przerwij': stop the running job(s) at the next safe point; our Comfy prompt is removed / interrupted."""
    QUEUE_STATE["stop"] = True
    return {"success": True, "cancelled": JOBS.cancel()}


# Request models
class OpenProjectRequest(BaseModel):
    manifest_path: str


class CreateProjectRequest(BaseModel):
    shot_name: str
    work_dir: str
    beauty_pattern: str
    first_frame: int
    last_frame: int
    plate_w: int
    plate_h: int
    fps: float = 24.0
    colorspace: str = "ACES2065-1"
    source_type: str = "exr_seq"
    bit_depth: int = 16
    backlit_pattern: str = ""
    cleanplate_pattern: str = ""


class UpdatePlateRequest(BaseModel):
    plate_name: str
    pattern: str
    generate_proxies: bool = False


class InspectPathRequest(BaseModel):
    path: str


class PickFolderRequest(BaseModel):
    initial_dir: Optional[str] = ""
    title: Optional[str] = "Wybierz katalog roboczy projektu"


class InspectWorkingDirRequest(BaseModel):
    dir_path: str


class OpenOrCreateProjectRequest(BaseModel):
    work_dir: str
    shot_name: Optional[str] = ""
    plates: Dict[str, str] = {}
    fps: Optional[float] = 24.0
    colorspace: Optional[str] = "ACES2065-1"


def _activate_project(data: Dict[str, Any]) -> Dict[str, Any]:
    """Store an opened ShotProject in STATE. STATE["manifest"] is the *effective* manifest
    (user manifest + project.json overrides); the manifest file itself is never written."""
    proj: ShotProject = data["shot_project"]
    project = proj.to_dict()
    STATE["shot_project"] = proj
    STATE["project"] = project
    STATE["manifest"] = proj.effective_manifest()
    STATE["manifest_path"] = str(proj.manifest_path.resolve())
    STATE["work_dir"] = str(proj.work_dir)
    STATE["frame_mapper"] = FrameMapper(project["first_frame"], project["last_frame"])
    STATE["coord_transformer"] = CoordTransformer(project["plate_w"], project["plate_h"])
    STATE["comfy_client"] = ComfyClient.from_config()
    STATE["display_cache"] = {}   # Z8 tiles: full-frame display images of the previous project are stale
    return project


def _refresh_manifest_view() -> Dict[str, Any]:
    proj: Optional[ShotProject] = STATE.get("shot_project")
    if proj is None:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    STATE["manifest"] = proj.effective_manifest()
    STATE["project"] = proj.to_dict()
    return STATE["manifest"]


def _plate_size() -> tuple:
    proj: Optional[ShotProject] = STATE.get("shot_project")
    if proj is None:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    return proj.plate_size


@app.get("/api/doctor")
def get_doctor_report():
    doc = EnvironmentDoctor()
    report = doc.run_all_checks()
    data = report.to_dict()
    data["config"] = {
        "comfy_host": doc.config.get("comfy_host", "127.0.0.1"),
        "comfy_port": doc.config.get("comfy_port", 8000),
        "comfy_flavor": doc.config.get("comfy_flavor", "desktop")
    }
    return data


class ComfyFlavorRequest(BaseModel):
    flavor: str                                   # desktop | git (ComfyUI from git; old name "portable" accepted)
    comfy_dir: Optional[str] = None               # git: folder with main.py
    comfy_port: Optional[int] = None
    shared_models: Optional[bool] = None          # git: use ComfyUI Desktop's models folder
    autostart: Optional[bool] = None              # start ComfyUI with MatteDesk (comfy_autostart)


@app.get("/api/config/comfy")
def get_comfy_config():
    """Doctor: which ComfyUI is used (Desktop / from git), its folders, port and autostart."""
    cfg = load_config()
    local = engine_config._read(engine_config.LOCAL)
    return {"flavor": cfg.get("comfy_flavor"), "comfy_dir": cfg.get("comfy_dir"), "models_dir": cfg.get("models_dir"),
            "comfy_port": cfg.get("comfy_port"), "git": local.get("comfy_git") or {},
            "autostart": bool(cfg.get("comfy_autostart", True))}


@app.post("/api/config/comfy")
@when_idle("Zmiana ComfyUI")
def set_comfy_config(req: ComfyFlavorRequest):
    """Doctor's switch: ComfyUI Desktop <-> ComfyUI from git. Installs/downloads go to the active one."""
    prof = {k: v for k, v in (("comfy_dir", req.comfy_dir), ("comfy_port", req.comfy_port),
                              ("shared_models", req.shared_models)) if v is not None}
    try:
        engine_config.set_comfy_flavor(req.flavor, prof or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.autostart is not None:
        engine_config.write_local({"comfy_autostart": bool(req.autostart)})
    return get_comfy_config()


@app.get("/api/comfy/launch")
def comfy_launch_state():
    """State of the ComfyUI start by MatteDesk: idle | starting | running | failed (+ how / error)."""
    from engine import comfy_launch
    return dict(comfy_launch.STATE)


@app.post("/api/comfy/start")
def comfy_start():
    """Doctor button "Uruchom ComfyUI": start the chosen ComfyUI (Desktop app / ComfyUI from git) in the background."""
    from engine import comfy_launch
    return comfy_launch.start_comfy(load_config())


@app.get("/api/install/progress")
def install_progress():
    """Live state of the running "Zainstaluj" (step, file, done/total bytes) - the Doctor polls it for the progress bar."""
    from engine import installer
    return dict(installer.PROGRESS)


@app.post("/api/install/{what}")
@when_idle("Instalacja")
def install_component(what: str):
    """Doctor's "Zainstaluj" button (user-triggered): our Comfy node packages, third-party packs, weights, CoTracker3."""
    from engine import installer, tools_install
    cfg = load_config()
    installer.progress_begin(what)
    try:
        tools_install.prepare_env(cfg)   # git / ffmpeg installed by the Doctor -> on PATH for git clone, pip git+https
        if what in tools_install.TOOLS:
            res = tools_install.install_tool(cfg, what)
        elif what == "restart_comfy":
            res = installer.restart_comfy(cfg)
        elif what == "nodes":
            res = installer.install_nodes(cfg)
        elif what == "cotracker":
            res = installer.install_cotracker(cfg)
        elif what == "vitmatte":
            res = installer.install_vitmatte(cfg)
        elif what == "ltx":
            res = installer.install_ltx(cfg)
        elif what == "videomama":
            res = installer.install_videomama(cfg)
        elif what in ("sam3", "matanyone2_weights", "lama") or what in installer.HF_FOLDERS:
            res = installer.install_model(cfg, what)
        elif what in installer.GIT_NODE_PACKS:
            res = installer.install_git_node(cfg, what)
        else:
            installer.progress_end(False, "unknown")
            raise HTTPException(status_code=404, detail="ffmpeg | git | nodes | cotracker | vitmatte | ltx | videomama | sam3 | matanyone2_weights | lama | matanyone2_node | inpaint_nodes | vhs_nodes | restart_comfy")
    except HTTPException:
        raise
    except Exception as e:  # report, the doctor re-checks afterwards
        installer.progress_end(False, str(e))
        raise HTTPException(status_code=400, detail=str(e))
    installer.progress_end(bool(res.get("success", True)))
    if res.get("restart_comfy") and res.get("success", True):
        installer.mark_pending_restart(what)
    return res


@app.post("/api/project/open")
@when_idle("Otwarcie projektu")
async def open_project(req: OpenProjectRequest):
    creator = ProjectCreator()
    try:
        data = creator.init_project_from_manifest(req.manifest_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    manifest = data["manifest"]
    project = _activate_project(data)
    _remember_work_dir(Path(req.manifest_path).resolve().parent)
    first_f = project["first_frame"]
    last_f = project["last_frame"]

    await ws_manager.broadcast({
        "type": "project_opened",
        "shot": manifest.get("shot"),
        "first_frame": first_f,
        "last_frame": last_f
    })

    return {
        "success": True,
        "project": data["project"],
        "manifest": manifest,
        "work_dir": data["work_dir"]
    }


@app.post("/api/project/new")
@when_idle("Nowy projekt")
async def create_new_project(req: CreateProjectRequest):
    creator = ProjectCreator()
    manifest_file = Path(req.work_dir) / f"shot_manifest_{req.shot_name}.json"
    if manifest_file.exists():
        raise HTTPException(status_code=409, detail="Manifest ujęcia już istnieje – otwórz go zamiast tworzyć nowy (manifest jest tylko do odczytu)")
    creator.create_manifest(
        shot_name=req.shot_name,
        work_dir=req.work_dir,
        beauty_pattern=req.beauty_pattern,
        first_frame=req.first_frame,
        last_frame=req.last_frame,
        plate_w=req.plate_w,
        plate_h=req.plate_h,
        fps=req.fps,
        colorspace=req.colorspace,
        source_type=req.source_type,
        bit_depth=req.bit_depth,
        backlit_pattern=req.backlit_pattern,
        cleanplate_pattern=req.cleanplate_pattern
    )

    data = creator.init_project_from_manifest(manifest_file)
    project = _activate_project(data)
    _remember_work_dir(req.work_dir)

    return {"success": True, "manifest_path": str(manifest_file), "project": project}


try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


@app.post("/api/dialog/pick_folder")
async def pick_folder_dialog(req: PickFolderRequest):
    loop = asyncio.get_running_loop()
    def _ask():
        if not filedialog or not tk:
            return ""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            init_d = req.initial_dir if req.initial_dir and Path(req.initial_dir).exists() else None
            folder = filedialog.askdirectory(title=req.title, initialdir=init_d)
            root.destroy()
            return folder or ""
        except Exception as e:
            logger.warning(f"Native folder dialog error: {e}")
            return ""
    folder = await loop.run_in_executor(None, _ask)
    return {"folder": folder.replace("\\", "/")}


class PickFileRequest(BaseModel):
    initial_dir: Optional[str] = ""
    title: Optional[str] = "Choose a file"


@app.post("/api/dialog/pick_file")
async def pick_file_dialog(req: PickFileRequest):
    """Native file picker (a single image / video: target background BG, MP4 plates); "" when cancelled."""
    loop = asyncio.get_running_loop()
    def _ask():
        if not filedialog or not tk:
            return ""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            init_d = req.initial_dir if req.initial_dir and Path(req.initial_dir).exists() else None
            path = filedialog.askopenfilename(title=req.title, initialdir=init_d, filetypes=[
                ("Images / video", "*.exr *.png *.jpg *.jpeg *.tif *.tiff *.dpx *.mp4 *.mov *.mxf *.avi *.mkv"), ("All files", "*.*")])
            root.destroy()
            return path or ""
        except Exception as e:
            logger.warning(f"Native file dialog error: {e}")
            return ""
    path = await loop.run_in_executor(None, _ask)
    return {"path": path.replace("\\", "/")}


@app.post("/api/project/inspect_working_dir")
def inspect_working_dir_endpoint(req: InspectWorkingDirRequest):
    info = ProjectCreator.inspect_working_dir(req.dir_path)
    return {"success": True, "info": info}


def _open_work_dir(req: OpenOrCreateProjectRequest) -> Dict[str, Any]:
    """Open the shot in `req.work_dir` (manifest found -> read only, plates that differ -> project.json) or create
    a new manifest (wizard). Activates the project in STATE and returns the opened data. Shared by the
    open_or_create endpoint and the startup reopen of the last project."""
    work_p = Path(req.work_dir)
    if not work_p.exists():
        work_p.mkdir(parents=True, exist_ok=True)

    creator = ProjectCreator()
    inspection = creator.inspect_working_dir(work_p)

    if inspection.get("manifest_found"):
        # Existing manifest = user input, read-only. Plates given in the form that differ from
        # the manifest are stored as project overrides in project.json.
        manifest_file = Path(inspection["manifest_path"])
        data = creator.init_project_from_manifest(manifest_file)
        proj: ShotProject = data["shot_project"]
        manifest_plates = proj.manifest.get("plates") or {}
        for k, v in req.plates.items():
            if v and v.strip():
                resolved = MediaInspector.resolve_sequence_pattern(v.strip()).get("pattern", v.strip())
                if manifest_plates.get(k) != resolved:
                    proj.set_plate(k, resolved)
        data["manifest"] = proj.effective_manifest()
    else:
        beauty_raw = req.plates.get("beauty", "").strip()
        if not beauty_raw:
            raise HTTPException(status_code=400, detail="Beauty Plate is required to initialize project")

        shot_name = req.shot_name.strip() or work_p.name or "SHOT"
        if Path(beauty_raw).suffix.lower() in VIDEO_EXT and Path(beauty_raw).is_file():
            # video -> sequence in the shot work folder (ffmpeg), bit depth from the header (SPEC 1.0)
            from engine.project import work_subdir_name
            beauty_info = decode_video(beauty_raw, work_p / work_subdir_name() / "source" / Path(beauty_raw).stem)
            if not req.fps or req.fps == 24.0:
                req.fps = beauty_info["fps"]
            if not req.colorspace or req.colorspace == "ACES2065-1":
                req.colorspace = beauty_info.get("colorspace", "sRGB")
        else:
            beauty_info = MediaInspector.resolve_sequence_pattern(beauty_raw)
        beauty_pat = beauty_info.get("pattern", beauty_raw)

        first_f = beauty_info.get("first_frame", 1)
        last_f = beauty_info.get("last_frame", 100)
        plate_w = beauty_info.get("width")
        plate_h = beauty_info.get("height")
        if not plate_w or not plate_h:
            raise HTTPException(status_code=400, detail="Nie udało się odczytać rozdzielczości plate'u beauty – podaj ją ręcznie")
        colorspace = req.colorspace or beauty_info.get("colorspace", "ACES2065-1")
        source_type = beauty_info.get("source_type", "exr_seq")
        bit_depth = beauty_info.get("bit_depth", 16)

        cleanplate_pat = req.plates.get("cleanplate", "").strip()
        if cleanplate_pat:
            cleanplate_pat = MediaInspector.resolve_sequence_pattern(cleanplate_pat).get("pattern", cleanplate_pat)

        backlit_pat = req.plates.get("backlit", "").strip()
        if backlit_pat:
            backlit_pat = MediaInspector.resolve_sequence_pattern(backlit_pat).get("pattern", backlit_pat)

        creator.create_manifest(
            shot_name=shot_name,
            work_dir=str(work_p),
            beauty_pattern=beauty_pat,
            first_frame=first_f,
            last_frame=last_f,
            plate_w=plate_w,
            plate_h=plate_h,
            fps=req.fps or 24.0,
            colorspace=colorspace,
            source_type=source_type,
            bit_depth=bit_depth,
            backlit_pattern=backlit_pat,
            cleanplate_pattern=cleanplate_pat
        )

        manifest_file = work_p / f"shot_manifest_{shot_name}.json"
        data = creator.init_project_from_manifest(manifest_file)
        depth_pat = req.plates.get("depth", "").strip()
        if depth_pat:
            depth_pat = MediaInspector.resolve_sequence_pattern(depth_pat).get("pattern", depth_pat)
            data["shot_project"].set_plate("depth", depth_pat)
            data["manifest"] = data["shot_project"].effective_manifest()

    data["project"] = _activate_project(data)
    return data


def _remember_work_dir(work_dir) -> None:
    """Last / recent shot folders -> config.local.json (this machine). Never fails the open."""
    try:
        engine_config.remember_work_dir(str(work_dir))
    except Exception as e:  # pragma: no cover - disk / permission problems only
        logger.warning(f"Could not save the last project in config.local.json: {e}")


def _reopen_last_project() -> Optional[str]:
    """Startup: reopen `last_work_dir` from config.local.json (if it still has a shot manifest). Reads only;
    an error is logged, never raised."""
    try:
        last = engine_config.load_config().get("last_work_dir") or ""
        if not last or STATE.get("project"):
            return None
        wd = Path(last)
        if not wd.is_dir() or not any(wd.glob("shot_manifest_*.json")):
            logger.warning(f"Last project {last}: folder or shot_manifest_*.json missing - skipped")
            return None
        _open_work_dir(OpenOrCreateProjectRequest(work_dir=last, plates={}))
        logger.info(f"Otwarto ostatni projekt: {last}")
        return last
    except Exception as e:
        detail = getattr(e, "detail", None) or e
        logger.warning(f"Could not reopen the last project: {detail}")
        return None


@app.post("/api/project/open_or_create")
@when_idle("Otwarcie projektu")
async def open_or_create_project(req: OpenOrCreateProjectRequest):
    data = _open_work_dir(req)
    _remember_work_dir(req.work_dir)
    manifest = data["manifest"]
    project = data["project"]
    first_f = project["first_frame"]
    last_f = project["last_frame"]

    await ws_manager.broadcast({
        "type": "project_opened",
        "shot": manifest.get("shot"),
        "first_frame": first_f,
        "last_frame": last_f
    })

    return {
        "success": True,
        "project": project,
        "manifest": manifest,
        "work_dir": data["work_dir"]
    }


@app.get("/api/project/recent")
def get_recent_projects():
    """Recently opened shot folders (config.local.json, newest first) for the project dialog."""
    cfg = engine_config.load_config()
    recent = [d for d in (cfg.get("recent_work_dirs") or []) if isinstance(d, str) and d]
    return {"last_work_dir": cfg.get("last_work_dir") or "",
            "recent": [{"work_dir": d, "exists": Path(d).is_dir()} for d in recent[:engine_config.RECENT_MAX]]}


@app.get("/api/project/current")
def get_current_project():
    if not STATE["project"]:
        return {"loaded": False}
    return {
        "loaded": True,
        "project": STATE["project"],
        "manifest": STATE["manifest"],
        "work_dir": STATE["work_dir"]
    }


@app.post("/api/project/generate_proxies")
@long_job("Proxy ({plate})")
async def generate_proxies(plate: str = "beauty", max_side: int = 1920):
    if not STATE["manifest"]:
        raise HTTPException(status_code=400, detail="No project loaded")

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    plate_pattern = plates.get(plate)
    if not plate_pattern:
        raise HTTPException(status_code=400, detail=f"Plate '{plate}' has no path pattern in manifest")

    first_f = manifest.get("first", 1)
    last_f = manifest.get("last", 100)
    colorspace = manifest.get("colorspace", "ACES2065-1")
    proxy_dir = Path(STATE["work_dir"]) / "proxy" / plate

    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int):
        JOBS.progress(done, total)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({
                "type": "proxy_progress",
                "done": done,
                "total": total,
                "plate": plate
            }),
            loop
        )

    proxies = await loop.run_in_executor(
        None,
        lambda: ProxyGenerator.generate_plate_proxies(
            plate_pattern=plate_pattern,
            first_frame=first_f,
            last_frame=last_f,
            output_dir=proxy_dir,
            colorspace=colorspace,
            max_side=max_side,
            progress_cb=on_progress
        )
    )

    return {"success": True, "rendered_count": len(proxies), "proxy_dir": str(proxy_dir)}


@app.get("/api/project/plates")
def get_project_plates():
    if not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    shot = manifest.get("shot", "unknown")
    work_dir = STATE.get("work_dir")

    labels = {
        "beauty": "Beauty Plate",
        "cleanplate": "Clean Plate",
        "clean_plate": "Clean Plate",
        "backlit": "Backlit Plate",
        "depth": "Depth Plate",
        "bg": "Tło docelowe (BG, podgląd compu)"
    }

    results = []
    for name, pattern in plates.items():
        pattern_str = pattern or ""
        exists = False
        frame_count = 0
        if pattern_str:
            info = MediaInspector.resolve_sequence_pattern(pattern_str)
            exists = info.get("exists", False)
            frame_count = info.get("frame_count", 0)

        proxy_dir = Path(work_dir) / "proxy" / name if work_dir else None
        proxy_count = len(list(proxy_dir.glob("p_*.jpg"))) if proxy_dir and proxy_dir.is_dir() else 0

        results.append({
            "name": name,
            "label": labels.get(name, name),
            "pattern": pattern_str,
            "exists": exists,
            "frame_count": frame_count,
            "proxy_count": proxy_count,
            "is_primary": (name == "beauty")
        })

    return {"success": True, "shot": shot, "plates": results}


@app.post("/api/project/plate/inspect")
def inspect_plate_path(req: InspectPathRequest):
    info = MediaInspector.resolve_sequence_pattern(req.path)
    return {"success": True, "info": info}


@app.post("/api/project/plate/update")
@long_job("Plate: dekodowanie / proxy")
async def update_project_plate(req: UpdatePlateRequest):
    if not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    plate_name = req.plate_name.strip()
    if not plate_name:
        raise HTTPException(status_code=400, detail="Nazwa warstwy nie może być pusta")

    pattern = req.pattern.strip()
    info = MediaInspector.resolve_sequence_pattern(pattern)
    # Plates added in the tool go to project.json – the shot manifest is never written.
    STATE["shot_project"].set_plate(plate_name, info.get("pattern", pattern))
    manifest = _refresh_manifest_view()

    # Broadcast updated plates to all connected clients
    await ws_manager.broadcast({
        "type": "plates_updated",
        "shot": manifest.get("shot"),
        "plates": manifest.get("plates", {})
    })

    rendered_proxies = 0
    resolved_pat = manifest.get("plates", {}).get(plate_name, "")
    # the target background ("bg") is fitted per frame on demand (single image / sequence / video), no plate proxies
    if req.generate_proxies and resolved_pat and STATE.get("work_dir") and plate_name != "bg":
        first_f = manifest.get("first", 1)
        last_f = manifest.get("last", 100)
        colorspace = manifest.get("colorspace", "ACES2065-1")
        proxy_dir = Path(STATE["work_dir"]) / "proxy" / plate_name
        loop = asyncio.get_running_loop()
        proxies = await loop.run_in_executor(
            None,
            lambda: ProxyGenerator.generate_plate_proxies(
                plate_pattern=resolved_pat,
                first_frame=first_f,
                last_frame=last_f,
                output_dir=proxy_dir,
                colorspace=colorspace
            )
        )
        rendered_proxies = len(proxies)

    return {
        "success": True,
        "plate_name": plate_name,
        "pattern": resolved_pat,
        "manifest": manifest,
        "rendered_proxies": rendered_proxies,
        "info": info
    }


@app.delete("/api/project/plate/{plate_name}")
@when_idle("Usunięcie plate'u")
async def delete_project_plate(plate_name: str):
    if not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    if plate_name.lower() == "beauty":
        raise HTTPException(status_code=400, detail="Nie można usunąć głównej warstwy beauty")

    # Plate hidden / removed in project.json only; the manifest file stays untouched.
    STATE["shot_project"].remove_plate(plate_name)
    manifest = _refresh_manifest_view()

    await ws_manager.broadcast({
        "type": "plates_updated",
        "shot": manifest.get("shot"),
        "plates": manifest.get("plates", {})
    })

    return {"success": True, "manifest": manifest}


def _beauty_proxy_size() -> Tuple[int, int]:
    """(w, h) of the beauty proxy = the size every preview layer is drawn at (read from a proxy file, cached)."""
    key = STATE.get("work_dir")
    cached = STATE.get("beauty_proxy_size")
    if cached and cached[0] == key:
        return cached[1]
    m = STATE.get("manifest") or {}
    first = int(m.get("first", 1))
    try:
        img = cv2.imread(str(_proxy_path("beauty", first)), cv2.IMREAD_COLOR)
    except HTTPException:
        img = None
    if img is not None:
        size = (int(img.shape[1]), int(img.shape[0]))
    else:
        w, h = _plate_size()
        s = min(1920.0 / w, 1920.0 / h, 1.0)
        size = (max(1, round(w * s)), max(1, round(h * s)))
    STATE["beauty_proxy_size"] = (key, size)
    return size


def _bg_proxy_path(frame_num: int) -> Path:
    """Target background (plate role "bg", preview only): single image / sequence / video fitted to the beauty proxy."""
    from engine import bgplate
    manifest = STATE.get("manifest") or {}
    pattern = (manifest.get("plates") or {}).get(bgplate.BG_PLATE, "")
    if not pattern:
        raise HTTPException(status_code=404, detail="Brak tła docelowego (BG) – dodaj je w ⚙️ Plate'y")
    try:
        return bgplate.proxy_path(Path(STATE["work_dir"]), pattern, frame_num, int(manifest.get("first", 1)),
                                  _beauty_proxy_size(), manifest.get("colorspace", "ACES2065-1"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _proxy_path(plate: str, frame_num: int) -> Path:
    """Proxy JPEG of a plate frame, rendered on demand (like the old /api/frame)."""
    if plate == "bg":
        return _bg_proxy_path(frame_num)
    proxy_dir = Path(STATE["work_dir"]) / "proxy" / plate
    proxy_file = proxy_dir / f"p_{frame_num:04d}.jpg"
    if proxy_file.exists():
        return proxy_file
    manifest = STATE.get("manifest") or {}
    pattern = manifest.get("plates", {}).get(plate, "")
    if pattern:
        src_file = ProxyGenerator.resolve_frame_path(pattern, frame_num)
        if src_file.exists():
            ProxyGenerator.render_proxy_file(source_frame_path=src_file, dest_proxy_path=proxy_file,
                                             colorspace=manifest.get("colorspace", "ACES2065-1"))
            if proxy_file.exists():
                return proxy_file
    raise HTTPException(status_code=404, detail=f"Proxy frame {frame_num} not found for plate '{plate}'")


def _aligned_proxy_path(plate: str, frame_num: int) -> Path:
    """Proxy of an auxiliary plate warped by its alignment report (Align_<plate>.json) – the 'po dopasowaniu'
    preview (SPEC 4a pkt 2) without writing the <plate>_align sequence. Cached next to the proxies, invalidated when
    the report is newer. Transforms are measured in plate px and rescaled to the proxy."""
    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "align" / f"Align_{plate}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak pomiaru dopasowania dla '{plate}' – najpierw „Zmierz dopasowanie”")
    out_dir = work_dir / "proxy" / f"{plate}_aligned_preview"
    out = out_dir / f"p_{frame_num:04d}.jpg"
    if out.exists() and out.stat().st_mtime >= json_path.stat().st_mtime:
        return out
    from engine.align import AlignmentReport, PlateWarper
    report = AlignmentReport.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
    fres = next((f for f in report.frames if f.frame == frame_num), None)
    src = cv2.imread(str(_proxy_path(plate, frame_num)), cv2.IMREAD_COLOR)
    if src is None:
        raise HTTPException(status_code=404, detail="Brak proxy")
    if fres is None:
        raise HTTPException(status_code=404, detail=f"Klatka {frame_num} poza zakresem pomiaru (F{report.frame_range[0]}–F{report.frame_range[1]})")
    w, _h = _plate_size()
    sc = src.shape[1] / float(w)
    import copy
    fr = copy.deepcopy(fres)
    fr.dx, fr.dy = fres.dx * sc, fres.dy * sc
    if fres.transform_matrix is not None:
        M = np.array(fres.transform_matrix, dtype=np.float64)
        if M.shape == (2, 3):
            M = np.vstack([M, [0.0, 0.0, 1.0]])
        S = np.diag([sc, sc, 1.0])
        Mp = S @ M @ np.linalg.inv(S)
        fr.transform_matrix = (Mp[:2] if fres.method in ("similarity", "affine") else Mp).tolist()
    if fres.method == "local_stmap":
        # engine.align.compose_remap: the global model, then the alignment grid, both scaled to the proxy inside
        # (plate px -> proxy px); a missing grid (old report) = the global model alone
        gp = work_dir / "align" / f"grid_{plate}" / (fres.grid_rel_path or f"grid_{frame_num:04d}.exr")
        warped = PlateWarper.warp_frame(src, fres, field_path=gp if gp.exists() else None, plate_w=w)
    else:
        warped = PlateWarper.warp_frame(src, fr)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


@app.get("/api/frame/{frame_num}")
def get_proxy_frame(frame_num: int, plate: str = "beauty", aligned: int = 0):
    """Proxy JPEG of a frame. aligned=1 (auxiliary plates only): warped by the alignment report = preview of the
    result of 'Skoryguj plate' without writing anything to the plate folders."""
    if not STATE["work_dir"]:
        raise HTTPException(status_code=400, detail="No project loaded")
    if aligned:
        return FileResponse(_aligned_proxy_path(plate, frame_num), media_type="image/jpeg")
    return FileResponse(_proxy_path(plate, frame_num), media_type="image/jpeg")


TILE_MAX = 2048          # Z8: max side of a tile (plate px)
DISPLAY_CACHE_FRAMES = 3  # full-frame display images kept in memory (4096x2731x3 B ~ 33 MB each)


def _full_display_u8(plate: str, frame_num: int) -> np.ndarray:
    """Full-resolution display image (BGR uint8) of a plate frame – the SAME conversion as the proxy
    (`ProxyGenerator.load_frame_rgb`), so the tile has the proxy's colours. LRU of DISPLAY_CACHE_FRAMES in STATE."""
    cache = STATE.setdefault("display_cache", {})
    key = (plate, int(frame_num))
    img = cache.get(key)
    if img is None:
        manifest = STATE.get("manifest") or {}
        pattern = manifest.get("plates", {}).get(plate, "")
        src = ProxyGenerator.resolve_frame_path(pattern, frame_num) if pattern else None
        if not src or not src.exists():
            raise HTTPException(status_code=404, detail=f"Brak klatki {frame_num} plate'u '{plate}'")
        rgb = ProxyGenerator.load_frame_rgb(src, colorspace=manifest.get("colorspace", "ACES2065-1"))
        img = np.ascontiguousarray(rgb[..., ::-1])
        while len(cache) >= DISPLAY_CACHE_FRAMES:
            cache.pop(next(iter(cache)))
        cache[key] = img
    return img


def _tile_rect(x: int, y: int, w: int, h: int, full_w: int, full_h: int) -> Tuple[int, int, int, int]:
    if w <= 0 or h <= 0 or w > TILE_MAX or h > TILE_MAX:
        raise HTTPException(status_code=400, detail=f"Kafel: 1 ≤ w, h ≤ {TILE_MAX} px")
    if x < 0 or y < 0 or x + w > full_w or y + h > full_h:
        raise HTTPException(status_code=400, detail=f"Kafel poza klatką ({full_w}×{full_h} px)")
    return int(x), int(y), int(w), int(h)


def _png_response(img: np.ndarray):
    from fastapi.responses import Response
    ok, buf = cv2.imencode(".png", img)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/frame/{frame_num}/tile")
def get_frame_tile(frame_num: int, x: int, y: int, w: int, h: int, plate: str = "beauty", fmt: str = "jpg"):
    """Z8: full-resolution crop [x, y, w, h] (plate px) of a plate frame for zoom > 100 % of the proxy – the same
    display conversion as the proxy. JPEG q95 by default, fmt=png = lossless (tests: pixels == load_frame_rgb crop)."""
    if not STATE["work_dir"]:
        raise HTTPException(status_code=400, detail="No project loaded")
    img = _full_display_u8(plate, frame_num)
    x, y, w, h = _tile_rect(x, y, w, h, img.shape[1], img.shape[0])
    crop = np.ascontiguousarray(img[y:y + h, x:x + w])
    if fmt == "png":
        return _png_response(crop)
    from fastapi.responses import Response
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/project/stages")
def project_stages():
    """Stage 0 (Plate'y) of the shot: what exists for the stage bar / panel – plates & proxies, alignment reports of
    the auxiliary plates (backlit / cleanplate: the only ones alignment applies to), cleaning (deflicker / denoise,
    any plate), camera 2D, depth. Nothing here blocks the object stages – it is information."""
    if not STATE.get("manifest") or not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    from engine.align import stale_report_reason
    manifest = STATE["manifest"]
    work_dir = Path(STATE["work_dir"])
    plates = {k: v for k, v in (manifest.get("plates") or {}).items() if v}
    bg_pattern = plates.pop("bg", "")   # target background: preview of the comp only, not a plate of the shot
    bg: Dict[str, Any] = {"pattern": bg_pattern}
    if bg_pattern:
        from engine import bgplate
        src = bgplate.source_info(bg_pattern)
        bg.update({"kind": src["kind"], "count": src["count"], "exists": bool(src["count"])})
    out_plates = []
    for name, pattern in plates.items():
        info = MediaInspector.resolve_sequence_pattern(pattern) if pattern else {}
        pdir = work_dir / "proxy" / name
        out_plates.append({"name": name, "pattern": pattern, "exists": bool(info.get("exists")),
                           "frame_count": int(info.get("frame_count") or 0),
                           "proxy_count": len(list(pdir.glob("p_*.jpg"))) if pdir.is_dir() else 0,
                           "aux": name in ("backlit", "cleanplate", "clean_plate", "cleanplate_auto") or name.endswith("_align"),
                           "aligned_of": name[:-6] if name.endswith("_align") else None})
    align = {}
    for name in ("backlit", "cleanplate"):
        if name not in plates:
            continue
        jp = work_dir / "align" / f"Align_{name}.json"
        entry: Dict[str, Any] = {"measured": jp.exists(), "applied": f"{name}_align" in plates}
        if jp.exists():
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                entry.update({"method": d.get("method"), "residual_median": d.get("residual_median"),
                              "residual_max": d.get("residual_max"), "needs_correction": d.get("needs_correction"),
                              "frame_range": d.get("frame_range"), "recommendation": d.get("recommendation"),
                              "nk": (work_dir / "align" / f"Align_{name}.nk").exists(),
                              "frames_over": [f["frame"] for f in d.get("frames", []) if (f.get("residual_px") or 0) > 0.3],
                              "created_at": d.get("created_at"),
                              "device": d.get("device") or "", "device_note": d.get("device_note") or "",
                              "stale": stale_report_reason(d)})
            except (OSError, ValueError):
                entry["measured"] = False
        align[name] = entry
    clean = {}
    for name in plates:
        rp = work_dir / "deflicker" / f"Deflicker_{name}.json"
        e: Dict[str, Any] = {"done": rp.exists()}
        if rp.exists():
            try:
                d = json.loads(rp.read_text(encoding="utf-8"))
                e.update({"reduction_pct": d.get("reduction_pct"), "noise_reduction_pct": d.get("noise_reduction_pct"),
                          "stages": d.get("stages_applied"), "nk": bool(d.get("nk_group_path")),
                          "export_mode": d.get("export_mode")})
            except (OSError, ValueError):
                e["done"] = False
        clean[name] = e
    cam = work_dir / "camera" / "camera2d.json"
    camera: Dict[str, Any] = {"done": cam.exists()}
    if cam.exists():
        try:
            d = json.loads(cam.read_text(encoding="utf-8"))
            camera.update({"motion_type": d.get("motion_type"), "residual_median_px": d.get("residual_median_px")})
        except (OSError, ValueError):
            pass
    dep = work_dir / "depth" / "depth_report.json"
    depth: Dict[str, Any] = {"done": dep.exists()}
    nobj = len(list((work_dir / "objects").glob("*/object.json"))) if (work_dir / "objects").is_dir() else 0
    try:
        n_frames = len(_shot_frames())
    except Exception:
        n_frames = 0
    gate = _plate_gate()     # Z26: plate QA per plate + the stage-1 gate
    return {"plates": out_plates, "bg": bg, "align": align, "clean": clean, "camera": camera, "depth": depth,
            "objects": nobj, "n_frames": n_frames, "mask_plate_denoise": _mask_plate_denoise_info(),
            "beauty_proxy_ok": any(p["name"] == "beauty" and p["proxy_count"] > 0 for p in out_plates),
            "plate_qa": gate["plates"], "plate_qa_block": gate["block"]}


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# -----------------------------------------------------------------------------
# Plate Alignment Endpoints (Stage E1b, SPEC Section 4a, 10, 12)
# -----------------------------------------------------------------------------

class AlignMeasureRequest(BaseModel):
    plate: str
    aux_plate_type: str = "backlit"
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    device: Optional[str] = None   # auto | gpu | cpu; None = config `align_device` (UI: Auto / GPU / CPU)


class AlignExportNkRequest(BaseModel):
    plate: str


class AlignApplyRequest(BaseModel):
    plate: str
    confirmed: bool = False


@app.post("/api/plate/align/measure")
@long_job("Pomiar wyrównania")
async def measure_plate_alignment(req: AlignMeasureRequest):
    if not STATE.get("manifest") or not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    device = (req.device or "").strip().lower() or None
    if device not in (None, "auto", "gpu", "cpu"):
        raise HTTPException(status_code=400, detail="device: auto | gpu | cpu")

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    beauty_pat = plates.get("beauty")
    if not beauty_pat:
        raise HTTPException(status_code=400, detail="Brak zdefiniowanego plate'u beauty w manifeście")

    aux_pat = plates.get(req.plate)
    if not aux_pat:
        # Check if plate key matches a plate path directly
        aux_pat = plates.get("backlit" if req.aux_plate_type == "backlit" else "cleanplate")
        if not aux_pat:
            raise HTTPException(status_code=400, detail=f"Nie znaleziono ścieżki dla plate'u '{req.plate}' w manifeście")

    first_f = req.first_frame or manifest.get("first", 1)
    last_f = req.last_frame or manifest.get("last", 100)
    w, h = _plate_size()

    work_dir = Path(STATE["work_dir"])
    align_dir = work_dir / "align"
    align_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int, f_num: int):
        JOBS.progress(done, total, f_num)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({
                "type": "align_progress",
                "plate": req.plate,
                "done": done,
                "total": total,
                "frame": f_num
            }),
            loop
        )

    from engine.align import AlignmentPipeline, NukeAlignGroupGenerator

    report = await loop.run_in_executor(
        None,
        lambda: AlignmentPipeline.measure_sequence(
            beauty_pattern=beauty_pat,
            aux_pattern=aux_pat,
            first_frame=first_f,
            last_frame=last_f,
            plate_name=req.plate,
            aux_plate_type=req.aux_plate_type,
            plate_w=w,
            plate_h=h,
            progress_cb=on_progress,
            grid_dir=align_dir / f"grid_{req.plate}",
            step_cb=JOBS.progress,
            cancel_check=JOBS.check,
            device=device,
        )
    )

    # Save report JSON
    json_path = align_dir / f"Align_{req.plate}.json"
    nk_path = align_dir / f"Align_{req.plate}.nk"
    NukeAlignGroupGenerator.generate_script(
        plate_name=req.plate,
        report=report,
        plate_w=w,
        plate_h=h,
        output_nk_path=nk_path,
        output_json_path=json_path
    )

    report.nk_script_path = str(nk_path)
    report.json_path = str(json_path)

    await ws_manager.broadcast({
        "type": "align_complete",
        "plate": req.plate,
        "recommendation": report.recommendation,
        "residual_median": report.residual_median,
        "residual_max": report.residual_max,
        "method": report.method,
        "needs_correction": report.needs_correction
    })

    return report.to_dict()


@app.post("/api/plate/align/export_nk")
def export_align_nk(req: AlignExportNkRequest):
    if not STATE.get("work_dir") or not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "align" / f"Align_{req.plate}.json"
    nk_path = work_dir / "align" / f"Align_{req.plate}.nk"

    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak zapisanego pomiaru wyrównania dla '{req.plate}'. Najpierw uruchom pomiar.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from engine.align import AlignmentReport, NukeAlignGroupGenerator
    report = AlignmentReport.from_dict(data)

    manifest = STATE["manifest"]
    w, h = _plate_size()

    script = NukeAlignGroupGenerator.generate_script(
        plate_name=req.plate,
        report=report,
        plate_w=w,
        plate_h=h,
        output_nk_path=nk_path,
        output_json_path=json_path
    )

    valid, msg = NukeAlignGroupGenerator.validate_nuke_syntax(script)
    if not valid:
        raise HTTPException(status_code=500, detail=f"Błąd składni skryptu Nuke: {msg}")

    return {
        "success": True,
        "nk_path": str(nk_path),
        "json_path": str(json_path),
        "script": script
    }


@app.post("/api/plate/align/apply")
@long_job("Zapis wyrównania")
async def apply_plate_alignment(req: AlignApplyRequest):
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="Operacja wymaga potwierdzenia użytkownika ('confirmed': true)")

    if not STATE.get("work_dir") or not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "align" / f"Align_{req.plate}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Brak pomiaru wyrównania dla '{req.plate}'")

    with open(json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    from engine.align import AlignmentReport, PlateWarper
    report = AlignmentReport.from_dict(report_data)

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    aux_pat = plates.get(req.plate)
    if not aux_pat:
        aux_pat = plates.get("backlit" if report.aux_plate_type == "backlit" else "cleanplate")
        if not aux_pat:
            raise HTTPException(status_code=400, detail=f"Brak ścieżki plate'u '{req.plate}'")

    # Output directory alongside the original plate folder: <plate folder>_align (SPEC 4a pkt 3)
    sample_path = Path(aux_pat.replace("####", "0001").replace("%04d", "0001"))
    src_parent = sample_path.parent
    align_dest_dir = src_parent.parent / f"{src_parent.name}_align"

    grid_dir = work_dir / "align" / f"grid_{req.plate}"

    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int, f_num: int):
        JOBS.progress(done, total, f_num)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({
                "type": "align_apply_progress",
                "plate": req.plate,
                "done": done,
                "total": total,
                "frame": f_num
            }),
            loop
        )

    written = await loop.run_in_executor(
        None,
        lambda: PlateWarper.warp_sequence(
            aux_pattern=aux_pat,
            report=report,
            dest_dir=align_dest_dir,
            grid_dir=grid_dir if grid_dir.exists() else None,
            progress_cb=on_progress
        )
    )

    # Register the aligned plate in project.json (manifest is read-only)
    align_pat = str(align_dest_dir / sample_path.name.replace("0001", "####"))
    STATE["shot_project"].set_plate(f"{req.plate}_align", align_pat)
    _refresh_manifest_view()

    return {
        "success": True,
        "aligned_count": len(written),
        "aligned_dir": str(align_dest_dir),
        "pattern": align_pat
    }


@app.get("/api/plate/align/report")
def get_align_report(plate: str):
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "align" / f"Align_{plate}.json"
    if not json_path.exists():
        return {"exists": False}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from engine.align import stale_report_reason
    return {"exists": True, "report": data, "stale": stale_report_reason(data)}


# -----------------------------------------------------------------------------
# Plate Cleaning Endpoints: Deflicker & Denoise (Stage E1c, SPEC Section 4c)
# -----------------------------------------------------------------------------

class CleanMeasureRequest(BaseModel):
    plate: str = "beauty"
    max_frames: int = 16


class CleanRunRequest(BaseModel):
    plate: str = "beauty"
    target_reduction: float = 0.70
    enable_stage1: bool = True
    enable_stage2: bool = True
    enable_stage3: bool = False
    enable_stage4: bool = True
    enable_denoise: bool = False
    export_mode: str = "nuke_group"  # "nuke_group" | "exr_deflicker" | "exr_deflicker_dn"
    confirmed: bool = False


class CleanExportNkRequest(BaseModel):
    plate: str = "beauty"


@app.post("/api/plate/clean/measure")
@long_job("Pomiar migotania")
async def measure_plate_clean(req: CleanMeasureRequest):
    if not STATE.get("manifest") or not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    pat = plates.get(req.plate)
    if not pat:
        raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{req.plate}' w manifeście")

    first_f = manifest.get("first", 1)
    last_f = manifest.get("last", 100)
    total_frames = last_f - first_f + 1
    sample_count = min(req.max_frames, total_frames) if total_frames > 24 else total_frames

    frames = []
    for f_idx in range(sample_count):
        fn = first_f + f_idx
        f_path = Path(pat.replace("####", f"{fn:04d}"))
        if not f_path.exists():
            break
        img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames.append(img)

    if len(frames) < 2:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać klatek z {pat}")

    loop = asyncio.get_running_loop()
    flicker = await loop.run_in_executor(None, lambda: DeflickerMetrics.measure_flow_aligned(frames))
    variance = await loop.run_in_executor(None, lambda: DeflickerMetrics.measure_temporal_variance(frames))

    return {
        "success": True,
        "flicker": flicker.to_dict(),
        "temporal_variance": round(variance, 4),
        "needs_deflicker": flicker.mean > 4.0 or flicker.p95 > 6.0
    }


@app.post("/api/plate/clean/run")
@long_job("Deflicker")
async def run_plate_clean(req: CleanRunRequest):
    if not STATE.get("manifest") or not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    pat = plates.get(req.plate)
    if not pat:
        raise HTTPException(status_code=400, detail=f"Brak zdefiniowanego plate'u '{req.plate}' w manifeście")

    if req.export_mode in ("exr_deflicker", "exr_deflicker_dn") and not req.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Wymagane jest potwierdzenie zapisu pełnych klatek EXR na dysku (zasada IZOLACJI)."
        )

    first_f = manifest.get("first", 1)
    last_f = manifest.get("last", 100)
    work_dir = Path(STATE["work_dir"])
    plate_w, plate_h = _plate_size()

    # Read frames (RGB only, BGR order; alpha / other layers are copied untouched on export)
    from engine.exr_io import read_rgb
    frames = []
    file_paths = []
    for fn in range(first_f, last_f + 1):
        f_path = ProxyGenerator.resolve_frame_path(pat, fn)
        if not f_path.exists():
            break
        img = read_rgb(f_path) if f_path.suffix.lower() == ".exr" else cv2.imread(str(f_path), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img.astype(np.float32))
            file_paths.append(f_path)

    if len(frames) < 2:
        raise HTTPException(status_code=400, detail=f"Brak wystarczającej liczby klatek ({len(frames)}) dla plate'u {req.plate}")

    loop = asyncio.get_running_loop()

    def on_progress(step_name: str, done: int, total: int):
        JOBS.progress(step_name, done, total)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({
                "type": "clean_progress",
                "step": step_name,
                "done": done,
                "total": total,
                "plate": req.plate
            }),
            loop
        )

    cleaned, gains, report, residuals = await loop.run_in_executor(
        None,
        lambda: PlateCleaner.process_sequence(
            frames=frames,
            plate_name=req.plate,
            target_reduction=req.target_reduction,
            enable_stage1=req.stage1 if hasattr(req, "stage1") else req.enable_stage1,
            enable_stage2=req.enable_stage2,
            enable_stage3=req.enable_stage3,
            enable_stage4=req.enable_stage4,
            enable_denoise=req.enable_denoise,
            progress_cb=on_progress
        )
    )

    clean_dir = work_dir / "deflicker"
    clean_dir.mkdir(parents=True, exist_ok=True)
    gain_dir = clean_dir / f"gain_{req.plate}"
    gain_pattern = PlateCleaner.export_gain_maps(gains, first_frame=first_f, output_dir=gain_dir)
    report.gain_map_pattern = gain_pattern

    # Generate Nuke Group script
    nk_file = clean_dir / f"Deflicker_{req.plate}.nk"
    NukeDeflickerGroupGenerator.generate_script(
        plate_name=req.plate,
        gain_pattern=gain_pattern,
        plate_w=plate_w,
        plate_h=plate_h,
        first_frame=first_f,
        last_frame=last_f,
        report=report,
        output_nk_path=nk_file
    )
    report.nk_group_path = str(nk_file).replace("\\", "/")

    # Export EXRs if selected
    export_dir_str = ""
    if req.export_mode in ("exr_deflicker", "exr_deflicker_dn"):
        # <plate folder>_deflicker / _deflicker_dn next to the original (SPEC 4c.3)
        suffix = "deflicker_dn" if req.export_mode == "exr_deflicker_dn" else "deflicker"
        source_p = Path(pat)
        dest_exr_dir = source_p.parent.parent / f"{source_p.parent.name}_{suffix}"
        dest_exr_dir.mkdir(parents=True, exist_ok=True)

        # Same encoding as the source (channels, half/float, compression, metadata) – SPEC 4c.3
        from engine.exr_io import write_rgb_like
        for idx, cl_frame in enumerate(cleaned):
            src_file = file_paths[idx]
            dest_file = dest_exr_dir / src_file.name
            if src_file.suffix.lower() == ".exr":
                write_rgb_like(dest_file, cl_frame, src_file,
                               extra_attrs={"matteDesk": f"deflicker {'+'.join(report.stages_applied)}".encode()})
            else:
                cv2.imwrite(str(dest_file), cl_frame)

        export_dir_str = str(dest_exr_dir).replace("\\", "/")
        report.export_dir = export_dir_str

        # Update manifest to point beauty or plate to cleaned version if beauty
        new_pat = str(dest_exr_dir / source_p.name).replace("\\", "/")
        STATE["shot_project"].set_plate(f"{req.plate}_clean", new_pat)
        _refresh_manifest_view()

    # Save JSON report
    json_path = clean_dir / f"Deflicker_{req.plate}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    return {
        "success": True,
        "report": report.to_dict(),
        "nk_script_path": report.nk_group_path,
        "export_dir": export_dir_str
    }


@app.get("/api/plate/clean/report")
def get_clean_report(plate: str):
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "deflicker" / f"Deflicker_{plate}.json"
    if not json_path.exists():
        return {"exists": False}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"exists": True, "report": data}


# -----------------------------------------------------------------------------
# 2D Camera Motion Endpoints (Stage E1d, SPEC Section 4d, 12, 16)
# -----------------------------------------------------------------------------

class CameraTrackRequest(BaseModel):
    ref_frame: Optional[int] = None
    eval_w: int = 1024
    anchor_interval: int = 24
    export_nk: bool = True


@app.post("/api/camera/track")
@long_job("Kamera 2D")
async def track_camera_motion(req: CameraTrackRequest):
    if not STATE.get("manifest") or not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    manifest = STATE["manifest"]
    shot_name = manifest.get("shot", "shot")
    fps = float(manifest.get("fps", 24.0))
    first_f = int(manifest.get("first", 1))
    last_f = int(manifest.get("last", 1))
    ref_frame = req.ref_frame if req.ref_frame is not None else first_f

    work_dir = Path(STATE["work_dir"])
    cam_dir = work_dir / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather frames: check proxy frames first for lightning fast processing, fallback to source
    proxy_dir = work_dir / "proxy" / "beauty"
    frames: List[np.ndarray] = []
    frame_nums: List[int] = []

    if proxy_dir.exists() and len(list(proxy_dir.glob("p_*.jpg"))) >= (last_f - first_f + 1):
        for fn in range(first_f, last_f + 1):
            pf = proxy_dir / f"p_{fn:04d}.jpg"
            if pf.exists():
                img = cv2.imread(str(pf))
                if img is not None:
                    frames.append(img)
                    frame_nums.append(fn)
    else:
        # Load from plate pattern
        plates = manifest.get("plates", {})
        pat = plates.get("beauty")
        if not pat:
            raise HTTPException(status_code=400, detail="Brak plate'u beauty w manifeście")
        for fn in range(first_f, last_f + 1):
            src_file = ProxyGenerator.resolve_frame_path(pat, fn)
            if src_file.exists():
                img = cv2.imread(str(src_file), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    frames.append(img)
                    frame_nums.append(fn)

    if not frames:
        raise HTTPException(status_code=404, detail="Nie znaleziono klatek do analizy ruchu kamery")

    loop = asyncio.get_running_loop()

    def on_progress(step_name: str, done: int, total: int):
        JOBS.progress(step_name, done, total)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({
                "type": "camera_progress",
                "step": step_name,
                "done": done,
                "total": total
            }),
            loop
        )

    cam_data: Camera2DData = await loop.run_in_executor(
        None,
        lambda: Camera2DTracker.track_sequence(
            frames=frames,
            frame_numbers=frame_nums,
            ref_frame=ref_frame,
            eval_w=req.eval_w,
            anchor_interval=req.anchor_interval,
            shot_name=shot_name,
            fps=fps,
            progress_cb=on_progress,
            plate_w=_plate_size()[0],
            plate_h=_plate_size()[1],
        )
    )

    json_path = cam_dir / "camera2d.json"
    cam_data.save(json_path)

    nk_path_str = ""
    if req.export_nk:
        nk_file = cam_dir / f"Camera2D_{shot_name}.nk"
        NukeCamera2DGroupGenerator.generate_script(
            shot_name=shot_name,
            camera_data=cam_data,
            output_nk_path=nk_file,
            output_json_path=json_path
        )
        nk_path_str = str(nk_file).replace("\\", "/")

    from engine.outputs import update_outputs
    update_outputs(work_dir, "camera2d", {
        "json_path": str(json_path).replace("\\", "/"),
        "nk_path": nk_path_str,
        "motion_type": cam_data.motion_type,
        "residual_median_px": cam_data.residual_median_px,
        "anchor_drift_px": cam_data.anchor_drift_px
    })

    return {
        "success": True,
        "camera2d": cam_data.to_dict(),
        "nk_script_path": nk_path_str,
        "json_path": str(json_path).replace("\\", "/")
    }


@app.get("/api/camera/report")
def get_camera_report():
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "camera" / "camera2d.json"
    if not json_path.exists():
        return {"exists": False}

    try:
        cam_data = Camera2DData.load(json_path)
        nk_file = work_dir / "camera" / f"Camera2D_{cam_data.shot_name}.nk"
        return {
            "exists": True,
            "camera2d": cam_data.to_dict(),
            "nk_exists": nk_file.exists(),
            "nk_path": str(nk_file).replace("\\", "/") if nk_file.exists() else ""
        }
    except Exception as e:
        logger.error(f"Error loading camera2d.json: {e}")
        return {"exists": False, "error": str(e)}


@app.post("/api/camera/export_nk")
def export_camera_nuke_group():
    if not STATE.get("work_dir") or not STATE.get("manifest"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")

    work_dir = Path(STATE["work_dir"])
    json_path = work_dir / "camera" / "camera2d.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Brak pliku camera2d.json. Najpierw przeprowadź analizę ruchu.")

    cam_data = Camera2DData.load(json_path)
    shot_name = cam_data.shot_name
    nk_file = work_dir / "camera" / f"Camera2D_{shot_name}.nk"

    nk_content = NukeCamera2DGroupGenerator.generate_script(
        shot_name=shot_name,
        camera_data=cam_data,
        output_nk_path=nk_file,
        output_json_path=json_path
    )

    return {
        "success": True,
        "nk_path": str(nk_file).replace("\\", "/"),
        "script": nk_content
    }



# -----------------------------------------------------------------------------
# Depth map Endpoints (Stage E1e, SPEC Section 4e, 10)
# -----------------------------------------------------------------------------

class DepthComputeRequest(BaseModel):
    plate: str = "beauty"
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    workflow_path: Optional[str] = None


def _depth_workflow_path(explicit: Optional[str]) -> Path:
    """The user's DepthCrafter workflow (Comfy UI JSON). Config `depth_workflow`, else Comfy's workflows dir."""
    if explicit:
        return Path(explicit)
    cfg = load_config()
    if cfg.get("depth_workflow"):
        return Path(cfg["depth_workflow"])
    return Path(cfg.get("comfy_dir", "")) / "user" / "default" / "workflows" / "workflow_comfyui_depthcrafter.json"


@app.post("/api/depth/compute")
@long_job("Głębia (DepthCrafter)")
async def compute_depth(req: DepthComputeRequest):
    if not STATE.get("shot_project"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    from engine.depth import run_depth

    proj: ShotProject = STATE["shot_project"]
    manifest = STATE["manifest"]
    plates = manifest.get("plates", {})
    work_dir = Path(STATE["work_dir"])
    wf = _depth_workflow_path(req.workflow_path)
    if not wf.exists():
        raise HTTPException(status_code=400, detail=f"Nie znaleziono workflowu DepthCrafter: {wf}")

    # Input after deflicker (4e.1 pkt 1): the `_deflicker` EXR if exported, otherwise plate x gain maps
    plate_pat = plates.get(f"{req.plate}_clean") or plates.get(req.plate)
    if not plate_pat:
        raise HTTPException(status_code=400, detail=f"Brak plate'u '{req.plate}'")
    gain_dir = work_dir / "deflicker" / f"gain_{req.plate}"
    gain_pat = str(gain_dir / "gain_%04d.exr") if not plates.get(f"{req.plate}_clean") and gain_dir.exists() else None

    first_f = req.first_frame or int(manifest.get("first", 1))
    last_f = req.last_frame or int(manifest.get("last", 1))
    frames = list(range(first_f, last_f + 1))
    w, h = proj.plate_size

    homs = None
    cam_json = work_dir / "camera" / "camera2d.json"
    if cam_json.exists():
        cam = Camera2DData.load(cam_json)
        by_f = {fr.frame: np.array(fr.homography, dtype=np.float64) for fr in cam.frames}
        if all(f in by_f for f in frames):
            sx, sy = w / float(cam.plate_w), h / float(cam.plate_h)
            S = np.diag([sx, sy, 1.0])
            homs = [S @ by_f[f] @ np.linalg.inv(S) for f in frames]

    client = STATE.get("comfy_client") or ComfyClient.from_config()
    loop = asyncio.get_running_loop()

    def on_progress(step: str, done: int, total: int):
        JOBS.progress(step, done, total)
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast({"type": "depth_progress", "step": step, "done": done, "total": total}), loop)

    try:
        rep = await loop.run_in_executor(None, lambda: run_depth(
            client, wf, manifest.get("shot", "shot"), plate_pat, frames, w, h, work_dir, work_dir / "export",
            gain_pattern=gain_pat, homographies=homs, progress_cb=on_progress))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "report": rep.to_dict()}


@app.get("/api/depth/report")
def get_depth_report():
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    p = Path(STATE["work_dir"]) / "depth" / "depth_report.json"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, "report": json.loads(p.read_text(encoding="utf-8"))}


@app.post("/api/depth/export_nk")
def export_depth_nk():
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    from engine.depth import DepthReport, NukeDepthGroupGenerator

    p = Path(STATE["work_dir"]) / "depth" / "depth_report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Najpierw policz głębię")
    rep = DepthReport(**json.loads(p.read_text(encoding="utf-8")))
    nk = Path(STATE["work_dir"]) / "depth" / f"Depth_{rep.shot}.nk"
    text = NukeDepthGroupGenerator.generate_script(rep.shot, rep, nk)
    return {"success": True, "nk_path": str(nk).replace("\\", "/"), "script": text}


# -----------------------------------------------------------------------------
# Objects, instance choice and crop engine (Stages E2/E3, SPEC 5.2, 6, 10)
# -----------------------------------------------------------------------------

class ObjectCreateRequest(BaseModel):
    name: str
    prompts: List[str] = []
    depth_order: Optional[int] = None
    flags: List[str] = []


class ObjectDetectRequest(BaseModel):
    frame: int
    prompt: Optional[str] = None
    positive: List[List[float]] = []   # plate px
    negative: List[List[float]] = []


class ObjectChooseRequest(BaseModel):
    frame: int
    candidate: int = 0


class CropOverrideRequest(BaseModel):
    crop: List[int]   # [x, y, side, side] plate px


def _objects():
    from engine.objects import ObjectStore
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    return ObjectStore(STATE["work_dir"])


def _proxy_scale() -> float:
    w, h = _plate_size()
    return min(1.0, 1920.0 / max(w, h))


def _load_proxy_display(frame: int, plate: str = "beauty") -> np.ndarray:
    """Display proxy (BGR float 0..1) – rendered on demand like /api/frame."""
    proxy_dir = Path(STATE["work_dir"]) / "proxy" / plate
    pf = proxy_dir / f"p_{frame:04d}.jpg"
    if not pf.exists():
        pattern = STATE["manifest"].get("plates", {}).get(plate, "")
        src = ProxyGenerator.resolve_frame_path(pattern, frame) if pattern else None
        if not src or not src.exists():
            raise HTTPException(status_code=404, detail=f"Brak klatki {frame} plate'u {plate}")
        ProxyGenerator.render_proxy_file(source_frame_path=src, dest_proxy_path=pf,
                                         colorspace=STATE["manifest"].get("colorspace", "ACES2065-1"))
    img = cv2.imread(str(pf), cv2.IMREAD_COLOR)
    w, h = _plate_size()
    s = _proxy_scale()
    want = (max(1, round(w * s)), max(1, round(h * s)))
    if (img.shape[1], img.shape[0]) != want:
        img = cv2.resize(img, want, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _sam3_runner():
    from engine.sam3 import Sam3Runner
    client = STATE.get("comfy_client") or ComfyClient.from_config()
    return Sam3Runner(client, Path(STATE["work_dir"]) / "comfy_io", io_mode=_comfy_io_mode())


def _comfy_io_mode() -> str:
    """How images go to Comfy (decision 2026-10-13): 8-bit sources -> PNG + VHS_LoadImagesPath, EXR -> as before."""
    from engine.comfy_io import io_mode_for
    return io_mode_for(STATE.get("manifest"))


def _plate_display(lin: np.ndarray) -> np.ndarray:
    """Scene-linear -> display for SAM3 / MatAnyone inputs (PlateToDisplay formula; 4b replaces it in E5b)."""
    from engine.pipeline import default_display
    return default_display(lin)


_plate_display.elementwise = True   # Z30: per-pixel formula (crop-first reading, half LUT)
_plate_display.lut_white = 1.0


class PreviewFrameRequest(BaseModel):
    frame: int
    prompt: Optional[str] = None
    positive: List[List[float]] = []
    negative: List[List[float]] = []


def _sam3_preview_mask(name: str, frame: int, prompt: Optional[str], positive: List[List[float]],
                       negative: List[List[float]], use_crops: bool = True) -> Dict[str, Any]:
    """SAM3_Detect on the crop of one frame (SPEC 6 pkt 2, < 3 s): mask saved as preview/p_####.png (plate px).
    Prompt (the request's, else the object's) and points together (user decision 2026-10-15): prompt = base, positive
    points add, negative points subtract what they point at (engine.sam3.combine_prompt_points)."""
    import time as _time
    from engine.crop import CropSegment, frame_segments, load_crops, paste_crop, square_crop
    from engine.pipeline import PlateCropLoader
    t0 = _time.time()
    store = _objects()
    obj = store.get(name)
    w, h = _plate_size()
    crops_p = store.dir(name) / "crops.json"
    seg = None
    # use_crops=False („▶ Przelicz klatkę” in stage Wykryj, user decision 2026-10-15): the object's crops come from
    # „▶▶ Przelicz całość” – while the selection is being decided they must not restrict SAM3
    if use_crops and crops_p.exists():
        segs = frame_segments(load_crops(crops_p), frame)
        seg = segs[0] if segs else None
        pts_all = list(positive or []) + list(negative or [])
        if seg is not None and pts_all:
            cx0, cy0, side0 = seg.crop[0], seg.crop[1], seg.crop[2]
            if any(not (cx0 <= x < cx0 + side0 and cy0 <= y < cy0 + side0) for x, y in pts_all):
                # a point outside the object's crop (bug 2026-10-15: a click on the wall gave the hair – SAM3 got the
                # point outside its image): one square around the crop AND every point
                xs = [x for x, _ in pts_all] + [cx0, cx0 + side0]
                ys = [y for _, y in pts_all] + [cy0, cy0 + side0]
                side = max(max(xs) - min(xs), max(ys) - min(ys), 64)
                crop, pad = square_crop((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side, w, h)
                seg = CropSegment("preview", frame, frame, 0, 0, crop, pad, round(1008 / crop[2], 4))
    if seg is None:  # no crops (yet): the whole frame for a prompt, else a square around the points / the start key
        pts = list(positive or []) + list(negative or [])
        keys = store.keys(name)
        eff_prompt = prompt if prompt is not None else (obj.get("detect", {}).get("prompts") or [None])[0]
        if (eff_prompt or "").strip():
            bb = (0, 0, w, h)
            pts = []
        elif pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bb = (min(xs), min(ys), max(xs), max(ys))
        elif keys:
            kf = min(keys, key=lambda k: abs(k - frame))
            ys, xs = np.where(store.key_mask(name, kf))
            bb = (xs.min(), ys.min(), xs.max(), ys.max())
        else:
            bb = (0, 0, w, h)
        side = max(bb[2] - bb[0], bb[3] - bb[1], 64)
        if pts:   # points alone say little about the object's extent: give SAM3 context around them
            side = max(side * 2.5, min(w, h) * 0.35)
        crop, pad = square_crop((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, side, w, h)
        seg = CropSegment("preview", frame, frame, 0, 0, crop, pad, round(1008 / crop[2], 4))
    pattern = STATE["manifest"]["plates"].get("beauty")
    img = PlateCropLoader(pattern, _plate_display).crop(frame, seg)
    x0, y0 = seg.crop[0], seg.crop[1]
    shift = lambda pts: [(x - x0, y - y0) for x, y in pts] or None
    from engine.sam3 import combine_prompt_points
    if prompt is None:   # None = the object's prompt; "" = explicitly no prompt (field cleared in the palette)
        prompt = (obj.get("detect", {}).get("prompts") or [None])[0]
    prompt = (prompt or "").strip() or None
    runner = _sam3_runner()
    info: Dict[str, Any] = {"mode": "points" if (positive or negative) else "prompt"}
    try:
        if prompt and (positive or negative):
            info["mode"] = "prompt+points"
            first = lambda ms: ms[0] if ms else None
            base = first(runner.detect(img, prompt=prompt, individual=False))
            plus = first(runner.detect(img, positive=shift(positive), negative=shift(negative or []),
                                       individual=False)) if positive else None
            minus = first(runner.detect(img, positive=shift(negative), negative=shift(positive or []),
                                        individual=False)) if negative else None
            if base is None and plus is None and minus is None:
                base = np.zeros((seg.side, seg.side), np.float32)
            m, counts = combine_prompt_points(base, plus, minus)
            info.update(counts)
            masks = [m.astype(np.float32)]
        else:
            masks = runner.detect(img, prompt=prompt, positive=shift(positive or []),
                                  negative=shift(negative or []), individual=False)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    full = paste_crop(masks[0] if masks else np.zeros((seg.side, seg.side), np.float32), seg, w, h) > 0.5
    out = store.dir(name) / "preview" / f"p_{frame:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), full.astype(np.uint8) * 255)
    return {"frame": frame, "mask": full, "mask_path": str(out), "crop": seg.crop,
            "seconds": round(_time.time() - t0, 3), "empty": not full.any(), "prompt": prompt, **info}


@app.post("/api/object/{name}/preview_frame")
@long_job("Podgląd klatki: {name}", quiet=True)
def preview_frame(name: str, req: PreviewFrameRequest):
    """SAM3_Detect on the crop of one frame (SPEC 6 pkt 2, < 3 s): quick loop before computing the shot."""
    r = _sam3_preview_mask(name, req.frame, req.prompt, req.positive, req.negative)
    r.pop("mask")
    return {"success": True, **r}


class PointsRequest(BaseModel):
    positive: List[List[float]] = []   # plate px
    negative: List[List[float]] = []
    prompt: Optional[str] = None


@app.post("/api/object/{name}/points/{frame}")
@object_edit
@long_job("Punkty SAM3: {name}", quiet=True)
def set_points(name: str, frame: int, req: PointsRequest):
    """Stage 1 (Wykryj): the points of a frame ARE its key. Any change -> SAM3_Detect from the points -> key
    `kind: points` saved at that frame (author user; undo/redo in history.json). No points left -> key deleted
    (only a points key; a brush/polygon key stays and must be deleted explicitly)."""
    from engine.keys import KeyHistory
    w, h = _plate_size()
    bad = [pt for pt in list(req.positive) + list(req.negative)
           if len(pt) < 2 or not (0 <= float(pt[0]) < w and 0 <= float(pt[1]) < h)]
    if bad:   # Z11: a point outside the plate (SH010 F20: [2189, -422]) gives SAM3 a wrong mask -> empty core
        raise HTTPException(status_code=400, detail=f"Punkt poza kadrem {bad[0]} (kadr {w}×{h} px) – kliknij na obrazie")
    store = _objects()
    keys = store.keys(name)
    existing = keys.get(frame)
    if req.prompt is not None:   # the palette sends its field: it is also the object's prompt (no race with the blur save)
        pr = req.prompt.strip()
        if pr != ((store.get(name).get("detect", {}).get("prompts") or [""])[0] or ""):
            store.update(name, detect={**store.get(name).get("detect", {}), "prompts": [pr] if pr else []})
    prompt_now = (req.prompt.strip() if req.prompt is not None
                  else ((store.get(name).get("detect", {}).get("prompts") or [""])[0] or "")) or None
    if not req.positive and not req.negative and not prompt_now:
        if existing and existing.get("kind") == "points":
            KeyHistory(store, name).delete_key(frame)
            store.update(name, tool={"final_valid": False, "stale_from": frame})
            pp = store.dir(name) / "preview" / f"p_{frame:04d}.png"
            if pp.exists():
                pp.unlink()
            return {"success": True, "deleted": True, "frame": frame}
        return {"success": True, "deleted": False, "frame": frame, "kept": existing.get("kind") if existing else None}
    r = _sam3_preview_mask(name, frame, prompt_now or "", req.positive, req.negative, use_crops=False)
    if r["empty"]:
        # no mask for these points: a stale points-key would no longer match its points -> remove it
        if existing and existing.get("kind") == "points":
            KeyHistory(store, name).delete_key(frame)
            store.update(name, tool={"final_valid": False, "stale_from": frame})
        return {"success": True, "frame": frame, "empty": True, "deleted": bool(existing and existing.get("kind") == "points"),
                "seconds": r["seconds"], "crop": r["crop"], "detail": ("SAM3 nie znalazł obiektu z promptu „" + prompt_now + "” – zmień prompt albo dodaj punkt +"
                                              if r["mode"] == "prompt" else
                                              "SAM3 nie znalazł obiektu przy tych punktach – dodaj punkt + wyraźnie na obiekcie")}
    if existing and existing.get("kind") not in (None, "points", "mask") and existing.get("author") == "user":
        # a brush / polygon correction exists in this frame: keep it, store the points only
        store.update(name, tool={"final_valid": False, "stale_from": frame})
        return {"success": True, "frame": frame, "kept": existing.get("kind"), "seconds": r["seconds"],
                "detail": f"klatka {frame} ma klucz „{existing.get('kind')}” – punkty nie nadpisują go"}
    key = KeyHistory(store, name).save_key(frame, r["mask"], author="user", kind="points",
                                           positive=req.positive, negative=req.negative,
                                           prompt=r["prompt"] if r["mode"] in ("prompt+points", "prompt") else None)
    if not store.get(name).get("tool", {}).get("start_frame"):
        store.update(name, tool={"start_frame": frame})
    store.update(name, tool={"final_valid": False, "stale_from": frame, "status": "klucze z punktów"})
    return {"success": True, "frame": frame, "key": {"frame": frame, "author": "user", "kind": "points"},
            "seconds": r["seconds"], "crop": r["crop"], "mode": r["mode"], "prompt": r["prompt"],
            "prompt_px": r.get("prompt_px"), "plus_px": r.get("plus_px"), "minus_px": r.get("minus_px")}


def _ltx_workflow() -> Optional[Dict[str, Any]]:
    cfg = load_config()
    p = cfg.get("ltx_alpha_workflow")
    if p and Path(p).exists():
        return json.loads(Path(p).read_text(encoding="utf-8"))
    return None


def _backlit_pattern() -> Optional[str]:
    plates = STATE["manifest"].get("plates", {})
    return plates.get("backlit_align") or plates.get("backlit") or None


def _mplate_global_enabled() -> bool:
    """defaults.json mplate.enabled: global kill-switch of the plate-to-mask denoise (false = off everywhere)."""
    try:
        d = json.loads((REPO_DIR / "defaults.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return bool(d.get("mplate", {}).get("enabled", True))


def _mask_plate_denoise_info(proj: Any = None, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Project switch of the plate-to-mask denoise (SPEC 4b, decision 2026-10-13): project.json mask_plate_denoise
    auto (key absent) / on / off -> effective value. Shared by the core and the final pass."""
    from engine.mplate import DENOISE_KEY, resolve_mask_plate_denoise
    proj = proj if proj is not None else STATE.get("shot_project")
    setting = proj.data.get(DENOISE_KEY) if proj else None
    return resolve_mask_plate_denoise(setting, manifest if manifest is not None else STATE.get("manifest"),
                                      _mplate_global_enabled())


def _mask_plate_denoise() -> bool:
    return _mask_plate_denoise_info()["effective"]


def _pipeline():
    from engine.pipeline import ObjectPipeline
    w, h = _plate_size()
    defaults = json.loads((REPO_DIR / "defaults.json").read_text(encoding="utf-8"))
    m = STATE["manifest"]
    gain_dir = Path(STATE["work_dir"]) / "deflicker" / "gain_beauty"
    plate = m["plates"].get("beauty_clean") or m["plates"]["beauty"]   # exported _deflicker EXR, if chosen (4b pkt 1)
    gain = str(gain_dir / "gain_%04d.exr") if (not m["plates"].get("beauty_clean") and gain_dir.exists()) else None
    return ObjectPipeline(Path(STATE["work_dir"]), plate, w, h,
                          STATE.get("comfy_client") or ComfyClient.from_config(), display=_plate_display,
                          backlit_pattern=_backlit_pattern(), backlit_params=defaults.get("backlit", {}),
                          ltx_workflow=_ltx_workflow(), use_mplate=_mask_plate_denoise(),
                          gain_pattern=gain, frames=_shot_frames(), shutter_deg=float(m.get("shutter_deg", 180)),
                          cleanplate_pattern=m["plates"].get("cleanplate_align") or m["plates"].get("cleanplate") or None,
                          io_mode=_comfy_io_mode())


EDGE_MODES = ("auto", "backlit", "matanyone", "core", "ltx")


@app.get("/api/object/{name}/edge_modes")
def edge_modes(name: str):
    """'Krawędź' list (SPEC 5.5): Backlit only when a backlit plate exists. LTX is no longer an edge mode (user decision
    2026-10-16: its own stage „LTX Alpha-Gen” before the export, POST /api/object/{name}/ltx)."""
    obj = _objects().get(name)
    has_bl = bool(_backlit_pattern())
    items = [{"mode": "auto", "label": f"Auto (→ {obj.get('edge_mode_auto') or '?'})", "enabled": True}]
    if has_bl:
        items.append({"mode": "backlit", "label": "Backlit", "enabled": True})
    items += [{"mode": "matanyone", "label": "MatAnyone", "enabled": True},
              {"mode": "core", "label": "Ostra (rdzeń)", "enabled": True}]
    return {"current": obj.get("edge_mode", "auto"), "auto": obj.get("edge_mode_auto"),
            "reason": obj["tool"].get("edge_auto_reason", ""), "use_backlit": obj.get("use_backlit", True), "modes": items}


class EdgeModeRequest(BaseModel):
    mode: str
    use_backlit: Optional[bool] = None


@app.put("/api/object/{name}/edge_mode")
@object_edit
def set_edge_mode(name: str, req: EdgeModeRequest):
    if req.mode not in EDGE_MODES:
        raise HTTPException(status_code=400, detail=f"nieznany tryb {req.mode}")
    if req.mode == "ltx":
        raise HTTPException(status_code=400, detail="LTX to teraz osobny etap „LTX Alpha-Gen” przed eksportem, nie tryb krawędzi")
    if req.mode == "backlit" and not _backlit_pattern():
        raise HTTPException(status_code=400, detail="Brak plate'u backlit w projekcie")
    fields = {"edge_mode": req.mode}
    if req.use_backlit is not None:
        fields["use_backlit"] = req.use_backlit
    obj = _objects().update(name, **fields, tool={"final_valid": False})
    return {"success": True, "object": obj}


def _shot_frames() -> List[int]:
    m = STATE["manifest"]
    return list(range(int(m.get("first", 1)), int(m.get("last", 1)) + 1))


def _shot_range() -> Tuple[int, int]:
    fr = _shot_frames()
    return (fr[0], fr[-1])


def _work_area() -> Optional[List[int]]:
    """Work area of the shot (project.json -> work_area, engine/workarea.py); None = the whole shot."""
    from engine.workarea import clamp_range
    proj = STATE.get("shot_project")
    try:
        return clamp_range((proj.data.get("work_area") if proj else None), _shot_range())
    except ValueError:
        return None


def _work_frames() -> List[int]:
    """Frames of the work area (the whole shot when none is set)."""
    wa = _work_area()
    return list(range(wa[0], wa[1] + 1)) if wa else _shot_frames()


def _require_stage(name: str, stage: str) -> None:
    """Stages in order (engine/stages.py): refuse an action whose previous stage is not done (on the work area)."""
    from engine.stage_models import model_gate
    from engine.stages import require
    _require_plates_ok()
    g = model_gate(load_config()).get(stage)
    if g:   # user rule 2026-10-16: a stage without its model is unavailable
        raise HTTPException(status_code=409, detail=g["why"])
    try:
        require(_objects(), name, Path(STATE["work_dir"]), stage, mplate=_mask_plate_denoise(),
                frame_range=_shot_range(), work_range=_work_area())
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/project/work_area")
def get_work_area():
    """Work area (AE / Nuke style, user request 2026-10-15): `range` = [first, last] or None (whole shot), `shot`,
    `effective` = the frames „▶▶ Przelicz całość”, „Przelicz finalnie” and the export "work area" compute."""
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    wa = _work_area()
    sh = list(_shot_range())
    return {"range": wa, "shot": sh, "effective": wa or sh, "is_whole": wa is None}


class WorkAreaRequest(BaseModel):
    range: Optional[List[int]] = None     # [first, last]; None / the whole shot = no work area


@app.put("/api/project/work_area")
def set_work_area(req: WorkAreaRequest):
    """Stored in project.json -> work_area (the shot manifest is never written). A running job keeps the frames it
    started with."""
    from engine.workarea import clamp_range
    proj = STATE.get("shot_project")
    if not proj:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    if req.range is not None and len(req.range) != 2:
        raise HTTPException(status_code=400, detail="Zakres = [pierwsza, ostatnia] klatka")
    try:
        wa = clamp_range(req.range, _shot_range())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if wa:
        proj.data["work_area"] = wa
    else:
        proj.data.pop("work_area", None)
    proj.save()
    return get_work_area()


@app.get("/api/object/{name}/life")
def get_object_life(name: str):
    """Object life range (zakres życia, user request 2026-10-15): `life` = visible intervals [[a, b], ...] or None (the
    whole shot), `shot`, `hints` = suggestions from the last core (object leaves / enters at the frame border),
    `hidden_keys` = keys outside the life (ignored by the computation)."""
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    from engine.workarea import is_visible, object_life
    o = _objects()
    try:
        obj = o.get(name)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    hints = []
    cj = o.dir(name) / "core" / "core.json"
    if cj.exists():
        try:
            hints = json.loads(cj.read_text(encoding="utf-8")).get("life_hints") or []
        except (OSError, ValueError):
            hints = []
    life = object_life(obj)
    # only the hints that would still change the life range (some of their empty frames are visible)
    hints = [h for h in hints if any(is_visible(f, life) for f in range(int(h["empty"][0]), int(h["empty"][1]) + 1))]
    return {"life": life, "shot": list(_shot_range()), "is_whole": life is None, "hints": hints,
            "hidden_keys": sorted(int(k) for k in o.keys(name) if not is_visible(int(k), life))}


class LifeRequest(BaseModel):
    life: Optional[List[List[int]]] = None     # visible intervals; None / the whole shot = no life range


@app.put("/api/object/{name}/life")
@object_edit
def set_object_life(name: str, req: LifeRequest):
    """Stored in objects/<name>/object.json -> life (the shot manifest is never written). Outside the intervals the
    matte is empty and nothing is computed; the core goes stale for the frames whose visibility changed."""
    from engine.workarea import normalize_life
    o = _objects()
    try:
        obj = o.get(name)
        life = normalize_life(req.life, _shot_range())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if life:
        obj["life"] = life
    else:
        obj.pop("life", None)
    o._write(o.dir(name) / "object.json", obj)
    return get_object_life(name)


def _core_frames(name: str) -> List[int]:
    """Frames of a core run: the work area extended to the nearest key on each side (whole intervals between keys, the
    same cache as on the whole shot); the whole shot without a work area.
    Object life range (engine/workarea.py): every visible piece of the work area needs a key and is extended only
    inside its life interval; the frames of the work area outside the life are passed too (empty core, not computed)."""
    from engine.workarea import expand_to_keys, fmt_intervals, is_visible, life_pieces, object_life
    wa = _work_area()
    life = object_life(_objects().get(name))
    if life is not None:
        rng = wa or list(_shot_range())
        keys = [int(k) for k in _objects().keys(name) if is_visible(int(k), life)]
        pieces = life_pieces(rng, life)
        where = "zakresie pracy" if wa else "ujęciu"
        if not pieces:
            raise HTTPException(status_code=409, detail={
                "code": "not_visible", "range": rng,
                "detail": f"Obiekt „{name}” jest poza swoim zakresem życia w całym {where} F{rng[0]}–{rng[1]} – nic do "
                          f"liczenia (zmień zakres życia albo zakres pracy)"})
        nokey = [p for p in pieces if not any(p[0] <= k <= p[1] for k in keys)]
        if nokey:
            raise HTTPException(status_code=409, detail={
                "code": "no_key_in_range", "range": rng, "pieces": nokey,
                "detail": f"Każdy widoczny kawałek obiektu w {where} musi mieć klatkę kluczową – brak klucza w: "
                          f"{fmt_intervals(nokey)} (dodaj klucz albo zmień zakres życia)"})
        out = set(range(int(rng[0]), int(rng[1]) + 1))
        for pc in pieces:
            iv = next(L for L in life if L[0] <= pc[0] <= L[1])     # the life interval of this piece
            a, b = expand_to_keys(pc, [k for k in keys if iv[0] <= k <= iv[1]], iv)
            out |= set(range(a, b + 1))
        return sorted(out)
    if not wa:
        return _shot_frames()
    keys = [int(k) for k in _objects().keys(name)]
    if not any(wa[0] <= k <= wa[1] for k in keys):
        # user decision 2026-10-15: SAM3 needs at least one key inside the work area (no silent propagation from outside)
        raise HTTPException(status_code=409, detail={
            "code": "no_key_in_range", "range": wa,
            "detail": f"W zakresie pracy F{wa[0]}–{wa[1]} musi być co najmniej jedna klatka kluczowa – dodaj klucz "
                      f"(punkty + „▶ Przelicz klatkę”) albo zmień zakres"})
    a, b = expand_to_keys(wa, keys, _shot_range())
    return list(range(a, b + 1))


def _require_core_on(name: str, frames: List[int], what: str = "Rdzeń") -> None:
    """409 (code range_missing + missing intervals) when the object's core does not cover `frames`."""
    from engine.workarea import core_coverage, missing_detail, need_missing, object_life
    cov = core_coverage(_objects().dir(name))
    if cov is None:
        return               # no core at all: the stage check says „najpierw …”
    miss = need_missing((frames[0], frames[-1]), cov, object_life(_objects().get(name)))
    if miss:
        lbl = "całym ujęciu" if frames == _shot_frames() else f"zakresie pracy F{frames[0]}–{frames[-1]}"
        raise HTTPException(status_code=409, detail=missing_detail(what, lbl, {name: miss}))


@app.get("/api/object/{name}/stages")
def get_object_stages(name: str):
    """Stage strip of the objects panel: done / stale / todo / locked + the next action."""
    from engine.stages import object_stages
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    fr = _shot_frames()
    info = object_stages(_objects(), name, Path(STATE["work_dir"]), (fr[0], fr[-1]) if fr else None,
                         mplate=_mask_plate_denoise(), work_range=_work_area())
    # user rule 2026-10-16: a stage whose model is not downloaded / installed is unavailable (locked, with the reason)
    from engine.stage_models import apply_gate, model_gate
    from engine.stages import STEPS
    apply_gate(info, model_gate(load_config()), {sid: members for _n, sid, _l, members in STEPS})
    block = _plate_gate()["block"]
    if block:   # Z26: plate QA undecided -> nothing of the object can run (results on disk stay visible)
        for x in info["stages"] + info["steps"]:
            if x["state"] != "done":
                x["state"], x["why"] = "locked", block
        info["steps"][0].update(state="locked", why=block)
        info["plate_block"] = block
    return info


def _ws_progress(kind: str, name: str):
    loop = asyncio.get_running_loop()

    def cb(step: str, done: int, total: int):
        JOBS.progress(step, done, total)
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(
            {"type": kind, "object": name, "step": step, "done": done, "total": total}), loop)
    return cb


def _crops_needed(name: str) -> Optional[str]:
    """Why the crops must be (re)computed before the core, None = the current ones hold every user key."""
    from engine.crop import crops_cover_keys, load_crops
    store = _objects()
    cp = store.dir(name) / "crops.json"
    if not cp.exists():
        return "brak kadrów"
    boxes = {}
    for f, k in store.keys(name).items():
        if k.get("author") != "user":
            continue
        ys, xs = np.where(store.key_mask(name, f))
        if len(xs):
            boxes[f] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return crops_cover_keys(load_crops(cp), boxes)


@app.post("/api/object/{name}/compute_all")
@long_job("Przelicz całość: {name}")
async def compute_all(name: str):
    """„▶▶ Przelicz całość” (user decision 2026-10-15): crops only when missing or when a user key no longer fits its
    crop, then the core on the whole shot (unchanged intervals from the cache) – the mask in every frame. Repeated
    after adding / removing keys it recomputes only the changed intervals."""
    _require_stage(name, "crops")
    frames = _core_frames(name)       # work area extended to the nearest keys (crops always cover the whole shot)
    loop = asyncio.get_running_loop()
    reason = await loop.run_in_executor(None, lambda: _crops_needed(name))
    crops = None
    if reason:
        JOBS.progress(f"Kadrowanie ({reason})")
        crops = await loop.run_in_executor(None, lambda: _compute_crops(name))
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        res = await loop.run_in_executor(None, lambda: pipe.run_core(name, frames, cb))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "frames": [frames[0], frames[-1]], "work_area": _work_area(),
            "whole_shot": frames == _shot_frames(), "crops_recomputed": bool(reason), "crops_reason": reason,
            "crop_segments": len(crops["segments"]) if crops else None, "crop_flags": (crops or {}).get("flags") or {},
            "life_hints": getattr(res, "life_hints", []),
            "uncertain": res.uncertain, "suggested_keys": res.suggested_keys,
            "recomputed": res.recomputed_intervals, "cached": res.cached_intervals}


@app.post("/api/object/{name}/run")
@long_job("Licz ujęcie: {name}")
async def run_object_core(name: str):
    """Core SAM3 on the whole shot from the keys (7.2); intervals with unchanged inputs come from cache."""
    _require_stage(name, "core")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    frames = _core_frames(name)
    try:
        res = await asyncio.get_running_loop().run_in_executor(None, lambda: pipe.run_core(name, frames, cb))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "frames": [frames[0], frames[-1]], "life_hints": getattr(res, "life_hints", []),
            "uncertain": res.uncertain, "suggested_keys": res.suggested_keys,
            "recomputed": res.recomputed_intervals, "cached": res.cached_intervals}


class FinalizeRequest(BaseModel):
    edge_mode: Optional[str] = None
    stage: Optional[str] = None      # "matanyone" = stage 3: MatAnyone2 on the mask 1+2, matte zones skipped (base version)


@app.post("/api/object/{name}/core_fix")
@long_job("Zastosuj poprawki: {name}")
async def core_fix_object(name: str):
    """Stage 2 „▶▶ Zastosuj poprawki” (2026-10-15): the mask 1+2 (SAM3 core + brush corrections propagated by
    CoTracker3) on every frame the core covers -> core_fix/ (view=core_fix, export source "core")."""
    from engine.workarea import core_coverage
    _require_stage(name, "cotracker")
    cov = core_coverage(_objects().dir(name)) or []
    frames = [f for f in _shot_frames() if any(a <= f <= b for a, b in cov)]
    if not frames:
        raise HTTPException(status_code=409, detail="Brak rdzenia – „▶▶ Przelicz całość” (etap 1)")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        info = await asyncio.get_running_loop().run_in_executor(None, lambda: pipe.build_core_fix(name, frames, cb))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "frames": [frames[0], frames[-1]], **info}


@app.post("/api/object/{name}/finalize")
@long_job("Przelicz finalnie: {name}")
async def finalize_object(name: str, req: FinalizeRequest = FinalizeRequest()):
    """'Przelicz finalnie' (7.3a) -> runs/v### on the work area (whole shot without one); the core must cover it.
    `stage="matanyone"`: stage 3 of the 2026-10-15 layout (edge MatAnyone2, only the brush corrections, no matte zones)."""
    if req.stage not in (None, "matanyone"):
        raise HTTPException(status_code=400, detail="stage: matanyone (albo brak = „Przelicz finalnie” z etapu 5)")
    asked = name
    if req.stage == "matanyone":   # a part of a group: the group is computed as a whole on its parent (2026-10-15)
        name = _objects().get(name).get("part_of") or name
    frames = _work_frames()
    _require_core_on(name, frames)
    _require_stage(name, req.stage or "final")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    from engine.zones import MissingFront
    try:
        _, rep, run_dir = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.finalize(name, frames, req.edge_mode, cb, stage=req.stage))
    except MissingFront as e:
        raise _missing_front_409(e)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    from engine.export import mark_dependents
    deps = mark_dependents(_objects(), name)
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {"success": True, "run": run_dir.name, "report": rep.to_dict(), "dependents_to_recompute": deps,
            "frames": [frames[0], frames[-1]], "whole_shot": frames == _shot_frames(), "object": name, "asked": asked,
            "group": rj.get("group"), "group_outside_crop_px": rj.get("group_outside_crop_px")}


class LtxSettingsRequest(BaseModel):
    guide: Optional[str] = None          # plain | isolate | steer
    max_side: Optional[int] = None
    max_frames: Optional[int] = None
    overlap: Optional[int] = None
    seed: Optional[int] = None
    region_px: Optional[float] = None
    region_soft_px: Optional[float] = None
    combine: Optional[str] = None        # region | max_core | raw
    core_erode_px: Optional[int] = None
    mask_source: Optional[str] = None    # core | matanyone | vitmatte (2026-10-16)


def _ltx_status() -> Dict[str, Any]:
    """Models of stage LTX (present / missing with size and repo) + the Comfy nodes it needs."""
    from engine.ltx_alpha import missing_nodes, models_status
    cfg = load_config()
    models = models_status(cfg.get("models_dir"))
    try:
        nodes = missing_nodes((STATE.get("comfy_client") or ComfyClient.from_config()).get_object_info())
    except Exception:
        nodes = None    # Comfy not running: unknown
    miss = [m for m in models if not m["present"]]
    return {"models": models, "missing_models": [m["file"] for m in miss],
            "missing_gb": round(sum(m["gb"] for m in miss), 1), "missing_nodes": nodes,
            "ready": not miss and nodes == []}


@app.get("/api/object/{name}/ltx")
def ltx_info(name: str):
    """Stage „LTX Alpha-Gen”: the object's settings (defaults.json "ltx" <- object `ltx`), the stage result, models."""
    from engine.ltx_alpha import COMBINES, GUIDES, object_settings
    from engine.stages import stage_run
    store = _objects()
    obj = store.get(name)
    run, ok, why = stage_run(store, name, "ltx")
    return {"settings": object_settings(obj), "guides": list(GUIDES), "combines": list(COMBINES),
            "run": run, "valid": ok, "why": why, **_ltx_status()}


@app.put("/api/object/{name}/ltx")
@object_edit
def ltx_set(name: str, req: LtxSettingsRequest):
    from engine.ltx_alpha import ltx_settings, object_settings
    store = _objects()
    cur = dict(store.get(name).get("ltx") or {})
    cur.update({k: v for k, v in req.model_dump().items() if v is not None})
    clean = ltx_settings({"ltx": cur})
    store.update(name, ltx={k: clean[k] for k in cur})
    return {"success": True, "settings": object_settings(store.get(name))}


def _ltx_preview_settings(name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """The object's LTX settings with the palette values not saved yet (query parameters)."""
    from engine.ltx_alpha import ltx_settings
    cur = dict(_objects().get(name).get("ltx") or {})
    cur.update({k: v for k, v in overrides.items() if v is not None})
    from engine.ltx_alpha import object_settings
    base = object_settings(_objects().get(name))
    return ltx_settings({"ltx": {**base, **cur}})


@app.get("/api/object/{name}/ltx_preview/{frame}")
def ltx_preview(name: str, frame: int, what: str = "band", region_px: Optional[float] = None,
                region_soft_px: Optional[float] = None, core_erode_px: Optional[int] = None,
                combine: Optional[str] = None, mask_source: Optional[str] = None):
    """LTX palette (2026-10-16), live: what=band – 128 = where the LTX matte is kept, 255 = forced 1 (proxy PNG);
    what=alpha – the current LTX version's raw matte re-limited with these values (404 without an LTX version)."""
    name = _objects().get(name).get("part_of") or name
    s = _ltx_preview_settings(name, {"region_px": region_px, "region_soft_px": region_soft_px,
                                     "core_erode_px": core_erode_px, "combine": combine, "mask_source": mask_source})
    try:
        band, alpha = _pipeline().ltx_frame_preview(name, frame, s)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    if what == "alpha" and alpha is None:
        raise HTTPException(status_code=404, detail="Brak wyniku LTX – podgląd wyniku po „▶ Policz LTX”")
    m = band if what != "alpha" else (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    sc = _proxy_scale()
    w, h = _plate_size()
    small = cv2.resize(m, (max(1, round(w * sc)), max(1, round(h * sc))),
                       interpolation=cv2.INTER_NEAREST if what != "alpha" else cv2.INTER_AREA)
    return _png_response(small)


@app.post("/api/object/{name}/ltx/refinish")
@long_job("LTX – suwaki na wynik: {name}")
async def ltx_refinish(name: str):
    """„Zastosuj suwaki na wynik” (2026-10-16): the current LTX version's raw mattes limited again with the saved
    region / mask settings -> a new version (no model run)."""
    name = _objects().get(name).get("part_of") or name
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        run = await asyncio.get_running_loop().run_in_executor(None, lambda: pipe.ltx_refinish(name, _work_frames(), progress_cb=cb))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "run": run, "object": name}


@app.post("/api/object/{name}/ltx")
@long_job("LTX Alpha-Gen: {name}")
async def ltx_compute(name: str):
    """Stage „LTX Alpha-Gen” (user decision 2026-10-16): LTX-2.5 IC-LoRA Alpha-Gen on the work area, guided by the
    mask 1+2 (other objects painted out of the guide), limited to the object -> runs/v### (tool.stage_runs.ltx).
    A part of a group: computed as the group on its parent (as stages 3 / 4)."""
    asked = name
    name = _objects().get(name).get("part_of") or name
    st = _ltx_status()
    if st["missing_models"]:
        raise HTTPException(status_code=409, detail="Brak modeli LTX-2.5 (" + ", ".join(st["missing_models"]) +
                            f"; {st['missing_gb']} GB) – Doktor → „Zainstaluj LTX”")
    if st["missing_nodes"]:
        raise HTTPException(status_code=409, detail="Comfy bez węzłów LTX: " + ", ".join(st["missing_nodes"]) +
                            " – zaktualizuj ComfyUI (≥ 0.39) / zainstaluj ComfyUI-PlateMatte")
    frames = _work_frames()
    _require_core_on(name, frames)
    _require_stage(name, "ltx")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        run = await asyncio.get_running_loop().run_in_executor(None, lambda: pipe.ltx_run(name, frames, progress_cb=cb))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    rj = json.loads((_objects().dir(name) / "runs" / run / "run.json").read_text(encoding="utf-8"))
    return {"success": True, "run": run, "object": name, "asked": asked, "frames": [frames[0], frames[-1]],
            "seconds": rj.get("seconds"), "batches": rj.get("ltx_batches"), "group": rj.get("group")}


class VmmSettingsRequest(BaseModel):
    base: Optional[str] = None           # core (mask 1+2) | matanyone (stage-3 result)
    reach_px: Optional[float] = None
    protect_px: Optional[float] = None
    protect_auto: Optional[bool] = None  # True = protect band by base (0 core / 3 matanyone)
    max_side: Optional[int] = None
    batch: Optional[int] = None
    overlap: Optional[int] = None
    seed: Optional[int] = None
    offload: Optional[str] = None        # auto | on | off


def _vmm_status() -> Dict[str, Any]:
    """Stage VideoMaMa: weight files (present / missing), the Comfy nodes it needs (None = Comfy not answering)."""
    from engine.videomama import missing_nodes, models_status
    cfg = load_config()
    models = models_status(cfg.get("models_dir"))
    try:
        nodes = missing_nodes((STATE.get("comfy_client") or ComfyClient.from_config()).get_object_info())
    except Exception:
        nodes = None
    miss = [m for m in models if not m["present"]]
    return {"models": models, "missing_models": [f"{m['dir']}/{m['file']}" for m in miss],
            "missing_gb": round(sum(m["gb"] for m in miss), 1), "missing_nodes": nodes,
            "ready": not miss and nodes == []}


@app.get("/api/object/{name}/videomama")
def vmm_info(name: str):
    """Stage „VideoMaMa” (2026-10-16): the object's settings (defaults.json "videomama" <- object `videomama`), the
    stage result, weights / nodes."""
    from engine.stages import stage_run
    from engine.videomama import BASES, object_settings
    store = _objects()
    run, ok, why = stage_run(store, name, "videomama")
    return {"settings": object_settings(store.get(name)), "bases": list(BASES), "run": run, "valid": ok, "why": why,
            **_vmm_status()}


@app.put("/api/object/{name}/videomama")
@object_edit
def vmm_set(name: str, req: VmmSettingsRequest):
    """Stage VideoMaMa settings of the object (base, reach / protect px, resolution, batch / overlap); protect_auto =
    the protect band by base again."""
    from engine.videomama import object_settings, vmm_settings
    store = _objects()
    cur = dict(store.get(name).get("videomama") or {})
    body = req.model_dump()
    cur.update({k: v for k, v in body.items() if v is not None and k != "protect_auto"})
    if body.get("protect_auto"):
        cur.pop("protect_px", None)
    clean = vmm_settings({"videomama": cur})
    store.update(name, videomama={k: clean[k] for k in cur})
    return {"success": True, "settings": object_settings(store.get(name))}


@app.get("/api/object/{name}/videomama_preview/{frame}")
def vmm_preview(name: str, frame: int, what: str = "band", base: Optional[str] = None, reach_px: Optional[float] = None,
                protect_px: Optional[float] = None, protect_auto: Optional[int] = None):
    """VideoMaMa palette, live: what=band – 128 = where VideoMaMa adds, 255 = protect band (proxy PNG); what=alpha –
    the current VideoMaMa version's raw matte put over the base with these values (404 without a VideoMaMa version)."""
    from engine.videomama import object_settings, vmm_settings
    name = _objects().get(name).get("part_of") or name
    cur = {k: v for k, v in object_settings(_objects().get(name)).items() if k != "protect"}
    cur.update({k: v for k, v in {"base": base, "reach_px": reach_px, "protect_px": protect_px}.items() if v is not None})
    if protect_auto:
        cur["protect_px"] = None
    s = vmm_settings({"videomama": cur})
    try:
        band, alpha = _pipeline().vmm_frame_preview(name, frame, s)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    if what == "alpha" and alpha is None:
        raise HTTPException(status_code=404, detail="Brak wyniku VideoMaMa – podgląd wyniku po „▶ Policz VideoMaMa”")
    m = band if what != "alpha" else (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    sc = _proxy_scale()
    w, h = _plate_size()
    small = cv2.resize(m, (max(1, round(w * sc)), max(1, round(h * sc))),
                       interpolation=cv2.INTER_NEAREST if what != "alpha" else cv2.INTER_AREA)
    return _png_response(small)


@app.post("/api/object/{name}/videomama/refinish")
@long_job("VideoMaMa – suwaki na wynik: {name}")
async def vmm_refinish(name: str):
    """„✚ Suwaki na wynik”: the current VideoMaMa version's raw mattes put over the base again with the saved base /
    reach / protect -> a new version (no model run)."""
    name = _objects().get(name).get("part_of") or name
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        run = await asyncio.get_running_loop().run_in_executor(None, lambda: pipe.vmm_refinish(name, _work_frames(), progress_cb=cb))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "run": run, "object": name}


@app.post("/api/object/{name}/videomama")
@long_job("VideoMaMa: {name}")
async def vmm_compute(name: str):
    """Stage „VideoMaMa” (user decision 2026-10-16): VideoMaMa on the work area guided by the mask 1+2, put over the
    base (mask 1+2 or the stage-3 result) -> runs/v### (tool.stage_runs.videomama). A part: its group on the parent."""
    asked = name
    name = _objects().get(name).get("part_of") or name
    st = _vmm_status()
    if st["missing_models"]:
        raise HTTPException(status_code=409, detail=f"Brak wag VideoMaMa ({len(st['missing_models'])} plików, "
                            f"{st['missing_gb']} GB) – Doktor → „Zainstaluj VideoMaMa”")
    if st["missing_nodes"]:
        raise HTTPException(status_code=409, detail="Comfy bez węzłów VideoMaMa: " + ", ".join(st["missing_nodes"]) +
                            " – Doktor → „Zainstaluj VideoMaMa”, potem restart ComfyUI")
    frames = _work_frames()
    _require_core_on(name, frames)
    _require_stage(name, "videomama")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    models_dir = load_config().get("models_dir")
    try:
        run = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.vmm_run(name, frames, Path(models_dir) if models_dir else None, progress_cb=cb))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    rj = json.loads((_objects().dir(name) / "runs" / run / "run.json").read_text(encoding="utf-8"))
    return {"success": True, "run": run, "object": name, "asked": asked, "frames": [frames[0], frames[-1]],
            "seconds": rj.get("seconds"), "batches": rj.get("vmm_batches"), "group": rj.get("group")}
class AnalyzeRequest(BaseModel):
    run: str                           # the base version (or a stage name: matanyone | videomama | vitmatte | ltx | core_base)
    thr_px: float = 0.5                # proposal: |median offset| > thr_px ...
    min_frames: int = 5                # ... for >= min_frames frames in a row


def _analysis_target(name: str, run: str) -> Tuple[str, str]:
    """(object, version) of an analysis request: a part -> its group's parent; a stage name -> that stage's version."""
    from engine.stages import stage_run
    o = _objects()
    name = o.get(name).get("part_of") or name
    if run in ("matanyone", "videomama", "vitmatte", "ltx", "core_base"):
        r, _, _ = stage_run(o, name, run)
        if not r:
            raise HTTPException(status_code=404, detail=f"Brak wyniku etapu „{run}” – policz go najpierw")
        run = r
    if not (o.dir(name) / "runs" / run / "run.json").exists():
        raise HTTPException(status_code=404, detail=f"Brak wersji {run}")
    return name, run


@app.post("/api/object/{name}/analyze")
@long_job("Analiza wersji (QA + krawędź): {name}")
async def analyze_version(name: str, req: AnalyzeRequest):
    """Stage Eksperymenty (2026-10-16): QA + edge-correction proposals of the chosen base version (measured once)."""
    name, run = _analysis_target(name, req.run)
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        return {"success": True, "object": name, **(await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.analyze_run(name, run, req.thr_px, req.min_frames, cb)))}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/object/{name}/edge_recount")
def edge_recount(name: str, v: str, thr_px: float = 0.5, min_frames: int = 5):
    """Live count of the proposals with other thresholds (saved measurements only; nothing written)."""
    name, run = _analysis_target(name, v)
    rdir = _objects().dir(name) / "runs" / run
    if not any((rdir / "edge").glob("e_*.npz")):
        raise HTTPException(status_code=404, detail="Wersja bez pomiaru krawędzi – „▶ Analizuj”")
    rc = _pipeline().edge_recount(rdir, thr_px, min_frames)
    return {"run": run, "n_fix": rc["n_fix"], "median_abs": rc["median_abs"],
            "pieces": [{k: p[k] for k in ("id", "first", "last", "n_frames", "median_px", "kind")} for p in rc["pieces"]]}


class ExperimentRequest(BaseModel):
    base: str                          # base version (or a stage name: matanyone | videomama | vitmatte | ltx | core_base)
    mode: str = "matanyone"            # edge engine: matanyone | backlit | core
    fix: List[str] = []                # edge pieces of the base to correct first (from „▶ Analizuj”)


@app.post("/api/object/{name}/experiment")
@long_job("Eksperyment z bazy: {name}")
async def experiment_from_base(name: str, req: ExperimentRequest):
    """Stage Eksperymenty (2026-10-16): „Przelicz z tej bazy” – the base version's mask -> threshold zones -> optional
    edge correction of the base -> edge engine -> a new version (QA + edge report, origin in run.json)."""
    name, run = _analysis_target(name, req.base)
    if req.mode == "backlit" and not _backlit_pattern():
        raise HTTPException(status_code=409, detail="Brak plate'u backlit – silnik „backlit” niedostępny")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    from engine.zones import MissingFront
    try:
        new = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.experiment_from_base(name, run, req.mode, req.fix, cb))
    except MissingFront as e:
        raise _missing_front_409(e)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    from engine.export import mark_dependents
    deps = mark_dependents(_objects(), name)
    qa = json.loads((_objects().dir(name) / "runs" / new / "qa.json").read_text(encoding="utf-8"))
    return {"success": True, "run": new, "object": name, "base": run, "mode": req.mode, "fixed": len(req.fix),
            "gate_ok": qa.get("gate_ok"), "flicker_median": qa.get("flicker_median"),
            "pieces_to_fix": (qa.get("edge") or {}).get("pieces_to_fix"), "dependents_to_recompute": deps}


class EdgeFixRequest(BaseModel):
    run: Optional[str] = None          # base version (default: the active one)
    skip: List[str] = []               # piece ids the user unticked (never shifted)


@app.post("/api/object/{name}/edge_fix")
@long_job("Poprawka krawędzi: {name}")
async def apply_edge_fix(name: str, req: EdgeFixRequest = EdgeFixRequest()):
    """'Zastosuj poprawkę krawędzi' (Z27 pkt 1): the former automatic second round, as a new runs/v###."""
    _require_stage(name, "final")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    from engine.zones import MissingFront
    try:
        _, rep, run_dir = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.apply_edge_fix(name, _shot_frames(), req.run, req.skip, cb))
    except MissingFront as e:
        raise _missing_front_409(e)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    from engine.export import mark_dependents
    deps = mark_dependents(_objects(), name)
    return {"success": True, "run": run_dir.name, "report": rep.to_dict(), "dependents_to_recompute": deps}


@app.get("/api/object/{name}/motion_blur/{frame}")
def motion_blur_preview(name: str, frame: int, v: Optional[str] = None, strength: float = 1.0,
                        shutter: Optional[float] = None):
    """Stage Eksperymenty „Motion blur alfy” (2026-10-16): preview of one frame – the version `v` (default the active
    one) with the object's motion blur at `strength` and `shutter` (deg, default the manifest's shutter_deg)
    (engine/motionblur.py), proxy PNG like view=alpha.
    Headers X-MD-MB-Max-Px (longest blur, plate px), X-MD-MB-Moving-Px, X-MD-MB-Run."""
    _objects().get(name)
    try:
        a, st = _pipeline().motion_blur_frame(name, frame, _shot_frames(), v, strength, shutter)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    s = _proxy_scale()
    w, h = _plate_size()
    small = cv2.resize(_export_quantize(a), (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    r = _png_response(_alpha_u8(small))
    r.headers["X-MD-MB-Max-Px"] = str(st["max_px"])
    r.headers["X-MD-MB-Moving-Px"] = str(st["moving_px"])
    r.headers["X-MD-MB-Run"] = st["run"]
    return r


class MotionBlurRequest(BaseModel):
    run: Optional[str] = None      # base version (default: the active one)
    strength: float = 1.0          # x the physical blur of the shutter (0-10)
    shutter_deg: Optional[float] = None   # shutter angle (deg, 0-720); None = the manifest's shutter_deg


@app.post("/api/object/{name}/motion_blur")
@object_edit
@long_job("Motion blur alfy: {name}")
async def apply_motion_blur(name: str, req: MotionBlurRequest = MotionBlurRequest()):
    """„Zastosuj motion blur” (stage Eksperymenty, 2026-10-16): new runs/v### = the base version with the object's
    motion blur in every frame (alpha only)."""
    _require_stage(name, "final")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        run_dir, mb = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.apply_motion_blur(name, _shot_frames(), req.run, req.strength, cb, req.shutter_deg))
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    from engine.export import mark_dependents
    deps = mark_dependents(_objects(), name)
    return {"success": True, "run": run_dir.name, "motion_blur": mb, "dependents_to_recompute": deps}


@app.get("/api/object/{name}/edge_preview/{frame}")
def edge_preview_frame(name: str, frame: int, band_scale: Optional[float] = None, shift_px: Optional[float] = None,
                       band_out_px: Optional[float] = None, band_in_px: Optional[float] = None):
    """Stage 3 palette: preview of the MatAnyone edge before computing – proxy PNG, 255 = mask kept (mask 1+2 + shrink /
    grow), 128 = adaptive band where MatAnyone decides, 0 = background; slider values not yet saved can be passed.
    Headers X-MD-Source (mask12 | mask12+group), X-MD-Band-Px and X-MD-Outside-Crop-Px."""
    _objects().get(name)
    try:
        m, info = _pipeline().edge_preview(name, frame, band_scale, shift_px, band_out_px, band_in_px)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    s = _proxy_scale()
    w, h = _plate_size()
    small = cv2.resize(m, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_NEAREST)
    r = _png_response(small)
    r.headers["X-MD-Source"] = info["source"]
    r.headers["X-MD-Band-Px"] = str(info["band_px"])
    r.headers["X-MD-Outside-Crop-Px"] = str(info["outside_crop_px"])
    return r


@app.get("/api/object/{name}/edge_pieces/{frame}")
def edge_pieces_frame(name: str, frame: int, v: Optional[str] = None):
    """Stage-4 overlay: contour samples of the pieces to correct in this frame, in plate px
    (kind halo = red / shrink, cut = blue / grow)."""
    from engine.crop import CropSegment
    from engine.pipeline import load_edge_measure, piece_id
    store = _objects()
    run = v or store.get(name)["tool"].get("active_run")
    if not run:
        return {"frame": frame, "pieces": []}
    rdir = store.dir(name) / "runs" / run
    ej, ep = rdir / "edge" / "edge.json", rdir / "edge" / f"e_{frame:04d}.npz"
    if not ej.exists() or not ep.exists():
        return {"frame": frame, "pieces": []}
    fix = set(json.loads(ej.read_text(encoding="utf-8")).get("fix", []))
    rj = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
    segs = {d["seg_id"]: CropSegment.from_dict(d) for d in rj.get("segments_def", [])}
    seg_id, m = load_edge_measure(ep)
    sg = segs.get(seg_id)
    if sg is None:
        return {"frame": frame, "pieces": []}
    by: Dict[str, Dict[str, Any]] = {}
    for pt, pc in zip(m.points, m.pieces):
        pid = piece_id(pc)
        if pid not in fix or pc not in m.piece_median:
            continue
        med = m.piece_median[pc]
        e = by.setdefault(pid, {"id": pid, "median_px": round(med, 3), "kind": "halo" if med > 0 else "cut", "pts": []})
        e["pts"].append([round(sg.crop[0] + float(pt[0]), 1), round(sg.crop[1] + float(pt[1]), 1)])
    return {"frame": frame, "run": run, "pieces": list(by.values())}


def _export_context(store) -> List[Dict[str, Any]]:
    """Objects with a valid final result = the context of an export (front objects subtracted, parts split)."""
    from engine.stages import blocked_for_export
    every = store.list()
    bad = set(blocked_for_export(store, [o["name"] for o in every], Path(STATE["work_dir"]), _mask_plate_denoise()))
    return [o for o in every if o["name"] not in bad]


def _object_alpha_full(name: str, frame: int, run: Optional[str], export: bool) -> Optional[np.ndarray]:
    """Full-frame alpha of the object's version in `frame`, assembled exactly like the export (crop segments pasted,
    overlaps crossfaded). `export=True`: the matte as `export_objects` writes it – objects in front (with a valid
    final result) subtracted, parts of a whole split – for the active version only. None without a result."""
    from engine.export import FrameAlphas, run_alpha_loader
    store = _objects()
    obj = store.get(name)
    w, h = _plate_size()
    run = run or obj["tool"].get("active_run")
    if not run or not (store.dir(name) / "runs" / run / "run.json").exists():
        return None
    if export and run == obj["tool"].get("active_run"):
        ctx = _export_context(store)
        if any(o["name"] == name for o in ctx):
            return _export_quantize(FrameAlphas(Path(STATE["work_dir"]), ctx, w, h).frame(frame)[name])
    return _export_quantize(run_alpha_loader(store.dir(name), run, w, h)(frame))


WHOLE_SOURCES = ("alpha", "core", "core_fix")


def _member_mask(store, name: str, frame: int, w: int, h: int, src: str = "alpha",
                 run: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(alpha, core) of one object in `frame` for the whole-of-parts preview, from the layer of the view (bug 2026-10-15:
    the whole was always the active alpha, so the Rdzeń / maska 1+2 / Alpha views of a linked object looked the same):
    src "alpha" = the version `run` (default the active one), else the mask 1+2 / SAM3 core; "core_fix" = the mask 1+2,
    else the core; "core" = the SAM3 core. core = core/c_####.png (seam detection, like FrameAlphas.core_of)."""
    from engine.export import run_alpha_loader
    from engine.workarea import is_visible, object_life
    o = store.get(name)
    if not is_visible(frame, object_life(o)):
        return None, None
    od = store.dir(name)
    cm = cv2.imread(str(od / "core" / f"c_{frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
    core = None if cm is None else (cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST) if cm.shape != (h, w) else cm) > 127
    run = run or (o.get("tool") or {}).get("active_run")
    if src == "alpha" and run and (od / "runs" / run / "run.json").exists():
        return run_alpha_loader(od, run, w, h)(frame), core
    for sub in (("core",) if src == "core" else ("core_fix", "core")):
        m = cv2.imread(str(od / sub / f"c_{frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            return (m > 127).astype(np.float32), core
    return None, core


def _whole_of_parts(store, obj: Dict[str, Any], frame: int, pw: int, ph: int, src: str = "alpha",
                    run: Optional[str] = None) -> np.ndarray:
    """8-bit proxy PNG of the whole (parent + parts, seam filled with the parent's part_seam_px) – the viewer's
    „całość” layer. Members without a final result use their core, so it works from stage 1 on; objects in front are
    not subtracted here (the export does that)."""
    from engine.export import part_groups, seam_px, seamless_union
    parent = obj.get("part_of") or obj["name"]
    every = store.list()
    members = part_groups(every).get(parent)
    if not members:
        raise HTTPException(status_code=404, detail=f"„{parent}” nie ma części – całość = sam obiekt")
    w, h = _plate_size()
    alphas, cores = [], []
    if src not in WHOLE_SOURCES:
        raise HTTPException(status_code=400, detail=f"src: {' | '.join(WHOLE_SOURCES)}")
    for n in members:   # `run` (a version shown for comparison) belongs to the requested object only
        a, c = _member_mask(store, n, frame, w, h, src, run if n == obj["name"] else None)
        if a is not None:
            alphas.append(a); cores.append(c)
    if not alphas:
        raise HTTPException(status_code=404, detail="Brak masek obiektu i jego części w tej klatce")
    u = seamless_union(alphas, cores, gap_px=seam_px(store.get(parent)))
    return _alpha_u8(cv2.resize(u, (pw, ph), interpolation=cv2.INTER_AREA))


def _export_quantize(a: np.ndarray) -> np.ndarray:
    """The preview shows what the export FILE holds: an EXR export stores half floats (half round trip here); a PNG
    export stores 8 bits (the 8-bit quantization of the preview / tile below is the same `* 255 + 0.5`)."""
    from engine.export import resolve_export_format
    fmt = resolve_export_format(_export_format_setting(), int(STATE["manifest"].get("bit_depth", 16)))
    return a.astype(np.float16).astype(np.float32) if fmt == "exr" else np.clip(a, 0, 1).astype(np.float32)


def _alpha_u8(a: np.ndarray) -> np.ndarray:
    """8-bit alpha exactly as `export_objects` writes PNG mattes (round half up)."""
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


@app.get("/api/object/{name}/frame/{frame}")
def object_frame(name: str, frame: int, view: str = "overlay", v: Optional[str] = None, export: int = 0, cand: int = 0,
                 v2: Optional[str] = None, src: str = "alpha"):
    """Proxy-size PNG of the object in a frame: view=core | alpha | overlay (contour on the plate proxy).
    view=alpha is the faithful preview (plan step 7): the version's alpha assembled like the export (INTER_AREA to the
    proxy); export=1 = the matte as exported (front objects subtracted, parts split) when the object can be exported.
    Z9: view=flicker = the version's flicker map (runs/v###/flicker_map.png, INFERNO, QA bbox) as a BGRA PNG (alpha =
    brightness, so it overlays only where the edge flickers). Z10: view=cand&cand=i = mask of SAM3 candidate i."""
    store = _objects()
    obj = store.get(name)
    w, h = _plate_size()
    s = _proxy_scale()
    pw, ph = max(1, round(w * s)), max(1, round(h * s))
    run = v or obj["tool"].get("active_run")
    alpha = None
    if view == "flicker":
        rd = store.dir(name) / "runs" / (run or "")
        fm = cv2.imread(str(rd / "flicker_map.png"), cv2.IMREAD_COLOR) if run else None
        if fm is None:
            raise HTTPException(status_code=404, detail="Brak mapy migotania – „Przelicz finalnie” (QA zapisuje runs/v###/flicker_map.png)")
        try:
            x0, y0, x1, y1 = json.loads((rd / "qa.json").read_text(encoding="utf-8")).get("bbox") or (0, 0, w, h)
        except (OSError, ValueError):
            x0, y0, x1, y1 = 0, 0, w, h
        full = np.zeros((h, w, 3), np.uint8)
        if fm.shape[:2] != (y1 - y0, x1 - x0):   # the map covers the QA bbox (+40 px); anything else -> fit it there
            fm = cv2.resize(fm, (max(1, x1 - x0), max(1, y1 - y0)), interpolation=cv2.INTER_LINEAR)
        full[y0:y1, x0:x1] = fm
        small = cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA)
        bgra = cv2.cvtColor(small, cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = small.max(axis=2)
        return _png_response(bgra)
    if view == "full":   # the whole of an object made of parts (parent + parts), as `<parent>_full` is assembled
        _png = _whole_of_parts(store, obj, frame, pw, ph, src, v)   # src = the layer of the view (alpha | core | core_fix)
        return _png_response(_png)
    if view == "cand":   # Z10: SAM3 candidate mask (stored at proxy size by /detect)
        m = cv2.imread(str(store.dir(name) / "candidates" / f"c_{int(cand):02d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak kandydata – najpierw „Wykryj kandydatów”")
        if m.shape != (ph, pw):
            m = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_LINEAR)
        return _png_response(m)
    if view == "alpha_diff":   # comparison of two versions (stage 4, 2026-10-15): |alpha(v) - alpha(v2)| as a mask
        if not v or not v2:
            raise HTTPException(status_code=400, detail="alpha_diff: podaj v i v2 (dwie wersje)")
        a, b = _object_alpha_full(name, frame, v, False), _object_alpha_full(name, frame, v2, False)
        if a is None or b is None:
            raise HTTPException(status_code=404, detail="Brak wyniku jednej z wersji w tej klatce")
        alpha = np.clip(np.abs(a - b) * 3.0, 0, 1).astype(np.float32)   # x3: small soft differences stay visible
    elif view in ("alpha", "overlay") and run:
        alpha = _object_alpha_full(name, frame, run, bool(export))
    if view == "preview":
        pp = store.dir(name) / "preview" / f"p_{frame:04d}.png"
        m = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak podglądu tej klatki")
        alpha = m.astype(np.float32) / 255.0
    if view in ("delta_add", "delta_remove", "zone_add", "zone_remove", "zone_alpha", "zone_trimap", "zone_raw", "zone_area"):
        # delta_*: region changed by the last local correction (stage 3 overlay); zone_*: what the previewed zone
        # adds / removes on the core (stage 4 "Podgląd (ta klatka)"); zone_alpha / zone_trimap: `matte` zone preview
        tag = {"delta_add": "da", "delta_remove": "dr", "zone_add": "za", "zone_remove": "zr", "zone_alpha": "zalpha",
               "zone_trimap": "ztrimap", "zone_raw": "zraw", "zone_area": "zarea"}[view]
        pp = store.dir(name) / "preview" / f"{tag}_{frame:04d}.png"
        m = cv2.imread(str(pp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak korekty w tej klatce" if tag[0] == "d" else "Brak podglądu strefy w tej klatce")
        alpha = m.astype(np.float32) / 255.0
    if view == "zone_engines":   # engine weight map of the `matte` zone preview (RGB PNG, proxy size)
        pp = store.dir(name) / "preview" / f"zeng_{frame:04d}.png"
        m = cv2.imread(str(pp), cv2.IMREAD_COLOR)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak podglądu strefy w tej klatce")
        ok, buf = cv2.imencode(".png", cv2.resize(m, (pw, ph), interpolation=cv2.INTER_AREA))
        from fastapi.responses import Response
        return Response(content=buf.tobytes(), media_type="image/png")
    if view == "key":   # the key mask of this frame (points / candidate / brush) – stage 1 overlay
        m = cv2.imread(str(store.dir(name) / "keys" / f"k_{frame:04d}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak klucza w tej klatce")
        alpha = m.astype(np.float32) / 255.0
    if view == "core_fix":   # stage 2: the mask 1+2 (core + brush corrections, „Zastosuj poprawki”)
        m = cv2.imread(str(store.dir(name) / "core_fix" / f"c_{frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak maski 1+2 w tej klatce – „▶▶ Zastosuj poprawki” (etap 2)")
        alpha = m.astype(np.float32) / 255.0
    if alpha is None:
        cp = store.dir(name) / "core" / f"c_{frame:04d}.png"
        m = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=404, detail="Brak wyniku dla tej klatki")
        alpha = m.astype(np.float32) / 255.0
    small = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_AREA)
    if view == "alpha":
        img = _alpha_u8(small)   # faithful preview: the export's 8-bit rounding (tests: hash of the preview = hash of the export)
    elif view in ("core", "core_fix", "alpha_diff", "preview", "key", "delta_add", "delta_remove", "zone_add", "zone_remove", "zone_alpha",
                  "zone_trimap", "zone_raw", "zone_area"):
        img = (np.clip(small, 0, 1) * 255).astype(np.uint8)
    else:
        base = (_load_proxy_display(frame) * 255).astype(np.uint8)
        cnts, _ = cv2.findContours((small > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        img = base.copy()
        cv2.drawContours(img, cnts, -1, (80, 255, 80), 1)
    ok, buf = cv2.imencode(".png", img)
    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/object/{name}/frame/{frame}/tile")
def object_frame_tile(name: str, frame: int, x: int, y: int, w: int, h: int, view: str = "alpha", v: Optional[str] = None,
                      export: int = 0):
    """Z8 / plan step 7: full-resolution crop [x, y, w, h] (plate px) of a mask of the object in `frame` as an 8-bit
    PNG: view=core (core/c_####.png) | key (keys/k_####_mask.png) | alpha (the version's alpha like the export;
    export=1 as exported) | zone_alpha (preview of the `matte` zone). Drawn by the UI instead of the proxy overlay
    when the plate tile is active, so strands are judged at plate resolution."""
    store = _objects()
    store.get(name)
    pw, ph = _plate_size()
    x, y, w, h = _tile_rect(x, y, w, h, pw, ph)
    d = store.dir(name)
    if view == "alpha":
        a = _object_alpha_full(name, frame, v, bool(export))
        if a is None:
            raise HTTPException(status_code=404, detail="Brak wyniku finalnego w tej klatce")
        return _png_response(_alpha_u8(a[y:y + h, x:x + w]))
    paths = {"core": d / "core" / f"c_{frame:04d}.png", "key": d / "keys" / f"k_{frame:04d}_mask.png",
             "zone_alpha": d / "preview" / f"zalpha_{frame:04d}.png", "core_fix": d / "core_fix" / f"c_{frame:04d}.png"}
    if view not in paths:
        raise HTTPException(status_code=400, detail="view: core | core_fix | key | alpha | zone_alpha")
    m = cv2.imread(str(paths[view]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise HTTPException(status_code=404, detail={"core": "Brak rdzenia w tej klatce", "key": "Brak klucza w tej klatce",
                                                     "zone_alpha": "Brak podglądu strefy w tej klatce",
                                                     "core_fix": "Brak maski 1+2 w tej klatce"}[view])
    if m.shape != (ph, pw):   # a mask in another resolution (never expected): bring it to plate px first
        m = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_LINEAR)
    return _png_response(np.ascontiguousarray(m[y:y + h, x:x + w]))


class KeySaveRequest(BaseModel):
    source: str = "core"        # core | preview | empty – base mask before strokes
    strokes: List[Dict[str, Any]] = []
    polygons: List[Dict[str, Any]] = []
    positive: List[List[float]] = []
    negative: List[List[float]] = []


@app.post("/api/object/{name}/keys/{frame}")
@object_edit
def save_key(name: str, frame: int, req: KeySaveRequest):
    """User key (SPEC 7.1): base (current core / SAM3 preview) +/- brush strokes and polygons -> hard mask."""
    from engine.keys import KeyHistory, apply_strokes
    store = _objects()
    w, h = _plate_size()
    base = np.zeros((h, w), bool)
    if req.source in ("core", "preview"):
        d = "core" if req.source == "core" else "preview"
        fn = f"c_{frame:04d}.png" if d == "core" else f"p_{frame:04d}.png"
        if d == "core" and frame in store.keys(name):   # the frame result is its key: corrections accumulate on it
            m = store.key_mask(name, frame).astype(np.uint8) * 255
        else:
            m = cv2.imread(str(store.dir(name) / d / fn), cv2.IMREAD_GRAYSCALE)
        if m is None and frame in store.keys(name):
            m = store.key_mask(name, frame).astype(np.uint8) * 255
        if m is None:
            raise HTTPException(status_code=404, detail=f"Brak maski '{req.source}' w klatce {frame}")
        base = m > 127
    mask = apply_strokes(base, req.strokes, req.polygons)
    if not mask.any():
        raise HTTPException(status_code=400, detail="Pusta maska klucza")
    kind = "brush" if req.strokes else ("polygon" if req.polygons else ("points" if req.positive else "mask"))
    key = KeyHistory(store, name).save_key(frame, mask, author="user", kind=kind, positive=req.positive,
                                           negative=req.negative, strokes=req.strokes)
    store.update(name, tool={"final_valid": False, "stale_from": frame})
    return {"success": True, "key": key}


@app.delete("/api/object/{name}/keys/{frame}")
@object_edit
def delete_key(name: str, frame: int, confirm: bool = False):
    from engine.keys import KeyHistory
    store = _objects()
    keys = store.keys(name)
    if frame not in keys:
        raise HTTPException(status_code=404, detail="Brak klucza")
    if keys[frame].get("author") == "user" and not confirm:
        raise HTTPException(status_code=400, detail="Usunięcie klucza użytkownika wymaga potwierdzenia (confirm=true)")
    KeyHistory(store, name).delete_key(frame)
    store.update(name, tool={"final_valid": False, "stale_from": frame})
    return {"success": True}


class CorrectRequest(BaseModel):
    mode: str                 # add (hole / missed part) | remove (leak)
    point: List[float]        # plate px


@app.post("/api/object/{name}/correct/{frame}")
@object_edit
@long_job("Korekta SAM3: {name}", quiet=True)
def correct_frame(name: str, frame: int, req: CorrectRequest):
    """Stage 3 local correction (user request 2026-10-12): change ONLY the clicked hole / leak of the current frame
    mask (its key, else the SAM3 core). SAM3_Detect from seed points of the current mask + the click; the connected
    region under the click is added / removed, every other pixel stays identical. Saved at once as a key
    `correction` (undo in history.json); the changed region -> preview/da_####.png | dr_####.png for the overlay."""
    from engine.correct import local_correction, seed_points
    from engine.keys import KeyHistory
    if req.mode not in ("add", "remove") or len(req.point) != 2:
        raise HTTPException(status_code=400, detail="mode: add | remove, point: [x, y]")
    store = _objects()
    w, h = _plate_size()
    x, y = float(req.point[0]), float(req.point[1])
    if not (0 <= x < w and 0 <= y < h):
        raise HTTPException(status_code=400, detail="Klik poza kadrem")
    keys = store.keys(name)
    if frame in keys:
        base = store.key_mask(name, frame) > 0.5
    else:
        m = cv2.imread(str(store.dir(name) / "core" / f"c_{frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise HTTPException(status_code=400, detail="Brak rdzenia w tej klatce – najpierw „▶▶ Przelicz całość”")
        base = m > 127
    if not base.any():
        raise HTTPException(status_code=400, detail="Maska tej klatki jest pusta – postaw punkty obiektu w etapie 1")
    ys, xs = np.nonzero(base)
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    if req.mode == "add":
        pos, neg = seed_points(base) + [[x, y]], []
    else:
        pos, neg = seed_points(base, exclude=(x, y), exclude_r=max(40.0, 0.12 * diag)), [[x, y]]
    r = _sam3_preview_mask(name, frame, None, pos, neg)
    new, comp = local_correction(base, r["mask"], (x, y), req.mode)
    n = int(comp.sum())
    if n == 0:
        what = "nic nowego do dodania" if req.mode == "add" else "nic do usunięcia"
        return {"success": True, "frame": frame, "changed_px": 0, "seconds": r["seconds"],
                "detail": f"SAM3: {what} przy tym punkcie – kliknij bliżej brzegu albo użyj pędzla "
                          f"({'G' if req.mode == 'add' else 'R'})"}
    old = keys.get(frame) or {}
    corr = list(old.get("corrections") or []) + [{"mode": req.mode, "point": [x, y], "px": n}]
    key = KeyHistory(store, name).save_key(frame, new, author="user", kind="correction", corrections=corr,
                                           positive=old.get("positive") or [], negative=old.get("negative") or [])
    pdir = store.dir(name) / "preview"
    pdir.mkdir(parents=True, exist_ok=True)
    add = comp if req.mode == "add" else np.zeros_like(comp)
    rem = comp if req.mode == "remove" else np.zeros_like(comp)
    cv2.imwrite(str(pdir / f"da_{frame:04d}.png"), add.astype(np.uint8) * 255)
    cv2.imwrite(str(pdir / f"dr_{frame:04d}.png"), rem.astype(np.uint8) * 255)
    from engine.correct_region import save_delta
    save_delta(store.dir(name), frame, comp)     # Z21: the region of every correction of this key (union)
    store.update(name, tool={"final_valid": False, "stale_from": frame})
    return {"success": True, "frame": frame, "changed_px": n, "mode": req.mode, "seconds": r["seconds"], "key": key}


@app.get("/api/object/{name}/keys")
def list_keys(name: str):
    return {"keys": [{"frame": f, "author": k.get("author"), "kind": k.get("kind"), "t": k.get("t"),
                      "positive": k.get("positive") or [], "negative": k.get("negative") or [],
                      "has_strokes": bool(k.get("strokes")), "prompt": k.get("prompt")}
                     for f, k in sorted(_objects().keys(name).items())]}


@app.post("/api/object/{name}/undo")
@object_edit
def undo_key(name: str):
    from engine.keys import KeyHistory
    r = KeyHistory(_objects(), name).undo()
    return {"success": r is not None, "result": r}


@app.post("/api/object/{name}/redo")
@object_edit
def redo_key(name: str):
    from engine.keys import KeyHistory
    r = KeyHistory(_objects(), name).redo()
    return {"success": r is not None, "result": r}


@app.post("/api/object/{name}/rerun_from/{frame}")
@long_job("Przelicz zmienione przedziały: {name}")
async def rerun_from(name: str, frame: int):
    """Incremental core (7.3): only intervals whose inputs changed are recomputed (cache hash), the rest is reused."""
    import time as _time
    t0 = _time.time()
    r = await run_object_core(name)
    r["seconds"] = round(_time.time() - t0, 2)
    r["from_frame"] = frame
    return r


class ZoneRequest(BaseModel):
    mode: str
    criterion: str
    frame: int
    polygon: List[List[float]] = []
    strokes: List[Dict[str, Any]] = []
    param: float = 0.15
    soft: bool = False
    softness: float = 0.05
    color_lab: Optional[List[float]] = None
    static: bool = False
    enabled: bool = True
    # criterion `matte` (docs/PLAN_HAIR_TOOL.md, zone@1.1)
    preset: str = "hair"
    matting: Optional[str] = None       # None = the preset's default (defaults.json localmatte.presets.<preset>.matting)
    prompt: str = ""
    prompt_history: List[str] = []
    bg_strokes: List[Dict[str, Any]] = []
    bg_static: bool = True
    keys: Dict[str, Dict[str, Any]] = {}
    holdout: bool = True
    distractors: List[Dict[str, Any]] = []
    engine_hints: Dict[str, float] = {}
    params: Dict[str, Any] = {}


def _missing_front_409(e) -> HTTPException:
    """MissingFront (plan section 6) -> 409 {"detail", "missing_front", "queue"} for the UI dialog / md.py code 2."""
    return HTTPException(status_code=409, detail={"detail": str(e), "missing_front": e.missing, "queue": e.queue})


def _write_matte_preview(d: Path, frame: int, core: np.ndarray, res) -> Dict[str, Any]:
    """Stage-4 overlays of a `matte` zone preview: za_/zr_ (added / removed vs the core, as for the other criteria) +
    zalpha_ (soft alpha), ztrimap_ (0 / 128 / 255), zeng_ (engine weights: chroma = blue, knn = red, other = green)."""
    pv = d / "preview"
    pv.mkdir(parents=True, exist_ok=True)
    a = res.alpha
    added = res.area & ~core & (a > 0.02)
    removed = np.zeros_like(core)
    cv2.imwrite(str(pv / f"za_{frame:04d}.png"), added.astype(np.uint8) * 255)
    cv2.imwrite(str(pv / f"zr_{frame:04d}.png"), removed.astype(np.uint8) * 255)
    cv2.imwrite(str(pv / f"zalpha_{frame:04d}.png"), (np.clip(a, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(str(pv / f"ztrimap_{frame:04d}.png"), res.trimap)
    eng = np.zeros(core.shape + (3,), np.uint8)
    colours = {"chroma": 0, "knn": 2, "clean": 1}
    for k, w in res.weights.items():
        ch = colours.get(k, 1)
        eng[..., ch] = np.maximum(eng[..., ch], (np.clip(w, 0, 1) * 255).astype(np.uint8))
    if not res.weights and res.engines:   # forced engine: the whole area in its colour
        k = next(iter(res.engines))
        eng[..., colours.get(k, 1)] = res.area.astype(np.uint8) * 255
    cv2.imwrite(str(pv / f"zeng_{frame:04d}.png"), eng)
    st = {k: v for k, v in res.stats.items() if k not in ("box",)}
    return {"added_px": int(added.sum()), "removed_px": 0, "soft_px": int((res.area & (a > 0.02) & (a < 0.98)).sum()),
            "stats": st, "flags": list(res.flags)}


def _zone_status(name: str, zones) -> Dict[str, Dict[str, Any]]:
    """Card state of every zone for the stage-4 panel (plan step 5): policzona (in the active version and the final
    result is valid) / do przeliczenia / liczona… (a job runs on the object) / wyłączona; keys, display name
    (prompt or id), stats of the zone in the active version (run.json -> zones)."""
    from engine.zones import zone_keys
    store = _objects()
    tool = store.get(name).get("tool", {})
    run = tool.get("active_run")
    run_zones: Dict[str, Any] = {}
    if run:
        try:
            run_zones = json.loads((store.dir(name) / "runs" / run / "run.json").read_text(encoding="utf-8")).get("zones") or {}
        except (OSError, ValueError):
            run_zones = {}
    job = JOBS.busy_object(name)
    out: Dict[str, Dict[str, Any]] = {}
    for z in zones:
        keys = sorted(zone_keys(z))
        if job:
            state = "liczona"
        elif not z.enabled:
            state = "wyłączona"
        elif tool.get("final_valid") and z.zone_id in run_zones:
            state = "policzona"
        else:
            state = "do przeliczenia"
        prompt = (z.prompt or "").strip()
        out[z.zone_id] = {"state": state, "n_keys": len(keys), "keys": keys, "name": prompt[:40] if prompt else z.zone_id,
                          "run": run if z.zone_id in run_zones else None, "run_stats": run_zones.get(z.zone_id)}
    return out


@app.get("/api/object/{name}/zones")
def list_zones(name: str):
    from engine.zones import load_zones
    zones = load_zones(_objects().dir(name))
    return {"zones": [z.to_dict() for z in zones], "status": _zone_status(name, zones)}


@app.get("/api/zones/preset_defaults/{preset}")
def zone_preset_defaults(preset: str):
    """Default engine and params of a zone preset (defaults.json) – the zone card's "↺ z presetu"."""
    from engine.localmatte import PRESETS, preset_defaults
    if preset not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset: {'|'.join(PRESETS)}")
    matting, params = preset_defaults(preset)
    return {"preset": preset, "matting": matting, "params": params}


class ZoneInterpretRequest(BaseModel):
    prompt: str = ""
    frame: Optional[int] = None       # with a frame: Hough "-" strokes in the background of the zone / core
    zone_id: Optional[str] = None     # region = this zone's key / propagated drawing in `frame` (else the core bbox + 60 px)


@app.post("/api/object/{name}/zones/interpret")
def zone_interpret(name: str, req: ZoneInterpretRequest):
    """Plan step 9: what the zone description means without a VLM – preset, SAM3 phrase (`fg_en`), distractors,
    routing multipliers, warnings; with `frame`: proposed "-" strokes along straight lines in the background (Hough),
    to confirm in the UI (nothing is written here)."""
    from engine.localmatte_prompt import detect_bg_lines, interpret_prompt
    from engine.zones import key_zone, zone_keys, zone_mask_at
    store = _objects()
    store.get(name)
    plan = interpret_prompt(req.prompt)
    out: Dict[str, Any] = {"success": True, "plan": plan.to_dict(), "bg_strokes": []}
    if req.frame is None:
        return out
    d = store.dir(name)
    core = cv2.imread(str(d / "core" / f"c_{req.frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
    if core is None:
        raise HTTPException(status_code=404, detail=f"Brak rdzenia w klatce {req.frame} – „▶▶ Przelicz całość” (etap 1 / 2)")
    cb = core > 127
    region = None
    if req.zone_id:
        z = _load_zone_or_404(name, req.zone_id)
        kz = key_zone(z, req.frame) if req.frame in zone_keys(z) else z
        region = zone_mask_at(kz, cb.shape)
    if region is None or not region.any():
        ys, xs = np.where(cb)
        region = np.zeros_like(cb)
        if len(xs):
            region[max(0, ys.min() - 60):ys.max() + 61, max(0, xs.min() - 60):xs.max() + 61] = True
    img = _pipeline().loader.frame(req.frame)
    out["bg_strokes"] = detect_bg_lines(img, region, cb) if (plan.bg_line or not req.prompt.strip()) else []
    out["frame"] = int(req.frame)
    return out


class ZoneProposeRequest(BaseModel):
    frame: int
    prompt: Optional[str] = None      # zone description; None = the object's detect prompt
    min_area: int = 400


@app.post("/api/object/{name}/zones/propose")
@long_job("Zaproponuj strefę: {name}", quiet=True)
def zone_propose(name: str, req: ZoneProposeRequest):
    """Plan step 9 „Zaproponuj strefę”: SAM3_Detect with the prompt's subject on the crop around the core, minus the
    core, plus soft edge pixels outside the core -> zone proposals (polygons in plate px) to accept in the UI (each
    accepted one = `POST /zones/new` with that polygon). Read-only."""
    from engine.crop import choose_instance
    from engine.localmatte_prompt import interpret_prompt, propose_zones, soft_edge_candidates
    store = _objects()
    obj = store.get(name)
    d = store.dir(name)
    core = cv2.imread(str(d / "core" / f"c_{req.frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
    if core is None:
        raise HTTPException(status_code=404, detail=f"Brak rdzenia w klatce {req.frame} – „▶▶ Przelicz całość” (etap 1 / 2)")
    cb = core > 127
    plan = interpret_prompt(req.prompt or "")
    fg = plan.fg_en or (obj.get("detect", {}).get("prompts") or [None])[0] or None
    img = _pipeline().loader.frame(req.frame)
    h, w = cb.shape
    ys, xs = np.where(cb)
    sam_mask = None
    sam_error = None
    if len(xs):
        mx, my = max(64, int((xs.max() - xs.min()) * 0.3)), max(64, int((ys.max() - ys.min()) * 0.3))
        x0, x1 = max(0, int(xs.min()) - mx), min(w, int(xs.max()) + 1 + mx)
        y0, y1 = max(0, int(ys.min()) - my), min(h, int(ys.max()) + 1 + my)
        crop = np.clip(img[y0:y1, x0:x1], 0, 1).astype(np.float32)
        sc = min(1.0, 1008.0 / max(crop.shape[0], crop.shape[1]))
        small = cv2.resize(crop, (max(1, round(crop.shape[1] * sc)), max(1, round(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA) if sc < 1.0 else crop
        try:
            cands = choose_instance(_sam3_runner(), small, prompt=fg)
        except RuntimeError as e:
            cands, sam_error = [], str(e)
        core_small = cv2.resize(cb[y0:y1, x0:x1].astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        keep = [m for m in cands if ((m > 0.5) & core_small).sum() > 0.2 * max(1, core_small.sum())]   # the object, not another one
        if keep:
            u = np.max(np.stack([m for m in keep]), axis=0)
            sam_mask = np.zeros((h, w), np.float32)
            sam_mask[y0:y1, x0:x1] = cv2.resize(u.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    soft = soft_edge_candidates(img, cb)
    props = propose_zones(cb, sam_mask, soft, min_area=int(req.min_area))
    return {"success": True, "frame": int(req.frame), "fg": fg, "preset": plan.preset or "hair", "plan": plan.to_dict(),
            "sam3": sam_mask is not None, "sam3_error": sam_error, "proposals": props}


@app.post("/api/object/{name}/zones/{zone_id}")
@object_edit
def save_zone_ep(name: str, zone_id: str, req: ZoneRequest, preview_frame: Optional[int] = None):
    """Create (zone_id='new') or update a zone (SPEC 7.7). `?preview_frame=` -> live preview of that frame (< 1 s):
    what the zone adds / removes on the current core."""
    from engine.zones import CRITERIA, Zone, apply_zones, next_zone_id, save_zone
    if req.mode not in ("add", "remove") or req.criterion not in CRITERIA:
        raise HTTPException(status_code=400, detail="tryb add|remove, kryterium dark|bright|color|all|matte")
    fields = req.model_dump()
    if req.criterion == "matte":
        from engine.localmatte import ENGINES, PRESETS, preset_defaults
        if req.mode != "add":
            raise HTTPException(status_code=400, detail="Strefa „matte”: tryb Usuń jeszcze nieobsługiwany (kreski „−” wymuszają tło)")
        if req.preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"Strefa „matte”: preset {'|'.join(PRESETS)}")
        # step 9: the description names the kind of detail -> preset, unless one was chosen explicitly (request keeps `hair`)
        plan = None
        if req.prompt.strip():
            from engine.localmatte_prompt import interpret_prompt
            plan = interpret_prompt(req.prompt)
            if plan.preset and req.preset == "hair" and plan.preset != fields["preset"]:
                fields["preset"] = plan.preset
        # engine / params omitted -> the preset's defaults (hair: vitmatte + flow_mean r5 x2, black 0.1, white 0.9); explicit values win
        def_matting, def_params = preset_defaults(fields["preset"])
        if fields.get("matting") is None:
            fields["matting"] = def_matting
            fields["params"] = {**def_params, **(fields.get("params") or {})}
        if fields["matting"] not in ENGINES:
            raise HTTPException(status_code=400, detail=f"Strefa „matte”: silnik {'|'.join(ENGINES)}, preset {'|'.join(PRESETS)}")
        outline = str((fields.get("params") or {}).get("region", "draw")) == "outline"
        if len(req.polygon) < 3 and not req.strokes and not outline:   # outline mode: the drawing is an optional exclusion
            raise HTTPException(status_code=400, detail="Strefa „matte”: narysuj obszar (wielokąt albo pędzel +)")
    if fields.get("matting") is None:
        fields["matting"] = "matanyone"
    store = _objects()
    d = store.dir(name)
    zid = next_zone_id(d) if zone_id == "new" else zone_id
    z = Zone(zid, **fields)       # station: engine / params resolved from the (prompt-derived) preset above (hair = vitmatte + flow r3 x2)
    if req.criterion != "matte":
        plan = None
    if z.criterion == "color" and not z.color_lab:
        # Z24 bug: the UI did not send color_lab -> the key compared to the default grey [50, 0, 0]. Without a pipette
        # pick, the colour = median inside the drawn area at the zone's frame
        from engine.pipeline import PlateCropLoader
        from engine.zones import zone_median_lab
        try:
            z.color_lab = zone_median_lab(z, PlateCropLoader(STATE["manifest"]["plates"]["beauty"], _plate_display).frame(z.frame))
        except (FileNotFoundError, KeyError):
            z.color_lab = None
        if not z.color_lab:
            raise HTTPException(status_code=400, detail="Kryterium „kolor”: wybierz kolor pipetą albo narysuj obszar strefy")
    if z.criterion == "matte" and z.holdout:
        from engine.zones import MissingFront
        try:   # the objects in front must have masks (plan section 6) – checked at save time, not silently later
            _pipeline().front_holdout(name, [preview_frame] if preview_frame is not None else _shot_frames())
        except MissingFront as e:
            raise _missing_front_409(e)
    save_zone(d, z)
    store.update(name, tool={"final_valid": False})
    out: Dict[str, Any] = {"success": True, "zone": z.to_dict(), **({"plan": plan.to_dict()} if plan else {})}
    if preview_frame is not None and z.criterion == "matte":
        pv = _matte_preview(name, d, z, preview_frame)
        if pv is not None:
            out["preview"] = pv
        return out
    if preview_frame is not None:
        cp = d / "core" / f"c_{preview_frame:04d}.png"
        core = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if core is not None:
            from engine.pipeline import PlateCropLoader
            loader = PlateCropLoader(STATE["manifest"]["plates"]["beauty"], _plate_display)
            c = {preview_frame: core > 127}
            r = apply_zones([preview_frame], c, loader.frame, [z],
                            {zid: {preview_frame: np.eye(2, 3, dtype=np.float32)}})
            new = r.core[preview_frame]
            pv = d / "preview" / f"zone_{preview_frame:04d}.png"
            pv.parent.mkdir(parents=True, exist_ok=True)
            vis = np.zeros(new.shape + (3,), np.uint8)
            vis[new & c[preview_frame]] = (255, 255, 255)
            vis[new & ~c[preview_frame]] = (0, 255, 0)      # added
            vis[~new & c[preview_frame]] = (0, 0, 255)      # removed (red)
            cv2.imwrite(str(pv), vis)
            # stage-4 viewer overlay (view=zone_add / zone_remove): the preview used to exist only as pixel counts
            cv2.imwrite(str(d / "preview" / f"za_{preview_frame:04d}.png"), (new & ~c[preview_frame]).astype(np.uint8) * 255)
            cv2.imwrite(str(d / "preview" / f"zr_{preview_frame:04d}.png"), (~new & c[preview_frame]).astype(np.uint8) * 255)
            out["preview"] = {"path": str(pv), "added_px": int((new & ~c[preview_frame]).sum()),
                              "removed_px": int((~new & c[preview_frame]).sum())}
    return out


def _matte_preview(name: str, d: Path, z, frame: int) -> Optional[Dict[str, Any]]:
    """Preview of a `matte` zone in `frame` (< 2 s, crop of the zone): the region is the zone's KEY at that frame when
    one exists (plan step 5: painting = key, the preview follows the key), else the base drawing; `matanyone` shows its
    key hint (clean / chroma) with a `note`. None when the frame has no core yet. Overlays -> frame?view=zone_*."""
    from engine.localmatte import local_matte
    from engine.zones import key_zone, zone_bg_mask_at, zone_hints, zone_keys, zone_mask_at
    core = cv2.imread(str(d / "core" / f"c_{frame:04d}.png"), cv2.IMREAD_GRAYSCALE)
    if core is None:
        return None
    pipe = _pipeline()
    hold_fn, used = pipe.front_holdout(name, [frame], need=z.holdout)
    cb = (core > 127)
    if pipe.group_members(name):   # a group: the zone works on the whole (members' masks 1+2 + seam), as stage 4 does
        try:
            cb = pipe.preview_core(name, [frame])[frame]
        except FileNotFoundError:
            return None
    is_key = frame in zone_keys(z)
    kz = key_zone(z, frame) if is_key else z
    region = zone_mask_at(kz, cb.shape)
    from engine.zones import region_mode
    try:
        outline = region_mode(z) == "outline"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    from engine.zones import track_mode
    local = track_mode(z) == "local"
    if outline or (local and not is_key):
        # outline mode / local tracking without a key here (2026-10-15): the region the shot gets – propagated from the
        # keys (CoTracker, cached), minus the excluded objects and the parts of this object
        # the frames of the active version (as the dashed band, GET .../region): the cores exist only in the object's
        # life range – the whole shot asked for F1 and failed with 409 (user report 2026-10-16, CLIP2_A F44–70)
        frames = [x for x in _active_run_frames(name) if (d / "core" / f"c_{x:04d}.png").exists()] or [frame]
        try:
            regs, _, _ = pipe.zone_regions(name, z, frames, pipe.preview_core(name, frames), only=[frame])
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        region = regs.get(frame, np.zeros_like(cb))
    try:
        fg_fn, _ = pipe.zone_fg(name, z, [frame])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    fgm = None
    if fg_fn is not None:     # parts of this object (one matte with it): out of the region, sure foreground of the trimap
        fgm = fg_fn(frame)
        region = region & ~fgm
        cb = cb | fgm
    clean_fn, clean_src = pipe.clean_source()
    if z.matting == "clean" and clean_fn is None:
        raise HTTPException(status_code=409, detail={"detail": "Silnik „clean” wymaga clean plate'u – „Policz z ujęcia (auto)” "
                                                     "(POST /api/project/cleanplate_auto) albo wczytaj clean plate w etapie 0",
                                                     "cleanplate": "brak"})
    cp = clean_fn(frame) if clean_fn else None
    eng = z.matting
    note = None
    if eng == "matanyone":   # the sequence engine runs at "Przelicz finalnie" / "Zastosuj"; the preview shows its key hint
        eng = "clean" if cp else "chroma"
        note = "podgląd = podpowiedź klucza (clean / chroma); MatAnyone2 na kadrze strefy liczy się przy „Przelicz finalnie” / „Zastosuj”"
    if eng == "vitmatte":    # per-frame engine on the zone crop: the preview IS the engine (one frame, no cache; station 2026-10-13)
        from engine.localmatte import LocalMatteResult
        from engine.vitmatte import get_runner
        from engine.zones import vitmatte_tile_px, vitmatte_zone
        hold_m = hold_fn(frame) if z.holdout else None
        bgm = zone_bg_mask_at(kz, cb.shape)
        pipe.release_comfy_gpu("podgląd ViTMatte")      # Comfy's models off the GPU first (idle queue only)
        from engine.vitmatte import model_id_for
        al, tris, st = vitmatte_zone(kz, [frame], {frame: cb}, {frame: region}, lambda f: pipe.loader.frame(f),
                                     get_runner(model_id_for((kz.params or {}).get("vitmatte_model"))),
                                     holdout=(lambda f: hold_m), bg=(lambda f: bgm),
                                     fg_erode_px=int((kz.params or {}).get("fg_erode_px", 2)),
                                     model_px=int((kz.params or {}).get("model_px", 1008)),
                                     unknown_value=float((kz.params or {}).get("unknown_value", 0.5)),
                                     tile_px=vitmatte_tile_px(kz.params),
                                     fg_sure={frame: fgm} if fg_fn is not None else None)
        a = al.get(frame, np.zeros(cb.shape, np.float32))
        res = LocalMatteResult(alpha=a, area=region, trimap=tris.get(frame, np.zeros(cb.shape, np.uint8)),
                               box=tuple(st.get("bbox", (0, 0, 0, 0))), scale=float(st.get("scale", 1.0)),
                               engines={"vitmatte": a}, stats={**st, "seconds": st.get("seconds", 0.0)})
        note = "podgląd = ViTMatte na kadrze strefy (bez wygładzenia czasowego; params.temporal działa przy liczeniu)"
    else:
        res = local_matte(pipe.loader.frame(frame), cb, region, bg_mask=zone_bg_mask_at(kz, cb.shape),
                          holdout=hold_fn(frame) if z.holdout else None, engine=eng, preset=z.preset,
                          params=z.params or None, hints=zone_hints(kz),
                          clean=cp[0] if cp else None, clean_conf=cp[1] if cp else None)
    from engine.zones import finish_zone_alpha
    # live finish sliders in the UI (2026-10-16): the alpha BEFORE the finish + the area it acts on (0 = outside,
    # 128 = region, 255 = region ∩ core – forced to 1), so black / white / blur recompute in the browser without the model
    pvd = d / "preview"; pvd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pvd / f"zraw_{frame:04d}.png"), (np.clip(res.alpha, 0, 1) * 255).astype(np.uint8))
    area = region.astype(np.uint8) * 128
    area[region.astype(bool) & cb.astype(bool)] = 255
    cv2.imwrite(str(pvd / f"zarea_{frame:04d}.png"), area)
    res.alpha = finish_zone_alpha(res.alpha, region, cb, kz.params or {})   # black / white / blur sliders, live on the key frame
    return {**_write_matte_preview(d, frame, cb, res), "holdout_objects": used, "cleanplate": clean_src,
            "is_key": is_key, "frame": int(frame), "engine_used": eng, **({"note": note} if note else {})}


class ZoneKeyRequest(BaseModel):
    polygon: List[List[float]] = []
    strokes: List[Dict[str, Any]] = []
    bg_strokes: List[Dict[str, Any]] = []
    prompt: Optional[str] = None     # iteration note of the correction loop (plan step 7); None = keep, "" = clear


def _load_zone_or_404(name: str, zone_id: str):
    from engine.zones import load_zones
    z = next((x for x in load_zones(_objects().dir(name)) if x.zone_id == zone_id), None)
    if z is None:
        raise HTTPException(status_code=404, detail="Brak strefy")
    return z


class ZonePatchRequest(BaseModel):
    """Partial update of a zone from the panel card (plan step 5); absent fields stay as they are."""
    enabled: Optional[bool] = None
    prompt: Optional[str] = None
    preset: Optional[str] = None
    matting: Optional[str] = None
    holdout: Optional[bool] = None
    bg_static: Optional[bool] = None
    static: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    engine_hints: Optional[Dict[str, float]] = None


@app.patch("/api/object/{name}/zones/{zone_id}")
@object_edit
def patch_zone(name: str, zone_id: str, req: ZonePatchRequest):
    """Zone card (plan step 5): enable / disable, description (prompt; the previous one goes to prompt_history), preset,
    engine, holdout ("Licz mimo to bez holdoutu"). Any change invalidates the final result (new version on recompute)."""
    from engine.localmatte import ENGINES, PRESETS
    from engine.zones import save_zone
    z = _load_zone_or_404(name, zone_id)
    changes = {k: v for k, v in req.model_dump().items() if v is not None}
    if not changes:
        raise HTTPException(status_code=400, detail="Nic do zmiany")
    if "matting" in changes and changes["matting"] not in ENGINES:
        raise HTTPException(status_code=400, detail=f"Silnik: {'|'.join(ENGINES)}")
    if "preset" in changes and changes["preset"] not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Preset: {'|'.join(PRESETS)}")
    plan = None
    if "prompt" in changes:
        changes["prompt"] = changes["prompt"].strip()
        if z.prompt and changes["prompt"] != z.prompt:
            z.prompt_history = list(z.prompt_history or []) + [z.prompt]
        from engine.localmatte_prompt import interpret_prompt
        plan = interpret_prompt(changes["prompt"])
        if plan.preset and "preset" not in changes and plan.preset != z.preset and z.criterion == "matte":
            changes["preset"] = plan.preset     # step 9: the description names the kind of detail
    for k, v in changes.items():
        setattr(z, k, v)
    d = _objects().dir(name)
    save_zone(d, z)
    _objects().update(name, tool={"final_valid": False})
    return {"success": True, "zone": z.to_dict(), "changed": sorted(changes), "status": _zone_status(name, [z])[zone_id],
            **({"plan": plan.to_dict()} if plan else {})}


@app.post("/api/object/{name}/zones/{zone_id}/preview/{frame}")
def zone_preview_ep(name: str, zone_id: str, frame: int):
    """Automatic preview of the panel (plan step 5, debounce in the UI): the `matte` zone as saved, in `frame`
    (its key there when one exists) – nothing is written to the zone. 404 without a core in that frame."""
    from engine.zones import MissingFront
    z = _load_zone_or_404(name, zone_id)
    if z.criterion != "matte":
        raise HTTPException(status_code=400, detail="Podgląd automatyczny tylko dla strefy „matte” (inne: „Podgląd (ta klatka)”)")
    try:
        pv = _matte_preview(name, _objects().dir(name), z, frame)
    except MissingFront as e:
        raise _missing_front_409(e)
    if pv is None:
        raise HTTPException(status_code=404, detail=f"Brak rdzenia w klatce {frame} – „▶▶ Przelicz całość” (etap 1 / 2)")
    return {"success": True, "zone_id": zone_id, "preview": pv}


@app.put("/api/object/{name}/zones/{zone_id}/keys/{frame}")
@object_edit
def put_zone_key(name: str, zone_id: str, frame: int, req: ZoneKeyRequest):
    """Zone key (plan step 4): the drawing of the zone in `frame` (polygon / '+' strokes / '-' strokes). Painting in a
    frame without a key creates one; the base key (zone.frame) is updated when frame == zone.frame."""
    from engine.zones import save_zone, set_zone_key, zone_key_prompt, zone_keys
    from engine.stages import zone_kind
    z = _load_zone_or_404(name, zone_id)
    brush = zone_kind(z) == "brush"     # stage 2 correction: an empty key = the correction ends here (fades to it)
    if z.criterion != "matte" and not brush:
        raise HTTPException(status_code=400, detail="Klucze strefy dotyczą stref „matte” i poprawek pędzlem (etap 2)")
    outline = str((z.params or {}).get("region", "draw")) == "outline"   # drawing = optional exclusion
    if len(req.polygon) < 3 and not req.strokes and not outline and not brush:
        raise HTTPException(status_code=400, detail="Klucz strefy: narysuj obszar (wielokąt albo pędzel +)")
    set_zone_key(z, frame, req.polygon, req.strokes, req.bg_strokes, req.prompt)
    save_zone(_objects().dir(name), z)
    _objects().update(name, tool={"final_valid": False})
    return {"success": True, "zone": z.to_dict(), "keys": list(zone_keys(z)),
            "key": {**zone_keys(z)[int(frame)], "prompt": zone_key_prompt(z, int(frame))}}


@app.delete("/api/object/{name}/zones/{zone_id}/keys/{frame}")
@object_edit
def delete_zone_key(name: str, zone_id: str, frame: int):
    from engine.zones import delete_zone_key, save_zone, zone_keys
    z = _load_zone_or_404(name, zone_id)
    try:
        delete_zone_key(z, frame)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    save_zone(_objects().dir(name), z)
    _objects().update(name, tool={"final_valid": False})
    return {"success": True, "zone": z.to_dict(), "keys": list(zone_keys(z))}


@app.get("/api/object/{name}/zones/{zone_id}/keys")
def get_zone_keys(name: str, zone_id: str):
    from engine.zones import zone_key_prompt, zone_keys
    z = _load_zone_or_404(name, zone_id)
    return {"zone_id": zone_id, "base": int(z.frame), "prompt": z.prompt, "prompt_history": list(z.prompt_history or []),
            "keys": {str(k): {"polygon_pts": len(v["polygon"]), "strokes": len(v["strokes"]), "bg_strokes": len(v["bg_strokes"]),
                              "prompt": zone_key_prompt(z, k)} for k, v in zone_keys(z).items()}}


@app.get("/api/object/{name}/zones/{zone_id}/region/{frame}")
@long_job("Obszar strefy {zone_id} w klatce {frame}", quiet=True)
def get_zone_region(name: str, zone_id: str, frame: int):
    """The propagated region of the zone in `frame` (contours in plate px, bbox) – the starting point of a correction.
    The propagation transforms are cached (zones_cache/<zone>/transforms_*.npz); the first call tracks (CoTracker3)."""
    _load_zone_or_404(name, zone_id)
    try:
        return _pipeline().zone_region(name, zone_id, _active_run_frames(name), frame)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/object/{name}/zones/{zone_id}/apply")
@long_job("Zastosuj strefę {zone_id}: {name}")
async def apply_zone_ep(name: str, zone_id: str, stage: Optional[str] = None):
    """„Zastosuj” (plan step 4): only the zones are recomputed (cache) on the base alpha of the active version -> a new
    version runs/v### (QA + edge report); 409 with `missing_front` when objects in front lack masks.
    `?stage=vitmatte` (stage 4, 2026-10-15): on the stage-3 MatAnyone version, brush + matte zones only."""
    from engine.zones import MissingFront
    _load_zone_or_404(name, zone_id)
    if stage not in (None, "vitmatte"):
        raise HTTPException(status_code=400, detail="stage: vitmatte (albo brak)")
    par = _objects().get(name).get("part_of")
    if stage and par and par != name:
        raise HTTPException(status_code=409, detail=f"„{name}” jest częścią „{par}”: grupę liczy się jako całość – strefy "
                                                    f"ViTMatte zakładaj na obiekcie głównym „{par}”")
    if stage:
        _require_stage(name, "vitmatte")
    pipe = _pipeline()
    cb = _ws_progress("object_progress", name)
    try:
        _, run_dir, info = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.apply_zone(name, _shot_frames(), zone_id, cb, stage=stage))
    except MissingFront as e:
        raise _missing_front_409(e)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    from engine.export import mark_dependents
    deps = mark_dependents(_objects(), name)
    return {"success": True, "run": run_dir.name, **info, "dependents_to_recompute": deps}


@app.get("/api/object/{name}/color_lab/{frame}")
def zone_color_pick(name: str, frame: int, x: float, y: float):
    """Pipette of the "color" zone criterion (Z24): median Lab of 5 x 5 px of the plate-to-mask display image (the image
    the zone key is computed on) around (x, y) in plate px; only those scanlines are decoded."""
    from engine.pipeline import PlateCropLoader
    from engine.zones import sample_lab
    w, h = _plate_size()
    if not (0 <= x < w and 0 <= y < h):
        raise HTTPException(status_code=400, detail="Punkt poza kadrem")
    xi, yi = int(x), int(y)
    y0, y1, x0, x1 = max(0, yi - 2), min(h, yi + 3), max(0, xi - 2), min(w, xi + 3)
    loader = PlateCropLoader(STATE["manifest"]["plates"]["beauty"], _plate_display)
    try:
        patch = loader._rows(frame, y0, y1, x0, x1)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Brak klatki {frame}")
    bgr = np.median(np.clip(patch, 0, 1).reshape(-1, 3), axis=0)
    hexc = "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in bgr[::-1])
    return {"frame": frame, "x": xi, "y": yi, "lab": sample_lab(patch), "hex": hexc}


@app.delete("/api/object/{name}/zones/{zone_id}")
@object_edit
def delete_zone(name: str, zone_id: str):
    p = _objects().dir(name) / "zones" / f"{zone_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Brak strefy")
    p.unlink()
    _objects().update(name, tool={"final_valid": False})
    return {"success": True}


@app.get("/api/object/{name}/runs")
def object_runs(name: str):
    """Versions of the final result (Z2): runs/v### with run.json + qa.json summary; `active` = tool.active_run."""
    store = _objects()
    active = store.get(name)["tool"].get("active_run")
    out = []
    rd = store.dir(name) / "runs"
    # origin of the Experiments versions (2026-10-16): is their base still the current result of its stage?
    cur_stage = dict((store.get(name)["tool"].get("stage_runs") or {}))
    for d in sorted(rd.glob("v[0-9][0-9][0-9]")) if rd.is_dir() else []:
        try:
            rj = json.loads((d / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue   # unfinished version (no run.json)
        qa = {}
        if (d / "qa.json").exists():
            try:
                qa = json.loads((d / "qa.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                qa = {}
        flags = qa.get("flags") or {}
        zones = {}
        for zid, st in (rj.get("zones") or {}).items():   # "Szczegóły" of the version: engines and time per frame
            if not isinstance(st, dict):
                continue
            fr, sec = st.get("frames") or 0, st.get("seconds")
            zones[zid] = {k: st.get(k) for k in ("engine", "preset", "frames", "seconds", "computed", "cached", "engine_share", "cleanplate")}
            zones[zid]["s_per_frame"] = round(float(sec) / fr, 2) if sec is not None and fr else None
        out.append({"run": d.name, "edge_mode": rj.get("edge_mode"), "seconds": rj.get("seconds"),
                    "created_at": rj.get("created_at"), "smoothing": rj.get("smoothing"), "band_px": rj.get("band_px"),
                    "gate_ok": qa.get("gate_ok") if qa else None, "flicker_median": qa.get("flicker_median"),
                    "n_flags": sum(1 for v in flags.values() if v),   # frames with at least one flag
                    "edge_corrected": (qa.get("edge") or {}).get("corrected"), "active": d.name == active,
                    "zones": zones, "applied_from": rj.get("applied_from"), "applied_zone": rj.get("applied_zone"),
                    "stage": rj.get("stage"),         # matanyone / vitmatte / core (stage 3 / 4 result, base 1+2) or None
                    "base": rj.get("base"),           # vitmatte: core | matanyone
                    "motion_blur": rj.get("motion_blur"),   # „Motion blur alfy” (2026-10-16): strength, base_run, longest_px
                    "base_run": rj.get("base_run"), "base_stage": rj.get("base_stage"),   # Experiments: made from this version
                    "base_outdated": bool(rj.get("base_run") and rj.get("base_stage") in ("matanyone", "videomama", "vitmatte", "ltx", "core")
                                          and rj.get("base_run") not in cur_stage.values()),
                    "ltx_mask_run": (rj.get("ltx") or {}).get("mask_source")
                    if re.fullmatch(r"v\d{3,}", str((rj.get("ltx") or {}).get("mask_source", ""))) else None})
    return {"object": name, "active_run": active, "runs": out}


class ActiveRunRequest(BaseModel):
    run: str


@app.put("/api/object/{name}/active_run")
@object_edit
def set_active_run(name: str, req: ActiveRunRequest):
    """'Ustaw jako aktywną' (Z2): export and dependent objects use this version; final_valid from its qa.json."""
    store = _objects()
    d = store.dir(name) / "runs" / req.run
    if not (d / "run.json").exists():
        raise HTTPException(status_code=404, detail=f"Brak wersji {req.run}")
    qp = d / "qa.json"
    qa = json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else None
    store.update(name, tool={"active_run": req.run, "final_valid": qa is not None,
                             "status": ("QA OK" if qa.get("gate_ok") else "QA: flagi") if qa else "brak QA"})
    from engine.export import mark_dependents
    deps = mark_dependents(store, name)
    return {"success": True, "active_run": req.run, "final_valid": qa is not None, "dependents_to_recompute": deps}


@app.get("/api/object/{name}/qa")
def object_qa(name: str, v: Optional[str] = None):
    store = _objects()
    run = v or store.get(name)["tool"].get("active_run")
    p = store.dir(name) / "runs" / (run or "") / "qa.json"
    if not run or not p.exists():
        return {}   # no version yet (Z17: 200 {} instead of a 404 in the console)
    return {**json.loads(p.read_text(encoding="utf-8")), "run": run}


class ObjectSettingsRequest(BaseModel):
    band_scale: Optional[float] = None
    shift_px: Optional[float] = None
    temporal_smoothing: Optional[str] = None
    thin_lens: Optional[str] = None
    use_backlit: Optional[bool] = None
    lupa_box: Optional[List[float]] = None
    depth_order: Optional[int] = None
    flags: Optional[List[str]] = None
    part_of: Optional[str] = None      # "" = not a part; name of the parent object (e.g. Figure's dress -> "figure")
    part_seam_px: Optional[int] = None  # parent of parts: contact gap between the parts filled in <parent>_full (0..40 px)
    prompts: Optional[List[str]] = None  # SAM3 text prompt(s); changing it does not touch user keys (SPEC 7.1)
    ma_scale_mode: Optional[str] = None  # MatAnyone scale in 'Przelicz finalnie': auto (VRAM probe) | 1.0 | 0.75 | 0.5
    core_stabilize: Optional[bool] = None  # temporal stabilisation of the SAM3 core before the edge mode / zones (engine/corestab)
    core_soft_start: Optional[bool] = None  # MatAnyone edge band starts from the SAM3 probabilities (CoreCombine core_mode=soft)
    key_band_same: Optional[bool] = None  # key frames get the same edge band as the other frames (not the spec's +-3 px)
    correction_region_only: Optional[bool] = None  # Z21: recompute after a correction changes the core only in its tracked region
    vitmatte_base: Optional[str] = None  # stage 4 base (2026-10-15): "core" = mask 1+2 (stages 1+2) | "matanyone" = stage 3
    band_out_px: Optional[float] = None  # px mode of the edge band (2026-10-16): at least this far out of the mask 1+2 edge
    band_in_px: Optional[float] = None   # ... and into it; the adaptive width wins where larger; no shift of the mask


@app.put("/api/object/{name}/settings")
@object_edit
def object_settings(name: str, req: ObjectSettingsRequest):
    """Object panel options (SPEC 1.2); any change invalidates the final result (new version on recompute)."""
    upd: Dict[str, Any] = {}
    if req.vitmatte_base is not None:   # only which base stage 4 lays its zones on: no result is invalidated
        from engine.stages import VITMATTE_BASES
        if req.vitmatte_base not in VITMATTE_BASES:
            raise HTTPException(status_code=400, detail=f"baza ViTMatte: {' | '.join(VITMATTE_BASES)}")
        if len([v for v in req.model_dump().values() if v is not None]) == 1:
            return {"success": True, "object": _objects().update(name, vitmatte_base=req.vitmatte_base)}
        upd["vitmatte_base"] = req.vitmatte_base
    from engine.edge import clamp_band_scale, clamp_shift_px
    if req.band_scale is not None:   # any typed value (sanity guards only; was 0.5–2 / ±3 px before 2026-10-16)
        upd["band_scale"] = clamp_band_scale(req.band_scale)
    if req.shift_px is not None:
        upd["shift_px"] = clamp_shift_px(req.shift_px)
    from engine.edge import clamp_band_px
    if req.band_out_px is not None:
        upd["band_out_px"] = clamp_band_px(req.band_out_px)
    if req.band_in_px is not None:
        upd["band_in_px"] = clamp_band_px(req.band_in_px)
    if req.temporal_smoothing is not None:
        if req.temporal_smoothing not in ("weak", "normal", "strong"):
            raise HTTPException(status_code=400, detail="wygładzanie: weak | normal | strong")
        upd["temporal_smoothing"] = req.temporal_smoothing
    if req.thin_lens is not None:
        if req.thin_lens not in ("auto", "off", "box"):
            raise HTTPException(status_code=400, detail="lupa: auto | off | box")
        upd["thin_lens"] = req.thin_lens
    if req.use_backlit is not None:
        upd["use_backlit"] = bool(req.use_backlit)
    if req.ma_scale_mode is not None:
        from engine.refine import MA_SCALE_MODES
        m = str(req.ma_scale_mode).strip()
        m = {"1": "1.0", "0.50": "0.5"}.get(m, m)
        if m not in MA_SCALE_MODES:
            raise HTTPException(status_code=400, detail="skala MatAnyone: auto | 1.0 | 0.75 | 0.5")
        upd["ma_scale_mode"] = m
    if req.core_stabilize is not None:
        upd["core_stabilize"] = bool(req.core_stabilize)
    if req.core_soft_start is not None:
        upd["core_soft_start"] = bool(req.core_soft_start)
    if req.key_band_same is not None:
        upd["key_band_same"] = bool(req.key_band_same)
    if req.correction_region_only is not None:
        upd["correction_region_only"] = bool(req.correction_region_only)
    if req.depth_order is not None:
        upd["depth_order"] = int(req.depth_order)
    if req.flags is not None:
        upd["flags"] = list(req.flags)
    store = _objects()
    if req.prompts is not None:
        det = dict(store.get(name).get("detect") or {})
        det["prompts"] = [p.strip() for p in req.prompts if p and p.strip()]
        det.setdefault("type", "text")
        if len(upd) == 0 and req.lupa_box is None and req.part_of is None:
            return {"success": True, "object": store.update(name, detect=det)}
        upd["detect"] = det
    if req.part_seam_px is not None:   # export-level (the whole of the parts): no recompute of the masks needed
        from engine.export import SEAM_PX_MAX
        seam = max(0, min(SEAM_PX_MAX, int(req.part_seam_px)))
        if len(upd) == 0 and req.lupa_box is None and req.part_of is None and req.prompts is None:
            return {"success": True, "object": store.update(name, part_seam_px=seam)}
        upd["part_seam_px"] = seam
    if req.part_of is not None:   # export-level relation: no recompute of the masks needed
        if req.part_of:
            if req.part_of == name:
                raise HTTPException(status_code=400, detail="obiekt nie może być częścią samego siebie")
            parent = store.get(req.part_of)
            if parent.get("part_of"):
                raise HTTPException(status_code=400, detail=f"„{req.part_of}” sam jest częścią „{parent['part_of']}” – wskaż obiekt główny")
            if any(o.get("part_of") == name for o in store.list()):
                raise HTTPException(status_code=400, detail=f"„{name}” ma własne części – nie może być częścią innego obiektu")
        if len(upd) == 0 and req.lupa_box is None:
            obj = store.update(name, part_of=req.part_of or None)
            return {"success": True, "object": obj}
        upd["part_of"] = req.part_of or None
    tool: Dict[str, Any] = {"final_valid": False}
    if req.lupa_box is not None:
        tool["lupa_box"] = req.lupa_box
    obj = store.update(name, **upd, tool=tool)
    return {"success": True, "object": obj}


def _export_dir() -> Path:
    """Export folder (Z29): project.json -> export_dir, default <work>/matte_desk/export."""
    proj = STATE.get("shot_project")
    d = (proj.data.get("export_dir") if proj else None) or ""
    return Path(d) if d else Path(STATE["work_dir"]) / "export"


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@app.get("/api/project/export_dir")
def get_export_dir():
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    default = Path(STATE["work_dir"]) / "export"
    cur = _export_dir()
    shot_root = Path(STATE["work_dir"]).parent
    return {"path": str(cur).replace("\\", "/"), "default": str(default).replace("\\", "/"),
            "is_default": cur.resolve() == default.resolve(), "outside_work": not _inside(cur, shot_root)}


class ExportDirRequest(BaseModel):
    path: Optional[str] = None          # None / "" = default
    confirmed: bool = False             # a folder outside the shot's <work> needs the user's confirmation (EXR written there)


@app.put("/api/project/export_dir")
@when_idle("Zmiana katalogu eksportu")
def set_export_dir(req: ExportDirRequest):
    """Z29: export folder stored in project.json (the shot manifest is never written)."""
    proj = STATE.get("shot_project")
    if not proj:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    if req.path:
        p = Path(req.path)
        if not p.is_absolute():
            raise HTTPException(status_code=400, detail="Podaj pełną ścieżkę katalogu")
        if p.exists() and not p.is_dir():
            raise HTTPException(status_code=400, detail="To nie jest katalog")
        if not _inside(p, Path(STATE["work_dir"]).parent) and not req.confirmed:
            raise HTTPException(status_code=400, detail=f"Katalog poza projektem ({p}) – eksport zapisze tam EXR; potwierdź ('confirmed': true)")
        proj.data["export_dir"] = str(p).replace("\\", "/")
    else:
        proj.data.pop("export_dir", None)
    proj.save()
    return get_export_dir()


def _export_format_setting() -> str:
    """project.json -> export_format: auto (default) / png / exr. Never stored in the shot manifest."""
    proj = STATE.get("shot_project")
    return ((proj.data.get("export_format") if proj else None) or "auto").lower()


def _export_format_info(fmt: Optional[str] = None) -> Dict[str, Any]:
    from engine.export import EXPORT_FORMATS, resolve_export_format
    bit = int(STATE["manifest"].get("bit_depth", 16))
    cur = fmt or _export_format_setting()
    return {"format": cur, "resolved": resolve_export_format(cur, bit), "auto_resolves_to": resolve_export_format("auto", bit),
            "bit_depth": bit, "choices": list(EXPORT_FORMATS)}


@app.get("/api/project/export_format")
def get_export_format():
    """Mask export format (decision 2026-10-13): auto = PNG sequence for 8-bit sources, EXR for EXR sources."""
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    return _export_format_info()


class ExportFormatRequest(BaseModel):
    format: str = "auto"                 # auto / png / exr


@app.put("/api/project/export_format")
@when_idle("Zmiana formatu eksportu")
def set_export_format(req: ExportFormatRequest):
    """Export format stored in project.json -> export_format (the shot manifest is never written)."""
    from engine.export import EXPORT_FORMATS
    proj = STATE.get("shot_project")
    if not proj:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    f = (req.format or "auto").lower()
    if f not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Format musi być jednym z: {', '.join(EXPORT_FORMATS)}")
    if f == "auto":
        proj.data.pop("export_format", None)
    else:
        proj.data["export_format"] = f
    proj.save()
    return get_export_format()


@app.get("/api/project/mask_plate_denoise")
def get_mask_plate_denoise():
    """Plate-to-mask denoise (SPEC 4b, stage 0): auto = on for EXR / > 8-bit sources, off for 8-bit; `effective` used by
    the core and the final pass; defaults.json mplate.enabled = false turns it off everywhere."""
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    return _mask_plate_denoise_info()


class MaskPlateDenoiseRequest(BaseModel):
    setting: str = "auto"                # auto / on / off


@app.put("/api/project/mask_plate_denoise")
@when_idle("Zmiana odszumiania plate'u do maskowania")
def set_mask_plate_denoise(req: MaskPlateDenoiseRequest):
    """Stored in project.json -> mask_plate_denoise (never in the shot manifest). Objects whose core was computed with
    the other effective value become stale ('do przeliczenia'); the interval cache hash includes the flag."""
    from engine.mplate import DENOISE_CHOICES, DENOISE_KEY
    from engine.stages import denoise_changed
    proj = STATE.get("shot_project")
    if not proj:
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    v = (req.setting or "auto").lower()
    if v not in DENOISE_CHOICES:
        raise HTTPException(status_code=400, detail=f"Ustawienie musi być jednym z: {', '.join(DENOISE_CHOICES)}")
    before = _mask_plate_denoise()
    if v == "auto":
        proj.data.pop(DENOISE_KEY, None)
    else:
        proj.data[DENOISE_KEY] = v
    proj.save()
    info = _mask_plate_denoise_info()
    store = _objects()
    info["changed"] = info["effective"] != before
    info["stale_objects"] = [o["name"] for o in store.list() if denoise_changed(store, o["name"], info["effective"])]
    return info


@app.post("/api/export/nk")
def export_nk_only():
    """'Tylko .nk' (Z29): the Nuke script for the existing export (outputs.json), without writing EXR again."""
    from engine.nuke_mattes import write_mattes_nk
    from engine.outputs import load_outputs, update_outputs
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    data = load_outputs(STATE["work_dir"])
    if not data.get("objects") and not data.get("mattes_all"):
        raise HTTPException(status_code=409, detail="Brak eksportu – najpierw „Eksport”")
    m = STATE["manifest"]
    w, h = _plate_size()
    fr = data.get("frame_range") or [_shot_frames()[0], _shot_frames()[-1]]
    shot = data.get("shot") or m.get("shot", "shot")
    root = Path(data.get("export_dir") or _export_dir())
    nk = write_mattes_nk(root / f"{shot}_mattes.nk", data, shot, int(fr[0]), int(fr[1]), w, h,
                         channels=data.get("mattes_all_channels") if data.get("mattes_all") else None)
    update_outputs(STATE["work_dir"], "mattes_nk", str(nk).replace("\\", "/"))
    return {"success": True, "nk": str(nk).replace("\\", "/")}


def _active_run_frames(name: str) -> List[int]:
    """Frames of the object's active version (a version computed on a work area covers only it)."""
    from engine.workarea import run_frames
    o = _objects()
    fr = run_frames(o.dir(name), o.get(name)["tool"].get("active_run"))
    return [f for f in _shot_frames() if f in set(fr)] if fr else _shot_frames()


class ExportRequest(BaseModel):
    objects: List[str] = []          # empty = all objects with a final result
    scope: str = "all"               # "all" = the whole shot, "work_area" = the work area (project.json work_area)
    multichannel: bool = True
    with_core: bool = False
    with_depth: bool = False
    format: Optional[str] = None     # auto / png / exr for this export only; None = project.json export_format
    # source of the matte (stage layout 2026-10-15): active = the active version with a valid final result (as before);
    # core = the mask 1+2 (SAM3 core + brush corrections, stage 2); matanyone / vitmatte = the result of stage 3 / 4;
    # version = the version `run` (stage 5 list; default the active one) of the one exported object
    source: str = "active"
    run: Optional[str] = None


EXPORT_SOURCES = ("active", "core", "matanyone", "videomama", "vitmatte", "ltx", "version")
EXPORT_SOURCE_LABEL = {"active": "aktywna wersja", "core": "rdzeń (1+2)", "matanyone": "MatAnyone (etap 3)",
                       "videomama": "VideoMaMa (etap 4)", "vitmatte": "ViTMatte (etap 5)", "ltx": "LTX Alpha-Gen (etap 7)",
                       "version": "wybrana wersja"}
# when an object of a full export / of the context has no result of the chosen source (e.g. a hand without ViTMatte
# zones): the next source down, reported in the warning – a written object chosen by name never falls back
EXPORT_FALLBACK = {"vitmatte": ["vitmatte", "videomama", "matanyone", "active", "core"],
                   "matanyone": ["matanyone", "active", "core"],
                   "videomama": ["videomama", "matanyone", "active", "core"],
                   "ltx": ["ltx", "vitmatte", "videomama", "matanyone", "active", "core"],
                   "core": ["core"], "version": ["version", "active"], "active": ["active"]}


def _export_source_run(name: str, source: str, run: Optional[str] = None, build_core: bool = False):
    """(run | "core" | None, why) of an object for an export source. "core" needs a current core (stage 1); the mask
    1+2 is (re)built by the caller when `build_core` (core_fix stale / missing)."""
    from engine.stages import blocked_for_export, core_fix_state, object_stages, stage_run
    store = _objects()
    tool = store.get(name)["tool"]
    if source == "active":
        if blocked_for_export(store, [name], Path(STATE["work_dir"]), _mask_plate_denoise()):
            return None, "brak aktualnego wyniku finalnego (aktywnej wersji)"
        return tool.get("active_run"), ""
    if source in ("matanyone", "videomama", "vitmatte", "ltx"):
        r, ok, why = stage_run(store, name, source)
        if r and ok:
            return r, ""
        return None, why or {"matanyone": "brak wyniku etapu 3 MatAnyone", "videomama": "brak wyniku etapu 4 VideoMaMa",
                             "vitmatte": "brak wyniku etapu 5 ViTMatte", "ltx": "brak wyniku etapu 7 LTX Alpha-Gen"}[source]
    if source == "version":
        r = run or tool.get("active_run")
        if not r or not (store.dir(name) / "runs" / r / "run.json").exists():
            return None, f"brak wersji {r or ''}".strip()
        return r, ""
    st = {s["id"]: s["state"] for s in object_stages(store, name, Path(STATE["work_dir"]), mplate=_mask_plate_denoise())["stages"]}
    if st["core"] != "done":
        return None, "rdzeń SAM3 nieaktualny albo nie policzony – „▶▶ Przelicz całość” (etap 1)"
    if not build_core and not core_fix_state(store, name)[1]:
        return None, "maska 1+2 nieaktualna – „▶▶ Zastosuj poprawki” (etap 2)"
    return "core", ""


async def _export_by_source(req: "ExportRequest"):
    """Export with a chosen source (2026-10-15): every written object strictly from `req.source` (409 with the reasons),
    the context objects (front subtraction, parts) from the same source with EXPORT_FALLBACK."""
    from engine.export import export_objects, part_groups
    from engine.workarea import missing_detail, need_missing, object_life, run_coverage
    if req.source not in EXPORT_SOURCES:
        raise HTTPException(status_code=400, detail=f"source: {' | '.join(EXPORT_SOURCES)}")
    if req.scope not in ("all", "work_area"):
        raise HTTPException(status_code=400, detail="scope = all | work_area")
    store = _objects()
    every = store.list()
    groups = part_groups(every)
    names = {o["name"] for o in every}
    want, needed = None, set(names)
    if req.objects:
        want, needed = [], set()
        for n in req.objects:
            if n.endswith("_full") and n[:-5] in groups:
                needed.update(groups[n[:-5]])
            elif n in names:
                needed.add(n)
            else:
                raise HTTPException(status_code=404, detail=f"nieznany obiekt / całość: {n}")
            want.append(n)
    one = req.objects[0] if (req.run and len(req.objects) == 1 and req.objects[0] in names) else None
    runs, bad, fell = {}, {}, {}
    for o in every:
        n = o["name"]
        strict = want is not None and n in needed
        chain = [req.source] if strict else EXPORT_FALLBACK[req.source]
        why0 = ""
        for src in chain:
            r, why = _export_source_run(n, src, req.run if n == one else None, build_core=(src == "core" and n in needed))
            if r:
                runs[n] = r
                if src != req.source:
                    fell[n] = EXPORT_SOURCE_LABEL[src]
                break
            why0 = why0 or why
        if n not in runs and n in needed:
            bad[n] = why0
    if want is None:      # full export: objects without any result are left out (like before: no final = not exported)
        needed = {n for n in needed if n in runs}
        bad = {}
    if bad:
        raise HTTPException(status_code=409, detail="Brak wyniku ze źródła „" + EXPORT_SOURCE_LABEL[req.source] + "”: " +
                            "; ".join(f"{n}: {w}" for n, w in sorted(bad.items())))
    # the mask 1+2 of the written objects: rebuilt when stale (stage 2 „Zastosuj poprawki” done on the way)
    from engine.stages import core_fix_state
    pipe = None
    for n in sorted(needed):
        if runs.get(n) == "core" and not core_fix_state(store, n)[1]:
            pipe = pipe or _pipeline()
            JOBS.progress(f"Maska 1+2: {n}")
            from engine.workarea import core_coverage
            cov = core_coverage(store.dir(n)) or []
            fr = [f for f in _shot_frames() if any(a <= f <= b for a, b in cov)]
            await asyncio.get_running_loop().run_in_executor(None, lambda n=n, fr=fr: pipe.build_core_fix(n, fr))
    frames = _work_frames() if req.scope == "work_area" else _shot_frames()
    def owner(n: str) -> str:   # a part resolved to its parent's group version: the version lives in the parent's runs/
        r, par = runs.get(n), store.get(n).get("part_of")
        if r and r != "core" and par and not (store.dir(n) / "runs" / r / "run.json").exists() \
                and (store.dir(par) / "runs" / r / "run.json").exists():
            return par
        return n
    objs = []
    for o in every:
        if o["name"] in runs:
            ow = owner(o["name"])
            objs.append({**o, "tool": {**o["tool"], "active_run": runs[o["name"]]}, **({"_run_owner": ow} if ow != o["name"] else {})})

    def coverage(n: str):
        if runs[n] == "core":
            try:
                return json.loads((store.dir(n) / "core_fix" / "core_fix.json").read_text(encoding="utf-8")).get("coverage") or []
            except (OSError, ValueError):
                return []
        return run_coverage(store.dir(owner(n)), runs[n]) or []
    relevant = set(needed)
    if want is not None:
        back = max((o.get("depth_order", 0) for o in every if o["name"] in needed), default=0)
        relevant |= {o["name"] for o in objs if o.get("depth_order", 0) < back}
    holes = {}
    for n in sorted(relevant & set(runs)):
        miss = need_missing((frames[0], frames[-1]), coverage(n), object_life(store.get(n)))
        if miss:
            holes[n] = miss
    if holes:
        lbl = "całym ujęciu" if req.scope == "all" else f"zakresie pracy F{frames[0]}–{frames[-1]}"
        raise HTTPException(status_code=409, detail=missing_detail(EXPORT_SOURCE_LABEL[req.source], lbl, holes))
    missing_front = []
    if want is not None:
        first = min((o.get("depth_order", 0) for o in every if o["name"] in needed), default=0)
        missing_front = [o["name"] for o in every if o["name"] not in runs and o.get("depth_order", 0) < first]
    m = STATE["manifest"]
    w, h = _plate_size()
    depth = None
    if req.with_depth:
        from engine.outputs import load_outputs
        depth = load_outputs(STATE["work_dir"]).get("depth", {}).get("path")
    from engine.export import EXPORT_FORMATS
    fmt = (req.format or _export_format_setting()).lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Format musi być jednym z: {', '.join(EXPORT_FORMATS)}")
    cb = _ws_progress("export_progress", "*")
    out = await asyncio.get_running_loop().run_in_executor(None, lambda: export_objects(
        Path(STATE["work_dir"]), m.get("shot", "shot"), objs, frames, w, h, int(m.get("bit_depth", 16)),
        req.multichannel and want is None, depth, req.with_core, cb, want, export_dir=_export_dir(), export_format=fmt))
    from engine.contract import validate
    warn = []
    if missing_front:
        warn.append(f"Obiekty z przodu bez wyniku nie zostały odjęte: {', '.join(missing_front)}")
    if fell:
        warn.append("Inne źródło (brak wyniku „" + EXPORT_SOURCE_LABEL[req.source] + "”): " +
                    ", ".join(f"{n} → {s}" for n, s in sorted(fell.items())))
    return {"success": True, "outputs": out, "contract_errors": validate(out), "written": want or "all",
            "warning": " · ".join(warn), "format": out.get("export_format"), "scope": req.scope,
            "frames": [frames[0], frames[-1]], "source": req.source, "runs": {n: runs[n] for n in sorted(needed)}}


@app.post("/api/export")
@long_job("Eksport mattów")
async def export_mattes(req: ExportRequest = ExportRequest()):
    """Export (SPEC 9): full-frame alpha per object (front-to-back subtraction), multi-EXR, outputs.json.
    `objects` empty = everything (+ wholes of objects with parts, + multi-EXR); otherwise only these mattes – object
    names and/or "<parent>_full" (the whole of an object with parts). Mattes are always computed in the context of all
    objects with a final result, so a single export equals that object in a full export.
    `source` / `run` (2026-10-15): mattes from the mask 1+2, the stage-3 / stage-4 result or a chosen version
    (_export_by_source); the default "active" is the export as before."""
    if req.source != "active" or req.run:
        return await _export_by_source(req)
    from engine.export import export_objects, part_groups
    from engine.stages import blocked_for_export
    store = _objects()
    work = Path(STATE["work_dir"])
    dn = _mask_plate_denoise()
    every = store.list()
    groups = part_groups(every)
    want = None
    if req.objects:
        want, needed = [], set()
        names = {o["name"] for o in every}
        for n in req.objects:
            if n.endswith("_full") and n[:-5] in groups:
                needed.update(groups[n[:-5]])
            elif n in names:
                needed.add(n)
            else:
                raise HTTPException(status_code=404, detail=f"nieznany obiekt / całość: {n}")
            want.append(n)
        stale = blocked_for_export(store, sorted(needed), work, dn)
    else:
        stale = blocked_for_export(store, [o["name"] for o in every], work, dn)
    if stale:
        raise HTTPException(status_code=409, detail=f"Najpierw „Przelicz finalnie” (eksport nie bierze rdzenia ani nieaktualnego wyniku): {', '.join(stale)}")
    if req.scope not in ("all", "work_area"):
        raise HTTPException(status_code=400, detail="scope = all | work_area")
    frames = _work_frames() if req.scope == "work_area" else _shot_frames()
    # context: every object with a valid final result (front objects are subtracted, parts split the whole)
    ok = set(o["name"] for o in every) - set(blocked_for_export(store, [o["name"] for o in every], work, dn))
    objs = [o for o in every if o["name"] in ok]
    # every written matte and every object it depends on (front objects, parts of the same whole) must have a final
    # result on all exported frames – never an export with silent holes (409 + the missing intervals)
    from engine.workarea import missing_detail, need_missing, object_life, run_coverage
    if want is None:
        relevant = set(ok)
    else:
        relevant = set(needed)
        back = max((o.get("depth_order", 0) for o in every if o["name"] in needed), default=0)
        relevant |= {o["name"] for o in objs if o.get("depth_order", 0) < back}
    holes = {}
    for n in sorted(relevant & ok):
        miss = need_missing((frames[0], frames[-1]), run_coverage(store.dir(n), store.get(n)["tool"].get("active_run")) or [],
                            object_life(store.get(n)))     # frames outside the object's life = black, counted
        if miss:
            holes[n] = miss
    if holes:
        lbl = "całym ujęciu" if req.scope == "all" else f"zakresie pracy F{frames[0]}–{frames[-1]}"
        raise HTTPException(status_code=409, detail=missing_detail("Wynik finalny", lbl, holes))
    missing_front = []
    if want is not None:
        first = min((o.get("depth_order", 0) for o in every if o["name"] in set(want) | {w[:-5] for w in want}), default=0)
        missing_front = [o["name"] for o in every if o["name"] not in ok and o.get("depth_order", 0) < first]
    m = STATE["manifest"]
    w, h = _plate_size()
    depth = None
    if req.with_depth:
        from engine.outputs import load_outputs
        depth = load_outputs(STATE["work_dir"]).get("depth", {}).get("path")
    bit = int(m.get("bit_depth", 16))
    from engine.export import EXPORT_FORMATS
    fmt = (req.format or _export_format_setting()).lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Format musi być jednym z: {', '.join(EXPORT_FORMATS)}")
    cb = _ws_progress("export_progress", "*")
    out = await asyncio.get_running_loop().run_in_executor(None, lambda: export_objects(
        Path(STATE["work_dir"]), m.get("shot", "shot"), objs, frames, w, h, bit, req.multichannel, depth,
        req.with_core, cb, want, export_dir=_export_dir(), export_format=fmt))
    from engine.contract import validate
    warn = (f"Obiekty z przodu bez aktualnego wyniku nie zostały odjęte: {', '.join(missing_front)}"
            if missing_front else "")
    return {"success": True, "outputs": out, "contract_errors": validate(out), "written": want or "all", "warning": warn,
            "format": out.get("export_format"), "scope": req.scope, "frames": [frames[0], frames[-1]]}


class ObjectExportRequest(BaseModel):
    format: Optional[str] = None     # auto / png / exr; None = project.json export_format
    scope: str = "all"               # all | work_area
    source: str = "active"           # active | core | matanyone | vitmatte | version (ExportRequest)
    run: Optional[str] = None


@app.post("/api/object/{name}/export")
async def export_one(name: str, req: Optional[ObjectExportRequest] = None):
    return await export_mattes(ExportRequest(objects=[name], format=req.format if req else None,
                                             scope=req.scope if req else "all", source=req.source if req else "active",
                                             run=req.run if req else None))


@app.post("/api/object/{name}/recompute_dependents")
@long_job("Przelicz zależne: {name}")
async def recompute_dependents(name: str):
    """'Przelicz zależne' (7.3): objects behind `name` marked 'do przeliczenia' are recomputed (core + final)."""
    store = _objects()
    todo = [o["name"] for o in store.list() if name in o["tool"].get("depends_on", []) and o["tool"].get("status") == "do przeliczenia"]
    done = []
    for o in todo:
        await run_object_core(o)
        await finalize_object(o, FinalizeRequest())
        store.update(o, tool={"depends_on": [d for d in store.get(o)["tool"].get("depends_on", []) if d != name]})
        done.append(o)
    return {"success": True, "recomputed": done}


class LibrarySaveRequest(BaseModel):
    object: str
    project: Optional[str] = None


@app.post("/api/library")
def library_save(req: LibrarySaveRequest):
    from engine.crop import frame_segments, load_crops
    from engine.library import save_to_library
    store = _objects()
    obj = store.get(req.object)
    thumbs = []
    keys = store.keys(req.object)
    for f in sorted(keys)[:3]:
        m = store.key_mask(req.object, f)
        ys, xs = np.where(m)
        if len(xs):
            img = _load_proxy_display(f)
            s = _proxy_scale()
            x0, y0, x1, y1 = int(xs.min() * s), int(ys.min() * s), int(xs.max() * s) + 1, int(ys.max() * s) + 1
            thumbs.append(img[y0:y1, x0:x1])
    project = req.project or STATE["manifest"].get("project") or STATE["manifest"].get("shot", "PROJECT")
    p = save_to_library(obj, project, thumbs)
    return {"success": True, "path": str(p)}


@app.get("/api/library")
def library_list():
    from engine.library import list_library
    return {"items": list_library()}


class LibraryAddRequest(BaseModel):
    item: str
    name: Optional[str] = None


@app.post("/api/library/add")
def library_add(req: LibraryAddRequest):
    from engine.library import add_from_library
    try:
        obj = add_from_library(_objects(), req.item, req.name)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "object": obj}


class ReviewImportRequest(BaseModel):
    set_entry: Dict[str, Any]
    marks: List[Dict[str, Any]]
    radius_px: float = 6.0


@app.post("/api/object/{name}/import_review")
@object_edit
def import_review(name: str, req: ReviewImportRequest):
    """Strokes from 'Przegląd klatek' (r = remove, g = add) -> user keys (core +/- strokes)."""
    from engine.keys import KeyHistory, apply_strokes
    from engine.review import strokes_from_marks
    store = _objects()
    hist = KeyHistory(store, name)
    saved = []
    for f, strokes in strokes_from_marks(req.set_entry, req.marks, req.radius_px).items():
        cp = store.dir(name) / "core" / f"c_{f:04d}.png"
        base = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if base is None:
            continue
        hist.save_key(f, apply_strokes(base > 127, strokes), author="user", kind="brush", strokes=strokes, source="review")
        saved.append(f)
    if saved:
        store.update(name, tool={"final_valid": False, "stale_from": min(saved)})
    return {"success": True, "keys": saved}


QUEUE_STATE: Dict[str, Any] = {"thread": None, "stop": False}


def _queue():
    from engine.queue import JobQueue
    return JobQueue(REPO_DIR / "queue" / "queue.json")


class QueueAddRequest(BaseModel):
    object: str
    manifest_path: Optional[str] = None


@app.post("/api/queue")
def queue_add(req: QueueAddRequest):
    mp = req.manifest_path or STATE.get("manifest_path")
    if not mp:
        raise HTTPException(status_code=400, detail="Brak manifestu")
    return {"success": True, "job": _queue().add(mp, req.object), "report": _queue().report()}


class CleanplateAutoRequest(BaseModel):
    frames: Optional[List[int]] = None     # [first, last] subset (default: whole shot)
    grow_px: int = 40
    max_samples: int = 60


@app.post("/api/project/cleanplate_auto")
@long_job("Clean plate (auto)")
async def cleanplate_auto_build(req: CleanplateAutoRequest = CleanplateAutoRequest()):
    """„Policz z ujęcia (auto)” (docs/PLAN_CLEANPLATE.md): median of the uncovered pixels of the shot (cores of ALL objects
    needed -> 409 with the missing ones), holes by LaMa (Comfy) or OpenCV; registered as plate `cleanplate_auto` in
    project.json (manifest untouched); the stage-0 compare tools and the `clean` zone engine use it."""
    from engine.zones import MissingFront
    frames = _shot_frames()
    if req.frames and len(req.frames) == 2:
        frames = [f for f in frames if req.frames[0] <= f <= req.frames[1]]
    pipe = _pipeline()
    try:
        stats = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pipe.build_auto_cleanplate(frames, req.grow_px, req.max_samples))
    except MissingFront as e:
        raise HTTPException(status_code=409, detail={"detail": "Brak rdzeni obiektów: " + ", ".join(e.missing) +
                                                     " – najpierw „Licz ujęcie” (clean plate liczy się po rdzeniu)",
                                                     "missing_front": e.missing, "queue": e.queue})
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    proj = STATE.get("shot_project")
    if proj is not None:
        proj.set_plate("cleanplate_auto", stats["pattern"])
        STATE["manifest"] = proj.effective_manifest()
        STATE["project"] = proj.to_dict()
    return {"success": True, **stats}


@app.get("/api/project/cleanplate_auto")
def cleanplate_auto_status():
    """Clean plate state for the zone card: source (projekt / auto / brak), stats of the auto plate when built."""
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    from engine.cleanplate_auto import load_cleanplate
    m = STATE.get("manifest") or {}
    plates = m.get("plates") or {}
    src = "projekt" if (plates.get("cleanplate_align") or plates.get("cleanplate")) else "brak"
    cp = load_cleanplate(Path(STATE["work_dir"]) / "cleanplate_auto")
    out: Dict[str, Any] = {"source": src, "auto": None}
    if cp is not None:
        out["auto"] = {k: v for k, v in cp[2].items() if k != "frames_sampled"}
        if src == "brak":
            out["source"] = "auto"
    return out


@app.post("/api/queue/preset/all")
def queue_preset_all():
    """„Policz wszystko” (plan section 6): every object of the open project without a valid final result, front to back
    (depth_order) – core + „Przelicz finalnie” each (the queue worker does both; unchanged cores come from cache)."""
    mp = STATE.get("manifest_path")
    if not mp:
        raise HTTPException(status_code=400, detail="Brak projektu")
    store = _objects()
    q = _queue()
    added = []
    for o in store.list():
        if o["tool"].get("final_valid") and o["tool"].get("active_run"):
            continue
        added.append(q.add(mp, o["name"], "run+finalize")["id"])
    return {"success": True, "added": added, "objects": [o["name"] for o in store.list() if not (o["tool"].get("final_valid") and o["tool"].get("active_run"))],
            "report": q.report()}


@app.get("/api/queue")
def queue_report():
    r = _queue().report()
    r["running"] = bool(QUEUE_STATE["thread"] and QUEUE_STATE["thread"].is_alive())
    return r


def _queue_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """One queued object: open its shot (read-only manifest), core + 'Przelicz finalnie' + QA. Keys untouched."""
    from engine.ingest import ProjectCreator
    from engine.comfy_io import io_mode_for
    from engine.pipeline import ObjectPipeline
    data = ProjectCreator().init_project_from_manifest(job["manifest_path"])
    proj = data["shot_project"]
    m = proj.effective_manifest()
    w, h = proj.plate_size
    frames = list(range(int(m["first"]), int(m["last"]) + 1))
    pipe = ObjectPipeline(proj.work_dir, m["plates"]["beauty"], w, h, STATE.get("comfy_client") or ComfyClient.from_config(),
                          display=_plate_display, frames=frames, io_mode=io_mode_for(m),
                          use_mplate=_mask_plate_denoise_info(proj, m)["effective"])
    pipe.run_core(job["object"], frames)
    _, rep, run_dir = pipe.finalize(job["object"], frames)
    qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    flagged = sorted(int(f) for f, c in qa["colours"].items() if c != "green")
    return {"run": run_dir.name, "qa_status": "ok" if qa["gate_ok"] else "flags", "first_flag": flagged[0] if flagged else None}


@app.post("/api/queue/run")
def queue_run():
    import threading
    if QUEUE_STATE["thread"] and QUEUE_STATE["thread"].is_alive():
        return {"success": True, "already_running": True}
    try:
        jid = JOBS.try_start("queue", "Kolejka na noc", "*")
    except JobBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    QUEUE_STATE["stop"] = False

    def run():
        JOBS.bind(jid)
        try:
            _queue().run(_queue_worker, lambda: QUEUE_STATE["stop"] or JOBS.cancelled())
            JOBS.finish(jid, "cancelled" if JOBS.cancelled() else "done")
        except Exception as e:
            JOBS.finish(jid, "cancelled" if isinstance(e, JobCancelled) else "error", str(e))
    t = threading.Thread(target=run, daemon=True)
    QUEUE_STATE["thread"] = t
    t.start()
    return {"success": True, "eta_s": _queue().eta_s()}


class IntervalEngineRequest(BaseModel):
    frame: int
    engine: str   # sam3 | points


@app.put("/api/object/{name}/interval_engine")
@object_edit
def set_interval_engine(name: str, req: IntervalEngineRequest):
    """Propagation engine of the key interval containing `frame` (7.2 pkt 7): SAM3 or CoTracker3 points."""
    if req.engine not in ("sam3", "points"):
        raise HTTPException(status_code=400, detail="silnik: sam3 | points")
    store = _objects()
    ks = sorted(store.keys(name))
    if not ks:
        raise HTTPException(status_code=400, detail="Brak kluczy")
    before = [k for k in ks if k <= req.frame]
    after = [k for k in ks if k > req.frame]
    a = before[-1] if before else None
    b = after[0] if after else None
    if a is not None and a == req.frame and after:
        b = after[0]
    iname = f"{'start' if a is None else a}-{'end' if b is None else b}"
    eng = dict(store.get(name)["tool"].get("interval_engines", {}))
    eng[iname] = req.engine
    store.update(name, tool={"interval_engines": eng, "final_valid": False})
    return {"success": True, "interval": iname, "engines": eng}


@app.get("/api/objects")
def list_objects():
    """Objects in depth order; `has` = results on disk (core / final alpha of the active version) – the UI shows the masks
    of the OTHER computed objects as extra overlay layers."""
    store = _objects()
    out = []
    for o in store.list():
        od = store.dir(o["name"])
        run = (o.get("tool") or {}).get("active_run")
        o = dict(o)
        o["has"] = {"core": (od / "core" / "core.json").exists(),
                    "final": bool(run) and (od / "runs" / run / "run.json").exists()}
        out.append(o)
    return {"objects": out}


class ObjectsOrderRequest(BaseModel):
    names: List[str]          # all objects, front (closest to the camera) first


@app.put("/api/objects/order")
@when_idle("Zmiana kolejności obiektów")
def set_objects_order(req: ObjectsOrderRequest):
    """Depth order by drag & drop (spec 6 pkt 2, Z3): depth_order = index, atomically, no gaps / duplicates. A part
    (part_of) stays behind its parent. Objects whose set of objects in front changed and that have a final result
    -> 'do przeliczenia' (their mattes subtract a different set of front objects)."""
    store = _objects()
    objs = store.list()
    names = [o["name"] for o in objs]
    if len(set(req.names)) != len(req.names) or sorted(req.names) != sorted(names):
        raise HTTPException(status_code=400, detail=f"Kolejność musi zawierać dokładnie wszystkie obiekty bez powtórzeń: {', '.join(names)}")
    pos = {n: i for i, n in enumerate(req.names)}
    for o in objs:
        par = o.get("part_of")
        if par and par in pos and pos[par] > pos[o["name"]]:
            raise HTTPException(status_code=400, detail=f"Część „{o['name']}” musi być za obiektem „{par}” (rodzic przed częściami)")
    old_front = {o["name"]: {x["name"] for x in objs if x.get("depth_order", 0) < o.get("depth_order", 0)} for o in objs}
    changed = []
    for o in objs:
        n = o["name"]
        new_front = set(req.names[:pos[n]])
        upd: Dict[str, Any] = {}
        tool: Dict[str, Any] = {}
        if new_front != old_front[n] and o["tool"].get("active_run"):
            tool = {"status": "do przeliczenia", "final_valid": False}
            changed.append(n)
        if o.get("depth_order") != pos[n] or tool:
            store.update(n, depth_order=pos[n], tool=tool, **upd)
    return {"success": True, "objects": store.list(), "to_recompute": changed}


@app.post("/api/object")
def create_object(req: ObjectCreateRequest):
    try:
        obj = _objects().create(req.name, req.prompts, req.depth_order, req.flags)
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "object": obj}


def _export_files(work: Path, name: str) -> List[Path]:
    """Exported files of an object from outputs.json (matte / core sequences of `name` and of `<name>_full`); the
    multichannel file of all objects (`mattes_all`) is not one of them."""
    import re as _re
    from engine.outputs import load_outputs
    outs = load_outputs(work)
    files: List[Path] = []
    for e in outs.get("objects", []):
        if e.get("name") not in (name, f"{name}_full"):
            continue
        for key in ("matte", "core"):
            pat = e.get(key)
            if not pat:
                continue
            pp = Path(pat)
            rx = _re.compile("^" + _re.sub(r"#+", lambda m: r"\d{%d,}" % len(m.group(0)), _re.escape(pp.name).replace(r"\#", "#")) + "$")
            if pp.parent.is_dir():
                files += sorted(f for f in pp.parent.iterdir() if f.is_file() and rx.match(f.name))
    return list(dict.fromkeys(files))


def _size_info(files: List[Path]) -> Dict[str, Any]:
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return {"files": len(files), "mb": round(total / 2**20, 1)}


@app.get("/api/object/{name}/delete_info")
def object_delete_info(name: str):
    """What deleting the object removes (for the confirmation dialog): its folder (-> project trash) and, optionally,
    its exported masks (outputs.json)."""
    store = _objects()
    try:
        store.get(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    od = store.dir(name)
    folder = [f for f in od.rglob("*") if f.is_file()]
    exports = _export_files(Path(STATE["work_dir"]), name)
    return {"name": name, "folder": {**_size_info(folder), "path": str(od)},
            "exports": {**_size_info(exports), "dirs": sorted({str(f.parent) for f in exports})},
            "parts": [o["name"] for o in store.list() if o.get("part_of") == name]}


class RenameObjectRequest(BaseModel):
    new_name: str


@app.put("/api/object/{name}/rename")
@when_idle("Zmiana nazwy obiektu")
def rename_object(name: str, req: RenameObjectRequest):
    """Rename an object: folder objects/<name> -> objects/<new_name> and every exact reference (parts, holdouts,
    groups). Only when nothing is being computed. Exported masks keep their old file names."""
    new = req.new_name.strip()
    if new == name:
        return {"success": True, "name": name, "files": 0}
    store = _objects()
    try:
        n = store.rename(name, new)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "name": new, "files": n}


@app.delete("/api/object/{name}")
@object_edit
def delete_object(name: str, exports: int = 0):
    """Delete an object: folder -> <work>/trash/objects/ (recoverable), entry removed from outputs.json, objects behind
    it that used it as a front object get 'do przeliczenia'. exports=1: its exported masks (outputs.json) are moved too,
    to <work>/trash/exports/<name>_<time>/ (also recoverable); without it the exported files on disk stay."""
    import shutil
    import time as _time
    from engine.export import mark_dependents
    from engine.outputs import load_outputs, update_outputs
    store = _objects()
    try:
        store.get(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    work = Path(STATE["work_dir"])
    exp_files = _export_files(work, name) if exports else []
    marked = mark_dependents(store, name)
    trash = store.remove(name)
    exp_trash = None
    if exp_files:
        exp_trash = work / "trash" / "exports" / f"{name}_{_time.strftime('%Y%m%d_%H%M%S')}"
        for f in exp_files:
            dest = exp_trash / f.parent.name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
        for d in {f.parent for f in exp_files}:
            try:
                d.rmdir()          # only when empty
            except OSError:
                pass
    outs = load_outputs(work)
    drop = {name, f"{name}_full"} if exports else {name}
    if any(e.get("name") in drop for e in outs.get("objects", [])):
        update_outputs(work, "objects", [e for e in outs["objects"] if e.get("name") not in drop])
    return {"success": True, "trash": str(trash), "marked_for_recompute": marked,
            "exports_trash": str(exp_trash) if exp_trash else None, "exports_moved": len(exp_files)}


@app.post("/api/object/{name}/detect")
@long_job("Wykryj: {name}", quiet=True)
def detect_object(name: str, req: ObjectDetectRequest):
    """Instance choice (SPEC 5.2 pkt 0): SAM3_Detect on the letterboxed proxy, candidates stored as PNG."""
    from engine.crop import choose_instance
    _require_plates_ok()
    store = _objects()
    obj = store.get(name)
    prompt = req.prompt or (obj.get("detect", {}).get("prompts") or [None])[0]
    s = _proxy_scale()
    try:
        cands = choose_instance(_sam3_runner(), _load_proxy_display(req.frame), prompt=prompt,
                                positive=[(x * s, y * s) for x, y in req.positive] or None,
                                negative=[(x * s, y * s) for x, y in req.negative] or None)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    cdir = store.dir(name) / "candidates"
    if cdir.exists():
        for p in cdir.glob("c_*.png"):
            p.unlink()
    cdir.mkdir(parents=True, exist_ok=True)
    boxes = []
    for i, m in enumerate(cands):
        cv2.imwrite(str(cdir / f"c_{i:02d}.png"), ((m > 0.5) * 255).astype(np.uint8))
        ys, xs = np.where(m > 0.5)
        boxes.append([float(xs.min() / s), float(ys.min() / s), float((xs.max() + 1) / s), float((ys.max() + 1) / s)])
    store.update(name, tool={"candidates_frame": req.frame})
    return {"success": True, "frame": req.frame, "candidates": boxes,
            "auto_choice": 0 if len(cands) == 1 else None}


@app.get("/api/object/{name}/candidate/{index}")
def candidate_thumbnail(name: str, index: int, side: int = 160):
    """Z10: thumbnail of SAM3 candidate `index` – the proxy crop of its bbox with the mask as alpha (BGRA PNG, longer
    side <= `side` px) for the candidate list of stage 1."""
    store = _objects()
    obj = store.get(name)
    m = cv2.imread(str(store.dir(name) / "candidates" / f"c_{int(index):02d}.png"), cv2.IMREAD_GRAYSCALE)
    f = obj["tool"].get("candidates_frame")
    if m is None or f is None:
        raise HTTPException(status_code=404, detail="Brak kandydata – najpierw „Wykryj kandydatów”")
    img = (_load_proxy_display(int(f)) * 255).astype(np.uint8)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(m > 127)
    if not len(xs):
        raise HTTPException(status_code=404, detail="Pusta maska kandydata")
    pad = 8
    x0, x1 = max(0, int(xs.min()) - pad), min(img.shape[1], int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(img.shape[0], int(ys.max()) + 1 + pad)
    crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
    crop[..., 3] = np.where(m[y0:y1, x0:x1] > 127, 255, 70).astype(np.uint8)   # the rest of the crop dimmed, not hidden
    side = max(16, min(512, int(side)))
    sc = min(1.0, side / max(crop.shape[0], crop.shape[1]))
    if sc < 1.0:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * sc)), max(1, round(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    return _png_response(np.ascontiguousarray(crop))


@app.post("/api/object/{name}/choose")
@object_edit
def choose_object_instance(name: str, req: ObjectChooseRequest):
    """Chosen candidate -> automatic key (author=auto) at full plate resolution."""
    store = _objects()
    p = store.dir(name) / "candidates" / f"c_{req.candidate:02d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise HTTPException(status_code=404, detail="Brak kandydata – najpierw wykryj obiekt")
    w, h = _plate_size()
    full = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    key = store.save_key(name, req.frame, full, author="auto", kind="mask")
    store.update(name, tool={"start_frame": req.frame, "status": "wybrano instancję"})
    return {"success": True, "key": key}


@app.post("/api/object/{name}/crops")
@long_job("Kadrowanie: {name}")
async def compute_crops(name: str):
    return await asyncio.get_running_loop().run_in_executor(None, lambda: _compute_crops(name))


def _compute_crops(name: str) -> Dict[str, Any]:
    """Crop segments of the object from its keys (stage Kadruj; automatic in „Przelicz całość”)."""
    from engine.crop import run_crop_engine, save_crops
    _require_plates_ok()
    store = _objects()
    obj = store.get(name)
    keys = store.keys(name)
    start = obj["tool"].get("start_frame")
    if start is None or start not in keys:
        raise HTTPException(status_code=400, detail="Najpierw wybierz instancję obiektu (klucz startowy)")
    manifest = STATE["manifest"]
    frames = list(range(int(manifest.get("first", 1)), int(manifest.get("last", 1)) + 1))
    w, h = _plate_size()
    s = _proxy_scale()
    to_proxy = lambda m: cv2.resize(m.astype(np.float32), (max(1, round(w * s)), max(1, round(h * s))),
                                    interpolation=cv2.INTER_AREA)
    key_masks = {f: to_proxy(store.key_mask(name, f)) > 0.5 for f, k in keys.items() if k.get("author") == "user"}
    defaults = json.loads((REPO_DIR / "defaults.json").read_text(encoding="utf-8"))
    try:
        res = run_crop_engine(
            _sam3_runner(), _load_proxy_display, frames, start, to_proxy(store.key_mask(name, start)), s, w, h,
            cfg=defaults.get("crop", {}), thin_parts="thin_parts" in obj.get("flags", []),
            key_masks_proxy=key_masks, for_ltx=obj.get("edge_mode") == "ltx")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    save_crops(store.dir(name) / "crops.json", res.segments,
               {"flags": {str(k): v for k, v in res.flags.items()},
                "bboxes": {str(k): v for k, v in res.bboxes.items()}})
    store.update(name, tool={"status": "kadrowanie gotowe"})
    return {"success": True, "segments": [x.to_dict() for x in res.segments],
            "flags": {str(k): v for k, v in res.flags.items()}}


@app.get("/api/object/{name}/crops")
def get_crops(name: str):
    p = _objects().dir(name) / "crops.json"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, **json.loads(p.read_text(encoding="utf-8"))}


@app.put("/api/object/{name}/crops/{seg_id}")
@object_edit
def override_crop(name: str, seg_id: str, req: CropOverrideRequest):
    """Manual frame of a segment (SPEC 5.2 pkt 8): stored with manual=true, snapped square, padding recomputed."""
    from engine.crop import load_crops, save_crops, square_crop
    p = _objects().dir(name) / "crops.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Brak crops.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    segs = load_crops(p)
    w, h = _plate_size()
    x, y, side = req.crop[0], req.crop[1], req.crop[2]
    for sgm in segs:
        if sgm.seg_id == seg_id:
            crop, pad = square_crop(x + side / 2, y + side / 2, side, w, h, pad_ratio=1.0, margin=0.0, snap=16)
            sgm.crop, sgm.pad, sgm.manual = crop, pad, True
            sgm.scale_to_1008 = round(1008 / crop[2], 4)
            break
    else:
        raise HTTPException(status_code=404, detail=f"Brak odcinka {seg_id}")
    save_crops(p, segs, {k: v for k, v in data.items() if k not in ("schema", "segments")})
    return {"success": True, "segments": [s.to_dict() for s in segs]}


# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>MatteDesk</h1><p>Frontend static files loading...</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)


# ----------------------------------------------------------------------------- Z26: plate QA + fix (stage 0)
class PlateQARequest(BaseModel):
    plate: str = "beauty"
    frames: Optional[List[int]] = None     # [first, last]
    source: bool = False                   # plate switched to a MatteDesk `_fix`: check its source again (change the repair)


class PlateDecisionsRequest(BaseModel):
    decisions: Dict[str, str] = {}         # frame -> mean | flow | gain | keep ("" removes the decision)


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


class PlateFixRequest(BaseModel):
    plate: str = "beauty"
    decisions: Dict[str, str] = {}         # frame -> mean | flow | gain | keep
    confirmed: bool = False                # EXR / GBs are written only after the user confirmed (plan with sizes first)


@app.post("/api/plate/qa")
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
    thr = json.loads((REPO_DIR / "defaults.json").read_text(encoding="utf-8")).get("plate_qa") or {}
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


@app.get("/api/plate/qa/{plate}")
def plate_qa_report(plate: str):
    from engine.plate_qa import load_report
    if not STATE.get("work_dir"):
        raise HTTPException(status_code=400, detail="Brak otwartego projektu")
    rep = load_report(Path(STATE["work_dir"]), plate)
    if rep is None:
        return {"exists": False, "plate": plate}
    rep["frames"] = {str(k): v for k, v in rep["frames"].items()}
    return {"exists": True, **rep, "gate": _plate_gate()["plates"].get(plate), "fix": _plate_fix_rec(plate)}


@app.put("/api/plate/qa/{plate}/decisions")
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


@app.get("/api/plate/qa/{plate}/preview/{frame}")
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


@app.post("/api/plate/fix")
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
        STATE["display_cache"] = {}
        from engine.stages import plate_stale_objects
        stale = plate_stale_objects(_objects(), Path(STATE["work_dir"]), req.plate)
    # the decisions sent (CLI / UI) become the report's decisions of that source
    rp = load_report(Path(STATE["work_dir"]), req.plate)
    if rp is not None and _norm(rp.get("pattern")) == _norm(pattern):
        flagged = {int(f) for f in rp.get("flagged") or []}
        save_decisions(Path(STATE["work_dir"]), req.plate, {str(f): v for f, v in decisions.items() if f in flagged})
    return {"success": True, "plate": req.plate, **{k: v for k, v in res.items() if k != "repaired"},
            "repaired": {str(k): v for k, v in res["repaired"].items()}, "stale_objects": stale}
