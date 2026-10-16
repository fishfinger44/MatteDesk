# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/config.py - Tool configuration, per machine.

Layers (later wins):
  1. built-in defaults (no machine paths)
  2. `config.json`        – in the repo, shared by all machines (no machine paths)
  3. `config.local.json`  – this machine only (gitignored): comfy_dir, models_dir, comfy_port, ffmpeg_path, ...
  4. autodetect for empty values: ComfyUI Desktop base path from its own settings
     (%APPDATA%\\ComfyUI\\config.json -> "basePath"), models_dir = <comfy_dir>/models.

Paths may be relative: they are resolved against the MatteDesk repo folder (e.g. "../ComfyUI").
`python doctor.py --write-local` writes what it detected into config.local.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

REPO_DIR = Path(__file__).resolve().parent.parent
SHARED = REPO_DIR / "config.json"
LOCAL = REPO_DIR / "config.local.json"

DEFAULTS: Dict[str, Any] = {
    "comfy_flavor": "desktop",
    "comfy_dir": "",
    "comfy_host": "127.0.0.1",
    "comfy_port": 8000,
    "comfy_python": "",
    "comfy_autostart": True,      # start the chosen ComfyUI with MatteDesk when it is not running
    "models_dir": "",
    "ffmpeg_path": "ffmpeg",
    "git_path": "git",            # the Doctor clones node packs from GitHub (engine/tools_install.py)
    "backend_host": "127.0.0.1",
    "backend_port": 8765,
    "sam3_checkpoint": "sam3.1_multiplex_fp16.safetensors",
    "align_device": "auto",       # plate alignment: auto (GPU in ComfyUI's Python when it runs) | gpu | cpu
    "mb_device": "auto",          # motion blur of the alpha (the same GPU worker): auto | gpu | cpu
}

PATH_KEYS = ("comfy_dir", "models_dir", "comfy_python", "ffmpeg_path", "git_path", "depth_workflow", "ltx_alpha_workflow")


def resolve_path(v: str, base: Path = REPO_DIR) -> str:
    """Relative path -> absolute against `base`. A bare program name (e.g. "ffmpeg") stays as is (PATH lookup)."""
    if not v:
        return v
    p = Path(os.path.expandvars(os.path.expanduser(v)))
    if p.is_absolute() or (len(v) > 1 and v[1] == ":"):
        return str(p).replace("\\", "/")
    if "/" not in v and "\\" not in v:
        return v
    return str((base / p).resolve()).replace("\\", "/")


def _read(p: Path) -> Dict[str, Any]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return _migrate(d) if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


# ComfyUI variants: "desktop" (ComfyUI Desktop app) | "git" (ComfyUI from git, `python main.py`). Before 2026-10-15
# the git variant was called "portable" with its profile in `comfy_portable` – old config.local.json files still load.
FLAVORS = ("desktop", "git")
_OLD_FLAVORS = {"portable": "git"}


def normalize_flavor(v: Any) -> str:
    v = str(v or "desktop").lower()
    return _OLD_FLAVORS.get(v, v)


def _migrate(d: Dict[str, Any]) -> Dict[str, Any]:
    """Old key names -> current ones (comfy_flavor "portable" -> "git", `comfy_portable` -> `comfy_git`)."""
    d = dict(d)
    if "comfy_flavor" in d and d["comfy_flavor"] not in ("", None):
        d["comfy_flavor"] = normalize_flavor(d["comfy_flavor"])
    if "comfy_portable" in d:
        old = d.pop("comfy_portable")
        d.setdefault("comfy_git", old)
    return d


def desktop_base_path(appdata: Optional[str] = None) -> Optional[str]:
    """ComfyUI Desktop keeps its base path (models/, custom_nodes/, user/) in %APPDATA%/ComfyUI/config.json."""
    root = appdata if appdata is not None else os.environ.get("APPDATA", "")
    if not root:
        return None
    base = _read(Path(root) / "ComfyUI" / "config.json").get("basePath")
    return str(base) if base else None


def desktop2_paths(appdata: Optional[str] = None, localappdata: Optional[str] = None) -> Dict[str, str]:
    """New ComfyUI Desktop ("Comfy Desktop 2"): code in %LOCALAPPDATA%/Comfy-Desktop/ComfyUI-Installs/<name>/ComfyUI
    (custom_nodes/, user/), models in a shared folder named by `base_path` in
    %APPDATA%/Comfy Desktop/instance-model-paths/*.yaml. Returns {"comfy_dir", "models_dir"} for what was found."""
    out: Dict[str, str] = {}
    la = localappdata if localappdata is not None else os.environ.get("LOCALAPPDATA", "")
    ad = appdata if appdata is not None else os.environ.get("APPDATA", "")
    if la:
        installs = Path(la) / "Comfy-Desktop" / "ComfyUI-Installs"
        found = sorted(installs.glob("*/ComfyUI/main.py"), key=lambda p: p.stat().st_mtime, reverse=True) \
            if installs.is_dir() else []
        if found:
            out["comfy_dir"] = str(found[0].parent).replace("\\", "/")
    if ad:
        yamls = sorted((Path(ad) / "Comfy Desktop" / "instance-model-paths").glob("*.yaml"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for y in yamls:
            try:
                for line in y.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip()
                    if s.startswith("base_path:"):
                        out["models_dir"] = s.split(":", 1)[1].strip().strip("'\"").replace("\\", "/")
                        break
            except OSError:
                continue
            if "models_dir" in out:
                break
    return out


def load_config(shared: Optional[Path] = None, local: Optional[Path] = None, appdata: Optional[str] = None,
                localappdata: Optional[str] = None) -> Dict[str, Any]:
    # module paths resolved at call time (tests redirect LOCAL with monkeypatch)
    if localappdata is None and appdata is not None:
        localappdata = ""   # explicit appdata (tests) -> do not peek at the real %LOCALAPPDATA%
    shared = SHARED if shared is None else shared
    local = LOCAL if local is None else local
    cfg = dict(DEFAULTS)
    sources = {k: "default" for k in cfg}
    for name, p in (("config.json", shared), ("config.local.json", local)):
        for k, v in _read(p).items():
            if v not in ("", None):
                cfg[k] = v
                sources[k] = name
    cfg["comfy_flavor"] = normalize_flavor(cfg.get("comfy_flavor"))
    git = cfg["comfy_flavor"] == "git"
    desktop_models = cfg.get("models_dir") or ""   # top-level keys describe ComfyUI Desktop
    if git:
        # ComfyUI from git: own profile `comfy_git` {comfy_dir, comfy_port, models_dir, comfy_python,
        # shared_models}; the top-level (Desktop) paths must not leak into it.
        prof = cfg.get("comfy_git") if isinstance(cfg.get("comfy_git"), dict) else {}
        cfg.update(comfy_dir="", models_dir="", comfy_python="", comfy_port=8188)
        for k in ("comfy_dir", "models_dir", "comfy_python", "comfy_port"):
            if prof.get(k) not in ("", None):
                cfg[k], sources[k] = prof[k], "config.local.json (comfy_git)"
        if prof.get("shared_models") and not cfg["models_dir"]:
            d2 = desktop2_paths(appdata, localappdata)
            shared_dir = desktop_models or d2.get("models_dir") or ""
            if shared_dir:
                cfg["models_dir"], sources["models_dir"] = shared_dir, "wspólne z ComfyUI Desktop"
    for k in PATH_KEYS:
        if isinstance(cfg.get(k), str):
            cfg[k] = resolve_path(cfg[k])
    if not git and not cfg.get("comfy_dir"):
        base = desktop_base_path(appdata)
        if base:
            cfg["comfy_dir"], sources["comfy_dir"] = base, "autodetect (ComfyUI Desktop)"
    if not git and (not cfg.get("comfy_dir") or not cfg.get("models_dir")):
        d2 = desktop2_paths(appdata, localappdata)
        if not cfg.get("comfy_dir") and d2.get("comfy_dir"):
            cfg["comfy_dir"], sources["comfy_dir"] = d2["comfy_dir"], "autodetect (Comfy Desktop 2)"
        if not cfg.get("models_dir") and d2.get("models_dir"):
            cfg["models_dir"], sources["models_dir"] = d2["models_dir"], "autodetect (Comfy Desktop 2, instance-model-paths)"
    if not cfg.get("models_dir") and cfg.get("comfy_dir"):
        cfg["models_dir"] = str(Path(cfg["comfy_dir"]) / "models").replace("\\", "/")
        sources["models_dir"] = "comfy_dir/models"
    cfg["_sources"] = sources
    return cfg


def write_local(values: Dict[str, Any], local: Optional[Path] = None) -> Path:
    """Merge values into config.local.json (this machine only)."""
    local = LOCAL if local is None else local
    cur = _read(local)
    cur.update({k: v for k, v in values.items() if not k.startswith("_")})
    local.write_text(json.dumps(cur, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return local


PROFILE_KEYS = ("comfy_dir", "models_dir", "comfy_port", "comfy_python")


def write_local_active(values: Dict[str, Any], local: Optional[Path] = None) -> Path:
    """Like write_local, but machine-path keys go to the ACTIVE ComfyUI profile: top level for Desktop, the
    `comfy_git` block for ComfyUI from git (so switching the flavor never overwrites the other one's paths)."""
    local = LOCAL if local is None else local
    cur = _read(local)
    if normalize_flavor(cur.get("comfy_flavor") or _read(SHARED).get("comfy_flavor")) == "git":
        prof = dict(cur.get("comfy_git") or {})
        prof.update({k: v for k, v in values.items() if k in PROFILE_KEYS})
        values = {**{k: v for k, v in values.items() if k not in PROFILE_KEYS}, "comfy_git": prof}
    return write_local(values, local)


def set_comfy_flavor(flavor: str, profile: Optional[Dict[str, Any]] = None, local: Optional[Path] = None) -> Path:
    """Switch between ComfyUI Desktop and ComfyUI from git; `profile` = the git variant's profile (merged)."""
    flavor = normalize_flavor(flavor)
    if flavor not in FLAVORS:
        raise ValueError("flavor: desktop | git")
    local = LOCAL if local is None else local
    vals: Dict[str, Any] = {"comfy_flavor": flavor}
    if profile is not None:
        prof = dict(_read(local).get("comfy_git") or {})
        prof.update(profile)
        vals["comfy_git"] = prof
    return write_local(vals, local)


RECENT_MAX = 5


def _same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def remember_work_dir(work_dir: str, local: Optional[Path] = None, keep: int = RECENT_MAX) -> Dict[str, Any]:
    """Shot folder opened in the tool -> config.local.json: `last_work_dir` + `recent_work_dirs` (newest first,
    no duplicates, at most `keep`). The shot manifest is not touched."""
    local = LOCAL if local is None else local
    wd = str(work_dir).replace("\\", "/").rstrip("/")
    old = _read(local).get("recent_work_dirs") or []
    recent = [wd] + [d for d in old if isinstance(d, str) and d and not _same_dir(d, wd)]
    vals = {"last_work_dir": wd, "recent_work_dirs": recent[:keep]}
    write_local(vals, local)
    return vals
