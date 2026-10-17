# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/comfy_launch.py - start ComfyUI when MatteDesk starts and ComfyUI is not running (user decision 2026-10-14).

Which ComfyUI: the one chosen in the Doctor (`comfy_flavor`):
- desktop  -> a Comfy Desktop 2 install (%APPDATA%/Comfy Desktop/installations.json, code in
              %LOCALAPPDATA%/Comfy-Desktop/ComfyUI-Installs/<name>/ComfyUI): its server is started headless – the
              install's `.venv` python `-s main.py` with Desktop's model-paths yaml, shared input / output folders and
              the install's launchArgs, exactly as Desktop runs it, plus `--listen <comfy_host> --port <comfy_port>`,
              in its own console window. The Desktop 2 app itself only opens its launcher and waits for a click; it
              has no CLI flag or deep link that starts an install (checked in app.asar v1.1.7: deep links are only
              install-update / open-settings; the `autoLaunchOnStartup` setting opens the install window).
              No Desktop 2 install -> the older ComfyUI Desktop app (`comfy_desktop_exe` in config, else
              %LOCALAPPDATA%/Programs/...), which starts its own server (port from its settings, usually 8188),
- git -> ComfyUI from git: `<comfy python> main.py --port <comfy_port>` in `comfy_dir`, in its own console window
              (logs visible, closing the window stops ComfyUI).
`comfy_autostart` (config.local.json, default true) switches the start-up launch off; the Doctor also has a button.
`comfy_cpu_affinity` (config.local.json, optional hex mask, e.g. "FFFFF0FF") pins a ComfyUI started from its own
console window to those logical CPUs (Windows; children started later inherit it).
STATE is polled by the UI (GET /api/comfy/launch): idle | starting | running | failed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE: Dict[str, Any] = {"state": "idle"}
_LOCK = threading.Lock()

HOW_GIT = "ComfyUI z gita"
HOW_PORTABLE = "ComfyUI portable"
HOW_DESKTOP2 = "Comfy Desktop 2 (serwer bez okna aplikacji)"
HOW_DESKTOP = "ComfyUI Desktop"


def comfy_url(cfg: Dict[str, Any]) -> str:
    return f"http://{cfg.get('comfy_host') or '127.0.0.1'}:{int(cfg.get('comfy_port') or 8188)}"


def comfy_alive(cfg: Dict[str, Any], timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(comfy_url(cfg) + "/system_stats",
                                                           headers={"User-Agent": "MatteDesk"}), timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def desktop_exe(cfg: Dict[str, Any]) -> Optional[Path]:
    cands = [cfg.get("comfy_desktop_exe") or ""]
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        cands += [Path(la) / "Programs" / "Comfy Desktop" / "Comfy Desktop.exe",
                  Path(la) / "Programs" / "@comfyorgcomfyui-electron" / "ComfyUI.exe"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _venv_python(comfy_dir: Path) -> Optional[Path]:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (comfy_dir / rel).exists():
            return comfy_dir / rel
    return None


def _norm(p: Any) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def desktop2_install(cfg: Dict[str, Any], appdata: Optional[str] = None,
                     localappdata: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A Comfy Desktop 2 install whose server can be started headless, as Desktop itself starts it:
    {comfy_dir, python, yaml, input_dir, output_dir, launch_args, id}; None when there is none.
    Which install: `comfy_dir` from config when it is a Desktop 2 install, else the most recently launched local one."""
    ad = appdata if appdata is not None else os.environ.get("APPDATA", "")
    la = localappdata if localappdata is not None else os.environ.get("LOCALAPPDATA", "")
    root = Path(ad) / "Comfy Desktop" if ad else None
    settings = (_read_json(root / "settings.json") if root else None) or {}
    installs = (_read_json(root / "installations.json") if root else None) or []
    installs = [i for i in installs if isinstance(i, dict) and i.get("installPath") and i.get("sourceId") != "cloud"
                and i.get("status", "installed") == "installed"]
    installs.sort(key=lambda i: i.get("lastLaunchedAt") or 0, reverse=True)
    cands: List[Tuple[Path, Dict[str, Any]]] = [(Path(i["installPath"]) / "ComfyUI", i) for i in installs]
    inst_root = settings.get("installDir") or (str(Path(la) / "Comfy-Desktop" / "ComfyUI-Installs") if la else "")
    if inst_root and Path(inst_root).is_dir():   # installs missing from installations.json
        cands += [(m.parent, {}) for m in sorted(Path(inst_root).glob("*/ComfyUI/main.py"),
                                                 key=lambda m: m.stat().st_mtime, reverse=True)]
    want = cfg.get("comfy_dir") or ""
    if want:   # the folder chosen in config wins when it is a Desktop 2 install (main.py + its own .venv)
        hit = [c for c in cands if _norm(c[0]) == _norm(want)] or [(Path(want), {})]
        cands = hit + cands
    yamls = sorted((root / "instance-model-paths").glob("*.yaml"), key=lambda y: y.stat().st_mtime, reverse=True) \
        if root and (root / "instance-model-paths").is_dir() else []
    shared = Path(la) / "Comfy-Desktop" / "ComfyUI-Shared" if la else None
    for d, inst in cands:
        py = _venv_python(d)
        if not (py and (d / "main.py").exists()):
            continue
        own = [y for y in yamls if inst.get("id") and y.stem == inst["id"]]
        io: Dict[str, Optional[str]] = {}
        for key, setting in (("input", "inputDir"), ("output", "outputDir")):
            v = settings.get(setting) or (str(shared / key) if shared else "")
            io[key + "_dir"] = v if v and Path(v).is_dir() else None
        try:
            extra = shlex.split(str(inst.get("launchArgs") or ""), posix=False)
        except ValueError:
            extra = []
        return {"comfy_dir": d, "python": py, "yaml": (own or yamls or [None])[0], "launch_args": extra,
                "id": inst.get("id"), **io}
    return None


def _desktop2_argv(cfg: Dict[str, Any], inst: Dict[str, Any]) -> List[str]:
    """Desktop 2's own command line (`python -s main.py <launchArgs> --extra-model-paths-config ...
    --input-directory ... --output-directory ...`) + our host / port (last, so they win over launchArgs)."""
    argv = [str(inst["python"]), "-s", "main.py", *inst.get("launch_args", [])]
    if inst.get("yaml"):
        argv += ["--extra-model-paths-config", str(inst["yaml"])]
    if inst.get("input_dir"):
        argv += ["--input-directory", str(inst["input_dir"])]
    if inst.get("output_dir"):
        argv += ["--output-directory", str(inst["output_dir"])]
    return argv + ["--listen", str(cfg.get("comfy_host") or "127.0.0.1"),
                   "--port", str(int(cfg.get("comfy_port") or 8188))]


def _launch_cmd(cfg: Dict[str, Any]):
    """(argv, cwd, how) for the chosen ComfyUI, or raises RuntimeError with what is missing."""
    flavor = str(cfg.get("comfy_flavor") or "desktop").lower()
    if flavor in ("git", "portable"):   # "portable" = old name of the git variant
        d = Path(cfg.get("comfy_dir") or "")
        if not (d / "main.py").exists():
            raise RuntimeError(f"Nie ma {d / 'main.py'} – ustaw folder ComfyUI z gita w Doktorze.")
        from engine.comfy_portable import embedded_python
        from engine.installer import comfy_python
        port = ["--port", str(int(cfg.get("comfy_port") or 8188))]
        emb = None if cfg.get("comfy_python") else embedded_python(d)
        if emb is not None:   # official portable package: the same flags as its run_nvidia_gpu.bat
            return [str(emb), "-s", "main.py", "--windows-standalone-build"] + port, str(d), HOW_PORTABLE
        py = comfy_python(cfg)
        return [str(py) if py else "python", "main.py"] + port, str(d), HOW_GIT
    inst = desktop2_install(cfg)
    if inst is not None:
        return _desktop2_argv(cfg, inst), str(inst["comfy_dir"]), HOW_DESKTOP2
    exe = desktop_exe(cfg)
    if exe is None:
        raise RuntimeError("Nie znaleziono instalacji Comfy Desktop 2 ani aplikacji ComfyUI Desktop – uruchom ComfyUI "
                           "ręcznie albo ustaw `comfy_dir` / `comfy_desktop_exe` w config.local.json.")
    return [str(exe)], str(exe.parent), HOW_DESKTOP


def _affinity_mask(cfg: Dict[str, Any]) -> int:
    v = cfg.get("comfy_cpu_affinity")
    if v in (None, "", 0):
        return 0
    try:
        return int(v, 16) if isinstance(v, str) else int(v)
    except ValueError:
        return 0


def _set_affinity(proc: Any, mask: int) -> bool:
    """Pin the started process (and the children it starts later) to `mask` (Windows; elsewhere a no-op)."""
    if os.name != "nt" or not mask or getattr(proc, "_handle", None) is None:
        return False
    import ctypes
    return bool(ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask)))


def start_comfy(cfg: Dict[str, Any], wait_s: float = 240.0, popen=subprocess.Popen) -> Dict[str, Any]:
    """Start ComfyUI (if not running) and wait in a background thread until it answers. Returns the state at once."""
    with _LOCK:
        if STATE.get("state") == "starting":
            return dict(STATE)
        if comfy_alive(cfg):
            STATE.clear()
            STATE.update(state="running", how="już działał", since=time.time())
            return dict(STATE)
        try:
            argv, cwd, how = _launch_cmd(cfg)
            console = how != HOW_DESKTOP   # our own `python main.py`: a console window with its log
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if console else \
                getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = popen(argv, cwd=cwd, creationflags=flags, stdout=None if console else subprocess.DEVNULL,
                         stderr=None if console else subprocess.DEVNULL)
            mask = _affinity_mask(cfg) if console else 0
            pinned = bool(mask) and _set_affinity(proc, mask)
        except Exception as e:
            STATE.clear()
            STATE.update(state="failed", error=str(e), since=time.time())
            return dict(STATE)
        STATE.clear()
        STATE.update(state="starting", how=how, since=time.time())
        if mask:
            STATE["affinity"] = f"{mask:X}" if pinned else f"{mask:X} – nie ustawiono"

    def _wait():
        t0 = time.time()
        while time.time() - t0 < wait_s:
            time.sleep(2.0)
            if comfy_alive(cfg):
                STATE.update(state="running", took_s=round(time.time() - t0))
                return
        STATE.update(state="failed", error=f"{STATE.get('how', 'ComfyUI')} nie odpowiada po {wait_s:.0f} s "
                                           f"(sprawdź jego okno; port w config = {cfg.get('comfy_port')}).")
    result = dict(STATE)   # "starting" – taken before the waiting thread can change it
    threading.Thread(target=_wait, daemon=True).start()
    return result


def autostart(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """At MatteDesk start: launch ComfyUI unless switched off or already running."""
    if not cfg.get("comfy_autostart", True):
        return {"state": "idle", "autostart": False}
    return start_comfy(cfg)
