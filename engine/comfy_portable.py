# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/comfy_portable.py - the official ComfyUI portable package for Windows (user decision 2026-10-17).

Layout of the package (`ComfyUI_windows_portable_nvidia.7z` from Comfy-Org/ComfyUI releases):

    ComfyUI_windows_portable/
      python_embeded/python.exe      <- ComfyUI's own Python (torch + CUDA); pip works through it
      ComfyUI/main.py                <- comfy_dir (custom_nodes/, models/)
      run_nvidia_gpu.bat             <- `python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build`

In MatteDesk it is the git profile (`comfy_flavor: git`, block `comfy_git`: a folder with main.py, started with
`python main.py`) whose Python is `python_embeded` next to the folder – `kind()` tells them apart for the Doctor and the
autostart (`-s ... --windows-standalone-build`, like the package's own .bat). Old configs keep working: nothing new
is stored; the kind is read from the folder.

`install()` = download (resume + SHA-256 from GitHub) -> unpack with 7-Zip (`7zr.exe`, shipped with the installer, or
7z on PATH) -> write the git profile to config.local.json. Used by the Windows installer (`tools/installer_setup.py`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engine import download as dl

REPO = "Comfy-Org/ComfyUI"
ASSET = "ComfyUI_windows_portable_nvidia.7z"
PACKAGE_DIR = "ComfyUI_windows_portable"
UNPACKED_GB = 6.5     # what the unpacked package takes on disk (v0.41: ~6 GB) – shown before the download
PORT = 8188


def embedded_python(comfy_dir: str | Path) -> Optional[Path]:
    """`python_embeded/python.exe` next to a ComfyUI folder (portable package), else None."""
    if not str(comfy_dir or ""):
        return None
    p = Path(comfy_dir).parent / "python_embeded" / "python.exe"
    return p if p.exists() else None


def kind(cfg: Dict[str, Any]) -> str:
    """desktop | git | portable – what the chosen ComfyUI is (portable = git profile with python_embeded)."""
    from engine.config import normalize_flavor
    if normalize_flavor(cfg.get("comfy_flavor")) != "git":
        return "desktop"
    return "portable" if embedded_python(cfg.get("comfy_dir") or "") else "git"


def find_comfy_dir(root: str | Path) -> Optional[Path]:
    """The ComfyUI folder (with main.py) for a folder the user points at: the package root, its parent, or ComfyUI itself."""
    r = Path(root)
    for c in (r, r / "ComfyUI", r / PACKAGE_DIR / "ComfyUI"):
        if (c / "main.py").exists():
            return c
    return None


def _drives() -> List[Path]:
    import os
    import string
    if os.name != "nt":
        return [Path("/")]
    return [Path(f"{c}:/") for c in string.ascii_uppercase if Path(f"{c}:/").exists()]


def scan_roots(env: Optional[Dict[str, str]] = None) -> List[Path]:
    """Where people usually unpack ComfyUI portable / clone ComfyUI: drive roots, home, Desktop, Downloads, Documents."""
    import os
    env = dict(os.environ) if env is None else env
    home = Path(env.get("USERPROFILE") or Path.home())
    return _drives() + [home, home / "Desktop", home / "Downloads", home / "Documents", home / "OneDrive" / "Desktop"]


def find_installed(roots: Optional[List[Path]] = None, depth: int = 2) -> List[Path]:
    """ComfyUI folders (with main.py) of portable packages and git clones found up to `depth` levels below `roots`,
    newest first. Cheap: only folder names are listed, no recursion into big trees."""
    seen: Dict[str, Path] = {}

    def visit(d: Path, level: int) -> None:
        try:
            c = find_comfy_dir(d)
            if c is not None and ((c / "custom_nodes").is_dir()):
                seen.setdefault(str(c.resolve()).lower(), c)
                return
            if level >= depth:
                return
            for sub in d.iterdir():
                if sub.is_dir() and not sub.name.startswith((".", "$")) and sub.name.lower() not in _SKIP:
                    visit(sub, level + 1)
        except (OSError, PermissionError):
            return

    for r in roots if roots is not None else scan_roots():
        visit(Path(r), 0)

    def mtime(c: Path) -> float:
        try:
            return (c / "main.py").stat().st_mtime
        except OSError:
            return 0.0
    return sorted(seen.values(), key=mtime, reverse=True)


_SKIP = {"windows", "program files", "program files (x86)", "programdata", "appdata", "$recycle.bin",
         "system volume information", "node_modules", "site-packages"}


PROBE_PORTS = (8188, 8000, 8001, 8189, 8190)


def _process_on_port(port: int) -> Dict[str, str]:
    """{exe, cmd} of the process listening on `port` (Windows, PowerShell); {} when unknown."""
    import json
    import os
    if os.name != "nt":
        return {}
    ps = (f"$c=Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select -First 1;"
          "if($c){Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\" | "
          "Select @{n='exe';e={$_.ExecutablePath}},@{n='cmd';e={$_.CommandLine}} | ConvertTo-Json -Compress}")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        d = json.loads(out) if out else {}
        return {"exe": d.get("exe") or "", "cmd": d.get("cmd") or ""}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def running_comfy(ports=PROBE_PORTS, get_json: Optional[Callable[[str], Optional[dict]]] = None,
                  proc: Callable[[int], Dict[str, str]] = _process_on_port) -> Optional[Dict[str, Any]]:
    """A ComfyUI answering on localhost: {port, python, comfy_dir (when its command line names main.py by full path)}."""
    import json
    import re
    import urllib.request
    if get_json is None:
        def get_json(url: str) -> Optional[dict]:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (OSError, ValueError):
                return None
    for port in ports:
        st = get_json(f"http://127.0.0.1:{port}/system_stats")
        if not isinstance(st, dict) or "system" not in st:
            continue
        p = proc(port)
        m = re.search(r'"?([^"]*?main\.py)"?', p.get("cmd") or "")
        main = Path(m.group(1).strip()) if m else None
        main = main if main is not None and main.is_absolute() and main.exists() else None
        return {"port": port, "python": p.get("exe") or "", "comfy_dir": str(main.parent) if main else ""}
    return None


def detect(user_dir: str = "", roots: Optional[List[Path]] = None, desktop: Optional[Callable[[], Dict[str, str]]] = None,
           running: Optional[Callable[[], Optional[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """"I already have ComfyUI" – which one: the folder the user gave (any kind), else a ComfyUI running now (its
    port and Python are taken over), else ComfyUI Desktop, else the newest portable / git ComfyUI found on disk.
    {kind: desktop | portable | git | none, comfy_dir, others: [...], port?, python?}"""
    run = (running or running_comfy)()
    if run:
        out = _from_running(run, user_dir, roots)
        if out:
            return out
    if user_dir:
        c = find_comfy_dir(user_dir)
        if c is not None:
            return {"kind": "portable" if embedded_python(c) else "git", "comfy_dir": str(c), "others": []}
    if desktop is None:
        from engine.config import desktop2_paths, desktop_base_path

        def desktop() -> Dict[str, str]:
            d2 = desktop2_paths()
            if d2.get("comfy_dir"):
                return d2
            base = desktop_base_path()
            return {"comfy_dir": base} if base and Path(base).is_dir() else {}
    found = find_installed(roots)
    d = desktop()
    if d.get("comfy_dir"):
        return {"kind": "desktop", "comfy_dir": d["comfy_dir"], "others": [str(c) for c in found]}
    if found:
        c = found[0]
        return {"kind": "portable" if embedded_python(c) else "git", "comfy_dir": str(c),
                "others": [str(x) for x in found[1:]]}
    return {"kind": "none", "comfy_dir": "", "others": []}


def _from_running(run: Dict[str, Any], user_dir: str, roots) -> Optional[Dict[str, Any]]:
    py = run.get("python") or ""
    comfy = find_comfy_dir(user_dir) if user_dir else None
    comfy = comfy or (Path(run["comfy_dir"]) if run.get("comfy_dir") else None)
    if comfy is None and py and Path(py).parent.name.lower() == "python_embeded":
        comfy = find_comfy_dir(Path(py).parent.parent)
    if comfy is None and "comfy-desktop" in py.lower().replace(" ", "-"):
        return {"kind": "desktop", "comfy_dir": "", "port": run["port"], "others": []}
    found = [] if comfy else find_installed(roots)
    comfy = comfy or (found[0] if found else None)
    if comfy is None:
        return None
    kind_ = "portable" if embedded_python(comfy) else "git"
    venv = (Path(comfy) / ".venv").exists()
    return {"kind": kind_, "comfy_dir": str(comfy), "port": run["port"],
            "python": py if (py and kind_ == "git" and not venv) else "",
            "others": [str(x) for x in found[1:]]}


def sevenzip(app_dir: Optional[Path] = None) -> Optional[str]:
    """7-Zip able to unpack .7z: `bin/7zr.exe` shipped with MatteDesk, else 7z / 7za / 7zr on PATH or Program Files."""
    from engine.config import REPO_DIR
    for c in ((app_dir or REPO_DIR) / "bin" / "7zr.exe", Path(r"C:\Program Files\7-Zip\7z.exe")):
        if c.exists():
            return str(c)
    for name in ("7z", "7za", "7zr"):
        w = shutil.which(name)
        if w:
            return w
    return None


def release_info(api_get: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """{url, size, sha256, tag} of the latest official portable package (+ unpacked size estimate)."""
    info = dl.github_asset(REPO, ASSET, api_get=api_get)
    info["unpacked_gb"] = UNPACKED_GB
    return info


def install(dest: str | Path, *, info: Optional[Dict[str, Any]] = None, progress: Optional[dl.Progress] = None,
            run: Callable[..., Any] = subprocess.run, download: Callable[..., Any] = dl.download,
            seven: Optional[str] = None, write_config: bool = True, log: Optional[List[str]] = None) -> Dict[str, Any]:
    """Download + unpack the portable package into `dest` and make it the active ComfyUI. Returns
    {comfy_dir, python, tag}. An existing unpacked package in `dest` is reused (no second download)."""
    log = [] if log is None else log
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    comfy = find_comfy_dir(dest)
    tag = None
    if comfy is None:
        info = info or release_info()
        tag = info.get("tag")
        archive = dest / ASSET
        if not archive.exists():
            log.append(f"pobieranie {ASSET} {tag} ({info['size'] / 1e9:.1f} GB) -> {archive}")
            download(info["url"], archive, sha256=info.get("sha256"), progress=progress)
        seven = seven or sevenzip()
        if not seven:
            raise RuntimeError("Brak 7-Zip do rozpakowania paczki .7z (bin/7zr.exe albo https://www.7-zip.org).")
        log.append(f"rozpakowywanie do {dest} …")
        r = run([seven, "x", "-y", f"-o{dest}", str(archive)])
        if getattr(r, "returncode", 0) != 0:
            raise RuntimeError(f"7-Zip zakończył się błędem ({r.returncode}) – archiwum zostaje w {archive}")
        comfy = find_comfy_dir(dest)
        if comfy is None:
            raise RuntimeError(f"Po rozpakowaniu nie ma ComfyUI/main.py w {dest}")
        archive.unlink(missing_ok=True)
    py = embedded_python(comfy)
    if py is None:
        raise RuntimeError(f"{comfy} nie ma python_embeded obok – to nie jest paczka portable")
    if write_config:
        configure(comfy)
    log.append(f"ComfyUI portable gotowe: {comfy}")
    return {"comfy_dir": str(comfy).replace("\\", "/"), "python": str(py), "tag": tag}


def configure(comfy_dir: str | Path, port: int = PORT, local: Optional[Path] = None, python: str = "") -> Path:
    """Make this ComfyUI folder (portable or from git) the active one: git profile in config.local.json
    (`python` = the interpreter a running git ComfyUI uses, when it is not its own .venv / python_embeded)."""
    from engine.config import set_comfy_flavor
    prof: Dict[str, Any] = {"comfy_dir": str(Path(comfy_dir)).replace("\\", "/"), "comfy_port": int(port)}
    if python:
        prof["comfy_python"] = str(python).replace("\\", "/")
    return set_comfy_flavor("git", prof, local=local)
