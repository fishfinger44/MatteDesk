# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/tools_install.py - system tools the Doctor installs itself (user-triggered only): ffmpeg and Git.

ffmpeg: video I/O (ingest of MP4 / MOV, previews). Git: the Doctor clones ComfyUI node packs from GitHub
(ComfyUI-MatAnyone2, VideoHelperSuite, DepthCrafter, ...) and pip installs from GitHub (git+https: CoTracker3,
MatAnyone2's install.py). Python is installed by start_v….bat (MatteDesk runs on it).

Order of attempts:
1. already there: config `<tool>_path`, PATH, PATH from the registry (installed after this process started), known folders;
2. winget (Windows Package Manager, part of Windows 10 / 11): `winget install --id <id> -e --silent ...`;
3. ffmpeg only: the portable build from gyan.dev unpacked into <repo>/bin/ffmpeg (no installer, no admin rights).
The found path is written to config.local.json (`ffmpeg_path` / `git_path`) and prepended to this process' PATH, so the
next install step (git clone, pip git+https) finds the tool without restarting MatteDesk.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_DIR / "bin"

FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

TOOLS: Dict[str, Dict[str, Any]] = {
    "ffmpeg": {"exe": "ffmpeg.exe", "cfg": "ffmpeg_path", "winget": "Gyan.FFmpeg", "size": "ok. 100 MB",
               "link": "https://www.gyan.dev/ffmpeg/builds/"},
    "git": {"exe": "git.exe", "cfg": "git_path", "winget": "Git.Git", "size": "ok. 60 MB",
            "link": "https://git-scm.com/download/win"},
}


def _known_locations(key: str) -> List[str]:
    la = os.environ.get("LOCALAPPDATA", "")
    if key == "git":
        pf = [d for d in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")) if d]
        out = [str(Path(d) / "Git" / "cmd" / "git.exe") for d in pf]
        return out + ([str(Path(la) / "Programs" / "Git" / "cmd" / "git.exe")] if la else [])
    out = [str(BIN_DIR / "ffmpeg" / "bin" / "ffmpeg.exe")]
    if la:
        out.append(str(Path(la) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
        out += sorted(glob.glob(str(Path(la) / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg*" / "*" / "bin" / "ffmpeg.exe")),
                      reverse=True)
    out.append(r"C:\ffmpeg\bin\ffmpeg.exe")
    return out


def registry_path() -> List[str]:
    """PATH entries from the registry (machine + user) – a tool installed after this process started is there, not in os.environ."""
    try:
        import winreg
    except ImportError:
        return []
    out: List[str] = []
    for root, sub in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(root, sub) as k:
                val, _ = winreg.QueryValueEx(k, "Path")
        except OSError:
            continue
        out += [os.path.expandvars(p) for p in str(val).split(os.pathsep) if p.strip()]
    return out


def use_dir(directory: str) -> None:
    """Prepend a folder to this process' PATH (subprocesses – git clone, pip git+https, ffmpeg – inherit it)."""
    cur = os.environ.get("PATH", "").split(os.pathsep)
    if not any(os.path.normcase(c) == os.path.normcase(directory) for c in cur):
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


def find_tool(key: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Absolute path of ffmpeg / git, or None."""
    t = TOOLS[key]
    p = str((cfg or {}).get(t["cfg"]) or "")
    if p and Path(p).is_file():
        return str(Path(p))
    hit = shutil.which(p or key) or shutil.which(key)
    if hit:
        return hit
    for d in registry_path():
        c = Path(d) / t["exe"]
        if c.is_file():
            return str(c)
    for c in _known_locations(key):
        if Path(c).is_file():
            return c
    return None


def winget_exe() -> Optional[str]:
    hit = shutil.which("winget")
    if hit:
        return hit
    la = os.environ.get("LOCALAPPDATA", "")
    c = Path(la) / "Microsoft" / "WindowsApps" / "winget.exe" if la else None
    return str(c) if c and c.exists() else None


def _step(text: str) -> None:
    from engine.installer import _step as step
    step(text)


def _remember(key: str, path: str, write_local: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
    use_dir(str(Path(path).parent))
    if write_local is None:
        from engine.config import write_local as write_local
    write_local({TOOLS[key]["cfg"]: path})


def _ffmpeg_portable(download: Callable[[str, Path], None], log: List[str]) -> Optional[str]:
    """gyan.dev essentials zip -> <repo>/bin/ffmpeg/{bin,...}; returns ffmpeg.exe or None."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    zpath = BIN_DIR / "ffmpeg-release-essentials.zip"
    _step("pobieranie ffmpeg (gyan.dev)")
    download(FFMPEG_ZIP_URL, zpath)
    _step("rozpakowywanie ffmpeg")
    dst = BIN_DIR / "ffmpeg"
    tmp = BIN_DIR / "ffmpeg_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    exe = next(iter(sorted(tmp.rglob("ffmpeg.exe"))), None)
    if exe is None:
        log.append("w paczce nie ma ffmpeg.exe")
        return None
    root = exe.parent.parent             # ffmpeg-<ver>-essentials_build/{bin,doc,...}
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(root), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    out = dst / "bin" / "ffmpeg.exe"
    log.append(f"ffmpeg rozpakowany do {dst}")
    return str(out) if out.is_file() else None


def install_tool(cfg: Dict[str, Any], key: str, run: Callable[..., Any] = subprocess.run,
                 download: Optional[Callable[[str, Path], None]] = None,
                 winget: Optional[str] = "auto",
                 write_local: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """Doctor button "Zainstaluj" for ffmpeg / Git. `winget`: "auto" = look it up, None = pretend there is none (tests)."""
    if key not in TOOLS:
        raise ValueError(f"nieznane narzędzie: {key}")
    t = TOOLS[key]
    log: List[str] = []
    found = find_tool(key, cfg)
    if found:
        _remember(key, found, write_local)
        log.append(f"{key}: już jest ({found}) – zapisane w config.local.json")
        return {"success": True, "log": log, "path": found}

    wg = winget_exe() if winget == "auto" else winget
    if wg:
        _step(f"winget install {t['winget']} ({t['size']})")
        cmd = [wg, "install", "--id", t["winget"], "-e", "--silent",
               "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"]
        try:
            r = run(cmd, capture_output=True, text=True, timeout=1200)
            code = r.returncode
            out = (r.stdout or "") + (r.stderr or "")
        except Exception as e:   # winget broken / timed out -> fall back below
            code, out = -1, str(e)
        log.append(f"winget install {t['winget']} -> kod {code}")
        if code:
            log.append(out[-600:])
        found = find_tool(key, cfg)
        if found:
            _remember(key, found, write_local)
            log.append(f"{key}: zainstalowany ({found})")
            return {"success": True, "log": log, "path": found}
    else:
        log.append("winget niedostępny na tym komputerze")

    if key == "ffmpeg":
        from engine.installer import _download
        try:
            found = _ffmpeg_portable(download or _download, log)
        except Exception as e:
            log.append(f"pobranie paczki ffmpeg nie powiodło się: {e}")
            found = None
        if found:
            _remember(key, found, write_local)
            return {"success": True, "log": log, "path": found}
    raise RuntimeError(f"Nie udało się zainstalować {key} automatycznie. Zainstaluj ręcznie ({t['link']}), "
                       f"potem „Odśwież test”. Szczegóły: " + " | ".join(log[-3:]))


def prepare_env(cfg: Dict[str, Any]) -> None:
    """Before any install step: put the found git / ffmpeg folders on this process' PATH (git clone, pip git+https)."""
    for key in TOOLS:
        p = find_tool(key, cfg)
        if p:
            use_dir(str(Path(p).parent))
