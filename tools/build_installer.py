# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the Windows installer:  python tools/build_installer.py  ->  dist/MatteDesk-<version>-setup.exe

1. Code: the public copy (`tools/export_public.py` – only committed files, no shot data / plans, neutral names)
   into build/installer/stage. Commit first: uncommitted changes are not in the installer.
2. Python: python-build-standalone CPython 3.11 for Windows x64 ("install_only": full Python with tkinter, which the
   folder picker needs; the python.org embeddable zip has no tkinter) -> stage/python, then
   `pip install -r requirements.txt` into it. The user's machine never runs pip or winget for MatteDesk.
3. bin/7zr.exe (7-Zip standalone, LGPL) – unpacks the ComfyUI portable .7z when the user has no ComfyUI.
4. ISCC (Inno Setup 6) compiles installer/MatteDesk.iss with the version and the current ComfyUI portable size.

Downloads go to build/cache (resumed, SHA-256 checked: GitHub `digest` for Python, pinned hash for 7zr.exe).
build/ and dist/ are not in git.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engine import download as dl  # noqa: E402

BUILD = REPO / "build" / "installer"
DIST = REPO / "dist"
PY_REPO = "astral-sh/python-build-standalone"
PY_ASSET = re.compile(r"^cpython-3\.11\.\d+\+\d+-x86_64-pc-windows-msvc-install_only\.tar\.gz$")
SEVENZR_URL = "https://www.7-zip.org/a/7zr.exe"
SEVENZR_SHA256 = "256feca8e274e5da655e2a284fabafd9f554365eb164862089dacd4e8276d282"   # 7zr.exe 26.x, pinned 2026-10-17


def _api(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": dl.UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def python_asset(api_get: Callable[[str], Dict[str, Any]] = _api) -> Dict[str, Any]:
    """Newest python-build-standalone CPython 3.11 install_only build for Windows x64 (+ its SHA-256)."""
    rel = api_get(f"https://api.github.com/repos/{PY_REPO}/releases/latest")
    for a in rel.get("assets") or []:
        if PY_ASSET.match(a.get("name", "")):
            dig = str(a.get("digest") or "")
            return {"name": a["name"], "url": a["browser_download_url"], "size": int(a.get("size") or 0),
                    "sha256": dig.split(":", 1)[1] if dig.startswith("sha256:") else None}
    raise RuntimeError(f"{PY_REPO} {rel.get('tag_name')}: brak CPython 3.11 install_only dla Windows x64")


def find_iscc(env: Optional[dict] = None) -> Optional[str]:
    env = os.environ if env is None else env
    for root in (env.get("LOCALAPPDATA", "") + r"\Programs", env.get("ProgramFiles(x86)"), env.get("ProgramFiles")):
        if root:
            p = Path(root) / "Inno Setup 6" / "ISCC.exe"
            if p.exists():
                return str(p)
    return shutil.which("ISCC")


def iscc_argv(iscc: str, version: str, stage: Path, out: Path, portable_gb: float, portable_disk_gb: float) -> List[str]:
    return [iscc, "/Qp", f"/DAppVersion={version}", f"/DStageDir={stage}", f"/DOutputDir={out}",
            f"/DPortableGB={portable_gb:.1f}", f"/DPortableDiskGB={portable_disk_gb:.1f}",
            str(REPO / "installer" / "MatteDesk.iss")]


def export_code(stage: Path, run: Callable[..., Any] = subprocess.run, changelog: Optional[Path] = None) -> None:
    if stage.exists():
        shutil.rmtree(stage)     # our own build output
    r = run([sys.executable, str(REPO / "tools" / "export_public.py"), str(stage)]
            + ([f"--changelog={changelog}"] if changelog else []))
    if r.returncode != 0:
        raise RuntimeError("export_public.py nie powiódł się")


def bundle_python(stage: Path, cache: Path, run: Callable[..., Any] = subprocess.run,
                  progress: Optional[dl.Progress] = None) -> Path:
    a = python_asset()
    arc = cache / a["name"]
    if not arc.exists():
        print(f"Python: {a['name']} ({a['size'] / 1e6:.0f} MB)")
        dl.download(a["url"], arc, sha256=a["sha256"], progress=progress)
    with tarfile.open(arc) as t:            # archive root = python/
        t.extractall(stage, filter="data")
    py = stage / "python" / "python.exe"
    if not py.exists():
        raise RuntimeError(f"brak {py} po rozpakowaniu")
    req = stage / "requirements.txt"
    r = run([str(py), "-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check",
             "-r", str(req)])
    if r.returncode != 0:
        raise RuntimeError("pip install -r requirements.txt do wbudowanego Pythona nie powiódł się")
    trim_python(stage / "python")
    return py


def trim_python(root: Path) -> int:
    """Drop what a user never needs (debug symbols, C headers / import libs): ~85 MB less. Returns bytes removed."""
    gone = 0
    for p in list(root.rglob("*.pdb")):
        gone += p.stat().st_size
        p.unlink()
    for d in ("include", "libs"):
        if (root / d).is_dir():
            gone += sum(f.stat().st_size for f in (root / d).rglob("*") if f.is_file())
            shutil.rmtree(root / d)
    return gone


def bundle_7zr(stage: Path, cache: Path) -> Path:
    exe = cache / "7zr.exe"
    if not exe.exists():
        dl.download(SEVENZR_URL, exe, sha256=SEVENZR_SHA256, verify_server_hash=False)
    got = dl.sha256_file(exe)
    if SEVENZR_SHA256 is None:
        print(f"7zr.exe sha256 = {got} (wpisz do SEVENZR_SHA256)")
    elif got != SEVENZR_SHA256:
        raise RuntimeError(f"7zr.exe: suma SHA-256 {got} ≠ przypięta {SEVENZR_SHA256}")
    dst = stage / "bin" / "7zr.exe"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, dst)
    return dst


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true", help="only prepare build/installer/stage")
    ap.add_argument("--changelog", type=Path, default=None,
                    help="public CHANGELOG.md (public repo clone) to ship instead of a generated one")
    ap.add_argument("--reuse-python", action="store_true", help="keep the bundled Python of the last build (faster)")
    args = ap.parse_args(argv)
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    stage, cache = BUILD / "stage", REPO / "build" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("UWAGA: niezacommitowane zmiany nie trafią do instalatora:\n" + dirty)
    if args.reuse_python and (stage / "python" / "python.exe").exists():
        keep = BUILD / "python_keep"
        shutil.move(str(stage / "python"), str(keep))
        export_code(stage, changelog=args.changelog)
        shutil.move(str(keep), str(stage / "python"))
        r = subprocess.run([str(stage / "python" / "python.exe"), "-m", "pip", "install", "-q", "--no-warn-script-location",
                            "--disable-pip-version-check", "-r", str(stage / "requirements.txt")])   # new requirements
        if r.returncode != 0:
            raise RuntimeError("pip install -r requirements.txt do wbudowanego Pythona nie powiódł się")
        trim_python(stage / "python")
    else:
        export_code(stage, changelog=args.changelog)
        bundle_python(stage, cache)
    bundle_7zr(stage, cache)
    if args.skip_compile:
        print(f"stage gotowy: {stage}")
        return 0
    iscc = find_iscc()
    if not iscc:
        print("Brak Inno Setup 6 (ISCC.exe): winget install JRSoftware.InnoSetup")
        return 1
    from engine.comfy_portable import UNPACKED_GB, release_info
    try:
        gb = release_info()["size"] / 1e9
    except Exception:
        gb = 2.0
    DIST.mkdir(exist_ok=True)
    r = subprocess.run(iscc_argv(iscc, version, stage, DIST, gb, UNPACKED_GB))
    if r.returncode != 0:
        return r.returncode
    out = DIST / f"MatteDesk-{version}-setup.exe"
    print(f"gotowe: {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
