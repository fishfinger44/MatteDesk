# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/installer.py - "Zainstaluj" buttons of the Doctor (user-triggered only, never automatic).

1. Our own Comfy node packages (not in any registry) -> <comfy_dir>/custom_nodes/:
   ComfyUI-PlateMatte, ComfyUI-MatAnyone2-Keyframes (source: handoff/comfy/custom_nodes/ in this repo).
   Only these two folders are touched. An existing, different copy is first moved to
   <repo>/backups/custom_nodes/<name>_<timestamp>/ (outside custom_nodes, so Comfy does not load it).
   Their requirements.txt is installed into Comfy's Python when it is found (Desktop: <comfy_dir>/.venv).
2. CoTracker3: `pip install` of facebookresearch/co-tracker into MatteDesk's Python + weights
   scaled_offline.pth -> <models_dir>/cotracker/.
After 1. Comfy must be restarted (Desktop: in the app).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_DIR = Path(__file__).resolve().parent.parent
NODE_SRC = REPO_DIR / "handoff" / "comfy" / "custom_nodes"
NODE_PACKAGES = ("ComfyUI-PlateMatte", "ComfyUI-MatAnyone2-Keyframes")
BACKUP_DIR = REPO_DIR / "backups" / "custom_nodes"

COTRACKER_PIP = "git+https://github.com/facebookresearch/co-tracker"
COTRACKER_WEIGHTS_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COTRACKER_REL = Path("cotracker") / "scaled_offline.pth"

_SKIP = {"__pycache__", ".git"}
PIP_BUILD_CONSTRAINTS = Path(__file__).resolve().parent / "pip_build_constraints.txt"

# Live state of the running install (polled by the Doctor UI: GET /api/install/progress)
PROGRESS: Dict[str, Any] = {"active": False}


def progress_begin(what: str) -> None:
    PROGRESS.clear()
    PROGRESS.update(active=True, what=what, step="start", file="", done=0, total=0, started=time.time())


def progress_end(ok: bool, error: str = "") -> None:
    PROGRESS.update(active=False, ok=ok, error=error, step="gotowe" if ok else "błąd")


def _step(text: str) -> None:
    if PROGRESS.get("active"):
        PROGRESS.update(step=text, file="", done=0, total=0)


def _file_bytes_normalized(p: Path) -> bytes:
    """File content for the digest: text files (no NUL byte) with CRLF -> LF, so a copy checked out / saved with
    Windows line endings is not reported as "inna wersja niż w repo" (user request 2026-10-13). Binary files unchanged."""
    data = p.read_bytes()
    return data if b"\0" in data else data.replace(b"\r\n", b"\n")


def _tree_digest(root: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (_SKIP & set(p.relative_to(root).parts)) and p.suffix != ".pyc":
            h.update(str(p.relative_to(root)).replace("\\", "/").encode())
            h.update(_file_bytes_normalized(p))
    return h.hexdigest()


_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)


def package_version(pkg_dir: Path) -> Optional[str]:
    """`__version__` of a node package (its __init__.py); None = missing / older copy without a version."""
    init = Path(pkg_dir) / "__init__.py"
    try:
        m = _VERSION_RE.search(init.read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(1) if m else None


def node_versions(comfy_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """{package: {"installed": version | None, "repo": version | None}} – shown by the Doctor next to the status.
    The install decision itself stays the content digest (node_status): any change in the repo offers the update."""
    cn = Path(comfy_dir or "") / "custom_nodes"
    return {name: {"installed": package_version(cn / name) if comfy_dir else None,
                   "repo": package_version(NODE_SRC / name)} for name in NODE_PACKAGES}


def comfy_python(cfg: Dict[str, Any]) -> Optional[Path]:
    """Python of the Comfy install: config `comfy_python`, else Desktop/venv or portable embedded."""
    if cfg.get("comfy_python"):
        p = Path(cfg["comfy_python"])
        return p if p.exists() else None
    d = Path(cfg.get("comfy_dir") or "")
    if not str(d):
        return None
    for c in (d / ".venv" / "Scripts" / "python.exe", d / ".venv" / "bin" / "python",
              d.parent / "python_embeded" / "python.exe"):
        if c.exists():
            return c
    return None


def node_status(comfy_dir: str) -> Dict[str, str]:
    """{package: missing | same | different | no_comfy}."""
    cn = Path(comfy_dir or "") / "custom_nodes"
    out = {}
    for name in NODE_PACKAGES:
        if not comfy_dir or not cn.is_dir():
            out[name] = "no_comfy"
        elif not (cn / name).exists():
            out[name] = "missing"
        else:
            out[name] = "same" if _tree_digest(cn / name) == _tree_digest(NODE_SRC / name) else "different"
    return out


def _pip_requirements(cfg: Dict[str, Any], name: str, pkg_dir: Path, run: Callable[..., Any], log: List[str]) -> bool:
    """pip install -r <pkg>/requirements.txt into ComfyUI's Python (always, also for an up-to-date package: a node pack
    copied earlier may still miss its dependencies -> "IMPORT FAILED" in Comfy). False = pip failed."""
    req = pkg_dir / "requirements.txt"
    if not (req.exists() and req.read_text(encoding="utf-8", errors="ignore").strip()):
        return True
    py = comfy_python(cfg)
    if py is None:
        log.append(f"{name}: nie znaleziono Pythona ComfyUI – doinstaluj ręcznie: pip install -r {req}")
        return False
    _step(f"pip install ({name})")
    r = run([str(py), "-m", "pip", "install", "-r", str(req)], capture_output=True, text=True)
    log.append(f"{name}: zależności (pip install -r requirements.txt) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        log.append((r.stderr or r.stdout or "")[-800:])
    return r.returncode == 0


def install_nodes(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    comfy_dir = cfg.get("comfy_dir") or ""
    cn = Path(comfy_dir) / "custom_nodes"
    if not comfy_dir or not cn.is_dir():
        raise RuntimeError(f"Nie znaleziono ComfyUI (brak folderu {cn}). Ustaw comfy_dir w config.local.json.")
    log: List[str] = []
    ok = True
    status = node_status(comfy_dir)
    for name in NODE_PACKAGES:
        _step(f"kopiowanie {name}")
        dst = cn / name
        if status[name] == "same":
            log.append(f"{name}: pliki aktualne")
        else:
            if status[name] == "different":
                BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                bak = BACKUP_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(dst), str(bak))
                log.append(f"{name}: poprzednia wersja przeniesiona do {bak}")
            shutil.copytree(NODE_SRC / name, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            log.append(f"{name}: skopiowany do {dst}")
        ok = _pip_requirements(cfg, name, dst, run, log) and ok
    log.append("Zrestartuj ComfyUI (Desktop: w aplikacji), potem „Odśwież test”.")
    return {"success": ok, "log": log, "status": node_status(comfy_dir), "restart_comfy": True}


def _progress_cb(dst: Path) -> Callable[[int, int], None]:
    def progress(done: int, total: int) -> None:
        if PROGRESS.get("active"):
            PROGRESS.update(step="pobieranie", file=dst.name, done=done, total=total)
    return progress


def _download(url: str, dst: Path, progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Resumable download with a SHA-256 check when the server publishes one (Hugging Face LFS) – engine/download.py."""
    from engine.download import download
    download(url, dst, progress=progress or _progress_cb(dst))


# --- model weights and third-party node packs (Doctor buttons) -------------------------------------------------
# key -> (url, path relative to models_dir, size hint for the confirmation dialog)
def model_download(key: str, cfg: Dict[str, Any]) -> tuple:
    sam3 = cfg.get("sam3_checkpoint") or "sam3.1_multiplex_fp16.safetensors"
    table = {
        "sam3": (f"https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/{sam3}",
                 Path("checkpoints") / sam3),
        "matanyone2_weights": ("https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
                               Path("matanyone2") / "matanyone2.pth"),
        "lama": ("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
                 Path("inpaint") / "big-lama.pt"),
    }
    return table[key]


# key -> (git url, folder name in custom_nodes). Third-party packs: cloned as they are (no patching).
GIT_NODE_PACKS = {
    "matanyone2_node": ("https://github.com/spiritform/comfy-matanyone2", "ComfyUI-MatAnyone2"),
    "inpaint_nodes": ("https://github.com/Acly/comfyui-inpaint-nodes", "comfyui-inpaint-nodes"),
    "depthcrafter_nodes": ("https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes", "ComfyUI-DepthCrafter-Nodes"),
    "minimax_nodes": ("https://github.com/1038lab/ComfyUI-MiniMax-Remover", "ComfyUI-MiniMax-Remover"),
    # Comfy I/O "png8" (8-bit sources): VHS_LoadImagesPath
    "vhs_nodes": ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "ComfyUI-VideoHelperSuite"),
    # stage VideoMaMa (2026-10-16): VideoMaMaPipelineLoader + VideoMaMaSampler (diffusers / transformers / accelerate)
    "videomama_node": ("https://github.com/okdalto/ComfyUI-VideoMaMa", "ComfyUI-VideoMaMa"),
}
# Other folder names of the same pack (ComfyUI Manager clones under the registry id) – never clone a second copy
GIT_NODE_ALIASES = {"vhs_nodes": ("comfyui-videohelpersuite",)}

# Multi-file weights: key -> (Hugging Face repo, [files], folder in models_dir)
HF_FOLDERS = {
    "minimax_weights": ("zibojia/minimax-remover",
                        ["scheduler/scheduler_config.json", "transformer/config.json",
                         "transformer/diffusion_pytorch_model.safetensors", "vae/config.json",
                         "vae/diffusion_pytorch_model.safetensors"], "MiniMax-Remover"),
}


def install_model(cfg: Dict[str, Any], key: str, download: Callable[[str, Path], None] = _download) -> Dict[str, Any]:
    models_dir = cfg.get("models_dir") or ""
    if not models_dir:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if key in HF_FOLDERS:
        repo, files, folder = HF_FOLDERS[key]
        log: List[str] = []
        for f in files:
            dst = Path(models_dir) / folder / f
            if dst.exists():
                log.append(f"już jest: {dst}")
                continue
            download(f"https://huggingface.co/{repo}/resolve/main/{f}", dst)
            log.append(f"pobrane: {dst} ({dst.stat().st_size / 1048576:.0f} MB)")
        return {"success": True, "log": log, "weights": str(Path(models_dir) / folder)}
    url, rel = model_download(key, cfg)
    dst = Path(models_dir) / rel
    if dst.exists():
        return {"success": True, "log": [f"już jest: {dst}"], "weights": str(dst)}
    download(url, dst)
    return {"success": True, "log": [f"pobrane: {dst} ({dst.stat().st_size / 1048576:.0f} MB)",
                                     "Odśwież listę modeli w ComfyUI (klawisz R) albo zrestartuj Comfy."], "weights": str(dst)}


def hf_token() -> Optional[str]:
    """The user's Hugging Face login (`hf auth login` -> HF_HOME/token, or HF_TOKEN). Never logged or returned to the UI."""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    try:
        t = (home / "token").read_text(encoding="utf-8").strip()
        return t or None
    except OSError:
        return None


def _download_hf(url: str, dst: Path, token: Optional[str], progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Download with the HF login (gated repos), resume (`.part` + Range – 20 GB files must not start from zero) and
    the SHA-256 Hugging Face publishes for LFS files."""
    from engine.download import download
    download(url, dst, token=token, progress=progress or _progress_cb(dst))


def install_ltx(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Stage „LTX-2.5 Alpha-Gen”: the five LTX-2.5 files (~40 GB) into models_dir, from the gated Lightricks repos with the
    user's HF login. Present files are skipped; interrupted downloads resume."""
    from engine.ltx_alpha import HF_PATH, LTX_MODELS
    models_dir = cfg.get("models_dir") or ""
    if not models_dir:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    token = hf_token()
    if download is None:
        if not token:
            raise RuntimeError("Brak logowania do Hugging Face: modele LTX-2.5 są „gated”. Zaloguj się („hf auth login”) "
                               "i zaakceptuj licencję LTX-2.x na stronach Lightricks/LTX-2.5 i "
                               "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen.")
        download = lambda url, dst: _download_hf(url, dst, token)
    log: List[str] = []
    for key, (folder, fn, repo, gb) in LTX_MODELS.items():
        if keys and key not in keys:
            continue
        dst = Path(models_dir) / folder / fn
        if dst.exists() and dst.stat().st_size > 0.9 * gb * 1e9:
            log.append(f"już jest: {dst}")
            continue
        _step(f"LTX-2.5: {fn} ({gb:.1f} GB)")
        download(f"https://huggingface.co/{repo}/resolve/main/{HF_PATH[key]}{fn}", dst)
        log.append(f"pobrane: {dst} ({dst.stat().st_size / 2 ** 30:.1f} GB)")
    log.append("Odśwież listę modeli w ComfyUI (klawisz R) albo zrestartuj Comfy.")
    return {"success": True, "log": log}


def install_videomama(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                      run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """Stage „VideoMaMa”: the ComfyUI-VideoMaMa node pack (git clone + requirements) and the weights it loads – only the
    needed parts of SVD-xt + the VideoMaMa UNet (7.1 GB, public repos) into <models_dir>/videomama. Present files are
    skipped; interrupted downloads resume."""
    from engine.videomama import VMM_FILES, models_root
    models_dir = cfg.get("models_dir") or ""
    if not models_dir:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    cn = Path(cfg.get("comfy_dir") or "") / "custom_nodes"
    had_node = (cn / GIT_NODE_PACKS["videomama_node"][1]).is_dir()
    res = install_git_node(cfg, "videomama_node", run=run)
    log: List[str] = list(res.get("log") or [])
    if not res.get("success"):
        return {"success": False, "log": log, "restart_comfy": not had_node}
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    root = models_root(Path(models_dir))
    for sub, repo, fn, gb in VMM_FILES:
        dst = root / sub / fn
        if dst.exists() and dst.stat().st_size > (0.9 * gb * 1e9 if gb else 0):
            continue
        _step(f"VideoMaMa: {sub}/{fn}" + (f" ({gb:.1f} GB)" if gb else ""))
        download(f"https://huggingface.co/{repo}/resolve/main/{fn}", dst)
        log.append(f"pobrane: {dst}" + (f" ({dst.stat().st_size / 2 ** 30:.1f} GB)" if gb else ""))
    log.append("Wagi VideoMaMa: OK." + (" Zrestartuj ComfyUI (nowy węzeł), potem „Odśwież test”." if not had_node else ""))
    return {"success": True, "log": log, "restart_comfy": not had_node}


def install_git_node(cfg: Dict[str, Any], key: str, run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    comfy_dir = cfg.get("comfy_dir") or ""
    cn = Path(comfy_dir) / "custom_nodes"
    if not comfy_dir or not cn.is_dir():
        raise RuntimeError(f"Nie znaleziono ComfyUI (brak folderu {cn}). Ustaw comfy_dir w config.local.json.")
    from engine.tools_install import find_tool
    git = find_tool("git", cfg)
    if not git:
        raise RuntimeError("Brak Gita – kliknij „Zainstaluj” przy pozycji „Git” w Doktorze (albo winget install Git.Git), potem spróbuj ponownie.")
    url, name = GIT_NODE_PACKS[key]
    dst = next((cn / n for n in GIT_NODE_ALIASES.get(key, ()) if (cn / n).is_dir()), cn / name)
    name = dst.name
    log: List[str] = []
    if dst.exists():
        log.append(f"{name}: już jest ({dst}) – sprawdzam zależności")
    else:
        _step(f"git clone {name}")
        r = run([git, "clone", "--depth", "1", url, str(dst)], capture_output=True, text=True)
        log.append(f"git clone {url} -> kod {r.returncode}")
        if r.returncode:
            log.append((r.stderr or r.stdout or "")[-800:])
            return {"success": False, "log": log}
    ok = _pip_requirements(cfg, name, dst, run, log)
    ok = _run_install_py(cfg, name, dst, run, log) and ok
    log.append("Zrestartuj ComfyUI (Desktop: w aplikacji), potem „Odśwież test”.")
    return {"success": ok, "log": log, "restart_comfy": True}


def _run_install_py(cfg: Dict[str, Any], name: str, pkg_dir: Path, run: Callable[..., Any], log: List[str]) -> bool:
    """<pack>/install.py with ComfyUI's Python (ComfyUI Manager convention; e.g. ComfyUI-MatAnyone2 installs the
    matanyone2 package, hydra-core and thinplate there – they are not in its requirements.txt). False = it failed."""
    script = pkg_dir / "install.py"
    if not script.exists():
        return True
    py = comfy_python(cfg)
    if py is None:
        log.append(f"{name}: nie znaleziono Pythona ComfyUI – uruchom ręcznie: <python ComfyUI> {script}")
        return False
    _step(f"install.py ({name})")
    # its pip calls inherit the environment: build constraints (pip_build_constraints.txt) apply to their builds
    env = {**os.environ, "PIP_BUILD_CONSTRAINT": str(PIP_BUILD_CONSTRAINTS)}
    r = run([str(py), str(script)], capture_output=True, text=True, cwd=str(pkg_dir), env=env)
    log.append(f"{name}: install.py -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        cause = [l.strip() for l in out.splitlines() if re.search(r"\w*Error\b|^\s*error:", l)
                 and not l.strip().startswith(("File ", "raise "))]
        if cause:   # the real reason (pip's build error) sits far above the traceback tail
            log.append("przyczyna: " + " | ".join(dict.fromkeys(cause[-4:])))
        log.append(out.strip()[-800:])
    return r.returncode == 0


# --- "installed, waiting for a ComfyUI restart" + the restart button ------------------------------------------
# what -> time of a successful install that needs a ComfyUI restart (node packs / their dependencies). The Doctor shows
# "Zrestartuj ComfyUI" instead of "Zainstaluj" for these until ComfyUI was (re)started after that time.
# Kept in a file too: a MatteDesk restart (e.g. the PyTorch CUDA install) must not forget it – otherwise the old
# "Cannot import ... No module named cv2" in ComfyUI's log made the Doctor offer the same install again.
PENDING_FILE = REPO_DIR / ".comfy_restart_pending.json"


def _load_pending() -> Dict[str, float]:
    try:
        d = __import__("json").loads(PENDING_FILE.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_pending() -> None:
    try:
        if PENDING_RESTART:
            PENDING_FILE.write_text(__import__("json").dumps(PENDING_RESTART), encoding="utf-8")
        else:
            PENDING_FILE.unlink(missing_ok=True)
    except OSError:
        pass


PENDING_RESTART: Dict[str, float] = _load_pending()


def mark_pending_restart(what: str) -> None:
    PENDING_RESTART[what] = time.time()
    _save_pending()


def _comfy_url(cfg: Dict[str, Any]) -> str:
    return f"http://{cfg.get('comfy_host') or '127.0.0.1'}:{int(cfg.get('comfy_port') or 8188)}"


def comfy_started_at(base_url: str, timeout: float = 2.5) -> Optional[float]:
    """Start time of the running ComfyUI = timestamp of the first line of its log (/internal/logs); None = unknown."""
    import datetime
    import json as _json
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{base_url}/internal/logs",
                                                           headers={"User-Agent": "MatteDesk"}), timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="ignore")
        if text.startswith('"'):
            text = _json.loads(text, strict=False)
    except Exception:
        return None
    m = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)", text)
    try:
        return datetime.datetime.fromisoformat(m.group(1)).timestamp() if m else None
    except ValueError:
        return None


def pending_restart(base_url: Optional[str]) -> Dict[str, float]:
    """Installs still waiting for a restart: installed after the running ComfyUI started (unknown start = still waiting)."""
    if not PENDING_RESTART:
        return {}
    started = comfy_started_at(base_url) if base_url else None
    if started is not None:
        done = [k for k, t in PENDING_RESTART.items() if started > t]
        for k in done:
            del PENDING_RESTART[k]
        if done:
            _save_pending()
    return dict(PENDING_RESTART)


def restart_comfy(cfg: Dict[str, Any], wait_s: float = 180.0) -> Dict[str, Any]:
    """Restart ComfyUI through ComfyUI-Manager (POST /v2/manager/reboot; Desktop restarts the process itself) and wait
    until it answers again with a new start time. Refuses while ComfyUI is computing something."""
    base = _comfy_url(cfg)
    def _req(path: str, method: str = "GET", data: Optional[bytes] = None, timeout: float = 5.0):
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"User-Agent": "MatteDesk", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    import json as _json
    try:
        _, q = _req("/queue")
        qd = _json.loads(q.decode("utf-8"))
        if qd.get("queue_running") or qd.get("queue_pending"):
            raise RuntimeError("ComfyUI właśnie coś liczy – poczekaj, aż skończy, i zrestartuj ponownie.")
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError(f"ComfyUI nie odpowiada na {base} – uruchom go (Desktop: otwórz aplikację).")
    before = comfy_started_at(base)
    _step("restart ComfyUI")
    try:
        _req("/v2/manager/reboot", method="POST", data=b"{}", timeout=10)
    except urllib.error.HTTPError as e:
        hint = "Zrestartuj ComfyUI ręcznie (Desktop: zamknij i otwórz aplikację)."
        if e.code == 404:
            raise RuntimeError(f"Ten ComfyUI nie ma ComfyUI-Managera z restartem. {hint}")
        raise RuntimeError(f"ComfyUI odrzucił restart (HTTP {e.code}, poziom bezpieczeństwa Managera). {hint}")
    except Exception:
        pass   # the process exits while answering -> connection reset = restart started
    _step("czekam, aż ComfyUI wstanie")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(2.0)
        try:
            _req("/system_stats", timeout=2)
        except Exception:
            continue
        now = comfy_started_at(base)
        if now is not None and (before is None or now > before + 1):
            PENDING_RESTART.clear()
            _save_pending()
            return {"success": True, "log": [f"ComfyUI zrestartowany ({time.time() - t0:.0f} s)."], "restarted": True}
    raise RuntimeError(f"ComfyUI nie wstał w {wait_s:.0f} s – sprawdź okno ComfyUI Desktop (może pokazuje błąd).")


def install_cotracker(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run,
                      download: Callable[[str, Path], None] = _download, python: Optional[str] = None) -> Dict[str, Any]:
    """CoTracker3 runs in ComfyUI's Python (engine/cotracker_worker.py): the `cotracker` package goes THERE
    (pip --no-deps – it has no dependencies, ComfyUI's torch stays untouched) + weights into models_dir/cotracker."""
    log: List[str] = []
    py = python or (str(comfy_python(cfg)) if comfy_python(cfg) else "")
    if not py:
        return {"success": False, "log": ["Nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python w config.local.json)."]}
    chk = run([py, "-c", "import cotracker.predictor"], capture_output=True, text=True)
    if chk.returncode == 0:
        log.append("pakiet cotracker w Pythonie ComfyUI: już zainstalowany")
    else:
        _step("pip install CoTracker3 (Python ComfyUI)")
        r = run([py, "-m", "pip", "install", "--no-deps", COTRACKER_PIP], capture_output=True, text=True)
        log.append(f"pip install {COTRACKER_PIP} (Python ComfyUI) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
        if r.returncode:
            log.append((r.stderr or r.stdout or "")[-800:])
            return {"success": False, "log": log}
    try:   # the tracker caches "is cotracker importable" per interpreter
        from engine import tracking
        tracking._CHECK_CACHE.clear()
    except Exception:
        pass
    models_dir = cfg.get("models_dir") or ""
    if not models_dir:
        log.append("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json) – wagi nie pobrane.")
        return {"success": False, "log": log}
    dst = Path(models_dir) / COTRACKER_REL
    if dst.exists():
        log.append(f"wagi: już są ({dst})")
    else:
        download(COTRACKER_WEIGHTS_URL, dst)
        log.append(f"wagi pobrane: {dst}")
    return {"success": True, "log": log, "weights": str(dst)}


def install_vitmatte(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run, python: Optional[str] = None,
                     model_id: Optional[str] = None) -> Dict[str, Any]:
    """ViTMatte runs in ComfyUI's Python (engine/vitmatte_worker.py). `transformers` comes with ComfyUI itself; it is
    pip-installed there only when missing. Then the weights (~390 MB) go into the Hugging Face cache."""
    from engine import vitmatte
    log: List[str] = []
    py = python or (str(comfy_python(cfg)) if comfy_python(cfg) else "")
    if not py:
        return {"success": False, "log": ["Nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python w config.local.json)."]}
    mid = model_id or vitmatte.DEFAULT_MODEL
    chk = run([py, "-c", "import transformers; from transformers import VitMatteForImageMatting"], capture_output=True, text=True)
    if chk.returncode == 0:
        log.append("transformers w Pythonie ComfyUI: już są")
    else:
        _step("pip install transformers (Python ComfyUI)")
        r = run([py, "-m", "pip", "install", "transformers"], capture_output=True, text=True)
        log.append(f"pip install transformers (Python ComfyUI) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
        if r.returncode:
            log.append((r.stderr or r.stdout or "")[-800:])
            return {"success": False, "log": log}
    _step(f"wagi ViTMatte {mid} (~390 MB, cache Hugging Face)")
    r = run([py, "-I", str(vitmatte.WORKER), "--download", mid], capture_output=True, text=True)
    log.append(f"wagi ViTMatte {mid} -> {'pobrane' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    vitmatte._CHECK_CACHE.clear()
    if r.returncode:
        log.append((r.stderr or r.stdout or "")[-800:])
        return {"success": False, "log": log}
    return {"success": True, "log": log, "model": mid}


def install_sapiens2(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run, python: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Sapiens2-seg-0.4B (optional „Zaproponuj pas”, Sapiens2 License) in ComfyUI's Python (engine/sapiens2_worker.py):
    needs transformers with Sapiens2 (5.x) there – never upgraded here (it could break other nodes); the weights go into
    the Hugging Face cache (1.63 GB)."""
    from engine import sapiens2
    log: List[str] = []
    py = python or (str(comfy_python(cfg)) if comfy_python(cfg) else "")
    if not py:
        return {"success": False, "log": ["Nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python w config.local.json)."]}
    r = run([py, "-c", "from transformers import Sapiens2ForSemanticSegmentation"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"success": False, "log": ["W Pythonie ComfyUI brak transformers z Sapiens2 (potrzebny transformers ≥ 5) – Doktor "
                                          "go nie aktualizuje (mógłby zepsuć inne węzły). Zaktualizuj ComfyUI."]}
    _step(f"wagi {sapiens2.DEFAULT_MODEL} (cache Hugging Face, 1,63 GB)")
    r = run([py, "-I", str(sapiens2.WORKER), "--download", sapiens2.DEFAULT_MODEL], capture_output=True, text=True)
    log.append(f"wagi {sapiens2.DEFAULT_MODEL} -> {'pobrane' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        log.append((r.stderr or r.stdout or "")[-800:])
        return {"success": False, "log": log}
    from engine import vitmatte
    vitmatte._CHECK_CACHE.clear()
    return {"success": True, "log": log}


def install_birefnet(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run, python: Optional[str] = None,
                     model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """BiRefNet_HR-matting / Lucida (optional edge models of the ViTMatte zone engine, MIT weights) run in ComfyUI's
    Python (engine/birefnet_worker.py). Their remote code needs timm, kornia, einops (+ torchvision): the missing ones
    are pip-installed there – never torch / torchvision (that would change ComfyUI's PyTorch). Then the weights go into
    the Hugging Face cache (444 MB + 885 MB)."""
    from engine import birefnet, birefnet_worker
    log: List[str] = []
    py = python or (str(comfy_python(cfg)) if comfy_python(cfg) else "")
    if not py:
        return {"success": False, "log": ["Nie znaleziono Pythona ComfyUI (ustaw comfy_dir / comfy_python w config.local.json)."]}
    missing = [p for p in birefnet_worker.REMOTE_CODE_PACKAGES
               if run([py, "-c", f"import {p}"], capture_output=True, text=True).returncode != 0]
    if "torchvision" in missing:
        return {"success": False, "log": ["W Pythonie ComfyUI brak torchvision – Doktor go nie instaluje (zmieniłby PyTorch "
                                          "ComfyUI). Zaktualizuj ComfyUI albo doinstaluj torchvision zgodny z jego torch."]}
    if missing:
        _step(f"pip install {' '.join(missing)} (Python ComfyUI)")
        r = run([py, "-m", "pip", "install", *missing], capture_output=True, text=True)
        log.append(f"pip install {' '.join(missing)} (Python ComfyUI) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
        if r.returncode:
            log.append((r.stderr or r.stdout or "")[-800:])
            return {"success": False, "log": log}
    else:
        log.append("timm / kornia / einops w Pythonie ComfyUI: już są")
    for mid in (model_ids or list(birefnet.MODELS.values())):
        _step(f"wagi {mid} (cache Hugging Face)")
        r = run([py, "-I", str(birefnet.WORKER), "--download", mid], capture_output=True, text=True)
        log.append(f"wagi {mid} -> {'pobrane' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
        if r.returncode:
            log.append((r.stderr or r.stdout or "")[-800:])
            return {"success": False, "log": log}
    from engine import vitmatte
    vitmatte._CHECK_CACHE.clear()
    return {"success": True, "log": log}
