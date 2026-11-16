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
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("matte_desk.installer")

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


# how much of a failed step's output the Doctor's log (and the report package built from it) keeps
FAIL_TAIL = 3000


def failure_cause(out: str) -> List[str]:
    """The lines of a failed pip / install.py / worker output that name the reason, oldest first (at most 4,
    deduplicated): pip's `ERROR: ...` lines (any case), `...Error: ...` / `...Exception: ...` lines of a traceback,
    and the last non-empty line – pip's exit code 2 is an exception of pip itself (e.g. `InvalidVersion: ...`) whose
    name does not end in "Error"; the `~~~^^^` / `File ...` lines around it say nothing (issue 10, 2026-10-29)."""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    tb_frame = lambda l: l.startswith(("File ", "raise ")) or set(l) <= set("~^ ")
    cause, in_tb = [], False
    for i, l in enumerate(lines):
        if l.startswith("Traceback "):
            in_tb = True
        elif tb_frame(l):
            continue
        elif in_tb and tb_frame(lines[i - 1]):
            cause.append(l)   # the line that ends a traceback = the exception, whatever its name
            in_tb = False
        elif re.search(r"(\w*Error|\w*Exception)\b|^error\b", l, re.IGNORECASE) or i == len(lines) - 1:
            cause.append(l)
    return list(dict.fromkeys(cause[-4:]))


def _log_failure(r: Any, log: List[str]) -> None:
    """A failed install step (log[-1] names it): its reason + output tail into the Doctor's log, the whole output
    into MatteDesk's log (server.log). Until 2026-10-29 the Doctor's window – and the report package built from it –
    kept 800 characters of the tail and a case-sensitive `Error` filter, which hid pip's real error (issue 10)."""
    step = log[-1] if log else "?"
    out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    cause = failure_cause(out)
    if cause:
        log.append("przyczyna: " + " | ".join(cause))
    if out:
        log.append(out[-FAIL_TAIL:])
    logger.error("%s\n%s", step, out[-20000:])


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
    if not cfg.get("comfy_dir"):     # (Path("") is "." – it used to find MatteDesk's own .venv in the working folder)
        return None
    d = Path(cfg["comfy_dir"])
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


def _pip_requirements(cfg: Dict[str, Any], name: str, pkg_dir: Path, run: Callable[..., Any], log: List[str],
                      constraints: Optional[Path] = None) -> bool:
    """pip install -r <pkg>/requirements.txt into ComfyUI's Python (always, also for an up-to-date package: a node pack
    copied earlier may still miss its dependencies -> "IMPORT FAILED" in Comfy). `constraints` = pip -c file (e.g.
    torch_constraints: keep ComfyUI's PyTorch). False = pip failed."""
    req = pkg_dir / "requirements.txt"
    if not (req.exists() and req.read_text(encoding="utf-8", errors="ignore").strip()):
        return True
    py = comfy_python(cfg)
    if py is None:
        log.append(f"{name}: nie znaleziono Pythona ComfyUI – doinstaluj ręcznie: pip install -r {req}")
        return False
    _step(f"pip install ({name})")
    r = run([str(py), "-m", "pip", "install", "-r", str(req)] + (["-c", str(constraints)] if constraints else []),
            capture_output=True, text=True)
    log.append(f"{name}: zależności (pip install -r requirements.txt) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        _log_failure(r, log)
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
    # Comfy I/O "png8" (8-bit sources): VHS_LoadImagesPath
    "vhs_nodes": ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "ComfyUI-VideoHelperSuite"),
    # stage VideoMaMa (2026-10-16): VideoMaMaPipelineLoader + VideoMaMaSampler (diffusers / transformers / accelerate)
    "videomama_node": ("https://github.com/okdalto/ComfyUI-VideoMaMa", "ComfyUI-VideoMaMa"),
    # Cleanup 3D „SeedVR2: podbij klucze karty” (2026-11-10): MatteDesk runs its inference_cli.py with ComfyUI's Python
    "seedvr2_node": ("https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler", "ComfyUI-SeedVR2_VideoUpscaler"),
    # Inpaint C1 „Wan 2.2 Fun VACE” (2026-11-11): UnetLoaderGGUF for the Q4_K_S experts
    "gguf_node": ("https://github.com/city96/ComfyUI-GGUF", "ComfyUI-GGUF"),
}
# Other folder names of the same pack (ComfyUI Manager clones under the registry id) – never clone a second copy
GIT_NODE_ALIASES = {"vhs_nodes": ("comfyui-videohelpersuite",), "gguf_node": ("comfyui-gguf",)}

# Multi-file weights: key -> (Hugging Face repo, [files], folder in models_dir)
HF_FOLDERS: Dict[str, Tuple[str, List[str], str]] = {}   # (MiniMax-Remover weights until 2026-11-09)


# weights whose ComfyUI node reads ONLY folder_paths.models_dir/<folder> (not the extra model paths): ComfyUI Desktop keeps
# the models in ComfyUI-Shared\models (--extra-model-paths-config) while models_dir is the install's empty models\ – the
# MiniMax-Remover loader (dropped 2026-11-09) then downloaded its weights again. A directory junction install\models\<folder> ->
# models_dir\<folder> lets it find them (nothing copied, custom_nodes untouched).
COMFY_MODELS_DIR_LINKS: Tuple[str, ...] = ()


def comfy_models_link(cfg: Dict[str, Any], folder: str) -> Tuple[Optional[Path], Optional[Path]]:
    """(link in the Comfy install's models\, target in models_dir) when they differ; (None, None) when the same folder."""
    comfy_dir, models_dir = cfg.get("comfy_dir") or "", cfg.get("models_dir") or ""
    if not comfy_dir or not models_dir:
        return None, None
    link, target = Path(comfy_dir) / "models" / folder, Path(models_dir) / folder
    try:
        if link.resolve() == target.resolve():
            return None, None
    except OSError:
        pass
    return link, target


def link_into_comfy_models(cfg: Dict[str, Any], folder: str) -> str:
    """Create the junction (Windows) / symlink of COMFY_MODELS_DIR_LINKS; a log line."""
    link, target = comfy_models_link(cfg, folder)
    if link is None:
        return f"{folder}: modele w folderze ComfyUI – połączenie niepotrzebne"
    if link.exists():
        return f"{folder}: ComfyUI widzi już {link}"
    if not target.is_dir():
        return f"{folder}: brak {target} – połączenie pominięte"
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=30)
        if r.returncode != 0:
            return f"{folder}: nie udało się połączyć {link} -> {target}: {(r.stderr or r.stdout).strip()[:200]}"
    else:
        link.symlink_to(target, target_is_directory=True)
    return f"{folder}: połączono {link} -> {target} (węzeł ComfyUI czyta tylko swój folder models)"


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
        if key in COMFY_MODELS_DIR_LINKS:
            log.append(link_into_comfy_models(cfg, folder))
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
    from engine.ltx_alpha import HF_PATH, LTX_MODELS, models_status
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
    needed = {m["key"]: m["needed"] for m in models_status(models_dir)}
    for key, (folder, fn, repo, gb) in LTX_MODELS.items():
        if keys and key not in keys:
            continue
        dst = Path(models_dir) / folder / fn
        if not needed[key] and not dst.exists():
            log.append(f"niepotrzebne: {fn} (pusty prompt już zakodowany – ltx_conditioning)")
            continue
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


# --- Cleanup 3D (docs/PLAN_CLEANUP_3D.md, package P3) -------------------------------------------------------------
def _klein_texts() -> Tuple[str, str]:
    from engine.stage_models import KLEIN_PAGE
    login = ("Brak logowania do Hugging Face: FLUX 3 Image Klein jest „gated”. Zaloguj się („hf auth login”, token Read z "
             "https://huggingface.co/settings/tokens) i zaakceptuj dostęp (FLUX Kommunity License) na stronie modelu "
             f"{KLEIN_PAGE}, potem kliknij „Zainstaluj” jeszcze raz.")
    denied = ("Hugging Face odmówił dostępu do FLUX 3 Image Klein (401/403): zaakceptuj dostęp (FLUX Kommunity License) "
              f"na stronie modelu {KLEIN_PAGE} tym samym kontem, którym zalogowano „hf auth login”, potem kliknij "
              "„Zainstaluj” jeszcze raz – pobieranie wznowi się.")
    return login, denied


def _denied(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (401, 403)
    return "401/403" in str(e)


def _install_hf_files(cfg: Dict[str, Any], files: List[Dict[str, Any]], what: str,
                      download: Callable[[str, Path], None], denied_msg: Optional[str] = None) -> Dict[str, Any]:
    """Download the missing files of `files` (stage_models.*_status rows) into models_dir; present files are skipped,
    interrupted downloads resume (engine/download.py, SHA-256 from Hugging Face)."""
    models_dir = Path(cfg["models_dir"])
    log: List[str] = []
    for m in files:
        dst = models_dir / m["folder"] / m["file"]
        if m["present"]:
            log.append(f"już jest: {dst}")
            continue
        _step(f"{what}: {m['file']} ({m['gb']:.2f} GB)")
        try:
            download(m["url"], dst)
        except (RuntimeError, urllib.error.HTTPError) as e:
            if denied_msg and _denied(e):
                raise RuntimeError(denied_msg) from None
            raise
        log.append(f"pobrane: {dst} ({dst.stat().st_size / 2 ** 30:.2f} GB)")
    log.append("Odśwież listę modeli w ComfyUI (klawisz R) albo zrestartuj Comfy.")
    return {"success": True, "log": log}


def install_flux3_klein(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None) -> Dict[str, Any]:
    """Cleanup 3D engine „FLUX 3 Image Klein + LoRA Restore”: the three FLUX 3 Image Klein files (~20.3 GB) from the gated
    black-forest-labs repo with the user's HF login (as LTX-2.5)."""
    from engine.stage_models import klein_status
    if not cfg.get("models_dir"):
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    login, denied = _klein_texts()
    if download is None:
        token = hf_token()
        if not token:
            raise RuntimeError(login)
        download = lambda url, dst: _download_hf(url, dst, token)
    return _install_hf_files(cfg, klein_status(Path(cfg["models_dir"])), "FLUX 3 Image Klein", download, denied)


def install_klein_restore_lora(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None
                               ) -> Dict[str, Any]:
    """LoRA Restore for FLUX 3 Image Klein (public repo, 161 MB) -> models/loras."""
    from engine.stage_models import klein_lora_status
    if not cfg.get("models_dir"):
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    return _install_hf_files(cfg, klein_lora_status(Path(cfg["models_dir"])), "LoRA Restore (FLUX 3 Image Klein)", download)


def install_qwen21(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None) -> Dict[str, Any]:
    """Cleanup 3D engine „Qwen-Image-2.1”: the three files (~17.3 GB) of Comfy-Org/Qwen-Image-2.1 (the HF login is used
    when there is one; the repo is not gated as far as checked 2026-11-10)."""
    from engine.stage_models import QWEN21_PAGE, qwen21_status
    if not cfg.get("models_dir"):
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    denied = (f"Hugging Face odmówił dostępu do Qwen-Image-2.1 (401/403): otwórz stronę modelu {QWEN21_PAGE}, zaakceptuj "
              "warunki, jeśli strona o to prosi, zaloguj się („hf auth login”) i kliknij „Zainstaluj” jeszcze raz.")
    return _install_hf_files(cfg, qwen21_status(Path(cfg["models_dir"])), "Qwen-Image-2.1", download, denied)


# COLMAP (CUDA) for the camera solve of Cleanup 3D (package P10): the official Windows build, BSD. Checked at the source
# 2026-11-09 (GitHub releases API: size + `digest`). Unpacked next to the portable ffmpeg (engine/tools_install.py BIN_DIR).
COLMAP_TAG = "4.2.1"
COLMAP_ASSET = "colmap-x64-windows-cuda.zip"
COLMAP_URL = f"https://github.com/colmap/colmap/releases/download/{COLMAP_TAG}/{COLMAP_ASSET}"
COLMAP_SHA256 = "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"
COLMAP_BYTES = 414666626
COLMAP_DIR = REPO_DIR / "bin" / "colmap"


def colmap_exe(cfg: Optional[Dict[str, Any]] = None, root: Optional[Path] = None) -> Optional[Path]:
    """colmap.exe: config `colmap_path`, else <repo>/bin/colmap (bin/colmap.exe of the official package); None = missing."""
    p = str((cfg or {}).get("colmap_path") or "")
    if p and Path(p).is_file():
        return Path(p)
    root = Path(root or COLMAP_DIR)
    for c in (root / "bin" / "colmap.exe", root / "colmap.exe"):
        if c.is_file():
            return c
    return None


def install_colmap(cfg: Dict[str, Any], download: Optional[Callable[[str, Path, str], None]] = None,
                   root: Optional[Path] = None) -> Dict[str, Any]:
    """Doctor's „Zainstaluj COLMAP (CUDA)”: the release zip (resumable, SHA-256 checked) unpacked into <repo>/bin/colmap."""
    import zipfile
    root = Path(root or COLMAP_DIR)
    found = colmap_exe(cfg, root)
    if found:
        return {"success": True, "log": [f"COLMAP już jest: {found}"], "path": str(found)}
    if download is None:
        from engine.download import download as _dl
        download = lambda url, dst, sha: _dl(url, dst, sha256=sha, progress=_progress_cb(dst))
    root.parent.mkdir(parents=True, exist_ok=True)
    zpath = root.parent / COLMAP_ASSET
    _step(f"COLMAP {COLMAP_TAG} (CUDA): {COLMAP_ASSET} ({COLMAP_BYTES / 2 ** 20:.0f} MB)")
    download(COLMAP_URL, zpath, COLMAP_SHA256)
    _step("rozpakowywanie COLMAP")
    tmp = root.parent / "colmap_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    exe = next(iter(sorted(tmp.rglob("colmap.exe"))), None)
    if exe is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"W paczce {COLMAP_ASSET} nie ma colmap.exe – pobierz ręcznie z "
                           "https://github.com/colmap/colmap/releases")
    top = exe.parent.parent if exe.parent.name.lower() == "bin" else exe.parent
    shutil.rmtree(root, ignore_errors=True)
    shutil.move(str(top), str(root))
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    found = colmap_exe(cfg, root)
    if not found:
        raise RuntimeError(f"COLMAP rozpakowany do {root}, ale nie znaleziono colmap.exe")
    return {"success": True, "log": [f"COLMAP {COLMAP_TAG} (CUDA) rozpakowany do {root}"], "path": str(found)}


# --- Cleanup 3D P12: SeedVR2 inside ComfyUI, VOID in a venv on ComfyUI's Python (user decision 2026-11-10) ---------
TORCH_PKGS = ("torch", "torchvision", "torchaudio")


def torch_constraints(py: Path, run: Callable[..., Any], dst: Path) -> Optional[Path]:
    """A pip constraints file pinning the PyTorch packages `py` already has (exact version incl. +cuXXX) – pip then
    never swaps ComfyUI's CUDA PyTorch for another build. None = no PyTorch there."""
    code = ("import importlib.metadata as m\n"
            f"for p in {TORCH_PKGS!r}:\n"
            "    try: print(p + '==' + m.version(p))\n"
            "    except m.PackageNotFoundError: pass\n")
    r = run([str(py), "-c", code], capture_output=True, text=True)
    pins = [ln.strip() for ln in (getattr(r, "stdout", "") or "").splitlines() if "==" in ln]
    if getattr(r, "returncode", 1) != 0 or not any(p.startswith("torch==") for p in pins):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return dst


def _hf_files_to(repo: str, files: List[Tuple[str, float]], dst_dir: Path, what: str,
                 download: Callable[[str, Path], None], log: List[str]) -> None:
    """Download `files` [(path in repo, GB)] of the public HF `repo` into dst_dir; present files (>= 90 % of the size)
    are skipped, interrupted ones resume."""
    for rel, gb in files:
        dst = dst_dir / rel
        if dst.exists() and dst.stat().st_size >= 0.9 * gb * 1e9:
            log.append(f"już jest: {dst}")
            continue
        _step(f"{what}: {rel}" + (f" ({gb:.1f} GB)" if gb >= 0.05 else ""))
        download(f"https://huggingface.co/{repo}/resolve/main/{rel}", dst)
        log.append(f"pobrane: {dst}" + (f" ({dst.stat().st_size / 2 ** 30:.2f} GB)" if gb >= 0.05 else ""))


SEEDVR2_HF = "numz/SeedVR2_comfyUI"
SEEDVR2_FILES = [("seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors", 7.67), ("ema_vae_fp16.safetensors", 0.47)]


def seedvr2_files(card_gb: Optional[float] = None) -> List[Tuple[str, float]]:
    """The DiT of the card's variant (void_engine.seedvr2_variant: 3B fp8 under 16 GB, else 7B sharp fp8) + the VAE."""
    from engine.cleanup3d.void_engine import SEEDVR2_DITS, seedvr2_variant
    return [SEEDVR2_DITS[seedvr2_variant(card_gb)], SEEDVR2_FILES[1]]


def install_seedvr2(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                    run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """Doctor's „Zainstaluj SeedVR2”: the ComfyUI-SeedVR2_VideoUpscaler node into ComfyUI's custom_nodes, its
    requirements into ComfyUI's Python with ComfyUI's PyTorch pinned, and the DiT of the card's variant (3B fp8 3.4 GB
    under 16 GB VRAM, else 7B sharp fp8 8.2 GB) + the VAE into
    <models_dir>/SEEDVR2. MatteDesk runs its inference_cli.py with ComfyUI's Python – no ComfyUI restart needed."""
    from engine.cleanup3d.void_engine import tool_paths
    py = comfy_python(cfg)
    if py is None:
        raise RuntimeError("Nie znaleziono Pythona ComfyUI – SeedVR2 instaluje się w ComfyUI. Ustaw ComfyUI w Doktorze.")
    pins = torch_constraints(py, run, REPO_DIR / "bin" / "torch_pins_comfy.txt")   # bin/ is outside git
    res = install_git_node(cfg, "seedvr2_node", run=run, constraints=pins)
    log = [ln for ln in (res.get("log") or []) if not ln.startswith("Zrestartuj ComfyUI")]
    log.append("PyTorch ComfyUI przypięty (pip -c): " + (pins.read_text(encoding="utf-8").replace("\n", " ").strip()
                                                         if pins else "nie odczytano wersji – pip mógł go zmienić"))
    if not res.get("success"):
        return {"success": False, "log": log}
    models = tool_paths(cfg, "seedvr2")["models"]
    if models is None:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    _hf_files_to(SEEDVR2_HF, seedvr2_files(), Path(models), "SeedVR2", download, log)
    log.append("SeedVR2 gotowy – „Odśwież test”.")
    return {"success": True, "log": log, "restart_comfy": False}


VOID_GIT = "https://github.com/Netflix/void-model"
VOID_HF = "netflix/void-model"
VOID_BASE_HF = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
# the repo's requirements.txt pins torch==2.7.1 (the PyPI build for Windows has no CUDA) and training tools (deepspeed,
# tensorboard, datasets): MatteDesk installs only what inference (predict_v2v.py and the videox_fun modules it imports,
# checked 2026-11-10) needs; the API-sensitive libraries at the repo's versions, the rest as ComfyUI has them.
VOID_PIP = ["diffusers==0.33.1", "transformers==4.57.1", "accelerate==1.12.0", "peft==0.17.1", "safetensors>=0.6.2",
            "absl-py", "ml_collections", "loguru", "mediapy", "einops", "imageio", "imageio-ffmpeg", "scikit-image",
            "matplotlib", "scipy", "albumentations", "decord", "func-timeout", "sentencepiece", "opencv-python",
            "Pillow", "omegaconf", "tqdm", "ftfy", "beautifulsoup4", "timm"]
# predict_v2v.py loads the VAE without tiling: at 1344x768 it then fills a 24 GB card and stalls (experiment 2026-11-08,
# experiments/removers/void/patches/void_windows.patch). Inserted after the VAE load, idempotent.
VOID_VAE_ANCHOR = 'subfolder="vae"\n    ).to(weight_dtype)\n'
VOID_VAE_PATCH = "    vae.enable_tiling(); vae.enable_slicing()  # MatteDesk: hi-res VAE without filling the card\n"
VOID_BASE_SKIP = (".gitattributes", "README.md", "README_en.md", "LICENSE")


def patch_void_vae(repo: Path) -> str:
    """Add VAE tiling / slicing to the clone's predict_v2v.py (our clone in bin/void, not a ComfyUI node)."""
    f = Path(repo) / "inference" / "cogvideox_fun" / "predict_v2v.py"
    src = f.read_text(encoding="utf-8")
    if VOID_VAE_PATCH in src:
        return "predict_v2v.py: kafelkowanie VAE już jest"
    if VOID_VAE_ANCHOR not in src:
        return "predict_v2v.py: nie znaleziono miejsca łatki kafelkowania VAE – przy 1344×768 VOID może stanąć"
    f.write_text(src.replace(VOID_VAE_ANCHOR, VOID_VAE_ANCHOR + VOID_VAE_PATCH, 1), encoding="utf-8")
    return "predict_v2v.py: dodane kafelkowanie VAE (enable_tiling / enable_slicing)"


def _hf_tree(repo: str, token: Optional[str]) -> List[Tuple[str, float]]:
    """[(path, GB)] of every file in a Hugging Face model repo (the public tree API)."""
    import json as _json
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true",
                                 headers={"User-Agent": "MatteDesk", **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = _json.loads(r.read().decode("utf-8"))
    return [(x["path"], float(x.get("size") or 0) / 1e9) for x in rows if x.get("type") == "file"]


def install_void_legacy(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                 run: Callable[..., Any] = subprocess.run, tree: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
                 home: Optional[Path] = None) -> Dict[str, Any]:
    """Doctor's „Zainstaluj VOID”: git clone Netflix/void-model into bin/void, a venv made with ComfyUI's Python that
    sees ComfyUI's packages through a .pth (its CUDA PyTorch is reused, not downloaded again; the venv's own packages –
    diffusers / transformers at VOID's versions – come first, ComfyUI stays untouched), the VAE tiling patch, and the
    weights: CogVideoX-Fun-V1.5-5b-InP (19.7 GB) + void_pass1.safetensors (10.4 GB) into <models_dir>/void."""
    from engine.cleanup3d import void_engine as VE
    py = comfy_python(cfg)
    if py is None:
        raise RuntimeError("Nie znaleziono Pythona ComfyUI – venv VOID powstaje na nim (jego PyTorch z CUDA). "
                           "Ustaw ComfyUI w Doktorze.")
    from engine.tools_install import find_tool
    git = find_tool("git", cfg)
    if not git:
        raise RuntimeError("Brak Gita – kliknij „Zainstaluj” przy pozycji „Git” w Doktorze, potem spróbuj ponownie.")
    home = Path(home or VE.VOID_HOME)
    paths = VE.tool_paths(cfg, "void")
    repo = Path(cfg.get("void_repo") or home / "void-model")
    venv = home / "venv"
    vpy = Path(cfg.get("void_python") or VE.venv_python(venv))
    log: List[str] = []
    fail = lambda r: (_log_failure(r, log), {"success": False, "log": log})[1]   # noqa: E731
    home.mkdir(parents=True, exist_ok=True)
    if not (repo / VE.VOID_SCRIPT).exists():
        _step("git clone void-model")
        r = run([git, "clone", "--depth", "1", VOID_GIT, str(repo)], capture_output=True, text=True)
        log.append(f"git clone {VOID_GIT} -> kod {r.returncode}")
        if r.returncode:
            return fail(r)
    else:
        log.append(f"void-model: już jest ({repo})")
    log.append(patch_void_vae(repo))
    if not vpy.exists():
        _step("venv VOID (Python ComfyUI)")
        r = run([str(py), "-m", "venv", str(venv)], capture_output=True, text=True)
        log.append(f"venv {venv} z {py} -> kod {r.returncode}")
        if r.returncode:
            return fail(r)
    # ComfyUI's site-packages AFTER the venv's own (a .pth appends to sys.path): its PyTorch is reused
    r = run([str(py), "-c", "import os, torch; print(os.path.dirname(os.path.dirname(torch.__file__)))"],
            capture_output=True, text=True)
    comfy_site = (getattr(r, "stdout", "") or "").strip().splitlines()[-1:] if r.returncode == 0 else []
    if not comfy_site:
        log.append("Python ComfyUI nie ma PyTorcha – venv VOID nie ma czego współdzielić.")
        return fail(r)
    r = run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True)
    if r.returncode:
        return fail(r)
    site = Path((r.stdout or "").strip().splitlines()[-1])
    site.mkdir(parents=True, exist_ok=True)
    (site / "_matte_desk_comfy.pth").write_text(comfy_site[0] + "\n", encoding="utf-8")
    log.append(f"venv VOID widzi pakiety ComfyUI: {comfy_site[0]}")
    pins = torch_constraints(py, run, home / "torch_pins.txt")
    _step("pip install (VOID, tylko inferencja)")
    r = run([str(vpy), "-m", "pip", "install", *VOID_PIP] + (["-c", str(pins)] if pins else []),
            capture_output=True, text=True)
    log.append(f"pip install (VOID) -> {'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        return fail(r)
    r = run([str(vpy), "-c", "import torch, diffusers, transformers; "
                             "print(torch.__version__, torch.cuda.is_available(), diffusers.__version__, transformers.__version__)"],
            capture_output=True, text=True)
    log.append("venv VOID: torch / CUDA / diffusers / transformers = " + ((r.stdout or "").strip() or "?"))
    if r.returncode:
        return fail(r)
    models = Path(paths["models"]) if cfg.get("void_models") or cfg.get("models_dir") else repo
    token = hf_token()
    if download is None:
        download = lambda url, dst: _download_hf(url, dst, token)
    tree = tree or (lambda rp: _hf_tree(rp, token))
    base = [(p, gb) for p, gb in tree(VOID_BASE_HF) if Path(p).name not in VOID_BASE_SKIP]
    _hf_files_to(VOID_BASE_HF, base, models / VE.VOID_BASE, "VOID: CogVideoX-Fun-V1.5-5b-InP", download, log)
    _hf_files_to(VOID_HF, [(VE.VOID_TRANSFORMER, 10.38)], models, "VOID", download, log)
    log.append("VOID gotowy – „Odśwież test”.")
    return {"success": True, "log": log, "restart_comfy": False}


def install_void(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                 run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """Doctor's „Zainstaluj VOID” (2026-11-11: natively in ComfyUI ≥ 0.45, no repo / venv): Comfy-Org/void-model
    void_pass1 (11.1 GB, a whole transformer) + cogvideox_vae (0.4 GB) and – only when no t5xxl is there – t5xxl_fp16
    (9.8 GB) into ComfyUI's models. The old repo + venv setup (install_void_legacy) stays usable when complete."""
    from engine.cleanup3d import void_engine as VE
    md = cfg.get("models_dir")
    if not md:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    have = VE.void_native_files(cfg)
    log: List[str] = []
    for rel, folder, name, gb in VE.VOID_NATIVE_FILES:
        if folder == "text_encoders" and have["clip"]:
            log.append(f"już jest: {have['clip']} (koder T5 wspólny z innymi modelami)")
            continue
        dst = Path(md) / folder / name
        if dst.exists() and dst.stat().st_size >= 0.99 * gb * 1e9:
            log.append(f"już jest: {dst}")
            continue
        _step(f"VOID: {name} ({gb:.1f} GB)")
        download(f"https://huggingface.co/{VE.VOID_NATIVE_REPO}/resolve/main/{rel}", dst)
        log.append(f"pobrane: {dst}")
    log.append("VOID gotowy (natywnie w ComfyUI, wymaga ComfyUI ≥ 0.45) – „Odśwież test”.")
    return {"success": True, "log": log, "restart_comfy": False}


def install_vace(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                 run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """Doctor's „Zainstaluj Wan 2.2 Fun VACE” (Inpaint C1, 2026-11-11): the ComfyUI-GGUF node (UnetLoaderGGUF) into
    ComfyUI's custom_nodes with ComfyUI's PyTorch pinned, the two Q4_K_S experts (2 x 10.6 GB, QuantStack, Apache-2.0)
    into models/diffusion_models, the two Lightning LoRAs (2 x 1.23 GB – user 2026-11-13: installed WITH the model, so
    „szybko – 4 kroki” never silently runs 20) and – only when missing – the umt5 text encoder (6.7 GB) and the Wan VAE
    (0.25 GB). A ComfyUI restart is needed for the new node."""
    from engine import vace_fill as VF
    py = comfy_python(cfg)
    if py is None:
        raise RuntimeError("Nie znaleziono Pythona ComfyUI – Wan 2.2 Fun VACE działa w ComfyUI. Ustaw ComfyUI w Doktorze.")
    md = cfg.get("models_dir")
    if not md:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    pins = torch_constraints(py, run, REPO_DIR / "bin" / "torch_pins_comfy.txt")
    res = install_git_node(cfg, "gguf_node", run=run, constraints=pins)
    log = list(res.get("log") or [])
    if not res.get("success"):
        return {"success": False, "log": log}
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    rows = [(VF.REPO, rel, folder, name, size) for rel, folder, name, size, _sha in VF.HF_FILES]
    rows += [(VF.LIGHTNING_REPO, rel, folder, name, size) for rel, folder, name, size in VF.LIGHTNING_FILES]
    have = VF.files(cfg)
    rows += [(VF.WAN_REPO, rel, folder, name, size) for rel, folder, name, size in VF.SHARED_FILES
             if not have["clip" if folder == "text_encoders" else "vae"]]
    for repo, rel, folder, name, size in rows:
        dst = Path(md) / folder / name
        if dst.exists() and dst.stat().st_size >= 0.99 * size:
            log.append(f"już jest: {dst}")
            continue
        _step(f"{VF.LABEL}: {name} ({size / 1e9:.1f} GB)")
        download(f"https://huggingface.co/{repo}/resolve/main/{rel}", dst)
        log.append(f"pobrane: {dst}")
    log.append("Wan 2.2 Fun VACE gotowy – zrestartuj ComfyUI (nowy węzeł ComfyUI-GGUF), potem „Odśwież test”.")
    return {"success": True, "log": log, "restart_comfy": True}


def install_vace_lightning(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None,
                           run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """Doctor's „Zainstaluj LoRA Lightning (Wan 2.2 Fun VACE)” (2026-11-11): the two lightx2v 4-step LoRAs (T2V A14B,
    2 x 1.23 GB, Comfy-Org/Wan_2.2_ComfyUI_Repackaged) into models/loras – Wan 2.2 Fun VACE in 4 steps at CFG 1."""
    from engine import vace_fill as VF
    md = cfg.get("models_dir")
    if not md:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    log: List[str] = []
    for rel, folder, name, size in VF.LIGHTNING_FILES:
        dst = Path(md) / folder / name
        if dst.exists() and dst.stat().st_size >= 0.99 * size:
            log.append(f"już jest: {dst}")
            continue
        _step(f"LoRA Lightning: {name} ({size / 1e9:.2f} GB)")
        download(f"https://huggingface.co/{VF.LIGHTNING_REPO}/resolve/main/{rel}", dst)
        log.append(f"pobrane: {dst}")
    log.append("LoRA Lightning gotowe – Wan 2.2 Fun VACE liczy „szybko” w 4 krokach (restart ComfyUI niepotrzebny).")
    return {"success": True, "log": log, "restart_comfy": False}


def install_ltx_inpaint(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None) -> Dict[str, Any]:
    """Doctor's „Zainstaluj LoRA inpaintu LTX-2.5” (Inpaint C1, 2026-11-13): the in-/outpainting IC-LoRA (1.31 GB,
    Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting, gated – the user's HF login with the LTX-2.x licence accepted) into
    models/loras. The base LTX-2.5 files are stage 7's install („ltx”)."""
    from engine import ltx_fill as LF
    md = cfg.get("models_dir")
    if not md:
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    folder, name, repo, size = LF.LORA
    if download is None:
        token = hf_token()
        if not token:
            raise RuntimeError(f"Brak logowania do Hugging Face: {repo} jest „gated”. Zaloguj się („hf auth login”) i "
                               f"zaakceptuj licencję LTX-2.x na stronie huggingface.co/{repo}.")
        download = lambda url, dst: _download_hf(url, dst, token)
    dst = Path(md) / folder / name
    log: List[str] = []
    if dst.exists() and dst.stat().st_size >= 0.99 * size:
        log.append(f"już jest: {dst}")
    else:
        _step(f"LoRA inpaintu LTX-2.5: {name} ({size / 1e9:.2f} GB)")
        download(f"https://huggingface.co/{repo}/resolve/main/{name}", dst)
        log.append(f"pobrane: {dst}")
    log.append("LoRA inpaintu gotowa – silnik „LTX-2.5” w Inpaint (C1); odśwież listę modeli w ComfyUI (R), restart niepotrzebny.")
    return {"success": True, "log": log, "restart_comfy": False}


def install_prompt_assist(cfg: Dict[str, Any], download: Optional[Callable[[str, Path], None]] = None) -> Dict[str, Any]:
    """Inpaint (C1) „✨ Popraw opis”: the Qwen3-VL-4B-Instruct snapshot (~8.9 GB, Apache-2.0, not gated) into
    models/LLM/Qwen3-VL-4B-Instruct; runs in ComfyUI's Python with its transformers (engine/prompt_worker.py)."""
    from engine import prompt_assist as PA
    if not cfg.get("models_dir"):
        raise RuntimeError("Brak models_dir (ustaw comfy_dir/models_dir w config.local.json).")
    if download is None:
        token = hf_token()
        download = lambda url, dst: _download_hf(url, dst, token)
    r = _install_hf_files(cfg, PA.status(Path(cfg["models_dir"])), PA.LABEL, download)
    r["log"] = [x for x in r["log"] if "Odśwież listę modeli" not in x] + [
        f"{PA.LABEL} gotowy – „✨ Popraw opis” w palecie Inpaint (restart ComfyUI niepotrzebny)."]
    r["restart_comfy"] = False
    return r


# Doctor actions without their own branch in app.py's /api/install/{what}: what -> installer(cfg)
INSTALL_ACTIONS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "prompt_assist": install_prompt_assist,
    "flux3_klein": install_flux3_klein,
    "klein_restore_lora": install_klein_restore_lora,
    "qwen21": install_qwen21,
    "colmap": install_colmap,
    "seedvr2": install_seedvr2,
    "void": install_void,
    "vace": install_vace,
    "vace_lightning": install_vace_lightning,
    "ltx_inpaint": install_ltx_inpaint,
    # ComfyUI from git (2026-11-10): engine/comfy_update.py
    "comfy_update": lambda cfg: __import__("engine.comfy_update", fromlist=["update"]).update(cfg),
    "comfy_rollback": lambda cfg: __import__("engine.comfy_update", fromlist=["rollback"]).rollback(cfg),
}


def install_git_node(cfg: Dict[str, Any], key: str, run: Callable[..., Any] = subprocess.run,
                     constraints: Optional[Path] = None) -> Dict[str, Any]:
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
            _log_failure(r, log)
            return {"success": False, "log": log}
    ok = _pip_requirements(cfg, name, dst, run, log, constraints)
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
        _log_failure(r, log)   # the real reason (pip's build error) sits far above the traceback tail
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
            _log_failure(r, log)
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
            _log_failure(r, log)
            return {"success": False, "log": log}
    _step(f"wagi ViTMatte {mid} (~390 MB, cache Hugging Face)")
    r = run([py, "-I", str(vitmatte.WORKER), "--download", mid], capture_output=True, text=True)
    log.append(f"wagi ViTMatte {mid} -> {'pobrane' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    vitmatte._CHECK_CACHE.clear()
    if r.returncode:
        _log_failure(r, log)
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
        _log_failure(r, log)
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
            _log_failure(r, log)
            return {"success": False, "log": log}
    else:
        log.append("timm / kornia / einops w Pythonie ComfyUI: już są")
    for mid in (model_ids or list(birefnet.MODELS.values())):
        _step(f"wagi {mid} (cache Hugging Face)")
        r = run([py, "-I", str(birefnet.WORKER), "--download", mid], capture_output=True, text=True)
        log.append(f"wagi {mid} -> {'pobrane' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
        if r.returncode:
            _log_failure(r, log)
            return {"success": False, "log": log}
    from engine import vitmatte
    vitmatte._CHECK_CACHE.clear()
    return {"success": True, "log": log}
