# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor's „Zaktualizuj ComfyUI” / „Cofnij aktualizację ComfyUI” for a ComfyUI installed from git (user request
2026-11-10: the workstation's ComfyUI 0.33 lacked TextEncodeQwenImage21, 503 commits behind origin/master).

Update: the checkout must have no local changes to tracked files (untracked files – models, user folders – are fine);
the current commit is remembered (bin/comfy_update.json) so it can be rolled back; `git pull --ff-only` (never a merge);
ComfyUI's requirements.txt installed into its own Python with its PyTorch pinned (installer.torch_constraints – the
CUDA build stays); then the Doctor asks for a ComfyUI restart. Rollback: `git reset --keep <remembered commit>` (keeps
nothing lost: the tree was clean) + the requirements again. ComfyUI Desktop updates itself in its app – not here.
How far behind is read with `git fetch` at most once per FETCH_EVERY_S (the Doctor opens often).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_DIR / "bin" / "comfy_update.json"     # bin/ is outside git
FETCH_EVERY_S = 3600.0
BRANCHES = ("master", "main")


def _git(cfg: Dict[str, Any]) -> Optional[str]:
    from engine.tools_install import find_tool
    return find_tool("git", cfg)


def _run(run: Callable[..., Any], git: str, d: Path, *args: str, timeout: float = 600) -> Any:
    return run([git, "-C", str(d), *args], capture_output=True, text=True, timeout=timeout)


def _out(r: Any) -> str:
    return (getattr(r, "stdout", "") or "").strip()


def load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(st: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")


def comfy_git_dir(cfg: Dict[str, Any]) -> Optional[Path]:
    d = Path(str(cfg.get("comfy_dir") or ""))
    return d if cfg.get("comfy_dir") and (d / ".git").exists() else None


def info(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run, fetch: bool = True,
         now: Optional[float] = None) -> Dict[str, Any]:
    """{kind: git | other, dir, branch, head, behind, dirty: [files], checked, can_rollback, before}. `behind` None =
    unknown (no network / no upstream)."""
    d = comfy_git_dir(cfg)
    st = load_state()
    out: Dict[str, Any] = {"kind": "git" if d else "other", "dir": str(d) if d else None,
                           "can_rollback": bool(st.get("before") and d and st.get("dir") == str(d)),
                           "before": st.get("before"), "updated": st.get("date")}
    git = _git(cfg) if d else None
    if not d or not git:
        return out
    try:
        branch = _out(_run(run, git, d, "rev-parse", "--abbrev-ref", "HEAD", timeout=30))
        head = _out(_run(run, git, d, "rev-parse", "--short", "HEAD", timeout=30))
        dirty = [ln[3:] for ln in _out(_run(run, git, d, "status", "--porcelain", "--untracked-files=no",
                                            timeout=60)).splitlines() if ln.strip()]
        now = time.time() if now is None else now
        fetched = float(st.get("fetched") or 0)
        if fetch and now - fetched > FETCH_EVERY_S:
            r = _run(run, git, d, "fetch", "-q", "origin", timeout=120)
            if getattr(r, "returncode", 1) == 0:
                st["fetched"] = now
                _save_state(st)
        r = _run(run, git, d, "rev-list", "--count", f"HEAD..origin/{branch}", timeout=30)
        behind = int(_out(r)) if getattr(r, "returncode", 1) == 0 and _out(r).isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return out
    out.update(branch=branch, head=head, dirty=dirty, behind=behind, checked=st.get("fetched"))
    return out


def _requirements(cfg: Dict[str, Any], d: Path, run: Callable[..., Any], log: List[str]) -> bool:
    from engine.installer import _log_failure, comfy_python, torch_constraints
    req = d / "requirements.txt"
    py = comfy_python(cfg)
    if not req.exists() or py is None:
        log.append("requirements.txt / Python ComfyUI nie znaleziony – zależności pominięte")
        return True
    pins = torch_constraints(py, run, REPO_DIR / "bin" / "torch_pins_comfy.txt")
    r = run([str(py), "-m", "pip", "install", "-r", str(req)] + (["-c", str(pins)] if pins else []),
            capture_output=True, text=True)
    log.append(f"pip install -r requirements.txt (PyTorch przypięty: {'tak' if pins else 'nie'}) -> "
               f"{'OK' if r.returncode == 0 else 'BŁĄD, kod ' + str(r.returncode)}")
    if r.returncode:
        _log_failure(r, log)
    return r.returncode == 0


def update(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    from engine.installer import _log_failure, _step
    d = comfy_git_dir(cfg)
    if d is None:
        raise RuntimeError("Aktualizacja z MatteDeska działa dla ComfyUI z gita. ComfyUI Desktop aktualizuje się sam "
                           "(w aplikacji: menu → Check for updates).")
    git = _git(cfg)
    if not git:
        raise RuntimeError("Brak Gita – Doktor → „Zainstaluj” przy pozycji Git.")
    i = info(cfg, run, fetch=False)
    if i.get("dirty"):
        raise RuntimeError("ComfyUI ma lokalne zmiany w plikach z gita – aktualizacja przerwana, nic nie zmieniono: "
                           + ", ".join(i["dirty"][:6]) + (" …" if len(i["dirty"]) > 6 else ""))
    if i.get("branch") not in BRANCHES:
        raise RuntimeError(f"ComfyUI jest na gałęzi „{i.get('branch')}” – aktualizacja tylko dla master / main.")
    log: List[str] = []
    before = _out(_run(run, git, d, "rev-parse", "HEAD", timeout=30))
    _save_state({**load_state(), "dir": str(d), "before": before, "date": time.strftime("%Y-%m-%d %H:%M")})
    log.append(f"zapamiętana wersja do cofnięcia: {before[:10]}")
    _step("git pull ComfyUI")
    r = _run(run, git, d, "pull", "--ff-only", "origin", i["branch"])
    log.append(f"git pull --ff-only -> kod {r.returncode}")
    if r.returncode:
        _log_failure(r, log)
        return {"success": False, "log": log}
    after = _out(_run(run, git, d, "rev-parse", "--short", "HEAD", timeout=30))
    log.append(f"ComfyUI: {before[:10]} -> {after}")
    _step("zależności ComfyUI")
    ok = _requirements(cfg, d, run, log)
    log.append("Zrestartuj ComfyUI, potem „Odśwież test” – Doktor pokaże węzły, które się nie ładują "
               "(„Cofnij aktualizację ComfyUI” wraca do poprzedniej wersji).")
    return {"success": ok, "log": log, "restart_comfy": True}


def rollback(cfg: Dict[str, Any], run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    from engine.installer import _log_failure, _step
    d = comfy_git_dir(cfg)
    st = load_state()
    if d is None or not st.get("before") or st.get("dir") != str(d):
        raise RuntimeError("Brak zapamiętanej wersji ComfyUI do cofnięcia.")
    git = _git(cfg)
    if not git:
        raise RuntimeError("Brak Gita – Doktor → „Zainstaluj” przy pozycji Git.")
    log: List[str] = []
    _step("cofanie ComfyUI")
    r = _run(run, git, d, "reset", "--keep", st["before"])
    log.append(f"git reset --keep {st['before'][:10]} -> kod {r.returncode}")
    if r.returncode:
        _log_failure(r, log)
        return {"success": False, "log": log}
    ok = _requirements(cfg, d, run, log)
    _save_state({k: v for k, v in st.items() if k not in ("before", "date")})
    log.append("Zrestartuj ComfyUI, potem „Odśwież test”.")
    return {"success": ok, "log": log, "restart_comfy": True}
