# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/updates.py - "is there a newer MatteDesk?" (user decision 2026-10-17).

At start the UI asks `GET /api/update`; at most once a day this reads the latest release of the public repo
(GitHub Releases of fishfinger44/MatteDesk – one release per public version, tag `v<version>`, the installer
`MatteDesk-<version>-setup.exe` attached, the release text = short list of changes and fixes). A newer version ->
a window with that list, a link to the full CHANGELOG and: Update / Later / Skip this version.

- Installed with the Windows installer (`python/` next to the code, no `.git`): "Update" downloads the new setup
  (resume + SHA-256 from GitHub) and starts it; the setup replaces the program in place, config.local.json stays.
- A git checkout: no setup – the window says `git pull` (+ restart).
Settings in config.local.json: `update_check` (on by default; switch in the Doctor), `update_skip` (skipped version),
`update_last` (time + result of the last check, so a restart does not ask GitHub again).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from engine import download as dl

REPO = "fishfinger44/MatteDesk"
RELEASES = f"https://github.com/{REPO}/releases"
CHANGELOG_URL = f"https://github.com/{REPO}/blob/main/CHANGELOG.md"
DAY = 24 * 3600
NOTES_LINES = 12


def parse_version(v: str) -> tuple:
    """MatteDesk numbering 0.D / 0.DS (0.3 < 0.31 < 0.32 < 0.4; 0.11 = small change of 0.1) -> comparable tuple."""
    m = re.match(r"^v?(\d+)\.(\d)(\d*)$", str(v or "").strip())
    if not m:
        return (-1,)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def install_kind(app_dir: Path) -> str:
    """installer | git"""
    return "installer" if (app_dir / "python" / "python.exe").exists() and not (app_dir / ".git").exists() else "git"


def short_notes(body: str, n: int = NOTES_LINES) -> str:
    """The release text, at most `n` non-empty lines (the window links the full changelog)."""
    # plain text for the window: markdown links / bold removed, the release page's "⬇ Download" line skipped (the window
    # has its own Update button)
    def plain(ln: str) -> str:
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", ln).replace("**", "").rstrip()
    lines = [plain(ln) for ln in (body or "").replace("\r", "").split("\n") if ln.strip() and "⬇" not in ln]
    out = lines[:n]
    if len(lines) > n:
        out.append("…")
    return "\n".join(out)


def _api(url: str) -> Dict[str, Any]:
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": dl.UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def latest_release(api_get: Callable[[str], Dict[str, Any]] = _api) -> Optional[Dict[str, Any]]:
    """{version, notes, url, setup: {url, sha256, size} | None}; None when the repo has no release yet."""
    import urllib.error
    try:
        rel = api_get(f"https://api.github.com/repos/{REPO}/releases/latest")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    ver = str(rel.get("tag_name") or "").lstrip("v")
    setup = None
    for a in rel.get("assets") or []:
        if re.match(r"^MatteDesk-.*-setup\.exe$", a.get("name", "")):
            dig = str(a.get("digest") or "")
            setup = {"url": a["browser_download_url"], "size": int(a.get("size") or 0),
                     "sha256": dig.split(":", 1)[1] if dig.startswith("sha256:") else None, "name": a["name"]}
    return {"version": ver, "notes": short_notes(rel.get("body") or ""), "url": rel.get("html_url") or RELEASES,
            "setup": setup}


def check(current: str, cfg: Dict[str, Any], app_dir: Path, force: bool = False, now: Optional[float] = None,
          api_get: Callable[[str], Dict[str, Any]] = _api,
          save: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """{available, current, version, notes, url, changelog, kind, can_install, skipped, enabled, error?}"""
    now = time.time() if now is None else now
    out: Dict[str, Any] = {"available": False, "current": current, "changelog": CHANGELOG_URL,
                           "kind": install_kind(app_dir), "enabled": cfg.get("update_check", True) is not False}
    if not out["enabled"] and not force:
        return out
    last = cfg.get("update_last") if isinstance(cfg.get("update_last"), dict) else {}
    fresh = bool(last) and not force and now - float(last.get("t") or 0) < DAY
    rel = last.get("release") if fresh else None
    if not fresh:
        try:
            rel = latest_release(api_get) or {}
        except Exception as e:   # offline, rate limit: try again next start
            out["error"] = str(e)
            return out
        if save:
            save({"update_last": {"t": now, "release": rel}})
    rel = rel or {}
    if not rel.get("version"):
        return out
    newer = parse_version(rel["version"]) > parse_version(current)
    out.update(version=rel["version"], notes=rel.get("notes", ""), url=rel.get("url", RELEASES),
               available=newer, skipped=newer and str(cfg.get("update_skip") or "") == rel["version"],
               can_install=newer and out["kind"] == "installer" and bool(rel.get("setup")),
               setup=rel.get("setup"))
    return out


STATE: Dict[str, Any] = {"state": "idle"}


def download_and_run(setup: Dict[str, Any], dest_dir: Optional[Path] = None,
                     download: Callable[..., Any] = dl.download, popen: Callable[..., Any] = subprocess.Popen) -> Path:
    """Download the new setup (resume + SHA-256) and start it (its own wizard; an update keeps the ComfyUI setting).
    The caller then stops the server so the setup can replace the files."""
    dest_dir = dest_dir or Path(tempfile.gettempdir()) / "MatteDesk-update"
    exe = dest_dir / setup.get("name", "MatteDesk-setup.exe")
    STATE.clear()
    STATE.update(state="downloading", done=0, total=int(setup.get("size") or 0))
    download(setup["url"], exe, sha256=setup.get("sha256"), progress=lambda d, t: STATE.update(done=d, total=t))
    STATE.update(state="starting")
    popen([str(exe), "/SP-"], close_fds=True)
    STATE.update(state="started", path=str(exe))
    return exe
