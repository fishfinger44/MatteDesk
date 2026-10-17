# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/download.py - one downloader for everything MatteDesk fetches (models, ComfyUI portable, installer parts).

- **Resume:** data goes to `<file>.part`; an interrupted download continues from there (HTTP Range). A server that
  ignores the Range starts over; 416 on a complete `.part` finishes it.
- **Checksum:** SHA-256 given by the caller, or the one the server publishes – Hugging Face LFS files
  (`X-Linked-Etag` on the huggingface.co response, read with a HEAD that does not follow the CDN redirect) and
  GitHub release assets (`digest` in the releases API, see `github_asset`). A mismatch deletes the `.part`
  (a corrupted part must not be resumed) and raises.
- No third-party packages (urllib only): the installer runs it with the bundled Python before anything else exists.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

UA = "MatteDeskInstaller"
CHUNK = 8 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

Progress = Callable[[int, int], None]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):   # keep the first response (its headers carry the LFS hash)
        return None


def hf_sha256(url: str, token: Optional[str] = None, opener: Any = None) -> Optional[str]:
    """SHA-256 of a Hugging Face LFS file (`.../resolve/...`) from the `X-Linked-Etag` header; None when unknown."""
    if "huggingface.co/" not in url or "/resolve/" not in url:
        return None
    hdr = {"User-Agent": UA}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    op = opener or urllib.request.build_opener(_NoRedirect)
    try:
        r = op.open(urllib.request.Request(url, headers=hdr, method="HEAD"), timeout=30)
        headers = r.headers
    except urllib.error.HTTPError as e:     # 302 without following = HTTPError carrying the headers
        headers = e.headers
    except (urllib.error.URLError, OSError):
        return None
    etag = (headers.get("X-Linked-Etag") or "").strip().strip('"').lower() if headers else ""
    return etag if _HEX64.match(etag) else None


def github_asset(repo: str, name: str, tag: str = "latest", api_get: Optional[Callable[[str], Dict[str, Any]]] = None
                 ) -> Dict[str, Any]:
    """{url, size, sha256, tag} of a release asset (GitHub publishes `digest: sha256:<hex>` per asset)."""
    path = "latest" if tag == "latest" else f"tags/{tag}"
    api = f"https://api.github.com/repos/{repo}/releases/{path}"
    if api_get is None:
        def api_get(u: str) -> Dict[str, Any]:
            req = urllib.request.Request(u, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
    rel = api_get(api)
    for a in rel.get("assets") or []:
        if a.get("name") == name:
            dig = str(a.get("digest") or "")
            sha = dig.split(":", 1)[1].lower() if dig.startswith("sha256:") else None
            return {"url": a["browser_download_url"], "size": int(a.get("size") or 0), "sha256": sha,
                    "tag": rel.get("tag_name") or tag}
    raise RuntimeError(f"{repo} {rel.get('tag_name') or tag}: brak pliku {name} w wydaniu")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, sha256: Optional[str] = None, token: Optional[str] = None,
             progress: Optional[Progress] = None, verify_server_hash: bool = True,
             urlopen: Callable[..., Any] = urllib.request.urlopen) -> Path:
    """Download `url` to `dst` with resume and checksum. `sha256=None` -> the server's hash when it publishes one."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sha256 is None and verify_server_hash:
        sha256 = hf_sha256(url, token)
    have = tmp.stat().st_size if tmp.exists() else 0
    hdr = {"User-Agent": UA}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    if have:
        hdr["Range"] = f"bytes={have}-"
    try:
        r = urlopen(urllib.request.Request(url, headers=hdr), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and "huggingface.co" in url:
            raise RuntimeError("Hugging Face odmówił dostępu (401/403): zaloguj się („hf auth login”) i kliknij "
                               "„Agree and Access” na stronie modelu (repo gated).") from None
        if e.code == 416 and have:          # the .part is already complete
            return _finish(tmp, dst, sha256)
        raise
    with r:
        status = getattr(r, "status", 200)
        if have and status != 206:          # server ignored the Range: start over
            have = 0
        total = int(r.headers.get("Content-Length") or 0) + have
        done = have
        with open(tmp, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total and tmp.stat().st_size != total:
        raise RuntimeError(f"{dst.name}: pobrane {tmp.stat().st_size} z {total} B – spróbuj ponownie (wznowi)")
    return _finish(tmp, dst, sha256)


def _finish(tmp: Path, dst: Path, sha256: Optional[str]) -> Path:
    if sha256:
        got = sha256_file(tmp)
        if got != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{dst.name}: suma SHA-256 się nie zgadza (oczekiwana {sha256[:12]}…, jest {got[:12]}…) "
                               f"– plik usunięty, pobierz ponownie")
    tmp.replace(dst)
    return dst
