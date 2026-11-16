# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/deletion.py - the one way the program deletes project data (user decision 2026-11-13).

Deleting something (an object, a version, a key, a card, an anchor, a generation …) removes it from disk FOR GOOD –
nothing is moved to a project trash folder any more (disk use kept growing). The UI asks first with a clear warning
(permanent, cannot be undone, how much space it frees). Every deletion goes through `remove`, which returns the bytes
freed, so the API can report `freed_bytes`. Old `<work>/trash/` folders from before this decision are left alone – the
user decides about them separately; nothing here empties them.

  size_of(path) -> int        bytes of a file or a folder (0 when missing)
  remove(*paths) -> int       hard delete files / folders, returns bytes freed; missing paths are ignored; refuses an
                              empty path and a filesystem / drive root; `inside=` refuses anything outside that folder
  human(n) -> str             "1,2 GB" (Polish decimal comma)
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def size_of(path: Optional[PathLike]) -> int:
    """Bytes of a file, or of every file under a folder; 0 when missing (symlinks are not followed)."""
    if not path:
        return 0
    p = Path(path)
    try:
        if p.is_symlink() or p.is_file():
            return p.lstat().st_size
        if not p.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _check(p: Path, inside: Optional[Path]) -> Path:
    if not str(p).strip() or str(p).strip() in (".", ".."):
        raise ValueError("Usuwanie: pusta ścieżka")
    a = p.absolute()
    if a.parent == a or len(a.parts) < 2:
        raise ValueError(f"Usuwanie: odmowa – {a} to katalog główny dysku")
    if inside is not None:
        root = Path(inside).absolute()
        try:
            a.resolve().relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"Usuwanie: odmowa – {a} leży poza {root}")
        if a.resolve() == root.resolve():
            raise ValueError(f"Usuwanie: odmowa – {a} to sam folder nadrzędny")
    return a


def _writable(func, path, _exc) -> None:
    """rmtree error hook: a read-only file (Windows) is made writable and removed again."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def remove(*paths: Optional[PathLike], inside: Optional[PathLike] = None) -> int:
    """Delete files / folders from disk for good. Returns the bytes freed. Missing paths (and None) are ignored; an
    empty path, a drive root, or (with `inside`) anything outside that folder raises ValueError before anything goes."""
    root = Path(inside) if inside is not None else None
    checked = [_check(Path(p), root) for p in paths if p is not None]
    freed = 0
    for p in checked:
        if not (p.exists() or p.is_symlink()):
            continue
        n = size_of(p)
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, onerror=_writable)
        else:
            try:
                p.unlink()
            except PermissionError:
                os.chmod(p, stat.S_IWRITE)
                p.unlink()
        freed += n
    return freed


def human(n: Optional[int]) -> str:
    """Bytes as text with a Polish decimal comma: 0 B, 512 B, 3,4 kB, 12,0 MB, 1,2 GB."""
    n = int(n or 0)
    if n < 1024:
        return f"{n} B"
    for unit in ("kB", "MB", "GB", "TB"):
        n_f = n / 1024.0
        if n_f < 1024 or unit == "TB":
            return f"{n_f:.1f} {unit}".replace(".", ",")
        n = n_f
    return f"{n} B"
