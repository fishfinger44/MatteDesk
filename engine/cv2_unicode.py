# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unicode-safe cv2.imread / cv2.imwrite (Windows).

OpenCV on Windows opens files through the ANSI API, so any path with characters outside the system code page
(e.g. Polish "październik") makes cv2.imread return None and cv2.imwrite return False - silently. MatteDesk has
~70 call sites, so instead of touching each one, `install()` (called from engine/__init__.py) wraps both functions
once: plain ASCII paths keep the native call, other paths go through np.fromfile + cv2.imdecode /
cv2.imencode + ndarray.tofile. Flags and imwrite params are passed through unchanged.
"""
from __future__ import annotations

import os

import numpy as np

_installed = False


def _is_ascii(path) -> bool:
    try:
        os.fspath(path).encode("ascii")
        return True
    except (UnicodeEncodeError, TypeError):
        return False


def install() -> None:
    global _installed
    if _installed:
        return
    import cv2
    native_read, native_write = cv2.imread, cv2.imwrite

    def imread(filename, flags=cv2.IMREAD_COLOR):
        if _is_ascii(filename):
            return native_read(filename, flags)
        try:
            buf = np.fromfile(os.fspath(filename), dtype=np.uint8)
        except OSError:
            return None
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)

    def imwrite(filename, img, params=None):
        if _is_ascii(filename):
            return native_write(filename, img, params) if params is not None else native_write(filename, img)
        name = os.fspath(filename)
        ext = os.path.splitext(name)[1] or ".png"
        ok, buf = cv2.imencode(ext, img, params) if params is not None else cv2.imencode(ext, img)
        if not ok:
            return False
        try:
            buf.tofile(name)
        except OSError:
            return False
        return True

    imread.__wrapped__, imwrite.__wrapped__ = native_read, native_write
    cv2.imread, cv2.imwrite = imread, imwrite
    _installed = True
