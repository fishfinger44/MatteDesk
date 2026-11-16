# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/lru.py - The small in-memory cache used next to a running job: sync endpoints run in a thread pool, so get /
put / eviction happen in parallel (audit 2026-11-02 #20). Values are shared, not copied – callers copy what they change."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable


class LockedLRU:
    def __init__(self, maxsize: int):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            if key not in self._d:
                return default
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._d
