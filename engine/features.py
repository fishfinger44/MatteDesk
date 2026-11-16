# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/features.py - parts of the program switched off in a release (user decision 2026-11-16).

`feature_cleanup3d` (config.json / config.local.json, default on): Cleanup 3D = the steps C2 Kamera 3D · C3 Karty ·
C4 Klucze · C5 Projekcja · C6 Eksport of the „Czyszczenie” mode. The public copy (tools/export_public.py) writes it
off until Cleanup 3D is finished; C1 Inpaint stays. Off = the steps are hidden in the stage bar (static/cleanup3d.js
reads `GET /api/version` → features), their routes answer 404 (`/api/cleanup3d/...`; `GET /api/cleanup3d` itself
stays – the stage bar reads C1's state from it) and the Doctor hides the tools only those steps use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

FEATURES = ("cleanup3d",)
OFF_WHY = {"cleanup3d": "Cleanup 3D (kamera 3D, karty, klucze, projekcja) jest wyłączone w tym wydaniu – "
                        "włączysz je w config.local.json: \"feature_cleanup3d\": true"}


def enabled(name: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    if cfg is None:
        from engine.config import load_config
        cfg = load_config()
    return bool(cfg.get(f"feature_{name}", True))


def all_features(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    if cfg is None:
        from engine.config import load_config
        cfg = load_config()
    return {n: enabled(n, cfg) for n in FEATURES}
