# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/outputs.py - `<work>/matte_desk/outputs.json` registry (data contract, SPEC 16).

Every module that produces something other tools may read (camera2d, deflicker maps, depth, mattes, .nk
Groups) records it here. Writes are atomic; unknown keys written by other modules are preserved.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from engine.jsonio import StateFileError, read_state, write_json  # noqa: F401 - StateFileError: its public name here too

OUTPUTS_SCHEMA = "matte_desk/outputs@1.0"


def load_outputs(work_dir: Union[str, Path]) -> Dict[str, Any]:
    """The registry; missing = empty. A corrupt file raises `StateFileError` (never `{}` + overwrite, audit 2026-11-02 #3)."""
    p = Path(work_dir) / "outputs.json"
    data: Dict[str, Any] = read_state(p) or {}
    data.setdefault("schema", OUTPUTS_SCHEMA)
    data.setdefault("objects", [])
    return data


def update_outputs(work_dir: Union[str, Path], key: str, value: Any) -> Dict[str, Any]:
    data = load_outputs(work_dir)
    data[key] = value
    return _save(work_dir, data)


def remove_outputs(work_dir: Union[str, Path], *keys: str) -> Dict[str, Any]:
    """Drop entries that no longer describe the export (e.g. the multi-EXR after a PNG export)."""
    data = load_outputs(work_dir)
    if not any(k in data for k in keys):
        return data
    for k in keys:
        data.pop(k, None)
    return _save(work_dir, data)


def _save(work_dir: Union[str, Path], data: Dict[str, Any]) -> Dict[str, Any]:
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p = Path(work_dir) / "outputs.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, data, indent=2, ensure_ascii=False)
    return data
