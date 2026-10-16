# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/project.py - Shot project state, kept apart from the user's shot manifest.

Rules (CLAUDE.md, SPEC sections 4, 16):
- `shot_manifest_<shot>.json` is user input and READ-ONLY. The tool never writes to an existing manifest.
  A manifest is written only by the new-shot wizard, when none exists yet (`ProjectCreator.create_manifest`).
- Everything the tool decides or adds (extra plates, `_align` / `_deflicker` variants, hidden plates)
  lives in `<work>/<work_subdir>/project.json` (default subdir `matte_desk`, from defaults.json).
- The rest of the app reads an *effective manifest*: the manifest with project.json overrides applied
  (manifest dict is deep-copied, never mutated).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

PROJECT_SCHEMA = "matte_desk/project@1.0"
DEFAULT_WORK_SUBDIR = "matte_desk"


def load_defaults(defaults_path: Optional[Path] = None) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent.parent
    p = Path(defaults_path) if defaults_path else repo_dir / "defaults.json"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def work_subdir_name(defaults: Optional[Dict[str, Any]] = None) -> str:
    d = defaults if defaults is not None else load_defaults()
    return str(d.get("project", {}).get("work_subdir", DEFAULT_WORK_SUBDIR)) or DEFAULT_WORK_SUBDIR


def resolve_plate_size(manifest: Dict[str, Any]) -> Tuple[int, int]:
    """Plate resolution from the manifest (`plate_w` / `plate_h`).

    No hard-coded fallback: if the manifest lacks the fields, the beauty plate header is read.
    Raises ValueError when the size cannot be determined.
    """
    w, h = manifest.get("plate_w"), manifest.get("plate_h")
    if w and h:
        return int(w), int(h)
    beauty = (manifest.get("plates") or {}).get("beauty", "")
    if beauty:
        from engine.ingest import MediaInspector  # local import: ingest imports cv2 at module load

        info = MediaInspector.resolve_sequence_pattern(beauty)
        if info.get("width") and info.get("height"):
            return int(info["width"]), int(info["height"])
    raise ValueError("Manifest nie podaje rozdzielczości (plate_w / plate_h) i nie da się jej odczytać z plate'u beauty")


class ShotProject:
    """A shot opened in MatteDesk: read-only manifest + tool state in project.json."""

    def __init__(self, manifest_path: Union[str, Path], manifest: Dict[str, Any], work_dir: Path, data: Dict[str, Any]):
        self.manifest_path = Path(manifest_path)
        self._manifest = manifest
        self.work_dir = Path(work_dir)
        self.data = data

    # ------------------------------------------------------------------ open / save
    @classmethod
    def open(cls, manifest_path: Union[str, Path], defaults: Optional[Dict[str, Any]] = None) -> "ShotProject":
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        defaults = defaults if defaults is not None else load_defaults()
        work_root = Path(manifest.get("work") or m_path.parent)
        work_dir = work_root / work_subdir_name(defaults)
        for sub in ("", "proxy", "objects", "export"):
            (work_dir / sub).mkdir(parents=True, exist_ok=True)

        pj = work_dir / "project.json"
        data: Dict[str, Any] = {}
        if pj.exists():
            try:
                with open(pj, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = {}
        data.setdefault("schema", PROJECT_SCHEMA)
        data.setdefault("plates_added", {})
        data.setdefault("plates_hidden", [])
        data["manifest_path"] = str(m_path.resolve()).replace("\\", "/")
        data["work_dir"] = str(work_dir.resolve()).replace("\\", "/")

        proj = cls(m_path, manifest, work_dir, data)
        proj._refresh_summary(defaults)
        proj.save()
        return proj

    def _refresh_summary(self, defaults: Dict[str, Any]) -> None:
        m = self._manifest
        plate_w, plate_h = resolve_plate_size(self.effective_manifest())
        self.data.update({
            "shot": m.get("shot", "unknown"),
            "colorspace": m.get("colorspace", defaults.get("project", {}).get("colorspace", "ACES2065-1")),
            "fps": m.get("fps", 24),
            "first_frame": int(m.get("first", 1)),
            "last_frame": int(m.get("last", 1)),
            "plate_w": plate_w,
            "plate_h": plate_h,
        })

    def save(self) -> Path:
        pj = self.work_dir / "project.json"
        tmp = pj.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(pj)
        return pj

    # ------------------------------------------------------------------ views
    @property
    def manifest(self) -> Dict[str, Any]:
        """Raw manifest (copy – callers cannot mutate the original)."""
        return copy.deepcopy(self._manifest)

    def effective_manifest(self) -> Dict[str, Any]:
        m = copy.deepcopy(self._manifest)
        plates = dict(m.get("plates") or {})
        for name in self.data.get("plates_hidden", []):
            plates.pop(name, None)
        plates.update(self.data.get("plates_added", {}))
        m["plates"] = plates
        return m

    @property
    def plate_size(self) -> Tuple[int, int]:
        return int(self.data["plate_w"]), int(self.data["plate_h"])

    def plate_origin(self, name: str) -> str:
        if name in self.data.get("plates_added", {}):
            return "project"
        if name in (self._manifest.get("plates") or {}):
            return "manifest"
        return ""

    # ------------------------------------------------------------------ plate edits (project.json only)
    def set_plate(self, name: str, pattern: str) -> None:
        self.data.setdefault("plates_added", {})[name] = pattern
        hidden = self.data.setdefault("plates_hidden", [])
        if name in hidden:
            hidden.remove(name)
        self.save()

    def remove_plate(self, name: str) -> None:
        added = self.data.setdefault("plates_added", {})
        added.pop(name, None)
        if name in (self._manifest.get("plates") or {}):
            hidden = self.data.setdefault("plates_hidden", [])
            if name not in hidden:
                hidden.append(name)
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
