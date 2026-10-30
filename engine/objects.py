# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/objects.py - Objects of a shot (SPEC 4): `<work>/matte_desk/objects/<obj>/`.

object.json  – prompt, flags, depth_order, edge settings, tool state (extension of manifest objects[])
keys/        – k_####.json + k_####_mask.png; author=user is never overwritten by automation
crops.json   – crop engine result (engine.crop)
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

OBJECT_SCHEMA = "matte_desk/object@1.0"
KEY_SCHEMA = "matte_desk/key@1.0"
_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _defaults_object() -> Dict[str, Any]:
    try:
        d = json.loads((Path(__file__).resolve().parent.parent / "defaults.json").read_text(encoding="utf-8"))
        return dict(d.get("object", {}))
    except (OSError, ValueError):
        return {}


class ObjectStore:
    def __init__(self, work_dir: Union[str, Path]):
        self.root = Path(work_dir) / "objects"
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ objects
    def dir(self, name: str) -> Path:
        if not _NAME.match(name):
            raise ValueError(f"niepoprawna nazwa obiektu: {name!r} (litery, cyfry, _ -)")
        return self.root / name

    def create(self, name: str, prompts: Optional[List[str]] = None, depth_order: Optional[int] = None,
               flags: Optional[List[str]] = None, detect: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        d = self.dir(name)
        if (d / "object.json").exists():
            raise FileExistsError(f"obiekt {name} już istnieje")
        defs = _defaults_object()
        if depth_order is None:
            depth_order = len(self.list())
        obj = {
            "schema": OBJECT_SCHEMA, "name": name, "depth_order": int(depth_order),
            "detect": detect or {"type": "text", "prompts": prompts or []},
            "flags": flags or [],
            "edge_mode": defs.get("edge_mode", "auto"), "edge_mode_auto": "", "use_backlit": defs.get("use_backlit", True),
            "band_scale": defs.get("band_scale", 1.0), "shift_px": defs.get("shift_px", 0.0),
            "temporal_smoothing": defs.get("temporal_smoothing", "normal"), "thin_lens": defs.get("thin_lens", "auto"),
            "ma_scale": defs.get("ma_scale", 1.0), "ma_scale_mode": defs.get("ma_scale_mode", "auto"),
            "core_stabilize": bool(defs.get("core_stabilize", True)),
            "correction_region_only": bool(defs.get("correction_region_only", True)),
            "band": {"band_points": [], "band_ref_out": None, "band_ref_in": None}, "band_modes": {},   # engine/objband.py
            "tool": {"active_run": "", "final_valid": False, "auto_attempts": 0, "locked_frames": [],
                     "status": "nowy", "depends_on": [], "start_frame": None},
        }
        for sub in ("", "keys", "zones", "runs", "cache"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        self._write(d / "object.json", obj)
        return obj

    def get(self, name: str) -> Dict[str, Any]:
        p = self.dir(name) / "object.json"
        if not p.exists():
            raise FileNotFoundError(f"brak obiektu {name}")
        obj = self._read(p)
        if not isinstance(obj.get("band"), dict):   # the shared object band (2026-10-22): migrated on the first read
            from engine.objband import migrate
            migrate(self.root, name)
            obj = self._read(p)
        root = obj if not obj.get("part_of") else None
        if root is not None and "out_px" not in (obj.get("band") or {}):   # the sliders join the band (2026-10-24)
            from engine.objband import migrate_sliders
            migrate_sliders(self.root, name)
            obj = self._read(p)
        return obj

    def update(self, name: str, **fields: Any) -> Dict[str, Any]:
        obj = self.get(name)
        for k, v in fields.items():
            if k == "tool":
                obj["tool"].update(v)
            else:
                obj[k] = v
        self._write(self.dir(name) / "object.json", obj)
        return obj

    def remove(self, name: str) -> Path:
        """Delete an object = move its folder to <work>/trash/objects/<name>_<time> (recoverable by hand), then
        renumber depth_order of the rest (same order, no gaps) and drop the name from their depends_on."""
        import shutil
        import time
        d = self.dir(name)
        if not (d / "object.json").exists():
            raise FileNotFoundError(f"brak obiektu {name}")
        trash = self.root.parent / "trash" / "objects" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        gone = self._read(d / "object.json")
        shutil.move(str(d), str(trash))
        for g in self.list(groups=True):   # a group entity (engine/groups.py): a deleted member leaves it
            if g.get("kind") == "group" and name in (g.get("members") or []):
                self.update(g["name"], members=[m for m in g["members"] if m != name])
        if gone.get("kind") == "group":    # a deleted group: its members become standalone
            for m in gone.get("members") or []:
                if (self.dir(m) / "object.json").exists():
                    self.update(m, group=None, group_source=None, group_on=None, group_seam_px=None)
        for i, o in enumerate(self.list()):
            deps = [x for x in o["tool"].get("depends_on", []) if x != name]
            if o.get("depth_order") != i or deps != o["tool"].get("depends_on", []):
                self.update(o["name"], depth_order=i, tool={"depends_on": deps})
        return trash

    def rename(self, old: str, new: str) -> int:
        """Rename an object (user request 2026-10-16): move objects/<old> to objects/<new>, then replace every JSON
        string that IS the old name (exact match, never a substring – prompts stay) in all objects' files: its own
        name, group / members / depends_on of other objects, groups in run.json, front objects (holdouts) in zones.
        Returns the number of JSON files rewritten. Exported files keep their old names on disk."""
        src, dst = self.dir(old), self.dir(new)
        if not (src / "object.json").exists():
            raise FileNotFoundError(f"brak obiektu {old}")
        if dst.exists():
            raise FileExistsError(f"obiekt {new} już istnieje")
        src.rename(dst)

        def sub(x: Any) -> Any:
            if isinstance(x, str):
                return new if x == old else x
            if isinstance(x, list):
                return [sub(v) for v in x]
            if isinstance(x, dict):
                return {(new if k == old else k): sub(v) for k, v in x.items()}
            return x

        n = 0
        for p in sorted(self.root.glob("*/**/*.json")) + sorted(self.root.glob("*/*.json")):
            try:
                raw = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if f'"{old}"' not in raw:
                continue
            data = json.loads(raw)
            new_data = sub(data)
            if new_data != data:
                self._write(p, new_data)
                n += 1
        return n

    def list(self, groups: bool = False) -> List[Dict[str, Any]]:
        """Objects in depth order. Group entities (`kind: group`, engine/groups.py) only with `groups=True`."""
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            p = d / "object.json"
            if p.exists():
                o = self._read(p)
                if o.get("kind") == "group" and not groups:
                    continue
                out.append(o if isinstance(o.get("band"), dict) else self.get(d.name))
        return sorted(out, key=lambda o: o.get("depth_order", 0))

    # ------------------------------------------------------------------ keys
    def save_key(self, name: str, frame: int, mask: np.ndarray, author: str = "user", kind: str = "mask",
                 **extra: Any) -> Dict[str, Any]:
        """Full-frame binary key mask. An existing `user` key is never overwritten by `auto`."""
        kd = self.dir(name) / "keys"
        kp = kd / f"k_{frame:04d}.json"
        if kp.exists() and author == "auto":
            old = json.loads(kp.read_text(encoding="utf-8"))
            if old.get("author") == "user":
                return old
        kd.mkdir(parents=True, exist_ok=True)
        mpath = kd / f"k_{frame:04d}_mask.png"
        cv2.imwrite(str(mpath), ((mask > 0.5) * 255).astype(np.uint8))
        key = {"schema": KEY_SCHEMA, "frame": int(frame), "kind": kind, "author": author,
               "t": datetime.now().isoformat(timespec="seconds"), "mask_path": f"keys/{mpath.name}",
               "hardness": "hard", **extra}
        self._write(kp, key)
        return key

    def keys(self, name: str) -> Dict[int, Dict[str, Any]]:
        kd = self.dir(name) / "keys"
        out = {}
        for p in sorted(kd.glob("k_*.json")) if kd.exists() else []:
            k = json.loads(p.read_text(encoding="utf-8"))
            out[int(k["frame"])] = k
        return out

    def key_mask(self, name: str, frame: int) -> np.ndarray:
        m = cv2.imread(str(self.dir(name) / "keys" / f"k_{frame:04d}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"brak maski klucza {name} k_{frame:04d}")
        return m > 127

    def delete_key(self, name: str, frame: int) -> None:
        kd = self.dir(name) / "keys"
        for p in (kd / f"k_{frame:04d}.json", kd / f"k_{frame:04d}_mask.png", kd / f"k_{frame:04d}_delta.png"):
            if p.exists():
                p.unlink()

    @staticmethod
    def _read(p: Path) -> Dict[str, Any]:
        """object.json (JSON); Windows: opening fails (errno 13) for a moment while another thread replaces the file –
        retried like the replace in _write (2026-10-26)."""
        import time
        for i in range(8):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except PermissionError:
                if i == 7:
                    raise
                time.sleep(0.05 * (i + 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _write(p: Path, data: Dict[str, Any]) -> None:
        import threading
        import time
        # a tmp of its own per thread: two threads writing the same object (a request + the mask 1+2 background batch,
        # both migrating on the first read) overwrote one shared .tmp (test_zone_outline flaky, 2026-10-26)
        tmp = p.with_name(f"{p.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        for i in range(8):   # Windows: the replace fails (WinError 5) while another process reads the file – a second
            try:             # server, a viewer, an antivirus (seen 2026-10-24: a slider save lost with a 500)
                tmp.replace(p)
                return
            except PermissionError:
                if i == 7:
                    raise
                time.sleep(0.05 * (i + 1))
