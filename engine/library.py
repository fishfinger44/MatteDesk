# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/library.py - Object library shared between shots (SPEC 6.1): `<repo>/library/<project>/<obj>.json`.

Stores name, prompts, edge mode, band width, smoothing, flags, lupa settings and 1-3 thumbnails (to recognise the
instance). Keys and zones are NOT carried over (they belong to a shot).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

LIB_SCHEMA = "matte_desk/library_object@1.0"
FIELDS = ("detect", "flags", "edge_mode", "use_backlit", "band_scale", "shift_px", "temporal_smoothing", "thin_lens")


def library_root() -> Path:
    return Path(__file__).resolve().parent.parent / "library"


def save_to_library(obj: Dict[str, Any], project: str, thumbs: Sequence[np.ndarray] = (), root: Optional[Path] = None) -> Path:
    d = (root or library_root()) / project
    d.mkdir(parents=True, exist_ok=True)
    enc = []
    for t in list(thumbs)[:3]:
        ok, buf = cv2.imencode(".jpg", (np.clip(t, 0, 1) * 255).astype(np.uint8) if t.dtype != np.uint8 else t,
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            enc.append(base64.b64encode(buf.tobytes()).decode())
    data = {"schema": LIB_SCHEMA, "name": obj["name"], "project": project, "saved": datetime.now().isoformat(timespec="seconds"),
            **{k: obj.get(k) for k in FIELDS}, "lupa_box": None, "thumbnails_jpg_b64": enc}
    p = d / f"{obj['name']}.json"
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def list_library(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    r = root or library_root()
    out = []
    for p in sorted(r.glob("*/*.json")) if r.exists() else []:
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": f"{d['project']}/{d['name']}", "name": d["name"], "project": d["project"],
                    "prompts": (d.get("detect") or {}).get("prompts", []), "edge_mode": d.get("edge_mode"),
                    "thumbs": len(d.get("thumbnails_jpg_b64", []))})
    return out


def load_library_item(item_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    project, name = item_id.split("/", 1)
    return json.loads(((root or library_root()) / project / f"{name}.json").read_text(encoding="utf-8"))


def add_from_library(store: Any, item_id: str, new_name: Optional[str] = None, root: Optional[Path] = None) -> Dict[str, Any]:
    d = load_library_item(item_id, root)
    name = new_name or d["name"]
    det = d.get("detect") or {}
    obj = store.create(name, det.get("prompts", []), flags=d.get("flags") or [], detect=det)
    upd = {k: d[k] for k in FIELDS if k not in ("detect", "flags") and d.get(k) is not None}
    return store.update(name, **upd, tool={"library_source": item_id})


def thumbnail_similarity(candidate: np.ndarray, item: Dict[str, Any]) -> float:
    """Colour-histogram similarity of a candidate crop to the stored thumbnails (instance hint, 6.1)."""
    best = 0.0
    c = (np.clip(candidate, 0, 1) * 255).astype(np.uint8) if candidate.dtype != np.uint8 else candidate
    hc = cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hc, hc)
    for b64 in item.get("thumbnails_jpg_b64", []):
        t = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        ht = cv2.calcHist([cv2.cvtColor(t, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(ht, ht)
        best = max(best, float(cv2.compareHist(hc, ht, cv2.HISTCMP_CORREL)))
    return best
