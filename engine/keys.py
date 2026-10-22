# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/keys.py - Building user keys and the undo/redo history (SPEC 7.1, 6 pkt 5, stage E4).

A key = full, hard object mask in one frame, built as: current core of the frame (or a SAM 3.1 preview mask)
+/- brush strokes / polygons (plate px). Saving or deleting a key is recorded in `history.json` with the previous
state of the key files, so Ctrl+Z / Ctrl+Shift+Z restore exactly. Strokes and points before saving are client-side.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from engine.objects import ObjectStore


_SH = 4                      # sub-pixel bits for the variable-radius strokes (cv2 `shift`)


def has_point_radius(points: Sequence[Sequence[float]]) -> bool:
    return any(len(q) >= 3 for q in points)


def draw_variable_stroke(m: np.ndarray, points: Sequence[Sequence[float]], default_r: float, val: int) -> None:
    """Stroke with a radius per point ([[x, y, r], ...], Z23 – pen pressure / size changes): every pair of neighbours =
    the convex hull of their two discs ("capsule" with outer tangents), so the key equals what the preview draws."""
    k = float(1 << _SH)
    pts = [(float(q[0]), float(q[1]), max(0.5, float(q[2]) if len(q) >= 3 else float(default_r))) for q in points]
    disc = lambda x, y, r: cv2.circle(m, (int(round(x * k)), int(round(y * k))), int(round(r * k)), val, -1, cv2.LINE_8, _SH)
    for x, y, r in pts:
        disc(x, y, r)
    for (x0, y0, r0), (x1, y1, r1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        d = float(np.hypot(dx, dy))
        if d <= abs(r1 - r0) + 1e-6:
            continue                                   # one disc contains the other: the discs are the hull
        ux, uy = dx / d, dy / d
        nx, ny = -uy, ux
        sa = (r0 - r1) / d
        ca = float(np.sqrt(max(0.0, 1.0 - sa * sa)))
        quad = []
        for sgn in (1.0, -1.0):
            vx, vy = sgn * nx * ca + ux * sa, sgn * ny * ca + uy * sa
            quad.append((x0 + r0 * vx, y0 + r0 * vy, x1 + r1 * vx, y1 + r1 * vy))
        poly = np.array([[quad[0][0], quad[0][1]], [quad[0][2], quad[0][3]], [quad[1][2], quad[1][3]], [quad[1][0], quad[1][1]]])
        cv2.fillConvexPoly(m, np.round(poly * k).astype(np.int32), val, cv2.LINE_8, _SH)


def apply_strokes(base: np.ndarray, strokes: Sequence[Dict[str, Any]] = (), polygons: Sequence[Dict[str, Any]] = ()
                  ) -> np.ndarray:
    """strokes: [{"mode": "add|remove", "radius_px": r, "p": [[x, y], ...]}] – constant radius (unchanged, bit for bit),
    or "p": [[x, y, r], ...] – radius per point (Z23, pen pressure); polygons: [{"mode", "p"}]."""
    m = (base > 0.5).astype(np.uint8)
    for s in strokes:
        val = 1 if s.get("mode", "add") == "add" else 0
        if has_point_radius(s.get("p", [])):
            draw_variable_stroke(m, s.get("p", []), s.get("radius_px", 8), val)
            continue
        r = max(1, int(round(s.get("radius_px", 8))))
        pts = np.round(np.array(s.get("p", []), np.float32)).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(m, tuple(pts[0]), r, val, -1)
        elif len(pts) > 1:
            cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, val, thickness=2 * r, lineType=cv2.LINE_8)
            for p in pts:
                cv2.circle(m, tuple(p), r, val, -1)
    for poly in polygons:
        val = 1 if poly.get("mode", "add") == "add" else 0
        pts = np.round(np.array(poly.get("p", []), np.float32)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], val)
    return m.astype(bool)


class KeyHistory:
    """Undo / redo of key saves and deletions for one object (`objects/<obj>/history.json`)."""

    def __init__(self, store: ObjectStore, name: str):
        self.store = store
        self.name = name
        self.path = store.dir(name) / "history.json"

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"undo": [], "redo": []}

    def _save(self, h: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")

    def _snapshot(self, frame: int) -> Optional[Dict[str, Any]]:
        kd = self.store.dir(self.name) / "keys"
        jp, mp = kd / f"k_{frame:04d}.json", kd / f"k_{frame:04d}_mask.png"
        if not jp.exists():
            return None
        return {"json": json.loads(jp.read_text(encoding="utf-8")),
                "mask_png_b64": base64.b64encode(mp.read_bytes()).decode() if mp.exists() else None}

    def _restore(self, frame: int, snap: Optional[Dict[str, Any]]) -> None:
        kd = self.store.dir(self.name) / "keys"
        kd.mkdir(parents=True, exist_ok=True)
        if snap is None:
            self.store.delete_key(self.name, frame)
            return
        (kd / f"k_{frame:04d}.json").write_text(json.dumps(snap["json"], indent=2, ensure_ascii=False), encoding="utf-8")
        if snap.get("mask_png_b64"):
            (kd / f"k_{frame:04d}_mask.png").write_bytes(base64.b64decode(snap["mask_png_b64"]))

    def record(self, op: str, frame: int, before: Optional[Dict[str, Any]]) -> None:
        h = self._load()
        h["undo"].append({"op": op, "frame": frame, "before": before, "after": self._snapshot(frame),
                          "t": datetime.now().isoformat(timespec="seconds")})
        h["undo"] = h["undo"][-200:]
        h["redo"] = []
        self._save(h)

    def save_key(self, frame: int, mask: np.ndarray, author: str = "user", **extra: Any) -> Dict[str, Any]:
        before = self._snapshot(frame)
        key = self.store.save_key(self.name, frame, mask, author=author, **extra)
        self.record("save_key", frame, before)
        return key

    def delete_key(self, frame: int) -> None:
        before = self._snapshot(frame)
        self.store.delete_key(self.name, frame)
        self.record("delete_key", frame, before)

    def undo(self) -> Optional[Dict[str, Any]]:
        h = self._load()
        if not h["undo"]:
            return None
        e = h["undo"].pop()
        self._restore(e["frame"], e["before"])
        h["redo"].append(e)
        self._save(h)
        return {"op": e["op"], "frame": e["frame"], "undone": True}

    def redo(self) -> Optional[Dict[str, Any]]:
        h = self._load()
        if not h["redo"]:
            return None
        e = h["redo"].pop()
        self._restore(e["frame"], e["after"])
        h["undo"].append(e)
        self._save(h)
        return {"op": e["op"], "frame": e["frame"], "redone": True}
