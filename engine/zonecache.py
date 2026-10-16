# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/zonecache.py - cache of `matte` zone results per frame (docs/PLAN_HAIR_TOOL.md section 5, step 4).

Key = hash of (zone definition incl. keys / engine / params, engine version, core of the frame, plate frame signature,
region of the frame); per-frame engines (auto / clean / chroma / knn …) cache every frame on its own, the sequence engine
`matanyone` caches the whole range under one hash (its result depends on every frame). Stored as the bbox crop of the
zone alpha in 16-bit PNG + a small JSON next to it: `<obj>/zones_cache/<zone_id>/<hash>/a_####.png`. "Zastosuj" /
"Przelicz finalnie" compute only frames whose hash is missing. Transforms of the propagation (CoTracker3) are cached
too (`transforms_<hash>.npz`) so `GET …/region/{f}` is instant after the first pass.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

ENGINE_VERSION = "localmatte-1"     # bump when an engine's numerics change (invalidates every cached zone)


def _sha(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def zone_signature(zone_dict: Dict[str, Any]) -> str:
    """Hash of everything in the zone that changes the result (not `enabled`, not the id)."""
    d = {k: v for k, v in zone_dict.items() if k not in ("enabled", "zone_id", "schema", "prompt_history")}
    return _sha(d, ENGINE_VERSION)[:16]


def frame_hash(zone_sig: str, core: np.ndarray, plate_sig: Any, region: np.ndarray, extra: Any = None) -> str:
    return _sha(zone_sig, core.astype(np.uint8), plate_sig, region.astype(np.uint8), extra)[:16]


def sequence_hash(zone_sig: str, cores: Sequence[np.ndarray], plate_sigs: Sequence[Any], regions: Sequence[np.ndarray],
                  extra: Any = None) -> str:
    h = hashlib.sha1(zone_sig.encode())
    for c, p, r in zip(cores, plate_sigs, regions):
        h.update(frame_hash("", c, p, r).encode())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class ZoneCache:
    def __init__(self, obj_dir: Path, zone_id: str):
        self.dir = Path(obj_dir) / "zones_cache" / zone_id
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ alpha per frame
    def _paths(self, h: str, f: int) -> Tuple[Path, Path]:
        d = self.dir / h
        return d / f"a_{f:04d}.png", d / f"a_{f:04d}.json"

    def get(self, h: str, f: int, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        p, j = self._paths(h, f)
        if not p.exists() or not j.exists():
            self.misses += 1
            return None
        meta = json.loads(j.read_text(encoding="utf-8"))
        a = np.zeros(shape, np.float32)
        x0, y0, x1, y1 = meta["bbox"]
        if x1 > x0 and y1 > y0:
            crop = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if crop is None or crop.shape[:2] != (y1 - y0, x1 - x0):
                self.misses += 1
                return None
            a[y0:y1, x0:x1] = crop.astype(np.float32) / 65535.0
        self.hits += 1
        return a

    def put(self, h: str, f: int, alpha: np.ndarray, area: np.ndarray) -> None:
        p, j = self._paths(h, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        ys, xs = np.where(area)
        if len(xs):
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            crop = (np.clip(alpha[y0:y1, x0:x1], 0, 1) * 65535.0 + 0.5).astype(np.uint16)
            cv2.imwrite(str(p), crop)
        else:
            x0 = y0 = x1 = y1 = 0
            cv2.imwrite(str(p), np.zeros((1, 1), np.uint16))
        j.write_text(json.dumps({"bbox": [x0, y0, x1, y1], "frame": f, "hash": h}), encoding="utf-8")

    def has(self, h: str, f: int) -> bool:
        p, j = self._paths(h, f)
        return p.exists() and j.exists()

    def prune(self, keep_hashes: Sequence[str]) -> int:
        """Remove cached results of other hashes (old definitions of the zone). Returns the number of folders removed."""
        n = 0
        if not self.dir.exists():
            return 0
        for d in self.dir.iterdir():
            if d.is_dir() and d.name not in keep_hashes and not d.name.startswith("transforms"):
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n

    # ------------------------------------------------------------------ propagation transforms
    def get_transforms(self, h: str) -> Optional[Dict[int, np.ndarray]]:
        p = self.dir / f"transforms_{h}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return {int(k): z[k] for k in z.files}

    def put_transforms(self, h: str, tr: Dict[int, np.ndarray]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.dir / f"transforms_{h}.npz", **{str(f): np.asarray(m, np.float32) for f, m in tr.items()})

    def get_tracks(self, h: str) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
        """Local zone tracking (2026-10-15): key frame -> {pts (N,2), dst (T,N,2), vis (T,N), fallback (T,2,3)}."""
        p = self.dir / f"tracks_{h}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        keys = sorted({n.split("_", 1)[0] for n in z.files})
        return {int(k): {f: z[f"{k}_{f}"] for f in ("pts", "dst", "vis", "fallback")} for k in keys}

    def put_tracks(self, h: str, tr: Dict[int, Dict[int, Any]], frames) -> None:
        """tr: key frame -> {frame: zones.LocalWarp} (frames without a warp get an empty, invisible track)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        arrs: Dict[str, np.ndarray] = {}
        for k, by_f in tr.items():
            any_w = next((w for w in by_f.values() if hasattr(w, "pts")), None)
            if any_w is None:
                continue
            n = len(any_w.pts)
            eye = np.eye(2, 3, dtype=np.float32)
            get = lambda f: by_f.get(f) if hasattr(by_f.get(f), "pts") else None
            arrs[f"{k}_pts"] = any_w.pts.astype(np.float32)
            arrs[f"{k}_dst"] = np.stack([get(f).dst if get(f) is not None else np.zeros((n, 2), np.float32) for f in frames])
            arrs[f"{k}_vis"] = np.stack([get(f).vis if get(f) is not None else np.zeros(n, bool) for f in frames])
            arrs[f"{k}_fallback"] = np.stack([get(f).fallback if get(f) is not None else eye for f in frames])
        if arrs:
            np.savez_compressed(self.dir / f"tracks_{h}.npz", **arrs)

    def stats(self) -> Dict[str, int]:
        return {"cached": self.hits, "computed": self.misses}
