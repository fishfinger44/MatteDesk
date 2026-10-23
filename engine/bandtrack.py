# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/bandtrack.py - positions of the band control points (engine/bandmap.py) in every frame.

Each key of a point is a CoTracker3 query (frame, x, y); in a frame the point takes the track of its NEAREST key (as
the zone drawing follows its nearest key, zones.propagate_regions), so a corrected key re-anchors the point from there.
CoTracker3 visibility 0 = occluded: the point does not shape the band in that frame. Without a tracker (CoTracker3
missing, tests): the nearest key's position – the snap to the outline (bandmap.width_maps) keeps it on the edge.
The tracks depend on the key POSITIONS only (widths do not move the points): cached per zone in
zones_cache/<zone>/bandtracks_<hash>.npz, so a whisker drag never re-runs the tracker.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.bandmap import static_positions

Positions = Dict[int, Dict[str, Tuple[float, float, bool]]]


def _queries(points: Sequence[Dict[str, Any]], frames: Sequence[int]) -> List[Tuple[str, int, float, float]]:
    fs = set(frames)
    return [(p["id"], int(k), float(v["x"]), float(v["y"]))
            for p in points for k, v in sorted(p["keys"].items(), key=lambda kv: int(kv[0])) if int(k) in fs]


def tracks_sig(qs: Sequence[Tuple[str, int, float, float]], frames: Sequence[int], variant: str = "") -> str:
    q = [(pid, k, round(x, 2), round(y, 2)) for pid, k, x, y in qs]
    d: Dict[str, Any] = {"q": q, "frames": [int(frames[0]), int(frames[-1]), len(frames)]}
    if variant:                                   # how the tracker frames are made (absent = the old full frames)
        d["variant"] = variant
    return hashlib.sha1(json.dumps(d).encode()).hexdigest()[:16]


def _motion(points, frames, geo, want, occluder_at, out) -> None:
    """Core-motion positions (see track_positions) written into `out` for the frames asked."""
    for f in frames:
        if want is not None and f not in want:
            continue
        for p in points:
            k = min((int(kk) for kk in p["keys"]), key=lambda kk: (abs(kk - f), kk))
            v = p["keys"][str(k)]
            x, y = (float(v["x"]), float(v["y"])) if k == f else geo.shifted(k, f, float(v["x"]), float(v["y"]))
            out[f][p["id"]] = (x, y, not (occluder_at is not None and _covers(occluder_at(f), x, y)))


# Where CoTracker3 follows a point (station test 2026-10-21, beach shot Woman_01, fast pan ~190 px / frame): a point ON the
# hair / sky edge was carried away by the background (and flagged occluded); a point a little INSIDE the object stays
# with it. So each key is tracked from its outline pixel moved inwards along the normal by this fraction of the object
# size (sqrt(area)), clamped in px; in every frame the band snaps it back to the nearest outline (bandmap.width_maps).
TRACK_INSET_FRAC = 0.08
TRACK_INSET_PX = (6.0, 60.0)


def inset_queries(qs: Sequence[Tuple[str, int, float, float]], core_at: Callable[[int], Optional[np.ndarray]]
                  ) -> List[Tuple[str, int, float, float]]:
    from engine.bandmap import _contours, snap_normal
    out = []
    per: Dict[int, Any] = {}     # key frame -> (core, contours, area): read + traced once, not once per point
    for pid, k, x, y in qs:
        if k not in per:
            c = core_at(k)
            per[k] = (c, _contours(c), float(c.sum())) if c is not None and c.any() else (c, None, 0.0)
        c, cs, area = per[k]
        sn = snap_normal(c, x, y, cs) if cs is not None else None
        if sn is None:
            out.append((pid, k, x, y))
            continue
        sx, sy, nx, ny = sn
        d = float(np.clip(TRACK_INSET_FRAC * np.sqrt(area), *TRACK_INSET_PX))
        qx, qy = sx - nx * d, sy - ny * d
        h, w = c.shape
        inside = 0 <= int(round(qx)) < w and 0 <= int(round(qy)) < h and c[int(round(qy)), int(round(qx))]
        out.append((pid, k, qx, qy) if inside else (pid, k, sx, sy))
    return out


def track_positions(points: Sequence[Dict[str, Any]], frames: Sequence[int], tracker: Any = None,
                    tracker_frames: Optional[Callable[[Sequence[int]], Tuple[List[np.ndarray], float]]] = None,
                    cache_dir: Optional[Path] = None,
                    core_at: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                    occluder_at: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                    resolve: Optional[Sequence[int]] = None, wait: bool = True,
                    info: Optional[Dict[str, Any]] = None) -> Positions:
    """{frame: {point id: (x, y, visible)}} for every frame (`resolve`: only these frames get tracked positions, the
    rest keep the nearest key's – the outline work is per frame). `core_at(f)`: the object's outline mask – the point
    is tracked from a little inside the object (inset_queries), a track that ends far from the object follows the
    object's centroid instead, and `occluder_at(f)` (objects in front) decides visibility; without `core_at`: the raw
    track and CoTracker3's visibility. `wait=False` (interactive calls): tracks not cached yet are computed in the
    background and the nearest keys' positions are returned now, `info["pending"] = True`."""
    frames = list(frames)
    if not points or not frames:
        return {f: {} for f in frames}
    out = static_positions(points, frames)
    geo = _Geometry(core_at) if core_at is not None else None
    want = set(resolve) if resolve is not None else None
    if geo is not None:
        # without (or before) the tracker: each key carried by the object's own motion – the core's centroid and scale
        # from the key frame (user idea 2026-10-21: use the core mask we have); the band snaps it onto the outline
        _motion(points, frames, geo, want, occluder_at, out)
    qs = _queries(points, frames)
    if tracker is None or tracker_frames is None or not qs:
        return out
    if core_at is not None:
        qs = inset_queries(qs, core_at)
    h = tracks_sig(qs, frames, getattr(tracker_frames, "variant", ""))
    path = Path(cache_dir) / f"bandtracks_{h}.npz" if cache_dir is not None else None
    got = _tracks(h, path, qs, frames, tracker, tracker_frames, wait)
    if got is None:                               # tracking in the background: the core-motion positions for now
        if info is not None:
            info["pending"] = True
        return out
    tracks, vis = got
    qi = {(pid, k): j for j, (pid, k, _x, _y) in enumerate(qs)}
    for i, f in enumerate(frames):
        if want is not None and f not in want:
            continue
        for p in points:
            keys_in = [int(k) for k in p["keys"] if (p["id"], int(k)) in qi]
            if not keys_in:
                continue                          # no key inside the range: the nearest key's position (static)
            k = min(keys_in, key=lambda kk: (abs(kk - f), kk))
            v = p["keys"][str(k)]
            if k == f:
                out[f][p["id"]] = (float(v["x"]), float(v["y"]), True)
                continue
            j = qi[(p["id"], k)]
            x, y, seen = float(tracks[i, j, 0]), float(tracks[i, j, 1]), bool(vis[i, j])
            if geo is not None:
                if geo.lost(f, x, y):             # carried off by the background (wispy hair): move with the object
                    x, y = geo.shifted(k, f, float(v["x"]), float(v["y"]))
                # occlusion = an object in front covers the point (user decision 2026-10-18, depth_order); CoTracker3's
                # own visibility flagged points on the body as occluded in fast motion (station 2026-10-21)
                seen = not (occluder_at is not None and _covers(occluder_at(f), x, y))
            out[f][p["id"]] = (x, y, seen)
    return out


# One band tracking at a time, shared per signature (station 2026-10-21: every band / region / preview request started
# its own CoTracker3 worker – 7 at once filled the GPU and each took minutes). Waiting callers join the running one.
_LOCK = threading.Lock()
_RUNNING: Dict[str, threading.Event] = {}
_RUN_GUARD = threading.Lock()


def _load(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    try:
        d = np.load(path)
        return d["tracks"], d["vis"]
    except Exception:
        return None


def _tracks(h, path, qs, frames, tracker, tracker_frames, wait):
    got = _load(path)
    if got is not None:
        return got
    with _RUN_GUARD:
        ev = _RUNNING.get(h)
        owner = ev is None
        if owner:
            ev = _RUNNING[h] = threading.Event()

    def compute():
        try:
            with _LOCK:
                if _load(path) is None:
                    got_frames = tracker_frames(frames)
                    imgs, scale = got_frames[0], got_frames[1]
                    # optional per-frame crop origins (plate px): a crop that follows the object (pipeline
                    # _band_tracker_frames) – queries go into crop coordinates, tracks come back to the plate
                    offs = (np.asarray(got_frames[2], np.float32) if len(got_frames) > 2
                            else np.zeros((len(frames), 2), np.float32))
                    t_of = {f: i for i, f in enumerate(frames)}
                    q = np.array([[t_of[k], (x - offs[t_of[k], 0]) * scale, (y - offs[t_of[k], 1]) * scale]
                                  for _pid, k, x, y in qs], np.float32)
                    res = tracker.track(imgs, q)
                    tr = np.asarray(res.tracks, np.float32) / scale + offs[:, None, :]
                    vs = np.asarray(res.visible, bool)
                    if path is None:
                        _MEM[h] = (tr, vs)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = path.with_name("~" + path.name)          # whole file or nothing for a waiting reader
                        with open(tmp, "wb") as fh:
                            np.savez_compressed(fh, tracks=tr, vis=vs)
                        tmp.replace(path)
        finally:
            with _RUN_GUARD:
                _RUNNING.pop(h, None)
            ev.set()

    if owner:
        if wait:
            compute()
        else:
            threading.Thread(target=compute, name=f"bandtrack-{h}", daemon=True).start()
            return None
    elif not wait:
        return None
    else:
        ev.wait()
    return _load(path) if path is not None else _MEM.get(h)


_MEM: Dict[str, Any] = {}          # tracks without a cache dir (tests)


# a tracked point farther than this from the object (fraction of sqrt(area), px clamp) is lost, not occluded
LOST_FRAC = 0.12
LOST_PX = (24.0, 200.0)


def _covers(mask: Optional[np.ndarray], x: float, y: float) -> bool:
    if mask is None:
        return False
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi])


class _Geometry:
    """Per-frame outline facts for the band points, computed only for the frames asked (lazy, cached)."""

    def __init__(self, core_at: Callable[[int], Optional[np.ndarray]]):
        self.core_at = core_at
        self._c: Dict[int, Any] = {}

    def _get(self, f: int):
        if f not in self._c:
            c = self.core_at(f)
            if c is None or not c.any():
                self._c[f] = None
            else:
                import cv2
                cs, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                pix = np.concatenate([q[:, 0, :] for q in cs]).astype(np.float32) if cs else np.zeros((0, 2), np.float32)
                ys, xs = np.nonzero(c)
                self._c[f] = (c, pix, (float(xs.mean()), float(ys.mean())), float(np.sqrt(len(xs))))
        return self._c[f]

    def lost(self, f: int, x: float, y: float) -> bool:
        g = self._get(f)
        if g is None or not len(g[1]) or _covers(g[0], x, y):
            return False
        d = float(np.sqrt(((g[1] - (x, y)) ** 2).sum(1)).min())
        return d > float(np.clip(LOST_FRAC * g[3], *LOST_PX))

    def shifted(self, k: int, f: int, x: float, y: float) -> Tuple[float, float]:
        a, b = self._get(k), self._get(f)
        if a is None or b is None:
            return x, y
        s = b[3] / a[3] if a[3] else 1.0          # the object's scale change (sqrt of the area ratio)
        return b[2][0] + (x - a[2][0]) * s, b[2][1] + (y - a[2][1]) * s
