# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/zones.py - "Strefa" Add / Remove (SPEC 7.7): fix a constant error for the whole shot at once.

Criterion `matte` (docs/PLAN_HAIR_TOOL.md, zone@1.1): local matting of the zone (engine/localmatte.py) – hair / fur /
fibres / smoke / motion blur / thin structures; the result is a soft override (alpha + area) that wins over the edge
mode in refine.finalize. Forced background = "-" strokes (`bg_strokes`) ∪ accepted distractor masks ∪ pixels removed by
earlier Remove zones; holdout = masks of the objects in front (pipeline.apply_zones provides them, else MissingFront).
A user key does NOT skip a `matte` zone in its frame (the key defines the core, the zone adds soft alpha outside it).

Zone = direction (add | remove) + criterion (dark | bright | color | all) + one slider + polygon / strokes drawn
once (frame `frame`), following the object (CoTracker3 points of the zone -> per-frame similarity transform;
fallback: core bbox shift) or static (garbage matte). Soft mode (dark / bright / color): a ramp instead of a
threshold -> semi-transparent alpha that wins over the edge mode inside its area.
Order in the pipeline (7.7 pkt 5): on the core, before the edge mode; removed pixels (+2 px) leave the edge band.
Safety: Add keeps only pixels connected to the object (MaskRecoverFromGuide-like, reach 200 px), Remove acts only
inside the zone, results stabilized in time, a user key wins in its frame.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ZONE_SCHEMA = "matte_desk/zone@1.2"
# zone@1.2 (2026-10-24): params.outer_feather_px is literal like in every stage – 0 = a hard outer band edge; absent = as
# the inset feather. In zone@1.1 an explicit 0 meant „jak wcięcie”: load_zones writes the inset value there (same look)
CRITERIA = ("dark", "bright", "color", "all", "matte")


class MissingFront(RuntimeError):
    """A `matte` zone needs the masks of the objects in front, and some have none yet (plan section 6):
    the UI / CLI offer "Policz obiekty z przodu i kontynuuj" (queue) or "Licz mimo to bez holdoutu"."""

    def __init__(self, missing: List[str], queue: List[Dict[str, str]]):
        self.missing = list(missing)
        self.queue = list(queue)
        super().__init__("Obiekty z przodu (" + ", ".join(self.missing) + ") nie mają jeszcze maski – przelicz je w "
                         "pierwszej kolejności („Policz obiekty z przodu i kontynuuj”) albo licz bez holdoutu")


@dataclass
class Zone:
    zone_id: str
    mode: str                     # add | remove
    criterion: str                # dark | bright | color | all
    frame: int
    polygon: List[List[float]] = field(default_factory=list)   # plate px at `frame`
    strokes: List[Dict[str, Any]] = field(default_factory=list)
    param: float = 0.15           # luma threshold (dark/bright) or ΔE tolerance (color)
    soft: bool = False
    softness: float = 0.05        # ramp half-width (luma) or ΔE ramp width
    color_lab: Optional[List[float]] = None
    static: bool = False
    grow_px: int = 200
    enabled: bool = True
    schema: str = ZONE_SCHEMA
    # --- criterion `matte` (zone@1.1; unused by the other criteria) ---
    preset: str = "hair"                 # hair | fibres | smoke | motion | thin
    matting: str = "matanyone"           # matanyone (default, user decision 2026-10-13) | auto | chroma | knn | cf | hybrid | clean
    prompt: str = ""                     # zone description (step 9: keyword rules / VLM)
    prompt_history: List[str] = field(default_factory=list)
    bg_strokes: List[Dict[str, Any]] = field(default_factory=list)   # "-" strokes = forced background (base key)
    bg_static: bool = True               # "-" strokes mark background elements (a line on the wall): they do not follow the object
    keys: Dict[str, Dict[str, Any]] = field(default_factory=dict)     # zone keys {frame: {polygon, strokes, bg_strokes}} (step 4)
    holdout: bool = True                 # False = "Licz mimo to bez holdoutu" (yellow QA flag)
    distractors: List[Dict[str, Any]] = field(default_factory=list)   # [{prompt, mask_path, accepted}] (step 9-10)
    engine_hints: Dict[str, float] = field(default_factory=dict)      # engine -> weight multiplier (prompt / VLM)
    params: Dict[str, Any] = field(default_factory=dict)              # chroma_lo / chroma_hi / bg_color / engines…

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_zone(obj_dir: Path, z: Zone) -> Path:
    d = obj_dir / "zones"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{z.zone_id}.json"
    p.write_text(json.dumps(z.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_zones(obj_dir: Path) -> List[Zone]:
    """The zones of an object; a whole-outline zone on the shared object band carries the band's points AND sliders
    (engine/objband.zone_params, 2026-10-24) – everything downstream reads the effective params."""
    from engine.objband import zone_params
    d = obj_dir / "zones"
    try:
        obj = json.loads((obj_dir / "object.json").read_text(encoding="utf-8"))
        root = obj     # a group member has its own band (PLAN_GROUP_COMPOSE 2026-10-29)
    except (OSError, ValueError, KeyError):
        obj, root = None, None
    out = []
    for p in sorted(d.glob("z_*.json")) if d.exists() else []:
        zd = json.loads(p.read_text(encoding="utf-8"))
        if obj is not None and isinstance(obj.get("band"), dict) and zd.get("criterion") == "matte" \
                and (zd.get("params") or {}).get("region") == "outline":
            zd = {**zd, "params": zone_params(zd, obj, root or {})}
        out.append(_migrated(Zone(**zd)))
    return out


def _migrated(z: "Zone") -> "Zone":
    """zone@1.1 -> 1.2: an outer feather of 0 meant „as the inset feather” – it gets that value, so the zone looks the
    same while 0 now means a hard edge (one meaning in every stage, user decision 2026-10-24). Read-time; saved with the
    next edit of the zone."""
    if z.schema != ZONE_SCHEMA:
        p = dict(z.params or {})
        if not float(p.get("outer_feather_px") or 0.0):
            p["outer_feather_px"] = zone_feather_px(z)
        z.params = p
        z.schema = ZONE_SCHEMA
    return z


def next_zone_id(obj_dir: Path) -> str:
    ids = [int(z.zone_id[2:]) for z in load_zones(obj_dir) if z.zone_id[2:].isdigit()]
    return f"z_{(max(ids) + 1) if ids else 1:02d}"


# ----------------------------------------------------------------------------- region of the zone per frame
def zone_mask_at(z: Zone, shape: Tuple[int, int], transform: Optional[np.ndarray] = None) -> np.ndarray:
    """Polygon / strokes rasterized at a frame; `transform` (2x3) moves the drawing from `z.frame`."""
    return raster_drawing(z.polygon, z.strokes, shape, transform)


def zone_bg_mask_at(z: Zone, shape: Tuple[int, int], transform: Optional[np.ndarray] = None) -> np.ndarray:
    """"-" strokes of a `matte` zone (forced background) rasterized at a frame."""
    if not z.bg_strokes:
        return np.zeros(shape, bool)
    return raster_drawing([], z.bg_strokes, shape, transform)


def raster_drawing(polygon: Sequence[Sequence[float]], strokes: Sequence[Dict[str, Any]], shape: Tuple[int, int],
                   transform: Optional[np.ndarray] = None) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    def tr(pts):
        p = np.asarray(pts, np.float32).reshape(-1, 2)
        if transform is not None and len(p):
            p = cv2.transform(p[None], transform)[0]
        return np.round(p).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillPoly(m, [tr(polygon).reshape(-1, 1, 2)], 1)
    # strokes with "erase": true remove from the drawing (palette "Zmaż obszar", 2026-10-15) – drawn after the others
    for s in sorted(strokes, key=lambda q: bool(q.get("erase"))):
        val = 0 if s.get("erase") else 1
        from engine.keys import draw_variable_stroke, has_point_radius
        if has_point_radius(s.get("p", [])):   # Z23: radius per point (pen); the zone moves, radii stay
            raw = [list(q) for q in s.get("p", [])]
            xy = np.asarray([q[:2] for q in raw], np.float32).reshape(-1, 2)
            if transform is not None and len(xy):
                xy = cv2.transform(xy[None], transform)[0]
            draw_variable_stroke(m, [[float(a), float(b)] + (q[2:3] if len(q) >= 3 else []) for (a, b), q in zip(xy, raw)],
                                 s.get("radius_px", 8), val)
            continue
        pts = tr(s.get("p", []))
        r = max(1, int(round(s.get("radius_px", 8))))
        if len(pts) > 1:
            cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, val, thickness=2 * r)
        for q in pts:
            cv2.circle(m, tuple(int(v) for v in q), r, val, -1)
    return m.astype(bool)


TRACK_MODES = ("rigid", "local")
LOCAL_SIGMA_PX = 60.0        # local tracking: reach of one CoTracker point's displacement (Gaussian, plate px)
LOCAL_POINTS = 96


def track_mode(z: "Zone") -> str:
    m = str((z.params or {}).get("track", "rigid") or "rigid")
    if m not in TRACK_MODES:
        raise ValueError(f"strefa {z.zone_id}: params.track musi być jednym z {TRACK_MODES}, jest {m!r}")
    return m


@dataclass
class LocalWarp:
    """Local (non-rigid) motion of a zone drawing from its key frame to one frame (user decision 2026-10-15, rotobrush-
    like painted band): CoTracker3 points `pts` (key frame, plate px) seen at `dst` (this frame), `vis` = visible; the
    drawing is warped by the Gaussian-weighted mean displacement of the visible points near each pixel. `fallback` (2x3)
    = the rigid / core-bbox transform for places no point reaches (and frames with < 6 visible points)."""
    pts: np.ndarray
    dst: np.ndarray
    vis: np.ndarray
    fallback: np.ndarray
    sigma: float = LOCAL_SIGMA_PX


def local_warp_mask(mask: np.ndarray, lw: "LocalWarp") -> np.ndarray:
    """Warp a key-frame mask (bool / float) to the frame of `lw`: for every target pixel p the source is p - d(p), d =
    normalised Gaussian splat (on a 1/8 grid) of the visible points' displacements placed at their positions in THIS
    frame; where the points do not reach, the fallback transform's displacement is used."""
    H, W = mask.shape
    m = mask.astype(np.float32)
    v = lw.vis.astype(bool)
    if v.sum() < 6:
        return cv2.warpAffine(m, lw.fallback, (W, H), flags=cv2.INTER_LINEAR) > 0.5
    g = 8
    gh, gw = (H + g - 1) // g, (W + g - 1) // g
    acc = np.zeros((2, gh, gw), np.float32)
    wsum = np.zeros((gh, gw), np.float32)
    d = (lw.dst[v] - lw.pts[v]).astype(np.float32)
    q = lw.dst[v] / g
    ix = np.clip(np.round(q[:, 0]).astype(int), 0, gw - 1)
    iy = np.clip(np.round(q[:, 1]).astype(int), 0, gh - 1)
    np.add.at(wsum, (iy, ix), 1.0)
    np.add.at(acc[0], (iy, ix), d[:, 0])
    np.add.at(acc[1], (iy, ix), d[:, 1])
    sg = max(1.0, lw.sigma / g)
    k = int(round(sg * 3)) * 2 + 1
    wb = cv2.GaussianBlur(wsum, (k, k), sg)
    ax, ay = cv2.GaussianBlur(acc[0], (k, k), sg), cv2.GaussianBlur(acc[1], (k, k), sg)
    # fallback displacement field (target -> source of the rigid transform) where the points do not reach
    Mi = cv2.invertAffineTransform(lw.fallback.astype(np.float32))
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * g
    fx = gx - (Mi[0, 0] * gx + Mi[0, 1] * gy + Mi[0, 2])
    fy = gy - (Mi[1, 0] * gx + Mi[1, 1] * gy + Mi[1, 2])
    peak = 1.0 / (2 * np.pi * sg * sg)                         # weight of one point right under it
    trust = np.clip(wb / (0.25 * peak), 0.0, 1.0)              # 1 near the points, 0 far away
    dx = trust * (ax / np.maximum(wb, 1e-9)) + (1 - trust) * fx
    dy = trust * (ay / np.maximum(wb, 1e-9)) + (1 - trust) * fy
    dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(m, X - dx, Y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5


def follow_transforms(z: Zone, frames: Sequence[int], cores: Dict[int, np.ndarray],
                      tracker_frames: Optional[Callable[[], Tuple[List[np.ndarray], float]]] = None,
                      tracker: Any = None, point_exclude: Optional[np.ndarray] = None) -> Dict[int, Any]:
    """Per-frame 2x3 transforms moving the zone with the object. Static zone -> identity.
    With a point tracker (CoTracker3) the zone's points are tracked on display frames (downscaled by `scale`);
    otherwise the core's bbox shift + scale is used (fallback, SPEC 7.7 pkt 3)."""
    eye = np.eye(2, 3, dtype=np.float32)
    if z.static:
        return {f: eye for f in frames}
    fallback = _bbox_transforms(z, frames, cores)
    local = track_mode(z) == "local"
    pts = (_object_points(z, cores.get(z.frame), n=LOCAL_POINTS if local else 48, margin=60 if local else 40,
                          exclude=point_exclude)
           if tracker is not None and tracker_frames is not None else None)
    if pts is not None and local:
        # local mode: every frame keeps the point tracks (warped per pixel); the rigid fit is only the fallback
        imgs, scale = tracker_frames()
        t0 = list(frames).index(z.frame)
        q = np.column_stack([np.full(len(pts), t0), pts * scale]).astype(np.float32)
        res = tracker.track(imgs, q)
        out = {}
        for i, f in enumerate(frames):
            vis = res.visible[i].astype(bool)
            dst = (res.tracks[i] / scale).astype(np.float32)
            M = None
            if vis.sum() >= 6:
                M, inl = cv2.estimateAffinePartial2D(pts[vis], dst[vis], method=cv2.RANSAC, ransacReprojThreshold=3.0)
            out[f] = LocalWarp(pts=pts.astype(np.float32), dst=dst, vis=vis,
                               fallback=(M if M is not None else fallback[f]).astype(np.float32))
        return out
    if pts is not None:
        # Track points ON THE OBJECT around the zone (never the static background around it: RANSAC would pick the
        # background motion and the zone would stay behind). Frames where tracking fails use the core bbox shift.
        imgs, scale = tracker_frames()
        t0 = list(frames).index(z.frame)
        q = np.column_stack([np.full(len(pts), t0), pts * scale]).astype(np.float32)
        res = tracker.track(imgs, q)
        out = {}
        for i, f in enumerate(frames):
            vis = res.visible[i]
            M = None
            if vis.sum() >= 6:
                M, inl = cv2.estimateAffinePartial2D(pts[vis], res.tracks[i][vis] / scale, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0)
                if M is not None and (inl is None or inl.sum() < 0.5 * vis.sum()):
                    M = None
            out[f] = M.astype(np.float32) if M is not None else fallback[f]
        return out
    return fallback


def _object_points(z: Zone, core: Optional[np.ndarray], n: int = 48, margin: int = 40,
                   exclude: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Up to n points of the object's core inside the zone grown by `margin` px (the part of the object the zone
    belongs to). None when the core there is too small -> bbox fallback. `exclude` (key frame mask, e.g. a hand moving
    over the head – a part / an object in front, 2026-10-15): no point on it (grown 8 px), it would drag the zone."""
    if core is None or not core.any():
        return None
    area = cv2.dilate(zone_mask_at(z, core.shape).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))).astype(bool)
    inner = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # off the edge: stable texture
    if exclude is not None and exclude.any():
        inner &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    ys, xs = np.where(inner & area)
    if len(xs) < 12:
        return None
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return np.column_stack([xs[idx], ys[idx]]).astype(np.float32)


def _bbox_transforms(z: Zone, frames: Sequence[int], cores: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    eye = np.eye(2, 3, dtype=np.float32)
    def bb(m):
        ys, xs = np.where(m)
        return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
    ref = bb(cores[z.frame])
    out = {}
    for f in frames:
        b = bb(cores[f]) if f in cores else None
        if ref is None or b is None:
            out[f] = eye
            continue
        s = ((b[2] - b[0]) + (b[3] - b[1])) / max(1e-6, (ref[2] - ref[0]) + (ref[3] - ref[1]))
        cx0, cy0 = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2
        cx1, cy1 = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        out[f] = np.array([[s, 0, cx1 - s * cx0], [0, s, cy1 - s * cy0]], np.float32)
    return out


# ----------------------------------------------------------------------------- zone keys (step 4)
def zone_keys(z: Zone) -> Dict[int, Dict[str, Any]]:
    """All keys of the zone, frame -> {polygon, strokes, bg_strokes}; the base drawing (z.frame) is always a key.
    (Only the drawing: the cache hashes of the propagation are built from this – the iteration note of a key is read
    with `zone_key_prompt`.)"""
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in (z.keys or {}).items():
        try:
            out[int(k)] = {"polygon": list(v.get("polygon", [])), "strokes": list(v.get("strokes", [])),
                           "bg_strokes": list(v.get("bg_strokes", []))}
        except (TypeError, ValueError, AttributeError):
            continue
    out[int(z.frame)] = {"polygon": list(z.polygon), "strokes": list(z.strokes), "bg_strokes": list(z.bg_strokes)}
    return dict(sorted(out.items()))


def zone_outer_feather_px(z: "Zone") -> float:
    """Feather (px) of a whole-outline zone at its OUTER band edge (towards the background), separate from the inset
    feather on the core side (user request 2026-10-21). Literal like every stage (2026-10-24): 0 = a hard outer edge;
    absent = the inset feather (the default of a new zone; zone@1.1 zones are migrated in load_zones)."""
    v = (z.params or {}).get("outer_feather_px")
    if v is None:
        return zone_feather_px(z)
    try:
        return max(0.0, float(v))
    except (TypeError, ValueError):
        return zone_feather_px(z)


# the feathers of a band (inner from the inner edge, outer from the outer edge, never overlapping; user rule 2026-10-22)
# are the one band system: engine/bandsys.py (area_dists / lengths / weight – the former feather_maps / feather_weight
# of this module moved there on 2026-10-24, so stages 3 / 4 / 5 / 7 share one rule)


def zone_feather_px(z: "Zone") -> float:
    """Feather (px) of a matte zone over its base inside its area – the core inset blends into the core over this many px
    instead of a hard step (user request 2026-10-17); default = the former fixed value."""
    from engine.bandsys import BAND_DEFAULTS
    v = (z.params or {}).get("inset_feather_px")
    try:
        return max(0.5, float(v)) if v is not None else BAND_DEFAULTS["inset_feather_px"]
    except (TypeError, ValueError):
        return BAND_DEFAULTS["inset_feather_px"]


def outline_without_drawing(z: Zone) -> bool:
    """A `matte` zone in „whole outline” mode with nothing painted in any key (no exclusion, no background stroke):
    its area is the core outline ± the band in every frame – nothing to track (2026-10-17)."""
    try:
        if z.criterion != "matte" or region_mode(z) != "outline":
            return False
    except ValueError:
        return False
    return not any(k["polygon"] or k["strokes"] or k["bg_strokes"] for k in zone_keys(z).values())


def zone_hints(z: Zone) -> Optional[Dict[str, float]]:
    """Routing multipliers of a `matte` zone = prompt rules (`localmatte_prompt.interpret_prompt(z.prompt)`, step 9)
    x explicit `engine_hints`; None when neither says anything (the presets decide)."""
    from engine.localmatte_prompt import interpret_prompt, merge_hints
    out = merge_hints(z.engine_hints, interpret_prompt(z.prompt) if z.prompt else None)
    return out or None


def zone_key_prompt(z: Zone, f: int) -> str:
    """Iteration note of the correction loop stored in the key at f (plan step 7); "" when none."""
    v = (z.keys or {}).get(str(int(f)))
    return str(v.get("prompt") or "") if isinstance(v, dict) else ""


def key_zone(z: Zone, f: int) -> Zone:
    """Copy of the zone whose base drawing is its key at frame f (for follow_transforms / zone_mask_at). The key's
    iteration note (if any) is appended to the zone prompt, so the prompt rules (step 9) see it in that frame."""
    k = zone_keys(z)[int(f)]
    d = z.to_dict()
    d.update(frame=int(f), polygon=k["polygon"], strokes=k["strokes"], bg_strokes=k["bg_strokes"])
    note = zone_key_prompt(z, f)
    if note:
        d["prompt"] = f"{z.prompt}; {note}" if z.prompt else note
    return Zone(**d)


def set_zone_key(z: Zone, f: int, polygon: Sequence[Sequence[float]], strokes: Sequence[Dict[str, Any]] = (),
                 bg_strokes: Sequence[Dict[str, Any]] = (), prompt: Optional[str] = None) -> Zone:
    """Add / replace the key at frame f (the base key when f == z.frame). `prompt` = iteration note of the correction
    loop (plan step 7): stored in the key, appended to `prompt_history` (list of iterations) when new; None keeps the
    key's previous note, "" clears it. Returns the zone (mutated)."""
    f = int(f)
    prev = (z.keys or {}).get(str(f)) or {}
    note = str(prev.get("prompt") or "") if prompt is None else prompt.strip()
    entry = {"polygon": [list(map(float, q)) for q in polygon], "strokes": [dict(x) for x in strokes],
             "bg_strokes": [dict(x) for x in bg_strokes]}
    if note:
        entry["prompt"] = note
        if note != (prev.get("prompt") or "") and note not in (z.prompt_history or []):
            z.prompt_history = list(z.prompt_history or []) + [note]
    if f == int(z.frame):
        z.polygon, z.strokes, z.bg_strokes = entry["polygon"], entry["strokes"], entry["bg_strokes"]
    z.keys = {**{k: v for k, v in (z.keys or {}).items() if int(k) != f}, str(f): entry}
    return z


def delete_zone_key(z: Zone, f: int) -> Zone:
    """Remove the key at f; removing the base key promotes the nearest remaining key to base. The last key stays."""
    f = int(f)
    keys = zone_keys(z)
    if f not in keys or len(keys) <= 1:
        raise ValueError("Strefa musi mieć co najmniej jeden klucz" if len(keys) <= 1 else f"Brak klucza strefy w klatce {f}")
    keys.pop(f)
    if f == int(z.frame):
        nf = min(keys, key=lambda k: abs(k - f))
        z.frame = nf
        z.polygon, z.strokes, z.bg_strokes = keys[nf]["polygon"], keys[nf]["strokes"], keys[nf]["bg_strokes"]
    z.keys = {str(k): v for k, v in keys.items()}
    return z


def _lerp_polygon(pa: np.ndarray, pb: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * pa + t * pb


def propagate_regions(z: Zone, frames: Sequence[int], cores: Dict[int, np.ndarray], shape: Tuple[int, int],
                      tracker_frames_for: Optional[Callable[[Sequence[int]], Tuple[List[np.ndarray], float]]] = None,
                      tracker: Any = None, transforms_cache: Optional[Dict[str, Dict[int, Any]]] = None,
                      point_exclude: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                      only: Optional[Sequence[int]] = None
                      ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Dict[int, Any]], Dict[str, Any]]:
    """Region and forced-BG mask of the zone in every frame from its keys (plan section 5):
    - before the first / after the last key: that key moved by follow_transforms (CoTracker3, fallback core bbox);
    - between two keys a < f < b: polygons with the same vertex count (no strokes) -> vertices interpolated between
      key a moved forward and key b moved backward; otherwise the union of the two moved drawings;
    - a key frame = exactly its drawing. "-" strokes: static (bg_static) = the nearest key's strokes as drawn, else moved
      with that key. `transforms_cache` (key frame -> transforms) avoids re-tracking (ZoneCache.put_transforms).
    `only` (2026-10-30, the frame previews after scrubbing): regions / BG masks of these frames only – the tracking
    and the transforms still span `frames` (same cache, same result in those frames, bit for bit); ~3.5 s -> well
    under a second per previewed frame on a 50-frame work area.
    Returns (regions, bg_masks, transforms per key as {str(key): {f: M}}, info)."""
    frames = list(frames)
    want = set(int(f) for f in only) if only is not None else None
    need = (lambda f: True) if want is None else (lambda f: f in want)
    _kz: Dict[int, Zone] = {}

    def kz(k: int) -> Zone:   # key_zone once per key and call (a deep copy of the zone each time was ~30 ms)
        if k not in _kz:
            _kz[k] = key_zone(z, k)
        return _kz[k]
    keys = zone_keys(z)
    kf = sorted(k for k in keys if k in frames) or [int(z.frame)]
    eye = np.eye(2, 3, dtype=np.float32)
    tr: Dict[str, Dict[int, np.ndarray]] = {}
    tracked = []

    def transforms_of(k: int, sub: Sequence[int]) -> Dict[int, np.ndarray]:
        key = str(k)
        have = (transforms_cache or {}).get(key) or tr.get(key) or {}
        missing = [f for f in sub if f not in have]
        if missing:
            zk = kz(k)
            sub_all = list(range(min(min(sub), k), max(max(sub), k) + 1))   # continuous: a key outside `sub` too
            tf = (lambda: tracker_frames_for(sub_all)) if (tracker_frames_for is not None and tracker is not None) else None
            got = follow_transforms(zk, sub_all, cores, tf, tracker,
                                    point_exclude=point_exclude(k) if point_exclude is not None else None)
            have = {**have, **got}
            tracked.append((k, sub_all[0], sub_all[-1]))
        tr[key] = have
        return have

    regions: Dict[int, np.ndarray] = {}
    bgs: Dict[int, np.ndarray] = {}

    def rigid(M):    # the 2x3 part of a transform (a LocalWarp's fallback for the "-" strokes)
        return M.fallback if isinstance(M, LocalWarp) else M

    def bg_of(k: int, f: int, M: Optional[Any]) -> np.ndarray:
        zk = kz(k)
        return zone_bg_mask_at(zk, shape, None if z.bg_static else rigid(M))

    def region_of(k: int, M: Optional[Any]) -> np.ndarray:
        zk = kz(k)
        if isinstance(M, LocalWarp):
            return local_warp_mask(zone_mask_at(zk, shape, None), M)
        return zone_mask_at(zk, shape, M)

    first, last = kf[0], kf[-1]
    # before the first key / after the last key / single key
    head = [f for f in frames if f <= first]
    tail = [f for f in frames if f >= last]
    if head:
        T = transforms_of(first, head)
        for f in head:
            if not need(f):
                continue
            M = None if f == first else T.get(f, eye)
            regions[f] = region_of(first, M)
            bgs[f] = bg_of(first, f, M)
    if tail:
        T = transforms_of(last, tail)
        for f in tail:
            if f in regions or not need(f):
                continue
            M = None if f == last else T.get(f, eye)
            regions[f] = region_of(last, M)
            bgs[f] = bg_of(last, f, M)
    # between consecutive keys
    for a, b in zip(kf[:-1], kf[1:]):
        mid = [f for f in frames if a <= f <= b]
        if len(mid) <= 2:
            continue
        Ta = transforms_of(a, mid)
        Tb = transforms_of(b, mid)
        ka, kb = keys[a], keys[b]
        same_poly = (len(ka["polygon"]) >= 3 and len(ka["polygon"]) == len(kb["polygon"]) and not ka["strokes"]
                     and not kb["strokes"] and track_mode(z) != "local")
        for f in mid:
            if not need(f):
                continue
            if f in (a, b):
                regions[f] = zone_mask_at(kz(f), shape, None)
                bgs[f] = bg_of(f, f, None)
                continue
            Ma, Mb = Ta.get(f, eye), Tb.get(f, eye)
            t = (f - a) / float(b - a)
            if same_poly:
                pa = cv2.transform(np.asarray(ka["polygon"], np.float32).reshape(1, -1, 2), Ma)[0]
                pb = cv2.transform(np.asarray(kb["polygon"], np.float32).reshape(1, -1, 2), Mb)[0]
                regions[f] = raster_drawing(_lerp_polygon(pa, pb, t).tolist(), [], shape, None)
            elif isinstance(Ma, LocalWarp) or isinstance(Mb, LocalWarp):
                # local mode: both keys warped to f, cross-faded by the distance to each key (no union doubling)
                ra = region_of(a, Ma).astype(np.float32)
                rb = region_of(b, Mb).astype(np.float32)
                regions[f] = ((1.0 - t) * ra + t * rb) >= 0.5
            else:
                regions[f] = zone_mask_at(kz(a), shape, Ma) | zone_mask_at(kz(b), shape, Mb)
            near = a if t < 0.5 else b
            bgs[f] = bg_of(near, f, Ma if near == a else Mb)
    info = {"keys": kf, "tracked": tracked, "interpolated": any(len([f for f in frames if a < f < b]) for a, b in zip(kf[:-1], kf[1:]))}
    return regions, bgs, tr, info


# ----------------------------------------------------------------------------- criteria
def _luma(img: np.ndarray) -> np.ndarray:
    return (0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2]).astype(np.float32)  # BGR


def criterion_key(img: np.ndarray, z: Zone) -> np.ndarray:
    """0..1 key of 'what is in the zone' on the plate-to-mask display image."""
    if z.criterion == "all":
        return np.ones(img.shape[:2], np.float32)
    if z.criterion in ("dark", "bright"):
        y = _luma(img)
        if z.soft:
            k = np.clip((z.param + z.softness - y) / (2 * max(z.softness, 1e-4)), 0, 1)
        else:
            k = (y < z.param).astype(np.float32)
        return k if z.criterion == "dark" else (1.0 - k if z.soft else (y > z.param).astype(np.float32))
    if z.criterion == "color":
        lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)
        ref = np.asarray(z.color_lab or [50, 0, 0], np.float32)
        de = np.linalg.norm(lab - ref, axis=2)
        if z.soft:
            return np.clip((z.param + z.softness - de) / (2 * max(z.softness, 1e-4)), 0, 1).astype(np.float32)
        return (de < z.param).astype(np.float32)
    raise ValueError(f"nieznane kryterium {z.criterion}")


def lab_of(img: np.ndarray) -> np.ndarray:
    """Lab (L 0..100) of a display BGR float image – the same conversion as the "color" criterion."""
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)


def sample_lab(patch: np.ndarray) -> List[float]:
    """Median Lab of a small display patch (the pipette: 5 x 5 px of the plate-to-mask image), rounded."""
    lab = lab_of(patch).reshape(-1, 3)
    return [round(float(v), 2) for v in np.median(lab, axis=0)]


def zone_median_lab(z: "Zone", img: np.ndarray) -> Optional[List[float]]:
    """Fallback colour of a "color" zone without a pipette pick: median Lab inside the drawn area at its frame."""
    reg = zone_mask_at(z, img.shape[:2])
    if not reg.any():
        return None
    lab = lab_of(img)[reg]
    return [round(float(v), 2) for v in np.median(lab, axis=0)]


def connected_to(cand: np.ndarray, core: np.ndarray, reach: int = 200) -> np.ndarray:
    """Candidate pixels in components touching the object and within `reach` px of it (no floating islands)."""
    if not cand.any() or not core.any():
        return np.zeros_like(cand)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    cand = cand & (dist <= reach)
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    touch = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[touch & cand])] = True
    keep[0] = False
    return keep[lab]


@dataclass
class ZoneResult:
    core: Dict[int, np.ndarray]                  # corrected binary core per frame
    removed: Dict[int, np.ndarray]               # removed pixels (+2 px) -> out of the edge band
    soft: Dict[int, Tuple[np.ndarray, np.ndarray]]  # frame -> (soft alpha, area where it wins)
    flags: Dict[int, List[str]] = field(default_factory=dict)   # frame -> zone QA hints (engines disagree, no holdout…)
    stats: Dict[str, Any] = field(default_factory=dict)         # zone id -> summary for run.json / the UI card
    feather_px: Optional[float] = None   # feather of the soft zones over the base (max of the matte zones' inset_feather_px)
    # frame -> uint8 blend weight (255 = the zone) when a zone has its own outer-edge feather (outer_feather_px, 2026-10-21);
    # frames without one keep the distance feather of apply_soft_override (bit-identical)
    weights: Dict[int, np.ndarray] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)      # plan A6b: wall seconds of the shared composing


MA_ZONE_TARGET_PX = 1008     # MatAnyone zone engine: crop side scaled up to this (<= MA_ZONE_MAX_UP x)
MA_ZONE_MAX_UP = 3.0
MA_ZONE_PAD_PX = 48


def zone_crop_segment(regions: Dict[int, np.ndarray], frames: Sequence[int], zone_id: str,
                      pad_px: int = MA_ZONE_PAD_PX):
    """Square crop segment (engine/crop.CropSegment) around the union of the zone regions over the shot."""
    from engine.crop import CropSegment, square_crop
    H, W = regions[frames[0]].shape
    union = np.zeros((H, W), bool)
    for f in frames:
        union |= regions[f]
    if not union.any():
        return None
    ys, xs = np.where(union)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    need = max(xs.max() - xs.min(), ys.max() - ys.min()) + 2 * pad_px
    crop, pad = square_crop(cx, cy, float(need), W, H, pad_ratio=1.0, margin=float(pad_px), snap=16)
    return CropSegment(f"zone_{zone_id}", int(frames[0]), int(frames[-1]), 0, 0, crop, pad, 1.0)


# ----------------------------------------------------------------------------- region mode "outline" (user decision 2026-10-15)
# The zone is a band around the WHOLE edge of the SAM3 core (outline_out_px outside, the inset inside) and the drawing
# (polygon / "+" strokes) plus the chosen objects (params.exclude_objects) are EXCLUDED from it: the user marks where
# ViTMatte must not work (skin, face, a hand in front), not where the hair is. The band follows the object by definition.
REGION_MODES = ("draw", "outline")
OUTLINE_OUT_PX = 40          # band outside the core edge (px) = engine/bandsys.BAND_DEFAULTS["out_px"] (one set of defaults)
EXCLUDE_GROW_PX = 6          # excluded objects grown by this many px (their soft edge stays outside ViTMatte)
OUTLINE_TILE_PX = 448        # outline mode: ViTMatte on tiles of this side (plate px) – the whole outline in one crop
#                              would be scaled down (CLIP2_A: 815 px union -> x1.24 instead of x2.25)


def band_is_unknown(z: "Zone") -> bool:
    """Whole-outline (band) zones, user decision 2026-10-25: the band IS the unknown of the ViTMatte trimap – inside it
    the model replaces the core (sure FG = the core outside the band, nothing forced back to 1), the feathers blend at
    its edges only (engine/bandsys). Until then the sure FG was the core eroded by the inset slider and the result was
    forced to 1 there, so band points whose inner whisker reached deeper than the inset (background between the strands
    that SAM 3.1 took for hair) stayed 1 whatever the model said."""
    return z.criterion == "matte" and region_mode(z) == "outline"


def zone_whole(z: "Zone") -> bool:
    """„Wcięcie: cały rdzeń” of a whole-outline zone (params `whole_core`; shared: engine/objband.zone_params) – the band
    covers the whole core, the inner band edge = the middle of the core (engine/bandsys `whole`, depth_deficit)."""
    return band_is_unknown(z) and bool((z.params or {}).get("whole_core"))


def zone_feather_at(z: "Zone", f: int) -> Any:
    """The inset feather of a zone in frame f: with „cały rdzeń” and band points the map shaped by the points'
    inner whiskers (set by outline_regions), else zone_feather_px."""
    m = (getattr(z, "whole_feather", None) or {}).get(f) if zone_whole(z) else None
    return zone_feather_px(z) if m is None else m


def zone_force_core(z: "Zone", core: np.ndarray, region: np.ndarray, force_px: int) -> np.ndarray:
    """Where a ViTMatte zone result is forced to 1 (full run and frame preview alike): the core outside the band for a
    whole-outline zone (`band_is_unknown`), else the core eroded by `force_px` (0 = the whole core, as before)."""
    c = np.asarray(core).astype(bool)
    if band_is_unknown(z):
        return c & ~np.asarray(region).astype(bool)
    if force_px > 0:
        kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * force_px + 1, 2 * force_px + 1))
        return cv2.erode(c.astype(np.uint8), kf).astype(bool)
    return c


def region_mode(z: "Zone") -> str:
    m = str((z.params or {}).get("region", "draw") or "draw")
    if m not in REGION_MODES:
        raise ValueError(f"strefa {z.zone_id}: params.region musi być jednym z {REGION_MODES}, jest {m!r}")
    return m


def outline_in_px(params: Optional[Dict[str, Any]]) -> int:
    """Band inside the core edge: explicit outline_in_px, else the inset slider „Wcięcie w rdzeń” (fg_erode_px /
    force_core_erode_px) – the slider IS the band inside, as in stages 3 / 4 / 7 (user decision 2026-10-24; until then
    a hidden +4 px margin was added, so 0 still put the band 4 px into the core)."""
    p = params or {}
    if p.get("outline_in_px") is not None:
        return max(0, int(p["outline_in_px"]))
    from engine.bandsys import BAND_DEFAULTS
    return max(int(p.get("fg_erode_px", BAND_DEFAULTS["in_px"]) or 0), int(p.get("force_core_erode_px", 0) or 0))


def band_scales(params: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """(outer, inner) scale of the band points' widths (user decision 2026-10-21: the sliders „Pas na zewnątrz” /
    „Wcięcie w rdzeń” keep sizing the whole band, the points only shape it): the slider values now over their values
    when the first point was set (params band_ref_out, band_ref_in; engine/objband.scales); no reference = 1."""
    from engine.objband import BandSettings, scales
    p = params or {}
    return scales(BandSettings(float(p.get("outline_out_px", OUTLINE_OUT_PX)), float(outline_in_px(p)), 0.0, 0.0,
                               band_ref_out=p.get("band_ref_out"), band_ref_in=p.get("band_ref_in")))


def band_refs(params: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """The slider values a new set of band points is measured against (see band_scales)."""
    p = params or {}
    return {"band_ref_out": float(p.get("outline_out_px", OUTLINE_OUT_PX)), "band_ref_in": float(outline_in_px(p))}


def _disk_k(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def outline_band(core: np.ndarray, out_px: int, in_px: int, whole: bool = False) -> np.ndarray:
    """Pixels within out_px outside the core edge or within in_px inside it (`whole`: the whole core inside)."""
    c = core.astype(np.uint8)
    outer = cv2.dilate(c, _disk_k(out_px)).astype(bool) if out_px > 0 else core.astype(bool)
    if whole:
        return outer | core.astype(bool)
    inner = cv2.erode(c, _disk_k(in_px)).astype(bool) if in_px > 0 else core.astype(bool)
    return outer & ~inner


def outline_regions(z: "Zone", cores: Dict[int, np.ndarray], drawn: Dict[int, np.ndarray],
                    exclude: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                    band_at: Optional[Callable[[int], List[Any]]] = None) -> Dict[int, np.ndarray]:
    """Region of an outline-mode zone per frame: outline_band(core) minus the drawing (exclusion, propagated like any
    zone drawing) minus the excluded objects' masks grown by exclude_grow_px. `band_at(f)` (2026-10-21): the visible
    control points of frame f (engine/bandmap.py) – the band width then varies along the outline; none = constant."""
    p = z.params or {}
    out_px = int(p.get("outline_out_px", OUTLINE_OUT_PX))
    in_px = outline_in_px(p)
    whole = zone_whole(z)   # „cały rdzeń”: the band covers the whole core
    grow = int(p.get("exclude_grow_px", EXCLUDE_GROW_PX))
    out: Dict[int, np.ndarray] = {}

    def band(c, pts):
        if pts:
            from engine.bandmap import variable_band, width_maps
            om, im = width_maps(c, pts, out_px, 0 if whole else in_px)
            return variable_band(c, om, np.full_like(im, np.inf) if whole else im)
        return outline_band(c, out_px, in_px, whole)
    # the band of each frame is independent and cv2 / numpy release the GIL: frames in a thread pool (plan
    # docs/PLAN_EDGE_SPEED.md A6b, 2026-10-28 – 0.23 s/frame on a 4K sh060 frame); the same function per frame, so the
    # same bits; `band_at` / `exclude` / the subtraction stay in this thread, in frame order
    pts_of = {f: (band_at(f) if band_at is not None else None) for f in cores}
    if whole:   # „cały rdzeń”: the inset feather of each frame shaped by the points' inner whiskers (zone_feather_at)
        from engine.bandsys import whole_feather
        z.whole_feather = {f: whole_feather(cores[f], zone_feather_px(z), pts_of[f], in_px) for f in cores if pts_of[f]}
    if len(cores) > 1:
        from concurrent.futures import ThreadPoolExecutor
        from engine.qa import default_workers
        pool = ThreadPoolExecutor(max_workers=min(len(cores), default_workers()))
        futs = {f: pool.submit(band, c, pts_of[f]) for f, c in cores.items()}
        get = lambda f: futs.pop(f).result()
    else:
        pool = None
        get = lambda f: band(cores[f], pts_of[f])
    try:
        for f in cores:
            reg = get(f)
            d = drawn.get(f)
            if d is not None:
                reg &= ~d.astype(bool)
            m = exclude(f) if exclude is not None else None
            if m is not None and m.any():
                reg &= ~(cv2.dilate(m.astype(np.uint8), _disk_k(grow)).astype(bool) if grow > 0 else m.astype(bool))
            out[f] = reg
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return out


TILE_AUTO_SIDE = 672         # a drawn zone whose crop is larger than this is tiled too (a band painted round a head)


def vitmatte_tile_px(params: Optional[Dict[str, Any]]) -> int:
    """ViTMatte tile side: params.tile_px; default OUTLINE_TILE_PX in outline mode; otherwise -1 = automatic (tiles of
    OUTLINE_TILE_PX only when the zone crop is larger than TILE_AUTO_SIDE – smaller zones: one crop, as before)."""
    p = params or {}
    if p.get("tile_px") is not None:
        return max(0, int(p["tile_px"]))
    if str(p.get("vitmatte_model") or "") in BIREFNET_MODEL_PX:
        return 0                       # BiRefNet: one crop, never tiles (see BIREFNET_MODEL_PX)
    return OUTLINE_TILE_PX if str(p.get("region", "draw")) == "outline" else -1


def _tile_starts(lo: int, hi: int, size: int, t: int, step: int) -> List[int]:
    """Start positions of tiles of side t covering [lo, hi) inside [0, size)."""
    if hi - lo <= t:
        return [int(min(max(0, (lo + hi) // 2 - t // 2), size - t))]
    st = list(range(lo, hi - t, step)) + [hi - t]
    return sorted({int(min(max(0, x), size - t)) for x in st})


def _tile_weight(t: int, ov: int, x0: int, y0: int, W: int, H: int) -> np.ndarray:
    """Blend weight of a tile: linear ramp over `ov` px at the sides that are not the plate border."""
    i = np.arange(t, dtype=np.float32)
    def ramp(start: int, size: int) -> np.ndarray:
        r = np.ones(t, np.float32)
        if start > 0:
            r = np.minimum(r, (i + 1) / ov)
        if start + t < size:
            r = np.minimum(r, (t - i) / ov)
        return np.clip(r, 1e-3, 1.0)
    return np.outer(ramp(y0, H), ramp(x0, W)).astype(np.float32)


def matanyone_zone(z: Zone, frames: Sequence[int], core: Dict[int, np.ndarray], regions: Dict[int, np.ndarray],
                   init_alpha: Optional[np.ndarray], ma_run: Callable[..., Dict[int, np.ndarray]],
                   holdout: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                   bg: Optional[Callable[[int], np.ndarray]] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """MatAnyone2 on the crop of the zone (docs/PLAN_HAIR_TOOL.md section 3, pulled forward from step 11 on the user's
    request 2026-10-13): a tight crop around the union of the zone regions, scaled UP to ~1008 px (strands get 2-3 x the
    pixels of the full-frame pass), one MatAnyone2RunKeyframes pass over the zone frames with the key at `z.frame` =
    core ∪ (region ∩ init_alpha > 0.5) (the difference-key hint of the strands); result only inside the region, core = 1,
    forced BG / holdout = 0. `ma_run(seg, frames, core, keys, start, scale, tag)` = Finalizer._matanyone_segment."""
    from engine.crop import paste_crop
    frames = list(frames)
    H, W = core[frames[0]].shape
    seg = zone_crop_segment(regions, frames, z.zone_id)
    if seg is None:
        return {}, {"frames": 0}
    start = int(z.frame) if z.frame in frames else frames[0]
    hint = (regions[start] & (init_alpha > 0.5)) if init_alpha is not None else np.zeros((H, W), bool)
    key = core[start] | hint
    scale = min(MA_ZONE_MAX_UP, MA_ZONE_TARGET_PX / float(seg.side))
    alphas_crop = ma_run(seg, frames, core, {start: key}, start, scale, f"zone_{z.zone_id}")
    out: Dict[int, np.ndarray] = {}
    for f in frames:
        a = paste_crop(alphas_crop[f], seg, W, H)
        reg = regions[f]
        a = np.where(reg, a, 0.0).astype(np.float32)
        a[reg & core[f]] = 1.0
        if bg is not None:
            a[reg & bg(f) & ~core[f]] = 0.0
        if holdout is not None and z.holdout:
            hm = holdout(f)
            if hm is not None:
                a[reg & hm & ~core[f]] = 0.0
        out[f] = a
    stats = {"frames": len(out), "engine": "matanyone", "preset": z.preset, "crop": list(seg.crop), "side": seg.side,
             "scale": round(scale, 3), "key_frame": start, "hint_px": int(hint.sum()), "bbox": [seg.crop[0], seg.crop[1],
             seg.crop[0] + seg.side, seg.crop[1] + seg.side]}
    return out, stats


VITMATTE_TEMPORAL = ("none", "median", "flow", "flow_mean")   # z.params["temporal"] of the `vitmatte` engine (default: median)
# flow_mean (user decision 2026-10-15): levels (alpha_black / alpha_white) BEFORE the smoothing, then refine.temporal_mean_flow
VITMATTE_TEMPORAL_DEFAULT = "median"
TRIMAP_SOURCES = ("core", "base")   # z.params["trimap_source"] of the `vitmatte` engine: SAM3 core | version's base alpha
BASE_TRIMAP_FG = 0.98                # trimap_source=base: the zone's "core" = base alpha >= this
FINISH_KEYS = ("alpha_black", "alpha_white", "alpha_blur_px",   # zone alpha finish (user 2026-10-14), not in the engine hash
               "alpha_black_after")                             # ViTMatte: black levels again after the smoothing (2026-10-15)
# flow confidence of temporal = flow_mean (experiment 2026-10-15; zone param -> refine.temporal_mean_flow kwarg), off by default
TEMPORAL_CONF_PARAMS = (("temporal_fb_sigma", "fb_sigma", float), ("temporal_photo_sigma", "photo_sigma", float),
                        ("temporal_tex", "tex_k", float), ("temporal_conf_mode", "conf_mode", str),
                        ("temporal_multiscale", "multiscale", int),
                        # plan docs/PLAN_EDGE_SPEED.md B2 (2026-10-28): DIS on grays shrunk by this factor (1 = as before)
                        ("temporal_flow_scale", "flow_scale", float))
# plan B1 (2026-10-28): z.params["temporal_crop"] – the smoothing works on the union bbox of the zone over the whole shot
# ("union", as before) or on a crop of one size that follows the zone frame by frame ("follow")
TEMPORAL_CROPS = ("union", "follow")
# user decision 2026-10-28 after the A/B (local_results/ab_zone_smooth/RAPORT.md: the same result, sh060 1.9x faster,
# CLIP2_A strands 11-13 % less flicker): `flow_mean` without these params in the zone = follow crop + the DIS flow at
# the plate's resolution (temporal_flow_scale = 1 / temporal_scale); a zone that saves them keeps its own values
FLOW_MEAN_CROP_DEFAULT = "follow"
FOLLOW_CROP_MARGIN = 0.12            # "follow": the crop = the largest per-frame zone bbox + this share of it on each side
HYBRID_BASES = ("", "matanyone")    # z.params["hybrid"] of the `vitmatte` engine (experiment 2026-10-15), "" = off


def edge_model_runner(name: Optional[str]) -> Any:
    """Runner of the per-frame `vitmatte` zone engine for z.params["vitmatte_model"] (key kept for saved projects):
    a ViTMatte model (base / small / *-distinctions – Adobe-trained weights, non-commercial) or, since 2026-10-21,
    `birefnet_hr_matting` / `lucida` (BiRefNet, MIT weights; engine/birefnet.py). Same contract: crop + trimap -> alpha."""
    from engine import birefnet
    if name and str(name) in birefnet.MODELS:
        return birefnet.get_runner(birefnet.MODELS[str(name)])
    from engine.vitmatte import get_runner, model_id_for
    return get_runner(model_id_for(name))


# BiRefNet sees the whole object (salient-object model): on 448 px tiles it lost the arm edges and let the sky into
# the strands (station 2026-10-21, Woman_01 F56); one crop at about its own input side was clean -> its defaults
BIREFNET_MODEL_PX = {"birefnet_hr_matting": 2016, "lucida": 1008}


def edge_model_px(params: Optional[Dict[str, Any]]) -> int:
    """Crop side the zone's edge model sees: params.model_px; default 1008 (ViTMatte) or BIREFNET_MODEL_PX."""
    p = params or {}
    if p.get("model_px") is not None:
        return int(p["model_px"])
    return BIREFNET_MODEL_PX.get(str(p.get("vitmatte_model") or ""), MA_ZONE_TARGET_PX)


def ridge_map(a: np.ndarray, sigma: float, k: float = 0.1) -> np.ndarray:
    """Thin bright lines (strands) in an alpha: -(smaller Hessian eigenvalue) of the alpha smoothed at `sigma`, scaled by
    sigma^2 (scale-normalised) and / k -> 0..1, dilated 1 px (the response is narrower than the strand)."""
    ks = int(round(sigma * 3)) * 2 + 1
    g = cv2.GaussianBlur(a.astype(np.float32), (ks, ks), sigma)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    lmin = 0.5 * (dxx + dyy - np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
    r = np.clip(-lmin * sigma * sigma / max(k, 1e-6), 0.0, 1.0).astype(np.float32)
    return cv2.dilate(r, np.ones((3, 3), np.uint8))


def hybrid_blend(vm: np.ndarray, ma: np.ndarray, lo: float = 0.05, hi: float = 0.25, ridge_px: float = 0.0,
                 blur_px: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid zone alpha (experiment 2026-10-15): the temporally stable MatAnyone2 alpha `ma` is the base, the ViTMatte
    alpha `vm` takes over only where it is ABOVE the base – strands MatAnyone2 lost: weight w = linear ramp of (vm - ma)
    from lo to hi (a continuous function of the two alphas -> no seam), optionally x ridge_map(vm, ridge_px) (only thin
    lines; spatial -> blur_px widens it: w = max(w, blur(w)), the strand itself stays at full weight);
    result = ma + w * max(vm - ma, 0) (between the two, never beyond). Returns (alpha, w)."""
    d = vm.astype(np.float32) - ma.astype(np.float32)
    w = np.clip((d - lo) / max(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)
    if ridge_px > 0:
        w = w * ridge_map(vm, ridge_px)
    if blur_px > 0:
        kb = int(round(blur_px * 3)) * 2 + 1
        w = np.maximum(w, cv2.GaussianBlur(w, (kb, kb), blur_px))
    out = ma.astype(np.float32) + np.clip(w, 0.0, 1.0) * np.maximum(d, 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32), w


def finish_params(params: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(engine params without the finish keys, finish params)."""
    p = dict(params or {})
    fin = {k: p.pop(k) for k in FINISH_KEYS if k in p}
    p.pop("outer_feather_px", None)       # only how the zone is laid over its base (apply_zones weights), not the engine
    return p, fin


def levels_alpha(a: np.ndarray, black: Optional[float], white: Optional[float]) -> np.ndarray:
    """Levels of the finish sliders: <= black -> 0, >= white -> 1, linear in between (continuous, no step)."""
    b = float(black or 0.0)
    wt = float(white) if white is not None else 1.0
    if b <= 0.0 and wt >= 1.0:
        return a
    return np.clip((a - b) / (max(wt, b + 1e-3) - b), 0.0, 1.0).astype(np.float32)


def finish_zone_alpha(a: np.ndarray, region: np.ndarray, core: np.ndarray, params: Optional[Dict[str, Any]]) -> np.ndarray:
    """Levels + blur of a zone alpha (user 2026-10-14: sliders to remove haze / garbage): values <= alpha_black -> 0,
    >= alpha_white -> 1, linear in between (continuous, no step), then a Gaussian blur of alpha_blur_px; only inside
    the region (outside = as given). The core is forced back to 1 AFTER the blur and the result is max(alpha, core),
    so the clipped white joins the core without a hard cut at its edge."""
    p = params or {}
    black = float(p.get("alpha_black", 0.0) or 0.0)
    white = float(p.get("alpha_white", 1.0) if p.get("alpha_white") is not None else 1.0)
    blur = float(p.get("alpha_blur_px", 0.0) or 0.0)
    if black <= 0.0 and white >= 1.0 and blur <= 0.0:
        return a
    out = a.astype(np.float32).copy()
    if black > 0.0 or white < 1.0:
        w = max(white, black + 1e-3)
        out = np.clip((out - black) / (w - black), 0.0, 1.0).astype(np.float32)
    if blur > 0.0:
        k = int(round(blur * 3)) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), blur)
    reg = region.astype(bool)
    out = np.where(reg, out, a).astype(np.float32)
    out[reg & core.astype(bool)] = 1.0
    return out


def vitmatte_zone(z: Zone, frames: Sequence[int], core: Dict[int, np.ndarray], regions: Dict[int, np.ndarray],
                  images: Callable[[int], np.ndarray], runner: Any,
                  holdout: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                  bg: Optional[Callable[[int], np.ndarray]] = None,
                  progress: Optional[Callable[[int, int], None]] = None,
                  only: Optional[Sequence[int]] = None, fg_erode_px: int = 2, model_px: int = MA_ZONE_TARGET_PX,
                  unknown_value: float = 0.5, probs: Optional[Dict[int, np.ndarray]] = None,
                  force_core: Optional[Dict[int, np.ndarray]] = None, tile_px: int = 0,
                  fg_sure: Optional[Dict[int, np.ndarray]] = None, band_unknown: bool = False
                  ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
    """ViTMatte per frame on the SAME crop as the `matanyone` engine (square union crop of the zone over the shot,
    scaled up to ~1008 px, <= 3x; experiment 2026-10-13). Trimap per frame = engine/localmatte.build_trimap on the crop
    (FG = core eroded, unknown = region ∪ core rim, forced BG = "-" strokes / holdout); result only inside the region,
    core = 1, forced BG / holdout = 0. `only` = frames to compute (the rest come from the cache); returns
    (alphas per computed frame, trimaps per computed frame (full-frame u8), stats).
    `probs` (frame -> SAM 3.1 probability map, uint8 full frame, consistent with the core; keys absent): those frames get
    the trimap from SAM 3.1's uncertainty (build_trimap prob=) instead of the eroded core. `force_core` (frame -> mask):
    where the result is forced to 1 (default: the whole core; an eroded core leaves its rim to the model).
    `tile_px` > 0 (outline mode, 2026-10-15): per frame, square tiles of that side over the region's bbox (overlap 1/6,
    linear blend), each scaled up like the one crop (model_px / tile_px, <= 3x); tiles without unknown pixels skipped.
    `fg_sure` (frame -> mask; parts of the object, e.g. a hand that is one matte with the head): sure foreground of the
    trimap AFTER the FG erosion – the hair right next to the part is never left half-transparent.
    `band_unknown` (whole-outline zones, 2026-10-25): the region is the trimap's unknown and nothing in it is forced
    to 1 (build_trimap band_unknown; the default forcing = the core outside the region)."""
    from engine.crop import extract_crop, paste_crop
    from engine.localmatte import build_trimap
    frames = list(frames)
    H, W = core[frames[0]].shape
    seg = zone_crop_segment(regions, frames, z.zone_id)
    if seg is None:
        return {}, {}, {"frames": 0}
    # model_px (Nuke "Downrez" in reverse): the crop side the model sees – 1008 = the MatAnyone-style up-scale (<= 3x),
    # smaller values down-scale (e.g. 504 ~ Nuke 1/2 on a 1008 crop)
    scale = min(MA_ZONE_MAX_UP, float(model_px) / float(seg.side))
    ss = max(16, int(round(seg.side * scale)))
    todo = [f for f in (only if only is not None else frames) if regions[f].any()]
    out: Dict[int, np.ndarray] = {}
    tris: Dict[int, np.ndarray] = {}
    secs = 0.0
    rs = lambda m: cv2.resize(extract_crop(m.astype(np.float32), seg), (ss, ss), interpolation=cv2.INTER_LINEAR) > 0.5
    rp = lambda p: cv2.resize(extract_crop(p.astype(np.float32) / (255.0 if p.dtype == np.uint8 else 1.0), seg), (ss, ss),
                              interpolation=cv2.INTER_LINEAR)
    n_soft = 0
    if tile_px < 0:          # automatic: tiles only for a large zone (one crop would scale the strands down)
        tile_px = OUTLINE_TILE_PX if seg.side > TILE_AUTO_SIDE else 0
    T = int(min(tile_px, W, H)) if tile_px > 0 else 0
    if T:
        scale = min(MA_ZONE_MAX_UP, float(model_px) / float(T))
        ss = max(16, int(round(T * scale)))
        ov = max(8, T // 6)
    n_tiles = 0
    run_model = lambda img_, tri_: (runner.run(img_, tri_, unknown_value) if abs(unknown_value - 0.5) > 1e-6
                                    else runner.run(img_, tri_))

    # plan docs/PLAN_EDGE_SPEED.md A6: wall seconds per phase of the frame loop (stats "phases"); the model call is
    # split further by the runner itself ("runner_phases": prepare / IPC / forward / finish)
    from engine.vitmatte import prof_delta, prof_snapshot
    ph: Dict[str, float] = {}

    def tick(key: str, t0: float) -> float:
        t1 = time.perf_counter()
        ph[key] = ph.get(key, 0.0) + (t1 - t0)
        return t1
    prof0 = prof_snapshot(runner)
    wall0 = time.perf_counter()

    def run_tiles(imgs_: list, tris_: list) -> list:
        """All tiles of a frame: one worker request when the runner has `run_many` (same forward per tile, batch 1 –
        the alphas are those of `run`), else one by one (mocks, in-process runners without it)."""
        many = getattr(runner, "run_many", None)
        if many is not None and len(imgs_) > 1:
            return list(many(imgs_, tris_, unknown_value))
        return [run_model(im_, tr_) for im_, tr_ in zip(imgs_, tris_)]
    for i, f in enumerate(todo):
        if progress is not None:
            progress(i, len(todo))
        reg = regions[f]
        tp = time.perf_counter()
        bgm = bg(f) if bg is not None else np.zeros((H, W), bool)
        hm = holdout(f) if (holdout is not None and z.holdout) else None
        forced = bgm | (hm if hm is not None else False)
        tp = tick("masks", tp)
        if T:
            img_f = images(f)
            tp = tick("decode", tp)
            pff = (probs[f].astype(np.float32) / (255.0 if probs[f].dtype == np.uint8 else 1.0)) if probs and f in probs else None
            n_soft += pff is not None
            acc = np.zeros((H, W), np.float32)
            wsum = np.zeros((H, W), np.float32)
            ys_, xs_ = np.where(reg)
            tiles: list = []          # (slice, tx, ty, image tile, trimap u8) – the tiles with unknown pixels, in grid order
            for ty in _tile_starts(int(ys_.min()), int(ys_.max()) + 1, H, T, T - ov):
                for tx in _tile_starts(int(xs_.min()), int(xs_.max()) + 1, W, T, T - ov):
                    sl = (slice(ty, ty + T), slice(tx, tx + T))
                    if not reg[sl].any():
                        continue
                    rt = lambda m: cv2.resize(m[sl].astype(np.float32), (ss, ss), interpolation=cv2.INTER_LINEAR) > 0.5
                    pt = cv2.resize(pff[sl], (ss, ss), interpolation=cv2.INTER_LINEAR) if pff is not None else None
                    tm = build_trimap(rt(core[f]), rt(reg), rt(forced), max(1, int(round(fg_erode_px * scale))), prob=pt,
                                      band_unknown=band_unknown)
                    if fg_sure is not None and f in fg_sure:
                        fs = rt(fg_sure[f])
                        tm.fg |= fs
                        tm.unknown &= ~fs
                    if not tm.unknown.any():
                        tp = tick("trimap", tp)
                        continue
                    tp = tick("trimap", tp)
                    img = cv2.resize(img_f[sl], (ss, ss), interpolation=cv2.INTER_LINEAR if ss > T else cv2.INTER_AREA)
                    tiles.append((sl, tx, ty, img, tm.as_u8()))
                    tp = tick("tile_image", tp)
            t0 = time.time()
            outs = run_tiles([t_[3] for t_ in tiles], [t_[4] for t_ in tiles]) if tiles else []
            secs += time.time() - t0
            tp = time.perf_counter()
            n_tiles += len(tiles)
            for (sl, tx, ty, _img, _tri), a_t in zip(tiles, outs):   # grid order: the same sums as tile by tile
                a_t = cv2.resize(a_t.astype(np.float32), (T, T), interpolation=cv2.INTER_AREA if ss > T else cv2.INTER_LINEAR)
                wt = _tile_weight(T, ov, tx, ty, W, H)
                acc[sl] += a_t * wt
                wsum[sl] += wt
            a = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0.0)
            a = np.where(reg, a, 0.0).astype(np.float32)
            a[reg & (force_core[f] if force_core is not None else (core[f] & ~reg if band_unknown else core[f]))] = 1.0
            a[reg & bgm & ~core[f]] = 0.0
            if hm is not None:
                a[reg & hm & ~core[f]] = 0.0
            out[f] = a
            tp = tick("assemble", tp)
            tfull = build_trimap(core[f], reg, forced, max(1, int(fg_erode_px)), prob=pff, band_unknown=band_unknown)
            if fg_sure is not None and f in fg_sure:
                tfull.fg |= fg_sure[f]
                tfull.unknown &= ~fg_sure[f]
            tris[f] = tfull.as_u8()
            tick("trimap_full", tp)
            continue
        pf = rp(probs[f]) if probs and f in probs else None
        n_soft += pf is not None
        tm = build_trimap(rs(core[f]), rs(reg), rs(forced), max(1, int(round(fg_erode_px * scale))), prob=pf,
                          band_unknown=band_unknown)
        if fg_sure is not None and f in fg_sure:
            fs = rs(fg_sure[f])
            tm.fg |= fs
            tm.unknown &= ~fs
        tp = tick("trimap", tp)
        img_f = images(f)
        tp = tick("decode", tp)
        img = extract_crop(img_f, seg)
        if ss != seg.side:
            img = cv2.resize(img, (ss, ss), interpolation=cv2.INTER_LINEAR if ss > seg.side else cv2.INTER_AREA)
        tick("tile_image", tp)
        t0 = time.time()
        a_crop = run_model(img, tm.as_u8())
        secs += time.time() - t0
        tp = time.perf_counter()
        a = paste_crop(a_crop, seg, W, H)
        a = np.where(reg, a, 0.0).astype(np.float32)
        a[reg & (force_core[f] if force_core is not None else (core[f] & ~reg if band_unknown else core[f]))] = 1.0
        a[reg & bgm & ~core[f]] = 0.0
        if hm is not None:
            a[reg & hm & ~core[f]] = 0.0
        out[f] = a
        tp = tick("assemble", tp)
        t8 = cv2.resize(tm.as_u8(), (seg.side, seg.side), interpolation=cv2.INTER_NEAREST) if ss != seg.side else tm.as_u8()
        tris[f] = paste_crop(t8.astype(np.float32), seg, W, H).astype(np.uint8)
        tick("trimap_full", tp)
    if progress is not None:
        progress(len(todo), len(todo))
    ph["model"] = secs
    wall = time.perf_counter() - wall0
    ph["other"] = max(0.0, wall - sum(ph.values()))      # progress callbacks (cancel check), loop overhead
    from engine.vitmatte import runner_info
    stats = {"frames": len(out), "engine": "vitmatte", "preset": z.preset, "crop": list(seg.crop), "side": seg.side,
             "scale": round(scale, 3), "model_px": ss, "unknown_value": unknown_value, "soft_trimap_frames": int(n_soft),
             **({"tile_px": T, "tiles": n_tiles, "tiles_per_frame": round(n_tiles / max(1, len(out)), 1)} if T else {}),
             "seconds": round(secs, 1),
             "sec_per_frame": round(secs / max(1, len(out)), 3), **runner_info(runner),
             "wall_seconds": round(wall, 1), "phases": {k: round(v, 2) for k, v in ph.items()},
             "runner_phases": prof_delta(runner, prof0),
             "bbox": [seg.crop[0], seg.crop[1], seg.crop[0] + seg.side, seg.crop[1] + seg.side]}
    return out, tris, stats


def follow_crops(regions: Dict[int, np.ndarray], fl: Sequence[int],
                 margin: float = FOLLOW_CROP_MARGIN) -> Dict[int, Tuple[int, int, int, int]]:
    """Plan B1 (2026-10-28): per frame a crop (y0, y1, x0, x1) of ONE size for all frames – the largest per-frame bbox of
    the zone region + `margin` of it on each side (capped by the frame) – centred on that frame's region bbox and pushed
    inside the frame. The smoothing then aligns neighbours inside crops that follow the object (a pan does not grow
    the area as the union bbox of the whole shot does); DIS measures the motion between the crops directly."""
    H, W = regions[fl[0]].shape
    boxes = {}
    for f in fl:
        ys, xs = np.where(regions[f])
        if len(ys):
            boxes[f] = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    if not boxes:
        return {f: (0, H, 0, W) for f in fl}
    bh = max(b[1] - b[0] for b in boxes.values())
    bw = max(b[3] - b[2] for b in boxes.values())
    ch = min(H, bh + 2 * int(round(margin * bh)))
    cw = min(W, bw + 2 * int(round(margin * bw)))
    u = (min(b[0] for b in boxes.values()), max(b[1] for b in boxes.values()),
         min(b[2] for b in boxes.values()), max(b[3] for b in boxes.values()))
    if ch * cw >= (u[1] - u[0]) * (u[3] - u[2]):
        # the zone hardly moves: the union bbox is not larger than the following crop -> the union (= "union" mode)
        return {f: u for f in fl}
    out: Dict[int, Tuple[int, int, int, int]] = {}
    last = next(iter(boxes.values()))
    for f in fl:
        b = boxes.get(f, last)
        last = b
        cy, cx = (b[0] + b[1]) // 2, (b[2] + b[3]) // 2
        y0 = min(max(0, cy - ch // 2), H - ch)
        x0 = min(max(0, cx - cw // 2), W - cw)
        out[f] = (y0, y0 + ch, x0, x0 + cw)
    return out


def smooth_zone_alphas(alphas: Dict[int, np.ndarray], fl: Sequence[int], bbox: Tuple[int, int, int, int],
                       regions: Dict[int, np.ndarray], force: Dict[int, np.ndarray], temporal: str, t_radius: int,
                       t_scale: float, t_conf: Dict[str, Any], fin: Dict[str, Any], images: Callable[[int], np.ndarray],
                       fg_sure: Optional[Dict[int, np.ndarray]] = None,
                       progress: Optional[Callable[[int, int], None]] = None,
                       crops: Optional[Dict[int, Tuple[int, int, int, int]]] = None) -> None:
    """Temporal smoothing of a ViTMatte zone IN PLACE (alphas[f] replaced for f in `fl`): the crops of the union bbox
    (y0, y1, x0, x1) – parts of the object forced to 1 (`fg_sure`), flow_mean: levels first – smoothed by
    refine.temporal_median / temporal_median_flow / temporal_mean_flow (`temporal`), on the bbox stack up-scaled
    `t_scale` x when > 1 (user decision 2026-10-13: strands wider, flow holds), of which only the smoothing DELTA comes
    back down (INTER_AREA: unchanged frames stay bit-identical); outside the region 0, the forced core 1.
    Since 2026-10-25 (plan A1) the frames are sources, not stacks: each crop / gray / up-scale is made when the
    smoother's window first asks for it and dropped once the window has moved on – four full stacks (41 GB on a 4K
    shot) became ~2 r + 1 frames; the smoother hands the frames back one by one (`emit`) and reports `progress` per
    frame (cancel). Every value is computed by the same calls on the same inputs as the former stack code, so the
    result is bit-identical (tests/test_zone_smooth_window.py). `crops` (plan B1, follow_crops): a crop per frame, all of
    one size, instead of the one `bbox` – frame i is smoothed in ITS crop, the neighbours are read in theirs."""
    if temporal == "none" or len(fl) < 3:
        return
    from engine.refine import temporal_mean_flow, temporal_median, temporal_median_flow
    box_of = (lambda i: crops[fl[i]]) if crops else (lambda i: bbox)
    y0, y1, x0, x1 = box_of(0)
    bh, bw = y1 - y0, x1 - x0
    up = t_scale > 1.0
    up_size = (int(round(bw * t_scale)), int(round(bh * t_scale)))
    levels = temporal == "flow_mean" and ("alpha_black" in fin or "alpha_white" in fin)
    base: Dict[int, np.ndarray] = {}        # i -> the 1x crop the smoothing starts from (window only)

    def base_i(i: int) -> np.ndarray:
        b = base.get(i)
        if b is None:
            f = fl[i]
            y0, y1, x0, x1 = box_of(i)
            b = alphas[f][y0:y1, x0:x1]
            if fg_sure and f in fg_sure:
                # parts of the object are foreground (1) for the smoothing: a hand moving over the hair must not pull
                # the aligned neighbours' hair alpha down to 0 at its edge (they are cut out of the result again below)
                b = b.copy()
                b[fg_sure[f][y0:y1, x0:x1]] = 1.0
            if levels:   # garbage clipped in its own frame first: a zeroed speck is never averaged onto the neighbours
                b = levels_alpha(b, fin.get("alpha_black"), fin.get("alpha_white"))
            base[i] = b
        return b

    def up_i(i: int) -> np.ndarray:
        b = base_i(i)
        return cv2.resize(b, up_size, interpolation=cv2.INTER_LINEAR) if up else b

    class _Src:
        def __init__(self, get: Callable[[int], np.ndarray]):
            self.get = get

        def __len__(self) -> int:
            return len(fl)

        def __getitem__(self, i: int) -> np.ndarray:
            return self.get(i)

    def gray_i(i: int) -> np.ndarray:
        y0, y1, x0, x1 = box_of(i)
        g = cv2.cvtColor(np.clip(images(fl[i])[y0:y1, x0:x1], 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, up_size, interpolation=cv2.INTER_LINEAR) if up else g

    def emit(i: int, sm: np.ndarray) -> None:
        f = fl[i]
        if up:
            # only the smoothing DELTA comes back down (no resampling blur of the full-res alpha: unchanged frames stay
            # bit-identical): INTER_AREA = box mean of the 2x delta
            delta = cv2.resize(sm - up_i(i), (bw, bh), interpolation=cv2.INTER_AREA)
            sm = np.clip(base_i(i) + delta, 0.0, 1.0).astype(np.float32)
        a = alphas[f].copy()
        y0, y1, x0, x1 = box_of(i)
        a[y0:y1, x0:x1] = sm
        a[~regions[f]] = 0.0
        a[regions[f] & force[f]] = 1.0
        alphas[f] = a
        for k in [k for k in base if k < i - t_radius]:
            del base[k]

    src = _Src(up_i)
    if temporal == "median":
        temporal_median(src, t_radius, 0.35, progress=progress, emit=emit)
    elif temporal == "flow_mean":
        temporal_mean_flow(src, _Src(gray_i), t_radius, progress=progress, emit=emit, **t_conf)
    else:
        temporal_median_flow(src, _Src(gray_i), t_radius, progress=progress, emit=emit)


def _vitmatte_sequence(z: Zone, frames: Sequence[int], core: Dict[int, np.ndarray], images: Callable[[int], np.ndarray],
                       regions: Dict[int, np.ndarray], bgm: Callable[[int], np.ndarray],
                       holdout: Optional[Callable[[int], Optional[np.ndarray]]], runner: Any,
                       progress: Optional[Callable[[int, int], None]], cache: Any, psig: Callable[[int], Any],
                       probs: Optional[Dict[int, np.ndarray]] = None,
                       base_alpha: Optional[Callable[[int], np.ndarray]] = None,
                       hybrid_base: Optional[Dict[int, np.ndarray]] = None,
                       fg_extra: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                       progress_smooth: Optional[Callable[[int, int], None]] = None
                       ) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], Dict[str, Any]]:
    """`matting=vitmatte` inside _matte_zone: per-frame cache of the RAW model result (hash without the temporal
    option, so none / median / flow share the cached frames), then the temporal smoothing chosen in
    z.params["temporal"]: `none`, `median` (refine.temporal_median, motion-gated, as the per-frame engines) or `flow`
    (refine.temporal_median_flow: ±2 frames warped by DIS flow to frame f, median) – only inside the region.
    `hybrid_base` (params.hybrid = matanyone, experiment 2026-10-15): the MatAnyone2 zone alphas; the smoothed ViTMatte
    alpha only adds what is above them (hybrid_blend), the rest of the zone is MatAnyone2."""
    from engine.zonecache import frame_hash, zone_signature
    from engine.vitmatte import runner_info
    frames = list(frames)
    shape = core[frames[0]].shape
    fg_sure = None
    if fg_extra is not None:
        # parts of this object (e.g. the hairdresser's hand that is one matte with the head, 2026-10-15): part of the
        # core (hash / forcing) and sure foreground of the trimap AFTER the FG erosion; the region already excludes them
        fg_sure = {f: m for f in frames if (m := fg_extra(f)) is not None}
        core = {f: (core[f] | fg_sure[f]) if f in fg_sure else core[f] for f in frames}
    params, fin = finish_params(z.params)
    temporal = str(params.pop("temporal", VITMATTE_TEMPORAL_DEFAULT))
    if temporal not in VITMATTE_TEMPORAL:
        raise ValueError(f"vitmatte: temporal musi być jednym z {VITMATTE_TEMPORAL}, jest {temporal!r}")
    # FG erosion of the trimap in plate px (default 2 as build_trimap): a larger value hides the per-frame jitter of the
    # SAM3 core edge inside the unknown band, so the model – not the core – decides the edge (CLIP2_A test 2026-10-13)
    fg_erode_px = int(params.get("fg_erode_px", 2))
    model_px = edge_model_px(params)          # Nuke "Downrez": 504 ~ 1/2 of the 1008 default; BiRefNet: its own side
    unknown_value = float(params.get("unknown_value", 0.5))           # Nuke "Fine Tune": trimap level of the unknown band
    # temporal_radius (default 2) and temporal_scale (default 1 = full resolution; e.g. 2 = the smoothing runs on the
    # bbox stack upscaled 2x, where strands are wider and the flow alignment holds – user decision 2026-10-13); both are
    # popped BEFORE the hash: they only change the smoothing, so the per-frame cache is shared
    t_radius = max(1, int(params.pop("temporal_radius", 2)))
    t_scale = float(params.pop("temporal_scale", 1.0))
    # flow confidence of flow_mean + hybrid with MatAnyone (experiment 2026-10-15): smoothing / blending only -> popped
    # before the hash (the raw per-frame cache is shared); absent = bit-identical to before
    t_conf = {kw: cast(params.pop(pk)) for pk, kw, cast in TEMPORAL_CONF_PARAMS if pk in params}
    t_crop = str(params.pop("temporal_crop", FLOW_MEAN_CROP_DEFAULT if temporal == "flow_mean" else "union") or "union")
    if temporal == "flow_mean" and "flow_scale" not in t_conf and t_scale > 1.0:
        t_conf["flow_scale"] = 1.0 / t_scale         # B2: the flow at the plate's resolution, brought up to the stack
    if t_crop not in TEMPORAL_CROPS:
        raise ValueError(f"vitmatte: temporal_crop musi być jednym z {TEMPORAL_CROPS}, jest {t_crop!r}")
    hybrid = str(params.pop("hybrid", "") or "")
    if hybrid not in HYBRID_BASES:
        raise ValueError(f"vitmatte: hybrid musi być jednym z {HYBRID_BASES}, jest {hybrid!r}")
    hyb = {"lo": float(params.pop("hybrid_lo", 0.05)), "hi": float(params.pop("hybrid_hi", 0.25)),
           "ridge_px": float(params.pop("hybrid_ridge_px", 0.0)), "blur_px": float(params.pop("hybrid_blur_px", 0.0))}
    # soft trimap from the SAM3 probabilities (user decision 2026-10-14; zone param soft_trimap, default on): frames
    # with a map get it; the map's digest joins the frame hash only then (no maps = the old hashes, the old cache)
    if not bool(params.get("soft_trimap", True)):
        probs = None
    # trimap_source (user idea 2026-10-15): "base" = the zone's "core" is the solid part (a >= BASE_TRIMAP_FG) of the
    # version's base alpha (MatAnyone over the whole object, temporally coherent – on CLIP2_A its rim flickers ~6x less
    # than the SAM3 core) instead of the SAM3 core; trimap / forcing / finish then follow it. Needs the base alpha
    # ("Zastosuj" on a version with alpha_base); without it the SAM3 core is used and the stats say so.
    tsrc = str(params.get("trimap_source", "core"))
    if tsrc not in TRIMAP_SOURCES:
        raise ValueError(f"vitmatte: trimap_source musi być jednym z {TRIMAP_SOURCES}, jest {tsrc!r}")
    tsrc_used = tsrc
    if tsrc == "base":
        if base_alpha is None:
            tsrc_used = "core (brak alphy bazowej – „Zastosuj” na wersji ze strefami)"
        else:
            core = {f: base_alpha(f) >= BASE_TRIMAP_FG for f in frames}
            probs = None
    # force_core_erode_px (user idea 2026-10-15, CLIP2_A F50: the zone alpha had a hard step exactly on the SAM3 core
    # edge – the core was forced back to 1 after the model): the result is forced to 1 only on the core eroded by this
    # many px, the rim is left to ViTMatte. 0 = the whole core (as before, bit-identical).
    force_px = int(params.get("force_core_erode_px", 0))
    # a whole-outline zone (2026-10-25): the band is the unknown, forced = the core outside it (zone_force_core); the
    # signature token invalidates the per-frame cache computed with the old (eroded-core) trimap once
    band_unk = band_is_unknown(z)
    force = {f: zone_force_core(z, core[f], regions[f], force_px) for f in frames}
    zsig = zone_signature({**z.to_dict(), "params": params, "vitmatte_model": getattr(runner, "model_id", "mock"),
                           **({"band_unknown": 1} if band_unk else {})})
    seg = zone_crop_segment(regions, frames, z.zone_id)
    if seg is None:
        return {}, {}, {"frames": 0, "engine": "vitmatte"}

    def extra(f: int) -> list:
        e = [int(bgm(f).sum()), list(seg.crop)]
        if fg_sure is not None and f in fg_sure:     # parts = sure FG after the erosion (2026-10-15): another trimap
            e.append("fg_sure")
        if probs and f in probs:
            import hashlib
            e.append("p:" + hashlib.sha1(np.ascontiguousarray(probs[f]).tobytes()).hexdigest()[:16])
        return e
    tq = time.perf_counter()
    hashes = {f: frame_hash(zsig, core[f], psig(f), regions[f], extra=extra(f)) for f in frames if regions[f].any()}
    seq_ph = {"hash": time.perf_counter() - tq}          # plan A6, next to the model loop's phases
    tq = time.perf_counter()
    alphas: Dict[int, np.ndarray] = {}
    todo = []
    for f, h in hashes.items():
        got = cache.get(h, f, shape) if cache is not None else None
        if got is not None:
            alphas[f] = got
        else:
            todo.append(f)
    seq_ph["cache_get"] = time.perf_counter() - tq
    st: Dict[str, Any] = {"frames": 0, "engine": "vitmatte", "preset": z.preset, "crop": list(seg.crop), "side": seg.side,
                          "seconds": 0.0, **runner_info(runner)}
    if todo:
        new, _, st = vitmatte_zone(z, frames, core, regions, images, runner, holdout, bgm, progress, only=todo,
                                   fg_erode_px=fg_erode_px, model_px=model_px, unknown_value=unknown_value, probs=probs,
                                   force_core=force if (force_px > 0 or band_unk) else None, tile_px=vitmatte_tile_px(params),
                                   fg_sure=fg_sure, band_unknown=band_unk)
        tq = time.perf_counter()
        for f, a in new.items():
            if cache is not None:
                cache.put(hashes[f], f, a, regions[f])
                back = cache.get(hashes[f], f, shape)     # 16-bit cache = the single source (cached pass bit-identical)
                if back is not None:
                    a = back
            alphas[f] = a
        seq_ph["cache_put"] = time.perf_counter() - tq
    elif progress is not None:
        progress(len(frames), len(frames))
    if not alphas:
        return {}, {}, {"frames": 0, "engine": "vitmatte"}
    fl = [f for f in frames if f in alphas]
    ys, xs = np.where(np.any(np.stack([regions[f] for f in fl]), axis=0))
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    t0 = time.time()
    smooth_zone_alphas(alphas, fl, (y0, y1, x0, x1), regions, force, temporal, t_radius, t_scale, t_conf, fin, images,
                       fg_sure=fg_sure, progress=progress_smooth,
                       crops=follow_crops(regions, fl) if t_crop == "follow" else None)
    # flow_mean applied the levels before the smoothing: only the blur is left for the finish
    levels_done = temporal == "flow_mean" and len(fl) >= 3
    fin_after = ({k: v for k, v in fin.items() if k not in ("alpha_black", "alpha_white")} if levels_done else dict(fin))
    hyb_stats: Dict[str, Any] = {}
    if hybrid and hybrid_base is not None:
        n_vm = []
        for f in fl:
            ma = hybrid_base.get(f)
            if ma is None:
                continue
            reg = regions[f]
            if levels_done:      # the same levels as the ViTMatte stack got, so the two meet on one scale
                ma = np.where(reg, levels_alpha(ma, fin.get("alpha_black"), fin.get("alpha_white")), ma)
            a, wmap = hybrid_blend(alphas[f], ma, **hyb)
            a[~reg] = 0.0
            a[reg & force[f]] = 1.0
            alphas[f] = a
            n_vm.append(int(((wmap > 0.5) & reg).sum()))
        hyb_stats = {"hybrid": hybrid, "hybrid_params": hyb, "hybrid_frames": len(n_vm),
                     "hybrid_vm_px_mean": round(float(np.mean(n_vm)), 1) if n_vm else 0.0}
    # alpha_black_after (user 2026-10-15, CLIP2_A F18): the temporal mean re-creates low values (haze) from neighbours
    # whose strands were clipped elsewhere -> one more black-only levels AFTER the smoothing / hybrid, before the blur
    pblack = float(fin_after.pop("alpha_black_after", 0.0) or 0.0) if fin_after else 0.0
    if pblack > 0.0:
        for f in fl:
            a = alphas[f]
            alphas[f] = np.where(regions[f] & ~force[f], levels_alpha(a, pblack, None), a).astype(np.float32)
    if fin_after:
        for f in fl:
            alphas[f] = finish_zone_alpha(alphas[f], regions[f], force[f], fin_after)
    flags: Dict[int, List[str]] = {}
    if not z.holdout:
        flags = {f: ["strefa bez holdoutu obiektów z przodu"] for f in alphas}
    stats = {**st, "frames": len(alphas), "trimap_source": tsrc_used, "force_core_erode_px": force_px,
             "region": str(params.get("region", "draw")),
             "temporal": temporal, "temporal_radius": t_radius,
             "temporal_scale": t_scale,
             "temporal_seconds": round(time.time() - t0, 1), "finish": fin,
             "cached": len(alphas) - len(todo), "computed": len(todo), "bbox": [x0, y0, x1, y1]}
    stats["phases"] = {**(st.get("phases") or {}), **{k: round(v, 2) for k, v in seq_ph.items()}}
    if t_conf:
        stats["temporal_conf"] = t_conf
    stats.update(hyb_stats)
    return alphas, flags, stats


def _matanyone_sequence(z: Zone, frames: Sequence[int], core: Dict[int, np.ndarray], images: Callable[[int], np.ndarray],
                        regions: Dict[int, np.ndarray], bgm: Callable[[int], np.ndarray],
                        holdout: Optional[Callable[[int], Optional[np.ndarray]]],
                        progress: Optional[Callable[[int, int], None]],
                        clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]],
                        ma_run: Optional[Callable[..., Dict[int, np.ndarray]]], cache: Any, psig: Callable[[int], Any],
                        zsig: str) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], Dict[str, Any]]:
    """`matting=matanyone` inside _matte_zone (also the base of the vitmatte hybrid): one MatAnyone2 pass on the zone
    crop (matanyone_zone), whole-sequence cache."""
    from engine.localmatte import local_matte
    from engine.zonecache import sequence_hash
    shape = core[frames[0]].shape
    if ma_run is None:
        raise ValueError("silnik „matanyone” strefy wymaga ComfyUI (Finalizer) – niedostępny w tym kontekście")
    start = int(z.frame) if z.frame in frames else frames[0]
    hseq = sequence_hash(zsig, [core[f] for f in frames], [psig(f) for f in frames], [regions[f] for f in frames],
                         extra=[int(bgm(f).sum()) for f in frames])
    if cache is not None and all(cache.has(hseq, f) for f in frames):
        alphas = {f: cache.get(hseq, f, shape) for f in frames}
        if all(a is not None for a in alphas.values()):
            if progress is not None:
                progress(len(frames), len(frames))
            flags = {f: ["strefa bez holdoutu obiektów z przodu"] for f in alphas} if not z.holdout else {}
            return alphas, flags, {"frames": len(alphas), "engine": "matanyone", "preset": z.preset, "cleanplate": clean is not None,
                                   "cached": len(alphas), "computed": 0, "hash": hseq}
    init = None
    if regions[start].any():
        bg0 = bgm(start)
        hold0 = holdout(start) if (holdout is not None and z.holdout) else None
        if (z.params or {}).get("hint") == "vitmatte":
            # experiment 2026-10-13 (plan step 6 preview): the key hint for MatAnyone = ViTMatte alpha at the key
            # frame (strands > 0.5 become part of the key; MatAnyone then carries them consistently in time)
            from engine.vitmatte import get_runner
            vm, _, _ = vitmatte_zone(z, frames, core, regions, images, get_runner(), holdout, bgm, only=[start],
                                     fg_erode_px=int((z.params or {}).get("fg_erode_px", 2)))
            init = vm.get(start)
        else:
            cp = clean(start) if clean is not None else None
            r0 = local_matte(images(start), core[start], regions[start], bg_mask=bg0, holdout=hold0,
                             engine="clean" if cp else "chroma", preset=z.preset, params=z.params or None,
                             hints=zone_hints(z), clean=cp[0] if cp else None, clean_conf=cp[1] if cp else None)
            init = r0.alpha
    if progress is not None:
        progress(0, len(frames))
    alphas, st = matanyone_zone(z, frames, core, regions, init, ma_run, holdout, bgm)
    if cache is not None:   # the cache (16-bit) is the single source of the value: a cached pass is bit-identical
        for f, a in list(alphas.items()):
            cache.put(hseq, f, a, regions[f])
            back = cache.get(hseq, f, shape)
            if back is not None:
                alphas[f] = back
    if progress is not None:
        progress(len(frames), len(frames))
    flags: Dict[int, List[str]] = {}
    if not z.holdout:
        flags = {f: ["strefa bez holdoutu obiektów z przodu"] for f in alphas}
    return alphas, flags, {**st, "cleanplate": clean is not None, "cached": 0, "computed": len(alphas), "hash": hseq}


def _matte_zone(z: Zone, frames: Sequence[int], core: Dict[int, np.ndarray], images: Callable[[int], np.ndarray],
                regions: Dict[int, np.ndarray], transforms: Dict[int, np.ndarray], removed: Dict[int, np.ndarray],
                holdout: Optional[Callable[[int], Optional[np.ndarray]]],
                progress: Optional[Callable[[int, int], None]] = None,
                clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]] = None,
                ma_run: Optional[Callable[..., Dict[int, np.ndarray]]] = None,
                bg_masks: Optional[Dict[int, np.ndarray]] = None, cache: Any = None,
                plate_sig: Optional[Callable[[int], Any]] = None, probs: Optional[Dict[int, np.ndarray]] = None,
                base_alpha: Optional[Callable[[int], np.ndarray]] = None,
                fg_extra: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                progress_smooth: Optional[Callable[[int, int], None]] = None
                ) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], Dict[str, Any]]:
    """Per-frame local matting on the propagated region (step 3 mode); one temporal stabilization on the stacked
    alphas inside the union bbox of the regions (memory: crops, not full frames). `matting=matanyone`: one MatAnyone2
    pass on the zone crop instead (matanyone_zone; the hint comes from the clean / chroma key at the zone's key frame)."""
    from engine.localmatte import local_matte
    from engine.zonecache import frame_hash, sequence_hash, zone_signature
    shape = core[frames[0]].shape
    _eng_params, _fin = finish_params(z.params)
    zsig = zone_signature({**z.to_dict(), "params": _eng_params})
    bgm = lambda f: ((bg_masks[f] if bg_masks is not None and f in bg_masks
                      else zone_bg_mask_at(z, shape, None if z.bg_static else transforms.get(f))) | removed[f])
    psig = plate_sig if plate_sig is not None else (lambda f: f)
    if z.matting == "vitmatte":
        hyb_base = None
        if str((z.params or {}).get("hybrid", "") or "") == "matanyone":
            # hybrid (experiment 2026-10-15): MatAnyone in the zone with the key from ViTMatte = the stable base
            if ma_run is None:
                raise ValueError("hybryda „matanyone” strefy ViTMatte wymaga ComfyUI (Finalizer) – niedostępna w tym kontekście")
            from dataclasses import replace
            zp = {"hint": "vitmatte", **({"fg_erode_px": z.params["fg_erode_px"]} if "fg_erode_px" in (z.params or {}) else {})}
            z_ma = replace(z, matting="matanyone", params=zp)
            hyb_base, _, _ = _matanyone_sequence(z_ma, frames, core, images, regions, bgm, holdout, None, clean, ma_run,
                                                 cache, psig, zone_signature({**z_ma.to_dict(), "params": zp}))
        alphas, flags, st = _vitmatte_sequence(z, frames, core, images, regions, bgm, holdout,
                                               edge_model_runner((z.params or {}).get("vitmatte_model")), progress,
                                               cache, psig, probs=probs, base_alpha=base_alpha, hybrid_base=hyb_base,
                                               fg_extra=fg_extra, progress_smooth=progress_smooth)
        return alphas, flags, {**st, "cleanplate": clean is not None}
    if z.matting == "matanyone":
        return _matanyone_sequence(z, frames, core, images, regions, bgm, holdout, progress, clean, ma_run, cache, psig, zsig)
    alphas: Dict[int, np.ndarray] = {}
    flags: Dict[int, List[str]] = {}
    per_engine: Dict[str, List[float]] = {}
    secs = 0.0
    n_cached = n_computed = 0
    hashes: Dict[int, str] = {}
    for i, f in enumerate(frames):
        if progress is not None:
            progress(i + 1, len(frames))
        reg = regions[f]
        if not reg.any():
            continue
        bg = bgm(f)
        h = frame_hash(zsig, core[f], psig(f), reg, extra=int(bg.sum()))
        hashes[f] = h
        if cache is not None:
            got = cache.get(h, f, shape)
            if got is not None:
                alphas[f] = got
                n_cached += 1
                continue
        hold = holdout(f) if (holdout is not None and z.holdout) else None
        cp = clean(f) if clean is not None else None
        r = local_matte(images(f), core[f], reg, bg_mask=bg, holdout=hold, engine=z.matting, preset=z.preset,
                        params=z.params or None, hints=zone_hints(z),
                        clean=cp[0] if cp else None, clean_conf=cp[1] if cp else None)
        alphas[f] = r.alpha
        n_computed += 1
        if cache is not None:
            cache.put(h, f, r.alpha, reg)       # cached BEFORE the temporal median (per-frame result, stable key)
            back = cache.get(h, f, shape)       # 16-bit cache = the single source of the value (cached pass bit-identical)
            if back is not None:
                alphas[f] = back
        fl = list(r.flags)
        if not z.holdout:
            fl.append("strefa bez holdoutu obiektów z przodu")
        if fl:
            flags[f] = fl
        for k, v in (r.stats.get("engine_share") or {}).items():
            per_engine.setdefault(k, []).append(float(v))
        secs += float(r.stats.get("seconds", 0.0))
    if not alphas:
        return {}, flags, {"frames": 0}
    # one temporal stabilization (plan section 2) on the union bbox of the zone over the shot
    ys, xs = np.where(np.any(np.stack([regions[f] for f in alphas]), axis=0))
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    fl = [f for f in frames if f in alphas]
    if len(fl) >= 3:
        stack = np.stack([alphas[f][y0:y1, x0:x1] for f in fl])
        from engine.refine import temporal_median
        stack = temporal_median(stack, 2, 0.35)
        for i, f in enumerate(fl):
            alphas[f][y0:y1, x0:x1] = stack[i]
            alphas[f][~regions[f]] = 0.0
    if _fin:
        for f in list(alphas):
            alphas[f] = finish_zone_alpha(alphas[f], regions[f], core[f], _fin)
    stats = {"frames": len(alphas), "engine": z.matting, "preset": z.preset, "seconds": round(secs, 1), "finish": _fin,
             "cleanplate": clean is not None, "cached": n_cached, "computed": n_computed,
             "engine_share": {k: round(float(np.mean(v)), 3) for k, v in per_engine.items()},
             "bbox": [x0, y0, x1, y1]}
    return alphas, flags, stats


def stabilize_keys(keys: List[np.ndarray], radius: int = 2, motion_limit: float = 0.35) -> List[np.ndarray]:
    from engine.refine import temporal_median
    return list(temporal_median(np.stack(keys), radius, motion_limit))


def apply_zones(frames: Sequence[int], cores: Dict[int, np.ndarray], images: Callable[[int], np.ndarray],
                zones: Sequence[Zone], transforms: Dict[str, Dict[int, np.ndarray]],
                user_keys: Sequence[int] = (), holdout: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                progress: Optional[Callable[[str, int, int], None]] = None,
                clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]] = None,
                ma_run: Optional[Callable[..., Dict[int, np.ndarray]]] = None,
                regions_by_zone: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
                bg_by_zone: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
                cache_for: Optional[Callable[[str], Any]] = None,
                plate_sig: Optional[Callable[[int], Any]] = None,
                probs: Optional[Dict[int, np.ndarray]] = None,
                base_alpha: Optional[Callable[[int], np.ndarray]] = None,
                fg_by_zone: Optional[Dict[str, Callable[[int], Optional[np.ndarray]]]] = None) -> ZoneResult:
    """Zones in creation order on the core (step 2 of the pipeline). `holdout(f)` = full-frame mask of the objects in
    front (None = none) for `matte` zones; `progress(label, done, total)` for the task bar. `probs`: SAM 3.1 probability
    maps consistent with `cores` (ViTMatte zones: trimap from SAM 3.1's uncertainty); frames whose core an earlier zone
    changed fall back to the binary trimap."""
    frames = list(frames)
    core = {f: cores[f].copy() for f in frames}
    removed = {f: np.zeros_like(cores[f]) for f in frames}
    soft: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    result = ZoneResult(core=core, removed=removed, soft=soft)
    feathers: Dict[int, List[Tuple[np.ndarray, float, float, bool]]] = {}   # frame -> (area, inset / outer feather, outline)
    for z in [z for z in zones if z.enabled]:
        shape = cores[frames[0]].shape
        if regions_by_zone is not None and z.zone_id in regions_by_zone:
            regions = regions_by_zone[z.zone_id]
        else:
            regions = {f: zone_mask_at(z, shape, transforms.get(z.zone_id, {}).get(f)) for f in frames}
            if z.criterion == "matte" and region_mode(z) == "outline":   # drawing = exclusion (objects: pipeline only)
                regions = outline_regions(z, {f: core[f] for f in frames}, regions)
        if z.criterion == "matte":
            if z.mode != "add":
                raise ValueError(f"strefa {z.zone_id}: tryb Usuń dla kryterium „matte” jeszcze nieobsługiwany")
            cb = (lambda d, t: progress(f"Strefa {z.zone_id} ({z.preset}): matting", d, t)) if progress else None
            cb2 = (lambda d, t: progress(f"Strefa {z.zone_id} ({z.preset}): wygładzanie czasowe", d, t)) if progress else None
            tz = time.perf_counter()
            alphas, zflags, zstats = _matte_zone(z, frames, core, images, regions, transforms.get(z.zone_id, {}),
                                                 removed, holdout, cb, clean, ma_run,
                                                 bg_masks=(bg_by_zone or {}).get(z.zone_id),
                                                 cache=cache_for(z.zone_id) if cache_for else None, plate_sig=plate_sig,
                                                 probs=({f: p for f, p in probs.items() if f in core and np.array_equal(core[f], cores[f])}
                                                        if probs else None),
                                                 base_alpha=base_alpha, fg_extra=(fg_by_zone or {}).get(z.zone_id),
                                                 progress_smooth=cb2)
            fo = zone_outer_feather_px(z) if region_mode(z) == "outline" else 0.0
            for f, a in alphas.items():       # never skipped in user-key frames (plan section 12)
                area = regions[f]
                prev = soft.get(f)
                merged = np.where(area, a, prev[0] if prev else 0.0).astype(np.float32)
                soft[f] = (merged, area | (prev[1] if prev else np.zeros_like(area)))
                feathers.setdefault(f, []).append((area, zone_feather_at(z, f), fo, region_mode(z) == "outline", zone_whole(z)))
            for f, fl in zflags.items():
                result.flags.setdefault(f, []).extend(fl)
            if isinstance(zstats.get("phases"), dict):     # plan A6b: the whole zone (model loop, smoothing, finish, merge)
                zstats["phases"]["zone_total"] = round(time.perf_counter() - tz, 2)
            result.stats[z.zone_id] = zstats
            fz = zone_feather_px(z)
            result.feather_px = fz if result.feather_px is None else max(result.feather_px, fz)
            continue
        # stage-2 brush correction (params.source = brush, 2026-10-15): the painted region itself (criterion "all",
        # already coherent in time – propagated by CoTracker3 from the zone keys), applied in EVERY frame: it corrects
        # the SAM3 result after stage 1, so a SAM3 user key in the frame does not win over it
        brush = (z.params or {}).get("source") == "brush"
        keys = ([regions[f].astype(np.float32) for f in frames] if brush else
                stabilize_keys([criterion_key(images(f), z) * regions[f] for f in frames]))
        for i, f in enumerate(frames):
            if f in user_keys and not brush:
                continue  # a user key has priority in its frame
            k, reg = keys[i], regions[f]
            if z.soft and z.criterion != "all":
                alpha = k if z.mode == "add" else (1.0 - k)
                area = reg & ((k > 0.02) if z.mode == "add" else core[f])
                if z.mode == "add":
                    area = area & connected_to(k > 0.02, core[f], z.grow_px)
                prev = soft.get(f)
                a = np.where(area, alpha, prev[0] if prev else 0.0).astype(np.float32)
                soft[f] = (a, area | (prev[1] if prev else np.zeros_like(area)))
                continue
            hit = (k > 0.5) & reg
            if z.mode == "add":
                core[f] = core[f] | connected_to(hit, core[f], z.grow_px)
            else:
                rm = core[f] & hit
                core[f] = core[f] & ~hit
                removed[f] |= cv2.dilate(rm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & reg
    # frames with an outer-edge feather: one weight map for the whole soft area (each zone its own feathers)
    t_base = t_comp = 0.0
    tc0 = time.perf_counter()
    for f, lst in feathers.items():
        if not any(fo > 0 or ol for _a, _fi, fo, ol, _wh in lst):
            continue
        w = np.zeros(soft[f][1].shape, np.float32)
        sa = soft[f][0]
        for area, fi, fo, ol, wh in lst:
            if ol:
                # whole outline (user rule 2026-10-22): both feathers blend the zone's matte into max(core, base) –
                # white where the core is (the inner side, and an outer edge that lies on the core), the stage base
                # where it is not (user decision 2026-10-22, later that day: the outer edge stays continuous with a soft
                # base such as MatAnyone2 / VideoMaMa – with the mask 1+2 as base this is black, as before); the area
                # then wins outright; the feathers = the one band system (engine/bandsys.compose_area, 2026-10-24)
                from engine.bandsys import compose_area_crop
                partner = cores[f].astype(np.float32)
                tb = time.perf_counter()
                if base_alpha is not None:
                    partner = np.maximum(partner, np.clip(base_alpha(f), 0.0, 1.0).astype(np.float32))
                tc = time.perf_counter()
                t_base += tc - tb
                # on the band's bbox only (bit-identical to compose_area, A6b 2026-10-28)
                sa = compose_area_crop(cores[f], sa, area, fi, fo, outside=sa, partner=partner, whole=wh)   # fo literal (2026-10-24)
                t_comp += time.perf_counter() - tc
                w = np.where(area, 1.0, w)
                continue
            from engine.bandsys import area_weight
            zw = (area_weight(area, cores[f], fi, fo) if fo > 0 else
                  np.clip(cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5) / max(0.5, fi), 0, 1))
            w = np.where(area, zw, w)
        soft[f] = (sa, soft[f][1])
        result.weights[f] = (np.clip(w, 0, 1) * 255).round().astype(np.uint8)
    result.timings = {"feather_total": time.perf_counter() - tc0, "feather_read_base": t_base,
                      "feather_compose_area": t_comp}
    return result
