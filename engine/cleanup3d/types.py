# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – the shared contract (docs/PLAN_CLEANUP_3D.md, package P0, user decisions 2026-11-09).

One place for what every Cleanup 3D package agrees on: the data types, where they live on disk and the interface of the
geometry (P1) that the key store (P2), the regeneration (P5) and the render (P6) call. No heavy imports here (numpy only):
the API skeleton and the tests of every package import this module.

Layout, next to `<work>/matte_desk/` (never the shot manifest):
  <work>/inpaint/camera3d/cams.json          Camera3D          schema matte_desk/camera3d@1.0
  <work>/inpaint/cards.json                  [Card]            schema matte_desk/cards3d@1.0
  <work>/inpaint/keys/<card>/k_<frame>/      key.json (Key, schema matte_desk/card_key@1.0), v001.png ... (versions),
                                             areas.json ([Area], schema matte_desk/key_areas@1.0), a_<id>.png (brush layers)
  <work>/inpaint/v###/                       results, as the Cleanup MVP writes them (engine/cleanup.py)
Coordinates: plate pixels, origin top-left, y down, pixel centre at integer coordinates (as DATA_CONTRACT). Camera:
OpenCV convention – x right, y down, z forward; X_cam = R @ X_world + t; distortion (k1, k2, p1, p2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

SCHEMA_CAMERA = "matte_desk/camera3d@1.0"
SCHEMA_CARDS = "matte_desk/cards3d@1.0"
SCHEMA_KEY = "matte_desk/card_key@1.0"
SCHEMA_AREAS = "matte_desk/key_areas@1.0"

# the steps of the Cleanup mode (the stage bar of Cleanup, static/cleanup3d.js): id -> order
STEPS: Tuple[str, ...] = ("remove", "camera", "cards", "keys", "project", "export")
# engines that regenerate a marked area of a key (P5); "void" comes with stage III
# (user 2026-11-10: Qwen-Image-2.1 edit = the recommended engine for large areas, LaMa for small ones; FLUX 3 Image
# Klein + LoRA Restore stays but does not fill holes – it draws shapes into a neutral hole)
ENGINES: Tuple[str, ...] = ("lama", "qwen21_edit", "klein_restore")
ENGINE_LABEL = {"lama": "LaMa", "qwen21_edit": "Qwen-Image-2.1", "klein_restore": "FLUX 3 Image Klein + LoRA Restore"}
# how a card gets its key frames: manual / every (keys.py), auto = where the projection error grows, one = ONE master
# key accumulated from all frames (keymodes.py, package P13)
KEY_MODES: Tuple[str, ...] = ("manual", "every", "auto", "one")
CARD_KINDS: Tuple[str, ...] = ("plane", "sky", "depth") # sky = at infinity (rotation only); depth = surface of a depth map (P18)
DEPTH_SOURCES: Tuple[str, ...] = ("computed", "plate")  # depth map of the tool (DepthCrafter, outputs.json) / the shot's depth plate
DEPTH_CONVS: Tuple[str, ...] = ("disparity", "z")       # disparity: 1/z = scale d + offset (nearer = larger); z: z = scale d + offset
VIEWS: Tuple[str, ...] = ("key", "proj", "cards")       # previews of the viewer besides the plates
KEY_FORMATS: Tuple[str, ...] = ("png", "png16", "exr")  # export / import of a key
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")         # same rule as object names (engine/objects.py)


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"Niepoprawna nazwa: {name!r}")
    return name


# ----------------------------------------------------------------------------- paths
def root_dir(work_dir: Path) -> Path:
    """`<work>/inpaint/` – the Cleanup folder (engine.cleanup.cleanup_dir), next to the tool's matte_desk/."""
    return Path(work_dir).parent / "inpaint"


def camera_file(work_dir: Path) -> Path:
    return root_dir(work_dir) / "camera3d" / "cams.json"


def cards_file(work_dir: Path) -> Path:
    return root_dir(work_dir) / "cards.json"


def key_dir(work_dir: Path, card: str, frame: int) -> Path:
    return root_dir(work_dir) / "keys" / check_name(card) / f"k_{int(frame):04d}"


def key_file(work_dir: Path, card: str, frame: int) -> Path:
    return key_dir(work_dir, card, frame) / "key.json"


def areas_file(work_dir: Path, card: str, frame: int) -> Path:
    return key_dir(work_dir, card, frame) / "areas.json"


def version_name(n: int) -> str:
    return f"v{int(n):03d}"


# ----------------------------------------------------------------------------- camera
@dataclass
class Camera3D:
    """A solved camera of the shot: one set of intrinsics, a pose per plate frame."""
    width: int
    height: int
    K: np.ndarray                                   # 3x3
    dist: np.ndarray                                # (k1, k2, p1, p2)
    R: Dict[int, np.ndarray] = field(default_factory=dict)   # plate frame -> 3x3
    t: Dict[int, np.ndarray] = field(default_factory=dict)   # plate frame -> (3,)
    source: str = ""                                # how it was made ("colmap+cotracker3", "experiment sfm8", ...)
    report: Dict[str, Any] = field(default_factory=dict)     # reprojection px, key error, horizon, worst frames

    @property
    def frames(self) -> List[int]:
        return sorted(self.R)

    def centre(self, frame: int) -> np.ndarray:
        return -self.R[frame].T @ self.t[frame]

    def to_json(self) -> Dict[str, Any]:
        return {"schema": SCHEMA_CAMERA, "width": int(self.width), "height": int(self.height),
                "K": np.asarray(self.K, float).tolist(), "dist": np.asarray(self.dist, float).tolist(),
                "frames": [{"frame": int(f), "R": np.asarray(self.R[f], float).tolist(),
                            "t": np.asarray(self.t[f], float).tolist()} for f in self.frames],
                "source": self.source, "report": self.report}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Camera3D":
        K = np.asarray(d["K"], np.float64)
        dist = np.asarray(d.get("dist") or [0, 0, 0, 0], np.float64)
        if K.shape != (3, 3) or dist.shape != (4,):
            raise ValueError("camera3d: K musi być 3x3, dist = (k1, k2, p1, p2)")
        R, t = {}, {}
        for fr in d.get("frames") or []:
            f = int(fr["frame"])
            R[f], t[f] = np.asarray(fr["R"], np.float64), np.asarray(fr["t"], np.float64)
            if R[f].shape != (3, 3) or t[f].shape != (3,):
                raise ValueError(f"camera3d: zła poza klatki {f}")
        return cls(int(d["width"]), int(d["height"]), K, dist, R, t, str(d.get("source") or ""), dict(d.get("report") or {}))


# ----------------------------------------------------------------------------- areas (a card's extent, a key's regeneration areas)
@dataclass
class Area:
    """One layer of a marked region: a closed Bezier curve (engine/curves.py JSON) or a brush layer (PNG next to the json).
    `op`: add | remove (layers are applied in order). Pixels of the KEY (or of the card's reference key for an extent)."""
    id: str
    kind: str                                        # "curve" | "brush"
    op: str = "add"
    curve: Optional[Dict[str, Any]] = None           # {type: "bezier", closed: true, pts: [...]}
    png: Optional[str] = None                        # file name of the brush layer (a_<id>.png), 8-bit mask
    label: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = {"id": self.id, "kind": self.kind, "op": self.op, "label": self.label}
        if self.kind == "curve":
            d["curve"] = self.curve
        else:
            d["png"] = self.png
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Area":
        kind, op = d.get("kind"), d.get("op", "add")
        if kind not in ("curve", "brush") or op not in ("add", "remove"):
            raise ValueError(f"area: kind / op niepoprawne ({kind!r}, {op!r})")
        if kind == "curve" and not (isinstance(d.get("curve"), dict) and d["curve"].get("type") == "bezier"):
            raise ValueError("area: krzywa musi być w formacie engine/curves.py (type: bezier)")
        if kind == "brush" and not d.get("png"):
            raise ValueError("area: warstwa pędzla bez pliku png")
        return cls(check_name(str(d["id"])), kind, op, d.get("curve"), d.get("png"), str(d.get("label") or ""))


# ----------------------------------------------------------------------------- cards
@dataclass
class Card:
    """A card = a plane of the world (or the sky at infinity) + its EXTENT + its keys' mode (user decisions 2026-11-09:
    keys per card; the projection and the regeneration of a card never enter a neighbouring card).
    plane: n . X + d = 0 with n toward the cameras (None for the sky). adjust: the user's sliders on top of the solved
    plane (dh along the normal, pitch / roll in degrees) – as removers_test/cards.py. extent: areas drawn ONCE in the
    key frame `extent_frame`; they lie on the card, the camera carries them to every frame (empty = the whole card)."""
    id: str
    name: str
    kind: str = "plane"
    plane: Optional[Dict[str, Any]] = None           # {"n": [x, y, z], "d": float}
    adjust: Dict[str, float] = field(default_factory=lambda: {"dh": 0.0, "pitch": 0.0, "roll": 0.0})
    extent: List[Area] = field(default_factory=list)
    extent_frame: Optional[int] = None
    seam_px: float = 16.0                            # soft transition to the neighbouring cards
    order: int = 0                                   # 0 = in front (as the object order)
    color: str = "#4ae0ff"                           # key squares on the timeline, the card's grid
    key_mode: str = "manual"
    key_every: int = 8                               # frames between keys for key_mode "every"
    warps: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)   # key frame -> edge warps (curves src / dst)
    # key_mode "auto": threshold / min_gap / max_gap / max_keys of engine/cleanup3d/keymodes.py (missing = defaults)
    key_auto: Dict[str, float] = field(default_factory=dict)
    # P16 (2026-11-10): a card turned off for good (the timeline eye) – left out of the render, the previews and the Nuke
    # export
    off: bool = False
    # P16 solo (user 2026-11-10, like After Effects): when any card is soloed only the soloed cards (that are not off)
    # go into the render, the previews and the Nuke export (`active_cards`). Stored in the JSON only when set, so a card
    # without solo keeps its JSON (render_sig) and turning solo back off gives the old signature
    solo: bool = False
    # kind "depth" (P18): {"frame": reference plate frame, "source": computed | plate, "conv": disparity | z,
    # "scale", "offset" (relative -> metric, fitted to the camera), "mul", "shift" (the user's sliders: z = mul z0 + shift),
    # "file": snapshot .npy of the reference frame's depth, relative to <work>/inpaint/ (or absolute), "stamp",
    # "edge" (relative depth jump that splits the surface), "fit": report of the fit}. `plane` = the approximate plane
    # of the surface (grid of the previews, Nuke export). None for the other kinds.
    depth: Optional[Dict[str, Any]] = None
    # not stored: the `<work>/inpaint/` folder the card was read from (store.load_cards) – resolves depth["file"]
    root: Optional[str] = field(default=None, compare=False, repr=False)

    def to_json(self) -> Dict[str, Any]:
        d = {"id": self.id, "name": self.name, "kind": self.kind, "plane": self.plane, "adjust": dict(self.adjust),
             "extent": [a.to_json() for a in self.extent], "extent_frame": self.extent_frame,
             "seam_px": float(self.seam_px), "order": int(self.order), "color": self.color,
             "key_mode": self.key_mode, "key_every": int(self.key_every), "warps": self.warps,
             "key_auto": dict(self.key_auto), "off": bool(self.off)}
        if self.kind == "depth" and self.depth is not None:      # only then: other cards keep their JSON (render_sig)
            d["depth"] = dict(self.depth)
        if self.solo:                                            # only then (render_sig: solo off = the old JSON)
            d["solo"] = True
        return d

    def depth_file(self) -> Optional[Path]:
        """The snapshot of the reference depth (absolute), None when the card has none yet."""
        f = (self.depth or {}).get("file")
        if not f:
            return None
        p = Path(f)
        return p if p.is_absolute() or not self.root else Path(self.root) / p

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Card":
        kind, mode = d.get("kind", "plane"), d.get("key_mode", "manual")
        if kind not in CARD_KINDS or mode not in KEY_MODES:
            raise ValueError(f"card: kind / key_mode niepoprawne ({kind!r}, {mode!r})")
        plane = d.get("plane")
        depth = None
        if kind == "plane":
            if not (isinstance(plane, dict) and len(plane.get("n") or []) == 3 and "d" in plane):
                raise ValueError("card: karta-płaszczyzna wymaga plane {n: [x, y, z], d}")
        elif kind == "depth":
            depth = parse_depth(d.get("depth"))
            if not (isinstance(plane, dict) and len(plane.get("n") or []) == 3 and "d" in plane):
                plane = None                          # the approximate plane is optional (computed from the depth)
        else:
            plane = None
        adj = {"dh": 0.0, "pitch": 0.0, "roll": 0.0, **{k: float(v) for k, v in (d.get("adjust") or {}).items()}}
        card = cls(check_name(str(d["id"])), str(d.get("name") or d["id"]), kind, plane, adj,
                   [Area.from_json(a) for a in d.get("extent") or []],
                   None if d.get("extent_frame") is None else int(d["extent_frame"]),
                   float(d.get("seam_px", 16.0)), int(d.get("order", 0)), str(d.get("color") or "#4ae0ff"),
                   mode, max(1, int(d.get("key_every", 8))), dict(d.get("warps") or {}),
                   {str(k): float(v) for k, v in (d.get("key_auto") or {}).items()
                    if k in ("threshold", "min_gap", "max_gap", "max_keys") and v is not None}, depth=depth)
        card.off = bool(d.get("off", False))
        card.solo = bool(d.get("solo", False))
        return card


def active_cards(cards: List["Card"]) -> List["Card"]:
    """P16: the cards that go into the render, the previews, the timeline coverage and the Nuke export – like After
    Effects: a card turned off (eye) never; when any card is soloed, only the soloed ones."""
    any_solo = any(c.solo for c in cards)
    return [c for c in cards if not c.off and (not any_solo or c.solo)]


DEPTH_DEFAULTS: Dict[str, Any] = {"source": "computed", "conv": "disparity", "scale": 1.0, "offset": 0.0, "mul": 1.0,
                                  "shift": 0.0, "edge": 0.04, "file": None, "stamp": ""}


def parse_depth(raw: Any) -> Dict[str, Any]:
    """The `depth` dict of a depth card, checked and completed with DEPTH_DEFAULTS (ValueError in Polish)."""
    if not isinstance(raw, dict) or raw.get("frame") is None:
        raise ValueError("card: karta głębi wymaga depth {frame: klatka odniesienia, ...}")
    out = {**DEPTH_DEFAULTS, **{k: v for k, v in raw.items() if v is not None or k == "file"}}
    out["frame"] = int(raw["frame"])
    if out["source"] not in DEPTH_SOURCES or out["conv"] not in DEPTH_CONVS:
        raise ValueError(f"card: depth.source / depth.conv niepoprawne ({out['source']!r}, {out['conv']!r})")
    for k in ("scale", "offset", "mul", "shift", "edge"):
        out[k] = float(out[k])
    if out["mul"] <= 0 or out["edge"] <= 0:
        raise ValueError("card: skala głębi i próg krawędzi muszą być dodatnie")
    if out["file"] is not None:
        out["file"] = str(out["file"])
    out["stamp"] = str(out.get("stamp") or "")
    if out.get("fit") is not None and not isinstance(out["fit"], dict):
        raise ValueError("card: depth.fit musi być obiektem")
    return out


def cards_to_json(cards: Sequence[Card]) -> Dict[str, Any]:
    return {"schema": SCHEMA_CARDS, "cards": [c.to_json() for c in cards]}


def cards_from_json(d: Dict[str, Any]) -> List[Card]:
    cards = [Card.from_json(c) for c in d.get("cards") or []]
    ids = [c.id for c in cards]
    if len(set(ids)) != len(ids):
        raise ValueError("cards: powtórzone id karty")
    return sorted(cards, key=lambda c: c.order)


# ----------------------------------------------------------------------------- keys
@dataclass
class Key:
    """A key frame of a card: the image projected onto the card through the camera. Versions v001... (the active one is
    projected; older ones stay for undo). canvas: the key's own pinhole view when it is larger than the plate frame
    (stage III „one key”): {"w", "h", "K": 3x3, "frame": pose of this plate frame}; None = the plate frame as shot."""
    card: str
    frame: int
    versions: List[Dict[str, Any]] = field(default_factory=list)   # [{"v": "v001", "ext": "png", "source": "...", "date": "..."}]
    active: Optional[str] = None
    canvas: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {"schema": SCHEMA_KEY, "card": self.card, "frame": int(self.frame), "versions": self.versions,
                "active": self.active, "canvas": self.canvas}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Key":
        vs = list(d.get("versions") or [])
        active = d.get("active")
        if active is not None and active not in [v.get("v") for v in vs]:
            raise ValueError(f"key: aktywna wersja {active!r} nie istnieje")
        return cls(check_name(str(d["card"])), int(d["frame"]), vs, active, d.get("canvas"))


# ----------------------------------------------------------------------------- geometry interface (implemented by P1: engine/cleanup3d/geom.py)
class Geometry(Protocol):
    """What the other packages may ask the geometry. Arrays are float32; maps are (h, w) for cv2.remap-style sampling:
    map_x / map_y give, for every pixel of the TARGET image, where to read in the SOURCE image (< 0 = no data)."""

    def frame_to_key(self, frame: int, key: Key, card: Card) -> Tuple[np.ndarray, np.ndarray]:
        """Sampling maps to draw key `key` in plate frame `frame` through `card` (target = the frame, source = the key)."""

    def key_to_frame(self, key: Key, card: Card, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sampling maps to draw plate frame `frame` in the key's view (target = the key, source = the frame)."""

    def points_frame_to_key(self, pts: np.ndarray, frame: int, key: Key, card: Card) -> np.ndarray:
        """(n, 2) plate pixels of `frame` -> (n, 2) pixels of the key (NaN where the ray misses the card)."""

    def card_weight(self, card: Card, frame: int) -> np.ndarray:
        """(h, w) 0..1: where plate frame `frame` shows `card` (its plane hit first among the cards in front, inside its
        extent, soft over `seam_px` at the border to the neighbouring cards)."""

    def coverage(self, frame: int, key: Key, card: Card, edge_px: float = 120.0) -> np.ndarray:
        """(h, w) 0..1: how far key `key` reaches in `frame` (fades to 0 over `edge_px` at the key's own border)."""
