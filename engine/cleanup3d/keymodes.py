# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – key modes of a card (docs/PLAN_CLEANUP_3D.md, package P13; user decisions 2026-11-09).

Every card has its own key frames and a key mode (types.KEY_MODES): "manual" (frames picked by hand) and "every"
(keys.frames_every) live in keys.py; this module adds the two modes of stage III:

- "auto"  – key frames placed where the PROJECTION ERROR grows: for a frame between two keys, the plates of those key
  frames are projected onto it through the card (render.project_frame – the same crossfade as the render, the key
  frames' removal masks cut out like kept objects) and compared with the frame's own plate outside the objects to
  remove; a pixel of the card the keys do not reach counts as UNCOVERED_ERR. A new key goes where the error is largest,
  above `threshold`, at least `min_gap` frames from the other keys; no gap is longer than `max_gap`. Iterative, on a
  small copy of the shot (long side AUTO_SIDE) – the error says WHERE a key is needed, not how it looks.
- "one"   – ONE master key per card, an accumulation („klucz-suma”, removers_test/master_accum.py): a canvas in the view
  of the key frame (same focal, no distortion, larger than the frame – at most OVER of the frame past each side) filled
  with REAL plate pixels of all frames projected through the card: each frame contributes only pixels outside the
  removal masks (grown by `mask_grow_px`, soft over `edge_px`), the frame nearest to the key frame wins (front-to-back
  accumulation), each frame's colour is normalised to what is already there (one gain per channel on the overlap).
  The key frame itself goes first, so wherever it sees the background the key IS its plate (the plate rule of
  keys.rebase_on_plate). What no frame sees stays a hole: filled with the surrounding colour for now and marked as a
  regeneration area of the key (`UNSEEN_AREA`), for LaMa / FLUX 3 Image Klein + LoRA Restore.

Geometry only through engine/cleanup3d/geom.CardGeometry (device "cpu" here); images float32 BGR, plate pixels.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine import cleanup as cu
from engine.cleanup3d import keys as KS
from engine.cleanup3d import render as RD
from engine.cleanup3d import store
from engine.cleanup3d import types as T
from engine.cleanup3d.geom import CardGeometry

Progress = Optional[Callable[[str, int, int], None]]

# ----------------------------------------------------------------------------- auto
AUTO_DEFAULTS: Dict[str, float] = {"threshold": 0.03, "min_gap": 4, "max_gap": 48, "max_keys": 64}
AUTO_LIMITS: Dict[str, Tuple[float, float]] = {"threshold": (0.002, 0.5), "min_gap": (1, 500), "max_gap": (0, 5000),
                                               "max_keys": (2, 1000)}
AUTO_SIDE = 480              # the error is measured on frames of at most this long side
AUTO_MASK_GROW = 16.0        # px (full frame) the removal masks are grown by before the error / the key frames' cut
UNCOVERED_ERR = 0.25         # error of a pixel of the card the neighbouring keys do not reach (0..1 per channel)
MIN_CARD_SHARE = 0.002       # a frame showing less of the card than this share has no error
AUTO_LABEL = "Klucze automatycznie: błąd rzutu"

# ----------------------------------------------------------------------------- one (master key)
OVER = 0.25                  # canvas at most this much of the frame past each side of the key frame (master_accum OVER)
MASTER_MASK_GROW = 16.0      # removed object's edge (motion blur, halo) never sampled (master_accum MASK_GROW)
MASTER_EDGE_PX = 24.0        # soft edge of a frame's contribution (master_accum FE)
MASTER_GRID_PX = 16          # the canvas extent is found on every 16th pixel of each frame
MASTER_FULL = 0.98           # accumulated weight from which a canvas pixel counts as seen
MASTER_HOLE = 0.5            # ... below which a pixel some frame shows counts as never seen clean (regeneration area);
                             # between the two the real pixels are blended with the surrounding colour
MASTER_SOURCE = "master"
MASTER_LABEL = "Klucz-suma"
UNSEEN_AREA = "unseen"       # id of the regeneration area of what no frame sees
UNSEEN_LABEL = "niewidoczne w żadnej klatce"


def auto_settings(d: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """AUTO_DEFAULTS + the given values, clamped to AUTO_LIMITS (gaps / max_keys as ints)."""
    s = dict(AUTO_DEFAULTS)
    for k, v in (d or {}).items():
        if k in s and v is not None:
            lo, hi = AUTO_LIMITS[k]
            s[k] = min(hi, max(lo, float(v)))
    for k in ("min_gap", "max_gap", "max_keys"):
        s[k] = int(round(s[k]))
    return s


def _small_camera(cam: T.Camera3D, side: int) -> T.Camera3D:
    """The camera of a copy of the shot scaled so its long side is at most `side` (pixel centres at integers)."""
    s = min(1.0, float(side) / float(max(cam.width, cam.height)))
    w2, h2 = max(8, int(round(cam.width * s))), max(8, int(round(cam.height * s)))
    sx, sy = w2 / float(cam.width), h2 / float(cam.height)
    K = np.asarray(cam.K, np.float64).copy()
    K2 = K.copy()
    K2[0, 0], K2[0, 1], K2[0, 2] = K[0, 0] * sx, K[0, 1] * sx, (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 1], K2[1, 2] = K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5
    return T.Camera3D(w2, h2, K2, np.asarray(cam.dist, np.float64), dict(cam.R), dict(cam.t), cam.source, {})


def _bare(card: T.Card) -> T.Card:
    """The card's plane alone (no extent, no edge warps): the small geometry maps, the weights come from the full one."""
    return dataclasses.replace(card, extent=[], extent_frame=None, warps={}, order=0)


def _mask01(m: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if m is None:
        return np.zeros(shape, np.float32)
    a = np.asarray(m)
    a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 65535.0 if a.dtype == np.uint16 else 1.0)
    if a.ndim == 3:
        a = a.max(axis=2)
    return a


def _grow(m: np.ndarray, px: float) -> np.ndarray:
    g = int(round(px))
    b = (m > 0.02).astype(np.uint8)
    if g > 0 and b.any():
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    return b > 0


def _gaps(keys: List[int], frames: List[int], max_gap: int) -> List[int]:
    """`keys` + evenly spaced frames of `frames` so that no two neighbouring keys are more than `max_gap` apart."""
    if max_gap <= 0:
        return keys
    out = set(keys)
    ks = sorted(keys)
    for a, b in zip(ks, ks[1:]):
        if b - a <= max_gap:
            continue
        n = int(np.ceil((b - a) / float(max_gap)))
        inner = [f for f in frames if a < f < b]
        for i in range(1, n):
            target = a + (b - a) * i / float(n)
            if inner:
                out.add(min(inner, key=lambda f: (abs(f - target), f)))
    return sorted(out)


class _Small:
    """Small copies of the plates, removal masks and card weights of the frames (computed once per frame)."""

    def __init__(self, geometry: CardGeometry, card: T.Card, cam: T.Camera3D, plate: Callable[[int], np.ndarray],
                 remove_mask: Callable[[int], Optional[np.ndarray]], linear: bool, grow_px: float):
        self.g, self.card, self.cam = geometry, card, cam
        self.plate, self.remove_mask, self.linear = plate, remove_mask, linear
        self.scale = cam.width / float(geometry.camera.width)
        self.grow_px = grow_px
        self.d: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(plate in display tone (h, w, 3) float32, removed objects grown (h, w) float32 0 / 1, card weight (h, w))."""
        if f not in self.d:
            sz = (int(self.cam.width), int(self.cam.height))
            full = self.plate(f)
            pl = KS._bgr3(KS._to_float(np.asarray(full)))
            if pl.shape[:2] != (int(self.g.camera.height), int(self.g.camera.width)):
                raise ValueError(f"Klatka {f}: plate {pl.shape[1]}×{pl.shape[0]} nie pasuje do kamery 3D "
                                 f"{self.g.camera.width}×{self.g.camera.height}")
            pl = cu.to_model(cv2.resize(pl, sz, interpolation=cv2.INTER_AREA), self.linear)
            full_hw = (int(self.g.camera.height), int(self.g.camera.width))
            rm = _grow(_mask01(self.remove_mask(f), full_hw), self.grow_px).astype(np.float32)
            rm = (cv2.resize(rm, sz, interpolation=cv2.INTER_AREA) > 0.01).astype(np.float32)
            cw = cv2.resize(np.asarray(self.g.card_weight(self.card, f), np.float32), sz, interpolation=cv2.INTER_AREA)
            self.d[f] = (pl.astype(np.float32), rm, cw)
        return self.d[f]


def frame_error(small: _Small, sgeo: CardGeometry, card: T.Card, frame: int, key_frames: Sequence[int],
                cache: RD.KeyCache, settings: Dict[str, Any]) -> float:
    """Mean error of `frame` when its card is projected from the neighbouring keys among `key_frames` (each key = the
    plate of its frame, its removed objects cut out): |plate − projection| per channel where the keys reach, UNCOVERED_ERR
    where they do not, over the card's pixels outside the frame's removed objects. 0 when the frame hardly shows the card."""
    pl, rm, cw = small.get(frame)
    use = (cw > 0.5) & (rm < 0.5)
    if use.mean() < MIN_CARD_SHARE:
        return 0.0
    keys = [T.Key(card.id, int(k), [{"v": "plate", "ext": "png"}], "plate") for k in key_frames]
    fill, wsum = RD.project_frame(sgeo, [card], {card.id: keys}, int(frame), None,
                                  lambda f: small.get(int(f))[1], settings=settings, cache=cache)
    d = np.abs(fill - pl).mean(axis=2)
    e = wsum * d + (1.0 - wsum) * UNCOVERED_ERR
    return float(e[use].mean())


def auto_key_frames(geometry: CardGeometry, card: T.Card, frames: Iterable[int], plate: Callable[[int], np.ndarray],
                    remove_mask: Callable[[int], Optional[np.ndarray]], *, threshold: float = AUTO_DEFAULTS["threshold"],
                    min_gap: int = int(AUTO_DEFAULTS["min_gap"]), max_gap: int = int(AUTO_DEFAULTS["max_gap"]),
                    max_keys: int = int(AUTO_DEFAULTS["max_keys"]), side: int = AUTO_SIDE, linear: bool = False,
                    progress: Progress = None, cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """Key mode "auto": the key frames of `card` on `frames` (work range ∩ frames of the camera). Starts from the
    first and the last frame (+ keys so no gap exceeds `max_gap`, 0 = no limit), then adds, one at a time, the frame
    with the largest projection error (`frame_error`) above `threshold` that is at least `min_gap` frames from every
    key, until no frame is above the threshold or `max_keys` is reached.
    -> {"frames": [...], "errors": {frame: error with the final keys}, "max_error", "threshold", "added": n}."""
    s = auto_settings({"threshold": threshold, "min_gap": min_gap, "max_gap": max_gap, "max_keys": max_keys})
    fr = sorted({int(f) for f in frames} & set(geometry.camera.R))
    if not fr:
        raise ValueError("Kamera 3D nie ma żadnej klatki z zakresu pracy")
    prog = progress or (lambda *a: None)
    check = cancel_check or (lambda: None)
    if len(fr) == 1:
        return {"frames": fr, "errors": {fr[0]: 0.0}, "max_error": 0.0, "threshold": s["threshold"], "added": 0}
    scam = _small_camera(geometry.camera, side)
    bare = _bare(card)
    sgeo = CardGeometry(scam, [bare], device="cpu")
    small = _Small(geometry, card, scam, plate, remove_mask, linear, AUTO_MASK_GROW)
    sc = scam.width / float(geometry.camera.width)
    rs = {"edge_px": max(2.0, 120.0 * sc), "edge_hard_px": 1.0, "keep_grow_px": 1.0}
    cache = RD.KeyCache(lambda k: small.get(int(k.frame))[0], size=16)
    memo: Dict[Tuple[int, int, int], float] = {}

    def pair(f: int, ks: List[int]) -> Tuple[int, int]:
        a = max(k for k in ks if k <= f)
        b = min(k for k in ks if k > f) if any(k > f for k in ks) else a
        return a, b

    def errors(ks: List[int]) -> Dict[int, float]:
        out = {}
        for i, f in enumerate(fr):
            if f in ks:
                out[f] = 0.0
                continue
            a, b = pair(f, ks)
            if (f, a, b) not in memo:
                check()
                memo[(f, a, b)] = frame_error(small, sgeo, bare, f, sorted({a, b}), cache, rs)
            out[f] = memo[(f, a, b)]
        return out

    keys = _gaps([fr[0], fr[-1]], fr, s["max_gap"])
    start = len(keys)
    n_est = max(len(fr), 1)
    try:
        while True:
            prog(AUTO_LABEL, min(len(memo), n_est), n_est + len(keys))
            errs = errors(keys)
            cand = [f for f in fr if f not in keys and errs[f] > s["threshold"]
                    and min(abs(f - k) for k in keys) >= s["min_gap"]]
            if not cand or len(keys) >= s["max_keys"]:
                break
            keys = sorted(keys + [max(cand, key=lambda f: (errs[f], -f))])
    finally:
        sgeo.close()
    prog(AUTO_LABEL, 1, 1)
    errs = {int(f): round(float(v), 5) for f, v in errors(keys).items()}
    return {"frames": keys, "errors": errs, "max_error": max(errs.values()) if errs else 0.0,
            "threshold": s["threshold"], "min_gap": s["min_gap"], "max_gap": s["max_gap"], "added": len(keys) - start}


# ----------------------------------------------------------------------------- one: the master key
def master_canvas(geometry: CardGeometry, card: T.Card, frames: Sequence[int], key_frame: int, over: float = OVER,
                  grid_px: int = MASTER_GRID_PX, cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """The canvas of the master key: the view of `key_frame` (its focal and principal point, no distortion) grown so it
    holds what every frame of `frames` shows of the card (sampled every `grid_px`), at most `over` of the frame past each
    side, never smaller than the frame. -> {"w", "h", "K" (3x3 list), "frame", "x0", "y0"} (x0 / y0: where the frame's
    origin lies in the canvas, negated)."""
    cam = geometry.camera
    W, H = int(cam.width), int(cam.height)
    K0 = np.asarray(cam.K, np.float64)
    probe = T.Key(card.id, int(key_frame), canvas={"w": W, "h": H, "K": K0.tolist(), "frame": int(key_frame)})
    g = max(1, int(grid_px))
    ys, xs = np.mgrid[0:H:g, 0:W:g]
    grid = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    px, py = [], []
    for f in frames:
        if cancel_check:
            cancel_check()
        cw = np.asarray(geometry.card_weight(card, int(f)))[ys, xs].ravel()
        pts = grid[cw > 0.01]
        if not len(pts):
            continue
        q = geometry.points_frame_to_key(pts, int(f), probe, card)
        q = q[np.isfinite(q).all(axis=1)]
        keep = ((q[:, 0] > -over * W) & (q[:, 0] < (1 + over) * W) & (q[:, 1] > -over * H) & (q[:, 1] < (1 + over) * H))
        px.append(q[keep, 0])
        py.append(q[keep, 1])
    x0, y0, x1, y1 = 0.0, 0.0, W - 1.0, H - 1.0
    if px and sum(len(a) for a in px):
        ax, ay = np.concatenate(px), np.concatenate(py)
        x0 = min(x0, float(np.floor(np.percentile(ax, 0.05))) - g)
        y0 = min(y0, float(np.floor(np.percentile(ay, 0.05))) - g)
        x1 = max(x1, float(np.ceil(np.percentile(ax, 99.95))) + g)
        y1 = max(y1, float(np.ceil(np.percentile(ay, 99.95))) + g)
    x0, y0 = max(x0, -float(np.ceil(over * W))), max(y0, -float(np.ceil(over * H)))
    x1, y1 = min(x1, W - 1.0 + float(np.ceil(over * W))), min(y1, H - 1.0 + float(np.ceil(over * H)))
    KC = K0.copy()
    KC[0, 2] -= x0
    KC[1, 2] -= y0
    return {"w": int(x1 - x0 + 1), "h": int(y1 - y0 + 1), "K": KC.tolist(), "frame": int(key_frame),
            "x0": float(x0), "y0": float(y0)}


def _gain(cur: np.ndarray, s: np.ndarray, sel: np.ndarray, limit: int = 200_000) -> np.ndarray:
    """One gain per channel that brings `s` to `cur` on the pixels `sel` (medians, clamped 0.5..2)."""
    idx = np.flatnonzero(sel.ravel())
    if len(idx) > limit:
        idx = idx[:: max(1, len(idx) // limit)]
    a, b = cur.reshape(-1, 3)[idx], s.reshape(-1, 3)[idx]
    g = np.median(a, axis=0) / np.maximum(np.median(b, axis=0), 1e-4)
    return np.clip(g, 0.5, 2.0).astype(np.float32)


def build_master(geometry: CardGeometry, card: T.Card, frames: Iterable[int], plate: Callable[[int], np.ndarray],
                 remove_mask: Callable[[int], Optional[np.ndarray]], *, key_frame: Optional[int] = None,
                 canvas: Optional[Dict[str, Any]] = None, over: float = OVER, mask_grow_px: float = MASTER_MASK_GROW,
                 edge_px: float = MASTER_EDGE_PX, normalise: bool = True, progress: Progress = None,
                 cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """Key mode "one": the master key of `card` accumulated from `frames` (see the module doc).
    `key_frame`: the view of the canvas (default the middle frame); `canvas`: reuse this canvas (a rebuild keeps the
    size of the key's earlier versions). -> {"image" float32 BGR (h, w, 3), "canvas", "acc" float32 (h, w),
    "hole" uint8 0 / 255 (shown by some frame, accumulated weight < MASTER_HOLE), "frames": [frames that added pixels],
    "seen_share": share of the needed canvas covered, "hole_px"}."""
    fr = sorted({int(f) for f in frames} & set(geometry.camera.R))
    if not fr:
        raise ValueError("Kamera 3D nie ma żadnej klatki z zakresu pracy")
    M = int(key_frame) if key_frame is not None else fr[len(fr) // 2]
    if M not in geometry.camera.R:
        raise ValueError(f"Kamera 3D nie ma klatki klucza {M}")
    prog = progress or (lambda *a: None)
    check = cancel_check or (lambda: None)
    W, H = int(geometry.camera.width), int(geometry.camera.height)
    if canvas is None:
        prog(f"{MASTER_LABEL}: płótno", 0, len(fr))
        canvas = master_canvas(geometry, card, fr, M, over, cancel_check=check)
    master = T.Key(card.id, M, canvas={k: canvas[k] for k in ("w", "h", "K", "frame")})
    CW, CH = int(canvas["w"]), int(canvas["h"])
    img = np.zeros((CH, CW, 3), np.float32)
    acc = np.zeros((CH, CW), np.float32)
    seen = np.zeros((CH, CW), bool)
    used: List[int] = []
    order = sorted(fr, key=lambda t: (abs(t - M), t))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(mask_grow_px)) + 1,) * 2) if mask_grow_px > 0 else None
    e = max(1e-3, float(edge_px))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # soft frame border of a frame's contribution (the key frame itself counts whole: its plate is the key)
    border = np.clip((np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy)) + 1.0) / e, 0, 1).astype(np.float32)
    for i, t in enumerate(order):
        check()
        prog(f"{MASTER_LABEL}: klatki", i, len(order))
        mx, my = geometry.key_to_frame(master, card, t)
        cw = np.asarray(geometry.card_weight(card, t), np.float32)
        on_card = (cw > 0.01).astype(np.float32)    # the card's pixels (with its seam), hard
        bw = on_card if t == M else border * on_card
        vis = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        seen |= vis >= MASTER_FULL
        if acc.min() >= MASTER_FULL:
            continue
        rm = _mask01(remove_mask(t), (H, W)) > 0.02
        if se is not None and rm.any():
            rm = cv2.dilate(rm.astype(np.uint8), se) > 0
        if rm.any():
            wf = np.clip(cv2.distanceTransform((~rm).astype(np.uint8), cv2.DIST_L2, 5) / e, 0, 1).astype(np.float32)
            wf = np.minimum(wf, bw)
        else:
            wf = bw
        w = cv2.remap(wf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = (1.0 - acc) * w
        if not (a > 1e-3).any():
            continue
        pl = KS._bgr3(KS._to_float(np.asarray(plate(t))))
        if pl.shape[:2] != (H, W):
            raise ValueError(f"Klatka {t}: plate {pl.shape[1]}×{pl.shape[0]} nie pasuje do kamery 3D {W}×{H}")
        s = cv2.remap(pl, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if normalise and used:
            ov = (acc > 0.9) & (w > 0.9)
            if int(ov.sum()) >= 500:
                s = s * _gain(img / np.maximum(acc, 1e-6)[..., None], s, ov)[None, None, :]
        img += a[..., None] * s
        acc += a
        used.append(int(t))
    prog(f"{MASTER_LABEL}: klatki", len(order), len(order))
    if not used:
        raise ValueError(f"Żadna klatka nie pokazuje czystego tła karty {card.id} – klucz-suma niemożliwy")
    col = img / np.maximum(acc, 1e-6)[..., None]
    full = acc >= MASTER_FULL
    if not full.all():                        # never-clean pixels: the surrounding colour (the regenerator's start)
        fill = RD._diffuse(col, full.astype(np.float32), max(4.0, float(edge_px)))
        col = col * acc[..., None] + fill * (1.0 - acc[..., None])
        col[full] = (img[full] / acc[full][:, None])
    hole = (seen & (acc < MASTER_HOLE)).astype(np.uint8)
    if hole.any():
        hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    need = int(seen.sum())
    return {"image": col.astype(np.float32), "canvas": canvas, "acc": acc, "hole": hole * 255, "frames": used,
            "seen_share": round(float((seen & full).sum()) / max(1, need), 5), "hole_px": int((hole > 0).sum()),
            "key_frame": M}


def save_master(work: Path, card: str, res: Dict[str, Any], ext: str = "png16") -> T.Key:
    """Write a master key built by `build_master`: a new version of the key at its key frame with the key's canvas
    (`source: "master"`), and the never-clean hole as the brush area UNSEEN_AREA of that key (removed when there is
    no hole). Returns the Key."""
    work = Path(work)
    M = int(res["key_frame"])
    cv = {k: res["canvas"][k] for k in ("w", "h", "K", "frame")}
    key = store.load_key(work, card, M)
    if key is not None and key.versions and key.canvas != cv:   # older versions would be projected through a wrong view
        what = "jest zwykłą klatką" if key.canvas is None else "ma inne płótno"
        raise ValueError(f"Klucz {card} / klatka {M} {what} – usuń go i zbuduj klucz-sumę od nowa")
    if key is None or not key.versions:
        store.save_key(work, T.Key(T.check_name(card), M, [], None, cv))
    key = KS.add_version(work, card, M, res["image"], MASTER_SOURCE, ext)
    if key.canvas != cv:
        key.canvas = cv
        store.save_key(work, key)
    areas = store.load_areas(work, card, M)
    if int(res.get("hole_px") or 0) > 0:
        KS.save_brush_layer(work, card, M, UNSEEN_AREA, res["hole"], "add", UNSEEN_LABEL)
    elif any(a.id == UNSEEN_AREA for a in areas):
        store.save_areas(work, card, M, [a for a in areas if a.id != UNSEEN_AREA])
    return store.load_key(work, card, M)
