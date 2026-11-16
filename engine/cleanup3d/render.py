# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – projection and compositing (docs/PLAN_CLEANUP_3D.md, package P6): the keys of the cards projected
through the solved 3D camera onto every plate frame, blurred like the plate, composed into the plate by the Cleaner of
engine/cleanup.py as its source „klucz”. Ported from the experiment removers_test/ (chain v3c):

- `project_frame` (vkey3d.py): per card, frame t = smoothstep crossfade of the card's two nearest keys (active
  versions), each weighted by its soft coverage (0 at the key's own border, 1 from `edge_px` inside) and by the card's
  weight in the frame (`CardGeometry.card_weights` – plane hit, extent, order, soft seam). A key's edge warps
  (`Card.warps`, engine/curves.py) are applied to the key before it is projected. Kept (front) objects are cut out of a
  key's coverage – their pixels in a key would otherwise smear onto the background. Where no key reaches: weight 0.
- `MotionBlur` (mb_match.py, CRIT=hf, MOTION=3d): the fill blurred along the camera's own per-pixel motion (the cards
  projected to t-1 / t+1), the length chosen so the high-frequency energy of the fill in a ring just outside the
  removal mask equals the plate's; one value for the shot + a per-frame value pulled toward it, separately for the sky
  cards and the plane cards. The blur itself = `engine.shutter.path_blur` (the plate's exposure model).
- tone (void_comp.py): plate - fill measured on a ring just outside the object, diffused multi-scale into the mask.
- grain (grain.py, P17): grain added to the patch when the plate is clearly noisier than the keys (after the tone).
- `render`: fill -> blur -> tone -> `engine.cleanup.Cleaner(key_fill=...)`: the removal mask (grown, softened outward),
  holdout, the fallback sources (clean plate, LaMa, OpenCV) for pixels no key covers; output = a normal Cleanup
  version `<work>/inpaint/v###/` (patch / comp / source with code 6 = klucz / run.json with a "c3d" report).
- previews for the UI (8-bit BGR): `view_key`, `view_projection`, `view_cards`.

Devices: the geometry's ("cpu" – NumPy kernel in this process, only the region around the mask; "gpu" – the geometry
worker under GPU.use). On "gpu" the key fill runs in the worker end to end (`KeyFill`, 8-bit / display plates): the
card weights stay resident there, the projection, the motion field, the blur measure of pass 1 (all candidates in one
request), the blur and the tone match of pass 2 are its ops, and only the plate region, small masks and the finished
fill cross the process boundary. The per-pixel rules of those steps are written once in geom_gpu.py (`motion_grid`,
`mb_rows`, `diffuse` / `tone_rule`) over image primitives: OpenCV here (the CPU path, results as before), torch there.
The patch (effects + band) and the grain stay in this process (connected components, distance transforms, medians).
Images: BGR float32, 0..1 display for 8-bit plates, scene-linear for EXR (`linear`: measurements and the tone
correction then run in ACEScct like the Cleaner's inpaint; the blur stays in linear light).
"""
from __future__ import annotations

import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine import cleanup as cu
from engine import curves
from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d import grain as GR
from engine.cleanup3d import store
from engine.cleanup3d import types as T
from engine.cleanup3d.geom import CardGeometry
from engine.cleanup3d.horizon import apply_keyfill as HZ_apply, blend_fix as HZ_blend

SCHEMA = "matte_desk/cleanup3d_render@1.0"
BLUR_MODES = ("auto", "off", "manual")
DEFAULTS: Dict[str, Any] = {
    "edge_px": 120.0,        # soft border of a key's coverage: the relative weight between two keys (vkey3d EDGE)
    "edge_hard_px": 8.0,     # ... and how close to its border a key still fills alone
    "keep_grow_px": 12.0,    # kept objects grown by this in a key before they are cut out of its coverage (KEEP_GROW)
    "motion_blur": "auto",   # auto = matched to the plate per frame | off | manual = `mb_length` everywhere
    "mb_length": 0.25,       # manual: blur length as a fraction of the frame-to-frame motion (= shutter / 360)
    "key_soft_px": 32.0,     # soft outward edge of the key patch (beyond the removal mask + the object's effects)
    "key_grow_px": 16.0,     # the object part of the key patch grown by this (its motion-blurred edge / hair fringe
                             # lies outside the C1 removal mask; the keys are the clean background, so growing is safe)
    "patch_band": None,      # band points of the key patch (P14, engine/cleanup3d/band.py: c3d_band.json)
    "key_effects": True,     # the object's shadow / kicked sand in each frame join the patch (keys.effects_mask)
    "clean_first": False,    # the clear plate (aligned) before the keys: motion control – the keys fill only the rest
    "key_effects_thr": 0.10, # plate vs projected keys (0..1, one gain per channel) above which a pixel is an effect
    "mb_lam": 0.3,           # auto: pull of a frame's own estimate toward the shot's value (per unit of length; LAM)
    "mb_per_frame": True,    # auto: per-frame lengths (False = the shot's value on every frame)
    "mb_scale": 1.0,         # multiplies the chosen lengths (the user's "more / less blur")
    "tone": True,            # tone correction from the ring around the object
    "tone_ring_px": 48.0,    # width of that ring (void_comp RING)
    "grain": "auto",         # grain match of the patch (P17, grain.py): auto = only when the plate is clearly noisier | off | on
    "grain_strength": 1.0,   # multiplies the missing grain (0..2)
    "device": "cpu",         # cpu | gpu (the geometry)
    "horizon_match": False,  # P15: the projected fill moved onto the plate's horizon (engine/cleanup3d/horizon.py)
    "horizon_fade_px": 120.0,    # ... the displacement fades to 0 over this many px above / below the horizon
    "horizon_search_px": 60.0,   # ... the band around the cards' horizon where the edges are searched
}
MB_CANDIDATES = tuple(round(float(v), 4) for v in np.arange(0.0, 0.51, 0.0625))
MB_SIDE = 1920               # the blur is matched on images of at most this long side (4K -> half, as mb_match.py)
MB_RING_OUT, MB_RING_IN = 100, 15     # ring around the mask, px at MB_SIDE (201 / 31 kernels at half of 4K)
MB_MIN_MOTION = 4.0          # px / frame at MB_SIDE: slower pixels say nothing about the blur
MB_MIN_PIXELS = 2000         # at MB_SIDE x 1080; fewer usable ring pixels = the frame takes the shot's value
MOTION_STEP = 4              # the motion field is computed on every 4th pixel (it is smooth) and scaled up
BACKUP = 1e-6                # coverage sum below which a pixel has no key of the crossfade


_CV: List[Any] = []


def _cv() -> Any:
    """The OpenCV primitives of the shared per-frame rules (geom_gpu.CvOps), one instance."""
    if not _CV:
        _CV.append(K_.CvOps())
    return _CV[0]


def _mb_consts() -> Dict[str, float]:
    return {"ring_out": MB_RING_OUT, "ring_in": MB_RING_IN, "min_motion": MB_MIN_MOTION, "min_pixels": MB_MIN_PIXELS}


def render_settings(d: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    s = dict(DEFAULTS)
    if d:
        s.update({k: v for k, v in d.items() if k in DEFAULTS and v is not None})
    for k, lo, hi in (("edge_px", 1.0, 2000.0), ("edge_hard_px", 1.0, 2000.0), ("keep_grow_px", 0.0, 200.0),
                      ("mb_length", 0.0, 2.0), ("key_soft_px", 0.0, 400.0), ("key_grow_px", 0.0, 200.0), ("key_effects_thr", 0.01, 1.0), ("mb_lam", 0.0, 10.0), ("mb_scale", 0.0, 4.0), ("tone_ring_px", 2.0, 400.0)):
        s[k] = float(min(hi, max(lo, float(s[k]))))
    s["edge_hard_px"] = min(s["edge_hard_px"], s["edge_px"])
    s["motion_blur"] = s["motion_blur"] if s["motion_blur"] in BLUR_MODES else "auto"
    s["mb_per_frame"], s["tone"], s["key_effects"] = bool(s["mb_per_frame"]), bool(s["tone"]), bool(s["key_effects"])
    s["clean_first"] = bool(s["clean_first"])
    s["device"] = s["device"] if s["device"] in ("cpu", "gpu") else "cpu"
    from engine.cleanup3d import horizon as HZ
    s.update(HZ.settings(s))
    s["grain"], s["grain_strength"] = GR.clean_mode(s["grain"]), GR.clean_strength(s["grain_strength"])
    return s


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def key_pair(frame: int, keys_of_card: Sequence[T.Key]) -> List[Tuple[T.Key, float]]:
    """[(key, weight)] of the crossfade at `frame`: the last key at or before it and the first after it with the
    smoothstep weights (1 - w, w); outside the keys' range the nearest key alone. A key of weight 0 (the next key when
    the frame IS a key) is listed too: a backup that fills only where the frame's own key does not reach."""
    ks = sorted((k for k in keys_of_card if k.active), key=lambda k: k.frame)
    a = [k for k in ks if k.frame <= frame]
    b = [k for k in ks if k.frame > frame]
    if not a or not b:
        return [((a or b)[-1 if a else 0], 1.0)] if (a or b) else []
    ka, kb = a[-1], b[0]
    w = smoothstep((frame - ka.frame) / float(kb.frame - ka.frame))
    return [(ka, 1.0 - w), (kb, w)]


# ----------------------------------------------------------------------------------------------- key images
KEY_PATCH_PAD = 320                # px more around the removal mask: the object's effects (shadow, sand) reach further


class KeyCache:
    """Key images ready to project (read, resized to the key's view when the file is smaller, edge warps applied) and
    the kept objects' cut of a key – a small LRU (a 4K key = 100 MB)."""

    def __init__(self, read_key: Callable[[T.Key], np.ndarray], size: int = 8):
        self.read_key, self.size = read_key, max(2, int(size))
        self._d: "OrderedDict[Any, np.ndarray]" = OrderedDict()

    def _get(self, k: Any, make: Callable[[], np.ndarray]) -> np.ndarray:
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        v = make()
        self._d[k] = v
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return v

    def image(self, geometry: CardGeometry, card: T.Card, key: T.Key, warped: bool = True) -> np.ndarray:
        def make():
            img = np.asarray(self.read_key(key), np.float32)
            if img.ndim != 3 or img.shape[2] < 3:
                raise ValueError(f"Klucz {key.card} / klatka {key.frame}: obraz musi mieć 3 kanały koloru")
            img = img[..., :3]
            v = geometry.key_view(key)
            if img.shape[:2] != (int(v["h"]), int(v["w"])):
                img = cv2.resize(img, (int(v["w"]), int(v["h"])), interpolation=cv2.INTER_CUBIC)
            wp = (card.warps or {}).get(str(int(key.frame))) if warped else None
            return np.ascontiguousarray(curves.apply_warp(img, wp) if wp else img)
        return self._get(("img", card.id, key.frame, key.active, warped), make)

    def keep_cut(self, geometry: CardGeometry, card: T.Card, key: T.Key, keep_alpha: Callable[[int], Optional[np.ndarray]],
                 grow_px: float) -> Optional[np.ndarray]:
        """(key h, w) factor 0..1 on the key's coverage: 0 on the kept objects of the key's frame (grown, soft)."""
        cv = key.canvas
        src_frame = int(cv.get("frame", key.frame)) if cv else int(key.frame)

        def make():
            a = keep_alpha(src_frame)
            if a is None:
                return np.zeros((0, 0), np.float32)
            a = np.asarray(a, np.float32)
            if cv:                                    # the key's own canvas: same pose, other intrinsics
                mx, my = geometry.key_to_frame(key, card, src_frame)
                a = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            g = int(round(grow_px))
            if g > 0:
                a = cv2.dilate(a, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
                a = cv2.GaussianBlur(a, (0, 0), g / 2.0)
            return np.clip(1.0 - a * 1.5, 0.0, 1.0).astype(np.float32)
        f = self._get(("keep", key.frame, None if not cv else (card.id, key.active), float(grow_px)), make)
        return f if f.size else None


# ----------------------------------------------------------------------------------------------- projection
Roi = Tuple[int, int, int, int]


def _roi(geometry: CardGeometry, roi: Optional[Roi]) -> Roi:
    w, h = int(geometry.camera.width), int(geometry.camera.height)
    if roi is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = (int(v) for v in roi)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _frame_to_key(geometry: CardGeometry, frame: int, key: T.Key, card: T.Card, roi: Roi) -> Tuple[np.ndarray, np.ndarray]:
    """Sampling maps of the region `roi` of the frame into the key. GPU: the worker's full-frame maps, cropped;
    CPU: the same kernel on the region's rays only."""
    x0, y0, x1, y1 = roi
    full = (x0, y0, x1, y1) == (0, 0, int(geometry.camera.width), int(geometry.camera.height))
    rays = getattr(geometry, "_frame_rays", None)
    if geometry.device == "gpu" or full or rays is None:
        mx, my = geometry.frame_to_key(frame, key, card)
        return mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
    tgt = geometry.frame_view(frame)
    x, y = rays(tgt)
    mx, my, _ok = K_.project(K_.NP, x[y0:y1, x0:x1], y[y0:y1, x0:x1], tgt, geometry.key_view(key), geometry.card_spec(card))
    return mx.astype(np.float32), my.astype(np.float32)


def _gpu_key_id(cache: "KeyCache", card: T.Card, key: T.Key, keep: bool, s: Mapping[str, Any]) -> str:
    """The id of a key image resident in the GPU worker: one KeyCache (= one Scene) x card x key version x warp x cut."""
    wp = (card.warps or {}).get(str(int(key.frame)))
    return (f"{id(cache)}|{card.id}|{int(key.frame)}|{key.active}|{hash(repr(wp)) if wp else 0}"
            f"|{('cut', float(s['keep_grow_px'])) if keep else ''}")


def _project_gpu(geometry: CardGeometry, cards: Sequence[T.Card], keys_by_card: Mapping[str, Sequence[T.Key]],
                 frame: int, cache: "KeyCache", keep_alpha: Optional[Callable[[int], Optional[np.ndarray]]],
                 s: Mapping[str, Any], roi: Roi, parts: Optional[Dict[str, Any]],
                 blur: Optional[Tuple[np.ndarray, np.ndarray]] = None, resident: Optional[Dict[str, Any]] = None
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """`project_frame` in the GPU worker (geom_gpu op "project"): the key images stay resident there (sent once per
    version, `keys_put`), the frame region's maps, crossfade, coverage and cut are computed on the GPU; `blur` =
    (lx, ly) per-pixel blur vectors applied on the GPU too (render.blur_fill).
    `resident` (KeyFill on the GPU): the card weights stay in the worker (op frame_weights), the float projection is
    kept there for the blur measure, {"save": {"fill", "wgt"}} = pass 1's files written by the worker; nothing comes
    back - returns None, resident["any"] = whether any key reaches the region."""
    from engine.cleanup3d.geom import _jsonable_view
    x0, y0, x1, y1 = roi
    rh, rw = y1 - y0, x1 - x0
    stats = geometry.frame_weights(frame, roi) if resident is not None else None
    weights = geometry.card_weights(frame) if stats is None else None
    temp: List[str] = []
    req_cards, need = [], {}
    for card in sorted(cards, key=lambda c: c.order):
        pair = key_pair(frame, keys_by_card.get(card.id) or [])
        if stats is not None:
            hit, mean_w = stats[card.id]
            if not pair or not hit:
                continue
            wref: Any = {"resident": geometry.card_index(card)}
        else:
            cw = weights[card.id][y0:y1, x0:x1]
            if not pair or not (cw > 1e-4).any():
                continue
            wref = geometry._tmp("cw")
            np.save(wref, cw.astype(np.float16))
            temp.append(wref)
            mean_w = float(cw.mean())
        ks = []
        for key, tw in pair:
            kid = _gpu_key_id(cache, card, key, keep_alpha is not None, s)
            need[kid] = (card, key)
            ks.append({"id": kid, "view": _jsonable_view(geometry.key_view(key)), "tw": float(tw)})
        req_cards.append({"spec": geometry.card_spec(card), "weight": wref, "keys": ks})
        if parts is not None:
            parts[card.id] = {"keys": [(int(k.frame), round(float(tw), 4)) for k, tw in pair], "weight": mean_w}
    if not req_cards:
        if resident is not None:
            resident["any"] = False
            return None
        return np.zeros((rh, rw, 3), np.float32), np.zeros((rh, rw), np.float32)
    out = geometry._tmp("proj")
    req: Dict[str, Any] = {"op": "project", "frame": _jsonable_view(geometry.frame_view(frame)), "roi": [x0, y0, x1, y1],
                           "edge_px": float(s["edge_px"]), "hard_k": float(s["edge_px"]) / float(s["edge_hard_px"]),
                           "cards": req_cards, "out": out}
    if resident is not None:
        req.update({"out": None, "keep": True, "save": resident.get("save")})
    if blur is not None:
        lxp, lyp = geometry._tmp("lx"), geometry._tmp("ly")
        np.save(lxp, np.asarray(blur[0], np.float32))
        np.save(lyp, np.asarray(blur[1], np.float32))
        temp += [lxp, lyp]
        req["blur"] = {"lx": lxp, "ly": lyp}
    wk = geometry._worker()
    try:
        for _ in range(3):
            resp = wk.call(req)
            missing = resp.get("missing") or []
            if not missing:
                break
            put = []
            for kid in missing:
                card, key = need[kid]
                ip = geometry._tmp("key")
                np.save(ip, cache.image(geometry, card, key).astype(np.float16))
                temp.append(ip)
                cp = None
                if keep_alpha is not None:
                    cut = cache.keep_cut(geometry, card, key, keep_alpha, s["keep_grow_px"])
                    if cut is not None:
                        cp = geometry._tmp("cut")
                        np.save(cp, cut.astype(np.float32))
                        temp.append(cp)
                put.append({"id": kid, "img": ip, "cut": cp})
            wk.call({"op": "keys_put", "keys": put})
        else:
            raise RuntimeError("worker geometrii 3D nie przyjął kluczy")
        if resident is not None:
            resident["any"] = bool(resp.get("any"))
            return None
        arr = np.load(out)
    finally:
        for p in temp + [out]:
            try:
                os.remove(p)
            except OSError:
                pass
    return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3])


def project_frame(geometry: CardGeometry, cards: Sequence[T.Card], keys_by_card: Mapping[str, Sequence[T.Key]],
                  frame: int, read_key: Callable[[T.Key], np.ndarray],
                  keep_alpha: Optional[Callable[[int], Optional[np.ndarray]]] = None, *,
                  settings: Optional[Mapping[str, Any]] = None, roi: Optional[Roi] = None,
                  cache: Optional[KeyCache] = None, parts: Optional[Dict[str, Any]] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The cards' keys projected onto plate frame `frame` -> (fill BGR float32, weight 0..1 float32), of the frame's
    size or of `roi` (x0, y0, x1, y1). read_key(key) -> the key's active image (BGR float); keep_alpha(f) -> the
    alpha (0..1, plate size) of the objects that stay in plate frame f, or None. `cache`: a KeyCache kept between
    frames (else every call reads and warps its keys again). `parts`, when a dict, receives per card
    {"keys": [(frame, temporal weight)], "weight": mean card weight x key weight}."""
    s = render_settings(settings)
    roi = _roi(geometry, roi)
    x0, y0, x1, y1 = roi
    rh, rw = y1 - y0, x1 - x0
    cache = cache or (read_key if isinstance(read_key, KeyCache) else KeyCache(read_key))
    if getattr(geometry, "device", "cpu") == "gpu" and hasattr(geometry, "_worker") and isinstance(cache, KeyCache):
        return _project_gpu(geometry, cards, keys_by_card, frame, cache, keep_alpha, s, roi, parts)
    weights = geometry.card_weights(frame)
    acc = np.zeros((rh, rw, 3), np.float32)
    wsum = np.zeros((rh, rw), np.float32)
    hard_k = s["edge_px"] / s["edge_hard_px"]
    for card in sorted(cards, key=lambda c: c.order):
        cw = weights[card.id][y0:y1, x0:x1]
        pair = key_pair(frame, keys_by_card.get(card.id) or [])
        if not pair or not (cw > 1e-4).any():
            continue
        num = np.zeros((rh, rw, 3), np.float32)
        den = np.zeros((rh, rw), np.float32)
        reach = np.zeros((rh, rw), np.float32)
        col: Optional[np.ndarray] = None              # the crossfade, once a backup key tops it up
        for key, tw in sorted(pair, key=lambda p: -p[1]):
            backup = tw <= 0.0
            if backup:
                if col is None:
                    col = num / np.maximum(den, BACKUP)[..., None]
                    reach = np.where(den > BACKUP, reach, 0.0).astype(np.float32)
                if not ((reach < 0.999) & (cw > 1e-4)).any():
                    continue                          # the frame's own key covers everything this card shows
            img = cache.image(geometry, card, key)
            mx, my = _frame_to_key(geometry, frame, key, card, roi)
            cov = K_.coverage(K_.NP, mx, my, img.shape[1], img.shape[0], s["edge_px"]).astype(np.float32)
            hard = np.minimum(cov * hard_k, 1.0)
            if keep_alpha is not None:
                cut = cache.keep_cut(geometry, card, key, keep_alpha, s["keep_grow_px"])
                if cut is not None:
                    c = cv2.remap(cut, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    cov, hard = cov * c, hard * c
            pix = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            if backup:                                # only where the crossfade's keys do not reach (fully)
                wb = (1.0 - reach) * hard
                col = (col * reach[..., None] + pix * wb[..., None]) / np.maximum(reach + wb, BACKUP)[..., None]
                reach = reach + wb
                continue
            rel = cov * np.float32(tw)
            num += pix * rel[..., None]
            reach = np.where(rel > 0, np.maximum(reach, hard), reach)
            den += rel
        if col is None:
            col = num / np.maximum(den, BACKUP)[..., None]
            reach = np.where(den > BACKUP, reach, 0.0)
        a = (cw * reach).astype(np.float32)
        acc += col * a[..., None]
        wsum += a
        if parts is not None:
            parts[card.id] = {"keys": [(int(k.frame), round(float(tw), 4)) for k, tw in pair], "weight": float(a.mean())}
    fill = acc / np.maximum(wsum, 1e-6)[..., None]
    return fill, np.clip(wsum, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- motion blur
def motion_field(geometry: CardGeometry, frame: int, roi: Optional[Roi] = None, step: int = MOTION_STEP
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dx, dy, sky) of the region: the camera's own motion per frame at every pixel (px; half the path from t-1 to
    t+1 through the cards, as mb_match.py MOTION=3d) and the weight 0..1 of the sky cards there. Computed with the
    geometry's kernel on every `step`-th pixel and scaled up (the field is smooth)."""
    x0, y0, x1, y1 = _roi(geometry, roi)
    rh, rw = y1 - y0, x1 - x0
    weights = geometry.card_weights(frame)
    ws = [weights[c.id] for c in geometry.cards]
    # sample points = the centres of nx x ny equal blocks of the region (so the scaling up is aligned) – geom_gpu
    dx, dy, sky = K_.motion_grid(K_.NP, geometry.frame_view(frame), *_neighbour_views(geometry, frame),
                                 [(geometry.card_spec(c), c.kind) for c in geometry.cards],
                                 lambda i, iy, ix: ws[i][iy[:, None], ix[None, :]].astype(np.float64),
                                 (x0, y0, x1, y1), step)

    def up(a):
        return cv2.resize(a.astype(np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR) if a.shape != (rh, rw) \
            else a.astype(np.float32)
    return up(dx), up(dy), np.clip(up(sky), 0.0, 1.0)


def _neighbour_views(geometry: CardGeometry, frame: int) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The views of frames t-1 / t+1 of the motion field (None where the camera has no such frame)."""
    cam = geometry.camera
    return tuple(geometry.frame_view(f2) if f2 in cam.R else None for f2 in (frame - 1, frame + 1))


def _motion_gpu(geometry: CardGeometry, frame: int, roi: Roi, out: Optional[str] = None) -> None:
    """`motion_field` of the region in the geometry worker, kept resident there (op "motion"; the frame's card
    weights resident first)."""
    from engine.cleanup3d.geom import _jsonable_view
    geometry.frame_weights(frame, roi)
    pv, nv = _neighbour_views(geometry, frame)
    geometry._worker().call({"op": "motion", "frame": _jsonable_view(geometry.frame_view(frame)),
                             "prev": None if pv is None else _jsonable_view(pv),
                             "next": None if nv is None else _jsonable_view(nv), "roi": [int(v) for v in roi],
                             "step": MOTION_STEP, "out": out,
                             "cards": [{"spec": geometry.card_spec(c), "kind": c.kind} for c in geometry.cards]})


def _gray(img: np.ndarray, linear: bool) -> np.ndarray:
    """Gray of the measurements, 0..255 float (8-bit scale of mb_match.py; linear plates through ACEScct)."""
    return cv2.cvtColor(cu.to_model(img, linear), cv2.COLOR_BGR2GRAY) * 255.0


def hf_energy_map(g: np.ndarray) -> np.ndarray:
    return np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))


def blur_fill(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, length: Any, gpu: Any = None) -> np.ndarray:
    """`img` averaged along length x (dx, dy) at every pixel (engine.shutter.path_blur per channel); `length` a
    number or a per-pixel map. `gpu`: a CardGeometry on device "gpu" – the same blur in its worker."""
    from engine.shutter import path_blur
    lx, ly = (dx * length).astype(np.float32), (dy * length).astype(np.float32)
    if gpu is not None and getattr(gpu, "device", "cpu") == "gpu" and hasattr(gpu, "_worker"):
        if float(np.hypot(lx, ly).max() if lx.size else 0.0) < 0.5:
            return np.asarray(img, np.float32)
        ip, lxp, lyp, op = gpu._tmp("bimg"), gpu._tmp("blx"), gpu._tmp("bly"), gpu._tmp("bout")
        try:
            np.save(ip, np.asarray(img, np.float32))
            np.save(lxp, lx)
            np.save(lyp, ly)
            gpu._worker().call({"op": "path_blur", "img": ip, "lx": lxp, "ly": lyp, "out": op})
            return np.load(op)
        finally:
            for p in (ip, lxp, lyp, op):
                try:
                    os.remove(p)
                except OSError:
                    pass
    if img.ndim == 2:
        return path_blur(img, lx, ly)
    return np.dstack([path_blur(np.ascontiguousarray(img[..., c]), lx, ly) for c in range(img.shape[2])])


PATCH_BLUR_LENGTHS = tuple(round(float(v), 4) for v in np.arange(0.0, 1.51, 0.0625))
PATCH_BLUR_MARGIN = 160     # px around the hole where the generator's output is compared with the plate


def match_patch_blur(geometry: CardGeometry, frame: int, fill: np.ndarray, plate: np.ndarray, hole: np.ndarray,
                     known: Optional[np.ndarray] = None, margin: int = PATCH_BLUR_MARGIN,
                     lengths: Sequence[float] = PATCH_BLUR_LENGTHS) -> Tuple[np.ndarray, float]:
    """The motion blur of a GENERATED patch, matched to the plate of its own frame: a generator (VOID + upscaler,
    FLUX 3 Image Klein) redraws the area around the hole too, so there its output and the plate show the same scene;
    the length along the camera's motion (`motion_field` of `frame`, fraction of the frame-to-frame motion) that makes
    the output's detail energy equal to the plate's THERE is applied to the whole crop around the hole.
    `known`: where the output is comparable (default: the crop outside the hole). Returns (fill, length); the fill is
    unchanged outside the crop (and when there is nothing to compare or no motion)."""
    hole = np.asarray(hole) > 0
    if not hole.any():
        return fill, 0.0
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
    x1, y1 = min(w, int(xs.max()) + margin + 1), min(h, int(ys.max()) + margin + 1)
    dx, dy, _sky = motion_field(geometry, frame, roi=(x0, y0, x1, y1))
    crop = np.asarray(fill, np.float32)[y0:y1, x0:x1]
    cmp_ = ~hole[y0:y1, x0:x1] if known is None else (np.asarray(known)[y0:y1, x0:x1] > 0) & ~hole[y0:y1, x0:x1]
    cmp_ &= np.hypot(dx, dy) > MB_MIN_MOTION
    if int(cmp_.sum()) < 500:
        return fill, 0.0
    q = min(1.0, MB_SIDE / float(max(crop.shape[:2])))       # measured at most MB_SIDE wide, applied at full size
    rs = (lambda a, i: cv2.resize(a, (max(1, round(a.shape[1] * q)), max(1, round(a.shape[0] * q))), interpolation=i)) \
        if q < 1.0 else (lambda a, i: a)
    gf = rs(_gray(crop, False), cv2.INTER_AREA)
    gp = rs(_gray(np.asarray(plate, np.float32)[y0:y1, x0:x1], False), cv2.INTER_AREA)
    dxq, dyq = rs(dx, cv2.INTER_LINEAR) * q, rs(dy, cv2.INTER_LINEAR) * q
    sel = rs(cmp_.astype(np.uint8), cv2.INTER_NEAREST) > 0
    target = np.log(max(1e-6, float(hf_energy_map(gp)[sel].mean())))
    best, length = None, 0.0
    for L in lengths:
        e = hf_energy_map(blur_fill(gf, dxq, dyq, L) if L > 0 else gf)
        d = abs(np.log(max(1e-6, float(e[sel].mean()))) - target)
        if best is None or d < best - 1e-9:
            best, length = d, float(L)
    if length <= 0:
        return fill, 0.0
    out = np.array(fill, np.float32, copy=True)
    out[y0:y1, x0:x1] = blur_fill(crop, dx, dy, length)
    return out, length


class MotionBlur:
    """The blur lengths of a shot (mb_match.py CRIT=hf): `measure` every frame, then `solve`.
    Length = fraction of the frame-to-frame motion (x 360 = the shutter angle the fill gets)."""

    def __init__(self, frame_size: Tuple[int, int], lam: float = 0.3, per_frame: bool = True,
                 candidates: Sequence[float] = MB_CANDIDATES):
        self.lam, self.per_frame, self.cand = float(lam), bool(per_frame), tuple(candidates)
        side = float(max(frame_size))
        self.q = min(1.0, MB_SIDE / side)             # plate px -> analysis px
        self.k = self.q * side / MB_SIDE              # analysis px relative to mb_match.py's 1920-wide images (<= 1)
        self.rows: Dict[str, Dict[int, Dict[float, float]]] = {"ground": {}, "sky": {}}
        self.shot: Dict[str, float] = {}
        self.lengths: Dict[int, Tuple[float, float]] = {}

    def measure(self, frame: int, fill: np.ndarray, plate: np.ndarray, mask: np.ndarray, usable: np.ndarray,
                dx: np.ndarray, dy: np.ndarray, sky: np.ndarray, linear: bool = False) -> Dict[str, int]:
        """One frame (arrays of one region): fill / plate BGR, mask = the removal mask (bool), usable = where the
        fill is the keys alone and the plate is the background (bool), (dx, dy, sky) = motion_field."""
        rows, n = K_.mb_rows(_cv(), cu.to_model(fill, linear), cu.to_model(plate, linear), mask, usable, dx, dy, sky,
                             self.q, self.k, self.cand, _mb_consts())
        return self.add_rows(frame, rows, n)

    def add_rows(self, frame: int, rows: Mapping[str, Mapping[float, float]], n: Mapping[str, int]) -> Dict[str, int]:
        """One frame's errors per region and candidate (geom_gpu.mb_rows, here or in the GPU worker)."""
        for r, v in rows.items():
            for L, e in v.items():
                self.rows[r].setdefault(frame, {})[float(L)] = e
        return {r: int(c) for r, c in n.items()}

    def measure_gpu(self, geometry: CardGeometry, frame: int, roi: Roi, plate: np.ndarray, mask: np.ndarray,
                    keep: Optional[np.ndarray]) -> Dict[str, int]:
        """`measure` in the geometry worker on its resident projection and motion field of `roi` (8-bit plates):
        plate = the region's plate, mask = the removal object (bool), keep = the kept objects > 0.02 (bool) or None;
        the usable pixels (keys fully cover, no kept object) are made there from the float weight."""
        from engine.cleanup3d.geom import _jsonable_view
        paths = {"plate": geometry._tmp("mbp"), "obj": geometry._tmp("mbo")}
        np.save(paths["plate"], np.asarray(plate, np.float32))
        np.save(paths["obj"], np.asarray(mask).astype(np.uint8))
        if keep is not None:
            paths["keep"] = geometry._tmp("mbk")
            np.save(paths["keep"], np.asarray(keep).astype(np.uint8))
        try:
            resp = geometry._worker().call({"op": "mb_measure", "frame": _jsonable_view(geometry.frame_view(frame)),
                                            "roi": [int(v) for v in roi], "q": self.q, "k": self.k,
                                            "cands": list(self.cand), "consts": _mb_consts(),
                                            "key_full": cu.KEY_FULL, **paths})
        finally:
            for p in paths.values():
                try:
                    os.remove(p)
                except OSError:
                    pass
        rows = {r: {float(L): float(e) for L, e in v} for r, v in (resp.get("rows") or {}).items()}
        return self.add_rows(frame, rows, resp.get("n") or {})

    @property
    def ring_px(self) -> int:
        """Reach of the measuring ring around the mask, plate px."""
        return int(np.ceil(MB_RING_OUT * self.k / self.q))

    def solve(self, frames: Sequence[int]) -> Dict[int, Tuple[float, float]]:
        """frame -> (ground length, sky length): the shot's value = least summed error; a frame's own best pulled
        toward it by `lam`, median over 5 frames. A region never measured takes the other's (both none = 0)."""
        frames = list(frames)
        per: Dict[str, List[float]] = {}
        for r, rows in self.rows.items():
            if not rows:
                continue
            err = {L: sum(row[L] for row in rows.values()) for L in self.cand}
            g = min(self.cand, key=lambda L: err[L])
            self.shot[r] = g
            raw = [min(self.cand, key=lambda L: rows[f][L] + self.lam * abs(L - g)) if f in rows else g for f in frames]
            per[r] = [float(np.median(raw[max(0, i - 2):i + 3])) for i in range(len(frames))] if self.per_frame \
                else [g] * len(frames)
        zero = [0.0] * len(frames)
        gr = per.get("ground") or per.get("sky") or zero
        sk = per.get("sky") or per.get("ground") or zero
        self.lengths = {f: (gr[i], sk[i]) for i, f in enumerate(frames)}
        return self.lengths


# ----------------------------------------------------------------------------------------------- tone
def _diffuse(val: np.ndarray, w: np.ndarray, sigma: float, ds: int = 8) -> np.ndarray:
    """`val` (h, w, 3), known where w > 0, spread everywhere: multi-scale normalised convolution at 1 / ds resolution
    (void_comp.py `diffuse`; the weight is applied BEFORE the reduction, so pixels outside the ring never leak in)."""
    return K_.diffuse(_cv(), val, w, sigma, ds)


def tone_match(fill: np.ndarray, plate: np.ndarray, obj: np.ndarray, usable: np.ndarray, ring_px: float,
               linear: bool = False) -> np.ndarray:
    """The fill brought to the plate's tone: plate - fill on a ring of `ring_px` just outside the object `obj` (bool),
    only where `usable` (the fill is the keys, the plate is the background), diffused into the object."""
    out = K_.tone_rule(_cv(), cu.to_model(fill, linear), cu.to_model(plate, linear), np.asarray(obj, bool),
                       np.asarray(usable, bool), ring_px)
    if out is None:
        return fill
    return cu.from_model(out, linear) if linear else out


# ----------------------------------------------------------------------------------------------- the scene of a shot
def extent_layers(work: Path, cards: Sequence[T.Card]) -> Dict[str, Dict[str, np.ndarray]]:
    """Brush layers of the cards' extents: `a_<id>.png` in `<work>/inpaint/cards/<card>/` (where the API writes them,
    store.extent_dir), else next to the key of `extent_frame`. The card's own folder goes FIRST: a regeneration area
    of that key may carry the same id (the editor numbers both a1, a2, ...) and is not the extent."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c in cards:
        for a in c.extent:
            if a.kind != "brush":
                continue
            dirs = [store.extent_dir(work, c.id)]
            if c.extent_frame is not None:
                dirs.append(T.key_dir(work, c.id, int(c.extent_frame)))
            for d in dirs:
                m = cv2.imread(str(d / str(a.png)), cv2.IMREAD_GRAYSCALE) if (d / str(a.png)).is_file() else None
                if m is not None:
                    out.setdefault(c.id, {})[a.id] = m
                    break
    return out


def depth_needs_gpu(cards: Sequence[T.Card]) -> bool:
    """True when the scene has a depth card (P18) that is on and ComfyUI's Python is there: such a scene runs its
    geometry on the GPU whatever the project's device (MD_C3D_DEPTH_GPU=0 keeps the asked device – tests)."""
    import os
    if os.environ.get("MD_C3D_DEPTH_GPU", "1") == "0":
        return False
    if not any(c.kind == "depth" and not getattr(c, "off", False) for c in cards):
        return False
    from engine.cleanup3d.geom import worker_python
    py = worker_python()
    return py is not None and Path(py).exists()


class Scene:
    """What a render / a preview needs of a shot, read from the contract files of `work` (`<work>/matte_desk`):
    the camera, the cards, the geometry, the keys (with an active version) per card, a key cache.
    Raises ValueError (Polish, for the UI) when the camera or the cards are missing. `close()` ends an own GPU worker."""

    def __init__(self, work: Path, device: str = "cpu", gpu: Any = None,
                 layers: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None, key_cache: int = 8):
        from engine.cleanup3d import keys as KS
        self.work = Path(work)
        cam = store.load_camera(self.work)
        if cam is None or not cam.frames:
            raise ValueError("Brak kamery 3D – policz ją w kroku „Kamera 3D”")
        cards = store.load_cards(self.work)
        if not cards:
            raise ValueError("Brak kart – dodaj je w kroku „Karty”")
        cards = T.active_cards(cards)                     # P16: off (eye) / not soloed (S) = not in the scene
        if not cards:
            raise ValueError("Wszystkie karty są wyłączone (👁) albo solo (S) mają tylko wyłączone karty – włącz którąś "
                             "na osi czasu (wiersz karty)")
        self.camera, self.cards = cam, cards
        if device == "cpu" and gpu is None and depth_needs_gpu(cards):
            device = "gpu"                     # user 2026-11-10: depth cards on the GPU (CPU ~2 min / 4K map, CUDA ~11 s)
        self.geometry = CardGeometry(cam, cards, layers if layers is not None else extent_layers(self.work, cards),
                                     device=device, gpu=gpu)
        self.keys_by_card: Dict[str, List[T.Key]] = {
            c.id: [k for k in store.list_keys(self.work, c.id) if k.active] for c in cards}
        self.cache = KeyCache(lambda k: KS.read_key_image(self.work, k), key_cache)

    @property
    def size(self) -> Tuple[int, int]:
        return int(self.camera.width), int(self.camera.height)

    def has_keys(self) -> bool:
        return any(self.keys_by_card.values())

    def card(self, card_id: str) -> T.Card:
        for c in self.cards:
            if c.id == card_id:
                return c
        raise ValueError(f"Nieznana karta: {card_id}")

    def key(self, card_id: str, frame: int) -> T.Key:
        for k in self.keys_by_card.get(card_id) or []:
            if k.frame == int(frame):
                return k
        raise ValueError(f"Brak klucza {card_id} / klatka {int(frame)}")

    def close(self) -> None:
        self.geometry.close()


# ----------------------------------------------------------------------------------------------- render
class KeyFill:
    """The per-frame fill of the Cleaner's source „klucz”: projection -> motion blur -> tone, in the region around
    the frame's removal mask. With motion blur "auto", `prepare(frames)` projects and measures every frame first
    (the lengths need the whole range) and keeps the projections in `tmp` until they are used.
    `gpu` (the scene's geometry on device "gpu", 8-bit plates): pass 1 = ops project (weights resident, pass 1's
    files written by the worker) -> motion -> mb_measure, pass 2 = op keyfill (blur + tone) – the same rules as the
    CPU path (geom_gpu.py), the patch and the grain here."""

    def __init__(self, scene: Scene, s: Dict[str, Any], cs: Dict[str, Any], linear: bool,
                 plate: Callable[[int], np.ndarray], remove_alphas: Callable[[int], List[np.ndarray]],
                 keep_alphas: Callable[[int], List[np.ndarray]], tmp: Path,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 cancel_check: Optional[Callable[[], None]] = None):
        self.scene, self.s, self.cs, self.linear = scene, s, cs, linear
        self.plate, self.remove_alphas, self.keep_alphas = plate, remove_alphas, keep_alphas
        self.tmp = Path(tmp)
        self.progress = progress or (lambda *a: None)
        self.cancel_check = cancel_check or (lambda: None)
        self.w, self.h = scene.size
        self.mb = MotionBlur((self.w, self.h), s["mb_lam"], s["mb_per_frame"])
        self.stats: Dict[int, Dict[str, Any]] = {}
        self.seconds = {"project": 0.0, "blur": 0.0, "tone": 0.0}
        self._saved: Dict[int, Roi] = {}
        self._lengths: Dict[int, Tuple[float, float]] = {}
        geo = scene.geometry
        self.gpu = (getattr(geo, "device", "cpu") == "gpu" and hasattr(geo, "_worker") and not linear
                    and isinstance(scene.cache, KeyCache))
        self._fr: Optional[Tuple[int, Optional[Dict[str, Any]]]] = None

    # -- inputs of a frame
    def _keep(self, f: int) -> Optional[np.ndarray]:
        al = self.keep_alphas(f) if self.cs["holdout"] else []
        return np.max(np.stack([np.asarray(a, np.float32) for a in al]), axis=0) if al else None

    def _frame(self, f: int) -> Optional[Dict[str, Any]]:
        """The frame's masks and region (the last frame cached: the Cleaner's mask and the fill both ask)."""
        if self._fr is not None and self._fr[0] == f:
            return self._fr[1]
        fr = self._frame_new(f)
        self._fr = (f, fr)
        return fr

    def _frame_new(self, f: int) -> Optional[Dict[str, Any]]:
        rm = self.remove_alphas(f)
        if not rm:
            return None
        obj = np.max(np.stack([np.asarray(a, np.float32) for a in rm]), axis=0) > 0.02
        keep = self._keep(f)
        if not hasattr(self, "_c1_pts"):      # the C1 „Inpaint” band points of the cleaned thing (engine/inpaint_band)
            from engine import inpaint_band as IB
            self._c1_pts = IB.points_fn(self.cs, self.remove_alphas)
        g, sf = cu.band_at(self.cs, f)        # the C1 slider keys of this frame
        m = cu.removal_mask(rm, g, sf, self._c1_pts(f) if self._c1_pts else None)
        if keep is not None:
            m = m * (1.0 - (keep > 0.5))
        pts = self._band_points(f)
        reach = max([float(self.s["key_soft_px"])] + [float(p[2]) for p in pts or []])   # band points push the patch out
        pad = int(self.s["tone_ring_px"]) + 8 + KEY_PATCH_PAD + int(np.ceil(reach)) + int(self.s["key_grow_px"])
        if self.s["motion_blur"] != "off":
            pad = max(pad, self.mb.ring_px + 48)       # the measuring ring + the reach of the blur's samples
        box = cu.mask_bbox([m, obj], pad, self.w, self.h)
        if box is None:
            return None
        return {"roi": box, "obj": obj, "keep": keep, "mask": m}

    def _band(self):
        """The key patch band (engine/objband.cleanup_band: key_soft_px + the project's band points, outer side)."""
        if getattr(self, "_band_v", None) is None:
            from engine.objband import cleanup_band
            self._band_v = cleanup_band(self.s, self.s.get("patch_band"))
        return self._band_v

    def _band_points(self, f: int):
        """The band points of frame f (plate px, widths scaled by the slider), None without points – carried between
        their keys by the removal objects' motion (engine/cleanup3d/band.points_fn)."""
        if getattr(self, "_band_fn", 0) == 0:
            from engine.cleanup3d import band as C3B
            self._band_fn = C3B.points_fn(self._band(), lambda k: C3B.object_mask(self.remove_alphas(k)))
        return self._band_fn(f) if self._band_fn else None

    def _patch(self, f: int, fr: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """(hole, patch) of the frame (since P14: the patch is the band of engine/cleanup3d/band.py – key_soft_px shaped by
        the band points, outer side; the object part grown by key_grow_px): hole = the removal mask U the object's EFFECTS in this frame – where the plate
        differs from the projected keys (the clean background) next to the object: shadow, kicked-up sand
        (keys.effects_mask, measured where the keys fully cover); patch = 1 in the hole, a smoothstep ramp of
        `key_soft_px` outward (there the projected key and the plate show the same background), cut by the kept objects.
        Cached per frame with the projection (the Cleaner asks for its mask before the fill)."""
        if getattr(self, "_patch_f", None) == f:
            return self._patch_v
        from engine.cleanup3d import keys as KS
        from engine.cleanup3d import band as C3B
        x0, y0, x1, y1 = fr["roi"]
        fill, wgt, parts = self._projection(f, fr["roi"])
        hole = fr["mask"] > 0
        # the object part grown by key_grow_px (its motion-blurred edge / hair fringe outside the C1 mask), the kept
        # objects still cut out (as the removal mask)
        grown = C3B.grow(fr["obj"][y0:y1, x0:x1], float(self.s["key_grow_px"]))
        if fr["keep"] is not None:
            grown &= ~(fr["keep"][y0:y1, x0:x1] > 0.5)
        hole[y0:y1, x0:x1] |= grown
        core = hole[y0:y1, x0:x1].copy()           # the object part: what the band points snap to (band.py)
        if (wgt > 0).any() and self.s["key_effects"]:
            plate = self.plate(f)[y0:y1, x0:x1].astype(np.float32)
            full = wgt >= cu.KEY_FULL
            fl = np.where(full[..., None], fill, plate)                 # no key = no measurement there
            eff = KS.effects_mask(cu.to_model(plate, self.linear), cu.to_model(fl, self.linear),
                                  fr["obj"][y0:y1, x0:x1].astype(np.uint8) * 255, float(cu.band_at(self.cs, f)[0]),
                                  float(self.s["key_effects_thr"])) > 0
            hole[y0:y1, x0:x1] |= eff & full
        # the patch band (P14, one band system): 1 in the hole, smoothstep of the bandsys weight in the outer band
        # (out = key_soft_px shaped by the band points, outer feather = key_soft_px) – engine/cleanup3d/band.py
        b = self._band()
        patch = np.zeros(hole.shape, np.float32)
        patch[y0:y1, x0:x1] = C3B.patch_weight(hole[y0:y1, x0:x1], core, b.out_px, b.outer_feather_px,
                                               C3B.shift_points(self._band_points(f), x0, y0))[0]
        patch = np.maximum(patch, fr["mask"])
        if fr["keep"] is not None:
            patch *= 1.0 - (fr["keep"] > 0.5)
        self._patch_f, self._patch_v = f, (hole, patch)
        return hole, patch

    def _projection(self, f: int, roi: Roi) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """The keys projected onto frame f in `roi` – from pass 1's saved file when there is one, else projected now;
        kept until the fill of the same frame takes it."""
        pc = getattr(self, "_proj", None)
        if pc is not None and pc[0] == f and pc[1] == roi:
            return pc[2], pc[3], pc[4]
        if self._saved.get(f) == roi and (self.tmp / f"f_{f:04d}.npy").is_file():
            fill = np.load(self.tmp / f"f_{f:04d}.npy").astype(np.float32)
            wgt = np.load(self.tmp / f"w_{f:04d}.npy").astype(np.float32) / 255.0
            for n in ("f", "w"):
                (self.tmp / f"{n}_{f:04d}.npy").unlink(missing_ok=True)
            parts = (self.stats.get(f) or {}).get("cards", {})
            self._proj_half = True                     # the fill is float16 exactly (pass 1's file)
        else:
            fill, wgt, parts = self._project(f, roi)
            self._proj_half = False
        self._proj = (f, roi, fill, wgt, parts)
        return fill, wgt, parts

    def patch_mask(self, f: int) -> Optional[np.ndarray]:
        """The Cleaner's mask of frame f for the key source (`_patch`), None without objects to remove."""
        fr = self._frame(f)
        return None if fr is None else self._patch(f, fr)[1]

    def _project(self, f: int, roi: Roi) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        parts: Dict[str, Any] = {}
        keep = (lambda kf: self._keep(kf)) if self.cs["holdout"] else None
        fill, wgt = project_frame(self.scene.geometry, self.scene.cards, self.scene.keys_by_card, f,
                                  self.scene.cache, keep, settings=self.s, roi=roi, cache=self.scene.cache, parts=parts)
        self.seconds["project"] += time.time() - t0
        return fill, wgt, parts

    def _usable(self, fr: Dict[str, Any], wgt: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = fr["roi"]
        us = wgt >= cu.KEY_FULL
        if fr["keep"] is not None:
            us &= ~(cv2.dilate((fr["keep"][y0:y1, x0:x1] > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        return us

    # -- pass 1 (motion blur "auto")
    def prepare(self, frames: Sequence[int]) -> None:
        if self.s["motion_blur"] != "auto":
            return
        frames = list(frames)
        self.tmp.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            self.cancel_check()
            self.progress(f"Projekcja 3D: rzut kluczy i pomiar rozmycia F{f}", i, len(frames))
            fr = self._frame(f)
            if fr is None:
                continue
            if self.gpu:
                self._prepare_gpu(f, fr)
                continue
            x0, y0, x1, y1 = fr["roi"]
            fill, wgt, parts = self._project(f, fr["roi"])
            if not (wgt > 0).any():
                continue
            t0 = time.time()
            dx, dy, sky = motion_field(self.scene.geometry, f, fr["roi"])
            plate = self.plate(f)[y0:y1, x0:x1].astype(np.float32)
            n = self.mb.measure(f, fill, plate, fr["obj"][y0:y1, x0:x1], self._usable(fr, wgt) & ~fr["obj"][y0:y1, x0:x1],
                                dx, dy, sky, self.linear)
            self.seconds["blur"] += time.time() - t0
            np.save(self.tmp / f"f_{f:04d}.npy", fill.astype(np.float16))
            np.save(self.tmp / f"w_{f:04d}.npy", np.round(wgt * 255).astype(np.uint8))
            self._saved[f] = fr["roi"]
            self.stats[f] = {"cards": parts, "mb_ring_px": n}
        self._lengths = self.mb.solve(frames)

    def _prepare_gpu(self, f: int, fr: Dict[str, Any]) -> None:
        """Pass 1 of one frame in the geometry worker: project (pass 1's files written there), motion field, blur
        measure of every candidate – the plate region and two small masks go in, the errors come back."""
        geo = self.scene.geometry
        roi = fr["roi"]
        x0, y0, x1, y1 = roi
        t0 = time.time()
        parts: Dict[str, Any] = {}
        keep = (lambda kf: self._keep(kf)) if self.cs["holdout"] else None
        fp, wp = self.tmp / f"f_{f:04d}.npy", self.tmp / f"w_{f:04d}.npy"
        res: Dict[str, Any] = {"save": {"fill": str(fp), "wgt": str(wp)}}
        _project_gpu(geo, self.scene.cards, self.scene.keys_by_card, f, self.scene.cache, keep, self.s, roi, parts,
                     resident=res)
        self.seconds["project"] += time.time() - t0
        if not res.get("any"):
            for p in (fp, wp):
                p.unlink(missing_ok=True)
            return
        t0 = time.time()
        _motion_gpu(geo, f, roi)
        kp = fr["keep"]
        n = self.mb.measure_gpu(geo, f, roi, self.plate(f)[y0:y1, x0:x1].astype(np.float32), fr["obj"][y0:y1, x0:x1],
                                None if kp is None else kp[y0:y1, x0:x1] > 0.02)
        self.seconds["blur"] += time.time() - t0
        self._saved[f] = roi
        self.stats[f] = {"cards": parts, "mb_ring_px": n}

    def _levels_gpu(self, f: int, fr: Dict[str, Any], resident: Tuple[bool, bool]) -> Callable[..., Any]:
        """The grain measurement's heavy part (grain.levels_cpu) in the geometry worker (op grain_levels) on the fill /
        plate op keyfill left there (`resident` = (fill, plate)); what is not resident is sent."""
        from engine.cleanup3d.geom import _jsonable_view
        geo = self.scene.geometry
        roi = fr["roi"]

        def levels(p: np.ndarray, fl: np.ndarray, reg: np.ndarray, sel: np.ndarray, idx: Optional[np.ndarray]):
            tmp: Dict[str, str] = {}

            def put(tag: str, a: np.ndarray) -> str:
                tmp[tag] = geo._tmp(tag)
                np.save(tmp[tag], a)
                return tmp[tag]
            req = {"op": "grain_levels", "frame": _jsonable_view(geo.frame_view(f)), "roi": list(roi),
                   "fill": None if resident[0] else put("glf", np.asarray(fl, np.float32)),
                   "plate": None if resident[1] else put("glp", np.asarray(p, np.float32)),
                   "reg": put("glr", np.asarray(reg).astype(np.uint8)), "sel": put("gls", np.asarray(sel).astype(np.uint8)),
                   "idx": None if idx is None else put("gli", np.asarray(idx).astype(np.uint8)),
                   "sigmas": list(GR.SIGMAS), "cblock": GR.CBLOCK, "bins": GR.LUM_BINS}
            try:
                resp = geo._worker().call(req)
            finally:
                for q in tmp.values():
                    try:
                        os.remove(q)
                    except OSError:
                        pass
            arr = lambda v: None if v is None else np.asarray(v, np.float32)
            return (arr(resp["bp"]), arr(resp["bf"]), None if resp["binp"] is None else [arr(v) for v in resp["binp"]],
                    None if resp["binf"] is None else [arr(v) for v in resp["binf"]])
        return levels

    def _blur_tone_gpu(self, f: int, fr: Dict[str, Any], fill: np.ndarray, wgt: np.ndarray, hole: np.ndarray,
                       patch: np.ndarray, lengths: Optional[Tuple[float, float]], st: Dict[str, Any]) -> np.ndarray:
        """Pass 2's blur (lengths, along the motion field) and tone match of one frame in the geometry worker (op
        keyfill): the projection goes as float16 when it came from pass 1's file (exact), else float32."""
        from engine.cleanup3d.geom import _jsonable_view
        geo = self.scene.geometry
        roi = fr["roi"]
        x0, y0, x1, y1 = roi
        tmp: Dict[str, str] = {}

        def put(tag: str, a: np.ndarray) -> str:
            tmp[tag] = geo._tmp(tag)
            np.save(tmp[tag], a)
            return tmp[tag]
        req: Dict[str, Any] = {"op": "keyfill", "frame": _jsonable_view(geo.frame_view(f)), "roi": list(roi),
                               "fill": put("kff", fill.astype(np.float16 if self._proj_half else np.float32)),
                               "lengths": None, "tone": None, "out": geo._tmp("kfo")}
        tmp["out"] = req["out"]
        t0 = time.time()
        try:
            if lengths is not None:
                _motion_gpu(geo, f, roi)
                req["lengths"] = [float(lengths[0]), float(lengths[1])]
                req["sel"] = put("kfs", (patch[y0:y1, x0:x1] > 0).astype(np.uint8))
            if self.s["tone"]:
                req["tone"] = {"plate": put("kfp", self.plate(f)[y0:y1, x0:x1].astype(np.float32)),
                               "hole": put("kfh", hole[y0:y1, x0:x1].astype(np.uint8)),
                               "usable": put("kfu", self._usable(fr, wgt).astype(np.uint8)),
                               "ring_px": float(self.s["tone_ring_px"])}
            resp = geo._worker().call(req)
            out = np.load(req["out"])
        finally:
            for p in tmp.values():
                try:
                    os.remove(p)
                except OSError:
                    pass
        if "motion_px" in resp:
            st["motion_px"] = resp["motion_px"]
        self.seconds["blur_tone_gpu"] = self.seconds.get("blur_tone_gpu", 0.0) + time.time() - t0
        return out

    # -- pass 2: the Cleaner's key_fill
    def __call__(self, f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        self.cancel_check()
        fr = self._frame(f)
        if fr is None:
            return None
        roi = fr["roi"]
        x0, y0, x1, y1 = roi
        st = self.stats.setdefault(f, {})
        hole, patch = self._patch(f, fr)
        fill, wgt, parts = self._projection(f, roi)
        self._proj = None
        if parts and "cards" not in st:
            st["cards"] = parts
        if not (wgt > 0).any():
            return None
        fill, wgt = HZ_apply(self, f, fr, fill, wgt)          # P15: the projection onto the plate's horizon
        mode = self.s["motion_blur"]
        if self.gpu:                                         # blur + tone in the geometry worker (op keyfill)
            lengths = None
            if mode != "off":
                lg, ls = self._lengths.get(f, (0.0, 0.0)) if mode == "auto" else (self.s["mb_length"],) * 2
                lg, ls = lg * self.s["mb_scale"], ls * self.s["mb_scale"]
                st["mb_length"] = [round(lg, 4), round(ls, 4)]
                lengths = (lg, ls) if (lg > 0 or ls > 0) else None
            resident = (False, False)
            if lengths is not None or self.s["tone"]:
                if self.s.get("horizon_match"):              # the horizon moved the fill: it is no longer pass 1's file
                    self._proj_half = False
                fill = self._blur_tone_gpu(f, fr, fill, wgt, hole, patch, lengths, st)
                resident = (True, bool(self.s["tone"]))
            self._grain_levels = self._levels_gpu(f, fr, resident)
            mode = "off"
        if mode != "off":
            t0 = time.time()
            lg, ls = self._lengths.get(f, (0.0, 0.0)) if mode == "auto" else (self.s["mb_length"],) * 2
            lg, ls = lg * self.s["mb_scale"], ls * self.s["mb_scale"]
            st["mb_length"] = [round(lg, 4), round(ls, 4)]
            if lg > 0 or ls > 0:
                dx, dy, sky = motion_field(self.scene.geometry, f, roi)
                fill = blur_fill(fill, dx, dy, ls * sky + lg * (1.0 - sky), gpu=self.scene.geometry)
                sel = patch[y0:y1, x0:x1] > 0
                st["motion_px"] = round(float(np.hypot(dx, dy)[sel].mean()), 2) if sel.any() else 0.0
            self.seconds["blur"] += time.time() - t0
        if self.s["tone"] and not self.gpu:
            t0 = time.time()
            plate = self.plate(f)[y0:y1, x0:x1].astype(np.float32)
            fill = tone_match(fill, plate, hole[y0:y1, x0:x1], self._usable(fr, wgt), self.s["tone_ring_px"], self.linear)
            self.seconds["tone"] += time.time() - t0
        fill = GR.apply(fill, self.plate(f)[y0:y1, x0:x1].astype(np.float32), hole[y0:y1, x0:x1], self._usable(fr, wgt),
                        patch[y0:y1, x0:x1], f, self.s, self.linear, st, self.seconds,
                        levels=self._grain_levels if self.gpu else None)      # P17 grain match
        img = np.zeros((self.h, self.w, 3), np.float32)
        full = np.zeros((self.h, self.w), np.float32)
        img[y0:y1, x0:x1] = fill
        full[y0:y1, x0:x1] = wgt
        st["key_weight"] = round(float(wgt[patch[y0:y1, x0:x1] > 0].mean()), 4) if (patch[y0:y1, x0:x1] > 0).any() else 0.0
        st["patch_px"] = int((patch > 0).sum())
        st["effects_px"] = int((hole & ~(fr["mask"] > 0)).sum())
        return img, full


class _clean_first:
    """The key fill with weight 0 where the clear plate is usable (`clean(f)` -> (image, ok)), so the Cleaner, which
    tries the keys first, takes the clear plate there. The patch mask (`patch_mask`) stays the keys' own."""

    def __init__(self, fill: "KeyFill", clean: Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]):
        self.fill, self.clean = fill, clean

    def patch_mask(self, f: int) -> Optional[np.ndarray]:
        return self.fill.patch_mask(f)

    def __call__(self, f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        got = self.fill(f)
        if got is None:
            return None
        img, wgt = got
        cp = self.clean(f)
        if cp is not None:
            ok = np.asarray(cp[1], np.float32) > 0.5
            wgt = np.where(ok, 0.0, wgt).astype(np.float32)
            self.fill.stats.setdefault(f, {})["clean_first_px"] = int(ok.sum())
        return img, wgt


def _last(fn: Callable[[int], Any], n: int = 2) -> Callable[[int], Any]:
    """`fn(frame)` remembering its last `n` results (the plate / the alphas are asked for by the fill and the Cleaner)."""
    d: "OrderedDict[int, Any]" = OrderedDict()

    def get(f: int):
        if f not in d:
            d[f] = fn(f)
            while len(d) > n:
                d.popitem(last=False)
        return d[f]
    return get


def render(work: Path, frames: Sequence[int], *, plate: Callable[[int], np.ndarray],
           remove_alphas: Callable[[int], List[np.ndarray]],
           keep_alphas: Optional[Callable[[int], List[np.ndarray]]] = None,
           settings: Optional[Mapping[str, Any]] = None,
           progress: Optional[Callable[[str, int, int], None]] = None,
           clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
           auto_clean: Optional[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
           inpaint: Optional[Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], List[np.ndarray]]] = None,
           linear: bool = False, ext: Optional[str] = None, scene: Optional[Scene] = None,
           out_dir: Optional[Path] = None, cancel_check: Optional[Callable[[], None]] = None,
           keep_names: Optional[Sequence[str]] = None,
           plate_file: Optional[Callable[[int], str]] = None) -> Dict[str, Any]:
    """Render the Cleanup 3D result of `frames` into a new Cleanup version (`<work>/inpaint/v###/`, or `out_dir`).
    work = the tool's folder of the shot (`<work>/matte_desk`); plate(f) -> BGR float; remove_alphas(f) /
    keep_alphas(f) -> lists of full-frame alphas (as engine.cleanup.Cleaner); `settings`: one dict with the keys of
    DEFAULTS here and of engine.cleanup.DEFAULTS (grow_px, soft_px, holdout, fill, ...); clean / auto_clean / inpaint:
    the Cleaner's fallback sources for pixels no key covers; keep_names: the kept objects' names in keep_alphas' order
    (the kept-object check `engine.cleanup.kept_guard`, run here before the projection, shared with the plain
    Cleanup). Returns run.json (+ "version", "dir").
    Raises ValueError (Polish) when the camera, the cards or the keys are missing."""
    s = render_settings(settings)
    cs = cu.settings(dict(settings or {}))
    frames = [int(f) for f in frames]
    if not frames:
        raise ValueError("Pusty zakres klatek")
    own = scene is None
    scene = scene or Scene(work, device=s["device"])
    try:
        if not scene.has_keys():
            raise ValueError("Brak kluczy – dodaj klucz karty w kroku „Klucze”")
        missing = [f for f in frames if f not in scene.camera.R]
        if missing:
            raise ValueError(f"Kamera 3D nie ma klatek: {missing[0]}–{missing[-1]} ({len(missing)})")
        w, h = scene.size
        out = Path(out_dir) if out_dir is not None else cu.next_version(Path(work))
        ext = ext or ("exr" if linear else "png")
        plate_c, rm_c = _last(plate), _last(remove_alphas)
        keep_c = _last(keep_alphas or (lambda f: []), 4)
        # kept objects that look lost (SAM 3.1 on the removed object's outline): warned in run.json; their holdout is
        # left out only with the setting holdout_skip_lost – the same guarded loader for the fill and the Cleaner
        keep_g, kept = cu.kept_guard(frames, rm_c, keep_c, cs, keep_names, progress)
        keep_c = _last(keep_g, 4) if keep_g is not keep_c else keep_c
        tmp = out / "_fill"
        fill = KeyFill(scene, s, cs, linear, plate_c, rm_c, keep_c, tmp, progress, cancel_check)
        t0 = time.time()
        try:
            fill.prepare(frames)
            t_prep = time.time() - t0
            key_fill = fill
            if s["clean_first"] and clean is not None:
                key_fill = _clean_first(fill, clean)
            cl = cu.Cleaner(cs, w, h, linear, plate=plate_c, remove_alphas=rm_c, keep_alphas=keep_c, clean=clean,
                            auto_clean=auto_clean, inpaint=inpaint, progress=progress, key_fill=key_fill,
                            kept_check=kept, plate_file=plate_file)
            cl.report = {"c3d": {
                "schema": SCHEMA, "settings": s, "device": scene.geometry.device,
                "camera": scene.camera.source,
                "cards": [{"id": c.id, "name": c.name, "kind": c.kind, "order": c.order,
                           "keys": [{"frame": k.frame, "version": k.active} for k in scene.keys_by_card.get(c.id) or []]}
                          for c in scene.cards],
                "motion_blur": {"mode": s["motion_blur"], "shot": fill.mb.shot}, "frames": fill.stats,
                "seconds": fill.seconds, "seconds_prepare": round(t_prep, 1)}}
            rj = cl.run(frames, out, ext)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {**rj, "version": out.name, "dir": str(out).replace("\\", "/")}
    finally:
        if own:
            scene.close()


# ----------------------------------------------------------------------------------------------- previews (8-bit BGR)
def _to8(img: np.ndarray, linear: bool = False, max_side: Optional[int] = None) -> np.ndarray:
    out = np.clip(np.rint(cu.to_model(img, linear) * 255.0), 0, 255).astype(np.uint8)
    if max_side and max(out.shape[:2]) > max_side:
        k = max_side / float(max(out.shape[:2]))
        out = cv2.resize(out, (max(1, round(out.shape[1] * k)), max(1, round(out.shape[0] * k))), interpolation=cv2.INTER_AREA)
    return out


def view_key(scene: Scene, card: str, frame: int, *, warped: bool = True, linear: bool = False,
             max_side: Optional[int] = None) -> np.ndarray:
    """The key of `card` at key frame `frame` (active version; with its edge warps applied unless warped=False)."""
    c = scene.card(card)
    return _to8(scene.cache.image(scene.geometry, c, scene.key(card, frame), warped), linear, max_side)


def _shown(cards: Sequence[T.Card], only: Optional[Collection[str]]) -> List[T.Card]:
    """P16: the cards a PREVIEW shows (timeline eye / solo): `only` = card ids, None = every card. The weights stay those
    of every card (a hidden card leaves its area empty, the cards behind it do not take it over)."""
    return list(cards) if only is None else [c for c in cards if c.id in set(only)]


def _project_grid(scene: Scene, frame: int, step: int, settings: Optional[Mapping[str, Any]] = None,
                  only: Optional[Collection[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """`project_frame` on every `step`-th pixel of the frame (preview): rays of the grid only, sampled straight from the
    full-size keys. No kept objects (a preview). `only`: the cards shown (P16 filter, None = all). Returns (fill,
    weight) of the grid's size."""
    s = render_settings(settings)
    geo = scene.geometry
    w, h = scene.size
    PX, PY = np.meshgrid(np.arange(step // 2, w, step, dtype=np.float64), np.arange(step // 2, h, step, dtype=np.float64))
    gh, gw = PX.shape
    tgt = geo.frame_view(frame)
    x, y = K_.view_rays(K_.NP, tgt, PX, PY)
    weights = geo.card_weights(frame)
    acc = np.zeros((gh, gw, 3), np.float32)
    wsum = np.zeros((gh, gw), np.float32)
    hard_k = s["edge_px"] / s["edge_hard_px"]
    for card in sorted(_shown(scene.cards, only), key=lambda c: c.order):
        cw = weights[card.id][step // 2::step, step // 2::step][:gh, :gw]
        pair = key_pair(frame, scene.keys_by_card.get(card.id) or [])
        if not pair or not (cw > 1e-4).any():
            continue
        spec = geo.card_spec(card)
        num = np.zeros((gh, gw, 3), np.float32)
        den = np.zeros((gh, gw), np.float32)
        reach = np.zeros((gh, gw), np.float32)
        col: Optional[np.ndarray] = None
        for key, tw in sorted(pair, key=lambda p: -p[1]):
            backup = tw <= 0.0
            if backup:
                if col is None:
                    col = num / np.maximum(den, BACKUP)[..., None]
                    reach = np.where(den > BACKUP, reach, 0.0).astype(np.float32)
                if not ((reach < 0.999) & (cw > 1e-4)).any():
                    continue
            img = scene.cache.image(geo, card, key)
            mx, my, _ok = K_.project(K_.NP, x, y, tgt, geo.key_view(key), spec)
            mx, my = mx.astype(np.float32), my.astype(np.float32)
            cov = K_.coverage(K_.NP, mx, my, img.shape[1], img.shape[0], s["edge_px"]).astype(np.float32)
            hard = np.minimum(cov * hard_k, 1.0)
            pix = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if backup:
                wb = (1.0 - reach) * hard
                col = (col * reach[..., None] + pix * wb[..., None]) / np.maximum(reach + wb, BACKUP)[..., None]
                reach = reach + wb
                continue
            rel = cov * np.float32(tw)
            num += pix * rel[..., None]
            reach = np.where(rel > 0, np.maximum(reach, hard), reach)
            den += rel
        if col is None:
            col = num / np.maximum(den, BACKUP)[..., None]
            reach = np.where(den > BACKUP, reach, 0.0)
        a = (cw * reach).astype(np.float32)
        acc += col * a[..., None]
        wsum += a
    return acc / np.maximum(wsum, 1e-6)[..., None], np.clip(wsum, 0.0, 1.0)


BLEND_DIFF_MAX = 0.5        # the difference map of blend_inputs: 0..BLEND_DIFF_MAX -> 0..255


def blend_inputs(scene: Scene, frame: int, plate: np.ndarray, remove_alpha: np.ndarray, *, grow_px: float = 8.0,
                 key_grow_px: float = 0.0, settings: Optional[Mapping[str, Any]] = None, linear: bool = False,
                 max_side: int = 1600, only: Optional[Collection[str]] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """The inputs of the patch blend at preview size, for the live preview of C5 (the browser applies the sliders):
    (fill 8-bit BGR = the keys projected onto the frame, the plate where they do not fully cover;
     maps 8-bit BGR: B = the difference plate vs keys after one gain per channel, blurred as keys.effects_mask
     (0..BLEND_DIFF_MAX), G = the object part of the patch hole (object grown by max(grow_px, key_grow_px) – the
     C1 removal mask U the key patch's own growth), R = where the keys fully cover;
     scale = preview px per frame px). The same rule as KeyFill._patch, without the tone match and the blur; the band
     points' widths come with GET /api/cleanup3d/blend/{f}?part=band (engine/cleanup3d/band.py). `only`: the cards
     shown (P16 timeline eye / solo; None = all) – a hidden card's area shows the plate."""
    w, h = scene.size
    step = max(1, int(np.ceil(max(w, h) / float(max_side))))
    pl = cu.to_model(np.asarray(plate, np.float32)[..., :3], linear)
    if step > 1:
        fill, wgt = _project_grid(scene, frame, step, settings, only)
        fill = cu.to_model(fill, linear)
        gh, gw = wgt.shape
        pl = cv2.resize(pl, (gw, gh), interpolation=cv2.INTER_AREA)
    else:
        fill, wgt = project_frame(scene.geometry, _shown(scene.cards, only), scene.keys_by_card, frame, scene.cache,
                                  settings=settings, cache=scene.cache)
        fill = cu.to_model(fill, linear)
        gh, gw = wgt.shape
    fill, wgt = HZ_blend(scene, frame, fill, wgt, pl, remove_alpha, settings, step)   # P15: horizon match
    scale = gw / float(w)
    from engine.cleanup3d import band as C3B
    obj = C3B.preview_object(remove_alpha, (gw, gh), scale, grow_px, key_grow_px).astype(np.uint8)
    full = wgt >= cu.KEY_FULL
    fl = np.where(full[..., None], fill, pl)
    out = (obj == 0) & full
    if out.sum() > 100:
        gain = np.array([float(np.median(pl[..., c][out])) / max(float(np.median(fl[..., c][out])), 1e-4)
                         for c in range(3)], np.float32)
    else:
        gain = np.ones(3, np.float32)
    d = np.abs(fl * gain - pl).max(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), max(1.0, 3.0 * gw / 960.0))
    maps = np.dstack([np.clip(d / BLEND_DIFF_MAX * 255.0, 0, 255), obj * 255.0, full * 255.0]).astype(np.uint8)
    return np.clip(np.rint(fl * 255.0), 0, 255).astype(np.uint8), maps, scale


def view_projection(scene: Scene, frame: int, *, plate: Optional[np.ndarray] = None,
                    keep_alpha: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                    settings: Optional[Mapping[str, Any]] = None, linear: bool = False,
                    max_side: Optional[int] = None, only: Optional[Collection[str]] = None) -> np.ndarray:
    """The keys projected onto plate frame `frame` (no blur, no tone); where no key reaches: the plate darkened to a
    quarter when given, else black. With `max_side` below the frame's size and no `keep_alpha`, computed directly on
    the preview's pixel grid (`_project_grid` – the same crossfade, cards and coverage, a fraction of the time).
    `only`: the cards shown (P16 timeline eye / solo; None = all)."""
    w, h = scene.size
    step = max(1, int(np.ceil(max(w, h) / float(max_side)))) if max_side else 1
    if step > 1 and keep_alpha is None:
        fill, wgt = _project_grid(scene, frame, step, settings, only)
        img = cu.to_model(fill, linear) * wgt[..., None]
        if plate is not None:
            base = cv2.resize(cu.to_model(np.asarray(plate, np.float32)[..., :3], linear), (wgt.shape[1], wgt.shape[0]),
                              interpolation=cv2.INTER_AREA)
            img += base * 0.25 * (1.0 - wgt[..., None])
        return _to8(img, False, None)
    fill, wgt = project_frame(scene.geometry, _shown(scene.cards, only), scene.keys_by_card, frame, scene.cache,
                              keep_alpha, settings=settings, cache=scene.cache)
    img = cu.to_model(fill, linear) * wgt[..., None]
    if plate is not None:
        img += cu.to_model(np.asarray(plate, np.float32)[..., :3], linear) * 0.25 * (1.0 - wgt[..., None])
    return _to8(img, False, max_side)


def _hex_bgr(color: str) -> Tuple[float, float, float]:
    c = (color or "").lstrip("#")
    try:
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        r, g, b = 0.29, 0.88, 1.0
    return b, g, r


def _grid_lines(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """0..1 anti-aliased lines of the integer grid of (u, v); fades where a cell is smaller than ~3 px."""
    out = np.zeros(u.shape, np.float32)
    for c in (u, v):
        gy, gx = np.gradient(c)
        fw = np.maximum(np.hypot(gx, gy), 1e-9)
        d = np.abs(c - np.rint(c)) / fw                       # px to the nearest line
        line = np.clip(1.25 - d, 0.0, 1.0) * np.clip((0.33 - fw) / 0.2, 0.0, 1.0)
        out = np.maximum(out, np.where(np.isfinite(line), line, 0.0).astype(np.float32))
    return out


def view_cards(scene: Scene, frame: int, *, plate: Optional[np.ndarray] = None, linear: bool = False,
               max_side: Optional[int] = 1280, tint: float = 0.3, only: Optional[Collection[str]] = None) -> np.ndarray:
    """The cards in plate frame `frame`: each card's extent tinted with its colour (its weight in the frame) and a
    grid lying on it (plane: squares of half the camera's height above the card; sky: every 10 degrees), over the
    plate when given. Computed directly at the preview's size. `only`: the cards shown (P16; None = all)."""
    geo = scene.geometry
    w, h = scene.size
    step = max(1, int(np.ceil(max(w, h) / float(max_side)))) if max_side else 1
    PX, PY = np.meshgrid(np.arange(step // 2, w, step, dtype=np.float64), np.arange(step // 2, h, step, dtype=np.float64))
    view = geo.frame_view(frame)
    x, y = K_.view_rays(K_.NP, view, PX, PY)
    R = np.asarray(view["R"], np.float64)
    C = -R.T @ np.asarray(view["t"], np.float64)
    wd = np.stack([x, y, np.ones_like(x)], -1) @ R            # world directions (R^T r per pixel)
    weights = geo.card_weights(frame)
    if plate is not None:
        base = cu.to_model(np.asarray(plate, np.float32)[..., :3], linear)
        base = cv2.resize(base, (PX.shape[1], PX.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        base = np.full(PX.shape + (3,), 0.12, np.float32)
    out = base.astype(np.float32)
    mid = scene.camera.centre(scene.camera.frames[len(scene.camera.frames) // 2])
    for card in _shown(geo.cards, only):
        cw = weights[card.id][step // 2::step, step // 2::step][:PX.shape[0], :PX.shape[1]]
        if not (cw > 0).any():
            continue
        col = np.array(_hex_bgr(card.color), np.float32)
        spec = geo.card_spec(card)
        with np.errstate(divide="ignore", invalid="ignore"):
            if spec["kind"] == "sky":
                u = np.degrees(np.arctan2(wd[..., 0], wd[..., 2])) / 10.0
                v = np.degrees(np.arctan2(-wd[..., 1], np.hypot(wd[..., 0], wd[..., 2]))) / 10.0
            else:
                n, d = np.asarray(spec["n"], np.float64), float(spec["d"])
                zp = -(n @ C + d) / (wd @ n)
                P = C + zp[..., None] * wd
                a = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
                a /= np.linalg.norm(a)
                b = np.cross(n, a)
                cell = max(1e-9, abs(float(n @ mid + d)) / 2.0)
                u, v = (P @ a) / cell, (P @ b) / cell
            lines = _grid_lines(u, v)
        k = (cw * tint)[..., None]
        out = out * (1.0 - k) + col * k
        k = (cw * lines * 0.9)[..., None]
        out = out * (1.0 - k) + col * k
    return _to8(out, False, None)
