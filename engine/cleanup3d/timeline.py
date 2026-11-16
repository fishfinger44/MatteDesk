# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – the layered card timeline (package P16, docs/PLAN_CLEANUP_3D.md „Następna fala”, user 2026-11-10, like
the After Effects timeline: one row per card with its keys, a coverage bar, eye / solo).

The eye (Card.off) and solo (Card.solo) are stored on the card and work like After Effects (types.active_cards): a card
turned off, or – when any card is soloed – a card without solo, is out of render.Scene, so out of the render, the
previews, the coverage below and the Nuke export (the render signature changes, the result shows ↻).

Two things the rows need from the engine:

* `coverage(scene, frames)` – per card, per frame, on a coarse grid of the frame (long side COVER_SIDE px): `share` =
  the mean weight the card really fills in the frame (card weight x how far its keys reach) and `miss` = the part of
  what the card shows that its keys do NOT fully cover (there the render falls back to the next source: the clear
  plate, the auto clean plate, the inpainter). The same rule as render._project_grid (key_pair crossfade, the backup
  key, coverage with edge_px / edge_hard_px), without reading a single key image – only the camera rays.
* `move_key(...)` – a key dragged to another frame. Honest and simple: a key holds the plate of ITS frame plus the fill
  in the removed area, so the moved key is the PLATE OF THE TARGET FRAME (v001, source „plate”) and, when the source
  key carries a fill, a second version = that plate + the source key's active version (with its edge warps) projected
  through the 3D camera onto the target frame, only in the target's removal mask (grown, soft edge, tone-matched, the
  object's effects – keys.patch_on_plate, the rule of keys.rebase_on_plate) and only where the projection fully covers
  (source „move:F<src>:<version>”). Nothing is generated anew. The old key is deleted from disk for good
  (keys.delete_key, user decision 2026-11-13) only after the new one is written. Refused (ValueError, Polish): a master
  key (own canvas, stage III – rebuild it instead), the target already has a key of the card, a frame without the
  camera, a card turned off or outside the solo.
  The source key's areas go with it and its edge warps stay in the card's warps of that frame: they are in the source
  key's pixels and do not apply to another frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d import keys as KS
from engine.cleanup3d import store
from engine.cleanup3d import types as T

COVER_SIDE = 192          # long side of the coverage grid (px): a 4K frame -> every 20th pixel
FULL = 0.999              # reach at or above = the keys fully cover the pixel (render: KEY_FULL)
MOVE_SOURCE = "move"


# ----------------------------------------------------------------------------------------------- coverage
def _grid(w: int, h: int, side: int):
    step = max(1, int(np.ceil(max(w, h) / float(max(16, int(side))))))
    PX, PY = np.meshgrid(np.arange(step // 2, w, step, dtype=np.float64), np.arange(step // 2, h, step, dtype=np.float64))
    return step, PX, PY


def _specs(geo, step: int) -> List[Dict[str, Any]]:
    """The card specs of geom.CardGeometry.card_weights (CPU path) for a grid of `step` px: the seam in grid px."""
    out = []
    for c in geo.cards:
        ext = geo._extent(c)
        if ext is not None:
            ext = {**ext, "canvas_arr": ext["canvas_u8"].astype(np.float64) / 255.0}
        out.append({**geo.card_spec(c), "seam_px": float(c.seam_px) / float(step), "ext": ext})
    return out


def coverage(scene, frames: Sequence[int], *, side: int = COVER_SIDE,
             settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"frames": [...frames the camera has], "step": grid px, "cards": {card id: {"share": [...], "miss": [...]}}} –
    one number per frame (rounded to 4 places). `share` 0..1 of the frame; `miss` 0..1 of what the card shows (0 where
    the card shows nothing). Cards of the scene only (a card turned off or outside the solo is not in it)."""
    from engine.cleanup3d import render as RD
    s = RD.render_settings(settings)
    geo = scene.geometry
    w, h = scene.size
    step, PX, PY = _grid(w, h, side)
    hard_k = s["edge_px"] / s["edge_hard_px"]
    specs = _specs(geo, step)
    fr = [int(f) for f in frames if int(f) in scene.camera.R]
    out: Dict[str, Dict[str, List[float]]] = {c.id: {"share": [], "miss": []} for c in geo.cards}
    for f in fr:
        view = geo.frame_view(f)
        x, y = K_.view_rays(K_.NP, view, PX, PY)
        ws = K_.weights(K_.NP, view, specs, x, y)
        for card, cw in zip(geo.cards, ws):
            cw = np.asarray(cw, np.float32)
            tot = float(cw.sum())
            reach = np.zeros(cw.shape, np.float32)
            if tot > 1e-6:
                spec = geo.card_spec(card)
                den = np.zeros(cw.shape, np.float32)
                topped = False
                for key, tw in sorted(RD.key_pair(f, scene.keys_by_card.get(card.id) or []), key=lambda p: -p[1]):
                    backup = tw <= 0.0
                    if backup:
                        if not topped:
                            reach = np.where(den > RD.BACKUP, reach, 0.0).astype(np.float32)
                            topped = True
                        if not ((reach < FULL) & (cw > 1e-4)).any():
                            continue
                    src = geo.key_view(key)
                    mx, my, _ok = K_.project(K_.NP, x, y, view, src, spec)
                    cov = np.asarray(K_.coverage(K_.NP, mx, my, int(src["w"]), int(src["h"]), s["edge_px"]), np.float32)
                    hard = np.minimum(cov * hard_k, 1.0)
                    if backup:
                        reach = reach + (1.0 - reach) * hard
                        continue
                    rel = cov * np.float32(tw)
                    reach = np.where(rel > 0, np.maximum(reach, hard), reach).astype(np.float32)
                    den += rel
                if not topped:
                    reach = np.where(den > RD.BACKUP, reach, 0.0).astype(np.float32)
            share = float((cw * reach).mean())
            miss = float((cw * (reach < FULL)).sum()) / tot if tot > 1e-6 else 0.0
            out[card.id]["share"].append(round(share, 4))
            out[card.id]["miss"].append(round(miss, 4))
    return {"frames": fr, "step": step, "cards": out}


# ----------------------------------------------------------------------------------------------- key move
def _fmt(work: Path, key: T.Key) -> str:
    _img, fmt = KS.read_image(KS.version_path(work, key))
    return fmt


def move_key(scene, work: Path, card: str, src: int, dst: int, *, plate: Callable[[int], np.ndarray],
             remove_mask: Callable[[int], Optional[np.ndarray]], effects: bool = True) -> Dict[str, Any]:
    """Move the key of `card` from frame `src` to frame `dst` (see the module doc). Returns {"key": the new Key,
    "freed_bytes": what deleting the old key freed, "patched": a fill version was made, "covered": share of the target's removal
    mask the projection fully covers (None without a mask)}."""
    work = Path(work)
    src, dst = int(src), int(dst)
    cards = store.load_cards(work)
    c = next((x for x in cards if x.id == card), None)
    if c is None:
        raise ValueError(f"Brak karty {card}")
    if c.off:
        raise ValueError(f"Karta {c.name} jest wyłączona – włącz ją, żeby przenieść klucz")
    if c.id not in {x.id for x in T.active_cards(cards)}:
        raise ValueError(f"Karta {c.name} jest poza solo – dodaj jej S albo wyłącz solo, żeby przenieść klucz")
    key = store.load_key(work, card, src)
    if key is None or not key.versions or not key.active:
        raise ValueError(f"Brak klucza karty {c.name} w klatce {src}")
    if src == dst:
        raise ValueError("Klucz już jest w tej klatce")
    if key.canvas is not None:
        raise ValueError("Klucz główny (jeden klucz, własne płótno) nie przenosi się – zbuduj go od nowa w innej klatce")
    have = store.load_key(work, card, dst)
    if have is not None and have.versions:
        raise ValueError(f"Karta {c.name} ma już klucz w klatce {dst} – usuń go albo przenieś w inną klatkę")
    for f in (src, dst):
        if f not in scene.camera.R:
            raise ValueError(f"Kamera 3D nie ma klatki {f}")
    entry = next(e for e in key.versions if e.get("v") == key.active)
    fmt = _fmt(work, key)
    pl = KS._bgr3(KS._to_float(plate(dst)))
    mask = remove_mask(dst)
    patched, covered = False, None
    fill = None
    if str(entry.get("source") or "") != "plate" and mask is not None and float(np.max(mask)) > 0.02:
        from engine.cleanup3d import render as RD
        s = RD.render_settings(store.load_render_settings(work))
        sc_card = scene.card(card)
        img = scene.cache.image(scene.geometry, sc_card, key)          # with the key's edge warps, as the render reads it
        mx, my = scene.geometry.frame_to_key(dst, key, sc_card)
        kv = scene.geometry.key_view(key)
        cov = np.asarray(K_.coverage(K_.NP, mx.astype(np.float64), my.astype(np.float64), int(kv["w"]), int(kv["h"]),
                                     s["edge_px"]), np.float32)
        u = np.minimum(cov * (s["edge_px"] / s["edge_hard_px"]), 1.0)
        u *= np.asarray(scene.geometry.card_weight(sc_card, dst), np.float32)
        proj = cv2.remap(np.asarray(img, np.float32), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
        if proj.shape[:2] != pl.shape[:2]:
            raise ValueError("Rozmiar plate'u nie zgadza się z kamerą 3D")
        fill = proj * u[..., None] + pl * (1.0 - u[..., None])        # the target plate where the projection is short
        m = (np.asarray(mask, np.float32) > 0.02)
        covered = round(float((u[m] >= FULL).mean()) if m.any() else 0.0, 4)
    KS.add_version(work, card, dst, pl, "plate", fmt)
    if fill is not None:
        out = KS.patch_on_plate(pl, fill, mask, effects=effects)
        KS.add_version(work, card, dst, out, f"{MOVE_SOURCE}:F{src}:{entry['v']}", fmt,
                       extra={"moved_from": src})
        patched = True
    freed = KS.delete_key(work, card, src) or 0                       # only now: a failure above keeps the old key
    return {"key": store.load_key(work, card, dst), "freed_bytes": freed, "patched": patched, "covered": covered}


def band_keys(work: Path) -> List[int]:
    """Frames of the patch band's keys (P14, inpaint/c3d_band.json) – the „Łata” row's triangles."""
    from engine.cleanup3d import band as C3B
    b = C3B.load(work) or {}
    return sorted({int(k) for p in b.get("band_points") or [] for k in (p.get("keys") or {})})
