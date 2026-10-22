# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/corestab.py - temporal stabilisation of the binary SAM 3.1 core before the edge mode / zone trimaps
(user decision 2026-10-13 after the CLIP2_A flicker split: the core edge jumps ~8 % of the 4 px rim per frame even after
motion compensation, the base MatAnyone2 edge inherits it, the zone trimap too).

`stabilize_core(cores, gray, radius=2, band_px=8, keys=())`: for frame i the cores of frames i±1..i±radius are warped
INTO frame i by DIS optical flow measured on the display gray (crop around the union of the cores + margin, so a 1080p
shot costs ~15 ms per flow), a neighbour failing the forward/backward check (> 1 px) votes with the centre value, and
the per-pixel MAJORITY of the aligned window replaces the core – only inside `band_px` of the core edge (the interior
and everything far from the edge stay bit-identical) and never in frames listed in `keys` (user keys stay hard).
Pure function on dicts of bool masks; the caller decides where it applies (pipeline._final_core / apply_zone when the
object option `core_stabilize` is on; off = the old path untouched).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def consistent_soft(p8: np.ndarray, core: np.ndarray) -> np.ndarray:
    """8-bit probability map made threshold-consistent with a binary core (`p >= 128` <=> core): where the core and the
    map disagree (zones, shrink/grow, front-object subtraction, keys) the core wins with a hard 255 / 0."""
    p = np.asarray(p8, np.uint8).copy()
    c = core.astype(bool)
    p[c & (p < 128)] = 255
    p[~c & (p >= 128)] = 0
    return p


def _crop_box(cores: Dict[int, np.ndarray], margin: int) -> Tuple[int, int, int, int]:
    H, W = next(iter(cores.values())).shape
    u = np.zeros((H, W), bool)
    for m in cores.values():
        u |= m
    if not u.any():
        return 0, 0, W, H
    ys, xs = np.where(u)
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(W, int(xs.max()) + 1 + margin), min(H, int(ys.max()) + 1 + margin))


def stabilize_core(cores: Dict[int, np.ndarray], gray: Callable[[int], np.ndarray], radius: int = 2, band_px: int = 8,
                   keys: Iterable[int] = (), occl_px: float = 1.0, margin: int = 64,
                   preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                   progress: Optional[Callable[[int, int], None]] = None,
                   probs: Optional[Dict[int, np.ndarray]] = None, flip_margin: float = 0.0,
                   protect_thin: bool = True,
                   probs_out: Optional[Dict[int, np.ndarray]] = None,
                   only: Optional[Iterable[int]] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    """Returns (stabilised cores, stats {changed_px_per_frame, frames, mode}).

    Two guards against eroding what is not jitter (synthetic puppet 2026-10-14: a 2-4 px antenna vanished entirely,
    10-17 % of the core per frame removed when the flow is imprecise): `protect_thin` – parts thinner than `band_px`
    (components >= band_px^2 px of what the opening removes) and their `band_px` surroundings keep the frame's own
    value (a thin part lies entirely inside the band and cannot be voted on safely); `flip_margin` – optional: a pixel
    changes only when the aligned average is < 0.5 - margin / > 0.5 + margin. Sweep 2026-10-14 (synthetic jitter,
    puppet antenna, CLIP2_A): thin protection alone keeps the full gain (CLIP2_A rim 0.133 -> 0.096, antenna 0 % lost);
    a margin of 0.1-0.2 halves the jitter repair and most of the gain -> default margin 0.

    `probs` (frame -> SAM 3.1 soft probabilities, uint8 0..255 or float 0..1, full frame): PROBABILITY mode – the aligned
    window averages the probabilities with per-pixel confidence weights |2p - 1| (+0.05 so the centre always counts;
    an uncertain 0.5 barely votes, as the entropy weighting of inference-time temporal probability smoothing) and the
    core becomes `average > 0.5` inside the band. Frames without a map fall back to their binary core (p = 0 / 1).

    `probs_out` (dict, filled in place; probability mode only): the STABILISED probability map of every frame that has a
    soft map and is not a key – the aligned average inside the band, the frame's own p elsewhere – as uint8, made
    consistent with the returned binary core (consistent_soft). The soft start of the edge band (MatAnyone2 CoreCombine
    core_mode=soft, ViTMatte zone trimap; user decision 2026-10-14).

    `only` (previews of one frame, 2026-10-22): stabilise just these frames – the others come back as they are (the
    neighbours still vote; the result of an `only` frame is the same as without `only`); flows only for its window."""
    frames = sorted(cores)
    n = len(frames)
    if n < 3 or radius < 1:
        if probs_out is not None and probs:
            for f in frames:
                if f not in set(int(k) for k in keys) and probs.get(f) is not None:
                    p = probs[f]
                    p8 = p if p.dtype == np.uint8 else (np.clip(p, 0, 1) * 255.0 + 0.5).astype(np.uint8)
                    probs_out[f] = consistent_soft(p8, cores[f])
        return {f: cores[f].copy() for f in frames}, {"frames": n, "changed_px_per_frame": 0.0,
                                                      "mode": "prob" if probs else "binary"}
    use_prob = bool(probs)

    def prob_of(f: int, box) -> np.ndarray:
        """Probability crop of frame f (float32 0..1): the soft map, or the binary core where none."""
        y0, y1, x0, x1 = box
        p = probs.get(f) if use_prob else None
        if p is None:
            return cores[f][y0:y1, x0:x1].astype(np.float32)
        p = p[y0:y1, x0:x1]
        return (p.astype(np.float32) / 255.0) if p.dtype == np.uint8 else np.clip(p, 0, 1).astype(np.float32)
    keys = set(int(k) for k in keys)
    only_set = None if only is None else set(int(f) for f in only)
    x0, y0, x1, y1 = _crop_box(cores, margin)
    h, w = y1 - y0, x1 - x0
    dis = cv2.DISOpticalFlow_create(preset)
    g8: Dict[int, np.ndarray] = {}

    def gray8(f: int) -> np.ndarray:
        if f not in g8:
            g = np.ascontiguousarray(gray(f)[y0:y1, x0:x1])          # DIS needs a contiguous image
            g8[f] = g if g.dtype == np.uint8 else cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return g8[f]

    flows: Dict[Tuple[int, int], np.ndarray] = {}

    def flow(i: int, j: int) -> np.ndarray:        # for every pixel of frame i: where it is in frame j
        k = (i, j)
        if k not in flows:
            flows[k] = dis.calc(gray8(frames[i]), gray8(frames[j]), None)
        return flows[k]

    if only_set is not None:   # a preview: the few flows of its window at once on several cores (cv2 releases the GIL)
        pairs = sorted({pq for i, f in enumerate(frames) if f in only_set and f not in keys
                        for d in range(-radius, radius + 1) if d and 0 <= i + d < n for pq in ((i, i + d), (i + d, i))})
        if len(pairs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            for i in sorted({q for pq in pairs for q in pq}):
                gray8(frames[i])

            def calc(pq):
                return pq, cv2.DISOpticalFlow_create(preset).calc(g8[frames[pq[0]]], g8[frames[pq[1]]], None)
            with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as ex:
                flows.update(dict(ex.map(calc, pairs)))
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out: Dict[int, np.ndarray] = {}
    changed = 0
    for i, f in enumerate(frames):
        if progress is not None:
            progress(i + 1, n)
        c = cores[f]
        if f in keys or (only_set is not None and f not in only_set):
            out[f] = c.copy()
            continue
        cc = c[y0:y1, x0:x1].astype(np.float32)
        box = (y0, y1, x0, x1)
        pc = prob_of(f, box) if use_prob else cc
        conf = lambda p: (np.abs(2.0 * p - 1.0) + 0.05) if use_prob else np.ones_like(p)
        w0 = conf(pc)
        vsum, vcnt = pc * w0, w0.copy()
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            j = i + d
            if j < 0 or j >= n:
                continue                          # clamped window: fewer voters at the ends
            fij, fji = flow(i, j), flow(j, i)
            mx, my = gx + fij[..., 0], gy + fij[..., 1]
            src = prob_of(frames[j], box) if use_prob else cores[frames[j]][y0:y1, x0:x1].astype(np.float32)
            warped = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            back = cv2.remap(fji, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            valid = (np.linalg.norm(fij + back, axis=2) <= occl_px).astype(np.float32)   # unreliable flow abstains
            wj = valid * conf(warped)
            vsum += warped * wj
            vcnt += wj
        avg = vsum / vcnt
        own = cc > 0.5
        # flip only with a clear majority: remove below 0.5 - margin, add above 0.5 + margin, else keep the frame's value
        maj = np.where(own, avg >= 0.5 - flip_margin, avg > 0.5 + flip_margin)
        # only the band around the edge of the frame's own core (interior / far background untouched)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
        c8 = cc.astype(np.uint8)
        band = cv2.dilate(c8, k).astype(bool) & ~cv2.erode(c8, k).astype(bool)
        if protect_thin:
            # thin parts = what the opening removes, but only components of at least band_px^2 px (an antenna / strand
            # strip); the few-pixel residues at corners and notches are exactly the jitter the vote should repair
            resid = (own & ~cv2.morphologyEx(c8, cv2.MORPH_OPEN, k).astype(bool)).astype(np.uint8)
            if resid.any():
                n_lab, lab, st_, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
                keep = np.zeros(n_lab, bool)
                keep[1:] = st_[1:, cv2.CC_STAT_AREA] >= band_px * band_px
                thin = keep[lab]
                if thin.any():
                    band &= ~cv2.dilate(thin.astype(np.uint8), k).astype(bool)
        new = c.copy()
        sub = np.where(band, maj, own)
        changed += int((sub != (cc > 0.5)).sum())
        new[y0:y1, x0:x1] = sub
        out[f] = new
        if probs_out is not None and use_prob and probs.get(f) is not None:
            p = probs[f]
            p8 = (p.copy() if p.dtype == np.uint8 else (np.clip(p, 0, 1) * 255.0 + 0.5).astype(np.uint8))
            p8[y0:y1, x0:x1] = np.where(band, (np.clip(avg, 0, 1) * 255.0 + 0.5).astype(np.uint8), p8[y0:y1, x0:x1])
            probs_out[f] = consistent_soft(p8, new)
        # drop flows no longer needed (window moved on)
        for old in [k_ for k_ in flows if min(k_) < i - radius]:
            flows.pop(old, None)
        for old in [k_ for k_ in g8 if frames.index(k_) < i - radius]:
            g8.pop(old, None)
    return out, {"frames": n, "changed_px_per_frame": round(changed / max(1, n - len(keys)), 1),
                 "box": [x0, y0, x1, y1], "radius": radius, "band_px": band_px,
                 "mode": "prob" if use_prob else "binary",
                 "prob_frames": sum(1 for f in frames if use_prob and probs.get(f) is not None)}
