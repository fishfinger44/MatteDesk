# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/crop.py - Crop engine for SAM 3.1 (SPEC 5.2) – "the heart".

0. instance choice: SAM3_Detect on the full proxy letterboxed (black padding) to a square, individual masks.
1. coarse localization: SAM3_VideoTrack on the same square proxies, initial mask = chosen instance, no text,
   forward and backward from the start frame -> bbox per frame; user keys' bboxes are truth; lost object
   (empty mask / bbox jump > 30 %) -> flag; gap >= 3 empty frames -> "obiekt wraca?" flag.
2. bbox trajectory smoothing: median +-2 + zero-phase low-pass, never smaller than the raw bbox.
3. constant-crop segments: greedy, union bbox (+margin) must fit a square where the object fills >= min_fill
   (0.35) of the side in every frame; keys do NOT split segments; 8-frame overlap.
4. square crop per segment: side = max(w, h) of the union x 1.10, at least bbox + 64, multiple of 16 (32 for
   LTX-2.5); centred on the union; shifted inside the plate, mirror padding when the plate is too small.
5. tiles for objects larger than 2 x 1008 with thin parts (1008-px squares, 25 % overlap).
7. crops.json: [{seg_id, first, last, overlap_in, overlap_out, crop:[x, y, side, side], pad:[l, t, r, b],
   scale_to_1008, tiles:[...], manual}] – crop x/y = top-left of the virtual square (may be < 0 with padding).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from engine.coords import CoordTransformer, CropBox

SAM_RES = 1008
# Padding outside the plate = constant black, NOT a mirror: a mirror duplicates an object that touches the frame
# edge (upside-down head / body copy) and SAM3 tracks or detects the copy (SH030 2026-10-14: coarse track lost the
# person after ~25 frames, crops cut her off).
PAD_BORDER = cv2.BORDER_CONSTANT


def pad_border(img: np.ndarray, t: int, b: int, l: int, r: int) -> np.ndarray:
    return cv2.copyMakeBorder(img, t, b, l, r, PAD_BORDER, value=0)
CROPS_SCHEMA = "matte_desk/crops@1.0"


@dataclass
class CropSegment:
    seg_id: str
    first: int
    last: int
    overlap_in: int
    overlap_out: int
    crop: List[int]                 # [x, y, side, side] of the virtual square in plate px (x, y may be < 0)
    pad: List[int]                  # [l, t, r, b] mirror padding (px of the square outside the plate)
    scale_to_1008: float
    tiles: List[List[int]] = field(default_factory=list)  # [x, y, side, side] in crop px
    manual: bool = False
    fill_min: float = 0.0           # min over frames of max(bbox w, h) / side
    flags: List[str] = field(default_factory=list)

    @property
    def side(self) -> int:
        return int(self.crop[2])

    def box(self) -> CropBox:
        x, y, s, _ = self.crop
        l, t, r, b = self.pad
        return CropBox(x=float(x + l), y=float(y + t), w=float(s - l - r), h=float(s - t - b),
                       pad_l=float(l), pad_t=float(t), pad_r=float(r), pad_b=float(b))

    def frames(self) -> range:
        return range(self.first, self.last + 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CropSegment":
        return cls(**d)


# ----------------------------------------------------------------------------- geometry helpers
def mask_bbox(m: np.ndarray, thr: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(m > thr)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def letterbox_square(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad to a square with black (no anisotropy, SPEC 5.2 pkt 0). Returns (square, (offset_x, offset_y))."""
    h, w = img.shape[:2]
    s = max(h, w)
    l, t = (s - w) // 2, (s - h) // 2
    sq = pad_border(img, t, s - h - t, l, s - w - l)
    return sq, (l, t)


def extract_crop(img: np.ndarray, seg: CropSegment) -> np.ndarray:
    """Square crop of a full-resolution plate frame, black padding outside the plate."""
    x, y, s, _ = seg.crop
    l, t, r, b = seg.pad
    patch = img[y + t:y + s - b, x + l:x + s - r]
    if l or t or r or b:
        patch = pad_border(patch, t, b, l, r)
    return patch


def paste_crop(m: np.ndarray, seg: CropSegment, plate_w: int, plate_h: int) -> np.ndarray:
    """Crop-sized result -> full frame (padding removed). `m` may be at any square size (resized to side)."""
    x, y, s, _ = seg.crop
    l, t, r, b = seg.pad
    if m.shape[0] != s:
        m = cv2.resize(m.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((plate_h, plate_w), np.float32)
    full[y + t:y + s - b, x + l:x + s - r] = m[t:s - b, l:s - r]
    return full


def square_crop(cx: float, cy: float, need: float, plate_w: int, plate_h: int, pad_ratio: float = 1.10,
                margin: float = 64.0, snap: int = 16) -> Tuple[List[int], List[int]]:
    box = CoordTransformer(plate_w, plate_h).create_square_crop(cx, cy, need, pad_ratio, margin, snap)
    side = int(round(box.side))
    x = int(round(box.x - box.pad_l))
    y = int(round(box.y - box.pad_t))
    return [x, y, side, side], [int(box.pad_l), int(box.pad_t), int(box.pad_r), int(box.pad_b)]


# ----------------------------------------------------------------------------- 1. coarse localization
@dataclass
class CoarseTrack:
    bboxes: List[Optional[Tuple[float, float, float, float]]]   # plate px per frame index
    flags: Dict[int, str]                                     # frame index -> reason
    gaps: List[Tuple[int, int]]                               # index ranges with an empty mask >= 3 frames


def coarse_localize(runner: Any, proxies: Sequence[np.ndarray], start_idx: int, init_mask_proxy: np.ndarray,
                    proxy_scale: float, key_masks: Optional[Dict[int, np.ndarray]] = None,
                    frame_numbers: Optional[Sequence[int]] = None) -> CoarseTrack:
    """SAM3_VideoTrack on letterboxed proxies, forward and backward from `start_idx` (no text)."""
    from engine.comfy_io import LazySeq, progress_prefix
    # letterboxed on access (a 1000-frame shot as a list of 1920 x 1920 float frames would need ~47 GB RAM)
    sq = LazySeq(lambda i: letterbox_square(proxies[i])[0], range(len(proxies)))
    p0 = proxies[0]
    _, (ox, oy) = letterbox_square(p0)
    h, w = p0.shape[:2]
    del p0
    init_sq, _ = letterbox_square(init_mask_proxy.astype(np.float32))
    n = len(proxies)

    def bbox_of(i: int, m: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        m = m[oy:oy + h, ox:ox + w]
        if key_masks and i in key_masks:  # user keys are the truth in their frames
            m = key_masks[i].astype(np.float32)
        bb = mask_bbox(m)
        return None if bb is None else tuple(v / proxy_scale for v in bb)

    # only the bbox of each mask is kept (not the masks themselves)
    bboxes: List[Optional[Tuple[float, float, float, float]]] = [None] * n
    fn = list(frame_numbers) if frame_numbers is not None else list(range(n))
    # the crop pass always covers the whole shot (stable crops keep the interval cache): say so in the progress
    with progress_prefix(f"Kadrowanie całego ujęcia F{fn[start_idx]}→F{fn[-1]}"):
        fwd = runner.track(sq[start_idx:], init_sq)
    for k, m in enumerate(fwd):
        bboxes[start_idx + k] = bbox_of(start_idx + k, m)
    if start_idx > 0:
        with progress_prefix(f"Kadrowanie całego ujęcia F{fn[start_idx]}→F{fn[0]}"):
            bwd = runner.track(sq[start_idx::-1], init_sq)
        for k, m in enumerate(bwd):
            if k:
                bboxes[start_idx - k] = bbox_of(start_idx - k, m)
    flags: Dict[int, str] = {}
    for i in range(1, n):
        a, b = bboxes[i - 1], bboxes[i]
        if b is None:
            flags[i] = "pusta maska – kliknij obiekt w tej klatce"
        elif a is not None:
            area = lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])
            jump = max(abs(b[k] - a[k]) for k in range(4)) / max(a[2] - a[0], a[3] - a[1], 1.0)
            if abs(area(b) / max(area(a), 1.0) - 1.0) > 0.3 or jump > 0.3:
                flags[i] = "skok bbox > 30 % – sprawdź, czy to ten sam obiekt"
    gaps, run = [], None
    for i, bb in enumerate(bboxes + [(0, 0, 1, 1)]):
        if bb is None and run is None:
            run = i
        elif bb is not None and run is not None:
            if i - run >= 3:
                gaps.append((run, i - 1))
                if i < n:
                    flags[i] = "obiekt wraca? kliknij go"
            run = None
    return CoarseTrack(bboxes=bboxes, flags=flags, gaps=gaps)


# ----------------------------------------------------------------------------- 2. smoothing
def smooth_bboxes(bboxes: Sequence[Optional[Sequence[float]]]) -> List[Optional[Tuple[float, float, float, float]]]:
    """Median +-2 then zero-phase Butterworth low-pass; result always contains the raw bbox (never cuts)."""
    idx = [i for i, b in enumerate(bboxes) if b is not None]
    if not idx:
        return [None] * len(bboxes)
    arr = np.array([bboxes[i] for i in idx], np.float64)
    sm = median_filter(arr, size=(5, 1), mode="nearest")
    if len(idx) > 12:
        b, a = butter(2, 0.15)
        sm = filtfilt(b, a, sm, axis=0, padlen=min(9, len(idx) - 1))
    sm[:, 0] = np.minimum(sm[:, 0], arr[:, 0])
    sm[:, 1] = np.minimum(sm[:, 1], arr[:, 1])
    sm[:, 2] = np.maximum(sm[:, 2], arr[:, 2])
    sm[:, 3] = np.maximum(sm[:, 3], arr[:, 3])
    out: List[Optional[Tuple[float, float, float, float]]] = [None] * len(bboxes)
    for k, i in enumerate(idx):
        out[i] = tuple(float(v) for v in sm[k])
    return out


# ----------------------------------------------------------------------------- 3-5. segments
def _union(bbs: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    a = np.array(bbs)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())


def _tiles(side: int, tile: int = SAM_RES, overlap: float = 0.25) -> List[List[int]]:
    if side <= tile:
        return []
    n = int(math.ceil((side - tile) / (tile * (1 - overlap)))) + 1
    pos = np.linspace(0, side - tile, n).round().astype(int)
    return [[int(x), int(y), tile, tile] for y in pos for x in pos]


def plan_segments(bboxes: Sequence[Optional[Sequence[float]]], first_frame: int, plate_w: int, plate_h: int,
                  cfg: Optional[Dict[str, Any]] = None, thin_parts: bool = False, for_ltx: bool = False
                  ) -> List[CropSegment]:
    """Greedy constant-crop segments over frames that have a bbox (plate px)."""
    c = {"min_fill": 0.35, "pad_ratio": 1.10, "min_margin_px": 64, "snap_multiple": 16, "ltx_snap_multiple": 32,
         "overlap_frames": 8}
    c.update(cfg or {})
    snap = c["ltx_snap_multiple"] if for_ltx else c["snap_multiple"]
    ov = int(c["overlap_frames"])
    valid = [i for i, b in enumerate(bboxes) if b is not None]
    if not valid:
        return []
    lo, hi = valid[0], valid[-1]
    bb = list(bboxes)
    for i in range(lo, hi + 1):  # inside the presence range, gaps take the neighbours' union
        if bb[i] is None:
            prev = next(bb[j] for j in range(i, lo - 1, -1) if bb[j] is not None)
            nxt = next(bb[j] for j in range(i, hi + 1) if bb[j] is not None)
            bb[i] = _union([prev, nxt])

    def need_side(u):
        m = max(u[2] - u[0], u[3] - u[1])
        side = max(m * c["pad_ratio"], m + c["min_margin_px"])
        return int(math.ceil(side / snap) * snap)

    def size(b):
        return max(b[2] - b[0], b[3] - b[1])

    segs: List[CropSegment] = []
    s = lo
    while True:
        e = s
        s_max_fill = size(bb[s]) / c["min_fill"]
        single_max = need_side(bb[s])
        while e + 1 <= hi:
            cand_fill = min(s_max_fill, size(bb[e + 1]) / c["min_fill"])
            cand_single = max(single_max, need_side(bb[e + 1]))
            if need_side(_union(bb[s:e + 2])) <= max(cand_fill, cand_single):
                e += 1
                s_max_fill, single_max = cand_fill, cand_single
            else:
                break
        flags = []
        if e < hi and e - s + 1 <= ov:  # too short to progress past the overlap: extend (fill rule broken)
            flags.append("odcinek wydłużony ponad min_fill (szybka zmiana rozmiaru)")
            e = min(hi, s + ov)
        u = _union(bb[s:e + 1])
        crop, pad = square_crop((u[0] + u[2]) / 2, (u[1] + u[3]) / 2, max(u[2] - u[0], u[3] - u[1]),
                                plate_w, plate_h, c["pad_ratio"], c["min_margin_px"], snap)
        side = crop[2]
        fill = min(size(bb[i]) / side for i in range(s, e + 1))
        tiles = _tiles(side) if (thin_parts and side > 2 * SAM_RES) else []
        segs.append(CropSegment(
            seg_id=f"s{len(segs):02d}", first=first_frame + s, last=first_frame + e,
            overlap_in=0 if not segs else ov, overlap_out=0, crop=crop, pad=pad,
            scale_to_1008=round(SAM_RES / side, 4), tiles=tiles, fill_min=round(fill, 3), flags=flags))
        if e >= hi:
            break
        segs[-1].overlap_out = ov
        s = e - ov + 1
    if segs:
        segs[0].overlap_in = 0
    return segs


def group_segments(parent: Sequence[CropSegment], members: Sequence[Sequence[CropSegment]], plate_w: int, plate_h: int,
                   snap: int = 16) -> List[CropSegment]:
    """Crops of a group (parent + parts, computed as one whole in stages 3 / 4): every parent segment grown to the union
    of its own plate box and the plate boxes of the members' segments overlapping it in time (user report 2026-10-16:
    the head of CLIP2_A reached out of the hands' crop and lost its edge band there). A segment already covering
    everything is returned unchanged (same geometry -> same caches and results as before)."""
    out = []
    for s in parent:
        b = s.box()
        u = [b.x, b.y, b.x + b.w, b.y + b.h]
        for ms in members:
            for m in ms:
                if m.last < s.first or m.first > s.last:
                    continue
                mb = m.box()
                u = [min(u[0], mb.x), min(u[1], mb.y), max(u[2], mb.x + mb.w), max(u[3], mb.y + mb.h)]
        if u[0] >= b.x and u[1] >= b.y and u[2] <= b.x + b.w and u[3] <= b.y + b.h:
            out.append(s)
            continue
        # the boxes already carry the crop margins: no extra pad / margin, only the snap
        crop, pad = square_crop((u[0] + u[2]) / 2, (u[1] + u[3]) / 2, max(u[2] - u[0], u[3] - u[1], s.side),
                                plate_w, plate_h, 1.0, 0.0, snap)
        side = crop[2]
        out.append(CropSegment(
            seg_id=s.seg_id, first=s.first, last=s.last, overlap_in=s.overlap_in, overlap_out=s.overlap_out,
            crop=crop, pad=pad, scale_to_1008=round(SAM_RES / side, 4),
            tiles=_tiles(side) if (s.tiles and side > 2 * SAM_RES) else [], manual=s.manual,
            fill_min=round(s.fill_min * s.side / side, 3), flags=list(s.flags) + ["kadr grupy"]))
    return out


def frame_segments(segs: Sequence[CropSegment], frame: int) -> List[CropSegment]:
    return [s for s in segs if s.first <= frame <= s.last]


def crops_cover_keys(segs: Sequence[CropSegment], key_boxes: Dict[int, Tuple[int, int, int, int]],
                     margin: float = 0.02) -> Optional[str]:
    """None when every user key's mask box (x0, y0, x1, y1, plate px) lies inside the crop of a segment of its frame
    (with `margin` × side), else why the crops must be recomputed („Przelicz całość”, 2026-10-15). Crops are only
    redone when needed: new crops change the core inputs, so the interval cache would be lost."""
    for f, (x0, y0, x1, y1) in sorted(key_boxes.items()):
        segs_f = frame_segments(segs, f)
        if not segs_f:
            return f"klucz F{f} poza odcinkami kadru"
        ok = False
        for s in segs_f:
            cx, cy, side = s.crop[0], s.crop[1], s.crop[2]
            m = side * margin
            if x0 >= cx + m and y0 >= cy + m and x1 <= cx + side - m and y1 <= cy + side - m:
                ok = True
                break
        if not ok:
            return f"maska klucza F{f} wychodzi poza kadr odcinka"
    return None


def save_crops(path: Union[str, Path], segs: Sequence[CropSegment], info: Optional[Dict[str, Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": CROPS_SCHEMA, "segments": [s.to_dict() for s in segs], **(info or {})}
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_crops(path: Union[str, Path]) -> List[CropSegment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CropSegment.from_dict(d) for d in data.get("segments", [])]


# ----------------------------------------------------------------------------- pipeline
@dataclass
class CropResult:
    segments: List[CropSegment]
    candidates: int
    start_frame: int
    flags: Dict[int, str]
    bboxes: Dict[int, Optional[Tuple[float, float, float, float]]]


def choose_instance(runner: Any, proxy: np.ndarray, prompt: Optional[str] = None,
                    positive: Optional[Sequence[Tuple[float, float]]] = None,
                    negative: Optional[Sequence[Tuple[float, float]]] = None) -> List[np.ndarray]:
    """SAM3_Detect on the letterboxed square proxy (individual masks) -> candidate masks at proxy size."""
    sq, (ox, oy) = letterbox_square(proxy)
    h, w = proxy.shape[:2]
    sh = lambda pts: [(x + ox, y + oy) for x, y in pts] if pts else None
    masks = runner.detect(sq, prompt=prompt, positive=sh(positive), negative=sh(negative), individual=True)
    return [m[oy:oy + h, ox:ox + w] for m in masks if (m[oy:oy + h, ox:ox + w] > 0.5).any()]


def run_crop_engine(runner: Any, load_proxy, frames: Sequence[int], start_frame: int, init_mask_proxy: np.ndarray,
                    proxy_scale: float, plate_w: int, plate_h: int, cfg: Optional[Dict[str, Any]] = None,
                    thin_parts: bool = False, key_masks_proxy: Optional[Dict[int, np.ndarray]] = None,
                    for_ltx: bool = False) -> CropResult:
    """Steps 1-5: coarse track on proxies -> smoothed plate bboxes -> constant-crop segments."""
    from engine.comfy_io import LazySeq
    proxies = LazySeq(lambda i: load_proxy(frames[i]), range(len(frames)))   # read on access (long shots)
    idx_keys = {frames.index(f): m for f, m in (key_masks_proxy or {}).items() if f in frames}
    tr = coarse_localize(runner, proxies, frames.index(start_frame), init_mask_proxy, proxy_scale, idx_keys,
                         frame_numbers=frames)
    # rounding safety: proxy px -> plate px, grow by one proxy pixel
    g = 1.0 / proxy_scale
    raw = [None if b is None else (max(0.0, b[0] - g), max(0.0, b[1] - g), min(plate_w, b[2] + g), min(plate_h, b[3] + g))
           for b in tr.bboxes]
    sm = smooth_bboxes(raw)
    segs = plan_segments(sm, frames[0], plate_w, plate_h, cfg, thin_parts=thin_parts, for_ltx=for_ltx)
    return CropResult(segments=segs, candidates=1, start_frame=start_frame,
                      flags={frames[i]: v for i, v in tr.flags.items()},
                      bboxes={f: sm[i] for i, f in enumerate(frames)})
