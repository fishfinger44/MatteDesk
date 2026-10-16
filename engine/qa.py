# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/qa.py - Object QA gate (SPEC 8): whole object, every frame, flicker map.

Metrics (per frame, full plate coordinates, inside the whole-object bbox = union(alpha > 0.05) + 40 px):
- flicker: mean |a(t) - a(t-1)| after camera compensation (ECC translation on plate luma at 1/4, exactly as
  tools/pm_qa_figure.py); gate: median over frames <= 0.004; frames > 2.5 x median flagged; per-pixel flicker map
- area jump / IoU with the previous frame (camera-compensated) -> flags
- interior holes = 0 (unless from a Remove zone / user key), islands < 600 px = 0
- seams (multi-object split): pixels < 0.9 surrounded by > 0.99 in the union = 0
- fwd/bwd agreement of the core (7.2 pkt 4), jumps at crop-segment boundaries
Flags -> timeline colours: red (metric gate), yellow (suspicious), green; blue = key frames (UI).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

FLICKER_GATE = 0.004
ISLAND_MIN_PX = 600


ROI_WARP = True   # flicker warp on the object's box only (bitwise = full frame; tests switch it off for the reference)


def default_workers() -> int:
    """Threads for the QA stage (EXR decoding + per-frame cv2 work release the GIL). config.json -> qa_workers."""
    try:
        from engine.config import load_config
        n = int(load_config().get("qa_workers") or 0)
    except Exception:
        n = 0
    return n if n > 0 else max(1, (os.cpu_count() or 4) - 2)


class _GrayCache:
    """Each plate frame decoded ONCE (in parallel) and kept compactly: uint8 when the gray is integral 0..255
    (lossless for the display grays used here), float16 as given (log luminance), else float32.
    get(f) returns float32 like the loader."""

    def __init__(self, frames: Sequence[int], load: Callable[[int], np.ndarray], pool: ThreadPoolExecutor):
        self._d: Dict[int, np.ndarray] = {}
        for f, g in zip(frames, pool.map(load, frames)):
            g = np.asarray(g)
            if g.dtype == np.float16:      # raw-EXR log luminance (Z27 2b): kept compact as is
                pass
            elif g.dtype != np.uint8 and g.size and float(g.min()) >= 0 and float(g.max()) <= 255 and np.array_equal(g, np.round(g)):
                g = g.astype(np.uint8)
            self._d[f] = g

    def get(self, f: int) -> np.ndarray:
        return self._d[f].astype(np.float32)


def camera_shift(g_prev: np.ndarray, g_cur: np.ndarray) -> np.ndarray:
    """ECC translation (prev -> cur) at 1/4 resolution, as in pm_qa_figure.py."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(cv2.resize(g_cur, None, fx=0.25, fy=0.25), cv2.resize(g_prev, None, fx=0.25, fy=0.25),
                                       warp, cv2.MOTION_TRANSLATION,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), None, 5)
        warp[:, 2] *= 4
    except cv2.error:
        pass
    return warp


def interior_holes(a: np.ndarray, min_px: int = 16, allowed: Optional[np.ndarray] = None) -> int:
    """Background components fully enclosed by the object (not touching the frame border)."""
    fg = (a > 0.5).astype(np.uint8)
    if not fg.any():
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - fg, connectivity=4)
    h, w = fg.shape
    cnt = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or area < min_px:
            continue
        if allowed is not None and (allowed[lab == i]).mean() > 0.5:
            continue
        cnt += 1
    return cnt


def small_islands(a: np.ndarray, min_px: int = ISLAND_MIN_PX, allowed: Optional[np.ndarray] = None) -> int:
    """Components (alpha > 0.5) smaller than min_px besides the main one. `allowed` (area of soft `matte` zones): a
    small component touching it is a strand the zone drew on purpose (hygiene keeps it too), not an island."""
    fg = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 2:
        return 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnt = 0
    for i in range(1, n):
        if i == main or stats[i, cv2.CC_STAT_AREA] >= min_px:
            continue
        if allowed is not None and allowed[lab == i].any():
            continue
        cnt += 1
    return cnt


def seam_count(union: np.ndarray, alphas: Optional[Sequence[np.ndarray]] = None,
               valid: Optional[np.ndarray] = None) -> int:
    """Pixels < 0.9 sandwiched between > 0.99 neighbours (left & right or up & down; 1-2 px wide seams).
    With per-object `alphas`, only where two objects meet (both within 2 px) – a notch of one object is not a seam."""
    hi = union > 0.99
    lo = union < 0.9
    sand = np.zeros_like(hi)
    for d in (1, 2):
        lr = np.zeros_like(hi)
        ud = np.zeros_like(hi)
        lr[:, d:-d] = hi[:, :-2 * d] & hi[:, 2 * d:]
        ud[d:-d, :] = hi[:-2 * d, :] & hi[2 * d:, :]
        sand |= lr | ud
    if alphas is not None and len(alphas) >= 2:
        near = [cv2.dilate((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for a in alphas]
        meet = np.zeros_like(hi)
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                meet |= near[i] & near[j]
        sand &= meet
    if valid is not None:  # e.g. ground truth says "object here" (a real 1-px background gap is not a seam)
        sand &= valid
    return int((lo & sand).sum())


@dataclass
class QAReport:
    frames: List[int]
    flicker: Dict[int, float]
    flicker_median: float
    flags: Dict[int, List[Tuple[str, str]]]          # frame -> [(level, reason)]
    colours: Dict[int, str]
    holes: Dict[int, int]
    islands: Dict[int, int]
    seams: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    gate_ok: bool = False
    banner: str = ""
    suggested_keys: List[int] = field(default_factory=list)
    hygiene: Dict[str, Any] = field(default_factory=dict)
    edge: Dict[str, Any] = field(default_factory=dict)
    plate_gray: Dict[str, Any] = field(default_factory=dict)   # what the auxiliary measures read (Z27 2b)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flags"] = {str(k): v for k, v in self.flags.items()}
        return d


def run_qa(frames: Sequence[int], load_alpha: Callable[[int], np.ndarray], load_gray: Optional[Callable[[int], np.ndarray]],
           fwd_bwd_iou: Optional[Dict[int, float]] = None, seg_boundaries: Sequence[int] = (),
           allowed_holes: Optional[Callable[[int], np.ndarray]] = None, attempts: int = 0,
           allowed_islands: Optional[Callable[[int], Optional[np.ndarray]]] = None,
           flicker_map_path: Optional[Path] = None, flow_check: bool = False,
           load_clean_gray: Optional[Callable[[int], np.ndarray]] = None, workers: Optional[int] = None,
           progress_cb: Optional[Callable[[str, int, int], None]] = None,
           load_gray_rows: Optional[Callable[[int, int, int], np.ndarray]] = None,
           load_clean_gray_rows: Optional[Callable[[int, int, int], np.ndarray]] = None,
           gray_source: str = "") -> QAReport:
    """Same metrics and numbers as the former sequential version (Z27): plates decoded once, in parallel; per-frame
    work in threads; everything that is summed / ordered is combined in frame order (identical qa.json).
    `load_gray_rows(f, y0, y1)` (Z27 2b): the plate gray is read only for the band of rows around the whole object
    (`gray_band`); camera shift (translation), flow and clear-plate agreement are measured on that band. The alpha
    measures (holes, islands; flicker for the same camera shift) do not depend on it."""
    frames = list(frames)
    n_w = workers or default_workers()
    prog = progress_cb or (lambda *a: None)
    with ThreadPoolExecutor(max_workers=n_w) as pool:
        al = {f: load_alpha(f).astype(np.float32) for f in frames}
        H, W = al[frames[0]].shape
        u = al[frames[0]].copy()          # running max = np.max(np.stack(...), 0) without an 8 GB copy (193 x 4K)
        for f in frames[1:]:
            np.maximum(u, al[f], out=u)
        u = u > 0.05
        m = 40
        if u.any():
            ys, xs = np.nonzero(u)
            y0, y1, x0, x1 = max(0, ys.min() - m), min(H, ys.max() + m), max(0, xs.min() - m), min(W, xs.max() + m)
        else:
            y0, y1, x0, x1 = 0, H, 0, W
        flags: Dict[int, List[Tuple[str, str]]] = {f: [] for f in frames}
        score: Dict[int, float] = {frames[0]: 0.0}
        fmap = np.zeros((y1 - y0, x1 - x0), np.float32)
        prog("QA: wczytywanie klatek", 0, len(frames))
        by0, by1 = 0, H
        if load_gray_rows is not None:
            by0, by1 = gray_band(y0, y1, H)
            G = _GrayCache(frames, lambda f: load_gray_rows(f, by0, by1), pool)
            C = _GrayCache(frames, lambda f: load_clean_gray_rows(f, by0, by1), pool) if load_clean_gray_rows else None
        else:
            G = _GrayCache(frames, load_gray, pool) if load_gray else None
            C = _GrayCache(frames, load_clean_gray, pool) if (load_clean_gray and load_gray) else None
        ab = lambda f: al[f][by0:by1]      # alpha on the rows of the gray (whole frame without a band)
        pairs = list(zip(frames[:-1], frames[1:]))

        def pair_metrics(pr):
            fp, f = pr
            warp = camera_shift(G.get(fp), G.get(f)) if G else np.eye(2, 3, dtype=np.float32)
            if ROI_WARP and max(abs(float(warp[0, 2])), abs(float(warp[1, 2]))) < m - 1:
                # bitwise the same as the full-frame warp, only the object's box (12 % of SH010): alpha > 0.05 lies >= m px
                # inside the box, so with a translation < m - 1 px nothing > 0.5 can come from / go outside it
                w2 = warp.astype(np.float64)
                w2[0, 2] += x0
                w2[1, 2] += y0
                prev = cv2.warpAffine(al[fp], w2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                cur = al[f][y0:y1, x0:x1]
                d = np.abs(cur - prev)
            else:
                prev = cv2.warpAffine(al[fp], warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                cur = al[f]
                d = np.abs(cur - prev)[y0:y1, x0:x1]
            out = []
            a, b = cur > 0.5, prev > 0.5
            area_a, area_b = a.sum(), b.sum()
            if area_b > 0 and abs(area_a / area_b - 1.0) > 0.15:
                out.append(("red", f"skok powierzchni {100 * (area_a / area_b - 1):+.0f}%"))
            un = (a | b).sum()
            if un and (a & b).sum() / un < 0.9:
                out.append(("yellow", "IoU z poprzednią < 0.90"))
            if f in seg_boundaries and un and (a & b).sum() / un < 0.95:
                out.append(("yellow", "skok na granicy odcinka kadru"))
            return d, float(d.mean()), out

        # ordered consumption in chunks: fmap summed in frame order (bitwise as before), bounded memory
        step = max(1, 2 * n_w)
        for c0 in range(0, len(pairs), step):
            chunk = pairs[c0:c0 + step]
            for (fp, f), (d, sc, fl) in zip(chunk, pool.map(pair_metrics, chunk)):
                fmap += d
                score[f] = sc
                flags[f].extend(fl)
            prog("QA: migotanie", min(len(pairs), c0 + step), len(pairs))
        fmap /= max(1, len(frames) - 1)
        edge_rep: Dict[str, Any] = {}
        if flow_check and G:
            # 7.8 pkt 3: alpha(t) carried by DIS flow to t+1 vs alpha(t+1) in the edge band (occlusions excluded)
            from engine.edge import flow_consistency
            s = min(1.0, 1024.0 / max(W, H))
            rs = lambda x: cv2.resize(x.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x
            fl = {}
            for (fp, f), v in zip(pairs, pool.map(lambda pr: flow_consistency(rs(ab(pr[0])), rs(ab(pr[1])), rs(G.get(pr[0])), rs(G.get(pr[1])))[0], pairs)):
                fl[f] = v
                if v < 0.95:
                    flags[f].append(("yellow", f"zgodność po przepływie {v:.2f} < 0.95"))
            prog("QA: przepływ", len(pairs), len(pairs))
            edge_rep["flow_iou_min"] = round(min(fl.values()), 4) if fl else 1.0
        if C is not None:
            # clear plate as a yardstick (never a mask): |beauty - clear plate| key vs the matte inside the edge band
            def agree_f(f):
                b, c = G.get(f), C.get(f)
                c = c * (np.median(b) / max(float(np.median(c)), 1e-6))
                d = cv2.GaussianBlur(np.abs(b - c), (0, 0), 1.0)
                d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, k8 = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # half-way edge of the difference
                key = k8 > 0
                af = ab(f)
                band = cv2.dilate((af > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & \
                    ~cv2.erode((af > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                a_, k_ = (af > 0.5) & band, key & band
                un = (a_ | k_).sum()
                return 1.0 if un == 0 else float((a_ & k_).sum() / un)
            agree = {}
            for f, v in zip(frames, pool.map(agree_f, frames)):
                agree[f] = v
                if v < 0.8:
                    flags[f].append(("yellow", f"krawędź vs clear plate: zgodność {v:.2f}"))
            prog("QA: clear plate", len(frames), len(frames))
            edge_rep["cleanplate_agreement_median"] = round(float(np.median(list(agree.values()))), 4)
        med = float(np.median([score[f] for f in frames[1:]])) if len(frames) > 1 else 0.0
        for f in frames[1:]:
            if med > 0 and score[f] > 2.5 * med:
                flags[f].append(("yellow", f"migotanie {score[f]:.4f} > 2.5 × mediana"))
        hi = lambda f: (interior_holes(al[f], allowed=allowed_holes(f) if allowed_holes else None),
                        small_islands(al[f], allowed=allowed_islands(f) if allowed_islands else None))
        holes, islands = {}, {}
        for f, (h_, i_) in zip(frames, pool.map(hi, frames)):
            holes[f], islands[f] = h_, i_
            if h_:
                flags[f].append(("red", f"dziury wewnętrzne: {h_}"))
            if i_:
                flags[f].append(("red", f"wyspy < {ISLAND_MIN_PX} px: {i_}"))
    for f, v in (fwd_bwd_iou or {}).items():
        if f in flags and v < 0.9:
            flags[f].append(("yellow", f"przód/tył IoU {v:.2f}"))
    colours = {f: ("red" if any(l == "red" for l, _ in fl) else "yellow" if fl else "green") for f, fl in flags.items()}
    gate = med <= FLICKER_GATE and not any(holes.values()) and not any(islands.values())
    rep = QAReport(frames=frames, flicker={f: round(v, 6) for f, v in score.items()}, flicker_median=round(med, 6),
                   flags=flags, colours=colours, holes=holes, islands=islands,
                   bbox=(int(x0), int(y0), int(x1), int(y1)), gate_ok=bool(gate), edge=edge_rep)
    if G is not None:
        rep.plate_gray = {"source": gray_source or ("rows" if load_gray_rows else "full"), "rows": [int(by0), int(by1)]}
    bad = [f for f in frames if colours[f] == "red"] or [f for f in frames if colours[f] == "yellow"]
    rep.suggested_keys = spread_keys(bad, 5)
    if not gate and attempts >= 2 and bad:
        rep.banner = (f"Automat nie daje rady w klatkach {bad[0]}–{bad[-1]}. Proponuję: narysuj 3–5 kluczy "
                      f"(podpowiedziane klatki: {', '.join(map(str, rep.suggested_keys))}), propagacja od kluczy.")
    if flicker_map_path is not None:
        flicker_map_path.parent.mkdir(parents=True, exist_ok=True)
        fm = (np.clip(fmap / max(1e-4, float(np.percentile(fmap, 99.8))), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(flicker_map_path), cv2.applyColorMap(fm, cv2.COLORMAP_INFERNO))
    return rep


def gray_band(y0: int, y1: int, h: int, margin: int = 128, min_frac: float = 0.25) -> Tuple[int, int]:
    """Rows of the plate read for QA (Z27 2b): object bbox rows (already +40 px) + >= 128 px of background for the
    camera shift; at least 25 % of the frame height (grown around the middle, kept inside the frame)."""
    a, b = max(0, int(y0) - margin), min(h, int(y1) + margin)
    need = int(np.ceil(min_frac * h))
    if b - a < need:
        c = (a + b) / 2.0
        a = int(round(c - need / 2.0))
        a = min(max(0, a), max(0, h - need))
        b = min(h, a + need)
    return a, b


def spread_keys(frames: Sequence[int], k: int = 5) -> List[int]:
    """Up to k frames spread over the problematic ones (middles of runs first)."""
    if not frames:
        return []
    runs, cur = [], [frames[0]]
    for f in frames[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    picks = [r[len(r) // 2] for r in sorted(runs, key=len, reverse=True)]
    return sorted(picks[:k])
