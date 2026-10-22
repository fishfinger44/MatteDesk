# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_measure.py – measure the flicker of a plate window before deflickering (docs/PLAN_DEFLICKER_AUTO.md § 2).

Every number comes from consecutive frame pairs aligned with DIS optical flow (flows computed on the original), at
`eval_w` px (1024 by default – the same scale as DeflickerMetrics), so the measurement sees light changes, not motion.
Per region ("all" + the masks given, e.g. a SAM 3.1 object of the current frame):
- metric_pct   – the flow-aligned flicker metric (|log2 ratio| after a σ2 blur) = what the report calls "flicker";
- floor_pct    – the same metric on the same frame warped there and back through the flow: resampling + flow error,
                 i.e. what no deflicker can remove ("achievable");
- level_pct    – mean |step| of ONE gain for the whole region (lamp intensity) and its series; level_autocorr = lag-1
                 autocorrelation of the per-frame level after removing the trend (≈ 0: random per frame, > 0.5: slow
                 variation with memory); drift_pct ± drift_sigma_pct = linear trend over the window (kept, not removed);
- colour_pct   – mean |step| of R/G and B/G (lamp colour);
- local_pct    – what one gain per region does not explain (per-pixel residual), with its own flow floor;
- texture_bands_pct – that residual split by spatial scale (< 2 px grain, 2–8, 8–40, > 40 px);
- texture_fixed_share – PCA of the pair residual maps (8–40 px band, aligned to the middle frame): share of variance
                 in the first 3 modes. High = a fixed footprint whose amplitude changes (a lamp spot: a temporal median
                 removes it); low = a new pattern every frame (surface 'boiling', grain);
- period_frames / period_share / period_p – regular rhythm of the per-frame level (lamp beat, e.g. LED / fluorescent
                 on video): strongest period of the detrended level series, its share of the variance and the chance a random
                 series of the same values gives as strong a peak (p < 0.01 = a real rhythm). SH010 2026-10-22: no rhythm (tree
                 p 0.25) – stop-motion frames get a random lamp phase each;
- motion_resid_px – DIS flow minus the frame's median flow (camera) on the region's TEXTURED pixels: an object moving on
                 its own; None when the region has no texture (flow is meaningless on a plain wall) – such a region's mask
                 is moved with the camera, textured ones rigidly with their own median flow (texture_share tells which).
Frame-wide: global_shift_px (camera motion per pair), tiles_level_pct (9×6 grid) and region_corr (do the tiles
flicker together or independently → whether the 9×6 stage is needed).
Reference values: SH010 X1 left tree 2026-10-21 – metric 4.19 / floor 1.13 / level 1.26 / local 4.8 (floor 2.2) /
texture_fixed_share 0.91 (modes 1–3), wall–tree level correlation −0.08.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from engine.clean import _map_frames, as_bgr, clean_workers

LN2 = float(np.log(2.0))
BANDS = (("<2", 0.0, 2.0), ("2-8", 2.0, 8.0), ("8-40", 8.0, 40.0), (">40", 40.0, None))
TEXTURE_BAND = (8.0, 40.0)   # band used for the fixed-pattern PCA (the SH010 lamp footprint lives here)
TEXTURED_GRAD = 0.02          # |gradient of log2 luma| per px above this = texture the flow can hold on to
MIN_TEXTURED_PX = 64
MIN_TEXTURED_SHARE = 0.10      # below this share the region's flow is its edge with other things: use the camera motion


@dataclass
class RegionMeasure:
    name: str
    px: int
    metric_pct: float
    floor_pct: float
    level_pct: float
    level_series_pct: List[float]
    level_autocorr: Optional[float]
    drift_pct: float
    drift_sigma_pct: float          # standard error of the trend: a drift is real when |drift| > ~2 sigma
    colour_pct: Dict[str, float]
    local_pct: float
    local_floor_pct: float
    texture_bands_pct: Dict[str, float]
    texture_floor_bands_pct: Dict[str, float]   # the same bands on the flow round trip (resampling + flow error)
    texture_fixed_share: Optional[float]        # None when the 8-40 px band is not above its floor (nothing fixed to measure)
    motion_resid_px: Optional[float]
    texture_share: float            # share of the region's pixels with texture (flow / motion numbers rest on them)
    period_frames: Optional[float] = None   # strongest period of the level series (frames), None when < 8 frames
    period_share: Optional[float] = None    # its share of the (detrended) level variance
    period_p: Optional[float] = None        # chance of as strong a peak in a shuffled series (rhythm when < 0.01)


@dataclass
class FlickerMeasure:
    n_frames: int
    eval_w: int
    eval_h: int
    global_shift_px: float          # camera motion per pair: |median DIS flow| of the whole frame (px at eval width)
    region_corr: Optional[float]
    tiles_level_pct: List[List[float]]
    regions: Dict[str, RegionMeasure] = field(default_factory=dict)
    maps: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False, compare=False)   # "metric": per-pixel flicker map (eval size, %)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("maps", None)
        return d


MaskSpec = Union[np.ndarray, Sequence[np.ndarray]]


def level_period(steps_pct: Sequence[float], shuffles: int = 400, seed: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(period frames, share of variance, p vs shuffled) of the per-frame level built from its frame-to-frame steps; the slow
    drift (quadratic) is removed first. None when too short (< 8 frames)."""
    st = np.asarray(steps_pct, np.float64)
    if len(st) < 7:
        return None, None, None
    lv = np.concatenate([[0.0], np.cumsum(st)])
    t = np.arange(len(lv))
    x = lv - np.polyval(np.polyfit(t, lv, 2), t)
    x = x - x.mean()
    if x.std() < 1e-9:
        return None, None, None

    def peak(v: np.ndarray) -> Tuple[int, float]:
        s = np.abs(np.fft.rfft(v)) ** 2
        s[0] = 0.0
        k = int(np.argmax(s[1:]) + 1)
        return k, float(s[k] / max(s[1:].sum(), 1e-12))

    k, share = peak(x)
    rng = np.random.default_rng(seed)
    null = np.array([peak(rng.permutation(x))[1] for _ in range(shuffles)])
    p = float((null >= share).mean())
    period = len(x) / k
    return round(float(period), 2), round(share, 3), round(p, 4)


def _g8(x: np.ndarray) -> np.ndarray:
    return cv2.normalize(np.log(x.mean(axis=2)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _warp(img: np.ndarray, flow: np.ndarray, gx: np.ndarray, gy: np.ndarray, nearest: bool = False) -> np.ndarray:
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _blur_in(mask_f: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of `x` inside the mask (normalised convolution: nothing leaks in from outside)."""
    if sigma <= 0:
        return x
    num = cv2.GaussianBlur(x * mask_f, (0, 0), sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def _bands(resid: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean |band| (log2 %) of the residual inside the mask, bands by Gaussian scale."""
    mf = mask.astype(np.float32)
    out = {}
    prev = resid
    for name, lo, hi in BANDS:
        if hi is None:
            band = _blur_in(mf, resid, lo)
        else:
            band = prev - _blur_in(mf, resid, hi)
            prev = _blur_in(mf, resid, hi)
        out[name] = float(np.abs(band[mask]).mean() * 100) if mask.any() else 0.0
    return out


def _texture_band_map(resid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mf = mask.astype(np.float32)
    lo, hi = TEXTURE_BAND
    return (_blur_in(mf, resid, lo) - _blur_in(mf, resid, hi)) * mf


def measure_flicker(frames: Sequence[np.ndarray], masks: Optional[Dict[str, MaskSpec]] = None, eval_w: int = 1024,
                    tiles: Tuple[int, int] = (9, 6), pca_modes: int = 3, mask_erode_px: int = 3,
                    progress_cb: Optional[Callable[[int, int], None]] = None,
                    flow_from: Optional[Sequence[np.ndarray]] = None, return_maps: bool = False,
                    inner_border: float = 0.06) -> FlickerMeasure:
    """Measure `frames` (BGR float, scene-linear, any size; >= 3 frames). `masks`: name -> bool mask of the MIDDLE frame
    (moved rigidly to every frame by the median flow of the object – per-pixel flow is unreliable on plain surfaces) or a
    sequence of per-frame masks (e.g. SAM 3.1 cores); full-frame resolution or eval size.
    `inner_border`: fraction of the frame left out on every side for the always-present region "inner" (the frame
    border pumps under camera motion – new content enters the aligned stack; 0 = no such region).
    `flow_from`: frames the flows / masks / texture are taken from instead of `frames` (measure a deflickered result with
    the ORIGINAL's flows, so before / after are compared on identical alignment). `return_maps`: also the per-pixel
    flicker map ("metric", mean over pairs, %) in `.maps` for the preview heat map."""
    n = len(frames)
    if n < 3:
        raise ValueError("pomiar migotania potrzebuje co najmniej 3 klatek")
    f0 = as_bgr(np.asarray(frames[0], np.float32))
    H0, W0 = f0.shape[:2]
    ew = min(int(eval_w), W0)
    eh = max(8, round(H0 * ew / W0))

    def small(a: np.ndarray) -> np.ndarray:
        a = as_bgr(np.asarray(a, np.float32))
        a = cv2.resize(a, (ew, eh), interpolation=cv2.INTER_AREA) if a.shape[:2] != (eh, ew) else a
        return np.maximum(np.nan_to_num(a, nan=0.0), 1e-4)

    seq = [small(frames[i]) for i in range(n)]
    ref = seq if flow_from is None else [small(flow_from[i]) for i in range(n)]
    if len(ref) != n:
        raise ValueError("flow_from: inna liczba klatek niż frames")
    g8 = [_g8(x) for x in ref]

    def textured(x: np.ndarray) -> np.ndarray:
        lg = cv2.GaussianBlur(np.log2(x.mean(axis=2)).astype(np.float32), (0, 0), 1.5)
        return np.hypot(cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3)) / 8.0 > TEXTURED_GRAD

    tex = [textured(x) for x in ref]
    gx, gy = np.meshgrid(np.arange(ew, dtype=np.float32), np.arange(eh, dtype=np.float32))
    mid = n // 2

    # ---- masks per frame at eval size
    def to_eval_mask(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m)
        m = m if m.shape[:2] == (eh, ew) else cv2.resize(m.astype(np.uint8), (ew, eh), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)

    region_masks: Dict[str, List[np.ndarray]] = {"all": [np.ones((eh, ew), bool)] * n}
    if inner_border > 0 and min(eh, ew) >= 32:
        inner = np.zeros((eh, ew), bool)
        by, bx = int(round(eh * inner_border)), int(round(ew * inner_border))
        inner[by:eh - by, bx:ew - bx] = True
        region_masks["inner"] = [inner] * n
    static_names = []
    for name, spec in (masks or {}).items():
        if isinstance(spec, np.ndarray) and spec.ndim == 2:
            static_names.append(name)
            region_masks[name] = [to_eval_mask(spec)] * n     # replaced below by the flow-propagated masks
        else:
            seq_m = list(spec)
            if len(seq_m) != n:
                raise ValueError(f"maska {name}: {len(seq_m)} klatek zamiast {n}")
            region_masks[name] = [to_eval_mask(m) for m in seq_m]
    if static_names:
        def prop(i: int) -> Dict[str, np.ndarray]:
            if i == mid:
                return {nm: region_masks[nm][mid] for nm in static_names}
            fl = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g8[mid], g8[i], None)   # mid pixel -> frame i
            cam = np.median(fl.reshape(-1, 2), axis=0)
            out = {}
            for nm in static_names:
                m0 = region_masks[nm][mid]
                mt = m0 & tex[mid]
                ok_tex = mt.sum() >= MIN_TEXTURED_PX and mt.sum() >= MIN_TEXTURED_SHARE * m0.sum()
                v = np.median(fl[mt].reshape(-1, 2), axis=0) if ok_tex else cam   # own motion, else camera
                M = np.array([[1, 0, float(v[0])], [0, 1, float(v[1])]], np.float32)
                out[nm] = cv2.warpAffine(m0.astype(np.uint8), M, (ew, eh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
            return out
        per = _map_frames(prop, n, clean_workers(eh * ew * 4 * 8))
        for nm in static_names:
            region_masks[nm] = [per[i][nm] for i in range(n)]
    if mask_erode_px > 0:
        k = np.ones((2 * mask_erode_px + 1,) * 2, np.uint8)
        for nm in list(region_masks):
            if nm not in ("all", "inner"):
                region_masks[nm] = [cv2.erode(m.astype(np.uint8), k).astype(bool) for m in region_masks[nm]]
    names = list(region_masks)
    gxn, gyn = tiles
    tile_edges_x = np.linspace(0, ew, gxn + 1).astype(int)
    tile_edges_y = np.linspace(0, eh, gyn + 1).astype(int)

    # ---- per pair
    def pair(k: int) -> Dict[str, Any]:
        i = k + 1
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        cur, prv = seq[i], seq[i - 1]
        fwd = dis.calc(g8[i], g8[i - 1], None)                # cur pixel -> where it was in prev
        back = dis.calc(g8[i - 1], g8[i], None)
        prv_w = _warp(prv, fwd, gx, gy)
        rt = _warp(_warp(cur, fwd, gx, gy), back, gx, gy)      # round trip: resampling + flow error only
        # camera motion = median flow of the frame; an object's own motion = flow minus that
        cam = np.median(fwd.reshape(-1, 2), axis=0)
        shift = float(np.hypot(*cam))
        resid_flow = np.linalg.norm(fwd - cam[None, None, :], axis=2)
        txm = tex[i]   # textured pixels of this frame (not `tx`: the tile loop below uses that name)
        # photometric change per pixel (log2), per channel
        d = (np.log(cur) - np.log(prv_w)) / LN2
        d_rt = (np.log(cur) - np.log(rt)) / LN2
        dl, dl_rt = d.mean(axis=2), d_rt.mean(axis=2)
        cb = cv2.GaussianBlur(cur, (0, 0), 2.0)
        pb = cv2.GaussianBlur(prv_w, (0, 0), 2.0)
        rb = cv2.GaussianBlur(rt, (0, 0), 2.0)
        met = np.abs(np.log2((cb + 0.02) / (pb + 0.02))).mean(axis=2) * 100
        met_floor = np.abs(np.log2((cb + 0.02) / (rb + 0.02))).mean(axis=2) * 100
        out_map = met if return_maps else None
        to_mid = None if i == mid else dis.calc(g8[mid], g8[i], None)   # mid grid -> frame i
        out: Dict[str, Any] = {"shift": shift, "regions": {}, "tiles": np.zeros((gyn, gxn), np.float32), "map": out_map}
        for ty in range(gyn):
            for tx in range(gxn):
                sl = (slice(tile_edges_y[ty], tile_edges_y[ty + 1]), slice(tile_edges_x[tx], tile_edges_x[tx + 1]))
                out["tiles"][ty, tx] = float(dl[sl].mean())
        for nm in names:
            m = region_masks[nm][i]
            if m.sum() < 16:
                out["regions"][nm] = None
                continue
            level = float(dl[m].mean())
            resid = dl - level
            r = {"metric": float(met[m].mean()), "floor": float(met_floor[m].mean()), "level": level * 100,
                 "rg": float((d[..., 2] - d[..., 1])[m].mean() * 100), "bg": float((d[..., 0] - d[..., 1])[m].mean() * 100),
                 "local": float(np.abs(resid[m]).mean() * 100),
                 "local_floor": float(np.abs(dl_rt[m] - dl_rt[m].mean()).mean() * 100),
                 "bands": _bands(resid, m), "floor_bands": _bands(dl_rt - float(dl_rt[m].mean()), m),
                 "motion": (float(resid_flow[m & txm].mean())
                            if (m & txm).sum() >= max(MIN_TEXTURED_PX, MIN_TEXTURED_SHARE * m.sum()) else None),
                 "tex_share": float((m & txm).sum() / m.sum()), "px": int(m.sum())}
            band = _texture_band_map(resid, m)
            r["band_mid"] = band if to_mid is None else _warp(band, to_mid, gx, gy)   # in mid's grid, for the PCA
            out["regions"][nm] = r
        if progress_cb:
            progress_cb(k + 1, n - 1)
        return out

    pairs = _map_frames(pair, n - 1, clean_workers(eh * ew * 4 * 40))

    # ---- per-frame mean log2 luma per region (drift), no alignment needed: the mask follows the object
    def lum_series(nm: str) -> np.ndarray:
        return np.array([float(np.log2(seq[i].mean(axis=2)[region_masks[nm][i]]).mean()) if region_masks[nm][i].sum() >= 16 else np.nan
                         for i in range(n)])

    regions: Dict[str, RegionMeasure] = {}
    for nm in names:
        rs = [p["regions"][nm] for p in pairs]
        valid = [r for r in rs if r is not None]
        if len(valid) < 2:
            continue
        levels = np.array([r["level"] for r in valid])
        ls = lum_series(nm)
        ok = ~np.isnan(ls)
        t = np.arange(n)[ok]
        drift, drift_sigma, ac = 0.0, 0.0, None
        if ok.sum() >= 3:
            slope, icpt = np.polyfit(t, ls[ok], 1)
            fit = slope * t + icpt
            dof = max(1, ok.sum() - 2)
            se = float(np.sqrt(((ls[ok] - fit) ** 2).sum() / dof / max(((t - t.mean()) ** 2).sum(), 1e-9)))
            drift, drift_sigma = float(slope * (n - 1) * 100), float(se * (n - 1) * 100)
            dev = ls[ok] - fit                                   # per-frame level without the trend
            if len(dev) >= 5 and dev.std() > 1e-9:
                ac = float(np.corrcoef(dev[:-1], dev[1:])[0, 1])
        # fixed-pattern share: PCA of the residual band maps aligned to mid, inside the mid mask
        share = None
        mm = region_masks[nm][mid]
        if mm.sum() >= 64:
            X = np.stack([r["band_mid"][mm] for r in valid])
            X = X - X.mean(axis=1, keepdims=True)
            if np.abs(X).sum() > 0:
                s = np.linalg.svd(X, compute_uv=False)
                var = s ** 2
                share = float(var[:pca_modes].sum() / max(var.sum(), 1e-12))
        mot = [r["motion"] for r in valid if r["motion"] is not None]
        per = level_period(levels) if len(valid) == n - 1 else (None, None, None)   # gaps (lost pairs) would break the rhythm
        bands = {b: float(np.mean([r["bands"][b] for r in valid])) for b, _, _ in BANDS}
        floor_bands = {b: float(np.mean([r["floor_bands"][b] for r in valid])) for b, _, _ in BANDS}
        if share is not None and bands["8-40"] <= 1.5 * floor_bands["8-40"] + 0.05:
            share = None   # the band is resampling / flow error only: its 'fixed pattern' is the texture, not a lamp
        regions[nm] = RegionMeasure(
            name=nm, px=int(np.median([r["px"] for r in valid])),
            metric_pct=round(float(np.mean([r["metric"] for r in valid])), 3),
            floor_pct=round(float(np.mean([r["floor"] for r in valid])), 3),
            level_pct=round(float(np.abs(levels).mean()), 3), level_series_pct=[round(float(x), 3) for x in levels],
            level_autocorr=None if ac is None else round(ac, 3), drift_pct=round(drift, 3), drift_sigma_pct=round(drift_sigma, 3),
            colour_pct={"rg": round(float(np.mean([abs(r["rg"]) for r in valid])), 3), "bg": round(float(np.mean([abs(r["bg"]) for r in valid])), 3)},
            local_pct=round(float(np.mean([r["local"] for r in valid])), 3),
            local_floor_pct=round(float(np.mean([r["local_floor"] for r in valid])), 3),
            texture_bands_pct={k: round(v, 3) for k, v in bands.items()},
            texture_floor_bands_pct={k: round(v, 3) for k, v in floor_bands.items()},
            texture_fixed_share=None if share is None else round(share, 3),
            motion_resid_px=(round(float(np.mean(mot)), 3) if len(mot) >= len(valid) / 2 else None),   # most pairs must have texture
            period_frames=per[0], period_share=per[1], period_p=per[2],
            texture_share=round(float(np.mean([r["tex_share"] for r in valid])), 3))

    tiles_series = np.stack([p["tiles"] for p in pairs])           # (pairs, gy, gx) level steps (log2)
    tiles_level = (np.abs(tiles_series).mean(axis=0) * 100).round(3).tolist()
    flat = tiles_series.reshape(len(pairs), -1).T                   # (tiles, pairs)
    good = flat[flat.std(axis=1) > 1e-6]
    region_corr = None
    if len(good) >= 2 and len(pairs) >= 3:
        C = np.corrcoef(good)
        off = C[~np.eye(len(C), dtype=bool)]
        region_corr = round(float(np.nanmean(off)), 3)
    maps = {"metric": np.mean([p["map"] for p in pairs], axis=0).astype(np.float32)} if return_maps else None
    return FlickerMeasure(n_frames=n, eval_w=ew, eval_h=eh, global_shift_px=round(float(np.mean([p["shift"] for p in pairs])), 3),
                          region_corr=region_corr, tiles_level_pct=tiles_level, regions=regions, maps=maps)
