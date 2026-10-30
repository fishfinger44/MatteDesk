# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/localmatte.py - local matting of a zone (docs/PLAN_HAIR_TOOL.md, sections 2-4): hair / fur / fibres / smoke /
motion blur / thin structures inside a user-drawn area, on any (non-uniform) background.

Pure function `local_matte(img, core, region, ...)` -> LocalMatteResult (no zones, no UI, no Comfy):
- crop = bbox(region) + `pad` px, longer side scaled to <= `max_side` (INTER_AREA; alpha back with INTER_LINEAR);
- trimap: FG = core eroded `fg_erode_px`, unknown = region \ FG, BG = rest of the crop ∪ forced BG ("-" strokes,
  holdout of front objects, accepted distractor masks, pixels removed by Remove zones) – forced BG wins over unknown
  but never over the core (semantic core errors are a SAM 3.1 key, not a zone);
- engines on the unknown pixels:
  `chroma` – colour difference (ΔE in Lab of the display image) from the LOCAL background colour (mean of the
             background pixels in a 65 px window; pipette = one global colour); lo = local noise, hi = ΔE of the local
             core colour next to the unknown; wins on locally uniform backgrounds;
  `knn` / `cf` – pymatting KNN / closed-form matting on the crop trimap (motley, sharp backgrounds);
  `hybrid` – chroma as the initialisation / trimap of knn (unknown narrowed to 0.05 < α < 0.95);
  `auto` – per-patch weight map and seamless blend (section 4, step 2 of the plan);
- result only inside the zone: alpha = 1 on the core, matting in the unknown, 0 on forced BG; strands connected to
  the core on α > `connect_thr` within `connect_reach_px` (SPEC 7.7 pkt 4 with the soft threshold of plan section 12);
  `area` = the region (where the zone result wins over the edge mode in refine.finalize).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ENGINES = ("chroma", "knn", "cf", "hybrid", "clean", "matanyone", "vitmatte", "auto")   # matanyone / vitmatte: zone-crop engines (engine/zones.py)
PRESETS = ("hair", "fibres", "smoke", "motion", "thin")


def preset_defaults(preset: str) -> Tuple[str, Dict[str, Any]]:
    """Default zone engine and params of a preset (defaults.json -> localmatte.presets.<preset>.matting / .params;
    user decision 2026-10-14: hair = vitmatte + flow r3 x2; 2026-10-15 (CLIP2_A v038): hair = vitmatte + flow_mean r5 x2,
    black 0.1, white 0.9, binary trimap; 2026-10-28: smoothing x1 (plate-res flow, A/B local_results/ab_zone_smooth/);
    other presets = matanyone, no params)."""
    import json
    from pathlib import Path
    try:
        d = json.loads((Path(__file__).resolve().parents[1] / "defaults.json").read_text(encoding="utf-8"))
        p = (d.get("localmatte", {}).get("presets", {}) or {}).get(preset, {}) or {}
    except Exception:
        p = {}
    return str(p.get("matting", "matanyone")), dict(p.get("params", {}) or {})

# engine defaults (defaults.json -> "localmatte" overrides via `params`)
DEFAULTS: Dict[str, Any] = {
    "max_side": 1024, "pad_px": 48, "fg_erode_px": 2, "ring_in_px": 8, "ring_out_px": 24,
    "patch_px": 32, "chroma_min_range": 2.0,
    "hybrid_lo": 0.05, "hybrid_hi": 0.95, "chroma_uniform_std": 6.0,
    "connect_thr": 0.02, "connect_reach_px": 200,
}


@dataclass
class LocalMatteResult:
    alpha: np.ndarray                      # full-frame float32: the zone's alpha (0 outside `area`)
    area: np.ndarray                       # full-frame bool: where the zone result wins (= region)
    trimap: np.ndarray                     # full-frame uint8: 0 BG, 128 unknown, 255 FG (0 outside the crop)
    box: Tuple[int, int, int, int]         # crop x0, y0, x1, y1 (plate px, exclusive end)
    scale: float                           # crop -> working scale (<= 1)
    engines: Dict[str, np.ndarray] = field(default_factory=dict)   # engine -> full-frame alpha inside the unknown
    weights: Dict[str, np.ndarray] = field(default_factory=dict)   # engine -> full-frame weight (auto)
    stats: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)                 # QA hints ("niezgodność silników", ...)


def _p(params: Optional[Dict[str, Any]], key: str) -> Any:
    return (params or {}).get(key, DEFAULTS[key])


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _bool(m: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if m is None:
        return np.zeros(shape, bool)
    return m.astype(bool)


def _resize_mask(m: np.ndarray, size: Tuple[int, int], thr: float = 0.5) -> np.ndarray:
    if m.shape[1] == size[0] and m.shape[0] == size[1]:
        return m.astype(bool)
    return cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA) > thr


def lab_of(img_bgr: np.ndarray) -> np.ndarray:
    """Lab (L 0..100) of a display BGR float image (same conversion as the "color" zone criterion)."""
    return cv2.cvtColor(np.clip(img_bgr, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)


# ----------------------------------------------------------------------------- trimap
@dataclass
class Trimap:
    fg: np.ndarray          # bool (crop scale)
    unknown: np.ndarray     # bool
    forced_bg: np.ndarray   # bool: "-" strokes / holdout / distractors / removed, inside the crop
    core: np.ndarray        # bool: the (unresized-threshold) core at crop scale
    region: np.ndarray      # bool: the zone area at crop scale

    def as_float(self) -> np.ndarray:
        """pymatting trimap: 1 = FG, 0 = BG, 0.5 = unknown."""
        t = np.zeros(self.fg.shape, np.float64)
        t[self.unknown] = 0.5
        t[self.fg] = 1.0
        return t

    def as_u8(self) -> np.ndarray:
        t = np.zeros(self.fg.shape, np.uint8)
        t[self.unknown] = 128
        t[self.fg] = 255
        return t


SOFT_FG_P = 0.95   # soft trimap (SAM3 probabilities): FG where p >= this
SOFT_BG_P = 0.05   # ... unknown where SOFT_BG_P < p < SOFT_FG_P


def build_trimap(core: np.ndarray, region: np.ndarray, forced_bg: np.ndarray, fg_erode_px: int = 2,
                 prob: Optional[np.ndarray] = None, band_unknown: bool = False) -> Trimap:
    """FG = core eroded, unknown = region \\ FG \\ forced BG, BG = rest (incl. forced BG).
    `prob` (SAM 3.1 probability 0..1, consistent with the core; user decision 2026-10-14): the trimap from SAM 3.1's own
    uncertainty instead of the erosion – FG = core ∩ p >= 0.95, unknown = region ∪ (0.05 < p < 0.95) ∪ the rest of the
    core rim; `fg_erode_px` unused.
    `band_unknown` (whole-outline zones, user decision 2026-10-25): the region IS the unknown – no pixel of it is sure
    FG, whatever the erosion or SAM 3.1's confidence says (the band points' inner whiskers reach deeper than the inset,
    where SAM 3.1 took background for hair); the sure FG is the core outside the region."""
    core = core.astype(bool)
    region = region.astype(bool)
    if prob is not None:
        p = np.asarray(prob, np.float32)
        fg = core & (p >= SOFT_FG_P)
        uncertain = (p > SOFT_BG_P) & (p < SOFT_FG_P)
    else:
        fg = cv2.erode(core.astype(np.uint8), _disk(fg_erode_px)).astype(bool) if fg_erode_px > 0 else core.copy()
        fg &= core
        uncertain = np.zeros_like(core)
    if band_unknown:
        fg &= ~region
        if not fg.any() and core.any() and not (core & ~region).any():
            # „Wcięcie: cały rdzeń” (2026-10-29): the band covers the whole core – its middle (the ridge of depth) is
            # the sure FG seed ViTMatte needs (the composition keeps the inset feather around it, engine/bandsys `whole`)
            from engine.bandsys import core_middle
            fg = core_middle(core)
    forced = forced_bg.astype(bool) & ~core          # forced BG never touches the core (semantic errors = SAM3 key)
    # the eroded 2 px rim of the core is unknown everywhere (also outside the region): never a BG seed next to FG
    unknown = (region | core | uncertain) & ~fg & ~forced
    return Trimap(fg=fg, unknown=unknown, forced_bg=forced, core=core, region=region)


# ----------------------------------------------------------------------------- engines
def bg_ring(tm: Trimap, ring_in: int, ring_out: int) -> np.ndarray:
    """Background pixels 8-24 px outside the unknown: not core, not unknown, not forced BG (foreign colours)."""
    u = tm.unknown.astype(np.uint8)
    ring = cv2.dilate(u, _disk(ring_out)).astype(bool) & ~cv2.dilate(u, _disk(ring_in)).astype(bool)
    ring &= ~tm.core & ~tm.unknown & ~tm.forced_bg
    if not ring.any():   # unknown reaches the crop border everywhere: any free background of the crop
        ring = ~tm.core & ~tm.unknown & ~tm.forced_bg
    return ring


def masked_local_stats(lab: np.ndarray, mask: np.ndarray, ksize: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel mean Lab and std of the colour vector (sqrt of the summed channel variances = ΔE-like) over `mask`
    pixels in a ksize box; (mean HxWx3, std HxW with NaN where < 4 pixels, count HxW)."""
    m = mask.astype(np.float32)
    k = (int(ksize) | 1, int(ksize) | 1)
    n = cv2.boxFilter(m, -1, k, normalize=False)
    sum1 = cv2.boxFilter(lab * m[..., None], -1, k, normalize=False)
    sum2 = cv2.boxFilter((lab * lab) * m[..., None], -1, k, normalize=False)
    nn = np.maximum(n, 1.0)[..., None]
    mean = (sum1 / nn).astype(np.float32)
    var = np.clip(sum2 / nn - mean * mean, 0, None)
    std = np.sqrt(var.sum(axis=2)).astype(np.float32)
    bad = n < 4
    std[bad] = np.nan
    mean[bad] = np.nan
    return mean, std, n


def fill_nan_nearest(a: np.ndarray, fallback: Any) -> np.ndarray:
    """NaN -> value of the nearest finite pixel (distance transform labels); all NaN -> fallback."""
    bad = ~np.isfinite(a if a.ndim == 2 else a[..., 0])
    if not bad.any():
        return a
    out = a.copy()
    if bad.all():
        out[...] = fallback
        return out
    _, lab = cv2.distanceTransformWithLabels(bad.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(~bad)
    out[bad] = a[ys, xs][lab[bad] - 1]     # label i = i-th finite pixel in raster order
    return out


def chroma_alpha(img: np.ndarray, tm: Trimap, params: Optional[Dict[str, Any]] = None,
                 bg_color_bgr: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Colour-difference key of the unknown: ΔE (Lab, display) from the LOCAL background colour (mean of the background
    pixels in a 2 x patch + 1 window around every pixel – "half the hair on a bright, half on a dark wall" is handled
    per patch; pipette = one global colour), black point lo = local background noise (1.45 x local std ≈ 90th percentile
    of a 3-D normal), white point hi = ΔE between the local foreground colour (core next to the unknown) and the local
    background. `chroma_lo` / `chroma_hi` in params override lo / hi globally. Returns (alpha crop float32, info)."""
    lab = lab_of(img)
    ring = bg_ring(tm, int(_p(params, "ring_in_px")), int(_p(params, "ring_out_px")))
    win = 2 * int(_p(params, "patch_px")) + 1
    bgpx = ~tm.core & ~tm.unknown & ~tm.forced_bg
    glob = np.median(lab[ring].reshape(-1, 3), axis=0).astype(np.float32) if ring.any() else np.array([50, 0, 0], np.float32)
    if bg_color_bgr is not None:
        ref = lab_of(np.asarray(bg_color_bgr, np.float32).reshape(1, 1, 3))[0, 0]
        mean_bg = np.broadcast_to(ref, lab.shape).astype(np.float32)
        src, chip = "pipeta", ref
        _, std_bg, _ = masked_local_stats(lab, bgpx, win)
    elif bgpx.any():
        mean_bg, std_bg, _ = masked_local_stats(lab, bgpx, win)
        mean_bg = fill_nan_nearest(mean_bg, glob)
        src, chip = "pierścień BG (lokalnie)", glob
    else:
        mean_bg = np.broadcast_to(glob, lab.shape).astype(np.float32)
        std_bg = np.full(lab.shape[:2], np.nan, np.float32)
        src, chip = "brak tła (domyślny szary)", glob
    d_ring = np.linalg.norm(lab - glob[None, None, :], axis=2).astype(np.float32)
    ring_std = float(d_ring[ring].std()) if ring.any() else None
    std_bg = fill_nan_nearest(std_bg, ring_std if ring_std is not None else 0.0)
    d = np.linalg.norm(lab - mean_bg, axis=2).astype(np.float32)
    lo = np.maximum(1.45 * std_bg, 0.5).astype(np.float32)
    near_fg = tm.fg & cv2.dilate(tm.unknown.astype(np.uint8), _disk(int(_p(params, "ring_in_px")))).astype(bool)
    if near_fg.any():
        mean_fg, _, _ = masked_local_stats(lab, near_fg, 2 * win + 1)
        fg_glob = np.median(lab[near_fg].reshape(-1, 3), axis=0).astype(np.float32)
        mean_fg = fill_nan_nearest(mean_fg, fg_glob)
        hi = np.linalg.norm(mean_fg - mean_bg, axis=2).astype(np.float32)
    elif tm.unknown.any():
        hi = np.full(lab.shape[:2], float(np.percentile(d[tm.unknown], 95.0)), np.float32)
    else:
        hi = lo + 1.0
    hi = np.maximum(hi, lo + float(_p(params, "chroma_min_range")))
    lo_u = params.get("chroma_lo") if params else None   # user black / white point (plate ΔE units)
    hi_u = params.get("chroma_hi") if params else None
    if lo_u is not None:
        lo = np.full(lab.shape[:2], float(lo_u), np.float32)
    if hi_u is not None:
        hi = np.maximum(np.full(lab.shape[:2], float(hi_u), np.float32), lo + 1e-3)
    a = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    bgr = cv2.cvtColor(np.asarray(chip, np.float32).reshape(1, 1, 3), cv2.COLOR_Lab2BGR)[0, 0]
    u = tm.unknown if tm.unknown.any() else np.ones(lab.shape[:2], bool)
    info = {"bg_lab": [round(float(v), 2) for v in chip], "bg_source": src,
            "lo": round(float(np.median(lo[u])), 3), "hi": round(float(np.median(hi[u])), 3),
            "bg_hex": "#" + "".join(f"{int(round(float(np.clip(c, 0, 1)) * 255)):02x}" for c in bgr[::-1]),
            "ring_px": int(ring.sum()), "ring_std": round(ring_std, 3) if ring_std is not None else None}
    return a, info


def clean_alpha(img: np.ndarray, clean: np.ndarray, tm: Trimap, params: Optional[Dict[str, Any]] = None
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Difference key against the clean plate (docs/PLAN_CLEANPLATE.md): d = ΔE(pixel, background in the same place);
    lo = local noise of d on the background pixels (1.45 x std ≈ 90th pct), hi = ΔE(local foreground colour next to the
    unknown, clean plate there). A line / texture that is in the clean plate keys out without any "-" stroke; hair over
    it keeps its own alpha. Returns (alpha crop float32, info)."""
    lab = lab_of(img)
    labc = lab_of(clean)
    d = np.linalg.norm(lab - labc, axis=2).astype(np.float32)
    win = 2 * int(_p(params, "patch_px")) + 1
    bgpx = ~tm.core & ~tm.unknown & ~tm.forced_bg
    if bgpx.any():
        _, std_d, _ = masked_local_stats(d[..., None].repeat(3, axis=2) / np.sqrt(3.0), bgpx, win)   # std of d itself
        mean_d, _, _ = masked_local_stats(d[..., None].repeat(3, axis=2) / 3.0, bgpx, win)
        glob_std = float(np.nanmedian(std_d)) if np.isfinite(std_d).any() else 0.0
        std_d = fill_nan_nearest(std_d, glob_std)
        mean_d = fill_nan_nearest(mean_d, np.array([0, 0, 0], np.float32)).sum(axis=2)
    else:
        std_d = np.zeros(d.shape, np.float32)
        mean_d = np.zeros(d.shape, np.float32)
    lo = np.maximum(mean_d + 1.45 * std_d, 0.5).astype(np.float32)
    near_fg = tm.fg & cv2.dilate(tm.unknown.astype(np.uint8), _disk(int(_p(params, "ring_in_px")))).astype(bool)
    if near_fg.any():
        mean_fg, _, _ = masked_local_stats(lab, near_fg, 2 * win + 1)
        fg_glob = np.median(lab[near_fg].reshape(-1, 3), axis=0).astype(np.float32)
        mean_fg = fill_nan_nearest(mean_fg, fg_glob)
        hi = np.linalg.norm(mean_fg - labc, axis=2).astype(np.float32)
    elif tm.unknown.any():
        hi = np.full(d.shape, float(np.percentile(d[tm.unknown], 95.0)), np.float32)
    else:
        hi = lo + 1.0
    hi = np.maximum(hi, lo + float(_p(params, "chroma_min_range")))
    a = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    u = tm.unknown if tm.unknown.any() else np.ones(d.shape, bool)
    info = {"clean_lo": round(float(np.median(lo[u])), 3), "clean_hi": round(float(np.median(hi[u])), 3),
            "clean_bg_resid": round(float(np.median(d[bgpx])), 3) if bgpx.any() else None}
    return a, info


def _pymatting(img: np.ndarray, trimap: np.ndarray, kind: str) -> np.ndarray:
    import pymatting
    rgb = np.ascontiguousarray(np.clip(img[..., ::-1], 0, 1), dtype=np.float64)
    fn = pymatting.estimate_alpha_knn if kind == "knn" else pymatting.estimate_alpha_cf
    a = fn(rgb, np.ascontiguousarray(trimap, dtype=np.float64))
    return np.clip(a, 0, 1).astype(np.float32)


def matting_alpha(img: np.ndarray, tm: Trimap, kind: str = "knn") -> np.ndarray:
    """pymatting KNN / closed-form on the crop trimap; unknown pixels only (FG = 1, BG = 0 elsewhere)."""
    if not tm.unknown.any():
        return tm.fg.astype(np.float32)
    bg = ~tm.fg & ~tm.unknown
    if not tm.fg.any() or not bg.any():
        # pymatting needs both FG and BG samples; a crop without one of them (key drawn away from the core, or all object)
        # has no colour evidence for the unknown band: no FG -> 0 there, no BG -> 1 there (step 9 robustness fix)
        a = np.zeros(tm.unknown.shape, np.float32)
        a[tm.unknown] = 0.0 if not tm.fg.any() else 1.0
        a[tm.fg] = 1.0
        return a
    a = _pymatting(img, tm.as_float(), kind)
    a[tm.fg] = 1.0
    a[~tm.fg & ~tm.unknown] = 0.0
    return a


def hybrid_alpha(img: np.ndarray, tm: Trimap, params: Optional[Dict[str, Any]] = None,
                 bg_color_bgr: Optional[Tuple[float, float, float]] = None,
                 kind: str = "knn") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Chroma first; pixels it is sure about (α < lo / > hi) become BG / FG and knn solves only the rest."""
    ac, info = chroma_alpha(img, tm, params, bg_color_bgr)
    lo, hi = float(_p(params, "hybrid_lo")), float(_p(params, "hybrid_hi"))
    ring_std = info.get("ring_std")
    if bg_color_bgr is None and (ring_std is None or ring_std > float(_p(params, "chroma_uniform_std"))):
        # motley background: the colour difference from one "background colour" means nothing – plain knn
        a = matting_alpha(img, tm, kind)
        return a, {**info, "hybrid_unknown_px": int(tm.unknown.sum()), "hybrid_sure_fg_px": 0, "hybrid_sure_bg_px": 0,
                   "chroma_skipped": "tło niejednolite (odch. std pierścienia %.1f ΔE)" % (ring_std or 0.0)}
    sure_fg = tm.unknown & (ac >= hi)
    sure_bg = tm.unknown & (ac <= lo)
    rest = tm.unknown & ~sure_fg & ~sure_bg
    info = {**info, "hybrid_unknown_px": int(rest.sum()), "hybrid_sure_fg_px": int(sure_fg.sum()),
            "hybrid_sure_bg_px": int(sure_bg.sum())}
    if not rest.any():
        return ac, info
    tm2 = Trimap(fg=tm.fg | sure_fg, unknown=rest, forced_bg=tm.forced_bg | sure_bg, core=tm.core, region=tm.region)
    ak = matting_alpha(img, tm2, kind)
    a = ak.copy()
    a[tm.fg] = 1.0
    a[sure_fg] = 1.0
    a[sure_bg] = 0.0
    return a.astype(np.float32), info


# ----------------------------------------------------------------------------- connectivity (SPEC 7.7 pkt 4, soft)
def connect_strands(alpha: np.ndarray, core: np.ndarray, thr: float = 0.02, reach_px: int = 200) -> np.ndarray:
    """Keep only alpha in components (α > thr, 8-connected) touching the core and within `reach_px` of it."""
    cand = (alpha > thr) & ~core
    if not cand.any():
        return alpha
    if not core.any():
        return np.where(cand, 0.0, alpha).astype(np.float32)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    cand &= dist <= reach_px
    n, lab = cv2.connectedComponents((cand | core).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[core])] = True
    keep[0] = False
    ok = keep[lab]
    out = alpha.copy()
    out[~core & ~ok] = 0.0
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- main entry
def local_matte(img: np.ndarray, core: np.ndarray, region: np.ndarray, bg_mask: Optional[np.ndarray] = None,
                holdout: Optional[np.ndarray] = None, engine: str = "auto", preset: str = "hair",
                params: Optional[Dict[str, Any]] = None, hints: Optional[Dict[str, Any]] = None,
                connect: bool = True, clean: Optional[np.ndarray] = None,
                clean_conf: Optional[np.ndarray] = None) -> LocalMatteResult:
    """Zone matting on one frame. `img` = display BGR float32 (0..1) full frame; `core`, `region`, `bg_mask`
    ("-" strokes ∪ accepted distractors ∪ Remove-zone pixels), `holdout` (front objects) = full-frame bool;
    `clean` = clean plate (same size / colour space as `img`) + `clean_conf` (0..1) -> engine `clean`."""
    t0 = time.time()
    if engine not in ENGINES:
        raise ValueError(f"nieznany silnik strefy: {engine} (dozwolone: {', '.join(ENGINES)})")
    if engine == "clean" and clean is None:
        raise ValueError("silnik „clean” wymaga clean plate'u (projektowego albo auto – „Policz z ujęcia”)")
    if engine == "matanyone":
        raise ValueError("silnik „matanyone” liczy całą sekwencję strefy (zones.matanyone_zone), nie jedną klatkę")
    if engine == "vitmatte":
        raise ValueError("silnik „vitmatte” liczy na kadrze strefy (zones.vitmatte_zone), nie przez local_matte")
    H, W = img.shape[:2]
    core = _bool(core, (H, W))
    region = _bool(region, (H, W))
    forced_full = _bool(bg_mask, (H, W)) | _bool(holdout, (H, W))
    res = LocalMatteResult(alpha=np.zeros((H, W), np.float32), area=region.copy(),
                           trimap=np.zeros((H, W), np.uint8), box=(0, 0, 0, 0), scale=1.0)
    if not region.any():
        res.stats = {"empty": True, "seconds": 0.0}
        return res
    pad = int(_p(params, "pad_px"))
    ys, xs = np.where(region)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1)
    cw, ch = x1 - x0, y1 - y0
    s = min(1.0, float(_p(params, "max_side")) / max(cw, ch))
    sw, sh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    crop = img[y0:y1, x0:x1]
    if s < 1.0:
        crop = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    clean_s = conf_s = None
    if clean is not None:
        clean_s = clean[y0:y1, x0:x1]
        if s < 1.0:
            clean_s = cv2.resize(clean_s, (sw, sh), interpolation=cv2.INTER_AREA)
        cf = clean_conf if clean_conf is not None else np.ones((H, W), np.float32)
        conf_s = cf[y0:y1, x0:x1].astype(np.float32)
        if s < 1.0:
            conf_s = cv2.resize(conf_s, (sw, sh), interpolation=cv2.INTER_AREA)
    core_s = _resize_mask(core[y0:y1, x0:x1], (sw, sh))
    region_s = _resize_mask(region[y0:y1, x0:x1], (sw, sh))
    forced_s = _resize_mask(forced_full[y0:y1, x0:x1], (sw, sh))
    fg_erode = int(round(int(_p(params, "fg_erode_px")) * s)) if s < 1.0 else int(_p(params, "fg_erode_px"))
    tm = build_trimap(core_s, region_s, forced_s, fg_erode)
    bg_color = (params or {}).get("bg_color")
    bg_color = tuple(float(v) for v in bg_color) if bg_color else None
    engines: Dict[str, np.ndarray] = {}
    weights: Dict[str, np.ndarray] = {}
    info: Dict[str, Any] = {}
    tim: Dict[str, float] = {}
    if engine == "chroma":
        t = time.time(); a, info = chroma_alpha(crop, tm, params, bg_color); tim["chroma"] = time.time() - t
        engines["chroma"] = a
    elif engine in ("knn", "cf"):
        t = time.time(); a = matting_alpha(crop, tm, engine); tim[engine] = time.time() - t
        engines[engine] = a
    elif engine == "hybrid":
        t = time.time(); a, info = hybrid_alpha(crop, tm, params, bg_color); tim["hybrid"] = time.time() - t
        engines["hybrid"] = a
    elif engine == "clean":
        t = time.time(); a, info = clean_alpha(crop, clean_s, tm, params); tim["clean"] = time.time() - t
        engines["clean"] = a
    else:  # auto: per-patch routing + seamless blend (engine/localmatte_route.py, plan section 4)
        from engine.localmatte_route import route_and_blend
        a, engines, weights, rinfo = route_and_blend(crop, tm, preset, params, hints, bg_color, tim,
                                                     clean=clean_s, clean_conf=conf_s)
        info.update(rinfo)
        res.flags.extend(rinfo.get("flags", []))
    a = a.astype(np.float32)
    a[tm.fg] = 1.0
    a[tm.forced_bg & ~tm.fg] = 0.0
    a[~tm.unknown & ~tm.fg] = 0.0
    if connect:
        a = connect_strands(a, tm.core | tm.fg, float(_p(params, "connect_thr")), int(round(int(_p(params, "connect_reach_px")) * s)))
    # back to full resolution, result only inside the region
    def up(m: np.ndarray, interp: int) -> np.ndarray:
        return m if s >= 1.0 else cv2.resize(m, (cw, ch), interpolation=interp)
    full_a = np.zeros((H, W), np.float32)
    full_a[y0:y1, x0:x1] = up(a, cv2.INTER_LINEAR)
    full_a[~region] = 0.0
    full_a[region & core] = 1.0   # the core is the object – exact 1 also where s < 1 blurred its edge
    full_a[region & forced_full & ~core] = 0.0
    res.alpha = full_a
    tri = np.zeros((H, W), np.uint8)
    tri[y0:y1, x0:x1] = up(tm.as_u8(), cv2.INTER_NEAREST)
    res.trimap = tri
    res.box, res.scale = (x0, y0, x1, y1), s
    for k, m in engines.items():
        fm = np.zeros((H, W), np.float32)
        fm[y0:y1, x0:x1] = up(m.astype(np.float32), cv2.INTER_LINEAR)
        res.engines[k] = fm
    for k, m in weights.items():
        fm = np.zeros((H, W), np.float32)
        fm[y0:y1, x0:x1] = up(m.astype(np.float32), cv2.INTER_LINEAR)
        res.weights[k] = fm
    res.stats = {"engine": engine, "preset": preset, "box": list(res.box), "scale": round(s, 4),
                 "cleanplate": clean is not None,
                 "unknown_px": int(tm.unknown.sum()), "fg_px": int(tm.fg.sum()), "forced_bg_px": int(tm.forced_bg.sum()),
                 "times": {k: round(v, 3) for k, v in tim.items()}, "seconds": round(time.time() - t0, 3), **info}
    return res
