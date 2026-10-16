# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/localmatte_route.py - automatic engine choice per fragment of the zone and a seamless blend
(docs/PLAN_HAIR_TOOL.md section 4, step 2).

- weight map per pixel (32 px patches, smoothed): local background homogeneity (std of ΔE of the background around
  each pixel) -> `chroma` score 1 - std / sigma_ref; `knn` = constant baseline; preset weights (`defaults.json ->
  localmatte.presets`) and `hints` (prompt / VLM, step 9-10) multiply the scores; softmax with temperature; an engine
  with weight 0 is off (a single remaining engine = its result bitwise, as if forced);
- every engine computes on the whole unknown (quality first; restricting to its fragments + 32 px is a later speedup);
- feather of the weights (Gaussian, >= 16 px), level calibration of engine B to A (gain / offset, LSQ on the overlap
  band = transition where both weights > 0.2, at 0.05 < α < 0.95), α = Σ wᵢ αᵢ;
- disagreement |α_A - α_B| > 0.25 on > 20 % of the overlap band -> the more confident engine wins there (chroma:
  background homogeneity, knn: decisiveness of its alpha); both unsure -> hybrid (knn initialised by chroma) + flag;
- seam QA: mean |∇α| on the weight boundaries vs inside the fragments (ratio <= 1.5) and the engine map for the UI.
Measures not used yet (need more engines): edge blur width and motion -> `matanyone` / `ltx` (step 11).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.localmatte import (DEFAULTS, Trimap, bg_ring, chroma_alpha, clean_alpha, fill_nan_nearest, hybrid_alpha,
                               lab_of, masked_local_stats, matting_alpha)

ROUTE_DEFAULTS: Dict[str, Any] = {
    "patch_px": 32, "sigma_ref": 8.0, "temperature": 0.05, "feather_px": 16, "knn_baseline": 0.5,
    "overlap_min_w": 0.2, "disagree_thr": 0.25, "disagree_frac": 0.2, "seam_max_ratio": 1.5,
}
PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "hair":   {"engines": {"clean": 1.2, "chroma": 1.0, "knn": 1.0}},
    "fibres": {"engines": {"clean": 1.2, "chroma": 0.8, "knn": 1.0, "backlit": 1.0}},
    "smoke":  {"engines": {"clean": 1.3, "chroma": 1.3, "knn": 0.6}},
    "motion": {"engines": {"clean": 1.0, "chroma": 0.7, "knn": 1.0, "matanyone": 1.0}},
    "thin":   {"engines": {"clean": 1.2, "chroma": 1.0, "knn": 1.0}, "keep_islands": True},
}
AVAILABLE = ("clean", "chroma", "knn")   # engines this module can run today (backlit / matanyone / ltx: step 11)


def localmatte_defaults() -> Dict[str, Any]:
    """defaults.json -> "localmatte": {params..., "presets": {...}} merged over the built-ins."""
    out: Dict[str, Any] = {**DEFAULTS, **ROUTE_DEFAULTS, "presets": {k: json.loads(json.dumps(v)) for k, v in PRESET_DEFAULTS.items()}}
    try:
        d = json.loads((Path(__file__).resolve().parent.parent / "defaults.json").read_text(encoding="utf-8"))
        lm = d.get("localmatte") or {}
        for k, v in lm.items():
            if k == "presets":
                for name, pv in (v or {}).items():
                    cur = out["presets"].setdefault(name, {"engines": {}})
                    cur.update({kk: vv for kk, vv in pv.items() if kk != "engines"})
                    cur["engines"] = {**cur.get("engines", {}), **(pv.get("engines") or {})}
            else:
                out[k] = v
    except (OSError, ValueError):
        pass
    return out


def _rp(params: Optional[Dict[str, Any]], key: str) -> Any:
    return (params or {}).get(key, ROUTE_DEFAULTS[key])


def homogeneity_map(img: np.ndarray, tm: Trimap, patch_px: int, ring_in: int, ring_out: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Local background std of ΔE around every pixel (background = BG ring pixels + free background of the crop,
    never core / unknown / forced BG); patches without background take the nearest measured value."""
    lab = lab_of(img)
    bgpx = ~tm.core & ~tm.unknown & ~tm.forced_bg
    _, std, _ = masked_local_stats(lab, bgpx, 2 * patch_px + 1)
    ring = bg_ring(tm, ring_in, ring_out)
    glob = float(np.nanmedian(std[ring])) if ring.any() and np.isfinite(std[ring]).any() else float(np.nanmedian(std)) if np.isfinite(std).any() else 0.0
    std = fill_nan_nearest(std, glob)
    std = cv2.GaussianBlur(std, (0, 0), patch_px / 2.0)
    return std, {"bg_std_median": round(glob, 3), "bg_std_unknown_median": round(float(np.median(std[tm.unknown])), 3) if tm.unknown.any() else None}


def engine_weights(std_map: np.ndarray, tm: Trimap, preset: Dict[str, Any], params: Optional[Dict[str, Any]],
                   hints: Optional[Dict[str, Any]], clean_score: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Softmax weights of the available engines per pixel of the unknown (0 outside). `clean_score` (0..1 per pixel:
    clean-plate confidence x background agreement) enables the `clean` engine; None = no clean plate."""
    sigma_ref = float(_rp(params, "sigma_ref"))
    T = max(1e-3, float(_rp(params, "temperature")))
    eng_w = {k: float(v) for k, v in (preset.get("engines") or {}).items() if k in AVAILABLE}
    if clean_score is None:
        eng_w.pop("clean", None)
    for k, v in (hints or {}).items():            # prompt / VLM multipliers (e.g. "plamy kolorowe" -> chroma 0.3)
        if k in eng_w:
            eng_w[k] = eng_w[k] * float(v)
    eng_w = {k: v for k, v in eng_w.items() if v > 0}
    if not eng_w:
        eng_w = {"knn": 1.0}
    scores: Dict[str, np.ndarray] = {}
    if "clean" in eng_w:
        scores["clean"] = np.clip(clean_score, 0.0, 1.0).astype(np.float32) * eng_w["clean"]
    if "chroma" in eng_w:
        scores["chroma"] = np.clip(1.0 - std_map / sigma_ref, 0.0, 1.0) * eng_w["chroma"]
    if "knn" in eng_w:
        scores["knn"] = np.full(std_map.shape, float(_rp(params, "knn_baseline")) * eng_w["knn"], np.float32)
    names = list(scores)
    if len(names) == 1:
        return {names[0]: tm.unknown.astype(np.float32)}
    stack = np.stack([scores[n] / T for n in names])
    stack -= stack.max(axis=0, keepdims=True)
    e = np.exp(stack)
    w = e / e.sum(axis=0, keepdims=True)
    return {n: (w[i] * tm.unknown).astype(np.float32) for i, n in enumerate(names)}


def feather_weights(weights: Dict[str, np.ndarray], unknown: np.ndarray, feather_px: float) -> Dict[str, np.ndarray]:
    """Gaussian feather (>= 16 px) of the weight maps, renormalised to sum 1 on the unknown."""
    if len(weights) == 1:
        k = next(iter(weights))
        return {k: unknown.astype(np.float32)}
    sig = max(16.0, float(feather_px)) / 2.0
    um = unknown.astype(np.float32)
    out = {k: cv2.GaussianBlur(v, (0, 0), sig) for k, v in weights.items()}
    norm = cv2.GaussianBlur(um, (0, 0), sig)
    tot = sum(out.values())
    tot = np.where(tot > 1e-6, tot, 1.0)
    return {k: (v / tot * um).astype(np.float32) for k, v in out.items()}


def calibrate(a_ref: np.ndarray, a_other: np.ndarray, band: np.ndarray, min_n: int = 50) -> Tuple[float, float, int]:
    """Gain / offset of `a_other` to `a_ref` on the overlap band (both in transition 0.05..0.95), clipped."""
    sel = band & (a_ref > 0.05) & (a_ref < 0.95) & (a_other > 0.05) & (a_other < 0.95)
    n = int(sel.sum())
    if n < min_n:
        return 1.0, 0.0, n
    x, y = a_other[sel].astype(np.float64), a_ref[sel].astype(np.float64)
    A = np.column_stack([x, np.ones_like(x)])
    (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.clip(g, 0.5, 2.0)), float(np.clip(o, -0.3, 0.3)), n


def seam_qa(alpha: np.ndarray, weights: Dict[str, np.ndarray], unknown: np.ndarray) -> Dict[str, Any]:
    """Mean |∇α| on the weight boundaries (0.2 < w < 0.8) vs inside fragments (w > 0.9)."""
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g = np.sqrt(gx * gx + gy * gy)
    if len(weights) < 2:
        return {"seam_ratio": None, "boundary_px": 0}
    wmax = np.max(np.stack(list(weights.values())), axis=0)
    boundary = unknown & (wmax > 0.2) & (wmax < 0.8)
    inside = unknown & (wmax > 0.9)
    if boundary.sum() < 20 or inside.sum() < 20:
        return {"seam_ratio": None, "boundary_px": int(boundary.sum())}
    gb, gi = float(g[boundary].mean()), float(g[inside].mean())
    return {"seam_ratio": round(gb / max(gi, 1e-4), 3), "boundary_px": int(boundary.sum()),
            "grad_boundary": round(gb, 4), "grad_inside": round(gi, 4)}


def clean_score_map(img: np.ndarray, clean: np.ndarray, conf: np.ndarray, tm: Trimap, patch_px: int,
                    sigma_ref: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Where the clean plate can be trusted: its confidence x (1 - local background residual ΔE(img, clean) / sigma_ref);
    the residual is measured on free background pixels, patches without background take the nearest value."""
    d = np.linalg.norm(lab_of(img) - lab_of(clean), axis=2).astype(np.float32)
    bgpx = ~tm.core & ~tm.unknown & ~tm.forced_bg
    if bgpx.any():
        m, _, _ = masked_local_stats(d[..., None].repeat(3, axis=2) / 3.0, bgpx, 2 * patch_px + 1)
        glob = float(np.median(d[bgpx]))
        resid = fill_nan_nearest(m, np.array([glob / 3.0] * 3, np.float32)).sum(axis=2)
    else:
        resid = np.full(d.shape, sigma_ref, np.float32)
    resid = cv2.GaussianBlur(resid, (0, 0), patch_px / 2.0)
    score = np.clip(conf, 0, 1) * np.clip(1.0 - resid / sigma_ref, 0.0, 1.0)
    return score.astype(np.float32), {"clean_resid_median": round(float(np.median(resid[tm.unknown])), 3) if tm.unknown.any() else None,
                                      "clean_conf_median": round(float(np.median(conf[tm.unknown])), 3) if tm.unknown.any() else None}


def route_and_blend(img: np.ndarray, tm: Trimap, preset_name: str, params: Optional[Dict[str, Any]],
                    hints: Optional[Dict[str, Any]], bg_color: Optional[Tuple[float, float, float]],
                    tim: Dict[str, float], clean: Optional[np.ndarray] = None,
                    clean_conf: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Returns (alpha crop, engine alphas, weights, info)."""
    cfg = localmatte_defaults()
    preset = dict(cfg["presets"].get(preset_name) or cfg["presets"]["hair"])
    if params:
        preset["engines"] = {**preset.get("engines", {}), **(params.get("engines") or {})}
    p = {**cfg, **(params or {})}
    info: Dict[str, Any] = {"preset_engines": preset.get("engines"), "flags": []}
    if not tm.unknown.any():
        return tm.fg.astype(np.float32), {}, {}, info
    t = time.time()
    std_map, hinfo = homogeneity_map(img, tm, int(p["patch_px"]), int(p["ring_in_px"]), int(p["ring_out_px"]))
    info.update(hinfo)
    cscore = None
    if clean is not None:
        cscore, cinfo = clean_score_map(img, clean, clean_conf if clean_conf is not None else np.ones(img.shape[:2], np.float32),
                                        tm, int(p["patch_px"]), float(p["sigma_ref"]))
        info.update(cinfo)
    w0 = engine_weights(std_map, tm, preset, p, hints, cscore)
    weights = feather_weights(w0, tm.unknown, float(p["feather_px"]))
    tim["routing"] = time.time() - t
    engines: Dict[str, np.ndarray] = {}
    if "clean" in weights:
        t = time.time(); a, kinfo = clean_alpha(img, clean, tm, p); tim["clean"] = time.time() - t
        engines["clean"] = a
        info.update(kinfo)
    if "chroma" in weights:
        t = time.time(); a, cinfo = chroma_alpha(img, tm, p, bg_color); tim["chroma"] = time.time() - t
        engines["chroma"] = a
        info.update({k: v for k, v in cinfo.items()})
    if "knn" in weights:
        t = time.time(); engines["knn"] = matting_alpha(img, tm, "knn"); tim["knn"] = time.time() - t
    names = list(weights)
    info["engine_share"] = {n: round(float(weights[n][tm.unknown].mean()), 3) for n in names}
    if len(names) == 1:
        return engines[names[0]], engines, weights, info
    # calibration of every other engine B to A (largest share) on their overlap band = the feathered transition where
    # neither dominates; disagreement on that band -> the more confident of the pair wins there
    a_name = max(names, key=lambda n: info["engine_share"][n])
    info["calibration"] = {}
    info["disagreement_frac"] = {}
    for b_name in [n for n in names if n != a_name]:
        band = tm.unknown & (weights[a_name] > float(p["overlap_min_w"])) & (weights[b_name] > float(p["overlap_min_w"]))
        g, o, n = calibrate(engines[a_name], engines[b_name], band)
        cal = np.clip(engines[b_name] * g + o, 0, 1).astype(np.float32)
        cal[~tm.unknown] = engines[b_name][~tm.unknown]
        engines[b_name] = cal
        info["calibration"][b_name] = {"ref": a_name, "gain": round(g, 4), "offset": round(o, 4), "samples": n}
        if not band.any():
            continue
        diff = np.abs(engines[a_name] - engines[b_name])
        frac = float((diff[band] > float(p["disagree_thr"])).mean())
        info["disagreement_frac"][b_name] = round(frac, 3)
        if frac > float(p["disagree_frac"]):
            conf = {}
            for n in (a_name, b_name):
                if n == "chroma":
                    conf[n] = float(np.clip(1.0 - std_map[band] / float(p["sigma_ref"]), 0, 1).mean())
                elif n == "knn":
                    ak = engines["knn"][band]
                    conf[n] = float(1.0 - 2.0 * np.minimum(ak, 1.0 - ak).mean())
                elif n == "clean" and cscore is not None:
                    conf[n] = float(cscore[band].mean())
            info.setdefault("confidence", {})[b_name] = {k: round(v, 3) for k, v in conf.items()}
            win = max(conf, key=conf.get)
            if conf[win] < 0.5:
                t = time.time(); hy, _ = hybrid_alpha(img, tm, p, bg_color); tim["hybrid"] = time.time() - t
                for n in (a_name, b_name):
                    engines[n] = np.where(band, hy, engines[n]).astype(np.float32)
                info["flags"].append("niezgodność silników %s/%s (%.0f %% pasa), oba niepewne → knn z inicjalizacją chroma"
                                     % (a_name, b_name, 100 * frac))
                info.setdefault("disagreement_winner", {})[b_name] = "hybrid"
            else:
                lose = b_name if win == a_name else a_name
                weights[win] = np.where(band, weights[win] + weights[lose], weights[win]).astype(np.float32)
                weights[lose] = np.where(band, 0.0, weights[lose]).astype(np.float32)
                weights = feather_weights(weights, tm.unknown, float(p["feather_px"]))
                info.setdefault("disagreement_winner", {})[b_name] = win
    alpha = np.zeros(tm.unknown.shape, np.float32)
    for n in names:
        alpha += weights[n] * engines[n]
    alpha[tm.fg] = 1.0
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    sq = seam_qa(alpha, weights, tm.unknown)
    info.update(sq)
    if sq.get("seam_ratio") is not None and sq["seam_ratio"] > float(p["seam_max_ratio"]):
        info["flags"].append("szew między silnikami: |∇α| na granicy %.2f × wnętrze" % sq["seam_ratio"])
    return alpha, engines, weights, info
