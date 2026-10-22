# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/edge_mode.py - Automatic choice of the edge source (SPEC 5.5) + routing metrics (plan 5.2).

Metrics on the SAM 3.1 core and the plate-to-mask display image (sampled frames):
- soft_frac : share of edge normals whose 10-90 % transition on the plate is wider than 3 px (soft/blurred edges)
- thin_score: share of the core area that disappears under an opening with a 3-px disc (antennae, wires, fingers)
- motion_px : 95th percentile of the core centroid / bbox motion per frame
- dark_edge : share of edge samples with contrast < 3x noise (dark on dark)
Rules: backlit (exists + "Użyj backlit" + the object really blocks the light on it: backlit_cover >= 0.5) > ltx (semi_transparent / volumetric / defocused / thin_parts flags,
workflow available, no backlit) > matanyone (soft_frac >= 0.25 or a person/hair/fur prompt) > core (rigid or soft_frac
< 0.25). Borderline (soft_frac within +-20 % of 0.25) -> both candidates on a test batch, lower flicker wins.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SOFT_THR = 0.25
HUMAN = re.compile(r"\b(person|people|man|woman|men|women|girl|boy|child|kid|human|hair|fur|face|actor|actress|dancer)\b", re.I)
LTX_FLAGS = {"semi_transparent", "volumetric", "defocused", "thin_parts"}
MODE_LABEL = {"backlit": "Backlit", "matanyone": "MatAnyone2", "core": "Ostra (rdzeń)", "ltx": "LTX-2.5 (półprzezroczyste)"}


def _edge_samples(core: np.ndarray, step: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    cnts, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts, nrm = [], []
    for c in cnts:
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 10:
            continue
        for i in range(0, len(c), step):
            p0, p1 = c[i - 2], c[(i + 2) % len(c)]
            t = p1 - p0
            n = np.array([t[1], -t[0]], np.float32)
            ln = np.linalg.norm(n)
            if ln > 0:
                pts.append(c[i])
                nrm.append(n / ln)
    return np.array(pts, np.float32).reshape(-1, 2), np.array(nrm, np.float32).reshape(-1, 2)


def edge_profiles(img_gray: np.ndarray, core: np.ndarray, half: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Plate luma profiles along edge normals: (profiles N x (2*half+1), noise estimate)."""
    pts, nrm = _edge_samples(core)
    if len(pts) == 0:
        return np.zeros((0, 2 * half + 1), np.float32), 0.0
    ts = np.arange(-half, half + 1, dtype=np.float32)
    xs = (pts[:, None, 0] + nrm[:, None, 0] * ts[None]).astype(np.float32)
    ys = (pts[:, None, 1] + nrm[:, None, 1] * ts[None]).astype(np.float32)
    prof = cv2.remap(img_gray.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lap = cv2.Laplacian(img_gray.astype(np.float32), cv2.CV_32F)
    noise = 1.4826 * float(np.median(np.abs(lap - np.median(lap)))) / 2.0 + 1e-4
    return prof, noise


def transition_width(prof: np.ndarray) -> np.ndarray:
    """10-90 % width (px) of monotone-ish edge profiles; inf where there is no contrast."""
    lo, hi = prof[:, :3].mean(1), prof[:, -3:].mean(1)
    rng = hi - lo
    out = np.full(len(prof), np.inf, np.float32)
    for i, p in enumerate(prof):
        if abs(rng[i]) < 1e-6:
            continue
        q = (p - lo[i]) / rng[i]
        idx10 = np.argmax(q >= 0.1) if (q >= 0.1).any() else 0
        idx90 = np.argmax(q >= 0.9) if (q >= 0.9).any() else len(q) - 1
        out[i] = max(0.0, float(idx90 - idx10))
    return out


def routing_metrics(cores: Sequence[np.ndarray], images: Sequence[np.ndarray]) -> Dict[str, float]:
    soft, thin, dark, cents = [], [], [], []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for c, im in zip(cores, images):
        c = (c > 0.5).astype(np.uint8)
        if not c.any():
            continue
        g = im.mean(axis=2) if im.ndim == 3 else im
        prof, noise = edge_profiles(g, c)
        if len(prof):
            contrast = np.abs(prof[:, -3:].mean(1) - prof[:, :3].mean(1))
            ok = contrast > 3 * noise
            dark.append(1.0 - ok.mean())
            if ok.any():
                soft.append(float((transition_width(prof[ok]) > 3).mean()))
        opened = cv2.morphologyEx(c, cv2.MORPH_OPEN, ker)
        thin.append(float((c & ~opened.astype(bool)).sum() / max(c.sum(), 1)))
        ys, xs = np.where(c)
        cents.append((xs.mean(), ys.mean()))
    mv = np.linalg.norm(np.diff(np.array(cents), axis=0), axis=1) if len(cents) > 1 else np.zeros(1)
    return {"soft_frac": float(np.mean(soft)) if soft else 0.0, "thin_score": float(np.mean(thin)) if thin else 0.0,
            "dark_edge": float(np.mean(dark)) if dark else 0.0, "motion_px": float(np.percentile(mv, 95))}


BACKLIT_COVER_THR = 0.5
_LUM = np.array([0.2722, 0.6741, 0.0537], np.float32)  # same weights as PlateMatte BacklitCoverage (ACES AP0)


def backlit_cover(cores: Sequence[np.ndarray], backlit_imgs: Sequence[np.ndarray], off: float = 5.0) -> float:
    """Does the object's OUTLINE appear on the backlit plate? Coverage cov = 1 - lum / bright wall; along the core
    contour, cov a few px inside minus cov a few px outside, median over contour points and sampled frames.
    ~1 = the object is a silhouette on the backlit (valid matte source); ~0 = no edge there – the object is not on
    the backlit exposure, even when whatever is behind it is dark too (SH010: Figure absent from X2, dark nest /
    trees behind her -> the old darkness-only test said "blocks light")."""
    vals = []
    for c, im in zip(cores, backlit_imgs):
        c = (c > 0.5).astype(np.uint8)
        if c.sum() < 50:
            continue
        lum = im[..., :3].astype(np.float32) @ _LUM if im.ndim == 3 else im.astype(np.float32)
        wall = float(np.percentile(lum, 95))
        if wall <= 1e-4:
            continue
        cov = np.clip(1.0 - lum / wall, 0.0, 1.0)
        pts, nrm = _edge_samples(c, step=3)
        if len(pts) < 10:
            continue
        a = pts + nrm * off
        b = pts - nrm * off
        h, w = c.shape

        def at(img, q):
            x = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            return img[y, x]
        a_in = at(c, a).astype(bool)          # which side of the contour is inside the object
        cin = np.where(a_in, at(cov, a), at(cov, b))
        cout = np.where(a_in, at(cov, b), at(cov, a))
        vals.append(float(np.median(cin - cout)))
    return float(np.clip(np.median(vals), 0.0, 1.0)) if vals else 0.0


def choose_edge_mode(obj: Dict[str, Any], metrics: Dict[str, float], has_backlit: bool, ltx_available: bool
                     ) -> Dict[str, Any]:
    """Returns {"mode", "reason", "borderline", "alternative"} (Polish one-line reason for the UI)."""
    flags = set(obj.get("flags", []))
    prompt = " ".join(obj.get("detect", {}).get("prompts", []))
    sf = metrics.get("soft_frac", 0.0)
    cover = metrics.get("backlit_cover")
    note = ""
    if has_backlit and obj.get("use_backlit", True):
        if cover is None or cover >= BACKLIT_COVER_THR:
            return {"mode": "backlit", "reason": "jest backlit, obiekt zasłania światło → maska fizyczna", "borderline": False}
        note = f"backlit jest, ale obiekt go nie zasłania (pokrycie {cover * 100:.0f}%) · "
    if flags & {"semi_transparent", "volumetric", "defocused"} and ltx_available:
        f = ", ".join(sorted(flags & LTX_FLAGS))
        return {"mode": "ltx", "reason": note + f"flagi {f} → LTX-2.5", "borderline": False}
    if HUMAN.search(prompt):
        return {"mode": "matanyone", "reason": note + "człowiek / włosy w prompcie → MatAnyone2", "borderline": False}
    if "rigid" in flags:
        return {"mode": "core", "reason": note + "obiekt sztywny → ostra krawędź rdzenia", "borderline": False}
    border = abs(sf - SOFT_THR) <= 0.2 * SOFT_THR
    if sf >= SOFT_THR:
        mode, alt = "matanyone", "core"
    else:
        mode, alt = "core", "matanyone"
    return {"mode": mode, "reason": note + f"miękka krawędź {sf * 100:.0f}% → {MODE_LABEL[mode]}", "borderline": border,
            "alternative": alt if border else None}


def edge_flicker(alphas: Sequence[np.ndarray], cores: Sequence[np.ndarray], band_px: float = 6.0) -> float:
    """Mean |a_t - a_t-1| in the edge band of consecutive frames (motion-agnostic tie-break for borderline Auto)."""
    from engine.segment import sdf
    vals = []
    for t in range(1, len(alphas)):
        band = np.abs(sdf(cores[t])) <= band_px
        if band.any():
            vals.append(float(np.abs(alphas[t] - alphas[t - 1])[band].mean()))
    return float(np.mean(vals)) if vals else 0.0
