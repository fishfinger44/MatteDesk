# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen copy (git 7b2f49d, 2026-10-25) of the temporal smoothing functions BEFORE the sliding-window / parallel-flow
rewrite (docs/PLAN_EDGE_SPEED.md A1-A3). Reference for tests/test_temporal_window_identity.py: the new code must
return bit-identical arrays. Do not edit."""
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

TEMPORAL_CONF_MODES = ("drop", "centre")

def _median_odd(win: np.ndarray) -> np.ndarray:
    """Element-wise median of an odd number of frames (axis 0) – the middle element of the sorted values, which is
    exactly what np.median returns for an odd count (bit-identical), via min/max sorting networks for 3 and 5
    (no partition along the slow axis: 4x faster on 279 x 704^2); other counts -> np.median."""
    n = win.shape[0]
    mn, mx = np.minimum, np.maximum
    if n == 3:
        a, b, c = win
        return mx(mn(a, b), mn(mx(a, b), c))
    if n == 5:
        a, b, c, d, e = win
        # 5-element median network (7 comparators), values are only selected -> exact
        f, g = mn(a, b), mx(a, b)
        h, i = mn(c, d), mx(c, d)
        # min(f, h) is the minimum and max(g, i) the maximum of {a, b, c, d}: neither can be the median of five
        m1, m2 = mx(f, h), mn(g, i)          # the two middle candidates of {a, b, c, d}
        p, q = mn(m1, m2), mx(m1, m2)
        return mx(p, mn(q, e))               # median of {p, q, e} with p <= q
    return np.median(win, axis=0)


def temporal_median(alphas: np.ndarray, radius: int, motion_limit: float, band: Optional[np.ndarray] = None,
                    progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Per-pixel median over +-radius frames, skipped where the alpha range in the window is large (real
    motion -> no ghosts, lesson MBA) – soft gate like CoreCombine's motion_limit. Only inside `band` if given."""
    n = alphas.shape[0]
    out = alphas.copy()
    for i in range(n):
        if progress is not None:
            progress(i + 1, n)
        idx = [min(n - 1, max(0, j)) for j in range(i - radius, i + radius + 1)]
        win = alphas[idx]
        med = _median_odd(win)
        # range over the window incl. frame i (as MatAnyone2CoreCombine): thin parts moving faster than their
        # width must not be mistaken for glitches. Large one-frame jumps are a core problem -> key, not smoothing.
        rng = win.max(axis=0) - win.min(axis=0)
        calm = 1.0 - np.clip((rng - motion_limit * 0.6) / (motion_limit * 0.4), 0.0, 1.0)
        sm = alphas[i] + (med - alphas[i]) * calm
        out[i] = sm if band is None else np.where(band[i], sm, alphas[i])
    return out


def _drop_flows(cache: Dict[Tuple[int, int], Any], lo: int) -> None:
    """Frame i of a temporal window needs the pairs (i, j) / (j, i) with |i - j| <= radius; the frames go up, so a pair
    with a frame below i - radius is never used again: dropped (the flows of every pair of a 4K shot do not fit in RAM)."""
    for k in [k for k in cache if min(k) < lo]:
        del cache[k]


def temporal_median_flow(alphas: np.ndarray, grays: np.ndarray, radius: int = 2, band: Optional[np.ndarray] = None,
                         occl_px: float = 1.0, motion_limit: float = 0.35,
                         preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) -> np.ndarray:
    """Motion-compensated temporal median (experiment 2026-10-13, ViTMatte zone engine): for frame i the alphas of
    frames i±1..i±radius are warped INTO frame i by DIS optical flow measured on `grays` (same crop as `alphas`,
    float 0..1 or uint8) and the per-pixel median of the ALIGNED window is taken, with the same soft motion gate as
    `temporal_median` (range of the aligned window > motion_limit -> the frame keeps its value: a wrong flow on a
    smooth background, or a real one-frame change, is never "repaired" into a ghost; motion_limit <= 0 = plain
    median). A neighbour whose flow fails the forward/backward check (|f_ij + f_ji(warped)| > occl_px: occlusion /
    unreliable) contributes the centre value instead, so it never votes. Only inside `band` if given. Frames at the
    ends use the available neighbours (window clamped as in temporal_median)."""
    n, h, w = alphas.shape
    if n < 3:
        return alphas.copy()
    dis = cv2.DISOpticalFlow_create(preset)
    u8 = [cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for g in grays]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    flows: Dict[Tuple[int, int], np.ndarray] = {}

    def flow(i: int, j: int) -> np.ndarray:        # for every pixel of frame i: where it is in frame j
        k = (i, j)
        if k not in flows:
            flows[k] = dis.calc(u8[i], u8[j], None)
        return flows[k]

    out = alphas.copy()
    for i in range(n):
        _drop_flows(flows, i - radius)   # pairs no later frame needs (RAM: 4K x2 flows, user report 2026-10-22)
        samples = []
        for d in range(-radius, radius + 1):
            j = min(n - 1, max(0, i + d))
            if j == i:
                samples.append(alphas[i])
                continue
            fij, fji = flow(i, j), flow(j, i)
            mx, my = gx + fij[..., 0], gy + fij[..., 1]
            warped = cv2.remap(alphas[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            back = cv2.remap(fji, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            occl = np.linalg.norm(fij + back, axis=2) > occl_px
            samples.append(np.where(occl, alphas[i], warped).astype(np.float32))
        win = np.stack(samples)
        med = _median_odd(win)
        if motion_limit > 0:
            rng = win.max(axis=0) - win.min(axis=0)
            calm = 1.0 - np.clip((rng - motion_limit * 0.6) / (motion_limit * 0.4), 0.0, 1.0)
            med = alphas[i] + (med - alphas[i]) * calm
        out[i] = med if band is None else np.where(band[i], med, alphas[i])
    return out


def _min_eig(g: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smaller eigenvalue of the structure tensor of `g` (0..1): ~0 on a smooth area / a single straight edge (the flow
    there is a guess), large on a corner / texture."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    k = int(round(sigma * 3)) * 2 + 1
    jxx = cv2.GaussianBlur(gx * gx, (k, k), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (k, k), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (k, k), sigma)
    return np.maximum(0.5 * (jxx + jyy - np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)), 0.0).astype(np.float32)


def temporal_mean_flow(alphas: np.ndarray, grays: np.ndarray, radius: int = 3, sigma_t: Optional[float] = None,
                       tol: float = 0.25, occl_px: float = 1.0,
                       preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                       fb_sigma: float = 0.0, photo_sigma: float = 0.0, tex_k: float = 0.0, conf_mode: str = "drop",
                       multiscale: int = 1) -> np.ndarray:
    """Motion-compensated, median-anchored weighted MEAN (user decision 2026-10-15: stronger temporal stabilisation of the
    ViTMatte zone alpha). As temporal_median_flow the frames i±1..i±radius are warped into frame i by DIS flow (a
    neighbour failing the forward/backward check contributes the centre value); then each aligned sample gets the weight
    gauss(d / sigma_t) * clip(1 - |sample - median of the window| / tol, 0, 1): samples near the ROBUST centre average
    out the per-frame noise, an outlier – garbage present in one frame only, the centre included – gets weight ~0 (no
    smearing of garbage onto the neighbours, no ghost of a real change: the median follows it). The caller clips the
    garbage (levels) BEFORE this, so a zeroed speck is never averaged back in. sigma_t default = radius.

    Flow confidence (experiment 2026-10-15, all off by default = bit-identical): with any of them on, a neighbour enters
    the MEAN with its own warped value and an extra weight c = c_fb * c_photo * c_tex (the median anchor keeps the binary
    fwd/bwd substitution): c_fb = gauss(|f_ij + f_ji| / fb_sigma) (fb_sigma 0 -> binary: 0 above occl_px),
    c_photo = gauss(|gray_i - warped gray_j| / photo_sigma), c_tex = t / (t + tex_k) with t = min eigenvalue of the
    structure tensor of gray_i (a smooth background, where DIS only interpolates, does not vote). conf_mode "drop" =
    the lost weight is dropped (the other samples decide), "centre" = it goes to the frame's own value. multiscale 2 =
    the flow is also measured on the half-size grays (3: also quarter ...), per pixel the one with the smaller fwd/bwd
    error is used (also a
    confidence mode: with nothing else on, c = the binary fwd/bwd test)."""
    n, h, w = alphas.shape
    if n < 3:
        return alphas.copy()
    if conf_mode not in TEMPORAL_CONF_MODES:
        raise ValueError(f"conf_mode musi być jednym z {TEMPORAL_CONF_MODES}, jest {conf_mode!r}")
    conf = fb_sigma > 0 or photo_sigma > 0 or tex_k > 0 or multiscale > 1
    st = float(sigma_t) if sigma_t else max(0.5, float(radius))     # near-flat window: ~1.5x the median's efficiency
    dis = cv2.DISOpticalFlow_create(preset)
    u8 = [cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for g in grays]
    pyr = [[cv2.resize(u, (max(8, w >> k), max(8, h >> k)), interpolation=cv2.INTER_AREA) for u in u8]
           for k in range(1, max(1, int(multiscale)))]          # half, quarter ... size grays
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    flows: Dict[Tuple[int, int], np.ndarray] = {}
    fbs: Dict[Tuple[int, int], np.ndarray] = {}

    def fb_err(fij: np.ndarray, fji: np.ndarray) -> np.ndarray:
        back = cv2.remap(fji, gx + fij[..., 0], gy + fij[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.linalg.norm(fij + back, axis=2)

    def flow(i: int, j: int) -> np.ndarray:        # for every pixel of frame i: where it is in frame j
        k = (i, j)
        if k not in flows:
            flows[k] = dis.calc(u8[i], u8[j], None)
        return flows[k]

    def flow_ms(i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:   # (flow i->j, its fwd/bwd error), best of 2 scales
        k = (i, j)
        if k not in fbs:
            f1, b1 = flow(i, j), flow(j, i)
            e1 = fb_err(f1, b1)
            for lv in pyr:
                sc = np.array([w / lv[0].shape[1], h / lv[0].shape[0]], np.float32)
                hs = lambda a, b: cv2.resize(dis.calc(lv[a], lv[b], None), (w, h), interpolation=cv2.INTER_LINEAR) * sc
                f2, b2 = hs(i, j), hs(j, i)
                e2 = fb_err(f2, b2)
                pick = e2 < e1
                f1 = np.where(pick[..., None], f2, f1)
                e1 = np.where(pick, e2, e1)
            fbs[k] = (f1, e1)
        return fbs[k]

    tex = None
    out = alphas.copy()
    for i in range(n):
        _drop_flows(flows, i - radius)   # pairs no later frame needs (RAM: 4K x2 flows, user report 2026-10-22)
        _drop_flows(fbs, i - radius)
        samples, tw, conf_w, raw = [], [], [], []
        if conf and tex_k > 0:
            t = _min_eig(u8[i].astype(np.float32) / 255.0)
            tex = t / (t + float(tex_k))
        for d in range(-radius, radius + 1):
            j = min(n - 1, max(0, i + d))
            tw.append(float(np.exp(-0.5 * (d / st) ** 2)))
            if j == i:
                samples.append(alphas[i])
                if conf:
                    raw.append(alphas[i])
                    conf_w.append(np.ones((h, w), np.float32))
                continue
            if conf:
                fij, err = flow_ms(i, j)
                mx, my = gx + fij[..., 0], gy + fij[..., 1]
                warped = cv2.remap(alphas[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                occl = err > occl_px
                c = (np.exp(-0.5 * (err / fb_sigma) ** 2) if fb_sigma > 0 else (~occl).astype(np.float32))
                if photo_sigma > 0:
                    gw = cv2.remap(u8[j].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    c = c * np.exp(-0.5 * ((gw - u8[i].astype(np.float32)) / (255.0 * photo_sigma)) ** 2)
                if tex is not None:
                    c = c * tex
                conf_w.append(c.astype(np.float32))
                raw.append(warped.astype(np.float32))
                samples.append(np.where(occl, alphas[i], warped).astype(np.float32))
                continue
            fij, fji = flow(i, j), flow(j, i)
            mx, my = gx + fij[..., 0], gy + fij[..., 1]
            warped = cv2.remap(alphas[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            back = cv2.remap(fji, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            occl = np.linalg.norm(fij + back, axis=2) > occl_px
            samples.append(np.where(occl, alphas[i], warped).astype(np.float32))
        win = np.stack(samples)
        med = _median_odd(win)
        if conf:
            rw = np.stack(raw)
            cw = np.stack(conf_w)
            twa = np.asarray(tw, np.float32)[:, None, None]
            close = np.clip(1.0 - np.abs(rw - med[None]) / max(tol, 1e-3), 0.0, 1.0)
            wgt = close * twa * cw
            if conf_mode == "centre":       # the weight a doubtful neighbour loses goes to the frame's own value
                ci = radius
                lost = (twa * (1.0 - cw)).sum(axis=0)
                wgt[ci] = wgt[ci] + lost * close[ci]
            s = wgt.sum(axis=0)
            mean = np.where(s > 1e-6, (wgt * rw).sum(axis=0) / np.maximum(s, 1e-6), med)
        else:
            wgt = np.clip(1.0 - np.abs(win - med[None]) / max(tol, 1e-3), 0.0, 1.0) * np.asarray(tw, np.float32)[:, None, None]
            s = wgt.sum(axis=0)
            mean = np.where(s > 1e-6, (wgt * win).sum(axis=0) / np.maximum(s, 1e-6), med)
        out[i] = np.clip(mean, 0.0, 1.0).astype(np.float32)
        for k_ in [k_ for k_ in flows if min(k_) < i - radius]:
            flows.pop(k_, None)
        for k_ in [k_ for k_ in fbs if min(k_) < i - radius]:
            fbs.pop(k_, None)
    return out


