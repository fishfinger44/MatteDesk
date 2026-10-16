# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion blur of the alpha (stage Eksperymenty, user decision 2026-10-16).

MatAnyone2 makes a soft but narrow edge (10-90 % ~4-5 px) whatever the motion – on the moving hand of CLIP2_A the plate's
transition is 12.5-17 px (PROGRESS 2026-10-16), and a wider edge band does not help. So the blur of a moving object is
added by a separate step on an existing version: a vector blur of the alpha along the object's OWN motion.

1. Velocity (px / frame) of the object: dense optical flow of the plate (DIS on gray, at most `FLOW_SIDE` px on the long
   side), centred: (flow f->f+1 - flow f->f-1) / 2 (one side at the shot's ends). Taken only inside the object (alpha >
   0.5, eroded – at the edge the flow mixes with the background), smoothed there (normalised Gaussian) and extended
   outward to the nearest object pixel, so the pixels just outside the object carry its motion too (the blur reaches
   out of the object, like a vector blur on a dilated velocity pass). The background's own motion is never used.
2. Blur vector per pixel = velocity x shutter / 360 x strength (strength 1 = the physical blur of the manifest's
   shutter); the alpha is averaged along it (box over the exposure, `n` samples, bilinear). Where the vector is shorter
   than `min_px` the alpha stays as it is (static parts of the object stay sharp), with a 1 px ramp in between.
RGB is never touched (alpha only).
GPU (config `mb_device` auto | gpu | cpu, env MD_MB_DEVICE): the vector blur runs in the GPU worker of the plate alignment
(engine/align_gpu_worker.py `vector_blur`, ComfyUI's torch) on a crop around the moving part; the flow stays here
(DIS at <= 1280 px, cheap). GPU = CPU within the bilinear rounding of cv2.remap (tests/test_motion_blur.py)."""
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

FLOW_SIDE = 1280          # flow computed at <= this many px on the long side (velocity scaled back)
MB_DEFAULTS = {"strength": 1.0, "min_px": 1.0, "max_px": 64.0}
STRENGTH_RANGE = (0.0, 10.0)   # field: any value in this range (slider 0-2)
SHUTTER_RANGE = (0.0, 720.0)   # shutter angle override (deg; the manifest's shutter_deg by default), > 360 for experiments


def clamp_strength(v: float) -> float:
    return float(min(STRENGTH_RANGE[1], max(STRENGTH_RANGE[0], float(v))))


def clamp_shutter(v: float) -> float:
    return float(min(SHUTTER_RANGE[1], max(SHUTTER_RANGE[0], float(v))))


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Plate frame (display-referred float BGR 0..1, or uint8 / gray) -> uint8 gray for the flow."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def object_velocity(mask: np.ndarray, g: np.ndarray, g_prev: Optional[np.ndarray], g_next: Optional[np.ndarray],
                    reach_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity field (vx, vy) of the object, full resolution, px / frame: the object's own flow inside it, extended
    to the nearest object pixel up to `reach_px` (full-res px) outside it, 0 elsewhere (see the module doc)."""
    h, w = mask.shape
    vs, s = velocity_small(mask, g, g_prev, g_next, reach_px)
    if vs is None:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    vf = vs
    if s < 1.0:
        vf = cv2.resize(vs, (w, h), interpolation=cv2.INTER_LINEAR) / s
    return np.ascontiguousarray(vf[..., 0]), np.ascontiguousarray(vf[..., 1])


def velocity_small(mask: np.ndarray, g: np.ndarray, g_prev: Optional[np.ndarray], g_next: Optional[np.ndarray],
                   reach_px: float) -> Tuple[Optional[np.ndarray], float]:
    """(velocity at flow scale (sh, sw, 2) in flow-scale px / frame, scale s) – `object_velocity` before the upsample;
    (None, s) when there is nothing to measure."""
    h, w = mask.shape
    s = min(1.0, FLOW_SIDE / float(max(w, h)))
    if not mask.any() or (g_prev is None and g_next is None):
        return None, s
    sw, sh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    small = lambda x: cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1.0 else x
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gs = small(g)
    vs = []
    if g_next is not None:
        vs.append(dis.calc(gs, small(g_next), None))
    if g_prev is not None:
        vs.append(-dis.calc(gs, small(g_prev), None))
    v = (sum(vs) / len(vs)).astype(np.float32)
    m = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) if s < 1.0 else mask.astype(np.uint8)
    er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if int(er.sum()) < 20:
        er = m
    if not er.any():
        return None, s
    # smooth inside the object (normalised convolution: only object pixels vote)
    wt = er.astype(np.float32)
    den = cv2.GaussianBlur(wt, (0, 0), 6.0)
    vin = np.empty_like(v)
    for c in range(2):
        num = cv2.GaussianBlur(v[..., c] * wt, (0, 0), 6.0)
        vin[..., c] = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    # extend outward: velocity of the nearest object pixel (within reach)
    src = (er == 0).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.where(src == 0)
    lut_y = np.zeros(int(lab.max()) + 1, np.int32)
    lut_x = np.zeros_like(lut_y)
    lut_y[lab[zy, zx]] = zy
    lut_x[lab[zy, zx]] = zx
    vf = vin[lut_y[lab], lut_x[lab]]
    vf[dist > reach_px * s + 2.0] = 0.0
    vf = cv2.GaussianBlur(vf, (0, 0), 2.0)            # no seams between the nearest-pixel cells
    return np.ascontiguousarray(vf.astype(np.float32)), s


def vector_blur(alpha: np.ndarray, vx: np.ndarray, vy: np.ndarray, k: float, min_px: float = 1.0,
                max_px: float = 64.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alpha averaged along the blur vector k x (vx, vy) of every pixel (box over the exposure), only where the vector
    is >= `min_px` (1 px ramp below it). Vectors longer than `max_px` are shortened to it."""
    a = alpha.astype(np.float32)
    lx, ly = vx * k, vy * k
    ln = np.hypot(lx, ly)
    over = ln > max_px
    if over.any():
        f = np.where(over, max_px / np.maximum(ln, 1e-6), 1.0).astype(np.float32)
        lx, ly, ln = lx * f, ly * f, np.minimum(ln, max_px)
    wgt = np.clip(ln - (min_px - 1.0), 0.0, 1.0)
    act = wgt > 0
    stats = {"moving_px": int((ln >= min_px).sum()), "max_px": round(float(ln.max()) if ln.size else 0.0, 2),
             "samples": 0}
    if not act.any():
        return a, stats
    ys, xs = np.where(act)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    lmax = float(ln[y0:y1, x0:x1].max())
    n = int(min(65, max(2, np.ceil(lmax) + 1)))
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    bx, by = lx[y0:y1, x0:x1], ly[y0:y1, x0:x1]
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for i in range(n):
        t = (i + 0.5) / n - 0.5
        acc += cv2.remap(a, gx + t * bx, gy + t * by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = a.copy()
    wb = wgt[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = a[y0:y1, x0:x1] + wb * (acc / n - a[y0:y1, x0:x1])
    stats["samples"] = n
    return np.clip(out, 0.0, 1.0), stats


def vector_blur_gpu(gpu: Any, alpha: np.ndarray, vs: np.ndarray, s: float, k: float, min_px: float,
                    max_px: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """`vector_blur` in the GPU worker (`gpu` = engine.align_gpu.GpuAligner) on the crop around the moving part:
    the moving region found at flow scale, grown by the interpolation reach and half the longest blur (the samples)."""
    a = alpha.astype(np.float32)
    h, w = a.shape
    ln = np.minimum(np.hypot(vs[..., 0], vs[..., 1]) / s * k, max_px)
    act = ln > max(0.05, min_px - 1.0)        # weight < 0.05 of a < 0.05 px blur outside: no visible change
    if not act.any():
        return a, {"moving_px": 0, "max_px": round(float(ln.max()) if ln.size else 0.0, 2), "samples": 0}
    ys, xs = np.where(act)
    m = int(np.ceil(float(ln.max()) * 0.5)) + 2 + int(np.ceil(2.0 / s))
    x0, x1 = max(0, int(np.floor(xs.min() / s)) - m), min(w, int(np.ceil((xs.max() + 1) / s)) + m)
    y0, y1 = max(0, int(np.floor(ys.min() / s)) - m), min(h, int(np.ceil((ys.max() + 1) / s)) + m)
    tmp = gpu.tmp
    pa, pv, po = tmp / "mb_alpha.npy", tmp / "mb_vel.npy", tmp / "mb_out.npy"
    np.save(pa, np.ascontiguousarray(a[y0:y1, x0:x1]))
    np.save(pv, vs.astype(np.float32))
    resp = gpu.call({"op": "vector_blur", "alpha": str(pa), "vel": str(pv), "out": str(po), "x0": x0, "y0": y0,
                     "plate_w": w, "plate_h": h, "scale": s, "k": k, "min_px": min_px, "max_px": max_px})
    out = a.copy()
    out[y0:y1, x0:x1] = np.load(po)
    return out, {"moving_px": int(resp.get("moving_px", 0)), "max_px": float(resp.get("max_px", 0.0)),
                 "samples": int(resp.get("samples", 0)), "device": "gpu"}


def motion_blur_alpha(alpha: np.ndarray, g: np.ndarray, g_prev: Optional[np.ndarray], g_next: Optional[np.ndarray],
                      shutter_deg: float, strength: float, min_px: float = MB_DEFAULTS["min_px"],
                      max_px: float = MB_DEFAULTS["max_px"], gpu: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One frame: alpha (full frame, 0..1) + the plate's gray of f, f-1, f+1 (None at the shot's ends) -> the alpha
    with the object's motion blur. strength 0 = unchanged. `gpu`: an open GPU worker (engine.align_gpu) = the vector
    blur there."""
    k = float(shutter_deg) / 360.0 * float(strength)
    if k <= 0:
        return alpha.astype(np.float32), {"moving_px": 0, "max_px": 0.0, "samples": 0}
    reach = max_px * 0.5 + 2.0
    if gpu is not None:
        vs, s = velocity_small(alpha > 0.5, g, g_prev, g_next, reach)
        if vs is None:
            return alpha.astype(np.float32), {"moving_px": 0, "max_px": 0.0, "samples": 0, "device": "gpu"}
        return vector_blur_gpu(gpu, alpha, vs, s, k, min_px, max_px)
    vx, vy = object_velocity(alpha > 0.5, g, g_prev, g_next, reach_px=reach)
    return vector_blur(alpha, vx, vy, k, min_px, max_px)


def mb_device(requested: Optional[str] = None) -> Tuple[str, str]:
    """('gpu' | 'cpu', note) for „Zastosuj motion blur”: `requested` > env MD_MB_DEVICE (tests: cpu) > config
    `mb_device` (auto). auto = GPU when ComfyUI's Python has CUDA and ComfyUI's queue is idle."""
    import os
    from engine import align_gpu
    requested = requested or os.environ.get("MD_MB_DEVICE")
    return align_gpu.resolve_device(requested, key="mb_device")
