# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align_gpu_worker.py - plate alignment (engine/align.py) on the GPU, run in ComfyUI's Python (its PyTorch +
CUDA). Standalone on purpose: imports only numpy and torch - nothing from MatteDesk (run with `python -I`).

The CPU code in engine/align.py stays the reference; every step here reproduces its OpenCV / SciPy call so the
result matches it (synthetic tests: < 0.02 px). No ready-made GPU replacement exists (checked 2026-10-14: OpenCV
CUDA has no phaseCorrelate / findTransformECC, kornia's registrator optimises a different, photometric objective).
- features from luma (log-luma + Sobel 3x3 / exposure normalisation), static mask (cleanplate difference, blur,
  median + MAD, ellipse dilation);
- cv2.phaseCorrelate (Hanning window = sqrt(wr * wc), zero pad to the optimal DFT size, 5x5 weighted centroid);
- cv2.warpAffine / warpPerspective sampling (coordinates quantised to 1/32 px, cubic A = -0.75, zero border);
- GlobalAligner.evaluate_model_residual (21 px patches every 12 px, batched), select_best_model hierarchy;
- cv2.findTransformECC iteration (ECC criterion maximised by Gauss-Newton steps, input mask, 5x5 pre-blur);
- LocalPatchAligner._refine_translation (Gauss-Newton on patches, batched, cv2.getRectSubPix sampling);
- engine.align.reject_outliers (identical numpy copy: correlation response, +-8 px clamp, 8-neighbour median);
- scipy RBFInterpolator(thin_plate_spline, smoothing 0.1, degree 1) solved in float64, field on a 16 px grid,
  evaluated at the nodes of the alignment grid (engine.align_grid: cell centres, one per 16 px) and returned
  as the grid; the upsampling (align_grid.upsample) and the grid EXR are MatteDesk's.
CPU stays responsible for EXR decoding, luma, the INTER_AREA downscales and the global pre-warp of the aux
(engine.align.warp_global in the decode pool) - bit-identical inputs.

    <py> align_gpu_worker.py --check   -> "ok cuda <gpu name>" / "ok cpu" (exit 0) or the error (exit 1)
    <py> align_gpu_worker.py --serve   -> request loop, one JSON line per request on stdin, one JSON line back:
        {"op": "global", ...}       global model of one frame; may answer {"need_sift": true, "dump": {...}} and then
                                    read one {"op": "sift_models", "models": {...}} line before the final answer
        {"op": "patches", ...}      patch phase correlation + Gauss-Newton (anchor offsets of the local model)
        {"op": "local_frame", ...}  full-res refinement of tracked offsets + thin-plate field -> out .npy (2, H, W)
        {"op": "vector_blur", ...}  motion blur of the alpha (engine/motionblur.py, 2026-10-16): crop of the alpha +
                                    the object's velocity at flow scale -> blurred crop .npy
        {"op": "release"}           frees cached GPU memory;  {"op": "quit"}
    Errors: {"ok": false, "error": "..."}.

MD_ALIGN_GPU_FAKE=1: no torch, fixed fake results (protocol tests in MatteDesk's .venv); a request with
"fake_crash": true (or MD_ALIGN_GPU_FAKE_CRASH=1) exits the process, "fake_crash_once": <flag file> exits once,
"fake_sleep": s sleeps first.
"""

import json
import math
import os
import sys
import time

import numpy as np

FAKE = os.environ.get("MD_ALIGN_GPU_FAKE") == "1"
torch = None
F = None
DEV = None


def _init_torch():
    global torch, F, DEV
    if torch is None:
        import torch as _torch
        import torch.nn.functional as _f
        torch, F = _torch, _f
        DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEV


# ----------------------------------------------------------------------------------------------------- helpers

def optimal_dft_size(n):
    """cv::getOptimalDFTSize: smallest m >= n with m = 2^a 3^b 5^c."""
    m = max(1, int(n))
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def percentile(x, q):
    """np.percentile (linear) of a tensor, as a Python float."""
    n = x.numel()
    if n == 0:
        return float("nan")
    s, _ = torch.sort(x.double().reshape(-1))
    idx = q / 100.0 * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    a, b = float(s[lo]), float(s[hi])
    return a + (b - a) * (idx - lo)


def hann(h, w):
    """cv2.createHanningWindow((w, h), CV_32F) = sqrt(wr[i] * wc[j])."""
    wc = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(w, dtype=torch.float64, device=DEV) / (w - 1)))
    wr = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(h, dtype=torch.float64, device=DEV) / (h - 1)))
    return torch.sqrt(wr[:, None] * wc[None, :]).float()


def sep_filter(x, kx, ky):
    """Separable correlation of (H, W) float32 with 1-D kernels, BORDER_REFLECT_101 (cv2.filter2D / Sobel / blur)."""
    kx = torch.as_tensor(np.asarray(kx, np.float32), device=DEV)
    ky = torch.as_tensor(np.asarray(ky, np.float32), device=DEV)
    rx, ry = (len(kx) - 1) // 2, (len(ky) - 1) // 2
    y = x[None, None]
    if rx:
        y = F.pad(y, (rx, rx, 0, 0), mode="reflect")
    y = F.conv2d(y, kx.view(1, 1, 1, -1))
    if ry:
        y = F.pad(y, (0, 0, ry, ry), mode="reflect")
    y = F.conv2d(y, ky.view(1, 1, -1, 1))
    return y[0, 0]


def gaussian_kernel(ksize, sigma):
    """cv::getGaussianKernel as float32."""
    if sigma <= 0 and ksize in (1, 3, 5, 7):
        tab = {1: [1.0], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
               7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}
        return np.array(tab[ksize], np.float32)
    if sigma <= 0:
        sigma = ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) * 0.5
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(x, sigma):
    """cv2.GaussianBlur(x, (0, 0), sigma) for float32: ksize = cvRound(sigma * 8 + 1) | 1."""
    ksize = int(round(sigma * 8 + 1)) | 1
    k = gaussian_kernel(ksize, sigma)
    return sep_filter(x, k, k)


def np_gradient(p):
    """np.gradient of (N, h, w) -> (gy, gx), edge_order 1."""
    gy = torch.empty_like(p)
    gx = torch.empty_like(p)
    gy[:, 1:-1] = (p[:, 2:] - p[:, :-2]) * 0.5
    gy[:, 0] = p[:, 1] - p[:, 0]
    gy[:, -1] = p[:, -1] - p[:, -2]
    gx[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) * 0.5
    gx[:, :, 0] = p[:, :, 1] - p[:, :, 0]
    gx[:, :, -1] = p[:, :, -1] - p[:, :, -2]
    return gy, gx


# ---------------------------------------------------------------------------------------------------- features

def features(luma, mode):
    """AlignmentFeatureExtractor.extract on an already scaled luma (H, W) float32 tensor."""
    L = torch.log(torch.clamp(luma, min=1e-4) + 0.02)
    if mode == "cleanplate":
        Ld = L.double()
        m = float(Ld.mean())
        s = float(Ld.std(unbiased=False))
        return ((L - np.float32(m)) / np.float32(max(s, 1e-6))).float()
    gx = sep_filter(L, [-1.0, 0.0, 1.0], [1.0, 2.0, 1.0])
    gy = sep_filter(L, [1.0, 2.0, 1.0], [-1.0, 0.0, 1.0])
    g = torch.hypot(gx, gy)
    clip_val = percentile(g, 99.5) if g.numel() > 0 else 1.0
    c = torch.clamp(g, 0, max(clip_val, 1e-5))
    mx = float(c.max())
    return (c / mx if mx > 1e-6 else c).float()


def static_mask(lm_b, lm_a, mode, border_xy, box, kernel):
    """compute_static_mask at scale 0.5 from the INTER_AREA-resized lumas (the caller computed the integers)."""
    h, w = lm_b.shape
    mask = torch.ones((h, w), dtype=torch.float32, device=DEV)
    bx, by = border_xy
    mask[:by, :] = 0
    mask[-by:, :] = 0
    mask[:, :bx] = 0
    mask[:, -bx:] = 0
    if box is not None:
        sx0, sy0, sx1, sy1 = box
        mask[sy0:sy1, sx0:sx1] = 0
    if mode == "cleanplate":
        sel = mask > 0.5
        mb = percentile(lm_b[sel], 50)
        ma = percentile(lm_a[sel], 50)
        scale_exp = np.float32(mb / max(1e-4, ma))
        diff = torch.abs(gaussian_blur(lm_b, 4.0) - gaussian_blur(lm_a * scale_exp, 4.0))
        vals = diff[sel] if bool(sel.any()) else diff.reshape(-1)
        med = percentile(vals, 50)
        mad = 1.4826 * percentile(torch.abs(vals.double() - med), 50)
        thr = max(med + 6.0 * mad, 0.05 * percentile(lm_b, 50))
        fg = (diff > thr).float()
        k = torch.as_tensor(np.asarray(kernel, np.float32), device=DEV)
        kh, kw = k.shape
        ay, ax = kh // 2, kw // 2
        d = F.conv2d(F.pad(fg[None, None], (ax, kw - 1 - ax, ay, kh - 1 - ay)), k[None, None])[0, 0]
        mask[d > 0.5] = 0
    return mask


# ------------------------------------------------------------------------------------------- phase correlation

def _fftshift_cv(c):
    """cv's fftShift (quadrant swap with cx = cols / 2, cy = rows / 2) on (N, M, N') tensors."""
    m, n = c.shape[-2], c.shape[-1]
    cy, cx = m // 2, n // 2
    out = c.clone()
    out[:, :cy, :cx] = c[:, cy:2 * cy, cx:2 * cx]
    out[:, cy:2 * cy, cx:2 * cx] = c[:, :cy, :cx]
    out[:, :cy, cx:2 * cx] = c[:, cy:2 * cy, :cx]
    out[:, cy:2 * cy, :cx] = c[:, :cy, cx:2 * cx]
    return out


def phase_correlate(a, b, win=None):
    """cv2.phaseCorrelate(a[i], b[i], win) for batches a, b (N, h, w) float32 -> (dx, dy, response) float64 (N,)."""
    if win is not None:
        a = a * win
        b = b * win
    n, h, w = a.shape
    M, N = optimal_dft_size(h), optimal_dft_size(w)
    if (M, N) != (h, w):
        a = F.pad(a, (0, N - w, 0, M - h))
        b = F.pad(b, (0, N - w, 0, M - h))
    P = (torch.fft.fft2(a.float()) * torch.conj(torch.fft.fft2(b.float()))).to(torch.complex128)  # float32 like cv
    # cv magSpectrums + divSpectrums: C = P * |P| / (|P|^2 + FLT_EPSILON) - the epsilon damps weak frequencies
    mag = torch.abs(P)
    C = P * (mag / (mag * mag + 1.1920928955078125e-07))
    c = _fftshift_cv(torch.fft.ifft2(C).real)          # 1/(MN) normalised: response needs no further scaling
    flat = c.reshape(n, -1)
    peak = torch.argmax(flat, dim=1)
    py, px = peak // N, peak % N
    o = torch.arange(-2, 3, device=DEV)
    ys = py[:, None] + o[None, :]
    xs = px[:, None] + o[None, :]
    valid = (((ys >= 0) & (ys < M))[:, :, None] & ((xs >= 0) & (xs < N))[:, None, :]).double()
    idx = ys.clamp(0, M - 1)[:, :, None] * N + xs.clamp(0, N - 1)[:, None, :]
    vals = torch.gather(flat, 1, idx.reshape(n, -1)).reshape(n, 5, 5) * valid
    s = vals.sum(dim=(1, 2))
    sx = (vals * xs[:, None, :].double()).sum(dim=(1, 2))
    sy = (vals * ys[:, :, None].double()).sum(dim=(1, 2))
    resp = s.clone()
    s = s + 2.220446049250313e-16
    return N / 2.0 - sx / s, M / 2.0 - sy / s, resp


# ------------------------------------------------------------------------------------------------- sampling

def _cubic_coeffs(x):
    A = -0.75
    c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
    c1 = ((A + 2) * x - (A + 3)) * x * x + 1
    c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
    c3 = 1.0 - c0 - c1 - c2
    return torch.stack([c0, c1, c2, c3], dim=-1)


def _affine_fixed(Mi, h, w, nearest):
    """cv warpAffine coordinates for the src-from-dst map Mi (2x3), fixed point (AB_BITS 10, INTER_BITS 5).
    Returns int64 X, Y in 1/32 px (linear / cubic) or whole px (nearest)."""
    x = torch.arange(w, dtype=torch.float64, device=DEV)
    y = torch.arange(h, dtype=torch.float64, device=DEV)
    ad = torch.round(Mi[0][0] * x * 1024).long()
    bd = torch.round(Mi[1][0] * x * 1024).long()
    rd = 512 if nearest else 16
    X0 = torch.round((Mi[0][1] * y + Mi[0][2]) * 1024).long() + rd
    Y0 = torch.round((Mi[1][1] * y + Mi[1][2]) * 1024).long() + rd
    sh = 10 if nearest else 5
    X = torch.div(X0[:, None] + ad[None, :], 1 << sh, rounding_mode="floor")
    Y = torch.div(Y0[:, None] + bd[None, :], 1 << sh, rounding_mode="floor")
    return X, Y


def _persp_fixed(Hi, h, w, nearest):
    x = torch.arange(w, dtype=torch.float64, device=DEV)[None, :]
    y = torch.arange(h, dtype=torch.float64, device=DEV)[:, None]
    X0 = Hi[0][0] * x + Hi[0][1] * y + Hi[0][2]
    Y0 = Hi[1][0] * x + Hi[1][1] * y + Hi[1][2]
    W = Hi[2][0] * x + Hi[2][1] * y + Hi[2][2]
    s = 1.0 if nearest else 32.0
    nz = W != 0
    W = torch.where(nz, s / torch.where(nz, W, torch.ones_like(W)), torch.zeros_like(W))
    lim = 2147483647.0
    X = torch.round(torch.clamp(X0 * W, -lim - 1, lim)).long()
    Y = torch.round(torch.clamp(Y0 * W, -lim - 1, lim)).long()
    return X, Y


def _sample(src, X, Y, kind):
    """cv remap of src (H, W) or (C, H, W) float32 with its tables: 'linear' / 'cubic' (X, Y in 1/32 px) or
    'nearest'; zero border. Channels share the coordinates (one gather per tap)."""
    if src.dim() == 2:
        return _sample(src[None], X, Y, kind)[0]
    C, H, W = src.shape
    flat = src.reshape(C, -1)
    shape = (C,) + tuple(X.shape)
    if kind == "nearest":
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        v = flat[:, (Y.clamp(0, H - 1) * W + X.clamp(0, W - 1)).reshape(-1)].reshape(shape)
        return torch.where(ok, v, torch.zeros_like(v))
    ix = torch.div(X, 32, rounding_mode="floor")
    iy = torch.div(Y, 32, rounding_mode="floor")
    fx = (X - ix * 32).float() / 32.0
    fy = (Y - iy * 32).float() / 32.0
    if kind == "linear":
        cx = torch.stack([1.0 - fx, fx], dim=-1)
        cy = torch.stack([1.0 - fy, fy], dim=-1)
        offs = (0, 1)
    else:
        cx = _cubic_coeffs(fx)
        cy = _cubic_coeffs(fy)
        offs = (-1, 0, 1, 2)
    out = torch.zeros(shape, dtype=torch.float32, device=DEV)
    for i, oy in enumerate(offs):
        yy = iy + oy
        vy = (yy >= 0) & (yy < H)
        row = yy.clamp(0, H - 1) * W
        for j, ox in enumerate(offs):
            xx = ix + ox
            ok = vy & (xx >= 0) & (xx < W)
            v = flat[:, (row + xx.clamp(0, W - 1)).reshape(-1)].reshape(shape)
            out = out + torch.where(ok, v, torch.zeros_like(v)) * (cy[..., i] * cx[..., j])
    return out


def _inv_affine(M):
    return np.linalg.inv(np.vstack([np.asarray(M, np.float64), [0, 0, 1]]))[:2]


def warp_coords(model_type, M, h, w, inverse_map=False, kind="cubic"):
    """Source coordinates of cv2.warpAffine / warpPerspective for an (h, w) output (see _affine_fixed)."""
    M = np.asarray(M, np.float64)
    if model_type == "homography":
        Hi = M if inverse_map else np.linalg.inv(M)
        return _persp_fixed(Hi.tolist(), h, w, kind == "nearest")
    Mi = M if inverse_map else _inv_affine(M)
    return _affine_fixed(Mi.tolist(), h, w, kind == "nearest")


def warp(src, model_type, M, inverse_map=False, kind="cubic"):
    """cv2.warpAffine / warpPerspective(src, M, size, flags=kind [+ WARP_INVERSE_MAP]), zero border;
    src (H, W) or (C, H, W)."""
    h, w = src.shape[-2:]
    X, Y = warp_coords(model_type, M, h, w, inverse_map, kind)
    return _sample(src, X, Y, kind)


def rect_subpix(img, cx, cy, ps):
    """cv2.getRectSubPix(img, (ps, ps), (cx[i], cy[i])) for a batch -> (N, ps, ps); bilinear, replicated border.
    Centres become float32 like cv::Point2f."""
    H, W = img.shape
    cx = cx.float() - np.float32((ps - 1) * 0.5)
    cy = cy.float() - np.float32((ps - 1) * 0.5)
    fx0 = torch.floor(cx)
    fy0 = torch.floor(cy)
    ax = (cx - fx0)[:, None, None]
    ay = (cy - fy0)[:, None, None]
    r = torch.arange(ps, device=DEV)
    xs0 = fx0.long()[:, None] + r[None, :]
    ys0 = fy0.long()[:, None] + r[None, :]
    flat = img.reshape(-1)
    n = len(cx)

    def g(ys, xs):
        return flat[(ys.clamp(0, H - 1)[:, :, None] * W + xs.clamp(0, W - 1)[:, None, :]).reshape(-1)].reshape(n, ps, ps)

    v00, v01 = g(ys0, xs0), g(ys0, xs0 + 1)
    v10, v11 = g(ys0 + 1, xs0), g(ys0 + 1, xs0 + 1)
    return (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay


# --------------------------------------------------------------------------------------------- patch matching

def gn_refine(f_a, p_b, xy, d, win, iters=10):
    """LocalPatchAligner._refine_translation for a batch: xy (N, 2) int, d (N, 2) float64 start -> (N, 2)."""
    ps = p_b.shape[-1]
    cx = xy[:, 0].double() - 0.5
    cy = xy[:, 1].double() - 0.5
    gby, gbx = np_gradient(p_b)
    d = d.clone()
    active = torch.ones(len(d), dtype=torch.bool, device=DEV)
    w = win[None]
    for _ in range(iters):
        if not bool(active.any()):
            break
        ii = torch.nonzero(active).reshape(-1)
        pa = rect_subpix(f_a, cx[ii] - d[ii, 0], cy[ii] - d[ii, 1], ps)
        gy, gx = np_gradient(pa)
        gx = 0.5 * (gx + gbx[ii])
        gy = 0.5 * (gy + gby[ii])
        r = (p_b[ii] - pa) * w
        a11 = (w * gx * gx).double().sum(dim=(1, 2))
        a12 = (w * gx * gy).double().sum(dim=(1, 2))
        a22 = (w * gy * gy).double().sum(dim=(1, 2))
        b1 = (r * gx).double().sum(dim=(1, 2))
        b2 = (r * gy).double().sum(dim=(1, 2))
        det = a11 * a22 - a12 * a12
        ok = det >= 1e-9
        detc = torch.where(ok, det, torch.ones_like(det))
        ex = -(a22 * b1 - a12 * b2) / detc
        ey = -(a11 * b2 - a12 * b1) / detc
        d[ii[ok], 0] += ex[ok]
        d[ii[ok], 1] += ey[ok]
        conv = ok & (ex.abs() < 1e-3) & (ey.abs() < 1e-3)
        active[ii[~ok | conv]] = False
    return d


def match_patches(f_b, f_a, xy, d0, ps, chunk=None):
    """Per point: beauty patch (integer slice centred at xy), aux patch getRectSubPix at xy - 0.5 - d0, windowed
    phase correlation, then Gauss-Newton from d0 + r. Returns numpy (N, 5): rx, ry, resp, gn_dx, gn_dy.
    `chunk` points per batch: 4096 at 64 px, scaled with the patch area (128 px patches at 4096 filled the 24 GB
    card next to CoTracker3 on SH010, 2026-10-15)."""
    if chunk is None:
        chunk = max(256, int(4096 * (64.0 / ps) ** 2))
    win = hann(ps, ps)
    half = ps // 2
    H, W = f_b.shape
    r = torch.arange(ps, device=DEV)
    outs = []
    for s in range(0, len(xy), chunk):
        x = xy[s:s + chunk]
        dd = d0[s:s + chunk]
        rows = x[:, 1, None] - half + r[None, :]
        cols = x[:, 0, None] - half + r[None, :]
        pb = f_b.reshape(-1)[(rows[:, :, None] * W + cols[:, None, :]).reshape(-1)].reshape(len(x), ps, ps)
        pa = rect_subpix(f_a, x[:, 0].double() - 0.5 - dd[:, 0], x[:, 1].double() - 0.5 - dd[:, 1], ps)
        rx, ry, resp = phase_correlate(pa, pb, win)
        start = torch.stack([dd[:, 0] + rx, dd[:, 1] + ry], dim=1)
        start = torch.where(torch.isfinite(start).all(dim=1)[:, None], start, dd)
        g = gn_refine(f_a, pb, x, start, win)
        outs.append(torch.stack([rx, ry, resp, g[:, 0], g[:, 1]], dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 5), np.float64)


# ------------------------------------------------------------------------------------------------ global model

class GlobalGPU:
    """GlobalAligner on the GPU (same hierarchy and thresholds)."""

    def __init__(self, scale):
        self.scale = scale

    def measure_translation(self, fa, fb, mask):
        h, w = fb.shape
        dx, dy, resp = phase_correlate(fa[None], fb[None], hann(h, w) * mask)
        return float(dx[0]) / self.scale, float(dy[0]) / self.scale, float(resp[0])

    def evaluate(self, fa, fb, mask, model_type, params):
        h, w = fb.shape
        s = self.scale
        if model_type == "translation":
            M = np.array([[1, 0, params["dx"] * s], [0, 1, params["dy"] * s]], np.float32)
            wa = warp(fa, "affine", M)
        elif model_type in ("similarity", "affine"):
            M = np.array(params["matrix"], np.float32).copy()
            M[0, 2] *= s
            M[1, 2] *= s
            wa = warp(fa, "affine", M)
        elif model_type == "homography":
            Hm = np.array(params["matrix"], np.float32)
            Hs = np.diag([s, s, 1.0]) @ Hm @ np.diag([1.0 / s, 1.0 / s, 1.0])
            wa = warp(fa, "homography", Hs.astype(np.float32))
        else:
            return 0.0, 0.0
        edge = (mask > 0.5) & (fb > 0.05)
        step = max(12, int(round(24 * s)))
        pr = 10
        gy, gx = torch.meshgrid(torch.arange(step // 2, h, step, device=DEV),
                                torch.arange(step // 2, w, step, device=DEV), indexing="ij")
        valid = edge[gy, gx]
        if not bool(valid.any()):
            return 0.0, 0.0
        px, py = gx[valid], gy[valid]
        inside = (px >= pr) & (px < w - pr) & (py >= pr) & (py < h - pr)
        px, py = px[inside], py[inside]
        res = torch.zeros(0, device=DEV)
        if len(px):
            r = torch.arange(-pr, pr + 1, device=DEV)
            idx = ((py[:, None] + r[None, :])[:, :, None] * w + (px[:, None] + r[None, :])[:, None, :]).reshape(-1)
            n = len(px)
            p_b = fb.reshape(-1)[idx].reshape(n, 21, 21)
            p_w = wa.reshape(-1)[idx].reshape(n, 21, 21)
            keep = p_b.double().std(dim=(1, 2), unbiased=False) > 0.02
            if bool(keep.any()):
                dx, dy, resp = phase_correlate(p_w[keep], p_b[keep])
                ok = (resp > 0.1) & torch.isfinite(dx) & torch.isfinite(dy)
                res = (torch.hypot(dx[ok], dy[ok]) / s).float()
        if res.numel() == 0:
            diff = float(torch.abs(fb[edge] - wa[edge]).double().mean())
            return diff, diff * 2.0
        return percentile(res, 50), percentile(res, 95)

    def ecc(self, kind, fa, fb, mask, init_shift, iters=50, eps=1e-4):
        """cv2.findTransformECC(template=fa, input=fb, warp, motion, criteria, inputMask=mask) -> (rho, map) or None
        (None where OpenCV raises: NaN, lambda_d <= 0)."""
        hs, ws = fa.shape
        k5 = gaussian_kernel(5, 0)
        tmpl = sep_filter(fa, k5, k5)
        img = sep_filter(fb, k5, k5)
        pre = sep_filter((mask > 0.5).float(), k5, k5) * np.float32(0.5 / 0.95)
        pre = torch.round(pre)                         # convertTo(CV_8U): cvRound, half to even
        dxk = [-0.5, 0.0, 0.5]
        gxi = sep_filter(img, dxk, [1.0]) * pre
        gyi = sep_filter(img, [1.0], dxk) * pre
        X = torch.arange(ws, dtype=torch.float32, device=DEV)[None, :].expand(hs, ws)
        Y = torch.arange(hs, dtype=torch.float32, device=DEV)[:, None].expand(hs, ws)
        mp = np.eye(3, dtype=np.float32) if kind == "homography" else np.eye(2, 3, dtype=np.float32)
        mp[0, 2] = init_shift[0] * self.scale
        mp[1, 2] = init_shift[1] * self.scale
        mt = "homography" if kind == "homography" else "affine"
        rho, last = -1.0, -eps
        i = 1
        stack = torch.stack([img, gxi, gyi])
        while i <= iters and abs(rho - last) >= eps:
            # image and its gradients share the coordinates: one gather per tap for all three
            iw, gxw, gyw = warp(stack, mt, mp, inverse_map=True, kind="linear")
            m = warp(pre, mt, mp, inverse_map=True, kind="nearest") > 0
            cnt = int(m.sum())
            if cnt == 0:
                return None
            vi = iw[m].double()
            vt = tmpl[m].double()
            imean, tmean = float(vi.mean()), float(vt.mean())
            istd = math.sqrt(max(float((vi * vi).mean()) - imean * imean, 0.0))
            tstd = math.sqrt(max(float((vt * vt).mean()) - tmean * tmean, 0.0))
            iw = torch.where(m, iw - np.float32(imean), iw)
            tz = torch.where(m, tmpl - np.float32(tmean), torch.zeros_like(tmpl))
            tnorm = math.sqrt(cnt * tstd * tstd)
            inorm = math.sqrt(cnt * istd * istd)
            if kind == "affine":
                J = torch.stack([gxw * X, gyw * X, gxw * Y, gyw * Y, gxw, gyw])
            elif kind == "similarity":
                h0, h1 = float(mp[0, 0]), float(mp[1, 0])
                hx = -(X * h1) - (Y * h0)
                hy = (X * h0) - (Y * h1)
                J = torch.stack([gxw * hx + gyw * hy, gxw, gyw])
            else:
                h = [float(v) for v in mp.reshape(-1)]
                den = X * h[6] + Y * h[7] + 1.0
                hx = (-X * h[0] - Y * h[1] - h[2]) / den
                hy = (-X * h[3] - Y * h[4] - h[5]) / den
                g1 = gxw / den
                g2 = gyw / den
                tmp = hx * g1 + hy * g2
                J = torch.stack([g1 * X, g2 * X, tmp * X, g1 * Y, g2 * Y, tmp * Y, g1, g2])
            Jd = J.reshape(J.shape[0], -1).double()
            hess = (Jd @ Jd.T).float()
            try:
                hinv = torch.linalg.inv(hess.double()).float()
            except Exception:
                return None
            vecs = torch.stack([iw.reshape(-1), tz.reshape(-1)]).double()
            corr = float((vecs[0] * vecs[1]).sum())
            last = rho
            den_r = inorm * tnorm
            rho = corr / den_r if den_r != 0 else float("nan")
            if math.isnan(rho):
                return None
            proj = (Jd @ vecs.T).float()
            ip, tp = proj[:, 0], proj[:, 1]
            iph = hinv @ ip
            ln = inorm * inorm - float(ip.double() @ iph.double())
            ld = corr - float(tp.double() @ iph.double())
            if ld <= 0.0:
                return None
            lam = ln / ld
            err = (tz.double() * lam - iw.double()).float()
            ep = (Jd @ err.reshape(-1).double()).float()
            dp = (hinv @ ep).double().cpu().numpy()
            if kind == "affine":
                mp[0, 0] += dp[0]; mp[1, 0] += dp[1]; mp[0, 1] += dp[2]
                mp[1, 1] += dp[3]; mp[0, 2] += dp[4]; mp[1, 2] += dp[5]
            elif kind == "similarity":
                th = float(dp[0]) + math.asin(float(mp[1, 0]))
                mp[0, 2] += dp[1]
                mp[1, 2] += dp[2]
                mp[0, 0] = mp[1, 1] = math.cos(th)
                mp[1, 0] = math.sin(th)
                mp[0, 1] = -mp[1, 0]
            else:
                mp[0, 0] += dp[0]; mp[1, 0] += dp[1]; mp[2, 0] += dp[2]
                mp[0, 1] += dp[3]; mp[1, 1] += dp[4]; mp[2, 1] += dp[5]
                mp[0, 2] += dp[6]; mp[1, 2] += dp[7]
            i += 1
        return rho, mp

    def fit_ecc(self, kind, fa, fb, mask, init):
        """GlobalAligner.fit_ecc: full-scale matrix or None."""
        r = self.ecc(kind, fa, fb, mask, init)
        if r is None:
            return None
        cc, M = r
        if not (cc > 0.3) or np.isnan(M).any():
            return None
        s = self.scale
        if kind == "homography":
            return np.diag([1.0 / s, 1.0 / s, 1.0]) @ M @ np.diag([s, s, 1.0])
        M = M.copy()
        M[0, 2] /= s
        M[1, 2] /= s
        return M

    def select_best_model(self, fa, fb, mask, sift_cb):
        """GlobalAligner.select_best_model; sift_cb() -> SIFT + RANSAC models from MatteDesk (CPU)."""
        dx, dy, resp = self.measure_translation(fa, fb, mask)
        tp = {"dx": dx, "dy": dy, "response": resp}
        r_med, r_max = self.evaluate(fa, fb, mask, "translation", tp)
        if r_med < 0.3 and r_max < 0.75:
            return "translation", tp, r_med, r_max
        init = (dx, dy) if math.isfinite(dx) and math.isfinite(dy) else (0.0, 0.0)
        models = {}

        def model(kind):
            if kind not in models:
                models[kind] = self.fit_ecc(kind, fa, fb, mask, init)
                if kind == "similarity" and models[kind] is None:
                    for k in ("affine", "homography"):
                        models[k] = self.fit_ecc(k, fa, fb, mask, init)
                    for k, M in sift_cb().items():
                        if models.get(k) is None:
                            models[k] = np.asarray(M, np.float64)
            return models[kind]

        M = model("similarity")
        if M is not None:
            sp = {"matrix": M.tolist(), "dx": float(M[0, 2]), "dy": float(M[1, 2]),
                  "scale": float(math.hypot(M[0, 0], M[0, 1])),
                  "rotation_deg": float(math.degrees(math.atan2(M[1, 0], M[0, 0])))}
            s_med, s_max = self.evaluate(fa, fb, mask, "similarity", sp)
            if s_med < 0.3 and s_max < 0.75:
                return "similarity", sp, s_med, s_max
        M = model("affine")
        if M is not None:
            ap = {"matrix": M.tolist(), "dx": float(M[0, 2]), "dy": float(M[1, 2])}
            a_med, a_max = self.evaluate(fa, fb, mask, "affine", ap)
            if a_med < 0.3 and a_max < 0.75:
                return "affine", ap, a_med, a_max
        Hm = model("homography")
        if Hm is not None:
            hp = {"matrix": Hm.tolist()}
            h_med, h_max = self.evaluate(fa, fb, mask, "homography", hp)
            return "homography", hp, h_med, h_max
        return "translation", tp, r_med, r_max


# ------------------------------------------------------------------------------------------------- RBF field

def _tps(r2):
    """thin-plate r^2 log r from squared distances (0 at r = 0)."""
    pos = r2 > 0
    return torch.where(pos, 0.5 * r2 * torch.log(torch.where(pos, r2, torch.ones_like(r2))), torch.zeros_like(r2))


def _sqdist(a, b):
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


def grid_nodes(plate_w, plate_h, step=16):
    """= engine.align_grid.node_axes: gw + 2 nodes at the cell centres (k + 0.5) * w / gw - 0.5, k = -1..gw."""
    gw, gh = max(2, math.ceil(plate_w / step)), max(2, math.ceil(plate_h / step))
    # one guard node beyond each edge (align_grid.PAD) so the bicubic at the plate edge interpolates, not extrapolates
    xs = (torch.arange(-1, gw + 1, dtype=torch.float64, device=DEV) + 0.5) * (plate_w / gw) - 0.5
    ys = (torch.arange(-1, gh + 1, dtype=torch.float64, device=DEV) + 0.5) * (plate_h / gh) - 0.5
    return xs, ys


MIN_FIT_POINTS = 8
MIN_KEPT_FRAC = 0.2
OUTLIER_RESP_MIN = 0.1
OUTLIER_CLIP_PX = 8.0
OUTLIER_NBR_K = 8
OUTLIER_NBR_PX = 1.0


def reject_outliers(points, displacements, resp=None, resp_min=OUTLIER_RESP_MIN, clip_px=OUTLIER_CLIP_PX,
                    nbr_k=OUTLIER_NBR_K, nbr_px=OUTLIER_NBR_PX):
    """IDENTICAL copy of engine.align.reject_outliers (this file imports nothing from MatteDesk); change both.
    -> (keep (N,) bool, {"n", "resp", "clip", "nbr"})."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    d = np.asarray(displacements, np.float64).reshape(-1, 2)
    n = len(pts)
    keep = np.isfinite(d).all(axis=1) & np.isfinite(pts).all(axis=1)
    if resp is not None:
        r = np.asarray(resp, np.float64).reshape(-1)
        bad_resp = np.isfinite(r) & (r <= resp_min)
    else:
        bad_resp = np.zeros(n, bool)
    bad_clip = (np.abs(d) >= clip_px - 1e-3).any(axis=1) & ~bad_resp
    keep &= ~bad_resp & ~bad_clip
    n_nbr = 0
    idx = np.nonzero(keep)[0]
    if len(idx) > nbr_k:
        p, q = pts[idx], d[idx]
        dev = np.empty(len(idx), np.float64)
        for s in range(0, len(idx), 1024):
            blk = p[s:s + 1024]
            dist = ((blk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            dist[np.arange(len(blk)), s + np.arange(len(blk))] = np.inf      # not itself
            nb = np.argpartition(dist, nbr_k, axis=1)[:, :nbr_k]
            qb = q[s:s + 1024]
            med = np.median(q[nb], axis=1)
            dev_med = np.hypot(qb[:, 0] - med[:, 0], qb[:, 1] - med[:, 1])
            # a plane through the neighbours, evaluated at the point: at the plate edge the neighbour ring is
            # one-sided and the median of a sloping field is off, the plane is not; next to an outlier the plane
            # is pulled but the median is not - a point is dropped only when BOTH disagree with it
            rel = (p[nb] - blk[:, None, :]) / 100.0
            A = np.concatenate([np.ones(rel.shape[:2] + (1,)), rel], axis=2)
            AtA = A.transpose(0, 2, 1) @ A + 1e-6 * np.eye(3)
            beta = np.linalg.solve(AtA, A.transpose(0, 2, 1) @ q[nb])
            dev_plane = np.hypot(qb[:, 0] - beta[:, 0, 0], qb[:, 1] - beta[:, 0, 1])
            dev[s:s + 1024] = np.minimum(dev_med, dev_plane)
        bad = dev > nbr_px
        keep[idx[bad]] = False
        n_nbr = int(bad.sum())
    return keep, {"n": int(n), "resp": int(bad_resp.sum()), "clip": int(bad_clip.sum()), "nbr": n_nbr}


def rbf_field(points, disps, plate_w, plate_h, step=16, smoothing=0.1, resp=None, reject=True):
    """LocalPatchAligner.fit_rbf_grid -> (grid (2, gh, gw) float32 tensor, ho_med, ho_max, frac_ok, n_rejected,
    reject info); outliers dropped first (reject_outliers) like the CPU code."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    disps = np.asarray(disps, np.float64).reshape(-1, 2)
    if reject:
        keep, info = reject_outliers(points, disps, resp)
    else:
        keep, info = np.ones(len(points), bool), {"n": len(points), "resp": 0, "clip": 0, "nbr": 0}
    n_rej = int((~keep).sum())
    n_all = len(points)
    points, disps = points[keep], disps[keep]
    n = len(points)
    xs, ys = grid_nodes(plate_w, plate_h, step)
    zeros = torch.zeros((2, len(ys), len(xs)), dtype=torch.float32, device=DEV)
    if n < MIN_FIT_POINTS or n < MIN_KEPT_FRAC * n_all:
        return zeros, 0.0, 0.0, 0.0, n_rej, info
    perm = np.random.RandomState(42).permutation(n)      # = np.random.seed(42); np.random.permutation(n)
    n_ho = max(4, int(round(n * 0.1)))
    tr, ho = perm[n_ho:], perm[:n_ho]
    pts64 = points
    dsp64 = disps
    y = _t(pts64[tr])
    p = len(y)
    mins, maxs = y.min(0).values, y.max(0).values
    shift = (maxs + mins) / 2
    scl = (maxs - mins) / 2
    scl = torch.where(scl == 0, torch.ones_like(scl), scl)

    def poly(x):
        return torch.cat([torch.ones((len(x), 1), dtype=torch.float64, device=DEV), (x - shift) / scl], dim=1)

    lhs = torch.zeros((p + 3, p + 3), dtype=torch.float64, device=DEV)
    lhs[:p, :p] = _tps(_sqdist(y, y)) + smoothing * torch.eye(p, dtype=torch.float64, device=DEV)
    P = poly(y)
    lhs[:p, p:] = P
    lhs[p:, :p] = P.T
    rhs = torch.zeros((p + 3, 2), dtype=torch.float64, device=DEV)
    rhs[:p] = _t(dsp64[tr])
    try:
        coef = torch.linalg.solve(lhs, rhs)
    except Exception:                      # singular (collinear / duplicate points): = the CPU code's zero grid
        return zeros, 0.0, 0.0, 0.0, n_rej, info

    def ev(x, chunk=2048):
        return torch.cat([_tps(_sqdist(x[s:s + chunk], y)) @ coef[:p] + poly(x[s:s + chunk]) @ coef[p:]
                          for s in range(0, len(x), chunk)], dim=0)

    pred = ev(_t(pts64[ho])).cpu().numpy()
    err = np.hypot(pred[:, 0] - dsp64[ho, 0], pred[:, 1] - dsp64[ho, 1])
    frac_ok = float(np.mean(err < 0.3))
    ho_med = float(np.median(err))
    ho_max = float(np.percentile(err, 95))
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    e = ev(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
    ex = torch.clamp(e[:, 0].reshape(gy.shape), -8.0, 8.0).float()
    ey = torch.clamp(e[:, 1].reshape(gy.shape), -8.0, 8.0).float()
    return torch.stack([ex, ey]), ho_med, ho_max, frac_ok, n_rej, info


# ------------------------------------------------------------------------------------------------- requests

def _load(path):
    return _t(np.load(path).astype(np.float32, copy=False))


def op_global(req, ask_sift):
    mode = req["mode"]
    fb = features(_load(req["lf_b"]), mode)
    fa = features(_load(req["lf_a"]), mode)
    mask = static_mask(_load(req["lm_b"]), _load(req["lm_a"]), mode, req["border"], req.get("box"), req.get("kernel"))

    def sift():
        paths = {k: os.path.join(req["dump_dir"], f"{k}.npy") for k in ("feat_a", "feat_b", "mask")}
        for k, t in (("feat_a", fa), ("feat_b", fb), ("mask", mask)):
            np.save(paths[k], t.cpu().numpy())
        return ask_sift(paths)

    method, params, med, mx = GlobalGPU(float(req["scale"])).select_best_model(fa, fb, mask, sift)
    return {"method": method, "params": params, "med": float(med), "max": float(mx)}


def op_patches(req):
    fb = features(_load(req["luma_b"]), req["mode"])
    fa = features(_load(req["luma_a"]), req["mode"])
    xy = _t(np.load(req["xy"]).astype(np.int64).reshape(-1, 2))
    d0 = _t(np.load(req["d0"]).astype(np.float64).reshape(-1, 2))
    res = match_patches(fb, fa, xy, d0, int(req["patch"])) if len(xy) else np.zeros((0, 5))
    np.save(req["out"], res)
    return {"n": int(len(res))}


def op_local_frame(req):
    """_local_frame_job without the EXR write: refine_local_offsets (on the aux luma the decoder already warped by
    the global model) + reject_outliers + thin-plate field at the grid nodes."""
    mode = req["mode"]
    fb = features(_load(req["luma_b"]), mode)
    fa = features(_load(req["luma_a"]), mode)
    pts = np.load(req["pts"]).astype(np.float32).reshape(-1, 2)
    disp = np.load(req["disp"]).astype(np.float32).reshape(-1, 2).copy()
    ps = int(req.get("patch", 128))     # engine.align.LOCAL_PATCH_PX, sent by align_gpu.local_frames
    max_change = float(req.get("max_change_px", 4.0))
    h, w = fb.shape
    half = ps // 2
    xy = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in pts], np.int64).reshape(-1, 2)
    inside = (xy[:, 0] >= half) & (xy[:, 0] < w - half) & (xy[:, 1] >= half) & (xy[:, 1] < h - half)
    refined = np.zeros(len(disp), bool)
    response = np.full(len(disp), np.nan, np.float64)
    ii = np.nonzero(inside)[0]
    if len(ii):
        r = match_patches(fb, fa, _t(xy[ii]), _t(disp[ii].astype(np.float64)), ps)
        for k, (rx, ry, resp, gx, gy) in zip(ii, r):
            response[k] = resp
            if not (resp > 0.1) or not (math.isfinite(rx) and math.isfinite(ry)):
                continue
            d0 = disp[k]
            if not (math.isfinite(gx) and math.isfinite(gy)) or math.hypot(gx - d0[0], gy - d0[1]) > max_change:
                continue
            disp[k] = (np.clip(gx, -8.0, 8.0), np.clip(gy, -8.0, 8.0))
            refined[k] = True
    f, ho_med, ho_max, frac, n_rej, info = rbf_field(pts, disp, int(req["plate_w"]), int(req["plate_h"]), resp=response)
    np.save(req["out"], f.cpu().numpy())
    return {"ho_med": ho_med, "ho_max": ho_max, "ho_frac_ok": frac,
            "refined_frac": float(refined.mean()) if len(refined) else 0.0,
            "n_points": int(len(pts)), "n_rejected": n_rej, "reject": info}


def op_vector_blur(req):
    """engine/motionblur.vector_blur on the GPU: the velocity (flow scale, (sh, sw, 2)) is upsampled to the crop's
    pixels exactly like cv2.resize INTER_LINEAR (half-pixel centres, clamped edges) and divided by `scale`; the alpha
    is averaged along k x velocity (n samples, bilinear, border clamp) with the same weights as the CPU code."""
    a = torch.from_numpy(np.load(req["alpha"]).astype(np.float32)).to(DEV)
    vel = torch.from_numpy(np.load(req["vel"]).astype(np.float32)).to(DEV)          # (sh, sw, 2)
    x0, y0 = int(req["x0"]), int(req["y0"])
    W, H = int(req["plate_w"]), int(req["plate_h"])
    s, k = float(req["scale"]), float(req["k"])
    min_px, max_px = float(req["min_px"]), float(req["max_px"])
    h, w = a.shape
    ys = torch.arange(y0, y0 + h, device=DEV, dtype=torch.float32)
    xs = torch.arange(x0, x0 + w, device=DEV, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # full-frame pixel -> normalised coordinate of the flow-scale image (= cv2.resize to (W, H), align_corners=False)
    g = torch.stack(((2 * gx + 1) / W - 1, (2 * gy + 1) / H - 1), dim=-1)[None]
    v = F.grid_sample(vel.permute(2, 0, 1)[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0]
    if s < 1.0:
        v = v / s
    lx, ly = v[0] * k, v[1] * k
    ln = torch.hypot(lx, ly)
    f = torch.where(ln > max_px, max_px / ln.clamp_min(1e-6), torch.ones_like(ln))
    lx, ly, ln = lx * f, ly * f, torch.minimum(ln, torch.tensor(max_px, device=DEV))
    wgt = (ln - (min_px - 1.0)).clamp(0.0, 1.0)
    act = wgt > 0
    moving = int((ln >= min_px).sum())
    if not bool(act.any()):
        np.save(req["out"], a.cpu().numpy())
        return {"moving_px": moving, "max_px": round(float(ln.max()) if ln.numel() else 0.0, 2), "samples": 0}
    n = int(min(65, max(2, math.ceil(float(ln[act].max())) + 1)))
    lx_l, ly_l = gx - x0, gy - y0                                       # crop pixel coordinates
    acc = torch.zeros_like(a)
    src = a[None, None]
    for i in range(n):
        t = (i + 0.5) / n - 0.5
        sx, sy = lx_l + t * lx, ly_l + t * ly
        gg = torch.stack(((2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1), dim=-1)[None]
        acc += F.grid_sample(src, gg, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
    out = (a + wgt * (acc / n - a)).clamp(0.0, 1.0)
    np.save(req["out"], out.cpu().numpy())
    return {"moving_px": moving, "max_px": round(float(ln.max()), 2), "samples": n}


def _fake(req):
    if req.get("fake_sleep"):
        time.sleep(float(req["fake_sleep"]))
    if req.get("fake_crash") or os.environ.get("MD_ALIGN_GPU_FAKE_CRASH") == "1":
        os._exit(3)
    once = req.get("fake_crash_once")
    if once and not os.path.exists(once):
        open(once, "w").close()
        os._exit(3)
    op = req.get("op")
    if op == "global":
        np.load(req["lf_b"])
        np.load(req["lf_a"])
        return {"method": "translation", "params": {"dx": 0.0, "dy": 0.0, "response": 1.0}, "med": 0.0, "max": 0.0}
    if op == "patches":
        n = len(np.load(req["xy"]))
        np.save(req["out"], np.tile(np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]), (n, 1)))
        return {"n": n}
    if op == "local_frame":
        np.save(req["out"], np.zeros((2, int(req["plate_h"]), int(req["plate_w"])), np.float32))
        return {"ho_med": 0.0, "ho_max": 0.0, "ho_frac_ok": 1.0, "refined_frac": 1.0, "n_points": 0, "n_rejected": 0}
    if op == "vector_blur":
        np.save(req["out"], np.load(req["alpha"]))
        return {"moving_px": 0, "max_px": 0.0, "samples": 0}
    if op == "release":
        return {}
    raise ValueError(f"unknown op {op}")


def _serve():
    proto = sys.stdout
    sys.stdout = sys.stderr      # library prints must not break the protocol
    device = "fake" if FAKE else str(_init_torch())

    def send(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    def ask_sift(paths):
        send({"ok": True, "need_sift": True, "dump": paths})
        line = sys.stdin.readline()
        msg = json.loads(line) if line.strip() else {}
        return msg.get("models") or {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                break
            t0 = time.time()
            if FAKE:
                out = _fake(req)
            elif op == "global":
                out = op_global(req, ask_sift)
            elif op == "patches":
                out = op_patches(req)
            elif op == "local_frame":
                out = op_local_frame(req)
            elif op == "vector_blur":
                out = op_vector_blur(req)
            elif op == "release":
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                out = {}
            else:
                raise ValueError(f"unknown op {op}")
            resp = {"ok": True, "device": device, "seconds": round(time.time() - t0, 4), **out}
            if not FAKE and DEV.type == "cuda":
                resp["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
                resp["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / 2 ** 20)
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        send(resp)
    return 0


def _check():
    if FAKE:
        print("ok fake")
        return 0
    try:
        dev = _init_torch()
        name = torch.cuda.get_device_name(0) if dev.type == "cuda" else ""
        float(torch.fft.fft2(torch.ones((8, 8), device=dev)).real.sum())
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1
    print("ok", dev.type, name)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check()
    if len(argv) >= 2 and argv[1] == "--serve":
        return _serve()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
