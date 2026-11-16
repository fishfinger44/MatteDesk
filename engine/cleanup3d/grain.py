# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cleanup3d/grain.py - grain match of the Cleanup 3D patch (P17, docs/PLAN_CLEANUP_3D.md, user decision
2026-11-10): the patch from the keys (generators, projected and resampled keys) is cleaner than the plate when the
footage was NOT denoised on input; grain is then added to the patch so it matches the plate.

- measurement (`measure`): plate and patch (the projected keys after the tone match) on the SAME pixels – outside the
  hole, where the keys fully cover (there both show the same background), in the flattest blocks of the patch
  (lowest gradient energy of the clean fill), away from the hole and the region border. Per channel, per frequency
  band (difference of Gaussians: sigma 0-1, 1-2, 2-4 px), robust level = 1.4826 * median absolute deviation. A level is
  also given as the "equivalent sigma" of white noise (level / the band's response to unit white noise), so plate and
  patch noise read as one number per channel.
- decision: "off" = nothing; "auto" = grain only when the plate is clearly noisier than the patch (total equivalent
  sigma > RATIO x the patch's and more than MIN_DIFF above it – a denoised plate changes nothing); "on" = whatever is
  missing, without the threshold.
- synthesis (`synth`): white noise seeded by the frame (new noise every frame, the same on a re-render), split into the
  same bands and each band scaled to what is missing: sqrt(plate^2 - patch^2) per band and channel (the bands of white
  noise telescope, so the result has the plate's band energies), times the user's strength, times a brightness gain
  (noise as a function of brightness: three brightness bins of the measured pixels, when each has enough of them).
- applied (`apply`) in the render after the tone match, before the composite, only where the patch weight > 0.

Images: BGR float32, 0..1 display for 8-bit plates; scene-linear (EXR) is measured and grained in ACEScct
(engine.cleanup.to_model), like the tone match. The live preview of C5 does not add grain."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from engine import cleanup as cu

MODES = ("auto", "off", "on")
SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)   # band k = G(SIGMAS[k]) - G(SIGMAS[k + 1]); G(0) = the image itself
BLOCK = 16                           # flat-block size (px)
CBLOCK = 32                          # blocks whose median is taken out of a band before its level (centred)
FLAT_Q = 40.0                        # blocks with texture at or below this percentile of the region's blocks
BLOCK_FILL = 0.75                    # a block counts when this share of its pixels is measurable
MIN_PX = 1500                        # fewer measurable pixels = no measurement, no grain
RATIO = 1.3                          # auto: the plate's noise above this x the patch's ...
MIN_DIFF = 0.002                     # ... and more than this above it (model units, ~0.5 of an 8-bit level)
LUM_BINS = 3
LUM_MIN_PX = 800                     # per brightness bin; fewer = one gain for every brightness
LUM_GAIN = (0.3, 3.0)
SEED = 1717                          # + the frame number: deterministic, different per frame
_CAL = 384                           # calibration noise: _CAL x _CAL, fixed seed
_CACHE: Dict[str, Any] = {}


def clean_mode(v: Any) -> str:
    return v if v in MODES else "auto"


def clean_strength(v: Any) -> float:
    try:
        return float(min(2.0, max(0.0, float(v))))
    except (TypeError, ValueError):
        return 1.0


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT).reshape(img.shape)   # (h, w, 1) stays


def bands(img: np.ndarray, mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Difference-of-Gaussians bands of `img` (h, w[, c]), finest first; they sum to img - G(SIGMAS[-1]). With `mask`
    (bool / 0..1, h x w) the blurs are normalised convolutions inside it: pixels outside the mask (the object in the
    hole) never reach a band."""
    img = img.astype(np.float32)
    if mask is None:
        g = [_blur(img, s) for s in SIGMAS]
    else:
        m = mask.astype(np.float32).reshape(mask.shape + (1,) * (img.ndim - 2))
        im = img * m
        g = [img] + [_blur(im, s) / np.maximum(_blur(m, s), 1e-4) for s in SIGMAS[1:]]
    return [g[k] - g[k + 1] for k in range(len(SIGMAS) - 1)]


def _mad(v: np.ndarray) -> np.ndarray:
    """Robust sigma along axis 0: 1.4826 * median absolute deviation."""
    return 1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)


def centred(b: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """(n, c) values of band `b` (h, w, c) at the pixels of `sel`, in the order of b[sel], minus the median of their
    CBLOCK x CBLOCK block: what is left of the picture's structure in a band is nearly constant over a block, the
    noise is not (the bias this puts on the noise is calibrated away – white_response / band_gains measure the same
    way)."""
    h, w = sel.shape
    c = b.shape[2]
    ph, pw = -h % CBLOCK, -w % CBLOCK
    s = np.pad(sel, ((0, ph), (0, pw)))
    by, bx = s.shape[0] // CBLOCK, s.shape[1] // CBLOCK
    v = np.where(s[..., None], np.pad(b, ((0, ph), (0, pw), (0, 0))), np.nan)
    v = v.reshape(by, CBLOCK, bx, CBLOCK, c).transpose(0, 2, 1, 3, 4)
    used = s.reshape(by, CBLOCK, bx, CBLOCK).any(axis=(1, 3))
    med = np.zeros((by, bx, c), np.float32)
    if used.any():
        med[used] = np.nanmedian(v[used].reshape(int(used.sum()), CBLOCK * CBLOCK, c), axis=1)
    mi = np.repeat(np.repeat(med, CBLOCK, 0), CBLOCK, 1)[:h, :w]
    return b[sel] - mi[sel]


def _level(v: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(v), axis=0)


def band_levels(img: np.ndarray, sel: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(bands, c): robust noise level of every band of `img` (h, w, c) on `sel` (block-centred MAD)."""
    return np.stack([_level(centred(b, sel)) for b in bands(img, mask)])


def _cal() -> Dict[str, Any]:
    """Unit white noise (fixed seed), its bands and the measured pixels – the calibration of levels and gains."""
    if not _CACHE:
        n = np.random.default_rng(0).standard_normal((_CAL, _CAL, 1)).astype(np.float32)
        sel = np.zeros((_CAL, _CAL), bool)
        sel[CBLOCK:-CBLOCK, CBLOCK:-CBLOCK] = True
        _CACHE.update({"bands": bands(n), "sel": sel})
        _CACHE["white"] = np.stack([_level(centred(b, sel)) for b in _CACHE["bands"]])[:, 0]
    return _CACHE


def white_response() -> np.ndarray:
    """Level of each band of unit white noise, measured like the plate (band_levels; computed once, fixed seed)."""
    return _cal()["white"]


def equivalent_sigma(levels: np.ndarray) -> np.ndarray:
    """(c,): the sigma of white noise with the same total band energy as `levels` (bands, c) – white noise of sigma s
    gives s; the coarse bands, where picture structure is left, weigh by their (small) absolute energy."""
    S = white_response()[:, None]
    return np.sqrt((levels ** 2).sum(0) / (S ** 2).sum())


def _cross() -> np.ndarray:
    """(bands j, bands k, bands l): covariance of band j of band k and band j of band l of unit white noise (centred
    like the measurement) – level_j(sum_k g_k band_k(noise))^2 ~ g^T M_j g."""
    cal = _cal()
    if "cross" not in cal:
        nb = len(SIGMAS) - 1
        sub = [[centred(bj, cal["sel"])[:, 0] for bj in bands(bk)] for bk in cal["bands"]]    # [k][j] -> (n,)
        M = np.zeros((nb, nb, nb))
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    M[j, k, l] = float(np.mean(sub[k][j] * sub[l][j]))
        cal["cross"] = M
    return cal["cross"]


def band_gains(target: np.ndarray) -> np.ndarray:
    """(bands, c) gains g such that sum_k g_k * band_k(white noise) has the band levels `target` (bands, c). The DoG
    bands overlap (a band of a band is not 0), so per channel the gains solve sqrt(g^T M_j g) = target_j in the least
    squares sense (relative errors; the signs are free, so a fine band below the leak of the coarser ones is
    reachable), starting from target / white_response()."""
    from scipy.optimize import least_squares
    M = _cross()
    S = white_response()
    out = np.zeros_like(target, dtype=np.float64)
    for c in range(target.shape[1]):
        t = target[:, c].astype(np.float64)
        if not (t > 1e-9).any():
            continue
        tt = np.maximum(t, 1e-3 * t.max())

        def res(g):
            return np.sqrt(np.maximum(np.einsum("k,jkl,l->j", g, M, g), 1e-18)) / tt - 1.0
        out[:, c] = least_squares(res, t / S, method="lm", xtol=1e-10, ftol=1e-10).x
    return out


def flat_pixels(fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """bool (h, w): the pixels of `mask` in its flattest BLOCK x BLOCK blocks (gradient energy of the slightly blurred
    gray of the clean fill), at least MIN_PX of them when the mask has that many."""
    h, w = mask.shape
    g = cv2.GaussianBlur(cv2.cvtColor(fill.astype(np.float32), cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    tex = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2
    by, bx = h // BLOCK, w // BLOCK
    if by == 0 or bx == 0:
        return np.zeros_like(mask)
    hh, ww = by * BLOCK, bx * BLOCK
    m = mask[:hh, :ww].reshape(by, BLOCK, bx, BLOCK).astype(np.float32)
    share = m.mean(axis=(1, 3))
    t = (tex[:hh, :ww].reshape(by, BLOCK, bx, BLOCK) * m).sum(axis=(1, 3)) / np.maximum(m.sum(axis=(1, 3)), 1.0)
    ok = share >= BLOCK_FILL
    if not ok.any():
        return np.zeros_like(mask)
    order = np.argsort(np.where(ok, t, np.inf), axis=None)
    n_ok = int(ok.sum())
    thr = np.percentile(t[ok], FLAT_Q)
    pick = np.zeros(by * bx, bool)
    px = 0
    per = share.ravel() * BLOCK * BLOCK
    for i in order[:n_ok]:
        if t.ravel()[i] > thr and px >= MIN_PX:
            break
        pick[i] = True
        px += per[i]
    sel = np.repeat(np.repeat(pick.reshape(by, bx), BLOCK, 0), BLOCK, 1)
    out = np.zeros_like(mask)
    out[:hh, :ww] = sel & mask[:hh, :ww]
    return out


def levels_cpu(p: np.ndarray, f: np.ndarray, reg: np.ndarray, sel: np.ndarray, idx: Optional[np.ndarray]):
    """(plate levels (bands, 3), fill levels, plate finest-band level per brightness bin, fill ...) – the heavy part of
    `measure` (the bands as normalised convolutions inside `reg`, block-centred, robust levels on `sel`); idx = the
    brightness bin of every measured pixel (in the order of sel's pixels) or None. The GPU twin: geom_gpu op
    "grain_levels" (render.KeyFill on device "gpu")."""
    cp = [centred(b, sel) for b in bands(p, reg)]                 # (n, 3) per band
    cf = [centred(b, sel) for b in bands(f, reg)]
    bp, bf = np.stack([_level(v) for v in cp]), np.stack([_level(v) for v in cf])     # (bands, 3)
    if idx is None:
        return bp, bf, None, None
    ks = [idx == k for k in range(LUM_BINS)]
    return (bp, bf, [_level(cp[0][s]) if s.any() else None for s in ks],
            [_level(cf[0][s]) if s.any() else None for s in ks])


def measure(plate: np.ndarray, fill: np.ndarray, region: np.ndarray, linear: bool = False,
            levels: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Noise of the plate and of the fill on the flat pixels of `region` (bool: where both show the same background).
    Returns None when fewer than MIN_PX pixels are measurable, else {"plate" / "patch": per band x channel levels,
    "plate_sigma" / "patch_sigma": per channel equivalent white-noise sigma, "px", "lum": (bin centres, plate, patch)
    equivalent sigma per brightness bin or None, "sel": the measured pixels}. Model units (to_model).
    `levels`: in place of `levels_cpu` (the same signature; the GPU worker's)."""
    p, f = cu.to_model(plate, linear), cu.to_model(fill, linear)
    # the bands are normalised convolutions inside the region (the object in the hole, the fill where no key reaches
    # never enter); near the region's edge the blurs see fewer pixels, so the measured pixels keep a margin from it
    margin = int(np.ceil(1.5 * SIGMAS[-1])) + 1
    reg = region.astype(bool)
    m = cv2.erode(reg.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    m[:margin], m[-margin:], m[:, :margin], m[:, -margin:] = False, False, False, False
    if int(m.sum()) < MIN_PX:
        return None
    rf = reg.astype(np.float32)
    lum = _blur(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) * rf, 2.0) / np.maximum(_blur(rf, 2.0), 1e-4)
    m &= (lum > 0.02) & (lum < 0.98)                                               # clipped pixels have no noise
    if int(m.sum()) < MIN_PX:
        return None
    sel = flat_pixels(np.where(reg[..., None], f, 0.5).astype(np.float32), m)
    n = int(sel.sum())
    if n < MIN_PX:
        return None
    S = white_response()[:, None]
    # noise as a function of brightness: equivalent sigma (all channels, finest band) in LUM_BINS brightness bins
    fl = (_blur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) * rf, 2.0) / np.maximum(_blur(rf, 2.0), 1e-4))[sel]
    idx = None
    if n >= LUM_BINS * LUM_MIN_PX:
        edges = np.percentile(fl, np.linspace(0, 100, LUM_BINS + 1))
        if edges[-1] - edges[0] > 0.05:
            idx = np.clip(np.searchsorted(edges, fl, side="right") - 1, 0, LUM_BINS - 1)
    bp, bf, binp, binf = (levels or levels_cpu)(p, f, reg, sel, idx)
    out: Dict[str, Any] = {"plate": bp, "patch": bf, "px": n, "sel": sel,
                           "plate_sigma": equivalent_sigma(bp), "patch_sigma": equivalent_sigma(bf),
                           "lum": None}
    if idx is not None:
        cen, lp, lf = [], [], []
        for k in range(LUM_BINS):
            s = idx == k
            if int(s.sum()) < LUM_MIN_PX:
                break
            cen.append(float(np.median(fl[s])))
            lp.append(float(binp[k].mean() / S[0, 0]))
            lf.append(float(binf[k].mean() / S[0, 0]))
        else:
            out["lum"] = (np.array(cen), np.array(lp), np.array(lf))
    return out


def missing(meas: Mapping[str, Any]) -> np.ndarray:
    """(bands, 3): the band levels the patch lacks, sqrt(plate^2 - patch^2), 0 where the patch is the noisier."""
    return np.sqrt(np.maximum(meas["plate"] ** 2 - meas["patch"] ** 2, 0.0))


def decide(meas: Optional[Mapping[str, Any]], mode: str) -> bool:
    """Whether grain is added: never "off" / unmeasured; "on" whenever something is missing; "auto" only when the
    plate is clearly noisier (RATIO, MIN_DIFF)."""
    if mode == "off" or meas is None:
        return False
    if not (missing(meas) > 0).any():
        return False
    if mode == "on":
        return True
    ps, qs = float(np.mean(meas["plate_sigma"])), float(np.mean(meas["patch_sigma"]))
    return ps > RATIO * qs and ps - qs > MIN_DIFF


def lum_gain(meas: Mapping[str, Any], lum: np.ndarray) -> Optional[np.ndarray]:
    """Per-pixel gain of the grain by the fill's brightness `lum` (missing noise in the pixel's brightness bin / the
    missing noise overall), None when the brightness bins were not measured."""
    if meas.get("lum") is None:
        return None
    cen, lp, lf = meas["lum"]
    mb = np.sqrt(np.maximum(lp ** 2 - lf ** 2, 0.0))
    S0 = white_response()[0]
    allm = float(np.sqrt(np.maximum(meas["plate"][0] ** 2 - meas["patch"][0] ** 2, 0.0)).mean() / S0)
    if allm <= 1e-6:
        return None
    g = np.clip(mb / allm, *LUM_GAIN)
    return np.interp(lum, cen, g).astype(np.float32)


def synth(shape: Sequence[int], meas: Mapping[str, Any], frame: int, strength: float = 1.0,
          lum: Optional[np.ndarray] = None) -> np.ndarray:
    """Grain (h, w, 3) for frame `frame`: white noise seeded by the frame, each band scaled to the missing level."""
    h, w = int(shape[0]), int(shape[1])
    rng = np.random.default_rng([SEED, int(frame)])
    noise = rng.standard_normal((h, w, 3)).astype(np.float32)
    gains = band_gains(missing(meas)) * float(strength)                       # (bands, 3)
    out = np.zeros((h, w, 3), np.float32)
    for k, b in enumerate(bands(noise)):
        out += b * gains[k][None, None, :].astype(np.float32)
    if lum is not None:
        g = lum_gain(meas, lum)
        if g is not None:
            out *= g[..., None]
    return out


def apply(fill: np.ndarray, plate: np.ndarray, hole: np.ndarray, usable: np.ndarray, patch: np.ndarray, frame: int,
          settings: Mapping[str, Any], linear: bool = False, stats: Optional[Dict[str, Any]] = None,
          seconds: Optional[Dict[str, float]] = None, levels: Optional[Any] = None,
          meas: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """The render step (KeyFill, after the tone match): fill (h, w, 3) of the region, plate of the same region, hole
    (bool), usable (bool: the keys fully cover, no kept object), patch (the patch weight of the region). Returns the
    fill with grain where patch > 0 (unchanged when nothing is added) and writes stats["grain"] =
    {plate, patch: per channel equivalent sigma (model units) or None, added, strength, px}. `meas` = a measurement
    made elsewhere (Inpaint C1 measures once per shot, `combine`) – then this frame is not measured."""
    mode = clean_mode(settings.get("grain", "auto"))
    if mode == "off":
        return fill
    t0 = time.time()
    if meas is None:
        meas = measure(plate, fill, usable & ~hole, linear, levels)
    added = decide(meas, mode)
    strength = clean_strength(settings.get("grain_strength", 1.0))
    rep: Dict[str, Any] = {"plate": None, "patch": None, "added": False, "mode": mode, "strength": strength, "px": 0}
    if meas is not None:
        rep.update({"plate": [round(float(v), 5) for v in meas["plate_sigma"]],
                    "patch": [round(float(v), 5) for v in meas["patch_sigma"]], "px": meas["px"],
                    "bands": {"plate": np.round(meas["plate"], 5).tolist(), "patch": np.round(meas["patch"], 5).tolist()}})
    out = fill
    on = patch > 0
    if added and strength > 0 and on.any():
        fm = cu.to_model(fill, linear) if linear else fill.astype(np.float32)
        lum = cv2.GaussianBlur(cv2.cvtColor(np.clip(fm, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY), (0, 0), 2.0)
        g = synth(fill.shape, meas, frame, strength, lum)
        grained = fm + g
        if linear:
            grained = cu.from_model(grained, linear)
        out = np.where(on[..., None], grained, fill).astype(np.float32)
        rep["added"] = True
    if stats is not None:
        stats["grain"] = rep
    if seconds is not None:
        seconds["grain"] = round(seconds.get("grain", 0.0) + time.time() - t0, 3)
    return out


def combine(meas: Sequence[Optional[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """ONE measurement of several frames of a shot (Inpaint C1, 2026-11-14: measuring every frame cost ~0.8 s each and
    the grain of a plate does not change frame to frame): the median of the band levels and equivalent sigmas, the
    brightness bins of the frame with the most measured pixels. None = no frame measured."""
    ok = [m for m in meas if m is not None]
    if not ok:
        return None
    best = max(ok, key=lambda m: int(m.get("px") or 0))
    out = dict(best)
    for k in ("plate", "patch", "plate_sigma", "patch_sigma"):
        out[k] = np.median(np.stack([np.asarray(m[k], np.float64) for m in ok]), axis=0)
    out["px"] = int(sum(int(m.get("px") or 0) for m in ok))
    out["frames"] = len(ok)
    return out


def summary(frames: Optional[Mapping[Any, Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The render's grain in one line for the UI: median equivalent sigma (mean of channels, model units) of the plate
    and the patch over the measured frames, frames with grain added / measured / all."""
    rows = [(v or {}).get("grain") for v in (frames or {}).values()]
    rows = [r for r in rows if r]
    if not rows:
        return None
    got = [r for r in rows if r.get("plate") is not None]
    med = (lambda k: round(float(np.median([np.mean(r[k]) for r in got])), 5)) if got else (lambda k: None)
    return {"plate": med("plate"), "patch": med("patch"), "added": sum(1 for r in rows if r.get("added")),
            "measured": len(got), "frames": len(rows), "mode": rows[0].get("mode")}
