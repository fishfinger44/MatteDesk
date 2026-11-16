# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python port of the MergeMatte Nuke group (engine/merge_matte_nk.py builds the group; handoff/nuke/MergeMatte_v01.nk),
node by node. The ONLY implementation of the edge-colour comp in MatteDesk (PLAN_FINAL_TOUCH.md §11): the Final Touch
preview, the README comparisons and the tests call `merge_matte()` here. New operations go into the Nuke group first.

Conventions (Nuke): images are float32 H×W×4 RGBA, premultiplied unless a function says "straight"; a missing BG alpha is 0
(a Read of an RGB plate has no alpha); outside the format is black (Blur pads with 0). Knob names = MergeMatte knobs
(`merge_matte_nk.KNOB_SPECS`). Blur `size` → gaussian sigma and FilterErode are calibrated against Nuke
(tests/test_merge_matte.py, `real`).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from engine import log_curves as lc
from engine.log_curves import LOG_CURVES   # noqa: F401  (the lin ↔ log curves of the `log_curve` knob)
from engine.merge_matte_nk import KNOB_DEFAULTS, KNOB_RANGES, PRESET_SH010, normalize_knobs   # noqa: F401  (one source of knobs)

# Nuke Blur (filter gaussian) `size` → gaussian sigma, measured 2026-10-29 in the user's Nuke 15.0v1 (impulse response,
# second moment, sizes 2.5 / 3 / 5 / 23.2 / 100 → 0.4229 / 0.4237 / 0.4236 / 0.4269 / 0.4250; gaussian shape within 0.4 %
# of the peak up to size 5).
BLUR_SIGMA_PER_SIZE = 0.4236

REC709 = np.array([0.2126, 0.7152, 0.0722], np.float32)

# ---------------------------------------------------------------- colour spaces
# lin ↔ log of the `log_curve` knob: engine/log_curves.py (ACEScct kept under its old names for the callers / tests)
lin_to_acescct = LOG_CURVES["acescct"].lin_to_log
acescct_to_lin = LOG_CURVES["acescct"].log_to_lin


# sRGB (display PNG plate) ↔ ACEScg, i.e. ACES "Utility - sRGB - Texture": sRGB curve + matrix sRGB → AP1 (D65 → D60 via Bradford).
_SRGB_TO_AP1 = np.array([[0.6131, 0.3395, 0.0474], [0.0702, 0.9164, 0.0134], [0.0206, 0.1096, 0.8698]], np.float32)
_AP1_TO_SRGB = np.linalg.inv(_SRGB_TO_AP1).astype(np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055).astype(np.float32)


def srgb_to_acescg(rgb: np.ndarray) -> np.ndarray:
    """Display sRGB 0..1 (RGB order) → scene-linear ACEScg."""
    return srgb_to_linear(rgb) @ _SRGB_TO_AP1.T


def acescg_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(linear_to_srgb(rgb @ _AP1_TO_SRGB.T), 0, 1)


# Node channel defaults checked in the user's Nuke 15.0v1 (2026-10-29): Multiply / Clamp / Keymix / Blur / Invert / Dissolve
# = `all`, Saturation / Grade = `rgb`, Merge2 output rgba, FilterErode channels alpha (the group sets gaussian explicitly).
# ---------------------------------------------------------------- Nuke nodes as numpy functions
def unpremult(img: np.ndarray) -> np.ndarray:
    """Unpremult: rgb / alpha where alpha != 0 (pixels with alpha 0 are left as they are)."""
    out = img.copy()
    a = img[..., 3:4]
    nz = a != 0
    out[..., :3] = np.where(nz, img[..., :3] / np.where(nz, a, 1), img[..., :3])
    return out


def premult(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[..., :3] *= img[..., 3:4]
    return out


def blur(img: np.ndarray, size: float) -> np.ndarray:
    """Blur, filter gaussian, `size` px; outside the image = black (Nuke pads the bbox with 0)."""
    if size <= 0:
        return img.copy()
    s = size * BLUR_SIGMA_PER_SIZE
    return cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_CONSTANT).reshape(img.shape)


def blur_masked(img: np.ndarray, size: float, mask: np.ndarray) -> np.ndarray:
    """Blur with its mask input: out = blurred * mask + in * (1 - mask)."""
    if size <= 0:
        return img.copy()
    m = np.clip(mask, 0, 1)[..., None]
    return blur(img, size) * m + img * (1 - m)


FE_SIGMA_PER_SIZE, FE_SIGMA_OFFSET = 0.85, 0.42   # FilterErode gaussian: sigma = 0.85 |size| + 0.42 (fitted 2026-11-06)


def fe_sigma(size: float) -> float:
    return FE_SIGMA_PER_SIZE * abs(float(size)) + FE_SIGMA_OFFSET


def _dilate_gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """max over y of a(x + y) · exp(−|y|² / 2σ²) – separable (non-negative values: the max of products splits into a
    row pass and a column pass); outside the image = 0 (never wins); support 4σ (exp(−8) ≈ 3e-4)."""
    r = int(np.ceil(4 * sigma))
    out = np.asarray(a, np.float32).copy()
    for axis in (1, 0):
        src = out
        res = src.copy()
        for d in range(1, r + 1):
            g = np.float32(np.exp(-d * d / (2.0 * sigma * sigma)))
            if axis == 1:
                np.maximum(res[:, d:], src[:, :-d] * g, out=res[:, d:])
                np.maximum(res[:, :-d], src[:, d:] * g, out=res[:, :-d])
            else:
                np.maximum(res[d:], src[:-d] * g, out=res[d:])
                np.maximum(res[:-d], src[d:] * g, out=res[:-d])
        out = res
    return out


def filter_erode(a: np.ndarray, size: float) -> np.ndarray:
    """FilterErode, filter gaussian: size > 0 erodes, < 0 dilates (alpha only) – Nuke's node 1:1 (2026-11-06,
    tools/filtererode_probe.py on Nuke 11.1v3: steps, ramps 2–40 px, lines 1–5 px, half-level steps, discs, corners,
    a 1 px diagonal, sizes ±0.5…±20 – max |Δ| 0.0035 alpha, mean < 1e-4). Dilate = a non-flat gaussian max:
    max_y a(x + y) · g(y), g = exp(−|y|² / 2σ²), σ = 0.85 |size| + 0.42; erode = 1 − dilate(1 − a). A hard edge gets
    a gaussian tail, a soft ramp barely moves (its own values beat the attenuated neighbours) – what the earlier disc
    erosion could not do."""
    if size == 0:
        return np.asarray(a, np.float32).copy()
    a = np.asarray(a, np.float32)
    s = fe_sigma(size)
    return _dilate_gauss(a, s) if size < 0 else (1.0 - _dilate_gauss(1.0 - a, s)).astype(np.float32)


def keymix(b: np.ndarray, a: np.ndarray, mask: np.ndarray, mix: float = 1.0, channels: str = "rgba") -> np.ndarray:
    """Keymix: B where mask is 0, A where mask is 1 (× mix); `channels` rgb or rgba."""
    m = (np.clip(mask, 0, 1) * mix)[..., None]
    out = b.copy()
    n = 3 if channels == "rgb" else 4
    out[..., :n] = b[..., :n] * (1 - m) + a[..., :n] * m
    return out


def merge_over(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge over: A + B (1 - a), all channels."""
    return a + b * (1 - a[..., 3:4])


def merge_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * (1 - b[..., 3:4]) + b * (1 - a[..., 3:4])


def merge_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge mask: B a."""
    return b * a[..., 3:4]


def merge_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def merge_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def merge_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merge divide: A / B (0 where B is 0)."""
    return np.where(b != 0, a / np.where(b != 0, b, 1), 0).astype(np.float32)


def saturation(rgb: np.ndarray, s: float) -> np.ndarray:
    """Saturation node (Rec 709 luminance)."""
    lum = (rgb[..., :3] * REC709).sum(-1, keepdims=True)
    return (lum + (rgb[..., :3] - lum) * s).astype(np.float32)


def keyer_luminance(rgb: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Keyer, luminance key, range {a b c d}: 0 below a, ramp to 1 at b, 1 to c, ramp to 0 at d."""
    x = (rgb[..., :3] * REC709).sum(-1)
    out = np.zeros_like(x)
    if b != a:
        ramp = (x - a) / (b - a)
        out = np.where((x >= a) & (x < b), ramp, out)
    out = np.where((x >= b) & (x <= c), 1.0, out)
    if d != c:
        out = np.where((x > c) & (x < d), (d - x) / (d - c), out)
    return out.astype(np.float32)


def grade_gain(img: np.ndarray, gain: float, channels: str = "rgb") -> np.ndarray:
    out = img.copy()
    n = 3 if channels == "rgb" else 4
    out[..., :n] *= gain
    return out


def grade_whitepoint(img: np.ndarray, whitepoint: float, white_clamp: bool = True) -> np.ndarray:
    """Grade channels rgba, whitepoint w (black_clamp on by default): clip(in / w, 0, 1)."""
    out = img / whitepoint
    return np.clip(out, 0, 1 if white_clamp else np.inf).astype(np.float32)


def _wrap_chain(x: np.ndarray, size: float) -> np.ndarray:
    """flare/wrap/bleed blur stack: Blur(size), Blur(2 size), Blur(4 size), Blur(8 size) merged over each other
    (add2: blur2 over blur1, add3: blur3 over that, add4: blur4 over that)."""
    b1, b2, b3, b4 = (blur(x, size * m) for m in (1, 2, 4, 8))
    r = merge_over(b2, b1)
    r = merge_over(b3, r)
    return merge_over(b4, r)


def _with_alpha(img: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[..., 3] = a
    return out


# ---------------------------------------------------------------- the group
def knobs(overrides: Optional[Mapping[str, object]] = None, project_curve: Optional[str] = None) -> Dict[str, object]:
    """Node defaults with `overrides` (unknown names ignored; the old bool `log` → `log_curve`; `log_curve` = `project`
    resolved to `project_curve`, ACEScct when None – see merge_matte_nk.normalize_knobs)."""
    k = dict(KNOB_DEFAULTS)
    if overrides:
        k.update(normalize_knobs(overrides, project_curve))
    return k


def merge_matte(fg: np.ndarray, bg: np.ndarray, k: Optional[Mapping[str, object]] = None) -> np.ndarray:
    """MergeMatte v01: FG = premultiplied RGBA (scene-linear), BG = RGBA (alpha usually 0 for a plain plate /
    Constant). Returns the comp RGBA (RGB = FG over BG with the edge treatments, scene-linear). `log_curve` (an id of
    engine/log_curves.py, none = linear) = the curve the operations run in."""
    k = knobs(k)
    fg = np.asarray(fg, np.float32).copy()
    bg = np.asarray(bg, np.float32).copy()
    curve = lc.get(k["log_curve"])
    log = curve is not None

    # Alpha: choke / expand (FilterErode) + soften (Blur) on the FG alpha
    if k["choke"] != 0 or k["soften"] != 0:
        s = unpremult(fg)
        a = filter_erode(s[..., 3], k["choke"])
        a = blur(a, k["soften"])
        fg = premult(_with_alpha(s, a))
    # Log: both inputs through the curve (FG unpremultiplied around the conversion)
    if log:
        s = unpremult(fg)
        s[..., :3] = curve.lin_to_log(s[..., :3])
        fg = premult(s)
        bg[..., :3] = curve.lin_to_log(bg[..., :3])

    regular = merge_over(fg, bg)
    fgu = unpremult(fg)                                                   # Dot_decon
    # Decontaminate: interior colour spread (premult → Blur → unpremult) keymixed into the edge by 1 − alpha
    if k["decon"] != 0:
        spread = unpremult(blur(premult(fgu), k["decon_spread"]))
        fgu = keymix(fgu, spread, 1 - fgu[..., 3], k["decon"], "rgb")
    fgd = fgu                                                             # Dot3 / Dot12

    # BG outside the FG: BG with FG alpha → Invert alpha → Premult
    bg_out = premult(_with_alpha(bg, 1 - fg[..., 3]))                    # Dot4
    # A section with amount 0 contributes nothing in the group (Grade white 0 = black, Merge over black = identity;
    # bleed 0 = Saturation 1 + Merge mix 0 = the input), so it is skipped here – same result, less work.
    fg2 = fgd
    if k["bleed"] != 0:
        # Bleed colour: BG chroma (blur stack / its luminance) multiplied into the desaturated FG near the edge
        bleed = _wrap_chain(bg_out, k["bleed_size"])
        bleed = np.clip(bleed, 0.001, 100)                                # Clamp, default channels all (rgba)
        luma = bleed.copy()
        luma[..., :3] = saturation(bleed, 0)
        chroma = merge_divide(luma, bleed)                                # Merge divide: A = luma, B = colour (as in the group)
        fg_desat = fgd.copy()
        fg_desat[..., :3] = saturation(fgd, 1 - k["bleed"])
        bleed_apply = fg_desat + (merge_multiply(chroma, fg_desat) - fg_desat) * k["bleed"]
        fg2 = keymix(fgd, bleed_apply, bleed[..., 3], 1.0, "rgba")       # bleed_near_edge (mask = blur stack alpha)
    fg4 = fg2
    if k["wrap"] != 0:   # Light wrap – wrap: all BG outside the FG
        fg4 = merge_over(grade_gain(_wrap_chain(bg_out, k["wrap_size"]), k["wrap"], "rgba"), fg4)          # add_wrap
    if k["flare"] != 0:  # Light wrap – flare: bright BG (luminance key threshold..threshold+softness) outside the FG
        key = keyer_luminance(bg, k["flare_threshold"], k["flare_threshold"] + k["flare_softness"], 1.0, 1.0)
        flare0 = merge_mask(bg_out, premult(_with_alpha(bg, key)))
        fg4 = merge_over(grade_gain(_wrap_chain(flare0, k["wrap_size"]), k["flare"], "rgba"), fg4)        # add_flare
    fgp = premult(_with_alpha(fg4, fgd[..., 3]))                          # restore_fg_alpha → Premult
    comp = merge_over(fgp, bg)                                            # fg_over_bg

    # Darken edge: min(comp, BG) inside the band xor(Blur(alpha), Blur(alpha))
    if k["darken"] != 0:
        bl = blur(comp[..., 3], k["darken_width"])
        band = 2 * bl * (1 - bl)
        mn = merge_min(comp, bg)
        comp = keymix(comp, mn, band, k["darken"], "rgba")
    # Match edge: exposure / saturation of the comp inside a thinner band
    if k["match"] != 0:
        bl = blur(comp[..., 3], k["match_width"])
        band = 2 * bl * (1 - bl)
        g = grade_gain(comp, 2.0 ** k["match_exposure"], "rgb")
        g[..., :3] = saturation(g, k["match_saturation"])
        comp = keymix(comp, g, band, k["match"], "rgb")
    # Edge blur: blur the comp where xor(Blur 3, Blur 3) / 0.9 (clamped) says "edge"
    bl = blur(comp, 3.0)
    band = grade_whitepoint(merge_xor(bl, bl), 0.9)
    out = blur_masked(comp, k["edge_blur"], band[..., 3])
    if k["bypass"]:
        out = regular
    out = bg + (out - bg) * k["mix"]                                      # Dissolve(BG, comp, mix)
    if log:
        out[..., :3] = curve.log_to_lin(out[..., :3])
    return out


# ---------------------------------------------------------------- candidate for a later version: background unmix (§11 step 4 – not needed per step 0)
def unmix_bg(fgu: np.ndarray, plate_rgb: np.ndarray, amount: float, size: float, min_alpha: float = 0.08) -> np.ndarray:
    """"Unmix BG" on the straight FG colour: B = plate weighted by (1 − α) and blurred (÷ blurred 1 − α);
    F = (I − (1 − α) B) / α, clamped to the interior colour range (1–99 % of α > 0.98); below `min_alpha`
    the incoming colour (decontaminate) stays. Same colour space as `fgu`."""
    a = np.clip(fgu[..., 3], 0, 1)
    bw = (1 - a) * (a < 0.98)
    num = blur(plate_rgb * bw[..., None], size)
    den = blur(bw, size)[..., None]
    b = np.where(den > 1e-4, num / np.maximum(den, 1e-4), plate_rgb)
    f = (plate_rgb - (1 - a)[..., None] * b) / np.maximum(a, min_alpha)[..., None]
    solid = a > 0.98
    if solid.any():
        lo = np.percentile(plate_rgb[solid], 1, axis=0)
        hi = np.percentile(plate_rgb[solid], 99, axis=0)
        f = np.clip(f, lo, hi)
    w = np.clip((a - min_alpha) / max(min_alpha, 1e-4), 0, 1) * amount
    out = fgu.copy()
    out[..., :3] = np.where(solid[..., None], fgu[..., :3], fgu[..., :3] * (1 - w[..., None]) + f * w[..., None])
    return out


def _bg_rgba(bg_srgb, h: int, w: int) -> np.ndarray:
    """BG input: a flat display-sRGB colour (3 values) or an H×W×3 display-sRGB image → ACEScg RGBA, alpha 0 (a plate)."""
    bg = np.zeros((h, w, 4), np.float32)
    b = np.asarray(bg_srgb, np.float32)
    bg[..., :3] = srgb_to_acescg(b.reshape(1, 1, 3) if b.ndim == 1 else b[..., :3])
    return bg


def comp_srgb(plate_srgb: np.ndarray, alpha: np.ndarray, bg_srgb, k: Optional[Mapping[str, float]] = None,
              unmix: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Convenience for display-referred plates: plate RGB (0..1, RGB order) + alpha → MergeMatte over the BG (a flat
    colour or an image, display sRGB), returns display sRGB RGB 0..1 (ACEScg inside, as the group in an ACES script).
    `unmix` = {"amount", "size", "min_alpha"} runs Unmix BG on the straight FG colour before the group (measured in step 0,
    not needed)."""
    lin = srgb_to_acescg(plate_srgb.astype(np.float32))
    a = np.clip(alpha, 0, 1).astype(np.float32)
    h, w = a.shape
    bg = _bg_rgba(bg_srgb, h, w)
    if unmix and unmix.get("amount", 0) > 0:
        fgu = np.dstack([lin, a])
        lin = unmix_bg(fgu, lin, unmix["amount"], unmix.get("size", 30.0), unmix.get("min_alpha", 0.08))[..., :3]
    fg = np.dstack([lin * a[..., None], a])
    out = merge_matte(fg, bg, k)
    return acescg_to_srgb(out[..., :3])


# ---------------------------------------------------------------- preview of a region (Final Touch palette)
# knobs measured in px: scaled together with the image when the preview is computed smaller
PX_KNOBS = ("choke", "soften", "decon_spread", "darken_width", "match_width", "wrap_size", "bleed_size", "edge_blur")


def context_px(k: Mapping[str, float]) -> int:
    """How far (px) a pixel of the result depends on its neighbours: 3 sigma of the widest blur in use (+ choke's
    FilterErode reach, 4 σ of its gaussian)."""
    k = knobs(k)
    sizes = [3.0, k["edge_blur"], k["soften"], k["darken_width"] if k["darken"] else 0, k["match_width"] if k["match"] else 0,
             k["decon_spread"] if k["decon"] else 0, 8 * k["wrap_size"] if (k["wrap"] or k["flare"]) else 0,
             8 * k["bleed_size"] if k["bleed"] else 0]
    choke = 4 * fe_sigma(k["choke"]) if k["choke"] else 0.0
    return int(np.ceil(3 * BLUR_SIGMA_PER_SIZE * max(sizes) + choke + 2))


def preview_region(plate_srgb: np.ndarray, alpha: np.ndarray, bg_srgb, k: Optional[Mapping[str, float]] = None,
                   rect: Optional[tuple] = None, max_px: int = 2_500_000):
    """MergeMatte of the plate region rect = (x, y, w, h) (default: the whole frame), computed on the region plus the
    context the blurs need (so the region's pixels equal a whole-frame run); a region larger than max_px is computed
    smaller with the px knobs scaled (approximate – the palette says so). Returns (display sRGB uint8 RGB of the region,
    scale) – scale 1 = plate pixels."""
    H, W = alpha.shape[:2]
    x, y, w, h = rect if rect else (0, 0, W, H)
    x, y = max(0, int(x)), max(0, int(y))
    w, h = max(1, min(int(w), W - x)), max(1, min(int(h), H - y))
    k = knobs(k)
    m = context_px(k)
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    s = min(1.0, float(np.sqrt(max_px / float((x1 - x0) * (y1 - y0)))))
    pl, a = plate_srgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    b = np.asarray(bg_srgb, np.float32)
    b = b[y0:y1, x0:x1] if b.ndim == 3 else b
    if s < 1.0:
        sz = (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))
        pl = cv2.resize(pl, sz, interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, sz, interpolation=cv2.INTER_AREA)
        if b.ndim == 3:
            b = cv2.resize(b, sz, interpolation=cv2.INTER_AREA)
        k = {**k, **{n: k[n] * s for n in PX_KNOBS}}
    out = comp_srgb(pl, a, b, k)
    ox, oy = round((x - x0) * s), round((y - y0) * s)
    out = out[oy:oy + max(1, round(h * s)), ox:ox + max(1, round(w * s))]
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), s


# ---------------------------------------------------------------- QA: background left in the edge (PLAN_FINAL_TOUCH §11 step 5)
EDGE_LO, EDGE_HI = 0.05, 0.95        # the edge band: 0.05 < alpha < 0.95
EDGE_SPREAD = 30.0                   # Blur size (px) of the interior colour spread into the edge


def edge_bg(plate_srgb: np.ndarray, alpha: np.ndarray, comp_srgb: np.ndarray, bg_srgb, spread: float = EDGE_SPREAD,
            comp_alpha: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """„Tło w krawędzi” (as in rik-python): how far the comp's edge is from what a clean edge would show –
    ideal = α · F + (1 − α) · BG, F = the object's own colour spread from its solid interior (α ≥ 0.95, premult → Blur →
    unpremult) – on the band 0.05 < α < 0.95, display sRGB 0..1. Old plate background (bright sky) carried by the soft
    edge shows up here; the signed luminance says whether the edge is brighter (+, a halo) or darker (−) than clean.
    `comp_alpha` = the alpha the comp really used (MergeMatte's choke / soften) for the ideal; the band and the interior
    stay those of `alpha`, so before / after are measured on the same pixels.
    Returns {band_px, mean_abs, luma} (mean_abs / luma in % of full scale; None without a band or an interior)."""
    a = np.clip(np.asarray(alpha, np.float32), 0, 1)
    band = (a > EDGE_LO) & (a < EDGE_HI)
    solid = (a >= EDGE_HI).astype(np.float32)
    n = int(band.sum())
    if n == 0 or solid.sum() == 0:
        return {"band_px": n, "mean_abs": None, "luma": None}
    pl = np.asarray(plate_srgb, np.float32)[..., :3]
    num = blur(pl * solid[..., None], spread)
    den = blur(solid, spread)[..., None]
    f = np.where(den > 1e-3, num / np.maximum(den, 1e-3), pl)
    bg = np.asarray(bg_srgb, np.float32)
    bg = np.broadcast_to(bg.reshape(1, 1, 3) if bg.ndim == 1 else bg[..., :3], pl.shape)
    ca = a if comp_alpha is None else np.clip(np.asarray(comp_alpha, np.float32), 0, 1)
    ideal = ca[..., None] * f + (1 - ca[..., None]) * bg
    d = np.asarray(comp_srgb, np.float32)[..., :3] - ideal
    db = d[band]
    return {"band_px": n, "mean_abs": round(float(np.abs(db).mean()) * 100, 2),
            "luma": round(float((db * REC709).sum(-1).mean()) * 100, 2)}


def edge_bg_region(plate_srgb: np.ndarray, alpha: np.ndarray, bg_srgb, k: Optional[Mapping[str, object]] = None,
                   rect: Optional[tuple] = None, max_px: int = 2_500_000) -> Dict[str, Any]:
    """„Tło w krawędzi” of a region before (a plain Merge over = the neutral knobs) and after MergeMatte with `k` – the
    same pixels as preview_region (scaled down the same way when the region is large)."""
    H, W = alpha.shape[:2]
    x, y, w, h = rect if rect else (0, 0, W, H)
    x, y = max(0, int(x)), max(0, int(y))
    w, h = max(1, min(int(w), W - x)), max(1, min(int(h), H - y))
    neutral = {**KNOB_DEFAULTS, "log_curve": lc.NONE}
    before, s = preview_region(plate_srgb, alpha, bg_srgb, neutral, (x, y, w, h), max_px)
    after, _ = preview_region(plate_srgb, alpha, bg_srgb, k, (x, y, w, h), max_px)
    sz = (before.shape[1], before.shape[0])
    rs = (lambda im: im) if s >= 1.0 else (lambda im: cv2.resize(im, sz, interpolation=cv2.INTER_AREA))
    pl = rs(np.ascontiguousarray(plate_srgb[y:y + h, x:x + w]))
    a = rs(np.ascontiguousarray(alpha[y:y + h, x:x + w]))
    b = np.asarray(bg_srgb, np.float32)
    b = rs(np.ascontiguousarray(b[y:y + h, x:x + w])) if b.ndim == 3 else b
    sp = EDGE_SPREAD * s
    kk = knobs(k)
    a2 = a   # the alpha MergeMatte composes with: choke (FilterErode port) + soften, px knobs at the preview's scale
    if kk["choke"] != 0 or kk["soften"] != 0:
        a2 = blur(filter_erode(a, kk["choke"] * s), kk["soften"] * s)
    return {"before": edge_bg(pl, a, before.astype(np.float32) / 255, b, sp),
            "after": edge_bg(pl, a, after.astype(np.float32) / 255, b, sp, comp_alpha=a2), "scale": round(s, 6)}


def checker_srgb(h: int, w: int, cell: int = 16) -> np.ndarray:
    """The checker background of the Comp view (#999 / #666, 16 px cells in plate px) as display sRGB."""
    yy, xx = np.mgrid[0:h, 0:w]
    odd = ((xx // cell + yy // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.float32)
    out[odd], out[~odd] = 0x99 / 255, 0x66 / 255
    return out


def hex_srgb(c: str) -> np.ndarray:
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
