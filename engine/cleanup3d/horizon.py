# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – horizon match (docs/PLAN_CLEANUP_3D.md, package P15).

The sea card is a level plane: a small error of the camera solve / the lens / the plane height shows as a projected
horizon that is not exactly where the plate has it, and the projected sea "steps" against the plate where the patch
meets it (sh060 2026-11-10: image difference 1.21 levels overall, 1.61 at the horizon). Ported from the experiment
removers_test/ (hz_detect.py = the edge detector, hz_step.py = the step measure, horizon_fix.py = a vertical shift that
fades out away from the horizon – the two-edge band of horizon_snap.py made steps and is not used).

Per frame:
1. predicted horizon = where a plane card meets the sky card in the frame (the cards' hard ownership on a column grid,
   as engine/cleanup3d/geom_gpu.weights without the seam) – only the centre of the search band;
2. plate horizon = the user's line of that frame (engine/curves.py JSON, drawn with static/curves.js), between two drawn
   frames their offsets from the predicted horizon interpolated; elsewhere detected automatically: the sea top (the
   LOWEST strong bright-above-dark edge, hz_detect.py) in a band of `horizon_search_px` around the prediction, away
   from the removal object, fitted by a robust quadratic; when that fails the nearest drawn line's offset;
3. projection horizon = the same edge in the projected keys, near the plate's line, per column (median-smoothed);
4. correction: the fill (and its weight) resampled with a vertical displacement d(x) at the horizon line that fades to 0
   over `horizon_fade_px` above and below (smoothstep, no hard band); pixels farther away are not touched;
5. report: the step = |projection horizon - plate horizon| (median / p90 px) in the columns where the removal object
   crosses the horizon, before and after the correction – run.json c3d.frames[f].horizon.

Coordinates: the images of one call are a crop or a preview grid of the plate: frame px = o + k * image px
(`grid` = (ox, oy, k)); every px value of the settings and of the report is in frame px.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d import types as T

SCHEMA = "matte_desk/c3d_horizon@1.0"
DEFAULTS: Dict[str, Any] = {
    "horizon_match": False,     # the correction in the render and the live preview (C5 „Dopasuj horyzont”)
    "horizon_fade_px": 120.0,   # the displacement fades to 0 over this many px above / below the horizon line
    "horizon_search_px": 60.0,  # half-height of the band around the predicted horizon where the edges are searched
}
LIMITS = {"horizon_fade_px": (8.0, 800.0), "horizon_search_px": (8.0, 400.0)}
GRID = 8                     # px between the measured columns (frame px)
FRAC = 0.45                  # hz_detect.py: a peak counts when >= FRAC x the strongest in the profile
PLATE_STRENGTH = 5.0         # edge strength (peak / median |profile|) of a plate column (hz_step.py: 5)
FILL_STRENGTH = 4.0          # ... of a projection column (hz_step.py: 4)
MIN_POINTS = 12              # plate columns needed for the automatic fit
MIN_FILL_POINTS = 5          # projection columns needed for a correction
MIN_PX = 0.25                # median |offset| below which nothing is resampled
OBJ_MARGIN_X = 30.0          # frame px: plate columns this close to the removal object are not measured
STEP_ROWS = 3.0              # frame px: a column is "at the object" when the object is within this of the plate line
KEY_FULL = 0.999             # engine.cleanup.KEY_FULL (no import: the module stays light)


def settings(d: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The horizon settings of a settings dict, defaults for what is missing, numbers clamped to LIMITS."""
    s = dict(DEFAULTS)
    if d:
        s.update({k: d[k] for k in DEFAULTS if d.get(k) is not None})
    s["horizon_match"] = bool(s["horizon_match"])
    for k, (lo, hi) in LIMITS.items():
        s[k] = float(min(hi, max(lo, float(s[k]))))
    return s


# ----------------------------------------------------------------------------------------------- the user's lines
def lines_file(work_dir: Path) -> Path:
    """`<work>/inpaint/c3d_horizon.json` – the horizon lines the user drew, per plate frame (project data)."""
    return T.root_dir(work_dir) / "c3d_horizon.json"


def _check_curve(curve: Any) -> Dict[str, Any]:
    from engine import curves
    if not (isinstance(curve, (dict, list))):
        raise ValueError("Linia horyzontu: oczekiwana krzywa (type: bezier)")
    sp = curves.normalize(curve)
    if sp.get("closed"):
        raise ValueError("Linia horyzontu musi być otwartą krzywą")
    if len(sp.get("pts") or []) < 2:
        raise ValueError("Linia horyzontu potrzebuje co najmniej dwóch punktów")
    return sp


def load_lines(work_dir: Path) -> Dict[int, Dict[str, Any]]:
    """{plate frame: curve} of the project; a missing file = none, a broken one = ValueError (Polish)."""
    import json
    p = lines_file(work_dir)
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Uszkodzony plik {p.name}: {e}")
    return {int(k): _check_curve(v) for k, v in (d.get("lines") or {}).items()}


def save_line(work_dir: Path, frame: int, curve: Optional[Any]) -> Dict[int, Dict[str, Any]]:
    """Set (curve) or remove (None) the user's horizon line of `frame`; returns all lines."""
    from engine.jsonio import write_json
    lines = load_lines(work_dir)
    if curve is None:
        lines.pop(int(frame), None)
    else:
        lines[int(frame)] = _check_curve(curve)
    p = lines_file(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {"schema": SCHEMA, "lines": {str(f): lines[f] for f in sorted(lines)}}, indent=1, ensure_ascii=False)
    return lines


def curve_rows(curve: Mapping[str, Any], xs: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Rows of a horizon curve at columns xs (frame px; clamped outside its x range) and that range."""
    from engine import curves
    xy, _u, _p = curves.sample(curves.normalize(curve), step=2.0)
    o = np.argsort(xy[:, 0], kind="stable")
    x, y = xy[o, 0], xy[o, 1]
    x, i = np.unique(x, return_index=True)
    y = y[i]
    if len(x) == 1:
        return np.full(len(xs), float(y[0])), (float(x[0]), float(x[0]))
    return np.interp(xs, x, y), (float(x[0]), float(x[-1]))


# ----------------------------------------------------------------------------------------------- edge detection
def gray255(img: np.ndarray) -> np.ndarray:
    """(h, w) float32 0..255 luminance of a BGR image in model space (0..1)."""
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        a = a[..., 0] * 0.114 + a[..., 1] * 0.587 + a[..., 2] * 0.299
    return np.clip(a, 0.0, 1.0) * 255.0


def pick(prof: np.ndarray, frac: float = FRAC) -> Tuple[float, float]:
    """hz_detect.py: index (sub-pixel) of the LOWEST strong peak of a signed edge profile (bright above darker) and its
    strength relative to the profile's median |value| – near the sun the horizon is a soft band with two edges and the
    sea top is the lower one."""
    n = len(prof)
    top = float(prof.max()) if n else 0.0
    if n < 3 or top <= 0:
        return (float(np.argmax(prof)) if n else 0.0), 0.0
    pk = [i for i in range(1, n - 1) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] >= frac * top]
    j = max(pk) if pk else int(np.argmax(prof))
    den = prof[j - 1] - 2 * prof[j] + prof[j + 1] if 0 < j < n - 1 else 0.0
    jj = j + (0.5 * (prof[j - 1] - prof[j + 1]) / den if 0 < j < n - 1 and abs(den) > 1e-9 else 0.0)
    return float(jj), float(prof[j] / (np.median(np.abs(prof)) + 1e-3))


def edge_rows(gray: np.ndarray, xs: np.ndarray, centre: np.ndarray, r: float, k: float = 1.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Sea-top rows near `centre` (+- r image px) at the columns xs of a gray image -> (rows, strengths); NaN / 0 where
    the centre is NaN or the window leaves the image. Blur and column window scale with k (frame px per image px)."""
    h, w = gray.shape
    ys = np.full(len(xs), np.nan)
    st = np.zeros(len(xs))
    ok = np.isfinite(centre)
    if not ok.any():
        return ys, st
    r = max(2.0, float(r))
    top = int(max(0, np.floor(np.nanmin(centre[ok]) - r - 12)))
    bot = int(min(h, np.ceil(np.nanmax(centre[ok]) + r + 13)))
    if bot - top < 5:
        return ys, st
    g = cv2.GaussianBlur(np.ascontiguousarray(gray[top:bot], np.float32), (0, 0), max(0.7, 2.0 / k))
    gy = -cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)              # positive where it is brighter above
    half = max(1, int(round(6.0 / k)))
    for i, (x, c) in enumerate(zip(xs, centre)):
        if not np.isfinite(c):
            continue
        lo, hi = int(max(1, c - r)), int(min(h - 1, c + r + 1))
        if hi - lo < 3:
            continue
        xi = int(round(x))
        prof = gy[lo - top:hi - top, max(0, xi - half):min(w, xi + half + 1)].mean(1)
        j, s = pick(prof)
        ys[i], st[i] = lo + j, s
    return ys, st


def robust_fit(x: np.ndarray, y: np.ndarray, floor: float = 2.0, iters: int = 5, deg: int = 2) -> Optional[np.ndarray]:
    """Robust polynomial of degree `deg` through (x, y), outliers rejected iteratively (hz_step.py); None with too
    few points."""
    if len(x) < MIN_POINTS:
        return None
    keep = np.ones(len(x), bool)
    q = None
    for _ in range(iters):
        if keep.sum() < MIN_POINTS:
            break
        q = np.polyfit(x[keep], y[keep], deg)
        e = np.abs(np.polyval(q, x) - y)
        keep = e < max(floor, 2.5 * float(np.median(e[keep])))
    return q


def _smooth(v: np.ndarray, n: int = 2) -> np.ndarray:
    return np.array([np.median(v[max(0, i - n):i + n + 1]) for i in range(len(v))]) if len(v) else v


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------------------------- the correction
def warp_rows(img: np.ndarray, d: np.ndarray, line: np.ndarray, fade: float) -> np.ndarray:
    """Output (x, y) samples img at (x, y + d(x) * smoothstep(1 - |y - line(x)| / fade)): the content at row
    line(x) + d(x) lands on line(x), the displacement fades to 0 over `fade` rows above and below; rows farther than
    `fade` from the line are copied unchanged. d, line: (w,) image px (NaN line = column untouched)."""
    h, w = img.shape[:2]
    line = np.asarray(line, np.float64)
    d = np.where(np.isfinite(line), np.asarray(d, np.float64), 0.0)
    ok = np.isfinite(line) & (np.abs(d) > 1e-6)
    if not ok.any() or fade <= 0:
        return img
    y0 = int(max(0, np.floor(line[ok].min() - fade)))
    y1 = int(min(h, np.ceil(line[ok].max() + fade) + 1))
    if y1 <= y0:
        return img
    Y = np.arange(y0, y1, dtype=np.float64)[:, None]
    ln = np.where(np.isfinite(line), line, -1e9)[None, :]
    fall = smoothstep(1.0 - np.abs(Y - ln) / float(fade))
    my = (Y + d[None, :] * fall).astype(np.float32)
    mx = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], my.shape).copy()
    out = img.copy()
    src = np.ascontiguousarray(img)
    out[y0:y1] = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(out[y0:y1].shape)
    return out


def _r1(v: Optional[float]) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else round(float(v), 2)


# ----------------------------------------------------------------------------------------------- the matcher
class Horizon:
    """Horizon match of one shot: the predicted horizon of the cards (cached per frame), the user's lines, the
    measurement and the correction of a fill. `lines` None = read from `work` (none without it)."""

    def __init__(self, geometry: Any, work: Optional[Path] = None, s: Optional[Mapping[str, Any]] = None,
                 lines: Optional[Mapping[int, Mapping[str, Any]]] = None):
        self.geo = geometry
        self.s = settings(s)
        self.lines = dict(lines) if lines is not None else (load_lines(Path(work)) if work is not None else {})
        cam = geometry.camera
        self.w, self.h = int(cam.width), int(cam.height)
        self.grid_x = np.arange(GRID // 2, self.w, GRID, dtype=np.float64)
        self._pred: "OrderedDict[int, np.ndarray]" = OrderedDict()

    # -- the predicted horizon (the cards)
    def predicted(self, frame: int) -> np.ndarray:
        """Rows (frame px, on grid_x) where a plane card meets the sky card below it (the topmost such boundary per
        column; NaN = none) – from the cards' hard ownership: the first card in order whose plane the ray hits in front
        of the camera, inside its extent."""
        if frame in self._pred:
            self._pred.move_to_end(frame)
            return self._pred[frame]
        geo = self.geo
        cards = list(geo.cards)
        out = np.full(len(self.grid_x), np.nan)
        sky = [i for i, c in enumerate(cards) if c.kind == "sky"]
        plane = [i for i, c in enumerate(cards) if c.kind != "sky"]
        if sky and plane:
            view = geo.frame_view(int(frame))
            PX, PY = np.meshgrid(self.grid_x, np.arange(self.h, dtype=np.float64))
            x, y = K_.view_rays(K_.NP, view, PX, PY)
            owner = np.full(PX.shape, -1, np.int32)
            for i, c in enumerate(cards):
                spec = geo.card_spec(c)
                hit = K_.plane_hit(K_.NP, x, y, view, spec)
                ext = geo._extent(c)
                if ext is not None:
                    e = K_.extent_mask(K_.NP, x, y, view, spec,
                                       {**ext, "canvas_arr": ext["canvas_u8"].astype(np.float64) / 255.0})
                    hit = hit & (e >= 0.5)
                owner = np.where((owner < 0) & hit, i, owner)
            is_sky, is_plane = np.isin(owner, sky), np.isin(owner, plane)
            tr = is_sky[:-1] & is_plane[1:]                     # sky at row r, a plane card at row r + 1
            anyc = tr.any(0)
            first = np.argmax(tr, axis=0)
            out = np.where(anyc, first + 0.5, np.nan)
        self._pred[frame] = out
        while len(self._pred) > 32:
            self._pred.popitem(last=False)
        return out

    def _residual(self, k: int) -> Optional[np.ndarray]:
        """The user's line of frame k minus the predicted horizon (grid_x; nearest value outside the line's x range)."""
        pred = self.predicted(k)
        rows, (xa, xb) = curve_rows(self.lines[k], self.grid_x)
        res = rows - pred
        inside = (self.grid_x >= xa - GRID) & (self.grid_x <= xb + GRID) & np.isfinite(res)
        if not inside.any():
            return None
        return np.interp(self.grid_x, self.grid_x[inside], res[inside])

    def user_rows(self, frame: int, near: bool = False) -> Optional[np.ndarray]:
        """The plate horizon from the user's lines (grid_x, frame px): the frame's own line; between two drawn frames
        their offsets from the predicted horizon interpolated in time. Outside the drawn range: None, or with `near`
        the nearest drawn line's offset on this frame's predicted horizon."""
        if not self.lines:
            return None
        fs = sorted(self.lines)
        f = int(frame)
        if f in self.lines:
            rows, _r = curve_rows(self.lines[f], self.grid_x)
            return rows
        pred = self.predicted(f)
        if not np.isfinite(pred).any():
            return None
        if fs[0] < f < fs[-1]:
            a = max(k for k in fs if k < f)
            b = min(k for k in fs if k > f)
            ra, rb = self._residual(a), self._residual(b)
            if ra is None or rb is None:
                res = ra if rb is None else rb
            else:
                t = (f - a) / float(b - a)
                res = ra * (1.0 - t) + rb * t
        elif near:
            res = self._residual(min(fs, key=lambda k: (abs(k - f), k)))
        else:
            return None
        return None if res is None else pred + res

    # -- measure + correct
    def match(self, frame: int, plate: np.ndarray, fill: np.ndarray, wgt: np.ndarray, obj: Optional[np.ndarray],
              grid: Tuple[float, float, float] = (0.0, 0.0, 1.0), linear: bool = False, correct: bool = True,
              lines: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """(fill, weight, report) of frame `frame`: the plate's and the projection's horizon measured in these images
        (a crop / preview grid: frame px = o + k * image px), the fill and the weight resampled so the projection's
        horizon lies on the plate's (`correct`). obj = the removal object (bool / 0..1, image size). `lines`: the
        report carries the lines in frame px (predicted / plate / fill) for the preview."""
        from engine import cleanup as cu
        ox, oy, k = float(grid[0]), float(grid[1]), float(grid[2])
        h, w = fill.shape[:2]
        s = self.s
        r = s["horizon_search_px"] / k
        fade = s["horizon_fade_px"] / k
        info: Dict[str, Any] = {"status": "ok", "source": None, "warped": False}
        step_i = max(1.0, GRID / k)
        xs = np.arange(step_i / 2.0, w - 0.5, step_i)
        fx = ox + k * xs                                        # the columns in frame px
        pred_g = self.predicted(int(frame))
        okp = np.isfinite(pred_g)
        if not okp.any():
            info["status"] = "no_horizon"
            return fill, wgt, info
        pred = (np.interp(fx, self.grid_x[okp], pred_g[okp]) - oy) / k
        if lines:
            info["predicted"] = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(self.grid_x[okp], pred_g[okp])]
        if np.nanmax(pred) < -r or np.nanmin(pred) > h + r:
            info["status"] = "no_horizon"
            return fill, wgt, info
        objm = np.zeros((h, w), bool) if obj is None else (np.asarray(obj, np.float32) > 0.02)
        pl_m = cu.to_model(np.asarray(plate, np.float32)[..., :3], linear)
        fl_m = cu.to_model(np.asarray(fill, np.float32)[..., :3], linear)
        # 1. the plate's horizon: the user's line, else the automatic fit, else the nearest user line
        yp = None
        user = self.user_rows(int(frame))
        if user is not None:
            yp = (np.interp(fx, self.grid_x, user) - oy) / k
            info["source"] = "user"
        else:
            yr, sr = edge_rows(gray255(pl_m), xs, pred, r, k)
            mx = max(1, int(round(OBJ_MARGIN_X / k)))
            near_obj = cv2.dilate(objm.astype(np.uint8), np.ones((1, 2 * mx + 1), np.uint8)) > 0 if objm.any() else objm
            free = np.ones(len(xs), bool)
            if near_obj.any():
                for i, (x, y) in enumerate(zip(xs, yr)):
                    if np.isfinite(y):
                        xi = int(round(x))
                        free[i] = not near_obj[int(max(0, y - r)):int(min(h, y + r + 1)), xi].any()
            good = free & np.isfinite(yr) & (sr > PLATE_STRENGTH)
            info["plate_pts"] = int(good.sum())
            span = float(xs[good].max() - xs[good].min()) if good.any() else 0.0
            q = robust_fit(xs[good], yr[good], floor=max(0.5, 2.0 / k), deg=2 if span > 0.4 * w else 1) if good.any() else None
            if q is not None:
                yp = np.polyval(q, xs)
                info["source"] = "auto"
            else:
                near = self.user_rows(int(frame), near=True)
                if near is not None:
                    yp = (np.interp(fx, self.grid_x, near) - oy) / k
                    info["source"] = "user_near"
        if yp is None:
            info["status"] = "no_plate_line"
            return fill, wgt, info
        if lines:
            info["plate"] = [[round(float(a), 1), round(float(oy + k * b), 1)] for a, b in zip(fx, yp)]
        # 2. the projection's horizon near the plate's line, where the keys fully cover
        yi = np.clip(np.round(yp).astype(int), 0, h - 1)
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        inside = (yp >= 0) & (yp <= h - 1)
        covered = inside & (np.asarray(wgt)[yi, xi] >= KEY_FULL)
        yf, sf = edge_rows(gray255(fl_m), xs, np.where(covered, yp, np.nan), r, k)
        d = yf - yp
        good = covered & np.isfinite(d) & (sf > FILL_STRENGTH) & (np.abs(d) < r - 1.0)
        info["fill_pts"] = int(good.sum())
        if good.sum() < MIN_FILL_POINTS:
            info["status"] = "no_fill_line"
            return fill, wgt, info
        if lines:
            info["fill"] = [[round(float(a), 1), round(float(oy + k * b), 1)] for a, b, g in zip(fx, yf, good) if g]
        ds = _smooth(d[good])
        dcol = np.interp(np.arange(w, dtype=np.float64), xs[good], ds)          # every column, constant beyond
        line = np.interp(np.arange(w, dtype=np.float64), xs, yp)
        info["offset_px"] = _r1(float(np.median(ds)) * k)
        info["offset_max_px"] = _r1(float(np.abs(ds).max()) * k)
        # 3. the step where the object crosses the horizon (hz_step.py), before / after
        sr_ = max(1.0, STEP_ROWS / k)
        at_obj = np.array([inside[i] and objm[int(max(0, yp[i] - sr_)):int(min(h, yp[i] + sr_ + 1)), xi[i]].any()
                           for i in range(len(xs))]) if objm.any() else np.zeros(len(xs), bool)

        def step_of(dd: np.ndarray, ok: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
            sel = at_obj & ok
            if sel.sum() < 3:
                return None, None
            a = np.abs(dd[sel]) * k
            return _r1(float(np.median(a))), _r1(float(np.percentile(a, 90)))
        info["step_px"], info["step_p90_px"] = step_of(d, good)
        info["step_cols"] = int((at_obj & good).sum())
        out_f, out_w = fill, wgt
        need = abs(float(np.median(ds))) * k >= MIN_PX or float(np.abs(ds).max()) * k >= 4 * MIN_PX
        if correct and need:
            out_f = warp_rows(np.asarray(fill, np.float32), dcol, line, fade)
            out_w = warp_rows(np.asarray(wgt, np.float32), dcol, line, fade)
            info["warped"] = True
            fl2 = cu.to_model(np.asarray(out_f, np.float32)[..., :3], linear)
            y2, s2 = edge_rows(gray255(fl2), xs, np.where(covered, yp, np.nan), r, k)
            d2 = y2 - yp
            info["step_after_px"], info["step_after_p90_px"] = step_of(d2, good & np.isfinite(d2) & (s2 > FILL_STRENGTH))
        else:
            info["step_after_px"], info["step_after_p90_px"] = info["step_px"], info["step_p90_px"]
        return out_f, out_w, info


# ----------------------------------------------------------------------------------------------- hooks
def apply_keyfill(kf: Any, f: int, fr: Mapping[str, Any], fill: np.ndarray, wgt: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The render's hook (render.KeyFill, before the blur / tone): the projected fill of frame f in its ROI corrected
    to the plate's horizon when „Dopasuj horyzont” is on; the report goes to the frame's stats ("horizon")."""
    if not kf.s.get("horizon_match") or not (np.asarray(wgt) > 0).any():
        return fill, wgt
    t0 = time.time()
    hz = getattr(kf, "_hz", None)
    if hz is None:
        hz = kf._hz = Horizon(kf.scene.geometry, kf.scene.work, kf.s)
    x0, y0, x1, y1 = fr["roi"]
    plate = kf.plate(f)[y0:y1, x0:x1]
    out_f, out_w, info = hz.match(int(f), plate, fill, wgt, fr["obj"][y0:y1, x0:x1], (x0, y0, 1.0), kf.linear)
    kf.stats.setdefault(f, {})["horizon"] = info
    kf.seconds["horizon"] = kf.seconds.get("horizon", 0.0) + time.time() - t0
    return out_f, out_w


def _grid_obj(remove_alpha: Optional[np.ndarray], shape: Tuple[int, int], step: int) -> Optional[np.ndarray]:
    if remove_alpha is None:
        return None
    a = np.asarray(remove_alpha, np.float32)
    if a.ndim == 3:
        a = a.max(axis=2)
    gh, gw = shape
    if step > 1:
        a = a[step // 2::step, step // 2::step][:gh, :gw]
    if a.shape != (gh, gw):
        a = cv2.resize(a, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return a > 0.02


def blend_fix(scene: Any, frame: int, fill: np.ndarray, wgt: np.ndarray, plate: np.ndarray,
              remove_alpha: Optional[np.ndarray], s: Optional[Mapping[str, Any]], step: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    """The live preview's hook (render.blend_inputs): fill / plate in model space on the preview grid (`step` frame px
    per preview px, the grid's first sample at step // 2), corrected like the render when „Dopasuj horyzont” is on."""
    hs = settings(s)
    if not hs["horizon_match"]:
        return fill, wgt
    try:
        hz = Horizon(scene.geometry, scene.work, hs)
    except ValueError:                                        # a broken lines file: the preview stays uncorrected
        return fill, wgt
    o = float(step // 2) if step > 1 else 0.0
    out_f, out_w, _info = hz.match(int(frame), plate, fill, wgt, _grid_obj(remove_alpha, wgt.shape, step),
                                   (o, o, float(step)), linear=False)
    return out_f, out_w


def preview(scene: Any, frame: int, plate: np.ndarray, remove_alpha: Optional[np.ndarray],
            s: Optional[Mapping[str, Any]] = None, linear: bool = False, max_side: int = 1600) -> Dict[str, Any]:
    """The horizon of frame `frame` measured on the preview grid (GET /api/cleanup3d/horizon/{frame}): the report of
    `Horizon.match` with the lines in frame px (predicted / plate / fill), the user's line of the frame and the drawn
    frames. Measured whether or not the correction is on."""
    from engine import cleanup as cu
    from engine.cleanup3d import render as RD
    hs = settings(s)
    w, h = scene.size
    step = max(1, int(np.ceil(max(w, h) / float(max_side))))
    hz = Horizon(scene.geometry, scene.work, hs)
    if step > 1:
        fill, wgt = RD._project_grid(scene, int(frame), step)
        gh, gw = wgt.shape
        pl = cv2.resize(cu.to_model(np.asarray(plate, np.float32)[..., :3], linear), (gw, gh), interpolation=cv2.INTER_AREA)
        fill = cu.to_model(fill, linear)
    else:
        fill, wgt = RD.project_frame(scene.geometry, scene.cards, scene.keys_by_card, int(frame), scene.cache,
                                     cache=scene.cache)
        fill = cu.to_model(fill, linear)
        pl = cu.to_model(np.asarray(plate, np.float32)[..., :3], linear)
    o = float(step // 2) if step > 1 else 0.0
    _f, _w, info = hz.match(int(frame), pl, fill, wgt, _grid_obj(remove_alpha, wgt.shape, step), (o, o, float(step)),
                            linear=False, lines=True)
    info.update({"frame": int(frame), "settings": hs, "user": hz.lines.get(int(frame)), "drawn": sorted(hz.lines)})
    return info
