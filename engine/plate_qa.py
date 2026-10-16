# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
r"""engine/plate_qa.py - Z26: control of damaged frames in a plate (stage 0, before anything is cut; user rule
"fix the plate once, use the fixed plate everywhere").

Per frame of the sequence (analysis on the frame downscaled to <= MEASURE_SIDE px, linear values for EXR / display
values for 8/16-bit sources):
- file missing / unreadable / NaN-Inf / size differs from the sequence  -> `missing`, `unreadable`, `nan`, `size`;
- exposure jump: median log2 luminance of the frame vs the median of its neighbours t±1, t±2 (self excluded) –
  deviation > `ev_jump` while the neighbours agree with each other (|L(t-1) - L(t+1)| < `ev_smooth`: a smooth trend
  is never an alarm; a real cut / light change moves the neighbours too) -> `exposure_jump`; `overexposed_frac`
  (fraction of pixels >= 1 in any channel) reported separately;
- image jump: the frame's interpolation error (MSE of the frame vs the mean of t-1 / t+1, on normalised gray) is
  > `interp_ratio` x the error between the two neighbours themselves (a damaged / shifted frame is far from both
  neighbours although they agree) and above the noise floor `interp_min`; the phase-correlation shift vs t-1 is
  reported (`shift_px`, full-resolution px) and flagged when it deviates from the neighbours' trend by > `shift_px`
  -> `image_jump`. A real camera move (SH010 X1 F4) moves t-1 -> t+1 as well -> no alarm;
- duplicate: the frame equals its previous neighbour (max |diff| < `dup_eps` on the small image) -> `duplicate`.
Thresholds: defaults.json -> "plate_qa". Report: {frames: {f: {...}}, flagged: [...], thresholds, pattern}; the UI /
CLI decide per flagged frame (repair / keep) – engine/plate_fix.py applies the repair.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MEASURE_SIDE = 512
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "ev_jump": 0.5,        # EV: |L(t) - median(L of neighbours)| above this = exposure jump
    "ev_smooth": 0.25,     # EV: neighbours t-1 / t+1 must agree within this for the alarm (smooth trend otherwise)
    "interp_ratio": 3.0,   # frame vs neighbour-mean MSE > ratio x MSE(t-1, t+1)
    "interp_min": 2e-3,    # noise floor of the MSE (normalised gray 0..1 of the small image)
    "shift_px": 8.0,       # phase-correlation shift deviating from the trend by more than this (full-res px)
    "dup_eps": 1e-6,       # max |diff| on the small image below this = duplicate of the previous frame
    "overexposed_frac": 0.5,
}
FLAGS = ("missing", "unreadable", "nan", "size", "exposure_jump", "image_jump", "duplicate", "overexposed")


def frame_path(pattern: str, f: int) -> Path:
    from engine.ingest import ProxyGenerator
    return ProxyGenerator.resolve_frame_path(pattern, f)


def read_frame_linear(path: Path) -> Optional[np.ndarray]:
    """BGR float32: EXR scene-linear values; 8 / 16-bit sources scaled to 0..1. None when unreadable."""
    try:
        if path.suffix.lower() == ".exr":
            from engine.exr_io import read_half_rgb, read_rgb
            try:
                a = read_half_rgb(path)
            except Exception:
                a = None
            if a is None:
                a = read_rgb(path)
            return np.asarray(a[..., :3], np.float32)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img[..., :3].astype(np.float32)
        return img / (65535.0 if img.max() > 255 else 255.0)
    except Exception:
        return None


def small_of(img: np.ndarray, side: int = MEASURE_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, side / float(max(h, w)))
    return img if s >= 1.0 else cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)


def log_luma_median(img: np.ndarray) -> float:
    lum = np.clip(img.mean(axis=2), 1e-6, None)
    return float(np.median(np.log2(lum)))


def norm_gray(img: np.ndarray) -> np.ndarray:
    """Gray normalised by its own median (exposure-independent geometry compare), float32."""
    g = img.mean(axis=2).astype(np.float32)
    m = float(np.median(g))
    return g / max(m, 1e-6)


def phase_shift(g0: np.ndarray, g1: np.ndarray) -> Tuple[float, float]:
    win = cv2.createHanningWindow((g0.shape[1], g0.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(g0.astype(np.float32), g1.astype(np.float32), win)
    return float(dx), float(dy)


def analyze_frames(frames: Sequence[int], load: Callable[[int], Optional[np.ndarray]],
                   thresholds: Optional[Dict[str, float]] = None,
                   progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Pure analysis: `load(f)` -> BGR float frame or None. Returns the report dict (frames keyed by int)."""
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    frames = list(frames)
    n = len(frames)
    per: Dict[int, Dict[str, Any]] = {}
    smalls: Dict[int, np.ndarray] = {}
    grays: Dict[int, np.ndarray] = {}
    sizes: List[Tuple[int, int]] = []
    t0 = time.time()
    for i, f in enumerate(frames):
        if progress:
            progress(i + 1, n)
        rec: Dict[str, Any] = {"flags": []}
        img = load(f)
        if img is None:
            rec["flags"].append("missing")
            per[f] = rec
            continue
        if not np.isfinite(img).all():
            rec["flags"].append("nan")
            rec["nan_px"] = int((~np.isfinite(img)).any(axis=2).sum())
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        rec["size"] = [int(img.shape[1]), int(img.shape[0])]
        sizes.append((img.shape[1], img.shape[0]))
        sm = small_of(img)
        smalls[f] = sm
        grays[f] = norm_gray(sm)
        rec["log_luma"] = round(log_luma_median(sm), 4)
        rec["overexposed_frac"] = round(float((sm >= 1.0).any(axis=2).mean()), 4)
        per[f] = rec
    if not sizes:
        return {"frames": per, "flagged": frames, "thresholds": thr, "seconds": round(time.time() - t0, 1)}
    ref_size = max(set(sizes), key=sizes.count)
    scale_small = min(1.0, MEASURE_SIDE / float(max(ref_size)))
    for f in frames:
        rec = per[f]
        if "size" in rec and tuple(rec["size"]) != ref_size:
            rec["flags"].append("size")
            smalls.pop(f, None); grays.pop(f, None)
        if rec.get("overexposed_frac", 0.0) >= thr["overexposed_frac"]:
            rec["flags"].append("overexposed")
    ok = [f for f in frames if f in smalls]
    idx = {f: i for i, f in enumerate(ok)}
    # exposure: vs neighbours t±1, t±2 (ok frames only)
    L = {f: per[f]["log_luma"] for f in ok}
    for f in ok:
        i = idx[f]
        nb = [ok[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(ok)]
        if len(nb) < 2:
            continue
        med = float(np.median([L[g] for g in nb]))
        dev = L[f] - med
        per[f]["ev_dev"] = round(dev, 4)
        prev_, next_ = (ok[i - 1] if i > 0 else None), (ok[i + 1] if i + 1 < len(ok) else None)
        smooth = (prev_ is not None and next_ is not None and abs(L[prev_] - L[next_]) < thr["ev_smooth"]) or \
                 (prev_ is None or next_ is None)
        if abs(dev) > thr["ev_jump"] and smooth:
            per[f]["flags"].append("exposure_jump")
    # geometry: interpolation error vs neighbours, phase shift, duplicates
    shifts: Dict[int, Tuple[float, float]] = {}
    for i, f in enumerate(ok):
        if i > 0:
            p = ok[i - 1]
            dx, dy = phase_shift(grays[p], grays[f])
            shifts[f] = (dx / scale_small, dy / scale_small)
            per[f]["shift_px"] = [round(shifts[f][0], 2), round(shifts[f][1], 2)]
            d = float(np.abs(smalls[f] - smalls[p]).max())
            per[f]["diff_prev_max"] = round(d, 6)
            if d < thr["dup_eps"]:
                per[f]["flags"].append("duplicate")
    for i, f in enumerate(ok):
        if 0 < i < len(ok) - 1:
            p, q = ok[i - 1], ok[i + 1]
            gp, gq, gf = grays[p], grays[q], grays[f]
            err_self = float(np.mean((gf - 0.5 * (gp + gq)) ** 2))
            err_nb = float(np.mean((gp - gq) ** 2))
            per[f]["interp_err"] = round(err_self, 6)
            per[f]["neighbour_err"] = round(err_nb, 6)
            jump = err_self > thr["interp_min"] and err_self > thr["interp_ratio"] * err_nb
            # single-frame displacement: the shift INTO f breaks the motion of the frame before it (s(f) - s(f-1))
            # and the shift into f+1 breaks it the opposite way (s(f+1) - s(f+2)); a camera reversal (SH010 plate 03
            # F4..F6: -24, -22, +24) keeps s(f) ~ s(f-1) and a cut moves once and stays -> neither is flagged
            s_f, s_prev, s_next, s_next2 = shifts.get(f), shifts.get(p), shifts.get(q), shifts.get(ok[i + 2]) if i + 2 < len(ok) else None
            if s_f is not None and s_prev is not None:
                d_before = np.asarray(s_f) - np.asarray(s_prev)
                per[f]["shift_dev_px"] = round(float(np.hypot(*d_before)), 2)
                if s_next is not None and s_next2 is not None and float(np.hypot(*d_before)) > thr["shift_px"]:
                    d_after = np.asarray(s_next) - np.asarray(s_next2)
                    opposite = float(np.hypot(*d_after)) > thr["shift_px"] and \
                        float(np.hypot(*(d_before + d_after))) < 0.5 * float(np.hypot(*d_before))
                    if opposite:
                        jump = True
            if jump and "duplicate" not in per[f]["flags"]:
                per[f]["flags"].append("image_jump")
    flagged = [f for f in frames if per[f]["flags"]]
    return {"frames": per, "flagged": flagged, "thresholds": thr, "seconds": round(time.time() - t0, 1),
            "ref_size": list(ref_size), "n": n}


def analyze_plate(pattern: str, frames: Sequence[int], thresholds: Optional[Dict[str, float]] = None,
                  progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    rep = analyze_frames(frames, lambda f: read_frame_linear(frame_path(pattern, f)), thresholds, progress)
    rep["pattern"] = pattern
    return rep


def save_report(work: Path, plate: str, rep: Dict[str, Any]) -> Path:
    d = Path(work) / "plate_qa"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{plate}.json"
    out = dict(rep)
    out["frames"] = {str(k): v for k, v in rep["frames"].items()}
    out["plate"] = plate
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def load_report(work: Path, plate: str) -> Optional[Dict[str, Any]]:
    p = Path(work) / "plate_qa" / f"{plate}.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    rep["frames"] = {int(k): v for k, v in rep["frames"].items()}
    return rep


def suggest_method(rec: Dict[str, Any]) -> str:
    """Default repair per flag: missing / unreadable / nan / size / duplicate / image jump -> neighbours (mean or flow),
    exposure jump alone -> gain."""
    fl = set(rec.get("flags") or [])
    if fl & {"missing", "unreadable", "nan", "size", "duplicate", "image_jump"}:
        sh = rec.get("shift_px") or [0, 0]
        return "flow" if float(np.hypot(*sh)) > 4.0 else "mean"
    if "exposure_jump" in fl:
        return "gain"
    return "keep"


# ----------------------------------------------------------------------------- decisions + gate of stage 1 (UI, Z26)
DECISIONS = ("mean", "flow", "gain", "keep")
BLOCKING = ("undecided", "pending")


def _norm(p: Optional[str]) -> str:
    return Path(p).as_posix().lower() if p else ""


def save_decisions(work: Path, plate: str, decisions: Dict[str, str]) -> Dict[str, Any]:
    """Store the user's decision per flagged frame ({frame: mean|flow|gain|keep}) in the report; frames not flagged
    are refused, an empty method removes the decision. Returns the updated report."""
    rep = load_report(work, plate)
    if rep is None:
        raise FileNotFoundError(f"brak raportu kontroli klatek dla {plate}")
    flagged = {int(f) for f in rep.get("flagged") or []}
    dec = {str(k): v for k, v in (rep.get("decisions") or {}).items()}
    for f, m in decisions.items():
        if int(f) not in flagged:
            raise ValueError(f"klatka {f} nie jest oflagowana")
        if not m:
            dec.pop(str(int(f)), None)
        elif m not in DECISIONS:
            raise ValueError(f"metoda naprawy: {'|'.join(DECISIONS)}")
        else:
            dec[str(int(f))] = m
    rep["decisions"] = dec
    save_report(work, plate, rep)
    return rep


def gate_state(rep: Optional[Dict[str, Any]], current: str, fix_rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """State of one plate for the stage-1 gate (like Z18): every flagged frame needs a decision and the decided repairs
    must be written (`<plate>_fix` in use). `current` = the plate's pattern now, `fix_rec` = project.json
    plate_fix[plate] ({source, pattern, decisions, at}) of the last written repair.
    state: none (not checked) | clean | kept (all flags kept consciously) | fixed | undecided | pending (repairs
    decided but not written / changed since the write) | other (report of another file – check again)."""
    if rep is None:
        return {"state": "none", "blocking": False}
    flagged = [int(f) for f in rep.get("flagged") or []]
    dec = {str(k): v for k, v in (rep.get("decisions") or {}).items() if int(k) in flagged}
    undecided = [f for f in flagged if str(f) not in dec]
    repairs = {k: v for k, v in dec.items() if v != "keep"}
    same = _norm(rep.get("pattern")) == _norm(current)
    fixed = bool(fix_rec) and _norm(fix_rec.get("source")) == _norm(rep.get("pattern")) and \
        _norm(fix_rec.get("pattern")) == _norm(current)
    written = {str(k): v for k, v in ((fix_rec or {}).get("decisions") or {}).items() if v != "keep"} if fixed else {}
    if not same and not fixed:
        state = "other"
    elif undecided:
        state = "undecided"
    elif repairs != written:
        state = "pending"
    elif fixed and repairs:
        state = "fixed"
    else:
        state = "kept" if flagged else "clean"
    return {"state": state, "blocking": state in BLOCKING, "flagged": flagged, "undecided": undecided,
            "decisions": dec, "repairs": repairs, "written": written, "source": rep.get("pattern"), "fixed": fixed}
