# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate CONTENT check (NEXT_STEPS „sprawdzanie zgodności plate'ów przy ładowaniu”, user report 2026-10-17): a backlit /
clear plate of the right resolution but from another shot, or with the frame shifted, used to pass silently.

Method (exposure-independent – backlit and clear plates have other exposures and no object): per frame, luminance →
log → gradient magnitude at ≤ WORK_SIDE px; the beauty's and the plate's gradient images are compared by phase
correlation (shift, px of the full plate) and by normalised cross-correlation after that shift (score: same scene ≈ edges
in the same places). A few frames spread over the common range; the median decides.

    state  ok       same scene, shift <= SHIFT_OK_PX
           shifted  same scene, shifted by more  (→ „Dopasuj” = the existing alignment)
           other    low score: another shot – not used by the stages (like a wrong resolution)
           unknown  no common frame / unreadable

Also the frame range against the beauty (missing frames = a note). BG (the comp target) and depth are not checked
(another image by design). Thresholds: `SCORE_OTHER`, `SHIFT_OK_PX` – calibrated on SH010 (tools/plate_match_calib.py).
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORK_SIDE = 768          # long side of the compared gradient images
N_FRAMES = 3             # frames compared (spread over the common range)
SCORE_OTHER = 0.40       # median score below: another shot. SH010 2026-11-06: same scene 0.60–0.73, another shot 0.05–0.15, a mirrored beauty 0.16–0.29
SHIFT_OK_PX = 4.0        # full-plate px; above = shifted
CHECKED = ("backlit", "cleanplate", "clean_plate")


def is_checked(name: str) -> bool:
    return name.startswith(CHECKED)


def frames_of(pattern: str) -> Dict[int, str]:
    """frame number -> file of a #### / %04d sequence."""
    rx = re.escape(pattern).replace(re.escape("####"), r"(\d{4})").replace(re.escape("%04d"), r"(\d{4})")
    out = {}
    for f in glob.glob(pattern.replace("####", "[0-9]" * 4).replace("%04d", "[0-9]" * 4)):
        m = re.fullmatch(rx, f.replace("\\", "/")) or re.fullmatch(rx, f)
        if m:
            out[int(m.group(1))] = f
    return out


def _lum(path: str) -> Optional[np.ndarray]:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.dtype == np.uint8:
        img = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        img = raw.astype(np.float32) / 65535.0
    else:
        img = raw.astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.0722, 0.7152, 0.2126], np.float32)   # BGR → luminance
    return img


def grad_image(lum: np.ndarray, side: int = WORK_SIDE) -> Tuple[np.ndarray, float]:
    """(normalised gradient magnitude of log luminance at <= side px, scale to full px)."""
    h, w = lum.shape
    s = min(1.0, side / max(h, w))
    small = cv2.resize(lum, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else lum
    lg = np.log(np.maximum(small, 1e-4))
    gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    g = np.minimum(g, np.percentile(g, 99.5))          # a few hot edges must not rule the correlation
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.astype(np.float32), 1.0 / s


def compare(a: np.ndarray, b: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """((dx, dy) of b against a in the gradient image's px, NCC after undoing it)."""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    bb = cv2.warpAffine(b, m, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mg = int(np.ceil(max(abs(dx), abs(dy)))) + 4      # the borders the shift invented do not count
    if 2 * mg + 8 > min(a.shape):                     # a tiny image / a huge shift: nothing left to compare
        return (float(dx), float(dy)), 0.0
    aa, bc = a[mg:-mg, mg:-mg], bb[mg:-mg, mg:-mg]
    aa = (aa - aa.mean()) / (aa.std() + 1e-6)
    bc = (bc - bc.mean()) / (bc.std() + 1e-6)
    return (float(dx), float(dy)), float((aa * bc).mean())


def pick_frames(common: Sequence[int], n: int = N_FRAMES) -> List[int]:
    c = sorted(common)
    if len(c) <= n:
        return c
    return [c[int(round(i * (len(c) - 1) / (n - 1)))] for i in range(n)]


def check(beauty_pattern: str, plate_pattern: str, n: int = N_FRAMES) -> Dict[str, Any]:
    """Content check of a backlit / clear plate against the beauty (module docstring)."""
    bf, pf = frames_of(beauty_pattern), frames_of(plate_pattern)
    out: Dict[str, Any] = {"plate_frames": [min(pf), max(pf)] if pf else None, "plate_count": len(pf),
                           "beauty_frames": [min(bf), max(bf)] if bf else None}
    missing = sorted(set(bf) - set(pf))
    out["missing"] = len(missing)
    if missing:
        out["missing_first"] = missing[:5]
    common = sorted(set(bf) & set(pf))
    if not common:
        return {**out, "state": "unknown", "why": "brak wspólnych numerów klatek z beauty"}
    rows = []
    for f in pick_frames(common, n):
        la, lb = _lum(bf[f]), _lum(pf[f])
        if la is None or lb is None or la.shape != lb.shape:
            continue
        ga, k = grad_image(la)
        gb, _ = grad_image(lb)
        (dx, dy), score = compare(ga, gb)
        rows.append({"frame": f, "dx": round(dx * k, 1), "dy": round(dy * k, 1), "score": round(score, 3)})
    if not rows:
        return {**out, "state": "unknown", "why": "nie udało się odczytać klatek (inna rozdzielczość?)"}
    score = float(np.median([r["score"] for r in rows]))
    dx = float(np.median([r["dx"] for r in rows]))
    dy = float(np.median([r["dy"] for r in rows]))
    shift = float(np.hypot(dx, dy))
    out.update(frames=rows, score=round(score, 3), dx=round(dx, 1), dy=round(dy, 1), shift_px=round(shift, 1))
    if score < SCORE_OTHER:
        state = "other"
    elif shift > SHIFT_OK_PX:
        state = "shifted"
    else:
        state = "ok"
    return {**out, "state": state}


def describe(r: Optional[Dict[str, Any]]) -> Optional[str]:
    """One Polish line for stage 0 (EN in static/i18n/en.json); None = nothing to say."""
    if not r:
        return None
    st = r.get("state")
    miss = f"; brak {r['missing']} klatek z zakresu beauty" if r.get("missing") else ""
    if st == "other" and r.get("override"):
        return f"⚠ inne ujęcie? (podobieństwo {r['score']:.2f}) – używany mimo to (Twoja decyzja){miss}"
    if st == "other":
        return f"✗ inne ujęcie? (podobieństwo {r['score']:.2f}) – nie używany w etapach{miss}"
    if st == "shifted":
        return f"⚠ przesunięty o {r['shift_px']:.0f} px względem beauty – „Dopasuj” (wyrównanie){miss}"
    if st == "ok":
        return f"✓ ta sama scena (podobieństwo {r['score']:.2f}){miss}"
    return f"? {r.get('why', 'nie sprawdzono')}{miss}"


# ----------------------------------------------------------------------------- cache in the project
def cache_key(beauty_pattern: str, plate_pattern: str) -> str:
    def stamp(p: str) -> str:
        fs = sorted(frames_of(p).items())
        if not fs:
            return "0"
        st = Path(fs[0][1]).stat()
        return f"{len(fs)}:{st.st_size}:{int(st.st_mtime)}"
    return f"{beauty_pattern}|{stamp(beauty_pattern)}|{plate_pattern}|{stamp(plate_pattern)}"


CACHE_NAME = "plate_match.json"   # in the tool's work folder (<work>/matte_desk/) – never the shot manifest
_LOCK = None
_RUNNING: set = set()


def _lock():
    global _LOCK
    if _LOCK is None:
        import threading
        _LOCK = threading.Lock()
    return _LOCK


def _load(work_dir: Path) -> Dict[str, Any]:
    import json
    try:
        return json.loads((Path(work_dir) / CACHE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


OVERRIDES = "_overrides"   # {plate pattern: true} – „użyj mimo to”: the user says a plate flagged „other” is this shot


def cached(work_dir: Optional[Path], beauty_pattern: str, plate_pattern: str) -> Optional[Dict[str, Any]]:
    """The stored result for exactly these sequences (key = patterns + frame count / size / mtime of the first frame),
    with `override: True` when the user chose „użyj mimo to” for this plate."""
    if not work_dir or not beauty_pattern or not plate_pattern:
        return None
    d = _load(Path(work_dir))
    r = d.get(cache_key(beauty_pattern, plate_pattern))
    if r is None:
        return None
    return {**r, "override": True} if (d.get(OVERRIDES) or {}).get(plate_pattern) else r


def set_override(work_dir: Path, plate_pattern: str, on: bool) -> None:
    """„Użyj mimo to” (on) / back to the check's verdict (off). Kept across re-measurements of the same plate pattern."""
    from engine.jsonio import write_json
    with _lock():
        d = _load(Path(work_dir))
        ov = dict(d.get(OVERRIDES) or {})
        if on:
            ov[plate_pattern] = True
        else:
            ov.pop(plate_pattern, None)
        d[OVERRIDES] = ov
        write_json(Path(work_dir) / CACHE_NAME, d)


def forget(work_dir: Path, plate_pattern: str) -> None:
    """„Zmierz ponownie”: drop the stored results of this plate (the override stays), the next ensure() measures again."""
    from engine.jsonio import write_json
    with _lock():
        d = _load(Path(work_dir))
        d = {k: v for k, v in d.items() if k == OVERRIDES or not k.split("|")[2:3] == [plate_pattern]}
        write_json(Path(work_dir) / CACHE_NAME, d)


def run(work_dir: Path, beauty_pattern: str, plate_pattern: str) -> Dict[str, Any]:
    """Compute and store (a few seconds: 3 frame pairs at full size are read)."""
    from engine.jsonio import write_json
    key = cache_key(beauty_pattern, plate_pattern)
    r = check(beauty_pattern, plate_pattern)
    with _lock():
        d = _load(Path(work_dir))
        d = {k: v for k, v in d.items() if k == OVERRIDES or not k.split("|")[2:3] == [plate_pattern]}   # older results of this plate go
        d[key] = r
        write_json(Path(work_dir) / CACHE_NAME, d)
    return r


def ensure(work_dir: Optional[Path], beauty_pattern: str, plate_pattern: str) -> Optional[Dict[str, Any]]:
    """The stored result, or None while it is computed in a background thread (started once per key)."""
    r = cached(work_dir, beauty_pattern, plate_pattern)
    if r is not None or not work_dir:
        return r
    key = cache_key(beauty_pattern, plate_pattern)
    with _lock():
        if key in _RUNNING:
            return None
        _RUNNING.add(key)
    import threading

    def go():
        try:
            run(Path(work_dir), beauty_pattern, plate_pattern)
        except Exception:   # unreadable plate etc.: no result, stage 0 keeps saying "sprawdzam"; never breaks the page
            pass
        finally:
            with _lock():
                _RUNNING.discard(key)
    threading.Thread(target=go, daemon=True, name="plate-match").start()
    return None


def content_mismatch(work_dir: Optional[Path], beauty_pattern: str, plate_pattern: Optional[str]) -> Optional[str]:
    """Reason a backlit / clear plate is not used: the stored check says another shot (nothing is computed here)."""
    if not plate_pattern:
        return None
    r = cached(work_dir, beauty_pattern, plate_pattern)
    if r and r.get("state") == "other" and not r.get("override"):
        return f"inne ujęcie niż beauty (podobieństwo {r.get('score')})"
    return None
