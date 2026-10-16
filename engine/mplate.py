# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/mplate.py - Plate-to-mask (SPEC 4b): input of SAM3 / MatAnyone / LTX and of edge measurements only –
never RGB, never exported.

1. display (PlateToDisplay for scene-linear) x the 4c gain maps (same deflicker as the comp) – or, when the user
   exported `_deflicker_dn`, crops straight from that file;
2. temporal denoise with motion compensation: neighbours +-2 aligned with DIS flow, median weighted by agreement
   (pixels with a large alignment error are not averaged -> no ghosts); then a light spatial bilateral whose strength
   comes from the noise measured on flat areas;
3. gate: static edges must not move (edge position vs the display image < 0.1 px) and noise down >= 40 %;
   otherwise weaker settings; still failing -> display without denoise (logged);
4. computed only for crop segments, on demand, cached in `<work>/matte_desk/mplate/<seg>/` (16-bit PNG).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

STRENGTHS = [  # (agreement sigma, bilateral sigma multiplier)
    (0.06, 1.0),
    (0.04, 0.6),
    (0.025, 0.3),
]


def noise_level(img: np.ndarray) -> float:
    """Robust noise sigma from the Laplacian of the luma on flat areas."""
    g = img.mean(axis=2) if img.ndim == 3 else img
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    gm = cv2.GaussianBlur(np.abs(cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 1)), (0, 0), 2)
    flat = gm < np.percentile(gm, 40)
    v = lap[flat] if flat.any() else lap.ravel()
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(20.0))


def _u8(g: np.ndarray) -> np.ndarray:
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def denoise_frame(stack: Sequence[np.ndarray], center: int, sigma_agree: float, bil_mult: float, noise: float
                  ) -> np.ndarray:
    """Weighted median-ish average of flow-aligned neighbours (+-2) around stack[center]."""
    ref = stack[center].astype(np.float32)
    h, w = ref.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g_ref = _u8(ref.mean(axis=2))
    samples, weights = [ref], [np.ones((h, w), np.float32)]
    for j, img in enumerate(stack):
        if j == center:
            continue
        g = _u8(img.mean(axis=2))
        flow = dis.calc(g_ref, g, None)
        warped = cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        err = np.abs(warped - ref).mean(axis=2)
        wgt = np.exp(-(err / max(sigma_agree, 1e-4)) ** 2)
        samples.append(warped)
        weights.append(wgt.astype(np.float32))
    S = np.stack(samples)
    W = np.stack(weights)[..., None]
    out = (S * W).sum(0) / W.sum(0)
    if bil_mult > 0 and noise > 0:
        sc = float(min(0.2, 3.0 * noise * bil_mult))
        out = cv2.bilateralFilter(out.astype(np.float32), 5, sc, 2.0)
    return out.astype(np.float32)


def static_edge_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Median sub-pixel displacement of textured patches between two images (noise-robust: both lumas blurred
    sigma 1.5, phase correlation + Gauss-Newton refinement) – the 4b gate "static edges must not move"."""
    from engine.align import LocalPatchAligner, phase_correlate
    la = cv2.GaussianBlur(a.mean(axis=2).astype(np.float32), (0, 0), 1.5)
    lb = cv2.GaussianBlur(b.mean(axis=2).astype(np.float32), (0, 0), 1.5)
    h, w = la.shape
    win = cv2.createHanningWindow((32, 32), cv2.CV_32F)
    shifts = []
    for y in range(32, h - 32, 40):
        for x in range(32, w - 32, 40):
            pa, pb = la[y - 16:y + 16, x - 16:x + 16], lb[y - 16:y + 16, x - 16:x + 16]
            if pa.std() < 0.02:
                continue
            (dx, dy), r = phase_correlate(pa, pb, win)
            dx, dy = LocalPatchAligner._refine_translation(lb, pa, x, y, dx, dy, win)
            shifts.append(np.hypot(dx, dy))
    return float(np.median(shifts)) if shifts else 0.0


@dataclass
class MPlateReport:
    seg_id: str
    strength: int = -1                  # index in STRENGTHS, -1 = display without denoise
    noise_before: float = 0.0
    noise_after: float = 0.0
    edge_shift_px: float = 0.0
    log: List[str] = field(default_factory=list)


class MaskPlate:
    """Plate-to-mask crops for one crop segment. `display_crop(f)` -> display crop (already deflickered x gains)."""

    def __init__(self, cache_dir: Path, seg_id: str, frames: Sequence[int], display_crop: Callable[[int], np.ndarray],
                 static_mask: Optional[Callable[[int], np.ndarray]] = None):
        self.dir = Path(cache_dir) / seg_id
        self.seg_id = seg_id
        self.frames = list(frames)
        self.src = display_crop
        self.static_mask = static_mask
        self.report: Optional[MPlateReport] = None

    def _cached(self, f: int) -> Optional[np.ndarray]:
        p = self.dir / f"m_{f:04d}.png"
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        return None

    def build(self, probe: Optional[Sequence[int]] = None) -> MPlateReport:
        """Pick the strongest setting that passes the gate on a few probe frames, then fill the cache."""
        if (self.dir / "mplate.json").exists():
            self.report = MPlateReport(**json.loads((self.dir / "mplate.json").read_text()))
            return self.report
        self.dir.mkdir(parents=True, exist_ok=True)
        rep = MPlateReport(self.seg_id)
        probe = list(probe or self.frames[len(self.frames) // 2: len(self.frames) // 2 + 1])
        cache: Dict[int, np.ndarray] = {}
        get = lambda f: cache.setdefault(f, self.src(f))
        chosen = -1
        for si, (sa, bm) in enumerate(STRENGTHS):
            ok = True
            for f in probe:
                i = self.frames.index(f)
                nb = self.frames[max(0, i - 2): i + 3]
                stack = [get(x) for x in nb]
                out = denoise_frame(stack, nb.index(f), sa, bm, noise_level(stack[nb.index(f)]))
                n0, n1 = noise_level(get(f)), noise_level(out)
                shift = static_edge_shift(get(f), out)
                rep.log.append(f"ustawienie {si}: szum {n0:.4f} -> {n1:.4f}, krawędzie {shift:.3f} px")
                if not (shift < 0.1 and n1 <= 0.6 * n0):
                    ok = False
                    break
                rep.noise_before, rep.noise_after, rep.edge_shift_px = n0, n1, shift
            if ok:
                chosen = si
                break
        rep.strength = chosen
        if chosen < 0:
            rep.log.append("bramka nie przeszła – plate do maskowania = display bez odszumienia")
        for i, f in enumerate(self.frames):
            if chosen < 0:
                out = get(f)
            else:
                nb = self.frames[max(0, i - 2): i + 3]
                out = denoise_frame([get(x) for x in nb], nb.index(f), *STRENGTHS[chosen], noise_level(get(f)))
            cv2.imwrite(str(self.dir / f"m_{f:04d}.png"), (np.clip(out, 0, 1) * 65535).astype(np.uint16))
            for k in list(cache):
                if k < f - 2:
                    cache.pop(k)
        (self.dir / "mplate.json").write_text(json.dumps(asdict(rep), indent=1, ensure_ascii=False))
        self.report = rep
        return rep

    def crop(self, f: int) -> np.ndarray:
        m = self._cached(f)
        if m is None:
            self.build()
            m = self._cached(f)
        return m


# ---------------------------------------------------------------------------------------------------------------------
# Project switch (user decision 2026-10-13): project.json -> mask_plate_denoise = auto (default, key absent) | on | off.
# One setting per project: the core (SAM3) and the final pass (MatAnyone) share the cached plate-to-mask, so both must
# use the same value. auto = ON for EXR / > 8-bit sources, OFF for 8-bit sources (same detection as the Comfy I/O).
# defaults.json mplate.enabled = false stays a global kill-switch (effective value off regardless).
DENOISE_CHOICES = ("auto", "on", "off")
DENOISE_KEY = "mask_plate_denoise"


def resolve_mask_plate_denoise(setting: Optional[str], manifest: Optional[Dict], global_enabled: bool = True) -> Dict:
    """-> {setting, effective (bool), auto_resolves_to ('on'/'off'), reason, global_enabled, choices}."""
    from engine.comfy_io import PNG8, io_mode_for
    s = str(setting or "auto").lower()
    if s not in DENOISE_CHOICES:
        s = "auto"
    eight = io_mode_for(manifest) == PNG8
    auto_on = not eight
    src = "źródło 8-bit (MP4 / PNG / JPG)" if eight else "źródło EXR / > 8 bit"
    if not global_enabled:
        eff, reason = False, "wyłączone globalnie (defaults.json → mplate.enabled = false)"
    elif s == "auto":
        eff, reason = auto_on, src
    else:
        eff, reason = s == "on", "ustawione ręcznie"
    return {"setting": s, "effective": bool(eff), "auto_resolves_to": "on" if auto_on else "off", "reason": reason,
            "source_8bit": eight, "global_enabled": bool(global_enabled), "choices": list(DENOISE_CHOICES)}
