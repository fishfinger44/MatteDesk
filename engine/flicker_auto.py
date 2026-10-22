# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/flicker_auto.py – deflicker settings from a flicker measurement (docs/PLAN_DEFLICKER_AUTO.md § 3).

`auto_settings(measure)` turns a `FlickerMeasure` (engine/flicker_measure.py) into `DeflickerSettings`: the slider values of
the Deflicker palette plus Polish notes that say WHY (shown in the UI next to the sliders). The user decided (2026-10-21) that
every plate starts with a measurement and the sliders set themselves from it; the thresholds below are the initial rules from
SH010 X1 and are meant to be tuned on further plates – they are module constants on purpose.

Rules (plan § 3):
- strength (stage-1 iterations) from the strongest region's level flicker: < 0.3 % → 2, < 1 % → 3, < 3 % → 5, < 5 % → 6, else 8;
- time window (stage-1 ± frames): random pulsing (autocorrelation < 0.2) → ±10; variation with memory → 2.5 × correlation
  length, clamped to 10..20; a linear drift is reported as kept (trend), it does not change the window;
- 9×6 grid: on when the tiles do not flicker together (region correlation < 0.5) or unknown;
- whole-frame correction per channel when the colour (R/G, B/G) steps exceed 0.3 %, else luma (informational until the
  stage-4 knob exists);
- texture / shadow flicker (stage 3, EXR export only): on when a region's 8–40 px band exceeds its flow floor by > 0.5 % AND
  the pattern is fixed (PCA share > 0.5); radius 4 for random pulsing, 3 / 2 with memory;
- denoise: never automatic – a hint when the < 2 px band is clearly above its floor;
- early stop after stage 1 (target_reduction) is disabled: the measurement decides what runs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine.flicker_measure import FlickerMeasure, RegionMeasure

# ---- thresholds (initial, from SH010 X1 2026-10-21 – tune on further plates)
# level_pct upper bound -> stage-1 iterations; capped at 3: more iterations re-imprint moving objects into the gain map
# (E1c ghost 15 / 3 0.0046, 15 / 5 0.0066 > 0.005, 2026-10-22) – stronger values only by hand on the slider
LEVEL_STEPS: Tuple[Tuple[float, int], ...] = ((0.3, 2), (math.inf, 3))
MAX_AUTO_ITERATIONS = 3
RANDOM_AUTOCORR = 0.2        # below: pulsing without memory
WINDOW_RANDOM = 10
WINDOW_MIN, WINDOW_MAX = 10, 20
WINDOW_CORR_FACTOR = 2.5     # window = factor x correlation length (frames) for variation with memory
GRID_CORR = 0.5              # tiles' level correlation below this: they flicker apart -> 9x6 grid on
COLOUR_PER_CHANNEL = 0.3     # % – colour steps above this: correct R / G / B separately
TEXTURE_MIN_ABOVE_FLOOR = 0.5   # % – 8-40 px band above its floor
TEXTURE_MIN_SHARE = 0.5      # fixed-pattern share (PCA modes 1-3)
TEXTURE_RADIUS_RANDOM, TEXTURE_RADIUS_MEMORY, TEXTURE_RADIUS_SLOW = 4, 3, 2
DENOISE_HINT_FACTOR, DENOISE_HINT_MIN = 2.0, 0.3   # < 2 px band > factor x its floor + min
DRIFT_SIGMAS, DRIFT_MIN_PCT = 2.0, 2.0
MOTION_WARN_PX = 1.0         # px at eval width (1024): an object moving on its own
CAMERA_GRID_OFF_PX = 2.0     # camera motion above this: the 9x6 grid hurts (SH010 F1-16: inner 2.29 -> 2.42 %, objects worse) -> off
SMOOTHNESS_DEFAULT_PX = 4    # stage-1 sigma at 512 px (2026-10-22 grid: 4 beat 6 on ghost and on the SH010 tree)
PERIOD_P, PERIOD_SHARE = 0.01, 0.3   # a regular rhythm (lamp beat): p vs shuffled < 0.01 and >= 30 % of the level variance
PERIOD_MAX = 8.0             # longer "periods" in a ±10-frame window are drift, not a lamp beat


@dataclass
class DeflickerSettings:
    strength: int                      # stage1_iterations
    window: int                        # stage1_window (± frames)
    grid: bool                         # enable_stage2 (9x6 tiles)
    whole_frame: bool = True           # enable_stage4
    per_channel: bool = True           # stage 4 per channel (False = one luma gain for R / G / B)
    texture: bool = False              # enable_stage3 recommended (EXR export only)
    texture_radius: int = TEXTURE_RADIUS_RANDOM
    smoothness_px: int = SMOOTHNESS_DEFAULT_PX   # stage-1 sigma at 512 px
    denoise_hint: bool = False
    driver_region: str = "all"         # the region whose level flicker set the strength
    texture_region: Optional[str] = None
    achievable: Dict[str, Dict[str, float]] = field(default_factory=dict)   # region -> {now, floor}
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_process_kwargs(self, exr: bool) -> Dict[str, Any]:
        """Arguments for PlateCleaner.process_sequence. Stage 3 only with an EXR export (the Nuke Group carries gains only)."""
        return {"enable_stage1": True, "enable_stage2": self.grid, "enable_stage4": self.whole_frame,
                "enable_stage3": bool(self.texture and exr), "enable_denoise": False,
                "stage1_window": int(self.window), "stage1_iterations": int(self.strength),
                "stage3_radius": int(self.texture_radius), "stage1_sigma": float(self.smoothness_px),
                "stage4_luma": not self.per_channel, "target_reduction": 2.0}


def _pct(x: Optional[float], nd: int = 2) -> str:
    return "?" if x is None else f"{x:.{nd}f}"


def strength_for(level_pct: float) -> int:
    for bound, it in LEVEL_STEPS:
        if level_pct < bound:
            return it
    return LEVEL_STEPS[-1][1]


def window_for(autocorr: Optional[float]) -> Tuple[int, str]:
    """(± frames, kind) – kind: 'random' | 'memory' | 'unknown'."""
    if autocorr is None:
        return WINDOW_RANDOM, "unknown"
    if autocorr < RANDOM_AUTOCORR:
        return WINDOW_RANDOM, "random"
    rho = min(float(autocorr), 0.95)
    tau = -1.0 / math.log(rho)                      # AR(1) correlation length in frames
    return int(max(WINDOW_MIN, min(WINDOW_MAX, round(WINDOW_CORR_FACTOR * tau)))), "memory"


def texture_radius_for(autocorr: Optional[float]) -> int:
    if autocorr is None or autocorr < RANDOM_AUTOCORR:
        return TEXTURE_RADIUS_RANDOM
    return TEXTURE_RADIUS_SLOW if autocorr > 0.7 else TEXTURE_RADIUS_MEMORY


def _texture_excess(r: RegionMeasure) -> float:
    return r.texture_bands_pct.get("8-40", 0.0) - r.texture_floor_bands_pct.get("8-40", 0.0)


def auto_settings(m: FlickerMeasure) -> DeflickerSettings:
    regions = m.regions
    if not regions:
        raise ValueError("pomiar bez regionów")
    objects = {k: v for k, v in regions.items() if k not in ("all", "inner")} or dict(regions)

    # ---- strength: the strongest level flicker among the regions (objects first, "all" otherwise)
    driver_name, driver = max(objects.items(), key=lambda kv: kv[1].level_pct)
    if "all" in regions and regions["all"].level_pct > driver.level_pct:
        driver_name, driver = "all", regions["all"]
    strength = strength_for(driver.level_pct)

    # ---- window from the driver's memory; a regular rhythm -> a whole number of periods (averaging cancels it exactly)
    window, kind = window_for(driver.level_autocorr)
    rhythm = None
    for nm, r in regions.items():
        if (r.period_p is not None and r.period_p < PERIOD_P and (r.period_share or 0) >= PERIOD_SHARE
                and r.period_frames is not None and 1.9 <= r.period_frames <= PERIOD_MAX):
            if rhythm is None or (r.period_share or 0) > rhythm[1].period_share:
                rhythm = (nm, r)
    if rhythm is not None:
        per = rhythm[1].period_frames
        k = max(1, int(np.ceil(WINDOW_MIN / per)))           # smallest whole number of periods >= the minimum window
        window, kind = int(min(WINDOW_MAX, round(k * per))), "rhythm"

    # ---- grid (off under camera motion: the tiles chase content moving through them)
    camera_moves = m.global_shift_px > CAMERA_GRID_OFF_PX
    grid = (m.region_corr is None or m.region_corr < GRID_CORR) and not camera_moves

    # ---- colour
    col = max((max(r.colour_pct.get("rg", 0.0), r.colour_pct.get("bg", 0.0)) for r in regions.values()), default=0.0)
    per_channel = col > COLOUR_PER_CHANNEL

    s = DeflickerSettings(strength=strength, window=window, grid=grid, per_channel=per_channel, driver_region=driver_name)
    s.notes.append(f"Migotanie poziomu {driver.level_pct:.2f} % ({driver_name}) → siła {strength}")
    if kind == "rhythm":
        nm, r = rhythm
        s.notes.append(f"Regularny rytm migania co {r.period_frames:.1f} kl. w „{nm}” (lampa? udział {r.period_share:.2f}, p {r.period_p:.3f}) "
                       f"→ okno czasowe ±{window} kl. = pełne okresy, uśrednianie go wygasza")
    elif kind == "random":
        s.notes.append(f"Pulsowanie losowe klatka→klatka (autokorelacja {_pct(driver.level_autocorr)}) → okno czasowe ±{window} kl.")
    elif kind == "memory":
        s.notes.append(f"Zmiany jasności z pamięcią (autokorelacja {_pct(driver.level_autocorr)}) → okno czasowe ±{window} kl.")
    else:
        s.notes.append(f"Za mało klatek na ocenę rytmu → okno czasowe ±{window} kl.")
    if camera_moves:
        s.notes.append(f"Kamera porusza się ({m.global_shift_px:.1f} px/kl.) → kratka 9×6 wyłączona (przy ruchu kamery kafle psują wynik)")
    elif m.region_corr is None:
        s.notes.append("Brak oceny, czy obszary migają razem → kratka 9×6 włączona (bezpiecznie)")
    elif grid:
        s.notes.append(f"Obszary kadru migają osobno (korelacja kafli {m.region_corr:.2f}) → kratka 9×6 włączona")
    else:
        s.notes.append(f"Cały kadr miga razem (korelacja kafli {m.region_corr:.2f}) → kratka 9×6 wyłączona")
    if per_channel:
        s.notes.append(f"Kolor światła pływa (R/G, B/G do {col:.2f} %) → korekta R / G / B osobno")
    else:
        s.notes.append(f"Kolor stabilny (R/G, B/G ≤ {col:.2f} %) → korekta jasności (luma)")

    # ---- texture / shadow flicker: the region with the largest excess over the floor that has a fixed pattern
    cand = [(nm, r, _texture_excess(r)) for nm, r in regions.items()
            if r.texture_fixed_share is not None and r.texture_fixed_share > TEXTURE_MIN_SHARE and _texture_excess(r) > TEXTURE_MIN_ABOVE_FLOOR]
    if cand:
        nm, r, exc = max(cand, key=lambda c: c[2])
        s.texture, s.texture_region = True, nm
        s.texture_radius = texture_radius_for(r.level_autocorr)
        s.notes.append(f"Migotanie faktury i cieni: stały wzór w „{nm}” ({exc:.2f} % ponad podłogę, udział {r.texture_fixed_share:.2f}) "
                       f"→ włącz przy zapisie EXR, okno ±{s.texture_radius} kl.")
    else:
        s.notes.append("Migotanie faktury i cieni: brak stałego wzoru ponad podłogą → wyłączone")

    # ---- denoise hint
    for nm, r in regions.items():
        lo, fl = r.texture_bands_pct.get("<2", 0.0), r.texture_floor_bands_pct.get("<2", 0.0)
        if lo > DENOISE_HINT_FACTOR * fl + DENOISE_HINT_MIN:
            s.denoise_hint = True
            s.notes.append(f"Ziarno / szum ponad podłogę w „{nm}” ({lo:.2f} % vs {fl:.2f} %) – rozważ odszumianie (tylko EXR, decyzja artystyczna)")
            break

    # ---- drift, motion, achievable
    for nm, r in regions.items():
        if abs(r.drift_pct) >= DRIFT_MIN_PCT and abs(r.drift_pct) > DRIFT_SIGMAS * max(r.drift_sigma_pct, 1e-6):
            s.notes.append(f"Powolna zmiana jasności {r.drift_pct:+.1f} % na {m.n_frames} kl. („{nm}”) – zachowana jako trend, nie usuwana")
        if r.motion_resid_px is not None and r.motion_resid_px > MOTION_WARN_PX and nm not in ("all", "inner"):
            s.warnings.append(f"„{nm}” rusza się niezależnie od kamery ({r.motion_resid_px:.1f} px) – deflicker po masce obiektu w przyszłości")
        s.achievable[nm] = {"now": r.metric_pct, "floor": r.floor_pct}
    if camera_moves:
        s.warnings.append(f"Kamera porusza się ({m.global_shift_px:.1f} px/kl.) – brzeg kadru może pulsować po deflickerze (nowa treść wchodzi "
                          "w kadr); oceń na podglądzie region „bez brzegu”, a w razie potrzeby skróć zakres do części bez ruchu kamery")
    return s
