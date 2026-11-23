# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„⟲ Złóż ponownie” of an Inpaint (C1) version (user decision 2026-11-13): the version's composite made again WITHOUT
the model – its stored model frames (v###/model/m_####.png, the engine's canvas size) go through the same Cleaner as a
run, as the key fill of the engine that made them, with the CURRENT composite band (grow_px, soft_px, band points and
their keys), holdout, levelling and grain. What the model drew stays the source version's: its generation margin (the
model's mask) and its model / description / anchors. A new version; model/ and anchors/ are copied into it, so it can
be recomposed again. The live preview of one frame uses the same pieces (`Cleaner.preview`)."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np

from engine import cleanup as cu

# the composite band of the cleaned thing (its own, PER_TARGET) and the global composite options – what the CURRENT
# settings give a recomposite; everything else (model, description, anchors, masks, the generation margin) is the
# source version's
BAND_FIELDS = ("grow_px", "soft_px", "band_points", "band_ref_out", "band_ref_in", "band_keys")
COMPOSITE_FIELDS = ("holdout", "holdout_skip_lost", "tone", "grain", "grain_strength")
# composite options that are the cleaned thing's own (PER_TARGET) – taken from it like the band (2026-11-15)
TARGET_FIELDS = ("holdout_off",)


def model_dir(vdir: Path) -> Path:
    return Path(vdir) / "model"


def model_frames(vdir: Path) -> List[int]:
    """The frames whose model picture the version keeps (model/m_####.png)."""
    out = []
    d = model_dir(vdir)
    for p in sorted(d.glob("m_*.png")) if d.is_dir() else []:
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def missing_frames(vdir: Path, frames: List[int]) -> List[int]:
    have = set(model_frames(vdir))
    return [f for f in frames if f not in have]


class StoredFrames(Mapping):
    """frame -> uint8 BGR model picture, read from model/ when asked (a whole clip is never held in memory)."""

    def __init__(self, vdir: Path, frames: List[int]):
        self.dir, self.frames = model_dir(vdir), [int(f) for f in frames]
        self._last: Dict[int, np.ndarray] = {}

    def __getitem__(self, f: int) -> np.ndarray:
        f = int(f)
        last = self._last.get(f)            # one reference: another thread may replace the dict meanwhile
        if last is not None:
            return last
        if f not in self.frames:
            raise KeyError(f)
        im = cv2.imread(str(self.dir / f"m_{f:04d}.png"), cv2.IMREAD_COLOR)
        if im is None:
            raise KeyError(f)
        self._last = {f: im}
        return im

    def __iter__(self) -> Iterator[int]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)


def stored_fill(vdir: Path, model: Optional[str], frames: List[int], w: int, h: int, linear: bool):
    """The key fill of the engine that made the version (its grow / grid – what it drew), reading the stored frames."""
    from engine import h3_fill
    cls = h3_fill.H3Fill
    if model == "vace":
        from engine import vace_fill
        cls = vace_fill.VaceFill
    elif model == "ltx":
        from engine import ltx_fill
        cls = ltx_fill.LtxFill
    return cls(StoredFrames(vdir, frames), w, h, linear)


def recompose_settings(cur: Dict[str, Any], vs: Dict[str, Any],
                       overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The settings of a recomposite: the source version's (`vs`: model, description, anchors, masks, the generation
    margin = what the model drew) with the CURRENT composite band of the version's cleaned thing (`cur`: its own band
    when it is the thing shown, else the one stored for it under per_target, else the version's) and the current
    composite options; `overrides` = unsaved slider values of the palette (the live preview)."""
    # a version saved before 2026-11-23 has no gen_band key: read with the old rule (its margin derivation too)
    src = cu.settings({"gen_band": False, **(vs or {})})
    cur = cu.settings(cur)
    who = (src.get("remove") or [None])[0]
    shown = (cur.get("remove") or [None])[0]
    band = cur if who == shown else (cur.get("per_target") or {}).get(who) or src
    out = dict(src)
    out.update({k: band[k] for k in BAND_FIELDS + TARGET_FIELDS if k in band})
    out.update({k: cur[k] for k in COMPOSITE_FIELDS})
    for k, v in (overrides or {}).items():
        if v is not None and k in BAND_FIELDS + COMPOSITE_FIELDS:
            out[k] = v
    if any((overrides or {}).get(k) is not None for k in ("grow_px", "soft_px")):
        out["band_keys"] = {}       # the palette's values of the shown frame win there (the live preview, one frame)
    out["gen_margin_px"] = src["gen_margin_px"]
    # the model's mask covered the source's band (2026-11-23)? A version before has no key: it did not
    out["gen_band"] = bool((vs or {}).get("gen_band", False))
    out["gen_band_src"] = ((src.get("gen_band_src") or {k: src[k] for k in BAND_FIELDS})   # a recomposite of a
                           if out["gen_band"] else None)                                    # recomposite: the model's band
    out["gen_holdout"] = bool((vs or {}).get("gen_holdout", True))   # what the model drew: a version before 2026-11-15
    # had the kept objects cut out of its mask
    return cu.settings(out)
