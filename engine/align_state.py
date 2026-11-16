# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align_state.py - which auxiliary plate (backlit / clear plate) a stage computed with and whether it was the
aligned one (align_v2 step 4, 2026-11-03).

Stages 3 (MatAnyone2: backlit edge), 5 (ViTMatte zones: the clear plate of `matte` zones) and the auto clean plate
read `<plate>_align` when stage 0 wrote it, else the raw plate. Until now nothing recorded that, so a plate aligned
AFTER a run left the run looking current. Now `run.json` (and cleanplate_auto.json) carries `aux_plates`
(`aux_plate_state`: per plate the pattern used, whether it is the aligned one and the signature of the alignment
report), and engine/stages.stage_run compares it with the project's current state (`stale_reason`): „liczone na
niewyrównanym plate'cie” / „liczone na starszym wyrównaniu” -> ↻ in the stage bar, never an automatic recompute.
Light on purpose (json + hashlib only): engine/stages.py imports it for every stage-bar refresh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

AUX_PLATES = ("backlit", "cleanplate")
LABEL = {"backlit": "backlit", "cleanplate": "clear plate"}


def align_signature(work_dir: Path, plate: str) -> Optional[str]:
    """sha1 of <work>/align/Align_<plate>.json (the report of stage 0) - None when the plate was never measured."""
    p = Path(work_dir) / "align" / f"Align_{plate}.json"
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def aux_plate_state(work_dir: Path, plates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{plate: {"pattern", "aligned", "align_sig"}} for the auxiliary plates present in `plates` (the effective
    manifest's plates): what a stage started now would read (aligned version first, engine.pipeline.from_project)."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in AUX_PLATES:
        raw, al = plates.get(name), plates.get(f"{name}_align")
        if not raw and not al:
            continue
        out[name] = {"pattern": al or raw, "aligned": bool(al), "align_sig": align_signature(work_dir, name) if al else None}
    return out


def current_plates(work_dir: Path) -> Dict[str, Any]:
    """The project's effective plates read from disk (manifest next to / above the work dir + project.json
    plates_added / plates_hidden), for callers without a ShotProject (engine/stages.py)."""
    work = Path(work_dir)
    plates: Dict[str, Any] = {}
    try:
        data = json.loads((work / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    m_path = data.get("manifest_path")
    cands = [Path(m_path)] if m_path else []
    cands += sorted(work.parent.glob("shot_manifest_*.json"))
    for mp in cands:
        try:
            plates = dict((json.loads(Path(mp).read_text(encoding="utf-8")).get("plates") or {}))
            break
        except (OSError, ValueError):
            continue
    for n in data.get("plates_hidden") or []:
        plates.pop(n, None)
    plates.update(data.get("plates_added") or {})
    return {k: v for k, v in plates.items() if v}


def current_state(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    return aux_plate_state(work_dir, current_plates(work_dir))


def stale_reason(recorded: Optional[Dict[str, Any]], now: Dict[str, Dict[str, Any]], used: Iterable[str],
                 what: str = "") -> str:
    """Why a result is behind stage 0's alignment ("" = current). `recorded` = the run's `aux_plates` (None / missing
    plate = an older run: nothing to compare, stays current), `now` = `current_state`, `used` = the plates the stage
    reads. Only alignment changes count (a plate added or replaced is Z26's business)."""
    if not recorded:
        return ""
    for plate in used:
        r, n = recorded.get(plate), now.get(plate)
        if not r or not n:
            continue
        lab = LABEL.get(plate, plate)
        if n["aligned"] and not r.get("aligned"):
            return f"liczone na niewyrównanym plate'cie {lab} – etap 0 zapisał wersję wyrównaną (_align){what}"
        if n["aligned"] and r.get("aligned") and n.get("align_sig") != r.get("align_sig"):
            return f"liczone na starszym wyrównaniu plate'u {lab} – etap 0 zmierzył go ponownie{what}"
    return ""
