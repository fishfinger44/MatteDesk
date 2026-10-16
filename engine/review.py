# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/review.py - Import strokes from the user's "Przegląd klatek" review Artifact (SPEC 9).

Input: the set's entry from manifest.json (frames, first, pw, ph, crops[x0, y0, w, h] per frame, plate px) and the
`marks` documents (ArtifactData query): {set, frame, status, note, strokes: [{c: 'r'|'g'|'y', p: [x*10, y*10, ...]}]}
in panel pixels x10. Panel px -> plate: x = x0 + (p / 10) / pw * w,  y = y0 + (p / 10) / ph * h.
r = remove, g = add (strokes for a key: core +/- strokes), y = note only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def strokes_from_marks(set_entry: Dict[str, Any], marks: Sequence[Dict[str, Any]], radius_px: float = 6.0
                       ) -> Dict[int, List[Dict[str, Any]]]:
    pw, ph = float(set_entry["pw"]), float(set_entry["ph"])
    first = int(set_entry.get("first", 1))
    crops = set_entry.get("crops") or []
    out: Dict[int, List[Dict[str, Any]]] = {}
    for m in marks:
        if m.get("set") not in (None, set_entry.get("id")):
            continue
        f = int(m["frame"])
        idx = f - first
        x0, y0, cw, chh = crops[idx] if 0 <= idx < len(crops) else (crops[0] if crops else (0, 0, pw, ph))
        for s in m.get("strokes", []):
            mode = {"r": "remove", "g": "add"}.get(s.get("c"))
            if not mode:
                continue
            p = s.get("p", [])
            pts = [[x0 + (p[i] / 10.0) / pw * cw, y0 + (p[i + 1] / 10.0) / ph * chh] for i in range(0, len(p) - 1, 2)]
            if pts:
                out.setdefault(f, []).append({"mode": mode, "radius_px": radius_px, "p": pts})
    return out
