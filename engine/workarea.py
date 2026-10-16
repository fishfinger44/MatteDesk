# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/workarea.py - Work area (frame range of the shot, like the AE / Nuke work area; user request 2026-10-15).

The work area is stored per shot in `<work>/matte_desk/project.json` -> `work_area: [first, last]` (absent = the whole
shot; the shot manifest is never written). „▶▶ Przelicz całość”, „Przelicz finalnie” and the export (scope
"work_area") compute only its frames.

Coverage = which frames a result was computed on, with the current inputs:
- core: `core/core.json` -> `coverage` (list of [a, b]); a core from before the work area has no key -> the whole shot
  (it was always computed on the whole shot; when merged with a new run: the frames that have a `core/c_####.png`).
- final: `runs/<run>/run.json` -> `frames` ([a, b]); a run from before the work area has no key -> its segments.
A result that does not cover the frames an action needs is never used silently: the backend answers 409 with the list
of missing frame intervals (`missing_detail`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Interval = List[int]


def intervals_from_frames(frames: Iterable[int]) -> List[Interval]:
    """Sorted frames -> contiguous [a, b] intervals."""
    out: List[Interval] = []
    for f in sorted(set(int(x) for x in frames)):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def union(*groups: Optional[Sequence[Sequence[int]]]) -> List[Interval]:
    ivs = sorted([int(a), int(b)] for g in groups if g for a, b in g if int(a) <= int(b))
    out: List[Interval] = []
    for a, b in ivs:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def missing(rng: Sequence[int], coverage: Optional[Sequence[Sequence[int]]]) -> List[Interval]:
    """Intervals of `rng` ([a, b]) not inside `coverage`."""
    lo, hi = int(rng[0]), int(rng[1])
    out: List[Interval] = []
    cur = lo
    for a, b in union(coverage):
        if b < cur:
            continue
        if a > hi:
            break
        if a > cur:
            out.append([cur, min(hi, a - 1)])
        cur = max(cur, b + 1)
        if cur > hi:
            break
    if cur <= hi:
        out.append([cur, hi])
    return out


def fmt_intervals(ivs: Sequence[Sequence[int]], limit: int = 8) -> str:
    s = [f"F{a}" if a == b else f"F{a}–{b}" for a, b in ivs[:limit]]
    if len(ivs) > limit:
        s.append(f"… (+{len(ivs) - limit})")
    return ", ".join(s)


def clamp_range(rng: Optional[Sequence[int]], shot: Sequence[int]) -> Optional[Interval]:
    """[a, b] inside the shot (a <= b); None / the whole shot -> None (= no work area)."""
    if not rng:
        return None
    a, b = sorted((int(rng[0]), int(rng[1])))
    a, b = max(int(shot[0]), a), min(int(shot[1]), b)
    if a > b:
        raise ValueError(f"zakres F{rng[0]}–{rng[1]} poza ujęciem F{shot[0]}–{shot[1]}")
    if [a, b] == [int(shot[0]), int(shot[1])]:
        return None
    return [a, b]


def expand_to_keys(rng: Sequence[int], keys: Iterable[int], shot: Sequence[int]) -> Interval:
    """Core frames for a work area: extended outwards to the nearest key on each side (when there is one), so every
    interval between two keys is propagated whole – the same intervals (and cache hashes) as a run on the whole shot."""
    lo, hi = int(rng[0]), int(rng[1])
    ks = sorted(int(k) for k in keys if int(shot[0]) <= int(k) <= int(shot[1]))
    before = [k for k in ks if k <= lo]
    after = [k for k in ks if k >= hi]
    if before:
        lo = before[-1]
    if after:
        hi = after[0]
    return [lo, hi]


_CORE_RE = re.compile(r"c_(\d+)\.png$")


WHOLE = [[-10 ** 9, 10 ** 9]]      # a core from before the work area: computed on the whole shot


def core_coverage(obj_dir: Path, legacy_whole: bool = True) -> Optional[List[Interval]]:
    """Frames the object's core covers (None = no core). A core.json without `coverage` (before the work area, always
    the whole shot): `legacy_whole` -> WHOLE, else the frames that have a c_####.png."""
    cj = Path(obj_dir) / "core" / "core.json"
    if not cj.exists():
        return None
    try:
        cov = json.loads(cj.read_text(encoding="utf-8")).get("coverage")
    except (OSError, ValueError):
        cov = None
    if cov is not None:
        return union(cov)
    if legacy_whole:
        return [list(x) for x in WHOLE]
    return intervals_from_frames(int(m.group(1)) for p in (Path(obj_dir) / "core").glob("c_*.png")
                                 if (m := _CORE_RE.search(p.name)))


def run_coverage(obj_dir: Path, run: Optional[str]) -> Optional[List[Interval]]:
    """Frames a final run (runs/<run>) covers (None = no run)."""
    if not run:
        return None
    rj = Path(obj_dir) / "runs" / run / "run.json"
    try:
        d = json.loads(rj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("coverage") is not None:      # a version with a life range: only its visible pieces were computed
        return union(d["coverage"])
    if d.get("frames"):
        return union([d["frames"]])
    return union([[s["first"], s["last"]] for s in d.get("segments_def", [])])


def run_frames(obj_dir: Path, run: Optional[str]) -> Optional[List[int]]:
    """Frames a run computed (gaps outside the object's life), None = unknown."""
    cov = run_coverage(obj_dir, run)
    if not cov:
        return None
    return [f for a, b in cov for f in range(a, b + 1)]


def missing_detail(what: str, scope_label: str, per_object: Dict[str, List[Interval]]) -> Dict[str, Any]:
    """409 body: {code, detail (one readable sentence per object), missing: {obj: [[a, b]]}}."""
    lines = [f"„{n}”: brak {fmt_intervals(iv)}" for n, iv in per_object.items() if iv]
    return {"code": "range_missing", "what": what, "scope": scope_label,
            "detail": f"{what} nie jest policzony na {scope_label} – " + "; ".join(lines),
            "missing": {n: iv for n, iv in per_object.items() if iv}}


# ---------------------------------------------------------------------------------------------------------------------
# Object life range (zakres życia obiektu, user request 2026-10-15): an object can leave the frame and disappear (or come
# back). `object.json -> life: [[a, b], ...]` = the intervals where the object is visible (absent = the whole shot; the
# shot manifest is never written). Outside them the matte is empty (alpha 0) and nothing is computed: SAM3 propagates
# each visible piece separately (no tracker jump onto other objects across the gap), MatAnyone runs on contiguous
# visible pieces, the export writes black frames. Coverage: frames outside the life count as computed.
# Together with the work area the intersection counts; the „≥ 1 key” rule applies to every visible piece of it.

def intersect(a: Optional[Sequence[Sequence[int]]], b: Optional[Sequence[Sequence[int]]]) -> List[Interval]:
    out: List[Interval] = []
    for x0, x1 in union(a):
        for y0, y1 in union(b):
            lo, hi = max(x0, y0), min(x1, y1)
            if lo <= hi:
                out.append([lo, hi])
    return union(out)


def normalize_life(life: Optional[Sequence[Sequence[int]]], shot: Sequence[int]) -> Optional[List[Interval]]:
    """Life intervals clipped to the shot, merged; None = the whole shot (= no life range). ValueError when no frame of
    the shot stays visible (an object visible nowhere is deleted, not hidden)."""
    if life is None:
        return None
    ivs = []
    for iv in life:
        if len(iv) != 2:
            raise ValueError("przedział życia = [pierwsza, ostatnia] klatka")
        a, b = sorted((int(iv[0]), int(iv[1])))
        ivs.append([a, b])
    out = intersect(ivs, [[int(shot[0]), int(shot[1])]])
    if not out:
        raise ValueError(f"zakres życia poza ujęciem F{shot[0]}–{shot[1]} – obiekt musi być widoczny w co najmniej jednej klatce")
    if out == [[int(shot[0]), int(shot[1])]]:
        return None
    return out


def object_life(obj: Dict[str, Any]) -> Optional[List[Interval]]:
    """The object's life intervals (None = the whole shot)."""
    life = obj.get("life") if obj else None
    return union(life) if life else None


def is_visible(f: int, life: Optional[Sequence[Sequence[int]]]) -> bool:
    return life is None or any(int(a) <= int(f) <= int(b) for a, b in life)


def visible_frames(frames: Iterable[int], life: Optional[Sequence[Sequence[int]]]) -> List[int]:
    return [int(f) for f in frames if is_visible(f, life)]


def life_pieces(rng: Sequence[int], life: Optional[Sequence[Sequence[int]]]) -> List[Interval]:
    """Visible pieces of the range [a, b] (the whole range without a life range)."""
    r = [[int(rng[0]), int(rng[1])]]
    return r if life is None else intersect(r, life)


def need_missing(rng: Sequence[int], coverage: Optional[Sequence[Sequence[int]]],
                 life: Optional[Sequence[Sequence[int]]]) -> List[Interval]:
    """Frames of `rng` a result must have but does not: visible frames (life) outside `coverage`; frames outside the
    life count as computed (empty matte)."""
    out: List[Interval] = []
    for a, b in life_pieces(rng, life):
        out += missing((a, b), coverage)
    return union(out)


def life_changed_intervals(old: Optional[Sequence[Sequence[int]]], new: Optional[Sequence[Sequence[int]]],
                           shot: Sequence[int]) -> List[Interval]:
    """Frames whose visibility differs between two life ranges (None = the whole shot)."""
    whole = [[int(shot[0]), int(shot[1])]]
    o = union(old) if old else whole
    n = union(new) if new else whole
    return union(intersect(missing(whole[0], o), n), intersect(missing(whole[0], n), o))


# hint: the object leaves / enters the frame (core gets empty after touching the frame border)
LIFE_EMPTY_FRAC = 2e-5          # core area / frame area below this = empty
LIFE_BORDER_PX = 3              # core within this many px of the frame border = at the border


def life_hints(areas: Dict[int, float], border: Dict[int, bool], life: Optional[Sequence[Sequence[int]]] = None,
               min_gap: int = 2) -> List[Dict[str, Any]]:
    """`areas` = {frame: core area fraction}, `border` = {frame: core touches the frame border}. A hint where the core
    becomes empty right after a frame touching the border (`end`: object leaves from F) or appears at the border after
    empty frames (`start`). Runs of empty frames shorter than `min_gap` are ignored (SAM3 drop-out, not an exit)."""
    fs = sorted(areas)
    empty = {f: areas[f] < LIFE_EMPTY_FRAC for f in fs}
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(fs):
        if not empty[fs[i]]:
            i += 1
            continue
        j = i
        while j + 1 < len(fs) and empty[fs[j + 1]] and fs[j + 1] == fs[j] + 1:
            j += 1
        a, b = fs[i], fs[j]
        if b - a + 1 >= min_gap:
            prev = fs[i - 1] if i > 0 and fs[i - 1] == a - 1 else None
            nxt = fs[j + 1] if j + 1 < len(fs) and fs[j + 1] == b + 1 else None
            if prev is not None and border.get(prev):
                out.append({"kind": "end", "frame": a, "last_visible": prev, "empty": [a, b]})
            if nxt is not None and border.get(nxt):
                out.append({"kind": "start", "frame": nxt, "first_visible": nxt, "empty": [a, b]})
        i = j + 1
    # only hints that would change the life range (frames still inside it)
    return [h for h in out if is_visible(h["empty"][0], life) or is_visible(h["empty"][1], life)]


def mask_stats(m: Any) -> Tuple[float, bool]:
    """(area fraction, touches the frame border) of a binary mask (numpy)."""
    import numpy as np
    m = np.asarray(m) > 0.5
    if not m.any():
        return 0.0, False
    b = LIFE_BORDER_PX
    touch = bool(m[:b].any() or m[-b:].any() or m[:, :b].any() or m[:, -b:].any())
    return float(m.mean()), touch
