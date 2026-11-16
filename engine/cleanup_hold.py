# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removed objects that lose their mask for a few frames keep the nearest mask (Cleanup, 2026-11-12).

When SAM 3.1 loses the removed object for a few frames its alpha is empty there, the removal mask is empty and the
object pops back into the cleaned plate. The removal loaders (api/cleanup.py `_alpha_loaders`) are wrapped here, so
every consumer (the Cleaner, the mask preview, the MiniMax-H3 / Wan 2.2 Fun VACE mask, Cleanup 3D) sees the same held
mask. The idea (not the code) is the „Follow Matte” of AInVFX Paint a plate: one matte per frame, the last good one
where the model returns none. The kept objects have their own check (engine/cleanup.py `check_kept`).

The rule, per removed object over the work range (`find_lost`):
  1. present = alpha area (> 0.5) of at least cu.LOST_MIN_PX; a present frame under HOLD_TINY of the median area of its
     present neighbours within +-HOLD_WINDOW frames is absent too (a few stray pixels are not the object);
  2. a solid stretch = at least cu.LOST_MIN_FRAMES present frames in a row (one stray detection is no evidence);
  3. a gap between solid stretches (or between one and an end of the range) of at most HOLD_MAX_GAP frames is a LOST
     object – unless it is a real exit: the solid frame next to the gap touches the frame border AND its area is under
     HOLD_EXIT_SHRINK of the largest area of the HOLD_WINDOW frames further from the gap (shrinking into the border).
     A gap with a solid stretch on both sides is an exit only when both sides say so (out and back in);
  4. a lost frame takes the alpha of the nearest solid frame (a tie: the earlier one), as it is – no warp, no tracking;
     frames outside the object's life are never held; every other frame is returned as the loader gave it (bit for bit).
The areas of the whole range are read once per object (the loader itself) and cached by what the mask is made of.
The scan is made only by `Hold.plan` / `held_report` – every run calls it up front (api/cleanup.py `_held`); a loader
call never scans, so an interactive preview before the first run shows the loader's own mask (no wait), and the held
one once a run (or anything else) has scanned that mask.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine import cleanup as cu
from engine.lru import LockedLRU

logger = logging.getLogger(__name__)

HOLD_TINY = 0.10          # a frame under this x the median area of its neighbours = the object is not there
HOLD_WINDOW = 5           # +- frames of that neighbourhood, and of the "was it shrinking" look-back
HOLD_MAX_GAP = 8          # frames: a longer hole is not a lost track but an object that is really gone (never held)
HOLD_EXIT_SHRINK = 0.70   # touching the border with under this x its recent largest area = leaving / entering the frame

Loader = Callable[[int], np.ndarray]
Progress = Optional[Callable[[str, int, int], None]]

# the scans of the last removed objects: {"area": {f: px}, "held": {f: source frame}} by (mask stamp, frames)
_PLANS = LockedLRU(32)


def _px(a: Any) -> int:
    return int(np.count_nonzero(np.asarray(a) > 0.5))


def measure(a: Any) -> Tuple[int, bool]:
    """(area in px of alpha > 0.5, does it touch the frame border) – the border band is the kept-object check's
    (cu.LOST_BAND_REL x the long side, at least cu.LOST_BAND_MIN)."""
    m = np.asarray(a) > 0.5
    px = int(np.count_nonzero(m))
    if not px:
        return 0, False
    e = max(cu.LOST_BAND_MIN, int(round(cu.LOST_BAND_REL * max(m.shape[:2]))))
    return px, bool(m[:e].any() or m[-e:].any() or m[:, :e].any() or m[:, -e:].any())


def _runs(flags: Sequence[bool], value: bool) -> List[Tuple[int, int]]:
    out, i, n = [], 0, len(flags)
    while i < n:
        if bool(flags[i]) != value:
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]) == value:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def find_lost(frames: Sequence[int], area: Dict[int, int], touch: Dict[int, bool],
              alive: Optional[Callable[[int], bool]] = None) -> Dict[int, int]:
    """{lost frame: the frame whose alpha it takes} – the rule of the module's docstring. `frames` in order (positions
    count, not numbers); area / touch per frame as `measure` gives them; alive(f) False = outside the object's life."""
    frames = [int(f) for f in frames]
    n = len(frames)
    a = np.array([area.get(f, 0) if area.get(f, 0) >= cu.LOST_MIN_PX else 0 for f in frames], np.float64)
    present = a > 0
    while True:                              # stray pixels: far under the neighbours (they leave the medians together)
        drop = []
        for i in np.flatnonzero(present):
            nb = [a[j] for j in range(max(0, i - HOLD_WINDOW), min(n, i + HOLD_WINDOW + 1)) if j != i and present[j]]
            if len(nb) >= 2 and a[i] < HOLD_TINY * float(np.median(nb)):
                drop.append(i)
        if not drop:
            break
        present[drop] = False
    solid = np.zeros(n, bool)
    for s, e in _runs(present, True):
        if e - s + 1 >= cu.LOST_MIN_FRAMES:
            solid[s:e + 1] = True

    def leaving(b: int, step: int) -> bool:
        """The solid frame `b` next to a gap: at the border and shrunk against the frames further away (`step`)."""
        far = [a[j] for j in range(b + step, b + step * (HOLD_WINDOW + 1), step) if 0 <= j < n and solid[j]]
        return bool(touch.get(frames[b])) and bool(far) and a[b] < HOLD_EXIT_SHRINK * max(far)

    held: Dict[int, int] = {}
    for s, e in _runs(solid, False):
        left = s - 1 if s > 0 else None
        right = e + 1 if e < n - 1 else None
        if (left is None and right is None) or e - s + 1 > HOLD_MAX_GAP:
            continue
        out = left is not None and leaving(left, -1)
        back = right is not None and leaving(right, +1)
        if (out and back) if (left is not None and right is not None) else (out or back):
            continue                          # a real exit from the frame (or an entry): nothing to hold
        for i in range(s, e + 1):
            if present[i] or (alive is not None and not alive(frames[i])):
                continue                      # a short detection in the gap stays as it is
            dl = i - left if left is not None else None
            dr = right - i if right is not None else None
            src = left if dr is None or (dl is not None and dl <= dr) else right
            held[frames[i]] = frames[src]
    return held


class Hold:
    """A removal alpha loader that holds the nearest mask in the frames where the object is lost. `frames` = the work
    range (a list, or a function read at the first call); `key` = what the mask is made of (None: no shared cache);
    `alive(f)` = inside the object's life. Frames outside the range, and every frame that is not lost, pass through."""

    def __init__(self, fn: Loader, name: str, frames: Any, key: Any = None,
                 alive: Optional[Callable[[int], bool]] = None):
        self.fn, self.name, self.key, self.alive = fn, str(name), key, alive
        self._frames = frames
        self._own: Optional[Tuple[Tuple[int, ...], Dict[str, Any]]] = None
        self._lock = threading.Lock()

    def frames(self) -> Tuple[int, ...]:
        fr = self._frames() if callable(self._frames) else self._frames
        return tuple(int(f) for f in (fr or ()))

    def _scan(self, frames: Tuple[int, ...], progress: Progress, known: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        area: Dict[int, int] = {}
        touch: Dict[int, bool] = {}
        for i, f in enumerate(frames):
            if progress:
                progress(f"Czyszczenie: sprawdzam maskę „{self.name}” F{f}", i, len(frames))
            a = known[f] if known and f in known else self.fn(f)
            area[f], touch[f] = measure(a)
        return {"area": area, "held": find_lost(frames, area, touch, self.alive)}

    def plan(self, progress: Progress = None, fresh: bool = False, known: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        """{"area": {f: px}, "held": {f: source}} of the work range – scanned once (one loader read per frame), then
        from the cache. A scan that fails holds nothing (the loaders then behave as without this module)."""
        with self._lock:
            try:
                frames = self.frames()
            except Exception as e:  # noqa: BLE001 – no range = nothing to judge
                logger.warning("cleanup_hold %s: brak zakresu pracy (%s)", self.name, e)
                return {"area": {}, "held": {}}
            k = (self.key, frames) if self.key is not None else None
            if not fresh:
                hit = _PLANS.get(k) if k is not None else (self._own[1] if self._own and self._own[0] == frames else None)
                if hit is not None:
                    return hit
            try:
                p = self._scan(frames, progress, known)
            except Exception as e:  # noqa: BLE001 – e.g. a broken file: the run itself will say so at that frame
                logger.warning("cleanup_hold %s: sprawdzenie maski nie powiodło się (%s)", self.name, e)
                return {"area": {}, "held": {}}
            if k is not None:
                _PLANS.put(k, p)
            self._own = (frames, p)
            if p["held"]:
                logger.info("cleanup_hold %s: %s", self.name, held_text(p["held"]))
            return p

    def cached(self) -> Optional[Dict[str, Any]]:
        """The plan when the range has been scanned already (never scans)."""
        try:
            frames = self.frames()
        except Exception:  # noqa: BLE001
            return None
        if self.key is not None:
            return _PLANS.get((self.key, frames))
        return self._own[1] if self._own and self._own[0] == frames else None

    def forget(self) -> None:
        """Drop the scan (the mask changed): the next `plan` reads the range again."""
        try:
            frames = self.frames()
        except Exception:  # noqa: BLE001
            return
        if self.key is not None:
            _PLANS.put((self.key, frames), None)
        self._own = None

    def __call__(self, f: int) -> np.ndarray:
        """The frame's alpha, held where the scan found it lost. A call NEVER scans the range (review 2026-11-13: a
        lazy scan here made the first mask preview read every frame of the range – tens of seconds on a 4K group):
        the scan is made by `plan` / `held_report`, which every run calls up front; until then, and after the mask
        changed (the frame's area differs from the scan – the scan is dropped), the loader's own alpha passes through."""
        f = int(f)
        a = self.fn(f)
        p = self.cached()
        if p is None:
            return a
        if f in p["area"] and _px(a) != p["area"][f]:      # the mask changed since the scan: it no longer holds
            self.forget()
            return a
        src = p["held"].get(f)
        return a if src is None else self.fn(src)


def wrap(loaders: Sequence[Loader], names: Sequence[str], frames: Any, keys: Optional[Sequence[Any]] = None,
         alive: Optional[Sequence[Optional[Callable[[int], bool]]]] = None) -> List[Hold]:
    """The removal loaders with the hold, each object judged on its own."""
    return [Hold(fn, names[i], frames, keys[i] if keys else None, alive[i] if alive else None)
            for i, fn in enumerate(loaders)]


def stamp(store: Any, obj: Dict[str, Any], run: Optional[str]) -> Optional[tuple]:
    """A cheap signature of what the removal alpha of `obj` is made of (a group: its composition), for the scan cache:
    sources, life ranges and the mtimes of the results' json files. None = cannot tell (no shared cache). A change it
    misses is caught per frame by the loader (the area differs from the scan)."""
    try:
        from engine import groups
        name = obj["name"]
        if obj.get("kind") == "group":
            pairs = [(m, groups.member_run(store, m)["run"]) for m in groups.active_members(store, name)]
            out: List[Any] = [groups.group_sig(store, name)]
        else:
            pairs, out = [(name, run)], []
        for m, r in pairs:
            od = store.dir(m)
            out += [m, str(r), repr(store.get(m).get("life"))]
            for p in (od / "runs" / str(r or "").split(":")[-1] / "run.json", od / "core" / "core.json",
                      od / "core_fix" / "core_fix.json", od / "final_touch" / "final_touch.json"):
                try:
                    st = p.stat()
                    out.append((st.st_mtime_ns, st.st_size))
                except OSError:
                    out.append(0)
        return (str(store.dir(name)),) + tuple(out)
    except Exception as e:  # noqa: BLE001 – the hold still works, only without the shared cache
        logger.warning("cleanup_hold: sygnatura maski nie powiodła się (%s)", e)
        return None


def life_fn(store: Any, obj: Dict[str, Any]) -> Optional[Callable[[int], bool]]:
    """alive(f) of a removed object (a group: any active member alive); None = the whole shot."""
    from engine.workarea import is_visible, object_life
    if obj.get("kind") == "group":
        from engine import groups
        lives = [object_life(store.get(m)) for m in groups.active_members(store, obj["name"])]
    else:
        lives = [object_life(obj)]
    if not lives or any(lv is None for lv in lives):
        return None
    return lambda f: any(is_visible(f, lv) for lv in lives)


def _pieces(held: Dict[int, int]) -> List[Dict[str, Any]]:
    """[{frames: [...], from: source}] – consecutive held frames of one source together, in frame order."""
    out: List[Dict[str, Any]] = []
    for f in sorted(held):
        if out and out[-1]["from"] == held[f] and out[-1]["frames"][-1] == f - 1:
            out[-1]["frames"].append(f)
        else:
            out.append({"frames": [f], "from": int(held[f])})
    return out


def held_text(held: Dict[int, int]) -> str:
    return ", ".join(f"{cu.frame_ranges(p['frames'])} ← F{p['from']}" for p in _pieces(held))


def held_warning(name: str, held: Dict[int, int]) -> str:
    return (f"Obiekt usuwany „{name}” nie ma maski w {cu.frame_ranges(list(held))} (zgubiony) – podstawiona maska z "
            f"najbliższej klatki ({held_text(held)}); popraw maskę w tych klatkach, jeśli łata nie trafia w obiekt")


def held_report(loaders: Sequence[Any], frames: Optional[Sequence[int]] = None, progress: Progress = None) -> Dict[str, Any]:
    """What the removal loaders hold: {"objects": {name: [{"frames": [...], "from": frame}]}, "warnings": [Polish text]}
    – JSON-ready (run.json `held`, the API). `frames`: only these (a run shorter than the work range). Loaders
    without the hold are skipped. The first call scans the range (progress as the Cleaner's)."""
    only = None if frames is None else {int(f) for f in frames}
    objects: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    for ld in loaders:
        if not isinstance(ld, Hold):
            continue
        held = {f: s for f, s in ld.plan(progress)["held"].items() if only is None or f in only}
        if held:
            objects[ld.name] = _pieces(held)
            warnings.append(held_warning(ld.name, held))
    return {"objects": objects, "warnings": warnings}
