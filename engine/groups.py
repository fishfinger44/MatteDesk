# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/groups.py - A group of objects as its own entity (docs/PLAN_GROUP_COMPOSE.md, user decisions 2026-10-29).

Storage (variant A, accepted 2026-10-29): `objects/<group>/object.json` with `"kind": "group"` – so the group gets the
object machinery for free (runs/, tool.stage_runs, active_run, band, final_touch/, trash, rename, version deletion):
  {"kind": "group", "name": "body", "members": ["head", "torso", "legs"], "seam_px": 4,
   "band": {...like an object...}, "tool": {"active_run": "", "stage_runs": {}}, "export_source": null}
A member keeps its own object.json and gets:
  "group": "<group>"                          – at most one group per object; groups are never members
  "group_source": {"source": ..., "run": ...}  – „Do grupy z:” (GROUP_SOURCES; run only for "version"); default Core.
  "group_on": false                           – switched off in its group for now (palette checkbox, 2026-10-29): out of
                                                the composition (alpha, preview, whole recompute input), still a member.
  "group_seam_px": 3                          – its own seam reach (2026-10-29); absent = the group's seam_px.
The member's own `active_run` stays its export choice – it never feeds the group.
Depth order: the group has no number of its own – it is one layer at its front-most member (`group_depth`); members
keep their order inside it. `ObjectStore.list()` lists objects only, groups via `list_groups`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from engine.export import SEAM_PX_DEFAULT, SEAM_PX_MAX
from engine.objects import ObjectStore

GROUP_KIND = "group"
GROUP_SOURCES = ("core", "matanyone", "videomama", "vitmatte", "ltx", "version", "final_touch")
DEFAULT_SOURCE = {"source": "core", "run": None}

StoreLike = Union[ObjectStore, str, Path]


def _store(s: StoreLike) -> ObjectStore:
    return s if isinstance(s, ObjectStore) else ObjectStore(s)


def is_group(obj: Optional[Dict[str, Any]]) -> bool:
    return bool(obj) and obj.get("kind") == GROUP_KIND


def _exists(store: ObjectStore, name: str) -> bool:
    return (store.dir(name) / "object.json").exists()


def get_group(s: StoreLike, group: str) -> Dict[str, Any]:
    store = _store(s)
    g = store.get(group)
    if not is_group(g):
        raise ValueError(f"„{group}” nie jest grupą")
    return g


def list_groups(s: StoreLike) -> List[Dict[str, Any]]:
    store = _store(s)
    out = []
    for d in sorted(store.root.iterdir()) if store.root.exists() else []:
        if (d / "object.json").exists():
            o = store._read(d / "object.json")
            if is_group(o):
                out.append(o)
    return out


def members(s: StoreLike, group: str) -> List[str]:
    """Members of the group in depth order (front first)."""
    store = _store(s)
    ms = list(get_group(store, group).get("members") or [])
    return sorted(ms, key=lambda n: store.get(n).get("depth_order", 0))


def is_on(obj: Dict[str, Any]) -> bool:
    """A member switched on in its group (palette checkbox; default on)."""
    return obj.get("group_on", True) is not False


def active_members(s: StoreLike, group: str) -> List[str]:
    """Members that make the composition (switched on), depth order."""
    store = _store(s)
    return [m for m in members(store, group) if is_on(store.get(m))]


def set_member_on(s: StoreLike, group: str, name: str, on: bool) -> bool:
    """Switch a member on / off in its group (user decision 2026-10-29): off = out of the composition (and so ↻ on a
    whole recompute, like a change of its source), still a member; its own results are untouched."""
    store = _store(s)
    if name not in (get_group(store, group).get("members") or []):
        raise ValueError(f"„{name}” nie należy do grupy „{group}”")
    store.update(name, group_on=bool(on))
    return bool(on)


def group_of(s: StoreLike, name: str) -> Optional[str]:
    return _store(s).get(name).get("group") or None


def member_source(obj: Dict[str, Any]) -> Dict[str, Any]:
    """„Do grupy z:” of a member ({"source", "run"}); missing / broken = Core."""
    src = obj.get("group_source")
    if isinstance(src, dict) and src.get("source") in GROUP_SOURCES:
        return {"source": src["source"], "run": src.get("run") if src["source"] == "version" else None}
    return dict(DEFAULT_SOURCE)


def group_depth(s: StoreLike, group: str) -> Optional[int]:
    """The group's place in the depth order = its front-most member's; None for an empty group."""
    store = _store(s)
    ds = [store.get(n).get("depth_order", 0) for n in get_group(store, group).get("members") or []]
    return min(ds) if ds else None


def seam(s: StoreLike, group: str) -> int:
    from engine.export import seam_px
    return seam_px(get_group(s, group))


def member_seam(s: StoreLike, group: str, name: str) -> int:
    """A member's seam reach (user decision 2026-10-29: hair small, a dress large): its own `group_seam_px`, else the
    group's seam."""
    store = _store(s)
    v = store.get(name).get("group_seam_px")
    return seam(store, group) if v is None else _clamp_seam(v)


def seam_gaps(s: StoreLike, group: str, names: List[str]) -> List[int]:
    """The seam reach of each of `names` (engine/export.seamless_union `gaps`)."""
    store = _store(s)
    return [member_seam(store, group, n) for n in names]


def set_member_seam(s: StoreLike, group: str, name: str, px: Any) -> Optional[int]:
    """A member's own seam reach (px, clamped) or None = back to the group's seam. Part of the group signature (↻)."""
    store = _store(s)
    if name not in (get_group(store, group).get("members") or []):
        raise ValueError(f"„{name}” nie należy do grupy „{group}”")
    v = None if px is None else _clamp_seam(px)
    store.update(name, group_seam_px=v)
    return v


def _check_member(store: ObjectStore, name: str, group: str) -> Dict[str, Any]:
    if name == group:
        raise ValueError("grupa nie może należeć do siebie")
    if not _exists(store, name):
        raise FileNotFoundError(f"brak obiektu {name}")
    o = store.get(name)
    if is_group(o):
        raise ValueError(f"„{name}” jest grupą – grupy nie mogą należeć do grup")
    if o.get("group") and o["group"] != group:
        raise ValueError(f"„{name}” należy już do grupy „{o['group']}”")
    if o.get("part_of"):
        raise ValueError(f"„{name}” jest częścią „{o['part_of']}” (stary model części) – najpierw odłącz")
    return o


def create_group(s: StoreLike, group: str, member_names: Optional[List[str]] = None,
                 seam_px: int = SEAM_PX_DEFAULT) -> Dict[str, Any]:
    """New group `group` (may start empty – objects are dropped into it in the palette)."""
    store = _store(s)
    d = store.dir(group)
    if (d / "object.json").exists():
        raise FileExistsError(f"obiekt {group} już istnieje")
    ms = list(dict.fromkeys(member_names or []))
    for m in ms:
        _check_member(store, m, group)
    # a full object (edge settings, band, zones/, runs/ … – the stages run on it as on an object, step 4) + the group
    # fields; its depth_order is not used (group_depth)
    store.create(group, [], depth_order=0)
    store.update(group, kind=GROUP_KIND, members=[], seam_px=_clamp_seam(seam_px), export_source=None,
                 tool={"stage_runs": {}})
    for m in ms:
        add_member(store, group, m)
    return get_group(store, group)


def add_member(s: StoreLike, group: str, name: str) -> Dict[str, Any]:
    store = _store(s)
    g = get_group(store, group)
    o = _check_member(store, name, group)
    if name not in g["members"]:
        store.update(group, members=g["members"] + [name])
    store.update(name, group=group, group_source=member_source(o) if o.get("group") == group else dict(DEFAULT_SOURCE))
    return get_group(store, group)


def whole_results(s: StoreLike, group: str) -> List[str]:
    """What the group computed as a whole: its stage results (tool.stage_runs keys) + "final_touch" when it has hand
    corrections. Empty = nothing is lost when its members change."""
    store = _store(s)
    g = get_group(store, group)
    out = sorted(k for k, v in ((g.get("tool") or {}).get("stage_runs") or {}).items() if v)
    if any((store.dir(group) / "final_touch").glob("*.json")):
        out.append("final_touch")
    return out


def trash_whole(s: StoreLike, group: str) -> Optional[Path]:
    """The group's whole results -> <work>/trash/groups/<group>_<time>/ (runs/, final_touch/, previews; recoverable
    by hand, like a deleted object). Version numbers are never reused (runs/.last_run). None = nothing to move."""
    import shutil
    import time
    from engine.run_trash import LAST_RUN, last_run_number
    store = _store(s)
    get_group(store, group)
    d = store.dir(group)
    subs = [x for x in ("runs", "final_touch", "stage_preview") if (d / x).is_dir() and any((d / x).iterdir())]
    if not subs and not whole_results(store, group):
        return None
    n = last_run_number(d / "runs")
    dest = store.root.parent / "trash" / "groups" / f"{group}_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    for x in subs:
        shutil.move(str(d / x), str(dest / x))
    (d / "runs").mkdir(exist_ok=True)
    if n:
        (d / "runs" / LAST_RUN).write_text(str(n), encoding="utf-8")
    store.update(group, tool={"stage_runs": {}, "active_run": "", "final_valid": False})
    return dest


def _whole_warning(store: ObjectStore, group: str, what: str) -> Optional[Dict[str, Any]]:
    stages = whole_results(store, group)
    if not stages:
        return None
    label = {"matanyone": "MatAnyone2", "videomama": "VideoMaMa", "ltx": "LTX-2.5", "final_touch": "Final Touch"}
    names = ", ".join(label.get(k.split("@")[0], "ViTMatte" if k.startswith("vitmatte") else k) for k in stages)
    return {"success": False, "needs_confirm": True, "whole_stages": stages,
            "warning": f"{what}: wynik przeliczenia całości grupy „{group}” ({names}) zostanie przeniesiony do kosza "
                       f"projektu. Członkowie zachowują swoje wyniki."}


def remove_member(s: StoreLike, group: str, name: str, confirm: bool = False) -> Dict[str, Any]:
    """Detach a member; its own results stay untouched. When the group has whole results (a recompute of the whole or
    Final Touch): without `confirm` nothing happens and the warning comes back (needs_confirm, whole_stages); with
    it they go to the project trash (trash_whole) – user decision 2026-10-29."""
    store = _store(s)
    g = get_group(store, group)
    if name not in (g.get("members") or []):
        raise ValueError(f"„{name}” nie należy do grupy „{group}”")
    w = _whole_warning(store, group, f"Odłączenie „{name}”")
    if w and not confirm:
        return w
    trash = trash_whole(store, group) if w else None
    store.update(group, members=[m for m in get_group(store, group)["members"] if m != name])
    if _exists(store, name):
        store.update(name, group=None, group_source=None, group_on=None, group_seam_px=None)
    return {"success": True, "trash": str(trash) if trash else None}


def set_member_source(s: StoreLike, group: str, name: str, source: Dict[str, Any]) -> Dict[str, Any]:
    """„Do grupy z:” of a member: {"source": GROUP_SOURCES, "run": version for "version"}."""
    store = _store(s)
    if name not in (get_group(store, group).get("members") or []):
        raise ValueError(f"„{name}” nie należy do grupy „{group}”")
    src = (source or {}).get("source")
    if src not in GROUP_SOURCES:
        raise ValueError(f"źródło musi być jednym z {GROUP_SOURCES}, jest {src!r}")
    run = (source or {}).get("run") if src == "version" else None
    if src == "version" and not (run and (store.dir(name) / "runs" / run / "run.json").exists()):
        raise ValueError(f"brak wersji {run or ''} obiektu „{name}”".replace("  ", " "))
    store.update(name, group_source={"source": src, "run": run})
    return member_source(store.get(name))


def _clamp_seam(px: Any) -> int:
    try:
        v = int(round(float(px)))
    except (TypeError, ValueError):
        raise ValueError(f"szew musi być liczbą px, jest {px!r}")
    return max(0, min(SEAM_PX_MAX, v))


def set_seam(s: StoreLike, group: str, px: Any) -> int:
    store = _store(s)
    get_group(store, group)
    v = _clamp_seam(px)
    store.update(group, seam_px=v)
    return v


def ungroup(s: StoreLike, group: str, confirm: bool = False) -> Dict[str, Any]:
    """Delete the group: members become standalone (their results untouched), the group's folder (with its whole
    results) -> project trash. With whole results: the warning first, as remove_member."""
    store = _store(s)
    w = _whole_warning(store, group, "Rozgrupowanie")
    if w and not confirm:
        return w
    for m in list(get_group(store, group).get("members") or []):
        if _exists(store, m):
            store.update(m, group=None, group_source=None, group_on=None, group_seam_px=None)
    store.update(group, members=[])
    return {"success": True, "trash": str(store.remove(group))}


# ---------------------------------------------------------------- the composition (step 3)
SOURCE_LABEL = {"core": "Core (maska 1+2)", "matanyone": "MatAnyone2 (etap 3)", "videomama": "VideoMaMa (etap 4)",
                "vitmatte": "ViTMatte (etap 5)", "ltx": "LTX-2.5 (etap 7)", "version": "wersja",
                "final_touch": "Final Touch (etap 8)"}


def member_run(s: StoreLike, name: str) -> Dict[str, Any]:
    """What a member brings into its group („Do grupy z:”): {member, source, run, valid, why}. run = the version token
    for engine/export.run_alpha_loader ("core" = the mask 1+2), None = nothing to bring (why says so – never a silent
    substitute). valid False = a stale result (still used, the UI marks it ↻)."""
    store = _store(s)
    obj = store.get(name)
    src = member_source(obj)
    out = {"member": name, "source": src["source"], "run": None, "valid": False, "why": ""}
    if src["source"] == "final_touch":   # the member's Final Touch: its base (chosen on stage 8) + its hand corrections
        run, ok, why = ft_run(store, name)
        out.update(run=run, valid=ok, why=why)
        return out
    run, ok, why = _source_run(store, name, src["source"], src["run"])
    out.update(run=run, valid=ok, why=why)
    return out


def _source_run(store: ObjectStore, name: str, source: str, run: Optional[str]) -> tuple:
    """(run token | None, valid, why) of one source of an object: core (mask 1+2), a stage, a version, the active one."""
    from engine.export import CORE_RUN
    from engine.stages import core_fix_state, stage_run
    od = store.dir(name)
    src = {"source": source, "run": run}
    if src["source"] == "core":
        if not (od / "core").is_dir() or not any((od / "core").glob("c_*.png")):
            return None, False, "brak rdzenia SAM 3.1 – policz etap 1"
        exists, ok = core_fix_state(store, name)
        return CORE_RUN, ok or not exists, "" if ok or not exists else "maska 1+2 nieaktualna – „Zastosuj poprawki” (etap 2)"
    if src["source"] in ("version", "active"):
        run = src["run"] if src["source"] == "version" else (store.get(name).get("tool") or {}).get("active_run")
        if not run or not (od / "runs" / run / "run.json").exists():
            return None, False, f"brak wersji {run or ''}".strip()
        return run, True, ""
    run, ok, why = stage_run(store, name, src["source"])
    if not run:
        return None, False, why or f"brak wyniku: {SOURCE_LABEL.get(src['source'], src['source'])}"
    return run, ok, why


def ft_run(s: StoreLike, name: str) -> tuple:
    """(run token "ft:<base>", valid, why) of an object's Final Touch: the base chosen on stage 8, else the result of
    its last computed stage (final_touch.DEFAULT_BASE_ORDER; a stale one stays the base, marked not valid – like the
    app's default base, user decision 2026-10-30: no silent jump to the next current stage)."""
    from engine.final_touch import DEFAULT_BASE_ORDER, FT_PREFIX, load as ft_load
    store = _store(s)
    b = ft_load(store.dir(name)).get("base") or {}
    if b.get("source"):
        run, ok, why = _source_run(store, name, b["source"], b.get("run"))
        return (FT_PREFIX + run, ok, why) if run else (None, False, f"Final Touch – baza: {why}")
    for src in DEFAULT_BASE_ORDER:
        run, ok, why = _source_run(store, name, src, None)
        if run:
            return FT_PREFIX + run, ok, "" if ok else f"Final Touch – baza nieaktualna: {why}"
    return None, False, "Final Touch – brak bazy (najpierw rdzeń SAM 3.1, etap 1)"


class GroupSourceMissing(ValueError):
    """A member has nothing to bring into the composition (`missing` = [member_run dicts])."""

    def __init__(self, group: str, missing: List[Dict[str, Any]]):
        self.missing = missing
        super().__init__(f"Grupa „{group}”: " + "; ".join(f"„{m['member']}” – {m['why']}" for m in missing))


# the last compositions (frame previews ask for the same frames again – Final Touch, the overlay, the zone base):
# keyed by the group's signature (members, „Do grupy z:” sources and their mask 1+2, on / off, seam) and the members'
# result files' mtimes, so any change of the input is a new entry; ~8 MB per 1080p frame
_CACHE: "Dict[tuple, Any]" = {}
_CACHE_MAX = 16


def _mtimes(store: ObjectStore, runs: List[Dict[str, Any]]) -> tuple:
    out = []
    for r in runs:
        rd = store.dir(r["member"]) / "runs" / str(r["run"]).split(":")[-1]
        for p in (rd / "run.json", store.dir(r["member"]) / "core" / "core.json",
                  store.dir(r["member"]) / "final_touch" / "final_touch.json"):
            try:
                out.append(p.stat().st_mtime_ns)
            except OSError:
                out.append(0)
    return tuple(out)


def group_alpha_loader(s: StoreLike, group: str, w: int, h: int):
    """f -> the basic group matte: seamless_union of the members' chosen alphas („Do grupy z:”), seams between their
    SAM 3.1 cores filled (the group's seam_px) – the same rule as the legacy `<parent>_full`. Nothing is computed;
    objects in front are not subtracted (the export does that). A member outside its life range brings nothing.
    GroupSourceMissing when a member has no result of its source; an empty group -> ValueError."""
    import cv2
    import numpy as np
    from engine.export import run_alpha_loader, seamless_union
    from engine.workarea import is_visible, object_life
    store = _store(s)
    ms = active_members(store, group)
    if not ms:
        raise ValueError(f"Grupa „{group}” jest pusta" if not members(store, group)
                         else f"Grupa „{group}”: wszyscy członkowie wyłączeni (checkbox w palecie Obiekty)")
    runs = [member_run(store, m) for m in ms]
    missing = [r for r in runs if not r["run"]]
    if missing:
        raise GroupSourceMissing(group, missing)
    loaders = [run_alpha_loader(store.dir(r["member"]), r["run"], w, h) for r in runs]
    lives = [object_life(store.get(m)) for m in ms]
    gap = seam(store, group)
    gaps = seam_gaps(store, group, ms)
    ckey = (str(store.root.resolve()), group, group_sig(store, group), _mtimes(store, runs), int(w), int(h))

    def core(m: str, f: int):
        c = cv2.imread(str(store.dir(m) / "core" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if c is None:
            return None
        if c.shape != (h, w):
            c = cv2.resize(c, (w, h), interpolation=cv2.INTER_NEAREST)
        return c > 127

    def load(f: int) -> "np.ndarray":
        k = ckey + (int(f),)
        hit = _CACHE.get(k)
        if hit is not None:
            _CACHE[k] = _CACHE.pop(k)   # most recent last
            return hit.copy()
        vis = [is_visible(f, life) for life in lives]
        alphas = [ld(f) if v else np.zeros((h, w), np.float32) for ld, v in zip(loaders, vis)]
        out = seamless_union(alphas, [core(m, f) if v else None for m, v in zip(ms, vis)], gap_px=gap, gaps=gaps)
        _CACHE[k] = out.copy()
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        return out
    return load


def group_sig(s: StoreLike, group: str) -> str:
    """Signature of the group's input (the mask its whole recompute starts from, stages.mask_sig of a group): the
    members in depth order, what each brings („Do grupy z:” source + version; the mask 1+2 signature – also the core
    of the seam) and the seam width. A change of any = ↻ on the whole recompute (user decision 2026-10-29); a member's
    own export choice (active_run) is not part of it."""
    import hashlib
    import json
    from engine.stages import core_fix_sig
    store = _store(s)
    parts = []
    for m in active_members(store, group):   # a member switched off is out of the input
        r = member_run(store, m)
        p = [m, r["source"], r["run"], core_fix_sig(store, m)]
        own = store.get(m).get("group_seam_px")
        if own is not None:   # its own seam reach (2026-10-29) – only when set: older signatures stay
            p.append(["seam", _clamp_seam(own)])
        if r["source"] == "final_touch":   # the hand corrections (and finishing) are part of what it brings
            from engine.final_touch import corrections_sig, load as ft_load
            p.append(corrections_sig(ft_load(store.dir(m))))
        parts.append(p)
    blob = json.dumps({"members": parts, "seam": seam(store, group)}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- migration of the legacy parts (step 6)
# object fields that make the identity / state of the parent itself – everything else (edge settings, band, stage
# settings of VideoMaMa / LTX-2.5 / ViTMatte base …) is copied to the new group, so its migrated whole results keep
# their settings signatures
_NOT_COPIED = {"schema", "name", "detect", "flags", "part_of", "part_seam_px", "tool", "life", "depth_order", "group",
               "group_source", "group_on", "group_seam_px", "export_selected", "kind", "members", "seam_px", "edge_mode_auto"}


def _legacy_mask_sig(store: ObjectStore, parent: str, parts: List[str]) -> str:
    """The input signature a whole computed on the parent recorded before 2026-10-29 (stages.mask_sig of a parent
    with parts) – a migrated version that matches it was current, it gets the group's signature instead of ↻."""
    from engine.stages import _sha, core_fix_sig
    return _sha({"members": [[m, core_fix_sig(store, m)] for m in [parent] + parts],
                 "seam": store.get(parent).get("part_seam_px")})


def migrate_part_of(s: StoreLike) -> List[Dict[str, Any]]:
    """Legacy parts (`part_of`: a parent + its parts, whole computed on the parent) -> a group entity (user decision
    2026-10-29, simple migration – the feature was not exposed): group `<parent>_grp` with members [parent, parts]
    (all „Do grupy z:” Core), seam = the parent's part_seam_px; the parent's whole versions (run.json "group") move to
    the group with their stage results (still current ones get the group signature), the parent's Final Touch and
    ViTMatte zones of the whole go with them, its band and stage settings are copied (the parent keeps its own copy).
    Idempotent; returns what was migrated."""
    import json
    import shutil

    from engine.stages import mask_sig
    store = _store(s)
    done: List[Dict[str, Any]] = []
    objs = store.list()
    names = {o["name"] for o in objs}
    parents: Dict[str, List[str]] = {}
    for o in objs:
        p = o.get("part_of")
        if p and p != o["name"] and p in names:
            parents.setdefault(p, []).append(o["name"])
        elif p:
            store.update(o["name"], part_of=None)        # a dangling relation: just dropped
    for parent, parts in parents.items():
        parts = sorted(parts, key=lambda n: store.get(n).get("depth_order", 0))
        pobj = store.get(parent)
        old_sig = _legacy_mask_sig(store, parent, parts)
        gname, i = f"{parent}_grp", 2
        while (store.dir(gname) / "object.json").exists():
            gname, i = f"{parent}_grp{i}", i + 1
        for m in [parent] + parts:
            store.update(m, part_of=None)
        create_group(store, gname, [parent] + parts, seam_px=pobj.get("part_seam_px", SEAM_PX_DEFAULT))
        store.update(gname, **{k: v for k, v in pobj.items() if k not in _NOT_COPIED})
        pd, gd = store.dir(parent), store.dir(gname)
        moved = []
        for rd in sorted((pd / "runs").glob("v[0-9][0-9][0-9]")):
            try:
                rj = json.loads((rd / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not rj.get("group"):
                continue
            shutil.move(str(rd), str(gd / "runs" / rd.name))
            moved.append(rd.name)
        new_sig = mask_sig(store, gname)
        for r in moved:   # a whole that was current before stays current (the same input, now the group's)
            p = gd / "runs" / r / "run.json"
            rj = json.loads(p.read_text(encoding="utf-8"))
            if rj.get("core_fix_sig") == old_sig:
                rj["core_fix_sig"] = new_sig
                p.write_text(json.dumps(rj, indent=2, ensure_ascii=False), encoding="utf-8")
        ptool = store.get(parent).get("tool") or {}
        sr = dict(ptool.get("stage_runs") or {})
        gsr = {k: v for k, v in sr.items() if v in moved}
        store.update(parent, tool={"stage_runs": {k: v for k, v in sr.items() if v not in moved},
                                   **({"active_run": "", "final_valid": False} if ptool.get("active_run") in moved else {})})
        store.update(gname, tool={"stage_runs": gsr,
                                  **({"active_run": ptool["active_run"], "final_valid": bool(ptool.get("final_valid"))}
                                     if ptool.get("active_run") in moved else {})})
        if moved and (pd / "final_touch").is_dir():   # Final Touch of the whole goes with it
            shutil.move(str(pd / "final_touch"), str(gd / "final_touch"))
        if moved and (pd / "zones").is_dir():          # ViTMatte zones of the whole: the group's (copies)
            (gd / "zones").mkdir(exist_ok=True)
            for zp in sorted((pd / "zones").glob("z_*.json")):
                try:
                    z = json.loads(zp.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if z.get("criterion") == "matte":
                    shutil.copy2(str(zp), str(gd / "zones" / zp.name))
        store.update(parent, part_seam_px=None)
        done.append({"group": gname, "members": [parent] + parts, "runs": moved})
    return done
