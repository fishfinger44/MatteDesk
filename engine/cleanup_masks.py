# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which mask the Cleanup mode removes for an object (user decisions 2026-11-10, step C1 „Inpaint”, until 2026-11-11 „Co usuwamy”).

The objects to remove are picked in the Objects palette – a whole group or single objects – and the mask is the one
chosen AT the object:
  * a group: its composition (the members' „Do grupy z:” alphas + seams, engine/groups.group_alpha_loader = run
    token "core" of the group entity);
  * a member of a group: its „Do grupy z:” (engine/groups.member_run) – the same choice the alpha uses;
  * an object without a group: its own choice in the palette, kept in the Cleanup settings `sources[name]` (a stage
    "core" / "matanyone" / "videomama" / "vitmatte" / "ltx" / "final_touch", or a version token "v###"); empty = the
    active version, else the mask 1+2.
An object with nothing to give („brak maski”) cannot be picked; `resolve` says why.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

STAGE_SOURCES = ("core", "matanyone", "videomama", "vitmatte", "ltx", "final_touch")


def available(store: Any, name: str) -> List[Dict[str, Any]]:
    """The masks an object without a group can give: [{source, run, valid}] – only what exists (as api/groups.py
    lists a member's „Do grupy z:”), the versions by number after the stages."""
    from engine import groups
    out: List[Dict[str, Any]] = []
    for src in STAGE_SOURCES:
        if src == "final_touch":
            run, ok, _ = groups.ft_run(store, name)
        else:
            run, ok, _ = groups._source_run(store, name, src, None)
        if run:
            out.append({"source": src, "run": run, "valid": bool(ok)})
    runs = store.dir(name) / "runs"
    for d in sorted(runs.glob("v[0-9][0-9][0-9]")) if runs.is_dir() else []:
        if (d / "run.json").exists():
            out.append({"source": d.name, "run": d.name, "valid": True})
    return out


def resolve(store: Any, obj: Dict[str, Any], sources: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """{run, source, valid, why, by}: the run token engine/export.run_alpha_loader reads for `obj` (None = no mask,
    `why` says so). `by` = where the choice lives: group | member („Do grupy z:”) | own (the Cleanup setting)."""
    from engine import groups
    from engine.export import CORE_RUN
    name = obj["name"]
    if obj.get("kind") == "group":
        ms = groups.active_members(store, name)
        if not ms:
            return {"run": None, "source": "group", "valid": False, "by": "group",
                    "why": "grupa pusta albo wszyscy członkowie wyłączeni"}
        miss = [m for m in (groups.member_run(store, x) for x in ms) if not m["run"]]
        if miss:
            return {"run": None, "source": "group", "valid": False, "by": "group",
                    "why": "; ".join(f"„{m['member']}” – {m['why']}" for m in miss)}
        return {"run": CORE_RUN, "source": "group", "valid": True, "by": "group", "why": ""}
    if obj.get("group"):
        m = groups.member_run(store, name)
        return {"run": m["run"], "source": m["source"], "valid": bool(m["valid"]), "by": "member",
                "why": "" if m["run"] else (m["why"] or "brak maski")}
    spec = str((sources or {}).get(name) or "")
    if spec in STAGE_SOURCES:
        run, ok, why = (groups.ft_run(store, name) if spec == "final_touch"
                        else groups._source_run(store, name, spec, None))
        return {"run": run, "source": spec, "valid": bool(ok), "by": "own", "why": "" if run else (why or "brak maski")}
    if spec:   # a version token (an older setting: "v###")
        ok = (store.dir(name) / "runs" / spec / "run.json").exists()
        return {"run": spec if ok else None, "source": spec, "valid": ok, "by": "own",
                "why": "" if ok else f"brak wersji {spec}"}
    if groups.has_own_source(obj):   # the object's own mask choice (palette Obiekty, 2026-11-15) – before the active one
        m = groups.member_run(store, name)
        return {"run": m["run"], "source": m["source"], "valid": bool(m["valid"]), "by": "object",
                "why": "" if m["run"] else (m["why"] or "brak maski")}
    act = (obj.get("tool") or {}).get("active_run")
    if act and (store.dir(name) / "runs" / act / "run.json").exists():
        return {"run": act, "source": "active", "valid": True, "by": "own", "why": ""}
    run, ok, why = groups._source_run(store, name, "core", None)
    return {"run": run, "source": "core", "valid": bool(ok), "by": "own",
            "why": "" if run else "brak maski – policz etap 1 (SAM 3.1) w trybie Wycinanie"}
