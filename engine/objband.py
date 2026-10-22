# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/objband.py - the shared band of an object (user decision 2026-10-22, docs/PLAN_WSPOLNY_PAS.md).

The edge band (band control points, their keys in time, CoTracker3 tracking) is a feature of the OBJECT, not of a
stage. One set of band points per object lives in the object field `band`
({band_points, band_ref_out, band_ref_in}) of the group's main object (a part reads its group's band). Every stage with
a band – 3 MatAnyone2 (`matanyone`), 4 VideoMaMa (`videomama`), 7 LTX-2.5 (`ltx`), each whole-outline zone of stage 5
(`zone:<id>`) – reads it with its own sliders (scale = slider / band_ref, engine/zones.band_scales) and its own feathers.

`band_modes` of an object: {key: "own"} = „odłącz → własny pas”: the stage keeps its own copy of the points (the
behaviour before 2026-10-22 – the stage's own fields); absent = "shared". Zones of a part keep their mode on the part.

An object without the `band` field is not migrated yet: everything reads the stage's own points (the former behaviour)
until `migrate` runs (ObjectStore.get, the first read). Migration: exactly one owner with points (or several with the
very same points) → those become the object band and the owners shared; several different → they stay own (nothing is
lost), the object band empty until „ustaw jako wspólny”."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KEYS = ("band_points", "band_ref_out", "band_ref_in")
STAGES = ("matanyone", "videomama", "ltx")
SHARED_OWNER = "band"          # tracking cache of the object band (ZoneCache owner id)


def empty() -> Dict[str, Any]:
    return {"band_points": [], "band_ref_out": None, "band_ref_in": None}


def migrated(obj: Optional[Dict[str, Any]]) -> bool:
    return isinstance((obj or {}).get("band"), dict)


def shared(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The object band as stored (an object of the group's main object)."""
    b = (obj or {}).get("band") or {}
    return {"band_points": list(b.get("band_points") or []), "band_ref_out": b.get("band_ref_out"),
            "band_ref_in": b.get("band_ref_in")}


def mode(obj: Optional[Dict[str, Any]], key: str) -> str:
    """"shared" | "own" of a band owner; an object not migrated yet = "own" (its stages read their own points)."""
    if not migrated(obj):
        return "own"
    return "own" if ((obj or {}).get("band_modes") or {}).get(key) == "own" else "shared"


def own_params(obj: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """The stage's own copy of the points (zones: use the zone's params)."""
    o = obj or {}
    src = {"videomama": o.get("videomama"), "ltx": o.get("ltx"), "matanyone": o.get("edge_band")}.get(key) or {}
    return {k: src.get(k) for k in KEYS}


def effective(obj: Optional[Dict[str, Any]], key: str, own: Optional[Dict[str, Any]] = None,
              root: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
    """(params with band_points / band_ref_*, tracking owner id) the owner `key` of `obj` works with: the object band
    of `root` (default `obj`) when shared, else its own copy `own` (default: own_params)."""
    if mode(obj, key) == "shared":
        return shared(root if root is not None else obj), SHARED_OWNER
    p = own if own is not None else own_params(obj, key)
    return {k: p.get(k) for k in KEYS}, (key if key.startswith("zone:") else f"stage_{key}")


def apply(obj: Optional[Dict[str, Any]], key: str, s: Dict[str, Any]) -> Dict[str, Any]:
    """Stage settings `s` (vmm_settings / ltx_settings) with the band the stage works with + `band_owner`."""
    if mode(obj, key) == "shared":
        from engine.bandmap import band_points
        b = shared(obj)
        s["band_points"] = band_points({"band_points": b["band_points"]}) if b["band_points"] else []
        s["band_ref_out"], s["band_ref_in"] = b["band_ref_out"], b["band_ref_in"]
        s["band_owner"] = SHARED_OWNER
    else:
        s["band_owner"] = f"stage_{key}"
    return s


def edge_band(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The band points of stage 3 MatAnyone2 (former object field `edge_band`): None = no points. Shared: only in the
    px mode of the edge band (the points need it – the adaptive mode never reads them, so they stay out of its sig)."""
    if mode(obj, "matanyone") == "own":
        return obj.get("edge_band") or None
    from engine.edge import band_px_limits
    b = shared(obj)
    return b if (b["band_points"] and band_px_limits(obj)) else None


def zone_params(zone: Dict[str, Any], obj: Optional[Dict[str, Any]], root: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The params of a zone (raw dict) with the band it works with (shared: the band keys replaced by the object band
    of `root`; an empty object band = none, the constant band)."""
    p = dict(zone.get("params") or {})
    if mode(obj, f"zone:{zone.get('zone_id')}") != "shared":
        return p
    p = {k: v for k, v in p.items() if k not in KEYS}
    b = shared(root)
    if b["band_points"]:
        p.update(b)
    return p


# ----------------------------------------------------------------------------- migration
def _read(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(p: Path, d: Dict[str, Any]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _canon(p: Dict[str, Any]) -> str:
    return json.dumps([p.get("band_points") or [], p.get("band_ref_out"), p.get("band_ref_in")], sort_keys=True)


def migrate(root_dir: Path, name: str) -> None:
    """Migrate the object `name` (objects/<name>) and its group: the main object gets `band`, every member `band`
    (a part: {} – it reads its group's band) and `band_modes`. Idempotent; never touches a migrated object."""
    od = Path(root_dir) / name
    obj = _read(od / "object.json")
    if obj is None or migrated(obj):
        return
    main = obj.get("part_of") or name
    md = Path(root_dir) / main
    mobj = _read(md / "object.json") if main != name else obj
    if mobj is None:            # an orphan part: migrate it as a standalone object
        main, md, mobj = name, od, obj
    if migrated(mobj) and main != name:     # the group is migrated, this part was added later
        obj["band"] = {}
        obj.setdefault("band_modes", {})
        _write(od / "object.json", obj)
        return
    members: List[Tuple[str, Path, Dict[str, Any]]] = [(main, md, mobj)]
    for p in sorted(Path(root_dir).glob("*/object.json")):
        o = _read(p)
        if o and o.get("part_of") == main and o.get("name") != main and not migrated(o):
            members.append((o.get("name") or p.parent.name, p.parent, o))
    owners: List[Tuple[str, str, Dict[str, Any]]] = []      # (member, key, params)
    for key in STAGES:
        p = own_params(mobj, key)
        if p.get("band_points"):
            owners.append((main, key, p))
    zone_files: Dict[Tuple[str, str], Tuple[Path, Dict[str, Any]]] = {}
    for mname, mdir, _o in members:
        for zp in sorted((mdir / "zones").glob("z_*.json")) if (mdir / "zones").exists() else []:
            z = _read(zp)
            if z and (z.get("params") or {}).get("band_points"):
                key = f"zone:{z.get('zone_id') or zp.stem}"
                zone_files[(mname, key)] = (zp, z)
                owners.append((mname, key, {k: z["params"].get(k) for k in KEYS}))
    band = empty()
    modes: Dict[str, Dict[str, str]] = {m: {} for m, _d, _o in members}
    if owners and len({_canon(p) for _m, _k, p in owners}) == 1:
        band = {k: owners[0][2].get(k) for k in KEYS}
        for mname, key, _p in owners:            # the points now live in the object band: clear the own copies
            if key.startswith("zone:"):
                zp, z = zone_files[(mname, key)]
                z["params"] = {k: v for k, v in (z.get("params") or {}).items() if k not in KEYS}
                _write(zp, z)
            elif key == "matanyone":
                mobj["edge_band"] = None
            else:
                mobj[key] = {**(mobj.get(key) or {}), "band_points": [], "band_ref_out": None, "band_ref_in": None}
    else:
        for mname, key, _p in owners:
            modes[mname][key] = "own"
    for mname, mdir, o in members:
        o["band"] = band if mname == main else {}
        o["band_modes"] = {**(o.get("band_modes") or {}), **modes[mname]}
        _write(mdir / "object.json", o)
