# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/objband.py - the shared band of an object (user decision 2026-10-22, docs/PLAN_WSPOLNY_PAS.md).

The edge band (band control points, their keys in time, CoTracker3 tracking) is a feature of the OBJECT, not of a
stage. One set of band points per object lives in the object field `band`
({band_points, band_ref_out, band_ref_in}) of the group's main object (a part reads its group's band). Every stage with
a band – 3 MatAnyone2 (`matanyone`), 4 VideoMaMa (`videomama`), 7 LTX-2.5 (`ltx`), each whole-outline zone of stage 5
(`zone:<id>`) – reads it with its own sliders (scale = slider / band_ref, engine/zones.band_scales) and its own feathers.

Since 2026-10-24 (user decision) the band SLIDERS are a feature of the object too: `band` also holds `out_px`,
`in_px`, `inset_feather_px`, `outer_feather_px` (None = engine/bandsys.BAND_DEFAULTS) – every owner on the shared
band reads them (`sliders`), a slider moved in any stage moves the band of every shared stage (↻ there, as a point edit).
`band_modes` of an object: {key: "own"} = „odłącz → własny pas”: ONE detach for the points and the sliders – the
stage keeps its own copy of both (the stage's own fields: band_out_px…, videomama.reach_px…, ltx.region_px…, the zone's
outline_out_px / fg_erode_px / force_core_erode_px / inset_feather_px / outer_feather_px); absent = "shared".
Zones of a part keep their mode on the part.

An object without the `band` field is not migrated yet: everything reads the stage's own points (the former behaviour)
until `migrate` runs (ObjectStore.get, the first read). Migration: exactly one owner with points (or several with the
very same points) → those become the object band and the owners shared; several different → they stay own (nothing is
lost), the object band empty until „ustaw jako wspólny”.

ONE band settings model (user request 2026-10-24, „jeden wzór, reszta instancje”): `BandSettings` is what every band
owner works with – the four sliders, the points with their reference, the tracking owner id, shared / detached – and
`band_for(obj, key, root, zone, overrides, own)` is the ONLY resolver of it (the shared object band unless the owner is
detached, unset = engine/bandsys.BAND_DEFAULTS, `overrides` = palette values not saved yet). The stage settings
functions (edge.edge_settings, videomama.vmm_settings, ltx_alpha.ltx_settings, zone_params) embed it as `s["band"]` and
keep their legacy field names only as a thin mapping (`legacy_fields`) for old readers; `scales` is the one
slider / band_ref arithmetic, `pipeline.ObjectPipeline.band_points_fn` the one points function, `save_sliders` the one
save path of a slider moved in any palette. A change of band behaviour touches this module / bandsys.py only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KEYS = ("band_points", "band_ref_out", "band_ref_in")
SLIDERS = ("out_px", "in_px", "inset_feather_px", "outer_feather_px")   # the shared band sliders (2026-10-24)
STAGES = ("matanyone", "videomama", "ltx")
SHARED_OWNER = "band"          # tracking cache of the object band (ZoneCache owner id)
# the stage's own fields of the four sliders (own mode / before the slider migration)
OWN_SLIDER_FIELDS = {
    "matanyone": {"out_px": "band_out_px", "in_px": "band_in_px", "inset_feather_px": "band_in_feather_px",
                  "outer_feather_px": "band_out_feather_px"},
    "videomama": {"out_px": "reach_px", "in_px": "inset_px", "inset_feather_px": "feather_px", "outer_feather_px": "outer_feather_px"},
    "ltx": {"out_px": "region_px", "in_px": "core_erode_px", "inset_feather_px": "feather_px", "outer_feather_px": "region_soft_px"},
}


def empty() -> Dict[str, Any]:
    return {"band_points": [], "band_ref_out": None, "band_ref_in": None}


def migrated(obj: Optional[Dict[str, Any]]) -> bool:
    return isinstance((obj or {}).get("band"), dict)


def shared(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The object band as stored (an object of the group's main object)."""
    b = (obj or {}).get("band") or {}
    return {"band_points": list(b.get("band_points") or []), "band_ref_out": b.get("band_ref_out"),
            "band_ref_in": b.get("band_ref_in")}


def sliders_migrated(obj: Optional[Dict[str, Any]]) -> bool:
    """The object band carries the sliders (the second migration, 2026-10-24); a part: its group's main object."""
    return migrated(obj) and "out_px" in ((obj or {}).get("band") or {})


def shared_sliders(obj: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """The four sliders of the object band as stored (None = BAND_DEFAULTS)."""
    b = (obj or {}).get("band") or {}
    return {k: (None if b.get(k) is None else float(b[k])) for k in SLIDERS}


def _own_slider_values(obj: Optional[Dict[str, Any]], key: str, zone: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float]]:
    """The stage's own slider values (None where unset)."""
    o = obj or {}
    if key.startswith("zone:"):
        p = (zone or {}).get("params") or {}
        ins = p.get("outline_in_px")
        if ins is None:   # the zone's inset = its trimap inset (engine/zones.outline_in_px, the same rule)
            from engine.bandsys import BAND_DEFAULTS
            fg, fc = p.get("fg_erode_px"), p.get("force_core_erode_px")
            ins = None if fg is None and fc is None else max(float(p.get("fg_erode_px", BAND_DEFAULTS["in_px"]) or 0), float(fc or 0))
        return {"out_px": p.get("outline_out_px"), "in_px": ins, "inset_feather_px": p.get("inset_feather_px"),
                "outer_feather_px": p.get("outer_feather_px")}
    src = o if key == "matanyone" else (o.get(key) or {})
    f = OWN_SLIDER_FIELDS[key]
    return {k: (None if src.get(f[k]) is None else float(src[f[k]])) for k in SLIDERS}


def sliders(obj: Optional[Dict[str, Any]], key: str, root: Optional[Dict[str, Any]] = None,
            zone: Optional[Dict[str, Any]] = None) -> Tuple[float, float, float, float]:
    """(out, in, inset feather, outer feather) the owner `key` works with – `band_for(...).sliders`."""
    return band_for(obj, key, root=root, zone=zone).sliders


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


# ----------------------------------------------------------------------------- the one band settings model
# the zone's own field names of the four sliders (its inset = the ViTMatte trimap inset: fg_erode_px / force_core_erode_px)
ZONE_SLIDER_FIELDS = {"out_px": "outline_out_px", "in_px": "fg_erode_px", "inset_feather_px": "inset_feather_px",
                      "outer_feather_px": "outer_feather_px"}
ZONE_SLIDER_KEYS = ("outline_out_px", "fg_erode_px", "force_core_erode_px", "inset_feather_px", "outer_feather_px", "outline_in_px")
INT_SLIDERS = {"ltx": ("in_px",), "zone": ("in_px",)}   # fields stored as ints (LTX core_erode_px, the zone's trimap inset)
# „Wcięcie: cały rdzeń” (2026-10-29): the band covers the whole core (engine/bandsys `whole`); the shared band and the
# zone params store it as `whole_core`, stage 3 on the object as `band_whole_core`, stages 4 / 7 in their dict
WHOLE_FIELD = {"matanyone": "band_whole_core"}


class BandSettings(dict):
    """What a band owner works with: the four sliders (effective px), the points with the sliders' reference they were
    set against, the tracking cache owner id, shared / detached and the owner key. Built by `band_for` only. A dict
    with attribute access (so a settings dict that embeds it stays JSON – run.json keeps it as `band`):
      out_px, in_px, inset_feather_px, outer_feather_px   the sliders (effective px)
      band_points (normalized, engine/bandmap.band_points), band_ref_out, band_ref_in
      owner_id   ZoneCache owner of the points' tracking (SHARED_OWNER | stage_<key> | <zone_id>)
      shared     the shared object band (tracked once per group on its main object)
      key        matanyone | videomama | ltx | zone:<id>
      active     False = the stage has no band now (LTX-2.5 „surowy”): no points, no lines
      whole_core „Wcięcie: cały rdzeń” – the band covers the whole core (`in_w` = inf for the geometry; in_px stays the
                 slider value)"""
    FIELDS = ("out_px", "in_px", "inset_feather_px", "outer_feather_px", "band_points", "band_ref_out", "band_ref_in",
              "owner_id", "shared", "key", "active", "whole_core")

    def __init__(self, out_px: float, in_px: float, inset_feather_px: float, outer_feather_px: float,
                 band_points: Optional[List[Dict[str, Any]]] = None, band_ref_out: Optional[float] = None,
                 band_ref_in: Optional[float] = None, owner_id: str = SHARED_OWNER, shared: bool = True, key: str = "",
                 active: bool = True, whole_core: bool = False):
        super().__init__(out_px=float(out_px), in_px=float(in_px), inset_feather_px=float(inset_feather_px),
                         outer_feather_px=float(outer_feather_px), band_points=list(band_points or []),
                         band_ref_out=band_ref_out, band_ref_in=band_ref_in, owner_id=owner_id, shared=bool(shared),
                         key=key, active=bool(active), whole_core=bool(whole_core))

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    @property
    def sliders(self) -> Tuple[float, float, float, float]:
        return self.out_px, self.in_px, self.inset_feather_px, self.outer_feather_px

    @property
    def in_w(self) -> float:
        """The inset width for the band geometry (engine/bandsys.band_widths): inf with „cały rdzeń”."""
        return float("inf") if self.get("whole_core") else self.in_px

    def slider_dict(self) -> Dict[str, float]:
        return dict(zip(SLIDERS, self.sliders))

    def points_params(self) -> Dict[str, Any]:
        """The points as a params dict (band_points / band_ref_out / band_ref_in)."""
        return {"band_points": [dict(p) for p in self.band_points], "band_ref_out": self.band_ref_out,
                "band_ref_in": self.band_ref_in}

    @property
    def scales(self) -> Tuple[float, float]:
        return scales(self)

    def legacy(self) -> Dict[str, Any]:
        return legacy_fields(self.key, self)


def scales(s: BandSettings) -> Tuple[float, float]:
    """(outer, inner) scale of the points' widths: the sliders now over their values when the points were set
    (user decision 2026-10-21: the sliders size the whole band, the points shape it); no reference = 1. The ONE
    slider / band_ref arithmetic."""
    so = float(s.out_px) / float(s.band_ref_out) if s.band_ref_out else 1.0
    si = float(s.in_px) / float(s.band_ref_in) if s.band_ref_in else 1.0
    return so, si


def _kind(key: str) -> str:
    return "zone" if key.startswith("zone:") else key


def slider_fields(key: str) -> Dict[str, str]:
    """canonical slider -> the owner's own field name (the storage of a detached owner / an object not migrated);
    the object band itself (key SHARED_OWNER) has the canonical names."""
    if key.startswith("zone:"):
        return ZONE_SLIDER_FIELDS
    return OWN_SLIDER_FIELDS.get(key) or {k: k for k in SLIDERS}


def canon_sliders(key: str, d: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Slider values given with canonical (SLIDERS) or the owner's legacy names -> canonical; absent / None = None.
    A zone's inset = max(fg_erode_px, force_core_erode_px) (outline_in_px when explicit)."""
    d = d or {}
    out: Dict[str, Optional[float]] = {k: None for k in SLIDERS}
    f = slider_fields(key)
    for k in SLIDERS:
        v = d.get(k)
        if v is None:
            v = d.get(f[k])
        out[k] = None if v is None else float(v)
    if key.startswith("zone:"):
        if d.get("outline_in_px") is not None:
            out["in_px"] = float(d["outline_in_px"])
        elif d.get("fg_erode_px") is not None or d.get("force_core_erode_px") is not None:
            out["in_px"] = max(float(d.get("fg_erode_px") or 0), float(d.get("force_core_erode_px") or 0))
    return out


def legacy_fields(key: str, s: BandSettings) -> Dict[str, Any]:
    """The owner's legacy field names of `s` (the thin mapping old readers get in their settings dict): the four
    sliders under the stage's names (+ `core_erode_px` / `fg_erode_px` + `force_core_erode_px` as ints), the points,
    the reference and (stages) `band_owner`."""
    f = slider_fields(key)
    d: Dict[str, Any] = {f[k]: float(v) for k, v in zip(SLIDERS, s.sliders)}
    for k in INT_SLIDERS.get(_kind(key), ()):
        d[f[k]] = int(round(getattr(s, k)))
    if key.startswith("zone:"):
        d["force_core_erode_px"] = d["fg_erode_px"]
    d.update(s.points_params())
    if not key.startswith("zone:"):
        d["band_owner"] = s.owner_id
    if s.get("whole_core"):   # only when on: results computed before the option keep their signatures
        d[whole_field(key)] = True
    return d


def whole_field(key: str) -> str:
    """The owner's own field of „cały rdzeń” (`whole_core`; stage 3: `band_whole_core` on the object)."""
    return WHOLE_FIELD.get(key, "whole_core")


def _own_whole(obj: Optional[Dict[str, Any]], key: str, zone: Optional[Dict[str, Any]], own: Optional[Dict[str, Any]]) -> bool:
    o = obj or {}
    if own is not None and own.get(whole_field(key)) is not None:
        return bool(own.get(whole_field(key)))
    if key.startswith("zone:"):   # the stage dicts validated by their settings drop unknown keys: read the object
        return bool(((zone or {}).get("params") or {}).get("whole_core"))
    src = o if key == "matanyone" else (o.get(key) or {})
    return bool(src.get(whole_field(key)))


def _zone_dict(zone: Any) -> Optional[Dict[str, Any]]:
    if zone is None or isinstance(zone, dict):
        return zone
    return {"zone_id": getattr(zone, "zone_id", None), "params": getattr(zone, "params", None) or {}}


def band_for(obj: Optional[Dict[str, Any]], key: str, root: Optional[Dict[str, Any]] = None, zone: Any = None,
             overrides: Optional[Dict[str, Any]] = None, own: Optional[Dict[str, Any]] = None) -> BandSettings:
    """THE resolver: the band the owner `key` of `obj` works with. Shared (default) = the object band of `root`
    (default `obj`; a part: its group's main object) – points and, once migrated, sliders; detached („own”, or an object
    not migrated yet) = the owner's own fields (`own`: the stage's validated settings dict with its legacy names, default
    read from `obj`; a zone: `zone` – a Zone or its dict). Unset = engine/bandsys.BAND_DEFAULTS; a zone's outer
    feather unset = its inset feather (the zone's rule; the zone composition's 0.5 px floor of the inset feather stays
    in engine/zones.zone_feather_px). `overrides` (canonical or
    legacy names, None = ignored) = palette values not saved yet, on top in every mode."""
    from engine.bandmap import band_points
    from engine.bandsys import BAND_DEFAULTS
    o = obj or {}
    r = root if root is not None else o
    zd = _zone_dict(zone)
    sh = mode(o, key) == "shared"
    is_zone = key.startswith("zone:")
    if sh and sliders_migrated(r):
        v = shared_sliders(r)
    elif own is not None:
        v = canon_sliders(key, own)
    else:
        v = _own_slider_values(o, key, zd)
    for k, val in canon_sliders(key, overrides).items():
        if val is not None:
            v[k] = val
    out = BAND_DEFAULTS["out_px"] if v["out_px"] is None else float(v["out_px"])
    inn = BAND_DEFAULTS["in_px"] if v["in_px"] is None else float(v["in_px"])
    fi = BAND_DEFAULTS["inset_feather_px"] if v["inset_feather_px"] is None else float(v["inset_feather_px"])
    fo = (fi if is_zone else BAND_DEFAULTS["outer_feather_px"]) if v["outer_feather_px"] is None else float(v["outer_feather_px"])
    if sh:
        p, owner = shared(r), SHARED_OWNER
    elif is_zone:
        p, owner = {k: ((zd or {}).get("params") or {}).get(k) for k in KEYS}, key[5:]
    else:
        p, owner = ({k: own.get(k) for k in KEYS} if own is not None else own_params(o, key)), f"stage_{key}"
    pts = band_points({"band_points": p.get("band_points")}) if p.get("band_points") else []
    if overrides is not None and overrides.get("whole_core") is not None:
        whole = bool(overrides["whole_core"])
    elif sh and migrated(r):
        whole = bool(((r.get("band") or {}).get("whole_core")))
    else:
        whole = _own_whole(o, key, zd, own)
    return BandSettings(out, inn, fi, fo, pts, p.get("band_ref_out"), p.get("band_ref_in"), owner, sh, key,
                        whole_core=whole)


def save_sliders(store: Any, name: str, key: str, values: Optional[Dict[str, Any]], zone: Any = None) -> str:
    """THE save path of band sliders moved in any palette (`values`: canonical or legacy names, None = untouched).
    Shared owner on a migrated band: the object band of the group's main object (every shared stage follows, ↻);
    detached (or not migrated): the owner's own fields – the object's (stage 3), the stage dict (4 / 7, LTX-2.5 inset as
    int) or the zone's params (`zone`, saved). Returns "shared" | "own" | "" (nothing to save). Invalidates the final."""
    vals = {k: v for k, v in canon_sliders(key, values).items() if v is not None}
    if not vals:
        return ""
    obj = store.get(name)
    root = name    # a group member has its own band (PLAN_GROUP_COMPOSE 2026-10-29)
    if mode(obj, key) == "shared" and sliders_migrated(store.get(root)):
        band = dict(store.get(root).get("band") or {})
        band.update({k: float(v) for k, v in vals.items()})
        store.update(root, band=band)
        store.update(root, tool={"final_valid": False})
        return "shared"
    f = slider_fields(key)
    ints = INT_SLIDERS.get(_kind(key), ())
    upd = {f[k]: (int(round(v)) if k in ints else float(v)) for k, v in vals.items()}
    if key.startswith("zone:"):
        if zone is not None:
            from engine.zones import save_zone
            p = dict(zone.params or {})
            v = values or {}
            given = {k: v[k] for k in ZONE_SLIDER_KEYS if v.get(k) is not None}   # the zone's own names: as given
            for k in ("out_px", "inset_feather_px", "outer_feather_px"):            # canonical names (a detach, the API)
                if v.get(k) is not None:
                    given[f[k]] = float(v[k])
            if v.get("in_px") is not None:   # a canonical inset = the trimap inset too, no explicit outline_in_px left
                given["fg_erode_px"] = given["force_core_erode_px"] = v["in_px"]
                p.pop("outline_in_px", None)
            for k in ("fg_erode_px", "force_core_erode_px", "outline_in_px"):
                if k in given:
                    given[k] = int(round(float(given[k])))
            p.update(given)
            # always written: `zone` may carry the EFFECTIVE params (load_zones – the shared values) while its file still
            # holds stale own values; a detach copies the shared sliders into the file (user report 2026-10-24: after
            # „odłącz” the zone showed its old own values instead of the shared ones)
            zone.params = p
            save_zone(store.dir(name), zone)
    elif key == "matanyone":
        store.update(name, **upd)
    else:
        st = dict(obj.get(key) or {})
        st.update(upd)
        store.update(name, **{key: st})
    store.update(name, tool={"final_valid": False})
    return "own"


def save_whole(store: Any, name: str, key: str, on: bool, zone: Any = None) -> str:
    """„Wcięcie: cały rdzeń” of a band owner (2026-10-29), saved like the sliders: the shared object band (every shared
    stage follows, ↻) or the owner's own field (stage 3 on the object, stages 4 / 7 in their dict, a zone's params).
    Returns "shared" | "own". Invalidates the final."""
    obj = store.get(name)
    on = bool(on)
    if mode(obj, key) == "shared" and migrated(obj):
        band = dict(obj.get("band") or {})
        if on:
            band["whole_core"] = True
        else:
            band.pop("whole_core", None)
        store.update(name, band=band, tool={"final_valid": False})
        return "shared"
    f = whole_field(key)
    if key.startswith("zone:"):
        if zone is not None:
            from engine.zones import save_zone
            p = dict(zone.params or {})
            if on:
                p[f] = True
            else:
                p.pop(f, None)
            zone.params = p
            save_zone(store.dir(name), zone)
    elif key == "matanyone":
        store.update(name, **{f: (True if on else None)})
    else:
        st = dict(obj.get(key) or {})
        if on:
            st[f] = True
        else:
            st.pop(f, None)
        store.update(name, **{key: st})
    store.update(name, tool={"final_valid": False})
    return "own"


def zone_params(zone: Dict[str, Any], obj: Optional[Dict[str, Any]], root: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The params of a zone (raw dict) with the band it works with (`band_for`): shared – the band keys replaced by the
    object band of `root` (an empty object band = none, the constant band) and, once the sliders are migrated, the
    sliders of a whole-outline zone under the zone's legacy names (`legacy_fields`: also the ViTMatte trimap inset
    fg_erode_px / force_core_erode_px); detached – the zone's own params as they are. Zone params are JSON (saved in
    the zone file), so the BandSettings itself is not embedded – read it with `band_for(obj, "zone:<id>", root, zone)`."""
    p = dict(zone.get("params") or {})
    key = f"zone:{zone.get('zone_id')}"
    if mode(obj, key) != "shared":
        return p
    b = band_for(obj, key, root=root, zone=zone)
    p = {k: v for k, v in p.items() if k not in KEYS}
    if b.band_points:
        p.update(shared(root))   # the points as stored (the zone signature reads them as the zone's own)
    if sliders_migrated(root) and p.get("region") == "outline":   # the shared sliders (2026-10-24): the band of a
        # whole-outline zone and its trimap inset follow them (a drawn zone has no band: its params stay its own)
        p.pop("outline_in_px", None)
        p.pop("whole_core", None)
        p.update({k: v for k, v in b.legacy().items() if k not in KEYS})
    elif p.get("region") == "outline":   # „cały rdzeń” of the shared band (also before the sliders' migration)
        p.pop("whole_core", None)
        if b.whole_core:
            p["whole_core"] = True
    return p


# ----------------------------------------------------------------------------- migration
def _read(p: Path) -> Optional[Dict[str, Any]]:
    from engine.objects import ObjectStore
    try:
        return ObjectStore._read(p)   # retries a read that collides with another thread's replace (Windows)
    except (OSError, ValueError):
        return None


def _write(p: Path, d: Dict[str, Any]) -> None:
    # the store's writer (retries the Windows replace while another thread reads the file): the migration runs on the
    # first read, so a request and the mask 1+2 background batch migrated the same object at once and the bare replace
    # failed with WinError 5 – the object then looked as if it had no mask (2026-10-26)
    from engine.objects import ObjectStore
    ObjectStore._write(p, d)


def _slider_tuple(obj, key, zone=None):
    """The owner's own slider values with the defaults filled – what its result is computed with today."""
    from engine.bandsys import BAND_DEFAULTS
    v = _own_slider_values(obj, key, zone)
    fi = BAND_DEFAULTS["inset_feather_px"] if v["inset_feather_px"] is None else float(v["inset_feather_px"])
    fo = (fi if key.startswith("zone:") else BAND_DEFAULTS["outer_feather_px"]) if v["outer_feather_px"] is None else float(v["outer_feather_px"])
    return (BAND_DEFAULTS["out_px"] if v["out_px"] is None else float(v["out_px"]),
            BAND_DEFAULTS["in_px"] if v["in_px"] is None else float(v["in_px"]), fi, fo)


def migrate_sliders(root_dir: Path, name: str) -> None:
    """The second migration (user decision 2026-10-24): the sliders join the object band. Every owner on the shared
    band keeps the result it has today – the band gets the sliders of the first shared owner (a whole-outline zone
    first – the pattern of the band system – then stages 3 / 4 / 7); every other shared owner whose own sliders differ
    is detached („own”: a copy of the shared points + its own sliders, nothing changes in its result). Owners already
    detached stay so. Idempotent (`band.out_px` present = done); a part is migrated through its group's main object."""
    od = Path(root_dir) / name
    obj = _read(od / "object.json")
    if obj is None or not migrated(obj):
        return
    main = obj.get("part_of") or name
    md = Path(root_dir) / main
    mobj = _read(md / "object.json") if main != name else obj
    if mobj is None or not migrated(mobj) or "out_px" in (mobj.get("band") or {}):
        return
    members: List[Tuple[str, Path, Dict[str, Any]]] = [(main, md, mobj)]
    for pth in sorted(Path(root_dir).glob("*/object.json")):
        o = _read(pth)
        if o and o.get("part_of") == main and o.get("name") != main:
            members.append((o.get("name") or pth.parent.name, pth.parent, o))
    owners: List[Tuple[str, str, Tuple[float, float, float, float], Optional[Path], Optional[Dict[str, Any]]]] = []
    for mname, mdir, o in members:
        for zp in sorted((mdir / "zones").glob("z_*.json")) if (mdir / "zones").exists() else []:
            z = _read(zp)
            zpar = (z or {}).get("params") or {}
            if z and z.get("criterion") == "matte" and zpar.get("region") == "outline":
                key = f"zone:{z.get('zone_id') or zp.stem}"
                if mode(o, key) == "shared":
                    owners.append((mname, key, _slider_tuple(o, key, z), zp, z))
    for key in STAGES:
        if mode(mobj, key) == "shared":
            owners.append((main, key, _slider_tuple(mobj, key), None, None))
    band = dict(mobj.get("band") or {})

    def explicit(mname, key, z):
        o = next(o for n, _d, o in members if n == mname)
        return any(v is not None for v in _own_slider_values(o, key, z).values())
    lead = next((ow for ow in owners if explicit(ow[0], ow[1], ow[4])), owners[0] if owners else None)
    canon = lead[2] if lead else None
    if canon:
        band.update(out_px=canon[0], in_px=canon[1], inset_feather_px=canon[2], outer_feather_px=canon[3])
    else:
        band.update(out_px=None, in_px=None, inset_feather_px=None, outer_feather_px=None)
    pts = shared(mobj)
    for mname, key, vals, zp, z in owners:
        if vals == canon:
            continue
        # detached: a copy of the shared points (the band it had), its own sliders stay in its own fields
        o = next(o for n, _d, o in members if n == mname)
        o["band_modes"] = {**(o.get("band_modes") or {}), key: "own"}
        if key.startswith("zone:"):
            z["params"] = {**(z.get("params") or {}), **pts}
            _write(zp, z)
        elif key == "matanyone":
            o["edge_band"] = dict(pts) if pts["band_points"] else None
            for k, fld in OWN_SLIDER_FIELDS[key].items():
                if o.get(fld) is None:
                    o[fld] = vals[SLIDERS.index(k)]
        else:
            st = dict(o.get(key) or {})
            st.update(pts)
            for k, fld in OWN_SLIDER_FIELDS[key].items():
                if st.get(fld) is None:
                    st[fld] = vals[SLIDERS.index(k)]
            o[key] = st
    mobj["band"] = band
    for mname, mdir, o in members:
        _write(mdir / "object.json", o)


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
