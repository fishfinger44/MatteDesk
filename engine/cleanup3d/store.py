# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – reading and writing the contract's json files (package P0). Atomic writes through engine/jsonio.py;
a missing file = None / empty, a broken one = ValueError with the file's name (409 in the API). Images of the keys and
brush layers are the key store's job (P2: engine/cleanup3d/keys.py)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.cleanup3d import types as T
from engine.jsonio import write_json


def _read(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Uszkodzony plik {path.name}: {e}")


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data, indent=1, ensure_ascii=False)


def load_camera(work_dir: Path) -> Optional[T.Camera3D]:
    d = _read(T.camera_file(work_dir))
    return None if d is None else T.Camera3D.from_json(d)


def save_camera(work_dir: Path, cam: T.Camera3D) -> None:
    _write(T.camera_file(work_dir), cam.to_json())


def load_cards(work_dir: Path) -> List[T.Card]:
    d = _read(T.cards_file(work_dir))
    cards = [] if d is None else T.cards_from_json(d)
    for c in cards:                                      # a depth card's snapshot is relative to <work>/inpaint/ (P18)
        c.root = str(T.root_dir(work_dir))
    return cards


def save_cards(work_dir: Path, cards: List[T.Card]) -> None:
    T.cards_from_json(T.cards_to_json(cards))            # the same checks a reader will run (duplicate ids, kinds)
    _write(T.cards_file(work_dir), T.cards_to_json(cards))


def load_key(work_dir: Path, card: str, frame: int) -> Optional[T.Key]:
    d = _read(T.key_file(work_dir, card, frame))
    return None if d is None else T.Key.from_json(d)


def save_key(work_dir: Path, key: T.Key) -> None:
    _write(T.key_file(work_dir, key.card, key.frame), key.to_json())


def list_keys(work_dir: Path, card: Optional[str] = None) -> List[T.Key]:
    """Every key on disk (of one card or of all), by card then frame."""
    base = T.root_dir(work_dir) / "keys"
    out: List[T.Key] = []
    for cd in sorted(p for p in base.glob("*") if p.is_dir() and (card is None or p.name == card)):
        for kd in sorted(cd.glob("k_[0-9][0-9][0-9][0-9]")):
            k = load_key(work_dir, cd.name, int(kd.name[2:]))
            if k is not None:
                out.append(k)
    return out


def load_areas(work_dir: Path, card: str, frame: int) -> List[T.Area]:
    d = _read(T.areas_file(work_dir, card, frame))
    return [] if d is None else [T.Area.from_json(a) for a in d.get("areas") or []]


def save_areas(work_dir: Path, card: str, frame: int, areas: List[T.Area]) -> None:
    _write(T.areas_file(work_dir, card, frame), {"schema": T.SCHEMA_AREAS, "card": card, "frame": int(frame),
                                                 "areas": [a.to_json() for a in areas]})


def delete_card(work_dir: Path, card: str) -> int:
    """Remove a card (user request 2026-11-13: there was no way to): its keys (every key with all versions) and the
    brush layers of its extent are deleted from disk for good (user decision 2026-11-13: no trash), then the card leaves
    cards.json. Returns the bytes freed. ValueError (Polish): no such card."""
    from engine import deletion as DEL
    cards = load_cards(work_dir)
    hit = next((c for c in cards if c.id == card), None)
    if hit is None:
        raise ValueError(f"Nie ma karty „{card}”")
    root = T.root_dir(work_dir)
    freed = DEL.remove(root / "keys" / T.check_name(card), extent_dir(work_dir, card), inside=root)
    save_cards(work_dir, [c for c in cards if c.id != card])
    return freed


def extent_dir(work_dir: Path, card: str) -> Path:
    """`<work>/inpaint/cards/<card>/` – the brush layers (`a_<id>.png`) of a card's extent. Not the key folder: a key
    can be deleted on its own and its regeneration areas use the same file names."""
    return T.root_dir(work_dir) / "cards" / T.check_name(card)


# ----------------------------------------------------------------------------- what the projection is made of
def _stat(p: Path) -> str:
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "-"


def scene_stamp(work_dir: Path) -> str:
    """Cheap stamp (file times and sizes, nothing is read) of everything a `render.Scene` is built from: the camera,
    the cards, the extents' brush layers and every key.json. Changes whenever one of them is written – the API's
    Scene and preview caches are keyed by it."""
    root = T.root_dir(work_dir)
    files = [T.camera_file(work_dir), T.cards_file(work_dir)]
    files += sorted(root.glob("keys/*/k_[0-9][0-9][0-9][0-9]/key.json")) + sorted(root.glob("cards/*/a_*.png"))
    return hashlib.sha1("|".join(f"{p.parent.parent.name}/{p.parent.name}/{p.name}={_stat(p)}"
                                 for p in files).encode("utf-8")).hexdigest()


# the Cleanup (C1) settings the render's image depends on (engine/cleanup.py, inpaint/cleanup.json)
SIG_CLEANUP = ("remove", "sources", "grow_px", "soft_px", "band_keys", "holdout", "holdout_skip_lost", "fill", "model")


def render_inputs(work_dir: Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The settings a render's IMAGE depends on besides the camera / cards / keys: the C5 patch blend
    (load_render_settings), the patch band points (inpaint/c3d_band.json) and the Cleanup settings the render uses
    (inpaint/cleanup.json: remove list, mask versions, grow / soft, holdout, fill). `overrides`: values the render
    request set for itself (C5 „Miękkość maski” = soft_px)."""
    from engine import cleanup as cu
    from engine.cleanup3d import band as C3B
    cs = cu.load_settings(work_dir)
    cs.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cs = cu.settings(cs)
    b = C3B.load(work_dir)
    return {"render": load_render_settings(work_dir),
            "band": b if b.get("band_points") else None,
            "cleanup": {k: cs.get(k) for k in SIG_CLEANUP}}


def render_sig(work_dir: Path, overrides: Optional[Dict[str, Any]] = None) -> str:
    """Signature of the CONTENT a projection depends on: the camera file, the cards' geometry (plane, sliders, extent,
    seam, order, warps – not the name or the colour), the active version of every key and (2026-11-10) the render
    settings that change the image (`render_inputs`). A render stores it (`save_render_mark`); a different value
    later = the result is stale (↻). A mark written before the settings were part of it reads as stale once."""
    h = hashlib.sha1()
    h.update(json.dumps(render_inputs(work_dir, overrides), sort_keys=True, default=str).encode("utf-8"))
    cf = T.camera_file(work_dir)
    h.update(cf.read_bytes() if cf.is_file() else b"-")
    for c in load_cards(work_dir):
        d = c.to_json()
        for k in ("name", "color", "key_mode", "key_every", "key_auto"):
            d.pop(k, None)
        h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
        for a in c.extent:
            if a.kind == "brush":
                h.update(_stat(extent_dir(work_dir, c.id) / str(a.png)).encode("utf-8"))
    for k in list_keys(work_dir):
        if k.active:
            h.update(f"{k.card}/{k.frame}/{k.active}".encode("utf-8"))
    return h.hexdigest()


RENDER_SETTINGS = {"key_soft_px": 32.0, "key_grow_px": 16.0, "key_effects": True, "key_effects_thr": 0.10,
                   "clean_first": False, "grain": "auto", "grain_strength": 1.0,
                   # P15 horizon match (engine/cleanup3d/horizon.py DEFAULTS / LIMITS – the same values)
                   "horizon_match": False, "horizon_fade_px": 120.0, "horizon_search_px": 60.0}
RENDER_LIMITS = {"key_soft_px": (0.0, 400.0), "key_grow_px": (0.0, 200.0), "key_effects_thr": (0.01, 1.0),
                 "horizon_fade_px": (8.0, 800.0), "horizon_search_px": (8.0, 400.0), "grain_strength": (0.0, 2.0)}
GRAIN_MODES = ("auto", "off", "on")          # P17 grain match of the patch (engine/cleanup3d/grain.py)


def render_settings_file(work_dir: Path) -> Path:
    return T.root_dir(work_dir) / "c3d_settings.json"


def load_render_settings(work_dir: Path) -> Dict[str, Any]:
    """The Cleanup 3D render settings the user set in C5 (patch blend), defaults for what is missing."""
    try:
        d = _read(render_settings_file(work_dir)) or {}
    except ValueError:
        d = {}
    return clean_render_settings({**RENDER_SETTINGS, **{k: d[k] for k in RENDER_SETTINGS if k in d}})


def clean_render_settings(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(RENDER_SETTINGS)
    for k, v in d.items():
        if k not in RENDER_SETTINGS or v is None:
            continue
        if k in ("key_effects", "clean_first", "horizon_match"):
            out[k] = bool(v)
        elif k == "grain":
            out[k] = v if v in GRAIN_MODES else RENDER_SETTINGS["grain"]
        else:
            lo, hi = RENDER_LIMITS[k]
            out[k] = float(min(hi, max(lo, float(v))))
    return out


def save_render_settings(work_dir: Path, d: Dict[str, Any]) -> Dict[str, Any]:
    s = clean_render_settings({**load_render_settings(work_dir), **d})
    _write(render_settings_file(work_dir), {"schema": "matte_desk/c3d_settings@1.0", **s})
    return s


def render_mark_file(work_dir: Path) -> Path:
    return T.root_dir(work_dir) / "c3d_render.json"


def save_render_mark(work_dir: Path, version: str, overrides: Optional[Dict[str, Any]] = None,
                     sig: Optional[str] = None, skipped: Optional[List[int]] = None) -> Dict[str, Any]:
    """Remember which Cleanup version (`v###`) is the projection of the present camera / cards / keys / settings.
    `overrides`: the request's own settings (kept, the stale check uses them again); `sig`: the signature taken when
    the render started (a change during the render then reads as stale); `skipped`: work-range frames the camera
    does not have."""
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    d: Dict[str, Any] = {"version": str(version), "sig": sig or render_sig(work_dir, ov),
                         "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    if ov:
        d["overrides"] = ov
    if skipped:
        d["skipped"] = [int(f) for f in skipped]
    _write(render_mark_file(work_dir), d)
    return d


def render_state(work_dir: Path) -> Optional[Dict[str, Any]]:
    """{"version", "date", "stale", "warnings", "skipped"} of the last projection, None when there is none (or its
    folder is gone). `warnings`: the kept-object check of its run.json; `skipped`: frames without the 3D camera."""
    try:
        d = _read(render_mark_file(work_dir))
    except ValueError:
        return None
    rj = T.root_dir(work_dir) / str(d.get("version")) / "run.json" if d else None
    if not d or not rj.is_file():
        return None
    try:
        warn = list(((json.loads(rj.read_text(encoding="utf-8")).get("kept_check")) or {}).get("warnings") or [])
    except (OSError, ValueError):
        warn = []
    ov = d.get("overrides") if isinstance(d.get("overrides"), dict) else None
    return {"version": d["version"], "date": d.get("date"), "stale": d.get("sig") != render_sig(work_dir, ov),
            "warnings": warn, "skipped": list(d.get("skipped") or [])}


def state(work_dir: Path) -> Dict[str, Any]:
    """What the UI needs to draw the Cleanup steps (GET /api/cleanup3d): which steps have data, cards and their keys.
    `done.project`: a projection exists; `stale.project`: it is older than the camera / cards / keys it was made of."""
    cam = load_camera(work_dir)
    cards = load_cards(work_dir)
    keys = list_keys(work_dir)
    by_card: Dict[str, List[Dict[str, Any]]] = {}
    for k in keys:
        said = [str(e["prompt"]) for e in k.versions if e.get("prompt")]     # the last description used on this key
        by_card.setdefault(k.card, []).append({"frame": k.frame, "active": k.active, "versions": len(k.versions),
                                               "prompt": said[-1] if said else "",
                                               # the version list of the C4 pane / the „Z wersji:” select (2026-11-10)
                                               "list": [{f: e[f] for f in ("v", "source", "base", "prompt", "date",
                                                                           "area", "strength") if e.get(f) is not None}
                                                        for e in k.versions],
                                               "master": k.canvas is not None})   # own canvas = the master key (P13)
    rs = render_state(work_dir)
    return {
        "steps": list(T.STEPS),
        "done": {"camera": cam is not None, "cards": bool(cards), "keys": any(k.active for k in keys),
                 "project": rs is not None},
        "stale": {"project": bool(rs and rs["stale"])},
        "render": rs,
        "render_settings": load_render_settings(work_dir),
        "camera": None if cam is None else {"frames": [cam.frames[0], cam.frames[-1]] if cam.frames else None,
                                            "n_frames": len(cam.frames), "source": cam.source, "report": cam.report},
        "cards": [{**c.to_json(), "keys": by_card.get(c.id, [])} for c in cards],
        "engines": [{"id": e, "label": T.ENGINE_LABEL[e]} for e in T.ENGINES],
        "key_modes": list(T.KEY_MODES), "views": list(T.VIEWS), "key_formats": list(T.KEY_FORMATS),
    }
