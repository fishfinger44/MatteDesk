# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MiniMax-H3 anchors of step C1 (user design 2026-11-11): a list of anchors on the timeline, each with the history
of its Qwen-Image-2.1 generations.

Kinds:
- main     – ONE: generated from the shared description; also the whole-clip reference <Picture 1> (shared style)
- linked   – follows the main one: Qwen-Image-2.1 gets its frame + the main anchor's chosen generation as Picture 2
- own      – detached: its own description (a morph step); detaching copies the generation it showed
- original – the plate frame itself (a transition from / to the original look)
- projection – the Cleanup 3D render (step C5, card projection) of its frame, copied in as a generation (no Qwen):
  the projection holds the static geometry, MiniMax-H3 adds what it cannot (waves, water). NOT a transition – the
  whole-clip reference and the shared prompt stay (user request 2026-11-12, sh060 test vs VOID)

Shared style (only main + linked) = one prompt for the anchors and MiniMax-H3, the look held as a state. Any own /
original anchor = a TRANSITION: MiniMax-H3 gets its own clip prompt (`h3_prompt`) and neither the "no change over
time" sentence nor the whole-clip reference (they would block the change).

Storage: <inpaint>/anchors/<id>/g_###.png + g_###.json (prompt, seed, ref, frame, time); the anchor's `gen` names the
chosen generation. Deleting a generation / an anchor removes it from disk for good (user decision 2026-11-13)."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

KINDS = ("main", "linked", "own", "original", "projection")
ID_RE = re.compile(r"^a[0-9a-f]{6}$")
GEN_RE = re.compile(r"^g_\d{3}$")


def new_id() -> str:
    return "a" + uuid.uuid4().hex[:6]


def normalize(items: Any) -> List[Dict[str, Any]]:
    """Valid anchors sorted by frame, one per frame, at most one main (the first main; none = the first linked one)."""
    out, seen = [], set()
    for a in items or []:
        if not isinstance(a, dict):
            continue
        try:
            f = int(a.get("frame"))
        except (TypeError, ValueError):
            continue
        aid = str(a.get("id") or "")
        if not ID_RE.match(aid) or f in seen:
            continue
        seen.add(f)
        kind = a.get("kind") if a.get("kind") in KINDS else "linked"
        gen = a.get("gen") if isinstance(a.get("gen"), str) and GEN_RE.match(a.get("gen")) else None
        out.append({"id": aid, "frame": f, "kind": kind, "prompt": str(a.get("prompt") or "")[:1000], "gen": gen})
    out.sort(key=lambda a: a["frame"])
    mains = [a for a in out if a["kind"] == "main"]
    for a in mains[1:]:
        a["kind"] = "linked"
    if not mains:   # no main: the first linked one becomes main; own / original anchors keep what they are
        first = next((a for a in out if a["kind"] == "linked"), None)
        if first:
            first["kind"] = "main"
    return out


def main_of(anchors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((a for a in anchors if a["kind"] == "main"), None)


def is_transition(anchors: List[Dict[str, Any]]) -> bool:
    return any(a["kind"] in ("own", "original") for a in anchors)


def morphs(s: Dict[str, Any], anchors: List[Dict[str, Any]]) -> bool:
    """„Przejście między kotwicami” (user 2026-11-15): linked anchors with different designs (Qwen draws each anew) –
    the video model morphs from one pinned keyframe to the next instead of holding the main one's design and jumping
    at every pinned frame. Needs the setting and at least two anchors with a linked one."""
    return bool(s.get("anchor_morph")) and len(anchors) > 1 and any(a.get("kind") == "linked" for a in anchors)


def transition_of(s: Dict[str, Any], anchors: List[Dict[str, Any]]) -> bool:
    """The clip is a transition (no single <Picture 1> look for the whole clip): own / original anchors or a morph."""
    return is_transition(anchors) or morphs(s, anchors)


def defaults(main_frame: Optional[int], derived: List[int]) -> List[Dict[str, Any]]:
    if main_frame is None:
        return []
    return normalize([{"id": new_id(), "frame": int(main_frame), "kind": "main"}] +
                     [{"id": new_id(), "frame": int(f), "kind": "linked"} for f in derived if int(f) != int(main_frame)])


# ----------------------------------------------------------------------------- generations
def anchor_dir(root: Path, aid: str) -> Path:
    if not ID_RE.match(aid or ""):
        raise ValueError("zły identyfikator kotwicy")
    return Path(root) / aid


def gens(root: Path, aid: str) -> List[Dict[str, Any]]:
    d = anchor_dir(root, aid)
    out = []
    for png in sorted(d.glob("g_*.png")) if d.exists() else []:
        if not GEN_RE.match(png.stem):
            continue
        try:
            meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        out.append({"name": png.stem, **meta})
    return out


def gen_file(root: Path, aid: str, name: str) -> Path:
    if not GEN_RE.match(name or ""):
        raise ValueError("zła nazwa generacji")
    return anchor_dir(root, aid) / f"{name}.png"


def add_gen(root: Path, aid: str, png_bytes: bytes, meta: Dict[str, Any]) -> str:
    d = anchor_dir(root, aid)
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem[2:]) for p in d.glob("g_*.png") if GEN_RE.match(p.stem)]
    name = f"g_{(max(nums) + 1) if nums else 1:03d}"
    (d / f"{name}.png").write_bytes(png_bytes)
    (d / f"{name}.json").write_text(json.dumps({**meta, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False,
                                               indent=1), encoding="utf-8")
    return name


def delete(src: Path) -> int:
    """Delete a generation (its .png and .json) or an anchor folder from disk for good (user decision 2026-11-13).
    Returns the bytes freed (0 = nothing there)."""
    from engine import deletion as DEL
    src = Path(src)
    if src.is_dir():
        return DEL.remove(src)
    return DEL.remove(src, src.with_suffix(".json"))


def valid_gen(root: Path, a: Dict[str, Any], main: Optional[Dict[str, Any]]) -> Optional[str]:
    """The chosen generation the run may use as is: made for this frame; a linked one also from the main anchor's
    CURRENT generation (the main one changed = the linked follow it again). None = generate."""
    if a["kind"] == "original" or not a.get("gen") or not gen_file(root, a["id"], a["gen"]).exists():
        return None
    meta = next((g for g in gens(root, a["id"]) if g["name"] == a["gen"]), {})
    if int(meta.get("frame", a["frame"])) != a["frame"]:
        return None
    if a["kind"] == "linked" and (not main or meta.get("ref") != f"{main['id']}/{main.get('gen')}"):
        return None
    return a["gen"]


# ----------------------------------------------------------------------------- projection anchors (C5 render)
def projection_frames(work: Path, frames, step: int = 0) -> List[int]:
    """Frames for projection anchors inside `frames`: every frame where an active card key brings a new image
    (engine/cleanup3d/store.list_keys, all cards) + with `step` > 0 every step-th frame of the range (variant „denser”)."""
    from engine.cleanup3d import store
    fs = [int(f) for f in frames]
    have = set(fs)
    out = {k.frame for k in store.list_keys(work) if k.active and int(k.frame) in have}
    if step and step > 0 and fs:
        out |= set(range(fs[0], fs[-1] + 1, int(step))) & have
    return sorted(out)


def projection_version(work: Path) -> Dict[str, Any]:
    """The last Cleanup 3D render ({version, stale, ...}); ValueError when there is none."""
    from engine.cleanup3d import store
    rs = store.render_state(work)
    if not rs:
        raise ValueError("Brak renderu projekcji kart (Czyszczenie 3D, C5) – najpierw „Renderuj”")
    return rs


def projection_png(work: Path, version: str, frame: int, linear: bool = False) -> bytes:
    """The comp of `version` at `frame` as an 8-bit display PNG (what the video model gets)."""
    import cv2
    import numpy as np
    from engine import cleanup as cu
    from engine.cleanup3d.void_engine import to_display8
    d = cu.cleanup_dir(work) / str(version) / "comp"
    f = next((d / f"c_{int(frame):04d}{e}" for e in (".png", ".exr") if (d / f"c_{int(frame):04d}{e}").exists()), None)
    if f is None:
        raise ValueError(f"Render {version} nie ma klatki F{frame}")
    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Nie da się odczytać {f.name}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    ok, buf = cv2.imencode(".png", to_display8(img, linear or f.suffix == ".exr"))
    if not ok:
        raise ValueError("Błąd zapisu PNG")
    return np.asarray(buf).tobytes()


def snapshot_projection(work: Path, root: Path, a: Dict[str, Any], linear: bool = False) -> str:
    """A new generation of projection anchor `a` = the last C5 render of its frame; chosen at once. Returns its name."""
    rs = projection_version(work)
    name = add_gen(root, a["id"], projection_png(work, rs["version"], a["frame"], linear),
                   {"prompt": "", "frame": a["frame"], "kind": "projection", "ref": None,
                    "projection": rs["version"], "stale": bool(rs.get("stale"))})
    a["gen"] = name
    return name
