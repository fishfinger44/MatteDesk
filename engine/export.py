# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/export.py - Export (SPEC 9, 16): alpha only, full frames assembled from crop segments.

- `export/mattes/<obj>/<obj>_####.exr` – full frame, rgb = a, half, ZIP, plate frame numbers; or `<obj>_####.png`
  (8-bit) – format = project.json `export_format` ("auto" = PNG for 8-bit sources, EXR otherwise; "png"; "exr");
  source of truth per object.
- optional `export/mattes/all/<shot>_mattes_####.exr` – channels `<obj>.alpha`, `union.alpha` (+ `depth.disparity`);
  only when the format is EXR.
- several objects: ML mattes -> MatteSubtractFront in depth_order (front first): back = clamp(back - max(front)).
  Never `max` on contacts.
- a group (engine/groups.py, PLAN_GROUP_COMPOSE 2026-10-29) is one layer at its front-most member: its whole result,
  its Final Touch or the composition of its members (`seamless_union` of their „Do grupy z:” alphas, contact seams
  filled); it never cuts / is cut by its own members, which are exported with their own choice. (Physical mattes split
  by regions = backlit objects, MatteSplitByRegions in Comfy.)
- optional core / confidence (`<obj>_core/`, `<obj>_conf/`) for the 3D tool (visual hull).
- outputs.json updated (objects: name, depth_order, paths, run version, QA status, date, params hash).
No RGB is ever written (Z3).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.exr_io import write_gray_exr
from engine.outputs import load_outputs, remove_outputs, update_outputs
from engine.refine import assemble_full, read_alpha

logger = logging.getLogger(__name__)

CORE_RUN = "core"     # export source "rdzeń (1+2)" (2026-10-15): the mask 1+2 of core_fix/ instead of a run version

EXPORT_SOURCES = ("active", "core", "matanyone", "videomama", "vitmatte", "ltx", "version", "final_touch")
EXPORT_SOURCE_LABEL = {"active": "aktywna wersja", "core": "rdzeń (1+2)", "matanyone": "MatAnyone2 (etap 3)",
                       "videomama": "VideoMaMa (etap 4)", "vitmatte": "ViTMatte (etap 5)", "ltx": "LTX-2.5 Alpha-Gen (etap 7)",
                       "version": "wybrana wersja", "final_touch": "Final Touch (baza + korekty)"}
# when an object of a full export / of the context has no result of the chosen source (e.g. a hand without ViTMatte
# zones): the next source down, reported in the warning – a written object chosen by name never falls back
EXPORT_FALLBACK = {"vitmatte": ["vitmatte", "videomama", "matanyone", "active", "core"],
                   "matanyone": ["matanyone", "active", "core"],
                   "videomama": ["videomama", "matanyone", "active", "core"],
                   "ltx": ["ltx", "vitmatte", "videomama", "matanyone", "active", "core"],
                   "core": ["core"], "version": ["version", "active"], "active": ["active"],
                   # an object without its own corrections: base = the result of the last computed stage
                   "final_touch": ["final_touch"]}


def core_fix_holes(obj_dir: Path, frames: Sequence[int]) -> List[int]:
    """Frames of `frames` without the mask 1+2 in a core_fix/ that has other frames: they fall back on the SAM 3.1 core,
    without the corrections. No core_fix/ at all = no corrections to lose = no holes (audit 2026-11-02 #6)."""
    d = Path(obj_dir) / "core_fix"
    if not d.is_dir() or next(d.glob("c_*.png"), None) is None:
        return []
    return [int(f) for f in frames if not (d / f"c_{int(f):04d}.png").exists()]


def core_fix_warning(outputs: Dict[str, Any]) -> str:
    """The export's warning about mixed frames (`core_fix_fallback` of export_objects), "" when none."""
    holes = outputs.get("core_fix_fallback") or {}
    if not holes:
        return ""
    return "Klatki bez maski 1+2 (rdzeń SAM 3.1 bez poprawek – „Zastosuj poprawki” na całym zakresie): " + "; ".join(
        f"{n}: " + ", ".join(f"F{a}–{b}" for a, b in iv) for n, iv in sorted(holes.items()))


def core_fix_loader(obj_dir: Path, w: int, h: int) -> Callable[[int], np.ndarray]:
    """Full-frame 0 / 1 alpha of the mask 1+2 (core_fix/c_####.png, stage 2); a frame without it -> the SAM 3.1 core:
    logged once per loader here, reported by the export (core_fix_holes)."""
    warned = [False]

    def load(f: int) -> np.ndarray:
        for sub in ("core_fix", "core"):
            p = Path(obj_dir) / sub / f"c_{f:04d}.png"
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None   # no OpenCV warning per frame in the log
            if m is not None:
                if sub == "core" and not warned[0] and core_fix_holes(obj_dir, [f]):
                    warned[0] = True
                    logger.warning("%s: F%d bez maski 1+2 – rdzeń SAM 3.1 bez poprawek", Path(obj_dir).name, f)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.float32)
        return np.zeros((h, w), np.float32)
    return load


def run_alpha_loader(obj_dir: Path, run: str, w: int, h: int, sub: str = "alpha",
                     reader: Optional[Callable[[Path], np.ndarray]] = None) -> Callable[[int], np.ndarray]:
    """Full-frame alpha of a run version (crop segments pasted back, overlaps crossfaded). `sub="alpha_base"`: the
    version's base alpha (edge mode without zones, written when the version has zones). `run == CORE_RUN`: the
    mask 1+2 (core_fix_loader). `run == "ft:<base>"` (stage Final Touch): the base version + the object's hand
    corrections (engine/final_touch.py). `reader` (default read_alpha): how one segment's EXR is read – „Zastosuj”
    passes a caching one, so its overlay does not decode the same base file again (A6b, 2026-10-28)."""
    from engine.final_touch import base_run, ft_loader, is_ft
    if is_ft(run):
        return ft_loader(obj_dir, run_alpha_loader(obj_dir, base_run(run), w, h, sub, reader))
    if run == CORE_RUN:
        if _is_group_dir(obj_dir):   # a group entity: "core" = its composition (engine/groups, PLAN_GROUP_COMPOSE)
            from engine.groups import group_alpha_loader
            from engine.objects import ObjectStore
            return group_alpha_loader(ObjectStore(Path(obj_dir).parent.parent), Path(obj_dir).name, w, h)
        return core_fix_loader(obj_dir, w, h)
    rd = obj_dir / "runs" / run
    rj = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    segs = [CropSegment.from_dict(d) for d in rj["segments_def"]]

    def load(f: int) -> np.ndarray:
        cover = [s for s in segs if s.first <= f <= s.last]
        per = {}
        for s in cover:
            p = rd / sub / s.seg_id / f"a_{f:04d}.exr"
            if p.exists():
                per[s.seg_id] = {f: (reader or read_alpha)(p)}
        return assemble_full(per, [s for s in cover if s.seg_id in per], [f], w, h)[f]
    return load


def _is_group_dir(obj_dir: Path) -> bool:
    try:
        return json.loads((Path(obj_dir) / "object.json").read_text(encoding="utf-8")).get("kind") == "group"
    except (OSError, ValueError):
        return False


EXPORT_FORMATS = ("auto", "png", "exr")


def resolve_export_format(fmt: Optional[str], bit_depth: int) -> str:
    """'auto' (or empty) -> 'png' for 8-bit sources, 'exr' otherwise; 'png' / 'exr' as given."""
    f = (fmt or "auto").lower()
    if f not in EXPORT_FORMATS:
        raise ValueError(f"export_format musi być jednym z {EXPORT_FORMATS}, jest {fmt!r}")
    if f == "auto":
        return "png" if int(bit_depth) <= 8 else "exr"
    return f


SEAM_PX_DEFAULT = 4    # contact gap between the members' cores filled in a group's composition (group field `seam_px`)
SEAM_PX_MAX = 40


def seam_px(obj: Optional[Dict[str, Any]]) -> int:
    """Seam width of a group's composition: its `seam_px` (0..SEAM_PX_MAX), default SEAM_PX_DEFAULT."""
    try:
        v = int(round(float((obj or {}).get("seam_px", SEAM_PX_DEFAULT))))
    except (TypeError, ValueError):
        v = SEAM_PX_DEFAULT
    return max(0, min(SEAM_PX_MAX, v))


def seamless_union(alphas: Sequence[np.ndarray], cores: Sequence[Optional[np.ndarray]], gap_px: int = 4,
                    gaps: Optional[Sequence[int]] = None) -> np.ndarray:
    """Composition of a group: max of the members' alphas, with the contact seams between the members' cores filled
    to 1. The outer contour is untouched (filling only where two members meet). `gaps` (one per core, user decision
    2026-10-29): each member's seam reach – hair small, so the light gaps by the face stay open, a dress large, so the
    slit at the contact closes; the pair i, j bridges where d_i + d_j <= g_i + g_j + 1, closed with max(g). None = every
    member `gap_px` (bit for bit the single seam)."""
    u = np.max(np.stack([a.astype(np.float32) for a in alphas]), axis=0)
    gl = list(gaps) if gaps is not None else [gap_px] * len(cores)
    pairs = [(c.astype(bool), max(0, int(g))) for c, g in zip(cores, gl) if c is not None]
    cs = [c for c, _ in pairs]
    gs = [g for _, g in pairs]
    if len(cs) < 2 or max(gs) <= 0:
        return u
    gmax = max(gs)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gmax + 1, 2 * gmax + 1))
    # the bridge between two parts: d_i + d_j <= g_i + g_j + 1 (distance to each part's core) – the WHOLE gap up to
    # ~g_i + g_j wide (was: only pixels within gap_px of both = a thin strip in the middle, holes along both edges)
    dist = [cv2.distanceTransform((~c).astype(np.uint8), cv2.DIST_L2, 5) for c in cs]
    contact = np.zeros_like(cs[0])
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            if gs[i] + gs[j] > 0:
                contact |= (dist[i] + dist[j]) <= gs[i] + gs[j] + 1
    closed = cv2.morphologyEx(np.logical_or.reduce(cs).astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    u[closed & contact] = 1.0
    return u


def subtract_front(alpha: np.ndarray, fronts: Sequence[np.ndarray]) -> np.ndarray:
    """Visible part of a layer behind `fronts`: clamp(alpha − max(fronts)) – as MatteSubtractFront (spec 5.2). Not the
    running difference: the export context holds a group AND its members, so a soft edge was subtracted twice
    (audit 2026-11-02 #2)."""
    out = alpha.astype(np.float32, copy=True)
    if not fronts:
        return out
    top = np.maximum.reduce([np.asarray(fr, np.float32) for fr in fronts])
    return np.clip(out - top, 0.0, 1.0)


def params_hash(obj: Dict[str, Any], run_json: Dict[str, Any]) -> str:
    keys = ("edge_mode", "band_scale", "shift_px", "temporal_smoothing", "ma_scale")
    o = {k: obj.get(k) for k in keys}
    o.update({k: obj[k] for k in ("band_out_px", "band_in_px") if obj.get(k) is not None})   # px mode only when set
    if obj.get("ma_scale_mode", "auto") != "auto":   # a chosen MatAnyone scale -> new hash (auto keeps old hashes)
        o["ma_scale_mode"] = obj["ma_scale_mode"]
    blob = json.dumps({"obj": o, "run": run_json.get("params"),
                       "segments": run_json.get("segments_def")}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


class FrameAlphas:
    """Per-frame mattes exactly as `export_objects` writes them (plan step 7, faithful preview): front-to-back
    subtraction by depth_order; a group entity is one layer at its front-most member that never cuts / is cut by its
    own members. `objects` = the context (every object / group with a result). The export and the stage-4/5 preview
    (`frame?view=alpha&export=1`, full-res tiles) share this code, so what the viewer shows is the exported alpha."""

    def __init__(self, work_dir: Path, objects: Sequence[Dict[str, Any]], w: int, h: int):
        self.work_dir = Path(work_dir)
        self.objs = sorted((self._group_layer(o) for o in objects), key=lambda o: o.get("depth_order", 0))
        self.loaders = {o["name"]: run_alpha_loader(self.work_dir / "objects" / o["name"], o["tool"]["active_run"], w, h)
                        for o in self.objs}
        # object life range (engine/workarea.py): outside it the object's matte is empty (black frames, nothing
        # subtracted from the objects behind)
        from engine.workarea import object_life
        self.life = {o["name"]: object_life(o) for o in self.objs}
        self.w, self.h = w, h
        self.fronts: Dict[str, Optional[np.ndarray]] = {}   # of the last frame(): max of the raw layers each one is cut by

    def _group_layer(self, o: Dict[str, Any]) -> Dict[str, Any]:
        """A group entity (engine/groups.py) in the context: its depth = its front-most member (copy of the dict)."""
        if o.get("kind") != "group":
            return o
        from engine.groups import group_depth
        from engine.objects import ObjectStore
        d = group_depth(ObjectStore(self.work_dir), o["name"])
        return {**o, "depth_order": d if d is not None else o.get("depth_order", 0)}

    def _excluded(self, o: Dict[str, Any]) -> set:
        """Layers `o` is never subtracted by: a group's own members, a member's own group."""
        if o.get("kind") == "group":
            return set(o.get("members") or [])
        return {o["group"]} if o.get("group") else set()

    def visible(self, name: str, f: int) -> bool:
        from engine.workarea import is_visible
        return is_visible(f, self.life.get(name))

    def core_fix_holes(self, frames: Sequence[int]) -> Dict[str, List[List[int]]]:
        """{object: intervals} of `frames` where a layer of the context reads the SAM 3.1 core in a place of a missing mask
        1+2 (a group: through its members with the source „core”); frames outside the object's life do not count."""
        from engine.final_touch import base_run
        from engine.workarea import intervals_from_frames, is_visible, object_life
        store = None
        out: Dict[str, List[List[int]]] = {}
        for o in self.objs:
            if base_run(o["tool"]["active_run"]) != CORE_RUN:
                continue
            if o.get("kind") == "group":
                from engine.groups import active_members, member_run
                from engine.objects import ObjectStore
                store = store or ObjectStore(self.work_dir)
                ms = [(m, object_life(store.get(m))) for m in active_members(store, o["name"])
                      if base_run(member_run(store, m)["run"] or "") == CORE_RUN]
            else:
                ms = [(o["name"], self.life.get(o["name"]))]
            for m, life in ms:
                miss = core_fix_holes(self.work_dir / "objects" / m, [f for f in frames if is_visible(f, life)])
                if miss:
                    out[m] = intervals_from_frames(miss)
        return out

    def has_front(self, name: str) -> bool:
        """Whether any layer of the context is in front of `name` and cuts it (export.subtract_front)."""
        o = next(x for x in self.objs if x["name"] == name)
        skip = self._excluded(o)
        return any(x["name"] not in skip for x in self.objs[:self.objs.index(o)])

    def run_of(self, name: str) -> str:
        return next(x for x in self.objs if x["name"] == name)["tool"]["active_run"]

    def frame(self, f: int) -> Dict[str, np.ndarray]:
        """{name: alpha} for every object / group of the context – the exported mattes of f."""
        alphas: Dict[str, np.ndarray] = {}
        front_layers: List[tuple] = []         # (name, raw alpha) of every layer in front so far
        for o in self.objs:
            raw = self.loaders[o["name"]](f) if self.visible(o["name"], f) else np.zeros((self.h, self.w), np.float32)
            skip = self._excluded(o)
            fr = [a_ for n_, a_ in front_layers if n_ not in skip]
            alphas[o["name"]] = subtract_front(raw, fr)
            self.fronts[o["name"]] = np.maximum.reduce([np.asarray(a_, np.float32) for a_ in fr]) if fr else None
            front_layers.append((o["name"], raw))
        return alphas


def frame_alphas(work_dir: Path, objects: Sequence[Dict[str, Any]], f: int, w: int, h: int) -> Dict[str, np.ndarray]:
    """One frame of the export without writing anything (see FrameAlphas)."""
    return FrameAlphas(work_dir, objects, w, h).frame(f)


def export_workers() -> int:
    """Processes of the export: config `export_workers` (0 / missing = auto: cores - 4, at most 8 – the frames go to
    one disk, more processes only queue on it), capped by the free RAM in run_frames. 1 = in this process.
    Env MD_EXPORT_WORKERS overrides (the tests run in this process)."""
    import os
    if os.environ.get("MD_EXPORT_WORKERS", "").strip().isdigit():
        return max(1, int(os.environ["MD_EXPORT_WORKERS"]))
    try:
        from engine.config import load_config
        n = int(load_config().get("export_workers") or 0)
    except Exception:
        n = 0
    return n if n > 0 else max(1, min(8, (os.cpu_count() or 4) - 4))


def _export_frame(job: Dict[str, Any]) -> Dict[str, Any]:
    """One frame of export_objects (a process-pool job; plain values in, written paths out): the mattes of every
    written object, the Final Touch base / front sequences, the core PNGs and the multi-EXR of frame job["f"]."""
    f, w, h = job["f"], job["w"], job["h"]
    work_dir, out_root, png = Path(job["work_dir"]), Path(job["out_root"]), job["png"]
    want = None if job["want"] is None else set(job["want"])
    fa = FrameAlphas(work_dir, job["objects"], w, h)

    def write_matte(name: str, a: np.ndarray, run: str) -> str:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        if png:
            cv2.imwrite(str(d / f"{name}_{f:04d}.png"), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        else:
            write_gray_exr(d / f"{name}_{f:04d}.exr", a, half=True, channels=("R", "G", "B"),
                           extra_attrs={"matteDesk/object": name.encode(), "matteDesk/run": run.encode()})
        return str(d / (f"{name}_####." + ("png" if png else "exr"))).replace("\\", "/")

    out: Dict[str, Any] = {"written": {}, "ft": {}, "multi": None, "depth": False}
    alphas = fa.frame(f)
    if job["ftn"]:
        from engine import final_touch as _ft
        for n_, sp in job["ftn"].items():
            od = work_dir / "objects" / n_
            run = fa.run_of(n_)
            load = _ft.partial_loader(od, run_alpha_loader(od, _ft.base_run(run), w, h), sp["mode"] == "grade")
            raw = load(f) if fa.visible(n_, f) else np.zeros((h, w), np.float32)
            paths = {"base_matte": write_matte(f"{n_}_ftbase", raw, "ft-base")}
            if sp["front"]:
                fr = fa.fronts.get(n_)
                paths["front_matte"] = write_matte(f"{n_}_front", fr if fr is not None else np.zeros((h, w), np.float32), "front")
            out["ft"][n_] = paths
    for o in fa.objs:
        if want is not None and o["name"] not in want:
            continue
        out["written"][o["name"]] = write_matte(o["name"], alphas[o["name"]], o["tool"]["active_run"])
        if job["with_core"]:
            cp = work_dir / "objects" / o["name"] / "core" / f"c_{f:04d}.png"
            if cp.exists():
                cd = out_root / f"{o['name']}_core"
                cd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cd / f"{o['name']}_core_{f:04d}.png"), cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE))
    if job["multichannel"] and not png:
        import Imath
        import OpenEXR
        if want is not None:   # the palette's selection: its mattes only (the context objects cut, not written)
            alphas = {n: a for n, a in alphas.items() if n in want}
        union = np.clip(sum(alphas.values()), 0.0, 1.0) if alphas else np.zeros((h, w), np.float32)
        hdr = OpenEXR.Header(w, h)
        half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        chans = {f"{n}.alpha": a for n, a in alphas.items()}
        chans["union.alpha"] = union
        out["multi"] = list(alphas) + ["union"]
        if job["depth_pattern"]:
            from engine.depth import frame_path, read_depth
            dp = frame_path(job["depth_pattern"], f)
            if dp.exists():
                chans["depth.disparity"] = read_depth(dp)
                out["depth"] = True
        hdr["channels"] = {k: (Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) if k.startswith("depth.") else half)
                           for k in chans}
        hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        md = out_root / "all"
        md.mkdir(parents=True, exist_ok=True)
        o_ = OpenEXR.OutputFile(str(md / f"{job['shot']}_mattes_{f:04d}.exr"), hdr)
        o_.writePixels({k: np.ascontiguousarray(v, np.float32 if k.startswith("depth.") else np.float16).tobytes()
                        for k, v in chans.items()})
        o_.close()
    return out


def export_objects(work_dir: Path, shot: str, objects: Sequence[Dict[str, Any]], frames: Sequence[int], w: int, h: int,
                   bit_depth: int = 16, multichannel: bool = True, depth_pattern: Optional[str] = None,
                   with_core: bool = False, progress_cb: Optional[Callable[[str, int, int], None]] = None,
                   write: Optional[Sequence[str]] = None, export_dir: Optional[Path] = None,
                   export_format: str = "auto", selection: bool = False,
                   merge_plate: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    """Export objects (each must have an active run). Front-to-back subtraction by depth_order.
    `objects` = the context (all objects / groups with a result) – mattes are always computed with every front layer,
    so a single-object export equals the same object in a full export. `write` = names to write (objects / groups);
    None = everything + the multi-EXR.
    `export_format` auto / png / exr (resolve_export_format with `bit_depth`); the multi-EXR only for exr.
    `merge_plate` = (beauty pattern, colorspace) of the manifest: objects with MergeMatte on get the backdrop MERGEMATTE
    in the .nk (their Final Touch settings in the group's knobs; PLAN_FINAL_TOUCH §11)."""
    want = None if write is None else set(write)
    # a partial export never overwrites the shot's multi-EXR; `selection` = `write` is what the palette Obiekty has checked
    # (the full export of the shot, PLAN_GROUP_COMPOSE): the multi-EXR is written, with the written mattes as channels
    multichannel = multichannel and (want is None or selection)
    work_dir = Path(work_dir)
    export_dir = Path(export_dir) if export_dir else work_dir / "export"   # Z29: chosen in project.json -> export_dir
    out_root = export_dir / "mattes"
    fa = FrameAlphas(work_dir, objects, w, h)
    objs = fa.objs
    fmt = resolve_export_format(export_format, bit_depth)
    png = fmt == "png"
    written: Dict[str, str] = {}

    # Final Touch as Nuke nodes (engine/ft_nuke, user decision 2026-11-16): the objects written from Final Touch whose
    # nuke_mode is grade / paint get their base sequence (raw, not cut) + the max of their front layers; their matte
    # stays the baked one (the multi-EXR / union unchanged), the .nk rebuilds it from the base with live nodes
    from engine import final_touch as _ft
    from engine.ft_nuke import available as ft_available
    ftn: Dict[str, Dict[str, Any]] = {}
    for o in objs:
        run = o["tool"]["active_run"]
        if (want is not None and o["name"] not in want) or not _ft.is_ft(run):
            continue
        od = work_dir / "objects" / o["name"]
        d = _ft.load(od)
        if d["nuke_mode"] == "bake":
            continue
        ok, why = ft_available(d.get("finish") or {})
        if not ok:
            ftn[o["name"]] = {"mode": "bake", "requested": d["nuke_mode"], "why": why}
            continue
        ftn[o["name"]] = {"mode": d["nuke_mode"], "front": fa.has_front(o["name"])}
    # frames are independent: written in a process pool (engine/procpool; 2026-11-20 – one by one it was ~2.2 s a
    # frame on a 6K plate, the EXR decode / ZIP encode hold the GIL so threads gave only 1.6x). Each job rebuilds its
    # own FrameAlphas from these plain values, so a frame is written bit for bit as in this process (workers=1).
    ctx = {"work_dir": str(work_dir), "objects": list(objects), "w": int(w), "h": int(h), "out_root": str(out_root),
           "png": png, "want": None if want is None else sorted(want), "with_core": bool(with_core),
           "multichannel": bool(multichannel), "shot": shot, "depth_pattern": depth_pattern,
           "ftn": {n: {"mode": sp["mode"], "front": sp["front"]} for n, sp in ftn.items() if sp["mode"] != "bake"}}
    from engine.procpool import run_frames
    n_layers = len(objs) + 2 * len(ctx["ftn"])
    per_worker = int(w) * int(h) * 4 * (3 * n_layers + 6)      # the float32 layers of one frame + copies / union
    res = run_frames(_export_frame, [{**ctx, "f": int(f)} for f in frames], workers=export_workers(),
                     on_done=(lambda n, _i: progress_cb("Eksport", n, len(frames))) if progress_cb else None,
                     per_worker_bytes=per_worker)
    multi_channels: List[str] = []
    for r in res:                                               # frame order: the same as the former serial loop
        written.update(r["written"])
        for n_, paths in r["ft"].items():
            ftn[n_].update(paths)
        if r["multi"] and not multi_channels:
            multi_channels = list(r["multi"])
        if r.get("depth") and multi_channels and "depth" not in multi_channels:
            multi_channels.append("depth")
    # outputs.json (data contract)
    holes = fa.core_fix_holes(frames)
    entries = []
    for o in objs:
        if o["name"] not in written:
            continue
        from engine.final_touch import base_run, corrections_sig, is_ft, load as ft_load
        ver = o["tool"]["active_run"]
        rd = work_dir / "objects" / o["name"] / "runs" / base_run(ver)
        if base_run(ver) == CORE_RUN:   # mask 1+2: no run.json / QA of its own
            cf = work_dir / "objects" / o["name"] / "core_fix" / "core_fix.json"
            rj = {"edge_mode": "core", "params": json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else None}
            qa = {}
        else:
            rj = json.loads((rd / "run.json").read_text(encoding="utf-8"))
            qa = json.loads((rd / "qa.json").read_text(encoding="utf-8")) if (rd / "qa.json").exists() else {}
        entries.append({"name": o["name"], "depth_order": o.get("depth_order", 0), "matte": written[o["name"]],
                        "core": str(out_root / f"{o['name']}_core" / f"{o['name']}_core_####.png").replace("\\", "/") if with_core else None,
                        "version": o["tool"]["active_run"], "qa_status": "ok" if qa.get("gate_ok") else ("flags" if qa else "none"),
                        "date": datetime.now().isoformat(timespec="seconds"), "params_hash": params_hash(o, rj),
                        "edge_mode": rj.get("edge_mode"),
                        **({"members": list(o.get("members") or [])} if o.get("kind") == "group" else {}),
                        # Final Touch: which base and which corrections (hash) – the export is reproducible (plan §3)
                        **({"final_touch": {"base": base_run(ver), "corrections_sig":
                                            corrections_sig(ft_load(work_dir / "objects" / o["name"])),
                                            **({"nuke": {k: v for k, v in ftn[o["name"]].items() if k in
                                                         ("mode", "requested", "why", "base_matte", "front_matte")}}
                                               if o["name"] in ftn else {})}} if is_ft(ver) else {})})
    data = load_outputs(work_dir)
    keep = [e for e in data.get("objects", []) if e["name"] not in written]
    update_outputs(work_dir, "objects", keep + entries)
    remove_outputs(work_dir, "groups")   # the legacy `<parent>_full` wholes (parts -> groups, PLAN_GROUP_COMPOSE step 6)
    if want is None and png:   # full export as PNG: no multi-EXR of this export (drop a stale entry)
        remove_outputs(work_dir, "mattes_all", "mattes_all_channels")
    update_outputs(work_dir, "export_format", fmt)
    if multichannel and not png:
        update_outputs(work_dir, "mattes_all", str(out_root / "all" / f"{shot}_mattes_####.exr").replace("\\", "/"))
    if multichannel and not png and multi_channels:
        update_outputs(work_dir, "mattes_all_channels", multi_channels)
    update_outputs(work_dir, "shot", shot)
    update_outputs(work_dir, "frame_range", [int(frames[0]), int(frames[-1])])
    update_outputs(work_dir, "plate_size", [int(w), int(h)])
    update_outputs(work_dir, "export_dir", str(export_dir).replace("\\", "/"))
    if holes:
        update_outputs(work_dir, "core_fix_fallback", holes)
    else:
        remove_outputs(work_dir, "core_fix_fallback")
    # Z29: Nuke script with the mattes (backdrop MASKI, Shuffle2 + anchor + copy stamp per channel)
    from engine.nuke_mattes import write_mattes_nk
    data = load_outputs(work_dir)
    from engine.final_touch import merge_matte_export
    merge = merge_matte_export(work_dir, [e["name"] for e in data.get("objects", [])], *merge_plate) if merge_plate else None
    from engine.ft_nuke import specs as ft_specs
    nk = write_mattes_nk(export_dir / f"{shot}_mattes.nk", data, shot, int(frames[0]), int(frames[-1]), w, h,
                         channels=data.get("mattes_all_channels") if data.get("mattes_all") else None,
                         merge=merge, ft=ft_specs(work_dir, data))
    update_outputs(work_dir, "mattes_nk", str(nk).replace("\\", "/"))
    return load_outputs(work_dir)


def mark_dependents(store: Any, name: str) -> List[str]:
    """After a change of a front object, objects behind it that overlap get status 'do przeliczenia' (7.3) –
    nothing is recomputed silently."""
    objs = store.list()
    me = store.get(name)
    depth, mine = me.get("depth_order", 0), set()
    if me.get("kind") == "group":   # a group entity: one layer at its front-most member, never its own members
        from engine.groups import group_depth
        mine = set(me.get("members") or [])
        depth = group_depth(store, name)
        if depth is None:
            return []
    changed = []
    for o in objs:
        if o["name"] not in mine and o.get("depth_order", 0) > depth and o["tool"].get("active_run"):
            deps = set(o["tool"].get("depends_on", [])) | {name}
            store.update(o["name"], tool={"status": "do przeliczenia", "depends_on": sorted(deps), "final_valid": False})
            changed.append(o["name"])
    return changed
