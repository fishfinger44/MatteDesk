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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.exr_io import write_gray_exr
from engine.outputs import load_outputs, remove_outputs, update_outputs
from engine.refine import assemble_full, read_alpha


CORE_RUN = "core"     # export source "rdzeń (1+2)" (2026-10-15): the mask 1+2 of core_fix/ instead of a run version


def core_fix_loader(obj_dir: Path, w: int, h: int) -> Callable[[int], np.ndarray]:
    """Full-frame 0 / 1 alpha of the mask 1+2 (core_fix/c_####.png, stage 2); a frame without it -> the SAM 3.1 core."""
    def load(f: int) -> np.ndarray:
        for sub in ("core_fix", "core"):
            m = cv2.imread(str(Path(obj_dir) / sub / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
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
    out = alpha.astype(np.float32)
    for fr in fronts:
        out = np.clip(out - fr, 0.0, 1.0)
    return out


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

    def run_of(self, name: str) -> str:
        return next(x for x in self.objs if x["name"] == name)["tool"]["active_run"]

    def frame(self, f: int) -> Dict[str, np.ndarray]:
        """{name: alpha} for every object / group of the context – the exported mattes of f."""
        alphas: Dict[str, np.ndarray] = {}
        front_layers: List[tuple] = []         # (name, raw alpha) of every layer in front so far
        for o in self.objs:
            raw = self.loaders[o["name"]](f) if self.visible(o["name"], f) else np.zeros((self.h, self.w), np.float32)
            skip = self._excluded(o)
            alphas[o["name"]] = subtract_front(raw, [a_ for n_, a_ in front_layers if n_ not in skip])
            front_layers.append((o["name"], raw))
        return alphas


def frame_alphas(work_dir: Path, objects: Sequence[Dict[str, Any]], f: int, w: int, h: int) -> Dict[str, np.ndarray]:
    """One frame of the export without writing anything (see FrameAlphas)."""
    return FrameAlphas(work_dir, objects, w, h).frame(f)


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

    def write_matte(name: str, a: np.ndarray, f: int, run: str) -> str:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        if png:
            cv2.imwrite(str(d / f"{name}_{f:04d}.png"), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        else:
            write_gray_exr(d / f"{name}_{f:04d}.exr", a, half=True, channels=("R", "G", "B"),
                           extra_attrs={"matteDesk/object": name.encode(), "matteDesk/run": run.encode()})
        return str(d / (f"{name}_####." + ("png" if png else "exr"))).replace("\\", "/")

    multi_channels: List[str] = []
    for i, f in enumerate(frames):
        alphas = fa.frame(f)
        for o in objs:
            if want is not None and o["name"] not in want:
                continue
            written[o["name"]] = write_matte(o["name"], alphas[o["name"]], f, o["tool"]["active_run"])
            if with_core:
                cp = work_dir / "objects" / o["name"] / "core" / f"c_{f:04d}.png"
                if cp.exists():
                    cd = out_root / f"{o['name']}_core"
                    cd.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(cd / f"{o['name']}_core_{f:04d}.png"), cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE))
        if multichannel and not png:
            import Imath
            import OpenEXR
            if want is not None:   # the palette's selection: its mattes only (the context objects cut, not written)
                alphas = {n: a for n, a in alphas.items() if n in want}
            union = np.clip(sum(alphas.values()), 0.0, 1.0) if alphas else np.zeros((h, w), np.float32)
            hdr = OpenEXR.Header(w, h)
            half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
            chans = {f"{n}.alpha": a for n, a in alphas.items()}
            chans["union.alpha"] = union
            if not multi_channels:
                multi_channels = list(alphas) + ["union"]
            if depth_pattern:
                from engine.depth import frame_path, read_depth
                dp = frame_path(depth_pattern, f)
                if dp.exists():
                    chans["depth.disparity"] = read_depth(dp)
                    if "depth" not in multi_channels:
                        multi_channels.append("depth")
            hdr["channels"] = {k: (Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) if k.startswith("depth.") else half)
                               for k in chans}
            hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
            md = out_root / "all"
            md.mkdir(parents=True, exist_ok=True)
            o_ = OpenEXR.OutputFile(str(md / f"{shot}_mattes_{f:04d}.exr"), hdr)
            o_.writePixels({k: np.ascontiguousarray(v, np.float32 if k.startswith("depth.") else np.float16).tobytes()
                            for k, v in chans.items()})
            o_.close()
        if progress_cb:
            progress_cb("Eksport", i + 1, len(frames))
    # outputs.json (data contract)
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
                                            corrections_sig(ft_load(work_dir / "objects" / o["name"]))}} if is_ft(ver) else {})})
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
    # Z29: Nuke script with the mattes (backdrop MASKI, Shuffle2 + anchor + copy stamp per channel)
    from engine.nuke_mattes import write_mattes_nk
    data = load_outputs(work_dir)
    from engine.final_touch import merge_matte_export
    merge = merge_matte_export(work_dir, [e["name"] for e in data.get("objects", [])], *merge_plate) if merge_plate else None
    nk = write_mattes_nk(export_dir / f"{shot}_mattes.nk", data, shot, int(frames[0]), int(frames[-1]), w, h,
                         channels=data.get("mattes_all_channels") if data.get("mattes_all") else None,
                         merge=merge)
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
