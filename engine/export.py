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
- parts of one object (object field `part_of`, e.g. Figure's dress `part_of: figure`): the whole (parent + parts) is one
  seamless matte `<parent>_full` (internal contact gaps between the parts' cores filled); the first part in depth order
  keeps its own alpha, the next ones get the rest: part = clamp(whole - earlier parts) -> parts sum to the whole exactly
  (no gap, no overlap at the contact). Channel `<parent>_full.alpha` + folder `export/mattes/<parent>_full/`. (Physical mattes split by regions = backlit objects, MatteSplitByRegions in Comfy.)
- optional core / confidence (`<obj>_core/`, `<obj>_conf/`) for the 3D tool (visual hull).
- outputs.json updated (objects: name, depth_order, paths, run version, QA status, date, params hash).
No RGB is ever written (Z3).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.exr_io import write_gray_exr
from engine.outputs import load_outputs, remove_outputs, update_outputs
from engine.refine import assemble_full, read_alpha


CORE_RUN = "core"     # export source "rdzeń (1+2)" (2026-10-15): the mask 1+2 of core_fix/ instead of a run version


def core_fix_loader(obj_dir: Path, w: int, h: int) -> Callable[[int], np.ndarray]:
    """Full-frame 0 / 1 alpha of the mask 1+2 (core_fix/c_####.png, stage 2); a frame without it -> the SAM3 core."""
    def load(f: int) -> np.ndarray:
        for sub in ("core_fix", "core"):
            m = cv2.imread(str(Path(obj_dir) / sub / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.float32)
        return np.zeros((h, w), np.float32)
    return load


def run_alpha_loader(obj_dir: Path, run: str, w: int, h: int, sub: str = "alpha") -> Callable[[int], np.ndarray]:
    """Full-frame alpha of a run version (crop segments pasted back, overlaps crossfaded). `sub="alpha_base"`: the
    version's base alpha (edge mode without zones, written when the version has zones). `run == CORE_RUN`: the
    mask 1+2 (core_fix_loader)."""
    if run == CORE_RUN:
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
                per[s.seg_id] = {f: read_alpha(p)}
        return assemble_full(per, [s for s in cover if s.seg_id in per], [f], w, h)[f]
    return load


EXPORT_FORMATS = ("auto", "png", "exr")


def resolve_export_format(fmt: Optional[str], bit_depth: int) -> str:
    """'auto' (or empty) -> 'png' for 8-bit sources, 'exr' otherwise; 'png' / 'exr' as given."""
    f = (fmt or "auto").lower()
    if f not in EXPORT_FORMATS:
        raise ValueError(f"export_format musi być jednym z {EXPORT_FORMATS}, jest {fmt!r}")
    if f == "auto":
        return "png" if int(bit_depth) <= 8 else "exr"
    return f


def part_groups(objs: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """parent -> [parent + its parts] in depth order (only parents that have parts present in `objs`)."""
    by = {o["name"]: o for o in objs}
    groups: Dict[str, List[str]] = {}
    for o in objs:
        p = o.get("part_of")
        if p and p in by and p != o["name"]:
            groups.setdefault(p, [p]).append(o["name"])
    return {p: sorted(m, key=lambda n: by[n].get("depth_order", 0)) for p, m in groups.items()}


SEAM_PX_DEFAULT = 4    # contact gap between the parts' cores filled in `<parent>_full` (object field `part_seam_px`)
SEAM_PX_MAX = 40


def seam_px(obj: Optional[Dict[str, Any]]) -> int:
    """Seam width of a whole made of parts: the parent's `part_seam_px` (0..SEAM_PX_MAX), default SEAM_PX_DEFAULT."""
    try:
        v = int(round(float((obj or {}).get("part_seam_px", SEAM_PX_DEFAULT))))
    except (TypeError, ValueError):
        v = SEAM_PX_DEFAULT
    return max(0, min(SEAM_PX_MAX, v))


def seamless_union(alphas: Sequence[np.ndarray], cores: Sequence[Optional[np.ndarray]], gap_px: int = 4) -> np.ndarray:
    """Whole matte of an object made of parts: max of the parts' alphas, with the contact seams between the parts'
    cores (up to `gap_px` apart) filled to 1. The outer contour is untouched (filling only where two parts meet)."""
    u = np.max(np.stack([a.astype(np.float32) for a in alphas]), axis=0)
    cs = [c.astype(bool) for c in cores if c is not None]
    if len(cs) < 2 or gap_px <= 0:
        return u
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    # the bridge between two parts: d_i + d_j <= 2 * gap_px + 1 (distance to each part's core) – the WHOLE gap up to
    # ~2 x gap_px wide (was: only pixels within gap_px of both = a thin strip in the middle, holes along both edges)
    dist = [cv2.distanceTransform((~c).astype(np.uint8), cv2.DIST_L2, 5) for c in cs]
    contact = np.zeros_like(cs[0])
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            contact |= (dist[i] + dist[j]) <= 2 * gap_px + 1
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
    subtraction by depth_order, wholes of objects with parts (`<parent>_full`) split into parts. `objects` = the context
    (every object with a final result). The export and the stage-4/5 preview (`frame?view=alpha&export=1`, full-res
    tiles) share this code, so what the viewer shows is the exported alpha of the version."""

    def __init__(self, work_dir: Path, objects: Sequence[Dict[str, Any]], w: int, h: int):
        self.work_dir = Path(work_dir)
        # a group computed as a whole (stages 3 / 4 on a parent with parts, run.json "group", 2026-10-15): its parts
        # take the parent's version – the whole – and get their share of it (group_split)
        self.objs = with_group_runs(self.work_dir, sorted(objects, key=lambda o: o.get("depth_order", 0)))
        self.group_run = {o["name"] for o in self.objs if group_of_run(self.work_dir, o)}
        self.loaders = {o["name"]: run_alpha_loader(self.work_dir / "objects" / run_owner(o), o["tool"]["active_run"], w, h)
                        for o in self.objs}
        self.groups = part_groups(self.objs)
        self.member_of = {n: p for p, ms in self.groups.items() for n in ms}
        # object life range (engine/workarea.py): outside it the object's matte is empty (black frames, nothing
        # subtracted from the objects behind)
        from engine.workarea import object_life
        self.life = {o["name"]: object_life(o) for o in self.objs}
        self.w, self.h = w, h

    def visible(self, name: str, f: int) -> bool:
        from engine.workarea import is_visible
        return is_visible(f, self.life.get(name))

    def run_of(self, name: str) -> str:
        return next(x for x in self.objs if x["name"] == name)["tool"]["active_run"]

    def core_of(self, name: str, f: int) -> Optional[np.ndarray]:
        cp = self.work_dir / "objects" / name / "core" / f"c_{f:04d}.png"
        m = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE) if cp.exists() else None
        return None if m is None else m > 127

    def mask12_of(self, name: str, f: int) -> Optional[np.ndarray]:
        """Mask 1+2 of an object (core_fix/, else the SAM3 core) – which part of a whole a pixel belongs to."""
        for sub in ("core_fix", "core"):
            m = cv2.imread(str(self.work_dir / "objects" / name / sub / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != (self.h, self.w):
                    m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                return m > 127
        return None

    def frame(self, f: int) -> Dict[str, np.ndarray]:
        """{name: alpha} for every object of the context + `<parent>_full` for every group – the exported mattes of f."""
        alphas: Dict[str, np.ndarray] = {}
        fronts: List[np.ndarray] = []
        raws = {o["name"]: (self.loaders[o["name"]](f) if self.visible(o["name"], f)
                            else np.zeros((self.h, self.w), np.float32)) for o in self.objs}
        whole: Dict[str, np.ndarray] = {}      # parent -> remaining part of the whole (after earlier parts)
        for o in self.objs:
            raw = raws[o["name"]]
            p = self.member_of.get(o["name"])
            if p is None:
                a = subtract_front(raw, fronts)
            elif p in self.group_run:   # the whole was computed at once: parts = shares of it (they sum to the whole)
                ms = self.groups[p]
                if p not in whole:
                    u = subtract_front(raws[p], fronts)
                    alphas[f"{p}_full"] = u.copy()
                    whole[p] = dict(zip(ms, group_split(u, [self.mask12_of(n, f) if self.visible(n, f) else None for n in ms])))
                a = whole[p][o["name"]]
                fronts.append(a)
                alphas[o["name"]] = a
                continue
            else:
                ms = self.groups[p]
                if p not in whole:
                    u = seamless_union([raws[n] for n in ms], [self.core_of(n, f) if self.visible(n, f) else None for n in ms],
                                       gap_px=seam_px(next((x for x in self.objs if x["name"] == p), None)))
                    whole[p] = subtract_front(u, fronts)
                    alphas[f"{p}_full"] = whole[p].copy()
                last = o["name"] == ms[-1]
                a = whole[p] if last else np.minimum(subtract_front(raw, fronts), whole[p])
                whole[p] = np.clip(whole[p] - a, 0.0, 1.0)
            fronts.append(raw)
            alphas[o["name"]] = a
        return alphas


def run_owner(o: Dict[str, Any]) -> str:
    """Object whose runs/ hold the version of `o` (a part of a group takes its parent's group version)."""
    return o.get("_run_owner") or o["name"]


def group_of_run(work_dir: Path, o: Dict[str, Any]) -> Optional[List[str]]:
    """Members of the group the object's version was computed for as a whole (run.json "group"), else None."""
    run = (o.get("tool") or {}).get("active_run")
    if not run:
        return None
    try:
        g = json.loads((Path(work_dir) / "objects" / run_owner(o) / "runs" / run / "run.json").read_text(encoding="utf-8")).get("group")
    except (OSError, ValueError):
        return None
    return list(g) if g else None


def with_group_runs(work_dir: Path, objs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parts of a group whose parent's version is a whole (run.json "group") take that version (copies of the dicts)."""
    by = {o["name"]: o for o in objs}
    out = []
    for o in objs:
        par = by.get(o.get("part_of") or "")
        g = group_of_run(work_dir, par) if par else None
        if g and o["name"] in g:
            o = {**o, "_run_owner": par["name"], "tool": {**(o.get("tool") or {}), "active_run": par["tool"]["active_run"]}}
        out.append(o)
    return out


def group_split(whole: np.ndarray, masks: Sequence[Optional[np.ndarray]]) -> List[np.ndarray]:
    """Shares of a whole computed at once: every pixel goes to the member whose mask 1+2 is nearest (inside a mask:
    that member; ties: the earlier = closer to the camera) -> the shares sum to the whole exactly, no gap, no overlap."""
    whole = whole.astype(np.float32)
    n = len(masks)
    if n == 0:
        return []
    big = np.float32(1e9)
    dist = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) if (m is not None and m.any())
                     else np.full(whole.shape, big, np.float32) for m in masks])
    lab = np.argmin(dist, axis=0)
    return [np.where(lab == i, whole, 0.0).astype(np.float32) for i in range(n)]


def frame_alphas(work_dir: Path, objects: Sequence[Dict[str, Any]], f: int, w: int, h: int) -> Dict[str, np.ndarray]:
    """One frame of the export without writing anything (see FrameAlphas)."""
    return FrameAlphas(work_dir, objects, w, h).frame(f)


def export_objects(work_dir: Path, shot: str, objects: Sequence[Dict[str, Any]], frames: Sequence[int], w: int, h: int,
                   bit_depth: int = 16, multichannel: bool = True, depth_pattern: Optional[str] = None,
                   with_core: bool = False, progress_cb: Optional[Callable[[str, int, int], None]] = None,
                   write: Optional[Sequence[str]] = None, export_dir: Optional[Path] = None,
                   export_format: str = "auto") -> Dict[str, Any]:
    """Export objects (each must have an active run). Front-to-back subtraction by depth_order.
    `objects` = the context (all objects with a final result) – mattes are always computed with every front object and
    every part, so a single-object export equals the same object in a full export. `write` = names to write (object
    names and/or "<parent>_full" for a whole made of parts); None = everything + the multi-EXR.
    `export_format` auto / png / exr (resolve_export_format with `bit_depth`); the multi-EXR only for exr."""
    want = None if write is None else set(write)
    multichannel = multichannel and want is None   # a partial export never overwrites the shot's multi-EXR
    work_dir = Path(work_dir)
    export_dir = Path(export_dir) if export_dir else work_dir / "export"   # Z29: chosen in project.json -> export_dir
    out_root = export_dir / "mattes"
    fa = FrameAlphas(work_dir, objects, w, h)
    objs = fa.objs
    fmt = resolve_export_format(export_format, bit_depth)
    png = fmt == "png"
    written: Dict[str, str] = {}
    groups = fa.groups

    def write_matte(name: str, a: np.ndarray, f: int, run: str) -> str:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        if png:
            cv2.imwrite(str(d / f"{name}_{f:04d}.png"), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        else:
            write_gray_exr(d / f"{name}_{f:04d}.exr", a, half=True, channels=("R", "G", "B"),
                           extra_attrs={"matteDesk/object": name.encode(), "matteDesk/run": run.encode()})
        return str(d / (f"{name}_####." + ("png" if png else "exr"))).replace("\\", "/")

    group_written: Dict[str, str] = {}
    multi_channels: List[str] = []
    for i, f in enumerate(frames):
        alphas = fa.frame(f)
        for p, ms in groups.items():
            if want is None or f"{p}_full" in want:
                group_written[p] = write_matte(f"{p}_full", alphas[f"{p}_full"], f, "+".join(fa.run_of(n) for n in ms))
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
        rd = work_dir / "objects" / run_owner(o) / "runs" / o["tool"]["active_run"]
        if o["tool"]["active_run"] == CORE_RUN:   # mask 1+2: no run.json / QA of its own
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
                        "edge_mode": rj.get("edge_mode")})
    data = load_outputs(work_dir)
    keep = [e for e in data.get("objects", []) if e["name"] not in written]
    update_outputs(work_dir, "objects", keep + entries)
    if want is None and png:   # full export as PNG: no multi-EXR of this export (drop a stale entry)
        remove_outputs(work_dir, "mattes_all", "mattes_all_channels")
    update_outputs(work_dir, "export_format", fmt)
    if multichannel and not png:
        update_outputs(work_dir, "mattes_all", str(out_root / "all" / f"{shot}_mattes_####.exr").replace("\\", "/"))
    if group_written:
        g = dict(data.get("groups", {}))
        g.update({p: {"matte": group_written[p], "members": groups[p]} for p in group_written})
        update_outputs(work_dir, "groups", g)
    if multichannel and not png and multi_channels:
        update_outputs(work_dir, "mattes_all_channels", multi_channels)
    update_outputs(work_dir, "shot", shot)
    update_outputs(work_dir, "frame_range", [int(frames[0]), int(frames[-1])])
    update_outputs(work_dir, "plate_size", [int(w), int(h)])
    update_outputs(work_dir, "export_dir", str(export_dir).replace("\\", "/"))
    # Z29: Nuke script with the mattes (backdrop MASKI, Shuffle2 + anchor + copy stamp per channel)
    from engine.nuke_mattes import write_mattes_nk
    data = load_outputs(work_dir)
    nk = write_mattes_nk(export_dir / f"{shot}_mattes.nk", data, shot, int(frames[0]), int(frames[-1]), w, h,
                         channels=data.get("mattes_all_channels") if data.get("mattes_all") else None)
    update_outputs(work_dir, "mattes_nk", str(nk).replace("\\", "/"))
    return load_outputs(work_dir)


def mark_dependents(store: Any, name: str) -> List[str]:
    """After a change of a front object, objects behind it that overlap get status 'do przeliczenia' (7.3) –
    nothing is recomputed silently."""
    objs = store.list()
    me = next(o for o in objs if o["name"] == name)
    changed = []
    for o in objs:
        if o.get("depth_order", 0) > me.get("depth_order", 0) and o["tool"].get("active_run"):
            deps = set(o["tool"].get("depends_on", [])) | {name}
            store.update(o["name"], tool={"status": "do przeliczenia", "depends_on": sorted(deps), "final_valid": False})
            changed.append(o["name"])
    return changed
