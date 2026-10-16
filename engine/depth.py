# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/depth.py - Depth map (SPEC 4e, stage E1e).

1. Compute: the user's DepthCrafter workflow (UI JSON) converted to API; the tool only patches paths and the
   frame range (PlateLoadEXR / PlateSaveEXR). Parameters stay the user's (20 steps / guidance 0.5 / window 60 /
   overlap 20) – reported, never changed. Input = deflickered plate (4c): `_deflicker` EXR when it exists,
   otherwise the plate multiplied by the 4c gain maps, written once to `<work>/matte_desk/depth/input/`.
2. No per-frame edge snapping to the plate (lesson: grain, boiling edges).
3. Stabilization: static-camera segments (from camera2d.json) -> one temporal median outside the motion mask
   (plate deviation from the segment median + object masks), ~20-frame crossfade where the camera moves.
4. QA: temporal 2nd derivative |d[t+1] - 2 d[t] + d[t-1]| at 512 px, static / moving, mean accel map image.
5. Hints only (never edges): depth_order from median disparity of object cores, tracker-jump flags.
6. Output: `export/depth/<shot>_depth_####.exr` float32, channels `depth.disparity` + copy in `R`
   (relative disparity: nearer = larger), `Depth_<shot>.nk` Group, `outputs.json` entry `depth`.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from engine.comfy_graph import find_nodes, load_workflow, one_node, ui_to_api
from engine.nuke_nk import validate_nk

logger = logging.getLogger("matte_desk.depth")

DEPTHCRAFTER_USER_PARAMS = {"num_inference_steps": 20, "guidance_scale": 0.5, "window_size": 60, "overlap": 20}
SEC_PER_4K_FRAME = 3.0      # 3080 Ti, measured by the user (reference_depthcrafter_comfy.md)
RAM_GB_PER_FRAME_4K = 34.0 / 806.0


def frame_path(pattern: str, f: int) -> Path:
    return Path(pattern.replace("####", f"{f:04d}").replace("%04d", f"{f:04d}"))


# ----------------------------------------------------------------------------- 1. Comfy graph
def build_depthcrafter_graph(workflow: Union[str, Path, Dict[str, Any]], in_pattern: str, first: int, last: int,
                             out_pattern: str) -> Dict[str, Any]:
    """User's workflow -> API graph with only paths and frame range replaced."""
    ui = load_workflow(workflow) if not isinstance(workflow, dict) else workflow
    api = ui_to_api(ui)
    one_node(api, "DepthCrafter")
    _, load = one_node(api, "PlateLoadEXR")
    _, save = one_node(api, "PlateSaveEXR")
    load["inputs"]["path_pattern"] = str(in_pattern).replace("\\", "/")
    load["inputs"]["first_frame"] = int(first)
    load["inputs"]["last_frame"] = int(last)
    save["inputs"]["out_pattern"] = str(out_pattern).replace("\\", "/")
    return api


def depthcrafter_params(api: Dict[str, Any]) -> Dict[str, Any]:
    _, dc = one_node(api, "DepthCrafter")
    return {k: dc["inputs"].get(k) for k in ("force_size", *DEPTHCRAFTER_USER_PARAMS)}


def estimate_cost(n_frames: int, plate_w: int, plate_h: int) -> Dict[str, float]:
    px = plate_w * plate_h / (4096.0 * 2731.0)
    return {"eta_s": round(SEC_PER_4K_FRAME * n_frames * px, 1),
            "ram_gb": round(4.0 + RAM_GB_PER_FRAME_4K * n_frames * px, 1)}


def prepare_input(plate_pattern: str, frames: Sequence[int], out_dir: Path,
                  gain_pattern: Optional[str] = None) -> str:
    """Deflickered input for DepthCrafter. With a gain-map pattern (4c Group variant) the plate is multiplied
    by the stored maps exactly like the Nuke Group; without it the plate is used as is."""
    if not gain_pattern:
        return plate_pattern
    from engine.clean import PlateCleaner
    from engine.exr_io import read_rgb, write_rgb_like

    out_dir.mkdir(parents=True, exist_ok=True)
    name = frame_path(plate_pattern, 0).name.replace("0000", "####")
    for f in frames:
        src = frame_path(plate_pattern, f)
        dst = out_dir / src.name
        if dst.exists():
            continue
        img = read_rgb(src)
        low = cv2.imread(str(frame_path(gain_pattern, f)), cv2.IMREAD_UNCHANGED)
        if low is None:
            raise FileNotFoundError(frame_path(gain_pattern, f))
        g = cv2.resize(low, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        write_rgb_like(dst, img * g, src)
    return str(out_dir / name).replace("\\", "/")


# ----------------------------------------------------------------------------- I/O
def read_depth(path: Union[str, Path]) -> np.ndarray:
    """Single-plane disparity from a DepthCrafter/PlateSaveEXR frame (R) or our export (depth.disparity)."""
    from engine.exr_io import read_exr

    ch, _ = read_exr(path)
    for name in ("depth.disparity", "R", "rgba.R", "Y"):
        if name in ch:
            return ch[name]
    return next(iter(ch.values()))


def write_depth(path: Union[str, Path], disp: np.ndarray, meta: Dict[str, str]) -> None:
    import Imath
    import OpenEXR

    h, w = disp.shape
    hdr = OpenEXR.Header(w, h)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    hdr["channels"] = {"depth.disparity": Imath.Channel(pt), "R": Imath.Channel(pt)}
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    for k, v in meta.items():
        hdr[f"matteDesk/{k}"] = str(v).encode()
    buf = np.ascontiguousarray(disp, dtype=np.float32).tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o = OpenEXR.OutputFile(str(path), hdr)
    try:
        o.writePixels({"depth.disparity": buf, "R": buf})
    finally:
        o.close()


def _small(img: np.ndarray, aw: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= aw:
        return img.astype(np.float32)
    return cv2.resize(img, (aw, max(1, round(h * aw / w))), interpolation=cv2.INTER_AREA).astype(np.float32)


# ----------------------------------------------------------------------------- 3. stabilization
def static_segments(homographies: Sequence[np.ndarray], w: int, h: int, thr_px: float = 0.5,
                    min_len: int = 10) -> List[Tuple[int, int]]:
    """Index ranges [a, b] (inclusive) where the camera is static: corners mapped to the reference by H
    (frame -> ref) move < thr_px relative to the segment's first frame."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T

    def proj(H):
        p = np.asarray(H, dtype=np.float64) @ corners
        return (p[:2] / p[2:]).T

    segs, a = [], 0
    pa = proj(homographies[0])
    for i in range(1, len(homographies) + 1):
        if i == len(homographies) or np.abs(proj(homographies[i]) - pa).max() >= thr_px:
            if i - a >= min_len:
                segs.append((a, i - 1))
            if i < len(homographies):
                a, pa = i, proj(homographies[i])
    return segs


def _motion_mask(plate_small: np.ndarray, plate_med: np.ndarray, extra: Optional[np.ndarray]) -> np.ndarray:
    """Soft mask of moving content: plate deviation from the segment median (log luma) + object masks."""
    dev = np.abs(np.log(plate_small + 0.01) - np.log(plate_med + 0.01))
    mad = float(np.median(np.abs(dev - np.median(dev)))) * 1.4826 + 1e-4
    m = (dev > max(0.05, np.median(dev) + 6 * mad)).astype(np.uint8)
    if extra is not None:
        m |= (extra > 0.5).astype(np.uint8)
    k = max(3, int(round(plate_small.shape[1] / 64)) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), k / 2.0) * 1.5, 0.0, 1.0)


def stabilize_small(depth: np.ndarray, plate_luma: Optional[np.ndarray], segments: Sequence[Tuple[int, int]],
                    object_masks: Optional[np.ndarray] = None, crossfade: int = 20
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """depth (T,h,w) -> (stabilized (T,h,w), weight of the static median (T,h,w), motion mask (T,h,w),
    per-segment medians)."""
    T = depth.shape[0]
    out = depth.copy()
    wts = np.zeros_like(depth)
    motion = np.zeros_like(depth)
    meds = []
    moving_frames = np.ones(T, bool)
    for a, b in segments:
        moving_frames[a:b + 1] = False
    for a, b in segments:
        sl = slice(a, b + 1)
        if plate_luma is not None:
            pmed = np.median(plate_luma[sl], axis=0)
            for t in range(a, b + 1):
                motion[t] = _motion_mask(plate_luma[t], pmed, None if object_masks is None else object_masks[t])
        elif object_masks is not None:
            motion[sl] = object_masks[sl].astype(np.float32)
        # median of depth over the segment, only where the pixel is static (moving pixels excluded)
        d = np.where(motion[sl] > 0.5, np.nan, depth[sl])
        med = np.nanmedian(d, axis=0)
        med = np.where(np.isnan(med), np.median(depth[sl], axis=0), med)
        meds.append(med)
        # crossfade ~20 frames at ends that border moving-camera frames (inside the static segment)
        tw = np.ones(b - a + 1, np.float32)
        cf = min(crossfade, (b - a + 1) // 2)
        if a > 0 and moving_frames[a - 1] and cf > 0:
            tw[:cf] = np.linspace(0.0, 1.0, cf + 2)[1:-1]
        if b < T - 1 and moving_frames[b + 1] and cf > 0:
            tw[-cf:] = np.minimum(tw[-cf:], np.linspace(1.0, 0.0, cf + 2)[1:-1])
        for k, t in enumerate(range(a, b + 1)):
            wt = tw[k] * (1.0 - motion[t])
            wts[t] = wt
            out[t] = depth[t] * (1.0 - wt) + med * wt
    for t in np.where(moving_frames)[0]:
        motion[t] = 1.0
    return out, wts, motion, meds


# ----------------------------------------------------------------------------- 4. QA
def accel_qa(depth: np.ndarray, motion: Optional[np.ndarray] = None, size: int = 512) -> Dict[str, Any]:
    """Temporal 2nd derivative at `size` px: mean over static / moving pixels + mean accel map."""
    small = np.stack([_small(d, size) for d in depth])
    rng = float(np.percentile(small, 99) - np.percentile(small, 1)) or 1.0
    small = small / rng  # scale-free (disparity has an arbitrary scale)
    acc = np.abs(small[2:] - 2 * small[1:-1] + small[:-2])
    res: Dict[str, Any] = {"accel_mean": float(acc.mean())}
    if motion is not None:
        m = np.stack([_small(x, size) for x in motion])[1:-1] > 0.5
        res["accel_static"] = float(acc[~m].mean()) if (~m).any() else 0.0
        res["accel_moving"] = float(acc[m].mean()) if m.any() else 0.0
    res["accel_map"] = acc.mean(axis=0)
    return res


def save_accel_png(acc_map: np.ndarray, path: Path, vmax: Optional[float] = None) -> None:
    vmax = vmax or float(np.percentile(acc_map, 99.5)) or 1.0
    img = (np.clip(acc_map / vmax, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.applyColorMap(img, cv2.COLORMAP_INFERNO))


# ----------------------------------------------------------------------------- 5. hints
def suggest_depth_order(depth: np.ndarray, object_masks: Dict[str, np.ndarray], erode_frac: float = 0.02
                        ) -> List[Dict[str, Any]]:
    """Median disparity in the eroded core of each object (all frames) -> depth_order (0 = nearest).
    Masks: (T,h,w) at the depth resolution. Objects not visible (empty core) are reported as such."""
    rows = []
    for name, m in object_masks.items():
        k = max(3, int(round(m.shape[-1] * erode_frac)) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        vals = []
        for t in range(m.shape[0]):
            core = cv2.erode((m[t] > 0.5).astype(np.uint8), ker).astype(bool)
            if core.any():
                vals.append(depth[t][core])
        if vals:
            v = np.concatenate(vals)
            rows.append({"name": name, "median_disparity": float(np.median(v)), "visible": True,
                         "spread": float(np.percentile(v, 90) - np.percentile(v, 10))})
        else:
            rows.append({"name": name, "median_disparity": float("nan"), "visible": False, "spread": 0.0})
    vis = sorted([r for r in rows if r["visible"]], key=lambda r: -r["median_disparity"])
    for i, r in enumerate(vis):
        r["depth_order"] = i
    return vis + [r for r in rows if not r["visible"]]


def tracker_jump_flags(depth: np.ndarray, core_masks: np.ndarray, k: float = 3.0) -> List[int]:
    """Frame indices where the median core disparity jumps > k x the local variability (SPEC 4e.2)."""
    med = np.array([np.median(depth[t][core_masks[t] > 0.5]) if (core_masks[t] > 0.5).any() else np.nan
                    for t in range(depth.shape[0])])
    d = np.abs(np.diff(med))
    flags = []
    for t in range(1, len(med) - 1):
        nb = np.concatenate([d[max(0, t - 6):t - 1], d[t + 1:t + 6]])
        nb = nb[~np.isnan(nb)]
        if len(nb) >= 3 and not np.isnan(d[t - 1]) and d[t - 1] > k * (np.median(nb) + 1e-6) and d[t - 1] > 1e-3:
            flags.append(t)
    return flags


# ----------------------------------------------------------------------------- report / pipeline
@dataclass
class DepthReport:
    schema: str = "matte_desk/depth@1.0"
    shot: str = ""
    kind: str = "disparity"  # relative disparity, nearer = larger, arbitrary scale
    method: str = "DepthCrafter"
    params: Dict[str, Any] = field(default_factory=dict)
    input_pattern: str = ""
    raw_pattern: str = ""
    export_pattern: str = ""
    nk_path: str = ""
    first_frame: int = 1
    last_frame: int = 1
    plate_w: int = 0
    plate_h: int = 0
    value_min: float = 0.0
    value_max: float = 0.0
    static_segments: List[Tuple[int, int]] = field(default_factory=list)
    qa_raw: Dict[str, float] = field(default_factory=dict)
    qa_stabilized: Dict[str, float] = field(default_factory=dict)
    depth_order_hint: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def postprocess_depth(raw_pattern: str, frames: Sequence[int], out_pattern: str, plate_pattern: Optional[str],
                      homographies: Optional[Sequence[np.ndarray]], plate_w: int, plate_h: int,
                      object_masks: Optional[Dict[str, np.ndarray]] = None, qa_dir: Optional[Path] = None,
                      analysis_w: int = 1024, crossfade: int = 20, meta: Optional[Dict[str, str]] = None,
                      progress_cb: Optional[Callable[[str, int, int], None]] = None) -> DepthReport:
    """Stabilize + QA + export full-resolution float EXRs. Heavy maths runs at `analysis_w` (DepthCrafter
    itself works at ~1/4 of 4K), full-resolution frames are streamed one at a time."""
    rep = DepthReport(raw_pattern=raw_pattern, export_pattern=out_pattern, first_frame=int(frames[0]),
                      last_frame=int(frames[-1]), plate_w=plate_w, plate_h=plate_h)
    raw_small = np.stack([_small(read_depth(frame_path(raw_pattern, f)), analysis_w) for f in frames])
    T, h, w = raw_small.shape

    luma = None
    if plate_pattern:
        from engine.exr_io import read_rgb

        def _luma(f):
            p = frame_path(plate_pattern, f)
            img = read_rgb(p) if p.suffix.lower() == ".exr" else cv2.imread(str(p), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            return cv2.resize(img.mean(axis=2), (w, h), interpolation=cv2.INTER_AREA)
        luma = np.stack([_luma(f) for f in frames])

    obj_union = None
    if object_masks:
        obj_union = np.zeros((T, h, w), np.float32)
        for m in object_masks.values():
            mm = np.stack([cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for x in m])
            obj_union = np.maximum(obj_union, mm)

    if homographies is None:
        segs = []
        rep.notes.append("brak camera2d.json – bez stabilizacji odcinków statycznych")
    else:
        segs = static_segments(homographies, plate_w, plate_h)
    rep.static_segments = [(int(frames[a]), int(frames[b])) for a, b in segs]

    stab, wts, motion, meds = stabilize_small(raw_small, luma, segs, obj_union, crossfade)
    q_raw, q_stab = accel_qa(raw_small, motion), accel_qa(stab, motion)
    rep.qa_raw = {k: round(v, 6) for k, v in q_raw.items() if k != "accel_map"}
    rep.qa_stabilized = {k: round(v, 6) for k, v in q_stab.items() if k != "accel_map"}
    if qa_dir is not None:
        vmax = float(np.percentile(q_raw["accel_map"], 99.5)) or 1.0
        save_accel_png(q_raw["accel_map"], qa_dir / "accel_raw.png", vmax)
        save_accel_png(q_stab["accel_map"], qa_dir / "accel_stabilized.png", vmax)

    if object_masks:
        small_masks = {k: np.stack([cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) for x in m])
                       for k, m in object_masks.items()}
        rep.depth_order_hint = suggest_depth_order(stab, small_masks)

    seg_of = {}
    for si, (a, b) in enumerate(segs):
        for t in range(a, b + 1):
            seg_of[t] = si
    vmin, vmax_ = math.inf, -math.inf
    for t, f in enumerate(frames):
        full = read_depth(frame_path(raw_pattern, f)).astype(np.float32)
        if full.shape != (plate_h, plate_w):
            full = cv2.resize(full, (plate_w, plate_h), interpolation=cv2.INTER_CUBIC)
        if t in seg_of and wts[t].max() > 0:
            wf = cv2.resize(wts[t], (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
            mf = cv2.resize(meds[seg_of[t]], (plate_w, plate_h), interpolation=cv2.INTER_CUBIC)
            full = full * (1.0 - wf) + mf * wf
        vmin, vmax_ = min(vmin, float(full.min())), max(vmax_, float(full.max()))
        write_depth(frame_path(out_pattern, f), full, {"kind": "disparity", "method": rep.method, **(meta or {})})
        if progress_cb:
            progress_cb("Zapis głębi", t + 1, T)
    rep.value_min, rep.value_max = vmin, vmax_
    return rep


# ----------------------------------------------------------------------------- Nuke
class NukeDepthGroupGenerator:
    """`Depth_<shot>.nk`: Input1 = plate stream; inside Read(depth seq) -> Expression (disparity | Z-like)
    -> Copy into depth.Z of the stream -> Output1. ZDefocus reads depth.Z."""

    @staticmethod
    def generate_script(shot: str, report: DepthReport, output_nk_path: Path) -> str:
        name = f"Depth_{shot}"
        pat = report.export_pattern.replace("\\", "/").replace("####", "%04d")
        w, h = report.plate_w, report.plate_h
        dmin = max(report.value_min, 1e-6)
        dmax = max(report.value_max, dmin * 1.0001)
        qa = report.qa_stabilized.get("accel_static", report.qa_stabilized.get("accel_mean", 0.0))
        text = f"""Group {{
 name {name}
 tile_color 0x6b4f9eff
 label "[value depth_mode] ({report.method})"
 addUserKnob {{20 DepthTab l Depth}}
 addUserKnob {{4 depth_mode l Output M {{disparity "Z-like (1/disparity)"}}}}
 depth_mode 0
 addUserKnob {{7 near l near R 0 1000}}
 near 1
 addUserKnob {{7 far l far R 0 10000}}
 far 100
 addUserKnob {{6 invert l invert +STARTLINE}}
 addUserKnob {{7 disp_min l "disparity min" R 0 1}}
 disp_min {dmin:.6g}
 addUserKnob {{7 disp_max l "disparity max" R 0 1}}
 disp_max {dmax:.6g}
 addUserKnob {{26 info_src l Source T "{report.method}, {report.kind} (bliżej = więcej)"}}
 addUserKnob {{26 info_date l Data T "{report.created_at}"}}
 addUserKnob {{26 info_qa l "QA accel (static)" T "{qa:.6f}"}}
}}
 Input {{
  inputs 0
  name Input1
  xpos 0
  ypos -80
 }}
 Dot {{
  name Dot1
  xpos 34
  ypos -20
 }}
set N_dot1 [stack 0]
 Read {{
  inputs 0
  file "{pat}"
  first {report.first_frame}
  last {report.last_frame}
  origfirst {report.first_frame}
  origlast {report.last_frame}
  format "{w} {h} 0 0 {w} {h} 1 depth_format"
  name Read_Depth
  xpos 180
  ypos -80
 }}
 Expression {{
  temp_name0 zl
  temp_expr0 "1 / max(r, parent.disp_min)"
  temp_name1 zn
  temp_expr1 "clamp((zl - 1 / parent.disp_max) / (1 / parent.disp_min - 1 / parent.disp_max))"
  expr0 "parent.depth_mode == 0 ? r : (parent.invert ? parent.far - (parent.far - parent.near) * zn : parent.near + (parent.far - parent.near) * zn)"
  channel0 {{rgba.red -rgba.green -rgba.blue none}}
  channel1 none
  channel2 none
  name Expression_DepthMode
  xpos 180
  ypos 0
 }}
push $N_dot1
 Copy {{
  inputs 2
  from0 rgba.red
  to0 depth.Z
  name Copy_Depth
  xpos 0
  ypos 60
 }}
 Output {{
  name Output1
  xpos 0
  ypos 140
 }}
end_group
"""
        problems = validate_nk(text, name)
        if problems:
            raise ValueError("; ".join(problems))
        output_nk_path.parent.mkdir(parents=True, exist_ok=True)
        output_nk_path.write_text(text, encoding="utf-8")
        return text


def run_depth(client: Any, workflow: Union[str, Path, Dict[str, Any]], shot: str, plate_pattern: str,
              frames: Sequence[int], plate_w: int, plate_h: int, work_dir: Path, export_dir: Path,
              gain_pattern: Optional[str] = None, homographies: Optional[Sequence[np.ndarray]] = None,
              object_masks: Optional[Dict[str, np.ndarray]] = None, timeout_s: Optional[float] = None,
              progress_cb: Optional[Callable[[str, int, int], None]] = None) -> DepthReport:
    """Whole 4e.1 chain: deflickered input -> user's DepthCrafter workflow in Comfy -> stabilize/QA ->
    export EXR + Depth_<shot>.nk + outputs.json entry."""
    from engine.outputs import update_outputs

    depth_dir = Path(work_dir) / "depth"
    first, last = int(frames[0]), int(frames[-1])
    in_pattern = prepare_input(plate_pattern, frames, depth_dir / "input", gain_pattern)
    raw_pattern = str(depth_dir / "raw" / f"{shot}_depth_raw_####.exr").replace("\\", "/")
    graph = build_depthcrafter_graph(workflow, in_pattern, first, last, raw_pattern)
    cost = estimate_cost(len(frames), plate_w, plate_h)
    if progress_cb:
        progress_cb(f"DepthCrafter (ETA ~{cost['eta_s']:.0f} s, RAM ~{cost['ram_gb']} GB)", 0, len(frames))

    q = client.get_queue()
    if q.get("busy"):
        raise RuntimeError("ComfyUI zajęte – poczekaj, aż skończy inne zadanie")
    client.free_vram()
    sub = client.queue_prompt(graph)
    if not sub.get("success"):
        raise RuntimeError(f"Comfy odrzucił graf: {sub.get('error')}")
    res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=timeout_s or max(600.0, cost["eta_s"] * 3))
    if not res.get("success"):
        raise RuntimeError(f"DepthCrafter nie powiódł się: {res.get('error')}")

    out_pattern = str(Path(export_dir) / "depth" / f"{shot}_depth_####.exr").replace("\\", "/")
    params = depthcrafter_params(graph)
    rep = postprocess_depth(raw_pattern, frames, out_pattern, in_pattern, homographies, plate_w, plate_h,
                            object_masks=object_masks, qa_dir=depth_dir / "qa",
                            meta={"params": json.dumps(params)}, progress_cb=progress_cb)
    rep.shot, rep.params, rep.input_pattern = shot, params, in_pattern
    nk = depth_dir / f"Depth_{shot}.nk"
    NukeDepthGroupGenerator.generate_script(shot, rep, nk)
    rep.nk_path = str(nk).replace("\\", "/")
    (depth_dir / "depth_report.json").write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    update_outputs(work_dir, "depth", {
        "path": out_pattern, "kind": rep.kind, "channel": "depth.disparity", "range": [rep.value_min, rep.value_max],
        "method": rep.method, "params": params, "nk_path": rep.nk_path, "report": str(depth_dir / "depth_report.json"),
        "qa": rep.qa_stabilized})
    return rep
