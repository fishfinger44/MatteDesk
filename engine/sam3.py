# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/sam3.py - SAM 3.1 through ComfyUI (SPEC 5.1, 5.3, 7.2). Graphs use only core SAM 3.1 nodes, core I/O and
the PlateMatte loaders; frames go to Comfy as files written by the tool (display values, already cropped /
squared), never as full 4K plates.

- `detect`: SAM3_Detect on one image (text and/or points), `individual_masks` for instance choice (5.2 pkt 0).
- `track`:  SAM3_VideoTrack WITHOUT conditioning (tracker only follows, never detects – 5.1), max_objects 1,
  initial_mask = key mask for the first frame of the batch. Backward tracking = reversed frame order.
Frame indices inside one Comfy call are 0..N-1 (SPEC 4: plate numbers everywhere else).
Input files by source bit depth (engine/comfy_io.py): "exr" = float32 EXR + PlateLoadEXR (unchanged);
"png8" = 8-bit PNG + VHS_LoadImagesPath, track frames > SAM_RES downscaled, masks resized back.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SAM_RES = 1008


def default_checkpoint() -> str:
    from engine.config import load_config
    return load_config().get("sam3_checkpoint") or "sam3.1_multiplex_fp16.safetensors"


def _write_exr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.ascontiguousarray(img, dtype=np.float32))


def detect_graph(image_pattern: str, frame: int, prompt: Optional[str], positive: Optional[List[Dict[str, int]]],
                 negative: Optional[List[Dict[str, int]]], individual: bool, prefix: str, ckpt: str,
                 threshold: float = 0.5) -> Dict[str, Any]:
    g: Dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "PlateLoadEXR", "inputs": {"path_pattern": image_pattern, "first_frame": frame,
                                                       "last_frame": frame, "crop_x": 0, "crop_y": 0, "crop_w": 0,
                                                       "crop_h": 0, "scale": 1.0, "bad_frames": ""}},
        "6": {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["2", 0], "threshold": threshold,
                                                      "refine_iterations": 2, "individual_masks": bool(individual)}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }
    if prompt:
        g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}}
        g["6"]["inputs"]["conditioning"] = ["3", 0]
    if positive:
        g["4"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(positive)}}
        g["6"]["inputs"]["positive_coords"] = ["4", 0]
    if negative:
        g["5"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(negative)}}
        g["6"]["inputs"]["negative_coords"] = ["5", 0]
    if not (prompt or positive):
        raise ValueError("SAM3_Detect needs a text prompt or positive points")
    return g


def detect_graph_png8(image_dir: str, prompt: Optional[str], positive: Optional[List[Dict[str, int]]],
                      negative: Optional[List[Dict[str, int]]], individual: bool, prefix: str, ckpt: str,
                      threshold: float = 0.5) -> Dict[str, Any]:
    """detect_graph with the image from an 8-bit PNG folder (VHS_LoadImagesPath) instead of PlateLoadEXR."""
    from engine.comfy_io import vhs_load
    g = detect_graph("", 0, prompt, positive, negative, individual, prefix, ckpt, threshold)
    g["2"] = vhs_load(image_dir)
    return g


def track_graph_png8(frames_dir: str, init_dir: str, prefix: str, ckpt: str) -> Dict[str, Any]:
    """track_graph for 8-bit sources: frames = every PNG of `frames_dir` (sorted), initial mask = the one PNG of
    `init_dir` (VHS_LoadImagesPath + ImageToMask red -> ONE object)."""
    from engine.comfy_io import image_to_mask, vhs_load
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": vhs_load(frames_dir),
        "3": vhs_load(init_dir),
        "4": image_to_mask(["3", 0]),
        "5": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["2", 0], "model": ["1", 0],
                                                          "initial_mask": ["4", 0], "detection_threshold": 0.5,
                                                          "max_objects": 1, "detect_interval": 1}},
        "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": ""}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def track_graph(frames_pattern: str, n: int, init_mask_pattern: str, prefix: str, ckpt: str) -> Dict[str, Any]:
    """Frames 0..n-1; initial mask = PNG of frame 0 (loaded via a 1-frame plate_info so it is ONE object)."""
    load = {"path_pattern": frames_pattern, "crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0, "scale": 1.0,
            "bad_frames": ""}
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "PlateLoadEXR", "inputs": {**load, "first_frame": 0, "last_frame": n - 1}},
        "3": {"class_type": "PlateLoadEXR", "inputs": {**load, "first_frame": 0, "last_frame": 0}},
        "4": {"class_type": "PlateLoadMaskSeq", "inputs": {"path_pattern": init_mask_pattern, "plate_info": ["3", 1],
                                                           "frame_offset": 0}},
        "5": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["2", 0], "model": ["1", 0],
                                                          "initial_mask": ["4", 0], "detection_threshold": 0.5,
                                                          "max_objects": 1, "detect_interval": 1}},
        "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": ""}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


SOFT_NODE = "SAM3_VideoTrackSoft"     # PlateMatte node: core SAM3_VideoTrack + sigmoid(logits) (engine/corestab prob mode)


def with_soft_output(g: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Track graph -> node "5" = SAM3_VideoTrackSoft (same inputs; output 0 = track_data as before, so the binary
    masks of node "8" are bit-identical), soft masks saved as 8-bit PNG by node "10" (prefix + "_soft")."""
    out = {k: dict(v) for k, v in g.items()}
    out["5"] = {"class_type": SOFT_NODE, "inputs": dict(g["5"]["inputs"])}
    out["9"] = {"class_type": "MaskToImage", "inputs": {"mask": ["5", 1]}}
    out["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix + "_soft"}}
    return out


def combine_prompt_points(prompt_mask: Optional[np.ndarray], plus_mask: Optional[np.ndarray],
                          minus_mask: Optional[np.ndarray], thr: float = 0.5) -> Tuple[np.ndarray, Dict[str, int]]:
    """„▶ Przelicz klatkę” with a prompt (user decision 2026-10-15, second version): the prompt is the base, positive
    points ADD what they point at, negative points SUBTRACT what they point at: (prompt ∪ plus) − minus.
    prompt_mask = SAM 3.1 from the prompt, plus_mask = SAM 3.1 from the positive points (negatives as negatives), minus_mask =
    SAM 3.1 from the negative points used as positives (what was clicked; positives as negatives). None = not computed.
    (SAM3_Detect given a prompt and points at once returns their union and ignores negative points – measured.)"""
    shape = next(m.shape for m in (prompt_mask, plus_mask, minus_mask) if m is not None)
    b = lambda m: (m > thr) if m is not None else np.zeros(shape, bool)
    base, plus, minus = b(prompt_mask), b(plus_mask), b(minus_mask)
    out = (base | plus) & ~minus
    return out, {"prompt_px": int(base.sum()), "plus_px": int(plus.sum()), "minus_px": int((minus & (base | plus)).sum())}


class Sam3Runner:
    """Runs SAM 3.1 graphs through a ComfyClient-like object (real or mock)."""

    def __init__(self, client: Any, scratch_dir: Path, ckpt: Optional[str] = None, io_mode: str = "exr"):
        from engine.comfy_io import EXR, PNG8
        if io_mode not in (EXR, PNG8):
            raise ValueError(f"io_mode {io_mode!r}")
        self.client = client
        self.scratch = Path(scratch_dir)
        self.ckpt = ckpt or default_checkpoint()
        self.io_mode = io_mode
        # EXR mode: frames > SAM_RES downscaled before tracking (png8 always did); MD_SAM_FULLRES=1 = the old full crop
        import os
        self.sam_downscale = os.environ.get("MD_SAM_FULLRES", "0") != "1"
        self._soft: Optional[bool] = None

    def soft_available(self) -> bool:
        """True when ComfyUI has the PlateMatte node SAM3_VideoTrackSoft (object_info); cached per runner."""
        if self._soft is None:
            try:
                info = self.client.get_object_info(SOFT_NODE)
                self._soft = bool(info)
            except Exception:
                self._soft = False
        return self._soft

    def _run_nodes(self, graph: Dict[str, Any], raw: bool, nodes: Sequence[str]) -> Dict[str, List[np.ndarray]]:
        """Images of the given SaveImage nodes in file order: float 0..1, or (raw) the decoded uint8 PNGs."""
        busy = self.client.busy_error()
        if busy:
            raise RuntimeError(busy)
        from engine.gpu_manager import comfy_lease
        with comfy_lease(self.client, "SAM 3.1"):
            sub = self.client.queue_prompt(graph)
            if not sub.get("success"):
                raise RuntimeError(f"Comfy odrzucił graf SAM 3.1: {sub.get('error')}")
            res = self.client.wait_for_prompt(sub["prompt_id"], timeout_sec=1800)
        if not res.get("success"):
            raise RuntimeError(f"SAM 3.1 nie powiódł się: {res.get('error')}")
        result: Dict[str, List[np.ndarray]] = {}
        for nid in nodes:
            imgs = res.get("outputs", {}).get(nid, {}).get("images", [])
            out = []
            for info in sorted(imgs, key=lambda i: i["filename"]):
                buf = np.frombuffer(self.client.download_output(info), np.uint8)
                m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                out.append(m if raw else m.astype(np.float32) / 255.0)
            result[nid] = out
        return result

    def _run(self, graph: Dict[str, Any], raw: bool = False) -> List[np.ndarray]:
        """Masks of SaveImage node "8" in file order: float 0..1, or (raw) the decoded uint8 PNGs."""
        return self._run_nodes(graph, raw, ("8",))["8"]

    def detect(self, image: np.ndarray, prompt: Optional[str] = None,
               positive: Optional[Sequence[Tuple[float, float]]] = None,
               negative: Optional[Sequence[Tuple[float, float]]] = None, individual: bool = True,
               threshold: float = 0.5) -> List[np.ndarray]:
        """image: HxWx3 display 0..1 (BGR). Points in image pixels. Returns masks (HxW float 0..1)."""
        job = self.scratch / f"detect_{uuid.uuid4().hex[:8]}"
        pts = lambda p: [{"x": int(round(x)), "y": int(round(y))} for x, y in p] if p else None
        if self.io_mode == "png8":
            from engine.comfy_io import require_vhs, write_png8
            require_vhs(self.client)
            write_png8(job / "img" / "img_0000.png", image)
            return self._run(detect_graph_png8(str(job / "img").replace("\\", "/"), prompt, pts(positive),
                                               pts(negative), individual, f"matte_desk/{job.name}/m", self.ckpt,
                                               threshold))
        _write_exr(job / "img_0000.exr", image)
        g = detect_graph(str(job / "img_####.exr").replace("\\", "/"), 0, prompt, pts(positive), pts(negative),
                         individual, f"matte_desk/{job.name}/m", self.ckpt, threshold)
        return self._run(g)

    def track(self, frames: Sequence[np.ndarray], init_mask: np.ndarray, soft: bool = False):
        """frames: list of HxWx3 display images (constant crop!), init_mask for frames[0]. Returns len(frames) masks;
        with `soft=True` (node SAM3_VideoTrackSoft available) a tuple (masks, soft probabilities 0..1) – the masks
        are bit-identical to the plain run (same track_data, same SAM3_TrackToMask)."""
        from engine.comfy_io import fit_sam_res, job_progress
        job = self.scratch / f"track_{uuid.uuid4().hex[:8]}"
        n = len(frames)
        soft = bool(soft) and self.soft_available()
        if self.io_mode == "png8":
            from engine.comfy_io import require_vhs
            require_vhs(self.client)
            return self._track_png8(job, frames, init_mask, soft)
        # SAM 3.1 resizes every frame to SAM_RES² itself (bicubic, comfy/ldm/sam3/tracker.py), so a larger crop only costs
        # time: 96 frames of a 2688 px crop were 4 GB of float EXR and ~8 GB in Comfy, which froze it (SH010 2026-10-21).
        # Frames are downscaled here with the SAME filter (bicubic: A/B on SH010 20 fr IoU 0.9988, 19/20 frames >= 0.998;
        # INTER_AREA gave 0.9947), the key mask with INTER_AREA, the masks come back to the input size (INTER_LINEAR).
        # MatAnyone2 still gets full resolution.
        in_hw, down = (0, 0), False
        for i, f in enumerate(frames):                       # each frame read once (lazy sequences load on access)
            f = np.asarray(f)
            if i == 0:
                in_hw = f.shape[:2]
                down = self.sam_downscale and max(in_hw) > SAM_RES
            _write_exr(job / f"f_{i:04d}.exr", fit_sam_res(f, SAM_RES, cv2.INTER_CUBIC) if down else f)
            job_progress("SAM 3.1: zapis klatek do Comfy", i + 1, n)
        job.mkdir(parents=True, exist_ok=True)
        k0 = np.clip(np.asarray(init_mask, np.float32), 0, 1)
        k0 = fit_sam_res(k0, SAM_RES) if down else k0
        cv2.imwrite(str(job / "k_0000.png"), (k0 * 255).round().astype(np.uint8) if down else (k0 * 255).astype(np.uint8))
        prefix = f"matte_desk/{job.name}/t"
        g = track_graph(str(job / "f_####.exr").replace("\\", "/"), len(frames),
                        str(job / "k_####.png").replace("\\", "/"), prefix, self.ckpt)
        if soft:
            g = with_soft_output(g, prefix)
        outs = self._run_nodes(g, False, ("8", "10") if soft else ("8",))
        masks = outs["8"]
        if len(masks) != len(frames):
            raise RuntimeError(f"SAM3_VideoTrack zwrócił {len(masks)} masek zamiast {len(frames)}")
        h, w = in_hw
        back = (lambda ms: [m if m.shape[:2] == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in ms])
        if not soft:
            return back(masks) if down else masks
        softs = outs["10"]
        if len(softs) != len(frames):
            raise RuntimeError(f"SAM3_VideoTrackSoft zwrócił {len(softs)} miękkich masek zamiast {len(frames)}")
        return (back(masks), back(softs)) if down else (masks, softs)

    def _track_png8(self, job: Path, frames: Sequence[np.ndarray], init_mask: np.ndarray, soft: bool = False):
        """8-bit sources: PNG frames (downscaled to fit SAM_RES) in job/frames, initial mask in job/init. Returns a
        sequence of masks at the input size (resized back INTER_LINEAR on access); with `soft` also the soft masks."""
        from engine.comfy_io import fit_sam_res, job_progress, write_png8
        n = len(frames)
        in_hw = None
        for i, f in enumerate(frames):
            if in_hw is None:
                in_hw = f.shape[:2]
            write_png8(job / "frames" / f"f_{i:04d}.png", fit_sam_res(f, SAM_RES))
            job_progress("SAM 3.1: zapis klatek do Comfy", i + 1, n)
        small = fit_sam_res(np.ascontiguousarray(np.clip(init_mask, 0, 1), np.float32), SAM_RES)
        write_png8(job / "init" / "k_0000.png", small)
        prefix = f"matte_desk/{job.name}/t"
        g = track_graph_png8(str(job / "frames").replace("\\", "/"), str(job / "init").replace("\\", "/"),
                             prefix, self.ckpt)
        if soft:
            g = with_soft_output(g, prefix)
        outs = self._run_nodes(g, True, ("8", "10") if soft else ("8",))   # uint8 at SAM size: ~1 MB / frame
        small = outs["8"]
        if len(small) != n:
            raise RuntimeError(f"SAM3_VideoTrack zwrócił {len(small)} masek zamiast {n}")
        h, w = in_hw
        from engine.comfy_io import LazySeq

        def seq(arrs):
            def mask(i: int) -> np.ndarray:
                m = arrs[i].astype(np.float32) / 255.0
                return m if m.shape[:2] == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            return LazySeq(mask, range(n))
        if not soft:
            return seq(small)
        ssoft = outs["10"]
        if len(ssoft) != n:
            raise RuntimeError(f"SAM3_VideoTrackSoft zwrócił {len(ssoft)} miękkich masek zamiast {n}")
        return seq(small), seq(ssoft)
