# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/synth.py - Synthetic regression shot for the cloud (no SH010): a felt-puppet-like object with two thin
antennae moving / scaling over a textured set, a static second object, scene-linear EXR plates, exact masks.
Also a colour-key "SAM3" for the mock Comfy client (segments the puppet colour, follows the component that
overlaps the initial mask – like the real tracker, without text)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.comfy_graph import find_nodes
from engine.depth import frame_path

PUPPET_BGR = np.array([0.15, 0.25, 0.85], np.float32)   # display colour of the puppet (reddish)
TREE_BGR = np.array([0.20, 0.60, 0.25], np.float32)


@dataclass
class SynthShot:
    plate_pattern: str
    first: int
    last: int
    w: int
    h: int
    masks: Dict[int, np.ndarray]        # puppet GT alpha (float 0..1, antialiased) per frame
    tree: np.ndarray                     # static second object (bool)
    centers: Dict[int, tuple]


def puppet_mask(w, h, cx, cy, s, ss=4):
    """Body ellipse + head + two 2-px antennae; antialiased by supersampling."""
    W, H = w * ss, h * ss
    m = np.zeros((H, W), np.uint8)
    c = (int(cx * ss), int(cy * ss))
    cv2.ellipse(m, c, (int(40 * s * ss), int(60 * s * ss)), 0, 0, 360, 255, -1)
    head = (c[0], int(c[1] - 75 * s * ss))
    cv2.circle(m, head, int(25 * s * ss), 255, -1)
    for dx in (-1, 1):
        tip = (int(head[0] + dx * 45 * s * ss), int(head[1] - 70 * s * ss))
        cv2.line(m, head, tip, 255, max(1, int(round(2 * ss))))
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


DARK_BGR = np.array([0.04, 0.04, 0.05], np.float32)


def dark_part_mask(w, h, cx, cy, s, ss=4):
    """Dark appendage under the body ("nest bottom"): part of the object, invisible to the colour-key SAM."""
    m = np.zeros((h * ss, w * ss), np.uint8)
    cv2.ellipse(m, (int(cx * ss), int((cy + 70 * s) * ss)), (int(35 * s * ss), int(22 * s * ss)), 0, 0, 360, 255, -1)
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def make_shot(root: Path, n: int = 40, w: int = 1280, h: int = 720, first: int = 1001, seed: int = 3,
              exit_frame: bool = True, dark_part: bool = False, tree_x: int = 1000) -> SynthShot:
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0) * 0.35 + 0.25
    tree = np.zeros((h, w), bool)
    tree[150:650, tree_x:tree_x + 80] = True
    pdir = Path(root) / "plates"
    pdir.mkdir(parents=True, exist_ok=True)
    masks, centers = {}, {}
    for i in range(n):
        f = first + i
        t = i / 39.0                                                # motion speed independent of n
        cx = 200 + 900 * t if not exit_frame else 120 + 1100 * t   # n=40: ends partly outside the right edge
        cy = 420 + 40 * np.sin(6 * t)
        s = 1.0 + 0.6 * t                                           # grows (approaches the camera)
        a = puppet_mask(w, h, cx, cy, s)
        img = tex.copy()
        img[tree] = TREE_BGR
        img = img * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        if dark_part:
            dk = np.clip(dark_part_mask(w, h, cx, cy, s) - a, 0, 1)
            img = img * (1 - dk[..., None]) + DARK_BGR * dk[..., None]
            a = np.clip(a + dk, 0, 1)
        lin = (img ** 2.2).astype(np.float32) * 1.5                 # pseudo scene-linear
        cv2.imwrite(str(pdir / f"shot_{f:04d}.exr"), lin)
        masks[f], centers[f] = a, (cx, cy, s)
    return SynthShot(str(pdir / "shot_####.exr").replace("\\", "/"), first, first + n - 1, w, h, masks, tree, centers)


def display(lin: np.ndarray) -> np.ndarray:
    """Inverse of the pseudo-linear encoding above (stands in for PlateToDisplay in tests)."""
    return np.clip(lin / 1.5, 0, None) ** (1 / 2.2)


def colour_key(img: np.ndarray, colour=PUPPET_BGR, tol: float = 0.12) -> np.ndarray:
    d = np.linalg.norm(img[..., :3] - colour, axis=2)
    return (d < tol).astype(np.float32)


def _component_overlapping(m: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents((m > 0.5).astype(np.uint8))
    if n <= 1:
        return np.zeros_like(m)
    ov = [((lab == k) & (ref > 0.5)).sum() for k in range(1, n)]
    if max(ov) == 0:
        return np.zeros_like(m)
    return (lab == 1 + int(np.argmax(ov))).astype(np.float32)


def _sam_view(img: np.ndarray, simulate_res: bool) -> Tuple[np.ndarray, Tuple[int, int]]:
    """What SAM3 sees: the input rescaled to 1008 x 1008 (SPEC 5.1) – thin parts melt on large crops."""
    if not simulate_res:
        return img, img.shape[:2]
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > 1008 else cv2.INTER_LINEAR
    return cv2.resize(img, (1008, 1008), interpolation=interp), (h, w)


def _back(m: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    if m.shape[:2] == hw:
        return m
    return (cv2.resize(m.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)


def _grow(m: np.ndarray, px: float) -> np.ndarray:
    """SAM halo emulation: mask widened by `px` (sub-pixel, via SDF of a 4x supersampled mask)."""
    if not px:
        return m
    from engine.segment import sdf
    h, w = m.shape
    big = cv2.resize(m.astype(np.float32), (w * 4, h * 4), interpolation=cv2.INTER_LINEAR) > 0.5
    g = (sdf(big) > -px * 4).astype(np.float32)
    return (cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)


def load_node_images(graph, nid):
    """Images a loader node gives to the graph: PlateLoadEXR (float EXR, path pattern + frame range) or
    VHS_LoadImagesPath (every image of the folder, sorted, 8-bit -> float 0..1) – BGR like cv2."""
    node = graph[nid]
    inp = node["inputs"]
    if node["class_type"] == "VHS_LoadImagesPath":
        files = sorted(p for p in Path(inp["directory"]).iterdir()
                       if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return [cv2.imread(str(p), cv2.IMREAD_COLOR).astype(np.float32) / 255.0 for p in files]
    return [cv2.imread(str(frame_path(inp["path_pattern"], f)), cv2.IMREAD_UNCHANGED)
            for f in range(inp["first_frame"], inp["last_frame"] + 1)]


def load_node_masks(graph, nid, n=None):
    """Masks (float 0..1) from PlateLoadMaskSeq (PNG pattern, frames 0..n-1) or ImageToMask(VHS_LoadImagesPath)."""
    node = graph[nid]
    if node["class_type"] == "ImageToMask":
        ch = {"blue": 0, "green": 1, "red": 2}[node["inputs"]["channel"]]
        return [im[..., ch] for im in load_node_images(graph, node["inputs"]["image"][0])]
    pat = node["inputs"]["path_pattern"]
    n = n if n is not None else 1
    return [cv2.imread(str(frame_path(pat, i)), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for i in range(n)]


def sam3_handlers(outputs, colour=PUPPET_BGR, simulate_res: bool = False, grow_px: float = 0.0, tol: float = 0.12):
    """Handlers for MockComfyClient emulating SAM3_Detect (text -> all colour components, points -> the
    component under the point) and SAM3_VideoTrack (follows the component overlapping the previous mask)."""

    def _load(graph, nid):
        return load_node_images(graph, nid)

    def detect(graph):
        (sid, sam), = find_nodes(graph, "SAM3_Detect")
        img0 = _load(graph, sam["inputs"]["image"][0])[0]
        img, hw = _sam_view(img0, simulate_res)
        col = colour
        cond = sam["inputs"].get("conditioning")
        if cond and "tree" in graph[cond[0]]["inputs"].get("text", ""):
            col = TREE_BGR
        key = colour_key(img, col, tol)
        n, lab = cv2.connectedComponents((key > 0.5).astype(np.uint8))
        comps = [(lab == k).astype(np.float32) for k in range(1, n) if (lab == k).sum() > 30]
        pos = sam["inputs"].get("positive_coords")
        if pos:
            pts = json.loads(graph[pos[0]]["inputs"]["value"])
            comps = [c for c in comps if any(c[p["y"], p["x"]] > 0 for p in pts)]
        if not sam["inputs"].get("individual_masks") and comps:
            comps = [np.clip(sum(comps), 0, 1)]
        comps = [_grow(_back(c, hw), grow_px) for c in comps]
        (save_id, save), = find_nodes(graph, "SaveImage")
        return outputs.save_masks(save_id, save["inputs"]["filename_prefix"], comps)

    def track(graph):
        nodes = find_nodes(graph, "SAM3_VideoTrack") or find_nodes(graph, "SAM3_VideoTrackSoft")
        (tid, tr), = nodes
        soft_node = tr["class_type"] == "SAM3_VideoTrackSoft"
        assert "conditioning" not in tr["inputs"], "VideoTrack must not get text (SPEC 5.1)"
        assert tr["inputs"]["max_objects"] == 1
        frames = _load(graph, tr["inputs"]["images"][0])
        init = load_node_masks(graph, tr["inputs"]["initial_mask"][0])[0]
        out, prev = [], init
        # the tracker follows what the initial mask covers (no text): its colour = median under the mask
        f0 = frames[0][..., :3]
        col = np.median(f0[init > 0.5], axis=0) if (init > 0.5).any() else colour
        if np.linalg.norm(col - TREE_BGR) < np.linalg.norm(col - colour):
            col = TREE_BGR
        else:
            col = colour
        for img0 in frames:
            img, hw = _sam_view(img0, simulate_res)
            pv = cv2.resize(prev, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) if prev.shape != img.shape[:2] else prev
            m = _component_overlapping(colour_key(img, col, tol), cv2.dilate(pv, np.ones((15, 15), np.uint8)))
            m = _grow(_back(m, hw), grow_px)
            out.append(m)
            if m.any():
                prev = m
        saves = find_nodes(graph, "SaveImage")
        res = {}
        for save_id, save in saves:
            prefix = save["inputs"]["filename_prefix"]
            if prefix.endswith("_soft"):
                # soft stand-in: the binary mask blurred (sigma 1.5) = probabilities around the edge
                res.update(outputs.save_masks(save_id, prefix, [cv2.GaussianBlur(np.asarray(m, np.float32), (0, 0), 1.5) for m in out]))
            else:
                res.update(outputs.save_masks(save_id, prefix, out))
        assert soft_node or len(saves) == 1
        return res

    return {"SAM3_Detect": detect, "SAM3_VideoTrack": track, "SAM3_VideoTrackSoft": track}


def soft_key(img: np.ndarray, colour=PUPPET_BGR, ref_dist: float = 0.45) -> np.ndarray:
    """Soft alpha from the colour distance (stands in for MatAnyone2's soft edge)."""
    d = np.linalg.norm(img[..., :3] - colour, axis=2)
    return np.clip(1.0 - d / ref_dist, 0.0, 1.0).astype(np.float32)


def matanyone_handlers(colour=PUPPET_BGR, outputs=None):
    """MatAnyone2RunKeyframes + CoreCombine stand-in: soft colour key, max with the eroded core, written by
    'PlateSaveEXR' (RGBA EXR at crop size, frame indices). Also checks the graph contract."""

    def run(graph):
        (_, ma), = find_nodes(graph, "MatAnyone2RunKeyframes")
        # node >= 1.1.0 (object_info has vram_release_frac): max_mem_frames from defaults.json "matanyone" (user
        # decision 2026-10-13: 5, 0 would be auto); older node: the official 5
        from engine.refine import ma_node_defaults
        assert ma["inputs"]["max_mem_frames"] == (ma_node_defaults()["max_mem_frames"] if "vram_release_frac" in ma["inputs"] else 5)
        kf = [int(x) for x in graph[ma["inputs"]["keyframes"][0]]["inputs"]["value"].split(",")]
        assert len(kf) <= 3 and ma["inputs"]["start_frame"] in kf
        imgs = load_node_images(graph, ma["inputs"]["images"][0])
        (_, cc), = find_nodes(graph, "MatAnyone2CoreCombine")
        cores = load_node_masks(graph, cc["inputs"]["core_masks"][0], len(imgs))
        kms = load_node_masks(graph, ma["inputs"]["keyframe_masks"][0], len(kf))
        assert len(kms) == len(kf) and all(k.shape == imgs[0].shape[:2] for k in kms)
        saves = find_nodes(graph, "PlateSaveEXR")
        sv = saves[0][1] if saves else None
        alphas = []
        soft_mode = cc["inputs"].get("core_mode") == "soft"      # CoreCombine >= 1.1.2: probabilities as the core
        for i, (img, core) in enumerate(zip(imgs, cores)):
            pc = core
            core = core > 0.5
            # MatAnyone is guided by the key masks / core: the object region is kept, colour gives the soft edge
            from engine.refine import antialiased_core
            col = colour
            if core.any():
                med = np.median(img[..., :3][core], axis=0)
                col = TREE_BGR if np.linalg.norm(med - TREE_BGR) < np.linalg.norm(med - colour) else colour
            a = np.maximum(soft_key(img, col), antialiased_core(core))
            if soft_mode and ((pc > 0.002) & (pc < 0.998)).any():
                a = np.maximum(soft_key(img, col), pc)
            if sv is None:
                alphas.append(a)
                continue
            out = frame_path(sv["inputs"]["out_pattern"], i)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.dstack([a, a, a, a]).astype(np.float32))
        if sv is None:   # 8-bit sources: SaveImage (PNG), downloaded by the Finalizer
            (save_id, save), = find_nodes(graph, "SaveImage")
            return outputs.save_masks(save_id, save["inputs"]["filename_prefix"], alphas)
        return {}
    return {"MatAnyone2RunKeyframes": run}


def write_backlit(shot: SynthShot, root: Path, wall: float = 2.0, seed: int = 5, with_object: bool = True,
                  dark_behind: bool = False) -> str:
    """Backlit (X2-like) plate: bright wall, object = silhouette; scene-linear. with_object=False: the object is not
    on the backlit exposure (SH010: Figure is absent from X2, only the trees are)."""
    rng = np.random.default_rng(seed)
    d = Path(root) / "backlit"
    d.mkdir(parents=True, exist_ok=True)
    for f, a in shot.masks.items():
        if dark_behind:   # something dark and larger than the object behind it (SH010: nest / trees behind Figure)
            big = cv2.dilate((a > 0.5).astype(np.uint8), np.ones((121, 121), np.uint8)).astype(np.float32)
            a = big if not with_object else np.maximum(a, big)
        a = a if with_object or dark_behind else np.zeros_like(a)
        lum = wall * (1.0 - a) + rng.normal(0, 0.01, a.shape)
        cv2.imwrite(str(d / f"x2_{f:04d}.exr"), np.dstack([lum, lum, lum]).astype(np.float32))
    return str(d / "x2_####.exr").replace("\\", "/")


def backlit_handlers(wall: float = 2.0):
    """BacklitCoverage + MaskRegionLimit stand-in (coverage = 1 - lum / wall, region limited by the core)."""
    from engine.refine import region_limit

    def run(graph):
        (_, bc), = find_nodes(graph, "BacklitCoverage")
        ld = graph[bc["inputs"]["backlit"][0]]["inputs"]
        (_, rl), = find_nodes(graph, "MaskRegionLimit")
        reg = graph[rl["inputs"]["region"][0]]["inputs"]
        (_, sv), = find_nodes(graph, "PlateSaveEXR")
        wall_ld = graph[bc["inputs"]["wall"][0]]["inputs"] if "wall" in bc["inputs"] else None   # Z25: external wall
        for i in range(ld["first_frame"], ld["last_frame"] + 1):
            x = cv2.imread(str(frame_path(ld["path_pattern"], i)), cv2.IMREAD_UNCHANGED)
            w_i = wall
            if wall_ld is not None:
                wimg = cv2.imread(str(frame_path(wall_ld["path_pattern"], i)), cv2.IMREAD_UNCHANGED)
                w_i = np.maximum(wimg.mean(axis=2), 0.3)
            a = np.clip(1.0 - x.mean(axis=2) / w_i, 0, 1).astype(np.float32)
            core = cv2.imread(str(frame_path(reg["path_pattern"], i)), cv2.IMREAD_GRAYSCALE) > 127
            a = region_limit(a, core, rl["inputs"]["grow_px"], rl["inputs"]["soft_px"])
            out = frame_path(sv["inputs"]["out_pattern"], i)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.dstack([a, a, a, a]).astype(np.float32))
        return {}
    return {"BacklitCoverage": run}


LTX_UI_WORKFLOW = {  # minimal stand-in of the user's LTX Alpha Gen workflow (input / output nodes are what matters)
    "nodes": [
        {"id": 1, "type": "PlateLoadEXR", "inputs": [{"name": n, "widget": {"name": n}} for n in
                                                     ("path_pattern", "first_frame", "last_frame", "crop_x", "crop_y", "crop_w", "crop_h", "scale", "bad_frames")],
         "outputs": [{"name": "images", "links": [1]}, {"name": "plate_info", "links": [2]}],
         "widgets_values": ["x_####.exr", 0, 24, 0, 0, 0, 0, 1.0, ""]},
        {"id": 2, "type": "LTXVAddGuide", "inputs": [{"name": "image", "link": 1}], "outputs": [{"name": "alpha", "links": [3]}],
         "widgets_values": []},
        {"id": 3, "type": "PlateSaveEXR", "inputs": [{"name": "rgba", "link": 3}, {"name": "plate_info", "link": 2},
                                                     {"name": "out_pattern", "widget": {"name": "out_pattern"}},
                                                     {"name": "half_float", "widget": {"name": "half_float"}}],
         "outputs": [], "widgets_values": ["y_####.exr", True]},
    ],
    "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 1, 1, 3, 1, "PLATE_INFO"], [3, 2, 0, 3, 0, "IMAGE"]],
}


def ltx_handlers(colour=PUPPET_BGR, seen=None):
    """LTX Alpha Gen stand-in: soft colour key incl. thin parts; records batch length / side for assertions."""
    def run(graph):
        (_, ld), = find_nodes(graph, "PlateLoadEXR")
        (_, sv), = find_nodes(graph, "PlateSaveEXR")
        n = ld["inputs"]["last_frame"] - ld["inputs"]["first_frame"] + 1
        for i in range(ld["inputs"]["first_frame"], ld["inputs"]["last_frame"] + 1):
            img = cv2.imread(str(frame_path(ld["inputs"]["path_pattern"], i)), cv2.IMREAD_UNCHANGED)
            if seen is not None:
                seen.append((n, img.shape[0]))
            a = soft_key(img, colour, ref_dist=0.3)
            out = frame_path(sv["inputs"]["out_pattern"], i)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.dstack([a, a, a, a]).astype(np.float32))
        return {}
    return {"LTXVAddGuide": run}
