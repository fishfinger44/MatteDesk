# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/videomama.py - stage "VideoMaMa" between stage 3 (MatAnyone2) and stage 4 (ViTMatte) (user decision 2026-10-16,
after the A/B test local_results/videomama_m2/REPORT.md).

VideoMaMa (CVPR 2026, cvlab-kaist; Stable Video Diffusion prior, one diffusion step in the VAE latent) turns a video +
a binary guide mask into an alpha matte. It runs in ComfyUI through the community node ComfyUI-VideoMaMa (@okdalto:
VideoMaMaPipelineLoader + VideoMaMaSampler); the guide is ALWAYS the mask 1+2 (SAM 3.1 core + brush corrections, a group =
its whole), the image the plate-to-mask crop.
Its matte is then put over a BASE chosen by the user (setting `base`):
  "core"      – the mask 1+2 (default; with it VideoMaMa makes the whole edge – on CLIP2_A the calmest hard edge),
  "matanyone" – the stage-3 result (its hard edge is kept inside the protect band).
Composition per frame, distances in plate px from the base edge (alpha 0.5):
  inside the base, deeper than `protect` -> base; protect band (both sides) -> linear ramp base -> max(base, vmm);
  outside, protect < d <= reach -> max(base, vmm) (the strands); beyond `reach` -> base (strays / islands cut off).
The node runs every frame it gets in ONE UNet pass, so frames go in batches (separate prompts); batches overlap and are
crossfaded – without the overlap every batch start jumps (CLIP2_A: hard-edge flicker x1.9, hair x2.8 on the boundaries;
24 / overlap 8 -> +4 %).
Measured on SH050 (RTX 4090, Comfy 0.40): full 1280 x 720 frames, batches of 24 = 0.73 s / frame (with overlap 8:
1.14 s), 16.2 GB VRAM peak (whole GPU); a 448 px crop: 0.16 s / frame, 8.1 GB.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.ltx_alpha import batches, crossfade_weight, decode_matte, sdf_px

NODE_PACK = "ComfyUI-VideoMaMa"
NODE_REPO = "https://github.com/okdalto/ComfyUI-VideoMaMa"
VMM_NODES = ("VideoMaMaPipelineLoader", "VideoMaMaSampler", "PlateLoadEXR", "ImageToMask", "MaskToImage",
             "SaveImageAdvanced")
MODELS_SUBDIR = "videomama"
# files the node loads (models_dir/videomama/...): repo, path in the repo, GB. Only these parts of SVD-xt (the node
# loads feature_extractor, image_encoder fp16, vae fp16 from the base; the UNet comes from VideoMaMa) – 7.1 GB in all
# instead of the 32.6 GB of the whole SVD-xt repo.
SVD_DIR, UNET_DIR = "stable-video-diffusion-img2vid-xt", "VideoMaMa"
VMM_FILES = [
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "model_index.json", 0.0),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "feature_extractor/preprocessor_config.json", 0.0),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "scheduler/scheduler_config.json", 0.0),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "image_encoder/config.json", 0.0),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "image_encoder/model.fp16.safetensors", 1.26),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "vae/config.json", 0.0),
    (SVD_DIR, "stabilityai/stable-video-diffusion-img2vid-xt", "vae/diffusion_pytorch_model.fp16.safetensors", 0.2),
    (UNET_DIR, "SammyLim/VideoMaMa", "unet/config.json", 0.0),
    (UNET_DIR, "SammyLim/VideoMaMa", "unet/diffusion_pytorch_model.safetensors", 6.1),
]
BASES = ("core", "matanyone")
OFFLOADS = ("auto", "on", "off")
VMM_DEFAULTS: Dict[str, Any] = {
    "base": "core",           # core = mask 1+2 | matanyone = stage-3 result (CLIP2_A: the MA base keeps MA's boiling edge)
    "reach_px": 40.0,         # how far out of the base edge VideoMaMa may add (strands); beyond = base
    "protect_px": None,       # band around the base edge where the base wins (ramp); None = auto: 0 core / 3 matanyone
    # the sliders of the other edge stages (user decision 2026-10-17): VideoMaMa REPLACES the base from reach_px outside
    # to inset_px inside its edge, blended into the deeper base over feather_px. None = the former composition
    # (VideoMaMa only adds outside the base, protect band) – objects saved before keep their results
    "inset_px": None,
    "feather_px": 0.0,
    # one band system (user decision 2026-10-22, engine/bandsys.py): outer-edge feather (0 = hard, as before) and band
    # control points shaping the band (engine/bandmap.py; [] = constant band), the sliders' reference for scaling them
    "outer_feather_px": 0.0,
    "band_points": [],
    "band_ref_out": None,
    "band_ref_in": None,
    "max_side": 1280,         # long side VideoMaMa works at (the crop is scaled DOWN only; multiples of 8)
    "batch": 24,              # frames per UNet pass
    "overlap": 8,             # frames shared by neighbouring batches (linear crossfade)
    "seed": 42,
    "offload": "auto",        # model CPU offload: auto = on when the Comfy GPU has < 20 GB (16 GB peak at 1280 px)
}
SETTING_KEYS = tuple(VMM_DEFAULTS)


def vmm_settings(obj: Optional[Dict[str, Any]] = None, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """VMM_DEFAULTS <- defaults.json "videomama" <- object field `videomama` (known keys, validated); `protect`
    = the effective protect band (auto by base)."""
    s = dict(VMM_DEFAULTS)
    for src in (defaults or {}, (obj or {}).get("videomama") or {}):
        s.update({k: v for k, v in src.items() if k in VMM_DEFAULTS and (v is not None or k in ("protect_px", "inset_px", "band_ref_out", "band_ref_in"))})
    s["base"] = s["base"] if s["base"] in BASES else "core"
    s["offload"] = s["offload"] if s["offload"] in OFFLOADS else "auto"
    s["reach_px"] = float(min(400.0, max(0.0, float(s["reach_px"]))))
    s["protect_px"] = None if s["protect_px"] in (None, "", "auto") else float(min(40.0, max(0.0, float(s["protect_px"]))))
    s["inset_px"] = None if s["inset_px"] in (None, "", "auto") else float(min(200.0, max(0.0, float(s["inset_px"]))))
    s["feather_px"] = float(min(64.0, max(0.0, float(s["feather_px"] or 0.0))))
    s["outer_feather_px"] = float(min(200.0, max(0.0, float(s["outer_feather_px"] or 0.0))))
    from engine.bandmap import band_points
    s["band_points"] = band_points({"band_points": s.get("band_points")}) if s.get("band_points") else []
    s["max_side"] = int(min(2048, max(256, int(s["max_side"]) // 8 * 8)))
    s["batch"] = int(min(64, max(4, int(s["batch"]))))
    s["overlap"] = int(max(0, min(int(s["overlap"]), s["batch"] // 2)))
    s["seed"] = int(s["seed"])
    s["protect"] = s["protect_px"] if s["protect_px"] is not None else (3.0 if s["base"] == "matanyone" else 0.0)
    from engine.objband import apply   # the shared object band unless the stage is detached (2026-10-22)
    return apply(obj, "videomama", s)


def object_settings(obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """vmm_settings with defaults.json "videomama" (project-wide defaults) under the object's own `videomama`."""
    try:
        d = json.loads((Path(__file__).resolve().parent.parent / "defaults.json").read_text(encoding="utf-8")).get("videomama")
    except (OSError, ValueError):
        d = None
    return vmm_settings(obj, d or {})


def model_sig(s: Dict[str, Any]) -> str:
    """What the RAW VideoMaMa matte depends on (a change = run the model again)."""
    d = {k: s.get(k) for k in ("max_side", "batch", "overlap", "seed")}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def settings_sig(s: Dict[str, Any]) -> str:
    """The whole stage result: the model run + the composition (base, reach, protect)."""
    d = {k: s.get(k) for k in ("base", "reach_px", "protect", "max_side", "batch", "overlap", "seed")}
    if s.get("inset_px") is not None:   # the unified sliders (absent = the former signature: old results stay current)
        d.update(inset_px=s["inset_px"], feather_px=s.get("feather_px", 0.0))
    if s.get("outer_feather_px") or s.get("band_points"):   # the band system (absent = the signature above)
        d.update(outer_feather_px=s.get("outer_feather_px", 0.0), band_points=s.get("band_points") or [],
                 band_ref=[s.get("band_ref_out"), s.get("band_ref_in")])
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- models / nodes
def models_root(models_dir: Optional[Path]) -> Optional[Path]:
    return Path(models_dir) / MODELS_SUBDIR if models_dir else None


def models_status(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Every file the node loads: {dir, repo, file, gb, present}."""
    root = models_root(models_dir)
    out = []
    for sub, repo, fn, gb in VMM_FILES:
        p = root / sub / fn if root else None
        ok = bool(p and p.exists() and p.stat().st_size > (0.9 * gb * 1e9 if gb else 0))
        out.append({"dir": sub, "repo": repo, "file": fn, "gb": gb, "present": ok})
    return out


def missing_nodes(object_info: Optional[Dict[str, Any]]) -> List[str]:
    return [n for n in VMM_NODES if not object_info or n not in object_info]


def offload_on(s: Dict[str, Any], system_stats: Optional[Dict[str, Any]]) -> bool:
    if s["offload"] != "auto":
        return s["offload"] == "on"
    try:
        vram = float(((system_stats or {}).get("devices") or [{}])[0].get("vram_total") or 0)
    except (TypeError, ValueError, IndexError):
        vram = 0.0
    return 0 < vram < 20e9


def work_side(w: int, h: int, max_side: int) -> int:
    """`max_resolution` of the node (long side, multiple of 8): the crop's own size, never upscaled."""
    return int(max(64, min(max_side, max(w, h)) // 8 * 8))


# ----------------------------------------------------------------------------- composition
def compose(base: np.ndarray, vmm: np.ndarray, protect: float, reach: float, inset: Optional[float] = None,
            feather: float = 0.0, outer_feather: float = 0.0, pts=None) -> np.ndarray:
    """Base (0..1, the mask 1+2 or the stage-3 alpha) + VideoMaMa matte -> the stage alpha (see the module doc).
    `inset` given (the unified sliders, 2026-10-17; the band system since 2026-10-22, engine/bandsys.py): VideoMaMa in
    the band reach px outside .. inset px inside the base edge, blended into the base (the stage's core) over `feather` px
    from the inner band edge and `outer_feather` px from the outer one; band points (x, y, out, in in the crop's pixels)
    shape the band; beyond the band = the base."""
    b = np.clip(base.astype(np.float32), 0, 1)
    v = np.clip(vmm.astype(np.float32), 0, 1)
    d = sdf_px(b)                       # > 0 inside the base
    if inset is not None:
        from engine.bandsys import band_widths
        from engine.bandsys import compose as band_compose
        o, i = band_widths(b, reach, inset, pts)
        return band_compose(b, v, d, o, i, feather, outer_feather)
    mx = np.maximum(b, v)
    a = b.copy()
    add = (d < -protect) & (d >= -reach)
    a[add] = mx[add]
    if protect > 0:
        band = np.abs(d) <= protect
        t = np.clip(np.abs(d) / protect, 0, 1)
        a[band] = b[band] + t[band] * (mx[band] - b[band])
    return a.astype(np.float32)


def band_u8(base: np.ndarray, s: Dict[str, Any], pts=None) -> np.ndarray:
    """Band preview of the palette: 128 = where VideoMaMa adds (max), 255 = the protect band (base wins at the edge)."""
    b = np.clip(base.astype(np.float32), 0, 1)
    d = sdf_px(b)
    out = np.zeros(d.shape, np.uint8)
    if s.get("inset_px") is not None:   # unified sliders: 128 = VideoMaMa decides, 255 = the base stays (deeper inside)
        from engine.bandsys import band_widths
        o, i = band_widths(b, s["reach_px"], s["inset_px"], pts)
        out[(d >= -o) & (d <= i)] = 128
        out[d > i] = 255
        return out
    out[(d < -s["protect"]) & (d >= -s["reach_px"])] = 128
    if s["protect"] > 0:
        out[np.abs(d) <= s["protect"]] = 255
    return out


# ----------------------------------------------------------------------------- Comfy graph
def vmm_graph(img_pattern: str, mask_pattern: str, n_frames: int, max_res: int, out_prefix: str, models_root_: str,
              seed: int = 42, offload: bool = False) -> Dict[str, Any]:
    """API graph: PlateLoadEXR (crops 0..n-1, display 0..1) + guide masks -> VideoMaMa -> 16-bit PNG mattes."""
    L = lambda nid, slot=0: [str(nid), slot]
    ld = lambda pat: {"class_type": "PlateLoadEXR", "inputs": {"path_pattern": pat, "first_frame": 0, "last_frame": n_frames - 1,
                                                               "crop_x": 0, "crop_y": 0, "crop_w": 0, "crop_h": 0}}
    root = models_root_.replace("\\", "/").rstrip("/")
    return {
        "1": ld(img_pattern),
        "2": ld(mask_pattern),
        "3": {"class_type": "ImageToMask", "inputs": {"image": L(2), "channel": "red"}},
        "4": {"class_type": "VideoMaMaPipelineLoader",
              "inputs": {"base_model_path": f"{root}/{SVD_DIR}", "unet_checkpoint_path": f"{root}/{UNET_DIR}",
                         "precision": "fp16", "enable_model_cpu_offload": bool(offload), "vae_encode_chunk_size": 4,
                         "attention_mode": "auto", "enable_vae_tiling": False, "enable_vae_slicing": True}},
        "5": {"class_type": "VideoMaMaSampler",
              "inputs": {"pipeline": L(4), "images": L(1), "masks": L(3), "seed": int(seed), "max_resolution": int(max_res),
                         "fps": 7, "motion_bucket_id": 127, "noise_aug_strength": 0.0}},
        "6": {"class_type": "MaskToImage", "inputs": {"mask": L(5)}},
        "7": {"class_type": "SaveImageAdvanced", "inputs": {"images": L(6), "filename_prefix": out_prefix, "format": "png",
                                                            "format.bit_depth": "16-bit", "format.input_color_space": "sRGB"}},
    }


class VideoMaMa:
    """Runs VideoMaMa on one crop segment. `client` = ComfyClient (queue_prompt / wait_for_prompt / download_output)."""

    def __init__(self, client: Any, io_dir: Path, settings: Dict[str, Any], models_dir: Optional[Path],
                 progress: Optional[Callable[[str, int, int], None]] = None):
        self.client = client
        self.io = Path(io_dir)
        self.s = settings
        self.root = str(models_root(models_dir) or "")
        self.progress = progress or (lambda *a: None)
        self.log: List[Dict[str, Any]] = []
        try:
            stats = client.get_system_stats()
        except Exception:
            stats = None
        self.offload = offload_on(settings, stats)

    def _run(self, g: Dict[str, Any]) -> Dict[str, Any]:
        busy = self.client.busy_error()
        if busy:
            raise RuntimeError(busy)
        sub = self.client.queue_prompt(g)
        if not sub.get("success"):
            raise RuntimeError(f"Comfy odrzucił graf VideoMaMa: {sub.get('error')}")
        res = self.client.wait_for_prompt(sub["prompt_id"], timeout_sec=2 * 3600)
        if not res.get("success"):
            raise RuntimeError(f"VideoMaMa nie powiódł się: {res.get('error')}")
        return res

    def run(self, tag: str, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
        """images: plate-to-mask crops (float32 HxWx3 BGR 0..1), masks: guide masks (HxW 0/1), one per frame ->
        mattes at the crop size, same order (batches crossfaded over the overlap)."""
        import time
        n = len(images)
        H, W = images[0].shape[:2]
        res = work_side(W, H, self.s["max_side"])
        acc = [np.zeros((H, W), np.float32) for _ in range(n)]
        wsum = [0.0] * n
        bl = batches(n, self.s["batch"], self.s["overlap"])
        for bi, (a, b) in enumerate(bl):
            nb = b - a + 1
            job = (self.io / f"vmm_{tag}_{bi:02d}").resolve()   # Comfy reads it: absolute path (other cwd)
            job.mkdir(parents=True, exist_ok=True)
            for i in range(nb):
                cv2.imwrite(str(job / f"g_{i:04d}.exr"), np.ascontiguousarray(np.clip(images[a + i], 0, 1), np.float32))
                m = (np.asarray(masks[a + i]) > 0.5).astype(np.float32)
                cv2.imwrite(str(job / f"m_{i:04d}.exr"), np.ascontiguousarray(np.dstack([m, m, m]), np.float32))
            self.progress(f"VideoMaMa: paczka {bi + 1}/{len(bl)} ({nb} kl., {res} px)", bi, len(bl))
            t0 = time.time()
            prefix = f"matte_desk_vmm/{tag}_{bi:02d}_{int(t0)}"
            r = self._run(vmm_graph(str(job / "g_####.exr").replace("\\", "/"), str(job / "m_####.exr").replace("\\", "/"),
                                    nb, res, prefix, self.root, self.s["seed"], self.offload))
            files = sorted(r.get("outputs", {}).get("7", {}).get("images", []), key=lambda i: i["filename"])
            if len(files) < nb:
                raise RuntimeError(f"VideoMaMa: paczka {bi + 1} zwróciła {len(files)} klatek z {nb}")
            self.log.append({"batch": bi, "frames": [a, b], "side": res, "offload": self.offload,
                             "seconds": round(time.time() - t0, 1)})
            for i in range(nb):
                m = decode_matte(self.client.download_output(files[i]))
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                wgt = crossfade_weight(i, nb, bi == 0, bi == len(bl) - 1, self.s["overlap"])
                acc[a + i] += m * wgt
                wsum[a + i] += wgt
            shutil.rmtree(job, ignore_errors=True)    # Comfy's input only (a failed batch keeps it for diagnosis)
        return [acc[i] / max(wsum[i], 1e-6) for i in range(n)]
