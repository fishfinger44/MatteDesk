# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – regeneration of a marked area of a key (docs/PLAN_CLEANUP_3D.md, package P5 „Dogenerowanie”).

One entry point, `regenerate`: the active version of a key + its areas (engine/cleanup3d/keys.py) -> an engine fills the
area -> a NEW version of the key. Engines (types.ENGINES): `lama` (LaMa on a crop around each group of the area, through
engine.cleanup.lama_batch_comfy) and `klein_restore` (FLUX 3 Image Klein + LoRA Restore redraw in tiles of about 2 MP
through ComfyUI; proven in removers_test/sea_regen.py mode `klr`) and `qwen21_edit` (Qwen-Image-2.1 in EDIT mode on a
crop of about 1.4 MP around each group; proven in removers_test/remove_test.py mode `q21`).

Qwen-Image-2.1 (user decision 2026-11-10, after the sh060 test – the best engine for removing an object from a key):
an instruction edit, not an inpaint – the model gets the whole crop with the object STILL VISIBLE plus the instruction
(QWEN_PROMPT + the user's description), edits the whole crop, and the result is pasted ONLY in the marked area (same
soft edge, ring colour match and `blur` hook as the other engines). The "never sees the old fill" rule below is for the
generators that copy their reference (FLUX 3 Image Klein, LaMa); FLUX 3 Image Klein + LoRA Restore must not be used to
fill a hole (it draws shapes into a neutral hole) – Qwen-Image-2.1 for large areas, LaMa for small ones.

Rules (docs/FUTURE_MODULES.md „Poprawianie klucza modelem obrazowym”, user 2026-11-09):
  * the generator never sees the old fill: inside the (grown) area the model's input holds only the low-frequency colour
    of the surroundings (`neutral_fill`: inpaint at 1/8 resolution + blur);
  * the result is pasted ONLY inside the area grown by PASTE_GROW_PX (the soft edge lies in that growth); every other
    pixel of the new version is the old version bit for bit;
  * the colour of the result is matched to the real pixels on a ring around the area;
  * FLUX 3 Image Klein copies its reference faithfully, so seams in the reference survive: `soften_seams` blurs the
    reference (about 8 px) in the area for a first pass and a second pass (reference = the first result, no blur)
    restores the sharpness;
  * a card's regeneration never enters a neighbouring card: `extent` (0..1 mask of the card's extent in the key) cuts
    both the area and its growth.
Key steps (user 2026-11-10, docs/PLAN_CLEANUP_3D.md „Kroki na kluczu”): every step starts from a chosen BASE version
(default the active one; the new version records it as `base`), and works on one of two areas – „zaznaczenie” (the
key's marked areas, above) or „cała łata” (`area_mode="patch"`: where the base differs from the plate of the key frame +
the patch band of engine/cleanup3d/band.py; FLUX 3 Image Klein + LoRA Restore then redraws the existing fill at strength
0.9 from the tile's own latent, and the result is merged with keys.patch_on_plate's rules). FLUX 3 Image Klein + LoRA
Restore takes a strength (denoise) 0.3–1.0.
Scene-linear EXR keys go to the model through ACEScct and back (engine/cleanup.py to_model / from_model); the models work
in 8 bits, so the regenerated pixels carry 8-bit precision in that encoding whatever the key's own precision is.
Every ComfyUI prompt runs inside `comfy_lease` (engine/gpu_manager.py).
"""
from __future__ import annotations

import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # before cv2 (as api/__init__.py)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402

PathLike = Union[str, Path]
Box = Tuple[int, int, int, int]            # x0, y0, x1, y1 (x1 / y1 exclusive)

# ----------------------------------------------------------------------------- constants (tests shrink the tile ones)
PASTE_GROW_PX = 4            # the area grows by this much; the soft edge of the paste lies inside the growth
NEUTRAL_GROW_PX = 4          # the neutral fill reaches this much further than the pasted region
NEUTRAL_Q = 8                # the low-frequency colour is inpainted at 1 / NEUTRAL_Q of the resolution ...
NEUTRAL_BLUR = 6.0           # ... and blurred by this sigma at full resolution (qwen_inpaint.py NEUTRAL)
RING_PX = 24                 # width of the ring of real pixels the colour is matched on
GROUP_PX = 128               # parts of the area closer than 2 x this are regenerated together
LAMA_MARGIN_PX = 256         # LaMa: context around a group (key_lama.py MARGIN)
LAMA_MAX_SIDE = 2048         # LaMa: a larger crop is scaled down for the model and back
LAMA_MODEL = "big-lama.pt"
TILE_PX = 1920 * 1088        # FLUX 3 Image Klein: about 2 MP per tile
TILE_MAX_SIDE = 2048
TILE_MULT = 16               # tile sides are multiples of this
TILE_OVERLAP = 256           # neighbouring tiles share at least this much (less for small tiles: a third of the tile)
TILE_CONTEXT_PX = 256        # context around a group inside its tiles
SOFTEN_BLUR = 8.0            # soften_seams: sigma of the reference blur in the first pass
SOFTEN_GROW_PX = 16          # ... over the pasted region grown by this much
KLEIN_STEPS = 8
KLEIN_SEED = 42
KLEIN_SHIFT = math.log(16)
# the LoRA's trigger sentence (ausboss/FLUX-3-Image-Klein-Restore-LoRA)
KLEIN_PROMPT = ("Restore the photo: remove blur, noise and compression artifacts and recover sharp, natural detail, keeping "
                "the same composition, people and colors.")
KLEIN_SAVE_NODE = "14"
KLEIN_NAME = T.ENGINE_LABEL["klein_restore"]
QWEN_PX = 1024 * 1344        # Qwen-Image-2.1: the crop goes to the model at about 1.4 MP (remove_test.py CW x CH) ...
QWEN_MULT = 32               # ... sides multiples of this
QWEN_MARGIN_PX = 256         # context around a group in its crop ...
QWEN_MARGIN_FRAC = 0.25      # ... at least this fraction of the group's longer side
QWEN_STEPS = 40
QWEN_FEATHER_PX = 64         # the crop's context blended into the image over this many px at its inner edges (the
                             # raw Qwen picture as an H3 anchor showed the crop's rectangle, 2026-11-11); the region
                             # (>= QWEN_MARGIN_PX inside) is pasted whole
QWEN_SEED = 42
QWEN_SAVE_NODE = "14"
# the default instruction (user 2026-11-10; English = the model's language); the bar's description is appended
QWEN_PROMPT = ("Remove the marked object and everything it causes (shadow, footprints, kicked-up sand). Fill its place "
               "with the background that continues naturally around it, with the same light, colours, grain and motion "
               "blur. Change nothing else.")
QWEN_NAME = T.ENGINE_LABEL["qwen21_edit"]
PROMPT_ENGINES = ("klein_restore", "qwen21_edit")    # engines that take the bar's description (stored in the version)
# FLUX 3 Image Klein + LoRA Restore strength (denoise; user 2026-11-10): 1.0 for a marked area (the graph as before),
# 0.9 for the whole patch (the tested restore chain removers_test/flicker_up2.py) – below 1 from the tile's own latent
KLEIN_STRENGTH_RANGE = (0.3, 1.0)
KLEIN_STRENGTH = {"marked": 1.0, "patch": 0.9}
AREA_MODES = ("marked", "patch")    # „zaznaczenie” = the key's areas; „cała łata” = where the base differs from the plate
PATCH_THR = 2.5 / 255.0      # the whole patch: base vs the plate of the key frame above this (any channel, model encoding)
PATCH_CLOSE_PX = 6           # ... closed by this radius, holes inside filled ...
PATCH_MIN_PX = 16            # ... specks smaller than this dropped


# ----------------------------------------------------------------------------- the gate
def check_engine(engine: str, cfg: Optional[Dict[str, Any]] = None) -> None:
    """ValueError (Polish, the gate's own reason) when `engine` is unknown or its model files are missing."""
    if engine not in T.ENGINES:
        raise ValueError(f"Nieznany silnik dogenerowania: {engine}")
    from engine.stage_models import cleanup3d_engines
    if cfg is None:
        from engine.config import load_config
        cfg = load_config()
    st = cleanup3d_engines(cfg).get(engine) or {}
    if not st.get("ok"):
        raise ValueError(st.get("reason") or f"Brak modelu {T.ENGINE_LABEL[engine]} – Doktor")


# ----------------------------------------------------------------------------- masks
def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    return m if px <= 0 else cv2.dilate(m, _disk(px))


def paste_region(area: np.ndarray, extent: Optional[np.ndarray] = None) -> np.ndarray:
    """uint8 (h, w): the pixels a regeneration may change = (area inside the extent) grown by PASTE_GROW_PX, cut by the
    extent again. `area` / `extent`: 0..1 masks (> 0.5 counts)."""
    a = np.asarray(area) > 0.5
    if extent is None:
        return _grow(a, PASTE_GROW_PX)
    e = np.asarray(extent) > 0.5
    return _grow(a & e, PASTE_GROW_PX) & e.astype(np.uint8)


def paste_alpha(region: np.ndarray) -> np.ndarray:
    """float32 0..1: 0 outside `region`, a ramp of PASTE_GROW_PX px from its border inwards (so 1 on the area itself)."""
    d = cv2.distanceTransform((np.asarray(region) > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(d / float(max(1, PASTE_GROW_PX)), 0.0, 1.0).astype(np.float32)


def _soft_inside(hole: np.ndarray, sigma: float) -> np.ndarray:
    """float32 (h, w, 1): 1 on the whole hole, soft falloff OUTSIDE it (never below 1 inside)."""
    h = (np.asarray(hole) > 0).astype(np.float32)
    return np.maximum(h, cv2.GaussianBlur(h, (0, 0), sigma))[..., None]


def neutral_fill(img: np.ndarray, hole: np.ndarray, soft: bool = True) -> np.ndarray:
    """`img` (float BGR 0..1) with `hole` replaced by the low-frequency colour of its surroundings: inpainted at
    1 / NEUTRAL_Q of the resolution, blurred – no structure of the old fill is left for the model to copy.
    soft: a soft transition just OUTSIDE the hole (never less than the full fill inside); False = the pixels outside
    the hole stay exactly as they are (LaMa continues whatever touches its mask – a smoothed rim gave a grey blob)."""
    h, w = img.shape[:2]
    hole8 = (np.asarray(hole) > 0).astype(np.uint8)
    if not hole8.any():
        return img.astype(np.float32).copy()
    sw, sh = max(1, -(-w // NEUTRAL_Q)), max(1, -(-h // NEUTRAL_Q))
    small = np.clip(np.rint(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA) * 255.0), 0, 255).astype(np.uint8)
    ms = (cv2.resize(hole8.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    ms = cv2.dilate(ms, np.ones((3, 3), np.uint8))
    if ms.all():                                   # nothing real left: the mean colour outside the hole (or of all)
        src = img[hole8 == 0] if (hole8 == 0).any() else img.reshape(-1, 3)
        lo = np.broadcast_to(src.mean(axis=0).astype(np.float32), img.shape).copy()
    else:
        lo8 = cv2.inpaint(small, ms, 6, cv2.INPAINT_TELEA)
        lo = cv2.resize(lo8.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_CUBIC)
        lo = cv2.GaussianBlur(lo, (0, 0), NEUTRAL_BLUR)
    k = _soft_inside(hole8, 3.0) if soft else (hole8 > 0).astype(np.float32)[..., None]
    return np.clip(img * (1.0 - k) + lo * k, 0.0, 1.0).astype(np.float32)


def groups(region: np.ndarray) -> Tuple[np.ndarray, List[Box]]:
    """Parts of the region closer than about 2 x GROUP_PX belong to one group. -> (labels int32 (h, w): 0 = outside the
    region, g = 1..n; [bounding box of group g's pixels])."""
    r = (np.asarray(region) > 0).astype(np.uint8)
    h, w = r.shape
    q = 8
    sw, sh = max(1, -(-w // q)), max(1, -(-h // q))
    small = (cv2.resize(r.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    small = cv2.dilate(small, _disk(max(1, GROUP_PX // q)))
    _, lab = cv2.connectedComponents(small, connectivity=8)
    lab = cv2.resize(lab.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32) * r
    out_lab = np.zeros((h, w), np.int32)
    boxes: List[Box] = []
    for g in [int(v) for v in np.unique(lab) if v > 0]:
        sel = lab == g
        ys, xs = np.nonzero(sel)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        out_lab[sel] = len(boxes)
    left = (r > 0) & (out_lab == 0)                # a region pixel the coarse labels missed (never expected)
    if left.any():
        ys, xs = np.nonzero(left)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        out_lab[left] = len(boxes)
    return out_lab, boxes


def crop_ramp(box: Box, w: int, h: int, px: int) -> np.ndarray:
    """(h, w) of the crop: 0 at its edges inside the image -> 1 at px from them (an edge on the image border stays 1)."""
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    xs, ys = np.arange(cw, dtype=np.float32), np.arange(ch, dtype=np.float32)
    big_x, big_y = np.full(cw, 1e9, np.float32), np.full(ch, 1e9, np.float32)
    dx = np.minimum(xs + 0.5 if x0 > 0 else big_x, (cw - xs - 0.5) if x1 < w else big_x)
    dy = np.minimum(ys + 0.5 if y0 > 0 else big_y, (ch - ys - 0.5) if y1 < h else big_y)
    d = np.minimum(dy[:, None], dx[None, :])
    if px <= 0:
        return np.ones((ch, cw), np.float32)
    return np.clip(d / float(px), 0.0, 1.0).astype(np.float32)


def _pad_box(b: Box, pad: int, w: int, h: int) -> Box:
    return max(0, b[0] - pad), max(0, b[1] - pad), min(w, b[2] + pad), min(h, b[3] + pad)


# ----------------------------------------------------------------------------- tiles
def tile_size(rw: int, rh: int, w: int, h: int) -> Tuple[int, int]:
    """Tile (tw, th) for a region of rw x rh in an image of w x h: multiples of TILE_MULT, inside the image, at most
    TILE_MAX_SIDE a side and about TILE_PX pixels."""
    m = TILE_MULT
    if w < m or h < m:
        raise ValueError(f"Klucz jest za mały dla {KLEIN_NAME}: {w}×{h} px (minimum {m}×{m})")
    cap = max(m, TILE_MAX_SIDE // m * m)
    tw = max(m, min(-(-rw // m) * m, w // m * m, cap))
    th = max(m, min(-(-rh // m) * m, h // m * m, cap))
    while tw * th > TILE_PX and max(tw, th) > m:
        if tw >= th:
            tw -= m
        else:
            th -= m
    return tw, th


def tile_overlap(t: int) -> int:
    return max(0, min(TILE_OVERLAP, t // 3))


def _axis(lo: int, hi: int, size: int, tile: int) -> List[int]:
    """Tile starts along one axis: the tiles cover [lo, hi) and neighbours share >= tile_overlap(tile)."""
    if hi - lo <= tile:
        return [int(np.clip((lo + hi - tile) // 2, 0, size - tile))]
    ov = tile_overlap(tile)
    n = max(2, -(-(hi - lo - ov) // (tile - ov)))
    return [int(round(lo + i * (hi - lo - tile) / (n - 1))) for i in range(n)]


def tile_boxes(box: Box, shape: Tuple[int, int]) -> List[Box]:
    """Tiles covering `box` (already padded with its context) in an image of `shape` (h, w)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    tw, th = tile_size(x1 - x0, y1 - y0, w, h)
    return [(x, y, x + tw, y + th) for y in _axis(y0, y1, h, th) for x in _axis(x0, x1, w, tw)]


def tile_weight(tw: int, th: int) -> np.ndarray:
    """float32 (th, tw): 1 in the middle, a ramp to the tile's sides over its overlap (the blend of neighbours)."""
    def ramp(n: int) -> np.ndarray:
        i = np.arange(n, dtype=np.float32)
        return np.minimum(1.0, (np.minimum(i, n - 1 - i) + 1.0) / (tile_overlap(n) + 1.0))
    return np.minimum(ramp(th)[:, None], ramp(tw)[None, :]).astype(np.float32)


def plan_tiles(region: np.ndarray) -> List[Box]:
    """Every tile of a FLUX 3 Image Klein pass: per group, tiles over its bounding box + context; tiles that hold no
    pixel of the region and repeated tiles are dropped."""
    h, w = region.shape[:2]
    _, boxes = groups(region)
    out: List[Box] = []
    for b in boxes:
        for t in tile_boxes(_pad_box(b, TILE_CONTEXT_PX, w, h), (h, w)):
            if t not in out and region[t[1]:t[3], t[0]:t[2]].any():
                out.append(t)
    return out


# ----------------------------------------------------------------------------- FLUX 3 Image Klein + LoRA Restore
def klein_graph(tile_dir: PathLike, prefix: str, tw: int, th: int, prompt: Optional[str] = None,
                seed: int = KLEIN_SEED, strength: float = 1.0,
                files: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
    """The redraw graph of removers_test/sea_regen.py (mode `klr`): the tile is the reference; at strength (denoise) 1
    sampling starts from the reference node's latent, below 1 from the tile's OWN latent (VAEEncode of the tile – the
    restore chain of removers_test/flicker_up2.py, denoise 0.9: a partial denoise must start from the tile itself, not
    from the reference node's latent). File names from engine/stage_models.py (what the Doctor installs): the variant
    on disk for this card (bf16 / int8, stage_models.klein_files) unless `files` is given."""
    from engine.comfy_io import vhs_load
    from engine.stage_models import KLEIN_LORA, klein_files
    if files is None:
        from engine.config import load_config
        md = load_config().get("models_dir")
        files = klein_files(Path(md) if md else None)
    KLEIN_FILES = files   # noqa: N806 – the rows below read like the table
    den = float(strength)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN_FILES["unet"][1], "weight_dtype": "default"}},
        "7": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": KLEIN_LORA["lora"][1],
                                                              "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["7", 0], "max_shift": KLEIN_SHIFT,
                                                            "base_shift": KLEIN_SHIFT, "width": int(tw), "height": int(th)}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN_FILES["text_encoder"][1], "type": "flux3",
                                                     "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_FILES["vae"][1]}},
        "4": vhs_load(Path(tile_dir)),
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt or KLEIN_PROMPT}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "15": {"class_type": "ReferenceMultiLatent", "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
                                                                "resolution": "keep resolution", "images.image0": ["4", 0]}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["6", 0], "positive": ["15", 0], "negative": ["15", 1],
                                                    "latent_image": ["15", 2], "seed": int(seed), "steps": KLEIN_STEPS,
                                                    "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                    "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        KLEIN_SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }
    if den < 1.0:
        g["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}}
        g["12"]["inputs"]["latent_image"] = ["10", 0]
        g["12"]["inputs"]["denoise"] = den
    return g


def check_strength(strength: float) -> float:
    """FLUX 3 Image Klein + LoRA Restore strength (denoise) inside KLEIN_STRENGTH_RANGE, else ValueError (Polish)."""
    lo, hi = KLEIN_STRENGTH_RANGE
    v = float(strength)
    if not math.isfinite(v) or not (lo - 1e-9 <= v <= hi + 1e-9):
        raise ValueError(f"Siła {KLEIN_NAME} musi być w zakresie {lo:.1f}–{hi:.1f} (jest {v:g})")
    return round(min(hi, max(lo, v)), 3)


def _comfy_image(client: Any, g: Dict[str, Any], save_node: str, name: str, tw: int, th: int) -> np.ndarray:
    """Run one image graph under the GPU lease -> its SaveImage output as float BGR 0..1 of tw x th (resized when the
    model returned another size). `name`: the engine's full model name (lease label, errors)."""
    from engine.gpu_manager import comfy_lease
    with comfy_lease(client, f"{name} (dogenerowanie klucza)"):
        sub = client.queue_prompt(g)
        if not sub.get("success"):
            raise RuntimeError(f"Comfy odrzucił graf {name}: {sub.get('error')}")
        res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=3600)
    if not res.get("success"):
        raise RuntimeError(f"{name} nie powiódł się: {res.get('error')}")
    outs = res.get("outputs", {}).get(save_node, {}).get("images", [])
    if not outs:
        raise RuntimeError(f"{name} nie zwrócił obrazu")
    o8 = cv2.imdecode(np.frombuffer(client.download_output(outs[0]), np.uint8), cv2.IMREAD_COLOR)
    if o8 is None:
        raise RuntimeError(f"{name}: zły obraz wyjściowy")
    if o8.shape[:2] != (th, tw):
        o8 = cv2.resize(o8, (tw, th), interpolation=cv2.INTER_CUBIC)
    return o8.astype(np.float32) / 255.0


def _klein_tile(client: Any, job: Path, k: int, tile: np.ndarray, prompt: Optional[str],
                strength: float = 1.0) -> np.ndarray:
    """One tile (float BGR 0..1) through the graph -> float BGR 0..1 of the same size."""
    from engine.comfy_io import write_png8
    th, tw = tile.shape[:2]
    d = job / f"t_{k:03d}"
    write_png8(d / "tile.png", tile)
    g = klein_graph(d, f"matte_desk/c3regen_{job.name}/t{k:03d}", tw, th, prompt, strength=strength)
    return _comfy_image(client, g, KLEIN_SAVE_NODE, KLEIN_NAME, tw, th)


def _median_shift(out: np.ndarray, ref: np.ndarray, sel: np.ndarray, min_px: int = 64) -> np.ndarray:
    """`out` shifted per channel so that its median on `sel` equals the median of `ref` there (no change when `sel`
    is too small to trust)."""
    if int(sel.sum()) < min_px:
        return out
    o = out.copy()
    for c in range(3):
        o[..., c] += float(np.median(ref[..., c][sel]) - np.median(out[..., c][sel]))
    return o


def klein_pass(client: Any, job: Path, ref: np.ndarray, real: np.ndarray, known: np.ndarray, tiles: Sequence[Box],
               prompt: Optional[str], step: Callable[[], None], first: int = 0,
               strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """One pass over `tiles`: reference `ref` (model encoding) -> (generated image, weight sum; 0 = no tile there).
    Each tile's colour is shifted to the real pixels it holds (`real` where `known`), tiles are blended with ramps."""
    h, w = ref.shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for k, (x0, y0, x1, y1) in enumerate(tiles):
        out = _klein_tile(client, job, first + k, ref[y0:y1, x0:x1], prompt, strength)
        out = _median_shift(out, real[y0:y1, x0:x1], known[y0:y1, x0:x1])
        wt = tile_weight(x1 - x0, y1 - y0)
        acc[y0:y1, x0:x1] += out * wt[..., None]
        wsum[y0:y1, x0:x1] += wt
        step()
    return acc / np.maximum(wsum, 1e-6)[..., None], wsum


def soften_reference(ref: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The reference blurred by SOFTEN_BLUR inside the region grown by SOFTEN_GROW_PX (soft transition around it)."""
    zw = _soft_inside(_grow(region, SOFTEN_GROW_PX), SOFTEN_BLUR)
    return (ref * (1.0 - zw) + cv2.GaussianBlur(ref, (0, 0), SOFTEN_BLUR) * zw).astype(np.float32)


def _run_klein(client: Any, io_dir: Path, neutral: np.ndarray, real: np.ndarray, region: np.ndarray, hole: np.ndarray,
               soften: bool, prompt: Optional[str], progress: Optional[Callable[[int, int], None]],
               strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """`neutral` = the reference: the key with the hole neutral-filled, or – in the "patch" mode – the key itself (a
    redraw of the existing fill, `hole` empty = every pixel is known)."""
    from engine.comfy_io import require_vhs
    require_vhs(client)                            # the tile reaches ComfyUI through VHS_LoadImagesPath
    tiles = plan_tiles(region)
    total = len(tiles) * (2 if soften else 1)
    done = [0]

    def step() -> None:
        done[0] += 1
        if progress:
            progress(done[0], total)

    known = hole == 0
    job = Path(io_dir) / f"c3regen_{uuid.uuid4().hex[:8]}"
    try:
        ref = soften_reference(neutral, region) if soften else neutral
        gen, wsum = klein_pass(client, job, ref, real, known, tiles, prompt, step, strength=strength)
        if soften:                                 # second pass: the first result is the reference, no blur
            k = _soft_inside(hole, 3.0) * (wsum > 0)[..., None]
            ref2 = (neutral * (1.0 - k) + gen * k).astype(np.float32)
            gen, wsum = klein_pass(client, job, ref2, real, known, tiles, prompt, step, first=len(tiles),
                                   strength=strength)
    finally:
        shutil.rmtree(job, ignore_errors=True)
    return gen, wsum, len(tiles)


# ----------------------------------------------------------------------------- Qwen-Image-2.1 (edit)
PROMPT_MODES = ("remove", "replace")
# „Zastąp” (Cleanup step C1, user 2026-11-10): the description names the object and what goes in its place; the remove
# instruction ("fill with the background… change nothing else") fought such a description
QWEN_REPLACE_PROMPT = ("Replace {what}. Match the light, perspective, colours, grain and motion blur of the image. "
                       "Change nothing else.")


def qwen_prompt(description: Optional[str] = None, mode: str = "remove", what: Optional[str] = None) -> str:
    """The instruction for Qwen-Image-2.1: remove = QWEN_PROMPT plus the user's description of what should be in the
    area; replace = QWEN_REPLACE_PROMPT around the description (empty description: the remove instruction).
    `what` (user 2026-11-15, an earring stayed in every anchor): the removed object's noun. An edit model gets the real
    image with nothing marked on it – it removes only what the instruction NAMES (the video models never get the name:
    they would paint it)."""
    d = (description or "").strip()
    if mode == "replace" and d:
        return QWEN_REPLACE_PROMPT.format(what=d.rstrip("."))
    base = QWEN_PROMPT if not what else QWEN_PROMPT.replace("Remove the marked object", f"Remove the {what.strip()} (the marked object)", 1)
    if not d:
        return base
    return f"{base} In its place there should be: {d.rstrip('.')}."


def qwen_crop(box: Box, w: int, h: int) -> Box:
    """The crop the model edits for one group: its bounding box + QWEN_MARGIN_PX (at least QWEN_MARGIN_FRAC of the
    group's longer side) of context, clamped to the key."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = max(QWEN_MARGIN_PX, int(round(QWEN_MARGIN_FRAC * max(bw, bh))))
    return _pad_box(box, pad, w, h)


def qwen_model_size(cw: int, ch: int) -> Tuple[int, int]:
    """(mw, mh): the crop scaled (up or down) to about QWEN_PX pixels, sides multiples of QWEN_MULT."""
    m = QWEN_MULT
    s = math.sqrt(QWEN_PX / float(max(1, cw * ch)))
    return max(m, int(round(cw * s / m)) * m), max(m, int(round(ch * s / m)) * m)


QWEN_MASK_GROW_PX = 6      # model px the latent noise mask is grown by (the mask's soft edge / the latent's 8 px grid)


def qwen_graph(crop_dir: PathLike, prefix: str, prompt: str, seed: int = QWEN_SEED,
               ref_dir: Optional[PathLike] = None, denoise: float = 1.0,
               mask_dir: Optional[PathLike] = None, soft_mask: bool = False) -> Dict[str, Any]:
    """The edit graph of removers_test/remove_test.py (mode q21): the whole crop (object still visible) is the image of
    the instruction AND the latent the sampler starts from (VAEEncode, denoise 1). File names from engine/stage_models.py
    (what the Doctor checks)."""
    from engine.comfy_io import vhs_load
    from engine.stage_models import QWEN21_FILES
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN21_FILES["unet"][1], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN21_FILES["text_encoder"][1], "type": "qwen_image",
                                                     "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN21_FILES["vae"][1]}},
        "4": vhs_load(Path(crop_dir)),
        "8": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": "",
                                                                "vae": ["3", 0], "resolution": 0,
                                                                "images.image_1": ["4", 0]}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1],
                                                    "latent_image": ["10", 0], "seed": int(seed), "steps": QWEN_STEPS,
                                                    "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                    "denoise": float(denoise)}},   # < 1: a touch-up of the image (2026-11-11)
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        QWEN_SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
    }
    if ref_dir is not None:   # a second picture (the H3 main anchor: the look to copy, 2026-11-11)
        g["5"] = vhs_load(Path(ref_dir))
        g["8"]["inputs"]["images.image_2"] = ["5", 0]
    if mask_dir is not None:
        # inpainting (user 2026-11-15, an earring stayed in the anchors): only the latent under the mask is noised and
        # redrawn – the model knows WHERE, the rest of the crop stays; native nodes only (ImageToMask, GrowMask,
        # SetLatentNoiseMask – the „Qwen Image 2.1 Inpainting” recipe), no extra download
        g["20"] = vhs_load(Path(mask_dir))
        g["21"] = {"class_type": "ImageToMask", "inputs": {"image": ["20", 0], "channel": "red"}}
        g["22"] = {"class_type": "GrowMask", "inputs": {"mask": ["21", 0], "expand": QWEN_MASK_GROW_PX, "tapered_corners": True}}
        g["23"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["10", 0], "mask": ["22", 0]}}
        g["12"]["inputs"]["latent_image"] = ["23", 0]
        if soft_mask:   # a feathered mask (the anchors, user 2026-11-23): each pixel denoised as far as its mask value –
            # Qwen-Image-2.1 fades into the plate instead of drawing a hard outline at the mask edge (native node)
            g["24"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}}
            g["12"]["inputs"]["model"] = ["24", 0]
    return g


def _run_qwen(client: Any, io_dir: Path, real: np.ndarray, region: np.ndarray, prompt: Optional[str],
              progress: Optional[Callable[[int, int], None]], mode: str = "remove", seed: int = QWEN_SEED,
              ref: Optional[np.ndarray] = None, text: Optional[str] = None,
              denoise: float = 1.0, feather_px: int = 0, what: Optional[str] = None,
              latent_mask: bool = False, mask_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Qwen-Image-2.1 edit on a crop around each group (user 2026-11-10): the model gets the REAL crop (the object still
    visible – an edit model removes what the instruction names) scaled to about QWEN_PX, edits the whole crop; the
    output, scaled back, covers its crop in the generated image (so the ring around the area can be colour-matched),
    and only the region's pixels are pasted afterwards."""
    from engine.comfy_io import require_vhs, write_png8
    require_vhs(client)                            # the crop reaches ComfyUI through VHS_LoadImagesPath
    h, w = region.shape[:2]
    lab, boxes = groups(region)
    gen = real.copy()
    wsum = np.zeros((h, w), np.float32)
    text = text or qwen_prompt(prompt, mode, what)
    job = Path(io_dir) / f"c3regen_{uuid.uuid4().hex[:8]}"
    try:
        for g, b in enumerate(boxes, start=1):
            x0, y0, x1, y1 = qwen_crop(b, w, h)
            crop = real[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            mw, mh = qwen_model_size(cw, ch)
            down = mw * mh < cw * ch
            small = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_AREA if down else cv2.INTER_CUBIC)
            d = job / f"c_{g:03d}"
            write_png8(d / "crop.png", np.clip(small, 0.0, 1.0))
            rd = None
            if ref is not None:   # the reference crop: the same box of the main anchor (its pose differs a little)
                rd = job / f"r_{g:03d}"
                rc = cv2.resize(ref[y0:y1, x0:x1], (mw, mh), interpolation=cv2.INTER_AREA if down else cv2.INTER_CUBIC)
                write_png8(rd / "ref.png", np.clip(rc, 0.0, 1.0))
            md = None
            if latent_mask:   # the area to redraw, as a picture of the crop (white = redraw; the anchors, 2026-11-15)
                md = job / f"m_{g:03d}"
                src = region if mask_weight is None else mask_weight   # soft: 0..1 = how far to redraw
                mc = cv2.resize(src[y0:y1, x0:x1].astype(np.float32), (mw, mh), interpolation=cv2.INTER_AREA)
                write_png8(md / "mask.png", np.repeat(np.clip(mc, 0.0, 1.0)[..., None], 3, axis=2))
            gr = qwen_graph(d, f"matte_desk/c3regen_{job.name}/c{g:03d}", text, seed, rd, denoise, md,
                            soft_mask=md is not None and mask_weight is not None)
            out = _comfy_image(client, gr, QWEN_SAVE_NODE, QWEN_NAME, mw, mh)
            out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC if down else cv2.INTER_AREA)
            fill = (wsum[y0:y1, x0:x1] == 0) | (lab[y0:y1, x0:x1] == g)   # own region always; context: first crop
            ramp = crop_ramp((x0, y0, x1, y1), w, h, feather_px)   # 0 = the crop whole (as before)
            ramp[lab[y0:y1, x0:x1] == g] = 1.0
            r3 = ramp[..., None]
            blend = out * r3 + gen[y0:y1, x0:x1] * (1.0 - r3)
            gen[y0:y1, x0:x1][fill] = blend[fill]
            wsum[y0:y1, x0:x1] = 1.0
            if progress:
                progress(g, len(boxes))
    finally:
        shutil.rmtree(job, ignore_errors=True)
    return gen, wsum, len(boxes)


# ----------------------------------------------------------------------------- LaMa
def _run_lama(client: Any, io_dir: Path, neutral: np.ndarray, region: np.ndarray, mask: np.ndarray,
              progress: Optional[Callable[[int, int], None]], cfg: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """LaMa on a crop around each group (LAMA_MARGIN_PX of context; a crop over LAMA_MAX_SIDE is scaled down and back).
    `mask` = everything LaMa fills (the whole grown area, also its part outside the card's extent, so no old fill is
    context); `region` = what is pasted. The mask alone is enough for LaMa (it blanks the masked pixels itself, checked
    on sh060 2026-11-09); the crop is neutral-filled under the mask anyway and untouched outside it.
    The crop goes through LaMa at its FULL resolution (engine/lama_full.py, a worker in ComfyUI's Python; up to
    lama_full.MAX_SIDE) when that is available; else / when the worker fails, through the ComfyUI node (256 x 256)."""
    from engine import lama_full
    from engine.cleanup import lama_batch_comfy
    full = lama_full.available(cfg)
    h, w = region.shape[:2]
    lab, boxes = groups(region)
    gen = neutral.copy()
    wsum = np.zeros((h, w), np.float32)
    for g, b in enumerate(boxes, start=1):
        x0, y0, x1, y1 = _pad_box(b, LAMA_MARGIN_PX, w, h)
        crop, m = neutral[y0:y1, x0:x1], (mask[y0:y1, x0:x1] > 0).astype(np.float32)
        ch, cw = crop.shape[:2]
        s = LAMA_MAX_SIDE / float(max(ch, cw))
        out = None
        if full:
            try:
                out = lama_full.inpaint(crop, m, cfg)
            except lama_full.LamaFullError:
                full = False                           # the node for this and the remaining groups
        if out is not None:
            pass
        elif s < 1.0:
            sw, sh = max(8, int(round(cw * s))), max(8, int(round(ch * s)))
            small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            ms = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
            out = lama_batch_comfy(client, Path(io_dir), [small], [ms], LAMA_MODEL)[0]
            out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)
        else:
            out = lama_batch_comfy(client, Path(io_dir), [crop], [m], LAMA_MODEL)[0]
        sel = lab[y0:y1, x0:x1] == g
        gen[y0:y1, x0:x1][sel] = out[sel]
        wsum[y0:y1, x0:x1] = 1.0                   # the crop (also its untouched surroundings = the ring) is valid
        if progress:
            progress(g, len(boxes))
    return gen, wsum, len(boxes)


# ----------------------------------------------------------------------------- colour on the ring
def ring_match(gen: np.ndarray, real: np.ndarray, region: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per group: the generated pixels shifted so that the generated image and the real key agree (median per channel)
    on a ring of RING_PX px just outside the region (only where the engine produced pixels: `valid`)."""
    out = gen.copy()
    lab, boxes = groups(region)
    h, w = region.shape[:2]
    near = _grow(region, 2) > 0
    for g, b in enumerate(boxes, start=1):
        x0, y0, x1, y1 = _pad_box(b, RING_PX + 4, w, h)
        mine = (lab[y0:y1, x0:x1] == g).astype(np.uint8)
        ring = (_grow(mine, RING_PX + 2) > 0) & ~near[y0:y1, x0:x1] & valid[y0:y1, x0:x1]
        if int(ring.sum()) < 64:
            continue
        sub = out[y0:y1, x0:x1]
        for c in range(3):
            d = float(np.median(real[y0:y1, x0:x1, c][ring]) - np.median(gen[y0:y1, x0:x1, c][ring]))
            sub[..., c][mine > 0] += d
    return out


# ----------------------------------------------------------------------------- the whole patch („cała łata”)
def patch_hole(base: np.ndarray, plate: np.ndarray, linear: bool = False) -> np.ndarray:
    """bool (h, w): the patch of a key version = where it differs from the plate of the key frame by more than PATCH_THR
    (any channel, in the model encoding – ACEScct for linear EXR keys), closed by PATCH_CLOSE_PX, holes inside filled,
    specks under PATCH_MIN_PX dropped. Float BGR inputs of the same size (key encoding)."""
    from engine.cleanup import to_model
    b, p = np.asarray(base, np.float32), np.asarray(plate, np.float32)
    if b.shape[:2] != p.shape[:2]:
        raise ValueError(f"Plate {p.shape[1]}×{p.shape[0]} ma inny rozmiar niż klucz {b.shape[1]}×{b.shape[0]}")
    d = np.abs(to_model(b, linear) - to_model(p, linear)).max(axis=2) > PATCH_THR
    m = d.astype(np.uint8)
    if not m.any():
        return m > 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(PATCH_CLOSE_PX))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= PATCH_MIN_PX
    m = keep[lab].astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, thickness=cv2.FILLED)          # holes inside the patch belong to it
    return out > 0


def patch_weight(base: np.ndarray, plate: np.ndarray, out_px: float, feather_px: Optional[float] = None,
                 pts: Optional[Sequence[Tuple[float, ...]]] = None, linear: bool = False) -> np.ndarray:
    """float32 0..1 (h, w): the whole patch of a key version as the render blends a patch – 1 in `patch_hole`,
    the patch band of engine/cleanup3d/band.py outside it (outer offset `out_px` = „Wtopienie łaty” key_soft_px, outer
    feather `feather_px`, shaped by the band points `pts` of the key frame when there are any). No other band rule."""
    from engine.cleanup3d import band as C3B
    hole = patch_hole(base, plate, linear)
    if not hole.any():
        return np.zeros(hole.shape, np.float32)
    return C3B.patch_weight(hole, hole, float(out_px), feather_px, pts)[0]


# ----------------------------------------------------------------------------- entry point
def regenerate(work: PathLike, card: str, frame: int, engine: str, *, client: Any, io_dir: PathLike,
               soften_seams: bool = False, prompt: Optional[str] = None, extent: Optional[np.ndarray] = None,
               progress: Optional[Callable[[int, int], None]] = None,
               cfg: Optional[Dict[str, Any]] = None,
               blur: Optional[Callable[..., np.ndarray]] = None,
               base: Optional[str] = None, area_mode: str = "marked", patch: Optional[np.ndarray] = None,
               plate: Optional[np.ndarray] = None, strength: Optional[float] = None) -> Dict[str, Any]:
    """Regenerate an area of key `card` / `frame` with `engine` (types.ENGINES), starting from version `base` (None =
    the active one), and store the result as a NEW, active version of the key (`source: "regen:<engine>"`, `base`
    = the version it started from, same precision as that version).

    area_mode "marked" („zaznaczenie”): the key's areas (areas.json); the model never sees the old fill (except the
    edit model Qwen-Image-2.1), the result is pasted in the area grown by PASTE_GROW_PX, colour-matched on a ring.
    area_mode "patch" („cała łata”, user 2026-11-10): the whole patch of the base version – `patch` = its 0..1 weight
    (`patch_weight`: where the base differs from the plate of the key frame + the patch band; computed by the caller),
    `plate` = that plate frame (float BGR, key size and encoding). Every engine works on the patch area; FLUX 3 Image
    Klein + LoRA Restore REDRAWS the existing fill (the key itself is its reference, no neutral hole). The result is
    merged with keys.patch_on_plate's rules – the patch weight as the soft edge, the tone matched to the plate on the
    ring, the motion blur through `blur` – and only the pixels of the patch change; every other pixel stays the base.

    client: ComfyClient; io_dir: a folder ComfyUI can read (tiles / crops, removed afterwards); soften_seams:
    `klein_restore` in "marked" only; prompt: `klein_restore` (None = the LoRA's trigger sentence) and `qwen21_edit`
    (the user's description of what should be in the area, appended to QWEN_PROMPT); strength: `klein_restore` denoise
    in KLEIN_STRENGTH_RANGE (None = KLEIN_STRENGTH of the mode: 1.0 marked, 0.9 patch; below 1 the sampler starts from
    the tile's own latent); extent: 0..1 mask (key size) of the card's extent – nothing outside it changes;
    progress(done, total) per tile / crop; cfg: engine.config.load_config() when None; blur(gen, real, hole, known) ->
    gen: the motion blur of the generated pixels matched to the plate of the key frame (render.match_patch_blur through
    the API; `known` = where the model's output covers the real pixels).
    ValueError (Polish): unknown engine / area mode, bad strength, missing model (the gate's reason), no key, no such
    base version, no area (no patch), area outside the extent.
    -> {"version", "engine", "base", "area", "strength" (klein_restore), "tiles", "passes", "seconds", "area_px"}."""
    t0 = time.time()
    work = Path(work)
    check_engine(engine, cfg)
    if area_mode not in AREA_MODES:
        raise ValueError(f"Nieznany obszar dogenerowania: {area_mode} (dozwolone: zaznaczenie, cała łata)")
    klein = engine == "klein_restore"
    den = check_strength(KLEIN_STRENGTH[area_mode] if strength is None else strength) if klein else None
    key = store.load_key(work, card, frame)
    if key is None or not key.versions:
        raise ValueError(f"Brak klucza {card} / klatka {int(frame)}")
    base_v = KS._version(key, base)["v"]                            # ValueError when there is no such version
    img, fmt = KS.read_image(KS.version_path(work, key, base_v))    # float32 BGR of the base version + its precision
    h, w = img.shape[:2]
    if extent is not None:
        extent = np.asarray(extent)
        if extent.ndim == 3:
            extent = extent[..., 0]
        if extent.shape != (h, w):
            raise ValueError(f"Zasięg karty ma inny rozmiar niż klucz: {extent.shape[1]}×{extent.shape[0]}, "
                             f"klucz {w}×{h}")
    ext_ok = (lambda m: m & (extent > 0.5)) if extent is not None else (lambda m: m)

    weight = None
    if area_mode == "patch":
        if patch is None or plate is None:
            raise ValueError("Tryb „cała łata” potrzebuje łaty i plate'u klatki klucza")
        weight = np.asarray(patch, np.float32)
        if weight.ndim == 3:
            weight = weight[..., 0]
        plate = KS._bgr3(KS._to_float(np.asarray(plate)))
        if weight.shape != (h, w) or plate.shape[:2] != (h, w):
            raise ValueError(f"Łata / plate mają inny rozmiar niż klucz {w}×{h}")
        if extent is not None:
            weight = weight * (extent > 0.5)
        area = weight > 0
        if not area.any():
            raise ValueError(f"Wersja {base_v} klucza {card} / klatka {int(frame)} nie ma łaty (nie różni się od plate'u"
                             f"{' w zasięgu karty' if extent is not None else ''}) – nie ma czego dogenerować")
        region = area.astype(np.uint8)                              # the patch band is the soft edge: no extra growth
        area_px = int(area.sum())
    else:
        area = KS.areas_mask(work, card, frame, (h, w)) > 0.5
        if not area.any():
            raise ValueError(f"Brak zaznaczonego obszaru do dogenerowania – zaznacz obszar w kluczu {card} / klatka {int(frame)}")
        if extent is not None and not ext_ok(area).any():
            raise ValueError(f"Zaznaczony obszar leży poza zasięgiem karty {card} – nie ma czego dogenerować")
        region = paste_region(area, extent)                         # the only pixels that may change
        area_px = int(ext_ok(area).sum())

    from engine.cleanup import from_model, to_model
    linear = fmt == "exr"
    real = to_model(img, linear)                                    # 0..1 in the model's encoding
    if area_mode == "patch":
        marked = region.copy()
        hole = marked
    else:
        # the model never sees the old fill: the WHOLE marked area (also its part outside the extent) + a margin is neutral
        marked = _grow(area, PASTE_GROW_PX) | region
        hole = marked if engine == "lama" else _grow(marked, NEUTRAL_GROW_PX)   # LaMa: exactly its mask

    passes = 1
    soften = bool(soften_seams) and klein and area_mode == "marked"
    if engine == "qwen21_edit":                                     # an edit model: sees the object, no neutral fill
        from engine.stage_models import comfy_gate
        why = comfy_gate(client, "qwen21_edit")                     # an older ComfyUI: a clear reason, not Comfy's error
        if why:
            raise ValueError(why)
        gen, wsum, tiles = _run_qwen(client, Path(io_dir), real, region, prompt, progress)
    elif klein:
        if area_mode == "patch":                                    # a redraw: the existing fill is the reference
            gen, wsum, tiles = _run_klein(client, Path(io_dir), real, real, region, np.zeros_like(region), False,
                                          prompt, progress, den)
        else:
            neutral = neutral_fill(real, hole, soft=True)
            gen, wsum, tiles = _run_klein(client, Path(io_dir), neutral, real, region, hole, soften, prompt, progress,
                                          den)
        passes = 2 if soften else 1
    else:
        neutral = neutral_fill(real, hole, soft=False)
        gen, wsum, tiles = _run_lama(client, Path(io_dir), neutral, region, marked, progress, cfg)

    if area_mode == "patch":
        fill = img.copy()                                           # where no model output landed: the base itself
        cov = wsum > 0
        fill[cov] = from_model(np.clip(gen, 0.0, 1.0), linear)[cov]
        hook = None
        if blur is not None:
            def hook(fl, pl, hl):
                return np.asarray(blur(fl, pl, hl, cov), np.float32)
        core = weight >= 1.0 - 1e-6                                 # the patch hole; the band is its soft edge
        if not core.any():
            core = area
        # the tone ring lies just outside the hole, over the band (as in rebase_on_plate: the ring overlaps the edge)
        merged = KS.patch_on_plate(plate, fill, core.astype(np.float32), 0.0, 0.0, tone=True, blur=hook, weight=weight)
        out = img.copy()
        out[area] = merged[area]                                    # everything else stays the base bit for bit
    else:
        if blur is not None:
            gen = np.asarray(blur(gen, real, hole > 0, wsum > 0), np.float32)
        gen = ring_match(gen, real, region, (wsum > 0) & (hole == 0))
        a = paste_alpha(region)[..., None]
        sel = a[..., 0] > 0
        res = from_model(np.clip(real * (1.0 - a) + gen * a, 0.0, 1.0), linear)
        out = img.copy()
        out[sel] = res[sel]                                         # everything else stays bit for bit
    extra = {"area": area_mode if area_mode != "marked" else None, "strength": den if klein else None}
    key = KS.add_version(work, card, frame, out, f"regen:{engine}", fmt,
                         prompt=prompt if engine in PROMPT_ENGINES else None, base=base_v, extra=extra)
    res = {"version": key.active, "engine": engine, "base": base_v, "area": area_mode, "tiles": int(tiles),
           "passes": passes, "seconds": round(time.time() - t0, 2), "area_px": area_px}
    if klein:
        res["strength"] = den
    return res
