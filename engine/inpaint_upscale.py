# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„⤴ Upscale (SeedVR2)” of an Inpaint (C1) version (user decision 2026-11-11): the video models work at their canvas
(MiniMax-H3 768 short edge, Wan 2.2 Fun VACE 720 / 480) and their fill is scaled up to the plate – soft. SeedVR2
restores it as ONE clip (consistent texture): one crop around what the version filled in every frame (its source maps),
taken from the version's comp at the model's scale, upscaled back to the plate's scale. The result goes through the
same Cleaner as a key fill with the version's settings – the patch, the mask, the holdout and the output format are
those of a normal version; only the fill inside the mask changes. A new version, the old one stays."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LABEL = "SeedVR2"
MARGIN_PX = 48              # plate px of context around the filled area (SeedVR2 needs some)


def version_frames(vdir: Path) -> List[int]:
    """The frames the version wrote (comp/c_####.*)."""
    out = []
    for p in sorted((Path(vdir) / "comp").glob("c_*.*")):
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def comp_file(vdir: Path, f: int) -> Optional[Path]:
    return next(iter(sorted((Path(vdir) / "comp").glob(f"c_{int(f):04d}.*"))), None)


def filled_region(vdir: Path, f: int, shape: Tuple[int, int]) -> np.ndarray:
    """uint8 0/1 – where the version did not keep the plate (its source map: 0 = plate)."""
    p = Path(vdir) / "source" / f"s_{int(f):04d}.png"
    s = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else None
    if s is None:
        return np.zeros(shape, np.uint8)
    if s.ndim == 3:
        s = s[..., 0]
    if s.shape[:2] != shape:
        s = cv2.resize(s, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (s > 0).astype(np.uint8)


def crop_box(regions: Sequence[np.ndarray], shape: Tuple[int, int], margin: int = MARGIN_PX,
             mult: int = 16) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, w, h): the union of the regions + margin, inside the frame, sides multiples of `mult`; None = empty."""
    h, w = shape
    union = np.zeros(shape, np.uint8)
    for r in regions:
        union |= r
    ys, xs = np.nonzero(union)
    if not len(xs):
        return None
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin)
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin)
    cw = min(w, ((x1 - x0 + mult - 1) // mult) * mult)
    ch = min(h, ((y1 - y0 + mult - 1) // mult) * mult)
    x0, y0 = min(x0, w - cw), min(y0, h - ch)
    return x0, y0, cw, ch


def model_scale(model: str, settings: Dict[str, Any], w: int, h: int) -> float:
    """Plate -> model canvas scale of the version's engine (<= 1)."""
    if model == "vace":
        from engine import vace_fill
        cw, ch = vace_fill.canvas(w, h, int(settings.get("vace_res") or 720))
    elif model == "ltx":
        from engine import ltx_fill
        cw, ch = ltx_fill.canvas(w, h)
    else:
        from engine import h3_fill
        cw, ch = h3_fill.canvas(w, h)
    return min(1.0, max(cw, ch) / float(max(w, h)))


class UpFill:
    """The upscaled crop as the Cleaner's key fill: f -> (the version's comp with the crop replaced, in the plate's
    encoding; weight 1 inside the crop)."""
    parallel = True            # read-only: safe for the Cleaner's frame threads

    label = LABEL
    # grain match in the Cleaner (engine/cleanup.Cleaner._grain, user 2026-11-13): SeedVR2 restores a clean picture;
    # measured inside the crop beyond the patch, where the upscaled comp shows the plate's own background
    grain = True
    tone_gap_px = 0.0

    def __init__(self, vdir: Path, box: Tuple[int, int, int, int], up: Dict[int, np.ndarray], linear: bool):
        self.vdir, self.box, self.up, self.linear = Path(vdir), box, up, bool(linear)

    def __call__(self, f: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        from engine.cleanup3d.void_engine import from_display
        up = self.up.get(int(f))
        p = comp_file(self.vdir, f)
        if up is None or p is None:
            return None
        base = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if base is None:
            return None
        base = _to_f32(base[..., :3] if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
        x0, y0, cw, ch = self.box
        img = base.copy()
        img[y0:y0 + ch, x0:x0 + cw] = np.asarray(from_display(up, self.linear), np.float32)
        wgt = np.zeros(base.shape[:2], np.float32)
        wgt[y0:y0 + ch, x0:x0 + cw] = 1.0
        return img, wgt


def _to_f32(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def upscale(vdir: Path, frames: Sequence[int], model: str, settings: Dict[str, Any], linear: bool, runner: Any,
            ffmpeg: str, work_tmp: Path, progress: Optional[Callable[[str, int, int], None]] = None,
            cancel: Optional[Callable[[], None]] = None, client: Any = None) -> UpFill:
    """SeedVR2 over the crop of the version's comp (at the model's scale) -> the UpFill for the Cleaner."""
    from engine.cleanup3d.void_engine import _batch, encode_video, read_frames, to_display8
    frames = [int(f) for f in frames]
    first = cv2.imread(str(comp_file(vdir, frames[0])), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Wersja {Path(vdir).name}: brak klatek comp")
    h, w = first.shape[:2]
    regions = []
    for i, f in enumerate(frames):
        if progress:
            progress(f"{LABEL}: obszar wypełnienia F{f}", i, len(frames))
        regions.append(filled_region(vdir, f, (h, w)))
    box = crop_box(regions, (h, w))
    if box is None:
        raise ValueError(f"Wersja {Path(vdir).name} niczego nie wypełniła – nie ma czego podbijać")
    x0, y0, cw, ch = box
    s = model_scale(model, settings, w, h)
    iw, ih = max(16, int(round(cw * s / 2)) * 2), max(16, int(round(ch * s / 2)) * 2)
    root = Path(work_tmp)
    if root.exists():
        shutil.rmtree(root)
    src_dir, out = root / "clip", root / "out"
    src_dir.mkdir(parents=True)
    for i, f in enumerate(frames):
        if cancel:
            cancel()
        im = cv2.imread(str(comp_file(vdir, f)), cv2.IMREAD_UNCHANGED)
        im = _to_f32(im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        crop = to_display8(im[y0:y0 + ch, x0:x0 + cw], linear)
        cv2.imwrite(str(src_dir / f"{i:06d}.png"), cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA))
    clip = root / "clip.mkv"
    encode_video(ffmpeg, src_dir / "%06d.png", clip, True, codec="ffv1")
    from engine.gpu_manager import before_stage   # SeedVR2 is its own process: Comfy's models out of the card first
    before_stage("comfy", client, LABEL)
    if progress:
        progress(f"{LABEL}: podbicie wypełnienia ({iw}×{ih} → {cw}×{ch})", 0, 1)
    runner.run(clip, out, min(cw, ch), _batch(len(frames)), len(frames), cancel)
    got = read_frames(out)
    if len(got) < len(frames):
        raise RuntimeError(f"{LABEL} zwrócił {len(got)} klatek, potrzeba {len(frames)}")
    up = {}
    for f, im in zip(frames, got):
        up[f] = im if (im.shape[1], im.shape[0]) == (cw, ch) else cv2.resize(im, (cw, ch), interpolation=cv2.INTER_CUBIC)
    return UpFill(vdir, box, up, linear)
