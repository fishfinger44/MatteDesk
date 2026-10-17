# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/stage_preview.py - "preview a few frames" in stages 3 MatAnyone, 4 VideoMaMa, 5 ViTMatte (user request
2026-10-17: check the settings before the long run on the whole shot).

The models are temporal (MatAnyone2 carries memory between frames, VideoMaMa works on batches), so a preview computes a
short WINDOW around the frame – the same code path as the full run, only fewer frames:
  matanyone ±4, videomama ±6 (one batch), vitmatte ±2 (per frame; ±2 for the temporal smoothing of the zones).
A short window has less temporal context than the whole shot – the edge can differ a little from the full run.

NOTHING of the object state changes (user condition): no runs/v###, no tool.stage_runs / active_run / core_base, so a
preview never unlocks a base or a stage. Results: <object>/stage_preview/<stage>/a_####.png (full-frame alpha, 8-bit)
+ meta.json {stage, center, frames, sig, seconds}. `sig` = the settings the preview was made with; when they change the
preview is reported as stale (shown dimmed, with a note).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

WINDOW = {"matanyone": 4, "videomama": 6, "vitmatte": 2}
STAGES = tuple(WINDOW)


def preview_dir(store: Any, name: str, stage: str) -> Path:
    return store.dir(name) / "stage_preview" / stage


def window(frames: Sequence[int], f: int, half: int) -> List[int]:
    """`2*half+1` visible frames around `f` (shifted inwards at the ends of the shot / object life)."""
    fr = sorted(frames)
    if not fr:
        return []
    i = min(range(len(fr)), key=lambda k: abs(fr[k] - f))
    lo = max(0, min(i - half, len(fr) - (2 * half + 1)))
    return fr[lo:lo + 2 * half + 1]


def preview_sig(store: Any, name: str, stage: str) -> str:
    """What the preview depends on: the mask 1+2 and the stage settings (+ the base of ViTMatte / VideoMaMa)."""
    from engine.stages import mask_sig, matte_sig, settings_sig, stage_run, vitmatte_base
    obj = store.get(name)
    parts: List[Any] = [stage, mask_sig(store, name)]
    if stage == "matanyone":
        parts.append(settings_sig(obj))
    elif stage == "videomama":
        from engine.videomama import object_settings, settings_sig as vmm_sig
        s = object_settings(obj)
        parts += [vmm_sig(s), s.get("base")]
        if s.get("base") == "matanyone":
            parts.append(stage_run(store, name, "matanyone")[0] or "")
    else:
        vb = vitmatte_base(obj)
        parts += [vb, matte_sig(store.dir(name))]
        if vb != "core":
            parts.append(stage_run(store, name, vb)[0] or "")
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def read_meta(store: Any, name: str, stage: str) -> Optional[Dict[str, Any]]:
    try:
        m = json.loads((preview_dir(store, name, stage) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    m["stale"] = m.get("sig") != preview_sig(store, name, stage)
    return m


def frame_path(store: Any, name: str, stage: str, f: int) -> Path:
    return preview_dir(store, name, stage) / f"a_{f:04d}.png"


def run(pipe: Any, name: str, stage: str, f: int, frames: Sequence[int], models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Compute the preview window of `stage` around frame `f` (frames = the shot / work frames to pick it from)."""
    if stage not in STAGES:
        raise ValueError(f"stage: {' | '.join(STAGES)}")
    store = pipe.store
    vis = pipe._visible(name, frames)
    win = window(vis, f, WINDOW[stage])
    if not win:
        raise ValueError("obiekt nie jest widoczny w tym zakresie klatek")
    t0 = time.time()
    pdir = preview_dir(store, name, stage)
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    sig = preview_sig(store, name, stage)
    if stage == "matanyone":
        work = pdir / "_work"
        work.mkdir()
        full = pipe.finalize(name, win, None, None, stage="matanyone", preview_dir=work)
        shutil.rmtree(work, ignore_errors=True)
    elif stage == "videomama":
        full = _videomama(pipe, name, win, models_dir)
    else:
        full = _vitmatte(pipe, name, win)
    for g in win:
        if g in full:
            a = np.clip(np.asarray(full[g], np.float32), 0, 1)
            cv2.imwrite(str(pdir / f"a_{g:04d}.png"), (a * 255 + 0.5).astype(np.uint8))
    meta = {"stage": stage, "center": int(f), "frames": [int(g) for g in win if g in full], "sig": sig,
            "seconds": round(time.time() - t0, 1), "created_at": datetime.now().isoformat(timespec="seconds")}
    (pdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {**meta, "stale": False}


def _videomama(pipe: Any, name: str, frames: List[int], models_dir: Optional[Path]) -> Dict[int, np.ndarray]:
    """vmm_run on one batch of frames, composed over the base in memory (no version, no vmm_raw)."""
    from engine.comfy_io import job_progress
    from engine.crop import extract_crop
    from engine.refine import assemble_full, contiguous_chunks
    from engine.videomama import VideoMaMa, compose, object_settings
    obj = pipe.store.get(name)
    s = object_settings(obj)
    keys = pipe.keys(name)
    core, soft, members = pipe._stage_core(name, frames, obj, keys)
    zres = pipe.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
    core2 = zres.core if zres else core
    base, _ = pipe.vmm_base(name, frames, s, core2)
    segs = pipe._work_segments(name, frames) if members else pipe._segments_in(name, frames)
    try:
        pipe.client.free_vram()
    except Exception:
        pass
    gen = VideoMaMa(pipe.client, pipe.work / "comfy_io", s, models_dir, progress=lambda t, d, n: job_progress(t, d, n))
    per_seg: Dict[str, Dict[int, np.ndarray]] = {}
    for seg in segs:
        sf = [f for f in frames if seg.first <= f <= seg.last]
        res: Dict[int, np.ndarray] = {}
        for chunk in contiguous_chunks(sf):
            if not chunk:
                continue
            job_progress(f"Podgląd VideoMaMa: {seg.seg_id}", 0, len(chunk))
            imgs = [pipe.loader.crop(f, seg, 1.0) for f in chunk]
            guides = [extract_crop(core2[f].astype(np.float32), seg) for f in chunk]
            mattes = gen.run(f"pv_{name}_{seg.seg_id}_{chunk[0]}", imgs, guides)
            for f, m in zip(chunk, mattes):
                res[f] = compose(extract_crop(base[f], seg), m, s["protect"], s["reach_px"], s.get("inset_px"), s.get("feather_px", 0.0))
        per_seg[seg.seg_id] = res
    return assemble_full(per_seg, segs, frames, pipe.w, pipe.h)


def _vitmatte(pipe: Any, name: str, frames: List[int]) -> Dict[int, np.ndarray]:
    """The ViTMatte zones laid on the chosen base, like „Zastosuj na ujęcie” – without writing a version (and for the
    mask 1+2 base without making the core_base version, which a later full run would wrongly reuse as complete)."""
    from engine.comfy_io import job_progress
    from engine.export import run_alpha_loader
    from engine.refine import apply_soft_override
    from engine.stages import stage_run, vitmatte_base
    obj = pipe.store.get(name)
    vb = vitmatte_base(obj)
    keys = pipe.keys(name)
    core, soft, _ = pipe._stage_core(name, frames, obj, keys)
    if vb == "core":
        zb = pipe.apply_zones(name, frames, core, keys, probs=soft, kinds=("brush",))
        core2 = zb.core if zb else core
        base = {f: core2[f].astype(np.float32) for f in frames}
        base_alpha = lambda f: base[f]   # noqa: E731
    else:
        run, ok, why = stage_run(pipe.store, name, vb)
        if not run or not ok:
            raise ValueError(f"baza ViTMatte ({'MatAnyone' if vb == 'matanyone' else 'VideoMaMa'}) nie jest policzona"
                             + (f" – {why}" if why else "") + " – policz ten etap albo wybierz bazę „maska 1+2”")
        base_alpha = run_alpha_loader(pipe.store.dir(name), run, pipe.w, pipe.h, sub="alpha_base")
        base = {f: base_alpha(f).astype(np.float32) for f in frames}
    zres = pipe.apply_zones(name, frames, core, keys, probs=soft, base_alpha=base_alpha, kinds=("brush", "matte"))
    out: Dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        job_progress("Podgląd ViTMatte", i + 1, len(frames))
        a = base[f]
        if zres and f in zres.soft:
            sa, area = zres.soft[f]
            from engine.refine import ZONE_FEATHER_PX
            a = apply_soft_override(a, sa.astype(np.float32), area > 0.5, zres.feather_px or ZONE_FEATHER_PX)
        out[f] = a
    return out
