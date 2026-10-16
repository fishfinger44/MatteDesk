# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""SAM3_VideoTrackSoft – ComfyUI core SAM3_VideoTrack plus the SOFT mask probabilities (MatteDesk, 2026-10-14).

The core tracker (comfy/ldm/sam3/tracker.py, track_video_with_detection) computes mask logits at 1008^2 for every
frame and immediately bit-packs `masks > 0`; `track_data` keeps only 0/1. For the temporal stabilisation of the core
(engine/corestab.py, "inference-time temporal probability smoothing") MatteDesk needs the probabilities.

This node calls the unchanged core node and, ONLY for the duration of that call, wraps two names the tracker looks
up at run time:
- `comfy.ldm.sam3.tracker.pack_masks`  -> also stores sigmoid(logits) of the frame (uint8, union = max over objects),
- `comfy.utils.ProgressBar`            -> its `update` marks the end of a frame (frames without any object never
                                          reach pack_masks: they get an all-zero soft mask), so the soft stack is
                                          aligned with `packed_masks` frame by frame.
Both are restored in a `finally`. Nothing in ComfyUI is modified on disk. If a ComfyUI update renames either symbol
the node raises a clear error (MatteDesk's Doctor reports the node as unavailable) – the binary path is untouched.

Outputs: track_data (SAM3_TRACK_DATA, identical to SAM3_VideoTrack) and soft_masks (MASK [N, H, W], 0..1, bilinear
to the input size exactly like SAM3_TrackToMask).
"""

import threading

import torch
import torch.nn.functional as F

_LOCK = threading.Lock()


def _core():
    try:
        import comfy.utils
        import comfy.ldm.sam3.tracker as tracker
        from comfy_extras.nodes_sam3 import SAM3_VideoTrack
    except Exception as e:  # pragma: no cover - environment
        raise RuntimeError(f"SAM3_VideoTrackSoft: brak core SAM3 w tym ComfyUI ({e})")
    if not hasattr(tracker, "pack_masks") or not hasattr(comfy.utils, "ProgressBar"):
        raise RuntimeError("SAM3_VideoTrackSoft: ComfyUI zmieniło tracker.pack_masks / comfy.utils.ProgressBar – węzeł wymaga aktualizacji")
    return comfy.utils, tracker, SAM3_VideoTrack


class _Capture:
    """Per-frame soft masks captured from pack_masks, aligned by ProgressBar.update."""

    def __init__(self):
        self.pending = None      # soft of the current frame (uint8 [Hm, Wm]) or None
        self.frames = []         # one entry per frame: uint8 tensor or None

    def on_pack(self, masks):
        # masks: logits [N_obj, Hm, Wm] (float); union = max over objects of sigmoid
        with torch.no_grad():
            m = masks.detach()
            if m.dim() == 3:
                m = m.max(dim=0).values
            p = torch.sigmoid(m.float())
            self.pending = (p * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).cpu()

    def on_frame_end(self):
        self.frames.append(self.pending)
        self.pending = None


class SAM3_VideoTrackSoft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("MODEL",),
            },
            "optional": {
                "initial_mask": ("MASK",),
                "conditioning": ("CONDITIONING",),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_objects": ("INT", {"default": 0, "min": 0, "max": 256}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA", "MASK")
    RETURN_NAMES = ("track_data", "soft_masks")
    FUNCTION = "run"
    CATEGORY = "PlateMatte/SAM3"

    def run(self, images, model, initial_mask=None, conditioning=None, detection_threshold=0.5, max_objects=0,
            detect_interval=1):
        utils, tracker, core_node = _core()
        N, H, W, _ = images.shape
        cap = _Capture()
        orig_pack = tracker.pack_masks
        orig_bar = utils.ProgressBar

        def pack_and_capture(masks):
            cap.on_pack(masks)
            return orig_pack(masks)

        class Bar(orig_bar):
            def update(self, value):
                cap.on_frame_end()
                return super().update(value)

        with _LOCK:      # one capture at a time (the patched names are module globals)
            tracker.pack_masks = pack_and_capture
            utils.ProgressBar = Bar
            try:
                out = core_node.execute(images, model, initial_mask=initial_mask, conditioning=conditioning,
                                        detection_threshold=detection_threshold, max_objects=max_objects,
                                        detect_interval=detect_interval)
            finally:
                tracker.pack_masks = orig_pack
                utils.ProgressBar = orig_bar
        track_data = out.args[0] if hasattr(out, "args") else out[0]
        if cap.pending is not None:          # a frame packed without a closing update (defensive)
            cap.on_frame_end()
        frames = cap.frames[:N]
        if len(frames) < N:
            frames = frames + [None] * (N - len(frames))
        shapes = [f.shape for f in frames if f is not None]
        hm, wm = shapes[0] if shapes else (H, W)
        soft = torch.zeros(N, hm, wm, dtype=torch.uint8)
        for i, f in enumerate(frames):
            if f is not None:
                soft[i] = f
        soft = soft.float().div_(255.0).unsqueeze(1)
        if (hm, wm) != (H, W):
            soft = F.interpolate(soft, size=(H, W), mode="bilinear", align_corners=False)
        return (track_data, soft[:, 0])


NODE_CLASS_MAPPINGS = {"SAM3_VideoTrackSoft": SAM3_VideoTrackSoft}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM3_VideoTrackSoft": "SAM3 Video Track + soft masks (PlateMatte)"}
