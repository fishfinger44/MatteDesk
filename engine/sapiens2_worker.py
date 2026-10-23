# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/sapiens2_worker.py - Sapiens2 body-part segmentation (Meta, `facebook/sapiens2-seg-0.4b`) run in ComfyUI's
Python (its PyTorch + CUDA + transformers ≥ 5.x, which has `Sapiens2ForSemanticSegmentation` built in – no Meta repo),
used by engine/sapiens2.py. Standalone on purpose: imports only numpy, torch and transformers (run with `python -I`).

29 classes (0 background, 4 Hair, 3 Face_Neck, 6/15 hands, 1 Apparel …), trained on person crops at 1024 x 768 (H x W).

    <py> sapiens2_worker.py --check [model_id ...]   -> "ok <device> cached=<ids in the HF cache>" (exit 0) or error (1)
    <py> sapiens2_worker.py --download <model_id>    -> fetches the weights into the Hugging Face cache (1.63 GB)
    <py> sapiens2_worker.py --serve <model_id>       -> request loop, one JSON line per request on stdin:
        {"x": "<in.npy>", "out": "<out.npy>"}  ->  {"ok": true, "device": "cuda", "dtype": "float16", "load_seconds": 3.4}
                                                   {"ok": false, "error": "..."}
        in.npy : (H, W, 3) float32 RGB 0..1, a person crop (any size; resized to 1024 x 768 here, on the GPU)
        out.npy: (H, W) float32 class index 0..28 (argmax of the logits resized back to H x W, bilinear)

MD_SAPIENS2_FAKE=1: no torch / transformers, classes from the luminance (protocol tests in MatteDesk's .venv):
    < 0.25 -> 4 (Hair), < 0.6 -> 3 (Face_Neck), else 1 (Apparel).
"""

import json
import os
import sys
import time

FAKE = os.environ.get("MD_SAPIENS2_FAKE") == "1"
DEFAULT_MODEL = "facebook/sapiens2-seg-0.4b"
INPUT_HW = (1024, 768)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cached(model_ids):
    if FAKE:
        return []
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return []
    out = []
    for mid in model_ids:
        try:
            p = try_to_load_from_cache(mid, "model.safetensors")
        except Exception:
            p = None
        if isinstance(p, str) and os.path.exists(p):
            out.append(mid)
    return out


def _check(model_ids):
    if FAKE:
        print("ok fake cached=")
        return 0
    try:
        import torch
        from transformers import AutoModelForSemanticSegmentation  # noqa: F401
        from transformers import Sapiens2ForSemanticSegmentation  # noqa: F401  (transformers 5.x)
    except Exception as e:  # missing package / transformers too old
        print(f"{type(e).__name__}: {e}")
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("ok", device, "cached=" + ",".join(_cached(model_ids or [DEFAULT_MODEL])))
    return 0


def _load(model_id):
    import torch
    from transformers import AutoModelForSemanticSegmentation
    try:
        return AutoModelForSemanticSegmentation.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:
        return AutoModelForSemanticSegmentation.from_pretrained(model_id, torch_dtype=torch.float32)


def _download(model_id):
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json", "README.md"])
    print("ok", model_id)
    return 0


class _Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.device = "fake" if FAKE else None
        self.dtype = "fake" if FAKE else None
        self.load_seconds = 0.0

    def forward(self, x):
        import numpy as np
        if FAKE:
            y = x @ np.array([0.299, 0.587, 0.114], np.float32)
            return np.where(y < 0.25, 4, np.where(y < 0.6, 3, 1)).astype(np.float32)
        import torch
        import torch.nn.functional as F
        if self.model is None:
            t0 = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            m = _load(self.model_id).eval().to(self.device)
            if self.device == "cuda":
                m = m.half()
            self.model = m
            self.dtype = str(next(m.parameters()).dtype).replace("torch.", "")
            self.load_seconds = time.time() - t0
        h, w = x.shape[:2]
        dt = next(self.model.parameters()).dtype
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]).to(self.device, torch.float32)
        t = F.interpolate(t.clamp_(0, 1), size=INPUT_HW, mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        t = ((t - mean) / std).to(dt)
        with torch.inference_mode():
            logits = self.model(pixel_values=t).logits.float()
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            cls = logits.argmax(1)[0].to(torch.float32).cpu().numpy()
        return cls.astype(np.float32)


def _serve(model_id):
    import numpy as np
    proto = sys.stdout
    sys.stdout = sys.stderr      # library prints must not break the protocol
    m = _Model(model_id)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "quit":
                break
            x = np.load(req["x"])
            if x.ndim != 3 or x.shape[2] != 3:
                raise ValueError(f"wejście musi być (H, W, 3) RGB, jest {x.shape}")
            np.save(req["out"], m.forward(x.astype(np.float32)))
            resp = {"ok": True, "device": m.device, "dtype": m.dtype, "load_seconds": round(m.load_seconds, 3)}
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check(argv[2:])
    if len(argv) == 3 and argv[1] == "--download":
        return _download(argv[2])
    if len(argv) == 3 and argv[1] == "--serve":
        return _serve(argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
