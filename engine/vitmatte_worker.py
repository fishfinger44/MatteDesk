# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/vitmatte_worker.py - ViTMatte run in ComfyUI's Python (its PyTorch + CUDA + transformers), used by
engine/vitmatte.py when MatteDesk's own .venv has no PyTorch (since v0.11). Standalone on purpose: imports only
numpy, torch and transformers – nothing from MatteDesk (run with `python -I`).

    <py> vitmatte_worker.py --check [model_id ...]   -> "ok <device> cached=<ids in the HF cache>" (exit 0) or error (1)
    <py> vitmatte_worker.py --download <model_id>    -> fetches the weights into the Hugging Face cache
    <py> vitmatte_worker.py --serve <model_id>       -> request loop, one JSON line per request on stdin:
        {"x": "<in.npy>", "out": "<out.npy>"}  ->  {"ok": true, "device": "cuda", "load_seconds": 1.2}
                                                   {"ok": false, "error": "..."}
        in.npy : (H, W, 4) float32 – normalised RGB + trimap channel, already padded to a multiple of 32,
                 or an (N, H, W, 4) stack = the tiles of one frame (2026-10-25): each tile its own forward
        out.npy: (H, W) float32 alpha clamped to 0..1 (FG / BG forcing and the crop are done by the caller),
                 (N, H, W) for a stack

MD_VITMATTE_FAKE=1: no torch / transformers, alpha = trimap channel (protocol tests in MatteDesk's .venv).
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

# the common loop, loaded by its path: this file runs with -I, engine/ is not on the path (engine/worker_proto.py)
_spec = importlib.util.spec_from_file_location("worker_proto", Path(__file__).with_name("worker_proto.py"))
_wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wp)

FAKE = os.environ.get("MD_VITMATTE_FAKE") == "1"


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
            p = try_to_load_from_cache(mid, "config.json")
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
        from transformers import VitMatteForImageMatting  # noqa: F401
    except Exception as e:  # missing package / broken install
        print(f"{type(e).__name__}: {e}")
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("ok", device, "cached=" + ",".join(_cached(model_ids)))
    return 0


def _download(model_id):
    from transformers import VitMatteForImageMatting
    VitMatteForImageMatting.from_pretrained(model_id)
    print("ok", model_id)
    return 0


class _Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.device = "fake" if FAKE else None
        self.load_seconds = 0.0

    def forward(self, x):
        import numpy as np
        if FAKE:
            return np.clip(x[..., 3], 0, 1).astype(np.float32)
        import torch
        if self.model is None:
            from transformers import VitMatteForImageMatting
            t0 = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = VitMatteForImageMatting.from_pretrained(self.model_id).eval().to(self.device)
            self.load_seconds = time.time() - t0
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))[None]).to(self.device)
        with torch.inference_mode():
            a = self.model(pixel_values=t).alphas[0, 0].float().clamp_(0, 1).cpu().numpy()
        return a.astype(np.float32)


def _serve(model_id):
    import numpy as np

    def make():
        m = _Model(model_id)

        def handle(req, ask):
            t0 = time.perf_counter()
            x = np.load(req["x"])
            t1 = time.perf_counter()
            load0 = m.load_seconds
            a = np.stack([m.forward(xi) for xi in x]) if x.ndim == 4 else m.forward(x)
            t2 = time.perf_counter()
            np.save(req["out"], a)
            t3 = time.perf_counter()
            # phases for the zone stats (plan A6); the one-time model load is reported apart, not as forward
            loaded = m.load_seconds - load0
            return {"device": m.device, "load_seconds": round(m.load_seconds, 3),
                    "timings": {"read": round(t1 - t0, 4), "forward": round(t2 - t1 - loaded, 4),
                                "load": round(loaded, 4), "write": round(t3 - t2, 4)}}
        return handle
    return _wp.serve(make)


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
