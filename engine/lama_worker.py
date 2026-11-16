# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/lama_worker.py - LaMa (big-lama.pt, TorchScript) at the crop's FULL resolution, run in ComfyUI's Python (its
PyTorch + CUDA). The ComfyUI node INPAINT_InpaintWithModel always works at 256 x 256 and scales back (soft large
patches); LaMa is fully convolutional, so here it sees the crop as it is (user 2026-11-10).

Standalone on purpose: numpy (+ torch in the worker), nothing from MatteDesk (run with `python -I`).

Worker protocol (engine/worker_proto.py):
    <py> lama_worker.py --check   -> "ok cuda <gpu>" / "ok cpu" / "ok fake" (exit 0) or the error (exit 1)
    <py> lama_worker.py --serve   -> one JSON request per line:
        {"op": "inpaint", "model": path of big-lama.pt, "img": .npy (h, w, 3) float32 RGB 0..1,
         "mask": .npy (h, w) 0 / 1, "out": .npy}  -> (h, w, 3) float32 RGB 0..1 (outside the mask = the input)
        {"op": "release"}  drops the model and the cached GPU memory;  {"op": "quit"}
MD_LAMA_FAKE=1: no model - the hole is filled by diffusion of its surroundings (numpy; protocol tests in .venv).
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

FAKE = os.environ.get("MD_LAMA_FAKE") == "1"
PAD = 8                      # LaMa's FFC downsamples 3 x by 2: sides padded to a multiple of 8
torch = None
DEV = None
_MODEL = {"path": None, "net": None}


def _torch():
    global torch, DEV
    if torch is None:
        import torch as _t
        torch = _t
        DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch


def _net(path):
    if _MODEL["path"] != path:
        t = _torch()
        net = t.jit.load(path, map_location=DEV)
        net.eval()
        _MODEL.update(path=path, net=net)
    return _MODEL["net"]


def _fake_fill(img, mask):
    """Diffusion of the known pixels into the hole (repeated 3 x 3 averages) - only for the protocol tests."""
    out = img.copy()
    known = mask < 0.5
    out[~known] = 0.0
    w = known.astype(np.float32)
    for _ in range(200):
        pad = np.pad(out * w[..., None], ((1, 1), (1, 1), (0, 0)), mode="edge")
        pw = np.pad(w, 1, mode="edge")
        s = sum(pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        c = sum(pw[1 + dy:pw.shape[0] - 1 + dy, 1 + dx:pw.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        fill = s / np.maximum(c, 1e-6)[..., None]
        out = np.where(known[..., None], img, fill)
        w = np.where(known, 1.0, np.minimum(c, 1.0)).astype(np.float32)
    return out.astype(np.float32)


def inpaint(model_path, img, mask):
    """img (h, w, 3) RGB 0..1, mask (h, w) 0 / 1 -> (h, w, 3) RGB 0..1; outside the mask the input bit for bit."""
    img = np.asarray(img, np.float32)
    m = (np.asarray(mask, np.float32) > 0.5).astype(np.float32)
    if FAKE:
        res = _fake_fill(img, m)
    else:
        t = _torch()
        h, w = m.shape
        ph, pw = (-h) % PAD, (-w) % PAD
        x = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect" if min(h, w) > 1 else "edge")
        k = np.pad(m, ((0, ph), (0, pw)), mode="reflect" if min(h, w) > 1 else "edge")
        xt = t.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)), device=DEV)[None]
        kt = t.as_tensor(np.ascontiguousarray(k), device=DEV)[None, None]
        with t.no_grad():
            y = _net(model_path)(xt, kt)
        res = y[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()[:h, :w].astype(np.float32)
    return np.where(m[..., None] > 0.5, res, img).astype(np.float32)


def _handle(req, ask=None):
    op = req.get("op")
    if op == "release":
        _MODEL.update(path=None, net=None)
        if torch is not None and DEV is not None and DEV.type == "cuda":
            torch.cuda.empty_cache()
        return {}
    if op == "inpaint":
        out = inpaint(req["model"], np.load(req["img"]), np.load(req["mask"]))
        np.save(req["out"], out)
        return {}
    raise ValueError(f"unknown op {op}")


def _serve():
    spec = importlib.util.spec_from_file_location("worker_proto", Path(__file__).resolve().parent / "worker_proto.py")
    wp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wp)

    def make():
        device = "fake" if FAKE else str(_torch() and DEV)

        def handle(req, ask):
            t0 = time.time()
            out = _handle(req, ask)
            resp = {"device": device, "seconds": round(time.time() - t0, 4), **out}
            if not FAKE and DEV is not None and DEV.type == "cuda":
                resp["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            return resp
        return handle
    return wp.serve(make)


def _check():
    if FAKE:
        print("ok fake")
        return 0
    try:
        t = _torch()
        float(t.ones(2, 2, device=DEV).sum())
        name = t.cuda.get_device_name(0) if DEV.type == "cuda" else ""
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1
    print("ok", DEV.type, name)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check()
    if len(argv) >= 2 and argv[1] == "--serve":
        return _serve()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
