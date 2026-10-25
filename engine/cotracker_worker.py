# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cotracker_worker.py - CoTracker3 run in ComfyUI's Python (its PyTorch + CUDA), called by engine/tracking.py.
MatteDesk's own .venv has no PyTorch (Python 3.11 / numpy 1.x vs ComfyUI's 3.13 / numpy 2.x – the binaries cannot be
shared in one process), so the tracker runs as a separate process with ComfyUI's interpreter. Standalone on purpose:
imports only numpy, torch and cotracker – nothing from MatteDesk.
    <comfy python> cotracker_worker.py --check                     -> prints "ok <device>" (exit 0) or the error (exit 1)
    <comfy python> cotracker_worker.py <in.npz> <out.npz>          -> tracks (one-shot: model loaded for this call)
    <comfy python> cotracker_worker.py --serve                     -> request loop (2026-10-25): one JSON line per
        request on stdin {"in": "<in.npz>", "out": "<out.npz>"} -> {"ok": true, "device": "cuda", "peak_gib": 1.2,
        "load_seconds": 3.1} or {"ok": false, "error": "..."}; {"op": "quit"} ends it. The model is loaded once, for
        the first request (a band-point edit used to start a new process – torch + CUDA + weights – every time).
        The maths per request is the one-shot code, so the tracks are the same.
in.npz : n_videos, checkpoint (str), max_queries (optional, queries per model call);
         frames either per video  frames_i (T, H, W, 3) uint8 RGB (already downscaled)
         or from a shared pool:   n_pools, pool_k (M_k, H, W, 3) uint8, pool_members_k (M_k,) global pool indices,
                                  frames_idx_i (T,) global pool indices of video i;
         queries_i (N, 3) float32 [t, x, y]  – several videos share one model load
out.npz: per video i: tracks_i (T, N, 2) float32, visible_i (T, N) bool; device (str); peak_gib (GPU, 0 on CPU)
Queries go to the model in chunks of max_queries; a chunk that hits CUDA OOM is halved and retried (down to 1 query).
MD_COTRACKER_FAKE=1: no torch / cotracker – every query keeps its key position in all frames, visible everywhere
(protocol tests in MatteDesk's .venv).
"""
import json
import os
import sys
import time

FAKE = os.environ.get("MD_COTRACKER_FAKE") == "1"


class _Model:
    """CoTracker3 loaded once per checkpoint (the serve loop keeps the instance)."""

    def __init__(self):
        self.ckpt = None
        self.model = None
        self.device = "fake" if FAKE else None
        self.load_seconds = 0.0

    def get(self, ckpt: str):
        if FAKE:
            return None
        import torch
        from cotracker.predictor import CoTrackerPredictor
        if self.model is None or self.ckpt != ckpt:
            t0 = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CoTrackerPredictor(checkpoint=ckpt).to(self.device).eval()
            self.ckpt = ckpt
            self.load_seconds = time.time() - t0
        return self.model


def _track_file(m: "_Model", in_path: str, out_path: str) -> dict:
    """The one-shot computation: in.npz -> out.npz; returns {"device", "peak_gib"}."""
    import numpy as np
    d = np.load(in_path, allow_pickle=False)
    files = set(d.files)
    ckpt = str(d["checkpoint"])
    n = int(d["n_videos"])
    max_q = int(d["max_queries"]) if "max_queries" in files else 1 << 30
    pool = {}   # global pool index -> (pool array, row)
    for k in range(int(d["n_pools"]) if "n_pools" in files else 0):
        arr = d[f"pool_{k}"]                      # read once (npz members are re-read on every access)
        for row, g in enumerate(d[f"pool_members_{k}"]):
            pool[int(g)] = (arr, row)

    def frames_of(i):
        if f"frames_{i}" in files:
            return d[f"frames_{i}"]
        return np.stack([pool[int(g)][0][pool[int(g)][1]] for g in d[f"frames_idx_{i}"]])

    out = {}
    if FAKE:
        for i in range(n):
            q = d[f"queries_{i}"].astype(np.float32)
            T = int(len(d[f"frames_idx_{i}"]) if f"frames_idx_{i}" in files else d[f"frames_{i}"].shape[0])
            out[f"tracks_{i}"] = np.repeat(q[None, :, 1:], T, axis=0).astype(np.float32)
            out[f"visible_{i}"] = np.ones((T, len(q)), bool)
        out["device"] = np.array("fake")
        out["peak_gib"] = np.array(0.0)
        np.savez(out_path, **out)
        return {"device": "fake", "peak_gib": 0.0}
    import torch
    model = m.get(ckpt)
    device = m.device

    def run(v, q):
        """Track q (N, 3) in v (1, T, 3, H, W); halves the chunk on CUDA OOM."""
        try:
            with torch.inference_mode():
                tracks, vis = model(v, queries=torch.from_numpy(q)[None].to(device))
            return tracks[0].cpu().numpy().astype(np.float32), vis[0].cpu().numpy() > 0.5
        except torch.cuda.OutOfMemoryError:
            if len(q) <= 1:
                raise
            torch.cuda.empty_cache()
            h = len(q) // 2
            print(f"OOM at {len(q)} queries x {v.shape[1]} frames - splitting", file=sys.stderr, flush=True)
            (ta, va), (tb, vb) = run(v, q[:h]), run(v, q[h:])
            return np.concatenate([ta, tb], 1), np.concatenate([va, vb], 1)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out["device"] = np.array(device)
    for i in range(n):
        q_all = d[f"queries_{i}"].astype(np.float32)
        with torch.inference_mode():
            v = torch.from_numpy(frames_of(i)).permute(0, 3, 1, 2)[None].float().to(device)
        T = v.shape[1]
        parts = [run(v, q_all[j:j + max_q]) for j in range(0, len(q_all), max_q)]
        out[f"tracks_{i}"] = (np.concatenate([p[0] for p in parts], 1) if parts
                              else np.zeros((T, 0, 2), np.float32))
        out[f"visible_{i}"] = np.concatenate([p[1] for p in parts], 1) if parts else np.zeros((T, 0), bool)
        del v
        if device == "cuda":
            torch.cuda.empty_cache()
    peak = float(torch.cuda.max_memory_allocated() / 2 ** 30) if device == "cuda" else 0.0
    out["peak_gib"] = np.array(peak)
    np.savez(out_path, **out)
    return {"device": device, "peak_gib": peak}


def _check():
    if FAKE:
        print("ok fake")
        return 0
    try:
        import torch
        from cotracker.predictor import CoTrackerPredictor  # noqa: F401
    except Exception as e:  # missing package / broken install
        print(f"{type(e).__name__}: {e}")
        return 1
    print("ok", "cuda" if torch.cuda.is_available() else "cpu")
    return 0


def _serve():
    proto = sys.stdout
    sys.stdout = sys.stderr      # library prints must not break the protocol
    m = _Model()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "quit":
                break
            info = _track_file(m, req["in"], req["out"])
            resp = {"ok": True, **info, "load_seconds": round(m.load_seconds, 3)}
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check()
    if len(argv) == 2 and argv[1] == "--serve":
        return _serve()
    if len(argv) != 3:
        print(__doc__)
        return 2
    _track_file(_Model(), argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
