# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/cotracker_worker.py - CoTracker3 run in ComfyUI's Python (its PyTorch + CUDA), called by engine/tracking.py.

MatteDesk's own .venv has no PyTorch (Python 3.11 / numpy 1.x vs ComfyUI's 3.13 / numpy 2.x – the binaries cannot be
shared in one process), so the tracker runs as a separate process with ComfyUI's interpreter. Standalone on purpose:
imports only numpy, torch and cotracker – nothing from MatteDesk.

    <comfy python> cotracker_worker.py --check                     -> prints "ok <device>" (exit 0) or the error (exit 1)
    <comfy python> cotracker_worker.py <in.npz> <out.npz>          -> tracks

in.npz : n_videos, checkpoint (str), max_queries (optional, queries per model call);
         frames either per video  frames_i (T, H, W, 3) uint8 RGB (already downscaled)
         or from a shared pool:   n_pools, pool_k (M_k, H, W, 3) uint8, pool_members_k (M_k,) global pool indices,
                                  frames_idx_i (T,) global pool indices of video i;
         queries_i (N, 3) float32 [t, x, y]  – several videos share one model load
out.npz: per video i: tracks_i (T, N, 2) float32, visible_i (T, N) bool; device (str); peak_gib (GPU, 0 on CPU)

Queries go to the model in chunks of max_queries; a chunk that hits CUDA OOM is halved and retried (down to 1 query).
"""

import sys


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        try:
            import torch
            from cotracker.predictor import CoTrackerPredictor  # noqa: F401
        except Exception as e:  # missing package / broken install
            print(f"{type(e).__name__}: {e}")
            return 1
        print("ok", "cuda" if torch.cuda.is_available() else "cpu")
        return 0
    if len(argv) != 3:
        print(__doc__)
        return 2
    import numpy as np
    import torch
    from cotracker.predictor import CoTrackerPredictor

    d = np.load(argv[1], allow_pickle=False)
    files = set(d.files)
    ckpt = str(d["checkpoint"])
    n = int(d["n_videos"])
    max_q = int(d["max_queries"]) if "max_queries" in files else 1 << 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoTrackerPredictor(checkpoint=ckpt).to(device).eval()

    pool = {}   # global pool index -> (pool array, row)
    for k in range(int(d["n_pools"]) if "n_pools" in files else 0):
        arr = d[f"pool_{k}"]                      # read once (npz members are re-read on every access)
        for row, g in enumerate(d[f"pool_members_{k}"]):
            pool[int(g)] = (arr, row)

    def frames_of(i):
        if f"frames_{i}" in files:
            return d[f"frames_{i}"]
        return np.stack([pool[int(g)][0][pool[int(g)][1]] for g in d[f"frames_idx_{i}"]])

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
    out = {"device": np.array(device)}
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
    out["peak_gib"] = np.array(torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0)
    np.savez(argv[2], **out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
