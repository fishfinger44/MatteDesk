# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the peak GPU memory of VideoMaMa passes -> engine/vram_table.json (engine/vram_forecast.py).

    python tools/measure_vram.py [--sides 448,768,1024,1280,1536] [--frames 24,12] [--offload off,on] [--dry]

Needs ComfyUI running with the VideoMaMa nodes and weights (Doctor). Synthetic frames (memory does not depend on the
content). Per pass: Comfy /free, the card's memory before (base_gb), nvidia-smi sampled every 100 ms during the pass
(peak_gb). A pass that reaches the card's size spilled into shared RAM – its peak is not the need, so it is reported and
NOT stored. Measured points replace the 'seed' points of the same model / variant.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfy_client import ComfyClient          # noqa: E402
from engine import vram_forecast as vf       # noqa: E402
from engine.videomama import VideoMaMa, vmm_settings   # noqa: E402


class Peak:
    """nvidia-smi memory.used of GPU 0 every 100 ms in a thread: the maximum (GB)."""

    def __init__(self):
        self.peak = 0.0
        self._p = None
        self._t = None

    def __enter__(self):
        self._p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0",
                                    "-lms", "100"], stdout=subprocess.PIPE, text=True)

        def read():
            for line in self._p.stdout:
                try:
                    self.peak = max(self.peak, float(line.strip()) / 1024)
                except ValueError:
                    pass
        self._t = threading.Thread(target=read, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._p.terminate()
        self._p.wait(5)


def used_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out) / 1024


def card() -> tuple:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return float(out[0]) / 1024, out[1].strip().replace("NVIDIA GeForce ", "")


def settle(client: ComfyClient, s: float = 3.0) -> float:
    client.free_vram(unload_models=True, free_memory=True)
    time.sleep(s)
    return used_gb()


def frames_of(side: int, n: int):
    rng = np.random.default_rng(side + n)
    yy, xx = np.mgrid[:side, :side]
    mask = (((yy - side / 2) ** 2 + (xx - side / 2) ** 2) < (side * 0.3) ** 2).astype(np.float32)
    imgs = [np.clip(rng.random((side, side, 3), np.float32) * 0.3 + 0.35, 0, 1) for _ in range(n)]
    return imgs, [mask] * n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", default="448,768,1024,1280,1536")
    ap.add_argument("--frames", default="24,12")
    ap.add_argument("--offload", default="off,on")
    ap.add_argument("--dry", action="store_true", help="print, do not write engine/vram_table.json")
    a = ap.parse_args()
    from engine.config import load_config
    models_dir = load_config().get("models_dir")
    client = ComfyClient.from_config()
    if not client.is_alive():
        print("ComfyUI nie odpowiada – uruchom go i powtórz")
        return 2
    total, gpu = card()
    table = vf.load_table()
    new = {}
    io = Path(tempfile.mkdtemp(prefix="md_vram_"))
    try:
        for off in a.offload.split(","):
            variant = "offload" if off == "on" else "default"
            for n in [int(x) for x in a.frames.split(",")]:
                for side in [int(x) for x in a.sides.split(",")]:
                    s = vmm_settings({"videomama": {"max_side": side, "batch": n, "overlap": 0, "offload": off}})
                    base = settle(client)
                    imgs, masks = frames_of(side, n)
                    t0 = time.time()
                    err = None
                    with Peak() as pk:
                        try:
                            VideoMaMa(client, io, s, Path(models_dir) if models_dir else None).run(f"vram_{side}_{n}", imgs, masks)
                        except Exception as e:   # OOM etc.: report, keep going with the next size
                            err = str(e)[:200]
                    sec = time.time() - t0
                    capped = pk.peak >= total - 0.4
                    line = (f"{variant:8s} {side:5d} px × {n:2d} kl.  x={vf.mpx_frames(side, side, n):6.2f}  "
                            f"base {base:5.2f}  peak {pk.peak:5.2f} GB  {sec:6.1f} s" +
                            ("  PEŁNA KARTA – nie zapisuję" if capped else "") + (f"  BŁĄD: {err}" if err else ""))
                    print(line, flush=True)
                    if err or capped:
                        continue
                    new.setdefault(variant, []).append({"x": round(vf.mpx_frames(side, side, n), 3), "side": side, "frames": n,
                                                        "peak_gb": round(pk.peak, 2), "base_gb": round(base, 2), "gpu": gpu,
                                                        "seconds": round(sec, 1), "date": datetime.date.today().isoformat(),
                                                        "source": "tools/measure_vram.py"})
    finally:
        shutil.rmtree(io, ignore_errors=True)
        client.free_vram(unload_models=True, free_memory=True)
    for variant, pts in new.items():
        ab = vf.fit(pts)
        print(f"{variant}: a = {ab[0]:.2f} GB, b = {ab[1]:.3f} GB / Mpx·kl." if ab else f"{variant}: za mało punktów")
    if a.dry or not new:
        return 0
    vm = table.setdefault("models", {}).setdefault("videomama", {})
    for variant, pts in new.items():
        done = {(q["side"], q["frames"]) for q in pts}   # a new measurement replaces the same (side, frames) only
        vm[variant] = [q for q in vm.get(variant, []) if q.get("source") != "seed"
                       and (q.get("side"), q.get("frames")) not in done] + pts
    vf.TABLE_PATH.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"zapisane: {vf.TABLE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
