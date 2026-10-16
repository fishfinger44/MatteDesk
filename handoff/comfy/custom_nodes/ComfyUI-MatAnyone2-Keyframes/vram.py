# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""VRAM policy of MatAnyone2RunKeyframes (pure helpers, no ComfyUI import -> unit-testable with a mocked torch).

1. Conditional cache release (`VramReleaser`): the node used to call torch.cuda.empty_cache() after EVERY frame
   (Windows workaround: without expandable_segments the reserved pool ballooned to the whole 12 GB card and
   stalled). Freeing the cache every step forces cudaMalloc again on the next step -> the GPU idled (~20 %).
   Now the cache is released only when torch.cuda.memory_reserved() exceeds `frac` x the card's total memory
   (default 0.85). frac <= 0 = old behaviour (every frame). empty_cache never changes numerics.

2. Automatic working-memory size (`auto_max_mem_frames`, node input max_mem_frames = 0):
   MatAnyone2 (Cutie-style) reads memory with an affinity matrix  N_mem x HW  where
       HW    = ceil(h / 16) * ceil(w / 16)            (key tokens of one frame, stride 16)
       N_mem = (working frames + permanent frames) * HW
   get_similarity + top-k softmax keep ~3 fp32 copies of it (similarity, top-k buffers, scattered affinity), so
       bytes_per_memory_frame = HW^2 * 4 B * 3
   (1440 x 1440 -> HW = 8100 -> 0.73 GiB per memory frame). Budget (conservative):
       budget = 0.75 * available - 2 GiB            (backbone / decoder activations, fragmentation)
       available = free (cudaMemGetInfo) + reserved-but-unallocated torch cache
       n = floor(budget / bytes_per_memory_frame) - permanent_frames(3) - 1 (query frame)
   clamped to 5..15. 12 GiB card at 1440 px (even completely free): 0.75 * 12 - 2 = 7 GiB -> 9 frames -> 9 - 4 = 5
   = the official value used so far (1920 x 1080 has the same token count -> 5); 16 GiB -> 9; 24 GiB -> 15 (clamp);
   12 GiB at 720 px -> 15 (clamp).
"""
from __future__ import annotations

import math

GiB = 1024 ** 3
MEM_MIN, MEM_MAX = 5, 15
PERMANENT_FRAMES = 3          # <= 3 permanent keyframes per pass (node logic)
AFFINITY_COPIES = 3           # similarity + top-k buffers + affinity, fp32
SAFETY = 0.75                 # use at most 75 % of the available memory
FIXED_RESERVE = 2 * GiB       # backbone / decoder activations, allocator fragmentation


def bytes_per_memory_frame(h: int, w: int) -> int:
    hw = math.ceil(h / 16) * math.ceil(w / 16)
    return hw * hw * 4 * AFFINITY_COPIES


def auto_max_mem_frames(available_bytes: float, h: int, w: int) -> int:
    """Working memory frames from the available VRAM and the processing resolution (see module doc)."""
    per = bytes_per_memory_frame(h, w)
    budget = SAFETY * float(available_bytes) - FIXED_RESERVE
    n = int(budget // per) - PERMANENT_FRAMES - 1 if budget > 0 else 0
    return max(MEM_MIN, min(MEM_MAX, n))


def available_vram(torch_mod) -> float:
    """free (driver) + torch cache reserved but not allocated, in bytes."""
    free, _total = torch_mod.cuda.mem_get_info()
    return float(free) + float(torch_mod.cuda.memory_reserved() - torch_mod.cuda.memory_allocated())


def resolve_max_mem_frames(value: int, torch_mod, h: int, w: int, use_cuda: bool) -> int:
    """value > 0 -> explicit (>= 2); 0 -> auto (CPU / no CUDA -> the official 5)."""
    if value and value > 0:
        return max(2, int(value))
    if not use_cuda:
        return MEM_MIN
    return auto_max_mem_frames(available_vram(torch_mod), h, w)


class VramReleaser:
    """Call `step()` after every processed frame; it empties the CUDA cache only when needed."""

    def __init__(self, torch_mod, frac: float, enabled: bool = True):
        self.t = torch_mod
        self.frac = float(frac)
        self.enabled = enabled
        self.releases = 0
        self.steps = 0
        self.peak_reserved = 0
        self.limit = None
        if enabled and self.frac > 0:
            _free, total = torch_mod.cuda.mem_get_info()
            self.limit = self.frac * float(total)

    def step(self) -> bool:
        if not self.enabled:
            return False
        self.steps += 1
        reserved = self.t.cuda.memory_reserved()
        self.peak_reserved = max(self.peak_reserved, reserved)
        if self.limit is None or reserved > self.limit:
            self.t.cuda.empty_cache()
            self.releases += 1
            return True
        return False
