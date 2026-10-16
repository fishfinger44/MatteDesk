# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Object option key_band_same (user decision 2026-10-15): key frames get the same edge band as the other frames, so the
edge does not turn hard for one frame at every key (CLIP2_A F50: half the soft pixels of its neighbours with the spec's
+-3 px key band). key_band_same=False = the spec rule."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_core_soft import _propagate  # noqa: E402

from engine.export import run_alpha_loader  # noqa: E402
from engine.refine import key_band_same_on  # noqa: E402


def _soft_px(a):
    return int(((a > 0.02) & (a < 0.98)).sum())


def test_key_band_same_option_default_and_effect(tmp_path, monkeypatch):
    import cv2
    from engine import refine
    from engine.crop import extract_crop
    assert key_band_same_on({}) is True and key_band_same_on({"key_band_same": False}) is False

    def wide_ma(self, seg, frames, core, keys, start, scale, tag, soft=None):   # MatAnyone with a wide soft edge (8 px)
        self._last_keys = [start]
        return {f: cv2.GaussianBlur(extract_crop(core[f].astype(np.float32), seg), (0, 0), 4.0) for f in frames}
    monkeypatch.setattr(refine.Finalizer, "_matanyone_segment", wide_ma)
    shot, pipe, frames, start = _propagate(tmp_path, False)
    pipe.run_core("obj", frames)
    res = {}
    for val in (False, True):
        pipe.store.update("obj", key_band_same=val, ma_scale_mode="1.0")
        _, _, run_dir = pipe.finalize("obj", frames, edge_mode="matanyone")
        load = run_alpha_loader(pipe.store.dir("obj"), run_dir.name, shot.w, shot.h)
        res[val] = {f: _soft_px(load(f)) for f in frames}
    nb = [f for f in (start - 1, start + 1) if f in frames]
    key_off, key_on = res[False][start], res[True][start]
    neigh = np.mean([res[True][f] for f in nb])
    assert key_on > key_off, (key_on, key_off)                       # the key is no longer narrower
    assert key_on >= 0.7 * neigh, (key_on, neigh)                     # ... and close to its neighbours
    for f in frames:            # frames beyond the temporal median (+-2) of the key: the option changes nothing
        if abs(f - start) > 2:
            assert res[True][f] == res[False][f]
