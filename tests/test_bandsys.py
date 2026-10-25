# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""One band system for every stage with a band (engine/bandsys.py, user decision 2026-10-22)."""

import cv2
import numpy as np

from engine.bandsys import band_weight, compose, feather_lines
from engine.ltx_alpha import sdf_px
from engine.videomama import compose as vmm_compose


def _disk(h=240, w=240, r=60, c=(120, 120)):
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, c, r, 1.0, -1)
    return m


def test_videomama_without_points_is_close_to_the_former_composition():
    """No points, no outer feather: the unified-slider composition of 2026-10-17 up to the one measure of the band
    (2026-10-24: the distances come from the band area, ~1 px on the ramp); without feathers bit for bit."""
    base = _disk()
    rng = np.random.default_rng(1)
    vmm = rng.random(base.shape).astype(np.float32)
    reach, inset, feather = 30.0, 8.0, 5.0
    d = sdf_px(base)
    band = (d >= -reach) & (d <= inset)
    w = np.clip((inset - d) / feather, 0.0, 1.0)
    old = np.where(band, w * vmm + (1.0 - w) * base, base).astype(np.float32)
    new = vmm_compose(base, vmm, reach, inset, feather)
    assert np.abs(new - old).max() <= 1.0 / feather + 1e-6 and np.abs(new - old).mean() < 0.01
    hard = np.where(band, vmm, base).astype(np.float32)                    # feather 0 = hard on both sides
    assert np.array_equal(vmm_compose(base, vmm, reach, inset, 0.0), hard)


def test_feathers_blend_into_the_core_and_never_overlap():
    core = _disk()
    model = np.full(core.shape, 0.5, np.float32)
    d = sdf_px(core)
    a = compose(core, model, d, 20.0, 6.0, 4.0, 10.0)
    y = 120
    at = lambda px: a[y, int(np.argmin(np.abs(-d[y, 150:220] - px))) + 150]  # alpha px outside the core edge
    assert at(-7) == 1.0 and at(21) == 0.0                                  # beyond the band: the core (1 / 0)
    assert at(-6.2) > 0.97 and at(20) < 0.11                               # band edges: the core (first px: 1 / 10)
    assert abs(at(-1) - 0.5) < 1e-6 and abs(at(10) - 0.5) < 1e-6           # full model weight between the feathers
    assert 0.5 < at(-4) < 1.0 and 0.0 < at(15) < 0.5                       # the ramps
    _, w = band_weight(d, 3.0, 3.0, 4.0, 10.0)                             # band 6 px: inset 4 keeps, outer gets 2
    assert w.max() > 0.99


def test_points_widen_the_band_on_their_side():
    core = _disk()
    d = sdf_px(core)
    from engine.bandsys import band_widths
    o, i = band_widths(core, 10.0, 4.0, [(180.0, 120.0, 40.0, 4.0), (60.0, 120.0, 2.0, 4.0)])   # right 40, left 2
    band, _, _ = feather_lines(d, o, i, 0.0, 0.0)
    assert band[120, 180 + 35] and not band[120, 60 - 6]                   # right: 40 px out, left: 2 px
