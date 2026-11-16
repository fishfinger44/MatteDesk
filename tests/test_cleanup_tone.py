# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""C1 „wyrównanie poziomu łaty” and the mask on the latent grid (user decisions 2026-11-12 / 2026-11-13, after AInVFX
Paint a plate's HDR Match and its erase blocks): a video model's frames get ONE gain per channel for the whole shot,
measured where the model did not draw, blurred (detail is not level), robust to content the model re-draws; the masks
MiniMax-H3 / Wan 2.2 Fun VACE get cover whole latent cells."""
import numpy as np

from engine import cleanup as cu
from engine import h3_fill as H
from engine import vace_fill as V

W, H_ = 256, 192


def _scene():
    yy, xx = np.mgrid[0:H_, 0:W]
    bg = np.dstack([0.2 + 0.4 * xx / W, 0.3 + 0.3 * yy / H_, np.full((H_, W), 0.4)]).astype(np.float32)
    obj = (np.hypot(xx - 120, yy - 96) < 30).astype(np.float32)
    plate = bg.copy()
    plate[obj > 0] = (0.9, 0.1, 0.1)
    return bg, obj, plate


class _Fill:
    """A stand-in video model: the true background with a level error (a gain per channel) everywhere, optionally
    re-drawn detail (noise) and a band of different content right around the mask (what VACE does with hair)."""
    label = "MiniMax-H3"
    tone = True
    tone_gap_px = 6.0

    def __init__(self, bg, gain=(1.06, 0.97, 1.03), tone=True, detail=0.0, redraw=False, seed=0):
        rng = np.random.default_rng(seed)
        self.img = (bg * np.asarray(gain, np.float32)).astype(np.float32)
        if detail:
            self.img = self.img + rng.normal(0, detail, self.img.shape).astype(np.float32)
        if redraw:   # 20 px of other content beyond the mask (gap 6 px): the old ring rule took it for a level error
            yy, xx = np.mgrid[0:H_, 0:W]
            r = np.hypot(xx - 120, yy - 96)
            band = (r > 36) & (r < 56)
            self.img[band] = self.img[band] * 0.4
        self.tone = tone

    def __call__(self, f):
        return self.img, np.ones(self.img.shape[:2], np.float32)


def _run(tmp_path, fill, s=None, keep=None, frames=(1,)):
    bg, obj, plate = _scene()
    c = cu.Cleaner({"grow_px": 4, "soft_px": 0, "tone": True, **(s or {})}, W, H_, False, plate=lambda f: plate,
                   remove_alphas=lambda f: [obj], keep_alphas=lambda f: keep or [], key_fill=fill)
    rj = c.run(list(frames), tmp_path / "v001", "png16")
    import cv2
    comp = cv2.imread(str(tmp_path / "v001" / "comp" / f"c_{frames[0]:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    return rj, comp, bg, c.mask(frames[0])


def test_gain_of_the_model_is_removed_in_the_patch(tmp_path):
    bg, _, _ = _scene()
    rj, comp, bg, m = _run(tmp_path / "on", _Fill(bg))
    rj0, comp0, _, _ = _run(tmp_path / "off", _Fill(bg), {"tone": False})
    inside = m > 0.99
    err_on = np.abs(comp - bg)[inside].mean()
    err_off = np.abs(comp0 - bg)[inside].mean()
    assert err_off > 0.01 and err_on < err_off * 0.1
    g = rj["tone"]["bgr"]
    assert rj["tone"]["applied"] and abs(g[0] - 1 / 1.06) < 2e-3 and abs(g[1] - 1 / 0.97) < 2e-3 and "tone" not in rj0
    assert np.array_equal(comp[m == 0], comp0[m == 0])   # nothing outside the patch moves


def test_redrawn_content_and_detail_around_the_mask_do_not_fool_it(tmp_path):
    """The failure of the first rule on the real shot: content the model re-draws next to the mask is not a level."""
    bg, _, _ = _scene()
    rj, comp, bg, m = _run(tmp_path, _Fill(bg, detail=0.03, redraw=True))
    g = rj["tone"]["bgr"]
    assert abs(g[0] - 1 / 1.06) < 5e-3 and abs(g[1] - 1 / 0.97) < 5e-3 and abs(g[2] - 1 / 1.03) < 5e-3


def test_one_gain_for_the_whole_shot(tmp_path):
    bg, _, _ = _scene()
    fills = {f: _Fill(bg, gain=(1.05 + 0.01 * (f % 3), 1.0, 1.0), seed=f).img for f in range(1, 8)}

    class Var(_Fill):
        def __init__(self):
            self.tone = True

        def __call__(self, f):
            return fills[f], np.ones(fills[f].shape[:2], np.float32)
    rj, *_ = _run(tmp_path, Var(), frames=tuple(range(1, 8)))
    assert rj["tone"]["frames"] == 7 and rj["tone"]["spread"] > 0      # measured per frame, ONE gain applied
    assert abs(rj["tone"]["bgr"][0] - 1 / 1.06) < 5e-3


def test_unmarked_fill_right_level_and_content_mismatch(tmp_path):
    bg, _, _ = _scene()
    rj, comp, _, m = _run(tmp_path / "a", _Fill(bg, tone=False))
    assert "tone" not in rj and np.abs(comp - bg)[m > 0.99].mean() > 0.01        # e.g. Cleanup 3D keys: levelled there
    rj, comp, _, m = _run(tmp_path / "b", _Fill(bg, gain=(1.0, 1.0, 1.0)))
    assert not rj["tone"]["applied"] and not rj["tone"]["why"] and np.abs(comp - bg)[m > 0.99].max() < 2e-3
    rj, *_ = _run(tmp_path / "c", _Fill(bg, gain=(1.6, 1.0, 1.0)))
    assert not rj["tone"]["applied"] and "treścią" in rj["tone"]["why"] and any("pominięte" in n for n in rj["notes"])


def test_kept_object_is_not_measured(tmp_path):
    bg, _, _ = _scene()
    yy, xx = np.mgrid[0:H_, 0:W]
    keep = ((xx > 160) & (xx < 250)).astype(np.float32)          # a large kept object the model drew wrong
    fill = _Fill(bg)
    fill.img = fill.img.copy()
    fill.img[keep > 0] = fill.img[keep > 0] * 0.5
    rj, comp, bg, m = _run(tmp_path, fill, keep=[keep])
    assert abs(rj["tone"]["bgr"][0] - 1 / 1.06) < 3e-3


def test_settings_defaults():
    s = cu.settings({})
    assert s["tone"] is True and "tone_ring_px" not in s          # on again after the real test of the gain rule
    assert H.H3Fill.tone and V.VaceFill.tone and cu.VoidKeyFill.tone
    assert cu.shot_gain([]) is None


def test_mask_snapped_to_whole_latent_cells():
    m = np.zeros((64, 96), np.uint8)
    m[20:23, 37:41] = 255                                        # touches cells x 32..47, y 16..31
    g = H.snap_to_grid(m, 16)
    assert g[16:32, 32:48].all() and g.sum() == 16 * 16 * 255
    assert np.array_equal(H.snap_to_grid(m, 0), m)
    odd = H.snap_to_grid(np.pad(m, ((0, 5), (0, 3))), 16)        # a size that is not a multiple of the grid
    assert odd.shape == (69, 99) and odd[16:32, 32:48].all()
    grown = H.grow_masks([m.astype(np.float32) / 255], px=2, radius=0, grid=16)[0]
    cells = grown.reshape(4, 16, 6, 16)
    assert ((cells.max(axis=(1, 3)) == cells.min(axis=(1, 3)))).all()      # every cell is whole
    assert (grown >= H.grow_masks([m.astype(np.float32) / 255], px=2, radius=0)[0]).all()   # only outward
    assert H.MASK_GRID_PX == 16 and V.MASK_GRID_PX == 8


def test_tone_gap_follows_the_canvas_scale():
    f = H.H3Fill({1: np.zeros((96, 128, 3), np.uint8)}, 512, 384, False)
    assert f.tone_gap_px == (H.MASK_GROW_PX + H.MASK_GRID_PX) * 4
    v = V.VaceFill({1: np.zeros((96, 128, 3), np.uint8)}, 512, 384, False)
    assert v.tone_gap_px == (V.MASK_GROW_PX + V.MASK_GRID_PX) * 4
