# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""C1 grain match (user decision 2026-11-13): a video model's fill is cleaner than a noisy plate; the Cleaner adds grain
of the plate's spectrum inside the patch only, through engine/cleanup3d/grain.py (Cleanup 3D's module, not a copy),
after the levelling, seeded per frame; `grain` off = the result before the feature."""
import cv2
import numpy as np

from engine import cleanup as cu

W, H = 256, 192
SIGMA = 0.025


def _scene(seed=3):
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.dstack([0.25 + 0.3 * xx / W, 0.35 + 0.2 * yy / H, np.full((H, W), 0.45)]).astype(np.float32)
    obj = (np.hypot(xx - 128, yy - 96) < 28).astype(np.float32)
    rng = np.random.default_rng(seed)
    plate = bg + rng.normal(0, SIGMA, bg.shape).astype(np.float32)
    plate[obj > 0] = (0.9, 0.1, 0.1)
    return bg, obj, plate


class _Fill:
    """A stand-in video model: the clean background (no grain) everywhere, weight 1."""
    label = "MiniMax-H3"
    tone = True
    tone_gap_px = 6.0

    def __init__(self, bg, grain=None):
        self.img = bg.copy()
        if grain is not None:
            self.grain = grain

    def __call__(self, f):
        return self.img, np.ones(self.img.shape[:2], np.float32)


def _run(tmp, fill, s=None, frames=(1, 2)):
    bg, obj, plate = _scene()
    c = cu.Cleaner({"grow_px": 6, "soft_px": 2, "tone": False, **(s or {})}, W, H, False, plate=lambda f: plate,
                   remove_alphas=lambda f: [obj], keep_alphas=lambda f: [], key_fill=fill)
    rj = c.run(list(frames), tmp, "png16")
    comps = {f: cv2.imread(str(tmp / "comp" / f"c_{f:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
             for f in frames}
    return rj, comps, bg, plate, c.mask(frames[0])


def test_grain_added_only_inside_the_patch(tmp_path):
    bg, _, _ = _scene()
    rj, comps, bg, plate, m = _run(tmp_path / "on", _Fill(bg))
    rj0, comps0, *_ = _run(tmp_path / "off", _Fill(bg), {"grain": "off"})
    g = rj["grain"]
    assert g and g["added"] == 2 and g["measured"] == 2 and g["mode"] == "auto"
    assert g["plate"] > 2 * g["patch"]                          # the plate is the noisier, measured on the ring
    assert "grain" not in rj0                                   # off: nothing measured, nothing added
    inside = m > 0.999
    noise_on = float((comps[1] - bg)[inside].std())
    noise_off = float((comps0[1] - bg)[inside].std())
    assert noise_off < 0.002 and 0.5 * SIGMA < noise_on < 1.6 * SIGMA
    out = m == 0
    assert np.array_equal(comps[1][out], comps0[1][out])        # nothing outside the patch moves
    assert np.abs(comps[1][out] - np.clip(plate[out], 0, 1)).max() < 1e-4
    # new noise every frame (seeded by the frame)
    assert not np.array_equal(comps[1][inside], comps[2][inside])


def test_grain_is_seeded_a_rerun_is_identical(tmp_path):
    bg, _, _ = _scene()
    _, a, *_ = _run(tmp_path / "a", _Fill(bg))
    _, b, *_ = _run(tmp_path / "b", _Fill(bg))
    assert all(np.array_equal(a[f], b[f]) for f in a)


def test_grain_off_is_the_result_before_the_feature(tmp_path):
    """off = bit for bit a fill that never takes grain (what the Cleaner did before 2026-11-13)."""
    bg, _, _ = _scene()
    _, off, *_ = _run(tmp_path / "off", _Fill(bg), {"grain": "off"})
    _, never, *_ = _run(tmp_path / "never", _Fill(bg, grain=False))
    assert all(np.array_equal(off[f], never[f]) for f in off)


def test_strength_scales_and_clean_plate_adds_nothing(tmp_path):
    bg, obj, _ = _scene()
    _, half, bg, _, m = _run(tmp_path / "half", _Fill(bg), {"grain_strength": 0.5})
    inside = m > 0.999
    n_half = float((half[1] - bg)[inside].std())
    assert 0.25 * SIGMA < n_half < 0.8 * SIGMA
    # a clean plate (denoised on input): auto adds nothing
    c = cu.Cleaner({"grow_px": 6, "soft_px": 2, "tone": False}, W, H, False, plate=lambda f: np.where(obj[..., None] > 0, 0.5, bg),
                   remove_alphas=lambda f: [obj], keep_alphas=lambda f: [], key_fill=_Fill(bg))
    rj = c.run([1], tmp_path / "clean", "png16")
    assert rj["grain"]["added"] == 0


def test_settings_and_fill_marks():
    s = cu.settings()
    assert s["grain"] == "auto" and s["grain_strength"] == 1.0
    assert cu.settings({"grain": "bogus", "grain_strength": 9})["grain"] == "auto"
    assert cu.settings({"grain_strength": 9})["grain_strength"] == 2.0
    from engine import h3_fill, inpaint_upscale, ltx_fill, vace_fill
    for cls in (h3_fill.H3Fill, vace_fill.VaceFill, ltx_fill.LtxFill, cu.VoidKeyFill):
        assert getattr(cls, "grain", getattr(cls, "tone", False))
    assert inpaint_upscale.UpFill.grain is True
    # Cleanup 3D's key fill is grained in its own render, never again by the Cleaner
    from engine.cleanup3d import render as R
    assert not getattr(R.KeyFill, "grain", getattr(R.KeyFill, "tone", False))


def test_shot_grain_measured_once_and_combined_2026_11_14():
    """Speed: Inpaint measures the grain ONCE per run (grain.combine of a few frames) instead of every frame."""
    import numpy as np
    from engine.cleanup3d import grain as GR
    a = {"plate": np.ones((4, 3)), "patch": np.zeros((4, 3)), "plate_sigma": np.array([1.0, 1, 1]),
         "patch_sigma": np.array([0.0, 0, 0]), "px": 10, "lum": None}
    b = {**a, "plate": np.ones((4, 3)) * 3, "plate_sigma": np.array([3.0, 3, 3]), "px": 30}
    m = GR.combine([a, None, b])
    assert m["frames"] == 2 and m["px"] == 40 and float(m["plate_sigma"][0]) == 2.0 and GR.combine([None]) is None
