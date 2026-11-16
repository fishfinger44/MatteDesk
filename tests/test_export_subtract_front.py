"""Audit 2026-11-02 #2: `subtract_front` = clamp(back − max(front)) (module docstring, spec 5.2 / MatteSubtractFront),
never back − Σ front. The export context lists a group AND its members, so an object behind both used to lose its soft
edge twice (dark fringe)."""
import numpy as np

from engine.export import subtract_front


def test_subtract_front_uses_max_not_sum():
    back = np.full((4, 4), 1.0, np.float32)
    a = np.full((4, 4), 0.5, np.float32)
    b = np.full((4, 4), 0.5, np.float32)
    out = subtract_front(back, [a, b])
    assert np.allclose(out, 0.5)                       # sum would give 0.0


def test_subtract_front_single_and_empty():
    back = np.array([[0.2, 0.8]], np.float32)
    assert np.allclose(subtract_front(back, []), back)
    assert np.allclose(subtract_front(back, [np.array([[0.5, 0.5]], np.float32)]), [[0.0, 0.3]])
    assert subtract_front(back, []).dtype == np.float32
