# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z23 – brush strokes with a radius per point ([[x, y, r], ...], pen pressure / size changes).

The old format ([[x, y]] + radius_px) gives exactly the same mask as before (bit for bit); a variable-radius stroke is
the union of the convex hulls of neighbouring discs (pixels within +-1 px of the analytic shape); the key API accepts it.
"""

import cv2
import numpy as np

from engine.keys import apply_strokes
from engine.zones import Zone


def _old_apply(base, strokes):
    """Reference: apply_strokes before Z23 (constant radius)."""
    m = (base > 0.5).astype(np.uint8)
    for s in strokes:
        val = 1 if s.get("mode", "add") == "add" else 0
        r = max(1, int(round(s.get("radius_px", 8))))
        pts = np.round(np.array(s.get("p", []), np.float32)).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(m, tuple(pts[0]), r, val, -1)
        elif len(pts) > 1:
            cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, val, thickness=2 * r, lineType=cv2.LINE_8)
            for p in pts:
                cv2.circle(m, tuple(p), r, val, -1)
    return m.astype(bool)


def _hull_sdf(h, w, pts):
    """Signed distance (px, + inside) to the union over segments of {|p - c(t)| <= r(t)} (linear interpolation)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full((h, w), -1e9, np.float32)
    seq = pts if len(pts) > 1 else pts * 2
    for (x0, y0, r0), (x1, y1, r1) in zip(seq[:-1], seq[1:]):
        for t in np.linspace(0, 1, 400):
            cx, cy, r = x0 + t * (x1 - x0), y0 + t * (y1 - y0), r0 + t * (r1 - r0)
            best = np.maximum(best, r - np.hypot(xx - cx, yy - cy))
    return best


def test_old_format_bitwise_unchanged():
    rng = np.random.default_rng(4)
    base = rng.random((300, 400)) > 0.97
    strokes = [{"mode": "add", "radius_px": 7, "p": [[30.4, 40.6], [120.2, 90.9], [200, 95], [260.7, 30.1]]},
               {"mode": "remove", "radius_px": 3.6, "p": [[100, 100], [140, 160]]},
               {"mode": "add", "radius_px": 12, "p": [[350, 250]]}]
    assert np.array_equal(apply_strokes(base, strokes), _old_apply(base, strokes))


def test_variable_radius_matches_capsules():
    h, w = 160, 260
    pts = [(20.0, 50.0, 3.0), (120.0, 50.0, 15.0), (200.0, 110.0, 6.0), (230.0, 112.0, 30.0)]   # last: disc swallows a neighbour
    m = apply_strokes(np.zeros((h, w), bool), [{"mode": "add", "radius_px": 8, "p": [list(p) for p in pts]}])
    sdf = _hull_sdf(h, w, list(pts))
    want = sdf >= 0
    diff = m ^ want
    assert np.abs(sdf[diff]).max() <= 1.0, float(np.abs(sdf[diff]).max())     # disagreements only at the boundary (±1 px)
    assert diff.sum() < 0.05 * want.sum()
    # thin start, thick middle: the stroke really changes width
    assert m[:, 20].sum() <= 2 * 3 + 2 and m[:, 120].sum() >= 2 * 15 - 1
    # remove stroke with radii, mixed points (missing r -> radius_px)
    full = np.ones((h, w), bool)
    out = apply_strokes(full, [{"mode": "remove", "radius_px": 5, "p": [[60, 80, 10], [150, 80]]}])
    assert not out[80, 60] and not out[80, 150] and out[80 - 9, 150] and not out[80 - 9, 60]


def test_zone_mask_with_point_radii():
    z = Zone(zone_id="z_00", mode="add", criterion="all", frame=1, polygon=[],
             strokes=[{"mode": "add", "radius_px": 4, "p": [[20, 20, 2], [80, 20, 10]]}])
    from engine.zones import zone_mask_at
    m = zone_mask_at(z, (60, 120))
    assert m[20, 80 + 9] and not m[20 + 4, 22] and m.sum() > 0


def test_key_api_accepts_point_radii(tmp_path, monkeypatch):
    from test_ui_stages_api import _open
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    f = shot.first + 1
    st = [{"mode": "add", "radius_px": 6, "p": [[40, 40, 2], [140, 60, 20]]}]
    r = api.post(f"/api/object/puppet/keys/{f}", json={"source": "empty", "strokes": st})
    assert r.status_code == 200, r.text
    m = app_module._objects().key_mask("puppet", f)
    assert np.array_equal(m, apply_strokes(np.zeros((shot.h, shot.w), bool), st))
    assert mf.read_bytes() == before
