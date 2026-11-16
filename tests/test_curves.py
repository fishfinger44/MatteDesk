"""engine/curves.py - the one Bezier module (user decision 2026-11-09)."""
import json

import numpy as np
import pytest

from engine import curves as C


def _poly_err(sp, pts):
    s, _, _ = C.sample(sp, 0.25)
    return max(float(np.min(np.linalg.norm(s - p, axis=1))) for p in np.asarray(pts, float))


def test_from_points_is_catmull_rom_through_points():
    pts = [[0, 0], [100, 40], [200, 10], [300, 60]]
    sp = C.from_points(pts)
    s, u, par = C.sample(sp, 0.5)
    for i, p in enumerate(pts):                     # passes through every point at integer parameter
        j = int(np.argmin(np.abs(par - i)))
        assert np.allclose(s[j], p, atol=1e-6)
    assert u[0] == 0 and abs(u[-1] - 1) < 1e-12 and np.all(np.diff(u) >= 0)


def test_json_roundtrip_and_normalize():
    sp = C.fit([[x, 0.001 * (x - 500) ** 2] for x in range(0, 1001, 10)], tol=0.5)
    again = C.normalize(json.loads(json.dumps(sp)))
    assert again == sp


def test_fit_smooth_arc_few_points_within_tol():
    pts = [[x, 800 + 0.00002 * (x - 1900) ** 2] for x in np.arange(0, 3840, 10.0)]   # ~75 px sag (lens)
    sp = C.fit(pts, tol=1.0)
    assert len(sp["pts"]) <= 4                       # a horizon-like arc = a handful of points, not 384
    assert _poly_err(sp, pts) <= 1.0 + 1e-6
    assert not any(q["cusp"] for q in sp["pts"])


def test_fit_keeps_a_step_as_cusps():
    # VOID's horizon at key 1: flat, then a 16 px step, then flat (the case that survived 11 Catmull handles)
    pts = [[x, 882.0] for x in range(400, 1400, 10)] + [[1400, 890], [1410, 898]] + \
          [[x, 898.0] for x in range(1420, 2300, 10)]
    sp = C.fit(pts, tol=1.0, corner_deg=35)
    assert _poly_err(sp, pts) <= 1.0 + 1e-6
    assert sum(q["cusp"] for q in sp["pts"]) >= 2
    assert len(sp["pts"]) <= 8


def test_split_keeps_shape():
    sp = C.fit([[x, 50 * np.sin(x / 200)] for x in range(0, 1000, 5)], tol=0.5)
    a, _, _ = C.sample(sp, 0.25)
    sp2 = C.split(json.loads(json.dumps(sp)), 0.37)
    assert len(sp2["pts"]) == len(sp["pts"]) + 1
    b, _, _ = C.sample(sp2, 0.25)
    assert max(float(np.min(np.linalg.norm(b - p, axis=1))) for p in a[::7]) < 0.3


def test_correspond_by_x_matches_columns():
    src = C.from_points([[0, 100], [500, 120], [1000, 100]])
    dst = C.from_points([[0, 90], [1000, 90]])
    d, s = C.correspond(src, dst, "x", step=1.0)
    assert np.allclose(d[:, 0], s[:, 0])
    j = int(np.argmin(np.abs(d[:, 0] - 500)))
    assert abs(s[j, 1] - 120) < 0.5


def test_warp_moves_src_curve_onto_dst():
    cv2 = pytest.importorskip("cv2")
    h, w = 200, 400
    img = np.zeros((h, w), np.float32)
    img[:120] = 1.0                                 # an edge at y = 120 (the "VOID horizon")
    src = C.from_points([[0, 120], [400, 120]])
    dst = C.from_points([[0, 100], [400, 100]])
    out = C.apply_warp(img, [{"src": src, "dst": dst, "reach": 60, "match": "x"}])
    col = out[:, 200]
    edge = int(np.argmax(col < 0.5))
    assert abs(edge - 100) <= 1                     # the edge moved to the dst curve
    far = C.apply_warp(img, [{"src": src, "dst": dst, "reach": 10, "match": "x"}])
    assert np.allclose(far[:60], img[:60]) and np.allclose(far[160:], img[160:])   # untouched beyond reach


def test_legacy_point_lists_match_by_param():
    cv2 = pytest.importorskip("cv2")
    img = np.tile(np.linspace(0, 1, 300, dtype=np.float32)[:, None], (1, 300))
    wp = [{"src": [[0, 150], [150, 160], [299, 150]], "dst": [[0, 150], [150, 150], [299, 150]], "reach": 40}]
    dx, dy = C.warp_field(img.shape, wp)
    assert dy[150, 150] > 5 and abs(dy[150, 2]) < 2
