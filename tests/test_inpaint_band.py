# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""The band of Cleanup step C1 „Inpaint” (user decision 2026-11-11; engine/inpaint_band.py, engine/objband.inpaint_band):
„Pas na zewnątrz” = grow_px, „Wtopienie zewnętrzne” = soft_px, band points widen / narrow the patch locally, outer side
only, the cleaned thing's own band (per target), the preview, the anchors and the run read ONE mask function."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as A
from engine import cleanup as cu
from engine import inpaint_band as IB
from engine.objband import inpaint_band

W, H = 160, 120


def _obj():
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.hypot(xx - 60, yy - 60) < 18).astype(np.float32)


def _old_removal_mask(alphas, grow_px, soft_px):   # the mask before the band (2026-11-10), copied verbatim
    import cv2
    hard = (np.max(np.stack([np.asarray(a, np.float32) for a in alphas]), axis=0) > 0.02).astype(np.uint8)
    g = int(round(grow_px))
    if g > 0 and hard.any():
        hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    if soft_px <= 0 or not hard.any():
        return hard.astype(np.float32)
    d = cv2.distanceTransform((1 - hard).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - d / float(soft_px), 0.0, 1.0).astype(np.float32)


@pytest.mark.parametrize("grow,soft", [(0, 0), (8, 3), (4, 0), (0, 6), (23, 11)])
def test_no_points_is_bit_identical_to_before(grow, soft):
    obj = _obj()
    keep = np.zeros_like(obj)
    keep[40:80, 90:110] = 1
    s = cu.settings({"grow_px": grow, "soft_px": soft})
    old = _old_removal_mask([obj], grow, soft)
    assert np.array_equal(cu.removal_mask([obj], grow, soft), old)
    assert np.array_equal(cu.final_mask([obj], [keep], s), old * (1.0 - keep))
    m = cu.mask_fn(s, [lambda f: obj], [lambda f: keep])
    assert np.array_equal(m(3), old * (1.0 - keep))                      # no points: the band adds nothing


def test_settings_model_outer_side_only():
    b = inpaint_band(cu.settings({"grow_px": 12, "soft_px": 5}))
    assert (b.out_px, b.in_px, b.inset_feather_px, b.outer_feather_px) == (12.0, 0.0, 0.0, 5.0)
    assert b.key == "c1" and not b.shared and b.band_points == []


def test_points_widen_the_patch_locally():
    obj = _obj()
    # one point on the right edge (x=78, y=60) reaching 30 px out, set when the slider was 8
    s = cu.settings({"grow_px": 8, "soft_px": 0, "band_ref_out": 8.0,
                     "band_points": [{"id": "p1", "keys": {"1": {"x": 78.0, "y": 60.0, "out": 30.0, "in": 0.0}}}]})
    pf = IB.points_fn(s, lambda f: [obj])
    assert pf is not None and pf(1)
    m = cu.mask_fn(s, [lambda f: obj], [])(1)
    plain = cu.removal_mask([obj], 8, 0)
    assert m[60, 60 + 18 + 20] == 1 and plain[60, 60 + 18 + 20] == 0     # the whisker side reaches 30 px out
    assert (m >= plain).all() or m[60, 60 - 18 - 6] == plain[60, 60 - 18 - 6]
    # the slider scales the points' widths: grow 4 = half of the reference -> 15 px out on the whisker side
    s2 = {**s, "grow_px": 4.0}
    m2 = cu.mask_fn(s2, [lambda f: obj], [])(1)
    assert m2[60, 60 + 18 + 12] == 1 and m2[60, 60 + 18 + 20] == 0
    # the run (Cleaner.mask) uses the same points
    c = cu.Cleaner(s, W, H, False, plate=lambda f: np.zeros((H, W, 3), np.float32), remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [])
    assert np.array_equal(c.mask(1), m)


def test_lines_show_band_and_soft_edge():
    obj = _obj()
    L = IB.lines([obj], 8.0, 4.0)
    # the band's outer edge only (2026-11-11: the object's outline looked like a second band line) + the soft end
    assert len(L["band_contours"]) == 1 and len(L["feather_contours"]) == 1
    assert max(p[0] for p in L["band_contours"][0]) >= 60 + 18 + 8 - 1
    xs = [p[0] for p in L["feather_contours"][0]]
    assert 60 + 18 + 8 <= max(xs) <= 60 + 18 + 8 + 5
    assert IB.lines([obj], 0.0, 0.0)["band_contours"]                            # no band: the outline itself


def test_band_is_per_cleaned_thing():
    s = cu.settings({"remove": ["a"], "grow_px": 20, "band_points": [{"id": "p1", "keys": {}}], "band_ref_out": 20})
    cu.switch_target(s, "a", "b")
    assert s["grow_px"] == cu.DEFAULTS["grow_px"] and s["band_points"] == [] and s["band_ref_out"] is None
    s["grow_px"] = 3.0
    cu.switch_target(s, "b", "a")
    assert s["grow_px"] == 20 and s["band_points"] == [{"id": "p1", "keys": {}}] and s["band_ref_out"] == 20
    assert cu.settings(s)["per_target"]["b"]["grow_px"] == 3.0


@pytest.fixture
def band_app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(tmp_path / "b" / "f_####.png")}})
    monkeypatch.setattr(A, "_plate_size", lambda: (W, H))
    monkeypatch.setattr(A, "_work_frames", lambda: list(range(1, 11)))

    class Objs:
        def list(self, groups=True):
            return [{"name": "obj"}]

        def dir(self, n):
            return tmp_path / n
    import engine.cleanup_masks as CM
    import engine.export as EX
    monkeypatch.setattr(A, "_objects", lambda: Objs())
    monkeypatch.setattr(CM, "resolve", lambda store, o, src: {"run": "r1", "source": "core", "valid": True, "by": "own", "why": ""})
    obj = _obj()
    monkeypatch.setattr(EX, "run_alpha_loader", lambda d, run, w, h: (lambda f: obj))
    cu.save_settings(work, {"remove": ["obj"], "grow_px": 8, "soft_px": 2})
    return TestClient(A.app), work


def test_band_endpoints(band_app):
    c, work = band_app
    assert c.get("/api/cleanup/band/3").json()["points"] == []
    L = c.get("/api/cleanup/band/3/lines?grow_px=10&soft_px=0").json()
    assert L["band_contours"] and L["feather_contours"] == []
    r = c.put("/api/cleanup/band/3", json={"x": 78, "y": 60})
    assert r.status_code == 200, r.text
    pid = r.json()["point"]
    r = c.put("/api/cleanup/band/3", json={"id": pid, "out": 30})
    assert r.status_code == 200, r.text
    s = cu.load_settings(work)
    assert len(s["band_points"]) == 1 and s["band_ref_out"] == 8.0
    st = c.get("/api/cleanup/band/5").json()
    assert st["points"][0]["id"] == pid and abs(st["points"][0]["out"] - 30) < 1
    prev = c.get("/api/cleanup/band/params").json()
    assert prev["band_points"][0]["id"] == pid
    # the mask preview uses the point
    import cv2
    png = c.get("/api/cleanup/mask/3").content
    m = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[..., 2]   # R = the patch (G = model area)
    assert m is not None
    assert c.delete("/api/cleanup/band?frame=3").json()["removed"] == [pid]
    assert cu.load_settings(work)["band_points"] == [] and cu.load_settings(work)["band_ref_out"] is None
    assert c.put("/api/cleanup/band/params", json=prev).status_code == 200      # Ctrl+Z
    assert cu.load_settings(work)["band_points"][0]["id"] == pid
    assert c.delete(f"/api/cleanup/band/{pid}").status_code == 200
    assert cu.load_settings(work)["band_points"] == []


def test_preview_lines_match_the_mask_and_are_fast():
    """2026-11-11: the preview lines without points come from one distance transform (constant time) – within 1.5 px
    of removal_mask's own edges, and not slower for a wide band."""
    import time
    import cv2
    obj = np.zeros((600, 400), np.float32)
    cv2.circle(obj, (200, 300), 120, 1.0, -1)
    for g, s in ((12, 4), (90, 40)):
        L = IB.lines([obj], float(g), float(s))
        m = cu.removal_mask([obj], float(g), float(s))
        solid_edge = max(p[0] for p in L["band_contours"][0])
        want = max(np.where((m >= 0.999).any(axis=0))[0])
        assert abs(solid_edge - want) <= 1.5, (g, solid_edge, want)
        soft_edge = max(p[0] for p in L["feather_contours"][0])
        assert abs(soft_edge - max(np.where((m > 0).any(axis=0))[0])) <= 1.5
    t0 = time.perf_counter()
    IB.lines([obj], 150.0, 60.0)
    assert time.perf_counter() - t0 < 0.5
