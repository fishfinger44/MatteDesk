# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z24 – colour of the "color" zone criterion: pipette (median Lab of 5 x 5 px) or the median of the drawn area.

Bug fixed: the UI never sent `color_lab`, so `criterion_key` always compared with the default grey [50, 0, 0].
"""

import json

import numpy as np

from engine.zones import Zone, criterion_key, lab_of, sample_lab, zone_median_lab


def _img():
    img = np.zeros((60, 90, 3), np.float32)
    img[:, :30] = (0.1, 0.6, 0.2)       # green (BGR)
    img[:, 30:60] = (0.8, 0.2, 0.1)     # blue
    img[:, 60:] = (0.5, 0.5, 0.5)       # grey
    img += np.random.default_rng(1).normal(0, 0.01, img.shape).astype(np.float32)
    return img


def test_pipette_colour_selects_only_similar_pixels():
    img = _img()
    lab = sample_lab(img[28:33, 43:48])                     # 5 x 5 px inside the blue band
    z = Zone("z_00", "add", "color", 1, polygon=[[0, 0], [89, 0], [89, 59], [0, 59]], param=10.0, color_lab=lab)
    k = criterion_key(img, z) > 0.5
    de = np.linalg.norm(lab_of(img) - np.asarray(lab, np.float32), axis=2)
    assert np.array_equal(k, de < 10.0)
    assert k[:, 32:58].all() and not k[:, :28].any() and not k[:, 62:].any()   # only the blue band
    # the old behaviour (no colour -> grey [50, 0, 0]) would have picked the grey band instead
    z_old = Zone("z_01", "add", "color", 1, param=10.0)
    assert (criterion_key(img, z_old) > 0.5)[:, 62:].mean() > 0.9


def test_zone_without_pick_uses_area_median():
    img = _img()
    z = Zone("z_00", "add", "color", 1, polygon=[[2, 2], [25, 2], [25, 55], [2, 55]], param=10.0)
    lab = zone_median_lab(z, img)
    assert np.linalg.norm(np.asarray(lab) - np.asarray(sample_lab(img[20:25, 10:15]))) < 2.0


def test_color_pick_api_and_zone_gets_colour(tmp_path, monkeypatch):
    from test_ui_stages_api import _open
    import app as app_module
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    f = shot.first + 2
    ys, xs = np.where(shot.masks[f] > 0.5)
    px, py = int(np.median(xs)), int(np.median(ys))
    r = api.get(f"/api/object/puppet/color_lab/{f}?x={px}&y={py}")
    assert r.status_code == 200, r.text
    d = r.json()
    from engine.pipeline import PlateCropLoader
    ref = sample_lab(PlateCropLoader(shot.plate_pattern, app_module._plate_display).read(f)[py - 2:py + 3, px - 2:px + 3])
    assert d["lab"] == ref and d["hex"].startswith("#") and len(d["hex"]) == 7
    assert api.get(f"/api/object/puppet/color_lab/{f}?x=-3&y=5").status_code == 400
    # zone "color" saved without a pick -> median of the drawn area, not the default grey
    poly = [[px - 6, py - 6], [px + 6, py - 6], [px + 6, py + 6], [px - 6, py + 6]]
    r = api.post("/api/object/puppet/zones/new", json={"mode": "add", "criterion": "color", "frame": f, "polygon": poly, "param": 10})
    assert r.status_code == 200, r.text
    lab = r.json()["zone"]["color_lab"]
    assert lab and np.linalg.norm(np.asarray(lab) - np.asarray(d["lab"])) < 5.0 and lab != [50, 0, 0]
    # with a pipette pick the colour is kept as sent
    r = api.post("/api/object/puppet/zones/new", json={"mode": "add", "criterion": "color", "frame": f, "polygon": poly,
                                                        "param": 10, "color_lab": [40.0, 10.0, -20.0]})
    assert r.json()["zone"]["color_lab"] == [40.0, 10.0, -20.0]
    assert mf.read_bytes() == before
