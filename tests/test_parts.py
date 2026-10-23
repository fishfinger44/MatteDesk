# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parts of one object (part_of): Figure's dress is part of Figure. The whole is one seamless matte, the parts sum to it
exactly (no gap, no overlap at the contact); the dress keeps its own matte for holdouts (user, 2026-10-12)."""

import json

import cv2
import numpy as np

from engine.crop import CropSegment
from engine.exr_io import read_exr, write_gray_exr
from engine.export import export_objects
from engine.objects import ObjectStore

W, H = 200, 160


def _soft(mask, r=2.0):
    """Antialiased matte: soft ramp across the edge (both parts soft at the contact -> a seam under max())."""
    d_in = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(0.5 + (d_in - d_out) / (2 * r), 0, 1).astype(np.float32)


def _object(store, name, mask, depth, part_of=None):
    store.create(name, [name], depth_order=depth)
    od = store.dir(name)
    seg = CropSegment(seg_id="s00", first=1, last=1, overlap_in=0, overlap_out=0, crop=[0, 0, W, W],
                      pad=[0, 0, 0, W - H], scale_to_1008=5.04)
    (od / "runs" / "v001" / "alpha" / "s00").mkdir(parents=True)
    a = np.zeros((W, W), np.float32)
    a[:H] = _soft(mask)
    write_gray_exr(od / "runs" / "v001" / "alpha" / "s00" / "a_0001.exr", a)
    (od / "runs" / "v001" / "run.json").write_text(json.dumps({"edge_mode": "core", "segments_def": [seg.to_dict()]}))
    (od / "core").mkdir(exist_ok=True)
    cv2.imwrite(str(od / "core" / "c_0001.png"), mask.astype(np.uint8) * 255)
    store.update(name, part_of=part_of, tool={"active_run": "v001", "final_valid": True})


def _shot(tmp_path, part):
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    yy, xx = np.mgrid[0:H, 0:W]
    body = ((xx - 100) ** 2 + (yy - 50) ** 2 <= 30 ** 2) | ((abs(xx - 100) <= 12) & (yy >= 50) & (yy < 80))
    dress = (abs(xx - 100) <= 50) & (yy >= 80) & (yy < 140) & ~body
    _object(store, "figure", body, 0)
    _object(store, "suknia", dress, 1, part_of="figure" if part else None)
    export_objects(work, "shot", store.list(), [1], W, H)
    ch, _ = read_exr(work / "export" / "mattes" / "all" / "shot_mattes_0001.exr")
    return work, ch, body, dress


def test_parts_sum_to_seamless_whole(tmp_path):
    work, ch, body, dress = _shot(tmp_path, part=True)
    full, b, d = ch["figure_full.alpha"], ch["figure.alpha"], ch["suknia.alpha"]
    contact = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & \
        cv2.dilate(dress.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert contact.any() and full[contact].min() >= 0.99            # no seam between the body and the dress
    assert np.abs(b + d - full).max() < 2e-3                         # parts sum to the whole: no gap, no overlap
    assert d.min() >= 0 and (b > 0.5).sum() > 0 and (d > 0.5).sum() > 0
    out = json.loads((work / "outputs.json").read_text(encoding="utf-8"))
    assert out["groups"]["figure"]["members"] == ["figure", "suknia"]
    assert (work / "export" / "mattes" / "figure_full" / "figure_full_0001.exr").exists()
    # the outer silhouette stays soft (only the contact is filled)
    outer = cv2.dilate((body | dress).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~(body | dress) & ~contact
    assert 0.0 < full[outer].mean() < 0.6


def test_without_part_of_the_contact_has_a_seam(tmp_path):
    """Reference: two separate objects (front subtraction) leave a soft seam at the contact – that is why parts exist."""
    _, ch, body, dress = _shot(tmp_path, part=False)
    contact = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & \
        cv2.dilate(dress.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert ch["union.alpha"][contact].min() < 0.9


def test_api_part_of_rules(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    for n in ("figure", "suknia", "drzewo"):
        store.create(n, [n])
    store.update("figure", tool={"active_run": "v001", "final_valid": True})
    monkeypatch.setitem(app_module.STATE, "work_dir", str(work))
    api = TestClient(app_module.app)
    r = api.put("/api/object/suknia/settings", json={"part_of": "figure"})
    assert r.status_code == 200 and store.get("suknia")["part_of"] == "figure"
    assert store.get("figure")["tool"]["final_valid"]                       # relation only: no recompute needed
    assert api.put("/api/object/suknia/settings", json={"part_of": "suknia"}).status_code == 400
    assert api.put("/api/object/drzewo/settings", json={"part_of": "suknia"}).status_code == 400   # suknia is a part
    assert api.put("/api/object/figure/settings", json={"part_of": "drzewo"}).status_code == 400    # figure has parts
    assert api.delete("/api/object/figure").status_code == 200
    assert store.get("suknia").get("part_of") is None                      # orphaned part -> standalone


def test_three_parts_sum_to_whole(tmp_path):
    """A main object with several parts (e.g. Figure: body, dress, horns): parts still sum to the seamless whole."""
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    yy, xx = np.mgrid[0:H, 0:W]
    horns = (abs(xx - 100) <= 25) & (yy >= 10) & (yy < 30)
    body = (abs(xx - 100) <= 20) & (yy >= 30) & (yy < 80)
    dress = (abs(xx - 100) <= 50) & (yy >= 80) & (yy < 140)
    _object(store, "figure", body, 0)
    _object(store, "rogi", horns, 1, part_of="figure")
    _object(store, "suknia", dress, 2, part_of="figure")
    export_objects(work, "shot", store.list(), [1], W, H)
    ch, _ = read_exr(work / "export" / "mattes" / "all" / "shot_mattes_0001.exr")
    full = ch["figure_full.alpha"]
    total = ch["figure.alpha"] + ch["rogi.alpha"] + ch["suknia.alpha"]
    assert np.abs(total - full).max() < 3e-3
    for y0 in (30, 80):   # contact lines horns|body and body|dress, inside the narrower part's width
        assert full[y0 - 3:y0 + 3, 100 - 16:100 + 17].min() >= 0.99
    assert json.loads((work / "outputs.json").read_text())["groups"]["figure"]["members"] == ["figure", "rogi", "suknia"]


def test_single_export_equals_full_export(tmp_path):
    """Export only one object (or only the whole) -> the same matte as in the full export; nothing else written,
    the shot's multi-EXR untouched."""
    import shutil
    work, ch, body, dress = _shot(tmp_path, part=True)
    store = ObjectStore(work)
    mroot = work / "export" / "mattes"
    full_dress, _ = read_exr(mroot / "suknia" / "suknia_0001.exr")
    full_whole, _ = read_exr(mroot / "figure_full" / "figure_full_0001.exr")
    shutil.rmtree(mroot)
    export_objects(work, "shot", store.list(), [1], W, H, write=["suknia"])
    assert sorted(p.name for p in mroot.iterdir()) == ["suknia"]
    one, _ = read_exr(mroot / "suknia" / "suknia_0001.exr")
    assert np.abs(one["R"] - full_dress["R"]).max() < 1e-3
    export_objects(work, "shot", store.list(), [1], W, H, write=["figure_full"])
    assert sorted(p.name for p in mroot.iterdir()) == ["figure_full", "suknia"]
    whole, _ = read_exr(mroot / "figure_full" / "figure_full_0001.exr")
    assert np.abs(whole["R"] - full_whole["R"]).max() < 1e-3
    out = json.loads((work / "outputs.json").read_text(encoding="utf-8"))
    assert {e["name"] for e in out["objects"]} == {"figure", "suknia"}     # earlier entries kept


def test_three_objects_seam_fills_every_gap(tmp_path, monkeypatch):
    """Group of 3 objects (main + 2 parts, user check 2026-10-23): the main object's part_seam_px fills the gap at BOTH
    contacts of the whole; seam 0 leaves them open. Also via the API: two parts can be attached to one main object."""
    from fastapi.testclient import TestClient
    import app as app_module
    from engine.stages import group_parts
    yy, xx = np.mgrid[0:H, 0:W]
    top = (abs(xx - 100) <= 25) & (yy >= 10) & (yy < 30)
    mid = (abs(xx - 100) <= 25) & (yy >= 34) & (yy < 80)       # 4 px gap to top
    low = (abs(xx - 100) <= 25) & (yy >= 84) & (yy < 140)      # 4 px gap to mid
    gaps = [(slice(30, 34), slice(80, 121)), (slice(80, 84), slice(80, 121))]
    for seam, filled in ((4, True), (0, False)):
        work = tmp_path / f"s{seam}" / "matte_desk"
        store = ObjectStore(work)
        _object(store, "a", top, 0)                  # main = head: touches only the torso (b), never the legs (c)
        _object(store, "b", mid, 1, part_of="a")
        _object(store, "c", low, 2)                  # legs: touch only the torso – seams are filled between ANY two members
        monkeypatch.setitem(app_module.STATE, "work_dir", str(work))
        api = TestClient(app_module.app)
        assert api.put("/api/object/c/settings", json={"part_of": "a"}).status_code == 200   # 2nd part of the same main
        assert api.put("/api/object/a/settings", json={"part_seam_px": seam}).status_code == 200
        assert group_parts(store, "a") == ["b", "c"]
        export_objects(work, "shot", store.list(), [1], W, H)
        ch, _ = read_exr(work / "export" / "mattes" / "all" / "shot_mattes_0001.exr")
        full = ch["a_full.alpha"]
        for g in gaps:
            assert (full[g].min() >= 0.99) == filled, (seam, g)
        assert np.abs(ch["a.alpha"] + ch["b.alpha"] + ch["c.alpha"] - full).max() < 3e-3   # parts still sum to the whole
