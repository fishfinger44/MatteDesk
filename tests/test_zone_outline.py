# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Zone region mode "outline" (user decision 2026-10-15): the ViTMatte zone = a band around the WHOLE core edge; the
drawing and the chosen objects are EXCLUDED from it (the user marks where ViTMatte must not work, not where the hair
is); ViTMatte runs on tiles so the long outline keeps the x2.25 up-scale of the old zone crop."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine.zones import (OUTLINE_TILE_PX, Zone, apply_zones, outline_band, outline_in_px, outline_regions,  # noqa: E402
                          region_mode, vitmatte_tile_px, vitmatte_zone)


def _disc(h=200, w=260, r=60):
    c = np.zeros((h, w), np.uint8)
    cv2.circle(c, (w // 2, h // 2), r, 1, -1)
    return c.astype(bool)


def test_band_follows_the_core_and_drops_exclusions():
    core = _disc()
    band = outline_band(core, 20, 6)
    d_out = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    assert band[(~core) & (d_out <= 18)].all() and not band[(~core) & (d_out > 22)].any()
    assert band[core & (d_in <= 5)].all() and not band[core & (d_in > 8)].any()
    z = Zone("z_01", "add", "matte", 1, matting="vitmatte", params={"region": "outline", "outline_out_px": 20,
                                                                        "fg_erode_px": 2, "exclude_grow_px": 3})
    assert outline_in_px(z.params) == 6 and outline_in_px({"force_core_erode_px": 10}) == 14 and outline_in_px({"outline_in_px": 1}) == 1
    drawn = np.zeros_like(core); drawn[:, :90] = True                    # "skin" drawn on the left = excluded
    hand = np.zeros_like(core); hand[150:, 120:150] = True               # an object (hand) at the bottom
    reg = outline_regions(z, {1: core}, {1: drawn}, lambda f: hand)[1]
    assert not reg[drawn].any() and not reg[hand].any()
    assert not reg[147:, 117:153].any()                                 # grown by exclude_grow_px
    assert reg[:, 200:].any() and (reg & band).sum() == reg.sum()       # the rest of the band stays
    assert region_mode(z) == "outline" and region_mode(Zone("z", "add", "matte", 1)) == "draw"
    with pytest.raises(ValueError):
        region_mode(Zone("z", "add", "matte", 1, params={"region": "bogus"}))
    assert vitmatte_tile_px(z.params) == OUTLINE_TILE_PX and vitmatte_tile_px({}) == -1 and vitmatte_tile_px({"region": "outline", "tile_px": 0}) == 0


def test_tiles_match_one_crop_and_cover_the_outline():
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(1)
    f = frames[0]
    z = Zone("z_01", "add", "matte", f, matting="vitmatte")
    band = outline_band(core, 40, 4)
    r1, r2 = MockRunner(), MockRunner()
    one, _, st1 = vitmatte_zone(z, [f], {f: core}, {f: band}, lambda _: img, r1, model_px=256)
    til, tris, st2 = vitmatte_zone(z, [f], {f: core}, {f: band}, lambda _: img, r2, model_px=256, tile_px=96)
    assert len(r1.calls) == 1 and st2["tiles"] == len(r2.calls) > 2 and st2["tile_px"] == 96
    assert st2["scale"] > st1["scale"]                                  # each tile scaled up more than the one crop
    a1, a2 = one[f], til[f]
    m = band & ~core
    assert np.abs(a1[m] - a2[m]).mean() < 0.03, np.abs(a1[m] - a2[m]).mean()   # same matte (resampling only)
    assert (a2[hair & m] > 0.3).mean() > 0.8                           # the strands are there
    assert a2[~band].max() == 0.0 and a2[band & core].min() == 1.0 and tris[f].shape == core.shape


def test_apply_zones_outline_without_drawing_and_with_an_exclusion(monkeypatch):
    import engine.vitmatte as vm
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(3)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    params = {"region": "outline", "outline_out_px": 40, "tile_px": 128, "temporal": "none"}
    z0 = Zone("z_01", "add", "matte", frames[0], matting="vitmatte", params=params)            # no drawing at all
    r0 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z0], tr)
    band = outline_band(core, 40, outline_in_px(params))
    a0, area0 = r0.soft[frames[0]]
    assert np.array_equal(area0, band) and r0.stats["z_01"]["region"] == "outline" and r0.stats["z_01"]["tiles"] > 0
    assert (a0[hair & band & ~core] > 0.3).mean() > 0.8
    ys, xs = np.where(hair & band & ~core)
    x_mid = int(np.median(xs))
    poly = [[0, 0], [x_mid, 0], [x_mid, core.shape[0]], [0, core.shape[0]]]                    # left half excluded
    z1 = Zone("z_01", "add", "matte", frames[0], polygon=poly, matting="vitmatte", params=params)
    a1, area1 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z1], tr).soft[frames[0]]
    assert not area1[:, :x_mid].any() and np.array_equal(area1[:, x_mid + 1:], band[:, x_mid + 1:])


def test_outline_feathers_blend_into_core_and_black(monkeypatch):
    """User rule 2026-10-22: in a whole-outline zone the inset feather blends the CORE (1) into the matte from the inner
    band edge, the outer feather the matte into the core (black where there is none) – not into the stage base."""
    import engine.vitmatte as vm
    from engine.zones import feather_maps, outline_in_px
    from test_zone_vitmatte import MockRunner, _frames_scene
    img, core, hair, region, frames = _frames_scene(3)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    params = {"region": "outline", "outline_out_px": 40, "outer_feather_px": 10, "inset_feather_px": 6, "temporal": "none"}
    z = Zone("z_01", "add", "matte", frames[0], matting="vitmatte", params=params)
    r = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr)
    f = frames[0]
    a, area = r.soft[f]
    assert (r.weights[f][area] == 255).all()                       # the area wins outright (partners are in the alpha)
    d_in, d_out, li, lo = feather_maps(area, core, 6, 10)
    assert a[area & (d_in < 1.5)].min() > 0.8                      # at the inner band edge: the core (1)
    assert a[area & (d_out < 1.5)].max() < 0.2                     # at the outer band edge: black (0)
    # an outer band edge ON the core (band 0 px outside): the outer feather blends into the core there, not into black
    z0 = Zone("z_01", "add", "matte", frames[0], matting="vitmatte", params={**params, "outline_out_px": 0})
    a0, area0 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z0], tr).soft[f]
    _, d_out0, _, _ = feather_maps(area0, core, 6, 10)
    assert a0[area0 & core & (d_out0 < 1.5)].min() > 0.8
    # a soft stage base (MatAnyone2 / VideoMaMa, user decision 2026-10-22): the outer feather blends into max(core, base)
    # – the outer band edge is continuous with the base, no step; the inner edge still goes to the core (1)
    soft = np.full(core.shape, 0.4, np.float32); soft[core] = 1.0
    ab, areab = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, base_alpha=lambda f: soft).soft[f]
    assert abs(ab[areab & (d_out < 1.5)].mean() - 0.4) < 0.08 and ab[areab & (d_in < 1.5)].min() > 0.8


def test_api_outline_zone_needs_no_drawing_and_excludes_objects(tmp_path, monkeypatch):
    import engine.vitmatte as vm
    from test_zone_matte import _api_project
    from test_zone_vitmatte import MockRunner
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": {"region": "outline", "outline_out_px": 30, "tile_px": 160, "temporal": "none"}}
    bad = api.post("/api/object/obj/zones/new", json={**body, "params": {}})
    assert bad.status_code == 400                                   # the old mode still needs a drawing
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=body)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    st = r.json()["preview"]["stats"]
    assert st.get("tiles", 0) > 0 and r.json()["preview"]["soft_px"] > 0
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert reg["area_px"] > 0 and reg["contours"]
    # nothing painted (2026-10-17): the area of any frame comes straight from its core – no CoTracker, no propagation
    from engine.pipeline import ObjectPipeline
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: pytest.fail("tracker started for an outline without drawing"))
    other = next(f for f in frames if f != start)
    reg_o = api.get(f"/api/object/obj/zones/{zid}/region/{other}").json()
    assert reg_o["area_px"] > 0 and reg_o["region_mode"] == "outline"
    # an object to exclude (core only) cuts its area out of the band
    store = pipe.store
    store.create("hand", ["hand"], depth_order=5)
    cd = store.dir("hand") / "core"; cd.mkdir(parents=True)
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    m = np.zeros((shot.h, shot.w), np.uint8)
    m[:, int(np.median(xs)):] = 255                                 # right half of the object = "hand"
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    p = api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**body["params"], "exclude_objects": ["hand"]}})
    assert p.status_code == 200, p.text
    reg2 = api.get(f"/api/object/obj/zones/{zid}/region/{start}").json()
    assert 0 < reg2["area_px"] < reg["area_px"] and reg2["bbox"][2] <= int(np.median(xs)) + 1
    api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**body["params"], "exclude_objects": ["nobody"]}})
    e = api.post(f"/api/object/obj/zones/{zid}/preview/{start}")
    assert e.status_code == 409 and "nobody" in e.text


def test_preview_forces_only_the_eroded_core(tmp_path, monkeypatch):
    """User report 2026-10-17: the frame preview forced the WHOLE core to 1 (pixel staircase of the SAM3 edge), the full
    run only the core eroded by force_core_erode_px – the preview now does the same."""
    import engine.vitmatte as vm
    from test_zone_matte import _api_project
    from test_zone_vitmatte import MockRunner

    class Half(MockRunner):        # unknown = 0.5 everywhere: a pasted 1 in the core rim is then unmistakable
        def run(self, img, tri, unknown_value=0.5):
            a = np.full(tri.shape, 0.5, np.float32)
            a[tri == 255] = 1.0
            a[tri == 0] = 0.0
            return a
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: Half())
    params = {"region": "outline", "outline_out_px": 30, "tile_px": 160, "temporal": "none", "alpha_black": 0.05}
    body = {"mode": "add", "criterion": "matte", "frame": start, "polygon": [], "matting": "vitmatte", "preset": "hair",
            "params": params}
    zid = api.post("/api/object/obj/zones/new", json=body).json()["zone"]["zone_id"]
    od = pipe.store.dir("obj")

    core = pipe.load_core("obj", [start])[start].astype(bool)

    def forced():
        assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 200
        a = cv2.imread(str(od / "preview" / f"zarea_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(od / "preview" / f"zraw_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
        ring = core & (a > 0) & (a < 255)                      # the core rim the zone decides (inside the inset)
        return int((a == 255).sum()), int((a > 0).sum()), raw, ring
    whole, _, _, _ = forced()
    api.patch(f"/api/object/obj/zones/{zid}", json={"params": {**params, "fg_erode_px": 8, "force_core_erode_px": 8}})
    eroded, region, raw, ring = forced()
    assert 0 < eroded < whole and region > eroded
    # the raw ViTMatte alpha in the core rim is the engine's, not a pasted 1 (the hard SAM3 edge, 2026-10-17)
    assert ring.any() and (raw[ring] < 200).mean() > 0.5


def test_zone_feather_px():
    from engine.refine import ZONE_FEATHER_PX
    from engine.zones import Zone, zone_feather_px
    assert zone_feather_px(Zone("z", "add", "matte", 1)) == ZONE_FEATHER_PX              # default = the former fixed value
    assert zone_feather_px(Zone("z", "add", "matte", 1, params={"inset_feather_px": 3})) == 3.0
    assert zone_feather_px(Zone("z", "add", "matte", 1, params={"inset_feather_px": 0})) == 0.5   # never a hard 0


def test_zone_core_view_is_the_group_core(tmp_path, monkeypatch):
    """User report 2026-10-21: the stage-5 band preview of a group (head + hand) went round the hand inside the hair –
    it used the object's own core. view=zone_core = the core the zone is computed on: a group's members joined with the
    seam (pipeline.preview_core, as stages 3 / 4); not a group -> the object's core."""
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    own = api.get(f"/api/object/obj/frame/{start}?view=core")
    alone = api.get(f"/api/object/obj/frame/{start}?view=zone_core")
    assert alone.status_code == 200 and alone.content == own.content
    store = pipe.store
    store.create("hand", ["hand"], depth_order=1)
    store.update("hand", part_of="obj")
    cd = store.dir("hand") / "core"; cd.mkdir(parents=True)
    ys, xs = np.where(pipe.load_core("obj", [start])[start])
    m = np.zeros((shot.h, shot.w), np.uint8)
    m[int(ys.min()):int(ys.max()) + 1, int(xs.max()) + 1:min(shot.w, int(xs.max()) + 20)] = 255   # touching on the right
    for f in frames:
        cv2.imwrite(str(cd / f"c_{f:04d}.png"), m)
    grp = api.get(f"/api/object/obj/frame/{start}?view=zone_core")
    assert grp.status_code == 200
    g = cv2.imdecode(np.frombuffer(grp.content, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
    o = cv2.imdecode(np.frombuffer(own.content, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
    assert g.sum() > o.sum() and (g & o).sum() >= 0.98 * o.sum()      # the object + the hand, one piece
