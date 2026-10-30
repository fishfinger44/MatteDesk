# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„Wcięcie: cały rdzeń” (user decision 2026-10-29): the band covers the whole core – the model replaces the core
everywhere; the inner band edge is the middle of the core (its ridge of depth) and „Wtopienie wcięcia” keeps that much core around it
(0 px = nothing of the core is left). One switch of the band (engine/bandsys `whole`, engine/objband `whole_core`) for
stages 3 / 4 / 5 (whole-outline zones) / 7; off = the band exactly as before."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import bandsys  # noqa: E402
from engine.segment import sdf  # noqa: E402

INF = float("inf")


def _blob():
    c = np.zeros((90, 120), bool)
    c[15:75, 20:100] = True
    c[40:50, 100:115] = True       # a thin strand: 10 px wide
    return c


# ----------------------------------------------------------------------------- geometry (engine/bandsys)
def test_whole_core_band_replaces_the_core_and_the_inset_feather_keeps_its_middle():
    c = _blob()
    cf = c.astype(np.float32)
    model = np.full(c.shape, 0.3, np.float32)
    d = sdf(cf)
    a0 = bandsys.compose(cf, model, d, 8, INF, 0.0, 4.0)
    assert np.all(a0[c] == np.float32(0.3))                   # feather 0: the model decides in the whole core
    a6 = bandsys.compose(cf, model, d, 8, INF, 6.0, 4.0)
    mid = bandsys.core_middle(c, 6.0)
    assert mid[45, 60] and not mid[15, 60] and np.all(a6[mid] == 1.0)    # the middle stays core
    assert a6[45, 25] == np.float32(0.3) and a6[16, 60] == np.float32(0.3)   # near the edge: the model
    a20 = bandsys.compose(cf, model, d, 8, INF, 12.0, 4.0)
    assert a20[c].sum() > a6[c].sum() > a0[c].sum()          # a longer inset feather keeps more core
    tip = (slice(40, 50), slice(108, 115))                     # the tip of a 10 px strand: thinner than the feather
    assert a6[tip].max() < a6[mid].min() and a6[tip].min() == np.float32(0.3)     # less core there, its edge = model
    edge = c & (d <= 1.5)
    assert np.all(a6[edge] == np.float32(0.3))                 # the kept core never touches the edge
    # outside the core, beyond the reach of the old inset ramp (2 px inside + 6 px), the outer band is as before
    a_fin = bandsys.compose(cf, model, d, 8, 2.0, 6.0, 4.0)
    far = (d < -5) & ~c
    assert far.any() and np.array_equal(a6[far], a_fin[far])


def test_whole_core_band_points_shape_only_the_outside():
    c = _blob()
    pts = [(100.0, 45.0, 20.0, 0.0)]          # a point with no inside width: ignored inside with „cały rdzeń”
    o, i = bandsys.band_widths(c, 8.0, INF, pts)
    assert np.isinf(i) and isinstance(o, np.ndarray)
    assert bandsys.band_area(sdf(c.astype(np.float32)), o, i)[c].all()


def test_whole_core_area_paths_agree_and_the_crop_is_bit_identical():
    c = _blob()
    rng = np.random.default_rng(3)
    model = rng.random(c.shape).astype(np.float32)
    ring = bandsys.band_area(sdf(c.astype(np.float32)), 8.0, 2.0)
    full = bandsys.compose_area(c, model, ring, 6.0, 4.0, whole=True)
    crop = bandsys.compose_area_crop(c, model, ring, 6.0, 4.0, whole=True)
    assert np.array_equal(full, crop)
    w = bandsys.area_weight(ring, c, 6.0, 4.0, whole=True)
    assert np.allclose(full, w * model + (1 - w) * c.astype(np.float32), atol=1e-6)
    lines = bandsys.area_lines(ring, c, 6.0, 4.0, whole=True)
    assert lines["band_contours"] and lines["inset_feather_contours"]
    assert not bandsys.area_lines(ring, c, 0.0, 4.0, whole=True)["inset_feather_contours"]   # 0 px: no inset line


def test_off_is_the_band_as_before():
    c = _blob()
    model = np.full(c.shape, 0.3, np.float32)
    ring = bandsys.band_area(sdf(c.astype(np.float32)), 8.0, 2.0)
    assert np.array_equal(bandsys.compose_area(c, model, ring, 2.0, 8.0),
                          bandsys.compose_area(c, model, ring, 2.0, 8.0, whole=False))
    assert bandsys.band_widths(c, 8.0, 2.0) == (8.0, 2.0)


def test_trimap_of_a_band_over_the_whole_core_has_its_middle_as_sure_fg():
    from engine.localmatte import build_trimap
    from engine.zones import outline_band
    c = _blob()
    reg = outline_band(c, 8, 2, whole=True)
    assert reg[c].all()
    tm = build_trimap(c, reg, np.zeros_like(c), 2, band_unknown=True)
    assert tm.fg.any() and np.array_equal(tm.fg, bandsys.core_middle(c))
    assert not build_trimap(c, outline_band(c, 8, 2), np.zeros_like(c), 2, band_unknown=True).fg.sum() == 0


# ----------------------------------------------------------------------------- the setting (engine/objband, API)
ZONE = {"mode": "add", "criterion": "matte", "polygon": [], "matting": "vitmatte", "preset": "hair",
        "params": {"region": "outline", "temporal": "none"}}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def _bands(pipe, zid):
    from engine.edge import edge_settings
    from engine.ltx_alpha import object_settings as ltx_s
    from engine.videomama import object_settings as vmm_s
    from engine.zones import load_zones, zone_whole
    obj = pipe.store.get("obj")
    z = next(z for z in load_zones(pipe.store.dir("obj")) if z.zone_id == zid)
    return {"matanyone": edge_settings(obj), "videomama": vmm_s(obj), "ltx": ltx_s(obj)}, zone_whole(z)


def test_switch_on_the_shared_band_reaches_every_stage_and_their_signatures(proj):
    from engine.ltx_alpha import settings_sig as ltx_sig
    from engine.stages import settings_sig
    from engine.videomama import settings_sig as vmm_sig
    app_module, api, shot, pipe, frames, start = proj
    zid = api.post("/api/object/obj/zones/new", json={**ZONE, "frame": start}).json()["zone"]["zone_id"]
    st, zw = _bands(pipe, zid)
    assert not zw and not any(s["band"].whole_core for s in st.values())
    sig0 = (settings_sig(pipe.store.get("obj")), vmm_sig(st["videomama"]), ltx_sig(st["ltx"]))
    assert "whole_core" not in st["videomama"] and "band_whole_core" not in st["matanyone"]   # off: no new key
    r = api.put("/api/object/obj/band_whole", json={"stage": "videomama", "on": True})
    assert r.status_code == 200 and r.json()["mode"] == "shared"
    st, zw = _bands(pipe, zid)
    assert zw and all(s["band"].whole_core for s in st.values())
    assert all(np.isinf(s["band"].in_w) for s in st.values()) and st["ltx"]["core_erode_px"] == 2   # the slider stays
    sig1 = (settings_sig(pipe.store.get("obj")), vmm_sig(st["videomama"]), ltx_sig(st["ltx"]))
    assert all(a != b for a, b in zip(sig0, sig1))                    # ↻ in every stage on the shared band
    owners = api.get("/api/object/obj/band").json()
    assert owners["whole_core"] and all(o["whole_core"] for o in owners["owners"])
    # detach stage 7: it keeps its copy; switched off on the shared band, the detached one stays on
    assert api.put("/api/object/obj/stage_band/ltx/mode", json={"mode": "own"}).status_code == 200
    assert api.put("/api/object/obj/band_whole", json={"stage": "matanyone", "on": False}).json()["mode"] == "shared"
    st, zw = _bands(pipe, zid)
    assert not zw and st["ltx"]["band"].whole_core and not st["videomama"]["band"].whole_core
    sig2 = (settings_sig(pipe.store.get("obj")), vmm_sig(st["videomama"]))
    assert sig2 == sig0[:2]                                           # off again = the signatures as before
    assert api.put("/api/object/obj/band_whole", json={"stage": "bogus", "on": True}).status_code == 400


def test_every_band_composition_reads_the_inset_through_the_switch():
    """The stage-4 frame preview composed with the raw inset slider (user report 2026-10-29: „cały rdzeń” on, the
    preview kept the core) – every VideoMaMa / LTX-2.5 band geometry reads the inset through inset_w."""
    import re
    root = Path(__file__).resolve().parent.parent
    bad = []
    for p in list((root / "engine").glob("*.py")) + [root / "app.py"]:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'(compose|band_widths)\(.*s\["(inset_px|core_erode_px)"\]', line):
                bad.append(f"{p.name}:{i}")
    assert not bad, bad


def test_whole_core_points_shape_the_inset_feather_and_the_outer_feather_stays_outside():
    """„Cały rdzeń” + band points (user decision 2026-10-29): a point's inner whisker over the inset slider scales the
    inset feather there (0 = the model alone), interpolated along the outline; the outer feather acts on the outer
    band only (an outer band of 0 px keeps no core at the edge, whatever the outer feather)."""
    from engine.videomama import compose as vmm_compose
    c = np.zeros((90, 160), bool)
    c[20:70, 20:140] = True
    cf = c.astype(np.float32)
    model = np.full(c.shape, 0.3, np.float32)
    unit = 2.0                                                   # the inset slider: the unit of the whiskers
    pts = [(20.0, 45.0, 0.0, 0.0), (139.0, 45.0, 0.0, 2 * unit)]   # left: no core kept, right: twice the feather
    fmap = bandsys.whole_feather(cf, 10.0, pts, unit)
    assert fmap[45, 22] == 0.0 and abs(fmap[45, 137] - 20.0) < 1e-4
    a = vmm_compose(cf, model, 0.0, INF, 10.0, 25.0, pts, in_unit=unit)
    flat = vmm_compose(cf, model, 0.0, INF, 10.0, 25.0, None, in_unit=unit)
    assert a[45, 40] == np.float32(0.3) and flat[45, 40] > 0.3     # left: the model alone near the edge
    assert a[32, 115] > flat[32, 115]                              # right: more core kept (off the middle row)
    edge = c & (sdf(cf) <= 1.5)
    assert np.all(flat[edge] == np.float32(0.3))                   # out band 0: the outer feather keeps no core
    # off: the inset slider and the points as before, in_unit changes nothing
    assert np.array_equal(vmm_compose(cf, model, 8.0, 2.0, 2.0, 8.0, pts, in_unit=unit),
                          vmm_compose(cf, model, 8.0, 2.0, 2.0, 8.0, pts))
