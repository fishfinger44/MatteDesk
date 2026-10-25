# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""One band path (user request 2026-10-24): engine/bandsys.py is the only implementation of the band geometry, the
feather rule, the composition and the preview lines of stages 3 MatAnyone2, 4 VideoMaMa, 5 ViTMatte (whole-outline
zones) and 7 LTX-2.5. Bit-identity with the per-stage code it replaced (verbatim copies of the former formulas
below), and the two deliberate deviations spelled out."""

import cv2
import numpy as np
import pytest

from engine import bandsys as B
from engine.segment import sdf


# ----------------------------------------------------------------------------- scenes
def _blob(seed, h=160, w=220):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        c = (int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40)))
        ax = (int(rng.integers(14, 50)), int(rng.integers(10, 40)))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return m.astype(bool)


def _wmap(seed, shape):
    rng = np.random.default_rng(seed + 100)
    w = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 9)
    w = (w - w.min()) / max(1e-6, w.max() - w.min())
    return (3.0 + 21.0 * w).astype(np.float32)          # the adaptive band's 3..24 px


def _ring(core, out_px, in_px):
    """A constant band around `core` – an arbitrary area for the area path."""
    d = sdf(core)
    return (d >= -out_px) & (d <= in_px)


# ----------------------------------------------------------------------------- the former code (replaced 2026-10-24)
def ref_feather_maps(area, core, inner_px, outer_px):   # engine/zones.py feather_maps
    a = area.astype(bool)
    c = core.astype(bool)
    d_in = cv2.distanceTransform((~(c & ~a)).astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~(~c & ~a)).astype(np.uint8), cv2.DIST_L2, 5)
    width = d_in + d_out
    li = np.minimum(max(0.5, float(inner_px)), width)
    lo = np.minimum(max(0.5, float(outer_px)), np.maximum(0.0, width - li))
    return d_in, d_out, np.maximum(0.5, li), np.maximum(0.5, lo)


def ref_feather_weight(area, core, inner_px, outer_px):   # engine/zones.py feather_weight
    d_in, d_out, li, lo = ref_feather_maps(area, core, inner_px, outer_px)
    w = np.minimum(np.clip(d_in / li, 0, 1), np.clip(d_out / lo, 0, 1))
    return np.where(area.astype(bool), w, 0.0).astype(np.float32)


def ref_adaptive_band(core, width_map):   # engine/edge.py adaptive_band
    return np.abs(sdf(core)) <= width_map


def ref_band_px(core, width_map, out_px, in_px):   # engine/edge.py band_px
    d = sdf(core)
    return (d <= np.maximum(width_map, in_px)) & (d >= -np.maximum(width_map, out_px))


def ref_variable_band(core, out_map, in_map):   # engine/bandmap.py variable_band
    c = core.astype(bool)
    u8 = c.astype(np.uint8)
    d_out = cv2.distanceTransform((1 - u8), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(u8, cv2.DIST_L2, 5)
    return (~c & (d_out <= out_map)) | (c & (d_in <= in_map))


def ref_stage3_compose(c, soft, band, fe, fo):   # engine/refine.py, the feather branches of the edge band
    if fo > 0:
        d_in, d_out, li, lo = ref_feather_maps(band, c, fe, fo)
        w_in = np.clip(d_in / li, 0.0, 1.0) if fe > 0 else 1.0
        w = np.minimum(w_in, np.clip(d_out / lo, 0.0, 1.0))
        return np.where(band, w * soft + (1.0 - w) * c.astype(np.float32), c.astype(np.float32))
    if fe > 0:
        deep = c & ~band
        w = np.clip(cv2.distanceTransform((~deep).astype(np.uint8), cv2.DIST_L2, 5) / fe, 0.0, 1.0)
        return np.where(band, w * soft + (1.0 - w) * c.astype(np.float32), c.astype(np.float32))
    return np.where(band, soft, c.astype(np.float32))


def _ref_lines(m, mode):
    if m is None or not m.any():
        return []
    cnts, _ = cv2.findContours(m.astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in cnts if len(c) >= 3]


def ref_stage3_lines(band, core, fi, fo):   # app.py stage_band_lines, stage matanyone
    d_in, d_out, li, lo = ref_feather_maps(band, core, fi, fo)
    solid = core & ~band
    return {"band_contours": _ref_lines(band, cv2.RETR_CCOMP),
            "inset_feather_contours": _ref_lines(solid | (band & (d_in < li)), cv2.RETR_EXTERNAL) if fi > 0 else [],
            "feather_contours": _ref_lines(solid | (band & (d_out >= lo)), cv2.RETR_EXTERNAL) if fo > 0 else []}


def ref_zone_lines(reg, core, fi, fo):   # engine/pipeline.py zone_region (outline mode; fo = 0 -> fi at the call)
    d_in, d_out, li, lo = ref_feather_maps(reg, core, fi, fo if fo > 0 else fi)
    c0 = core.astype(bool) & ~reg
    return {"band_contours": _ref_lines(reg, cv2.RETR_CCOMP),
            "inset_feather_contours": _ref_lines(c0 | (reg & (d_in < li)), cv2.RETR_EXTERNAL),
            "feather_contours": _ref_lines(c0 | (reg & (d_out >= lo)), cv2.RETR_EXTERNAL)}


def ref_zone_compose(area, sa, core, fi, fo, partner):   # engine/zones.py apply_zones, the whole-outline branch
    d_in, d_out, li, lo = ref_feather_maps(area, core, fi, fo if fo > 0 else fi)
    zw = np.minimum(np.clip(d_in / li, 0, 1), np.clip(d_out / lo, 0, 1))
    return np.where(area, zw * sa + (1.0 - zw) * partner, sa).astype(np.float32)


def ref_band_weight(d, out_w, in_w, inner_px, outer_px):   # engine/bandsys.py band_weight before 2026-10-24
    d = d.astype(np.float32)
    o = np.asarray(out_w, np.float32)
    i = np.asarray(in_w, np.float32)
    area = (d >= -o) & (d <= i)
    d_in = i - d
    d_out = d + o
    width = o + i
    fi, fo = max(0.0, float(inner_px)), max(0.0, float(outer_px))
    li = np.minimum(fi, width)
    lo = np.minimum(fo, np.maximum(0.0, width - li))
    w_in = np.where(li > 0, np.clip(d_in / np.maximum(li, 1e-6), 0.0, 1.0), 1.0)
    w_out = np.where(lo > 0, np.clip(d_out / np.maximum(lo, 1e-6), 0.0, 1.0), 1.0)
    return area, np.where(area, np.minimum(w_in, w_out), 0.0).astype(np.float32)


def ref_compose(core, model, d, out_w, in_w, inner_px, outer_px):
    c = np.clip(core.astype(np.float32), 0, 1)
    m = np.clip(model.astype(np.float32), 0, 1)
    area, w = ref_band_weight(d, out_w, in_w, inner_px, outer_px)
    return np.where(area, w * m + (1.0 - w) * c, c).astype(np.float32)


def ref_feather_lines(d, out_w, in_w, inner_px, outer_px):
    d = d.astype(np.float32)
    o = np.asarray(out_w, np.float32)
    i = np.asarray(in_w, np.float32)
    band = (d >= -o) & (d <= i)
    width = o + i
    li = np.minimum(max(0.0, float(inner_px)), width)
    lo = np.minimum(max(0.0, float(outer_px)), np.maximum(0.0, width - li))
    inner_end = (d > i) | (band & (i - d < li)) if float(inner_px) > 0 else None
    outer_end = (d > i) | (band & (d + o >= lo)) if float(outer_px) > 0 else None
    return band, inner_end, outer_end


SEEDS = [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------- one band geometry
@pytest.mark.parametrize("seed", SEEDS)
def test_band_area_is_every_former_band_mask(seed):
    """The adaptive stage-3 band, its px mode and the band of points (bandmap) are band_area of width maps."""
    from engine import bandmap, edge
    core = _blob(seed)
    wmap = _wmap(seed, core.shape)
    d = sdf(core)
    assert np.array_equal(B.band_area(d, wmap, wmap), ref_adaptive_band(core, wmap))
    assert np.array_equal(edge.adaptive_band(core, wmap), ref_adaptive_band(core, wmap))
    for o, i in ((20.0, 4.0), (0.0, 0.0), (5.0, 30.0)):
        assert np.array_equal(edge.band_px(core, wmap, o, i), ref_band_px(core, wmap, o, i))
        assert np.array_equal(B.band_area(d, np.maximum(wmap, o), np.maximum(wmap, i)), ref_band_px(core, wmap, o, i))
    o_map, i_map = _wmap(seed + 7, core.shape), _wmap(seed + 9, core.shape) * 0.3
    assert np.array_equal(bandmap.variable_band(core, o_map, i_map), ref_variable_band(core, o_map, i_map))
    assert np.array_equal(bandmap.variable_band(core, np.zeros_like(o_map), np.zeros_like(i_map)),
                          ref_variable_band(core, np.zeros_like(o_map), np.zeros_like(i_map)))
    assert not bandmap.variable_band(np.zeros_like(core), o_map, i_map).any()


# ----------------------------------------------------------------------------- one feather rule (area path)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fi,fo", [(0.5, 0.0), (4.0, 0.0), (11.0, 51.0), (4.0, 20.0), (6.0, 10.0), (0.7, 0.3)])
def test_area_weight_is_the_former_zone_weight(seed, fi, fo):
    core = _blob(seed)
    for area in (_ring(core, 17, 0), _ring(core, 30, 12), B.band_area(sdf(core), _wmap(seed, core.shape), 4.0)):
        w = B.area_weight(area, core, fi, fo)
        assert w.dtype == np.float32 and np.array_equal(w, ref_feather_weight(area, core, fi, fo))
        d_in, d_out, solid = B.area_dists(area, core)
        r_in, r_out, r_li, r_lo = ref_feather_maps(area, core, fi, fo)
        li, lo = B.lengths(d_in + d_out, fi, fo)
        assert np.array_equal(d_in, r_in) and np.array_equal(d_out, r_out) and np.array_equal(solid, core & ~area)
        # the former lengths had a 0.5 px floor (a division guard); the first band pixel sits at distance 1, so the
        # weights are the same – the lengths agree above that floor
        assert np.array_equal(np.maximum(li, 0.5)[area], r_li[area]) and np.array_equal(np.maximum(lo, 0.5)[area], r_lo[area])


# ----------------------------------------------------------------------------- stage 3 MatAnyone2
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fe,fo", [(0.0, 0.0), (2.0, 5.0), (4.0, 10.0), (0.5, 3.0), (3.0, 0.0)])
def test_stage3_composition_and_lines_are_the_former_ones(seed, fe, fo):
    """refine.py (the edge result through the band with its feathers) and the stage-3 lines of app.py."""
    rng = np.random.default_rng(seed)
    core = _blob(seed)
    soft = rng.random(core.shape).astype(np.float32)
    band = B.band_area(sdf(core), _wmap(seed, core.shape), _wmap(seed, core.shape))   # the adaptive band (>= 6 px wide)
    band &= ~_blob(seed + 50)                                                           # less an exclusion
    new = B.compose_area(core, soft, band, fe, fo) if (fe > 0 or fo > 0) else np.where(band, soft, core.astype(np.float32))
    old = ref_stage3_compose(core, soft, band, fe, fo)
    assert new.dtype == np.float32
    if fo == 0 and fe > 0:
        # the former inset-only branch never clamped its ramp to the band width (the one rule does): identical where
        # the band is at least `fe` wide – elsewhere test_deviation_inset_feather_clamped_to_the_band_without_an_outer_feather
        d_in, d_out, _ = B.area_dists(band, core)
        same = ~band | (d_in + d_out >= fe)
        assert np.array_equal(new[same], old[same]) and same.sum() > 0.9 * same.size
    else:
        assert np.array_equal(new, old)
    assert B.area_lines(band, core, fe, fo) == ref_stage3_lines(band, core, fe, fo)


# ----------------------------------------------------------------------------- stage 5 whole-outline zone
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fi,fo", [(6.0, 10.0), (4.0, 0.0), (11.0, 51.0), (0.5, 15.0)])
def test_zone_composition_and_lines_are_the_former_ones(seed, fi, fo):
    rng = np.random.default_rng(seed + 3)
    core = _blob(seed)
    area = _ring(core, 17, 4) & ~_blob(seed + 60)
    sa = rng.random(core.shape).astype(np.float32)
    base = np.clip(rng.random(core.shape).astype(np.float32) * 0.5 + core, 0, 1)
    partner = np.maximum(core.astype(np.float32), base)
    new = B.compose_area(core, sa, area, fi, fo if fo > 0 else fi, outside=sa, partner=partner)
    assert np.array_equal(new, ref_zone_compose(area, sa, core, fi, fo, partner))
    assert B.area_lines(area, core, fi, fo if fo > 0 else fi) == ref_zone_lines(area, core, fi, fo)


# ----------------------------------------------------------------------------- stages 4 / 7 = the area path (2026-10-24)
@pytest.mark.parametrize("seed", SEEDS)
def test_width_band_is_the_area_band(seed):
    """One measure (user decision 2026-10-24): the stages that size the band from widths (4 VideoMaMa, 7 LTX-2.5) give
    exactly what stage 3 / the zones give for the same band area; and they stay within ~1 px of the former analytic
    ramps (the reference copies above)."""
    rng = np.random.default_rng(seed + 9)
    core = _blob(seed)
    d = sdf(core)
    model = rng.random(core.shape).astype(np.float32)
    o_map, i_map = _wmap(seed + 7, core.shape), _wmap(seed + 9, core.shape) * 0.3
    for o, i in ((30.0, 8.0), (12.0, 6.0), (0.0, 0.0), (o_map, i_map)):
        area = B.band_area(d, o, i)
        for fi, fo in ((0.0, 0.0), (5.0, 0.0), (3.0, 6.0), (4.0, 10.0), (40.0, 40.0)):
            a1, w1 = B.band_weight(d, o, i, fi, fo)
            assert np.array_equal(a1, area) and np.array_equal(w1, B.area_weight(area, core, fi, fo))
            assert np.array_equal(B.compose(core, model, d, o, i, fi, fo), B.compose_area(core, model, area, fi, fo))
            band, inner, outer = B.feather_lines(d, o, i, fi, fo)
            assert B.preview_lines(band, inner, outer) == B.area_lines(area, core, fi, fo)
            _, w0 = ref_band_weight(d, o, i, fi, fo)
            if fi == 0 and fo == 0:
                assert np.array_equal(w0, w1)                   # no feathers: the same, bit for bit
            elif max(fi, fo) < 40 and area.any():
                # ~1–2 px of ramp; single pixels in concave places / where the bands of close parts merge differ more –
                # the area measures to the real band edge there (as stage 3 and the zones always did): small on average
                assert np.abs(w0 - w1)[area].mean() < 0.1


def test_preview_lines_dict_shape():
    core = _blob(1)
    band = _ring(core, 10, 4)
    L = B.area_lines(band, core, 0.0, 0.0)
    assert set(L) == {"band_contours", "inset_feather_contours", "feather_contours"}
    assert L["band_contours"] and L["inset_feather_contours"] == [] and L["feather_contours"] == []
    assert B.preview_lines(np.zeros_like(core), None, None) == {"band_contours": [], "inset_feather_contours": [], "feather_contours": []}


# ----------------------------------------------------------------------------- the two deliberate deviations (2026-10-24)
def test_deviation_outer_feather_spans_the_whole_band_without_an_inset_feather():
    """Stage 3 with inset feather 0 and an outer feather wider than the band: the former feather_maps reserved 0.5 px
    for the (absent) inset feather, so the outer ramp ran over width − 0.5 px; the one rule gives it the whole band
    (as stages 4 / 7 always did). Anything else (inset >= 0.5 px, or an outer feather that fits) is bit-identical."""
    core = _blob(2)
    band = _ring(core, 3, 3)                                     # ~6 px wide
    d_in, d_out, _ = B.area_dists(band, core)
    w_new = B.area_weight(band, core, 0.0, 10.0)
    w_old = ref_feather_weight(band, core, 0.0, 10.0)
    width = d_in + d_out
    assert np.array_equal(w_new[band], np.clip(d_out / np.minimum(width, 10.0), 0, 1)[band])          # the ramp over the full band
    assert np.allclose(w_old[band], np.clip(d_out / np.minimum(width - 0.5, 10.0), 0, 1)[band])    # the former 0.5 px reserve
    # a longer ramp (the full band) = a little less model weight mid-band, 1 at the inner edge as before
    assert (w_new <= w_old + 1e-6).all() and (w_new < w_old - 1e-3).any() and np.abs(w_new - w_old).max() < 0.1
    assert np.array_equal(B.area_weight(band, core, 0.5, 10.0), ref_feather_weight(band, core, 0.5, 10.0))
    assert np.array_equal(B.area_weight(band, core, 0.0, 3.0), ref_feather_weight(band, core, 0.0, 3.0))


def test_deviation_inset_feather_clamped_to_the_band_without_an_outer_feather():
    """Stage 3 with an inset feather wider than the band and no outer feather: the former ramp d_in / fe stayed far
    from 1 inside a narrow band (the edge result was mostly core up to the outer edge); the one rule clamps the inset
    feather to the band width (user rule 2026-10-22: the inset feather keeps its length UP TO the band width), so the
    ramp spans the band – the model's weight at the outer band edge is (width − 1) / width instead of (width − 1) / fe.
    A band at least `fe` wide is bit-identical."""
    rng = np.random.default_rng(4)
    core = _blob(4)
    soft = rng.random(core.shape).astype(np.float32)
    band = _ring(core, 2, 2)                                     # ~4 px wide, feather 12 px
    new = B.compose_area(core, soft, band, 12.0, 0.0)
    old = ref_stage3_compose(core, soft, band, 12.0, 0.0)
    d_in, d_out, _ = B.area_dists(band, core)
    w_new, w_old = B.area_weight(band, core, 12.0, 0.0), np.where(band, np.clip(d_in / 12.0, 0, 1), 0).astype(np.float32)
    assert (w_new >= w_old - 1e-6).all() and (w_new[band] > w_old[band] + 0.3).any()   # closer to the model everywhere
    outer_edge = band & (d_out < 1.5) & ~core
    assert w_new[outer_edge].min() > 0.6 and w_old[outer_edge].max() < 0.45           # ~(w − 1) / w vs (w − 1) / 12
    assert np.allclose(new, w_new * soft + (1 - w_new) * core, atol=1e-6) and not np.array_equal(new, old)
    wide = _ring(core, 20, 20)                                   # a band wider than the feather: bit-identical
    assert np.array_equal(B.compose_area(core, soft, wide, 12.0, 0.0), ref_stage3_compose(core, soft, wide, 12.0, 0.0))
