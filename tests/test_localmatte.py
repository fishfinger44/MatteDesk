# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 1 of docs/PLAN_HAIR_TOOL.md (section 10, synthetic): local matting of a zone.

"Hair" = thin lines in the foreground colour on a motley background + a foreign (gold) line crossing the zone; the core
has no hair. Zone + a "-" stroke on the line -> alpha on the hair > 0.6, on the line < 0.1, nothing outside the zone;
`chroma` on a uniform background; a forced engine returns exactly its own alpha; holdout -> 0 under the front object.
"""

import numpy as np
import pytest

from engine.localmatte import (DEFAULTS, LocalMatteResult, build_trimap, chroma_alpha, connect_strands,
                               local_matte, matting_alpha)

FG = np.array([0.18, 0.28, 0.42], np.float32)      # brown (BGR)
GOLD = np.array([0.35, 0.72, 0.85], np.float32)
NAVY = np.array([0.30, 0.12, 0.08], np.float32)


def _motley(h, w, seed=1):
    rng = np.random.default_rng(seed)
    small = rng.uniform(0.2, 0.9, (h // 32 + 2, w // 32 + 2, 3)).astype(np.float32)
    import cv2
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big[..., 2] *= 0.55          # keep it bluish / greenish: never the warm hair colour
    big[..., 0] = np.clip(big[..., 0] + 0.2, 0, 1)
    return np.clip(big, 0, 1)


def _scene(bg: np.ndarray, with_line=True, seed=2):
    """Disc = core (no hair); hair = 2 px lines radiating from the disc top; gold line across the zone."""
    import cv2
    h, w = bg.shape[:2]
    img = bg.copy()
    core = np.zeros((h, w), np.uint8)
    cx, cy, r = w // 2, int(h * 0.62), int(h * 0.26)
    cv2.circle(core, (cx, cy), r, 1, -1)
    hair = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(seed)
    for ang in np.linspace(-150, -30, 14):
        a = np.deg2rad(ang)
        L = r + int(rng.uniform(28, 56))
        p0 = (int(cx + (r - 2) * np.cos(a)), int(cy + (r - 2) * np.sin(a)))
        p1 = (int(cx + L * np.cos(a)), int(cy + L * np.sin(a)))
        cv2.line(hair, p0, p1, 1, 2, cv2.LINE_8)
    line = np.zeros((h, w), np.uint8)
    if with_line:
        cv2.line(line, (0, cy - r - 30), (w - 1, cy - r - 70), 1, 3, cv2.LINE_8)
    img[core > 0] = FG
    img[hair > 0] = FG
    img[(line > 0) & (core == 0)] = GOLD
    # zone: the ring above the disc (where the hair is) – includes the line
    region = np.zeros((h, w), np.uint8)
    cv2.ellipse(region, (cx, cy), (r + 70, r + 70), 0, -170, -10, 1, -1)
    region[cy:] = 0
    return img, core.astype(bool), hair.astype(bool) & ~core.astype(bool), line.astype(bool) & ~core.astype(bool), \
        region.astype(bool)


@pytest.mark.parametrize("engine", ["knn", "hybrid", "cf"])
def test_hair_on_motley_background_with_foreign_line(engine):
    img, core, hair, line, region = _scene(_motley(256, 320))
    bg_stroke = line & region
    res = local_matte(img, core, region, bg_mask=bg_stroke, engine=engine)
    assert isinstance(res, LocalMatteResult)
    a = res.alpha
    hz, lz = hair & region, line & region & ~hair
    assert a[hz].mean() > 0.6, (engine, a[hz].mean())
    assert a[lz].mean() < 0.1, (engine, a[lz].mean())
    assert a[core & region].min() == 1.0
    # nothing outside the zone; the area is the zone
    assert a[~region].max() == 0.0
    assert np.array_equal(res.area, region)
    # background of the zone (no hair, no line) stays mostly transparent
    bgz = region & ~hair & ~line & ~core
    assert a[bgz].mean() < 0.15, (engine, a[bgz].mean())
    x0, y0, x1, y1 = res.box
    tri = res.trimap[y0:y1, x0:x1]
    assert tri[core[y0:y1, x0:x1]].min() in (128, 255) and tri[(region & ~core)[y0:y1, x0:x1]].max() in (0, 128)
    if engine == "hybrid":
        assert "chroma_skipped" in res.stats   # motley background -> knn alone


def test_chroma_on_uniform_background():
    bg = np.tile(NAVY, (256, 320, 1)).astype(np.float32)
    rng = np.random.default_rng(5)
    bg = np.clip(bg + rng.normal(0, 0.012, bg.shape).astype(np.float32), 0, 1)
    img, core, hair, line, region = _scene(bg, with_line=True)
    res = local_matte(img, core, region, bg_mask=line & region, engine="chroma")
    a = res.alpha
    assert a[hair & region].mean() > 0.6
    assert a[line & region & ~hair].mean() < 0.1
    assert a[region & ~hair & ~line & ~core].mean() < 0.05
    st = res.stats
    assert st["bg_source"].startswith("pierścień BG") and st["hi"] > st["lo"]
    # the measured background colour is the navy wall
    lab_navy = [round(float(v), 2) for v in __import__("engine.localmatte", fromlist=["lab_of"]).lab_of(NAVY.reshape(1, 1, 3))[0, 0]]
    assert abs(st["bg_lab"][0] - lab_navy[0]) < 3 and abs(st["bg_lab"][1] - lab_navy[1]) < 3
    # the gold line is NOT marked "-": chroma alone keys it in (foreign colour != background) – that is what the
    # "-" stroke / distractor masks are for
    res2 = local_matte(img, core, region, engine="chroma")
    assert res2.alpha[line & region & ~hair].mean() > 0.5


def test_forced_engine_is_bitwise_its_own_result():
    img, core, hair, line, region = _scene(_motley(192, 256))
    res = local_matte(img, core, region, bg_mask=line & region, engine="knn", connect=False)
    # recompute by hand on the same crop and trimap
    x0, y0, x1, y1 = res.box
    assert res.scale == 1.0
    tm = build_trimap(core[y0:y1, x0:x1], region[y0:y1, x0:x1], (line & region)[y0:y1, x0:x1], DEFAULTS["fg_erode_px"])
    a = matting_alpha(img[y0:y1, x0:x1], tm, "knn")
    a[tm.fg] = 1.0
    a[~tm.unknown & ~tm.fg] = 0.0
    a[~region[y0:y1, x0:x1]] = 0.0
    a[region[y0:y1, x0:x1] & core[y0:y1, x0:x1]] = 1.0
    assert np.array_equal(res.alpha[y0:y1, x0:x1], a)
    assert np.array_equal(res.engines["knn"][y0:y1, x0:x1] * (tm.unknown | tm.fg), matting_alpha(img[y0:y1, x0:x1], tm, "knn") * (tm.unknown | tm.fg))


def test_holdout_and_strokes_force_zero_but_not_the_core():
    img, core, hair, line, region = _scene(_motley(192, 256))
    hold = np.zeros_like(region)
    hold[:, :120] = True           # a front object covering the left half
    res = local_matte(img, core, region, holdout=hold, engine="chroma")
    assert res.alpha[hold & region & ~core].max() == 0.0
    assert res.alpha[hold & region & core].min() == 1.0      # forced BG never eats the core
    assert res.trimap[hold & region & ~core].max() == 0


def test_downscale_and_connectivity():
    img, core, hair, line, region = _scene(_motley(320, 400))
    big = __import__("cv2").resize(img, (1600, 1280), interpolation=__import__("cv2").INTER_LINEAR)
    import cv2
    up = lambda m: cv2.resize(m.astype(np.float32), (1600, 1280), interpolation=cv2.INTER_NEAREST) > 0.5
    res = local_matte(big, up(core), up(region), bg_mask=up(line & region), engine="chroma",
                      params={"max_side": 512})
    assert res.scale < 1.0 and res.alpha.shape == (1280, 1600)
    # thin structures lose their edges to the resample (INTER_AREA down / INTER_LINEAR up): judge the line interior
    interior = cv2.erode(up(hair).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert res.alpha[interior & up(region)].mean() > 0.6
    assert res.alpha[~up(region)].max() == 0.0
    # connectivity: an island of alpha far from the core is dropped, a strand touching the core stays
    a = np.zeros((64, 64), np.float32)
    c = np.zeros((64, 64), bool)
    c[20:44, 20:44] = True
    a[c] = 1.0
    a[44:50, 30] = 0.6         # strand touching the core
    a[2:5, 2:5] = 0.9          # island (> 2 px away, not connected)
    out = connect_strands(a, c, 0.02, 200)
    assert np.isclose(out[44:50, 30].min(), 0.6) and out[2:5, 2:5].max() == 0.0


def test_empty_region_and_bad_engine():
    img = np.zeros((32, 32, 3), np.float32)
    core = np.zeros((32, 32), bool)
    res = local_matte(img, core, np.zeros((32, 32), bool), engine="knn")
    assert res.stats.get("empty") and res.alpha.max() == 0.0
    with pytest.raises(ValueError):
        local_matte(img, core, np.ones((32, 32), bool), engine="vitmatte")
