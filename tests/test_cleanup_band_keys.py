# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""C1 band slider keys (user decision 2026-11-15, like the band of the alpha stages): a change of „Pas na zewnątrz” /
„Wtopienie” = a key of the current frame, linear between keys, held outside them (engine/cleanup.band_at – the mask,
the lines, the recomposite, Cleanup 3D); 🔑✕ in Cleanup asks when the frame has an anchor AND a band key."""
from pathlib import Path

import numpy as np

from engine import cleanup as cu
from engine import inpaint_recompose as IR

ROOT = Path(__file__).resolve().parents[1]


def _disc(r=30, size=128):
    y, x = np.mgrid[:size, :size]
    return (((x - 64) ** 2 + (y - 64) ** 2) <= r * r).astype(np.float32)


def test_band_at_interpolates_and_holds():
    s = cu.settings({"grow_px": 8, "soft_px": 3})
    assert cu.band_at(s, 10) == (8.0, 3.0)                                  # no keys: the sliders
    s = cu.settings({"band_keys": {"10": {"grow_px": 0, "soft_px": 0}, "20": {"grow_px": -10, "soft_px": 20}}})
    assert cu.band_at(s, 1) == (0.0, 0.0) and cu.band_at(s, 30) == (-10.0, 20.0)
    assert cu.band_at(s, 15) == (-5.0, 10.0)
    assert cu.band_reach_max(s) == 10.0
    assert "band_keys" in cu.PER_TARGET and "band_keys" in IR.BAND_FIELDS


def test_clean_band_keys_drops_junk_and_clamps():
    k = cu.clean_band_keys({"5": {"grow_px": -500, "soft_px": 999}, "x": {"grow_px": 1, "soft_px": 1}, "7": {"grow_px": 1}})
    assert k == {"5": {"grow_px": -100.0, "soft_px": 100.0}}


def test_mask_follows_the_keys_per_frame():
    a = _disc(30)
    s = cu.settings({"holdout": False, "band_keys": {"1": {"grow_px": 0, "soft_px": 0}, "11": {"grow_px": 20, "soft_px": 0}}})
    fn = cu.mask_fn(s, [lambda f: a], [])
    w1, w6, w11 = (int((fn(f) > 0.5).sum()) for f in (1, 6, 11))
    assert w1 < w6 < w11
    assert (fn(6) == cu.removal_mask([a], 10, 0)).all()


def test_override_wins_for_the_frame():
    s = cu.settings({"band_keys": {"1": {"grow_px": 40, "soft_px": 9}}})
    o = cu.at_override(s, 5, None, 1)
    assert o["band_keys"] == {} and o["grow_px"] == 5.0 and o["soft_px"] == 9.0
    assert cu.at_override(s, None, None, 1) is s
    r = IR.recompose_settings(s, {}, {"grow_px": 3, "soft_px": 1})
    assert r["band_keys"] == {} and r["grow_px"] == 3.0


def test_endpoints(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.inpaint_band as AB
    work = tmp_path / "w"; work.mkdir()
    router, _ = (lambda r: (r if isinstance(r, tuple) else (r, None)))(
        AB.make(STATE={}, plate_size=lambda: (64, 64), cleanup={"need": lambda: (work, {}), "alpha_loaders": None}))
    app = FastAPI(); app.include_router(router); c = TestClient(app)
    r = c.put("/api/cleanup/band_keys/10", json={"grow_px": -5}).json()
    assert r["band_keys"] == {"10": {"grow_px": -5.0, "soft_px": 3.0}}       # soft = the keyed (here slider) value
    r = c.put("/api/cleanup/band_keys/20", json={"soft_px": 12}).json()
    assert r["band_keys"]["20"] == {"grow_px": -5.0, "soft_px": 12.0}
    c.delete("/api/cleanup/band_keys/10")
    r = c.delete("/api/cleanup/band_keys/20").json()                          # the last one: its values stay
    assert r["band_keys"] == {} and r["grow_px"] == -5.0 and r["soft_px"] == 12.0
    assert c.delete("/api/cleanup/band_keys/20").status_code == 404
    r = c.put("/api/cleanup/band_keys", json={"band_keys": {"3": {"grow_px": 1, "soft_px": 2}}}).json()
    assert r["band_keys"] == {"3": {"grow_px": 1.0, "soft_px": 2.0}}


def test_ui_keys_lane_and_delete_choice():
    js = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "bandKeySet(field, val);" in js and "save({ [field]: val });" not in js
    assert "/api/cleanup/band_keys/${f}" in js and "B.setSliderKeys('c1', BAND_KEY_STORE)" in js
    assert "B.setSliderKey(f, field, val)" in js and "B.keyedAt(bandKeys(), f," in js
    assert "timelineCuBandKeys" not in js and "MD_CU_BANDKEYS" not in js   # no lane / undo of its own: the band system's
    assert "syncBandSliders(); renderBandKeysLane();" in js
    bp = (ROOT / "static" / "bandpoints.js").read_text(encoding="utf-8")
    for x in ("function keyedAt(keys, f, base)", "async function setSliderKey(f, field, value)", "const hasSliderKey",
              "if (u.restore) { await u.restore();", "keyedAt, setSliderKeys, setSliderKey, hasSliderKey,"):
        assert x in bp, x
    obj = (ROOT / "static" / "objects.js").read_text(encoding="utf-8")
    assert "function cleanKeyKinds(f)" in obj and "if (cleanMode()) return deleteCleanKey(f);" in obj
    assert "B.stage() === 'c1' && B.hasKeyAt(f)" in obj and "B.deleteKeyAt(f, asked ? { confirm: false } : undefined)" in obj
    assert "co usunąć?" in obj and 'value="both"' in obj


def test_bandsys_keyed_at_is_the_one_arithmetic():
    from engine import bandsys
    k = bandsys.clean_slider_keys({"0": {"a": 0, "b": 5}, "10": {"a": 10, "b": 5}, "z": {}}, {"a": (0, 8), "b": (0, 9)})
    assert k == {"0": {"a": 0.0, "b": 5.0}, "10": {"a": 8.0, "b": 5.0}}
    assert bandsys.keyed_at(k, 5, {"a": 1, "b": 1}) == {"a": 4.0, "b": 5.0}
    assert bandsys.keyed_at({}, 5, {"a": 1, "b": 2}) == {"a": 1.0, "b": 2.0}
    src = (ROOT / "engine" / "cleanup.py").read_text(encoding="utf-8")
    ba = src[src.index("def band_at("):src.index("def band_reach_max(")]
    assert "bandsys.keyed_at(" in ba and "fs[i - 1]" not in ba


def test_grain_text_handles_per_channel_arrays():
    js = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    g = js[js.index("function renderGrain"):js.index("function syncH3")]
    assert "Array.isArray(x)" in g and "R / G / B" in g
