# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""An object outside a group picks its mask too (user 2026-11-15): the same choice as „Do grupy z:” (group_source),
kept when it leaves a group, brought along when it joins one, and the default of Cleanup (removed / protected)."""
from pathlib import Path

import numpy as np
import pytest

from engine import cleanup_masks as CM
from engine import groups as G
from engine.objects import ObjectStore


@pytest.fixture
def store(tmp_path):
    st = ObjectStore(tmp_path / "objects")
    for n in ("a", "b"):
        st.create(n) if hasattr(st, "create") else st.add(n)
        core = st.dir(n) / "core"; core.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(core / "c_0001.png"), np.zeros((4, 4), np.uint8))
    return st


def test_own_source_kept_and_used(store):
    assert not G.has_own_source(store.get("a"))
    G.set_object_source(store, "a", {"source": "core"})
    assert G.has_own_source(store.get("a"))
    r = CM.resolve(store, store.get("a"))
    assert r["by"] == "object" and r["source"] == "core"
    with pytest.raises(ValueError):
        G.set_object_source(store, "a", {"source": "nope"})


def test_group_round_trip_keeps_the_choice(store):
    G.set_object_source(store, "a", {"source": "core"})
    G.create_group(store, "g", ["a", "b"]) if hasattr(G, "create_group") else G.make_group(store, "g", ["a", "b"])
    assert G.member_source(store.get("a"))["source"] == "core"
    G.ungroup(store, "g", confirm=True)
    assert G.has_own_source(store.get("a")) and store.get("a").get("group") is None


def test_ui_and_route():
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "objects.js").read_text(encoding="utf-8")
    assert "function soloSelect(o)" in js and "soloSelect(o)}${tail}" in js and "/mask_source`" in js
    api = (root / "api" / "groups.py").read_text(encoding="utf-8")
    assert '@router.put("/api/object/{name}/mask_source")' in api
    assert "ol-drag ol-clean-ico" in js and "${cuOn ?" in js


def test_cleaned_marker_is_the_icon_only_2026_11_15():
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "objects.js").read_text(encoding="utf-8")
    assert "🧹 czyszczony" not in js and "if (!mi || isGroup(o) || o.group) return '';" in js   # in Cleanup too (2026-11-15)
    css = (root / "static" / "index.css").read_text(encoding="utf-8")
    ico = css[css.index(".other-layers .ol-clean-ico {"):]
    assert "linear-gradient(135deg, #f2bb66, #e8803d)" in ico.split("}")[0]


def test_c1_has_no_mask_picker_and_cards_pick_the_model_2026_11_15():
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "cuMaskSrc" not in js and "maskOptions" not in js and "zmień w palecie Obiekty" in js
    assert 'data-cu-pick="${esc(x.id)}"' in js and "data-cu-minstall" in js and "$('cuGuide').addEventListener('click'" in js
    assert "if (m.available) return vramLine(m.vram);" in js
    src = (root / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "return resolve(objects(), o, {})" in src
    assert "    ltx: { best: 'cały klip naraz" in js and "mem: 3, memT: 'bardzo dużo (model 21,5 GB" in js   # LTX-2.5 card bars


def test_cleanup_palette_shows_the_cutout_selection_result_2026_11_15():
    js = (Path(__file__).resolve().parents[1] / "static" / "objects.js").read_text(encoding="utf-8")
    assert "(o.name !== S.obj || isCleanMode()) && !isGroup(o)" in js
