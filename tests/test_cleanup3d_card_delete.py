# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D: a card can be removed (user request 2026-11-13) – its cards.json entry, keys (all versions) and extent
are deleted from disk for good (user decision 2026-11-13: no trash); the other cards stay as they were."""
from pathlib import Path

import pytest

from engine.cleanup3d import store
from engine.cleanup3d import types as T


def _card(cid, name):
    return {"id": cid, "name": name, "kind": "plane", "plane": {"n": [0, -1, 0], "d": 1},
            "adjust": {"dh": 0, "pitch": 0, "roll": 0}, "extent": [], "extent_frame": None, "seam_px": 16,
            "color": "#ff0000", "key_mode": "manual", "key_every": 8, "warps": {}}


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "shot" / "matte_desk"
    w.mkdir(parents=True)
    store.save_cards(w, T.cards_from_json({"cards": [{**_card("card_1", "ziemia"), "order": 0},
                                                     {**_card("card_2", "morze"), "order": 1}]}))
    kd = T.key_dir(w, "card_1", 12)
    kd.mkdir(parents=True)
    (kd / "key.json").write_text("{}", encoding="utf-8")
    (kd / "v001.png").write_bytes(b"png")
    ed = store.extent_dir(w, "card_1")
    ed.mkdir(parents=True)
    (ed / "a_1.png").write_bytes(b"png")
    return w


def test_delete_card_deletes_everything_for_good(work):
    freed = store.delete_card(work, "card_1")
    assert freed == len(b"{}") + 2 * len(b"png")              # key.json + v001.png + the extent layer
    assert [c.id for c in store.load_cards(work)] == ["card_2"]
    assert not (T.root_dir(work) / "keys" / "card_1").exists() and not store.extent_dir(work, "card_1").exists()
    assert not (T.root_dir(work) / "trash").exists()          # nothing went to a trash folder
    assert store.delete_card(work, "card_2") == 0             # a card without keys / extent
    assert store.load_cards(work) == []
    with pytest.raises(ValueError):
        store.delete_card(work, "card_9")


def test_api_and_ui_are_wired():
    import inspect
    import api.cleanup3d as AC
    src = inspect.getsource(AC)
    assert '@router.delete("/api/cleanup3d/cards/{card}")' in src and "store.delete_card(work, card)" in src
    js = (Path(__file__).resolve().parents[1] / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    assert "data-c3del=" in js and "closest('[data-c3del]')" in js and "/api/cleanup3d/cards/${encodeURIComponent(id)}" in js
