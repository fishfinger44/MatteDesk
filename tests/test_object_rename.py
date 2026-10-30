# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Object rename (2026-10-16): folder moved, exact references replaced, prompts untouched."""
import json

import pytest

from engine.objects import ObjectStore


def test_rename_moves_folder_and_exact_references(tmp_path):
    st = ObjectStore(tmp_path)
    st.create("Hands", prompts=["Hands in the foreground"])
    st.create("Head")
    st.update("Head", tool={"depends_on": ["Hands"]})
    from engine.groups import create_group
    create_group(st, "Body", ["Hands", "Head"])
    (st.dir("Hands") / "zones").mkdir(exist_ok=True)
    (st.dir("Hands") / "zones" / "z_01.json").write_text(json.dumps({"params": {"sure_fg": ["Hands", "HandsX"]}}), encoding="utf-8")
    n = st.rename("Hands", "dogs")
    assert n >= 2
    assert not (tmp_path / "objects" / "Hands").exists() and (tmp_path / "objects" / "dogs" / "object.json").exists()
    d = st.get("dogs")
    assert d["name"] == "dogs" and d["detect"]["prompts"] == ["Hands in the foreground"]   # substring in a prompt kept
    h = st.get("Head")
    assert h["tool"]["depends_on"] == ["dogs"] and st.get("dogs")["group"] == "Body"
    assert st.get("Body")["members"] == ["dogs", "Head"]
    z = json.loads((st.dir("dogs") / "zones" / "z_01.json").read_text(encoding="utf-8"))
    assert z["params"]["sure_fg"] == ["dogs", "HandsX"]


def test_rename_refuses_existing_and_bad_names(tmp_path):
    st = ObjectStore(tmp_path)
    st.create("a"); st.create("b")
    with pytest.raises(FileExistsError):
        st.rename("a", "b")
    with pytest.raises(ValueError):
        st.rename("a", "zła nazwa")
    with pytest.raises(FileNotFoundError):
        st.rename("nope", "c")
