# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z29 – export folder chosen by the user (project.json -> export_dir) + Nuke script with the mattes from the multi-EXR.

Script rules (user's Nuke layout): one backdrop MASKI around everything, per channel Shuffle2 `<layer>.alpha` -> rgba named
MATTE_<obj> + anchor + free copy stamp under it, a Dot on every branch (chain of Dots), columns in depth order, labels with
the version and date. Opening in Nuke 15 -> LOCAL_CHECKS.
"""

import json
from pathlib import Path

import pytest

from engine.nuke_mattes import mattes_nk, nk_nodes


def _outputs(tmp_path):
    return {"schema": "matte_desk/outputs@1.0", "shot": "SYN",
            "objects": [{"name": "tree", "depth_order": 2, "matte": str(tmp_path / "mattes/tree/tree_####.exr"), "version": "v003"},
                        {"name": "figure", "depth_order": 0, "matte": str(tmp_path / "mattes/figure/figure_####.exr"), "version": "v002"},
                        {"name": "suknia", "depth_order": 1, "matte": str(tmp_path / "mattes/suknia/suknia_####.exr"), "version": "v001"}],
            "groups": {"figure": {"matte": str(tmp_path / "mattes/figure_full/figure_full_####.exr"), "members": ["figure", "suknia"]}},
            "mattes_all": str(tmp_path / "mattes/all/SYN_mattes_####.exr").replace("\\", "/")}


def _braces_ok(text):
    d = 0
    for ch in text:
        d += ch == "{"
        d -= ch == "}"
        assert d >= 0
    return d == 0


def _xy(node):
    return int(node["knobs"]["xpos"]), int(node["knobs"]["ypos"])


def test_multi_exr_script_structure(tmp_path):
    out = _outputs(tmp_path)
    chans = ["figure_full", "figure", "suknia", "tree", "union", "depth"]
    txt = mattes_nk(out, "SYN", 1001, 1193, 4000, 2000, date="2026-10-12 23:00", channels=chans)
    assert _braces_ok(txt) and "{{curve}} {{curve" not in txt
    nodes = nk_nodes(txt)
    by = lambda cls: [n for n in nodes if n["class"] == cls]
    reads = by("Read")
    assert len(reads) == 1
    r = reads[0]["knobs"]
    assert r["file"] == f'"{out["mattes_all"]}"' and r["first"] == "1001" and r["last"] == "1193" and r["raw"] == "true"
    shuf = by("Shuffle2")
    assert [n["knobs"]["name"] for n in shuf] == [f"MATTE_{c}" for c in chans]     # depth order, whole before its parts
    assert all("in1" in n["knobs"] for n in shuf)
    assert 'figure.alpha 0 0 rgba.alpha 0 3' in txt and "depth.Z 0 0 rgba.red 0 0" in txt
    anchors = [n["knobs"]["name"] for n in by("NoOp")]
    stamps = by("PostageStamp")
    assert anchors == [f"Anchor_{c}" for c in chans]
    assert [n["knobs"]["name"] for n in stamps] == [f"Stamp_copy_{c}" for c in chans]
    assert all(n["knobs"]["hide_input"] == "true" for n in stamps)
    assert len(by("Dot")) == len(chans)                                          # a Dot on every branch
    for c in ("figure", "suknia", "tree"):                                        # version + date in the labels
        lab = next(n for n in shuf if n["knobs"]["name"] == f"MATTE_{c}")["knobs"]["label"]
        assert out["objects"][[o["name"] for o in out["objects"]].index(c)]["version"] in lab and "2026-10-12" in lab
    assert "v002+v001" in next(n for n in shuf if n["knobs"]["name"] == "MATTE_figure_full")["knobs"]["label"]
    for c in chans:
        assert f"add_layer {{{c} " in txt or c == "depth"
    # backdrop MASKI around every node, no two nodes at the same place
    bd = by("BackdropNode")
    assert len(bd) == 1 and "MASKI" in bd[0]["knobs"]["label"]
    bx, byy = _xy(bd[0])
    bw, bh = int(bd[0]["knobs"]["bdwidth"]), int(bd[0]["knobs"]["bdheight"])
    pos = [_xy(n) for n in nodes if n["class"] != "BackdropNode"]
    assert len(pos) == len(set(pos))
    for x, y in pos:
        assert bx <= x and x + 80 <= bx + bw and byy <= y and y + 20 <= byy + bh, (x, y)
    # wiring: each Dot after the first is fed by the previous Dot (push $Dot_prev)
    assert txt.count("push $Dot_") == len(chans) - 1


def test_single_object_exports_get_their_own_read(tmp_path):
    out = _outputs(tmp_path)
    out.pop("mattes_all")
    txt = mattes_nk(out, "SYN", 1, 10, 640, 480, date="d")
    nodes = nk_nodes(txt)
    reads = [n["knobs"] for n in nodes if n["class"] == "Read"]
    files = [r["file"].strip('"') for r in reads]
    assert files == [o["matte"].replace("\\", "/") for o in sorted(out["objects"], key=lambda o: o["depth_order"])] + \
        [out["groups"]["figure"]["matte"].replace("\\", "/")]
    assert [n["knobs"]["name"] for n in nodes if n["class"] == "NoOp"] == ["Anchor_figure", "Anchor_suknia", "Anchor_tree", "Anchor_figure_full"]
    shuf = [n["knobs"] for n in nodes if n["class"] == "Shuffle2"]                  # matte is in rgb -> red into rgba
    assert [k["name"] for k in shuf] == ["MATTE_figure", "MATTE_suknia", "MATTE_tree", "MATTE_figure_full"]
    assert "rgba.red 0 0 rgba.alpha 0 3" in txt and _braces_ok(txt)
    pos = [_xy(n) for n in nodes if n["class"] != "BackdropNode"]
    assert len(pos) == len(set(pos))
    # Read (tall: postage stamp ~100 px) does not reach the Shuffle / anchor rows of its column
    assert all(_xy(n)[1] + 110 <= 190 for n in nodes if n["class"] == "Read")


def test_anchor_and_copy_stamp_are_stamps_nodes(tmp_path):
    """Stamps v1.1 knob sets (as in the user's scripts): anchor identifier = anchor + title; the copy is a Wired Stamp
    (identifier = wired, title, `anchor` = anchor node name) – plain PostageStamps are not recognised by Stamps."""
    txt = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 640, 480, date="d", channels=["figure", "union"])
    nodes = nk_nodes(txt)
    for c in ("figure", "union"):
        i = txt.index(f" name Anchor_{c}\n")
        a_blk = txt[i: txt.index("\n}\n", i)]
        assert "addUserKnob {26 identifier -STARTLINE +HIDDEN T anchor}" in a_blk and f' title "{c}"' in a_blk
        st = next(n["knobs"] for n in nodes if n["class"] == "PostageStamp" and n["knobs"]["name"] == f"Stamp_copy_{c}")
        assert st["anchor"] == f"Anchor_{c}" and st["hide_input"] == "true" and st["title"] == f'"{c}"'
        j = txt.index(f" name Stamp_copy_{c}\n")
        assert "addUserKnob {26 identifier -STARTLINE +HIDDEN T wired}" in txt[j: txt.index("\n}\n", j)]
    assert _braces_ok(txt)


def test_export_into_chosen_dir_writes_nk(tmp_path):
    from engine.contract import validate
    from engine.export import export_objects
    from test_e6_acceptance import _two_objects
    shot, pipe, frames, work = _two_objects(tmp_path, n=8)
    chosen = tmp_path / "chosen_export"
    out = export_objects(work, "SYN", pipe.store.list(), frames, shot.w, shot.h, bit_depth=16, multichannel=True,
                         export_dir=chosen)
    assert validate(out) == []
    assert (chosen / "mattes" / "all" / f"SYN_mattes_{frames[2]:04d}.exr").exists()
    assert not (work / "export" / "mattes" / "all").exists()                     # nothing in the default folder
    assert out["export_dir"] == str(chosen).replace("\\", "/") and out["mattes_all_channels"] == ["puppet", "tree", "union"]
    nk = Path(out["mattes_nk"])
    assert nk == chosen / "SYN_mattes.nk" and nk.exists()
    nodes = nk_nodes(nk.read_text(encoding="utf-8"))
    assert [n["knobs"]["name"] for n in nodes if n["class"] == "Shuffle2"] == ["MATTE_puppet", "MATTE_tree", "MATTE_union"]
    r = next(n for n in nodes if n["class"] == "Read")["knobs"]
    assert r["file"].strip('"') == out["mattes_all"] and r["first"] == str(frames[0]) and r["last"] == str(frames[-1])


def test_export_dir_api(tmp_path, monkeypatch):
    from test_ui_stages_api import _open
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    d = api.get("/api/project/export_dir").json()
    assert d["is_default"] and d["path"].endswith("matte_desk/export")
    inside = tmp_path / "work" / "deliver"
    d = api.put("/api/project/export_dir", json={"path": str(inside)}).json()
    assert d["path"] == str(inside).replace("\\", "/") and not d["is_default"] and not d["outside_work"]
    pj = json.loads((tmp_path / "work" / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    assert pj["export_dir"] == str(inside).replace("\\", "/")
    outside = tmp_path / "elsewhere"
    r = api.put("/api/project/export_dir", json={"path": str(outside)})
    assert r.status_code == 400 and "poza projektem" in r.json()["detail"]
    d = api.put("/api/project/export_dir", json={"path": str(outside), "confirmed": True}).json()
    assert d["outside_work"]
    assert api.put("/api/project/export_dir", json={"path": "relative/dir"}).status_code == 400
    d = api.put("/api/project/export_dir", json={"path": None}).json()
    assert d["is_default"]
    assert "export_dir" not in json.loads((tmp_path / "work" / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    assert api.post("/api/export/nk").status_code == 409                          # nothing exported yet
    assert mf.read_bytes() == before
