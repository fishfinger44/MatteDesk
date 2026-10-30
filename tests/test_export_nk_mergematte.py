# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export .nk with the backdrop MERGEMATTE (PLAN_FINAL_TOUCH §11, step 3): only objects whose MergeMatte is on, RGB from
the plate Read (manifest beauty, its colorspace), alpha from the MASKI anchor through a wired stamp, the group with the
object's knobs, anchor FUSE_<obj> + free stamp. The wiring is checked with a small simulator of the Nuke script stack
(push / set / inputs N, a Group's inner nodes skipped). Import in Nuke → LOCAL_CHECKS."""

from __future__ import annotations

import re
from pathlib import Path

from engine import final_touch as ft
from engine import merge_matte_nk as mmnk
from engine.nuke_mattes import mattes_nk


def _outputs(tmp_path):
    return {"shot": "SYN",
            "objects": [{"name": "hair", "depth_order": 0, "matte": str(tmp_path / "m/hair/hair_####.exr"), "version": "v002"},
                        {"name": "dress", "depth_order": 1, "matte": str(tmp_path / "m/dress/dress_####.exr"), "version": "v001"}],
            "mattes_all": str(tmp_path / "m/all/SYN_mattes_####.exr").replace("\\", "/")}


def wiring(text: str):
    """{node name: [input names]} of the top-level graph – a simulator of the .nk stack."""
    stack, vars_, inputs, cls = [], {}, {}, {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"^([A-Za-z0-9_]+) \{$", s)
        if s.startswith("push "):
            arg = s[5:]
            stack.append(None if arg == "0" else vars_[arg[1:]])
        elif s.startswith("set "):
            vars_[s.split()[1]] = stack[-1]
        elif m:
            c = m.group(1)
            body, depth = [], 1
            i += 1
            while depth:
                t = lines[i].strip()
                if t.endswith("{") and not t.startswith("addUserKnob"):
                    depth += 1
                if t == "}" or (t.endswith("}") and t.startswith("}")):
                    depth -= 1
                body.append(t)
                i += 1
            i -= 1
            knobs = dict((b.split(" ", 1) + [""])[:2] for b in body if b and not b.startswith("addUserKnob"))
            name = knobs.get("name", c)
            n_in = knobs.get("inputs", "1")
            n = 0 if c in ("Read", "BackdropNode") else int(n_in.split("+")[0]) if n_in else 1
            if c == "Group":   # skip the inner nodes up to end_group
                while lines[i + 1].strip() != "end_group":
                    i += 1
                i += 1
            ins = [stack.pop() if stack else None for _ in range(n)] if n else []
            inputs[name], cls[name] = ins, c
            if c != "BackdropNode":
                stack.append(name)
        i += 1
    return inputs, cls


def _merge(tmp_path, on=("hair",)):
    for o in ("hair", "dress"):
        (tmp_path / "objects" / o).mkdir(parents=True, exist_ok=True)
    for o in on:
        ft.set_merge_matte(tmp_path / "objects" / o, on=True, reset="sh010")
    return ft.merge_matte_export(tmp_path, ["hair", "dress"], "D:/plates/SYN/SYN_####.exr", "ACES2065-1")


def test_no_object_on_means_no_backdrop(tmp_path):
    assert _merge(tmp_path, on=()) is None
    txt = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 640, 480, date="d", channels=["hair", "dress", "union"])
    assert "MERGEMATTE" not in txt and "MergeMatte" not in txt


def test_backdrop_wiring_and_knobs(tmp_path):
    merge = _merge(tmp_path)
    assert list(merge["objects"]) == ["hair"]                      # only the object with MergeMatte on
    txt = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 640, 480, date="d", channels=["hair", "dress", "union"], merge=merge)
    assert txt.count("{") == txt.count("}")
    ins, cls = wiring(txt)
    assert cls["Read_SYN_plate"] == "Read" and "BD_MERGEMATTE" in cls
    assert 'colorspace "ACES - ACES2065-1"' in txt and 'file "D:/plates/SYN/SYN_####.exr"' in txt
    assert ins["Dot_plate_hair"] == ["Read_SYN_plate"]
    assert ins["Stamp_matte_hair"] == ["Anchor_hair"]               # alpha from the MASKI anchor, through a stamp
    assert ins["Copy_alpha_hair"] == ["Dot_plate_hair", "Stamp_matte_hair"]   # B = plate (RGB), A = matte (alpha)
    assert ins["Premult_hair"] == ["Copy_alpha_hair"]
    assert cls["MergeMatte_hair"] == "Group" and ins["MergeMatte_hair"] == ["Dot_BG_hair", "Premult_hair"]   # 0 = BG, 1 = FG
    assert ins["Dot_BG_hair"] == []
    assert ins["Anchor_FUSE_hair"] == ["MergeMatte_hair"] and ins["Stamp_copy_FUSE_hair"] == ["Anchor_FUSE_hair"]
    # the MASKI part is unchanged: anchors and their free stamps
    assert ins["Stamp_copy_hair"] == ["Anchor_hair"] and ins["Anchor_hair"] == ["MATTE_hair"]
    g = txt[txt.index("name MergeMatte_hair"):]
    for k, v in mmnk.PRESET_SH010.items():
        if k in ("log", "bypass"):
            continue
        assert re.search(rf"^ {k} {v:.6g}$", g, re.M), k


def test_unknown_colorspace_is_left_to_the_user(tmp_path):
    merge = _merge(tmp_path)
    merge["colorspace"] = "ARRI LogC4"
    txt = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 640, 480, date="d", channels=["hair", "dress", "union"], merge=merge)
    read = txt[txt.index("name Read_SYN_plate") - 400:txt.index("name Read_SYN_plate")]
    assert "colorspace" not in read.split("Read {")[-1] and "ustaw ręcznie" in txt
