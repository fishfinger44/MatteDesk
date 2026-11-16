# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Final Touch as live Nuke nodes in the export's .nk (engine/ft_nuke.py, user decision 2026-11-16): the RotoPaint text
(stroke order, single-frame lifetime, add / remove / erase, y flip, pen pressure), the finish nodes, the MASKI column
of such an object (wiring with the .nk stack simulator) and the export writing the base / front sequences. The node
values were calibrated against Nuke 15.0v1 renders; opening the script in Nuke → LOCAL_CHECKS."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import final_touch as ft
from engine import ft_nuke
from engine.nuke_mattes import mattes_nk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_export_nk_mergematte import wiring  # noqa: E402


def test_brush_params_follow_the_calibration():
    bs, h, bu = ft_nuke.brush_params(20, 1.0)
    assert bs == pytest.approx(42.0) and h == pytest.approx(1.0) and bu == 0
    bs, h, bu = ft_nuke.brush_params(10, 0.6)
    assert bs == pytest.approx(26.0) and h == pytest.approx(0.5) and bu == 1
    bs, h, _ = ft_nuke.brush_params(10, 0.65)          # between two rows: linear
    assert bs == pytest.approx(25.5) and h == pytest.approx(0.55)
    hs = [ft_nuke.brush_params(10, x / 10)[1] for x in range(11)]
    assert hs == sorted(hs) and hs[0] == 0.0 and hs[-1] == 1.0


def test_rotopaint_text_order_lifetime_modes_and_flip():
    frames = {"12": [{"mode": "add", "radius_px": 10, "hardness": 1.0, "p": [[10, 20], [30, 20]]},
                     {"mode": "remove", "radius_px": 5, "hardness": 0.6, "p": [[15, 25]]},
                     {"mode": "erase", "radius_px": 5, "hardness": 0.6, "p": [[16, 26, 2.5], [18, 26, 5]]}],
              "7": [{"mode": "add", "radius_px": 4, "hardness": 1.0, "p": [[1, 1]]}]}
    lines = ft_nuke.rotopaint_lines(frames, 100, 50, "FT_paint_x", 0, 0, "lbl")
    assert lines[0] == "RotoPaint {" and lines[-1] == "}"
    cur = next(x for x in lines if x.startswith(" curves "))
    assert cur.count("{") == cur.count("}")                        # one balanced line
    names = re.findall(r"cubiccurve (\S+) 512", cur)
    assert names == ["Erase_F12_3", "Remove_F12_2", "Add_F12_1", "Add_F7_1"]   # first = on top: reverse order
    curves = dict(zip(names, re.split(r"\{cubiccurve \S+ 512 ", cur)[1:]))
    assert "ltt 2 ltn 12 ltm 12" in curves["Add_F12_1"] and "ltt 2 ltn 7 ltm 7" in curves["Add_F7_1"]
    assert "{10.5 29.5 1} {30.5 29.5 1}" in curves["Add_F12_1"]      # x + 0.5, H − (y + 0.5)
    assert "r 0 g 0 b 0 a 0" in curves["Remove_F12_2"] and "src 2" not in curves["Remove_F12_2"]
    assert "src 2" in curves["Erase_F12_3"] and "ds 1" in curves["Erase_F12_3"] and "{16.5 23.5 0.5}" in curves["Erase_F12_3"]
    assert "ds 1" not in curves["Add_F12_1"] and "bs 21 h 1 bu 0" in curves["Add_F12_1"]


def test_finish_nodes_and_availability():
    assert ft_nuke.finish_nodes({}) == []
    n = dict(ft_nuke.finish_nodes({"black": 0.1, "white": 0.9, "gamma": 1.5, "erode_px": 3, "blur_px": 2}))
    assert n["Grade"]["blackpoint"] == "0.1" and n["Grade"]["whitepoint"] == "0.9" and n["Grade"]["gamma"] == "1.5"
    assert n["Grade"]["white_clamp"] == "true" and n["Grade"]["black_clamp"] == "true"
    assert n["Dilate"]["size"] == "-3"                               # Dilate: negative erodes
    assert n["Blur"]["size"] == "4.7096"                             # 2.3548 σ
    assert dict(ft_nuke.finish_nodes({"erode_px": -2}))["Dilate"]["size"] == "2"
    assert [c for c, _ in ft_nuke.finish_nodes({"gamma": 2})] == ["Grade"]
    assert ft_nuke.available({"mb_strength": 0})[0] and not ft_nuke.available({"mb_strength": 0.5})[0]


def _outputs(tmp_path):
    return {"shot": "SYN",
            "objects": [{"name": "hair", "depth_order": 0, "matte": "m/hair/hair_####.exr", "version": "ft:v002"},
                        {"name": "dress", "depth_order": 1, "matte": "m/dress/dress_####.exr", "version": "ft:v001"}],
            "mattes_all": "m/all/SYN_mattes_####.exr"}


def test_maski_column_rebuilds_the_matte_with_nodes(tmp_path):
    spec_paint = {"mode": "paint", "frames": {"3": [{"mode": "add", "radius_px": 5, "hardness": 1, "p": [[4, 4]]}]},
                  "finish": {"black": 0.1, "blur_px": 1}, "base": "m/dress_ftbase/dress_ftbase_####.exr",
                  "front": "m/dress_front/dress_front_####.exr"}
    txt = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 64, 48, date="d", channels=["hair", "dress", "union"],
                    ft={"dress": spec_paint})
    ins, cls = wiring(txt)
    # hair: the multi-EXR column as before; dress: its own Read of the Final Touch base + the chain + the front cut
    assert ins["MATTE_hair"] == ["Dot_hair"] and ins["Dot_hair"] == ["Read_SYN_mattes"]
    assert "Dot_dress" not in ins and cls["Read_dress"] == "Read"
    assert 'file "m/dress_ftbase/dress_ftbase_####.exr"' in txt
    assert ins["MATTE_dress"] == ["Read_dress"]
    assert ins["FT_paint_dress"] == ["MATTE_dress"] and cls["FT_paint_dress"] == "RotoPaint"
    assert ins["FT_grade_dress"] == ["FT_paint_dress"] and ins["FT_blur_dress"] == ["FT_grade_dress"]
    assert ins["FRONT_dress"] == ["Read_dress_front"]
    assert ins["FT_cut_front_dress"] == ["FRONT_dress", "FT_blur_dress"]   # input0 B = front, input1 A = chain: A − B
    assert "operation minus" in txt and ins["FT_clamp_dress"] == ["FT_cut_front_dress"]
    assert ins["Anchor_dress"] == ["FT_clamp_dress"]
    # every anchor moves down by the longest chain (no overlap), the stamps stay under their anchors
    ys = {m.group(1): int(m.group(2)) for m in re.finditer(r"name (Anchor_\w+)\n(?:.*\n)*? ypos (\d+)", txt)}
    assert len(set(ys.values())) == 1 and list(ys.values())[0] > 280
    # no Final Touch nodes: the column is the multi-EXR one again
    txt0 = mattes_nk(_outputs(tmp_path), "SYN", 1, 10, 64, 48, date="d", channels=["hair", "dress", "union"])
    assert "RotoPaint" not in txt0 and "Read_dress" not in txt0


def test_specs_from_outputs(tmp_path):
    od = tmp_path / "objects" / "dress"
    ft.set_frame(od, 3, [{"mode": "add", "radius_px": 5, "p": [[4, 4]]}])
    ft.set_finish(od, {"gamma": 1.4})
    out = _outputs(tmp_path)
    out["objects"][1]["final_touch"] = {"base": "v001", "corrections_sig": "x",
                                        "nuke": {"mode": "grade", "base_matte": "b_####.exr"}}
    out["objects"][0]["final_touch"] = {"base": "v002", "corrections_sig": "y", "nuke": {"mode": "bake", "requested": "paint", "why": "mb"}}
    sp = ft_nuke.specs(tmp_path, out)
    assert list(sp) == ["dress"] and sp["dress"]["mode"] == "grade" and sp["dress"]["frames"] == {}   # grade: strokes baked
    assert sp["dress"]["finish"]["gamma"] == pytest.approx(1.4) and sp["dress"]["front"] is None


def test_nuke_mode_store(tmp_path):
    od = tmp_path / "o"
    assert ft.load(od)["nuke_mode"] == "bake"
    ft.set_nuke_mode(od, "paint")
    assert ft.load(od)["nuke_mode"] == "paint"
    with pytest.raises(ValueError):
        ft.set_nuke_mode(od, "rotoscope")


# ------------------------------------------------------------------------------------------------ API + export
@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_export_paint_mode_writes_base_and_nodes(proj):
    app_module, api, shot, pipe, frames, start = proj
    c = pipe.load_core("obj", [start])[start]
    ys, xs = np.where(c)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    strokes = [{"mode": "remove", "radius_px": 6, "hardness": 1.0, "p": [[cx, cy]]}]
    assert api.put(f"/api/object/obj/final_touch/frames/{start}", json={"strokes": strokes}).status_code == 200
    assert api.put("/api/object/obj/final_touch/finish", json={"gamma": 1.3}).status_code == 200
    assert api.put("/api/object/obj/final_touch/nuke", json={"mode": "x"}).status_code == 400
    d = api.put("/api/object/obj/final_touch/nuke", json={"mode": "paint"}).json()
    assert d["nuke"] == {"mode": "paint", "available": True, "why": ""}
    r = api.post("/api/export", json={"objects": ["obj"], "format": "png", "source": "final_touch"})
    assert r.status_code == 200, r.text
    out = r.json()["outputs"]
    ent = next(e for e in out["objects"] if e["name"] == "obj")
    nk = ent["final_touch"]["nuke"]
    assert nk["mode"] == "paint" and "obj_ftbase" in nk["base_matte"] and "front_matte" not in nk
    root = Path(out["export_dir"]) / "mattes"
    baked = cv2.imread(str(root / "obj" / f"obj_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    base = cv2.imread(str(root / "obj_ftbase" / f"obj_ftbase_{start:04d}.png"), cv2.IMREAD_GRAYSCALE)
    assert baked[cy, cx] == 0 and base[cy, cx] > 127                 # the matte stays baked, the base has no stroke
    txt = Path(out["mattes_nk"]).read_text(encoding="utf-8")
    assert "RotoPaint {" in txt and f"ltn {start}" in txt and "FT_grade_obj" in txt
    from engine.contract import validate
    assert validate(out) == []
    # clipboard text = the regenerated file
    t = api.get("/api/export/nk_text").json()
    assert "RotoPaint {" in t["text"] and t["nk"] == out["mattes_nk"]
    # motion blur on: only baked, with the reason in the warning
    assert api.put("/api/object/obj/final_touch/finish", json={"mb_strength": 0.5}).status_code == 200
    assert api.get("/api/object/obj/final_touch").json()["nuke"]["available"] is False
    r = api.post("/api/export", json={"objects": ["obj"], "format": "png", "source": "final_touch"})
    assert r.status_code == 200, r.text
    ent = next(e for e in r.json()["outputs"]["objects"] if e["name"] == "obj")
    assert ent["final_touch"]["nuke"]["mode"] == "bake" and ent["final_touch"]["nuke"]["requested"] == "paint"
    assert "wpalony zamiast węzłów" in r.json()["warning"]
    assert "RotoPaint" not in Path(r.json()["outputs"]["mattes_nk"]).read_text(encoding="utf-8")


def test_section_paste_text_grade_and_paint(tmp_path):
    od = tmp_path / "o"
    ft.set_finish(od, {"black": 0.12, "blur_px": 6})
    ft.set_frame(od, 4, [{"mode": "add", "radius_px": 5, "p": [[3, 3]]}])
    txt, nodes = ft_nuke.paste_text("grade", ft.load(od), 64, 48, "Hair 1")
    assert txt.startswith("set cut_paste_input [stack 0]\nversion 15.0 v1\npush $cut_paste_input\n")
    assert nodes == ["FT_grade_Hair_1", "FT_dilate_Hair_1", "FT_blur_Hair_1"]   # all three, 0 = no-op to tweak in Nuke
    assert "blackpoint 0.12" in txt and "size 0" in txt and "size 14.1288" in txt
    ins, cls = wiring(txt.replace("push $cut_paste_input", "push 0").replace("set cut_paste_input [stack 0]\n", ""))
    assert ins["FT_dilate_Hair_1"] == ["FT_grade_Hair_1"] and ins["FT_blur_Hair_1"] == ["FT_dilate_Hair_1"]
    txt, nodes = ft_nuke.paste_text("paint", ft.load(od), 64, 48, "Hair 1")
    assert nodes == ["FT_paint_Hair_1"] and "RotoPaint {" in txt and "ltn 4" in txt
    with pytest.raises(ValueError):
        ft_nuke.paste_text("mb", ft.load(od), 64, 48, "o")


def test_section_export_endpoint(proj):
    app_module, api, shot, pipe, frames, start = proj
    assert api.put(f"/api/object/obj/final_touch/frames/{start}",
                   json={"strokes": [{"mode": "add", "radius_px": 4, "p": [[5, 5]]}]}).status_code == 200
    r = api.get("/api/object/obj/final_touch/nk?part=paint")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["nodes"] == ["FT_paint_obj"] and Path(d["path"]).read_text(encoding="utf-8") == d["text"]
    assert d["path"].endswith("final_touch/FT_paint_obj.nk")
    assert api.get("/api/object/obj/final_touch/nk?part=grade").json()["nodes"][0] == "FT_grade_obj"
    assert api.get("/api/object/obj/final_touch/nk?part=nope").status_code == 400
