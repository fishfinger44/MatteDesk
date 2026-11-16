# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„Wejście” of Inpaint (user 2026-11-15): Beauty (default) or an earlier Inpaint version's composite as the plate to
work on; frames that version lacks come from Beauty; the run, anchors, recomposite and upscale read the same input."""
import json
from pathlib import Path

from engine import cleanup as cu

ROOT = Path(__file__).resolve().parents[1]


def test_setting():
    assert cu.settings({})["input_plate"] == "" and "input_plate" in cu.PER_TARGET
    assert cu.settings({"input_plate": "v031"})["input_plate"] == "v031"
    assert cu.settings({"input_plate": "../x"})["input_plate"] == ""


def test_input_frame_fn(tmp_path):
    work = tmp_path
    d = cu.cleanup_dir(work) / "v005"
    (d / "comp").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"ext": "png", "comp_ext": "png", "frames": [1, 2]}))
    (d / "comp" / "c_0001.png").write_bytes(b"x")
    beauty = lambda f: f"B{f}"                                              # noqa: E731
    assert cu.input_frame_fn(work, {"input_plate": ""}, beauty) is beauty
    fn = cu.input_frame_fn(work, {"input_plate": "v005"}, beauty)
    assert fn(1).endswith("c_0001.png") and fn(2) == "B2"                   # missing frame: Beauty
    assert cu.input_missing(work, {"input_plate": "v005"}, [1, 2, 3]) == [2, 3]
    assert cu.input_frame_fn(work, {"input_plate": "v009"}, beauty) is beauty   # no such version


def test_api_and_ui_wired():
    src = (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert src.count("cu.input_frame_fn(work,") == 4 and "plate_file=src" in src
    assert 'cl.report["input_plate"] = s["input_plate"]' in src
    js = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert 'id="cuInput"' in js and "function renderInput(d)" in js and "save({ input_plate: e.target.value })" in js
    assert "(v.input_plate ? ` · na ${v.input_plate}` : '')" in js
