"""Audit 2026-11-02 #6: core_fix/ with a hole. A frame without the mask 1+2 still falls back on the SAM 3.1 core, but the
export says so (`core_fix_fallback` in outputs.json + the API warning): silently mixing frames with and without the
corrections is what the audit found. An object with no core_fix/ at all has no corrections to lose: nothing to flag."""
import json
import logging

import cv2
import numpy as np

from engine.export import CORE_RUN, core_fix_holes, core_fix_loader, core_fix_warning, export_objects
from engine.objects import ObjectStore
from engine.outputs import load_outputs

W, H = 32, 16


def _object(store, name, core_frames, fix_frames, depth=0):
    store.create(name, [name], depth_order=depth)
    od = store.dir(name)
    (od / "core").mkdir(exist_ok=True)
    for f in core_frames:
        cv2.imwrite(str(od / "core" / f"c_{f:04d}.png"), np.full((H, W), 255, np.uint8))
    if fix_frames:
        (od / "core_fix").mkdir()
    for f in fix_frames:
        cv2.imwrite(str(od / "core_fix" / f"c_{f:04d}.png"), np.zeros((H, W), np.uint8))
    store.update(name, tool={"active_run": CORE_RUN, "final_valid": True})
    return od


def _store(tmp_path):
    return ObjectStore(tmp_path / "work"), tmp_path / "work"


def test_loader_marks_the_hole_in_the_log_once(tmp_path, caplog):
    s, _ = _store(tmp_path)
    od = _object(s, "a", [1, 2, 3], [1])
    load = core_fix_loader(od, W, H)
    with caplog.at_level(logging.WARNING, logger="engine.export"):
        assert load(1).max() == 0.0          # corrected frame
        assert load(2).min() == 1.0          # the fallback itself stays: the SAM 3.1 core
        assert load(3).min() == 1.0
    msgs = [r.getMessage() for r in caplog.records if "maski 1+2" in r.getMessage()]
    assert len(msgs) == 1 and "a" in msgs[0]


def test_holes_cover_only_a_partial_core_fix(tmp_path):
    s, _ = _store(tmp_path)
    assert core_fix_holes(_object(s, "whole", [1, 2], [1, 2]), [1, 2]) == []
    assert core_fix_holes(_object(s, "none", [1, 2], []), [1, 2]) == []     # no corrections at all: the core IS the mask 1+2
    assert core_fix_holes(_object(s, "part", [1, 2, 3], [1]), [1, 2, 3]) == [2, 3]


def _export(s, work, names, frames):
    objs = [s.get(n) for n in names]
    return export_objects(work, "shot", objs, frames, W, H, 8, False, export_format="png")


def test_export_reports_the_mixed_frames_and_clears_them(tmp_path):
    s, work = _store(tmp_path)
    od = _object(s, "a", [1, 2, 3], [1])
    out = _export(s, work, ["a"], [1, 2, 3])
    assert out["core_fix_fallback"] == {"a": [[2, 3]]}
    assert json.loads((work / "outputs.json").read_text(encoding="utf-8"))["core_fix_fallback"] == {"a": [[2, 3]]}
    w = core_fix_warning(out)
    assert "a" in w and "F2–3" in w
    # the frames written are unchanged: alpha of the fallback frame = the core
    png = cv2.imread(str(work / "export" / "mattes" / "a" / "a_0002.png"), cv2.IMREAD_GRAYSCALE)
    assert png.min() == 255
    for f in (2, 3):
        cv2.imwrite(str(od / "core_fix" / f"c_{f:04d}.png"), np.zeros((H, W), np.uint8))
    out = _export(s, work, ["a"], [1, 2, 3])
    assert "core_fix_fallback" not in out and "core_fix_fallback" not in load_outputs(work)
    assert core_fix_warning(out) == ""


def test_frames_outside_the_objects_life_are_no_hole(tmp_path):
    s, work = _store(tmp_path)
    _object(s, "a", [1, 2], [1, 2])
    s.update("a", life=[[1, 2]])
    out = _export(s, work, ["a"], [1, 2, 3])
    assert "core_fix_fallback" not in out


def test_group_reports_its_members_on_the_core_source(tmp_path):
    from engine import groups
    s, work = _store(tmp_path)
    _object(s, "hair", [1, 2], [1], depth=0)
    _object(s, "dress", [1, 2], [1, 2], depth=1)
    groups.create_group(work, "g", ["hair", "dress"])
    obj = {**s.get("g"), "tool": {**s.get("g").get("tool", {}), "active_run": CORE_RUN}}
    out = export_objects(work, "shot", [obj], [1, 2], W, H, 8, False, export_format="png")
    assert out["core_fix_fallback"] == {"hair": [[2, 2]]}


def test_front_context_object_is_reported_too(tmp_path):
    """The hole of a front object cuts the written matte with the wrong mask: it counts as well."""
    s, work = _store(tmp_path)
    _object(s, "front", [1, 2], [1], depth=0)
    _object(s, "back", [1, 2], [1, 2], depth=1)
    objs = [s.get("front"), s.get("back")]
    out = export_objects(work, "shot", objs, [1, 2], W, H, 8, False, write=["back"], export_format="png")
    assert out["core_fix_fallback"] == {"front": [[2, 2]]}
