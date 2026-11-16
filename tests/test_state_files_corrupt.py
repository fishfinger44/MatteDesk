"""Audit 2026-11-02 #3: a corrupt `project.json` / `outputs.json` (power cut mid-write, a stray editor) must stop with
the file's name, not silently become `{}` and be overwritten by the next save (plates_added, export_dir, the outputs
registry were lost that way). A missing file is still fine (fresh project)."""
import json

import pytest

from engine.outputs import StateFileError as OutputsError, load_outputs, update_outputs
from engine.project import ShotProject, StateFileError


def _manifest(tmp_path):
    m = tmp_path / "shot_manifest_t.json"
    m.write_text(json.dumps({"shot": "t", "first": 1, "last": 3, "plate_w": 64, "plate_h": 32,
                             "plates": {"beauty": str(tmp_path / "p.%04d.exr")}, "work": str(tmp_path / "work")}),
                 encoding="utf-8")
    return m


def test_corrupt_project_json_raises_and_is_kept(tmp_path):
    m = _manifest(tmp_path)
    proj = ShotProject.open(m, {})
    proj.data["export_dir"] = "X"
    proj.save()
    pj = proj.work_dir / "project.json"
    pj.write_text('{"schema": "matte_desk/project@1.0", "export_dir": "X", "plates_ad', encoding="utf-8")   # truncated
    with pytest.raises(StateFileError) as e:
        ShotProject.open(m, {})
    assert "project.json" in str(e.value)
    assert pj.read_text(encoding="utf-8").endswith("plates_ad")        # untouched: the user can repair it


def test_missing_project_json_is_a_fresh_project(tmp_path):
    m = _manifest(tmp_path)
    proj = ShotProject.open(m, {})
    assert proj.data["plates_added"] == {}


def test_corrupt_outputs_json_raises_and_is_kept(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    update_outputs(work, "depth", {"path": "d"})
    p = work / "outputs.json"
    p.write_text('{"schema": "matte_desk/outputs@1.0", "objects": [', encoding="utf-8")
    with pytest.raises(OutputsError) as e:
        load_outputs(work)
    assert "outputs.json" in str(e.value)
    with pytest.raises(OutputsError):
        update_outputs(work, "depth", {"path": "e"})
    assert p.read_text(encoding="utf-8").endswith("[")
    assert load_outputs(tmp_path / "nothing")["objects"] == []         # missing = empty registry
