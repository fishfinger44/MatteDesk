# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 #1 (plan step 4): ONE way to build the pipeline of a shot – `ObjectPipeline.from_project` in the engine.
The UI („Przelicz” – app._pipeline) and the night queue („Policz wszystko” – app._queue_worker) used to build it apart: the
queue had no backlit, no gain maps, no clean plate, no shutter, so the overnight result differed from the UI's."""
import json
from pathlib import Path

import pytest

import app as app_module
from engine.pipeline import ObjectPipeline
from test_ui_stages_api import _open


def _rich_project(tmp_path, monkeypatch):
    """A synthetic shot whose manifest sets every input the pipeline takes: backlit + clean plate (the beauty files
    again: same size, so not skipped), shutter, 16-bit, gain maps on disk, denoise ON, a work area."""
    api, shot, mf = _open(tmp_path, monkeypatch)
    m = json.loads(mf.read_text(encoding="utf-8"))
    m["plates"]["backlit"] = shot.plate_pattern
    m["plates"]["cleanplate"] = shot.plate_pattern
    m["shutter_deg"] = 123.0
    m["bit_depth"] = 16
    mf.write_text(json.dumps(m), encoding="utf-8")
    (tmp_path / "work" / "matte_desk" / "deflicker" / "gain_beauty").mkdir(parents=True)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    proj = app_module.STATE["shot_project"]
    proj.data["mask_plate_denoise"] = "on"
    proj.data["work_area"] = [shot.first + 1, shot.first + 4]
    proj.save()
    app_module._refresh_manifest_view()
    return api, shot, mf, proj


class _Recorder:
    """Replaces ObjectPipeline.__init__ (a plain function, so it binds): keeps every constructor call, builds nothing."""
    def __init__(self):
        self.calls = []

    def init(self):
        def _init(pipe, *args, **kwargs):
            self.calls.append((args, kwargs))
            pipe.work_frames = None
            pipe.store = None
        return _init


def _normal(call):
    args, kw = call
    d = dict(kw)
    for i, k in enumerate(("work_dir", "plate_pattern", "plate_w", "plate_h", "client")):
        if i < len(args):
            d[k] = args[i]
    d["work_dir"] = str(d["work_dir"])
    d["client"] = id(d["client"])
    d["display"] = id(d.get("display"))
    return d


def test_ui_and_queue_build_the_same_pipeline(tmp_path, monkeypatch):
    api, shot, mf, proj = _rich_project(tmp_path, monkeypatch)
    rec = _Recorder()
    monkeypatch.setattr(ObjectPipeline, "__init__", rec.init())
    run_dir = tmp_path / "run_0001"
    run_dir.mkdir()
    (run_dir / "qa.json").write_text(json.dumps({"colours": {}, "gate_ok": True}), encoding="utf-8")
    monkeypatch.setattr(ObjectPipeline, "run_core", lambda self, name, frames, progress_cb=None: None)
    monkeypatch.setattr(ObjectPipeline, "finalize", lambda self, name, frames, *a, **k: (None, None, run_dir))

    ui = app_module._pipeline()
    out = app_module._queue_worker({"manifest_path": str(mf), "object": "puppet"})
    assert out["qa_status"] == "ok"
    assert len(rec.calls) == 2
    a, b = _normal(rec.calls[0]), _normal(rec.calls[1])
    assert a == b, {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    # the inputs really were there (the test would pass vacuously on a bare manifest)
    assert a["backlit_pattern"] == shot.plate_pattern and a["cleanplate_pattern"] == shot.plate_pattern
    assert a["shutter_deg"] == 123.0 and a["use_mplate"] is True and a["gain_pattern"]
    assert a["frames"] == list(range(shot.first, shot.last + 1)) and a["io_mode"] == "exr"
    # the work area is the UI's business: the queue computes the whole shot
    assert ui.work_frames == list(range(shot.first + 1, shot.first + 5))


def test_from_project_is_what_the_ui_builds(tmp_path, monkeypatch):
    api, shot, mf, proj = _rich_project(tmp_path, monkeypatch)
    rec = _Recorder()
    monkeypatch.setattr(ObjectPipeline, "__init__", rec.init())
    app_module._pipeline()
    ObjectPipeline.from_project(proj, app_module.STATE["comfy_client"], display=app_module._plate_display,
                               ltx_workflow=app_module._ltx_workflow())
    assert _normal(rec.calls[0]) == _normal(rec.calls[1])


def test_queue_respects_the_model_gate(tmp_path, monkeypatch):
    """The UI refuses a stage without its model (409); the queue did not ask – now it stops with the same reason."""
    import engine.stage_models as sm
    api, shot, mf, proj = _rich_project(tmp_path, monkeypatch)
    monkeypatch.setattr(sm, "model_gate", lambda cfg: {"core": {"missing": ["x"], "why": "Brak modelu SAM 3.1: x"}})
    monkeypatch.setattr(ObjectPipeline, "run_core", lambda *a, **k: pytest.fail("run_core must not start"))
    with pytest.raises(RuntimeError, match="Brak modelu SAM 3.1"):
        app_module._queue_worker({"manifest_path": str(mf), "object": "puppet"})


def test_app_has_one_pipeline_builder():
    """Only the engine calls the constructor; app.py goes through from_project (via _pipeline_for)."""
    src = Path(app_module.__file__).read_text(encoding="utf-8")
    assert "ObjectPipeline(" not in src.replace("ObjectPipeline.from_project(", "")
    assert src.count("ObjectPipeline.from_project(") == 1
