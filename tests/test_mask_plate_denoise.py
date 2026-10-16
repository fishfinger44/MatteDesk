# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project switch of the plate-to-mask denoise (SPEC 4b, user decision 2026-10-13).

- project.json mask_plate_denoise: auto (key absent) / on / off; auto = on for EXR / > 8-bit, off for 8-bit sources
  (same detection as engine/comfy_io.io_mode_for); defaults.json mplate.enabled = false = off everywhere.
- off: no MaskPlate, no `mplate/` folder – crops straight from the display (core and final pass).
- the effective flag is in the interval cache hash (ON hash unchanged = old caches valid) and in core.json;
  a core computed with the other value is stale; old core.json without the key = computed with ON.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engine.crop import choose_instance, run_crop_engine, save_crops
from engine.mplate import resolve_mask_plate_denoise
from engine.pipeline import ObjectPipeline
from engine.segment import CorePropagator
from engine.stages import core_denoise, object_stages, stale_intervals
from mock_comfy import MockComfyClient, MockOutputs
from synth import display, make_shot, matanyone_handlers, sam3_handlers


class StatsClient(MockComfyClient):
    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}


def test_resolve_auto_on_off_and_kill_switch():
    r = lambda s, m, en=True: resolve_mask_plate_denoise(s, m, en)
    assert r(None, {"bit_depth": 8, "source_type": "video"})["effective"] is False
    assert r("auto", {"bit_depth": 8, "source_type": "img_seq"})["effective"] is False
    assert r("auto", {"source_type": "video"})["effective"] is False            # MP4 without bit_depth
    assert r("auto", {"bit_depth": 16, "source_type": "exr_seq"})["effective"] is True
    assert r(None, {"bit_depth": 32})["effective"] is True
    assert r(None, {})["effective"] is True                                       # unknown source = EXR (as io_mode_for)
    d = r("on", {"bit_depth": 8})
    assert d["effective"] is True and d["auto_resolves_to"] == "off" and d["source_8bit"]
    d = r("off", {"bit_depth": 16})
    assert d["effective"] is False and d["auto_resolves_to"] == "on"
    assert r("bogus", {"bit_depth": 16})["setting"] == "auto"
    for s in ("auto", "on", "off"):                                               # global kill-switch
        d = r(s, {"bit_depth": 16}, False)
        assert d["effective"] is False and not d["global_enabled"]


# ------------------------------------------------------------------------------------------------------ pipeline
def _pipe(tmp_path, use_mplate, n=8):
    shot = make_shot(tmp_path, n=n)
    outs = MockOutputs(tmp_path / "comfy_out")
    client = StatsClient(handlers={**sam3_handlers(outs), **matanyone_handlers(outputs=outs)})
    client.output_root = outs.root
    frames = list(range(shot.first, shot.last + 1))
    work = tmp_path / "work" / "matte_desk"
    pipe = ObjectPipeline(work, shot.plate_pattern, shot.w, shot.h, client, display=display, sam3_ckpt="sam3.st",
                          use_mplate=use_mplate, frames=frames)
    pipe.store.create("obj", ["felt puppet"])
    start = frames[len(frames) // 2]
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt="felt puppet")
    pipe.store.save_key("obj", start, cands[0] > 0.5, author="auto")
    pipe.store.update("obj", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h)
    save_crops(pipe.store.dir("obj") / "crops.json", res.segments)
    return pipe, frames, work


def test_off_builds_no_mask_plate_core_and_final(tmp_path):
    pipe, frames, work = _pipe(tmp_path, use_mplate=False)
    assert pipe.loader.mplate_dir is None
    pipe.run_core("obj", frames)
    pipe.finalize("obj", frames, edge_mode="matanyone")
    assert not (work / "mplate").exists()
    assert not pipe.loader._mp                                                    # no MaskPlate object either
    assert core_denoise(pipe.store.dir("obj")) is False
    assert json.loads((pipe.store.dir("obj") / "core" / "core.json").read_text())["mask_plate_denoise"] is False


def test_on_uses_mask_plate_and_records_it(tmp_path):
    pipe, frames, work = _pipe(tmp_path, use_mplate=True)
    pipe.run_core("obj", frames)
    assert (work / "mplate").is_dir() and any((work / "mplate").iterdir())
    assert core_denoise(pipe.store.dir("obj")) is True


def test_cache_hash_differs_on_off_and_on_is_unchanged():
    from engine.crop import CropSegment
    seg = CropSegment("s00", 1, 10, 0, 0, [0, 0, 64, 64], [0, 0, 0, 0], 1.0)

    class L:
        pattern = "plate_####.exr"

    def hsh(use):
        p = ObjectPipeline.__new__(ObjectPipeline)
        p.use_mplate, p.loader = use, L()
        prop = CorePropagator(None, [seg], 64, 64, lambda f, s: None, params=p._core_params())
        return prop._hash(None, None, (1, 10)), p._core_params()

    h_on, p_on = hsh(True)
    h_off, p_off = hsh(False)
    assert h_on != h_off
    assert p_on == {"engine": "sam3", "plate": "plate_####.exr"}                  # = params before the switch existed
    legacy = CorePropagator(None, [seg], 64, 64, lambda f, s: None, params={"engine": "sam3", "plate": "plate_####.exr"})
    assert legacy._hash(None, None, (1, 10)) == h_on


def test_changed_setting_marks_core_stale(tmp_path):
    pipe, frames, work = _pipe(tmp_path, use_mplate=False)
    pipe.run_core("obj", frames)
    rng = (frames[0], frames[-1])
    st = lambda mp: {s["id"]: s["state"] for s in object_stages(pipe.store, "obj", work, rng, mplate=mp)["stages"]}
    assert st(False)["core"] == "done" and st(None)["core"] == "done"
    assert st(True)["core"] == "stale"
    info = object_stages(pipe.store, "obj", work, rng, mplate=True)
    assert "odszumianie" in next(s["why"] for s in info["stages"] if s["id"] == "core")
    assert stale_intervals(pipe.store, "obj", rng, mplate=True) == [list(rng)]
    assert stale_intervals(pipe.store, "obj", rng, mplate=False) == []
    # an old core.json (before the switch) has no key: it was computed with the denoise ON
    cj = pipe.store.dir("obj") / "core" / "core.json"
    d = json.loads(cj.read_text())
    d.pop("mask_plate_denoise")
    cj.write_text(json.dumps(d))
    assert core_denoise(pipe.store.dir("obj")) is True
    assert st(True)["core"] == "done" and st(False)["core"] == "stale"


# ------------------------------------------------------------------------------------------------------ API + CLI
def test_api_project_json_auto_8bit_kill_switch_and_stale(tmp_path, monkeypatch):
    import app as app_module
    from test_ui_stages_api import _open
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_bytes()
    pj = tmp_path / "work" / "matte_desk" / "project.json"
    assert "mask_plate_denoise" not in json.loads(pj.read_text(encoding="utf-8"))   # old project: no key -> auto
    d = api.get("/api/project/mask_plate_denoise").json()
    assert d["setting"] == "auto" and d["effective"] is True                         # synthetic shot = EXR
    assert api.get("/api/project/stages").json()["mask_plate_denoise"]["effective"] is True
    # an object with a core computed with the denoise ON (old core.json, no key)
    od = tmp_path / "work" / "matte_desk" / "objects" / "puppet" / "core"
    od.mkdir(parents=True, exist_ok=True)
    (od / "core.json").write_text(json.dumps({"keys_sig": []}))
    d = api.put("/api/project/mask_plate_denoise", json={"setting": "off"}).json()
    assert d["setting"] == "off" and d["effective"] is False and d["changed"] and d["stale_objects"] == ["puppet"]
    assert json.loads(pj.read_text(encoding="utf-8"))["mask_plate_denoise"] == "off"
    assert app_module._pipeline().loader.mplate_dir is None and not app_module._pipeline().use_mplate
    assert api.put("/api/project/mask_plate_denoise", json={"setting": "maybe"}).status_code == 400
    # 8-bit source: auto -> off, the key leaves project.json
    monkeypatch.setitem(app_module.STATE["manifest"], "bit_depth", 8)
    d = api.put("/api/project/mask_plate_denoise", json={"setting": "auto"}).json()
    assert d["setting"] == "auto" and d["effective"] is False and d["auto_resolves_to"] == "off" and not d["changed"]
    assert "mask_plate_denoise" not in json.loads(pj.read_text(encoding="utf-8"))
    d = api.put("/api/project/mask_plate_denoise", json={"setting": "on"}).json()
    assert d["effective"] is True and d["stale_objects"] == [] and app_module._pipeline().use_mplate
    # global kill-switch (defaults.json mplate.enabled = false) wins over "on"
    monkeypatch.setattr(app_module, "_mplate_global_enabled", lambda: False)
    d = api.get("/api/project/mask_plate_denoise").json()
    assert d["setting"] == "on" and d["effective"] is False and not d["global_enabled"]
    assert app_module._pipeline().loader.mplate_dir is None
    assert mf.read_bytes() == before                                                 # the shot manifest is never written


def test_md_cli_denoise_and_status():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import md
    calls, lines = [], []
    info = {"setting": "auto", "effective": False, "reason": "źródło 8-bit (MP4 / PNG / JPG)", "stale_objects": ["a"]}

    class H:
        def request(self, method, path, body=None, timeout=None):
            calls.append((method, path, body))
            if path == "/api/project/current":
                return 200, {"loaded": True, "manifest": {"shot": "S"}}
            if path == "/api/objects":
                return 200, {"objects": []}
            if path == "/api/jobs":
                return 200, {"jobs": []}
            return 200, info

    assert md.main(["denoise", "off"], http=H(), out=lines.append) == 0
    assert ("PUT", "/api/project/mask_plate_denoise", {"setting": "off"}) in calls
    assert md.main(["denoise"], http=H(), out=lines.append) == 0
    assert calls[-1][:2] == ("GET", "/api/project/mask_plate_denoise")
    assert md.main(["status"], http=H(), out=lines.append) == 0
    assert any("Odszumianie plate'u do maskowania: auto → wył." in s for s in lines)
    with pytest.raises(SystemExit):
        md.main(["denoise", "maybe"], http=H(), out=lines.append)
