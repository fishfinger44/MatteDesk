# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""align_v2 step 4 (2026-11-03): a stage-3 / stage-5 run records which backlit / clear plate it read (`aux_plates`
in run.json: aligned or raw + the alignment report's signature); after stage 0 writes or re-measures the aligned
plate the stage shows ↻ „liczone na niewyrównanym / starszym wyrównaniu” – and nothing is recomputed by itself.
Older runs without the record stay current."""

import json
import time
from pathlib import Path

import numpy as np

from engine.align_state import aux_plate_state, current_plates, current_state, stale_reason
from engine.objects import ObjectStore
from engine.stages import keys_signature, mask_sig, settings_sig, stage_run


def _project(tmp_path, plates):
    shot = tmp_path / "shot"
    work = shot / "matte_desk"
    work.mkdir(parents=True)
    (shot / "shot_manifest_SYN.json").write_text(json.dumps({"shot": "SYN", "first": 1, "last": 3, "plate_w": 64,
                                                             "plate_h": 48, "plates": plates}), encoding="utf-8")
    (work / "project.json").write_text(json.dumps({"manifest_path": str(shot / "shot_manifest_SYN.json"),
                                                   "plates_added": {}, "plates_hidden": []}), encoding="utf-8")
    return work


def _align_report(work, plate, tag):
    (work / "align").mkdir(exist_ok=True)
    (work / "align" / f"Align_{plate}.json").write_text(json.dumps({"plate_name": plate, "tag": tag}), encoding="utf-8")


def test_aux_plate_state_and_stale_reason(tmp_path):
    work = _project(tmp_path, {"beauty": "b_####.exr", "backlit": "x2_####.exr", "cleanplate": "cp_####.exr"})
    st = current_state(work)
    assert st == {"backlit": {"pattern": "x2_####.exr", "aligned": False, "align_sig": None},
                  "cleanplate": {"pattern": "cp_####.exr", "aligned": False, "align_sig": None}}
    # stage 0 writes the aligned backlit (project.json) after a measurement
    _align_report(work, "backlit", 1)
    pj = json.loads((work / "project.json").read_text(encoding="utf-8"))
    pj["plates_added"]["backlit_align"] = "x2_align/x2_####.exr"
    (work / "project.json").write_text(json.dumps(pj), encoding="utf-8")
    now = current_state(work)
    assert now["backlit"]["aligned"] and now["backlit"]["align_sig"] and now["backlit"]["pattern"].endswith("x2_align/x2_####.exr")
    assert current_plates(work)["backlit_align"]
    assert "niewyrównanym plate'cie backlit" in stale_reason(st, now, ("backlit",))
    assert stale_reason(st, now, ("cleanplate",)) == ""                      # the stage did not use the backlit
    assert stale_reason(None, now, ("backlit",)) == "" and stale_reason({}, now, ("backlit",)) == ""
    # a run made on the aligned plate, then stage 0 measures again -> older alignment
    rec = now
    _align_report(work, "backlit", 2)
    now2 = current_state(work)
    assert now2["backlit"]["align_sig"] != rec["backlit"]["align_sig"]
    assert "starszym wyrównaniu plate'u backlit" in stale_reason(rec, now2, ("backlit",), " – policz")
    assert stale_reason(now2, now2, ("backlit", "cleanplate")) == ""
    # aux_plate_state from a pipeline's own patterns
    assert aux_plate_state(work, {"backlit": "x2_align/x2_####.exr", "cleanplate": None}) == {
        "backlit": {"pattern": "x2_align/x2_####.exr", "aligned": False, "align_sig": None}}


def test_stage_runs_flag_an_alignment_written_after_them(tmp_path):
    work = _project(tmp_path, {"beauty": "b_####.exr", "backlit": "x2_####.exr", "cleanplate": "cp_####.exr"})
    store = ObjectStore(work)
    store.create("obj", ["puppet"])
    m = np.zeros((48, 64), bool); m[10:20, 10:20] = True
    store.save_key("obj", 1, m, author="user")
    cd = store.dir("obj") / "core"; cd.mkdir()
    (cd / "core.json").write_text(json.dumps({"keys_sig": keys_signature(store, "obj")}))
    obj = store.get("obj")
    rec = current_state(work)
    time.sleep(0.01)
    rd = store.dir("obj") / "runs" / "v001"; rd.mkdir(parents=True)
    (rd / "run.json").write_text(json.dumps({"stage": "matanyone", "core_fix_sig": mask_sig(store, "obj"),
                                             "settings_sig": settings_sig(obj), "aux_plates": rec}), encoding="utf-8")
    rd5 = store.dir("obj") / "runs" / "v002"; rd5.mkdir(parents=True)
    (rd5 / "run.json").write_text(json.dumps({"stage": "vitmatte", "base": "matanyone", "applied_from": "v001",
                                              "matte_sig": "x", "aux_plates": rec}), encoding="utf-8")
    store.update("obj", tool={"stage_runs": {"matanyone": "v001", "vitmatte@matanyone": "v002"}})
    from engine.stages import matte_sig
    (rd5 / "run.json").write_text(json.dumps({"stage": "vitmatte", "base": "matanyone", "applied_from": "v001",
                                              "matte_sig": matte_sig(store.dir("obj")), "aux_plates": rec}), encoding="utf-8")
    assert stage_run(store, "obj", "matanyone") == ("v001", True, "")
    assert stage_run(store, "obj", "vitmatte", "matanyone") == ("v002", True, "")
    # stage 0: the clear plate gets aligned -> stage 5 (matte zones read it) is behind, stage 3 (backlit) is not
    _align_report(work, "cleanplate", 1)
    pj = json.loads((work / "project.json").read_text(encoding="utf-8"))
    pj["plates_added"]["cleanplate_align"] = "cp_align/cp_####.exr"
    (work / "project.json").write_text(json.dumps(pj), encoding="utf-8")
    run, ok, why = stage_run(store, "obj", "vitmatte", "matanyone")
    assert run == "v002" and not ok and "niewyrównanym plate'cie clear plate" in why and "Zastosuj" in why
    assert stage_run(store, "obj", "matanyone") == ("v001", True, "")
    # ... and the backlit too -> stage 3 behind; a re-measured backlit later = older alignment
    _align_report(work, "backlit", 1)
    pj["plates_added"]["backlit_align"] = "x2_align/x2_####.exr"
    (work / "project.json").write_text(json.dumps(pj), encoding="utf-8")
    run, ok, why = stage_run(store, "obj", "matanyone")
    assert run == "v001" and not ok and "niewyrównanym plate'cie backlit" in why and "MatAnyone2" in why
    rec2 = current_state(work)
    (rd / "run.json").write_text(json.dumps({"stage": "matanyone", "core_fix_sig": mask_sig(store, "obj"),
                                             "settings_sig": settings_sig(obj), "aux_plates": rec2}), encoding="utf-8")
    assert stage_run(store, "obj", "matanyone")[1]
    _align_report(work, "backlit", 2)
    run, ok, why = stage_run(store, "obj", "matanyone")
    assert not ok and "starszym wyrównaniu plate'u backlit" in why
    # an old run without the record is never flagged
    (rd / "run.json").write_text(json.dumps({"stage": "matanyone", "core_fix_sig": mask_sig(store, "obj"),
                                             "settings_sig": settings_sig(obj)}), encoding="utf-8")
    assert stage_run(store, "obj", "matanyone")[1]


def test_finalize_records_aux_plates_and_stage_bar_flags_a_new_alignment(tmp_path):
    """End to end on the synthetic pipeline: stage 3 writes `aux_plates` into run.json; the stage bar shows ↻ after a
    backlit alignment appears in the project (nothing recomputed)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_e5_acceptance import _setup
    from engine.stages import object_stages
    shot, pipe, frames, start = _setup(tmp_path, n=6)
    assert pipe.aux_plates == {}                                   # no backlit / clear plate: nothing recorded
    pipe.aux_plates = {"backlit": {"pattern": "x2_####.exr", "aligned": False, "align_sig": None}}
    full, rep, run_dir = pipe.finalize("obj", frames, stage="matanyone")
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "matanyone" and rj["aux_plates"] == pipe.aux_plates
    work = pipe.work
    assert stage_run(pipe.store, "obj", "matanyone")[1]
    # stage 0 aligns the backlit (project.json + report) -> stage 3 ↻ with the reason, result still there
    (work / "align").mkdir(exist_ok=True)
    (work / "align" / "Align_backlit.json").write_text("{}", encoding="utf-8")
    pj = work / "project.json"
    d = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {}
    d.setdefault("plates_added", {})["backlit_align"] = "x2_align/x2_####.exr"
    d.setdefault("plates_hidden", [])
    pj.write_text(json.dumps(d), encoding="utf-8")
    run, ok, why = stage_run(pipe.store, "obj", "matanyone")
    assert run == run_dir.name and not ok and "niewyrównanym plate'cie backlit" in why
    st = {s["id"]: s for s in object_stages(pipe.store, "obj", work)["stages"]}
    assert st["matanyone"]["state"] == "stale" and "niewyrównanym" in (st["matanyone"].get("why") or "")
