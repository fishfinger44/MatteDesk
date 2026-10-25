# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Work area (zakres pracy, user request 2026-10-15): project.json work_area, „▶▶ Przelicz nieaktualne” / „Przelicz finalnie” /
export respect it; a core / final result computed only on the work area never passes for the whole shot (409 with the
missing frame intervals); SAM3 needs at least one key inside the work area."""

import json

import cv2
import pytest

from engine.workarea import clamp_range, expand_to_keys, intervals_from_frames, missing, union
from synth import display
from test_e3_acceptance import _setup, fp


def test_interval_helpers():
    assert intervals_from_frames([5, 1, 2, 3, 7]) == [[1, 3], [5, 5], [7, 7]]
    assert union([[1, 3]], [[4, 6], [10, 12]]) == [[1, 6], [10, 12]]
    assert missing((1, 20), [[5, 8], [12, 30]]) == [[1, 4], [9, 11]]
    assert missing((5, 8), [[1, 30]]) == []
    assert missing((1, 5), []) == [[1, 5]]
    assert clamp_range([80, 20], (1, 100)) == [20, 80]
    assert clamp_range([0, 200], (1, 100)) is None          # whole shot = no work area
    with pytest.raises(ValueError):
        clamp_range([150, 200], (1, 100))
    # extended to the nearest key on each side; no key on a side -> the edge of the work area
    assert expand_to_keys([20, 80], [10, 50, 90], (1, 100)) == [10, 90]
    assert expand_to_keys([20, 80], [50], (1, 100)) == [20, 80]
    assert expand_to_keys([20, 80], [20, 80, 95], (1, 100)) == [20, 80]


def _open(tmp_path, monkeypatch, n=24):
    from fastapi.testclient import TestClient
    import app as app_module
    shot, pipe, client, frames = _setup(tmp_path, n=n)
    mf = tmp_path / "work" / "shot_manifest_SYN.json"
    mf.write_text(json.dumps({"shot": "SYN", "first": shot.first, "last": shot.last, "plate_w": shot.w,
                              "plate_h": shot.h, "work": str(tmp_path / "work"),
                              "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    assert api.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", client)
    monkeypatch.setattr(app_module, "_plate_display", display)
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": display(
        cv2.imread(fp(shot.plate_pattern, f), cv2.IMREAD_UNCHANGED)))
    return api, shot, mf


def test_api_work_area_stored_in_project_not_manifest(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch)
    before = mf.read_text(encoding="utf-8")
    d = api.get("/api/project/work_area").json()
    assert d["range"] is None and d["effective"] == [shot.first, shot.last]
    d = api.put("/api/project/work_area", json={"range": [shot.first + 12, shot.first + 4]}).json()
    assert d["range"] == [shot.first + 4, shot.first + 12] and not d["is_whole"]
    pj = json.loads((tmp_path / "work" / "matte_desk" / "project.json").read_text(encoding="utf-8"))
    assert pj["work_area"] == [shot.first + 4, shot.first + 12]
    assert mf.read_text(encoding="utf-8") == before                      # manifest read-only
    assert api.put("/api/project/work_area", json={"range": [shot.last + 5, shot.last + 9]}).status_code == 400
    d = api.put("/api/project/work_area", json={"range": None}).json()
    assert d["range"] is None
    assert "work_area" not in json.loads((tmp_path / "work" / "matte_desk" / "project.json").read_text(encoding="utf-8"))


def test_api_work_area_core_final_export(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch, n=24)
    keys = sorted(int(k["frame"]) for k in api.get("/api/object/puppet/keys").json()["keys"])
    k0 = keys[0]
    # a work area without any key -> 409 no_key_in_range (SAM3 would track from a key outside)
    empty = [f for f in range(shot.first, shot.last + 1) if f not in keys]
    lo = next(f for f in empty if f + 2 in empty and f + 1 in empty)
    api.put("/api/project/work_area", json={"range": [lo, lo + 2]})
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_key_in_range", r.text
    # the dialog offers the work area extended to the object's nearest key (the work area is shared by all objects)
    d = r.json()["detail"]
    assert d["keys"] == keys and f"F{k0}" in d["detail"]
    near = min(keys, key=lambda k: min(abs(k - lo), abs(k - lo - 2)))
    assert d["suggest"] == [min(lo, near), max(lo + 2, near)]
    api.put("/api/project/work_area", json={"range": d["suggest"]})
    assert api.post("/api/object/puppet/compute_all").status_code == 200

    # work area around the key: core only there (extended to the nearest keys), not the whole shot
    wa = [max(shot.first, k0 - 2), min(shot.last, k0 + 3)]
    api.put("/api/project/work_area", json={"range": wa})
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["whole_shot"] is False and d["frames"][0] <= wa[0] and d["frames"][1] >= wa[1]
    st = api.get("/api/object/puppet/stages").json()
    core = next(s for s in st["stages"] if s["id"] == "core")
    assert core["state"] == "done" and core["scope"] == "range" and core["missing_shot"]   # done only in the range

    fz = api.post("/api/object/puppet/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    assert fz.json()["frames"] == wa and fz.json()["whole_shot"] is False

    # export "całość": no final result on the whole shot -> 409 with the missing intervals, nothing written
    exp_dir = tmp_path / "work" / "matte_desk" / "export" / "mattes" / "puppet"
    r = api.post("/api/export", json={"scope": "all", "multichannel": False})
    assert r.status_code == 409, r.text
    det = r.json()["detail"]
    assert det["code"] == "range_missing" and det["missing"]["puppet"] == missing((shot.first, shot.last), [wa])
    assert not exp_dir.exists()
    # export "work area": exactly the work-area frames
    r = api.post("/api/export", json={"scope": "work_area", "multichannel": False})
    assert r.status_code == 200, r.text
    assert r.json()["frames"] == wa
    written = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in exp_dir.iterdir())
    assert written == list(range(wa[0], wa[1] + 1))

    # whole shot as the work area -> the core is no longer "done" for it; finalize says what is missing
    api.put("/api/project/work_area", json={"range": None})
    st = api.get("/api/object/puppet/stages").json()
    assert next(s for s in st["stages"] if s["id"] == "core")["state"] == "stale"
    assert st["stale_intervals"]
    r = api.post("/api/object/puppet/finalize", json={"edge_mode": "core"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "range_missing", r.text
    # „▶▶ Przelicz nieaktualne” on the whole shot: the core covers everything (a single key -> both intervals are open-ended,
    # their span changed with the range, so they are recomputed; intervals between two keys would come from the cache)
    d = api.post("/api/object/puppet/compute_all").json()
    assert d["whole_shot"] and d["frames"] == [shot.first, shot.last]
    core = next(s for s in api.get("/api/object/puppet/stages").json()["stages"] if s["id"] == "core")
    assert core["state"] == "done" and core["scope"] == "all"
    assert api.post("/api/object/puppet/finalize", json={"edge_mode": "core"}).status_code == 200
    r = api.post("/api/export", json={"scope": "all", "multichannel": False})
    assert r.status_code == 200, r.text and r.json()["frames"] == [shot.first, shot.last]


def test_core_on_work_area_reuses_cache_between_keys(tmp_path):
    """A work area inside the interval between two keys is extended to both keys: the same interval (and cache hash)
    as on the whole shot -> taken from the cache, the result bit-identical; the coverage says what was computed."""
    import numpy as np
    from engine.workarea import core_coverage
    shot, pipe, client, frames = _setup(tmp_path, n=20)
    start = shot.first + 10
    full = pipe.run_core("puppet", frames)
    k2 = shot.first + 3
    pipe.store.save_key("puppet", k2, full.core[k2], author="auto")
    full = pipe.run_core("puppet", frames)
    assert core_coverage(pipe.store.dir("puppet")) == [[shot.first, shot.last]]
    wa = [k2 + 2, start - 2]
    a, b = expand_to_keys(wa, pipe.store.keys("puppet"), (shot.first, shot.last))
    assert (a, b) == (k2, start)
    part = pipe.run_core("puppet", list(range(a, b + 1)))
    assert part.recomputed_intervals == [] and part.cached_intervals == [(k2, start)]
    assert all(np.array_equal(part.core[f], full.core[f]) for f in range(a, b + 1))
    # same inputs -> the coverage keeps the whole shot
    assert core_coverage(pipe.store.dir("puppet")) == [[shot.first, shot.last]]
