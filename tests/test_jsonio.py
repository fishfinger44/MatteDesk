"""Audit 2026-11-02 #4 / #21: one atomic JSON writer (a cut write leaves the old file whole) and one rule for a corrupt
zone file: StateFileError with its name, the file stays for repair – whichever reader meets it first."""
import asyncio
import json
import re
import threading
from pathlib import Path

import pytest

from engine import jsonio
from engine.jsonio import StateFileError, read_state, write_json
from engine.project import StateFileError as ProjectStateFileError
from engine.stages import core_fix_sig, matte_sig, zone_dicts
from engine.zones import Zone, load_zones, save_zone

REPO = Path(__file__).resolve().parent.parent


def test_bytes_are_those_of_json_dumps(tmp_path):
    d = {"a": 1, "ż": "ą", "n": {"x": [1, 2]}}
    p = tmp_path / "x.json"
    write_json(p, d, indent=2, ensure_ascii=False)
    assert p.read_bytes() == json.dumps(d, indent=2, ensure_ascii=False).encode("utf-8")
    write_json(p, d, indent=1)
    assert p.read_bytes() == json.dumps(d, indent=1).encode("utf-8")


def test_unserialisable_data_keeps_the_old_file_and_no_tmp(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"old": 1})
    with pytest.raises(TypeError):
        write_json(p, {"new": object()})
    assert json.loads(p.read_text(encoding="utf-8")) == {"old": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_cut_write_keeps_the_old_file(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    write_json(p, {"old": 1})

    def boom(*a, **k):
        raise OSError("disk full / power cut")
    monkeypatch.setattr(jsonio.os, "replace", boom)
    with pytest.raises(OSError):
        write_json(p, {"new": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"old": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_replace_is_retried_when_windows_holds_the_file(tmp_path, monkeypatch):
    p = tmp_path / "x.json"
    real, calls = jsonio.os.replace, []

    def flaky(a, b):
        calls.append(a)
        if len(calls) < 3:
            raise PermissionError("WinError 5")
        real(a, b)
    monkeypatch.setattr(jsonio.os, "replace", flaky)
    monkeypatch.setattr(jsonio.time, "sleep", lambda s: None)
    write_json(p, {"v": 1})
    assert len(calls) == 3 and json.loads(p.read_text(encoding="utf-8")) == {"v": 1}


def test_threads_writing_one_file_leave_it_valid(tmp_path):
    p = tmp_path / "x.json"
    errors = []

    def work(i):
        try:
            for _ in range(20):
                write_json(p, {"i": i, "pad": "x" * 5000})
        except Exception as e:   # noqa: BLE001 - reported below
            errors.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert json.loads(p.read_text(encoding="utf-8"))["pad"] == "x" * 5000
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_read_state_missing_corrupt_not_an_object(tmp_path):
    p = tmp_path / "s.json"
    assert read_state(p) is None
    p.write_text('{"a": [', encoding="utf-8")
    with pytest.raises(StateFileError) as e:
        read_state(p)
    assert "s.json" in str(e.value)
    p.write_text("[1]", encoding="utf-8")
    with pytest.raises(StateFileError):
        read_state(p)
    assert StateFileError is ProjectStateFileError      # the one exception the API turns into 409


def _obj_dir(tmp_path):
    od = tmp_path / "obj"
    (od / "zones").mkdir(parents=True)
    (od / "object.json").write_text(json.dumps({"name": "obj", "core_stabilize": True}), encoding="utf-8")
    return od


class _Store:
    def __init__(self, od):
        self.od = od

    def dir(self, name):
        return self.od

    def get(self, name):
        return json.loads((self.od / "object.json").read_text(encoding="utf-8"))


def test_corrupt_zone_one_rule_everywhere(tmp_path):
    """#21: load_zones failed while zone_dicts skipped the file – the bar said current, every run ended in an exception."""
    od = _obj_dir(tmp_path)
    save_zone(od, Zone(zone_id="z_01", mode="add", criterion="matte", frame=1, params={"x": 1}))
    bad = od / "zones" / "z_02.json"
    bad.write_text('{"zone_id": "z_02", "crit', encoding="utf-8")
    for call in (lambda: load_zones(od), lambda: zone_dicts(od, ("matte",)), lambda: zone_dicts(od, ("brush",)),
                 lambda: matte_sig(od), lambda: core_fix_sig(_Store(od), "obj")):
        with pytest.raises(StateFileError) as e:
            call()
        assert "z_02.json" in str(e.value)
    assert bad.read_text(encoding="utf-8").endswith('"crit')      # kept for repair


def test_zone_with_foreign_fields_is_a_state_error_too(tmp_path):
    od = _obj_dir(tmp_path)
    (od / "zones" / "z_01.json").write_text(json.dumps({"zone_id": "z_01", "no_such_field": 1}), encoding="utf-8")
    with pytest.raises(StateFileError) as e:
        load_zones(od)
    assert "z_01.json" in str(e.value)


def test_no_zones_folder_is_empty(tmp_path):
    od = tmp_path / "obj"
    od.mkdir()
    assert load_zones(od) == [] and zone_dicts(od, ("matte",)) == []


class _Req:
    method = "GET"

    class url:
        path = "/api/object/x/stages"


def test_api_corrupt_zone_is_409_with_the_name(tmp_path):
    import app as app_module
    od = _obj_dir(tmp_path)
    (od / "zones" / "z_01.json").write_text("{", encoding="utf-8")
    with pytest.raises(StateFileError) as e:
        load_zones(od)
    r = asyncio.run(app_module._unhandled_error_as_json(_Req(), e.value))
    assert r.status_code == 409 and b"z_01.json" in r.body


@pytest.mark.parametrize("rel", ["engine/pipeline.py", "engine/refine.py", "engine/zones.py", "engine/groups.py"])
def test_state_json_goes_through_the_writer(rel):
    """No `x.write_text(json.dumps(...))` left where a version's / zone's state is written."""
    src = (REPO / rel).read_text(encoding="utf-8")
    assert not re.search(r"\.write_text\(\s*json\.dumps\(", src), rel
