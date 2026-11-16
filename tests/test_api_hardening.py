"""Audit 2026-11-02 #7 / #8: the local API is reachable only from its own origins, and names that build file paths
(zone id, plate role) are plain names."""
import pytest
from fastapi.middleware.cors import CORSMiddleware

import app as app_module
from engine.zones import zone_file


def test_cors_is_not_wildcard():
    cors = [m for m in app_module.app.user_middleware if m.cls is CORSMiddleware]
    assert cors, "CORS middleware missing"
    origins = cors[0].kwargs["allow_origins"]
    assert "*" not in origins
    assert all(o.startswith("http://127.0.0.1:") or o.startswith("http://localhost:") for o in origins)


@pytest.mark.parametrize("bad", ["..", "../x", "..\\..\\project", "a/b", "", "z 1", "x" * 65])
def test_zone_file_rejects_paths(tmp_path, bad):
    with pytest.raises(ValueError):
        zone_file(tmp_path, bad)


def test_zone_file_accepts_ids(tmp_path):
    assert zone_file(tmp_path, "z_01") == tmp_path / "zones" / "z_01.json"
    assert zone_file(tmp_path, "new").name == "new.json"


@pytest.mark.parametrize("bad", ["..", "../x", "..\\beauty", "a/b", ""])
def test_plate_name_rejects_paths(bad):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        app_module._plate_name_or_400(bad)
    assert e.value.status_code == 400


def test_plate_name_accepts_roles():
    for ok in ("beauty", "backlit", "clear_plate", "beauty_align", "x-1"):
        assert app_module._plate_name_or_400(ok) == ok
