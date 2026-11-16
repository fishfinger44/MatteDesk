# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/color_profile.py + api/color_profile.py (user decision 2026-11-03): the plate colour profile of a shot is a
lin ↔ log curve id proposed from the source type / manifest colorspace / EXR header (ACES flag, ARRI / Sony / RED
metadata) with a reason, overridable, stored ONLY in project.json (`color_profile`: auto | id) – the manifest is never
written – and read back by MergeMatte / the .nk export as the curve `project` resolves to."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine import color_profile as cp
from engine import log_curves as lc


def _m(**kw):
    m = {"colorspace": "ACES2065-1", "source_type": "exr_seq", "bit_depth": 16, "plates": {"beauty": "D:/x/sh_####.exr"}}
    m.update(kw)
    return m


@pytest.mark.parametrize("manifest, header, cid, why", [
    (_m(), {"acesImageContainerFlag": 1}, "acescct", "ACES"),
    (_m(), {}, "acescct", "colorspace"),                                     # ACES colorspace, nothing else
    (_m(), {"cameraModel": b"ARRI ALEXA Mini"}, "logc3", "ARRI"),
    (_m(), {"arri.cameraModel": "ALEXA 35", "arri.colorSpace": "LogC4"}, "logc4", "LogC4"),
    (_m(), {"sony.camera": "VENICE 2"}, "slog3", "Sony"),
    (_m(), {"red.camera": "RED V-RAPTOR", "redcode": "8:1"}, "cineon", "RED"),
    (_m(colorspace="ARRI LogC3"), {}, "logc3", "manifestu"),
    (_m(colorspace="Sony S-Log3"), {}, "slog3", "manifestu"),
    (_m(colorspace="Cineon"), {}, "cineon", "manifestu"),
    (_m(colorspace="sRGB", source_type="video", bit_depth=8, plates={"beauty": "F:/x/v_####.png"}), {}, "video", "sRGB"),
    (_m(colorspace="", source_type="img_seq", bit_depth=8, plates={"beauty": "F:/x/v_####.png"}), {}, "video", "8-bit"),
    (_m(colorspace="", source_type="exr_seq", bit_depth=16), {}, "cineon", "brak wskazówek"),
    (_m(colorspace="", source_type="unknown", bit_depth=None, plates={"beauty": "F:/x/v_####.dpx"}), {}, "cineon", "brak"),
])
def test_proposal_from_source_manifest_and_header(manifest, header, cid, why):
    p = cp.proposal(manifest, header)
    assert p["id"] == cid and why in p["reason"], p
    assert p["id"] in lc.LOG_CURVES


def test_camera_metadata_beats_the_manifest_colorspace():
    """Ingest writes ACES2065-1 into every EXR manifest, so the header's camera metadata is the stronger evidence."""
    assert cp.proposal(_m(colorspace="ACES2065-1"), {"cameraModel": "ARRI ALEXA LF"})["id"] == "logc3"
    assert cp.proposal(_m(colorspace="ACES2065-1"), {"acesImageContainerFlag": 1, "cameraModel": "SONY VENICE"})["id"] == "slog3"


def test_resolve_auto_and_override():
    r = cp.resolve(None, _m(), {})
    assert r == {"setting": "auto", "id": "acescct", "label": "ACEScct", "reason": r["auto"]["reason"], "auto": r["auto"]}
    r = cp.resolve("logc4", _m(), {})
    assert r["setting"] == "logc4" and r["id"] == "logc4" and r["label"] == "ARRI LogC4" and "ręcznie" in r["reason"]
    assert r["auto"]["id"] == "acescct"
    assert cp.resolve("nonsense", _m(), {})["setting"] == "auto"
    assert cp.validate("AUTO") == "auto" and cp.validate("slog3") == "slog3"
    with pytest.raises(ValueError):
        cp.validate("none")
    assert cp.CHOICES == ["auto"] + lc.CURVE_IDS[1:]


def test_sample_file_and_header(tmp_path):
    (tmp_path / "sh_0003.png").write_bytes(b"x")
    (tmp_path / "sh_0001.png").write_bytes(b"x")
    pat = str(tmp_path / "sh_####.png").replace("\\", "/")
    assert cp.sample_file({"plates": {"beauty": pat}}) == str(tmp_path / "sh_0001.png")
    assert cp.sample_file({"plates": {"beauty": str(tmp_path / "none_####.png")}}) is None
    assert cp.sample_file({"plates": {}}) is None
    assert cp.read_header(str(tmp_path / "sh_0001.png")) == {}   # not an EXR
    assert cp.header_hints({"a": b"ARRI", "b": 1, "c": object()}) == {"a": "arri", "b": "1"}


# ---------------------------------------------------------------- api/color_profile.py
class _Proj:
    def __init__(self, root: Path):
        self.data = {}
        self.root = root
        self.saved = 0

    def save(self):
        self.saved += 1
        (self.root / "project.json").write_text(json.dumps(self.data), "utf-8")


@pytest.fixture()
def client(tmp_path):
    from api.color_profile import make
    manifest = _m(colorspace="sRGB", source_type="video", bit_depth=8, plates={"beauty": str(tmp_path / "v_####.png")})
    mpath = tmp_path / "shot_manifest_x.json"
    mpath.write_text(json.dumps(manifest), "utf-8")
    STATE = {"shot_project": _Proj(tmp_path), "manifest": manifest, "work_dir": str(tmp_path), "manifest_path": str(mpath)}
    app = FastAPI()
    router, exports = make(STATE=STATE)
    app.include_router(router)
    c = TestClient(app)
    c.STATE, c.exports, c.mpath, c.manifest = STATE, exports, mpath, manifest
    return c


def test_get_put_round_trip_project_json_only(client):
    r = client.get("/api/project/color_profile").json()
    assert r["setting"] == "auto" and r["effective"]["id"] == "video" and "sRGB" in r["effective"]["reason"]
    assert [o["id"] for o in r["options"]] == cp.CHOICES and r["options"][0]["label"].startswith("auto (→ video (sRGB)")
    assert client.exports["_project_curve"]() == {"id": "video", "label": "video (sRGB)", "reason": r["effective"]["reason"]}
    r = client.put("/api/project/color_profile", json={"color_profile": "logc3"}).json()
    assert r["setting"] == "logc3" and r["effective"]["id"] == "logc3"
    proj = client.STATE["shot_project"]
    assert proj.data["color_profile"] == "logc3" and proj.saved == 1
    assert json.loads(client.mpath.read_text("utf-8")) == client.manifest        # the manifest is never written
    assert client.exports["_project_curve"]()["id"] == "logc3"
    assert client.put("/api/project/color_profile", json={"color_profile": "auto"}).json()["effective"]["id"] == "video"
    assert "color_profile" not in proj.data or proj.data["color_profile"] == "auto"
    assert client.put("/api/project/color_profile", json={"color_profile": "none"}).status_code == 400


def test_no_project_is_400_and_curve_none(client):
    client.STATE["shot_project"] = None
    assert client.get("/api/project/color_profile").status_code == 400
    assert client.put("/api/project/color_profile", json={"color_profile": "auto"}).status_code == 400
    assert client.exports["_project_curve"]() is None
