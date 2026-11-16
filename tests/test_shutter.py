"""Shutter angle measured on the plate (engine/shutter.py, user decision 2026-11-08): synthetic camera pans with a
known exposure, the resolve order (measured > manifest > default) and the pipeline using it."""
import cv2
import numpy as np
import pytest

from engine import shutter as sh


def _texture(h=540, w=1600, seed=3):
    rng = np.random.default_rng(seed)
    t = np.zeros((h, w), np.float32)
    for sg, a in ((1.0, 40), (2.5, 50), (6.0, 60)):           # multi-scale texture, isotropic
        t += cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sg) * a
    for _ in range(120):                                       # some hard edges in all directions
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(4, 30))
        cv2.circle(t, (x, y), r, float(rng.normal(0, 60)), -1)
    return np.clip(t + 128, 0, 255)


def _pan(shutter_deg, speeds, W=960, H=540):
    """Frames of a horizontal pan over the texture: frame f centred at x_f, exposure = shutter/360 of its speed."""
    tex = _texture()
    xs = np.cumsum([0.0] + list(speeds[:-1])) + 40.0
    frames = {}
    for f, (x, v) in enumerate(zip(xs, speeds), start=1):
        n = 32
        acc = np.zeros((H, W), np.float32)
        for i in range(n):
            o = x + ((i + 0.5) / n - 0.5) * v * shutter_deg / 360.0
            M = np.float32([[1, 0, -o], [0, 1, 0]])
            acc += cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames[f] = np.clip(acc / n, 0, 255).astype(np.uint8)
    return frames


SPEEDS = [3, 4, 6, 9, 13, 16, 12, 8, 5, 3, 4, 7, 11, 15, 10, 6]


@pytest.mark.parametrize("deg", [90.0, 180.0, 270.0])
def test_measure_recovers_the_pan_exposure(deg):
    fr = _pan(deg, SPEEDS)
    r = sh.measure(lambda f: fr.get(f), sorted(fr))
    assert r["confident"], r
    assert r["pairs"] >= sh.MIN_PAIRS
    assert abs(r["deg"] - deg) <= 45.0, r


def test_static_shot_is_not_measurable():
    fr = _pan(180.0, [0.0] * 8)
    r = sh.measure(lambda f: fr.get(f), sorted(fr))
    assert r["deg"] is None and not r["confident"] and r["why"]


def test_excluded_object_pixels_are_ignored():
    fr = _pan(180.0, SPEEDS)
    mask = np.zeros((540, 960), bool)
    mask[200:340, 400:560] = True
    r = sh.measure(lambda f: fr.get(f), sorted(fr), exclude=lambda f: mask)
    assert r["confident"] and abs(r["deg"] - 180.0) <= 45.0


def test_resolve_order():
    assert sh.resolve({"shutter_deg": 180.0}, {"shutter": {"deg": 60.0, "confident": True}}) == (60.0, "measured")
    assert sh.resolve({"shutter_deg": 172.8}, {}) == (172.8, "manifest")
    assert sh.resolve({}, {"shutter": {"deg": 60.0, "confident": False}}) == (sh.DEFAULT_DEG, "default")
    assert sh.resolve({}, None) == (sh.DEFAULT_DEG, "default")


def test_pipeline_and_api_use_the_measured_shutter(tmp_path, monkeypatch):
    """project.json `shutter` (the plate measurement) wins over the manifest's value; DELETE goes back to it; the
    measure endpoint stores the result in project.json, never in the manifest."""
    import json
    import app as app_module
    from test_pipeline_from_project import _rich_project
    api, shot, mf, proj = _rich_project(tmp_path, monkeypatch)       # manifest shutter_deg = 123
    assert app_module._pipeline().shutter == 123.0 and app_module._pipeline().shutter_source == "manifest"
    assert api.get("/api/project/shutter").json()["source"] == "manifest"
    monkeypatch.setattr(sh, "measure", lambda *a, **k: {"deg": 75.0, "confident": True, "pairs": 9, "why": "",
                                                        "errors": {}, "speeds": {}})
    r = api.post("/api/project/shutter/measure")
    assert r.status_code == 200, r.text
    assert r.json()["deg"] == 75.0 and r.json()["source"] == "measured"
    assert app_module._pipeline().shutter == 75.0 and app_module._pipeline().shutter_source == "measured"
    assert json.loads(mf.read_text(encoding="utf-8"))["shutter_deg"] == 123.0      # manifest untouched
    assert app_module.STATE["shot_project"].data["shutter"]["deg"] == 75.0
    assert api.delete("/api/project/shutter").json()["source"] == "manifest"
    assert app_module._pipeline().shutter == 123.0


def test_unmeasurable_shot_keeps_the_previous_value(tmp_path, monkeypatch):
    import app as app_module
    from test_pipeline_from_project import _rich_project
    api, shot, mf, proj = _rich_project(tmp_path, monkeypatch)
    monkeypatch.setattr(sh, "measure", lambda *a, **k: {"deg": None, "confident": False, "pairs": 0,
                                                        "why": "za mało klatek", "errors": {}, "speeds": {}})
    r = api.post("/api/project/shutter/measure")
    assert r.status_code == 409 and "za mało klatek" in r.text
    assert "shutter" not in app_module.STATE["shot_project"].data and app_module._pipeline().shutter == 123.0
