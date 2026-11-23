# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""One display transform for the C1 pictures in the viewer (user report 2026-11-23, SD160_150 v004 F27: „exposure
jump on the anchor frame”). Measured: the comp EXRs / plate proxies had no jump (F26 -> F27 block diff <= 1.6 / 255,
as any held pair), but the live composite overlay (and the model picture, the anchor previews) were plain sRGB of the
linear plate (to_display8, the models' encoding) drawn over proxies made with the PlateToDisplay curve – on an ACES
plate ~11 codes darker and harder on the frame the viewer stopped at. Now every such picture goes through
void_engine.to_view8 / model_to_view8 = the proxies' curve; the models' input stays to_display8. Synthetic."""
import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402
from engine import cleanup as cu  # noqa: E402
from engine import h3_anchors as HA  # noqa: E402
from engine import h3_fill  # noqa: E402
from engine.cleanup3d.void_engine import model_to_view8, to_display8, to_view8  # noqa: E402
from engine.ingest import ProxyGenerator  # noqa: E402

W, H = 240, 176
FRAMES = (1, 2, 3)


def _lin(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (0.02 + 0.6 * xx / W)[..., None] * np.array([0.8, 1.0, 1.2], np.float32)   # shadows .. mids, AP0-ish
    return np.clip(base + rng.normal(0, 0.005, (H, W, 3)), 0, None).astype(np.float32)


def test_view_curve_is_the_proxy_curve(tmp_path):
    lin = _lin()
    p = tmp_path / "f_0001.exr"
    assert cv2.imwrite(str(p), lin)
    proxy = ProxyGenerator.load_frame_rgb(p, colorspace="ACES2065-1")[..., ::-1]   # RGB -> BGR
    v = to_view8(lin, True)
    assert np.abs(v.astype(int) - proxy.astype(int)).max() <= 1                   # the same picture as the viewer's plate
    # the old overlay (to_display8) was a different picture: darker in the shadows / mids
    assert to_display8(lin, True).astype(float).mean() < v.astype(float).mean() - 8
    assert np.array_equal(to_view8(lin[..., :3].clip(0, 1), False), to_display8(lin.clip(0, 1), False))   # non-linear: as is


def test_model_picture_to_view_round_trip():
    lin = _lin(1).clip(0, 1)
    m8 = to_display8(lin, True)                       # what a model gets / gives back (sRGB)
    assert np.abs(model_to_view8(m8, True).astype(int) - to_view8(lin, True).astype(int)).max() <= 3
    assert np.array_equal(model_to_view8(m8, False), m8)


@pytest.fixture
def exr_app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    bdir = tmp_path / "b"
    bdir.mkdir()
    yy, xx = np.mgrid[0:H, 0:W]
    obj = (np.hypot(xx - 120, yy - 88) < 22).astype(np.float32)
    for f in FRAMES:
        p = _lin(f)
        p[obj > 0] = (0.01, 0.01, 0.9)
        assert cv2.imwrite(str(bdir / f"f_{f:04d}.exr"), p)
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(bdir / "f_####.exr")}, "colorspace": "ACES2065-1"})
    monkeypatch.setitem(A.STATE, "shot_project", None)
    monkeypatch.setattr(A, "_plate_size", lambda: (W, H))
    monkeypatch.setattr(A, "_work_frames", lambda: list(FRAMES))

    class Objs:
        def list(self, groups=True):
            return [{"name": "obj"}]

        def dir(self, n):
            return tmp_path / n
    import engine.cleanup_masks as CM
    import engine.export as EX
    monkeypatch.setattr(A, "_objects", lambda: Objs())
    monkeypatch.setattr(CM, "resolve", lambda store, o, src: {"run": "r1", "source": "core", "valid": True, "by": "own", "why": ""})
    monkeypatch.setattr(EX, "run_alpha_loader", lambda d, run, w, h: (lambda f: obj))
    vs = {"remove": ["obj"], "model": "h3", "grow_px": 6, "soft_px": 3, "gen_margin_px": 30, "prompt": "a wall",
          "grain": "off", "tone": False}
    cu.save_settings(work, vs)
    v = cu.next_version(work)
    (v / "model").mkdir(parents=True)
    cw, ch = h3_fill.canvas(W, H)
    for f in FRAMES:
        cv2.imwrite(str(v / "model" / f"m_{f:04d}.png"), to_display8(cv2.resize(_lin(f), (cw, ch)), True))
    (v / "run.json").write_text(json.dumps({"settings": cu.settings(vs), "frames": [1, 3], "backend": "h3", "linear": True,
                                            "ext": "exr", "date": "2026-11-23T10:00:00"}), encoding="utf-8")
    return TestClient(A.app), work, v, bdir, obj


def _dec(r):
    assert r.status_code == 200, r.text
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).astype(int)


def test_live_composite_and_model_picture_match_the_plate_proxy(exr_app, tmp_path):
    c, work, v, bdir, obj = exr_app
    pv = _dec(c.get(f"/api/cleanup/versions/{v.name}/preview/2"))
    proxy = tmp_path / "proxy.jpg"
    ProxyGenerator.render_proxy_file(bdir / "f_0002.exr", proxy, colorspace="ACES2065-1")
    px = cv2.imread(str(proxy), cv2.IMREAD_COLOR)
    px = cv2.resize(px, (pv.shape[1], pv.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
    far = cv2.resize((cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0).astype(np.uint8),
                     (pv.shape[1], pv.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    # outside the patch the live composite = the plate: the same picture as the proxy under it (JPEG noise only)
    assert abs((pv - px)[far].mean()) < 1.5 and np.abs(pv - px)[far].mean() < 3
    old = cv2.resize(to_display8(cv2.imread(str(bdir / "f_0002.exr"), cv2.IMREAD_UNCHANGED), True),
                     (pv.shape[1], pv.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
    assert (px - old)[far].mean() > 8                                          # the old overlay: the visible jump
    mp = _dec(c.get(f"/api/cleanup/versions/{v.name}/model/2"))
    assert abs((mp - px)[far].mean()) < 3                                      # the model picture too


def test_anchor_gen_view_vs_export(exr_app):
    c, work, v, bdir, obj = exr_app
    lin = cv2.imread(str(bdir / "f_0002.exr"), cv2.IMREAD_UNCHANGED)
    m8 = to_display8(lin, True)                                                 # an anchor in the models' space
    root = cu.cleanup_dir(work) / "anchors"
    ok, buf = cv2.imencode(".png", m8)
    g = HA.add_gen(root, "a000001", buf.tobytes(), {"frame": 2})
    url = f"/api/cleanup/anchors/a000001/gens/{g}"
    raw = c.get(url)
    assert raw.headers["content-type"] == "image/png" and raw.content == buf.tobytes()   # export: the stored PNG as is
    shown = _dec(c.get(url + "?view=1&t=x"))
    want = cv2.resize(model_to_view8(m8, True), (shown.shape[1], shown.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
    assert np.abs(shown - want).mean() < 3


def test_ui_previews_ask_for_the_view_curve():
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "view=1&t=" in js and 'href="${esc(g.url)}" download=' in js      # previews in the viewer's curve, export raw
