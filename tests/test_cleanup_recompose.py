# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„⟲ Złóż ponownie” (user decision 2026-11-13): a finished Inpaint version composited again WITHOUT the model – its
stored model frames (model/m_####.png) + the CURRENT composite band through the run's Cleaner -> a NEW version; the
live preview of one frame. Synthetic: no model runs."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as A
from api import cleanup as AC
from engine import cleanup as cu
from engine import h3_fill
from engine import inpaint_recompose as IR

W, H = 240, 176
FRAMES = (1, 2, 3)


def _obj():
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.hypot(xx - 120, yy - 88) < 22).astype(np.float32)


@pytest.fixture
def rec_app(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    bdir = tmp_path / "b"
    bdir.mkdir()
    rng = np.random.default_rng(1)
    obj = _obj()
    for f in FRAMES:
        p = np.clip(110 + rng.normal(0, 6, (H, W, 3)), 0, 255).astype(np.uint8)
        p[obj > 0] = (20, 20, 230)
        cv2.imwrite(str(bdir / f"f_{f:04d}.png"), p)
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(bdir / "f_####.png")}})
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
          "grain": "off"}
    cu.save_settings(work, vs)
    # the source version as a MiniMax-H3 run leaves it: model/ at the canvas, run.json
    v = cu.next_version(work)
    (v / "model").mkdir(parents=True)
    cw, ch = h3_fill.canvas(W, H)
    model = {}
    for f in FRAMES:
        im = np.full((ch, cw, 3), 60 + 10 * f, np.uint8)
        cv2.imwrite(str(v / "model" / f"m_{f:04d}.png"), im)
        model[f] = im
    (v / "anchors").mkdir()
    cv2.imwrite(str(v / "anchors" / "anchor_0002.png"), np.zeros((4, 4, 3), np.uint8))
    (v / "run.json").write_text(json.dumps({"settings": cu.settings(vs), "frames": [1, 3], "backend": "h3",
                                            "anchors": [2], "ext": "png", "date": "2026-11-13T10:00:00"}), encoding="utf-8")
    return TestClient(A.app), work, v, model, obj, bdir


def _comp(vdir, f):
    return cv2.imread(str(vdir / "comp" / f"c_{f:04d}.png"), cv2.IMREAD_UNCHANGED)


def _patch_alpha(vdir, f):
    p = cv2.imread(str(vdir / "patch" / f"p_{f:04d}.png"), cv2.IMREAD_UNCHANGED)
    return p[..., 3]


def test_recompose_equals_a_run_with_the_same_band(rec_app, tmp_path):
    c, work, v, model, obj, bdir = rec_app
    cu.save_settings(work, {**cu.load_settings(work), "grow_px": 14, "soft_px": 4})   # the band edited after the run
    r = c.post(f"/api/cleanup/versions/{v.name}/recompose")
    assert r.status_code == 200, r.text
    new = cu.cleanup_dir(work) / r.json()["version"]
    assert new.name == "v002" and r.json()["from"] == v.name
    rj = json.loads((new / "run.json").read_text(encoding="utf-8"))
    assert rj["recomposed_from"] == v.name and rj["source_model"]["label"] == "MiniMax-H3"
    assert rj["settings"]["grow_px"] == 14 and rj["settings"]["gen_margin_px"] == 30 and rj["settings"]["prompt"] == "a wall"
    assert rj["anchors"] == [2] and (new / "anchors" / "anchor_0002.png").exists()
    assert sorted(p.name for p in (new / "model").iterdir()) == sorted(p.name for p in (v / "model").iterdir())
    md = (new / "MODELS.md").read_text(encoding="utf-8")
    assert "MiniMax-H3" in md and f"Złożona ponownie z wersji {v.name}" in md
    assert any(x["version"] == "v002" and x["has_model"] and x["recomposed_from"] == v.name for x in cu.versions(work))
    # the same as a run of the Cleaner with that band and the same model frames
    s = IR.recompose_settings(cu.load_settings(work), json.loads((v / "run.json").read_text(encoding="utf-8"))["settings"])
    pat = str(bdir / "f_####.png")
    cl = cu.Cleaner(s, W, H, False, plate=lambda f: AC._read_plate(AC.frame_file(pat, f)),
                    remove_alphas=lambda f: [obj], keep_alphas=lambda f: [],
                    key_fill=h3_fill.H3Fill(model, W, H, False), plate_file=lambda f: AC.frame_file(pat, f),
                    gen_mask=cu.gen_mask_fn(s, [lambda f: obj], []))
    cl.run(list(FRAMES), tmp_path / "ref", "png")
    for f in FRAMES:
        assert np.array_equal(_comp(new, f), _comp(tmp_path / "ref", f))
    # it can be recomposed again (model/ copied)
    assert c.post(f"/api/cleanup/versions/{new.name}/recompose").status_code == 200


def test_narrower_band_smaller_patch_plate_outside_bit_for_bit(rec_app):
    c, work, v, model, obj, bdir = rec_app
    sizes = {}
    for g in (2, 20):
        cu.save_settings(work, {**cu.load_settings(work), "grow_px": g, "soft_px": 0})
        r = c.post(f"/api/cleanup/versions/{v.name}/recompose")
        assert r.status_code == 200, r.text
        new = cu.cleanup_dir(work) / r.json()["version"]
        a = _patch_alpha(new, 2)
        sizes[g] = int((a > 0).sum())
        plate = cv2.imread(str(bdir / "f_0002.png"), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(_comp(new, 2)[a == 0], plate[a == 0])     # outside the patch: the plate file's values
    assert 0 < sizes[2] < sizes[20]


def test_recompose_refuses_without_model_frames(rec_app):
    c, work, v, model, obj, bdir = rec_app
    old = cu.next_version(work)
    old.mkdir()
    (old / "run.json").write_text(json.dumps({"settings": {"remove": ["obj"], "model": "h3"}, "frames": [1, 3]}),
                                  encoding="utf-8")
    r = c.post(f"/api/cleanup/versions/{old.name}/recompose")
    assert r.status_code == 409 and "model/" in r.json()["detail"]
    (v / "model" / "m_0003.png").unlink()                           # model frames not covering the version's range
    r = c.post(f"/api/cleanup/versions/{v.name}/recompose")
    assert r.status_code == 409 and "F3" in r.json()["detail"]
    assert c.get(f"/api/cleanup/versions/{old.name}/preview/2").status_code == 409


def test_live_preview_jpeg_of_the_preview_size(rec_app):
    c, work, v, model, obj, bdir = rec_app
    r = c.get(f"/api/cleanup/versions/{v.name}/preview/2?grow_px=10&soft_px=2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"   # 2026-11-14: JPEG, ~10x smaller
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    sc = A._proxy_scale()
    assert img.shape[:2] == (round(H * sc), round(W * sc))
    # the unsaved slider value decides the patch: a wider band changes more pixels
    plate = cv2.imread(str(bdir / "f_0002.png"), cv2.IMREAD_COLOR)
    wide = cv2.imdecode(np.frombuffer(c.get(f"/api/cleanup/versions/{v.name}/preview/2?grow_px=24&soft_px=0").content,
                                      np.uint8), cv2.IMREAD_COLOR)
    narrow = cv2.imdecode(np.frombuffer(c.get(f"/api/cleanup/versions/{v.name}/preview/2?grow_px=1&soft_px=0").content,
                                        np.uint8), cv2.IMREAD_COLOR)
    diff = lambda a: int((np.abs(a.astype(int) - plate.astype(int)).max(axis=2) > 24).sum())   # noqa: E731 (JPEG noise)
    assert diff(wide) > diff(narrow) > 0
    assert cu.load_settings(work)["grow_px"] == 6                   # nothing saved by the preview
    assert c.get(f"/api/cleanup/versions/{v.name}/preview/9").status_code == 409


def test_recompose_settings_take_the_band_of_the_versions_thing():
    vs = {"remove": ["a"], "model": "vace", "grow_px": 5, "soft_px": 1, "gen_margin_px": 40, "prompt": "x"}
    cur = {"remove": ["a"], "model": "h3", "grow_px": 12, "soft_px": 2, "prompt": "y", "grain": "on"}
    s = IR.recompose_settings(cur, vs)
    assert (s["model"], s["prompt"], s["grow_px"], s["soft_px"], s["gen_margin_px"], s["grain"]) == ("vace", "x", 12, 2, 40, "on")
    cur2 = {"remove": ["b"], "grow_px": 30, "per_target": {"a": {"grow_px": 9}}}
    assert IR.recompose_settings(cur2, vs)["grow_px"] == 9
    assert IR.recompose_settings(cur, vs, {"grow_px": 3})["grow_px"] == 3
    old = {"remove": ["a"], "model": "h3", "grow_px": 30, "soft_px": 10}          # before the margin: its own band
    assert IR.recompose_settings(cur, old)["gen_margin_px"] == 40


def test_ui_wiring():
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "data-cu-recomp=" in js and "/recompose`" in js and "⟲ Złóż ponownie" in js and "v.has_model" in js
    assert 'id="cuLive"' in js and 'id="cuRecompose"' in js and "Zastosuj (nowa wersja)" in js
    assert "/preview/${f}?grow_px=" in js
    assert js.index("overlayHooks.push(drawLive)") < js.index("overlayHooks.push(drawBand)")   # under the band lines
    assert 'id="cuGen"' in js and "gen_margin_px" in js and 'id="cuBandLimit"' in js and "limit_contours" in js
    assert 'id="cuGrain"' in js and 'id="cuGrainStrength"' in js


def test_model_frame_preview_ui_2026_11_15():
    """„🎞 obraz modelu” in the palette Złożenie: the version's model picture over the frame (GET …/model/{f})."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    js = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert 'id="cuModelPv"' in js and "/model/${f}?t=" in js and "v.overlayHooks.push(drawModelPv)" in js
    src = (root / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert '@router.get("/api/cleanup/versions/{version}/model/{frame}")' in src and 'f"m_{int(frame):04d}.png"' in src
