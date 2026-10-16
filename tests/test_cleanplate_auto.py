# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 4b of docs/PLAN_HAIR_TOOL.md / docs/PLAN_CLEANPLATE.md (synthetic): auto clean plate (temporal median of the
uncovered pixels, holes inpainted with confidence 0), the `clean` zone engine (difference key against the clean plate:
a line behind the hair keys out WITHOUT a "-" stroke, the hair over it stays), routing with the clean score, feather of
the zone override at the area border, API `POST /api/project/cleanplate_auto` (+ 409 without cores, plate registered),
CLI `cleanplate-auto`."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.cleanplate_auto import (CONF_INPAINT, build_cleanplate, fill_holes, load_cleanplate, opencv_inpaint,
                                    sample_frames, write_cleanplate)
from engine.localmatte import local_matte
from tests.test_localmatte import FG, GOLD, NAVY, _motley, _scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def _moving_shot(n=24, h=160, w=240, seed=7):
    """Static textured background with a gold line; a disc moving left -> right; a second disc that never moves."""
    bg = _motley(h, w, seed)
    cv2.line(bg, (0, 40), (w - 1, 70), tuple(float(c) for c in GOLD), 3)
    frames = list(range(1, n + 1))
    imgs, cores = {}, {}
    for i, f in enumerate(frames):
        img = bg.copy()
        core = np.zeros((h, w), np.uint8)
        cx = 30 + int((w - 60) * i / (n - 1))
        cv2.circle(core, (cx, 100), 22, 1, -1)
        cv2.circle(core, (200, 130), 14, 1, -1)          # static object: pixels under it are never uncovered
        img[core > 0] = FG
        imgs[f], cores[f] = img, core.astype(bool)
    return bg, frames, imgs, cores


def test_median_cleanplate_recovers_background_and_marks_holes(tmp_path):
    bg, frames, imgs, cores = _moving_shot()
    clean, conf, stats = build_cleanplate(frames, lambda f: imgs[f], lambda f: cores[f], grow_px=4, max_samples=60)
    static = np.zeros(bg.shape[:2], np.uint8)
    cv2.circle(static, (200, 130), 14 + 4, 1, -1)
    static = static.astype(bool)
    good = conf > 0.5
    assert good.mean() > 0.8
    err = np.abs(clean - bg)[good].mean()
    assert err < 1e-3, err                                   # exact background where uncovered
    assert np.abs(clean - bg)[good].max() < 2e-3          # float16 sample precision
    assert conf[static].max() <= CONF_INPAINT["opencv"] + 1e-6 and stats["inpaint_backend"] == "opencv"
    assert 0 < stats["never_uncovered_px"] <= static.sum() * 1.1      # the ellipse kernel grows a bit more than the circle
    # moving disc: everything under its path was uncovered in other frames -> high confidence along the path
    path = np.zeros(bg.shape[:2], np.uint8)
    cv2.rectangle(path, (30, 78), (210, 122), 1, -1)
    path = path.astype(bool)
    assert conf[path & ~static].mean() > 0.8
    # files + reload, hard links for every frame
    pattern = write_cleanplate(tmp_path / "cleanplate_auto", clean, conf, frames, stats)
    assert "cp_####.png" in pattern
    assert all((tmp_path / "cleanplate_auto" / f"cp_{f:04d}.png").exists() for f in frames)
    back = load_cleanplate(tmp_path / "cleanplate_auto")
    assert back is not None
    c2, conf2, st2 = back
    assert np.abs(c2 - clean).max() < 1.0 / 65535 * 2 and np.abs(conf2 - conf).max() < 1.0 / 255 * 2
    assert st2["schema"].startswith("matte_desk/cleanplate_auto") and st2["first"] == frames[0]
    assert sample_frames(list(range(300)), 60) and len(sample_frames(list(range(300)), 60)) <= 60


def test_fill_holes_lama_callable_or_fallback():
    img = np.full((32, 32, 3), 0.5, np.float32)
    hole = np.zeros((32, 32), bool)
    hole[10:20, 10:20] = True
    img[hole] = 0.0
    out, backend = fill_holes(img, hole, lambda im, m: np.full_like(im, 0.5))
    assert backend == "comfy_lama" and np.allclose(out[hole], 0.5)

    def boom(im, m):
        raise RuntimeError("Comfy down")
    out2, backend2 = fill_holes(img, hole, boom)
    assert backend2 == "opencv" and np.allclose(out2[hole], 0.5, atol=0.02)
    assert np.allclose(opencv_inpaint(img, hole)[hole], 0.5, atol=0.02)


def test_clean_engine_keeps_hair_over_line_without_stroke():
    bg = np.tile(NAVY, (256, 320, 1)).astype(np.float32)
    rng = np.random.default_rng(5)
    bg = np.clip(bg + rng.normal(0, 0.012, bg.shape).astype(np.float32), 0, 1)
    img, core, hair, line, region = _scene(bg, with_line=True)
    clean = img.copy()                        # the true clean plate = background incl. the gold line
    clean[core | hair] = bg[core | hair]
    cv2.line(clean, (0, 256 * 62 // 100 - 256 * 26 // 100 - 30), (319, 256 * 62 // 100 - 256 * 26 // 100 - 70),
             tuple(float(c) for c in GOLD), 3)
    with pytest.raises(ValueError):
        local_matte(img, core, region, engine="clean")   # no clean plate
    res = local_matte(img, core, region, engine="clean", clean=clean)
    a = res.alpha
    assert a[hair & region].mean() > 0.6
    assert a[line & region & ~hair].mean() < 0.1          # the line keys out with NO "-" stroke
    assert a[region & ~hair & ~line & ~core].mean() < 0.05
    assert res.stats["cleanplate"] is True and "clean_lo" in res.stats
    # hair crossing the line: still foreground where the hair is
    cross = hair & cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & region
    if cross.any():
        assert a[cross].mean() > 0.5
    # auto routing: the clean engine gets the weight where the clean plate agrees with the plate (everywhere here)
    auto = local_matte(img, core, region, engine="auto", clean=clean, clean_conf=np.ones(core.shape, np.float32))
    assert "clean" in auto.weights and auto.stats["engine_share"]["clean"] > 0.6
    assert auto.alpha[line & region & ~hair].mean() < 0.15
    # confidence 0 -> no clean weight
    noconf = local_matte(img, core, region, engine="auto", clean=clean, clean_conf=np.zeros(core.shape, np.float32))
    assert noconf.stats["engine_share"].get("clean", 0.0) < 0.05


def test_api_cleanplate_auto_and_cli(tmp_path, monkeypatch):
    import md
    from test_md_cli import TCHttp, _run
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch)
    # status before: none
    r = api.get("/api/project/cleanplate_auto")
    assert r.status_code == 200 and r.json()["source"] == "brak" and r.json()["auto"] is None
    # an object without a core -> 409 listing it
    pipe.store.create("hand", ["hand"], depth_order=0)
    pipe.store.update("obj", depth_order=1)
    r = api.post("/api/project/cleanplate_auto", json={})
    assert r.status_code == 409 and r.json()["detail"]["missing_front"] == ["hand"], r.text
    assert api.delete("/api/object/hand").status_code == 200
    # build (OpenCV fallback: the mock Comfy has no LaMa node)
    code, out = _run(["cleanplate-auto", "--wait"], TCHttp(api))
    assert code == 0, out
    assert "Clean plate (auto): gotowy" in out and "cleanplate_auto" in out
    st = api.get("/api/project/cleanplate_auto").json()
    assert st["source"] == "auto" and st["auto"]["inpaint_backend"] in ("opencv", "comfy_lama")
    plates = api.get("/api/project/plates").json()
    names = [p["name"] for p in (plates if isinstance(plates, list) else plates.get("plates", []))]
    assert "cleanplate_auto" in names
    assert (Path(app_module.STATE["work_dir"]) / "cleanplate_auto" / "cleanplate_auto.json").exists()
    # the shot manifest was never written
    mf = Path(app_module.STATE["manifest_path"])
    assert "cleanplate_auto" not in json.loads(mf.read_text(encoding="utf-8"))["plates"]
    # a matte zone preview now reports the auto clean plate and the `clean` engine works
    cx, cy, s = shot.centers[start]
    poly = [[cx - 60, cy - 260], [cx + 60, cy - 260], [cx + 60, cy - 120], [cx - 60, cy - 120]]
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "clean", "preset": "hair"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    assert r.json()["preview"]["cleanplate"] == "auto" and r.json()["preview"]["stats"]["cleanplate"] is True
    code, out = _run(["status"], TCHttp(api))
    assert code == 0 and "Clean plate: auto" in out
