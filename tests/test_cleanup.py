# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate cleaning MVP (engine/cleanup.py, user decision 2026-11-06): the removal mask is the objects' alpha made hard,
grown and softened outward, never over the objects that stay; the clean plate fills first, LaMa the rest (here a
stand-in), OpenCV when LaMa fails; outside the mask the comp is the plate bit for bit; the patch is premultiplied RGBA
with the mask as alpha; scene-linear plates go through ACEScct for the model and back; versions live in <work>/inpaint."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import cleanup as cu

W, H = 160, 120


def scene():
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.dstack([xx / W, yy / H, np.full((H, W), 0.3)]).astype(np.float32)          # a gradient background
    obj = (np.hypot(xx - 60, yy - 60) < 18).astype(np.float32)                         # the object to remove
    keep = ((xx > 85) & (xx < 110) & (yy > 40) & (yy < 80)).astype(np.float32)         # an object that stays
    plate = bg.copy()
    plate[obj > 0] = (0.9, 0.1, 0.1)
    plate[keep > 0] = (0.1, 0.9, 0.1)
    return bg, obj, keep, plate


def test_mask_grow_soft_and_holdout():
    _, obj, keep, _ = scene()
    m = cu.removal_mask([obj], grow_px=4, soft_px=0)
    assert m[60, 60] == 1 and m[60, 60 + 18 + 3] == 1 and m[60, 60 + 18 + 6] == 0          # grown by 4
    soft = cu.removal_mask([obj], grow_px=4, soft_px=4)
    assert 0 < soft[60, 60 + 18 + 5] < 1 and soft[60, 60 + 18 + 9] == 0                     # falls outward
    big = cu.final_mask([obj], [keep], cu.settings({"grow_px": 40, "soft_px": 0}))
    assert big[60, 95] == 0 and big[60, 82] == 1                                             # holdout cuts the patch
    nohold = cu.final_mask([obj], [keep], cu.settings({"grow_px": 40, "soft_px": 0, "holdout": False}))
    assert nohold[60, 95] == 1
    with pytest.raises(ValueError):
        cu.removal_mask([], 4, 4)


def _cleaner(tmp_path, s, clean=None, inpaint=None, linear=False):
    bg, obj, keep, plate = scene()
    return cu.Cleaner(s, W, H, linear, plate=lambda f: plate, remove_alphas=lambda f: [obj],
                      keep_alphas=lambda f: [keep], clean=clean, inpaint=inpaint), bg, obj, keep, plate


def test_clean_plate_first_then_lama_and_plate_untouched_outside(tmp_path):
    bg, obj, keep, plate = scene()
    half = np.zeros((H, W), np.float32)
    half[:, :60] = 1                                       # the clean plate covers only the left half of the object
    calls = []

    def lama(imgs, masks):
        calls.append(len(imgs))
        return [np.where(m[..., None] > 0, 0.5, im).astype(np.float32) for im, m in zip(imgs, masks)]
    c, *_ = _cleaner(tmp_path, {"grow_px": 3, "soft_px": 2}, clean=lambda f: (bg, half), inpaint=lama)
    rj = c.run([1, 2, 3], tmp_path / "v001", "png")
    assert calls == [3] and rj["backend"] == "lama" and rj["counts"]["clear plate"] > 0 and rj["counts"]["LaMa"] > 0
    comp = cv2.imread(str(tmp_path / "v001" / "comp" / "c_0002.png")).astype(np.float32) / 255
    src = cv2.imread(str(tmp_path / "v001" / "source" / "s_0002.png"), cv2.IMREAD_GRAYSCALE)
    m = c.mask(2)
    out = m == 0
    assert np.abs(comp[out] - plate[out]).max() <= 1 / 255 + 1e-6                          # RGB of the plate outside
    assert src[60, 50] == cu.SRC_CLEAN and abs(comp[60, 50] - bg[60, 50]).max() < 2 / 255   # left: clean plate
    assert src[60, 70] == cu.SRC_LAMA and abs(comp[60, 70] - 0.5).max() < 2 / 255           # right: LaMa
    assert src[60, 95] == cu.SRC_PLATE                                                      # the kept object
    patch = cv2.imread(str(tmp_path / "v001" / "patch" / "p_0002.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    assert patch.shape[2] == 4 and patch[5, 5, 3] == 0 and patch[60, 60, 3] == 1
    assert np.allclose(patch[60, 60, :3], comp[60, 60], atol=2 / 255)                      # premultiplied, alpha 1 inside
    rj2 = json.loads((tmp_path / "v001" / "run.json").read_text(encoding="utf-8"))
    assert rj2["n_frames"] == 3 and rj2["settings"]["grow_px"] == 3


def test_lama_failure_falls_back_to_opencv_and_cleanplate_only(tmp_path):
    def broken(imgs, masks):
        raise RuntimeError("ComfyUI nie odpowiada")
    c, *_ = _cleaner(tmp_path, {}, inpaint=broken)
    rj = c.run([1], tmp_path / "a", "png")
    assert rj["backend"] == "opencv" and rj["counts"]["OpenCV"] > 0 and "LaMa niedostępna" in rj["notes"][0]
    c2, *_ = _cleaner(tmp_path, {"fill": "cleanplate"}, inpaint=broken)
    rj2 = c2.run([1], tmp_path / "b", "png")
    assert rj2["counts"]["OpenCV"] == 0 and rj2["counts"]["LaMa"] == 0 and rj2["notes"]


def test_linear_plates_round_trip_through_acescct(tmp_path):
    lin = np.array([[[0.0, 0.18, 1.0], [4.0, 0.01, 0.5]]], np.float32)
    assert np.allclose(cu.from_model(cu.to_model(lin, True), True), lin, rtol=1e-4, atol=1e-5)
    c, bg, obj, keep, plate = _cleaner(tmp_path, {}, inpaint=lambda i, m: [x.copy() for x in i], linear=True)
    rj = c.run([1], tmp_path / "lin", "exr")
    import os
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    assert rj["ext"] == "exr" and (tmp_path / "lin" / "comp" / "c_0001.exr").exists() and rj["linear"]


def test_settings_and_versions_live_next_to_matte_desk(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    s = cu.save_settings(work, {"remove": ["rig"], "grow_px": 999, "fill": "bogus"})
    assert s["grow_px"] == 200 and s["fill"] == "auto" and (tmp_path / "shot" / "inpaint" / "cleanup.json").exists()
    assert cu.load_settings(work)["remove"] == ["rig"]
    assert cu.next_version(work).name == "v001"
    (tmp_path / "shot" / "inpaint" / "v001").mkdir()
    (tmp_path / "shot" / "inpaint" / "v001" / "run.json").write_text('{"frames": [1, 3], "settings": {"remove": ["rig"]}}')
    assert cu.next_version(work).name == "v002" and cu.versions(work)[0]["remove"] == ["rig"]
    assert not any(p.name == "inpaint" for p in work.iterdir())                 # nothing inside matte_desk/


def test_mode_switch_and_routes_are_wired():
    """The UI switch lives in its own script (objects.js untouched), the router is included by app.py."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    js = (root / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "modeSwitch" in js and "/api/cleanup/run" in js and "/api/cleanup/mask/" in js and "updatePlateSelector" in js
    assert '<script src="/static/cleanup.js"></script>' in (root / "static" / "index.html").read_text(encoding="utf-8")
    assert "from api.cleanup import make as _make_cleanup" in (root / "app.py").read_text(encoding="utf-8")


def test_minimax_dropped_2026_11_09(tmp_path):
    """MiniMax-Remover dropped (user decision 2026-11-09, sh060: LaMa better): an old "minimax" setting reads as LaMa,
    the LaMa fill failing still falls back to OpenCV. (The model choice came back 2026-11-10 with LaMa / Qwen-Image-2.1 /
    VOID – tests/test_cleanup_models.py – never MiniMax-Remover.)"""
    assert "minimax" not in cu.MODELS and cu.MODELS[0] == "lama" and cu.settings({"model": "minimax", "steps": 7})["model"] == "lama"
    assert "steps" not in cu.settings() and not hasattr(cu, "minimax_batch_comfy")
    assert set(cu.SOURCE_NAMES.values()) == {"plate", "clear plate", "cleanplate_auto", "LaMa", "OpenCV"}

    def broken(imgs, masks):
        raise RuntimeError("Comfy gone")
    c, *_ = _cleaner(tmp_path, {"model": "minimax"}, inpaint=broken)
    rj = c.run([1], tmp_path / "x", "png")
    assert rj["backend"] == "opencv" and "LaMa niedostępna" in rj["notes"][0]
    root = Path(__file__).resolve().parent.parent
    for f in ("static/cleanup.js", "api/cleanup.py", "engine/cleanup.py", "static/app.js", "doctor.py",
              "engine/installer.py", "engine/model_trash.py"):
        txt = (root / f).read_text(encoding="utf-8")
        assert "minimax_batch" not in txt and "MiniMax-Remover\"" not in txt and "minimax_weights" not in txt, f


def test_output_keeps_the_plate_depth(tmp_path):
    """User 2026-11-10: the render is written at the plate's depth - 8-bit plates -> PNG8, 10 / 12 / 16-bit (DPX,
    PNG16) -> PNG16 (the full-frame comp keeps the plate's precision, was always 8-bit), linear -> EXR."""
    assert cu.output_ext(True, np.zeros((2, 2, 3), np.uint16)) == "exr"
    assert cu.output_ext(False, np.zeros((2, 2, 3), np.uint8)) == "png"
    assert cu.output_ext(False, np.zeros((2, 2, 3), np.uint16)) == "png16"
    assert cu.output_ext(False, None) == "png16" and cu.file_ext("png16") == "png"
    bg, obj, keep, plate = scene()
    plate = np.round(plate * 1023) / 1023                     # a 10-bit plate (0..1, like _read_plate of DPX)
    c = cu.Cleaner({"grow_px": 3, "soft_px": 2}, W, H, False, plate=lambda f: plate, remove_alphas=lambda f: [obj],
                   keep_alphas=lambda f: [keep], inpaint=lambda i, m: [np.where(mm[..., None] > 0, 0.5, im).astype(np.float32)
                                                                        for im, mm in zip(i, m)])
    rj = c.run([1], tmp_path / "v001", "png16")
    assert rj["ext"] == "png" and rj["bits"] == 16
    comp = cv2.imread(str(tmp_path / "v001" / "comp" / "c_0001.png"), cv2.IMREAD_UNCHANGED)
    patch = cv2.imread(str(tmp_path / "v001" / "patch" / "p_0001.png"), cv2.IMREAD_UNCHANGED)
    assert comp.dtype == np.uint16 and patch.dtype == np.uint16 and patch.shape[2] == 4
    out = c.mask(1) == 0
    assert np.abs(comp.astype(np.float64) / 65535 - plate)[out].max() < 1e-4      # 10-bit precision kept outside
    assert (comp[out] % 257 != 0).any()                                         # not 8-bit values stretched


def test_version_numbers_are_never_given_again(tmp_path):
    """Deleting is permanent (2026-11-13): a deleted version's number is not reused by the next run."""
    import shutil
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    v1 = cu.next_version(work); v1.mkdir()
    v2 = cu.next_version(work); v2.mkdir()
    assert (v1.name, v2.name) == ("v001", "v002")
    cu.remember_version(work)
    shutil.rmtree(v2)
    assert cu.next_version(work).name == "v003"

