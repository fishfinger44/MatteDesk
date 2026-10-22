# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""BiRefNet_HR-matting / Lucida worker (engine/birefnet_worker.py) and runner (engine/birefnet.py). The worker protocol
is tested with MD_BIREFNET_FAKE=1 in this .venv's Python (fake model = luminance of the RGB input)."""

import os
import sys

import numpy as np
import pytest

from engine import birefnet as br, vitmatte as vm

FAKE_ENV = {**os.environ, "MD_BIREFNET_FAKE": "1"}


def _crop(h=37, w=53, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3)).astype(np.float32)
    tri = np.full((h, w), 128, np.uint8)
    tri[: h // 3] = 0
    tri[-h // 4:] = 255
    return img, tri


def test_model_ids_and_labels():
    assert br.model_id_for(None) == br.DEFAULT_MODEL == "ZhengPeng7/BiRefNet_HR-matting"
    assert br.model_id_for("lucida") == "egeorcun/lucida"
    assert set(br.MODELS) == set(br.LABELS)
    assert br.ComfyBirefnetRunner("egeorcun/lucida", python=sys.executable).label == "Lucida"


def test_prepare_input_is_rgb_float_no_resize():
    img, _ = _crop()
    x = br.prepare_input(img)
    assert x.shape == img.shape and x.dtype == np.float32
    assert np.array_equal(x, img[..., ::-1])            # BGR -> RGB, nothing else (resize happens on the GPU)


def test_worker_protocol_with_fake_model():
    vm._CHECK_CACHE.clear()
    r = br.ComfyBirefnetRunner("ZhengPeng7/BiRefNet_HR-matting", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    assert r.available() == (True, "")
    img, tri = _crop()
    a = r.run(img, tri, unknown_value=0.9)              # unknown_value ignored – no trimap input
    lum = np.clip(img[..., ::-1] @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1)
    assert a.shape == tri.shape and a.dtype == np.float32
    assert np.all(a[tri == 255] == 1.0) and np.all(a[tri == 0] == 0.0)
    assert np.allclose(a[tri == 128], lum[tri == 128], atol=1e-6)   # float alpha, no 8-bit step
    assert r.device == "fake" and vm.runner_info(r)["backend"] == "comfy"
    pid = r._proc.pid
    r.run(img, tri)
    assert r._proc.pid == pid                           # one long-lived worker
    r.close()
    assert r._proc is None
    vm._CHECK_CACHE.clear()


def test_worker_bad_input_reports_error_and_survives():
    r = br.ComfyBirefnetRunner("egeorcun/lucida", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    with pytest.raises(RuntimeError, match="Lucida"):
        r.forward(np.zeros((32, 32, 4), np.float32))    # 4 channels -> worker error, named after the model
    img, tri = _crop()
    assert r.run(img, tri).shape == tri.shape
    r.close()


def test_check_uses_own_worker_cache_key():
    vm._CHECK_CACHE.clear()
    ok, dev, _ = vm.check_worker_python(sys.executable, ["egeorcun/lucida"], env=FAKE_ENV, worker=br.WORKER)
    assert ok and dev == "fake"
    assert any(str(br.WORKER) in k for k in vm._CHECK_CACHE) and not any(str(vm.WORKER) in k for k in vm._CHECK_CACHE)
    vm._CHECK_CACHE.clear()


def test_get_runner_explains_missing(monkeypatch):
    monkeypatch.setattr(br, "_RUNNERS", {})
    monkeypatch.setattr(br.ComfyBirefnetRunner, "available", lambda self: (False, "Lucida niedostępny: x"))
    with pytest.raises(RuntimeError, match="Doktor"):
        br.get_runner("egeorcun/lucida")


def test_zone_engine_dispatch(monkeypatch):
    """params.vitmatte_model picks the runner of the per-frame zone engine: ViTMatte names -> engine.vitmatte,
    birefnet_hr_matting / lucida -> engine.birefnet (user decision 2026-10-21: one list in the ViTMatte zone)."""
    from engine import zones
    got = []
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: got.append(("vm", model_id)) or "vm")
    monkeypatch.setattr(br, "get_runner", lambda model_id=None: got.append(("br", model_id)) or "br")
    assert zones.edge_model_runner(None) == "vm" and zones.edge_model_runner("small") == "vm"
    assert zones.edge_model_runner("lucida") == "br" and zones.edge_model_runner("birefnet_hr_matting") == "br"
    assert got == [("vm", vm.DEFAULT_MODEL), ("vm", vm.MODELS["small"]), ("br", "egeorcun/lucida"),
                   ("br", "ZhengPeng7/BiRefNet_HR-matting")]


def test_install_birefnet_pip_only_missing_never_torch():
    from types import SimpleNamespace
    from engine import installer
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        miss = cmd[1] == "-c" and cmd[2] in run.missing
        return SimpleNamespace(returncode=1 if miss else 0, stdout="", stderr="")

    run.missing = {"import timm"}
    res = installer.install_birefnet({}, run=run, python="py")
    assert res["success"] and ["py", "-m", "pip", "install", "timm"] in calls
    assert sum("--download" in c for c in calls) == len(br.MODELS)
    calls.clear(); run.missing = set()
    assert installer.install_birefnet({}, run=run, python="py")["success"] and not any("pip" in c for c in calls)
    run.missing = {"import torchvision"}
    calls.clear()
    bad = installer.install_birefnet({}, run=run, python="py")
    assert not bad["success"] and not any("pip" in c for c in calls)   # torchvision is never pip-installed


@pytest.mark.parametrize("res,status,action", [((True, "cuda", list(br.MODELS.values())), "ok", None),
                                               ((True, "cuda", ["egeorcun/lucida"]), "info", "install_birefnet"),
                                               ((False, "ModuleNotFoundError: timm", []), "info", "install_birefnet")])
def test_doctor_birefnet_item_optional_and_commercial(tmp_path, monkeypatch, res, status, action):
    from doctor import BIREFNET_LABEL, EnvironmentDoctor
    monkeypatch.setattr(vm, "check_worker_python", lambda *a, **k: res)
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    items = {i.name: i for i in doc.run_all_checks().items}
    it = items[BIREFNET_LABEL]
    assert it.status == status and it.action == action
    assert it.license and "MIT" in it.license and not it.license_nc                # commercial OK
    if status != "ok":
        assert "Wymagane teraz: NIE" in it.fix_suggestion
    assert items["Model: ViTMatte"].license_nc                                     # ViTMatte: Adobe-trained weights


def test_birefnet_zone_defaults_one_crop_at_model_side():
    """Station 2026-10-21 (Woman_01 F56): BiRefNet on 448 px outline tiles lost the arm edges, one crop was clean –
    BiRefNet models default to one crop at about their input side; explicit params and ViTMatte unchanged."""
    from engine import zones
    out = {"region": "outline"}
    assert zones.vitmatte_tile_px(out) == zones.OUTLINE_TILE_PX and zones.edge_model_px(out) == 1008
    assert zones.vitmatte_tile_px({**out, "vitmatte_model": "birefnet_hr_matting"}) == 0
    assert zones.edge_model_px({**out, "vitmatte_model": "birefnet_hr_matting"}) == 2016
    assert zones.edge_model_px({"vitmatte_model": "lucida"}) == 1008
    assert zones.vitmatte_tile_px({**out, "vitmatte_model": "lucida", "tile_px": 448}) == 448
    assert zones.edge_model_px({"vitmatte_model": "birefnet_hr_matting", "model_px": 756}) == 756
