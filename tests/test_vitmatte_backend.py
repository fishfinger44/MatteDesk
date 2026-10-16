# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ViTMatte backends: in this process (venv with torch) or a long-lived worker in ComfyUI's Python (no torch in
MatteDesk's .venv since v0.11). The worker protocol is tested with MD_VITMATTE_FAKE=1 in this .venv's Python."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from engine import installer, vitmatte as vm

FAKE_ENV = {**os.environ, "MD_VITMATTE_FAKE": "1"}


def _crop(h=37, w=53, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((h, w, 3)).astype(np.float32)
    tri = np.full((h, w), 128, np.uint8)
    tri[: h // 3] = 0
    tri[-h // 4:] = 255
    return img, tri


def _old_run(img_bgr, trimap_u8, unknown_value, forward):
    """engine/vitmatte.py VitMatteRunner.run before the backend split (forward = the model on the padded CHW input)."""
    h, w = trimap_u8.shape
    ph, pw = (-h) % 32, (-w) % 32
    rgb = np.clip(img_bgr[..., ::-1], 0, 1).astype(np.float32)
    rgb = (rgb - vm.IMAGENET_MEAN) / vm.IMAGENET_STD
    tri = (trimap_u8.astype(np.float32) / 255.0)
    if abs(float(unknown_value) - 0.5) > 1e-6:
        tri = np.where(trimap_u8 == 128, np.float32(unknown_value), tri).astype(np.float32)
    x = np.concatenate([rgb, tri[..., None]], axis=2)
    if ph or pw:
        x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    a = np.clip(forward(x)[:h, :w], 0, 1).astype(np.float32)
    a[trimap_u8 == 255] = 1.0
    a[trimap_u8 == 0] = 0.0
    return a


def _fake_forward(x):
    return (0.3 * x[..., 0] + 0.6 * x[..., 3] - 0.1 * x[..., 2]).astype(np.float32)


@pytest.mark.parametrize("uv", [0.5, 0.7])
def test_prepare_finish_bit_identical_to_old_run(uv):
    img, tri = _crop()
    old = _old_run(img, tri, uv, _fake_forward)
    x = vm.prepare_input(img, tri, uv)
    assert x.shape == (64, 64, 4) and x.dtype == np.float32
    assert np.array_equal(vm.finish_alpha(_fake_forward(x), tri), old)


def test_comfy_worker_protocol_with_fake_model(tmp_path):
    vm._CHECK_CACHE.clear()
    r = vm.ComfyVitMatteRunner("hustvl/fake", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    assert r.available() == (True, "")
    img, tri = _crop()
    a = r.run(img, tri, unknown_value=0.7)
    # fake model = trimap channel; FG / BG forced, unknown = the "Fine Tune" level
    assert a.shape == tri.shape and a.dtype == np.float32
    assert np.all(a[tri == 255] == 1.0) and np.all(a[tri == 0] == 0.0) and np.allclose(a[tri == 128], 0.7)
    assert r.device == "fake" and vm.runner_info(r)["backend"] == "comfy"
    pid = r._proc.pid
    r.run(img, tri)
    assert r._proc.pid == pid                       # one long-lived worker, model loaded once
    r._proc.kill(); r._proc.wait()
    assert np.allclose(r.run(img, tri)[tri == 128], 128 / 255)   # dead worker -> restarted (trained unknown level)
    assert r._proc.pid != pid
    r.close()
    assert r._proc is None


def test_comfy_worker_reports_error_without_hanging(tmp_path):
    r = vm.ComfyVitMatteRunner("hustvl/fake", python=sys.executable, env=FAKE_ENV, timeout_s=60)
    with pytest.raises(RuntimeError, match="ViTMatte"):
        r.forward(np.zeros((32, 32, 2), np.float32))   # fake model needs 4 channels -> worker error
    img, tri = _crop()
    assert r.run(img, tri).shape == tri.shape          # the worker survives a bad request
    r.close()


def test_check_worker_python_missing_and_fake(tmp_path):
    vm._CHECK_CACHE.clear()
    ok, why, cached = vm.check_worker_python(tmp_path / "nope.exe")
    assert not ok and "ComfyUI" in why
    ok, dev, cached = vm.check_worker_python(sys.executable, ["hustvl/fake"], env=FAKE_ENV)
    assert ok and dev == "fake" and cached == []
    vm._CHECK_CACHE.clear()


def test_get_runner_picks_backend_and_explains(monkeypatch):
    monkeypatch.setattr(vm, "_RUNNERS", {})
    monkeypatch.setattr(vm.VitMatteRunner, "available", staticmethod(lambda: (False, "no torch")))
    monkeypatch.setattr(vm.ComfyVitMatteRunner, "available", lambda self: (False, "ViTMatte niedostępny: x"))
    with pytest.raises(RuntimeError, match="Doktor"):
        vm.get_runner("hustvl/a")
    monkeypatch.setattr(vm.ComfyVitMatteRunner, "available", lambda self: (True, ""))
    r = vm.get_runner("hustvl/a")
    assert isinstance(r, vm.ComfyVitMatteRunner) and vm.get_runner("hustvl/a") is r
    monkeypatch.setattr(vm, "_RUNNERS", {})
    monkeypatch.setattr(vm.VitMatteRunner, "available", staticmethod(lambda: (True, "")))
    assert isinstance(vm.get_runner("hustvl/a"), vm.VitMatteRunner)


def test_install_vitmatte_pip_only_when_missing():
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        missing = "-c" in cmd and calls.count(cmd) == 1 and run.missing
        return SimpleNamespace(returncode=1 if missing else 0, stdout="", stderr="")

    run.missing = False
    res = installer.install_vitmatte({}, run=run, python="comfy-python")
    assert res["success"] and not any("pip" in c for c in calls)
    assert any("--download" in c and vm.DEFAULT_MODEL in c for c in calls)
    calls.clear(); run.missing = True
    res = installer.install_vitmatte({}, run=run, python="comfy-python")
    assert res["success"] and any(c[1:4] == ["-m", "pip", "install"] for c in calls)
    bad = installer.install_vitmatte({}, run=lambda c, **k: SimpleNamespace(returncode=0 if "-c" in c else 1,
                                                                           stdout="", stderr="no net"), python="py")
    assert not bad["success"] and any("BŁĄD" in l for l in bad["log"])


@pytest.mark.parametrize("state,status,action", [("ready", "ok", None), ("no_weights", "warning", "install_vitmatte"),
                                                 ("unavailable", "warning", "install_vitmatte")])
def test_doctor_vitmatte_item(tmp_path, monkeypatch, state, status, action):
    from doctor import EnvironmentDoctor
    monkeypatch.setattr(vm.VitMatteRunner, "available", staticmethod(lambda: (False, "no torch")))
    res = {"ready": (True, "cuda", [vm.DEFAULT_MODEL]), "no_weights": (True, "cuda", []),
           "unavailable": (False, "ModuleNotFoundError: transformers", [])}[state]
    monkeypatch.setattr(vm, "check_worker_python", lambda *a, **k: res)
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    it = {i.name: i for i in doc.run_all_checks().items}["Model: ViTMatte"]
    assert it.status == status and it.action == action
    if status != "ok":
        assert "Wymagane teraz: TAK" in it.fix_suggestion and it.links
