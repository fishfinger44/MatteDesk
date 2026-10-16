# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor: VideoHelperSuite (VHS_LoadImagesPath, Comfy I/O "png8" for 8-bit sources) – check (mocked /object_info,
custom_nodes when Comfy is offline) and the "Zainstaluj" action (git clone + requirements, no second copy)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from doctor import VHS_LABEL, EnvironmentDoctor
from engine import installer


def _doctor(tmp_path, **sim):
    (tmp_path / "custom_nodes").mkdir(exist_ok=True)
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True, **sim})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    return {it.name: it for it in doc.run_all_checks().items}


def test_vhs_missing_in_object_info_is_required_with_install_button(tmp_path):
    vhs = _doctor(tmp_path, object_info={"SAM3_Detect": {}})[VHS_LABEL]
    assert vhs.status == "error" and vhs.action == "install_vhs_nodes"
    assert "VHS_LoadImagesPath" in vhs.message and "Wymagane teraz: TAK" in vhs.fix_suggestion
    assert any("Kosinkadink/ComfyUI-VideoHelperSuite" in l for l in vhs.links)
    assert "licenc" not in vhs.fix_suggestion.lower()


def test_vhs_registered_is_ok(tmp_path):
    vhs = _doctor(tmp_path, object_info={"VHS_LoadImagesPath": {"input": {}}})[VHS_LABEL]
    assert vhs.status == "ok" and vhs.action is None


@pytest.mark.parametrize("folder", ["ComfyUI-VideoHelperSuite", "comfyui-videohelpersuite"])
def test_vhs_offline_checks_custom_nodes(tmp_path, folder):
    assert _doctor(tmp_path)[VHS_LABEL].action == "install_vhs_nodes"     # Comfy offline, folder missing
    (tmp_path / "custom_nodes" / folder).mkdir()
    vhs = _doctor(tmp_path)[VHS_LABEL]
    assert vhs.status == "ok" and vhs.action is None


def test_install_vhs_clones_and_installs_requirements(tmp_path, monkeypatch):
    (tmp_path / "custom_nodes").mkdir()
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    monkeypatch.setattr(installer, "comfy_python", lambda cfg: Path("comfy-python"))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True)
            (Path(cmd[-1]) / "requirements.txt").write_text("opencv-python\nimageio-ffmpeg\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    r = installer.install_git_node({"comfy_dir": str(tmp_path)}, "vhs_nodes", run=fake_run)
    assert r["success"] and r["restart_comfy"]
    assert calls[0][-2:] == ["https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
                             str(tmp_path / "custom_nodes" / "ComfyUI-VideoHelperSuite")]
    assert calls[1][:4] == ["comfy-python", "-m", "pip", "install"] and calls[1][-1].endswith("requirements.txt")


def test_install_vhs_reuses_manager_folder(tmp_path, monkeypatch):
    mgr = tmp_path / "custom_nodes" / "comfyui-videohelpersuite"
    mgr.mkdir(parents=True)
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    r = installer.install_git_node({"comfy_dir": str(tmp_path)}, "vhs_nodes",
                                   run=lambda cmd, **k: pytest.fail(f"no clone / pip expected: {cmd}"))
    assert r["success"] and "comfyui-videohelpersuite" in r["log"][0]
    assert sorted(p.name for p in (tmp_path / "custom_nodes").iterdir()) == ["comfyui-videohelpersuite"]


# --- user decision 2026-10-15: no EXR fallback; png8 without VHS = clear error BEFORE frames are written -----------

class _NoVhsClient:
    """ComfyUI answering /object_info/<unknown class> with {} (as the real one does); every other call is a failure."""
    base_url = "http://no-vhs.test"

    def __init__(self, has_vhs=False):
        self.asked = 0
        self.has_vhs = has_vhs

    def get_object_info(self, node_class=None):
        self.asked += 1
        return {node_class: {"input": {}}} if self.has_vhs else {}

    def __getattr__(self, name):
        raise AssertionError(f"nothing may reach ComfyUI without VHS: {name}")


def test_require_vhs_missing_offline_present_cached(monkeypatch):
    from engine import comfy_io
    monkeypatch.setattr(comfy_io, "_VHS_OK", set())
    with pytest.raises(comfy_io.MissingComfyNode, match="Doktor"):
        comfy_io.require_vhs(_NoVhsClient())
    comfy_io.require_vhs(SimpleNamespace(get_object_info=lambda c=None: None))   # Comfy not answering: no verdict
    ok = _NoVhsClient(has_vhs=True)
    comfy_io.require_vhs(ok)
    comfy_io.require_vhs(ok)
    assert ok.asked == 1                                                        # cached per ComfyUI address


def test_png8_sam3_and_matanyone_fail_before_writing(tmp_path, monkeypatch):
    import numpy as np
    from engine import comfy_io
    from engine.crop import CropSegment
    from engine.refine import Finalizer
    from engine.sam3 import Sam3Runner
    monkeypatch.setattr(comfy_io, "_VHS_OK", set())
    img = np.full((64, 64, 3), 0.5, np.float32)
    runner = Sam3Runner(_NoVhsClient(), tmp_path / "sam", io_mode="png8")
    with pytest.raises(RuntimeError, match="VideoHelperSuite"):
        runner.detect(img, prompt="person")
    with pytest.raises(RuntimeError, match="VideoHelperSuite"):
        runner.track([img] * 3, np.zeros((64, 64), np.float32))
    assert not list(tmp_path.rglob("*.png"))
    fin = Finalizer(_NoVhsClient(), tmp_path / "io", 64, 64, lambda f, s, sc: img, io_mode="png8")
    seg = CropSegment("s00", 1, 3, 0, 0, [0, 0, 64, 64], [0, 0, 0, 0], 1.0)
    core = {f: np.ones((64, 64), bool) for f in (1, 2, 3)}
    with pytest.raises(RuntimeError, match="Zainstaluj"):
        fin._matanyone_segment(seg, [1, 2, 3], core, {1: core[1]}, 1, 1.0, "final")
    assert not list(tmp_path.rglob("*.png"))


def test_lama_needs_vhs_too():
    from engine.cleanplate_auto import lama_available
    info = {"INPAINT_InpaintWithModel": {"INPAINT_InpaintWithModel": {}}, "VHS_LoadImagesPath": {}}
    assert not lama_available(SimpleNamespace(get_object_info=lambda c: info[c]))
    info["VHS_LoadImagesPath"] = {"VHS_LoadImagesPath": {}}
    assert lama_available(SimpleNamespace(get_object_info=lambda c: info[c]))
