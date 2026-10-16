# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-machine configuration: shared config.json + config.local.json, relative paths, ComfyUI Desktop autodetect."""

import json
from pathlib import Path

import pytest

from engine.config import REPO_DIR, load_config, resolve_path, write_local


def _w(p: Path, d):
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_local_overrides_shared_and_relative_paths(tmp_path):
    shared = _w(tmp_path / "config.json", {"comfy_port": 8000, "sam3_checkpoint": "a.safetensors"})
    local = _w(tmp_path / "config.local.json", {"comfy_port": 8001, "comfy_dir": "../ComfyUI"})
    c = load_config(shared, local, appdata="")
    assert c["comfy_port"] == 8001 and c["_sources"]["comfy_port"] == "config.local.json"
    assert c["sam3_checkpoint"] == "a.safetensors"
    assert c["comfy_dir"] == str((REPO_DIR / "../ComfyUI").resolve()).replace("\\", "/")  # relative to MatteDesk
    assert c["models_dir"] == c["comfy_dir"] + "/models"


def test_absolute_path_other_drive_kept(tmp_path):
    local = _w(tmp_path / "config.local.json", {"comfy_dir": "F:/Comfy/ComfyUI", "ffmpeg_path": "ffmpeg"})
    c = load_config(tmp_path / "none.json", local, appdata="")
    assert c["comfy_dir"] == "F:/Comfy/ComfyUI" and c["ffmpeg_path"] == "ffmpeg"
    assert resolve_path("D:\\tools\\ffmpeg.exe").startswith("D:")


def test_desktop_autodetect_when_empty(tmp_path):
    appdata = tmp_path / "AppData"
    (appdata / "ComfyUI").mkdir(parents=True)
    _w(appdata / "ComfyUI" / "config.json", {"basePath": "E:/ComfyUI_data"})
    c = load_config(tmp_path / "none.json", tmp_path / "none.local.json", appdata=str(appdata))
    assert c["comfy_dir"] == "E:/ComfyUI_data" and c["_sources"]["comfy_dir"].startswith("autodetect")
    assert c["models_dir"] == "E:/ComfyUI_data/models"


def test_desktop2_autodetect(tmp_path):
    la, ad = tmp_path / "Local", tmp_path / "Roaming"
    inst = la / "Comfy-Desktop" / "ComfyUI-Installs" / "ComfyUI" / "ComfyUI"
    inst.mkdir(parents=True)
    (inst / "main.py").write_text("", encoding="utf-8")
    ymls = ad / "Comfy Desktop" / "instance-model-paths"
    ymls.mkdir(parents=True)
    (ymls / "inst-1.yaml").write_text("comfy.desktop_0:\n  base_path: 'E:\\Shared\\models'\n  is_default: true\n",
                                      encoding="utf-8")
    c = load_config(tmp_path / "none.json", tmp_path / "none.local.json", appdata=str(ad), localappdata=str(la))
    assert c["comfy_dir"] == str(inst).replace("\\", "/") and "Desktop 2" in c["_sources"]["comfy_dir"]
    assert c["models_dir"] == "E:/Shared/models"


def test_write_local_merges(tmp_path):
    local = _w(tmp_path / "config.local.json", {"comfy_dir": "X:/c"})
    write_local({"comfy_port": 8188, "_sources": {}}, local)
    assert json.loads(local.read_text()) == {"comfy_dir": "X:/c", "comfy_port": 8188}


def test_repo_config_has_no_machine_paths():
    shared = json.loads((REPO_DIR / "config.json").read_text(encoding="utf-8"))
    assert not {"comfy_dir", "models_dir", "comfy_port"} & set(shared)


def test_git_profile_does_not_leak_desktop_paths(tmp_path):
    local = _w(tmp_path / "config.local.json", {
        "comfy_flavor": "git", "comfy_dir": "C:/Desktop/ComfyUI", "models_dir": "C:/Desktop/models", "comfy_port": 8001,
        "comfy_git": {"comfy_dir": "E:/ComfyUI_git", "comfy_port": 8189}})
    c = load_config(tmp_path / "none.json", local, appdata="")
    assert c["comfy_dir"] == "E:/ComfyUI_git" and c["comfy_port"] == 8189
    assert c["models_dir"] == "E:/ComfyUI_git/models"


def test_git_shared_models_uses_desktop_models(tmp_path):
    local = _w(tmp_path / "config.local.json", {
        "comfy_flavor": "git", "models_dir": "C:/Desktop/models",
        "comfy_git": {"comfy_dir": "E:/ComfyUI_git", "shared_models": True}})
    c = load_config(tmp_path / "none.json", local, appdata="")
    assert c["models_dir"] == "C:/Desktop/models" and "wspólne" in c["_sources"]["models_dir"]


def test_old_portable_name_still_loads(tmp_path):
    """config.local.json from before 2026-10-15: comfy_flavor "portable" + `comfy_portable` = the git variant."""
    local = _w(tmp_path / "config.local.json", {
        "comfy_flavor": "portable", "comfy_dir": "C:/Desktop/ComfyUI", "comfy_port": 8001,
        "comfy_portable": {"comfy_dir": "E:/ComfyUI_git", "comfy_port": 8189}})
    c = load_config(tmp_path / "none.json", local, appdata="")
    assert c["comfy_flavor"] == "git" and c["comfy_dir"] == "E:/ComfyUI_git" and c["comfy_port"] == 8189
    # the next write migrates the file to the new names, nothing of the profile is lost
    from engine.config import write_local
    write_local({"comfy_autostart": False}, local)
    d = json.loads(local.read_text())
    assert d["comfy_flavor"] == "git" and "comfy_portable" not in d
    assert d["comfy_git"] == {"comfy_dir": "E:/ComfyUI_git", "comfy_port": 8189} and d["comfy_dir"] == "C:/Desktop/ComfyUI"


def test_set_flavor_and_write_local_active(tmp_path):
    from engine.config import set_comfy_flavor, write_local_active
    local = _w(tmp_path / "config.local.json", {"comfy_dir": "C:/Desktop/ComfyUI"})
    set_comfy_flavor("git", {"comfy_dir": "E:/git"}, local)
    write_local_active({"comfy_port": 8190}, local)
    d = json.loads(local.read_text())
    assert d["comfy_dir"] == "C:/Desktop/ComfyUI" and "comfy_port" not in d
    assert d["comfy_git"] == {"comfy_dir": "E:/git", "comfy_port": 8190} and d["comfy_flavor"] == "git"
    set_comfy_flavor("portable", None, local)   # old name accepted (an old page / script), stored as "git"
    assert json.loads(local.read_text())["comfy_flavor"] == "git"
    with pytest.raises(ValueError):
        set_comfy_flavor("standalone", None, local)
