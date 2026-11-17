# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comfy Desktop 2 detection on testers' machines (MatteDesk-dev issues 11 and 12).

- issue 12: Desktop 2 installed outside %LOCALAPPDATA% (another drive, path only in installations.json) -> comfy_dir
  was not autodetected, so the Doctor could not install our nodes and MatAnyone2 failed in ComfyUI.
- issue 11: a folder behind a junction Windows 11 does not trust -> Path.is_dir raised OSError (WinError 448) and the
  whole Doctor returned 500. Such a folder must count as missing."""

import json
from pathlib import Path

from engine import comfy_launch as cl
from engine import config as ec


def _install_elsewhere(tmp_path):
    """Desktop 2 with the install on 'another drive': only installations.json knows where it is."""
    ad, la = tmp_path / "Roaming", tmp_path / "Local"
    root = ad / "Comfy Desktop"
    (root / "instance-model-paths").mkdir(parents=True)
    (root / "instance-model-paths" / "inst-d.yaml").write_text("comfy.desktop_0:\n  base_path: 'D:/models'\n")
    inst_path = tmp_path / "D_drive" / "Comfy" / "Main"
    comfy = inst_path / "ComfyUI"
    (comfy / ".venv" / "Scripts").mkdir(parents=True)
    (comfy / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    (comfy / "main.py").write_text("")
    (comfy / "custom_nodes").mkdir()
    (root / "settings.json").write_text(json.dumps({"inputDir": str(tmp_path / "junction_in")}))
    (root / "installations.json").write_text(json.dumps([
        {"id": "inst-d", "installPath": str(inst_path), "sourceId": "standalone", "status": "installed",
         "lastLaunchedAt": 5}]))
    la.mkdir()
    return ad, la, comfy


def test_comfy_dir_autodetected_from_installations_json(tmp_path):
    ad, la, comfy = _install_elsewhere(tmp_path)
    d2 = ec.desktop2_paths(str(ad), str(la))
    assert Path(d2["comfy_dir"]) == comfy and d2["models_dir"] == "D:/models"
    cfg = ec.load_config(shared=tmp_path / "none.json", local=tmp_path / "none_local.json",
                         appdata=str(ad), localappdata=str(la))
    assert Path(cfg["comfy_dir"]) == comfy
    assert cfg["_sources"]["comfy_dir"] == "autodetect (Comfy Desktop 2)"
    assert Path(ec.desktop_autodetect(str(ad), str(la))["comfy_dir"]) == comfy


def test_untrusted_mount_point_counts_as_missing(tmp_path, monkeypatch):
    ad, la, comfy = _install_elsewhere(tmp_path)
    bad = str(tmp_path / "junction_in")
    real_stat = Path.stat

    def stat(self, *a, **k):
        if str(self).startswith(bad):
            raise OSError(22, "Nie można przejść do tej ścieżki … niezaufany punkt instalacji", str(self), 448)
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    assert ec.safe_is_dir(bad) is False and ec.safe_exists(bad) is False and ec.safe_is_file(bad) is False
    assert ec.safe_glob(bad, "*") == []
    inst = cl.desktop2_install({}, str(ad), str(la))   # used to raise WinError 448 -> Doctor 500
    assert inst["comfy_dir"] == comfy and inst["input_dir"] is None


def test_desktop2_install_root_behind_bad_junction(tmp_path, monkeypatch):
    """installDir itself unreadable: the install from installations.json is still found, nothing raises."""
    ad, la, comfy = _install_elsewhere(tmp_path)
    blocked = tmp_path / "blocked"
    s = json.loads((ad / "Comfy Desktop" / "settings.json").read_text())
    s["installDir"] = str(blocked)
    (ad / "Comfy Desktop" / "settings.json").write_text(json.dumps(s))
    real_stat = Path.stat

    def stat(self, *a, **k):
        if str(self).startswith(str(blocked)):
            raise OSError(22, "untrusted mount point", str(self), 448)
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    assert Path(ec.desktop2_paths(str(ad), str(la))["comfy_dir"]) == comfy
    assert cl.desktop2_install({}, str(ad), str(la))["comfy_dir"] == comfy


def test_portable_detection_survives_untrusted_mount_point(tmp_path, monkeypatch):
    """ComfyUI portable / git: a folder behind an untrusted junction = not ComfyUI, nothing raises."""
    from engine import comfy_portable as cp
    bad = tmp_path / "junction"
    good = tmp_path / "ComfyUI_windows_portable"
    (good / "ComfyUI" / "custom_nodes").mkdir(parents=True)
    (good / "ComfyUI" / "main.py").write_text("")
    (good / "python_embeded").mkdir()
    (good / "python_embeded" / "python.exe").write_bytes(b"")
    real_stat = Path.stat

    def stat(self, *a, **k):
        if str(self).startswith(str(bad)):
            raise OSError(22, "untrusted mount point", str(self), 448)
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    assert cp.find_comfy_dir(bad) is None and cp.embedded_python(bad / "ComfyUI") is None
    assert cp.kind({"comfy_flavor": "git", "comfy_dir": str(bad / "ComfyUI")}) == "git"
    assert cp.find_comfy_dir(good) == good / "ComfyUI"
    assert cp.kind({"comfy_flavor": "git", "comfy_dir": str(good / "ComfyUI")}) == "portable"
    out = cp.detect(user_dir=str(bad), roots=[tmp_path], desktop=lambda: {}, running=lambda: None)
    assert out["kind"] == "portable" and Path(out["comfy_dir"]) == good / "ComfyUI"
