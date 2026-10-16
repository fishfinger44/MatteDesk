# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor installs ffmpeg / Git itself (engine/tools_install.py) and links ComfyUI Desktop when no ComfyUI is found."""

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import tools_install as ti


@pytest.fixture
def no_tools(monkeypatch, tmp_path):
    """Nothing on PATH, nothing in the registry or the known folders; bin/ under tmp."""
    monkeypatch.setattr(ti.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(ti, "registry_path", lambda: [])
    monkeypatch.setattr(ti, "_known_locations", lambda key: [str(tmp_path / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe")])
    monkeypatch.setattr(ti, "BIN_DIR", tmp_path / "bin")
    monkeypatch.setattr(ti, "use_dir", lambda d: None)
    return tmp_path


def test_ffmpeg_portable_fallback_without_winget(no_tools):
    def fake_download(url, dst):
        assert url == ti.FFMPEG_ZIP_URL
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("ffmpeg-7.1-essentials_build/bin/ffmpeg.exe", b"x")
            z.writestr("ffmpeg-7.1-essentials_build/bin/ffprobe.exe", b"x")
    saved = {}
    res = ti.install_tool({}, "ffmpeg", download=fake_download, winget=None, write_local=saved.update)
    exe = no_tools / "bin" / "ffmpeg" / "bin" / "ffmpeg.exe"
    assert res["success"] and Path(res["path"]) == exe and exe.is_file()
    assert (exe.parent / "ffprobe.exe").is_file()          # ingest derives ffprobe from ffmpeg_path
    assert saved == {"ffmpeg_path": str(exe)}
    assert not (no_tools / "bin" / "ffmpeg-release-essentials.zip").exists()


def test_git_via_winget(no_tools, monkeypatch, tmp_path):
    git = tmp_path / "Git" / "cmd" / "git.exe"
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        git.parent.mkdir(parents=True)
        git.write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(ti, "_known_locations", lambda key: [str(git)])
    saved = {}
    res = ti.install_tool({}, "git", run=fake_run, winget="winget.exe", write_local=saved.update)
    assert calls and calls[0][:4] == ["winget.exe", "install", "--id", "Git.Git"]
    assert "--accept-package-agreements" in calls[0] and "--silent" in calls[0]
    assert res["path"] == str(git) and saved == {"git_path": str(git)}


def test_git_without_winget_fails_with_link(no_tools):
    with pytest.raises(RuntimeError, match="git-scm.com"):
        ti.install_tool({}, "git", winget=None, write_local=lambda v: None)


def test_already_installed_only_remembered(no_tools, tmp_path):
    exe = tmp_path / "ff.exe"
    exe.write_bytes(b"x")
    saved = {}
    res = ti.install_tool({"ffmpeg_path": str(exe)}, "ffmpeg", run=lambda *a, **k: pytest.fail("no install"),
                          write_local=saved.update)
    assert res["path"] == str(exe) and saved == {"ffmpeg_path": str(exe)}


def test_doctor_items_offer_install(no_tools, monkeypatch):
    from doctor import EnvironmentDoctor
    d = EnvironmentDoctor()
    assert d.check_git().action == "install_git" and d.check_git().status == "error"
    ff = d.check_ffmpeg()
    assert ff.action == "install_ffmpeg" and ff.status == "error"


def test_doctor_links_comfy_desktop_when_missing(tmp_path):
    from doctor import EnvironmentDoctor
    d = EnvironmentDoctor()
    d.config = {**d.config, "comfy_flavor": "desktop", "comfy_dir": str(tmp_path / "nie_ma")}
    it = d.check_comfy_installed()
    assert it is not None and it.status == "error"
    assert any(l.startswith("https://www.comfy.org/download") for l in it.links)
    d.config = {**d.config, "comfy_flavor": "git", "comfy_dir": str(tmp_path)}
    assert "main.py" in d.check_comfy_installed().message
    (tmp_path / "main.py").write_text("")
    assert d.check_comfy_installed() is None
