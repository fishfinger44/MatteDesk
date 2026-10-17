# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows installer: build script (no network, no real pip / ISCC), Inno script consistency, app-window launcher."""
import io
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

import tools.build_installer as bi
import tools.launch_app as la

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "MatteDesk.iss").read_text(encoding="utf-8")


def test_python_asset_picks_311_windows_install_only():
    rel = {"tag_name": "20261013", "assets": [
        {"name": "cpython-3.12.12+20261013-x86_64-pc-windows-msvc-install_only.tar.gz", "browser_download_url": "a"},
        {"name": "cpython-3.11.17+20261013-x86_64-unknown-linux-gnu-install_only.tar.gz", "browser_download_url": "b"},
        {"name": "cpython-3.11.17+20261013-x86_64-pc-windows-msvc-install_only.tar.gz", "browser_download_url": "c",
         "size": 48_000_000, "digest": "sha256:" + "d" * 64}]}
    a = bi.python_asset(api_get=lambda u: rel)
    assert a["url"] == "c" and a["sha256"] == "d" * 64


def test_bundle_python_runs_only_pip_into_bundled_python(tmp_path, monkeypatch):
    """Test copy of the install step: the real pip must never run here – the fake `run` records and asserts it."""
    cache, stage = tmp_path / "cache", tmp_path / "stage"
    cache.mkdir()
    stage.mkdir()
    (stage / "requirements.txt").write_text("fastapi\n")
    name = "cpython-3.11.17+20261013-x86_64-pc-windows-msvc-install_only.tar.gz"
    with tarfile.open(cache / name, "w:gz") as t:
        for f in ("python/python.exe", "python/pythonw.exe"):
            info = tarfile.TarInfo(f)
            info.size = 0
            t.addfile(info, io.BytesIO(b""))
    monkeypatch.setattr(bi, "python_asset", lambda: {"name": name, "url": "never", "size": 1, "sha256": None})
    monkeypatch.setattr(bi.dl, "download", lambda *a, **k: pytest.fail("cached archive must not be downloaded"))
    calls = []
    py = bi.bundle_python(stage, cache, run=lambda argv: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    assert py == stage / "python" / "python.exe"
    assert len(calls) == 1
    argv = calls[0]
    assert Path(argv[0]) == py and argv[1:4] == ["-m", "pip", "install"] and argv[-2:] == ["-r", str(stage / "requirements.txt")]
    assert not any("winget" in str(a) for a in argv)


def test_iscc_argv_and_iss_defines():
    argv = bi.iscc_argv("ISCC.exe", "0.4", Path("S"), Path("D"), 2.01, 6.5)
    assert "/DAppVersion=0.4" in argv and "/DPortableGB=2.0" in argv and argv[-1].endswith("MatteDesk.iss")
    for d in ("AppVersion", "StageDir", "OutputDir", "PortableGB", "PortableDiskGB"):
        assert f"#ifndef {d}" in ISS


def test_iss_per_user_two_languages_shortcuts():
    assert "PrivilegesRequired=lowest" in ISS and "{autopf}\\MatteDesk" in ISS
    assert "compiler:Languages\\Polish.isl" in ISS
    assert ISS.count("tools\\launch_app.py") == 3          # Start menu, desktop, "start now"
    assert "pythonw.exe" in ISS
    en = set(re.findall(r"^en\.(\w+)=", ISS, re.M))
    pl = set(re.findall(r"^pl\.(\w+)=", ISS, re.M))
    assert en == pl and {"ComfyQuestion", "ComfyYes", "ComfyNo", "ComfyLater"} <= en
    for used in re.findall(r"CustomMessage\('(\w+)'\)", ISS):
        assert used in en


def test_iss_calls_installer_setup_with_valid_choices():
    import tools.installer_setup as st
    src = (ROOT / "tools" / "installer_setup.py").read_text(encoding="utf-8")
    choices = re.search(r'choices=\(([^)]*)\)', src).group(1)
    for c in re.findall(r"--comfy (\w+)", ISS):
        assert f'"{c}"' in choices
    assert "installer_setup.py" in ISS and st.main


def test_launcher_browser_and_argv(tmp_path):
    env = {"ProgramFiles(x86)": str(tmp_path / "pf86"), "ProgramFiles": str(tmp_path / "pf"), "LOCALAPPDATA": str(tmp_path / "la")}
    edge = tmp_path / "pf86" / "Microsoft/Edge/Application/msedge.exe"
    chrome = tmp_path / "pf" / "Google/Chrome/Application/chrome.exe"
    chrome.parent.mkdir(parents=True)
    chrome.write_bytes(b"")
    assert Path(la.find_browser(env)) == chrome
    edge.parent.mkdir(parents=True)
    edge.write_bytes(b"")
    assert Path(la.find_browser(env)) == edge                 # Edge first
    argv = la.browser_argv("msedge.exe", Path("P"))
    assert f"--app={la.URL}" in argv and "--user-data-dir=P" in argv
    assert la.profile_dir(env) == tmp_path / "la" / "MatteDesk" / "browser"


def test_launcher_server_start_and_jobs(tmp_path, monkeypatch):
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "python.exe").write_bytes(b"")
    seen = {}

    def popen(argv, **kw):
        seen.update(argv=argv, **kw)
        return object()
    la.start_server(tmp_path, popen=popen)
    assert seen["argv"][0] == str(tmp_path / "python" / "python.exe")
    assert seen["argv"][1:4] == ["-m", "uvicorn", "app:app"] and seen["cwd"] == str(tmp_path)
    assert seen["creationflags"] == la.NO_WINDOW and (tmp_path / "logs").is_dir()
    assert la.running_job(lambda p: {"jobs": [{"state": "done", "label": "a"}, {"state": "running", "label": "SAM3"}]}) == "SAM3"
    assert la.running_job(lambda p: None) is None


def test_launcher_port_state(monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    monkeypatch.setattr(la, "PORT", s.getsockname()[1])
    try:
        assert la.port_state(lambda p: {"version": "0.4"}) == "mattedesk"
        assert la.port_state(lambda p: None) == "other"
    finally:
        s.close()
    assert la.port_state(lambda p: None) == "free"


def test_launcher_quiet_edge(tmp_path):
    argv = la.browser_argv("msedge.exe", tmp_path)
    for f in ("--no-first-run", "--disable-sync", "--hide-crash-restore-bubble"):
        assert f in argv
    assert any(a.startswith("--disable-features=") and "msImplicitSignin" in a for a in argv)
    la.mark_clean_exit(tmp_path)                       # no profile yet: nothing happens
    pref = tmp_path / "Default" / "Preferences"
    pref.parent.mkdir()
    pref.write_text('{"profile": {"exit_type": "Crashed", "name": "x"}}', encoding="utf-8")
    la.mark_clean_exit(tmp_path)
    import json
    d = json.loads(pref.read_text(encoding="utf-8"))
    assert d["profile"] == {"exit_type": "Normal", "exited_cleanly": True, "name": "x"}
    assert d["signin"]["allowed"] is False


def test_choose_browser_default_first(tmp_path):
    comet = tmp_path / "Comet" / "Application"
    (comet / "154.0").mkdir(parents=True)
    (comet / "comet.exe").write_bytes(b"")
    (comet / "154.0" / "chrome_elf.dll").write_bytes(b"")
    fx = tmp_path / "Mozilla Firefox"
    fx.mkdir()
    (fx / "firefox.exe").write_bytes(b"")
    (fx / "xul.dll").write_bytes(b"")
    assert la.is_chromium(str(comet / "comet.exe")) and not la.is_chromium(str(fx / "firefox.exe"))
    edge = lambda: "msedge.exe"
    assert la.choose_browser("default", default=lambda: str(comet / "comet.exe"), edge=edge) == (str(comet / "comet.exe"), "app")
    assert la.choose_browser("default", default=lambda: str(fx / "firefox.exe"), edge=edge) == (str(fx / "firefox.exe"), "tab")
    assert la.choose_browser("default", default=lambda: None, edge=edge) == ("msedge.exe", "app")
    assert la.choose_browser("edge", default=lambda: str(fx / "firefox.exe"), edge=edge) == ("msedge.exe", "app")
    assert la.choose_browser("default", default=lambda: None, edge=lambda: None) == (None, "tab")
    env = {"LOCALAPPDATA": str(tmp_path)}
    assert la.profile_dir(env, "C:/x/comet.exe").name == "browser-comet"
    assert la.profile_dir(env, "C:/x/msedge.exe").name == "browser"


def test_app_browser_setting(tmp_path):
    assert la.app_browser_setting(tmp_path) == "default"
    (tmp_path / "config.local.json").write_text('{"app_browser": "edge"}', encoding="utf-8")
    assert la.app_browser_setting(tmp_path) == "edge"


def test_replace_old_server(monkeypatch):
    monkeypatch.setattr(la, "disk_version", lambda app_dir=None: "0.31")
    stopped = []
    stopper = lambda: stopped.append(1) or True
    same = lambda p: {"version": "0.31", "stale": False} if p == "/api/version" else {"jobs": []}
    assert la.replace_old_server(same, ask=lambda *a, **k: 6, stopper=stopper) is False and not stopped
    old = lambda p: {"version": "0.3", "stale": True} if p == "/api/version" else {"jobs": []}
    asked = []
    assert la.replace_old_server(old, ask=lambda t, flags=0: asked.append(t) or 7, stopper=stopper) is False
    assert "0.3" in asked[0] and "0.31" in asked[0] and not stopped
    assert la.replace_old_server(old, ask=lambda *a, **k: 6, stopper=stopper) is True and stopped
