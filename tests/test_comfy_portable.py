# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI portable (engine/comfy_portable.py, user decision 2026-10-17): detection, autostart command, download +
unpack (no network, no real 7-Zip), detection of installed ComfyUI, installer post-install step."""
import json
from pathlib import Path

import pytest

from engine import comfy_portable as cp
from engine import config as engine_config


def make_portable(root: Path) -> Path:
    pkg = root / cp.PACKAGE_DIR
    (pkg / "python_embeded").mkdir(parents=True)
    (pkg / "python_embeded" / "python.exe").write_bytes(b"")
    (pkg / "ComfyUI" / "custom_nodes").mkdir(parents=True)
    (pkg / "ComfyUI" / "main.py").write_text("")
    return pkg / "ComfyUI"


def make_git(root: Path) -> Path:
    (root / "custom_nodes").mkdir(parents=True)
    (root / "main.py").write_text("")
    return root


@pytest.fixture
def local_cfg(tmp_path, monkeypatch):
    p = tmp_path / "config.local.json"
    monkeypatch.setattr(engine_config, "LOCAL", p)
    return p


def test_find_and_kind(tmp_path):
    comfy = make_portable(tmp_path / "a")
    for hint in (tmp_path / "a", tmp_path / "a" / cp.PACKAGE_DIR, comfy):
        assert cp.find_comfy_dir(hint) == comfy
    assert cp.embedded_python(comfy) is not None
    git = make_git(tmp_path / "g" / "ComfyUI")
    assert cp.embedded_python(git) is None
    assert cp.kind({"comfy_flavor": "git", "comfy_dir": str(comfy)}) == "portable"
    assert cp.kind({"comfy_flavor": "git", "comfy_dir": str(git)}) == "git"
    assert cp.kind({"comfy_flavor": "desktop", "comfy_dir": str(comfy)}) == "desktop"


def test_autostart_command_portable_vs_git(tmp_path):
    from engine.comfy_launch import HOW_GIT, HOW_PORTABLE, _launch_cmd
    comfy = make_portable(tmp_path / "p")
    argv, cwd, how = _launch_cmd({"comfy_flavor": "git", "comfy_dir": str(comfy), "comfy_port": 8190})
    assert how == HOW_PORTABLE and Path(cwd) == comfy
    assert argv[0].endswith("python.exe") and "python_embeded" in argv[0]
    assert argv[1:] == ["-s", "main.py", "--windows-standalone-build", "--port", "8190"]
    git = make_git(tmp_path / "g")
    argv, _, how = _launch_cmd({"comfy_flavor": "git", "comfy_dir": str(git), "comfy_port": 8188})
    assert how == HOW_GIT and argv[1:] == ["main.py", "--port", "8188"]


def test_install_download_unpack_config(tmp_path, local_cfg):
    calls = []

    def fake_download(url, dst, sha256=None, progress=None):
        calls.append(("dl", url, sha256))
        Path(dst).write_bytes(b"7z")

    def fake_7z(argv):
        calls.append(("7z", argv))
        out = Path(argv[3][2:])
        make_portable(out)
        return type("R", (), {"returncode": 0})()

    info = {"url": "https://example/p.7z", "size": 2_000_000_000, "sha256": "c" * 64, "tag": "v0.41.0"}
    res = cp.install(tmp_path / "dest", info=info, download=fake_download, run=fake_7z, seven="7zr.exe")
    assert calls[0] == ("dl", info["url"], info["sha256"])
    assert calls[1][1][:3] == ["7zr.exe", "x", "-y"]
    assert not (tmp_path / "dest" / cp.ASSET).exists()           # archive removed after unpacking
    cfg = json.loads(local_cfg.read_text(encoding="utf-8"))
    assert cfg["comfy_flavor"] == "git" and cfg["comfy_git"]["comfy_dir"] == res["comfy_dir"]
    assert cfg["comfy_git"]["comfy_port"] == cp.PORT
    # second run: already unpacked -> no download
    calls.clear()
    cp.install(tmp_path / "dest", info=info, download=fake_download, run=fake_7z, seven="7zr.exe")
    assert calls == []


def test_find_installed_and_detect(tmp_path):
    old = make_git(tmp_path / "work" / "ComfyUI")
    new = make_portable(tmp_path / "Downloads")
    import os
    os.utime(old / "main.py", (1, 1))
    found = cp.find_installed([tmp_path], depth=3)
    assert found == [new, old]
    d = cp.detect(roots=[tmp_path], desktop=lambda: {}, running=lambda: None)
    assert d["kind"] == "portable" and Path(d["comfy_dir"]) == new and d["others"] == [str(old)]
    d = cp.detect(roots=[tmp_path], desktop=lambda: {"comfy_dir": "C:/Desk/ComfyUI"}, running=lambda: None)
    assert d["kind"] == "desktop" and len(d["others"]) == 2
    d = cp.detect(str(old), roots=[], desktop=lambda: {"comfy_dir": "C:/Desk"}, running=lambda: None)
    assert d["kind"] == "git" and Path(d["comfy_dir"]) == old      # the user's folder wins
    assert cp.detect(roots=[tmp_path / "none"], desktop=lambda: {}, running=lambda: None)["kind"] == "none"


def test_installer_setup_cli(tmp_path, local_cfg, monkeypatch):
    import tools.installer_setup as st
    monkeypatch.setattr(cp, "running_comfy", lambda: None)
    assert st.main(["--comfy", "later", "--no-pause"]) == 0 and not local_cfg.exists()
    comfy = make_portable(tmp_path / "pk")
    assert st.main(["--comfy", "have", "--dir", str(tmp_path / "pk"), "--no-pause"]) == 0
    assert json.loads(local_cfg.read_text(encoding="utf-8"))["comfy_git"]["comfy_dir"] == str(comfy).replace("\\", "/")
    monkeypatch.setattr(cp, "find_installed", lambda roots=None: [])
    assert st.main(["--comfy", "have", "--dir", str(tmp_path / "empty"), "--no-pause"]) in (0, 1)
    assert st.main(["--comfy", "portable", "--no-pause"]) == 1      # no folder given


def test_running_comfy_takes_port_and_python(tmp_path):
    git = make_git(tmp_path / "Comfy" / "ComfyUI")
    stats = {"system": {"argv": ["main.py", "--port", "8001"]}}
    get = lambda url: stats if ":8001/" in url else None
    sys_py = str(tmp_path / "Python311" / "python.exe")
    run = cp.running_comfy(get_json=get, proc=lambda port: {"exe": sys_py, "cmd": "python  main.py --port 8001"})
    assert run == {"port": 8001, "python": sys_py, "comfy_dir": ""}   # relative main.py: folder from the disk scan
    d = cp.detect(roots=[tmp_path], desktop=lambda: {}, running=lambda: run)
    assert d["kind"] == "git" and Path(d["comfy_dir"]) == git and d["port"] == 8001 and d["python"] == sys_py
    # absolute main.py in the command line names the folder directly
    run2 = cp.running_comfy(get_json=get, proc=lambda port: {"exe": sys_py, "cmd": f'python "{git / "main.py"}" --port 8001'})
    assert Path(run2["comfy_dir"]) == git
    # portable: python_embeded -> its ComfyUI, no comfy_python stored (found next to the folder)
    pk = make_portable(tmp_path / "P")
    emb = str(pk.parent / "python_embeded" / "python.exe")
    d = cp.detect(roots=[], desktop=lambda: {}, running=lambda: {"port": 8188, "python": emb, "comfy_dir": ""})
    assert d["kind"] == "portable" and Path(d["comfy_dir"]) == pk and d["python"] == ""
    # Desktop 2 server
    d = cp.detect(roots=[], desktop=lambda: {}, running=lambda: {
        "port": 8000, "python": r"E:\Apps\Comfy-Desktop\ComfyUI-Installs\a\.venv\Scripts\python.exe", "comfy_dir": ""})
    assert d["kind"] == "desktop" and d["port"] == 8000
    assert cp.running_comfy(get_json=lambda u: None, proc=lambda p: {}) is None


def test_installer_setup_uses_running_comfy(tmp_path, local_cfg, monkeypatch):
    import tools.installer_setup as st
    git = make_git(tmp_path / "ComfyUI")
    monkeypatch.setattr(cp, "running_comfy", lambda: {"port": 8001, "python": "C:/Py/python.exe", "comfy_dir": str(git)})
    assert st.main(["--comfy", "have", "--no-pause"]) == 0
    prof = json.loads(local_cfg.read_text(encoding="utf-8"))["comfy_git"]
    assert prof["comfy_port"] == 8001 and prof["comfy_python"] == "C:/Py/python.exe"
