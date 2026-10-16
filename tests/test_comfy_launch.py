# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI autostart with MatteDesk: the ComfyUI chosen in the Doctor (Comfy Desktop 2 headless / Desktop app / git).
Synthetic only: fake install folders, popen mocked – ComfyUI is never launched."""

import json
import time

import pytest

from engine import comfy_launch as cl


def _reset():
    cl.STATE.clear()
    cl.STATE["state"] = "idle"


@pytest.fixture
def no_desktop2(tmp_path, monkeypatch):
    """No Comfy Desktop 2 on this 'machine' (the real %APPDATA% / %LOCALAPPDATA% must not leak into the tests)."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "empty_roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty_local"))


def _fake_desktop2(tmp_path, launch_args="--enable-manager", inst_id="inst-1", name="ComfyUI", last=1000):
    """Desktop 2 layout: Roaming/Comfy Desktop/{settings,installations}.json + instance-model-paths/<id>.yaml,
    Local/Comfy-Desktop/ComfyUI-Installs/<name>/ComfyUI/{main.py,.venv/Scripts/python.exe}, ComfyUI-Shared/{input,output}."""
    ad, la = tmp_path / "Roaming", tmp_path / "Local"
    root, shared = ad / "Comfy Desktop", la / "Comfy-Desktop" / "ComfyUI-Shared"
    inst_path = la / "Comfy-Desktop" / "ComfyUI-Installs" / name
    comfy = inst_path / "ComfyUI"
    (comfy / ".venv" / "Scripts").mkdir(parents=True)
    (comfy / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    (comfy / "main.py").write_text("")
    (root / "instance-model-paths").mkdir(parents=True, exist_ok=True)
    (root / "instance-model-paths" / f"{inst_id}.yaml").write_text("comfy.desktop_0:\n  base_path: 'x'\n")
    for d in ("input", "output", "models"):
        (shared / d).mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(json.dumps({
        "modelsDirs": [str(shared / "models")], "inputDir": str(shared / "input"), "outputDir": str(shared / "output"),
        "installDir": str(la / "Comfy-Desktop" / "ComfyUI-Installs")}))
    insts = json.loads((root / "installations.json").read_text()) if (root / "installations.json").exists() else []
    insts += [{"id": inst_id, "name": name, "installPath": str(inst_path), "sourceId": "standalone",
               "status": "installed", "launchArgs": launch_args, "lastLaunchedAt": last},
              {"id": "inst-cloud", "name": "Comfy Cloud", "sourceId": "cloud", "status": "installed"}]
    (root / "installations.json").write_text(json.dumps(insts))
    return ad, la, comfy, shared


def test_desktop_app_is_started_when_comfy_is_down(tmp_path, monkeypatch, no_desktop2):
    """Older ComfyUI Desktop (no Desktop 2 install): the app is started, it runs its own server."""
    _reset()
    exe = tmp_path / "Comfy Desktop.exe"
    exe.write_bytes(b"")
    started = []
    alive = iter([False] + [True] * 50)
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: next(alive))
    monkeypatch.setattr(cl.time, "sleep", lambda s: None)
    st = cl.start_comfy({"comfy_flavor": "desktop", "comfy_desktop_exe": str(exe)},
                        popen=lambda argv, **k: started.append(argv))
    assert st["state"] == "starting" and started == [[str(exe)]] and st["how"] == "ComfyUI Desktop"
    for _ in range(100):
        if cl.STATE["state"] != "starting":
            break
        time.sleep(0.01)
    assert cl.STATE["state"] == "running"


def test_desktop2_install_found_like_desktop_runs_it(tmp_path):
    ad, la, comfy, shared = _fake_desktop2(tmp_path)
    inst = cl.desktop2_install({}, appdata=str(ad), localappdata=str(la))
    assert inst["comfy_dir"] == comfy and inst["python"] == comfy / ".venv" / "Scripts" / "python.exe"
    assert inst["yaml"].name == "inst-1.yaml" and inst["id"] == "inst-1"
    assert inst["input_dir"] == str(shared / "input") and inst["output_dir"] == str(shared / "output")
    assert inst["launch_args"] == ["--enable-manager"]


def test_desktop2_started_headless_with_port_yaml_and_shared_dirs(tmp_path, monkeypatch):
    """Desktop 2 waits for a click in its launcher -> MatteDesk starts the install's own python main.py (no app)."""
    _reset()
    ad, la, comfy, shared = _fake_desktop2(tmp_path)
    monkeypatch.setenv("APPDATA", str(ad))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    monkeypatch.setattr(cl, "desktop_exe", lambda cfg: pytest.fail("the Desktop 2 app must not be launched"))
    started = []
    alive = iter([False] + [True] * 50)
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: next(alive))
    monkeypatch.setattr(cl.time, "sleep", lambda s: None)
    st = cl.start_comfy({"comfy_flavor": "desktop", "comfy_port": 8188, "comfy_host": "127.0.0.1"},
                        popen=lambda argv, cwd=None, creationflags=0, **k: started.append((argv, cwd, creationflags)))
    assert st["state"] == "starting" and st["how"] == cl.HOW_DESKTOP2
    argv, cwd, flags = started[0]
    assert cwd == str(comfy)
    assert argv[:4] == [str(comfy / ".venv" / "Scripts" / "python.exe"), "-s", "main.py", "--enable-manager"]
    assert argv[argv.index("--extra-model-paths-config") + 1].endswith("inst-1.yaml")
    assert argv[argv.index("--input-directory") + 1] == str(shared / "input")
    assert argv[argv.index("--output-directory") + 1] == str(shared / "output")
    assert argv[-4:] == ["--listen", "127.0.0.1", "--port", "8188"]   # last: wins over the install's launchArgs
    assert flags == getattr(cl.subprocess, "CREATE_NEW_CONSOLE", 0)    # own console window with the log
    for _ in range(100):
        if cl.STATE["state"] != "starting":
            break
        time.sleep(0.01)
    assert cl.STATE["state"] == "running"


def test_desktop2_picks_comfy_dir_from_config_else_last_launched(tmp_path):
    ad, la, old, _ = _fake_desktop2(tmp_path, inst_id="inst-old", name="Old", last=100)
    _, _, new, _ = _fake_desktop2(tmp_path, inst_id="inst-new", name="New", last=900)
    assert cl.desktop2_install({}, str(ad), str(la))["comfy_dir"] == new
    chosen = cl.desktop2_install({"comfy_dir": str(old)}, str(ad), str(la))
    assert chosen["comfy_dir"] == old and chosen["yaml"].name == "inst-old.yaml"


def test_desktop2_install_without_venv_or_json(tmp_path):
    """No .venv -> not startable headless (falls back to the app); an install missing from installations.json
    is still found in installDir."""
    ad, la, comfy, _ = _fake_desktop2(tmp_path)
    (comfy / ".venv" / "Scripts" / "python.exe").unlink()
    assert cl.desktop2_install({}, str(ad), str(la)) is None
    (comfy / ".venv" / "Scripts" / "python.exe").write_bytes(b"")
    (ad / "Comfy Desktop" / "installations.json").unlink()
    inst = cl.desktop2_install({}, str(ad), str(la))
    assert inst["comfy_dir"] == comfy and inst["launch_args"] == [] and inst["yaml"].name == "inst-1.yaml"
    assert cl.desktop2_install({}, "", "") is None


def test_cpu_affinity_applied_to_own_console_launch(tmp_path, monkeypatch):
    _reset()
    ad, la, _, _ = _fake_desktop2(tmp_path)
    monkeypatch.setenv("APPDATA", str(ad))
    monkeypatch.setenv("LOCALAPPDATA", str(la))
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: False)
    monkeypatch.setattr(cl.threading, "Thread", lambda **k: type("T", (), {"start": lambda self: None})())
    pinned = []
    monkeypatch.setattr(cl, "_set_affinity", lambda proc, mask: pinned.append((proc, mask)) or True)
    st = cl.start_comfy({"comfy_flavor": "desktop", "comfy_cpu_affinity": "FFFFF0FF"}, popen=lambda argv, **k: "proc")
    assert pinned == [("proc", 0xFFFFF0FF)] and st["affinity"] == "FFFFF0FF"
    assert cl._affinity_mask({}) == 0 and cl._affinity_mask({"comfy_cpu_affinity": "zzz"}) == 0


def test_git_comfy_started_with_main_py_and_port(tmp_path, monkeypatch):
    _reset()
    (tmp_path / "main.py").write_text("")
    started = []
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: False)
    monkeypatch.setattr(cl.threading, "Thread", lambda **k: type("T", (), {"start": lambda self: None})())
    st = cl.start_comfy({"comfy_flavor": "git", "comfy_dir": str(tmp_path), "comfy_port": 8190},
                        popen=lambda argv, cwd=None, **k: started.append((argv, cwd)))
    assert st["how"] == "ComfyUI z gita"
    argv, cwd = started[0]
    assert argv[1:] == ["main.py", "--port", "8190"] and cwd == str(tmp_path)


def test_no_launch_when_running_or_switched_off(monkeypatch):
    _reset()
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: True)
    assert cl.start_comfy({"comfy_flavor": "desktop"}, popen=lambda *a, **k: 1 / 0)["state"] == "running"
    _reset()
    assert cl.autostart({"comfy_autostart": False})["autostart"] is False


def test_missing_desktop_app_is_a_readable_failure(monkeypatch, no_desktop2):
    _reset()
    monkeypatch.setattr(cl, "comfy_alive", lambda cfg, timeout=1.5: False)
    monkeypatch.setattr(cl, "desktop_exe", lambda cfg: None)
    st = cl.start_comfy({"comfy_flavor": "desktop"}, popen=lambda *a, **k: 1 / 0)
    assert st["state"] == "failed" and "Comfy Desktop" in st["error"]
