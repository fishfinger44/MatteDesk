# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor: ComfyUI-MatAnyone2 registered in /object_info but its lazily imported Python packages (matanyone2, hydra-core,
thinplate) missing -> error + "Zainstaluj"; the installer runs the pack's install.py with ComfyUI's Python."""

from pathlib import Path
from types import SimpleNamespace

from doctor import MATANYONE_PY_MODULES, EnvironmentDoctor
from engine import installer

INFO = {"LoadMatAnyone2Model": {"input": {}}}


def _item(tmp_path, modules):
    (tmp_path / "custom_nodes").mkdir(exist_ok=True)
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True, "object_info": INFO, "comfy_modules": modules})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    return {it.name: it for it in doc.run_all_checks().items}["ComfyUI-MatAnyone2"]


def test_matanyone_node_without_package_is_error_with_install(tmp_path):
    it = _item(tmp_path, {"hydra": True, "thinplate": True})
    assert it.status == "error" and it.action == "install_matanyone2_node"
    assert it.details["missing_modules"] == ["matanyone2"] and "Wymagane teraz: TAK" in it.fix_suggestion


def test_matanyone_node_with_packages_is_ok(tmp_path):
    it = _item(tmp_path, {m: True for m in MATANYONE_PY_MODULES})
    assert it.status == "ok" and it.action is None


def test_install_git_node_runs_install_py(tmp_path, monkeypatch):
    pack = tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2"
    pack.mkdir(parents=True)
    (pack / "requirements.txt").write_text("einops\n")
    (pack / "install.py").write_text("# pip install matanyone2 ...\n")
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    monkeypatch.setattr(installer, "comfy_python", lambda cfg: Path("comfy-python"))
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw.get("cwd")))
        envs.append(kw.get("env") or {})
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    envs = []
    r = installer.install_git_node({"comfy_dir": str(tmp_path)}, "matanyone2_node", run=fake_run)
    assert r["success"] and r["restart_comfy"]
    assert calls[-1] == (["comfy-python", str(pack / "install.py")], str(pack))     # after requirements
    assert any("install.py -> OK" in l for l in r["log"])
    # newest hatchling refuses MatAnyone2's pyproject -> install.py's pip builds run with our build constraints
    bc = Path(envs[-1]["PIP_BUILD_CONSTRAINT"])
    assert bc.exists() and "hatchling<1.26" in bc.read_text(encoding="utf-8")
    # install.py failing -> not a success, its output in the log
    r2 = installer.install_git_node({"comfy_dir": str(tmp_path)}, "matanyone2_node",
                                    run=lambda cmd, **k: SimpleNamespace(returncode=1 if cmd[-1].endswith("install.py") else 0,
                                                                         stdout="  ValueError: A second file is being added\n" + "x\n" * 900,
                                                                         stderr="subprocess.CalledProcessError: Command"))
    assert not r2["success"] and any(l.startswith("przyczyna:") and "A second file" in l for l in r2["log"])


def test_missing_packages_keep_install_not_restart(tmp_path, monkeypatch):
    """Comfy running, pack folder on disk, an earlier install pending a restart: the node is loaded, its packages are
    missing -> still "Zainstaluj" (a restart never installs them – SH050 2026-10-15 looped on "Zrestartuj ComfyUI")."""
    from doctor import CheckItem
    (tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2").mkdir(parents=True)
    monkeypatch.setattr(installer, "pending_restart", lambda url: {"matanyone2_node": 1.0})
    doc = EnvironmentDoctor(simulation={"object_info": INFO, "comfy_modules": {}})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    monkeypatch.setattr(doc, "check_comfy_connectivity",
                        lambda: (CheckItem(name="ComfyUI Service", category="ComfyUI", status="ok", message=""), "http://comfy.test"))
    it = {i.name: i for i in doc.run_all_checks().items}["ComfyUI-MatAnyone2"]
    assert it.status == "error" and it.action == "install_matanyone2_node"
    assert "ZRESTARTUJ" not in it.message and not (it.details or {}).get("needs_restart")
