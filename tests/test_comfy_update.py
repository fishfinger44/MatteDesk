# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor's „Zaktualizuj ComfyUI” / „Cofnij” for a ComfyUI from git (user request 2026-11-10). A fake git answers:
clean tree -> remembered commit, pull --ff-only, requirements with PyTorch pinned; local changes -> refused, nothing
run; rollback = reset --keep to the remembered commit."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import comfy_update as CU


class FakeGit:
    def __init__(self, dirty="", behind="503"):
        self.calls, self.dirty, self.behind, self.head = [], dirty, behind, "a" * 40

    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        out = ""
        a = cmd[3:] if cmd[1:2] == ["-C"] else cmd[1:]
        if a[:2] == ["rev-parse", "--abbrev-ref"]:
            out = "master"
        elif a[:2] == ["rev-parse", "--short"]:
            out = self.head[:7]
        elif a[:1] == ["rev-parse"]:
            out = self.head
        elif a[:1] == ["status"]:
            out = self.dirty
        elif a[:2] == ["rev-list", "--count"]:
            out = self.behind
        elif a[:1] == ["pull"]:
            self.head = "b" * 40
        elif "importlib.metadata" in " ".join(cmd):
            out = "torch==2.13.0+cu130\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    d = tmp_path / "ComfyUI"
    (d / ".git").mkdir(parents=True)
    (d / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    py = tmp_path / "py.exe"
    py.write_bytes(b"x")
    monkeypatch.setattr(CU, "STATE_FILE", tmp_path / "bin" / "comfy_update.json")
    monkeypatch.setattr(CU, "REPO_DIR", tmp_path)
    monkeypatch.setattr(CU, "_git", lambda cfg: "git")
    return {"comfy_dir": str(d), "comfy_python": str(py)}, d


def test_info_behind_and_clean(comfy):
    cfg, d = comfy
    i = CU.info(cfg, FakeGit(), now=1e9)
    assert i["kind"] == "git" and i["branch"] == "master" and i["behind"] == 503 and i["dirty"] == []
    assert CU.info({"comfy_dir": str(d.parent / "none")}, FakeGit())["kind"] == "other"


def test_update_remembers_pulls_ff_only_and_pins_torch(comfy):
    cfg, d = comfy
    run = FakeGit()
    res = CU.update(cfg, run)
    assert res["success"] and res["restart_comfy"]
    pull = next(c for c in run.calls if "pull" in c)
    assert pull[-3:] == ["--ff-only", "origin", "master"]
    pip = next(c for c in run.calls if c[1:4] == ["-m", "pip", "install"])
    assert "-c" in pip and "torch==2.13.0+cu130" in Path(pip[pip.index("-c") + 1]).read_text(encoding="utf-8")
    st = CU.load_state()
    assert st["before"] == "a" * 40 and st["dir"] == str(d)
    i = CU.info(cfg, FakeGit(), fetch=False)
    assert i["can_rollback"]
    rb = FakeGit()
    assert CU.rollback(cfg, rb)["success"]
    assert any(c[3:] == ["reset", "--keep", "a" * 40] for c in rb.calls)
    assert "before" not in CU.load_state()


def test_local_changes_block_the_update(comfy):
    cfg, _ = comfy
    run = FakeGit(dirty=" M comfy/model_management.py")
    with pytest.raises(RuntimeError, match="lokalne zmiany"):
        CU.update(cfg, run)
    assert not any("pull" in c for c in run.calls) and not CU.load_state().get("before")


def test_desktop_is_not_updated_here(tmp_path):
    with pytest.raises(RuntimeError, match="Desktop"):
        CU.update({"comfy_dir": str(tmp_path)})


def test_doctor_and_ui_wiring():
    root = Path(__file__).resolve().parent.parent
    from engine import installer
    assert {"comfy_update", "comfy_rollback"} <= set(installer.INSTALL_ACTIONS)
    js = (root / "static" / "app.js").read_text(encoding="utf-8")
    assert "install_comfy_update: ['comfy_update'," in js and "install_comfy_rollback: ['comfy_rollback'," in js
    nb = js.split("const notInBatch = [", 1)[1].split("]", 1)[0]
    assert "'comfy_update'" in nb and "'comfy_rollback'" in nb             # never in „Zainstaluj wszystkie”
    import doctor
    assert doctor.COMFY_UPDATE_ITEM == "Aktualizacja ComfyUI"


def test_update_items_lead_the_report(monkeypatch):
    """User 2026-11-11: the ComfyUI update / rollback rows sit at the very top of the Doctor."""
    import doctor
    from doctor import CheckItem, EnvironmentDoctor
    rows = [CheckItem(name=doctor.COMFY_ROLLBACK_ITEM, category="ComfyUI", status="info", message="r"),
            CheckItem(name=doctor.COMFY_UPDATE_ITEM, category="ComfyUI", status="info", message="u")]
    monkeypatch.setattr(EnvironmentDoctor, "check_comfy_update", lambda self: rows)
    rep = EnvironmentDoctor(simulation={"simulate_comfy_offline": True}).run_all_checks()
    assert [i.name for i in rep.items[:2]] == [doctor.COMFY_UPDATE_ITEM, doctor.COMFY_ROLLBACK_ITEM]
