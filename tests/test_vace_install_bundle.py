"""Wan 2.2 Fun VACE installs WITH its Lightning LoRAs (user 2026-11-13: „szybko – 4 kroki” must never silently run 20)."""
from pathlib import Path

from engine import installer
from engine import vace_fill as VF


def test_install_vace_fetches_lightning(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "comfy_python", lambda cfg: Path("python.exe"))
    monkeypatch.setattr(installer, "torch_constraints", lambda *a, **k: None)
    monkeypatch.setattr(installer, "install_git_node", lambda *a, **k: {"success": True, "log": []})
    got = []
    r = installer.install_vace({"models_dir": str(tmp_path)}, download=lambda u, d: got.append(Path(d).name))
    assert r["success"]
    for name in VF.LIGHTNING.values():
        assert name in got
    assert VF.HIGH in got and VF.LOW in got
