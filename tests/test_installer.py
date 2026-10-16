# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor "Zainstaluj": our node packages -> custom_nodes (backup of a different copy), CoTracker3 pip + weights."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine import installer


def _ok(*a, **k):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_install_nodes_copies_and_backs_up(tmp_path, monkeypatch):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes" / "other-pack").mkdir(parents=True)
    old = comfy / "custom_nodes" / "ComfyUI-PlateMatte"
    old.mkdir()
    (old / "__init__.py").write_text("# old")
    monkeypatch.setattr(installer, "BACKUP_DIR", tmp_path / "backups")
    calls = []
    res = installer.install_nodes({"comfy_dir": str(comfy)}, run=lambda *a, **k: calls.append(a) or _ok())
    assert res["status"] == {n: "same" for n in installer.NODE_PACKAGES}
    assert (comfy / "custom_nodes" / "other-pack").exists()          # other packages untouched
    baks = list((tmp_path / "backups").iterdir())
    assert len(baks) == 1 and (baks[0] / "__init__.py").read_text() == "# old"  # backup outside custom_nodes
    # second run: nothing to do
    res2 = installer.install_nodes({"comfy_dir": str(comfy)}, run=_ok)
    assert any("pliki aktualne" in l for l in res2["log"])
    assert len(list((tmp_path / "backups").iterdir())) == 1


def test_install_nodes_requires_comfy(tmp_path):
    with pytest.raises(RuntimeError):
        installer.install_nodes({"comfy_dir": str(tmp_path / "nope")})


def test_install_cotracker_downloads_weights(tmp_path, monkeypatch):
    got = []
    res = installer.install_cotracker({"models_dir": str(tmp_path / "models")}, run=_ok, python="comfy-python",
                                      download=lambda url, dst: got.append((url, dst)) or dst.parent.mkdir(parents=True) or dst.write_bytes(b"w"))
    assert res["success"] and got[0][0] == installer.COTRACKER_WEIGHTS_URL
    assert Path(res["weights"]) == tmp_path / "models" / "cotracker" / "scaled_offline.pth"
    res2 = installer.install_cotracker({"models_dir": str(tmp_path / "models")}, run=_ok, python="comfy-python",
                                       download=lambda *a: 1 / 0)
    assert any("już są" in l for l in res2["log"])


def test_doctor_offers_install_button(tmp_path):
    from doctor import EnvironmentDoctor
    (tmp_path / "custom_nodes").mkdir()
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models")})
    items = {it.name: it for it in doc.run_all_checks().items}
    assert items["Węzły MatteDesk (pliki)"].action == "install_nodes"
    assert items["Model: CoTracker3"].action == "install_cotracker"


def test_install_model_downloads_to_models_dir(tmp_path):
    got = {}

    def fake_download(url, dst):
        got["url"] = url
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    cfg = {"models_dir": str(tmp_path / "models"), "sam3_checkpoint": "sam3.1_multiplex_fp16.safetensors"}
    r = installer.install_model(cfg, "sam3", download=fake_download)
    assert r["success"] and (tmp_path / "models" / "checkpoints" / "sam3.1_multiplex_fp16.safetensors").exists()
    assert got["url"].startswith("https://huggingface.co/Comfy-Org/sam3.1/")
    again = installer.install_model(cfg, "sam3", download=lambda *a: pytest.fail("must not download twice"))
    assert "już jest" in again["log"][0]


def test_install_git_node_clones_once(tmp_path, monkeypatch):
    (tmp_path / "custom_nodes").mkdir()
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    cfg = {"comfy_dir": str(tmp_path)}
    r = installer.install_git_node(cfg, "matanyone2_node", run=fake_run)
    assert r["success"] and (tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2").is_dir()
    assert calls[0][:3] == ["git", "clone", "--depth"]
    installer.install_git_node(cfg, "matanyone2_node", run=lambda *a, **k: pytest.fail("already cloned"))


def test_download_reports_progress(tmp_path):
    import http.server
    import threading
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "f.bin").write_bytes(b"a" * (3 << 20))

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path / "srv"), **k)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        installer.progress_begin("test")
        installer._download(f"http://127.0.0.1:{srv.server_port}/f.bin", tmp_path / "out" / "f.bin")
    finally:
        srv.shutdown()
    assert (tmp_path / "out" / "f.bin").stat().st_size == 3 << 20
    assert installer.PROGRESS["file"] == "f.bin" and installer.PROGRESS["done"] == installer.PROGRESS["total"] == 3 << 20
    installer.progress_end(True)
    assert installer.PROGRESS["active"] is False


def test_vram_patch_is_warning_on_big_gpu(tmp_path):
    from doctor import EnvironmentDoctor
    nodes = tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2"
    nodes.mkdir(parents=True)
    (nodes / "nodes.py").write_text("x = 1\n", encoding="utf-8")
    doc = EnvironmentDoctor()
    doc.config["comfy_dir"] = str(tmp_path)
    doc.comfy_vram_gb = 24.0
    assert doc.check_matanyone_vram_patch().status == "warning"
    doc.comfy_vram_gb = 12.0
    assert doc.check_matanyone_vram_patch().status == "error"
    doc.comfy_vram_gb = None
    assert doc.check_matanyone_vram_patch().status == "error"


def test_install_nodes_installs_deps_even_when_files_are_current(tmp_path, monkeypatch):
    """Package copied earlier without its dependencies (IMPORT FAILED in Comfy): "Zainstaluj" must still run pip."""
    cn = tmp_path / "comfy" / "custom_nodes"
    cn.mkdir(parents=True)
    for name in installer.NODE_PACKAGES:
        installer.shutil.copytree(installer.NODE_SRC / name, cn / name)
    py = tmp_path / "comfy" / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("")
    calls = []
    r = installer.install_nodes({"comfy_dir": str(tmp_path / "comfy")},
                                run=lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert r["success"] and any("ComfyUI-PlateMatte" in " ".join(c) for c in calls)
    assert all(c[1:4] == ["-m", "pip", "install"] for c in calls)


def test_platematte_declares_its_dependencies():
    req = (installer.NODE_SRC / "ComfyUI-PlateMatte" / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "opencv" in req and "openexr" in req


def test_doctor_reads_import_errors_from_comfy_log(monkeypatch):
    import io
    from doctor import EnvironmentDoctor
    log = ('2026 - [WARNING] Cannot import C:\\X\\custom_nodes\\ComfyUI-PlateMatte module for custom nodes: '
           "No module named 'cv2'\n2026 - [INFO] ok\n")

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: R(('"' + log + '"').encode()))
    doc = EnvironmentDoctor()
    assert doc.comfy_import_errors("http://x") == {"ComfyUI-PlateMatte": "No module named 'cv2'"}


def test_pending_restart_cleared_when_comfy_started_later(monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "PENDING_FILE", tmp_path / "pending.json")
    installer.PENDING_RESTART.clear()
    installer.PENDING_RESTART["nodes"] = 1000.0
    monkeypatch.setattr(installer, "comfy_started_at", lambda url, timeout=2.5: 900.0)    # Comfy older than install
    assert installer.pending_restart("http://x") == {"nodes": 1000.0}
    monkeypatch.setattr(installer, "comfy_started_at", lambda url, timeout=2.5: 1100.0)   # restarted after it
    assert installer.pending_restart("http://x") == {}


def test_doctor_offers_restart_instead_of_install(monkeypatch, tmp_path):
    from doctor import CheckItem, EnvironmentDoctor
    monkeypatch.setattr(installer, "PENDING_FILE", tmp_path / "pending.json")
    installer.PENDING_RESTART.clear()
    installer.PENDING_RESTART["matanyone2_node"] = 1000.0
    monkeypatch.setattr(installer, "comfy_started_at", lambda url, timeout=2.5: 900.0)
    doc = EnvironmentDoctor(simulation={"simulate_flavor": "desktop"})
    doc.config["comfy_dir"] = str(tmp_path)
    doc.config["models_dir"] = str(tmp_path / "models")
    item = CheckItem(name="ComfyUI-MatAnyone2", category="ComfyUI Nodes", status="error", message="Brak węzłów")
    monkeypatch.setattr(doc, "check_comfy_nodes", lambda url: [item])
    report = doc.run_all_checks()
    it = next(i for i in report.items if i.name == "ComfyUI-MatAnyone2")
    assert it.action == "restart_comfy" and "Zainstalowane" in it.message
    installer.PENDING_RESTART.clear()


def test_comfy_started_at_parses_first_log_line(monkeypatch):
    import io

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(installer.urllib.request, "urlopen",
                        lambda *a, **k: R(b'"2026-10-14T15:04:16.328611 - [INFO] start\n2026-10-14T15:09:00 - x"'))
    import datetime
    assert installer.comfy_started_at("http://x") == datetime.datetime(2026, 10, 14, 15, 4, 16, 328611).timestamp()


def test_cotracker_package_goes_to_comfy_python_without_deps(tmp_path):
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=1 if cmd[1] == "-c" else 0, stdout="", stderr="")   # not importable yet
    res = installer.install_cotracker({"models_dir": str(tmp_path)}, run=run, python="C:/comfy/python.exe",
                                      download=lambda url, dst: dst.parent.mkdir(parents=True) or dst.write_bytes(b"w"))
    pip = [c for c in calls if "pip" in c][0]
    assert res["success"] and pip[0] == "C:/comfy/python.exe" and "--no-deps" in pip


def test_cotracker_runs_in_comfy_python_through_the_worker(tmp_path, monkeypatch):
    """The tracker hands frames (downscaled, in a shared pool) to a separate interpreter (ComfyUI's) and scales the
    tracks back."""
    import sys

    from engine import tracking
    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import sys, numpy as np\n"
        "d = np.load(sys.argv[1]); out = {'device': np.array('cpu')}\n"
        "for i in range(int(d['n_videos'])):\n"
        "    q = d[f'queries_{i}']; T, N = len(d[f'frames_idx_{i}']), q.shape[0]   # frames: shared pool\n"
        "    assert d['pool_0'].shape[1:] == (50, 100, 3)\n"
        "    out[f'tracks_{i}'] = np.repeat(q[None, :, 1:], T, axis=0).astype(np.float32)\n"
        "    out[f'visible_{i}'] = np.ones((T, N), bool)\n"
        "np.savez(sys.argv[2], **out)\n", encoding="utf-8")
    monkeypatch.setattr(tracking, "WORKER", fake)
    monkeypatch.setitem(tracking._CHECK_CACHE, sys.executable, (True, "cpu"))
    ckpt = tmp_path / "scaled_offline.pth"
    ckpt.write_bytes(b"w")
    t = tracking.CoTracker3Tracker(ckpt, python=sys.executable, max_side=100)
    frames = [np.zeros((200, 400, 3), np.uint8)] * 3            # downscaled x0.25 for the worker
    res = t.track(frames, np.array([[0, 40.0, 20.0]], np.float32))
    assert res.engine == "cotracker3" and res.tracks.shape == (3, 1, 2)
    assert np.allclose(res.tracks[2, 0], [40.0, 20.0])          # scaled back to full resolution

def test_pending_restart_survives_a_matte_desk_restart(monkeypatch, tmp_path):
    """Installed before a MatteDesk restart -> still 'Zrestartuj ComfyUI', not 'Zainstaluj' again."""
    monkeypatch.setattr(installer, "PENDING_FILE", tmp_path / "pending.json")
    installer.PENDING_RESTART.clear()
    installer.mark_pending_restart("nodes")
    installer.PENDING_RESTART.clear()                       # new server process: memory empty ...
    installer.PENDING_RESTART.update(installer._load_pending())   # ... loaded from the file at import
    monkeypatch.setattr(installer, "comfy_started_at", lambda url, timeout=2.5: 1.0)   # Comfy older than the install
    assert "nodes" in installer.pending_restart("http://x")
    monkeypatch.setattr(installer, "comfy_started_at", lambda url, timeout=2.5: 4e9)   # Comfy restarted later
    assert installer.pending_restart("http://x") == {} and not (tmp_path / "pending.json").exists()


def test_install_minimax_weights_downloads_all_files(tmp_path):
    got = []

    def fake_download(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    r = installer.install_model({"models_dir": str(tmp_path)}, "minimax_weights", download=fake_download)
    assert r["success"] and len(got) == 5 and all("zibojia/minimax-remover" in u for u in got)
    assert (tmp_path / "MiniMax-Remover" / "transformer" / "diffusion_pytorch_model.safetensors").exists()
    assert (tmp_path / "MiniMax-Remover" / "vae" / "config.json").exists()
