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
    monkeypatch.setattr(tracking, "check_worker", lambda *a, **k: (True, "ok cpu"))
    ckpt = tmp_path / "scaled_offline.pth"
    ckpt.write_bytes(b"w")
    # persistent=False: this fake worker speaks the one-shot argv protocol (the served loop is tested with the real
    # worker in MD_COTRACKER_FAKE mode, tests/test_cotracker_worker_serve.py)
    t = tracking.CoTracker3Tracker(ckpt, python=sys.executable, max_side=100, persistent=False)
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


def test_install_hf_folder_downloads_all_files(tmp_path, monkeypatch):
    """HF_FOLDERS (multi-file weights): every file lands in its folder (empty since MiniMax-Remover was dropped
    2026-11-09 – the mechanism stays for the next such model)."""
    got = []

    def fake_download(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setitem(installer.HF_FOLDERS, "demo_weights",
                        ("org/demo", ["transformer/config.json", "vae/config.json"], "Demo"))
    r = installer.install_model({"models_dir": str(tmp_path)}, "demo_weights", download=fake_download)
    assert r["success"] and len(got) == 2 and all("org/demo" in u for u in got)
    assert (tmp_path / "Demo" / "transformer" / "config.json").exists() and (tmp_path / "Demo" / "vae" / "config.json").exists()


def test_node_packages_survive_public_export_unchanged():
    """tools/export_public.py renames shot names in text files; inside our Comfy node packages that would make the
    public copy differ from nodes installed from the dev repo (Doctor: "different version than in the repo")."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    if not (root / "tools" / "export_public.py").exists():
        pytest.skip("export_public.py is not part of the public copy")
    spec = importlib.util.spec_from_file_location("export_public", root / "tools" / "export_public.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    changed = []
    for p in sorted((root / "handoff" / "comfy" / "custom_nodes").rglob("*")):
        if p.is_file() and p.suffix in ep.TEXT_EXT and "__pycache__" not in p.parts:
            text = p.read_text(encoding="utf-8")
            if ep.neutralize(text) != text:
                changed.append(str(p.relative_to(root)))
    assert not changed, changed


def test_readme_download_button_points_at_current_setup():
    """2026-10-17: the README button downloads the installer file itself; the public export sets the version in the link."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    if not (root / "tools" / "export_public.py").exists():
        pytest.skip("export_public.py is not part of the public copy")
    spec = importlib.util.spec_from_file_location("export_public", root / "tools" / "export_public.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    for name in ("README.md", "README.pl.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert ep.SETUP_LINK.search(text), name
    out = ep.setup_link("x releases/download/v0.31/MatteDesk-0.31-setup.exe y", "0.4")
    assert out == "x releases/download/v0.4/MatteDesk-0.4-setup.exe y"


# --- Cleanup 3D (docs/PLAN_CLEANUP_3D.md, P3): FLUX 3 Image Klein, LoRA Restore, COLMAP (CUDA) -----------------------
@pytest.fixture
def small_klein(monkeypatch):
    from engine import stage_models as G
    monkeypatch.setattr(G, "KLEIN_FILES", {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.KLEIN_FILES.items()})
    monkeypatch.setattr(G, "KLEIN_LORA", {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.KLEIN_LORA.items()})


def _fake_dl(got):
    def dl(url, dst):
        got.append((url, dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"w" * 8)
    return dl


def test_install_flux3_klein_downloads_three_files_once(tmp_path, small_klein):
    got = []
    cfg = {"models_dir": str(tmp_path / "models")}
    res = installer.install_flux3_klein(cfg, download=_fake_dl(got))
    base = "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/"
    assert res["success"] and [u for u, _ in got] == [base + "vae/flux2-vae.safetensors",
                                                     base + "text_encoders/qwen3vl_4b_bf16.safetensors",
                                                     base + "diffusion_models/flux3-image-klein-bf16.safetensors"]
    assert [d.relative_to(tmp_path / "models").as_posix() for _, d in got] == [
        "vae/flux2-vae.safetensors", "text_encoders/qwen3vl_4b_bf16.safetensors",
        "diffusion_models/flux3-image-klein-bf16.safetensors"]
    again = installer.install_flux3_klein(cfg, download=lambda *a: pytest.fail("must not download twice"))
    assert sum("już jest" in l for l in again["log"]) == 3


def test_install_flux3_klein_gated_messages(tmp_path, small_klein, monkeypatch):
    import urllib.error
    cfg = {"models_dir": str(tmp_path / "models")}
    monkeypatch.setattr(installer, "hf_token", lambda: None)
    with pytest.raises(RuntimeError) as e:
        installer.install_flux3_klein(cfg)
    assert "hf auth login" in str(e.value) and "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy" in str(e.value)
    assert "„gated”" in str(e.value)

    def denied(url, dst):   # what engine/download.py raises on 401 / 403 from huggingface.co
        raise RuntimeError("Hugging Face odmówił dostępu (401/403): zaloguj się („hf auth login”) i kliknij "
                           "„Agree and Access” na stronie modelu (repo gated).")
    with pytest.raises(RuntimeError) as e:
        installer.install_flux3_klein(cfg, download=denied)
    msg = str(e.value)
    assert "odmówił dostępu do FLUX 3 Image Klein" in msg and "zaakceptuj dostęp" in msg
    assert "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy" in msg

    def http403(url, dst):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
    with pytest.raises(RuntimeError, match="odmówił dostępu do FLUX 3 Image Klein"):
        installer.install_flux3_klein(cfg, download=http403)
    with pytest.raises(RuntimeError, match="models_dir"):
        installer.install_flux3_klein({}, download=_fake_dl([]))


def test_install_klein_restore_lora(tmp_path, small_klein):
    got = []
    res = installer.install_klein_restore_lora({"models_dir": str(tmp_path / "models")}, download=_fake_dl(got))
    assert res["success"] and got[0][0] == ("https://huggingface.co/ausboss/FLUX-3-Image-Klein-Restore-LoRA/resolve/main/"
                                            "flux-3-image-klein-restore.safetensors")
    assert got[0][1] == tmp_path / "models" / "loras" / "flux-3-image-klein-restore.safetensors"


def test_install_actions_table():
    assert set(installer.INSTALL_ACTIONS) == {"flux3_klein", "klein_restore_lora", "qwen21", "colmap", "seedvr2", "void",
                                              "vace", "vace_lightning", "comfy_update", "comfy_rollback",
                                              "prompt_assist", "ltx_inpaint"}


def test_install_qwen21_downloads_three_files_once(tmp_path, monkeypatch):
    """Qwen-Image-2.1 (Cleanup 3D, 2026-11-10): Comfy-Org/Qwen-Image-2.1, resumable, present files skipped."""
    import urllib.error
    from engine import stage_models as G
    monkeypatch.setattr(G, "QWEN21_FILES", {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.QWEN21_FILES.items()})
    got = []
    cfg = {"models_dir": str(tmp_path / "models")}
    res = installer.install_qwen21(cfg, download=_fake_dl(got))
    base = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/"
    assert res["success"] and [u for u, _ in got] == [base + "vae/qwen_image_2.1_vae_bf16.safetensors",
                                                     base + "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                                                     base + "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"]
    assert [d.relative_to(tmp_path / "models").as_posix() for _, d in got] == [
        "vae/qwen_image_2.1_vae_bf16.safetensors", "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"]
    again = installer.install_qwen21(cfg, download=lambda *a: pytest.fail("must not download twice"))
    assert sum("już jest" in l for l in again["log"]) == 3
    assert installer.INSTALL_ACTIONS["qwen21"] is installer.install_qwen21

    def http403(url, dst):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
    with pytest.raises(RuntimeError, match="odmówił dostępu do Qwen-Image-2.1"):
        installer.install_qwen21({"models_dir": str(tmp_path / "m2")}, download=http403)
    with pytest.raises(RuntimeError, match="models_dir"):
        installer.install_qwen21({}, download=_fake_dl([]))


def _colmap_zip(path: Path, with_exe: bool = True) -> None:
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("COLMAP-4.2.1-windows-cuda/COLMAP.bat", "@echo off")
        if with_exe:
            z.writestr("COLMAP-4.2.1-windows-cuda/bin/colmap.exe", "MZ")
        z.writestr("COLMAP-4.2.1-windows-cuda/lib/x.dll", "dll")


def test_install_colmap_unpacks_with_checksum(tmp_path):
    src = tmp_path / "src.zip"
    _colmap_zip(src)
    got = []

    def dl(url, dst, sha):
        got.append((url, sha))
        dst.write_bytes(src.read_bytes())
    root = tmp_path / "bin" / "colmap"
    res = installer.install_colmap({}, download=dl, root=root)
    assert got == [(installer.COLMAP_URL, installer.COLMAP_SHA256)]
    assert installer.COLMAP_URL.endswith("/4.2.1/colmap-x64-windows-cuda.zip")
    assert Path(res["path"]) == root / "bin" / "colmap.exe" and (root / "lib" / "x.dll").exists()
    assert not (tmp_path / "bin" / installer.COLMAP_ASSET).exists() and not (tmp_path / "bin" / "colmap_unpack").exists()
    again = installer.install_colmap({}, download=lambda *a: pytest.fail("already there"), root=root)
    assert "już jest" in again["log"][0]
    assert installer.colmap_exe({"colmap_path": str(src)}, root=tmp_path / "none") == src   # config path wins


def test_install_colmap_without_exe_fails(tmp_path):
    src = tmp_path / "src.zip"
    _colmap_zip(src, with_exe=False)
    with pytest.raises(RuntimeError, match="colmap.exe"):
        installer.install_colmap({}, download=lambda url, dst, sha: dst.write_bytes(src.read_bytes()),
                                 root=tmp_path / "bin" / "colmap")
    assert not (tmp_path / "bin" / "colmap").exists()


def test_doctor_offers_cleanup3d_installs(tmp_path, monkeypatch):
    from doctor import EnvironmentDoctor
    monkeypatch.setattr(installer, "COLMAP_DIR", tmp_path / "bin" / "colmap")
    (tmp_path / "custom_nodes").mkdir()
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models"), "colmap_path": ""})
    items = {it.name: it for it in doc.run_all_checks().items}
    k, lo, col = items["Model: FLUX 3 Image Klein"], items["Model: LoRA Restore (FLUX 3 Image Klein)"], items["COLMAP (CUDA)"]
    assert k.status == "warning" and k.action == "install_flux3_klein"
    assert lo.status == "warning" and lo.action == "install_klein_restore_lora"
    assert col.status == "info" and col.action == "install_colmap" and col.license == "BSD" and not col.license_limited
    for it in (k, lo):
        assert it.license.startswith("FLUX Kommunity License") and "5 mln USD" in it.license
        assert it.license_limited and not it.license_nc
        assert it.fix_suggestion.startswith("Wymagane teraz: NIE – tylko dogenerowanie klucza w trybie Czyszczenie")
    assert any("black-forest-labs/flux-3-image-klein-comfy" in l for l in k.links)
    assert any("ausboss/FLUX-3-Image-Klein-Restore-LoRA" in l for l in lo.links)
    assert any("colmap/colmap/releases" in l for l in col.links)
    v = items["Wersja ComfyUI (FLUX 3 Image Klein)"]
    assert v.status == "info" and "nie odpowiada" in v.message


def test_doctor_offers_qwen21_install(tmp_path, monkeypatch):
    """The Doctor lists Qwen-Image-2.1 (optional, non-commercial licence, own install button, links)."""
    from doctor import EnvironmentDoctor
    monkeypatch.setattr(installer, "COLMAP_DIR", tmp_path / "bin" / "colmap")
    (tmp_path / "custom_nodes").mkdir()
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path), "models_dir": str(tmp_path / "models"), "colmap_path": ""})
    q = {it.name: it for it in doc.run_all_checks().items}["Model: Qwen-Image-2.1"]
    assert q.status == "warning" and q.action == "install_qwen21"
    assert q.license.startswith("Qwen Research License") and q.license_nc and not q.license_limited
    assert q.fix_suggestion.startswith("Wymagane teraz: NIE – tylko dogenerowanie klucza w trybie Czyszczenie")
    assert "17,3 GB" in q.fix_suggestion and "diffusion_models/qwen_image_2.1_int8_convrot.safetensors" in q.message
    assert any("huggingface.co/Comfy-Org/Qwen-Image-2.1" in l for l in q.links)
    from engine import stage_models as G
    md = tmp_path / "models"
    for folder, fn, *_ in G.QWEN21_FILES.values():
        (md / folder).mkdir(parents=True, exist_ok=True)
        (md / folder / fn).write_bytes(b"x" * 4)
    monkeypatch.setattr(G, "QWEN21_FILES", {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.QWEN21_FILES.items()})
    ok = doc.check_qwen21_model(md)
    assert ok.status == "ok" and ok.fix_suggestion is None and "Qwen-Image-2.1 obecny" in ok.message
