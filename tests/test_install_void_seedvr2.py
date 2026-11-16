# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor installs SeedVR2 into ComfyUI and VOID into a venv on ComfyUI's Python (user decision 2026-11-10):
default places, ComfyUI's PyTorch kept (pip -c / .pth), VOID inference-only packages, VAE tiling patch, weights,
VOID's memory mode by the card. Synthetic: every process and download is faked."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import installer
from engine.cleanup3d import void_engine as VE

ROOT = Path(__file__).resolve().parent.parent
PREDICT_SNIPPET = ('    vae = AutoencoderKLCogVideoX.from_pretrained(\n        model_name,\n        subfolder="vae"\n'
                   '    ).to(weight_dtype)\n\n    if config.video_model.vae_path:\n')   # as in void-model 2026-11-10


def _comfy(tmp):
    py = tmp / "comfy" / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"x")
    (tmp / "comfy" / "custom_nodes").mkdir(parents=True)
    return {"comfy_dir": str(tmp / "comfy"), "models_dir": str(tmp / "models"), "comfy_python": str(py)}, py


def test_default_places(tmp_path, monkeypatch):
    cfg, py = _comfy(tmp_path)
    monkeypatch.setattr(VE, "VOID_HOME", tmp_path / "bin" / "void")
    s = VE.tool_paths(cfg, "seedvr2")
    assert s["python"] == py and s["repo"] == tmp_path / "comfy" / "custom_nodes" / VE.SEEDVR2_NODE
    assert s["models"] == tmp_path / "models" / "SEEDVR2"
    v = VE.tool_paths(cfg, "void")
    assert v["repo"] == tmp_path / "bin" / "void" / "void-model" and v["models"] == tmp_path / "models" / "void"
    assert v["python"] == VE.venv_python(tmp_path / "bin" / "void" / "venv")
    over = VE.tool_paths({**cfg, "void_repo": str(tmp_path / "x"), "seedvr2_models": str(tmp_path / "m")}, "void")
    assert over["repo"] == tmp_path / "x"
    assert VE.tool_paths({**cfg, "seedvr2_models": str(tmp_path / "m")}, "seedvr2")["models"] == tmp_path / "m"
    assert VE.tool_paths({}, "seedvr2")["python"] is None                     # no ComfyUI: unknown


def test_void_memory_mode_follows_the_card(tmp_path):
    assert VE.void_memory_mode({}, card=24.0) is None                          # the repo's default mode
    assert VE.void_memory_mode({}, card=12.0) == "sequential_cpu_offload"
    assert VE.void_memory_mode({"void_gpu_memory_mode": "model_cpu_offload"}, card=12.0) == "model_cpu_offload"
    v = VE.VoidRunner(tmp_path / "py", tmp_path / "r", tmp_path / "m", gpu_memory_mode="sequential_cpu_offload")
    cmd = v.command(tmp_path / "in", "shot", tmp_path / "out", (1344, 768), 57, 57)
    assert "--config.system.gpu_memory_mode=sequential_cpu_offload" in cmd
    assert not any("gpu_memory_mode" in c for c in VE.VoidRunner(tmp_path / "py", tmp_path / "r", tmp_path / "m")
                   .command(tmp_path / "in", "shot", tmp_path / "out", (1344, 768), 57, 57))


class FakeRun:
    """Answers the installer's subprocess calls like a working machine; records them."""

    def __init__(self, tmp):
        self.calls, self.tmp = [], tmp

    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        out = ""
        if cmd[1:3] == ["clone", "--depth"]:
            f = Path(cmd[-1]) / "inference" / "cogvideox_fun" / "predict_v2v.py"
            f.parent.mkdir(parents=True)
            f.write_text(PREDICT_SNIPPET, encoding="utf-8")
            (Path(cmd[-1]) / "requirements.txt").write_text("torch\nrotary_embedding_torch\n", encoding="utf-8")
        elif "importlib.metadata" in " ".join(cmd):
            out = "torch==2.13.0+cu130\ntorchvision==0.28.0+cu130\n"
        elif cmd[1:3] == ["-m", "venv"]:
            VE.venv_python(Path(cmd[3])).parent.mkdir(parents=True, exist_ok=True)
            VE.venv_python(Path(cmd[3])).write_bytes(b"x")
        elif "torch.__file__" in " ".join(cmd):
            out = str(self.tmp / "comfy_site") + "\n"
        elif "purelib" in " ".join(cmd):
            out = str(self.tmp / "bin" / "void" / "venv" / "Lib" / "site-packages") + "\n"
        elif "cuda.is_available" in " ".join(cmd):
            out = "2.13.0+cu130 True 0.33.1 4.57.1\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def _fake_download(got):
    def dl(url, dst):
        got.append((url, Path(dst)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"w")
    return dl


def test_install_seedvr2_into_comfy(tmp_path, monkeypatch):
    cfg, py = _comfy(tmp_path)
    monkeypatch.setattr(installer, "REPO_DIR", tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    run, got = FakeRun(tmp_path), []
    res = installer.install_seedvr2(cfg, download=_fake_download(got), run=run)
    assert res["success"] and res["restart_comfy"] is False
    clone = next(c for c in run.calls if c[1:2] == ["clone"])
    assert clone[-1] == str(tmp_path / "comfy" / "custom_nodes" / "ComfyUI-SeedVR2_VideoUpscaler")
    pip = next(c for c in run.calls if c[1:4] == ["-m", "pip", "install"])
    assert pip[0] == str(py) and "-c" in pip                                   # ComfyUI's Python, PyTorch pinned
    assert "torch==2.13.0+cu130" in Path(pip[pip.index("-c") + 1]).read_text(encoding="utf-8")
    assert [d.name for _u, d in got] == [f for f, _gb in installer.SEEDVR2_FILES]
    assert all(d.parent == tmp_path / "models" / "SEEDVR2" for _u, d in got)
    assert got[0][0].startswith("https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/")


def test_install_void_on_comfy_python(tmp_path, monkeypatch):
    cfg, py = _comfy(tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda n: "git")
    home = tmp_path / "bin" / "void"
    monkeypatch.setattr(VE, "VOID_HOME", home)
    run, got = FakeRun(tmp_path), []
    tree = {installer.VOID_BASE_HF: [("README.md", 0.0), ("model_index.json", 0.0), ("vae/config.json", 0.0),
                                     ("transformer/diffusion_pytorch_model.safetensors", 10.38)]}
    res = installer.install_void_legacy(cfg, download=_fake_download(got), run=run, tree=lambda r: tree[r], home=home)
    assert res["success"], res["log"]
    # venv from ComfyUI's Python, ComfyUI's site-packages appended through a .pth
    assert [str(py), "-m", "venv", str(home / "venv")] in run.calls
    pth = home / "venv" / "Lib" / "site-packages" / "_matte_desk_comfy.pth"
    assert pth.read_text(encoding="utf-8").strip() == str(tmp_path / "comfy_site")
    pip = next(c for c in run.calls if c[1:4] == ["-m", "pip", "install"])
    assert pip[0] == str(VE.venv_python(home / "venv"))
    assert "diffusers==0.33.1" in pip and "transformers==4.57.1" in pip
    assert not any(a.startswith(("torch==", "deepspeed", "tensorboard")) for a in pip)   # no CPU PyTorch, no training
    assert "-c" in pip
    # VAE tiling patch, once
    src = (home / "void-model" / "inference" / "cogvideox_fun" / "predict_v2v.py").read_text(encoding="utf-8")
    assert src.count("vae.enable_tiling(); vae.enable_slicing()") == 1
    assert "już jest" in installer.patch_void_vae(home / "void-model")
    # weights: base model (no README) + void_pass1 into models/void
    names = [str(d.relative_to(tmp_path / "models" / "void")).replace("\\", "/") for _u, d in got]
    assert names == ["CogVideoX-Fun-V1.5-5b-InP/model_index.json", "CogVideoX-Fun-V1.5-5b-InP/vae/config.json",
                     "CogVideoX-Fun-V1.5-5b-InP/transformer/diffusion_pytorch_model.safetensors",
                     "void_pass1.safetensors"]
    # the installed tree is what the gate expects
    assert VE._legacy_void_rows(cfg)[1][1] == home / "void-model" / VE.VOID_SCRIPT
    assert VE._legacy_void_rows(cfg)[4][1] == tmp_path / "models" / "void" / VE.VOID_TRANSFORMER


def test_install_void_needs_comfy_python(tmp_path):
    with pytest.raises(RuntimeError, match="Pythona ComfyUI"):
        installer.install_void_legacy({"comfy_dir": str(tmp_path)})


def test_cleanup3d_state_carries_the_install_card(tmp_path, monkeypatch):
    """C4 panel (user 2026-11-10): short info + the card's GPU verdict + the install action, not the gate's paths."""
    from engine import vram_fit
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    monkeypatch.setattr(VE, "VOID_HOME", tmp_path / "bin" / "void")
    monkeypatch.setattr(vram_fit, "card_gb", lambda comfy=None: 12.0)
    st = VE.state(tmp_path / "work", {})
    for tool in ("void", "seedvr2"):
        t = st[tool]
        assert not t["available"] and t["install"] == tool and t["partly"] is False
        assert t["info"]["download"] and t["info"]["license"]
    assert st["void"]["vram"]["level"] == "spill" and st["seedvr2"]["vram"]["level"] == "ok"   # 3B fp8 on 12 GB (2026-11-11)
    js = (ROOT / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    assert 'data-c3install="${esc(t.install)}"' in js and "ui.installWithProgress(tool)" in js


def test_void_pip_list_is_inference_only_and_in_sync_with_the_doctor_text():
    for bad in ("deepspeed", "tensorboard", "datasets", "came-pytorch"):
        assert not any(p.startswith(bad) for p in installer.VOID_PIP)
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "install_void: ['void'," in js and "install_seedvr2: ['seedvr2'," in js
    assert "'seedvr2', 'void', 'vace', 'vace_lightning', 'ltx_inpaint', 'comfy_update', 'comfy_rollback', 'prompt_assist']" in js                                          # not part of "Zainstaluj wszystkie"
