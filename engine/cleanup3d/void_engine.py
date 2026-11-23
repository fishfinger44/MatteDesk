# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P12 – VOID and SeedVR2 as key engines (docs/PLAN_CLEANUP_3D.md P12, user decisions 2026-11-10).

VOID (Netflix, video object removal on CogVideoX-Fun-V1.5-5b-InP + the void_pass1 transformer, quadmask input) is the BASE
fill source when a moving object is removed: it runs ONCE over the whole shot (1344 x 768 like the experiment
removers_test/void_hr3.sh) and the keys take their frames from it. A key built from VOID is ALWAYS the original plate +
the VOID fill only where the removed object and its traces are (`keys.patch_on_plate`, never the whole model frame).
SeedVR2 (7B sharp fp8, a video upscaler) then upscales the key patches: the SEQUENCE of a card's keys as one clip
(temporal consistency – dense keys), on the patch crop only (one crop for all keys), colour correction "lab", pasted
back by the same patch_on_plate rule. Every result is a NEW key version: `source` "void" / "seedvr2:<base version>",
`base` = the version it was built from (the version chain; an optional FLUX 3 Image Klein + LoRA Restore step on top is
the "steps on a key" package).

Both tools run in their OWN Python environments (configured per machine in config.local.json, see `TOOL_KEYS`; empty =
engine unavailable – the gate `stage_models.cleanup3d_tools`), a process per command under the GPU lease
(`VoidRunner.run` / `SeedVR2Runner.run` -> `GPU.use`, engine/gpu_manager.py LEASED_CALLS); „Przerwij” kills the process
(tree). On disk: `<work>/inpaint/void/void.json` (runs, the last one) and `<work>/inpaint/void/<run>/` (input/ = the
VOID input video + quadmask + the PNG frames they were made of, frames/f_######.png = the VOID output per plate frame);
SeedVR2 clips in `<work>/inpaint/void/seedvr2/<card>_<stamp>/`.

Quadmask (the experiment's void/prep_void.py): 0 = the removed object (its alpha > 0.25, grown 2 px at VOID size),
127 = what it affected (traces mask, default the object grown by `affect_px` plate px, minus the object and minus the
objects kept), 63 = object ∩ affected in its lowest 20 % (feet in the sand), 255 = keep. Lossless H.264 (qp 0,
yuv444p) so the four values survive.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402

PathLike = Union[str, Path]
Progress = Optional[Callable[[str, int, int], None]]
Cancel = Optional[Callable[[], None]]

# config.local.json keys (paths; relative = against the MatteDesk folder, engine/config.PATH_KEYS)
TOOL_KEYS = {
    "void": ("void_python", "void_repo", "void_models"),
    "seedvr2": ("seedvr2_python", "seedvr2_repo", "seedvr2_models"),
}
VOID_LABEL = "VOID"
SEEDVR2_LABEL = "SeedVR2"
VOID_SCRIPT = "inference/cogvideox_fun/predict_v2v.py"
VOID_CONFIG = "config/quadmask_cogvideox.py"
VOID_BASE = "CogVideoX-Fun-V1.5-5b-InP"
VOID_TRANSFORMER = "void_pass1.safetensors"
SEEDVR2_SCRIPT = "inference_cli.py"
SEEDVR2_DIT = "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
# the variant by the card (user 2026-11-11, as FLUX 3 Image Klein): 3B fp8 (3.4 GB) under SEEDVR2_7B_MIN_GB – the 7B
# (8.2 GB) without block swap was „unstable” on 12 GB –, 7B sharp fp8 from there; MD_SEEDVR2_VARIANT forces one (tests)
SEEDVR2_DITS = {"7b": (SEEDVR2_DIT, 8.24), "3b": ("seedvr2_ema_3b_fp8_e4m3fn.safetensors", 3.39)}
SEEDVR2_7B_MIN_GB = 16.0
SEEDVR2_VAE = "ema_vae_fp16.safetensors"

VOID_LONG, VOID_SHORT = 1344, 768     # the experiment's size (void_hr3.sh: sample_size 768x1344)
VOID_WINDOW = 57                      # temporal window of the experiment (57 frames = 4 x 14 + 1)
VOID_FPS = 12                         # the input video's frame rate (as prep_void.py; VOID's config.data.fps)
VOID_SEQ = "shot"                     # the sequence folder name inside the data root
VOID_OBJ_THR = 0.25                   # object alpha above this = quadmask 0 (prep_void.py: > 64 / 255)
VOID_OBJ_GROW = 2                     # object grown by this at VOID size (prep_void.py: 5 x 5 dilation)
VOID_FEET_PX = 4                      # affected reach that turns the object's bottom into 63 (prep_void.py: 9 x 9)
VOID_FEET_FRAC = 0.8                  # ... only below this fraction of the object's height
AFFECT_PX = 40.0                      # default traces area: the object grown by this (plate px) = REBASE_GROW_PX
VOID_PROMPT = ("The same scene with the object removed: the background continues naturally with the same light, "
               "texture and camera motion.")
SEEDVR2_MARGIN = 160                  # px of plate around the union of the keys' patch holes (flicker_up_test crop_box)
SEEDVR2_MULT = 16                     # crop sides multiples of this
SEEDVR2_MIN_SIDE = 256                # the clip SeedVR2 gets is never smaller than this (short side)
SEEDVR2_BATCH_MAX = 21                # frames per SeedVR2 batch (4n + 1); more keys = batches with a temporal overlap
SEEDVR2_OVERLAP = 3
SEEDVR2_COLOR = "lab"
VOID_SOURCE = "void"
SEEDVR2_SOURCE = "seedvr2"

# Default places (user decision 2026-11-10: SeedVR2 inside ComfyUI, VOID in its own venv built on ComfyUI's Python;
# both installed by the Doctor – engine/installer.install_seedvr2 / install_void). config.local.json keys override them.
REPO_DIR = Path(__file__).resolve().parents[2]
VOID_HOME = REPO_DIR / "bin" / "void"                 # <home>/void-model (git clone) + <home>/venv
VOID_MODELS_SUB = "void"                              # <models_dir>/void: CogVideoX-Fun-V1.5-5b-InP/ + void_pass1.safetensors
SEEDVR2_NODE = "ComfyUI-SeedVR2_VideoUpscaler"        # <comfy_dir>/custom_nodes/<this>, run with ComfyUI's Python
SEEDVR2_MODELS_SUB = "SEEDVR2"                        # <models_dir>/SEEDVR2 (the node's own folder)
VOID_FULL_GB = 19.0                                   # card memory for the repo's default mode (16.4 GB peak measured)
VOID_LOWMEM_MODE = "sequential_cpu_offload"           # predict_v2v config.system.gpu_memory_mode on smaller cards


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class ToolError(RuntimeError):
    """VOID / SeedVR2 cannot run or finish – Polish message for the user (409 in the API)."""


class ToolCancelled(ToolError):
    def __init__(self, msg: str = "Przerwane"):
        super().__init__(msg)


# ----------------------------------------------------------------------------- paths of the tools (config.local.json)
def tool_paths(cfg: Optional[Dict[str, Any]], tool: str) -> Dict[str, Optional[Path]]:
    """{"python", "repo", "models"} of `tool`: the config keys, else the places the Doctor installs to (None = cannot be
    known, e.g. no ComfyUI configured for SeedVR2).
    VOID: <MatteDesk>/bin/void/venv + bin/void/void-model, models <models_dir>/void (no models_dir: the repo, as its README).
    SeedVR2: ComfyUI's Python + <comfy_dir>/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler, models <models_dir>/SEEDVR2
    (no models_dir: <repo>/models/SEEDVR2, the CLI's default)."""
    cfg = cfg or {}
    kp, kr, km = TOOL_KEYS[tool]
    get = lambda k: Path(str(cfg.get(k))) if cfg.get(k) else None   # noqa: E731
    py, repo, models = get(kp), get(kr), get(km)
    mdir = Path(str(cfg["models_dir"])) if cfg.get("models_dir") else None
    if tool == "void":
        py = py or venv_python(VOID_HOME / "venv")
        repo = repo or VOID_HOME / "void-model"
        models = models or ((mdir / VOID_MODELS_SUB) if mdir else repo)
    else:
        if py is None:
            from engine.installer import comfy_python
            py = comfy_python(cfg)
        if repo is None and cfg.get("comfy_dir"):
            repo = Path(str(cfg["comfy_dir"])) / "custom_nodes" / SEEDVR2_NODE
        if models is None:
            models = (mdir / SEEDVR2_MODELS_SUB) if mdir else (repo / "models" / SEEDVR2_MODELS_SUB if repo else None)
    return {"python": py, "repo": repo, "models": models}


def seedvr2_variant(card_gb: Optional[float] = None) -> str:
    """"3b" on a card under SEEDVR2_7B_MIN_GB, else "7b" (unknown card = 7b)."""
    forced = os.environ.get("MD_SEEDVR2_VARIANT", "")
    if forced in SEEDVR2_DITS:
        return forced
    if card_gb is None:
        try:
            from engine.vram_fit import card_gb as _card
            card_gb = _card()
        except Exception:  # noqa: BLE001
            card_gb = None
    return "3b" if card_gb is not None and card_gb < SEEDVR2_7B_MIN_GB - 0.5 else "7b"


def seedvr2_dit(models: Optional[Path], card_gb: Optional[float] = None) -> str:
    """The DiT file to use: the card's variant when on disk, else the other one when that one is (installed earlier),
    else the card's variant (what the Doctor will download)."""
    want = SEEDVR2_DITS[seedvr2_variant(card_gb)][0]
    if models is not None and not (Path(models) / want).is_file():
        other = next((fn for fn, _gb in SEEDVR2_DITS.values() if fn != want and (Path(models) / fn).is_file()), None)
        if other:
            return other
    return want


def required_files(cfg: Optional[Dict[str, Any]], tool: str) -> List[Tuple[str, Optional[Path]]]:
    """(description, path) of every file `tool` needs; path None = its config key is empty. VOID: the native ComfyUI
    files (Comfy-Org/void-model), the old repo + venv rows only while that setup is complete."""
    if tool == "void":
        if void_legacy_complete(cfg):
            return _legacy_void_rows(cfg)
        md = Path(str((cfg or {}).get("models_dir"))) if (cfg or {}).get("models_dir") else None
        f = void_native_files(cfg)
        clip = f["clip"] or (md / "text_encoders" / VOID_NATIVE_CLIPS[0] if md else None)
        return [(f"diffusion_models/{VOID_NATIVE_UNET}", f["unet"] or (md / "diffusion_models" / VOID_NATIVE_UNET if md else None)),
                (f"vae/{VOID_NATIVE_VAE}", f["vae"] or (md / "vae" / VOID_NATIVE_VAE if md else None)),
                (f"text_encoders/{VOID_NATIVE_CLIPS[0]}", clip)]
    return _seedvr2_rows(cfg)


def _legacy_void_rows(cfg: Optional[Dict[str, Any]]) -> List[Tuple[str, Optional[Path]]]:
    p = tool_paths(cfg, "void")
    kp, kr, km = TOOL_KEYS["void"]
    j = lambda base, rel: (base / rel) if base is not None else None   # noqa: E731
    return [(f"{kp} (Python środowiska VOID)", p["python"]),
            (f"{kr}/{VOID_SCRIPT}", j(p["repo"], VOID_SCRIPT)),
            (f"{kr}/{VOID_CONFIG}", j(p["repo"], VOID_CONFIG)),
            (f"{km}/{VOID_BASE}/model_index.json", j(p["models"], f"{VOID_BASE}/model_index.json")),
            (f"{km}/{VOID_TRANSFORMER}", j(p["models"], VOID_TRANSFORMER))]


def _seedvr2_rows(cfg: Optional[Dict[str, Any]]) -> List[Tuple[str, Optional[Path]]]:
    p = tool_paths(cfg, "seedvr2")
    kp, kr, km = TOOL_KEYS["seedvr2"]
    j = lambda base, rel: (base / rel) if base is not None else None   # noqa: E731
    return [(f"{kp} (Python środowiska SeedVR2)", p["python"]),
            (f"{kr}/{SEEDVR2_SCRIPT}", j(p["repo"], SEEDVR2_SCRIPT)),
            (f"{km}/{seedvr2_dit(p['models'])}", j(p["models"], seedvr2_dit(p["models"]))),
            (f"{km}/{SEEDVR2_VAE}", j(p["models"], SEEDVR2_VAE))]


# ----------------------------------------------------------------------------- a process per command, under the lease
def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the tool and its children (a torch process may start helpers)."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


class _ProcRunner:
    """Runs `<python> <args...>` in `cwd`; output appended to `log`. Subclasses wrap `_run` in the GPU lease."""

    label = "tool"

    def __init__(self, python: PathLike, cwd: PathLike, log: Optional[PathLike] = None,
                 env: Optional[Dict[str, str]] = None, timeout_s: float = 12 * 3600.0):
        self.python = Path(python)
        self.cwd = Path(cwd)
        self.log = Path(log) if log else None
        self.env = dict(env or {})
        self.timeout_s = float(timeout_s)

    def _environ(self) -> Dict[str, str]:
        e = dict(os.environ)
        e.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        for k in ("PYTHONHOME", "PYTHONPATH"):          # never leak MatteDesk's interpreter into the tool's venv
            e.pop(k, None)
        e.update(self.env)
        return e

    def _run(self, args: Sequence[Any], cancel_check: Cancel) -> None:
        cmd = [str(self.python)] + [str(a) for a in args]
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL
        try:
            if self.log:
                out.write(("\n>>> " + " ".join(cmd) + "\n").encode("utf-8"))
                out.flush()
            try:
                proc = subprocess.Popen(cmd, cwd=str(self.cwd), stdout=out, stderr=subprocess.STDOUT,
                                        env=self._environ(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as e:
                raise ToolError(f"{self.label}: nie można uruchomić {self.python} ({e})")
            t0 = time.time()
            try:
                while True:
                    try:
                        proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        if cancel_check:
                            cancel_check()
                        if time.time() - t0 > self.timeout_s:
                            raise ToolError(f"{self.label}: przekroczony czas {self.timeout_s:.0f} s")
            except BaseException:
                _kill_tree(proc)
                raise
        finally:
            if self.log:
                out.close()
        if proc.returncode != 0:
            raise ToolError(f"{self.label} zakończył się błędem {proc.returncode}: {self._tail()}")

    def _tail(self, n: int = 600) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace").strip()[-n:] if self.log else ""
        except OSError:
            return ""


def _env_path_with(exe: Optional[str]) -> Dict[str, str]:
    """PATH with the folder of the configured ffmpeg first (VOID reads its videos through ffmpeg; the experiment put
    void/ffbin on PATH). A bare program name = PATH as it is."""
    if not exe or ("/" not in exe and "\\" not in exe):
        return {}
    d = str(Path(exe).parent)
    return {"PATH": d + os.pathsep + os.environ.get("PATH", "")}


class VoidRunner(_ProcRunner):
    """VOID pass 1 (inference/cogvideox_fun/predict_v2v.py of the void-model repo) – the command of void_hr3.sh."""

    label = VOID_LABEL

    def __init__(self, python: PathLike, repo: PathLike, models: PathLike, log: Optional[PathLike] = None,
                 ffmpeg: Optional[str] = None, timeout_s: float = 12 * 3600.0, gpu_memory_mode: Optional[str] = None):
        super().__init__(python, repo, log, _env_path_with(ffmpeg), timeout_s)
        self.repo = Path(repo)
        self.models = Path(models)
        self.gpu_memory_mode = gpu_memory_mode       # None = the repo's config default

    def command(self, data_root: Path, seq: str, save_path: Path, size: Tuple[int, int], length: int,
                window: int) -> List[str]:
        w, h = int(size[0]), int(size[1])
        mem = [f"--config.system.gpu_memory_mode={self.gpu_memory_mode}"] if self.gpu_memory_mode else []
        return [VOID_SCRIPT, "--config", VOID_CONFIG, *mem,
                f"--config.data.data_rootdir={Path(data_root).as_posix()}",
                f"--config.data.sample_size={h}x{w}",
                f"--config.data.max_video_length={int(length)}",
                f"--config.video_model.temporal_window_size={int(window)}",
                f"--config.experiment.run_seqs={seq}",
                f"--config.experiment.save_path={Path(save_path).as_posix()}",
                f"--config.video_model.model_name={(self.models / VOID_BASE).as_posix()}",
                f"--config.video_model.transformer_path={(self.models / VOID_TRANSFORMER).as_posix()}"]

    def run(self, data_root: Path, seq: str, save_path: Path, size: Tuple[int, int], length: int, window: int,
            cancel_check: Cancel = None) -> Path:
        """One VOID pass under the GPU lease -> the output video (`<seq>-fg=-1-0001.mp4`, not the `_tuple` one)."""
        from engine.gpu_manager import GPU
        save_path = Path(save_path)
        with GPU.use(("void", str(self.python)), VOID_LABEL, None, cancel_check):
            self._run(self.command(data_root, seq, save_path, size, length, window), cancel_check)
        vids = sorted(p for p in save_path.glob(f"{seq}*.mp4") if not p.stem.endswith("_tuple"))
        if not vids:
            raise ToolError(f"VOID nie zapisał wyniku w {save_path}: {self._tail()}")
        return vids[0]


class SeedVR2Runner(_ProcRunner):
    """SeedVR2 CLI (inference_cli.py of numz/ComfyUI-SeedVR2_VideoUpscaler) – the command of sv_keys.sh, but on a VIDEO
    (a folder would be upscaled frame by frame; a clip keeps the texture consistent between keys)."""

    label = SEEDVR2_LABEL

    def __init__(self, python: PathLike, repo: PathLike, models: PathLike, log: Optional[PathLike] = None,
                 timeout_s: float = 12 * 3600.0, dit: str = SEEDVR2_DIT):
        super().__init__(python, repo, log, None, timeout_s)
        self.repo = Path(repo)
        self.models = Path(models)
        self.dit = dit

    def command(self, src: Path, out_dir: Path, resolution: int, batch: int, frames: int) -> List[str]:
        cmd = [SEEDVR2_SCRIPT, str(Path(src)), "--output", str(Path(out_dir)), "--output_format", "png",
               "--model_dir", str(self.models), "--dit_model", self.dit, "--resolution", str(int(resolution)),
               "--batch_size", str(int(batch)), "--seed", "42", "--vae_encode_tiled", "--vae_decode_tiled",
               "--color_correction", SEEDVR2_COLOR]
        if frames > 1:
            cmd += ["--uniform_batch_size"]
        if frames > batch:
            cmd += ["--temporal_overlap", str(SEEDVR2_OVERLAP)]
        return cmd

    def run(self, src: Path, out_dir: Path, resolution: int, batch: int, frames: int,
            cancel_check: Cancel = None) -> Path:
        """Upscale the clip `src` under the GPU lease -> the folder with its PNG frames (in order)."""
        from engine.gpu_manager import GPU
        out_dir = Path(out_dir)
        with GPU.use(("seedvr2", str(self.python)), SEEDVR2_LABEL, None, cancel_check):
            self._run(self.command(src, out_dir, resolution, batch, frames), cancel_check)
        return out_dir


def void_runner(cfg: Dict[str, Any], log: Optional[PathLike] = None) -> Any:
    """The configured VOID (the gate `stage_models.cleanup3d_tools` first: ToolError with its Polish reason): natively
    in ComfyUI (2026-11-11), the old repo + venv only when that setup is complete."""
    _require(cfg, "void")
    if not void_legacy_complete(cfg):
        return ComfyVoidRunner(cfg)
    p = tool_paths(cfg, "void")
    return VoidRunner(p["python"], p["repo"], p["models"], log=log, ffmpeg=str(cfg.get("ffmpeg_path") or ""),
                      gpu_memory_mode=void_memory_mode(cfg))



# ----------------------------------------------------------------------------- VOID natively in ComfyUI (2026-11-11)
# ComfyUI ≥ 0.45 has VOID's nodes (comfy_extras/nodes_void.py) and Comfy-Org repacked the model: void_pass1 is a
# whole transformer (no CogVideoX-Fun base, no repo, no venv), the text encoder is ComfyUI's t5xxl (shared with other
# models), so the download drops from ~30 GB to ~11.6 GB (+9.8 GB T5 when missing). The graph = the user's ComfyUI
# workflow video_void_remove_subject.json, pass 1 (pass 2 is an optional refinement with optical flow – not used).
VOID_NATIVE_REPO = "Comfy-Org/void-model"
VOID_NATIVE_UNET = "void_pass1.safetensors"
VOID_NATIVE_VAE = "cogvideox_vae.safetensors"
VOID_NATIVE_CLIPS = ("t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "t5xxl_fp8_e4m3fn.safetensors")
VOID_NATIVE_FILES = [("diffusion_models/" + VOID_NATIVE_UNET, "diffusion_models", VOID_NATIVE_UNET, 11.14),
                     ("vae/" + VOID_NATIVE_VAE, "vae", VOID_NATIVE_VAE, 0.43),
                     ("text_encoders/" + VOID_NATIVE_CLIPS[0], "text_encoders", VOID_NATIVE_CLIPS[0], 9.79)]
VOID_NATIVE_NODES = ["VOIDQuadmaskPreprocess", "VOIDInpaintConditioning", "VOIDSampler", "CFGGuider",
                     "SamplerCustomAdvanced"]
VOID_NATIVE_STEPS, VOID_NATIVE_CFG = 30, 6.0
VOID_NATIVE_SAVE = "40"


def void_native_files(cfg: Optional[Dict[str, Any]]) -> Dict[str, Optional[Path]]:
    """{"unet", "vae", "clip"}: the native VOID files in models_dir (None = missing; the clip = any t5xxl variant)."""
    md = Path(str((cfg or {}).get("models_dir"))) if (cfg or {}).get("models_dir") else None
    if md is None:
        return {"unet": None, "vae": None, "clip": None}
    find = lambda folder, names: next((md / folder / n for n in names if (md / folder / n).is_file()), None)  # noqa: E731
    return {"unet": find("diffusion_models", (VOID_NATIVE_UNET,)), "vae": find("vae", (VOID_NATIVE_VAE,)),
            "clip": find("text_encoders", VOID_NATIVE_CLIPS)}


def void_legacy_complete(cfg: Optional[Dict[str, Any]]) -> bool:
    """The old setup (repo + own venv + CogVideoX-Fun base, P12) is complete – kept as it is when there."""
    rows = _legacy_void_rows(cfg)
    return all(p is not None and p.exists() for _d, p in rows)


def void_native_length(n: int) -> int:
    """Frames for VOIDInpaintConditioning: >= n, 4k + 1 AND an even latent count (patch_size_t = 2 – the node rounds an
    odd one down, e.g. 49 -> 45, which would drop frames): 8m + 5."""
    n = max(5, int(n))
    while n % 8 != 5:
        n += 1
    return n


def void_native_graph(rgb_dir: Path, qm_dir: Path, prompt: str, w: int, h: int, length: int,
                      files: Dict[str, str], seed: int = 42) -> Dict[str, Any]:
    from engine.comfy_io import image_to_mask, vhs_load
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "cogvideox", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "10": vhs_load(Path(rgb_dir)),
        "11": vhs_load(Path(qm_dir)),
        "12": image_to_mask(["11", 0]),
        "13": {"class_type": "VOIDQuadmaskPreprocess", "inputs": {"mask": ["12", 0], "dilate_width": 0}},
        "20": {"class_type": "VOIDInpaintConditioning", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0], "video": ["10", 0], "quadmask": ["13", 0],
            "width": int(w), "height": int(h), "length": int(length), "batch_size": 1}},
        "21": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["20", 0], "negative": ["20", 1],
                                                     "cfg": VOID_NATIVE_CFG}},
        "22": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple",
                                                          "steps": VOID_NATIVE_STEPS, "denoise": 1.0}},
        "23": {"class_type": "VOIDSampler", "inputs": {}},
        "24": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["24", 0], "guider": ["21", 0],
                                                                 "sampler": ["23", 0], "sigmas": ["22", 0],
                                                                 "latent_image": ["20", 2]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["3", 0]}},
        VOID_NATIVE_SAVE: {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": "matte_desk/void"}},
    }


class ComfyVoidRunner:
    """VOID pass 1 as a ComfyUI graph – the same `run` as VoidRunner (the input folders prepare_void_input wrote, the
    output folder of PNG frames), so void_fill and everything after it stay as they are."""

    label = VOID_LABEL

    def __init__(self, cfg: Dict[str, Any], client: Any = None, seed: int = 42):
        self.cfg, self.client, self.seed = cfg, client, int(seed)

    def run(self, data_root: Path, seq: str, save_path: Path, size: Tuple[int, int], length: int, window: int,
            cancel_check: Cancel = None) -> Path:
        from engine.comfy_io import require_vhs
        from engine.gpu_manager import comfy_lease
        client = self.client
        if client is None:
            from comfy_client import ComfyClient
            client = ComfyClient.from_config()
        from engine.comfy_io import missing_nodes
        miss = missing_nodes(client, VOID_NATIVE_NODES)
        if miss:
            raise ToolError(f"Twój ComfyUI nie ma węzłów VOID ({', '.join(miss)}) – zaktualizuj ComfyUI (≥ 0.45)")
        require_vhs(client)
        f = void_native_files(self.cfg)
        if not all(f.values()):
            raise ToolError("Brak plików VOID (Comfy-Org/void-model) – Doktor → Zainstaluj VOID")
        data = Path(data_root) / seq
        rgb = sorted((data / "_rgb").glob("*.png"))
        qm = sorted((data / "_qm").glob("*.png"))
        n = min(len(rgb), len(qm))
        if not n:
            raise ToolError("VOID: brak klatek wejściowych")
        L = void_native_length(n)
        tmp = Path(save_path) / "_in"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "rgb").mkdir(parents=True)
        (tmp / "qm").mkdir(parents=True)
        for i in range(L):                                # padded with the last frame (dropped on read)
            j = min(i, n - 1)
            shutil.copyfile(rgb[j], tmp / "rgb" / f"{i:06d}.png")
            shutil.copyfile(qm[j], tmp / "qm" / f"{i:06d}.png")
        prompt = ""
        try:
            prompt = json.loads((data / "prompt.json").read_text(encoding="utf-8")).get("bg") or ""
        except (OSError, ValueError):
            pass
        names = {k: p.name for k, p in f.items()}
        g = void_native_graph(tmp / "rgb", tmp / "qm", prompt or VOID_PROMPT, int(size[0]), int(size[1]), L, names,
                              self.seed)
        out = Path(save_path) / "frames_out"
        out.mkdir(parents=True, exist_ok=True)
        with comfy_lease(client, f"{VOID_LABEL} (ComfyUI)"):
            sub = client.queue_prompt(g)
            if not sub.get("success"):
                raise ToolError(f"ComfyUI odrzucił graf VOID: {sub.get('error')}")
            res = client.wait_for_prompt(sub["prompt_id"], timeout_sec=12 * 3600)
        if not res.get("success"):
            raise ToolError(f"VOID nie powiódł się: {res.get('error')}")
        imgs = res.get("outputs", {}).get(VOID_NATIVE_SAVE, {}).get("images", [])
        for i, info in enumerate(imgs):
            (out / f"{i:06d}.png").write_bytes(client.download_output(info))
        shutil.rmtree(tmp, ignore_errors=True)
        return out

def void_memory_mode(cfg: Optional[Dict[str, Any]] = None, card: Optional[float] = None) -> Optional[str]:
    """predict_v2v's config.system.gpu_memory_mode for this card: None = the repo's default (model_cpu_offload_and_qfloat8,
    16.4 GB peak measured at 1344x768) on >= VOID_FULL_GB; smaller cards -> layer-by-layer CPU offload (much slower).
    config.local.json `void_gpu_memory_mode` forces a mode."""
    forced = str((cfg or {}).get("void_gpu_memory_mode") or "").strip()
    if forced:
        return forced
    if card is None:
        from engine import vram_fit
        card = vram_fit.card_gb()
    from engine.vram_fit import TOL_GB
    return VOID_LOWMEM_MODE if card and card + TOL_GB < VOID_FULL_GB else None


def seedvr2_runner(cfg: Dict[str, Any], log: Optional[PathLike] = None) -> SeedVR2Runner:
    _require(cfg, "seedvr2")
    p = tool_paths(cfg, "seedvr2")
    return SeedVR2Runner(p["python"], p["repo"], p["models"], log=log, dit=seedvr2_dit(p["models"]))


def _require(cfg: Dict[str, Any], tool: str) -> None:
    from engine.stage_models import cleanup3d_tools
    g = cleanup3d_tools(cfg).get(tool) or {}
    if not g.get("ok"):
        raise ToolError(g.get("reason") or f"{tool}: niedostępny")


# ----------------------------------------------------------------------------- image helpers
def _f32(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    return a[..., :3]


def _lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _srgb_to_lin(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.04045, y / 12.92, np.power((y + 0.055) / 1.055, 2.4)).astype(np.float32)


def to_display8(img: np.ndarray, linear: bool = False) -> np.ndarray:
    """Plate (float / uint8 / uint16 BGR; `linear` = scene-linear EXR) -> uint8 BGR for the models (display sRGB)."""
    a = _f32(img)
    if linear:
        a = _lin_to_srgb(a)
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def from_display(img8: np.ndarray, linear: bool = False) -> np.ndarray:
    """A model's uint8 BGR output -> float32 BGR in the plate's space."""
    a = _f32(img8)
    return _srgb_to_lin(a) if linear else a


# the VIEWER's picture (2026-11-23, user report SD160_150 v004 F27 „exposure jump at the anchor”): the plate proxies are
# drawn with the PlateToDisplay curve (engine/ingest.tonemap_plate_to_display), the models get plain sRGB (to_display8).
# Every C1 image drawn in the viewer over / next to a plate proxy (the live composite, the model picture, the anchor
# previews) goes through to_view8 / model_to_view8 – the same curve as the proxies; a plain sRGB overlay looked darker
# and harder on the frame it was paused on. The models' own input / output stays to_display8 / from_display.
def to_view8(img: np.ndarray, linear: bool = False) -> np.ndarray:
    """Plate-space image (float / uint8 / uint16 BGR) -> uint8 BGR as the viewer shows the plate (its proxies)."""
    if not linear:
        return to_display8(img, False)
    from engine.ingest import tonemap_plate_to_display
    return tonemap_plate_to_display(np.asarray(_f32(img), np.float32))


def model_to_view8(img8: np.ndarray, linear: bool = False) -> np.ndarray:
    """A model-space uint8 BGR picture (to_display8 encoding: a model's frame, an anchor) -> uint8 BGR as the viewer."""
    return to_view8(from_display(img8, linear), linear)


def _mask01(m: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if m is None:
        return np.zeros(shape, np.float32)
    a = np.asarray(m)
    a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 65535.0 if a.dtype == np.uint16 else 1.0)
    if a.ndim == 3:
        a = a.max(axis=2)
    if a.shape != shape:
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return a


def void_size(plate_w: int, plate_h: int) -> Tuple[int, int]:
    """(w, h) of the VOID input: the experiment's 1344 x 768 in the plate's orientation."""
    return (VOID_LONG, VOID_SHORT) if plate_w >= plate_h else (VOID_SHORT, VOID_LONG)


def video_length(n: int) -> int:
    """Frames VOID gets: n padded up to 4k + 1 (the CogVideoX VAE compresses time by 4; VOID truncates otherwise)."""
    return 1 if n <= 1 else ((n - 1 + 3) // 4) * 4 + 1


def quadmask(obj: np.ndarray, affected: np.ndarray, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """uint8 quadmask at VOID size from 0..1 masks at that size: 0 object, 127 affected, 63 object ∩ affected in its
    lowest 20 % (feet), 255 keep (prep_void.py)."""
    h, w = obj.shape[:2]
    o = obj > VOID_OBJ_THR
    if VOID_OBJ_GROW > 0:
        k = 2 * VOID_OBJ_GROW + 1
        o = cv2.dilate(o.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    aff = (affected > VOID_OBJ_THR) & ~o
    if keep is not None:
        aff &= ~(keep > VOID_OBJ_THR)
    k = 2 * VOID_FEET_PX + 1
    feet = o & (cv2.dilate(aff.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)
    ys = np.nonzero(o)[0]
    if len(ys):
        feet &= np.arange(h)[:, None] > ys.min() + VOID_FEET_FRAC * (ys.max() - ys.min())
    else:
        feet[:] = False
    q = np.full((h, w), 255, np.uint8)
    q[aff] = 127
    q[o] = 0
    q[feet] = 63
    return q


def grown(mask01: np.ndarray, px: float) -> np.ndarray:
    """0..1 mask -> 0 / 1 mask grown by `px` (the default traces area of the object)."""
    m = (mask01 > 0.02).astype(np.uint8)
    if px <= 0 or not m.any():
        return m.astype(np.float32)
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return (d <= float(px)).astype(np.float32)


# ----------------------------------------------------------------------------- video io (ffmpeg)
def _ffmpeg(cfg: Optional[Dict[str, Any]]) -> str:
    return str((cfg or {}).get("ffmpeg_path") or "ffmpeg")


def encode_video(ffmpeg: str, pattern: Path, dst: Path, lossless: bool, fps: int = VOID_FPS,
                 codec: str = "h264") -> None:
    """PNG frames (`pattern` with %06d) -> a video. lossless h264 = qp 0 yuv444p (the quadmask's four values survive);
    otherwise crf 10 yuv420p (the plate, as prep_void.py); codec "ffv1" = lossless BGR in .mkv (the SeedVR2 clip)."""
    ff = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(int(fps)), "-start_number", "0", "-i", str(pattern)]
    if codec == "ffv1":
        args = ["-c:v", "ffv1", "-pix_fmt", "bgr0"]
    elif lossless:
        args = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv444p"]
    else:
        args = ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    try:
        r = subprocess.run(ff + args + [str(dst)], capture_output=True, text=True, timeout=3600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as e:
        raise ToolError(f"ffmpeg nie działa ({ffmpeg}): {e} – Doktor → ffmpeg")
    if r.returncode != 0:
        raise ToolError(f"ffmpeg: nie udało się zapisać {dst.name}: {(r.stderr or '').strip()[-400:]}")


def read_frames(src: PathLike) -> List[np.ndarray]:
    """Frames of a video file or of a folder of PNG / JPG files (sorted, recursively) as uint8 BGR."""
    src = Path(src)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        out = []
        for p in files:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise ToolError(f"Nie można odczytać klatki {p}")
            out.append(im)
        return out
    cap = cv2.VideoCapture(str(src))
    out = []
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(fr)
    finally:
        cap.release()
    if not out:
        raise ToolError(f"Nie można odczytać wideo {src}")
    return out


# ----------------------------------------------------------------------------- runs on disk
def void_root(work: PathLike) -> Path:
    return T.root_dir(Path(work)) / "void"


def _runs_file(work: PathLike) -> Path:
    return void_root(work) / "void.json"


def load_runs(work: PathLike) -> Dict[str, Any]:
    p = _runs_file(work)
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        d = {}
    return {"last": d.get("last"), "runs": list(d.get("runs") or [])}


def _save_runs(work: PathLike, d: Dict[str, Any]) -> None:
    p = _runs_file(work)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def last_run(work: PathLike) -> Optional[Dict[str, Any]]:
    """The last FINISHED VOID run (its frames on disk) or None."""
    d = load_runs(work)
    r = next((x for x in d["runs"] if x.get("id") == d["last"]), None)
    if r and r.get("status") == "done":
        return r
    return None


def frame_path(work: PathLike, run: Dict[str, Any], frame: int) -> Path:
    return void_root(work) / str(run["id"]) / "frames" / f"f_{int(frame):06d}.png"


def void_frame(work: PathLike, run: Dict[str, Any], frame: int, size: Tuple[int, int]) -> np.ndarray:
    """uint8 BGR VOID frame `frame` of `run`, scaled (bicubic, like vid2frames.py) to `size` = (w, h)."""
    p = frame_path(work, run, frame)
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Przebieg VOID {run.get('id')} nie ma klatki F{int(frame)}")
    if (im.shape[1], im.shape[0]) != tuple(size):
        im = cv2.resize(im, (int(size[0]), int(size[1])), interpolation=cv2.INTER_CUBIC)
    return im


# ----------------------------------------------------------------------------- (1) VOID over the whole shot
def prepare_void_input(frames: Sequence[int], plate_fn: Callable[[int], np.ndarray],
                       mask_fn: Callable[[int], Optional[np.ndarray]], data_dir: Path, *,
                       affect_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                       keep_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
                       affect_px: float = AFFECT_PX, size: Optional[Tuple[int, int]] = None, linear: bool = False,
                       prompt: str = VOID_PROMPT, progress: Progress = None, cancel_check: Cancel = None,
                       ) -> Dict[str, Any]:
    """`data_dir/_rgb/%06d.png` (the plate at VOID size, display sRGB), `data_dir/_qm/%06d.png` (the quadmask) and
    `prompt.json` ({"bg": prompt}); the frame list padded to 4k + 1 by repeating the last frame. Masks 0..1 at plate
    size: `mask_fn` = the removed objects, `affect_fn` = their traces (None = the objects grown by `affect_px` plate
    px), `keep_fn` = objects kept (never "affected"). Returns {"size", "length", "frames", "qm_values"}."""
    frames = [int(f) for f in frames]
    if not frames:
        raise ValueError("VOID: pusty zakres klatek")
    rgb, qmd = data_dir / "_rgb", data_dir / "_qm"
    for d in (rgb, qmd):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    length = video_length(len(frames))
    seen: set = set()
    W = H = None
    for i, f in enumerate(frames):
        if cancel_check:
            cancel_check()
        if progress:
            progress("VOID: przygotowanie wejścia", i, len(frames))
        pl = plate_fn(f)
        ph, pw = np.asarray(pl).shape[:2]
        if W is None:
            W, H = size or void_size(pw, ph)
        obj = _mask01(mask_fn(f), (ph, pw))
        aff = _mask01(affect_fn(f), (ph, pw)) if affect_fn is not None else None
        if aff is None or not (aff > 0).any():
            aff = grown(obj, affect_px * 1.0)
        aff = np.maximum(aff, (obj > VOID_OBJ_THR).astype(np.float32))
        keep = _mask01(keep_fn(f), (ph, pw)) if keep_fn is not None else None
        rs = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)   # noqa: E731
        q = quadmask(rs(obj), rs(aff), None if keep is None else rs(keep))
        seen |= set(np.unique(q).tolist())
        cv2.imwrite(str(rgb / f"{i:06d}.png"), rs(to_display8(pl, linear)))
        cv2.imwrite(str(qmd / f"{i:06d}.png"), q)
    for i in range(len(frames), length):                  # pad to 4k + 1 with the last frame (dropped on read)
        shutil.copyfile(rgb / f"{len(frames) - 1:06d}.png", rgb / f"{i:06d}.png")
        shutil.copyfile(qmd / f"{len(frames) - 1:06d}.png", qmd / f"{i:06d}.png")
    (data_dir / "prompt.json").write_text(json.dumps({"bg": prompt or VOID_PROMPT}, ensure_ascii=False),
                                          encoding="utf-8")
    if progress:
        progress("VOID: przygotowanie wejścia", len(frames), len(frames))
    return {"size": [int(W), int(H)], "length": int(length), "frames": frames, "qm_values": sorted(seen)}


def void_fill(work: PathLike, frames: Sequence[int], plate_fn: Callable[[int], np.ndarray],
              mask_fn: Callable[[int], Optional[np.ndarray]], runner: Any, out_dir: Optional[PathLike] = None,
              progress: Progress = None, cancel: Cancel = None, *,
              affect_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
              keep_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None, affect_px: float = AFFECT_PX,
              size: Optional[Tuple[int, int]] = None, window: int = VOID_WINDOW, linear: bool = False,
              prompt: str = VOID_PROMPT, ffmpeg: str = "ffmpeg", encode: Optional[Callable[..., None]] = None,
              client: Any = None) -> Dict[str, Any]:
    """VOID once over `frames` (the whole shot): input video + quadmask from the removal masks, one pass, the output
    frames extracted per plate frame -> `<run>/frames/f_######.png`. Recorded in void.json (status running -> done /
    failed); returns the run {"id", "frames", "size", "dir", "paths": {frame: path}, ...}. `runner` = VoidRunner (or a
    fake with the same `run`); `encode(pattern, dst, lossless)` replaces ffmpeg (tests); `client` = ComfyUI, released
    before the run (VOID needs the whole card)."""
    from engine.gpu_manager import before_stage
    work = Path(work)
    frames = sorted({int(f) for f in frames})
    if not frames:
        raise ValueError("VOID: pusty zakres klatek")
    rid = time.strftime("run_%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir else void_root(work) / rid
    if root.exists():
        shutil.rmtree(root)
    data_root = root / "input"
    data_dir = data_root / VOID_SEQ
    save = root / "output"
    data_dir.mkdir(parents=True)
    runs = load_runs(work)
    rec = {"id": root.name if out_dir else rid, "status": "running", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "frames": frames, "prompt": prompt or VOID_PROMPT, "affect_px": float(affect_px),
           "dir": str(root).replace("\\", "/")}
    runs["runs"] = [r for r in runs["runs"] if r.get("id") != rec["id"]] + [rec]
    _save_runs(work, runs)
    try:
        info = prepare_void_input(frames, plate_fn, mask_fn, data_dir, affect_fn=affect_fn, keep_fn=keep_fn,
                                  affect_px=affect_px, size=size, linear=linear, prompt=prompt, progress=progress,
                                  cancel_check=cancel)
        enc = encode or (lambda pat, dst, lossless: encode_video(ffmpeg, pat, dst, lossless))
        enc(data_dir / "_rgb" / "%06d.png", data_dir / "input_video.mp4", False)
        enc(data_dir / "_qm" / "%06d.png", data_dir / "quadmask_0.mp4", True)
        before_stage("comfy", client, "VOID")
        if progress:
            progress("VOID: usuwanie obiektu z ujęcia", 0, 1)
        win = min(int(window), info["length"])
        win = max(1, ((win - 1) // 4) * 4 + 1)
        out = runner.run(data_root, VOID_SEQ, save, tuple(info["size"]), info["length"], win, cancel)
        got = read_frames(out)
        if len(got) < len(frames):
            raise ToolError(f"VOID zwrócił {len(got)} klatek, potrzeba {len(frames)}")
        fdir = root / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        paths = {}
        W, H = info["size"]
        for f, im in zip(frames, got):
            if (im.shape[1], im.shape[0]) != (W, H):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            p = fdir / f"f_{f:06d}.png"
            cv2.imwrite(str(p), im)
            paths[f] = str(p).replace("\\", "/")
        if progress:
            progress("VOID: usuwanie obiektu z ujęcia", 1, 1)
        rec.update(status="done", size=info["size"], length=info["length"], window=win, qm_values=info["qm_values"],
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        runs = load_runs(work)
        runs["runs"] = [r for r in runs["runs"] if r.get("id") != rec["id"]] + [rec]
        runs["last"] = rec["id"]
        _save_runs(work, runs)
        return {**rec, "paths": paths}
    except BaseException as e:
        rec.update(status="cancelled" if type(e).__name__ in ("JobCancelled", "ToolCancelled") else "failed",
                   error=str(e)[:400])
        runs = load_runs(work)
        runs["runs"] = [r for r in runs["runs"] if r.get("id") != rec["id"]] + [rec]
        _save_runs(work, runs)
        raise


# ----------------------------------------------------------------------------- key versions with a chain
def add_chained(work: PathLike, card: str, frame: int, image: np.ndarray, source: str, ext: str,
                base: Optional[str], **extra: Any) -> T.Key:
    """A new active key version (keys.add_version) with `base` = the version it was built from (+ extra fields)."""
    key = KS.add_version(work, card, frame, image, source, ext)
    e = key.versions[-1]
    e["base"] = base
    e.update({k: v for k, v in extra.items() if v is not None})
    store.save_key(Path(work), key)
    return key


def _ext_of(key: Optional[T.Key], default: str) -> str:
    """Key format of a new version: the active version's precision (8-bit png, 16-bit png, exr), else `default`."""
    if key is None or not key.versions:
        return default
    e = next((x for x in key.versions if x.get("v") == key.active), key.versions[-1])
    if e.get("ext") == "exr":
        return "exr"
    return "png16" if int(e.get("bits") or 8) > 8 else "png"


# ----------------------------------------------------------------------------- (2) keys from the VOID run
def keys_from_void(work: PathLike, cards: Sequence[str], key_frames: Optional[Dict[str, Sequence[int]]] = None, *,
                   plate_fn: Callable[[int], np.ndarray], mask_fn: Callable[[int], Optional[np.ndarray]],
                   run: Optional[Dict[str, Any]] = None, blur_fn: Optional[Callable[[int], Any]] = None,
                   ext: str = "png16", linear: bool = False, grow_px: float = KS.REBASE_GROW_PX,
                   soft_px: float = KS.REBASE_SOFT_PX, effects: bool = True, progress: Progress = None,
                   cancel: Cancel = None) -> Dict[str, Any]:
    """New key versions from the VOID run (default: the last finished one) = keys.patch_on_plate(plate, the VOID frame
    upscaled to the plate, removal mask, effects, blur = `blur_fn(frame)` -> patch_on_plate's blur hook or None).
    `key_frames`: {card: frames}; None / a card missing = the card's existing keys. A key that does not exist yet gets
    the plate frame as its first version (undo goes back to it). The patch is the same for every card of a frame
    (computed once). Source "void", base = the active version before, `void_run` = the run id."""
    work = Path(work)
    run = run or last_run(work)
    if not run:
        raise ValueError("Brak policzonego wypełnienia VOID – najpierw „VOID – wypełnienie ujęcia”")
    have = set(int(f) for f in run.get("frames") or [])
    plan: List[Tuple[int, str]] = []
    for c in cards:
        fr = (key_frames or {}).get(c)
        if fr is None:
            fr = [k.frame for k in store.list_keys(work, c) if k.versions and k.canvas is None]
        for f in sorted({int(x) for x in fr}):
            plan.append((f, c))
    if not plan:
        raise ValueError("Brak kluczy do zbudowania z VOID – dodaj klucze kart (C4) albo podaj klatki")
    miss = sorted({f for f, _ in plan if f not in have})
    if miss:
        raise ValueError(f"Przebieg VOID nie obejmuje klatek F{', F'.join(str(f) for f in miss[:8])} – policz VOID "
                         "na zakresie z tymi klatkami")
    plan.sort()
    done: List[Dict[str, Any]] = []
    cache: Dict[int, np.ndarray] = {}
    for i, (f, c) in enumerate(plan):
        if cancel:
            cancel()
        if progress:
            progress("Klucze z VOID", i, len(plan))
        key = store.load_key(work, c, f)
        if key is not None and key.canvas is not None:
            continue                                     # a master key has its own canvas – not a plate frame
        if f not in cache:
            pl = _f32(plate_fn(f))
            h, w = pl.shape[:2]
            vf = from_display(void_frame(work, run, f, (w, h)), linear)
            m = _mask01(mask_fn(f), (h, w))
            cache = {f: KS.patch_on_plate(pl, vf, m, grow_px, soft_px, effects=effects,
                                          blur=blur_fn(f) if blur_fn else None)}
        if key is None or not key.versions:
            key = KS.create_keys(work, c, [f], plate_fn, ext)[0]
        base = key.active
        nk = add_chained(work, c, f, cache[f], VOID_SOURCE, _ext_of(key, ext), base, void_run=run["id"])
        done.append({"card": c, "frame": f, "version": nk.active, "base": base})
    if progress:
        progress("Klucze z VOID", len(plan), len(plan))
    return {"run": run["id"], "keys": done}


# ----------------------------------------------------------------------------- (3) SeedVR2 on the keys of a card
def patch_crop(masks: Sequence[np.ndarray], shape: Tuple[int, int], grow_px: float = KS.REBASE_GROW_PX,
               margin: int = SEEDVR2_MARGIN, mult: int = SEEDVR2_MULT) -> Tuple[int, int, int, int]:
    """ONE crop (x0, y0, w, h) for all keys: the union of their patch holes (masks grown by `grow_px`) + `margin`,
    clipped to the frame, sides multiples of `mult` (grown inside the frame when possible)."""
    h, w = shape
    union = np.zeros((h, w), bool)
    for m in masks:
        union |= KS.fill_region(m, grow_px, 0.0) > 0
    if not union.any():
        raise ValueError("SeedVR2: klucze karty nie mają obszaru usuwanego obiektu (pusta maska)")
    ys, xs = np.nonzero(union)
    x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
    x1, y1 = min(w, int(xs.max()) + margin + 1), min(h, int(ys.max()) + margin + 1)

    def fit(a0: int, a1: int, n: int) -> Tuple[int, int]:
        size = a1 - a0
        want = min(n - n % mult if n >= mult else n, ((size + mult - 1) // mult) * mult)
        a0 = max(0, min(a0 - (want - size) // 2, n - want))
        return a0, want
    x0, cw = fit(x0, x1, w)
    y0, ch = fit(y0, y1, h)
    return int(x0), int(y0), int(cw), int(ch)


def _batch(n: int, cap: int = SEEDVR2_BATCH_MAX) -> int:
    """SeedVR2 batch (4k + 1) covering `n` frames, at most `cap`."""
    b = 1 if n <= 1 else ((n - 1 + 3) // 4) * 4 + 1
    cap = max(1, ((int(cap) - 1) // 4) * 4 + 1)
    return min(b, cap)


def seedvr2_keys(work: PathLike, card: str, *, plate_fn: Callable[[int], np.ndarray],
                 mask_fn: Callable[[int], Optional[np.ndarray]], runner: Any, frames: Optional[Sequence[int]] = None,
                 in_scale: Optional[float] = None, blur_fn: Optional[Callable[[int], Any]] = None,
                 linear: bool = False, grow_px: float = KS.REBASE_GROW_PX, soft_px: float = KS.REBASE_SOFT_PX,
                 effects: bool = True, batch_max: int = SEEDVR2_BATCH_MAX, ffmpeg: str = "ffmpeg",
                 encode: Optional[Callable[..., None]] = None, progress: Progress = None, cancel: Cancel = None,
                 client: Any = None, out_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """SeedVR2 on the ACTIVE versions of the card's keys (`frames` = a subset; default all keys whose active version is
    not the plate itself and that have no own canvas): one crop for all (patch_crop), the crops as ONE clip in frame
    order (scaled by `in_scale` first – default: the plate / VOID size ratio of the last VOID run, so SeedVR2 does the
    upscale VOID left; 1 = restoration at full size), upscaled back to the crop size, pasted into the active version and
    merged with the plate by patch_on_plate over the SAME patch region as the base version (its object + traces – not
    detected again on the upscaled output). New version per key: source "seedvr2:<base>", base = that version."""
    from engine.gpu_manager import before_stage
    work = Path(work)
    keys = [k for k in store.list_keys(work, card) if k.versions and k.canvas is None]
    if frames is not None:
        want = {int(f) for f in frames}
        keys = [k for k in keys if k.frame in want]
    keys = [k for k in keys if str(KS._version(k, None).get("source") or "") != "plate"]
    if not keys:
        raise ValueError(f"Karta {card} nie ma kluczy z wypełnieniem do podbicia (aktywne wersje to plate)")
    keys.sort(key=lambda k: k.frame)
    plates, masks, bases, holes = {}, {}, {}, {}
    for i, k in enumerate(keys):
        if cancel:
            cancel()
        if progress:
            progress("SeedVR2: wycinek kluczy", i, len(keys))
        plates[k.frame] = _f32(plate_fn(k.frame))
        h, w = plates[k.frame].shape[:2]
        masks[k.frame] = _mask01(mask_fn(k.frame), (h, w))
        bases[k.frame] = KS.read_key_image(work, k)
        if bases[k.frame].shape[:2] != (h, w):
            raise ValueError(f"Klucz {card} F{k.frame} ma inny rozmiar niż plate")
        # the patch region of the BASE version (where it is not the plate: object + its traces), kept as it is – the
        # upscaled crop differs from the plate everywhere a little (detail, colour), so re-detecting the traces on the
        # SeedVR2 output would grow the patch over plate pixels
        if effects:
            holes[k.frame] = KS.effects_mask(plates[k.frame], bases[k.frame], masks[k.frame], grow_px)
        else:
            holes[k.frame] = (KS.fill_region(masks[k.frame], grow_px, 0.0) > 0).astype(np.uint8) * 255
    h, w = next(iter(plates.values())).shape[:2]
    hole_grow = KS.EFFECT_EDGE_PX if effects else 0.0
    x0, y0, cw, ch = patch_crop(list(holes.values()), (h, w), hole_grow + soft_px)
    if in_scale is None:
        run = last_run(work)
        in_scale = 1.0
        if run and run.get("size"):
            in_scale = min(1.0, float(max(run["size"])) / float(max(w, h)))
    s = float(in_scale)
    iw, ih = max(2, int(round(cw * s / 2)) * 2), max(2, int(round(ch * s / 2)) * 2)
    if min(iw, ih) < SEEDVR2_MIN_SIDE:                    # never a tiny clip (SeedVR2 needs some context)
        k = min(1.0, SEEDVR2_MIN_SIDE / float(min(cw, ch)))
        iw, ih = max(2, int(round(cw * k / 2)) * 2), max(2, int(round(ch * k / 2)) * 2)
    root = Path(out_dir) if out_dir else void_root(work) / "seedvr2" / f"{card}_{time.strftime('%Y%m%d_%H%M%S')}"
    if root.exists():
        shutil.rmtree(root)
    src_dir, out = root / "clip", root / "out"
    src_dir.mkdir(parents=True)
    for i, k in enumerate(keys):
        crop = to_display8(bases[k.frame][y0:y0 + ch, x0:x0 + cw], linear)
        if (iw, ih) != (cw, ch):
            crop = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(src_dir / f"{i:06d}.png"), crop)
    clip = root / "clip.mkv"
    enc = encode or (lambda pat, dst, lossless: encode_video(ffmpeg, pat, dst, lossless, codec="ffv1"))
    enc(src_dir / "%06d.png", clip, True)
    before_stage("comfy", client, "SeedVR2")
    if progress:
        progress("SeedVR2: podbicie kluczy", 0, 1)
    res = min(cw, ch)
    runner.run(clip, out, res, _batch(len(keys), batch_max), len(keys), cancel)
    got = read_frames(out)
    if len(got) < len(keys):
        raise ToolError(f"SeedVR2 zwrócił {len(got)} klatek, potrzeba {len(keys)}")
    done = []
    for i, k in enumerate(keys):
        if cancel:
            cancel()
        if progress:
            progress("SeedVR2: łączenie z plate'em", i, len(keys))
        up = got[i]
        if (up.shape[1], up.shape[0]) != (cw, ch):
            up = cv2.resize(up, (cw, ch), interpolation=cv2.INTER_CUBIC)
        fill = bases[k.frame].copy()
        fill[y0:y0 + ch, x0:x0 + cw] = from_display(up, linear)
        img = KS.patch_on_plate(plates[k.frame], fill, holes[k.frame], hole_grow, soft_px, effects=False,
                                blur=blur_fn(k.frame) if blur_fn else None)
        base = k.active
        nk = add_chained(work, card, k.frame, img, f"{SEEDVR2_SOURCE}:{base}", _ext_of(k, "png16"), base,
                         crop=[x0, y0, cw, ch])
        done.append({"card": card, "frame": k.frame, "version": nk.active, "base": base})
    if progress:
        progress("SeedVR2: łączenie z plate'em", len(keys), len(keys))
    return {"card": card, "crop": [x0, y0, cw, ch], "clip_size": [iw, ih], "resolution": int(res),
            "keys": done, "dir": str(root).replace("\\", "/")}


def state(work: PathLike, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """For the UI: availability of both tools (the gate) and the last VOID run."""
    from engine.stage_models import cleanup3d_tools
    try:
        g = cleanup3d_tools(cfg or {})
    except Exception as e:  # noqa: BLE001 – a broken check never blocks silently
        g = {t: {"ok": False, "reason": f"sprawdzenie nie powiodło się: {e}"} for t in TOOL_KEYS}
    d = load_runs(work)
    cur = next((x for x in d["runs"] if x.get("id") == d["last"]), None)
    newest = d["runs"][-1] if d["runs"] else None
    pick = lambda r: None if r is None else {k: r.get(k) for k in   # noqa: E731
                                             ("id", "status", "date", "finished", "size", "error", "prompt",
                                              "affect_px")} | {"frames": len(r.get("frames") or [])}
    return {"void": {"available": bool(g.get("void", {}).get("ok")), "reason": str(g.get("void", {}).get("reason") or ""),
                     "last": pick(cur), "newest": pick(newest), **tool_card(cfg, "void")},
            "seedvr2": {"available": bool(g.get("seedvr2", {}).get("ok")),
                        "reason": str(g.get("seedvr2", {}).get("reason") or ""), **tool_card(cfg, "seedvr2")}}


# Short install card of a tool for the Cleanup 3D panel (user 2026-11-10: "short info about the models and their needs
# for this card + an Install button" instead of the gate's list of paths). The install itself = the Doctor's action.
TOOL_INFO = {
    "void": {"label": "VOID", "download": "11,6 GB (+9,8 GB T5, gdy go brak)",
             "what": "CogVideoX-Fun-V1.5-5b-InP 19,7 GB + void_pass1 10,4 GB; własny venv na Pythonie ComfyUI "
                     "(PyTorch z ComfyUI, bez ponownego pobierania)",
             "license": "kod Apache-2.0; CogVideoX License – komercyjnie po bezpłatnej rejestracji"},
    "seedvr2": {"label": "SeedVR2", "download": "3,9 GB (3B, karta < 16 GB) albo 8,7 GB (7B)",
                "what": "węzeł ComfyUI-SeedVR2_VideoUpscaler w ComfyUI + wagi wg karty: 3B fp8 (3,4 GB) poniżej 16 GB VRAM, "
                        "7B sharp fp8 (8,2 GB) od 16 GB; VAE (0,5 GB)",
                "license": "Apache-2.0"},
}


def tool_card(cfg: Optional[Dict[str, Any]], tool: str) -> Dict[str, Any]:
    """{info, install, partly, vram}: what the install brings, the Doctor action, whether some files are already there
    (= „Dokończ instalację”) and the GPU memory verdict on this card (engine/vram_fit)."""
    try:
        partly = any(p is not None and p.exists() for _d, p in required_files(cfg or {}, tool)[1:])
    except Exception:  # noqa: BLE001 – the card is informative only
        partly = False
    try:
        from engine import vram_fit
        vram = vram_fit.assess(tool, vram_fit.card_gb())
    except Exception:  # noqa: BLE001
        vram = None
    return {"info": TOOL_INFO[tool], "install": tool, "partly": partly, "vram": vram}
