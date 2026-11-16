# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""doctor.py - Environment diagnosis and configuration validator for MatteDesk.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 3a, 12: Stage E0).
Strict isolation:
- Never modifies F:\\Comfy or system packages without explicit user consent.
- Can create/verify venv in matte_desk\\.venv.
- Generates INSTALL_<date>.md with exact PowerShell commands when external components are missing.
- Saves report to doctor_report.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckItem:
    name: str
    category: str
    status: str  # "ok", "warning", "error", "info"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_suggestion: Optional[str] = None
    fix_command: Optional[str] = None
    links: List[str] = field(default_factory=list)
    action: Optional[str] = None   # "install_nodes" | "install_cotracker" | "install_vitmatte" ... -> button "Zainstaluj" in the UI
    license: Optional[str] = None  # licence of the third-party component (LICENSES), shown in the UI
    license_nc: bool = False       # non-commercial / research only -> highlighted in the UI
    license_limited: bool = False  # commercial use only under a condition (LICENSE_LIMITED) -> highlighted, own wording
    vram: Optional[Dict[str, Any]] = None   # engine/vram_fit.assess on THIS card: {level ok|spill|unstable|no, text, ...}
    module: str = "system"         # MODULE_OF: system | alpha | cleanup | both -> the Doctor's module filter


@dataclass
class DoctorReport:
    timestamp: str
    all_ok: bool
    summary: Dict[str, int]
    comfy_flavor: str
    comfy_url: Optional[str]
    items: List[CheckItem] = field(default_factory=list)
    licences: List[Dict[str, Any]] = field(default_factory=list)   # licence_summary(): commercial vs research use

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EXPECTED_SAM3_INPUTS = {
    "SAM3_Detect": [
        "model", "image", "conditioning", "bboxes",
        "positive_coords", "negative_coords", "threshold",
        "refine_iterations", "individual_masks"
    ],
    "SAM3_VideoTrack": [
        "images", "model", "initial_mask", "conditioning",
        "detection_threshold", "max_objects", "detect_interval"
    ],
    "SAM3_TrackToMask": [
        "track_data", "object_indices"
    ]
}

REQUIRED_PLATEMATTE_NODES = [
    "PlateLoadEXR", "PlateToDisplay", "PlateSaveEXR",
    "BacklitCoverage", "MaskRegionLimit", "MatteSplitByRegions",
    "MaskTemporalMedian", "MaskRecoverFromGuide"
]

REQUIRED_MATANYONE_KEYFRAMES_NODES = [
    "MatAnyone2RunKeyframes", "MatAnyone2CoreCombine"
]

BIG_GPU_GB = 20   # >= this much VRAM in Comfy: the MatAnyone2 VRAM patch is only a warning
REQUIRED_MATANYONE_BASE_NODES = ["LoadMatAnyone2Model"]
# The node imports these lazily (only when it runs) -> /object_info lists it even without them. Installed by the pack's
# install.py (ComfyUI Manager convention), not by its requirements.txt. Import names: matanyone2, hydra-core, thinplate.
MATANYONE_PY_MODULES = ["matanyone2", "hydra", "thinplate"]
REQUIRED_DEPTHCRAFTER_NODES = ["DownloadAndLoadDepthCrafterModel", "DepthCrafter"]
# Optional inpaint nodes (docs/PLAN_CLEANPLATE.md): LaMa for the holes of the auto clean plate
REQUIRED_INPAINT_LAMA_NODES = ["INPAINT_LoadInpaintModel", "INPAINT_InpaintWithModel"]
# VideoHelperSuite: Comfy I/O "png8" (8-bit sources: PNG folders -> VHS_LoadImagesPath; engine/comfy_io.py vhs_load)
REQUIRED_VHS_NODES = ["VHS_LoadImagesPath"]
VHS_LABEL = "VideoHelperSuite (ujęcia 8-bit)"
VHS_DIRS = ("ComfyUI-VideoHelperSuite", "comfyui-videohelpersuite")   # git clone | ComfyUI Manager (registry id)
VHS_FIX = ("Wymagane teraz: TAK – ujęcia 8-bitowe (PNG/JPG, np. CLIP2_A, SH020): SAM 3.1, MatAnyone2 i strefy wysyłają do ComfyUI "
           "foldery PNG czytane węzłem VHS_LoadImagesPath; bez niego liczenie takich ujęć pada („Node 'VHS_LoadImagesPath' not found”). "
           "Ujęcia EXR go nie potrzebują. Pobranie: węzeł ok. 1 MB + zależności pip (imageio-ffmpeg ok. 30 MB; opencv zwykle już jest). "
           "Kliknij „Zainstaluj” albo ComfyUI Manager: „ComfyUI-VideoHelperSuite”.")

BIREFNET_LABEL = "Model: BiRefNet_HR-matting / Lucida"
SAPIENS2_LABEL = "Model: Sapiens2-seg-0.4B"

# Licences of third-party components (not part of MatteDesk; the user downloads them and accepts them).
# key = LINKS key -> (text, non-commercial). Keep in sync with THIRD_PARTY_LICENSES.md.
# Short on purpose (user 2026-10-21: no clutter) – the UI adds "commercial: yes" / "non-commercial only"; details in
# THIRD_PARTY_LICENSES.md.
# Checked at the source 2026-10-21 (LICENSE files on GitHub, Hugging Face model cards).
LICENSES = {
    "comfyui": ("GPL-3.0", False),
    "sam3": ("SAM License (bez zastosowań wojskowych / ITAR)", False),
    "matanyone2_node": ("S-Lab 1.0 (MatAnyone2)", True),
    "matanyone2_weights": ("S-Lab 1.0", True),
    "cotracker3": ("CC BY-NC 4.0", True),
    "vitmatte": ("wagi trenowane na zbiorze Adobe", True),
    "birefnet": ("MIT", False),
    "sapiens2": ("Sapiens2 License (bez nadzoru / biometrii / deepfake)", False),
    "depthcrafter": ("Tencent, tylko badawczo", True),
    "local_nodes": ("GPL-3.0 (MatteDesk)", False),
    "inpaint_lama": ("Apache-2.0 / GPL-3.0", False),
    "vhs": ("GPL-3.0", False),
    "ltx": ("LTX-2.x Community, darmowo do 10 mln USD przychodu/rok", False),
    "ltx_fill": ("LTX-2.x Community, darmowo do 10 mln USD przychodu/rok (IC-LoRA in-/outpainting, gated)", False),
    "videomama": ("kod CC BY-NC 4.0", True),
    # FLUX 3 Image Klein + LoRA Restore (Cleanup 3D, 2026-11-09): read at the source (LICENSE.md of
    # ausboss/FLUX-3-Image-Klein-Restore-LoRA = FLUX Kommunity License v1.0, as the gated base repo): outputs may be used
    # commercially only by a "Qualifying User" (gross annualized revenue incl. affiliates < US$5M); otherwise non-commercial.
    "flux3_klein": ("FLUX Kommunity License: komercyjnie tylko przy rocznym przychodzie firmy (z powiązanymi) "
                    "< 5 mln USD, powyżej – licencja od Black Forest Labs", False),
    "colmap": ("BSD", False),
    # Qwen-Image-2.1 (Cleanup 3D, 2026-11-10): Comfy-Org/Qwen-Image-2.1 points to Qwen/Qwen-Image-2.1 LICENSE = Qwen
    # Research License (read at the source 2026-11-10): the Materials for non-commercial purposes only, commercial use
    # needs a separate licence; the status of generated images is not spelled out (to verify with Qwen before paid work).
    "qwen21": ("Qwen Research License – tylko niekomercyjnie (status obrazów wynikowych do sprawdzenia)", True),
    # Qwen3-VL-4B-Instruct (C1 prompt assistant, 2026-11-12): the HF model card says license: apache-2.0 (HF API, read)
    "prompt_assist": ("Apache-2.0", False),
    # VOID (Cleanup 3D P12, 2026-11-10; read in the local copies): void-model LICENSE = Apache-2.0 (code); its base
    # CogVideoX-Fun-V1.5-5b-InP: README "CogVideoX-5B (Transformers module) under the CogVideoX LICENSE" = academic use
    # free, commercial use after registering for the basic commercial licence (free, up to 1M visits / month); the
    # void_pass1 weights (HF netflix/void-model) – licence of the model card not read (no copy on disk): to check.
    "void": ("kod Apache-2.0; baza CogVideoX-5B: CogVideoX License – komercyjnie po bezpłatnej rejestracji (do 1 mln "
             "wizyt/mies.); licencja wag void_pass1 do sprawdzenia", False),
    # SeedVR2 (Cleanup 3D P12): numz/ComfyUI-SeedVR2_VideoUpscaler LICENSE = Apache-2.0 (README: "Apache 2.0 to match
    # ByteDance Seed project"); the weights (numz/SeedVR2_comfyUI, from ByteDance-Seed/SeedVR2) – not read locally.
    "seedvr2": ("Apache-2.0 (kod; wagi ByteDance-Seed SeedVR2 – sprawdź kartę modelu)", False),
    # MiniMax-H3 (Cleanup C1, 2026-11-10): the HF model cards (also alibaba-pai Fun ControlNet Union, 2026-11-13) name
    # "minimax-h3-community-license-agreement" – the text not read yet: to check
    "h3": ("MiniMax-H3 Community License Agreement (także Fun ControlNet) – treść do sprawdzenia przed pracą komercyjną", False),
    "vace": ("Apache-2.0 (Wan 2.2 Fun VACE, kwantyzacja GGUF QuantStack; węzeł ComfyUI-GGUF Apache-2.0)", False),
    "vace_lightning": ("Apache-2.0 (lightx2v Wan 2.2 Lightning)", False),
}
# Licences that allow commercial use only under a condition (not "non-commercial"): highlighted in the UI with their own
# wording; the licence summary lists them as use "limited".
LICENSE_LIMITED = {"flux3_klein", "void"}

# Doctor's licence summary (user decision 2026-10-21: "what is commercial and what is research use") – one row per
# third-party component, installed or not: (shown name, LICENSES key, use). use: "commercial" (check the terms) |
# "research" (non-commercial / research only = LICENSES nc) | "check" (licence not established).
LICENCE_SUMMARY = [
    ("ComfyUI", "comfyui", "commercial"), ("SAM 3.1", "sam3", "commercial"),
    ("BiRefNet_HR-matting / Lucida", "birefnet", "commercial"), ("Sapiens2-seg-0.4B", "sapiens2", "commercial"),
    ("LaMa", "inpaint_lama", "commercial"),
    ("LTX-2.5 Alpha-Gen", "ltx", "commercial"), ("VideoHelperSuite", "vhs", "commercial"),
    ("MatAnyone2", "matanyone2_weights", "research"), ("CoTracker3", "cotracker3", "research"),
    ("ViTMatte", "vitmatte", "research"), ("VideoMaMa", "videomama", "research"),
    ("DepthCrafter", "depthcrafter", "research"),
    ("FLUX 3 Image Klein + LoRA Restore", "flux3_klein", "limited"), ("COLMAP", "colmap", "commercial"),
    ("Qwen-Image-2.1", "qwen21", "research"), ("Qwen3-VL-4B-Instruct (asystent opisu)", "prompt_assist", "commercial"),
    ("VOID (CogVideoX-Fun-V1.5-5b-InP + void_pass1)", "void", "limited"), ("SeedVR2", "seedvr2", "commercial"),
    ("MiniMax-H3", "h3", "check"),
]


def licence_summary() -> List[Dict[str, Any]]:
    return [{"name": n, "use": use, "license": LICENSES[k][0]} for n, k, use in LICENCE_SUMMARY]

# Where to get missing components (official sources). Shown in the UI and in INSTALL_<date>.md.
LINKS = {
    "python": ["https://www.python.org/downloads/release/python-3119/"],
    "ffmpeg": ["https://www.gyan.dev/ffmpeg/builds/"],
    "git": ["https://git-scm.com/download/win (Git for Windows)"],
    "comfyui_install": ["https://www.comfy.org/download (ComfyUI Desktop – zalecane: pobierz, zainstaluj, uruchom raz)",
                        "https://docs.comfy.org/installation/desktop/overview (instrukcja instalacji)",
                        "https://github.com/comfyanonymous/ComfyUI (ComfyUI z gita – opcja)"],
    "comfyui": ["https://docs.comfy.org/installation/desktop/overview (ComfyUI Desktop)",
                "https://github.com/comfyanonymous/ComfyUI"],
    "sam3": ["https://huggingface.co/Comfy-Org/sam3.1 (plik checkpoints/sam3.1_multiplex_fp16.safetensors)",
             "https://docs.comfy.org/tutorials/utility/video-segment-sam3"],
    "matanyone2_node": ["https://github.com/spiritform/comfy-matanyone2",
                        "https://github.com/pq-yang/MatAnyone2"],
    "matanyone2_weights": ["https://github.com/pq-yang/MatAnyone2/releases (matanyone2.pth; węzeł pobiera go też sam przy 1. użyciu)"],
    "cotracker3": ["https://github.com/facebookresearch/co-tracker",
                   "https://huggingface.co/facebook/cotracker3 (plik scaled_offline.pth)"],
    "vitmatte": ["https://huggingface.co/hustvl/vitmatte-base-composition-1k (pobiera Doktor / 1. użycie strefy)",
                 "https://github.com/hustvl/ViTMatte"],
    "birefnet": ["https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting (444 MB)",
                 "https://huggingface.co/egeorcun/lucida (885 MB)"],
    "sapiens2": ["https://huggingface.co/facebook/sapiens2-seg-0.4b (1,63 GB)",
                 "https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md (Sapiens2 License)"],
    "depthcrafter": ["https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes (model pobiera się sam przy 1. użyciu)",
                     "https://github.com/Tencent/DepthCrafter"],
    "local_nodes": ["handoff/comfy/custom_nodes/ w repo MatteDesk (nie ma ich w internecie)"],
    "inpaint_lama": ["https://github.com/Acly/comfyui-inpaint-nodes (ComfyUI Manager: „ComfyUI Inpaint Nodes”)",
                     "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt (→ models/inpaint/)"],
    "vhs": ["https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite (ComfyUI Manager: „ComfyUI-VideoHelperSuite”)"],
    "ltx_fill": ["https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting (Agree and Access – osobno od LTX-2.5; ltx-2.3-22b-ic-lora-in-outpainting-0.9, 1,31 GB, do models/loras)"],
    "ltx": ["https://huggingface.co/Lightricks/LTX-2.5 (Agree and Access; transformer int8, Gemma int8, VAE)",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen (Agree and Access; LoRA 1,3 GB)",
            "https://huggingface.co/settings/tokens (token Read → „hf auth login”)"],
    "videomama": ["https://github.com/okdalto/ComfyUI-VideoMaMa (węzeł; Doktor klonuje go do custom_nodes)",
                  "https://huggingface.co/SammyLim/VideoMaMa (UNet 6,1 GB)",
                  "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt (tylko image_encoder + vae fp16)"],
    "flux3_klein": ["https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy (akceptacja dostępu; model 11,1 GB, "
                    "koder 8,9 GB, VAE 0,34 GB)",
                    "https://huggingface.co/settings/tokens (token Read → „hf auth login”)"],
    "klein_restore_lora": ["https://huggingface.co/ausboss/FLUX-3-Image-Klein-Restore-LoRA (161 MB → models/loras/)"],
    "comfy_version": ["https://www.comfy.org/download (ComfyUI Desktop – aktualizacja w aplikacji)",
                      "https://github.com/comfyanonymous/ComfyUI (ComfyUI z gita: git pull)"],
    "colmap": ["https://github.com/colmap/colmap/releases/tag/4.2.1 (colmap-x64-windows-cuda.zip, 415 MB)"],
    "prompt_assist": ["https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct (8,9 GB → models/LLM/Qwen3-VL-4B-Instruct/, "
                      "Apache-2.0)"],
    "qwen21": ["https://huggingface.co/Comfy-Org/Qwen-Image-2.1 (model int8 7,3 GB → diffusion_models/, koder "
               "qwen3vl_8b int8 9,4 GB → text_encoders/, VAE 0,68 GB → vae/)",
               "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE (Qwen Research License)"],
    "void": ["https://github.com/Netflix/void-model (kod; Doktor klonuje go do bin/void)",
             "https://huggingface.co/netflix/void-model (void_pass1.safetensors, 10,4 GB)",
             "https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP (model bazowy, 19,7 GB)",
             "https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE (CogVideoX License)"],
    "vace_lightning": ["https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged (split_files/loras: "
                       "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise / low_noise, 2 × 1,23 GB)"],
    "vace": ["https://huggingface.co/QuantStack/Wan2.2-VACE-Fun-A14B-GGUF (HighNoise / LowNoise Q4_K_S, 2 × 10,6 GB)",
             "https://github.com/city96/ComfyUI-GGUF (węzeł UnetLoaderGGUF)",
             "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged (umt5_xxl_fp8, wan_2.1_vae)"],
    "h3": ["https://huggingface.co/Comfy-Org/MiniMax-H3 (diffusion_models/minimax_h3_fl2va_pruned_int8_convrot 19,5 GB, "
           "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot 25,3 GB albo nvfp4 14,6 GB – tylko RTX 50, "
           "vae/minimax_h3_video_vae_fp16 4,9 GB, loras/minimax_h3_fl2v_turbo_8step 1,8 GB; opcjonalnie "
           "model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot 4,3 GB – Fun ControlNet, jeszcze "
           "nieużywany przez żaden etap)"],
    "seedvr2": ["https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (węzeł ComfyUI; MatteDesk woła jego inference_cli.py)",
                "https://huggingface.co/numz/SeedVR2_comfyUI (seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors 7,7 GB + "
                "ema_vae_fp16.safetensors 0,5 GB)"],
}

KLEIN_ITEM = "Model: FLUX 3 Image Klein"
KLEIN_LORA_ITEM = "Model: LoRA Restore (FLUX 3 Image Klein)"
COMFY_VERSION_ITEM = "Wersja ComfyUI (FLUX 3 Image Klein)"
COLMAP_ITEM = "COLMAP (CUDA)"
QWEN21_ITEM = "Model: Qwen-Image-2.1"
VOID_ITEM = "VOID (usuwanie obiektu z wideo)"            # = engine/stage_models.TOOL_LABEL (the gate names it)
SEEDVR2_ITEM = "SeedVR2 (podbicie kluczy)"
H3_ITEM = "Model: MiniMax-H3"
VACE_ITEM = "Model: Wan 2.2 Fun VACE"
VACE_LIGHTNING_ITEM = "LoRA Lightning (Wan 2.2 Fun VACE)"
LTX_INPAINT_ITEM = "LoRA inpaintu LTX-2.5 (Inpaint C1)"
COMFY_UPDATE_ITEM = "Aktualizacja ComfyUI"
COMFY_ROLLBACK_ITEM = "Cofnij aktualizację ComfyUI"
PROMPT_ASSIST_ITEM = "Model: Qwen3-VL-4B (asystent opisu)"
PROMPT_ASSIST_NEED = ("Wymagane teraz: NIE – tylko przycisk „✨ Popraw opis” w palecie Inpaint (C1): tłumaczy opis na "
                      "angielski i przepisuje go według zasad MiniMax-H3 / Wan 2.2 Fun VACE / Qwen-Image-2.1, patrząc na "
                      "klatkę z zaznaczonym obiektem. Bez niego zostaje sprawdzanie opisu bez modelu.")
QWEN21_NEED = ("Wymagane teraz: NIE – tylko dogenerowanie klucza w trybie Czyszczenie (silnik „Qwen-Image-2.1”, zalecany "
               "do usuwania obiektu z dużych obszarów; bez niego zostają LaMa i FLUX 3 Image Klein + LoRA Restore).")
# Which MatteDesk module needs the component (user 2026-11-10: the Doctor splits the models by module – Alpha = masks /
# mattes, Cleanup = plate cleanup / Cleanup 3D; "both" = shown under either; anything not listed = "system", the base
# runtime: Python, ComfyUI, ffmpeg, Git, GPU, disk). A new Doctor item for a model = add it here
# (tests/test_vram_fit.py::test_every_model_row_has_a_module).
MODULE_OF = {
    "SAM 3.1 Core Nodes": "alpha", "SAM 3.1 Node Signatures": "alpha", "Model: SAM 3.1": "alpha",
    "ComfyUI-PlateMatte": "alpha", "ComfyUI-MatAnyone2-Keyframes": "alpha", "Węzły MatteDesk (pliki)": "alpha",
    "Custom Nodes (Filesystem)": "alpha", "ComfyUI-MatAnyone2": "alpha", "MatAnyone2 VRAM Patch": "alpha",
    "Model: MatAnyone2": "alpha", VHS_LABEL: "alpha", "Model: LTX-2.5 Alpha-Gen": "alpha", "Model: VideoMaMa": "alpha",
    "Model: ViTMatte": "alpha", BIREFNET_LABEL: "alpha", SAPIENS2_LABEL: "alpha",
    "Model: CoTracker3": "both", "ComfyUI-DepthCrafter-Nodes": "both", "Inpaint LaMa (comfyui-inpaint-nodes)": "both",
    "Model: LaMa (big-lama.pt)": "both", "Wyrównanie na GPU": "both",
    "Wersja ComfyUI (FLUX 3 Image Klein)": "cleanup", "Model: FLUX 3 Image Klein": "cleanup",
    "Model: LoRA Restore (FLUX 3 Image Klein)": "cleanup", "COLMAP (CUDA)": "cleanup", "Model: Qwen-Image-2.1": "cleanup",
    "VOID (usuwanie obiektu z wideo)": "cleanup", "SeedVR2 (podbicie kluczy)": "cleanup", "Model: MiniMax-H3": "cleanup", "Model: Wan 2.2 Fun VACE": "cleanup", "LoRA Lightning (Wan 2.2 Fun VACE)": "cleanup", "LoRA inpaintu LTX-2.5 (Inpaint C1)": "cleanup",
    "Model: Qwen3-VL-4B (asystent opisu)": "cleanup",
}

KLEIN_NEED = ("Wymagane teraz: NIE – tylko dogenerowanie klucza w trybie Czyszczenie (silnik „FLUX 3 Image Klein + "
              "LoRA Restore”; bez niego zostaje LaMa).")


def parse_version(v: Any) -> Optional[tuple]:
    """'0.44.2' / 'v0.41' -> (0, 44, 2) / (0, 41, 0); None when not a version."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(v or ""))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


class EnvironmentDoctor:
    def __init__(self, config_path: Optional[Path] = None, simulation: Optional[Dict[str, Any]] = None):
        self.repo_dir = Path(__file__).resolve().parent
        self.config_path = config_path or (self.repo_dir / "config.json")
        self.simulation = simulation or {}
        self.comfy_vram_gb: Optional[float] = None
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """config.json + config.local.json (per machine) + Desktop autodetect, relative paths resolved."""
        from engine.config import LOCAL, load_config
        local = LOCAL if self.config_path == self.repo_dir / "config.json" else self.config_path.with_name("config.local.json")
        return load_config(self.config_path, local)

    def run_all_checks(self) -> DoctorReport:
        items: List[CheckItem] = []

        # 1. Python & Virtual Environment
        items.append(self.check_python_version())
        items.append(self.check_venv())

        src = self.config.get("_sources", {})
        items.append(CheckItem(
            name="Konfiguracja", category="Config", status="ok" if self.config.get("comfy_dir") else "warning",
            message=(f"comfy_dir = {self.config.get('comfy_dir') or '(nie ustawiony)'} [{src.get('comfy_dir', '?')}], "
                     f"models_dir = {self.config.get('models_dir') or '(nie ustawiony)'} [{src.get('models_dir', '?')}], "
                     f"comfy_port = {self.config.get('comfy_port')} [{src.get('comfy_port', '?')}]"),
            details={"sources": src},
            fix_suggestion=None if self.config.get("comfy_dir") else
            "Ustaw comfy_dir w config.local.json (ścieżka może być względna do folderu MatteDesk, np. \"../ComfyUI\") "
            "albo uruchom: python doctor.py --write-local"))

        # 2. GPU & CUDA
        items.append(self.check_gpu())

        # 3. ffmpeg + Git (the Doctor installs both: engine/tools_install.py)
        items.append(self.check_ffmpeg())
        items.append(self.check_git())

        # ComfyUI not installed at all -> link to ComfyUI Desktop (the Doctor does not install ComfyUI itself)
        comfy_missing = self.check_comfy_installed()
        if comfy_missing:
            items.append(comfy_missing)

        # 4. ComfyUI Environment & Flavor
        flavor_item, detected_flavor = self.check_comfy_flavor()
        items.append(flavor_item)

        # 5. ComfyUI Service Connectivity & Port
        conn_item, active_url = self.check_comfy_connectivity()
        if comfy_missing and not active_url:
            conn_item.action = None   # nothing to start yet: install ComfyUI first (item above)
            conn_item.fix_suggestion = "Najpierw zainstaluj ComfyUI (pozycja „ComfyUI (instalacja)”)."
        items.append(conn_item)
        items.append(self.check_comfy_version((conn_item.details or {}).get("stats") if active_url else None))
        items.extend(self.check_comfy_update())

        # 6. ComfyUI Nodes & SAM3 Signatures
        items.extend(self.check_comfy_nodes(active_url))

        # 7. MatAnyone2 VRAM Patch
        items.append(self.check_matanyone_vram_patch())

        # 8. Model Weights
        items.extend(self.check_model_weights())

        # 9. Disk Space & Storage
        items.append(self.check_disk_space())

        # Our node packages on disk (copied by the "Zainstaluj" button)
        items.append(self.check_node_files())
        items.append(self.check_report_endpoint())
        model_actions = {"Model: LTX-2.5 Alpha-Gen": "install_ltx",
                         "Model: VideoMaMa": "install_videomama",
                         "Model: CoTracker3": "install_cotracker", "Model: ViTMatte": "install_vitmatte", "Model: SAM 3.1": "install_sam3",
                         BIREFNET_LABEL: "install_birefnet", SAPIENS2_LABEL: "install_sapiens2",
                       "Model: MatAnyone2": "install_matanyone2_weights", "Model: LaMa (big-lama.pt)": "install_lama",
                         KLEIN_ITEM: "install_flux3_klein", KLEIN_LORA_ITEM: "install_klein_restore_lora",
                         QWEN21_ITEM: "install_qwen21", PROMPT_ASSIST_ITEM: "install_prompt_assist"}
        node_actions = {"ComfyUI-MatAnyone2": "install_matanyone2_node",
                        "Inpaint LaMa (comfyui-inpaint-nodes)": "install_inpaint_nodes",
                        "ComfyUI-DepthCrafter-Nodes": "install_depthcrafter_nodes",
                        VHS_LABEL: "install_vhs_nodes"}
        for item in items:
            if item.name in model_actions and item.status != "ok" and self.config.get("models_dir"):
                item.action = model_actions[item.name]
            if item.name in node_actions and item.status != "ok" and self.config.get("comfy_dir"):
                item.action = node_actions[item.name]
            if item.name in ("ComfyUI-PlateMatte", "ComfyUI-MatAnyone2-Keyframes", "Custom Nodes (Filesystem)") \
                    and item.status != "ok" and self.config.get("comfy_dir"):
                item.action = "install_nodes"
            if item.name == COLMAP_ITEM and item.status != "ok":
                item.action = "install_colmap"

        # Installed in this session (or on disk but not loaded) and ComfyUI not restarted since -> "Zrestartuj ComfyUI"
        try:
            from engine.installer import pending_restart
            pending = pending_restart(active_url)
        except Exception:
            pending = {}
        for item in items:
            if item.status == "ok":
                continue
            if (item.details or {}).get("missing_modules"):
                continue   # checked in ComfyUI's Python directly: a restart does not install them, "Zainstaluj" does
            what = (item.action or "").replace("install_", "", 1)
            if (what and what in pending) or (item.details or {}).get("needs_restart"):
                item.action = "restart_comfy"
                item.message = "✔ Zainstalowane – ComfyUI musi się zrestartować, żeby to załadować. (" + item.message + ")"
                item.fix_suggestion = "Kliknij „Zrestartuj ComfyUI” (albo zamknij i otwórz ComfyUI Desktop), potem „Odśwież test”."
                item.details = {**(item.details or {}), "needs_restart": True}

        # Where to get what is missing
        link_keys = {"Python Version": "python", "ffmpeg": "ffmpeg", "Git": "git", "ComfyUI Service": "comfyui", "ComfyUI Flavor": "comfyui",
                     "SAM 3.1 Core Nodes": "comfyui", "SAM 3.1 Node Signatures": "comfyui", "Model: SAM 3.1": "sam3",
                     "ComfyUI-MatAnyone2": "matanyone2_node", "Model: MatAnyone2": "matanyone2_weights",
                     "Model: CoTracker3": "cotracker3", "Model: ViTMatte": "vitmatte", "ComfyUI-DepthCrafter-Nodes": "depthcrafter",
                     BIREFNET_LABEL: "birefnet", SAPIENS2_LABEL: "sapiens2",
                     "ComfyUI-PlateMatte": "local_nodes", "ComfyUI-MatAnyone2-Keyframes": "local_nodes",
                     "Custom Nodes (Filesystem)": "local_nodes",
                     "Inpaint LaMa (comfyui-inpaint-nodes)": "inpaint_lama", "Model: LaMa (big-lama.pt)": "inpaint_lama",
                     "Model: LTX-2.5 Alpha-Gen": "ltx", "Model: VideoMaMa": "videomama", VHS_LABEL: "vhs",
                     KLEIN_ITEM: "flux3_klein", KLEIN_LORA_ITEM: "klein_restore_lora", COLMAP_ITEM: "colmap",
                     COMFY_VERSION_ITEM: "comfy_version", QWEN21_ITEM: "qwen21", VOID_ITEM: "void",
                     PROMPT_ASSIST_ITEM: "prompt_assist",
                     SEEDVR2_ITEM: "seedvr2", H3_ITEM: "h3", VACE_ITEM: "vace", VACE_LIGHTNING_ITEM: "vace_lightning", LTX_INPAINT_ITEM: "ltx_fill"}
        lic_keys = {**link_keys, KLEIN_LORA_ITEM: "flux3_klein", COMFY_VERSION_ITEM: ""}
        for item in items:
            # an optional model not installed yet (info + „Zainstaluj”) keeps its „where to get it” too (2026-10-23:
            # ViTMatte became optional and lost its link)
            missing = item.status in ("error", "warning") or (item.status == "info" and item.action)
            if missing and item.name in link_keys and not item.links:
                item.links = list(LINKS[link_keys[item.name]])
            lic = LICENSES.get(lic_keys.get(item.name, ""))
            if lic and not item.license:
                item.license, item.license_nc = lic
                item.license_limited = lic_keys.get(item.name) in LICENSE_LIMITED

        # Will it fit THIS card (user 2026-11-10: warn at the install what spills or runs unstably here)
        items_by_name = {i.name: i for i in items}
        for item in items:
            item.module = MODULE_OF.get(item.name, "system")
        try:
            from engine import vram_fit
            card = vram_fit.card_gb(self.comfy_vram_gb) if not self.simulation.get("simulate_no_gpu") else None
            for name, key in link_keys.items():
                if name in items_by_name and key in vram_fit.rated_keys():
                    items_by_name[name].vram = vram_fit.assess(key, card)
        except Exception:
            pass

        # ComfyUI update / rollback at the very top (user 2026-11-11): they are actions on the whole environment, not
        # one component – seen first, not between the version and the node checks
        top = (COMFY_UPDATE_ITEM, COMFY_ROLLBACK_ITEM)
        items = [i for n in top for i in items if i.name == n] + [i for i in items if i.name not in top]

        # Compute summary
        counts = {"ok": 0, "warning": 0, "error": 0, "info": 0}
        for item in items:
            counts[item.status] = counts.get(item.status, 0) + 1

        all_ok = counts["error"] == 0

        report = DoctorReport(
            timestamp=datetime.datetime.now().isoformat(),
            all_ok=all_ok,
            summary=counts,
            comfy_flavor=detected_flavor,
            comfy_url=active_url,
            items=items,
            licences=licence_summary(),
        )

        return report

    def check_python_version(self) -> CheckItem:
        ver = sys.version_info
        ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        if ver.major == 3 and ver.minor >= 11:
            return CheckItem(
                name="Python Version",
                category="Runtime",
                status="ok",
                message=f"Python {ver_str} (>= 3.11 wymagane)",
                details={"version": ver_str, "executable": sys.executable}
            )
        else:
            return CheckItem(
                name="Python Version",
                category="Runtime",
                status="error",
                message=f"Nieobsługiwana wersja Pythona: {ver_str}. Wymagany Python 3.11+.",
                details={"version": ver_str, "executable": sys.executable},
                fix_suggestion="Zainstaluj Python 3.11 i uruchamiaj narzędzie przez odpowiedni interpreter."
            )

    def check_venv(self) -> CheckItem:
        venv_dir = self.repo_dir / ".venv"
        if Path(sys.executable).resolve().parent == Path(__file__).resolve().parent / "python":
            # Windows installer: MatteDesk's own Python next to the code, no .venv needed
            return CheckItem(name="Virtual Environment", category="Runtime", status="ok",
                             message="Wbudowany Python MatteDesk (instalator) – .venv niepotrzebny",
                             details={"prefix": sys.prefix})
        in_venv = (sys.prefix != sys.base_prefix) or venv_dir.exists()
        if in_venv:
            return CheckItem(
                name="Virtual Environment",
                category="Runtime",
                status="ok",
                message="Wirtualne środowisko (.venv) jest obecne i aktywne",
                details={"venv_path": str(venv_dir), "prefix": sys.prefix}
            )
        else:
            return CheckItem(
                name="Virtual Environment",
                category="Runtime",
                status="warning",
                message="Brak wirtualnego środowiska w repozytorium (.venv)",
                details={"venv_path": str(venv_dir)},
                fix_suggestion="Utwórz venv narzędzia: python -m venv --system-site-packages .venv",
                fix_command="python -m venv --system-site-packages .venv"
            )

    def _other_comfy_on_port(self, url: str) -> Optional[str]:
        """Two ComfyUIs on one machine (Desktop, git, portable) may both use 8188: the running server tells its
        custom_nodes folder (/internal/folder_paths). Returns that folder when it is NOT the chosen comfy_dir's."""
        want = self.config.get("comfy_dir") or ""
        if not want:
            return None
        try:
            req = urllib.request.Request(f"{url}/internal/folder_paths", headers={"User-Agent": "MatteDeskDoctor"})
            with urllib.request.urlopen(req, timeout=1.5) as r:
                dirs = json.loads(r.read().decode("utf-8")).get("custom_nodes") or []
        except Exception:
            return None   # older ComfyUI without the route: cannot tell, do not warn
        norm = lambda p: os.path.normcase(os.path.normpath(str(p)))
        if not dirs or norm(Path(want) / "custom_nodes") in {norm(d) for d in dirs}:
            return None
        return str(Path(dirs[0]).parent)

    def check_gpu(self) -> CheckItem:
        """GPU as ComfyUI sees it (/system_stats): all GPU work (SAM 3.1, MatAnyone2, DepthCrafter, CoTracker3) runs in
        ComfyUI's Python – MatteDesk's own .venv has no PyTorch."""
        if self.simulation.get("simulate_no_gpu"):
            return CheckItem(name="GPU & CUDA", category="Hardware", status="warning",
                             message="Brak akceleracji CUDA. Tylko CPU (podgląd/QA).", details={"cuda_available": False})
        url = f"http://{self.config.get('comfy_host', '127.0.0.1')}:{self.config.get('comfy_port', 8188)}/system_stats"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "MatteDeskDoctor"}), timeout=2) as r:
                stats = json.loads(r.read().decode("utf-8"))
        except Exception:
            return CheckItem(name="GPU & CUDA", category="Hardware", status="info",
                             message="GPU sprawdzane przez ComfyUI – ComfyUI nie odpowiada (uruchom go).")
        devs = [d for d in stats.get("devices", []) if str(d.get("type", "")).lower() == "cuda"]
        torch_v = (stats.get("system") or {}).get("pytorch_version", "?")
        if not devs:
            return CheckItem(name="GPU & CUDA", category="Hardware", status="warning",
                             message=f"ComfyUI nie widzi karty CUDA (PyTorch {torch_v}) – modele liczą się na CPU, bardzo wolno.",
                             details={"pytorch": torch_v})
        d = max(devs, key=lambda x: x.get("vram_total", 0))
        # "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync" -> "NVIDIA GeForce RTX 4090"
        name = re.sub(r"^\w+:\d+\s*", "", str(d.get("name", "GPU"))).split(" : ")[0].strip() or "GPU"
        vram = d.get("vram_total", 0) / 1024 ** 3
        self.comfy_vram_gb = vram
        return CheckItem(name="GPU & CUDA", category="Hardware", status="ok",
                         message=f"{name} ({vram:.1f} GB VRAM, PyTorch {torch_v} w ComfyUI) – na nim liczą się SAM 3.1, MatAnyone2 i CoTracker3",
                         details={"device_name": name, "vram_gb": round(vram, 2), "pytorch": torch_v})

    def check_report_endpoint(self) -> CheckItem:
        """„Zgłoś problem” (engine/report.py): the report proxy from config.json `report_endpoint` (Cloudflare Worker,
        infra/report_proxy/). Optional – without it the rest of the app works; reports cannot be sent."""
        from engine.report import endpoint, health
        url = endpoint(self.config)
        need = "Wymagane teraz: NIE – tylko do przycisku „Zgłoś problem” (wysyła opis i diagnostykę jako zgłoszenie)."
        if not url:
            return CheckItem(name="Zgłaszanie problemów", category="Config", status="info",
                             message=f"Serwer zgłoszeń nie jest ustawiony (report_endpoint w config.json). {need}")
        ok, why = (True, "pominięte (symulacja)") if self.simulation.get("skip_network") else health(url)
        return CheckItem(name="Zgłaszanie problemów", category="Config", status="ok" if ok else "warning",
                         message=f"Serwer zgłoszeń {url}: {why}. {need}", details={"report_endpoint": url},
                         fix_suggestion=None if ok else "Sprawdź połączenie z internetem albo zaporę; zgłoszenie możesz też "
                                                        "opisać na GitHubie (link w oknie „O programie”).")

    def check_ffmpeg(self) -> CheckItem:
        if self.simulation.get("simulate_no_ffmpeg"):
            return CheckItem(name="ffmpeg", category="Tools", status="error", message="ffmpeg nie jest dostępny",
                             fix_suggestion="Kliknij „Zainstaluj” (winget albo paczka portable z gyan.dev).",
                             fix_command="winget install Gyan.FFmpeg", action="install_ffmpeg")
        from engine.tools_install import find_tool
        path = find_tool("ffmpeg", self.config)
        if path:
            try:
                res = subprocess.run([path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                if res.returncode == 0:
                    first = res.stdout.splitlines()[0] if res.stdout else "ffmpeg"
                    return CheckItem(name="ffmpeg", category="Tools", status="ok", message=f"{first} ({path})",
                                     details={"path": path})
            except Exception:
                pass
        return CheckItem(
            name="ffmpeg", category="Tools", status="error",
            message="ffmpeg nie został znaleziony (PATH, config, typowe foldery). Wymagane teraz: TAK – wczytywanie ujęć "
                    "wideo (MP4 / MOV) i podglądy.",
            fix_suggestion="Kliknij „Zainstaluj” – Doktor zainstaluje ffmpeg przez winget (ok. 100 MB), a gdy winget nie "
                           "działa, rozpakuje paczkę portable z gyan.dev do bin/ffmpeg i zapisze ścieżkę w config.local.json.",
            fix_command="winget install Gyan.FFmpeg", action="install_ffmpeg")

    def check_git(self) -> CheckItem:
        from engine.tools_install import find_tool
        path = None if self.simulation.get("simulate_no_git") else find_tool("git", self.config)
        if path:
            return CheckItem(name="Git", category="Tools", status="ok", message=f"Git: {path}", details={"path": path})
        return CheckItem(
            name="Git", category="Tools", status="error",
            message="Git nie został znaleziony. Wymagane teraz: TAK – Doktor klonuje nim węzły ComfyUI z GitHuba "
                    "(ComfyUI-MatAnyone2, VideoHelperSuite, DepthCrafter), a pip instaluje z GitHuba CoTracker3 i pakiety MatAnyone2.",
            fix_suggestion="Kliknij „Zainstaluj” – Doktor zainstaluje Git for Windows przez winget (ok. 60 MB; Windows może "
                           "zapytać o zgodę administratora).",
            fix_command="winget install Git.Git", action="install_git")

    def check_comfy_installed(self) -> Optional[CheckItem]:
        """ComfyUI itself is not installed by the Doctor: when none is found, say so first and link the download."""
        if self.simulation.get("simulate_flavor"):
            return None
        flavor = self.config.get("comfy_flavor", "desktop")
        d = Path(self.config.get("comfy_dir") or "")
        if flavor == "git":
            if str(d) and (d / "main.py").exists():
                return None
            msg = (f"Folder ComfyUI z gita / portable nie zawiera main.py ({d})." if str(d) else
                   "Wybrano ComfyUI z gita / portable, ale nie podano folderu.")
            tip = ("Podaj u góry folder z main.py (w paczce portable: ComfyUI_windows_portable\\ComfyUI) i port, albo "
                   "przełącz na ComfyUI Desktop i zainstaluj go z linku niżej.")
        else:
            if str(d) and d.is_dir():
                return None
            msg = "Nie znaleziono ComfyUI Desktop na tym komputerze. MatteDesk liczy wszystkie modele w ComfyUI – bez niego nic nie policzy."
            tip = ("Pobierz i zainstaluj ComfyUI Desktop (link niżej), uruchom go raz, zamknij, potem kliknij „Odśwież test” – "
                   "MatteDesk sam znajdzie jego foldery i port. Masz ComfyUI z gita? Przełącz wariant u góry.")
        return CheckItem(name="ComfyUI (instalacja)", category="ComfyUI", status="error", message=msg, fix_suggestion=tip,
                         links=list(LINKS["comfyui_install"]))

    def check_node_files(self) -> CheckItem:
        from engine.installer import node_status, node_versions
        st = node_status(self.config.get("comfy_dir", ""))
        vers = node_versions(self.config.get("comfy_dir", ""))
        if all(v == "no_comfy" for v in st.values()):
            return CheckItem(name="Węzły MatteDesk (pliki)", category="ComfyUI Nodes", status="warning",
                             message="Nie znaleziono folderu custom_nodes ComfyUI – ustaw comfy_dir.", details=st)
        bad = {k: v for k, v in st.items() if v != "same"}
        names = {"missing": "brak", "different": "inna wersja niż w repo"}

        def _ver(k: str, v: str) -> str:
            ins, repo = vers.get(k, {}).get("installed"), vers.get(k, {}).get("repo")
            if v == "different" and repo:
                return f" (zainstalowana {ins or 'starsza, bez numeru'} → repo {repo})"
            return ""
        return CheckItem(
            name="Węzły MatteDesk (pliki)", category="ComfyUI Nodes", status="error" if bad else "ok",
            message=("; ".join(f"{k}: {names.get(v, v)}{_ver(k, v)}" for k, v in bad.items()) if bad
                     else "PlateMatte i MatAnyone2-Keyframes zgodne z repo"),
            details={**st, "versions": vers},
            fix_suggestion="Kliknij „Zainstaluj” (kopia z handoff/comfy/custom_nodes, stara wersja do backups/), potem restart ComfyUI." if bad else None,
            action="install_nodes" if bad else None)

    def check_comfy_flavor(self) -> tuple[CheckItem, str]:
        sim_flavor = self.simulation.get("simulate_flavor")
        if sim_flavor:
            flavor = sim_flavor.lower()
            flavor_display = "Desktop (restart w aplikacji)" if flavor == "desktop" else "ComfyUI z gita"
            details = {"flavor": flavor, "simulated": True}
            return CheckItem(
                name="ComfyUI Flavor",
                category="ComfyUI",
                status="ok",
                message=f"Wykryto wariant ComfyUI: {flavor_display} (symulacja)",
                details=details
            ), flavor

        comfy_dir = Path(self.config.get("comfy_dir", "ComfyUI"))
        config_flavor = self.config.get("comfy_flavor", "desktop")

        # ComfyUI from git: folder with main.py
        is_git = (comfy_dir / "main.py").exists()
        # Check desktop characteristics (app data or .venv in desktop structure)
        is_desktop = (comfy_dir / ".venv").exists() and not is_git

        if is_desktop or config_flavor == "desktop":
            flavor = "desktop"
            msg = "Wykryto ComfyUI Desktop (restart i zarządzanie procesem po stronie użytkownika)"
        else:
            flavor = "git"
            from engine.comfy_portable import embedded_python
            msg = ("Wykryto ComfyUI portable (katalog z main.py, python_embeded obok)" if embedded_python(comfy_dir)
                   else "Wykryto ComfyUI z gita (katalog z main.py)")

        return CheckItem(
            name="ComfyUI Flavor",
            category="ComfyUI",
            status="ok",
            message=msg,
            details={"flavor": flavor, "comfy_dir": str(comfy_dir)}
        ), flavor

    def _start_hint(self, comfy_port: int) -> tuple[str, Optional[str]]:
        """How to start Comfy: Desktop = the app (port in its settings); git = main.py."""
        comfy_dir = Path(self.config.get("comfy_dir", "ComfyUI"))
        cd_path = str(comfy_dir).replace("/", "\\")
        if self.config.get("comfy_flavor", "desktop") == "desktop" or (
                (comfy_dir / ".venv").exists() and not (comfy_dir / "main.py").exists()):
            from engine.comfy_launch import desktop2_install
            if desktop2_install(self.config) is not None:
                # Comfy Desktop 2 starts its server only after a click in its window -> MatteDesk starts it headless
                return (f"Comfy Desktop 2: przycisk / autostart uruchamia serwer tej instalacji bez okna aplikacji "
                        f"(własne okno konsoli z logiem, port {comfy_port}; zamknięcie okna zatrzymuje ComfyUI). "
                        f"Ręcznie: otwórz Comfy Desktop i kliknij instalację – wtedy port w Desktopie musi być "
                        f"równy `comfy_port` = {comfy_port}."), None
            return (f"Uruchom aplikację ComfyUI Desktop. Port serwera: Ustawienia → Server-Config → Port "
                    f"(musi być równy `comfy_port` = {comfy_port} w config.json albo zmień config.json)."), None
        return (f"Uruchom ComfyUI na porcie {comfy_port}: cd {cd_path} && python main.py --port {comfy_port}",
                f"cd {cd_path}; python main.py --port {comfy_port}")

    def check_comfy_connectivity(self) -> tuple[CheckItem, Optional[str]]:
        comfy_port = self.config.get("comfy_port", 8000)
        hint, cmd = self._start_hint(comfy_port)

        if self.simulation.get("simulate_comfy_offline"):
            return CheckItem(
                name="ComfyUI Service",
                category="ComfyUI",
                status="warning",
                message="ComfyUI nie odpowiada (symulacja offline).",
                fix_suggestion=hint, fix_command=cmd
            ), None

        host = self.config.get("comfy_host", "127.0.0.1")
        ports_to_try = [
            comfy_port,
            8001,
            8000,
            8188
        ]
        # Deduplicate while preserving order
        seen = set()
        ports_to_try = [p for p in ports_to_try if not (p in seen or seen.add(p))]

        for p in ports_to_try:
            url = f"http://{host}:{p}"
            try:
                req = urllib.request.Request(f"{url}/system_stats", headers={"User-Agent": "MatteDeskDoctor"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        try:   # GPU of the Comfy process (not this venv's torch) -> decides how serious the VRAM patch is
                            self.comfy_vram_gb = max(d.get("vram_total", 0) for d in data.get("devices", [])) / 1024 ** 3
                        except (ValueError, TypeError):
                            pass
                        if p != comfy_port:  # the tool talks to the configured port only
                            other = self._other_comfy_on_port(url)   # which install answers there (2026-10-23)
                            if other:
                                return CheckItem(
                                    name="ComfyUI Service", category="ComfyUI", status="error",
                                    message=f"Na porcie {comfy_port} (ustawionym) ComfyUI nie odpowiada. Na {url} działa INNY "
                                            f"ComfyUI ({other}) – nie ten wybrany ({self.config.get('comfy_dir') or '-'}).",
                                    details={"url": url, "running_custom_nodes": other},
                                    fix_suggestion="Uruchom wybrany ComfyUI („▶ Uruchom ComfyUI”) albo w „Używany ComfyUI” "
                                                   f"wybierz tamten (folder {other}, port {p})."
                                ), None
                            return CheckItem(
                                name="ComfyUI Service", category="ComfyUI", status="error",
                                message=f"Wybrany ComfyUI działa na {url}, ale w MatteDesk ustawiony jest port {comfy_port}.",
                                details={"url": url, "stats": data},
                                fix_suggestion=f"W Doktorze → „Używany ComfyUI” ustaw port {p} i „Zapisz”."
                            ), url
                        other = self._other_comfy_on_port(url)
                        if other:
                            return CheckItem(
                                name="ComfyUI Service", category="ComfyUI", status="error",
                                message=f"Na {url} działa INNY ComfyUI ({other}) niż wybrany w MatteDesk "
                                        f"({self.config.get('comfy_dir') or '-'}). Węzły i modele trafią do wybranego, "
                                        f"a liczy ten działający.",
                                details={"url": url, "running_custom_nodes": other},
                                fix_suggestion="Zamknij tamten ComfyUI albo w „Używany ComfyUI” wybierz go "
                                               "(folder / port), albo uruchom go na innym porcie."
                            ), url
                        return CheckItem(
                            name="ComfyUI Service",
                            category="ComfyUI",
                            status="ok",
                            message=f"ComfyUI aktywne na {url} (wersja/system: OK)",
                            details={"url": url, "stats": data}
                        ), url
            except Exception:
                continue

        return CheckItem(
            name="ComfyUI Service",
            category="ComfyUI",
            status="warning",
            message=f"ComfyUI nie odpowiada na żadnym z portów ({ports_to_try}).",
            fix_suggestion="Kliknij „▶ Uruchom ComfyUI” (uruchomi wybrany niżej ComfyUI). " + hint, fix_command=cmd,
            action="start_comfy"
        ), None

    # Doctor item -> folder of the node pack in custom_nodes (for "on disk but not loaded" diagnostics)
    NODE_PACK_DIRS = {"ComfyUI-PlateMatte": "ComfyUI-PlateMatte", "ComfyUI-MatAnyone2-Keyframes": "ComfyUI-MatAnyone2-Keyframes",
                      "ComfyUI-MatAnyone2": "ComfyUI-MatAnyone2", "Inpaint LaMa (comfyui-inpaint-nodes)": "comfyui-inpaint-nodes",
                      "ComfyUI-DepthCrafter-Nodes": "ComfyUI-DepthCrafter-Nodes",
                      VHS_LABEL: "ComfyUI-VideoHelperSuite"}

    def comfy_import_errors(self, active_url: Optional[str]) -> Dict[str, str]:
        """{node pack folder: reason} from ComfyUI's log (/internal/logs): "Cannot import <dir> module for custom nodes: <err>"."""
        if not active_url or self.simulation:
            return {}
        try:
            req = urllib.request.Request(f"{active_url}/internal/logs", headers={"User-Agent": "MatteDeskDoctor"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return {}
        try:   # Comfy returns a JSON string
            text = json.loads(text, strict=False) if text.startswith('"') else text
        except ValueError:
            pass
        out: Dict[str, str] = {}
        for m in re.finditer(r"Cannot import (.+?) module for custom nodes: ([^\r\n]+)", str(text)):
            out[Path(m.group(1).strip()).name] = m.group(2).strip()
        return out

    def _missing_module_now_installed(self, error: str) -> bool:
        """"No module named 'cv2'" from ComfyUI's log, but ComfyUI's Python imports cv2 now -> the error is stale."""
        m = re.search(r"No module named '([\w.]+)'", error or "")
        if not m:
            return False
        try:
            from engine.installer import comfy_python
            py = comfy_python(self.config)
            if py is None:
                return False
            r = subprocess.run([str(py), "-c", f"import {m.group(1).split('.')[0]}"], capture_output=True, timeout=60)
            return r.returncode == 0
        except Exception:
            return False

    def comfy_missing_modules(self, modules: List[str]) -> Optional[List[str]]:
        """Modules ComfyUI's Python cannot find (importlib find_spec: fast, imports nothing). None = not checked."""
        if "comfy_modules" in self.simulation:   # tests: {module: installed}
            return [m for m in modules if not self.simulation["comfy_modules"].get(m)]
        if self.simulation:
            return None
        try:
            from engine.installer import comfy_python
            py = comfy_python(self.config)
            if py is None:
                return None
            code = ("import importlib.util, sys; "
                    "print(','.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))")
            r = subprocess.run([str(py), "-c", code, *modules], capture_output=True, text=True, timeout=60)
            if r.returncode:
                return None
            return [m for m in r.stdout.strip().split(",") if m]
        except Exception:
            return None

    def check_comfy_nodes(self, active_url: Optional[str]) -> List[CheckItem]:
        items = self._check_comfy_nodes(active_url)
        for it in items:
            if it.name == "ComfyUI-MatAnyone2" and it.status == "ok":
                miss = self.comfy_missing_modules(MATANYONE_PY_MODULES)
                if miss:
                    it.status = "error"
                    it.message = (f"Węzeł LoadMatAnyone2Model jest zarejestrowany, ale w Pythonie ComfyUI brak pakietów {miss} – "
                                  "MatAnyone2 („Przelicz finalnie”, strefy) pada: „No module named 'matanyone2'”.")
                    it.fix_suggestion = ("Wymagane teraz: TAK – MatAnyone2 (rdzeń → finalnie, strefy). Kliknij „Zainstaluj”: uruchomi "
                                         "install.py węzła w Pythonie ComfyUI (matanyone2 z GitHub pq-yang/MatAnyone2, hydra-core, "
                                         "thinplate; kilka MB, PyTorch bez zmian). Potem „Odśwież test”.")
                    it.details = {**(it.details or {}), "missing_modules": miss}
        if not active_url:
            return items
        errors = self.comfy_import_errors(active_url)
        cn = Path(self.config.get("comfy_dir") or "") / "custom_nodes"
        for it in items:
            folder = self.NODE_PACK_DIRS.get(it.name)
            if not folder or it.status == "ok" or not self.config.get("comfy_dir") or (it.details or {}).get("missing_modules"):
                continue   # missing_modules: the node IS loaded, its Python packages are not -> no "restart" advice
            if folder in errors and self._missing_module_now_installed(errors[folder]):
                # the log line is from before our dependency install (ComfyUI not restarted since) -> restart, not install
                it.message = (f"Pakiet {folder} i jego zależności są już zainstalowane, ale ComfyUI działa jeszcze sprzed instalacji "
                              f"(stary błąd z logu: {errors[folder]}) – ZRESTARTUJ ComfyUI, potem „Odśwież test”.")
                it.fix_suggestion = "Kliknij „Zrestartuj ComfyUI” (albo zamknij i otwórz ComfyUI Desktop)."
                it.details = {**(it.details or {}), "needs_restart": True, "on_disk": True, "stale_import_error": errors[folder]}
            elif folder in errors:
                it.message = f"Pakiet {folder} jest w custom_nodes, ale ComfyUI nie mógł go załadować: {errors[folder]}"
                it.fix_suggestion = ("Kliknij „Zainstaluj” – doinstaluje brakujące zależności do Pythona ComfyUI. "
                                     "Potem ZRESTARTUJ ComfyUI (w aplikacji) i kliknij „Odśwież test”.")
                it.details = {**(it.details or {}), "import_error": errors[folder], "on_disk": True}
            elif (cn / folder).is_dir():
                it.message = (f"Pakiet {folder} jest już w custom_nodes, ale ComfyUI go jeszcze nie załadował – "
                              "ZRESTARTUJ ComfyUI (w aplikacji), potem „Odśwież test”.")
                it.fix_suggestion = "Jeśli po restarcie dalej brak – kliknij „Zainstaluj” (zależności) i zrestartuj ComfyUI ponownie."
                it.details = {**(it.details or {}), "needs_restart": True, "on_disk": True}
        return items

    def _check_comfy_nodes(self, active_url: Optional[str]) -> List[CheckItem]:
        items: List[CheckItem] = []
        object_info: Optional[Dict[str, Any]] = None

        if self.simulation.get("simulate_altered_sam3"):
            object_info = {
                "SAM3_Detect": {"input": {"required": {"model": [], "image": []}}},
                "SAM3_VideoTrack": {"input": {"required": {"model": []}}},
                "SAM3_TrackToMask": {"input": {"required": {}}}
            }
        elif self.simulation.get("simulate_missing_nodes"):
            object_info = {}
        elif "object_info" in self.simulation:   # tests: a mocked /object_info
            object_info = self.simulation["object_info"]
        elif active_url:
            try:
                req = urllib.request.Request(f"{active_url}/object_info", headers={"User-Agent": "MatteDeskDoctor"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    if resp.status == 200:
                        object_info = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass

        # Fallback offline check via custom_nodes folder if Comfy is offline and not in simulation
        comfy_dir = Path(self.config.get("comfy_dir", "ComfyUI"))
        custom_nodes_dir = comfy_dir / "custom_nodes"
        custom_nodes_path = str(custom_nodes_dir).replace("/", "\\")

        # Check SAM3 nodes
        if object_info is not None:
            sam3_missing = []
            sam3_schema_mismatch = []
            for node_name, expected_inputs in EXPECTED_SAM3_INPUTS.items():
                if node_name not in object_info:
                    sam3_missing.append(node_name)
                else:
                    node_desc = object_info[node_name]
                    inputs_dict = node_desc.get("input", {})
                    req = set(inputs_dict.get("required", {}).keys())
                    opt = set(inputs_dict.get("optional", {}).keys())
                    avail = req | opt
                    missing_fields = [f for f in expected_inputs if f not in avail]
                    if missing_fields:
                        sam3_schema_mismatch.append(f"{node_name}: brak {missing_fields}")

            if sam3_missing:
                items.append(CheckItem(
                    name="SAM 3.1 Core Nodes",
                    category="ComfyUI Nodes",
                    status="error",
                    message=f"Brak węzłów SAM 3.1 w ComfyUI: {sam3_missing}",
                    details={"missing": sam3_missing},
                    fix_suggestion="Zaktualizuj ComfyUI do wersji z obsługą SAM 3.1 (nodes_sam3 w comfy_extras)."
                ))
            elif sam3_schema_mismatch:
                items.append(CheckItem(
                    name="SAM 3.1 Node Signatures",
                    category="ComfyUI Nodes",
                    status="error",
                    message=f"Niezgodność wejść węzłów SAM 3.1: {'; '.join(sam3_schema_mismatch)}",
                    details={"mismatches": sam3_schema_mismatch},
                    fix_suggestion="Zaktualizuj ComfyUI: węzły SAM 3.1 w rdzeniu ComfyUI mają inne wejścia, niż oczekuje MatteDesk (initial_mask, conditioning)."
                ))
            else:
                items.append(CheckItem(
                    name="SAM 3.1 Core Nodes",
                    category="ComfyUI Nodes",
                    status="ok",
                    message="Węzły SAM3_Detect, SAM3_VideoTrack, SAM3_TrackToMask są zgodne ze specyfikacją",
                    details={"nodes": list(EXPECTED_SAM3_INPUTS.keys())}
                ))

            # PlateMatte nodes
            pl_missing = [n for n in REQUIRED_PLATEMATTE_NODES if n not in object_info]
            if pl_missing:
                items.append(CheckItem(
                    name="ComfyUI-PlateMatte",
                    category="ComfyUI Nodes",
                    status="error",
                    message=f"Brak węzłów PlateMatte w ComfyUI: {pl_missing}",
                    details={"missing": pl_missing},
                    fix_suggestion=f"Skopiuj ComfyUI-PlateMatte z paczki do {custom_nodes_path}\\ (za zgodą użytkownika)",
                    fix_command=f"Copy-Item -Recurse handoff/comfy/custom_nodes/ComfyUI-PlateMatte {custom_nodes_path}\\"
                ))
            else:
                items.append(CheckItem(
                    name="ComfyUI-PlateMatte",
                    category="ComfyUI Nodes",
                    status="ok",
                    message="Wszystkie węzły ComfyUI-PlateMatte są zarejestrowane",
                    details={"nodes": REQUIRED_PLATEMATTE_NODES}
                ))

            # MatAnyone2-Keyframes
            ma_missing = [n for n in REQUIRED_MATANYONE_KEYFRAMES_NODES if n not in object_info]
            if ma_missing:
                items.append(CheckItem(
                    name="ComfyUI-MatAnyone2-Keyframes",
                    category="ComfyUI Nodes",
                    status="error",
                    message=f"Brak węzłów MatAnyone2-Keyframes: {ma_missing}",
                    details={"missing": ma_missing},
                    fix_suggestion=f"Skopiuj ComfyUI-MatAnyone2-Keyframes z paczki do {custom_nodes_path}\\",
                    fix_command=f"Copy-Item -Recurse handoff/comfy/custom_nodes/ComfyUI-MatAnyone2-Keyframes {custom_nodes_path}\\"
                ))
            else:
                items.append(CheckItem(
                    name="ComfyUI-MatAnyone2-Keyframes",
                    category="ComfyUI Nodes",
                    status="ok",
                    message="Węzły MatAnyone2RunKeyframes i MatAnyone2CoreCombine są zarejestrowane",
                    details={"nodes": REQUIRED_MATANYONE_KEYFRAMES_NODES}
                ))

            # MatAnyone2 base node pack (LoadMatAnyone2Model) and DepthCrafter (4e, optional)
            for label, req, status, fix in (
                ("ComfyUI-MatAnyone2", REQUIRED_MATANYONE_BASE_NODES, "error",
                 f"Kliknij „Zainstaluj” (pobiera węzeł z GitHuba do {custom_nodes_path}\\ComfyUI-MatAnyone2) albo ComfyUI Manager: „matanyone2”."),
                ("ComfyUI-DepthCrafter-Nodes", REQUIRED_DEPTHCRAFTER_NODES, "warning",
                 "Wymagane teraz: NIE – potrzebne tylko do funkcji „Głębia” (etap 0); bez niego reszta programu działa. "
                 "Do czego: mapa głębi ujęcia (etap 0, 4e) – podpowiada kolejność obiektów od przodu i pomaga przy strefach. "
                 "Pobranie: węzeł < 1 MB, model ok. 4–6 GB pobiera się sam przy 1. użyciu. "
                 "Kliknij „Zainstaluj” albo ComfyUI Manager: „DepthCrafter Nodes”."),
                ("Inpaint LaMa (comfyui-inpaint-nodes)", REQUIRED_INPAINT_LAMA_NODES, "warning",
                 "Opcjonalne (clean plate auto – dziury): ComfyUI Manager „ComfyUI Inpaint Nodes” + models/inpaint/big-lama.pt; bez węzła dziury wypełnia OpenCV (pewność 0)."),
                (VHS_LABEL, REQUIRED_VHS_NODES, "error", VHS_FIX),
            ):
                miss = [n for n in req if n not in object_info]
                items.append(CheckItem(
                    name=label, category="ComfyUI Nodes", status=status if miss else "ok",
                    message=f"Brak węzłów: {miss}" if miss else f"Węzły {req} są zarejestrowane",
                    details={"missing": miss}, fix_suggestion=fix if miss else None))
        else:
            # Comfy offline fallback: inspect custom_nodes directory
            has_platematte = (custom_nodes_dir / "ComfyUI-PlateMatte").exists()
            has_keyframes = (custom_nodes_dir / "ComfyUI-MatAnyone2-Keyframes").exists()

            items.append(CheckItem(
                name="Custom Nodes (Filesystem)",
                category="ComfyUI Nodes",
                status="ok" if (has_platematte and has_keyframes) else "warning",
                message=f"Katalogi węzłów: PlateMatte ({'OK' if has_platematte else 'BRAK'}), MatAnyone2-Keyframes ({'OK' if has_keyframes else 'BRAK'})",
                details={"has_platematte": has_platematte, "has_keyframes": has_keyframes}
            ))
            vhs_dir = next((custom_nodes_dir / d for d in VHS_DIRS if (custom_nodes_dir / d).is_dir()), None)
            items.append(CheckItem(
                name=VHS_LABEL, category="ComfyUI Nodes", status="ok" if vhs_dir else "warning",
                message=(f"Katalog {vhs_dir.name} jest w custom_nodes (ComfyUI nie działa – nie sprawdzono, czy się ładuje)"
                         if vhs_dir else f"Brak ComfyUI-VideoHelperSuite w {custom_nodes_path} (VHS_LoadImagesPath)"),
                details={"missing": [] if vhs_dir else REQUIRED_VHS_NODES}, fix_suggestion=None if vhs_dir else VHS_FIX))

        return items

    def check_matanyone_vram_patch(self) -> CheckItem:
        if self.simulation.get("simulate_missing_patch"):
            return CheckItem(
                name="MatAnyone2 VRAM Patch",
                category="Patches",
                status="error",
                message="Brak łatki VRAM max_mem_frames=5 w ComfyUI-MatAnyone2/nodes.py (symulacja brakującej łatki).",
                fix_suggestion="Przywróć łatkę VRAM (max_mem_frames=5, empty_cache).",
                fix_command="# Propozycja: Zaaplikuj łatkę z folderu patches/ za zgodą użytkownika"
            )

        comfy_dir = Path(self.config.get("comfy_dir", "ComfyUI"))
        nodes_file = comfy_dir / "custom_nodes" / "ComfyUI-MatAnyone2" / "nodes.py"

        if not nodes_file.exists():
            return CheckItem(
                name="MatAnyone2 VRAM Patch",
                category="Patches",
                status="warning",
                message=f"Nie znaleziono pliku {nodes_file}",
                details={"path": str(nodes_file)}
            )

        try:
            content = nodes_file.read_text(encoding="utf-8", errors="ignore")
            if "max_mem_frames = 5" in content or "max_mem_frames=5" in content:
                return CheckItem(
                    name="MatAnyone2 VRAM Patch",
                    category="Patches",
                    status="ok",
                    message="Łatka VRAM obecna (cfg.max_mem_frames = 5)",
                    details={"path": str(nodes_file)}
                )
            else:
                big = self.comfy_vram_gb is not None and self.comfy_vram_gb >= BIG_GPU_GB
                return CheckItem(
                    name="MatAnyone2 VRAM Patch",
                    category="Patches",
                    status="warning" if big else "error",
                    message=(f"Brak łatki VRAM (max_mem_frames=5) w ComfyUI-MatAnyone2/nodes.py – GPU ma {self.comfy_vram_gb:.0f} GB, "
                             "łatka niepotrzebna (ostrzeżenie)." if big else
                             "Brak łatki VRAM (max_mem_frames=5) w ComfyUI-MatAnyone2/nodes.py. Ryzyko OOM na kartach 12 GB."),
                    details={"path": str(nodes_file)},
                    fix_suggestion=None if big else "Ustaw cfg.max_mem_frames = 5 w ComfyUI-MatAnyone2/nodes.py (patrz reference_matanyone2_comfy.md)."
                )
        except Exception as e:
            return CheckItem(
                name="MatAnyone2 VRAM Patch",
                category="Patches",
                status="warning",
                message=f"Błąd odczytu nodes.py: {e}",
                details={"error": str(e)}
            )

    def prewarm_python_checks(self) -> None:
        """The ComfyUI-Python checks (ViTMatte, BiRefNet, Sapiens2, GPU alignment, CoTracker3) each start ComfyUI's Python and import
        torch (~6 s each). Run them side by side, so the checks below only read their caches (~24 s -> ~7 s cold)."""
        if self.simulation:
            return
        from concurrent.futures import ThreadPoolExecutor
        from engine import align_gpu, birefnet, sapiens2
        from engine import vitmatte as vm
        from engine.installer import comfy_python
        from engine.tracking import check_tracker_python
        py = comfy_python(self.config)
        if py is None:
            return
        jobs =[lambda: vm.check_worker_python(py, [vm.DEFAULT_MODEL]),
                lambda: vm.check_worker_python(py, list(birefnet.MODELS.values()), worker=birefnet.WORKER),
                lambda: vm.check_worker_python(py, [sapiens2.DEFAULT_MODEL], worker=sapiens2.WORKER),
                lambda: align_gpu.check_worker(py),
                lambda: check_tracker_python(py)]
        with ThreadPoolExecutor(len(jobs)) as ex:
            for fut in [ex.submit(j) for j in jobs]:
                try:
                    fut.result()
                except Exception:
                    pass   # the check itself reports it

    def check_model_weights(self) -> List[CheckItem]:
        try:
            self.prewarm_python_checks()
        except Exception:
            pass
        items: List[CheckItem] = []
        models_dir = Path(self.config.get("models_dir", "ComfyUI/models"))
        models_path = str(models_dir).replace("/", "\\")

        if self.simulation.get("simulate_missing_model"):
            models_dir = Path("D:/non_existent_models_dir")
            models_path = str(models_dir).replace("/", "\\")

        # 1. SAM3 – loaded by CheckpointLoaderSimple, so it lives in models/checkpoints (name from config.json)
        sam3_name = self.config.get("sam3_checkpoint", "sam3.1_multiplex_fp16.safetensors")
        sam3_file = models_dir / "checkpoints" / sam3_name
        if sam3_file.exists():
            items.append(CheckItem(
                name="Model: SAM 3.1",
                category="Models",
                status="ok",
                message=f"SAM 3.1 wagi obecne ({sam3_name}, {sam3_file.stat().st_size / (1024**2):.0f} MB)",
                details={"path": str(sam3_file)}
            ))
        else:
            items.append(CheckItem(
                name="Model: SAM 3.1",
                category="Models",
                status="error",
                message=f"Brak wag SAM 3.1: {sam3_file}",
                fix_suggestion=f"Pobierz {sam3_name} do {models_path}\\checkpoints\\ (nazwa pliku = `sam3_checkpoint` w config.json)"
            ))

        # 2. MatAnyone2
        ma2_file = models_dir / "matanyone2" / "matanyone2.pth"
        if ma2_file.exists():
            size_mb = ma2_file.stat().st_size / (1024**2)
            items.append(CheckItem(
                name="Model: MatAnyone2",
                category="Models",
                status="ok",
                message=f"Wagi MatAnyone2 obecne ({size_mb:.1f} MB)",
                details={"path": str(ma2_file)}
            ))
        else:
            items.append(CheckItem(
                name="Model: MatAnyone2",
                category="Models",
                status="error",
                message=f"Brak pliku {ma2_file}",
                fix_suggestion=f"Pobierz matanyone2.pth do {models_path}\\matanyone2\\"
            ))

        # 2b. Inpaint model (optional): LaMa for the auto clean plate and the Cleanup mode
        lama_file = models_dir / "inpaint" / "big-lama.pt"
        items.append(CheckItem(
            name="Model: LaMa (big-lama.pt)", category="Models", status="ok" if lama_file.exists() else "warning",
            message=(f"LaMa obecny ({lama_file.stat().st_size / (1024**2):.0f} MB)" if lama_file.exists()
                     else f"Brak {lama_file} – dziury clean plate'u auto wypełni OpenCV (pewność 0)"),
            details={"path": str(lama_file)},
            fix_suggestion=None if lama_file.exists() else f"Pobierz big-lama.pt do {models_path}\\inpaint\\ (link w „skąd pobrać”)"))

        # 2c. LTX-2.5 Alpha-Gen (stage 6 „LTX”, optional): five files ~40 GB from the gated Lightricks repos (HF login)
        from engine.ltx_alpha import models_status as ltx_models_status
        ltx = ltx_models_status(models_dir)
        ltx_miss = [m for m in ltx if m["needed"] and not m["present"]]
        ltx_cond = not next(m for m in ltx if m["key"] == "text_encoder")["needed"]
        items.append(CheckItem(
            name="Model: LTX-2.5 Alpha-Gen", category="Models", status="ok" if not ltx_miss else "warning",
            message=(("Modele LTX-2.5 obecne (transformer int8, VAE, LoRA Alpha-Gen; pusty prompt zakodowany raz – "
                      "Gemma niepotrzebna)" if ltx_cond else
                      "Modele LTX-2.5 obecne (transformer int8, Gemma int8, VAE, LoRA Alpha-Gen)") if not ltx_miss else
                     f"Brak {len(ltx_miss)} z {sum(m['needed'] for m in ltx)} plików LTX-2.5 "
                     f"({sum(m['gb'] for m in ltx_miss):.1f} GB) – etap 7 „LTX-2.5” wyłączony"),
            details={"missing": [f"{m['folder']}/{m['file']}" for m in ltx_miss]},
            fix_suggestion=None if not ltx_miss else (
                "Wymagane teraz: NIE – tylko etap 7 „LTX-2.5 Alpha-Gen” (opcjonalny). Do czego: generatywna alfa włosów, dymu, "
                "rozmycia ruchu. Pobranie: ok. 40 GB (transformer 21,5 + Gemma 15,4 + VAE 1,8 + LoRA 1,3) z Hugging Face; "
                "repozytoria Lightricks są „gated”: najpierw „hf auth login” i „Agree and Access” na stronach modeli. "
                "VRAM: działa na 24 GB z przelewaniem do RAM-u (zalecane ≥ 64 GB RAM).")))

        # 2d. VideoMaMa (stage 4 „VideoMaMa”, optional, 2026-10-16): node pack ComfyUI-VideoMaMa + 7.1 GB of weights
        from engine.videomama import NODE_PACK, models_status as vmm_models_status
        vmm_miss = [m for m in vmm_models_status(models_dir) if not m["present"]]
        cn = Path(self.config.get("comfy_dir") or "") / "custom_nodes"
        vmm_node = bool(self.config.get("comfy_dir")) and (cn / NODE_PACK).is_dir()
        vmm_ok = not vmm_miss and vmm_node
        items.append(CheckItem(
            name="Model: VideoMaMa", category="Models", status="ok" if vmm_ok else "warning",
            message=("Węzeł ComfyUI-VideoMaMa i wagi obecne (UNet VideoMaMa + części SVD-xt)" if vmm_ok else
                     "Brak " + " i ".join(([f"węzła {NODE_PACK}"] if not vmm_node else []) +
                                          ([f"{len(vmm_miss)} plików wag ({sum(m['gb'] for m in vmm_miss):.1f} GB)"] if vmm_miss else []))
                     + " – etap 4 „VideoMaMa” wyłączony"),
            details={"missing": [f"{m['dir']}/{m['file']}" for m in vmm_miss], "node": vmm_node},
            fix_suggestion=None if vmm_ok else (
                "Wymagane teraz: NIE – tylko etap 4 „VideoMaMa” (opcjonalny). Do czego: kosmyki włosów i detal krawędzi "
                "z maski 1+2 (model na priorze Stable Video Diffusion). Pobranie: węzeł z GitHuba + 7,1 GB wag z Hugging Face "
                "(UNet VideoMaMa 6,1 GB, image encoder + VAE SVD-xt 1,5 GB) do models/videomama; po instalacji węzła restart "
                "ComfyUI. VRAM: ~8 GB przy kadrze 448 px, ~16 GB przy 1280 px (przy < 20 GB VRAM model przelewa się do RAM-u).")))

        # 2e. Cleanup 3D (docs/PLAN_CLEANUP_3D.md, optional): FLUX 3 Image Klein + LoRA Restore, COLMAP (CUDA)
        # Cleanup 3D switched off in this release (engine/features.py): no checks of the tools only C2–C6 use (VOID,
        # Qwen-Image-2.1 and the prompt assistant serve C1 Inpaint and stay)
        from engine.features import enabled as _feature_on
        c3_only = set() if _feature_on("cleanup3d", self.config) else {KLEIN_ITEM, KLEIN_LORA_ITEM, COMFY_VERSION_ITEM,
                                                                       COLMAP_ITEM, SEEDVR2_ITEM}
        items.extend(i for i in self.check_cleanup3d_models(models_dir) if i.name not in c3_only)
        items.append(self.check_qwen21_model(models_dir))
        items.append(self.check_prompt_assist_model(models_dir))
        items.extend(i for i in self.check_cleanup3d_tools() if i.name not in c3_only)

        # 3. CoTracker3 – weights in models_dir + the `cotracker` package in ComfyUI's Python (runs there)
        cotracker_file = models_dir / "cotracker" / "scaled_offline.pth"
        try:
            from engine.installer import comfy_python
            from engine.tracking import check_tracker_python
            pkg_ok, pkg_why = check_tracker_python(comfy_python(self.config))
        except Exception as e:
            pkg_ok, pkg_why = False, str(e)
        if cotracker_file.exists() and pkg_ok:
            items.append(CheckItem(
                name="Model: CoTracker3", category="Models", status="ok",
                message=f"CoTracker3 gotowy (scaled_offline.pth {cotracker_file.stat().st_size / (1024**2):.1f} MB, "
                        f"liczy w Pythonie ComfyUI na {pkg_why})",
                details={"path": str(cotracker_file), "device": pkg_why}))
        else:
            missing = ([] if cotracker_file.exists() else ["wagi scaled_offline.pth (97 MB)"]) + \
                      ([] if pkg_ok else ["pakiet cotracker w Pythonie ComfyUI (< 1 MB)"])
            items.append(CheckItem(
                name="Model: CoTracker3", category="Models", status="warning",
                message=f"CoTracker3: brak {', '.join(missing)}. Śledzenie punktów zastępczo przez OpenCV (LK) – mniej dokładne.",
                details={"path": str(cotracker_file), "package": pkg_why},
                fix_suggestion="Wymagane teraz: zalecane – długie śledzenie punktów (wyrównanie plate'ów, propagacja, strefy). "
                               "Kliknij „Zainstaluj” (pakiet do Pythona ComfyUI + wagi)."))

        # 4. ViTMatte – zone engine `vitmatte` / preset „włosy”; in MatteDesk's Python when it has torch, else in ComfyUI's
        try:
            from engine import vitmatte as vm
            if vm.VitMatteRunner.available()[0]:
                # the runtime is there – the weights must be too (2026-11-10: the HF cache was recreated on 2026-11-04,
                # the Doctor still said „gotowy” while stage 5's gate locked it)
                from engine.stage_models import hf_cached
                vm_ok, vm_dev, vm_where = True, "venv", "w Pythonie MatteDesk"
                vm_cached = [vm.DEFAULT_MODEL] if hf_cached(vm.DEFAULT_MODEL, self.config.get("models_dir")) else []
            else:
                from engine.installer import comfy_python
                vm_ok, vm_dev, vm_cached = vm.check_worker_python(comfy_python(self.config), [vm.DEFAULT_MODEL])
                vm_where = "w Pythonie ComfyUI"
        except Exception as e:
            vm_ok, vm_dev, vm_cached, vm_where = False, str(e), [], ""
        vm_need = ("Wymagane teraz: NIE – silnik ViTMatte w etapie 5 (wagi tylko niekomercyjne); bez niego etap 5 liczy BiRefNet_HR-matting / Lucida (wymagany). Pobiera: wagi "
                   "hustvl/vitmatte-base-composition-1k (~390 MB, cache Hugging Face); transformers tylko, jeśli ich brak "
                   "(ComfyUI ma je zwykle sam). Kliknij „Zainstaluj”. Opcjonalnie (wybór „Model” w palecie Strefa): "
                   "small / base – distinctions pobierają się same przy pierwszym użyciu (~100 / ~390 MB).")
        if vm_ok and vm.DEFAULT_MODEL in vm_cached:
            items.append(CheckItem(name="Model: ViTMatte", category="Models", status="ok",
                                   message=f"ViTMatte gotowy (liczy {vm_where} na {vm_dev})",
                                   details={"device": vm_dev, "cached": vm_cached}))
        elif vm_ok:
            items.append(CheckItem(name="Model: ViTMatte", category="Models", status="info",
                                   message=f"Opcjonalne: ViTMatte – brak wag w cache Hugging Face – pobiorą się przy 1. użyciu strefy "
                                           f"(~390 MB); liczy {vm_where} na {vm_dev}",
                                   details={"device": vm_dev, "cached": vm_cached}, fix_suggestion=vm_need))
        else:
            items.append(CheckItem(name="Model: ViTMatte", category="Models", status="info",
                                   message=f"Opcjonalne: ViTMatte niedostępny ({vm_dev}) – etap 5 liczy "
                                           "BiRefNet_HR-matting / Lucida",
                                   details={"reason": vm_dev}, fix_suggestion=vm_need))

        # 4b. BiRefNet_HR-matting / Lucida – the REQUIRED stage-5 edge engine (MIT weights; ViTMatte optional since 2026-10-23)
        items.append(self.check_birefnet())
        # 4c. Sapiens2-seg-0.4B – optional „Zaproponuj pas” (body parts -> band points, 2026-10-22)
        items.append(self.check_sapiens2())

        # 5. Plate alignment on the GPU (engine/align_gpu_worker.py in ComfyUI's Python; numpy + torch only)
        items.append(self.check_align_gpu())

        return items

    def check_cleanup3d_models(self, models_dir: Path) -> List[CheckItem]:
        """FLUX 3 Image Klein (3 files, gated repo), its LoRA Restore and COLMAP (CUDA) – all optional (Cleanup 3D)."""
        from engine.installer import COLMAP_BYTES, COLMAP_DIR, COLMAP_TAG, colmap_exe
        from engine.stage_models import (KLEIN_BF16_MIN_GB, KLEIN_VARIANT_LABEL, klein_lora_status, klein_status,
                                         klein_variant)
        items: List[CheckItem] = []
        kl = klein_status(models_dir)
        var = klein_variant(models_dir)
        miss = [m for m in kl if not m["present"]]
        listed = lambda ms: ", ".join(f"{m['folder']}/{m['file']} {m['gb']:.2f} GB" for m in ms)
        # the variant by the card (user 2026-11-10): int8 under KLEIN_BF16_MIN_GB of VRAM, bf16 above
        from engine.vram_fit import card_gb
        card = card_gb(self.comfy_vram_gb)
        why_var = (f"wariant {KLEIN_VARIANT_LABEL[var]} – karta {card:.0f} GB VRAM"
                   f" ({'< ' if var == 'int8' else '≥ '}{KLEIN_BF16_MIN_GB:.0f} GB)" if card else f"wariant {KLEIN_VARIANT_LABEL[var]}")
        tot = sum(m["gb"] for m in kl)
        items.append(CheckItem(
            name=KLEIN_ITEM, category="Models", status="ok" if not miss else "warning",
            message=(f"FLUX 3 Image Klein obecny, {why_var} ({listed(kl)})" if not miss else
                     f"Brak {len(miss)} z {len(kl)} plików FLUX 3 Image Klein, {why_var} ({sum(m['gb'] for m in miss):.1f} GB: "
                     f"{listed(miss)}) – silnik „FLUX 3 Image Klein + LoRA Restore” w Czyszczeniu wyłączony"),
            details={"missing": [f"{m['folder']}/{m['file']}" for m in miss], "variant": var},
            fix_suggestion=None if not miss else (
                KLEIN_NEED + " Do czego: dogenerowanie obszaru obiektu w kluczu karty (lepsze od LaMa na dużych "
                f"obszarach i fakturach). Pobranie: ok. {tot:.1f} GB ({listed(kl)}) z Hugging Face – wariant dobrany do "
                f"karty: int8 poniżej {KLEIN_BF16_MIN_GB:.0f} GB VRAM, bf16 powyżej; repozytorium "
                "black-forest-labs/flux-3-image-klein-comfy jest „gated”: najpierw „hf auth login” i akceptacja dostępu "
                "na stronie modelu. Wymaga ComfyUI ≥ 0.41.")))
        lo = klein_lora_status(models_dir)
        lmiss = [m for m in lo if not m["present"]]
        lf = f"{lo[0]['folder']}/{lo[0]['file']}, {lo[0]['bytes'] / 2 ** 20:.0f} MB"
        items.append(CheckItem(
            name=KLEIN_LORA_ITEM, category="Models", status="ok" if not lmiss else "warning",
            message=(f"LoRA Restore (FLUX 3 Image Klein) obecna ({lf})" if not lmiss else
                     f"Brak LoRA Restore (FLUX 3 Image Klein): {lf} – silnik „FLUX 3 Image Klein + LoRA Restore” "
                     "w Czyszczeniu wyłączony"),
            details={"missing": [f"{m['folder']}/{m['file']}" for m in lmiss]},
            fix_suggestion=None if not lmiss else (
                KLEIN_NEED + " Do czego: LoRA Restore na FLUX 3 Image Klein odtwarza fakturę plate'u zamiast ją "
                "wymyślać. Pobranie: 161 MB z Hugging Face (ausboss/FLUX-3-Image-Klein-Restore-LoRA, publiczne) do "
                "models/loras.")))
        # MiniMax-H3 (Cleanup C1, 2026-11-10): files only here (ComfyUI's nodes are checked when the panel asks)
        from engine import h3_fill
        hs = h3_fill.status(self.config, None)
        hf = hs["files"]
        items.append(CheckItem(
            name=H3_ITEM, category="Models", status="ok" if hs["ok"] else "info",
            message=(f"MiniMax-H3 obecny ({hf['unet']}, {hf['clip']}, {hf['vae']}"
                     f"{', LoRA turbo ' + hf['turbo'] if hf.get('turbo') else ', bez LoRA turbo – 30 kroków'}"
                     f"{', Fun ControlNet ' + hf['control'] if hf.get('control') else ''})" if hs["ok"]
                     else f"Opcjonalne: {hs['reason']}"),
            details={"files": hf},
            fix_suggestion=None if hs["ok"] else (
                "Wymagane teraz: NIE – tylko model „MiniMax-H3” w Czyszczeniu C1 (usuwanie / zastępowanie obiektu w całym "
                "klipie, spójne w czasie). Instalacja ręczna (ok. 55 GB, Doktor nie pobiera): pliki z Comfy-Org/MiniMax-H3 "
                "do models ComfyUI; potrzebny ComfyUI z węzłami MiniMax H3. VRAM: na 12 GB przelewa się do RAM-u.")))
        # Wan 2.2 Fun VACE (Inpaint C1, 2026-11-11): the 12 GB alternative to MiniMax-H3, installed by the Doctor
        from engine import vace_fill
        vs = vace_fill.status(self.config, None)
        vf = vs["files"]
        items.append(CheckItem(
            name=VACE_ITEM, category="Models", status="ok" if vs["ok"] else "info",
            message=(f"Wan 2.2 Fun VACE obecny ({vf['high']}, {vf['low']}, {vf['clip']}, {vf['vae']})" if vs["ok"]
                     else f"Opcjonalne: {vs['reason']}"),
            details={"files": vf},
            action=None if vs["ok"] else "install_vace",
            fix_suggestion=None if vs["ok"] else (
                "Wymagane teraz: NIE – tylko silnik „Wan 2.2 Fun VACE” w Inpaint (C1): usuń / zastąp obiekt w całym "
                "klipie z kotwicami Qwen-Image-2.1, alternatywa dla MiniMax-H3 na kartach 12 GB. „Zainstaluj”: węzeł "
                "ComfyUI-GGUF (github.com/city96) do custom_nodes ComfyUI (PyTorch przypięty), dwa eksperty "
                "Wan2.2-VACE-Fun-A14B Q4_K_S (2 × 10,6 GB, huggingface.co/QuantStack/Wan2.2-VACE-Fun-A14B-GGUF) do "
                "models/diffusion_models, LoRA Lightning 4-step high + low (2 × 1,23 GB, do „szybko – 4 kroki”) do models/loras "
                "oraz – jeśli ich nie ma – koder umt5_xxl_fp8 (6,7 GB) i wan_2.1_vae (0,25 GB). "
                "Potem restart ComfyUI. VRAM: na 12 GB ekspert 10,6 GB ładowany po kolei, możliwe lekkie przelewanie.")))
        lf = vace_fill.has_lightning(vf)
        items.append(CheckItem(
            name=VACE_LIGHTNING_ITEM, category="Models", status="ok" if lf else "info",
            message=("LoRA Lightning obecne – Wan 2.2 Fun VACE liczy „szybko” w 4 krokach" if lf
                     else "Opcjonalne: brak LoRA Lightning – Wan 2.2 Fun VACE liczy 20 kroków (wolniej)"),
            details={"files": {k: vf.get(k) for k in ("lora_high", "lora_low")}},
            action=None if lf else "install_vace_lightning",
            fix_suggestion=None if lf else (
                "Wymagane teraz: NIE – przyspiesza silnik „Wan 2.2 Fun VACE” w Inpaint (C1) ok. 5–10 razy: 4 kroki "
                "przy CFG 1 zamiast 20 przy CFG 3,5 (jak LoRA turbo MiniMax-H3), kosztem trochę szczegółu. "
                "„Zainstaluj”: LoRA lightx2v 4-step T2V v1.1 high + low (2 × 1,23 GB) z huggingface.co/Comfy-Org/"
                "Wan_2.2_ComfyUI_Repackaged do models/loras. Licencja Apache-2.0.")))
        # LTX-2.5 as an Inpaint (C1) engine (2026-11-13): stage 7's files + the in-/outpainting IC-LoRA
        from engine import ltx_fill
        xs = ltx_fill.status(self.config, None)
        xl = bool(xs["files"].get("lora"))
        items.append(CheckItem(
            name=LTX_INPAINT_ITEM, category="Models", status="ok" if xl else "info",
            message=(f"LoRA inpaintu obecna ({ltx_fill.LORA[1]})" + ("" if xs["ok"] else f" – {xs['reason']}") if xl
                     else "Opcjonalne: brak IC-LoRA inpaintu – silnik „LTX-2.5” w Inpaint (C1) niedostępny"),
            details={"files": xs["files"]},
            action=None if xl else "install_ltx_inpaint",
            fix_suggestion=None if xl else (
                "Wymagane teraz: NIE – tylko silnik „LTX-2.5” w Inpaint (C1): usuń / zastąp obiekt w całym klipie "
                "(IC-LoRA in-/outpainting Lightricks, zielony klucz pod maską) z kotwicami Qwen-Image-2.1. „Zainstaluj”: "
                f"{ltx_fill.LORA[1]} (1,31 GB) z huggingface.co/{ltx_fill.LORA[2]} do models/loras – repo „gated”: "
                "potrzebne logowanie Hugging Face („hf auth login”) i akceptacja licencji LTX-2.x na stronie repo. "
                "Reszta (transformer, Gemma, VAE ok. 40 GB) = pliki etapu 7 LTX-2.5. Licencja LTX-2.x Community "
                "(darmowo do 10 mln USD przychodu/rok). VRAM: transformer int8 21,5 GB – na 12 GB przelewa się do RAM-u.")))
        exe = colmap_exe(self.config)
        items.append(CheckItem(
            name=COLMAP_ITEM, category="Tools", status="ok" if exe else "info",
            message=(f"COLMAP obecny ({exe})" if exe else
                     f"Opcjonalne: COLMAP {COLMAP_TAG} (CUDA) nie jest zainstalowany (miejsce: {COLMAP_DIR})"),
            details={"path": str(exe) if exe else None},
            fix_suggestion=None if exe else (
                "Wymagane teraz: NIE – przyszłe liczenie kamery 3D w trybie Czyszczenie (COLMAP + CoTracker3). "
                f"Pobranie: oficjalna paczka colmap-x64-windows-cuda.zip {COLMAP_TAG} (~{COLMAP_BYTES / 2 ** 20:.0f} MB, "
                "BSD) z GitHuba, sprawdzana sumą SHA-256, rozpakowana do bin/colmap MatteDesk.")))
        return items

    def check_cleanup3d_tools(self) -> List[CheckItem]:
        """VOID and SeedVR2 (Cleanup 3D P12) – optional. Installed by the Doctor (user decision 2026-11-10): SeedVR2 into
        ComfyUI (node + ComfyUI's Python), VOID into bin/void (venv on ComfyUI's Python). Nothing installed = info,
        partly installed = warning; config.local.json keys still override the places."""
        from engine.cleanup3d.void_engine import TOOL_KEYS, required_files, tool_paths
        from engine.stage_models import _cleanup3d_tool
        need = {
            "void": ("Wymagane teraz: NIE – tylko „VOID – wypełnienie ujęcia” w Czyszczeniu 3D (usunięcie ruchomego "
                     "obiektu z całego ujęcia razem z jego śladami, klucze kart biorą z niego klatki). VOID działa "
                     "natywnie w ComfyUI (≥ 0.45, bez repozytorium i venv). „Zainstaluj”: z huggingface.co/Comfy-Org/"
                     "void-model void_pass1 (11,1 GB) → models/diffusion_models i cogvideox_vae (0,4 GB) → models/vae; "
                     "koder t5xxl_fp16 (9,8 GB) → models/text_encoders tylko, gdy żadnego t5xxl nie ma."),
            "seedvr2": ("Wymagane teraz: NIE – tylko „SeedVR2: podbij klucze karty” w Czyszczeniu 3D (podbicie łaty "
                        "kluczy jako jeden klip – spójna faktura przy gęstych kluczach). „Zainstaluj”: węzeł "
                        "numz/ComfyUI-SeedVR2_VideoUpscaler do custom_nodes ComfyUI, jego wymagania w Pythonie ComfyUI "
                        "(PyTorch przypięty, bez zmian) i wagi wg karty – 3B fp8 (3,4 GB) poniżej 16 GB VRAM, 7B sharp fp8 "
                        "(8,2 GB) od 16 GB – + VAE (0,5 GB) do models/SEEDVR2. Też „⤴ Upscale” wersji w Inpaint (C1)."),
        }
        items: List[CheckItem] = []
        for tool, name in (("void", VOID_ITEM), ("seedvr2", SEEDVR2_ITEM)):
            g = _cleanup3d_tool(self.config, tool)
            rows = required_files(self.config, tool)
            # the tool's own files (not its Python: for SeedVR2 that is ComfyUI's, always there)
            present = [p for _d, p in rows[1:] if p is not None and p.exists()]
            where = tool_paths(self.config, tool)["repo"]
            items.append(CheckItem(
                name=name, category="Models", status="ok" if g["ok"] else ("warning" if present else "info"),
                message=(f"{name}: zainstalowany ({where})" if g["ok"] else f"Opcjonalne: {g['reason']}"),
                details={"missing": g.get("missing") or [], "keys": list(TOOL_KEYS[tool])},
                links=[] if g["ok"] else list(LINKS[tool]),
                action=None if g["ok"] else f"install_{tool}",
                fix_suggestion=None if g["ok"] else need[tool]))
        return items

    def check_comfy_update(self) -> List[CheckItem]:
        """ComfyUI from git: how far behind origin, the update button and (after an update) the rollback button
        (engine/comfy_update.py, 2026-11-10). ComfyUI Desktop / no ComfyUI: nothing (it updates itself)."""
        if self.simulation.get("simulate_comfy_offline") or self.simulation.get("simulate_flavor"):
            return []
        try:
            from engine import comfy_update as CU
            i = CU.info(self.config)
        except Exception:  # noqa: BLE001 – the Doctor never fails on this
            return []
        if i.get("kind") != "git" or i.get("head") is None:
            return []
        items: List[CheckItem] = []
        behind, dirty = i.get("behind"), i.get("dirty") or []
        where = f"{i.get('dir')} ({i.get('branch')} @ {i.get('head')})"
        if dirty:
            st, msg, act = "warning", (f"ComfyUI z gita {where}: lokalne zmiany w {len(dirty)} plikach – aktualizacja "
                                       f"z MatteDeska zablokowana ({', '.join(dirty[:4])}{' …' if len(dirty) > 4 else ''})"), None
        elif behind is None:
            st, msg, act = "info", f"ComfyUI z gita {where}: nie udało się sprawdzić nowszej wersji (brak sieci?)", None
        elif behind == 0:
            st, msg, act = "ok", f"ComfyUI z gita {where}: aktualny", None
        else:
            st, msg, act = "info", f"ComfyUI z gita {where}: {behind} commitów za origin/{i.get('branch')}", "install_comfy_update"
        items.append(CheckItem(
            name=COMFY_UPDATE_ITEM, category="ComfyUI", status=st, message=msg, action=act,
            details={"behind": behind, "head": i.get("head"), "dirty": dirty},
            fix_suggestion=None if not act else (
                "Nowsze modele potrzebują nowszych węzłów ComfyUI (np. Qwen-Image-2.1: TextEncodeQwenImage21). "
                "„Zainstaluj” = git pull (tylko szybkie przewinięcie), zależności z przypiętym PyTorchem, potem restart "
                "ComfyUI. Poprzednia wersja jest zapamiętana – „Cofnij aktualizację ComfyUI” wraca do niej. Węzły "
                "dodatkowe (custom_nodes) mogą wymagać aktualizacji – Doktor pokaże, które się nie ładują.")))
        if i.get("can_rollback"):
            items.append(CheckItem(
                name=COMFY_ROLLBACK_ITEM, category="ComfyUI", status="info", action="install_comfy_rollback",
                message=f"Poprzednia wersja ComfyUI zapamiętana: {str(i.get('before'))[:10]} (aktualizacja {i.get('updated')})",
                fix_suggestion="Gdy po aktualizacji coś przestało działać: „Zainstaluj” = powrót do tej wersji + restart ComfyUI."))
        return items

    def check_qwen21_model(self, models_dir: Path) -> CheckItem:
        """Qwen-Image-2.1 (3 files of Comfy-Org/Qwen-Image-2.1) – optional engine of the key regeneration (Cleanup 3D)."""
        from engine.stage_models import qwen21_status
        rows = qwen21_status(models_dir)
        miss = [m for m in rows if not m["present"]]
        listed = ", ".join(f"{m['folder']}/{m['file']} {m['gb']:.2f} GB" for m in (miss or rows))
        return CheckItem(
            name=QWEN21_ITEM, category="Models", status="ok" if not miss else "warning",
            message=(f"Qwen-Image-2.1 obecny ({listed})" if not miss else
                     f"Brak {len(miss)} z {len(rows)} plików Qwen-Image-2.1 ({sum(m['gb'] for m in miss):.1f} GB: "
                     f"{listed}) – silnik „Qwen-Image-2.1” w Czyszczeniu wyłączony"),
            details={"missing": [f"{m['folder']}/{m['file']}" for m in miss]},
            fix_suggestion=None if not miss else (
                QWEN21_NEED + " Do czego: usunięcie obiektu z klucza karty – model edytuje cały wycinek wokół "
                "zaznaczenia według polecenia, wklejany jest tylko zaznaczony obszar. Pobranie: ok. 17,3 GB (model int8 "
                "7,3 GB, koder Qwen3-VL 8B int8 9,4 GB, VAE 0,68 GB) z Hugging Face (Comfy-Org/Qwen-Image-2.1). "
                "Licencja Qwen Research License – tylko niekomercyjnie. Wymaga ComfyUI z węzłem TextEncodeQwenImage21 "
                "(sprawdzone na 0.44.2)."))

    def check_prompt_assist_model(self, models_dir: Path) -> CheckItem:
        """Qwen3-VL-4B-Instruct (HF snapshot in models/LLM) – optional, C1 „✨ Popraw opis” (engine/prompt_assist.py)."""
        from engine import prompt_assist as PA
        rows = PA.status(models_dir)
        miss = [m for m in rows if not m["present"]]
        return CheckItem(
            name=PROMPT_ASSIST_ITEM, category="Models", status="ok" if not miss else "info",
            message=(f"{PA.LABEL} obecny ({PA.FOLDER}, {PA.DOWNLOAD_GB:.1f} GB)" if not miss else
                     f"Opcjonalne: brak {len(miss)} z {len(rows)} plików {PA.LABEL} ({sum(m['gb'] for m in miss):.1f} GB) – "
                     "„✨ Popraw opis” w Inpaint niedostępny"),
            details={"missing": [f"{m['folder']}/{m['file']}" for m in miss]},
            fix_suggestion=None if not miss else (
                PROMPT_ASSIST_NEED + f" Pobiera {PA.DOWNLOAD_GB:.1f} GB (Qwen/Qwen3-VL-4B-Instruct, bf16) do "
                f"models/{PA.FOLDER}; liczy w Pythonie ComfyUI (transformers z ComfyUI), model ładowany na czas pracy i "
                "wyładowywany po 3 min bezczynności. Licencja Apache-2.0."))

    def check_comfy_version(self, stats: Optional[Dict[str, Any]]) -> CheckItem:
        """ComfyUI >= 0.41 for FLUX 3 Image Klein (CLIPLoader type flux3, ReferenceMultiLatent); `stats` = /system_stats."""
        from engine.stage_models import KLEIN_MIN_COMFY
        need = ".".join(map(str, KLEIN_MIN_COMFY))
        raw = ((stats or {}).get("system") or {}).get("comfyui_version")
        ver = parse_version(raw)
        if ver is None:
            return CheckItem(name=COMFY_VERSION_ITEM, category="ComfyUI", status="info",
                             message=(f"Nie sprawdzono wersji ComfyUI (FLUX 3 Image Klein wymaga ≥ {need})"
                                      + (" – ComfyUI nie odpowiada" if stats is None else "")))
        if ver[:2] >= tuple(KLEIN_MIN_COMFY):
            return CheckItem(name=COMFY_VERSION_ITEM, category="ComfyUI", status="ok",
                             message=f"ComfyUI {raw} (FLUX 3 Image Klein wymaga ≥ {need})", details={"version": raw})
        return CheckItem(
            name=COMFY_VERSION_ITEM, category="ComfyUI", status="warning",
            message=f"ComfyUI {raw} jest za stary dla FLUX 3 Image Klein (wymaga ≥ {need}) – silnik „FLUX 3 Image Klein + "
                    "LoRA Restore” w Czyszczeniu nie zadziała",
            details={"version": raw, "required": need},
            fix_suggestion=(KLEIN_NEED + f" Zaktualizuj ComfyUI Desktop do wersji ≥ {need} (aktualizacja w aplikacji "
                            "ComfyUI Desktop; ComfyUI z gita: „git pull” w jego folderze), potem „Odśwież test”."))

    def check_birefnet(self) -> CheckItem:
        """BiRefNet_HR-matting / Lucida in ComfyUI's Python (engine/birefnet_worker.py): packages + weights in the HF cache.
        Required since 2026-10-23 (user decision): the stage-5 edge engine that stays when ViTMatte is removed."""
        from engine import birefnet
        need = ("Wymagane teraz: TAK – silnik krawędzi etapu 5 (zostaje, gdy ViTMatte jest usunięty). Do czego: "
                "alternatywa ViTMatte do pracy komercyjnej (wagi MIT; wagi ViTMatte są tylko niekomercyjne). Pobiera: "
                "BiRefNet_HR-matting 444 MB + Lucida 885 MB do cache Hugging Face; timm (~3 MB, czasem też kornia / einops) "
                "do Pythona ComfyUI, tylko jeśli ich brak – PyTorch ComfyUI zostaje bez zmian. Kliknij „Zainstaluj”.")
        try:
            from engine.installer import comfy_python
            from engine.vitmatte import check_worker_python
            ok, dev, cached = check_worker_python(comfy_python(self.config), list(birefnet.MODELS.values()),
                                                  worker=birefnet.WORKER)
        except Exception as e:
            ok, dev, cached = False, str(e), []
        names = {v: birefnet.LABELS[k] for k, v in birefnet.MODELS.items()}
        have = [names[m] for m in cached if m in names]
        if ok and len(have) == len(names):
            return CheckItem(name=BIREFNET_LABEL, category="Models", status="ok",
                             message=f"BiRefNet_HR-matting i Lucida gotowe (liczą w Pythonie ComfyUI na {dev})",
                             details={"device": dev, "cached": cached})
        if ok:
            miss = [n for n in names.values() if n not in have]
            return CheckItem(name=BIREFNET_LABEL, category="Models", status="warning",
                             message=f"Brak wag {', '.join(miss)}" + (f" (są: {', '.join(have)})" if have else ""),
                             details={"device": dev, "cached": cached}, fix_suggestion=need)
        return CheckItem(name=BIREFNET_LABEL, category="Models", status="warning",
                         message=f"BiRefNet_HR-matting / Lucida niedostępne w Pythonie ComfyUI ({dev}) – etap 5 bez nich "
                                 "liczy tylko ViTMatte",
                         details={"reason": dev}, fix_suggestion=need)

    def check_sapiens2(self) -> CheckItem:
        """Sapiens2-seg-0.4B in ComfyUI's Python (engine/sapiens2_worker.py): transformers with Sapiens2 + weights in the
        HF cache. Optional: the „Zaproponuj pas (Sapiens2)” button of the band (people only)."""
        from engine import sapiens2
        need = ("Wymagane teraz: NIE – opcjonalny przycisk „Zaproponuj pas (Sapiens2)” przy punktach pasa (etapy 3 / 4 / 5 / 7): "
                "rozpoznaje części ciała (włosy, skóra, ubranie) i ustawia pas sam – tylko dla ludzi. Pobiera: wagi "
                "Sapiens2-seg-0.4B 1,63 GB do cache Hugging Face; potrzebuje transformers ≥ 5 w Pythonie ComfyUI (Doktor go nie "
                "aktualizuje). Licencja Sapiens2 License (Meta): praca komercyjna dozwolona, zakaz nadzoru / biometrii / deepfake. "
                "Kliknij „Zainstaluj”.")
        try:
            from engine.installer import comfy_python
            from engine.vitmatte import check_worker_python
            ok, dev, cached = check_worker_python(comfy_python(self.config), [sapiens2.DEFAULT_MODEL], worker=sapiens2.WORKER)
        except Exception as e:
            ok, dev, cached = False, str(e), []
        if ok and sapiens2.DEFAULT_MODEL in cached:
            return CheckItem(name=SAPIENS2_LABEL, category="Models", status="ok",
                             message=f"Sapiens2-seg-0.4B gotowy (liczy w Pythonie ComfyUI na {dev})",
                             details={"device": dev, "cached": cached})
        if ok:
            return CheckItem(name=SAPIENS2_LABEL, category="Models", status="info",
                             message="Opcjonalne: brak wag Sapiens2-seg-0.4B", details={"device": dev}, fix_suggestion=need)
        return CheckItem(name=SAPIENS2_LABEL, category="Models", status="info",
                         message=f"Opcjonalne: Sapiens2 niedostępny w Pythonie ComfyUI ({dev})",
                         details={"reason": dev}, fix_suggestion=need)

    def check_align_gpu(self) -> CheckItem:
        """Alignment worker in ComfyUI's Python: optional speed-up, the CPU code works without it."""
        need = ("Wymagane teraz: NIE – przyspieszenie wyrównania plate'ów (backlit / clear plate), „Motion blur alfy” "
                "(rozmycie ruchu wersji, 2,5–4× szybciej na 4K) i odszumiania plate'u w kroku Plates oraz mniejsze obciążenie "
                "CPU; bez tego wszystkie liczą się na CPU. Nic nie pobiera: używa PyTorcha i CUDA, które ma ComfyUI. Ustawienia "
                "w config.local.json: `align_device` (wyrównanie), `mb_device` (motion blur) i `denoise_device` (odszumianie): "
                "auto (domyślnie) | gpu | cpu.")
        dev_cfg = str(self.config.get("align_device") or "auto")
        try:
            from engine import align_gpu
            from engine.installer import comfy_python
            ok, info = align_gpu.check_worker(comfy_python(self.config))
        except Exception as e:
            ok, info = False, str(e)
        details = {"align_device": dev_cfg, "mb_device": str(self.config.get("mb_device") or "auto"),
                   "denoise_device": str(self.config.get("denoise_device") or "auto"), "worker": info}
        if ok and info.startswith("cuda"):
            used = "nie (align_device = cpu)" if dev_cfg.lower() == "cpu" else "tak"
            return CheckItem(name="Wyrównanie na GPU", category="Models", status="ok",
                             message=f"Wyrównanie, motion blur alfy i odszumianie plate'u na GPU gotowe ({info}); wyrównanie na GPU: {used}",
                             details=details)
        why = f"Python ComfyUI bez CUDA ({info})" if ok else info
        return CheckItem(name="Wyrównanie na GPU", category="Models", status="warning",
                         message=f"Wyrównanie na GPU niedostępne: {why} – wyrównanie, motion blur i odszumianie liczą się na CPU",
                         details=details, fix_suggestion=need)

    def check_disk_space(self) -> CheckItem:
        work_drive = self.repo_dir.drive or "D:"
        try:
            total, used, free = shutil.disk_usage(work_drive)
            free_gb = free / (1024**3)
            status = "ok" if free_gb >= 50 else "warning"
            msg = f"Wolne miejsce na dysku {work_drive}: {free_gb:.1f} GB"
            if free_gb < 50:
                msg += " (Ostrzeżenie: zalecane co najmniej 50 GB wolnego miejsca)"
            return CheckItem(
                name="Disk Space",
                category="Storage",
                status=status,
                message=msg,
                details={"drive": work_drive, "free_gb": round(free_gb, 2)}
            )
        except Exception as e:
            return CheckItem(
                name="Disk Space",
                category="Storage",
                status="info",
                message=f"Nie można sprawdzić dysku {work_drive}: {e}"
            )

    def generate_install_markdown(self, report: DoctorReport) -> Optional[Path]:
        errors_or_warnings = [
            it for it in report.items if it.status in ("error", "warning") and (it.fix_suggestion or it.fix_command or it.links)
        ]
        if not errors_or_warnings:
            return None

        today = datetime.date.today().strftime("%Y-%m-%d")
        md_path = self.repo_dir / f"INSTALL_{today}.md"

        lines = [
            f"# Instrukcja instalacji / naprawy środowiska MatteDesk – {today}",
            "",
            "> **Uwaga:** Zgodnie z zasadami izolacji, polecenia ingerujące poza `.venv` narzędzia",
            "> (np. w folderze ComfyUI z konfiguracji lub pakiety systemowe) są generowane wyłącznie jako propozycja do wykonania przez użytkownika.",
            "",
            "## Wykryte braki i sugerowane kroki naprawcze:",
            ""
        ]

        for idx, it in enumerate(errors_or_warnings, 1):
            lines.append(f"### {idx}. {it.name} [{it.status.upper()}]")
            lines.append(f"- **Problem:** {it.message}")
            if it.fix_suggestion:
                lines.append(f"- **Rozwiązanie:** {it.fix_suggestion}")
            for ln in it.links:
                lines.append(f"- **Skąd pobrać:** {ln}")
            if it.fix_command:
                lines.append("```powershell")
                lines.append(it.fix_command)
                lines.append("```")
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

    def save_report(self, report: DoctorReport, output_path: Optional[Path] = None) -> Path:
        out = output_path or (self.repo_dir / "doctor_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return out


def run_doctor(config_path: Optional[Path] = None, simulation: Optional[Dict[str, Any]] = None) -> DoctorReport:
    doc = EnvironmentDoctor(config_path=config_path, simulation=simulation)
    report = doc.run_all_checks()
    doc.save_report(report)
    doc.generate_install_markdown(report)
    return report


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="MatteDesk Environment Doctor")
    parser.add_argument("--json", action="store_true", help="Output pure JSON")
    parser.add_argument("--simulate-desktop", action="store_true", help="Simulate ComfyUI Desktop flavor")
    parser.add_argument("--simulate-offline", action="store_true", help="Simulate ComfyUI service offline")
    parser.add_argument("--simulate-missing-model", action="store_true", help="Simulate missing model weights")
    parser.add_argument("--simulate-altered-sam3", action="store_true", help="Simulate altered SAM 3.1 node schemas")
    parser.add_argument("--write-local", action="store_true",
                        help="Zapisz wykryty port Comfy i folder ComfyUI do config.local.json (tylko ta maszyna)")
    args = parser.parse_args()

    sim = {}
    if args.simulate_desktop:
        sim["simulate_flavor"] = "desktop"
    if args.simulate_offline:
        sim["simulate_comfy_offline"] = True
    if args.simulate_missing_model:
        sim["simulate_missing_model"] = True
    if args.simulate_altered_sam3:
        sim["simulate_altered_sam3"] = True

    report = run_doctor(simulation=sim)

    if args.write_local:
        from engine.config import write_local_active as write_local
        doc = EnvironmentDoctor()
        vals = {}
        if report.comfy_url:
            vals["comfy_port"] = int(report.comfy_url.rsplit(":", 1)[1])
        if doc.config.get("comfy_dir"):
            vals["comfy_dir"] = doc.config["comfy_dir"]
        if doc.config.get("models_dir"):
            vals["models_dir"] = doc.config["models_dir"]
        print(f"config.local.json <- {vals}: {write_local(vals)}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return

    print("=" * 65)
    print(" MATTEDESK ENVIRONMENT DOCTOR - RAPORT SRODOWISKA")
    print("=" * 65)
    print(f"Data: {report.timestamp}")
    print(f"Wariant ComfyUI: {report.comfy_flavor}")
    print(f"Status ogolny: {'[OK] Wszystko gotowe' if report.all_ok else '[UWAGA] Wymagane sprawdzenie'}")
    print(f"Podsumowanie: OK={report.summary.get('ok', 0)}, Ostrzezenia={report.summary.get('warning', 0)}, Bledy={report.summary.get('error', 0)}")
    print("-" * 65)

    for it in report.items:
        prefix = {
            "ok": "[OK]     ",
            "warning": "[UWAGA]  ",
            "error": "[BLAD]   ",
            "info": "[INFO]   "
        }.get(it.status, "[?]      ")
        print(f"{prefix} {it.category} > {it.name}: {it.message}")
        if it.status in ("error", "warning") and it.fix_suggestion:
            print(f"         --> Sugestia: {it.fix_suggestion}")

    print("=" * 65)
    print(f"Raport zapisany w: doctor_report.json")


if __name__ == "__main__":
    main()
