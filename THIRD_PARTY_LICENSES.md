# Third-party licences

MatteDesk (this code) is licensed under **GPL-3.0-or-later** – see [LICENSE](LICENSE).

**MatteDesk does not contain the code or the weights of any of the projects below.** The environment Doctor downloads them
from their original sources only when the user asks, to the user's computer. By downloading them you accept their licences,
and you are responsible for complying with them – especially for commercial work. The Doctor shows the licence next to each item.

As of 2026-10-21: every licence below was read at the source (the LICENSE file on GitHub or the Hugging Face model card).
Licences can change – check the current licence at the source before commercial use. The Doctor shows a short version.

## ⚠️ Non-commercial / research use only

| Component | Used in MatteDesk for | Licence (source) |
|---|---|---|
| **MatAnyone2** (code + weights) | stage 3, "matanyone" zone engine | S-Lab License 1.0 – non-commercial only; commercial use needs the authors' permission (`pq-yang/MatAnyone2` LICENSE.txt) |
| comfy-matanyone2 (node) | runs MatAnyone2 in ComfyUI | no licence file; README: MatAnyone2 under S-Lab License 1.0 (`spiritform/comfy-matanyone2`) |
| **CoTracker3** (code + weights) | stage 2, point tracking | CC BY-NC 4.0 (`facebookresearch/co-tracker` LICENSE.md, HF `facebook/cotracker3`) |
| **ViTMatte** (weights `hustvl/vitmatte-*`) | stage 5 zones, model "ViTMatte" | code MIT, model card Apache-2.0, but the composition-1k weights are trained on the **Adobe Deep Image Matting** dataset, whose licence agreement allows models trained on it to be used and distributed for non-commercial purposes only (FBA Matting README says the same of its Adobe-trained models); distinctions-646 weights: academic dataset, licence not published |
| **VideoMaMa** (code; UNet weights) | stage 4 | code CC BY-NC 4.0 (`cvlab-kaist/VideoMaMa` README / License.md); UNet + `dino_projection_mlp` checkpoints: Stability AI Community License (HF `SammyLim/VideoMaMa`); the base `stable-video-diffusion-img2vid-xt`: Stability AI Community License (free commercial use only under USD 1M annual revenue, with registration). The code is NC, so the stage is NC |
| ComfyUI-VideoMaMa (node) | runs VideoMaMa in ComfyUI | no licence file; README: "follows the original VideoMaMa license" (`okdalto/ComfyUI-VideoMaMa`) |
| **Qwen-Image-2.1** (`Comfy-Org/Qwen-Image-2.1`, optional) | Cleanup mode: key regeneration engine "Qwen-Image-2.1" | Qwen Research License (the Comfy-Org repo links `Qwen/Qwen-Image-2.1` LICENSE, read 2026-11-10): the Materials for non-commercial purposes only, commercial use needs a separate licence from Qwen; the licence does not say whether generated images count as Materials – **to verify with Qwen** before using the output in paid work |
| **DepthCrafter** (model + nodes) | depth maps (optional) | Tencent: "only for academic, research and education purposes … no commercial or production purposes" (`Tencent/DepthCrafter`, `akatz-ai/ComfyUI-DepthCrafter-Nodes` LICENSE) |

MatteDesk keeps working without these components. The features that need them are unavailable, and the Doctor says so.

## ✔ Commercial use allowed (check the conditions)

| Component | Used in MatteDesk for | Licence (source) |
|---|---|---|
| ComfyUI | runs the models | GPL-3.0 |
| **SAM 3.1** (weights `sam3.1_multiplex_fp16`) | stage 1 | SAM License (Meta, 2025-11-19): royalty-free use incl. commercial; no military / ITAR / Trade-Controls uses (HF `Comfy-Org/sam3.1`, `facebookresearch/sam3`) |
| **BiRefNet_HR-matting** (`ZhengPeng7/BiRefNet_HR-matting`) | stage 5 zones, model "BiRefNet_HR-matting" | MIT (code + weights); the training sets of this checkpoint are not listed (BiRefNet's matting sets – P3M-10k, AM-2k, AIM-500, Distinctions-646, HIM2K… – are academic) |
| **Lucida** (`egeorcun/lucida`) | stage 5 zones, model "Lucida" | MIT (code + weights); the model card says part of the training data (DIS5K, P3M-10k, COD10K) is research-only |
| **Sapiens2-seg-0.4B** (`facebook/sapiens2-seg-0.4b`, optional) | „Propose band” of the band points (stages 3 / 4 / 5 / 7) | Sapiens2 License (Meta, `facebookresearch/sapiens2` LICENSE.md): non-exclusive, worldwide, royalty-free, no non-commercial clause; prohibited: surveillance, biometric processing, identifying people, deepfakes / impersonation, ITAR uses; weights redistributed only with the agreement (MatteDesk does not ship them) |
| **LTX-2.5 + IC-LoRA Alpha-Gen** (optional) | stage 7 | LTX-2.x Community License (2026-08-11, gated on HF): free incl. commercial for entities under USD 10M annual revenue; above that a paid commercial licence (`Lightricks/LTX-2` LICENSE-2_x) |
| LaMa (`big-lama.pt`) | inpainting (optional) | Apache-2.0 (`advimman/lama`) |
| comfyui-inpaint-nodes, ComfyUI-VideoHelperSuite | nodes | GPL-3.0 |

## ⚠️ Commercial use only under a revenue limit

| Component | Used in MatteDesk for | Licence (source) |
|---|---|---|
| **FLUX 3 Image Klein** (`black-forest-labs/flux-3-image-klein-comfy`, gated, optional) + **LoRA Restore** (`ausboss/FLUX-3-Image-Klein-Restore-LoRA`, optional) | Cleanup mode: key fill engine "FLUX 3 Image Klein + LoRA Restore" | FLUX Kommunity License v1.0 (Black Forest Labs; LICENSE.md of the LoRA repo, read 2026-11-09 – the LoRA is a Derivative under the same licence, no extra terms): free for non-commercial purposes; Outputs may be used commercially only by a "Qualifying User" – gross annualized revenue together with affiliates below USD 5,000,000; above that a licence from Black Forest Labs (bfl.ai/licensing). Prohibited: military, surveillance, biometric processing, unlawful content. **Not** "non-commercial only", but check your company's revenue before commercial work |

## ⚠️ Commercial use after registration (Cleanup 3D, own Python environments)

| Component | Used in MatteDesk for | Licence (source) |
|---|---|---|
| **VOID** (`Comfy-Org/void-model` repack of `netflix/void-model` `void_pass1` on CogVideoX-Fun-V1.5-5b-InP, optional; natively in ComfyUI since 2026-11-11, the old `Netflix/void-model` repo + venv still supported) | Cleanup 3D: „VOID – wypełnienie ujęcia” (the moving object removed from the whole shot, key fills) | code of the original repo: Apache-2.0; model: the CogVideoX-5B transformer is under the **CogVideoX License** (free for research, commercial use after registering for the free basic commercial licence, up to 1 million visits / month); check the Comfy-Org / netflix model cards before paid work. Downloaded by the Doctor on the user's click, not shipped |

## Tools

| Component | Used in MatteDesk for | Licence (source) |
|---|---|---|
| **COLMAP** 4.2.1 (`colmap-x64-windows-cuda.zip`, optional) | camera solve of Cleanup 3D (future) | new BSD (`colmap/colmap` COPYING.txt, read 2026-11-09); the dependencies bundled in the Windows package (e.g. the CUDA libraries) are licensed separately |
| **SeedVR2** (`numz/ComfyUI-SeedVR2_VideoUpscaler` CLI + weights `seedvr2_ema_7b_sharp_fp8_e4m3fn`, optional) | Cleanup 3D: „SeedVR2: podbij klucze karty” (key patches upscaled as one clip) | code: Apache-2.0 (LICENSE of the local copy; README: "Reverted License from MIT to Apache 2.0 to match ByteDance Seed project", read 2026-11-10); the weights come from ByteDance-Seed SeedVR2 – check the model card. Not installed or shipped by MatteDesk (own venv, `config.local.json`) |
| **Wan 2.2 Fun VACE** A14B (GGUF Q4_K_S by `QuantStack/Wan2.2-VACE-Fun-A14B-GGUF`, optional) + **ComfyUI-GGUF** (`city96`) | Inpaint C1: engine „Wan 2.2 Fun VACE” (remove / replace over the whole clip with the Qwen-Image-2.1 anchors) | weights: Apache-2.0 (Hugging Face model card `license: apache-2.0`, read 2026-11-11; Wan 2.2 is Apache-2.0); ComfyUI-GGUF: Apache-2.0. Downloaded by the Doctor on the user's click, not shipped |
| **Qwen3-VL-4B-Instruct** (`Qwen/Qwen3-VL-4B-Instruct`, optional) | Inpaint C1: „✨ Popraw opis” (local prompt assistant) | Apache-2.0 (Hugging Face model card `license: apache-2.0`, read through the HF API 2026-11-12). Downloaded by the Doctor on the user's click, not shipped |
| **Wan 2.2 Lightning LoRAs** (lightx2v 4-step T2V v1.1 high / low, `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`, optional) | Inpaint C1: Wan 2.2 Fun VACE „szybko – 4 kroki” | Apache-2.0 (lightx2v). Downloaded by the Doctor on the user's click, not shipped |
| **LTX-2.3 IC-LoRA In-/Outpainting** (`Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting`, optional) | Inpaint C1: engine „LTX-2.5” | LTX-2.x Community License (gated on Hugging Face; free under USD 10 M revenue / year). Downloaded by the Doctor on the user's click with the user's HF login, not shipped |

## Python libraries and fonts

| Component | Licence |
|---|---|
| FastAPI, Pydantic, PyMatting | MIT |
| Uvicorn, httpx, websockets, NumPy, SciPy, OpenEXR, PyTorch, psutil | BSD |
| pystray | LGPL-3.0 |
| OpenCV, Transformers, huggingface_hub | Apache-2.0 |
| Fonts Inter, JetBrains Mono (Google Fonts) | SIL Open Font License 1.1 |
| html2canvas 1.4.1 (loaded from cdnjs only when a report includes a window screenshot) | MIT |

## ComfyUI nodes from this repository

`handoff/comfy/custom_nodes/ComfyUI-PlateMatte` and `ComfyUI-MatAnyone2-Keyframes` are MatteDesk code (GPL-3.0-or-later).
They import ComfyUI modules (GPL-3.0). `ComfyUI-MatAnyone2-Keyframes` imports the `matanyone2` library, which MatteDesk does
not contain – for this node to work, the user installs MatAnyone2 themselves (non-commercial licence, see above).
