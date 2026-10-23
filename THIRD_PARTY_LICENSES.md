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
| **DepthCrafter** (model + nodes) | depth maps (optional) | Tencent: "only for academic, research and education purposes … no commercial or production purposes" (`Tencent/DepthCrafter`, `akatz-ai/ComfyUI-DepthCrafter-Nodes` LICENSE) |
| **MiniMax-Remover** (weights) | inpainting (planned, optional) | CC BY-NC 4.0 (HF `zibojia/minimax-remover`; the GitHub repo has no licence file) |

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
| comfyui-inpaint-nodes, ComfyUI-MiniMax-Remover, ComfyUI-VideoHelperSuite | nodes | GPL-3.0 |

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
