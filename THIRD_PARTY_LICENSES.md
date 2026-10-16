# Third-party licences

MatteDesk (this code) is licensed under **GPL-3.0-or-later** – see [LICENSE](LICENSE).

**MatteDesk does not contain the code or the weights of any of the projects below.** The environment Doctor downloads them
from their original sources only when the user asks, to the user's computer. By downloading them you accept their licences,
and you are responsible for complying with them – especially for commercial work. The Doctor shows the licence next to each item.

As of 2026-10-16. Licences can change – check the current licence at the source before commercial use.
"Checked" = the licence file was read from a local installation; "to confirm" = taken from the project's documentation, not verified.

## ⚠️ Non-commercial / research use only

| Component | Used in MatteDesk for | Licence | Status | Source |
|---|---|---|---|---|
| **MatAnyone2** (code + weights) | final edge pass, "matanyone" zone engine | S-Lab License 1.0 – **non-commercial only**; commercial use needs the authors' permission | checked | https://github.com/pq-yang/MatAnyone2 |
| **CoTracker3** (code + weights) | point tracking, roto propagation | CC BY-NC 4.0 – **non-commercial only** | checked | https://github.com/facebookresearch/co-tracker |
| **VideoMaMa** (code + UNet; SVD weights) | VideoMaMa stage (edge between MatAnyone and ViTMatte) | code CC BY-NC 4.0 – **non-commercial only**; Stable Video Diffusion weights: Stability AI Community License | from the test notes, to confirm | https://huggingface.co/SammyLim/VideoMaMa |
| **DepthCrafter** (model) | depth maps | Tencent – **academic, research and education only, no production use** | checked | https://github.com/Tencent/DepthCrafter |
| **LTX-2.5 + IC-LoRA Alpha-Gen** (optional) | stage "LTX" | LTX-2 Community License ("gated" repo, accepted on Hugging Face) | to confirm | https://huggingface.co/Lightricks |

MatteDesk keeps working without these components. The features that need them are unavailable, and the Doctor says so.

## ComfyUI models and nodes

| Component | Licence | Status | Source |
|---|---|---|---|
| ComfyUI | GPL-3.0 | checked | https://github.com/comfyanonymous/ComfyUI |
| SAM 3.1 (weights `sam3.1_multiplex_fp16`) | Meta SAM License | to confirm | https://huggingface.co/Comfy-Org/sam3.1 |
| ViTMatte (`hustvl/vitmatte-*`) | MIT (code); weights per the model card | to confirm | https://github.com/hustvl/ViTMatte |
| LaMa (`big-lama.pt`) | Apache-2.0 | to confirm | https://github.com/advimman/lama |
| MiniMax-Remover (weights) | per the model card | to confirm | https://github.com/zibojia/MiniMax-Remover |
| comfyui-inpaint-nodes | GPL-3.0 | checked | https://github.com/Acly/comfyui-inpaint-nodes |
| ComfyUI-MiniMax-Remover | GPL-3.0 | checked | https://github.com/1038lab/ComfyUI-MiniMax-Remover |
| ComfyUI-VideoHelperSuite | GPL-3.0 | checked | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite |
| ComfyUI-DepthCrafter-Nodes | Tencent (as DepthCrafter) | checked | https://github.com/akatz-ai/ComfyUI-DepthCrafter-Nodes |
| ComfyUI-VideoMaMa | CC BY-NC 4.0 (as VideoMaMa) | to confirm | https://github.com/okdalto/ComfyUI-VideoMaMa |
| comfy-matanyone2 | **no licence file** in the repo | checked | https://github.com/spiritform/comfy-matanyone2 |

## Python libraries and fonts

| Component | Licence |
|---|---|
| FastAPI, Pydantic, PyMatting | MIT |
| Uvicorn, httpx, websockets, NumPy, SciPy, OpenEXR, PyTorch | BSD |
| OpenCV, Transformers, huggingface_hub | Apache-2.0 |
| Fonts Inter, JetBrains Mono (Google Fonts) | SIL Open Font License 1.1 |

## ComfyUI nodes from this repository

`handoff/comfy/custom_nodes/ComfyUI-PlateMatte` and `ComfyUI-MatAnyone2-Keyframes` are MatteDesk code (GPL-3.0-or-later).
They import ComfyUI modules (GPL-3.0). `ComfyUI-MatAnyone2-Keyframes` imports the `matanyone2` library, which MatteDesk does
not contain – for this node to work, the user installs MatAnyone2 themselves (non-commercial licence, see above).
