# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/prompt_worker.py - the local prompt assistant (Qwen3-VL-4B-Instruct, transformers) in ComfyUI's Python
(user 2026-11-12): reads the user's description (any language) + the frame with the masked object outlined and writes
the English description by the rules of engine/prompt_rules.ASSIST_RULES (sent in the request).

Standalone on purpose: nothing from MatteDesk (run with `python -I`); the model is loaded on the first request and
stays until {"op": "release"} / the worker is closed by the GPU manager (idle 180 s).

Worker protocol (engine/worker_proto.py):
    <py> prompt_worker.py --check  -> "ok cuda <gpu>" / "ok cpu" / "ok fake" (exit 0) or the error (exit 1)
    <py> prompt_worker.py --serve  -> one JSON request per line:
        {"op": "assist", "model": folder of the HF snapshot, "system": text, "user": text, "image": png path or null,
         "max_new_tokens": int}  -> {"text": the raw answer, "load_s": s, "gen_s": s, "tokens": n}
        {"op": "release"};  {"op": "quit"}
MD_PROMPT_FAKE=1: no model - a fixed JSON answer built from the request (protocol tests in .venv).
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

FAKE = os.environ.get("MD_PROMPT_FAKE") == "1"
_M = {"path": None, "model": None, "proc": None}


def _load(path):
    if _M["path"] == path:
        return _M["model"], _M["proc"], 0.0
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    _M.update(path=None, model=None, proc=None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                            device_map=dev)
    model.eval()
    proc = AutoProcessor.from_pretrained(path)
    _M.update(path=path, model=model, proc=proc)
    return model, proc, time.time() - t0


def assist(req):
    if FAKE:
        user = req.get("user") or ""
        return {"text": json.dumps({"description": "FAKE: " + user[-60:], "clip": "", "notes": ["fake"]}),
                "load_s": 0.0, "gen_s": 0.0, "tokens": 0}
    import torch
    model, proc, load_s = _load(req["model"])
    content = []
    if req.get("image"):
        from PIL import Image
        content.append({"type": "image", "image": Image.open(req["image"]).convert("RGB")})
    content.append({"type": "text", "text": req.get("user") or ""})
    msgs = [{"role": "system", "content": [{"type": "text", "text": req.get("system") or ""}]},
            {"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(req.get("max_new_tokens") or 500), do_sample=False)
    new = out[0][inputs["input_ids"].shape[1]:]
    text = proc.decode(new, skip_special_tokens=True)
    return {"text": text, "load_s": round(load_s, 2), "gen_s": round(time.time() - t0, 2), "tokens": int(new.shape[0])}


def _handle(req, ask=None):
    op = req.get("op")
    if op == "release":
        _M.update(path=None, model=None, proc=None)
        if not FAKE:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {}
    if op == "assist":
        return assist(req)
    raise ValueError(f"unknown op {op}")


def _serve():
    spec = importlib.util.spec_from_file_location("worker_proto", Path(__file__).resolve().parent / "worker_proto.py")
    wp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wp)

    def make():
        device = "fake"
        if not FAKE:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def handle(req, ask):
            out = _handle(req, ask)
            resp = {"device": device, **out}
            if device == "cuda":
                import torch
                resp["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            return resp
        return handle
    return wp.serve(make)


def _check():
    if FAKE:
        print("ok fake")
        return 0
    try:
        import torch
        import transformers  # noqa: F401
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = torch.cuda.get_device_name(0) if dev == "cuda" else ""
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1
    print("ok", dev, name)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        return _check()
    if len(argv) >= 2 and argv[1] == "--serve":
        return _serve()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
