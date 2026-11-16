# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„✨ Popraw opis” of Inpaint (C1) – a local prompt assistant (user 2026-11-12): Qwen3-VL-4B-Instruct (Apache-2.0) in a
worker in ComfyUI's Python (engine/prompt_worker.py) writes the English description by engine/prompt_rules.assist_rules
from what the user typed (any language) and the frame with the masked object outlined. REMOVE: the description is the
background behind the object – the object is never named (the instruction + `guard` on the answer). Every call runs under the GPU
manager's lease (`GPU.use`, LEASED_CALLS): another work's models are unloaded first, the idle worker is closed after
`gpu_idle_release_s` (180 s) – the model leaves the card on its own.

Optional: Doctor „Model: Qwen3-VL-4B (asystent opisu)” → Zainstaluj (8,9 GB, models/LLM/Qwen3-VL-4B-Instruct).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from engine.worker_proto import WorkerDied, WorkerProcess, WorkerTimeout

WORKER = Path(__file__).resolve().with_name("prompt_worker.py")
LABEL = "Qwen3-VL-4B"
REPO = "Qwen/Qwen3-VL-4B-Instruct"
FOLDER = "LLM/Qwen3-VL-4B-Instruct"
PAGE = "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
LICENSE = "Apache-2.0"
IMAGE_SIDE = 1024            # the frame the assistant sees: long side (the mask outlined in red)
# key -> (folder, file, repo, subfolder, bytes): the HF snapshot (sizes from the HF API 2026-11-12)
FILES: Dict[str, tuple] = {k: (FOLDER, f, REPO, "", b) for k, (f, b) in {
    "w1": ("model-00001-of-00002.safetensors", 4967229296),
    "w2": ("model-00002-of-00002.safetensors", 3908490048),
    "index": ("model.safetensors.index.json", 64742),
    "config": ("config.json", 1505),
    "gen": ("generation_config.json", 269),
    "chat": ("chat_template.json", 5502),
    "pre": ("preprocessor_config.json", 390),
    "vpre": ("video_preprocessor_config.json", 385),
    "tok": ("tokenizer.json", 7032403),
    "tokc": ("tokenizer_config.json", 10868),
    "vocab": ("vocab.json", 2776833),
    "merges": ("merges.txt", 1671839),
}.items()}
DOWNLOAD_GB = sum(v[4] for v in FILES.values()) / 1e9


class PromptAssistError(RuntimeError):
    pass


def status(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """The snapshot's files (stage_models._status rows: key, folder, file, url, bytes, gb, present)."""
    from engine.stage_models import _status
    return _status(FILES, models_dir)


def model_dir(cfg: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """The model folder when every file is on disk, else None."""
    from engine.stage_models import _md
    if cfg is None:
        from engine.config import load_config
        cfg = load_config()
    md = _md(cfg)
    if not md or not all(r["present"] for r in status(md)):
        return None
    return Path(md) / FOLDER


def worker_python() -> Optional[Path]:
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


def available(cfg: Optional[Dict[str, Any]] = None) -> bool:
    if os.environ.get("MD_PROMPT_FAKE") == "1":
        return True
    py = worker_python()
    return model_dir(cfg) is not None and py is not None and Path(py).exists()


class PromptWorker(WorkerProcess):
    worker = WORKER
    label = "worker asystenta opisu"
    doctor_item = "Wyrównanie na GPU"          # same requirement (ComfyUI's Python with torch + CUDA)
    tmp_prefix = "md_prompt_"
    error_cls = PromptAssistError

    def __init__(self, python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None,
                 timeout_s: float = 600.0):
        super().__init__(python if python is not None else worker_python(), env, timeout_s)
        self.tmp = self.tmp_dir()

    def call(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        from engine.gpu_manager import GPU
        with GPU.use(self, f"{LABEL} (asystent opisu)", self.kill, cancel_check):
            try:
                resp = self.request(req, cancel_check, restarts=1)
            except WorkerDied as e:
                raise PromptAssistError(f"worker {LABEL} zakończył się dwa razy: {e}") from None
            except WorkerTimeout as e:
                raise PromptAssistError(str(e)) from None
            if not resp.get("ok"):
                raise PromptAssistError(f"worker {LABEL}: {resp.get('error')}")
            return resp


_LOCK = threading.Lock()
_WORKER: Dict[str, Optional[PromptWorker]] = {"w": None}


def _worker() -> PromptWorker:
    if _WORKER["w"] is None:
        _WORKER["w"] = PromptWorker()
    return _WORKER["w"]


def outlined(frame_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """uint8 BGR at IMAGE_SIDE (long side, never up) with the mask's outline in red (3 px)."""
    import cv2
    img = np.asarray(frame_bgr)
    if img.dtype != np.uint8:
        img = np.clip(np.rint(np.asarray(img, np.float32) * 255.0), 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    s = min(1.0, IMAGE_SIDE / float(max(h, w)))
    out = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA) \
        if s < 1.0 else img.copy()
    if mask is not None:
        m = (cv2.resize(np.asarray(mask, np.float32), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA) > 0.5)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 0, 255), 3)
    return np.ascontiguousarray(out)


def user_text(text: str, mode: str, transition: bool, image: bool, names: Sequence[str] = ()) -> str:
    """The request of one job; the user's text stays the last part. `names` (the tool's labels of the cleaned thing)
    are NOT written into the request (review 2026-11-13: "ignore the label kobieta" puts the very word in front of a
    small model, which then tends to repeat it) – they only feed `guard`, which checks the answer."""
    remove = mode != "replace"
    what = "REMOVE (the background continues)" if remove else "REPLACE (a new look inside the mask)"
    parts = [f"Mode: {what}."]
    if image:
        parts.append("The image is a frame of the shot; the masked object is outlined in red.")
    if remove:
        parts.append("Describe only the background behind the outlined object. Do not name or describe the object.")
    elif text.strip():
        parts.append("The new look inside the mask is the user's text below – build the description around it and keep "
                     "every idea of it; do not describe how the object looks now.")
    parts.append("TRANSITION: also write the clip text." if transition else "No transition: leave \"clip\" empty.")
    empty = ("(empty – propose a description of the background from the image)" if remove
             else "(empty – propose a description from the image)")
    parts.append("The user's text (any language):\n" + (text.strip() or empty))
    return "\n".join(parts)


def guard(out: Dict[str, Any], mode: str, names: Sequence[str]) -> Dict[str, Any]:
    """REMOVE only: the sentences of the assistant's answer (description, clip) that carry a label of the removed
    object are dropped, with a Polish note saying so. Dropping, not re-asking: the worker decodes greedily
    (do_sample=False), so the same request gives the same answer and a changed one is a second 10–40 s GPU call with
    no guarantee; a dropped sentence is deterministic and costs nothing, and an empty description is a valid REMOVE
    (the program's own remove instruction alone). REPLACE and the user's own text are never touched."""
    from engine import prompt_rules as PR
    out["guard"] = {"dropped": [], "words": []}
    if mode == "replace" or not PR.name_terms(names):
        return out
    for k in ("description", "clip"):
        r = PR.strip_named(out.get(k) or "", names)
        if r["dropped"]:
            out[k] = r["text"]
            out["guard"]["dropped"] += r["dropped"]
            out["guard"]["words"] += [w for w in r["words"] if w not in out["guard"]["words"]]
    if out["guard"]["dropped"]:
        w = ", ".join(out["guard"]["words"])
        out["notes"] = list(out.get("notes") or []) + [
            f"Pominięto {len(out['guard']['dropped'])} zdanie(-a) z nazwą usuwanego obiektu („{w}”) – nazwa w opisie "
            "może przywrócić obiekt."
            + ("" if out["description"] else " Propozycja jest pusta: pusty opis = samo polecenie usunięcia.")]
    return out


def parse(raw: str) -> Dict[str, Any]:
    """The model's JSON answer -> {description, clip, notes}; text around / code fences tolerated."""
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    try:
        d = json.loads(m.group(0) if m else t)
    except ValueError:
        raise PromptAssistError(f"{LABEL} nie odpowiedział w formacie JSON: {raw[:200]}") from None
    desc = str(d.get("description") or "").strip()
    if not desc:
        raise PromptAssistError(f"{LABEL} nie zwrócił opisu: {raw[:200]}")
    notes = d.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    return {"description": desc, "clip": str(d.get("clip") or "").strip(),
            "notes": [str(n) for n in notes if str(n).strip()][:6]}


def assist(text: str, mode: str = "remove", transition: bool = False, frame_bgr: Optional[np.ndarray] = None,
           mask: Optional[np.ndarray] = None, cfg: Optional[Dict[str, Any]] = None,
           worker: Optional[PromptWorker] = None, cancel_check: Optional[Callable[[], None]] = None,
           names: Sequence[str] = ()) -> Dict[str, Any]:
    """The proposal: {description, clip, notes, lint, guard, seconds: {load, gen}, tokens}. Nothing is saved.
    `names` = the labels of the cleaned thing (REMOVE: told to the model as the thing to leave out + `guard`)."""
    import cv2
    from engine import prompt_rules as PR
    md = model_dir(cfg)
    if md is None and os.environ.get("MD_PROMPT_FAKE") != "1":
        raise PromptAssistError(f"Brak modelu {LABEL} – Doktor → „Zainstaluj” przy „Model: {LABEL} (asystent opisu)”")
    with _LOCK:
        wk = worker or _worker()
        n = len(os.listdir(wk.tmp))
        ip = wk.tmp / f"frame_{os.getpid()}_{n}.png"
        try:
            if frame_bgr is not None:
                cv2.imwrite(str(ip), outlined(frame_bgr, mask))
            resp = wk.call({"op": "assist", "model": str(md or ""), "system": PR.assist_rules(mode),
                            "user": user_text(text, mode, transition, frame_bgr is not None, names),
                            "image": str(ip) if frame_bgr is not None else None, "max_new_tokens": 500}, cancel_check)
        finally:
            try:
                os.remove(ip)
            except OSError:
                pass
    out = parse(resp.get("text") or "")
    if not transition:
        out["clip"] = ""
    guard(out, mode, names)
    if mode == "replace":   # the user's own words the proposal dropped (2026-11-13: "psychedelic" came back as porcelain)
        miss = PR.missing_terms(text, out.get("description"))
        out["missing"] = miss
        if miss:
            out["notes"] = list(out.get("notes") or []) + [
                f"⚠ Propozycja pomija Twoje słowa: {', '.join(miss)} – popraw ją albo wstaw swój opis."]
    out["lint"] = PR.lint(out["description"], mode, transition)
    out["seconds"] = {"load": resp.get("load_s"), "gen": resp.get("gen_s")}
    out["tokens"] = resp.get("tokens")
    out["gpu_mem_peak_mb"] = resp.get("gpu_mem_peak_mb")
    return out


def close() -> None:
    w = _WORKER["w"]
    if w is not None:
        w.close()
        from engine.gpu_manager import GPU
        GPU.forget(w)
        _WORKER["w"] = None
