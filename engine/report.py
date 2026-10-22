# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/report.py - "Zgłoś problem" (PLAN_2026-10-22 I10): a report package built on this machine, anonymised, shown
to the user in full and sent only after their consent to the report proxy (`report_endpoint` in config.json, a Cloudflare
Worker – infra/report_proxy/), which files it as a GitHub issue. The app never holds a GitHub token.

Package: what happened / steps / e-mail (optional, given by the user) + diagnostics: MatteDesk version, OS, Python, GPU
(nvidia-smi), ComfyUI version and VRAM (/system_stats), Doctor items that are not OK, the last jobs (state, error), the
tail of the server log + an optional screenshot of the window (PNG). Anonymisation: every path of this machine (config
values, the open shot's manifest / work folder / plates, the home folder, the app folder) and the names of the shot and
its objects, the user and the computer are replaced with <path> / <shot> / <obj1>… / <user> / <host> / <app> (custom
plate names <plate1>…); any other absolute path left in a text becomes <path>. Sizes are capped (text fields, log,
screenshot). `project` (user 2026-10-22): the objects as <objN> with their role and SAM 3.1 prompts AS TYPED (only
paths / e-mails / user / host removed – the user sees them in the preview), the plate keys.

Preview -> send: `preview()` keeps the built package under an id for a while; `send(id)` sends exactly that package,
so what the user saw is what leaves the computer."""
from __future__ import annotations

import base64
import collections
import json
import logging
import os
import platform
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "matte_desk/report@1"
MAX_TEXT = 5000                  # each user text field
MAX_LOG_LINES = 300
MAX_LOG_BYTES = 48 * 1024
MAX_PACKAGE_BYTES = 200 * 1024   # JSON without the screenshot
MAX_SCREENSHOT_BYTES = 3 * 1024 * 1024
MIN_WHAT = 10
KEEP_S = 30 * 60                 # previewed packages kept for sending

REPO_DIR = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- log buffer
class _Ring(logging.Handler):
    def __init__(self, n: int = MAX_LOG_LINES):
        super().__init__(level=logging.INFO)
        self.lines: collections.deque = collections.deque(maxlen=n)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass


_RING: Optional[_Ring] = None


def install_log_buffer() -> None:
    """Keep the last log lines of this server in memory (the console of start_v*.bat is not a file)."""
    global _RING
    if _RING is None:
        _RING = _Ring()
        logging.getLogger().addHandler(_RING)


def log_tail(app_dir: Path = REPO_DIR) -> List[str]:
    """Last lines of the in-memory log, else of logs/server.log (installed app: the launcher writes it)."""
    lines = list(_RING.lines) if _RING is not None and _RING.lines else []
    if not lines:
        p = Path(app_dir) / "logs" / "server.log"
        try:
            with open(p, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - MAX_LOG_BYTES))
                lines = fh.read().decode("utf-8", "replace").splitlines()[1:]
        except OSError:
            lines = []
    lines = lines[-MAX_LOG_LINES:]
    while lines and sum(len(x) + 1 for x in lines) > MAX_LOG_BYTES:
        lines.pop(0)
    return lines


# ----------------------------------------------------------------------------- anonymisation
# drive / UNC / home-like absolute paths that survived the exact replacements (stops at quotes, spaces, brackets)
_WIN_PATH = re.compile(r"(?<![\w<])(?:[A-Za-z]:[\\/]|\\\\[\w.$-]+\\)[^\s\"'<>|*?\]\[(){},;]*")
_NIX_PATH = re.compile(r"(?<![\w<./~-])/(?:home|Users|mnt|media|root|tmp|var|opt|srv|data)/[^\s\"'<>|*?\]\[(){},;]*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class Anonymizer:
    """Exact secrets (paths, names) -> placeholders, longest first; paths matched with either slash and any case."""

    def __init__(self, paths: Dict[str, str] = None, names: Dict[str, str] = None):
        self.pairs: List[Tuple[re.Pattern, str]] = []
        seen = set()
        items = [(p, tag, True) for p, tag in (paths or {}).items()] + [(n, tag, False) for n, tag in (names or {}).items()]
        for val, tag, is_path in sorted(items, key=lambda x: -len(x[0])):
            v = str(val or "").strip().rstrip("\\/")
            if len(v) < (3 if is_path else 2) or v.lower() in seen:
                continue
            seen.add(v.lower())
            if is_path:
                parts = [re.escape(x) for x in re.split(r"[\\/]+", v) if x != ""]
                lead = r"[\\/]+" if v[:1] in "\\/" else ""
                rx = re.compile(lead + r"[\\/]+".join(parts) + r"(?![^\\/\s\"'])", re.I) if parts else None
            else:   # a name: whole word only ("Woman_01", not inside "Woman_011")
                rx = re.compile(r"(?<![\w])" + re.escape(v) + r"(?![\w])", re.I)
            if rx is not None:
                self.pairs.append((rx, tag))

    def text(self, s: str) -> str:
        for rx, tag in self.pairs:
            s = rx.sub(tag, s)
        s = _WIN_PATH.sub("<path>", s)
        s = _NIX_PATH.sub("<path>", s)
        return _EMAIL.sub("<email>", s)

    def obj(self, x: Any) -> Any:
        if isinstance(x, str):
            return self.text(x)
        if isinstance(x, dict):
            return {self.text(str(k)): self.obj(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self.obj(v) for v in x]
        return x


def _strings(x: Any) -> Iterable[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def _looks_path(s: str) -> bool:
    return bool(re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|/|\.{1,2}[\\/]|~)", s)) or ("/" in s or "\\" in s) and len(s) > 3


def secrets_of(config: Optional[Dict[str, Any]] = None, state: Optional[Dict[str, Any]] = None,
               objects: Sequence[str] = (), app_dir: Path = REPO_DIR,
               env: Optional[Dict[str, str]] = None) -> Anonymizer:
    """Everything that identifies this machine, its user or the open project (see module doc)."""
    env = dict(os.environ) if env is None else env
    paths: Dict[str, str] = {}
    names: Dict[str, str] = {}
    for s in _strings(config or {}):
        if _looks_path(s):
            paths[s] = "<path>"
    st = state or {}
    m = st.get("manifest") or {}
    for s in _strings(m.get("plates") or {}):
        if _looks_path(s):
            paths[s] = "<path>"
            paths[str(Path(s).parent)] = "<path>"
    for k in ("manifest_path", "work_dir"):
        if st.get(k):
            paths[str(st[k])] = "<path>"
            paths[str(Path(str(st[k])).parent)] = "<path>"
    if m.get("work"):
        paths[str(m["work"])] = "<path>"
    home = env.get("USERPROFILE") or env.get("HOME") or str(Path.home())
    if home:
        paths[home] = "<home>"
    paths[str(Path(app_dir).resolve())] = "<app>"
    for k in ("LOCALAPPDATA", "APPDATA", "TEMP", "TMP"):
        if env.get(k):
            paths[env[k]] = "<path>"
    machine: Dict[str, str] = {}
    for k in ("USERNAME", "USER"):
        if env.get(k):
            machine[env[k]] = "<user>"
    host = env.get("COMPUTERNAME") or socket.gethostname()
    if host:
        machine[host] = "<host>"
    shot = m.get("shot") or (st.get("project") or {}).get("shot")
    if shot:
        names[str(shot)] = "<shot>"
    # objects numbered in their list order (user 2026-10-22): the log / jobs say WHICH object, project.objects describes it
    for i, o in enumerate(objects, 1):
        names[str(o)] = f"<obj{i}>"
    # custom plate names (a client / brand name may be in them); the standard keys say nothing and stay
    for i, k in enumerate([k for k in (m.get("plates") or {}) if k not in STANDARD_PLATES], 1):
        names[str(k)] = f"<plate{i}>"
    anon = Anonymizer(paths, {**names, **machine})
    # SAM 3.1 prompts are sent as typed (user decision 2026-10-17 / 2026-10-22): only paths, e-mails, user and host go
    anon.prompts = Anonymizer(paths, machine)
    return anon


# ----------------------------------------------------------------------------- project (objects, prompts, plates)
STANDARD_PLATES = ("beauty", "backlit", "cleanplate", "clean_plate", "depth", "bg")


def project_summary(objects: Sequence[Dict[str, Any]], key_prompts: Dict[str, Sequence[str]],
                    plates: Dict[str, Any], anon: Anonymizer) -> Dict[str, Any]:
    """What the shot holds, without its names: objects as <objN> (same numbers as in the anonymised texts) with their
    role and the SAM 3.1 prompts as typed (detect prompts + key prompts), plates as standard keys / <plateN>."""
    pa = getattr(anon, "prompts", anon)
    out = []
    for o in objects:
        det = o.get("detect") or {}
        prompts: List[str] = []
        for t in list(det.get("prompts") or []) + list(key_prompts.get(o.get("name"), []) or []):
            t = _clip(pa.text(str(t or "")), 200)
            if t and t not in prompts:
                prompts.append(t)
        out.append({k: v for k, v in {
            "id": anon.text(str(o.get("name", ""))),
            "prompts": prompts[:20],
            "detect": det.get("type"),
            "part_of": anon.text(str(o["part_of"])) if o.get("part_of") else None,
            "depth_order": o.get("depth_order"),
            "edge_mode": o.get("edge_mode"),
            "flags": o.get("flags") or None,
            "status": anon.text(str((o.get("tool") or {}).get("status") or "")) or None,
        }.items() if v not in (None, "", [])})
    return {"objects": out, "plates": [k if k in STANDARD_PLATES else anon.text(str(k)) for k in (plates or {})]}


# ----------------------------------------------------------------------------- package
def _clip(s: Any, n: int = MAX_TEXT) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + " …[obcięte]"


def _doctor_items(run_doctor: Optional[Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    if run_doctor is None:
        return {}
    try:
        rep = run_doctor() or {}
    except Exception as e:   # the report must work exactly when something is broken
        return {"error": f"{type(e).__name__}: {e}"}
    items = [{k: it.get(k) for k in ("name", "status", "message")} for it in rep.get("items", [])
             if it.get("status") not in ("ok", "info")]
    return {"summary": rep.get("summary"), "comfy_flavor": rep.get("comfy_flavor"), "items": items}


def _jobs(snapshot: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ("kind", "label", "state", "step", "error", "elapsed_s", "done", "total")
    return [{k: j.get(k) for k in keys if j.get(k) not in (None, "")} for j in list(snapshot)[-6:]]


def _screenshot(data_url: Optional[str]) -> Optional[bytes]:
    """data:image/png;base64,... -> PNG bytes (None when not given); ValueError when not a PNG or too large."""
    if not data_url:
        return None
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError):
        raise ValueError("Zrzut ekranu: nieprawidłowe dane obrazu")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Zrzut ekranu: oczekiwany plik PNG")
    if len(raw) > MAX_SCREENSHOT_BYTES:
        raise ValueError(f"Zrzut ekranu za duży ({len(raw) // 1024} KB > {MAX_SCREENSHOT_BYTES // 1024} KB)")
    return raw


def build_package(user: Dict[str, Any], *, version: str, anon: Anonymizer,
                  gpu: Optional[List[Dict[str, Any]]] = None, comfy: Optional[Dict[str, Any]] = None,
                  doctor: Optional[Dict[str, Any]] = None, jobs: Sequence[Dict[str, Any]] = (),
                  log: Sequence[str] = (), context: Optional[Dict[str, Any]] = None,
                  project: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The report as it will be sent (screenshot excluded – see preview). Raises ValueError on invalid user input."""
    what = _clip(user.get("what"))
    if len(what) < MIN_WHAT:
        raise ValueError(f"Opisz, co się stało (co najmniej {MIN_WHAT} znaków)")
    email = str(user.get("email") or "").strip()
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Adres e-mail wygląda na nieprawidłowy (pole jest opcjonalne – możesz je zostawić puste)")
    ctx = {k: _clip(v, 2000) for k, v in (context or {}).items() if k in ("source", "error_title", "error_text") and v}
    pkg = {
        "schema": SCHEMA,
        # the user's own words: paths / names in them are anonymised as well (an e-mail they typed in the text too)
        "user": {"what": anon.text(what), "steps": anon.text(_clip(user.get("steps"))), "email": email},
        "context": anon.obj(ctx),
        "app": {"version": version, "os": f"{platform.system()} {platform.release()} ({platform.version()})",
                "python": platform.python_version(), "machine": platform.machine()},
        "gpu": anon.obj(gpu),
        "comfy": anon.obj(comfy or {}),
        "doctor": anon.obj(doctor or {}),
        "jobs": anon.obj(_jobs(jobs)),
        "project": project or {},          # project_summary(): already anonymised (prompts as typed)
        "log_tail": anon.obj(list(log)),
    }
    while len(json.dumps(pkg, ensure_ascii=False).encode("utf-8")) > MAX_PACKAGE_BYTES and pkg["log_tail"]:
        pkg["log_tail"] = pkg["log_tail"][len(pkg["log_tail"]) // 4 + 1:]
    return pkg


def comfy_summary(client: Any) -> Dict[str, Any]:
    from engine.sysinfo import comfy_info
    out = comfy_info(client)
    info: Dict[str, Any] = {"alive": out.get("alive"), "devices": out.get("devices"), "ram": out.get("ram")}
    try:
        st = client.get_system_stats() if client is not None and out.get("alive") else None
        sysd = (st or {}).get("system") or {}
        info.update({k: sysd.get(k) for k in ("comfyui_version", "python_version", "pytorch_version", "os") if sysd.get(k)})
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------- preview / send
class ReportStore:
    """Previewed packages by id (in memory, KEEP_S); send() takes exactly what was previewed."""

    def __init__(self):
        self._lock = threading.Lock()
        self._items: Dict[str, Tuple[float, Dict[str, Any], Optional[bytes]]] = {}

    def put(self, pkg: Dict[str, Any], shot: Optional[bytes]) -> str:
        rid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            for k in [k for k, v in self._items.items() if now - v[0] > KEEP_S]:
                self._items.pop(k, None)
            self._items[rid] = (now, pkg, shot)
        return rid

    def pop(self, rid: str) -> Optional[Tuple[Dict[str, Any], Optional[bytes]]]:
        with self._lock:
            v = self._items.pop(rid, None)
        if v is None or time.time() - v[0] > KEEP_S:
            return None
        return v[1], v[2]

    def restore(self, rid: str, pkg: Dict[str, Any], shot: Optional[bytes]) -> None:
        with self._lock:
            self._items[rid] = (time.time(), pkg, shot)


STORE = ReportStore()


def preview_text(pkg: Dict[str, Any], shot: Optional[bytes]) -> str:
    """Everything that will be sent, as the user reads it (the screenshot only by size – shown as an image in the UI)."""
    body = json.dumps(pkg, indent=1, ensure_ascii=False)
    return body + (f"\n\n+ zrzut okna PNG, {len(shot) // 1024} KB" if shot else "")


def endpoint(config: Dict[str, Any]) -> str:
    return str(config.get("report_endpoint") or "").strip().rstrip("/")


def send(pkg: Dict[str, Any], shot: Optional[bytes], url: str, post: Optional[Callable[..., Any]] = None,
         timeout: float = 30.0) -> Dict[str, Any]:
    """POST {package, screenshot_png_b64} to <report_endpoint>/report -> {ok, number} (the proxy's answer).
    RuntimeError with a message for the user when the proxy is not set, unreachable or refuses."""
    if not url:
        raise RuntimeError("Zgłaszanie problemów nie jest jeszcze skonfigurowane (brak report_endpoint w config.json).")
    body = {"package": pkg, "screenshot_png_b64": base64.b64encode(shot).decode("ascii") if shot else None}
    if post is None:
        import httpx
        post = httpx.post
    try:
        r = post(f"{url}/report", json=body, timeout=timeout, headers={"User-Agent": f"MatteDesk/{pkg['app']['version']}"})
    except Exception as e:
        raise RuntimeError(f"Nie udało się połączyć z serwerem zgłoszeń ({type(e).__name__}). Sprawdź internet i spróbuj ponownie.")
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code == 429:
        raise RuntimeError(data.get("error") or "Za dużo zgłoszeń z tego adresu – spróbuj za godzinę.")
    if r.status_code != 200 or not data.get("ok"):
        raise RuntimeError(data.get("error") or f"Serwer zgłoszeń odpowiedział błędem HTTP {r.status_code}.")
    return {"ok": True, "number": data.get("number")}


def health(url: str, get: Optional[Callable[..., Any]] = None, timeout: float = 3.0) -> Tuple[bool, str]:
    """Doctor: is the report proxy reachable (GET <url>/health -> {ok: true})."""
    if not url:
        return False, "nie ustawiony"
    if get is None:
        import httpx
        get = httpx.get
    try:
        r = get(f"{url}/health", timeout=timeout)
        ok = r.status_code == 200 and bool((r.json() or {}).get("ok"))
        return ok, "osiągalny" if ok else f"odpowiada błędem HTTP {r.status_code}"
    except Exception as e:
        return False, f"nieosiągalny ({type(e).__name__})"
