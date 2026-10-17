# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MatteDesk as a desktop app (Windows shortcut of the installer):  pythonw.exe tools/launch_app.py

1. Port 8765: a MatteDesk server already answering -> reuse it (only open a window); another program -> message, stop.
2. Otherwise start the server (`python -m uvicorn app:app`, no console window, log in `logs/server.log`) and wait for it.
3. Open the UI in the system default browser (user decision 2026-10-17; `app_browser: edge` in config.local.json /
   the Doctor forces Edge): Chromium-based (Chrome, Edge, Brave, Vivaldi, Comet ...) -> its own window without an
   address bar (`--app`, own quiet profile in `%LOCALAPPDATA%/MatteDesk/browser-<name>`, no prompts, the process
   lives as long as the window); Firefox / others -> a tab + a tray icon "Stop MatteDesk"; nothing detected -> Edge.
4. Window closed -> a running computation asks first (Yes = stop it, No = reopen the window); then the server we started
   is stopped. A server we did not start is left alone. ComfyUI (own console) keeps running, as with start_v….bat.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, List, Optional

APP_DIR = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = int(os.environ.get("MATTEDESK_PORT") or 8765)   # other port: tests next to a running MatteDesk
URL = f"http://{HOST}:{PORT}"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _get_json(path: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}" + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def port_state(get_json: Callable[[str], Optional[dict]] = _get_json) -> str:
    """free | mattedesk | other"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((HOST, PORT)) != 0:
            return "free"
    v = get_json("/api/version")
    return "mattedesk" if isinstance(v, dict) and "version" in v else "other"


def server_python(app_dir: Path = APP_DIR) -> str:
    """The console-less pythonw.exe is fine for the launcher, but uvicorn gets python.exe (CREATE_NO_WINDOW hides it)."""
    for c in (app_dir / "python" / "python.exe", app_dir / ".venv" / "Scripts" / "python.exe"):
        if c.exists():
            return str(c)
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


def start_server(app_dir: Path = APP_DIR, popen: Callable[..., Any] = subprocess.Popen) -> Any:
    logs = app_dir / "logs"
    logs.mkdir(exist_ok=True)
    log = open(logs / "server.log", "ab")
    return popen([server_python(app_dir), "-m", "uvicorn", "app:app", "--host", HOST, "--port", str(PORT)],
                 cwd=str(app_dir), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                 creationflags=NO_WINDOW)


def wait_ready(proc: Any, timeout: float = 90.0, get_json: Callable[[str], Optional[dict]] = _get_json) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        if get_json("/api/version"):
            return True
        time.sleep(0.5)
    return False


EDGE_REL = Path("Microsoft/Edge/Application/msedge.exe")
CHROME_REL = Path("Google/Chrome/Application/chrome.exe")


def find_edge(env: Optional[dict] = None, exists: Callable[[Path], bool] = Path.exists) -> Optional[str]:
    """Edge (part of Windows 10/11), then Chrome; None when neither is found."""
    env = os.environ if env is None else env
    roots = [env.get("ProgramFiles(x86)"), env.get("ProgramFiles"), env.get("LOCALAPPDATA")]
    for r in (EDGE_REL, CHROME_REL):
        for root in roots:
            if root and exists(Path(root) / r):
                return str(Path(root) / r)
    for name in ("msedge", "chrome"):
        w = shutil.which(name)
        if w:
            return w
    return None


find_browser = find_edge   # old name


def default_browser() -> Optional[str]:
    """The exe of the system default browser (https handler of the current user), None when unknown."""
    if os.name != "nt":
        return None
    import shlex
    import winreg
    for proto in ("https", "http"):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 rf"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\{proto}\UserChoice")
            prog = winreg.QueryValueEx(key, "ProgId")[0]
            cmd = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, prog + r"\shell\open\command")
        except OSError:
            continue
        exe = cmd.split('"')[1] if cmd.startswith('"') else shlex.split(cmd, posix=False)[0]
        if exe and Path(exe).exists():
            return exe
    return None


def is_chromium(exe: str) -> bool:
    """Chrome, Edge, Brave, Vivaldi, Comet, Chromium ... (all support `--app`): their install has chrome.dll /
    *_elf.dll next to the exe or in the version folder. Firefox (xul.dll) and others -> False."""
    d = Path(exe).parent
    for pat in ("chrome.dll", "*_elf.dll", "*/chrome.dll", "*/*_elf.dll"):
        if any(d.glob(pat)):
            return True
    return Path(exe).stem.lower() in ("msedge", "chrome", "brave", "vivaldi", "chromium", "comet")


def app_browser_setting(app_dir: Path = APP_DIR) -> str:
    """config.local.json `app_browser`: default (system default browser, user decision 2026-10-17) | edge."""
    try:
        v = json.loads((app_dir / "config.local.json").read_text(encoding="utf-8")).get("app_browser")
    except (OSError, ValueError):
        v = None
    return "edge" if str(v or "").lower() == "edge" else "default"


def choose_browser(setting: str, default: Callable[[], Optional[str]] = default_browser,
                   edge: Callable[[], Optional[str]] = find_edge,
                   chromium: Callable[[str], bool] = is_chromium) -> tuple:
    """(exe | None, "app" | "tab"): the default browser in app mode when it is Chromium-based, as a tab otherwise
    (Firefox); Edge (app) when nothing is detected or the setting says edge; (None, "tab") = let Windows open it."""
    if setting != "edge":
        exe = default()
        if exe:
            return exe, ("app" if chromium(exe) else "tab")
    e = edge()
    return (e, "app") if e else (None, "tab")


# Quiet app window (user request 2026-10-17): the own profile is fresh, so Edge would show its welcome / sign-in /
# sync / "make Edge default" prompts and, after an unclean exit, "Restore pages?". None of that belongs in a tool window.
QUIET_FLAGS = ["--no-first-run", "--no-default-browser-check", "--disable-sync", "--hide-crash-restore-bubble",
               "--disable-session-crashed-bubble", "--disable-features=msImplicitSignin,msEdgeFirstRunExperience,"
               "msEdgeSidebarV2,msUndersideButton,msShowcaseFRE,EdgeCollections,msEdgeShopping,"
               "msWelcomePage,msHubApps,TranslateUI", "--no-pings", "--disable-background-networking"]


def browser_argv(browser: str, profile: Path) -> List[str]:
    return [browser, f"--app={URL}", f"--user-data-dir={profile}", *QUIET_FLAGS, "--window-size=1600,1000"]


def mark_clean_exit(profile: Path) -> None:
    """Mark the window profile as closed normally (no "Restore pages?" after a crash / forced stop) and switch off
    the profile prompts Edge keeps in its Preferences file. Missing / unreadable file: nothing to do."""
    pref = profile / "Default" / "Preferences"
    try:
        d = json.loads(pref.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    d.setdefault("profile", {}).update(exit_type="Normal", exited_cleanly=True)
    d.setdefault("browser", {})["has_seen_welcome_page"] = True
    d.setdefault("signin", {})["allowed"] = False
    d.setdefault("translate", {})["enabled"] = False
    try:
        pref.write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


def profile_dir(env: Optional[dict] = None, browser: str = "") -> Path:
    """Own window profile per browser (never the user's own profile: no tabs restored, no extensions, no sign-in)."""
    env = os.environ if env is None else env
    base = env.get("LOCALAPPDATA") or str(Path.home())
    stem = Path(browser).stem.lower() if browser else ""
    return Path(base) / "MatteDesk" / ("browser" if stem in ("", "msedge") else f"browser-{stem}")


def running_job(get_json: Callable[[str], Optional[dict]] = _get_json) -> Optional[str]:
    """Label of a computation in progress, else None."""
    d = get_json("/api/jobs") or {}
    for j in d.get("jobs") or []:
        if j.get("state") == "running":
            return str(j.get("label") or j.get("name") or "liczenie")
    return None


def message(text: str, title: str = "MatteDesk", flags: int = 0x40) -> int:
    """Windows message box (pythonw has no console). Returns the button id (6 = Yes, 7 = No, 1 = OK)."""
    if os.name == "nt":
        import ctypes
        # MB_TOPMOST | MB_SETFOREGROUND: the box must not hide behind other windows (pythonw has no window of its own)
        return int(ctypes.windll.user32.MessageBoxW(None, text, title, flags | 0x40000 | 0x10000))
    print(f"{title}: {text}")
    return 1


def stop(proc: Any) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    os.chdir(APP_DIR)
    state = port_state()
    if state == "other":
        message(f"Port {PORT} jest zajęty przez inny program – zamknij go i uruchom MatteDesk ponownie.", flags=0x10)
        return 1
    proc = None
    if state == "mattedesk" and replace_old_server():
        state = "free"
    if state == "free":
        proc = start_server()
        if not wait_ready(proc):
            stop(proc)
            message(f"Serwer MatteDesk nie wystartował. Log: {APP_DIR / 'logs' / 'server.log'}", flags=0x10)
            return 1
    browser, mode = choose_browser(app_browser_setting())
    if mode == "tab":
        open_tab(browser)
        if proc is not None:          # our server: a tray icon stops it (a browser tab cannot tell us it was closed)
            tray(proc, browser)
        return 0
    prof = profile_dir(browser=browser)
    prof.mkdir(parents=True, exist_ok=True)
    while True:
        mark_clean_exit(prof)
        subprocess.run(browser_argv(browser, prof))
        if proc is None:          # a server started elsewhere (other window, start_v….bat): leave it running
            return 0
        if ask_keep_running():
            continue
        stop(proc)
        return 0


def disk_version(app_dir: Path = APP_DIR) -> str:
    try:
        return (app_dir / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def replace_old_server(get_json: Callable[[str], Optional[dict]] = _get_json, ask: Callable[..., int] = None,
                       stopper: Callable[[], bool] = None) -> bool:
    """A MatteDesk server of another version (or older code) already holds the port – like start_v….bat: ask, then
    stop it so this version starts. True = stopped (start ours), False = keep using the running one."""
    v = get_json("/api/version") or {}
    mine = disk_version()
    if not v or (str(v.get("version")) == mine and not v.get("stale")):
        return False
    ask = ask or message
    job = running_job(get_json)
    text = (f"Działa inna wersja serwera MatteDesk ({v.get('version')}), ta to {mine}.\n"
            + (f"Trwa liczenie: {job} – zatrzymanie je przerwie.\n" if job else "")
            + f"Zatrzymać starą i uruchomić {mine}? (Nie = użyj działającej)")
    if ask(text, flags=0x24) != 6:
        return False
    return (stopper or stop_port_server)()


def stop_port_server() -> bool:
    """Stop the MatteDesk server on PORT (tools/server_port.ps1 -Stop – only `uvicorn app:app` processes)."""
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(APP_DIR / "tools" / "server_port.ps1"), "-Port", str(PORT), "-Stop"],
                       creationflags=NO_WINDOW, capture_output=True)
    if r.returncode != 0:
        message(f"Nie udało się zatrzymać starego serwera na porcie {PORT}.", flags=0x10)
        return False
    return True


def ask_keep_running() -> bool:
    """A computation in progress: ask before stopping. True = keep running (reopen the window)."""
    job = running_job()
    return bool(job) and message(f"Trwa liczenie: {job}.\nZatrzymać MatteDesk? (Nie = otwórz okno ponownie)",
                                 flags=0x24) == 7


def open_tab(browser: Optional[str], popen: Callable[..., Any] = subprocess.Popen) -> None:
    """A tab in the user's browser (Firefox and other non-Chromium browsers) – no extra window, no prompts."""
    if browser:
        popen([browser, URL], close_fds=True)
    else:
        import webbrowser
        webbrowser.open(URL)


def tray(proc: Any, browser: Optional[str]) -> None:
    """Tray icon "MatteDesk": Open (a new tab) / Stop MatteDesk. Without pystray: a message box with OK = stop."""
    try:
        import pystray
        from PIL import Image
    except ImportError:
        message("MatteDesk działa w przeglądarce.\nKliknij OK, aby zatrzymać MatteDesk.")
        stop(proc)
        return

    def on_open(icon, item):
        open_tab(browser)

    def on_stop(icon, item):
        if ask_keep_running():
            return
        stop(proc)
        icon.stop()

    try:
        img = Image.open(APP_DIR / "installer" / "MatteDesk.ico")
    except OSError:
        img = Image.new("RGB", (32, 32), (232, 163, 61))
    icon = pystray.Icon("MatteDesk", img, f"MatteDesk ({URL})", menu=pystray.Menu(
        pystray.MenuItem("Otwórz MatteDesk", on_open, default=True),
        pystray.MenuItem("Zatrzymaj MatteDesk", on_stop)))
    icon.run()


if __name__ == "__main__":
    sys.exit(main())
