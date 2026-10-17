# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step run by the Windows installer after the files are copied (bundled Python, own console window):

    python tools/installer_setup.py --comfy have [--dir X]       "I already have ComfyUI": detected by itself –
                                                                 Desktop, portable or from git; X = optional hint
                                                                 (the package folder, its ComfyUI folder or the parent)
    python tools/installer_setup.py --comfy portable --dir X     "No": download the official ComfyUI portable into X
    python tools/installer_setup.py --comfy later                nothing now (the Doctor asks on the first start)

Writes only config.local.json of this MatteDesk. Models stay where the user's ComfyUI keeps them; our nodes and
models are installed later by the Doctor (it opens by itself when something is missing).
Exit code 0 = done; 1 = failed (the window waits for Enter so the message can be read).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import comfy_portable  # noqa: E402
from engine.config import set_comfy_flavor, write_local  # noqa: E402


class Bar:
    """Console progress: MB, speed, time left (one line, rewritten)."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.first = None
        self.last = 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.time()
        if self.first is None:
            self.first = done          # resumed downloads: speed from what this run fetched
        if now - self.last < 0.5 and done < total:
            return
        self.last = now
        rate = (done - self.first) / max(now - self.t0, 1e-3)
        left = (total - done) / rate if rate > 0 and total else 0
        pct = 100.0 * done / total if total else 0
        sys.stdout.write(f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB  {pct:5.1f} %  {rate / 1e6:6.1f} MB/s  "
                         f"zostało ~{int(left // 60)} min {int(left % 60):02d} s   ")
        sys.stdout.flush()


def run(args: argparse.Namespace) -> int:
    if args.comfy == "later":
        print("ComfyUI: skonfigurujesz później w Doktorze MatteDesk.")
        return 0
    if args.comfy == "have":
        found = comfy_portable.detect(args.dir)
        if found["kind"] == "none":
            print("Nie znaleziono ComfyUI" + (f" w {args.dir}" if args.dir else "") + ". Wskaż go w Doktorze MatteDesk "
                  "(„Używany ComfyUI”) przy pierwszym starcie.")
            return 1
        port = int(found.get("port") or args.port)
        if found["kind"] == "desktop":
            set_comfy_flavor("desktop")
            if found.get("port"):
                write_local({"comfy_port": port})
            print(f"ComfyUI Desktop: {found['comfy_dir'] or '(działający)'} – foldery MatteDesk wykrywa sam, port {port}.")
        else:
            comfy_portable.configure(found["comfy_dir"], port=port, python=found.get("python") or "")
            what = "portable" if found["kind"] == "portable" else "z gita"
            print(f"ComfyUI {what}: {found['comfy_dir']} (port {port}"
                  + (f", Python {found['python']}" if found.get("python") else "") + "). Modele zostają w jego folderze models.")
        if found["others"]:
            print("Inne znalezione ComfyUI (przełączysz w Doktorze): " + ", ".join(found["others"][:5]))
        return 0
    if not args.dir:
        print("Brak folderu (--dir).")
        return 1
    # portable download
    info = comfy_portable.release_info()
    print(f"ComfyUI portable {info['tag']}: pobieranie {info['size'] / 1e9:.1f} GB, po rozpakowaniu "
          f"~{info['unpacked_gb']:.1f} GB w {args.dir}")
    print("Przerwane pobieranie wznowi się przy następnej próbie. Suma SHA-256 sprawdzana na końcu.")
    log: list = []
    res = comfy_portable.install(args.dir, info=info, progress=Bar(), log=log)
    print()
    for line in log:
        print(line)
    print(f"Gotowe: {res['comfy_dir']}. Nasze węzły i modele zainstaluje Doktor przy pierwszym starcie MatteDesk.")
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", choices=("have", "portable", "later"), required=True)
    ap.add_argument("--dir", default="")
    ap.add_argument("--port", type=int, default=comfy_portable.PORT)
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter after an error (tests)")
    args = ap.parse_args(argv)
    try:
        code = run(args)
    except Exception as e:   # shown to the user in the installer's console
        print(f"\nBŁĄD: {e}")
        code = 1
    if code and not args.no_pause:
        print("\nMatteDesk jest zainstalowany; ComfyUI ustawisz w Doktorze. Naciśnij Enter, aby zamknąć to okno.")
        try:
            input()
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
