# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bump the MatteDesk version: VERSION, start_v<ver>.bat (git mv), CHANGELOG.md.

Scheme (user decision 2026-10-14): 0.D = feature release, 0.DS = small change in it.
    python tools/bump_version.py "small fix"            # 0.1 -> 0.11 -> 0.12 ... -> 0.19
    python tools/bump_version.py "big change" --major   # 0.1x -> 0.2 (0.9x -> 1.0)
    python tools/bump_version.py "what changed" --set 0.5
"""
import argparse
import datetime
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def next_version(old: str, major: bool) -> str:
    m = re.fullmatch(r"(\d+)\.(\d)(\d?)", old)
    if not m:
        raise SystemExit(f"VERSION '{old}' is not N.D or N.DS")
    n, d, s = int(m.group(1)), int(m.group(2)), m.group(3)
    if major:
        return f"{n + 1}.0" if d == 9 else f"{n}.{d + 1}"
    if not s:
        return f"{n}.{d}1"
    if s == "9":
        raise SystemExit(f"{old}: no small step left after .9 – use --major")
    return f"{n}.{d}{int(s) + 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--major", action="store_true", help="feature release: 0.1x -> 0.2")
    ap.add_argument("--set", dest="new")
    a = ap.parse_args()
    old = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    new = a.new or next_version(old, a.major)
    (ROOT / "VERSION").write_text(new + "\n", encoding="utf-8")
    src, dst = ROOT / f"start_v{old}.bat", ROOT / f"start_v{new}.bat"
    if src.exists():
        r = subprocess.run(["git", "mv", str(src), str(dst)], cwd=ROOT)
        if r.returncode:
            src.rename(dst)
    cl = ROOT / "CHANGELOG.md"
    text = cl.read_text(encoding="utf-8")
    entry = f"## {new} – {datetime.date.today().isoformat()}\n- {a.message}\n\n"
    i = text.find("\n## ")
    cl.write_text(text[: i + 1] + entry + text[i + 1:] if i >= 0 else text + "\n" + entry, encoding="utf-8")
    print(f"{old} -> {new}: VERSION, {dst.name}, CHANGELOG.md")


if __name__ == "__main__":
    main()
