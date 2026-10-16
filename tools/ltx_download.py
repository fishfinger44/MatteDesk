# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tools/ltx_download.py - download the LTX-2.5 Alpha-Gen models (~40 GB) with the user's HF login, resumable.

Same code as the Doctor's „Zainstaluj LTX” (engine.installer.install_ltx). Prints one progress line per ~2 %.
Usage: python tools/ltx_download.py [key ...]   (keys: unet lora text_encoder video_vae audio_vae; default all)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import load_config            # noqa: E402
from engine import installer                     # noqa: E402


def main() -> int:
    cfg = load_config()
    token = installer.hf_token()
    if not token:
        print("BRAK logowania HF (hf auth login)", flush=True)
        return 2
    last = {"pct": -1, "t": time.time()}

    def prog(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        if pct >= last["pct"] + 2:
            print(f"  {pct:3d}%  {done / 2**30:6.2f}/{total / 2**30:.2f} GB", flush=True)
            last.update(pct=pct, t=time.time())

    def dl(url: str, dst: Path) -> None:
        last.update(pct=-1)
        print(f"POBIERAM {dst.name}", flush=True)
        installer._download_hf(url, dst, token, prog)
        print(f"OK {dst.name} {dst.stat().st_size / 2**30:.2f} GB", flush=True)

    res = installer.install_ltx(cfg, download=dl, keys=sys.argv[1:] or None)
    for line in res["log"]:
        print(line, flush=True)
    print("GOTOWE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:      # one line the monitor can see
        print(f"BLAD {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
