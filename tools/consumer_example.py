# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tools/consumer_example.py - Example of a dependent tool reading MatteDesk data (SPEC 16) WITHOUT importing
any MatteDesk code: only json + numpy + OpenEXR (+ OpenCV for PNG mattes).

    python tools/consumer_example.py <work>/matte_desk

Prints the objects, checks that every matte frame exists with the plate size, reads one matte and the 2D camera.
Exit code 0 = the contract holds for this shot.
"""

import json
import sys
from pathlib import Path

import numpy as np


def read_exr_plane(path, names=("R", "Y", "A")):
    import Imath
    import OpenEXR
    f = OpenEXR.InputFile(str(path))
    hdr = f.header()
    dw = hdr["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    chans = list(hdr["channels"])
    name = next((n for n in names if n in chans), chans[0])
    pt = hdr["channels"][name].type
    dt = np.float16 if str(pt) == "HALF" else np.float32
    a = np.frombuffer(f.channel(name, pt), dtype=dt).reshape(h, w).astype(np.float32)
    f.close()
    return a, chans


def main(work):
    work = Path(work)
    out = json.loads((work / "outputs.json").read_text(encoding="utf-8"))
    assert out["schema"].startswith("matte_desk/outputs@"), out.get("schema")
    first, last = out["frame_range"]
    pw, ph = out["plate_size"]
    print(f"shot {out.get('shot')}  frames {first}-{last}  plate {pw}x{ph}")
    for o in sorted(out["objects"], key=lambda x: x["depth_order"]):
        pat = o["matte"]
        missing = [f for f in range(first, last + 1) if not Path(pat.replace("####", f"{f:04d}")).exists()]
        assert not missing, f"{o['name']}: missing frames {missing[:5]}"
        p = Path(pat.replace("####", f"{first:04d}"))
        if p.suffix == ".exr":
            a, _ = read_exr_plane(p)
        else:
            import cv2
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        assert a.shape == (ph, pw), (o["name"], a.shape)
        assert 0.0 <= a.min() and a.max() <= 1.0
        print(f"  {o['depth_order']}: {o['name']:<12} {o['version']}  QA {o['qa_status']:<5}  coverage {float((a > 0.5).mean()):.4f}")
    if out.get("mattes_all"):
        _, chans = read_exr_plane(Path(out["mattes_all"].replace("####", f"{first:04d}")), ("union.alpha",))
        print("  multi-EXR channels:", sorted(chans))
    cam = out.get("camera2d", {}).get("json_path")
    if cam and Path(cam).exists():
        c = json.loads(Path(cam).read_text(encoding="utf-8"))
        H = np.array(c["frames"][-1]["homography"])
        assert H.shape == (3, 3)
        print(f"  camera2d: {c['motion_type']}, ref {c['ref_frame']}, {len(c['frames'])} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
