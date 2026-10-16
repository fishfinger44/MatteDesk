# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z27 2b – QA reads the raw EXR (log luminance, one OpenEXR call) and only the object's band of scanlines.

Alpha measures (holes, islands) are identical; the auxiliary ones (camera shift, flow, clear plate) may differ in
the numbers, but on the synthetic shot the conclusions – frame colours and flags – are the same as with the former
full-frame display gray.
"""

import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import Imath  # noqa: E402
import numpy as np  # noqa: E402
import OpenEXR  # noqa: E402

from engine.exr_io import log_luma, read_half_rgb, read_log_luma_rows  # noqa: E402
from engine.pipeline import PlateCropLoader, qa_gray_rows  # noqa: E402
from engine.qa import gray_band, run_qa  # noqa: E402

W, H, N = 800, 600, 8


def _write(path, bgr):
    hdr = OpenEXR.Header(bgr.shape[1], bgr.shape[0])
    pt = Imath.PixelType(Imath.PixelType.HALF)
    hdr["channels"] = {c: Imath.Channel(pt) for c in "BGR"}
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
    o = OpenEXR.OutputFile(str(path), hdr)
    o.writePixels({c: np.ascontiguousarray(bgr[..., i].astype(np.float16)).tobytes() for i, c in enumerate("BGR")})
    o.close()


def _shot(root, hole_frame=5):
    """Textured background panning 3 px / frame, an ellipse riding with it; one frame with an interior hole."""
    rng = np.random.default_rng(2)
    big = cv2.GaussianBlur(rng.random((H, W + 64, 3)).astype(np.float32), (0, 0), 2.5) * 1.6 + 0.05
    alphas = {}
    for i in range(N):
        f = 1001 + i
        dx = 3 * i
        img = big[:, dx:dx + W].copy()
        a = np.zeros((H, W), np.float32)
        cv2.ellipse(a, (360 - dx, 290), (120, 45), 0, 0, 360, 1.0, -1)
        if f == 1001 + hole_frame:
            cv2.circle(a, (360 - dx, 290), 10, 0.0, -1)
        img = img * (1 - a[..., None]) + np.array([0.08, 0.5, 0.9], np.float32) * a[..., None]
        _write(root / f"pl_{f:04d}.exr", img)
        alphas[f] = a
    return str(root / "pl_####.exr"), alphas


def test_gray_band():
    assert gray_band(500, 1800, 2731) == (372, 1928)             # +128 px each side, >= 25 % (683) already
    a, b = gray_band(1300, 1340, 2731)                           # tiny object: grown to 25 % around the middle
    assert b - a == 683 and a <= 1172 and b >= 1468
    assert gray_band(0, 100, 400) == (0, 228)
    assert gray_band(390, 398, 400) == (262, 400)                # kept inside the frame
    assert gray_band(0, 2731, 2731) == (0, 2731)


def test_log_luma_rows_are_a_band_of_the_full_frame(tmp_path):
    pat, _ = _shot(tmp_path)
    p = tmp_path / "pl_1001.exr"
    full = read_log_luma_rows(p)
    assert full.dtype == np.float16 and full.shape == (H, W)
    assert np.array_equal(read_log_luma_rows(p, 100, 333), full[100:333])
    assert np.array_equal(full, log_luma(read_half_rgb(p)))
    assert full.min() < full.max()                                # fixed scale, not clipped flat
    rows = qa_gray_rows(pat, [1001])
    assert rows is not None and np.array_equal(rows(1001, 5, 50), full[5:50])
    assert qa_gray_rows(str(tmp_path / "x_####.png"), [1]) is None


def test_qa_conclusions_same_as_display_gray(tmp_path):
    pat, alphas = _shot(tmp_path)
    frames = sorted(alphas)
    ld = PlateCropLoader(pat)
    old_gray = lambda f: cv2.cvtColor((np.clip(ld.read_reference(f), 0, 1) * 255).astype(np.uint8),
                                      cv2.COLOR_BGR2GRAY).astype(np.float32)
    calls = []
    rows = qa_gray_rows(pat, frames)

    def spy(f, y0, y1):
        calls.append((f, y0, y1))
        return rows(f, y0, y1)

    old = run_qa(frames, lambda f: alphas[f], old_gray, flow_check=True, load_clean_gray=old_gray)
    new = run_qa(frames, lambda f: alphas[f], None, flow_check=True, load_gray_rows=spy, load_clean_gray_rows=rows,
                 gray_source="exr_log_luma")
    # only the band of rows was read, every frame once
    y0, y1 = new.plate_gray["rows"]
    assert new.plate_gray["source"] == "exr_log_luma" and 0 < y0 < y1 < H
    assert sorted(c[0] for c in calls) == frames and all(c[1:] == (y0, y1) for c in calls)
    # alpha measures identical, conclusions identical
    assert new.holes == old.holes and new.islands == old.islands and new.holes[1006] == 1
    assert new.colours == old.colours, (old.colours, new.colours)
    # flags identical except the flow check, which is an auxiliary measure (its numbers may differ): at the hole
    # frame the flow IoU sits at the 0.95 threshold by construction (display / whole frame 0.956, log / band 0.931)
    noflow = lambda rep: {f: sorted(r for _, r in v if not r.startswith("zgodność po przepływie") and not r.startswith("migotanie"))
                          for f, v in rep.flags.items()}
    assert noflow(new) == noflow(old)
    flick = lambda rep: sorted(f for f, v in rep.flags.items() if any(r.startswith("migotanie") for _, r in v))
    assert flick(new) == flick(old)
    assert abs(new.edge["flow_iou_min"] - old.edge["flow_iou_min"]) < 0.05
    assert new.gate_ok == old.gate_ok and new.bbox == old.bbox
    # camera shift found on the band too: flicker after compensation as small as with the display gray
    assert abs(new.flicker_median - old.flicker_median) < 2e-3, (old.flicker_median, new.flicker_median)
    json.dumps(new.to_dict())


def test_pipeline_qa_uses_full_display_gray(tmp_path):
    """User decision 2026-10-13 (quality first): the pipeline QA measures camera shift / flow / clear plate on the
    display gray of the whole frame, as before Z27 2b (the log-luma band compensated the camera worse on SH010)."""
    from test_e5b_acceptance import _pipe, _rigid
    shot = _rigid(tmp_path)
    pipe, frames = _pipe(tmp_path, shot, use_mplate=False)
    pipe.run_core("obj", frames)
    _full, _rep, run_dir = pipe.finalize("obj", frames, edge_mode="core")
    qa = json.loads((run_dir / "qa.json").read_text())
    assert qa["plate_gray"] == {"source": "display_full", "rows": [0, shot.h]}
