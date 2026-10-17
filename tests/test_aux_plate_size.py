"""2026-10-17: SH010 backlit / clear plate (4096x2731) left in a 1440x2560 project (CLIP_A) – QA of every stage crashed
on `operands could not be broadcast (2560,1440) (2731,4096)`. A backlit / clear plate of another resolution is not used."""

import cv2
import numpy as np

from engine import pipeline as P


def _seq(d, w, h, n=2):
    d.mkdir(parents=True, exist_ok=True)
    for f in range(1, n + 1):
        cv2.imwrite(str(d / f"p_{f:04d}.png"), np.zeros((h, w, 3), np.uint8))
    return str(d / "p_####.png").replace("\\", "/")


def test_mismatch_detected(tmp_path):
    P._AUX_SIZE.clear()
    ok = _seq(tmp_path / "ok", 48, 64)
    bad = _seq(tmp_path / "bad", 96, 40)
    assert P.aux_plate_mismatch(ok, 48, 64) is None
    assert P.aux_plate_mismatch(bad, 48, 64) == "96x40 ≠ beauty 48x64"
    assert P.aux_plate_mismatch(None, 48, 64) is None
    assert P.aux_plate_mismatch(str(tmp_path / "none_####.png"), 48, 64) is None   # nothing to read = not judged


def test_pipeline_skips_wrong_aux_plates(tmp_path):
    P._AUX_SIZE.clear()
    beauty = _seq(tmp_path / "beauty", 48, 64)
    bad = _seq(tmp_path / "bad", 96, 40)
    good = _seq(tmp_path / "good", 48, 64)
    pipe = P.ObjectPipeline(tmp_path / "work", beauty, 48, 64, client=None, backlit_pattern=bad, cleanplate_pattern=bad)
    assert pipe.backlit is None and pipe.clean is None
    assert set(pipe.plates_skipped) == {"backlit", "cleanplate"}
    pipe2 = P.ObjectPipeline(tmp_path / "work2", beauty, 48, 64, client=None, backlit_pattern=good, cleanplate_pattern=good)
    assert pipe2.backlit is not None and pipe2.clean is not None and pipe2.plates_skipped == {}
