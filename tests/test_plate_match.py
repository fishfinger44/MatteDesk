# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate CONTENT check (2026-11-06, engine/plate_match.py): a backlit / clear plate of the right size from another shot or
with a shifted frame used to pass. Same scene at another exposure without the object = ok, shifted = shifted (with the
shift), another image = other (and not used by the stages – like a wrong resolution), missing frames counted; the
result is cached in the work folder, never in the shot manifest. Thresholds come from SH010 (real) – here synthetic."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import plate_match as pm  # noqa: E402

W, H = 640, 400


def scene(seed: int) -> np.ndarray:
    """A textured scene (blobs + lines) as linear float RGB."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 0.2, np.float32)
    for _ in range(60):
        c = tuple(float(x) for x in rng.random(3) * 0.8 + 0.1)
        cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(5, 40)), c, -1)
    for _ in range(25):
        c = tuple(float(x) for x in rng.random(3))
        cv2.line(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), (int(rng.integers(0, W)), int(rng.integers(0, H))), c, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def write_seq(d, name, frames, make):
    d.mkdir(parents=True, exist_ok=True)
    for f in frames:
        cv2.imwrite(str(d / f"{name}_{f:04d}.exr"), make(f).astype(np.float32))
    return str(d / f"{name}_####.exr").replace("\\", "/")


@pytest.fixture()
def seqs(tmp_path):
    base = scene(1)
    obj = lambda f: (cv2.circle(base.copy(), (200 + 4 * f, 200), 50, (0.9, 0.6, 0.5), -1))   # noqa: E731 – the object moves
    beauty = write_seq(tmp_path / "b", "beauty", range(1, 13), obj)
    clean = write_seq(tmp_path / "c", "clean", range(1, 13), lambda f: base * 0.25)              # no object, 2 stops darker
    shift = np.float32([[1, 0, 9], [0, 1, -6]])
    shifted = write_seq(tmp_path / "s", "shifted", range(1, 13), lambda f: cv2.warpAffine(base * 2.0, shift, (W, H), borderMode=cv2.BORDER_REFLECT))
    other = write_seq(tmp_path / "o", "other", range(1, 13), lambda f: scene(7))
    partial = write_seq(tmp_path / "p", "partial", range(1, 9), lambda f: base)
    return {"beauty": beauty, "clean": clean, "shifted": shifted, "other": other, "partial": partial, "tmp": tmp_path}


def test_same_scene_shifted_other_and_missing(seqs):
    b = seqs["beauty"]
    ok = pm.check(b, seqs["clean"])
    assert ok["state"] == "ok" and ok["score"] > 0.4 and ok["shift_px"] <= pm.SHIFT_OK_PX and ok["missing"] == 0
    sh = pm.check(b, seqs["shifted"])
    assert sh["state"] == "shifted" and abs(sh["dx"] - 9) < 1.5 and abs(sh["dy"] + 6) < 1.5
    ot = pm.check(b, seqs["other"])
    assert ot["state"] == "other" and ot["score"] < pm.SCORE_OTHER
    pa = pm.check(b, seqs["partial"])
    assert pa["state"] == "ok" and pa["missing"] == 4 and pa["missing_first"] == [9, 10, 11, 12]
    assert "brak 4 klatek" in pm.describe(pa) and pm.describe(ot).startswith("✗") and pm.describe(sh).startswith("⚠")
    assert pm.check(b, str(seqs["tmp"] / "none" / "x_####.exr"))["state"] == "unknown"


def test_cache_in_the_work_folder_and_the_stages_skip_another_shot(seqs):
    work = seqs["tmp"] / "work"
    work.mkdir()
    b = seqs["beauty"]
    assert pm.cached(work, b, seqs["other"]) is None and pm.content_mismatch(work, b, seqs["other"]) is None
    r = pm.run(work, b, seqs["other"])
    assert (work / pm.CACHE_NAME).exists() and pm.cached(work, b, seqs["other"]) == r
    assert "inne ujęcie" in pm.content_mismatch(work, b, seqs["other"])
    pm.run(work, b, seqs["clean"])
    assert pm.content_mismatch(work, b, seqs["clean"]) is None
    # the pipeline: a backlit found to be another shot is not used (plates_skipped), the clear plate stays
    from engine.pipeline import ObjectPipeline
    pipe = ObjectPipeline(work, b, W, H, client=None, backlit_pattern=seqs["other"], cleanplate_pattern=seqs["clean"])
    assert "backlit" in pipe.plates_skipped and "inne ujęcie" in pipe.plates_skipped["backlit"]
    assert "cleanplate" not in pipe.plates_skipped and pipe.clean is not None


def test_ensure_runs_once_in_the_background(seqs):
    import time
    work = seqs["tmp"] / "work2"
    work.mkdir()
    assert pm.ensure(work, seqs["beauty"], seqs["shifted"]) is None   # started
    t0 = time.time()
    while pm.cached(work, seqs["beauty"], seqs["shifted"]) is None and time.time() - t0 < 30:
        time.sleep(0.1)
    assert pm.ensure(work, seqs["beauty"], seqs["shifted"])["state"] == "shifted"


def test_stage0_pill_and_api_fields():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    js = (root / "static" / "stage0.js").read_text(encoding="utf-8")
    assert "p.content_note" in js and "'pending'" in js
    app = (root / "app.py").read_text(encoding="utf-8")
    assert "_plate_content(name, pattern" in app and '"mismatch": _aux_mismatch(name, pattern, manifest)' in app


def test_override_and_remeasure_2026_11_07(seqs):
    """NEXT_STEPS 5: „użyj mimo to” makes a plate flagged „other” usable (and back); „↻” forgets the stored result,
    the override survives the re-measurement; the stages follow the override."""
    work = seqs["tmp"] / "work3"
    work.mkdir()
    b, o = seqs["beauty"], seqs["other"]
    pm.run(work, b, o)
    pm.run(work, b, seqs["clean"])
    assert pm.content_mismatch(work, b, o)
    pm.set_override(work, o, True)
    r = pm.cached(work, b, o)
    assert r["override"] and r["state"] == "other" and pm.content_mismatch(work, b, o) is None
    assert "mimo to" in pm.describe(r)
    from engine.pipeline import ObjectPipeline
    pipe = ObjectPipeline(work, b, W, H, client=None, backlit_pattern=o)
    assert "backlit" not in pipe.plates_skipped
    pm.forget(work, o)
    assert pm.cached(work, b, o) is None and pm.cached(work, b, seqs["clean"]) is not None   # only this plate forgotten
    pm.run(work, b, o)
    assert pm.cached(work, b, o)["override"]                                                    # the decision stays
    pm.set_override(work, o, False)
    assert "inne ujęcie" in pm.content_mismatch(work, b, o) and "override" not in pm.cached(work, b, o)


def test_stage0_content_buttons_2026_11_07():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    js = (root / "static" / "stage0.js").read_text(encoding="utf-8")
    assert "contentButtons(p)" in js and "'remeasure'" in js and "'override'" in js and "'clear_override'" in js
    assert '@app.post("/api/project/plate/{name}/content")' in (root / "app.py").read_text(encoding="utf-8")
