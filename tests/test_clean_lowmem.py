# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Low-memory deflicker: the lazy / low-res-gain implementation must reproduce the original (frozen in
tests/ref_clean_v1.py, which held every full-res frame, gain map and corrected frame in RAM) and the endpoint must
turn a MemoryError into a readable JSON error."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import ref_clean_v1 as old
from app import app
from engine.clean import LazyFrames, PlateCleaner

OLD_PARAMS = dict(stage1_window=15, stage1_iterations=2, stage3_radius=3, stage1_sigma=6.0, stage1_content_check=0.0)   # the first port's values: these tests check memory / threads / cache, not the defaults


def _sequence(n=10, w=192, h=112):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.2 + 0.5 * xx / w)[..., None] * np.array([1.0, 0.9, 0.8], np.float32) + 0.05 * np.sin(yy / 9.0)[..., None]
    return [(base * (1 + 0.10 * np.sin(i * 1.7)) * (1 + 0.05 * (xx / w)[..., None] * (i % 3 - 1))).astype(np.float32)
            for i in range(n)]


@pytest.mark.parametrize("s1,s2,s4", [(True, True, True), (False, True, True), (True, False, False), (False, False, True)])
def test_lowmem_matches_original(s1, s2, s4):
    frames = _sequence()
    kw = dict(enable_stage1=s1, enable_stage2=s2, enable_stage4=s4, target_reduction=2.0, **OLD_PARAMS)   # 2.0: never stop after stage 1
    o_frames, o_gains, o_rep, _ = old.PlateCleaner.process_sequence([f.copy() for f in frames], **kw)
    # new engine on a lazy source, like the endpoint (frames re-read on every access)
    n_frames, n_gains, n_rep, _ = PlateCleaner.process_sequence(LazyFrames(len(frames), lambda i: frames[i].copy()), **kw)

    assert n_rep.stages_applied == o_rep.stages_applied
    assert len(n_frames) == len(o_frames) and len(n_gains) == len(o_gains)
    for i in range(len(frames)):
        np.testing.assert_allclose(n_frames[i], o_frames[i], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(n_gains[i], o_gains[i], rtol=1e-5, atol=1e-6)
    assert n_rep.reduction_pct == pytest.approx(o_rep.reduction_pct, abs=1e-3)
    assert (n_rep.gain_min, n_rep.gain_max) == pytest.approx((o_rep.gain_min, o_rep.gain_max), abs=1e-5)


def test_lowmem_gain_export_matches_original(tmp_path):
    frames = _sequence(n=6)
    _, o_gains, _, _ = old.PlateCleaner.process_sequence([f.copy() for f in frames], **OLD_PARAMS)
    _, n_gains, _, _ = PlateCleaner.process_sequence(LazyFrames(6, lambda i: frames[i].copy()), **OLD_PARAMS)
    old.PlateCleaner.export_gain_maps(o_gains, 1, tmp_path / "old")
    PlateCleaner.export_gain_maps(n_gains, 1, tmp_path / "new")
    for i in range(1, 7):
        a = cv2.imread(str(tmp_path / "old" / f"gain_{i:04d}.exr"), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(tmp_path / "new" / f"gain_{i:04d}.exr"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


def test_lazy_frames_sequence():
    lf = LazyFrames(3, lambda i: np.full((2, 2), i, np.float32))
    assert len(lf) == 3 and lf[-1][0, 0] == 2 and [int(f[0, 0]) for f in lf] == [0, 1, 2]
    with pytest.raises(IndexError):
        lf[3]


def test_clean_run_out_of_memory_is_json(tmp_path, monkeypatch):
    work = tmp_path / "OOM"
    plates = work / "plates_beauty"
    plates.mkdir(parents=True)
    for f in range(1, 5):
        cv2.imwrite(str(plates / f"b_{f:04d}.png"), np.full((64, 64, 3), 100 + f, np.uint8))
    mf = work / "shot_manifest_OOM.json"
    mf.write_text(json.dumps({"shot": "OOM", "first": 1, "last": 4, "plate_w": 64, "plate_h": 64, "colorspace": "ACES2065-1",
                              "source_type": "exr_seq", "plates": {"beauty": str(plates / "b_####.png")}, "work": str(work)}))
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200

    def boom(*a, **k):
        raise MemoryError("Unable to allocate 128. MiB")
    monkeypatch.setattr(PlateCleaner, "process_sequence", staticmethod(boom))
    res = client.post("/api/plate/clean/run", json={"plate": "beauty", "export_mode": "nuke_group"})
    assert res.status_code == 507
    assert "Brak pamięci RAM przy 64x64, 4 klatek" in res.json()["detail"]
