# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plate alignment RAM guard: worker count capped by free RAM, a dying pool process and a local model that cannot fit
become readable errors (HTTP 507 JSON) instead of a bare BrokenProcessPool / "Internal Server Error"."""
import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from engine import align, procpool
from engine.align import AlignmentPipeline
from engine.procpool import PoolMemoryError, RamLimitError, ram_workers, run_frames

GB = 2 ** 30


def _die_on_two(x):
    if x == 2:
        os._exit(1)        # what the OS does to a process that runs out of memory
    return x * 10


def test_ram_workers_cap(monkeypatch):
    monkeypatch.setattr(procpool, "available_ram_bytes", lambda: 10 * GB)
    assert ram_workers(12, GB) == 6                 # 60 % of 10 GB / 1 GB
    assert ram_workers(12, 100 * GB) == 1           # never below one process
    assert ram_workers(12, 0) == 12                 # no estimate: unchanged
    assert ram_workers(1, GB) == 1
    monkeypatch.setattr(procpool, "available_ram_bytes", lambda: None)
    assert ram_workers(12, GB) == 12                # unknown RAM: unchanged


def test_default_workers_respects_ram(monkeypatch):
    monkeypatch.setattr(procpool, "available_ram_bytes", lambda: 2 * GB)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert procpool.default_workers() == 16
    assert procpool.default_workers(GB) == 1


def test_run_frames_dead_worker_is_readable():
    with pytest.raises(PoolMemoryError) as ei:
        run_frames(_die_on_two, [0, 1, 2, 3, 4, 5], workers=2)
    assert isinstance(ei.value, MemoryError) and "pamięci RAM" in str(ei.value)


def test_run_frames_ok_and_in_process(monkeypatch):
    assert run_frames(_die_on_two, [0, 1, 3], workers=2) == [0, 10, 30]
    assert run_frames(_die_on_two, [0, 1, 3], workers=1) == [0, 10, 30]
    monkeypatch.setattr(procpool, "available_ram_bytes", lambda: GB // 100)   # tight RAM: 1 process, same result
    assert run_frames(_die_on_two, [0, 1, 3], workers=4, per_worker_bytes=GB) == [0, 10, 30]


def test_job_ram_covers_measured_4k():
    assert align.job_ram_bytes(4096, 2731) >= int(0.53 * GB)    # measured peak of the global job at 4K (incl. interpreter)
    assert align.job_ram_bytes(4096, 2731) < 2 * GB


def test_local_model_ram_check(monkeypatch):
    need = align.local_model_ram_bytes(100, 4096, 2731)
    assert 8 * GB < need < 16 * GB                              # ~67 MB x 1.3 per frame at 4K
    monkeypatch.setattr(align, "available_ram_bytes", lambda: 8 * GB)
    with pytest.raises(RamLimitError) as ei:
        align.check_local_model_ram(100, 4096, 2731)
    assert "Skróć zakres klatek" in str(ei.value) and "100 klatek 4096x2731" in str(ei.value)
    align.check_local_model_ram(20, 4096, 2731)                 # fits
    monkeypatch.setattr(align, "available_ram_bytes", lambda: None)
    align.check_local_model_ram(100, 4096, 2731)                # unknown RAM: no check


def _project(tmp_path):
    work = tmp_path / "AL"
    for name in ("plates_beauty", "plates_clean"):
        d = work / name
        d.mkdir(parents=True)
        for f in range(1, 5):
            cv2.imwrite(str(d / f"p_{f:04d}.png"), np.full((64, 64, 3), 100 + f, np.uint8))
    mf = work / "shot_manifest_AL.json"
    mf.write_text(json.dumps({"shot": "AL", "first": 1, "last": 4, "plate_w": 64, "plate_h": 64, "colorspace": "ACES2065-1",
                              "source_type": "exr_seq", "work": str(work),
                              "plates": {"beauty": str(work / "plates_beauty" / "p_####.png"),
                                         "cleanplate": str(work / "plates_clean" / "p_####.png")}}))
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    return client


@pytest.mark.parametrize("exc, expect", [
    (MemoryError("Unable to allocate 128. MiB"), "Brak pamięci RAM przy 64x64, 4 klatek"),
    (RamLimitError("Dopasowanie lokalne potrzebuje ok. 9.0 GB RAM"), "Dopasowanie lokalne potrzebuje ok. 9.0 GB RAM"),
    (PoolMemoryError("Proces roboczy zakończył się nagle"), "Proces roboczy zakończył się nagle"),
])
def test_align_measure_out_of_memory_is_json(tmp_path, monkeypatch, exc, expect):
    client = _project(tmp_path)

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(AlignmentPipeline, "measure_sequence", classmethod(lambda cls, *a, **k: boom()))
    res = client.post("/api/plate/align/measure", json={"plate": "cleanplate", "aux_plate_type": "cleanplate"})
    assert res.status_code == 507
    assert expect in res.json()["detail"]
