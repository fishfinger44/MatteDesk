# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 #9: heavy plate work never runs on the event loop – while it runs, GET /api/jobs („Przerwij”, the job
bar) and the WebSocket keep answering. One asyncio loop for both requests (httpx ASGITransport): a TestClient without
its context manager starts a loop per request, so it would not see a blocked loop."""

import asyncio
import json
import time

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.plate_clean as api_plate_clean
import app as app_module

MAX_ANSWER_S = 0.5     # GET /api/jobs during the work
WORK_S = 1.5           # how long the work blocks in total: well above the limit


def _open_shot(tmp_path, n=8):
    plates = tmp_path / "plates_beauty"
    plates.mkdir()
    rng = np.random.default_rng(3)
    for f in range(1, n + 1):
        img = (rng.random((64, 64, 3)) * 0.2 + 0.4 + 0.05 * (f % 2)).astype(np.float32)
        cv2.imwrite(str(plates / f"b_{f:04d}.exr"), img)
    manifest = {"shot": "SHOT_LOOP", "first": 1, "last": n, "plate_w": 64, "plate_h": 64, "colorspace": "ACES2065-1",
                "source_type": "exr_seq", "plates": {"beauty": str(plates / "b_####.exr")}, "work": str(tmp_path)}
    path = tmp_path / "shot_manifest_SHOT_LOOP.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert TestClient(app_module.app).post("/api/project/open", json={"manifest_path": str(path)}).status_code == 200
    return n


def _slow_imread(monkeypatch, n):
    real = cv2.imread

    def slow(*a, **k):
        time.sleep(WORK_S / n)
        return real(*a, **k)
    monkeypatch.setattr(cv2, "imread", slow)


def _post_and_probe(url, body):
    """POST `url` and ask GET /api/jobs all the time: (response, slowest answer in s, a running job was seen)."""
    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://md", timeout=120) as c:
            task = asyncio.create_task(c.post(url, json=body))
            worst, seen = 0.0, False
            while not task.done():
                t0 = time.perf_counter()
                await asyncio.sleep(0.02)                  # in the measure: the loop may stall right here too
                jobs = (await c.get("/api/jobs")).json()["jobs"]
                worst = max(worst, time.perf_counter() - t0)
                seen = seen or any(j["state"] == "running" for j in jobs)
            return await task, worst, seen
    return asyncio.run(main())


def _broadcasts(monkeypatch):
    got = []
    real = app_module.ws_manager.broadcast

    async def spy(msg):
        got.append(msg)
        await real(msg)
    monkeypatch.setattr(app_module.ws_manager, "broadcast", spy)
    return got


def test_flicker_measure_keeps_the_loop_free(tmp_path, monkeypatch):
    n = _open_shot(tmp_path)
    _slow_imread(monkeypatch, n)
    r, worst, seen = _post_and_probe("/api/plate/clean/measure", {"plate": "beauty", "max_frames": n})
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    assert seen and worst < MAX_ANSWER_S, f"/api/jobs answered after {worst:.2f} s"


def test_deflicker_run_keeps_the_loop_free(tmp_path, monkeypatch):
    _open_shot(tmp_path)
    got = _broadcasts(monkeypatch)
    real = api_plate_clean.PlateCleaner.export_gain_maps

    def slow(*a, **k):      # a step after the cleaner's own executor call: it ran on the loop
        time.sleep(WORK_S)
        return real(*a, **k)
    monkeypatch.setattr(api_plate_clean.PlateCleaner, "export_gain_maps", staticmethod(slow))
    r, worst, seen = _post_and_probe("/api/plate/clean/run", {"plate": "beauty", "export_mode": "nuke_group",
                                                              "enable_stage2": False, "enable_denoise": False})
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    assert seen and worst < MAX_ANSWER_S, f"/api/jobs answered after {worst:.2f} s"
    assert any(m.get("type") == "clean_progress" for m in got), "progress still reaches the WebSocket"
    assert (tmp_path / "matte_desk" / "deflicker" / "Deflicker_beauty.json").exists()


def test_camera_track_keeps_the_loop_free(tmp_path, monkeypatch):
    n = _open_shot(tmp_path)
    got = _broadcasts(monkeypatch)
    _slow_imread(monkeypatch, n)
    r, worst, seen = _post_and_probe("/api/camera/track", {"eval_w": 64, "anchor_interval": 3, "export_nk": True})
    assert r.status_code == 200 and r.json()["success"] is True, r.text
    assert len(r.json()["camera2d"]["frames"]) == n
    assert seen and worst < MAX_ANSWER_S, f"/api/jobs answered after {worst:.2f} s"
    assert any(m.get("type") == "camera_progress" for m in got), "progress still reaches the WebSocket"


@pytest.mark.parametrize("url, body", [("/api/plate/clean/measure", {"plate": "beauty", "max_frames": 8}),
                                       ("/api/camera/track", {"eval_w": 64, "anchor_interval": 3, "export_nk": False})])
def test_cancel_stops_during_frame_decoding(tmp_path, monkeypatch, url, body):
    """„Przerwij” answers AND works: the frame loop checks it per frame (the usual 409 „przerwane” of long_job)."""
    n = _open_shot(tmp_path)
    _slow_imread(monkeypatch, n)

    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://md", timeout=120) as c:
            task = asyncio.create_task(c.post(url, json=body))
            for _ in range(200):
                await asyncio.sleep(0.02)
                if any(j["state"] == "running" for j in (await c.get("/api/jobs")).json()["jobs"]):
                    break
            t0 = time.perf_counter()
            assert (await c.post("/api/jobs/cancel")).status_code == 200
            return await task, time.perf_counter() - t0
    r, took = asyncio.run(main())
    assert r.status_code == 409 and "przerwane" in r.json()["detail"], r.text
    assert took < WORK_S * 0.7, f"stopped after {took:.2f} s"


def test_cancel_during_the_flicker_metrics_is_a_cancel(tmp_path, monkeypatch):
    """The metrics themselves cannot stop midway; a „Przerwij” in them still ends as „przerwane”, not as „gotowe”."""
    from engine.jobs import JOBS
    n = _open_shot(tmp_path)
    real = api_plate_clean.DeflickerMetrics.measure_temporal_variance

    def cancel_then_measure(*a, **k):
        JOBS.cancel()
        return real(*a, **k)
    monkeypatch.setattr(api_plate_clean.DeflickerMetrics, "measure_temporal_variance", staticmethod(cancel_then_measure))
    r = TestClient(app_module.app).post("/api/plate/clean/measure", json={"plate": "beauty", "max_frames": n})
    assert r.status_code == 409 and "przerwane" in r.json()["detail"], r.text


@pytest.mark.parametrize("url, body", [("/api/plate/clean/measure", {"plate": "nie_ma"}), ("/api/plate/clean/run", {"plate": "nie_ma"})])
def test_refusals_still_answer_400(tmp_path, url, body):
    """The HTTP errors raised in the worker thread reach the client as before."""
    _open_shot(tmp_path, n=3)
    r = TestClient(app_module.app).post(url, json=body)
    assert r.status_code == 400 and "nie_ma" in r.json()["detail"]
