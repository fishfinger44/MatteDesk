# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress window thumbnail (user request 2026-11-23): a job keeps the mask of the frame it has just finished
(JOBS.thumb, throttled, small, in memory), GET /api/jobs/thumb composes it on the plate proxy, stages report it."""

import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from engine.comfy_io import job_thumb
from engine.corestab import stabilize_core
from engine.jobs import JOBS


def _job():
    jid = JOBS.start("t", "Thumb")
    JOBS.bind(jid)
    return jid


def test_thumb_is_small_throttled_and_dropped_at_finish(monkeypatch):
    jid = _job()
    try:
        assert JOBS.thumb_of(jid) is None
        big = np.zeros((2000, 3000), bool)
        big[500:1500, 1000:2000] = True
        JOBS.thumb(7, big)
        t = JOBS.thumb_of(jid)
        assert t["frame"] == 7 and t["crop"] is None and max(t["mask"].shape) == JOBS.THUMB_SIDE
        assert t["mask"].dtype == np.uint8 and t["mask"].max() == 255
        j = next(x for x in JOBS.snapshot() if x["id"] == jid)
        assert j["thumb"] == {"frame": 7, "seq": 1}          # the job dict carries no array (JSON for /api/jobs)
        JOBS.thumb(8, np.ones((10, 10), np.float32), crop=[-4, 5, 10, 10])
        assert JOBS.thumb_of(jid)["frame"] == 7               # throttled
        monkeypatch.setattr(JOBS, "THUMB_EVERY_S", 0.0)
        JOBS.thumb(8, np.ones((10, 10), np.float32), crop=[-4, 5, 10, 10])
        t = JOBS.thumb_of(jid)
        assert t["frame"] == 8 and t["crop"] == [-4, 5, 10, 10] and t["mask"].shape == (10, 10)
        assert next(x for x in JOBS.snapshot() if x["id"] == jid)["thumb"]["seq"] == 2
    finally:
        JOBS.finish(jid)
    assert JOBS.thumb_of(jid) is None


def test_job_thumb_outside_a_job_and_bad_input_never_raise():
    for j in [x for x in JOBS.snapshot() if x["state"] == "running"]:
        JOBS.finish(j["id"])
    job_thumb(1, np.ones((4, 4)))                   # no job: no-op
    jid = _job()
    try:
        job_thumb(1, np.ones((4, 4, 3)))            # not a 2-D mask: ignored
        job_thumb(1, "nonsense")
        assert JOBS.thumb_of(jid) is None
    finally:
        JOBS.finish(jid)


def test_corestab_emits_each_frame_with_its_crop_box():
    cores = {}
    for f in range(1, 6):
        m = np.zeros((120, 160), bool)
        m[40:80, 50 + f:100 + f] = True
        cores[f] = m
    got = []
    stab, st = stabilize_core(cores, lambda f: cores[f].astype(np.float32), radius=1, margin=8,
                              emit=lambda f, m, box: got.append((f, m.shape, box)))
    x0, y0, x1, y1 = st["box"]
    assert [g[0] for g in got] == [1, 2, 3, 4, 5]
    assert all(g[1] == (y1 - y0, x1 - x0) and g[2] == (x0, y0, x1 - x0, y1 - y0) for g in got)


def test_endpoint_composes_the_mask_on_the_proxy(monkeypatch):
    W, H = 3840, 2160                                       # proxy 1920 x 1080 (scale 0.5)
    proxy = np.full((1080, 1920, 3), 0.5, np.float32)
    monkeypatch.setitem(app_module.STATE, "work_dir", "x")
    monkeypatch.setattr(app_module, "_plate_size", lambda: (W, H))
    monkeypatch.setattr(app_module, "_load_proxy_display", lambda f, plate="beauty": proxy)
    monkeypatch.setattr(JOBS, "THUMB_EVERY_S", 0.0)
    client = TestClient(app_module.app)
    for j in [x for x in JOBS.snapshot() if x["state"] == "running"]:
        JOBS.finish(j["id"])
    assert client.get("/api/jobs/thumb").status_code == 204          # no job
    jid = _job()
    try:
        assert client.get(f"/api/jobs/thumb?job={jid}").status_code == 204   # nothing finished yet
        # a crop square partly outside the plate (x < 0): only the visible part, mask in its left half
        m = np.zeros((800, 800), np.float32)
        m[:, :400] = 1.0
        JOBS.thumb(3, m, crop=[-200, 400, 800, 800])
        r = client.get(f"/api/jobs/thumb?job={jid}&s=1")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (320, 240)                           # visible 600 x 800 plate px -> 300 x 400 proxy px
        left, right = img[160, 40].astype(int), img[160, 200].astype(int)
        assert left[1] > left[0] + 40                                  # green tint where the mask is
        assert abs(right[1] - right[0]) < 12                           # plain grey plate where it is not
        # full-frame mask: zoomed to its bbox (+ margin), not the whole frame
        full = np.zeros((H, W), bool)
        full[1000:1200, 1800:2000] = True
        JOBS.thumb(4, full)
        img = cv2.imdecode(np.frombuffer(client.get(f"/api/jobs/thumb?job={jid}&s=2").content, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) < 320 and img.shape[0] == img.shape[1]
    finally:
        JOBS.finish(jid)
