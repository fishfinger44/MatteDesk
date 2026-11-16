# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU memory forecast before a stage (2026-11-06, engine/vram_forecast.py + api/vram.py + static/vramcheck.js):
a line fitted on measured passes, the worst pass of a run against the card, a resolution that fits, the window text;
the table in git is valid; the UI asks before VideoMaMa and never blocks on an unknown forecast."""
import json
from pathlib import Path

import pytest

from engine import vram_forecast as vf

ROOT = Path(__file__).resolve().parent.parent
# x of a point = megapixels of ONE work frame (side²): 1000 px = 1 Mpx, 2000 px = 4 Mpx (2026-11-06: the frame count of a
# VideoMaMa batch barely changes the peak, the frame size does)
T = {"models": {"vm": {"default": [{"side": 1000, "frames": 24, "peak_gb": 9.0, "base_gb": 1.0},
                                   {"side": 2000, "frames": 24, "peak_gb": 17.0, "base_gb": 1.0}],
                       "offload": [{"side": 1000, "frames": 24, "peak_gb": 4.0}]}}}


def test_fit_line_without_the_desktop_share():
    a, b = vf.fit(T["models"]["vm"]["default"])
    assert b == pytest.approx(8 / 3) and a == pytest.approx(8 - 8 / 3)   # (9-1) and (17-1) at 1 and 4 Mpx
    assert vf.predict("vm", 2.5, table=T) == pytest.approx(12.0)
    assert vf.predict("vm", 2.5, "offload", T) is None                  # one point: no line
    assert vf.predict("nope", 1.0, table=T) is None


def test_check_ok_over_unknown_and_the_side_that_fits():
    passes = [(1024, 24), (512, 24), (3000, 12)]                         # worst: the largest frame, 3000² = 9 Mpx
    r = vf.check("vm", passes, card=16.0, table=T)
    assert (r["side"], r["frames"]) == (3000, 12) and r["state"] == "over" and r["peak_gb"] == pytest.approx(29.3, abs=0.1)
    assert r["extrapolated"] is True
    s = r["suggest_side"]
    assert s % 64 == 0 and vf.predict("vm", vf.mpx(s, s), table=T) <= r["budget_gb"]
    assert vf.predict("vm", vf.mpx(s + 64, s + 64), table=T) > r["budget_gb"]   # the largest that fits
    assert vf.check("vm", [(512, 24)], card=24.0, table=T)["state"] == "ok"
    assert vf.check("vm", [(512, 24)], card=None, table=T)["state"] == "unknown"           # Comfy not answering
    assert vf.check("vm", [(512, 24)], card=24.0, variant="offload", table=T)["state"] == "unknown"
    assert vf.check("vm", [], card=24.0, table=T)["state"] == "unknown"
    # other programs on the card shrink the budget (2000 px = 16 GB of the model)
    assert vf.check("vm", [(2000, 24)], card=24.0, table=T)["state"] == "ok"
    assert vf.check("vm", [(2000, 24)], card=24.0, others_gb=8.0, table=T)["state"] == "over"


def test_videomama_passes_follow_the_stage():
    s = {"max_side": 1280, "batch": 24}
    assert vf.videomama_passes([3728, 600, 900], [100, 5, 0], s) == [(1280, 24), (600, 5)]   # scaled down, batch cap, empty chunk


def test_card_gb_from_comfy_then_nvidia_smi():
    assert vf.card_gb({"devices": [{"vram_total": 24 * 2 ** 30}]}) == pytest.approx(24.0)
    assert vf.card_gb(None, [{"mem_total_mb": 12288}]) == pytest.approx(12.0)
    assert vf.card_gb(None, None) is None


def test_table_in_git_is_valid():
    t = json.loads((ROOT / "engine" / "vram_table.json").read_text(encoding="utf-8"))
    pts = t["models"]["videomama"]["default"]
    assert vf.fit(pts) is not None
    for p in pts:
        assert p["side"] > 0 and p["peak_gb"] > 0 and p.get("source")
    # measured on the RTX 4090: the line holds every point within 0.5 GB (no offload) / 1.2 GB (offload)
    for v, tol in (("default", 0.5), ("offload", 1.2)):
        ps = t["models"]["videomama"][v]
        a, b = vf.fit(ps)
        assert max(abs(p["peak_gb"] - p["base_gb"] - (a + b * vf.point_x(p))) for p in ps) <= tol


def test_window_text_and_api(monkeypatch):
    from api.vram import forecast_text
    r = {"state": "over", "peak_gb": 26.1, "side": 2048, "frames": 12, "card_gb": 24.0, "budget_gb": 23.2,
         "suggest_side": 1920, "extrapolated": True}
    t = forecast_text(r, "VideoMaMa")
    assert len(t) == 4 and "26.1 GB" in t[0] and "1920 px" in t[2]
    assert forecast_text({"state": "ok"}, "VideoMaMa") == []
    en = json.loads((ROOT / "static" / "i18n" / "en.json").read_text(encoding="utf-8"))["strings"]
    for k in ("Nadmiar przeleje się do RAM: liczenie kilka razy wolniejsze, może skończyć się brakiem pamięci po wielu minutach.",
              "Prognoza poza zakresem pomiarów – przybliżona.", "Zmieści się „Rozdzielczość” ≤ {0} px.",
              "{0}: prognoza pamięci GPU {1} GB przy kadrze {2} px × {3} kl. – karta ma {4} GB (wolne na model ok. {5} GB)."):
        assert k in en


def test_api_endpoint_uses_the_work_segments(monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.vram import make

    class Pipe:
        def _visible(self, name, frames):
            return list(frames)

        def _work_segments(self, name, frames):
            return [SimpleNamespace(first=1, last=20, side=3000), SimpleNamespace(first=21, last=40, side=400)]

    class Store:
        def get(self, name):
            if name != "o":
                raise FileNotFoundError(name)
            return {"videomama": {"max_side": 2048}}

    class Client:
        def get_system_stats(self):
            return {"devices": [{"vram_total": 12 * 2 ** 30, "torch_vram_total": 0}]}

    monkeypatch.setattr("engine.sysinfo.gpu_info", lambda *a, **k: None)
    monkeypatch.setattr(vf, "TABLE_PATH", ROOT / "engine" / "vram_table.json")
    router, _ = make(objects=lambda: Store(), pipeline=lambda: Pipe(), work_frames=lambda: list(range(1, 41)),
                     shot_frames=lambda: list(range(1, 41)), comfy_client=lambda: Client())
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    r = c.get("/api/object/o/vram_forecast?stage=videomama").json()
    # 12 GB card -> VideoMaMa offload (auto < 20 GB); the table has no offload points yet -> unknown, never blocks
    assert r["variant"] == "offload" and r["card_gb"] == 12.0 and r["state"] == "over"     # 2048 px on 12 GB
    Client.get_system_stats = lambda self: {"devices": [{"vram_total": 24 * 2 ** 30, "torch_vram_total": 0}]}
    r = c.get("/api/object/o/vram_forecast?stage=videomama").json()
    assert r["variant"] == "default" and (r["side"], r["frames"]) == (2048, 20) and r["state"] == "over"
    assert r["suggest_side"] and r["suggest_side"] < 2048 and r["text"]
    r = c.get("/api/object/o/vram_forecast?stage=videomama&frame=38").json()   # the preview: ±6 frames, all in the small segment
    assert r["frames"] <= 13 and r["side"] == 400 and r["state"] == "ok"
    assert c.get("/api/object/o/vram_forecast?stage=ltx").status_code == 400
    assert c.get("/api/object/x/vram_forecast").status_code == 404


def test_ui_asks_before_videomama_and_never_blocks_unknown():
    js = (ROOT / "static" / "vramcheck.js").read_text(encoding="utf-8")
    assert "vmRun" in js and "vmStagePv" in js and "/vram_forecast?stage=videomama" in js
    assert "r.state !== 'over'" in js and "catch (e) { return 'go'; }" in js
    assert '<script src="/static/vramcheck.js"></script>' in (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    jobs = (ROOT / "static" / "jobs.js").read_text(encoding="utf-8")
    assert "quietWarned" in jobs and "ci.spill_gb" in jobs   # a quiet frame preview says the spill / Comfy error too
