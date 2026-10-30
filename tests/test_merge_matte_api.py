# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract of the MergeMatte palette routes (engine/merge_matte_api.py) and settings (engine/final_touch.py):
settings live in final_touch.json, never touch the alpha / corrections hash, knobs are clamped to the node's ranges, the
preview of a region equals the same region of a whole-frame run, every Comp background works."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from engine import final_touch as ft
from engine import merge_matte as mm
from engine import merge_matte_nk as mmnk
from engine.merge_matte_api import make_router

W, H = 160, 120


class _Store:
    def __init__(self, root: Path):
        self.root = root

    def dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def _scene():
    rng = np.random.default_rng(5)
    plate = (rng.random((H, W, 3)) * 200 + 30).astype(np.uint8)                 # display BGR u8
    yy, xx = np.mgrid[0:H, 0:W]
    a = np.clip((40 - np.hypot(xx - 80, yy - 60)) / 10, 0, 1).astype(np.float32)   # soft disc
    bgp = np.full((H, W, 3), 90, np.uint8)
    return plate, a, bgp


@pytest.fixture()
def client(tmp_path):
    plate, a, bgp = _scene()
    calls = {"require": []}

    def require(n, s):
        calls["require"].append((n, s))

    app = FastAPI()
    app.include_router(make_router(objects=lambda: _Store(tmp_path), require_stage=require,
                                   ft_base=lambda n: {"run": "v001"}, ft_alpha=lambda od, b, f, w, h: a,
                                   plate_size=lambda: (W, H), display_u8=lambda p, f: plate if p == "beauty" else bgp,
                                   has_project=lambda: True))
    c = TestClient(app)
    c.scene = (plate, a, bgp)
    c.calls = calls
    c.root = tmp_path
    return c


def test_get_defaults_and_knob_table(client):
    r = client.get("/api/object/Hair/merge_matte").json()
    assert r["on"] is False and r["node"] == "MergeMatte_v01"
    assert r["knobs"] == mmnk.KNOB_DEFAULTS
    assert [s["name"] for s in r["specs"]] == [n for n, *_ in mmnk.KNOB_SPECS]
    assert r["presets"]["sh010"]["choke"] == 6.0


def test_put_stores_in_final_touch_json_clamps_and_keeps_alpha_sig(client):
    od = client.root / "Hair"
    od.mkdir(exist_ok=True)
    sig0 = ft.corrections_sig(ft.load(od))
    # a typed value may pass the slider (1..100) up to the field range (4x: 400), not further (2026-10-30)
    r = client.put("/api/object/Hair/merge_matte", json={"on": True, "knobs": {"decon_spread": 500, "darken": 0.5}}).json()
    assert r["on"] is True and r["knobs"]["decon_spread"] == 400 and r["knobs"]["darken"] == 0.5
    assert client.calls["require"] == [("Hair", "final_touch")]
    d = ft.load(od)
    assert d["merge_matte"]["knobs"] == {"decon_spread": 400.0, "darken": 0.5}   # only what differs from the node
    assert client.put("/api/object/Hair/merge_matte", json={"knobs": {"decon_spread": 150}}).json()["knobs"]["decon_spread"] == 150
    assert ft.corrections_sig(d) == sig0                                           # the alpha / export signature unchanged
    assert client.put("/api/object/Hair/merge_matte", json={"knobs": {"nope": 1}}).status_code == 400
    r = client.put("/api/object/Hair/merge_matte", json={"reset": "sh010"}).json()
    assert r["knobs"] == mmnk.PRESET_SH010 and r["on"] is True


def _png(resp):
    assert resp.status_code == 200, resp.text
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


@pytest.mark.parametrize("bg", ["#1b2a4a", "checker", "bg"])
def test_region_preview_equals_whole_frame(client, bg):
    plate, a, bgp = client.scene
    client.put("/api/object/Hair/merge_matte", json={"reset": "sh010"})
    resp = client.get(f"/api/object/Hair/merge_matte/frame/7?x=50&y=30&w=60&h=50&bg={bg.replace('#', '%23')}")
    img = _png(resp)
    assert img.shape == (50, 60, 3) and float(resp.headers["X-MM-Scale"]) == 1.0
    bgv = mm.checker_srgb(H, W) if bg == "checker" else bgp[..., ::-1].astype(np.float32) / 255 if bg == "bg" else mm.hex_srgb(bg)
    full = mm.comp_srgb(plate[..., ::-1].astype(np.float32) / 255, a, bgv, mmnk.PRESET_SH010)
    ref = (np.clip(full[30:80, 50:110], 0, 1) * 255 + 0.5).astype(np.uint8)
    assert int(np.abs(img.astype(int) - ref).max()) <= 1


def test_large_region_is_computed_smaller(client):
    resp = client.get("/api/object/Hair/merge_matte/frame/1?max_px=100000")
    s = float(resp.headers["X-MM-Scale"])
    img = _png(resp)
    assert s == 1.0 and img.shape == (H, W, 3)                  # 160×120 + context < 100 000 px → full size
    plate, a, _ = client.scene
    big = np.tile(plate, (4, 4, 1)), np.tile(a, (4, 4))
    out, s2 = mm.preview_region(big[0][..., ::-1].astype(np.float32) / 255, big[1], mm.hex_srgb("#000000"), None, None, max_px=100_000)
    assert s2 < 1.0 and out.shape[1] == round(W * 4 * s2)


def test_bad_background_is_400(client):
    assert client.get("/api/object/Hair/merge_matte/frame/1?bg=zzz").status_code == 400


def test_skipped_sections_equal_the_group_formula():
    """A section with amount 0 is skipped in the port; with a tiny amount the result converges to the skipped one."""
    rng = np.random.default_rng(2)
    a = np.clip(rng.random((40, 40)) * 1.5 - 0.25, 0, 1).astype(np.float32)
    fg = np.dstack([rng.random((40, 40, 3)).astype(np.float32) * a[..., None], a])
    bg = np.zeros((40, 40, 4), np.float32)
    bg[..., :3] = 0.3
    base = mm.merge_matte(fg, bg, {"wrap": 0, "flare": 0, "bleed": 0})
    tiny = mm.merge_matte(fg, bg, {"wrap": 1e-6, "flare": 1e-6, "bleed": 1e-6, "flare_threshold": 0.1})
    assert np.abs(base - tiny).max() < 1e-4


def test_nk_route_gives_pasteable_group_with_the_object_knobs(client):
    """The palette's „⧉ Nuke” button (2026-10-30): clipboard text of the group with the stored knobs + a file copy."""
    client.put("/api/object/Hair 2/merge_matte", json={"knobs": {"darken": 0.25}})
    r = client.get("/api/object/Hair 2/merge_matte/nk").json()
    assert r["node"] == "MergeMatte_Hair_2"
    assert r["text"].startswith("set cut_paste_input [stack 0]") and " name MergeMatte_Hair_2" in r["text"]
    assert r["text"] == mmnk.paste_text({**mmnk.KNOB_DEFAULTS, "darken": 0.25}, name="MergeMatte_Hair_2")
    assert Path(r["path"]).read_text("utf-8") == r["text"]


def test_defaults_are_a_plain_over():
    """User decision 2026-10-30: the MergeMatte defaults change nothing – the group at its defaults = a plain linear
    Merge over (FG premultiplied over BG), so moving a slider away from its default is the only thing that alters the
    comp."""
    rng = np.random.default_rng(3)
    h, w = 64, 80
    fgc = rng.random((h, w, 3)).astype(np.float32) * 0.9
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.clip((24 - np.hypot(xx - 40, yy - 32)) / 8, 0, 1).astype(np.float32)
    fg = np.dstack([fgc * a[..., None], a])
    bg = np.dstack([rng.random((h, w, 3)).astype(np.float32), np.zeros((h, w), np.float32)])
    out = mm.merge_matte(fg, bg, mmnk.KNOB_DEFAULTS)
    plain = fg[..., :3] + bg[..., :3] * (1 - a[..., None])
    assert np.abs(out[..., :3] - plain).max() < 1e-5
    for n, *_ in mmnk.KNOB_SPECS:   # every amount knob is 0 at its default
        if n in ("decon", "darken", "match", "wrap", "flare", "bleed", "edge_blur", "choke", "soften"):
            assert mmnk.KNOB_DEFAULTS[n] == 0, n
