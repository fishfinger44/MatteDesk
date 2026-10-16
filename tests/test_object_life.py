# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Object life range (zakres życia obiektu, user request 2026-10-15): `object.json -> life` = visible intervals. Outside
them the matte is empty and nothing is computed (SAM3 propagates each visible piece on its own, MatAnyone runs on
contiguous visible pieces), the export writes black frames and counts them as covered; with a work area the
intersection counts and every visible piece of it needs a key."""

import json

import cv2
import numpy as np
import pytest

from engine.workarea import (life_changed_intervals, life_hints, life_pieces, mask_stats, need_missing,
                             normalize_life, visible_frames)
from test_e3_acceptance import _setup
from test_work_area import _open


def test_life_helpers():
    shot = (1, 100)
    assert normalize_life(None, shot) is None
    assert normalize_life([[1, 100]], shot) is None                       # whole shot = no life range
    assert normalize_life([[30, 10], [80, 200], [25, 40]], shot) == [[10, 40], [80, 100]]
    with pytest.raises(ValueError):
        normalize_life([[150, 200]], shot)                                # visible nowhere
    life = [[10, 40], [80, 100]]
    assert visible_frames(range(38, 43), life) == [38, 39, 40]
    assert life_pieces((30, 90), life) == [[30, 40], [80, 90]]
    assert life_pieces((50, 60), life) == []
    # frames outside the life count as computed
    assert need_missing((1, 100), [[10, 40], [80, 100]], life) == []
    assert need_missing((1, 100), [[10, 30]], life) == [[31, 40], [80, 100]]
    assert need_missing((1, 100), [[10, 30]], None) == [[1, 9], [31, 100]]
    assert life_changed_intervals(None, life, shot) == [[1, 9], [41, 79]]
    assert life_changed_intervals([[10, 40]], [[10, 45]], shot) == [[41, 45]]
    assert life_changed_intervals(life, life, shot) == []


def test_life_hints_exit_and_entry_at_border():
    h, w = 100, 200
    m_in = np.zeros((h, w), bool)
    m_in[40:60, 80:120] = True                     # inside the frame
    m_edge = np.zeros((h, w), bool)
    m_edge[40:60, 190:200] = True                  # touching the right border
    empty = np.zeros((h, w), bool)
    seq = {1: m_in, 2: m_in, 3: m_edge, 4: empty, 5: empty, 6: empty, 7: m_edge, 8: m_in, 9: empty, 10: m_in}
    st = {f: mask_stats(m) for f, m in seq.items()}
    assert st[3][1] and not st[1][1] and st[4] == (0.0, False)
    hints = life_hints({f: v[0] for f, v in st.items()}, {f: v[1] for f, v in st.items()})
    assert {(x["kind"], x["frame"]) for x in hints} == {("end", 4), ("start", 7)}     # F9: 1-frame drop-out, inside
    # already outside the life -> no hint
    assert life_hints({f: v[0] for f, v in st.items()}, {f: v[1] for f, v in st.items()}, [[1, 3], [7, 10]]) == []


def test_core_and_final_only_on_visible_pieces(tmp_path, monkeypatch):
    shot, pipe, client, frames = _setup(tmp_path, n=24)
    start = shot.first + 12
    full = pipe.run_core("puppet", frames)
    k2 = shot.first + 20
    pipe.store.save_key("puppet", k2, full.core[k2], author="auto")
    hidden_key = shot.first + 17
    pipe.store.save_key("puppet", hidden_key, full.core[hidden_key], author="auto")   # outside the life: ignored
    life = [[shot.first, start + 2], [start + 6, shot.last]]
    gap = list(range(start + 3, start + 6))
    pipe.store.update("puppet", life=life)

    loaded = []
    crop0 = pipe.loader.crop
    monkeypatch.setattr(pipe.loader, "crop", lambda f, s, *a, **k: (loaded.append(int(f)), crop0(f, s, *a, **k))[1])
    res = pipe.run_core("puppet", frames)
    assert loaded and not set(loaded) & set(gap)                 # SAM3 never sees the invisible frames
    assert sorted(res.core) == [f for f in frames if f not in gap]
    cdir = pipe.store.dir("puppet") / "core"
    for f in gap:                                                # empty core, written without computing
        m = cv2.imread(str(cdir / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
        assert m is not None and not m.any()
    cj = json.loads((cdir / "core.json").read_text(encoding="utf-8"))
    assert cj["coverage"] == life and cj["life"] == life
    assert all(not (gap[0] <= a <= gap[-1] or gap[0] <= b <= gap[-1]) for a, b in res.recomputed_intervals)
    # the same inputs again -> every interval from the cache
    again = pipe.run_core("puppet", frames)
    assert again.recomputed_intervals == [] and all(np.array_equal(again.core[f], res.core[f]) for f in res.core)

    # MatAnyone: one video per contiguous visible piece, never across the gap
    from engine.refine import Finalizer
    chunks = []
    orig = Finalizer._matanyone_segment

    def rec(self, seg, fr, *a, **k):
        chunks.append(list(fr))
        return orig(self, seg, fr, *a, **k)
    monkeypatch.setattr(Finalizer, "_matanyone_segment", rec)
    fullf, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")
    assert chunks and all(c == list(range(c[0], c[-1] + 1)) for c in chunks)
    assert not any(set(c) & set(gap) for c in chunks)
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert rj["coverage"] == life and rj["life"] == life
    from engine.workarea import run_coverage
    assert run_coverage(pipe.store.dir("puppet"), run_dir.name) == life


def test_api_life_core_final_export_black_frames(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch, n=24)
    before = mf.read_text(encoding="utf-8")
    k0 = sorted(int(k["frame"]) for k in api.get("/api/object/puppet/keys").json()["keys"])[0]
    d = api.get("/api/object/puppet/life").json()
    assert d["life"] is None and d["is_whole"] and d["shot"] == [shot.first, shot.last]
    assert api.put("/api/object/puppet/life", json={"life": [[shot.last + 3, shot.last + 9]]}).status_code == 400

    # a visible piece without a key -> 409 no_key_in_range listing the piece
    life = [[shot.first, shot.first + 2], [k0 - 3, k0 + 4]]
    d = api.put("/api/object/puppet/life", json={"life": life}).json()
    assert d["life"] == life
    obj = json.loads((tmp_path / "work" / "matte_desk" / "objects" / "puppet" / "object.json").read_text(encoding="utf-8"))
    assert obj["life"] == life and mf.read_text(encoding="utf-8") == before        # object data, never the manifest
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "no_key_in_range", r.text
    assert r.json()["detail"]["pieces"] == [[shot.first, shot.first + 2]]

    life = [[k0 - 3, k0 + 4]]
    api.put("/api/object/puppet/life", json={"life": life})
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 200, r.text
    st = api.get("/api/object/puppet/stages").json()
    core = next(s for s in st["stages"] if s["id"] == "core")
    assert core["state"] == "done" and core["scope"] == "all" and core["missing_shot"] == []   # hidden frames = covered
    assert st["life"] == life
    assert api.post("/api/object/puppet/finalize", json={"edge_mode": "core"}).status_code == 200

    # export "całość": no false holes, black frames outside the life
    r = api.post("/api/export", json={"scope": "all", "multichannel": False, "format": "png"})
    assert r.status_code == 200, r.text
    exp_dir = tmp_path / "work" / "matte_desk" / "export" / "mattes" / "puppet"
    files = {int(p.stem.rsplit("_", 1)[-1]): p for p in exp_dir.iterdir()}
    assert sorted(files) == list(range(shot.first, shot.last + 1))
    for f, p in files.items():
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if life[0][0] <= f <= life[0][1]:
            assert a.any(), f
        else:
            assert not a.any(), f

    # the life range grows -> the core is stale for the frames that became visible; export refuses the hole
    api.put("/api/object/puppet/life", json={"life": [[k0 - 3, k0 + 6]]})
    st = api.get("/api/object/puppet/stages").json()
    core = next(s for s in st["stages"] if s["id"] == "core")
    assert core["state"] == "stale" and "zakres życia" in core["why"]
    assert any(a <= k0 + 5 <= b for a, b in st["stale_intervals"])
    r = api.post("/api/export", json={"scope": "all", "multichannel": False, "format": "png"})
    assert r.status_code == 409, r.text
    # back to the whole shot: no life key left
    d = api.put("/api/object/puppet/life", json={"life": None}).json()
    assert d["life"] is None
    obj = json.loads((tmp_path / "work" / "matte_desk" / "objects" / "puppet" / "object.json").read_text(encoding="utf-8"))
    assert "life" not in obj


def test_api_life_with_work_area_intersection(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch, n=24)
    k0 = sorted(int(k["frame"]) for k in api.get("/api/object/puppet/keys").json()["keys"])[0]
    # work area and life overlap only around the key -> core on the intersection, out-of-life frames of the work
    # area empty; the work-area export covers everything
    api.put("/api/project/work_area", json={"range": [k0 - 4, k0 + 4]})
    api.put("/api/object/puppet/life", json={"life": [[shot.first, k0 + 1]]})
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 200, r.text
    assert api.post("/api/object/puppet/finalize", json={"edge_mode": "core"}).status_code == 200
    r = api.post("/api/export", json={"scope": "work_area", "multichannel": False, "format": "png"})
    assert r.status_code == 200, r.text
    # a work area entirely outside the life -> 409 not_visible
    api.put("/api/project/work_area", json={"range": [k0 + 3, k0 + 6]})
    r = api.post("/api/object/puppet/compute_all")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "not_visible", r.text
