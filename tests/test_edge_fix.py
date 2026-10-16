# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z27 pkt 1 – the edge correction is a user decision, not an automatic second round.

`finalize` writes one version + the edge report (qa.json -> edge.pieces, runs/v###/edge/e_####.npz);
`apply_edge_fix` ("Zastosuj poprawkę krawędzi") = the former automatic second round, bit for bit, as a new version;
an unticked piece is never shifted.
"""

import json

import numpy as np

from engine.edge import apply_shift
from engine.lupa import thin_parts
from test_e5b_acceptance import _pipe, _rigid


def _legacy_two_rounds(pipe, name, frames, mode):
    """Reference: the pre-Z27 `finalize` (first pass + automatic in-memory correction pass), verbatim logic."""
    from engine.edge import correction_map, measure_edge, needs_correction
    from engine.crop import extract_crop, paste_crop
    obj = pipe.store.get(name)
    segs = pipe.segments(name)
    keys = pipe.keys(name)
    start = obj["tool"].get("start_frame") or min(keys)
    core2, zres = pipe._final_core(name, frames, obj, keys)
    full, _rep, _rd, _qa = pipe._finalize_once(name, obj, segs, frames, core2, keys, start, mode, None, zres)
    per, meas = [], {}
    for f in frames:
        s = next((s for s in segs if s.first <= f <= s.last), None)
        if s is None:
            per.append({})
            continue
        m = measure_edge(extract_crop(full[f].astype(np.float32), s), pipe.loader.crop(f, s).mean(axis=2))
        meas[f] = (s, m)
        per.append(m.piece_median)
    fix = needs_correction(per)
    assert any(fix.values())
    disp = {}
    for f, (s, m) in meas.items():
        m.piece_median = {k: v for k, v in m.piece_median.items() if fix.get(k)}
        c = extract_crop(core2[f].astype(np.float32), s) > 0.5
        disp[f] = paste_crop(correction_map(c, m), s, pipe.w, pipe.h)
    core3 = {f: (core2[f] if f in keys or f not in disp else apply_shift(core2[f], disp[f]) | thin_parts(core2[f]))
             for f in frames}
    full2, _rep2, rd2, _qa2 = pipe._finalize_once(name, obj, segs, frames, core3, keys, start, mode, None, zres)
    return full2, rd2


def _setup(tmp_path):
    shot = _rigid(tmp_path)
    pipe, frames = _pipe(tmp_path, shot, use_mplate=False, grow_px=1.5)   # SAM core 1.5 px too wide (halo)
    pipe.store.update("obj", flags=["rigid"])
    pipe.run_core("obj", frames)
    return shot, pipe, frames


def test_finalize_has_no_automatic_second_round(tmp_path):
    _shot, pipe, frames = _setup(tmp_path)
    _full, _rep, run_dir = pipe.finalize("obj", frames, edge_mode="core")
    runs = sorted(p.name for p in (pipe.store.dir("obj") / "runs").iterdir())
    assert runs == ["v001"], runs                       # one pass only
    qa = json.loads((run_dir / "qa.json").read_text())
    e = qa["edge"]
    assert e["corrected"] is False and e["fix_available"] is True
    assert e["pieces_to_fix"] == len(e["pieces"]) > 0 and e["before"] >= 0.75
    p0 = e["pieces"][0]
    assert {"id", "first", "last", "median_px", "kind", "per_frame"} <= set(p0)
    assert p0["kind"] == "halo"                          # mask too wide
    assert (run_dir / "edge" / "edge.json").exists() and list((run_dir / "edge").glob("e_*.npz"))


def test_apply_equals_legacy_second_round_bitwise(tmp_path):
    _shot, ref_pipe, frames = _setup(tmp_path / "ref")
    ref_full, _ = _legacy_two_rounds(ref_pipe, "obj", frames, "core")

    _shot, pipe, frames = _setup(tmp_path / "new")
    pipe.finalize("obj", frames, edge_mode="core")
    full, _rep, run_dir = pipe.apply_edge_fix("obj", frames)
    assert run_dir.name == "v002"
    for f in frames:
        assert np.array_equal(full[f], ref_full[f]), f
    # the written EXR alphas too
    for p in (run_dir / "alpha").rglob("*.exr"):
        q = tmp_path / "ref" / p.relative_to(tmp_path / "new")
        assert p.read_bytes() == q.read_bytes(), p
    qa = json.loads((run_dir / "qa.json").read_text())
    assert qa["edge"]["corrected"] and qa["edge"]["base_run"] == "v001"
    assert qa["edge"]["after"] <= 0.5                   # same threshold as E5b
    assert pipe.store.get("obj")["tool"]["active_run"] == "v002"


def test_unticked_piece_is_not_shifted(tmp_path):
    _shot, pipe, frames = _setup(tmp_path)
    _full, _rep, run_dir = pipe.finalize("obj", frames, edge_mode="core")
    ids = [p["id"] for p in json.loads((run_dir / "qa.json").read_text())["edge"]["pieces"]]
    assert len(ids) >= 2
    skip = ids[0]
    core2, _ = pipe._final_core("obj", frames, pipe.store.get("obj"), pipe.keys("obj"))
    d_all = pipe.edge_disp(run_dir / "edge", pipe.segments("obj"), frames, core2, set(ids))
    d_skip = pipe.edge_disp(run_dir / "edge", pipe.segments("obj"), frames, core2, set(ids[1:]))
    from engine.pipeline import load_edge_measure, piece_id
    f = next(iter(d_all))
    seg_id, m = load_edge_measure(run_dir / "edge" / f"e_{f:04d}.npz")
    seg = next(s for s in pipe.segments("obj") if s.seg_id == seg_id)
    pts = [p for p, pc in zip(m.points, m.pieces) if piece_id(pc) == skip]
    far = [p for p, pc in zip(m.points, m.pieces) if piece_id(pc) not in ids]
    assert pts
    # at the samples of the unticked piece the displacement drops (its own median no longer contributes)
    x, y = int(round(pts[len(pts) // 2][0])) + seg.crop[0], int(round(pts[len(pts) // 2][1])) + seg.crop[1]
    assert abs(d_skip[f][y, x]) < abs(d_all[f][y, x])
    _full2, _rep2, rd2 = pipe.apply_edge_fix("obj", frames, "v001", skip=[skip])
    assert json.loads((rd2 / "qa.json").read_text())["edge"]["skipped"] == [skip]
    _ = far


def test_edge_fix_api(tmp_path, monkeypatch):
    import app as app_mod
    from fastapi.testclient import TestClient
    _shot, pipe, frames = _setup(tmp_path)
    pipe.finalize("obj", frames, edge_mode="core")
    monkeypatch.setattr(app_mod, "_pipeline", lambda: pipe)
    monkeypatch.setattr(app_mod, "_shot_frames", lambda: frames)
    monkeypatch.setattr(app_mod, "_objects", lambda: pipe.store)
    monkeypatch.setattr(app_mod, "_require_stage", lambda name, stage: None)
    monkeypatch.setattr(app_mod, "_proxy_scale", lambda: 0.5)
    c = TestClient(app_mod.app)
    r = c.get(f"/api/object/obj/edge_pieces/{frames[3]}")
    assert r.status_code == 200, r.text
    pcs = r.json()["pieces"]
    assert pcs and all(p["kind"] in ("halo", "cut") and len(p["pts"]) > 0 for p in pcs)
    r = c.post("/api/object/obj/edge_fix", json={"skip": []})
    assert r.status_code == 200, r.text
    assert r.json()["run"] == "v002"
    # all pieces unticked -> 409 with a message, no new version
    ids = [p["id"] for p in json.loads((pipe.store.dir("obj") / "runs" / "v001" / "qa.json").read_text())["edge"]["pieces"]]
    r = c.post("/api/object/obj/edge_fix", json={"run": "v001", "skip": ids})
    assert r.status_code == 409 and "odcink" in r.json()["detail"]
    assert sorted(p.name for p in (pipe.store.dir("obj") / "runs").iterdir()) == ["v001", "v002"]
