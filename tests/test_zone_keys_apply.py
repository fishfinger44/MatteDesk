# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 4 of docs/PLAN_HAIR_TOOL.md (synthetic): zone keys, propagation between keys (interpolation / union, bbox
fallback), the per-frame / per-sequence zone cache (second pass = 0 frames computed), "Zastosuj" = a new version with
only the zones re-applied on the base alpha (outside the zones bitwise), API keys / region / apply, CLI."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.zonecache import ZoneCache, frame_hash, sequence_hash, zone_signature
from engine.zones import (Zone, apply_zones, delete_zone_key, key_zone, propagate_regions, set_zone_key, zone_keys,
                          zone_mask_at)
from tests.test_localmatte import _motley, _scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_keys_helpers():
    z = Zone("z_01", "add", "matte", 10, polygon=_rect(0, 0, 10, 10))
    assert list(zone_keys(z)) == [10]
    set_zone_key(z, 20, _rect(5, 5, 15, 15), bg_strokes=[{"p": [[1, 1], [2, 2]], "radius_px": 2}])
    set_zone_key(z, 10, _rect(1, 1, 11, 11))                 # base key updated in place
    assert list(zone_keys(z)) == [10, 20] and z.polygon == [[1.0, 1.0], [11.0, 1.0], [11.0, 11.0], [1.0, 11.0]]
    kz = key_zone(z, 20)
    assert kz.frame == 20 and kz.polygon == [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]] and len(kz.bg_strokes) == 1
    delete_zone_key(z, 10)                                   # base removed -> nearest key promoted
    assert z.frame == 20 and list(zone_keys(z)) == [20] and z.polygon[0] == [5.0, 5.0]
    with pytest.raises(ValueError):
        delete_zone_key(z, 20)
    d = z.to_dict()
    assert d["keys"] == {"20": {"polygon": z.polygon, "strokes": [], "bg_strokes": z.bg_strokes}}


def test_propagation_interpolates_between_keys_and_tracks_outside():
    H, W = 200, 300
    frames = list(range(1, 21))
    cores = {}
    for f in frames:                          # a block moving right 4 px per frame (bbox fallback tracking)
        c = np.zeros((H, W), bool)
        x = 40 + 4 * (f - 1)
        c[100:160, x:x + 60] = True
        cores[f] = c
    z = Zone("z_01", "add", "matte", 1, polygon=_rect(30, 60, 70, 100), static=False)
    set_zone_key(z, 11, _rect(80, 60, 120, 100))               # same vertex count -> interpolation
    regions, bgs, tr, info = propagate_regions(z, frames, cores, (H, W))
    assert info["keys"] == [1, 11] and info["interpolated"]
    # key frames = exactly the drawing
    assert np.array_equal(regions[1], zone_mask_at(key_zone(z, 1), (H, W)))
    assert np.array_equal(regions[11], zone_mask_at(key_zone(z, 11), (H, W)))
    # halfway: the interpolated rectangle sits halfway (x 55..95)
    ys, xs = np.where(regions[6])
    assert abs(xs.min() - 55) <= 2 and abs(xs.max() - 95) <= 2 and abs(ys.min() - 60) <= 1
    # after the last key: moved with the core (bbox fallback: +4 px per frame)
    ys, xs = np.where(regions[16])
    assert abs(xs.min() - (80 + 4 * 5)) <= 2
    # union when the vertex counts differ
    z2 = Zone("z_02", "add", "matte", 1, polygon=_rect(30, 60, 70, 100))
    set_zone_key(z2, 11, [[80, 60], [120, 60], [120, 100], [100, 110], [80, 100]])
    regions2, _, _, info2 = propagate_regions(z2, frames, cores, (H, W))
    r6 = regions2[6]
    assert r6[80, 55] and r6[80, 95] and not r6[80, 110]          # union of key a moved +20 px and key b moved -20 px
    assert all(bgs[f].dtype == bool for f in frames)


def test_zone_cache_roundtrip_and_hashes(tmp_path):
    zc = ZoneCache(tmp_path / "obj", "z_01")
    a = np.zeros((64, 64), np.float32); a[10:30, 20:40] = 0.6
    area = a > 0
    h = frame_hash("sig", area.astype(np.uint8), ["plate", 1, 2], area)
    assert zc.get(h, 5, (64, 64)) is None and not zc.has(h, 5)
    zc.put(h, 5, a, area)
    back = zc.get(h, 5, (64, 64))
    assert back is not None and np.abs(back - a).max() < 1.0 / 65535 * 2 and zc.has(h, 5)
    assert zc.stats() == {"cached": 1, "computed": 1}
    # the hash changes with the core / plate / region / zone definition
    z = Zone("z_01", "add", "matte", 1, polygon=_rect(0, 0, 10, 10))
    s1 = zone_signature(z.to_dict())
    z.params = {"chroma_lo": 2.0}
    assert zone_signature(z.to_dict()) != s1
    z.enabled = False
    assert zone_signature(z.to_dict()) == zone_signature({**z.to_dict(), "enabled": True})
    assert frame_hash("sig", area.astype(np.uint8), ["plate", 1, 3], area) != h
    hs = sequence_hash("sig", [area.astype(np.uint8)] * 2, [[1], [2]], [area, area])
    assert len(hs) == 16 and hs != sequence_hash("sig", [area.astype(np.uint8)] * 2, [[1], [9]], [area, area])
    zc.put_transforms("t1", {1: np.stack([np.eye(2, 3, dtype=np.float32)] * 3)})
    assert zc.get_transforms("t1")[1].shape == (3, 2, 3)
    assert zc.prune(["other"]) == 1 and zc.get(h, 5, (64, 64)) is None


def test_apply_zones_uses_cache_second_pass_zero_computed(tmp_path):
    img, core, hair, line, region = _scene(_motley(192, 256))
    frames = [1, 2, 3]
    ys, xs = np.where(region)
    z = Zone("z_01", "add", "matte", 1, polygon=_rect(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), matting="chroma")
    tr = {"z_01": {f: np.eye(2, 3, dtype=np.float32) for f in frames}}
    cache_for = lambda zid: ZoneCache(tmp_path / "obj", zid)
    r1 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, cache_for=cache_for, plate_sig=lambda f: ["p", f])
    assert r1.stats["z_01"]["computed"] == 3 and r1.stats["z_01"]["cached"] == 0
    r2 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z], tr, cache_for=cache_for, plate_sig=lambda f: ["p", f])
    assert r2.stats["z_01"]["computed"] == 0 and r2.stats["z_01"]["cached"] == 3
    for f in frames:   # same result (cache holds the per-frame alpha; the temporal median is re-applied identically)
        assert np.abs(r2.soft[f][0] - r1.soft[f][0]).max() < 2.0 / 65535
    # a changed core in one frame -> only that frame recomputed
    core2 = core.copy(); core2[0:5, 0:5] = True
    r3 = apply_zones(frames, {1: core, 2: core2, 3: core}, lambda f: img, [z], tr, cache_for=cache_for, plate_sig=lambda f: ["p", f])
    assert r3.stats["z_01"]["computed"] == 1 and r3.stats["z_01"]["cached"] == 2
    # the sequence engine caches the whole range under one hash
    calls = []
    z2 = Zone("z_02", "add", "matte", 1, polygon=z.polygon, matting="matanyone")

    def ma_run(seg, sf, core_, keys_, start, scale, tag):
        calls.append(1)
        return {f: np.ones((seg.side, seg.side), np.float32) * 0.5 for f in sf}
    r4 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z2], {"z_02": tr["z_01"]}, ma_run=ma_run,
                     cache_for=cache_for, plate_sig=lambda f: ["p", f])
    r5 = apply_zones(frames, {f: core for f in frames}, lambda f: img, [z2], {"z_02": tr["z_01"]}, ma_run=ma_run,
                     cache_for=cache_for, plate_sig=lambda f: ["p", f])
    assert len(calls) == 1 and r4.stats["z_02"]["computed"] == 3 and r5.stats["z_02"]["cached"] == 3
    for f in frames:
        assert np.abs(r5.soft[f][0] - r4.soft[f][0]).max() < 2.0 / 65535


def test_api_keys_region_apply_and_cli(tmp_path, monkeypatch):
    import md
    from test_md_cli import TCHttp, _run
    from test_zone_matte import _api_project
    from engine.export import run_alpha_loader
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=8)
    cx, cy, s = shot.centers[start]
    poly = _rect(cx - 60, cy - 260, cx + 60, cy - 120)
    z = {"mode": "add", "criterion": "matte", "frame": start, "polygon": poly, "matting": "chroma", "preset": "hair"}
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json=z)
    assert r.status_code == 200, r.text
    zid = r.json()["zone"]["zone_id"]
    # apply before any finalize -> 409 (no version)
    assert api.post(f"/api/object/obj/zones/{zid}/apply").status_code == 409
    # finalize (core edge mode, mock Comfy) -> v001 with alpha_base
    fz = api.post("/api/object/obj/finalize", json={"edge_mode": "core"})
    assert fz.status_code == 200, fz.text
    v1 = fz.json()["run"]
    assert (pipe.store.dir("obj") / "runs" / v1 / "alpha_base").exists()
    # "Zastosuj" right after "Przelicz finalnie" (nothing changed): 0 frames computed, the version is bitwise identical
    ap0 = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert ap0.status_code == 200 and ap0.json()["computed"] == 0 and ap0.json()["cached"] == len(frames), ap0.text
    v1b = ap0.json()["run"]
    l1, l1b = (run_alpha_loader(pipe.store.dir("obj"), v, shot.w, shot.h) for v in (v1, v1b))
    for f in frames:
        assert np.array_equal(l1(f), l1b(f)), f
    # keys: add a second key, list, region, delete
    f2 = frames[-1]
    r = api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx - 40, cy - 240, cx + 80, cy - 100)})
    assert r.status_code == 200 and r.json()["keys"] == sorted({start, f2}), r.text
    assert api.get(f"/api/object/obj/zones/{zid}/keys").json()["base"] == start
    mid = next(f for f in frames if f not in (start, f2))
    reg = api.get(f"/api/object/obj/zones/{zid}/region/{mid}")
    assert reg.status_code == 200 and reg.json()["area_px"] > 0 and reg.json()["keys"] == sorted({start, f2}), reg.text
    assert reg.json()["bbox"] is not None and not reg.json()["is_key"]
    assert api.put(f"/api/object/obj/zones/{zid}/keys/{mid}", json={"polygon": []}).status_code == 400
    # apply -> new version, outside the zone bitwise = v1 (same base), cache: second apply = 0 computed
    ap = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert ap.status_code == 200, ap.text
    v2 = ap.json()["run"]
    assert ap.json()["computed"] == len(frames) and ap.json()["cached"] == 0
    ap2 = api.post(f"/api/object/obj/zones/{zid}/apply")
    assert ap2.status_code == 200 and ap2.json()["computed"] == 0 and ap2.json()["cached"] == len(frames), ap2.text
    v3 = ap2.json()["run"]
    la, lb, lc = (run_alpha_loader(pipe.store.dir("obj"), v, shot.w, shot.h) for v in (v1, v2, v3))
    w0 = min(p[0] for p in poly) - 80; w1 = max(p[0] for p in poly) + 100
    for f in frames:
        a1, a2, a3 = la(f), lb(f), lc(f)
        assert np.array_equal(a2, a3)                                     # cache: identical version
        outside = np.ones_like(a1, bool); outside[:, int(w0):int(w1)] = False
        assert np.array_equal(a1[outside], a2[outside])                   # outside the zone: bitwise the base
    rj = json.loads((pipe.store.dir("obj") / "runs" / v3 / "run.json").read_text(encoding="utf-8"))
    assert rj["applied_from"] == v2 and zid in rj["zones"] and (pipe.store.dir("obj") / "runs" / v3 / "qa.json").exists()
    assert pipe.store.get("obj")["tool"]["active_run"] == v3
    # delete the extra key; the base key cannot be deleted when it is the last
    assert api.delete(f"/api/object/obj/zones/{zid}/keys/{f2}").status_code == 200
    assert api.delete(f"/api/object/obj/zones/{zid}/keys/{start}").status_code == 409
    # CLI
    draw = tmp_path / "key.json"; draw.write_text(json.dumps({"polygon": _rect(cx - 50, cy - 250, cx + 70, cy - 110)}), encoding="utf-8")
    code, out = _run(["zone-keys", "obj", zid, "--set", str(f2), "--draw", str(draw)], TCHttp(api))
    assert code == 0 and f"F{f2} zapisany" in out, out
    code, out = _run(["zone-keys", "obj", zid], TCHttp(api))
    assert code == 0 and f"klucz bazowy F{start}" in out
    code, out = _run(["zone-keys", "obj", zid, "--region", str(mid)], TCHttp(api))
    assert code == 0 and "Obszar strefy" in out
    code, out = _run(["zone-apply", "obj", zid, "--wait"], TCHttp(api))
    assert code == 0 and "Zastosuj" in out and "wersja v" in out, out
    code, out = _run(["zone-keys", "obj", zid, "--delete", str(f2)], TCHttp(api))
    assert code == 0 and "usunięty" in out
