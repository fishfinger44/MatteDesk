# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tools/ab_zone_smooth.py – A/B of the ViTMatte zone temporal smoothing `flow_mean` (plan docs/PLAN_EDGE_SPEED.md B1 / B2,
2026-10-28) on a real object, all variants on THE SAME raw zone alphas:

  A      = union bbox of the whole shot, DIS on the up-scaled stack (as the app today)
  B1     = temporal_crop "follow" (one-size crop following the zone frame by frame)
  B2     = temporal_flow_scale <s> (DIS on grays shrunk by s, brought back up)
  B1B2   = both
  none   = no smoothing (the raw alpha with the zone levels) – the flicker reference
  median / flow / mean [_rN] [_s1] = the palette's modes (mean = flow_mean as the app runs it since 2026-10-28)

The zone's region / raw per-frame alphas come from the normal code path (pipeline.apply_zones with the zone file
untouched – load_zones is wrapped in-process to give the zone `temporal: flow_mean`); zones.smooth_zone_alphas is wrapped
to run every variant on a copy of its inputs and time it, then the run stops (nothing else of the zone is computed).
Writes into the shot only what apply_zones writes anyway (zone cache) – run it on a test shot or a COPY.

Metrics (tools/_abmeasure.mc_alpha_flicker, one DIS flow on the plate shared by all variants): motion-compensated
flicker inside the zone region, split rim (≤ 6 px from the core edge) / strands (farther out); difference to A
(mean / p99 inside the region); smoothing wall time. Panels: alpha tiles A / B… / |B − A| ×4, whole bbox + a zoom where
B1B2 differs most.

  python tools/ab_zone_smooth.py <shot_dir> <obj> <zone> <out_dir> [--first F --last L] [--flow-scale 0.5]
         [--radius 3 --scale 2] [--variants A,B1,B2,B1B2,none]
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import engine.zones as Z  # noqa: E402
from _abmeasure import FlowCache, gray_loader, mc_alpha_flicker, open_shot, panel  # noqa: E402


class _Captured(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("obj")
    ap.add_argument("zone")
    ap.add_argument("out")
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--flow-scale", type=float, default=0.5)
    ap.add_argument("--variants", default="A,B1,B2,B1B2,none")
    ap.add_argument("--panels", default="", help="frames for the panels (default: 3 spread over the range)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = a.variants.split(",")
    pipe, m = open_shot(a.shot)
    name = a.obj
    obj = pipe.store.get(name)
    run = obj["tool"].get("active_run")
    from engine.workarea import run_frames
    frames = list(range(int(m.get("first", 1)), int(m.get("last", 1)) + 1))
    rf = run_frames(pipe.store.dir(name), run) if run else None
    if rf:
        frames = [f for f in frames if f in set(rf)]
    if a.first is not None:
        frames = [f for f in frames if f >= a.first]
    if a.last is not None:
        frames = [f for f in frames if f <= a.last]
    rj = json.loads((pipe.store.dir(name) / "runs" / run / "run.json").read_text(encoding="utf-8")) if run else {}
    keys = pipe.keys(name)
    if rj.get("group"):
        core, soft = pipe.group_core(name, frames), {}
    else:
        core, soft = pipe.core_with_soft(name, frames, keys, bool(rj.get("core_stabilize")))

    orig_load = Z.load_zones

    def load(obj_dir):
        zs = orig_load(obj_dir)
        for z in zs:
            if z.zone_id == a.zone:
                z.params = {**(z.params or {}), "temporal": "flow_mean", "temporal_radius": a.radius,
                            "temporal_scale": a.scale}
                z.enabled = True
            else:
                z.enabled = False
        return zs
    Z.load_zones = load
    import engine.pipeline as P
    if hasattr(P, "load_zones"):
        P.load_zones = load
    orig_smooth = Z.smooth_zone_alphas
    res = {}

    def capture(alphas, fl, bbox, regions, force, temporal, t_radius, t_scale, t_conf, fin, images, fg_sure=None,
                progress=None, crops=None):
        y0, y1, x0, x1 = bbox
        raw = {f: alphas[f].copy() for f in fl}
        res.update(fl=list(fl), bbox=bbox, regions={f: regions[f] for f in fl}, force={f: force[f] for f in fl},
                   core={f: core[f] for f in fl})
        for v in variants:
            al = {f: raw[f].copy() for f in fl}
            t0 = time.perf_counter()
            if v == "none":
                for f in fl:
                    b = al[f]
                    if "alpha_black" in fin or "alpha_white" in fin:
                        b = Z.levels_alpha(b, fin.get("alpha_black"), fin.get("alpha_white"))
                    b = b.copy()
                    b[~regions[f]] = 0.0
                    b[regions[f] & force[f]] = 1.0
                    al[f] = b
            elif v.split("_")[0] in ("median", "flow", "mean"):
                # a mode of the palette (2026-10-28, help texts): median | flow | mean (= flow_mean as the app runs it
                # now: follow crop + plate-res flow); _rN = radius, _s1 = no x2 up-scale
                mode = {"median": "median", "flow": "flow", "mean": "flow_mean"}[v.split("_")[0]]
                r = next((int(x[1:]) for x in v.split("_")[1:] if x.startswith("r")), t_radius)
                sc = 1.0 if "_s1" in v else t_scale
                tc = dict(t_conf)
                cr = None
                if mode == "flow_mean":
                    cr = Z.follow_crops(regions, fl)
                    if sc > 1.0:
                        tc["flow_scale"] = 1.0 / sc
                orig_smooth(al, fl, bbox, regions, force, mode, r, sc, tc if mode == "flow_mean" else {}, fin, images,
                            fg_sure=fg_sure, crops=cr)
                if mode != "flow_mean" and ("alpha_black" in fin or "alpha_white" in fin):
                    for f in fl:          # the finish applies the levels after median / flow (flow_mean: before)
                        b = Z.levels_alpha(al[f], fin.get("alpha_black"), fin.get("alpha_white")).copy()
                        b[~regions[f]] = 0.0
                        b[regions[f] & force[f]] = 1.0
                        al[f] = b
            else:
                tc = dict(t_conf)
                if "B2" in v:
                    tc["flow_scale"] = a.flow_scale
                cr = Z.follow_crops(regions, fl) if "B1" in v else None
                orig_smooth(al, fl, bbox, regions, force, "flow_mean", t_radius, t_scale, tc, fin, images,
                            fg_sure=fg_sure, crops=cr)
            dt = time.perf_counter() - t0
            res.setdefault("alpha", {})[v] = {f: al[f][y0:y1, x0:x1].astype(np.float32) for f in fl}
            res.setdefault("seconds", {})[v] = round(dt, 1)
            if v.startswith("B1"):
                c = next(iter(Z.follow_crops(regions, fl).values()))
                res["follow_px"] = (c[1] - c[0]) * (c[3] - c[2])
            print(f"{v}: {dt:.1f} s", flush=True)
        raise _Captured()
    Z.smooth_zone_alphas = capture
    t0 = time.perf_counter()
    try:
        pipe.apply_zones(name, frames, core, keys, probs=soft, base_alpha=None, kinds=("matte",))
        raise SystemExit("smoothing never reached (zone without frames?)")
    except _Captured:
        pass
    print(f"prep + variants {time.perf_counter() - t0:.0f} s", flush=True)
    fl, (y0, y1, x0, x1) = res["fl"], res["bbox"]
    cut = lambda mm: mm[y0:y1, x0:x1]
    region = {f: cut(res["regions"][f]) for f in fl}
    core_c = {f: cut(res["core"][f].astype(bool)) for f in fl}
    rim, strands = {}, {}
    for f in fl:
        d_out = cv2.distanceTransform((~core_c[f]).astype(np.uint8), cv2.DIST_L2, 5)
        d_in = cv2.distanceTransform(core_c[f].astype(np.uint8), cv2.DIST_L2, 5)
        near = np.where(core_c[f], d_in, d_out) <= 6
        rim[f] = region[f] & near
        strands[f] = region[f] & ~near & ~core_c[f]
    flow = FlowCache(gray_loader(pipe.loader.read), roi=(x0, y0, x1, y1))
    pad = lambda d: {f: _place(d[f], (y0, y1, x0, x1), res["regions"][fl[0]].shape) for f in d}
    report = {"shot": a.shot, "obj": name, "zone": a.zone, "frames": [fl[0], fl[-1], len(fl)],
              "radius": a.radius, "scale": a.scale, "flow_scale": a.flow_scale,
              "union_bbox_px": (y1 - y0) * (x1 - x0), "follow_crop_px": res.get("follow_px"),
              "seconds": res["seconds"], "variants": {}}
    for v in variants:
        al = pad(res["alpha"][v])
        rep = {"seconds": res["seconds"][v]}
        for key, msk in (("region", region), ("rim", rim), ("strands", strands)):
            rep[f"flicker_{key}"] = mc_alpha_flicker(al, fl, flow, region=pad(msk))["median"]
        if v != "A" and "A" in res["alpha"]:
            d = np.concatenate([np.abs(res["alpha"][v][f] - res["alpha"]["A"][f])[region[f]] for f in fl])
            rep["diff_vs_A_mean"] = round(float(d.mean()), 5)
            rep["diff_vs_A_p99"] = round(float(np.percentile(d, 99)), 4)
            rep["diff_vs_A_max"] = round(float(d.max()), 4)
        report["variants"][v] = rep
        print(v, rep, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    # panels
    pf = [int(x) for x in a.panels.split(",") if x] or [fl[len(fl) // 6], fl[len(fl) // 2], fl[5 * len(fl) // 6]]
    pv = [v for v in variants if v != "none"] + (["none"] if "none" in variants else [])
    for f in pf:
        if f not in fl:
            continue
        g = cut(pipe.loader.read(f))
        tiles, labels = [], []
        for v in pv:
            tiles.append(_comp(g, res["alpha"][v][f]))
            labels.append(f"{v} F{f}")
        for v in pv:
            if v not in ("A", "none") and "A" in res["alpha"]:
                tiles.append(np.clip(np.abs(res["alpha"][v][f] - res["alpha"]["A"][f]) * 4 * 255, 0, 255).astype(np.uint8))
                labels.append(f"|{v}-A| x4")
        cv2.imwrite(str(out / f"panel_F{f:04d}.jpg"), panel(tiles, labels, max_h=700), [cv2.IMWRITE_JPEG_QUALITY, 90])
        # zoom: 480 px around where the last B variant differs most from A (box mean 31 px)
        vz = next((v for v in ("B1B2", "B2", "B1") if v in res["alpha"]), None)
        if vz and "A" in res["alpha"]:
            d = cv2.blur(np.abs(res["alpha"][vz][f] - res["alpha"]["A"][f]), (31, 31))
            cy, cx = np.unravel_index(int(np.argmax(d)), d.shape)
        else:
            ys, xs = np.where(region[f])
            cy, cx = int(np.median(ys)), int(np.median(xs))
        hh, ww = g.shape[:2]
        s = min(480, hh, ww)
        zy0, zx0 = int(min(max(0, cy - s // 2), hh - s)), int(min(max(0, cx - s // 2), ww - s))
        zs = (slice(zy0, zy0 + s), slice(zx0, zx0 + s))
        zt = [_comp(g[zs], res["alpha"][v][f][zs]) for v in pv]
        zt += [cv2.resize(res["alpha"][v][f][zs], None, fx=1, fy=1) for v in pv]
        zt = [t if t.ndim == 3 else (np.clip(t, 0, 1) * 255).astype(np.uint8) for t in zt]
        cv2.imwrite(str(out / f"zoom_F{f:04d}.jpg"), panel(zt, [f"{v}" for v in pv] + [f"alpha {v}" for v in pv],
                                                            max_h=480), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(report, indent=1, ensure_ascii=False))


def _place(crop, box, shape):
    y0, y1, x0, x1 = box
    m = np.zeros(shape, crop.dtype)
    m[y0:y1, x0:x1] = crop
    return m


def _comp(gray_bgr, alpha):
    """Plate x alpha over a mid-grey checker-free red background (strands readable on any plate)."""
    img = np.clip(gray_bgr, 0, 1).astype(np.float32)
    al = np.clip(alpha, 0, 1)[..., None]
    bg = np.zeros_like(img)
    bg[..., 2] = 0.55
    return (np.clip(img * al + bg * (1 - al), 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    main()
