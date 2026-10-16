# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E3b – remaining edge modes: core (antialiased core), backlit (BacklitCoverage limited by the core),
ltx (user's LTX Alpha Gen workflow, 8n+1 batches, region), tiles (5.2 pkt 5), Auto choice + "Krawędź" list (5.5).

Acceptance (spec 12): backlit on SH010 trees = current comfy_mattes/tree_* (IoU >= 0.98, station);
core on a rigid object: edge flicker < MatAnyone; ltx on Figure's horns: tips present in >= 90 % of frames.
Cloud: synthetic puppet / backlit / rigid shots with mocks emulating the node chains.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.crop import choose_instance, plan_segments, run_crop_engine, save_crops, smooth_bboxes
from engine.edge_mode import choose_edge_mode, edge_flicker, routing_metrics
from engine.pipeline import ObjectPipeline
from engine.refine import ltx_batches, ltx_side
from engine.segment import CorePropagator, iou
from mock_comfy import MockComfyClient, MockOutputs
from synth import (LTX_UI_WORKFLOW, backlit_handlers, display, ltx_handlers, make_shot, matanyone_handlers,
                   puppet_mask, sam3_handlers, write_backlit)


class StatsClient(MockComfyClient):
    def get_system_stats(self):
        return {"devices": [{"torch_vram_total": 12 * 2 ** 30, "torch_vram_free": 6 * 2 ** 30}]}


def _pipe(tmp_path, shot, extra_handlers, flags=(), prompt="felt puppet with two long antennae", **kw):
    outs = MockOutputs(tmp_path / "comfy_out")
    client = StatsClient(handlers={**sam3_handlers(outs), **extra_handlers})
    client.output_root = outs.root
    pipe = ObjectPipeline(tmp_path / "work" / "matte_desk", shot.plate_pattern, shot.w, shot.h, client,
                          display=display, sam3_ckpt="sam3.st", use_mplate=False, **kw)
    pipe.store.create("obj", [prompt], flags=list(flags))
    frames = list(range(shot.first, shot.last + 1))
    start = frames[len(frames) // 2]
    cands = choose_instance(pipe.runner, pipe.loader.frame(start), prompt=prompt)
    pipe.store.save_key("obj", start, cands[0] > 0.5, author="auto")
    pipe.store.update("obj", tool={"start_frame": start})
    res = run_crop_engine(pipe.runner, pipe.loader.frame, frames, start, cands[0], 1.0, shot.w, shot.h,
                          thin_parts="thin_parts" in flags)
    save_crops(pipe.store.dir("obj") / "crops.json", res.segments)
    pipe.run_core("obj", frames)
    return pipe, client, frames


def test_backlit_mode_reproduces_physical_matte(tmp_path):
    shot = make_shot(tmp_path, n=16)
    bl = write_backlit(shot, tmp_path)
    pipe, client, frames = _pipe(tmp_path, shot, backlit_handlers(), backlit_pattern=bl)
    choice = pipe.auto_edge_mode("obj", frames)
    assert choice["mode"] == "backlit"
    full, rep, _ = pipe.finalize("obj", frames)
    assert rep.edge_mode == "backlit"
    ious = [iou(full[f] > 0.5, shot.masks[f] > 0.5) for f in frames]
    assert min(ious) >= 0.98, ious
    # soft values follow the physical coverage (not a binary core)
    f = frames[8]
    edge = (shot.masks[f] > 0.1) & (shot.masks[f] < 0.9)
    assert np.abs(full[f][edge] - shot.masks[f][edge]).mean() < 0.1


def _rigid_shot(root, n=12):
    """Static rigid box (sharp edges) on texture."""
    from synth import PUPPET_BGR, SynthShot
    rng = np.random.default_rng(2)
    w, h = 640, 480
    tex = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3.0) * 0.35 + 0.25
    big = np.zeros((h * 4, w * 4), np.uint8)
    box = cv2.boxPoints(((320 * 4, 240 * 4), (200 * 4, 180 * 4), 7.0)).astype(np.int32)
    cv2.fillPoly(big, [box], 255)
    a = cv2.resize(big.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)  # antialiased edge
    pdir = Path(root) / "plates"
    pdir.mkdir(parents=True, exist_ok=True)
    masks = {}
    for i in range(n):
        f = 1001 + i
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        cv2.imwrite(str(pdir / f"r_{f:04d}.exr"), ((img ** 2.2) * 1.5).astype(np.float32))
        masks[f] = a
    return SynthShot(str(pdir / "r_####.exr"), 1001, 1000 + n, w, h, masks, np.zeros((h, w), bool), {})


def _noisy_matanyone(seed=0):
    """MatAnyone stand-in with per-frame edge noise (what made MatAnyone flicker on rigid props, MBA chair)."""
    base = matanyone_handlers()["MatAnyone2RunKeyframes"]
    rng = np.random.default_rng(seed)

    def run(graph):
        base(graph)
        from engine.comfy_graph import find_nodes
        from synth import frame_path
        (_, sv), = find_nodes(graph, "PlateSaveEXR")
        (_, ld), = find_nodes(graph, "MatAnyone2RunKeyframes")
        img_node = graph[ld["inputs"]["images"][0]]["inputs"]
        for i in range(img_node["first_frame"], img_node["last_frame"] + 1):
            p = frame_path(sv["inputs"]["out_pattern"], i)
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., 3]
            edge = (a > 0.02) & (a < 0.98)
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            a = np.clip(a + edge * rng.normal(0, 0.15, a.shape), 0, 1).astype(np.float32)
            cv2.imwrite(str(p), np.dstack([a, a, a, a]))
        return {}
    return {"MatAnyone2RunKeyframes": run}


@pytest.mark.parametrize("dark_behind", [False, True])
def test_auto_skips_backlit_when_object_not_on_it(tmp_path, dark_behind):
    """SH010: the backlit exposure X2 shows only the trees – Figure is absent, and what is behind her (nest, trees) is
    dark too. Auto must not pick backlit for her (station: backlit chosen twice -> matte = grown core)."""
    shot = make_shot(tmp_path, n=16)
    bl = write_backlit(shot, tmp_path, with_object=False, dark_behind=dark_behind)
    pipe, client, frames = _pipe(tmp_path, shot, {**backlit_handlers(), **matanyone_handlers()}, backlit_pattern=bl)
    choice = pipe.auto_edge_mode("obj", frames)
    assert choice["mode"] != "backlit", choice
    assert choice["metrics"]["backlit_cover"] < 0.2 and "nie zasłania" in choice["reason"]


def test_core_mode_flickers_less_than_matanyone_on_rigid(tmp_path):
    shot = _rigid_shot(tmp_path)
    pipe, client, frames = _pipe(tmp_path, shot, _noisy_matanyone(), flags=["rigid"], prompt="red box")
    assert pipe.auto_edge_mode("obj", frames)["mode"] == "core"
    core = pipe.load_core("obj", frames)
    f_core, _, _ = pipe.finalize("obj", frames, edge_mode="core")
    f_ma, _, _ = pipe.finalize("obj", frames, edge_mode="matanyone")
    fl_core = edge_flicker([f_core[f] for f in frames], [core[f] for f in frames])
    fl_ma = edge_flicker([f_ma[f] for f in frames], [core[f] for f in frames])
    assert fl_core < fl_ma, (fl_core, fl_ma)
    assert min(iou(f_core[f] > 0.5, shot.masks[f] > 0.5) for f in frames) >= 0.98


def test_ltx_mode_keeps_thin_tips(tmp_path):
    shot = make_shot(tmp_path, n=20)
    seen = []
    pipe, client, frames = _pipe(tmp_path, shot, ltx_handlers(seen=seen), flags=["semi_transparent", "thin_parts"],
                                 ltx_workflow=LTX_UI_WORKFLOW)
    # 2026-10-16: LTX is its own stage (engine/ltx_alpha.py) – Auto never picks it; the old user-workflow edge mode
    # still runs when asked explicitly (objects saved with edge_mode "ltx")
    assert pipe.auto_edge_mode("obj", frames)["mode"] != "ltx"
    full, rep, _ = pipe.finalize("obj", frames, edge_mode="ltx")
    assert all((n - 1) % 8 == 0 for n, _ in seen) and all(s % 32 == 0 and s <= 1088 for _, s in seen)
    # antenna tips: the last 6 px of each antenna (GT) covered by alpha > 0.5
    present = 0
    for f in frames:
        cx, cy, s = shot.centers[f]
        head = (cx, cy - 75 * s)
        tips = [(head[0] + dx * 45 * s * 0.95, head[1] - 70 * s * 0.95) for dx in (-1, 1)]
        ok = all(full[f][int(round(ty)) - 1:int(round(ty)) + 2, int(round(tx)) - 1:int(round(tx)) + 2].max() > 0.5
                 for tx, ty in tips if 0 <= tx < shot.w and 0 <= ty < shot.h)
        present += ok
    assert present >= 0.9 * len(frames), present


def test_ltx_batches_and_side():
    assert ltx_batches(25) == [(0, 24)]
    b = ltx_batches(60, 25)
    assert all((e - s + 1) == 25 for s, e in b) and b[-1][1] == 59
    assert all(b[i][1] - b[i + 1][0] + 1 >= 8 for i in range(len(b) - 1))
    assert ltx_side(1500) == 1088 and ltx_side(530) == 512


def test_tiles_for_big_object_with_thin_parts(tmp_path):
    """Object > 2 x 1008 px with thin parts -> crop split into 1008-px tiles; tracking per tile, cos^2 blend."""
    from synth import PUPPET_BGR, SynthShot
    w, h, n = 2600, 2400, 4
    rng = np.random.default_rng(4)
    tex = cv2.GaussianBlur(rng.random((h // 4, w // 4, 3)).astype(np.float32), (0, 0), 2.0) * 0.35 + 0.25
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)
    pdir = tmp_path / "plates"
    pdir.mkdir()
    masks = {}
    for i in range(n):
        f = 1001 + i
        a = puppet_mask(w, h, 1300 + 5 * i, 1500, 9.0, ss=1)
        img = tex * (1 - a[..., None]) + PUPPET_BGR * a[..., None]
        cv2.imwrite(str(pdir / f"b_{f:04d}.exr"), ((img ** 2.2) * 1.5).astype(np.float32))
        masks[f] = a
    shot = SynthShot(str(pdir / "b_####.exr"), 1001, 1000 + n, w, h, masks, np.zeros((h, w), bool), {})
    pipe, client, frames = _pipe(tmp_path, shot, {}, flags=["thin_parts"])
    segs = pipe.segments("obj")
    assert segs[0].side > 2 * 1008 and segs[0].tiles
    core = pipe.load_core("obj", frames)
    assert min(iou(core[f], masks[f] > 0.5) for f in frames) >= 0.97
    n_track_graphs = sum(any(v["class_type"] == "SAM3_VideoTrack" for v in g.values()) for g in client.sent)
    assert n_track_graphs > len(segs)  # several tiles tracked


def test_auto_routing_rules_and_metrics():
    obj = lambda **k: {"flags": k.get("flags", []), "detect": {"prompts": [k.get("p", "thing")]}, "use_backlit": True}
    assert choose_edge_mode(obj(), {"soft_frac": 0.1}, True, False)["mode"] == "backlit"
    assert choose_edge_mode(obj(), {"soft_frac": 0.1, "backlit_cover": 0.9}, True, False)["mode"] == "backlit"
    assert choose_edge_mode(obj(), {"soft_frac": 0.1, "backlit_cover": 0.05}, True, False)["mode"] == "core"
    assert choose_edge_mode(obj(flags=["volumetric"]), {}, False, True)["mode"] == "ltx"
    assert choose_edge_mode(obj(flags=["volumetric"]), {"soft_frac": 0.4}, False, False)["mode"] == "matanyone"
    assert choose_edge_mode(obj(p="woman in red coat"), {"soft_frac": 0.0}, False, False)["mode"] == "matanyone"
    c = choose_edge_mode(obj(), {"soft_frac": 0.31}, False, False)
    assert c["mode"] == "matanyone" and "31%" in c["reason"] and not c["borderline"]
    assert choose_edge_mode(obj(), {"soft_frac": 0.26}, False, False)["borderline"]
    # metrics: blurred edge -> soft, sharp edge -> hard
    core = np.zeros((200, 200), bool)
    core[50:150, 50:150] = True
    sharp = np.where(core, 0.9, 0.1).astype(np.float32)
    soft = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    assert routing_metrics([core], [sharp])["soft_frac"] < 0.1
    assert routing_metrics([core], [soft])["soft_frac"] > 0.8


def test_api_edge_modes_list(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module
    shot = make_shot(tmp_path, n=4)
    work = tmp_path / "w"
    work.mkdir()
    mf = work / "shot_manifest_S.json"
    mf.write_text(json.dumps({"shot": "S", "first": shot.first, "last": shot.last, "plate_w": shot.w, "plate_h": shot.h,
                              "work": str(work), "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    api.post("/api/object", json={"name": "o", "prompts": ["x"]})
    d = api.get("/api/object/o/edge_modes").json()
    modes = {m["mode"]: m for m in d["modes"]}
    assert "backlit" not in modes and "ltx" not in modes        # LTX: stage 6 since 2026-10-16, not an edge mode
    assert api.put("/api/object/o/edge_mode", json={"mode": "ltx"}).status_code == 400
    assert api.put("/api/object/o/edge_mode", json={"mode": "core"}).status_code == 200
    assert api.get("/api/object/o/edge_modes").json()["current"] == "core"


def _run_vs_reference(obj_dir: Path, ref_dir: Path):
    import os
    from engine.crop import CropSegment
    from engine.exr_io import read_exr
    from engine.refine import assemble_full, read_alpha
    obj = json.loads((obj_dir / "object.json").read_text(encoding="utf-8"))
    run = obj_dir / "runs" / obj["tool"]["active_run"]
    rj = json.loads((run / "run.json").read_text(encoding="utf-8"))
    segs = [CropSegment.from_dict(d) for d in rj["segments_def"]]
    out = []
    for p in sorted(x for x in os.listdir(ref_dir) if x.lower().endswith(".exr")):
        f = int(p.rsplit("_", 1)[-1].split(".")[0])
        ch, _ = read_exr(ref_dir / p)
        ref = ch.get("A", ch.get("R"))
        per = {s.seg_id: {f: read_alpha(run / "alpha" / s.seg_id / f"a_{f:04d}.exr")} for s in segs if s.first <= f <= s.last}
        ours = assemble_full(per, [s for s in segs if s.seg_id in per], [f], ref.shape[1], ref.shape[0])[f]
        out.append((f, ours, ref))
    return rj, out


@pytest.mark.parametrize("tree", ["tree_FG", "tree_M", "tree_L", "tree_RB"])
def test_e3b_real_backlit_trees(tree):
    """Station (D:/MD_REAL_PROJECT): object `<tree>` computed in the UI with edge mode Backlit -> IoU >= 0.98 with
    render/comfy_mattes/<tree> (made by the same BacklitCoverage chain + MatteSplitByRegions; until E6 the split
    between trees is approximated by MaskRegionLimit on each tree's core)."""
    work = Path("C:\\shots\\SH010\\frame_####.exr")
    obj_dir, ref = work / "matte_desk" / "objects" / tree, work / "render" / "comfy_mattes" / tree
    assert obj_dir.exists() and ref.exists(), f"compute object {tree} (Backlit) in the UI first"
    rj, rows = _run_vs_reference(obj_dir, ref)
    assert rj["edge_mode"] == "backlit"
    ious = [iou(o > 0.5, r > 0.5) for _, o, r in rows]
    print(tree, "IoU min", min(ious), "mean", np.mean(ious))
    assert min(ious) >= 0.98
