# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E1e – depth map (SPEC 4e). Cloud: synthetic plates + mock Comfy that plays DepthCrafter.

Acceptance (spec 12): workflow run with the user's parameters (20 / 0.5 / 60 / 20 – checked in the sent graph);
QA static flicker down vs raw depth; depth_order suggestion matches the reference order for visible objects;
EXR: `depth.disparity`, full resolution, plate frame numbers; Group: disparity / Z switch, output to depth.Z.
Real DepthCrafter on SH010 + Nuke check: LOCAL_CHECKS.md.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.comfy_graph import find_nodes, ui_to_api
from engine.depth import (DEPTHCRAFTER_USER_PARAMS, build_depthcrafter_graph, frame_path, prepare_input,
                          read_depth, run_depth, static_segments)
from engine.exr_io import read_exr
from engine.nuke_nk import validate_nk
from mock_comfy import MockComfyClient

WF = Path(__file__).resolve().parent.parent / "handoff" / "comfy" / "workflows" / "workflow_comfyui_depthcrafter.json"
F0, N, W, H = 1001, 40, 320, 240
PAN_FROM = 25  # camera static for frames 0..24, then pans


def _scene(tmp_path):
    rng = np.random.default_rng(7)
    tex = cv2.GaussianBlur(rng.random((H, W + 60, 3)).astype(np.float32), (0, 0), 2.0) * 0.6 + 0.2
    plates, truth, masks_near, masks_far, Hs = [], [], [], [], []
    for t in range(N):
        ox = 0 if t < PAN_FROM else 2 * (t - PAN_FROM + 1)
        img = tex[:, ox:ox + W].copy()
        disp = np.full((H, W), 0.1, np.float32)
        far = np.zeros((H, W), bool)
        far[40:120, 200 - ox:260 - ox] = True
        near = np.zeros((H, W), bool)
        nx = 30 + 3 * t
        near[130:200, nx % (W - 60):nx % (W - 60) + 50] = True
        img[far] = 0.45
        disp[far] = 0.3
        img[near] = 0.9
        disp[near] = 0.8
        plates.append(img)
        truth.append(disp)
        masks_near.append(near)
        masks_far.append(far)
        Hs.append(np.array([[1, 0, ox], [0, 1, 0], [0, 0, 1]], np.float64))  # frame -> ref
    pdir = tmp_path / "plates"
    pdir.mkdir()
    from engine.exr_io import write_gray_exr  # noqa: F401 (ensures module import ok)
    for t, img in enumerate(plates):
        cv2.imwrite(str(pdir / f"shot_{F0 + t:04d}.exr"), img)
    return str(pdir / "shot_####.exr"), truth, np.array(masks_near), np.array(masks_far), Hs


def _depthcrafter_handler(truth):
    rng = np.random.default_rng(1)

    def run(graph):
        _, load = find_nodes(graph, "PlateLoadEXR")[0]
        _, save = find_nodes(graph, "PlateSaveEXR")[0]
        first, last = load["inputs"]["first_frame"], load["inputs"]["last_frame"]
        for f in range(first, last + 1):
            assert frame_path(load["inputs"]["path_pattern"], f).exists()
            d = truth[f - first] * (1.0 + 0.04 * np.sin(f * 1.7)) + rng.normal(0, 0.01, truth[0].shape)
            out = frame_path(save["inputs"]["out_pattern"], f)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.dstack([d, d, d]).astype(np.float32))  # PlateSaveEXR: RGB(A) EXR
        return {}
    return run


def test_ui_to_api_keeps_only_connected_nodes_and_user_params():
    api = ui_to_api(json.loads(WF.read_text(encoding="utf-8")))
    classes = sorted(v["class_type"] for v in api.values())
    assert "VHS_LoadVideo" not in classes  # not connected to the saver
    assert classes.count("ImageScaleBy") == 2 and "DepthCrafter" in classes
    (_, dc), = find_nodes(api, "DepthCrafter")
    assert dc["inputs"]["depthcrafter_model"][1] == 0 and isinstance(dc["inputs"]["images"], list)


def test_graph_patches_only_paths_and_range():
    api = build_depthcrafter_graph(WF, "D:/x/in_####.exr", 1001, 1193, "D:/x/out_####.exr")
    (_, dc), = find_nodes(api, "DepthCrafter")
    for k, v in DEPTHCRAFTER_USER_PARAMS.items():
        assert dc["inputs"][k] == v
    (_, ld), = find_nodes(api, "PlateLoadEXR")
    assert (ld["inputs"]["path_pattern"], ld["inputs"]["first_frame"], ld["inputs"]["last_frame"]) == ("D:/x/in_####.exr", 1001, 1193)
    assert ld["inputs"]["scale"] == 0.5  # user's value untouched
    (_, sv), = find_nodes(api, "PlateSaveEXR")
    assert sv["inputs"]["out_pattern"] == "D:/x/out_####.exr" and sv["inputs"]["half_float"] is True


def test_static_segments_from_camera():
    Hs = [np.eye(3)] * 30 + [np.array([[1, 0, 3.0 * k], [0, 1, 0], [0, 0, 1]]) for k in range(1, 11)]
    assert static_segments(Hs, 320, 240) == [(0, 29)]


def test_e1e_acceptance_pipeline(tmp_path):
    plate_pat, truth, near, far, Hs = _scene(tmp_path)
    client = MockComfyClient(handlers={"DepthCrafter": _depthcrafter_handler(truth)})
    frames = list(range(F0, F0 + N))
    work, export = tmp_path / "work" / "matte_desk", tmp_path / "work" / "matte_desk" / "export"
    rep = run_depth(client, WF, "SHOT", plate_pat, frames, W, H, work, export, homographies=Hs,
                    object_masks={"near_obj": near, "far_obj": far})

    # 1. sent graph carries the user's parameters
    (_, dc), = find_nodes(client.sent[0], "DepthCrafter")
    assert {k: dc["inputs"][k] for k in DEPTHCRAFTER_USER_PARAMS} == DEPTHCRAFTER_USER_PARAMS
    assert rep.params["num_inference_steps"] == 20 and client.freed == 1

    # 2. QA: static flicker down vs raw depth
    assert rep.static_segments == [(F0, F0 + PAN_FROM - 1)]
    assert rep.qa_stabilized["accel_static"] < 0.5 * rep.qa_raw["accel_static"], (rep.qa_raw, rep.qa_stabilized)
    assert (work / "depth" / "qa" / "accel_stabilized.png").exists()

    # 3. depth_order hint = reference order (near object first)
    order = {r["name"]: r["depth_order"] for r in rep.depth_order_hint}
    assert order == {"near_obj": 0, "far_obj": 1}

    # 4. EXR: depth.disparity, full resolution, plate frame numbers, float
    f = frame_path(rep.export_pattern, F0 + 10)
    ch, hdr = read_exr(f)
    assert "depth.disparity" in ch and "R" in ch
    assert ch["depth.disparity"].shape == (H, W)
    assert str(hdr["channels"]["depth.disparity"].type) == "FLOAT"
    assert not frame_path(rep.export_pattern, F0 - 1).exists() and frame_path(rep.export_pattern, F0 + N - 1).exists()
    # moving object keeps its own (per-frame) disparity – not replaced by the static median
    d10 = ch["depth.disparity"]
    assert abs(float(np.median(d10[near[10]])) - 0.8) < 0.08

    # 5. Group: switch disparity / Z-like, output into depth.Z
    nk = Path(rep.nk_path).read_text(encoding="utf-8")
    assert validate_nk(nk, "Depth_SHOT") == []
    assert "depth_mode" in nk and "to0 depth.Z" in nk and "near" in nk and "far" in nk

    # 6. outputs.json (data contract)
    out = json.loads((work / "outputs.json").read_text(encoding="utf-8"))
    assert out["depth"]["kind"] == "disparity" and out["depth"]["channel"] == "depth.disparity"


def test_prepare_input_applies_gain_maps_like_the_group(tmp_path):
    plate_pat, _, _, _, _ = _scene(tmp_path)
    gdir = tmp_path / "gain"
    gdir.mkdir()
    for f in (F0, F0 + 1):
        cv2.imwrite(str(gdir / f"gain_{f:04d}.exr"), np.full((H // 2, W // 2, 3), 1.25, np.float32))
    pat = prepare_input(plate_pat, [F0, F0 + 1], tmp_path / "depth_in", str(gdir / "gain_%04d.exr"))
    from engine.exr_io import read_rgb
    a, b = read_rgb(frame_path(plate_pat, F0)), read_rgb(frame_path(pat, F0))
    assert np.allclose(b, a * 1.25, atol=2e-3)


def test_busy_comfy_is_not_interrupted(tmp_path):
    plate_pat, truth, _, _, Hs = _scene(tmp_path)
    client = MockComfyClient(busy=True)
    with pytest.raises(RuntimeError, match="zajęte"):
        run_depth(client, WF, "S", plate_pat, list(range(F0, F0 + N)), W, H, tmp_path / "w", tmp_path / "e")
    assert client.sent == []


def test_api_depth_compute_with_mock_comfy(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app as app_module

    plate_pat, truth, _, _, _ = _scene(tmp_path)
    work = tmp_path / "shot"
    work.mkdir()
    mf = work / "shot_manifest_SHOT.json"
    mf.write_text(json.dumps({"shot": "SHOT", "first": F0, "last": F0 + N - 1, "plate_w": W, "plate_h": H,
                              "work": str(work), "plates": {"beauty": plate_pat}}), encoding="utf-8")
    client = TestClient(app_module.app)
    assert client.post("/api/project/open", json={"manifest_path": str(mf)}).status_code == 200
    monkeypatch.setitem(app_module.STATE, "comfy_client", MockComfyClient(handlers={"DepthCrafter": _depthcrafter_handler(truth)}))
    r = client.post("/api/depth/compute", json={"workflow_path": str(WF)})
    assert r.status_code == 200, r.text
    assert r.json()["report"]["params"]["window_size"] == 60
    assert client.get("/api/depth/report").json()["exists"] is True
    assert "to0 depth.Z" in client.post("/api/depth/export_nk").json()["script"]
