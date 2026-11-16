# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P11 (docs/PLAN_CLEANUP_3D.md): the export to Nuke (engine/cleanup3d/nuke.py, POST
/api/cleanup3d/export_nuke) on a synthetic camera / cards / keys.

What is checked here (synthetically – opening the script in Nuke is LOCAL_CHECKS.md): the axis conversion (a world
point projected by our OpenCV camera = the same point projected by the converted Nuke camera, with the model of
Nuke's camera maths in nuke.nuke_project), Euler ZXY round trip, the card placement (on the plane, normal toward the
cameras, covering what the camera sees), the Dissolve chain = the render's key_pair weights, the STMaps (inverse of
each other), the .nk text (balanced, animated XYZ knob syntax, frame range, node counts, every node's inputs wired as
intended – a stack simulation of the .nk push / inputs / set lines), the .chan file, the files on disk and the route.
"""
import os
import re
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import nuke_nk as NK  # noqa: E402
from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import nuke as NU  # noqa: E402
from engine.cleanup3d import render as RD  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402
from engine.cleanup3d.geom import card_plane  # noqa: E402
from tests.test_cleanup3d_render import FRAMES, GROUND_Y, H, KEY_FRAMES, W, make_camera, make_cards  # noqa: E402


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def _keys():
    return [T.Key("ground", f, [{"v": "v001", "ext": "png"}], "v001") for f in KEY_FRAMES] + \
           [T.Key("sky", f, [{"v": "v001", "ext": "png"}], "v001") for f in (1, 9)]


def _key_files(keys):
    return {(k.card, k.frame): f"D:\\keys\\{k.card}\\k_{k.frame:04d}\\v001.png" for k in keys}


# ----------------------------------------------------------------------------------------------- maths
def test_euler_zxy_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        Rm = _rand_rot(rng)
        assert np.allclose(NU.rot_zxy(*NU.euler_zxy(Rm)), Rm, atol=1e-9)
    # the order: rotate about Z first, then X, then Y
    assert np.allclose(NU.rot_zxy(10, 20, 30), NU.rot_zxy(0, 20, 0) @ NU.rot_zxy(10, 0, 0) @ NU.rot_zxy(0, 0, 30))


def test_unwrap_keeps_curves_continuous():
    assert NU.unwrap_deg([179.0, -170.0, 0.0], [-179.0, 175.0, 359.0]) == [181.0, -185.0, -1.0]


def test_axis_conversion_projects_like_our_camera():
    """A world point through our camera (OpenCV, X_cam = R X + t, y down) = through the Nuke camera of the export
    (WORLD @ X, Nuke pose + lens, y up, looking down -Z); principal point off-centre and fx != fy on purpose."""
    cam = make_camera()
    cam.K = np.array([[270.0, 0, 151.3], [0, 262.0, 108.7], [0, 0, 1]])
    lens = NU.lens_knobs(cam.K, W, H)
    fr, tr, rot = NU.camera_curves(cam)
    rng = np.random.default_rng(2)
    n_checked = 0
    for f, Tn, rn in zip(fr, tr, rot):
        X = cam.centre(f) + (cam.R[f].T @ np.array([0, 0, 1.0])) * 6 + rng.normal(size=(40, 3))
        for p in X:
            pc = cam.R[f] @ p + cam.t[f]
            if pc[2] <= 0.1:
                continue
            u, v = cam.K[0, 0] * pc[0] / pc[2] + cam.K[0, 2], cam.K[1, 1] * pc[1] / pc[2] + cam.K[1, 2]
            xn, yn = NU.nuke_project(NU.WORLD @ p, Tn, rn, lens, W, H)
            assert abs((xn - 0.5) - u) < 1e-6 and abs((H - 0.5 - yn) - v) < 1e-6
            n_checked += 1
    assert n_checked > 200
    # a point behind our camera is behind the Nuke camera too
    f = fr[0]
    behind = cam.centre(f) - cam.R[f].T @ np.array([0, 0, 1.0]) * 3
    assert NU.nuke_project(NU.WORLD @ behind, tr[0], rot[0], lens, W, H) is None


def test_known_pose_identity_camera():
    """Identity pose: the Nuke camera sits at the origin with no rotation except the 180 deg of the axis flips, which
    cancel (WORLD @ I @ F = I) – Nuke's default camera looking down -Z of a world that is ours turned about X."""
    T_, Rn = NU.nuke_pose(np.eye(3), np.zeros(3))
    assert np.allclose(T_, 0) and np.allclose(Rn, np.eye(3))
    T_, Rn = NU.nuke_pose(np.eye(3), np.array([0.0, 0.0, -5.0]))         # our camera at z = +5 (world)
    assert np.allclose(T_, [0, 0, -5]) and np.allclose(NU.euler_zxy(Rn), 0)
    lens = NU.lens_knobs(np.array([[1000.0, 0, 959.5], [0, 1000.0, 539.5], [0, 0, 1]]), 1920, 1080)
    assert lens["focal"] == pytest.approx(1000 * 24.576 / 1920) and np.allclose(lens["win_translate"], 0)


def test_card_geometry_lies_on_the_plane_and_covers_the_view():
    cam = make_camera()
    ground, sky = make_cards()
    g = NU.card_geometry(cam, ground)
    n, d = card_plane(cam, ground)
    Rc = NU.rot_zxy(*g["rotate"])
    assert np.allclose(Rc[:, 2], NU.WORLD @ n, atol=1e-9)              # local +Z = the normal toward the cameras
    our_c = NU.WORLD.T @ g["translate"]
    assert abs(our_c @ n + d) < 1e-9                                    # centre on the plane
    for f in FRAMES:                                                    # every hit of a frame's ground pixel is on the card
        xx, yy = np.meshgrid(np.linspace(0, W - 1, 15), np.linspace(H * 0.55, H - 1, 6))
        r = np.stack([(xx.ravel() - cam.K[0, 2]) / cam.K[0, 0], (yy.ravel() - cam.K[1, 2]) / cam.K[1, 1],
                      np.ones(xx.size)], 1) @ cam.R[f]
        C = cam.centre(f)
        s = -(C @ n + d) / (r @ n)
        P = C + r * s[:, None]
        loc = Rc.T @ (np.array([NU.WORLD @ p for p in P[s > 0]]) - g["translate"]).T
        assert np.all(np.abs(loc[0]) <= g["scaling"][0] / 2) and np.all(np.abs(loc[1]) <= g["scaling"][1] / 2)
        assert np.allclose(loc[2], 0, atol=1e-9)
    gs = NU.card_geometry(cam, sky)
    assert gs["kind"] == "sky" and gs["radius"] >= NU.SKY_FACTOR * NU.scene_scale(cam) - 1e-9


def test_dissolve_chain_equals_render_key_pair():
    keys = [T.Key("ground", f, [{"v": "v001", "ext": "png"}], "v001") for f in (3, 7, 12)]
    frames = list(range(1, 16))
    wts = NU.dissolve_weights([k.frame for k in keys], frames)
    for j, f in enumerate(frames):
        mix = np.eye(3)[0]                                              # chain D_i = (1 - a) D_{i-1} + a key_i
        for i, a in enumerate(wts, start=1):
            mix = (1 - a[j]) * mix + a[j] * np.eye(3)[i]
        want = np.zeros(3)
        for k, wk in RD.key_pair(f, keys):
            want[[3, 7, 12].index(k.frame)] += wk
        assert np.allclose(mix, want), f


def test_stmaps_are_inverse_of_each_other():
    K = np.array([[300.0, 0, 161.0], [0, 300.0, 98.0], [0, 0, 1]])
    dist = [-0.12, 0.03, 0.001, -0.002]
    und, dis = NU.stmaps(K, dist, W, H)
    assert und.shape == (H, W, 3) and und.dtype == np.float32
    for x, y in [(40, 30), (160, 100), (290, 170), (10, 190)]:
        s, t = dis[y, x, 2], dis[y, x, 1]                               # distorted plate pixel -> pinhole position
        xu, yu = s * W - 0.5, H - 0.5 - t * H
        mx = cv2.remap(und[..., 2], np.float32([[xu]]), np.float32([[yu]]), cv2.INTER_LINEAR)[0, 0]
        my = cv2.remap(und[..., 1], np.float32([[xu]]), np.float32([[yu]]), cv2.INTER_LINEAR)[0, 0]
        assert abs((mx * W - 0.5) - x) < 0.05 and abs((H - 0.5 - my * H) - y) < 0.05
    und0, dis0 = NU.stmaps(K, [0, 0, 0, 0], W, H)
    assert np.allclose(und0[5, 7, 2], 7.5 / W) and np.allclose(und0[5, 7, 1], (H - 5.5) / H) and np.allclose(und0, dis0)


# ----------------------------------------------------------------------------------------------- .nk text
def _parse(text):
    """Stack simulation of the script: {name: (class, [inputs], {knob: rest of line})}. Input 0 = the top of the stack
    (the last push), as Nuke writes scripts."""
    nodes, stack, varmap = {}, [], {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("push "):
            v = ln[5:].strip()
            stack.append(None if v == "0" else varmap[v[1:]])
        elif ln.startswith("set "):
            m = re.match(r"set (\S+) \[stack 0\]", ln)
            varmap[m.group(1)] = stack[-1]
        elif re.match(r"^[A-Za-z0-9]+ \{$", ln) and not ln.startswith("Root"):
            cls = ln.split()[0]
            body = {}
            i += 1
            while lines[i] != "}":
                k, _, rest = lines[i].strip().partition(" ")
                body[k] = rest
                i += 1
            n = int(body["inputs"])
            ins = stack[len(stack) - n:][::-1] if n else []
            del stack[len(stack) - n:]
            nodes[body["name"]] = (cls, ins, body)
            stack.append(body["name"])
        i += 1
    return nodes


def _script(**kw):
    cam = make_camera()
    cards = make_cards()
    keys = _keys()
    args = dict(key_files=_key_files(keys), plate_pattern="D:\\shot\\plates\\p_####.exr", nk_path="D:\\shot\\x.nk",
                patch={"pattern": "D:/shot/inpaint/v003/patch/p_####.png", "first": 1, "last": 9, "version": "v003"})
    args.update(kw)
    return cam, cards, keys, NU.build_script(cam, cards, keys, **args)


def test_script_structure_counts_and_wiring():
    cam, cards, keys, (text, rep) = _script()
    assert NK.validate_script(text) == []
    assert text.startswith("#! nuke -nx\nversion 14.0")
    assert re.search(r"\n first_frame 1\n last_frame 9\n", text)
    nodes = _parse(text)
    by = {}
    for name, (cls, _i, _b) in nodes.items():
        by[cls] = by.get(cls, 0) + 1
    nk = len(keys)
    assert by["Camera3"] == 1 + nk                                      # shot camera + one projector per key
    assert by["Card2"] == 1 and by["Sphere"] == 1
    assert by["Project3D2"] == nk and by["ScanlineRender"] == nk and by["ApplyMaterial"] == nk
    assert by["Dissolve"] == (len(KEY_FRAMES) - 1) + (2 - 1)
    assert by["Merge2"] == 2                                            # ground over sky, comp over plate
    assert "STMap" not in by                                            # no lens distortion in the synthetic camera
    assert rep["counts"]["keys"] == nk and rep["frames"] == [1, 9]
    # wiring
    for name, (cls, ins, body) in nodes.items():
        if cls == "ScanlineRender":
            assert ins[0] is None and nodes[ins[1]][0] == "ApplyMaterial" and ins[2] == "Camera_shot"
        if cls == "Project3D2":
            assert nodes[ins[0]][0] == "Shuffle" and nodes[ins[1]][0] == "Camera3" and ins[1].startswith("Proj_")
        if cls == "ApplyMaterial":
            assert nodes[ins[0]][0] in ("Card2", "Sphere") and nodes[ins[1]][0] == "Project3D2"
    # the text itself: last input pushed first, input 0 (the empty bg) right above the node
    assert "push $N_Camera_shot\npush $N_Material_ground_0001\npush 0\nScanlineRender {\n inputs 3\n" in text
    comp = nodes["Comp_cleanup3d"]
    assert comp[1][0] == "Read_plate" and comp[1][2] == "Read_removal_mask" and comp[2]["maskChannelInput"] == "rgba.alpha"
    assert nodes["Over_ground"][1][1].startswith("Dissolve_ground")     # ground (order 0) is A = in front
    assert nodes["Over_ground"][1][0].startswith("Dissolve_sky")
    # paths absolute with forward slashes
    assert 'file "D:/shot/plates/p_####.exr"' in text and "\\keys" not in text
    assert 'file "D:/keys/ground/k_0005/v001.png"' in text


def test_animated_knob_syntax_and_values():
    cam, cards, keys, (text, rep) = _script()
    nodes = _parse(text)
    sh = nodes["Camera_shot"][2]
    assert re.fullmatch(r"\{\{curve x1 [^{}]+\} \{curve x1 [^{}]+\} \{curve x1 [^{}]+\}\}", sh["translate"])
    assert "}} {{curve" not in text
    assert NK.vec_knob("t", [1, 2], [[1, 1], [0, 1], [2, 2]]) == "t {1.0000 {curve x1 0.0000 x2 1.0000} 2.0000}"
    fr, tr, rot = NU.camera_curves(cam)
    comps = re.findall(r"\{curve ([^{}]+)\}", sh["rotate"])
    for c, axis in zip(comps, range(3)):
        vals = dict((int(a[1:]), float(b)) for a, b in zip(c.split()[::2], c.split()[1::2]))
        assert sorted(vals) == FRAMES
        assert all(abs(vals[f] - r[axis]) < 1e-5 for f, r in zip(fr, rot))
    # the dissolve between ground keys 1 and 5 at frame 3 = smoothstep(0.5)
    dv = nodes["Dissolve_ground_1"][2]["which"]
    vals = dict((int(a[1:]), float(b)) for a, b in zip(dv.strip("{}").split()[1::2], dv.strip("{}").split()[2::2]))
    assert vals[1] == 0 and vals[3] == pytest.approx(0.5) and vals[5] == 1 and vals[9] == 1
    # a projector camera = constant pose of its key frame
    pc = nodes["Proj_ground_0005"][2]
    T5, R5 = NU.nuke_pose(cam.R[5], cam.t[5])
    assert np.allclose([float(v) for v in pc["translate"].strip("{}").split()], T5, atol=1e-6)


def test_distortion_and_extent_add_stmaps_and_masks():
    cam = make_camera()
    cam.dist = np.array([-0.1, 0.02, 0.0, 0.0])
    cards = make_cards()
    keys = _keys()
    ext = {"ground": {"file": "D:/x/ext.png", "frame": 1, "K": cam.K, "w": W * 2, "h": H * 2, "seam_px": 4.0}}
    text, rep = NU.build_script(cam, cards, keys, key_files=_key_files(keys), plate_pattern="D:/p_####.exr",
                                nk_path="D:/x.nk", stmap_files={"undistort": "D:/u.exr", "distort": "D:/d.exr"},
                                extents=ext)
    nodes = _parse(text)
    stm = [n for n, (c, _i, _b) in nodes.items() if c == "STMap"]
    assert len(stm) == len(keys) + 1                                    # every key undistorted + the comp distorted
    assert nodes["Distort_lens"][1][1] == "Read_stmap_distort"
    assert nodes["Extent_ground"][2]["operation"] == "mask" and nodes["Extent_ground"][1][1] == "Seam_ground"
    assert rep["counts"]["extents"] == 1
    assert "Read_removal_mask" not in nodes and rep["notes"]            # no projection yet: said, no mask


def test_chan_file():
    cam = make_camera()
    txt = NU.chan_text(cam)
    rows = [ln.split() for ln in txt.strip().splitlines()]
    assert len(rows) == len(FRAMES) and all(len(r) == 8 for r in rows)
    fr, tr, rot = NU.camera_curves(cam)
    assert int(rows[4][0]) == fr[4] and np.allclose([float(v) for v in rows[4][1:4]], tr[4], atol=1e-6)
    assert np.allclose([float(v) for v in rows[4][4:7]], rot[4], atol=1e-6)


def test_refusals():
    cam = make_camera()
    with pytest.raises(ValueError):
        NU.build_script(cam, make_cards(), [], key_files={}, plate_pattern="p", nk_path="x")
    with pytest.raises(ValueError):
        NU.build_script(cam, make_cards(), _keys(), key_files=_key_files(_keys()), plate_pattern="p", nk_path="x",
                        frames=[1, 2, 99])


# ----------------------------------------------------------------------------------------------- files + route
def _shot(work, dist=(0, 0, 0, 0)):
    cam = make_camera()
    cam.dist = np.array(dist, float)
    store.save_camera(work, cam)
    cards = make_cards()
    cards[0].extent = [T.Area("e1", "curve", "add", {"type": "bezier", "closed": True, "pts": [
        {"p": [0, 120]}, {"p": [319, 120]}, {"p": [319, 199]}, {"p": [0, 199]}]})]
    cards[0].extent_frame = 1
    store.save_cards(work, cards)
    for k in _keys():
        KS.add_version(work, k.card, k.frame, np.full((H, W, 3), 0.5, np.float32), "plate", "png")
    return cam


def test_export_writes_files(tmp_path):
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    _shot(work, dist=(-0.08, 0.01, 0, 0))
    out = NU.export(work, plate_pattern=str(tmp_path / "plates" / "p_####.png"), tag="v007")
    nk, chan = Path(out["nk"]), Path(out["chan"])
    assert nk.name == "c3d_v007.nk" and nk.parent == (T.root_dir(work) / "nuke").resolve()
    assert nk.is_file() and chan.is_file() and len(out["stmaps"]) == 2 and len(out["extents"]) == 1
    for p in out["stmaps"]:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert img is not None and img.shape[:2] == (H, W) and img.dtype == np.float32
    ext = cv2.imread(out["extents"][0], cv2.IMREAD_GRAYSCALE)
    assert ext.shape == (H * 2, W * 2) and ext.max() == 255 and ext.min() == 0
    text = nk.read_text(encoding="utf-8")
    assert "\\" not in re.sub(r'\\[\[\]$"{}]', "", text)               # no backslash paths
    assert str(KS.version_path(work, store.load_key(work, "ground", 5)).resolve()).replace("\\", "/") in text
    assert len(chan.read_text().strip().splitlines()) == len(FRAMES)


def test_export_leaves_out_cards_off_or_outside_the_solo(tmp_path):
    """P16 (user 2026-11-10, like After Effects): a card turned off, or – when any card is soloed – a card without solo,
    is not exported (its keys, STMaps and extent masks neither)."""
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    _shot(work)
    ground_key = str(KS.version_path(work, store.load_key(work, "ground", 5)).resolve()).replace("\\", "/")
    full = NU.export(work, plate_pattern=str(tmp_path / "p_####.png"), tag="all")
    assert ground_key in Path(full["nk"]).read_text(encoding="utf-8") and len(full["extents"]) == 1
    cards = store.load_cards(work)
    other = next(c for c in cards if c.id != "ground")
    other.solo = True
    store.save_cards(work, cards)
    solo = NU.export(work, plate_pattern=str(tmp_path / "p_####.png"), tag="solo")
    text = Path(solo["nk"]).read_text(encoding="utf-8")
    assert ground_key not in text and len(solo["extents"]) == 0       # the ground (with the extent) is out
    other.off = True                                                  # the only soloed card off: nothing to export
    store.save_cards(work, cards)
    with pytest.raises(ValueError, match="solo"):
        NU.export(work, plate_pattern=str(tmp_path / "p_####.png"), tag="none")


def test_route_export_nuke(tmp_path, monkeypatch):
    from tests.test_cleanup3d_api import Env
    env = Env(tmp_path, monkeypatch)
    r = env.client.post("/api/cleanup3d/export_nuke", json={})
    assert r.status_code == 409 and "kamer" in r.json()["detail"].lower()
    _shot(env.work)
    r = env.client.post("/api/cleanup3d/export_nuke", json={})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["success"] and Path(d["nk"]).is_file() and Path(d["chan"]).is_file()
    assert re.match(r"c3d_\d{8}_\d{6}\.nk$", Path(d["nk"]).name)        # no projection yet: dated
    assert d["render"] is None and any("maski usuwania" in n or "maska" in n.lower() for n in d["notes"])
    assert "Read_removal_mask" not in Path(d["nk"]).read_text(encoding="utf-8")
    r = env.client.post("/api/cleanup3d/export_nuke")                   # body is optional
    assert r.status_code == 200


def test_c6_button_calls_the_route():
    js = (Path(__file__).resolve().parents[1] / "static" / "cleanup3d.js").read_text(encoding="utf-8")
    assert "'/api/cleanup3d/export_nuke'" in js and "addEventListener('click', exportNuke)" in js
    assert "etap II, jeszcze niedostępne" not in js                    # the stage-II stub is gone
