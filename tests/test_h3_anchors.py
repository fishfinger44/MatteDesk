# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MiniMax-H3 anchors of step C1 (user design 2026-11-11): a list (main / linked / own / original), each with the
history of its Qwen-Image-2.1 generations; detaching keeps the image; a linked anchor follows the main one's CURRENT
generation; own / original anchors = a transition (own clip prompt, no whole-clip reference). Synthetic: Qwen is a
stand-in, no ComfyUI."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as A
from engine import cleanup as cu
from engine import h3_anchors as HA
from engine import h3_fill

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_one_main_one_per_frame():
    a = HA.normalize([{"id": "a000001", "frame": 9, "kind": "linked"}, {"id": "a000002", "frame": 3, "kind": "linked"},
                      {"id": "a000003", "frame": 3, "kind": "main"}, {"id": "bad", "frame": 5}])
    assert [x["frame"] for x in a] == [3, 9] and a[0]["kind"] == "main"          # first one per frame, no main -> first
    b = HA.normalize([{"id": "a000001", "frame": 1, "kind": "main"}, {"id": "a000002", "frame": 2, "kind": "main"}])
    assert [x["kind"] for x in b] == ["main", "linked"]
    assert not HA.is_transition(b) and HA.is_transition(b + [{"id": "a000003", "frame": 4, "kind": "original"}])
    d = HA.defaults(31, [5, 50])
    assert [(x["frame"], x["kind"]) for x in d] == [(5, "linked"), (31, "main"), (50, "linked")]
    assert cu.settings({"anchors": d})["anchors"] == d


def test_generations_store_and_validity(tmp_path):
    root = tmp_path / "anchors"
    m = {"id": "a00000a", "frame": 31, "kind": "main", "gen": None}
    g1 = HA.add_gen(root, m["id"], b"png1", {"frame": 31, "prompt": "p", "seed": 1})
    g2 = HA.add_gen(root, m["id"], b"png2", {"frame": 31, "prompt": "p2", "seed": 2})
    assert (g1, g2) == ("g_001", "g_002") and [g["name"] for g in HA.gens(root, m["id"])] == ["g_001", "g_002"]
    m["gen"] = g2
    assert HA.valid_gen(root, m, None) == "g_002"
    lk = {"id": "a00000b", "frame": 5, "kind": "linked", "gen": None}
    lk["gen"] = HA.add_gen(root, lk["id"], b"x", {"frame": 5, "ref": "a00000a/g_002"})
    assert HA.valid_gen(root, lk, m) == lk["gen"]
    m["gen"] = g1                                                              # another main generation chosen
    assert HA.valid_gen(root, lk, m) is None                                   # -> the linked one follows again
    moved = {**m, "frame": 40}
    assert HA.valid_gen(root, moved, None) is None                             # made for another frame
    g = HA.gen_file(root, m["id"], "g_001")
    size = g.stat().st_size + g.with_suffix(".json").stat().st_size
    assert HA.delete(g) == size and not g.exists() and not g.with_suffix(".json").exists()
    assert HA.delete(g) == 0 and not (tmp_path / "trash").exists()              # deleted for good, no trash
    with pytest.raises(ValueError):
        HA.gen_file(root, "../x", "g_001")


def test_transition_prompt_and_no_reference(tmp_path):
    assert "nothing transforms" in h3_fill.h3_prompt("a porcelain face", "replace")
    tp = h3_fill.h3_prompt("her face slowly turns into porcelain", "replace", transition=True)
    assert "turns into porcelain" in tp and "nothing transforms" not in tp
    files = {"unet": "u", "clip": "c", "vae": "v", "turbo": "t"}
    g = h3_fill.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, files, guides=[(0, tmp_path)], reference=False)
    assert g["20"]["class_type"] == "MiniMaxH3ImageToVideo" and g["61"]["class_type"] == "MiniMaxH3AddGuide"


@pytest.fixture
def anchors_app(tmp_path, monkeypatch):
    """The anchor endpoints on a fake project: Qwen-Image-2.1 = a grey image with the seed in its value."""
    import api.cleanup as AC
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setitem(A.STATE, "work_dir", str(work))
    monkeypatch.setitem(A.STATE, "manifest", {"plates": {"beauty": str(tmp_path / "b" / "f_####.png")}})
    calls = []

    def fake_anchor(client, io, plate, mask, prompt, mode, linear=False, seed=None, ref=None, init=None, denoise=1.0, region=None, text=None, raw=None, plate_denoise=None, what=None):
        calls.append({"prompt": prompt, "seed": seed, "ref": ref is not None, "init": init is not None, "denoise": denoise})
        if raw is not None:
            raw.append(np.zeros((8, 8, 3), np.uint8))
        return np.full((8, 8, 3), (seed or 0) % 250, np.uint8)
    monkeypatch.setattr(h3_fill, "make_anchor", fake_anchor)
    import engine.tracking as TRK

    class StillTracker:   # linked anchors are warped from the main one (engine/anchor_warp): points stay put
        engine = "fake"

        def track(self, frames, queries):
            p = np.asarray(queries, np.float32)[:, 1:]
            return TRK.TrackResult(np.stack([p] * len(frames)), np.ones((len(frames), len(p)), bool), "fake")
    monkeypatch.setattr(TRK, "make_point_tracker", lambda *a, **k: StillTracker())
    monkeypatch.setattr(AC, "_read_plate", lambda p: np.zeros((8, 8, 3), np.float32))
    monkeypatch.setitem(A.STATE, "comfy_client", object())
    monkeypatch.setattr(A, "_work_frames", lambda: list(range(1, 51)))
    monkeypatch.setattr(A, "_plate_size", lambda: (8, 8))
    class Objs:   # one object with a computed mask (the mask itself is a stand-in)
        def list(self, groups=True):
            return [{"name": "obj"}]

        def dir(self, n):
            return tmp_path / n
    import engine.cleanup_masks as CM
    import engine.export as EX
    monkeypatch.setattr(A, "_objects", lambda: Objs())
    monkeypatch.setattr(CM, "resolve", lambda store, o, src: {"run": "r1", "source": "core", "valid": True, "by": "own", "why": ""})
    monkeypatch.setattr(EX, "run_alpha_loader", lambda d, run, w, h: (lambda f: np.ones((8, 8), np.float32)))
    cu.save_settings(work, {"model": "h3", "prompt": "a porcelain face", "prompt_mode": "replace", "remove": ["obj"]})
    return TestClient(A.app), work, calls


def test_anchor_endpoints_detach_and_history(anchors_app, monkeypatch):
    c, work, calls = anchors_app
    r = c.post("/api/cleanup/anchors", json={"frame": 31, "kind": "linked"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["anchors"][0]["kind"] == "main" and d["on"]                       # the first anchor = main
    d = c.post("/api/cleanup/anchors", json={"frame": 5, "kind": "linked"}).json()
    main = next(a for a in d["anchors"] if a["kind"] == "main")
    lk = next(a for a in d["anchors"] if a["kind"] == "linked")
    assert c.post("/api/cleanup/anchors", json={"frame": 5}).status_code == 409
    # a linked generation makes the main one first, with the main as Picture 2
    r = c.post(f"/api/cleanup/anchors/{lk['id']}/generate", json={"seed": 7})
    assert r.status_code == 200, r.text
    d = r.json()
    assert calls[0]["ref"] is False and calls[1]["ref"] is True
    lk = next(a for a in d["anchors"] if a["id"] == lk["id"])
    assert lk["valid"] and len(lk["gens"]) == 1
    # history: another main generation, choose the first again, delete one
    d = c.post(f"/api/cleanup/anchors/{main['id']}/generate", json={"prompt": "gold filigree porcelain"}).json()
    main = next(a for a in d["anchors"] if a["id"] == main["id"])
    assert len(main["gens"]) == 2 and main["gen"] == "g_002"
    assert cu.load_settings(work)["prompt"] == "gold filigree porcelain"       # main = the shared description
    assert not next(a for a in d["anchors"] if a["id"] == lk["id"])["valid"]  # linked follows the new main
    d = c.patch(f"/api/cleanup/anchors/{main['id']}", json={"gen": "g_001"}).json()
    assert next(a for a in d["anchors"] if a["id"] == main["id"])["gen"] == "g_001"
    d = c.delete(f"/api/cleanup/anchors/{main['id']}/gens/g_001").json()
    assert next(a for a in d["anchors"] if a["id"] == main["id"])["gen"] == "g_002"
    assert d["freed_bytes"] > 0 and not (work / "trash").exists()            # deleted for good (2026-11-13)
    # detach: the same image as an own generation, a transition from now on
    d = c.patch(f"/api/cleanup/anchors/{lk['id']}", json={"kind": "own"}).json()
    own = next(a for a in d["anchors"] if a["id"] == lk["id"])
    assert own["kind"] == "own" and len(own["gens"]) == 2 and own["valid"] and d["transition"]
    first, second = own["gens"][-1], own["gens"][0]
    assert first["seed"] == second["seed"] and second.get("detached_from")
    # an original anchor needs no generation
    d = c.post("/api/cleanup/anchors", json={"frame": 50, "kind": "original"}).json()
    orig = next(a for a in d["anchors"] if a["frame"] == 50)
    assert orig["valid"] and c.post(f"/api/cleanup/anchors/{orig['id']}/generate", json={}).status_code == 409
    # set another main
    d = c.patch(f"/api/cleanup/anchors/{own['id']}", json={"kind": "main"}).json()
    assert [a["kind"] for a in d["anchors"]] == ["main", "linked", "original"]


def test_ui_anchor_module():
    js = (ROOT / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    for s in ("'/api/cleanup/anchors'", "/generate`", "data-cut=\"add\"", "data-cut=\"orig\"", "timelineCuAnchor",
              "data-gdel", "data-anc-kind=\"own\"", "cu-anc-hover", "/import`", "data-anc-export", "cuAnchorPal", "makeFloating"):
        assert s in js, s
    assert 'id="cuH3Prompt"' in (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")   # Inpaint palette (2026-11-11)
    assert '<script src="/static/cleanup_anchors.js">' in (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def test_anchors_import_the_newest_version_images(anchors_app):
    """Runs made before the generation history: their pinned anchors become the first generations."""
    c, work, calls = anchors_app
    for v, frames in (("v005", (1, 31, 50)), ("v006", (5, 31, 50))):
        d = work.parent / "inpaint" / v / "anchors"
        d.mkdir(parents=True)
        (work.parent / "inpaint" / v / "run.json").write_text("{}", encoding="utf-8")
        for f in frames:
            (d / f"anchor_{f:04d}.png").write_bytes(f"{v}-{f}".encode())
    s = cu.load_settings(work)
    s.update(anchor="frame", anchors=HA.defaults(31, [5, 50]))
    cu.save_settings(work, s)
    d = c.get("/api/cleanup/anchors").json()
    assert all(a["gens"] and a["gens"][0]["imported"] == "v006" for a in d["anchors"])
    assert all(a["valid"] for a in d["anchors"]) and not calls                   # linked count as made from the main
    assert len(c.get("/api/cleanup/anchors").json()["anchors"][0]["gens"]) == 1  # imported once


def test_anchor_import_own_image(anchors_app):
    import base64
    import cv2
    c, work, calls = anchors_app
    d = c.post("/api/cleanup/anchors", json={"frame": 31}).json()
    d = c.post("/api/cleanup/anchors", json={"frame": 5}).json()
    lk = next(a for a in d["anchors"] if a["frame"] == 5)
    ok, buf = cv2.imencode(".png", np.full((4, 6, 3), 99, np.uint8))
    r = c.post(f"/api/cleanup/anchors/{lk['id']}/import", json={"data": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), "name": "ps.png"})
    assert r.status_code == 200, r.text
    a = next(x for x in r.json()["anchors"] if x["id"] == lk["id"])
    assert a["kind"] == "own" and a["valid"] and a["gens"][0]["imported"] == "ps.png" and not calls
    img = cv2.imread(str(HA.gen_file(work.parent / "inpaint" / "anchors", lk["id"], a["gen"])))
    assert img.shape[:2] == (8, 8)                                              # scaled to the plate frame
    assert c.post(f"/api/cleanup/anchors/{lk['id']}/import", json={"data": "xx"}).status_code == 400


def test_preview_shows_what_the_video_model_gets(anchors_app):
    """anchor_raw on: the generation's preview / export = Qwen-Image-2.1's raw picture (the one the run pins), off or no
    raw saved = the pasted one (user 2026-11-12: the pasted one was shown while the raw one went to MiniMax-H3)."""
    c, work, calls = anchors_app
    d = c.post("/api/cleanup/anchors", json={"frame": 31}).json()
    aid = d["anchors"][0]["id"]
    d = c.post(f"/api/cleanup/anchors/{aid}/generate", json={"seed": 1}).json()
    g = d["anchors"][0]["gens"][0]
    assert g["has_raw"] and g["shown"] == "raw" and g["url"].endswith("?raw=1")
    f = HA.gen_file(work.parent / "inpaint" / "anchors", aid, g["name"])
    raw_png = f.with_name(f"{g['name']}_raw.png")
    assert c.get(g["url"]).content == raw_png.read_bytes()
    assert c.get(g["url"] + "?t=old-ui").content == raw_png.read_bytes()   # a cached old UI glues a second "?"
    assert c.get(g["pasted_url"]).content == f.read_bytes()
    assert c.put("/api/cleanup", json={"anchor_raw": False}).status_code == 200
    g = c.get("/api/cleanup/anchors").json()["anchors"][0]["gens"][0]
    assert g["shown"] == "pasted" and g["url"] == g["pasted_url"]
    raw_png.unlink()                                                     # an old generation without a raw picture
    c.put("/api/cleanup", json={"anchor_raw": True})
    g = c.get("/api/cleanup/anchors").json()["anchors"][0]["gens"][0]
    assert not g["has_raw"] and g["shown"] == "pasted"
    assert c.get(g["pasted_url"] + "?raw=1").content == f.read_bytes()   # falls back to the pasted one


def test_prompt_check_endpoint_uses_the_typed_text(anchors_app):
    """POST /api/cleanup/prompt/check: warnings + the final texts for what is typed (not saved yet)."""
    c, work, calls = anchors_app
    c.post("/api/cleanup/anchors", json={"frame": 31})
    r = c.post("/api/cleanup/prompt/check", json={"prompt": "her face slowly transforms into porcelain [cite: 1]"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert {"event", "cite"} <= {w["code"] for w in d["lint"]} and d["clip_lint"] == []
    keys = [i["key"] for i in d["final"]["items"]]
    assert keys == ["anchor_main", "video"] and "slowly transforms" in d["final"]["items"][0]["text"]
    assert cu.load_settings(work)["prompt"] == "a porcelain face"          # nothing saved


def test_deleting_the_main_anchor_takes_its_linked_ones(anchors_app):
    c, work, calls = anchors_app
    c.post("/api/cleanup/anchors", json={"frame": 31})
    c.post("/api/cleanup/anchors", json={"frame": 5})
    c.post("/api/cleanup/anchors", json={"frame": 50})
    d = c.post("/api/cleanup/anchors", json={"frame": 40, "kind": "original"}).json()
    main = next(a for a in d["anchors"] if a["kind"] == "main")
    d = c.delete(f"/api/cleanup/anchors/{main['id']}").json()
    assert [(a["frame"], a["kind"]) for a in d["anchors"]] == [(40, "original")]   # linked gone, the original stays as is
    d = c.delete(f"/api/cleanup/anchors/{d['anchors'][0]['id']}").json()
    assert d["anchors"] == [] and not d["on"]


def test_settings_are_per_cleaned_thing():
    """Another object / group picked: its own model, description and anchors (user 2026-11-11)."""
    s = cu.settings({"remove": ["face"], "model": "h3", "prompt": "porcelain", "anchor": "frame",
                     "anchors": HA.defaults(31, [5])})
    s = cu.switch_target(s, "face", "grp")
    assert s["prompt"] == "" and s["anchors"] == [] and s["anchor"] == "none" and s["model"] == "h3"
    s["prompt"], s["model"] = "a wall", "lama"
    s = cu.switch_target(s, "grp", "face")
    assert s["prompt"] == "porcelain" and s["model"] == "h3" and len(s["anchors"]) == 2
    s = cu.settings(s)
    assert s["per_target"]["grp"]["prompt"] == "a wall" and "face" not in s["per_target"]


def test_showing_a_version_brings_back_its_anchors(anchors_app, monkeypatch):
    """Click on a version = the plate „cleaned” shows it and its model / description / anchors come back (user
    2026-11-11); a run made before the anchor list: one anchor per pinned image, the same ids on a second click."""
    import json
    c, work, calls = anchors_app
    inp = work.parent / "inpaint"
    v = inp / "v006"
    (v / "anchors").mkdir(parents=True)
    for f in (5, 31, 50):
        (v / "anchors" / f"anchor_{f:04d}.png").write_bytes(f"img{f}".encode())
    (v / "run.json").write_text(json.dumps({"frames": [1, 50], "anchors": [5, 31, 50], "ext": "png", "settings": {
        "remove": ["obj"], "model": "h3", "prompt": "porcelain", "prompt_mode": "replace", "anchor": "frame",
        "anchor_frame": 31, "seed": 9}}), encoding="utf-8")
    monkeypatch.setitem(A.STATE, "shot_project", None)                          # no project file: no plate switch
    r = c.post("/api/cleanup/versions/v006/show")
    assert r.status_code == 200, r.text
    s = cu.load_settings(work)
    assert s["prompt"] == "porcelain" and s["seed"] == 9 and s["anchor"] == "frame"
    assert [(a["frame"], a["kind"]) for a in s["anchors"]] == [(5, "linked"), (31, "main"), (50, "linked")]
    ids = [a["id"] for a in s["anchors"]]
    d = c.get("/api/cleanup/anchors").json()
    assert all(a["valid"] for a in d["anchors"])                                 # linked made from the main one
    c.post("/api/cleanup/versions/v006/show")
    s2 = cu.load_settings(work)
    assert [a["id"] for a in s2["anchors"]] == ids and all(len(a["gens"]) == 1 for a in c.get("/api/cleanup/anchors").json()["anchors"])
    assert cu.versions(work)[0]["model"] == "h3" and cu.versions(work)[0]["anchors"] == [5, 31, 50]


def test_generate_linked_from_the_main_choice(anchors_app):
    """„🔗 Generuj podłączone”: the linked ones from the main anchor's CHOSEN image; the main one is not generated
    again; anchors never move (user 2026-11-11)."""
    c, work, calls = anchors_app
    c.post("/api/cleanup/anchors", json={"frame": 31})
    c.post("/api/cleanup/anchors", json={"frame": 5})
    d = c.post("/api/cleanup/anchors", json={"frame": 50}).json()
    main = next(a for a in d["anchors"] if a["kind"] == "main")
    assert c.post("/api/cleanup/anchors/linked/generate").status_code == 409   # no main generation yet
    c.post(f"/api/cleanup/anchors/{main['id']}/generate", json={"seed": 3})
    assert c.patch(f"/api/cleanup/anchors/{main['id']}", json={"frame": 27}).status_code == 409   # locked to F31
    assert c.put("/api/cleanup", json={"link_method": "warp_qwen"}).status_code == 200   # default = qwen (below)
    calls.clear()
    r = c.post("/api/cleanup/anchors/linked/generate")
    assert r.status_code == 200, r.text
    assert len(calls) == 2 and all(x["ref"] for x in calls)                       # only the two linked, from the main
    d = r.json()
    assert all(a["valid"] for a in d["anchors"] if a["kind"] == "linked")
    assert len(next(a for a in d["anchors"] if a["kind"] == "main")["gens"]) == 1
    # warp_qwen: the main pattern is moved onto them (anchor_warp) and Qwen-Image-2.1 only touches it up
    assert all(x["init"] and x["denoise"] == 0.4 for x in calls)
    lk = next(a for a in d["anchors"] if a["kind"] == "linked")
    chosen = lambda a: next(g for g in a["gens"] if g["name"] == a["gen"])   # noqa: E731
    assert chosen(lk).get("link") == "warp_qwen"
    adir = cu.cleanup_dir(work) / "anchors" / lk["id"]
    assert (adir / f"{lk['gen']}_raw.png").exists() and (adir / f"{lk['gen']}_warp.png").exists()   # kept for the user
    calls.clear()                                               # all valid: nothing to do, all=1 = every linked again
    assert c.post("/api/cleanup/anchors/linked/generate").status_code == 200 and calls == []
    assert c.put("/api/cleanup", json={"link_method": "warp"}).status_code == 200
    r = c.post("/api/cleanup/anchors/linked/generate?all=1")
    assert r.status_code == 200, r.text
    assert calls == [] and chosen(next(a for a in r.json()["anchors"] if a["kind"] == "linked"))["link"] == "warp"
    c.put("/api/cleanup", json={"link_method": "qwen"})
    c.post("/api/cleanup/anchors/linked/generate?all=1")
    assert len(calls) == 2 and not any(x["init"] for x in calls)                # the old way: Picture 2 only


def test_anchor_paste_is_hard_over_the_repainted_area(monkeypatch):
    """User 2026-11-11: the patch's soft edge / soft holdout must not blend the anchor with the plate – the anchor holds
    the edit over the whole area the video model repaints (mask > 0.02); the blend is the Cleaner's, at the end."""
    import engine.cleanup3d.regen as R
    import engine.stage_models as SM
    monkeypatch.setattr(SM, "comfy_gate", lambda client, name: "")
    monkeypatch.setattr(R, "_run_qwen", lambda client, io, real, region, prompt, progress, mode, **k: (np.ones_like(real), None, 1))
    plate = np.zeros((4, 4, 3), np.float32)
    m = np.zeros((4, 4), np.float32)
    m[1, 1], m[2, 2] = 0.3, 1.0
    out = h3_fill.make_anchor(None, Path("."), plate, m, "p", "replace")
    assert out[1, 1, 0] == 255 and out[2, 2, 0] == 255 and out[0, 0, 0] == 0   # 0.3 -> fully the edit, not 30 %


@pytest.fixture
def c5_render(anchors_app, monkeypatch):
    """A fake Cleanup 3D scene: card keys at F1 / F9 / F17 (+ an inactive one at F13), the last render = v003 whose comp
    frame f has the value f (so the copied anchor says which frame it came from)."""
    import cv2
    from types import SimpleNamespace
    import engine.cleanup3d.store as ST
    c, work, calls = anchors_app
    keys = [SimpleNamespace(frame=f, active="v001", card="sand") for f in (1, 9, 17)] + \
        [SimpleNamespace(frame=13, active=None, card="sand"), SimpleNamespace(frame=9, active="v001", card="sky")]
    monkeypatch.setattr(ST, "list_keys", lambda w, card=None: keys)
    state = {"version": "v003", "stale": False}
    monkeypatch.setattr(ST, "render_state", lambda w: dict(state))
    comp = work.parent / "inpaint" / "v003" / "comp"
    comp.mkdir(parents=True)
    for f in range(1, 51):
        cv2.imwrite(str(comp / f"c_{f:04d}.png"), np.full((8, 8, 3), f, np.uint8))
    return c, work, calls, state


def test_projection_anchors_from_card_keys(c5_render):
    """User request 2026-11-12 (MiniMax-H3 with the card projection instead of VOID): „⚓ z kluczy kart” = one anchor per
    active card-key frame, its image = the last C5 render of that frame, no Qwen-Image-2.1, NOT a transition."""
    import cv2
    c, work, calls, state = c5_render
    s = cu.load_settings(work)
    s.update(anchor="frame", anchors=HA.defaults(31, [5, 50]))   # the automatic ones (no image yet) get replaced
    cu.save_settings(work, s)
    r = c.post("/api/cleanup/anchors/projection", json={})
    assert r.status_code == 200, r.text
    d = r.json()
    assert [(a["frame"], a["kind"]) for a in d["anchors"]] == [(1, "projection"), (9, "projection"), (17, "projection")]
    assert all(a["valid"] for a in d["anchors"]) and not d["transition"] and d["main"] is None and not calls
    assert d["projection"] == {"render": "v003", "stale": False, "outdated": []}
    a9 = d["anchors"][1]
    img = cv2.imread(str(HA.gen_file(work.parent / "inpaint" / "anchors", a9["id"], a9["gen"])))
    assert int(img[0, 0, 0]) == 9 and a9["gens"][0]["projection"] == "v003"     # the render of ITS frame
    # a new render: the copies are outdated until the button is pressed again (a new generation, history kept)
    state["version"] = "v004"
    (work.parent / "inpaint" / "v004").mkdir()
    import shutil
    shutil.copytree(work.parent / "inpaint" / "v003" / "comp", work.parent / "inpaint" / "v004" / "comp")
    assert len(c.get("/api/cleanup/anchors").json()["projection"]["outdated"]) == 3
    d = c.post("/api/cleanup/anchors/projection", json={"step": 20}).json()      # denser: + F1, F21, F41
    assert [a["frame"] for a in d["anchors"]] == [1, 9, 17, 21, 41] and not d["projection"]["outdated"]
    assert len(next(a for a in d["anchors"] if a["frame"] == 9)["gens"]) == 2
    # the prompt check shows no Qwen-Image-2.1 anchor prompt and the plain clip prompt with the <Picture 1> wrap
    from engine import prompt_rules as PR
    fin = PR.final_prompts(cu.load_settings(work))
    assert [i["key"] for i in fin["items"]] == ["video"] and "<Picture 1>" in fin["items"][0]["text"]


def test_projection_anchor_needs_a_render(anchors_app, monkeypatch):
    import engine.cleanup3d.store as ST
    c, work, calls = anchors_app
    monkeypatch.setattr(ST, "render_state", lambda w: None)
    r = c.post("/api/cleanup/anchors/projection", json={})
    assert r.status_code == 409 and "C5" in r.json()["detail"]
    assert c.post("/api/cleanup/anchors", json={"frame": 3, "kind": "projection"}).status_code == 409


def test_anchors_off_means_no_anchor_for_any_model(anchors_app):
    """„Kotwica: brak” = no model gets an anchor, even with a saved list (user 2026-11-13: a remove run pinned four face
    anchors of an earlier session). The list itself stays, so switching the anchors back on brings it again."""
    import inspect
    import api.cleanup as AC
    saved = HA.defaults(31, [5, 50])
    assert cu.active_anchors({"anchor": "none", "anchors": saved}) == []
    assert cu.active_anchors({"anchor": "frame", "anchors": saved}) == saved
    assert cu.active_anchors({"anchor": "frame"}) == []
    c, work, calls = anchors_app
    c.post("/api/cleanup/anchors", json={"frame": 31})
    s = cu.load_settings(work / "matte_desk") if (work / "matte_desk").is_dir() else cu.load_settings(work)
    assert s["anchor"] == "frame" and s["anchors"]
    # the run builds its anchor list only through _ensure_anchors, which answers through cu.active_anchors when off
    src = inspect.getsource(AC)
    run_src = src[src.index("async def cleanup_run"):]
    assert "_ensure_anchors(work, s, frames, rmask)" in run_src and "s.get(\"anchors\")" not in run_src.split("def ")[0]
    ens = src[src.index("def _ensure_anchors"):src.index("def _gen_image")]
    assert "cu.active_anchors(s)" in ens
    from engine import prompt_rules as PR
    off = PR.final_prompts({**s, "anchor": "none", "model": "h3", "prompt": "a wall"})
    assert not any(i["key"].startswith("anchor") for i in off["items"]) and not off["transition"]


def test_token_clash_2026_11_16():
    """Two anchors in one MiniMax-H3 token (frames 1 | 2-5 | 6-9 ... of the range) = only one holds: the palette and
    the run notes warn (user test a test shot, v004: F2 + F5 linked, the earring came back in F2)."""
    frames = list(range(1, 51))
    assert [HA.h3_token(i) for i in (0, 1, 4, 5, 8, 9)] == [0, 1, 1, 2, 2, 3]
    a = [{"id": "m", "frame": 49, "kind": "main"}, {"id": "x", "frame": 2, "kind": "linked"},
         {"id": "y", "frame": 5, "kind": "linked"}]
    c = HA.token_clashes(a, frames)
    assert len(c) == 1 and c[0]["frames"] == [2, 5] and c[0]["ids"] == ["x", "y"] and c[0]["span"] == [2, 5]
    assert "F2, F5" in HA.clash_note(c[0]) and "F2–F5" in HA.clash_note(c[0])
    # F1 is a token of its own; F5 and F6 are neighbouring tokens; outside the range = ignored
    assert HA.token_clashes([{"id": "p", "frame": 1}, {"id": "q", "frame": 2}, {"id": "r", "frame": 6},
                             {"id": "s", "frame": 60}, {"id": "t", "frame": 61}], frames) == []
    # the range need not start at 1: tokens count from its first frame
    c2 = HA.token_clashes([{"id": "u", "frame": 102}, {"id": "v", "frame": 105}], list(range(101, 131)))
    assert c2 and c2[0]["span"] == [102, 105]
    # wired: the anchors view (model h3 only) and the H3 run notes
    src = (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert '"token_clash": clashes' in src and "HA.token_clashes(lst, frames)" in src and "clash_notes" in src
    js = (ROOT / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "token_clash" in js and "clashHtml()" in js


def test_h3_token_layout_and_masks_2026_11_16():
    """Tokens of 1, 4, 4, 4, 4 frames repeated every 17 (comfy FRAME_PER_TOKEN[k % 5]); h3_fill writes ONE noise mask
    per token = the max over its frames, so ComfyUI's trilinear resize to the latent is identity (v005: the token of
    F2-F5 sampled the mask near F5.4 and left 51 % of the F2 earring unmasked)."""
    sp = HA.token_spans(56)
    assert sp[:7] == [(0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 17), (18, 21)] and len(sp) == 17
    assert sum(b - a + 1 for a, b in sp) == 56
    for k, (a, b) in enumerate(sp):   # h3_token agrees with the spans
        assert all(HA.h3_token(i) == k for i in range(a, b + 1))
    fr = list(range(1, 57))
    assert HA.h3_token_span(5, fr) == [18, 18] and HA.h3_token_span(6, fr) == [19, 22]
    # F18 (idx 17) is a token of its own: F17 + F18 do not clash, F19 + F22 do
    assert HA.token_clashes([{"id": "a", "frame": 17}, {"id": "b", "frame": 18}], fr) == []
    assert HA.token_clashes([{"id": "a", "frame": 19}, {"id": "b", "frame": 22}], fr)[0]["span"] == [19, 22]
    # one mask per token, the union of its frames: an object only in F2 is masked in the token of F2-F5
    masks = [np.zeros((4, 4), np.uint8) for _ in range(56)]
    masks[1][0, 0] = 255
    tm = h3_fill.token_masks(masks)
    assert len(tm) == 17 and tm[1][0, 0] == 255 and tm[0][0, 0] == 0 and tm[2][0, 0] == 0
    src = (ROOT / "engine" / "h3_fill.py").read_text(encoding="utf-8")
    assert "enumerate(token_masks(grown))" in src


def test_frame_mask_per_model_2026_11_16():
    """„Maska klatki” shows the ENGINE's mask of the frame (user 2026-11-16): MiniMax-H3 = its whole latent token
    ±2 frames, Wan 2.2 Fun VACE / LTX-2.5 = the frame ±2, other models = the frame itself."""
    fr = list(range(1, 51))
    assert cu.model_frames("h3", 2, fr) == list(range(1, 8))     # token F2-F5 (idx 1-4) ± 2, clipped at F1
    assert cu.model_frames("h3", 1, fr) == [1, 2, 3]             # F1 = a token of its own
    assert cu.model_frames("h3", 18, fr) == list(range(16, 21))   # idx 17 = a single-frame token
    assert cu.model_frames("vace", 10, fr) == list(range(8, 13))
    assert cu.model_frames("ltx", 50, fr) == [48, 49, 50]
    assert cu.model_frames("lama", 10, fr) == [10] and cu.model_frames("h3", 99, fr) == [99]
    # the limit line follows the union, `cut` stays at this frame's own area (what the composite is cut at)
    from engine import inpaint_band as IB
    a = np.zeros((64, 64), np.float32); a[30:34, 10:14] = 1
    b = np.zeros((64, 64), np.float32); b[30:34, 50:54] = 1
    one = IB.lines([a], 0, 0, None, 1.0, 3.0)
    both = IB.lines([a], 0, 0, None, 1.0, 3.0, limit_alphas=[a, b])
    assert len(both["limit_contours"]) == 2 and len(one["limit_contours"]) == 1 and both["cut"] == one["cut"]
    src = (ROOT / "api" / "inpaint_band.py").read_text(encoding="utf-8")
    assert "cu.model_frames(" in src and '"h3_token"' in src
    assert "cu.model_frames(" in (ROOT / "api" / "cleanup.py").read_text(encoding="utf-8")
    assert "dataset.modelNote" in (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
