# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plan step 9 (docs/PLAN_HAIR_TOOL.md section 7 without a VLM), synthetic, mock Comfy:
- prompt rules PL/EN (`engine/localmatte_prompt.py`): preset, subject -> SAM3 phrase, distractors, line in the
  background, routing multipliers (busy / plain / same-colour background, named / forced engine), warnings;
- Hough "-" strokes along straight lines in the background of the zone (never through the core);
- zone proposals from "SAM3 foreground outside the core" + soft edge pixels;
- API: create / PATCH apply the preset from the description, `zone_engines` of a preview follow the prompt (same colour
  as the background -> chroma off), `POST zones/interpret`, `POST zones/propose` (+ CLI zone-interpret / zone-propose
  --accept)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from engine.localmatte_prompt import detect_bg_lines, interpret_prompt, merge_hints, propose_zones, soft_edge_candidates  # noqa: E402


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ---------------------------------------------------------------------------------------------- rules
def test_prompt_rules_pl_en():
    p = interpret_prompt("włosy bez złotej linii")
    assert p.preset == "hair" and p.fg == "włosy" and p.fg_en == "hair" and p.distractors == ["zlotej linii"] and p.bg_line
    assert p.engine_hints == {} and "distractor" in p.matched and "bg:line" in p.matched
    p = interpret_prompt("hair without the pole behind, plain sky")
    assert p.preset == "hair" and p.fg_en == "hair" and p.distractors == ["pole behind"] and p.bg_line and p.engine_hints == {"chroma": 1.5}
    p = interpret_prompt("fur, patterned background")
    assert p.preset == "hair" and p.fg_en == "fur" and p.engine_hints == {"chroma": 0.3} and not p.bg_line and p.distractors == []
    p = interpret_prompt("włókna filcu bez słupa")
    assert p.preset == "fibres" and p.fg_en == "fibres felt" and p.bg_line
    p = interpret_prompt("dym nad kubkiem")
    assert p.preset == "smoke" and p.fg_en.startswith("smoke")
    p = interpret_prompt("rozmyte włosy w ruchu")
    assert p.preset == "motion" and p.fg_en == "hair"
    p = interpret_prompt("trawa i gałęzie; pomiń kabel")
    assert p.preset == "thin" and p.fg_en == "grass branches" and p.distractors == ["kabel"] and p.bg_line
    p = interpret_prompt("włosy w kolorze tła")
    assert p.same_color_bg and p.engine_hints["chroma"] == 0.0 and p.engine_hints["knn"] == 1.3 and p.warnings and p.fg_en == "hair"
    p = interpret_prompt("tylko knn, clip")
    assert p.preset == "hair" and p.forced_engine == "knn" and p.engine_hints["knn"] == 1.5 and p.engine_hints["chroma"] == 0.0 and p.engine_hints["clean"] == 0.0
    p = interpret_prompt("chroma na jednolitej ścianie")
    assert p.engine_hints["chroma"] == 1.5 and p.forced_engine is None and p.preset is None
    assert interpret_prompt("").preset is None and interpret_prompt("   ").to_dict()["text"] == ""
    assert interpret_prompt("coś zupełnie innego").preset is None and interpret_prompt("coś zupełnie innego").engine_hints == {}
    # explicit zone hints always multiply the prompt-derived ones
    assert merge_hints({"knn": 2.0}, interpret_prompt("pstre tło")) == {"chroma": 0.3, "knn": 2.0}
    assert merge_hints({"chroma": 0.5}, interpret_prompt("pstre tło")) == {"chroma": 0.15}
    assert merge_hints(None, None) == {}


# ---------------------------------------------------------------------------------------------- Hough lines in the background
def _scene(line_y=60, through_core=False):
    h, w = 300, 400
    img = np.full((h, w, 3), 0.55, np.float32)
    core = np.zeros((h, w), bool)
    cv2.circle(core.view(np.uint8), (200, 170), 60, 1, -1)
    y = 170 if through_core else line_y
    cv2.line(img, (10, y), (390, y), (0.05, 0.05, 0.05), 2)      # a dark horizontal line (a pole / the golden line)
    region = np.zeros((h, w), bool)
    region[80:140, 120:280] = True                                  # zone: above the head, the line crosses it
    if through_core:
        region[150:190, 120:280] = True
    return img, core, region


def test_bg_lines_found_above_the_core_and_never_through_it():
    img, core, region = _scene(line_y=100)
    out = detect_bg_lines(img, region, core)
    assert len(out) == 1, out
    st = out[0]
    assert st["source"] == "hough" and st["radius_px"] == 3 and abs(st["angle"]) < 2 and st["length_px"] >= 0.45 * 160
    assert all(abs(q[1] - 100) <= 2 for q in st["p"])                       # along the line
    xs = [q[0] for q in st["p"]]
    assert min(xs) >= 120 - 24 - 2 and max(xs) <= 280 + 24 + 2             # inside bbox(region) + margin
    # the same line through the core: rejected; no line at all: nothing; empty region: nothing
    img2, core2, region2 = _scene(through_core=True)
    assert detect_bg_lines(img2, region2, core2) == []
    plain = np.full_like(img, 0.55)
    assert detect_bg_lines(plain, region, core) == []
    assert detect_bg_lines(img, np.zeros_like(region), core) == []
    # two parallel lines -> two strokes, longest first; max_lines caps
    cv2.line(img, (10, 90), (390, 90), (0.05, 0.05, 0.05), 2)
    two = detect_bg_lines(img, region, core)
    assert len(two) == 2 and two[0]["length_px"] >= two[1]["length_px"]
    assert len(detect_bg_lines(img, region, core, max_lines=1)) == 1


# ---------------------------------------------------------------------------------------------- proposals
def test_zone_proposals_from_fg_outside_core_and_soft_edges():
    h, w = 300, 400
    gt = np.zeros((h, w), np.uint8)
    cv2.circle(gt, (200, 150), 70, 255, -1)
    core = cv2.erode(gt, np.ones((19, 19), np.uint8)) > 0           # the core misses a 9 px ring
    fg = gt.astype(np.float32) / 255.0
    props = propose_zones(core, fg, None, min_area=200)
    assert len(props) == 1 and props[0]["source"] == "sam3" and props[0]["id"] == 0
    poly = np.array(props[0]["polygon"], np.int32)
    assert len(poly) >= 8 and props[0]["area_px"] > 2000
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [poly], 1)
    ring = (gt > 0) & ~core
    assert (m.astype(bool) & ring).sum() >= 0.9 * ring.sum()           # the proposal covers the missed ring
    # two lobes left / right of the core -> two proposals with sides; too small ones dropped; nothing outside -> []
    fg2 = core.astype(np.float32).copy()
    cv2.rectangle(fg2, (90, 120), (128, 180), 1.0, -1)
    cv2.rectangle(fg2, (272, 120), (330, 180), 1.0, -1)
    cv2.rectangle(fg2, (200, 10), (205, 14), 1.0, -1)              # 30 px: below min_area
    p2 = propose_zones(core, fg2, None, min_area=200)
    assert [p["side"] for p in p2] == ["prawo", "lewo"] and [p["id"] for p in p2] == [0, 1]   # sorted by area
    assert propose_zones(core, core.astype(np.float32), None) == []
    # soft edge pixels: an antialiased object on a plain background -> the mixed ring is proposed (source soft)
    img = np.full((h, w, 3), 0.9, np.float32)
    a = cv2.GaussianBlur(gt.astype(np.float32) / 255.0, (0, 0), 4)
    img = img * (1 - a[..., None]) + np.array([0.1, 0.1, 0.1], np.float32) * a[..., None]
    soft = soft_edge_candidates(img, core)
    assert soft.any() and not (soft & core).any()
    p3 = propose_zones(core, None, soft, min_area=200)
    assert len(p3) >= 1 and p3[0]["source"] == "soft"
    assert not soft_edge_candidates(np.full((h, w, 3), 0.1, np.float32), core).any()   # object == background colour: no evidence


# ---------------------------------------------------------------------------------------------- API + CLI
def test_prompt_api_preset_hints_interpret_propose_cli(tmp_path, monkeypatch):
    import md
    from test_md_cli import TCHttp, _run
    from test_zone_matte import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=4)
    cx, cy, s = shot.centers[start]
    poly = _rect(cx - 60, cy - 260, cx + 60, cy - 120)
    # create: the description names the kind of detail (default preset `hair` -> fibres); an explicit preset wins
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json={"mode": "add", "criterion": "matte", "frame": start, "polygon": poly,
                                                                             "matting": "auto", "prompt": "włókna filcu bez słupa"})
    assert r.status_code == 200, r.text
    z1 = r.json()["zone"]; zid = z1["zone_id"]
    assert z1["preset"] == "fibres" and r.json()["plan"]["bg_line"] is True and r.json()["plan"]["fg_en"] == "fibres felt"
    # one matte zone per object (user decision 2026-10-25): a second one is refused; the explicit preset is checked by PATCH
    r = api.post(f"/api/object/obj/zones/new?preview_frame={start}", json={"mode": "add", "criterion": "matte", "frame": start, "polygon": poly,
                                                                             "matting": "auto", "prompt": "włosy", "preset": "thin"})
    assert r.status_code == 409 and zid in r.json()["detail"]
    # PATCH prompt -> preset + plan; an explicit preset in the same PATCH wins
    p = api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "dym nad kubkiem"})
    assert p.status_code == 200 and p.json()["zone"]["preset"] == "smoke" and p.json()["plan"]["preset"] == "smoke" and "preset" in p.json()["changed"]
    p = api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy", "preset": "thin"})
    assert p.json()["zone"]["preset"] == "thin"
    # engines follow the prompt: "same colour as the background" -> chroma off in the preview's engine map (blue channel)
    def chroma_px(f):
        rr = api.get(f"/api/object/obj/frame/{f}?view=zone_engines"); assert rr.status_code == 200
        m = cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
        return int((m[..., 0] > 0).sum()), int((m[..., 2] > 0).sum())
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy", "preset": "hair", "matting": "auto"}).status_code == 200
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 200
    chroma_plain, knn_plain = chroma_px(start)
    assert chroma_plain > 0
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy w kolorze tła"}).status_code == 200
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 200
    chroma_same, knn_same = chroma_px(start)
    assert chroma_same == 0 and knn_same > 0, (chroma_same, knn_same)
    # a key's iteration note also feeds the rules in its frame (key_zone appends it)
    f2 = frames[-1]
    assert api.patch(f"/api/object/obj/zones/{zid}", json={"prompt": "włosy"}).status_code == 200
    assert api.put(f"/api/object/obj/zones/{zid}/keys/{f2}", json={"polygon": _rect(cx + 100, cy - 260, cx + 220, cy - 120), "prompt": "w kolorze tła"}).status_code == 200
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{f2}").status_code == 200
    assert chroma_px(f2)[0] == 0
    assert api.post(f"/api/object/obj/zones/{zid}/preview/{start}").status_code == 200
    assert chroma_px(start)[0] > 0                                      # the base key has no such note
    # interpret: plan only / with a frame -> Hough strokes (structure; none may cross the core)
    i = api.post("/api/object/obj/zones/interpret", json={"prompt": "włosy bez linii"})
    assert i.status_code == 200 and i.json()["plan"]["bg_line"] is True and i.json()["bg_strokes"] == [] and "frame" not in i.json()
    i = api.post("/api/object/obj/zones/interpret", json={"prompt": "włosy bez linii", "frame": start, "zone_id": zid})
    assert i.status_code == 200 and i.json()["frame"] == start and isinstance(i.json()["bg_strokes"], list)
    core = cv2.imread(str(pipe.store.dir("obj") / "core" / f"c_{start:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    for st in i.json()["bg_strokes"]:
        for q in st["p"]:
            assert not core[int(q[1]), int(q[0])]
    assert api.post("/api/object/obj/zones/interpret", json={"prompt": "x", "frame": start + 500}).status_code == 404
    assert api.post("/api/object/obj/zones/interpret", json={"prompt": "x", "frame": start, "zone_id": "z_99"}).status_code == 404
    # propose: the core eroded by 9 px -> the missed ring is proposed (SAM3 mock keys the whole puppet)
    cp = pipe.store.dir("obj") / "core" / f"c_{start:04d}.png"
    full = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(cp), cv2.erode(full, np.ones((19, 19), np.uint8)))
    pr = api.post("/api/object/obj/zones/propose", json={"frame": start, "prompt": "włosy"})
    assert pr.status_code == 200, pr.text
    d = pr.json()
    assert d["sam3"] is True and d["fg"] == "hair" and d["preset"] == "hair" and len(d["proposals"]) >= 1
    top = d["proposals"][0]
    assert top["area_px"] > 500 and len(top["polygon"]) >= 4 and top["source"] in ("sam3", "both")
    eroded = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE) > 127
    m = np.zeros(eroded.shape, np.uint8); cv2.fillPoly(m, [np.array(top["polygon"], np.int32)], 1)
    ring = (full > 127) & ~eroded
    assert (m.astype(bool) & ring).sum() >= 0.6 * ring.sum()
    assert api.post("/api/object/obj/zones/propose", json={"frame": start + 500}).status_code == 404
    # CLI
    code, out = _run(["zone-interpret", "obj", "włosy bez złotej linii", "--frame", str(start), "--zone", zid], TCHttp(api))
    assert code == 0 and "preset hair" in out and "SAM3 „hair”" in out and "zlotej linii" in out, out
    # one matte zone per object (user decision 2026-10-25): accepting a proposal next to a zone is refused ...
    code, out = _run(["zone-propose", "obj", "--frame", str(start), "--prompt", "włosy", "--accept", "0"], TCHttp(api))
    assert code != 0 and "409" in out, out
    # ... and works once the object has none
    assert api.delete(f"/api/object/obj/zones/{zid}").status_code == 200
    code, out = _run(["zone-propose", "obj", "--frame", str(start), "--prompt", "włosy", "--accept", "0"], TCHttp(api))
    assert code == 0 and "propozycji" in out and "zapisana" in out, out
    zs = api.get("/api/object/obj/zones").json()["zones"]
    assert len(zs) == 1 and zs[-1]["preset"] == "hair" and len(zs[-1]["polygon"]) == len(top["polygon"])
    cv2.imwrite(str(cp), full)
