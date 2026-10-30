# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests of the group as its own entity (docs/PLAN_GROUP_COMPOSE.md, step 0, user decisions 2026-10-29).

(a) a group of members on Core (mask 1+2) = bit for bit the legacy `<parent>_full` (no whole recompute) – green from step 0;
    since step 6 (parts migrated to groups) checked against the frozen formula `_core_composition`;
(b) a member with source X brings exactly the alpha of X into the composition, outside the seam;
(c) a change of a member's source / the seam / the members = the whole recompute is stale;
(d) detaching a member while the group has a whole result -> a warning first, the result goes to the project trash.

(b)–(d) describe the new behaviour and are xfail(strict) until their step lands. Every call of the not-yet-built group
API goes through `G` below – step 1 (data model) adjusts only that adapter, never the assertions."""

import json

import cv2
import numpy as np
import pytest

from engine.crop import CropSegment
from engine.exr_io import read_exr, write_gray_exr
from engine.export import CORE_RUN, core_fix_loader, export_objects, frame_alphas, seamless_union
from engine.objects import ObjectStore

W, H = 200, 160
SEAM = 6          # != SEAM_PX_DEFAULT: the seam setting must really reach the composition
# the seam fill reaches pixels with d_a + d_b <= 2 * SEAM + 1 -> up to 2 * SEAM + 1 px from the other member
_SEAM_ZONE = np.ones((2 * (2 * SEAM + 3) + 1,) * 2, np.uint8)


def _soft(mask, r=2.0):
    d_in = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(0.5 + (d_in - d_out) / (2 * r), 0, 1).astype(np.float32)


def _masks():
    """Head / torso / legs with 4 px gaps between them (the seam has something to fill)."""
    yy, xx = np.mgrid[0:H, 0:W]
    head = (abs(xx - 100) <= 25) & (yy >= 10) & (yy < 30)
    torso = (abs(xx - 100) <= 25) & (yy >= 34) & (yy < 80)
    legs = (abs(xx - 100) <= 25) & (yy >= 84) & (yy < 140)
    return {"head": head, "torso": torso, "legs": legs}


def _object(store, name, mask, depth, active_run=CORE_RUN, soft_r=2.0):
    """Object with a SAM 3.1 core (core/) and one computed version v001 (soft alpha, `soft_r` px ramp)."""
    store.create(name, [name], depth_order=depth)
    od = store.dir(name)
    seg = CropSegment(seg_id="s00", first=1, last=1, overlap_in=0, overlap_out=0, crop=[0, 0, W, W],
                      pad=[0, 0, 0, W - H], scale_to_1008=5.04)
    (od / "runs" / "v001" / "alpha" / "s00").mkdir(parents=True)
    a = np.zeros((W, W), np.float32)
    a[:H] = _soft(mask, soft_r)
    write_gray_exr(od / "runs" / "v001" / "alpha" / "s00" / "a_0001.exr", a)
    (od / "runs" / "v001" / "run.json").write_text(json.dumps({"edge_mode": "core", "segments_def": [seg.to_dict()]}))
    (od / "core").mkdir(exist_ok=True)
    cv2.imwrite(str(od / "core" / "c_0001.png"), mask.astype(np.uint8) * 255)
    store.update(name, tool={"active_run": active_run, "final_valid": True})


def _core_composition(work, members, seam):
    """THE contract of the basic group matte on Core: seamless_union of the members' masks 1+2, seams between their
    cores filled (`seam` px). Frozen here – the group loader of step 3 must give exactly this."""
    root = work / "objects"
    alphas = [core_fix_loader(root / m, W, H)(1) for m in members]
    cores = [cv2.imread(str(root / m / "core" / "c_0001.png"), cv2.IMREAD_GRAYSCALE) > 127 for m in members]
    return seamless_union(alphas, cores, gap_px=seam)


# ---------------------------------------------------------------- (a) green today
def test_a_core_group_equals_the_frozen_composition(tmp_path):
    """Group of Core members (no whole recompute) = the composition the legacy `<parent>_full` was, bit for bit – in
    the frame the viewer shows (FrameAlphas) and in the exported file."""
    from engine.groups import create_group
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    _object(store, "head", ms["head"], 0)
    _object(store, "torso", ms["torso"], 1)
    _object(store, "legs", ms["legs"], 2)
    create_group(store, "body", ["head", "torso", "legs"], seam_px=SEAM)
    expected = _core_composition(work, ["head", "torso", "legs"], SEAM)
    assert expected[30:34, 80:121].min() == 1.0 and expected[80:84, 80:121].min() == 1.0   # both seams filled
    grp = {**store.get("body"), "tool": {**store.get("body")["tool"], "active_run": CORE_RUN}}
    got = frame_alphas(work, store.list() + [grp], 1, W, H)["body"]
    assert got.dtype == expected.dtype and np.array_equal(got, expected)
    export_objects(work, "shot", store.list() + [grp], [1], W, H, write=["body"])
    ch, _ = read_exr(work / "export" / "mattes" / "body" / "body_0001.exr")
    assert np.array_equal(ch["R"].astype(np.float32), expected)          # 0 / 1 values: exact in half float too


def test_single_export_equals_full_export(tmp_path):
    """Export only the group (or only one member) -> the same matte as in the full export; nothing else written."""
    import shutil
    from engine.groups import create_group
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(store, n, ms[n], i)
    create_group(store, "body", ["head", "torso"], seam_px=SEAM)
    ctx = store.list() + [{**store.get("body"), "tool": {**store.get("body")["tool"], "active_run": CORE_RUN}}]
    export_objects(work, "shot", ctx, [1], W, H)
    root = work / "export" / "mattes"
    full = {n: read_exr(root / n / f"{n}_0001.exr")[0]["R"] for n in ("body", "head", "legs")}
    shutil.rmtree(root)
    for n in ("body", "head", "legs"):
        export_objects(work, "shot", ctx, [1], W, H, write=[n])
        one, _ = read_exr(root / n / f"{n}_0001.exr")
        assert np.array_equal(one["R"], full[n]), n
    assert sorted(p.name for p in root.iterdir()) == ["body", "head", "legs"]
    out = json.loads((work / "outputs.json").read_text(encoding="utf-8"))
    assert "groups" not in out and next(e for e in out["objects"] if e["name"] == "body")["members"] == ["head", "torso"]


# ---------------------------------------------------------------- adapter of the group API (steps 1–4)
class G:
    """The only place (b)–(d) touch the group API. Proposed in step 1, adjusted there to the accepted model."""

    @staticmethod
    def create(work, group, members, seam=SEAM):
        from engine import groups
        groups.create_group(work, group, members, seam_px=seam)

    @staticmethod
    def set_source(work, group, member, source, run=None):
        from engine import groups
        groups.set_member_source(work, group, member, {"source": source, "run": run})

    @staticmethod
    def set_seam(work, group, px):
        from engine import groups
        groups.set_seam(work, group, px)

    @staticmethod
    def add(work, group, member):
        from engine import groups
        groups.add_member(work, group, member)

    @staticmethod
    def alpha(work, group, f):
        from engine import groups
        return groups.group_alpha_loader(work, group, W, H)(f)

    @staticmethod
    def fake_whole_run(work, group, stage, run="v001"):
        """A whole recompute of `stage` done now, written as the pipeline writes it: run.json with the group's input
        signature (core_fix_sig = stages.mask_sig = groups.group_sig) + the edge settings, tool.stage_runs[stage]."""
        from engine import groups
        from engine.stages import mask_sig, settings_sig
        store = ObjectStore(work)
        rd = store.dir(group) / "runs" / run
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "run.json").write_text(json.dumps({
            "stage": stage, "group": groups.members(store, group), "segments_def": [],
            "core_fix_sig": mask_sig(store, group), "settings_sig": settings_sig(store.get(group))}))
        store.update(group, tool={"stage_runs": {stage: run}, "active_run": run})
        return run

    @staticmethod
    def whole_valid(work, group, stage):
        from engine.stages import stage_run
        return stage_run(ObjectStore(work), group, stage)[1]

    @staticmethod
    def detach(work, group, member, confirm):
        from engine import groups
        return groups.remove_member(work, group, member, confirm=confirm)


def _three(tmp_path):
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(store, n, ms[n], i, soft_r=4.0)
    G.create(work, "body", ["head", "torso", "legs"])
    return work, store, ms


# ---------------------------------------------------------------- (b)
def test_b_member_source_brings_exactly_its_alpha(tmp_path):
    """„Do grupy z:” the version v001 (soft, like a MatAnyone2 result) on the torso, Core on the rest: outside the seam
    the composition IS the torso's v001 alpha – independent of the torso's own export choice (its active_run stays Core)."""
    from engine.export import run_alpha_loader
    work, store, ms = _three(tmp_path)
    G.set_source(work, "body", "torso", "version", run="v001")
    assert store.get("torso")["tool"]["active_run"] == CORE_RUN                   # own export choice untouched
    got = G.alpha(work, "body", 1)
    torso = run_alpha_loader(work / "objects" / "torso", "v001", W, H)(1)
    near_other = cv2.dilate((ms["head"] | ms["legs"]).astype(np.uint8), _SEAM_ZONE) > 0
    own = (torso > 0) & ~near_other                                              # the torso's alpha, away from any seam
    assert own.sum() > 1000 and np.array_equal(got[own], torso[own])
    assert 0.0 < got[own].min() < 1.0                                             # really the soft v001, not the core
    # Core members still bring their mask 1+2 bit for bit away from the torso
    core_only = (ms["head"] | ms["legs"]) & ~(cv2.dilate(ms["torso"].astype(np.uint8), _SEAM_ZONE) > 0)
    assert np.array_equal(got[core_only], np.ones(core_only.sum(), np.float32))
    # default source of a new member = Core: the group of Core members is the frozen contract of (a)
    G.set_source(work, "body", "torso", "core")
    assert np.array_equal(G.alpha(work, "body", 1), _core_composition(work, ["head", "torso", "legs"], SEAM))


# ---------------------------------------------------------------- (c)
@pytest.mark.parametrize("change", ["source", "seam", "members_add", "members_remove", "member_off"])
def test_c_group_change_makes_whole_recompute_stale(tmp_path, change):
    work, store, ms = _three(tmp_path)
    G.fake_whole_run(work, "body", "matanyone")
    assert G.whole_valid(work, "body", "matanyone")
    store.update("torso", tool={"active_run": "v001"})              # a member's own export choice: not the group's input
    assert G.whole_valid(work, "body", "matanyone")
    if change == "source":
        G.set_source(work, "body", "torso", "version", run="v001")
    elif change == "seam":
        G.set_seam(work, "body", SEAM + 2)
    elif change == "member_off":   # the palette checkbox (2026-10-29): out of the composition
        from engine.groups import set_member_on
        set_member_on(work, "body", "legs", False)
    elif change == "members_add":
        yy, xx = np.mgrid[0:H, 0:W]
        _object(store, "arm", (abs(xx - 140) <= 8) & (yy >= 40) & (yy < 70), 3)
        G.add(work, "body", "arm")
    else:
        G.detach(work, "body", "legs", confirm=True)      # the whole result goes to the trash (d)
    assert not G.whole_valid(work, "body", "matanyone")


# ---------------------------------------------------------------- (d)
def test_d_detach_with_whole_result_warns_then_trashes(tmp_path):
    work, store, ms = _three(tmp_path)
    G.fake_whole_run(work, "body", "matanyone")
    before = {n: sorted(p.relative_to(store.dir(n)).as_posix() for p in store.dir(n).rglob("*"))
              for n in ("head", "torso", "legs")}
    r = G.detach(work, "body", "legs", confirm=False)
    assert r["needs_confirm"] and "matanyone" in r["whole_stages"] and r["warning"]   # nothing done yet
    assert G.whole_valid(work, "body", "matanyone")
    assert not (work / "trash").exists()
    r = G.detach(work, "body", "legs", confirm=True)
    assert r["success"] and r["trash"]
    trash = work / "trash"
    assert trash.exists() and any(p.name == "run.json" for p in trash.rglob("run.json"))   # the whole result is recoverable
    assert not G.whole_valid(work, "body", "matanyone")
    for n, files in before.items():                                    # members keep their own results untouched
        assert sorted(p.relative_to(store.dir(n)).as_posix() for p in store.dir(n).rglob("*")) == files
    # without a whole result: detaching needs no confirmation
    r = G.detach(work, "body", "torso", confirm=False)
    assert r["success"] and not r.get("needs_confirm")


# ---------------------------------------------------------------- step 3: sources, reasons, preview endpoint
def test_member_runs_say_why_and_preview_endpoint(tmp_path, monkeypatch):
    """„Do grupy z:” without a result = an explicit reason (no silent substitute); the preview PNG = the composition."""
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import app as app_module
    from engine import groups
    work, store, ms = _three(tmp_path)
    assert [r["run"] for r in map(lambda m: groups.member_run(store, m), ("head", "torso", "legs"))] == [CORE_RUN] * 3
    monkeypatch.setitem(app_module.STATE, "work_dir", str(work))
    monkeypatch.setitem(app_module.STATE, "shot_project", SimpleNamespace(plate_size=(W, H)))
    api = TestClient(app_module.app)
    r = api.get("/api/group/body")
    assert r.status_code == 200 and [m["member"] for m in r.json()["members"]] == ["head", "torso", "legs"]
    assert r.json()["seam_px"] == SEAM and r.json()["depth_order"] == 0
    G.set_source(work, "body", "torso", "version", run="v001")
    png = api.get("/api/group/body/frame/1")
    assert png.status_code == 200
    got = cv2.imdecode(np.frombuffer(png.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(got, (np.clip(G.alpha(work, "body", 1), 0, 1) * 255 + 0.5).astype(np.uint8))
    groups.set_member_source(store, "body", "legs", {"source": "matanyone"})    # no stage-3 result
    r = groups.member_run(store, "legs")
    assert r["run"] is None and r["why"]
    bad = api.get("/api/group/body/frame/1")
    assert bad.status_code == 409 and [m["member"] for m in bad.json()["detail"]["missing"]] == ["legs"]
    assert api.get("/api/group/head").status_code == 404                        # an object, not a group
    import shutil
    shutil.rmtree(store.dir("head") / "core")
    groups.set_member_source(store, "body", "legs", {"source": "core"})
    assert "rdzenia" in groups.member_run(store, "head")["why"]


def test_group_is_one_layer_in_the_depth_order(tmp_path):
    """The group stands at its front-most member: an object in front is subtracted from the group, an object behind
    is cut by the group; the group and its members never cut each other."""
    from engine.export import frame_alphas
    work, store, ms = _three(tmp_path)
    yy, xx = np.mgrid[0:H, 0:W]
    _object(store, "leaf", (abs(xx - 100) <= 6) & (abs(yy - 50) <= 6), 0)          # in front of everything
    for i, n in enumerate(("head", "torso", "legs")):
        store.update(n, depth_order=i + 1)
    _object(store, "wall", (yy >= 0) & (xx < 110), 4)                               # behind everything
    objs = [{**o, "tool": {**o["tool"], "active_run": CORE_RUN}} for o in store.list(groups=True)]
    al = frame_alphas(work, objs, 1, W, H)
    comp = G.alpha(work, "body", 1)
    leaf = al["leaf"]
    assert np.allclose(al["body"], np.clip(comp - leaf, 0, 1))                       # the leaf in front cuts the group
    assert al["torso"][60, 100] == 1.0 and al["body"][60, 100] == 1.0                # member and group: no mutual cut
    assert al["wall"][60, 100] == 0.0 and al["wall"][150, 50] == 1.0                 # the group cuts what is behind


def test_member_switched_off_leaves_the_composition(tmp_path):
    """Palette checkbox of a member off (user decision 2026-10-29): the group's alpha = the composition of the members
    still on (bit for bit), the member stays in the group; on again = the full composition."""
    from engine.groups import group_alpha_loader, members, set_member_on
    work, store, ms = _three(tmp_path)
    full = group_alpha_loader(store, "body", W, H)(1)
    set_member_on(work, "body", "legs", False)
    assert members(store, "body") == ["head", "torso", "legs"]
    got = group_alpha_loader(store, "body", W, H)(1)
    assert np.array_equal(got, _core_composition(work, ["head", "torso"], SEAM))
    assert got.sum() < full.sum()
    set_member_on(work, "body", "legs", True)
    assert np.array_equal(group_alpha_loader(store, "body", W, H)(1), full)


# ---------------------------------------------------------------- the seam reach of each member (2026-10-29)
def test_member_seams_on_the_group_value_are_the_single_seam_bit_for_bit(tmp_path):
    """Contract: every member on the group's seam (its own field unset or set to that value) = the composition with
    the one seam of today, bit for bit; the union with `gaps` all equal = the union with `gap_px`."""
    from engine.groups import create_group, group_alpha_loader, group_sig, set_member_seam
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(store, n, ms[n], i)
    create_group(store, "body", ["head", "torso", "legs"], seam_px=SEAM)
    expected = _core_composition(work, ["head", "torso", "legs"], SEAM)
    assert np.array_equal(group_alpha_loader(store, "body", W, H)(1), expected)
    sig = group_sig(store, "body")
    for n in ("head", "torso", "legs"):
        set_member_seam(store, "body", n, SEAM)
    assert np.array_equal(group_alpha_loader(store, "body", W, H)(1), expected)
    assert group_sig(store, "body") != sig                                    # an own value is part of the signature
    for n in ("head", "torso", "legs"):
        set_member_seam(store, "body", n, None)
    assert group_sig(store, "body") == sig                                    # back to the group's seam = as before
    alphas = [np.zeros((H, W), np.float32)] * 3
    cores = [ms[n] for n in ("head", "torso", "legs")]
    assert np.array_equal(seamless_union(alphas, cores, gap_px=SEAM), seamless_union(alphas, cores, gaps=[SEAM] * 3))


def test_a_larger_member_seam_closes_only_its_own_contacts(tmp_path):
    """Legs with a larger seam reach close the slit at the torso / legs contact, the head / torso slit (both on the
    small group seam) stays open – hair keeps its gaps by the face, the dress closes its slit."""
    from engine.groups import create_group, group_alpha_loader, set_member_seam
    work = tmp_path / "matte_desk"
    store = ObjectStore(work)
    ms = _masks()
    for i, n in enumerate(("head", "torso", "legs")):
        _object(store, n, ms[n], i)
    create_group(store, "body", ["head", "torso", "legs"], seam_px=1)
    a = group_alpha_loader(store, "body", W, H)(1)
    assert a[31:33, 100].max() < 1.0 and a[81:83, 100].max() < 1.0            # 4 px slits: both open on 1 px
    set_member_seam(store, "body", "legs", 3)
    b = group_alpha_loader(store, "body", W, H)(1)
    assert np.all(b[80:84, 90:111] == 1.0)                                    # the legs' contact closed
    assert np.array_equal(b[25:40], a[25:40])                                 # the head / torso contact untouched
