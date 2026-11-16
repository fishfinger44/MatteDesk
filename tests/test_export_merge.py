"""Audit 2026-11-02 step 5: `export_mattes` and `_export_by_source` carried ~60 duplicated lines; now one function. The
plain export (source `active`, no version, no group named) must stay what it was, file by file: `_old_export_mattes` is
the removed branch, frozen, and both write the same synthetic project."""
import asyncio
import json
import shutil
from pathlib import Path

import pytest
from fastapi import HTTPException

import app as A
from engine import groups
from test_work_area import _open


async def _old_export_mattes(req):
    """The branch of export_mattes at 396b1e6, verbatim but for the `A.` prefix."""
    from engine.export import export_objects
    from engine.stages import blocked_for_export
    store = A._objects()
    work = Path(A.STATE["work_dir"])
    dn = A._mask_plate_denoise()
    every = store.list()
    want = None
    if req.objects:
        want, needed = [], set()
        names = {o["name"] for o in every}
        for n in req.objects:
            if n not in names:
                raise HTTPException(status_code=404, detail=f"nieznany obiekt / grupa: {n}")
            needed.add(n)
            want.append(n)
        stale = blocked_for_export(store, sorted(needed), work, dn)
    else:
        stale = blocked_for_export(store, [o["name"] for o in every], work, dn)
    if stale:
        raise HTTPException(status_code=409, detail=f"Najpierw „Przelicz finalnie” (eksport nie bierze rdzenia ani nieaktualnego wyniku): {', '.join(stale)}")
    if req.scope not in ("all", "work_area"):
        raise HTTPException(status_code=400, detail="scope = all | work_area")
    frames = A._work_frames() if req.scope == "work_area" else A._shot_frames()
    ok = set(o["name"] for o in every) - set(blocked_for_export(store, [o["name"] for o in every], work, dn))
    objs = [o for o in every if o["name"] in ok]
    from engine.workarea import missing_detail, need_missing, object_life, run_coverage
    if want is None:
        relevant = set(ok)
    else:
        relevant = set(needed)
        back = max((o.get("depth_order", 0) for o in every if o["name"] in needed), default=0)
        relevant |= {o["name"] for o in objs if o.get("depth_order", 0) < back}
    holes = {}
    for n in sorted(relevant & ok):
        miss = need_missing((frames[0], frames[-1]), run_coverage(store.dir(n), store.get(n)["tool"].get("active_run")) or [],
                            object_life(store.get(n)))
        if miss:
            holes[n] = miss
    if holes:
        lbl = "całym ujęciu" if req.scope == "all" else f"zakresie pracy F{frames[0]}–{frames[-1]}"
        raise HTTPException(status_code=409, detail=missing_detail("Wynik finalny", lbl, holes))
    missing_front = []
    if want is not None:
        first = min((o.get("depth_order", 0) for o in every if o["name"] in set(want)), default=0)
        missing_front = [o["name"] for o in every if o["name"] not in ok and o.get("depth_order", 0) < first]
    m = A.STATE["manifest"]
    w, h = A._plate_size()
    depth = None
    if req.with_depth:
        from engine.outputs import load_outputs
        depth = load_outputs(A.STATE["work_dir"]).get("depth", {}).get("path")
    bit = int(m.get("bit_depth", 16))
    from engine.export import EXPORT_FORMATS
    fmt = (req.format or A._export_format_setting()).lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Format musi być jednym z: {', '.join(EXPORT_FORMATS)}")
    cb = A._ws_progress("export_progress", "*")
    out = await asyncio.get_running_loop().run_in_executor(None, lambda: export_objects(
        Path(A.STATE["work_dir"]), m.get("shot", "shot"), objs, frames, w, h, bit, req.multichannel, depth,
        req.with_core, cb, want, export_dir=A._export_dir(), export_format=fmt, selection=req.selection,
        merge_plate=A._merge_plate()))
    from engine.contract import validate
    warn = (f"Obiekty z przodu bez aktualnego wyniku nie zostały odjęte: {', '.join(missing_front)}"
            if missing_front else "")
    return {"success": True, "outputs": out, "contract_errors": validate(out), "written": want or "all", "warning": warn,
            "format": out.get("export_format"), "scope": req.scope, "frames": [frames[0], frames[-1]]}


def _project(tmp_path, monkeypatch):
    """`hand` in front of `puppet`, plus `member` alone in the group `g`: every one with the same computed final result."""
    api, shot, _mf = _open(tmp_path, monkeypatch, n=6)
    assert api.post("/api/object/puppet/compute_all").status_code == 200
    assert api.post("/api/object/puppet/finalize", json={"edge_mode": "core"}).status_code == 200
    store = A._objects()
    for name, depth in (("hand", -2), ("member", -1)):
        shutil.copytree(store.dir("puppet"), store.dir(name))
        p = store.dir(name) / "object.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d.update(name=name, depth_order=depth)
        p.write_text(json.dumps(d), encoding="utf-8")
    groups.create_group(A.STATE["work_dir"], "g", ["member"])
    return api


def _tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _normal(res):
    """A response without what is expected to differ: the date in outputs, and the two keys only the new code gives."""
    res = json.loads(json.dumps(res))
    res.pop("source", None)
    res.pop("runs", None)
    out = res.get("outputs") or {}
    out.pop("updated_at", None)
    for o in out.get("objects") or []:
        o.pop("date", None)
    return res


def _old(body):
    try:
        return 200, _normal(asyncio.run(_old_export_mattes(A.ExportRequest(**body))))
    except HTTPException as e:
        return e.status_code, e.detail


def _new(api, body):
    r = api.post("/api/export", json=body)
    return r.status_code, _normal(r.json()) if r.status_code == 200 else r.json()["detail"]


CASES = [
    {"multichannel": True},
    {"multichannel": False, "format": "png"},
    {"objects": ["puppet"], "format": "exr"},
    {"objects": ["puppet"], "format": "png", "with_core": True},
    {"objects": ["puppet", "hand"], "selection": True, "format": "exr"},
    {"objects": ["nobody"]},
    {"format": "tiff"},
    {"objects": ["puppet"], "scope": "work_area", "format": "png"},
]


def test_plain_export_is_the_old_one_file_by_file(tmp_path, monkeypatch):
    api = _project(tmp_path, monkeypatch)
    root = tmp_path / "work" / "matte_desk"
    out_dir = root / "export"
    seen = set()
    for body in CASES:
        shutil.rmtree(out_dir, ignore_errors=True)
        (root / "outputs.json").unlink(missing_ok=True)
        code_old, res_old = _old(body)
        tree_old = _tree(out_dir) if out_dir.exists() else {}
        out_old = json.loads((root / "outputs.json").read_text(encoding="utf-8")) if (root / "outputs.json").exists() else None
        shutil.rmtree(out_dir, ignore_errors=True)
        (root / "outputs.json").unlink(missing_ok=True)
        code_new, res_new = _new(api, body)
        tree_new = _tree(out_dir) if out_dir.exists() else {}
        out_new = json.loads((root / "outputs.json").read_text(encoding="utf-8")) if (root / "outputs.json").exists() else None
        assert (code_new, res_new) == (code_old, res_old), body
        assert sorted(tree_new) == sorted(tree_old), body
        for k in tree_old:
            if k.endswith(".nk"):      # holds no date, but checked as text for a readable diff
                assert tree_new[k].decode("utf-8") == tree_old[k].decode("utf-8"), (body, k)
            else:
                assert tree_new[k] == tree_old[k], (body, k)
        if out_old is not None:
            out_old.pop("updated_at")
            out_new.pop("updated_at")
            for d in (out_old, out_new):
                for o in d.get("objects") or []:
                    o.pop("date", None)
            assert out_new == out_old, body
        if code_old == 200 and not body.get("objects"):
            assert {o["name"] for o in res_old["outputs"]["objects"]} == {"hand", "member", "puppet"}
        seen.add(code_old)
    assert seen >= {200, 400, 404}


def test_refusals_stay_the_same(tmp_path, monkeypatch):
    api = _project(tmp_path, monkeypatch)
    store = A._objects()
    assert _new(api, {"multichannel": False, "scope": "nope"}) == (400, "scope = all | work_area")
    # a part of the shot computed: 409 with the missing intervals, under the old label
    store.update("hand", life=None)
    tool = store.get("puppet")["tool"]
    store.update("puppet", tool={"final_valid": False})
    code_o, det_o = _old({"multichannel": False})
    code_n, det_n = _new(api, {"multichannel": False})
    assert (code_n, det_n) == (code_o, det_o) and code_n == 409 and "Przelicz finalnie" in det_n and "puppet" in det_n
    code_o, det_o = _old({"objects": ["puppet"]})
    assert _new(api, {"objects": ["puppet"]}) == (code_o, det_o)
    store.update("puppet", tool={"final_valid": tool.get("final_valid", True)})


def test_group_named_or_other_source_still_take_their_way(tmp_path, monkeypatch):
    """The two ways that were already through _export_by_source keep their extras."""
    api = _project(tmp_path, monkeypatch)
    r = api.post("/api/export", json={"objects": ["g"], "format": "png"})
    assert r.status_code == 200, r.text
    assert "source" in r.json() and r.json()["runs"]["g"]
    bad = api.post("/api/export", json={"source": "nope"})
    assert bad.status_code == 400 and "active" in bad.json()["detail"]


def test_export_constants_live_in_the_engine():
    import engine.export as E
    assert E.EXPORT_SOURCES and E.EXPORT_SOURCE_LABEL and E.EXPORT_FALLBACK
    assert A.EXPORT_SOURCES is E.EXPORT_SOURCES
    src = (Path(A.__file__)).read_text(encoding="utf-8")
    assert "EXPORT_SOURCES = (" not in src and "EXPORT_FALLBACK = {" not in src
    assert src.count("export_objects(") == 1, "one call of the writer in app.py"
    for k in E.EXPORT_SOURCES:
        assert k in E.EXPORT_SOURCE_LABEL and k in E.EXPORT_FALLBACK


@pytest.mark.parametrize("src", ["core", "vitmatte"])
def test_other_source_full_export_may_drop_objects_without_refusal(tmp_path, monkeypatch, src):
    """A full export of another source leaves out what has no result – no „Przelicz finalnie” refusal there."""
    api = _project(tmp_path, monkeypatch)
    r = api.post("/api/export", json={"source": src, "format": "png", "multichannel": False})
    assert r.status_code in (200, 409) and "Przelicz finalnie" not in r.text
