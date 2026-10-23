# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Z7 – smoke test of the front end without a browser.

- `node --check` of every static/*.js (skipped with a message when Node is not installed);
- every `<obj>.save()` on a canvas context is followed by `<obj>.restore()` in the same block (the scroll bug: save without
  restore left the transform growing on every render);
- every id used through `$('…')` / `getElementById('…')` exists in index.html or is created by a JS template (`id="…"`);
- every <button> / <select> / <input> of the stage panels has a tooltip (inline `title` or an entry in static/tips.js, Z24),
  apart from an explicit list of exceptions.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
JS = sorted(STATIC.glob("*.js"))
HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def test_node_check_all_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – pomijam `node --check` (asercje tekstowe niżej działają)")
    bad = []
    for p in JS:
        r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{p.name}: {r.stderr.strip()[:300]}")
    assert not bad, "\n".join(bad)


def _strip_strings(src: str) -> str:
    """Comments and quoted strings blanked (same length), so braces inside them do not count. Template literals keep
    their `${…}` structure (balanced)."""
    out, i, n = list(src), 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                out[k] = " " if src[k] != "\n" else "\n"
            i = j
        elif c in "'\"":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def unrestored_saves(src: str):
    """[(line, obj)] of `obj.save()` calls with no `obj.restore()` before their enclosing block closes."""
    s = _strip_strings(src)
    bad = []
    for m in re.finditer(r"\b(\w+)\.save\(\)", s):
        obj, depth, ok = m.group(1), 0, False
        for j in range(m.end(), len(s)):
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    break
            if s.startswith(f"{obj}.restore()", j):
                ok = True
                break
        if not ok:
            bad.append((src.count("\n", 0, m.start()) + 1, obj))
    return bad


@pytest.mark.parametrize("name", ["objects.js", "app.js", "stage0.js", "depth.js", "jobs.js", "zones.js", "brush.js"])
def test_every_canvas_save_is_restored(name):
    src = (STATIC / name).read_text(encoding="utf-8")
    assert unrestored_saves(src) == [], f"{name}: save() bez restore() w tym samym bloku"


def test_save_check_detects_a_missing_restore():
    assert unrestored_saves("function f(){ ctx.save(); ctx.scale(2,2); }") == [(1, "ctx")]
    assert unrestored_saves("function f(){ ctx.save(); try { x(); } finally { ctx.restore(); } }") == []


def _used_ids(src: str):
    return set(re.findall(r"\$\('([\w-]+)'\)", src)) | set(re.findall(r"getElementById\('([\w-]+)'\)", src))


def _defined_ids(html: str, js_sources):
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    for src in js_sources:
        ids |= set(re.findall(r'\bid="([\w-]+)"', src))
        ids |= set(re.findall(r"\.id\s*=\s*'([\w-]+)'", src))
    return ids


def missing_ids(html: str, js_by_name):
    defined = _defined_ids(html, js_by_name.values())
    return {name: sorted(_used_ids(src) - defined) for name, src in js_by_name.items() if _used_ids(src) - defined}


def test_all_used_ids_exist():
    srcs = {p.name: p.read_text(encoding="utf-8") for p in JS}
    assert missing_ids(HTML, srcs) == {}


def test_id_check_detects_a_removed_id():
    srcs = {p.name: p.read_text(encoding="utf-8") for p in JS}
    # an id that only index.html defines and objects.js uses
    used = _used_ids(srcs["objects.js"])
    js_defined = _defined_ids("", srcs.values())
    victim = sorted(i for i in used if i not in js_defined and f'id="{i}"' in HTML)[0]
    broken = HTML.replace(f'id="{victim}"', 'id="__removed__"')
    assert victim in missing_ids(broken, srcs).get("objects.js", [])


# Controls without a tooltip on purpose (self-explanatory or with a visible label right next to them)
NO_TITLE_OK = {
    "st4Edge",            # options carry the tooltips (tips.js OPTION_TIPS), the label has one too
    "z4Crit",             # same
    "brushSize", "brushSizeN", "penPressure",   # inside the titled .tool-radius box / label
    "st1PtsRedo",         # titled inline
}


def _panel_controls(src: str):
    """(id, has_inline_title) of <button|select|input id="…"> in JS template strings."""
    out = []
    for m in re.finditer(r"<(button|select|input)\b([^>]*)>", src):
        attrs = m.group(2)
        mid = re.search(r'\bid="([\w-]+)"', attrs)
        if not mid or "${" in mid.group(1):
            continue
        out.append((mid.group(1), "title=" in attrs))
    return out


def test_panel_controls_have_tooltips():
    tips = (STATIC / "tips.js").read_text(encoding="utf-8")
    tip_ids = set(re.findall(r"^\s{4}(\w+):", tips, re.M))
    missing = []
    for name in ("objects.js", "stage0.js", "zones.js", "brush.js"):
        src = (STATIC / name).read_text(encoding="utf-8")
        for cid, titled in _panel_controls(src):
            if not titled and cid not in tip_ids and cid not in NO_TITLE_OK:
                missing.append(f"{name}:{cid}")
    assert not missing, f"brak podpowiedzi (title albo static/tips.js): {missing}"


# ---------------------------------------------------------------- stage-4 zone panel (static/zones.js, plan step 5)
ZONE_PANEL_IDS = ["st4ZonePanel", "z4Panel", "z4Adv", "st4Zones", "z4New", "z4Presets", "z4Prompt", "z4Engine", "z4Holdout",
                  "z4CleanRow", "z4CleanTxt", "z4CpAuto", "z4CpFile", "z4CpNone", "z4Ov", "z4Refresh", "z4Loupe", "z4PvInfo",
                  "z4ApplyShot", "z4KeyDel", "timelineZones", "zoneFrontModal"]


def test_zone_panel_ids_and_hooks():
    zones = (STATIC / "zones.js").read_text(encoding="utf-8")
    objects = (STATIC / "objects.js").read_text(encoding="utf-8")
    defined = _defined_ids(HTML, [zones, objects])
    missing = [i for i in ZONE_PANEL_IDS if i not in defined]
    assert not missing, f"brak id panelu Strefy: {missing}"
    assert 'src="/static/zones.js"' in HTML and HTML.index("objects.js") < HTML.index("zones.js")
    # the panel's public API used by objects.js exists in zones.js
    for fn in ("init", "load", "reset", "draw", "drawLoupe", "renderLane", "onFrame", "wantsDraw", "onDraw", "undo", "runDetailsHtml", "flush"):
        assert re.search(rf"\b{fn}\b", zones), fn
        assert f"MD_ZONES.{fn}" in objects or fn in ("load", "reset", "draw", "drawLoupe", "renderLane", "onFrame", "wantsDraw", "onDraw", "undo", "runDetailsHtml", "flush", "init"), fn
    for call in ("MD_ZONES.init(", "MD_ZONES.load()", "MD_ZONES.draw(", "MD_ZONES.drawLoupe(", "MD_ZONES.onDraw(", "MD_ZONES.undo(", "MD_ZONES.flush()", "MD_ZONES.runDetailsHtml("):
        assert call in objects, call
    # the API routes the panel runs on
    for route in ("/preview/${f}", "/keys/${f}", "/region/${f}", "/apply?stage=vitmatte`", "/api/project/cleanplate_auto", "/api/queue/preset/all", "/api/queue/run",
                  "'PATCH'", "zone_alpha", "zone_trimap", "zone_engines"):
        assert route in zones, route
    # overlay views and presets of the engines (names must match engine/localmatte.py)
    for p in ("hair", "fibres", "smoke", "motion", "thin"):
        assert f"['{p}'" in zones, p
    assert '<option value="matanyone">' in zones and zones.index('value="matanyone"') < zones.index('value="auto"'), "matanyone = domyślny silnik"


def test_zone_preview_invalidates_only_its_frame():
    """Plan step 5 pkt 6 / Z20: after an automatic preview only the current frame's overlays are dropped."""
    zones = (STATIC / "zones.js").read_text(encoding="utf-8")
    body = _strip_strings(zones[zones.index("function setPreview("):zones.index("function refreshPreview(")])
    assert "invalidateFrame(f)" in body and "clearOverlays(" not in body
    # the preview debounce = 600 ms like the stage-3 brush key
    assert "delay == null ? 600" in zones


def test_no_native_confirm():
    """Chrome may block native dialogs – every yes/no question goes through viewer.uiConfirm / uiDialog."""
    native = re.compile(r"(?<![\w.])confirm\(|window\.confirm\(")
    bad = [f"{p.name}:{i}" for p in JS for i, line in enumerate(_strip_strings(p.read_text(encoding="utf-8")).splitlines(), 1)
           if native.search(line)]
    assert not bad, bad


def test_other_objects_layers_wired():
    """Other computed objects as extra overlay layers (user idea 2026-10-15): drawn under the selected object, never in
    Comp, per-object colour / opacity / on-off kept in localStorage, panel refreshed when the selection changes."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "if (!S.auxMode) drawOthers(ctx, img, f);" in src
    draw = src[src.index("function drawOverlay()"):]
    assert draw.index("drawOthers(ctx, img, f)") < draw.index("overlayFor('alpha', f)")   # under the selected object's layer
    assert "S.view === 'none' || S.view === 'core' || S.view === 'alpha'" in src   # not in Comp
    assert "md_other_layers" in src and "await refreshOthersHas();" in src
    for cls in ("ol-eye", "ol-color", "ol-op", "olAll"):
        assert cls in src
    # floating palette (user request 2026-10-15): dragged by the header, position saved, kept inside the view
    assert "function makeFloating(box, w, head, pos)" in src and "makeFloating(box, w, '.ol-head', othersPos)" in src
    assert "pos: saved.pos || null" in src and "Math.min(Math.max(0, p.x), maxX)" in src


def test_new_and_delete_object_dialogs_wired():
    """New object = dialog in the middle (name + optional prompt), delete = 🗑 on the objects list / next to "+" with a
    confirmation listing what goes to the trash; Enter never deletes."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "$('objNewBtn').addEventListener('click', () => newObjectDialog());" in src
    assert "st1NewBox" not in src and "st1NewName" not in src
    assert "OBJ_NAME_RE = /^[A-Za-z0-9_\-]{1,64}$/" in src
    objects_py = (Path(__file__).resolve().parents[1] / "engine" / "objects.py").read_text(encoding="utf-8")
    assert 'r"^[A-Za-z0-9_\-]{1,64}$"' in objects_py               # same name rule as the server
    # delete: 🗑 per row of the floating layers palette (2026-10-15, was data-del on the stage-1 list) + next to "+"
    assert "class=\"ol-del\"" in src and "deleteObjectDialog(o.name)" in src and "$('objDelBtn').addEventListener" in src
    dlg = src[src.index("async function deleteObjectDialog"):src.index("async function refreshList")]
    assert "delete_info" in dlg and "?exports=1" in dlg and "Enter does NOT delete" in dlg


def test_kadruj_is_automatic_przelicz_calosc():
    """2026-10-15: Kadruj is automatic – no stage of its own (the crops live in a collapsed box of stage SAM3),
    „▶▶ Przelicz całość” in stage SAM3 (panel + palette) calls POST /compute_all; nothing crops by itself."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "autoCrop" not in src and "id: 'crops'" not in src and 'id="st1CropsBox"' in src
    assert "compute_all" in src and src.count("run('st1All', computeAll)") == 1 and "run('ppAll', computeAll)" in src
    assert src.count("S.view === 'none' && S.stage === ST.SAM3) return 'key'") == 2


def test_stage_layout_2026_10_15():
    """User decision 2026-10-15: 0 Plate'y · 1 SAM3 · 2 CoTracker · 3 MatAnyone · 4 ViTMatte · 5 Eksperymenty · 6 Eksport;
    2026-10-16: 6 LTX Alpha-Gen (out of the experiments' edge modes) before 7 Eksport.
    Stage numbers are constants (STAGES order -> ST.<key>), never literals; every stage pane starts with a description
    (what it does / what it is best for); stage 2 = static/brush.js, stage 3 = finalize stage=matanyone, stage 4 = apply
    ?stage=vitmatte, export with a source."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    keys = re.findall(r"\{ key: '(\w+)', id: '(\w+)'", src)
    assert keys == [("PLATES", "plates"), ("SAM3", "sam3"), ("BRUSH", "cotracker"), ("MATANYONE", "matanyone"),
                    ("VIDEOMAMA", "videomama"), ("VITMATTE", "vitmatte"), ("EXPERIMENTS", "final"), ("LTX", "ltx"),
                    ("FINAL_TOUCH", "final_touch"), ("EXPORT", "export")]
    # 2026-10-16: 4 VideoMaMa between MatAnyone and ViTMatte; 2026-10-22: 8 Final Touch (hand corrections) before Eksport
    assert "const ST = Object.fromEntries(STAGES.map((s) => [s.key, s.n]))" in src
    code = _strip_strings(src)
    assert not re.search(r"S\.stage\s*[=!<>]==?\s*\d", code), "numer etapu wpisany na sztywno (użyj ST.<klucz>)"
    assert not re.search(r"setStage\(\d", code)
    for z in ("zones.js", "brush.js"):
        zc = _strip_strings((STATIC / z).read_text(encoding="utf-8"))
        assert not re.search(r"stage\s*[=!]==\s*\d", zc), z
    # every stage pane has its description at the top, every object stage has one in STAGES
    assert src.count("${descHtml('") == 9
    assert all(f"key: '{k}'" in src and "desc: '<b>Co robi:</b>" in src
               for k in ("SAM3", "BRUSH", "MATANYONE", "VIDEOMAMA", "VITMATTE", "EXPERIMENTS", "LTX", "FINAL_TOUCH", "EXPORT"))
    assert src.count("desc: '<b>Co robi:</b>") == 9 and src.count("<b>Najlepsze do:</b>") >= 8
    # stage Final Touch: static/finaltouch.js, its pane, export source with base + corrections, Alpha = base + corrections
    assert '<script src="/static/finaltouch.js"></script>' in HTML and HTML.index("brush.js") < HTML.index("finaltouch.js") < HTML.index("tips.js")
    ft = (STATIC / "finaltouch.js").read_text(encoding="utf-8")
    assert "/final_touch/frames/${f}`" in ft and "/final_touch/frame/${f}?b=" in ft and "MD_FT.ready = true" in ft
    assert 'id="ftPanel"' in src and '<option value="final_touch"' in src and "window.MD_FT.matteRef(f)" in src
    assert "window.MD_FT.onDraw(d.f)" in src and "window.MD_FT.undo(f)" in src and "window.MD_FT.deleteKeysAt(f)" in src
    assert not re.search(r"stage\s*[=!]==\s*\d", _strip_strings(ft))
    # stage VideoMaMa: its own pane + palette, export source, ViTMatte base
    assert "/videomama`" in src and 'id="vmRun"' in src and 'id="vmStagePv"' in src and "videomama_preview" in src
    assert '<option value="videomama"' in src and src.count('<option value="videomama"') >= 2
    # stage LTX: its own pane + export source; no longer an edge mode of the experiments
    assert "/ltx`" in src and 'id="lx6Run"' in src and '<option value="ltx"' in src
    assert "rdzeń / LTX)" not in src
    # stage 2 / 3 / 4 / 6 wiring
    assert '<script src="/static/brush.js"></script>' in HTML and HTML.index("zones.js") < HTML.index("brush.js") < HTML.index("tips.js")
    brush = (STATIC / "brush.js").read_text(encoding="utf-8")
    assert "source: 'brush'" in brush and "criterion: 'all'" in brush and "track: B.track ? 'local' : 'rigid'" in brush
    assert "/core_fix`" in brush and "/keys/${f}`" in brush and "/region/${f}`" in brush
    assert "MD_BRUSH.init(hooks)" in src and "window.MD_BRUSH.onDraw(d.f)" in src and "window.MD_BRUSH.undo(f)" in src
    assert "{ stage: 'matanyone' }" in src and "'core_fix'" in src
    assert "source: src" in src and 'id="e6Source"' in src and "e6Run" in src
    zones = (STATIC / "zones.js").read_text(encoding="utf-8")
    assert "S().stage === H.ST.VITMATTE" in zones and "params.source === 'brush'" in zones

def test_points_compute_only_on_przelicz():
    """User request 2026-10-15: a click only places a point; SAM3 runs on „▶ Przelicz” (floating palette, stage-1 panel,
    Enter); „✕ Wyczyść punkty” clears the frame; nothing computes by itself after a click (no timer, no re-run)."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "schedulePoints" not in src and "S.ptsTimer = setTimeout" not in src
    assert "if (S.pointsDirty[f] && !S.pointsFailed[f]) flushPoints(f);" not in src
    assert "makeFloating(box, w, '.pp-head', ptsPos)" in src and "md_points_palette" in src
    for i in ("ppCompute", "ppClear", "pointsPalette"):
        assert f'id="{i}"' in src
    assert "computePoints(f)" in src and "e.target.tagName !== 'BUTTON'" in src
    assert "await flushPendingPoints();" in src          # points still become keys before any computation


def test_points_palette_layout_v2():
    """User request 2026-10-15 (v2): Punkt + / − on the palette under the info line (not in the left bar in stage 1),
    the prompt above „▶ Przelicz klatkę” / „✕ Wyczyść punkty”, no „Wykryj kandydatów"."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    pal = src[src.index("function buildPointsPalette()"):src.index("function renderPointsPalette()")]
    order = [pal.index(x) for x in ('id="ppInfo"', 'data-t="pt_add"', 'id="st1Prompt"', 'id="ppCompute"', 'id="ppAll"')]
    assert order == sorted(order)
    assert "▶ Przelicz klatkę" in pal and 'id="st1Detect"' not in src and "$('st1Detect').addEventListener" not in src
    assert "if (S.stage === ST.SAM3) {   // Punkt + / − are on the floating palette" in src
    init = src[src.index("    buildPanes();"):]
    assert init.index("buildPointsPalette();") < init.index("wire();")


def test_next_button_goes_to_the_next_stage():
    """„Dalej” goes to the next stage of STAGES (no hidden Kadruj any more, 2026-10-15)."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "const nx = STAGES[st.n + 1];" in src and "Dalej → ${nx.d}. ${nx.label}" in src
    assert "addEventListener('click', () => setStage(st.n + 1))" in src


def test_work_area_timeline_and_shortcuts():
    """Work area (2026-10-15): brackets on the timeline (static/workarea.js, loaded), Shift+B / Shift+N / Alt+X in the
    shortcuts table; plain B / N of objects.js (A/B versions, next QA flag) ignore Shift; export has the scope choice."""
    assert '<script src="/static/workarea.js"></script>' in HTML
    assert "<kbd>Shift+B</kbd>" in HTML and "<kbd>Shift+N</kbd>" in HTML and "<kbd>Alt+X</kbd>" in HTML
    wa = (STATIC / "workarea.js").read_text(encoding="utf-8")
    assert "/api/project/work_area" in wa and "wa-handle" in wa and "KeyX" in wa
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "e.code === 'KeyN' && !e.shiftKey" in obj and "e.code === 'KeyB' && !e.ctrlKey && !e.metaKey && !e.shiftKey" in obj
    assert 'id="e5Scope"' in obj and "scope: $('e5Scope').value" in obj and "range_missing" in obj


def test_object_life_on_keys_lane():
    """Object life range (2026-10-15): static/life.js loaded; bands + [ ] handles over the object's keys lane, buttons
    „od / do tej klatki” in stages 1 / 3, hint dialog from the core; objects.js announces the stages and dims keys
    outside the life."""
    assert '<script src="/static/life.js"></script>' in HTML
    js = (STATIC / "life.js").read_text(encoding="utf-8")
    assert "/api/object/${encodeURIComponent(L.obj)}/life" in js and "life-handle" in js and "life-dim" in js
    assert "uiConfirm" in js and "md-object-stages" in js
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "new CustomEvent('md-object-stages'" in obj and "window.MD_OBJ = " in obj and "life-hidden" in obj


def test_one_mask_source_chooser_2026_10_21():
    """User request 2026-10-21: the base mask is chosen the same way in every stage – one helper (fillMaskSelect, same
    order / names / greyed-out state); each stage passes only the earlier stages it may use."""
    js = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "function fillMaskSelect(" in js and "function expBaseOptions" not in js
    for call in ("fillMaskSelect($('vmBase'), ['core', 'matanyone']",
                 "fillMaskSelect(bs, ['core', 'matanyone', 'videomama']",
                 "fillMaskSelect(sel, ['core', 'matanyone', 'vitmatte'], { versions: 'experiment'",
                 "fillMaskSelect(sel, ['core', 'matanyone', 'videomama', 'vitmatte', 'ltx'], { value: 'run', versions: 'all'",
                 "fillMaskSelect(sel, ['core', 'matanyone', 'videomama', 'vitmatte', 'ltx'], { avail:"):
        assert call in js, call
    order = [js.index(f"{{ k: '{k}', label:") for k in ("core", "matanyone", "videomama", "vitmatte", "ltx")]
    assert order == sorted(order)                    # one order everywhere: 1+2, 3, 4, 5, 7


def test_band_preview_lines_only_2026_10_23():
    """One band system (user decision 2026-10-22, report 2026-10-23): the band preview of stages 3 MatAnyone2 /
    4 VideoMaMa / 5 ViTMatte / 7 LTX-2.5 is dashed lines only – no yellow / blue fill, also while a slider is held, and no
    old yellow banner „Podgląd pasa krawędzi (…) – przed liczeniem”."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    zon = (STATIC / "zones.js").read_text(encoding="utf-8")
    bp = (STATIC / "bandpoints.js").read_text(encoding="utf-8")
    # the old fill colours of the band (yellow 250,204,21 / blue 80,160,255) are gone from the band canvases
    assert "function edgePvCanvas" not in obj and "px[i] = 250; px[i + 1] = 204" not in obj
    ins = zon[zon.index("function insetCanvas("):zon.index("function bandPngEdges(")]
    assert "p[o] = 250" not in ins and "p[o] = 80" not in ins and "bandEdgeCanvas(cls" in ins
    assert "bandPngEdges," in zon and "window.MD_ZONES.bandPngEdges(ov)" in obj
    # no banner from the stage-3 preview, no yellow / blue legend notes in stages 4 / 7
    assert "przed liczeniem; wyłącz „👁 podgląd” w palecie Krawędź`" not in obj
    assert "żółte = MatAnyone2 decyduje" not in obj and "Pas VideoMaMa: żółte" not in obj and "Pas LTX-2.5: żółte" not in obj
    # a held slider draws the server lines (live, one request at a time) and falls back to the browser's band edges
    assert "function drawLines(ctx, img, f, params, opts = {})" in bp and "opts.live" in bp
    edge = obj[obj.index("function drawEdgePreview("):obj.index("function expPaletteHtml(")]
    assert "drawLines(ctx, img, f, lp, { live: true })" in edge and "return '';" in edge
    assert "drawLines(ctx, img, f, lp2, { live: true })" in obj
    assert "drawLines(ctx, img, f, lp, { live: S.pvDrag === 'ltx' })" in obj


def test_overlay_cache_keeps_newest_2026_10_23():
    """Derived overlay canvases (tint / black-white / band edges): the cache keeps the NEWEST entries (measured
    2026-10-23: the old `{ [ck]: c, ...slice(-12) }` kept the 12 oldest, every render of stage 4 recomputed its
    overlays on the CPU, ~46 ms)."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    zon = (STATIC / "zones.js").read_text(encoding="utf-8")
    for js in (obj, zon):
        assert not re.search(r"compCache = \{ \[\w+\]: c, \.\.\.", js)
    assert "function putComp(ck, c)" in obj and "slice(-(COMP_KEEP - 1))), [ck]: c }" in obj
    assert "tintCanvas, putComp," in obj and "H.putComp(key, c)" in zon
