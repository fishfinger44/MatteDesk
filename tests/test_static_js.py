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


@pytest.mark.parametrize("name", ["objects.js", "app.js", "stage0.js", "depth.js", "jobs.js", "zones.js", "brush.js",
                                  "brushcore.js", "cleanup3d_areas.js"])
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
        for m in re.finditer(r"bandRow\('\w+', '[^']*', '([\w-]+)', '([\w-]+)'", src):   # the shared band rows (2026-10-24)
            ids |= {m.group(1), m.group(2)}
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


def test_outline_zone_retries_after_transient_error_2026_11_03():
    """Audit 2026-11-02 #17: the automatic whole-outline zone. A 409 (a job on the object) or a network error only waits:
    a retry at the end of a job, on entering the stage or on an object reload; only another refusal stays until reload.
    The retry never runs from the failing render itself (no POST loop) and never on a list not loaded yet."""
    zones = (STATIC / "zones.js").read_text(encoding="utf-8")
    fn = zones[zones.index("async function ensureOutlineZone()"):zones.index("function reset() {")]
    assert re.search(r"e\.status\s*==\s*null\s*\|\|\s*e\.status\s*===\s*409", fn), "transient = network (no status) or 409"
    assert "Z.autoWait" in fn and "Z.autoFailed" in fn
    want = zones[zones.index("const wantsAutoZone"):zones.index("async function ensureOutlineZone()")]
    assert "Z.autoWait" in want and "Z.autoFailed" in want, "a waiting object does not auto-create from a render"
    assert re.search(r"addEventListener\('matte-job-finished',.*retryOutlineZone", zones), "retry at the end of a job"
    stage = zones[zones.index("onStage() {"):zones.index("onStages() {")]
    assert "retryOutlineZone" in stage, "retry on entering the stage"
    reset = zones[zones.index("function reset() {"):zones.index("async function loadCleanplate()")]
    assert "Z.loadedObj = null" in reset and "autoWait" in reset, "object reload: retry, but only after the list is loaded"


def test_one_websocket_2026_11_02():
    """Audit 2026-11-02 #14: one server event stream (app.js initWebSocket, with reconnect) – the other modules listen to
    the `md-ws` DOM event; a second bare `new WebSocket` went silent after every server restart."""
    bad = [p.name for p in JS if p.name != "app.js" and "new WebSocket(" in _strip_strings(p.read_text(encoding="utf-8"))]
    assert not bad, bad
    assert "new CustomEvent('md-ws'" in (STATIC / "app.js").read_text(encoding="utf-8")
    assert "addEventListener('md-ws'" in (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "addEventListener('md-ws'" in (STATIC / "depth.js").read_text(encoding="utf-8")


def test_frame_range_keeps_frame_zero_2026_11_02():
    """Audit 2026-11-02 #13: a shot starting at frame 0 (`m.first === 0`) must not become 1 (`||`)."""
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "m.first ?? 1" in app and "m.last ?? 100" in app
    assert "m.first || 1" not in app


def _html_templates(src):
    """(line, text, ${} expressions) of the template literals of `src`, nested ones too."""
    out, n = [], len(src)

    def tpl(i):
        start, text, exprs = i, [], []
        i += 1
        while i < n and src[i] != "`":
            if src[i] == "\\":
                i += 2
            elif src.startswith("${", i):
                j, depth = i + 2, 1
                while j < n and depth:
                    c = src[j]
                    if c == "`":
                        j = tpl(j)
                        continue
                    if c in "'\"":
                        q, j = c, j + 1
                        while j < n and src[j] != q and src[j] != "\n":
                            j += 2 if src[j] == "\\" else 1
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                    j += 1
                exprs.append(src[i + 2:j - 1])
                i = j
            else:
                text.append(src[i])
                i += 1
        out.append((src.count("\n", 0, start) + 1, "".join(text), exprs))
        return i + 1

    i = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "'\"":
            q, i = c, i + 1
            while i < n and src[i] != q and src[i] != "\n":
                i += 2 if src[i] == "\\" else 1
            i += 1
        elif c == "`":
            i = tpl(i)
        elif c == "/" and i and src[i - 1] in "(,=:!&|?{};\n ":   # a regex literal, roughly
            j = i + 1
            while j < n and src[j] != "/" and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            i = j + 1 if j < n and src[j] == "/" else i + 1
        else:
            i += 1
    return out


def test_names_in_html_go_through_esc_2026_11_02():
    """Audit 2026-11-02 #18: a name, label or server detail printed into HTML goes through esc() (a plate named `<x` broke
    the card). A condition on a name (`sel.has(o.name) ? 'checked' : ''`) prints no name: only what a ternary gives counts."""
    bad = []
    for p in JS:
        for line, text, exprs in _html_templates(p.read_text(encoding="utf-8")):
            if "<" not in text:
                continue
            for e in exprs:
                shown = e.split("?", 1)[1] if "?" in e and "?." not in e and "??" not in e else e
                if re.search(r"\.(name|label|detail)\b", shown) and "esc(" not in shown and "`" not in shown:
                    bad.append(f"{p.name}:{line}: {e[:80]}")
    assert not bad, "\n".join(bad)


def test_no_native_dialogs_2026_11_02():
    """Chrome lets the user block page dialogs, then messages vanish silently: UI.dialog / UI.form, never prompt / confirm."""
    for p in JS:
        src = _strip_strings(p.read_text(encoding="utf-8"))
        assert not re.search(r"(?<![\w.])(window\.)?(prompt|confirm)\(", src), p.name


def test_one_api_toast_dialog_2026_11_02():
    """Audit 2026-11-02 step 7: one api(), one toast, one dialog, in static/ui.js, loaded before every other script."""
    html = HTML
    assert html.index('/static/ui.js') < html.index('/static/i18n.js')
    ui = (STATIC / "ui.js").read_text(encoding="utf-8")
    assert "window.UI = " in ui
    for name in ("api", "read", "toast", "dialog", "form"):
        assert re.search(rf"function {name}\(", ui), name
    all_js = {p.name: p.read_text(encoding="utf-8") for p in JS}
    for name, src in all_js.items():
        # no `(await fetch(x)).json()` – it reads an error body as if it were the answer (no res.ok)
        assert not re.search(r"\(await fetch\([^\n]*\)\)\.json\(\)", src), name
        if name == "ui.js":
            continue
        assert "md-modal-bg" not in src and "jobErrModal" not in src and "jobToast" not in src, name
        assert "getElementById('mdToast')" not in src, name
    assert "md-modal-bg" not in (STATIC / "index.css").read_text(encoding="utf-8")
    assert "async function postJSON" not in all_js["report.js"]
    obj = all_js["objects.js"]
    assert "const api = window.UI.api;" in obj
    app = all_js["app.js"]
    assert "showToast(msg, ms = 3500) { window.UI.toast(msg, ms); }" in app


def test_colours_are_tokens_2026_11_02():
    """A colour equal to a token is the token; --footer-height is defined once (it was twice, the second one won)."""
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    root = css[css.index(":root {"):css.index("}", css.index(":root {"))]
    tokens = {m.group(2).lower(): m.group(1) for m in re.finditer(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b", root)}
    rest = css[:css.index(":root {")] + css[css.index("}", css.index(":root {")):]
    used = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b", rest)} & set(tokens)
    assert not used, sorted(used)
    assert len(re.findall(r"--footer-height:", css)) == 1
    assert "--footer-height: 112px" in root


def test_refresh_cycle_2026_11_02():
    """loadStages renders each panel once (a whole line was doubled, every render ran twice); loadObject asks in parallel;
    a change of the compared plate is md-compare-ref, md-plate is the plate itself."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    body = obj[obj.index("async function loadStages()"):]
    body = body[:body.index("window.dispatchEvent(new CustomEvent('md-object-stages'")]
    for fn in ("renderStageBar", "updateViewTabs", "updateStageButtons", "renderSegsLane", "renderPointsPalette", "renderStageInfo",
               "renderExportSource", "renderEdgePalette", "renderLtxPalette", "renderVmmPalette", "renderExpPalette"):
        assert body.count(f"{fn}()") == 1, fn
    lo = obj[obj.index("async function loadObject()"):]
    lo = lo[:lo.index("function fillObjectForm()")]
    assert "Promise.all(" in lo and "await loadEdge()" not in lo
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "prev: this.activePlate } }));" not in app and "new CustomEvent('md-compare-ref'" in app
    assert "'md-compare-ref'" in (STATIC / "stage0.js").read_text(encoding="utf-8")


def test_job_windows_let_browser_keys_through_2026_11_02():
    """Audit 2026-11-02 #16: the progress / error windows swallow plain shortcuts only – Ctrl/Meta/Alt combinations and
    F-keys (F5, F12, Ctrl+L, Ctrl+Z of the sliders) reach the browser."""
    js = (STATIC / "jobs.js").read_text(encoding="utf-8")
    assert js.count("if (browserKey(e)) return;") == 1           # the progress window; the error window is UI.dialog
    assert "e.ctrlKey || e.metaKey || e.altKey" in js
    ui = (STATIC / "ui.js").read_text(encoding="utf-8")
    assert "if (browserKey(e)) return;" in ui and "e.ctrlKey || e.metaKey || e.altKey" in ui


def test_letter_shortcuts_ignore_ctrl_2026_11_02():
    """Audit 2026-11-02 #19: single-letter tool shortcuts never fire with Ctrl/Meta (Ctrl+P = print, Ctrl+V = paste)."""
    js = (STATIC / "objects.js").read_text(encoding="utf-8")
    for key in ("KeyG", "KeyK", "KeyL", "KeyP", "KeyV", "KeyN"):
        hits = [l for l in js.splitlines() if f"e.code === '{key}'" in l and "else if" in l]
        assert hits, key
        assert all("plain" in l or "!e.ctrlKey" in l for l in hits), (key, hits)


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
    confirmation listing what is deleted for good (UI.confirmDelete); Enter never deletes."""
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
    „▶▶ Przelicz nieaktualne” in stage SAM3 (panel + palette) calls POST /compute_all; nothing crops by itself."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "autoCrop" not in src and "id: 'crops'" not in src and 'id="st1CropsBox"' in src
    assert "compute_all" in src and src.count("run('st1All', computeAll)") == 1 and "run('ppAll', computeAll)" in src
    # 2026-10-26: the plate in stages SAM 3.1 / CoTracker3 shows the core like the Rdzeń view (not only the key frames)
    assert "S.view === 'none' && (S.stage === ST.SAM3 || S.stage === ST.BRUSH)" in src
    assert src.count("if (coreView()) return 'core';") == 2 and "return 'key';" not in src


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


def test_gpu_overlays_with_cpu_fallback():
    """Viewport overlays on the GPU (2026-10-23): static/gpu_overlay.js loaded before app.js / objects.js; tint / bw /
    comp try mdGpu first and keep the CPU loop as the fallback (no WebGL, context lost, md_gpu_overlay = '0')."""
    assert HTML.index('<script src="/static/gpu_overlay.js"></script>') < HTML.index('<script src="/static/app.js"></script>')
    js = (STATIC / "gpu_overlay.js").read_text(encoding="utf-8")
    assert "window.mdGpu" in js and "webglcontextlost" in js and "md_gpu_overlay" in js
    assert "premultipliedAlpha: true" in js and "floor(rgb * a / 255. + .5)" in js   # same stored values as putImageData
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "mdGpu.line(ov, rgb) : mdGpu.tint(ov, rgb)" in obj and "mdGpu.bw(ov)" in obj and "mdGpu.comp(img, ov, c" in obj
    assert obj.count("getImageData(0, 0, c.width, c.height)") >= 3   # the CPU paths are still there
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "mdGpu.comp(beauty, m.img, c, W, H)" in app


def test_spill_warning_in_the_progress_window_2026_10_26():
    """User request 2026-10-26: GPU memory spilled to RAM during a Comfy prompt (comfy_info.spill_gb, comfy_client) shows
    as a warning row in the progress window, next to „Przerwij”."""
    js = (STATIC / "jobs.js").read_text(encoding="utf-8")
    assert 'id="jmSpill"' in js and "const sp = (j.comfy_info || {}).spill_gb;" in js
    assert "przelało się do RAM" in js and "Przerwij" in js


def test_band_keys_are_of_the_band_not_the_frame_2026_10_26():
    """User report 2026-10-26: the band key triangles hid while scrubbing – they came from the current frame's points
    (empty until loaded, occluded points missing). The lane reads the stored band points' keys (GET …/params), cached per
    object + stage and dropped with every band change (dirty / reset)."""
    bp = (STATIC / "bandpoints.js").read_text(encoding="utf-8")
    assert "H.api('GET', `${base()}/params`)" in bp and "Object.keys(p.keys || {}).map(Number)" in bp
    assert "if (B.keyset && B.keyset.k === owner()) return B.keyset.keys;" in bp
    assert "B.keyset = null; dirtyLines();" in bp and "function reset() { B.data = {}; B.keyset = null;" in bp


def test_one_band_drawing_2026_10_24():
    """One band path (user request 2026-10-24): every band preview is drawn by MD_BANDPTS.strokeLines (dashed band
    edges + thin feather-end lines) – stage 5 zones.js no longer strokes its own yellow lines; stage 3 shows a refused
    lines request (lineNote) instead of „ładuję linie pasa…” for ever."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    zon = (STATIC / "zones.js").read_text(encoding="utf-8")
    bp = (STATIC / "bandpoints.js").read_text(encoding="utf-8")
    assert "function strokeLines(ctx, k, L, opts = {})" in bp and "strokeLines(ctx, k, L, { alpha: 0.5 })" in bp   # the previous lines dimmed until the new arrive (2026-11-11)
    assert "function dirtyLines() { B.lines = {}; }" in bp
    assert "rgba(255,212,0" not in zon and "window.MD_BANDPTS.strokeLines(ctx, k, reg)" in zon
    assert bp.count("rgba(255,212,0") == 2                       # the band edge and the feather ends, once
    assert "function lineNote(f, params)" in bp and "if (L && L.error) return false;" in bp
    assert "window.MD_BANDPTS.lineNote(f, lp) ||" in obj
    assert "if (e.status === 409) B.data[k] = { frame: f, points: [], defaults: {}, refused: e.message };" in bp   # no request storm


def test_one_band_palette_2026_10_24():
    """One band palette (user decision 2026-10-24): stages 3 / 4 / 7 build their four band sliders from
    MD_BANDPTS.slider (one label / range / wording) in the one order out, in, feather, outFeather; the stage-5 zone rows
    follow the same order and ranges; the zone's own band-point editor is gone (static/bandpoints.js edits every band)."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    zon = (STATIC / "zones.js").read_text(encoding="utf-8")
    bp = (STATIC / "bandpoints.js").read_text(encoding="utf-8")
    assert "SLIDER_ORDER = ['out', 'in', 'feather', 'outFeather']" in bp and "function slider(kind, model)" in bp
    rows = re.findall(r"bandRow\('(\w+)', '([^']*)', '([\w-]+)', '([\w-]+)'", obj)
    by_model = {}
    for kind, model, rid, nid in rows:
        by_model.setdefault(model, []).append(kind)
    assert set(by_model) == {"MatAnyone2", "VideoMaMa", "LTX-2.5"}
    assert all(v == ["out", "in", "feather", "outFeather"] for v in by_model.values()), by_model
    # the zone palette: the same order and ranges (its sliders also drive the trimap, so they stay its own rows)
    zi = [zon.index(f'id="{i}"') for i in ("z4Out", "z4Inset", "z4Feather", "z4OutFeather")]
    assert zi == sorted(zi)
    assert 'id="z4Out" min="0" max="120"' in zon and 'id="z4Inset" min="0" max="40"' in zon
    assert 'id="z4Feather" min="0" max="40"' in zon and 'id="z4OutFeather" min="0" max="80"' in zon
    # one band-point editor
    for gone in ("function drawBand(", "function bandHit(", "function wireBand(", "async function bandSave(", "function bandGeom("):
        assert gone not in zon, gone
    assert "zoneBand: () =>" in obj and "activeBandKey, pushBandUndo, reloadBand" in zon
    assert "S().stage === H.ST.VITMATTE && H.zoneBand ? H.zoneBand() : null" in bp


def test_one_version_comparison_2026_10_26():
    """One version comparison (user decision 2026-10-26): a single block (cmpHtml) mounted into the .cmp-slot of stages
    3 / 4 / 5 / 6 / 7; the list shows (and deletes) only the stage's versions, B = any version grouped by stage;
    keys A / B / D, Q = the stage's band preview. No per-stage copies (old st4Cmp*, vmShow*, lx6Show*)."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert obj.count("function cmpHtml()") == 1 and obj.count('id="stCmp"') == 1
    assert obj.count('<div class="cmp-slot"></div>') == 5
    assert "const CMP_STAGES = () => [ST.MATANYONE, ST.VIDEOMAMA, ST.VITMATTE, ST.EXPERIMENTS, ST.LTX]" in obj
    assert "own.map(" in obj and "<optgroup label=" in obj               # the list = this stage, B grouped by stage
    assert "mountCmp(n !== prevStage)" in obj
    for gone in ("st4CmpA", "st4CmpRuns", "vmShowMa", "lx6ShowMa"):
        assert gone not in obj, gone
    for key in ("KeyA", "KeyD", "KeyQ"):
        assert f"e.code === '{key}'" in obj, key


def test_one_stage_alpha_switch_2026_10_26():
    """„👁 alfa” is one switch with one meaning in stages 3 / 4 / 5 / 7 (user request 2026-10-26): every stage palette has
    it next to the band preview, the stage preview tint obeys it (stageAlphaOn), the stage result is tinted where nothing
    else is, and Alpha / Comp never get a green tint over the matte (live results of 4 / 7, zone preview of 5)."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    zon = (STATIC / "zones.js").read_text(encoding="utf-8")
    for cid in ("ePvAlpha", "vmPvAlpha", "lxPvAlpha"):
        assert re.search(rf'id="{cid}"[^>]*> 👁 alfa</label>', obj), cid
    assert "👁 wynik" not in obj
    assert "const alphaOff = !stageAlphaOn(st);" in obj
    assert "function drawStageResultTint(ctx, img, f)" in obj and "drawStageResultTint(ctx, img, f); if (t)" in obj
    assert "bandsOnly || !tintView() ||" in obj and "!bandsOnly && tintView() &&" in obj
    assert "const tintOk = S().view !== 'alpha' && S().view !== 'comp';" in zon
    assert "(tintOk || Z.ov !== 'alpha')" in zon
    # Comp: the stage frame preview is the matte, never tinted (drawStagePreview)
    assert "const bw = alphaBw() || S.view === 'comp';" in obj


def test_top_view_tab_rules_2026_10_26():
    """The view tab on top (Plate / Core / Alpha / Comp) is the user's in every stage (user decision 2026-10-26): the
    version comparison (entering a stage puts its version A on the view) and finished computations never switch it."""
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    body = obj[obj.index("function showRun(runName) {"):obj.index("function updateRunHud()")]
    assert "setView(" not in body
    body = obj[obj.index("function cmpShow(which) {"):obj.index("function cmpKey(which) {")]
    assert "setView(" not in body
    for name in ("objects.js", "brush.js", "zones.js"):
        src = (STATIC / name).read_text(encoding="utf-8")
        assert not re.search(r"loadStages\(\);\s*(H\.)?setView\('(alpha|core)'\)", src), name


def test_job_error_window_2026_10_28():
    """A computation that ends with an error gets a window that stays until "OK" (user report 2026-10-26: a Comfy OOM in
    VideoMaMa / a failed frame preview were only a short bar line or small red text), once per job, quiet jobs too;
    a Comfy error during the run is a red line in the progress window, not only in "Szczegóły"."""
    js = (STATIC / "jobs.js").read_text(encoding="utf-8")
    assert "function showError(j)" in js and "UI.dialog(" in js[js.index("function showError(j)"):]
    poll = js[js.index("async function poll()"):]
    assert "j.state === 'error' && j.ended && !errShown.has(j.id)" in poll
    assert poll.index("showError(failed[0])") < poll.index("if (busy && busy.quiet)")   # before the quiet-job return
    assert "ci.raw_error" in js[js.index("function showError(j)"):]
    assert "$('jmErr').style.display = ce ? '' : 'none';" in js


def test_slider_undo_2026_10_28():
    """Ctrl+Z / Ctrl+Shift+Z for sliders (user report 2026-10-28): ONE history module for every range (+ paired number
    field), replayed through the slider's own input / change handlers, per object + stage, newest action first."""
    js = (STATIC / "slider_undo.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert html.index("/static/slider_undo.js") < html.index("/static/objects.js")
    assert "el.dispatchEvent(new Event('input', { bubbles: true }));" in js and "el.dispatchEvent(new Event('change', { bubbles: true }));" in js
    assert "if (applying || !tracked(el)) return;" in js   # a replayed change is not recorded again
    assert "lastSliderAt >= lastCanvasAt" in js and "#canvasWrapper" in js
    assert "window.MD_SLIDER_UNDO.ctx = () => `${S.obj || ''}|${S.stage}`" in obj
    key_z = obj[obj.index("e.code === 'KeyZ' && (e.ctrlKey || e.metaKey)"):]
    assert key_z.index("su.wants(e.shiftKey)") < key_z.index("S.strokes[f].pop()")
    # no stage keeps its own slider history
    for f in STATIC.glob("*.js"):
        if f.name != "slider_undo.js":
            assert "sliderHistory" not in f.read_text(encoding="utf-8"), f.name


def test_group_member_opens_on_its_source_stage_2026_10_29():
    """A selected group member shows the mask its group takes: selecting it (or changing „Do grupy z:”) switches to the
    stage of that source (Core -> SAM 3.1 / CoTracker3, MatAnyone2 / VideoMaMa / ViTMatte / LTX-2.5 -> their stage)."""
    src = (STATIC / "objects.js").read_text(encoding="utf-8")
    fn = src[src.index("function sourceStage("):]
    fn = fn[:fn.index("\n  }\n")]
    for k in ("matanyone: 'MATANYONE'", "videomama: 'VIDEOMAMA'", "vitmatte: 'VITMATTE'", "ltx: 'LTX'"):
        assert k in src
    assert "ST.BRUSH : ST.SAM3" in fn
    sel = src[src.index("async function selectObject("):]
    sel = sel[:sel.index("\n  }\n")]
    assert "sourceStage(name)" in sel
    chg = src[src.index("sel.addEventListener('change', () => run('olSrc'"):]
    assert "sourceStage(n)" in chg[:chg.index("}));")]
    assert "if (S.obj === o.group)" in chg[:chg.index("}));")]   # the group stays selected, its composition reloads


def test_comp_is_linear_light_2026_10_30():
    """User decision 2026-10-30: the Comp view mixes plate and background in linear light like Nuke's Merge over (the
    MergeMatte port at its neutral defaults gives the same pixels) – GPU shader and CPU fallbacks alike."""
    gpu = (STATIC / "gpu_overlay.js").read_text(encoding="utf-8")
    assert "vec3 lin(vec3 c)" in gpu and "enc(lin(p) * (a / 255.) + lin(g) * (1. - a / 255.))" in gpu
    assert "window.mdLinComp = function (P, A, B)" in gpu
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "window.mdLinComp(p.data, a.data, b.data)" in obj and "window.mdLinComp(p.data, a.data, b.data)" in app
    assert "b.data[i + k] = p.data[i + k] * al + b.data[i + k] * (1 - al)" not in obj   # the old display-sRGB mix
    assert "cutout(" not in app


def test_merge_matte_gpu_preview_2026_10_30():
    """MergeMatte palette in real time (user request 2026-10-30): static/mm_gpu.js (WebGL2 port of the group, loaded before
    mergematte.js) draws the preview; the server port stays the fallback (no WebGL2) and the zoomed-in refinement."""
    assert HTML.index('<script src="/static/mm_gpu.js"></script>') < HTML.index('<script src="/static/mergematte.js"></script>')
    gpu = (STATIC / "mm_gpu.js").read_text(encoding="utf-8")
    assert "window.mdMmGpu = { render" in gpu and "EXT_color_buffer_float" in gpu and "0.4236" in gpu
    mm = (STATIC / "mergematte.js").read_text(encoding="utf-8")
    assert "function gpuOver(" in mm and "window.mdMmGpu.render(" in mm and "fetchPreview(f, reg, bg, key)" in mm


def test_final_touch_finish_on_gpu_2026_10_30():
    """Final Touch finishing live on the GPU (user request 2026-10-30): dragging erode / blur / levels draws base +
    strokes -> mdMmGpu.finish in the browser; with motion blur (server flow) the server image once it is in."""
    gpu = (STATIC / "mm_gpu.js").read_text(encoding="utf-8")
    assert "function finish(alpha, fin, W, H, into)" in gpu and "window.mdMmGpu = { render, finish" in gpu
    ft = (STATIC / "finaltouch.js").read_text(encoding="utf-8")
    # motion blur (server flow): the GPU draws the rest without it only while dragging / until the server's frame is in
    assert "function finishGpu(serverCur)" in ft and "(+n.mb_strength || 0) > 0) return !!F.liveFin || !serverCur" in ft
    assert "F.liveFin = { ...(F.liveFin || {}), [k]: +s.value }" in ft and "window.mdMmGpu.finish(raw" in ft


def _class_method(src: str, sig: str) -> str:
    """The text of a method of the viewer class (2-space indent), from `  <sig>` to its closing `  }`."""
    i = src.index("\n  " + sig) + 1
    return src[i:src.index("\n  }\n", i) + 4]


def test_project_opened_applied_once_2026_11_03():
    """Audit 2026-11-02 #15: the POST response and the WS `project_opened` of ONE open apply the project once, whichever
    comes first (the WS path fetches /api/project/current, so its fetch can also finish last). Another open (a new
    `open_id`) applies again, and loadCurrentProject() called directly (stage0.js after a plate change) still re-applies."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node nie jest zainstalowany – kolejność zdarzeń otwarcia projektu nie sprawdzona")
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "this.appliedOpenId = data.open_id" in _class_method(src, "applyProject(data) {")
    post = _class_method(src, "async openOrCreateProject(")
    assert "this.applyOpened(data)" in post and "this.applyProject(" not in post
    methods = "\n".join(_class_method(src, s) for s in ("handleWsMessage(msg) {", "async loadCurrentProject(", "applyOpened(data) {"))
    script = """
globalThis.document = { dispatchEvent() {} };
globalThis.CustomEvent = class { constructor(t, o) { this.detail = o && o.detail; } };
let pending = [], current = null, fetches = 0;
globalThis.fetch = () => { fetches++; const body = current;
  return new Promise(res => pending.push(() => res({ json: async () => body }))); };
class V { constructor() { this.applied = 0; }
  applyProject(data) { this.applied++; this.appliedOpenId = data.open_id; }
""" + methods + """
}
const settle = async () => { while (pending.length) pending.shift()(); await new Promise(r => setTimeout(r, 0)); };
const opened = id => ({ loaded: true, open_id: id, manifest: {} });
const ws = id => ({ type: 'project_opened', open_id: id });
const out = {};
(async () => {
  let v = new V(); current = opened('a');          // WS first, its fetch finishes before the POST response
  v.handleWsMessage(ws('a')); await settle(); v.applyOpened(opened('a')); out.ws_first = v.applied;
  v = new V(); current = opened('b');              // WS first, the POST response overtakes its fetch
  v.handleWsMessage(ws('b')); v.applyOpened(opened('b')); await settle(); out.response_overtakes = v.applied;
  v = new V(); current = opened('c'); fetches = 0; // the POST response first: the event needs no fetch at all
  v.applyOpened(opened('c')); v.handleWsMessage(ws('c')); await settle(); out.response_first = v.applied; out.fetches = fetches;
  current = opened('d');                           // the same page, another open (other client / agent / same shot again)
  v.handleWsMessage(ws('d')); await settle(); out.next_open = v.applied;
  const forced = v.loadCurrentProject(); await settle(); await forced;    // stage0.js: same open, the plates changed -> always applied
  out.forced = v.applied;
  console.log(JSON.stringify(out));
})();
"""
    r = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-1500:]
    import json
    assert json.loads(r.stdout) == {"ws_first": 1, "response_overtakes": 1, "response_first": 1, "fetches": 0,
                                    "next_open": 2, "forced": 3}


def test_project_switch_asks_for_no_old_object_2026_11_03():
    """After a project switch the prefetch asked the NEW project for the objects of the OLD one (20 x 500 in the log):
    applyProject moved to the first frame before anyone heard of the new project. md-project comes first now, and the
    object list is emptied until refreshList brings the new one."""
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    apply = _class_method(src, "applyProject(data) {")
    assert apply.index("new CustomEvent('md-project'") < apply.index("this.goToFrame(this.firstFrame)")
    objs = (STATIC / "objects.js").read_text(encoding="utf-8")
    handler = re.search(r"addEventListener\('md-project', \(\) => \{(.*)", objs).group(1)
    assert "S.objects = [];" in handler and handler.index("S.objects = [];") < handler.index("refreshList()")


def test_plate_color_profile_in_stage0_2026_11_03():
    """Plate colour profile (user decision 2026-11-03): a list next to the plates in stage 0 – auto (→ the proposed curve
    with its reason) or a curve id – read / written through /api/project/color_profile (project.json only), and the
    MergeMatte palette reloads its „z profilu plate'u” entry on the 'md-color-profile' event."""
    js = (STATIC / "stage0.js").read_text(encoding="utf-8")
    assert 'id="p0ColorProfile"' in js and 'id="p0ColorProfileInfo"' in js
    assert "window.MD.api('GET', '/api/project/color_profile')" in js
    assert "window.MD.api('PUT', '/api/project/color_profile', { color_profile: sel.value })" in js
    assert "new CustomEvent('md-color-profile'" in js and "loadColorProfile();" in js[js.index("async function refresh()"):]
    assert "project.json" in js and "shot_manifest" not in js.lower()


def test_merge_matte_log_curve_list_2026_11_03():
    """MergeMatte palette (user decisions 2026-11-03): the knob `log_curve` is a list „Krzywa log” – „z profilu plate'u”
    (the stage-0 colour profile, relabelled on 'md-color-profile') or an override per object – and the GPU preview
    computes the RESOLVED curve: mm_gpu.js carries every curve of engine/log_curves.py (l2g / g2l by the knob index,
    uniform `curve`), no OCIO-only ACEScct path (cct / logOn) any more."""
    mm = (STATIC / "mergematte.js").read_text(encoding="utf-8")
    assert "s.range === 'enum'" in mm and "<select data-mm=" in mm and "z profilu plate'u (→ ${esc(curveLabel(projectCurveId()))})" in mm
    assert "const resolvedCurve = (v) => (v === PROJECT ? projectCurveId() : v);" in mm
    assert "const k = { ...M.knobs, curve: curveIndex(M.knobs.log_curve) };" in mm
    assert "window.addEventListener('md-color-profile'" in mm and "M.info.project_curve = { id: p.id, label: p.label, reason: p.reason };" in mm
    gpu = (STATIC / "mm_gpu.js").read_text(encoding="utf-8")
    assert "vec3 l2g(vec3 x, int c){" in gpu and "vec3 g2l(vec3 y, int c){" in gpu
    for i in range(1, 7):
        assert f"if (c == {i})" in gpu[gpu.index("vec3 l2g("):gpu.index("vec3 g2l(")] and f"if (c == {i})" in gpu[gpu.index("vec3 g2l("):]
    assert "logOn" not in gpu and "cct(" not in gpu
    assert gpu.count("curve: { i: curve }") == 3                      # prepBg, fg, final
    assert "const curve = k.curve !== undefined ? (k.curve | 0) : (k.log ? 1 : 0);" in gpu


def test_camera_lock_switch_described_2026_11_04():
    """Camera lock of the plate alignment (user decision 2026-11-04): the user decides – auto (default) / locked / moving –
    and every choice has a plain description: the <select> tooltip and a visible help line under the row (p0LockHelp,
    refreshed on change) that says auto decides for the WHOLE measured range, so a mixed shot (camera still, then
    moving – SHOW) counts as moving; labels translated in en.json."""
    import json
    js = (STATIC / "stage0.js").read_text(encoding="utf-8")
    sel = js[js.index('title="Czy kamera stała'):js.index("</select>", js.index('<select id="p0Lock"'))]
    # 2026-11-10 (tidy panel): the select sits in a label „Kamera”, options auto / nieruchoma / ruchoma
    for v, label in (("auto", "auto"), ("on", "nieruchoma"), ("off", "ruchoma")):
        assert f'<option value="{v}">{label}</option>' in sel
    assert ">Kamera<select id=\"p0Lock\"" in js
    assert "CAŁEGO mierzonego zakresu" in sel and "ujęcie mieszane" in sel
    assert 'id="p0LockHelp"' in js and "const LOCK_HELP = {" in js
    for k in ("auto:", "on:", "off:"):
        assert k in js[js.index("const LOCK_HELP = {"):js.index("};", js.index("const LOCK_HELP = {"))]
    assert "lockHelp();" in js[js.index("function wire()"):]
    assert "body.camera_lock = $('p0Lock').value || 'auto';" in js
    en = json.loads((STATIC / "i18n/en.json").read_text(encoding="utf-8"))["strings"]
    for label in ("Kamera", "nieruchoma", "ruchoma"):
        assert en.get(label)
    assert all(en.get(m.strip()) for m in __import__("re").findall(r"^\s+(?:auto|on|off): '(.*)',$", js[js.index("const LOCK_HELP"):], __import__("re").M))


def test_one_number_field_for_every_slider_2026_11_11():
    """User 2026-11-11: every slider gets the typed number field of the right-hand palette – ONE module for all
    (static/slider_num.js, loaded on the page like range_fill.js), a converted label stays."""
    js = (STATIC / "slider_num.js").read_text(encoding="utf-8")
    for s in ("MutationObserver", "md-num", "typedStep", "dispatchEvent(new Event('input'", "dispatchEvent(new Event('change'",
              "hasOwnField", "r.dataset.numDone = 'label'"):
        assert s in js, s
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/range_fill.js") < html.index("/static/slider_num.js")


def test_final_touch_gpu_erode_is_the_ellipse_morphology_2026_11_11():
    """User 2026-11-11: a positive Erode broke the mask – the GPU preview used MergeMatte's separable erode without its
    uniforms (1 − alpha). Final Touch has its own disc min / max (cv2 MORPH_ELLIPSE)."""
    js = (STATIC / "mm_gpu.js").read_text(encoding="utf-8")
    assert "pass('morph', { S: T }" in js and "pass('erode', { S: T }, { r:" not in js
    assert "m = dil == 1 ? max(m, v) : min(m, v)" in js


def test_delete_key_in_cleanup_is_the_anchor_2026_11_11():
    """User report 2026-11-11: 🔑✕ in Cleanup acted on the cut-out keys of the selected object. In Cleanup it deletes the
    anchor of the frame (with the anchor's warning); the cut-out stage shortcuts are off there."""
    js = (STATIC / "objects.js").read_text(encoding="utf-8")
    # 2026-11-15: the anchor OR the C1 band key of the frame; both there = a window asks (test_cleanup_band_keys.py)
    assert "if (cleanMode()) return deleteCleanKey(f);" in js and "window.MD_CU_ANCHORS.hasAt(f)" in js
    assert "cleanKeyKinds(v.currentFrame).length > 0" in js and "updateDelKey: () => updateDelKeyBtn()," in js
    an = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "hasAt: (f) =>" in an and "removeAnchor(a.id)" in an and "window.MD.updateDelKey()" in an


def test_frame_mask_preview_lines_and_popup_2026_11_11():
    """User 2026-11-11: the frame mask shows the band lines and opens the big preview like the anchors – ONE popup
    module (static/pop_view.js) any panel can use."""
    pv = (STATIC / "pop_view.js").read_text(encoding="utf-8")
    assert "window.MD_POPVIEW = { open, close" in pv and "cu-anc-pop md-popview" in pv
    cu = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    assert "/api/cleanup/band/${f}/lines?grow_px=" in cu and "window.MD_POPVIEW.open(" in cu
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/pop_view.js") < html.index("/static/cleanup.js")
