# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup step C1 „Inpaint” UI (user decision 2026-11-11): the step's tools on a LEFT floating palette „🧹 Inpaint”
(the right panel keeps the description, the cleaned thing and the versions), the band instead of grow / soft (the one
band point editor, owner „c1”, outer side only), collapsible palettes, the anchor palette's head (on / off + the
Qwen-Image-2.1 card), its edit field and the big preview popup."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


def _fn(src: str, name: str) -> str:
    a = src.index(f"function {name}(")
    b = src.index("\n  }\n", a)
    return src[a:b]


def test_inpaint_palette_holds_the_tools_right_panel_the_rest():
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    pal, panel = _fn(js, "paletteHtml"), _fn(js, "panelHtml")
    # the model run on the left palette; the band, the protection and the grain on the right one („Złożenie”, 2026-11-15)
    order = ["cuFill", "cuModel", "cuModelInfo", "cuPromptMode", "cuPrompt", "cuH3Prompt", "cuSeed", "cuGen",
             "cuPreview", "cuRun"]
    pos = [pal.index(f'id="{i}"') for i in order]
    assert pos == sorted(pos), "Inpaint palette order"
    assert 'class="pts-palette cu-pal" id="cuPal"' in pal
    for i in ("cuObjects", "cuVersions", "cuWarn", "cuAnchors"):
        assert f'id="{i}"' in panel and f'id="{i}"' not in pal
    for i in order:
        assert f'id="{i}"' not in panel, i
    assert "Co robi:" in panel
    comp = _fn(js, "compPaletteHtml")
    # band preview + point tools above the band sliders (544eca7)
    corder = ["cuBandPv", "cuBandAdd", "cuGrow", "cuSoft", "cuLive", "cuRecompose", "cuHoldout", "cuSkipLost",
              "cuKept", "cuGrain", "cuGrainStrength"]
    cpos = [comp.index(f'id="{i}"') for i in corder]
    assert cpos == sorted(cpos), "Złożenie palette order"
    # no inset slider in the C1 band, the old wording is gone
    assert "Wcięcie" not in comp and "Powiększ (px)" not in js and "Miękkość (px)" not in js
    assert "Pas na zewnątrz" in comp and "Wtopienie zewn" in comp
    # the engine list stays server-driven (another session adds engines): no hard-coded <option> in the model select
    assert '<select id="cuModel" class="form-input"></select>' in pal
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert ':not(#cuAnchorPal):not(#cuPal):not(#cuCompPal) { display: none !important; }' in css
    assert "body:not(.mode-clean) #cuPal" in css and ".md-folded > :not(.md-fold-head)" in css


def test_band_editor_owner_c1():
    bp = (STATIC / "bandpoints.js").read_text(encoding="utf-8")
    assert "c1On() ? 'c1' : c3dOn() ? 'c3d'" in bp and "isC1() ? '/api/cleanup/band'" in bp
    assert "const isC3d = () => STAGE_OF() === 'c3d' || isC1();" in bp            # outer side only, like C5
    assert "if (s === 'c1') return window.MD_CLEANUP.bandPv();" in bp
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    assert "bandOwner, bandTool: () => C.bandTool || null, bandPv, reloadBand," in js
    assert "B.drawLines(ctx, img, v.currentFrame, bandVals()" in js and "v.overlayHooks.push(drawBand)" in js
    assert "window.MD_BANDPTS.reset();   // its own band (per cleaned thing)" in js


def test_palettes_collapse_and_remember():
    # the C1 palettes fold through the shared script (static/palette_fold.js via makeFloating), no own copy
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    anc = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "function foldable(" not in js and "foldable" not in anc
    assert "makeFloating(pal, cw, '.pp-head', palPos)" in js and "makeFloating(pal, cw, '.pp-head', palPos)" in anc
    fold = (STATIC / "palette_fold.js").read_text(encoding="utf-8")
    assert "cuPal: 'md_cu_fold_inpaint', cuAnchorPal: 'md_cu_fold_anchors'" in fold   # saved state kept


def test_anchor_palette_head_edit_field_and_popup():
    anc = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    head = anc[anc.index('<div class="cu-anc-head">'):anc.index('<div id="cuAncBody">')]
    assert 'id="cuAnchor"' in head and 'id="cuAnchorInfo"' in head
    edit = _fn(anc, "editHtml")
    assert "Opis: z palety Inpaint" in edit                       # main: no second field
    assert re.search(r"kind === 'linked'.*disabled", edit, re.S)  # linked: disabled until ⛓ Odłącz
    assert 'id="cuAncPrompt"' in edit
    det = _fn(anc, "detailHtml")
    assert "cu-gens" not in det and "cuAncCap" not in det and "g.prompt" not in det   # no generation prompt text on the palette
    assert "if (e.target.closest('.cu-anc-bigwrap')) { openPop(a.id); return; }" in anc
    pop = _fn(anc, "drawPop")
    for s in ("data-gdel", "data-anc-export", "data-anc-import", "data-pop-copy", "data-pop-use", "cuPopPrompt"):
        assert s in pop, s
    assert "e.key === 'Escape' && P.aid" in anc and "addEventListener('wheel'" in anc and "cuPopClose" in anc
    assert "navigator.clipboard.writeText(g.prompt)" in anc


def test_anchor_texts_follow_the_video_engine():
    """MiniMax-H3 and Wan 2.2 Fun VACE share the anchors and the transition clip prompt (user 2026-11-11): the palette
    names the chosen engine, the transition prompt shows for both, the version list names VACE."""
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    anc = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "const tr = !!(m && m.anchors && s.anchor === 'frame'" in js and "$('cuH3Eng').textContent = m.label" in js
    assert "vace: 'Wan 2.2 Fun VACE'" in js and "engine: m ? m.label : ''" in js
    assert "Kotwice MiniMax-H3" not in anc and "Prompt klipu MiniMax-H3" not in js
    assert "const ENG = () => (A.ctx && A.ctx.engine) || 'MiniMax-H3';" in anc and 'id="cuAncEng"' in anc


def test_anchor_popup_fits_small_windows():
    """User 2026-11-12: the big anchor preview never leaves a small window (checked in the browser at 1000x560 and
    700x460: the actions stay on screen, the left edge at 8 px); a dragged position is pulled back inside."""
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    pop = css[css.index(".cu-anc-pop {"):css.index("}", css.index(".cu-anc-pop {"))]
    assert "left: max(8px, calc(50% - 380px))" in pop and "max-height: calc(100vh - clamp(8px, 6vh, 60px) - 8px)" in pop
    assert "overflow-y: auto" in pop
    assert "calc(100vh - 380px)" in css[css.index(".cu-pop-view {"):]
    anc = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "function keepIn()" in anc and "window.addEventListener('resize', () => { if (P.aid) { keepIn(); fitView(); } });" in anc


def test_hud_names_the_cleaned_version():
    """User 2026-11-12: with the plate „cleaned” the viewer's HUD names the version on it (like the stage mask label)
    and says when the current frame is outside its range."""
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function shownLabel()" in js and "    shownLabel," in js
    assert "window.MD_CLEANUP.shownLabel()" in app and "poza zakresem: brak klatki F${this.currentFrame}" in app
    assert "if (changed && this.activePlate === 'cleaned') this.updateHudMode();" in app


def test_composite_palette_right_side_moved_blocks_no_duplicates():
    """User decision 2026-11-15: a RIGHT floating palette „Złożenie” (C1 only) with the band, „Ochrona” (+ the kept
    objects: swatch, name, mask, 👁 overlay, „chroń”, „→ Wycinanie”) and the grain – MOVED from „🧹 Inpaint” (each id
    exactly once in the file), „Zapas generowania” stays left; every handler kept."""
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    pal, comp = _fn(js, "paletteHtml"), _fn(js, "compPaletteHtml")
    assert 'class="pts-palette cu-pal cu-comp-pal" id="cuCompPal"' in comp and "🧩 Złożenie" in comp
    moved = ["cuBandPv", "cuBandAdd", "cuBandRemove", "cuBandClear", "cuGrowR", "cuGrow", "cuSoftR", "cuSoft",
             "cuBandInfo", "cuBandLimit", "cuRecompRow", "cuRecompInfo", "cuLive", "cuRecompose", "cuHoldout",
             "cuSkipLost", "cuKept", "cuGrain", "cuGrainStrength", "cuGrainStrengthV", "cuGrainInfo"]
    for i in moved:
        assert f'id="{i}"' in comp and f'id="{i}"' not in pal, i
    assert 'id="cuGen"' in pal and 'id="cuGen"' not in comp                    # the model run's margin stays left
    ids = re.findall(r'\bid="([A-Za-z][\w-]*)"', js)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, dup
    for i in moved:   # the handlers still find them
        if i not in ("cuBandInfo", "cuBandLimit", "cuRecompRow", "cuRecompInfo", "cuGrainInfo", "cuGrainStrengthV", "cuGrowR", "cuSoftR", "cuKept"):
            assert f"$('{i}')" in js, i
    assert "makeFloating($('cuCompPal'), cw, '.pp-head', compPos)" in js and "posStore('md_cu_comp_palette')" in js
    assert "[[pal, palPos], [$('cuCompPal'), compPos]]" in js                  # shown with the C1 palette only
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert ":not(#cuPal):not(#cuCompPal) { display: none !important; }" in css and "body:not(.mode-clean) #cuCompPal" in css
    assert ".cu-comp-pal { left: auto; right: 10px;" in css


def test_kept_objects_list_overlay_and_jump_to_cutting():
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    rk = _fn(js, "renderKept")
    for s in ("cu-kept-sw", "keptMaskLabel(k.mask)", "data-kept-eye", "data-kept-on", "data-kept-cut", "→ Wycinanie",
              "łata może go przykryć"):
        assert s in rk, s
    assert "if (m.source === 'final_touch') return 'Final Touch';" in js and "return 'rdzeń 1+2';" in js
    # the overlay: the server's holdout alpha + cut, drawn by an overlay hook, re-asked on frame / band / reload
    assert "/api/cleanup/holdout/${encodeURIComponent(n)}/${f}?grow_px=" in js
    assert "v.overlayHooks.push(drawHoldout)" in js
    assert js.index("overlayHooks.push(drawLive)") < js.index("overlayHooks.push(drawHoldout)") < js.index("overlayHooks.push(drawBand)")
    assert "X-Holdout-On" in js and "X-Holdout-Cut-Px" in js
    # „chroń” -> holdout_off, then the live preview / overlays again; back from cutting: load() bumps both
    tg = _fn(js, "keptToggle")
    assert "save({ holdout_off: C.info.settings.holdout_off })" in tg and "compositeChanged()" in tg
    ld = _fn(js, "load")
    assert "C.liveBump = (C.liveBump || 0) + 1; C.ho.bump += 1;" in ld
    cut = _fn(js, "keptToCut")
    assert "setMode('cut')" in cut and "window.MD.selectObject(n)" in cut
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "objColor: (n) => objColor(n)," in obj


def test_kept_row_like_objects_palette_no_new_object_in_cleanup_2026_11_15():
    """User 2026-11-15: the kept row = the Obiekty palette row (checkbox, eye 👁 / ◌, colour, name); no ＋ / 🔗 in the
    Obiekty palette of Cleanup (objects are made in Wycinanie)."""
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    rk = _fn(js, "renderKept")
    assert rk.index("data-kept-on") < rk.index("data-kept-eye") < rk.index("cu-kept-sw") < rk.index("cu-kept-name")
    assert "${ov ? '👁' : '◌'}" in rk and "> chroń</label>" not in rk
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "${isCleanMode() ? ''" in obj and "if ($('olNew')) $('olNew')" in obj and "if ($('olNewGrp')) $('olNewGrp')" in obj
    pk = _fn(js, "pick")
    assert "v.setActivePlate('cleaned')" in pk and "v.setActivePlate('beauty')" in pk


def test_anchor_popup_shows_every_anchor_2026_11_15():
    """User 2026-11-15: the anchor popup lays every anchor (the main one too) side by side, panned / zoomed as one;
    a click on a preview = that anchor, its versions in the strip below (click = show and choose)."""
    js = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert 'id="cuPopStrip"' in js and "cuPopImg" not in js
    ds = _fn(js, "drawStrip")
    assert "anchors().slice().sort((x, y) => x.frame - y.frame)" in ds and "data-pop-aid" in ds
    assert "if (click && tile && tile.dataset.popAid !== P.aid)" in js and "A.sel = a.id; if (V()) V().goToFrame(a.frame); draw();" in js
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert ".cu-pop-strip { position: absolute;" in css and ".cu-pop-strip .cu-pop-tile.sel" in css
    assert "data-pop-gen" in js and "if (e.target.closest('[data-pop-gen]')) { generate(a.id); return; }" in js


def test_beauty_plate_hides_the_composite_overlays_2026_11_15():
    """User 2026-11-15: Beauty picked, the live patch stayed over it and the version list kept a version marked."""
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    assert "const onCleaned = () => !!(window.viewer && window.viewer.activePlate === 'cleaned');" in js
    assert "shownRec() && onCleaned());" in js and js.count("&& onCleaned()") >= 3
    vh = _fn(js, "versionsHtml")
    assert "const shown = v.shown && onCleaned();" in vh and "v.shown ?" not in vh
    assert "window.addEventListener('md-plate', () => {" in js


def test_anchor_popup_resizable_2026_11_15():
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    pop = css[css.index(".cu-anc-pop {"):css.index(".cu-pop-head {")]
    assert "resize: both;" in pop and "min-width: 360px" in pop
    assert ".cu-pop-view { position: relative; flex: 1 1 auto; min-height: 160px;" in css


def test_anchor_popup_link_and_own_prompt_2026_11_15():
    """The anchor popup: ⛓ Odłącz / 🔗 Podłącz, and for an own anchor its description editable with „✨ Popraw opis”."""
    js = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert 'data-pop-kind="own"' in js and 'data-pop-kind="linked"' in js
    assert "patch(a.id, { kind: kd.dataset.popKind }, 'Kotwica')" in js
    assert 'id="cuPopAncPrompt"' in js and "window.MD_CU_PROMPT.assist({ ta: $('cuPopAncPrompt'), btn: b, frame: a.frame })" in js
    cp = (STATIC / "cleanup_prompt.js").read_text(encoding="utf-8")
    assert "async function assist(o) {" in cp and "const ta = ta0;" in cp and "$('cuAssist').addEventListener('click', () => assist());" in cp
    assert 'data-pop-del="1"' in js and "if (e.target.closest('[data-pop-del]')) { removeAnchor(a.id); return; }" in js


def test_c1_palette_rows_one_layout_2026_11_15():
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert ".cu-pal > label.zone-slider, .cu-pal > label.cu-gen, #cuAnchorPal .cu-anc-head label.cu-anc-dn {" in css
    assert "grid-template-columns: 112px minmax(0, 1fr) 62px" in css and "#cuAnchorPal .cu-anc-proj .pp-hint { flex-basis: 100%;" in css


def test_card_key_anchors_hidden_for_now_2026_11_15():
    js = (STATIC / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "const SHOW_PROJ = false;" in js and "${SHOW_PROJ ? projHtml() : ''}" in js


def test_progress_above_the_anchor_popup_2026_11_15():
    js = (STATIC / "jobs.js").read_text(encoding="utf-8")
    assert "#jobModal{position:fixed;inset:0;z-index:3000;" in js and "z-index:2990;" in js


def test_overlays_respect_the_comparison_2026_11_15():
    js = (STATIC / "cleanup.js").read_text(encoding="utf-8")
    assert "function cmpClip(ctx, img)" in js and js.count("if (cmpClip(ctx, img)) ctx.drawImage(") == 2
    assert "c.mode === 'wipe'" in js and "c.mode === 'diff'" in js


def test_cleanup_draws_nothing_of_the_cutout_stages_2026_11_15():
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    d = obj[obj.index("  function drawOverlay() {"):]
    d = d[:d.index("const covers = ")]
    assert "if (cleanMode()) {" in d and "drawOthers(ctx, img, f);" in d and "window.MD_BANDPTS.draw(ctx, img, k, f);" in d
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert "body.mode-clean .zone-palette { display: none !important; }" in css
