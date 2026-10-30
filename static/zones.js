// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/zones.js – stage ViTMatte (H.ST.VITMATTE, layout 2026-10-15) panel „Strefy” (docs/PLAN_HAIR_TOOL.md step 5, section 8 pkt 1–5 / 7): list of zones (cards),
// new zone = first stroke (brush + G / brush − R / polygon P) with the current preset, automatic preview (debounce 600 ms,
// POST /zones/{id}/preview/{f}) with overlays Alpha / Comp / Trimapa / Silniki, zone keys on the timeline (diamonds,
// painting = key, PUT/DELETE keys/{f}), propagated region in frames without a key (GET region/{f}), „Zastosuj na ujęcie”
// (POST apply, long job) with the "objects in front" dialog (409 missing_front), clean plate row (docs/PLAN_CLEANPLATE.md 4),
// loupe. Step 7 (correction loop on the final mask): strokes in the Alpha / Comp view go to the zone key of the frame,
// an iteration note (`z4Iter`) is stored in that key (PUT keys/{f} prompt -> zone.prompt_history), Ctrl+Z restores the
// previous key incl. its note; the zone-alpha preview is drawn on the full-resolution plate tiles (Z8) when they are active.
// The old threshold criteria (dark / bright / color / all) stay under „Zaawansowane” (stage Eksperymenty, objects.js);
// the stage-2 brush corrections (params.source = brush) belong to static/brush.js and are not listed here.
// „Zastosuj na ujęcie” lays the zones on the stage-3 MatAnyone version (POST apply?stage=vitmatte).
// Rules: only the current frame's overlay is invalidated after a preview (per-frame mechanism, never clearOverlays());
// the default engine is `matanyone` (user decision 2026-10-13); nothing here changes the engines' results.
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  // drawing a zone area (brush / polygon, painted-area mode) is hidden for now (user decision 2026-10-21: it misled –
  // the band of a whole-outline zone is set by control points); the code stays, true brings the tools back
  const DRAW_TOOLS = false;
  // presets, zone description / fix note, zone engine + holdout, clean plate (user decision 2026-10-25: one
  // whole-outline zone per object, its settings live in the „Strefa” palette); the code stays – a possible future
  // "settings preset" – and true brings the panel rows back
  const PANEL_EXTRAS = false;
  let H = null;                      // hooks from objects.js (see init)
  const S = () => H.S;
  const obj = () => H.S.obj;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const Z = {
    list: [], status: {}, active: null,        // zones of the object (GET /zones), card status, active zone id
    engAligned: {},                            // zone id -> its model was moved to the stage 5 engine once (syncForm)
    band: { modeSig: null },   // the band points themselves: static/bandpoints.js (one editor for every band, 2026-10-24)
    preset: 'hair', prompt: '', engine: 'matanyone', holdout: true,   // defaults of the NEXT new zone
    ov: 'alpha',                               // overlay of the preview: alpha | comp | trimap | engines | none
    preview: null,                             // { zid, f, data } of the last preview
    pvTimer: null, pvBusy: false, pvAgain: null,
    drawTimer: null, drawPending: null, drawBusy: false,
    regionCache: {}, regionBusy: {}, regionRetry: {},   // zid -> { f -> GET region/{f} }; retries while the app is busy
    undo: [],                                  // [{ zid, f, prev }] prev = key entry before the stroke, null = key created
    loupe: false, mouse: null,
    cp: null, cpNone: false,                   // GET /api/project/cleanplate_auto; "Bez clean plate'u" (per project, localStorage)
    obj: null,                                 // object the state belongs to (reset)
    plan: null,                                // step 9: last prompt interpretation { plan, bg_strokes, frame } (POST zones/interpret)
    props: null,                               // step 9: zone proposals { f, list, preset, hover } (POST zones/propose)
    brush: null,                               // palette brush: paint | erase | bg (null = the left toolbar as before)
    newPainted: false,                         // next new zone = painted ViTMatte band (local tracking)
  };
  const PRESETS = [
    ['hair', '💇', 'włosy / futro', 'Preset włosy / futro: kosmyki i futro na dowolnym tle (clean 1.2 · chroma 1.0 · knn 1.0)'],
    ['fibres', '🧵', 'włókna', 'Preset włókna (filc, wełna, sierść): + silnik backlit, gdy jest plate backlit'],
    ['smoke', '💨', 'dym / para', 'Preset dym / para / woalka: półprzezroczyste, chroma i clean ważniejsze'],
    ['motion', '🌀', 'rozmycie', 'Preset rozmycie ruchu / defocus: MatAnyone2 dla rozmytych, ruchomych włosów'],
    ['thin', '🌾', 'cienkie', 'Preset cienkie (trawa, gałęzie, anteny): wyspy w strefie nie są usuwane'],
  ];
  const PRESET_ICON = Object.fromEntries(PRESETS.map((p) => [p[0], p[1]]));
  const PRESET_NAME = Object.fromEntries(PRESETS.map((p) => [p[0], p[2]]));
  const CRIT_LABEL = { dark: 'ciemne', bright: 'jasne', color: 'kolor', all: 'wszystko' };
  // zone engine name as shown: the per-frame engine `vitmatte` is BiRefNet_HR-matting / Lucida when the zone's model is one
  const zEngLabel = (z) => (z && z.matting === 'vitmatte' && EDGE_MODEL_LABEL[pval(z, 'vitmatte_model')]
    ? EDGE_MODEL_LABEL[pval(z, 'vitmatte_model')] : (z ? ENGINE_LABEL[z.matting] || z.matting : ''));
  // engine lists of the zone follow the stage 5 engine (user decision 2026-10-21): BiRefNet -> `vitmatte` shown as BiRefNet,
  // MatAnyone2 (non-commercial) not offered; ViTMatte -> as before
  function applyEngineOptions(sel) {
    const bir = ((S().info && S().info.edge_engine) || 'vitmatte') === 'birefnet';
    for (const o of sel.options) {
      if (o.dataset.txt == null) o.dataset.txt = o.textContent;
      if (o.value === 'vitmatte') o.textContent = bir ? 'BiRefNet' : o.dataset.txt;
      if (o.value === 'matanyone') { o.hidden = bir; o.disabled = bir; }
    }
  }
  const ENGINE_LABEL = { matanyone: 'MatAnyone2', auto: 'auto', clean: 'clean', chroma: 'chroma', knn: 'knn', cf: 'cf', hybrid: 'hybrid', backlit: 'backlit', vitmatte: 'ViTMatte' };
  const STATE_LABEL = { policzona: ['✔ policzona', 'ok'], 'do przeliczenia': ['↻ do przeliczenia', 'warn'], liczona: ['⏳ liczona…', 'busy'], 'wyłączona': ['○ wyłączona', 'muted'] };
  const PALETTE = ['#f472b6', '#c8c8cc', '#34d399', '#fb923c', '#a78bfa', '#22d3ee', '#f87171', '#a3e635'];
  const OVS = [['alpha', 'Alpha', 'Miękka alpha strefy w tej klatce (zielona z przezroczystością = dodane)'],
               ['comp', 'Comp', 'Rdzeń + alpha strefy na tle compu (V zmienia tło)'],
               ['trimap', 'Trimapa', 'Trimapa strefy: niebieskie = FG (rdzeń), żółte = unknown (tu liczy matting), reszta = tło'],
               ['engines', 'Silniki', 'Mapa wag silników: niebieski = chroma, czerwony = knn, zielony = clean / inne'],
               ['none', 'Wył.', 'Bez nakładki podglądu strefy']];

  // ------------------------------------------------------------------------------------------------ helpers
  const api = (...a) => H.api(...a);
  const toast = (t, ms) => H.toast(t, ms);
  const status = (t, k) => H.status(t, k);
  const zoneColour = (zid) => { const i = Z.list.findIndex((z) => z.zone_id === zid); return PALETTE[(i < 0 ? 0 : i) % PALETTE.length]; };
  const active = () => Z.list.find((z) => z.zone_id === Z.active) || null;
  const activeMatte = () => { const z = active(); return z && z.criterion === 'matte' ? z : null; };
  // zone params edited on the card (user decision 2026-10-15): temporal smoothing of the ViTMatte alpha + finish levels.
  // Shown value = the zone's param, else the engine default (what the server uses when the key is absent).
  const TEMPORAL = [['none', 'brak'], ['median', 'mediana'], ['flow', 'mediana po przepływie'], ['flow_mean', 'średnia po przepływie (czerń/biel przed)']];
  // help of each smoothing mode: speed / flicker measured 2026-10-28 (local_results/ab_zone_smooth/, sh060 + CLIP2_A)
  const TEMPORAL_HELP = {
    none: 'Brak – najszybciej (~0 s/kl.), migotanie bez zmian. Kiedy: strefa jest prawie samo 0 / 1 (twardy obrys) albo szybki podgląd.',
    median: 'Mediana – szybka (~0,3 s/kl.), ale przy ruchu prawie nie zmniejsza migotania (CLIP2_A: bez zmian) – działa tylko tam, gdzie nic się nie rusza. Kiedy: statyczna kamera i nieruchomy obiekt.',
    flow: 'Mediana po przepływie – najwolniejsza (~2,8 s/kl.), migotanie −36 % (CLIP2_A). Zostawiona dla zgodności: średnia po przepływie jest szybsza i spokojniejsza.',
    flow_mean: 'Średnia po przepływie – POLECANA do włosów: najmniej migotania (CLIP2_A −44 % z ×2, −47 % z ×1), ~0,5 s/kl. z ×1, ~1,5 s/kl. z ×2. Najpierw czerń / biel w każdej klatce, potem średnia ważona z ± promień klatek dopasowanych przepływem – śmieć z jednej klatki nie rozmazuje się na sąsiednie.',
  };
  const PARAM_DEF = { temporal: 'median', temporal_radius: 2, temporal_scale: 1, alpha_black: 0, alpha_white: 1, alpha_blur_px: 0,
    fg_erode_px: 2, force_core_erode_px: 0, alpha_black_after: 0, region: 'draw', outline_out_px: 40, exclude_objects: [], track: 'rigid',
    vitmatte_model: 'base', model_px: 1008, unknown_value: 0.5 };
  // edge model of the ViTMatte zone engine -> [licence line, non-commercial] (user decision 2026-10-21; doctor.py LICENSES)
  const BIREFNET_MODELS = ['birefnet_hr_matting', 'lucida'];
  const BIREFNET_PX = { birefnet_hr_matting: 2016, lucida: 1008 };
  const VM_NC = '⚠ tylko niekomercyjnie';
  const EDGE_MODEL_LICENCE = {
    base: [VM_NC, true], small: [VM_NC, true], 'base-distinctions': [VM_NC, true],
    birefnet_hr_matting: ['MIT – komercyjnie OK', false], lucida: ['MIT – komercyjnie OK', false] };
  const EDGE_MODEL_LABEL = { base: 'ViTMatte base', small: 'ViTMatte small', 'base-distinctions': 'ViTMatte base – distinctions',
    birefnet_hr_matting: 'BiRefNet_HR-matting', lucida: 'Lucida' };
  // band preview switch of the palette (user request 2026-10-16, like the MatAnyone palette): remembered in this browser
  const bandPvGet = () => { try { return localStorage.getItem('md_zone_bandpv') !== '0'; } catch (e) { return true; } };
  const bandPvSet = (on) => { try { localStorage.setItem('md_zone_bandpv', on ? '1' : '0'); } catch (e) { /* storage blocked */ } };
  const isOutline = (z) => !!(z && z.params && z.params.region === 'outline');
  // any painted key (exclusion / background stroke)? – engine/zones.py outline_without_drawing on the server
  const hasDrawing = (z) => [z, ...Object.values(z.keys || {})].some((k) => k && ((k.polygon || []).length || (k.strokes || []).length || (k.bg_strokes || []).length));   // zone = band around the whole core minus exclusions
  const pval = (z, k) => (z && z.params && z.params[k] != null ? z.params[k] : PARAM_DEF[k]);
  const inZoneFlow = () => !!(H && obj() && !S().auxMode && S().stage === H.ST.VITMATTE);
  const curFrame = () => V().currentFrame;
  function keysOf(z) {
    const out = {};
    for (const [k, v] of Object.entries(z.keys || {})) out[+k] = { polygon: v.polygon || [], strokes: v.strokes || [], bg_strokes: v.bg_strokes || [], prompt: v.prompt || '' };
    const base = (z.keys || {})[String(z.frame)] || {};
    out[+z.frame] = { polygon: z.polygon || [], strokes: z.strokes || [], bg_strokes: z.bg_strokes || [], prompt: base.prompt || '' };
    return out;
  }
  const keyFrames = (z) => Object.keys(keysOf(z)).map(Number).sort((a, b) => a - b);
  // the zone keys the user sees (user decision 2026-10-21): a whole-outline zone has no key of its own – only a drawn
  // (old exclusion) key counts; its keys are the band point triangles
  const drawnKey = (k) => !!(k && ((k.polygon || []).length || (k.strokes || []).length || (k.bg_strokes || []).length));
  const laneKeys = (z) => (isOutline(z) ? keyFrames(z).filter((f) => drawnKey(keysOf(z)[f])) : keyFrames(z));
  function share(es) {
    const ks = Object.keys(es || {}); if (!ks.length) return '';
    return ks.sort((a, b) => es[b] - es[a]).map((k) => `${ENGINE_LABEL[k] || k} ${Math.round(es[k] * 100)} %`).join(' / ');
  }
  function replaceZone(zone, st) {
    const i = Z.list.findIndex((z) => z.zone_id === zone.zone_id);
    if (i >= 0) Z.list[i] = zone; else Z.list.push(zone);
    if (st) Z.status[zone.zone_id] = st;
    delete Z.regionCache[zone.zone_id];
    if (window.MD_BANDPTS) window.MD_BANDPTS.dirty();   // the shown widths depend on the sliders: ask again
    // a whole-outline zone shows the server's band: ask for it again now, not on the next frame change (user report
    // 2026-10-21: after a slider the band was gone until the frame changed)
    if (inZoneFlow() && Z.active === zone.zone_id && zone.criterion === 'matte' && isOutline(zone) && !hasDrawing(zone) && bandPvGet()) {
      setTimeout(() => { const a = activeMatte(); if (a && a.zone_id === zone.zone_id) regionFor(a, curFrame(), false); }, 0);
    }
  }
  function missingFront(e) { return e && e.status === 409 && e.detail && Array.isArray(e.detail.missing_front) ? e.detail : null; }

  // ------------------------------------------------------------------------------------------------ panel
  function panelHtml() {
    return `<div class="sec-title">Strefy <span class="muted">(matting lokalny: włosy, futro, dym, włókna)</span></div>
      <div id="st4Zones" class="zone-list"></div>
      <div class="row zone-new-row"><button class="btn btn-secondary btn-sm" id="z4New" title="Nowa strefa ViTMatte wzdłuż całego obrysu obiektu – jedna na obiekt">＋ Nowa strefa</button>
        <button class="btn btn-secondary btn-sm" id="z4Propose" title="Zaproponuj strefę (krok 9): SAM 3.1 z opisem strefy na kadrze obiektu minus rdzeń + miękkie krawędzie poza rdzeniem → propozycje (żółte wielokąty z numerami); ✓ tworzy strefę z wybranym presetem, ✕ odrzuca">✨ Zaproponuj</button>
        <span class="muted zone-hint" id="z4Hint"></span></div>
      <div id="z4Props" class="zone-props"></div>
      <div id="z4Extras"${PANEL_EXTRAS ? '' : ' style="display:none"'}>
      <div class="zone-presets" id="z4Presets">${PRESETS.map((p) => `<button class="tool-btn zp" data-preset="${p[0]}" title="${esc(p[3])}"><span class="tool-icon">${p[1]}</span><span class="tool-label">${p[2]}</span></button>`).join('')}</div>
      <label>Opis strefy<input id="z4Prompt" class="form-input" placeholder="co wyciąć, czego pozbyć się z tła (np. włosy, bez złotej linii)"></label>
      <div class="muted zone-plan" id="z4Plan"></div>
      <label>Uwaga do poprawki (ten klucz)<input id="z4Iter" class="form-input" placeholder="np. mniej tła przy uchu – zapisze się z następną kreską" title="Pętla korekty (krok 7): tekst trafia do klucza strefy tej klatki razem z następną kreską + / − i do historii iteracji strefy; reguły promptu (krok 9) czytają go w tej klatce. Ctrl+Z przywraca poprzedni klucz z jego uwagą"></label>
      <div class="row">
        <label>Silnik<select id="z4Engine" class="form-input"><option value="matanyone">MatAnyone2 (domyślnie)</option><option value="auto">auto (dobór per fragment)</option><option value="clean">clean (clean plate)</option><option value="chroma">chroma</option><option value="knn">knn</option><option value="vitmatte">ViTMatte (eksperyment)</option></select></label>
      </div>
      <div class="muted zone-tune-note">Suwaki strefy (obszar, wygładzanie w czasie, wykończenie): paleta „Strefa” na podglądzie (przeciągnij za ⠿)</div>
      <div class="row zone-cp" id="z4CleanRow"><span class="muted" id="z4CleanTxt">Clean plate: …</span>
        <button class="btn btn-secondary btn-xs" id="z4CpAuto">Policz z ujęcia (auto)</button>
        <button class="btn btn-secondary btn-xs" id="z4CpFile">Mam plik…</button>
        <button class="btn btn-ghost btn-xs" id="z4CpNone">Bez clean plate’u</button></div>
      </div>
      <div class="row"><label class="chk" title="Obiekty z przodu (wcześniej w kolejności głębi) są wycinane ze strefy: alfa 0 pod nimi, a silnik liczy krawędź aż do ich krawędzi"><input type="checkbox" id="z4Holdout" checked> holdout obiektów z przodu</label></div>
      <div class="zone-ov-row"><span class="muted">Nakładka</span>
        <div class="seg zone-ov" id="z4Ov">${OVS.map(([k, l, t]) => `<button data-ov="${k}" class="${k === Z.ov ? 'active' : ''}" title="${esc(t)}">${l}</button>`).join('')}</div></div>
      <div class="row"><button class="btn btn-secondary btn-xs" id="z4Refresh">↻ Podgląd</button>
        <button class="btn btn-secondary btn-xs" id="z4Loupe">🔍 Lupa L</button></div>
      <div class="muted zone-pv" id="z4PvInfo"></div>
      <div class="row"><button class="btn btn-primary btn-sm" id="z4ApplyShot">✚ Zastosuj na ujęcie</button><button class="btn btn-ghost btn-xs" id="z4KeyDel"${DRAW_TOOLS ? '' : ' style="display:none"'}>🗑 Usuń klucz strefy (ta klatka)</button></div>`;
  }

  // ------------------------------------------------------------------------------------------------ floating palette „Strefa”
  // (user decision 2026-10-15): every zone slider lives here, over the view (dragged by ⠿, position in md_zone_palette):
  // area (outline mode: band around the whole core minus exclusions – drawing / objects), temporal smoothing, finish.
  const palPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_zone_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_zone_palette', JSON.stringify(p)); else localStorage.removeItem('md_zone_palette'); } catch (e) { /* storage blocked */ } },
  };
  function paletteHtml() {
    return `<div class="zone-palette" id="zonePalette" style="display:none">
      <div class="zp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> <span id="zpTitle">Strefa</span></div>
      <div id="zpBaseSlot"></div>
      <div class="zp-pv"><label class="chk" title="Przerywany obrys obszaru ViTMatte (pas, w którym liczy) i podgląd „Wcięcie w rdzeń” przy przeciąganiu. Wyłącz, żeby oglądać sam wynik"><input type="checkbox" id="zpBandPv" title="Podgląd pasa ViTMatte na obrazie (wł. / wył.)"> 👁 podgląd pasa</label>
        <label class="chk" title="Zielona alfa podglądu strefy na obrazie (w wycięciu „Na tle” podgląd zostaje – to jest wynik)"><input type="checkbox" id="zpAlphaPv" title="Zielona alfa podglądu strefy (wł. / wył.)"> 👁 alfa strefy</label></div>
      <div class="band-seed" id="z4BandSeed"></div>
      <div class="zp-none" id="zpNone"><span class="muted" id="zpNoneTxt">Brak aktywnej strefy „matte”.</span>
        <button type="button" class="btn btn-primary btn-sm" id="zpPickZone" style="display:none" title="Wybierz istniejącą strefę obiektu – jedna strefa „cały obrys” na obiekt; jej ustawienia zmieniasz w tej palecie"></button>
        <button type="button" class="btn btn-primary btn-sm" id="zpNewOutline" title="Nowa strefa ViTMatte wzdłuż całego obrysu obiektu (preset włosy), jedna na obiekt. Grubość pasa ustawiasz punktami pasa. Dłoń / obiekt z przodu: ustaw go w kolejności PRZED tym obiektem (holdout)">＋ Strefa ViTMatte: cały obrys</button>
        <button type="button" class="btn btn-secondary btn-sm" id="zpNewPaint" title="Nowa strefa ViTMatte: namaluj pędzlem pas wzdłuż krawędzi włosy–tło (może wjechać na dłoń). Obszar idzie za obiektem (CoTracker3, lokalnie); poprawka pędzlem w dowolnej klatce = klucz. Części obiektu (np. dłoń) są z obszaru wycinane same">🖌 Maluj obszar włosów (ViTMatte)</button>
        </div>
      <div class="zone-tune" id="z4Tune">
        <div class="zone-tune-sec" id="z4AreaSec"><span class="zp-sub">Obszar ViTMatte</span>
          <div class="seg zone-region" id="z4Region">
            <button type="button" data-r="draw" title="Strefa = obszar namalowany pędzlem (wielokąt też działa); idzie za obiektem, poprawka = klucz">Malowany obszar</button>
            <button type="button" data-r="outline" title="Strefa = pas wokół całego obrysu rdzenia SAM 3.1; rysunek i zaznaczone obiekty są z niego WYCIĘTE">Cały obrys − wykluczenia</button></div>
          <div class="seg zone-brush" id="z4Brush">
            <button type="button" data-b="paint" title="Pędzel: maluj obszar, w którym liczy ViTMatte [G]; rozmiar: suwak pędzla po lewej albo [ ]">🖌 Maluj</button>
            <button type="button" data-b="erase" title="Pędzel: zmaż obszar (ViTMatte tu nie liczy) [R]">⌫ Zmaż</button>
            <button type="button" data-b="bg" title="Pędzel: wymuszone tło (np. linia na ścianie przechodząca przez włosy)">− Tło</button></div>
          <label class="chk" id="z4TrackRow" title="Obszar przenoszony między klatkami punktami CoTrackera lokalnie (każdy fragment za najbliższymi punktami – nadąża za ruchem i deformacją włosów); punkty nigdy na częściach obiektu (np. dłoń) ani na obiektach z przodu. Wyłączone = sztywny kształt"><input type="checkbox" id="z4Track" title="Śledzenie lokalne obszaru (CoTracker3): włączone = obszar nadąża za ruchem i deformacją włosów; wyłączone = sztywny kształt"> śledzenie lokalne (CoTracker3)</label>
          <div class="zp-hint" id="z4RegionHint"></div>
          <div class="zp-band" id="z4BandRow"><button type="button" class="btn btn-ghost btn-xs" id="z4BandTool" style="display:none" title="Dodawanie punktów kontrolnych pasa (Esc = koniec): klik na obrazie przy obrysie = nowy punkt; dwa wąsy – żółty = pas na zewnątrz (w tło), niebieski = do środka (w obiekt); przeciągnij koniec wąsa = grubość w tej klatce (klucz), przeciągnij punkt = jego położenie (klucz); 0 / 0 = przerwa w pasie (np. skóra, dłoń). Między kluczami grubość się animuje, punkty śledzi CoTracker3. PPM na punkcie = usuń klucz tej klatki (ostatni klucz usuwa punkt)">＋ Dodaj punkt pasa</button>
            <button type="button" class="btn btn-ghost btn-xs" id="z4BandClear" title="Usuń wszystkie punkty bieżącej klatki kluczowej pasa (= usuń klatkę kluczową)">🗑 Usuń wszystkie punkty (ta klatka)</button>
            <span class="zp-hint" id="z4BandInfo"></span><div class="band-mode" id="z4BandMode"></div></div>
          <label class="zone-slider band-out-row" id="z4OutRow">Pas na zewnątrz <input type="range" id="z4Out" min="0" max="120" step="1" title="Jak daleko poza obrys rdzenia SAM 3.1 liczy ViTMatte (px plate'u) – ten sam pas co w etapach 3 / 4 / 7. Podgląd przy przeciąganiu: przerywane linie = krawędzie pasa ViTMatte"><b id="z4OutV"></b></label>
          <label class="zone-slider band-in-row">Wcięcie w rdzeń <input type="range" id="z4Inset" min="0" max="40" step="1" title="Ile px w głąb rdzenia SAM 3.1 oddajemy ViTMatte: pewny środek trimapy i miejsce, gdzie alfa = 1, to rdzeń zwężony o tyle px – ten sam pas co w etapach 3 / 4 / 7. Podgląd przy przeciąganiu: przerywane linie = krawędzie pasa, o którym decyduje ViTMatte (w środku pewne 1). 0–2 = twardy obrys SAM 3.1"><b id="z4InsetV"></b></label>
          <label class="zone-slider">Wtopienie wcięcia <input type="range" id="z4Feather" min="0" max="40" step="1" title="Na ilu px wynik ViTMatte przenika się z bazą (rdzeniem) przy granicach obszaru strefy – miękkie wtopienie wcięcia zamiast twardego przejścia. Działa na żywo w podglądzie (bez modelu) i przy „Zastosuj”. Domyślnie 8 px"><b id="z4FeatherV"></b></label>
          <label class="zone-slider band-outf-row" id="z4OutFeatherRow">Wtopienie krawędzi zewn. <input type="range" id="z4OutFeather" min="0" max="80" step="1" title="Na ilu px od ZEWNĘTRZNEJ krawędzi pasa (od strony tła) wynik strefy przechodzi w bazę – cieńsza przerywana linia na podglądzie. 0 = twarde cięcie (jak w etapach 3 / 4 / 7). Nie przelicza modelu – tylko nakładanie"><b id="z4OutFeatherV"></b></label>
          <div class="zp-excl" id="z4ExclRow"><span class="muted" title="Maski innych policzonych obiektów (alpha aktywnej wersji, bez niej rdzeń) wycinane z pasa ViTMatte – zostaje tam wynik bazowy. Dla części tego samego obiektu (twarz, szyja). Obiekt z przodu (dłoń) lepiej ustawić w kolejności przed tym obiektem: holdout = alfa 0 pod nim, a ViTMatte liczy włosy aż do jego krawędzi (CLIP2_A v069 vs v067)">Wyklucz obiekty</span><div id="z4Excl" class="zp-excl-list"></div></div></div>
        <div class="zone-tune-sec" id="z4TempSec"><span class="zp-sub">Wygładzanie w czasie</span>
          <label>Tryb<select id="z4Temporal" class="form-input" title="Wygładzanie alfy w czasie (cała strefa, przy „Zastosuj”). Czas na strefie ~4 Mpx (sh060, 4K; mała strefa to ułamek), od najszybszego: brak ~0 s/kl. → mediana ~0,3 → średnia po przepływie ×1 ~0,5 → średnia ×2 ~1,5 → mediana po przepływie ~2,8 s/kl. Najmniej migotania: średnia po przepływie (CLIP2_A −44…−52 % z promieniem 3…5); mediana przy ruchu prawie nic nie daje. Szczegóły: pozycje listy i opis pod polem.">${TEMPORAL.map(([v, l]) => `<option value="${v}" title="${esc(TEMPORAL_HELP[v] || '')}">${l}</option>`).join('')}</select></label>
          <div class="zp-hint" id="z4TempHint"></div>
          <label class="zone-slider">Promień <input type="range" id="z4TRadius" min="1" max="6" step="1" title="Ile klatek w każdą stronę bierze wygładzanie. Więcej = spokojniej, ale wolniej i większe ryzyko smużenia przy szybkim ruchu. Średnia po przepływie na CLIP2_A: promień 2 −40 %, 3 −44 %, 5 −52 % migotania; promień 5 ≈ 1,5× czasu promienia 3. Kiedy: 3 domyślnie, 5 przy wolnym ruchu i mocnym migotaniu, 2 przy szybkim ruchu."><b id="z4TRadiusV"></b></label>
          <label class="chk"><input type="checkbox" id="z4TScale" title="Wygładzanie na kadrze powiększonym ×2 (do wyniku wraca tylko poprawka). Od 2026-10-28 przepływ liczy się w rozdzielczości plate'u, więc ×1 wyszło tak samo spokojne (CLIP2_A −47 % vs −44 %) i ~3× szybsze (sh060 0,5 vs 1,5 s/kl.). Kiedy ×2: bardzo cienkie, pojedyncze kosmyki; w pozostałych przypadkach wystarczy ×1."> ×2</label></div>
        <div class="zone-tune-sec" id="z4EngSec"><span class="zp-sub">Silnik strefy</span>
          <label>Silnik<select id="zpEngine" class="form-input" title="Czym liczona jest alfa w strefie. ViTMatte = najlepsze pojedyncze kosmyki (ustawienia jak węzeł ViTMatte w Nuke'u poniżej); MatAnyone2 = spójniejsze w czasie przy rozmyciu / ruchu"><option value="vitmatte">ViTMatte</option><option value="matanyone">MatAnyone2</option><option value="auto">auto (dobór per fragment)</option><option value="clean">clean (clean plate)</option><option value="chroma">chroma</option><option value="knn">knn</option></select></label>
          <div class="zp-hint" id="zpEngHint"></div></div>
        <div class="zone-tune-sec" id="z4GrpSec"><span class="zp-sub" title="Obiekt połączony z częściami szwem: strefa liczy na całości grupy. Zaznaczony członek = pewny pierwszy plan (wycięty z obszaru strefy, alfa 1); odznaczony = jego krawędź liczy silnik strefy (np. włosy części „głowa”)">Grupa – pewny pierwszy plan</span>
          <div id="z4Grp" class="zp-excl-list"></div>
          <div class="zp-hint">odznacz obiekt z włosami / futrem – wtedy jego krawędź liczy strefa</div></div>
        <div class="zone-tune-sec" id="z4VmSec"><span class="zp-sub" id="z4VmTitle">ViTMatte (jak węzeł w Nuke'u)</span>
          <label>Model<select id="z4VmModel" class="form-input" title="Model krawędzi w strefie. ViTMatte: base = „High detail” z Nuke'a (domyślnie, najlepsze kosmyki); small = szybszy, więcej mgiełki; base – distinctions = trenowany na innym zbiorze (czasem lepszy na futrze / włóknach) – wagi ViTMatte tylko do użytku niekomercyjnego (zbiór Adobe). BiRefNet_HR-matting (2048 px) / Lucida (1024 px) = wagi MIT, także komercyjnie; liczą alfę z samego obrazu, trimapa tylko przycina wynik do pasa. Modele są w cache Hugging Face (pobiera je Doktor)"><optgroup data-e="vitmatte" label="ViTMatte – tylko niekomercyjnie"><option value="base">ViTMatte base (High detail)</option><option value="small">ViTMatte small (szybki)</option><option value="base-distinctions">ViTMatte base – distinctions</option></optgroup><optgroup data-e="birefnet" label="BiRefNet – wagi MIT, komercyjnie"><option value="birefnet_hr_matting">BiRefNet_HR-matting</option><option value="lucida">Lucida</option></optgroup></select></label>
          <div class="zp-hint" id="z4VmLic"></div>
          <label>Rozdzielczość<select id="z4VmPx" class="form-input" title="Bok kadru strefy, który widzi model (Nuke „Downrez” odwrotnie): 1008 = kadr powiększony jak dla MatAnyone2 (≤ 3×, domyślnie); 504 = ½ (szybciej, pomiar CLIP2_A ≈ wynik Nuke'a); więcej = drobniejsze kosmyki, wolniej i więcej VRAM"><option value="504">504 (½, szybciej)</option><option value="756">756</option><option value="1008">1008 (domyślnie)</option><option value="1512">1512</option><option value="2016">2016 (wolno)</option></select></label>
          <label class="zone-slider">Fine tune <input type="range" id="z4VmUnk" min="0.2" max="0.8" step="0.05" title="Nuke „Fine Tune”: poziom, jaki pas „unknown” dostaje w kanale trimapy. 0.5 = neutralnie (domyślnie); mniej = model ostrożniej dodaje (0.3 obcina kosmyki); więcej = śmielej (0.7 = mgiełka w tle)"><b id="z4VmUnkV"></b></label>
          <div class="zp-hint">Wcięcie w rdzeń (wyżej) = „Erode” pewnego środka trimapy.</div></div>
        <div class="zone-tune-sec" id="z4FinSec"><span class="zp-sub">Wykończenie</span>
          <label class="zone-slider" style="display:none">Czerń <input type="range" id="z4Black" min="0" max="0.3" step="0.01" title="Alfa poniżej tej wartości → 0 (usuwa mgiełkę i drobne śmieci w tle); liniowo do bieli, bez schodka. Przy średniej po przepływie działa PRZED wygładzaniem"><b id="z4BlackV"></b></label>
          <label class="zone-slider" id="z4BlackAfterRow" style="display:none">Czerń po <input type="range" id="z4BlackAfter" min="0" max="0.3" step="0.01" title="Drugie obcięcie czerni PO wygładzaniu w czasie: średnia z sąsiednich klatek tworzy nowe niskie wartości (mgiełka przy szybkim ruchu, CLIP2_A F18) – ten suwak je usuwa (ViTMatte)"><b id="z4BlackAfterV"></b></label>
          <label class="zone-slider" style="display:none">Biel <input type="range" id="z4White" min="0.6" max="1" step="0.01" title="Alfa powyżej tej wartości → 1 (domyka prawie pełne kosmyki i brzeg)"><b id="z4WhiteV"></b></label>
          <label class="zone-slider" style="display:none">Rozmycie <input type="range" id="z4Blur" min="0" max="3" step="0.1" title="Rozmycie Gaussa alfy strefy w px (po wygładzaniu); rdzeń zostaje 1"><b id="z4BlurV"></b></label>
          <button class="btn btn-ghost btn-xs" id="z4TuneReset" title="Przywróć wygładzanie i wykończenie z presetu strefy">↺ z presetu</button></div>
        <div class="muted" id="z4FinNote" title="Czerń / biel / rozmycie całej alfy są teraz w etapie Final Touch (na żywo, bez przeliczania). Tu zostaje tylko próg szumu z presetu, obcinany PRZED wygładzaniem w czasie – bez niego średnia po przepływie robi mgłę z szumu modelu"></div>
        <div class="zp-btns">
          <button type="button" class="btn btn-primary btn-sm" id="z4PvFrame" style="display:none" title="Policz ViTMatte w tej klatce z bieżącymi ustawieniami i wykluczeniami (bez wygładzania w czasie – ono działa przy „Zastosuj”)">▶ Przelicz klatkę</button>
          <button type="button" class="btn btn-primary btn-sm" id="zpStagePv" title="Policz ViTMatte na wybranej bazie tylko na kilku klatkach wokół bieżącej (±2 kl.) – szybkie sprawdzenie ustawień przed długim liczeniem całości. To NIE jest wynik etapu: nic się nie zapisuje jako wersja i nic się nie odblokowuje. Modele czasowe dostają mniej kontekstu niż w pełnym przebiegu – krawędź może się trochę różnić">👁 Podgląd klatek</button>
          <button type="button" class="btn btn-secondary btn-sm" id="zpApply" title="Policz strefę na całym ujęciu (nowa wersja)">✚ Zastosuj na ujęcie</button></div>
      </div>
    </div>`;
  }
  function buildPalette() {   // once at init (hidden); wire() attaches the handlers to its controls
    if ($('zonePalette')) return $('zonePalette');
    const w = $('canvasWrapper') || document.body;
    const box = H.el(paletteHtml());
    w.appendChild(box);
    if (H.makeFloating) H.makeFloating(box, w, '.zp-head', palPos);
    return box;
  }
  function renderPalette() {
    const box = $('zonePalette'); if (!box || !H) return;
    const show = !!(obj() && !S().auxMode && S().stage === H.ST.VITMATTE && (!V() || V().projectLoaded !== false));
    box.style.display = show ? '' : 'none';
    if (!show) return;
    const z = activeMatte();
    $('zpBandPv').checked = bandPvGet();
    $('zpAlphaPv').checked = Z.ov !== 'none';
    const auto = !z && wantsAutoZone();   // the band comes by itself
    $('zpNone').style.display = z || auto ? 'none' : '';
    if (!z && !auto) {
      // one whole-outline zone per object (user decision 2026-10-25): with a zone already there, offer to pick it, never
      // to create another one (the server refuses with 409); the last enabled one is the one that wins in the alpha
      const have = Z.list.filter((x) => x.criterion === 'matte');
      const pick = have.filter((x) => x.enabled).pop() || have[have.length - 1];
      $('zpNoneTxt').textContent = pick ? `Obiekt ma już strefę ${pick.zone_id} – jedna strefa „cały obrys” na obiekt.` : 'Brak aktywnej strefy „matte”.';
      $('zpNewOutline').style.display = pick ? 'none' : '';
      $('zpPickZone').style.display = pick ? '' : 'none';
      $('zpPickZone').dataset.z = pick ? pick.zone_id : '';
      $('zpPickZone').textContent = pick ? `Wybierz ${pick.zone_id}` : '';
    }
    if (auto) ensureOutlineZone();
    $('z4Tune').style.display = z ? '' : 'none';
    $('zpTitle').textContent = z ? `Strefa ${z.zone_id} · ${zEngLabel(z)}${isOutline(z) ? ' · cały obrys' : ''}` : 'Strefa';
    const w = $('canvasWrapper');
    if (w && H.placeFloating) H.placeFloating(box, w, palPos.get());
  }
  function renderExcl(z) {
    const box = $('z4Excl'); if (!box) return;
    const sel = new Set((z && z.params && z.params.exclude_objects) || []);
    const others = (S().objects || []).filter((o) => o.name !== obj());
    if (!others.length) { box.innerHTML = '<span class="muted">brak innych obiektów – np. twarz jako osobny obiekt (etap 1); dłoń z przodu: obiekt ustawiony przed tym (holdout)</span>'; return; }
    box.innerHTML = others.map((o) => {
      const has = o.has ? (o.has.final || o.has.core) : true;
      return `<label class="chk ${has ? '' : 'off'}" title="${has ? `Wytnij maskę „${esc(o.name)}” z pasa ViTMatte (${o.has && o.has.final ? 'alpha aktywnej wersji' : 'rdzeń'})` : 'brak maski – policz ten obiekt najpierw'}">
        <input type="checkbox" data-ex="${esc(o.name)}" ${sel.has(o.name) ? 'checked' : ''} ${has ? '' : 'disabled'}> ${esc(o.name)}</label>`;
    }).join('');
  }

  // members of the selected group entity (PLAN_GROUP_COMPOSE) and which of them are sure foreground of zone z
  // (params.sure_fg; absent = the members in front of the backmost one)
  function groupMembers() {
    const all = S().objects || []; const me = all.find((o) => o.name === obj()); if (!me || me.kind !== 'group') return [];
    const ms = all.filter((o) => o.group === me.name);
    return ms.length > 1 ? ms : [];
  }
  function sureFg(z, ms) {   // default = pipeline.group_sure_default: the members in front of the backmost one
    const sf = z && z.params && z.params.sure_fg;
    if (Array.isArray(sf)) return new Set(sf);
    const back = Math.max(...ms.map((o) => +o.depth_order || 0));
    const front = ms.filter((o) => (+o.depth_order || 0) < back).map((o) => o.name);
    return new Set(front);
  }
  function renderGrp(z) {
    const box = $('z4Grp'); if (!box) return;
    const ms = groupMembers(); const sure = sureFg(z, ms);
    box.innerHTML = ms.map((o) => `<label class="chk" title="${sure.has(o.name) ? 'pewny pierwszy plan – strefa go nie liczy' : 'krawędź liczy silnik strefy'}">
      <input type="checkbox" data-sf="${esc(o.name)}" ${sure.has(o.name) ? 'checked' : ''}> ↳ ${esc(o.name)}</label>`).join('');
  }
  function renderList() {
    const box = $('st4Zones'); if (!box) return;
    if (!Z.list.length) {
      box.innerHTML = DRAW_TOOLS ? '<div class="muted">brak stref – pierwsza kreska pędzlem + (G) albo wielokąt (P) na klatce tworzy nową strefę z wybranym presetem</div>'
        : '<div class="muted">brak strefy – strefa „cały obrys” powstaje sama po wejściu w etap (albo „＋ Nowa strefa”)</div>';
    } else {
      box.innerHTML = Z.list.map((z) => {
        const st = Z.status[z.zone_id] || {};
        const [sl, sc] = STATE_LABEL[st.state] || [st.state || '', ''];
        const matte = z.criterion === 'matte';
        const icon = st.state === 'liczona' ? '⏳' : (matte ? (PRESET_ICON[z.preset] || '▣') : '◐');
        const rs = st.run_stats;
        const stats = rs ? ` · ${ENGINE_LABEL[rs.engine] || rs.engine || ''}${rs.engine_share && Object.keys(rs.engine_share).length ? ' (' + share(rs.engine_share) + ')' : ''}`
          + (rs.seconds != null && rs.frames ? ` · ${(rs.seconds / rs.frames).toFixed(2)} s/kl.` : '') : '';
        const sub = matte ? `${z.zone_id} · ${PRESET_NAME[z.preset] || z.preset} · ${zEngLabel(z)}${z.holdout === false ? ' · bez holdoutu' : ''}`
                          : `${z.zone_id} · ${z.mode === 'add' ? 'Dodaj' : 'Usuń'} ${CRIT_LABEL[z.criterion] || z.criterion}${z.soft ? ' (miękko)' : ''}`;
        return `<div class="zone-card ${z.zone_id === Z.active ? 'cur' : ''} ${z.enabled ? '' : 'off'}" data-z="${esc(z.zone_id)}" style="--zc:${zoneColour(z.zone_id)}" title="Klik = aktywna strefa (jej klucze na osi czasu, kreski trafiają do niej)">
          <span class="zone-ic" title="${esc(matte ? 'preset ' + (PRESET_NAME[z.preset] || z.preset) : 'strefa progowa')}">${icon}</span>
          <span class="zone-name"><b>${esc(st.name || z.zone_id)}</b><br><span class="muted">${esc(sub)}</span></span>
          <span class="zone-meta"><span title="liczba kluczy strefy">🔑 ${st.n_keys ?? 1}</span> · <span class="zs ${sc}" title="${esc(st.run ? 'w wersji ' + st.run : 'stan strefy')}">${esc(sl)}</span><span class="muted">${esc(stats)}</span></span>
          <span class="zone-btns">
            <label class="chk" title="Włącz / wyłącz strefę (wyłączona nie liczy się przy „Zastosuj” / „Przelicz finalnie”)"><input type="checkbox" data-en="${esc(z.zone_id)}" ${z.enabled ? 'checked' : ''}></label>
            ${matte ? `<button class="btn btn-ghost btn-xs" data-eng="${esc(z.zone_id)}" title="Pokaż mapę silników w tej klatce (niebieski = chroma, czerwony = knn, zielony = clean / inne)">🗺</button>` : ''}
            <button class="btn btn-ghost btn-xs" data-del="${esc(z.zone_id)}" title="Usuń strefę (jej klucze przepadają; policzone wersje zostają)">✕</button></span></div>`;
      }).join('');
      box.querySelectorAll('.zone-card').forEach((c) => c.addEventListener('click', (e) => {
        if (e.target.closest('button') || e.target.closest('input')) return;
        setActive(c.dataset.z === Z.active ? null : c.dataset.z);
      }));
      box.querySelectorAll('input[data-en]').forEach((c) => c.addEventListener('change', () => patchZone(c.dataset.en, { enabled: c.checked }, c.checked ? 'włączona' : 'wyłączona')));
      box.querySelectorAll('button[data-eng]').forEach((b) => b.addEventListener('click', () => { setActive(b.dataset.eng); setOv('engines'); refreshPreview(); }));
      box.querySelectorAll('button[data-del]').forEach((b) => b.addEventListener('click', () => deleteZone(b.dataset.del)));
    }
    syncForm(); renderLane();
  }

  function syncForm() {
    if (!$('z4Prompt')) return;
    const z = activeMatte();
    const preset = z ? z.preset : Z.preset;
    document.querySelectorAll('#z4Presets .zp').forEach((b) => b.classList.toggle('active', b.dataset.preset === preset));
    if (document.activeElement !== $('z4Prompt')) $('z4Prompt').value = z ? (z.prompt || '') : Z.prompt;   // not while typing
    applyEngineOptions($('z4Engine'));
    if (Z.engine === 'matanyone' && ((S().info && S().info.edge_engine) === 'birefnet')) Z.engine = 'vitmatte';   // new zones: BiRefNet
    $('z4Engine').value = z ? z.matting : Z.engine;
    if (![...$('z4Engine').options].some((o) => o.value === $('z4Engine').value)) $('z4Engine').value = 'auto';
    $('z4Holdout').checked = z ? z.holdout !== false : Z.holdout;
    syncTune(z);
    const f = V() ? curFrame() : null;
    const hint = $('z4Hint');
    if (hint) {
      if (z) {
        const ks = keyFrames(z);
        const kn = ks.includes(f) ? (keysOf(z)[f].prompt || '') : '';
        hint.innerHTML = !DRAW_TOOLS
          ? `aktywna: <b style="color:${zoneColour(z.zone_id)}">${esc(z.zone_id)}</b>${isOutline(z) ? ' · <b>cały obrys</b>: grubość pasa = punkty pasa („Dodaj punkt”, G, na pasku po lewej)' : ''}`
          : `aktywna: <b style="color:${zoneColour(z.zone_id)}">${esc(z.zone_id)}</b>${isOutline(z) ? ' · <b>cały obrys</b>: G / P = miejsca BEZ włosów' : ''} · klucze ${ks.length} · F${f}: ${ks.includes(f) ? '<b>klucz</b> – kreski poprawiają ten klucz' : 'bez klucza – pierwsza kreska tworzy klucz'}${kn ? ` · uwaga: <i>${esc(kn)}</i>` : ''}`;
      } else if (active()) hint.textContent = DRAW_TOOLS ? `aktywna: ${active().zone_id} (strefa progowa) – kreski utworzą NOWĄ strefę „matte”` : `aktywna: ${active().zone_id} (strefa progowa)`;
      else hint.textContent = Z.list.length ? (DRAW_TOOLS ? 'brak aktywnej strefy – kreska (G / R / P) tworzy nową z tym presetem' : 'brak aktywnej strefy – kliknij kartę strefy') : '';
    }
    const nz = $('z4New');
    if (nz) {   // one whole-outline zone per object (user decision 2026-10-25)
      const have = Z.list.filter((x) => x.criterion === 'matte').map((x) => x.zone_id);
      nz.disabled = have.length > 0;
      nz.title = have.length ? `Obiekt ma już strefę ${have.join(', ')} – jedna strefa „cały obrys” na obiekt; zmieniaj jej ustawienia (klik na karcie)`
        : 'Nowa strefa ViTMatte wzdłuż całego obrysu obiektu – jedna na obiekt';
    }
    const kd = $('z4KeyDel'); if (kd) kd.disabled = !(z && keyFrames(z).includes(f));
    const lock = H.stageState ? H.stageState('vitmatte') : { state: 'todo' };   // needs a current stage-3 MatAnyone version
    for (const id of ['z4ApplyShot', 'zpApply']) {
      const ap = $(id); if (!ap) continue;
      if (ap.dataset.t0 == null) ap.dataset.t0 = ap.title;   // the button's own tooltip, back when unlocked
      ap.disabled = !Z.list.some((x) => x.enabled) || lock.state === 'locked';
      const vPrev = S().info && S().info.stage_runs && S().info.stage_runs.vitmatte;   // „again” when there is a result (2026-10-17)
      if (ap.dataset.l0 == null) ap.dataset.l0 = ap.textContent;
      ap.textContent = vPrev ? '✚ Zastosuj ponownie' : ap.dataset.l0;
      ap.title = lock.state === 'locked' ? `Baza ViTMatte niegotowa – ${lock.why || 'najpierw etap ' + H.ST.MATANYONE + ' albo baza = maska 1+2'}`
        : (vPrev ? `Policz strefy od nowa – nowa wersja zastąpi wynik etapu ${vPrev} (zostaje na liście wersji)` : ap.dataset.t0);
    }
  }

  function syncTune(z) {
    const box = $('z4Tune'); if (!box) return;
    box.classList.toggle('off', !z);
    box.querySelectorAll('input, select, button').forEach((el) => { el.disabled = !z; });
    box.title = z ? '' : 'Najpierw utwórz lub wybierz strefę „matte” (nowa strefa dostaje ustawienia z presetu)';
    $('z4TempSec').style.display = !z || z.matting === 'vitmatte' ? '' : 'none';   // temporal smoothing: ViTMatte engine only
    $('z4AreaSec').style.display = $('z4TempSec').style.display;                   // area / inset / exclusions: ViTMatte only
    // engine of the zone + the ViTMatte node knobs (user request 2026-10-16: „the settings that were in Nuke”)
    const grp = groupMembers();
    $('z4GrpSec').style.display = z && grp.length ? '' : 'none';
    if (z && grp.length) renderGrp(z);
    const vm = !!z && z.matting === 'vitmatte';
    $('z4VmSec').style.display = vm ? '' : 'none';
    if (document.activeElement !== $('zpEngine')) $('zpEngine').value = z ? z.matting : 'vitmatte';
    applyEngineOptions($('zpEngine'));
    $('zpEngHint').textContent = z && !vm ? `Ustawienia modelu (model, rozdzielczość, wygładzanie w czasie) są przy silniku ${$('zpEngine').querySelector('option[value="vitmatte"]').textContent} – ta strefa liczy ${zEngLabel(z)}.` : '';
    if (vm) {
      // models of the stage 5 engine only (objects.js arrow menu). A zone on the other engine's model (set before the
      // engine switch moved zones, 2026-10-21) is moved to the stage engine's default model once, as the switch does
      const stEng = (S().info && S().info.edge_engine) || 'vitmatte';
      const zEng = BIREFNET_MODELS.includes(pval(z, 'vitmatte_model')) ? 'birefnet' : 'vitmatte';
      if (zEng !== stEng && S().info && inZoneFlow() && !Z.engAligned[z.zone_id]) {   // only in the open stage 5, never in the background
        Z.engAligned[z.zone_id] = true;
        patchTune({ vitmatte_model: stEng === 'birefnet' ? 'birefnet_hr_matting' : 'base' }, `model ${stEng === 'birefnet' ? 'BiRefNet_HR-matting' : 'ViTMatte base'} (silnik etapu)`);
      }
      $('z4VmModel').querySelectorAll('optgroup').forEach((g) => { const on = g.dataset.e === stEng || g.dataset.e === zEng; g.hidden = !on; g.disabled = !on; });
      $('z4VmTitle').textContent = zEng === 'birefnet' ? 'BiRefNet' : "ViTMatte (jak węzeł w Nuke'u)";
      if (document.activeElement !== $('z4VmModel')) $('z4VmModel').value = pval(z, 'vitmatte_model');
      // BiRefNet default: one crop at about its own input side (engine/zones.py BIREFNET_MODEL_PX)
      const pxDef = BIREFNET_PX[pval(z, 'vitmatte_model')];
      if (document.activeElement !== $('z4VmPx')) $('z4VmPx').value = String(z.params && z.params.model_px != null ? z.params.model_px : (pxDef || PARAM_DEF.model_px));
      if (document.activeElement !== $('z4VmUnk')) $('z4VmUnk').value = pval(z, 'unknown_value');
      $('z4VmUnkV').textContent = (+$('z4VmUnk').value).toFixed(2);
      const lic = EDGE_MODEL_LICENCE[pval(z, 'vitmatte_model')] || EDGE_MODEL_LICENCE.base;
      $('z4VmLic').textContent = lic[0];
      $('z4VmLic').classList.toggle('zp-hint-nc', lic[1]);
      $('z4VmUnk').parentElement.style.display = BIREFNET_MODELS.includes(pval(z, 'vitmatte_model')) ? 'none' : '';   // no trimap input
    }
    $('z4BlackAfterRow').style.display = 'none';   // finish sliders live in Final Touch (2026-10-22)
    const out = isOutline(z);
    $('z4Region').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.r === (out ? 'outline' : 'draw')));
    $('z4OutRow').style.display = out ? '' : 'none';
    $('z4OutFeatherRow').style.display = out ? '' : 'none';
    $('z4BandRow').style.display = out ? '' : 'none';
    if (!out && bandTool()) H.setTool(S().tool);   // tool off
    bandUi();
    $('z4ExclRow').style.display = out ? '' : 'none';
    $('z4TrackRow').style.display = out || !DRAW_TOOLS ? 'none' : '';
    $('z4Region').style.display = DRAW_TOOLS ? '' : 'none';
    $('z4Brush').style.display = DRAW_TOOLS ? '' : 'none';
    $('z4Track').checked = pval(z, 'track') === 'local';
    $('z4Brush').querySelectorAll('button').forEach((b) => b.classList.toggle('active', Z.brush === b.dataset.b));
    $('z4RegionHint').innerHTML = !DRAW_TOOLS
      ? (out ? 'Strefa = pas wokół <b>całego obrysu</b>. Grubość ustawiasz <b>punktami pasa</b>: narzędzie „Dodaj punkt” (G) na pasku po lewej, potem klik przy obrysie; wąsy: żółty = na zewnątrz, niebieski = do środka; 0 / 0 = przerwa (skóra, dłoń).'
        : 'Ta strefa ma <b>malowany obszar</b> (starszy tryb) – rysowanie jest teraz wyłączone; nowe strefy: „Nowa strefa – cały obrys”.')
      : out
      ? 'ViTMatte liczy wzdłuż <b>całego obrysu</b>. Pędzel G / wielokąt P zaznaczają miejsca, których ViTMatte ma <b>NIE</b> brać pod uwagę (twarz, szyja, skóra); R = tło. Tam zostaje wynik bazowy. Dłoń / obiekt z przodu: ustaw go w kolejności <b>przed</b> tym obiektem (holdout), nie wykluczaj.'
      : 'Namaluj pas wzdłuż krawędzi <b>włosy–tło</b> (może wjechać na dłoń). ViTMatte liczy tylko w nim; obszar idzie za obiektem, poprawka pędzlem w dowolnej klatce = <b>klucz</b>. Części obiektu (np. dłoń, „część” na liście obiektów) są wycinane z obszaru same i liczą się jako pewny pierwszy plan; obiekty z przodu = holdout.';
    if (out) renderExcl(z);
    const busy = (id) => document.activeElement === $(id);
    if (!busy('z4Temporal')) $('z4Temporal').value = pval(z, 'temporal');
    $('z4TempHint').textContent = TEMPORAL_HELP[$('z4Temporal').value] || '';
    const set = (id, k, fmt) => { if (!busy(id)) $(id).value = pval(z, k); $(`${id}V`).textContent = fmt(+$(id).value); };
    set('z4TRadius', 'temporal_radius', (v) => `${v}`);
    if (!busy('z4Inset')) $('z4Inset').value = Math.max(+pval(z, 'force_core_erode_px'), +pval(z, 'fg_erode_px'));
    $('z4InsetV').textContent = `${$('z4Inset').value} px`;
    if ($('z4Feather')) {
      if (!busy('z4Feather')) $('z4Feather').value = z && z.params && z.params.inset_feather_px != null ? z.params.inset_feather_px : 8;
      $('z4FeatherV').textContent = `${$('z4Feather').value} px`;
    }
    $('z4TScale').checked = +pval(z, 'temporal_scale') > 1;
    set('z4Black', 'alpha_black', (v) => v.toFixed(2));
    set('z4BlackAfter', 'alpha_black_after', (v) => v.toFixed(2));
    set('z4White', 'alpha_white', (v) => v.toFixed(2));
    set('z4Blur', 'alpha_blur_px', (v) => `${v.toFixed(1)} px`);
    {   // the finish lives in Final Touch now (user decision 2026-10-22): say what the preset still clips here
      const n = $('z4FinNote');
      if (n) n.textContent = `Próg szumu przed wygładzaniem: czerń ${(+pval(z, 'alpha_black')).toFixed(2)} / biel ${(+pval(z, 'alpha_white')).toFixed(2)} (preset). Poziomy, erode i rozmycie całej alfy: etap Final Touch.`;
    }
    set('z4Out', 'outline_out_px', (v) => `${v} px`);
    {   // literal like every stage (2026-10-24): 0 = a hard outer edge; absent = as the inset feather
      const v = z && z.params && z.params.outer_feather_px != null ? +z.params.outer_feather_px : +$('z4Feather').value;
      if (!busy('z4OutFeather')) $('z4OutFeather').value = v;
      $('z4OutFeatherV').textContent = `${$('z4OutFeather').value} px`;
    }
    renderPalette();
  }
  // live inset preview on the mask (user 2026-10-15): distance of every core pixel to the core edge (chamfer 3-4, two
  // passes, cached per frame), coloured for the slider value without a server round trip
  function coreDistance(img, key) {
    if (Z.insetDist && Z.insetDist.key === key) return Z.insetDist;
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    if (!w || !h) return null;
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, w, h).data;
    const inside = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) inside[i] = px[i * 4] > 127 ? 1 : 0;
    // d: core pixels -> distance to the background; dOut: background pixels -> distance to the core (outline band)
    Z.insetDist = { key, w, h, inside, d: chamfer(inside, 1, w, h), dOut: chamfer(inside, 0, w, h) };
    return Z.insetDist;
  }
  function chamfer(inside, val, w, h) {   // chamfer 3-4 distance of the pixels with inside == val to the nearest other pixel
    const BIG = 1e9; const d = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) d[i] = inside[i] === val ? BIG : 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!d[i]) continue;
      let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + 3);
      if (y > 0) { v = Math.min(v, d[i - w] + 3); if (x > 0) v = Math.min(v, d[i - w - 1] + 4); if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4); }
      d[i] = v;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; if (!d[i]) continue;
      let v = d[i];
      if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
      if (y < h - 1) { v = Math.min(v, d[i + w] + 3); if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4); if (x > 0) v = Math.min(v, d[i + w - 1] + 4); }
      d[i] = v;
    }
    return d;
  }
  // n = inset, out = band outside (outline mode, 0 = none), both in plate px; the mask may be a proxy -> scaled to its width.
  // One band system (user decision 2026-10-22): the band is drawn as its dashed edges only, no fill (the old yellow / blue
  // fill hid the edge being judged)
  function insetCanvas(dist, n, out) {
    const ps = H.plateSize ? H.plateSize() : null; const k = ps && ps[0] ? dist.w / ps[0] : 1;
    const lim = n * 3 * k, limOut = (out || 0) * 3 * k;   // chamfer units
    const cls = new Uint8Array(dist.d.length);   // 0 outside, 1 band, 2 kept core
    for (let i = 0; i < cls.length; i++) {
      if (dist.inside[i]) cls[i] = dist.d[i] > lim ? 2 : 1;
      else if (limOut && dist.dOut[i] <= limOut) cls[i] = 1;
    }
    return bandEdgeCanvas(cls, dist.w, dist.h);
  }
  // dashed edges of the band class (1) of a class image: a band pixel next to another class is an edge pixel, dashed
  // along x + y (the same yellow as the server's band lines)
  function bandEdgeCanvas(cls, w, h) {
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d'); const im = g.createImageData(w, h); const p = im.data;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (cls[i] !== 1 || ((x + y) & 15) >= 10) continue;
      if ((x > 0 && cls[i - 1] !== 1) || (x < w - 1 && cls[i + 1] !== 1) || (y > 0 && cls[i - w] !== 1) || (y < h - 1 && cls[i + w] !== 1)) {
        const o = i * 4; p[o] = 255; p[o + 1] = 212; p[o + 2] = 0; p[o + 3] = 240;
      }
    }
    g.putImageData(im, 0, 0); return c;
  }
  // a server band PNG (255 kept / 128 band / 0 outside) -> its dashed band edges
  function bandPngEdges(img) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    if (!w || !h) return null;
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d'); g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, w, h).data;
    const cls = new Uint8Array(w * h);
    for (let i = 0; i < cls.length; i++) { const v = px[i * 4]; cls[i] = v > 200 ? 2 : v > 90 ? 1 : 0; }
    return bandEdgeCanvas(cls, w, h);
  }
  async function patchTune(upd, what, opts = {}) {
    const z = activeMatte(); if (!z) return;
    await patchZone(z.zone_id, { params: { ...(z.params || {}), ...upd } }, what, opts).catch(() => { syncForm(); });   // refused (e.g. a job runs): controls back to the zone's values
  }

  // ------------------------------------------------------------------------------------------------ step 9: prompt rules, Hough lines, proposals
  const HINT_TXT = (h) => Object.entries(h || {}).map(([k, v]) => `${ENGINE_LABEL[k] || k} ×${(+v).toFixed(2).replace(/\.?0+$/, '')}`).join(', ');
  function renderPlan(pl, bgStrokes) {
    const box = $('z4Plan'); if (!box) return;
    if (!pl || !pl.text) { box.innerHTML = ''; box.style.display = 'none'; return; }
    box.style.display = '';
    const parts = [];
    parts.push(`preset: <b>${esc(pl.preset ? PRESET_NAME[pl.preset] || pl.preset : 'bez zmian')}</b>`);
    if (pl.fg_en) parts.push(`SAM 3.1: „${esc(pl.fg_en)}”`);
    if (pl.distractors && pl.distractors.length) parts.push(`tło: ${esc(pl.distractors.join('; '))}`);
    if (pl.engine_hints && Object.keys(pl.engine_hints).length) parts.push(`silniki: ${esc(HINT_TXT(pl.engine_hints))}${pl.forced_engine ? ` (tylko ${esc(ENGINE_LABEL[pl.forced_engine] || pl.forced_engine)})` : ''}`);
    let html = `→ ${parts.join(' · ')}`;
    if (pl.warnings && pl.warnings.length) html += `<br><span class="warn">⚑ ${esc(pl.warnings.join('; '))}</span>`;
    if (pl.bg_line) {
      const n = bgStrokes ? bgStrokes.length : null;
      html += ` <button class="btn btn-ghost btn-xs" id="z4Lines" title="Szukaj prostych linii w tle strefy (Hough) i dodaj je jako kreski − do klucza tej klatki (do potwierdzenia: Ctrl+Z cofa)">${n == null ? '— Znajdź linie w tle' : n ? `— Dodaj ${n} ${n === 1 ? 'linię' : 'linie'} jako kreski −` : '— brak linii w tle (Hough)'}</button>`;
    }
    box.innerHTML = html;
    const lb = $('z4Lines');
    if (lb) lb.addEventListener('click', () => H.run('z4Lines', () => (bgStrokes && bgStrokes.length ? addBgLines(bgStrokes) : findLines())));
  }
  async function findLines() {
    const z = activeMatte(); const f = curFrame();
    const prompt = z ? (z.prompt || '') : ($('z4Prompt') ? $('z4Prompt').value.trim() : '');
    const d = await api('POST', `/api/object/${obj()}/zones/interpret`, { prompt, frame: f, zone_id: z ? z.zone_id : null });
    Z.plan = { ...d, f };
    renderPlan(d.plan, d.bg_strokes);
    status(d.bg_strokes.length ? `F${f}: ${d.bg_strokes.length} linii w tle – „Dodaj … jako kreski −” wpisuje je do klucza strefy (podgląd przeliczy się)` : `F${f}: brak prostych linii w tle strefy`, d.bg_strokes.length ? 'ok' : 'warn');
    V().render();
  }
  async function addBgLines(strokes) {
    const z = activeMatte(); if (!z) { status('Najpierw strefa „matte” (kreska G / P) – kreski − dopiszą się do jej klucza', 'warn'); return; }
    const f = curFrame();
    const keys = keysOf(z);
    let base = keys[f];
    if (!base) {
      const reg = await regionFor(z, f, true);
      const poly = reg && reg.contours && reg.contours.length ? reg.contours.slice().sort((a, b) => b.length - a.length)[0] : null;
      base = poly ? { polygon: poly, strokes: [], bg_strokes: [], prompt: '' } : null;
      if (!base) { status(`F${f}: brak obszaru strefy – narysuj obszar (G / P)`, 'warn'); return; }
    }
    const minus = strokes.map((st) => ({ p: st.p, radius_px: st.radius_px || 3 }));
    const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, { polygon: base.polygon, strokes: base.strokes, bg_strokes: base.bg_strokes.concat(minus) });
    Z.undo.push({ zid: z.zone_id, f, prev: keys[f] ? { polygon: base.polygon, strokes: base.strokes, bg_strokes: base.bg_strokes, prompt: base.prompt || '' } : null });
    replaceZone(d.zone, Z.status[z.zone_id]);
    Z.status[z.zone_id] = { ...(Z.status[z.zone_id] || {}), state: 'do przeliczenia', n_keys: d.keys.length, keys: d.keys };
    if (Z.plan) Z.plan.bg_strokes = [];
    renderList(); renderPlan(Z.plan && Z.plan.plan, []);
    status(`Klucz strefy ${z.zone_id} F${f}: dodano ${minus.length} kresek − z linii w tle (Ctrl+Z cofa). Liczę podgląd…`, 'ok');
    H.loadStages().catch(() => {});
    schedulePreview(f, 0);
  }
  async function propose() {
    if (!obj()) return;
    const f = curFrame();
    const z = activeMatte();
    const prompt = z ? (z.prompt || '') : ($('z4Prompt') ? $('z4Prompt').value.trim() : '');
    status(`F${f}: szukam propozycji strefy (SAM 3.1 na kadrze obiektu minus rdzeń + miękkie krawędzie)…`);
    const d = await api('POST', `/api/object/${obj()}/zones/propose`, { frame: f, prompt: prompt || null });
    Z.props = { f, list: d.proposals || [], preset: d.preset || Z.preset, hover: null, fg: d.fg, sam3: d.sam3 };
    renderProps();
    status(Z.props.list.length ? `F${f}: ${Z.props.list.length} propozycji (SAM 3.1 „${d.fg || '–'}”${d.sam3 ? '' : ' bez maski'}) – ✓ tworzy strefę, ✕ odrzuca, najechanie podświetla`
                                : `F${f}: brak propozycji – rdzeń pokrywa wszystko, co SAM 3.1 widzi; narysuj strefę ręcznie (G / P)`, Z.props.list.length ? 'ok' : 'warn');
    V().render();
  }
  function renderProps() {
    const box = $('z4Props'); if (!box) return;
    const P = Z.props;
    if (!P || !P.list.length) { box.innerHTML = ''; box.style.display = 'none'; return; }
    box.style.display = '';
    box.innerHTML = `<div class="muted">Propozycje stref F${P.f} (preset ${esc(PRESET_NAME[P.preset] || P.preset)}):</div>` + P.list.map((pr) =>
      `<div class="zone-prop" data-i="${pr.id}" title="Propozycja #${pr.id}: ${pr.area_px} px, ${esc(pr.side)}, źródło ${esc(pr.source === 'both' ? 'SAM 3.1 + miękka krawędź' : pr.source === 'sam3' ? 'SAM 3.1 poza rdzeniem' : 'miękka krawędź')}">
        <span>#${pr.id} · ${esc(pr.side)} · ${pr.area_px} px</span>
        <button class="btn btn-primary btn-xs" data-acc="${pr.id}" title="Utwórz strefę „matte” z tego wielokąta (klucz w tej klatce) i policz podgląd">✓ strefa</button>
        <button class="btn btn-ghost btn-xs" data-rej="${pr.id}" title="Odrzuć propozycję">✕</button></div>`).join('')
      + `<div class="row"><button class="btn btn-ghost btn-xs" id="z4PropsClear" title="Schowaj wszystkie propozycje">Schowaj</button></div>`;
    box.querySelectorAll('.zone-prop').forEach((r) => {
      r.addEventListener('mouseenter', () => { P.hover = +r.dataset.i; V().render(); });
      r.addEventListener('mouseleave', () => { P.hover = null; V().render(); });
    });
    box.querySelectorAll('button[data-acc]').forEach((b) => b.addEventListener('click', () => H.run('z4Propose', () => acceptProposal(+b.dataset.acc))));
    box.querySelectorAll('button[data-rej]').forEach((b) => b.addEventListener('click', () => { P.list = P.list.filter((x) => x.id !== +b.dataset.rej); renderProps(); V().render(); }));
    const cl = $('z4PropsClear'); if (cl) cl.addEventListener('click', () => { Z.props = null; renderProps(); V().render(); });
  }
  async function acceptProposal(i) {
    const P = Z.props; if (!P) return;
    const pr = P.list.find((x) => x.id === i); if (!pr) return;
    const prompt = $('z4Prompt') ? $('z4Prompt').value.trim() : Z.prompt;
    const body = { mode: 'add', criterion: 'matte', frame: P.f, polygon: pr.polygon, strokes: [], bg_strokes: [],
                   preset: P.preset || Z.preset, matting: Z.engine, prompt: activeMatte() ? '' : prompt, holdout: Z.holdout, static: false, bg_static: true };
    let d;
    try { d = await api('POST', `/api/object/${obj()}/zones/new?preview_frame=${P.f}`, body); }
    catch (e) { const mf = missingFront(e); if (mf) { frontDialog(mf, { noHoldout: async () => { Z.holdout = false; $('z4Holdout').checked = false; await acceptProposal(i); } }); return; } throw e; }
    P.list = P.list.filter((x) => x.id !== i);
    if (!P.list.length) Z.props = null;
    Z.active = d.zone.zone_id; Z.undo.push({ zid: Z.active, f: P.f, prev: null, created: true });
    await load();
    renderProps();
    toast(`Strefa ${Z.active} z propozycji #${i} (${PRESET_NAME[d.zone.preset] || d.zone.preset}) – klucz F${P.f}`);
    setPreview(d.zone.zone_id, P.f, d.preview);
    H.loadStages().catch(() => {});
  }
  function drawProps(ctx, k, f) {
    const P = Z.props; if (!P || P.f !== f || !P.list.length) return;
    const v = V();
    ctx.save();
    try {
      ctx.lineWidth = 1.5 / v.zoom; ctx.font = `${12 / v.zoom}px "IBM Plex Sans", sans-serif`; ctx.textBaseline = 'top';
      for (const pr of P.list) {
        const hot = P.hover === pr.id;
        ctx.setLineDash(hot ? [] : [5 / v.zoom, 4 / v.zoom]);
        ctx.strokeStyle = hot ? '#fde047' : 'rgba(200,200,204,0.85)'; ctx.fillStyle = hot ? 'rgba(200,200,204,0.25)' : 'rgba(200,200,204,0.10)';
        ctx.beginPath(); pr.polygon.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); ctx.fill(); ctx.stroke();
        const [bx, by] = [pr.bbox[0] * k, pr.bbox[1] * k];
        ctx.setLineDash([]); ctx.fillStyle = hot ? '#fde047' : 'rgba(200,200,204,0.85)';
        const label = `#${pr.id}`; const tw = ctx.measureText(label).width + 6 / v.zoom;
        ctx.fillRect(bx, by - 15 / v.zoom, tw, 15 / v.zoom); ctx.fillStyle = '#2a2a2e'; ctx.fillText(label, bx + 3 / v.zoom, by - 14 / v.zoom);
      }
    } finally { ctx.restore(); }
  }

  function setActive(zid) {
    Z.active = zid;
    Z.preview = null;
    Z.inset = null;   // the slider band belongs to the previous zone
    renderList();
    const z = activeMatte();
    if (z && z.prompt) api('POST', `/api/object/${obj()}/zones/interpret`, { prompt: z.prompt }).then((d) => { Z.plan = { ...d, f: null }; renderPlan(d.plan, null); }).catch(() => renderPlan(null));
    else renderPlan(null);
    if (z) status(`Strefa ${z.zone_id} aktywna (${PRESET_NAME[z.preset] || z.preset}, ${zEngLabel(z)}): maluj G / R / P w klatce = klucz; „Zastosuj na ujęcie”, gdy podgląd pasuje.`);
    onFrame();
    V().render();
  }

  async function patchZone(zid, body, what, opts = {}) {
    try {
      const d = await api('PATCH', `/api/object/${obj()}/zones/${zid}`, body);
      replaceZone(d.zone, d.status);
      renderList();
      // the model runs only on demand (user rule 2026-10-17: sliders never start the model – it loads the system):
      // „▶ Przelicz klatkę” / „👁 Podgląd klatek” / a change of the ViTMatte base; finish sliders stay live without it
      const pvNote = !opts.noPreview && Z.active === zid ? ' · podgląd modelu: „👁 Podgląd klatek”' : '';
      status(`Strefa ${zid}: ${what || d.changed.join(', ')} – etap ${H.ST.VITMATTE} ViTMatte do przeliczenia („Zastosuj na ujęcie”)${pvNote}`, 'ok');
      H.loadStages().catch(() => {});
      // a band slider of a whole-outline zone moves the shared band of the object: the palettes of stages 3 / 4 / 7 again
      if (H.reloadBandPalettes && body.params && ['outline_out_px', 'fg_erode_px', 'force_core_erode_px', 'inset_feather_px', 'outer_feather_px'].some((k) => k in body.params)) H.reloadBandPalettes({ zones: false }).catch(() => {});
      if (opts.preview && Z.active === zid && (keyFrames(d.zone).includes(curFrame()) || isOutline(d.zone))) schedulePreview(curFrame(), 0);
      return d;
    } catch (e) { status(`Strefa ${zid} NIE zmieniona: ${e.message}`, 'err'); toast(e.message, 5000); throw e; }
  }

  async function deleteZone(zid) {
    const z = Z.list.find((x) => x.zone_id === zid); if (!z) return;
    if (!(await V().uiConfirm(`Usunąć strefę ${zid}${z.prompt ? ` („${z.prompt}”)` : ''}?\nJej klucze (${keyFrames(z).length}) przepadają; policzone wersje zostają.`, { ok: 'Usuń', tone: 'error' }))) return;
    try {
      await api('DELETE', `/api/object/${obj()}/zones/${zid}`);
      if (Z.active === zid) { Z.active = null; Z.preview = null; }
      await load(); H.loadStages().catch(() => {});
      status(`Strefa ${zid} usunięta`, 'ok'); V().render();
    } catch (e) { status(`Nie usunięto ${zid}: ${e.message}`, 'err'); }
  }

  // ------------------------------------------------------------------------------------------------ loading
  async function load() {
    if (!obj()) { Z.list = []; Z.status = {}; Z.active = null; renderList(); return; }
    const d = await api('GET', `/api/object/${obj()}/zones`);
    Z.list = (d.zones || []).filter((z) => !(z.params && z.params.source === 'brush')); Z.status = d.status || {};   // brush = stage 2 (brush.js)
    Z.regionCache = {};
    if (Z.active && !Z.list.some((z) => z.zone_id === Z.active)) Z.active = null;
    if (!Z.active) { const m = Z.list.filter((z) => z.criterion === 'matte'); const o = m.filter(isOutline); if (o.length === 1 || m.length === 1) Z.active = (o[0] || m[0]).zone_id; }
    Z.loadedObj = obj();
    renderList();
    // the cache is empty now: ask for this frame's band at once (a finished job reloads the object – the lines vanished
    // until the next frame change, user report 2026-10-22); as replaceZone
    const a = activeMatte();
    if (a && inZoneFlow() && isOutline(a) && bandPvGet()) setTimeout(() => { const b = activeMatte(); if (b && b.zone_id === a.zone_id && !(Z.regionCache[b.zone_id] || {})[curFrame()]) regionFor(b, curFrame(), false); }, 0);
  }
  // stage 5 opens on the band of the chosen base mask (user decision 2026-10-21): the whole-outline zone is created by
  // itself – no preview, no undo entry, no key on the lane; keys come only from band points
  const wantsAutoZone = () => !!obj() && !(Z.autoFailed || {})[obj()] && !Z.list.some((x) => x.criterion === 'matte');
  async function ensureOutlineZone() {
    const o = obj();
    if (!o || Z.autoBusy || Z.loadedObj !== o || !wantsAutoZone()) return;
    Z.autoBusy = true;
    try {
      let def = {};
      try { def = (await api('GET', '/api/zones/preset_defaults/hair')).params || {}; } catch (e) { /* older server: engine defaults */ }
      const body = { mode: 'add', criterion: 'matte', frame: curFrame(), polygon: [], strokes: [], bg_strokes: [], preset: 'hair', matting: 'vitmatte',
                     prompt: '', holdout: Z.holdout, static: false, bg_static: true, params: { ...def, region: 'outline', vitmatte_model: undefined } };   // model: the stage engine (server)
      const d = await api('POST', `/api/object/${o}/zones/new`, body);
      if (obj() !== o) return;
      Z.active = d.zone.zone_id;
      await load(); H.loadStages().catch(() => {}); onFrame(); V().render();
    } catch (e) {
      (Z.autoFailed = Z.autoFailed || {})[o] = true; renderPalette();
      status(`Pas „cały obrys” nie powstał sam (${e.message || e}) – „＋ Strefa ViTMatte: cały obrys”`, 'warn');
    } finally { Z.autoBusy = false; }
  }
  function reset() {   // object (re)load: a different object forgets the active zone / undo / preview; the same one (after a job) keeps them
    const same = Z.obj === obj();
    Z.obj = obj();
    Z.list = []; Z.status = {}; Z.regionCache = {};
    if (!same) { Z.active = null; Z.preview = null; Z.undo = []; Z.plan = null; Z.props = null; renderProps(); renderPlan(null); }
    clearTimeout(Z.pvTimer); clearTimeout(Z.drawTimer); Z.drawPending = null; renderLane();
  }

  async function loadCleanplate() {
    const row = $('z4CleanRow'); if (!row) return;
    try { Z.cp = await api('GET', '/api/project/cleanplate_auto'); } catch (e) { Z.cp = null; }
    try { Z.cpNone = localStorage.getItem(`md_cp_none_${(V().projectData || {}).work || ''}`) === '1'; } catch (e) { Z.cpNone = false; }
    const txt = $('z4CleanTxt'); const cp = Z.cp;
    const has = cp && cp.source !== 'brak';
    ['z4CpAuto', 'z4CpFile'].forEach((id) => { $(id).style.display = has || Z.cpNone ? 'none' : ''; });
    $('z4CpNone').textContent = Z.cpNone ? 'cofnij „bez clean plate’u”' : 'Bez clean plate’u';
    $('z4CpNone').style.display = has ? 'none' : '';
    if (!cp) txt.textContent = 'Clean plate: brak projektu';
    else if (cp.source === 'projekt') txt.textContent = 'Clean plate: z projektu (plik) ✓ – silnik clean dostępny';
    else if (cp.source === 'auto') txt.textContent = `Clean plate: auto z ujęcia ✓${cp.auto && cp.auto.confidence_mean != null ? ` (pewność ${(+cp.auto.confidence_mean).toFixed(2)})` : ''}${cp.auto && cp.auto.inpaint_backend ? ` · dziury: ${cp.auto.inpaint_backend}` : ''}`;
    else txt.textContent = Z.cpNone ? 'Clean plate: brak (pominięty – silniki chroma / knn / MatAnyone2)' : 'Clean plate: brak →';
    row.classList.toggle('warn', !!cp && cp.source === 'brak' && !Z.cpNone);
  }

  // ------------------------------------------------------------------------------------------------ drawing -> zone / key
  function wantsDraw() { return DRAW_TOOLS && inZoneFlow(); }
  function onDraw(f) {
    clearTimeout(Z.drawTimer); Z.drawPending = f;
    Z.drawTimer = setTimeout(() => commitDraw(f).catch(() => {}), 600);
    status(activeMatte() ? `Klatka ${f}: kreska strefy ${Z.active} – zapisuję klucz…` : `Klatka ${f}: pierwsza kreska – tworzę nową strefę (${PRESET_NAME[Z.preset]})…`);
  }
  async function commitDraw(f) {
    clearTimeout(Z.drawTimer); Z.drawPending = null;
    if (Z.drawBusy) { Z.drawPending = f; Z.drawTimer = setTimeout(() => commitDraw(f).catch(() => {}), 300); return; }
    const s = S();
    const strokes = s.strokes[f] || [], polys = s.polys[f] || [];
    if (!strokes.length && !polys.length) return;
    const plus = strokes.filter((x) => x.mode === 'add').map((x) => ({ p: x.p, radius_px: x.radius_px }));
    const erasing = Z.brush === 'erase';   // palette "Zmaż": the R strokes remove from the zone drawing
    const minus = erasing ? [] : strokes.filter((x) => x.mode === 'remove').map((x) => ({ p: x.p, radius_px: x.radius_px }));
    const erase = erasing ? strokes.filter((x) => x.mode === 'remove').map((x) => ({ p: x.p, radius_px: x.radius_px, erase: true })) : [];
    const polyAdd = polys.filter((q) => q.mode === 'add').slice(-1)[0] || null;
    if (polys.some((q) => q.mode !== 'add')) toast('Wielokąt usuwający (PPM) nie działa w strefie „matte” – tło zaznacza pędzel − (R)', 5000);
    s.strokes[f] = []; s.polys[f] = [];
    Z.drawBusy = true;
    try {
      let z = activeMatte();
      if (!z) {
        if (!polyAdd && !plus.length) {
          s.strokes[f] = strokes.concat(s.strokes[f] || []);
          status('Nowa strefa: najpierw obszar – pędzel + (G) albo wielokąt (P). Sama kreska − (R) nie tworzy strefy.', 'warn'); return;
        }
        const body = { mode: 'add', criterion: 'matte', frame: f, polygon: polyAdd ? polyAdd.p : [], strokes: plus, bg_strokes: minus,
                       preset: Z.preset, matting: Z.engine, prompt: Z.prompt, holdout: Z.holdout, static: false, bg_static: true };
        if (Z.newPainted) {   // palette "Maluj obszar włosów": ViTMatte with the preset's settings + local tracking
          body.matting = 'vitmatte'; body.params = { ...(await presetParams(Z.preset)), track: 'local' }; Z.newPainted = false;
        }
        let d;
        try { d = await api('POST', `/api/object/${obj()}/zones/new?preview_frame=${f}`, body); }
        catch (e) {
          const mf = missingFront(e);
          if (mf) { s.strokes[f] = strokes.concat(s.strokes[f] || []); s.polys[f] = polys.concat(s.polys[f] || []); V().render();
                    frontDialog(mf, { noHoldout: async () => { Z.holdout = false; $('z4Holdout').checked = false; await commitDraw(f); } }); return; }
          throw e;
        }
        Z.active = d.zone.zone_id; Z.undo.push({ zid: Z.active, f, prev: null, created: true });
        await load();
        toast(`Nowa strefa ${Z.active} (${PRESET_NAME[Z.preset]}) – klucz strefy F${f}`);
        setPreview(d.zone.zone_id, f, d.preview);
        H.loadStages().catch(() => {});
        return;
      }
      const keys = keysOf(z);
      let base = keys[f], created = false;
      if (!base) {
        created = true;
        let reg = (Z.regionCache[z.zone_id] || {})[f] || null;
        if (!reg) { status(`Klatka ${f}: obszar strefy z propagacji (pierwsze wywołanie może trwać)…`); reg = await regionFor(z, f, true); }
        // outline mode: the key's drawing = the EXCLUSION, so a new key starts from the propagated exclusion (never the band)
        const src = isOutline(z) ? (reg && reg.exclude_contours) || [] : (reg && reg.contours) || [];
        const poly = src.length ? src.slice().sort((a, b) => b.length - a.length)[0] : null;
        if (isOutline(z)) base = { polygon: poly || [], strokes: [], bg_strokes: [] };
        else if (!poly) {   // propagation unavailable (busy server): the nearest key's drawing
          const ks = keyFrames(z); const near = ks.reduce((a, b) => (Math.abs(b - f) < Math.abs(a - f) ? b : a));
          base = { polygon: keys[near].polygon, strokes: keys[near].strokes, bg_strokes: keys[near].bg_strokes };
          toast(`Obszar z propagacji niedostępny – klucz F${f} startuje z rysunku klucza F${near}`, 5000);
        } else base = { polygon: poly, strokes: [], bg_strokes: [] };
      }
      if (!isOutline(z) && !polyAdd && !base.polygon.length && !plus.length && !base.strokes.some((x) => !x.erase)) {
        s.strokes[f] = strokes.concat(s.strokes[f] || []);
        status(`Klucz F${f}: brak obszaru – narysuj obszar (G / P), kreska − tylko zaznacza tło`, 'warn'); return;
      }
      const iter = ($('z4Iter') && $('z4Iter').value.trim()) || '';
      const body = { polygon: polyAdd ? polyAdd.p : base.polygon, strokes: base.strokes.concat(plus, erase), bg_strokes: base.bg_strokes.concat(minus) };
      if (iter) body.prompt = iter;                       // step 7: iteration note stored in this key (+ prompt_history)
      const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, body);
      Z.undo.push({ zid: z.zone_id, f, prev: created ? null : { polygon: base.polygon, strokes: base.strokes, bg_strokes: base.bg_strokes, prompt: base.prompt || '' } });
      if (iter && $('z4Iter')) $('z4Iter').value = '';
      replaceZone(d.zone, Z.status[z.zone_id]);
      Z.status[z.zone_id] = { ...(Z.status[z.zone_id] || {}), state: 'do przeliczenia', n_keys: d.keys.length, keys: d.keys };
      renderList();
      if (created) toast(`Klucz strefy F${f} (${z.zone_id})`);
      status(`Klucz strefy ${z.zone_id} F${f} zapisany (${plus.length} kresek +, ${minus.length} −${erase.length ? `, ${erase.length} zmazań` : ''}${polyAdd ? ', wielokąt' : ''}${iter ? `, uwaga „${iter}”` : ''}). Ctrl+Z cofa. Liczę podgląd…`, 'ok');
      H.loadStages().catch(() => {});
      schedulePreview(f, 0);
    } catch (e) {
      s.strokes[f] = strokes.concat(s.strokes[f] || []); s.polys[f] = polys.concat(s.polys[f] || []);
      const m = `Kreska strefy F${f} NIE zapisana: ${e.message}`; status(m, 'err'); toast(m, 6000);
    } finally { Z.drawBusy = false; V().render(); }
  }

  async function undo(f) {
    const u = Z.undo[Z.undo.length - 1];
    if (!obj()) return false;
    if (!u) { status('Nic do cofnięcia w tym etapie', 'warn'); return true; }   // never the SAM 3.1 key history from here
    Z.undo.pop();
    try {
      if (u.bandpts) {   // a band point edit (static/bandpoints.js keeps the state; the marker keeps the order here)
        return window.MD_BANDPTS ? window.MD_BANDPTS.undo(f) : true;
      } else if (u.created && u.prev === null && Z.list.some((z) => z.zone_id === u.zid) && keyFrames(Z.list.find((z) => z.zone_id === u.zid)).length <= 1) {
        await api('DELETE', `/api/object/${obj()}/zones/${u.zid}`);
        if (Z.active === u.zid) Z.active = null;
        Z.preview = null; await load(); status(`Cofnięto: strefa ${u.zid} usunięta (była tylko jedna kreska)`, 'ok');
      } else if (u.prev === null) {
        const d = await api('DELETE', `/api/object/${obj()}/zones/${u.zid}/keys/${u.f}`);
        replaceZone(d.zone); Z.preview = null; renderList(); status(`Cofnięto: klucz strefy ${u.zid} F${u.f} usunięty`, 'ok');
      } else {
        const d = await api('PUT', `/api/object/${obj()}/zones/${u.zid}/keys/${u.f}`, u.prev);
        replaceZone(d.zone); renderList(); status(`Cofnięto kreskę strefy ${u.zid} F${u.f}`, 'ok');
        if (u.f === f) schedulePreview(f, 0);
      }
      H.loadStages().catch(() => {}); V().render();
    } catch (e) { status(`Cofanie strefy: ${e.message}`, 'err'); }
    return true;
  }

  async function deleteKey(zid, f) {
    const z = Z.list.find((x) => x.zone_id === zid); if (!z) return;
    const ks = keyFrames(z);
    if (!ks.includes(f)) { toast(`F${f}: brak klucza strefy ${zid}`); return; }
    if (ks.length <= 1) {   // a zone needs at least one key: the last one goes with the zone (asked first)
      if (!(await V().uiConfirm(`F${f} to jedyny klucz strefy ${zid}.\nUsunąć całą strefę?`, { ok: 'Usuń strefę', tone: 'error' }))) return;
      try {
        await api('DELETE', `/api/object/${obj()}/zones/${zid}`);
        if (Z.active === zid) { Z.active = null; Z.preview = null; }
        await load(); H.loadStages().catch(() => {});
        status(`Strefa ${zid} usunięta (z jedynym kluczem F${f})`, 'ok'); V().render();
      } catch (e) { status(`Nie usunięto ${zid}: ${e.message}`, 'err'); }
      return;
    }
    if (!(await V().uiConfirm(`Usunąć klucz strefy ${zid} w klatce ${f}?`, { ok: 'Usuń' }))) return;
    try {
      const d = await api('DELETE', `/api/object/${obj()}/zones/${zid}/keys/${f}`);
      replaceZone(d.zone); Z.status[zid] = { ...(Z.status[zid] || {}), state: 'do przeliczenia', n_keys: d.keys.length, keys: d.keys };
      if (Z.preview && Z.preview.f === f) Z.preview = null;
      renderList(); onFrame(); H.loadStages().catch(() => {});
      status(`Klucz strefy ${zid} F${f} usunięty – obszar w tej klatce znów z propagacji`, 'ok'); V().render();
    } catch (e) { status(`Nie usunięto klucza: ${e.message}`, 'err'); }
  }

  // ------------------------------------------------------------------------------------------------ propagated region (frames without a key)
  async function regionFor(z, f, wait) {
    const c = Z.regionCache[z.zone_id] || (Z.regionCache[z.zone_id] = {});
    if (c[f]) return c[f];
    const k = `${z.zone_id}_${f}`;
    if (Z.regionBusy[k]) return wait ? Z.regionBusy[k] : null;
    let stale = false, busy409 = false;
    const p = (async () => {
      try {
        const d = await api('GET', `/api/object/${obj()}/zones/${z.zone_id}/region/${f}`);
        // the zone changed while this was asked (a slider, a band point – its cache was dropped): the answer belongs
        // to the old settings, ask again for this frame (user report 2026-10-21: the band vanished after a slider)
        stale = Z.regionCache[z.zone_id] !== c;
        if (!stale) { c[f] = d; if (curFrame() === f) V().render(); }
        // the band on the quick stand-in mask 1+2 (the exact one comes from the background batch, 2026-10-24): asked
        // again once, when the mask is ready (MD_BANDPTS.watchMask12 polls the cheap status)
        if (!stale && d && d.approx && window.MD_BANDPTS && window.MD_BANDPTS.watchMask12) {
          window.MD_BANDPTS.watchMask12(() => { if (c[f] === d) { delete c[f]; if (curFrame() === f) regionFor(z, f, false); } });
        }
        return stale ? null : d;
      }
      catch (e) {
        if (e.status !== 409 && Z.regionCache[z.zone_id] === c) c[f] = { contours: [], error: e.message };
        if (e.status === 409) busy409 = true;   // another job runs (e.g. the zone preview): try again shortly
        return null;
      }
      finally {
        delete Z.regionBusy[k];
        const a = activeMatte();
        const again = a && a.zone_id === z.zone_id && curFrame() === f && bandPvGet();
        if (stale && again) regionFor(a, f, false);
        else if (busy409 && again && (Z.regionRetry[k] = (Z.regionRetry[k] || 0) + 1) <= 20) setTimeout(() => regionFor(activeMatte() || a, f, false), 600);
        else if (!busy409) delete Z.regionRetry[k];
      }
    })();
    Z.regionBusy[k] = p;
    return wait ? p : null;
  }
  let frameTimer = null;
  function onFrame() {
    renderLane(); syncForm();
    if (Z.props && Z.props.f !== curFrame()) { Z.props = null; renderProps(); }
    clearTimeout(frameTimer);
    const z = activeMatte(); if (!z || !inZoneFlow()) return;
    const f = curFrame();
    if ((keyFrames(z).includes(f) && !isOutline(z)) || (Z.regionCache[z.zone_id] || {})[f]) return;   // outline: the band also on keys
    // no automatic CoTracker run on every frame (user rule 2026-10-17: heavy work only on a click) – the area is computed
    // by „▶ Przelicz klatkę” / „👁 Podgląd klatek” / painting a correction, and stays cached for the frames visited
    // „whole outline” with nothing painted: the area comes straight from the core on the server (no CoTracker, a fraction
    // of a second) – that one still follows the frame by itself
    const cheap = isOutline(z) && !hasDrawing(z);
    frameTimer = setTimeout(() => {
      if (curFrame() !== f || !bandPvGet()) return;
      if (cheap) {
        const ask = (n) => {   // busy (another job): ask again a little later instead of skipping (user report 2026-10-21)
          if (curFrame() !== f || (Z.regionCache[z.zone_id] || {})[f]) return;
          if (!document.body.classList.contains('job-busy')) regionFor(z, f, false);
          else if (n > 0) setTimeout(() => ask(n - 1), 500);
        };
        ask(20);
      }
      else status(`F${f}: obszar strefy ${z.zone_id} jeszcze niepoliczony w tej klatce – „👁 Podgląd klatek” policzy go (CoTracker3) razem z podglądem`);
    }, 350);
  }

  // ------------------------------------------------------------------------------------------------ preview
  function setOv(k) {
    if (Z.ov && Z.ov !== 'none') Z.ovLast = Z.ov;
    Z.ov = k; document.querySelectorAll('#z4Ov button').forEach((b) => b.classList.toggle('active', b.dataset.ov === k));
    const ap = $('zpAlphaPv'); if (ap) ap.checked = k !== 'none';
    V().render();
  }
  // one-frame model preview („▶ Przelicz klatkę”) switched off (user decision 2026-10-22): it blended the zone with a
  // ramp of its own in the browser and had no temporal smoothing, so it did not match „Zastosuj”; „👁 Podgląd klatek”
  // (±2 frames, the same code as the result) is the preview. The code stays behind this switch.
  const FRAME_PV = false;
  function schedulePreview(f, delay) {
    clearTimeout(Z.pvTimer);
    if (!FRAME_PV) return;
    Z.pvTimer = setTimeout(() => runPreview(f).catch(() => {}), delay == null ? 600 : delay);
  }
  async function runPreview(f) {
    const z = activeMatte(); if (!z || !obj()) return;
    if (Z.pvBusy) { Z.pvAgain = f; return; }
    Z.pvBusy = true; const info = $('z4PvInfo'); if (info) info.textContent = `⏳ liczę podgląd F${f} (${zEngLabel(z)})…`;
    try {
      const d = await api('POST', `/api/object/${obj()}/zones/${z.zone_id}/preview/${f}`);
      setPreview(z.zone_id, f, d.preview);
      if (!(Z.regionCache[z.zone_id] || {})[f] && (isOutline(z) || !keyFrames(z).includes(f))) regionFor(z, f, false);   // the band of this frame, on demand
    } catch (e) {
      const mf = missingFront(e);
      if (mf) { frontDialog(mf, { noHoldout: () => patchZone(z.zone_id, { holdout: false }, 'bez holdoutu') }); }
      else { if (info) info.textContent = `Podgląd F${f}: ${e.message}`; status(`Podgląd strefy F${f}: ${e.message}`, 'err'); }
    } finally {
      Z.pvBusy = false;
      if (Z.pvAgain != null) { const g = Z.pvAgain; Z.pvAgain = null; schedulePreview(g, 0); }
    }
  }
  function setPreview(zid, f, pv) {
    H.invalidateFrame(f);              // only this frame's overlays (Z20 per-frame mechanism), never clearOverlays()
    Z.preview = pv ? { zid, f, data: pv, seq: Date.now() } : null;
    const info = $('z4PvInfo');
    if (info) {
      if (!pv) info.textContent = `Podgląd F${f}: brak rdzenia w tej klatce – „▶▶ Przelicz nieaktualne” (etap ${H.ST.SAM3})`;
      else {
        const st = pv.stats || {};
        const zz = Z.list.find((x) => x.zone_id === zid);
        info.innerHTML = `Podgląd <b>F${f}</b>${isOutline(zz) ? ' (cały obrys − wykluczenia)' : pv.is_key === false ? ' <span class="warn">(bez klucza: obszar klucza bazowego – namaluj kreskę, by utworzyć klucz)</span>' : ' (klucz)'}`
          + ` · ${esc(ENGINE_LABEL[pv.engine_used] || pv.engine_used || st.engine || '')}${st.engine_share ? ' (' + esc(share(st.engine_share)) + ')' : ''}`
          + ` · +${pv.added_px} px, miękkie ${pv.soft_px}${st.seconds != null ? ` · ${(+st.seconds).toFixed(2)} s` : ''}`
          + (pv.holdout_objects && pv.holdout_objects.length ? ` · holdout: ${esc(pv.holdout_objects.join(', '))}` : '')
          + (pv.cleanplate ? ` · clean plate: ${esc(pv.cleanplate)}` : '')
          + (pv.flags && pv.flags.length ? `<br><span class="warn">⚑ ${esc(pv.flags.join('; '))}</span>` : '')
          + (pv.note ? `<br><span class="muted">ℹ ${esc(pv.note)}</span>` : '');
      }
    }
    if (pv && Z.ov === 'none' && !Z.ovUserOff) setOv('alpha');   // not when the user switched the overlay off (2026-10-16)
    V().render();
  }
  function refreshPreview(silent = false) {   // silent: called by itself (ViTMatte base changed) – no warning without a zone
    const z = activeMatte();
    if (!z) { if (silent !== true) status('Wybierz strefę „matte” (karta) albo narysuj nową', 'warn'); return; }
    schedulePreview(curFrame(), 0);
  }

  // ------------------------------------------------------------------------------------------------ overlays on the canvas (inside pan / zoom)
  function strokePath(ctx, st, k) {
    if (st.p.some((q) => q.length >= 3) && H.drawVarStroke) { H.drawVarStroke(ctx, st.p, k, st.radius_px); return; }
    ctx.lineWidth = 2 * (st.radius_px || 8) * k; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); st.p.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k)));
    if (st.p.length === 1) ctx.lineTo(st.p[0][0] * k + 0.01, st.p[0][1] * k);
    ctx.stroke();
  }
  function cachedCanvas(key, make) {
    const cc = S().compCache;
    if (cc[key]) return cc[key];
    const c = make();
    H.putComp(key, c);   // the newest kept (objects.js putComp)
    return c;
  }
  function trimapCanvas(ov, key) {
    return cachedCanvas(key, () => {
      const c = document.createElement('canvas'); c.width = ov.naturalWidth; c.height = ov.naturalHeight;
      const g = c.getContext('2d'); g.drawImage(ov, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height); const p = d.data;
      for (let i = 0; i < p.length; i += 4) {
        const v = p[i];
        if (v > 200) { p[i] = 80; p[i + 1] = 160; p[i + 2] = 255; p[i + 3] = 110; }
        else if (v > 90) { p[i] = 250; p[i + 1] = 204; p[i + 2] = 21; p[i + 3] = 150; }
        else p[i + 3] = 0;
      }
      g.putImageData(d, 0, 0); return c;
    });
  }
  function enginesCanvas(ov, key) {
    return cachedCanvas(key, () => {
      const c = document.createElement('canvas'); c.width = ov.naturalWidth; c.height = ov.naturalHeight;
      const g = c.getContext('2d'); g.drawImage(ov, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height); const p = d.data;
      for (let i = 0; i < p.length; i += 4) p[i + 3] = Math.max(p[i], p[i + 1], p[i + 2]);
      g.putImageData(d, 0, 0); return c;
    });
  }
  function unionCanvas(core, za, key) {
    return cachedCanvas(key, () => {
      const c = document.createElement('canvas'); c.width = za.naturalWidth; c.height = za.naturalHeight;
      const g = c.getContext('2d'); g.drawImage(core, 0, 0, c.width, c.height);
      g.globalCompositeOperation = 'lighten'; g.drawImage(za, 0, 0, c.width, c.height);
      return c;
    });
  }
  // live finish (user decision 2026-10-16): black / white / blur act AFTER the model – recomputed here from the preview's
  // raw alpha (view zone_raw) and area (zone_area: 128 = region, 255 = region ∩ core = 1), the same formula as
  // engine/zones.finish_zone_alpha (levels, Gaussian blur, only inside the region, core forced to 1); no model call
  // the SAVED feather of the active zone (not the slider while it is dragged)
  const featherPx = () => { const z = activeMatte(); const v = z && z.params && z.params.inset_feather_px != null ? +z.params.inset_feather_px : 8; return isFinite(v) ? Math.max(0.5, v) : 8; };
  function finVals() {
    const w = $('z4White') ? +$('z4White').value : 1;
    return { b: Math.max(0, +($('z4Black') || {}).value || 0), w: isFinite(w) && w > 0 ? w : 1, s: Math.max(0, +($('z4Blur') || {}).value || 0) };
  }
  function finishedCanvas(f) {
    if (!Z.preview || Z.preview.f !== f) return null;
    const raw = H.overlayFor('zone_raw', f), area = H.overlayFor('zone_area', f);
    if (!raw || raw === 'missing' || !area || area === 'missing') return null;
    const fv = finVals();
    const pd = V().projectData; const kpx = pd && pd.plate_w ? raw.naturalWidth / pd.plate_w : 1;
    const fe = featherPx();
    const key = `zfin_${obj()}_${f}_${Z.preview.seq || 0}_${fv.b}_${fv.w}_${fv.s}_${fe}`;
    const c = cachedCanvas(key, () => {
      const W = raw.naturalWidth, Hh = raw.naturalHeight;
      const cv = document.createElement('canvas'); cv.width = W; cv.height = Hh; const g = cv.getContext('2d');
      g.drawImage(raw, 0, 0); const rp = g.getImageData(0, 0, W, Hh).data;
      const lv = g.createImageData(W, Hh); const lp = lv.data;
      const lev = fv.b > 0 || fv.w < 1; const ww = Math.max(fv.w, fv.b + 1e-3);
      for (let i = 0; i < rp.length; i += 4) {
        let a = rp[i] / 255;
        if (lev) a = Math.min(1, Math.max(0, (a - fv.b) / (ww - fv.b)));
        lp[i] = lp[i + 1] = lp[i + 2] = Math.round(a * 255); lp[i + 3] = 255;
      }
      let bp = lp;
      if (fv.s > 0) {
        const t = document.createElement('canvas'); t.width = W; t.height = Hh; t.getContext('2d').putImageData(lv, 0, 0);
        const t2 = document.createElement('canvas'); t2.width = W; t2.height = Hh; const g2 = t2.getContext('2d');
        g2.fillStyle = '#000'; g2.fillRect(0, 0, W, Hh); g2.filter = `blur(${fv.s * kpx}px)`; g2.drawImage(t, 0, 0);
        bp = g2.getImageData(0, 0, W, Hh).data;
      }
      g.clearRect(0, 0, W, Hh); g.drawImage(area, 0, 0, W, Hh); const ap = g.getImageData(0, 0, W, Hh).data;
      const out = g.createImageData(W, Hh); const op = out.data;
      for (let i = 0; i < op.length; i += 4) {
        const v = ap[i] > 192 ? 255 : ap[i] > 64 ? bp[i] : rp[i];
        op[i] = op[i + 1] = op[i + 2] = v; op[i + 3] = 255;
      }
      g.putImageData(out, 0, 0);
      return cv;
    });
    // feather weight inside the area (like apply_soft_override of the result): 0 at its border -> 1 after `fe` px
    const weight = cachedCanvas(`${key}_w`, () => {
      const W = area.naturalWidth, Hh = area.naturalHeight;
      const cv = document.createElement('canvas'); cv.width = W; cv.height = Hh; const g = cv.getContext('2d');
      g.drawImage(area, 0, 0); const ap = g.getImageData(0, 0, W, Hh).data;
      const inside = new Uint8Array(W * Hh); for (let i = 0; i < inside.length; i++) inside[i] = ap[i * 4] > 64 ? 1 : 0;
      const d = chamfer(inside, 1, W, Hh); const lim = 3 * fe * kpx;
      const out = g.createImageData(W, Hh); const op = out.data;
      for (let i = 0; i < inside.length; i++) { const w = inside[i] ? Math.min(1, d[i] / lim) : 0; op[i * 4] = op[i * 4 + 1] = op[i * 4 + 2] = Math.round(w * 255); op[i * 4 + 3] = 255; }
      g.putImageData(out, 0, 0); return cv;
    });
    return { canvas: c, key, area, weight };
  }
  function previewMatte(f) {   // for the cut-out „Na tle” (objects.js matteFor): {canvas, key, area} of the active zone's preview
    const z = activeMatte();
    if (!z || !Z.preview || Z.preview.zid !== z.zone_id) return null;
    return finishedCanvas(f);
  }
  function previewImage(f) {   // the zone alpha overlay of the preview (Image) or null
    if (!Z.preview || Z.preview.f !== f) return null;
    const ov = H.overlayFor('zone_alpha', f);
    return ov && ov !== 'missing' ? ov : null;
  }
  // ------------------------------------------------------------------------------------------------ band control points
  // (2026-10-21, engine/bandmap.py): an outline zone's band width set by points on the outline – yellow whisker = band
  // into the background (out), blue = into the object (in); a drag = a key in this frame; 0 / 0 = a gap in the band
  const bandPts = (z) => ((z && z.params && z.params.band_points) || []);
  // The points, their whiskers, dragging, keys, Ctrl+Z, the timeline triangles and the badge of the shared band are
  // static/bandpoints.js – the same editor as stages 3 / 4 / 7 (user decision 2026-10-24); here only the zone's UI around it
  function bandUi() {
    const z = activeMatte(); if (!z) return;
    const n = bandPts(z).length;
    const B = window.MD_BANDPTS && window.MD_BANDPTS.ready ? window.MD_BANDPTS : null;
    if (B) { B.info(); B.renderMode(); }
    const kf = !!(B && B.hasKeyAt(curFrame()));
    $('z4BandClear').style.display = n ? '' : 'none';
    $('z4BandClear').disabled = !kf;
    $('z4BandClear').title = kf ? `Usuń wszystkie punkty tej klatki kluczowej (F${curFrame()}) = usuń klatkę kluczową pasa; punkt bez innych kluczy znika (Ctrl+Z cofa)`
      : `F${curFrame()} nie jest klatką kluczową pasa – przejdź na trójkąt na osi`;
  }
  // „Pas na zewnątrz” / „Wcięcie w rdzeń” size the whole band, the points shape it (user decision 2026-10-21): while
  // dragging, the whiskers scale by the slider ratio (the server's band follows on release)
  function liveBandScale() {
    const z = activeMatte(); if (!z || !window.MD_BANDPTS || !window.MD_BANDPTS.ready) return;
    const inPx = (fg, fc) => (z.params && z.params.outline_in_px != null ? +z.params.outline_in_px : Math.max(fg, fc));   // the slider = the band inside (no hidden margin, 2026-10-24)
    const o0 = +pval(z, 'outline_out_px') || 1;
    const i0 = inPx(+pval(z, 'fg_erode_px'), +pval(z, 'force_core_erode_px'));
    const n = +$('z4Inset').value;
    window.MD_BANDPTS.setLive(o0, i0, +$('z4Out').value, inPx(n, n));
  }

  // band point tools on the left rail (user decision 2026-10-22): objects.js S.tool band_add / band_remove
  const bandTool = () => (H && inZoneFlow() && ['band_add', 'band_remove'].includes(S().tool) ? S().tool : null);
  function setBandTool(on) { if (H.setTool && !!on !== (S().tool === 'band_add')) H.setTool('band_add'); }
  function onTool() {   // objects.js updateDrawTools: the rail tool changed
    const t = bandTool();
    if (t && Z.brush) setBrush(Z.brush);              // the palette brush off: a click is a point now
    $('z4BandTool').classList.toggle('active', t === 'band_add');
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.onTool(); else V().canvas.style.cursor = t ? 'crosshair' : '';
    bandUi(); V().render();
  }
  // the key of the active whole-outline zone's band for bandpoints.js („zone:<id>”), null outside the zone flow
  const activeBandKey = () => { const z = inZoneFlow() ? activeMatte() : null; return z && isOutline(z) ? `zone:${z.zone_id}` : null; };
  const pushBandUndo = () => { Z.undo.push({ bandpts: true }); };
  async function reloadBand() {   // after a band edit (bandpoints.js): the zone list again (its params carry the points), the region anew
    const z = activeMatte();
    if (z) delete Z.regionCache[z.zone_id];
    Z.band.modeSig = null;
    await load(); onFrame(); renderLane(); bandUi(); V().render();
  }
  function draw(ctx, img, k, f) {
    if (!H || !obj() || S().auxMode || S().stage !== H.ST.VITMATTE) return;
    drawProps(ctx, k, f);   // step 9 proposals (also without an active zone)
    const z = activeMatte();
    if (!z) return;
    const v = V();
    const col = zoneColour(z.zone_id);
    const keys = keysOf(z);
    const key = keys[f];
    ctx.save();
    try {
      const poly = (cnt) => { ctx.beginPath(); cnt.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); };
      if (isOutline(z)) {
        // outline mode: dashed = the band ViTMatte works in (from the server, also on key frames); red = the exclusion
        // (the key's drawing in its frame, the propagated drawing elsewhere)
        const reg = (Z.regionCache[z.zone_id] || {})[f];
        ctx.lineWidth = 1.5 / v.zoom;
        if (reg && reg.contours && bandPvGet()) {
          if (reg.band_contours && reg.band_contours.length && window.MD_BANDPTS && window.MD_BANDPTS.strokeLines) {
            // the band from the server – outer + inner edge, its width follows the band points – as dashed lines only
            // (user decision 2026-10-21: a fill hid the hair being judged); thinner dashes = where the inner and the
            // outer feather end (full zone weight between them, 2026-10-22). One drawing for every band: stages
            // 3 / 4 / 7 draw their lines with the same function (static/bandpoints.js strokeLines, 2026-10-24)
            window.MD_BANDPTS.strokeLines(ctx, k, reg);
          } else {
            ctx.strokeStyle = col; ctx.setLineDash([6 / v.zoom, 4 / v.zoom]);
            for (const cnt of reg.contours) { poly(cnt); ctx.stroke(); }
          }
          ctx.setLineDash([]);
        }
        ctx.strokeStyle = 'rgba(255,70,70,0.95)'; ctx.fillStyle = 'rgba(255,70,70,0.18)';
        if (key) {
          if (key.polygon && key.polygon.length >= 3) { poly(key.polygon); ctx.fill(); ctx.stroke(); }
          for (const st of key.strokes || []) { ctx.strokeStyle = ctx.fillStyle = 'rgba(255,70,70,0.30)'; strokePath(ctx, st, k); }
        } else if (reg && reg.exclude_contours) {
          for (const cnt of reg.exclude_contours) { poly(cnt); ctx.fill(); ctx.stroke(); }
        }
        for (const st of (key && key.bg_strokes) || []) { ctx.strokeStyle = ctx.fillStyle = 'rgba(255,60,90,0.32)'; strokePath(ctx, st, k); }
      } else if (key) {
        if (key.polygon && key.polygon.length >= 3) {
          ctx.strokeStyle = col; ctx.lineWidth = 1.5 / v.zoom;
          ctx.beginPath(); key.polygon.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); ctx.stroke();
        }
        for (const st of (key.strokes || []).filter((x) => !x.erase)) { ctx.strokeStyle = ctx.fillStyle = 'rgba(80,255,120,0.28)'; strokePath(ctx, st, k); }
        for (const st of (key.strokes || []).filter((x) => x.erase)) { ctx.strokeStyle = ctx.fillStyle = 'rgba(20,20,30,0.55)'; strokePath(ctx, st, k); }
        for (const st of key.bg_strokes || []) { ctx.strokeStyle = ctx.fillStyle = 'rgba(255,60,90,0.32)'; strokePath(ctx, st, k); }
      } else {
        const reg = (Z.regionCache[z.zone_id] || {})[f];
        if (reg && reg.contours && bandPvGet()) {   // the propagated area of this frame: filled lightly + dashed edge (painted band = where ViTMatte works)
          ctx.strokeStyle = col; ctx.fillStyle = 'rgba(80,255,120,0.16)'; ctx.lineWidth = 1.5 / v.zoom; ctx.setLineDash([6 / v.zoom, 4 / v.zoom]);
          ctx.beginPath(); for (const cnt of reg.contours) { cnt.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); }
          ctx.fill('evenodd'); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      if (Z.inset && (bandPvGet() || Z.inset.live) && !(isOutline(z) && bandPts(z).length)) {   // "Pas na zewnątrz" / "Wcięcie w rdzeń" band (dashed band edges, no fill – one band system 2026-10-22), any frame; while dragging also with the band preview off (2026-10-17)
        // a group: the core the zone is computed on (members joined with the seam, as stages 3 / 4), not the object's
        // own core – that one has a hole where the hand is, the band went round it (user report 2026-10-21)
        const kind = groupMembers().length ? 'zone_core' : 'core';
        const core = H.overlayFor(kind, f);
        const dist = core && core !== 'missing' ? coreDistance(core, `${obj()}_${kind}_${f}`) : null;
        if (dist) {
          ctx.drawImage(insetCanvas(dist, Z.inset.n, Z.inset.out || 0), 0, 0, img.width, img.height);
          if (Z.inset.live) return;   // while dragging: the band alone; after release the zone preview is drawn too
        }
      }
      const maskOff = !!(window.MD_OBJ && window.MD_OBJ.maskHidden && window.MD_OBJ.maskHidden());   // 👁 of the object off
      // Alpha / Comp: the zone preview is in the matte itself (objects.js matteFor → previewMatte), no green tint over it
      // (user report 2026-10-26); the other overlays (trimap, engines, comp) stay
      const tintOk = S().view !== 'alpha' && S().view !== 'comp';
      if (Z.preview && Z.preview.f === f && Z.preview.zid === z.zone_id && Z.ov !== 'none' && !maskOff && (tintOk || Z.ov !== 'alpha')) {
        const o = obj();
        const fc = finishedCanvas(f);
        // „👁 podgląd pasa” off: the trimap (blue sure / yellow band) is the band too – the zone alpha instead (2026-10-16)
        const ov = Z.ov === 'trimap' && !bandPvGet() ? 'alpha' : Z.ov;
        if (ov === 'alpha' && fc) {   // live finish (black / white / blur sliders) of the preview
          ctx.globalAlpha = 0.75; ctx.drawImage(H.tintCanvas(fc.canvas, [60, 255, 90], fc.key), 0, 0, img.width, img.height);
        } else if (ov === 'alpha') {
          const T = H.tileOverlays ? H.tileOverlays('zone_alpha', f) : null;   // Z8: strands at plate resolution when the plate tiles are active
          if (T) {
            ctx.globalAlpha = 0.75;
            for (const { c, img: t } of T.tiles) ctx.drawImage(H.tintCanvas(t, [60, 255, 90], `${o}_zone_alpha_tile_${c.x}_${c.y}_${c.w}_${c.h}_${f}`), c.x / T.reg.kp, c.y / T.reg.kp, c.w / T.reg.kp, c.h / T.reg.kp);
          } else {
            const za = H.overlayFor('zone_alpha', f);
            if (za && za !== 'missing') { ctx.globalAlpha = 0.75; ctx.drawImage(H.tintCanvas(za, [60, 255, 90], `${o}_zone_alpha_${f}`), 0, 0, img.width, img.height); }
          }
        } else if (ov === 'comp') {
          const za = H.overlayFor('zone_alpha', f); const core = H.overlayFor(S().keys.some((x) => x.frame === f) ? 'key' : 'core', f);
          if (za && za !== 'missing' && core && core !== 'missing') ctx.drawImage(H.compCanvas(img, unionCanvas(core, za, `zunion_${o}_${f}_`), `${o}_zcomp_${f}`, f), 0, 0);
          else if (za && za !== 'missing') ctx.drawImage(H.compCanvas(img, za, `${o}_zcomp_${f}`, f), 0, 0);
        } else if (ov === 'trimap') {
          const tm = H.overlayFor('zone_trimap', f);
          if (tm && tm !== 'missing') ctx.drawImage(trimapCanvas(tm, `ztri_${o}_${f}_`), 0, 0, img.width, img.height);
        } else if (ov === 'engines') {
          const en = H.overlayFor('zone_engines', f);
          if (en && en !== 'missing') { ctx.globalAlpha = 0.7; ctx.drawImage(enginesCanvas(en, `zeng_${o}_${f}_`), 0, 0, img.width, img.height); }
        }
      } else if (Z.ov === 'alpha' && !maskOff && H.stageRunAlpha && !H.stagePreviewAt(f) && S().view !== 'alpha' && S().view !== 'comp') {
        // no preview of this frame (user report 2026-10-25: „zone alpha” on, only the core visible): the stage 5 result
        // (the active version ★) in green instead – „↻ Podgląd” / „👁 Podgląd klatek” computes the current settings
        const ra = H.stageRunAlpha(f);
        if (ra) { ctx.globalAlpha = 0.6; ctx.drawImage(H.tintCanvas(ra.img, [60, 255, 90], `${obj()}_zrun_${ra.run}_${f}`), 0, 0, img.width, img.height); }
      }
    } finally { ctx.restore(); }
  }
  // loupe (screen space, after the pan / zoom transform): plate + zone alpha of the preview, 3x
  function drawLoupe(ctx, img, k) {
    if (!H || !Z.loupe || !Z.mouse || S().stage !== H.ST.VITMATTE || !obj() || S().auxMode) return;
    const v = V(); const r = v.canvas.getBoundingClientRect();
    const mx = Z.mouse[0] - r.left, my = Z.mouse[1] - r.top;
    if (mx < 0 || my < 0 || mx > r.width || my > r.height) return;
    const R = 90, M = 3;
    const ix = (mx - v.panX) / v.zoom, iy = (my - v.panY) / v.zoom;        // image px under the cursor
    const sw = (2 * R) / (v.zoom * M);
    let cx = mx + R + 24, cy = my - R - 24;
    if (cx + R > r.width) cx = mx - R - 24;
    if (cy - R < 0) cy = my + R + 24;
    ctx.save();
    try {
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.clip();
      ctx.fillStyle = '#000'; ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, ix - sw / 2, iy - sw / 2, sw, sw, cx - R, cy - R, 2 * R, 2 * R);
      const za = previewImage(v.currentFrame);
      if (za && Z.ov !== 'none' && !(window.MD_OBJ && window.MD_OBJ.maskHidden && window.MD_OBJ.maskHidden())) {
        const t = H.tintCanvas(za, [60, 255, 90], `${obj()}_zone_alpha_${v.currentFrame}`);
        const s = t.width / img.width;
        ctx.globalAlpha = 0.7; ctx.drawImage(t, (ix - sw / 2) * s, (iy - sw / 2) * s, sw * s, sw * s, cx - R, cy - R, 2 * R, 2 * R); ctx.globalAlpha = 1;
      }
    } finally { ctx.restore(); }
    ctx.save();
    try {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = '11px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.fillText(`lupa ${M}×`, cx, cy + R - 8);
    } finally { ctx.restore(); }
  }
  function setLoupe(on) {
    Z.loupe = on; const b = $('z4Loupe'); if (b) b.classList.toggle('active', on);
    if (!on) Z.mouse = null;
    V().render();
  }

  // ------------------------------------------------------------------------------------------------ timeline lane (diamonds = zone keys, bar = range)
  function laneEl() {
    let lane = $('timelineZones'); if (lane) return lane;
    const keys = $('timelineKeys'); if (!keys) return null;
    lane = H.el('<div class="timeline-zones" id="timelineZones" title="Klucze aktywnej strefy (romby w kolorze strefy): klik = skok, PPM = usuń klucz; pasek = zakres między kluczami"></div>');
    keys.parentNode.insertBefore(lane, keys);
    return lane;
  }
  function renderLane() {
    const lane = laneEl(); const v = V(); if (!lane || !v) return;
    const z = H && obj() && !S().auxMode && S().stage === H.ST.VITMATTE ? activeMatte() : null;
    lane.style.display = z ? 'block' : 'none';
    lane.innerHTML = '';
    if (!z) return;
    const col = zoneColour(z.zone_id);
    const range = Math.max(1, v.lastFrame - v.firstFrame);
    const ks = laneKeys(z);
    const bar = document.createElement('div'); bar.className = 'zone-range';
    bar.style.cssText = `left:${100 * (ks[0] - v.firstFrame) / range}%;width:${Math.max(0.3, 100 * (ks[ks.length - 1] - ks[0]) / range)}%;background:${col}`;
    bar.title = `Strefa ${z.zone_id}: klucze F${ks[0]}–F${ks[ks.length - 1]}; poza nimi obszar jedzie z obiektem (propagacja)`;
    if (ks.length) lane.appendChild(bar);
    for (const f of ks) {
      const m = document.createElement('div');
      m.className = `zone-key ${f === v.currentFrame ? 'cur' : ''} ${f === +z.frame ? 'base' : ''}`;
      m.style.left = `${100 * (f - v.firstFrame) / range}%`; m.style.background = col;
      const kn = keysOf(z)[f].prompt;
      m.title = `Klucz strefy ${z.zone_id} F${f}${f === +z.frame ? ' (bazowy)' : ''}${kn ? ` · uwaga: ${kn}` : ''} · klik = skok · PPM = usuń klucz`;
      m.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
      m.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); deleteKey(z.zone_id, f); });
      lane.appendChild(m);
    }
    if (Z.preview && Z.preview.zid === z.zone_id) {
      const p = document.createElement('div'); p.className = 'zone-pv'; p.style.left = `${100 * (Z.preview.f - v.firstFrame) / range}%`; p.title = `podgląd strefy policzony w F${Z.preview.f}`;
      lane.appendChild(p);
    }
  }

  // ------------------------------------------------------------------------------------------------ „Zastosuj na ujęcie” + objects in front
  // palette brush (rotobrush-like, 2026-10-15): paint = "+" strokes of the zone key, erase = strokes with erase: true
  // (removed from the drawing), bg = the "-" strokes (forced background); the left toolbar brush is switched along
  function setBrush(kind) {
    Z.brush = Z.brush === kind ? null : kind;
    const want = Z.brush === 'paint' ? 'brush_add' : Z.brush ? 'brush_remove' : null;
    if (H.setTool && want && S().tool !== want) H.setTool(want);
    if (H.setTool && !want && (S().tool === 'brush_add' || S().tool === 'brush_remove')) H.setTool(S().tool);   // off
    $('z4Brush').querySelectorAll('button').forEach((b) => b.classList.toggle('active', Z.brush === b.dataset.b));
  }
  async function presetParams(preset) {
    try { return (await api('GET', `/api/zones/preset_defaults/${preset}`)).params || {}; } catch (e) { return {}; }
  }

  // area mode of the active zone (palette): switching a drawn hair area to "outline" turns that drawing into an exclusion,
  // so the user is asked whether to clear it (keys keep their "-" strokes)
  async function setRegionMode(r) {
    const z = activeMatte(); if (!z || (r === 'outline') === isOutline(z)) return;
    const keys = keysOf(z); const ks = keyFrames(z);
    const drawn = ks.some((f) => (keys[f].polygon || []).length >= 3 || (keys[f].strokes || []).length);
    let clear = false;
    if (r === 'outline' && drawn) {
      clear = await V().uiConfirm('W trybie „cały obrys” rysunek strefy oznacza miejsca BEZ włosów (wykluczenie z pasa ViTMatte).\n'
        + `Wyczyścić dotychczasowy obszar włosów w kluczach (${ks.length})? Kreski tła (R) zostają.\n„Anuluj” = rysunek zostaje i działa jako wykluczenie.`, { ok: 'Wyczyść rysunek' });
    }
    await patchTune({ region: r }, r === 'outline' ? 'obszar: cały obrys − wykluczenia' : 'obszar: narysowany');
    if (clear) {
      for (const f of ks) {
        const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, { polygon: [], strokes: [], bg_strokes: keys[f].bg_strokes || [] });
        replaceZone(d.zone);
      }
      renderList(); schedulePreview(curFrame(), 0);
      status(`Strefa ${z.zone_id}: cały obrys, rysunek wyczyszczony – zaznacz G / P miejsca bez włosów`, 'ok');
    }
    if (r === 'draw' && !drawn) status(`Strefa ${z.zone_id}: narysuj obszar włosów (G / P) – bez rysunku ViTMatte nic nie liczy`, 'warn');
    delete Z.regionCache[z.zone_id]; onFrame();
  }
  // "＋ Strefa ViTMatte: cały obrys" (palette): a hair zone with the preset's ViTMatte settings, region = outline, no drawing
  async function newOutlineZone() {
    const f = curFrame();
    let def = {};
    try { def = (await api('GET', '/api/zones/preset_defaults/hair')).params || {}; } catch (e) { /* older server: engine defaults */ }
    const body = { mode: 'add', criterion: 'matte', frame: f, polygon: [], strokes: [], bg_strokes: [], preset: 'hair', matting: 'vitmatte',
                   prompt: Z.prompt, holdout: Z.holdout, static: false, bg_static: true, params: { ...def, region: 'outline', vitmatte_model: undefined } };   // model: the stage engine (server)
    let d;
    // creating the zone computes its first preview (ViTMatte on this frame): say so – it looked frozen (2026-10-17)
    status(`Tworzę strefę ViTMatte wzdłuż całego obrysu i liczę podgląd F${f} (ViTMatte, kilka sekund)…`);
    toast(`Tworzę strefę ViTMatte wzdłuż całego obrysu i liczę podgląd F${f} (ViTMatte, kilka sekund)…`, 6000);
    const info = $('z4PvInfo'); if (info) info.textContent = `⏳ liczę podgląd F${f} (ViTMatte)…`;
    try { d = await api('POST', `/api/object/${obj()}/zones/new?preview_frame=${f}`, body); }
    catch (e) {
      const mf = missingFront(e);
      if (mf) { frontDialog(mf, { noHoldout: async () => { Z.holdout = false; $('z4Holdout').checked = false; await newOutlineZone(); } }); return; }
      throw e;
    }
    Z.active = d.zone.zone_id; Z.undo.push({ zid: Z.active, f, prev: null, created: true });
    await load();
    toast(DRAW_TOOLS ? `Nowa strefa ${Z.active}: ViTMatte wzdłuż całego obrysu – zaznacz G / P miejsca bez włosów (twarz, szyja); dłoń z przodu = obiekt przed tym (holdout)`
      : `Nowa strefa ${Z.active}: ViTMatte wzdłuż całego obrysu – grubość pasa punktami pasa (G); dłoń z przodu = obiekt przed tym (holdout)`);
    setPreview(d.zone.zone_id, f, d.preview);
    H.loadStages().catch(() => {}); onFrame();
  }

  async function applyShot() {
    const z = activeMatte() || Z.list.find((x) => x.enabled && x.criterion === 'matte');
    if (!z) { status(DRAW_TOOLS ? 'Brak strefy „matte” do zastosowania – narysuj strefę (G / P)' : 'Brak strefy „matte” do zastosowania – „＋ Nowa strefa”', 'warn'); return; }
    await H.ensurePointsSaved();
    if (Z.drawPending != null) await commitDraw(Z.drawPending);
    const base = (S().info && S().info.vitmatte_base) === 'core' ? `masce 1+2 (etapy ${H.ST.SAM3} + ${H.ST.BRUSH})` : `wersji MatAnyone2 (etap ${H.ST.MATANYONE})`;
    status(`Zastosuj ${z.zone_id}: przeliczam strefy ViTMatte na ujęciu na ${base} (poza strefami baza bez zmian)…`);
    let d;
    try { d = await api('POST', `/api/object/${obj()}/zones/${z.zone_id}/apply?stage=vitmatte`); }
    catch (e) {
      const mf = missingFront(e);
      if (mf) { frontDialog(mf, { noHoldout: async () => { await patchZone(z.zone_id, { holdout: false }, 'bez holdoutu'); await applyShot(); } }); return; }
      throw e;
    }
    H.clearOverlays();                 // a new version: every frame's alpha changed
    await H.loadStages(); H.restoreUserView();
    await load(); H.loadQA();
    status(`Wersja ${d.run}: strefa ${z.zone_id} – przeliczone ${d.computed} kl., z cache ${d.cached}${d.dependents_to_recompute && d.dependents_to_recompute.length ? ` · do przeliczenia: ${d.dependents_to_recompute.join(', ')}` : ''}. Etap ${H.ST.VITMATTE} ViTMatte zaliczony; „Porównanie wersji” niżej (A / B / różnica, klawisze A / B / D).`, 'ok');
  }
  function frontDialog(mf, opts) {
    let m = $('zoneFrontModal');
    if (!m) { m = H.el('<div id="zoneFrontModal" class="modal-backdrop zone-front" style="display:none"></div>'); document.body.appendChild(m); }
    const list = (mf.missing_front || []).join(', ');
    m.innerHTML = `<div class="modal-card zone-front-card">
      <div class="modal-title">Obiekty z przodu (${esc(list)}) nie mają jeszcze maski</div>
      <div class="muted">${esc(mf.detail || '')}</div>
      <div class="muted">Holdout strefy bierze maski obiektów z przodu (finalna alpha albo rdzeń) – bez nich kosmyki mogłyby wejść na rękę / psa. Kolejka „Policz wszystko” liczy obiekty bez wyniku od przodu (rdzeń + finalnie), a potem ten obiekt ze strefą.</div>
      <div class="row" style="justify-content:flex-end;gap:8px;margin-top:10px">
        <button class="btn btn-ghost" data-act="cancel">Anuluj</button>
        <button class="btn btn-secondary" data-act="noholdout" title="Strefa liczy się bez holdoutu (żółta flaga QA „strefa bez holdoutu”)">Licz mimo to bez holdoutu</button>
        <button class="btn btn-primary" data-act="queue" title="POST /api/queue/preset/all + uruchomienie kolejki">Policz obiekty z przodu i kontynuuj</button></div></div>`;
    m.style.display = 'flex';
    const close = () => { m.style.display = 'none'; };
    m.querySelector('[data-act="cancel"]').addEventListener('click', close);
    m.querySelector('[data-act="noholdout"]').addEventListener('click', async () => { close(); try { await opts.noHoldout(); } catch (e) { status(`Błąd: ${e.message}`, 'err'); } });
    m.querySelector('[data-act="queue"]').addEventListener('click', async () => {
      close();
      try {
        const q = await api('POST', '/api/queue/preset/all');
        await api('POST', '/api/queue/run');
        status(`Kolejka „Policz wszystko”: ${(q.objects || []).join(' → ')} (od przodu; ten obiekt ze strefą na końcu). Postęp na pasku zadań; po kolejce obiekty będą miały wynik.`, 'ok');
        toast(`Kolejka: ${(q.objects || []).length} obiektów`);
      } catch (e) { status(`Kolejka: ${e.message}`, 'err'); }
    });
  }

  // ------------------------------------------------------------------------------------------------ versions: „Szczegóły” (engines, time per frame)
  function runDetailsHtml(r) {
    const zs = r.zones || {}; const ids = Object.keys(zs); if (!ids.length) return '';
    const rows = ids.map((zid) => {
      const s = zs[zid];
      return `<div>${esc(zid)} · ${esc(ENGINE_LABEL[s.engine] || s.engine || '')}${s.engine_share && Object.keys(s.engine_share).length ? ` (${esc(share(s.engine_share))})` : ''} · ${s.frames ?? '?'} kl.`
        + `${s.seconds != null ? ` · ${s.seconds} s${s.s_per_frame != null ? ` (${s.s_per_frame} s/kl.)` : ''}` : ''}`
        + `${s.computed != null ? ` · przeliczone ${s.computed} / z cache ${s.cached ?? 0}` : ''}${s.cleanplate ? ' · clean plate' : ''}</div>`;
    }).join('');
    return `<details class="run-zones"><summary title="Strefy tej wersji: silniki, udziały, czas per klatka (run.json → zones)">Szczegóły stref (${ids.length})${r.applied_from ? ` · „Zastosuj” z ${esc(r.applied_from)}` : ''}</summary>${rows}</details>`;
  }

  // ------------------------------------------------------------------------------------------------ wiring
  function wire() {
    $('z4New').addEventListener('click', () => {
      if (!DRAW_TOOLS) { H.run('z4New', newOutlineZone); return; }   // the whole-outline zone itself (one per object)
      setActive(null); status(`Nowa strefa: wybierz preset, opcjonalnie opis i silnik, potem maluj G (obszar) / R (tło) albo wielokąt P – pierwsza kreska tworzy strefę i klucz tej klatki`, 'ok');
    });
    if (!DRAW_TOOLS) $('z4Propose').style.display = 'none';   // proposals become painted zones (hidden since 2026-10-21)
    document.querySelectorAll('#z4Presets .zp').forEach((b) => b.addEventListener('click', () => {
      const z = activeMatte();
      if (z) patchZone(z.zone_id, { preset: b.dataset.preset }, `preset ${PRESET_NAME[b.dataset.preset]}`);
      else { Z.preset = b.dataset.preset; syncForm(); status(`Preset nowej strefy: ${PRESET_NAME[Z.preset]}`); }
    }));
    $('z4Prompt').addEventListener('change', async () => {
      const z = activeMatte(); const t = $('z4Prompt').value.trim();
      if (z) {
        const d = await patchZone(z.zone_id, { prompt: t }, 'opis zapisany – reguły promptu (preset / silniki / linie w tle) niżej').catch(() => null);
        if (d && d.plan) { Z.plan = { plan: d.plan, bg_strokes: null, f: curFrame() }; renderPlan(d.plan, null); if (d.plan.bg_line) findLines().catch(() => {}); }
        else if (d) renderPlan(null);
      } else {
        Z.prompt = t;
        if (!t) { renderPlan(null); return; }
        try { const d = await api('POST', `/api/object/${obj()}/zones/interpret`, { prompt: t }); Z.plan = { ...d, f: null }; renderPlan(d.plan, null); if (d.plan.preset && Z.preset === 'hair') { Z.preset = d.plan.preset; syncForm(); } }   // like the server: the description sets the preset unless one was chosen
        catch (e) { renderPlan(null); }
      }
    });
    $('z4Propose').addEventListener('click', () => H.run('z4Propose', propose));
    $('z4Engine').addEventListener('change', () => {
      const z = activeMatte(); const e = $('z4Engine').value;
      if (e === 'clean' && Z.cp && Z.cp.source === 'brak') toast('Silnik clean wymaga clean plate’u – „Policz z ujęcia (auto)” albo „Mam plik…”', 6000);
      if (z) patchZone(z.zone_id, { matting: e }, `silnik ${ENGINE_LABEL[e] || e}`); else Z.engine = e;
    });
    $('z4Grp').addEventListener('change', (e) => {
      const n = e.target && e.target.dataset && e.target.dataset.sf; if (!n) return;
      const z = activeMatte(); if (!z) return;
      const sel = [...$('z4Grp').querySelectorAll('input[data-sf]')].filter((x) => x.checked).map((x) => x.dataset.sf);
      patchTune({ sure_fg: sel }, `pewny pierwszy plan: ${sel.length ? sel.join(', ') : 'nikt'}`);
    });
    $('zpBandPv').addEventListener('change', () => { bandPvSet($('zpBandPv').checked); V().render(); });
    $('zpAlphaPv').addEventListener('change', () => (Z.ovUserOff = !$('zpAlphaPv').checked, setOv($('zpAlphaPv').checked ? (Z.ovLast && Z.ovLast !== 'none' ? Z.ovLast : 'alpha') : 'none')));
    $('zpEngine').addEventListener('change', () => {   // palette copy of the panel's engine select
      const z = activeMatte(); const e = $('zpEngine').value;
      if (z) patchZone(z.zone_id, { matting: e }, `silnik ${ENGINE_LABEL[e] || e}`);
    });
    $('z4VmModel').addEventListener('change', () => patchTune({ vitmatte_model: $('z4VmModel').value }, `model ${EDGE_MODEL_LABEL[$('z4VmModel').value] || $('z4VmModel').value}`));
    $('z4VmPx').addEventListener('change', () => patchTune({ model_px: +$('z4VmPx').value }, `rozdzielczość modelu ${$('z4VmPx').value}`));
    $('z4VmUnk').addEventListener('input', () => { $('z4VmUnkV').textContent = (+$('z4VmUnk').value).toFixed(2); });
    $('z4VmUnk').addEventListener('change', () => patchTune({ unknown_value: +$('z4VmUnk').value }, `fine tune ${(+$('z4VmUnk').value).toFixed(2)}`));
    $('z4Holdout').addEventListener('change', () => {
      const z = activeMatte(); const h = $('z4Holdout').checked;
      if (z) patchZone(z.zone_id, { holdout: h }, h ? 'z holdoutem' : 'bez holdoutu (żółta flaga QA)'); else Z.holdout = h;
    });
    $('z4Temporal').addEventListener('change', () => { $('z4TempHint').textContent = TEMPORAL_HELP[$('z4Temporal').value] || ''; });
    $('z4Temporal').addEventListener('change', () => patchTune({ temporal: $('z4Temporal').value }, `wygładzanie: ${$('z4Temporal').selectedOptions[0].textContent}`));
    [['z4TRadius', 'temporal_radius', (v) => `${v}`, 'promień'], ['z4Black', 'alpha_black', (v) => v.toFixed(2), 'czerń'],
     ['z4BlackAfter', 'alpha_black_after', (v) => v.toFixed(2), 'czerń po wygładzaniu'],
     ['z4White', 'alpha_white', (v) => v.toFixed(2), 'biel'], ['z4Blur', 'alpha_blur_px', (v) => `${v.toFixed(1)} px`, 'rozmycie']].forEach(([id, k, fmt, l]) => {
      const fin = ['alpha_black', 'alpha_white', 'alpha_blur_px'].includes(k);   // after the model: live, no recompute
      $(id).addEventListener('input', () => { $(`${id}V`).textContent = fmt(+$(id).value); if (fin) V().render(); });   // live number (+ preview) while dragging
      $(id).addEventListener('change', () => patchTune({ [k]: +$(id).value }, `${l} ${fmt(+$(id).value)}`, { noPreview: fin }));
    });
    // live band preview on the mask while dragging "Pas na zewnątrz" / "Wcięcie w rdzeń" (blue = sure 1, yellow = ViTMatte)
    const livePreview = (live = true) => {
      const z = activeMatte();
      Z.inset = { f: curFrame(), n: +$('z4Inset').value, out: isOutline(z) ? +$('z4Out').value : 0, live }; V().render();
    };
    $('z4Inset').addEventListener('input', () => { $('z4InsetV').textContent = `${$('z4Inset').value} px`; liveBandScale(); livePreview(); });
    // feather: only the number while dragging (the live recompute of the blend weights lagged, 2026-10-17); applied to the
    // preview cut-out once on release – no model
    $('z4Feather').addEventListener('input', () => { $('z4FeatherV').textContent = `${$('z4Feather').value} px`; });
    $('z4Feather').addEventListener('change', () => patchTune({ inset_feather_px: +$('z4Feather').value }, `wtopienie wcięcia ${$('z4Feather').value} px`, { noPreview: true }).then(() => V().render()));
    // the live band of „Pas na zewnątrz” / „Wcięcie” ends with the pointer – also when no „change” comes (a click
    // without moving, release outside the slider): it stayed on with the band preview off (2026-10-17)
    window.addEventListener('pointerup', () => { if (Z.inset && Z.inset.live) { Z.inset.live = false; V().render(); } });
    $('z4Out').addEventListener('input', () => { $('z4OutV').textContent = `${$('z4Out').value} px`; liveBandScale(); livePreview(); });
    $('z4Inset').addEventListener('change', async () => {
      const n = +$('z4Inset').value;
      await patchTune({ fg_erode_px: n, force_core_erode_px: n }, `wcięcie w rdzeń ${n} px`);
      // the model does not run on a slider (2026-10-17): keep the live band with the saved values (it vanished waiting
      // for a server trimap that is no longer recomputed)
      livePreview(false);
    });
    $('z4OutFeather').addEventListener('input', () => { $('z4OutFeatherV').textContent = `${$('z4OutFeather').value} px`; });
    $('z4OutFeather').addEventListener('change', async () => {
      const z = activeMatte(); const n = +$('z4OutFeather').value;
      await patchTune({ outer_feather_px: n }, n ? `wtopienie krawędzi zewnętrznej ${n} px` : 'wtopienie krawędzi zewnętrznej 0 px – twarde cięcie', { noPreview: true });
      if (z) delete Z.regionCache[z.zone_id];   // the feather line comes with the band area
      onFrame(); V().render();
    });
    $('z4Out').addEventListener('change', async () => {
      const n = +$('z4Out').value;
      await patchTune({ outline_out_px: n }, `pas ViTMatte ${n} px poza obrysem`);
      livePreview(false);
    });
    $('z4Region').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => H.run('z4Region', () => setRegionMode(b.dataset.r))));
    $('z4Excl').addEventListener('change', (e) => {
      const c = e.target.closest('input[data-ex]'); if (!c) return;
      const z = activeMatte(); if (!z) return;
      const cur = new Set((z.params && z.params.exclude_objects) || []);
      if (c.checked) cur.add(c.dataset.ex); else cur.delete(c.dataset.ex);
      patchTune({ exclude_objects: [...cur] }, c.checked ? `wykluczony obiekt ${c.dataset.ex}` : `obiekt ${c.dataset.ex} znów w pasie ViTMatte`);
    });
    $('zpNewOutline').addEventListener('click', () => H.run('zpNewOutline', newOutlineZone));
    $('zpPickZone').addEventListener('click', () => { if ($('zpPickZone').dataset.z) setActive($('zpPickZone').dataset.z); });
    if (!DRAW_TOOLS) $('zpNewPaint').style.display = 'none';
    $('z4Brush').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => setBrush(b.dataset.b)));
    $('z4Track').addEventListener('change', () => patchTune({ track: $('z4Track').checked ? 'local' : 'rigid' }, $('z4Track').checked ? 'śledzenie lokalne (CoTracker3)' : 'śledzenie sztywne'));
    $('zpNewPaint').addEventListener('click', () => {
      Z.newPainted = true; Z.engine = 'vitmatte'; $('z4Engine').value = 'vitmatte'; setBrush('paint');
      status('Maluj pędzlem pas wzdłuż krawędzi włosy–tło (może wjechać na dłoń) – pierwsza kreska tworzy strefę ViTMatte ze śledzeniem lokalnym', 'ok');
    });
    $('z4PvFrame').addEventListener('click', refreshPreview);
    $('zpApply').addEventListener('click', () => H.run('zpApply', applyShot));
    $('zpStagePv').addEventListener('click', () => window.MD_OBJ && window.MD_OBJ.stagePreview('vitmatte', 'zpStagePv'));
    $('z4TScale').addEventListener('change', () => patchTune({ temporal_scale: $('z4TScale').checked ? 2 : 1 }, $('z4TScale').checked ? 'wygładzanie ×2' : 'wygładzanie ×1'));
    $('z4TuneReset').addEventListener('click', async () => {
      const z = activeMatte(); if (!z) return;
      let def = {};
      try { def = (await api('GET', `/api/zones/preset_defaults/${z.preset}`)).params || {}; } catch (e) { /* older server */ }
      const RESET = ['temporal', 'temporal_radius', 'temporal_scale', 'alpha_black', 'alpha_white', 'alpha_blur_px', 'alpha_black_after'];   // not the area
      const keep = Object.fromEntries(Object.entries(z.params || {}).filter(([k]) => !RESET.includes(k)));
      await patchZone(z.zone_id, { params: { ...keep, ...Object.fromEntries(Object.entries(def).filter(([k]) => RESET.includes(k))) } }, `wygładzanie i wykończenie z presetu ${PRESET_NAME[z.preset] || z.preset}`).catch(() => {});
    });
    $('z4CpAuto').addEventListener('click', () => H.run('z4CpAuto', async () => {
      status('Clean plate auto: mediana nieprzykrytych pikseli (+ LaMa / OpenCV na dziurach)… postęp na pasku zadań');
      let d;
      try { d = await api('POST', '/api/project/cleanplate_auto', {}); }
      catch (e) {
        const mf = missingFront(e);
        if (mf) { frontDialog({ ...mf, detail: mf.detail }, { noHoldout: async () => { toast('Clean plate auto wymaga rdzeni wszystkich obiektów – policz je (kolejka)'); } }); return; }
        throw e;
      }
      await loadCleanplate();
      status(`Clean plate auto gotowy: ${d.frames_sampled ?? ''} kl. próbki, pewność ${d.confidence_mean != null ? (+d.confidence_mean).toFixed(2) : '–'}, dziury: ${d.inpaint_backend || '–'} – plate „cleanplate_auto” w etapie 0`, 'ok');
    }));
    $('z4CpFile').addEventListener('click', () => { if (V().openPlatesModal) V().openPlatesModal(); status('Dodaj plate „cleanplate” (clear plate) w oknie plate’ów – zapis w project.json, manifest bez zmian', 'ok'); });
    $('z4CpNone').addEventListener('click', () => {
      Z.cpNone = !Z.cpNone;
      try { localStorage.setItem(`md_cp_none_${(V().projectData || {}).work || ''}`, Z.cpNone ? '1' : '0'); } catch (e) { /* storage blocked */ }
      loadCleanplate();
    });
    document.querySelectorAll('#z4Ov button').forEach((b) => b.addEventListener('click', () => { Z.ovUserOff = b.dataset.ov === 'none'; setOv(b.dataset.ov); }));
    $('z4Refresh').addEventListener('click', refreshPreview);
    $('z4Loupe').addEventListener('click', () => setLoupe(!Z.loupe));
    $('z4ApplyShot').addEventListener('click', () => H.run('z4ApplyShot', applyShot));
    $('z4KeyDel').addEventListener('click', () => { const z = activeMatte(); if (z) deleteKey(z.zone_id, curFrame()); });
    const adv = $('z4Adv');
    if (adv) adv.addEventListener('toggle', () => { renderLane(); syncForm(); V().render(); });

    const cv = V().canvas;
    $('z4BandTool').addEventListener('click', () => setBandTool(S().tool !== 'band_add'));
    $('z4BandClear').addEventListener('click', () => { if (window.MD_BANDPTS) window.MD_BANDPTS.deleteKeyAt(curFrame()); });
    cv.addEventListener('pointermove', (e) => { if (Z.loupe && S().stage === H.ST.VITMATTE) { Z.mouse = [e.clientX, e.clientY]; V().render(); } });
    cv.addEventListener('pointerleave', () => { if (Z.mouse) { Z.mouse = null; V().render(); } });
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || !obj() || S().auxMode || S().stage !== H.ST.VITMATTE) return;
      if (e.code === 'KeyL' && !e.ctrlKey && !e.metaKey) setLoupe(!Z.loupe);
    });
    window.addEventListener('md-frame', onFrame);
  }

  function init(hooks) {
    H = hooks;
    const host = $('z4Panel'); if (!host) return;
    host.innerHTML = panelHtml();
    buildPalette();
    wire();
    renderList();
    window.MD_ZONES.ready = true;
  }

  // playback bar 🔑✕ / Delete in stage ViTMatte (objects.js): the key of the active zone in frame f, else of any zone there
  // band point keys of the active zone go first (user report 2026-10-21: 🔑✕ did not remove the band key frame)
  const zoneWithKeyAt = (f) => { const a = activeMatte(); if (a && laneKeys(a).includes(f)) return a; return Z.list.find((z) => laneKeys(z).includes(f)) || null; };
  const bandKeyAt = (f) => !!(activeBandKey() && window.MD_BANDPTS && window.MD_BANDPTS.ready && window.MD_BANDPTS.hasKeyAt(f));
  const hasKeyAt = (f) => !!(bandKeyAt(f) || zoneWithKeyAt(f));
  async function deleteKeyAt(f) {
    if (bandKeyAt(f)) return window.MD_BANDPTS.deleteKeyAt(f);
    const z = zoneWithKeyAt(f); if (!z) { toast(`F${f}: brak klucza strefy`); return false; }
    await deleteKey(z.zone_id, f); return true;
  }
  window.MD_ZONES = {
    hasKeyAt, deleteKeyAt, previewMatte, refreshPreview, onTool,
    activeBandKey, pushBandUndo, reloadBand,   // the zone's band for static/bandpoints.js (one band editor, 2026-10-24)
    get alphaPvOn() { return Z.ov !== 'none'; },   // „👁 alfa strefy” – objects.js hides the stage preview tint with it (2026-10-22)
    // instant band from a mask (dashed band edges, no fill): objects.js uses it for the stage 3 / 4 band while a slider
    // is dragged – the exact (adaptive, band points) lines come from the server (2026-10-17, lines 2026-10-23)
    liveBand(maskImg, key, inPx, outPx) { const d = coreDistance(maskImg, key); return d ? insetCanvas(d, inPx, outPx) : null; },
    bandPngEdges,   // a server band PNG as dashed edges (objects.js: previews without a lines endpoint, e.g. LTX-2.5 region)
    compareMode(covers) {   // objects.js 'md-compare': band / zone alpha off in the cut-out view, previous state back after it
      if (covers) { Z.beforeComp = { band: bandPvGet(), ov: Z.ov, off: Z.ovUserOff }; bandPvSet(false); if (Z.ov !== 'none') setOv('none'); }
      else if (Z.beforeComp) { const b = Z.beforeComp; Z.beforeComp = null; bandPvSet(b.band); Z.ovUserOff = b.off; setOv(b.ov || 'none'); }
      if ($('zpBandPv')) { $('zpBandPv').checked = bandPvGet(); $('zpAlphaPv').checked = Z.ov !== 'none'; }
    },
    drawTools: DRAW_TOOLS,
    init, load, reset, draw, drawLoupe, renderLane, onFrame, wantsDraw, onDraw, undo, runDetailsHtml,
    onShow() { loadCleanplate().catch(() => {}); renderList(); onFrame(); },
    onStage() { renderLane(); renderPalette(); },
    onStages() { syncForm(); },   // object stages reloaded (objects.js loadStages): lock of „Zastosuj” (needs stage 3)
    reload() { return load(); },  // zones changed on the server (stage 5 engine switch, objects.js)
    get state() { return Z; },
    hasPending() { return Z.drawPending != null; },
    flush() { return Z.drawPending != null ? commitDraw(Z.drawPending) : Promise.resolve(); },
  };
})();
