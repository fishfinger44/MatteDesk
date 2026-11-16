// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/cleanup3d.js – the steps of the „🧹 Czyszczenie” mode (docs/PLAN_CLEANUP_3D.md §2, package P4 „Szkielet UI”).
// In Cleanup mode the stage bar (#stageBar) shows the Cleanup steps instead of the object stages:
//   C1 Inpaint · C2 Kamera 3D · C3 Karty · C4 Klucze · C5 Projekcja · C6 Eksport
// with the same look and states (✔ done / ↻ to recompute / ○ empty) taken from GET /api/cleanup3d (`done`, optional
// `stale`). C1 = the MVP panel of static/cleanup.js (#cleanupPanel, unchanged). C2–C6: a left floating palette (main
// functions, classes and drag as „Poprawki maski” – window.MD.makeFloating, double click on the head = back to the corner)
// and a right pane (.c3-pane: „Co robi / Najlepsze do”, rare settings, reports). The layered card timeline (#timelineC3,
// P16: one row per card – keys as diamonds, eye / solo (stored on the card, like After Effects), coverage bar – and the patch band row), a preview switch
// in the header (Plate · Klucz · Rzut · Wynik · Karty).
// Every action calls the routes of api/cleanup3d.py; an answer 501 (a later package) is shown as a toast with the
// server's text – never hidden. The object stage bar (#stageList, #stageHint, #stageObjectBox) is only hidden, never
// rewritten, so „✂ Wycinanie” gets it back exactly as it was.
//
// Hook object for other scripts – window.MD_CLEANUP3D:
//   .ready                  true after init (built lazily by the first onMode / refresh)
//   .step                   current step id: 'remove' | 'camera' | 'cards' | 'keys' | 'project' | 'export'
//   .setStep(id)            switch step (remembered per project in localStorage md_c3_step:<work>)
//   .card / .selectCard(id) selected card id (C3 / C4)
//   .key  / .selectKey(card, frame)   selected key {card, frame} or null
//   .state                  last GET /api/cleanup3d answer (null before the first load)
//   .refresh()              reload the state and redraw (call after an action that changed data)
//   .areaMount()            element of the „Klucz karty” palette where the area editor lives (#c3AreaMount; package P7,
//                           static/cleanup3d_areas.js)
//   .extentMount()          the same in the „Karty” palette for the card's extent (#c3ExtentMount)
//   .toolMount()            element of the left tool bar („Rysuj”) for the step's drawing tools (#c3ToolMount, P7)
//   .setCardExtent(id, areas, frame)   the extent saved through PUT …/cards/{card}/extent: kept in the card list, so
//                           a later PUT /api/cleanup3d/cards (order, sliders) does not send the old extent back
//   .onChange(fn)           fn({step, card, key, state}) after every change of step / selection / state
//   .onViewImage            set by the viewer side (P6 / P8): fn({kind, frame, card, url}) when a preview image arrived
//   .onMode(clean)          called by static/cleanup.js when the mode switches
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const V = () => window.viewer;
  const api = (m, u, b) => window.UI.api(m, u, b);
  const toast = (t, ms) => window.UI.toast(t, ms);

  const STEPS = [
    { id: 'remove', n: 'C1', label: 'Inpaint (beta)', hint: 'Inpaint: usuń / zastąp obiekt – maska z MatteDeska, LaMa / MiniMax-H3 / VOID, kotwice wyglądu, wersje' },
    { id: 'camera', n: 'C2', label: 'Kamera 3D', hint: 'Kamera liczona w programie (COLMAP + CoTracker3) – podstawa rzutowania kart' },
    { id: 'cards', n: 'C3', label: 'Karty', hint: 'Płaszczyzny świata (niebo, morze, piasek …) z zasięgiem i kolejnością' },
    { id: 'keys', n: 'C4', label: 'Klucze', hint: 'Klucze kart: eksport / wczytanie (Photoshop), dogenerowanie obszaru' },
    { id: 'project', n: 'C5', label: 'Projekcja', hint: 'Klucze rzucone przez kamerę na każdą klatkę i złożone w plate' },
    { id: 'export', n: 'C6', label: 'Eksport', hint: 'Łata RGBA / pełny kompozyt, Nuke (kamera + karty)' },
  ];
  // Cleanup 3D switched off in this release (engine/features.py; GET /api/version → features.cleanup3d): the bar shows
  // only C1 Inpaint, nothing calls the C2–C6 routes (they answer 404)
  const ALL_STEPS = STEPS.slice();
  let featuresLoaded = null;
  function loadFeatures() {
    featuresLoaded = featuresLoaded || api('GET', '/api/version').then((v) => {
      const on = !(v && v.features && v.features.cleanup3d === false);
      STEPS.splice(0, STEPS.length, ...(on ? ALL_STEPS : ALL_STEPS.slice(0, 1)));
      C3.c3on = on;
      if (!STEPS.some((x) => x.id === C3.step)) C3.step = 'remove';
    }).catch(() => {});
    return featuresLoaded;
  }
  const ICON = { done: '✔', stale: '↻', todo: '○' };
  const STATE_WORD = { done: 'policzone', stale: 'do przeliczenia', todo: 'jeszcze nie policzone' };
  const PAL = { camera: 'c3PalCamera', cards: 'c3PalCards', keys: 'c3PalKeys', project: 'c3PalProject', export: 'c3PalExport' };
  const PANE = { camera: 'c3PaneCamera', cards: 'c3PaneCards', keys: 'c3PaneKeys', project: 'c3PaneProject', export: 'c3PaneExport' };
  const KEY_MODE_LABEL = { manual: 'ręcznie (klatki)', every: 'co N klatek', auto: 'automat', one: 'jeden klucz' };
  const AUTO_DEFAULTS = { threshold: 0.03, min_gap: 4, max_gap: 48 };   // engine/cleanup3d/keymodes.py AUTO_DEFAULTS
  const CARD_COLORS = ['#4ae0ff', '#ffb84a', '#7cff6b', '#ff6bd0', '#b08cff', '#ff7a5c', '#5cffc8', '#ffe45c'];
  const VIEW_TABS = [
    { v: 'plate', label: 'Plate', title: 'Sam plate (wybór plate\'u jak zawsze)' },
    { v: 'key', label: 'Klucz', title: 'Sam klucz wybranej karty (obraz rzutowany na kartę)' },
    { v: 'proj', label: 'Rzut', title: 'Klucze rzucone przez kamerę 3D na bieżącą klatkę' },
    { v: 'cleaned', label: 'Wynik', title: 'Plate „cleaned” – wynik czyszczenia (porównanie A/B / wipe / różnica jak zawsze)' },
    { v: 'cards', label: 'Karty', title: 'Siatka kart na bieżącej klatce' },
  ];

  const C3 = { ready: false, clean: false, c3on: null, step: 'remove', state: null, card: null, key: null, view: 'plate', viewUrl: null,
    cards: [], saveT: null, listeners: [], dragId: null, hidden: {}, bandTool: null };

  // ------------------------------------------------------------------------------------------------ storage
  const projKey = () => { const v = V(); return `md_c3_step:${(v && (v.currentWorkDir || (v.projectData && v.projectData.work))) || ''}`; };
  function loadStep() {
    try { const s = localStorage.getItem(projKey()); if (STEPS.some((x) => x.id === s)) return s; } catch (e) { /* storage blocked */ }
    return 'remove';
  }
  function saveStep() { try { localStorage.setItem(projKey(), C3.step); } catch (e) { /* storage blocked */ } }
  const palPos = (step) => ({
    get: () => { try { return JSON.parse(localStorage.getItem(`md_c3_palette_${step}`) || 'null'); } catch (e) { return null; } },
    set: (q) => { try { if (q) localStorage.setItem(`md_c3_palette_${step}`, JSON.stringify(q)); else localStorage.removeItem(`md_c3_palette_${step}`); } catch (e) { /* storage blocked */ } },
  });

  // ------------------------------------------------------------------------------------------------ state
  function stepState(id) {
    const d = C3.state;
    if (id === 'remove' && !(d && d.done && 'remove' in d.done)) {   // the MVP's result: the plate „cleaned”
      const pl = V() && V().manifest && V().manifest.plates;
      return pl && pl.cleaned ? 'done' : 'todo';
    }
    if (!d) return 'todo';
    const v = (d.done || {})[id];
    if (v === 'stale' || (d.stale && d.stale[id])) return 'stale';
    return v === true || v === 'done' ? 'done' : 'todo';
  }
  const cardById = (id) => C3.cards.find((c) => c.id === id) || null;
  const keysOf = (id) => { const c = ((C3.state && C3.state.cards) || []).find((x) => x.id === id); return (c && c.keys) || []; };
  function emit() { for (const fn of C3.listeners) { try { fn({ step: C3.step, card: C3.card, key: C3.key, state: C3.state }); } catch (e) { console.error(e); } } }

  async function refresh() {
    if (!C3.ready) init();
    await loadFeatures();
    try { C3.state = await api('GET', '/api/cleanup3d'); } catch (e) {
      C3.state = null;
      if (C3.clean && e.status !== 400) toast(`Czyszczenie 3D: ${e.message}`, 6000);
    }
    C3.cards = ((C3.state && C3.state.cards) || []).map((c) => { const o = { ...c }; delete o.keys; return o; });
    if (!cardById(C3.card)) C3.card = C3.cards.length ? C3.cards[0].id : null;
    if (C3.key && !keysOf(C3.key.card).some((k) => k.frame === C3.key.frame)) C3.key = null;
    renderAll();
    emit();
    loadTimeline();                // the rows' coverage numbers (cached on the server by the scene stamp)
    return C3.state;
  }

  // ------------------------------------------------------------------------------------------------ actions
  // one wrapper for every server action: button disabled while it runs, the server's message (also 501) in a toast
  async function act(btnId, method, url, body, okMsg) {
    const b = btnId ? $(btnId) : null;
    if (b) b.disabled = true;
    try {
      const r = await api(method, url, body);
      if (okMsg) toast(typeof okMsg === 'function' ? okMsg(r) : okMsg, 5000);
      await refresh();
      return r;
    } catch (e) {
      toast(e.message || String(e), 8000);
      return null;
    } finally { if (b) { b.disabled = false; syncButtons(); } }
  }

  // ------------------------------------------------------------------------------------------------ step bar
  function renderBar() {
    const list = $('c3StepList'); if (!list) return;
    list.innerHTML = STEPS.map((s) => {
      const st = stepState(s.id);
      return `<li class="stage-item ${st} ${C3.step === s.id ? 'current' : ''}" data-c3step="${s.id}" title="${esc(`${window.mdT ? window.mdT(STATE_WORD[st]) : STATE_WORD[st]} – ${s.hint}`)}">
        <span class="stage-n c3-n">${s.n}</span><span class="stage-ic">${ICON[st]}</span><span class="stage-label">${esc(s.label)}</span></li>`;
    }).join('');
    const hint = $('c3StepHint');
    if (hint) { const s = STEPS.find((x) => x.id === C3.step); hint.textContent = `${s.n} ${s.label}: ${s.hint}`; }
  }

  function setStep(id, opts = {}) {
    if (!STEPS.some((s) => s.id === id)) return;
    C3.step = id;
    if (!opts.noSave) saveStep();
    renderAll();
    if (id === 'remove' && C3.clean && window.MD_CLEANUP && window.MD_CLEANUP.reload) window.MD_CLEANUP.reload();
    emit();
  }

  // the panes / palettes / tools of the current step; nothing of Cleanup 3D outside Cleanup mode
  function showStep() {
    const on = C3.clean;
    const loaded = !!(V() && V().projectLoaded);
    const cu = $('cleanupPanel'); if (cu && on) cu.style.display = C3.step === 'remove' ? '' : 'none';
    for (const [s, id] of Object.entries(PANE)) { const p = $(id); if (p) p.style.display = on && C3.step === s ? '' : 'none'; }
    const cw = $('canvasWrapper');
    for (const [s, id] of Object.entries(PAL)) {
      const p = $(id); if (!p) continue;
      const vis = on && loaded && C3.step === s;
      p.style.display = vis ? '' : 'none';
      if (vis && cw && window.MD && window.MD.placeFloating) window.MD.placeFloating(p, cw, palPos(s).get());
    }
    const tools = $('c3Tools');
    if (tools) {
      const has = on && (C3.step === 'cards' || C3.step === 'keys');
      tools.style.display = has ? '' : 'none';
      const ph = $('c3ToolHint');
      if (ph) ph.textContent = C3.step === 'cards' ? 'Zasięg karty: obrys rysowany raz, w bieżącej klatce – kamera 3D przenosi go na pozostałe' : 'Obszar do dogenerowania: na kluczu albo na dowolnej klatce (przenoszony na klucze)';
    }
  }

  // ------------------------------------------------------------------------------------------------ HTML
  const head = (title, sub) => `<div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> ${title}${sub ? ` <span class="muted">${sub}</span>` : ''}</div>`;
  // label and sentence are SEPARATE text nodes: static/i18n.js translates a node by its exact catalog entry first; one
  // node „Co robi: …” went through the placeholder patterns, where a generic pattern of the same length won for C3
  // („{0}: {1} nie w {2}”) and the description stayed Polish in the English UI
  const what = (does, best) => `<div class="muted pp-hint c3-what"><span>Co robi:</span> <span>${does}</span></div><div class="muted pp-hint c3-what"><span>Najlepsze do:</span> <span>${best}</span></div>`;

  function palettesHtml() {
    return [
      `<div class="pts-palette c3-palette" id="c3PalCamera" style="display:none">${head('Kamera 3D', 'COLMAP + CoTracker3')}
        <div class="pp-sec"><div class="zp-sub" title="Obiekty, które się ruszają (ludzie, auta, fale przy brzegu) – ich piksele nie biorą udziału w liczeniu kamery">Obiekty ruchome (poza solve'em)</div>
          <div id="c3Moving" class="c3-list"></div></div>
        <label class="chk" title="Jedna ogniskowa na całe ujęcie (bez zoomu) – stabilniejszy solve"><input type="checkbox" id="c3SingleFocal" checked title="Jedna ogniskowa"> jedna ogniskowa</label>
        <label class="chk" title="Użyj linii horyzontu (morze / ziemia) jako więzu, gdy jest widoczna"><input type="checkbox" id="c3Horizon" checked title="Horyzont jako więz"> horyzont jako więz</label>
        <button type="button" class="btn btn-primary" id="c3Solve" style="width:100%" title="Policz kamerę 3D ujęcia: punkty COLMAP + ścieżki CoTracker3, wspólna optymalizacja (ogniskowa, dystorsja, poza w każdej klatce)">▶ Policz kamerę</button>
        <button type="button" class="btn btn-secondary btn-sm" id="c3Static" style="width:100%;margin-top:4px" title="Ujęcie ze statywu, kamera się nie rusza: bez COLMAP – jedna poza dla wszystkich klatek. W kroku Karty „Zaproponuj” daje kartę „Plan” na wprost kamery: klucz z C4 przenosi się 1:1 na każdą klatkę (szybkie usuwanie LaMą / VOID)">📌 Kamera nieruchoma (bez liczenia)</button>
        <div class="muted pp-hint" id="c3CamInfo"></div></div>`,
      `<div class="pts-palette c3-palette" id="c3PalCards" style="display:none">${head('Karty', 'kolejność = kto zasłania')}
        <div id="c3CardList" class="c3-list" title="Kolejność kart: góra = najbliżej kamery (zasłania karty niżej). Przeciągnij wiersz albo użyj ▲ / ▼"></div>
        <button type="button" class="btn btn-ghost btn-xs" id="c3CardAdd" title="Nowa karta-płaszczyzna (położenie z kamery 3D ustawisz suwakami)">＋ karta</button>
        <button type="button" class="btn btn-ghost btn-xs" id="c3CardPropose" title="Karty z policzonej kamery 3D: ziemia, morze (gdy widać horyzont) i niebo. Istniejące karty zostają bez zmian">✦ Zaproponuj karty</button>
        <div class="pp-sec" id="c3CardEdit"></div>
        <div class="pp-sec"><div class="zp-sub" title="Obrys zasięgu karty: pędzel albo zamknięta krzywa, rysowany raz w bieżącej klatce; pusty = cała karta">Zasięg karty</div>
          <div id="c3ExtentMount" class="c3-areas"></div></div></div>`,
      `<div class="pts-palette c3-palette" id="c3PalKeys" style="display:none">${head('Klucz karty')}
        <div class="pp-sec"><div class="row c3-row">
          <select id="c3KeyCard" class="form-input" title="Karta, której klucze edytujesz"></select>
          <select id="c3KeyMode" class="form-input" title="Skąd karta bierze klatki kluczy: ręcznie (wybrane klatki), co N klatek zakresu pracy, automat (tam, gdzie rośnie błąd rzutu) albo jeden klucz-suma ze wszystkich klatek"></select>
          <input type="number" id="c3KeyEvery" class="form-input c3-num" min="1" max="500" step="1" title="Co ile klatek klucz (tryb „co N klatek”)"></div>
          <div id="c3AutoBox" style="display:none">
            <div class="row c3-row">
              <label title="Średni błąd rzutu karty z sąsiednich kluczy (różnica plate / rzut poza usuwanymi obiektami), powyżej którego klatka dostaje klucz. Mniej = więcej kluczy">Próg błędu (%)<input type="number" id="c3AutoThr" class="form-input c3-num" min="0.2" max="50" step="0.5" title="Próg błędu rzutu w procentach"></label>
              <label title="Nowy klucz co najmniej tyle klatek od innych kluczy">Min. odstęp<input type="number" id="c3AutoMin" class="form-input c3-num" min="1" max="500" step="1" title="Minimalny odstęp kluczy (klatki)"></label>
              <label title="Nigdy dłuższa przerwa między kluczami (0 = bez limitu)">Maks. odstęp<input type="number" id="c3AutoMax" class="form-input c3-num" min="0" max="5000" step="1" title="Maksymalny odstęp kluczy (klatki, 0 = bez limitu)"></label></div>
            <button type="button" class="btn btn-secondary btn-xs" id="c3KeyAuto" title="Policz błąd rzutu karty na zakresie pracy i dodaj klucze tam, gdzie przekracza próg (pierwsza wersja klucza = klatka plate'u). Istniejące klucze zostają">▶ Wyznacz klucze (automat)</button></div>
          <div id="c3OneBox" style="display:none">
            <button type="button" class="btn btn-secondary btn-xs" id="c3KeyMaster" title="Jeden klucz karty w bieżącej klatce, złożony z czystego tła wszystkich klatek rzuconych przez kamerę 3D (płótno większe niż kadr). Czego nie widzi żadna klatka, zostaje zaznaczone jako obszar do dogenerowania">✦ Zbuduj klucz-sumę</button>
            <div class="muted pp-hint">Klucz-suma powstaje w bieżącej klatce i zastępuje pozostałe klucze karty (są usuwane z dysku na stałe – przed tym zapytam).</div></div>
          <div class="muted pp-hint" id="c3KeyModeInfo"></div></div>
        <div class="pp-sec"><div class="zp-sub">Klucze</div>
          <div id="c3KeyList" class="c3-list" title="Klucze tej karty: klik = skok na klatkę klucza (kwadraty na osi czasu w kolorze karty)"></div>
          <div class="row c3-row"><button type="button" class="btn btn-ghost btn-xs" id="c3KeyAdd" title="Nowy klucz tej karty w bieżącej klatce (pierwsza wersja = klatka plate'u)">＋ klucz w tej klatce</button>
            <button type="button" class="btn btn-ghost btn-xs" id="c3KeyAddEvery" title="Klucze co N klatek na zakresie pracy (oś czasu)">＋ klucze co N</button>
            <button type="button" class="btn btn-ghost btn-xs" id="c3KeyDel" title="Usuń wybrany klucz (wszystkie jego wersje)">🗑</button></div></div>
        <div class="pp-sec"><div class="row c3-row">
          <select id="c3KeyFormat" class="form-input" title="Format eksportu klucza"></select>
          <button type="button" class="btn btn-secondary btn-xs" id="c3KeyExport" title="Zapisz wybrany klucz do pliku (do poprawki w Photoshopie) – domyślnie <work>/inpaint/keys_export/">⬇ Eksportuj klucz</button></div>
          <div class="row c3-row"><input type="text" id="c3KeyPath" class="form-input" placeholder="ścieżka pliku PNG / EXR" title="Plik na tym komputerze, ten sam rozmiar co klucz">
            <button type="button" class="btn btn-secondary btn-xs" id="c3KeyImport" title="Wczytaj poprawiony klucz jako nową wersję (poprzednia zostaje do cofnięcia)">⬆ Wczytaj klucz</button></div></div>
        <div class="pp-sec">
          <button type="button" class="btn btn-secondary btn-xs" id="c3KeyRebase" title="Klucze, które są całymi klatkami z modelu (np. VOID przemalowuje cały kadr): nowa wersja = oryginalny plate, a wypełnienie z modelu tylko w miejscu usuwanych obiektów z kroku C1 (z zapasem i miękką krawędzią, kolor dopasowany do plate'u). Klucze będące już plate'em zostają">⟲ Tło kluczy z plate'u</button></div>
        <div class="pp-sec"><div class="zp-sub" title="Warstwy obszaru jak w „Poprawkach maski”: pędzel albo zamknięta krzywa, widoczność / krycie. Zaznaczenie na zwykłej klatce jest przenoszone na klucze">Obszary do dogenerowania</div>
          <div id="c3AreaMount" class="c3-areas"></div></div>
        <div class="pp-sec">
          <div class="muted pp-hint" id="c3GenHint">Zaznacz obszar w kluczu (pędzel, krzywa albo prostokąt) – przy zaznaczeniu pojawi się pasek: silnik, opis i „✦ Generuj”.</div>
          <button type="button" class="btn btn-ghost btn-xs" id="c3GenPatch" title="Pasek generowania dla całej łaty wybranej wersji (tam, gdzie różni się od plate'u klatki klucza, z pasem łaty z C5) – bez zaznaczania. Np. FLUX 3 Image Klein + LoRA Restore z siłą 0,9 przerysowuje całe wypełnienie">✦ Cała łata…</button>
          <button type="button" class="btn btn-ghost btn-xs" id="c3KeyUndo" title="Wróć do poprzedniej wersji klucza (nowsza zostaje na dysku)">↶ Poprzednia wersja klucza</button>
          <div class="muted pp-hint" id="c3KeyInfo"></div></div></div>`,
      `<div class="pts-palette c3-palette" id="c3PalProject" style="display:none">${head('Projekcja')}
        <div class="muted pp-hint">Zakres pracy: znaczniki [ ] na osi czasu.</div>
        <div class="pp-hint" id="c3RangeHint" style="display:none;color:#fbbf24"></div>
        <label title="auto = rozmycie ruchu dopasowane w każdej klatce do ruchu kamery; wył. = ostre klucze">Rozmycie ruchu<select id="c3Blur" class="form-input">
          <option value="auto">auto</option><option value="off">wył.</option></select></label>
        <div class="pp-sec"><div class="zp-sub" title="Jak łata z kluczy przechodzi w plate: obiekt + jego ślady (cień, kopnięty piasek) + miękka rampa na zewnątrz">Wtopienie łaty</div>
          <label class="chk" title="Podgląd złożenia na bieżącej klatce, liczony w przeglądarce przy każdym ruchu suwaka (bez dopasowania tonu i rozmycia ruchu – te dochodzą w „Policz”)"><input type="checkbox" id="c3BlendPreview" title="Podgląd wtopienia na żywo"> podgląd na żywo</label>
          <label class="zone-slider" title="Szerokość miękkiego przejścia łaty w plate, na zewnątrz od obiektu i jego śladów (px klatki)">Wtopienie (px) <span id="c3KeySoftV" class="muted"></span><input type="range" id="c3KeySoft" min="0" max="200" step="1"></label>
          <label class="zone-slider" title="O ile px łata obejmuje więcej niż sam obiekt (alfa z C1): rozmyta ruchem krawędź i włosy leżą poza maską usuwania – bez poszerzenia zostaje cienki kontur usuniętego obiektu. Klucze to czyste tło, więc poszerzenie jest bezpieczne; działa tylko dla źródła „klucz”">Poszerzenie łaty (px) <span id="c3KeyGrowV" class="muted"></span><input type="range" id="c3KeyGrow" min="0" max="200" step="1"></label>
          <div class="c3-band-row" title="Punkty pasa łaty (jak pas krawędzi w etapach wycinania, tylko strona zewnętrzna): żółty wąs = jak daleko łata sięga na zewnątrz w tym miejscu – dalej, żeby złapać cień, którego automat nie znalazł, albo bliżej. Przeciągnięcie = klucz tej klatki, między kluczami punkt idzie za obiektem">
            <span class="zp-sub">Pas łaty</span>
            <button type="button" class="btn btn-ghost btn-xs" id="c3BandAdd" title="Dodaj punkt pasa łaty: klik przy krawędzi łaty = nowy punkt (klucz tej klatki); przeciągnij żółty wąs = jak daleko łata sięga na zewnątrz. Esc = koniec">⊕ Dodaj punkt</button>
            <button type="button" class="btn btn-ghost btn-xs" id="c3BandRemove" title="Usuń punkt pasa łaty: klik na punkt (albo PPM na punkcie). Esc = koniec">⊖ Usuń punkt</button>
            <button type="button" class="btn btn-ghost btn-xs" id="c3BandClear" style="display:none" title="Usuń klucze pasa łaty w tej klatce">🔑✕</button>
            <div class="muted pp-hint" id="c3BandInfo"></div></div>
          <label class="chk" title="Cień i ślady obiektu (kopnięty piasek) w każdej klatce: tam, gdzie plate wyraźnie różni się od rzutu kluczy i styka się z obiektem"><input type="checkbox" id="c3KeyEffects" title="Usuwaj cień i ślady obiektu"> cień i ślady obiektu</label>
          <label class="zone-slider" title="Jak duża różnica plate / klucze liczy się jako ślad obiektu: mniej = łapie słabsze cienie, ale też więcej przypadkowych miejsc">Próg śladów <span id="c3KeyThrV" class="muted"></span><input type="range" id="c3KeyThr" min="0.02" max="0.5" step="0.01"></label>
          <div class="muted pp-hint" id="c3BlendInfo"></div>
          <label class="chk" id="c3CleanFirstRow" title="Ujęcia z motion control: clear plate po wyrównaniu pasuje do każdej klatki bez projekcji – łata bierze najpierw clear plate, klucze i LaMa tylko tam, gdzie go nie ma albo nie pasuje. Uwaga: rzeczy ruchome (fale, chmury) i zmiana światła między przejazdami"><input type="checkbox" id="c3CleanFirst" title="Clear plate przed kluczami"> clear plate przed kluczami (motion control)</label>
          <div class="muted pp-hint" id="c3CleanFirstHint"></div>
          <div class="c3-grain" title="Ziarno łaty: łata z kluczy (generatory, rzut) jest czystsza od plate'u, gdy materiał nie był odszumiony na wejściu – wtedy dokładamy jej ziarno o widmie plate'u, nowe w każdej klatce. Liczone w „Policz”; podgląd na żywo pokazuje łatę bez ziarna">
            <div class="zp-sub">Ziarno</div>
            <label title="auto = ziarno tylko gdy plate jest wyraźnie bardziej zaszumiony od łaty (odszumiony plate = bez zmian); wł. = zawsze dokładaj brakujące ziarno; wył. = nigdy. Podgląd na żywo nie pokazuje ziarna">Ziarno łaty<select id="c3Grain" class="form-input">
              <option value="auto">auto</option><option value="off">wył.</option><option value="on">wł.</option></select></label>
            <label class="zone-slider" title="Mnoży brakujące ziarno (sqrt(plate² − łata²) w każdym paśmie częstotliwości): 1 = jak plate, mniej = delikatniej, więcej = mocniej. Podgląd na żywo nie pokazuje ziarna">Siła ziarna <span id="c3GrainStrengthV" class="muted"></span><input type="range" id="c3GrainStrength" min="0" max="2" step="0.05"></label>
            <div class="muted pp-hint" id="c3GrainInfo"></div></div></div>
        <div class="pp-sec" id="c3HzSec"><div class="zp-sub" title="Morze to płaska karta: mały błąd kamery, obiektywu albo wysokości karty przesuwa rzucony horyzont o kilka px i łata robi schodek na styku z plate'em. Dopasowanie przesuwa rzut pionowo na horyzont plate'u, z płynnym wygaszeniem w górę i w dół">Horyzont</div>
          <label class="chk" title="Przesuń rzut kluczy pionowo tak, żeby jego horyzont leżał na horyzoncie plate'u (w „Policz” i w podglądzie na żywo). Horyzont plate'u: narysowana linia albo automat"><input type="checkbox" id="c3HzOn" title="Dopasuj horyzont"> Dopasuj horyzont</label>
          <label class="zone-slider" title="Na ile px w górę i w dół od horyzontu przesunięcie płynnie maleje do zera (dalej rzut zostaje bez zmian)">Zasięg dopasowania (px) <span id="c3HzFadeV" class="muted"></span><input type="range" id="c3HzFade" min="16" max="600" step="4"></label>
          <div class="row c3-row"><button type="button" class="btn btn-ghost btn-xs" id="c3HzDraw" title="Narysuj linię horyzontu plate'u w bieżącej klatce edytorem krzywych (przeciągnij punkt, klik na linii = nowy punkt, Delete = usuń punkt, Esc = koniec). Między narysowanymi klatkami linia jest interpolowana, poza nimi działa automat">✎ Narysuj horyzont w tej klatce</button>
            <button type="button" class="btn btn-ghost btn-xs" id="c3HzDel" title="Usuń narysowaną linię horyzontu z bieżącej klatki (znów działa automat)">🗑</button></div>
          <label class="chk" title="Pokaż na klatce horyzont plate'u (niebieska przerywana), horyzont rzutu kluczy (pomarańczowa) i narysowaną linię (żółta)"><input type="checkbox" id="c3HzLines" title="Pokaż linie horyzontu"> pokaż linie horyzontu</label>
          <div class="muted pp-hint" id="c3HzInfo"></div>
          <div class="muted pp-hint" id="c3HzDrawn"></div></div>
        <label class="zone-slider" title="Miękkość samej maski usuwania z kroku C1 (px); puste = ustawienie Czyszczenia">Miękkość maski (px)<input type="number" id="c3Feather" class="form-input" min="0" max="200" step="1" placeholder="domyślne"></label>
        <button type="button" class="btn btn-primary" id="c3Render" style="width:100%" title="Rzuć klucze wszystkich kart na klatki zakresu pracy i złóż w plate → nowa wersja v### i plate „cleaned”">▶ Policz</button>
        <div class="pp-hint" id="c3RenderWarn" style="display:none;color:#fbbf24"></div></div>`,
      `<div class="pts-palette c3-palette" id="c3PalExport" style="display:none">${head('Eksport')}
        <div class="muted pp-hint">Łata RGBA i pełny kompozyt: wersje v### w &lt;work&gt;/inpaint (jak w C1).</div>
        <div class="pp-sec"><div class="zp-sub" title="Skrypt Nuke'a odtwarzający projekcję: kamera 3D, karty, klucze przez Project3D, dystorsja przez STMap, Merge na plate przez maskę usuwania">Nuke</div>
          <button type="button" class="btn btn-secondary btn-sm" id="c3NukeExport" disabled title="Zapisz skrypt Nuke'a (.nk) z kamerą (Camera3 + plik .chan), kartami (Card3D, niebo jako sfera), kluczami (Read → Project3D z własną kamerą projektora) i dystorsją (STMap) do <work>/inpaint/nuke/">⬇ Nuke: kamera + karty + klucze</button>
          <div class="muted pp-hint" id="c3NukeHint"></div></div></div>`,
    ].join('');
  }

  function panesHtml() {
    return `<div class="stage-pane c3-pane" id="c3PaneCamera" style="display:none">
        <div class="stage-title">C2 Kamera 3D</div>
        ${what('liczy kamerę ujęcia w programie – punkty COLMAP i ścieżki CoTracker3, jedna wspólna optymalizacja; obiekty ruchome są pomijane.', 'ruchomej kamery, z której klucze kart mają być rzucane na każdą klatkę.')}
        <div class="zp-sub">Raport solve'u</div><div id="c3CamReport" class="c3-report"></div></div>
      <div class="stage-pane c3-pane" id="c3PaneCards" style="display:none">
        <div class="stage-title">C3 Karty</div>
        ${what('dzieli świat na płaszczyzny (karty) – każda ma własne klucze i zasięg; projekcja i dogenerowanie jednej karty nie wchodzą na sąsiednią.', 'tła z wyraźnymi płaszczyznami: niebo, morze, piasek, ściana.')}
        <div class="muted pp-hint">Podgląd siatki: przełącznik „Karty” u góry. Kolejność na liście = która karta zasłania którą.</div>
        <div id="c3CardsInfo" class="muted pp-hint"></div></div>
      <div class="stage-pane c3-pane" id="c3PaneKeys" style="display:none">
        <div class="stage-title">C4 Klucze</div>
        ${what('klucz = obraz karty w wybranej klatce, rzutowany przez kamerę na pozostałe; obszary klucza można dogenerować albo poprawić w Photoshopie (eksport / wczytanie).', 'usunięcia obiektu z ruchomego ujęcia jednym czystym obrazem na kartę.')}
        <div class="zp-sub">Wersje klucza</div><div id="c3KeyVersions" class="muted pp-hint"></div>
        <div class="zp-sub">Silniki</div>
        <div class="muted pp-hint">LaMa: szybko, małe obszary, bez opisu.</div>
        <div class="muted pp-hint">FLUX 3 Image Klein + LoRA Restore: przerysowanie większych obszarów z opisem; licencja FLUX Kommunity – użycie komercyjne tylko poniżej progu przychodu (szczegóły: Doktor → licencje).</div></div>
      <div class="stage-pane c3-pane" id="c3PaneProject" style="display:none">
        <div class="stage-title">C5 Projekcja</div>
        ${what('rzuca aktywne wersje kluczy wszystkich kart przez kamerę 3D na każdą klatkę zakresu, przycina do zasięgu kart i składa łatę w plate.', 'końcowego przebiegu po poprawieniu kluczy.')}
        <div class="zp-sub">Wyniki</div><div id="c3ProjInfo" class="muted pp-hint">Migotanie per karta, horyzont i wersje v### pojawią się po pierwszym przebiegu.</div></div>
      <div class="stage-pane c3-pane" id="c3PaneExport" style="display:none">
        <div class="stage-title">C6 Eksport</div>
        ${what('łata RGBA (Merge over w Nuke\'u) i plate „cleaned” są zapisywane przy każdym przebiegu; „Nuke” zapisuje skrypt .nk, który odtwarza projekcję: kamera 3D (Camera3 + .chan), karty (Card3D), klucze (Project3D), dystorsja (STMap), maska usuwania z ostatniego przebiegu.', 'dokończenia projekcji w Nuke\'u.')}
        <div class="muted pp-hint" id="c3ExportInfo">Ścieżki i format: &lt;work&gt;/inpaint/v###/ (patch/, comp/).</div>
        <div class="muted pp-hint">W skrypcie nie ma dopasowania rozmycia ruchu, tonu ani zapasowego klucza tam, gdzie bieżący nie sięga – to robi przebieg C5.</div>
        <div class="c3-report" id="c3NukeInfo"></div></div>`;
  }

  // ------------------------------------------------------------------------------------------------ C2 camera
  function renderCamera() {
    const d = C3.state; const cam = d && d.camera;
    const info = $('c3CamInfo');
    if (info) info.textContent = cam ? `✔ ${cam.n_frames} kl.${cam.frames ? ` (F${cam.frames[0]}–F${cam.frames[1]})` : ''} · ${cam.source || ''}` : 'Kamera jeszcze nie policzona.';
    const mv = $('c3Moving');
    if (mv) {
      const objs = ((window.MD && window.MD.S && window.MD.S.objects) || []).filter((o) => o.kind !== 'group');
      const prev = new Set([...mv.querySelectorAll('input[data-c3mv]')].filter((x) => x.checked).map((x) => x.dataset.c3mv));
      mv.innerHTML = objs.length ? objs.map((o) => `<label class="chk"><input type="checkbox" data-c3mv="${esc(o.name)}"${prev.has(o.name) ? ' checked' : ''}> ${esc(o.name)}</label>`).join('')
        : '<div class="muted">brak obiektów (wszystko statyczne)</div>';
    }
    const rep = $('c3CamReport'); if (!rep) return;
    if (!cam) { rep.innerHTML = '<div class="muted">Jeszcze nic nie policzono.</div>'; return; }
    const r = cam.report || {};
    const pick = (...ks) => { for (const k of ks) if (r[k] != null) return r[k]; return null; };
    const num = (x, u) => (x == null ? '—' : `${typeof x === 'number' ? (+x).toFixed(2) : esc(x)}${u}`);
    const rows = [['Błąd reprojekcji', num(pick('reproj_px', 'reprojection_px', 'reproj'), ' px')],
      ['Błąd kluczy', num(pick('key_err_px', 'key_error_px', 'key_error'), ' px')],
      ['Horyzont', num(pick('horizon_px', 'horizon_err_px', 'horizon'), typeof pick('horizon_px', 'horizon_err_px', 'horizon') === 'number' ? ' px' : '')]];
    const known = new Set(['reproj_px', 'reprojection_px', 'reproj', 'key_err_px', 'key_error_px', 'key_error', 'horizon_px', 'horizon_err_px', 'horizon', 'worst', 'worst_frames']);
    for (const [k, v] of Object.entries(r)) if (!known.has(k) && (typeof v === 'number' || typeof v === 'string')) rows.push([k, num(v, '')]);
    const worst = pick('worst_frames', 'worst') || [];
    rep.innerHTML = `<table class="c3-table">${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join('')}</table>
      <div class="zp-sub">Najgorsze klatki</div><div class="c3-worst">${worst.length ? worst.map((w) => {
        const f = typeof w === 'object' ? w.frame : w; const e = typeof w === 'object' && w.err_px != null ? ` (${(+w.err_px).toFixed(2)} px)` : '';
        return `<button type="button" class="btn btn-ghost btn-xs" data-c3jump="${esc(f)}" title="Skocz do klatki">F${esc(f)}${esc(e)}</button>`; }).join('') : '<span class="muted">—</span>'}</div>`;
  }

  // ------------------------------------------------------------------------------------------------ C3 cards
  function cardsBody() { return C3.cards.map((c, i) => ({ ...c, order: i })); }
  async function saveCards(msg) {
    clearTimeout(C3.saveT); C3.saveT = null;
    try {
      const r = await api('PUT', '/api/cleanup3d/cards', { cards: cardsBody() });
      const keys = Object.fromEntries(((C3.state && C3.state.cards) || []).map((c) => [c.id, c.keys || []]));
      if (C3.state) C3.state.cards = (r.cards || []).map((c) => ({ ...c, keys: keys[c.id] || [] }));
      C3.cards = (r.cards || []).map((c) => ({ ...c }));
      if (msg) toast(msg, 3000);
      renderAll(); emit();
      // the previews drawn from the cards follow at once (2026-11-13: a slider moved the card on the server, but the
      // grid only changed with the next frame – it looked as if the sliders did nothing); setView drops a request in
      // flight, so a dragged slider asks only for the latest state
      if (['proj', 'cards', 'blend'].includes(C3.view)) setView(C3.view, true);
    } catch (e) { toast(e.message || String(e), 8000); await refresh(); }
  }
  function saveCardsSoon() { clearTimeout(C3.saveT); C3.saveT = setTimeout(() => saveCards(), 150); }
  function moveCard(id, to) {
    const i = C3.cards.findIndex((c) => c.id === id); if (i < 0) return;
    const j = Math.max(0, Math.min(C3.cards.length - 1, to)); if (i === j) return;
    const [c] = C3.cards.splice(i, 1); C3.cards.splice(j, 0, c);
    saveCards(`Kolejność kart: ${C3.cards.map((x) => x.name).join(' › ')}`);
  }
  function renderCards() {
    const box = $('c3CardList'); if (!box) return;
    box.innerHTML = C3.cards.length ? C3.cards.map((c, i) => `<div class="c3-card ${c.id === C3.card ? 'active' : ''}" draggable="true" data-c3card="${esc(c.id)}" title="Karta ${esc(c.name)}: klik = wybierz, przeciągnij = kolejność">
        <input type="color" class="c3-color" data-c3color="${esc(c.id)}" value="${esc(c.color)}" title="Kolor karty (kwadraty kluczy na osi czasu, siatka)">
        <input type="text" class="form-input c3-name" data-c3name="${esc(c.id)}" value="${esc(c.name)}" title="Nazwa karty">
        <select class="form-input c3-kind" data-c3kind="${esc(c.id)}" title="Płaszczyzna, niebo (nieskończoność – tylko obrót kamery) albo głębia (powierzchnia z mapy głębi klatki odniesienia – paralaksa bliskich obiektów)">
          <option value="plane"${c.kind === 'plane' ? ' selected' : ''}>płaszczyzna</option><option value="sky"${c.kind === 'sky' ? ' selected' : ''}>niebo</option><option value="depth"${c.kind === 'depth' ? ' selected' : ''}>głębia</option></select>
        <button type="button" class="btn btn-ghost btn-xs" data-c3up="${esc(c.id)}" title="Wyżej (bliżej kamery)"${i === 0 ? ' disabled' : ''}>▲</button>
        <button type="button" class="btn btn-ghost btn-xs" data-c3down="${esc(c.id)}" title="Niżej (dalej od kamery)"${i === C3.cards.length - 1 ? ' disabled' : ''}>▼</button>
        <button type="button" class="btn btn-ghost btn-xs" data-c3del="${esc(c.id)}" title="Usuń kartę z jej kluczami i zasięgiem – z dysku na stałe, nie da się cofnąć (przed usunięciem zapytam)">🗑</button></div>`).join('')
      : '<div class="muted">Brak kart – „＋ karta”.</div>';
    const ed = $('c3CardEdit'); const c = cardById(C3.card);
    if (ed && c && c.kind === 'depth') ed.innerHTML = depthEdit(c);
    else if (ed) {
      ed.innerHTML = c ? `<div class="zp-sub">${esc(c.name)}</div>
        ${slider('dh', 'Wysokość', c.adjust.dh, -5, 5, 0.01, '', 'Przesunięcie płaszczyzny wzdłuż jej normalnej (jednostki sceny)')}
        ${slider('pitch', 'Nachylenie', c.adjust.pitch, -30, 30, 0.1, '°', 'Obrót płaszczyzny w pionie (stopnie)')}
        ${slider('roll', 'Przechył', c.adjust.roll, -30, 30, 0.1, '°', 'Obrót płaszczyzny wokół osi widzenia (stopnie)')}
        ${slider('seam', 'Szew', c.seam_px, 0, 64, 1, ' px', 'Szerokość miękkiego przejścia na styku z sąsiednimi kartami')}`
        : '<div class="muted">Wybierz kartę z listy.</div>';
    }
    const info = $('c3CardsInfo');
    if (info) info.textContent = C3.cards.length ? `${C3.cards.length} kart: ${C3.cards.map((x) => `${x.name} (${keysOf(x.id).length} kl.)`).join(', ')}` : 'Brak kart.';
  }
  // depth card (P18): the surface of the reference frame's depth map; scale / offset fitted to the camera, sliders on top
  const FIT_LABEL = { points: 'punkty kamery 3D', planes: 'karty-płaszczyzny', none: 'bez dopasowania – ustaw suwakami' };
  function depthStep(c) { const zm = (c.depth && c.depth.fit && +c.depth.fit.z_median) || 10; return Math.max(1e-3, zm / 200); }
  function depthEdit(c) {
    const d = c.depth || {}; const fit = d.fit || {};
    if (!C3.depthSrc && !C3.depthSrcBusy) loadDepthSources();
    const srcs = C3.depthSrc || [];
    const zm = (fit && +fit.z_median) || 10; const st = depthStep(c);
    const fitTxt = fit.method ? `Dopasowanie: ${esc(FIT_LABEL[fit.method] || fit.method)}${fit.inliers ? ` (${esc(fit.inliers)} / ${esc(fit.anchors)} punktów${fit.err_rel != null ? `, błąd ${esc((100 * fit.err_rel).toFixed(1))} %` : ''})` : ''}` : 'Głębia jeszcze nie pobrana';
    return `<div class="zp-sub">${esc(c.name)} – głębia</div>
      <div class="c3-depth-row" title="Klatka, z której mapy głębi powstaje powierzchnia karty (najlepiej klatka jej klucza)">Klatka odniesienia: <b>F${esc(d.frame != null ? d.frame : '—')}</b>
        <button type="button" class="btn btn-ghost btn-xs" data-c3depth="frame" title="Pobierz głębię bieżącej klatki i dopasuj ją do kamery 3D">Bieżąca klatka</button></div>
      <label class="c3-depth-row" title="Skąd mapa głębi: DepthCrafter policzony w narzędziu (etap 0 „Głębia”) albo plate głębi ujęcia">Źródło
        <select class="form-input" data-c3dsrc>${srcs.map((s) => `<option value="${esc(s.id)}"${s.id === d.source ? ' selected' : ''}${s.available ? '' : ' disabled'}>${esc(s.label)}${s.available ? '' : ' – brak'}</option>`).join('') || `<option value="${esc(d.source || '')}">${esc(d.source || '—')}</option>`}</select></label>
      <label class="c3-depth-row" title="Jak zapisana jest mapa: dysparycja (bliżej = jaśniej, jak DepthCrafter) albo odległość Z">Zapis
        <select class="form-input" data-c3dconv><option value="disparity"${d.conv !== 'z' ? ' selected' : ''}>dysparycja (bliżej = jaśniej)</option><option value="z"${d.conv === 'z' ? ' selected' : ''}>odległość Z</option></select></label>
      <div class="c3-depth-row"><button type="button" class="btn btn-ghost btn-xs" data-c3depth="fit" title="Dopasuj skalę i przesunięcie głębi do punktów kamery 3D (albo do kart-płaszczyzn); zeruje suwaki">Dopasuj do kamery</button>
        <span class="muted">${fitTxt}</span></div>
      ${slider('mul', 'Skala głębi', d.mul != null ? d.mul : 1, 0.25, 4, 0.01, '×', 'Mnoży odległość całej powierzchni od kamery klatki odniesienia (1 = dopasowanie)')}
      ${slider('shift', 'Przesunięcie', d.shift || 0, -zm, zm, st, '', 'Przesuwa całą powierzchnię bliżej (−) albo dalej (+) od kamery (jednostki sceny)')}
      ${slider('seam', 'Szew', c.seam_px, 0, 64, 1, ' px', 'Szerokość miękkiego przejścia na styku z sąsiednimi kartami')}
      <div class="muted">Dziury (to, czego klatka odniesienia nie widzi) biorą karty dalej w kolejności.</div>`;
  }
  async function loadDepthSources() {
    C3.depthSrcBusy = true;
    try { const r = await api('GET', '/api/cleanup3d/depth_sources'); C3.depthSrc = r.sources || []; } catch (e) { C3.depthSrc = []; }
    C3.depthSrcBusy = false;
    const c = cardById(C3.card); if (c && c.kind === 'depth') renderCards();
  }
  // the card becomes / stays a depth card: the server stores the frame's depth and fits it to the camera
  async function makeDepth(c, body, msg) {
    C3.depthSrc = null;
    const r = await act(null, 'POST', `/api/cleanup3d/cards/${encodeURIComponent(c.id)}/depth`, body,
      (x) => `${msg}: ${(x && x.fit && (FIT_LABEL[x.fit.method] || x.fit.method)) || ''}`);
    if (!r) await refresh();
  }
  function slider(k, label, v, lo, hi, step, unit, title) {
    return `<label class="zone-slider c3-slider" title="${esc(title)}">${esc(label)}<input type="range" data-c3adj="${k}" min="${lo}" max="${hi}" step="${step}" value="${esc(v)}"><span class="muted" data-c3adjv="${k}">${esc((+v).toFixed(step < 1 ? (step < 0.1 ? 2 : 1) : 0))}${esc(unit)}</span></label>`;
  }
  async function deleteCard(id) {
    const c = cardById(id); if (!c) return;
    const nk = keysOf(id).length;
    const ok = await window.UI.confirmDelete({ title: 'Usunąć kartę?', ok: 'Usuń kartę na stałe',
      what: `<p>Karta <b>${esc(c.name)}</b>${nk ? ` i jej klucze (${nk}: ${esc(keysOf(id).map((k) => `F${k.frame}`).join(', '))}) z wszystkimi wersjami` : ' (bez kluczy)'} oraz zasięg karty.</p>
        <p class="muted">Render C5 trzeba będzie przeliczyć.</p>` });
    if (!ok) return;
    try {
      const r = await api('DELETE', `/api/cleanup3d/cards/${encodeURIComponent(id)}`);
      if (C3.card === id) C3.card = null;
      toast(`Karta ${c.name} usunięta z dysku (zwolniono ${window.UI.bytes(r.freed_bytes)})`, 6000);
    } catch (e) { toast(e.message || String(e), 8000); }
    await refresh();
    // the previews drawn from the cards must lose it too (2026-11-13: the deleted card's grid stayed on screen);
    // no card left = the plate itself
    if (['proj', 'cards', 'blend'].includes(C3.view)) setView(C3.cards.length ? C3.view : 'plate', true);
  }
  function addCard() {
    let n = C3.cards.length + 1; while (cardById(`card_${n}`)) n++;
    const c = { id: `card_${n}`, name: `karta ${n}`, kind: 'plane', plane: { n: [0, -1, 0], d: 1 }, adjust: { dh: 0, pitch: 0, roll: 0 },
      extent: [], extent_frame: null, seam_px: 16, color: CARD_COLORS[(n - 1) % CARD_COLORS.length], key_mode: 'manual', key_every: 8, warps: {} };
    C3.cards.push(c); C3.card = c.id;
    saveCards(`Nowa karta: ${c.name}`);
  }

  async function proposeCards() {
    try {
      const r = await api('POST', '/api/cleanup3d/cards/propose');
      const added = r.added || [];
      await refresh();
      if (added.length && !C3.card) C3.card = added[0];
      toast(added.length ? `Dodane karty: ${added.join(', ')}` : 'Nic nowego – karty ziemi / morza / nieba już są', 5000);
    } catch (e) { toast(e.message || String(e), 8000); }
  }

  // ------------------------------------------------------------------------------------------------ C4 keys
  function engines() { return (C3.state && C3.state.engines) || []; }
  function engineOk(e) { return e && e.available !== false; }
  // the reason of an engine without its model (stage gate, P3: „Brak modelu … – Doktor → Zainstaluj …”)
  function engineWhy(e) { return (e && (e.reason || e.why)) || 'niedostępny'; }
  function renderKeys() {
    const sel = $('c3KeyCard'); if (!sel) return;
    sel.innerHTML = C3.cards.length ? C3.cards.map((c) => `<option value="${esc(c.id)}"${c.id === C3.card ? ' selected' : ''}>${esc(c.name)}</option>`).join('') : '<option value="">— brak kart (C3) —</option>';
    const c = cardById(C3.card);
    const modes = (C3.state && C3.state.key_modes) || ['manual', 'every'];
    $('c3KeyMode').innerHTML = modes.map((m) => `<option value="${esc(m)}"${c && c.key_mode === m ? ' selected' : ''}>${esc(KEY_MODE_LABEL[m] || m)}</option>`).join('');
    $('c3KeyEvery').value = c ? c.key_every : 8;
    $('c3KeyEvery').style.display = c && c.key_mode === 'every' ? '' : 'none';
    renderKeyMode(c);
    const ks = c ? keysOf(c.id) : [];
    $('c3KeyList').innerHTML = ks.length ? ks.map((k) => `<button type="button" class="btn btn-ghost btn-xs c3-keybtn ${C3.key && C3.key.card === c.id && C3.key.frame === k.frame ? 'active' : ''}" data-c3key="${esc(k.frame)}" title="Klucz F${esc(k.frame)}: ${esc(k.versions)} wersji, aktywna ${esc(k.active || '—')} – klik = skok">
        <i class="c3-sq" style="background:${esc(c.color)}"></i>F${esc(k.frame)} <span class="muted">${esc(k.active || '')}</span></button>`).join('')
      : '<div class="muted">Brak kluczy tej karty.</div>';
    const fm = $('c3KeyFormat'); const fmts = (C3.state && C3.state.key_formats) || ['png', 'png16', 'exr'];
    const cur = fm.value; fm.innerHTML = fmts.map((f) => `<option value="${esc(f)}"${f === cur ? ' selected' : ''}>${esc(f.toUpperCase())}</option>`).join('');
    // engines: one without its model is greyed out with the reason (Doktor → Zainstaluj …)
    const es = engines(); const eng = $('c3Engine');
    const keep = es.find((e) => e.id === eng.value && engineOk(e)) ? eng.value : ((es.find(engineOk) || {}).id || '');
    eng.innerHTML = es.map((e) => `<option value="${esc(e.id)}"${e.id === keep ? ' selected' : ''}${engineOk(e) ? '' : ' disabled'} title="${engineOk(e) ? esc(e.label) : esc(engineWhy(e))}">${esc(e.label)}${engineOk(e) ? '' : ` – ${esc(engineWhy(e))}`}</option>`).join('');
    eng.value = keep;
    const off = es.filter((e) => !engineOk(e));
    $('c3EngWhy').textContent = off.map((e) => `${e.label}: ${e.reason || e.why || 'niedostępny'}`).join(' · ');
    $('c3SoftenRow').style.display = keep === 'klein_restore' ? '' : 'none';
    // the description: FLUX 3 Image Klein + LoRA Restore and Qwen-Image-2.1 (appended to its removal instruction)
    const pr = $('c3Prompt'); const takes = PROMPT_ENGINES.includes(keep);
    pr.disabled = !takes;
    pr.placeholder = keep === 'qwen21_edit' ? 'Co ma być w miejscu obiektu (opcjonalnie, po angielsku)'
      : (takes ? 'Opisz, co ma być w obszarze (opcjonalnie)' : 'LaMa wypełnia z otoczenia – bez opisu');
    pr.title = keep === 'qwen21_edit'
      ? 'Qwen-Image-2.1 dostaje cały wycinek z obiektem i polecenie „usuń obiekt z cieniem i śladami, przedłuż tło”; tu dopisz, co ma być w jego miejscu, np. „wet sand with footprints, the sea behind” – puste = samo polecenie. Enter = Generuj'
      : 'Co ma być w obszarze po usunięciu, np. „mokry piasek, ślady stóp, fale” – puste = domyślne zdanie LoRA Restore. Enter = Generuj';
    const kid = C3.key ? `${C3.key.card}:${C3.key.frame}` : '';
    if (pr.dataset.key !== kid) { pr.dataset.key = kid; pr.value = keyPrompt(); }   // the last description of this key
    renderGenSteps(keep);
    placeGenBar();
    renderVersions();
    syncButtons();
  }
  const curKey = () => (C3.key ? keysOf(C3.key.card).find((x) => x.frame === C3.key.frame) || null : null);
  const versionsOf = (k) => (k && Array.isArray(k.list) ? k.list : []);
  // „Z wersji:” (default = the active version; a chosen base stays while the key is selected, so the same step can be
  // re-run with another engine / setting from it), the area options and the FLUX 3 Image Klein strength
  function renderGenSteps(engine) {
    const k = curKey(); const kid = C3.key ? `${C3.key.card}:${C3.key.frame}` : '';
    if (!C3.genBase || C3.genBase.kid !== kid) C3.genBase = { kid, v: null };
    const vs = versionsOf(k);
    if (C3.genBase.v && !vs.some((v) => v.v === C3.genBase.v)) C3.genBase.v = null;
    const base = C3.genBase.v || (k && k.active) || '';
    const bs = $('c3GenBase');
    if (bs) {
      bs.innerHTML = vs.length ? vs.slice().reverse().map((v) => `<option value="${esc(v.v)}"${v.v === base ? ' selected' : ''}>${esc(v.v)}${v.v === (k && k.active) ? ' (aktywna)' : ''} · ${esc(v.source || '')}</option>`).join('')
        : `<option value="${esc((k && k.active) || '')}">${esc((k && k.active) || '—')}</option>`;
      bs.disabled = !vs.length;
    }
    const kl = engine === 'klein_restore';
    const row = $('c3StrengthRow'); if (row) row.style.display = kl ? '' : 'none';
    const st = $('c3Strength'); const s = genStrength();
    if (st) st.value = String(s);
    if ($('c3StrengthV')) $('c3StrengthV').textContent = s.toFixed(2);
    if (kl && $('c3SoftenRow')) $('c3SoftenRow').style.display = genArea() === 'marked' ? '' : 'none';   // no softening of a redraw
  }
  // the C4 right pane: the versions of the selected key – name, source, base, prompt, date; click = make active,
  // 🗑 = deleted from disk for good after a warning (not the active one, not the last one – delete the key instead)
  function renderVersions() {
    const box = $('c3KeyVersions'); if (!box) return;
    const k = curKey();
    if (!k) { box.textContent = 'Wybierz klucz.'; return; }
    const name = cardById(C3.key.card) ? cardById(C3.key.card).name : C3.key.card;
    const head = `<div>${esc(name)} F${esc(k.frame)}: ${esc(k.versions)} wersji, aktywna ${esc(k.active || '—')}</div>`;
    const vs = versionsOf(k);
    const one = vs.length <= 1;
    box.innerHTML = head + (vs.length ? `<div class="c3-list c3-vlist">${vs.slice().reverse().map((v) => {
      const act = v.v === k.active;
      const meta = [v.source || '', v.base ? `z ${v.base}` : '', v.area === 'patch' ? 'cała łata' : '', v.strength != null ? `siła ${(+v.strength).toFixed(2)}` : '', v.date || ''].filter(Boolean).map(esc).join(' · ');
      const why = act ? 'Aktywna wersja – żeby ją usunąć, ustaw najpierw aktywną inną' : (one ? 'Ostatnia wersja – usuń cały klucz' : 'Usuń tę wersję z dysku na stałe (przed usunięciem zapytam)');
      return `<div class="c3-vrow${act ? ' active' : ''}" data-c3ver="${esc(v.v)}" title="${act ? 'Aktywna wersja (rzutowana w C5)' : 'Klik = ustaw jako aktywną'}">
        <span class="c3-vname">${esc(v.v)}</span><span class="c3-vmeta">${meta}</span>${v.prompt ? `<span class="c3-vprompt">„${esc(v.prompt)}”</span>` : ''}
        <button type="button" class="btn btn-ghost btn-xs c3-vdel" data-c3verdel="${esc(v.v)}"${act || one ? ' disabled' : ''} title="${why}">🗑</button></div>`;
    }).join('')}</div>` : '');
  }
  async function deleteVersion(v) {
    const k = curKey(); if (!k || !C3.key) return;
    const ok = await window.UI.confirmDelete({ title: 'Usunąć wersję klucza?', ok: 'Usuń wersję na stałe',
      what: `<p>Wersja ${esc(v)} klucza F${esc(k.frame)} (obraz tej wersji). Jej numer nie zostanie użyty ponownie.</p>` });
    if (!ok) return;
    await act(null, 'DELETE', keyUrl(`/versions/${encodeURIComponent(v)}`), undefined,
      (r) => `Wersja ${v} usunięta z dysku (zwolniono ${window.UI.bytes(r && r.freed_bytes)})`);
  }
  // key modes of stage III (P13): automat = keys where the projection error grows, jeden klucz = the master key
  const autoOf = (c) => ({ ...AUTO_DEFAULTS, ...((c && c.key_auto) || {}) });
  function renderKeyMode(c) {
    const mode = c ? c.key_mode : 'manual';
    $('c3AutoBox').style.display = mode === 'auto' ? '' : 'none';
    $('c3OneBox').style.display = mode === 'one' ? '' : 'none';
    const a = autoOf(c);
    $('c3AutoThr').value = +(a.threshold * 100).toFixed(2);
    $('c3AutoMin').value = a.min_gap;
    $('c3AutoMax').value = a.max_gap;
    const ks = c ? keysOf(c.id) : [];
    const master = ks.find((k) => k.master);
    $('c3KeyModeInfo').textContent = mode === 'one' ? (master ? `Klucz-suma: F${master.frame}${ks.length > 1 ? ` (+ ${ks.length - 1} innych kluczy)` : ''}` : 'Karta nie ma jeszcze klucza-sumy.')
      : mode === 'auto' ? (C3.autoInfo && C3.autoInfo.card === (c && c.id) ? C3.autoInfo.text : '') : '';
  }
  function autoFromUi() {
    const c = cardById(C3.card); if (!c) return;
    const num = (id, d) => { const v = +$(id).value; return Number.isFinite(v) && $(id).value !== '' ? v : d; };
    c.key_auto = { ...(c.key_auto || {}), threshold: Math.max(0.002, num('c3AutoThr', 3) / 100), min_gap: Math.max(1, Math.round(num('c3AutoMin', 4))), max_gap: Math.max(0, Math.round(num('c3AutoMax', 48))) };
    saveCards();
  }
  async function runAuto() {
    const c = cardById(C3.card); if (!c) return;
    const a = autoOf(c);
    const r = await act('c3KeyAuto', 'POST', `/api/cleanup3d/keys/${encodeURIComponent(c.id)}/auto`, { threshold: a.threshold, min_gap: a.min_gap, max_gap: a.max_gap },
      (x) => `Automat: ${x.frames.length} kluczy (nowe: ${(x.added_keys || []).length}), największy błąd ${(x.max_error * 100).toFixed(1)} %`);
    if (r) { C3.autoInfo = { card: c.id, text: `Klucze: F${r.frames.join(', F')} · największy błąd ${(r.max_error * 100).toFixed(1)} % (próg ${(r.threshold * 100).toFixed(1)} %)` }; renderKeys(); }
  }
  async function runMaster() {
    const c = cardById(C3.card); if (!c || !V()) return;
    const f = V().currentFrame;
    const others = keysOf(c.id).filter((k) => !(k.frame === f && k.master));
    let replace = false;
    if (others.length) {
      const ok = await window.UI.dialog({ title: 'Zastąpić klucze karty?', html: `<p>Klucz-suma karty ${esc(c.name)} powstanie w klatce ${esc(f)} i zastąpi ${esc(others.length)} kluczy (F${esc(others.map((k) => k.frame).join(', F'))}).</p><p><b>Stare klucze (ze wszystkimi wersjami) zostaną usunięte z dysku na stałe – nie da się tego cofnąć.</b></p>`, ok: 'Zbuduj i usuń stare klucze', cancel: 'Anuluj', tone: 'danger' });
      if (!ok) return;
      replace = true;
    }
    const r = await act('c3KeyMaster', 'POST', `/api/cleanup3d/keys/${encodeURIComponent(c.id)}/master`, { frame: f, replace },
      (x) => `Klucz-suma F${x.key_frame}: płótno ${x.canvas.w}×${x.canvas.h}, ${x.frames.length} klatek${x.hole_px ? ' – niewidoczne w żadnej klatce zaznaczone do dogenerowania' : ''}`);
    if (r) selectKey(c.id, r.key_frame);
  }
  function syncButtons() {
    const has = !!C3.key; const card = !!cardById(C3.card);
    const eng = engines().find((e) => e.id === ($('c3Engine') || {}).value);
    for (const id of ['c3KeyExport', 'c3KeyImport', 'c3KeyUndo', 'c3KeyDel', 'c3GenPatch']) { const b = $(id); if (b) b.disabled = !has; }
    const gp = $('c3GenPatch'); const ck = has ? curKey() : null;
    if (gp && ck && ck.master) gp.disabled = true;                  // a master key has its own canvas: no plate to differ from
    for (const id of ['c3KeyAdd', 'c3KeyAddEvery', 'c3KeyAuto', 'c3KeyMaster']) { const b = $(id); if (b) b.disabled = !card; }
    const rg = $('c3Regen');
    if (rg) {
      rg.disabled = !has || !engineOk(eng);
      const info = $('c3KeyInfo');
      if (info) info.textContent = !has ? 'Najpierw wybierz klucz.' : (!engineOk(eng) ? `Silnik niedostępny: ${(eng && (eng.reason || eng.why)) || 'brak silnika'}` : '');
    }
    renderVoid();
  }
  const keyUrl = (tail) => `/api/cleanup3d/keys/${encodeURIComponent(C3.key.card)}/${C3.key.frame}${tail}`;
  function selectKey(card, frame) {
    C3.key = card && frame != null ? { card, frame: +frame } : null;
    if (card) C3.card = card;
    renderAll(); emit();
  }

  // ------------------------------------------------------------------------------------------------ timeline (P16)
  // Layered card timeline (user 2026-11-10, like After Effects): under the frame ruler one ROW per card – colour, name,
  // key mode, 👁 / S (solo) – with its keys as diamonds on the time axis (click = jump + select the key, drag = move
  // the key to another frame after a confirm: POST …/keys/{card}/{frame}/move) and a thin coverage bar (GET
  // /api/cleanup3d/timeline: per frame the card's weight in the frame; orange = its keys do not fully cover what it
  // shows, the render falls back to other sources there). The last row „Łata” = the patch band keys (P14, triangles).
  // 👁 = the card on / off in the render, the previews and the Nuke export (PUT …/cards/{card}/off; user 2026-11-10,
  // like After Effects); S = solo, also like After Effects (PUT …/cards/{card}/solo): when any card is soloed only the
  // soloed cards that are on go into the render, the previews and the Nuke export. Both are stored on the card (the
  // server leaves the others out of the scene, the result shows ↻). The rows fold into one lane of all keys (the fold
  // is the only per-browser state).
  const TL_MISS_WARN = 0.02;            // a frame whose card has more than 2 % of its area without full key cover = warning
  C3.tl = { data: null, err: '', folded: false, drag: null, seq: 0, loadedFor: null };
  const tlKey = () => { const v = V(); return `md_c3_tl:${(v && (v.currentWorkDir || (v.projectData && v.projectData.work))) || ''}`; };
  function tlLoad() {
    const k = tlKey(); if (C3.tl.loadedFor === k) return;
    C3.tl.loadedFor = k; C3.tl.folded = false;
    try {
      const d = JSON.parse(localStorage.getItem(k) || 'null');
      if (d) C3.tl.folded = !!d.folded;
    } catch (e) { /* storage blocked: defaults */ }
  }
  function tlSave() {
    try { localStorage.setItem(tlKey(), JSON.stringify({ folded: C3.tl.folded })); } catch (e) { /* storage blocked */ }
  }
  // the soloed cards (stored on the card); empty = no solo, every card that is on is in the scene
  const soloCards = () => ((C3.state && C3.state.cards) || []).filter((c) => c.solo);
  async function loadTimeline() {
    if (!C3.clean || C3.c3on === false) return;
    const seq = ++C3.tl.seq;
    try {
      const d = await api('GET', '/api/cleanup3d/timeline');
      if (seq !== C3.tl.seq) return;
      C3.tl.data = d; C3.tl.err = (d && d.reason) || '';
    } catch (e) {
      if (seq !== C3.tl.seq) return;
      C3.tl.data = null; C3.tl.err = (e && e.message) || String(e);
    }
    renderLane();
  }
  function laneEl() {
    let box = $('timelineC3'); if (box) return box;
    const keys = $('timelineKeys'); if (!keys) return null;
    box = document.createElement('div');
    box.className = 'c3-tl'; box.id = 'timelineC3';
    keys.parentNode.insertBefore(box, keys);
    box.addEventListener('click', onTlClick);
    return box;
  }
  const pct = (f, v) => `${100 * (f - v.firstFrame) / Math.max(1, v.lastFrame - v.firstFrame)}%`;
  function frameRanges(fr) {     // [3,4,5,9] -> "F3–F5, F9"
    const out = []; let a = null; let b = null;
    for (const f of fr) { if (a != null && f === b + 1) { b = f; continue; } if (a != null) out.push(a === b ? `F${a}` : `F${a}–F${b}`); a = f; b = f; }
    if (a != null) out.push(a === b ? `F${a}` : `F${a}–F${b}`);
    return out.join(', ');
  }
  function covOf(id) {
    const d = C3.tl.data; const c = d && d.cards && d.cards[id];
    if (!c) return null;
    const warn = d.frames.filter((f, i) => c.miss[i] > TL_MISS_WARN && c.share[i] + c.miss[i] > 0);
    return { frames: d.frames, share: c.share, miss: c.miss, warn };
  }
  function rgba(hex, a) {
    const h = String(hex || '#4ae0ff').replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((x) => x + x).join('') : h.slice(0, 6), 16) || 0;
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
  }
  function covBar(c, v) {
    const cv = document.createElement('canvas');
    cv.className = 'c3-tl-cov';
    const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
    // one column per frame, the column of frame f centred on its diamond (the axis puts F at (f - first) / range)
    cv.width = n; cv.height = 1;
    const half = n > 1 ? 50 / (n - 1) : 0;
    cv.style.left = `-${half}%`; cv.style.width = `${100 + 2 * half}%`;
    const cov = covOf(c.id);
    if (c.off) { cv.title = 'Karta wyłączona na stałe – nie wchodzi do renderu'; return cv; }
    if (!cov) { cv.title = C3.tl.err ? `Pokrycie: ${C3.tl.err}` : 'Pokrycie: liczę…'; return cv; }
    cv.dataset.warn = String(cov.warn.length);
    const max = Math.max(1e-6, ...cov.share);
    const g = cv.getContext && cv.getContext('2d');
    if (g) {
      cov.frames.forEach((f, i) => {
        const x = f - v.firstFrame; if (x < 0 || x >= n) return;
        const shown = cov.share[i] + cov.miss[i] > 0;
        if (!shown) return;
        g.fillStyle = cov.miss[i] > TL_MISS_WARN ? 'rgba(240,112,58,0.95)' : rgba(c.color, 0.2 + 0.8 * cov.share[i] / max);
        g.fillRect(x, 0, 1, 1);
      });
    }
    cv.title = cov.warn.length
      ? `Pokrycie kluczami: w ${cov.warn.length} kl. klucze nie pokrywają całej karty (${frameRanges(cov.warn)}) – tam render bierze inne źródło (clear plate / auto / generowanie). Jaśniej = większa waga karty w kadrze`
      : 'Pokrycie kluczami: klucze pokrywają kartę w każdej klatce. Jaśniej = większa waga karty w kadrze';
    cv.addEventListener('mousemove', (e) => {
      const r = cv.getBoundingClientRect(); if (!r.width) return;
      const f = v.firstFrame + Math.floor((e.clientX - r.left) / r.width * n);
      const i = cov.frames.indexOf(f);
      if (i >= 0) cv.title = `F${f}: waga karty ${Math.round(cov.share[i] * 100)} % kadru · bez pełnego pokrycia kluczami ${Math.round(cov.miss[i] * 100)} % karty`;
    });
    return cv;
  }
  function keyDiamond(c, k, v, lane) {
    const d = document.createElement('div');
    const sel = C3.key && C3.key.card === c.id && C3.key.frame === k.frame;
    d.className = `zone-key c3-tl-key ${k.frame === v.currentFrame ? 'cur' : ''} ${sel ? 'sel' : ''}`;
    d.dataset.card = c.id; d.dataset.frame = k.frame;
    d.style.left = pct(k.frame, v); d.style.background = c.color;
    d.title = `Klucz karty ${c.name} F${k.frame} (${k.versions} wersji) · klik = skok · przeciągnij = przenieś do innej klatki`;
    d.addEventListener('click', (e) => {
      e.stopPropagation();
      if (C3.tl.dragged) { C3.tl.dragged = false; return; }
      v.goToFrame(k.frame); selectKey(c.id, k.frame);
    });
    d.addEventListener('pointerdown', (e) => {
      if (e.button) return;
      e.stopPropagation();
      C3.tl.drag = { card: c, frame: k.frame, x0: e.clientX, lane, el: d, to: k.frame, moving: false };
      if (d.setPointerCapture && e.pointerId != null) { try { d.setPointerCapture(e.pointerId); } catch (err) { /* synthetic event */ } }
    });
    d.addEventListener('pointermove', onDragMove);
    d.addEventListener('pointerup', onDragEnd);
    d.addEventListener('pointercancel', () => { const g = C3.tl.drag; C3.tl.drag = null; if (g && g.moving) renderLane(); });
    return d;
  }
  function onDragMove(e) {
    const g = C3.tl.drag; const v = V(); if (!g || !v) return;
    if (!g.moving && Math.abs(e.clientX - g.x0) < 4) return;
    g.moving = true;
    const r = g.lane.getBoundingClientRect(); if (!r.width) return;
    const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    g.to = Math.round(v.firstFrame + t * (v.lastFrame - v.firstFrame));
    const taken = keysOf(g.card.id).some((k) => k.frame === g.to && k.frame !== g.frame);
    g.el.style.left = pct(g.to, v);
    g.el.classList.add('c3-tl-moving'); g.el.classList.toggle('c3-tl-bad', taken);
    g.el.title = taken ? `F${g.to}: karta ma już tu klucz` : `→ F${g.to}`;
  }
  async function onDragEnd(e) {
    const g = C3.tl.drag; C3.tl.drag = null;
    if (!g) return;
    if (!g.moving) return;                        // a click: the click handler jumps
    C3.tl.dragged = true; setTimeout(() => { C3.tl.dragged = false; }, 0);
    if (e) onDragMove(Object.assign({}, { clientX: e.clientX }));
    const to = g.to; const c = g.card;
    if (to === g.frame) { renderLane(); return; }
    if (keysOf(c.id).some((k) => k.frame === to)) { toast(`Karta ${c.name} ma już klucz w klatce ${to}`, 5000); renderLane(); return; }
    await moveKey(c, g.frame, to);
  }
  async function moveKey(c, from, to) {
    const k = keysOf(c.id).find((x) => x.frame === from);
    const ok = await window.UI.dialog({ title: 'Przenieść klucz?',
      html: `<p>Klucz karty ${esc(c.name)} z klatki F${esc(from)} do F${esc(to)}.</p>
        <p>Nowy klucz = plate klatki F${esc(to)} (wersja 1) i aktywna wersja ${esc((k && k.active) || '')} rzucona przez kamerę 3D na klatkę F${esc(to)} – tylko w obszarze usuwanym, ze zgraniem tonu (wersja 2). Nic nie jest generowane od nowa; obszary klucza i krzywe krawędzi nie przechodzą na nowy klucz.</p>
        <p><b>Stary klucz (ze wszystkimi wersjami i obszarami) zostanie usunięty z dysku na stałe – nie da się tego cofnąć.</b></p>`,
      ok: 'Przenieś', cancel: 'Anuluj', tone: 'danger' });
    if (!ok) { renderLane(); return; }
    try {
      const r = await api('POST', `/api/cleanup3d/keys/${encodeURIComponent(c.id)}/${from}/move`, { to });
      const cov = r && r.covered != null ? ` · rzut pokrywa ${Math.round(r.covered * 100)} % obszaru usuwanego` : '';
      toast(`Klucz przeniesiony: F${from} → F${to}${r && r.patched ? cov : ' (sam plate – klucz nie miał wypełnienia)'}`, 6000);
      C3.key = { card: c.id, frame: to }; C3.card = c.id;
      await refresh();
      const v = V(); if (v) v.goToFrame(to);
    } catch (err) {
      toast(err.message || String(err), 7000); renderLane();
    }
  }
  // 👁 = the card on / off in the render, the previews and the Nuke export (user 2026-11-10: like the eye of After
  // Effects; stored on the card, reversible – no dialog); its area goes to the cards behind it
  async function toggleOff(c) {
    const off = !c.off;
    try {
      await api('PUT', `/api/cleanup3d/cards/${encodeURIComponent(c.id)}/off`, { off });
      toast(off ? `Karta ${c.name} wyłączona (render do przeliczenia)` : `Karta ${c.name} włączona`, 5000);
      await refresh();
      if (['proj', 'cards', 'blend'].includes(C3.view)) setView(C3.view, true);
    } catch (err) { toast(err.message || String(err), 7000); }
  }
  // S = solo like After Effects (user 2026-11-10): stored on the card, reversible – no dialog; `cards` = several at once
  // („wyłącz solo” takes it from every soloed card)
  async function setSolo(cards, solo) {
    try {
      for (const c of cards) await api('PUT', `/api/cleanup3d/cards/${encodeURIComponent(c.id)}/solo`, { solo });
      if (cards.length === 1) toast(solo ? `Solo: ${cards[0].name} (render do przeliczenia)` : `Solo zdjęte: ${cards[0].name} (render do przeliczenia)`, 5000);
      else toast('Solo wyłączone – wszystkie włączone karty wchodzą do renderu (render do przeliczenia)', 5000);
      await refresh();
      if (['proj', 'cards', 'blend'].includes(C3.view)) setView(C3.view, true);
    } catch (err) { toast(err.message || String(err), 7000); await refresh(); }
  }
  function onTlClick(e) {
    const b = e.target.closest('[data-c3tl]'); if (!b) return;
    e.stopPropagation();
    const id = b.dataset.card; const c = id && ((C3.state && C3.state.cards) || []).find((x) => x.id === id);
    const what = b.dataset.c3tl;
    if (what === 'fold') { C3.tl.folded = !C3.tl.folded; tlSave(); renderLane(); return; }
    if (what === 'all') { setSolo(soloCards(), false); return; }
    if (!c) return;
    if (what === 'eye') toggleOff(c);
    else if (what === 'solo') setSolo([c], !c.solo);
    else if (what === 'name') { C3.card = id; renderAll(); emit(); }
  }
  function renderLane() {
    const box = laneEl(); const v = V(); if (!box) return;
    const cards = (C3.state && C3.state.cards) || [];
    const on = !!(C3.clean && v && cards.length);
    box.style.display = on ? '' : 'none';
    if (C3.tl.drag && C3.tl.drag.moving) return;          // a drag in progress keeps its diamond
    box.innerHTML = '';
    if (!on) return;
    tlLoad();
    const solo = soloCards();
    const names = solo.map((c) => c.name).join(', ');
    const nWarn = cards.reduce((s, c) => { const cv = !c.off && covOf(c.id); return s + (cv && cv.warn.length ? 1 : 0); }, 0);
    box.insertAdjacentHTML('beforeend', `<div class="c3-tl-group">
      <button type="button" class="c3-tl-fold" data-c3tl="fold" title="Zwiń / rozwiń wiersze kart (zwinięte = jeden pas ze wszystkimi kluczami)">${C3.tl.folded ? '▸' : '▾'} Karty (${cards.length})</button>
      ${solo.length ? `<span class="c3-tl-filter" title="Solo: tylko karty z S wchodzą do renderu, podglądów i eksportu do Nuke'a – jak w After Effects">Solo: ${esc(names)}</span>
      <button type="button" class="c3-tl-btn" data-c3tl="all" title="Wyłącz solo na wszystkich kartach – do renderu wchodzą znów wszystkie włączone karty">wyłącz solo</button>` : ''}
      ${nWarn ? `<span class="c3-tl-warn" title="Pomarańczowe klatki na paskach: klucze nie pokrywają całej karty – tam render bierze inne źródło">⚠ ${nWarn} ${nWarn === 1 ? 'karta' : 'karty'} z lukami w kluczach</span>` : ''}
    </div>`);
    if (C3.tl.folded) {
      const lane = document.createElement('div'); lane.className = 'timeline-zones c3-tl-lane';
      lane.title = 'Klucze kart (romby w kolorze karty): klik = skok, przeciągnij = przenieś';
      const order = cards.slice().sort((a, b) => (a.id === C3.card) - (b.id === C3.card));
      for (const c of order) for (const k of c.keys || []) lane.appendChild(keyDiamond(c, k, v, lane));
      box.appendChild(lane);
      return;
    }
    for (const c of cards) {
      const row = document.createElement('div');
      const hid = solo.length > 0 && !c.solo && !c.off;            // outside the solo: not in the render either
      row.className = `c3-tl-row ${c.off ? 'off' : ''} ${hid ? 'hid' : ''} ${c.id === C3.card ? 'cur' : ''}`;
      row.dataset.c3row = c.id;
      const nk = (c.keys || []).length;
      row.innerHTML = `<div class="c3-tl-head">
        <span class="c3-tl-sw" style="background:${esc(c.color)}"></span>
        <span class="c3-tl-name" data-c3tl="name" data-card="${esc(c.id)}" title="Wybierz kartę">${esc(c.name)}</span>
        <span class="c3-tl-mode" title="Tryb kluczy karty (C4)">${esc(KEY_MODE_LABEL[c.key_mode] || c.key_mode)} · ${nk} kl.</span>
        <button type="button" class="c3-tl-btn ${c.off ? '' : 'on'}" data-c3tl="eye" data-card="${esc(c.id)}" title="${c.off ? 'Karta wyłączona (poza renderem, podglądami i eksportem do Nuke\'a) – klik = włącz' : 'Widoczność karty: wyłączona nie wchodzi do renderu (C5), podglądów ani eksportu do Nuke\'a, jej obszar przejmują karty za nią'}">👁</button>
        <button type="button" class="c3-tl-btn ${c.solo ? 'on' : ''}" data-c3tl="solo" data-card="${esc(c.id)}" title="Solo: tylko karty z S wchodzą do renderu, podglądów i eksportu do Nuke'a – jak w After Effects">S</button>
      </div>`;
      const lane = document.createElement('div'); lane.className = 'timeline-zones c3-tl-lane';
      for (const k of c.keys || []) lane.appendChild(keyDiamond(c, k, v, lane));
      const wrap = document.createElement('div'); wrap.className = 'c3-tl-covwrap';
      wrap.appendChild(covBar(c, v));
      row.appendChild(lane); row.appendChild(wrap);
      box.appendChild(row);
    }
    // the patch band keys (P14): triangles like the band keys of the stages
    const bk = (C3.tl.data && C3.tl.data.band_keys) || [];
    const row = document.createElement('div'); row.className = 'c3-tl-row c3-tl-patch'; row.dataset.c3row = 'patch';
    row.innerHTML = `<div class="c3-tl-head"><span class="c3-tl-name">Łata</span>
      <span class="c3-tl-mode" title="Klucze punktów pasa łaty – edytujesz je w C5 w podglądzie „Wtopienie na żywo”">${bk.length ? `klucze pasa: ${bk.length}` : 'bez kluczy pasa'}</span></div>`;
    const lane = document.createElement('div'); lane.className = 'timeline-zones c3-tl-lane';
    for (const f of bk) {
      const m = document.createElement('div');
      m.className = `band-key ${f === v.currentFrame ? 'cur' : ''}`; m.dataset.frame = f;
      m.style.left = pct(f, v);
      m.title = `Klucz punktów pasa łaty F${f} · klik = skok`;
      m.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
      lane.appendChild(m);
    }
    row.appendChild(lane); box.appendChild(row);
  }

  // ------------------------------------------------------------------------------------------------ preview switch
  function renderViewTabs() {
    const box = $('c3ViewTabs'); if (!box) return;
    box.style.display = C3.clean ? '' : 'none';
    const v = V(); const plate = v && v.activePlate;
    const cur = C3.view === 'plate' && plate === 'cleaned' ? 'cleaned' : (C3.view === 'cleaned' && plate !== 'cleaned' ? 'plate' : C3.view);
    box.querySelectorAll('[data-c3view]').forEach((b) => b.classList.toggle('active', b.dataset.c3view === cur));
  }
  function hasPlate(name) { const s = $('plateSelect'); return !!(s && [...s.options].some((o) => o.value === name)); }
  async function setView(kind, quiet) {
    const v = V();
    if (kind === 'plate' || kind === 'cleaned') {
      if (kind === 'cleaned' && !hasPlate('cleaned')) { toast('Brak plate\'u „cleaned” – najpierw policz czyszczenie (C1 albo C5)', 6000); return; }
      C3.view = kind; C3.viewUrl = null; PV.img = null;
      if ($('c3BlendPreview')) $('c3BlendPreview').checked = false;
      if (v && v.setActivePlate) { if (kind === 'cleaned') v.setActivePlate('cleaned'); else if (v.activePlate === 'cleaned') v.setActivePlate('beauty'); }
      if (v && v.render) v.render();
      renderViewTabs(); emit();
      return;
    }
    if (kind === 'blend') { await loadBlend(quiet); return; }
    const f = v ? v.currentFrame : 0;
    const card = C3.card;
    // preview size: the viewer's canvas is rarely wider than this; the server computes the projection on this grid
    const q = [`max_side=${PREVIEW_SIDE}`]; if (card && kind === 'key') q.push(`card=${encodeURIComponent(card)}`);
    const url = `/api/cleanup3d/view/${kind}/${f}?${q.join('&')}`;
    if (C3.viewAbort) { try { C3.viewAbort.abort(); } catch (e) { /* old browser */ } }
    const ac = typeof AbortController === 'function' ? new AbortController() : null;
    C3.viewAbort = ac;
    try {
      const r = await fetch(url, ac ? { signal: ac.signal } : undefined);
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        const det = d && d.detail !== undefined ? d.detail : r.statusText;
        throw new Error(det && typeof det === 'object' ? (det.detail || JSON.stringify(det)) : det);
      }
      const blob = await r.blob();
      if (C3.viewAbort === ac) C3.viewAbort = null;
      if (C3.viewUrl && window.URL && URL.revokeObjectURL) URL.revokeObjectURL(C3.viewUrl);
      C3.viewUrl = window.URL && URL.createObjectURL ? URL.createObjectURL(blob) : url;
      C3.view = kind;
      if (typeof window.MD_CLEANUP3D.onViewImage === 'function') window.MD_CLEANUP3D.onViewImage({ kind, frame: f, card, url: C3.viewUrl });
      renderViewTabs(); emit();
    } catch (e) {
      if (e && e.name === 'AbortError') return;           // a newer frame asked for its own preview
      if (!quiet || C3.view === kind) toast(e.message || String(e), 7000);
      if (C3.view === kind) { C3.view = 'plate'; renderViewTabs(); }
    }
  }

  // ------------------------------------------------------------------------------------------------ mode
  // hide / show an element remembering its own display, so the cutting mode gets it back as it was
  function hide(id, on) {
    const e = $(id); if (!e) return;
    if (on) { if (!(id in C3.hidden)) { C3.hidden[id] = e.style.display; e.style.display = 'none'; } }
    else if (id in C3.hidden) { e.style.display = C3.hidden[id]; delete C3.hidden[id]; }
  }
  function onMode(clean) {
    if (!C3.ready) init();
    const was = C3.clean;
    C3.clean = !!clean;
    document.body.classList.toggle('c3-on', C3.clean);
    for (const id of ['stageObjectBox', 'stageList', 'stageHint', 'viewModeTabs']) hide(id, C3.clean);
    const list = $('c3StepList'); if (list) list.style.display = C3.clean ? '' : 'none';
    const hint = $('c3StepHint'); if (hint) hint.style.display = C3.clean ? '' : 'none';
    if (C3.clean && !was) { C3.step = loadStep(); renderAll(); refresh(); }
    else { renderAll(); if (was !== C3.clean) emit(); }   // the area editor leaves the viewer with the mode
  }

  function renderAll() {
    if (!C3.ready) return;
    renderBar(); showStep(); renderCamera(); renderCards(); renderKeys(); renderLane(); renderViewTabs(); renderBlendUi();
    renderNuke();
  }

  // ------------------------------------------------------------------------------------------------ C6 export to Nuke (P11)
  function renderNuke() {
    const b = $('c3NukeExport'); if (!b) return;
    const done = (C3.state && C3.state.done) || {};
    const why = !done.camera ? 'Najpierw policz kamerę 3D (C2).' : !done.cards ? 'Najpierw dodaj karty (C3).'
      : !done.keys ? 'Najpierw dodaj klucze kart (C4).' : '';
    b.disabled = !!why || !!C3.nukeBusy;
    const hint = $('c3NukeHint');
    if (hint) hint.textContent = why || (done.project ? 'Maska usuwania: ostatni przebieg C5.' : 'Bez przebiegu C5 skrypt nie ma maski usuwania.');
  }
  function nukeInfo(r) {
    const el = $('c3NukeInfo'); if (!el || !r) return;
    const c = r.counts || {};
    const files = [r.nk, r.chan, ...(r.stmaps || []), ...(r.extents || [])].filter(Boolean);
    el.innerHTML = `<div class="zp-sub">Ostatni eksport do Nuke'a</div>
      <div class="muted">${esc(`Karty: ${c.cards || 0} · klucze: ${c.keys || 0} · klatki F${(r.frames || [])[0]}–F${(r.frames || [])[1]}`)}</div>
      ${files.map((p) => `<div class="mono">${esc(p)}</div>`).join('')}
      ${(r.notes || []).map((n) => `<div class="muted">⚠ ${esc(n)}</div>`).join('')}`;
  }
  async function exportNuke() {
    C3.nukeBusy = true;
    const r = await act('c3NukeExport', 'POST', '/api/cleanup3d/export_nuke', {}, (x) => `Skrypt Nuke'a zapisany: ${(x && x.nk) || ''}`);
    C3.nukeBusy = false;
    nukeInfo(r); renderNuke();
  }

  // ------------------------------------------------------------------------------------------------ wiring
  function wire() {
    $('c3StepList').addEventListener('click', (e) => { const li = e.target.closest('[data-c3step]'); if (li) setStep(li.dataset.c3step); });
    $('c3ViewTabs').addEventListener('click', (e) => { const b = e.target.closest('[data-c3view]'); if (b) setView(b.dataset.c3view); });
    // C2
    $('c3Solve').addEventListener('click', () => {
      const moving = [...document.querySelectorAll('#c3Moving input[data-c3mv]')].filter((x) => x.checked).map((x) => x.dataset.c3mv);
      act('c3Solve', 'POST', '/api/cleanup3d/camera/solve', { moving: moving.length ? moving : null, single_focal: $('c3SingleFocal').checked, horizon: $('c3Horizon').checked }, 'Kamera 3D policzona');
    });
    $('c3Static').addEventListener('click', () => act('c3Static', 'POST', '/api/cleanup3d/camera/static', null,
      (r) => `Kamera nieruchoma: ${r.n_frames} kl. – w kroku Karty „Zaproponuj” doda kartę „Plan”`));
    $('c3PaneCamera').addEventListener('click', (e) => { const b = e.target.closest('[data-c3jump]'); if (b && V()) V().goToFrame(+b.dataset.c3jump); });
    // C3
    const list = $('c3CardList');
    list.addEventListener('click', (e) => {
      const up = e.target.closest('[data-c3up]'); const dn = e.target.closest('[data-c3down]');
      if (up) { const id = up.dataset.c3up; moveCard(id, C3.cards.findIndex((c) => c.id === id) - 1); return; }
      if (dn) { const id = dn.dataset.c3down; moveCard(id, C3.cards.findIndex((c) => c.id === id) + 1); return; }
      const del = e.target.closest('[data-c3del]');
      if (del) { e.stopPropagation(); deleteCard(del.dataset.c3del); return; }
      if (e.target.closest('input, select')) return;
      const row = e.target.closest('[data-c3card]'); if (row) { C3.card = row.dataset.c3card; renderAll(); emit(); }
    });
    list.addEventListener('change', (e) => {
      const t = e.target; const c = cardById(t.dataset.c3name || t.dataset.c3color || t.dataset.c3kind); if (!c) return;
      if (t.dataset.c3name !== undefined) c.name = t.value.trim() || c.id;
      else if (t.dataset.c3color !== undefined) c.color = t.value;
      else if (t.dataset.c3kind !== undefined) {
        if (t.value === 'depth') { C3.card = c.id; makeDepth(c, { frame: V() ? V().currentFrame : null, source: (c.depth && c.depth.source) || null }, `Karta głębi z klatki F${V() ? V().currentFrame : ''}`); return; }
        c.kind = t.value; if (c.kind === 'plane' && !c.plane) c.plane = { n: [0, -1, 0], d: 1 };
      }
      saveCards();
    });
    list.addEventListener('dragstart', (e) => { const row = e.target.closest('[data-c3card]'); if (row) { C3.dragId = row.dataset.c3card; if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move'; } });
    list.addEventListener('dragover', (e) => { if (C3.dragId) e.preventDefault(); });
    list.addEventListener('drop', (e) => {
      const row = e.target.closest('[data-c3card]'); const id = C3.dragId; C3.dragId = null;
      if (!row || !id || row.dataset.c3card === id) return;
      e.preventDefault();
      moveCard(id, C3.cards.findIndex((c) => c.id === row.dataset.c3card));
    });
    list.addEventListener('dragend', () => { C3.dragId = null; });
    $('c3CardAdd').addEventListener('click', addCard);
    $('c3CardPropose').addEventListener('click', proposeCards);
    const ed = $('c3CardEdit');
    ed.addEventListener('input', (e) => {
      const k = e.target.dataset.c3adj; const c = cardById(C3.card); if (!k || !c) return;
      const val = +e.target.value;
      if (k === 'seam') c.seam_px = val;
      else if (k === 'mul' || k === 'shift') c.depth = { ...c.depth, [k]: val };
      else c.adjust = { ...c.adjust, [k]: val };
      const lab = ed.querySelector(`[data-c3adjv="${k}"]`);
      if (lab) lab.textContent = `${k === 'seam' ? val.toFixed(0) + ' px' : k === 'mul' ? val.toFixed(2) + '×' : k === 'shift' ? val.toFixed(depthStep(c) < 0.01 ? 3 : 2) : k === 'dh' ? val.toFixed(2) : val.toFixed(1) + '°'}`;
      saveCardsSoon();
    });
    // depth card (P18): reference frame / source / notation / fit go through the server (snapshot + fit)
    ed.addEventListener('click', (e) => {
      const b = e.target.closest('[data-c3depth]'); const c = cardById(C3.card); if (!b || !c || c.kind !== 'depth') return;
      const d = c.depth || {};
      if (b.dataset.c3depth === 'frame' && V()) makeDepth(c, { frame: V().currentFrame, source: d.source || null, conv: d.conv || null }, `Głębia z klatki F${V().currentFrame}`);
      else makeDepth(c, { frame: d.frame, source: d.source || null, conv: d.conv || null }, 'Głębia dopasowana');
    });
    ed.addEventListener('change', (e) => {
      const c = cardById(C3.card); if (!c || c.kind !== 'depth') return;
      const d = c.depth || {};
      if (e.target.dataset.c3dsrc !== undefined) makeDepth(c, { frame: d.frame, source: e.target.value, conv: null }, 'Nowe źródło głębi');
      else if (e.target.dataset.c3dconv !== undefined) makeDepth(c, { frame: d.frame, source: d.source || null, conv: e.target.value }, 'Zapis głębi zmieniony');
    });
    // C4
    $('c3KeyCard').addEventListener('change', (e) => { C3.card = e.target.value || null; C3.key = null; renderAll(); emit(); });
    $('c3KeyMode').addEventListener('change', (e) => { const c = cardById(C3.card); if (!c) return; c.key_mode = e.target.value; renderKeyMode(c); saveCards(); });
    for (const id of ['c3AutoThr', 'c3AutoMin', 'c3AutoMax']) $(id).addEventListener('change', autoFromUi);
    $('c3KeyAuto').addEventListener('click', runAuto);
    $('c3KeyMaster').addEventListener('click', runMaster);
    $('c3KeyEvery').addEventListener('change', (e) => { const c = cardById(C3.card); if (!c) return; c.key_every = Math.max(1, +e.target.value || 8); saveCards(); });
    $('c3KeyList').addEventListener('click', (e) => {
      const b = e.target.closest('[data-c3key]'); if (!b || !C3.card) return;
      const f = +b.dataset.c3key; if (V()) V().goToFrame(f); selectKey(C3.card, f);
    });
    $('c3KeyAdd').addEventListener('click', () => { const c = C3.card; if (c && V()) act('c3KeyAdd', 'POST', `/api/cleanup3d/keys/${encodeURIComponent(c)}`, { frames: [V().currentFrame] }, 'Nowy klucz'); });
    $('c3KeyAddEvery').addEventListener('click', () => { const c = cardById(C3.card); if (c) act('c3KeyAddEvery', 'POST', `/api/cleanup3d/keys/${encodeURIComponent(c.id)}`, { every: c.key_every }, `Klucze co ${c.key_every} kl.`); });
    $('c3KeyDel').addEventListener('click', async () => {
      if (!C3.key) return;
      const ok = await window.UI.confirmDelete({ title: 'Usunąć klucz?', ok: 'Usuń klucz na stałe',
        what: `<p>Klucz karty ${esc(C3.key.card)} w klatce ${esc(C3.key.frame)} – wszystkie jego wersje i obszary.</p>` });
      if (ok) act('c3KeyDel', 'DELETE', keyUrl(''), undefined, (r) => `Klucz usunięty z dysku (zwolniono ${window.UI.bytes(r && r.freed_bytes)})`);
    });
    $('c3KeyExport').addEventListener('click', () => { if (C3.key) act('c3KeyExport', 'POST', keyUrl('/export'), { format: $('c3KeyFormat').value || 'png' }, (r) => `Klucz zapisany: ${(r && r.path) || ''}`); });
    $('c3KeyImport').addEventListener('click', () => {
      if (!C3.key) return;
      const p = $('c3KeyPath').value.trim(); if (!p) { toast('Podaj ścieżkę pliku klucza (PNG / EXR)', 5000); return; }
      act('c3KeyImport', 'POST', keyUrl('/import'), { path: p }, 'Klucz wczytany jako nowa wersja');
    });
    $('c3Engine').addEventListener('change', () => renderKeys());
    $('c3Prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !$('c3Regen').disabled) { e.preventDefault(); $('c3Regen').click(); } });
    window.addEventListener('resize', placeGenBar);
    $('c3Regen').addEventListener('click', () => {
      if (!C3.key) return;
      const engine = $('c3Engine').value; const area = genArea(); const kl = engine === 'klein_restore';
      act('c3Regen', 'POST', keyUrl('/regen'), { engine, soften_seams: kl && area === 'marked' && $('c3Soften').checked, prompt: (PROMPT_ENGINES.includes(engine) && $('c3Prompt').value.trim()) || null,
        base: $('c3GenBase').value || null, area, strength: kl ? genStrength() : null },
      (r) => (area === 'patch' ? `Cała łata dogenerowana – nowa wersja ${r.version || ''} (z ${r.base || ''})` : `Obszar dogenerowany – nowa wersja ${r.version || ''} (z ${r.base || ''})`));
    });
    $('c3GenBase').addEventListener('change', (e) => { if (C3.key) C3.genBase = { kid: `${C3.key.card}:${C3.key.frame}`, v: e.target.value || null }; });
    $('c3GenArea').addEventListener('change', () => renderKeys());
    $('c3Strength').addEventListener('input', (e) => {
      C3.strength = C3.strength || { ...STRENGTH_DEFAULT };
      C3.strength[genArea()] = Math.min(1, Math.max(0.3, +e.target.value || STRENGTH_DEFAULT[genArea()]));
      $('c3StrengthV').textContent = C3.strength[genArea()].toFixed(2);
    });
    $('c3GenPatch').addEventListener('click', () => { if (!C3.key) return; $('c3GenArea').value = 'patch'; renderKeys(); });
    $('c3KeyVersions').addEventListener('click', (e) => {
      const del = e.target.closest('[data-c3verdel]');
      if (del) { e.stopPropagation(); if (!del.disabled) deleteVersion(del.dataset.c3verdel); return; }
      const row = e.target.closest('[data-c3ver]');
      if (!row || !C3.key || row.classList.contains('active')) return;
      act(null, 'PUT', keyUrl('/active'), { version: row.dataset.c3ver }, `Wersja ${row.dataset.c3ver} aktywna`);
    });
    $('c3KeyRebase').addEventListener('click', () => act('c3KeyRebase', 'POST', '/api/cleanup3d/rebase_keys', {},
      (r) => ((r.rebased || []).length ? `Tło z plate'u: ${r.rebased.length} kluczy (bez zmian: ${(r.unchanged || []).length})` : 'Wszystkie klucze mają już tło z plate\'u')));
    $('c3KeyUndo').addEventListener('click', () => { if (C3.key) act('c3KeyUndo', 'POST', keyUrl('/undo'), undefined, 'Poprzednia wersja klucza aktywna'); });
    // C5
    $('c3BlendPreview').addEventListener('change', (e) => setView(e.target.checked ? 'blend' : 'plate'));
    for (const id of ['c3KeySoft', 'c3KeyThr']) $(id).addEventListener('input', () => { blendFromUi(); saveBlendSoon(); });
    for (const id of ['c3KeySoft', 'c3KeyThr']) $(id).addEventListener('change', () => { if (window.MD_BANDPTS) window.MD_BANDPTS.dirty(); });
    wireBand();
    $('c3KeyEffects').addEventListener('change', () => { blendFromUi(); saveBlendSoon(); });
    $('c3CleanFirst').addEventListener('change', () => { blendFromUi(); saveBlendSoon(); });
    $('c3Grain').addEventListener('change', () => { blendFromUi(); saveBlendSoon(); });
    $('c3GrainStrength').addEventListener('input', () => { blendFromUi(); saveBlendSoon(); });
    $('c3Render').addEventListener('click', () => {
      const fe = $('c3Feather').value;
      act('c3Render', 'POST', '/api/cleanup3d/render', { frames: null, motion_blur: $('c3Blur').value, feather_px: fe === '' ? null : +fe },
        (r) => { C3.grain = (r && r.grain) || null; renderGrainUi(); return `Projekcja: ${(r && r.version) || 'gotowe'}${r && r.skipped_text ? ` · pominięte klatki bez kamery 3D: ${r.skipped_text}` : ''}${r && (r.warnings || []).length ? ` · ${r.warnings.join('; ')}` : ''}`; });
    });
    // C6
    $('c3NukeExport').addEventListener('click', exportNuke);
    window.addEventListener('md-frame', () => { if (!C3.clean) return; renderLane(); if (['key', 'proj', 'cards', 'blend'].includes(C3.view)) setView(C3.view, true); });
    window.addEventListener('md-plate', () => { if (C3.clean) renderViewTabs(); });
    window.addEventListener('md-work-area', () => { if (C3.clean) { renderRangeInfo(); loadTimeline(); } });
  }

  // ------------------------------------------------------------------------------------------------ generate bar
  // Like Photoshop's Generative Fill: when the selected key has a marked area, a bar appears right under the selection
  // (above it when there is no room) – engine, description, „✦ Generuj”. Hidden while drawing. The description is
  // used by Qwen-Image-2.1 (appended to its removal instruction) and FLUX 3 Image Klein + LoRA Restore (LaMa fills from
  // the surroundings); the last one used on a key comes back with it (key.json version `prompt`).
  const PROMPT_ENGINES = ['qwen21_edit', 'klein_restore'];
  function genBarHtml() {
    return `<div class="c3-genbar" id="c3GenBar" style="display:none" title="Dogenerowanie zaznaczonego obszaru klucza">
      <select id="c3Engine" class="form-input c3-gen-engine" title="LaMa – szybko, małe obszary, bez opisu; Qwen-Image-2.1 – zalecany do usuwania obiektu z dużych obszarów (edycja wycinka poleceniem, ok. 40 s, licencja tylko niekomercyjna); FLUX 3 Image Klein + LoRA Restore – odtworzenie faktury, nie do wypełniania dziur"></select>
      <input type="text" id="c3Prompt" class="form-input c3-gen-prompt" placeholder="Opisz, co ma być w obszarze (opcjonalnie)" title="Co ma być w obszarze po usunięciu, np. „mokry piasek, ślady stóp, fale” – puste = domyślne zdanie LoRA Restore. Enter = Generuj">
      <label class="chk" id="c3SoftenRow" title="FLUX 3 Image Klein + LoRA Restore: rozmycie odniesienia w obszarze + drugi przebieg – mniej widoczne szwy"><input type="checkbox" id="c3Soften" title="Zmiękcz szwy"> zmiękcz szwy</label>
      <button type="button" class="btn btn-primary btn-xs" id="c3Regen" title="Dogeneruj obszar klucza wybranym silnikiem z wybranej wersji → nowa wersja klucza (wersja bazowa zostaje)">✦ Generuj</button>
      <label class="c3-gen-opt" title="Wersja klucza, od której zaczyna się ten krok (domyślnie aktywna) – ten sam krok z innym silnikiem albo ustawieniem z tej samej bazy daje wersje do porównania">Z wersji: <select id="c3GenBase" class="form-input"></select></label>
      <label class="c3-gen-opt" title="Zaznaczenie = obszary narysowane w kluczu (pędzel, krzywa, prostokąt). Cała łata = całe miejsce, w którym wersja bazowa różni się od plate'u klatki klucza, z pasem łaty z C5 – nie trzeba nic zaznaczać">Obszar: <select id="c3GenArea" class="form-input">
        <option value="marked">zaznaczenie</option><option value="patch">cała łata</option></select></label>
      <label class="c3-gen-opt" id="c3StrengthRow" title="FLUX 3 Image Klein + LoRA Restore: siła przerysowania (denoise). 1,0 = rysuje od nowa; mniej = trzyma się tego, co jest (start z obrazu wersji bazowej). Domyślnie 1,0 dla zaznaczenia, 0,9 dla całej łaty">Siła <span id="c3StrengthV" class="muted"></span><input type="range" id="c3Strength" min="0.3" max="1" step="0.05"></label>
      <div class="muted c3-gen-why" id="c3EngWhy"></div></div>`;
  }
  // key steps (user 2026-11-10): base version, area (zaznaczenie / cała łata), FLUX 3 Image Klein strength per area
  const STRENGTH_DEFAULT = { marked: 1.0, patch: 0.9 };            // engine/cleanup3d/regen.py KLEIN_STRENGTH
  function genArea() { const a = $('c3GenArea'); return a && a.value === 'patch' ? 'patch' : 'marked'; }
  function genStrength() { C3.strength = C3.strength || { ...STRENGTH_DEFAULT }; return C3.strength[genArea()]; }
  function placeGenBar() {
    const bar = $('c3GenBar'); const cw = $('canvasWrapper'); if (!bar || !cw) return;
    const ar = window.MD_C3AREAS;
    const on = C3.clean && C3.step === 'keys' && C3.key;
    const box = on && ar && ar.ready && ar.selectionBox ? ar.selectionBox() : null;
    const W = cw.clientWidth || 0; const H = cw.clientHeight || 0;
    const bw = bar.offsetWidth || 420; const bh = bar.offsetHeight || 40; const m = 8;
    // „cała łata” needs no marked area: without a selection the bar sits at the bottom of the viewer
    if (on && genArea() === 'patch' && (!box || box.drawing)) {
      bar.style.display = '';
      bar.style.left = `${Math.round(Math.max(m, (W - bw) / 2))}px`; bar.style.top = `${Math.round(Math.max(m, H - bh - m))}px`;
      return;
    }
    if (!box || box.drawing) { bar.style.display = 'none'; return; }
    bar.style.display = '';
    const x = Math.min(Math.max(box.x0, m), Math.max(m, W - bw - m));
    let y = box.y1 + 10;
    if (y + bh > H - m) y = box.y0 - bh - 10;                 // no room under the selection: above it
    y = Math.min(Math.max(y, m), Math.max(m, H - bh - m));
    bar.style.left = `${Math.round(x)}px`; bar.style.top = `${Math.round(y)}px`;
  }
  function keyPrompt() {
    const k = C3.key && keysOf(C3.key.card).find((x) => x.frame === C3.key.frame);
    return (k && k.prompt) || '';
  }

  // ------------------------------------------------------------------------------------------------ preview in the viewer
  // The previews are PNGs of the plate's aspect at preview size: drawn stretched over the plate in image space (the
  // viewer's overlayHooks run inside its pan / zoom). A preview of another frame stays until the new one arrives,
  // with its frame written in the corner, so scrubbing never flashes the plate.
  const PREVIEW_SIDE = 1600;
  const PV = { img: null, kind: null, frame: null, keyFrame: null };
  function showPreview({ kind, frame, url }) {
    const im = new Image();
    im.onload = () => {
      if (C3.view !== kind) return;
      PV.img = im; PV.kind = kind; PV.frame = frame;
      const v = V(); if (v && v.render) v.render();
    };
    im.src = url;
  }
  function drawPreview(ctx, img) {
    if (!C3.clean || !PV.img || PV.kind !== C3.view || !['key', 'proj', 'cards', 'blend'].includes(C3.view)) return;
    const w = img.naturalWidth || img.width; const h = img.naturalHeight || img.height;
    ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(PV.img, 0, 0, w, h); ctx.restore();
    if (C3.view === 'blend') drawBandLines(ctx, img);
    const v = V();
    if (v && PV.frame !== v.currentFrame) {
      const z = v.zoom || 1; const fs = 14 / z;
      ctx.save(); ctx.font = `${fs}px "IBM Plex Sans", sans-serif`; ctx.textAlign = 'left';
      const t = `F${PV.frame} → F${v.currentFrame}…`; const tw = ctx.measureText(t).width;
      ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(8 / z, 8 / z, tw + 12 / z, fs * 1.6);
      ctx.fillStyle = '#fbbf24'; ctx.fillText(t, 14 / z, 8 / z + fs * 1.15); ctx.restore();
    }
  }
  // ------------------------------------------------------------------------------------------------ live patch blend (C5)
  // The server gives, at preview size, the keys projected onto the frame (fill) and maps (B difference plate vs keys,
  // G removal mask, R full key coverage) – GET /api/cleanup3d/blend/{f}; the sliders are applied here, in the browser,
  // with the render's rule (KeyFill._patch): hole = removal mask U the object's traces (difference > threshold, in
  // components touching the object, where the keys cover), patch = hole + a smoothstep ramp outward. The overlay =
  // fill with alpha = patch, drawn over the plate by drawPreview. No tone match / motion blur (the render adds them).
  const BLEND_DIFF_MAX = 0.5;
  const BL = { f: null, w: 0, h: 0, scale: 1, fill: null, maps: null, canvas: null, saveT: null };
  function blendSettings() {
    const s = (C3.state && C3.state.render_settings) || {};
    return { key_soft_px: s.key_soft_px ?? 32, key_grow_px: s.key_grow_px ?? 16, key_effects: s.key_effects ?? true, key_effects_thr: s.key_effects_thr ?? 0.1, clean_first: !!s.clean_first,
      grain: s.grain || 'auto', grain_strength: s.grain_strength ?? 1 };
  }
  // P17 grain of the patch: the setting + the noise measured by the last render (equivalent sigma, 8-bit levels)
  function renderGrainUi() {
    if (!$('c3Grain')) return;
    const s = blendSettings();
    if (document.activeElement !== $('c3Grain')) $('c3Grain').value = s.grain;
    if (document.activeElement !== $('c3GrainStrength')) $('c3GrainStrength').value = s.grain_strength;
    $('c3GrainStrength').disabled = $('c3Grain').value === 'off';
    $('c3GrainStrengthV').textContent = `×${(+$('c3GrainStrength').value).toFixed(2)}`;
    const g = C3.grain;
    const lv = (v) => (v == null ? '–' : (v * 255).toFixed(1));
    $('c3GrainInfo').textContent = !g ? 'Pomiar szumu plate\'u i łaty pojawi się po „Policz”.'
      : !g.measured ? 'Ostatnie liczenie: za mało płaskich miejsc obok łaty, żeby zmierzyć szum – ziarno nie dodane.'
        : `Ostatnie liczenie – szum (poziomy 8-bit): plate ${lv(g.plate)}, łata ${lv(g.patch)}; ziarno dodane w ${g.added} z ${g.frames} klatek.`;
  }
  function renderBlendUi() {
    const s = blendSettings();
    if (!$('c3KeySoft')) return;
    if (document.activeElement !== $('c3KeySoft')) $('c3KeySoft').value = s.key_soft_px;
    if (document.activeElement !== $('c3KeyThr')) $('c3KeyThr').value = s.key_effects_thr;
    if ($('c3KeyGrow') && document.activeElement !== $('c3KeyGrow')) $('c3KeyGrow').value = s.key_grow_px;
    if ($('c3KeyGrowV')) $('c3KeyGrowV').textContent = `${Math.round(+$('c3KeyGrow').value)} px`;
    renderBandUi();
    $('c3KeyEffects').checked = !!s.key_effects;
    $('c3CleanFirst').checked = !!s.clean_first;
    $('c3CleanFirstHint').textContent = s.clean_first && !hasPlate('cleanplate') ? 'Projekt nie ma clear plate\'u – przełącznik nic nie zmieni.' : '';
    $('c3KeySoftV').textContent = `${Math.round(+$('c3KeySoft').value)} px`;
    $('c3KeyThrV').textContent = (+$('c3KeyThr').value).toFixed(2);
    $('c3KeyThr').disabled = !$('c3KeyEffects').checked;
    renderGrainUi();
    if ($('c3BlendPreview')) $('c3BlendPreview').checked = C3.view === 'blend';
    renderRangeInfo();
  }
  // C5: work range vs the solved camera (the render skips frames the camera does not have) and the last render's
  // warnings (kept objects that looked lost – engine/cleanup.py check_kept; skipped frames)
  function renderRangeInfo() {
    const hint = $('c3RangeHint'); const warn = $('c3RenderWarn');
    if (hint) {
      const cam = C3.state && C3.state.camera && C3.state.camera.frames;
      const wa = window.MD_WA && window.MD_WA.effective;
      const rng = (a, b) => (a === b ? `${a}` : `${a}–${b}`);
      const out = [];
      if (cam && wa) {
        if (wa[0] < cam[0]) out.push(rng(wa[0], Math.min(wa[1], cam[0] - 1)));
        if (wa[1] > cam[1]) out.push(rng(Math.max(wa[0], cam[1] + 1), wa[1]));
      }
      hint.style.display = out.length ? '' : 'none';
      hint.textContent = out.length ? `Kamera 3D ma klatki ${rng(cam[0], cam[1])}; zakres pracy ${rng(wa[0], wa[1])} – klatki ${out.join(', ')} zostaną pominięte` : '';
    }
    if (warn) {
      const r = C3.state && C3.state.render;
      const lines = r ? (r.warnings || []).slice() : [];
      if (r && (r.skipped || []).length) lines.unshift(`Pominięte klatki bez kamery 3D: ${r.skipped.length}`);
      warn.style.display = lines.length ? '' : 'none';
      warn.innerHTML = lines.map((w) => `<div>⚠ ${esc(r.version)}: ${esc(w)}</div>`).join('');
    }
  }
  function blendFromUi() {
    if (!C3.state) return;
    C3.state.render_settings = { ...C3.state.render_settings, key_soft_px: +$('c3KeySoft').value, key_grow_px: +$('c3KeyGrow').value, key_effects: $('c3KeyEffects').checked, key_effects_thr: +$('c3KeyThr').value, clean_first: $('c3CleanFirst').checked,
      grain: $('c3Grain').value, grain_strength: +$('c3GrainStrength').value };
    renderBlendUi();
    if (C3.view === 'blend') composeBlend();
  }
  function saveBlendSoon() {
    clearTimeout(BL.saveT);
    BL.saveT = setTimeout(async () => {
      try { const r = await api('PUT', '/api/cleanup3d/render_settings', blendSettings()); if (C3.state && r) C3.state.render_settings = r.render_settings; }
      catch (e) { toast(e.message || String(e), 6000); }
    }, 400);
  }
  async function fetchImage(url, signal) {
    const r = await fetch(url, signal ? { signal } : undefined);
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error((d && d.detail) || r.statusText);
    }
    const scale = +(r.headers.get('X-C3D-Scale') || 0);
    const soft = r.headers.get('X-C3D-Band-Soft');
    const blob = await r.blob();
    const bmp = await createImageBitmap(blob);
    const c = document.createElement('canvas'); c.width = bmp.width; c.height = bmp.height;
    const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(bmp, 0, 0);
    return { data: g.getImageData(0, 0, c.width, c.height), scale, soft: soft == null ? null : +soft };
  }
  async function loadBlend(quiet) {
    const v = V(); const f = v ? v.currentFrame : 0;
    if (C3.viewAbort) { try { C3.viewAbort.abort(); } catch (e) { /* old browser */ } }
    const ac = typeof AbortController === 'function' ? new AbortController() : null;
    C3.viewAbort = ac;
    const info = $('c3BlendInfo'); if (info) info.textContent = `Podgląd F${f}: rzut kluczy…`;
    try {
      const sig = ac ? ac.signal : undefined;
      const [fill, maps] = await Promise.all([fetchImage(`/api/cleanup3d/blend/${f}?part=fill`, sig), fetchImage(`/api/cleanup3d/blend/${f}?part=maps`, sig)]);
      const band = await fetchImage(`/api/cleanup3d/blend/${f}?part=band`, sig);
      if (C3.viewAbort === ac) C3.viewAbort = null;
      Object.assign(BL, { f, w: fill.data.width, h: fill.data.height, scale: fill.scale || maps.scale || (fill.data.width / 3840), fill: fill.data, maps: maps.data,
                          band: band.data.width === fill.data.width && band.data.height === fill.data.height ? band.data : null, bandSoft: band.soft });
      C3.view = 'blend';
      composeBlend();
      renderViewTabs(); renderBlendUi(); emit();
    } catch (e) {
      if (e && e.name === 'AbortError') return;
      if (info) info.textContent = '';
      if (!quiet || C3.view === 'blend') toast(e.message || String(e), 7000);
      if (C3.view === 'blend') { C3.view = 'plate'; PV.img = null; renderViewTabs(); renderBlendUi(); const vv = V(); if (vv && vv.render) vv.render(); }
    }
  }
  // two-pass chamfer distance (3-4) from the set pixels, in preview px
  function distanceFrom(set, w, h) {
    const INF = 1e9; const d = new Float32Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = set[i] ? 0 : INF;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; let v = d[i]; if (!v) continue;
      if (x > 0) v = Math.min(v, d[i - 1] + 3);
      if (y > 0) { v = Math.min(v, d[i - w] + 3); if (x > 0) v = Math.min(v, d[i - w - 1] + 4); if (x < w - 1) v = Math.min(v, d[i - w + 1] + 4); }
      d[i] = v;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x; let v = d[i]; if (!v) continue;
      if (x < w - 1) v = Math.min(v, d[i + 1] + 3);
      if (y < h - 1) { v = Math.min(v, d[i + w] + 3); if (x < w - 1) v = Math.min(v, d[i + w + 1] + 4); if (x > 0) v = Math.min(v, d[i + w - 1] + 4); }
      d[i] = v;
    }
    for (let i = 0; i < d.length; i++) d[i] /= 3;
    return d;
  }
  function composeBlend() {
    if (!BL.fill || !BL.maps) return;
    const t0 = performance.now();
    const s = blendSettings(); const { w, h } = BL; const n = w * h;
    const M = BL.maps.data; const F = BL.fill.data;
    const hole = new Uint8Array(n); const base = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (M[i * 4 + 1] > 127) { base[i] = 1; hole[i] = 1; }
    let traces = 0;
    if (s.key_effects) {
      const thr = Math.max(0, Math.min(255, s.key_effects_thr / BLEND_DIFF_MAX * 255));
      const cand = new Uint8Array(n);
      for (let i = 0; i < n; i++) cand[i] = (base[i] || M[i * 4 + 2] > thr) ? 1 : 0;
      // components of the candidates that touch the object (flood fill from the object), only where the keys cover
      const seen = new Uint8Array(n); const stack = new Int32Array(n); let sp = 0;
      for (let i = 0; i < n; i++) if (base[i]) { seen[i] = 1; stack[sp++] = i; }
      while (sp) {
        const i = stack[--sp]; const x = i % w;
        const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i - w, i + w];
        for (const j of nb) if (j >= 0 && j < n && !seen[j] && cand[j]) { seen[j] = 1; stack[sp++] = j; }
      }
      for (let i = 0; i < n; i++) if (seen[i] && !base[i] && M[i * 4] > 127) { hole[i] = 1; traces++; }
    }
    const soft = s.key_soft_px * BL.scale;
    // the band's outer width per pixel: the server's map (band points, GET …?part=band) scaled by the slider now over
    // the value it was computed with (widths are linear in the slider – objband.scales), else the slider everywhere
    const outW = new Float32Array(n);
    const k = BL.band && BL.bandSoft > 0 ? s.key_soft_px / BL.bandSoft : 0;
    for (let i = 0; i < n; i++) outW[i] = k ? BL.band.data[i * 4] * k : soft;
    const alpha = patchAlpha(hole, outW, soft, w, h);
    const out = new ImageData(w, h); const O = out.data;
    for (let i = 0; i < n; i++) {
      O[i * 4] = F[i * 4]; O[i * 4 + 1] = F[i * 4 + 1]; O[i * 4 + 2] = F[i * 4 + 2]; O[i * 4 + 3] = Math.round(alpha[i] * 255);
    }
    if (!BL.canvas) BL.canvas = document.createElement('canvas');
    BL.canvas.width = w; BL.canvas.height = h;
    BL.canvas.getContext('2d').putImageData(out, 0, 0);
    PV.img = BL.canvas; PV.kind = 'blend'; PV.frame = BL.f;
    const info = $('c3BlendInfo');
    if (info) info.textContent = `Podgląd F${BL.f}: ślady obiektu ${Math.round(traces / (BL.scale * BL.scale) / 1000)} tys. px · ${Math.round(performance.now() - t0)} ms`;
    const v = V(); if (v && v.render) v.render();
  }

  // THE patch rule on the browser's live hole (engine/cleanup3d/band.py patch_weight – one rule, P14): the band = outside
  // the hole within outW of it; w = bandsys weight with no inset: distance to the outer band edge over the outer feather
  // `fo`, the feather limited to the band's local width (bandsys.lengths); alpha = 1 in the hole, smoothstep(w) in the band
  function patchAlpha(hole, outW, fo, w, h) {
    const n = w * h; const a = new Float32Array(n);
    let any = false;
    for (let i = 0; i < n; i++) { if (hole[i]) a[i] = 1; else if (outW[i] > 0) any = true; }
    if (!any) return a;
    const dIn = distanceFrom(hole, w, h);
    const outside = new Uint8Array(n);
    for (let i = 0; i < n; i++) outside[i] = !hole[i] && !(dIn[i] <= outW[i]) ? 1 : 0;
    const dOut = distanceFrom(outside, w, h);
    for (let i = 0; i < n; i++) {
      if (hole[i] || outside[i]) continue;
      const lo = Math.min(Math.max(0, fo), dIn[i] + dOut[i]);
      const t = lo > 0 ? Math.max(0, Math.min(1, dOut[i] / lo)) : 1;
      a[i] = t * t * (3 - 2 * t);
    }
    return a;
  }

  // ------------------------------------------------------------------------------------------------ patch band (C5, P14)
  // Band points of the key patch = the one band point editor (static/bandpoints.js, owner „c3d”): active while the C5
  // live blend preview is shown; ⊕ / ⊖ on the palette are its tools, the lines drawn by MD_BANDPTS.strokeLines (drawLines).
  const bandOwner = () => (C3.clean && C3.step === 'project' && C3.view === 'blend' ? 'c3d' : null);
  function setBandTool(t) {
    C3.bandTool = C3.bandTool === t ? null : t;
    if (C3.bandTool && C3.view !== 'blend') setView('blend');
    renderBandUi();
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.onTool();
  }
  function renderBandUi() {
    if (!$('c3BandAdd')) return;
    if (!bandOwner() && C3.bandTool) C3.bandTool = null;
    $('c3BandAdd').classList.toggle('active', C3.bandTool === 'band_add');
    $('c3BandRemove').classList.toggle('active', C3.bandTool === 'band_remove');
    const B = window.MD_BANDPTS;
    if (B && B.ready) { B.info(); B.renderLane(); }
    if (!bandOwner()) { $('c3BandInfo').textContent = 'Punkty pasa łaty edytujesz w podglądzie na żywo.'; $('c3BandClear').style.display = 'none'; }
  }
  function wireBand() {
    $('c3BandAdd').addEventListener('click', () => setBandTool('band_add'));
    $('c3BandRemove').addEventListener('click', () => setBandTool('band_remove'));
    $('c3BandClear').addEventListener('click', () => { const v = V(); if (window.MD_BANDPTS && v) window.MD_BANDPTS.deleteKeyAt(v.currentFrame); });
    $('c3KeyGrow').addEventListener('input', () => { blendFromUi(); });
    // the grown object is the G map of the server: saved, then the preview inputs again
    $('c3KeyGrow').addEventListener('change', async () => {
      blendFromUi(); clearTimeout(BL.saveT);
      try { const r = await api('PUT', '/api/cleanup3d/render_settings', blendSettings()); if (C3.state && r) C3.state.render_settings = r.render_settings; }
      catch (e) { toast(e.message || String(e), 6000); return; }
      if (window.MD_BANDPTS) window.MD_BANDPTS.dirty();
      if (C3.view === 'blend') loadBlend(true);
    });
    document.addEventListener('keydown', (e) => {
      if (!bandOwner()) return;
      const tag = (e.target && e.target.tagName) || '';
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
      if (e.key === 'Escape' && C3.bandTool) { C3.bandTool = null; renderBandUi(); if (window.MD_BANDPTS) window.MD_BANDPTS.onTool(); }
      else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z') && window.MD_BANDPTS && window.MD_BANDPTS.ready) {
        e.preventDefault(); e.stopPropagation(); window.MD_BANDPTS.undo(V() ? V().currentFrame : 0);
      }
    }, true);
  }
  async function reloadBand() {   // a band edit: the band map of the preview again (fill / maps come from the server cache)
    if (C3.view === 'blend') await loadBlend(true);
  }
  function drawBandLines(ctx, img) {
    const B = window.MD_BANDPTS; const v = V();
    if (!B || !B.ready || !bandOwner() || !v || BL.f !== v.currentFrame) return;
    const s = blendSettings();
    B.drawLines(ctx, img, v.currentFrame, { key_soft_px: s.key_soft_px, key_effects: s.key_effects ? 1 : 0, key_effects_thr: s.key_effects_thr }, { live: true });
  }

  function hookViewer(tries = 0) {
    const v = V();
    if (!v || !Array.isArray(v.overlayHooks)) { if (tries < 40) setTimeout(() => hookViewer(tries + 1), 300); return; }
    if (!v.overlayHooks.includes(drawPreview)) v.overlayHooks.unshift(drawPreview);   // before the area editor's redraw
  }

  // ------------------------------------------------------------------------------------------------ VOID + SeedVR2 (P12)
  // Own block in the „Klucz karty” palette (mounted after the palettes, refreshed from syncButtons): VOID removes the
  // moving object from the whole shot ONCE (POST /api/cleanup3d/void – then every card's existing keys get a version =
  // the plate + the VOID fill only where the object and its traces are), „Klucze z VOID” rebuilds them from the last
  // run without a new pass (POST /api/cleanup3d/void/keys), SeedVR2 upscales the patches of the selected card's keys as
  // one clip (POST /api/cleanup3d/keys/{card}/seedvr2). Availability and the last run come with GET /api/cleanup3d
  // (`void`, `seedvr2`: gate of engine/stage_models.cleanup3d_tools – configured in config.local.json, Doctor).
  const VOID_STATUS = { running: 'przerwany albo trwa', failed: 'błąd', cancelled: 'przerwany' };
  function voidHtml() {
    return `<div class="pp-sec" id="c3VoidBox"><div class="zp-sub" title="VOID (Netflix) usuwa ruchomy obiekt z całego ujęcia razem z jego skutkami (cień, ślady). Liczony raz na zakresie pracy w 1344×768; klucze biorą z niego wypełnienie: oryginalny plate + VOID tylko w miejscu obiektu i jego śladów">VOID – wypełnienie ujęcia</div>
      <input type="text" id="c3VoidPrompt" class="form-input" placeholder="Co widać po usunięciu (opis tła, po angielsku)" title="Opis ujęcia PO usunięciu obiektu dla VOID, np. „an empty sandy beach at sunset, calm sea” – puste = zdanie ogólne">
      <div class="row c3-row"><button type="button" class="btn btn-secondary btn-xs" id="c3VoidRun" title="Policz VOID na zakresie pracy (obiekty z kroku C1, długo – kilka minut na GPU), potem nowe wersje istniejących kluczy wszystkich kart: plate + wypełnienie VOID tylko w miejscu obiektu i jego śladów">▶ Policz VOID</button>
        <button type="button" class="btn btn-ghost btn-xs" id="c3VoidKeys" title="Nowe wersje kluczy wszystkich kart z ostatniego przebiegu VOID (bez ponownego liczenia) – np. po dodaniu kluczy">⟲ Klucze z VOID</button></div>
      <div class="muted pp-hint" id="c3VoidInfo"></div>
      <div id="c3VoidInstall"></div>
      <button type="button" class="btn btn-secondary btn-xs" id="c3SeedVR2" title="SeedVR2 podbija łatę kluczy wybranej karty: jeden wycinek wokół obiektu, klucze jako jeden klip (spójna faktura – najlepiej przy gęstych kluczach, co 2–4 klatki), wynik wklejany w plate tylko w miejscu łaty. Nowa wersja każdego klucza (poprzednia zostaje do cofnięcia)">⇧ SeedVR2: podbij klucze karty</button>
      <div class="muted pp-hint" id="c3SvInfo"></div><div id="c3SvInstall"></div></div>`;
  }
  // Not installed: a short card (what it brings, download size, the GPU memory verdict for THIS card – engine/vram_fit)
  // and the Doctor's install with its progress bar (user 2026-11-10); the gate's full reason stays in the tooltip.
  const VRAM_HEAD = { ok: 'mieści się', spill: 'przelewa się do RAM-u', unstable: 'niestabilnie', no: 'nie zmieści się' };
  const VRAM_COL = { ok: 'var(--accent-success)', spill: '#f0a03a', unstable: '#f87171', no: '#f87171' };
  function vramLine(vr) {
    if (!vr) return '';
    // short: the verdict only; the reason (and where the numbers come from) in the tooltip and the install question
    return `<div class="c3-tool-vram" style="color:${VRAM_COL[vr.level] || 'inherit'}" title="${esc(`${vr.text}${vr.why ? ` (${vr.why})` : ''}`)}">${vr.level === 'ok' ? '✔' : '⚠'} Ta karta ${esc(vr.card_gb)} GB: ${esc(VRAM_HEAD[vr.level] || vr.level)}${vr.basis === 'estimate' ? ' (szacunek)' : ''}</div>`;
  }
  function installCardHtml(t) {
    const i = t.info || {};
    return `<div class="c3-tool-card" title="${esc(t.reason || '')}">
      <div><b>${esc(i.label)}</b> – ${t.partly ? 'instalacja niepełna' : 'nie zainstalowany'}</div>
      <div class="muted">${esc(i.what || '')} · pobiera ${esc(i.download || '')}</div>${vramLine(t.vram)}
      <button type="button" class="btn btn-primary btn-xs" data-c3install="${esc(t.install)}">⬇ ${t.partly ? 'Dokończ instalację' : 'Zainstaluj'} ${esc(i.label)} (${esc(i.download || '')})</button></div>`;
  }
  async function installTool(tool) {
    const t = ((C3.state || {})[tool]) || {};
    const i = t.info || {};
    const ui = window.viewer;                      // the app (static/app.js): Doctor's dialog + install with progress bar
    if (!ui || !ui.installWithProgress) return;
    const vr = t.vram && t.vram.level !== 'ok' ? `<p style="color:${VRAM_COL[t.vram.level]}">⚠ Ta karta (${esc(t.vram.card_gb)} GB): ${esc(VRAM_HEAD[t.vram.level])} – ${esc(t.vram.text)}</p>` : '';
    const ok = await ui.uiDialog({ title: `Zainstalować ${i.label}?`, ok: 'Zainstaluj', cancel: 'Anuluj', tone: vr ? 'warn' : 'info',
      html: `${vr}<p>${esc(i.what || '')}. Pobiera ${esc(i.download || '')}; przerwane pobieranie wznawia się.</p>
             <p>Licencja: ${esc(i.license || '')} – pobierając, akceptujesz ją.</p>` });
    if (!ok) return;
    const r = await ui.installWithProgress(tool);
    const d = (r && r.data) || {};
    toast(r && r.ok && d.success !== false ? `${i.label}: zainstalowany` : `${i.label}: instalacja nie powiodła się – ${d.detail || (d.log || []).slice(-1)[0] || 'szczegóły w Doktorze'}`, 9000);
    await refresh();
  }
  function mountVoid() {
    const pal = $('c3PalKeys'); if (!pal || $('c3VoidBox')) return;
    const after = $('c3KeyRebase') ? $('c3KeyRebase').closest('.pp-sec') : null;
    if (after) after.insertAdjacentHTML('afterend', voidHtml()); else pal.insertAdjacentHTML('beforeend', voidHtml());
    $('c3VoidRun').addEventListener('click', () => act('c3VoidRun', 'POST', '/api/cleanup3d/void', { prompt: $('c3VoidPrompt').value.trim() || null, keys: true },
      (r) => `VOID policzony: ${(r && r.run && r.run.frames ? r.run.frames.length : 0)} klatek · nowe wersje kluczy: ${(r && r.keys ? r.keys.length : 0)}`));
    $('c3VoidBox').addEventListener('click', (ev) => {
      const b = ev.target.closest('[data-c3install]');
      if (b) { b.disabled = true; installTool(b.dataset.c3install).finally(() => { b.disabled = false; }); }
    });
    $('c3VoidKeys').addEventListener('click', () => act('c3VoidKeys', 'POST', '/api/cleanup3d/void/keys', {},
      (r) => `Klucze z VOID: ${(r && r.keys ? r.keys.length : 0)} nowych wersji`));
    $('c3SeedVR2').addEventListener('click', () => {
      const c = cardById(C3.card); if (!c) return;
      act('c3SeedVR2', 'POST', `/api/cleanup3d/keys/${encodeURIComponent(c.id)}/seedvr2`, {},
        (r) => `SeedVR2: ${(r && r.keys ? r.keys.length : 0)} kluczy podbitych (wycinek ${r && r.crop ? `${r.crop[2]}×${r.crop[3]}` : '—'})`);
    });
  }
  function renderVoid() {
    const box = $('c3VoidBox'); if (!box) return;
    const st = C3.state || {}; const v = st.void || {}; const sv = st.seedvr2 || {};
    const last = v.last; const newest = v.newest;
    const pr = $('c3VoidPrompt');
    if (pr && document.activeElement !== pr && !pr.value && last && last.prompt) pr.value = last.prompt;
    const lines = [];
    $('c3VoidInstall').innerHTML = !v.available && v.info ? installCardHtml(v) : '';
    if (!v.available && !v.info) lines.push(`VOID niedostępny: ${v.reason || 'brak konfiguracji'}`);
    if (!v.available && !last) { /* the card says it; no „not computed yet” noise */ } else lines.push(last ? `Ostatni przebieg: ${last.finished || last.date || ''} · ${last.frames} kl.${last.size ? ` · ${last.size[0]}×${last.size[1]}` : ''}` : 'VOID jeszcze nie policzony.');
    if (newest && newest.status !== 'done' && (!last || newest.id !== last.id)) lines.push(`Ostatnia próba (${newest.date || ''}): ${VOID_STATUS[newest.status] || newest.status}${newest.error ? ` – ${newest.error}` : ''}`);
    $('c3VoidInfo').textContent = lines.join(' · ');
    const run = $('c3VoidRun'); const keys = $('c3VoidKeys'); const up = $('c3SeedVR2');
    run.disabled = !v.available; run.title = v.available ? run.title : (v.reason || run.title);
    keys.disabled = !last;
    up.disabled = !sv.available || !cardById(C3.card);
    $('c3SvInfo').textContent = !sv.available && !sv.info ? `SeedVR2 niedostępny: ${sv.reason || 'brak konfiguracji'}` : '';
    $('c3SvInstall').innerHTML = !sv.available && sv.info ? installCardHtml(sv) : '';
    // a tool that is not installed: its run button is hidden (the card's install button replaces it)
    run.style.display = keys.style.display = v.available || last ? '' : 'none';
    if (pr) pr.style.display = v.available ? '' : 'none';
    up.style.display = sv.available ? '' : 'none';
  }

  function init() {
    if (C3.ready) return;
    const bar = $('stageBar'); const panel = $('stagePanel'); const cw = $('canvasWrapper');
    if (!bar || !panel) return;
    bar.insertAdjacentHTML('beforeend', `<ol class="stage-list c3-steps" id="c3StepList" style="display:none" title="Kroki Czyszczenia: ✔ zielone = policzone · ↻ pomarańczowe = do przeliczenia · ○ szare = jeszcze nie policzone"></ol>
      <div class="stage-hint c3-hint" id="c3StepHint" style="display:none"></div>`);
    panel.insertAdjacentHTML('beforeend', panesHtml());
    if (cw) {
      cw.insertAdjacentHTML('beforeend', palettesHtml());
      cw.insertAdjacentHTML('beforeend', genBarHtml());
      if (window.MD && window.MD.makeFloating) for (const [s, id] of Object.entries(PAL)) window.MD.makeFloating($(id), cw, '.pp-head', palPos(s));
    }
    const tabsAfter = $('viewModeTabs');
    const tabs = `<div class="view-mode-tabs c3-view-tabs" id="c3ViewTabs" style="display:none" title="Podgląd Czyszczenia">${VIEW_TABS.map((t) => `<button type="button" class="tab-btn" data-c3view="${t.v}" title="${esc(t.title)}">${esc(t.label)}</button>`).join('')}</div>`;
    if (tabsAfter) tabsAfter.insertAdjacentHTML('afterend', tabs);
    else { const hc = document.querySelector('.header-center') || bar; hc.insertAdjacentHTML('beforeend', tabs); }
    const side = $('toolsSidebar');
    if (side) side.insertAdjacentHTML('beforeend', `<div class="c3-tools" id="c3Tools" style="display:none"><div class="tool-section-title">Rysuj</div>
      <div id="c3ToolMount"></div><div class="muted c3-tool-hint" id="c3ToolHint"></div></div>`);
    C3.ready = true;
    wire();
    mountVoid();                       // P12 block of the „Klucz karty” palette
    hookViewer();
  }

  window.MD_CLEANUP3D = {
    get ready() { return C3.ready; },
    get step() { return C3.step; }, setStep,
    get card() { return C3.card; }, selectCard(id) { if (cardById(id)) { C3.card = id; renderAll(); emit(); } },
    get key() { return C3.key ? { ...C3.key } : null; }, selectKey,
    get state() { return C3.state; },
    get view() { return C3.view; }, setView,
    refresh, onMode,
    areaMount: () => $('c3AreaMount'),
    extentMount: () => $('c3ExtentMount'),
    setCardExtent(id, areas, frame) {
      for (const list of [C3.cards, (C3.state && C3.state.cards) || []]) {
        const c = list.find((x) => x.id === id); if (c) { c.extent = JSON.parse(JSON.stringify(areas || [])); c.extent_frame = frame == null ? null : +frame; }
      }
    },
    toolMount: () => $('c3ToolMount'),
    onChange(fn) { if (typeof fn === 'function') C3.listeners.push(fn); },
    onAreasDrawn: placeGenBar,         // the area editor calls it after every redraw (selection moved / changed)
    onViewImage: showPreview,          // replaceable; the default draws the preview in the viewer (drawPreview)
    bandOwner,                         // static/bandpoints.js: „c3d” while the patch band is edited (P14)
    bandTool: () => C3.bandTool || null,
    reloadBand,
    patchAlpha,                        // the patch rule (tests compare it with engine/cleanup3d/band.py)
    STEPS: STEPS.map((s) => s.id),
  };
  function start() {
    init();
    if (window.MD_CLEANUP && window.MD_CLEANUP.refresh) window.MD_CLEANUP.refresh();   // cleanup.js may have applied the mode before us
    else if (window.MD_CLEANUP && window.MD_CLEANUP.mode === 'clean') onMode(true);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
