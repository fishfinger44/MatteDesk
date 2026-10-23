// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/objects.js – stage bar + object stages (layout of the user decision 2026-10-15): 1 SAM3 → 2 CoTracker (brush
// corrections, static/brush.js) → 3 MatAnyone → 4 VideoMaMa (2026-10-16) → 5 ViTMatte (static/zones.js) → 6 Eksperymenty
// → 7 LTX Alpha-Gen (2026-10-16, out of the experiments' edge modes) → 8 Final Touch (2026-10-22, hand corrections,
// static/finaltouch.js) → 9 Eksport; keys lane on
// the timeline, overlays of the view tabs (Rdzeń / Alpha / Comp) on the main canvas. Stage 0 (Plate'y) lives in stage0.js.
// Stage numbers are never written by hand: STAGES (order = number) and ST.<key> – reorder STAGES to reorder the bar.
// Rules: only the tools of the current stage are visible; any earlier stage can be reopened; a later stage opens only
// when the backend (engine/stages.py, `steps`) says it is not locked. Stage 1: the points of a frame ARE its key.
(function () {
  // ------------------------------------------------------------------------------------------------ stages model
  // key = name in code (ST.key), id = engine/stages.py step id; desc = what it does / what it is best for (panel top)
  const STAGES = [
    { key: 'PLATES', id: 'plates', label: "Plate'y", hint: "Plate'y ujęcia, zgodność clear plate / backlit z beauty, czyszczenie, kamera, głębia" },
    { key: 'SAM3', id: 'sam3', label: 'SAM 3.1', hint: 'Wskaż obiekt (punkty / prompt), „▶▶ Przelicz całość” = maska w każdej klatce; poprawiaj klucze SAM 3.1 i przeliczaj',
      desc: '<b>Co robi:</b> SAM 3.1 znajduje obiekt z punktów / promptu w klatkach kluczowych i propaguje maskę (rdzeń, ostra krawędź) na całe ujęcie; kadry pod obiekt liczą się same. <b>Najlepsze do:</b> wyznaczenia CO jest obiektem – sylwetka, części, zasłonięcia. Złą klatkę poprawiasz tu kluczem SAM 3.1 (punkty, Korekta + / −, pędzel / wielokąt klucza).' },
    { key: 'BRUSH', id: 'cotracker', label: 'CoTracker3', hint: 'Pędzel poprawek maski (dodaj / usuń) w klatkach kluczowych, propagowany punktami CoTracker3',
      desc: '<b>Co robi:</b> poprawki maski pędzlem (dodaj / usuń) w kilku klatkach – CoTracker3 przenosi je punktami na resztę ujęcia (bez SAM 3.1). Maska 1+2 = rdzeń SAM 3.1 ± poprawki. <b>Najlepsze do:</b> stałych błędów SAM 3.1, których kluczami nie da się dobić (cienki fragment, dziura, wyciek na tło), na obiektach, które się przesuwają / odkształcają. Etap opcjonalny; zmiany SAM 3.1 = powrót do etapu 1.' },
    { key: 'MATANYONE', id: 'matanyone', label: 'MatAnyone2', hint: 'Miękka krawędź MatAnyone2 z maski 1+2 (bez stref ViTMatte)',
      desc: '<b>Co robi:</b> MatAnyone2 liczy miękką krawędź alfy w paśmie wokół maski 1+2 (rdzeń SAM 3.1 + poprawki pędzlem), w czasie, bez stref ViTMatte – to wersja bazowa. Obiekt połączony z częściami (warstwy) liczy się raz jako całość ze szwem; części dostają swój udział w eksporcie. <b>Najlepsze do:</b> naturalnej krawędzi postaci i obiektów (rozmycie ruchu, nieostrość, miękkie kontury); MatAnyone2 nie dodaje ani nie usuwa większych fragmentów – tylko krawędź.' },
    { key: 'VIDEOMAMA', id: 'videomama', label: 'VideoMaMa', hint: 'VideoMaMa: kosmyki włosów i detal krawędzi z maski 1+2, nałożone na bazę (maska 1+2 albo wynik MatAnyone2)',
      desc: '<b>Co robi:</b> model VideoMaMa (na priorze Stable Video Diffusion) liczy alfę całego obiektu z obrazu i maski 1+2, w paczkach klatek z przenikaniem. Wynik nakłada się na bazę: maskę 1+2 (domyślnie – VideoMaMa robi wtedy całą krawędź) albo wynik MatAnyone2 (twarda krawędź MatAnyone2 zostaje w pasie ochrony); dalej niż „zasięg” od krawędzi bazy nic nie dochodzi (wysepki w tle odcięte). Suwaki działają na żywo w podglądzie, bez liczenia modelu; „👁 Podgląd klatek” i „▶ Policz” biorą ich bieżące wartości. <b>Najlepsze do:</b> włosów, kosmyków, futra – pojedyncze odstające włosy, których MatAnyone2 nie rozdziela. Wynik może być bazą ViTMatte i źródłem eksportu.' },
    { key: 'VITMATTE', id: 'vitmatte', label: 'ViTMatte', hint: 'Strefy ViTMatte (malowany pas / cały obrys) na wyniku MatAnyone2',
      desc: '<b>Co robi:</b> ViTMatte liczy półprzezroczystą alfę tylko w strefie – pasie wokół całego obrysu, który jedzie za obiektem; grubość pasa ustawiasz punktami pasa (wąsy: na zewnątrz / do środka, 0 = przerwa, np. skóra); wynik nakłada się na bazę do wyboru: wersję MatAnyone2 z etapu 3 albo maskę 1+2 (wtedy etap działa zaraz po etapach 1 + 2). Grupę (obiekt + części) liczy obiekt główny jako całość ze szwem. Porównanie A / B / różnica – niżej. <b>Najlepsze do:</b> włosów, futra, kosmyków, dymu, włókien – detali, których MatAnyone2 nie rozdziela od tła. <b>Silnik</b> (strzałka ▾ przy etapie, gdy zainstalowane są oba): ViTMatte – tylko niekomercyjnie; BiRefNet_HR-matting / Lucida – wagi MIT, także komercyjnie.' },
    { key: 'EXPERIMENTS', id: 'final', label: 'Eksperymenty', hint: 'Tryby krawędzi, strefy progowe, poprawka krawędzi, wersje, QA, mapa migotania',
      desc: '<b>Co robi:</b> najpierw wybierasz <b>maskę bazową</b> (wynik MatAnyone2 / ViTMatte / LTX-2.5, maska 1+2 albo dowolna wersja); QA i propozycje poprawki krawędzi liczą się na niej, a „Przelicz z tej bazy” robi z niej nową wersję (silnik krawędzi MatAnyone2 / backlit / twarda, strefy progowe „Zaawansowane”, opcjonalnie poprawka krawędzi bazy) z zapisanym pochodzeniem. <b>Najlepsze do:</b> eksperymentów – np. wynik ViTMatte jeszcze raz przez MatAnyone2, backlit, obiekty sztywne, garbage matte progiem.' },
    { key: 'LTX', id: 'ltx', label: 'LTX-2.5', hint: 'LTX-2.5 Alpha-Gen: generatywna alfa z samego obrazu, prowadzona maską 1+2 (inne obiekty zamalowane)',
      desc: '<b>Co robi:</b> model LTX-2.5 (22B) z IC-LoRA Alpha-Gen sam generuje alfę z kadru obrazu – bez maski i promptu; model sam wybiera pierwszy plan, a maska 1+2 przycina wynik do regionu obiektu (obiekty z przodu odejmuje eksport; „Izoluj” zamalowuje inne obiekty stojące OBOK). Paczki 8k+1 klatek z przenikaniem. <b>Najlepsze do:</b> półprzezroczystych i cienkich rzeczy – włosy pod światło, dym, tkaniny, rozmycie ruchu – gdy MatAnyone2 / ViTMatte gubią detal. Generatywny: sprawdź wynik (A / B z wersją MatAnyone2). Wymaga modeli LTX-2.5 (~40 GB, Doktor).' },
    { key: 'FINAL_TOUCH', id: 'final_touch', label: 'Final Touch', hint: 'Ręczne poprawki alfy pędzlem, klatka po klatce, na wybranej wersji – przed eksportem',
      desc: '<b>Co robi:</b> ręczne, poklatkowe poprawki alfy pędzlem bitmapowym (Dodaj = biel, Usuń = czerń, Zmaż = powrót do bazy; miękkość wg twardości) na <b>bazie</b> – dowolnej policzonej wersji (domyślnie wynik ostatniego etapu). Korekty są warstwą obiektu: zmiana bazy w każdej chwili, korekty zostają. Eksport domyślnie bierze bazę + korekty. Bez śledzenia – każda klatka osobno. <b>Najlepsze do:</b> ostatnich poprawek, które dziś robisz poza programem – pojedyncze złe klatki, palce, kosmyki, dziury, wycieki.' },
    { key: 'EXPORT', id: 'export', label: 'Eksport', hint: 'Maski do Nuke’a (EXR / PNG / multi-EXR) z wybranego źródła, biblioteka, kolejka',
      desc: '<b>Co robi:</b> zapisuje SAMĄ alfę (nigdy RGB) pełnych klatek do Nuke’a z wybranego źródła: rdzeń (1+2), MatAnyone2, VideoMaMa, ViTMatte, wersja z listy, LTX-2.5 albo Final Touch (baza + ręczne korekty – domyślne, gdy obiekt ma korekty); obiekty z przodu odejmowane od tych z tyłu, części składane w &lt;obiekt&gt;_full. <b>Dostępny</b> od razu po rdzeniu z etapu 1.' },
  ];
  STAGES.forEach((s, i) => { s.n = i; s.d = i; });
  const ST = Object.fromEntries(STAGES.map((s) => [s.key, s.n]));   // ST.SAM3, ST.BRUSH, … – never literal numbers
  const S = {
    objects: [], obj: null, stage: ST.SAM3, info: null, view: 'none',
    overlay: {}, compCache: {}, preview: {}, points: {}, pointsBusy: {}, pointsDirty: {}, pointsFailed: {},
    tool: 'none', radius: 12, strokes: {}, poly: null, polys: {}, drawing: null, keys: [], qa: null, crops: null,
    bg: 0, pickColor: '#00b140', auxMode: false, lastObjStage: ST.SAM3, clickStart: null,
    penSeen: false, penPressure: false, penGamma: 1.0, hover: null, resizing: null,
  };
  const $ = (id) => document.getElementById(id);
  const el = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstChild; };
  const V = () => window.viewer;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const det = d.detail;
      const e = new Error(det && typeof det === 'object' ? (det.detail || JSON.stringify(det)) : (det || r.statusText));
      e.status = r.status; e.detail = det; throw e;
    }
    return d;
  }
  const toast = (t, ms) => V() && V().showToast(t, ms);
  function status(t, kind) {
    const box = $(S.stage === ST.SAM3 ? 'st1Status2' : `st${S.stage}Status`);   // st1Status = the object line
    if (box) { box.textContent = t; box.className = `stage-status ${kind || ''}`; }
  }

  // stage bar states (user request 2026-10-16: computed / not computed / needs updating must be obvious at a glance)
  const ICON = { done: '✔', stale: '↻', todo: '○', locked: '○', info: '·' };
  const STATE_WORD = { done: 'policzone', stale: 'do przeliczenia', todo: 'jeszcze nie policzone', locked: 'niedostępne' };
  const PANES = {};        // n -> { el, show(), hide() }
  // fine-grained stage (detect / crops / core / cotracker / matanyone / vitmatte / final / export) or a bar step (sam3 …)
  const stageState = (id) => (S.info && ((S.info.stages || []).find((s) => s.id === id) || (S.info.steps || []).find((s) => s.id === id)))
    || { state: 'locked', why: '' };
  const stepState = (st) => (S.info && (S.info.steps || []).find((s) => s.id === st.id)) || { state: 'locked', why: '' };
  const stageOfId = (id) => {   // bar stage showing the fine-grained stage `id` (e.g. "core" -> SAM3)
    const step = S.info && (S.info.steps || []).find((s) => s.id === id || s.stage === id);
    return STAGES.find((x) => x.id === (step ? step.id : id)) || STAGES[ST.SAM3];
  };
  // stages with the left draw tools; stage 5 only while zones.js allows drawing a zone area (off since 2026-10-21)
  const drawStage = () => [ST.SAM3, ST.BRUSH, ST.EXPERIMENTS, ST.FINAL_TOUCH].includes(S.stage)
    || (S.stage === ST.VITMATTE && !!(window.MD_ZONES && window.MD_ZONES.drawTools));
  // stage 5 without the drawing tools: the rail holds the band point tools (user decision 2026-10-22)
  const BAND_STAGES = () => [ST.MATANYONE, ST.VIDEOMAMA, ST.LTX];   // stage bands with points (static/bandpoints.js)
  const bandStage = () => (S.stage === ST.VITMATTE && !!(window.MD_ZONES && window.MD_ZONES.ready && !window.MD_ZONES.drawTools))
    || (BAND_STAGES().includes(S.stage) && !!(window.MD_BANDPTS && window.MD_BANDPTS.ready));   // one band system (2026-10-22)
  const bandMod = () => (BAND_STAGES().includes(S.stage) ? window.MD_BANDPTS : window.MD_ZONES);
  const versionStage = () => [ST.MATANYONE, ST.VIDEOMAMA, ST.VITMATTE, ST.EXPERIMENTS, ST.LTX].includes(S.stage);    // stages showing versions (B, viewRun)

  // stage 5 engine (user decision 2026-10-21): ViTMatte by default; BiRefNet when only it is installed; an arrow menu
  // in the stage bar when both are (GET stages: edge_engines / edge_engine, PUT /api/object/{o}/edge_engine)
  const EDGE_ENGINE_LABEL = { vitmatte: 'ViTMatte', birefnet: 'BiRefNet' };
  const edgeEngine = () => (S.info && S.info.edge_engine) || 'vitmatte';
  const bothEdgeEngines = () => !!(S.info && S.info.edge_engines && S.info.edge_engines.vitmatte && S.info.edge_engines.birefnet);
  const stageLabel = (st) => (st.n === ST.VITMATTE && S.obj ? EDGE_ENGINE_LABEL[edgeEngine()] : st.label);
  function openEdgeEngineMenu(anchor) {
    document.querySelectorAll('.stage-engine-menu').forEach((m) => m.remove());
    const cur = edgeEngine();
    const menu = el(`<div class="stage-engine-menu">
      ${[['vitmatte', 'ViTMatte', 'tylko niekomercyjnie'], ['birefnet', 'BiRefNet', 'wagi MIT, komercyjnie']].map(([k, l, lic]) =>
        `<button type="button" data-e="${k}" class="${k === cur ? 'active' : ''}">${k === cur ? '✔ ' : ''}${l} <span class="muted">– ${lic}</span></button>`).join('')}
    </div>`);
    const r = anchor.getBoundingClientRect();
    menu.style.left = `${Math.round(r.left)}px`;
    menu.style.top = `${Math.round(r.bottom + 4)}px`;
    document.body.appendChild(menu);
    const close = (e) => { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener('mousedown', close, true); } };
    setTimeout(() => document.addEventListener('mousedown', close, true), 0);
    menu.querySelectorAll('button').forEach((b) => b.addEventListener('click', async () => {
      menu.remove(); document.removeEventListener('mousedown', close, true);
      if (b.dataset.e === cur) return;
      try {
        const d = await api('PUT', `/api/object/${S.obj}/edge_engine`, { engine: b.dataset.e });
        S.vmPick = null;   // the viewer follows the new stage engine (HUD ⇄ picks the other one)
        toast(`Silnik etapu ${ST.VITMATTE}: ${EDGE_ENGINE_LABEL[d.edge_engine]}${d.zones_changed.length ? ` – strefy do przeliczenia: ${d.zones_changed.join(', ')}` : ''}`, 5000);
        await loadStages();
        if (window.MD_ZONES && window.MD_ZONES.ready) await window.MD_ZONES.reload();
      } catch (e) { toast(`Nie udało się zmienić silnika: ${e.message || e}`, 6000); }
    }));
  }

  function renderStageBar() {
    const list = $('stageList'); if (!list) return;
    list.innerHTML = '';
    for (const st of STAGES) {
      let state = 'info', why = st.hint;
      if (st.n !== ST.PLATES) {
        if (!S.obj) { state = st.n === ST.SAM3 ? 'todo' : 'locked'; why = st.n === ST.SAM3 ? 'Dodaj albo wybierz obiekt' : 'Najpierw dodaj / wybierz obiekt'; }
        else { const x = stepState(st); state = x.state; why = x.why || st.hint; if (x.scope === 'range') state = 'done range'; }
        const pb = window.MD.plateBlock && window.MD.plateBlock();   // Z26: plate QA undecided -> object stages locked
        if (pb && state !== 'done') { state = 'locked'; why = pb; }
      } else if (window.MD.stage0State) state = window.MD.stage0State();
      const disabled = S.auxMode && st.n !== ST.PLATES;
      const edgeMenu = st.n === ST.VITMATTE && S.obj && bothEdgeEngines();
      const li = el(`<li class="stage-item ${state} ${S.stage === st.n ? 'current' : ''} ${disabled || state === 'locked' ? 'disabled' : ''}" data-n="${st.n}"
        title="${esc(disabled ? 'Narzędzia obiektów działają na beauty – przełącz plate na beauty' : (STATE_WORD[state.split(" ")[0]] && st.n !== ST.PLATES ? `${window.mdT(STATE_WORD[state.split(" ")[0]])}${state === 'done range' ? ' (część ujęcia)' : ''} – ` : '') + why)}">
        <span class="stage-n">${st.d}</span><span class="stage-ic">${state === 'done range' ? '◐' : (ICON[state] || '')}</span><span class="stage-label">${stageLabel(st)}</span>${edgeMenu ? '<span class="stage-menu" title="Silnik etapu: ViTMatte (tylko niekomercyjnie) albo BiRefNet (wagi MIT, komercyjnie)">▼</span>' : ''}</li>`);
      li.addEventListener('click', () => { if (!disabled && state !== 'locked') setStage(st.n); else toast(li.title); });
      if (edgeMenu) li.querySelector('.stage-menu').addEventListener('click', (e) => { e.stopPropagation(); openEdgeEngineMenu(li); });
      list.appendChild(li);
    }
    const hint = $('stageHint');
    if (hint) {
      const cur = STAGES[S.stage];
      let t = cur.hint;
      if (S.stage !== ST.PLATES && S.info) {
        const x = stepState(cur);
        if (x.state === 'done' && S.info.next && stageOfId(S.info.next).n !== S.stage) t = `✔ gotowe · następny krok: ${stageOfId(S.info.next).n}. ${S.info.next_label}`;
        else if (x.state === 'stale') t = `↻ ${x.why}`;
        else if ((x.state === 'todo' || x.state === 'done') && x.why) t = x.why;
      }
      hint.textContent = t;
    }
    // "Dalej" buttons of the panes
    for (const st of STAGES) {
      const b = $(`st${st.n}Next`); if (!b) continue;
      const nx = STAGES[st.n + 1];
      if (!nx) { b.style.display = 'none'; continue; }
      const pb = st.n === ST.PLATES && window.MD.plateBlock && window.MD.plateBlock();
      const ns = pb ? 'locked' : (st.n === ST.PLATES ? (S.obj ? stepState(nx).state : 'todo') : stepState(nx).state);
      b.disabled = S.auxMode || ns === 'locked';
      b.textContent = `Dalej → ${nx.d}. ${nx.label}`;
      b.title = b.disabled ? (pb || stepState(nx).why || 'najpierw ten etap') : `Przejdź do etapu ${nx.d}. ${nx.label}: ${nx.hint}`;
    }
  }

  // default view of a stage before the user picks a tab
  const STAGE_VIEW = { SAM3: 'core', BRUSH: 'core', MATANYONE: 'alpha', VIDEOMAMA: 'alpha', VITMATTE: 'alpha', EXPERIMENTS: 'alpha', LTX: 'alpha', FINAL_TOUCH: 'alpha', EXPORT: 'comp' };
  function setStage(n, opts = {}) {
    if (!STAGES[n]) return;
    if (S.auxMode && n !== ST.PLATES) n = ST.PLATES;
    const prevStage = S.stage;
    S.stage = n;
    if (n !== ST.PLATES) S.lastObjStage = n;
    if (CLICK_TOOLS[S.tool] && CLICK_TOOLS[S.tool] !== n) S.tool = 'none';
    if (S.tool === 'brush_erase' && n !== ST.BRUSH && n !== ST.FINAL_TOUCH) S.tool = 'none';
    if (String(S.tool).startsWith('band_') && ![ST.VITMATTE, ...BAND_STAGES()].includes(n)) S.tool = 'none';
    for (const [k, p] of Object.entries(PANES)) {
      const on = +k === n;
      p.el.style.display = on ? '' : 'none';
      if (on && p.show) p.show();
    }
    $('stagePanelEmpty').style.display = V() && V().projectLoaded ? 'none' : '';
    renderSegsLane();
    if (drawStage()) loadStageRadius();
    updateDrawTools();
    if (!opts.keepView) {   // the tab the user picked (Plate / Core / Alpha / Comp) rules in every stage (user decision
      // 2026-10-22); the stage default only before any pick, or when the picked view has nothing to show here
      const def = STAGE_VIEW[STAGES[n].key] || 'none';
      const want = S.userView || def;
      setView(viewAllowed(want) ? want : (viewAllowed(def) ? def : 'none'), false, false);
    }
    renderStageBar();
    if (n !== prevStage) {   // the shared object band (2026-10-22): an edit made in another stage shows here
      if (window.MD_BANDPTS && window.MD_BANDPTS.ready && BAND_STAGES().includes(n)) { window.MD_BANDPTS.dirty(); window.MD_BANDPTS.renderMode(); }
      if (n === ST.VITMATTE && prevStage != null) loadZones();
    }
    updateViewTabs();   // Alpha / Comp availability depends on the stage (its own mask, 2026-10-22)
    renderPointsPalette();
    renderEdgePalette();
    renderMbPalette();
    renderLtxPalette();
    renderVmmPalette();
    window.dispatchEvent(new CustomEvent('md-stage', { detail: { stage: n } }));   // stage0.js: its floating palette
    renderExpPalette();
    if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.onStage();
    if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.onStage();
    if (window.MD_FT && window.MD_FT.ready) window.MD_FT.onStage();
    V() && V().render();
  }

  // ------------------------------------------------------------------------------------------------ view tabs
  function viewAllowed(v) {
    if (v === 'none') return true;
    if (S.auxMode || !S.obj || !S.info) return false;
    const has = S.info.has || {};
    const coreOk = !!has.core || ['done', 'stale'].includes(stageState('core').state);
    if (v === 'core') return coreOk;
    // stages with their own mask show it in Alpha / Comp – the core / mask 1+2 until the stage is computed (2026-10-22)
    if (coreOk && [ST.SAM3, ST.BRUSH, ST.MATANYONE, ST.VIDEOMAMA, ST.VITMATTE, ST.LTX, ST.FINAL_TOUCH].includes(S.stage)) return true;
    // a stage result that is not the active version (LTX, stage 6) or a version picked for viewing also has an alpha
    const sr = S.info.stage_runs || {};
    return !!has.final || ['done', 'stale'].includes(stageState('final').state) || !!S.viewRun || !!sr.ltx;
  }
  function updateViewTabs() {
    document.querySelectorAll('#viewModeTabs .tab-btn').forEach((b) => {
      const ok = viewAllowed(b.dataset.view);
      b.disabled = !ok;
      b.classList.toggle('disabled', !ok);
      if (!ok && b.dataset.view !== 'none') b.title = S.auxMode ? 'Widoki wyniku tylko na beauty' : (!S.obj ? 'Wybierz obiekt' :
        (b.dataset.view === 'core' ? `Rdzeń jeszcze nie policzony – „▶▶ Przelicz całość” (etap ${ST.SAM3})`
          : `Brak wyniku z krawędzią – „▶ Przelicz MatAnyone2” (etap ${ST.MATANYONE}) albo „Przelicz finalnie” (etap ${ST.EXPERIMENTS})`));
      else if (ok && b.dataset.view !== 'none') {
        const stale = b.dataset.view === 'core' ? stageState('core').state !== 'done' : stageState('final').state !== 'done';
        b.title = stale ? 'Wynik nieaktualny (klucze / ustawienia zmienione po liczeniu) – do przeliczenia' : '';
        b.classList.toggle('stale', stale);
      }
    });
    if (!viewAllowed(S.view)) setView('none', false, false);
  }
  // remember = the layer the user chose (tabs, V, stage change) – kept when switching objects (bug 2026-10-15: switching
  // the object reset the view to Plate, the mask "vanished"); automatic fallbacks (no result yet) do not overwrite it
  function setView(v, render = true, remember = true) {
    S.view = viewAllowed(v) ? v : 'none';
    if (remember && S.view === v) S.userView = v;
    document.querySelectorAll('#viewModeTabs .tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.view === S.view));
    if (V()) { preloadOverlays(V().currentFrame, 0); renderLoadBar(); }   // Z20: the bar / sweep follow the layer of the view
    if (render && V()) V().render();
  }

  // ------------------------------------------------------------------------------------------------ overlays
  // Z20: S.overlay = key -> Image | 'missing'. After a recompute the frame's overlays are only marked STALE (shown until
  // the new image arrives: S.overlayNext holds the replacement load); 'missing' entries are dropped so they are asked again.
  const OVERLAY_LIMIT = 600;    // proxy-size mask PNGs kept in memory (a whole shot of a few layers)
  const SWEEP_PAR = 4;          // background sweep: at most 4 overlay requests in flight
  // derived overlay canvases (tint / black-white / band edges …): the newest COMP_KEEP are kept. The old form put the new
  // entry first and kept the 12 OLDEST, so the overlays of the frame on screen were dropped and recomputed (a CPU
  // pixel loop, ~25 ms each) on every render – stage 4 drew in ~46 ms per render (measured 2026-10-23)
  const COMP_KEEP = 16;
  function putComp(ck, c) {
    delete S.compCache[ck];
    S.compCache = { ...Object.fromEntries(Object.entries(S.compCache).slice(-(COMP_KEEP - 1))), [ck]: c };
  }
  function clearOverlays() { S.overlay = {}; S.compCache = {}; S.overlayNext = {}; renderLoadBar(); }
  function invalidateFrame(f) {
    for (const k of Object.keys(S.overlay)) {
      if (!k.endsWith(`_${f}`)) continue;
      const im = S.overlay[k];
      if (im && im !== 'missing' && im.complete && im.naturalWidth > 0) im.mdStale = true; else delete S.overlay[k];
    }
    for (const k of Object.keys(S.overlayNext || {})) if (k.endsWith(`_${f}`)) delete S.overlayNext[k];
    for (const k of Object.keys(S.compCache)) if (k.includes(`_${f}_`)) delete S.compCache[k];
    renderLoadBar();
  }
  // front objects (smaller depth_order) without a core: nothing was subtracted from this object there (2026-10-23)
  function frontMissingText(d) {
    return (d.front_missing || []).map((m) => `\n⚠ ${m.message}`).join('');
  }
  function frontMissingKind(d) { return (d.front_missing || []).length ? 'warn' : 'ok'; }
  function invalidateIntervals(ivs) {   // [[a, b], …] from run / rerun_from: only the recomputed frames (Z20)
    for (const iv of ivs || []) { const [a, b] = iv; for (let f = a; f <= b; f++) invalidateFrame(f); }
  }
  const keyOf = (f) => S.keys.find((x) => x.frame === f);
  function startOverlay(key, url, replace) {
    const im = new Image();
    const gen = S.projectGen || 0;   // an overlay of the previous project (same object name) must not land in this one
    const done = () => { S.loading = Math.max(0, (S.loading || 0) - 1); sweepTick(); renderLoadBar(); V().render(); };
    im.onload = () => {
      if (gen !== (S.projectGen || 0)) { done(); return; }
      if (replace) { S.overlay[key] = im; delete S.overlayNext[key]; const f = key.slice(key.lastIndexOf('_') + 1); for (const k of Object.keys(S.compCache)) if (k.includes(`_${f}_`)) delete S.compCache[k]; }
      done();
    };
    im.onerror = () => { if (gen !== (S.projectGen || 0)) { done(); return; } S.overlay[key] = 'missing'; if (replace) delete S.overlayNext[key]; done(); };
    S.loading = (S.loading || 0) + 1;
    im.src = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
    return im;
  }
  function evictOverlays() {
    const ks = Object.keys(S.overlay).filter((k) => !k.includes('_tile_'));
    if (ks.length <= OVERLAY_LIMIT) return;
    const cur = `_${V() ? V().currentFrame : -1}`;
    for (const k of ks.slice(0, ks.length - OVERLAY_LIMIT)) if (!k.endsWith(cur)) delete S.overlay[k];
  }
  // Image (complete) | 'missing' | null (loading); a stale image is returned while its replacement loads
  function loadOverlay(key, url) {
    const cur = S.overlay[key];
    if (cur === 'missing') return 'missing';
    if (cur && !cur.mdStale) return cur.complete && cur.naturalWidth > 0 ? cur : null;
    S.overlayNext = S.overlayNext || {};
    if (cur) { if (!S.overlayNext[key]) S.overlayNext[key] = startOverlay(key, url, true); return cur; }
    S.overlay[key] = startOverlay(key, url, false);
    evictOverlays();
    return null;
  }

  // Live server previews (edge band, VideoMaMa, LTX, motion blur) take seconds per frame and the browser runs 6 requests
  // at a time, so scrubbing queued a request for every frame passed and the frame you stop on waited behind them – the
  // preview "disappeared" (bug 2026-10-17). A frame is requested only once the playhead rests on it (~0.2 s).
  function settledOverlay(key, url, f) {
    const now = performance.now();
    if (S.pvFrame !== f) { S.pvFrame = f; S.pvFrameT = now; }
    if (S.overlay[key] === undefined && now - S.pvFrameT < 200) {
      clearTimeout(S.pvSettleT); S.pvSettleT = setTimeout(() => V().render(), 220);
      return null;
    }
    return loadOverlay(key, url);
  }
  // A band preview still loading: the last band of the same layer and settings stays on screen, dimmed (not for alpha
  // previews – an alpha of another frame would sit in the wrong place). Returns true when something was drawn.
  function drawBand(ctx, img, layer, q, ov, key) {
    S.pvLast = S.pvLast || {};
    if (ov && ov !== 'missing') {
      const c = bandEdges(ov, key);
      if (!c) return false;
      ctx.drawImage(c, 0, 0, img.width, img.height);
      S.pvLast[layer] = { obj: S.obj, q, c };
      return true;
    }
    const l = S.pvLast[layer];
    if (!ov && l && l.obj === S.obj && (l.q === q || S.pvDrag === layer)) {   // a slider held: the last lines of the layer
      ctx.save(); ctx.globalAlpha = 0.45; ctx.drawImage(l.c, 0, 0, img.width, img.height); ctx.restore();
      return true;
    }
    return false;
  }

  // Stage 3 local correction: a click changes ONLY the hole (+) / leak (−) under it; the key is saved at once.
  async function correctAt(f, p, mode) {
    if (S.corrBusy) { toast('Poprzednia korekta jeszcze się liczy – chwila'); return; }
    S.corrBusy = true; S.corrMark = { f, p, mode }; V().render();
    status(mode === 'add' ? `Korekta F${f}: dodaję brakujący fragment…` : `Korekta F${f}: usuwam wyciek…`);
    try {
      const d = await api('POST', `/api/object/${S.obj}/correct/${f}`, { mode, point: p });
      if (!d.changed_px) { status(d.detail, 'warn'); toast(d.detail, 5000); }
      else {
        invalidateFrame(f);
        status(`Klucz F${f} (korekta): ${mode === 'add' ? 'dodano' : 'usunięto'} ${d.changed_px} px (SAM 3.1 ${d.seconds}s), reszta maski bez zmian. ` +
               'Ctrl+Z cofa. Potem „Przelicz zmienione przedziały”.', 'ok');
        await loadKeys();
      }
      await loadStages();
    } catch (e) {
      const m = `Korekta F${f} NIE zapisana: ${e.message}`; status(m, 'err'); toast(m, 6000);
    } finally { S.corrBusy = false; S.corrMark = null; V().render(); }
  }

  // Stage 3 brush / polygon: the key is saved automatically (no "Zapisz klucz" needed)
  function scheduleBrushKey(f) {
    clearTimeout(S.brushTimer);
    S.brushPending = f;
    S.brushTimer = setTimeout(() => saveBrushKey(f).catch(() => {}), 600);
    status(`Klatka ${f}: poprawka pędzlem / wielokątem – zapisuję klucz…`);
  }
  async function saveBrushKey(f) {
    clearTimeout(S.brushTimer); S.brushPending = null;
    const strokes = S.strokes[f] || [], polyObjs = S.polys[f] || [], polys = polyObjs.map((q) => ({ mode: q.mode, p: q.p }));
    if (!strokes.length && !polys.length) return;
    S.strokes[f] = []; S.polys[f] = [];
    try {
      const d = await api('POST', `/api/object/${S.obj}/keys/${f}`, { source: 'core', strokes, polygons: polys });
      invalidateFrame(f);
      status(`Klucz F${f} zapisany (${d.key.kind}) – poprawka na obecnej masce. Ctrl+Z cofa. Potem „Przelicz zmienione przedziały”.`, 'ok');
      await loadKeys(); await loadStages();
    } catch (e) {
      S.strokes[f] = strokes.concat(S.strokes[f] || []); S.polys[f] = polyObjs.concat(S.polys[f] || []);
      const m = `Poprawka F${f} NIE zapisana jako klucz: ${e.message}`; status(m, 'err'); toast(m, 6000);
      throw e;
    } finally { V().render(); }
  }
  // stage Eksport: the alpha overlay is the matte as exported (objects in front subtracted, parts split) – plan step 7
  const exportCtx = () => (S.stage === ST.EXPORT && !S.viewRun ? '&export=1' : '');
  // the core layer of frame f: stage 2 shows the mask 1+2 (core_fix, „Zastosuj poprawki”) when it exists; elsewhere a
  // key frame shows its key (the result IS the key there), other frames the SAM3 core
  const coreKind = (f) => (S.stage === ST.BRUSH && S.info && S.info.has && S.info.has.core_fix ? 'core_fix' : (keyOf(f) ? 'key' : 'core'));
  const overlayKey = (kind, f) => {
    const ver = kind === 'alpha' && S.viewRun ? S.viewRun : '';     // Z2: preview of another version (A/B, key B)
    const ex = kind === 'alpha' ? exportCtx() : '';
    return `${S.obj}_${kind}${ver ? '@' + ver : ''}${ex ? '#x' : ''}_${f}`;
  };
  function overlayFor(kind, f) {
    // kind: core | alpha | key | preview | zone_* … Returns Image (complete), 'missing', or null (loading)
    const ver = kind === 'alpha' && S.viewRun ? S.viewRun : '';
    const ex = kind === 'alpha' ? exportCtx() : '';
    return loadOverlay(overlayKey(kind, f), `/api/object/${S.obj}/frame/${f}?view=${kind}${ver ? '&v=' + ver : ''}${ex}`);
  }
  // ---- Z20: overlays ahead of the playhead + background sweep of the whole shot + load bar over the timeline
  function neededLayer() {   // the overlay layer the current view draws (null = none)
    if (!S.obj || S.auxMode) return null;
    if (S.view === 'alpha' || S.view === 'comp') return 'alpha';
    if (S.view === 'core') return 'core';
    if (S.view === 'none' && S.stage === ST.SAM3) return 'key';
    return null;
  }
  function neededKind(f) {   // the overlay kind drawn in frame f for the current view (null = nothing to load)
    const l = neededLayer(); if (!l) return null;
    if (l === 'core') return coreKind(f);
    if (l === 'key') return keyOf(f) ? 'key' : null;
    return 'alpha';
  }
  function overlayState(f) {   // 'ready' | 'loading' | 'missing' | 'none' (no overlay needed in this frame)
    const kind = neededKind(f); if (!kind) return 'none';
    const cur = S.overlay[overlayKey(kind, f)];
    if (cur === 'missing') return 'missing';
    if (cur && cur.complete && cur.naturalWidth > 0 && !cur.mdStale) return 'ready';
    return 'loading';
  }
  function ensureOverlay(f) { const kind = neededKind(f); if (kind) overlayFor(kind, f); }
  function preloadOverlays(center, dir) {   // same range as the plate preload (±6, ±15 when playing, forward only while playing)
    const v = V(); if (!v || (!neededLayer() && !otherObjs().length)) return;
    const range = v.isPlaying ? 15 : 6;
    ensureOthers(center);
    for (let o = 1; o <= range; o++) {
      if (center + o <= v.lastFrame) { ensureOverlay(center + o); ensureOthers(center + o); }
      if (center - o >= v.firstFrame && !(v.isPlaying && dir > 0)) { ensureOverlay(center - o); ensureOthers(center - o); }
    }
    sweepTick();
  }
  function sweepTick() {   // whole shot in the background, SWEEP_PAR requests at a time; stops by itself when the view / object changes
    const v = V(); if (!v || !S.obj || !neededLayer() || !v.projectLoaded) return;
    if (document.body.classList.contains('job-busy')) return;   // a running job may be rewriting these masks
    let inflight = S.loading || 0;
    const cur = v.currentFrame;
    for (let o = 0; o <= v.lastFrame - v.firstFrame && inflight < SWEEP_PAR; o++) {   // outward from the playhead
      for (const f of (o ? [cur + o, cur - o] : [cur])) {
        if (f < v.firstFrame || f > v.lastFrame || inflight >= SWEEP_PAR) continue;
        const kind = neededKind(f); if (!kind) continue;
        const cur_ = S.overlay[overlayKey(kind, f)];
        if (cur_ && (!cur_.mdStale || (S.overlayNext && S.overlayNext[overlayKey(kind, f)]))) continue;   // loaded / loading / missing / replacing
        overlayFor(kind, f); inflight++;
      }
    }
  }
  function frameReady(f) {   // Play (Z20): wait for the plate AND the overlay of the view (like Nuke); missing = nothing to wait for
    const v = V(); if (!v || !v.hasFrame(f)) { ensureOverlay(f); return false; }
    const st = overlayState(f);
    if (st === 'loading') { ensureOverlay(f); return false; }
    return true;
  }
  function renderLoadBar() {
    const c = $('timelineLoad'); if (!c || S._lbRaf) return;
    S._lbRaf = requestAnimationFrame(() => { S._lbRaf = 0; drawLoadBar(c); });
  }
  function drawLoadBar(c) {
    const v = V(); if (!v) return;
    const show = !!(v.projectLoaded && S.obj && !S.auxMode);
    c.style.display = show ? '' : 'none';
    if (!show) return;
    const W = (c.parentNode && c.parentNode.clientWidth) || c.width || 300;
    if (c.width !== W) c.width = W;
    const g = c.getContext('2d'); if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    const n = v.lastFrame - v.firstFrame + 1; if (n <= 0) return;
    let ready = 0, missing = 0;
    const cw = c.width / n;
    for (let i = 0; i < n; i++) {
      const f = v.firstFrame + i;
      const st = overlayState(f);
      let col = '#ef4444';                                    // red = not loaded yet
      if (st === 'missing') { col = '#55555b'; missing++; }   // grey = no result in this frame
      else if (v.hasFrame(f) && (st === 'ready' || st === 'none')) { col = '#22c55e'; ready++; }
      g.fillStyle = col; g.fillRect(Math.floor(i * cw), 0, Math.max(1, Math.ceil(cw)), c.height);
    }
    const layer = neededLayer();
    c.title = `Wczytane ${ready} / ${n} klatek (plate${layer ? ` + nakładka „${{ core: 'Rdzeń', alpha: 'Alpha', key: 'Klucz' }[layer]}”` : ''})` +
      `${missing ? ` · ${missing} bez wyniku (szare)` : ''} · czerwone = jeszcze nie wczytane; Play czeka na gotową klatkę`;
  }
  // Z8 / plan step 7: full-resolution crop of a mask (core | alpha | key | zone_alpha) for one tile chunk {x, y, w, h} of
  // the plate tile region – GET /api/object/{o}/frame/{f}/tile. Same cache / invalidation as overlayFor (key ends with _f).
  const TILE_OVERLAYS_MAX = 24;
  function overlayTile(kind, f, c) {
    const ver = kind === 'alpha' && S.viewRun ? S.viewRun : '';
    const ex = kind === 'alpha' ? exportCtx() : '';
    const key = `${S.obj}_${kind}${ver ? '@' + ver : ''}${ex ? '#x' : ''}_tile_${c.x}_${c.y}_${c.w}_${c.h}_${f}`;
    if (!S.overlay[key]) {
      const tiles = Object.keys(S.overlay).filter((k) => k.includes('_tile_'));
      for (const k of tiles.slice(0, Math.max(0, tiles.length + 1 - TILE_OVERLAYS_MAX))) delete S.overlay[k];
    }
    return loadOverlay(key, `/api/object/${S.obj}/frame/${f}/tile?view=${kind}&x=${c.x}&y=${c.y}&w=${c.w}&h=${c.h}${ver ? '&v=' + ver : ''}${ex}`);
  }
  // tiles of a mask for the active plate-tile region: the list of {chunk, img} when EVERY chunk is loaded, else null
  // (the proxy overlay is drawn meanwhile; the loads are started here)
  function tileOverlays(kind, f) {
    const v = V(); const reg = v && v.tileRegion ? v.tileRegion() : null;
    if (!reg) return null;
    const out = [];
    for (const c of reg.chunks) {
      const t = overlayTile(kind, f, c);
      if (!t || t === 'missing') return null;
      out.push({ c, img: t });
    }
    return { reg, tiles: out };
  }
  // comp background (user decision 2026-10-22): the target background BG when the project has one, else green; V cycles
  const GREEN = '#00b140';
  const BGS = ['bg', GREEN, '#000000', '#ffffff', '#767676', 'checker', 'pick'];
  const BG_LABEL = { bg: 'BG (tło docelowe)', [GREEN]: 'zielone', '#000000': 'czarne', '#ffffff': 'białe', '#767676': 'szare 18 %', checker: 'szachownica' };
  const hasBgPlate = () => !!(V() && V()._bgPattern);
  const compBgName = () => (BGS[S.bg] === 'bg' && !hasBgPlate() ? GREEN : BGS[S.bg]);
  function nextCompBg() {
    let i = BGS[S.bg] === 'bg' && !hasBgPlate() ? BGS.indexOf(GREEN) : S.bg;
    do { i = (i + 1) % BGS.length; } while (BGS[i] === 'bg' && !hasBgPlate());
    S.bg = i;
    const b = BGS[i];
    return b === 'pick' ? S.pickColor : BG_LABEL[b] || b;
  }
  function compCanvas(img, ov, key, f) {
    let bg = compBgName(), bgImg = null;
    if (bg === 'bg') {   // the BG frame of this frame; green until it arrives (or when it cannot be read)
      const v = V();
      bgImg = v.getImage('bg', f, false);
      if (!bgImg) {
        if (!(v.frameErrors && v.frameErrors.has(v.frameKey('bg', f, false)))) v.loadFrame(f, () => { if (v.currentFrame === f) v.render(); }, 'bg', false);
        bg = GREEN;
      }
    }
    const ck = `${key}_${bgImg ? 'bg' : bg}_${S.pickColor}`;
    if (S.compCache[ck]) return S.compCache[ck];
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    const g = c.getContext('2d');
    if (bgImg) g.drawImage(bgImg, 0, 0, c.width, c.height);
    else if (bg === 'checker') { for (let y = 0; y < c.height; y += 16) for (let x = 0; x < c.width; x += 16) { g.fillStyle = ((x + y) / 16) % 2 ? '#999' : '#666'; g.fillRect(x, y, 16, 16); } }
    else { g.fillStyle = bg === 'pick' ? S.pickColor : bg; g.fillRect(0, 0, c.width, c.height); }
    const gc = window.mdGpu && mdGpu.comp(img, ov, c, c.width, c.height);   // same pixels on the GPU, else the loop below
    if (gc) { putComp(ck, gc); return gc; }
    const b = g.getImageData(0, 0, c.width, c.height);
    const t = document.createElement('canvas'); t.width = img.width; t.height = img.height;
    const tg = t.getContext('2d'); tg.drawImage(img, 0, 0); const p = tg.getImageData(0, 0, t.width, t.height);
    tg.clearRect(0, 0, t.width, t.height); tg.drawImage(ov, 0, 0, t.width, t.height); const a = tg.getImageData(0, 0, t.width, t.height);
    for (let i = 0; i < b.data.length; i += 4) { const al = a.data[i] / 255; for (let k = 0; k < 3; k++) b.data[i + k] = p.data[i + k] * al + b.data[i + k] * (1 - al); }
    g.putImageData(b, 0, 0);
    putComp(ck, c);
    return c;
  }
  // ---- overlay look per layer (core / alpha / key): colour, opacity, fill or outline; kept in localStorage
  const OVS_DEFAULT = { core: { c: '#50a0ff', o: 45, line: false }, alpha: { c: '#50ff78', o: 45, line: false }, key: { c: '#ff78ff', o: 45, line: false } };
  const OVS_LABEL = { core: 'Rdzeń', alpha: 'Alpha', key: 'Klucz' };
  S.ovs = (() => {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('md_overlay_style') || '{}') || {}; } catch (e) { saved = {}; }
    const out = {};
    for (const k of Object.keys(OVS_DEFAULT)) out[k] = { ...OVS_DEFAULT[k], ...(saved[k] || {}) };
    return out;
  })();
  function saveOvs() { try { localStorage.setItem('md_overlay_style', JSON.stringify(S.ovs)); } catch (e) { /* private mode */ } }
  const hexRgb = (h) => { const n = parseInt(String(h).replace('#', ''), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  function overlayLayer() {
    if (!S.obj || S.auxMode) return null;
    if (S.view === 'core') return 'core';
    if (S.view === 'none' && S.stage === ST.SAM3) return 'key';
    return null;   // comp has its own background (V); alpha is shown black / white
  }
  function syncOverlayStyleUi() {
    const box = $('overlayStyle'); if (!box) return;
    const layer = overlayLayer();
    if (layer === S._ovsUiLayer) return;
    S._ovsUiLayer = layer;
    box.classList.toggle('disabled', !layer);
    $('ovsLayer').textContent = layer ? OVS_LABEL[layer] : '—';
    ['ovsColor', 'ovsOpacity', 'ovsLine'].forEach((id) => { $(id).disabled = !layer; });
    if (!layer) { box.title = 'Brak nakładki w tym widoku (Rdzeń / Alpha albo etap 1 z kluczem)'; return; }
    const st = S.ovs[layer];
    box.title = `Wygląd nakładki: ${OVS_LABEL[layer]} – zapamiętany w przeglądarce`;
    $('ovsColor').value = st.c; $('ovsOpacity').value = st.o; $('ovsOpacityVal').textContent = `${st.o}%`;
    $('ovsLine').textContent = st.line ? '◻' : '▣';
    $('ovsLine').title = st.line ? 'Sam kontur (kliknij: wypełnienie)' : 'Wypełnienie (kliknij: sam kontur)';
  }
  function wireOverlayStyle() {
    if (!$('overlayStyle')) return;
    const cur = () => { const l = overlayLayer(); return l ? S.ovs[l] : null; };
    $('ovsColor').addEventListener('input', (e) => { const st = cur(); if (!st) return; st.c = e.target.value; saveOvs(); V().render(); });
    $('ovsOpacity').addEventListener('input', (e) => {
      const st = cur(); if (!st) return; st.o = +e.target.value; $('ovsOpacityVal').textContent = `${st.o}%`; saveOvs(); V().render();
    });
    $('ovsLine').addEventListener('click', () => { const st = cur(); if (!st) return; st.line = !st.line; saveOvs(); S._ovsUiLayer = undefined; syncOverlayStyleUi(); V().render(); });
  }
  // mask -> coloured canvas (full alpha; opacity is applied when drawing). line = outline 2 px (inner + outer edge pixel)
  function tintCanvas(ov, rgb, key, line = false) {
    const ck = `tint_${key}_${rgb.join()}_${line ? 'l' : 'f'}`;
    if (S.compCache[ck]) return S.compCache[ck];
    const gc = window.mdGpu && (line ? mdGpu.line(ov, rgb) : mdGpu.tint(ov, rgb));   // same pixels on the GPU, else the loop below
    if (gc) { putComp(ck, gc); return gc; }
    const c = document.createElement('canvas'); c.width = ov.naturalWidth || ov.width; c.height = ov.naturalHeight || ov.height;
    const g = c.getContext('2d'); g.drawImage(ov, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height); const p = d.data; const W = c.width, H = c.height;
    if (line) {
      const m = new Uint8Array(W * H);
      for (let i = 0, j = 0; i < p.length; i += 4, j++) m[j] = p[i] > 127 ? 1 : 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const j = y * W + x, v = m[j];
        const edge = (x > 0 && m[j - 1] !== v) || (x < W - 1 && m[j + 1] !== v) || (y > 0 && m[j - W] !== v) || (y < H - 1 && m[j + W] !== v);
        const i = j * 4; p[i] = rgb[0]; p[i + 1] = rgb[1]; p[i + 2] = rgb[2]; p[i + 3] = edge ? 255 : 0;
      }
    } else {
      for (let i = 0; i < p.length; i += 4) { const a = p[i]; p[i] = rgb[0]; p[i + 1] = rgb[1]; p[i + 2] = rgb[2]; p[i + 3] = a; }
    }
    g.putImageData(d, 0, 0);
    putComp(ck, c);
    return c;
  }
  // the Alpha view = the alpha channel itself, black / white (user decision 2026-10-22), opaque, no overlay colour
  function bwCanvas(ov, key) {
    const ck = `bw_${key}`;
    if (S.compCache[ck]) return S.compCache[ck];
    const gc = window.mdGpu && mdGpu.bw(ov);
    if (gc) { putComp(ck, gc); return gc; }
    const c = document.createElement('canvas'); c.width = ov.naturalWidth || ov.width; c.height = ov.naturalHeight || ov.height;
    const g = c.getContext('2d'); g.drawImage(ov, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height); const p = d.data;
    for (let i = 0; i < p.length; i += 4) { const a = p[i] * p[i + 3] / 255; p[i] = p[i + 1] = p[i + 2] = a; p[i + 3] = 255; }
    g.putImageData(d, 0, 0);
    putComp(ck, c);
    return c;
  }
  const alphaBw = () => S.view === 'alpha' && !!S.obj && !S.auxMode;
  function drawTinted(ctx, ov, layer, cacheKey, w, h, x = 0, y = 0) {
    if (layer === 'alpha' && alphaBw()) { ctx.drawImage(bwCanvas(ov, cacheKey), x, y, w, h); return; }
    const st = S.ovs[layer];
    const prev = ctx.globalAlpha;
    ctx.globalAlpha = st.line ? Math.max(st.o, 60) / 100 : st.o / 100;   // an outline below 60 % is hard to see
    try { ctx.drawImage(tintCanvas(ov, hexRgb(st.c), cacheKey, st.line), x, y, w, h); } finally { ctx.globalAlpha = prev; }
  }
  // Z8: the mask layer at plate resolution on the plate-tile chunks (when all are loaded), else the proxy overlay `ov`
  function drawLayer(ctx, kind, ov, layer, f, img) {
    const T = S.view !== 'comp' ? tileOverlays(kind, f) : null;
    if (T) {
      const kp = T.reg.kp;
      for (const { c, img: t } of T.tiles) drawTinted(ctx, t, layer, `${S.obj}_${kind}_tile_${c.x}_${c.y}_${c.w}_${c.h}_${f}`, c.w / kp, c.h / kp, c.x / kp, c.y / kp);
      return;
    }
    if (ov) drawTinted(ctx, ov, layer, `${S.obj}_${kind}${S.viewRun && kind === 'alpha' ? '@' + S.viewRun : ''}_${f}`, img.width, img.height);
  }
  // ---- other computed objects as extra overlay layers (user idea 2026-10-15): own colour, opacity, on/off per object;
  // alpha of the active version, else the core. Kept in localStorage per object name; drawn under the selected object.
  // away from the selected object's defaults (core blue, alpha green, key pink – OVS_DEFAULT)
  const OTHER_COLORS = ['#ffb020', '#ff4040', '#ffe14d', '#3ee0ff', '#b18cff', '#ffffff', '#c08040', '#4dffb5'];
  function othersCfg() {
    if (!S.othersCfg) {
      let saved = {};
      try { saved = JSON.parse(localStorage.getItem('md_other_layers') || '{}') || {}; } catch (e) { saved = {}; }
      S.othersCfg = { on: saved.on !== false, open: saved.open !== false, per: saved.per || {}, pos: saved.pos || null };
    }
    return S.othersCfg;
  }
  function saveOthersCfg() { try { localStorage.setItem('md_other_layers', JSON.stringify(S.othersCfg)); } catch (e) { /* storage blocked */ } }
  const otherKind = (o) => (o.has && o.has.final ? 'alpha' : 'core');
  function otherObjs() {   // computed objects other than the selected one, depth order
    return (S.objects || []).filter((o) => o.name !== S.obj && o.has && (o.has.final || o.has.core));
  }
  function otherStyle(o) {
    const cfg = othersCfg();
    if (!cfg.per[o.name]) {
      const used = new Set(Object.values(cfg.per).map((x) => x.c));
      cfg.per[o.name] = { c: OTHER_COLORS.find((c) => !used.has(c)) || OTHER_COLORS[0], o: 40, vis: true };
      saveOthersCfg();   // the colour stays the same when the selection changes
    }
    return cfg.per[o.name];
  }
  const othersVisibleIn = () => S.view === 'none' || S.view === 'core' || S.view === 'alpha';
  function otherOverlay(o, f) {   // same cache key / URL as overlayFor() when that object is selected (no version / export)
    const kind = otherKind(o);
    return loadOverlay(`${o.name}_${kind}_${f}`, `/api/object/${encodeURIComponent(o.name)}/frame/${f}?view=${kind}`);
  }
  // Linked objects (parent + parts, user request 2026-10-15) are ONE layer in the view: the whole as the export assembles
  // <parent>_full (GET …/frame/f?view=full – the contact seam filled with the parent's part_seam_px), in one colour /
  // opacity (the parts inherit the parent's); the filled seam shows in the same colour. Selected group: its look.
  const objByName = (n) => (S.objects || []).find((x) => x.name === n);
  function groupOf(name) {
    const o = objByName(name); if (!o) return null;
    const p = o.part_of || o.name;
    return (S.objects || []).some((x) => x.part_of === p) ? p : null;
  }
  const selGroup = () => (S.obj ? groupOf(S.obj) : null);
  function seamOf(p) {
    const v = (objByName(p) || {}).part_seam_px;
    return v === undefined || v === null || !Number.isFinite(+v) ? 4 : Math.max(0, Math.min(40, Math.round(+v)));
  }
  // src = the layer of the view (alpha | core | core_fix), v = the version shown (comparison) of the selected object only –
  // otherwise the Rdzeń / maska 1+2 / Alpha views of a linked object were the same picture (bug 2026-10-15)
  const wholeKey = (p, f, src = 'alpha', v = '', who = p) => `${p}_full_s${seamOf(p)}${src !== 'alpha' ? '_' + src : ''}${v ? '@' + who + ':' + v : ''}_${f}`;
  const wholeOverlay = (p, f, src = 'alpha', v = '', who = p) => loadOverlay(wholeKey(p, f, src, v, who),
    `/api/object/${encodeURIComponent(v ? who : p)}/frame/${f}?view=full&s=${seamOf(p)}&src=${src}${v ? '&v=' + v : ''}`);
  // the whole of the selected group in the current view: core view = the core (stage 2: the mask 1+2), alpha = the version shown
  const selWholeSrc = () => (S.view === 'core' ? (S.stage === ST.BRUSH && S.info && S.info.has && S.info.has.core_fix ? 'core_fix' : 'core') : 'alpha');
  function ensureOthers(f) {
    const cfg = othersCfg(); if (!cfg.on || !othersVisibleIn()) return;
    const sel = selGroup();
    for (const o of otherObjs()) {
      const g = groupOf(o.name);
      if (g) { if (g !== sel && otherStyle(objByName(g) || o).vis) wholeOverlay(g, f); continue; }
      if (otherStyle(o).vis) otherOverlay(o, f);
    }
  }
  function drawOthers(ctx, img, f) {
    const cfg = othersCfg(); if (!cfg.on || !othersVisibleIn()) return;
    const sel = selGroup(), done = new Set();
    for (const o of otherObjs().slice().reverse()) {   // farthest first, the one closest to the camera on top
      const g = groupOf(o.name);
      if (g) {   // a linked group: drawn once, as its whole, in the parent's colour (the selected group: drawOverlay)
        if (g === sel || done.has(g)) continue;
        done.add(g);
        const gs = otherStyle(objByName(g) || o); if (!gs.vis || gs.o <= 0) continue;
        const ov = wholeOverlay(g, f); if (!ov || ov === 'missing') continue;
        const pa = ctx.globalAlpha; ctx.globalAlpha = gs.o / 100;
        try { ctx.drawImage(tintCanvas(ov, hexRgb(gs.c), wholeKey(g, f), false), 0, 0, img.width, img.height); }
        finally { ctx.globalAlpha = pa; }
        continue;
      }
      const st = otherStyle(o); if (!st.vis || st.o <= 0) continue;
      const ov = otherOverlay(o, f); if (!ov || ov === 'missing') continue;
      const pa = ctx.globalAlpha; ctx.globalAlpha = st.o / 100;
      try { ctx.drawImage(tintCanvas(ov, hexRgb(st.c), `${o.name}_${otherKind(o)}_${f}`, false), 0, 0, img.width, img.height); }
      finally { ctx.globalAlpha = pa; }
    }
  }
  async function refreshOthersHas() {   // `has` of the other objects may have changed (computed while selected)
    try {
      const d = await api('GET', '/api/objects');
      const by = Object.fromEntries(d.objects.map((o) => [o.name, o]));
      for (const o of S.objects || []) if (by[o.name]) o.has = by[o.name].has;
    } catch (e) { /* keep the old state */ }
    renderOthersPanel();
  }
  // floating palette (user request 2026-10-15): dragged by its header (⠿), position kept in md_other_layers, always inside the view
  // Floating palette over the view: dragged by its header (`head` selector), kept inside the view (also after a resize),
  // double click on the header = back to its CSS corner. pos = {get(), set(pos|null)} (the caller stores it).
  function placeFloating(box, w, p) {
    if (!p || box.style.display === 'none') return;
    const maxX = Math.max(0, w.clientWidth - box.offsetWidth), maxY = Math.max(0, w.clientHeight - box.offsetHeight);
    box.style.left = `${Math.min(Math.max(0, p.x), maxX)}px`; box.style.top = `${Math.min(Math.max(0, p.y), maxY)}px`;
    box.style.right = 'auto'; box.style.bottom = 'auto';   // a palette anchored at the bottom (mb-palette) kept its CSS bottom and stretched
  }
  function makeFloating(box, w, head, pos) {
    for (const ev of ['mousedown', 'wheel', 'dblclick', 'contextmenu']) box.addEventListener(ev, (e) => e.stopPropagation());
    box.addEventListener('mousedown', (e) => {
      if (e.button !== 0 || !e.target.closest(head) || e.target.closest('input, button, label')) return;
      e.preventDefault();
      const b = box.getBoundingClientRect(), wr = w.getBoundingClientRect();
      const dx = e.clientX - b.left, dy = e.clientY - b.top;
      box.classList.add('dragging');
      const move = (ev) => placeFloating(box, w, { x: ev.clientX - wr.left - dx, y: ev.clientY - wr.top - dy });
      const up = () => {
        document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up);
        box.classList.remove('dragging');
        pos.set({ x: parseFloat(box.style.left) || 0, y: parseFloat(box.style.top) || 0 });
      };
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });
    box.addEventListener('dblclick', (e) => {
      if (!e.target.closest(head) || e.target.closest('input, button, label')) return;
      pos.set(null); box.style.left = ''; box.style.top = ''; box.style.right = ''; box.style.bottom = '';
    });
    if (window.ResizeObserver) new ResizeObserver(() => placeFloating(box, w, pos.get())).observe(w);
  }
  const othersPos = { get: () => othersCfg().pos, set: (p) => { othersCfg().pos = p; saveOthersCfg(); } };
  function placeOthersPanel(box, w) { placeFloating(box, w, othersCfg().pos); }
  // Layers panel (user request 2026-10-15, "like Photoshop"): ALL objects in depth order (top = closest to the camera),
  // the selected one highlighted. Click = select; drag onto the middle of a row = part of that object, onto its upper /
  // lower edge = depth order; eye / colour / opacity = the mask of the other computed objects drawn under the selected one.
  // seam of a linked group, under its last member: the gap between the parts' masks up to ~2 x this is filled (preview in
  // the group's colour = what <parent>_full exports)
  function seamRow(o, all, st) {
    const g = groupOf(o.name); if (!g) return '';
    const ms = all.filter((x) => x.name === g || x.part_of === g);
    if (ms[ms.length - 1].name !== o.name) return '';
    const sv = seamOf(g);
    const col = g === selGroup() ? (S.ovs[S.view === 'alpha' ? 'alpha' : 'core'] || {}).c : (st && st.c);
    return `<div class="ol-seam" data-p="${esc(g)}" title="Szew grupy „${esc(g)}”: szczelina między maskami połączonych obiektów do ~2 × tej wartości jest zalewana (zewnętrzny obrys bez zmian). Podgląd w kolorze grupy; ten sam szew ma eksport ${esc(g)}_full">
        <span class="ol-seam-l"><i style="background:${col || '#888'}"></i>szew ${esc(g)}</span><input type="range" class="ol-seam-r" min="0" max="40" step="1" value="${sv}"><span class="ol-seam-v">${sv} px</span></div>`;
  }
  function renderOthersPanel() {
    const w = $('canvasWrapper'); if (!w) return;
    let box = $('otherLayers');
    const all = S.objects || [];
    if (!all.length || !(V() && V().projectLoaded)) { if (box) box.style.display = 'none'; return; }
    if (!box) {
      box = el(`<div class="other-layers" id="otherLayers"></div>`);
      w.appendChild(box);
      makeFloating(box, w, '.ol-head', othersPos);
    }
    box.style.display = '';
    const cfg = othersCfg();
    const computed = new Set(otherObjs().map((o) => o.name));
    const grpHas = (g) => all.some((x) => (x.name === g || x.part_of === g) && computed.has(x.name));
    box.innerHTML = `<div class="ol-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)">
        <span class="ol-grip">⠿</span>
        <span class="ol-title">Obiekty (${all.length})</span>
        <label class="ol-all" title="Pokaż maski innych policzonych obiektów (pod maską wybranego obiektu; w widoku Comp ukryte)">
          <input type="checkbox" id="olAll" ${cfg.on ? 'checked' : ''} title="Włącz / wyłącz maski innych obiektów na podglądzie"> maski innych</label>
        <button type="button" class="ol-btn" id="olNew" title="Nowy obiekt">＋</button>
        <button type="button" class="ol-fold" id="olFold" title="${cfg.open ? 'Zwiń listę' : 'Rozwiń listę'}">${cfg.open ? '▾' : '▸'}</button>
      </div>` + (cfg.open ? all.map((o) => {
        const cur = o.name === S.obj, g = groupOf(o.name);
        const has = g ? grpHas(g) : computed.has(o.name);
        const st = has ? otherStyle(g ? (objByName(g) || o) : o) : null;
        const inherit = !!(g && o.name !== g);      // a part: colour / opacity of its parent (one layer)
        const selIn = !!(g && g === selGroup());    // the selected group: drawn in the top bar's look
        const kind = has && otherKind(o) === 'core' ? ' <i>(rdzeń)</i>' : '';
        const name = `<span class="ol-name">${o.part_of ? '<span class="ol-sub">↳</span> ' : ''}${o.depth_order}. ${esc(o.name)}${kind}</span>`;
        const offRow = has && !cur && !selIn && !(st.vis && cfg.on);
        const own = has && !cur && !inherit && !selIn;   // its own eye / colour / opacity
        const tail = own
          ? `<input type="range" class="ol-op" min="0" max="100" step="5" value="${st.o}" title="Krycie maski ${esc(o.name)}${g ? ' i jego części' : ''}"><span class="ol-val">${st.o}%</span>`
          : `<span class="ol-status" title="${cur ? 'Wybrany obiekt – wygląd nakładki (całej grupy) ustawiasz w górnym pasku' : inherit ? `Część „${esc(g)}” – kolor i krycie jak „${esc(g)}”` : selIn ? 'Grupa wybranego obiektu – wygląd z górnego paska' : 'Brak policzonego wyniku'}">${cur ? esc((o.tool && o.tool.status) || 'wybrany') : inherit ? `jak „${esc(g)}”` : selIn ? 'grupa wybranego' : 'brak wyniku'}</span>`;
        const eye = own
          ? `<button type="button" class="ol-eye" title="Pokaż / ukryj maskę obiektu ${esc(o.name)}${g ? ' z częściami' : ''}">${st.vis ? '👁' : '◌'}</button><input type="color" class="ol-color" value="${st.c}" title="Kolor maski ${esc(o.name)}${g ? ' i jego części' : ''}">`
          : cur ? `<button type="button" class="ol-eye ol-eye-sel" title="Pokaż / ukryj maskę wybranego obiektu na podglądzie (narzędzia – punkty, pędzel, strefy – zostają)">${S.hideSel ? '◌' : '👁'}</button>` + '<span></span>'
          : `<span class="ol-eye ol-eye-off" title="${inherit ? `Kolor jak „${esc(g)}”` : 'Brak wyniku – nie ma czego pokazać'}"></span>` +
            (inherit && has && !selIn ? `<span class="ol-swatch" style="background:${st.c}" title="Kolor jak „${esc(g)}”"></span>` : '<span></span>');
        // group look (user request 2026-10-23): the main object left, its parts indented under it with a guide line
        const gcls = g ? (o.name === g ? 'grp grp-main' : 'grp grp-part') : '';
        return `<div class="ol-row ${cur ? 'cur' : ''} ${gcls} ${offRow ? 'off' : ''}" data-o="${esc(o.name)}" draggable="true"
            title="Klik = wybierz obiekt · przeciągnij na środek innego = jego część · na górną / dolną krawędź = kolejność głębi (0 = najbliżej kamery)">
          <span class="ol-drag">⋮⋮</span>${eye}${name}${tail}<button type="button" class="ol-del" title="Usuń obiekt ${esc(o.name)} (z potwierdzeniem)">🗑</button>
        </div>` + seamRow(o, all, st);
      }).join('') : '');
    placeOthersPanel(box, w);
    const redraw = () => { saveOthersCfg(); V() && V().render(); };
    $('olAll').addEventListener('change', (e) => { cfg.on = e.target.checked; renderOthersPanel(); redraw(); });
    $('olFold').addEventListener('click', () => { cfg.open = !cfg.open; renderOthersPanel(); saveOthersCfg(); });
    $('olNew').addEventListener('click', () => newObjectDialog());
    box.querySelectorAll('.ol-row').forEach((row) => {
      const o = all.find((x) => x.name === row.dataset.o);
      const selEye = row.querySelector('button.ol-eye-sel');   // the selected object's own mask on / off (user request 2026-10-17)
      if (selEye) selEye.addEventListener('click', (e) => { e.stopPropagation(); S.hideSel = !S.hideSel; renderOthersPanel(); V() && V().render(); });
      const eye = row.querySelector('button.ol-eye:not(.ol-eye-sel)');
      if (eye) {
        const st = otherStyle(o);
        eye.addEventListener('click', (e) => { e.stopPropagation(); st.vis = !st.vis; renderOthersPanel(); redraw(); });
        row.querySelector('.ol-color').addEventListener('input', (e) => { st.c = e.target.value; redraw(); });
        const op = row.querySelector('.ol-op');
        if (op) op.addEventListener('input', (e) => { st.o = +e.target.value; row.querySelector('.ol-val').textContent = `${st.o}%`; redraw(); });
      }
      row.querySelector('.ol-del').addEventListener('click', (e) => { e.stopPropagation(); deleteObjectDialog(o.name); });
      // controls inside the row must not start a drag of the row (slider, colour)
      row.addEventListener('mousedown', (e) => { row.draggable = !e.target.closest('input, button'); });
      row.addEventListener('click', (e) => { if (!e.target.closest('input, button')) selectObject(o.name); });
      const nm = row.querySelector('.ol-name');
      if (nm) { nm.title = 'Dwuklik: zmień nazwę obiektu'; nm.addEventListener('dblclick', (e) => { e.stopPropagation(); renameInline(nm, o.name); }); }
    });
    box.querySelectorAll('.ol-seam').forEach((row) => {
      const p = row.dataset.p, r = row.querySelector('.ol-seam-r'), lab = row.querySelector('.ol-seam-v');
      row.addEventListener('mousedown', (e) => e.stopPropagation());
      r.addEventListener('input', () => { lab.textContent = `${r.value} px`; });
      r.addEventListener('change', async () => {
        const v = +r.value;
        try {
          await api('PUT', `/api/object/${encodeURIComponent(p)}/settings`, { part_seam_px: v });
          const po = objByName(p); if (po) po.part_seam_px = v;
          status(`Szew grupy „${p}”: ${v} px (podgląd i eksport ${p}_full)`, 'ok');
        } catch (er) { status(`Szew NIE zapisany: ${er.message}`, 'err'); r.value = seamOf(p); lab.textContent = `${seamOf(p)} px`; }
        if (V()) { preloadOverlays(V().currentFrame, 0); V().render(); }
      });
    });
    wireObjDrag(box, '.ol-row', 'o');
  }
  function note(t) {
    const n = $('canvasNote'); if (!n) return;
    n.style.display = t ? '' : 'none'; n.textContent = t || '';
    if (t) {   // always below the HUD row, also when it wraps (long „Maska: …”) – they overlapped (2026-10-17)
      const hud = $('canvasHud'), w = $('canvasWrapper');
      if (hud && w) n.style.top = `${Math.max(14, hud.getBoundingClientRect().bottom - w.getBoundingClientRect().top + 8)}px`;
    }
  }

  function plateSize() { const p = V() && V().projectData; return p ? [p.plate_w, p.plate_h] : null; }

  function drawOverlay() {
    const v = V(); if (!v) return;
    syncOverlayStyleUi();
    const img = v.getCurrentImage && v.getCurrentImage(); if (!img) { note(''); return; }
    const ps = plateSize(); if (!ps) return;
    const k = img.width / ps[0];
    const ctx = v.ctx;
    const f = v.currentFrame;
    ctx.save();
    try {
      ctx.translate(v.panX, v.panY); ctx.scale(v.zoom, v.zoom);
      let noteTxt = '';
      if (S.obj && !S.auxMode && S.holdC) {   // Z9: C held = the plate alone (stages 3–5)
        note('C: sam plate (bez nakładki)'); ctx.restore(); return;
      }
      // cut-out on a comp background: the mask layers are skipped (they would tint the comp), the tools are still drawn –
      // brush strokes, polygons, zones, points, brush cursor (user report 2026-10-16: painting showed nothing in „Na tle”)
      updateMaskHud(f);   // „Maska: …” – also in the comp-background view
      const covers = !!(v.compareCoversView && v.compareCoversView());
      if (!covers) {
      if (alphaBw()) { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, img.width, img.height); }   // black where there is no alpha
      else if (!S.auxMode) drawOthers(ctx, img, f);   // other computed objects, under the selected one
      // Alpha / Comp in a stage with its own mask = exactly the mask of the stage, the one „Maska: …” names (matteFor –
      // SAM 3.1 core, mask 1+2, the stage's version; user report 2026-10-22: stage SAM 3.1 showed the active soft alpha)
      const stageMask = S.obj && !S.auxMode && !S.viewRun && (S.view === 'alpha' || S.view === 'comp')
        && [ST.SAM3, ST.BRUSH, ST.MATANYONE, ST.VIDEOMAMA, ST.VITMATTE, ST.LTX, ST.FINAL_TOUCH].includes(S.stage);
      const sw = S.obj && !S.auxMode && (S.view === 'core' || (S.view === 'alpha' && !stageMask)) ? selGroup() : null;
      if (mbPreviewOn() && (S.view === 'alpha' || S.view === 'comp')) {   // „Motion blur alfy” preview of this frame
        const ov = mbOverlay(f), base = mbBase(), s = `${mbStrength()}s${mbShutter()}`;
        if (ov === 'missing') noteTxt = `Podgląd motion blura F${f}: brak wyniku wersji ${base} w tej klatce`;
        else if (ov) {
          if (S.view === 'comp') ctx.drawImage(compCanvas(img, ov, `${S.obj}@mb${base}x${s}_${f}`, f), 0, 0);
          else drawTinted(ctx, ov, 'alpha', `${S.obj}_mb@${base}x${s}_${f}`, img.width, img.height);
        }
        noteTxt = noteTxt || `Podgląd: ${base} + motion blur ×${mbStrength()}, shutter ${mbShutter()}° (wyłącz „👁 podgląd” w palecie Motion blur, żeby zobaczyć bez)`;
      } else if (S.hideSel && S.obj && !S.auxMode) {   // 👁 off in the Objects palette: no mask of the selected object
        noteTxt = noteTxt || 'Maska wybranego obiektu ukryta – 👁 w palecie Obiekty';
      } else if (stageMask) {
        const m = matteFor(f);
        if (!m) noteTxt = S.matteNote || `Brak maski etapu w klatce ${f}`;
        else if (m !== 'loading') {
          if (S.view === 'comp') ctx.drawImage(compCanvas(img, m.img, `stg_${m.key}`, f), 0, 0);
          else drawTinted(ctx, m.img, 'alpha', `stg_${m.key}`, img.width, img.height);
        }
      } else if (sw) {   // linked objects: one layer (whole + filled seam) in the selected layer's colour
        const src = selWholeSrc(), vr = src === 'alpha' && S.viewRun ? S.viewRun : '';
        const ov = wholeOverlay(sw, f, src, vr, S.obj);
        if (ov === 'missing') noteTxt = `Brak masek „${sw}” i jego części w klatce ${f}` + (src === 'alpha' ? '' : ` (${src === 'core' ? 'rdzeń' : 'maska 1+2'})`);
        else if (ov) drawTinted(ctx, ov, S.view === 'core' ? 'core' : 'alpha', wholeKey(sw, f, src, vr, S.obj), img.width, img.height);
        if (!noteTxt) noteTxt = `Grupa „${sw}” – całość ze szwem: ${src === 'core' ? 'rdzeń SAM 3.1' : src === 'core_fix' ? 'maska 1+2' : (vr ? `wersja ${vr} (${S.obj})` : 'aktywne wersje')}`;
      } else if (S.obj && !S.auxMode) {
        if (S.view === 'comp' || S.view === 'alpha') {
          const ov = overlayFor('alpha', f);
          if (ov === 'missing') noteTxt = `Brak wyniku w klatce ${f} – „▶ Przelicz MatAnyone2” (etap ${ST.MATANYONE}) albo „Przelicz finalnie” (etap ${ST.EXPERIMENTS})`;
          else if (ov) { const vk = S.viewRun ? '@' + S.viewRun : ''; if (S.view === 'comp') ctx.drawImage(compCanvas(img, ov, `${S.obj}${vk}${exportCtx() ? '#x' : ''}_${f}`, f), 0, 0); else drawLayer(ctx, 'alpha', ov, 'alpha', f, img); }
        } else if (S.view === 'core') {
          const kf = keyOf(f);   // in a key frame the result IS the key – show it, not the old core
          const ck = coreKind(f);  // stage 2: the mask 1+2 when „Zastosuj poprawki” made it
          const ov = overlayFor(ck, f);
          if (ov === 'missing') noteTxt = ck === 'core_fix' ? `Brak maski 1+2 w klatce ${f} – „▶▶ Zastosuj poprawki” (etap ${ST.BRUSH})`
            : `Brak rdzenia w klatce ${f} – „▶▶ Przelicz całość” (etap ${ST.SAM3})`;
          else if (ov) drawLayer(ctx, ck, ov, 'core', f, img);
          if (ck === 'core_fix' && !noteTxt && stageState('cotracker').state !== 'done') noteTxt = 'Maska 1+2 nieaktualna (poprawki zmienione) – „▶▶ Zastosuj poprawki”';
          if (S.stage === ST.SAM3 && kf && kf.kind === 'correction') {
            for (const [kind, rgb] of [['delta_add', [60, 255, 90]], ['delta_remove', [255, 60, 80]]]) {
              const d = overlayFor(kind, f);
              if (d && d !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.85; ctx.drawImage(tintCanvas(d, rgb, `${S.obj}_${kind}_${f}`, true), 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
            }
          }
        } else if (S.stage === ST.SAM3) {
          // stage SAM3 without a view: the SAM3 mask of the points / chosen candidate in this frame (if any)
          if (S.keys.some((x) => x.frame === f)) {
            const ov = overlayFor('key', f);
            if (ov && ov !== 'missing') drawLayer(ctx, 'key', ov, 'key', f, img);
          }
        }
      }
      { const t = drawEdgePreview(ctx, img, f); if (t) noteTxt = noteTxt || t; }   // stage 3 / 5: edge band before computing
      { const t = drawLtxPreview(ctx, img, f); if (t) noteTxt = noteTxt || t; }    // stage LTX: band / re-limited result
      { const t = drawVmmPreview(ctx, img, f); if (t) noteTxt = noteTxt || t; }    // stage VideoMaMa: band / recomposed result
      { const t = drawStagePreview(ctx, img, f); if (t) noteTxt = t; }               // stages 3 / 4 / 5: preview of a few frames
      if (S.obj && !S.auxMode && S.cmpDiff && (S.view === 'alpha' || S.view === 'comp')) {   // stage-4 comparison |A − B|
        const c = S.cmpDiff;
        const d = loadOverlay(`${S.obj}_adiff@${c.a}~${c.b}_${f}`, `/api/object/${S.obj}/frame/${f}?view=alpha_diff&v=${c.a}&v2=${c.b}`);
        if (d && d !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.9; ctx.drawImage(tintCanvas(d, [255, 40, 40], `${S.obj}_adiff@${c.a}~${c.b}_${f}`), 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
        noteTxt = noteTxt || `Różnica |A − B| ×3 (czerwone): ${c.a} − ${c.b}`;
      }
      if (S.obj && !S.auxMode && S.onion && (S.view === 'core' || S.view === 'alpha' || S.view === 'comp')) {   // Z9: onion skin ±1
        const kind = S.view === 'core' ? null : 'alpha';
        for (const [df, rgb] of [[-1, [255, 70, 70]], [1, [70, 140, 255]]]) {
          const g = f + df; if (g < v.firstFrame || g > v.lastFrame) continue;
          const kd = kind || coreKind(g);
          const ov = overlayFor(kd, g);
          if (ov && ov !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.9; ctx.drawImage(tintCanvas(ov, rgb, `${S.obj}_${kd}${kd === 'alpha' && S.viewRun ? '@' + S.viewRun : ''}_${g}`, true), 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
        }
        noteTxt = noteTxt || `Onion skin: czerwony = F${f - 1}, niebieski = F${f + 1} (O wyłącza)`;
      }
      if (S.obj && !S.auxMode && S.flicker && versionStage()) {   // Z9: flicker map of the version (M)
        const fm = loadOverlay(`${S.obj}_flicker@${S.viewRun || S.activeRun || 'cur'}_all`, `/api/object/${S.obj}/frame/${f}?view=flicker${S.viewRun ? '&v=' + S.viewRun : ''}`);
        if (fm === 'missing') noteTxt = 'Brak mapy migotania – „Przelicz finalnie”';
        else if (fm) { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.85; ctx.drawImage(fm, 0, 0, img.width, img.height); ctx.globalAlpha = pa; noteTxt = noteTxt || 'Mapa migotania wersji (INFERNO: jaśniej = krawędź bardziej migocze; M wyłącza)'; }
      }
      drawCandidates(ctx, img, k, f);   // Z10
      } else {   // cut-out on a background: the edge bands stay visible over it (user report 2026-10-17), no tinted alpha
        { const t = drawEdgePreview(ctx, img, f); if (t) noteTxt = noteTxt || t; }
        { const t = drawLtxPreview(ctx, img, f, true); if (t) noteTxt = noteTxt || t; }
        { const t = drawVmmPreview(ctx, img, f, true); if (t) noteTxt = noteTxt || t; }
        if (S.obj && !S.auxMode && S.flicker && versionStage()) {   // the flicker map, when ticked in this view
          const fm = loadOverlay(`${S.obj}_flicker@${S.viewRun || S.activeRun || 'cur'}_all`, `/api/object/${S.obj}/frame/${f}?view=flicker${S.viewRun ? '&v=' + S.viewRun : ''}`);
          if (fm && fm !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.85; ctx.drawImage(fm, 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
        }
      }   // !covers
      if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.draw(ctx, img, k, f);   // zone keys / region / preview overlays
      if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.draw(ctx, img, k, f);   // stage band points
      if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.draw(ctx, img, k, f);   // stage 2: brush corrections
      if (window.MD_FT && window.MD_FT.ready) window.MD_FT.draw(ctx, img, k, f);   // stage Final Touch: saved strokes
      if (S.obj && !S.auxMode && S.stage === ST.EXPERIMENTS && S.zonePreview && S.zonePreview.f === f) {
        // stage Eksperymenty threshold-zone preview ("Podgląd (ta klatka)"): what the zone adds (green) / removes (red) on the core
        for (const [kind, rgb] of [['zone_add', [60, 255, 90]], ['zone_remove', [255, 60, 80]]]) {
          const d = overlayFor(kind, f);
          if (d && d !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.85; ctx.drawImage(tintCanvas(d, rgb, `${S.obj}_${kind}_${f}`, true), 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
        }
      }
      if (!noteTxt && S.obj && versionStage() && S.viewRun && (S.view === 'alpha' || S.view === 'comp'))
        noteTxt = `Wersja ${S.viewRun} (aktywna: ${S.activeRun || '–'}) – B wraca do aktywnej`;
      note(noteTxt);
      const showEdge = !!(S.obj && !S.auxMode && S.stage === ST.EXPERIMENTS && S.edgeFix && !S.edgeFix.dismissed && expPvGet('edge', true));
      const lg = edgeLegendEl(); if (lg) lg.style.display = showEdge ? '' : 'none';
      if (showEdge) {
        const ep = edgePiecesFor(f);   // Z27 pkt 1: pieces of the proposed edge correction (red = halo, blue = cut)
        if (ep) {
          ctx.lineCap = 'round';
          for (const pc of ep) {
            const off = S.edgeFix.skip.has(pc.id);   // unchecked: stays visible (grey, dashed) so a click can re-check it
            const hot = S.edgeFix.hover === pc.id;
            ctx.lineWidth = (hot ? 5 : 3) / v.zoom;
            ctx.setLineDash(off ? [6 / v.zoom, 5 / v.zoom] : []);
            ctx.strokeStyle = off ? 'rgba(148,163,184,0.75)' : (pc.kind === 'halo' ? 'rgba(255,60,60,0.9)' : 'rgba(60,140,255,0.9)');
            ctx.beginPath(); pc.pts.forEach((q, i) => i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k)); ctx.stroke();
          }
          ctx.setLineDash([]);
        }
      }
      if (S.crops && S.stage === ST.SAM3 && S.showCrops) {
        ctx.lineWidth = 1 / v.zoom; ctx.strokeStyle = '#ffcc33';
        for (const s of S.crops) if (s.first <= f && f <= s.last) ctx.strokeRect(s.crop[0] * k, s.crop[1] * k, s.crop[2] * k, s.crop[3] * k);
      }
      for (const st of (S.strokes[f] || []).concat(S.drawing && S.drawing.f === f ? [S.drawing] : [])) {
        const col = st.mode === 'add' ? 'rgba(80,255,120,0.6)' : st.mode === 'erase' ? 'rgba(203,213,225,0.55)' : 'rgba(255,60,90,0.6)';
        if (st.p.some((q) => q.length >= 3)) {   // Z23: radius per point (pen pressure)
          ctx.fillStyle = col;
          drawVarStroke(ctx, st.p, k, st.radius_px);
          continue;
        }
        ctx.strokeStyle = col;
        ctx.lineWidth = 2 * st.radius_px * k; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath(); st.p.forEach((q, i) => i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k));
        if (st.p.length === 1) ctx.lineTo(st.p[0][0] * k + 0.01, st.p[0][1] * k);
        ctx.stroke();
      }
      for (const pg of (S.polys[f] || []).concat(S.poly && S.poly.f === f ? [S.poly] : [])) {
        ctx.strokeStyle = pg.mode === 'add' ? '#50ff78' : '#ff3c5a'; ctx.lineWidth = 1.5 / v.zoom;
        ctx.beginPath(); pg.p.forEach((q, i) => i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k));
        if (pg.closed) ctx.closePath(); ctx.stroke();
      }
      if (drawStage() && ['brush_add', 'brush_remove', 'brush_erase'].includes(S.tool)) {
        const c = S.resizing ? S.resizing.p : S.hover;   // Z23: brush reach before clicking / while resizing
        if (c) {
          ctx.save(); ctx.lineWidth = 1.5 / v.zoom; ctx.strokeStyle = S.tool === 'brush_add' ? '#50ff78' : S.tool === 'brush_erase' ? '#c4c4c8' : '#ff3c5a';
          ctx.beginPath(); ctx.arc(c[0] * k, c[1] * k, S.radius * k, 0, 2 * Math.PI); ctx.stroke();
          if (S.resizing) { ctx.fillStyle = '#fff'; ctx.font = `${13 / v.zoom}px "IBM Plex Sans", sans-serif`; ctx.textAlign = 'center'; ctx.fillText(`${S.radius} px`, c[0] * k, c[1] * k); }
          ctx.restore();
        }
      }
      if (S.corrMark && S.corrMark.f === f) {
        const c = S.corrMark, r = 9 / v.zoom;
        ctx.save(); ctx.setLineDash([4 / v.zoom, 3 / v.zoom]); ctx.lineWidth = 2 / v.zoom;
        ctx.strokeStyle = c.mode === 'add' ? '#3cff5a' : '#ff3c50';
        ctx.beginPath(); ctx.arc(c.p[0] * k, c.p[1] * k, r, 0, 7); ctx.stroke(); ctx.restore();
      }
      if (S.stage === ST.SAM3) {
        const pts = S.points[f] || [];
        pts.forEach((p, i) => {
          const x = p[0] * k, y = p[1] * k, r = 5 / v.zoom;
          const unsaved = (S.pointsDirty[f] || S.pointsFailed[f]) && !S.pointsBusy[f];
          ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
          ctx.fillStyle = unsaved ? '#f0703a' : (p[2] > 0 ? '#22dd55' : '#ff3355'); ctx.fill();
          ctx.lineWidth = 1.5 / v.zoom; ctx.strokeStyle = '#fff'; ctx.stroke();
          if (unsaved) { ctx.save(); ctx.setLineDash([3 / v.zoom, 3 / v.zoom]); ctx.strokeStyle = '#f0703a'; ctx.beginPath(); ctx.arc(x, y, r * 2, 0, 7); ctx.stroke(); ctx.restore(); }
          ctx.fillStyle = '#fff'; ctx.font = `${10 / v.zoom}px "IBM Plex Sans", sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(p[2] > 0 ? '+' : '−', x, y);
        });
      }
    } finally {
      ctx.restore();
    }
    if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.drawLoupe(ctx, img, k);   // screen space
  }

  // ------------------------------------------------------------------------------------------------ loading
  async function loadStages() {
    // the same object keeps its info while asking again (2026-10-21: null in between made the stage-5 zone form read
    // the stage engine as ViTMatte and move a BiRefNet zone to ViTMatte base)
    const o = S.obj;
    if (S.infoObj !== o) S.info = null;
    let info = null;
    if (o) { try { info = await api('GET', `/api/object/${o}/stages`); } catch (e) { info = null; } }
    if (S.obj === o) { S.info = info; S.infoObj = o; }
    renderStageBar(); updateViewTabs(); updateStageButtons(); renderSegsLane(); renderPointsPalette(); renderStageInfo(); renderExportSource(); renderEdgePalette(); renderMbPalette(); renderLtxPalette(); renderVmmPalette();
    renderStageBar(); updateViewTabs(); updateStageButtons(); renderSegsLane(); renderPointsPalette(); renderStageInfo(); renderExportSource(); renderEdgePalette(); renderMbPalette(); renderLtxPalette(); renderExpPalette();
    if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.onStages();
    if (window.MD_FT && window.MD_FT.ready) window.MD_FT.onStages();
    updateDelKeyBtn();   // every stage module calls loadStages after a key change
    if (window.MD_ZONES && window.MD_ZONES.ready && window.MD_ZONES.onStages) window.MD_ZONES.onStages();
    loadStagePreviews();
    window.dispatchEvent(new CustomEvent('md-object-stages', { detail: { name: S.obj, info: S.info } }));   // static/life.js
  }
  // matte of the selected object in frame f for the comp-background comparison (app.js bgComp): the final alpha, else
  // the key / core. { img, key, kind } | 'loading' | null (no object / no mask in this frame)
  // the stage's own result for the comparison „Na tle” (user report 2026-10-16: the cut-out always showed the active
  // version, whatever stage was picked): SAM3 = the SAM3 core / key, CoTracker = the mask 1+2, MatAnyone / ViTMatte / LTX
  // = that stage's version (else the mask the stage starts from), other stages = the viewed / active alpha as before
  // stage 5 with two engines (user request 2026-10-22): the mask of the engine picked in the HUD (default = the stage
  // engine) – its latest version on the current ViTMatte base; '' = not computed with that engine yet
  const vmEngine = () => S.vmPick || (S.info && S.info.edge_engine) || 'vitmatte';
  function vmRunFor(eng) {
    const b = (S.info && S.info.vitmatte_base) || 'matanyone';
    const sr = (S.info && S.info.stage_runs) || {};
    const mine = (S.runs || []).filter((r) => r.stage === 'vitmatte' && (r.edge_engine || 'vitmatte') === eng && (!r.base || r.base === b));
    const cur = sr[`vitmatte@${b}`] || (b === 'matanyone' ? sr.vitmatte : '');
    if (cur && mine.some((r) => r.run === cur)) return cur;
    mine.sort((x, y) => String(y.created_at || '').localeCompare(String(x.created_at || '')));
    return mine.length ? mine[0].run : '';
  }
  function stageAlphaRun() {
    const sr = (S.info && S.info.stage_runs) || {};
    if (S.stage === ST.MATANYONE) return sr.matanyone || '';
    if (S.stage === ST.VITMATTE) {   // no ViTMatte version yet: its base (MatAnyone of stage 3, or the mask 1+2 = '')
      const b = (S.info && S.info.vitmatte_base) || 'matanyone';
      return vmRunFor(vmEngine()) || (b === 'matanyone' ? sr.matanyone || '' : b === 'videomama' ? sr.videomama || '' : '');
    }
    if (S.stage === ST.LTX) return sr.ltx || '';
    if (S.stage === ST.VIDEOMAMA) return sr.videomama || '';
    return null;   // not a stage with its own version
  }
  // „Maska: …” next to the HUD mode (user request 2026-10-17): which mask the preview shows right now – the stage's
  // version with its frames, the mask 1+2 / SAM3 core, or a stage preview – and when the version misses this frame
  const STAGE_NAME = () => ({ [ST.MATANYONE]: 'MatAnyone2', [ST.VIDEOMAMA]: 'VideoMaMa', [ST.VITMATTE]: 'ViTMatte', [ST.LTX]: 'LTX' });
  function updateMaskHud(f) {
    const el = $('hudMask');
    if (!el) return;
    let txt = '', warn = false;
    if (S.obj && !S.auxMode && V() && V().projectLoaded && S.viewRun && (S.view === 'alpha' || S.view === 'comp')) {
      txt = `Maska: wersja ${S.viewRun} (wybrana z listy wersji)`;
    } else if (S.obj && !S.auxMode && V() && V().projectLoaded) {
      const has12 = S.info && S.info.has && S.info.has.core_fix;
      const st = spvStage(); const m = spvOf(st);
      const run = stageAlphaRun();
      if (m && (m.frames || []).includes(f) && spvWanted(st)) {
        txt = `Maska: podgląd ${spvLabel(st)} F${m.frames[0]}–${m.frames[m.frames.length - 1]} (nie wynik etapu)${m.stale ? ' · nieaktualny' : ''}`;
        warn = !!m.stale;
      } else if (S.stage === ST.SAM3) txt = 'Maska: rdzeń SAM 3.1';
      else if (S.stage === ST.BRUSH) txt = has12 ? 'Maska: maska 1+2 (rdzeń + poprawki)' : 'Maska: rdzeń SAM 3.1';
      else if (S.stage === ST.FINAL_TOUCH) {
        const ref = window.MD_FT && window.MD_FT.ready ? window.MD_FT.matteRef(f) : null;
        const n = window.MD_FT && window.MD_FT.ready && window.MD_FT.hasKeyAt(f);
        txt = ref ? `Maska: Final Touch – baza ${ref.run === 'core' ? 'maska 1+2' : ref.run}${n ? ' + korekty tej klatki' : ' (bez korekt w tej klatce)'}` : 'Maska: brak bazy Final Touch';
        warn = !ref;
      }
      else if (run === '') txt = `Maska: ${has12 ? 'maska 1+2' : 'rdzeń SAM 3.1'} – ${(S.stage === ST.VITMATTE ? EDGE_ENGINE_LABEL[vmEngine()] : STAGE_NAME()[S.stage]) || 'etap'} jeszcze niepoliczony`;
      else if (run) {
        const r = (S.runs || []).find((x) => x.run === run);
        const cov = r && r.coverage ? r.coverage : null;
        const covTxt = cov ? cov.map(([a, b]) => (a === b ? `F${a}` : `F${a}–${b}`)).join(', ') : '';
        const inCov = !cov || cov.some(([a, b]) => f >= a && f <= b);
        const who = S.stage === ST.VITMATTE && r && r.stage !== 'vitmatte' ? `baza ${EDGE_ENGINE_LABEL[vmEngine()] || 'ViTMatte'}: ${r.stage === 'videomama' ? 'VideoMaMa' : 'MatAnyone2'} (${EDGE_ENGINE_LABEL[vmEngine()] || 'ViTMatte'} jeszcze niepoliczony)`
          : S.stage === ST.VITMATTE ? (EDGE_ENGINE_LABEL[vmEngine()] || 'ViTMatte') : (STAGE_NAME()[S.stage] || 'wersja');
        // one placeholder per side of „·”: i18n patterns treat spaces next to a placeholder as optional, so
        // „{0} {1}” split „ViTMatte” after its first letter (user screenshots: „V iTMatte v007”, 2026-10-22)
        const miss = inCov ? '' : ` – ${window.mdT ? window.mdT(`BRAK w F${f}`) : `BRAK w F${f}`}`;
        txt = `Maska: ${who} · ${run + (covTxt ? ` (${covTxt})` : '') + miss}`;
        warn = !inCov;
      } else if (S.viewRun || S.activeRun) txt = `Maska: wersja ${S.viewRun || S.activeRun}`;
    }
    el.style.display = txt ? '' : 'none';
    el.textContent = txt;
    el.classList.toggle('warn', warn);
    const alt = $('hudMaskAlt');
    if (alt) {   // stage 5 with both engines: the other engine's mask, one click away (user request 2026-10-22)
      const on = !!txt && S.stage === ST.VITMATTE && !S.viewRun && bothEdgeEngines();
      alt.style.display = on ? '' : 'none';
      if (on) {
        const other = vmEngine() === 'birefnet' ? 'vitmatte' : 'birefnet';
        const run = vmRunFor(other);
        const pv = S.spv && S.spv[spvKey('vitmatte', other)];   // the other engine's preview counts too (2026-10-22)
        const pvTxt = pv && (pv.frames || []).length ? `(podgląd F${pv.frames[0]}–${pv.frames[pv.frames.length - 1]}${pv.stale ? ', nieaktualny' : ''})` : '';
        alt.textContent = `⇄ ${EDGE_ENGINE_LABEL[other]} ${run || pvTxt || '(niepoliczony)'}`;
        alt.disabled = !run && !pv;
        alt.dataset.eng = other;
      }
    }
  }
  function matteFor(f) {
    if (!S.obj) return null;
    const coreMatte = (kind) => {
      const ov = overlayFor(kind, f);
      if (ov && ov !== 'missing') return { img: ov, key: overlayKey(kind, f), kind };
      return ov === 'missing' ? null : 'loading';
    };
    // a group (parent + parts joined with the seam): the cut-out of the whole, like the core view's layer (user report
    // 2026-10-16: stage SAM3 showed only the selected object)
    const sw = selGroup();
    const wholeMatte = (src) => {
      const ov = wholeOverlay(sw, f, src, '', S.obj);
      if (ov && ov !== 'missing') return { img: ov, key: wholeKey(sw, f, src, '', S.obj), kind: src, group: sw };
      return ov === 'missing' ? null : 'loading';
    };
    const has12 = S.info && S.info.has && S.info.has.core_fix;
    if (S.stage === ST.SAM3) return sw ? wholeMatte('core') : coreMatte(keyOf(f) ? 'key' : 'core');
    if (S.stage === ST.BRUSH) return sw ? wholeMatte(has12 ? 'core_fix' : 'core') : coreMatte(coreKind(f));
    if (S.stage === ST.FINAL_TOUCH) {   // base + hand corrections of this frame (= the export source „Final Touch”)
      const ref = window.MD_FT && window.MD_FT.ready ? window.MD_FT.matteRef(f) : null;
      if (!ref) return coreMatte(has12 ? 'core_fix' : 'core');
      const al = loadOverlay(ref.key, ref.url);
      if (al && al !== 'missing') { S.matteNote = ''; return { img: al, key: ref.key, kind: 'alpha', run: `Final Touch (${ref.run})` }; }
      if (!al) return 'loading';
      S.matteNote = `Baza Final Touch (${ref.run}) nie obejmuje klatki ${f} – wybierz inną bazę albo przejdź do policzonych klatek`;
      return null;
    }
    const run = stageAlphaRun();
    const withZonePv = (m) => {   // user report 2026-10-16: „Przelicz klatkę” changed nothing on the cut-out
      const pm = S.stage === ST.VITMATTE && m && m !== 'loading' && window.MD_ZONES && window.MD_ZONES.previewMatte
        ? window.MD_ZONES.previewMatte(f) : null;
      if (!pm) return m;
      const key = `${m.key}|${pm.key}`;
      const ck = `zpvcut_${key}`;
      if (!S.compCache[ck]) {
        const W = pm.canvas.width, Hh = pm.canvas.height;
        const c = document.createElement('canvas'); c.width = W; c.height = Hh; const g = c.getContext('2d');
        g.drawImage(m.img, 0, 0, W, Hh); const bd = g.getImageData(0, 0, W, Hh);
        g.clearRect(0, 0, W, Hh); g.drawImage(pm.area, 0, 0, W, Hh); const ad = g.getImageData(0, 0, W, Hh).data;
        const zd = pm.canvas.getContext('2d').getImageData(0, 0, W, Hh).data; const p = bd.data;
        const wd = pm.weight ? (() => { g.clearRect(0, 0, W, Hh); g.drawImage(pm.weight, 0, 0, W, Hh); return g.getImageData(0, 0, W, Hh).data; })() : null;
        for (let i = 0; i < p.length; i += 4) {
          if (wd) { const w = wd[i] / 255; if (w > 0) p[i] = p[i + 1] = p[i + 2] = Math.round(w * zd[i] + (1 - w) * p[i]); }   // feather (2026-10-17)
          else if (ad[i] > 64) { p[i] = p[i + 1] = p[i + 2] = zd[i]; }
          p[i + 3] = 255;
        }
        g.putImageData(bd, 0, 0);
        putComp(ck, c);
      }
      return { img: S.compCache[ck], key, kind: 'alpha', run: `${m.run || 'baza'} + podgląd strefy F${f}` };
    };
    {   // „👁 Podgląd klatek” of this stage covers the frame (and the stage has no current result): the cut-out shows it
        // (user report 2026-10-17: the MatAnyone preview was computed but the cut-out still showed the SAM3 core)
      const st = spvStage(); const pv = spvOf(st);
      if (pv && (pv.frames || []).includes(f) && spvWanted(st)) {
        const key = `${S.obj}_spv_${spvKey(st)}_${pv.created_at}_${f}`;
        const al = settledOverlay(key, `/api/object/${S.obj}/stage_preview/${st}/${f}?c=${encodeURIComponent(pv.created_at)}${spvEngQ(st)}`, f);
        if (al && al !== 'missing') return { img: al, key, kind: 'alpha', run: `podgląd ${spvLabel(st)} F${pv.frames[0]}–${pv.frames[pv.frames.length - 1]}` };
        if (!al) return 'loading';
      }
    }
    if (run === '') return withZonePv(sw ? wholeMatte(has12 ? 'core_fix' : 'core') : coreMatte(has12 ? 'core_fix' : (keyOf(f) ? 'key' : 'core')));
    if (run) {
      const key = `${S.obj}_alpha@${run}_${f}`;
      const al = loadOverlay(key, `/api/object/${S.obj}/frame/${f}?view=alpha&v=${encodeURIComponent(run)}`);
      if (al && al !== 'missing') { S.matteNote = ''; return withZonePv({ img: al, key, kind: 'alpha', run }); }
      if (!al) return 'loading';
      // bug 2026-10-17: the stage's version does not cover this frame – say so (it showed the bare background)
      const r = (S.runs || []).find((x) => x.run === run);
      const cov = r && r.coverage ? r.coverage.map(([a, b]) => (a === b ? `F${a}` : `F${a}–${b}`)).join(', ') : '';
      S.matteNote = `Wersja ${run}${cov ? ` obejmuje ${cov}` : ''} – klatka ${f} jest poza wynikiem tego etapu (policz etap na tym zakresie albo przejdź do policzonych klatek)`;
      return null;
    }
    const al = overlayFor('alpha', f);
    if (al && al !== 'missing') return { img: al, key: overlayKey('alpha', f), kind: 'alpha' };
    if (!al) return 'loading';
    return coreMatte(keyOf(f) ? 'key' : 'core');
  }
  window.MD_OBJ = { get name() { return S.obj; }, reload: () => loadStages().then(() => renderKeysLane()), matteFor,
                    get matteNote() { return S.matteNote || ''; }, maskHidden: () => selMaskHidden(),
                    stagePreview: (stage, btn) => stagePreview(stage, btn) };   // static/life.js, app.js
  function updateStageButtons() {
    const st = (id) => stageState(id).state;
    const set = (id, state, primaryWhen) => { const b = $(id); if (!b) return; b.disabled = !S.obj || state === 'locked'; b.classList.toggle('btn-primary', !!primaryWhen); b.classList.toggle('btn-secondary', !primaryWhen); };
    set('st2Crops', st('crops'), false);
    set('st3Rerun', st('core'), false);
    set('st3Ma', st('matanyone'), st('matanyone') !== 'done');
    { const b = $('st3Ma'); if (b) b.classList.toggle('btn-cta', !!S.obj && st('matanyone') === 'todo'); }
    set('st4Final', st('final'), st('final') !== 'done');
    set('lx6Run', st('ltx'), st('ltx') !== 'done');
    set('vmRun', st('videomama'), st('videomama') !== 'done');
    {   // a stage that already has a result: „Policz ponownie” – a new version replaces the stage result (the old one stays
        // on the version list) – user request 2026-10-17
      const sr = (S.info && S.info.stage_runs) || {};
      for (const [id, k, first] of [['st3Ma', 'matanyone', '▶ Przelicz MatAnyone2'], ['vmRun', 'videomama', '▶ Policz VideoMaMa'],
                                     ['lx6Run', 'ltx', '▶ Policz LTX-2.5']]) {
        const b = $(id); if (!b) continue;
        if (b.dataset.t0 == null) { b.dataset.t0 = b.title; b.dataset.l0 = b.textContent; }
        const prev = sr[k];
        b.textContent = prev ? '▶ Policz ponownie' : (b.dataset.l0 || first);
        b.title = prev ? `Policz etap od nowa – nowa wersja zastąpi wynik etapu ${prev} (${prev} zostaje na liście wersji, można do niego wrócić)` : b.dataset.t0;
      }
    }
    const ex = $('st5Export'); if (ex) ex.disabled = false;
    const one = $('st5ExportOne'); if (one) { one.disabled = !S.obj || st('export') === 'locked'; one.title = one.disabled ? (stageState('export').why || 'najpierw rdzeń (etap 1)') : 'Tylko wybrany obiekt (liczony jak w pełnym eksporcie), z wybranego źródła'; }
    const canCore = st('core') !== 'locked';
    for (const id of ['st3KeySave', 'st3KeyDel']) { const b = $(id); if (b) b.disabled = !S.obj; }
    const si = (S.info && S.info.stale_intervals) || [];
    const r = $('st3Rerun'); if (r) r.title = canCore ? `Przelicza tylko przedziały zmienione przez klucze: ${si.length ? si.map(([a, b]) => `F${a}–${b}`).join(', ') : 'brak zmian'}` : 'Najpierw „Licz ujęcie”';
  }

  // New object (user request 2026-10-15): a dialog in the middle of the window – name (required) + prompt (optional, EN).
  // Name rule = engine/objects.py _NAME. Enter = add, Esc / click outside = cancel; server errors stay in the dialog.
  const OBJ_NAME_RE = /^[A-Za-z0-9_\-]{1,64}$/;
  // rename in place (double-click on the name in the Objects palette, user request 2026-10-16)
  function renameInline(span, old) {
    if (span.querySelector('input')) return;
    const inp = document.createElement('input');
    inp.className = 'form-input ol-rename'; inp.value = old; inp.maxLength = 64;
    inp.title = 'Enter = zapisz, Esc = anuluj (litery bez polskich znaków, cyfry, _ i -)';
    const keep = span.innerHTML; span.innerHTML = ''; span.appendChild(inp);
    const row = span.closest('.ol-row'); if (row) row.draggable = false;
    inp.focus(); inp.select();
    let done = false;
    const finish = async (save) => {
      if (done) return; done = true;
      document.removeEventListener('keydown', onKey, true);
      const n = inp.value.trim();
      if (!save || !n || n === old) { span.innerHTML = keep; if (row) row.draggable = true; return; }
      if (!OBJ_NAME_RE.test(n)) { status('Nazwa: tylko litery bez polskich znaków, cyfry, _ i - (bez spacji)', 'err'); span.innerHTML = keep; return; }
      if ((S.objects || []).some((o) => o.name === n)) { status(`Obiekt „${n}” już istnieje`, 'err'); span.innerHTML = keep; return; }
      try {
        await api('PUT', `/api/object/${encodeURIComponent(old)}/rename`, { new_name: n });
      } catch (er) { status(`Nie zmieniono nazwy: ${er.message}`, 'err'); span.innerHTML = keep; return; }
      await refreshList(S.obj === old ? n : S.obj);
      status(`Zmieniono nazwę: „${old}” → „${n}” (wyeksportowane pliki zostają pod starą nazwą)`, 'ok');
    };
    // capture phase on the document (like the "New object" dialog): the viewer's shortcuts must not see the typing
    const onKey = (e) => {
      if (document.activeElement !== inp) return;
      e.stopPropagation();
      if (e.key === 'Enter') { e.preventDefault(); finish(true); } else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    };
    document.addEventListener('keydown', onKey, true);
    inp.addEventListener('mousedown', (e) => e.stopPropagation());
    inp.addEventListener('click', (e) => e.stopPropagation());
    inp.addEventListener('blur', () => finish(true));
  }

  function newObjectDialog() {
    if ($('newObjModal')) { $('newObjName').focus(); return; }
    if (!(V() && V().projectLoaded)) { toast('Najpierw otwórz projekt (📁 Projekt)'); return; }
    const bg = el(`<div class="md-modal-bg" id="newObjModal">
      <div class="md-modal" role="dialog" aria-labelledby="newObjTitle">
        <div class="md-modal-title" id="newObjTitle">＋ Nowy obiekt</div>
        <label class="md-field">Nazwa <span class="muted">(litery bez polskich znaków, cyfry, _ i -)</span>
          <input id="newObjName" class="form-input" placeholder="np. osoba" maxlength="64" autocomplete="off" title="Nazwa obiektu = nazwa folderu i plików masek"></label>
        <label class="md-field">Prompt SAM 3.1 <span class="muted">(opcjonalnie, EN – krótka fraza rzeczownikowa)</span>
          <input id="newObjPrompt" class="form-input" placeholder="np. felt puppet with antennae" autocomplete="off" title="Opis dla SAM 3.1 przy „Wykryj kandydatów”; można dodać później"></label>
        <div class="md-modal-err" id="newObjErr"></div>
        <div class="muted md-modal-tip">💡 Wycinaj od pierwszego planu (najbliżej kamery) w stronę tła – gotowe maski z przodu pomagają obiektom za nimi.</div>
        <div class="md-modal-btns">
          <button type="button" class="btn btn-secondary" id="newObjCancel" title="Zamknij bez dodawania (Esc)">Anuluj</button>
          <button type="button" class="btn btn-primary" id="newObjOk" title="Dodaj obiekt i przejdź do etapu 1 (Enter)">Dodaj obiekt</button>
        </div>
      </div></div>`);
    document.body.appendChild(bg);
    const nameEl = $('newObjName'), promptEl = $('newObjPrompt'), err = $('newObjErr'), ok = $('newObjOk');
    const close = () => { document.removeEventListener('keydown', onKey, true); bg.remove(); };
    const check = () => {
      const n = nameEl.value.trim();
      const msg = !n ? '' : !OBJ_NAME_RE.test(n) ? 'Nazwa: tylko litery bez polskich znaków, cyfry, _ i - (bez spacji)'
        : (S.objects || []).some((o) => o.name === n) ? `Obiekt „${n}” już istnieje` : '';
      err.textContent = msg; ok.disabled = !n || !!msg;
      return !ok.disabled;
    };
    const submit = async () => {
      if (!check()) { nameEl.focus(); return; }
      const name = nameEl.value.trim(), prompt = promptEl.value.trim();
      ok.disabled = true; err.textContent = '';
      try {
        await api('POST', '/api/object', { name, prompts: prompt ? [prompt] : [] });
      } catch (e) { err.textContent = `Nie dodano: ${e.message}`; ok.disabled = false; return; }
      close();
      await refreshList(name);
      setStage(ST.SAM3);
      status(`Dodano obiekt „${name}”. Kliknij na obiekt na klatce (punkt +) albo „Wykryj kandydatów”.`, 'ok');
    };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); close(); }
      else if (e.key === 'Enter') { e.stopPropagation(); e.preventDefault(); submit(); }
      else e.stopPropagation();   // typing here must not trigger the viewer's shortcuts
    };
    document.addEventListener('keydown', onKey, true);
    bg.addEventListener('mousedown', (e) => { if (e.target === bg) close(); });
    nameEl.addEventListener('input', check);
    $('newObjCancel').addEventListener('click', close);
    ok.addEventListener('click', submit);
    check();
    nameEl.focus();
  }
  // Delete an object (user request 2026-10-15): confirmation in the middle of the window with what goes where; the
  // exported masks only when ticked (both go to the project trash, recoverable by hand).
  async function deleteObjectDialog(name) {
    if ($('delObjModal')) return;
    if (document.body.classList.contains('job-busy')) { toast('Trwa liczenie – usuwanie po jego zakończeniu'); return; }
    let info;
    try { info = await api('GET', `/api/object/${encodeURIComponent(name)}/delete_info`); }
    catch (e) { toast(`Nie można sprawdzić obiektu „${name}”: ${e.message}`, 6000); return; }
    const ex = info.exports || { files: 0, mb: 0, dirs: [] };
    const bg = el(`<div class="md-modal-bg" id="delObjModal">
      <div class="md-modal md-modal-danger" role="dialog" aria-labelledby="delObjTitle">
        <div class="md-modal-title" id="delObjTitle">🗑 Usunąć obiekt „${esc(name)}”?</div>
        <ul class="md-modal-list">
          <li>Folder obiektu (klucze, strefy, rdzeń, wersje): <b>${info.folder.files} plików, ${info.folder.mb} MB</b> → kosz projektu
            <span class="muted">matte_desk/trash/objects/</span></li>
          <li>Obiekty za nim, które używały go jako obiektu z przodu, dostaną status „do przeliczenia”.</li>
          ${info.parts && info.parts.length ? `<li>Części (${info.parts.map(esc).join(', ')}) staną się osobnymi obiektami.</li>` : ''}
        </ul>
        <label class="md-check ${ex.files ? '' : 'disabled'}" title="${ex.files ? esc(ex.dirs.join('\n')) : 'Obiekt nie ma wyeksportowanych masek'}">
          <input type="checkbox" id="delObjExports" ${ex.files ? '' : 'disabled'} title="Przenieś też wyeksportowane maski obiektu do kosza projektu">
          <span>Usuń też wyeksportowane maski: <b>${ex.files} plików, ${ex.mb} MB</b> → <span class="muted">matte_desk/trash/exports/</span></span></label>
        <div class="muted md-modal-tip">Bez zaznaczenia maski zostają na dysku (np. użyte już w Nuke'u). Z kosza projektu można odzyskać ręcznie.</div>
        <div class="md-modal-err" id="delObjErr"></div>
        <div class="md-modal-btns">
          <button type="button" class="btn btn-secondary" id="delObjCancel" title="Zamknij bez usuwania (Esc)">Anuluj</button>
          <button type="button" class="btn btn-danger" id="delObjOk" title="Usuń obiekt">Usuń</button>
        </div>
      </div></div>`);
    document.body.appendChild(bg);
    const close = () => { document.removeEventListener('keydown', onKey, true); bg.remove(); };
    const go = async () => {
      const okb = $('delObjOk'); okb.disabled = true;
      const withExports = $('delObjExports').checked;
      try {
        const d = await api('DELETE', `/api/object/${encodeURIComponent(name)}${withExports ? '?exports=1' : ''}`);
        close();
        const tail = (x) => String(x || '').split(/[\\/]/).slice(-3).join('/');   // matte_desk/trash/... is enough
        toast(`Usunięto „${name}” → kosz: ${tail(d.trash)}${d.exports_moved ? ` · maski (${d.exports_moved}) → ${tail(d.exports_trash)}` : ''}`, 6000);
        await refreshList(S.obj === name ? undefined : S.obj);
      } catch (e) { $('delObjErr').textContent = `Nie usunięto: ${e.message}`; okb.disabled = false; }
    };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); close(); }
      else e.stopPropagation();   // no viewer shortcuts while the dialog is open; Enter does NOT delete
    };
    document.addEventListener('keydown', onKey, true);
    bg.addEventListener('mousedown', (e) => { if (e.target === bg) close(); });
    $('delObjCancel').addEventListener('click', close);
    $('delObjOk').addEventListener('click', go);
    $('delObjCancel').focus();
  }
  // „▶▶ Przelicz całość” (user decision 2026-10-15): points -> keys, crops when needed, core on the whole shot; again after
  // adding / removing keys = only the changed intervals (server cache). The mask is then visible in every frame.
  async function computeAll() {
    if (!S.obj) return;
    await ensurePointsSaved();
    status(`Przelicz całość: kadry (gdy trzeba) + rdzeń SAM 3.1 – ${window.MD_WA && window.MD_WA.range ? `zakres pracy ${window.MD_WA.label()}` : 'całe ujęcie'}…`);
    const d = await api('POST', `/api/object/${S.obj}/compute_all`);
    if (d.crops_recomputed) { clearOverlays(); await loadCrops(); } else invalidateIntervals(d.recomputed);
    await loadStages(); setView('core');
    const fl = Object.entries(d.crop_flags || {});
    const where = d.whole_shot ? 'na całym ujęciu' : `w F${d.frames[0]}–${d.frames[1]} (zakres pracy F${d.work_area[0]}–${d.work_area[1]}, do najbliższych kluczy)`;
    status(`Maska policzona ${where}: przeliczono ${d.recomputed.length} przedz., z cache ${d.cached.length}` +
      (d.crops_recomputed ? ` · kadry od nowa (${d.crops_reason}, odcinków: ${d.crop_segments})` : '') +
      (fl.length ? `\nUwagi kadrowania: ${fl.map(([k, v]) => `${k}: ${v}`).join('; ')}` : '') +
      (d.uncertain.length ? `\nNiepewne klatki: ${d.uncertain.slice(0, 12).join(', ')}` : '') +
      (d.suggested_keys.length ? `\nSugerowane klucze: ${d.suggested_keys.slice(0, 8).join(', ')}` : '') +
      frontMissingText(d) +
      '\nObejrzyj klatki (←/→); złą klatkę popraw punktami (klucz) i „▶ Przelicz klatkę”, potem „▶▶ Przelicz zmiany”.', frontMissingKind(d));
    toast(`Maska ${d.whole_shot ? 'w każdej klatce' : `w F${d.frames[0]}–${d.frames[1]}`} gotowa${d.uncertain.length ? ` – niepewnych: ${d.uncertain.length}` : ''}` +
      ((d.front_missing || []).length ? ` · ⚠ obiekt z przodu bez rdzenia: ${d.front_missing.map((m) => m.object).join(', ')}` : ''), 5000);
    renderPointsPalette();
  }
  async function refreshList(select) {
    const d = await api('GET', '/api/objects');
    S.objects = d.objects;
    const sel = $('objSelect');
    sel.innerHTML = d.objects.length ? d.objects.map((o) => `<option value="${esc(o.name)}">${o.depth_order}. ${o.part_of ? '↳ ' : ''}${esc(o.name)}${o.part_of ? ` (część ${esc(o.part_of)})` : ''}</option>`).join('')
                                     : '<option value="">— brak obiektów —</option>';
    if (select) sel.value = select;
    S.obj = sel.value || null;
    await loadObject();
  }
  function renderOrder() { renderOthersPanel(); }   // the list of objects lives in the layers panel (2026-10-15)
  // drag & drop of object rows (data-<key> = object name): onto the middle of a row = part of that object (like
  // „Relacja” → „część obiektu”), onto its upper / lower edge = depth order (a part dropped away from its parent's
  // family becomes a standalone object again)
  function wireObjDrag(box, rowSel, key) {
    let dragged = null;
    const byName = (n) => S.objects.find((o) => o.name === n);
    const zoneOf = (row, e) => {
      const r = row.getBoundingClientRect(), y = (e.clientY - r.top) / Math.max(1, r.height);
      return y < 0.28 ? 'before' : y > 0.72 ? 'after' : 'into';
    };
    const canNest = (src, dst) => {   // one level of parts: the target is a main object, the dragged one has no parts
      const s = byName(src), t = byName(dst);
      if (!s || !t || src === dst) return '';
      const parent = t.part_of || t.name;   // dropped on a part = part of its parent
      if (parent === src) return '';
      if (S.objects.some((o) => o.part_of === src)) return '';
      return parent;
    };
    const clearMarks = () => box.querySelectorAll(rowSel).forEach((r) => r.classList.remove('drop-before', 'drop-after', 'drop-into', 'drop-target'));
    // a part dropped anywhere that is not a row (outside the panel, empty space) = detached from its object (user request
    // 2026-10-15, like dragging a layer out of a group); Esc / a cancelled drag fires no drop -> nothing changes
    const detachTo = (src) => run('st1PartOf', async () => {
      await api('PUT', `/api/object/${src}/settings`, { part_of: '' });
      await refreshList(S.obj);
      status(`„${src}” odczepiony – jest znów samodzielnym obiektem`, 'ok');
    });
    const outOver = (e) => { e.preventDefault(); if (e.dataTransfer) e.dataTransfer.dropEffect = 'move'; };
    const outDrop = (e) => { e.preventDefault(); const src = dragged; dragged = null; clearMarks(); if (src) detachTo(src); };
    const outOff = () => { document.removeEventListener('dragover', outOver); document.removeEventListener('drop', outDrop); box.classList.remove('dragging-part'); };
    box.querySelectorAll(rowSel).forEach((row) => {
      row.addEventListener('dragstart', (e) => {
        dragged = row.dataset[key]; e.dataTransfer.effectAllowed = 'move'; try { e.dataTransfer.setData('text/plain', dragged); } catch (er) { /* old browsers */ }
        const s = byName(dragged);
        if (s && s.part_of) {
          document.addEventListener('dragover', outOver); document.addEventListener('drop', outDrop); box.classList.add('dragging-part');
          status(`Upuść „${dragged}” poza warstwami, żeby odczepić go od „${s.part_of}”`);
        }
      });
      row.addEventListener('dragend', () => { dragged = null; clearMarks(); outOff(); });
      row.addEventListener('dragover', (e) => {
        if (!dragged) return;
        e.stopPropagation();   // over a row: not "outside"
        if (dragged === row.dataset[key]) { e.preventDefault(); e.dataTransfer.dropEffect = 'none'; clearMarks(); return; }   // over itself: no-op
        e.preventDefault();
        let z = zoneOf(row, e);
        if (z === 'into' && !canNest(dragged, row.dataset[key])) z = e.clientY - row.getBoundingClientRect().top < row.offsetHeight / 2 ? 'before' : 'after';
        clearMarks(); row.classList.add(`drop-${z}`);
      });
      row.addEventListener('dragleave', () => row.classList.remove('drop-before', 'drop-after', 'drop-into'));
      row.addEventListener('drop', (e) => {
        e.preventDefault(); e.stopPropagation();   // a row handles it (never the "outside = detach" drop)
        const z = row.classList.contains('drop-into') ? 'into' : row.classList.contains('drop-after') ? 'after' : 'before';
        const src = dragged, target = row.dataset[key]; dragged = null;
        if (!src || src === target) return;
        if (z === 'into') {
          const parent = canNest(src, target); if (!parent) return;
          run('st1PartOf', async () => {
            await api('PUT', `/api/object/${src}/settings`, { part_of: parent });
            await refreshList(S.obj);
            status(`„${src}” jest częścią „${parent}” – eksport: ${parent}_full + osobne maski części`, 'ok');
          });
          return;
        }
        const s = byName(src), t = byName(target);
        const detach = !!(s && s.part_of && t && t.name !== s.part_of && t.part_of !== s.part_of);
        const names = S.objects.map((o) => o.name).filter((n) => n !== src);
        names.splice(names.indexOf(target) + (z === 'after' ? 1 : 0), 0, src);
        run('st1Order', async () => {
          if (detach) await api('PUT', `/api/object/${src}/settings`, { part_of: '' });
          const d = await api('PUT', '/api/objects/order', { names });
          await refreshList(S.obj);
          status(`${detach ? `„${src}” jest znów samodzielnym obiektem · ` : ''}Kolejność: ${names.join(' → ')}${d.to_recompute.length ? ` · do przeliczenia: ${d.to_recompute.join(', ')}` : ''}`, 'ok');
        });
      });
    });
  }
  // switch the object (top select or a click in the list of objects): same stage, same view layer when the object has it
  // (alpha -> core when it has only a core), else Plate – the chosen layer comes back on the next object that has it
  async function selectObject(name) {
    if (!name || name === S.obj) return;
    const want = S.userView || S.view || 'none';
    S.obj = name; S.vmPick = null;   // the stage engine's mask again
    const sel = $('objSelect'); if (sel && sel.value !== name) sel.value = name;
    await loadObject();
    if (S.stage === ST.PLATES) return;
    const cur = STAGES[S.stage];
    let n = S.stage;
    // Z14: the stage shown must exist for this object – a locked one jumps to its next stage (GET /stages -> next)
    if (cur && stepState(cur).state === 'locked' && S.info && S.info.next) {
      const nx = stageOfId(S.info.next);
      if (nx && nx.n !== S.stage) { toast(`${S.obj}: etap „${cur.label}” jeszcze zablokowany – przechodzę do ${nx.n}. ${nx.label}`); n = nx.n; }
    }
    setStage(n, { keepView: true });
    const order = { comp: ['comp', 'alpha', 'core'], alpha: ['alpha', 'core'], core: ['core'] }[want] || [];
    setView(order.find((v) => viewAllowed(v)) || 'none', true, false);
  }
  async function loadObject() {
    renderOrder();
    clearOverlays(); S.preview = {}; S.points = {}; S.strokes = {}; S.polys = {}; S.poly = null; S.crops = null; S.keys = []; S.viewRun = null; S.runs = []; S.cmpDiff = null; S.cmpUser = false; S.edgePv = null; S.epvParams = null; S.pvLast = {};
    if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.reset();
    if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.reset();
    if (window.MD_FT && window.MD_FT.ready) window.MD_FT.reset();
    S.exportSourceUser = false;
    fillObjectForm();
    if (S.obj) {
      await loadCrops().catch(() => {});
      await loadKeys().catch(() => {});
      await loadEdge().catch(() => {});
      await loadZones().catch(() => {});
      if (window.MD_BRUSH && window.MD_BRUSH.ready) await window.MD_BRUSH.load().catch(() => {});
      if (window.MD_FT && window.MD_FT.ready) await window.MD_FT.load().catch(() => {});
      await loadQA().catch(() => {});
    } else { renderKeysLane(); }
    await loadStages();
    await refreshOthersHas();
    V() && V().render();
  }
  function fillObjectForm() {
    const me = S.objects.find((o) => o.name === S.obj);
    const box = $('st1ObjBox'); if (!box) return;
    box.style.display = me ? '' : 'none';
    if (!me) return;
    $('st1Name').textContent = me.name;
    $('st1Prompt').value = (me.detect && me.detect.prompts || [])[0] || '';
    const ps = $('st1PartOf');
    const hasParts = S.objects.some((o) => o.part_of === me.name);
    ps.innerHTML = '<option value="">samodzielny obiekt</option>' + S.objects.filter((o) => o.name !== me.name && !o.part_of).map((o) => `<option value="${esc(o.name)}">część obiektu: ${esc(o.name)}</option>`).join('');
    ps.value = me.part_of || ''; ps.disabled = hasParts; ps.title = hasParts ? 'Ten obiekt ma własne części – jest obiektem głównym' : '';
    $('st1Status').textContent = me.tool && me.tool.status ? `Stan: ${me.tool.status}` : '';
  }

  async function loadCrops() {
    S.crops = null;
    try { const d = await api('GET', `/api/object/${S.obj}/crops`); if (d.exists) S.crops = d.segments; } catch (e) { /* none */ }
    renderCrops();
  }
  function renderCrops() {
    const box = $('st2List'); if (!box) return;
    if ($('st1CropsBox')) $('st1CropsBox').querySelector('summary').textContent = `Kadry SAM 3.1 (automatyczne)${S.crops ? ` – ${S.crops.length} odc.` : ''}`;
    if (!S.crops) { box.innerHTML = '<div class="muted">Jeszcze nie kadrowano.</div>'; return; }
    box.innerHTML = S.crops.map((s) => `<div class="row-item" data-f="${s.first}" title="Kliknij: skok do początku odcinka"><b>${esc(s.seg_id)}</b> F${s.first}–${s.last} · ${s.crop[2]} px · SAM ${(+s.scale_to_1008).toFixed(2)} px/px</div>`).join('');
    box.querySelectorAll('.row-item').forEach((r) => r.addEventListener('click', () => V().goToFrame(+r.dataset.f)));
  }

  async function loadKeys() {
    const d = await api('GET', `/api/object/${S.obj}/keys`);
    S.keys = d.keys;
    S.points = {};
    for (const k of d.keys) if ((k.positive || []).length || (k.negative || []).length)
      S.points[k.frame] = (k.positive || []).map((p) => [p[0], p[1], 1]).concat((k.negative || []).map((p) => [p[0], p[1], -1]));
    renderKeysLane(); renderKeyLists();
    sweepTick(); renderLoadBar();
  }
  function renderSegsLane() {   // Z2: crop segments + intervals to recompute (stages 2–4)
    const lane = $('timelineSegs'); const v = V(); if (!lane || !v) return;
    const on = S.obj && !S.auxMode && S.stage === ST.SAM3 && (S.crops || (S.info && (S.info.stale_intervals || []).length));
    lane.style.display = on ? 'block' : 'none';
    lane.innerHTML = '';
    if (!on) return;
    const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
    const pos = (a, b) => `left:${100 * (a - v.firstFrame) / n}%;width:${100 * (b - a + 1) / n}%`;
    (S.crops || []).forEach((sg, i) => {
      const d = document.createElement('div'); d.className = `seg ${i % 2 ? 'b' : 'a'}`; d.style.cssText = pos(sg.first, sg.last);
      d.title = `${sg.seg_id}: F${sg.first}–${sg.last} (kadr ${sg.crop[2]} px)`;
      d.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(sg.first); });
      lane.appendChild(d);
    });
    for (const [a, b] of (S.info && S.info.stale_intervals) || []) {
      const d = document.createElement('div'); d.className = 'stale'; d.style.cssText = pos(a, b); lane.appendChild(d);
    }
    const st = (S.info && S.info.stale_intervals) || [];
    lane.title = `Odcinki kadru: ${(S.crops || []).length}${st.length ? ` · do przeliczenia: ${st.map(([a, b]) => `F${a}–${b}`).join(', ')}` : ''}. Klik = skok`;
  }
  // Z6: one place for "Usuń klucz" (stage-3 button, playback bar 🔑✕, Del / Backspace, double-click on the marker)
  async function deleteKeyAt(f) {
    if (!S.obj || S.auxMode) return false;
    const k = S.keys.find((x) => x.frame === f);
    if (!k) { status(`Brak klucza w klatce ${f}`, 'warn'); return false; }
    if (k.author === 'user' && !(await V().uiConfirm(`Usunąć klucz użytkownika w klatce ${f}?\nCtrl+Z cofa.`, { ok: 'Usuń' }))) return false;
    await api('DELETE', `/api/object/${S.obj}/keys/${f}?confirm=true`);
    S.points[f] = []; invalidateFrame(f);
    status(`Usunięto klucz ${f} (Ctrl+Z cofa)`, 'ok'); await loadKeys(); await loadStages(); V().render();
    return true;
  }
  // the key of the current stage in frame f (user report 2026-10-16: in stage ViTMatte 🔑✕ / Delete went for the SAM3 key):
  // ViTMatte = zone key (diamond), CoTracker = brush correction keys, else the SAM3 key
  const stageKeyMod = () => (S.stage === ST.VITMATTE && window.MD_ZONES && window.MD_ZONES.ready ? window.MD_ZONES
    : S.stage === ST.BRUSH && window.MD_BRUSH && window.MD_BRUSH.ready ? window.MD_BRUSH
      : BAND_STAGES().includes(S.stage) && window.MD_BANDPTS && window.MD_BANDPTS.ready ? window.MD_BANDPTS
      : S.stage === ST.FINAL_TOUCH && window.MD_FT && window.MD_FT.ready ? window.MD_FT : null);
  function stageHasKey(f) { const m = stageKeyMod(); return m && m.hasKeyAt ? m.hasKeyAt(f) : S.keys.some((k) => k.frame === f); }
  function deleteStageKey(f) {
    const m = stageKeyMod();
    if ((m === window.MD_ZONES || m === window.MD_BANDPTS) && m) return m.deleteKeyAt(f);
    if (m === window.MD_BRUSH && m) return m.deleteKeysAt(f);
    if (m === window.MD_FT && m) return m.deleteKeysAt(f);
    return deleteKeyAt(f);
  }
  // 🔑✕ follows the keys of the current stage right after any change (user report 2026-10-22: a new brush / zone /
  // Final Touch key could not be deleted until the playhead moved – the lane was redrawn only on a frame change)
  function updateDelKeyBtn() {
    const bdk = $('btnDelKey'); const v = V(); if (!bdk || !v) return;
    const has = !!(S.obj && !S.auxMode && stageHasKey(v.currentFrame));
    bdk.disabled = !has; bdk.classList.toggle('has-key', has);
  }
  function renderKeysLane() {
    renderSegsLane();
    if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.renderLane();
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.renderLane();
    if (window.MD_FT && window.MD_FT.ready) window.MD_FT.renderLane();
    const lane = $('timelineKeys'); const v = V(); if (!lane || !v) return;
    lane.innerHTML = '';
    const range = Math.max(1, v.lastFrame - v.firstFrame);
    for (const k of S.keys) {
      const m = document.createElement('div');
      const hid = window.MD_LIFE && !window.MD_LIFE.visible(k.frame);   // outside the object's life: not used
      m.className = `key-mark ${k.author} ${k.kind || ''} ${k.frame === v.currentFrame ? 'cur' : ''}${hid ? ' life-hidden' : ''}`;
      m.style.left = `${100 * (k.frame - v.firstFrame) / range}%`;
      m.title = `${hid ? 'POZA ZAKRESEM ŻYCIA (nieużywany) · ' : ''}Klucz F${k.frame} · ${k.author === 'user' ? 'użytkownik' : 'automatyczny'} · ${{ points: 'punkty', brush: 'pędzel', polygon: 'wielokąt', mask: 'maska', correction: 'korekta' }[k.kind] || k.kind || ''} · klik = skok · dwuklik = usuń klucz`;
      m.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(k.frame); });
      m.addEventListener('dblclick', (e) => { e.stopPropagation(); e.preventDefault(); v.goToFrame(k.frame); run('btnDelKey', () => deleteKeyAt(k.frame)); });   // Z6
      lane.appendChild(m);
    }
    updateDelKeyBtn();
    renderLoadBar();
    for (const f of pendingPointFrames()) {
      if (S.keys.some((k) => k.frame === f) && !S.pointsFailed[f]) continue;
      const m = document.createElement('div');
      m.className = `key-mark pending ${f === v.currentFrame ? 'cur' : ''}`;
      m.style.left = `${100 * (f - v.firstFrame) / range}%`;
      m.title = `F${f}: punkty jeszcze NIE są kluczem${S.pointsFailed[f] ? ` (${S.pointsFailed[f]})` : ' (zapis trwa)'}`;
      m.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
      lane.appendChild(m);
    }
    const badge = $('frameKeyBadge');
    if (badge) {
      const k = S.keys.find((x) => x.frame === v.currentFrame);
      const pts = S.points[v.currentFrame] || [];
      badge.style.display = k ? '' : 'none';
      if (k) badge.textContent = `🔑 ${k.kind === 'points' ? `punkty ${pts.filter((p) => p[2] > 0).length}+ / ${pts.filter((p) => p[2] < 0).length}−` : ({ brush: 'pędzel', polygon: 'wielokąt', correction: 'korekta', mask: k.author === 'auto' ? 'auto' : 'maska' }[k.kind] || k.kind)}`;
    }
  }
  function renderKeyLists() {
    const items = S.keys.map((k) => `<span class="key-chip ${k.author}" data-f="${k.frame}" title="skok do klatki">${k.frame}${k.kind === 'points' ? '•' : ''}${k.author === 'auto' ? 'a' : ''}</span>`).join(' ') || '<span class="muted">brak</span>';
    for (const id of ['st1Keys']) { const b = $(id); if (!b) continue; b.innerHTML = `Klucze: ${items}`; b.querySelectorAll('.key-chip').forEach((c) => c.addEventListener('click', () => V().goToFrame(+c.dataset.f))); }
    updatePointsBox();
  }
  function keyStep(dir) {
    const v = V(); if (!v || !S.keys.length) return;
    const fs = S.keys.map((k) => k.frame).sort((a, b) => a - b);
    const nx = dir > 0 ? fs.find((f) => f > v.currentFrame) : fs.slice().reverse().find((f) => f < v.currentFrame);
    if (nx !== undefined) v.goToFrame(nx); else toast(dir > 0 ? 'To ostatni klucz' : 'To pierwszy klucz');
  }

  // „Kopiuj punkty pasa z”: outline zones of stage 5 with points and the other stages' bands with points (2026-10-22)
  // the shared object band (2026-10-22) is one source; detached stages / zones with their own points are the others
  async function fillBandSources(pfx) {
    const sel = $(`${pfx}BandSrc`), btn = $(`${pfx}BandCopy`); if (!sel || !S.obj) return;
    const me = { ep: 'matanyone', vm: 'videomama', lx: 'ltx' }[pfx];
    const label = { matanyone: `etap ${ST.MATANYONE} MatAnyone2`, videomama: `etap ${ST.VIDEOMAMA} VideoMaMa`, ltx: `etap ${ST.LTX} LTX-2.5` };
    const opts = [];
    try {
      const d = await api('GET', `/api/object/${S.obj}/band`);
      if (d.points) opts.push('<option value="band">pas obiektu (wspólny)</option>');
      for (const o of d.owners || []) {
        if (o.mode !== 'own' || o.key === me || !o.points) continue;
        opts.push(`<option value="${o.key}">${o.key.startsWith('zone:') ? `strefa ${o.key.slice(5)} (etap ${ST.VITMATTE})` : label[o.key]} – własny pas</option>`);
      }
    } catch (e) { /* no band */ }
    sel.innerHTML = opts.join('') || '<option value="">— brak punktów w innych etapach —</option>';
    if (btn) btn.disabled = !opts.length;
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) window.MD_BANDPTS.renderMode();   // badge + copy row only when detached
  }

  async function loadEdge() {
    const d = await api('GET', `/api/object/${S.obj}/edge_modes`);
    $('st4Edge').innerHTML = d.modes.map((m) => `<option value="${m.mode}" ${m.enabled ? '' : 'disabled'} title="${esc(m.why || '')}">${esc(m.label)}${m.enabled ? '' : ' (niedostępne)'}</option>`).join('');
    $('st4Edge').value = d.current;
    $('st4EdgeWhy').textContent = d.reason || '';
    const o = S.objects.find((x) => x.name === S.obj) || {};
    S.edgeLegacy = (o.band_out_px == null && o.band_in_px == null) ? { b: o.band_scale ?? 1, s: o.shift_px ?? 0 } : null;
    S.edgePxTouched = false;
    for (const [id, k] of [['o4Out', 'band_out_px'], ['o4In', 'band_in_px'], ['o4Feather', 'band_in_feather_px'], ['o4OutFeather', 'band_out_feather_px']]) { $(id).value = o[k] ?? 0; $(`${id}V`).value = o[k] ?? 0; }
    fillBandSources('ep').catch(() => {});
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) { window.MD_BANDPTS.dirty(); window.MD_BANDPTS.onFrame(); }
    const lg = S.edgeLegacy && (+S.edgeLegacy.b !== 1 || +S.edgeLegacy.s !== 0);
    $('o4PxHint').textContent = lg ? `Obiekt ma stare ustawienie (szerokość ×${S.edgeLegacy.b}, środek ${S.edgeLegacy.s} px) – ruszenie suwaka przełącza na px`
      : '0 / 0 = sam automat (pas wg rozmycia, 3–24 px w każdą stronę)';
    $('o4Smooth').value = o.temporal_smoothing || 'normal'; $('o4Lupa').value = o.thin_lens || 'auto';
    $('o4MaScale').value = o.ma_scale_mode || 'auto';
    $('o4MaScaleL').style.display = d.modes.some((m) => m.mode === 'matanyone' && m.enabled) ? '' : 'none';
    $('o4Backlit').checked = o.use_backlit !== false;
    $('o4BacklitL').style.display = d.modes.some((m) => m.mode === 'backlit') ? '' : 'none';
    const wh = wholeOf(S.obj);
    $('st5ExportWhole').style.display = wh ? '' : 'none';
    $('st5ExportWhole').textContent = wh ? `⇪ Całość (${wh}_full)` : '⇪ Całość';
  }
  function wholeOf(name) {
    const me = S.objects.find((o) => o.name === name); if (!me) return null;
    if (me.part_of) return me.part_of;
    return S.objects.some((o) => o.part_of === name) ? name : null;
  }

  async function loadQA() {
    S.qa = null; S.edgeFix = null;
    const ebox = $('st4EdgeFix'); if (ebox) ebox.innerHTML = '';
    const tr = $('timelineTrack'); if (tr) tr.querySelectorAll('.qa-cell').forEach((x) => x.remove());
    const ban = $('st4Banner'); if (ban) ban.style.display = 'none';
    const sum = $('st4QA'); if (sum) sum.textContent = '';
    if (!S.obj) return;
    if (S.stage === ST.EXPERIMENTS && !S.exRun) {   // stage 5: QA of the chosen base only (user decision 2026-10-16)
      if (sum) sum.textContent = 'QA: wybierz maskę bazową w palecie „Eksperymenty” – QA i propozycje krawędzi policzą się na niej';
      return;
    }
    const qv = S.stage === ST.EXPERIMENTS && S.exRun ? `?v=${encodeURIComponent(S.exRun)}` : '';   // the chosen base (2026-10-16)
    let d; try { d = await api('GET', `/api/object/${S.obj}/qa${qv}`); } catch (e) { d = null; }
    if (!d || !d.colours) { if (sum) sum.textContent = S.stage === ST.EXPERIMENTS ? 'QA: wybierz maskę bazową w palecie „Eksperymenty” i „▶ Analizuj”' : 'QA: brak wyniku finalnego'; return; }   // {} = no version yet (Z17)
    S.qa = d;
    const v = V(); const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
    const col = { green: 'rgba(34,197,94,0.55)', yellow: 'rgba(234,179,8,0.75)', red: 'rgba(239,68,68,0.85)' };
    for (const [f, c] of Object.entries(d.colours)) {
      const x = document.createElement('div'); x.className = 'qa-cell';
      x.style.cssText = `left:${100 * (f - v.firstFrame) / n}%;width:${100 / n}%;background:${col[c]};`;
      const fl = d.flags[f] || []; x.title = `F${f}: ${fl.map((q) => q[1]).join('; ') || 'OK'}`; tr && tr.appendChild(x);
    }
    if (d.banner && ban) { ban.textContent = d.banner; ban.style.display = 'block'; }
    const bad = Object.values(d.colours).filter((c) => c !== 'green').length;
    if (sum) sum.textContent = `QA: migotanie (mediana) ${d.flicker_median} ${d.flicker_median <= 0.004 ? '✓' : '✗ > 0.004'} · flagi: ${bad} kl.`;
    renderEdgeFix(d);
    loadRuns().catch(() => {});
  }
  // ---- Z2: versions runs/v### – preview without recomputing, set active, B = active <-> previous
  async function loadRuns() {
    const box = $('st4Runs'); if (!box || !S.obj) return;
    const d = await api('GET', `/api/object/${S.obj}/runs`);
    S.runs = d.runs; S.activeRun = d.active_run;
    if (S.viewRun && !d.runs.some((r) => r.run === S.viewRun)) S.viewRun = null;
    box.innerHTML = d.runs.length ? d.runs.slice().reverse().map((r) => `<div class="row-item run-row ${(S.viewRun || d.active_run) === r.run ? 'cur' : ''}" data-run="${r.run}" style="cursor:pointer" title="Klik = pokaż tę wersję (nic nie przelicza)">
      <span><b>${r.run}</b>${r.active ? ' ★' : ''} · ${esc(r.edge_mode || '')}${r.stage ? ` · ${esc(stageOfRun(r))}` : ''} · ${r.seconds ?? '?'} s · QA ${r.gate_ok == null ? '–' : (r.gate_ok ? 'OK' : 'flagi')} · migotanie ${r.flicker_median ?? '–'} · flagi ${r.n_flags}${r.edge_corrected ? ' · poprawka krawędzi' : ''}${r.motion_blur ? ` · motion blur ×${r.motion_blur.strength}, ${r.motion_blur.shutter_deg}° (z ${esc(r.motion_blur.base_run || '?')})` : ''}${r.base_run ? ` · z bazy ${esc(r.base_run)}${r.base_outdated ? ' <span class="warn" title="Ten etap ma już nowszy wynik">(baza nieaktualna)</span>' : ''}` : ''}</span>
      ${r.active ? '' : `<button class="btn btn-ghost btn-sm" data-act="${r.run}" title="Eksport i obiekty zależne użyją tej wersji">Ustaw jako aktywną</button>`}</div>${window.MD_ZONES && window.MD_ZONES.ready ? window.MD_ZONES.runDetailsHtml(r) : ''}`).join('') : 'brak wersji';
    box.querySelectorAll('.run-row').forEach((el) => el.addEventListener('click', (e) => {
      if (e.target.dataset.act || e.target.closest('.run-zones')) return;
      showRun(el.dataset.run === d.active_run ? null : el.dataset.run);
    }));
    box.querySelectorAll('button[data-act]').forEach((b) => b.addEventListener('click', () => run('st4Runs', async () => {
      const r = await api('PUT', `/api/object/${S.obj}/active_run`, { run: b.dataset.act });
      S.viewRun = null; clearOverlays(); await refreshList(S.obj); await loadStages();
      status(`Aktywna wersja: ${r.active_run}${r.dependents_to_recompute.length ? ` · do przeliczenia: ${r.dependents_to_recompute.join(', ')}` : ''}`, 'ok');
      loadQA();
    })));
    updateRunHud();
    renderCmp();   // stage-4 comparison lists follow the versions
  }
  function showRun(runName) {
    S.viewRun = runName;
    if (runName && S.view !== 'alpha' && S.view !== 'comp') setView('alpha', false);
    V().render();
    const box = $('st4Runs'); if (box) box.querySelectorAll('.run-row').forEach((el) => el.classList.toggle('cur', el.dataset.run === (runName || S.activeRun)));
  }
  function updateRunHud() { V().render(); }
  function toggleRunB() {
    if (!versionStage()) return;                    // the core has no versions
    if (S.stage === ST.VITMATTE && $('st4CmpA') && $('st4CmpA').value && $('st4CmpB').value && $('st4CmpA').value !== $('st4CmpB').value) {
      cmpShow(S.viewRun === $('st4CmpA').value ? 'b' : 'a'); toast(`Porównanie: ${S.viewRun}`); return;
    }
    const runs = (S.runs || []).map((r) => r.run);
    if (runs.length < 2 || !S.activeRun) { toast('Jedna wersja – nie ma z czym porównać'); return; }
    const i = runs.indexOf(S.activeRun);
    const prev = runs[i > 0 ? i - 1 : runs.length - 1];
    showRun(S.viewRun ? null : prev);
    toast(`${S.activeRun} ↔ ${prev} · na ekranie: ${S.viewRun || S.activeRun}`);
  }
  // ---- Z27 pkt 1: edge correction proposal (no automatic second round)
  function edgePiecesFor(f) {
    const ef = S.edgeFix; if (!ef) return null;
    if (f in ef.cache) return ef.cache[f];
    ef.cache[f] = null;
    api('GET', `/api/object/${S.obj}/edge_pieces/${f}?v=${ef.run}`).then((d) => { ef.cache[f] = d.pieces || []; if (V().currentFrame === f) V().render(); }).catch(() => { ef.cache[f] = []; });
    return null;
  }
  function renderEdgeFix(d) {
    const box = $('st4EdgeFix'); if (!box) return;
    const e = (d && d.edge) || {};
    const activeRun = S.objects.find((o) => o.name === S.obj)?.tool?.active_run || null;
    if (!e.fix_available && e.fix_why && (e.pieces || []).length) {   // proposals measured, fix not for this version
      S.edgeFix = { run: d.run || activeRun, pieces: e.pieces, skip: new Set(), cache: {}, dismissed: false, available: false, why: e.fix_why };
      box.innerHTML = `<div class="sec-title">Krawędź – propozycje (${e.pieces.length})</div><div class="muted">Mediana ${e.before} px. ${esc(e.fix_why)}</div>`;
      return;   // (only backlit bases say why: their edge is not corrected)
    }
    if (!e.fix_available || !(e.pieces || []).length) {
      S.edgeFix = null;
      box.innerHTML = e.corrected ? `<div class="muted">Poprawka krawędzi zastosowana (z ${esc(e.base_run || '?')}): mediana ${e.before ?? '?'} → ${e.after} px</div>`
        : (e.pieces_to_fix === 0 ? '<div class="muted">Krawędź: brak odcinków do poprawy</div>' : '');
      return;
    }
    S.edgeFix = { run: d.run || activeRun, pieces: e.pieces, skip: new Set(), cache: {}, dismissed: false, available: true };
    if (!S.edgeFix.run) { S.edgeFix = null; box.innerHTML = ''; return; }
    const rows = e.pieces.map((p) => `<label class="chk edge-piece" data-first="${p.first}" title="F${p.first}–${p.last}, ${p.n_frames} kl.; klik na nazwę = skok do klatki">
      <input type="checkbox" data-piece="${esc(p.id)}" checked> <span style="color:${p.kind === 'halo' ? '#ff5050' : '#4c8cff'}">■</span>
      <a href="#" data-jump="${p.first}">${esc(p.id)}</a> F${p.first}–${p.last} · ${p.median_px > 0 ? '+' : ''}${p.median_px} px ${p.kind === 'halo' ? '(halo → zwęź)' : '(ucięcie → poszerz)'}</label>`).join('');
    box.innerHTML = `<div class="sec-title">Poprawka krawędzi</div>
      <div class="muted">${e.pieces.length} odcinków, mediana ${e.before} px. Czerwony = halo (zwęź), niebieski = ucięcie (poszerz). Odznacz odcinki na fałszywych krawędziach (wzór, tło o podobnej jasności).</div>
      <div class="edge-pieces" style="max-height:160px;overflow:auto">${rows}</div>
      <div class="row"><button class="btn btn-primary btn-sm" id="st4EdgeApply" title="Nowa wersja z poprawką zaznaczonych odcinków (aktywna wersja zostaje do porównania)">✔ Zastosuj poprawkę krawędzi</button>
      <button class="btn btn-ghost btn-sm" id="st4EdgeSkip" title="Zostaw wersję bez poprawki">Pomiń</button></div>`;
    box.querySelectorAll('input[data-piece]').forEach((c) => c.addEventListener('change', () => {
      if (c.checked) S.edgeFix.skip.delete(c.dataset.piece); else S.edgeFix.skip.add(c.dataset.piece);
      V().render();
    }));
    box.querySelectorAll('.edge-piece').forEach((row) => {   // hovering a row highlights its piece on the viewer
      const id = row.querySelector('input[data-piece]').dataset.piece;
      row.addEventListener('mouseenter', () => { if (S.edgeFix) { S.edgeFix.hover = id; V().render(); } });
      row.addEventListener('mouseleave', () => { if (S.edgeFix && S.edgeFix.hover === id) { S.edgeFix.hover = null; V().render(); } });
    });
    box.querySelectorAll('a[data-jump]').forEach((a) => a.addEventListener('click', (ev) => { ev.preventDefault(); V().goToFrame(+a.dataset.jump); }));
    $('st4EdgeSkip').addEventListener('click', () => { S.edgeFix.dismissed = true; box.innerHTML = '<div class="muted">Poprawka krawędzi pominięta (wersja bez zmian)</div>'; V().render(); });
    $('st4EdgeApply').addEventListener('click', () => run('st4EdgeApply', async () => {
      // 2026-10-16: the correction shifts the edge of THIS base and computes it with the palette's edge engine
      if (S.edgeFix && S.exRun !== S.edgeFix.run) S.exRun = S.edgeFix.run;
      await expCompute(true);
    }));
  }
  // legend of the edge pieces; built here too, so a cached old index.html without it still gets one
  function edgeLegendEl() {
    let lg = $('edgeLegend'); if (lg) return lg;
    const w = $('canvasWrapper'); if (!w) return null;
    lg = el(`<div class="edge-legend" id="edgeLegend" style="display:none">
      <span><i style="background:#ff3c3c"></i>halo → zwęź</span><span><i style="background:#3c8cff"></i>ucięcie → poszerz</span>
      <span><i class="off"></i>odznaczony</span><span class="muted">klik na odcinek = odznacz / zaznacz</span></div>`);
    w.appendChild(lg); return lg;
  }
  // piece of the current frame under the cursor (within 8 screen px), or null
  function edgePieceAt(e) {
    if (!(S.obj && !S.auxMode && S.stage === ST.EXPERIMENTS && S.edgeFix && !S.edgeFix.dismissed)) return null;
    const v = V(); const ep = S.edgeFix.cache[v.currentFrame]; if (!ep || !ep.length) return null;
    const img = v.getCurrentImage && v.getCurrentImage(); const ps = plateSize(); if (!img || !ps) return null;
    const k = img.width / ps[0]; const r = v.canvas.getBoundingClientRect();
    const x = (e.clientX - r.left - v.panX) / v.zoom / k, y = (e.clientY - r.top - v.panY) / v.zoom / k;
    const tol = 8 / (v.zoom * k);
    let best = null, bd = tol;
    for (const pc of ep) {
      const p = pc.pts;
      for (let i = 0; i < p.length; i++) {
        const a = p[i], b = p[Math.min(i + 1, p.length - 1)];
        const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
        const t = L ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L)) : 0;
        const d = Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
        if (d < bd) { bd = d; best = pc; }
      }
    }
    return best;
  }
  function toggleEdgePiece(pc) {
    const ef = S.edgeFix; const off = !ef.skip.has(pc.id);
    if (off) ef.skip.add(pc.id); else ef.skip.delete(pc.id);
    const c = document.querySelector(`#st4EdgeFix input[data-piece="${CSS.escape(pc.id)}"]`);
    if (c) {
      c.checked = !off;
      const row = c.closest('.edge-piece'); row.scrollIntoView({ block: 'nearest' });
      row.classList.add('flash'); setTimeout(() => row.classList.remove('flash'), 900);
    }
    status(`${pc.id}: ${off ? 'odznaczony (bez poprawki)' : 'zaznaczony'} · ${pc.kind === 'halo' ? 'halo → zwęź' : 'ucięcie → poszerz'}`, 'ok');
    V().render();
  }

  function nextFlag() {
    if (!S.qa) { toast('Brak QA – najpierw „Przelicz finalnie”'); return; }
    const v = V(); const fl = Object.entries(S.qa.colours).filter(([f, c]) => c !== 'green').map(([f]) => +f).sort((a, b) => a - b);
    const nx = fl.find((f) => f > v.currentFrame) ?? fl[0];
    if (nx !== undefined) { v.goToFrame(nx); status(`F${nx}: ${(S.qa.flags[nx] || []).map((q) => q[1]).join('; ')}`); } else toast('Brak flag – QA zielone');
  }

  async function loadZones() {   // the zone panel (static/zones.js, plan step 5) owns the list
    if (window.MD_ZONES && window.MD_ZONES.ready) await window.MD_ZONES.load();
  }

  // ------------------------------------------------------------------------------------------------ points (stage 1)
  function updatePointsBox() {
    const v = V(); if (!v || !$('st1PtsInfo')) return;
    const f = v.currentFrame; const pts = S.points[f] || [];
    const pos = pts.filter((p) => p[2] > 0).length, neg = pts.length - pos;
    const k = S.keys.find((x) => x.frame === f);
    const busy = S.pointsBusy[f];
    $('st1PtsInfo').innerHTML = `Klatka <b>${f}</b>: ${pts.length ? `${pos} punkt(y) +, ${neg} −` : 'brak punktów'}` +
      (k ? ` · <span class="ok">🔑 klucz (${{ points: 'z punktów', mask: k.author === 'auto' ? 'kandydat' : 'maska', brush: 'pędzel', polygon: 'wielokąt', correction: 'korekta' }[k.kind] || k.kind})</span>` : '') +
      (busy ? ' · <span class="busy">⏳ SAM 3.1 liczy…</span>' : '') +
      (!busy && (S.pointsDirty[f] || S.pointsFailed[f]) && pts.length ? ' · <span class="err">⚠ niezapisane – jeszcze nie klucz</span>' : '');
    $('st1PtsClear').disabled = !pts.length;
    const dirty = !!(S.pointsDirty[f] || S.pointsFailed[f]);
    const promptOnly = !pts.length && !!promptField() && !k;   // prompt alone (e.g. from the „Nowy obiekt” dialog)
    $('st1PtsRedo').disabled = !!busy || (!dirty && !pts.length && !promptField());
    $('st1PtsRedo').classList.toggle('btn-cta', (dirty || promptOnly) && !busy);
    renderPointsPalette();
  }
  // Points (user request 2026-10-15): a click only places / removes a point; SAM3 runs on „▶ Przelicz klatkę” (palette, panel, Enter).
  // Points not computed yet are orange; before any computation of the object they are computed anyway (ensurePointsSaved).
  function markPoints(f) {
    S.pointsDirty[f] = true;
    updatePointsBox(); V().render(); renderKeysLane();
  }
  const promptField = () => { const e = $('st1Prompt'); return e ? e.value.trim() : null; };
  function computePoints(f) {
    if (!S.obj) return;
    // recompute on demand: points, or the prompt alone (e.g. after „Wyczyść punkty” + a new prompt – bug 2026-10-15)
    if (!S.pointsDirty[f] && !S.pointsFailed[f]) S.pointsDirty[f] = !!(S.points[f] || []).length || !!promptField();
    if (!S.pointsDirty[f]) { status('Brak punktów i promptu – postaw punkt albo wpisz prompt', 'warn'); return; }
    delete S.pointsFailed[f];
    flushPoints(f);
  }
  function clearPoints(f) {
    if (!(S.points[f] || []).length) return;
    S.points[f] = []; markPoints(f);
    status(`F${f}: punkty wyczyszczone – „▶ Przelicz klatkę” usunie klucz z punktów tej klatki`, 'warn');
  }
  const ptsPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_points_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_points_palette', JSON.stringify(p)); else localStorage.removeItem('md_points_palette'); } catch (e) { /* storage blocked */ } },
  };
  function buildPointsPalette() {   // once, at start (hidden): wire() attaches the prompt / detect handlers to its controls
    const w = $('canvasWrapper'); if (!w) return null;
    let box = $('pointsPalette');
    if (!box) {
      box = el(`<div class="pts-palette" id="pointsPalette" style="display:none">
        <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Punkty SAM 3.1</div>
        <div class="pp-empty">
          <button type="button" class="btn btn-primary" id="ppFirstObj" title="Nazwij obiekt i (opcjonalnie) opisz go promptem – potem punkty / prompt w klatce i „▶ Przelicz klatkę”">＋ Dodaj pierwszy obiekt</button>
          <div class="muted">W ujęciu nie ma jeszcze obiektów. Dodaj obiekt, potem zaznacz go punktami albo opisz promptem i policz klatkę.</div>
        </div>
        <div class="pp-info" id="ppInfo"></div>
        <div class="pp-tools">
          <button type="button" class="tool-btn pp-tool" data-t="pt_add" title="Punkt + [G]: klik = DODAJ to, co wskazujesz (PPM = punkt −; Alt = łapka); klik na istniejący punkt = usuń. Kliknij ponownie, żeby wyłączyć"><span class="tool-icon">🎯+</span><span class="tool-label">Punkt + G</span></button>
          <button type="button" class="tool-btn pp-tool" data-t="pt_remove" title="Punkt − [R]: klik = ODEJMIJ to, co wskazujesz (PPM = punkt +; Alt = łapka); klik na istniejący punkt = usuń"><span class="tool-icon">🎯−</span><span class="tool-label">Punkt − R</span></button>
        </div>
        <input id="st1Prompt" class="form-input" placeholder="prompt EN (opcjonalnie), np. woman with long hair" autocomplete="off"
          title="Prompt = baza maski klatki; punkty + dodają, punkty − odejmują. Zapisywany w obiekcie (Enter / wyjście z pola)">
        <div class="pp-btns">
          <button type="button" class="btn btn-primary btn-sm" id="ppCompute" title="Policz maskę tej klatki: prompt (baza) + punkty + (dodaj) − punkty − (odejmij); zapis jako klucz (Enter)">▶ Przelicz klatkę</button>
          <button type="button" class="btn btn-secondary btn-sm" id="ppClear" title="Usuń wszystkie punkty tej klatki (klucz zniknie po „Przelicz klatkę”)">✕ Wyczyść punkty</button>
        </div>
        <button type="button" class="btn btn-primary btn-sm" id="ppAll" title="Kadry (automatycznie, gdy trzeba) + rdzeń SAM 3.1 na całym ujęciu – maska w każdej klatce. Po zmianie kluczy liczy tylko zmienione przedziały">▶▶ Przelicz całość</button>
        <div class="pp-hint muted">Klik = punkt + · PPM = punkt − · Alt = łapka · klik na punkt = usuń · Delete = ostatni</div>
      </div>`);
      w.appendChild(box);
      makeFloating(box, w, '.pp-head', ptsPos);
      $('ppFirstObj').addEventListener('click', () => newObjectDialog());
      $('ppCompute').addEventListener('click', () => computePoints(V().currentFrame));
      $('ppClear').addEventListener('click', () => clearPoints(V().currentFrame));
      $('ppAll').addEventListener('click', () => run('ppAll', computeAll));
      box.querySelectorAll('.pp-tool').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.t)));
    }
    return box;
  }
  function renderPointsPalette() {
    const w = $('canvasWrapper'); const box = buildPointsPalette(); if (!w || !box) return;
    // no object yet (user request 2026-10-16): the palette shows only "Add first object" – the obvious first step
    const empty = S.stage === ST.SAM3 && !S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.classList.toggle('empty', empty);
    if (empty) { box.style.display = ''; return; }
    const show = S.stage === ST.SAM3 && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    if (!show) { box.style.display = 'none'; return; }
    box.style.display = '';
    const f = V().currentFrame; const pts = S.points[f] || [];
    const pos = pts.filter((p) => p[2] > 0).length, neg = pts.length - pos;
    const busy = S.pointsBusy[f], dirty = !!(S.pointsDirty[f] || S.pointsFailed[f]);
    const k = S.keys.find((x) => x.frame === f);
    $('ppInfo').innerHTML = `F${f}: <b>${pos}</b> + / <b>${neg}</b> −` +
      (busy ? ' · <span class="busy">⏳ SAM 3.1 liczy…</span>' : dirty ? ' · <span class="err">● nieprzeliczone</span>' : k ? ' · <span class="ok">🔑 klucz</span>' : '');
    // a prompt alone is enough (bug 2026-10-15: the prompt typed in the „Nowy obiekt” dialog is in the field but was not
    // "typed here", so the button stayed disabled until the user retyped it)
    const promptOnly = !pts.length && !!promptField() && !k;
    $('ppCompute').disabled = !!busy || (!dirty && !pts.length && !promptField());
    $('ppCompute').classList.toggle('btn-cta', (dirty || promptOnly) && !busy);
    $('ppClear').disabled = !pts.length || !!busy;
    box.querySelectorAll('.pp-tool').forEach((b) => b.classList.toggle('active', S.tool === b.dataset.t));
    const core = stageState('core').state, canAll = stageState('detect').state === 'done';
    for (const id of ['ppAll', 'st1All']) {
      const b = $(id); if (!b) continue;
      b.disabled = !canAll || document.body.classList.contains('job-busy');
      b.classList.toggle('btn-cta', canAll && core !== 'done');
      b.textContent = core === 'done' || core === 'stale' ? (core === 'stale' ? '▶▶ Przelicz zmiany' : '▶▶ Przelicz całość') : '▶▶ Przelicz całość';
    }
    placeFloating(box, w, ptsPos.get());
  }
  async function flushPoints(f) {
    if (!S.obj || S.pointsBusy[f]) return;
    if (!S.pointsDirty[f]) return;
    S.pointsDirty[f] = false; S.pointsBusy[f] = true; updatePointsBox();
    const ps = plateSize();
    const inside = (p) => !ps || (p[0] >= 0 && p[1] >= 0 && p[0] < ps[0] && p[1] < ps[1]);
    if (!(S.points[f] || []).every(inside)) {   // Z11: e.g. an old key with a point outside the plate – the server refuses it
      S.points[f] = (S.points[f] || []).filter(inside); toast('Punkt poza kadrem – pominięty');
    }
    const pts = S.points[f] || [];
    try {
      const d = await api('POST', `/api/object/${S.obj}/points/${f}`, {
        positive: pts.filter((p) => p[2] > 0).map((p) => [p[0], p[1]]), negative: pts.filter((p) => p[2] < 0).map((p) => [p[0], p[1]]),
        prompt: promptField() });   // the field as it is now (also saved as the object's prompt by the server)
      delete S.overlay[`${S.obj}_key_${f}`];
      Object.keys(S.compCache).filter((k) => k.includes(`_key_${f}`)).forEach((k) => delete S.compCache[k]);
      if (d.empty) {   // keep the points on screen so the user can add a better one; the key (if any) is gone
        status(d.detail, 'warn');
        const keep = S.points[f];
        await loadKeys(); S.points[f] = keep; renderKeysLane();
      } else {
        if (d.deleted) status(`Klucz F${f} usunięty (brak punktów). Ctrl+Z cofa.`);
        else if (d.kept) status(d.detail || `Klatka ${f}: zachowany klucz ${d.kept}`, 'warn');
        else status(`Klucz F${f} zapisany (SAM 3.1 ${d.seconds}s) – ` + (d.mode === 'prompt+points'
          ? `prompt „${d.prompt}” + punkty: dodane ${d.plus_px} px, odjęte ${d.minus_px} px.`
          : d.mode === 'prompt' ? `z samego promptu „${d.prompt}” (bez punktów).` : 'z samych punktów (bez promptu).') + ' Zmiana punktów / promptu + „▶ Przelicz klatkę” = nowy klucz.', 'ok');
        await loadKeys();
      }
      await loadStages();
      delete S.pointsFailed[f];
    } catch (e) {
      // NOT saved: keep the frame pending (orange points + lane mark), retry after the running job / before computing
      S.pointsDirty[f] = true; S.pointsFailed[f] = e.message;
      const busy = e.status === 409;
      const msg = busy ? `Punkty F${f} NIE są jeszcze kluczem – ${e.message}. Po zakończeniu liczenia kliknij „▶ Przelicz klatkę” (albo policzą się same przed liczeniem obiektu).`
                       : `Punkty F${f} NIE zapisane jako klucz: ${e.message}`;
      status(msg, 'err'); toast(msg, 7000);
    } finally {
      S.pointsBusy[f] = false; updatePointsBox(); V().render(); renderKeysLane();   // points added meanwhile wait for „Przelicz”
    }
  }
  // frames whose points are not (yet) a key on the server
  const pendingPointFrames = () => Object.keys(S.pointsDirty).filter((f) => S.pointsDirty[f] || S.pointsBusy[f]).map(Number).sort((a, b) => a - b);
  async function flushPendingPoints() {
    clearTimeout(S.ptsTimer);
    for (const f of pendingPointFrames()) {
      if (S.pointsBusy[f]) { for (let i = 0; i < 120 && S.pointsBusy[f]; i++) await new Promise((z) => setTimeout(z, 250)); continue; }
      delete S.pointsFailed[f];
      await flushPoints(f);
    }
  }
  // before any computation of the object: every point must already be a key (else the run ignores it)
  async function ensurePointsSaved() {
    if (S.brushPending != null) await saveBrushKey(S.brushPending);
    if (window.MD_ZONES && window.MD_ZONES.ready) await window.MD_ZONES.flush();
    if (window.MD_BRUSH && window.MD_BRUSH.ready) await window.MD_BRUSH.flush();
    await flushPendingPoints();
    const left = pendingPointFrames();
    if (left.length) throw new Error(`punkty w klatkach ${left.join(', ')} nie są zapisane jako klucze – nic nie liczę (${S.pointsFailed[left[0]] || 'zapis trwa'})`);
  }
  function canvasToPlate(e) {
    const v = V(); const r = v.canvas.getBoundingClientRect(); const img = v.getCurrentImage(); const ps = plateSize();
    if (!img || !ps) { toast('Klatka jeszcze się ładuje – kliknij ponownie'); return null; }
    const x = (e.clientX - r.left - v.panX) / v.zoom, y = (e.clientY - r.top - v.panY) / v.zoom;
    return [x * ps[0] / img.width, y * ps[1] / img.height];
  }
  function clampPlate(p) {   // brush / polygon points stay inside the plate (Z11)
    const ps = plateSize(); if (!p || !ps) return p;
    return [Math.min(Math.max(p[0], 0), ps[0] - 1), Math.min(Math.max(p[1], 0), ps[1] - 1)];
  }
  function hitPoint(e) {
    const v = V(); const img = v.getCurrentImage(); const ps = plateSize(); if (!img || !ps) return -1;
    const r = v.canvas.getBoundingClientRect(); const k = img.width / ps[0];
    const pts = S.points[v.currentFrame] || [];
    for (let i = pts.length - 1; i >= 0; i--) {
      const sx = v.panX + pts[i][0] * k * v.zoom + r.left, sy = v.panY + pts[i][1] * k * v.zoom + r.top;
      if (Math.hypot(sx - e.clientX, sy - e.clientY) <= 9) return i;
    }
    return -1;
  }

  // ------------------------------------------------------------------------------------------------ draw tools (left bar)
  const CLICK_TOOLS = { pt_add: ST.SAM3, pt_remove: ST.SAM3, corr_add: ST.SAM3, corr_remove: ST.SAM3 };   // tool -> stage
  const clickTool = () => CLICK_TOOLS[S.tool] === S.stage;
  // ---- Z23: brush size (log slider 1–200 px of the plate, per stage in localStorage), pen pressure
  const R_MIN = 1, R_MAX = 200;
  const radiusKey = () => `md_brush_r_${STAGES[S.stage].id}`;
  function sliderToR(v) { return Math.max(R_MIN, Math.min(R_MAX, Math.round(Math.exp(Math.log(R_MIN) + (Math.log(R_MAX) - Math.log(R_MIN)) * v / 1000)))); }
  function rToSlider(r) { return Math.round(1000 * (Math.log(Math.max(R_MIN, r)) - Math.log(R_MIN)) / (Math.log(R_MAX) - Math.log(R_MIN))); }
  function setRadius(r, from) {
    S.radius = Math.max(R_MIN, Math.min(R_MAX, Math.round(r || 1)));
    try { localStorage.setItem(radiusKey(), String(S.radius)); } catch (e) { /* storage blocked */ }
    const bs = $('brushSize'), bn = $('brushSizeN');
    if (bs && from !== 'slider') bs.value = rToSlider(S.radius);
    if (bn && from !== 'field') bn.value = S.radius;
    V() && V().render();
  }
  function loadStageRadius() {
    try { const v = +localStorage.getItem(radiusKey()); if (v >= R_MIN && v <= R_MAX) S.radius = v; } catch (e) { /* storage blocked */ }
  }
  function penRadius(e) {   // light touch -> min 15 % of the size, gamma 1.0 by default
    const pr = e.pressure > 0 ? e.pressure : 0.5;
    return Math.max(0.15, Math.pow(pr, S.penGamma || 1.0)) * S.radius;
  }
  function drawVarStroke(ctx, pts, k, rDefault, outline = false) {   // same geometry as engine/keys.py:draw_variable_stroke; outline = stroke the reach instead of filling it
    const P = pts.map((q) => [q[0] * k, q[1] * k, (q.length >= 3 ? q[2] : rDefault) * k]);
    ctx.beginPath();
    for (const [x, y, r] of P) { ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 2 * Math.PI); }
    for (let i = 0; i + 1 < P.length; i++) {
      const [x0, y0, r0] = P[i], [x1, y1, r1] = P[i + 1];
      const dx = x1 - x0, dy = y1 - y0, d = Math.hypot(dx, dy);
      if (d <= Math.abs(r1 - r0) + 1e-6) continue;
      const ux = dx / d, uy = dy / d, nx = -uy, ny = ux, sa = (r0 - r1) / d, ca = Math.sqrt(Math.max(0, 1 - sa * sa));
      const v = [1, -1].map((g) => [g * nx * ca + ux * sa, g * ny * ca + uy * sa]);
      ctx.moveTo(x0 + r0 * v[0][0], y0 + r0 * v[0][1]); ctx.lineTo(x1 + r1 * v[0][0], y1 + r1 * v[0][1]);
      ctx.lineTo(x1 + r1 * v[1][0], y1 + r1 * v[1][1]); ctx.lineTo(x0 + r0 * v[1][0], y0 + r0 * v[1][1]); ctx.closePath();
    }
    if (outline) ctx.stroke(); else ctx.fill('nonzero');
  }
  function setTool(t) {
    S.tool = S.tool === t ? 'none' : t;
    updateDrawTools();
  }
  function updateDrawTools() {
    const box = $('drawTools'); const title = $('drawToolsTitle'); if (!box) return;
    const band = S.obj && !S.auxMode && bandStage();
    const show = S.obj && !S.auxMode && (drawStage() || band);
    box.style.display = show ? '' : 'none'; title.style.display = show ? '' : 'none';
    renderPointsPalette();
    if (!show) { S.tool = 'none'; return; }
    const cvs = V() && V().canvas; if (cvs) cvs.style.cursor = clickTool() ? 'crosshair' : '';
    const tb = (t, icon, label, title) => `<button class="tool-btn ${S.tool === t ? 'active' : ''}" data-t="${t}" title="${title}"><span class="tool-icon">${icon}</span><span class="tool-label">${label}</span></button>`;
    const wireTools = () => box.querySelectorAll('.tool-btn').forEach((b) => b.addEventListener('click', () => {
      if (b.dataset.t === 'clear') {
        const f = V().currentFrame;
        // Final Touch: the strokes are saved at once – „Wyczyść” removes this frame's corrections (user decision 2026-10-22)
        if (S.stage === ST.FINAL_TOUCH && window.MD_FT && window.MD_FT.ready) { run('btnDelKey', () => window.MD_FT.deleteKeysAt(f)); return; }
        S.strokes[f] = []; S.polys[f] = []; S.poly = null; V().render(); return;
      }
      setTool(b.dataset.t);
    }));
    if (band) {   // band control points of the whole-outline zone (static/zones.js); the palette removes a whole key frame
      title.textContent = 'Punkty pasa';
      if (!['band_add', 'band_remove'].includes(S.tool)) S.tool = 'none';
      box.innerHTML = tb('band_add', '⊕', 'Dodaj punkt G', 'Dodaj punkt pasa [G]: klik przy obrysie = nowy punkt (klucz tej klatki); wąsy: żółty = pas na zewnątrz, niebieski = do środka; przeciągnij koniec wąsa = grubość, punkt = położenie; 0 / 0 = przerwa. Esc = koniec')
        + tb('band_remove', '⊖', 'Usuń punkt R', 'Usuń punkt pasa [R]: klik na punkt = usuń go (wszystkie jego klucze; Ctrl+Z cofa). Klucze jednej klatki: „Usuń wszystkie punkty (ta klatka)” na palecie albo 🔑✕. Esc = koniec');
      wireTools();
      bandMod().onTool();
      return;
    }
    let tools;
    if (S.stage === ST.SAM3) {   // Punkt + / − are on the floating palette (user request 2026-10-15); here: corrections of a SAM3 key
      title.textContent = 'Korekta klucza SAM 3.1';
      tools = tb('corr_add', '⊕', 'Korekta + K', 'Korekta + [K]: klik w dziurę / pominięty fragment = SAM 3.1 dodaje tylko ten fragment (reszta maski bez zmian; PPM = odwrotnie). Zapis jako klucz SAM 3.1 tej klatki. Przeciąganie = przesuwanie obrazu') +
        tb('corr_remove', '⊖', 'Korekta − L', 'Korekta − [L]: klik na wyciek = SAM 3.1 usuwa tylko ten fragment (reszta maski bez zmian; PPM = odwrotnie). Zapis jako klucz SAM 3.1 tej klatki') +
        tb('brush_add', '🖌+', 'Klucz +', 'Pędzel klucza SAM 3.1 (dodaj): domalowana maska tej klatki staje się kluczem SAM 3.1 – potem „Przelicz zmienione przedziały”. Poprawka bez SAM 3.1 (CoTracker3) = etap 2') +
        tb('brush_remove', '🖌−', 'Klucz −', 'Pędzel klucza SAM 3.1 (usuń): wymazana maska tej klatki staje się kluczem SAM 3.1') +
        tb('poly', '⬠', 'Wielokąt P', 'Wielokąt klucza SAM 3.1 [P]: klikaj punkty, Enter zamyka (PPM na 1. punkcie = usuwający) – zapis jako klucz');
    } else if (S.stage === ST.BRUSH) {
      title.textContent = 'Pędzel poprawek (CoTracker3)';
      tools = tb('brush_add', '🖌+', 'Dodaj G', 'Dodaj do maski [G]: domaluj brakujący fragment obiektu (tylko piksele połączone z maską). Kreska = klucz poprawki tej klatki, CoTracker3 przenosi ją na inne klatki') +
        tb('brush_remove', '🖌−', 'Usuń R', 'Usuń z maski [R]: zamaluj wyciek na tło / fragment, który nie należy do obiektu. Kreska = klucz poprawki tej klatki') +
        tb('brush_erase', '🧽', 'Gumka E', 'Gumka [E]: wymazuje pędzel Dodaj / Usuń w tym miejscu tej klatki (maska wraca do SAM 3.1)');
    } else if (S.stage === ST.FINAL_TOUCH) {
      title.textContent = 'Pędzel korekt (Final Touch)';
      tools = tb('brush_add', '🖌+', 'Dodaj G', 'Dodaj [G]: maluje biel – alfa tej klatki → 1 (miękko wg twardości z panelu). Tylko ta klatka, bez śledzenia') +
        tb('brush_remove', '🖌−', 'Usuń R', 'Usuń [R]: maluje czerń – alfa tej klatki → 0') +
        tb('brush_erase', '🧽', 'Gumka E', 'Gumka [E]: wymazuje kreski Dodaj i Usuń w tym miejscu tej klatki – alfa wraca do bazy');
    } else {
      title.textContent = S.stage === ST.VITMATTE ? 'Obszar strefy ViTMatte' : 'Rysowanie (strefy progowe, lupa)';
      tools = `
      <button class="tool-btn ${S.tool === 'brush_add' ? 'active' : ''}" data-t="brush_add" title="Pędzel dodaj [G]; rozmiar [ ]"><span class="tool-icon">🖌+</span><span class="tool-label">Dodaj G</span></button>
      <button class="tool-btn ${S.tool === 'brush_remove' ? 'active' : ''}" data-t="brush_remove" title="Pędzel usuń [R]"><span class="tool-icon">🖌−</span><span class="tool-label">Usuń R</span></button>
      <button class="tool-btn ${S.tool === 'poly' ? 'active' : ''}" data-t="poly" title="Wielokąt [P]: klikaj punkty, Enter zamyka, PPM na 1. punkcie = usuwający"><span class="tool-icon">⬠</span><span class="tool-label">Wielokąt P</span></button>`;
    }
    box.innerHTML = tools + `
      <div class="tool-radius" title="Rozmiar pędzla (px plate'u): suwak, pole, [ ] albo Shift + przeciągnij w lewo / prawo na obrazie">
        <input type="range" id="brushSize" min="0" max="1000" value="${rToSlider(S.radius)}" style="width:100%">
        <input type="number" id="brushSizeN" min="1" max="200" value="${S.radius}" class="form-input" style="width:58px"> px</div>
      ${S.penSeen ? `<label class="chk" title="Piórko tabletu: siła nacisku zmienia grubość kreski (mysz – stały rozmiar)"><input type="checkbox" id="penPressure" ${S.penPressure ? 'checked' : ''}> Nacisk piórka</label>`
        : `<div class="pen-state muted" title="Przeglądarka nie dostała jeszcze zdarzenia piórka (pointerType = pen) – tylko mysz. Wacom: Wacom Tablet Properties → Pióro → Mapowanie → zaznacz „Użyj Windows Ink” (Use Windows Ink), potem odśwież stronę. Bez tego sterownik udaje mysz i nacisk nie dociera do przeglądarki.">✎ piórko: nie wykryto${S.penAsMouse ? ' (tablet jako mysz – włącz Windows Ink)' : ''}</div>`}
      <button class="tool-btn" data-t="clear" title="${S.stage === ST.FINAL_TOUCH ? 'Usuń wszystkie korekty tej klatki (klucz Final Touch) [Delete]; Ctrl+Z cofa' : 'Wyczyść kreski i wielokąty tej klatki'}"><span class="tool-icon">${S.stage === ST.FINAL_TOUCH ? '🗑' : '⌫'}</span><span class="tool-label">${S.stage === ST.FINAL_TOUCH ? 'Usuń klucz' : 'Wyczyść'}</span></button>`;
    wireTools();
    const bs = $('brushSize'), bn = $('brushSizeN'), pp = $('penPressure');
    if (bs) bs.addEventListener('input', () => setRadius(sliderToR(+bs.value), 'slider'));
    if (bn) bn.addEventListener('change', () => setRadius(+bn.value, 'field'));
    if (pp) pp.addEventListener('change', () => { S.penPressure = pp.checked; });
    if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.onStage();   // stage-2 panel mirrors the active tool
    if (window.MD_FT && window.MD_FT.ready) window.MD_FT.onStage();
  }

  // ------------------------------------------------------------------------------------------------ panes
  const nextBtn = (n) => `<button class="btn btn-primary next-btn" id="st${n}Next" disabled>Dalej →</button>`;
  // what the stage does / what it is best for – at the top of every stage pane (user request 2026-10-15)
  const descHtml = (key) => `<div class="stage-desc" title="Opis etapu">${STAGES[ST[key]].desc}</div>`;
  const paneTitle = (key, sub) => `<div class="pane-title">${ST[key]} · ${STAGES[ST[key]].label}${sub ? ` <span class="muted">(${sub})</span>` : ''}</div>`;
  // edge settings shared by stage MatAnyone and Eksperymenty: a floating palette over the view, left, like „Punkty SAM3”
  // (user request 2026-10-15), with a preview of the band before computing (GET edge_preview) that can be switched off
  function edgeSettingsHtml() {
    return `<div class="pts-palette edge-palette" id="edgePalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Krawędź (MatAnyone2)</div>
      <label class="chk" title="Podgląd przed liczeniem: przerywane linie = krawędzie pasa, w którym MatAnyone2 decyduje o alfie (adaptacyjny – szerszy przy rozmyciu / ruchu), cienkie przerywane = koniec wtopień; w środku zostaje maska 1+2. Wyłącz, gdy oglądasz policzony wynik"><input type="checkbox" id="ePv" title="Podgląd pasa krawędzi na obrazie (wł. / wył.)"> 👁 podgląd pasa krawędzi</label>
      <div class="band-seed" id="epBandSeed"></div>
      <div class="muted pp-hint" id="ePvInfo"></div>
      <div class="pp-sec edge-settings" id="edgeSettings">
      <label class="edge-px" title="Jak daleko poza krawędź maski 1+2 sięga pas (zewnętrzna przerywana linia), w którym MatAnyone2 liczy miękką alfę (kosmyki, rozmycie). To minimum: tam, gdzie automat mierzy większe rozmycie ostrości / ruchu, pas sam jest szerszy. Maska się nie przesuwa. Suwak 0–60 px, w polu dowolna wartość (do 200 px)">Pas na zewnątrz <input id="o4Out" type="range" min="0" max="60" step="1" value="0"><input id="o4OutV" type="number" class="form-input edge-num" min="0" step="1" value="0" title="Dowolna wartość (0–200 px) – wpisz i Enter"> px</label>
      <label class="edge-px" title="Ile px w głąb maski 1+2 oddajemy MatAnyone2 (wewnętrzna przerywana linia pasa); dalej w środku alfa = 1. Minimum jak wyżej – automat może dać więcej przy rozmyciu. 0 = sam automat (3–24 px). Suwak 0–30 px, w polu do 200 px">Wcięcie w rdzeń <input id="o4In" type="range" min="0" max="30" step="1" value="0"><input id="o4InV" type="number" class="form-input edge-num" min="0" step="1" value="0" title="Dowolna wartość (0–200 px) – wpisz i Enter"> px</label>
      <label class="edge-px" title="Na ilu px wynik MatAnyone2 przenika się z wnętrzem rdzenia na wewnętrznym brzegu pasa (wcięcia). 0 = twarde przejście (jak dotąd) – może zostać widoczna linia na granicy wcięcia; kilka px = miękkie wtopienie">Wtopienie wcięcia <input id="o4Feather" type="range" min="0" max="24" step="1" value="0" title="Wtopienie wcięcia (px)"><input id="o4FeatherV" type="number" class="form-input edge-num" min="0" max="64" step="1" value="0" title="Wtopienie wcięcia (px, 0–64)"> px</label>
      <label class="edge-px" title="Na ilu px od ZEWNĘTRZNEJ krawędzi pasa wynik MatAnyone2 przechodzi w rdzeń (zwykle czerń) – cieńsza przerywana linia na podglądzie. Nie nachodzi na wtopienie wcięcia. 0 = twarde cięcie (jak dotąd)">Wtopienie krawędzi zewn. <input id="o4OutFeather" type="range" min="0" max="80" step="1" value="0" title="Wtopienie krawędzi zewnętrznej (px)"><input id="o4OutFeatherV" type="number" class="form-input edge-num" min="0" max="200" step="1" value="0" title="Wtopienie krawędzi zewnętrznej (px, 0–200)"> px</label>
      <div class="pp-sec vm-band-pts"><div class="band-mode" id="epBandMode"></div><span class="muted pp-hint" id="epBandInfo"></span>
        <button type="button" class="btn btn-ghost btn-xs" id="epBandClear" title="Usuń wszystkie punkty bieżącej klatki kluczowej pasa (= usuń klatkę kluczową)">🗑 Usuń wszystkie punkty (ta klatka)</button>
        <label title="Punkty pasa z innego etapu (np. strefy „cały obrys” etapu 5) zastępują punkty tego pasa – szerokości takie, jak tam widać">Kopiuj punkty pasa z <select id="epBandSrc" class="form-input" title="Skąd skopiować punkty pasa"></select></label>
        <button type="button" class="btn btn-secondary btn-xs" id="epBandCopy" title="Skopiuj punkty pasa (z kluczami) z wybranego miejsca">⇩ Kopiuj</button></div>
      <div class="muted pp-hint" id="o4PxHint">0 / 0 = sam automat (pas wg rozmycia, 3–24 px w każdą stronę)</div>
      <div class="row">
        <label title="Migotanie vs duchy przy ruchu">Wygładzanie<select id="o4Smooth" class="form-input"><option value="weak">słabe</option><option value="normal">normalne</option><option value="strong">mocne</option></select></label>
        <label title="Lupa na cienkich częściach (czułki, rogi, druty)">Lupa<select id="o4Lupa" class="form-input"><option value="auto">auto</option><option value="off">wył.</option><option value="box">ramka (ostatni wielokąt)</option></select></label>
      </div>
      <label id="o4MaScaleL" title="Rozdzielczość, w której liczy MatAnyone2">Skala MatAnyone2<select id="o4MaScale" class="form-input"><option value="auto">auto</option><option value="1.0">1.0</option><option value="0.75">0.75</option><option value="0.5">0.5</option></select></label>
      </div>
      <div class="pp-btns"><button type="button" class="btn btn-secondary btn-sm" id="epStagePv" title="Policz MatAnyone2 tylko na kilku klatkach wokół bieżącej (±4 kl.) – szybkie sprawdzenie ustawień przed długim liczeniem całości. To NIE jest wynik etapu: nic się nie zapisuje jako wersja i nic się nie odblokowuje. Modele czasowe dostają mniej kontekstu niż w pełnym przebiegu – krawędź może się trochę różnić">👁 Podgląd klatek</button><button type="button" class="btn btn-primary btn-sm" id="epCompute" title="Policz z tymi ustawieniami (etap ${ST.MATANYONE}: „▶ Przelicz MatAnyone2”, etap ${ST.EXPERIMENTS}: „✔ Przelicz finalnie”)">▶ Przelicz</button></div>
    </div>`;
  }
  const epPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_edge_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_edge_palette', JSON.stringify(p)); else localStorage.removeItem('md_edge_palette'); } catch (e) { /* storage blocked */ } },
  };
  const edgeStage = () => S.stage === ST.MATANYONE;   // stage 5 has its own palette „Eksperymenty” (2026-10-16)
  // preview params: the slider values (also while dragging, before they are saved); debounced so a drag is one request
  // px mode (2026-10-16): once the object has band_out_px / band_in_px or a px slider moved; else its old ×scale / shift
  function edgePvParams() {
    if (S.edgeLegacy && !S.edgePxTouched) return { px: false, b: +S.edgeLegacy.b || 1, s: +S.edgeLegacy.s || 0 };
    return { px: true, o: Math.max(0, +$('o4OutV').value || 0), i: Math.max(0, +$('o4InV').value || 0) };
  }
  function renderEdgePalette() {
    const box = $('edgePalette'); const w = $('canvasWrapper'); if (!box || !w) return;
    const show = edgeStage() && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (!show) return;
    if (S.edgePv == null) S.edgePv = stageState('matanyone').state !== 'done';   // on until a MatAnyone result exists
    $('ePv').checked = !!S.edgePv;
    const b = $('epCompute');
    const maPrev = S.info && S.info.stage_runs && S.info.stage_runs.matanyone;
    b.textContent = S.stage === ST.MATANYONE ? (maPrev ? '▶ Policz ponownie' : '▶ Przelicz MatAnyone2') : '✔ Przelicz finalnie';
    if (S.stage === ST.MATANYONE && maPrev) b.title = `Policz MatAnyone2 od nowa – nowa wersja zastąpi wynik etapu ${maPrev} (zostaje na liście wersji)`;
    const src = S.stage === ST.MATANYONE ? $('st3Ma') : $('st4Final');
    b.disabled = !src || src.disabled || document.body.classList.contains('job-busy');
    b.classList.toggle('btn-cta', S.stage === ST.MATANYONE && stageState('matanyone').state === 'todo');
    if (S.stage === ST.MATANYONE) $('o4MaScaleL').style.display = '';   // stage 3 always runs MatAnyone
    if (!S.edgePv) $('ePvInfo').textContent = 'Podgląd wyłączony – widać policzony wynik';
    placeFloating(box, w, epPos.get());
  }
  // a server band PNG (255 kept / 128 band / 0 outside) as its dashed band edges – one band system (2026-10-22): a band
  // preview is never a fill (the old yellow / blue fill came back in stages 3 / 4 / 7, user report 2026-10-23)
  function bandEdges(ov, key) {
    const ck = `epv_${key}`;
    if (S.compCache[ck]) return S.compCache[ck];
    const c = window.MD_ZONES && window.MD_ZONES.bandPngEdges ? window.MD_ZONES.bandPngEdges(ov) : null;
    if (!c) return null;
    putComp(ck, c);
    return c;
  }
  // ---- stage preview (user request 2026-10-17): stages 3 MatAnyone / 4 VideoMaMa / 5 ViTMatte on a few frames around
  // the current one before the long run (engine/stage_preview.py). Never a stage result: drawn green only on the frames
  // of the window and only in its stage, dimmed when the settings changed, hidden once the stage itself is computed.
  const SPV_LABEL = { matanyone: 'MatAnyone2', videomama: 'VideoMaMa', vitmatte: 'ViTMatte' };
  const spvStage = () => ({ [ST.MATANYONE]: 'matanyone', [ST.VIDEOMAMA]: 'videomama', [ST.VITMATTE]: 'vitmatte' })[S.stage];
  // stage 5 keeps one preview per engine (2026-10-22): S.spv key = "vitmatte@<engine>", the engine picked in the HUD
  const spvKey = (st, eng) => (st === 'vitmatte' ? `vitmatte@${eng || vmEngine()}` : st);
  const spvEngQ = (st, eng) => (st === 'vitmatte' ? `&engine=${eng || vmEngine()}` : '');
  const spvLabel = (st, eng) => (st === 'vitmatte' ? EDGE_ENGINE_LABEL[eng || vmEngine()] : SPV_LABEL[st]);
  const spvOf = (st) => (st && S.spv ? S.spv[spvKey(st)] : null);
  // the preview of a stage shows while the stage has no result to show instead – stage 5: no version of the picked engine
  const spvWanted = (st) => (st === 'vitmatte' ? !vmRunFor(vmEngine()) : !(stageState(st) && stageState(st).state === 'done'));
  async function loadStagePreviews() {
    const obj = S.obj; S.spv = {};
    if (!obj) return;
    const slots = ['matanyone', 'videomama', ['vitmatte', 'vitmatte'], ['vitmatte', 'birefnet']];
    for (const s of slots) {
      const st = Array.isArray(s) ? s[0] : s, eng = Array.isArray(s) ? s[1] : null;
      try { const d = await api('GET', `/api/object/${obj}/stage_preview/${st}${eng ? `?engine=${eng}` : ''}`); if (S.obj === obj && d.exists) S.spv[spvKey(st, eng)] = d; } catch (e) { /* none */ }
    }
    if (S.obj === obj && V()) V().render();
  }
  async function stagePreview(stage, btnId) {
    if (!S.obj) return;
    const f = V().currentFrame;
    await run(btnId, async () => {
      const eng = stage === 'vitmatte' ? edgeEngine() : null;   // the stage engine (the zones' model), not the HUD pick
      status(`Podgląd ${spvLabel(stage, eng)}: liczę kilka klatek wokół F${f}…`);
      const d = await api('POST', `/api/object/${S.obj}/stage_preview/${f}?stage=${stage}`);
      S.spv = { ...(S.spv || {}), [spvKey(stage, d.engine || eng)]: d };
      if (stage === 'vitmatte') S.vmPick = null;   // show what was just computed
      status(`Podgląd ${spvLabel(stage, d.engine || eng)}: F${d.frames[0]}–${d.frames[d.frames.length - 1]} (${d.seconds} s) – zielone na obrazie. ` +
        'To nie jest wynik etapu: nic nie zostało zapisane jako wersja i nic się nie odblokowało.', 'ok');
      V().render();
    });
  }
  // 👁 of the selected object off in the Objects palette = NO mask of it in any stage (user report 2026-10-22: the green
  // preview of a few frames / of a result kept covering the image) – only the tools (band lines, points, strokes) stay
  const selMaskHidden = () => !!(S.hideSel && S.obj && !S.auxMode);
  function drawStagePreview(ctx, img, f) {
    const st = spvStage(); const m = spvOf(st);
    if (!m || !S.obj || S.auxMode || !(m.frames || []).includes(f)) return '';
    if (!spvWanted(st)) return '';   // the stage result is there – show it, not the preview
    if (selMaskHidden()) return 'Maska wybranego obiektu ukryta – 👁 w palecie Obiekty';
    const bw = alphaBw();   // the Alpha view is black / white only (user rule 2026-10-22): the preview is the matte there
    // (matteFor shows it), never a green tint over it; the note still says it is a preview
    // the stage's own „👁 alfa” switch hides the green tint too (user report 2026-10-22: „zone alpha” off, still green)
    const alphaOff = (st === 'vitmatte' && window.MD_ZONES && window.MD_ZONES.ready && !window.MD_ZONES.alphaPvOn)
      || (st === 'videomama' && !vmmPvGet('alpha', true));
    if (!bw && !alphaOff) {
      const key = `${S.obj}_spv_${spvKey(st)}_${m.created_at}_${f}`;
      const al = settledOverlay(key, `/api/object/${S.obj}/stage_preview/${st}/${f}?c=${encodeURIComponent(m.created_at)}${spvEngQ(st)}`, f);
      if (al && al !== 'missing') {
        ctx.save(); ctx.globalAlpha = m.stale ? 0.35 : 0.75;
        ctx.drawImage(tintCanvas(al, [60, 255, 90], key), 0, 0, img.width, img.height); ctx.restore();
      }
    }
    return `Podgląd ${spvLabel(st)} F${m.frames[0]}–${m.frames[m.frames.length - 1]}${bw || alphaOff ? '' : ' (zielone)'} – nie jest wynikiem etapu` +
      (m.stale ? ' · NIEAKTUALNY: zmieniono ustawienia – „👁 Podgląd klatek” jeszcze raz' : '');
  }
  function drawEdgePreview(ctx, img, f) {
    if (!((S.edgePv || S.pvDrag === 'edge') && edgeStage() && S.obj && !S.auxMode)) return '';
    const dragging = S.pvDrag === 'edge';
    const p = dragging ? edgePvParams() : (S.epvParams || edgePvParams());
    const q = p.px ? `band_out_px=${p.o}&band_in_px=${p.i}` : `band_scale=${p.b}&shift_px=${p.s}`;
    // instant band edges from the mask 1+2 in the browser (fixed px, no adaptive widening, dashed lines – no fill):
    // while the slider is dragged until the server's exact lines are loaded (user reports 2026-10-17 / 2026-10-23)
    const live = () => {
      if (!p.px || !(window.MD_ZONES && window.MD_ZONES.liveBand)) return null;
      const kind = S.info && S.info.has && S.info.has.core_fix ? 'core_fix' : 'core';
      const m = overlayFor(kind, f);
      return m && m !== 'missing' ? window.MD_ZONES.liveBand(m, `${S.obj}_${kind}_${f}`, p.i, p.o) : null;
    };
    // one band system (2026-10-22): dashed band edges + thin feather-end lines, no fill – in the px mode and in the
    // automatic (adaptive) mode too (user report 2026-10-22: stage 3 still showed the old filled preview at 0 / 0);
    // the adaptive band comes from the saved settings (no reach / inset in the request)
    const lp = { feather_px: +$('o4FeatherV').value || 0, outer_feather_px: +$('o4OutFeatherV').value || 0,
                 ...(p.px ? { reach_px: p.o, inset_px: p.i } : { band_scale: p.b, shift_px: p.s }) };
    if (dragging) {   // a slider held (user report 2026-10-23: the old filled band came back while dragging)
      if (!(window.MD_BANDPTS && window.MD_BANDPTS.ready && window.MD_BANDPTS.drawLines(ctx, img, f, lp, { live: true }))) {
        const c = live(); if (c) ctx.drawImage(c, 0, 0, img.width, img.height);
      }
    } else {
      if (window.MD_BANDPTS && window.MD_BANDPTS.ready) {
        // while the lines load nothing is drawn (user report 2026-10-22: the old filled band flashed on every stage
        // change before the lines replaced it)
        if (!window.MD_BANDPTS.drawLines(ctx, img, f, lp)) return `Podgląd pasa krawędzi F${f}: ładuję linie pasa…`;
      } else {   // no band-points module (older page): the filled preview as before
        const key = `${S.obj}_epv_${q}_${f}`;
        const ov = settledOverlay(key, `/api/object/${S.obj}/edge_preview/${f}?${q}`, f);
        if (ov === 'missing') return `Podgląd krawędzi F${f}: brak rdzenia / kadru w tej klatce`;
        if (ov) drawBand(ctx, img, 'edge', q, ov, key);
        else { const c = live(); if (c) ctx.drawImage(c, 0, 0, img.width, img.height); }
      }
    }
    const info = $('ePvInfo');
    const desc = p.px ? `na zewnątrz ≥ ${p.o} px, w rdzeń ≥ ${p.i} px (więcej, gdzie rozmycie)` : `pas automatyczny z rozmycia ×${p.b}, środek ${p.s} px`;
    if (info) info.textContent = `F${f}: ${desc} – przerywane = krawędzie pasa (między nimi decyduje MatAnyone2), cienkie = koniec wtopień`;
    return '';   // no banner: the lines say it (one band system, 2026-10-22)
  }
  // ---- palette „Eksperymenty” (user request 2026-10-16): first the BASE version (MatAnyone / ViTMatte / LTX / mask 1+2 /
  // any version), then QA and the edge-correction proposals are computed on it (measured once, into that version);
  // threshold sliders recount the proposals live from the saved measurements; previews can be switched off
  function expPaletteHtml() {
    return `<div class="pts-palette exp-palette" id="expPalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Eksperymenty – baza i QA</div>
      <label class="exp-base" title="Wersja, którą analizujemy: QA (migotanie, flagi) i propozycje poprawki krawędzi liczą się na niej (raz, potem z pamięci)">Maska bazowa<select id="exBase" class="form-input" title="Wersja, na której liczymy QA i propozycje krawędzi"></select></label>
      <div class="muted pp-hint" id="exInfo">Wybierz maskę bazową – QA i propozycje krawędzi policzą się na niej.</div>
      <div class="pp-chk"><label class="chk" title="Odcinki krawędzi do poprawy na obrazie (czerwony = halo → zwęź, niebieski = ucięcie → poszerz)"><input type="checkbox" id="exPvEdge" title="Propozycje poprawki krawędzi na obrazie (wł. / wył.)"> 👁 propozycje krawędzi</label>
        <label class="chk" title="Mapa migotania analizowanej wersji (jaśniej = krawędź bardziej migocze); to samo co klawisz M"><input type="checkbox" id="exPvFlick" title="Mapa migotania (wł. / wył.)"> 👁 migotanie</label></div>
      <div class="pp-sec">
        <label class="edge-px" title="Odcinek (~64 px obrysu) jest do poprawy, gdy jego krawędź odstaje od krawędzi na plate'cie o więcej niż tyle px (mediana). Mniej = więcej propozycji">Próg odchyłki <input id="exThrS" type="range" min="0.2" max="3" step="0.1" title="Próg odchyłki krawędzi (px)"><input id="exThr" type="number" min="0.05" step="0.05" class="form-input edge-num" title="Próg odchyłki (px)"> px</label>
        <label class="edge-px" title="… przez co najmniej tyle kolejnych klatek (po wygładzeniu ±4). Mniej = więcej krótkich propozycji">Min. klatek <input id="exMinS" type="range" min="1" max="20" step="1" title="Minimalna liczba kolejnych klatek"><input id="exMin" type="number" min="1" step="1" class="form-input edge-num" title="Minimalna liczba kolejnych klatek"> kl.</label>
        <div class="muted pp-hint" id="exCount"></div>
      </div>
      <div class="pp-sec">
        <label title="Czym liczona jest krawędź nowej wersji z maski bazy: MatAnyone2 (miękka krawędź), backlit (gdy jest plate backlit) albo bez krawędzi (twarda maska bazy po strefach / poprawce)">Silnik krawędzi<select id="exMode" class="form-input" title="Silnik krawędzi nowej wersji"><option value="matanyone">MatAnyone2</option><option value="backlit">backlit</option><option value="core">bez krawędzi (twarda)</option></select></label>
        <label class="chk" title="Najpierw przesuń krawędź BAZY w zaznaczonych odcinkach (lista w panelu po prawej), potem licz silnikiem"><input type="checkbox" id="exWithFix" title="Z poprawką krawędzi bazy (zaznaczone odcinki)"> z poprawką krawędzi bazy</label>
      </div>
      <div class="pp-btns"><button type="button" class="btn btn-secondary btn-sm" id="exAnalyze" title="Policz QA i propozycje krawędzi na wybranej wersji (pomiar raz, potem tylko progi)">▶ Analizuj</button>
        <button type="button" class="btn btn-primary btn-sm" id="exCompute" title="Nowa wersja z maski bazy: strefy progowe → (poprawka krawędzi) → silnik krawędzi → QA; pochodzenie zapisane w wersji">▶ Przelicz z tej bazy</button></div>
    </div>`;
  }
  const expPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_exp_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_exp_palette', JSON.stringify(p)); else localStorage.removeItem('md_exp_palette'); } catch (e) { /* storage blocked */ } },
  };
  const expPvGet = (k, d) => { try { const v = localStorage.getItem(`md_exp_${k}`); return v == null ? d : v === '1'; } catch (e) { return d; } };
  const expPvSet = (k, on) => { try { localStorage.setItem(`md_exp_${k}`, on ? '1' : '0'); } catch (e) { /* storage blocked */ } };
  // ---- one mask-source chooser for every stage (user request 2026-10-21: the base mask is picked the same way everywhere –
  // same order, names, version id and greyed-out state). A stage takes masks only from earlier stages, so each picker
  // passes its `keys`: VideoMaMa core / matanyone; ViTMatte core / matanyone / videomama; LTX core / matanyone / vitmatte
  // (+ Eksperymenty versions); Eksperymenty and Eksport everything. value 'stage' = option value is the stage key,
  // 'run' = the version id (Eksperymenty). avail(k) overrides "has a result" (Eksport: current results only);
  // keep = a saved value that stays selectable even without a result (its state is told next to the picker).
  function maskSources() {
    const sr = (S.info && S.info.stage_runs) || {};
    const vb = (S.info && S.info.vitmatte_base) || 'matanyone';
    return [
      { k: 'core', label: `Maska 1+2 (etapy ${ST.SAM3} + ${ST.BRUSH})`, run: sr.core_base || '', todo: `najpierw etapy ${ST.SAM3} + ${ST.BRUSH}` },
      { k: 'matanyone', label: `MatAnyone2 (etap ${ST.MATANYONE})`, run: sr.matanyone || '', todo: `najpierw etap ${ST.MATANYONE} („▶ Przelicz MatAnyone2”)` },
      { k: 'videomama', label: `VideoMaMa (etap ${ST.VIDEOMAMA})`, run: sr.videomama || '', todo: `najpierw etap ${ST.VIDEOMAMA} („▶ Policz VideoMaMa”)` },
      { k: 'vitmatte', label: `${EDGE_ENGINE_LABEL[edgeEngine()]} (etap ${ST.VITMATTE})`, run: sr[`vitmatte@${vb}`] || sr.vitmatte || '', todo: `najpierw etap ${ST.VITMATTE} („Zastosuj na ujęcie”)` },
      { k: 'ltx', label: `LTX-2.5 (etap ${ST.LTX})`, run: sr.ltx || '', todo: `najpierw etap ${ST.LTX} LTX-2.5` },
    ];
  }
  const T = (t) => (window.mdT ? window.mdT(t) : t);
  function fillMaskSelect(sel, keys, { value = 'stage', versions = false, placeholder = '', extra = '', avail = null, keep = null } = {}) {
    if (!sel) return;
    const mm = (S.info && S.info.models_missing) || {};
    const cur = keep != null ? keep : sel.value;
    let html = placeholder ? `<option value="">${esc(placeholder)}</option>` : '';
    for (const m of maskSources().filter((x) => keys.includes(x.k))) {
      const has = avail ? avail(m.k) : (m.k === 'core' && value === 'stage') || !!m.run;
      const v = value === 'run' ? m.run : m.k;
      const off = !has || !v;
      const dis = off && !(v && v === cur);
      html += `<option value="${esc(v || `none:${m.k}`)}"${dis ? ' disabled' : ''}${off ? ` title="${esc(mm[m.k] || m.todo)}"` : ''}>`
        + `${esc(T(m.label))}${m.run ? ` · ${esc(m.run)}` : ''}${off ? ` ${esc(T(m.run ? '– nieaktualny' : '– brak wyniku'))}` : ''}</option>`;   // one text node: translated here (i18n.js cannot match the joined parts)
    }
    if (versions) {
      const listed = new Set(value === 'run' ? maskSources().filter((x) => keys.includes(x.k)).map((x) => x.run) : []);
      const rs = (S.runs || []).filter((r) => (versions === 'all' || r.stage === 'experiment') && !listed.has(r.run)).slice().reverse();
      if (rs.length) {
        html += `<optgroup label="${esc(versions === 'all' ? 'Wszystkie wersje' : `Eksperymenty (etap ${ST.EXPERIMENTS})`)}">`
          + rs.map((r) => `<option value="${esc(r.run)}">${esc(runLabel(r))}</option>`).join('') + '</optgroup>';
      }
    }
    html += extra;
    if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; }
    if ([...sel.options].some((o) => o.value === cur && !o.disabled)) sel.value = cur;
  }
  function renderExpPalette() {
    const box = $('expPalette'); const w = $('canvasWrapper'); if (!box || !w) return;
    const show = S.stage === ST.EXPERIMENTS && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (!show) return;
    const sel = $('exBase');
    if (document.activeElement !== sel) {   // stage 6 may start from anything: every stage result, then every version
      fillMaskSelect(sel, ['core', 'matanyone', 'videomama', 'vitmatte', 'ltx'], { value: 'run', versions: 'all', placeholder: '— wybierz —', keep: S.exRun || '' });
      if (sel.value !== (S.exRun || '')) sel.value = '';
    }
    $('exPvEdge').checked = expPvGet('edge', true);
    $('exPvFlick').checked = !!S.flicker;
    if (document.activeElement !== $('exThr')) { $('exThr').value = S.exThr ?? 0.5; $('exThrS').value = S.exThr ?? 0.5; }
    if (document.activeElement !== $('exMin')) { $('exMin').value = S.exMin ?? 5; $('exMinS').value = S.exMin ?? 5; }
    $('exAnalyze').disabled = !S.exRun || document.body.classList.contains('job-busy');
    const canFix = !!(S.edgeFix && S.edgeFix.run === S.exRun && S.edgeFix.pieces.length);
    $('exWithFix').disabled = !canFix; if (!canFix) $('exWithFix').checked = false;
    $('exCompute').disabled = !S.exRun || document.body.classList.contains('job-busy');
    const bl = $('exMode').querySelector('option[value="backlit"]'); if (bl) bl.disabled = !(V() && V().manifest && V().manifest.plates && V().manifest.plates.backlit);
    placeFloating(box, w, expPos.get());
  }
  async function expCompute(withFix) {   // „▶ Przelicz z tej bazy” (replaces „Przelicz finalnie”, user decision 2026-10-16)
    if (!S.exRun) return;
    const fix = withFix && S.edgeFix && S.edgeFix.run === S.exRun ? S.edgeFix.pieces.map((q) => q.id).filter((id) => !S.edgeFix.skip.has(id)) : [];
    if (withFix && !fix.length) { status('Wszystkie odcinki odznaczone – nic do poprawy', 'warn'); return; }
    status(`Nowa wersja z bazy ${S.exRun} (${$('exMode').selectedOptions[0].textContent}${fix.length ? `, poprawka ${fix.length} odcinków` : ''})…`);
    const d = await api('POST', `/api/object/${S.obj}/experiment`, { base: S.exRun, mode: $('exMode').value, fix });
    clearOverlays(); await refreshList(S.obj); await loadStages(); await loadRuns().catch(() => {});
    S.exRun = d.run; showRun(null);   // the new version is active; it becomes the analysed base
    await loadQA(); renderExpPalette();
    status(`Wersja ${d.run} z bazy ${d.base}: QA ${d.gate_ok ? 'OK' : 'flagi'} · migotanie ${d.flicker_median} · krawędź do poprawy ${d.pieces_to_fix ?? 0}${d.dependents_to_recompute && d.dependents_to_recompute.length ? ` · do przeliczenia: ${d.dependents_to_recompute.join(', ')}` : ''}`, 'ok');
  }
  async function expAnalyze() {
    if (!S.exRun) return;
    const d = await api('POST', `/api/object/${S.obj}/analyze`, { run: S.exRun, thr_px: +($('exThr').value) || 0.5, min_frames: Math.max(1, Math.round(+$('exMin').value || 5)) });
    S.exThr = d.edge && d.edge.thr_px; S.exMin = d.edge && d.edge.min_frames;
    clearOverlays(); await loadQA(); renderExpPalette(); V().render();
    const e = d.edge || {};
    $('exInfo').textContent = `${d.run}: QA ${d.gate_ok ? 'OK' : 'flagi'} · migotanie ${d.flicker_median ?? '–'} · krawędź mediana ${e.before ?? '?'} px · do poprawy ${e.pieces_to_fix ?? 0}${d.fix_why ? ` · ${d.fix_why}` : ''}`;
    status(`Analiza ${d.run}: ${e.pieces_to_fix ?? 0} odcinków krawędzi do poprawy`, 'ok');
  }
  // ---- palette „LTX” (user request 2026-10-16): the stage settings over the view, left, like the other stages; the
  // region sliders in px with a live band preview and – when an LTX version exists – its raw matte re-limited live
  function ltxPaletteHtml() {
    return `<div class="pts-palette ltx-palette" id="ltxPalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> LTX-2.5 Alpha-Gen</div>
      <label title="Maska, która przycina i kieruje LTX-2.5: maska 1+2 (SAM 3.1 + poprawki; grupa = całość ze szwem) albo wynik etapu ${ST.MATANYONE} / ${ST.VITMATTE} (alfa > 0,5)">Maska<select id="lx6Mask" class="form-input" title="Z której maski korzysta LTX-2.5">
        <option value="core">Maska 1+2 (etapy ${ST.SAM3} + ${ST.BRUSH})</option><option value="matanyone">MatAnyone2 (etap ${ST.MATANYONE})</option><option value="vitmatte">ViTMatte (etap ${ST.VITMATTE})</option></select></label>
      <div class="pp-chk"><label class="chk" title="Przerywane linie = krawędzie obszaru, w którym zostaje alfa LTX-2.5 (maska + pas na zewnątrz; w środku wymuszone 1 przy „+ wnętrze” / „pas”), cienkie = koniec wtopień"><input type="checkbox" id="lxPvBand" title="Podgląd pasa LTX-2.5 (wł. / wył.)"> 👁 podgląd pasa</label>
        <label class="chk" title="Wynik LTX-2.5 tej klatki przycięty bieżącymi suwakami (z zapisanej surowej alfy LTX-2.5 – bez liczenia modelu)"><input type="checkbox" id="lxPvAlpha" title="Wynik LTX-2.5 z bieżącymi suwakami (wł. / wył.)"> 👁 wynik</label></div>
      <div class="band-seed" id="lxBandSeed"></div>
      <div class="muted pp-hint" id="lxPvInfo"></div>
      <div class="pp-sec">
      <label title="Co model dostaje jako obraz. Izoluj = kadr z zamalowanymi INNYMI obiektami projektu (ich maski 1+2) – model sam wybiera pierwszy plan, więc tak go kierujemy na ten obiekt. Czysty kadr = obraz bez zmian. Izoluj + rozmyj tło = dodatkowo rozmyte wszystko poza obiektem (gdy obiekt nie jest najbliżej kamery)">Przewodnik<select id="lx6Guide" class="form-input" title="Obraz, który dostaje model (izolacja innych obiektów)">
        <option value="plain">Czysty kadr (domyślnie)</option>
        <option value="isolate">Izoluj (inne obiekty zamalowane – tylko obiekty OBOK, nie zasłaniające)</option>
        <option value="steer">Izoluj + rozmyj tło</option></select></label>
      <label title="Dłuższy bok obrazu dla LTX-2.5 (tylko w dół; potem dopełnienie do wielokrotności 32, krótszy bok ≤ 1088). Więcej = więcej detalu, więcej VRAM / czasu">Rozdzielczość LTX-2.5<select id="lx6Side" class="form-input" title="Dłuższy bok obrazu dla LTX-2.5"><option value="768">768 px</option><option value="1024">1024 px</option><option value="1280">1280 px</option><option value="1536">1536 px</option><option value="1920">1920 px</option></select></label>
      <label title="Klatki w jednej paczce modelu (8k+1, maks. 145 – dłuższe paczki: model wpuszcza RGB do maski). Paczki zachodzą na siebie 8 klatek z przenikaniem. Dłuższa = stabilniej w czasie, więcej pamięci">Długość paczki<select id="lx6Frames" class="form-input" title="Klatki w jednej paczce modelu (8k+1)"><option value="25">25 kl.</option><option value="49">49 kl.</option><option value="73">73 kl.</option><option value="97">97 kl.</option><option value="121">121 kl.</option></select></label>
      <label title="Jak wynik modelu staje się alfą obiektu. Region = tylko w masce 1+2 poszerzonej o margines (inne rzeczy, które model uznał za pierwszy plan, odpadają). + wnętrze = dodatkowo pełne wnętrze z maski 1+2 (dziury modelu w obiekcie zamknięte). Surowy = wynik modelu bez zmian (diagnostyka)">Wynik<select id="lx6Combine" class="form-input" title="Jak wynik modelu staje się alfą obiektu"><option value="region">region maski 1+2</option><option value="max_core">region + wnętrze z maski 1+2</option><option value="raw">surowy (bez ograniczenia)</option><option value="band">pas (jeden system: wtopienia w rdzeń, punkty)</option></select></label>        <label class="edge-px" title="O ile px poza maskę alfa LTX-2.5 może wyjść (włosy, rozmycie ruchu); dalej = 0. Suwak 0–120, w polu do 400">Pas na zewnątrz <input id="lx6RegionS" type="range" min="0" max="120" step="1" title="Pas na zewnątrz maski (px) – podgląd na żywo, zapis po puszczeniu"><input id="lx6Region" type="number" min="0" max="400" step="1" class="form-input edge-num" title="Pas na zewnątrz maski (px)"> px</label>
        <label class="edge-px" title="Miękkie przejście na końcu pasa na zewnątrz (px)">Miękkość <input id="lx6SoftS" type="range" min="0" max="40" step="1" title="Miękkość końca pasa (px)"><input id="lx6Soft" type="number" min="0" max="200" step="1" class="form-input edge-num" title="Miękkość końca pasa (px)"> px</label>
        <label class="edge-px" title="Wynik „+ wnętrze”: maska zwężona o tyle px jest wymuszona na 1 (dziury LTX-2.5 w obiekcie zamknięte); bliżej krawędzi decyduje LTX-2.5">Wcięcie w rdzeń <input id="lx6ErodeS" type="range" min="0" max="40" step="1" title="Wcięcie w rdzeń (px), wynik „+ wnętrze”"><input id="lx6Erode" type="number" min="0" max="200" step="1" class="form-input edge-num" title="Wcięcie w rdzeń (px)"> px</label>
        <label class="edge-px lx-band-only" title="Wynik „pas”: na ilu px od wewnętrznej krawędzi pasa rdzeń przechodzi w wynik LTX-2.5 (cieńsza przerywana linia). „Miękkość” = wtopienie od zewnętrznej krawędzi w rdzeń (czerń); nie nachodzą na siebie">Wtopienie wcięcia <input id="lx6FeatherS" type="range" min="0" max="40" step="1" title="Wtopienie wcięcia (px)"><input id="lx6Feather" type="number" min="0" max="200" step="1" class="form-input edge-num" title="Wtopienie wcięcia (px)"> px</label>
        <div class="vm-band-pts lx-band-only"><div class="band-mode" id="lxBandMode"></div><span class="muted pp-hint" id="lxBandInfo"></span>
          <button type="button" class="btn btn-ghost btn-xs" id="lxBandClear" title="Usuń wszystkie punkty bieżącej klatki kluczowej pasa (= usuń klatkę kluczową)">🗑 Usuń wszystkie punkty (ta klatka)</button>
          <label title="Punkty pasa z innego etapu (np. strefy „cały obrys” etapu 5) zastępują punkty tego pasa – szerokości takie, jak tam widać">Kopiuj punkty pasa z <select id="lxBandSrc" class="form-input" title="Skąd skopiować punkty pasa"></select></label>
          <button type="button" class="btn btn-secondary btn-xs" id="lxBandCopy" title="Skopiuj punkty pasa (z kluczami) z wybranego miejsca">⇩ Kopiuj</button></div>
      </div>
      <div class="pp-btns"><button type="button" class="btn btn-primary btn-sm" id="lx6Run" title="LTX-2.5 Alpha-Gen na zakresie pracy (oś czasu) → nowa wersja (wynik etapu ${ST.LTX}, źródło eksportu „LTX-2.5”)">▶ Policz LTX-2.5</button></div>
    </div>`;
  }
  const ltxPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_ltx_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_ltx_palette', JSON.stringify(p)); else localStorage.removeItem('md_ltx_palette'); } catch (e) { /* storage blocked */ } },
  };
  const ltxPvGet = (k, d) => { try { const v = localStorage.getItem(`md_ltx_${k}`); return v == null ? d : v === '1'; } catch (e) { return d; } };
  const ltxPvSet = (k, on) => { try { localStorage.setItem(`md_ltx_${k}`, on ? '1' : '0'); } catch (e) { /* storage blocked */ } };
  function renderLtxPalette() {
    const box = $('ltxPalette'); const w = $('canvasWrapper'); if (!box || !w) return;
    const show = S.stage === ST.LTX && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (!show) return;
    $('lxPvBand').checked = ltxPvGet('band', true);
    $('lxPvAlpha').checked = ltxPvGet('alpha', true);
    const has = !!(S.ltx && S.ltx.run);
    {   // mask of LTX: mask 1+2, stage 3 / 5 results + versions of stage Eksperymenty (stage 6 may use them; 2026-10-16)
      const sel = $('lx6Mask'); const cur = (S.ltx && S.ltx.settings && S.ltx.settings.mask_source) || sel.value;
      fillMaskSelect(sel, ['core', 'matanyone', 'vitmatte'], { versions: 'experiment', keep: cur });
    }
    $('lxPvAlpha').disabled = !has;
    placeFloating(box, w, ltxPos.get());
  }
  // the palette values (also while dragging, before they are saved)
  function ltxPvParams() {
    const n = (id, d) => { const v = +$(id).value; return isFinite(v) && $(id).value !== '' ? Math.max(0, v) : d; };
    return { region_px: n('lx6Region', 24), region_soft_px: n('lx6Soft', 12), core_erode_px: Math.round(n('lx6Erode', 4)),
             combine: $('lx6Combine').value, mask_source: $('lx6Mask').value, feather_px: n('lx6Feather', 0) };
  }
  const ltxQuery = (p) => Object.entries(p).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
  function ltxAlphaOverlay(f) {   // the live re-limited LTX matte of frame f (Image), 'missing' or null (loading)
    if (!(S.ltx && S.ltx.run) || !ltxPvGet('alpha', true)) return 'missing';
    const q = ltxQuery(S.ltxPv || ltxPvParams());
    return settledOverlay(`${S.obj}_ltxa_${S.ltx.run}_${q}_${f}`, `/api/object/${S.obj}/ltx_preview/${f}?what=alpha&${q}`, f);
  }
  function drawLtxPreview(ctx, img, f, bandsOnly = false) {   // bandsOnly: the cut-out (comp) view – no tinted alpha
    if (!(S.stage === ST.LTX && S.obj && !S.auxMode)) return '';
    const p = S.ltxPv || ltxPvParams(); const q = ltxQuery(p);
    let note = '';
    const al = bandsOnly || selMaskHidden() ? null : ltxAlphaOverlay(f);
    if (al && al !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.75; ctx.drawImage(tintCanvas(al, [60, 255, 90], `${S.obj}_ltxa_${S.ltx.run}_${q}_${f}`), 0, 0, img.width, img.height); ctx.globalAlpha = pa; note = `Wynik LTX-2.5 ${S.ltx.run} z bieżącymi suwakami (zielone)`; }
    if (ltxPvGet('band', true) || S.pvDrag === 'ltx') {
      // band mode = the one band system (2026-10-22): dashed band edges + thin feather-end lines, no fill
      // the other result modes (region / + interior / raw): the edges of the server's result area, dashed too (2026-10-23)
      const lp = { reach_px: p.region_px, inset_px: p.core_erode_px, feather_px: p.feather_px, outer_feather_px: p.region_soft_px };
      if (!(p.combine === 'band' && window.MD_BANDPTS && window.MD_BANDPTS.ready && window.MD_BANDPTS.drawLines(ctx, img, f, lp, { live: S.pvDrag === 'ltx' }))) {
        const key = `${S.obj}_ltxb_${q}_${f}`;
        const ov = settledOverlay(key, `/api/object/${S.obj}/ltx_preview/${f}?what=band&${q}`, f);
        drawBand(ctx, img, 'ltx', q, ov, key);
      }
    }
    const info = $('lxPvInfo');
    if (info) info.textContent = `F${f}: pas na zewnątrz ${p.region_px} px, wcięcie ${p.core_erode_px} px, maska: ${{ core: '1+2', matanyone: 'MatAnyone2', vitmatte: 'ViTMatte' }[p.mask_source]}`;
    return note;
  }
  // ---- palette „VideoMaMa” (stage between MatAnyone and ViTMatte, 2026-10-16): base, reach / protect in px with a live
  // band preview and – when a VideoMaMa version exists – its raw matte put over the base live (no model run)
  function vmmPaletteHtml() {
    return `<div class="pts-palette vmm-palette" id="vmmPalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> VideoMaMa</div>
      <label title="Na co nakłada się wynik VideoMaMa. Maska 1+2 (domyślnie) – VideoMaMa robi całą krawędź (na CLIP2_A najspokojniejsza twarda krawędź). MatAnyone2 (etap ${ST.MATANYONE}) – twarda krawędź MatAnyone2 zostaje w pasie ochrony, VideoMaMa dokłada kosmyki dalej. VideoMaMa zawsze prowadzi maska 1+2">Baza<select id="vmBase" class="form-input" title="Na co nakłada się wynik VideoMaMa">
        <option value="core">Maska 1+2 (etapy ${ST.SAM3} + ${ST.BRUSH})</option><option value="matanyone">MatAnyone2 (etap ${ST.MATANYONE})</option></select></label>
      <div class="pp-chk"><label class="chk" title="Przerywane linie = krawędzie pasa, w którym decyduje VideoMaMa (pas na zewnątrz + wcięcie), cienkie = koniec wtopień; w środku zostaje baza"><input type="checkbox" id="vmPvBand" title="Podgląd pasa VideoMaMa (wł. / wył.)"> 👁 podgląd pasa</label>
        <label class="chk" title="Wynik VideoMaMa tej klatki nałożony na bazę bieżącymi suwakami (z zapisanej surowej alfy – bez liczenia modelu)"><input type="checkbox" id="vmPvAlpha" title="Wynik VideoMaMa z bieżącymi suwakami (wł. / wył.)"> 👁 wynik</label></div>
      <div class="band-seed" id="vmBandSeed"></div>
      <div class="muted pp-hint" id="vmPvInfo"></div>
      <div class="pp-sec">
        <label class="edge-px" title="Jak daleko od krawędzi bazy VideoMaMa może dokładać (kosmyki, odstające włosy); dalej zostaje baza – wysepki w tle odcięte. Suwak 0–120, w polu do 400">Pas na zewnątrz <input id="vmReachS" type="range" min="0" max="120" step="1" title="Zasięg od krawędzi bazy (px) – podgląd na żywo, zapis po puszczeniu"><input id="vmReach" type="number" min="0" max="400" step="1" class="form-input edge-num" title="Zasięg od krawędzi bazy (px)"> px</label>
        <label class="edge-px" title="Ile px w głąb bazy VideoMaMa ZASTĘPUJE bazę (wewnętrzna przerywana linia pasa); dalej w środku zostaje baza. 0 = VideoMaMa tylko od krawędzi bazy na zewnątrz. Suwak 0–30, w polu do 200">Wcięcie w rdzeń <input id="vmInsetS" type="range" min="0" max="30" step="1" title="Wcięcie w rdzeń (px) – podgląd na żywo, zapis po puszczeniu"><input id="vmInset" type="number" min="0" max="200" step="1" class="form-input edge-num" title="Wcięcie w rdzeń (px)"> px</label>
        <label class="edge-px" title="Na ilu px od ZEWNĘTRZNEJ krawędzi pasa wynik VideoMaMa przechodzi w rdzeń (bazę; zwykle czerń) – cieńsza przerywana linia na podglądzie. Nie nachodzi na wtopienie wcięcia. 0 = twarde cięcie">Wtopienie krawędzi zewn. <input id="vmOuterFeatherS" type="range" min="0" max="80" step="1" title="Wtopienie krawędzi zewnętrznej (px) – podgląd na żywo, zapis po puszczeniu"><input id="vmOuterFeather" type="number" min="0" max="200" step="1" class="form-input edge-num" title="Wtopienie krawędzi zewnętrznej (px)"> px</label>
        <label class="edge-px" title="Na ilu px wynik VideoMaMa przenika się z głębszą bazą na wewnętrznym brzegu pasa. 0 = twarde przejście">Wtopienie wcięcia <input id="vmFeatherS" type="range" min="0" max="24" step="1" title="Wtopienie wcięcia (px) – podgląd na żywo, zapis po puszczeniu"><input id="vmFeather" type="number" min="0" max="64" step="1" class="form-input edge-num" title="Wtopienie wcięcia (px)"> px</label>
      </div>
      <div class="pp-sec vm-band-pts"><div class="band-mode" id="vmBandMode"></div><span class="muted pp-hint" id="vmBandInfo"></span>
        <button type="button" class="btn btn-ghost btn-xs" id="vmBandClear" title="Usuń wszystkie punkty bieżącej klatki kluczowej pasa (= usuń klatkę kluczową)">🗑 Usuń wszystkie punkty (ta klatka)</button>
        <label title="Punkty pasa z innego etapu (np. strefy „cały obrys” etapu 5) zastępują punkty tego pasa – szerokości takie, jak tam widać">Kopiuj punkty pasa z <select id="vmBandSrc" class="form-input" title="Skąd skopiować punkty pasa"></select></label>
        <button type="button" class="btn btn-secondary btn-xs" id="vmBandCopy" title="Skopiuj punkty pasa (z kluczami) z wybranego miejsca">⇩ Kopiuj</button></div>
      <div class="pp-sec">
        <label title="Dłuższy bok kadru dla VideoMaMa (tylko w dół – kadr nigdy nie jest powiększany). Więcej = więcej detalu, więcej VRAM / czasu (1280 px: ~16 GB, 448 px: ~8 GB)">Rozdzielczość<select id="vmSide" class="form-input" title="Dłuższy bok kadru dla VideoMaMa"><option value="512">512 px</option><option value="768">768 px</option><option value="1024">1024 px</option><option value="1280">1280 px</option><option value="1536">1536 px</option><option value="2048">2048 px</option></select></label>
        <label title="Klatki w jednym przebiegu modelu. Sąsiednie paczki zachodzą na siebie o „zakładkę” z przenikaniem – bez zakładki początek każdej paczki skacze">Paczka<select id="vmBatch" class="form-input" title="Klatki w jednym przebiegu modelu"><option value="12">12 kl.</option><option value="16">16 kl.</option><option value="24">24 kl.</option><option value="32">32 kl.</option></select></label>
        <label title="Klatki wspólne sąsiednich paczek (płynne przejście). 8 przy paczce 24: szwy praktycznie znikają (+35 % czasu)">Zakładka<select id="vmOverlap" class="form-input" title="Klatki wspólne sąsiednich paczek"><option value="0">0 (bez)</option><option value="4">4 kl.</option><option value="8">8 kl.</option><option value="12">12 kl.</option></select></label>
      </div>
      <div class="pp-btns"><button type="button" class="btn btn-secondary btn-sm" id="vmStagePv" title="Policz VideoMaMa tylko na kilku klatkach wokół bieżącej (±6 kl., jedna paczka) – szybkie sprawdzenie ustawień przed długim liczeniem całości. To NIE jest wynik etapu: nic się nie zapisuje jako wersja i nic się nie odblokowuje. Modele czasowe dostają mniej kontekstu niż w pełnym przebiegu – krawędź może się trochę różnić">👁 Podgląd klatek</button><button type="button" class="btn btn-primary btn-sm" id="vmRun" title="VideoMaMa na zakresie pracy (oś czasu) → nowa wersja (wynik etapu ${ST.VIDEOMAMA}: baza ViTMatte do wyboru, źródło eksportu „VideoMaMa”)">▶ Policz VideoMaMa</button></div>
    </div>`;
  }
  const vmmPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_vmm_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_vmm_palette', JSON.stringify(p)); else localStorage.removeItem('md_vmm_palette'); } catch (e) { /* storage blocked */ } },
  };
  const vmmPvGet = (k, d) => { try { const v = localStorage.getItem(`md_vmm_${k}`); return v == null ? d : v === '1'; } catch (e) { return d; } };
  const vmmPvSet = (k, on) => { try { localStorage.setItem(`md_vmm_${k}`, on ? '1' : '0'); } catch (e) { /* storage blocked */ } };
  function renderVmmPalette() {
    const box = $('vmmPalette'); const w = $('canvasWrapper'); if (!box || !w) return;
    const show = S.stage === ST.VIDEOMAMA && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (!show) return;
    $('vmPvBand').checked = vmmPvGet('band', true);
    $('vmPvAlpha').checked = vmmPvGet('alpha', true);
    const has = !!(S.vmm && S.vmm.run);
    $('vmPvAlpha').disabled = !has;
    placeFloating(box, w, vmmPos.get());
  }
  // the palette values (also while dragging, before they are saved)
  function vmmPvParams() {
    const n = (id, d) => { const v = +$(id).value; return isFinite(v) && $(id).value !== '' ? Math.max(0, v) : d; };
    // the unified sliders: inset / feather are sent (and saved) as soon as they exist in the palette – an object never
    // touched still shows inset 0, feather 0, which for the mask 1+2 base is the former composition exactly
    return { base: $('vmBase').value, reach_px: n('vmReach', 40), inset_px: n('vmInset', 0), feather_px: n('vmFeather', 0),
             outer_feather_px: n('vmOuterFeather', 0) };
  }
  const vmmQuery = (p) => Object.entries(p).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
  function drawVmmPreview(ctx, img, f, bandsOnly = false) {
    if (!(S.stage === ST.VIDEOMAMA && S.obj && !S.auxMode && $('vmBase'))) return '';
    const p = S.vmmPv || vmmPvParams(); const q = vmmQuery(p);
    let note = '';
    if (!bandsOnly && !selMaskHidden() && S.vmm && S.vmm.run && vmmPvGet('alpha', true)) {
      const key = `${S.obj}_vmma_${S.vmm.run}_${q}_${f}`;
      const al = settledOverlay(key, `/api/object/${S.obj}/videomama_preview/${f}?what=alpha&${q}`, f);
      if (al && al !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.75; ctx.drawImage(tintCanvas(al, [60, 255, 90], key), 0, 0, img.width, img.height); ctx.globalAlpha = pa; note = `Wynik VideoMaMa ${S.vmm.run} na bazie, bieżące suwaki (zielone)`; }
    }
    if (vmmPvGet('band', true) || S.pvDrag === 'vmm') {
      // instant band in the browser while dragging / until the server band is loaded (2026-10-17): reach outside,
      // inset inside the mask 1+2 edge (the MatAnyone base is approximated by the mask 1+2 here)
      const lp = S.pvDrag === 'vmm' ? vmmPvParams() : p;
      const live = () => {
        if (!(window.MD_ZONES && window.MD_ZONES.liveBand)) return null;
        const kind = S.info && S.info.has && S.info.has.core_fix ? 'core_fix' : 'core';
        const m = overlayFor(kind, f);
        return m && m !== 'missing' ? window.MD_ZONES.liveBand(m, `${S.obj}_${kind}_${f}`, +lp.inset_px || 0, +lp.reach_px || 0) : null;
      };
      // one band system (2026-10-22): dashed band edges + thin dashes where the feathers end, no fill – also while a
      // slider is held (user report 2026-10-23: the old filled band came back while dragging)
      const lp2 = { reach_px: lp.reach_px, inset_px: lp.inset_px, feather_px: lp.feather_px, outer_feather_px: lp.outer_feather_px };
      const pts = window.MD_BANDPTS && window.MD_BANDPTS.ready;
      if (S.pvDrag === 'vmm') {
        if (!(pts && window.MD_BANDPTS.drawLines(ctx, img, f, lp2, { live: true }))) { const c = live(); if (c) ctx.drawImage(c, 0, 0, img.width, img.height); }
      } else {
        if (!(pts && window.MD_BANDPTS.drawLines(ctx, img, f, lp2))) {
          const key = `${S.obj}_vmmb_${q}_${f}`;
          const ov = settledOverlay(key, `/api/object/${S.obj}/videomama_preview/${f}?what=band&${q}`, f);
          if (ov) drawBand(ctx, img, 'vmm', q, ov, key);
          else { const c = live(); if (c) ctx.drawImage(c, 0, 0, img.width, img.height); }
        }
      }
    }
    const info = $('vmPvInfo');
    if (info) info.textContent = `F${f}: baza ${p.base === 'matanyone' ? 'MatAnyone2' : 'maska 1+2'}, pas na zewnątrz ${p.reach_px} px, wcięcie ${p.inset_px} px, wtopienie ${p.feather_px} / ${p.outer_feather_px || 0} px`;
    return note;
  }
  // stage VideoMaMa: settings (object field `videomama`), weights / nodes, the stage result
  async function loadVmm() {
    if (!S.obj || !$('vmBase')) return;
    const d = await api('GET', `/api/object/${S.obj}/videomama`);
    S.vmm = d;
    const s = d.settings;
    {   // base MatAnyone: only with a stage-3 result (user rule 2026-10-16; bug 2026-10-17: it was offered without one –
        // the band preview stayed empty). A saved base without its result falls back to the mask 1+2, with a note.
      const sr = (S.info && S.info.stage_runs) || {};
      fillMaskSelect($('vmBase'), ['core', 'matanyone']);
      $('vmBase').value = s.base === 'matanyone' && !sr.matanyone ? 'core' : s.base;
      if (s.base === 'matanyone' && !sr.matanyone) {
        $('vmBase').value = 'core';
        vmmSetting({ base: 'core' }).catch((er) => status(`Błąd: ${er.message}`, 'err'));
        status(`Baza VideoMaMa: maska 1+2 – wyniku MatAnyone2 (etap ${ST.MATANYONE}) jeszcze nie ma`, 'warn');
      }
    }
    for (const [id, v] of [['vmReach', s.reach_px], ['vmInset', s.inset_px ?? 0], ['vmFeather', s.feather_px ?? 0], ['vmOuterFeather', s.outer_feather_px ?? 0]]) { $(id).value = v; $(`${id}S`).value = v; }
    S.vmmPv = null;
    if (window.MD_BANDPTS && window.MD_BANDPTS.ready) { window.MD_BANDPTS.dirty(); window.MD_BANDPTS.onFrame(); }
    await fillBandSources('vm');
    for (const [id, v] of [['vmSide', s.max_side], ['vmBatch', s.batch], ['vmOverlap', s.overlap]]) {
      const sel = $(id);
      if (![...sel.options].some((o) => +o.value === +v)) sel.appendChild(el(`<option value="${v}">${v}</option>`));
      sel.value = String(v);
    }
    const miss = d.missing_models || [];
    $('vmModels').innerHTML = miss.length
      ? `<span class="err">Brak wag VideoMaMa (${miss.length} plików, ${d.missing_gb} GB)</span> – Doktor → „Zainstaluj VideoMaMa”`
      : (d.missing_nodes === null ? 'Wagi VideoMaMa: OK · ComfyUI nie odpowiada (węzły nie sprawdzone)'
        : (d.missing_nodes.length ? `<span class="err">Comfy bez węzłów: ${d.missing_nodes.map(esc).join(', ')}</span> – Doktor → „Zainstaluj VideoMaMa”, restart ComfyUI` : 'Wagi VideoMaMa i węzły Comfy: OK'));
    const st = stageState('videomama');
    $('vmInfo').textContent = d.run ? `Wynik etapu: ${d.run} ${d.valid ? '✔ aktualny' : `↻ nieaktualny – ${d.why || 'przelicz'}`}`
      : (st.state === 'locked' ? (st.why || '') : 'Brak wyniku – „▶ Policz VideoMaMa”');
  }
  async function vmmSetting(body) {
    if (!S.obj) return;
    await api('PUT', `/api/object/${S.obj}/videomama`, body);
    await loadStages(); await loadVmm();
  }
  // ---- „Motion blur alfy” (user decision 2026-10-16): MatAnyone2 does not reproduce the motion blur (4-5 px ramp on a
  // 12-17 px plate blur), so it is added on a version: vector blur of the alpha along the object's own motion (plate flow
  // x shutter x strength). Floating palette in stages MatAnyone / ViTMatte / Eksperymenty, on the version of that stage
  // (stage 3: its MatAnyone result, stage 4: its ViTMatte result – separately), preview of this frame, „Zastosuj” = new
  // version (a stage's result stays that stage's result, so ViTMatte on the MatAnyone base then reads the blurred one).
  function mbPaletteHtml() {
    return `<div class="pts-palette mb-palette" id="mbPalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Motion blur alfy</div>
      <div class="muted pp-hint" id="mbBase"></div>
      <label class="chk" title="Podgląd tej klatki: wersja etapu z dodanym rozmyciem ruchu (liczone na żywo, nic nie zapisuje). Wyłącz, żeby zobaczyć wersję bez rozmycia"><input type="checkbox" id="mbPv"> 👁 podgląd (ta klatka)</label>
      <div class="pp-sec">
        <label title="Siła rozmycia: mnożnik długości rozmycia (ruch obiektu z przepływu × shutter / 360°); 1 = fizycznie, 0 = bez zmian. Suwak 0–2×, w polu dowolna wartość (do 10×). Rozmycie tylko tam, gdzie obiekt się rusza (krótsze niż 1 px = bez zmian)">Siła: <input id="mbStrN" type="number" class="form-input edge-num" min="0" max="10" step="0.1" value="1" title="Dowolna wartość 0–10× – wpisz i Enter">×<input id="mbStr" type="range" min="0" max="2" step="0.05" value="1"></label>
        <label title="Kąt migawki (shutter, stopnie): jaka część klatki naświetla ruch – 180° = pół klatki (typowo), 360° = cała klatka, 90° = ćwierć. Domyślnie z manifestu ujęcia (shutter_deg). Długość rozmycia = ruch na klatkę × shutter / 360° × siła">Shutter: <input id="mbShutN" type="number" class="form-input edge-num" min="0" max="720" step="5" value="180" title="Kąt migawki w stopniach (0–720) – wpisz i Enter">°<input id="mbShut" type="range" min="0" max="360" step="5" value="180"></label>
        <button type="button" class="btn btn-ghost btn-xs" id="mbShutReset" title="Shutter z manifestu ujęcia">↺ shutter z manifestu</button>
      </div>
      <div class="muted pp-hint" id="mbInfo"></div>
      <div class="pp-btns"><button type="button" class="btn btn-primary btn-sm" id="mbApply" title="Rozmycie ruchu we wszystkich klatkach wersji → nowa wersja (stara zostaje na liście wersji). Wynik etapu zostaje wynikiem etapu">✔ Zastosuj motion blur</button></div>
    </div>`;
  }
  const mbPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_mb_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_mb_palette', JSON.stringify(p)); else localStorage.removeItem('md_mb_palette'); } catch (e) { /* storage blocked */ } },
  };
  const mbStage = () => S.stage === ST.MATANYONE || S.stage === ST.VITMATTE || S.stage === ST.EXPERIMENTS;
  function mbBase() {   // the version the palette works on: the result of the current stage
    const sr = (S.info && S.info.stage_runs) || {};
    if (S.stage === ST.MATANYONE) return sr.matanyone || '';
    if (S.stage === ST.VITMATTE) return vmRunFor(vmEngine());   // the engine's version shown in the HUD (2026-10-22)
    return S.viewRun || S.activeRun || '';
  }
  const mbStrength = () => { const v = +$('mbStrN').value; return isFinite(v) && v >= 0 ? Math.min(10, v) : 1; };
  const mbShutterManifest = () => { const m = V() && V().manifest; const v = m ? +m.shutter_deg : NaN; return isFinite(v) && v > 0 ? v : 180; };
  const mbShutter = () => { const v = +$('mbShutN').value; return isFinite(v) && v >= 0 ? Math.min(720, v) : mbShutterManifest(); };
  const mbPreviewOn = () => !!(S.mbPv && mbStage() && S.obj && !S.auxMode && mbBase());
  function renderMbPalette() {
    const box = $('mbPalette'); const w = $('canvasWrapper'); if (!box || !w) return;
    const show = mbStage() && !!S.obj && !S.auxMode && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (!show) return;
    const base = mbBase();
    const what = S.stage === ST.MATANYONE ? `wynik MatAnyone2 (etap ${ST.MATANYONE})` : S.stage === ST.VITMATTE ? `wynik ViTMatte (etap ${ST.VITMATTE})` : 'oglądana / aktywna wersja';
    const r = (S.runs || []).find((x) => x.run === base);
    $('mbBase').textContent = base ? `Na: ${base} – ${what}${r && r.motion_blur ? ` (ma już motion blur ×${r.motion_blur.strength})` : ''}`
      : `Brak wersji – najpierw ${S.stage === ST.VITMATTE ? '„Zastosuj na ujęcie” (ViTMatte)' : '„▶ Przelicz MatAnyone2”'}`;
    $('mbPv').checked = !!S.mbPv; $('mbPv').disabled = !base;
    if (S.mbShutFor !== S.obj + '|' + mbShutterManifest()) {   // a new project / object: the manifest's shutter
      S.mbShutFor = S.obj + '|' + mbShutterManifest();
      $('mbShutN').value = mbShutterManifest(); $('mbShut').value = mbShutterManifest();
    }
    $('mbApply').disabled = !base || document.body.classList.contains('job-busy') || mbStrength() <= 0 || mbShutter() <= 0;
    if (!S.mbPv) $('mbInfo').textContent = '';
    placeFloating(box, w, mbPos.get());
  }
  function mbOverlay(f) {   // Image | 'missing' | null – the preview alpha of frame f (same format as view=alpha)
    const base = mbBase(), s = mbStrength(), sh = mbShutter();
    return settledOverlay(`${S.obj}_mb@${base}x${s}s${sh}_${f}`, `/api/object/${S.obj}/motion_blur/${f}?v=${encodeURIComponent(base)}&strength=${s}&shutter=${sh}`, f);
  }

  function buildPanes() {
    const host = $('stagePanel');
    const add = (n, html, hooks) => { const e = el(html); e.style.display = 'none'; host.appendChild(e); PANES[n] = { el: e, ...hooks }; };

    add(ST.SAM3, `<div class="stage-pane" data-stage="sam3">
      ${paneTitle('SAM3', 'wykryj + rdzeń')}
      ${descHtml('SAM3')}
      <div class="pane-section" id="st1ObjBox">
        <div class="row"><b id="st1Name"></b><button class="btn btn-ghost btn-xs" id="st1Delete" title="Usuń obiekt (folder do kosza projektu)">🗑 usuń</button></div>
        <label title="Np. suknia Figure: część Figure. Eksport: cała postać bez szwu (<rodzic>_full) + osobne maski części">Relacja<select id="st1PartOf" class="form-input"></select></label>
        <div class="muted" id="st1Status"></div>
      </div>
      <div class="muted pane-section">Lista obiektów (wybór, kolejność głębi, części) – w palecie „Obiekty” nad obrazem.</div>
      <button class="btn btn-secondary" id="st1Create" title="Nowy obiekt do wycięcia – okno z nazwą i promptem">＋ Nowy obiekt</button>
      <div class="pane-section">
        <div class="sec-title">Punkty na klatce → klucz</div>
        <div class="muted">Narzędzia na pływającej palecie: <b class="ok">Punkt + (G)</b> – dodaj to, co wskazujesz, albo <b class="err">Punkt − (R)</b> – odejmij – bez narzędzia klik niczego nie dodaje (przeciąganie = przesuwanie). PPM = odwrotny znak, Alt (przytrzymaj) = łapka, klik na punkt = usuń, Esc = wyłącz narzędzie. Prompt (opcjonalny) to baza maski. Postaw punkty, potem <b>„▶ Przelicz klatkę” (Enter)</b> – SAM 3.1 liczy maskę i zapisuje ją jako klucz tej klatki (różowa nakładka). Punkty jeszcze nieprzeliczone są pomarańczowe. „✕ Wyczyść punkty” + „Przelicz klatkę” usuwa klucz. Te same przyciski i pole promptu są na pływającej palecie.</div>
        <div id="st1PtsInfo"></div>
        <div class="row"><button class="btn btn-primary btn-sm" id="st1PtsRedo" title="Policz maskę tej klatki: prompt (baza) + punkty + (dodaj) − punkty − (odejmij); zapis jako klucz (Enter)">▶ Przelicz klatkę</button><button class="btn btn-secondary btn-sm" id="st1PtsClear" title="Usuń wszystkie punkty tej klatki (klucz zniknie po „Przelicz”)">✕ Wyczyść punkty</button></div>
        <div class="row"><button class="btn btn-primary" id="st1All" title="Kadry (automatycznie, gdy trzeba) + rdzeń SAM 3.1 na całym ujęciu – maska w każdej klatce. Po zmianie kluczy liczy tylko zmienione przedziały">▶▶ Przelicz całość</button>
          <button class="btn btn-secondary" id="st3Rerun" title="Przelicza tylko przedziały zmienione przez klucze">↻ Przelicz zmienione przedziały</button></div>
      </div>
      <div class="pane-section">
        <div class="sec-title">Korekta klucza SAM 3.1 (zła klatka)</div>
        <div class="muted">Popraw <b>tylko</b> złe miejsce (lewy pasek): <b>Korekta + (K)</b> – klik w dziurę / pominięty fragment = dodaj, <b>Korekta − (L)</b> – klik na wyciek = usuń (PPM = odwrotnie) (SAM 3.1, reszta maski bez zmian; zielone = dodane, czerwone = usunięte). Pędzel klucza + / − i wielokąt P (Enter) – też od razu klucz SAM3. Potem „Przelicz zmienione przedziały”. W klatce z kluczem wynik = klucz. Poprawka bez SAM 3.1 (pędzel propagowany CoTrackerem) = etap ${ST.BRUSH}.</div>
        <div class="row"><button class="btn btn-secondary" id="st3KeySave" title="Zapisz bieżące kreski / wielokąty jako klucz SAM 3.1 tej klatki (zwykle zapisuje się samo)">🔑 Zapisz klucz (ta klatka)</button><button class="btn btn-ghost" id="st3KeyDel" title="Usuwa klucz bieżącej klatki (Ctrl+Z cofa)">🗑 Usuń klucz</button></div>
      </div>
      <div class="pane-section"><div id="st1Keys" class="muted"></div></div>
      <details class="pane-section" id="st1CropsBox"><summary title="Kadry pod obiekt dla SAM 3.1 – liczone same przez „▶▶ Przelicz całość”; rozwinięcie pokazuje ramki na podglądzie (żółte)">Kadry SAM 3.1 (automatyczne)</summary>
        <div class="muted">SAM 3.1 liczy maskę na obrazie <b>najwyżej 1008 px</b>; kadr przycięty pod obiekt daje mu dużo więcej szczegółu niż cała klatka 4K. „▶▶ Przelicz całość” liczy kadry sam, gdy ich brak albo klucz z nich wychodzi.</div>
        <button class="btn btn-secondary btn-sm" id="st2Crops" title="Wyznacz kadry obiektu dla SAM 3.1 od nowa (zwykle niepotrzebne – robi to „▶▶ Przelicz całość”)">⬚ Kadruj od nowa</button>
        <div id="st2List"></div></details>
      <div class="stage-status" id="st1Status2"></div>
      ${nextBtn(ST.SAM3)}
    </div>`, { show() { fillObjectForm(); updatePointsBox(); renderKeyLists(); renderCrops(); } });

    add(ST.BRUSH, `<div class="stage-pane" data-stage="cotracker">
      ${paneTitle('BRUSH', 'poprawki pędzlem')}
      ${descHtml('BRUSH')}
      <div id="b2Panel"></div>
      <div class="stage-status" id="st${ST.BRUSH}Status"></div>
      ${nextBtn(ST.BRUSH)}
    </div>`, { show() { if (window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.onShow(); } });

    add(ST.MATANYONE, `<div class="stage-pane" data-stage="matanyone">
      ${paneTitle('MATANYONE', 'krawędź z maski 1+2')}
      ${descHtml('MATANYONE')}
      <div class="muted">Suwaki krawędzi i podgląd pasa: paleta „Krawędź (MatAnyone2)” na podglądzie (przeciągnij za ⠿).</div>
      <button class="btn btn-primary" id="st3Ma" title="MatAnyone2 na masce 1+2 (rdzeń SAM 3.1 + poprawki pędzlem z etapu ${ST.BRUSH}), bez stref ViTMatte → nowa wersja (baza etapu ${ST.VITMATTE}); zakres pracy z osi czasu">▶ Przelicz MatAnyone2</button>
      <div class="muted" id="st3MaInfo"></div>
      <div class="stage-status" id="st${ST.MATANYONE}Status"></div>
      ${nextBtn(ST.MATANYONE)}
    </div>`, { show() { renderEdgePalette(); renderStageInfo(); } });

    add(ST.VIDEOMAMA, `<div class="stage-pane" data-stage="videomama">
      ${paneTitle('VIDEOMAMA', 'kosmyki z maski 1+2 na bazie')}
      ${descHtml('VIDEOMAMA')}
      <div class="muted" id="vmModels"></div>
      <div class="muted">Ustawienia i podgląd: paleta „VideoMaMa” na obrazie (po lewej, przeciągnij za ⠿).</div>
      <div class="row"><button class="btn btn-secondary btn-sm" id="vmShow" title="Pokaż wynik VideoMaMa (widok Alpha / Comp)">👁 Wynik VideoMaMa</button>
        <button class="btn btn-secondary btn-sm" id="vmShowMa" title="Pokaż wynik MatAnyone2 (etap ${ST.MATANYONE}) – do porównania">MatAnyone2</button>
        <button class="btn btn-ghost btn-sm" id="vmShowOff" title="Wróć do aktywnej wersji">Aktywna</button></div>
      <div class="muted" id="vmInfo"></div>
      <div class="stage-status" id="st${ST.VIDEOMAMA}Status"></div>
      ${nextBtn(ST.VIDEOMAMA)}
    </div>`, { show() { loadVmm().then(renderVmmPalette).catch((e) => status(`VideoMaMa: ${e.message}`, 'err')); } });

    add(ST.VITMATTE, `<div class="stage-pane" data-stage="vitmatte">
      ${paneTitle('VITMATTE', 'strefy na bazie: MatAnyone2 albo maska 1+2')}
      ${descHtml('VITMATTE')}
      <div id="st4BaseBox" class="zp-base"><label title="Na czym ViTMatte kładzie swoje strefy (poza strefami zostaje baza). MatAnyone2 (etap ${ST.MATANYONE}) = miękka krawędź modelu wszędzie poza strefą; maska 1+2 = twarda maska SAM 3.1 + poprawki pędzla – etap dostępny zaraz po etapach ${ST.SAM3} + ${ST.BRUSH}, bez liczenia MatAnyone2. Każda baza ma własny wynik – porównaj je niżej">Baza ViTMatte<select id="st4BaseSel" class="form-input">
        <option value="matanyone" title="Wersja MatAnyone2 z etapu ${ST.MATANYONE} (musi być aktualna)">MatAnyone2 (etap ${ST.MATANYONE})</option>
        <option value="core" title="Maska 1+2: rdzeń SAM 3.1 + poprawki pędzla (etapy ${ST.SAM3} + ${ST.BRUSH}), twarda 0 / 1 poza strefą">Maska 1+2 (etapy ${ST.SAM3} + ${ST.BRUSH})</option>
        <option value="videomama" title="Wynik VideoMaMa z etapu ${ST.VIDEOMAMA} (musi być aktualny) – kosmyki VideoMaMa poza strefą">VideoMaMa (etap ${ST.VIDEOMAMA})</option></select></label>
      <div class="muted" id="st4Base"></div></div>
      <div class="pane-section" id="st4ZonePanel"><div id="z4Panel"></div></div>
      <div class="pane-section" id="st4Cmp">
        <div class="sec-title" title="Porównanie dwóch wersji obiektu na podglądzie (widok Alpha): A, B albo różnica |A − B| (czerwona, ×3)">Porównanie wersji</div>
        <div class="row"><label title="Wersja A (domyślnie: ViTMatte na masce 1+2)">A<select id="st4CmpA" class="form-input"></select></label>
          <label title="Wersja B (domyślnie: ViTMatte na MatAnyone2)">B<select id="st4CmpB" class="form-input"></select></label></div>
        <div class="row"><button type="button" class="btn btn-secondary btn-sm" id="st4CmpShowA" title="Pokaż wersję A (widok Alpha / Comp)">A</button>
          <button type="button" class="btn btn-secondary btn-sm" id="st4CmpShowB" title="Pokaż wersję B">B</button>
          <button type="button" class="btn btn-secondary btn-sm" id="st4CmpDiff" title="Różnica |A − B| na czerwono (×3, żeby słabe różnice półprzezroczystości były widoczne) na wersji A">Różnica</button>
          <button type="button" class="btn btn-ghost btn-sm" id="st4CmpOff" title="Koniec porównania – aktywna wersja">Wył.</button></div>
        <div class="muted" id="st4CmpInfo"></div>
      </div>
      <div class="stage-status" id="st${ST.VITMATTE}Status"></div>
      ${nextBtn(ST.VITMATTE)}
    </div>`, { show() { renderStageInfo(); loadRuns().then(renderCmp).catch(() => {}); if (window.MD_ZONES && window.MD_ZONES.ready) window.MD_ZONES.onShow(); } });

    add(ST.EXPERIMENTS, `<div class="stage-pane" data-stage="experiments">
      ${paneTitle('EXPERIMENTS', 'tryby krawędzi, QA, wersje')}
      ${descHtml('EXPERIMENTS')}
      <div class="muted">Paleta „Eksperymenty” na obrazie: <b>maska bazowa</b> (wynik etapu ${ST.MATANYONE} / ${ST.VITMATTE} / ${ST.LTX}, maska 1+2 albo dowolna wersja) → „▶ Analizuj” (QA + propozycje krawędzi) → „▶ Przelicz z tej bazy” (silnik krawędzi, strefy progowe, opcjonalnie poprawka krawędzi bazy) = nowa wersja z zapisanym pochodzeniem. Ustawienia pasa krawędzi: paleta „Krawędź” w etapie ${ST.MATANYONE}.</div>
      <div style="display:none">      <label title="Źródło krawędzi (SPEC 5.5); zmiana = nowa wersja przy „Przelicz finalnie”">Krawędź<select id="st4Edge" class="form-input"></select></label>
      <div class="muted" id="st4EdgeWhy"></div>
      <label id="o4BacklitL" class="chk" title="Pozwól użyć backlitu jako źródła krawędzi (gdy jest wczytany)"><input type="checkbox" id="o4Backlit" checked> Użyj backlit</label>
      <button class="btn btn-primary" id="st4Final" title="Policz finalną alphę z rdzenia: wybrany tryb krawędzi + WSZYSTKIE strefy (pędzel, ViTMatte, progowe) → nowa wersja (poprzednie zostają)">✔ Przelicz finalnie</button>
      </div>
      <details class="pane-section" id="z4Adv"><summary title="Stare kryteria progowe (ciemne / jasne / kolor / wszystko); działają na masce bazy przy „▶ Przelicz z tej bazy”">Zaawansowane: strefa progowa (ciemne / jasne / kolor / wszystko)</summary>
        <div class="muted">Narysuj obszar wielokątem (P) albo pędzlem na tej klatce, ustaw kryterium i zastosuj – działa na rdzeniu przed krawędzią, podąża za obiektem.</div>
        <div class="row"><select id="z4Mode" class="form-input"><option value="add">Dodaj</option><option value="remove">Usuń</option></select>
          <select id="z4Crit" class="form-input"><option value="dark">ciemne</option><option value="bright">jasne</option><option value="color">kolor</option><option value="all">wszystko</option></select></div>
        <div class="row" id="z4ColorRow" style="display:none"><span id="z4Swatch" title="Kolor strefy (brak = mediana narysowanego obszaru)" style="display:inline-block;width:22px;height:22px;border-radius:4px;border:1px solid #6e6e75;background:repeating-linear-gradient(45deg,#3c3c42 0 4px,#2a2a2e 4px 8px)"></span>
          <button class="btn btn-secondary btn-sm" id="z4Pick">🎯 Pipeta</button><span class="muted" id="z4ColorTxt">kolor: mediana obszaru</span></div>
        <label id="z4ParamL">próg: <span id="z4ParamV">0.15</span><input id="z4Param" type="range" min="0" max="1" step="0.01" value="0.15"></label>
        <div class="row"><label class="chk"><input type="checkbox" id="z4Soft"> miękko</label><label class="chk"><input type="checkbox" id="z4Static"> nieruchoma</label></div>
        <label id="z4SoftL" style="display:none">miękkość: <span id="z4SoftV">0.05</span><input id="z4Softness" type="range" min="0.005" max="0.3" step="0.005" value="0.05"></label>
        <div class="row"><button class="btn btn-secondary btn-sm" id="z4Preview">👁 Podgląd (ta klatka)</button><button class="btn btn-secondary btn-sm" id="z4Apply">✚ Zapisz strefę progową</button></div>
      </details>
      <div class="pane-section">
        <div class="sec-title">QA</div>
        <div id="st4QA" class="muted"></div>
        <div id="st4Banner" class="banner" style="display:none"></div>
        <button class="btn btn-secondary btn-sm" id="st4NextFlag" title="Skok do następnej klatki z flagą QA [N]">⚑ Następna flaga (N)</button>
        <div class="muted">Comp: V zmienia tło (czarne / białe / 18 % / szachownica / kolor). M = mapa migotania wersji.</div>
      </div>
      <div class="pane-section" id="st4EdgeFix"></div>
      <div class="pane-section"><div class="sec-title">Wersje <span class="muted">(klik = podgląd, B = aktywna ↔ poprzednia)</span></div><div id="st4Runs" class="muted"></div></div>
      <div class="stage-status" id="st${ST.EXPERIMENTS}Status"></div>
      ${nextBtn(ST.EXPERIMENTS)}
    </div>`, { show() { renderEdgePalette(); loadQA().then(renderExpPalette).catch(() => {}); } });

    add(ST.LTX, `<div class="stage-pane" data-stage="ltx">
      ${paneTitle('LTX', 'generatywna alfa LTX-2.5 Alpha-Gen')}
      ${descHtml('LTX')}
      <div class="muted" id="lx6Models"></div>
      <div class="muted">Ustawienia LTX-2.5 i podgląd: paleta „LTX-2.5” na obrazie (po lewej, przeciągnij za ⠿).</div>
      <div class="row"><button class="btn btn-secondary btn-sm" id="lx6Show" title="Pokaż wynik LTX-2.5 (widok Alpha / Comp)">👁 Wynik LTX-2.5</button>
        <button class="btn btn-secondary btn-sm" id="lx6ShowMa" title="Pokaż wynik MatAnyone2 (etap ${ST.MATANYONE}) – do porównania">MatAnyone2</button>
        <button class="btn btn-ghost btn-sm" id="lx6ShowOff" title="Wróć do aktywnej wersji">Aktywna</button></div>
      <div class="muted" id="lx6Info"></div>
      <div class="stage-status" id="st${ST.LTX}Status"></div>
      ${nextBtn(ST.LTX)}
    </div>`, { show() { loadLtx().then(renderLtxPalette).catch((e) => status(`LTX-2.5: ${e.message}`, 'err')); } });

    add(ST.FINAL_TOUCH, `<div class="stage-pane" data-stage="final_touch">
      ${paneTitle('FINAL_TOUCH', 'ręczne korekty')}
      ${descHtml('FINAL_TOUCH')}
      <div id="ftPanel"></div>
      <div class="stage-status" id="st${ST.FINAL_TOUCH}Status"></div>
      ${nextBtn(ST.FINAL_TOUCH)}
    </div>`, { show() { if (window.MD_FT && window.MD_FT.ready) window.MD_FT.onShow(); } });
    add(ST.EXPORT, `<div class="stage-pane" data-stage="export">
      ${paneTitle('EXPORT')}
      ${descHtml('EXPORT')}
      <label title="Z którego etapu wziąć maskę. Rdzeń (1+2) = SAM 3.1 + poprawki pędzlem (twarda maska; nieaktualna maska 1+2 liczy się przy eksporcie); MatAnyone2 / ViTMatte = wynik etapu 3 / 4 (musi być aktualny); wersja z listy = dowolna wersja z etapu ${ST.EXPERIMENTS}; aktywna = jak dotąd (aktualny wynik finalny)">Źródło maski<select id="e6Source" class="form-input">
        <option value="core" title="Rdzeń SAM 3.1 + poprawki pędzlem z etapu ${ST.BRUSH} – twarda maska 0/1; dostępne zaraz po etapie ${ST.SAM3}">Rdzeń (1+2)</option>
        <option value="matanyone" title="Miękka krawędź MatAnyone2 z etapu ${ST.MATANYONE}">MatAnyone2 (etap ${ST.MATANYONE})</option>
        <option value="videomama" title="VideoMaMa na bazie (maska 1+2 albo MatAnyone2) z etapu ${ST.VIDEOMAMA}">VideoMaMa (etap ${ST.VIDEOMAMA})</option>
        <option value="vitmatte" title="Baza + strefy ViTMatte z etapu ${ST.VITMATTE}">ViTMatte (etap ${ST.VITMATTE})</option>
        <option value="ltx" title="Generatywna alfa LTX-2.5 Alpha-Gen z etapu ${ST.LTX}">LTX-2.5 (etap ${ST.LTX})</option>
        <option value="version" title="Wersja wybrana z listy niżej (lista wersji z etapu ${ST.EXPERIMENTS})">Wersja z listy (etap ${ST.EXPERIMENTS})</option>
        <option value="active" title="Aktywna wersja z aktualnym wynikiem finalnym (zachowanie sprzed 2026-10-15)">Aktywna wersja</option>
        <option value="final_touch" title="Baza + ręczne korekty z etapu ${ST.FINAL_TOUCH} (obiekt bez korekt = sama baza)">Final Touch (etap ${ST.FINAL_TOUCH})</option></select></label>
      <label id="e6RunL" style="display:none" title="Która wersja wybranego obiektu (tylko „⇪ Ten obiekt” / „⇪ Całość”; przy eksporcie wszystkich – aktywne wersje)">Wersja<select id="e6Run" class="form-input"></select></label>
      <div class="muted" id="e6SrcInfo"></div>
      <div class="muted">Pełne klatki z wycinków: <code>mattes/&lt;obiekt&gt;/</code> (alpha: EXR albo PNG 8-bit – wg formatu) + przy EXR multi-EXR ujęcia (<code>&lt;obiekt&gt;.alpha</code>, <code>union.alpha</code>) i <code>outputs.json</code>. Obiekty z przodu są odejmowane od tych z tyłu (z tego samego źródła, a gdy go nie mają – z niższego etapu).</div>
      <div class="row"><label class="chk" title="Jeden plik EXR na klatkę z kanałami wszystkich obiektów"><input type="checkbox" id="e5Multi" checked> multi-EXR</label><label class="chk" title="Dodaj też rdzeń SAM 3.1 (twarda maska) do eksportu"><input type="checkbox" id="e5Core"> rdzeń</label><label class="chk" title="Dodaj kanał głębi"><input type="checkbox" id="e5Depth"> głębia</label></div>
      <label title="Format masek (project.json → export_format; manifest bez zmian). Auto: PNG 8-bit dla źródeł 8-bit (MP4 / PNG / JPG), EXR dla EXR. Multi-EXR tylko przy EXR">Format masek<select id="e5Format" class="form-input"><option value="auto">auto</option><option value="png">PNG 8-bit</option><option value="exr">EXR (half)</option></select></label>
      <label title="Katalog eksportu (project.json → export_dir; manifest bez zmian). Poza projektem – potwierdzenie">Katalog eksportu<input id="e5Dir" class="form-input" readonly></label>
      <div class="row"><button class="btn btn-secondary btn-sm" id="e5DirPick" title="Wybierz katalog eksportu">📁 Wybierz…</button><button class="btn btn-ghost btn-sm" id="e5DirDefault" title="Przywróć domyślny katalog &lt;work&gt;/matte_desk/export">domyślny</button><button class="btn btn-ghost btn-sm" id="e5NkOnly" title="Tylko skrypt Nuke'a (backdrop MASKI, anchory + stampy) dla istniejącego eksportu – bez ponownego zapisu EXR">tylko .nk</button></div>
      <label title="Które klatki eksportować. „całość” wymaga wyniku na całym ujęciu – inaczej okno z listą brakujących klatek (nic nie zapisuje się z dziurami). Zakres pracy = znaczniki [ ] na osi czasu (Shift+B / Shift+N, Alt+X = całe ujęcie)">Zakres eksportu<select id="e5Scope" class="form-input" title="całość = całe ujęcie (wymaga wyniku na wszystkich klatkach); work area = tylko zakres pracy z osi czasu"><option value="all">całość (całe ujęcie)</option><option value="work_area">work area (zakres pracy)</option></select></label>
      <button class="btn btn-primary" id="st5Export" title="Eksport masek wszystkich obiektów z wybranego źródła (z przodu odejmowane od tych z tyłu)">⇪ Eksport wszystkich obiektów</button>
      <div class="row"><button class="btn btn-secondary" id="st5ExportOne" title="Tylko wybrany obiekt (liczony jak w pełnym eksporcie), z wybranego źródła">⇪ Ten obiekt</button><button class="btn btn-secondary" id="st5ExportWhole" style="display:none" title="Cała postać złożona z części, bez szwu (&lt;obiekt&gt;_full), z wybranego źródła">⇪ Całość</button></div>
      <div class="pane-section">
        <div class="sec-title">Biblioteka i kolejka</div>
        <div class="row"><button class="btn btn-secondary btn-sm" id="st5LibSave" title="Zapisuje nazwę, prompt, ustawienia krawędzi i miniatury">📚 Do biblioteki</button><button class="btn btn-secondary btn-sm" id="st5LibAdd" title="Utwórz obiekt z zapisanego w bibliotece (prompt + ustawienia)">📚 Z biblioteki</button></div>
        <div class="row"><button class="btn btn-secondary btn-sm" id="st5QAdd" title="Dodaje „Przelicz finalnie” tego obiektu do kolejki">🌙 Do kolejki</button><button class="btn btn-secondary btn-sm" id="st5QRun" title="Policz kolejkę zadań (np. na noc)">▶ Uruchom kolejkę</button></div>
        <label class="btn btn-secondary btn-sm file-btn" title="Wczytaj zaznaczenia z ‘Przeglądu klatek’ jako klucze">⤓ Import kresek z „Przeglądu klatek” (JSON)<input type="file" id="st5RevFile" accept=".json" style="display:none"></label>
      </div>
      <div class="stage-status" id="st${ST.EXPORT}Status"></div>
    </div>`, { show() { loadExportDir(); renderExportSource(); } });
    // the shared edge settings live in a hidden holder until the first pane that shows them
    const cw = $('canvasWrapper');
    if (cw) { const pal = el(edgeSettingsHtml()); cw.appendChild(pal); makeFloating(pal, cw, '.pp-head', epPos); }
    if (cw) { const mp = el(mbPaletteHtml()); cw.appendChild(mp); makeFloating(mp, cw, '.pp-head', mbPos); }
    if (cw) { const lp = el(ltxPaletteHtml()); cw.appendChild(lp); makeFloating(lp, cw, '.pp-head', ltxPos); }
    if (cw) { const vp = el(vmmPaletteHtml()); cw.appendChild(vp); makeFloating(vp, cw, '.pp-head', vmmPos); }
    if (cw) { const xp = el(expPaletteHtml()); cw.appendChild(xp); makeFloating(xp, cw, '.pp-head', expPos); }
  }
  // stage MatAnyone / ViTMatte: which version is the result of the stage, is it current
  function renderStageInfo() {
    const runs = (S.info && S.info.stage_runs) || {};
    const ma = $('st3MaInfo');
    if (ma) {
      const st = stageState('matanyone');
      ma.textContent = runs.matanyone ? `Wynik etapu: ${runs.matanyone} ${st.state === 'done' ? '✔ aktualny' : `↻ nieaktualny – ${st.why || 'przelicz'}`}`
        : (st.state === 'locked' ? (st.why || '') : 'Brak wyniku – „▶ Przelicz MatAnyone2”');
    }
    // the base selector lives at the top of the floating palette „Strefa” (user request 2026-10-16: what ViTMatte computes
    // from matters most) – moved once into its slot, ids and handlers unchanged
    const bb = $('st4BaseBox'), slot = $('zpBaseSlot');
    if (bb && slot && bb.parentNode !== slot) slot.appendChild(bb);
    const vb = $('st4Base');
    if (vb) {
      const base = (S.info && S.info.vitmatte_base) || 'matanyone';
      const bs = $('st4BaseSel'); if (bs && document.activeElement !== bs) bs.value = base;
      // user rule 2026-10-16: a base whose stage has no model (and no result) cannot be chosen
      // bug 2026-10-17: also a base whose stage has no result yet (MatAnyone / VideoMaMa not computed) is not offered
      if (bs && document.activeElement !== bs) fillMaskSelect(bs, ['core', 'matanyone', 'videomama'], { keep: base });
      const vs = stageState('vitmatte');
      const res = runs.vitmatte ? ` · wynik ViTMatte: ${runs.vitmatte} ${vs.state === 'done' ? '✔' : '↻ ' + (vs.why || '')}` : '';
      if (base === 'core') {
        const cs = stageState('cotracker');
        vb.textContent = `Baza: maska 1+2 (etapy ${ST.SAM3} + ${ST.BRUSH}) ${cs.state === 'done' ? '✔' : '↻ najpierw „▶▶ Zastosuj poprawki” (etap ' + ST.BRUSH + ')'}` + res;
      } else if (base === 'videomama') {
        const vs2 = stageState('videomama');
        vb.textContent = runs.videomama ? `Baza: ${runs.videomama} (VideoMaMa, etap ${ST.VIDEOMAMA}) ${vs2.state === 'done' ? '✔' : '↻ nieaktualna – ' + (vs2.why || 'najpierw przelicz etap ' + ST.VIDEOMAMA)}` + res
          : `Brak bazy – „▶ Policz VideoMaMa” (etap ${ST.VIDEOMAMA}) albo inna baza`;
      } else {
        const ms = stageState('matanyone');
        vb.textContent = runs.matanyone ? `Baza: ${runs.matanyone} (MatAnyone2, etap ${ST.MATANYONE}) ${ms.state === 'done' ? '✔' : '↻ nieaktualna – najpierw przelicz etap ' + ST.MATANYONE}` + res
          : `Brak bazy – „▶ Przelicz MatAnyone2” (etap ${ST.MATANYONE}) albo baza = maska 1+2`;
      }
    }
  }
  // ---- comparison of two versions (stage 4, 2026-10-15): A / B on the view, or |A − B| in red over A
  const baseName = (b) => ({ core: 'masce 1+2', videomama: 'VideoMaMa' }[b] || 'MatAnyone2');
  const stageOfRun = (r) => (r.stage === 'matanyone' ? `etap ${ST.MATANYONE} MatAnyone2`
    : r.stage === 'vitmatte' ? `etap ${ST.VITMATTE} ViTMatte na ${baseName(r.base)}`
    : r.stage === 'videomama' ? `etap ${ST.VIDEOMAMA} VideoMaMa na ${r.base === 'matanyone' ? 'MatAnyone2' : 'masce 1+2'}`
    : r.stage === 'core' ? 'maska 1+2 (baza ViTMatte)' : r.stage === 'ltx' ? `etap ${ST.LTX} LTX-2.5 Alpha-Gen`
    : r.stage === 'experiment' ? `eksperyment z ${r.base_run || '?'}` : r.stage);
  const runLabel = (r) => `${r.run}${r.active ? ' ★' : ''} · ${r.stage === 'vitmatte' ? `ViTMatte na ${baseName(r.base)}`
    : r.stage === 'videomama' ? `VideoMaMa na ${r.base === 'matanyone' ? 'MatAnyone2' : 'masce 1+2'}`
    : r.stage === 'matanyone' ? 'MatAnyone2' : r.stage === 'core' ? 'maska 1+2' : r.stage === 'ltx' ? 'LTX-2.5 Alpha-Gen'
    : r.stage === 'experiment' ? `eksperyment z ${r.base_run || '?'}` : (r.edge_mode || '')}${r.motion_blur ? ` + motion blur ×${r.motion_blur.strength}` : ''}`;
  function renderCmp() {
    const a = $('st4CmpA'), b = $('st4CmpB'); if (!a || !b) return;
    const runs = (S.runs || []).slice().reverse();
    const opts = runs.map((r) => `<option value="${r.run}">${esc(runLabel(r))}</option>`).join('');
    const keepA = a.value, keepB = b.value;
    a.innerHTML = opts; b.innerHTML = opts;
    const sr = (S.info && S.info.stage_runs) || {};
    const has = (v) => v && runs.some((r) => r.run === v);
    a.value = S.cmpUser && has(keepA) ? keepA : (has(sr['vitmatte@core']) ? sr['vitmatte@core'] : (runs[0] || {}).run || '');
    b.value = S.cmpUser && has(keepB) ? keepB : (has(sr['vitmatte@matanyone']) ? sr['vitmatte@matanyone'] : has(sr.matanyone) ? sr.matanyone : (runs[1] || runs[0] || {}).run || '');
    const off = runs.length < 2;
    ['st4CmpShowA', 'st4CmpShowB', 'st4CmpDiff'].forEach((id) => { $(id).disabled = off; });
    $('st4CmpInfo').textContent = off ? 'Potrzebne co najmniej dwie wersje (np. ViTMatte na masce 1+2 i na MatAnyone2 – zmień bazę i „Zastosuj na ujęcie”)'
      : S.cmpDiff ? `Różnica |A − B| (czerwone, ×3): ${S.cmpDiff.a} − ${S.cmpDiff.b}` : S.viewRun ? `Na podglądzie: ${S.viewRun}` : 'Na podglądzie: aktywna wersja';
  }
  function cmpShow(which) {
    const a = $('st4CmpA').value, b = $('st4CmpB').value;
    if (!a || !b) return;
    if (which === 'diff') { S.cmpDiff = { a, b }; showRun(a); }
    else { S.cmpDiff = null; showRun(which === 'a' ? a : b); }
    if (S.view !== 'alpha' && S.view !== 'comp') setView('alpha');
    renderCmp(); V().render();
  }
  // export source (2026-10-15): default = the highest current result of the selected object; "version" = the list of versions
  function renderExportSource() {
    const sel = $('e6Source'); if (!sel || !S.info) return;
    const avail = {
      core: stageState('core').state === 'done',
      matanyone: stageState('matanyone').state === 'done',
      videomama: stageState('videomama').state === 'done',
      vitmatte: stageState('vitmatte').state === 'done',
      ltx: stageState('ltx').state === 'done',
      version: !!(S.runs && S.runs.length) || !!(S.info.has && S.info.has.final),
      active: stageState('final').state === 'done',
      final_touch: stageState('final_touch').state === 'done',
    };
    // the shared mask chooser (same names / order as every stage); export needs CURRENT stage results (avail)
    const none = ` ${T('– brak wyniku')}`;
    const extra = `<option value="version"${avail.version ? '' : ' disabled'}>${esc(T(`Wersja z listy (etap ${ST.EXPERIMENTS})`))}${avail.version ? '' : esc(none)}</option>`
      + `<option value="active"${avail.active ? '' : ' disabled'}>${esc(T('Aktywna wersja'))}${avail.active ? '' : esc(none)}</option>`
      + `<option value="final_touch"${avail.final_touch ? '' : ' disabled'}>${esc(T(`Final Touch (etap ${ST.FINAL_TOUCH})`))}${avail.final_touch ? '' : esc(none)}</option>`;
    fillMaskSelect(sel, ['core', 'matanyone', 'videomama', 'vitmatte', 'ltx'], { avail: (k) => avail[k], extra });
    // DECYZJA (PLAN_FINAL_TOUCH.md): an object with hand corrections exports base + corrections by default
    const hasFt = !!(S.info.final_touch && (S.info.final_touch.frames || []).length);
    if (!S.exportSourceUser || !avail[sel.value]) sel.value = (hasFt && avail.final_touch ? 'final_touch' : null)
      || ['vitmatte', 'videomama', 'matanyone', 'active', 'core'].find((k) => avail[k]) || 'core';
    const runSel = $('e6Run');
    $('e6RunL').style.display = sel.value === 'version' ? '' : 'none';
    if (sel.value === 'version') {
      const cur = runSel.value;
      runSel.innerHTML = (S.runs || []).slice().reverse().map((r) => `<option value="${r.run}" title="${esc(r.edge_mode || '')} · QA ${r.gate_ok == null ? '–' : (r.gate_ok ? 'OK' : 'flagi')}">${r.run}${r.active ? ' ★' : ''} · ${esc(r.edge_mode || '')}${r.stage ? ` · ${esc(r.stage)}` : ''}</option>`).join('');
      if (cur && [...runSel.options].some((o) => o.value === cur)) runSel.value = cur;
      if (!S.runs) loadRuns().then(() => renderExportSource()).catch(() => {});
    }
    const runs = S.info.stage_runs || {};
    $('e6SrcInfo').textContent = { core: `Maska 1+2 ${S.info.has && S.info.has.core_fix ? (stageState('cotracker').state === 'done' ? '(aktualna)' : '(nieaktualna – policzy się przy eksporcie)') : '(policzy się przy eksporcie)'}`,
      matanyone: runs.matanyone ? `Wersja ${runs.matanyone}` : '', vitmatte: runs.vitmatte ? `Wersja ${runs.vitmatte}` : '',
      ltx: runs.ltx ? `Wersja ${runs.ltx} (LTX-2.5)` : '', videomama: runs.videomama ? `Wersja ${runs.videomama} (VideoMaMa)` : '',
      version: 'Przy eksporcie wszystkich obiektów: aktywne wersje', active: runs.active ? `Wersja ${runs.active}` : '',
      final_touch: S.info.final_touch ? `Baza + korekty w ${(S.info.final_touch.frames || []).length} klatkach (bazę wybierasz w etapie ${ST.FINAL_TOUCH})` : '' }[sel.value] || '';
  }
  const exportBody = (single) => {
    const src = $('e6Source') ? $('e6Source').value : 'active';
    return { source: src, ...(src === 'version' && single && $('e6Run').value ? { run: $('e6Run').value } : {}) };
  };
  async function loadExportDir() {
    try { const d = await api('GET', '/api/project/export_dir'); const el = $('e5Dir'); if (el) { el.value = d.path; el.title = d.is_default ? 'domyślny' : (d.outside_work ? 'poza projektem' : ''); } } catch (e) { /* no project */ }
    try { showExportFormat(await api('GET', '/api/project/export_format')); } catch (e) { /* no project */ }
  }
  function showExportFormat(d) {
    // export_format from project.json; the "auto" option shows what it resolves to for this source (8-bit -> PNG, EXR -> EXR)
    const sel = $('e5Format'); if (!sel) return;
    const name = (f) => (f === 'png' ? 'PNG 8-bit' : 'EXR');
    sel.options[0].textContent = `auto (→ ${name(d.auto_resolves_to)}, źródło ${d.bit_depth}-bit)`;
    sel.value = d.format;
    const multi = $('e5Multi'); if (multi) { multi.disabled = d.resolved !== 'exr'; multi.parentElement.title = d.resolved === 'exr' ? '' : 'multi-EXR tylko przy formacie EXR'; }
  }

  // ------------------------------------------------------------------------------------------------ actions
  // 409 of the work area (engine/workarea.py): missing frame intervals per object / no key inside the work area
  function rangeErrorDialog(e, retry) {
    const d = e && e.detail;
    if (!d || typeof d !== 'object' || !['range_missing', 'no_key_in_range'].includes(d.code)) return false;
    const iv = (l) => l.map(([a, b]) => (a === b ? `F${a}` : `F${a}–${b}`)).join(', ');
    let html;
    if (d.code === 'no_key_in_range' && d.suggest && window.MD_WA) {
      // the work area is one per shot (often set for another object): offer it extended to the object's nearest key
      const [a, b] = d.suggest;
      V().uiDialog({ title: 'Brak klucza w zakresie pracy', tone: 'warn', ok: `Rozszerz zakres do F${a}–${b}`, cancel: 'Zostaw',
        html: `${esc(d.detail)}<br><br>Zakres pracy jest wspólny dla wszystkich obiektów: <b>F${d.range[0]}–${d.range[1]}</b> (znaczniki [ ] na osi czasu, Alt+X = całe ujęcie).` })
        .then(async (yes) => { if (yes) { await window.MD_WA.set([a, b]); if (retry) retry(); } });
      return true;
    }
    if (d.code === 'no_key_in_range') {
      html = `${esc(d.detail)}<br><br>Zakres pracy: <b>F${d.range[0]}–${d.range[1]}</b> (znaczniki [ ] na osi czasu, Alt+X = całe ujęcie).`;
    } else {
      html = `<b>${esc(d.what)}</b> nie jest policzony na ${esc(d.scope)}. Brakujące klatki:<ul>` +
        Object.entries(d.missing || {}).map(([n, l]) => `<li><b>${esc(n)}</b>: ${esc(iv(l))}</li>`).join('') + '</ul>' +
        (d.what === 'Rdzeń' ? 'Policz rdzeń w tych klatkach: „▶▶ Przelicz całość” z zakresem pracy obejmującym je (Alt+X = całe ujęcie).'
                            : 'Przelicz finalnie w tych klatkach (zakres pracy obejmujący je, Alt+X = całe ujęcie) albo wyeksportuj tylko „work area”. Nic nie zostało zapisane.');
    }
    V().uiDialog({ title: d.code === 'no_key_in_range' ? 'Brak klucza w zakresie pracy' : 'Brakujące klatki', tone: 'warn', html });
    return true;
  }
  // stage LTX Alpha-Gen (2026-10-16): settings (object field `ltx`), models, the stage result
  async function loadLtx() {
    if (!S.obj || !$('lx6Guide')) return;
    const d = await api('GET', `/api/object/${S.obj}/ltx`);
    S.ltx = d;
    const s = d.settings;
    $('lx6Guide').value = s.guide; $('lx6Combine').value = s.combine; $('lx6Mask').value = s.mask_source || 'core';
    for (const [id, v] of [['lx6Region', s.region_px], ['lx6Soft', s.region_soft_px], ['lx6Erode', s.core_erode_px], ['lx6Feather', s.feather_px ?? 0]]) { $(id).value = v; $(`${id}S`).value = v; }
    S.ltxPv = null;
    document.querySelectorAll('.lx-band-only').forEach((e) => { e.style.display = s.combine === 'band' ? '' : 'none'; });
    if (s.combine === 'band') {
      await fillBandSources('lx');
      if (window.MD_BANDPTS && window.MD_BANDPTS.ready) { window.MD_BANDPTS.dirty(); window.MD_BANDPTS.onFrame(); }
    }
    for (const [id, v] of [['lx6Side', s.max_side], ['lx6Frames', s.max_frames]]) {
      const sel = $(id);
      if (![...sel.options].some((o) => +o.value === +v)) sel.appendChild(el(`<option value="${v}">${v}</option>`));
      sel.value = String(v);
    }
    const miss = d.missing_models || [];
    $('lx6Models').innerHTML = miss.length
      ? `<span class="err">Brak modeli LTX-2.5: ${miss.map(esc).join(', ')} (${d.missing_gb} GB)</span> – Doktor → „Zainstaluj LTX-2.5” (wymaga logowania do Hugging Face i akceptacji licencji LTX-2.x)`
      : (d.missing_nodes === null ? 'Modele LTX-2.5: OK · ComfyUI nie odpowiada (węzły nie sprawdzone)'
        : (d.missing_nodes.length ? `<span class="err">Comfy bez węzłów: ${d.missing_nodes.map(esc).join(', ')}</span>` : 'Modele LTX-2.5 i węzły Comfy: OK'));
    const st = stageState('ltx');
    $('lx6Info').textContent = d.run ? `Wynik etapu: ${d.run} ${d.valid ? '✔ aktualny' : `↻ nieaktualny – ${d.why || 'przelicz'}`}`
      : (st.state === 'locked' ? (st.why || '') : 'Brak wyniku – „▶ Policz LTX-2.5”');
  }
  async function ltxSetting(body) {
    if (!S.obj) return;
    await api('PUT', `/api/object/${S.obj}/ltx`, body);
    await loadStages(); await loadLtx();
  }

  async function run(btnId, fn) {
    const b = $(btnId); if (b) b.disabled = true;
    try { await fn(); } catch (e) { status(`Błąd: ${e.message}`, 'err'); if (!rangeErrorDialog(e, () => run(btnId, fn))) toast(`Błąd: ${e.message}`, 5000); }
    finally { if (b) b.disabled = false; loadStages().catch(() => {}); }
  }
  const curPts = () => { const pts = S.points[V().currentFrame] || []; return { positive: pts.filter((p) => p[2] > 0).map((p) => [p[0], p[1]]), negative: pts.filter((p) => p[2] < 0).map((p) => [p[0], p[1]]) }; };

  async function chooseCandidate(i) {   // Z10: candidate i -> automatic key in the detect frame
    if (!S.cands || !S.obj) return;
    const f = S.cands.f;
    await api('POST', `/api/object/${S.obj}/choose`, { frame: f, candidate: i });
    S.cands = null;
    invalidateFrame(f);
    status(`Klucz startowy (kandydat #${i}) zapisany w klatce ${f}.`, 'ok');
    await loadKeys(); await loadStages();
    toast(`Klucz z kandydata zapisany – „▶▶ Przelicz całość” policzy maskę w każdej klatce`, 5000);
  }
  function candAt(e) {   // index of the candidate box under the cursor (smallest box wins), -1 = none
    const v = V(); const img = v.getCurrentImage(); const ps = plateSize();
    if (!S.cands || !img || !ps || S.cands.f !== v.currentFrame) return -1;
    const r = v.canvas.getBoundingClientRect(); const k = img.width / ps[0];
    const x = (e.clientX - r.left - v.panX) / v.zoom / k, y = (e.clientY - r.top - v.panY) / v.zoom / k;
    let best = -1, area = Infinity;
    S.cands.boxes.forEach((b, i) => { if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) { const a = (b[2] - b[0]) * (b[3] - b[1]); if (a < area) { area = a; best = i; } } });
    return best;
  }
  function drawCandidates(ctx, img, k, f) {
    if (!S.cands || S.cands.f !== f || S.stage !== ST.SAM3) return;
    const v = V();
    const hov = S.cands.hover;
    if (hov != null) {
      const ov = loadOverlay(`${S.obj}_cand${hov}_${f}`, `/api/object/${S.obj}/frame/${f}?view=cand&cand=${hov}`);
      if (ov && ov !== 'missing') { const pa = ctx.globalAlpha; ctx.globalAlpha = 0.55; ctx.drawImage(tintCanvas(ov, [255, 200, 40], `${S.obj}_cand${hov}_${f}`), 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
    }
    ctx.lineWidth = 2 / v.zoom; ctx.font = `${13 / v.zoom}px "IBM Plex Sans", sans-serif`; ctx.textBaseline = 'top';
    S.cands.boxes.forEach((b, i) => {
      const hot = i === hov;
      ctx.strokeStyle = hot ? '#fde047' : 'rgba(200,200,204,0.85)'; ctx.setLineDash(hot ? [] : [6 / v.zoom, 4 / v.zoom]);
      ctx.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
      ctx.setLineDash([]);
      const label = `#${i}`; const tw = ctx.measureText(label).width + 8 / v.zoom;
      ctx.fillStyle = hot ? '#fde047' : 'rgba(200,200,204,0.85)'; ctx.fillRect(b[0] * k, b[1] * k - 17 / v.zoom, tw, 17 / v.zoom);
      ctx.fillStyle = '#2a2a2e'; ctx.fillText(label, b[0] * k + 4 / v.zoom, b[1] * k - 15 / v.zoom);
    });
  }
  function wire() {
    // object selector (stage bar)
    $('objSelect').addEventListener('change', (e) => selectObject(e.target.value));
    $('objNewBtn').addEventListener('click', () => newObjectDialog());
    $('st1Create').addEventListener('click', () => newObjectDialog());
    $('st1Delete').addEventListener('click', () => { if (S.obj) deleteObjectDialog(S.obj); });
    $('objDelBtn').addEventListener('click', () => { if (S.obj) deleteObjectDialog(S.obj); });
    // prompt field (palette): saved on blur WITHOUT reloading the object (that dropped unsaved points); typing marks the
    // frame „nieprzeliczona”; Enter = save + „▶ Przelicz klatkę” (bug 2026-10-15: Enter only saved, nothing recomputed)
    $('st1Prompt').addEventListener('change', async (e) => {
      if (!S.obj) return;
      const pr = e.target.value.trim();
      try {
        await api('PUT', `/api/object/${S.obj}/settings`, { prompts: pr ? [pr] : [] });
        const me = (S.objects || []).find((o) => o.name === S.obj); if (me) me.detect = { ...(me.detect || {}), prompts: pr ? [pr] : [] };
        status(`Prompt zapisany${pr ? ` („${pr}”)` : ' (pusty)'} – „▶ Przelicz klatkę” (Enter) liczy klatkę z nowym promptem.`, 'ok');
      } catch (er) { status(`Prompt NIE zapisany: ${er.message}`, 'err'); }
    });
    $('st1Prompt').addEventListener('input', () => { if (S.obj && V()) { S.pointsDirty[V().currentFrame] = true; updatePointsBox(); } });
    $('st1Prompt').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      e.preventDefault(); e.stopPropagation();
      if (S.obj && V()) computePoints(V().currentFrame);   // the request carries the field; the server saves it too
    });
    $('st1PartOf').addEventListener('change', (e) => run('st1PartOf', async () => {
      await api('PUT', `/api/object/${S.obj}/settings`, { part_of: e.target.value });
      status(e.target.value ? `„${S.obj}” jest częścią „${e.target.value}” – eksport: ${e.target.value}_full + osobne maski części` : `„${S.obj}” jest samodzielnym obiektem`, 'ok');
      await refreshList(S.obj);
    }));
    $('st1PtsClear').addEventListener('click', () => clearPoints(V().currentFrame));
    $('st1PtsRedo').addEventListener('click', () => computePoints(V().currentFrame));
    $('st1CropsBox').addEventListener('toggle', () => { S.showCrops = $('st1CropsBox').open; renderCrops(); V().render(); });
    $('st2Crops').addEventListener('click', () => run('st2Crops', async () => {
      const d = await api('POST', `/api/object/${S.obj}/crops`);
      S.crops = d.segments; renderCrops(); V().render();
      const fl = Object.entries(d.flags || {});
      status(`Odcinków: ${d.segments.length}` + (fl.length ? `\nFlagi: ${fl.map(([k, v]) => `${k}: ${v}`).join('\n')}` : ''), 'ok');
      await loadStages();
      S.showCrops = true; V().render();
    }));
    $('st1All').addEventListener('click', () => run('st1All', computeAll));
    $('st3Rerun').addEventListener('click', () => run('st3Rerun', async () => {
      await ensurePointsSaved();
      const f = V().currentFrame;
      const d = await api('POST', `/api/object/${S.obj}/rerun_from/${f}`);
      invalidateIntervals(d.recomputed); setView('core');   // Z20: only the recomputed intervals reload (old overlay shown meanwhile)
      status(`Przeliczono ${d.recomputed.length} przedz. (${d.seconds}s), z cache ${d.cached.length}.` + (d.suggested_keys.length ? `\nSugerowane klucze: ${d.suggested_keys.join(', ')}` : '') + frontMissingText(d), frontMissingKind(d));
    }));
    $('st3KeySave').addEventListener('click', () => run('st3KeySave', async () => {
      const f = V().currentFrame;
      if (!(S.strokes[f] || []).length && !(S.polys[f] || []).length && !(S.points[f] || []).length) { status('Najpierw narysuj poprawkę (G / R / P) albo postaw punkty.', 'warn'); return; }
      const source = (S.strokes[f] || []).length || (S.polys[f] || []).length ? 'core' : 'preview';
      const d = await api('POST', `/api/object/${S.obj}/keys/${f}`, { source, strokes: S.strokes[f] || [],
        polygons: (S.polys[f] || []).map((p) => ({ mode: p.mode, p: p.p })), ...curPts() });
      S.strokes[f] = []; S.polys[f] = []; clearOverlays();
      status(`Klucz zapisany w klatce ${f} (${d.key.kind}). „Przelicz zmienione przedziały”, żeby go użyć.`, 'ok');
      await loadKeys(); V().render();
    }));
    $('st3KeyDel').addEventListener('click', () => run('st3KeyDel', () => deleteKeyAt(V().currentFrame)));
    const bdk = $('btnDelKey'); if (bdk) bdk.addEventListener('click', () => run('btnDelKey', () => deleteStageKey(V().currentFrame)));   // Z6: playback bar, key of the stage
    // no interval-engine choice in stage 1 (user decision 2026-10-16): CoTracker3 is stage 2; the API (/interval_engine) stays for agents

    // stage 4
    const setting = (body) => api('PUT', `/api/object/${S.obj}/settings`, body).then(() => { status('Ustawienie zmienione – „Przelicz finalnie” zrobi nową wersję', 'ok'); loadStages(); }).catch((e) => status(`Błąd: ${e.message}`, 'err'));
    $('st4Edge').addEventListener('change', (e) => run('st4Final', async () => {
      await api('PUT', `/api/object/${S.obj}/edge_mode`, { mode: e.target.value }); await loadEdge();
      status('Tryb krawędzi zmieniony – „Przelicz finalnie” zrobi nową wersję (A/B)', 'ok');
    }));
    const epvLive = () => { clearTimeout(S.epvTimer); S.epvTimer = setTimeout(() => { S.epvParams = edgePvParams(); V().render(); }, 250); };
    const mbLive = () => { clearTimeout(S.mbTimer); S.mbTimer = setTimeout(() => { renderMbPalette(); V().render(); }, 250); };
    $('mbPv').addEventListener('change', () => {
      S.mbPv = $('mbPv').checked;
      if (S.mbPv && S.view !== 'alpha' && S.view !== 'comp') setView('alpha');
      if (S.mbPv && S.edgePv) { S.edgePv = false; renderEdgePalette(); }   // the band preview would cover it
      renderMbPalette(); V().render();
    });
    $('mbStr').addEventListener('input', () => { $('mbStrN').value = $('mbStr').value; mbLive(); });
    $('mbStrN').addEventListener('input', () => { $('mbStr').value = $('mbStrN').value; mbLive(); });
    $('mbShut').addEventListener('input', () => { $('mbShutN').value = $('mbShut').value; mbLive(); });
    $('mbShutN').addEventListener('input', () => { $('mbShut').value = $('mbShutN').value; mbLive(); });
    $('mbShutReset').addEventListener('click', () => { $('mbShutN').value = mbShutterManifest(); $('mbShut').value = mbShutterManifest(); mbLive(); });
    $('mbApply').addEventListener('click', () => run('mbApply', async () => {
      const base = mbBase(), s = mbStrength(), sh = mbShutter();
      status(`Motion blur alfy ×${s}, shutter ${sh}° na ${base} (wszystkie klatki → nowa wersja)…`);
      const d = await api('POST', `/api/object/${S.obj}/motion_blur`, { run: base, strength: s, shutter_deg: sh });
      S.mbPv = false;   // the new version has the blur: the preview would add it twice
      clearOverlays(); await refreshList(S.obj); await loadStages(); setView('alpha');
      const st = (d.motion_blur.stages || []).length ? ` – wynik etapu: ${d.motion_blur.stages.join(', ')}` : '';
      status(`Wersja ${d.run}: ${base} + motion blur ×${s}, shutter ${sh}° (najdłuższe rozmycie ${d.motion_blur.longest_px} px)${st}`, 'ok');
      loadRuns().catch(() => {}); renderStageInfo(); renderMbPalette();
    }));
    $('ePv').addEventListener('change', () => { S.edgePv = $('ePv').checked; if (S.edgePv && S.mbPv) { S.mbPv = false; renderMbPalette(); } S.epvParams = edgePvParams(); renderEdgePalette(); V().render(); });
    $('epStagePv').addEventListener('click', () => stagePreview('matanyone', 'epStagePv'));
    $('vmStagePv').addEventListener('click', () => stagePreview('videomama', 'vmStagePv'));
    $('epCompute').addEventListener('click', () => {
      const b = S.stage === ST.MATANYONE ? $('st3Ma') : $('st4Final');
      if (!b) return;
      if (b.disabled) { toast(stageState(S.stage === ST.MATANYONE ? 'matanyone' : 'final').why || 'Teraz niedostępne – zobacz panel etapu', 5000); return; }   // never a silent click
      b.click();
    });
    // slider = shortcut, number field = the value (any, typed for experiments – user request 2026-10-16)
    // px sliders: moving one switches a legacy object (×scale / shift) to the px mode – scale 1, no shift, both px saved
    const pxSave = () => setting({ band_out_px: Math.max(0, +$('o4OutV').value || 0), band_in_px: Math.max(0, +$('o4InV').value || 0),
                                   ...(S.edgeLegacy ? { band_scale: 1, shift_px: 0 } : {}) });
    // band preview while a slider is held (user request 2026-10-17): with the preview switched off, grabbing an edge
    // slider shows the band until it is released; then the preview is back to what was set (on / off)
    const pvHold = (key, ids) => ids.forEach((id) => {
      const e = $(id); if (!e) return;
      const on = () => { S.pvDrag = key; clearTimeout(S.pvDragT); V() && V().render(); };
      const off = () => { clearTimeout(S.pvDragT); S.pvDragT = setTimeout(() => { if (S.pvDrag === key && !S.pvHeld) { S.pvDrag = null; V() && V().render(); } }, 500); };
      e.addEventListener('pointerdown', () => { S.pvHeld = true; on(); });
      e.addEventListener('input', () => { on(); off(); });
      e.addEventListener('change', off);
    });
    window.addEventListener('pointerup', () => { if (S.pvHeld) { S.pvHeld = false; clearTimeout(S.pvDragT); S.pvDragT = setTimeout(() => { S.pvDrag = null; V() && V().render(); }, 250); } });
    pvHold('edge', ['o4Out', 'o4In', 'o4OutV', 'o4InV']);
    pvHold('vmm', ['vmReachS', 'vmReach', 'vmInsetS', 'vmInset', 'vmFeatherS', 'vmFeather', 'vmOuterFeatherS', 'vmOuterFeather']);
    pvHold('ltx', ['lx6RegionS', 'lx6Region', 'lx6SoftS', 'lx6Soft', 'lx6ErodeS', 'lx6Erode', 'lx6FeatherS', 'lx6Feather']);
    for (const id of ['o4Out', 'o4In']) {
      $(id).addEventListener('input', () => { $(`${id}V`).value = $(id).value; S.edgePxTouched = true; epvLive(); });
      $(id).addEventListener('change', pxSave);
      $(`${id}V`).addEventListener('input', () => { $(id).value = $(`${id}V`).value; S.edgePxTouched = true; epvLive(); });
      $(`${id}V`).addEventListener('change', () => { const v = $(`${id}V`).value; if (v !== '' && isFinite(+v) && +v >= 0) pxSave(); });
    }
    {   // outer edge feather (one band system, 2026-10-22): from the outer band edge into the core
      const ofSave = () => { const v = +$('o4OutFeatherV').value; if (isFinite(v) && v >= 0) setting({ band_out_feather_px: v }).then(() => { const o = S.objects.find((x) => x.name === S.obj); if (o) o.band_out_feather_px = v; V().render(); }); };
      $('o4OutFeather').addEventListener('input', () => { $('o4OutFeatherV').value = $('o4OutFeather').value; V().render(); });
      $('o4OutFeather').addEventListener('change', ofSave);
      $('o4OutFeatherV').addEventListener('input', () => { $('o4OutFeather').value = $('o4OutFeatherV').value; });
      $('o4OutFeatherV').addEventListener('change', ofSave);
      $('epBandClear').addEventListener('click', () => { if (window.MD_BANDPTS) window.MD_BANDPTS.deleteKeyAt(V().currentFrame); });
      $('epBandCopy').addEventListener('click', () => { const sel = $('epBandSrc'); if (sel.value && window.MD_BANDPTS) window.MD_BANDPTS.copyFrom(sel.value, sel.options[sel.selectedIndex].textContent); });
      // the whiskers of the band points scale with the px sliders while dragging (as stage 5)
      for (const id of ['o4Out', 'o4In', 'o4OutV', 'o4InV']) $(id).addEventListener('input', () => {
        const o = S.objects.find((x) => x.name === S.obj) || {};
        if (window.MD_BANDPTS) window.MD_BANDPTS.setLive(+o.band_out_px || 0, +o.band_in_px || 0, +$('o4OutV').value || 0, +$('o4InV').value || 0);
      });
    }
    {   // feather of the core inset (user request 2026-10-17): saved on release, used by „▶ Podgląd klatek” / „Run”
      const featherSave = () => { const v = +$('o4FeatherV').value; if (isFinite(v) && v >= 0) setting({ band_in_feather_px: v }); };
      $('o4Feather').addEventListener('input', () => { $('o4FeatherV').value = $('o4Feather').value; });
      $('o4Feather').addEventListener('change', featherSave);
      $('o4FeatherV').addEventListener('input', () => { $('o4Feather').value = $('o4FeatherV').value; });
      $('o4FeatherV').addEventListener('change', featherSave);
    }
    $('o4Smooth').addEventListener('change', () => setting({ temporal_smoothing: $('o4Smooth').value }));
    $('o4MaScale').addEventListener('change', () => setting({ ma_scale_mode: $('o4MaScale').value }));
    $('o4Backlit').addEventListener('change', () => setting({ use_backlit: $('o4Backlit').checked }));
    $('o4Lupa').addEventListener('change', () => {
      const v = $('o4Lupa').value; const f = V().currentFrame; const pg = (S.polys[f] || []).slice(-1)[0];
      if (v === 'box') {
        if (!pg) { status('Narysuj ramkę lupy wielokątem (P), potem wybierz „ramka”', 'warn'); $('o4Lupa').value = 'auto'; return; }
        const xs = pg.p.map((p) => p[0]), ys = pg.p.map((p) => p[1]);
        setting({ thin_lens: 'box', lupa_box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] });
      } else setting({ thin_lens: v });
    });
    $('st3Ma').addEventListener('click', () => run('st3Ma', async () => {
      status('MatAnyone2 na masce 1+2 (bez stref ViTMatte)…');
      await ensurePointsSaved();
      if (window.MD_BRUSH && window.MD_BRUSH.ready) await window.MD_BRUSH.flush();
      const d = await api('POST', `/api/object/${S.obj}/finalize`, { stage: 'matanyone' });
      S.edgePv = false;   // the result is there: the band preview would cover it (the checkbox brings it back)
      clearOverlays(); await loadStages(); setView('alpha');
      const grp = d.group ? ` · grupa liczona jako całość ze szwem: ${d.group.join(' + ')}${d.asked !== d.object ? ` (na obiekcie głównym „${d.object}”)` : ''}` : '';
      const out = d.group_outside_crop_px ? `
⚠ ${d.group_outside_crop_px} px całości grupy poza kadrami „${d.object}” (ucięte) – dodaj klucz obiektu głównego obejmujący część albo „Kadruj od nowa”` : '';
      status(`Wersja ${d.run} (MatAnyone2${d.whole_shot ? '' : `, zakres pracy F${d.frames[0]}–${d.frames[1]}`}): ma_scale ${d.report.ma_scale} (${d.report.ma_scale_mode || 'auto'}), ${d.report.seconds}s – baza etapu ${ST.VITMATTE} (ViTMatte)${grp}${out}`, out ? 'warn' : 'ok');
      loadEdge(); loadQA().catch(() => {}); renderStageInfo();
      toast(`Gotowe – wersja ${d.run} (MatAnyone2). Dalej: strefy ViTMatte (etap ${ST.VITMATTE}) albo eksport`, 6000);
    }));
    $('lx6Guide').addEventListener('change', (e) => ltxSetting({ guide: e.target.value }).catch((er) => status(`Błąd: ${er.message}`, 'err')));
    $('lx6Side').addEventListener('change', (e) => ltxSetting({ max_side: +e.target.value }).catch((er) => status(`Błąd: ${er.message}`, 'err')));
    $('lx6Frames').addEventListener('change', (e) => ltxSetting({ max_frames: +e.target.value }).catch((er) => status(`Błąd: ${er.message}`, 'err')));
    $('lx6Combine').addEventListener('change', (e) => { S.ltxPv = null; ltxSetting({ combine: e.target.value }).catch((er) => status(`Błąd: ${er.message}`, 'err')); });
    $('exBase').addEventListener('change', () => run('exBase', async () => {
      S.exRun = $('exBase').value || null; S.edgeFix = null;
      if (!S.exRun) { await loadQA(); renderExpPalette(); return; }
      showRun(S.exRun);
      await expAnalyze();   // measured once per version, then only the thresholds
    }));
    $('exAnalyze').addEventListener('click', () => run('exAnalyze', expAnalyze));
    $('exCompute').addEventListener('click', () => run('exCompute', () => expCompute($('exWithFix').checked)));
    $('exPvEdge').addEventListener('change', (e) => { expPvSet('edge', e.target.checked); V().render(); });
    $('exPvFlick').addEventListener('change', (e) => { S.flicker = e.target.checked; V().render(); });
    const exLive = () => {   // live count of the proposals from the saved measurements (nothing written)
      clearTimeout(S.exTimer);
      S.exTimer = setTimeout(async () => {
        if (!S.exRun) return;
        try {
          const d = await api('GET', `/api/object/${S.obj}/edge_recount?v=${encodeURIComponent(S.exRun)}&thr_px=${+$('exThr').value || 0.5}&min_frames=${Math.max(1, Math.round(+$('exMin').value || 5))}`);
          $('exCount').textContent = `z tymi progami: ${d.n_fix} odcinków do poprawy (mediana krawędzi ${d.median_abs} px) – puść suwak, żeby zapisać`;
        } catch (er) { $('exCount').textContent = er.message; }
      }, 200);
    };
    for (const [id, k] of [['exThr', 'exThr'], ['exMin', 'exMin']]) {
      $(`${id}S`).addEventListener('input', () => { $(id).value = $(`${id}S`).value; exLive(); });
      $(id).addEventListener('input', () => { $(`${id}S`).value = $(id).value; exLive(); });
      const save = () => { S[k] = +$(id).value; if (S.exRun) run('exAnalyze', expAnalyze); };
      $(`${id}S`).addEventListener('change', save); $(id).addEventListener('change', save);
    }
    const ltxLive = () => { clearTimeout(S.ltxTimer); S.ltxTimer = setTimeout(() => { S.ltxPv = ltxPvParams(); V().render(); }, 200); };
    const ltxSave = (body) => ltxSetting(body).catch((er) => status(`Błąd: ${er.message}`, 'err'));
    $('lxBandClear').addEventListener('click', () => { if (window.MD_BANDPTS) window.MD_BANDPTS.deleteKeyAt(V().currentFrame); });
    $('lxBandCopy').addEventListener('click', () => { const sel = $('lxBandSrc'); if (sel.value && window.MD_BANDPTS) window.MD_BANDPTS.copyFrom(sel.value, sel.options[sel.selectedIndex].textContent); });
    const ltxBandLive = () => { const s0 = (S.ltx && S.ltx.settings) || {}; if (window.MD_BANDPTS && s0.combine === 'band') window.MD_BANDPTS.setLive(+s0.region_px || 0, +s0.core_erode_px || 0, +$('lx6Region').value || 0, +$('lx6Erode').value || 0); };
    for (const [id, k, int] of [['lx6Region', 'region_px', false], ['lx6Soft', 'region_soft_px', false], ['lx6Erode', 'core_erode_px', true], ['lx6Feather', 'feather_px', false]]) {
      $(`${id}S`).addEventListener('input', () => { $(id).value = $(`${id}S`).value; ltxBandLive(); ltxLive(); });
      $(`${id}S`).addEventListener('change', () => ltxSave({ [k]: int ? Math.round(+$(id).value) : +$(id).value }));
      $(id).addEventListener('input', () => { $(`${id}S`).value = $(id).value; ltxLive(); });
      $(id).addEventListener('change', () => { const v = +$(id).value; if (isFinite(v) && v >= 0) ltxSave({ [k]: int ? Math.round(v) : v }); });
    }
    $('lx6Mask').addEventListener('change', (e) => { ltxLive(); ltxSave({ mask_source: e.target.value }); });
    $('lxPvBand').addEventListener('change', (e) => { ltxPvSet('band', e.target.checked); V().render(); });
    $('lxPvAlpha').addEventListener('change', (e) => { ltxPvSet('alpha', e.target.checked); V().render(); });
    $('lx6Run').addEventListener('click', () => run('lx6Run', async () => {
      status('LTX-2.5 Alpha-Gen… (pierwsza paczka ładuje model 22B – kilka minut)');
      await ensurePointsSaved();
      if (window.MD_BRUSH && window.MD_BRUSH.ready) await window.MD_BRUSH.flush();
      const d = await api('POST', `/api/object/${S.obj}/ltx`);
      clearOverlays(); await loadStages(); await loadRuns().catch(() => {});
      showRun(d.run);
      const grp = d.group ? ` · grupa liczona jako całość: ${d.group.join(' + ')}` : '';
      status(`Wersja ${d.run} (LTX-2.5, F${d.frames[0]}–${d.frames[1]}): ${(d.batches || []).length} paczek, ${d.seconds}s${grp}`, 'ok');
      await loadLtx();
      toast(`Gotowe – wersja ${d.run} (LTX-2.5). Porównaj z MatAnyone2, eksport: źródło „LTX-2.5”`, 6000);
    }));
    $('lx6Show').addEventListener('click', () => { const r = S.info && S.info.stage_runs && S.info.stage_runs.ltx; if (r) showRun(r); else status('Brak wyniku LTX-2.5', 'warn'); });
    $('lx6ShowMa').addEventListener('click', () => { const r = S.info && S.info.stage_runs && S.info.stage_runs.matanyone; if (r) showRun(r); else status(`Brak wyniku MatAnyone2 (etap ${ST.MATANYONE})`, 'warn'); });
    $('lx6ShowOff').addEventListener('click', () => showRun(null));
    // stage VideoMaMa (2026-10-16)
    const vmmLive = () => { clearTimeout(S.vmmTimer); S.vmmTimer = setTimeout(() => { S.vmmPv = vmmPvParams(); V().render(); }, 200); };
    const vmmSave = (body) => vmmSetting(body).catch((er) => status(`Błąd: ${er.message}`, 'err'));
    $('vmBase').addEventListener('change', (e) => { vmmLive(); vmmSave({ base: e.target.value }); });
    // saving one of them saves the three together (inset_px present = the unified composition)
    const vmmEdgeSave = () => vmmSave({ reach_px: +$('vmReach').value || 0, inset_px: +$('vmInset').value || 0, feather_px: +$('vmFeather').value || 0,
                                        outer_feather_px: +$('vmOuterFeather').value || 0 });
    // the whiskers of the band points scale with „Pas na zewnątrz” / „Wcięcie w rdzeń” while dragging (as stage 5)
    const vmmBandLive = () => { const s0 = (S.vmm && S.vmm.settings) || {}; if (window.MD_BANDPTS) window.MD_BANDPTS.setLive(+s0.reach_px || 0, +s0.inset_px || 0, +$('vmReach').value || 0, +$('vmInset').value || 0); };
    for (const id of ['vmReach', 'vmInset', 'vmFeather', 'vmOuterFeather']) {
      $(`${id}S`).addEventListener('input', () => { $(id).value = $(`${id}S`).value; vmmBandLive(); vmmLive(); });
      $(`${id}S`).addEventListener('change', vmmEdgeSave);
      $(id).addEventListener('input', () => { $(`${id}S`).value = $(id).value; vmmBandLive(); vmmLive(); });
      $(id).addEventListener('change', () => { const v = +$(id).value; if (isFinite(v) && v >= 0) vmmEdgeSave(); });
    }
    $('vmBandClear').addEventListener('click', () => { if (window.MD_BANDPTS) window.MD_BANDPTS.deleteKeyAt(V().currentFrame); });
    $('vmBandCopy').addEventListener('click', () => { const sel = $('vmBandSrc'); if (sel.value && window.MD_BANDPTS) window.MD_BANDPTS.copyFrom(sel.value, sel.options[sel.selectedIndex].textContent); });
    $('vmSide').addEventListener('change', (e) => vmmSave({ max_side: +e.target.value }));
    $('vmBatch').addEventListener('change', (e) => vmmSave({ batch: +e.target.value }));
    $('vmOverlap').addEventListener('change', (e) => vmmSave({ overlap: +e.target.value }));
    $('vmPvBand').addEventListener('change', (e) => { vmmPvSet('band', e.target.checked); V().render(); });
    $('vmPvAlpha').addEventListener('change', (e) => { vmmPvSet('alpha', e.target.checked); V().render(); });
    $('vmRun').addEventListener('click', () => run('vmRun', async () => {
      status('VideoMaMa… (pierwsza paczka ładuje model – ok. 10–30 s)');
      await ensurePointsSaved();
      if (window.MD_BRUSH && window.MD_BRUSH.ready) await window.MD_BRUSH.flush();
      const d = await api('POST', `/api/object/${S.obj}/videomama`);
      clearOverlays(); await loadStages(); await loadRuns().catch(() => {});
      showRun(d.run);
      const grp = d.group ? ` · grupa liczona jako całość: ${d.group.join(' + ')}` : '';
      status(`Wersja ${d.run} (VideoMaMa, F${d.frames[0]}–${d.frames[1]}): ${(d.batches || []).length} paczek, ${d.seconds}s${grp}`, 'ok');
      await loadVmm(); renderVmmPalette();
      toast(`Gotowe – wersja ${d.run} (VideoMaMa). Porównaj z MatAnyone2; dalej: ViTMatte (baza „VideoMaMa”) albo eksport`, 6000);
    }));
    $('vmShow').addEventListener('click', () => { const r = S.info && S.info.stage_runs && S.info.stage_runs.videomama; if (r) showRun(r); else status('Brak wyniku VideoMaMa', 'warn'); });
    $('vmShowMa').addEventListener('click', () => { const r = S.info && S.info.stage_runs && S.info.stage_runs.matanyone; if (r) showRun(r); else status(`Brak wyniku MatAnyone2 (etap ${ST.MATANYONE})`, 'warn'); });
    $('vmShowOff').addEventListener('click', () => showRun(null));
    $('st4Final').addEventListener('click', () => run('st4Final', async () => {
      status('Przeliczam finalnie (krawędź + QA)…');
      await ensurePointsSaved();
      const d = await api('POST', `/api/object/${S.obj}/finalize`, {});
      clearOverlays(); await loadStages(); setView('alpha');
      status(`Wersja ${d.run}${d.whole_shot ? '' : ` (zakres pracy F${d.frames[0]}–${d.frames[1]})`}: tryb ${d.report.edge_mode}, ma_scale ${d.report.ma_scale} (${d.report.ma_scale_mode || 'auto'}), ${d.report.seconds}s`, 'ok');
      loadEdge(); await loadQA();
      const bad = S.qa && S.qa.colours ? Object.values(S.qa.colours).filter((c) => c !== 'green').length : 0;   // Z6 toast
      toast(`Gotowe – wersja ${d.run}. Obejrzyj QA${bad ? `: ${bad} kl. z flagami, N = następna flaga` : ' (zielone)'}`, 6000);
    }));
    const zoneBody = (f) => {
      const poly = (S.polys[f] || []).slice(-1)[0];
      return { mode: $('z4Mode').value, criterion: $('z4Crit').value, frame: f, polygon: poly ? poly.p : [],
        strokes: (S.strokes[f] || []).map((s) => ({ mode: 'add', radius_px: s.radius_px, p: s.p })),
        param: +$('z4Param').value, soft: $('z4Soft').checked, softness: +$('z4Softness').value, static: $('z4Static').checked,
        color_lab: $('z4Crit').value === 'color' && S.zoneColor ? S.zoneColor.lab : null };
    };
    $('z4Param').addEventListener('input', () => { $('z4ParamV').textContent = $('z4Param').value; });
    $('z4Softness').addEventListener('input', () => { $('z4SoftV').textContent = $('z4Softness').value; });
    $('z4Soft').addEventListener('change', () => { $('z4SoftL').style.display = $('z4Soft').checked ? '' : 'none'; });
    $('z4Crit').addEventListener('change', () => {
      const c = $('z4Crit').value; $('z4ParamL').style.display = c === 'all' ? 'none' : '';
      if (c === 'color') { $('z4Param').max = 60; $('z4Param').step = 0.5; $('z4Param').value = 10; } else { $('z4Param').max = 1; $('z4Param').step = 0.01; $('z4Param').value = c === 'bright' ? 0.85 : 0.15; }
      $('z4ParamV').textContent = $('z4Param').value;
      $('z4ColorRow').style.display = c === 'color' ? '' : 'none';
    });
    $('z4Pick').addEventListener('click', () => { S.pipette = !S.pipette; $('z4Pick').classList.toggle('active', S.pipette); status(S.pipette ? 'Pipeta: kliknij na obrazie kolor strefy' : 'Pipeta wyłączona'); });
    $('z4Preview').addEventListener('click', () => run('z4Preview', async () => {
      const f = V().currentFrame; const b = zoneBody(f);
      if (!b.polygon.length && !b.strokes.length) { status('Najpierw narysuj obszar strefy (P / pędzel)', 'warn'); return; }
      const d = await api('POST', `/api/object/${S.obj}/zones/preview_tmp?preview_frame=${f}`, b);
      await api('DELETE', `/api/object/${S.obj}/zones/preview_tmp`);
      // show it: green = added by the zone, red = removed (until the next preview / apply / object change)
      delete S.overlay[`${S.obj}_zone_add_${f}`]; delete S.overlay[`${S.obj}_zone_remove_${f}`];
      S.zonePreview = d.preview ? { f } : null; V().render();
      status(d.preview ? `Strefa w klatce ${f}: +${d.preview.added_px} px (zielone) / −${d.preview.removed_px} px (czerwone) – podgląd na obrazie` : 'Brak rdzenia w tej klatce – policz ujęcie', d.preview ? 'ok' : 'warn');
    }));
    $('z4Apply').addEventListener('click', () => run('z4Apply', async () => {
      const f = V().currentFrame; const b = zoneBody(f);
      if (!b.polygon.length && !b.strokes.length) { status('Najpierw narysuj obszar strefy (P / pędzel)', 'warn'); return; }
      const d = await api('POST', `/api/object/${S.obj}/zones/new`, b);
      S.polys[f] = []; S.strokes[f] = []; S.zonePreview = null; V().render(); loadZones();
      status(`Strefa ${d.zone.zone_id} zapisana – działa przy „Przelicz finalnie”`, 'ok');
    }));
    $('st4NextFlag').addEventListener('click', nextFlag);

    // stage 5
    const srcLabel = () => ($('e6Source') ? $('e6Source').selectedOptions[0].textContent.replace(/ \((brak|missing)\)$/, '') : '');
    const exportSel = (names, btn) => run(btn, async () => {
      const d = await api('POST', '/api/export', { objects: names, with_core: $('e5Core').checked, scope: $('e5Scope').value, ...exportBody(true) });
      status(`Wyeksportowano: ${names.join(', ')} (F${d.frames[0]}–${d.frames[1]}, źródło: ${srcLabel()}${d.runs ? ` – ${Object.entries(d.runs).map(([n, r]) => `${n} ${r}`).join(', ')}` : ''})${d.warning ? '\n⚠ ' + d.warning : ''}`, 'ok');
    });
    $('st5Export').addEventListener('click', () => run('st5Export', async () => {
      const d = await api('POST', '/api/export', { multichannel: $('e5Multi').checked, with_core: $('e5Core').checked, with_depth: $('e5Depth').checked, scope: $('e5Scope').value, ...exportBody(false) });
      status(`Eksport F${d.frames[0]}–${d.frames[1]} (źródło: ${srcLabel()}): ${d.outputs.objects.map((o) => `${o.depth_order}. ${o.name} ${o.version} (QA ${o.qa_status})`).join(', ')}` +
        (d.warning ? `\n⚠ ${d.warning}` : '') +
        (d.contract_errors.length ? `\nKontrakt danych: ${d.contract_errors.join('; ')}` : '\noutputs.json zgodny z kontraktem'), d.contract_errors.length || d.warning ? 'warn' : 'ok');
    }));
    $('e6Source').addEventListener('change', () => { S.exportSourceUser = true; renderExportSource(); });
    $('st4BaseSel').addEventListener('change', () => run('st4BaseSel', async () => {
      const v = $('st4BaseSel').value;
      await api('PUT', `/api/object/${S.obj}/settings`, { vitmatte_base: v });
      await loadStages();
      // the frame preview and the cut-out follow the new base at once (user report 2026-10-17)
      if (window.MD_ZONES && window.MD_ZONES.refreshPreview) window.MD_ZONES.refreshPreview(true);
      V().render();
      status(`Baza ViTMatte: ${v === 'core' ? 'maska 1+2 (etapy ' + ST.SAM3 + ' + ' + ST.BRUSH + ')' : 'MatAnyone2 (etap ' + ST.MATANYONE + ')'} – „Zastosuj na ujęcie” zrobi wynik na tej bazie (drugi wynik zostaje do porównania)`, 'ok');
    }));
    $('st4CmpShowA').addEventListener('click', () => cmpShow('a'));
    $('st4CmpShowB').addEventListener('click', () => cmpShow('b'));
    $('st4CmpDiff').addEventListener('click', () => cmpShow('diff'));
    $('st4CmpOff').addEventListener('click', () => { S.cmpDiff = null; showRun(null); renderCmp(); V().render(); });
    ['st4CmpA', 'st4CmpB'].forEach((id) => $(id).addEventListener('change', () => { S.cmpUser = true; if (S.cmpDiff) cmpShow('diff'); else renderCmp(); }));
    $('e5DirPick').addEventListener('click', () => run('e5DirPick', async () => {
      const cur = await api('GET', '/api/project/export_dir');
      const r = await api('POST', '/api/dialog/pick_folder', { title: 'Katalog eksportu', initial_dir: cur.path });
      if (!r.folder) { status('Nie wybrano katalogu (albo okno wyboru niedostępne)', 'warn'); return; }
      let d;
      try { d = await api('PUT', '/api/project/export_dir', { path: r.folder }); }
      catch (e) {
        if (!/poza projektem/.test(e.message) || !(await V().uiConfirm(`${e.message}\n\nZapisać eksport (EXR) w tym katalogu?`, { title: 'Katalog eksportu poza projektem', ok: 'Zapisz tutaj' }))) throw e;
        d = await api('PUT', '/api/project/export_dir', { path: r.folder, confirmed: true });
      }
      await loadExportDir(); status(`Katalog eksportu: ${d.path}`, 'ok');
    }));
    $('e5Format').addEventListener('change', () => run('e5Format', async () => {
      const d = await api('PUT', '/api/project/export_format', { format: $('e5Format').value });
      showExportFormat(d); status(`Format masek: ${d.format}${d.format === 'auto' ? ` (→ ${d.resolved})` : ''}`, 'ok');
    }));
    $('e5DirDefault').addEventListener('click', () => run('e5DirDefault', async () => { const d = await api('PUT', '/api/project/export_dir', { path: null }); await loadExportDir(); status(`Katalog eksportu: ${d.path} (domyślny)`, 'ok'); }));
    $('e5NkOnly').addEventListener('click', () => run('e5NkOnly', async () => { const d = await api('POST', '/api/export/nk'); status(`Skrypt Nuke'a: ${d.nk}`, 'ok'); }));
    $('st5ExportOne').addEventListener('click', () => S.obj ? exportSel([S.obj], 'st5ExportOne') : status('Wybierz obiekt', 'warn'));
    $('st5ExportWhole').addEventListener('click', () => { const wh = wholeOf(S.obj); if (wh) exportSel([`${wh}_full`], 'st5ExportWhole'); });
    $('st5LibSave').addEventListener('click', () => run('st5LibSave', async () => { const d = await api('POST', '/api/library', { object: S.obj }); status(`Zapisano w bibliotece: ${d.path}`, 'ok'); }));
    $('st5LibAdd').addEventListener('click', () => run('st5LibAdd', async () => {
      const d = await api('GET', '/api/library');
      if (!d.items.length) { status('Biblioteka pusta', 'warn'); return; }
      const pick = prompt('Dodaj z biblioteki (wpisz numer):\n' + d.items.map((x, i) => `${i}: ${x.id} – ${x.prompts.join(', ')}`).join('\n'), '0');
      if (pick === null) return;
      const it = d.items[+pick]; const r = await api('POST', '/api/library/add', { item: it.id });
      await refreshList(r.object.name); status(`Dodano ${r.object.name} z biblioteki (klucze i strefy nie są przenoszone)`, 'ok');
    }));
    $('st5QAdd').addEventListener('click', () => run('st5QAdd', async () => { const d = await api('POST', '/api/queue', { object: S.obj }); status(`Kolejka: ${d.report.pending} zadań, ETA ~${Math.round(d.report.eta_s / 60)} min`, 'ok'); }));
    $('st5QRun').addEventListener('click', () => run('st5QRun', async () => {
      await api('POST', '/api/queue/run');
      const poll = async () => { const r = await api('GET', '/api/queue');
        status('Kolejka:\n' + r.jobs.map((j) => `${j.id} ${j.object}: ${j.status}${j.qa ? ' QA ' + j.qa : ''}${j.first_flag ? ' (1. flaga ' + j.first_flag + ')' : ''}${j.error ? ' – ' + j.error : ''}`).join('\n'));
        if (r.running) setTimeout(poll, 5000); };
      poll();
    }));
    $('st5RevFile').addEventListener('change', (e) => run('st5Export', async () => {
      const file = e.target.files[0]; if (!file) return;
      const data = JSON.parse(await file.text());
      const d = await api('POST', `/api/object/${S.obj}/import_review`, data);
      status(`Zaimportowano kreski jako klucze: ${d.keys.join(', ') || 'brak'} – etap ${ST.SAM3} „Przelicz zmienione przedziały”`, 'ok'); loadKeys();
    }));

    // "Dalej" buttons
    for (const st of STAGES) { const b = $(`st${st.n}Next`); if (b) b.addEventListener('click', () => setStage(st.n + 1)); }

    // "Przesuń" (Widok) = no drawing / correction tool, only panning
    if ($('toolSelectBtn')) $('toolSelectBtn').addEventListener('click', () => { if (S.tool !== 'none') { S.tool = 'none'; updateDrawTools(); } });
    if ($('hudMaskAlt')) $('hudMaskAlt').addEventListener('click', (e) => {   // stage 5: the other engine's mask (2026-10-22)
      const eng = e.currentTarget.dataset.eng; if (!eng) return;
      S.vmPick = eng === ((S.info && S.info.edge_engine) || 'vitmatte') ? null : eng;
      updateMaskHud(V().currentFrame); renderMbPalette(); V().render();
    });

    // view tabs
    document.querySelectorAll('#viewModeTabs .tab-btn').forEach((b) => b.addEventListener('click', () => {
      if (b.disabled) { toast(b.title); return; }
      setView(b.dataset.view);
    }));

    // keys navigation
    $('btnPrevKey').addEventListener('click', () => keyStep(-1));
    $('btnNextKey').addEventListener('click', () => keyStep(1));
    window.addEventListener('md-key-step', (e) => keyStep(e.detail));

    // canvas interactions -------------------------------------------------------------------
    const cv = V().canvas;
    cv.addEventListener('contextmenu', (e) => { if (S.obj && !S.auxMode && (S.stage === ST.SAM3 || S.tool !== 'none')) e.preventDefault(); });
    // Alt held = temporary hand (user request 2026-10-15): every tool lets the drag through to panning (app.js),
    // releasing Alt brings the chosen tool back. The "opposite sign" of points / corrections / polygon = PPM.
    const setAltPan = (on) => {
      if (!!S.altPan === on) return;
      S.altPan = on; cv.classList.toggle('alt-pan', on);
      if (on && S.hover) S.hover = null;
      V().render();
    };
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'Alt') return;
      e.preventDefault();   // Alt alone would focus the browser menu
      setAltPan(true);
    });
    window.addEventListener('keyup', (e) => { if (e.key === 'Alt') { e.preventDefault(); setAltPan(false); } });
    window.addEventListener('blur', () => setAltPan(false));
    cv.addEventListener('mousedown', (e) => {
      if (!S.obj || S.auxMode) return;
      if (e.altKey && e.button === 0) return;   // Alt = hand: no tool, app.js pans
      if (S.pipette && S.stage === ST.EXPERIMENTS && e.button === 0) {   // Z24: colour of the "color" zone criterion
        e.mdHandled = true; e.preventDefault(); e.stopPropagation();
        const p = canvasToPlate(e); if (!p) return;
        S.pipette = false; $('z4Pick') && $('z4Pick').classList.remove('active');
        api('GET', `/api/object/${S.obj}/color_lab/${V().currentFrame}?x=${p[0].toFixed(1)}&y=${p[1].toFixed(1)}`).then((d) => {
          S.zoneColor = d; $('z4Swatch').style.background = d.hex;
          $('z4ColorTxt').textContent = `Lab ${d.lab.map((v) => v.toFixed(1)).join(' / ')}`;
          status(`Kolor strefy: ${d.hex} (Lab ${d.lab.join(', ')})`, 'ok');
        }).catch((er) => status(`Pipeta: ${er.message}`, 'err'));
        return;
      }
      // stage Eksperymenty without a drawing tool: a click on an edge-correction piece = (un)check it; a drag still pans
      if (S.stage === ST.EXPERIMENTS && e.button === 0 && !['poly', 'brush_add', 'brush_remove'].includes(S.tool) && edgePieceAt(e)) {
        S.edgeClick = { x: e.clientX, y: e.clientY };
      }
      // Z10: stage SAM3 with candidates on the canvas – a click inside a frame chooses it (decided on mouseup, so a drag still pans)
      if (S.stage === ST.SAM3 && e.button === 0 && candAt(e) >= 0) { S.candClick = { x: e.clientX, y: e.clientY, i: candAt(e) }; return; }
      // stage SAM3: a click = point (SAM3 on „▶ Przelicz klatkę”, user request 2026-10-15) or, with Korekta + / −, a correction
      if (clickTool()) { S.clickStart = { x: e.clientX, y: e.clientY, b: e.button, alt: e.altKey }; return; }   // no click tool = only panning   // pan still allowed; click decided on mouseup
      if ([ST.SAM3, ST.VITMATTE, ST.EXPERIMENTS].includes(S.stage) && S.tool === 'poly' && (e.button === 0 || e.button === 2)) {   // brush: pointer events below (Z23)
        e.mdHandled = true; e.preventDefault();
        const f = V().currentFrame; const p = clampPlate(canvasToPlate(e)); if (!p) return;
        if (S.tool === 'poly') {
          if (!S.poly || S.poly.f !== f) S.poly = { f, mode: e.button === 2 ? 'remove' : 'add', p: [] };   // PPM on the 1st point = removing polygon
          S.poly.p.push(p); V().render(); return;
        }
        S.drawing = { f, mode: S.tool === 'brush_add' ? 'add' : 'remove', radius_px: S.radius, p: [p] };
        V().render();
      }
    }, true);
    // ---- Z23: brush on Pointer Events (pen pressure, eraser end, Shift + drag = size); mouse = constant radius as before
    cv.style.touchAction = 'none';
    const brushOn = () => S.obj && !S.auxMode && drawStage() && ['brush_add', 'brush_remove', 'brush_erase'].includes(S.tool);
    const platePt = (e) => {   // like canvasToPlate, without the "still loading" toast (hover)
      const v = V(); const img = v.getCurrentImage && v.getCurrentImage(); const ps = plateSize(); if (!img || !ps) return null;
      const r = v.canvas.getBoundingClientRect();
      return [((e.clientX - r.left - v.panX) / v.zoom) * ps[0] / img.width, ((e.clientY - r.top - v.panY) / v.zoom) * ps[1] / img.height];
    };
    cv.addEventListener('mousemove', (e) => {   // hover over an edge piece: highlight + pointer cursor
      if (!S.edgeFix || S.stage !== ST.EXPERIMENTS || e.buttons) return;
      const pc = ['poly', 'brush_add', 'brush_remove'].includes(S.tool) ? null : edgePieceAt(e);
      const id = pc ? pc.id : null;
      cv.classList.toggle('edge-hot', !!pc);
      cv.title = pc ? `${pc.id} · ${pc.kind === 'halo' ? 'halo → zwęź' : 'ucięcie → poszerz'} · ${pc.median_px > 0 ? '+' : ''}${pc.median_px} px · klik = ${S.edgeFix.skip.has(pc.id) ? 'zaznacz' : 'odznacz'}` : '';
      if (S.edgeFix.hover !== id) { S.edgeFix.hover = id; V().render(); }
    });
    let hoverRaf = 0;
    // pen detection in every stage and tool, also on hover (was: only on a brush pointerdown in stages 3–4, so the
    // "Nacisk piórka" checkbox never showed before the first stroke) – capture, before any handler stops the event
    const notePen = (e) => {
      if (e.pointerType === 'pen') {
        if (!S.penSeen) { S.penSeen = true; S.penPressure = true; updateDrawTools(); }
      } else if (!S.penSeen && e.pointerType === 'mouse' && e.type === 'pointerdown' && e.pressure > 0 && e.pressure !== 0.5 && !S.penAsMouse) {
        S.penAsMouse = true; updateDrawTools();   // a pressure value on a "mouse" = a tablet driver posing as a mouse
      }
    };
    cv.addEventListener('pointerdown', notePen, true);
    cv.addEventListener('pointermove', notePen, true);
    cv.addEventListener('pointerdown', (e) => {
      if (!brushOn() || e.altKey) return;   // Alt = hand
      // the pen's eraser end: button 5 on the way down, or the eraser bit WITHOUT the tip bit (some drivers raise the
      // eraser bit together with the tip – that is the tip, user report 2026-10-22: „−” painted white)
      // (a BOOLEAN: `buttons & 32` is a number, and `false !== 0` is true – with the pen every „−” stroke became „+”,
      // user report 2026-10-22)
      const eraser = e.pointerType === 'pen' && (e.button === 5 || (!!(e.buttons & 32) && !(e.buttons & 1)));
      if (e.button !== 0 && !eraser) return;
      e.preventDefault(); e.stopPropagation(); e.mdHandled = true;
      try { cv.setPointerCapture(e.pointerId); } catch (er) { /* not capturable */ }
      if (e.pointerType === 'pen' && !S.penSeen) { S.penSeen = true; S.penPressure = true; updateDrawTools(); }
      const p = clampPlate(canvasToPlate(e)); if (!p) return;
      if (e.shiftKey) { S.resizing = { x0: e.clientX, r0: S.radius, p, pid: e.pointerId }; V().render(); return; }
      const pen = e.pointerType === 'pen' && S.penPressure;
      // the pen's eraser end: in the correction stages (2 / 8) it is the eraser (wipes + and −), never the opposite brush
      // (user report 2026-10-22: the „−” pen painted white); the SAM3 key brush has no eraser – there it flips + / −
      const hasEraser = S.stage === ST.BRUSH || S.stage === ST.FINAL_TOUCH;
      const mode = S.tool === 'brush_erase' || (eraser && hasEraser) ? 'erase'
        : ((S.tool === 'brush_add') !== (eraser && !hasEraser) ? 'add' : 'remove');
      S.drawing = { f: V().currentFrame, mode, radius_px: S.radius,
                    p: [pen ? [p[0], p[1], penRadius(e)] : p], pid: e.pointerId, pen };
      if (eraser) toast(`Piórko: końcówka-gumka – ${mode === 'erase' ? 'gumka' : mode === 'add' ? 'Dodaj' : 'Usuń'}`, 1500);
      V().render();
    }, true);
    cv.addEventListener('pointermove', (e) => {
      if (S.resizing && e.pointerId === S.resizing.pid) { setRadius(S.resizing.r0 * Math.exp((e.clientX - S.resizing.x0) / 120)); return; }
      if (S.drawing && e.pointerId === S.drawing.pid) {
        const evs = (e.getCoalescedEvents && e.getCoalescedEvents().length) ? e.getCoalescedEvents() : [e];
        for (const ev of evs) {
          const p = clampPlate(platePt(ev)); if (!p) continue;
          S.drawing.p.push(S.drawing.pen ? [p[0], p[1], penRadius(ev)] : p);
        }
        V().render(); return;
      }
      if (brushOn() && !S.altPan) { S.hover = platePt(e); if (!hoverRaf) hoverRaf = requestAnimationFrame(() => { hoverRaf = 0; V().render(); }); }
      if (S.cands && S.stage === ST.SAM3 && S.cands.f === V().currentFrame) {   // Z10: hover = highlighted candidate + its mask
        const i = candAt(e); const ci = i >= 0 ? i : null;
        if (ci !== S.cands.hover) { S.cands.hover = ci; cv.style.cursor = ci != null ? 'pointer' : ''; document.querySelectorAll('#st1Cands .cand-row').forEach((r) => r.classList.toggle('hot', +r.dataset.i === ci)); V().render(); }
      }
    });
    cv.addEventListener('pointerleave', () => { if (S.hover) { S.hover = null; V().render(); } });
    const endPointer = (e) => {
      if (S.resizing && e.pointerId === S.resizing.pid) { S.resizing = null; setRadius(S.radius); updateDrawTools(); return; }
      if (S.drawing && e.pointerId === S.drawing.pid) {
        const d = S.drawing; S.drawing = null; (S.strokes[d.f] = S.strokes[d.f] || []).push({ mode: d.mode, radius_px: d.radius_px, p: d.p }); V().render();
        if (S.stage === ST.SAM3) scheduleBrushKey(d.f);
        else if (S.stage === ST.BRUSH && window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.onDraw(d.f);
        else if (S.stage === ST.FINAL_TOUCH && window.MD_FT && window.MD_FT.ready) window.MD_FT.onDraw(d.f);
        else if (S.stage === ST.VITMATTE && window.MD_ZONES && window.MD_ZONES.ready && window.MD_ZONES.wantsDraw()) window.MD_ZONES.onDraw(d.f);
      }
    };
    cv.addEventListener('pointerup', endPointer);
    cv.addEventListener('pointercancel', endPointer);
    window.addEventListener('mouseup', (e) => {
      if (S.candClick) {   // Z10
        const c = S.candClick; S.candClick = null;
        if (e.target === cv && Math.hypot(e.clientX - c.x, e.clientY - c.y) <= 4 && S.cands) { run('st1Detect', () => chooseCandidate(c.i)); return; }
      }
      if (S.edgeClick) {
        const c = S.edgeClick; S.edgeClick = null;
        if (e.target === cv && Math.hypot(e.clientX - c.x, e.clientY - c.y) <= 4) { const pc = edgePieceAt(e); if (pc) toggleEdgePiece(pc); }
      }
      if (S.clickStart && clickTool() && S.obj) {
        const c = S.clickStart; S.clickStart = null;
        if (Math.hypot(e.clientX - c.x, e.clientY - c.y) > 4) return;      // it was a pan
        if (e.target !== cv) return;
        const f = V().currentFrame;
        if (S.tool === 'corr_add' || S.tool === 'corr_remove') {   // local correction: + = fill the hole / missed part, − (PPM) = remove the leak
          const p = canvasToPlate(e); if (!p) return;
          const ps = plateSize(); if (ps && (p[0] < 0 || p[1] < 0 || p[0] >= ps[0] || p[1] >= ps[1])) { toast('Klik poza kadrem – pominięty'); return; }
          const inv = c.alt || c.b === 2;
          correctAt(f, p, (S.tool === 'corr_remove') !== inv ? 'remove' : 'add');
          return;
        }
        const hit = hitPoint(e);
        if (hit >= 0) { S.points[f].splice(hit, 1); }
        else {
          const p = canvasToPlate(e); if (!p) return;
          const ps = plateSize(); if (ps && (p[0] < 0 || p[1] < 0 || p[0] >= ps[0] || p[1] >= ps[1])) { toast('Klik poza kadrem – punkt pominięty'); return; }
          const neg = (S.tool === 'pt_remove') !== (c.alt || c.b === 2);
          (S.points[f] = S.points[f] || []).push([p[0], p[1], neg ? -1 : 1]);
        }
        markPoints(f);
      }
    });
    window.addEventListener('keydown', async (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || !S.obj || S.auxMode) return;
      const f = V() && V().currentFrame;
      const brushKeys = [ST.BRUSH, ST.VITMATTE, ST.EXPERIMENTS, ST.FINAL_TOUCH].includes(S.stage);   // G / R / P = brush / polygon (SAM3: points)
      const sam3 = S.stage === ST.SAM3;
      if (sam3 && (e.code === 'Enter' || e.code === 'NumpadEnter') && !S.poly && !e.ctrlKey && e.target.tagName !== 'BUTTON') { e.preventDefault(); computePoints(f); }
      else if (sam3 && e.code === 'KeyG') setTool('pt_add');
      else if (sam3 && e.code === 'KeyR' && !e.ctrlKey) setTool('pt_remove');
      else if (sam3 && e.code === 'KeyK') setTool('corr_add');
      else if (sam3 && e.code === 'KeyL') setTool('corr_remove');
      else if (sam3 && e.code === 'KeyP') setTool('poly');
      else if ((S.stage === ST.BRUSH || S.stage === ST.FINAL_TOUCH) && e.code === 'KeyE' && !e.ctrlKey) setTool('brush_erase');
      else if (e.code === 'Escape' && S.tool !== 'none' && !S.poly) setTool(S.tool);   // Esc = tool off (only panning) – Z6: every tool
      else if ((e.code === 'Delete' || e.code === 'Backspace') && !e.ctrlKey && !e.metaKey) {   // Z6
        e.preventDefault();
        if (sam3 && (S.points[f] || []).length) { S.points[f].pop(); markPoints(f); status(`F${f}: usunięto ostatni punkt (${S.points[f].length} zostało) – „▶ Przelicz” zapisze klucz`); }
        else if (S.stage === ST.BRUSH && window.MD_BRUSH && window.MD_BRUSH.ready) window.MD_BRUSH.deleteKeysAt(f);
        else if (S.stage === ST.FINAL_TOUCH && window.MD_FT && window.MD_FT.ready) window.MD_FT.deleteKeysAt(f);
        else if (sam3 || [ST.VITMATTE, ...BAND_STAGES()].includes(S.stage)) run('btnDelKey', () => deleteStageKey(f));
      }
      else if (bandStage() && e.code === 'KeyG') setTool('band_add');
      else if (bandStage() && e.code === 'KeyR' && !e.ctrlKey) setTool('band_remove');
      else if (brushKeys && e.code === 'KeyG') setTool('brush_add');
      else if (brushKeys && e.code === 'KeyR' && !e.ctrlKey) setTool('brush_remove');
      else if (brushKeys && S.stage !== ST.BRUSH && S.stage !== ST.FINAL_TOUCH && e.code === 'KeyP') setTool('poly');
      else if (e.code === 'KeyN' && !e.shiftKey) nextFlag();
      else if (e.code === 'KeyC' && !e.ctrlKey && !e.metaKey && S.stage !== ST.PLATES && S.stage !== ST.SAM3 && (V().activePlate || 'beauty') === 'beauty') { if (!S.holdC) { S.holdC = true; V().render(); } }   // Z9
      else if (e.code === 'KeyO' && !e.ctrlKey && !e.metaKey) { S.onion = !S.onion; toast(S.onion ? 'Onion skin ±1: czerwony = poprzednia, niebieski = następna klatka' : 'Onion skin wyłączony'); V().render(); }
      else if (e.code === 'KeyM' && !e.ctrlKey && !e.metaKey && versionStage()) { S.flicker = !S.flicker; toast(S.flicker ? 'Mapa migotania wersji (M wyłącza)' : 'Mapa migotania wyłączona'); V().render(); }
      else if (e.code === 'KeyB' && !e.ctrlKey && !e.metaKey && !e.shiftKey) toggleRunB();
      else if (e.code === 'KeyV' && viewAllowed('comp')) { const n = nextCompBg(); if (S.view !== 'comp') setView('comp', false); toast(`Tło compu: ${n}`); V().render(); }
      else if (drawStage() && e.code === 'BracketLeft') setRadius(S.radius / 1.25);
      else if (drawStage() && e.code === 'BracketRight') setRadius(S.radius * 1.25 + 1);
      else if (e.code === 'Enter' && S.poly && S.poly.p.length >= 3) {
        const pf = S.poly.f; S.poly.closed = true; (S.polys[pf] = S.polys[pf] || []).push(S.poly); S.poly = null; V().render();
        if (S.stage === ST.SAM3) scheduleBrushKey(pf);
        else if (S.stage === ST.VITMATTE && window.MD_ZONES && window.MD_ZONES.ready && window.MD_ZONES.wantsDraw()) window.MD_ZONES.onDraw(pf);
      } else if (e.code === 'Escape' && S.poly) { S.poly = null; V().render(); }
      else if (e.code === 'KeyZ' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (!e.shiftKey && (S.strokes[f] || []).length) { S.strokes[f].pop(); V().render(); return; }
        if (!e.shiftKey && (S.polys[f] || []).length) { S.polys[f].pop(); V().render(); return; }
        if (!e.shiftKey && S.stage === ST.VITMATTE && window.MD_ZONES && window.MD_ZONES.ready && await window.MD_ZONES.undo(f)) return;
        if (!e.shiftKey && BAND_STAGES().includes(S.stage) && window.MD_BANDPTS && window.MD_BANDPTS.ready && await window.MD_BANDPTS.undo(f)) return;
        if (!e.shiftKey && S.stage === ST.BRUSH && window.MD_BRUSH && window.MD_BRUSH.ready && await window.MD_BRUSH.undo(f)) return;
        if (S.stage === ST.FINAL_TOUCH && window.MD_FT && window.MD_FT.ready) { if (!e.shiftKey && !(await window.MD_FT.undo(f))) status('Nic do cofnięcia w Final Touch', 'warn'); return; }
        const d = await api('POST', `/api/object/${S.obj}/${e.shiftKey ? 'redo' : 'undo'}`);
        const OPS = { save_key: 'zapis klucza', delete_key: 'usunięcie klucza' };
        status(d.success ? `${e.shiftKey ? 'Ponowiono' : 'Cofnięto'} ${OPS[d.result.op] || d.result.op} F${d.result.frame}` : 'Nic do cofnięcia', d.success ? 'ok' : 'warn');
        clearOverlays(); await loadKeys(); loadStages(); V().render();
      }
    });

    window.addEventListener('keyup', (e) => { if (e.code === 'KeyC' && S.holdC) { S.holdC = false; note(''); V().render(); } });   // Z9
    window.addEventListener('md-frame', (e) => { renderKeysLane(); updatePointsBox(); const d = e.detail || {}; preloadOverlays(d.frame ?? V().currentFrame, d.dir || 0); });
    window.addEventListener('md-frame-loaded', () => renderLoadBar());   // plate frame arrived (app.js)
    window.addEventListener('resize', () => renderLoadBar());
    window.addEventListener('md-plate', (e) => setAuxMode(e.detail.plate !== 'beauty'));
    window.addEventListener('md-work-area', () => { const o = $('e5Scope') && $('e5Scope').querySelector('option[value="work_area"]'); if (o) o.textContent = `work area (${window.MD_WA.label()})`; loadStages().catch(() => {}); });
    // cut-out on a background (user request 2026-10-17): band / alpha previews off when entering it (they look like an
    // error over the cut-out), what the user ticks there is shown; leaving the view restores the previous state
    window.addEventListener('md-compare', (e) => {
      if (e.detail.covers) {
        S.pvBeforeComp = { edge: S.edgePv, vb: vmmPvGet('band', true), va: vmmPvGet('alpha', true), lb: ltxPvGet('band', true), la: ltxPvGet('alpha', true),
                           xe: expPvGet('edge', true), fl: !!S.flicker };
        S.edgePv = false; vmmPvSet('band', false); vmmPvSet('alpha', false); ltxPvSet('band', false); ltxPvSet('alpha', false);
        expPvSet('edge', false); S.flicker = false;
      } else if (S.pvBeforeComp) {
        const b = S.pvBeforeComp; S.pvBeforeComp = null;
        S.edgePv = b.edge; vmmPvSet('band', b.vb); vmmPvSet('alpha', b.va); ltxPvSet('band', b.lb); ltxPvSet('alpha', b.la);
        expPvSet('edge', b.xe); S.flicker = b.fl;
      }
      if (window.MD_ZONES && window.MD_ZONES.compareMode) window.MD_ZONES.compareMode(e.detail.covers);
      renderEdgePalette(); renderVmmPalette(); renderLtxPalette(); renderExpPalette(); if (V()) V().render();
    });
    window.addEventListener('md-project', () => { S.projectGen = (S.projectGen || 0) + 1; clearOverlays(); S.obj = null; refreshList().then(() => { setAuxMode((V().activePlate || 'beauty') !== 'beauty'); }); });
    window.addEventListener('matte-job-finished', (e) => {
      if (!S.obj) return;
      if (pendingPointFrames().length) setTimeout(() => flushPendingPoints().catch(() => {}), 300);
      if (e.detail && e.detail.quiet) { loadStages().catch(() => {}); return; }   // SAM3 on one frame: the points flow refreshes itself
      clearOverlays(); refreshList(S.obj).catch(() => {});
    });
  }

  function setAuxMode(on) {
    S.auxMode = on;
    if (on) { if (S.stage !== ST.PLATES) S.lastObjStage = S.stage; setStage(ST.PLATES); }
    else setStage(S.lastObjStage || ST.SAM3);
    updateViewTabs();
  }

  // ------------------------------------------------------------------------------------------------ public
  window.MD = {
    S, ST, STAGES, api, el, esc, toast, setStage, setView, loadStages, refreshList, renderStageBar,
    makeFloating: (...a) => makeFloating(...a), placeFloating: (...a) => placeFloating(...a),
    registerPane(n, pane) { pane.el.style.display = 'none'; $('stagePanel').appendChild(pane.el); PANES[n] = pane; },
    get stage() { return S.stage; }, get auxMode() { return S.auxMode; },
    status(t, kind) { status(t, kind); },
  };
  window.addEventListener('DOMContentLoaded', () => {
    const v0 = V(); if (!v0) return;
    const bar = $('stageObjectBox');
    bar.innerHTML = `<label class="obj-label">Obiekt</label><select id="objSelect" class="form-input"><option value="">— brak obiektów —</option></select><button class="btn btn-secondary btn-xs" id="objNewBtn" title="Nowy obiekt do wycięcia">＋</button><button class="btn btn-secondary btn-xs" id="objDelBtn" title="Usuń wybrany obiekt (z potwierdzeniem)">🗑</button>`;
    buildPanes();
    buildPointsPalette();   // prompt / detect controls live on the palette (wire() needs them)
    const origRender = v0.render.bind(v0);
    v0.render = function () { origRender(); drawOverlay(); };
    v0.frameReady = frameReady;   // Z20: Play waits for the plate + the overlay of the view
    wire();
    wireOverlayStyle();
    const hooks = { S, ST, api, el, esc, toast, status, run, invalidateFrame, clearOverlays, overlayFor, tintCanvas, putComp, compCanvas,
                    drawVarStroke, plateSize, loadStages, loadQA, setView, ensurePointsSaved, tileOverlays,
                    makeFloating, placeFloating, setTool, stageState, canvasToPlate, renderKeysLane, loadOverlay,
                    reloadStageBand: async () => {
                      await loadStages();
                      if (S.stage === ST.VIDEOMAMA) await loadVmm(); else if (S.stage === ST.LTX) await loadLtx();
                      else { S.epvParams = null; await fillBandSources('ep'); }
                    } };
    if (window.MD_ZONES) window.MD_ZONES.init(hooks);
    if (window.MD_BANDPTS) window.MD_BANDPTS.init(hooks);
    if (window.MD_BRUSH) window.MD_BRUSH.init(hooks);
    if (window.MD_FT) window.MD_FT.init(hooks);
    renderStageBar(); updateViewTabs(); updateDrawTools();
    setStage(ST.SAM3);
    const ws = new WebSocket(`ws://${location.host}/ws/events`);
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.type === 'object_progress') status(`${m.step}: ${m.done}/${m.total}`); };
    setTimeout(() => { if (v0.projectLoaded && !S.objects.length) refreshList(); }, 1200);
  });
})();
