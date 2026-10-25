// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/finaltouch.js – stage Final Touch (PLAN_FINAL_TOUCH.md step 1, 2026-10-22): hand corrections of the alpha,
// frame by frame, with a bitmap brush over a chosen computed version (the base) – no tracking yet. Strokes are stored as
// paths per frame (engine/final_touch.py, PUT final_touch/frames/{f}); „Dodaj” pulls the alpha to 1, „Usuń” to 0,
// „Zmaż” back to the base, soft by the hardness. The Alpha / Comp view of the stage = base + corrections
// (objects.js matteFor → GET final_touch/frame/{f}); the export source „Final Touch” writes the same.
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  let H = null;                       // hooks from objects.js
  const S = () => H.S;
  const obj = () => H.S.obj;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const F = { info: null, undo: [], rev: {}, timer: null, pending: null, busy: false, hardness: 0.6, showStrokes: true, obj: null };
  const MODES = { add: { label: 'Dodaj', css: '#50ff78', col: [80, 255, 120] }, remove: { label: 'Usuń', css: '#ff3c5a', col: [255, 60, 90] },
                  erase: { label: 'Zmaż', css: '#c4c4c8', col: [200, 200, 210] } };
  const BASES = [['', 'domyślna – wynik ostatniego etapu'], ['core', 'Maska 1+2 (etapy 1 + 2)'], ['matanyone', 'MatAnyone2'],
                 ['videomama', 'VideoMaMa'], ['vitmatte', 'ViTMatte'], ['ltx', 'LTX-2.5'], ['active', 'Aktywna wersja'], ['version', 'Wersja z listy']];
  // finish sliders (user decision 2026-10-22: moved here from the ViTMatte zone): key, label, min, max, step, title
  const FIN = [
    ['black', 'Czerń', 0, 0.9, 0.01, 'Alfa poniżej tej wartości → 0, liniowo do bieli (usuwa mgłę i śmieci w tle; bez schodka)'],
    ['white', 'Biel', 0.1, 1, 0.01, 'Alfa powyżej tej wartości → 1 (domyka prawie pełne kosmyki i brzeg)'],
    ['gamma', 'Gamma', 0.2, 5, 0.05, 'Gamma półprzezroczystości: > 1 rozjaśnia (grubsze kosmyki), < 1 ściemnia; 1 = bez zmian'],
    ['erode_px', 'Erode', -50, 50, 0.5, 'Cofnięcie krawędzi alfy w px (filtr min); ujemne = rozszerzenie (max). Miękka krawędź zostaje miękka'],
    ['mb_strength', 'Siła', 0, 3, 0.05, 'Rozmycie ruchu alfy wzdłuż ruchu OBIEKTU (przepływ plate\'u): 1 = fizyczne rozmycie migawki, 0 = wyłączone; pole = dowolna wartość do 10. Nieruchome części zostają ostre. Po erode, przed rozmyciem'],
    ['mb_shutter_deg', 'Shutter', 0, 360, 1, 'Kąt migawki (°) dla motion bluru; domyślnie z manifestu ujęcia („↺ z manifestu”)'],
    ['blur_px', 'Rozmycie', 0, 50, 0.5, 'Rozmycie Gaussa całej alfy w px (na końcu)'],
  ];
  const FIN_DEF = { black: 0, white: 1, gamma: 1, erode_px: 0, blur_px: 0, mb_strength: 0, mb_shutter_deg: null, mb_measured: true };
  const GRADE = ['black', 'white', 'gamma', 'erode_px', 'blur_px'];   // palette group „Grade”; the rest = „Motion blur”
  const finRow = ([k, l, min, max, step, t]) => `<label class="zone-slider" title="${esc(t)}">${l} <input type="range" data-fin="${k}" min="${min}" max="${max}" step="${step}" style="width:100%"><b data-finv="${k}"></b></label>`;
  const finFmt = (k, v) => (k === 'erode_px' || k === 'blur_px' ? `${(+v).toFixed(1)} px` : k === 'mb_shutter_deg' ? `${Math.round(+v)}°` : k === 'mb_strength' ? `×${(+v).toFixed(2)}` : (+v).toFixed(2));
  const STAGE_OF = { core: 'core', matanyone: 'matanyone', videomama: 'videomama', vitmatte: 'vitmatte', ltx: 'ltx', active: 'final' };
  const api = (...a) => H.api(...a);
  const status = (t, k) => H.status(t, k);
  const curFrame = () => V().currentFrame;
  const inStage = () => !!(H && obj() && !S().auxMode && S().stage === H.ST.FINAL_TOUCH);
  const framesOf = () => Object.keys((F.info && F.info.frames) || {}).map(Number).sort((a, b) => a - b);
  const strokesAt = (f) => ((F.info && F.info.frames) || {})[String(f)] || [];
  try { const h = +localStorage.getItem('md_ft_hardness'); if (h >= 0 && h <= 1 && localStorage.getItem('md_ft_hardness') !== null) F.hardness = h; } catch (e) { /* storage blocked */ }
  try { F.showStrokes = localStorage.getItem('md_ft_strokes') !== '0'; } catch (e) { /* storage blocked */ }   // as „👁 podgląd pasa”
  F.strokesOpacity = 30;   // % – opacity of the brush overlay (default 30, user decision 2026-10-22), as the mask overlay's slider
  try { const o = +localStorage.getItem('md_ft_strokes_o'); if (o >= 0 && o <= 100 && localStorage.getItem('md_ft_strokes_o') !== null) F.strokesOpacity = o; } catch (e) { /* storage blocked */ }

  // ------------------------------------------------------------------------------------------------ panel
  function panelHtml() {
    return `<div class="pane-section">
        <div class="muted">Narzędzia tego etapu: <b>paleta „Final Touch”</b> na obrazie (po lewej) – baza, poziomy i wykończenie, motion blur, twardość pędzla; <b>pędzel</b> Dodaj (G) / Usuń (R) / Gumka (E) i „Usuń klucz” (korekty tej klatki) – lewy pasek.</div>
      </div>
      <div class="pane-section">
        <div class="sec-title">Jak to działa</div>
        <div class="muted"><b>Baza</b> = dowolna policzona wersja (domyślnie wynik ostatniego etapu); zmiana bazy w każdej chwili – korekty zostają i nakładają się na nową bazę (sprawdź miejsca korekt).
          <b class="ok">Dodaj</b> maluje biel (obiekt), <b class="err">Usuń</b> czerń (tło), <b>Gumka</b> wymazuje obie (alfa wraca do bazy), „Usuń klucz” kasuje wszystkie korekty klatki; kreska dotyczy <b>tylko tej klatki</b> (bez śledzenia), miękkość wg twardości.
          <b>Poziomy i wykończenie</b> działają na całej alfie PO kreskach: czerń / biel usuwają mgłę i domykają kosmyki, gamma rozjaśnia / ściemnia półprzezroczystość, erode cofa krawędź (ujemne = rozszerza),
          <b>motion blur</b> rozmywa wzdłuż ruchu obiektu (×1 = fizyczne rozmycie migawki; „siła z pomiaru plate'u” = hybryda: siła per odcinek krawędzi z szerokości przejścia na plate'cie), rozmycie Gaussa na końcu.
          Widok Alpha pokazuje bazę + korekty + wykończenie – to samo idzie do eksportu (źródło „Final Touch”, domyślne dla obiektu z korektami).
          Skróty: [ ] rozmiar pędzla, Shift + przeciągnij = rozmiar, Ctrl+Z cofa kreskę, Delete / 🔑✕ usuwa korekty klatki.</div>
      </div>
      <div class="pane-section"><div class="sec-title">Klatki z korektą <span class="muted">(kwadraty na osi czasu; PPM = usuń)</span></div><div id="ftList" class="muted"></div></div>
      <div class="row"><button type="button" class="btn btn-ghost btn-xs" id="ftDelFrame" title="Usuń wszystkie korekty tej klatki [Delete] (Ctrl+Z cofa)">🗑 Usuń korekty (ta klatka)</button></div>`;
  }
  // floating palette on the canvas (left, draggable – as „Punkty SAM3” / „Krawędź” / „VideoMaMa”): the stage's tools
  function paletteHtml() {
    return `<div class="pts-palette ft-palette" id="ftPalette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Final Touch</div>
      <label title="Wersja, na którą nakładają się korekty: dowolny policzony wynik (jak źródło eksportu). Zmiana bazy w każdej chwili – korekty zostają i nakładają się na nową bazę (sprawdź miejsca korekt)">Baza<select id="ftBase" class="form-input"></select></label>
      <label id="ftRunL" style="display:none" title="Która wersja z listy jest bazą">Wersja<select id="ftRun" class="form-input"></select></label>
      <div class="muted pp-hint" id="ftBaseInfo"></div>
      <div class="pp-chk"><label class="chk" title="Warstwa korekt tej klatki na obrazie (zielone = dodaje, czerwone = odejmuje; gumka ją wymazuje). Wyłącz, żeby oceniać samą alfę"><input type="checkbox" id="ftShow" ${F.showStrokes ? 'checked' : ''}> 👁 kreski pędzla</label>
        <input type="range" id="ftShowO" min="0" max="100" step="5" value="${F.strokesOpacity}" style="width:70px" title="Krycie podglądu kresek (%)"><span class="muted" id="ftShowOV">${F.strokesOpacity} %</span></div>
      <div class="pp-sec">
        <div class="zp-sub" title="Grading alfy – cała alfa, po kreskach pędzla; na żywo, bez przeliczania modeli; to samo idzie do eksportu">Grade</div>
        <div class="muted pp-hint warn" id="ftHardHint" style="display:none">Baza „maska 1+2” jest twarda (tylko 0 i 1): czerń / biel / gamma zmieniają tylko półprzezroczystość, więc tu działają wyłącznie na miękkich kreskach pędzla. Na całej alfie działają erode, rozmycie i motion blur; czerń / biel / gamma – na bazie z miękką krawędzią (MatAnyone2, VideoMaMa, ViTMatte / BiRefNet, LTX-2.5).</div>
        ${FIN.filter(([k]) => GRADE.includes(k)).map(finRow).join('')}
        <button type="button" class="btn btn-ghost btn-xs" id="ftFinReset" title="Wyzeruj grade i motion blur (baza bez zmian)">↺ bez wykończenia</button>
      </div>
      <div class="pp-sec">
        <div class="zp-sub" title="Rozmycie ruchu alfy wzdłuż ruchu obiektu (po grade, przed rozmyciem Gaussa)">Motion blur</div>
        ${FIN.filter(([k]) => !GRADE.includes(k)).map(finRow).join('')}
        <div class="row"><label class="chk" title="Hybryda (plan 2026-10-22): kierunek rozmycia z ruchu obiektu, SIŁA per odcinek krawędzi zmierzona na plate'cie – szerokość przejścia 10–90 % w poprzek krawędzi wobec rozmycia przewidzianego z prędkości i migawki (łapie obrót, palce, rolling shutter, defocus). Wyłączone = jedna siła na cały obiekt"><input type="checkbox" id="ftMbMeasured" checked> siła z pomiaru plate'u</label>
          <button type="button" class="btn btn-ghost btn-xs" id="ftMbShutterManifest" title="Shutter z manifestu ujęcia">↺ z manifestu</button></div>
      </div>
      <div class="pp-sec">
        <div class="zp-sub" title="Dodaj (G) / Usuń (R) / Gumka (E) na lewym pasku; kreska = tylko ta klatka">Pędzel</div>
        <label title="Twardość pędzla: 100 % = ostra krawędź kreski, 0 % = miękka (krycie spada od środka do krawędzi kreski; nic nie wychodzi poza promień)">Twardość <span id="ftHardN">${Math.round(F.hardness * 100)} %</span>
          <input type="range" id="ftHard" min="0" max="100" value="${Math.round(F.hardness * 100)}" style="width:100%"></label>
      </div>
    </div>`;
  }
  const ftPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_ft_palette') || 'null'); } catch (e) { return null; } },
    set: (q) => { try { if (q) localStorage.setItem('md_ft_palette', JSON.stringify(q)); else localStorage.removeItem('md_ft_palette'); } catch (e) { /* storage blocked */ } },
  };
  function renderPalette() {
    const box = $('ftPalette'); if (!box) return;
    const show = inStage() && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (show && H.placeFloating) H.placeFloating(box, $('canvasWrapper'), ftPos.get());
  }
  function baseLabel(src) { return (BASES.find((b) => b[0] === src) || [src, src])[1]; }
  function stageDone(src) { return !STAGE_OF[src] || ['done'].includes(H.stageState(STAGE_OF[src]).state); }
  function renderBase() {
    const sel = $('ftBase'); if (!sel || !F.info) return;
    const ch = F.info.base_choice || {}; const b = F.info.base || {};
    const opts = BASES.map(([k, l]) => {
      const ok = k === '' || k === 'version' ? true : stageDone(k);
      const txt = k === '' && !ch.source && b.source ? `${l} (${baseLabel(b.source)})` : l;
      return `<option value="${k}"${ok || k === (ch.source || '') ? '' : ' disabled'}>${esc(window.mdT ? window.mdT(txt) : txt)}${ok ? '' : esc(window.mdT ? window.mdT(' – brak wyniku') : ' – brak wyniku')}</option>`;
    }).join('');
    if (sel.dataset.html !== opts) { sel.innerHTML = opts; sel.dataset.html = opts; }
    sel.value = ch.source || '';
    $('ftRunL').style.display = sel.value === 'version' ? '' : 'none';
    if (sel.value === 'version') {
      const rs = (S().runs || []).slice().reverse();
      $('ftRun').innerHTML = rs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}${r.stage ? ` · ${esc(r.stage)}` : ''}</option>`).join('');
      if (ch.run) $('ftRun').value = ch.run;
    }
    const info = $('ftBaseInfo');
    info.textContent = b.run ? `Baza: ${b.run === 'core' ? 'maska 1+2' : `wersja ${b.run}`} (${baseLabel(b.source)}${b.chosen ? '' : ', domyślna'})` : (b.why || 'Brak bazy');
    info.className = b.run ? 'muted' : 'muted warn';
    // a hard base (mask 1+2 = 0 / 1 only): levels and gamma have no semi-transparency to act on (user report 2026-10-23
    // "black / white sliders change nothing") – say so and dim them instead of leaving them looking broken
    const hard = b.run === 'core';
    $('ftHardHint').style.display = hard ? '' : 'none';
    for (const k of ['black', 'white', 'gamma']) {
      const row = document.querySelector(`#ftPalette [data-fin="${k}"]`);
      if (row) row.closest('label').style.opacity = hard ? '.5' : '';
    }
  }
  function renderList() {
    const box = $('ftList'); if (!box) return;
    const fr = framesOf();
    box.innerHTML = fr.length ? fr.map((f) => `<a href="#" data-f="${f}" title="Skok do klatki ${f} (${strokesAt(f).length} kresek)">F${f}</a>`).join(' · ')
      + ` <span class="muted">(${fr.length})</span>` : 'brak korekt – eksport „Final Touch” = sama baza';
    box.querySelectorAll('a[data-f]').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); V().goToFrame(+a.dataset.f); }));
    renderLane(); syncTools();
    // 🔑✕ / Delete follow the corrections at once, not only after a frame change (user report 2026-10-22: the key
    // of the first stroke could not be deleted until the playhead left and came back)
    if (H.renderKeysLane) H.renderKeysLane();
  }
  function syncTools() { /* the brush buttons live on the left bar (objects.js updateDrawTools) */ }

  // ------------------------------------------------------------------------------------------------ loading
  async function load() {
    if (!obj()) { F.info = null; renderAll(); return; }
    try { F.info = await api('GET', `/api/object/${obj()}/final_touch`); } catch (e) { F.info = null; }
    renderAll();
  }
  function renderFinish() {
    const fin = { ...FIN_DEF, ...((F.info && F.info.finish) || {}) };
    const shM = F.info && F.info.shutter_manifest != null ? +F.info.shutter_manifest : 180;
    for (const [k] of FIN) {
      const s = document.querySelector(`#ftPalette [data-fin="${k}"]`), v = document.querySelector(`#ftPalette [data-finv="${k}"]`);
      const val = k === 'mb_shutter_deg' && fin[k] == null ? shM : fin[k];
      if (s && document.activeElement !== s) s.value = val;
      if (v) v.textContent = finFmt(k, val) + (k === 'mb_shutter_deg' && fin[k] == null ? ' (manifest)' : '');
    }
    const cb = $('ftMbMeasured'); if (cb) cb.checked = fin.mb_measured !== false;
    const bm = $('ftMbShutterManifest'); if (bm) bm.disabled = fin.mb_shutter_deg == null;
    const on = FIN.some(([k]) => k !== 'mb_shutter_deg' && Math.abs(fin[k] - FIN_DEF[k]) > 1e-9);
    const b = $('ftFinReset'); if (b) b.disabled = !on;
  }
  async function putFinish(body, what) {
    try {
      F.info = await api('PUT', `/api/object/${obj()}/final_touch/finish`, body);
      renderFinish(); H.loadStages().catch(() => {}); V().render();
      status(`Wykończenie: ${what} – podgląd i eksport biorą to od razu`, 'ok');
    } catch (e) { status(`Wykończenie NIE zapisane: ${e.message}`, 'err'); renderFinish(); }
  }
  function renderAll() { renderBase(); renderFinish(); renderList(); renderPalette(); }
  function reset() {
    const same = F.obj === obj(); F.obj = obj();
    F.info = null; F.rev = {};
    if (!same) F.undo = [];
    clearTimeout(F.timer); F.pending = null; renderLane();
  }
  // the base + corrections image key of a frame (objects.js matteFor): changes with the base and with every edit of f
  function matteRef(f) {
    const b = F.info && F.info.base;
    if (!b || !b.run) return null;
    return { key: `${obj()}_ft@${b.run}_${F.info.sig || '0'}_${F.rev[f] || 0}_${f}`,
             url: `/api/object/${obj()}/final_touch/frame/${f}?b=${encodeURIComponent(b.run)}&c=${F.info.sig || '0'}.${F.rev[f] || 0}`,
             run: b.run };
  }

  // ------------------------------------------------------------------------------------------------ drawing -> frame strokes
  function thin(p, r) {   // fewer points for the JSON: drop points closer than r / 4 to the previous one (the ends stay)
    if (p.length <= 2) return p;
    const out = [p[0]]; const d = Math.max(0.5, r / 4);
    for (let i = 1; i < p.length - 1; i++) { const q = out[out.length - 1]; if (Math.hypot(p[i][0] - q[0], p[i][1] - q[1]) >= d) out.push(p[i]); }
    out.push(p[p.length - 1]);
    return out;
  }
  function onDraw(f) {
    clearTimeout(F.timer); F.pending = f;
    F.timer = setTimeout(() => commit(f).catch(() => {}), 150);
  }
  async function commit(f) {
    clearTimeout(F.timer); F.pending = null;
    if (F.busy) { F.pending = f; F.timer = setTimeout(() => commit(f).catch(() => {}), 200); return; }
    const drawn = S().strokes[f] || [];
    if (!drawn.length || !obj()) return;
    S().strokes[f] = [];
    const prev = strokesAt(f).slice();
    const add = drawn.map((s) => ({ mode: s.mode, radius_px: s.radius_px, hardness: F.hardness, p: thin(s.p, s.radius_px) }));
    F.busy = true;
    try {
      F.info = await api('PUT', `/api/object/${obj()}/final_touch/frames/${f}`, { strokes: prev.concat(add) });
      F.undo.push({ f, prev }); F.rev[f] = (F.rev[f] || 0) + 1;
      status(`F${f}: korekta zapisana (${add.map((s) => MODES[s.mode].label).join(', ')}) – Ctrl+Z cofa`, 'ok');
      H.toast(`F${f}: ${add.map((s) => MODES[s.mode].label).join(', ')}`, 1500);   // the panel's status line is far below (user 2026-10-22)
      H.loadStages().catch(() => {});
    } catch (e) {
      S().strokes[f] = drawn.concat(S().strokes[f] || []);
      status(`Korekta F${f} NIE zapisana: ${e.message}`, 'err'); H.toast(`Korekta F${f} NIE zapisana: ${e.message}`, 6000);
    } finally { F.busy = false; renderList(); V().render(); }
  }
  async function putFrame(f, strokes) {
    F.info = await api('PUT', `/api/object/${obj()}/final_touch/frames/${f}`, { strokes });
    F.rev[f] = (F.rev[f] || 0) + 1;
    H.loadStages().catch(() => {}); renderList(); V().render();
  }
  async function undo() {
    const u = F.undo[F.undo.length - 1];
    if (!u || !obj()) return false;
    F.undo.pop();
    try { await putFrame(u.f, u.prev); status(`Cofnięto kreskę korekty F${u.f}`, 'ok'); if (curFrame() !== u.f) V().goToFrame(u.f); }
    catch (e) { status(`Cofanie korekty: ${e.message}`, 'err'); }
    return true;
  }
  async function deleteKeysAt(f) {
    const prev = strokesAt(f).slice();
    if (!prev.length) { status(`F${f}: brak korekt w tej klatce`, 'warn'); return false; }
    if (!(await V().uiConfirm(`Usunąć korekty klatki ${f} (${prev.length} kresek)?\nCtrl+Z cofa.`, { ok: 'Usuń' }))) return false;
    try {
      F.info = await api('DELETE', `/api/object/${obj()}/final_touch/frames/${f}`);
      F.undo.push({ f, prev }); F.rev[f] = (F.rev[f] || 0) + 1;
      status(`F${f}: korekty usunięte (Ctrl+Z cofa)`, 'ok');
    } catch (e) { status(`Nie usunięto korekt F${f}: ${e.message}`, 'err'); }
    H.loadStages().catch(() => {}); renderList(); V().render();
    return true;
  }
  async function setBase(source, run) {
    F.info = await api('PUT', `/api/object/${obj()}/final_touch`, { source: source || null, run: run || null });
    F.rev = {}; renderAll(); H.loadStages().catch(() => {}); V().render();
    const b = F.info.base;
    status(b.run ? `Baza Final Touch: ${b.run === 'core' ? 'maska 1+2' : b.run} – korekty nałożone na nową bazę; sprawdź klatki z korektą` : (b.why || 'Brak bazy'), b.run ? 'ok' : 'warn');
  }

  // ------------------------------------------------------------------------------------------------ overlays (inside pan / zoom)
  function draw(ctx, img, k, f) {
    if (!inStage() || !F.showStrokes || !strokesAt(f).length) return;
    // the correction layer of the frame from the server (green = add, red = remove, alpha = strength): what the strokes
    // really do after the eraser and the hardness – stroke outlines showed every disc of the path and never the eraser
    // (user report 2026-10-22)
    const key = `${obj()}_ftlayer_${F.info.sig || '0'}_${F.rev[f] || 0}_${f}`;
    const ov = H.loadOverlay ? H.loadOverlay(key, `/api/object/${obj()}/final_touch/layer/${f}?c=${F.info.sig || '0'}.${F.rev[f] || 0}`) : null;
    F.lastLayer = F.lastLayer || {};
    const lk = `${obj()}_${f}`;
    if (ov && ov !== 'missing') F.lastLayer[lk] = ov;
    const show = ov === 'missing' ? null : (ov || F.lastLayer[lk]);   // the previous layer stays until the new one arrives
    if (show) { const pa = ctx.globalAlpha; ctx.globalAlpha = pa * F.strokesOpacity / 100; ctx.drawImage(show, 0, 0, img.width, img.height); ctx.globalAlpha = pa; }
  }

  // ------------------------------------------------------------------------------------------------ timeline lane
  function laneEl() {
    let lane = $('timelineFT'); if (lane) return lane;
    const keys = $('timelineKeys'); if (!keys) return null;
    lane = H.el('<div class="timeline-zones" id="timelineFT" title="Klatki z korektą Final Touch: klik = skok, PPM = usuń korekty klatki"></div>');
    keys.parentNode.insertBefore(lane, keys);
    return lane;
  }
  function renderLane() {
    const lane = laneEl(); const v = V(); if (!lane || !v || !H) return;
    const fr = framesOf();
    const on = inStage() && fr.length;
    lane.style.display = on ? 'block' : 'none';
    lane.innerHTML = '';
    if (!on) return;
    const range = Math.max(1, v.lastFrame - v.firstFrame);
    for (const f of fr) {
      const d = document.createElement('div');
      d.className = `zone-key ft-key ${f === v.currentFrame ? 'cur' : ''}`;
      d.style.left = `${100 * (f - v.firstFrame) / range}%`; d.style.background = '#f0c419'; d.style.transform = 'none';
      d.title = `Korekta Final Touch F${f} (${strokesAt(f).length} kresek) · klik = skok · PPM = usuń korekty klatki`;
      d.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
      d.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); v.goToFrame(f); deleteKeysAt(f); });
      lane.appendChild(d);
    }
  }

  // ------------------------------------------------------------------------------------------------ wiring
  async function flush() { if (F.pending != null) await commit(F.pending); }
  function wire() {
    $('ftHard').addEventListener('input', () => {
      F.hardness = +$('ftHard').value / 100; $('ftHardN').textContent = `${$('ftHard').value} %`;
      try { localStorage.setItem('md_ft_hardness', String(F.hardness)); } catch (e) { /* storage blocked */ }
    });
    $('ftShow').addEventListener('change', () => {
      F.showStrokes = $('ftShow').checked; V().render();
      try { localStorage.setItem('md_ft_strokes', F.showStrokes ? '1' : '0'); } catch (e) { /* storage blocked */ }
    });
    $('ftShowO').addEventListener('input', () => {
      F.strokesOpacity = +$('ftShowO').value; $('ftShowOV').textContent = `${F.strokesOpacity} %`; V().render();
      try { localStorage.setItem('md_ft_strokes_o', String(F.strokesOpacity)); } catch (e) { /* storage blocked */ }
    });
    document.querySelectorAll('#ftPalette [data-fin]').forEach((s) => {
      const k = s.dataset.fin; const lbl = FIN.find((x) => x[0] === k)[1];
      s.addEventListener('input', () => { const v = document.querySelector(`#ftPalette [data-finv="${k}"]`); if (v) v.textContent = finFmt(k, s.value); });
      s.addEventListener('change', () => H.run('ftPanel', () => putFinish({ [k]: +s.value }, `${lbl} ${finFmt(k, s.value)}`)));
    });
    $('ftFinReset').addEventListener('click', () => H.run('ftFinReset', () => putFinish({ ...FIN_DEF, mb_shutter_deg: undefined, mb_shutter_manifest: true }, 'wyzerowane')));
    $('ftMbMeasured').addEventListener('change', () => H.run('ftPanel', () => putFinish({ mb_measured: $('ftMbMeasured').checked }, $('ftMbMeasured').checked ? 'siła motion bluru z pomiaru plate\'u' : 'jedna siła motion bluru')));
    $('ftMbShutterManifest').addEventListener('click', () => H.run('ftMbShutterManifest', () => putFinish({ mb_shutter_manifest: true }, 'shutter z manifestu')));
    $('ftDelFrame').addEventListener('click', () => H.run('ftDelFrame', () => deleteKeysAt(curFrame())));
    $('ftBase').addEventListener('change', () => H.run('ftBase', async () => {
      const src = $('ftBase').value;
      if (src === 'version') {
        if (!(S().runs || []).length) S().runs = (await api('GET', `/api/object/${obj()}/runs`)).runs;
        renderBase(); $('ftBase').value = 'version'; $('ftRunL').style.display = '';
        const runs = (S().runs || []).slice().reverse();
        $('ftRun').innerHTML = runs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}${r.stage ? ` · ${esc(r.stage)}` : ''}</option>`).join('');
        if (!runs.length) { status('Brak wersji na liście', 'warn'); return; }
        await setBase('version', $('ftRun').value);
      } else await setBase(src, null);
    }));
    $('ftRun').addEventListener('change', () => H.run('ftRun', () => setBase('version', $('ftRun').value)));
    window.addEventListener('md-frame', () => { renderLane(); });
  }
  function init(hooks) {
    H = hooks;
    const host = $('ftPanel'); if (!host) return;
    host.innerHTML = panelHtml();
    const cw = $('canvasWrapper');
    if (cw) { const pal = H.el(paletteHtml()); cw.appendChild(pal); if (H.makeFloating) H.makeFloating(pal, cw, '.pp-head', ftPos); }
    wire();
    window.MD_FT.ready = true;
  }
  window.MD_FT = {
    init, load, reset, draw, onDraw, undo, flush, deleteKeysAt, renderLane, matteRef,
    hasKeyAt: (f) => strokesAt(f).length > 0,
    onShow() { load(); renderPalette(); },
    onStage() { renderLane(); syncTools(); renderPalette(); },
    onStages() { renderBase(); },
    get state() { return F; },
  };
})();
