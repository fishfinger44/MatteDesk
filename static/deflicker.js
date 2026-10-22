// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Deflicker palette (docs/PLAN_DEFLICKER_AUTO.md § 4–5): built into the clean modal at run time.
// "Zmierz migotanie" -> POST /api/plate/clean/analyze (±10 frames around the current frame, 1024 px) -> the sliders set
// themselves from the automat and the notes say why; "👁 Podgląd klatek" -> POST /api/plate/clean/preview (the deflicker
// on that window at 1024 px): before / after loop with a wipe, heat maps on one scale, numbers per region incl. the floor
// ("do osiągnięcia"). "⚡ Deflicker" (app.js runPlateClean) takes the palette's knobs through window.deflickerPalette.knobs().
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const P = {                     // palette state
    auto: null,                   // settings from the last measurement (DeflickerSettings.to_dict)
    measured: null,               // last analyze response
    preview: null,                // last preview response
    stale: false,                 // settings changed after the preview
    frames: [],                   // preloaded Image pairs [before, after]
    playing: false, idx: 0, timer: null, wipe: 1.0, mode: 'wipe',
  };

  const SLIDERS = [
    { key: 'strength', id: 'dfStrength', label: 'Siła', min: 1, max: 8, step: 1, unit: '', tip: 'Ile razy stopień „obszary” poprawia mapę jasności (iteracje). Automat: 2–3 z pomiaru. Powyżej 3 zdejmuje trochę więcej migotania, ale może zostawić ślady poruszających się obiektów (sprawdź na podglądzie)' },
    { key: 'window', id: 'dfWindow', label: 'Okno czasowe', min: 5, max: 30, step: 1, unit: ' kl.', pre: '±', tip: 'Z iloma sąsiednimi klatkami porównywana jest każda klatka. Pulsowanie losowe → ±10; zmiany z pamięcią → dłużej. Pełne liczenie ogranicza do połowy długości ujęcia' },
    { key: 'smoothness_px', id: 'dfSmooth', label: 'Gładkość mapy', min: 2, max: 16, step: 1, unit: ' px', tip: 'Najmniejsza plama jasności, jaką poprawia stopień „obszary” (σ w 512 px). Mniej = drobniejsze plamy, ale ryzyko śladów faktury' },
    { key: 'texture_radius', id: 'dfTexR', label: 'Faktura i cienie: okno', min: 1, max: 6, step: 1, unit: ' kl.', pre: '±', tip: 'Mediana wzoru faktury z ±N klatek (tylko przy zapisie EXR, gdy opcja „Migotanie faktury i cieni” jest włączona). Pulsowanie → ±4, zmiany z pamięcią → ±3 / ±2' },
  ];

  // slider rows with literal ids (tests/test_static_js.py checks every element lookup against id="…" literals)
  function row(s, id) {
    return `<label class="df-row" title="${esc(s.tip)}"><span class="df-l">${esc(s.label)}</span>
        <input type="range" id="${id}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.min}">
        <span class="df-v" id="${id}V">–</span></label>`;
  }
  function html() {
    const [st, wi, sm, tx] = SLIDERS;
    const rows = row(st, 'dfStrength').replace('id="dfStrength"', 'id="dfStrength"') + row(wi, 'dfWindow') + row(sm, 'dfSmooth') + row(tx, 'dfTexR');
    // ids used above: id="dfStrength" id="dfStrengthV" id="dfWindow" id="dfWindowV" id="dfSmooth" id="dfSmoothV" id="dfTexR" id="dfTexRV"
    return `
      <div class="clean-options-section" id="dfPalette" style="margin-top:12px">
        <div class="section-title" style="display:flex;align-items:center;gap:8px">Ustawienia deflickera
          <span class="badge badge-info" id="dfSource" style="margin-left:auto">brak pomiaru</span></div>
        <p class="align-summary-text" style="margin:6px 0 0">Najpierw „Zmierz migotanie”: suwaki ustawią się same z pomiaru ±10 klatek wokół bieżącej klatki, a lista niżej powie dlaczego. Suwaki możesz potem przesunąć ręcznie.</p>
        <div id="dfSliders" style="margin-top:8px">${rows}</div>
        <label class="checkbox-label" style="margin-top:6px" title="Stopień „cała klatka” liczy osobną korektę dla R, G i B (kolor światła pływa) albo jedną dla jasności (kolor stabilny). Automat: z pomiaru koloru"><input type="checkbox" id="dfPerChannel" checked><span>Korekta R / G / B osobno (kolor światła pływa)</span></label>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
          <button class="btn btn-secondary btn-sm" id="dfApplyAuto" disabled title="Przywróć wartości z ostatniego pomiaru">↺ Ustaw z pomiaru</button>
          <button class="btn btn-secondary btn-sm" id="dfPreview" disabled title="Policz deflicker tylko na zmierzonym oknie (±10 kl.) w 1024 px i pokaż przed / po. To NIE jest wynik: nic się nie zapisuje">👁 Podgląd klatek</button>
          <span id="dfStale" class="warn" style="display:none;align-self:center">NIEAKTUALNY – zmieniono ustawienia, policz podgląd jeszcze raz</span>
        </div>
        <ul id="dfNotes" class="align-summary-text" style="margin:8px 0 0 16px;padding:0;display:none"></ul>
        <div id="dfWarnings" class="warn" style="margin-top:4px;display:none"></div>
      </div>
      <div class="clean-options-section" id="dfPreviewBox" style="margin-top:12px;display:none">
        <div class="section-title" style="display:flex;align-items:center;gap:8px">Podgląd: przed / po <span class="badge badge-info" id="dfPvInfo" style="margin-left:auto"></span></div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap">
          <button class="btn btn-secondary btn-sm" id="dfPlay">▶ Odtwarzaj</button>
          <button class="btn btn-secondary btn-sm" id="dfModeWipe" title="Lewa część: przed, prawa: po; przesuń linię">Wipe</button>
          <button class="btn btn-secondary btn-sm" id="dfModeA" title="Pokaż tylko przed">Przed</button>
          <button class="btn btn-secondary btn-sm" id="dfModeB" title="Pokaż tylko po">Po deflickerze</button>
          <input type="range" id="dfWipe" min="0" max="100" value="100" style="flex:1;min-width:120px" title="Pozycja linii wipe">
          <span id="dfFrameNo" class="df-v"></span>
        </div>
        <canvas id="dfCanvas" style="width:100%;margin-top:6px;background:#111;border-radius:3px"></canvas>
        <div style="display:flex;gap:8px;margin-top:8px">
          <figure style="flex:1;margin:0"><img id="dfMapB" style="width:100%;border-radius:3px" alt=""><figcaption class="align-summary-text">Mapa migotania: przed</figcaption></figure>
          <figure style="flex:1;margin:0"><img id="dfMapA" style="width:100%;border-radius:3px" alt=""><figcaption class="align-summary-text">Mapa migotania: po (ta sama skala)</figcaption></figure>
        </div>
        <table class="df-table" id="dfTable" style="width:100%;margin-top:8px;font-size:12px;border-collapse:collapse"></table>
        <ul id="dfPvNotes" class="align-summary-text" style="margin:6px 0 0 16px;padding:0"></ul>
      </div>`;
  }

  const CSS = `
    .df-row{display:grid;grid-template-columns:150px 1fr 70px;align-items:center;gap:8px;margin:4px 0}
    .df-l{color:var(--text-muted,#9a9aa0)} .df-v{font-variant-numeric:tabular-nums;text-align:right}
    .df-table td,.df-table th{padding:3px 6px;border-bottom:1px solid var(--border,#333);text-align:right}
    .df-table th:first-child,.df-table td:first-child{text-align:left}
    #dfPalette .warn{color:var(--warning,#f0703a)}`;

  function mount() {
    const anchor = $('cleanExportModeSelect') && $('cleanExportModeSelect').closest('.form-group');
    if (!anchor || $('dfPalette')) return false;
    const style = document.createElement('style'); style.textContent = CSS; document.head.appendChild(style);
    anchor.insertAdjacentHTML('beforebegin', html());
    for (const s of SLIDERS) {
      const el = $(s.id);
      el.addEventListener('input', () => { show(s); markStale(); });
    }
    $('dfPerChannel').addEventListener('change', markStale);
    for (const id of ['cleanStage1Checkbox', 'cleanStage2Checkbox', 'cleanStage4Checkbox', 'cleanStage3Checkbox', 'cleanDenoiseCheckbox', 'cleanExportModeSelect']) {
      const el = $(id); if (el) el.addEventListener('change', markStale);
    }
    $('dfApplyAuto').addEventListener('click', () => { if (P.auto) applySettings(P.auto, 'z pomiaru'); });
    $('dfPreview').addEventListener('click', runPreview);
    $('dfPlay').addEventListener('click', () => (P.playing ? stop() : play()));
    $('dfModeWipe').addEventListener('click', () => { P.mode = 'wipe'; draw(); });
    $('dfModeA').addEventListener('click', () => { P.mode = 'a'; draw(); });
    $('dfModeB').addEventListener('click', () => { P.mode = 'b'; draw(); });
    $('dfWipe').addEventListener('input', () => { P.wipe = $('dfWipe').value / 100; P.mode = 'wipe'; draw(); });
    // drag the wipe line on the image itself (click or drag left / right), like the viewer's wipe
    const cv = $('dfCanvas');
    cv.style.cursor = 'ew-resize'; cv.style.touchAction = 'none';
    const wipeAt = (e) => {
      const r = cv.getBoundingClientRect();
      P.wipe = Math.min(1, Math.max(0, (e.clientX - r.left) / Math.max(1, r.width)));
      P.mode = 'wipe'; $('dfWipe').value = Math.round(P.wipe * 100); draw();
    };
    cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); wipeAt(e); });
    cv.addEventListener('pointermove', (e) => { if (cv.hasPointerCapture(e.pointerId)) wipeAt(e); });
    cv.addEventListener('pointerup', (e) => { if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId); });
    for (const s of SLIDERS) show(s);
    return true;
  }

  function show(s) {
    const v = $(s.id).value;
    $(s.id + 'V').textContent = `${s.pre || ''}${v}${s.unit}`;
  }

  function markStale() {
    if (P.preview) { P.stale = true; $('dfStale').style.display = ''; }
    if (P.auto) $('dfSource').textContent = settingsEqual(current(), P.auto) ? 'ustawienia z pomiaru' : 'ustawienia ręczne';
  }

  function current() {
    const g = (id) => $(id) && $(id).checked;
    return {
      strength: +$('dfStrength').value, window: +$('dfWindow').value, smoothness_px: +$('dfSmooth').value,
      texture_radius: +$('dfTexR').value, per_channel: $('dfPerChannel').checked,
      grid: !!g('cleanStage2Checkbox'), whole_frame: !!g('cleanStage4Checkbox'), texture: !!g('cleanStage3Checkbox'),
    };
  }

  function settingsEqual(a, b) {
    // texture / its window only count with an EXR export: with the Nuke Group the option is locked off, so the automat
    // proposing it must not turn the badge into "manual settings" right after the measurement
    const exr = $('cleanExportModeSelect').value !== 'nuke_group';
    const keys = ['strength', 'window', 'smoothness_px', 'per_channel', 'grid', 'whole_frame'].concat(exr ? ['texture', 'texture_radius'] : []);
    return keys.every((k) => a[k] === b[k]);
  }

  // palette -> body of "⚡ Deflicker" (app.js runPlateClean spreads this in; the stage checkboxes stay the master for enable_*)
  function knobs() {
    const c = current();
    return { stage1_iterations: c.strength, stage1_window: c.window, stage1_sigma: c.smoothness_px, stage3_radius: c.texture_radius,
             stage4_luma: !c.per_channel, target_reduction: 2.0 };
  }

  function applySettings(s, label) {
    $('dfStrength').value = s.strength; $('dfWindow').value = s.window; $('dfSmooth').value = s.smoothness_px; $('dfTexR').value = s.texture_radius;
    $('dfPerChannel').checked = !!s.per_channel;
    const set = (id, v) => { const el = $(id); if (el && !el.disabled) el.checked = !!v; };
    set('cleanStage2Checkbox', s.grid); set('cleanStage4Checkbox', s.whole_frame);
    const exr = $('cleanExportModeSelect').value !== 'nuke_group';
    set('cleanStage3Checkbox', s.texture && exr);
    for (const sl of SLIDERS) show(sl);
    $('dfSource').textContent = label ? `ustawienia ${label}` : 'ustawienia z pomiaru';
    markStale();
  }

  function renderNotes(s) {
    const ul = $('dfNotes'); ul.innerHTML = (s.notes || []).map((n) => `<li>${esc(n)}</li>`).join(''); ul.style.display = s.notes && s.notes.length ? '' : 'none';
    const w = $('dfWarnings'); w.innerHTML = (s.warnings || []).map((x) => `⚠ ${esc(x)}`).join('<br>'); w.style.display = s.warnings && s.warnings.length ? '' : 'none';
    const exr = $('cleanExportModeSelect').value !== 'nuke_group';
    if (s.texture && !exr) {
      const a = s.achievable && s.achievable[s.texture_region];
      ul.insertAdjacentHTML('beforeend', `<li class="warn">Stały wzór faktury w „${esc(s.texture_region)}” zdejmie dopiero zapis EXR z opcją „Migotanie faktury i cieni”${a ? ` (dziś ${a.now} %, do osiągnięcia ≈ ${a.floor} %)` : ''}</li>`);
      ul.style.display = '';
    }
  }

  // ---- "Zmierz migotanie": measurement + automat (replaces the viewer's measurePlateClean)
  async function analyze() {
    const V = window.viewer;
    const plate = $('cleanPlateSelect').value;
    V.btnMeasureClean.disabled = true;
    try {
      const res = await fetch('/api/plate/clean/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate, frame: V.currentFrame, radius: 10, objects: true, eval_w: 1024 }) });
      const api = await readApiResponse(res);
      const d = api.data;
      if (!api.ok || !d.success) { alert(`Błąd pomiaru: ${api.error || 'Nieznany błąd'}`); return; }
      P.measured = d; P.auto = d.settings; P.preview = null; P.stale = false; $('dfStale').style.display = 'none';
      const fl = d.flicker;
      V.cleanMetricsGrid.style.display = 'grid';
      V.metricCleanInit.textContent = `R ${fl.r}% / G ${fl.g}% / B ${fl.b}% (p95: ${fl.p95}%)`;
      const all = d.measure.regions.all;
      // the badge judges the flicker ABOVE what is removable at all (the flow floor), in the worst region – the old
      // spec thresholds (mean > 4 % / p95 > 6 %) called SH010's visible 2.4 % "low"
      const excess = Math.max(...Object.values(d.measure.regions).map((r) => r.metric_pct - r.floor_pct));
      const ex = Math.round(excess * 100) / 100;
      V.cleanBadgeStatus.className = `badge ${excess > 0.5 ? 'badge-warning' : 'badge-success'}`;
      V.cleanBadgeStatus.textContent = excess > 0.5 ? `Jest co usuwać: ${ex} % ponad osiągalne` : `Migotanie bliskie osiągalnego (+${ex} %)`;
      // the "after" numbers belong to the previous full run, not to this measurement: cleared until the next "⚡ Deflicker"
      V.metricCleanFinal.textContent = '–'; V.metricCleanRedPct.textContent = '–'; V.metricCleanNoiseRed.textContent = '–';
      const regs = Object.entries(d.measure.regions).filter(([k]) => k !== 'all')
        .map(([k, r]) => `${regionName(k)}: ${r.metric_pct} % (do osiągnięcia ≈ ${r.floor_pct} %)`).join(' · ');
      V.cleanSummaryText.textContent = `Pomiar F${d.window[0]}–F${d.window[1]} (${d.n_frames} kl., ${d.eval_w} px): cały kadr ${all.metric_pct} % migotania, do osiągnięcia ≈ ${all.floor_pct} % ${regs ? '· ' + regs : ''}`.trim() + '.';
      applySettings(d.settings, 'z pomiaru');
      renderNotes(d.settings);
      $('dfApplyAuto').disabled = false; $('dfPreview').disabled = false;
      V.btnRunClean.disabled = false;
    } catch (e) {
      alert(`Błąd połączenia z serwerem: ${e}`);
    } finally {
      V.btnMeasureClean.disabled = false;
    }
  }

  // ---- "👁 Podgląd klatek"
  async function runPreview() {
    const V = window.viewer;
    const plate = $('cleanPlateSelect').value;
    const frame = V.currentFrame;
    const win = P.measured ? P.measured.window : null;                 // the same frames as the measurement
    $('dfPreview').disabled = true;
    try {
      const res = await fetch('/api/plate/clean/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate, frame, radius: 10, window: win, objects: true, eval_w: 1024, settings: current(), export_mode: $('cleanExportModeSelect').value }) });
      const api = await readApiResponse(res);
      const d = api.data;
      if (!api.ok || !d.success) { alert(`Błąd podglądu: ${api.error || 'Nieznany błąd'}`); return; }
      P.preview = d; P.stale = false; $('dfStale').style.display = 'none';
      showPreview(d);
    } catch (e) {
      alert(`Błąd połączenia z serwerem: ${e}`);
    } finally {
      $('dfPreview').disabled = false;
    }
  }

  function showPreview(d) {
    stop();
    $('dfPreviewBox').style.display = '';
    $('dfPvInfo').textContent = `F${d.window[0]}–F${d.window[1]} · ${d.eval_w} px · ${d.seconds} s · ${d.stages_applied.map(stageName).join(' + ')}`;
    $('dfMapB').src = d.files.map_before; $('dfMapA').src = d.files.map_after;
    const rows = Object.entries(d.regions).map(([k, r]) =>
      `<tr><td>${esc(regionName(k))}</td><td>${r.before} %</td><td>${r.after} %</td><td>${r.floor} %</td><td>${r.removed_pct} %</td></tr>`).join('');
    $('dfTable').innerHTML = `<tr><th>Obszar</th><th>Przed</th><th>Po deflickerze</th><th>Do osiągnięcia</th><th>Usunięto (z możliwego)</th></tr>${rows}`;
    $('dfPvNotes').innerHTML = (d.notes || []).map((n) => `<li>${esc(n)}</li>`).join('');
    P.frames = d.files.frames.map(([b, a]) => { const ib = new Image(); ib.src = b; const ia = new Image(); ia.src = a; return [ib, ia]; });
    P.idx = 0; P.mode = 'wipe'; P.wipe = 0.5; $('dfWipe').value = 50;
    const first = P.frames[0][0];
    first.onload = () => { sizeCanvas(first); draw(); };
    if (first.complete) { sizeCanvas(first); draw(); }
    play();
  }

  function regionName(k) {
    return { all: 'cały kadr', inner: 'bez brzegu 6 %' }[k] || k;
  }
  function stageName(s) {
    // the same names as the report in app.js (all present in en.json)
    return { stage1_local_mc: 'jasność obszarów', stage2_tiles_mc: 'kratka 9×6', stage4_global_mc: 'cała klatka', stage3_detail_mc: 'migotanie faktury i cieni', denoise_mc: 'odszumianie w czasie' }[s] || s;
  }

  function sizeCanvas(img) {
    const c = $('dfCanvas');
    const w = Math.min(img.naturalWidth || 1024, 1024);
    c.width = w; c.height = Math.round(w * (img.naturalHeight || 683) / (img.naturalWidth || 1024));
  }

  function draw() {
    const c = $('dfCanvas'); if (!c || !P.frames.length) return;
    const ctx = c.getContext('2d');
    const [a, b] = P.frames[P.idx];
    if (!a.complete || !b.complete) return;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    if (P.mode === 'a') { ctx.drawImage(a, 0, 0, W, H); }
    else if (P.mode === 'b') { ctx.drawImage(b, 0, 0, W, H); }
    else {
      const x = Math.round(W * P.wipe);
      ctx.drawImage(a, 0, 0, W, H);
      ctx.save(); ctx.beginPath(); ctx.rect(x, 0, W - x, H); ctx.clip(); ctx.drawImage(b, 0, 0, W, H); ctx.restore();
      ctx.fillStyle = 'rgba(232,163,61,0.9)'; ctx.fillRect(x - 1, 0, 2, H);
      ctx.font = '13px IBM Plex Sans, sans-serif'; ctx.fillStyle = '#fff';
      ctx.fillText('przed', 8, 18); ctx.fillText('po', W - 28, 18);
    }
    const fr = P.preview ? P.preview.frames[P.idx] : P.idx;
    $('dfFrameNo').textContent = `F${fr} (${P.idx + 1}/${P.frames.length})`;
  }

  function play() {
    if (!P.frames.length) return;
    P.playing = true; $('dfPlay').textContent = '⏸ Pauza';
    clearInterval(P.timer);
    P.timer = setInterval(() => { P.idx = (P.idx + 1) % P.frames.length; draw(); }, 1000 / 12);
  }
  function stop() { P.playing = false; clearInterval(P.timer); P.timer = null; const b = $('dfPlay'); if (b) b.textContent = '▶ Odtwarzaj'; }

  function init() {
    const V = window.viewer;
    if (!V || !$('cleanExportModeSelect')) { setTimeout(init, 60); return; }
    if (!mount()) return;
    V.measurePlateClean = analyze;                         // the button is bound as () => this.measurePlateClean()
    const close = $('closeCleanBtn'); if (close) close.addEventListener('click', stop);
    window.deflickerPalette = { knobs, current, applySettings, state: P };
  }
  init();
})();
