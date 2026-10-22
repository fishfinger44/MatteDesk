// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/stage0.js – stage 0 "Plate'y" (shot level): plates & proxies, agreement of clear plate / backlit with beauty
// (compare on the canvas: A/B, wipe, difference, contours; measure; preview "po dopasowaniu" without writing files;
// then the user decides to write <plate>_align), cleaning (deflicker / denoise – any plate), camera 2D, depth.
// Alignment applies only to the auxiliary plates (that is what it is for); nothing is ever cut out of them.
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  const AUX = ['cleanplate', 'backlit'];
  const LABEL = { beauty: 'Beauty (X1)', cleanplate: 'Clear plate (tło)', clean_plate: 'Clear plate (tło)', backlit: 'Backlit (X2)', depth: 'Głębia' };
  const label = (n) => LABEL[n] || (n.endsWith('_align') ? `${LABEL[n.slice(0, -6)] || n.slice(0, -6)} · wyrównany` : n);
  let info = null;          // /api/project/stages
  let aux = null;           // auxiliary plate shown in the compare section
  let report = null;        // Align_<aux>.json (report dict) or null

  const pane = window.MD.el(`<div class="stage-pane" data-stage="0">
    <div class="pane-title">0 · Plate'y <span class="muted">(ujęcie)</span></div>
    <div class="pane-section">
      <div class="sec-title">Plate'y w projekcie</div>
      <div id="p0Plates"></div>
      <div class="row"><button class="btn btn-secondary btn-sm" id="p0Manage">⚙ Zarządzaj plate'ami</button><button class="btn btn-secondary btn-sm" id="p0Proxy" title="Proxy 1920 px dla oglądanego plate'u">⚡ Proxy (ten plate)</button></div>
    </div>
    <div class="pane-section" id="p0QaSec">
      <div class="sec-title">Kontrola klatek <span class="muted">(uszkodzone klatki – naprawa raz, potem wszędzie plate naprawiony)</span></div>
      <div class="row"><label>Plate<select id="p0QaPlate" class="form-input input-sm"></select></label>
        <button class="btn btn-primary btn-sm" id="p0QaRun">🔍 Sprawdź klatki</button>
        <button class="btn btn-secondary btn-sm" id="p0QaSource" style="display:none">↺ Zmień naprawę</button></div>
      <div class="status-card"><span class="badge badge-info" id="p0QaBadge">Nie sprawdzono</span><div id="p0QaText" class="muted"></div></div>
      <canvas id="p0QaChart" class="p0qa-chart" height="64" style="display:none"></canvas>
      <div id="p0QaList" class="p0qa-list"></div>
      <div id="p0QaPrev" class="p0qa-prev" style="display:none">
        <figure><img id="p0QaBefore" alt=""><figcaption id="p0QaBeforeL">przed</figcaption></figure>
        <figure><img id="p0QaAfter" alt=""><figcaption id="p0QaAfterL">po</figcaption></figure>
      </div>
      <div class="row"><button class="btn btn-secondary btn-sm" id="p0QaAccept" style="display:none">✔ Przyjmij propozycje</button>
        <button class="btn btn-warning btn-sm" id="p0QaWrite" disabled>💾 Zapisz naprawiony plate (_fix)</button></div>
    </div>
    <div class="pane-section" id="p0AlignSec">
      <div class="sec-title">Zgodność z beauty <span class="muted">(clear plate / backlit)</span></div>
      <div class="muted">Z tych plate'ów nic nie wycinamy – służą jako miarka krawędzi (clear plate) i źródło sylwetki (backlit). Motion control potrafi nie trzymać ruchu między przejazdami: sprawdź, czy plate pasuje do beauty, a jeśli nie – „Zmierz dopasowanie” i obejrzyj wynik zanim zapiszesz.</div>
      <div class="muted">Plate'y i przyciski porównań (Plate / A/B / Wipe / Różnica / Kontury) są w rzędzie pod górnym paskiem, pod wyborem plate'ów (lewy ↔ prawy). Pomiar dotyczy clear plate'u / backlitu z pary na górze.</div>
      <div class="status-card" id="p0AlignCard"><span class="badge badge-info" id="p0AlignBadge">Nie mierzono</span><div id="p0AlignText" class="muted"></div></div>
      <label class="chk" id="p0AlignedL" title="Proxy plate'u przekształcone pomiarem – podgląd wyniku „Skoryguj plate” bez zapisu na dysk"><input type="checkbox" id="p0Aligned"> Pokaż po dopasowaniu</label>
      <div class="row">
        <button class="btn btn-primary btn-sm" id="p0Measure">📐 Zmierz dopasowanie</button>
        <select id="p0Device" class="input-sm" title="Gdzie liczyć wyrównanie: Auto = GPU (Python ComfyUI), gdy działa i ComfyUI nic nie liczy, inaczej CPU (wszystkie rdzenie)">
          <option value="">Auto</option><option value="gpu">GPU</option><option value="cpu">CPU</option>
        </select>
        <button class="btn btn-warning btn-sm" id="p0Apply" disabled title="Zapisuje wyrównaną sekwencję <plate>_align obok oryginału i wpisuje ją do project.json">⚡ Skoryguj plate (_align)</button>
      </div>
      <div class="row"><button class="btn btn-secondary btn-sm" id="p0Nk" disabled title="Group Align_<plate> do Nuke'a (Transform / CornerPin / STMap z siatki wg metody)">💾 Zapisz .nk</button></div>
      <div class="progress-bar-container" id="p0AlignProg" style="display:none"><div class="progress-bar-fill" id="p0AlignFill" style="width:0%"></div><span class="progress-text" id="p0AlignTxt"></span></div>
    </div>
    <div class="pane-section">
      <div class="sec-title">Czyszczenie plate'u <span class="muted">(deflicker / odszumianie – każdy plate)</span></div>
      <div id="p0Clean"></div>
      <button class="btn btn-secondary btn-sm" id="p0CleanOpen">✨ Czyszczenie…</button>
    </div>
    <div class="pane-section">
      <div class="sec-title">Plate do maskowania <span class="muted">(wejście SAM 3.1 / MatAnyone2 – projekt)</span></div>
      <div class="row"><label>Odszumianie plate'u do maskowania<select id="p0Denoise" class="form-input"></select></label></div>
      <div id="p0DenoiseInfo" class="muted"></div>
    </div>
    <div class="pane-section">
      <div class="sec-title">Kamera 2D</div>
      <div id="p0Camera" class="muted"></div>
      <button class="btn btn-secondary btn-sm" id="p0CameraOpen">📹 Ruch kamery…</button>
    </div>
    <div class="pane-section">
      <div class="sec-title">Głębia</div>
      <div id="p0Depth" class="muted"></div>
      <button class="btn btn-secondary btn-sm" id="p0DepthOpen">🌗 Głębia…</button>
    </div>
    <div class="stage-status" id="st0Status"></div>
    <button class="btn btn-primary next-btn" id="st0Next">Dalej → 1. Wykryj</button>
  </div>`);

  async function refresh() {
    try { info = await window.MD.api('GET', '/api/project/stages'); } catch (e) { info = null; }
    render();
    window.MD.renderStageBar();
  }
  function render() {
    if (!info) { $('p0Plates').innerHTML = '<div class="muted">Otwórz projekt.</div>'; return; }
    const cur = V().activePlate || 'beauty';
    $('p0Plates').innerHTML = info.plates.map((p) => `
      <div class="row-item plate-row ${p.name === cur ? 'cur' : ''}" data-p="${p.name}" title="Kliknij: pokaż ten plate">
        <span class="plate-badge plate-badge-${p.name.replace('_align', '')}">${p.name}</span>
        <span class="plate-name">${label(p.name)}</span>
        <span class="pill ${p.exists ? 'ok' : 'warn'}">${p.exists ? `${p.frame_count} kl.` : 'brak plików'}</span>
        <span class="pill ${p.proxy_count ? 'ok' : ''}" title="proxy 1920 px">${p.proxy_count ? `proxy ${p.proxy_count}` : 'bez proxy'}</span>
        ${p.mismatch ? `<span class="pill warn" title="Inna rozdzielczość niż beauty – ten plate nie jest używany w żadnym etapie (pewnie z innego ujęcia). Podmień go albo usuń.">⚠ ${p.mismatch}</span>` : ''}
      </div>`).join('');
    $('p0Plates').querySelectorAll('.plate-row').forEach((r) => r.addEventListener('click', () => V().setActivePlate(r.dataset.p)));

    // alignment section: only for the auxiliary plates present
    const auxes = info.plates.filter((p) => AUX.includes(p.name)).map((p) => p.name);
    $('p0AlignSec').style.display = auxes.length ? '' : 'none';
    if (auxes.length) {
      // the plate to measure = the auxiliary one of the header pair (left first), else the last used / first available
      const refP = V().compare.ref;
      if (AUX.includes(cur)) aux = cur; else if (AUX.includes(refP)) aux = refP; else if (!aux || !auxes.includes(aux)) aux = auxes[0];
      renderAlign();
    }
    // cleaning per plate
    $('p0Clean').innerHTML = info.plates.map((p) => {
      const c = info.clean[p.name] || {};
      return `<div class="row-item"><span>${label(p.name)}</span><span class="pill ${c.done ? 'ok' : ''}">${c.done ? `−${c.reduction_pct}% migotania${c.noise_reduction_pct ? `, −${c.noise_reduction_pct}% szumu` : ''}${c.nk ? ' · .nk' : ''}` : 'nie czyszczono'}</span></div>`;
    }).join('');
    renderDenoise();
    $('p0Camera').textContent = info.camera.done ? `Policzono: ${info.camera.motion_type}, reszta ${info.camera.residual_median_px} px` : 'Nie policzono (opcjonalne – tło nieruchome przy QA i strefach).';
    $('p0Depth').textContent = info.depth.done ? 'Policzono (podpowiedź kolejności obiektów).' : 'Nie policzono (opcjonalne).';
    const on = AUX.includes(cur);
    V().syncCompareUi();
    $('st0Status').textContent = info.plate_qa_block ? `⚠ ${info.plate_qa_block}` : (on ? `Oglądasz „${cur}”. Narzędzia obiektów (etapy 1–5) są wyłączone – wycinamy tylko z beauty. Przytrzymaj C = beauty.` : '');
    loadQa();
  }
  // plate-to-mask denoise (project.json mask_plate_denoise): auto / on / off, shared by the core and the final pass
  function renderDenoise() {
    const d = info && info.mask_plate_denoise, sel = $('p0Denoise');
    if (!d || !sel) return;
    const onoff = (v) => (v ? 'wł.' : 'wył.');
    const autoOn = d.auto_resolves_to === 'on';
    sel.innerHTML = `<option value="auto">auto (→ ${onoff(autoOn)})</option><option value="on">wł.</option><option value="off">wył.</option>`;
    sel.value = d.setting;
    sel.disabled = !d.global_enabled;
    $('p0DenoiseInfo').textContent = d.global_enabled
      ? `Teraz: ${onoff(d.effective)} (${d.reason}). Zmiana = obiekty z rdzeniem liczonym inaczej do przeliczenia.`
      : `Wyłączone globalnie (defaults.json → mplate.enabled = false).`;
  }

  function renderAlign() {
    const a = info && info.align[aux];
    const badge = $('p0AlignBadge'), txt = $('p0AlignText');
    report = null;
    if (!a || !a.measured) {
      badge.className = 'badge badge-info'; badge.textContent = 'Nie mierzono';
      txt.textContent = 'Porównaj wzrokowo (Wipe / Różnica), potem „Zmierz dopasowanie” – pomiar tylko na statycznych obszarach, model od przesunięcia do dopasowania lokalnego (siatka przesunięć → STMap w Nuke).';
      $('p0Apply').disabled = true; $('p0Nk').disabled = true; $('p0Aligned').disabled = true; $('p0Aligned').checked = false;
      if (V().alignedPreview) V().setAlignedPreview(false);
      return;
    }
    report = a;
    if (a.stale) { badge.className = 'badge badge-warning'; badge.textContent = 'Zmierz ponownie'; }
    else if (a.applied) { badge.className = 'badge badge-success'; badge.textContent = 'Skorygowany (_align)'; }
    else if (a.needs_correction) { badge.className = 'badge badge-warning'; badge.textContent = 'Przesunięty – korekta zalecana'; }
    else { badge.className = 'badge badge-success'; badge.textContent = 'Zgodny'; }
    const over = (a.frames_over || []);
    const dev = a.device ? ` · liczone na <b>${window.MD.esc(a.device.replace('cuda', 'GPU').replace('cpu', 'CPU'))}</b>` : '';
    txt.innerHTML = (a.stale ? `<span class="p0-stale">⚠ ${window.MD.esc(a.stale)}</span><br>` : '') +
      `${window.MD.esc(a.recommendation || '')}<br>metoda <b>${a.method}</b> · reszta mediana <b>${(+a.residual_median).toFixed(2)} px</b>, max <b>${(+a.residual_max).toFixed(2)} px</b> · F${a.frame_range[0]}–${a.frame_range[1]}${dev}` +
      (a.device_note ? `<br>${window.MD.esc(a.device_note)}` : '') +
      (over.length ? `<br>klatki > 0.3 px: ${over.slice(0, 12).join(', ')}${over.length > 12 ? '…' : ''}` : '');
    $('p0Apply').disabled = !a.needs_correction || a.applied;
    $('p0Nk').disabled = false;
    $('p0Aligned').disabled = false;
    $('p0Aligned').checked = !!V().alignedPreview;
  }

  // ---------------------------------------------------------------- Z26: kontrola klatek (plate QA) + naprawa _fix
  const FLAG = { missing: 'brak pliku', unreadable: 'nieczytelna', nan: 'NaN / Inf', size: 'zły rozmiar', exposure_jump: 'skok ekspozycji',
    image_jump: 'skok obrazu', duplicate: 'duplikat', overexposed: 'prześwietlona' };
  const METHOD = { mean: 'średnia sąsiadów', flow: 'przepływ (duży ruch)', gain: 'wzmocnienie (ekspozycja)', keep: 'zostaw bez zmian' };
  const GATE = { none: ['badge-info', 'Nie sprawdzono'], clean: ['badge-success', 'Bez uszkodzeń'], kept: ['badge-success', 'Flagi zostawione świadomie'],
    fixed: ['badge-success', 'Naprawiony (_fix)'], undecided: ['badge-warning', 'Klatki bez decyzji'], pending: ['badge-warning', 'Zapisz naprawę'],
    other: ['badge-warning', 'Raport innego pliku – sprawdź ponownie'] };
  let qaPlate = null;       // plate shown in the QA section
  let qaRep = null;         // GET /api/plate/qa/<plate> (exists, frames, flagged, suggested, decisions, gate, fix)
  let qaSel = null;         // selected flagged frame (preview)
  const qaPlates = () => (info ? info.plates.map((p) => p.name).filter((n) => n !== 'bg') : []);
  const flagText = (r) => (r.flags || []).map((f) => (FLAG[f] || f) +
    (f === 'exposure_jump' && r.ev_dev != null ? ` ${r.ev_dev > 0 ? '+' : ''}${(+r.ev_dev).toFixed(2)} EV` : '') +
    (f === 'image_jump' && r.shift_dev_px ? ` (odchylenie ruchu ${(+r.shift_dev_px).toFixed(0)} px)` : '')).join(', ');

  // anomaly per frame, 1 = threshold of its measure (the chart): exposure, interpolation error, one-frame shift
  function qaScore(r, thr) {
    if ((r.flags || []).some((f) => ['missing', 'unreadable', 'nan', 'size', 'duplicate'].includes(f))) return 3;
    const ev = r.ev_dev != null ? Math.abs(r.ev_dev) / (thr.ev_jump || 0.5) : 0;
    const it = r.interp_err != null ? Math.min(r.interp_err / (thr.interp_min || 2e-3), r.interp_err / Math.max(1e-12, (thr.interp_ratio || 3) * (r.neighbour_err || 0))) : 0;
    const sh = r.shift_dev_px != null ? r.shift_dev_px / (thr.shift_px || 8) : 0;
    return Math.max(ev, it, sh);
  }
  function drawQaChart() {
    const c = $('p0QaChart');
    if (!qaRep || !qaRep.exists) { c.style.display = 'none'; return; }
    c.style.display = '';
    const w = c.clientWidth || 300, h = c.height; c.width = w;
    const ctx = c.getContext('2d');
    const fr = Object.keys(qaRep.frames).map(Number).sort((a, b) => a - b);
    const thr = qaRep.thresholds || {}, flagged = new Set(qaRep.flagged || []);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(18,18,20,.6)'; ctx.fillRect(0, 0, w, h);
    const bw = w / Math.max(1, fr.length), top = 3;
    fr.forEach((f, i) => {
      const s = Math.min(3, qaScore(qaRep.frames[f], thr)), bh = Math.max(1, (h - top) * s / 3);
      ctx.fillStyle = flagged.has(f) ? '#ef4444' : (s > 1 ? '#f0703a' : '#6e6e75');
      ctx.fillRect(i * bw, h - bh, Math.max(1, bw - (bw > 3 ? 1 : 0)), bh);
    });
    ctx.strokeStyle = 'rgba(253,230,138,.6)'; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, h - (h - top) / 3); ctx.lineTo(w, h - (h - top) / 3); ctx.stroke();
    ctx.setLineDash([]);
    const ci = fr.indexOf(V().currentFrame);
    if (ci >= 0) { ctx.fillStyle = 'rgba(242,187,102,.9)'; ctx.fillRect(ci * bw + bw / 2 - 1, 0, 2, h); }
    c._frames = fr; c._bw = bw;
  }
  async function loadQa() {
    const names = qaPlates();
    const sel = $('p0QaPlate');
    if (!names.length) { $('p0QaSec').style.display = 'none'; return; }
    $('p0QaSec').style.display = '';
    if (!qaPlate || !names.includes(qaPlate)) qaPlate = names.includes(V().activePlate) ? V().activePlate : (names.includes('beauty') ? 'beauty' : names[0]);
    sel.innerHTML = names.map((n) => {
      const g = (info.plate_qa || {})[n] || { state: 'none' };
      const mark = g.blocking ? ' ⚠' : (['clean', 'kept', 'fixed'].includes(g.state) ? ' ✔' : '');
      return `<option value="${n}">${window.MD.esc(label(n))}${mark}</option>`;
    }).join('');
    sel.value = qaPlate;
    try { qaRep = await window.MD.api('GET', `/api/plate/qa/${encodeURIComponent(qaPlate)}`); } catch (e) { qaRep = null; }
    if (qaRep && qaRep.exists && qaSel != null && !(qaRep.flagged || []).includes(qaSel)) qaSel = null;
    renderQa();
  }
  function renderQa() {
    const g = (qaRep && qaRep.gate) || ((info && info.plate_qa) || {})[qaPlate] || { state: 'none' };
    const [cls, txt] = GATE[g.state] || GATE.none;
    const badge = $('p0QaBadge');
    badge.className = `badge ${cls}`;
    badge.textContent = g.state === 'undecided' ? `${g.undecided.length} kl. bez decyzji` : txt;
    const rep = qaRep && qaRep.exists ? qaRep : null;
    const fix = qaRep && qaRep.fix;
    let t = '';
    if (!rep) t = 'Sprawdza każdą klatkę: brak pliku / błąd odczytu / NaN, skok ekspozycji, skok obrazu (uszkodzona albo przesunięta klatka), duplikat. Wynik → decyzja per klatka → zapis naprawionego plate\'u „_fix” obok oryginału.';
    else {
      t = `${rep.n || Object.keys(rep.frames).length} kl. sprawdzone w ${rep.seconds} s · oflagowane: <b>${(rep.flagged || []).length}</b>`;
      if (g.state === 'other') t += '<br>Plate wskazuje teraz inny plik niż sprawdzony – „Sprawdź klatki” ponownie.';
      if (g.state === 'undecided') t += '<br>Etap 1 zablokowany, dopóki każda oflagowana klatka nie ma decyzji (naprawa albo „zostaw”).';
      if (g.state === 'pending') t += '<br>Decyzje gotowe – „Zapisz naprawiony plate (_fix)” (etap 1 czeka na zapis).';
      if (g.state === 'fixed' && fix) t += `<br>W użyciu: <code>${window.MD.esc(fix.pattern)}</code>`;
    }
    $('p0QaText').innerHTML = t;
    $('p0QaSource').style.display = fix && !(g.fixed) ? '' : 'none';
    $('p0QaSource').title = fix ? `Sprawdź ponownie oryginał (${fix.source}), żeby zmienić naprawę` : '';
    drawQaChart();
    const list = $('p0QaList');
    const fl = rep ? rep.flagged || [] : [];
    const dec = (g.decisions) || (rep && rep.decisions) || {};
    list.innerHTML = fl.map((f) => {
      const r = rep.frames[String(f)] || {}, d = dec[String(f)], sug = (rep.suggested || {})[String(f)] || 'keep';
      const val = d || sug;
      return `<div class="row-item p0qa-row ${qaSel === f ? 'cur' : ''} ${d ? 'decided' : 'undecided'}" data-f="${f}" title="Kliknij: skok do klatki ${f} i podgląd przed / po">
        <span class="p0qa-f">F${f}</span><span class="p0qa-why">${window.MD.esc(flagText(r))}</span>
        <select class="input-sm p0qa-m" data-f="${f}">${Object.entries(METHOD).map(([k, v]) => `<option value="${k}" ${k === val ? 'selected' : ''}>${v}${k === sug ? ' · propozycja' : ''}</option>`).join('')}</select>
        ${d ? '<span class="pill ok">✔</span>' : `<button class="btn btn-secondary btn-xs p0qa-ok" data-f="${f}" title="Zatwierdź wybraną metodę dla tej klatki">✔</button>`}
      </div>`;
    }).join('');
    list.querySelectorAll('.p0qa-row').forEach((row) => row.addEventListener('click', (e) => {
      if (e.target.closest('select, button')) return;
      selectFlag(+row.dataset.f, true);
    }));
    list.querySelectorAll('.p0qa-m').forEach((s) => s.addEventListener('change', () => { decide({ [s.dataset.f]: s.value }); selectFlag(+s.dataset.f, false); }));
    list.querySelectorAll('.p0qa-ok').forEach((b) => b.addEventListener('click', () => {
      const s = list.querySelector(`.p0qa-m[data-f="${b.dataset.f}"]`); decide({ [b.dataset.f]: s.value });
    }));
    const undec = fl.filter((f) => !dec[String(f)]);
    $('p0QaAccept').style.display = undec.length ? '' : 'none';
    $('p0QaAccept').textContent = `✔ Przyjmij propozycje (${undec.length})`;
    const w = $('p0QaWrite');
    w.disabled = g.state !== 'pending';
    w.textContent = g.state === 'pending' && g.fixed && !Object.keys(g.repairs || {}).length ? '↺ Wróć do oryginału' : '💾 Zapisz naprawiony plate (_fix)';
    renderPreview();
  }
  function selectFlag(f, jump) {
    qaSel = f;
    $('p0QaList').querySelectorAll('.p0qa-row').forEach((r) => r.classList.toggle('cur', +r.dataset.f === f));
    if (jump) {
      if (V().activePlate !== qaPlate) V().setActivePlate(qaPlate);
      V().goToFrame(f);
    }
    renderPreview();
  }
  function renderPreview() {
    const box = $('p0QaPrev');
    if (qaSel == null || !qaRep || !qaRep.exists) { box.style.display = 'none'; return; }
    const s = $('p0QaList').querySelector(`.p0qa-m[data-f="${qaSel}"]`);
    const m = s ? s.value : 'keep';
    const u = (meth) => `/api/plate/qa/${encodeURIComponent(qaPlate)}/preview/${qaSel}?method=${meth}&t=${Date.now()}`;
    box.style.display = '';
    const set = (img, cap, meth, txt) => {
      cap.textContent = `${txt} – ładowanie…`;
      img.onload = () => { cap.textContent = txt; };
      img.onerror = () => { cap.textContent = `${txt} – ${meth === 'orig' ? 'brak pliku' : 'nie da się policzyć (brak sąsiadów?)'}`; img.removeAttribute('src'); };
      img.src = u(meth);
    };
    set($('p0QaBefore'), $('p0QaBeforeL'), 'orig', `przed · F${qaSel}`);
    set($('p0QaAfter'), $('p0QaAfterL'), m === 'keep' ? 'orig' : m, `po · ${METHOD[m]}`);
  }
  async function decide(d) {
    try {
      await window.MD.api('PUT', `/api/plate/qa/${encodeURIComponent(qaPlate)}/decisions`, { decisions: d });
    } catch (e) { window.MD.toast(`Decyzja: ${e.message}`, 6000); }
    await refresh();
    if (window.MD.loadStages) window.MD.loadStages().catch(() => {});
  }
  async function runQa(source) {
    const b = $('p0QaRun'); b.disabled = true;
    const n = (info && info.n_frames) || 0, prev = qaRep && qaRep.exists ? qaRep.seconds : null;
    $('p0QaBadge').className = 'badge badge-info';
    $('p0QaBadge').textContent = `Sprawdzanie… (≈ ${prev ? Math.round(prev) : Math.max(5, Math.round(n * 0.3))} s)`;
    try {
      const r = await window.MD.api('POST', '/api/plate/qa', { plate: qaPlate, source: !!source });
      window.MD.toast(r.flagged.length ? `„${qaPlate}”: oflagowane klatki ${r.flagged.slice(0, 10).join(', ')}${r.flagged.length > 10 ? '…' : ''} – wybierz decyzję` : `„${qaPlate}”: bez uszkodzeń (${r.n} kl., ${r.seconds} s)`, 6000);
      qaSel = r.flagged.length ? r.flagged[0] : null;
    } catch (e) { window.MD.toast(`Kontrola klatek: ${e.message}`, 6000); }
    finally { b.disabled = false; }
    await refresh();
    if (window.MD.loadStages) window.MD.loadStages().catch(() => {});
  }
  async function writeFix() {
    const g = qaRep && qaRep.gate;
    if (!g) return;
    const body = { plate: qaPlate, decisions: g.decisions };
    let plan;
    try { plan = await window.MD.api('POST', '/api/plate/fix', { ...body, confirmed: false }); } catch (e) { window.MD.toast(`Plan naprawy: ${e.message}`, 8000); return; }
    const p = plan.plan || {};
    const mb = (p.bytes_written_estimate || 0) / 1048576;
    const eta = p.eta_s >= 90 ? `${Math.round(p.eta_s / 60)} min` : `${Math.max(1, Math.round(p.eta_s || 0))} s`;
    const rep = Object.entries(p.repair || {}).map(([f, m]) => `F${f} (${METHOD[m] || m})`).join(', ');
    const msg = p.restore
      ? `Wszystkie flagi „${qaPlate}” zostawione bez zmian – wrócić do oryginalnego plate'u?\n${p.restore}\nFolder _fix zostaje na dysku. Obiekty liczone z wersji _fix → do przeliczenia.`
      : `Zapisać naprawiony plate „${qaPlate}”?\n\nNaprawiane klatki: ${rep}\nZapis: ~${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB (${Object.keys(p.repair || {}).length} kl.) + ${p.hardlinks} twardych dowiązań (bez kopiowania)\nCzas: ≈ ${eta}\nFolder: ${p.dir}${p.exists ? '\n(folder istnieje – zapisany wcześniej przez MatteDesk, zostanie nadpisany)' : ''}\n\nOryginał zostaje nietknięty; project.json wskaże wersję _fix (manifest ujęcia bez zmian). Obiekty liczone z tego plate'u → do przeliczenia.`;
    if (!(await V().uiConfirm(msg, { title: p.restore ? 'Wrócić do oryginału?' : 'Zapisać plate _fix?', ok: p.restore ? 'Wróć' : 'Zapisz', tone: 'info' }))) return;
    const w = $('p0QaWrite'); w.disabled = true;
    try {
      const r = await window.MD.api('POST', '/api/plate/fix', { ...body, confirmed: true });
      const st = r.stale_objects || [];
      window.MD.toast((r.restored ? `„${qaPlate}” znów czyta oryginał` : `Zapisano ${Object.keys(r.repaired || {}).length} kl. + ${r.hardlinks} dowiązań: ${r.dir}`) +
        (st.length ? ` · do przeliczenia: ${st.join(', ')}` : ''), 9000);
      await V().loadCurrentProject();
    } catch (e) { window.MD.toast(`Zapis _fix: ${e.message}`, 8000); }
    await refresh();
    if (window.MD.loadStages) window.MD.loadStages().catch(() => {});
  }
  function wireQa() {
    $('p0QaPlate').addEventListener('change', (e) => { qaPlate = e.target.value; qaSel = null; loadQa(); });
    $('p0QaRun').addEventListener('click', () => runQa(false));
    $('p0QaSource').addEventListener('click', () => runQa(true));
    $('p0QaAccept').addEventListener('click', () => {
      const d = {};
      $('p0QaList').querySelectorAll('.p0qa-row.undecided .p0qa-m').forEach((s) => { d[s.dataset.f] = s.value; });
      if (Object.keys(d).length) decide(d);
    });
    $('p0QaWrite').addEventListener('click', writeFix);
    ['p0QaBefore', 'p0QaAfter'].forEach((id) => $(id).addEventListener('click', (e) => { if (e.target.src) window.open(e.target.src.replace(/side=\d+&?/, '') + '&side=2048', '_blank'); }));
    const c = $('p0QaChart');
    const at = (e) => { const r = c.getBoundingClientRect(); const i = Math.floor((e.clientX - r.left) / (c._bw || 1)); return (c._frames || [])[Math.max(0, Math.min((c._frames || []).length - 1, i))]; };
    c.addEventListener('mousemove', (e) => {
      const f = at(e); if (f == null || !qaRep || !qaRep.frames) return;
      const r = qaRep.frames[String(f)] || {};
      c.title = `F${f}${(r.flags || []).length ? ` · ${flagText(r)}` : ''}${r.ev_dev != null ? ` · ΔEV ${(+r.ev_dev).toFixed(2)}` : ''}${r.shift_dev_px != null ? ` · odchylenie ruchu ${(+r.shift_dev_px).toFixed(1)} px` : ''} – kliknij: skok do klatki`;
    });
    c.addEventListener('click', (e) => {
      const f = at(e); if (f == null) return;
      if ((qaRep.flagged || []).includes(f)) selectFlag(f, true);
      else { if (V().activePlate !== qaPlate) V().setActivePlate(qaPlate); V().goToFrame(f); }
    });
    window.addEventListener('md-frame', () => { if (qaRep && qaRep.exists && $('p0QaSec').offsetParent) drawQaChart(); });
    window.addEventListener('resize', () => { if (qaRep && qaRep.exists) drawQaChart(); });
  }

  const prog = (show, pct, t) => { $('p0AlignProg').style.display = show ? 'flex' : 'none'; $('p0AlignFill').style.width = `${pct || 0}%`; $('p0AlignTxt').textContent = t || ''; };

  function wire() {
    $('p0Manage').addEventListener('click', () => V().openPlatesModal());
    $('p0Proxy').addEventListener('click', () => V().triggerProxyGeneration().then(refresh));
    $('p0Aligned').addEventListener('change', (e) => V().setAlignedPreview(e.target.checked));
    try { $('p0Device').value = localStorage.getItem('md.alignDevice') || ''; } catch (e) { /* no storage */ }
    $('p0Device').addEventListener('change', (e) => { try { localStorage.setItem('md.alignDevice', e.target.value); } catch (err) { /* no storage */ } });
    $('p0Measure').addEventListener('click', async () => {
      const b = $('p0Measure'); b.disabled = true; prog(true, 0, 'Pomiar…');
      try {
        const body = { plate: aux, aux_plate_type: aux === 'backlit' ? 'backlit' : 'cleanplate' };
        if ($('p0Device').value) body.device = $('p0Device').value;
        const r = await window.MD.api('POST', '/api/plate/align/measure', body);
        await refresh();
        $('p0Aligned').checked = true; V().setAlignedPreview(true);
        window.MD.toast(r.needs_correction ? 'Plate przesunięty – oglądasz podgląd po dopasowaniu (Wipe / Różnica)' : 'Plate zgodny z beauty – korekta niepotrzebna', 5000);
      } catch (e) { window.MD.toast(`Pomiar: ${e.message}`, 6000); }
      finally { b.disabled = false; prog(false); }
    });
    $('p0Apply').addEventListener('click', async () => {
      if (!(await V().uiConfirm(`Zapisać wyrównaną sekwencję OpenEXR „${aux}_align” obok oryginału?\nOryginał zostaje nietknięty; project.json wskaże wersję wyrównaną (manifest ujęcia bez zmian).`, { ok: 'Zapisz', tone: 'info' }))) return;
      const b = $('p0Apply'); b.disabled = true; prog(true, 5, 'Zapis klatek…');
      try {
        const r = await window.MD.api('POST', '/api/plate/align/apply', { plate: aux, confirmed: true });
        window.MD.toast(`Zapisano ${r.aligned_count} klatek: ${r.aligned_dir}`, 7000);
        await V().loadCurrentProject(); await refresh();
      } catch (e) { window.MD.toast(`Zapis: ${e.message}`, 6000); b.disabled = false; }
      finally { prog(false); }
    });
    $('p0Nk').addEventListener('click', async () => {
      try { const r = await window.MD.api('POST', '/api/plate/align/export_nk', { plate: aux }); window.MD.toast(`Zapisano ${r.nk_path} – File → Import Script albo wklej pod Read plate'u`, 7000); }
      catch (e) { window.MD.toast(`.nk: ${e.message}`, 6000); }
    });
    window.addEventListener('md-align-progress', (e) => { const m = e.detail; prog(true, m.total ? Math.round(100 * m.done / m.total) : 0, `${m.done}/${m.total} · F${m.frame}`); });
    $('p0Denoise').addEventListener('change', async (e) => {
      const sel = e.target; sel.disabled = true;
      try {
        const r = await window.MD.api('PUT', '/api/project/mask_plate_denoise', { setting: sel.value });
        window.MD.toast(`Odszumianie plate'u do maskowania: ${r.effective ? 'wł.' : 'wył.'}` +
          (r.stale_objects && r.stale_objects.length ? ` – do przeliczenia: ${r.stale_objects.join(', ')}` : ''), 6000);
        if (window.MD.loadStages) window.MD.loadStages().catch(() => {});
      } catch (err) { window.MD.toast(`Odszumianie: ${err.message}`, 6000); }
      finally { await refresh(); }
    });
    $('p0CleanOpen').addEventListener('click', () => V().openCleanModal());
    $('p0CameraOpen').addEventListener('click', () => V().openCameraModal());
    $('p0DepthOpen').addEventListener('click', () => window.openDepthModal && window.openDepthModal());
    window.addEventListener('md-plate', (e) => {
      if (info) render(); V().syncCompareUi();
      const p = info && e.detail && info.plates.find((x) => x.name === e.detail.plate);
      if (p && e.detail.plate !== e.detail.prev && info.n_frames && p.proxy_count < info.n_frames) window.MD.toast(`„${p.name}”: proxy ${p.proxy_count}/${info.n_frames} klatek – brakujące liczone na żywo (wolniej); pełne proxy: „Proxy (ten plate)”`, 6000);
    });
    window.addEventListener('md-project', refresh);
    window.addEventListener('md-plates-updated', refresh);
    window.addEventListener('matte-job-finished', refresh);
    document.querySelectorAll('#closeCleanModalBtn, #closeCleanBtn, #closeCameraModalBtn').forEach((b) => b.addEventListener('click', refresh));
  }

  window.MD.stage0State = () => (info && info.beauty_proxy_ok && !info.plate_qa_block ? 'done' : 'todo');
  window.MD.plateBlock = () => (info && info.plate_qa_block) || null;   // Z26: stage 1 locked until every flag is decided
  // floating palette on the left (user request 2026-10-16, like the other stages): plates, frame check, cleaning –
  // the same DOM sections moved out of the stage panel (ids and handlers unchanged)
  const palPos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_plates0_palette') || 'null'); } catch (e) { return null; } },
    set: (p) => { try { if (p) localStorage.setItem('md_plates0_palette', JSON.stringify(p)); else localStorage.removeItem('md_plates0_palette'); } catch (e) { /* storage blocked */ } },
  };
  let palette = null;
  function buildPalette() {
    const w = $('canvasWrapper'); if (!w || palette) return;
    palette = window.MD.el(`<div class="pts-palette p0-palette" id="plates0Palette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Plate'y ujęcia</div></div>`);
    for (const s of [$('p0Plates').closest('.pane-section'), $('p0QaSec'), $('p0Clean').closest('.pane-section')]) if (s) palette.appendChild(s);
    w.appendChild(palette);
    window.MD.makeFloating(palette, w, '.pp-head', palPos);
  }
  function showPalette(stage) {
    if (!palette) return;
    const on = stage === 0 && !!(V() && V().projectLoaded);
    palette.style.display = on ? '' : 'none';
    if (on) window.MD.placeFloating(palette, $('canvasWrapper'), palPos.get());
  }
  window.addEventListener('md-stage', (e) => showPalette(e.detail.stage));
  window.addEventListener('md-project', () => showPalette(window.MD.stage));
  window.addEventListener('DOMContentLoaded', () => {
    window.MD.registerPane(0, { el: pane, show() { refresh(); } });
    buildPalette(); showPalette(window.MD.stage);
    wire();
    wireQa();
    setTimeout(() => { if (V() && V().projectLoaded) refresh(); }, 1000);
  });
})();
