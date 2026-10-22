// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/brush.js – stage CoTracker (user decision 2026-10-15): brush corrections of the SAM3 mask, propagated by
// CoTracker3 points (no SAM3). Two zones of the engine (engine/zones.py, criterion "all", params.source = "brush"):
// "Dodaj" (mode add – pixels connected to the mask) and "Usuń" (mode remove). Painting in a frame = the zone key of that
// frame (PUT keys/{f}); a frame without a key starts from the propagated region (GET region/{f}); „Zmaż” = erase strokes
// in the keys; an empty key ends a correction (the region fades towards it). Points never on parts of the object / on
// objects in front (pipeline.zone_regions), local warp (params.track = local). „▶▶ Zastosuj poprawki” = POST core_fix
// -> the mask 1+2 on the shot (view core_fix). Stage 3 (MatAnyone) and the export source "rdzeń (1+2)" use the same zones.
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  let H = null;                       // hooks from objects.js
  const S = () => H.S;
  const obj = () => H.S.obj;
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const B = { list: [], regionCache: {}, regionBusy: {}, undo: [], drawTimer: null, drawPending: null, drawBusy: false, track: true, obj: null };
  const MODES = { add: { label: 'Dodaj', col: [80, 255, 120], css: '#50ff78' }, remove: { label: 'Usuń', col: [255, 60, 90], css: '#ff3c5a' } };
  const api = (...a) => H.api(...a);
  const status = (t, k) => H.status(t, k);
  const toast = (t, ms) => H.toast(t, ms);
  const curFrame = () => V().currentFrame;
  const inStage = () => !!(H && obj() && !S().auxMode && S().stage === H.ST.BRUSH);
  const isBrush = (z) => z && z.criterion !== 'matte' && z.params && z.params.source === 'brush';
  const zoneOf = (mode) => B.list.find((z) => z.mode === mode) || null;
  function keysOf(z) {
    const out = {};
    for (const [k, v] of Object.entries(z.keys || {})) out[+k] = { polygon: v.polygon || [], strokes: v.strokes || [], bg_strokes: v.bg_strokes || [] };
    out[+z.frame] = { polygon: z.polygon || [], strokes: z.strokes || [], bg_strokes: z.bg_strokes || [] };
    return out;
  }
  const keyFrames = (z) => Object.keys(keysOf(z)).map(Number).sort((a, b) => a - b);

  // ------------------------------------------------------------------------------------------------ panel
  function panelHtml() {
    return `<div class="pane-section">
        <div class="sec-title">Pędzel poprawek maski</div>
        <div class="muted">Popraw maskę SAM 3.1 tam, gdzie się myli: <b class="ok">🖌 Dodaj (G)</b> domalowuje brakujący fragment (tylko piksele połączone z maską),
          <b class="err">🖌 Usuń (R)</b> zamalowuje wyciek na tło, <b>⌫ Zmaż (E)</b> cofa poprawkę w tym miejscu. Kreska = <b>klucz poprawki</b> tej klatki;
          między kluczami i poza nimi CoTracker3 przenosi poprawkę punktami na obiekcie (nigdy na częściach obiektu ani obiektach z przodu).
          Popraw kilka klatek (najlepiej na początku, w środku i na końcu ruchu), potem <b>„▶▶ Zastosuj poprawki”</b> i obejrzyj widok Rdzeń.
          Rozmiar pędzla: [ ] albo Shift + przeciągnij; Ctrl+Z cofa; Delete usuwa klucze poprawek tej klatki.</div>
        <div class="row b2-tools">
          <button type="button" class="tool-btn" data-bt="brush_add" title="Dodaj do maski [G]: domaluj brakujący fragment obiektu (tylko piksele połączone z maską)"><span class="tool-icon">🖌+</span><span class="tool-label">Dodaj G</span></button>
          <button type="button" class="tool-btn" data-bt="brush_remove" title="Usuń z maski [R]: zamaluj wyciek na tło / fragment spoza obiektu"><span class="tool-icon">🖌−</span><span class="tool-label">Usuń R</span></button>
          <button type="button" class="tool-btn" data-bt="brush_erase" title="Zmaż poprawkę [E]: cofa Dodaj / Usuń w tym miejscu tej klatki (maska wraca do SAM 3.1)"><span class="tool-icon">⌫</span><span class="tool-label">Zmaż E</span></button>
        </div>
        <label class="chk" title="Poprawka przenoszona między klatkami punktami CoTracker3 lokalnie (każdy fragment za najbliższymi punktami – nadąża za ruchem i odkształceniem); wyłączone = sztywny kształt jadący za obiektem. Dotyczy nowych poprawek i obu istniejących"><input type="checkbox" id="b2Track" checked title="Śledzenie lokalne poprawek (CoTracker3)"> śledzenie lokalne (CoTracker3)</label>
      </div>
      <div class="pane-section"><div class="sec-title">Poprawki <span class="muted">(romby na osi czasu = klucze; PPM na rombie = usuń)</span></div><div id="b2List" class="zone-list"></div></div>
      <div class="row">
        <button type="button" class="btn btn-ghost btn-xs" id="b2EmptyKey" title="Pusty klucz w tej klatce: poprawki kończą się tutaj (między poprzednim kluczem a tą klatką obszar zanika, dalej brak poprawki)">⌀ Koniec poprawek w tej klatce</button>
        <button type="button" class="btn btn-ghost btn-xs" id="b2DelKey" title="Usuń klucze poprawek tej klatki (obszar w tej klatce znów z propagacji) [Delete]">🗑 Usuń klucze (ta klatka)</button></div>
      <button type="button" class="btn btn-primary" id="b2Apply" title="Policz maskę 1+2 (rdzeń SAM 3.1 + poprawki) na całym ujęciu – widok Rdzeń pokaże wynik w każdej klatce; ta sama maska idzie do MatAnyone2 (etap 3) i do eksportu „rdzeń (1+2)”">▶▶ Zastosuj poprawki na ujęcie</button>
      <div class="muted" id="b2Info"></div>`;
  }
  function renderList() {
    const box = $('b2List'); if (!box) return;
    box.innerHTML = B.list.length ? B.list.map((z) => {
      const m = MODES[z.mode] || MODES.add; const ks = keyFrames(z);
      return `<div class="zone-card ${z.enabled ? '' : 'off'}" data-z="${esc(z.zone_id)}" style="--zc:${m.css}" title="Poprawka ${m.label}: klucze ${ks.map((f) => 'F' + f).join(', ')} – klik = skok do najbliższego klucza">
        <span class="zone-ic">${z.mode === 'add' ? '🖌+' : '🖌−'}</span>
        <span class="zone-name"><b>${m.label}</b><br><span class="muted">${esc(z.zone_id)} · ${(z.params && z.params.track) === 'local' ? 'śledzenie lokalne' : 'sztywna'}</span></span>
        <span class="zone-meta"><span title="liczba kluczy poprawki">🔑 ${ks.length}</span></span>
        <span class="zone-btns">
          <label class="chk" title="Włącz / wyłącz tę poprawkę (wyłączona nie działa w etapach 2, 3 i eksporcie)"><input type="checkbox" data-en="${esc(z.zone_id)}" ${z.enabled ? 'checked' : ''}></label>
          <button class="btn btn-ghost btn-xs" data-del="${esc(z.zone_id)}" title="Usuń całą poprawkę ${m.label} (wszystkie jej klucze)">✕</button></span></div>`;
    }).join('') : '<div class="muted">brak poprawek – maska = rdzeń SAM 3.1 (etap opcjonalny). Pierwsza kreska G / R tworzy poprawkę.</div>';
    box.querySelectorAll('.zone-card').forEach((c) => c.addEventListener('click', (e) => {
      if (e.target.closest('button') || e.target.closest('input')) return;
      const z = B.list.find((x) => x.zone_id === c.dataset.z); if (!z) return;
      const f = curFrame(); const ks = keyFrames(z);
      V().goToFrame(ks.reduce((a, b) => (Math.abs(b - f) < Math.abs(a - f) ? b : a)));
    }));
    box.querySelectorAll('input[data-en]').forEach((c) => c.addEventListener('change', () => patch(c.dataset.en, { enabled: c.checked }, c.checked ? 'włączona' : 'wyłączona')));
    box.querySelectorAll('button[data-del]').forEach((b) => b.addEventListener('click', () => deleteZone(b.dataset.del)));
    renderInfo(); renderLane(); syncTools();
  }
  function renderInfo() {
    const info = $('b2Info'); if (!info || !H) return;
    const st = H.stageState('cotracker');
    const f = V() ? curFrame() : null;
    const here = B.list.filter((z) => keyFrames(z).includes(f)).map((z) => MODES[z.mode].label);
    info.textContent = `${st.state === 'done' ? '✔ ' : st.state === 'stale' ? '↻ ' : ''}${st.why || (S().info && S().info.has && S().info.has.core_fix ? 'Maska 1+2 aktualna – widok Rdzeń pokazuje ją w każdej klatce' : '')}`
      + (f != null ? ` · F${f}: ${here.length ? `klucz ${here.join(' + ')}` : 'bez klucza poprawki'}` : '');
    const ap = $('b2Apply');
    if (ap) { ap.disabled = st.state === 'locked' || document.body.classList.contains('job-busy'); ap.classList.toggle('btn-cta', st.state === 'todo' || st.state === 'stale'); }
  }
  function syncTools() {
    document.querySelectorAll('#b2Panel [data-bt]').forEach((b) => b.classList.toggle('active', S().tool === b.dataset.bt));
  }

  // ------------------------------------------------------------------------------------------------ loading
  async function load() {
    if (!obj()) { B.list = []; renderList(); return; }
    const d = await api('GET', `/api/object/${obj()}/zones`);
    B.list = (d.zones || []).filter(isBrush);
    B.regionCache = {};
    if (B.list.length) B.track = B.list.every((z) => (z.params && z.params.track) === 'local');
    if ($('b2Track')) $('b2Track').checked = B.track;
    renderList();
  }
  function reset() {
    const same = B.obj === obj(); B.obj = obj();
    B.list = []; B.regionCache = {};
    if (!same) B.undo = [];
    clearTimeout(B.drawTimer); B.drawPending = null; renderLane();
  }

  // ------------------------------------------------------------------------------------------------ propagated region
  async function regionFor(z, f, wait) {
    const c = B.regionCache[z.zone_id] || (B.regionCache[z.zone_id] = {});
    if (c[f]) return c[f];
    const k = `${z.zone_id}_${f}`;
    if (B.regionBusy[k]) return wait ? B.regionBusy[k] : null;
    const p = (async () => {
      try { const d = await api('GET', `/api/object/${obj()}/zones/${z.zone_id}/region/${f}`); c[f] = d; if (curFrame() === f) V().render(); return d; }
      catch (e) { if (e.status !== 409) c[f] = { contours: [], error: e.message }; return null; }
      finally { delete B.regionBusy[k]; }
    })();
    B.regionBusy[k] = p;
    return wait ? p : null;
  }
  let frameTimer = null;
  function onFrame() {
    renderLane(); renderInfo();
    clearTimeout(frameTimer);
    if (!inStage()) return;
    const f = curFrame();
    frameTimer = setTimeout(() => {
      if (curFrame() !== f || document.body.classList.contains('job-busy')) return;
      for (const z of B.list) if (z.enabled && !keyFrames(z).includes(f)) regionFor(z, f, false);
    }, 350);
  }
  // a key in a frame without one starts from the propagated region (the largest contour) – the correction continues
  async function baseKey(z, f) {
    const keys = keysOf(z);
    if (keys[f]) return { base: keys[f], created: false };
    let reg = (B.regionCache[z.zone_id] || {})[f];
    if (!reg) { status(`F${f}: obszar poprawki z propagacji (pierwsze wywołanie śledzi CoTrackerem – chwila)…`); reg = await regionFor(z, f, true); }
    const cnt = (reg && reg.contours) || [];
    if (cnt.length > 1) toast(`F${f}: obszar poprawki ${MODES[z.mode].label} ma ${cnt.length} części – nowy klucz bierze największą`, 5000);
    const poly = cnt.length ? cnt.slice().sort((a, b) => b.length - a.length)[0] : [];
    return { base: { polygon: poly, strokes: [], bg_strokes: [] }, created: true };
  }

  // ------------------------------------------------------------------------------------------------ drawing -> keys
  function onDraw(f) {
    clearTimeout(B.drawTimer); B.drawPending = f;
    B.drawTimer = setTimeout(() => commit(f).catch(() => {}), 400);
    status(`F${f}: kreska poprawki – zapisuję klucz…`);
  }
  async function commit(f) {
    clearTimeout(B.drawTimer); B.drawPending = null;
    if (B.drawBusy) { B.drawPending = f; B.drawTimer = setTimeout(() => commit(f).catch(() => {}), 300); return; }
    const strokes = S().strokes[f] || [];
    if (!strokes.length) return;
    S().strokes[f] = [];
    const pick = (mode) => strokes.filter((x) => x.mode === mode).map((x) => ({ p: x.p, radius_px: x.radius_px }));
    const plus = { add: pick('add'), remove: pick('remove') };
    const erase = pick('erase').map((x) => ({ ...x, erase: true }));
    B.drawBusy = true;
    const done = [];
    try {
      for (const mode of ['add', 'remove']) {
        let z = zoneOf(mode);
        if (!z && !plus[mode].length) continue;
        if (!z) {
          const d = await api('POST', `/api/object/${obj()}/zones/new`, { mode, criterion: 'all', frame: f, polygon: [], strokes: plus[mode], bg_strokes: [],
            static: false, params: { source: 'brush', track: B.track ? 'local' : 'rigid' } });
          B.list.push(d.zone); B.undo.push({ zid: d.zone.zone_id, f, prev: null, created: true });
          done.push(`nowa poprawka ${MODES[mode].label}`);
          continue;
        }
        if (!plus[mode].length && !erase.length) continue;
        const { base, created } = await baseKey(z, f);
        if (!plus[mode].length && created && !(base.polygon || []).length) continue;   // nothing of this zone here to erase
        const body = { polygon: base.polygon, strokes: (base.strokes || []).concat(plus[mode], erase), bg_strokes: base.bg_strokes || [] };
        const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, body);
        B.undo.push({ zid: z.zone_id, f, prev: created ? null : { polygon: base.polygon, strokes: base.strokes, bg_strokes: base.bg_strokes } });
        replace(d.zone);
        done.push(`${MODES[mode].label}${created ? ' (nowy klucz)' : ''}`);
      }
      if (!done.length && erase.length) status(`F${f}: nic do zmazania – w tej klatce nie ma poprawki`, 'warn');
      else status(`F${f}: klucz poprawki zapisany – ${done.join(', ')}. Ctrl+Z cofa. Potem „▶▶ Zastosuj poprawki”.`, 'ok');
      H.loadStages().catch(() => {});
    } catch (e) {
      S().strokes[f] = strokes.concat(S().strokes[f] || []);
      const m = `Poprawka F${f} NIE zapisana: ${e.message}`; status(m, 'err'); toast(m, 6000);
    } finally { B.drawBusy = false; renderList(); V().render(); }
  }
  function replace(zone) {
    const i = B.list.findIndex((z) => z.zone_id === zone.zone_id);
    if (i >= 0) B.list[i] = zone; else B.list.push(zone);
    delete B.regionCache[zone.zone_id];
  }
  async function undo(f) {
    const u = B.undo[B.undo.length - 1];
    if (!u || !obj()) return false;
    B.undo.pop();
    try {
      const z = B.list.find((x) => x.zone_id === u.zid);
      if (u.created && z && keyFrames(z).length <= 1) {
        await api('DELETE', `/api/object/${obj()}/zones/${u.zid}`);
        B.list = B.list.filter((x) => x.zone_id !== u.zid); status(`Cofnięto: poprawka ${u.zid} usunięta (była tylko jedna kreska)`, 'ok');
      } else if (u.prev === null) {
        const d = await api('DELETE', `/api/object/${obj()}/zones/${u.zid}/keys/${u.f}`);
        replace(d.zone); status(`Cofnięto: klucz poprawki ${u.zid} F${u.f} usunięty`, 'ok');
      } else {
        const d = await api('PUT', `/api/object/${obj()}/zones/${u.zid}/keys/${u.f}`, u.prev);
        replace(d.zone); status(`Cofnięto kreskę poprawki ${u.zid} F${u.f}`, 'ok');
      }
      H.loadStages().catch(() => {}); renderList(); V().render();
    } catch (e) { status(`Cofanie poprawki: ${e.message}`, 'err'); }
    return true;
  }
  async function patch(zid, body, what) {
    try {
      const d = await api('PATCH', `/api/object/${obj()}/zones/${zid}`, body);
      replace(d.zone); renderList(); H.loadStages().catch(() => {});
      status(`Poprawka ${zid}: ${what} – „▶▶ Zastosuj poprawki”`, 'ok'); V().render();
    } catch (e) { status(`Poprawka ${zid} NIE zmieniona: ${e.message}`, 'err'); renderList(); }
  }
  async function deleteZone(zid) {
    const z = B.list.find((x) => x.zone_id === zid); if (!z) return;
    if (!(await V().uiConfirm(`Usunąć całą poprawkę ${MODES[z.mode].label} (${zid})?\nJej klucze (${keyFrames(z).length}) przepadają.`, { ok: 'Usuń', tone: 'error' }))) return;
    try {
      await api('DELETE', `/api/object/${obj()}/zones/${zid}`);
      B.list = B.list.filter((x) => x.zone_id !== zid); renderList(); H.loadStages().catch(() => {}); V().render();
      status(`Poprawka ${zid} usunięta`, 'ok');
    } catch (e) { status(`Nie usunięto ${zid}: ${e.message}`, 'err'); }
  }
  async function deleteKeysAt(f) {
    const zs = B.list.filter((z) => keyFrames(z).includes(f));
    if (!zs.length) { status(`F${f}: brak klucza poprawki`, 'warn'); return; }
    if (!(await V().uiConfirm(`Usunąć klucze poprawek w klatce ${f} (${zs.map((z) => MODES[z.mode].label).join(', ')})?`, { ok: 'Usuń' }))) return;
    for (const z of zs) {
      if (keyFrames(z).length <= 1) { toast(`Ostatni klucz poprawki ${MODES[z.mode].label} – usuń całą poprawkę (✕ na liście)`, 5000); continue; }
      try { const d = await api('DELETE', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`); replace(d.zone); }
      catch (e) { status(`Nie usunięto klucza ${z.zone_id}: ${e.message}`, 'err'); }
    }
    renderList(); H.loadStages().catch(() => {}); V().render();
  }
  async function emptyKey(f) {
    if (!B.list.length) { status('Brak poprawek – nie ma czego kończyć', 'warn'); return; }
    for (const z of B.list) {
      const prev = keysOf(z)[f] || null;
      const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, { polygon: [], strokes: [], bg_strokes: [] });
      B.undo.push({ zid: z.zone_id, f, prev }); replace(d.zone);
    }
    renderList(); H.loadStages().catch(() => {}); V().render();
    status(`F${f}: pusty klucz – poprawki kończą się w tej klatce (Ctrl+Z cofa)`, 'ok');
  }
  async function applyShot() {
    await flush();
    status('Maska 1+2: rdzeń SAM 3.1 + poprawki na całym ujęciu (CoTracker3 dla kluczy, które nie mają jeszcze torów)…');
    const d = await api('POST', `/api/object/${obj()}/core_fix`);
    H.clearOverlays(); await H.loadStages(); H.setView('core');
    status(`Maska 1+2 policzona F${d.frames[0]}–${d.frames[1]}: zmienione ${d.changed_px} px względem SAM 3.1 (${(d.zones || []).length} poprawek). Widok Rdzeń pokazuje wynik; dalej etap ${H.ST.MATANYONE} (MatAnyone2).`, 'ok');
  }

  // ------------------------------------------------------------------------------------------------ overlays (inside pan / zoom)
  function strokePath(ctx, st, k) {
    if (st.p.some((q) => q.length >= 3) && H.drawVarStroke) { H.drawVarStroke(ctx, st.p, k, st.radius_px); return; }
    ctx.lineWidth = 2 * (st.radius_px || 8) * k; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); st.p.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k)));
    if (st.p.length === 1) ctx.lineTo(st.p[0][0] * k + 0.01, st.p[0][1] * k);
    ctx.stroke();
  }
  function draw(ctx, img, k, f) {
    if (!inStage()) return;
    const v = V();
    ctx.save();
    try {
      for (const z of B.list) {
        if (!z.enabled) continue;
        const [r, g, b] = (MODES[z.mode] || MODES.add).col;
        const key = keysOf(z)[f];
        if (key) {
          ctx.fillStyle = ctx.strokeStyle = `rgba(${r},${g},${b},0.35)`;
          if ((key.polygon || []).length >= 3) { ctx.beginPath(); key.polygon.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); ctx.fill(); }
          for (const st of (key.strokes || []).filter((x) => !x.erase)) strokePath(ctx, st, k);
          ctx.fillStyle = ctx.strokeStyle = 'rgba(20,20,30,0.55)';
          for (const st of (key.strokes || []).filter((x) => x.erase)) strokePath(ctx, st, k);
        } else {
          const reg = (B.regionCache[z.zone_id] || {})[f];
          if (reg && reg.contours && reg.contours.length) {   // propagated correction of this frame: light fill + dashed edge
            ctx.fillStyle = `rgba(${r},${g},${b},0.18)`; ctx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
            ctx.lineWidth = 1.5 / v.zoom; ctx.setLineDash([6 / v.zoom, 4 / v.zoom]);
            ctx.beginPath(); for (const cnt of reg.contours) { cnt.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); }
            ctx.fill('evenodd'); ctx.stroke(); ctx.setLineDash([]);
          }
        }
      }
    } finally { ctx.restore(); }
  }
  // ------------------------------------------------------------------------------------------------ timeline lane
  function laneEl() {
    let lane = $('timelineBrush'); if (lane) return lane;
    const keys = $('timelineKeys'); if (!keys) return null;
    lane = H.el('<div class="timeline-zones" id="timelineBrush" title="Klucze poprawek pędzlem (zielone = Dodaj, czerwone = Usuń): klik = skok, PPM = usuń klucz"></div>');
    keys.parentNode.insertBefore(lane, keys);
    return lane;
  }
  function renderLane() {
    const lane = laneEl(); const v = V(); if (!lane || !v || !H) return;
    const on = inStage() && B.list.length;
    lane.style.display = on ? 'block' : 'none';
    lane.innerHTML = '';
    if (!on) return;
    const range = Math.max(1, v.lastFrame - v.firstFrame);
    for (const z of B.list) {
      const m = MODES[z.mode] || MODES.add;
      for (const f of keyFrames(z)) {
        const d = document.createElement('div');
        const k = keysOf(z)[f]; const empty = !(k.polygon || []).length && !(k.strokes || []).some((x) => !x.erase);
        d.className = `zone-key ${f === v.currentFrame ? 'cur' : ''}`;
        d.style.left = `${100 * (f - v.firstFrame) / range}%`; d.style.background = empty ? '#6e6e75' : m.css;
        d.title = `Klucz poprawki ${m.label} F${f}${empty ? ' (pusty – koniec poprawki)' : ''} · klik = skok · PPM = usuń klucz`;
        d.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
        d.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); v.goToFrame(f); deleteKeysAt(f); });
        lane.appendChild(d);
      }
    }
  }

  // ------------------------------------------------------------------------------------------------ wiring
  async function flush() { if (B.drawPending != null) await commit(B.drawPending); }
  function wire() {
    document.querySelectorAll('#b2Panel [data-bt]').forEach((b) => b.addEventListener('click', () => { H.setTool(b.dataset.bt); syncTools(); }));
    $('b2Track').addEventListener('change', async () => {
      B.track = $('b2Track').checked;
      for (const z of B.list) await patch(z.zone_id, { params: { ...(z.params || {}), track: B.track ? 'local' : 'rigid' } }, B.track ? 'śledzenie lokalne' : 'śledzenie sztywne');
    });
    $('b2EmptyKey').addEventListener('click', () => H.run('b2EmptyKey', () => emptyKey(curFrame())));
    $('b2DelKey').addEventListener('click', () => H.run('b2DelKey', () => deleteKeysAt(curFrame())));
    $('b2Apply').addEventListener('click', () => H.run('b2Apply', applyShot));
    window.addEventListener('md-frame', onFrame);
  }
  function init(hooks) {
    H = hooks;
    const host = $('b2Panel'); if (!host) return;
    host.innerHTML = panelHtml();
    wire();
    renderList();
    window.MD_BRUSH.ready = true;
  }
  window.MD_BRUSH = {
    init, load, reset, draw, onDraw, undo, flush, deleteKeysAt, renderLane,
    hasKeyAt: (f) => B.list.some((z) => keyFrames(z).includes(f)),
    onShow() { renderList(); onFrame(); },
    onStage() { renderLane(); syncTools(); onFrame(); },
    onStages() { renderInfo(); },
    get state() { return B; },
  };
})();
