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
  const B = { list: [], regionCache: {}, regionBusy: {}, undo: [], drawTimer: null, drawPending: null, drawBusy: false, track: true, obj: null,
              ver: 0, live: null, show: true, op: 35, active: {}, pendingNew: null };
  try { const o = localStorage.getItem('md_b2_strokes_o'); if (o !== null && +o >= 0 && +o <= 100) B.op = +o; } catch (e) { /* storage blocked */ }
  const MODES = { add: { label: 'Dodaj', col: [80, 255, 120], css: '#50ff78' }, remove: { label: 'Usuń', col: [255, 60, 90], css: '#ff3c5a' } };
  const api = (...a) => H.api(...a);
  const status = (t, k) => H.status(t, k);
  const toast = (t, ms) => H.toast(t, ms);
  const curFrame = () => V().currentFrame;
  const inStage = () => !!(H && obj() && !S().auxMode && S().stage === H.ST.BRUSH);
  const isBrush = (z) => z && z.criterion !== 'matte' && z.params && z.params.source === 'brush';
  // fix LAYERS (user decision 2026-10-30): any number of Add / Remove zones; strokes go to the ACTIVE layer (per object),
  // its keys are on the timeline and its strokes on the image; „＋ warstwa” starts a new layer with the next stroke
  const activeZone = () => B.list.find((z) => z.zone_id === B.active[obj()]) || null;
  const setActive = (zid) => { B.active[obj()] = zid; B.pendingNew = null; };
  const layerName = (z) => `${(MODES[z.mode] || MODES.add).label} ${B.list.filter((x) => x.mode === z.mode).indexOf(z) + 1}`;
  // the layer a stroke of `mode` goes to: 'new' (a „＋ warstwa” of this mode waits), the active one when it has this
  // mode, else the latest layer of this mode (it becomes active), else 'new'
  function targetFor(mode) {
    if (B.pendingNew === mode) return 'new';
    const a = activeZone();
    if (a && a.mode === mode) return a;
    const same = B.list.filter((z) => z.mode === mode);
    return same.length ? same[same.length - 1] : 'new';
  }
  const useTool = (t) => { if (S().tool !== t) H.setTool(t); };   // setTool toggles: never switch the brush off
  // after a layer is gone: the last remaining one becomes active; no layer left after „Zastosuj poprawki” = core_fix/
  // still holds the deleted fixes (group composition, export „rdzeń (1+2)”) – rebuilt at once to the SAM 3.1 core
  async function afterLayerGone() {
    if (!activeZone() && B.list.length) setActive(B.list[B.list.length - 1].zone_id);
    if (B.list.some((z) => z.enabled)) return;   // every layer deleted or switched off
    await H.loadStages().catch(() => {});
    if (H.stageState('cotracker').state === 'stale') {
      status('Usunięto ostatnią warstwę – maska 1+2 wraca do rdzenia SAM 3.1 (przeliczam)…');
      await H.run('b2Apply', applyShot);
    }
  }
  const shown = () => { const a = activeZone(); return a ? [a] : B.list; };   // layers whose keys / strokes are drawn
  function keysOf(z) {
    const out = {};
    for (const [k, v] of Object.entries(z.keys || {})) out[+k] = { polygon: v.polygon || [], strokes: v.strokes || [], bg_strokes: v.bg_strokes || [] };
    out[+z.frame] = { polygon: z.polygon || [], strokes: z.strokes || [], bg_strokes: z.bg_strokes || [] };
    return out;
  }
  const keyFrames = (z) => Object.keys(keysOf(z)).map(Number).sort((a, b) => a - b);

  // ------------------------------------------------------------------------------------------------ panel
  function panelHtml() {   // side panel = what the stage does; the tools live on the left rail and in the palette
    return `<div class="pane-section">
        <div class="sec-title">Pędzel poprawek maski</div>
        <div class="muted">Popraw maskę SAM 3.1 tam, gdzie się myli: <b class="ok">🖌 Dodaj (G)</b> domalowuje brakujący fragment (tylko piksele połączone z maską),
          <b class="err">🖌 Usuń (R)</b> zamalowuje wyciek na tło, <b>⌫ Zmaż (E)</b> cofa poprawkę w tym miejscu (narzędzia na lewym pasku). Kreska = <b>klucz poprawki</b> tej klatki;
          między kluczami i poza nimi CoTracker3 przenosi poprawkę punktami na obiekcie (nigdy na częściach obiektu ani obiektach z przodu).
          Popraw kilka klatek (najlepiej na początku, w środku i na końcu ruchu), potem <b>„▶▶ Zastosuj poprawki”</b> w palecie na obrazie i obejrzyj widok Rdzeń.
          Rozmiar pędzla: [ ] albo Shift + przeciągnij; Ctrl+Z cofa; Delete usuwa klucze poprawek tej klatki.</div>
      </div>`;
  }
  // floating palette on the canvas (left, draggable – as „Final Touch”; user decision 2026-10-30: the stage's main
  // functions on the left palette, consistently across stages)
  function paletteHtml() {
    return `<div class="pts-palette b2-palette" id="b2Palette" style="display:none">
      <div class="pp-head" title="Przeciągnij, żeby przesunąć paletę (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> Poprawki maski <span class="muted">CoTracker3</span></div>
      <div class="pp-chk"><label class="chk" title="Kreski i obszary poprawek na obrazie (zielone = Dodaj, czerwone = Usuń). Wyłącz, żeby oceniać samą maskę (rdzeń + poprawka tej klatki liczona od razu w przeglądarce)"><input type="checkbox" id="b2Show" checked title="Pokaż kreski poprawek na obrazie"> 👁 kreski pędzla</label>
        <input type="range" id="b2ShowO" min="0" max="100" step="5" value="${B.op}" style="width:70px" title="Krycie podglądu kresek (%)"><span class="muted" id="b2ShowOV">${B.op} %</span></div>
      <div class="pp-chk"><label class="chk" title="Poprawka przenoszona między klatkami punktami CoTracker3 lokalnie (każdy fragment za najbliższymi punktami – nadąża za ruchem i odkształceniem); wyłączone = sztywny kształt jadący za obiektem. Dotyczy nowych poprawek i obu istniejących"><input type="checkbox" id="b2Track" checked title="Śledzenie lokalne poprawek (CoTracker3)"> śledzenie lokalne (CoTracker3)</label></div>
      <div class="pp-sec"><div class="zp-sub" title="Warstwy poprawek: kreski idą do AKTYWNEJ warstwy (klik na warstwie); romby na osi czasu = jej klucze, PPM na rombie = usuń. Każda warstwa ma własne punkty CoTracker3 – np. osobna warstwa na każdy kolczyk">Warstwy poprawek</div>
        <div class="row"><button type="button" class="btn btn-ghost btn-xs" id="b2NewAdd" title="Nowa warstwa Dodaj: następna kreska Dodaj (G) zaczyna nową warstwę">＋ warstwa Dodaj</button>
          <button type="button" class="btn btn-ghost btn-xs" id="b2NewRem" title="Nowa warstwa Usuń: następna kreska Usuń (R) zaczyna nową warstwę">＋ warstwa Usuń</button></div>
        <div id="b2List" class="zone-list"></div></div>
      <div class="pp-sec"><div class="row">
        <button type="button" class="btn btn-ghost btn-xs" id="b2EmptyKey" title="Pusty klucz w tej klatce: poprawki kończą się tutaj (między poprzednim kluczem a tą klatką obszar zanika, dalej brak poprawki)">⌀ Koniec poprawek w tej klatce</button>
        <button type="button" class="btn btn-ghost btn-xs" id="b2DelKey" title="Usuń klucze poprawek tej klatki (obszar w tej klatce znów z propagacji) [Delete]">🗑 Usuń klucze (ta klatka)</button></div>
      <button type="button" class="btn btn-primary" id="b2Apply" style="width:100%" title="Policz maskę 1+2 (rdzeń SAM 3.1 + poprawki) na całym ujęciu – widok Rdzeń pokaże wynik w każdej klatce; ta sama maska idzie do MatAnyone2 (etap 3) i do eksportu „rdzeń (1+2)”">▶▶ Zastosuj poprawki na ujęcie</button>
      <div class="muted pp-hint" id="b2Info"></div></div>
    </div>`;
  }
  const b2Pos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_b2_palette') || 'null'); } catch (e) { return null; } },
    set: (q) => { try { if (q) localStorage.setItem('md_b2_palette', JSON.stringify(q)); else localStorage.removeItem('md_b2_palette'); } catch (e) { /* storage blocked */ } },
  };
  function renderPalette() {
    const box = $('b2Palette'); if (!box) return;
    const show = inStage() && !!(V() && V().projectLoaded);
    box.style.display = show ? '' : 'none';
    if (show && H.placeFloating) H.placeFloating(box, $('canvasWrapper'), b2Pos.get());
  }
  function renderList() {
    const box = $('b2List'); if (!box) return;
    const act = activeZone();
    const pend = B.pendingNew ? `<div class="zone-card active" style="--zc:${MODES[B.pendingNew].css}" title="Nowa warstwa powstanie z następnej kreski ${MODES[B.pendingNew].label}"><span class="zone-ic">${B.pendingNew === 'add' ? '🖌+' : '🖌−'}</span><span class="zone-name"><b>nowa warstwa ${MODES[B.pendingNew].label}</b><br><span class="muted">następna kreska ją tworzy</span></span></div>` : '';
    box.innerHTML = B.list.length || pend ? B.list.map((z) => {
      const m = MODES[z.mode] || MODES.add; const ks = keyFrames(z);
      return `<div class="zone-card ${z.enabled ? '' : 'off'} ${z === act && !B.pendingNew ? 'active' : ''}" data-z="${esc(z.zone_id)}" style="--zc:${m.css}" title="Warstwa ${layerName(z)}: klucze ${ks.map((f) => 'F' + f).join(', ')} – klik = aktywna (kreski idą do niej), dwuklik = skok do najbliższego klucza">
        <span class="zone-ic">${z.mode === 'add' ? '🖌+' : '🖌−'}</span>
        <span class="zone-name"><b>${layerName(z)}</b>${z === act && !B.pendingNew ? '<span class="zone-act" title="Kreski idą do tej warstwy">● aktywna</span>' : ''}<br><span class="muted">${esc(z.zone_id)} · ${(z.params && z.params.track) === 'local' ? 'śledzenie lokalne' : 'sztywna'}</span></span>
        <span class="zone-meta"><span title="liczba kluczy poprawki">🔑 ${ks.length}</span></span>
        <span class="zone-btns">
          <label class="chk" title="Włącz / wyłącz tę poprawkę (wyłączona nie działa w etapach 2, 3 i eksporcie)"><input type="checkbox" data-en="${esc(z.zone_id)}" ${z.enabled ? 'checked' : ''}></label>
          <button class="btn btn-ghost btn-xs" data-del="${esc(z.zone_id)}" title="Usuń całą warstwę ${layerName(z)} (wszystkie jej klucze)">✕</button></span></div>`;
    }).join('') + pend : '<div class="muted">brak poprawek – maska = rdzeń SAM 3.1 (etap opcjonalny). Pierwsza kreska G / R tworzy warstwę.</div>';
    box.querySelectorAll('.zone-card[data-z]').forEach((c) => {
      c.addEventListener('click', (e) => {
        if (e.target.closest('button') || e.target.closest('input')) return;
        const z = B.list.find((x) => x.zone_id === c.dataset.z); if (!z) return;
        setActive(z.zone_id);
        useTool(z.mode === 'add' ? 'brush_add' : 'brush_remove');
        renderList(); onFrame(); V().render();
      });
      c.addEventListener('dblclick', (e) => {
        if (e.target.closest('button') || e.target.closest('input')) return;
        const z = B.list.find((x) => x.zone_id === c.dataset.z); if (!z) return;
        const f = curFrame(); const ks = keyFrames(z);
        V().goToFrame(ks.reduce((a, b) => (Math.abs(b - f) < Math.abs(a - f) ? b : a)));
      });
    });
    box.querySelectorAll('input[data-en]').forEach((c) => c.addEventListener('change', () => patch(c.dataset.en, { enabled: c.checked }, c.checked ? 'włączona' : 'wyłączona')));
    box.querySelectorAll('button[data-del]').forEach((b) => b.addEventListener('click', () => deleteZone(b.dataset.del)));
    renderInfo(); renderLane(); syncTools();
  }
  function renderInfo() {
    const info = $('b2Info'); if (!info || !H) return;
    const st = H.stageState('cotracker');
    const f = V() ? curFrame() : null;
    const here = shown().filter((z) => keyFrames(z).includes(f)).map(layerName);
    info.textContent = `${st.state === 'done' ? '✔ ' : st.state === 'stale' ? '↻ ' : ''}${st.why || (S().info && S().info.has && S().info.has.core_fix ? 'Maska 1+2 aktualna – widok Rdzeń pokazuje ją w każdej klatce' : '')}`
      + (f != null ? ` · F${f}: ${here.length ? `klucz ${here.join(' + ')}` : 'bez klucza poprawki'}` : '');
    const ap = $('b2Apply');
    if (ap) { ap.disabled = st.state === 'locked' || document.body.classList.contains('job-busy'); ap.classList.toggle('btn-cta', st.state === 'todo' || st.state === 'stale'); }
  }
  function syncTools() { /* the tools are on the left rail (objects.js) */ }

  // ------------------------------------------------------------------------------------------------ loading
  async function load() {
    if (!obj()) { B.list = []; renderList(); return; }
    const d = await api('GET', `/api/object/${obj()}/zones`);
    B.list = (d.zones || []).filter(isBrush);
    B.regionCache = {}; B.ver++;
    if (!activeZone() && B.list.length) B.active[obj()] = B.list[B.list.length - 1].zone_id;
    if (B.list.length) B.track = B.list.every((z) => (z.params && z.params.track) === 'local');
    if ($('b2Track')) $('b2Track').checked = B.track;
    renderList();
  }
  function reset() {
    const same = B.obj === obj(); B.obj = obj();
    B.list = []; B.regionCache = {};
    if (!same) { B.undo = []; B.pendingNew = null; }
    clearTimeout(B.drawTimer); B.drawPending = null; renderLane();
  }

  // ------------------------------------------------------------------------------------------------ propagated region
  async function regionFor(z, f, wait) {
    const c = B.regionCache[z.zone_id] || (B.regionCache[z.zone_id] = {});
    if (c[f]) return c[f];
    const k = `${z.zone_id}_${f}`;
    if (B.regionBusy[k]) return wait ? B.regionBusy[k] : null;
    const p = (async () => {
      try { const d = await api('GET', `/api/object/${obj()}/zones/${z.zone_id}/region/${f}`); c[f] = d; B.ver++; if (curFrame() === f) V().render(); return d; }
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
      const act = activeZone();
      for (const mode of ['add', 'remove']) {
        // the layer of this mode's strokes (layers, 2026-10-30); erase strokes touch only the active layer
        const t = plus[mode].length ? targetFor(mode) : (act && act.mode === mode ? act : null);
        if (!t) continue;
        const er = act && t === act ? erase : [];
        if (t === 'new') {
          const d = await api('POST', `/api/object/${obj()}/zones/new`, { mode, criterion: 'all', frame: f, polygon: [], strokes: plus[mode], bg_strokes: [],
            static: false, params: { source: 'brush', track: B.track ? 'local' : 'rigid' } });
          B.list.push(d.zone); B.ver++; B.undo.push({ zid: d.zone.zone_id, f, prev: null, created: true });
          setActive(d.zone.zone_id);
          done.push(`nowa warstwa ${layerName(d.zone)}`);
          continue;
        }
        const z = t;
        if (!plus[mode].length && !er.length) continue;
        const { base, created } = await baseKey(z, f);
        if (!plus[mode].length && created && !(base.polygon || []).length) continue;   // nothing of this layer here to erase
        const body = { polygon: base.polygon, strokes: (base.strokes || []).concat(plus[mode], er), bg_strokes: base.bg_strokes || [] };
        const d = await api('PUT', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`, body);
        B.undo.push({ zid: z.zone_id, f, prev: created ? null : { polygon: base.polygon, strokes: base.strokes, bg_strokes: base.bg_strokes } });
        replace(d.zone);
        if (plus[mode].length) setActive(z.zone_id);
        done.push(`${layerName(B.list.find((x) => x.zone_id === z.zone_id) || z)}${created ? ' (nowy klucz)' : ''}`);
      }
      if (!done.length && erase.length) status(`F${f}: nic do zmazania – aktywna warstwa nie ma tu poprawki`, 'warn');
      else status(`F${f}: klucz poprawki zapisany – ${done.join(', ')}. Ctrl+Z cofa. Potem „▶▶ Zastosuj poprawki”.`, 'ok');
      H.loadStages().catch(() => {});
    } catch (e) {
      S().strokes[f] = strokes.concat(S().strokes[f] || []);
      const m = `Poprawka F${f} NIE zapisana: ${e.message}`; status(m, 'err'); toast(m, 6000);
    } finally { B.drawBusy = false; renderList(); V().render(); }
  }
  function replace(zone) {
    B.ver++;
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
        B.list = B.list.filter((x) => x.zone_id !== u.zid); B.ver++; status(`Cofnięto: poprawka ${u.zid} usunięta (była tylko jedna kreska)`, 'ok');
        await afterLayerGone();
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
      if (body.enabled === false) await afterLayerGone();
    } catch (e) { status(`Poprawka ${zid} NIE zmieniona: ${e.message}`, 'err'); renderList(); }
  }
  async function deleteZone(zid) {
    const z = B.list.find((x) => x.zone_id === zid); if (!z) return;
    if (!(await V().uiConfirm(`Usunąć całą warstwę ${layerName(z)} (${zid})?\nJej klucze (${keyFrames(z).length}) przepadają.`, { ok: 'Usuń', tone: 'error' }))) return;
    try {
      await api('DELETE', `/api/object/${obj()}/zones/${zid}`);
      B.list = B.list.filter((x) => x.zone_id !== zid); B.ver++; renderList(); V().render();
      status(`Poprawka ${zid} usunięta`, 'ok');
      await afterLayerGone(); renderList();
    } catch (e) { status(`Nie usunięto ${zid}: ${e.message}`, 'err'); }
  }
  async function deleteKeysAt(f) {
    const zs = shown().filter((z) => keyFrames(z).includes(f));   // the active layer (all when none is active)
    if (!zs.length) { status(`F${f}: brak klucza poprawki`, 'warn'); return; }
    // the last key of a fix = the fix itself: deleting it removes the whole fix (user report 2026-10-30: the key could
    // not be deleted, only a toast pointed at ✕ in the list); the question says so
    const last = zs.filter((z) => keyFrames(z).length <= 1);
    const q = `Usunąć klucze poprawek w klatce ${f} (${zs.map(layerName).join(', ')})?`
      + (last.length ? `\nTo ostatni klucz poprawki ${last.map((z) => `${layerName(z)} (${z.zone_id})`).join(', ')} – usunie całą poprawkę.` : '');
    if (!(await V().uiConfirm(q, { ok: 'Usuń', tone: last.length ? 'error' : undefined }))) return;
    for (const z of zs) {
      try {
        if (keyFrames(z).length <= 1) {
          await api('DELETE', `/api/object/${obj()}/zones/${z.zone_id}`);
          B.list = B.list.filter((x) => x.zone_id !== z.zone_id); B.ver++;
          status(`Poprawka ${z.zone_id} usunięta (był to jej ostatni klucz)`, 'ok');
        } else { const d = await api('DELETE', `/api/object/${obj()}/zones/${z.zone_id}/keys/${f}`); replace(d.zone); }
      } catch (e) { status(`Nie usunięto klucza ${z.zone_id}: ${e.message}`, 'err'); }
    }
    renderList(); V().render();
    await afterLayerGone(); renderList(); H.loadStages().catch(() => {});
  }
  async function emptyKey(f) {
    if (!B.list.length) { status('Brak poprawek – nie ma czego kończyć', 'warn'); return; }
    for (const z of shown()) {   // the active layer (all when none is active)
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
    H.clearOverlays(); await H.loadStages(); H.restoreUserView();
    status(`Maska 1+2 policzona F${d.frames[0]}–${d.frames[1]}: zmienione ${d.changed_px} px względem SAM 3.1 (${(d.zones || []).length} poprawek). Widok Rdzeń pokazuje wynik; dalej etap ${H.ST.MATANYONE} (MatAnyone2).`, 'ok');
  }

  // ------------------------------------------------------------------------------------------------ live mask (as Final Touch)
  // The mask of a frame being corrected = SAM 3.1 core + this frame's fix, drawn in the browser at once (user request
  // 2026-10-30) instead of only after „▶▶ Zastosuj poprawki”. The same rule as engine/zones.py for a brush zone in a key
  // frame: region = polygon + strokes − erase strokes (rasterised by the canvas, on the GPU); Add = core ∪ the region's
  // components touching the core (connected_to; its 200 px reach is not applied here), Remove = core − region. Frames
  // without a key or a pending stroke keep the server's mask (propagated fixes need CoTracker3).
  function regionCanvas(W, H, kp, polys, strokes) {   // region of one zone in proxy px (white), erase strokes cut out
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    g.fillStyle = g.strokeStyle = '#fff';
    for (const poly of polys) if (poly && poly.length >= 3) { g.beginPath(); poly.forEach((q, i) => (i ? g.lineTo(q[0] * kp, q[1] * kp) : g.moveTo(q[0] * kp, q[1] * kp))); g.closePath(); g.fill(); }
    for (const st of strokes.filter((x) => !x.erase)) strokePath(g, st, kp);
    g.globalCompositeOperation = 'destination-out';
    for (const st of strokes.filter((x) => x.erase)) strokePath(g, st, kp);
    return c;
  }
  function liveMatte(f) {
    if (!inStage() || !obj()) return null;
    const pend = S().strokes[f] || [];
    const drawing = S().drawing && S().drawing.f === f && S().drawing.p.length ? S().drawing : null;
    const keyed = B.list.some((z) => z.enabled && keysOf(z)[f]);
    if (!pend.length && !drawing && !(keyed && H.stageState('cotracker').state !== 'done')) return null;
    const core = H.overlayFor('core', f);
    if (!core || core === 'missing') return null;
    const W = core.naturalWidth || core.width, Hh = core.naturalHeight || core.height;
    const kp = W / +V().projectData.plate_w;
    const npts = pend.reduce((s, x) => s + x.p.length, 0) + (drawing ? drawing.p.length : 0);
    const key = `${obj()}_b2live_${f}_${B.ver}_${pend.length}_${npts}_${W}_${B.active[obj()] || ''}_${B.pendingNew || ''}`;
    if (B.live && B.live.key === key && B.live.core === core) return B.live.m;
    const t = document.createElement('canvas'); t.width = W; t.height = Hh;
    const tg = t.getContext('2d', { willReadFrequently: true });
    tg.drawImage(core, 0, 0, W, Hh);
    const cd = tg.getImageData(0, 0, W, Hh).data;
    const N = W * Hh, m = new Uint8Array(N);
    for (let i = 0; i < N; i++) m[i] = cd[i * 4] > 127 ? 1 : 0;
    const all = pend.concat(drawing ? [drawing] : []);
    const act = activeZone();
    // pending strokes of the layer z (z = 'new:<mode>' for a layer the next stroke creates)
    const strokesOf = (z, mode) => all.filter((x) => (x.mode === mode && (targetFor(mode) === z || (z === `new:${mode}` && targetFor(mode) === 'new')))
      || (x.mode === 'erase' && z === act)).map((x) => ({ p: x.p, radius_px: x.radius_px, erase: x.mode === 'erase' }));
    const regionOf = (z, mode) => {
      const key0 = typeof z === 'object' ? keysOf(z)[f] : null;
      let polys = [], strokes = [];
      if (key0) { polys = [key0.polygon]; strokes = key0.strokes || []; }
      else if (typeof z === 'object') { const reg = (B.regionCache[z.zone_id] || {})[f]; if (reg && reg.contours) polys = reg.contours; }
      strokes = strokes.concat(strokesOf(z, mode));
      if (!polys.length && !strokes.some((s) => !s.erase)) return null;
      const rc = regionCanvas(W, Hh, kp, polys, strokes).getContext('2d').getImageData(0, 0, W, Hh).data;
      const r = new Uint8Array(N); let any = false;
      for (let i = 0; i < N; i++) if (rc[i * 4 + 3] > 127) { r[i] = 1; any = true; }
      return any ? r : null;
    };
    const layers = B.list.filter((z) => z.enabled).map((z) => [z, z.mode]);
    for (const mode of ['add', 'remove']) if (all.some((x) => x.mode === mode) && targetFor(mode) === 'new') layers.push([`new:${mode}`, mode]);
    for (const [z, mode] of layers) {
      const reg = regionOf(z, mode); if (!reg) continue;
      if (mode === 'remove') { for (let i = 0; i < N; i++) if (reg[i]) m[i] = 0; continue; }
      addConnected(m, reg, W, Hh, N);
    }
    const o = tg.createImageData(W, Hh);
    for (let i = 0; i < N; i++) { const v = m[i] ? 255 : 0; o.data[i * 4] = o.data[i * 4 + 1] = o.data[i * 4 + 2] = v; o.data[i * 4 + 3] = 255; }
    tg.putImageData(o, 0, 0);
    const res = { img: t, key, kind: 'core_fix', run: 'maska 1+2 (podgląd poprawki na żywo)' };
    B.live = { key, core, m: res };
    return res;
  }
  // Add of one layer (engine/zones.connected_to): the region's components touching the mask (8-connected) join it
  function addConnected(m, add, W, Hh, N) {
    {   // components of the region touching the core (8-connected): flood from the region pixels next to it
      const out = new Uint8Array(N), q = new Int32Array(N); let qh = 0, qt = 0;
      for (let y = 0; y < Hh; y++) for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!add[i]) continue;
        if (!(m[i] || (x > 0 && m[i - 1]) || (x < W - 1 && m[i + 1]) || (y > 0 && m[i - W]) || (y < Hh - 1 && m[i + W]))) continue;
        out[i] = 1; q[qt++] = i;
      }
      while (qh < qt) {
        const i = q[qh++], x = i % W, y = (i - x) / W;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= Hh) continue;
          const j = yy * W + xx;
          if (add[j] && !out[j]) { out[j] = 1; q[qt++] = j; }
        }
      }
      for (let i = 0; i < N; i++) if (out[i]) m[i] = 1;
    }
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
    if (!inStage() || !B.show) return;
    const v = V();
    ctx.save();
    try {
      for (const z of shown()) {   // the active layer's strokes / area only (layers, 2026-10-30)
        if (!z.enabled) continue;
        const [r, g, b] = (MODES[z.mode] || MODES.add).col;
        const key = keysOf(z)[f];
        if (key) {   // the key's area as ONE flat layer (polygon + strokes − eraser): one opacity, no darker overlaps
          const paint = (c) => {
            c.fillStyle = c.strokeStyle = `rgb(${r},${g},${b})`;
            if ((key.polygon || []).length >= 3) { c.beginPath(); key.polygon.forEach((q, i) => (i ? c.lineTo(q[0] * k, q[1] * k) : c.moveTo(q[0] * k, q[1] * k))); c.closePath(); c.fill(); }
            for (const st of (key.strokes || []).filter((x) => !x.erase)) strokePath(c, st, k);
            c.globalCompositeOperation = 'destination-out';
            for (const st of (key.strokes || []).filter((x) => x.erase)) strokePath(c, st, k);
            c.globalCompositeOperation = 'source-over';
          };
          if (H.flatDraw) H.flatDraw(ctx, B.op / 100, paint);
        } else {
          const reg = (B.regionCache[z.zone_id] || {})[f];
          if (reg && reg.contours && reg.contours.length) {   // propagated correction of this frame: light fill + dashed edge
            ctx.fillStyle = `rgba(${r},${g},${b},${0.18 * B.op / 35})`; ctx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
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
    for (const z of shown()) {   // the active layer's keys (all when none is active)
      const m = MODES[z.mode] || MODES.add;
      for (const f of keyFrames(z)) {
        const d = document.createElement('div');
        const k = keysOf(z)[f]; const empty = !(k.polygon || []).length && !(k.strokes || []).some((x) => !x.erase);
        d.className = `zone-key ${f === v.currentFrame ? 'cur' : ''}`;
        d.style.left = `${100 * (f - v.firstFrame) / range}%`; d.style.background = empty ? '#6e6e75' : m.css;
        d.title = `Klucz warstwy ${layerName(z)} F${f}${empty ? ' (pusty – koniec poprawki)' : ''} · klik = skok · PPM = usuń klucz`;
        d.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
        d.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); v.goToFrame(f); deleteKeysAt(f); });
        lane.appendChild(d);
      }
    }
  }

  // ------------------------------------------------------------------------------------------------ wiring
  async function flush() { if (B.drawPending != null) await commit(B.drawPending); }
  function wire() {
    $('b2Show').addEventListener('change', () => { B.show = $('b2Show').checked; V().render(); });
    $('b2ShowO').addEventListener('input', () => {
      B.op = +$('b2ShowO').value; $('b2ShowOV').textContent = `${B.op} %`;
      try { localStorage.setItem('md_b2_strokes_o', String(B.op)); } catch (e) { /* storage blocked */ }
      V().render();
    });
    $('b2Track').addEventListener('change', async () => {
      B.track = $('b2Track').checked;
      for (const z of B.list) await patch(z.zone_id, { params: { ...(z.params || {}), track: B.track ? 'local' : 'rigid' } }, B.track ? 'śledzenie lokalne' : 'śledzenie sztywne');
    });
    const newLayer = (mode) => { B.pendingNew = mode; useTool(mode === 'add' ? 'brush_add' : 'brush_remove'); renderList(); V().render();
      status(`Nowa warstwa ${MODES[mode].label}: następna kreska ${mode === 'add' ? 'Dodaj (G)' : 'Usuń (R)'} ją utworzy`); };
    $('b2NewAdd').addEventListener('click', () => newLayer('add'));
    $('b2NewRem').addEventListener('click', () => newLayer('remove'));
    $('b2EmptyKey').addEventListener('click', () => H.run('b2EmptyKey', () => emptyKey(curFrame())));
    $('b2DelKey').addEventListener('click', () => H.run('b2DelKey', () => deleteKeysAt(curFrame())));
    $('b2Apply').addEventListener('click', () => H.run('b2Apply', applyShot));
    window.addEventListener('md-frame', onFrame);
  }
  function init(hooks) {
    H = hooks;
    const host = $('b2Panel'); if (!host) return;
    host.innerHTML = panelHtml();
    const cw = $('canvasWrapper');
    if (cw) { const pal = H.el(paletteHtml()); cw.appendChild(pal); if (H.makeFloating) H.makeFloating(pal, cw, '.pp-head', b2Pos); }
    wire();
    renderList();
    window.MD_BRUSH.ready = true;
  }
  window.MD_BRUSH = {
    init, load, reset, draw, onDraw, undo, flush, deleteKeysAt, renderLane, liveMatte,
    hasKeyAt: (f) => shown().some((z) => keyFrames(z).includes(f)),
    onShow() { renderList(); onFrame(); renderPalette(); },
    onStage() { renderLane(); syncTools(); onFrame(); renderPalette(); },
    onStages() { renderInfo(); },
    get state() { return B; },
  };
})();
