// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/bandpoints.js – band control points of a STAGE band (one band system, user decision 2026-10-22; the pattern is
// the stage-5 whole-outline zone in static/zones.js): today stage 4 VideoMaMa (later MatAnyone2 / LTX-2.5). The band
// lies along the stage's base mask; the sliders of the stage palette size it, the points shape it: yellow whisker =
// band into the background, blue = into the object; a drag = a key in this frame; 0 / 0 = a gap. Tools on the left
// rail (objects.js S.tool band_add / band_remove), keys = triangles on the timeline, 🔑✕ / Delete / Ctrl+Z work on
// them, preview = dashed band edges + thin dashes where the feathers end (GET …/stage_band/{stage}/{f}/lines).
// API: GET / PUT / DELETE /api/object/{o}/stage_band/{stage}/…, POST …/copy (from a zone or another stage).
// The points are the shared object band by default (2026-10-22, engine/objband.py): the same endpoints edit it; a
// detached stage („odłącz”) keeps its own copy, PUT …/mode, POST /api/object/{o}/band/from („ustaw jako wspólny”).
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  let H = null;
  const S = () => H.S;
  const obj = () => H.S.obj;
  const B = { data: {}, lines: {}, busy: {}, drag: null, live: null, undo: [], tracking: false };
  // the stage whose band is edited: 4 VideoMaMa, 3 MatAnyone2 (px mode of the edge band)
  const STAGE_OF = () => (!H ? null : S().stage === H.ST.VIDEOMAMA ? 'videomama' : S().stage === H.ST.MATANYONE ? 'matanyone'
    : S().stage === H.ST.LTX ? 'ltx' : null);
  const PFX = () => ({ matanyone: 'ep', videomama: 'vm', ltx: 'lx' }[STAGE_OF()] || 'vm');   // palette ids: epBandInfo …
  const on = () => !!(H && obj() && !S().auxMode && STAGE_OF());
  const base = () => `/api/object/${obj()}/stage_band/${STAGE_OF()}`;
  const key = (f) => `${obj()}|${STAGE_OF()}|${f}`;
  const tr = (t) => (window.mdT ? window.mdT(t) : t);
  const MIN_WHISKER_SCREEN = 14;

  const st = (f) => B.data[key(f)] || null;
  const pts = (f) => ((st(f) || {}).points || []);
  const allKeys = () => { const s = st(V().currentFrame); return s ? [...new Set(s.points.flatMap((p) => p.keys || []))].sort((a, b) => a - b) : []; };

  async function load(f) {
    const k = key(f);
    if (B.busy[k] || !on()) return;
    B.busy[k] = true;
    try {
      const d = await H.api('GET', `${base()}/${f}`);
      B.data[k] = d;
      if (d.tracking) { B.tracking = true; setTimeout(() => { delete B.data[key(V().currentFrame)]; load(V().currentFrame); }, 2000); }
      else if (B.tracking) { B.tracking = false; dirtyLines(); }
      info(); renderLane(); V().render();
    } catch (e) { /* no core in this frame: asked again on the next draw */ }
    finally { delete B.busy[k]; }
  }
  function dirty() { B.data = {}; dirtyLines(); B.live = null; }
  function dirtyLines() { B.lines = {}; }
  // preview lines of the band: the palette's slider values (live while dragging = the whiskers scale, lines after release)
  async function loadLines(f, params, prefetch) {
    const q = Object.entries(params || {}).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
    const k = `${key(f)}|${q}`;
    if (B.lines[k] || B.busy[k]) return B.lines[k] || null;
    if (Object.keys(B.lines).length > 240) B.lines = {};
    B.busy[k] = true;
    const url = `${base()}/${f}/lines?${q}`;
    try { B.lines[k] = await H.api('GET', url); if (!prefetch) V().render(); }
    catch (e) { if (!prefetch) B.lines[k] = { band_contours: [] }; }
    finally { delete B.busy[k]; }
    // the neighbours in the background (user report 2026-10-22: the band showed late after a frame step): the next
    // step finds its lines here / in the server cache – one at a time, only while the frame stays put and not playing
    if (!prefetch) {
      for (const g of [f + 1, f - 1]) {
        const v = V();
        if (!v || v.currentFrame !== f || v.isPlaying || g < v.firstFrame || g > v.lastFrame || !on()) break;
        await loadLines(g, params, true);
      }
    }
    return null;
  }
  // a band edit: Ctrl+Z restores the points as they were (the stage settings carry band_points + reference)
  async function snap(f, what) {   // the points as they are now (GET …/params), for Ctrl+Z
    let prev = { band_points: [], band_ref_out: null, band_ref_in: null };
    try { const d = await H.api('GET', `${base()}/params`); prev = { band_points: d.band_points || [], band_ref_out: d.band_ref_out ?? null, band_ref_in: d.band_ref_in ?? null }; }
    catch (e) { /* no band yet (px mode off): nothing to restore */ }
    B.undo.push({ stage: STAGE_OF(), url: base(), f, what, prev });
  }
  async function save(f, body, what, del) {
    await snap(f, what);
    try {
      if (del != null) await H.api('DELETE', `${base()}${del === '' || del[0] === '?' ? '' : '/'}${del}`);
      else await H.api('PUT', `${base()}/${f}`, body);
      dirty(); await reloadStage(); renderMode();
      H.status(`Pas ${STAGE_OF()} F${f}: ${what}`, 'ok');
    } catch (e) { B.undo.pop(); dirty(); H.toast(`Punkt pasa: ${e.message || e}`, 6000); }
    renderLane(); V().render();
  }
  async function reloadStage() {   // the stage palette reads its settings again (objects.js loadVmm via the reload hook)
    if (H.reloadStageBand) await H.reloadStageBand();
  }
  async function undo() {
    const u = B.undo[B.undo.length - 1];
    if (!u) { H.status('Nic do cofnięcia w tym etapie', 'warn'); return true; }
    B.undo.pop();
    try {
      await H.api('PUT', `${u.url}/params`, u.prev);
      dirty(); await reloadStage(); renderLane(); V().render();
      H.status(`Cofnięto: ${u.what} (F${u.f})`, 'ok');
    } catch (e) { H.status(`Cofanie pasa: ${e.message}`, 'err'); }
    return true;
  }
  const hasKeyAt = (f) => pts(f).some((p) => (p.keys || []).includes(f));
  async function deleteKeyAt(f) {
    const ids = pts(f).filter((p) => (p.keys || []).includes(f)).map((p) => p.id);
    if (!ids.length) { H.toast(`F${f}: brak kluczy pasa`); return false; }
    const gone = pts(f).filter((p) => (p.keys || []).length === 1 && p.keys[0] === f).map((p) => p.id);
    const msg = `Usunąć klucze pasa w klatce ${f} (${ids.join(', ')})?` + (gone.length ? `\nPunkty bez innych kluczy znikną: ${gone.join(', ')}` : '');
    if (!(await V().uiConfirm(msg, { ok: 'Usuń', tone: 'error' }))) return false;
    await save(f, null, `klucze pasa usunięte (${ids.join(', ')})`, `?frame=${f}`);
    return true;
  }
  async function copyFrom(source, label) {
    await snap(V().currentFrame, `punkty pasa skopiowane z ${label}`);
    try {
      const d = await H.api('POST', `${base()}/copy`, { source });
      dirty(); await reloadStage(); renderLane(); renderMode(); V().render();
      H.status(`Skopiowano ${d.points} pkt pasa z ${label} (klucze: ${d.keys.map((k) => 'F' + k).join(' ')})`, 'ok');
    } catch (e) { B.undo.pop(); H.toast(`Kopiowanie pasa: ${e.message || e}`, 6000); }
  }

  // ---- shared object band (user decision 2026-10-22): the badge „pas obiektu (wspólny)” + „odłącz” / „wróć do wspólnego” /
  // „ustaw jako wspólny”; „Kopiuj punkty pasa z …” only for a detached band. `bkey` = a stage or "zone:<id>".
  async function renderModeInto(el, bkey, copyEls, after) {
    if (!el || !H || !obj()) return;
    let d;
    try { d = await H.api('GET', `/api/object/${obj()}/band`); } catch (e) { el.innerHTML = ''; return; }
    const me = (d.owners || []).find((o) => o.key === bkey) || { mode: 'shared', points: d.points };
    const own = me.mode === 'own';
    const keys = (d.keys || []).map((k) => 'F' + k).join(' ');
    const pk = (n) => tr(n === 1 ? 'punkt' : 'pkt');
    const idle = me.active === false   // the stage does not read points in its current mode
      ? ` <span class="muted" title="Etap czyta punkty pasa tylko w trybie px (MatAnyone2: „Pas na zewnątrz” / „Wcięcie” > 0) albo w wyniku „pas” (LTX-2.5)">· ${tr('nieużywany w tym trybie')}</span>` : '';
    el.innerHTML = own
      ? `<span class="muted" title="Ten etap ma własną kopię punktów pasa – zmiany wspólnego pasa obiektu go nie dotyczą">✂ ${tr('własny pas')} · ${me.points} ${pk(me.points)}</span>${idle}
         <button type="button" class="btn btn-ghost btn-xs" data-bm="shared" title="Wróć do wspólnego pasa obiektu – własne punkty tego etapu zostaną usunięte">🔗 ${tr('wróć do wspólnego')}</button>
         ${me.points ? `<button type="button" class="btn btn-ghost btn-xs" data-bm="from" title="Punkty tego etapu stają się wspólnym pasem obiektu – zmienią się wszystkie etapy na wspólnym pasie (↻)">⇧ ${tr('ustaw jako wspólny')}</button>` : ''}`
      : `<span class="muted" title="Jedne punkty pasa dla etapów 3 / 4 / 5 / 7 – każdy etap skaluje je własnymi suwakami; zmiana punktu = ↻ we wszystkich etapach na wspólnym pasie">🔗 ${tr('pas obiektu (wspólny)')} · ${d.points} ${pk(d.points)}${keys ? ` · ${tr('klucze')} ${keys}` : ''}</span>${idle}
         <button type="button" class="btn btn-ghost btn-xs" data-bm="own" title="Odłącz → własny pas: ten etap dostaje kopię punktów i dalej zmiany wspólnego pasa go nie dotyczą">⛓ ${tr('odłącz')}</button>`;
    for (const c of copyEls || []) if (c) c.style.display = own ? '' : 'none';
    el.querySelectorAll('[data-bm]').forEach((b) => b.addEventListener('click', () => setMode(bkey, b.dataset.bm, after)));
  }
  async function setMode(bkey, what, after) {
    try {
      if (what === 'shared') {
        if (!(await V().uiConfirm('Wrócić do wspólnego pasa obiektu? Własne punkty tego etapu zostaną usunięte.', { ok: 'Wróć', tone: 'error' }))) return;
        await H.api('PUT', `/api/object/${obj()}/stage_band/${bkey}/mode`, { mode: 'shared' });
      } else if (what === 'own') {
        await H.api('PUT', `/api/object/${obj()}/stage_band/${bkey}/mode`, { mode: 'own' });
      } else {
        if (!(await V().uiConfirm('Ustawić punkty tego etapu jako wspólny pas obiektu? Zmienią się wszystkie etapy na wspólnym pasie (↻).', { ok: 'Ustaw' }))) return;
        await H.api('POST', `/api/object/${obj()}/band/from`, { source: bkey });
      }
      B.undo = [];   // the snapshots belonged to the other band
      dirty(); await reloadStage(); renderLane(); V().render();
      H.status({ shared: 'Etap czyta wspólny pas obiektu', own: 'Etap odłączony – własny pas (kopia wspólnego)', from: 'Punkty etapu = wspólny pas obiektu' }[what], 'ok');
    } catch (e) { H.toast(`Pas obiektu: ${e.message || e}`, 6000); }
    if (after) after(); else renderMode();
  }
  function renderMode() {
    if (!on()) return;
    const p = PFX(); const src = $(`${p}BandSrc`);
    renderModeInto($(`${p}BandMode`), STAGE_OF(), [src && src.closest('label'), $(`${p}BandCopy`)]);
  }

  // ---- geometry / drawing (as static/zones.js)
  function plateScale() { const img = V().getCurrentImage(); const ps = H.plateSize && H.plateSize(); return img && ps ? img.width / ps[0] : 1; }
  function geom(p) {
    if (!p.visible || p.sx == null) return null;
    const d = B.drag && B.drag.p.id === p.id ? B.drag : null;
    const g = { sx: p.sx, sy: p.sy, nx: p.nx, ny: p.ny, out: p.out, in: p.in };
    if (B.live) { g.out *= B.live.out; g.in *= B.live.in; }
    if (d && d.kind === 'move') { g.sx = d.x; g.sy = d.y; }
    if (d && d.kind === 'out') g.out = d.val;
    if (d && d.kind === 'in') g.in = d.val;
    return g;
  }
  function ends(g) {
    const m = MIN_WHISKER_SCREEN / (V().zoom * plateScale());
    const lo = Math.max(g.out, m), li = Math.max(g.in, m);
    return { o: [g.sx + g.nx * lo, g.sy + g.ny * lo], i: [g.sx - g.nx * li, g.sy - g.ny * li] };
  }
  function hit(f, p) {
    const s = st(f); if (!s) return null;
    const tol = 9 / (V().zoom * plateScale());
    const near = (q) => Math.hypot(q[0] - p[0], q[1] - p[1]) <= tol;
    for (const pt of s.points) {
      const g = geom(pt); if (!g) continue;
      const e = ends(g);
      if (near(e.o)) return { kind: 'out', p: pt };
      if (near(e.i)) return { kind: 'in', p: pt };
      if (near([g.sx, g.sy])) return { kind: 'move', p: pt };
    }
    return null;
  }
  const poly = (ctx, k, cnt) => { ctx.beginPath(); cnt.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); };
  // the band preview lines (the stage palette calls it when „👁 podgląd pasa” is on)
  function drawLines(ctx, img, f, params) {
    if (!on()) return false;
    const k = img.width / ((H.plateSize && H.plateSize()[0]) || img.width);
    const q = Object.entries(params || {}).map(([kk, v]) => `${kk}=${encodeURIComponent(v)}`).join('&');
    const L = B.lines[`${key(f)}|${q}`];
    if (!L) { loadLines(f, params); return false; }
    const v = V();
    ctx.save();
    try {
      ctx.strokeStyle = 'rgba(255,212,0,0.95)'; ctx.lineWidth = 1.6 / v.zoom; ctx.setLineDash([6 / v.zoom, 4 / v.zoom]);
      for (const c of L.band_contours || []) { poly(ctx, k, c); ctx.stroke(); }
      ctx.strokeStyle = 'rgba(255,212,0,0.8)'; ctx.lineWidth = 0.8 / v.zoom; ctx.setLineDash([3 / v.zoom, 3 / v.zoom]);
      for (const c of [...(L.inset_feather_contours || []), ...(L.feather_contours || [])]) { poly(ctx, k, c); ctx.stroke(); }
    } finally { ctx.restore(); }
    return true;
  }
  function draw(ctx, img, k, f) {
    if (!on()) return;
    const s = st(f);
    if (!s) { load(f); return; }
    const v = V(); const lw = 2 / v.zoom; const r = 5 / v.zoom; const h = 4 / v.zoom;
    ctx.save();
    try {
      for (const p of s.points) {
        const g = geom(p);
        if (!g) {
          if (p.x != null) { ctx.strokeStyle = 'rgba(190,190,190,0.8)'; ctx.lineWidth = lw; ctx.beginPath(); ctx.arc(p.x * k, p.y * k, r, 0, 2 * Math.PI); ctx.stroke(); }
          continue;
        }
        const e = ends(g);
        for (const [end, col] of [[e.o, '#ffd400'], [e.i, '#4da3ff']]) {
          ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw;
          ctx.beginPath(); ctx.moveTo(g.sx * k, g.sy * k); ctx.lineTo(end[0] * k, end[1] * k); ctx.stroke();
          ctx.fillRect(end[0] * k - h, end[1] * k - h, 2 * h, 2 * h);
        }
        ctx.beginPath(); ctx.arc(g.sx * k, g.sy * k, r, 0, 2 * Math.PI);
        ctx.fillStyle = p.key ? '#ffffff' : '#ff6fb5'; ctx.fill();
        ctx.strokeStyle = '#111'; ctx.lineWidth = 1 / v.zoom; ctx.stroke();
      }
    } finally { ctx.restore(); }
  }
  // the palette sliders are being dragged: the whiskers scale by slider now / slider saved (as the stage-5 zone)
  function setLive(out0, in0, outNow, inNow) {
    B.live = (pts(V().currentFrame).length && out0 > 0 && in0 >= 0) ? { out: outNow / (out0 || 1), in: in0 > 0 ? inNow / in0 : 1 } : null;
    V().render();
  }

  // ---- tools (left rail) and pointer
  const tool = () => (on() && ['band_add', 'band_remove'].includes(S().tool) ? S().tool : null);
  function onTool() { V().canvas.style.cursor = tool() ? 'crosshair' : ''; info(); V().render(); }
  function wire(cv) {
    cv.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || e.altKey || !on()) return;
      const f = V().currentFrame; const p = H.canvasToPlate(e); if (!p) return;
      const ht = hit(f, p); const t = tool();
      if (t === 'band_remove') {
        if (!ht) return;
        e.preventDefault(); e.stopPropagation(); e.mdHandled = true;
        save(f, null, `punkt ${ht.p.id} usunięty`, ht.p.id); return;
      }
      if (!ht && t !== 'band_add') return;
      e.preventDefault(); e.stopPropagation(); e.mdHandled = true;
      if (!ht) { save(f, { x: p[0], y: p[1] }, 'nowy punkt'); return; }
      try { cv.setPointerCapture(e.pointerId); } catch (er) { /* not capturable */ }
      B.drag = { ...ht, pid: e.pointerId, f, x0: p[0], y0: p[1], x: p[0], y: p[1], val: ht.kind === 'out' ? ht.p.out : ht.kind === 'in' ? ht.p.in : 0 };
    }, true);
    cv.addEventListener('pointermove', (e) => {
      const d = B.drag; if (!d || e.pointerId !== d.pid) return;
      const p = H.canvasToPlate(e); if (!p) return;
      if (d.kind === 'move') { d.x = p[0]; d.y = p[1]; }
      else { const sg = d.kind === 'out' ? 1 : -1; d.val = Math.max(0, Math.round(sg * ((p[0] - d.p.sx) * d.p.nx + (p[1] - d.p.sy) * d.p.ny))); }
      V().render();
    }, true);
    const up = (e) => {
      const d = B.drag; if (!d || e.pointerId !== d.pid) return;
      B.drag = null;
      if (d.kind === 'move') {
        if (Math.hypot(d.x - d.x0, d.y - d.y0) < 1) { V().render(); return; }
        save(d.f, { id: d.p.id, x: d.x, y: d.y }, `punkt ${d.p.id} przesunięty (klucz)`);
      } else {
        const side = d.kind === 'out' ? 'na zewnątrz' : 'do środka';
        save(d.f, { id: d.p.id, [d.kind]: d.val }, `${d.p.id}: ${tr(side)} ${d.val} px (klucz)`);
      }
    };
    cv.addEventListener('pointerup', up, true);
    cv.addEventListener('pointercancel', up, true);
    cv.addEventListener('contextmenu', async (e) => {
      if (!on()) return;
      const f = V().currentFrame; const p = H.canvasToPlate(e); const ht = p ? hit(f, p) : null; if (!ht) return;
      e.preventDefault(); e.stopPropagation();
      save(f, null, `punkt ${ht.p.id} usunięty`, ht.p.id);
    }, true);
  }

  // ---- timeline: triangles of the band keys
  function laneEl() {
    let lane = $('timelineStageBand'); if (lane) return lane;
    const keys = $('timelineKeys'); if (!keys) return null;
    lane = H.el('<div class="timeline-zones" id="timelineStageBand" title="Klucze punktów pasa etapu (trójkąty): klik = skok, PPM = usuń klucze pasa tej klatki"></div>');
    keys.parentNode.insertBefore(lane, keys);
    return lane;
  }
  function renderLane() {
    const lane = laneEl(); const v = V(); if (!lane || !v) return;
    lane.innerHTML = '';
    const ks = on() ? allKeys() : [];
    lane.style.display = ks.length ? 'block' : 'none';
    const range = Math.max(1, v.lastFrame - v.firstFrame);
    for (const f of ks) {
      const m = document.createElement('div');
      m.className = `band-key ${f === v.currentFrame ? 'cur' : ''}`;
      m.style.left = `${100 * (f - v.firstFrame) / range}%`;
      m.title = `Klucz punktów pasa F${f} · klik = skok · PPM = usuń klucze pasa`;
      m.addEventListener('click', (e) => { e.stopPropagation(); v.goToFrame(f); });
      m.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); v.goToFrame(f); deleteKeyAt(f); });
      lane.appendChild(m);
    }
  }
  function info() {
    const el = $(`${PFX()}BandInfo`); if (!el || !on()) return;
    const f = V().currentFrame; const n = pts(f).length;
    if (B.tracking) { el.textContent = '⏳ śledzę punkty (CoTracker3) – na razie położenie z najbliższego klucza'; return; }
    el.textContent = tool() === 'band_add' ? 'klik przy krawędzi bazy = nowy punkt (Esc = koniec)'
      : tool() === 'band_remove' ? 'klik na punkt = usuń punkt (Esc = koniec)'
        : (n ? `${n} ${tr(n === 1 ? 'punkt' : 'pkt')} · klucze: ${allKeys().map((k) => 'F' + k).join(' ')} · PPM na punkcie = usuń`
          : 'brak – pas o stałej szerokości z suwaków; narzędzie „Dodaj punkt” (G) na pasku po lewej');
    const clr = $(`${PFX()}BandClear`);
    if (clr) { clr.style.display = n ? '' : 'none'; clr.disabled = !hasKeyAt(f); }
  }
  function onFrame() { if (on()) { if (!st(V().currentFrame)) load(V().currentFrame); renderLane(); info(); } else renderLane(); }
  function reset() { B.data = {}; B.lines = {}; B.undo = []; B.live = null; B.drag = null; renderLane(); }

  function init(hooks) {
    H = hooks;
    wire(V().canvas);
    window.addEventListener('md-frame', onFrame);
    window.MD_BANDPTS.ready = true;
  }
  window.MD_BANDPTS = { init, draw, drawLines, onTool, onFrame, reset, dirty, undo, hasKeyAt, deleteKeyAt, copyFrom, setLive,
                        renderLane, info, stage: STAGE_OF, renderMode, renderModeInto, ready: false };
})();
