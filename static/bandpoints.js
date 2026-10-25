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
  const B = { data: {}, lines: {}, lastLines: {}, busy: {}, drag: null, live: null, undo: [], tracking: false, keep: null };
  // the band being edited: a stage (3 MatAnyone2 / 4 VideoMaMa / 7 LTX-2.5) or, in stage 5, the active whole-outline zone
  // („zone:<id>” – one editor for every band, user decision 2026-10-24; the server's stage_band endpoints take both)
  const STAGE_OF = () => (!H ? null : S().stage === H.ST.VIDEOMAMA ? 'videomama' : S().stage === H.ST.MATANYONE ? 'matanyone'
    : S().stage === H.ST.LTX ? 'ltx' : S().stage === H.ST.VITMATTE && H.zoneBand ? H.zoneBand() : null);
  const isZone = () => (STAGE_OF() || '').startsWith('zone:');
  const PFX = () => (isZone() ? 'z4' : ({ matanyone: 'ep', videomama: 'vm', ltx: 'lx' }[STAGE_OF()] || 'vm'));   // palette ids: epBandInfo …
  const on = () => !!(H && obj() && !S().auxMode && STAGE_OF());
  const base = () => `/api/object/${obj()}/stage_band/${STAGE_OF()}`;
  const key = (f) => `${obj()}|${STAGE_OF()}|${f}`;
  // the band preview checkbox of the stage hides the band points with it (user report 2026-10-24); same sources as the palettes
  const ls = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : v === '1'; } catch (e) { return d; } };
  const pvShown = () => {
    const s = STAGE_OF();
    if (s === 'matanyone') return S().edgePv == null ? true : !!S().edgePv;
    if (s === 'videomama') return ls('md_vmm_band', true);
    if (s === 'ltx') return ls('md_ltx_band', true);
    if (isZone()) return (() => { try { return localStorage.getItem('md_zone_bandpv') !== '0'; } catch (e) { return true; } })();
    return true;
  };
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
    } catch (e) {
      // no core in this frame: asked again on the next draw; a refusal (409: the stage reads no points in this mode –
      // stage 3 adaptive) is kept per frame, every render asked again before (50 requests in 50 ms, 2026-10-24)
      if (e.status === 409) B.data[k] = { frame: f, points: [], defaults: {}, refused: e.message };
    }
    finally { delete B.busy[k]; }
  }
  function dirty() { B.data = B.keep ? { [B.keep.k]: B.keep.d } : {}; dirtyLines(); B.live = null; }
  function dirtyLines() { B.lines = {}; B.lastLines = {}; }
  // preview lines of the band: the palette's slider values (live while dragging = the whiskers scale, lines after release)
  async function loadLines(f, params, prefetch) {
    const q = Object.entries(params || {}).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
    const k = `${key(f)}|${q}`;
    if (B.lines[k] || B.busy[k]) return B.lines[k] || null;
    if (Object.keys(B.lines).length > 240) B.lines = {};
    B.busy[k] = true;
    const url = `${base()}/${f}/lines?${q}`;
    try {
      B.lines[k] = await H.api('GET', url); if (!prefetch) V().render();
      // the mask 1+2 of this frame was the quick stand-in (the exact one is computed in the background, 2026-10-24):
      // shown now; asked again ONCE, when the cheap status says the exact mask is ready (no re-asking every few seconds)
      if (B.lines[k] && B.lines[k].approx) watchMask12(() => { for (const kk of Object.keys(B.lines)) if (B.lines[kk] && B.lines[kk].approx) delete B.lines[kk]; if (V()) V().render(); });
    }
    catch (e) {   // the reason stays with the frame (lineNote) until the band changes – not „ładuję…” for ever (2026-10-24)
      if (!prefetch) { B.lines[k] = { band_contours: [], error: e.message || String(e) }; V().render(); }
    }
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
    // a zone band edit goes on the zone's own undo stack too (drawing and band edits in one order, Ctrl+Z in stage 5)
    if (isZone() && window.MD_ZONES && window.MD_ZONES.pushBandUndo) window.MD_ZONES.pushBandUndo();
  }
  async function save(f, body, what, del) {
    await snap(f, what);
    try {
      let r = null;
      if (del != null) await H.api('DELETE', `${base()}${del === '' || del[0] === '?' ? '' : '/'}${del}`);
      else r = await H.api('PUT', `${base()}/${f}`, body);
      // the PUT answers with the band of this frame: kept through the dirty() of the stage reload, so the same state is
      // not asked again (2026-10-23, ~2 s with 24 points)
      B.keep = r && r.band ? { k: key(f), d: r.band } : null;
      dirty();
      if (B.keep && B.keep.d.tracking && !B.tracking) { B.tracking = true; setTimeout(() => { delete B.data[key(V().currentFrame)]; load(V().currentFrame); }, 2000); }
      try { await reloadStage(); } finally { B.keep = null; }
      renderMode();
      H.status(`Pas ${STAGE_OF()} F${f}: ${what}`, 'ok');
    } catch (e) { B.undo.pop(); dirty(); H.toast(`Punkt pasa: ${e.message || e}`, 6000); }
    renderLane(); V().render();
  }
  async function reloadStage() {   // the stage palette reads its settings again (objects.js loadVmm via the reload hook)
    if (isZone()) {   // stage 5: the zone list again, AND the palettes of stages 3 / 4 / 7 – „odłącz” / „ustaw jako wspólny” / a
      // band edit of a zone changes the band they show (user report 2026-10-24: stage 3 kept the old sliders)
      if (window.MD_ZONES && window.MD_ZONES.reloadBand) await window.MD_ZONES.reloadBand();
      if (H.reloadBandPalettes) await H.reloadBandPalettes({ zones: false });
      return;
    }
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
      ? ` <span class="muted" title="Wynik LTX-2.5 „surowy” nie ma pasa – model sam decyduje, co wycina; punkty pasa działają w wyniku „pas”">· ${tr('nieużywany w tym trybie')}</span>` : '';
    el.innerHTML = own
      ? `<span class="muted" title="Ten etap ma własną kopię pasa – suwaków i punktów; zmiany wspólnego pasa obiektu go nie dotyczą">✂ ${tr('własny pas')} · ${me.points} ${pk(me.points)}</span>${idle}
         <button type="button" class="btn btn-ghost btn-xs" data-bm="shared" title="Wróć do wspólnego pasa obiektu – ten etap znów czyta wspólne suwaki i punkty; własne punkty tego etapu zostaną usunięte">🔗 ${tr('wróć do wspólnego')}</button>
         ${me.points ? `<button type="button" class="btn btn-ghost btn-xs" data-bm="from" title="Punkty tego etapu stają się wspólnym pasem obiektu – zmienią się wszystkie etapy na wspólnym pasie (↻)">⇧ ${tr('ustaw jako wspólny')}</button>` : ''}`
      : `<span class="muted" title="Jeden pas dla etapów 3 / 4 / 5 / 7 – te same suwaki (pas, wcięcie, wtopienia) i punkty; zmiana suwaka lub punktu = ↻ we wszystkich etapach na wspólnym pasie">🔗 ${tr('pas obiektu (wspólny)')} · ${d.points} ${pk(d.points)}${keys ? ` · ${tr('klucze')} ${keys}` : ''}</span>${idle}
         <button type="button" class="btn btn-ghost btn-xs" data-bm="own" title="Odłącz → własny pas: ten etap dostaje kopię suwaków i punktów i dalej zmiany wspólnego pasa go nie dotyczą">⛓ ${tr('odłącz')}</button>`;
    renderSeed(document.getElementById(el.id.replace(/BandMode$/, 'BandSeed')), bkey, own, after);
    for (const c of copyEls || []) if (c) c.style.display = own ? '' : 'none';
    el.querySelectorAll('[data-bm]').forEach((b) => b.addEventListener('click', () => setMode(bkey, b.dataset.bm, after)));
  }
  // „Zaproponuj pas” (Sapiens2-seg-0.4B, 2026-10-22), under „👁 podgląd pasa” of the stage palette: always shown,
  // greyed out with a Doctor hint until the model is downloaded (info.band_seed; user decision 2026-10-23)
  const SEED_ICON = '<svg class="bs-ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="6" r="2.6" fill="currentColor"/>'
    + '<path d="M8.2 20.5v-6.2c0-2.4 1.7-4.2 3.8-4.2s3.8 1.8 3.8 4.2v6.2z" fill="currentColor"/>'
    + '<path d="M12 1.2c3 0 5.4 2.3 5.4 5.1 0 1.3-.5 2.4-1.3 3.3 1.9 1.1 3.1 3.1 3.1 5.5v6.4M12 1.2c-3 0-5.4 2.3-5.4 5.1 0 1.3.5 2.4 1.3 3.3-1.9 1.1-3.1 3.1-3.1 5.5v6.4" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="2.2 1.8" opacity=".85"/></svg>';
  function renderSeed(el, bkey, own, after) {
    if (!el) return;
    const ready = !!(S().info && S().info.band_seed);
    el.innerHTML = `<button type="button" class="btn band-seed-btn" data-bseed="1"${ready ? '' : ' disabled'} title="${ready
      ? 'Sapiens2 rozpoznaje części ciała w tej klatce i ustawia punkty pasa: włosy szeroko, ubranie wąsko, skóra cienko (zawsze co najmniej tyle, ile automat mierzy na rozmyciu). Tylko ludzie. Jeden klucz w tej klatce, dalej punkty śledzi CoTracker3; poprawiasz wąsami. Ctrl+Z cofa.'
      : 'Brak modelu Sapiens2-seg-0.4B – Doktor → Zainstaluj Sapiens2'}">${SEED_ICON}<span>${tr('Zaproponuj pas (Sapiens2)')}</span></button>`
      + (ready ? '' : `<div class="band-seed-hint">${tr('Brak modelu Sapiens2-seg-0.4B – zainstaluj go w')} <a href="#" data-bdoc="1">🩺 ${tr('Doktorze')}</a> (${tr('Zainstaluj Sapiens2')}, 1,63 GB).</div>`);
    const b = el.querySelector('[data-bseed]');
    if (ready) b.addEventListener('click', () => proposeBand(bkey, own, after));
    const doc = el.querySelector('[data-bdoc]');
    if (doc) doc.addEventListener('click', (e) => { e.preventDefault(); const d = document.getElementById('openDoctorBtn'); if (d) d.click(); });
  }
  async function proposeBand(bkey, own, after) {
    const f = V().currentFrame;
    // Ctrl+Z: the band as it was (shared: the object band; a detached stage: its own params; a detached zone: no undo)
    const url = !own ? `/api/object/${obj()}/band` : (bkey.startsWith('zone:') ? null : `/api/object/${obj()}/stage_band/${bkey}`);
    let prev = null;
    if (url) { try { prev = await H.api('GET', `${url}/params`); } catch (e) { prev = null; } }
    const go = async (replace) => H.api('POST', `/api/object/${obj()}/band/propose`, { frame: f, stage: bkey, replace });
    H.status('Sapiens2: rozpoznaję części ciała…', 'info');
    try {
      let d;
      try { d = await go(false); } catch (e) {
        if (e.status !== 409 || !/zastąpić/.test(e.message || '')) throw e;
        if (!(await V().uiConfirm(`${e.message}`, { ok: 'Zastąp', tone: 'error' }))) return;
        d = await go(true);
      }
      if (prev) B.undo.push({ stage: STAGE_OF(), url, f, what: 'propozycja pasa (Sapiens2)',
                              prev: { band_points: prev.band_points || [], band_ref_out: prev.band_ref_out ?? null, band_ref_in: prev.band_ref_in ?? null } });
      dirty(); await reloadStage(); renderLane(); V().render();
      const sh = d.edge_share || {};
      H.status(`Pas z Sapiens2: ${d.points} pkt w F${f} (włosy ${Math.round((sh.hair || 0) * 100)} %, ubranie ${Math.round((sh.cloth || 0) * 100)} %, skóra ${Math.round((sh.skin || 0) * 100)} % obrysu)`, 'ok');
    } catch (e) { H.toast(`Propozycja pasa: ${e.message || e}`, 7000); }
    if (after) after(); else renderMode();
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
    const g = { sx: p.sx, sy: p.sy, nx: p.nx, ny: p.ny, tx: p.tx == null ? -p.ny : p.tx, ty: p.ty == null ? p.nx : p.ty,
                out: p.out, in: p.in, ang: p.ang || 0 };
    if (B.live) { g.out *= B.live.out; g.in *= B.live.in; }
    if (d && d.kind === 'move') { g.sx = d.x; g.sy = d.y; }
    if (d && d.kind === 'out') { g.out = d.val; g.ang = d.ang; if (d.sym) g.in = d.val; }
    if (d && d.kind === 'in') { g.in = d.val; g.ang = d.ang; if (d.sym) g.out = d.val; }
    return g;
  }
  // the whisker pair's direction: the outline's normal turned by `ang` degrees toward the contour's tangent (2026-10-26);
  // the yellow whisker always stays outside, the blue one inside – the pair only turns round the point, never swaps
  const wdir = (g) => { const a = (g.ang || 0) * Math.PI / 180; const c = Math.cos(a), s = Math.sin(a);
                        return [g.nx * c + g.tx * s, g.ny * c + g.ty * s]; };
  function ends(g) {
    const m = MIN_WHISKER_SCREEN / (V().zoom * plateScale());
    const lo = Math.max(g.out, m), li = Math.max(g.in, m); const [dx, dy] = wdir(g);
    return { o: [g.sx + dx * lo, g.sy + dy * lo], i: [g.sx - dx * li, g.sy - dy * li] };
  }
  function hit(f, p) {
    const s = st(f); if (!s || !pvShown()) return null;
    const tol = 9 / (V().zoom * plateScale());
    const near = (q) => Math.hypot(q[0] - p[0], q[1] - p[1]) <= tol;
    for (const pt of s.points) {
      const g = geom(pt); if (!g) continue;
      const e = ends(g);
      if (near(e.o)) return { kind: 'out', p: pt };
      if (near(e.i)) return { kind: 'in', p: pt };
      if (near([g.sx, g.sy])) return { kind: 'move', p: pt };
    }
    // the whisker line itself (2026-10-26): grabbing anywhere along it moves the point, like grabbing the point
    for (const pt of s.points) {
      const g = geom(pt); if (!g) continue;
      const e = ends(g);
      const ax = e.i[0], ay = e.i[1], bx = e.o[0], by = e.o[1];
      const l2 = (bx - ax) ** 2 + (by - ay) ** 2; if (l2 <= 0) continue;
      const t = Math.max(0, Math.min(1, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / l2));
      if (Math.hypot(p[0] - (ax + t * (bx - ax)), p[1] - (ay + t * (by - ay))) <= tol) return { kind: 'move', p: pt };
    }
    return null;
  }
  const poly = (ctx, k, cnt) => { ctx.beginPath(); cnt.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); };
  // the band preview lines (the stage palette calls it when „👁 podgląd pasa” is on)
  // opts.live (a palette slider is held): one lines request of this frame at a time (the latest values are asked once
  // it returns), and with band points the last lines of this frame stay, dimmed – the browser's live band ignores points
  function drawLines(ctx, img, f, params, opts = {}) {
    if (!on()) return false;
    const k = img.width / ((H.plateSize && H.plateSize()[0]) || img.width);
    const q = Object.entries(params || {}).map(([kk, v]) => `${kk}=${encodeURIComponent(v)}`).join('&');
    let L = B.lines[`${key(f)}|${q}`];
    let dim = false;
    if (L && L.error) return false;   // the server refused: nothing to draw, lineNote(f, params) says why
    if (!L) {
      const pre = `${key(f)}|`;
      if (!(opts.live && Object.keys(B.busy).some((b) => b.startsWith(pre)))) loadLines(f, params);
      L = opts.live && pts(f).length ? B.lastLines[key(f)] : null;
      if (!L) return false;
      dim = true;
    }
    B.lastLines[key(f)] = L;
    strokeLines(ctx, k, L, { alpha: dim ? 0.5 : null });
    return true;
  }
  // ONE drawing of a band preview (stages 3 / 4 / 5 / 7, 2026-10-24): `L` = a lines dict of the server (band_contours /
  // inset_feather_contours / feather_contours in plate px, engine/bandsys.preview_lines), `k` = plate -> image scale:
  // dashed band edges + thin dashes where the feathers end, no fill (user decision 2026-10-21)
  function strokeLines(ctx, k, L, opts = {}) {
    const v = V();
    ctx.save();
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    try {
      ctx.strokeStyle = 'rgba(255,212,0,0.95)'; ctx.lineWidth = 1.6 / v.zoom; ctx.setLineDash([6 / v.zoom, 4 / v.zoom]);
      for (const c of L.band_contours || []) { poly(ctx, k, c); ctx.stroke(); }
      ctx.strokeStyle = 'rgba(255,212,0,0.8)'; ctx.lineWidth = 0.8 / v.zoom; ctx.setLineDash([3 / v.zoom, 3 / v.zoom]);
      for (const c of [...(L.inset_feather_contours || []), ...(L.feather_contours || [])]) { poly(ctx, k, c); ctx.stroke(); }
    } finally { ctx.restore(); }
  }
  // the exact mask 1+2 of the work area is computed once in the background (pipeline.mask12_warm); the previews shown
  // on the stand-in mask register here and are refreshed once, when GET …/mask12_status says ready (one poll, 5 s)
  const W = { timer: null, cbs: [], obj: null };
  function watchMask12(cb) {
    if (!obj()) return;
    if (W.obj !== obj()) { W.cbs = []; W.obj = obj(); }
    if (cb && !W.cbs.includes(cb)) W.cbs.push(cb);
    if (W.timer) return;
    const tick = async () => {
      W.timer = null;
      if (!obj()) { W.cbs = []; return; }
      let st = null;
      try { st = await H.api('GET', `/api/object/${obj()}/mask12_status`); } catch (e) { st = null; }
      if (st && st.ready) { const cbs = W.cbs; W.cbs = []; cbs.forEach((c) => { try { c(); } catch (e) { /* a stale view */ } }); return; }
      if (st && !st.running && !st.ready) { W.cbs = []; return; }   // nothing is being computed (an error was logged): stop
      W.timer = setTimeout(tick, 5000);
    };
    W.timer = setTimeout(tick, 5000);
  }
  // why drawLines(f, params) drew nothing: the server's refusal of the lines ('' = still loading)
  function lineNote(f, params) {
    const q = Object.entries(params || {}).map(([kk, v]) => `${kk}=${encodeURIComponent(v)}`).join('&');
    const L = B.lines[`${key(f)}|${q}`];
    if (L && L.error) return `Podgląd pasa F${f}: ${L.error}`;
    return L && L.approx ? `Podgląd pasa F${f}: przybliżony – liczę dokładną maskę 1+2 obszaru roboczego (raz, w tle)…` : '';
  }
  function draw(ctx, img, k, f) {
    if (!on() || !pvShown()) return;
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
  // a band-point tool picked while the stage's band preview is off switches the preview on (user request 2026-10-24)
  function onTool() {
    if (tool() && !pvShown()) {
      const cb = $(isZone() ? 'zpBandPv' : { matanyone: 'ePv', videomama: 'vmPvBand', ltx: 'lxPvBand' }[STAGE_OF()]);
      if (cb && !cb.checked) { cb.checked = true; cb.dispatchEvent(new Event('change', { bubbles: true })); }
    }
    V().canvas.style.cursor = tool() ? 'crosshair' : ''; info(); V().render(); }
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
      B.drag = { ...ht, pid: e.pointerId, f, x0: p[0], y0: p[1], x: p[0], y: p[1], val: ht.kind === 'out' ? ht.p.out : ht.kind === 'in' ? ht.p.in : 0, ang: ht.p.ang || 0 };
    }, true);
    cv.addEventListener('pointermove', (e) => {
      const d = B.drag; if (!d || e.pointerId !== d.pid) return;
      const p = H.canvasToPlate(e); if (!p) return;
      if (d.kind === 'move') { d.x = d.p.sx + (p[0] - d.x0); d.y = d.p.sy + (p[1] - d.y0); }
      else {
        // polar drag (2026-10-26): the end follows the cursor – distance = the whisker's length, direction = the tilt of the
        // whole pair round the point (the other whisker stays collinear); within 4° of the normal it snaps back to 0
        d.sym = !!e.shiftKey; const sg = d.kind === 'out' ? 1 : -1;
        const vx = sg * (p[0] - d.p.sx), vy = sg * (p[1] - d.p.sy);
        const tx = d.p.tx == null ? -d.p.ny : d.p.tx, ty = d.p.ty == null ? d.p.nx : d.p.ty;
        let a = Math.atan2(vx * tx + vy * ty, vx * d.p.nx + vy * d.p.ny) * 180 / Math.PI;
        a = Math.max(-80, Math.min(80, a)); if (Math.abs(a) < 4) a = 0;
        d.ang = Math.round(a * 10) / 10;
        d.val = Math.max(0, Math.round(Math.hypot(vx, vy)));
      }
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
        const an = d.ang ? `, kąt ${d.ang}°` : '';
        if (d.sym) save(d.f, { id: d.p.id, out: d.val, in: d.val, ang: d.ang }, `${d.p.id}: oba wąsy ${d.val} px (symetrycznie${an}, klucz)`);
        else save(d.f, { id: d.p.id, [d.kind]: d.val, ang: d.ang }, `${d.p.id}: ${tr(side)} ${d.val} px${an} (klucz)`);
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
    // the band keys of a frame arrive asynchronously (load): 🔑✕ / Delete must follow them (user report 2026-10-26: the key of
    // F40 could not be deleted – the button was computed before the points were loaded and stayed disabled)
    if (H && H.updateDelKey) H.updateDelKey();
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
  function reset() { B.data = {}; B.lines = {}; B.lastLines = {}; B.undo = []; B.live = null; B.drag = null; renderLane(); }

  function init(hooks) {
    H = hooks;
    wire(V().canvas);
    window.addEventListener('md-frame', onFrame);
    window.MD_BANDPTS.ready = true;
  }

  // ---- ONE band palette (user decision 2026-10-24): every band stage (3 / 4 / 7; the stage-5 zone keeps its own
  // rows with the same ranges) shows the same four sliders in the same order, range and wording; a stage keeps only
  // its element ids and its model name. `slider(kind, model)` = the attribute strings for the stage's own <input id=…>
  const SLIDER_ORDER = ['out', 'in', 'feather', 'outFeather'];
  const SLIDERS = {
    out: [120, 400, 'Pas na zewnątrz', (m) => `Jak daleko poza krawędź maski bazowej sięga pas (zewnętrzna przerywana linia), w którym ${m} decyduje o alfie (kosmyki, rozmycie). Punkty pasa kształtują go wzdłuż obrysu, suwak skaluje cały pas – tak samo w etapach 3 / 4 / 5 / 7. Suwak 0–120 px, w polu do 400`],
    in: [40, 200, 'Wcięcie w rdzeń', (m) => `Ile px w głąb maski bazowej oddajemy ${m} (wewnętrzna przerywana linia pasa); dalej w środku alfa = 1. Punkty pasa kształtują je wzdłuż obrysu – tak samo w etapach 3 / 4 / 5 / 7. Suwak 0–40 px, w polu do 200`],
    feather: [40, 200, 'Wtopienie wcięcia', (m) => `Na ilu px od wewnętrznej krawędzi pasa rdzeń przechodzi w wynik ${m} (cieńsza przerywana linia); nie nachodzi na wtopienie krawędzi zewnętrznej. 0 = twarde przejście`],
    outFeather: [80, 200, 'Wtopienie krawędzi zewn.', (m) => `Na ilu px od ZEWNĘTRZNEJ krawędzi pasa wynik ${m} przechodzi w rdzeń (zwykle czerń) – cieńsza przerywana linia. Nie nachodzi na wtopienie wcięcia. 0 = twarde cięcie`],
  };
  function slider(kind, model) {
    const [smax, nmax, label, tip] = SLIDERS[kind];
    const rt = `${label} (px) – podgląd na żywo, zapis po puszczeniu`;
    const nt = `${label} (px) – wpisz i Enter`;
    return { label, title: tip(model),
             range: `type="range" min="0" max="${smax}" step="1" title="${rt}"`,
             num: `type="number" min="0" max="${nmax}" step="1" class="form-input edge-num" title="${nt}"` };
  }
  window.MD_BANDPTS = { init, draw, drawLines, strokeLines, lineNote, onTool, slider, SLIDER_ORDER, pointsAt: (f) => pts(f), watchMask12, onFrame, reset, dirty, undo, hasKeyAt, deleteKeyAt, copyFrom, setLive,
                        renderLane, info, stage: STAGE_OF, renderMode, renderModeInto, ready: false };
})();
