// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/mergematte.js – floating palette „MergeMatte” of stage Final Touch (PLAN_FINAL_TOUCH.md §11, user decisions
// 2026-10-29: own palette on the RIGHT of the view, knobs = the Nuke group's knobs). The settings never change the alpha:
// they are the comp the Nuke export writes into the MergeMatte group. Preview = the server's port of the group
// (engine/merge_matte.py, GET merge_matte/frame/{f}) of the visible region – in real time on the GPU (static/mm_gpu.js,
// the same chain, ±1 / 255 against the port) with the server as fallback – drawn over the Comp view (app.js drawBgCompare)
// on the same background (colour / checker / target plate „bg”); the checkbox „wł.” = A/B against the plain comp.
(function () {
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  let H = null;                                   // hooks of objects.js (via finaltouch.js)
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const M = { obj: null, info: null, knobs: null, on: false, rev: 0, putTimer: null, fetchTimer: null, seq: 0,
              shown: null, pending: null, note: '' };
  // section titles of the knob table's text knobs (the node's own English headings stay as tooltips)
  const SECTIONS = { t_alpha: 'Alfa', t_decon: 'Dekontaminacja', t_darken: 'Przyciemnienie krawędzi', t_match: 'Dopasowanie krawędzi',
                     t_wrap: 'Light wrap', t_bleed: 'Przenikanie koloru tła', t_edge: 'Rozmycie krawędzi', div: 'Całość' };
  const STEP = (lo, hi) => (hi - lo > 20 ? 0.5 : hi - lo > 4 ? 0.1 : 0.01);
  const inStage = () => !!(H && H.S.obj && !H.S.auxMode && H.S.stage === H.ST.FINAL_TOUCH);
  const pos = {
    get: () => { try { return JSON.parse(localStorage.getItem('md_mm_palette') || 'null'); } catch (e) { return null; } },
    set: (q) => { try { if (q) localStorage.setItem('md_mm_palette', JSON.stringify(q)); else localStorage.removeItem('md_mm_palette'); } catch (e) { /* storage blocked */ } },
  };

  function paletteHtml() {
    return `<div class="pts-palette mm-palette" id="mmPalette" style="display:none">
      <div class="pp-head" title="MergeMatte = węzeł Nuke (Merge over z obróbką krawędzi). Ustawienia idą do eksportu .nk; alfa eksportu się nie zmienia. Przeciągnij, żeby przesunąć (dwuklik = z powrotem w rogu)"><span class="ol-grip">⠿</span> MergeMatte <span class="muted">v01</span></div>
      <div class="pp-chk"><label class="chk" title="Podgląd MergeMatte w widoku Comp (wyłącz = zwykłe wycięcie, A/B)"><input type="checkbox" id="mmOn"> wł.</label>
        <button type="button" class="btn btn-ghost btn-xs" id="mmPresetDef" title="Wszystkie knoby = domyślne węzła">↺ domyślne</button>
        <button type="button" class="btn btn-ghost btn-xs" id="mmPresetSd" title="Zestaw startowy dla włosów na jasnym / kontrastowym tle: choke 6, decon 100 px, darken 23 px, match, wrap, flare, edge blur 2">preset włosy</button>
        <button type="button" class="btn btn-ghost btn-xs" id="mmNuke" title="Węzeł MergeMatte z tymi ustawieniami do schowka – w Nuke'u Ctrl+V w grafie węzłów (wejścia: FG = plate z alfą, BG = tło). Zapisany też jako plik .nk w folderze obiektu">⧉ Nuke</button></div>
      <div class="muted pp-hint" id="mmNote"></div>
      <div id="mmKnobs"></div>
    </div>`;
  }
  function fmt(v) { return Math.abs(v) >= 10 ? (+v).toFixed(1) : (+v).toFixed(2); }
  function renderKnobs() {
    const box = $('mmKnobs'); if (!box || !M.info) return;
    let html = '', open = false;
    for (const s of M.info.specs) {
      if (s.range === null) {
        if (open) html += '</div>';
        html += `<div class="pp-sec"><div class="zp-sub" title="${esc(s.label.replace(/<[^>]+>/g, ''))}">${esc(SECTIONS[s.name] || s.name)}</div>`;
        open = true;
      } else if (s.range === 'bool') {
        html += `<label class="chk" title="${esc(s.label)}${s.tip ? ' – ' + esc(s.tip) : ''}"><input type="checkbox" data-mm="${s.name}"> ${esc(s.name)}</label>`;
      } else {
        const [lo, hi] = s.range;
        const [flo, fhi] = s.field || s.range;   // a typed value may pass the slider (as in Nuke)
        html += `<label class="zone-slider" title="${esc(s.label)}${s.tip ? ' – ' + esc(s.tip) : ''}${M.info.approx.includes(s.name) ? ' – podgląd przybliżony (FilterErode z Nuke nie jest jeszcze odwzorowany 1:1)' : ''}">${esc(s.name)}${M.info.approx.includes(s.name) ? '*' : ''}
          <input type="range" data-mm="${s.name}" min="${lo}" max="${hi}" step="${STEP(lo, hi)}" style="width:100%"><input type="number" data-mmn="${s.name}" min="${flo}" max="${fhi}" step="0.1" class="form-input edge-num" title="${esc(s.name)} – wpisz i Enter (suwak ${lo} … ${hi}, w polu ${flo} … ${fhi}); strzałki ↑ / ↓ = ±0,1, po wpisaniu setnych ±0,01"></label>`;
      }
    }
    if (open) html += '</div>';
    box.innerHTML = html;
    box.querySelectorAll('[data-mm]').forEach((el) => {
      const n = el.dataset.mm;
      el.addEventListener('input', () => {
        M.knobs[n] = el.type === 'checkbox' ? (el.checked ? 1 : 0) : +el.value;
        const b = box.querySelector(`[data-mmn="${n}"]`); if (b) b.value = fmt(M.knobs[n]);
        M.rev++; schedulePut({ knobs: { [n]: M.knobs[n] } });
      });
      el.addEventListener('dblclick', () => {   // double click = the node's default
        const d = M.info.presets.default[n]; M.knobs[n] = d; syncValues(); M.rev++; schedulePut({ knobs: { [n]: d } });
      });
    });
    // typed values (like the band palette): Enter / leaving the field, clamped to the knob's range. Arrows (the field's
    // and the keyboard's ↑ / ↓) step 0.1; once hundredths (or finer) are typed they step by that digit (user request
    // 2026-10-30) – the preview follows every step at once
    const typedStep = (txt) => { const m = String(txt).match(/[.,](\d+)/); const d = m ? m[1].length : 0; return d >= 2 ? 10 ** -d : 0.1; };
    box.querySelectorAll('[data-mmn]').forEach((el) => {
      const n = el.dataset.mmn;
      const commit = (live) => {
        const v = parseFloat(String(el.value).replace(',', '.'));
        if (!Number.isFinite(v)) { if (!live) el.value = fmt(M.knobs[n]); return; }
        const c = Math.min(+el.max, Math.max(+el.min, +v.toFixed(6)));
        if (c === M.knobs[n]) { if (!live) el.value = fmt(c); return; }
        M.knobs[n] = c; syncValues(); M.rev++; schedulePut({ knobs: { [n]: c } });
      };
      el.addEventListener('change', () => commit(false));
      el.addEventListener('input', (e) => {
        if (e.inputType && e.inputType !== 'insertReplacementText') el.step = String(typedStep(el.value));   // typed: its precision
        commit(true);
      });
      el.addEventListener('blur', () => { el.step = '0.1'; });
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); el.blur(); } e.stopPropagation(); });
    });
    syncValues();
  }
  function syncValues() {
    const box = $('mmKnobs'); if (!box || !M.knobs) return;
    box.querySelectorAll('[data-mm]').forEach((el) => {
      const v = M.knobs[el.dataset.mm];
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
      const b = box.querySelector(`[data-mmn="${el.dataset.mm}"]`); if (b && document.activeElement !== b) b.value = fmt(v);
    });
    $('mmOn').checked = !!M.on;
  }
  let putBody = {};
  function schedulePut(body) {
    // a knob / preset change while the palette is off switches it on – otherwise the Comp view shows the plain cut-out
    // and the slider seems dead (user report 2026-10-30); only the checkbox itself turns it off
    if (body.on === undefined && !M.on) { body = { ...body, on: true }; M.on = true; const c = $('mmOn'); if (c) c.checked = true; note(); }
    putBody = { ...putBody, ...body, knobs: { ...(putBody.knobs || {}), ...(body.knobs || {}) } };
    if (body.knobs) refresh();   // the GPU preview follows a drag / typed value at once (the server save is debounced)
    clearTimeout(M.putTimer);
    M.putTimer = setTimeout(async () => {
      const b = putBody; putBody = {};
      const name = M.obj;
      try {
        const r = await H.api('PUT', `/api/object/${encodeURIComponent(name)}/merge_matte`, b);
        if (name === M.obj && !Object.keys(putBody).length) { M.knobs = r.knobs; M.on = r.on; syncValues(); }
        M.rev++; refresh();
      } catch (e) { H.status(`MergeMatte: ${e.message || e}`, 'err'); }
    }, 180);
  }
  async function load() {
    const name = H && H.S.obj; if (!name) return;
    if (name !== M.obj) { M.shown = null; M.pending = null; }
    M.obj = name;
    try {
      const r = await H.api('GET', `/api/object/${encodeURIComponent(name)}/merge_matte`);
      if (name !== M.obj) return;
      M.info = r; M.knobs = r.knobs; M.on = r.on;
      renderKnobs(); render(); refresh();
    } catch (e) { M.info = null; }
  }
  function render() {
    const box = $('mmPalette'); if (!box) return;
    const show = inStage() && !!(V() && V().projectLoaded) && !!M.info;
    box.style.display = show ? '' : 'none';
    if (show && H.placeFloating) H.placeFloating(box, $('canvasWrapper'), pos.get());
    note();
  }
  function note() {
    const n = $('mmNote'); if (!n) return;
    const v = V();
    const comp = (v && v.isBgRef && v.isBgRef() && (v.compare.mode === 'comp' || v.compare.mode === 'wipe' || v.compare.holdC))
      || !!(H && H.compView && H.compView());
    n.className = 'muted pp-hint' + (M.on && !comp ? ' warn' : '');
    n.textContent = !M.on ? 'Wyłączony – widok Comp pokazuje zwykłe wycięcie.'
      : !comp ? 'Podgląd jest w widoku Comp (zakładka Comp albo tło po prawej: kolor / szachownica / BG).'
        : M.note || '';
  }
  // ------------------------------------------------------------------------------------------------ preview
  function bgSpec(v) { return v.compare.ref === 'bg' ? 'bg' : (v.compare.bg.checker ? 'checker' : v.compare.bg.color); }
  function wantedRegion(v, img) {
    const pw = v.projectData.plate_w, ph = v.projectData.plate_h;
    const kp = v.plateScale(img);
    if (v.zoom > 1.0 && v.tileRegionFor) {
      const r = v.tileRegionFor(img);
      if (r && r.w * r.h < pw * ph * 0.9) return { x: r.x, y: r.y, w: r.w, h: r.h, kp };
    }
    return { x: 0, y: 0, w: pw, h: ph, kp };
  }
  function keyOf(f, reg, bg) { return `${M.obj}|${f}|${reg.x},${reg.y},${reg.w},${reg.h}|${bg}|${M.rev}|${(window.MD_FT && window.MD_FT.state && window.MD_FT.state.info && window.MD_FT.state.info.sig) || ''}`; }
  function refresh() { const v = V(); if (v && v.render) v.render(); }
  function fetchPreview(f, reg, bg, key) {
    clearTimeout(M.fetchTimer);
    M.fetchTimer = setTimeout(async () => {
      const seq = ++M.seq;
      M.pending = key; M.note = 'Liczę podgląd MergeMatte…'; note();
      const ftInfo = window.MD_FT && window.MD_FT.state && window.MD_FT.state.info;
      const b = ftInfo && ftInfo.base && ftInfo.base.run ? `&b=${encodeURIComponent(ftInfo.base.run)}` : '';
      const url = `/api/object/${encodeURIComponent(M.obj)}/merge_matte/frame/${f}?x=${reg.x}&y=${reg.y}&w=${reg.w}&h=${reg.h}&bg=${encodeURIComponent(bg)}${b}&k=${M.rev}`;
      const t0 = performance.now();
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || resp.statusText);
        const s = +resp.headers.get('X-MM-Scale') || 1;
        const blob = await resp.blob();
        const img = await createImageBitmap(blob);   // (Image.decode() of a blob URL never settled in the app's browser pane)
        if (seq !== M.seq) { img.close(); return; }
        if (M.shown && M.shown.img && M.shown.img.close) M.shown.img.close();
        M.shown = { key, f, reg, img };
        M.pending = null;
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        M.note = `Podgląd: ${reg.w}×${reg.h} px${s < 1 ? ` liczony w ${Math.round(s * 100)} % (przybliżenie – przybliż widok)` : ' w pełnej rozdzielczości'} · ${secs} s${M.knobs && M.knobs.choke ? ' · choke* przybliżony' : ''}`;
        note(); refresh();
      } catch (e) {
        if (seq !== M.seq) return;
        M.pending = null; M.note = `Błąd podglądu: ${e.message || e}`; note();
      }
    }, 250);
  }
  // app.js drawBgCompare / objects.js Comp view: after the plain comp was drawn (plate space of the proxy image), draw MergeMatte over it
  // objects.js passes the background of the Comp view tab (`bgOf`); app.js none = the compare background
  // GPU (static/mm_gpu.js, real time – user request 2026-10-30): the whole proxy frame from the matte the Comp view shows,
  // px knobs scaled to proxy px like the server's preview_region; cached per frame / matte / background / knob revision.
  // The server stays the fallback (no WebGL2) and refines a zoomed-in region when the proxy is smaller than the plate.
  const PX_KNOBS = ['choke', 'soften', 'decon_spread', 'darken_width', 'match_width', 'wrap_size', 'bleed_size', 'edge_blur'];
  function gpuOver(ctx, img, f, bg, kp) {
    if (!(window.mdMmGpu && window.mdMmGpu.on)) return false;
    const m = window.MD_OBJ && window.MD_OBJ.matteFor ? window.MD_OBJ.matteFor(f) : null;
    if (!m || m === 'loading' || !m.img) return true;   // no matte yet: the plain comp stays, no server request either
    const v = V();
    let bgo;
    if (bg === 'bg') {
      const bi = v.getImage('bg', f, false);
      if (!bi) { v.loadFrame(f, () => { if (v.currentFrame === f) v.render(); }, 'bg', false); return true; }
      bgo = { mode: 'image', img: bi };
    } else if (bg === 'checker') bgo = { mode: 'checker', cell: 16 };
    else { const h = String(bg).replace('#', ''); bgo = { mode: 'color', color: [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16) / 255) }; }
    const W = img.width, H = img.height;
    const key = `${M.obj}|${f}|${m.key || ''}|${bg}|${M.rev}|${W}x${H}`;
    let c = M.gpu && M.gpu.key === key && M.gpu.img === m.img && M.gpu.plate === img ? M.gpu.c : null;
    if (!c) {
      const k = { ...M.knobs };
      for (const n of PX_KNOBS) k[n] = k[n] / kp;
      const t0 = performance.now();
      c = window.mdMmGpu.render({ plate: img, alpha: m.img, bg: bgo, knobs: k, W, H, into: M.gpu && M.gpu.c });
      if (!c) return false;
      M.gpu = { key, img: m.img, plate: img, c };
      M.note = `Podgląd na GPU: ${W}×${H} px · ${Math.round(performance.now() - t0)} ms${M.knobs.choke ? ' · choke* przybliżony' : ''}`;
      note();
    }
    ctx.drawImage(c, 0, 0, W, H);
    return true;
  }
  function drawOver(ctx, img, bgOf) {
    const v = V();
    if (!M.on || !inStage() || !M.info || !v || !v.projectData) return;
    const f = v.currentFrame, bg = bgOf || bgSpec(v), reg = wantedRegion(v, img);
    if (gpuOver(ctx, img, f, bg, reg.kp) && !(reg.kp > 1.01 && v.zoom > 1.0)) return;   // proxy = plate detail: GPU only
    const key = keyOf(f, reg, bg);
    if ((!M.shown || M.shown.key !== key) && M.pending !== key) fetchPreview(f, reg, bg, key);
    const sh = M.shown;
    if (!sh || sh.f !== f || !sh.key.startsWith(`${M.obj}|${f}|`)) return;   // other frame / object: the plain comp stays
    const kp = reg.kp;
    ctx.drawImage(sh.img, sh.reg.x / kp, sh.reg.y / kp, sh.reg.w / kp, sh.reg.h / kp);
  }
  // „⧉ Nuke”: the object's MergeMatte group as Nuke clipboard text (pending slider saves go first)
  async function copyNode() {
    const name = M.obj; if (!name) return;
    try {
      if (Object.keys(putBody).length) { clearTimeout(M.putTimer); const b = putBody; putBody = {}; await H.api('PUT', `/api/object/${encodeURIComponent(name)}/merge_matte`, b); }
      const r = await H.api('GET', `/api/object/${encodeURIComponent(name)}/merge_matte/nk`);
      let ok = false;
      try { await navigator.clipboard.writeText(r.text); ok = true; } catch (e) {
        const t = document.createElement('textarea'); t.value = r.text; document.body.appendChild(t); t.select();
        try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
        t.remove();
      }
      H.status(ok ? `Węzeł ${r.node} w schowku – w Nuke'u Ctrl+V w grafie węzłów. Plik: ${r.path}` : `Schowek niedostępny – plik węzła: ${r.path}`, ok ? 'ok' : 'warn');
    } catch (e) { H.status(`MergeMatte → Nuke: ${e.message || e}`, 'err'); }
  }
  function init(hooks) {
    H = hooks;
    const cw = $('canvasWrapper'); if (!cw) return;
    const box = H.el(paletteHtml());
    cw.appendChild(box);
    if (H.makeFloating) H.makeFloating(box, cw, '.pp-head', pos);
    $('mmOn').addEventListener('change', () => { M.on = $('mmOn').checked; M.rev++; schedulePut({ on: M.on }); note(); refresh(); });
    $('mmPresetDef').addEventListener('click', () => schedulePut({ reset: 'default' }));
    $('mmPresetSd').addEventListener('click', () => schedulePut({ reset: 'sh010' }));
    $('mmNuke').addEventListener('click', () => copyNode());
    window.addEventListener('md-frame', () => note());
    window.MD_MM.ready = true;
  }
  window.MD_MM = {
    init, load, render, drawOver,
    onShow() { load(); },
    onStage() { if (inStage() && H.S.obj !== M.obj) load(); else render(); },
    get state() { return M; },
  };
})();
