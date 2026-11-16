// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Cleanup 3D – horizon match in C5 „Projekcja” (docs/PLAN_CLEANUP_3D.md, package P15; engine/cleanup3d/horizon.py).
// The section „Horyzont” of the C5 palette (its HTML lives in static/cleanup3d.js palettesHtml):
//   „Dopasuj horyzont”          render_settings.horizon_match (the render and the live preview move the projected keys
//                               onto the plate's horizon – a vertical shift that fades out over „Zasięg dopasowania”)
//   „Zasięg dopasowania (px)”   render_settings.horizon_fade_px
//   „✎ Narysuj horyzont…”       the user's line of the current frame, edited with THE curve editor (MD_CURVES.Editor,
//                               static/curves.js – no own curve code); saved on every finished edit
//                               (PUT /api/cleanup3d/horizon/{f}); 🗑 removes it (DELETE). Between drawn frames the
//                               render interpolates, elsewhere the automatic detector works.
//   status                      GET /api/cleanup3d/horizon/{f}: offset of the projection vs the plate and the step
//                               where the object crosses the horizon, before → after the correction.
// The lines (plate's horizon, projection's horizon, the user's) are drawn on an own canvas over the viewer, which
// follows its pan / zoom (the same canvas classes as the area editor: c3-area-canvas / c3-area-on).
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const V = () => window.viewer;
  const C3 = () => window.MD_CLEANUP3D;
  const CV = () => window.MD_CURVES;
  const api = (m, u, b) => window.UI.api(m, u, b);
  const toast = (t, ms) => window.UI.toast(t, ms);
  const DEF = { horizon_match: false, horizon_fade_px: 120, horizon_search_px: 60 };   // engine/cleanup3d/horizon.py DEFAULTS
  const SRC = { auto: 'automat', user: 'linia narysowana', user_near: 'najbliższa narysowana linia' };
  const HZ = { ready: false, s: null, info: null, drawing: false, curve: null, canvas: null, ed: null, saveT: null,
    loadT: null, seq: 0, drawn: [] };

  const on = () => document.body.classList.contains('c3-on') && !!(C3() && C3().ready && C3().step === 'project' && V());
  const frame = () => (V() ? V().currentFrame : 0);
  function settings() {
    const rs = (C3() && C3().state && C3().state.render_settings) || {};
    const s = { ...DEF };
    for (const k of Object.keys(DEF)) if (rs[k] != null) s[k] = rs[k];
    return HZ.s ? { ...s, ...HZ.s } : s;
  }
  function plateSize() {
    const v = V(); const m = (v && (v.projectData || v.manifest)) || {};
    if (m.plate_w && m.plate_h) return [+m.plate_w, +m.plate_h];
    const img = v && v.getCurrentImage && v.getCurrentImage();
    return img && img.width ? [img.width, img.height] : null;
  }

  // ------------------------------------------------------------------------------------------------ palette
  function renderUi() {
    if (!$('c3HzOn')) return;
    const s = settings();
    $('c3HzOn').checked = !!s.horizon_match;
    if (document.activeElement !== $('c3HzFade')) $('c3HzFade').value = s.horizon_fade_px;
    $('c3HzFadeV').textContent = `${Math.round(+$('c3HzFade').value)} px`;
    $('c3HzDraw').classList.toggle('active', HZ.drawing);
    $('c3HzDel').disabled = !HZ.drawn.includes(frame());
    const d = $('c3HzDrawn');
    if (d) d.textContent = HZ.drawn.length ? `Linie narysowane w klatkach: ${HZ.drawn.join(', ')}` : 'Brak narysowanych linii – działa automat';
  }
  function infoText(i) {
    if (!i) return '';
    if (i.status === 'no_horizon') return `F${i.frame}: brak horyzontu kart w tej klatce`;
    if (i.status === 'no_plate_line') return `F${i.frame}: automat nie znalazł horyzontu w plate'cie – narysuj go w tej klatce`;
    if (i.status === 'no_fill_line') return `F${i.frame}: rzut kluczy nie pokazuje horyzontu`;
    const src = SRC[i.source] || i.source || '';
    const step = i.step_px == null ? 'obiekt nie przecina horyzontu' : `schodek ${i.step_px} → ${i.step_after_px} px`;
    return `F${i.frame}: przesunięcie rzutu ${i.offset_px} px (${src}) · ${step}`;
  }
  async function saveSettings(patch) {
    HZ.s = { ...(HZ.s || {}), ...patch };
    renderUi();
    clearTimeout(HZ.saveT);
    HZ.saveT = setTimeout(async () => {
      try {
        const r = await api('PUT', '/api/cleanup3d/render_settings', settings());
        const c3 = C3(); if (c3 && c3.state && r) c3.state.render_settings = r.render_settings;
        HZ.s = null;
      } catch (e) { toast(e.message || String(e), 6000); return; }
      if (C3() && C3().view === 'blend') C3().setView('blend', true);   // the live preview with the new correction
    }, 300);
  }

  // ------------------------------------------------------------------------------------------------ measurement
  function load(quiet) {
    clearTimeout(HZ.loadT);
    HZ.loadT = setTimeout(async () => {
      if (!on()) return;
      const f = frame(); const seq = ++HZ.seq;
      const info = $('c3HzInfo'); if (info && !quiet) info.textContent = `Horyzont F${f}: pomiar…`;
      try {
        const d = await api('GET', `/api/cleanup3d/horizon/${f}`);
        if (seq !== HZ.seq) return;
        HZ.info = d; HZ.drawn = d.drawn || [];
        if (info) info.textContent = infoText(d);
        if (HZ.drawing && d.frame === frame()) HZ.curve = d.user ? JSON.parse(JSON.stringify(d.user)) : startCurve(d);
      } catch (e) {
        if (seq !== HZ.seq) return;
        HZ.info = null;
        if (info) info.textContent = '';
        if (!quiet) toast(e.message || String(e), 6000);
      }
      renderUi(); redraw();
    }, quiet ? 250 : 0);
  }
  // the first line of a frame: the plate's horizon as measured (else the cards' one) through a few points
  function startCurve(d) {
    const src = (d.plate && d.plate.length > 1) ? d.plate : (d.predicted || []);
    if (src.length < 2) { const ps = plateSize() || [1920, 1080]; return CV().fromPoints([[0, ps[1] / 2], [ps[0] - 1, ps[1] / 2]], false); }
    const n = Math.min(6, src.length); const pts = [];
    for (let i = 0; i < n; i++) pts.push(src[Math.round(i * (src.length - 1) / (n - 1))].slice());
    return CV().fromPoints(pts, false);
  }
  async function saveLine() {
    if (!HZ.curve) return;
    const f = frame();
    try {
      const r = await api('PUT', `/api/cleanup3d/horizon/${f}`, { curve: HZ.curve });
      HZ.drawn = r.drawn || HZ.drawn;
    } catch (e) { toast(e.message || String(e), 6000); return; }
    renderUi();
    load(true);
    if (settings().horizon_match && C3() && C3().view === 'blend') C3().setView('blend', true);
  }
  async function deleteLine() {
    const f = frame();
    try {
      const r = await api('DELETE', `/api/cleanup3d/horizon/${f}`);
      HZ.drawn = r.drawn || [];
    } catch (e) { toast(e.message || String(e), 6000); return; }
    HZ.curve = null;
    renderUi();
    load(false);
    if (settings().horizon_match && C3() && C3().view === 'blend') C3().setView('blend', true);
  }
  function setDrawing(v) {
    HZ.drawing = !!v && on();
    if (HZ.drawing) { HZ.curve = null; load(false); }
    else HZ.curve = null;
    if (HZ.canvas) HZ.canvas.classList.toggle('c3-area-on', HZ.drawing);
    renderUi(); redraw();
  }

  // ------------------------------------------------------------------------------------------------ canvas over the viewer
  function geom() {
    const v = V(); const img = v && v.getCurrentImage && v.getCurrentImage(); const ps = plateSize();
    if (!img || !ps || !img.width) return null;
    return { kx: (img.width / ps[0]) * v.zoom, ky: (img.height / ps[1]) * v.zoom, px: v.panX || 0, py: v.panY || 0 };
  }
  const toScreen = (p) => { const g = geom(); return g ? [g.px + p[0] * g.kx, g.py + p[1] * g.ky] : [0, 0]; };
  const toImage = (sx, sy) => { const g = geom(); return g ? [(sx - g.px) / g.kx, (sy - g.py) / g.ky] : [0, 0]; };
  function fitCanvas() {
    const c = HZ.canvas; const vc = V() && V().canvas; if (!c || !vc) return;
    if (c.width !== vc.width || c.height !== vc.height) { c.width = vc.width; c.height = vc.height; }
    const st = c.style;
    st.width = `${vc.offsetWidth || vc.width}px`; st.height = `${vc.offsetHeight || vc.height}px`;
    st.left = `${vc.offsetLeft || 0}px`; st.top = `${vc.offsetTop || 0}px`;
  }
  function thin(pts, n) {
    if (!pts || pts.length < 2) return null;
    const out = []; const k = Math.max(1, Math.floor(pts.length / n));
    for (let i = 0; i < pts.length; i += k) out.push(pts[i]);
    if (out[out.length - 1] !== pts[pts.length - 1]) out.push(pts[pts.length - 1]);
    return CV().fromPoints(out, false);
  }
  // layers for MD_CURVES.Editor: the measured lines (not editable) + the user's line while drawing
  function layers() {
    const out = []; const i = HZ.info;
    const show = HZ.drawing || ($('c3HzLines') && $('c3HzLines').checked);
    if (!on() || !show) return out;
    if (i && i.frame === frame()) {
      const pl = i.source !== 'user' ? thin(i.plate, 60) : null;
      if (pl) out.push({ id: '+plate', spline: pl, color: '#4ae0ff', dash: [6, 4], editable: false });
      const fl = thin(i.fill, 60);
      if (fl) out.push({ id: '+fill', spline: fl, color: '#ff9f43', dash: [2, 3], editable: false });
    }
    if (HZ.drawing && HZ.curve) out.push({ id: 'hz', spline: HZ.curve, color: '#ffd84a', editable: true });
    else if (i && i.user && i.frame === frame()) out.push({ id: '+user', spline: i.user, color: '#ffd84a', editable: false });
    return out;
  }
  function redraw() {
    if (!HZ.canvas || !HZ.ed) return;
    const act = on() && (HZ.drawing || ($('c3HzLines') && $('c3HzLines').checked));
    HZ.canvas.style.display = act ? '' : 'none';
    if (!act) return;
    fitCanvas(); HZ.ed.draw();
  }
  function onWheel(e) {
    if (e.shiftKey) return;                                   // Shift + wheel = the curve editor's soft radius
    const vc = V() && V().canvas; if (!vc) return;
    e.preventDefault();
    vc.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: e.clientX, clientY: e.clientY, deltaY: e.deltaY }));
  }

  // ------------------------------------------------------------------------------------------------ wiring
  function init() {
    if (HZ.ready) return true;
    const v = V(); const c3 = C3(); const cw = document.getElementById('canvasWrapper');
    if (!v || !c3 || !c3.ready || !CV() || !cw || !$('c3HzOn')) return false;
    const c = document.createElement('canvas');
    c.id = 'c3HzCanvas'; c.className = 'c3-area-canvas'; c.style.display = 'none';
    if (v.canvas && v.canvas.parentNode === cw) cw.insertBefore(c, v.canvas.nextSibling); else cw.appendChild(c);
    HZ.canvas = c;
    c.addEventListener('wheel', onWheel, { passive: false });
    HZ.ed = new (CV().Editor)({
      canvas: c, toScreen, toImage, layers,
      enabled: () => on() && HZ.drawing && !!HZ.curve,
      onChange: (id, sp, final) => { if (id === 'hz' && final) saveLine(); },
    });
    $('c3HzOn').addEventListener('change', (e) => saveSettings({ horizon_match: e.target.checked }));
    $('c3HzFade').addEventListener('input', () => { $('c3HzFadeV').textContent = `${Math.round(+$('c3HzFade').value)} px`; });
    $('c3HzFade').addEventListener('change', () => saveSettings({ horizon_fade_px: +$('c3HzFade').value }));
    $('c3HzDraw').addEventListener('click', () => setDrawing(!HZ.drawing));
    $('c3HzDel').addEventListener('click', deleteLine);
    $('c3HzLines').addEventListener('change', () => { if ($('c3HzLines').checked) load(false); else redraw(); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && HZ.drawing) { const tag = (e.target && e.target.tagName) || ''; if (!['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) setDrawing(false); }
    }, true);
    if (Array.isArray(v.overlayHooks)) v.overlayHooks.push(() => redraw());   // follow the viewer's pan / zoom
    window.addEventListener('md-frame', () => { if (!on()) return; if (HZ.drawing) HZ.curve = null; renderUi(); if (HZ.drawing || $('c3HzLines').checked) load(true); else redraw(); });
    window.addEventListener('md-project', () => { HZ.info = null; HZ.drawn = []; HZ.s = null; setDrawing(false); });
    window.addEventListener('resize', () => redraw());
    c3.onChange(() => { if (!on() && HZ.drawing) setDrawing(false); renderUi(); redraw(); });
    HZ.ready = true;
    renderUi();
    return true;
  }
  function start(tries = 0) { if (!init() && tries < 60) setTimeout(() => start(tries + 1), 300); }
  window.MD_C3HORIZON = {
    get ready() { return HZ.ready; }, get state() { return HZ; },
    load, setDrawing, saveLine, deleteLine, infoText, layers,
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => start()); else start();
})();
