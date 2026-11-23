// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/zones_harness.mjs – headless run of static/zones.js (stage ViTMatte zone panel) and static/brush.js (stage
// CoTracker brush corrections, 2026-10-15) in jsdom with a fake API / viewer.
// Exercises the panel flow that ships, without a browser or server: the automatic whole-outline zone (a 409 waits for the
// end of the job), palette controls -> PATCH, band area per frame, overlays (alpha / comp / trimap / engines), apply 409
// missing_front -> dialog -> "bez holdoutu" -> apply ok, one zone per object, clean plate row, delete.
// NOT covered: the zone drawing hidden since 2026-10-21 (DRAW_TOOLS = false in zones.js, code kept) – its assertions were
// dropped when this harness was revived on 2026-11-03; they are in git history (tests/js/zones_harness.mjs before that).
// Run: cd tests/js && npm ci && node zones_harness.mjs   (tests/test_zones_js_harness.py wraps it; skipped without jsdom)
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.resolve(here, '..', '..', 'static');

const html = `<!doctype html><html><body>
  <div id="stagePanel"><div class="stage-pane"><div class="pane-section" id="st4ZonePanel"><div id="z4Panel"></div>
  <details id="z4Adv"><summary>adv</summary></details></div><div class="stage-status" id="st4Status"></div><div id="st4Runs"></div></div>
  <div class="stage-pane"><div id="b2Panel"></div></div></div>
  <div id="timelineKeys"></div><div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;

// canvas 2D context stub (jsdom has none): enough for tint / trimap / engines / comp code paths
const fakeCtx = () => new Proxy({ canvas: null, globalAlpha: 1 }, {
  get(t, k) {
    if (k in t) return t[k];
    if (k === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h) * 4), width: w, height: h });
    if (k === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h) * 4), width: w, height: h });
    if (k === 'measureText') return () => ({ width: 10 });
    return () => {};
  },
  set(t, k, v) { t[k] = v; return true; },
});
window.HTMLCanvasElement.prototype.getContext = function () { return fakeCtx(); };
window.confirm = () => { throw new Error('native confirm() used - call viewer.uiConfirm'); };
window.requestAnimationFrame = (fn) => setTimeout(fn, 0);

// ---------------------------------------------------------------- fake server (shapes as in app.py)
const calls = [];
const F0 = 101, F1 = 140;
const srv = { zones: {}, next: 1, finalValid: false, activeRun: null, hand: false, holdoutRefusals: 0, cp: { source: 'brak', auto: null } };
const keysOf = (z) => { const o = {}; for (const [k, v] of Object.entries(z.keys)) o[k] = v; o[String(z.frame)] = { polygon: z.polygon, strokes: z.strokes, bg_strokes: z.bg_strokes }; return o; };
const sortedKeys = (z) => Object.keys(keysOf(z)).map(Number).sort((a, b) => a - b);
const statusOf = (z) => ({ state: !z.enabled ? 'wyłączona' : (srv.finalValid && srv.activeRun ? 'policzona' : 'do przeliczenia'), n_keys: sortedKeys(z).length, keys: sortedKeys(z),
  name: (z.prompt || '').trim() ? z.prompt.trim().slice(0, 40) : z.zone_id, run: srv.finalValid ? srv.activeRun : null,
  run_stats: srv.finalValid ? { engine: z.matting, frames: 40, seconds: 48.0, engine_share: { chroma: 0.6, knn: 0.4 }, computed: 40, cached: 0 } : null });
const err = (status, detail) => { const e = new Error(detail && typeof detail === 'object' ? detail.detail : detail); e.status = status; e.detail = detail; return e; };
const preview = (z, f) => {
  if (srv.hand && z.holdout) throw err(409, { detail: 'Obiekty z przodu bez maski: hand', missing_front: ['hand'], queue: [] });
  const isKey = sortedKeys(z).includes(f);
  const eng = z.matting === 'matanyone' ? 'chroma' : z.matting;
  return { added_px: 1234, removed_px: 0, soft_px: 321, stats: { engine: eng, seconds: 0.21, engine_share: eng === 'auto' ? { chroma: 0.7, knn: 0.3 } : {} }, flags: [],
    holdout_objects: [], cleanplate: 'brak', is_key: isKey, frame: f, engine_used: eng, ...(z.matting === 'matanyone' ? { note: 'podgląd = podpowiedź klucza' } : {}) };
};
async function fakeApi(method, url, body) {
  calls.push([method, url, body]);
  let m;
  if (method === 'GET' && url === '/api/object/obj/zones') return { zones: Object.values(srv.zones).map((z) => ({ ...z })), status: Object.fromEntries(Object.values(srv.zones).map((z) => [z.zone_id, statusOf(z)])) };
  if (method === 'GET' && url === '/api/project/cleanplate_auto') return srv.cp;
  if (method === 'POST' && url === '/api/queue/preset/all') return { success: true, added: [1, 2], objects: ['hand', 'obj'], report: {} };
  if (method === 'POST' && url === '/api/queue/run') return { success: true };
  if (url === '/api/object/obj/zones/interpret' && method === 'POST') {   // step 9
    const t = (body.prompt || '').toLowerCase();
    const plan = { text: body.prompt || '', preset: /dym|smoke/.test(t) ? 'smoke' : (/w[lł]os|hair/.test(t) ? 'hair' : null), fg: body.prompt || '', fg_en: /w[lł]os|hair/.test(t) ? 'hair' : (body.prompt || ''),
      distractors: /bez /.test(t) ? [t.split('bez ')[1]] : [], bg_line: /lini|line/.test(t), same_color_bg: false, engine_hints: /pstre/.test(t) ? { chroma: 0.3 } : {}, forced_engine: null, warnings: [], matched: [] };
    return { success: true, plan, bg_strokes: body.frame != null && plan.bg_line ? [{ p: [[100, 50], [500, 52]], radius_px: 3, length_px: 400, angle: 0.3, source: 'hough' }] : [], ...(body.frame != null ? { frame: body.frame } : {}) };
  }
  if (url === '/api/object/obj/zones/propose' && method === 'POST') {
    return { success: true, frame: body.frame, fg: 'hair', preset: 'hair', sam3: true, sam3_error: null, plan: { text: body.prompt || '' },
      proposals: [{ id: 0, polygon: [[300, 100], [420, 100], [420, 240], [300, 240]], area_px: 9000, bbox: [300, 100, 420, 240], side: 'prawo', source: 'sam3' },
                  { id: 1, polygon: [[100, 100], [160, 100], [160, 240], [100, 240]], area_px: 4000, bbox: [100, 100, 160, 240], side: 'lewo', source: 'soft' }] };
  }
  if (method === 'GET' && url === '/api/zones/preset_defaults/hair') return { params: { temporal: 'none', outline_in_px: 2 } };
  if (/^\/api\/object\/obj\/zones\/new/.test(url) && method === 'POST' && srv.refuseNew && srv.refuseNew.length) throw srv.refuseNew.shift();
  if (url === '/api/object/obj/zones/new' && method === 'POST') {   // no preview: the automatic outline zone, stage-2 brush correction
    const outline = body.criterion === 'matte' && body.params && body.params.region === 'outline';
    if (!outline && (body.criterion !== 'all' || !body.params || body.params.source !== 'brush')) throw err(400, 'harness: only outline / brush zones here');
    const z = { zone_id: `z_${String(srv.next++).padStart(2, '0')}`, schema: 'matte_desk/zone@1.1', keys: {}, prompt_history: [], engine_hints: {}, enabled: true,
      bg_static: true, holdout: true, preset: 'hair', matting: 'matanyone', prompt: '', ...body };
    srv.zones[z.zone_id] = z;
    return { success: true, zone: { ...z } };
  }
  if (url === '/api/object/obj/core_fix' && method === 'POST') {
    srv.coreFix = (srv.coreFix || 0) + 1;
    return { success: true, frames: [F0, F1], changed_px: 777, zones: Object.values(srv.zones).filter((z) => z.params && z.params.source === 'brush').map((z) => z.zone_id), sig: 'x' };
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/new\?preview_frame=(\d+)$/)) && method === 'POST') {
    if (body.criterion !== 'matte' || body.mode !== 'add') throw err(400, 'tryb');
    if (body.polygon.length < 3 && !body.strokes.length && !(body.params && body.params.region === 'outline')) throw err(400, 'Strefa „matte”: narysuj obszar (wielokąt albo pędzel +)');
    const z = { zone_id: `z_${String(srv.next++).padStart(2, '0')}`, schema: 'matte_desk/zone@1.1', keys: {}, prompt_history: [], params: {}, engine_hints: {}, enabled: true,
      static: false, bg_static: true, holdout: true, preset: 'hair', matting: 'matanyone', prompt: '', ...body };
    const pv = preview(z, +m[1]);
    srv.zones[z.zone_id] = z; srv.finalValid = false;
    return { success: true, zone: { ...z }, preview: pv };
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/(z_\d+)$/))) {
    const z = srv.zones[m[1]]; if (!z) throw err(404, 'Brak strefy');
    if (method === 'DELETE') { delete srv.zones[m[1]]; srv.finalValid = false; return { success: true }; }
    if (method === 'PATCH') {
      const ch = Object.fromEntries(Object.entries(body).filter(([, v]) => v != null));
      if (!Object.keys(ch).length) throw err(400, 'Nic do zmiany');
      if (ch.prompt != null && z.prompt && ch.prompt !== z.prompt) z.prompt_history.push(z.prompt);
      Object.assign(z, ch); srv.finalValid = false;
      const plan = ch.prompt != null ? (await fakeApi('POST', '/api/object/obj/zones/interpret', { prompt: ch.prompt })).plan : null;
      if (plan && plan.preset && ch.preset == null) z.preset = plan.preset;
      return { success: true, zone: { ...z }, changed: Object.keys(ch).sort(), status: statusOf(z), ...(plan ? { plan } : {}) };
    }
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/(z_\d+)\/preview\/(\d+)$/)) && method === 'POST') {
    const z = srv.zones[m[1]]; if (!z) throw err(404, 'Brak strefy');
    return { success: true, zone_id: z.zone_id, preview: preview(z, +m[2]) };
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/(z_\d+)\/keys\/(\d+)$/))) {
    const z = srv.zones[m[1]]; if (!z) throw err(404, 'Brak strefy'); const f = +m[2];
    if (method === 'PUT') {
      if (body.polygon.length < 3 && !body.strokes.length && !(z.params && (z.params.region === 'outline' || z.params.source === 'brush'))) throw err(400, 'Klucz strefy: narysuj obszar');
      const prev = z.keys[String(f)] || {};
      const note = body.prompt == null ? (prev.prompt || '') : body.prompt.trim();   // step 7: iteration note (None keeps, "" clears)
      const entry = { polygon: body.polygon, strokes: body.strokes || [], bg_strokes: body.bg_strokes || [] };
      if (note) { entry.prompt = note; if (note !== (prev.prompt || '') && !z.prompt_history.includes(note)) z.prompt_history.push(note); }
      if (f === z.frame) Object.assign(z, { polygon: entry.polygon, strokes: entry.strokes, bg_strokes: entry.bg_strokes });
      z.keys = { ...Object.fromEntries(Object.entries(z.keys).filter(([k]) => +k !== f)), [String(f)]: entry };
      srv.finalValid = false;
      return { success: true, zone: { ...z }, keys: sortedKeys(z), key: { ...entry, prompt: note } };
    }
    if (method === 'DELETE') {
      if (sortedKeys(z).length <= 1) throw err(409, 'Ostatni klucz strefy – usuń strefę');
      delete z.keys[String(f)];
      if (f === z.frame) { const nf = sortedKeys(z)[0]; Object.assign(z, { frame: nf, ...z.keys[String(nf)] }); }
      srv.finalValid = false;
      return { success: true, zone: { ...z }, keys: sortedKeys(z) };
    }
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/(z_\d+)\/region\/(\d+)$/)) && method === 'GET') {
    const z = srv.zones[m[1]]; const f = +m[2];
    return { zone_id: z.zone_id, frame: f, keys: sortedKeys(z), is_key: sortedKeys(z).includes(f), contours: [[[300, 100], [420, 100], [420, 240], [300, 240]], [[10, 10], [20, 10], [20, 20]]],
      bbox: [300, 100, 421, 241], area_px: 16800, bg_px: 0,
      ...(z.params && z.params.region === 'outline' ? { region_mode: 'outline', exclude_contours: [[[50, 60], [90, 60], [90, 120], [50, 120]]] } : {}) };
  }
  // whole outline with nothing painted: the band lines on the stages' path (2026-11-20), no exclusion to show
  if ((m = url.match(/^\/api\/object\/obj\/stage_band\/zone:(z_\d+)\/(\d+)\/lines$/)) && method === 'GET') {
    const z = srv.zones[m[1]]; const f = +m[2];
    return { zone_id: z.zone_id, frame: f, keys: sortedKeys(z), is_key: sortedKeys(z).includes(f), contours: [[[300, 100], [420, 100], [420, 240], [300, 240]]],
      bbox: [300, 100, 421, 241], area_px: 16800, bg_px: 0, region_mode: 'outline', exclude_contours: [],
      band_contours: [[[290, 90], [430, 90], [430, 250], [290, 250]]], inset_feather_contours: [], feather_contours: [] };
  }
  if ((m = url.match(/^\/api\/object\/obj\/zones\/(z_\d+)\/apply\?stage=vitmatte$/)) && method === 'POST') {
    const z = srv.zones[m[1]];
    if (srv.hand && z.holdout) { srv.holdoutRefusals++; throw err(409, { detail: 'Obiekty z przodu bez maski: hand', missing_front: ['hand'], queue: [] }); }
    srv.activeRun = srv.activeRun ? `v${String(+srv.activeRun.slice(1) + 1).padStart(3, '0')}` : 'v002'; srv.finalValid = true;
    return { success: true, run: srv.activeRun, computed: 12, cached: 28, dependents_to_recompute: [] };
  }
  throw err(404, `fake api: ${method} ${url}`);
}

// ---------------------------------------------------------------- fake viewer + hooks (objects.js contract)
const canvas = document.getElementById('cv');
const viewer = {
  currentFrame: 115, firstFrame: F0, lastFrame: F1, canvas, panX: 0, panY: 0, zoom: 1, projectData: { work: 'W', plate_w: 1280, plate_h: 720 },
  renders: 0, render() { this.renders++; }, showToast(t) { toasts.push(t); }, openPlatesModal() { opened.push(1); },
  goToFrame(f) { this.currentFrame = f; window.dispatchEvent(new window.CustomEvent('md-frame', { detail: { frame: f } })); },
  getCurrentImage() { return fakeImg; },
  uiConfirm(msg) { confirms.push(msg); return Promise.resolve(true); },
};
const confirms = [];
const toasts = [], opened = [], statuses = [], invalidated = []; let cleared = 0;
const fakeImg = { width: 640, height: 360, naturalWidth: 640, naturalHeight: 360, complete: true };
window.viewer = viewer;
const S = { obj: 'obj', auxMode: false, stage: 4, strokes: {}, polys: {}, keys: [], compCache: {}, overlay: {}, bg: 0, pickColor: '#000' };
const views = [];
const ST = { PLATES: 0, SAM3: 1, BRUSH: 2, MATANYONE: 3, VITMATTE: 4, EXPERIMENTS: 5, EXPORT: 6 };   // objects.js STAGES order
const hooks = {
  S, ST, api: fakeApi, stageState: (id) => ({ state: id === 'cotracker' ? 'todo' : 'done', why: '' }),
  setTool: (t) => { S.tool = S.tool === t ? 'none' : t; },
  el: (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstChild; },
  esc: (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])),
  toast: (t) => toasts.push(t), status: (t, k) => statuses.push([t, k || '']),
  run: async (btn, fn) => { try { await fn(); } catch (e) { statuses.push([`Błąd: ${e.message}`, 'err']); } },
  invalidateFrame: (f) => invalidated.push(f), clearOverlays: () => { cleared++; },
  overlayFor: (kind, f) => (kind.startsWith('zone_') || kind === 'core' ? fakeImg : 'missing'),
  tintCanvas: () => { const c = document.createElement('canvas'); c.width = 640; c.height = 360; return c; },
  compCanvas: () => { const c = document.createElement('canvas'); c.width = 640; c.height = 360; return c; },
  drawVarStroke: () => {}, plateSize: () => [1280, 720],
  loadStages: async () => {}, loadQA: () => {}, setView: (v) => views.push(v), ensurePointsSaved: async () => {}, restoreUserView: () => views.push('restored'),
  putComp: (k, c) => { S.compCache[k] = c; },
  tileOverlays: (kind) => { tileCalls.push(kind); return { reg: { kp: 2, chunks: [{ x: 0, y: 0, w: 640, h: 360 }] }, tiles: [{ c: { x: 0, y: 0, w: 640, h: 360 }, img: fakeImg }] }; },
};
const tileCalls = [];

// ---------------------------------------------------------------- load zones.js into the window
window.eval(readFileSync(path.join(STATIC, 'zones.js'), 'utf-8'));
const ZN = window.MD_ZONES;
const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);
const lastStatus = () => statuses[statuses.length - 1] || ['', ''];

ZN.init(hooks);
assert(ZN.ready, 'init');
for (const id of ['st4Zones', 'z4New', 'z4Presets', 'z4Prompt', 'z4Engine', 'z4Holdout', 'z4CleanRow', 'z4Ov', 'z4Refresh', 'z4Loupe', 'z4ApplyShot', 'z4KeyDel']) assert($(id), `id ${id}`);
assert($('z4Engine').value === 'matanyone', 'default engine matanyone');
srv.refuseNew = [err(409, 'Obiekt „obj” jest teraz liczony')];   // a job on the object while stage ViTMatte opens (block 1)
await ZN.load();
assert($('st4Zones').textContent.includes('brak stref'), 'empty list text');
ZN.onShow(); await wait(5);
assert($('z4CleanTxt').textContent.includes('brak'), 'clean plate row: brak');
assert($('z4CleanRow').classList.contains('warn'), 'clean plate row warns');

// 1. stage ViTMatte opens on the whole-outline zone, created by itself (user decision 2026-10-21): no preview, no key on
// the lane. A job on the object refuses it with a 409: it waits and tries again when the job ends (audit 2026-11-02 #17)
assert(!ZN.drawTools && !ZN.wantsDraw(), 'zone drawing hidden (DRAW_TOOLS = false)');
await wait(30);
assert(ZN.state.autoWait && ZN.state.autoWait.obj && !Object.keys(srv.zones).length, 'a passing 409 waits, no zone yet');
assert(lastStatus()[1] === 'warn' && lastStatus()[0].includes('po końcu zadania'), `status says it retries: ${lastStatus()[0]}`);
assert($('zpNone').style.display === '' && $('zpNewOutline').style.display === '', 'palette offers the zone meanwhile');
calls.length = 0;
window.dispatchEvent(new window.CustomEvent('matte-job-finished')); await wait(40);
const post = calls.find((c) => c[0] === 'POST' && c[1] === '/api/object/obj/zones/new');
assert(post && post[2].criterion === 'matte' && post[2].matting === 'vitmatte' && post[2].params.region === 'outline' && post[2].polygon.length === 0 && post[2].frame === 115,
  `automatic outline zone body: ${JSON.stringify(post && post[2])}`);
assert(calls.some((c) => c[1] === '/api/zones/preset_defaults/hair'), 'the preset defaults asked for');
const st = ZN.state;
assert(st.active === 'z_01' && !st.autoWait.obj, `active after the retry: ${st.active}`);
assert(document.querySelectorAll('.zone-card').length === 1 && document.querySelector('.zone-card').classList.contains('cur'), 'one active card');
assert($('timelineZones').querySelectorAll('.zone-key').length === 0, 'an outline zone without a drawing has no key on the lane');

// 2. the floating palette „Strefa”: every control of the zone; PATCH merges into the zone's params
const z1 = srv.zones.z_01;
assert($('zonePalette') && $('zonePalette').contains($('z4Inset')) && !$('z4Panel').contains($('z4Inset')), 'sliders on the palette, not in the card');
assert($('zonePalette').style.display === '' && $('zpNone').style.display === 'none' && $('z4Tune').style.display === '', 'palette shown with an active zone');
assert($('zpTitle').textContent.includes('cały obrys'), `title says outline: ${$('zpTitle').textContent}`);
assert($('z4OutRow').style.display === '' && $('z4ExclRow').style.display === '' && $('z4BandRow').style.display === '', 'outline controls shown');
for (const id of ['z4Region', 'z4Brush', 'z4TrackRow', 'zpNewPaint', 'z4Propose']) assert($(id).style.display === 'none', `hidden with the drawing: ${id}`);
assert($('z4BlackAfterRow').style.display === 'none', 'finish sliders live in Final Touch');
z1.params.fg_erode_px = 4; st.list.find((z) => z.zone_id === 'z_01').params.fg_erode_px = 4;   // a param set elsewhere (CLI)
calls.length = 0;
$('z4Temporal').value = 'flow_mean'; $('z4Temporal').dispatchEvent(new window.Event('change')); await wait(30);
$('z4TRadius').value = '5'; $('z4TRadius').dispatchEvent(new window.Event('change')); await wait(30);
$('z4TScale').checked = true; $('z4TScale').dispatchEvent(new window.Event('change')); await wait(30);
assert(z1.params.temporal === 'flow_mean' && z1.params.temporal_radius === 5 && z1.params.temporal_scale === 2 && z1.params.fg_erode_px === 4 && z1.params.region === 'outline',
  `temporal params merged: ${JSON.stringify(z1.params)}`);
assert(calls.filter((c) => c[0] === 'PATCH').length === 3, 'one PATCH per control');
// live band while dragging, release -> saved, the live band stays with the saved values
$('z4Inset').value = '8'; $('z4Inset').dispatchEvent(new window.Event('input')); await wait(10);
assert(st.inset && st.inset.n === 8 && $('z4InsetV').textContent === '8 px', 'live inset state');
ZN.draw(fakeCtx(), fakeImg, 1, viewer.currentFrame);                    // the preview draws without throwing
$('z4Inset').dispatchEvent(new window.Event('change')); await wait(30);
assert(z1.params.fg_erode_px === 8 && z1.params.force_core_erode_px === 8 && st.inset && st.inset.live === false, `inset saved: ${JSON.stringify(z1.params)}`);
$('z4Out').value = '60'; $('z4Out').dispatchEvent(new window.Event('input')); await wait(5);
assert(st.inset && st.inset.out === 60 && $('z4OutV').textContent === '60 px', 'live band state');
ZN.draw(fakeCtx(), fakeImg, 1, viewer.currentFrame);
$('z4Out').dispatchEvent(new window.Event('change')); await wait(40);
assert(z1.params.outline_out_px === 60 && z1.params.fg_erode_px === 8, `band saved, params merged: ${JSON.stringify(z1.params)}`);
// other objects can be cut out of the band; one without a mask cannot
S.objects = [{ name: 'obj', has: { core: true } }, { name: 'hand', has: { core: true, final: false } }, { name: 'face', has: { core: false, final: false } }];
await ZN.load();
const exBoxes = $('z4Excl').querySelectorAll('input[data-ex]');
assert(exBoxes.length === 2 && !exBoxes[0].disabled && exBoxes[1].disabled, 'other objects listed (not this one), one without a mask disabled');
exBoxes[0].checked = true; exBoxes[0].dispatchEvent(new window.Event('change', { bubbles: true })); await wait(40);
assert(JSON.stringify(z1.params.exclude_objects) === '["hand"]', `hand excluded: ${JSON.stringify(z1.params)}`);
// the engine of the palette; no one-frame model preview on a change (FRAME_PV off, its button hidden)
$('zpEngine').value = 'auto'; $('zpEngine').dispatchEvent(new window.Event('change')); await wait(30);
assert(z1.matting === 'auto', 'engine auto');
$('zpEngine').value = 'vitmatte'; $('zpEngine').dispatchEvent(new window.Event('change')); await wait(30);
assert(z1.matting === 'vitmatte', 'engine back on ViTMatte');
assert($('z4PvFrame').style.display === 'none' && !calls.some((c) => /\/preview\/\d+$/.test(c[1])), 'no one-frame preview');
// a frame change fetches the band area of that frame
calls.length = 0;
viewer.goToFrame(120); await wait(450);
// (a whole outline with nothing painted: the stages' lines path, not the job-locked region endpoint – 2026-11-20)
assert(calls.some((c) => c[0] === 'GET' && c[1] === '/api/object/obj/stage_band/zone:z_01/120/lines'), 'band lines fetched on frame change (stage path)');
assert(!calls.some((c) => c[1].endsWith('/region/120')), 'no job-locked region request for a zone without drawing');
assert(st.regionCache.z_01[120] && st.regionCache.z_01[120].band_contours.length === 1, 'outline band lines cached');
viewer.goToFrame(115); await wait(30);

// 3. overlays draw without throwing for every view; the loupe
const ctx = fakeCtx();
for (const ov of ['alpha', 'comp', 'trimap', 'engines', 'none']) { $('z4Ov').querySelector(`[data-ov="${ov}"]`).click(); ZN.draw(ctx, fakeImg, 0.5, 115); }
$('z4Ov').querySelector('[data-ov="alpha"]').click();
$('z4Loupe').click(); assert(st.loupe, 'loupe on');
const Ev = window.PointerEvent || window.MouseEvent; canvas.dispatchEvent(new Ev('pointermove', { clientX: 100, clientY: 80 }));
st.mouse = [100, 80]; canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 360 });
ZN.drawLoupe(ctx, fakeImg, 0.5); $('z4Loupe').click(); assert(!st.loupe, 'loupe off');

// 4. "Zastosuj": 409 missing_front -> dialog -> "bez holdoutu" -> PATCH + apply ok
srv.hand = true; calls.length = 0; statuses.length = 0;
$('z4ApplyShot').click(); await wait(60);
const modal = $('zoneFrontModal');
assert(modal && modal.style.display === 'flex' && modal.textContent.includes('hand'), 'objects-in-front dialog shown');
modal.querySelector('[data-act="noholdout"]').click(); await wait(120);
assert(z1.holdout === false, 'holdout off via PATCH');
assert(srv.activeRun === 'v002' && srv.holdoutRefusals === 1, 'apply retried and succeeded');
assert(lastStatus()[0].includes('Wersja v002') && lastStatus()[0].includes('przeliczone 12') && lastStatus()[1] === 'ok', `status after apply: ${lastStatus()[0]}`);
assert(cleared === 1, 'overlays cleared once after the new version');
assert(document.querySelector('.zone-card .zs').textContent.includes('policzona'), 'card state policzona');
assert(document.querySelector('.zone-card .zone-meta').textContent.includes('s/kl.'), 'card shows s/frame');
// the queue button of the dialog
z1.holdout = true; calls.length = 0;
$('zpApply').click(); await wait(60);
$('zoneFrontModal').querySelector('[data-act="queue"]').click(); await wait(60);
assert(calls.some((c) => c[1] === '/api/queue/preset/all') && calls.some((c) => c[1] === '/api/queue/run'), 'queue-all + run');
assert(lastStatus()[0].includes('Policz wszystko'), 'queue status');
srv.hand = false;

// 5. enable / disable from the card; version details html; one outline zone per object; clean plate row; delete
document.querySelector('.zone-card input[data-en]').click(); await wait(30);
assert(z1.enabled === false && document.querySelector('.zone-card').classList.contains('off'), 'disabled');
document.querySelector('.zone-card input[data-en]').click(); await wait(30);
assert(z1.enabled === true, 'enabled again');
const det = ZN.runDetailsHtml({ run: 'v002', applied_from: 'v001', zones: { z_01: { engine: 'auto', frames: 40, seconds: 48, s_per_frame: 1.2, computed: 12, cached: 28, engine_share: { chroma: 0.6, knn: 0.4 } } } });
assert(det.includes('1.2 s/kl.') && det.includes('chroma 60 %') && det.includes('v001'), 'run details html');
st.active = null; ZN.onStage();
assert($('zpNone').style.display === '' && $('zpNewOutline').style.display === 'none' && $('zpPickZone').textContent === 'Wybierz z_01', 'an object with a zone offers to pick it, not another one');
$('zpPickZone').click(); await wait(20);
assert(st.active === 'z_01', 'zone picked');
$('z4CpNone').click(); await wait(20);
assert($('z4CleanTxt').textContent.includes('pominięty'), 'clean plate skipped text');
$('z4CpFile').click(); assert(opened.length === 1, 'Mam plik -> plates modal');
// the automatic zone refused for good (not a passing refusal): deleting the zone leaves the button, which creates it with a preview
srv.refuseNew = [err(500, 'zepsute')];
calls.length = 0; statuses.length = 0;
document.querySelector('.zone-card button[data-del]').click(); await wait(60);
assert(confirms.some((m) => /^Usunąć strefę z_01/.test(m)), 'zone delete asks through viewer.uiConfirm');
assert(!srv.zones.z_01 && st.autoFailed && st.autoFailed.obj && $('zpNone').style.display === '' && $('zpNewOutline').style.display === '', 'palette offers a new outline zone');
assert($('st4Zones').textContent.includes('brak stref'), 'empty list again');
calls.length = 0;
$('zpNewOutline').click(); await wait(80);
const postO = calls.find((c) => c[0] === 'POST' && /zones\/new\?preview_frame=115$/.test(c[1]));
assert(postO && postO[2].polygon.length === 0 && postO[2].matting === 'vitmatte' && postO[2].params.region === 'outline', `outline zone body: ${JSON.stringify(postO && postO[2])}`);
assert(st.active && srv.zones[st.active].params.region === 'outline' && $('z4Tune').style.display === '', 'outline zone active, sliders shown');
assert(toasts.some((t) => /Nowa strefa z_02/.test(t)), 'toast new zone');
assert(st.preview && st.preview.f === 115 && st.preview.zid === 'z_02', 'the button shows the preview computed with the zone');
assert($('z4PvInfo').textContent.includes('F115'), `preview info: ${$('z4PvInfo').textContent}`);
for (const ov of ['alpha', 'comp', 'trimap', 'engines']) { $('z4Ov').querySelector(`[data-ov="${ov}"]`).click(); ZN.draw(ctx, fakeImg, 0.5, 115); }   // with a preview to show
// another stage -> no palette
S.stage = ST.EXPERIMENTS; ZN.onStage();
assert($('zonePalette').style.display === 'none' && !ZN.wantsDraw(), 'stage Eksperymenty = no zone palette');

// ---------------------------------------------------------------- stage CoTracker: brush corrections (static/brush.js)
window.eval(readFileSync(path.join(STATIC, 'brushcore.js'), 'utf-8'));   // stroke geometry shared with the Cleanup 3D area editor
window.eval(readFileSync(path.join(STATIC, 'brush.js'), 'utf-8'));
const BR = window.MD_BRUSH;
BR.init(hooks); assert(BR.ready, 'brush init');
for (const id of ['b2List', 'b2Track', 'b2EmptyKey', 'b2DelKey', 'b2Apply', 'b2Info']) assert($(id), `brush id ${id}`);
S.stage = ST.BRUSH; S.tool = 'none';
for (const z of Object.values(srv.zones)) delete srv.zones[z.zone_id];
await BR.load(); await ZN.load();
assert($('b2List').textContent.includes('brak poprawek'), 'brush: empty list text');
// a. first "+" stroke at F110 -> new brush zone (criterion all, mode add, source brush, local tracking)
calls.length = 0;
viewer.goToFrame(110);
S.strokes[110] = [{ mode: 'add', radius_px: 10, p: [[100, 100], [120, 120]] }];
BR.onDraw(110); await wait(500);
const pAdd = calls.find((c) => c[0] === 'POST' && c[1] === '/api/object/obj/zones/new');
assert(pAdd && pAdd[2].criterion === 'all' && pAdd[2].mode === 'add' && pAdd[2].params.source === 'brush' && pAdd[2].params.track === 'local' && pAdd[2].strokes.length === 1,
  `brush add body: ${JSON.stringify(pAdd && pAdd[2])}`);
// b. "-" stroke in the same frame -> second zone (remove)
calls.length = 0;
S.strokes[110] = [{ mode: 'remove', radius_px: 6, p: [[200, 200], [210, 220]] }];
BR.onDraw(110); await wait(500);
const pRem = calls.find((c) => c[0] === 'POST' && c[1] === '/api/object/obj/zones/new');
assert(pRem && pRem[2].mode === 'remove' && BR.state.list.length === 2, 'brush remove zone created');
// c. "+" in a frame without a key -> region of the propagation (largest contour) + the stroke as a new key
calls.length = 0;
viewer.goToFrame(120);
S.strokes[120] = [{ mode: 'add', radius_px: 10, p: [[310, 110], [330, 130]] }];
BR.onDraw(120); await wait(500);
const zAdd = BR.state.list.find((z) => z.mode === 'add');
assert(calls.some((c) => c[0] === 'GET' && c[1] === `/api/object/obj/zones/${zAdd.zone_id}/region/120`), 'region fetched for the new key');
const put120 = calls.find((c) => c[0] === 'PUT' && c[1] === `/api/object/obj/zones/${zAdd.zone_id}/keys/120`);
assert(put120 && put120[2].polygon.length === 4 && put120[2].strokes.length === 1, `key 120 from the region: ${JSON.stringify(put120 && put120[2])}`);
assert(toasts.some((t) => t.includes('2 części')), 'toast: region of several parts');
// d. "Zmaż" in F120 -> erase strokes only into the key of the active layer (fix layers, user decision 2026-10-30)
calls.length = 0;
S.strokes[120] = [{ mode: 'erase', radius_px: 8, p: [[315, 115]] }];
BR.onDraw(120); await wait(500);
const puts = calls.filter((c) => c[0] === 'PUT' && /\/keys\/120$/.test(c[1]));
assert(puts.length === 1 && puts[0][1].includes(zAdd.zone_id) && puts[0][2].strokes.some((s) => s.erase), `erase in the active layer: ${puts.length}`);
// e. Ctrl+Z restores the key before the erase
calls.length = 0;
assert(await BR.undo(120), 'brush undo handled');
assert(calls.some((c) => (c[0] === 'PUT' || c[0] === 'DELETE') && /\/keys\/120$/.test(c[1])), 'undo touched the key 120');
// f. empty key = end of the corrections in F130 (the active layer)
calls.length = 0;
viewer.goToFrame(130);
$('b2EmptyKey').click(); await wait(50);
const empt = calls.filter((c) => c[0] === 'PUT' && /\/keys\/130$/.test(c[1]));
assert(empt.length === 1 && empt[0][1].includes(zAdd.zone_id) && !empt[0][2].polygon.length && !empt[0][2].strokes.length, `empty key in F130: ${empt.length}`);
// g. drawing in stage 2 (keys + propagated regions) does not throw; the lane shows the keys of the active layer (F110, F120, F130)
BR.draw(window.HTMLCanvasElement.prototype.getContext(), fakeImg, 0.5, 120);
BR.draw(window.HTMLCanvasElement.prototype.getContext(), fakeImg, 0.5, 125);
assert($('timelineBrush') && $('timelineBrush').querySelectorAll('.zone-key').length === 3, `brush lane diamonds: ${$('timelineBrush') && $('timelineBrush').querySelectorAll('.zone-key').length}`);
// h. „Zastosuj poprawki” -> POST core_fix, the user's view back
calls.length = 0; views.length = 0;
$('b2Apply').click(); await wait(50);
assert(srv.coreFix === 1 && views.includes('restored') && lastStatus()[0].includes('Maska 1+2'), `apply: ${lastStatus()[0]}`);
// i. Delete in F130 -> DELETE keys/130 of the active layer (with a confirmation)
calls.length = 0; confirms.length = 0;
await BR.deleteKeysAt(130);
assert(confirms.length === 1 && calls.filter((c) => c[0] === 'DELETE' && /\/keys\/130$/.test(c[1])).length === 1, 'delete key in F130');
// j. brush zones never appear in the ViTMatte list
assert(!ZN.state.list.some((z) => z.params && z.params.source === 'brush'), 'brush zones hidden from the ViTMatte panel');
// k. „＋ warstwa”: the next stroke of that mode starts a new layer
const before = BR.state.list.length; calls.length = 0;
$('b2NewAdd').click();
S.strokes[120] = [{ mode: 'add', radius_px: 10, p: [[500, 300], [520, 310]] }];
BR.onDraw(120); await wait(500);
assert(BR.state.list.length === before + 1 && calls.some((c) => c[0] === 'POST' && c[1] === '/api/object/obj/zones/new'), `new fix layer: ${BR.state.list.length}`);

console.log(`OK zones harness: ${calls.length + 1} api calls in the last block, ${toasts.length} toasts, ${statuses.length} statuses, renders ${viewer.renders}`);
