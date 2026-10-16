// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/zones_harness.mjs – headless run of static/zones.js (stage ViTMatte zone panel) and static/brush.js (stage
// CoTracker brush corrections, 2026-10-15) in jsdom with a fake API / viewer.
// Exercises the whole panel flow without a browser or server: first stroke -> new zone + preview, preset PATCH,
// region in a frame without a key, "-" stroke there -> key from the propagated region, Ctrl+Z, apply 409 missing_front
// -> dialog -> "bez holdoutu" -> apply ok, overlays (alpha / comp / trimap / engines), clean plate row, delete.
// Run: NODE_PATH=<dir with jsdom> node tests/js/zones_harness.mjs   (tests/test_zones_js_harness.py wraps it; skipped without jsdom)
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
  <div id="timelineKeys"></div><canvas id="cv" width="640" height="360"></canvas></body></html>`;
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
  if (url === '/api/object/obj/zones/new' && method === 'POST') {   // stage-2 brush correction (no preview)
    if (body.criterion !== 'all' || !body.params || body.params.source !== 'brush') throw err(400, 'harness: only brush zones here');
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
  loadStages: async () => {}, loadQA: () => {}, setView: (v) => views.push(v), ensurePointsSaved: async () => {},
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
await ZN.load();
assert($('st4Zones').textContent.includes('brak stref'), 'empty list text');
ZN.onShow(); await wait(5);
assert($('z4CleanTxt').textContent.includes('brak'), 'clean plate row: brak');
assert($('z4CleanRow').classList.contains('warn'), 'clean plate row warns');

// 1. first stroke = new zone (preset fibres chosen first), preview at once, card + lane
$('z4Presets').querySelector('[data-preset="fibres"]').click();
$('z4Prompt').value = 'włosy bez linii'; $('z4Prompt').dispatchEvent(new window.Event('change'));
S.strokes[115] = [{ mode: 'add', radius_px: 12, p: [[500, 200], [520, 230]] }, { mode: 'remove', radius_px: 4, p: [[600, 300], [610, 300]] }];
assert(ZN.wantsDraw(), 'zone flow on (advanced closed)');
ZN.onDraw(115);
await wait(750);
const st = ZN.state;
assert(st.active === 'z_01', `active after first stroke: ${st.active}`);
const post = calls.find((c) => c[0] === 'POST' && c[1].startsWith('/api/object/obj/zones/new'));
assert(post && post[2].preset === 'fibres' && post[2].prompt === 'włosy bez linii' && post[2].matting === 'matanyone' && post[2].strokes.length === 1 && post[2].bg_strokes.length === 1, 'POST new body');
assert(st.preview && st.preview.f === 115 && st.preview.zid === 'z_01', 'preview set');
assert(invalidated.includes(115) && cleared === 0, 'only frame 115 invalidated, no clearOverlays');
assert(document.querySelectorAll('.zone-card').length === 1 && document.querySelector('.zone-card').classList.contains('cur'), 'one active card');
assert(document.querySelector('.zone-card .zone-name b').textContent === 'włosy bez linii', 'card name from prompt');
assert($('z4PvInfo').textContent.includes('F115') && $('z4PvInfo').textContent.includes('podpowiedź klucza'), 'preview info + matanyone note');
assert($('timelineZones').querySelectorAll('.zone-key').length === 1 && $('timelineZones').querySelector('.zone-range'), 'lane: 1 diamond + range');
assert(toasts.some((t) => /Nowa strefa z_01/.test(t)), 'toast new zone');
assert(S.strokes[115].length === 0, 'local strokes consumed');

// 2. preset / engine change on the active zone -> PATCH + preview
calls.length = 0;
$('z4Presets').querySelector('[data-preset="hair"]').click(); await wait(30);
assert(calls.some((c) => c[0] === 'PATCH' && c[2].preset === 'hair'), 'PATCH preset');
assert(srv.zones.z_01.preset === 'hair', 'server preset');
$('z4Engine').value = 'auto'; $('z4Engine').dispatchEvent(new window.Event('change')); await wait(30);
assert(srv.zones.z_01.matting === 'auto', 'engine auto');
assert(calls.some((c) => c[0] === 'POST' && c[1].endsWith('/preview/115')), 'preview after PATCH');
assert($('z4PvInfo').textContent.includes('chroma 70 %'), `engine share shown: ${$('z4PvInfo').textContent}`);
// 2b. card "Wygładzanie w czasie" (ViTMatte only) + "Wykończenie": PATCH merges into the zone's params
assert($('z4TempSec').style.display === 'none', 'temporal section hidden for a non-ViTMatte engine');
assert(!$('z4Black').disabled && $('z4BlackV').textContent === '0.00', 'finish: default black shown');
srv.zones.z_01.params = { fg_erode_px: 4 }; ZN.state.list.find((z) => z.zone_id === 'z_01').params = { fg_erode_px: 4 };   // a param set elsewhere (CLI)
$('z4Black').value = '0.12'; $('z4Black').dispatchEvent(new window.Event('input'));
assert($('z4BlackV').textContent === '0.12', 'live number while dragging');
$('z4Black').dispatchEvent(new window.Event('change')); await wait(30);
assert(srv.zones.z_01.params.alpha_black === 0.12 && srv.zones.z_01.params.fg_erode_px === 4, `black merged into params: ${JSON.stringify(srv.zones.z_01.params)}`);
$('z4Engine').value = 'vitmatte'; $('z4Engine').dispatchEvent(new window.Event('change')); await wait(30);
assert($('z4TempSec').style.display === '', 'temporal section shown for ViTMatte');
$('z4Temporal').value = 'flow_mean'; $('z4Temporal').dispatchEvent(new window.Event('change')); await wait(30);
$('z4TRadius').value = '5'; $('z4TRadius').dispatchEvent(new window.Event('change')); await wait(30);
$('z4TScale').checked = true; $('z4TScale').dispatchEvent(new window.Event('change')); await wait(30);
const zp = srv.zones.z_01.params;
assert(zp.temporal === 'flow_mean' && zp.temporal_radius === 5 && zp.temporal_scale === 2 && zp.alpha_black === 0.12 && zp.fg_erode_px === 4,
  `temporal params: ${JSON.stringify(zp)}`);
// 2c. "Wcięcie w rdzeń": live preview on the mask while dragging, release -> fg_erode_px + force_core_erode_px, trimap overlay
assert($('z4AreaSec').style.display === '' && $('z4InsetV').textContent === '4 px', `inset shown for ViTMatte: ${$('z4InsetV').textContent}`);
$('z4Inset').value = '8'; $('z4Inset').dispatchEvent(new window.Event('input')); await wait(10);
assert(ZN.state.inset && ZN.state.inset.n === 8 && $('z4InsetV').textContent === '8 px', 'live inset preview state');
ZN.draw(fakeCtx(), fakeImg, 1, viewer.currentFrame);                    // the preview draws without throwing
assert(ZN.state.insetDist && ZN.state.insetDist.w === 640, 'core distance map computed once');
$('z4Inset').dispatchEvent(new window.Event('change')); await wait(30);
assert(srv.zones.z_01.params.fg_erode_px === 8 && srv.zones.z_01.params.force_core_erode_px === 8 && ZN.state.inset === null && ZN.state.ov === 'trimap',
  `inset saved: ${JSON.stringify(srv.zones.z_01.params)} ov ${ZN.state.ov}`);
// 2d. floating palette „Strefa” (user decision 2026-10-15): every slider there; area "outline" = band around the whole core,
// the drawing and the chosen objects are exclusions
assert($('zonePalette') && $('zonePalette').contains($('z4Black')) && !$('z4Panel').contains($('z4Black')), 'sliders on the palette, not in the card');
assert($('zonePalette').style.display === '' && $('zpNone').style.display === 'none', 'palette shown with an active zone');
assert($('z4AreaSec').style.display === '' && $('z4OutRow').style.display === 'none', 'area section for ViTMatte; band slider only in outline mode');
S.objects = [{ name: 'obj', has: { core: true } }, { name: 'hand', has: { core: true, final: false } }, { name: 'face', has: { core: false, final: false } }];
confirms.length = 0;
$('z4Region').querySelector('[data-r="outline"]').click(); await wait(100);
assert(confirms.length === 1 && /BEZ włosów/.test(confirms[0]), 'switching a drawn zone to outline asks to clear the drawing');
const z1 = srv.zones.z_01;
assert(z1.params.region === 'outline' && z1.polygon.length === 0 && z1.strokes.length === 0 && z1.bg_strokes.length === 1 && z1.params.fg_erode_px === 8,
  `outline, hair drawing cleared, "-" kept: ${JSON.stringify({ p: z1.params, poly: z1.polygon, s: z1.strokes, b: z1.bg_strokes })}`);
assert($('z4OutRow').style.display === '' && $('z4ExclRow').style.display === '' && $('z4RegionHint').textContent.includes('NIE'), 'outline controls + hint');
assert($('zpTitle').textContent.includes('cały obrys') && $('z4Hint').textContent.includes('BEZ włosów'), 'title + card hint say outline');
const exBoxes = $('z4Excl').querySelectorAll('input[data-ex]');
assert(exBoxes.length === 2 && !exBoxes[0].disabled && exBoxes[1].disabled, 'other objects listed (not this one), one without a mask disabled');
exBoxes[0].checked = true; exBoxes[0].dispatchEvent(new window.Event('change', { bubbles: true })); await wait(40);
assert(JSON.stringify(srv.zones.z_01.params.exclude_objects) === '["hand"]', `hand excluded: ${JSON.stringify(srv.zones.z_01.params)}`);
$('z4Out').value = '60'; $('z4Out').dispatchEvent(new window.Event('input')); await wait(5);
assert(ZN.state.inset && ZN.state.inset.out === 60 && $('z4OutV').textContent === '60 px', 'live band preview state');
ZN.draw(fakeCtx(), fakeImg, 1, viewer.currentFrame);
assert(ZN.state.insetDist && ZN.state.insetDist.dOut, 'outside distance map for the band');
$('z4Out').dispatchEvent(new window.Event('change')); await wait(40);
assert(srv.zones.z_01.params.outline_out_px === 60 && srv.zones.z_01.params.fg_erode_px === 8 && srv.zones.z_01.params.exclude_objects.length === 1, 'band saved, params merged');
$('z4BlackAfter').value = '0.1'; $('z4BlackAfter').dispatchEvent(new window.Event('change')); await wait(30);
assert(srv.zones.z_01.params.alpha_black_after === 0.1 && $('z4BlackAfterV').textContent === '0.10', 'black after smoothing');
calls.length = 0;
viewer.goToFrame(118); await wait(450);
assert(ZN.state.regionCache.z_01[118] && ZN.state.regionCache.z_01[118].exclude_contours.length === 1, 'outline region with exclusion contours');
ZN.draw(fakeCtx(), fakeImg, 1, 118);
S.strokes[118] = [{ mode: 'add', radius_px: 6, p: [[200, 200]] }]; ZN.onDraw(118); await wait(750);
const putEx = calls.filter((c) => c[0] === 'PUT' && c[1].endsWith('/keys/118')).pop();
assert(putEx && putEx[2].polygon[0][0] === 50 && putEx[2].strokes.length === 1, `new key starts from the exclusion, not from the band: ${JSON.stringify(putEx && putEx[2])}`);
viewer.goToFrame(115); await wait(30); calls.length = 0;
$('z4PvFrame').click(); await wait(30);
assert(calls.some((c) => c[0] === 'POST' && c[1].endsWith('/preview/115')), '"Przelicz klatkę" = preview of the current frame');
assert($('z4PvInfo').textContent.includes('cały obrys'), `preview info says outline: ${$('z4PvInfo').textContent}`);
await fakeApi('DELETE', '/api/object/obj/zones/z_01/keys/118'); await ZN.load();
$('z4Region').querySelector('[data-r="draw"]').click(); await wait(60);
assert(srv.zones.z_01.params.region === 'draw' && $('z4OutRow').style.display === 'none', 'back to the drawn area');
srv.zones.z_01.polygon = [[300, 100], [420, 100], [420, 240], [300, 240]]; await ZN.load();
// 2e. rotobrush-like painted band (user decision 2026-10-15): brush paint / erase / bg on the palette, local tracking
assert($('z4Region').querySelector('[data-r="draw"]').classList.contains('active') && $('z4TrackRow').style.display === '', 'painted area: tracking option shown');
$('z4Track').checked = true; $('z4Track').dispatchEvent(new window.Event('change')); await wait(40);
assert(srv.zones.z_01.params.track === 'local' && srv.zones.z_01.params.fg_erode_px === 8, `local tracking saved: ${JSON.stringify(srv.zones.z_01.params)}`);
$('z4Brush').querySelector('[data-b="erase"]').click();
assert(ZN.state.brush === 'erase' && $('z4Brush').querySelector('[data-b="erase"]').classList.contains('active'), 'erase brush on');
const bgBefore = srv.zones.z_01.bg_strokes.length; calls.length = 0;
S.strokes[115] = [{ mode: 'remove', radius_px: 5, p: [[310, 120]] }]; ZN.onDraw(115); await wait(750);
const putE = calls.filter((c) => c[0] === 'PUT' && c[1].endsWith('/keys/115')).pop();
assert(putE && putE[2].strokes.some((x) => x.erase) && putE[2].bg_strokes.length === bgBefore, `erase stroke stored in the drawing: ${JSON.stringify(putE && putE[2])}`);
ZN.draw(fakeCtx(), fakeImg, 1, 115);
$('z4Brush').querySelector('[data-b="erase"]').click(); assert(ZN.state.brush === null, 'erase brush off');
ZN.state.ov = 'alpha';
$('z4Engine').value = 'auto'; $('z4Engine').dispatchEvent(new window.Event('change')); await wait(30);
assert($('z4AreaSec').style.display === 'none' && $('z4BlackAfterRow').style.display === 'none', 'area / black-after hidden for a non-ViTMatte engine');

// 3. overlays draw without throwing for every view
const ctx = fakeCtx();
for (const ov of ['alpha', 'comp', 'trimap', 'engines', 'none']) { $('z4Ov').querySelector(`[data-ov="${ov}"]`).click(); ZN.draw(ctx, fakeImg, 0.5, 115); }
$('z4Ov').querySelector('[data-ov="alpha"]').click();
$('z4Loupe').click(); assert(ZN.state.loupe, 'loupe on');
const Ev = window.PointerEvent || window.MouseEvent; canvas.dispatchEvent(new Ev('pointermove', { clientX: 100, clientY: 80 }));
ZN.state.mouse = [100, 80]; canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 360 });
ZN.drawLoupe(ctx, fakeImg, 0.5); $('z4Loupe').click(); assert(!ZN.state.loupe, 'loupe off');

// 4. frame without a key: region fetched; a "-" stroke there creates a key from the propagated region
calls.length = 0;
viewer.goToFrame(120); await wait(450);
assert(calls.some((c) => c[0] === 'GET' && c[1].endsWith('/region/120')), 'region fetched on frame change');
assert(ZN.state.regionCache.z_01[120].contours.length === 2, 'region cached');
assert($('z4Hint').textContent.includes('bez klucza'), 'hint: no key here');
ZN.draw(ctx, fakeImg, 0.5, 120);
S.strokes[120] = [{ mode: 'remove', radius_px: 3, p: [[350, 150], [360, 160]] }];
ZN.onDraw(120); await wait(750);
const put = calls.find((c) => c[0] === 'PUT' && c[1].endsWith('/keys/120'));
assert(put && put[2].polygon.length === 4 && put[2].polygon[0][0] === 300 && put[2].bg_strokes.length === 1 && put[2].strokes.length === 0, 'key from the largest region contour + "-" stroke');
assert(sortedKeys(srv.zones.z_01).join() === '115,120', 'server keys');
assert($('timelineZones').querySelectorAll('.zone-key').length === 2, 'lane: 2 diamonds');
assert(toasts.some((t) => /Klucz strefy F120/.test(t)), 'toast new key');
assert(document.querySelector('.zone-card .zone-meta').textContent.includes('🔑 2'), 'card keys = 2');
// a "+" stroke in the key frame extends the key (no new polygon); step 7: the iteration note goes with it into the key
assert($('z4Iter') && $('z4Iter').title, 'iteration note field with tooltip');
$('z4Iter').value = 'mniej tła przy uchu';
S.strokes[120] = [{ mode: 'add', radius_px: 8, p: [[400, 200]] }]; ZN.onDraw(120); await wait(750);
assert(srv.zones.z_01.keys['120'].strokes.length === 1 && srv.zones.z_01.keys['120'].bg_strokes.length === 1, 'key extended');
const putNote = calls.filter((c) => c[0] === 'PUT' && c[1].endsWith('/keys/120')).pop();
assert(putNote[2].prompt === 'mniej tła przy uchu' && srv.zones.z_01.keys['120'].prompt === 'mniej tła przy uchu', 'note sent with the key');
assert(srv.zones.z_01.prompt_history.includes('mniej tła przy uchu'), 'note in prompt_history');
assert($('z4Iter').value === '', 'note field cleared after the key');
assert($('z4Hint').textContent.includes('uwaga: mniej tła przy uchu'), `hint shows the key note: ${$('z4Hint').textContent}`);
assert([...$('timelineZones').querySelectorAll('.zone-key')].some((d) => d.title.includes('mniej tła')), 'lane diamond tooltip has the note');
assert(lastStatus()[0].includes('uwaga „mniej tła przy uchu”'), 'status mentions the note');
// the alpha overlay draws on the plate tiles when the viewer has them (Z8): tileOverlays hook used
tileCalls.length = 0; ZN.draw(ctx, fakeImg, 0.5, 120); assert(tileCalls.length === 1 && tileCalls[0] === 'zone_alpha', 'zone alpha asked for tile overlays');
// Ctrl+Z twice: stroke (restores the key WITHOUT the note), then the key itself
assert(await ZN.undo(120), 'undo handled'); await wait(20);
assert(srv.zones.z_01.keys['120'].strokes.length === 0 && !srv.zones.z_01.keys['120'].prompt, 'undo removed the + stroke and the note');
assert(await ZN.undo(120), 'undo 2 handled'); await wait(20);
assert(sortedKeys(srv.zones.z_01).join() === '115', 'undo deleted the created key');

// 4b. step 9: prompt rules line, Hough "-" strokes into the key, proposals -> zone
for (const id of ['z4Plan', 'z4Propose', 'z4Props']) assert($(id), `id ${id}`);
viewer.goToFrame(115); await wait(50);
calls.length = 0;
$('z4Prompt').value = 'włosy bez linii'; $('z4Prompt').dispatchEvent(new window.Event('change')); await wait(60);
assert(calls.some((c) => c[0] === 'PATCH' && c[2].prompt === 'włosy bez linii'), 'prompt PATCH');
assert($('z4Plan').textContent.includes('preset:') && $('z4Plan').textContent.includes('włosy / futro'), `plan line: ${$('z4Plan').textContent}`);
assert(calls.some((c) => c[1] === '/api/object/obj/zones/interpret' && c[2].frame === 115), 'lines searched automatically for a line prompt');
const lb = $('z4Lines'); assert(lb && lb.textContent.includes('Dodaj 1'), `lines button: ${lb && lb.textContent}`);
lb.click(); await wait(60);
const putL = calls.filter((c) => c[0] === 'PUT' && c[1].endsWith('/keys/115')).pop();
assert(putL && putL[2].bg_strokes.length === 2 && putL[2].bg_strokes[1].p[0][0] === 100, 'Hough stroke appended to the key as "-"');
assert(srv.zones.z_01.keys['115'].bg_strokes.length === 2, 'server key has 2 "-" strokes');
assert(await ZN.undo(115), 'undo lines'); await wait(20);
assert(srv.zones.z_01.keys['115'].bg_strokes.length === 1, 'undo removed the Hough stroke');
$('z4Propose').click(); await wait(60);
assert(calls.some((c) => c[1] === '/api/object/obj/zones/propose' && c[2].frame === 115), 'propose called');
assert(document.querySelectorAll('.zone-prop').length === 2 && ZN.state.props.list.length === 2, 'two proposals listed');
ZN.draw(ctx, fakeImg, 0.5, 115);
document.querySelector('.zone-prop[data-i="1"] button[data-rej]').click(); assert(document.querySelectorAll('.zone-prop').length === 1, 'rejected one');
document.querySelector('.zone-prop[data-i="0"] button[data-acc]').click(); await wait(60);
const newZ = calls.filter((c) => c[0] === 'POST' && /zones\/new/.test(c[1])).pop();
assert(newZ && newZ[2].polygon.length === 4 && newZ[2].polygon[0][0] === 300 && newZ[2].preset === 'hair', 'zone created from the proposal');
assert(Object.keys(srv.zones).length === 2 && ZN.state.active === 'z_02' && ZN.state.props === null, 'proposal zone active, proposals gone');
assert(document.querySelectorAll('.zone-card').length === 2, 'two cards');
await fakeApi('DELETE', '/api/object/obj/zones/z_02'); await ZN.load(); ZN.state.active = 'z_01'; await wait(10);
viewer.goToFrame(120); await wait(50);

// 5. "Zastosuj": 409 missing_front -> dialog -> "bez holdoutu" -> PATCH + apply ok
srv.hand = true; calls.length = 0; statuses.length = 0;
$('z4ApplyShot').click(); await wait(60);
const modal = $('zoneFrontModal');
assert(modal && modal.style.display === 'flex' && modal.textContent.includes('hand'), 'objects-in-front dialog shown');
modal.querySelector('[data-act="noholdout"]').click(); await wait(120);
assert(srv.zones.z_01.holdout === false, 'holdout off via PATCH');
assert(srv.activeRun === 'v002' && srv.holdoutRefusals === 1, 'apply retried and succeeded');
assert(lastStatus()[0].includes('Wersja v002') && lastStatus()[0].includes('przeliczone 12') && lastStatus()[1] === 'ok', `status after apply: ${lastStatus()[0]}`);
assert(cleared === 1, 'overlays cleared once after the new version');
assert(document.querySelector('.zone-card .zs').textContent.includes('policzona'), 'card state policzona');
assert(document.querySelector('.zone-card .zone-meta').textContent.includes('s/kl.'), 'card shows s/frame');
// the queue button of the dialog
srv.zones.z_01.holdout = true; calls.length = 0;
$('z4ApplyShot').click(); await wait(60);
$('zoneFrontModal').querySelector('[data-act="queue"]').click(); await wait(60);
assert(calls.some((c) => c[1] === '/api/queue/preset/all') && calls.some((c) => c[1] === '/api/queue/run'), 'queue-all + run');
assert(lastStatus()[0].includes('Policz wszystko'), 'queue status');
srv.hand = false;

// 6. enable / disable from the card; version details html; delete key (last) refused; delete zone
document.querySelector('.zone-card input[data-en]').click(); await wait(30);
assert(srv.zones.z_01.enabled === false && document.querySelector('.zone-card').classList.contains('off'), 'disabled');
document.querySelector('.zone-card input[data-en]').click(); await wait(30);
assert(srv.zones.z_01.enabled === true, 'enabled again');
const det = ZN.runDetailsHtml({ run: 'v002', applied_from: 'v001', zones: { z_01: { engine: 'auto', frames: 40, seconds: 48, s_per_frame: 1.2, computed: 12, cached: 28, engine_share: { chroma: 0.6, knn: 0.4 } } } });
assert(det.includes('1.2 s/kl.') && det.includes('chroma 60 %') && det.includes('v001'), 'run details html');
viewer.goToFrame(115); await wait(10);
$('z4KeyDel').click(); await wait(20);
assert(toasts.some((t) => /Ostatniego klucza/.test(t)), 'last key refused');
$('z4CpNone').click(); await wait(20);
assert($('z4CleanTxt').textContent.includes('pominięty'), 'clean plate skipped text');
$('z4CpFile').click(); assert(opened.length === 1, 'Mam plik -> plates modal');
document.querySelector('.zone-card button[data-del]').click(); await wait(40);
assert(Object.keys(srv.zones).length === 0 && ZN.state.active === null && $('st4Zones').textContent.includes('brak stref'), 'zone deleted');
assert(confirms.some((m) => /^Usunąć strefę z_01/.test(m)), 'zone delete asks through viewer.uiConfirm');
// palette without a zone: "＋ Strefa ViTMatte: cały obrys" creates an outline zone without any drawing
assert($('zpNone').style.display === '' && $('z4Tune').style.display === 'none', 'palette offers a new outline zone');
calls.length = 0;
$('zpNewOutline').click(); await wait(80);
const postO = calls.find((c) => c[0] === 'POST' && /zones\/new/.test(c[1]));
assert(postO && postO[2].polygon.length === 0 && postO[2].matting === 'vitmatte' && postO[2].params.region === 'outline', `outline zone body: ${JSON.stringify(postO && postO[2])}`);
assert(ZN.state.active && srv.zones[ZN.state.active].params.region === 'outline' && $('z4Tune').style.display === '', 'outline zone active, sliders shown');
await fakeApi('DELETE', `/api/object/obj/zones/${ZN.state.active}`); await ZN.load();
// "🖌 Maluj obszar włosów (ViTMatte)": the first paint stroke creates a ViTMatte zone with local tracking
$('zpNewPaint').click(); assert(ZN.state.newPainted && ZN.state.brush === 'paint', 'painted-zone mode armed');
calls.length = 0;
S.strokes[115] = [{ mode: 'add', radius_px: 20, p: [[300, 100], [420, 110]] }]; ZN.onDraw(115); await wait(750);
const postP = calls.find((c) => c[0] === 'POST' && /zones\/new/.test(c[1]));
assert(postP && postP[2].matting === 'vitmatte' && postP[2].params.track === 'local' && postP[2].strokes.length === 1 && !ZN.state.newPainted,
  `painted zone body: ${JSON.stringify(postP && postP[2])}`);
await fakeApi('DELETE', `/api/object/obj/zones/${ZN.state.active}`); ZN.state.active = null; await ZN.load(); $('z4Brush').querySelector('[data-b="paint"]').click();
// another stage (Eksperymenty: the old threshold criteria live there) -> no zone flow
S.stage = ST.EXPERIMENTS; assert(!ZN.wantsDraw(), 'stage Eksperymenty = no zone flow');

// ---------------------------------------------------------------- stage CoTracker: brush corrections (static/brush.js)
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
// d. "Zmaż" in F120 -> erase strokes into the keys of both zones (remove zone: from its propagated region)
calls.length = 0;
S.strokes[120] = [{ mode: 'erase', radius_px: 8, p: [[315, 115]] }];
BR.onDraw(120); await wait(500);
const puts = calls.filter((c) => c[0] === 'PUT' && /\/keys\/120$/.test(c[1]));
assert(puts.length === 2 && puts.every((c) => c[2].strokes.some((s) => s.erase)), `erase in both zones: ${puts.length}`);
// e. Ctrl+Z restores the key before the erase
calls.length = 0;
assert(await BR.undo(120), 'brush undo handled');
assert(calls.some((c) => (c[0] === 'PUT' || c[0] === 'DELETE') && /\/keys\/120$/.test(c[1])), 'undo touched the key 120');
// f. empty key = end of the corrections in F130 (both zones)
calls.length = 0;
viewer.goToFrame(130);
$('b2EmptyKey').click(); await wait(50);
const empt = calls.filter((c) => c[0] === 'PUT' && /\/keys\/130$/.test(c[1]));
assert(empt.length === 2 && empt.every((c) => !c[2].polygon.length && !c[2].strokes.length), 'empty keys in F130');
// g. drawing in stage 2 (keys + propagated regions) does not throw; the lane shows the keys
BR.draw(window.HTMLCanvasElement.prototype.getContext(), fakeImg, 0.5, 120);
BR.draw(window.HTMLCanvasElement.prototype.getContext(), fakeImg, 0.5, 125);
assert($('timelineBrush') && $('timelineBrush').querySelectorAll('.zone-key').length >= 4, 'brush lane diamonds');
// h. „Zastosuj poprawki” -> POST core_fix, view Rdzeń
calls.length = 0; views.length = 0;
$('b2Apply').click(); await wait(50);
assert(srv.coreFix === 1 && views.includes('core') && lastStatus()[0].includes('Maska 1+2'), `apply: ${lastStatus()[0]}`);
// i. Delete in F130 -> DELETE keys/130 of both zones (with a confirmation)
calls.length = 0; confirms.length = 0;
await BR.deleteKeysAt(130);
assert(confirms.length === 1 && calls.filter((c) => c[0] === 'DELETE' && /\/keys\/130$/.test(c[1])).length === 2, 'delete keys in F130');
// j. brush zones never appear in the ViTMatte list
assert(!ZN.state.list.some((z) => z.params && z.params.source === 'brush'), 'brush zones hidden from the ViTMatte panel');

console.log(`OK zones harness: ${calls.length + 1} api calls in the last block, ${toasts.length} toasts, ${statuses.length} statuses, renders ${viewer.renders}`);
