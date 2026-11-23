// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/cleanup3d_horizon_harness.mjs – headless run of the C5 section „Horyzont” (static/cleanup3d_horizon.js,
// package P15) in jsdom with a fake API / viewer: static/cleanup.js + static/cleanup3d.js + static/curves.js + the module.
// Checks: the section is in the C5 palette, „Dopasuj horyzont” / „Zasięg dopasowania” are saved through
// PUT /api/cleanup3d/render_settings (the patch blend sliders keep them), „✎ Narysuj horyzont” measures the frame
// (GET /api/cleanup3d/horizon/{f}) and starts the line from the plate's measured horizon with THE curve editor
// (MD_CURVES.Editor), a finished edit saves PUT /api/cleanup3d/horizon/{f}, 🗑 removes it, the status line says the
// offset and the step before → after, Esc ends drawing.
// Run: cd tests/js && npm ci && node cleanup3d_horizon_harness.mjs   (tests/test_cleanup3d_horizon.py wraps it)
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.resolve(here, '..', '..', 'static');

const html = `<!doctype html><html><body>
  <header class="app-header"><div class="header-left"></div><div class="header-center">
    <select id="plateSelect"><option value="beauty">Beauty</option></select>
    <div class="view-mode-tabs" id="viewModeTabs"><button class="tab-btn" data-view="none">Plate</button></div></div></header>
  <nav class="stage-bar" id="stageBar"><div class="stage-object" id="stageObjectBox">obj</div>
    <ol class="stage-list" id="stageList"></ol><div class="stage-hint" id="stageHint"></div></nav>
  <aside class="tools-sidebar" id="toolsSidebar"></aside>
  <div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div>
  <aside class="stage-panel" id="stagePanel"></aside>
  <div id="timelineTrackContainer"><div id="timelineKeys"></div></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;
window.HTMLCanvasElement.prototype.getContext = function () { return new Proxy({}, { get: () => () => ({ data: [] }) }); };

const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms = 30) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- fake server
const srv = { rs: { key_soft_px: 32, key_grow_px: 16, key_effects: true, key_effects_thr: 0.1, clean_first: false,
  horizon_match: false, horizon_fade_px: 120, horizon_search_px: 60 }, lines: {} };
const state = () => ({ steps: ['remove', 'camera', 'cards', 'keys', 'project', 'export'], done: { camera: true, cards: true, keys: true },
  camera: { frames: [1, 100], n_frames: 100, source: 'test', report: {} }, cards: [], render_settings: { ...srv.rs },
  engines: [], key_modes: ['manual'], views: ['key', 'proj', 'cards'], key_formats: ['png'] });
const plateLine = Array.from({ length: 40 }, (_, i) => [i * 98.0, 800 + i * 0.5]);
const calls = [];
async function fakeApi(method, url, body) {
  calls.push([method, url, body ? JSON.parse(JSON.stringify(body)) : body]);
  if (method === 'GET' && url === '/api/cleanup') return { objects: [], settings: { remove: [], sources: {}, grow_px: 8, soft_px: 4, holdout: true, fill: 'auto' }, sources: {}, versions: [], out_dir: 'W/inpaint' };
  if (method === 'GET' && url === '/api/cleanup3d') return state();
  if (method === 'PUT' && url === '/api/cleanup3d/render_settings') { Object.assign(srv.rs, body); return { render_settings: { ...srv.rs } }; }
  let m = /^\/api\/cleanup3d\/horizon\/(\d+)$/.exec(url);
  if (m && method === 'GET') {
    const f = +m[1];
    return { frame: f, status: 'ok', source: srv.lines[f] ? 'user' : 'auto', warped: true, offset_px: 2.4, offset_max_px: 3.1,
      step_px: 2.2, step_p90_px: 3.0, step_after_px: 0.3, step_after_p90_px: 0.5, plate: plateLine, fill: plateLine.map(([x, y]) => [x, y + 2.4]),
      predicted: plateLine, user: srv.lines[f] || null, drawn: Object.keys(srv.lines).map(Number).sort((a, b) => a - b), settings: {} };
  }
  if (m && method === 'PUT') { srv.lines[+m[1]] = body.curve; return { success: true, drawn: Object.keys(srv.lines).map(Number) }; }
  if (m && method === 'DELETE') { delete srv.lines[+m[1]]; return { success: true, drawn: Object.keys(srv.lines).map(Number) }; }
  if (method === 'GET' && /^\/api\/cleanup3d\/band\//.test(url)) return { frame: 30, points: [], defaults: {} };
  throw Object.assign(new Error(`fake api: ${method} ${url}`), { status: 404 });
}
window.fetch = async () => ({ ok: false, status: 404, statusText: 'nope', json: async () => ({}) });
const toasts = [];
window.UI = { exportBtn: () => '<button class="md-export">Export</button>', exportMenu: () => {}, api: fakeApi, toast: (t) => toasts.push(String(t)), dialog: async () => true, confirmDelete: async () => true, bytes: (n) => `${n} B` };

// ---------------------------------------------------------------- fake viewer
const cv = $('cv');
const viewer = {
  projectLoaded: true, currentWorkDir: 'D:/W/shot/matte_desk', manifest: { plates: { beauty: 'b.####.exr' }, plate_w: 3840, plate_h: 2160 },
  firstFrame: 1, lastFrame: 100, currentFrame: 30, activePlate: 'beauty', canvas: cv, zoom: 1, panX: 0, panY: 0, overlayHooks: [],
  getCurrentImage: () => ({ width: 640, height: 360 }),
  setActivePlate(n) { this.activePlate = n; },
  goToFrame(f) { this.currentFrame = f; window.dispatchEvent(new window.CustomEvent('md-frame', { detail: { frame: f } })); },
  render() {},
};
window.viewer = viewer;

// ---------------------------------------------------------------- load the scripts (index.html order)
for (const f of ['cleanup.js', 'cleanup3d.js', 'curves.js', 'cleanup3d_horizon.js']) window.eval(readFileSync(path.join(STATIC, f), 'utf-8'));
await wait();
const C3 = window.MD_CLEANUP3D;
assert(C3 && C3.ready, 'MD_CLEANUP3D ready');
document.querySelector('#modeSwitch [data-mode="clean"]').click();
await wait();
C3.setStep('project');
await wait(100);
const HZ = window.MD_C3HORIZON;
assert(HZ && HZ.ready, 'MD_C3HORIZON ready');
assert($('c3PalProject').contains($('c3HzSec')), 'the section is in the C5 palette');
assert($('c3HzCanvas') && $('c3HzCanvas').classList.contains('c3-area-canvas'), 'own canvas over the viewer');
assert(HZ.state.ed instanceof window.MD_CURVES.Editor, 'THE curve editor');
assert($('c3HzOn').checked === false && +$('c3HzFade').value === 120, 'settings from the state');

// ---------------------------------------------------------------- settings
$('c3HzOn').checked = true; $('c3HzOn').dispatchEvent(new window.Event('change'));
await wait(400);
let put = calls.filter(([m, u]) => m === 'PUT' && u === '/api/cleanup3d/render_settings').pop();
assert(put && put[2].horizon_match === true && put[2].horizon_fade_px === 120, 'horizon_match saved');
$('c3HzFade').value = '200'; $('c3HzFade').dispatchEvent(new window.Event('change'));
await wait(400);
put = calls.filter(([m, u]) => m === 'PUT' && u === '/api/cleanup3d/render_settings').pop();
assert(put[2].horizon_fade_px === 200 && put[2].horizon_match === true, 'fade saved with the switch');
assert(C3.state.render_settings.horizon_fade_px === 200, 'state updated');
// a patch blend slider keeps the horizon settings in the state
$('c3KeySoft').value = '40'; $('c3KeySoft').dispatchEvent(new window.Event('input'));
assert(C3.state.render_settings.horizon_match === true && C3.state.render_settings.key_soft_px === 40, 'blend sliders keep the horizon keys');

// ---------------------------------------------------------------- drawing a line
$('c3HzDraw').click();
await wait(100);
assert(HZ.state.drawing && $('c3HzDraw').classList.contains('active'), 'drawing on');
assert(calls.some(([m, u]) => m === 'GET' && u === '/api/cleanup3d/horizon/30'), 'frame measured');
const cur = HZ.state.curve;
assert(cur && cur.type === 'bezier' && !cur.closed && cur.pts.length === 6, 'line started from the plate horizon');
assert(Math.abs(cur.pts[0].p[1] - 800) < 1e-6, 'first point on the plate line');
assert($('c3HzCanvas').classList.contains('c3-area-on'), 'canvas takes the mouse while drawing');
const info = $('c3HzInfo').textContent;
assert(info.includes('2.4') && info.includes('2.2 → 0.3') && info.includes('automat'), `status: ${info}`);
const lay = HZ.layers();
assert(lay.some((l) => l.id === 'hz' && l.editable) && lay.some((l) => l.id === '+plate' && l.editable === false), 'layers');
cur.pts[2].p[1] += 5;
HZ.state.ed.o.onChange('hz', cur, true);                       // a finished edit of the editor
await wait(400);
const pl = calls.filter(([m, u]) => m === 'PUT' && u === '/api/cleanup3d/horizon/30').pop();
assert(pl && pl[2].curve.pts.length === 6 && Math.abs(pl[2].curve.pts[2].p[1] - cur.pts[2].p[1]) < 1e-9, 'line saved');
assert($('c3HzDrawn').textContent.includes('30') && !$('c3HzDel').disabled, 'drawn frames listed');
assert($('c3HzInfo').textContent.includes('linia narysowana'), 'source = the drawn line');
// Esc ends drawing
document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
assert(!HZ.state.drawing && !$('c3HzCanvas').classList.contains('c3-area-on'), 'Esc ends drawing');
// delete
$('c3HzDel').click();
await wait(100);
assert(calls.some(([m, u]) => m === 'DELETE' && u === '/api/cleanup3d/horizon/30'), 'line deleted');
assert($('c3HzDrawn').textContent.includes('automat') && $('c3HzDel').disabled, 'no drawn lines');
// statuses
assert(HZ.infoText({ frame: 5, status: 'no_plate_line' }).includes('narysuj'), 'no plate line text');
assert(HZ.infoText({ frame: 5, status: 'ok', source: 'auto', offset_px: 1, step_px: null }).includes('nie przecina'), 'no step text');
assert(!toasts.length, `no toasts: ${toasts.join(' | ')}`);
console.log('OK cleanup3d horizon harness');
