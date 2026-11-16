// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/cleanup3d_harness.mjs – headless run of static/cleanup.js + static/cleanup3d.js (Cleanup 3D steps C1–C6,
// docs/PLAN_CLEANUP_3D.md §2, package P4) in jsdom with a fake API / viewer. The floating palette code is the REAL one:
// placeFloating / makeFloating are cut out of static/objects.js and evaluated here.
// Checks: the stage bar switches to the steps and back (object stage bar untouched), step states from a mocked
// GET /api/cleanup3d, C1 = the MVP panel, palettes per step drag + double click back to the corner, the card order
// saved with ONE PUT, an engine without its model disabled with the reason, a 501 answer shown as a toast with the
// server's text, the layered card timeline of P16 (rows, diamonds click / drag-move, eye = off and solo stored on the
// card like After Effects (PUT, no preview query), coverage warnings, the patch band row, fold), the preview switch, the step remembered per project,
// the key modes of P13 (automat: settings saved on the card + request; jeden klucz: dialog before replacing the keys).
// Run: cd tests/js && npm ci && node cleanup3d_harness.mjs   (tests/test_cleanup3d_js.py wraps it; skipped without jsdom)
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.resolve(here, '..', '..', 'static');

const ORIG_STAGES = '<li class="stage-item done current" data-n="1"><span class="stage-n">1</span><span class="stage-label">SAM 3.1</span></li>';
const html = `<!doctype html><html><body>
  <header class="app-header"><div class="header-left"></div><div class="header-center">
    <select id="plateSelect"><option value="beauty">Beauty</option><option value="cleaned">cleaned</option></select>
    <div class="view-mode-tabs" id="viewModeTabs"><button class="tab-btn" data-view="none">Plate</button></div></div></header>
  <nav class="stage-bar" id="stageBar"><div class="stage-object" id="stageObjectBox">obj</div>
    <ol class="stage-list" id="stageList">${ORIG_STAGES}</ol><div class="stage-hint" id="stageHint">hint etapu</div></nav>
  <aside class="tools-sidebar" id="toolsSidebar"><div id="drawToolsTitle">Rysuj</div><div id="drawTools"></div></aside>
  <div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div>
  <aside class="stage-panel" id="stagePanel"><div class="stage-panel-empty" id="stagePanelEmpty" style="display:none"></div>
    <div class="stage-pane" id="objPane">etap obiektu</div><div class="stage-pane" id="objPaneHidden" style="display:none">inny etap</div></aside>
  <div id="timelineTrackContainer"><div id="timelineKeys"></div></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;
// the REAL stylesheet: Cleanup mode hides the cutting stages' panes with a rule of index.css, not with inline styles
{ const st = document.createElement('style'); st.textContent = readFileSync(path.join(STATIC, 'index.css'), 'utf-8'); document.head.appendChild(st); }
window.HTMLCanvasElement.prototype.getContext = function () { return new Proxy({}, { get: () => () => ({ data: [] }) }); };
window.confirm = () => { throw new Error('native confirm() used'); };

const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms = 15) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);
const shown = (id) => { const e = $(id); return !!e && window.getComputedStyle(e).display !== 'none'; };

// ---------------------------------------------------------------- fake server
const DETAIL = (p) => `Czyszczenie 3D: ta funkcja jeszcze nie jest zbudowana (${p})`;
const REASON = 'Brak modelu FLUX 3 Image Klein – Doktor → Zainstaluj FLUX 3 Image Klein';
const card = (id, name, kind, color, order, keys) => ({ id, name, kind, plane: kind === 'plane' ? { n: [0, -1, 0], d: 1 } : null,
  adjust: { dh: 0, pitch: 0, roll: 0 }, extent: [], extent_frame: null, seam_px: 16, order, color, key_mode: 'manual', key_every: 8, warps: {}, keys });
const srv = {
  cards: [card('sky', 'niebo', 'sky', '#4ae0ff', 0, [{ frame: 10, active: 'v001', versions: 1 }]),
    card('sea', 'morze', 'plane', '#ffb84a', 1, [{ frame: 10, active: 'v002', versions: 2 }, { frame: 60, active: 'v001', versions: 1 }])],
  kleinOk: false,
};
// key steps (2026-11-10): the version list of a key as GET /api/cleanup3d gives it (store.state `list`)
const VLIST = [{ v: 'v001', source: 'plate', date: '2026-11-10 10:00:00' },
  { v: 'v002', source: 'plate+fill:v001', date: '2026-11-10 10:05:00' },
  { v: 'v003', source: 'regen:klein_restore', base: 'v002', area: 'patch', strength: 0.9, prompt: 'wet sand', date: '2026-11-10 10:09:00' }];
const seaKey60 = () => srv.cards.find((c) => c.id === 'sea').keys.find((k) => k.frame === 60);
const state = () => ({ steps: ['remove', 'camera', 'cards', 'keys', 'project', 'export'], done: { camera: true, cards: 'stale', keys: false },
  camera: { frames: [1, 100], n_frames: 100, source: 'colmap+cotracker3', report: { reproj_px: 0.4321, key_error_px: 1.5, horizon_px: 0.8, worst_frames: [{ frame: 77, err_px: 2.5 }, 12] } },
  cards: srv.cards.map((c) => ({ ...c })),
  engines: [{ id: 'lama', label: 'LaMa' }, srv.kleinOk ? { id: 'klein_restore', label: 'FLUX 3 Image Klein + LoRA Restore', available: true }
    : { id: 'klein_restore', label: 'FLUX 3 Image Klein + LoRA Restore', available: false, reason: REASON }],
  key_modes: ['manual', 'every', 'auto', 'one'], views: ['key', 'proj', 'cards'], key_formats: ['png', 'png16', 'exr'] });
const calls = [];
const err = (status, detail) => { const e = new Error(detail); e.status = status; e.detail = detail; return e; };
async function fakeApi(method, url, body) {
  calls.push([method, url, body ? JSON.parse(JSON.stringify(body)) : body]);
  if (method === 'GET' && url === '/api/cleanup') return { objects: [], settings: { remove: [], sources: {}, grow_px: 8, soft_px: 4, holdout: true, fill: 'auto' }, sources: {}, versions: [], out_dir: 'W/inpaint' };
  if (method === 'GET' && url === '/api/cleanup3d') return state();
  if (method === 'PUT' && url === '/api/cleanup3d/cards') {
    const keys = Object.fromEntries(srv.cards.map((c) => [c.id, c.keys]));
    srv.cards = body.cards.map((c, i) => ({ ...c, order: i, keys: keys[c.id] || [] }));
    return { schema: 'matte_desk/cards3d@1.0', cards: body.cards.map((c, i) => ({ ...c, order: i })) };
  }
  if (method === 'GET' && url === '/api/cleanup3d/depth_sources') return { sources: [{ id: 'computed', label: 'Głębia policzona w narzędziu (DepthCrafter)', available: true, conv: 'disparity' }, { id: 'plate', label: 'Plate głębi ujęcia', available: false, conv: 'disparity' }], convs: ['disparity', 'z'] };
  const dm = /^\/api\/cleanup3d\/cards\/(\w+)\/depth$/.exec(url);
  if (method === 'POST' && dm) {          // P18: the card becomes a depth card (snapshot + fit on the server)
    const c = srv.cards.find((x) => x.id === dm[1]);
    c.kind = 'depth';
    c.depth = { frame: body.frame, source: body.source || 'computed', conv: body.conv || 'disparity', scale: 0.5, offset: 0.01, mul: 1, shift: 0, edge: 0.04,
      file: `cards/${c.id}/depth_${String(body.frame).padStart(4, '0')}.npy`, stamp: 'x', fit: { method: 'points', anchors: 300, inliers: 280, err_rel: 0.02, z_median: 12 } };
    return { card: c, fit: c.depth.fit, source: { id: 'computed', label: 'DepthCrafter' } };
  }
  if (method === 'POST' && url === '/api/cleanup3d/camera/solve') throw err(501, DETAIL('P10 solve kamery'));
  if (method === 'POST' && /^\/api\/cleanup3d\/keys\/\w+\/\d+\/regen$/.test(url)) {
    if (!srv.kleinOk) throw err(501, DETAIL('P5 dogenerowanie + P8'));
    return { success: true, version: 'v004', base: body.base, area: body.area, engine: body.engine };
  }
  if (method === 'PUT' && url === '/api/cleanup3d/keys/sea/60/active') { seaKey60().active = body.version; return { success: true, version: body.version }; }
  if (method === 'DELETE' && url === '/api/cleanup3d/keys/sea/60/versions/v002') {
    const k = seaKey60(); k.list = k.list.filter((v) => v.v !== 'v002'); k.versions = k.list.length; return { success: true, freed_bytes: 2 * 1048576 };
  }
  if (method === 'POST' && url === '/api/cleanup3d/render') throw err(501, DETAIL('P6 projekcja i składanie + P8'));
  if (method === 'GET' && url === '/api/cleanup3d/timeline') {              // P16: coverage per card per frame
    const frames = Array.from({ length: 100 }, (_, i) => i + 1);
    return { frames, step: 20, reason: '', band_keys: [20, 40], off: srv.cards.filter((c) => c.off).map((c) => c.id),
      cards: { sky: { share: frames.map(() => 0.3), miss: frames.map(() => 0) },
        sea: { share: frames.map(() => 0.5), miss: frames.map((f) => (f > 80 ? 0.2 : 0.01)) } } };
  }
  { const m = /^\/api\/cleanup3d\/cards\/(\w+)\/off$/.exec(url);
    if (method === 'PUT' && m) { const c = srv.cards.find((x) => x.id === m[1]); c.off = body.off; return { success: true, card: { ...c } }; } }
  { const m = /^\/api\/cleanup3d\/cards\/(\w+)\/solo$/.exec(url);
    if (method === 'PUT' && m) { const c = srv.cards.find((x) => x.id === m[1]); if (body.solo) c.solo = true; else delete c.solo; return { success: true, card: { ...c } }; } }
  { const m = /^\/api\/cleanup3d\/keys\/(\w+)\/(\d+)\/move$/.exec(url);
    if (method === 'POST' && m) {
      const c = srv.cards.find((x) => x.id === m[1]); const k = c.keys.find((x) => x.frame === +m[2]);
      k.frame = body.to; c.keys.sort((p, q) => p.frame - q.frame);
      return { success: true, from: +m[2], to: body.to, patched: true, covered: 0.97, freed_bytes: 1024 };
    } }
  if (method === 'POST' && url === '/api/cleanup3d/keys/sea/auto') return { success: true, frames: [1, 30, 60], added_keys: [1, 30], max_error: 0.021, threshold: body.threshold, errors: {} };
  if (method === 'POST' && url === '/api/cleanup3d/keys/sea/master') return { success: true, key_frame: body.frame, canvas: { w: 4800, h: 2700 }, frames: [30, 29, 31], hole_px: 120, replaced: [], freed_bytes: 0 };
  throw err(404, `fake api: ${method} ${url}`);
}
const fetches = [];
window.fetch = async (url) => { fetches.push(url); return { ok: false, status: 501, statusText: 'Not Implemented', json: async () => ({ detail: DETAIL('P6 projekcja i składanie + P8') }) }; };
const toasts = [];
window.UI = { api: fakeApi, toast: (t) => toasts.push(String(t)), dialog: async () => true, confirmDelete: async () => true,
  bytes: (n) => `${((+n || 0) / 1048576).toFixed(1).replace('.', ',')} MB` };
const lastToast = () => toasts[toasts.length - 1] || '';

// ---------------------------------------------------------------- real floating palette code (static/objects.js)
const objSrc = readFileSync(path.join(STATIC, 'objects.js'), 'utf-8');
const fl = objSrc.slice(objSrc.indexOf('  function placeFloating('), objSrc.indexOf('  const othersPos'));
assert(fl.includes('function makeFloating(') && fl.includes('dblclick'), 'makeFloating found in objects.js');
let stageBarRenders = 0;
window.eval(`(function () { ${fl}; window.MD = { placeFloating, makeFloating, renderStageBar: () => { window.__sbr = (window.__sbr || 0) + 1; }, S: { objects: [{ name: 'man' }, { name: 'rig' }] } }; })();`);

// ---------------------------------------------------------------- fake viewer
const plates = [];
const viewer = {
  projectLoaded: true, currentWorkDir: 'D:/W/shot/matte_desk', manifest: { plates: { beauty: 'b.####.exr' } },
  firstFrame: 1, lastFrame: 100, currentFrame: 30, activePlate: 'beauty',
  setActivePlate(n) { this.activePlate = n; plates.push(n); },
  goToFrame(f) { this.currentFrame = f; window.dispatchEvent(new window.CustomEvent('md-frame', { detail: { frame: f } })); },
  render() {},
};
window.viewer = viewer;

// ---------------------------------------------------------------- load the scripts (index.html order)
window.eval(readFileSync(path.join(STATIC, 'cleanup.js'), 'utf-8'));
window.eval(readFileSync(path.join(STATIC, 'cleanup3d.js'), 'utf-8'));
await wait();
const C3 = window.MD_CLEANUP3D;
assert(C3 && C3.ready, 'MD_CLEANUP3D ready');
assert(window.MD_CLEANUP.mode === 'cut', 'starts in cutting mode');
// cutting mode: nothing of Cleanup 3D shows, the object stage bar is as it was
assert(shown('stageBar') && shown('stageList') && $('stageList').innerHTML === ORIG_STAGES, 'object stage bar intact');
assert(!shown('c3StepList') && !shown('c3ViewTabs') && !shown('c3PalCamera') && !shown('c3PaneCamera'), 'c3 hidden in cut mode');
assert(shown('objPane') && !shown('cleanupPanel'), 'object pane shown in cut mode');

// ---------------------------------------------------------------- clean mode: the step bar
document.querySelector('#modeSwitch [data-mode="clean"]').click();
await wait();
assert(window.MD_CLEANUP.mode === 'clean', 'clean mode');
assert(shown('stageBar'), 'stage bar visible in Cleanup mode');
assert(shown('c3StepList') && shown('c3StepHint'), 'c3 step list shown');
assert(!shown('stageList') && !shown('stageHint') && !shown('stageObjectBox'), 'object stage bar hidden');
const items = [...document.querySelectorAll('#c3StepList .stage-item')];
assert(items.length === 6, `6 steps (${items.length})`);
assert(items.map((li) => li.querySelector('.stage-n').textContent).join(',') === 'C1,C2,C3,C4,C5,C6', 'step numbers');
assert(items.map((li) => li.querySelector('.stage-label').textContent).join('|') === 'Inpaint (beta)|Kamera 3D|Karty|Klucze|Projekcja|Eksport', 'step labels');
const li = (id) => document.querySelector(`#c3StepList [data-c3step="${id}"]`);
assert(li('camera').classList.contains('done') && li('camera').querySelector('.stage-ic').textContent === '✔', 'camera ✔ done');
assert(li('cards').classList.contains('stale') && li('cards').querySelector('.stage-ic').textContent === '↻', 'cards ↻ stale');
assert(li('keys').classList.contains('todo') && li('keys').querySelector('.stage-ic').textContent === '○', 'keys ○ todo');
assert(li('remove').classList.contains('current'), 'C1 current by default');
// C1 = the MVP panel, unchanged ids
assert(shown('cleanupPanel') && !shown('objPane'), 'C1 shows the cleanup panel, object pane hidden');
for (const id of ['cuObjects', 'cuGrow', 'cuSoft', 'cuHoldout', 'cuFill', 'cuPreview', 'cuRun', 'cuVersions']) assert($(id), `C1 id ${id}`);
assert($('cuGrow').value === '8', 'C1 panel loaded its settings');
assert(!shown('c3PalCamera') && !shown('c3PalCards'), 'no c3 palette in C1');
assert(shown('c3ViewTabs') && !shown('viewModeTabs'), 'preview switch replaces the object view tabs');
assert(document.body.classList.contains('c3-on'), 'body c3-on');

// ---------------------------------------------------------------- C2 camera
li('camera').click();
await wait();
assert(C3.step === 'camera' && li('camera').classList.contains('current'), 'C2 current');
assert(!shown('cleanupPanel') && shown('c3PaneCamera') && shown('c3PalCamera'), 'C2 pane + palette');
assert(window.localStorage.getItem('md_c3_step:D:/W/shot/matte_desk') === 'camera', 'step remembered per project');
const rep = $('c3CamReport').textContent;
assert(rep.includes('0.43') && rep.includes('1.50') && rep.includes('F77') && rep.includes('F12'), `solve report: ${rep}`);
assert($('c3PaneCamera').textContent.includes('Co robi:') && $('c3PaneCamera').textContent.includes('Najlepsze do:'), 'Co robi / Najlepsze do');
document.querySelector('#c3CamReport [data-c3jump="77"]').click();
assert(viewer.currentFrame === 77, 'worst frame jump');
assert(document.querySelectorAll('#c3Moving input[data-c3mv]').length === 2, 'moving objects list');
// palette drag (the real makeFloating) + double click back to the corner
const pal = $('c3PalCamera'); const cw = $('canvasWrapper');
Object.defineProperty(cw, 'clientWidth', { value: 800 }); Object.defineProperty(cw, 'clientHeight', { value: 600 });
Object.defineProperty(pal, 'offsetWidth', { value: 200 }); Object.defineProperty(pal, 'offsetHeight', { value: 300 });
const phead = pal.querySelector('.pp-head');
phead.dispatchEvent(new window.MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 10, clientY: 10 }));
document.dispatchEvent(new window.MouseEvent('mousemove', { bubbles: true, clientX: 110, clientY: 60 }));
document.dispatchEvent(new window.MouseEvent('mouseup', { bubbles: true, clientX: 110, clientY: 60 }));
assert(pal.style.left === '100px' && pal.style.top === '50px', `palette dragged (${pal.style.left}, ${pal.style.top})`);
assert(JSON.parse(window.localStorage.getItem('md_c3_palette_camera')).x === 100, 'palette position stored');
document.dispatchEvent(new window.MouseEvent('mousemove', { bubbles: true, clientX: 700, clientY: 700 }));
assert(pal.style.left === '100px', 'drag ended on mouseup');
phead.dispatchEvent(new window.MouseEvent('dblclick', { bubbles: true }));
assert(pal.style.left === '' && pal.style.top === '' && window.localStorage.getItem('md_c3_palette_camera') === null, 'double click = back to the corner');
// 501 -> toast with the server's text
$('c3SingleFocal').checked = false;
document.querySelector('#c3Moving input[data-c3mv="man"]').checked = true;
$('c3Solve').click();
await wait();
const solve = calls.find((c) => c[1] === '/api/cleanup3d/camera/solve');
assert(solve && solve[2].single_focal === false && solve[2].horizon === true && solve[2].moving.join() === 'man', `solve body ${JSON.stringify(solve)}`);
assert(lastToast() === DETAIL('P10 solve kamery'), `501 toast: ${lastToast()}`);
assert(!$('c3Solve').disabled, 'button enabled again');

// ---------------------------------------------------------------- C3 cards: order saved with one PUT
li('cards').click();
await wait();
assert(shown('c3PalCards') && !shown('c3PalCamera') && shown('c3PaneCards'), 'C3 palette only');
assert(shown('c3Tools') && $('c3ToolHint').textContent.includes('Zasięg karty'), 'C3 tool slot');
let rows = [...document.querySelectorAll('#c3CardList [data-c3card]')];
assert(rows.map((r) => r.dataset.c3card).join() === 'sky,sea', 'card list order');
let nPut = calls.filter((c) => c[0] === 'PUT').length;
document.querySelector('#c3CardList [data-c3down="sky"]').click();
await wait();
let puts = calls.filter((c) => c[0] === 'PUT');
assert(puts.length === nPut + 1, `one PUT for the order (${puts.length - nPut})`);
const put = puts[puts.length - 1];
assert(put[1] === '/api/cleanup3d/cards' && put[2].cards.map((c) => c.id).join() === 'sea,sky' && put[2].cards.map((c) => c.order).join() === '0,1', `PUT body ${JSON.stringify(put[2].cards.map((c) => [c.id, c.order]))}`);
assert(!put[2].cards.some((c) => 'keys' in c), 'keys not sent with the cards');
rows = [...document.querySelectorAll('#c3CardList [data-c3card]')];
assert(rows.map((r) => r.dataset.c3card).join() === 'sea,sky', 'list redrawn in the new order');
// drag and drop: sky back on top, again one PUT
nPut = calls.filter((c) => c[0] === 'PUT').length;
const ds = new window.Event('dragstart', { bubbles: true }); document.querySelector('#c3CardList [data-c3card="sky"]').dispatchEvent(ds);
const dr = new window.Event('drop', { bubbles: true, cancelable: true }); document.querySelector('#c3CardList [data-c3card="sea"]').dispatchEvent(dr);
await wait();
puts = calls.filter((c) => c[0] === 'PUT');
assert(puts.length === nPut + 1 && puts[puts.length - 1][2].cards.map((c) => c.id).join() === 'sky,sea', 'drag & drop order = one PUT');
// slider edit -> debounced PUT with adjust
document.querySelector('#c3CardList [data-c3card="sea"] .c3-kind').closest('[data-c3card]').click();
assert(C3.card === 'sea', 'card selected by click');
const pitch = document.querySelector('#c3CardEdit [data-c3adj="pitch"]');
pitch.value = '5.5'; pitch.dispatchEvent(new window.Event('input', { bubbles: true }));
await wait(400);
const pp = calls.filter((c) => c[0] === 'PUT').pop();
assert(pp[2].cards.find((c) => c.id === 'sea').adjust.pitch === 5.5, 'pitch saved');
// ＋ karta
$('c3CardAdd').click();
await wait();
assert(srv.cards.length === 3 && srv.cards[2].kind === 'plane' && srv.cards[2].plane, 'new card saved');
assert(C3.card === srv.cards[2].id, 'new card selected');
// P18 depth card: kind „głębia” -> POST .../depth with the current frame; the palette shows the depth sliders
const newId = srv.cards[2].id;
const f0 = viewer.currentFrame;
const kindSel = document.querySelector(`#c3CardList [data-c3kind="${newId}"]`);
assert([...kindSel.options].map((o) => o.value).join() === 'plane,sky,depth', 'kind select has głębia');
kindSel.value = 'depth'; kindSel.dispatchEvent(new window.Event('change', { bubbles: true }));
await wait(40);
const dpost = calls.filter((c) => c[0] === 'POST' && c[1] === `/api/cleanup3d/cards/${newId}/depth`).pop();
assert(dpost && dpost[2].frame === f0, `depth POST with the current frame ${JSON.stringify(dpost)}`);
assert(srv.cards[2].kind === 'depth' && C3.card === newId, 'card is a depth card');
assert(document.querySelector('#c3CardEdit [data-c3adj="mul"]') && document.querySelector('#c3CardEdit [data-c3adj="shift"]'), 'depth sliders shown');
assert(!document.querySelector('#c3CardEdit [data-c3adj="pitch"]'), 'plane sliders hidden for a depth card');
assert($('c3CardEdit').textContent.includes(`F${f0}`) && $('c3CardEdit').textContent.includes('punkty kamery 3D'), `depth panel: ${$('c3CardEdit').textContent}`);
assert([...document.querySelectorAll('#c3CardEdit [data-c3dsrc] option')].map((o) => o.value).join() === 'computed,plate', 'depth sources listed');
const mul = document.querySelector('#c3CardEdit [data-c3adj="mul"]');
mul.value = '1.5'; mul.dispatchEvent(new window.Event('input', { bubbles: true }));
await wait(400);
const mp = calls.filter((c) => c[0] === 'PUT').pop();
assert(mp[2].cards.find((c) => c.id === newId).depth.mul === 1.5, 'depth scale slider saved on the card');
viewer.currentFrame = 42;
document.querySelector('#c3CardEdit [data-c3depth="frame"]').click();
await wait(40);
const fp = calls.filter((c) => c[0] === 'POST' && c[1] === `/api/cleanup3d/cards/${newId}/depth`).pop();
assert(fp[2].frame === 42 && fp[2].source === 'computed', `reference frame from the current frame ${JSON.stringify(fp[2])}`);
viewer.currentFrame = f0;

// ---------------------------------------------------------------- C4 keys: engine gate, key list, 501 toast
li('keys').click();
await wait();
assert(shown('c3PalKeys') && shown('c3PaneKeys') && $('c3ToolHint').textContent.includes('Obszar do dogenerowania'), 'C4 palette + tools');
assert($('c3ExtentMount') && C3.extentMount() === $('c3ExtentMount'), 'extent mount in the Karty palette');
C3.selectCard('sea');
const eng = $('c3Engine');
const kl = [...eng.options].find((o) => o.value === 'klein_restore');
assert(kl && kl.disabled && kl.textContent.includes(REASON) && kl.title === REASON, 'engine without model disabled with the reason');
assert(eng.value === 'lama', 'available engine picked');
assert($('c3EngWhy').textContent.includes(REASON), 'reason shown under the engine');
assert($('c3Regen').disabled && $('c3KeyInfo').textContent.includes('klucz'), 'regen disabled without a key');
const kbtn = [...document.querySelectorAll('#c3KeyList [data-c3key]')];
assert(kbtn.map((b) => b.dataset.c3key).join() === '10,60', 'key list of the card');
kbtn[1].click();
await wait();
assert(viewer.currentFrame === 60 && C3.key && C3.key.card === 'sea' && C3.key.frame === 60, 'key click = jump + select');
assert(!$('c3Regen').disabled, 'regen enabled with a key and an available engine');
assert($('c3KeyVersions').textContent.includes('F60'), 'key versions in the right pane');
assert(C3.areaMount() === $('c3AreaMount') && C3.toolMount() === $('c3ToolMount'), 'P7 mount points');
$('c3Regen').click();
await wait();
const rg = calls.find((c) => c[1] === '/api/cleanup3d/keys/sea/60/regen');
assert(rg && rg[2].engine === 'lama' && rg[2].soften_seams === false, 'regen body');
assert(lastToast() === DETAIL('P5 dogenerowanie + P8'), `regen 501 toast: ${lastToast()}`);
assert(rg[2].base === 'v001' && rg[2].area === 'marked' && rg[2].strength === null, `regen body: base / area / strength ${JSON.stringify(rg[2])}`);

// ---------------------------------------------------------------- key steps (2026-11-10): versions, base, area, strength
Object.assign(seaKey60(), { list: VLIST.map((v) => ({ ...v })), versions: 3, active: 'v003' });
srv.kleinOk = true;
await C3.refresh();
await wait();
let vrows = [...document.querySelectorAll('#c3KeyVersions [data-c3ver]')];
assert(vrows.map((r) => r.dataset.c3ver).join() === 'v003,v002,v001', `version list newest first (${vrows.map((r) => r.dataset.c3ver)})`);
assert(vrows[0].classList.contains('active') && vrows[0].textContent.includes('regen:klein_restore') && vrows[0].textContent.includes('z v002')
  && vrows[0].textContent.includes('wet sand') && vrows[0].textContent.includes('2026-11-10 10:09'), `active row: ${vrows[0].textContent}`);
assert(vrows[1].textContent.includes('plate+fill:v001') && vrows[2].textContent.includes('plate'), 'sources shown');
assert(vrows[0].querySelector('[data-c3verdel]').disabled && !vrows[1].querySelector('[data-c3verdel]').disabled, 'the active version cannot be deleted');
assert([...$('c3GenBase').options].map((o) => o.value).join() === 'v003,v002,v001' && $('c3GenBase').value === 'v003', 'Z wersji: default = active');
// 🗑 = the permanent-delete warning (UI.confirmDelete, user decision 2026-11-13: no trash), then DELETE
const vdialogs = [];
window.UI.confirmDelete = async (o) => { vdialogs.push(o); return false; };
vrows[1].querySelector('[data-c3verdel]').click();
await wait();
assert(vdialogs.length === 1 && vdialogs[0].what.includes('v002') && !calls.some((c) => c[0] === 'DELETE'), 'cancelled dialog: nothing deleted');
window.UI.confirmDelete = async (o) => { vdialogs.push(o); return true; };
vrows[1].querySelector('[data-c3verdel]').click();
await wait();
const vdel = calls.find((c) => c[0] === 'DELETE' && c[1] === '/api/cleanup3d/keys/sea/60/versions/v002');
assert(vdel && vdialogs.length === 2 && lastToast().includes('v002') && lastToast().includes('2,0 MB'), `delete version: ${lastToast()}`);
window.UI.confirmDelete = async () => true;
vrows = [...document.querySelectorAll('#c3KeyVersions [data-c3ver]')];
assert(vrows.map((r) => r.dataset.c3ver).join() === 'v003,v001', 'deleted version gone from the list');
// click = make active
vrows[1].click();
await wait();
const vact = calls.find((c) => c[0] === 'PUT' && c[1] === '/api/cleanup3d/keys/sea/60/active');
assert(vact && vact[2].version === 'v001' && document.querySelector('#c3KeyVersions [data-c3ver="v001"]').classList.contains('active'), 'click = active');
assert($('c3GenBase').value === 'v001', 'base follows the active version while none was chosen');
// a chosen base stays; FLUX 3 Image Klein strength: 1.0 marked, 0.9 cała łata; soften only for the marked area
$('c3GenBase').value = 'v003'; $('c3GenBase').dispatchEvent(new window.Event('change', { bubbles: true }));
$('c3Engine').value = 'klein_restore'; $('c3Engine').dispatchEvent(new window.Event('change', { bubbles: true }));
await wait();
assert(shown('c3StrengthRow') && $('c3Strength').value === '1' && $('c3StrengthV').textContent === '1.00', `strength marked ${$('c3Strength').value}`);
assert(!shown('c3GenBar'), 'no selection, area zaznaczenie: no bar');
$('c3GenPatch').click();
await wait();
assert($('c3GenArea').value === 'patch' && shown('c3GenBar'), 'cała łata: the bar without a selection');
assert($('c3Strength').value === '0.9' && !shown('c3SoftenRow'), `strength patch ${$('c3Strength').value}`);
$('c3Strength').value = '0.75'; $('c3Strength').dispatchEvent(new window.Event('input', { bubbles: true }));
$('c3Soften').checked = true;
$('c3Regen').click();
await wait();
const rg2 = calls.filter((c) => c[1] === '/api/cleanup3d/keys/sea/60/regen').pop();
assert(rg2[2].engine === 'klein_restore' && rg2[2].area === 'patch' && rg2[2].base === 'v003' && rg2[2].strength === 0.75 && rg2[2].soften_seams === false,
  `regen body cała łata ${JSON.stringify(rg2[2])}`);
assert(lastToast().startsWith('Cała łata dogenerowana') && lastToast().includes('v004'), `regen toast ${lastToast()}`);
assert($('c3GenBase').value === 'v003', 'the chosen base stays after the step');
$('c3GenArea').value = 'marked'; $('c3GenArea').dispatchEvent(new window.Event('change', { bubbles: true }));
await wait();
assert($('c3Strength').value === '1' && !shown('c3GenBar'), 'back to zaznaczenie: its own strength, bar hidden without a selection');
window.UI.dialog = async () => true;
srv.kleinOk = false;
delete seaKey60().list; Object.assign(seaKey60(), { versions: 1, active: 'v001' });
await C3.refresh();
await wait();
const modes = [...$('c3KeyMode').options];
assert(modes.length === 4 && modes.every((o) => !o.disabled), 'all four key modes enabled (P13)');
assert(!shown('c3AutoBox') && !shown('c3OneBox'), 'manual mode: no auto / master controls');
// P13 automat: the mode and its settings saved on the card, the button sends them
const keyMode = (m) => { $('c3KeyMode').value = m; $('c3KeyMode').dispatchEvent(new window.Event('change', { bubbles: true })); };
keyMode('auto');
await wait();
assert(shown('c3AutoBox') && !shown('c3OneBox') && $('c3AutoThr').value === '3' && $('c3AutoMin').value === '4', 'auto controls with the defaults');
let mput = calls.filter((c) => c[0] === 'PUT').pop();
assert(mput[2].cards.find((c) => c.id === 'sea').key_mode === 'auto', 'auto mode saved');
$('c3AutoThr').value = '5'; $('c3AutoThr').dispatchEvent(new window.Event('change', { bubbles: true }));
await wait();
mput = calls.filter((c) => c[0] === 'PUT').pop();
const ka = mput[2].cards.find((c) => c.id === 'sea').key_auto;
assert(Math.abs(ka.threshold - 0.05) < 1e-9 && ka.min_gap === 4 && ka.max_gap === 48, `auto settings saved ${JSON.stringify(ka)}`);
$('c3KeyAuto').click();
await wait();
const au = calls.find((c) => c[1] === '/api/cleanup3d/keys/sea/auto');
assert(au && Math.abs(au[2].threshold - 0.05) < 1e-9 && au[2].max_gap === 48, 'auto request body');
assert(lastToast().startsWith('Automat: 3 kluczy') && $('c3KeyModeInfo').textContent.includes('F1, F30, F60'), `auto result shown: ${lastToast()}`);
// P13 jeden klucz: the card's other keys -> the dialog, then replace
keyMode('one');
await wait();
assert(shown('c3OneBox') && !shown('c3AutoBox') && $('c3KeyModeInfo').textContent.includes('nie ma jeszcze'), 'master controls');
const dialogs = [];
window.UI.dialog = async (o) => { dialogs.push(o); return true; };
viewer.currentFrame = 30;
$('c3KeyMaster').click();
await wait();
const ms = calls.find((c) => c[1] === '/api/cleanup3d/keys/sea/master');
assert(dialogs.length === 1 && dialogs[0].html.includes('F10, F60') && dialogs[0].html.includes('na stałe') && dialogs[0].tone === 'danger' && ms && ms[2].frame === 30 && ms[2].replace === true, 'master: dialog, then replace');
assert(lastToast().startsWith('Klucz-suma F30') && C3.key && C3.key.frame === 30, `master result: ${lastToast()}`);
window.UI.dialog = async () => true;
keyMode('manual');
await wait();
C3.selectKey('sea', 60);
await wait();

// ---------------------------------------------------------------- timeline rows (P16): one row per card + the patch row
const lane = $('timelineC3');
const row = (id) => lane.querySelector(`[data-c3row="${id}"]`);
await C3.refresh(); await wait();
assert(lane && lane.style.display !== 'none', 'timeline rows shown');
const rowIds = [...lane.querySelectorAll('[data-c3row]')].map((r) => r.dataset.c3row);
assert(rowIds.join('|') === [...srv.cards.map((c) => c.id), 'patch'].join('|'), `rows: cards in the card order + Łata (${rowIds})`);
assert(row('sea').querySelector('.c3-tl-name').textContent === 'morze' && row('sea').querySelector('.c3-tl-mode').textContent.includes('ręcznie'), 'row head: name + key mode');
const dia = (id) => [...row(id).querySelectorAll('.c3-tl-key')];
assert(dia('sea').length === 2 && dia('sky').length === 1, `diamonds per row (${dia('sea').length}, ${dia('sky').length})`);
assert(/74,\s*224,\s*255|#4ae0ff/i.test(dia('sky')[0].style.background), `sky diamond colour ${dia('sky')[0].style.background}`);
assert(dia('sea').find((d) => d.dataset.frame === '60').classList.contains('sel'), 'selected key highlighted');
// coverage bar: frames 81–100 of the sea have keys short of the card -> warning frames in the tooltip
const seaCov = row('sea').querySelector('.c3-tl-cov');
assert(seaCov.dataset.warn === '20' && seaCov.title.includes('F81–F100'), `sea coverage warning (${seaCov.dataset.warn}: ${seaCov.title})`);
assert(row('sky').querySelector('.c3-tl-cov').dataset.warn === '0', 'sky fully covered');
assert(lane.querySelector('.c3-tl-warn') && lane.querySelector('.c3-tl-warn').textContent.includes('1 karta'), 'group warning');
// the patch row: band keys as triangles, click = jump
const bk = [...row('patch').querySelectorAll('.band-key')];
assert(bk.length === 2 && bk[0].dataset.frame === '20', 'patch band keys');
bk[1].click(); await wait();
assert(viewer.currentFrame === 40, 'band key click = jump');
// solo = like After Effects (user 2026-11-10): stored on the card (PUT …/solo, no dialog), the render / previews / Nuke
// follow on the server – the preview URL carries no card filter any more; the rows outside the solo are dimmed
fetches.length = 0;
const soloDialogs = [];
window.UI.dialog = async (o) => { soloDialogs.push(o); return true; };
assert(!lane.querySelector('[data-c3tl="all"]') && !lane.querySelector('.c3-tl-filter'), 'no solo: no solo banner');
assert(row('sea').querySelector('[data-c3tl="solo"]').title.includes('jak w After Effects'), 'solo tooltip');
row('sea').querySelector('[data-c3tl="solo"]').click(); await wait(40);
assert(!soloDialogs.length && calls.some((c) => c[0] === 'PUT' && c[1] === '/api/cleanup3d/cards/sea/solo' && c[2].solo === true), 'solo: PUT, no dialog');
assert(srv.cards.find((c) => c.id === 'sea').solo === true, 'solo stored on the card');
assert(row('sea').querySelector('[data-c3tl="solo"]').classList.contains('on') && !row('sea').classList.contains('hid'), 'soloed row lit');
assert(row('sky').classList.contains('hid'), 'row outside the solo dimmed');
assert(lane.querySelector('.c3-tl-filter').textContent.includes('Solo: morze'), `solo banner: ${lane.querySelector('.c3-tl-filter').textContent}`);
assert(lastToast().startsWith('Solo: morze'), `solo toast: ${lastToast()}`);
await C3.setView('cards'); await wait();
assert(fetches.pop() === '/api/cleanup3d/view/cards/40?max_side=1600', 'preview URL without a card filter');
assert(!fetches.some((u) => u.includes('cards=')), 'no cards= query anywhere');
// a second solo, then „wyłącz solo” takes it from both
row('sky').querySelector('[data-c3tl="solo"]').click(); await wait(40);
assert(!row('sky').classList.contains('hid') && srv.cards.filter((c) => c.solo).length === 2, 'two soloed cards');
lane.querySelector('[data-c3tl="all"]').click(); await wait(60);
const soloOff = calls.filter((c) => c[0] === 'PUT' && /\/solo$/.test(c[1]) && c[2].solo === false).map((c) => c[1]).sort();
assert(soloOff.join('|') === '/api/cleanup3d/cards/sea/solo|/api/cleanup3d/cards/sky/solo', `solo off for every soloed card (${soloOff})`);
assert(!srv.cards.some((c) => c.solo) && !row('sky').classList.contains('hid') && !lane.querySelector('[data-c3tl="all"]'), 'solo off: rows back');
await C3.setView('proj'); await wait();
assert(fetches.pop() === '/api/cleanup3d/view/proj/40?max_side=1600', 'proj preview');
assert(!calls.some((c) => c[0] === 'PUT' && /\/off$/.test(c[1])), 'solo never touches the eye');
window.UI.dialog = async () => true;
await C3.setView('plate');
// 👁 = the card on / off in the render and the previews (stored on the card, no dialog), the row dimmed
const offDialogs = [];
window.UI.dialog = async (o) => { offDialogs.push(o); return true; };
assert(!row('sea').querySelector('[data-c3tl="off"]'), 'no separate off button');
row('sea').querySelector('[data-c3tl="eye"]').click(); await wait(40);
assert(!offDialogs.length && calls.some((c) => c[0] === 'PUT' && c[1] === '/api/cleanup3d/cards/sea/off' && c[2].off === true), 'eye off: PUT, no dialog');
assert(row('sea').classList.contains('off') && !row('sea').querySelector('[data-c3tl="eye"]').classList.contains('on'), 'eye off: row dimmed');
row('sea').querySelector('[data-c3tl="eye"]').click(); await wait(40);
assert(calls.filter((c) => c[1] === '/api/cleanup3d/cards/sea/off').pop()[2].off === false && !row('sea').classList.contains('off'), 'eye back on');
// drag a diamond: F60 -> F71 with a confirm, POST …/move, the moved key selected; onto a taken frame = refused
const ptr = (el, type, x) => el.dispatchEvent(new window.MouseEvent(type, { bubbles: true, clientX: x, button: 0 }));
const seaLane = () => row('sea').querySelector('.c3-tl-lane');
const grab = (f) => { const l = seaLane(); l.getBoundingClientRect = () => ({ left: 0, width: 990, top: 0, height: 10 }); return [l, dia('sea').find((d) => d.dataset.frame === String(f))]; };
let [, d60] = grab(60);
ptr(d60, 'pointerdown', 590); ptr(d60, 'pointermove', 90); ptr(d60, 'pointerup', 90); await wait(40);
assert(!calls.some((c) => /\/move$/.test(c[1])) && lastToast().includes('ma już klucz w klatce 10'), `drag onto a taken frame refused: ${lastToast()}`);
[, d60] = grab(60);
const mvDialogs = [];
window.UI.dialog = async (o) => { mvDialogs.push(o); return true; };
ptr(d60, 'pointerdown', 590); ptr(d60, 'pointermove', 700); ptr(d60, 'pointerup', 700); await wait(60);
const mv = calls.find((c) => c[0] === 'POST' && c[1] === '/api/cleanup3d/keys/sea/60/move');
assert(mvDialogs.length === 1 && mvDialogs[0].html.includes('F60') && mvDialogs[0].html.includes('F71') && mvDialogs[0].html.includes('na stałe') && !mvDialogs[0].html.includes('kosza'), 'move: confirm dialog');
assert(mv && mv[2].to === 71, `move request ${JSON.stringify(mv)}`);
assert(C3.key && C3.key.card === 'sea' && C3.key.frame === 71 && viewer.currentFrame === 71, 'moved key selected + jump');
assert(dia('sea').map((d) => d.dataset.frame).join(',') === '10,71', 'diamonds after the move');
window.UI.dialog = async () => true;
// fold: one lane with every key
lane.querySelector('[data-c3tl="fold"]').click(); await wait();
assert(!row('sea') && lane.querySelectorAll('.c3-tl-key').length === 3, 'folded = one lane');
lane.querySelector('[data-c3tl="fold"]').click(); await wait();
assert(row('sea') && row('patch'), 'unfolded again');
// click = jump + select
dia('sky')[0].click();
await wait();
assert(viewer.currentFrame === 10 && C3.key.card === 'sky' && C3.key.frame === 10, 'diamond click = jump + select');

// ---------------------------------------------------------------- preview switch
const tab = (v) => document.querySelector(`#c3ViewTabs [data-c3view="${v}"]`);
assert([...document.querySelectorAll('#c3ViewTabs [data-c3view]')].map((b) => b.textContent).join('|') === 'Plate|Klucz|Rzut|Wynik|Karty', 'preview tabs');
tab('key').click();
await wait();
assert(fetches.pop() === '/api/cleanup3d/view/key/10?max_side=1600&card=sky', 'key preview url');
assert(lastToast() === DETAIL('P6 projekcja i składanie + P8'), 'preview 501 toast');
assert(C3.view === 'plate', 'failed preview leaves the plate');
tab('cards').click(); await wait();
assert(fetches.pop() === '/api/cleanup3d/view/cards/10?max_side=1600', 'cards preview url');
tab('cleaned').click(); await wait();
assert(viewer.activePlate === 'cleaned' && tab('cleaned').classList.contains('active'), 'Wynik = plate cleaned');
tab('plate').click(); await wait();
assert(viewer.activePlate === 'beauty' && tab('plate').classList.contains('active'), 'Plate = beauty');

// ---------------------------------------------------------------- C5 / C6
li('project').click(); await wait();
assert(shown('c3PalProject') && shown('c3PaneProject'), 'C5');
$('c3Blur').value = 'off'; $('c3Feather').value = '12';
$('c3Render').click(); await wait();
const rd = calls.find((c) => c[1] === '/api/cleanup3d/render');
assert(rd && rd[2].motion_blur === 'off' && rd[2].feather_px === 12 && rd[2].frames === null, 'render body');
assert(lastToast() === DETAIL('P6 projekcja i składanie + P8'), 'render 501 toast');
li('export').click(); await wait();
assert(shown('c3PalExport') && shown('c3PaneExport') && $('c3NukeExport').disabled, 'C6 placeholder');
li('keys').click(); await wait();
// objects.js writes the display of its panes on its own events (setStage) – Cleanup keeps them out of sight (defect (a)
// of the wave-1 browser check; the reload case with the stage change: cleanup3d_reload_harness.mjs)
$('objPaneHidden').style.display = 'none'; $('objPane').style.display = '';
await wait();
assert(!shown('objPane') && !shown('objPaneHidden') && shown('c3PaneKeys'), 'a stage pane shown again by objects.js stays hidden in Cleanup');

// ---------------------------------------------------------------- back to cutting: everything as before
const sbr0 = window.__sbr || 0;
document.querySelector('#modeSwitch [data-mode="cut"]').click();
await wait();
assert(window.MD_CLEANUP.mode === 'cut', 'cut mode');
assert(shown('stageBar') && shown('stageList') && shown('stageHint') && shown('stageObjectBox'), 'object stage bar back');
assert($('stageList').innerHTML === ORIG_STAGES && $('stageHint').textContent === 'hint etapu', 'object stage bar content untouched');
assert((window.__sbr || 0) === sbr0 + 1, 'object stage bar re-rendered once');
assert(!shown('c3StepList') && !shown('c3StepHint') && !shown('c3ViewTabs') && shown('viewModeTabs'), 'c3 bar / tabs hidden, view tabs back');
assert(shown('objPane') && !shown('objPaneHidden') && !shown('cleanupPanel'), 'object panes back as they were');
for (const id of ['c3PalCamera', 'c3PalCards', 'c3PalKeys', 'c3PalProject', 'c3PalExport', 'c3PaneCamera', 'c3PaneCards', 'c3PaneKeys', 'c3PaneProject', 'c3PaneExport', 'c3Tools']) assert(!shown(id), `${id} hidden in cut mode`);
assert(lane.style.display === 'none', 'key lane hidden in cut mode');
assert(!document.body.classList.contains('c3-on'), 'c3-on removed');

// ---------------------------------------------------------------- again Cleanup: the step comes back (per project)
document.querySelector('#modeSwitch [data-mode="clean"]').click();
await wait();
assert(C3.step === 'keys' && li('keys').classList.contains('current') && shown('c3PalKeys') && !shown('cleanupPanel'), 'step remembered');
assert(!shown('objPane'), 'object pane hidden again (twice in clean mode)');
document.querySelector('#modeSwitch [data-mode="cut"]').click();
await wait();
assert(shown('objPane') && !shown('objPaneHidden'), 'object panes back after a second round');
// onChange hook
let seen = null; C3.onChange((e) => { seen = e; }); C3.setStep('cards');
assert(seen && seen.step === 'cards', 'onChange fires');

console.log(`OK cleanup3d harness (${calls.length} api calls, ${toasts.length} toasts)`);
