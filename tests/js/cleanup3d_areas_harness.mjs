// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/cleanup3d_areas_harness.mjs – headless run of the Cleanup 3D area editor (static/cleanup3d_areas.js, package
// P7) in jsdom with a fake API / viewer; the REAL static/curves.js (MD_CURVES.Editor) and static/brushcore.js are loaded.
// Checks: tools on the left bar; a brush stroke and a closed curve -> the PUT …/areas payload (plate pixels, brush layer
// as PNG data only when its pixels changed); layers (add / remove op, visibility, delete); Ctrl+Z; rectangle; eraser;
// a point dragged through the curve editor; a saved brush layer kept by file name; 501 keeps the drawing and shows the
// server's text; a mark on a frame that is not a key -> POST …/mark/{frame}; a card's extent saved with its frame and
// in the card's colour, editable only in that frame; nothing left on the viewer in the cutting mode.
// Run: cd tests/js && npm ci && node cleanup3d_areas_harness.mjs   (tests/test_cleanup3d_js.py wraps it)
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
    <div class="view-mode-tabs" id="viewModeTabs"></div></div></header>
  <nav class="stage-bar" id="stageBar"><div id="stageObjectBox"></div><ol id="stageList"></ol><div id="stageHint"></div></nav>
  <aside class="tools-sidebar" id="toolsSidebar"><div id="drawToolsTitle">Rysuj</div><div id="drawTools"></div></aside>
  <div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div>
  <aside class="stage-panel" id="stagePanel"><div class="stage-pane" id="objPane">etap obiektu</div></aside>
  <div id="timelineTrackContainer"><div id="timelineKeys"></div></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;

// canvas stubs (jsdom has no 2D context): every drawing call is accepted, sizes come from the element
const fakeCtx = (canvas) => new Proxy({ canvas, globalAlpha: 1 }, {
  get(t, k) {
    if (k in t) return t[k];
    if (k === 'getImageData') return (x, y, w, h) => ({ data: new Uint8ClampedArray(Math.max(1, w * h) * 4), width: w, height: h });
    return () => {};
  },
  set(t, k, v) { t[k] = v; return true; },
});
window.HTMLCanvasElement.prototype.getContext = function () { return this.__ctx || (this.__ctx = fakeCtx(this)); };
window.HTMLCanvasElement.prototype.toDataURL = function () { return `data:image/png;base64,W${this.width}H${this.height}`; };
window.HTMLCanvasElement.prototype.getBoundingClientRect = function () { return { left: 0, top: 0, width: this.width, height: this.height, right: this.width, bottom: this.height }; };

const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms = 20) => new Promise((r) => setTimeout(r, ms));
const SAVE = 750;                                    // > the editor's save debounce (600 ms)
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- fake server
const DETAIL = (p) => `Czyszczenie 3D: ta funkcja jeszcze nie jest zbudowana (${p})`;
const card = (id, name, kind, color, order, keys) => ({ id, name, kind, plane: kind === 'plane' ? { n: [0, -1, 0], d: 1 } : null,
  adjust: { dh: 0, pitch: 0, roll: 0 }, extent: [], extent_frame: null, seam_px: 16, order, color, key_mode: 'manual', key_every: 8, warps: {}, keys });
const SRV_CURVE = { type: 'bezier', closed: true, pts: [{ p: [10, 10], in: [0, 0], out: [0, 0], cusp: true }, { p: [90, 10], in: [0, 0], out: [0, 0], cusp: true }, { p: [50, 80], in: [0, 0], out: [0, 0], cusp: true }] };
const srv = {
  mode: 'ok',                                        // 'ok' | '501' for the writes of the area routes
  cards: [card('sky', 'niebo', 'sky', '#4ae0ff', 0, [{ frame: 10, active: 'v001', versions: 1 }]),
    card('sea', 'morze', 'plane', '#ffb84a', 1, [{ frame: 10, active: 'v002', versions: 2 }, { frame: 60, active: 'v001', versions: 1 }])],
  areas: { 'sea/10': [{ id: 'srv1', kind: 'brush', op: 'add', png: 'a_srv1.png', label: '' }, { id: 'srv2', kind: 'curve', op: 'add', curve: SRV_CURVE, label: '' }] },
};
const state = () => ({ steps: ['remove', 'camera', 'cards', 'keys', 'project', 'export'], done: {}, camera: null,
  cards: srv.cards.map((c) => JSON.parse(JSON.stringify(c))), engines: [{ id: 'lama', label: 'LaMa' }],
  key_modes: ['manual', 'every'], views: ['key', 'proj', 'cards'], key_formats: ['png'] });
const calls = [];
const err = (status, detail) => { const e = new Error(detail); e.status = status; e.detail = detail; return e; };
async function fakeApi(method, url, body) {
  calls.push([method, url, body ? JSON.parse(JSON.stringify(body)) : body]);
  let m;
  if (method === 'GET' && url === '/api/cleanup') return { objects: [], settings: { remove: [], sources: {}, grow_px: 8, soft_px: 4, holdout: true, fill: 'auto' }, sources: {}, versions: [], out_dir: 'W/inpaint' };
  if (method === 'GET' && url === '/api/cleanup3d') return state();
  if (method === 'PUT' && url === '/api/cleanup3d/cards') {
    const keys = Object.fromEntries(srv.cards.map((c) => [c.id, c.keys]));
    srv.cards = body.cards.map((c, i) => ({ ...c, order: i, keys: keys[c.id] || [] }));
    return { cards: body.cards.map((c, i) => ({ ...c, order: i })) };
  }
  if ((m = url.match(/^\/api\/cleanup3d\/keys\/(\w+)\/(\d+)\/areas$/))) {
    if (method === 'GET') return { areas: JSON.parse(JSON.stringify(srv.areas[`${m[1]}/${m[2]}`] || [])) };
    if (method === 'PUT') {
      if (srv.mode === '501') throw err(501, DETAIL('P2 magazyn kluczy + P8'));
      srv.areas[`${m[1]}/${m[2]}`] = body.areas.map((a) => { const o = { ...a }; delete o.png_data; delete o.width; delete o.height; return o; });
      return { areas: srv.areas[`${m[1]}/${m[2]}`] };
    }
  }
  if (method === 'PUT' && (m = url.match(/^\/api\/cleanup3d\/cards\/(\w+)\/extent$/))) {
    if (srv.mode === '501') throw err(501, DETAIL('P7 edytor zasięgu + P8'));
    return { success: true };
  }
  if (method === 'POST' && (m = url.match(/^\/api\/cleanup3d\/mark\/(\d+)$/))) {
    if (srv.mode === '501') throw err(501, DETAIL('P2 magazyn kluczy + P8'));
    return { keys: [{ card: 'sea', frame: 10 }, { card: 'sea', frame: 60 }] };
  }
  throw err(404, `fake api: ${method} ${url}`);
}
const toasts = [];
window.UI = { api: fakeApi, toast: (t) => toasts.push(String(t)), dialog: async () => true, confirmDelete: async () => true, bytes: (n) => `${n} B` };
const lastToast = () => toasts[toasts.length - 1] || '';
const last = (method, re) => calls.filter((c) => c[0] === method && re.test(c[1])).pop();
const count = (method, re) => calls.filter((c) => c[0] === method && re.test(c[1])).length;

// ---------------------------------------------------------------- fake viewer: proxy 640 × 360 of a 1280 × 720 plate
window.MD = { S: { objects: [] }, renderStageBar: () => {} };
const viewer = {
  projectLoaded: true, currentWorkDir: 'D:/W/shot/matte_desk', manifest: { plates: { beauty: 'b.####.exr' } },
  projectData: { plate_w: 1280, plate_h: 720 }, canvas: $('cv'), zoom: 1, panX: 0, panY: 0, overlayHooks: [],
  firstFrame: 1, lastFrame: 100, currentFrame: 60, activePlate: 'beauty',
  getCurrentImage() { return { width: 640, height: 360 }; },
  setActivePlate(n) { this.activePlate = n; },
  goToFrame(f) { this.currentFrame = f; window.dispatchEvent(new window.CustomEvent('md-frame', { detail: { frame: f } })); },
  render() { for (const fn of this.overlayHooks) fn(null, this.getCurrentImage()); },
};
window.viewer = viewer;
const panned = [];
viewer.canvas.addEventListener('mousedown', (e) => panned.push(['down', e.button]));
viewer.canvas.addEventListener('wheel', (e) => panned.push(['wheel', e.deltaY]));

// ---------------------------------------------------------------- scripts in index.html order
const indexHtml = readFileSync(path.join(STATIC, 'index.html'), 'utf-8');
const order = ['brushcore.js', 'brush.js', 'cleanup.js', 'cleanup3d.js', 'curves.js', 'cleanup3d_areas.js'].map((n) => indexHtml.indexOf(`/static/${n}"`));
assert(order.every((i) => i > 0) && order.join() === order.slice().sort((a, b) => a - b).join(), `script order in index.html: ${order}`);
for (const n of ['brushcore.js', 'cleanup.js', 'cleanup3d.js', 'curves.js', 'cleanup3d_areas.js']) window.eval(readFileSync(path.join(STATIC, n), 'utf-8'));
await wait();
const C3 = window.MD_CLEANUP3D; const AR = window.MD_C3AREAS; const A = AR.state;
assert(C3 && C3.ready && AR && AR.ready, 'Cleanup 3D + the area editor are ready');
const cvs = $('c3AreaCanvas');
assert(cvs && cvs.style.display === 'none', 'the editor canvas exists and is hidden in the cutting mode');

const ev = (type, x, y, o = {}) => new window.MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0, ...o });
const key = (k, o = {}) => window.dispatchEvent(new window.KeyboardEvent('keydown', { key: k, code: o.code || k, bubbles: true, cancelable: true, ...o }));
const stroke = (pts) => { cvs.dispatchEvent(ev('pointerdown', ...pts[0])); for (const p of pts.slice(1)) cvs.dispatchEvent(ev('pointermove', ...p)); cvs.dispatchEvent(ev('pointerup', ...pts[pts.length - 1])); };
const tool = (t) => document.querySelector(`#c3ToolMount [data-c3tool="${t}"]`);
const mount = () => $('c3AreaMount');
const rows = (m = mount()) => [...m.querySelectorAll('[data-ar-layer]')];
const q = (name, m = mount()) => m.querySelector(`[data-ar="${name}"]`);

// ---------------------------------------------------------------- C4 on a key frame: tools + an empty layer list
document.querySelector('#modeSwitch [data-mode="clean"]').click();
await wait();
document.querySelector('#c3StepList [data-c3step="keys"]').click();
C3.selectCard('sea');
await wait();
assert(A.cur && A.cur.type === 'key' && A.cur.key === 'key:sea:60', `target = the key of the current frame (${A.cur && A.cur.key})`);
assert(last('GET', /keys\/sea\/60\/areas$/), 'the key areas are loaded');
assert(['brush', 'curve', 'erase', 'rect'].every((t) => tool(t)) && !tool('brush').disabled, 'four tools on the left bar');
assert(tool('brush').textContent.includes('Obszar') && tool('curve').textContent.includes('Krzywa') && tool('erase').textContent.includes('Gumka') && tool('rect').textContent.includes('Prostokąt'), 'tool labels');
assert(q('info').textContent.includes('F60') && q('info').textContent.includes('morze'), `key info: ${q('info').textContent}`);
assert(q('list').textContent.includes('Brak warstw') && q('undo').disabled, 'no layers yet');
assert(cvs.style.display !== 'none' && !cvs.classList.contains('c3-area-on'), 'canvas shown, no tool = the viewer keeps the mouse');

// ---------------------------------------------------------------- brush stroke -> a brush layer -> PUT with PNG data
tool('brush').click();
assert(tool('brush').classList.contains('active') && cvs.classList.contains('c3-area-on'), 'brush tool on');
A.input.setRadius(20);
stroke([[100, 100], [120, 110], [140, 120]]);
assert(rows().length === 1 && A.cur.layers[0].kind === 'brush' && A.cur.layers[0].op === 'add', 'a stroke creates a brush layer');
assert(JSON.stringify(A.cur.layers[0].strokes[0].p[0]) === '[200,200]' && A.cur.layers[0].strokes[0].radius_px === 20, `stroke in plate px: ${JSON.stringify(A.cur.layers[0].strokes[0])}`);
assert(count('PUT', /areas$/) === 0 && q('status').textContent.includes('do zapisania'), 'saving is debounced');
stroke([[200, 100], [220, 100]]);
assert(rows().length === 1 && A.cur.layers[0].strokes.length === 2, 'the next stroke goes to the active brush layer');
await wait(SAVE);
let put = last('PUT', /keys\/sea\/60\/areas$/);
assert(put && count('PUT', /areas$/) === 1, 'one debounced PUT for both strokes');
let a0 = put[2].areas[0];
assert(put[2].areas.length === 1 && a0.id === 'a1' && a0.kind === 'brush' && a0.op === 'add' && a0.png === 'a_a1.png' && a0.label === '', `brush area: ${JSON.stringify({ ...a0, png_data: undefined })}`);
assert(a0.png_data === 'data:image/png;base64,W1280H720' && a0.width === 1280 && a0.height === 720, `brush layer as PNG data at the key's size: ${a0.png_data} ${a0.width}x${a0.height}`);
assert(!('strokes' in a0) && !('visible' in a0) && !('frame' in put[2]), 'only the contract fields are sent');
assert(q('status').textContent.includes('zapisane') && !A.cur.dirty, `saved status: ${q('status').textContent}`);
// [ ] and Shift + drag change the size
key('[', { code: 'BracketLeft' }); assert(A.input.radius === 16, `[ = smaller (${A.input.radius})`);
key(']', { code: 'BracketRight' }); assert(A.input.radius === 21, `] = bigger (${A.input.radius})`);
cvs.dispatchEvent(ev('pointerdown', 100, 100, { shiftKey: true })); cvs.dispatchEvent(ev('pointermove', 220, 100)); cvs.dispatchEvent(ev('pointerup', 220, 100));
assert(A.input.radius === Math.round(21 * Math.E) && A.cur.layers[0].strokes.length === 2, `Shift + drag = size, no stroke (${A.input.radius})`);

// ---------------------------------------------------------------- closed curve (MD_CURVES) -> a curve layer
tool('curve').click();
for (const p of [[300, 100], [400, 100], [350, 200]]) cvs.dispatchEvent(ev('mousedown', ...p));
assert(A.adding && A.adding.length === 3 && rows().length === 1, 'points of the new curve are collected');
key('Enter');
assert(!A.adding && rows().length === 2 && A.cur.layers[1].kind === 'curve', 'Enter closes the curve into a layer');
await wait(SAVE);
put = last('PUT', /keys\/sea\/60\/areas$/);
const a1 = put[2].areas[1];
assert(put[2].areas.length === 2 && a1.id === 'a2' && a1.kind === 'curve' && a1.op === 'add' && a1.curve.type === 'bezier' && a1.curve.closed === true && a1.curve.pts.length === 3, `curve area: ${JSON.stringify(a1).slice(0, 200)}`);
assert(JSON.stringify(a1.curve.pts.map((x) => x.p)) === '[[600,200],[800,200],[700,400]]', `curve in plate px: ${JSON.stringify(a1.curve.pts.map((x) => x.p))}`);
assert('in' in a1.curve.pts[0] && 'out' in a1.curve.pts[0] && 'cusp' in a1.curve.pts[0] && !('png' in a1), 'engine/curves.py point format');
assert(!('png_data' in put[2].areas[0]) && put[2].areas[0].png === 'a_a1.png', 'an unchanged brush layer is sent by file name only');
// the same curve edited through the ONE Bezier editor: drag a point
const nUndo = A.cur.undo.length;
cvs.dispatchEvent(ev('mousedown', 300, 100)); window.dispatchEvent(ev('mousemove', 310, 120)); window.dispatchEvent(ev('mouseup', 310, 120));
assert(JSON.stringify(A.cur.layers[1].curve.pts[0].p) === '[620,240]' && A.cur.undo.length === nUndo + 1, `point dragged by MD_CURVES.Editor: ${JSON.stringify(A.cur.layers[1].curve.pts[0].p)}`);
cvs.dispatchEvent(ev('mousedown', 400, 100)); window.dispatchEvent(ev('mouseup', 400, 100));
assert(A.cur.undo.length === nUndo + 1, 'a click that moves nothing leaves no undo step');
key('z', { ctrlKey: true, code: 'KeyZ' });
assert(JSON.stringify(A.cur.layers[1].curve.pts[0].p) === '[600,200]', 'Ctrl+Z restores the curve');
// a double click also closes; Esc cancels; fewer than 3 points = no curve
cvs.dispatchEvent(ev('mousedown', 500, 300)); key('Escape');
assert(!A.adding && rows().length === 2 && A.tool === 'curve', 'Esc cancels the new curve, the tool stays');
for (const p of [[500, 250], [560, 250], [530, 320], [530, 320]]) cvs.dispatchEvent(ev('mousedown', ...p));
cvs.dispatchEvent(ev('dblclick', 530, 320));
assert(rows().length === 3 && A.cur.layers[2].curve.pts.length === 3, `double click closes (${A.cur.layers[2] && A.cur.layers[2].curve.pts.length} pts)`);

// ---------------------------------------------------------------- layers: op, visibility, delete, undo
mount().querySelector('[data-ar-op="a2"]').click();
assert(A.cur.layers[1].op === 'remove' && mount().querySelector('[data-ar-op="a2"]').textContent.includes('odejmuje'), 'layer switched to remove');
const edCurve = () => A.ed.layers().find((l) => l.id === 'a2');
assert(edCurve().dash && edCurve().color === '#ff3c5a', 'a removing curve is red and dashed');
const vis = mount().querySelector('[data-ar-vis="a2"]');
const nPut = count('PUT', /areas$/);
vis.checked = false; vis.dispatchEvent(new window.Event('change', { bubbles: true }));
assert(!edCurve() && A.cur.layers[1].visible === false, 'hidden layer is not drawn');
mount().querySelector('[data-ar-del="a3"]').click();
assert(rows().length === 2, 'layer deleted');
await wait(SAVE);
put = last('PUT', /keys\/sea\/60\/areas$/);
assert(count('PUT', /areas$/) === nPut + 1 && put[2].areas.map((a) => `${a.id}:${a.op}`).join() === 'a1:add,a2:remove', `PUT after op + delete: ${put[2].areas.map((a) => `${a.id}:${a.op}`).join()}`);
key('z', { ctrlKey: true, code: 'KeyZ' });
assert(rows().length === 3, 'Ctrl+Z brings the deleted layer back');
q('undo').click();
assert(A.cur.layers[1].op === 'add', 'the ↶ button undoes the op switch');
const op = q('op'); op.value = '80'; op.dispatchEvent(new window.Event('input', { bubbles: true }));
assert(A.op === 80 && q('opv').textContent === '80 %', 'overlay opacity');

// ---------------------------------------------------------------- rectangle + eraser + a new brush layer
tool('rect').click();
cvs.dispatchEvent(ev('mousedown', 50, 50)); window.dispatchEvent(ev('mousemove', 150, 120)); window.dispatchEvent(ev('mouseup', 150, 120));
const rc = A.cur.layers[A.cur.layers.length - 1];
assert(rc.kind === 'curve' && rc.curve.closed && rc.curve.pts.length === 4 && rc.curve.pts.every((x) => x.cusp), 'rectangle = a closed curve of 4 cusp points');
assert(JSON.stringify(rc.curve.pts.map((x) => x.p)) === '[[100,100],[300,100],[300,240],[100,240]]', `rectangle in plate px: ${JSON.stringify(rc.curve.pts.map((x) => x.p))}`);
tool('erase').click();
toasts.length = 0;
stroke([[60, 60], [70, 70]]);
assert(lastToast().includes('Gumka') && A.cur.layers[0].strokes.length === 2, 'eraser needs an active brush layer');
rows()[0].click();
assert(A.cur.active === 'a1', 'click = active layer');
stroke([[110, 105], [130, 115]]);
assert(A.cur.layers[0].strokes.length === 3 && A.cur.layers[0].strokes[2].erase === true, 'eraser stroke on the active brush layer');
q('new').click();
assert(A.tool === 'brush' && A.pendingNew && q('list').textContent.includes('nowa warstwa'), '＋ warstwa pędzla waits for a stroke');
stroke([[400, 300], [420, 300]]);
assert(A.cur.layers.filter((l) => l.kind === 'brush').length === 2 && !A.pendingNew, 'the stroke started a new brush layer');
await wait(SAVE);
put = last('PUT', /keys\/sea\/60\/areas$/);
assert(put[2].areas.filter((a) => a.png_data).map((a) => a.id).join() === `a1,${A.cur.layers[A.cur.layers.length - 1].id}`, 'changed brush layers carry PNG data again');

// ---------------------------------------------------------------- pan / zoom pass through the editor canvas
panned.length = 0;
cvs.dispatchEvent(ev('mousedown', 10, 10, { button: 1 }));
cvs.dispatchEvent(new window.WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: -100, clientX: 10, clientY: 10 }));
assert(JSON.stringify(panned) === '[["down",1],["wheel",-100]]', `middle button + wheel reach the viewer: ${JSON.stringify(panned)}`);

// ---------------------------------------------------------------- a key with saved layers: brush kept by file name
viewer.goToFrame(10);
await wait();
assert(A.cur.key === 'key:sea:10' && rows().length === 2, `saved areas of the key loaded (${rows().length})`);
assert(A.imgs['key:sea:10/srv1'].img.src.endsWith('/api/cleanup3d/keys/sea/10/areas/srv1/png'), `pixels of a saved brush layer: ${A.imgs['key:sea:10/srv1'].img.src}`);
mount().querySelector('[data-ar-op="srv2"]').click();
await wait(SAVE);
put = last('PUT', /keys\/sea\/10\/areas$/);
assert(put && put[2].areas[0].id === 'srv1' && put[2].areas[0].png === 'a_srv1.png' && !('png_data' in put[2].areas[0]) && put[2].areas[1].op === 'remove', `saved brush layer untouched: ${JSON.stringify(put && put[2].areas[0])}`);
tool('brush').click(); if (A.tool !== 'brush') tool('brush').click();
rows()[0].click();
stroke([[20, 20], [30, 30]]);
assert(A.cur.layers.length === 3 && A.cur.layers[0].strokes.length === 0, 'pixels of the saved layer not loaded: the stroke goes to a new layer');
key('z', { ctrlKey: true, code: 'KeyZ' });
await wait(SAVE);

// ---------------------------------------------------------------- 501: the drawing stays, the server's text is shown
srv.mode = '501';
viewer.goToFrame(60);
await wait();
const n60 = A.cur.layers.length;
assert(A.cur.key === 'key:sea:60' && n60 === 5, `back on F60 with its layers (${n60})`);
toasts.length = 0;
stroke([[300, 300], [320, 310]]);
await wait(SAVE);
assert(lastToast() === DETAIL('P2 magazyn kluczy + P8'), `501 toast: ${lastToast()}`);
assert(A.cur.dirty && q('status').textContent.includes(DETAIL('P2 magazyn kluczy + P8')) && q('retry').style.display !== 'none', `status keeps the server's text: ${q('status').textContent}`);
stroke([[340, 300], [360, 310]]);
await wait(SAVE);
assert(toasts.length === 1, 'the same refusal is not toasted after every stroke');
const kept = JSON.stringify(A.cur.layers);
viewer.goToFrame(61); await wait();
viewer.goToFrame(60); await wait();
assert(JSON.stringify(A.cur.layers) === kept, 'the unsaved drawing survives leaving the frame and coming back');
srv.mode = 'ok';
q('retry').click();
await wait();
assert(!A.cur.dirty && !A.cur.error && q('status').textContent.includes('zapisane') && q('retry').style.display === 'none', 'retry saves when the server accepts');

// ---------------------------------------------------------------- a mark on a frame that is not a key
viewer.goToFrame(30);
await wait();
assert(A.cur.type === 'mark' && A.cur.key === 'mark:30', 'frame without a key = a mark');
assert(q('info').textContent.includes('F30') && q('mark').style.display !== 'none', `mark info: ${q('info').textContent}`);
assert(q('markto').textContent.includes('niebo: F10') && q('markto').textContent.includes('morze: F10, F60'), `keys the mark goes to: ${q('markto').textContent}`);
assert(q('send').disabled, 'nothing to carry yet');
const nMark = calls.length;
if (A.tool !== 'brush') tool('brush').click();
stroke([[100, 200], [150, 210]]);
await wait(SAVE);
assert(calls.length === nMark && !q('send').disabled, 'a mark is not saved by itself');
srv.mode = '501'; toasts.length = 0;
q('send').click();
await wait();
let mk = last('POST', /\/api\/cleanup3d\/mark\/30$/);
assert(mk && mk[2].cards === null && mk[2].areas.length === 1 && mk[2].areas[0].kind === 'brush' && mk[2].areas[0].png_data, `mark route called: ${JSON.stringify(mk && { ...mk[2], areas: mk[2].areas.length })}`);
assert(lastToast() === DETAIL('P2 magazyn kluczy + P8') && A.cur.layers.length === 1, 'mark 501: toast + the drawing stays');
viewer.goToFrame(31); await wait(); viewer.goToFrame(30); await wait();
assert(A.cur.layers.length === 1, 'the mark waits on its frame');
const only = q('only'); only.checked = true; only.dispatchEvent(new window.Event('change', { bubbles: true }));
assert(!q('markto').textContent.includes('niebo') && q('markto').textContent.includes('morze'), 'only the selected card');
srv.mode = 'ok';
q('send').click();
await wait();
mk = last('POST', /\/api\/cleanup3d\/mark\/30$/);
assert(mk[2].cards.join() === 'sea', 'cards of the mark request');
assert(lastToast().includes('morze F10') && lastToast().includes('morze F60') && A.cur.layers.length === 0, `mark carried: ${lastToast()}`);
assert(!A.cache['key:sea:10'], 'keys reload their areas after a mark');

// ---------------------------------------------------------------- C3: the extent of a card
document.querySelector('#c3StepList [data-c3step="cards"]').click();
await wait();
const em = $('c3ExtentMount');
assert(A.cur && A.cur.type === 'extent' && A.cur.key === 'ext:sea' && q('info', em).textContent.includes('cała karta'), 'extent target of the selected card');
assert(['brush', 'curve', 'erase', 'rect'].every((t) => tool(t)), 'the same tools in C3');
tool('rect').click();
cvs.dispatchEvent(ev('mousedown', 0, 180)); window.dispatchEvent(ev('mousemove', 640, 359)); window.dispatchEvent(ev('mouseup', 640, 359));
assert(A.cur.layers.length === 1 && A.cur.frame === 30, 'extent drawn in the current frame');
assert(A.ed.layers()[0].color === '#ffb84a', `extent outline in the card's colour (${A.ed.layers()[0].color})`);
await wait(SAVE);
let ex = last('PUT', /\/api\/cleanup3d\/cards\/sea\/extent$/);
assert(ex && ex[2].frame === 30 && ex[2].areas.length === 1 && ex[2].areas[0].kind === 'curve' && ex[2].areas[0].curve.closed, `extent PUT: ${JSON.stringify(ex && ex[2]).slice(0, 160)}`);
const sea = C3.state.cards.find((c) => c.id === 'sea');
assert(sea.extent.length === 1 && sea.extent_frame === 30, 'the saved extent is kept in the card list');
const pitch = document.querySelector('#c3CardEdit [data-c3adj="pitch"]');
pitch.value = '3'; pitch.dispatchEvent(new window.Event('input', { bubbles: true }));
await wait(400);
const cp = last('PUT', /\/api\/cleanup3d\/cards$/);
assert(cp[2].cards.find((c) => c.id === 'sea').extent.length === 1 && cp[2].cards.find((c) => c.id === 'sea').extent_frame === 30, 'a later PUT of the cards carries the saved extent, not the old one');
// another frame: the outline is not shown and not editable; „Skocz” goes back
viewer.goToFrame(31);
await wait();
assert(A.ed.layers().length === 0 && q('jumprow', em).style.display !== 'none' && q('jump', em).textContent.includes('F30'), 'extent editable only in its frame');
toasts.length = 0;
cvs.dispatchEvent(ev('mousedown', 10, 10)); window.dispatchEvent(ev('mouseup', 60, 60));
assert(A.cur.layers.length === 1 && lastToast().includes('F30'), `drawing refused off the extent frame: ${lastToast()}`);
q('jump', em).click();
await wait();
assert(viewer.currentFrame === 30 && A.ed.layers().length === 1 && q('jumprow', em).style.display === 'none', 'jump to the extent frame');
// brush extent + 501
srv.mode = '501'; toasts.length = 0;
tool('brush').click();
stroke([[100, 300], [500, 300]]);
await wait(SAVE);
ex = last('PUT', /\/api\/cleanup3d\/cards\/sea\/extent$/);
assert(ex[2].areas.length === 2 && ex[2].areas[1].kind === 'brush' && ex[2].areas[1].png_data && ex[2].frame === 30, 'brush extent sent as PNG data with the frame');
assert(lastToast() === DETAIL('P7 edytor zasięgu + P8') && A.cur.layers.length === 2 && sea.extent.length === 1, 'extent 501: toast, drawing kept, card list unchanged');
q('clear', em).click();
assert(A.cur.layers.length === 0 && A.cur.frame === null, 'clear = the whole card again');
key('z', { ctrlKey: true, code: 'KeyZ' });
assert(A.cur.layers.length === 2 && A.cur.frame === 30, 'Ctrl+Z restores the extent and its frame');
// another card has its own extent
C3.selectCard('sky');
await wait();
assert(A.cur.key === 'ext:sky' && A.cur.layers.length === 0, 'extent per card');

// ---------------------------------------------------------------- Esc = tool off; cutting mode = nothing on the viewer
key('Escape');
assert(A.tool === 'none' && !cvs.classList.contains('c3-area-on'), 'Esc switches the tool off');
document.querySelector('#modeSwitch [data-mode="cut"]').click();
await wait(40);
assert(!A.cur && cvs.style.display === 'none', 'cutting mode: the editor leaves the viewer');
const nKey = calls.length;
key('z', { ctrlKey: true, code: 'KeyZ' });
assert(calls.length === nKey, 'no editor shortcuts in the cutting mode');

if (process.env.C3_DEBUG) console.log(calls.map((c) => `${c[0]} ${c[1]}`).join('\n'));
console.log(`OK cleanup3d areas harness (${calls.length} api calls, ${toasts.length} toasts)`);
