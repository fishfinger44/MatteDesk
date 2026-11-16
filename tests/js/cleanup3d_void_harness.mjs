// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/cleanup3d_void_harness.mjs – the VOID + SeedVR2 block of static/cleanup3d.js (Cleanup 3D package P12) in jsdom
// with a fake API: mounted in the „Klucz karty” palette under „Tło kluczy z plate'u”, buttons disabled with the gate's
// reason when the tools are not configured, the last run / a failed attempt shown, the prompt taken from the last run,
// and the three requests (POST /api/cleanup3d/void, /void/keys, /keys/{card}/seedvr2) with the server's answer in a toast.
// Run: cd tests/js && npm ci && node cleanup3d_void_harness.mjs   (tests/test_cleanup3d_js.py wraps it)
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.resolve(here, '..', '..', 'static');

const html = `<!doctype html><html><body>
  <header class="app-header"><div class="header-left"></div><div class="header-center"><select id="plateSelect"><option value="beauty">Beauty</option></select>
    <div class="view-mode-tabs" id="viewModeTabs"></div></div></header>
  <nav class="stage-bar" id="stageBar"><div class="stage-object" id="stageObjectBox">obj</div><ol class="stage-list" id="stageList"></ol>
    <div class="stage-hint" id="stageHint"></div></nav>
  <aside class="tools-sidebar" id="toolsSidebar"></aside>
  <div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div>
  <aside class="stage-panel" id="stagePanel"><div class="stage-pane" id="objPane">etap</div></aside>
  <div id="timelineTrackContainer"><div id="timelineKeys"></div></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;
window.HTMLCanvasElement.prototype.getContext = function () { return new Proxy({}, { get: () => () => ({ data: [] }) }); };
const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms = 15) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);

const card = (id, keys) => ({ id, name: id, kind: 'plane', plane: { n: [0, -1, 0], d: 1 }, adjust: { dh: 0, pitch: 0, roll: 0 }, extent: [],
  extent_frame: null, seam_px: 16, order: 0, color: '#ffb84a', key_mode: 'manual', key_every: 4, warps: {}, keys });
let tools = { void: { available: false, reason: 'VOID (usuwanie obiektu z wideo) nie jest skonfigurowany: void_python w config.local.json', last: null, newest: null },
  seedvr2: { available: false, reason: 'SeedVR2 (podbicie kluczy) nie jest skonfigurowany' } };
const state = () => ({ steps: ['remove', 'camera', 'cards', 'keys', 'project', 'export'], done: { camera: true, cards: true, keys: true },
  camera: { frames: [1, 57], n_frames: 57, report: {} }, cards: [card('sea', [{ frame: 1, active: 'v002', versions: 2 }])],
  engines: [{ id: 'lama', label: 'LaMa' }], key_modes: ['manual', 'every'], views: ['key'], key_formats: ['png'], ...tools });
const calls = [];
async function fakeApi(method, url, body) {
  calls.push([method, url, body ? JSON.parse(JSON.stringify(body)) : body]);
  if (method === 'GET' && url === '/api/cleanup') return { objects: [], settings: { remove: [], sources: {} }, sources: {}, versions: [], out_dir: 'W' };
  if (method === 'GET' && url === '/api/cleanup3d') return state();
  if (method === 'POST' && url === '/api/cleanup3d/void') return { success: true, run: { id: 'run_1', status: 'done', frames: [1, 2, 3] }, keys: [{ card: 'sea', frame: 1 }] };
  if (method === 'POST' && url === '/api/cleanup3d/void/keys') return { success: true, run: 'run_1', keys: [{ card: 'sea', frame: 1 }, { card: 'sea', frame: 5 }] };
  if (method === 'POST' && url === '/api/cleanup3d/keys/sea/seedvr2') return { success: true, crop: [10, 20, 512, 640], keys: [{ frame: 1 }] };
  const e = new Error(`fake api: ${method} ${url}`); e.status = 404; throw e;
}
const toasts = [];
window.UI = { api: fakeApi, toast: (t) => toasts.push(String(t)), dialog: async () => true, confirmDelete: async () => true, bytes: (n) => `${n} B` };
window.viewer = { projectLoaded: true, currentWorkDir: 'D:/W/shot/matte_desk', manifest: { plates: { beauty: 'b.####.png' } },
  firstFrame: 1, lastFrame: 57, currentFrame: 1, activePlate: 'beauty', setActivePlate() {}, goToFrame() {}, render() {} };
window.MD = { placeFloating() {}, makeFloating() {}, renderStageBar() {}, S: { objects: [] } };
window.eval(readFileSync(path.join(STATIC, 'cleanup.js'), 'utf-8'));
window.eval(readFileSync(path.join(STATIC, 'cleanup3d.js'), 'utf-8'));
await wait();
const C3 = window.MD_CLEANUP3D;
assert(C3 && C3.ready, 'MD_CLEANUP3D ready');
document.querySelector('#modeSwitch [data-mode="clean"]').click();
await wait();
C3.setStep('keys');
await C3.refresh();

// mounted once, in the keys palette, right under „Tło kluczy z plate'u”
const box = $('c3VoidBox');
assert(box && box.closest('#c3PalKeys'), 'VOID block in the keys palette');
assert(document.querySelectorAll('#c3VoidBox').length === 1, 'mounted once');
assert($('c3KeyRebase').closest('.pp-sec').nextElementSibling === box, 'right after the rebase section');
// not configured: disabled with the gate's reason
assert($('c3VoidRun').disabled && $('c3VoidKeys').disabled && $('c3SeedVR2').disabled, 'disabled without the tools');
assert($('c3VoidInfo').textContent.includes('VOID niedostępny') && $('c3VoidInfo').textContent.includes('config.local.json'), `void reason: ${$('c3VoidInfo').textContent}`);
assert($('c3SvInfo').textContent.includes('SeedVR2 niedostępny'), 'seedvr2 reason');

// configured, a finished run and a newer failed attempt
tools = { void: { available: true, reason: '', last: { id: 'run_1', status: 'done', finished: '2026-11-10 20:00:00', frames: 57, size: [1344, 768], prompt: 'empty beach' },
  newest: { id: 'run_2', status: 'failed', date: '2026-11-10 21:00:00', error: 'CUDA out of memory', frames: 57 } },
seedvr2: { available: true, reason: '' } };
await C3.refresh();
assert(!$('c3VoidRun').disabled && !$('c3VoidKeys').disabled && !$('c3SeedVR2').disabled, 'enabled with the tools');
const info = $('c3VoidInfo').textContent;
assert(info.includes('Ostatni przebieg') && info.includes('57 kl.') && info.includes('1344×768'), `last run: ${info}`);
assert(info.includes('Ostatnia próba') && info.includes('CUDA out of memory'), `failed attempt: ${info}`);
assert($('c3VoidPrompt').value === 'empty beach', 'prompt from the last run');
assert($('c3SvInfo').textContent === '', 'no seedvr2 reason');

// the three requests
$('c3VoidPrompt').value = 'sunset beach, calm sea';
$('c3VoidRun').click();
await wait(40);
let c = calls.find((x) => x[0] === 'POST' && x[1] === '/api/cleanup3d/void');
assert(c && c[2].prompt === 'sunset beach, calm sea' && c[2].keys === true, `void request: ${JSON.stringify(c)}`);
assert(toasts.some((t) => t.includes('VOID policzony: 3 klatek') && t.includes('nowe wersje kluczy: 1')), `void toast: ${toasts}`);
$('c3VoidKeys').click();
await wait(40);
assert(calls.some((x) => x[0] === 'POST' && x[1] === '/api/cleanup3d/void/keys'), 'void keys request');
assert(toasts.some((t) => t.includes('Klucze z VOID: 2 nowych wersji')), 'void keys toast');
$('c3SeedVR2').click();
await wait(40);
assert(calls.some((x) => x[0] === 'POST' && x[1] === '/api/cleanup3d/keys/sea/seedvr2'), 'seedvr2 request for the selected card');
assert(toasts.some((t) => t.includes('SeedVR2: 1 kluczy podbitych') && t.includes('512×640')), `seedvr2 toast: ${toasts}`);
assert(!$('c3VoidRun').disabled && !$('c3SeedVR2').disabled, 'buttons back after the actions');

console.log(`OK cleanup3d void harness (${calls.length} api calls, ${toasts.length} toasts)`);
