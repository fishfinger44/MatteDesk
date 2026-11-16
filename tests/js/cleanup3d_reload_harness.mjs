// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// tests/js/cleanup3d_reload_harness.mjs – the two defects found in the real browser after wave 1 of Cleanup 3D, in
// jsdom with the REAL static/i18n.js and the real catalog static/i18n/en.json (English UI = the default language):
//  (a) a page reload in Cleanup mode with the step C3 „Karty”: objects.js shows its stage pane AFTER cleanup.js applied
//      the mode (project loaded -> setStage) – only the Cleanup pane may stay visible, and the cutting mode gets the
//      pane objects.js wanted last (visibility = computed style with the REAL static/index.css);
//  (b) the „Co robi / Najlepsze do” texts of all six steps come out in English (C3 stayed Polish: its one text node
//      matched a generic placeholder pattern before „Co robi: {0}”), and so do the texts of the area editor (P7).
// Run: cd tests/js && npm ci && node cleanup3d_reload_harness.mjs   (tests/test_cleanup3d_js.py wraps it)
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');
const here = path.dirname(fileURLToPath(import.meta.url));
const STATIC = path.resolve(here, '..', '..', 'static');
const CATALOG = JSON.parse(readFileSync(path.join(STATIC, 'i18n', 'en.json'), 'utf-8'));
const EN = CATALOG.strings;

const WORK = 'D:/W/shot/matte_desk';
const html = `<!doctype html><html><body>
  <header class="app-header"><div class="header-left"></div><div class="header-center"><button id="langToggle"></button>
    <select id="plateSelect"><option value="beauty">Beauty</option></select><div class="view-mode-tabs" id="viewModeTabs"></div></div></header>
  <nav class="stage-bar" id="stageBar"><div id="stageObjectBox"></div><ol id="stageList"></ol><div id="stageHint"></div></nav>
  <aside class="tools-sidebar" id="toolsSidebar"><div id="drawToolsTitle">Rysuj</div><div id="drawTools"></div></aside>
  <div id="canvasWrapper"><canvas id="cv" width="640" height="360"></canvas></div>
  <aside class="stage-panel" id="stagePanel"><div class="stage-panel-empty" id="stagePanelEmpty"></div>
    <div class="stage-pane" data-stage="sam3" id="paneSam3">1 · SAM 3.1</div>
    <div class="stage-pane" data-stage="cotracker" id="paneBrush" style="display:none">2 · CoTracker3</div></aside>
  <div id="timelineTrackContainer"><div id="timelineKeys"></div></div></body></html>`;
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://127.0.0.1:8765/' });
const { window } = dom;
const { document } = window;
// the REAL stylesheet: Cleanup mode hides the cutting stages' panes with a rule of index.css, not with inline styles
{ const st = document.createElement('style'); st.textContent = readFileSync(path.join(STATIC, 'index.css'), 'utf-8'); document.head.appendChild(st); }
const fakeCtx = (canvas) => new Proxy({ canvas, globalAlpha: 1 }, { get(t, k) { return k in t ? t[k] : () => ({ data: [] }); }, set(t, k, v) { t[k] = v; return true; } });
window.HTMLCanvasElement.prototype.getContext = function () { return this.__ctx || (this.__ctx = fakeCtx(this)); };
window.HTMLCanvasElement.prototype.getBoundingClientRect = function () { return { left: 0, top: 0, width: this.width, height: this.height }; };
if (!window.CanvasRenderingContext2D) { window.CanvasRenderingContext2D = function () {}; window.CanvasRenderingContext2D.prototype.fillText = function () {}; }

const assert = (c, msg) => { if (!c) throw new Error(`ASSERT: ${msg}`); };
const wait = (ms = 20) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);
const shown = (e) => !!e && window.getComputedStyle(e).display !== 'none';
const POLISH = /[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ„”]/;

// the page was left in Cleanup mode on the step „Karty” – this is the reload
window.localStorage.setItem('md_mode', 'clean');
window.localStorage.setItem(`md_c3_step:${WORK}`, 'cards');

const card = (id, name, color, keys) => ({ id, name, kind: 'plane', plane: { n: [0, -1, 0], d: 1 }, adjust: { dh: 0, pitch: 0, roll: 0 },
  extent: [], extent_frame: null, seam_px: 16, order: 0, color, key_mode: 'manual', key_every: 8, warps: {}, keys });
const DETAIL = 'Czyszczenie 3D: ta funkcja jeszcze nie jest zbudowana (P2 magazyn kluczy + P8)';
async function fakeApi(method, url) {
  if (method === 'GET' && url === '/api/cleanup') return { objects: [], settings: { remove: [], sources: {}, grow_px: 8, soft_px: 4, holdout: true, fill: 'auto' }, sources: {}, versions: [], out_dir: 'W/inpaint' };
  if (method === 'GET' && url === '/api/cleanup3d') return { steps: [], done: {}, camera: null, cards: [card('sea', 'morze', '#ffb84a', [{ frame: 10, active: 'v001', versions: 1 }])],
    engines: [{ id: 'lama', label: 'LaMa' }], key_modes: ['manual', 'every'], views: [], key_formats: ['png'] };
  if (method === 'GET') return { areas: [] };
  const e = new Error(DETAIL); e.status = 501; throw e;
}
const toasts = [];
window.UI = { api: fakeApi, toast: (t) => toasts.push(String(t)), dialog: async () => true, confirmDelete: async () => true, bytes: (n) => `${n} B` };
window.fetch = async (url) => { assert(String(url).includes('/static/i18n/en.json'), `fetch ${url}`); return { json: async () => CATALOG }; };
window.MD = { S: { objects: [] }, renderStageBar: () => {} };
window.viewer = { projectLoaded: true, currentWorkDir: WORK, manifest: { plates: {} }, projectData: { plate_w: 1280, plate_h: 720 }, canvas: $('cv'),
  zoom: 1, panX: 0, panY: 0, overlayHooks: [], firstFrame: 1, lastFrame: 100, currentFrame: 30, activePlate: 'beauty',
  getCurrentImage() { return { width: 640, height: 360 }; }, setActivePlate() {}, render() {},
  goToFrame(f) { this.currentFrame = f; window.dispatchEvent(new window.CustomEvent('md-frame', { detail: { frame: f } })); } };

for (const n of ['i18n.js', 'brushcore.js', 'cleanup.js', 'cleanup3d.js', 'curves.js', 'cleanup3d_areas.js']) window.eval(readFileSync(path.join(STATIC, n), 'utf-8'));
await wait(60);
assert(window.mdLang === 'en' && window.mdT('Anuluj') === EN.Anuluj, 'English UI with the real catalog');
const C3 = window.MD_CLEANUP3D;
assert(window.MD_CLEANUP.mode === 'clean' && C3.step === 'cards', `reload lands in Cleanup, step C3 (${window.MD_CLEANUP.mode}, ${C3.step})`);

// ---------------------------------------------------------------- (a) objects.js shows its pane after the reload
const visiblePanes = () => [...$('stagePanel').children].filter((c) => shown(c)).map((c) => c.id);
assert(visiblePanes().join() === 'c3PaneCards', `before the project loads: ${visiblePanes()}`);
const setStage = (on) => { for (const id of ['paneSam3', 'paneBrush']) $(id).style.display = id === on ? '' : 'none'; $('stagePanelEmpty').style.display = 'none'; };   // as objects.js setStage
setStage('paneSam3');                                   // project loaded -> setStage(1)
await wait();
assert(visiblePanes().join() === 'c3PaneCards', `(a) only the Cleanup pane after the reload: ${visiblePanes()}`);
setStage('paneBrush');                                  // the stage changes behind Cleanup (e.g. another object selected)
await wait();
assert(visiblePanes().join() === 'c3PaneCards', `(a) still only the Cleanup pane: ${visiblePanes()}`);
C3.setStep('remove'); await wait();
assert(visiblePanes().join() === 'cleanupPanel', `(a) C1 shows the MVP panel alone: ${visiblePanes()}`);
C3.setStep('cards'); await wait();

// ---------------------------------------------------------------- (b) all six step descriptions in English
const steps = { remove: 'cleanupPanel', camera: 'c3PaneCamera', cards: 'c3PaneCards', keys: 'c3PaneKeys', project: 'c3PaneProject', export: 'c3PaneExport' };
const js = readFileSync(path.join(STATIC, 'cleanup3d.js'), 'utf-8');
const sentences = [...js.matchAll(/\$\{what\('((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'\)\}/g)].map((m) => [m[1], m[2]].map((s) => s.replace(/\\'/g, "'")));
assert(sentences.length === 5, `five what() descriptions in cleanup3d.js (${sentences.length})`);
let n = 0;
for (const [step, pane] of Object.entries(steps)) {
  C3.setStep(step); await wait();
  const p = $(pane);
  assert(shown(p), `${step}: pane shown`);
  if (step === 'remove') {
    const t = p.querySelector('.pp-hint').textContent;
    assert(t.startsWith('What it does:') && t.includes('Best for:') && !POLISH.test(t), `C1 description in English: ${t}`);
    continue;
  }
  const lines = [...p.querySelectorAll('.c3-what')].map((d) => d.textContent.replace(/\s+/g, ' ').trim());
  const [does, best] = sentences[n++];
  assert(EN[does] && EN[best], `${step}: both sentences in the catalog`);
  assert(lines[0] === `What it does: ${EN[does]}` && lines[1] === `Best for: ${EN[best]}`, `${step} description in English:\n  ${lines.join('\n  ')}`);
  assert(!POLISH.test(lines.join(' ')), `${step}: no Polish left: ${lines.join(' | ')}`);
}

// ---------------------------------------------------------------- the area editor's texts (P7) in English too
const AR = window.MD_C3AREAS;
const noPolish = (el, what) => {
  const txt = [el.textContent, ...[...el.querySelectorAll('[title]')].map((x) => x.getAttribute('title'))].join(' | ');
  assert(!POLISH.test(txt) && !/\b(warstw|klucz|zaznaczeni|obszar|kart[aęy]|klatk)/i.test(txt), `${what} not fully translated: ${txt}`);
};
C3.setStep('cards'); await wait();
assert(AR.state.cur && AR.state.cur.type === 'extent', 'extent editor mounted');
noPolish($('c3ToolMount'), 'tools'); noPolish($('c3ExtentMount'), 'extent palette'); noPolish($('c3ToolHint'), 'tool hint');
AR.setTool('rect');
const cvs = $('c3AreaCanvas');
const ev = (type, x, y) => new window.MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 });
cvs.dispatchEvent(ev('mousedown', 10, 10)); window.dispatchEvent(ev('mousemove', 200, 100)); window.dispatchEvent(ev('mouseup', 200, 100));
await wait(750);
assert(toasts.pop() === DETAIL, 'the server\'s 501 text reaches the toast');
noPolish($('c3ExtentMount').querySelector('[data-ar="list"]'), 'extent layer list');
const st = $('c3ExtentMount').querySelector('[data-ar="status"]').textContent;
assert(st.startsWith('⚠ Not saved on the server:') && st.includes('kept here'), `status in English around the server text: ${st}`);
window.viewer.goToFrame(31); await wait();
noPolish($('c3ExtentMount').querySelector('[data-ar="info"]'), 'extent info off its frame');
assert($('c3ExtentMount').querySelector('[data-ar="jump"]').textContent === 'Jump to F30', 'jump button');
C3.setStep('keys'); await wait();
assert(AR.state.cur.type === 'mark', 'mark target on a frame without a key');
noPolish($('c3AreaMount'), 'mark palette');
assert($('c3AreaMount').querySelector('[data-ar="info"]').textContent === 'Frame F31 is not a key of card "morze" – the 3D camera carries a mark from this frame to the keys.', `mark info: ${$('c3AreaMount').querySelector('[data-ar="info"]').textContent}`);
window.viewer.goToFrame(10); await wait();
assert($('c3AreaMount').querySelector('[data-ar="info"]').textContent === 'Key F10 of card "morze": you draw the areas in this key\'s pixels.', `key info: ${$('c3AreaMount').querySelector('[data-ar="info"]').textContent}`);

// ---------------------------------------------------------------- (a) back in the cutting mode: the pane objects.js wanted last
document.querySelector('#modeSwitch [data-mode="cut"]').click();
await wait();
assert(visiblePanes().join() === 'paneBrush', `cutting mode shows the stage pane objects.js set last: ${visiblePanes()}`);
setStage('paneSam3'); await wait();
assert(visiblePanes().join() === 'paneSam3', 'in the cutting mode the panes are left alone');

console.log('OK cleanup3d reload harness');
