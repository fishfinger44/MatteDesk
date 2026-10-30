// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Ctrl+Z / Ctrl+Shift+Z for sliders (user report 2026-10-28). One history for every <input type=range> (and the number
// field paired with a slider in the same row), recorded on `change` = what the user committed. Undo puts the old value
// back and fires `input` + `change`, so the slider's own handlers save it exactly like a drag – no per-stage code.
// History per object + stage (MD_SLIDER_UNDO.ctx from objects.js); the newest action wins: a slider change after the
// last edit on the image / keys lane is undone first, an older one waits until those are undone.
(function () {
  'use strict';

  const MAX = 100;
  const stacks = new Map();     // ctx -> { undo: [], redo: [] }
  const before = new WeakMap(); // element -> value when the user started changing it
  let applying = false;
  let lastSliderAt = 0, lastCanvasAt = 0;

  const ctxKey = () => { try { return (API.ctx && API.ctx()) || ''; } catch (e) { return ''; } };
  const stack = (k) => { if (!stacks.has(k)) stacks.set(k, { undo: [], redo: [] }); return stacks.get(k); };
  const isRange = (el) => el instanceof HTMLInputElement && el.type === 'range';
  const tracked = (el) => el instanceof HTMLInputElement && !el.disabled && (isRange(el)
    || (el.type === 'number' && !!el.parentElement && !!el.parentElement.querySelector('input[type=range]')));

  // panels re-render: find the element again by id, or by its position under the nearest ancestor with an id / data-*
  function locate(el) {
    if (el.id) return { sel: `#${CSS.escape(el.id)}` };
    let a = el.parentElement;
    while (a && !a.id && !Object.keys(a.dataset || {}).length) a = a.parentElement;
    if (!a) return null;
    const asel = a.id ? `#${CSS.escape(a.id)}` : Object.entries(a.dataset).map(([k, v]) =>
      `[data-${k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())}="${CSS.escape(v)}"]`).join('');
    const own = Object.entries(el.dataset || {}).map(([k, v]) =>
      `[data-${k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())}="${CSS.escape(v)}"]`).join('');
    const isel = `input[type=${el.type}]${own}`;
    return { sel: asel, isel, idx: Array.from(a.querySelectorAll(isel)).indexOf(el) };
  }
  function find(loc) {
    if (!loc.isel) return document.querySelector(loc.sel);
    const a = document.querySelector(loc.sel);
    return a ? a.querySelectorAll(loc.isel)[loc.idx] || null : null;
  }

  // the value before the user's change; code may have set a new value since the last one (frame / object switch), so a
  // grab or an arrow key on a slider always takes the current value. A click on a track fires `input` BEFORE `focusin`,
  // so focus only arms a field not armed yet (Tab), and leaving the field forgets it
  const arm = (force) => (e) => {
    const el = e.target;
    if (tracked(el) && (force || !before.has(el))) before.set(el, el.value);
  };
  document.addEventListener('pointerdown', arm(true), true);
  document.addEventListener('focusin', arm(false), true);
  document.addEventListener('focusout', (e) => before.delete(e.target), true);
  document.addEventListener('keydown', (e) => { if (!(e.ctrlKey || e.metaKey) && e.key !== 'Shift') arm(isRange(e.target))(e); }, true);
  document.addEventListener('change', (e) => {
    const el = e.target;
    if (applying || !tracked(el)) return;
    const old = before.get(el); before.delete(el);
    if (old === undefined || old === el.value) return;
    const loc = locate(el); if (!loc) return;
    const s = stack(ctxKey());
    s.undo.push({ loc, old, val: el.value }); if (s.undo.length > MAX) s.undo.shift();
    s.redo.length = 0;
    lastSliderAt = performance.now();
    if (document.activeElement === el) before.set(el, el.value);   // keyboard: the next arrow press starts from here
  }, true);
  // edits on the image / keys lanes: their undo goes first when they are newer than the last slider change
  document.addEventListener('pointerdown', (e) => {
    if (e.target.closest && e.target.closest('#canvasWrapper, #timelineKeys, #timelineZones, #timelineSegs')) lastCanvasAt = performance.now();
  }, true);

  function apply(redo) {
    const s = stack(ctxKey());
    const from = redo ? s.redo : s.undo, to = redo ? s.undo : s.redo;
    while (from.length) {
      const it = from.pop();
      const el = find(it.loc);
      if (!el || el.disabled) continue;   // the panel is gone / locked now: that entry cannot be applied any more
      applying = true;
      try {
        el.value = redo ? it.val : it.old;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      } finally { applying = false; }
      before.delete(el);
      to.push(it);
      lastSliderAt = performance.now();
      const lab = el.closest('label');
      const name = lab ? (lab.firstChild && lab.firstChild.nodeType === 3 ? lab.firstChild.textContent : lab.textContent).trim().split('\n')[0] : '';
      if (window.viewer && window.viewer.showToast) window.viewer.showToast(redo ? `Ponowiono suwak „${name}”: ${el.value}` : `Cofnięto suwak „${name}”: ${el.value}`);
      return true;
    }
    return false;
  }

  // Ctrl+Z while a slider / its number field has the focus (the global shortcut skips focused inputs): a slider change
  // when it is the newest action, else the field lets go and the same key goes to the image's undo (objects.js)
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'KeyZ' || !(e.ctrlKey || e.metaKey) || !tracked(e.target)) return;
    e.preventDefault(); e.stopPropagation();
    if (API.wants(e.shiftKey) && apply(e.shiftKey)) return;
    e.target.blur();
    document.body.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyZ', key: e.key, ctrlKey: e.ctrlKey, metaKey: e.metaKey, shiftKey: e.shiftKey, bubbles: true }));
  }, true);

  const API = {
    ctx: null,   // set by objects.js: () => `${object}|${stage}`
    // the global Ctrl+Z asks first: is the newest undoable action a slider change?
    wants: (redo) => { const s = stack(ctxKey()); return redo ? s.redo.length > 0 && lastSliderAt >= lastCanvasAt
                                                              : s.undo.length > 0 && lastSliderAt >= lastCanvasAt; },
    undo: () => apply(false),
    redo: () => apply(true),
    clear: () => stacks.clear(),
  };
  window.MD_SLIDER_UNDO = API;
})();
