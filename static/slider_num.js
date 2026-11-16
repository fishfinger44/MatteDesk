// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// ONE number field for every slider (user request 2026-11-11: „the values can be typed and stepped with arrows like the
// palette on the right – unify it everywhere, a change in one place changes it everywhere”). Like range_fill.js this
// file enhances every <input type=range> of the page by itself (panels are rebuilt with innerHTML – a MutationObserver):
//   • a number field right after the slider: type + Enter / leave the field, ↑ / ↓ step by the slider's step, after
//     typing hundredths (or finer) by that digit (as the MergeMatte palette);
//   • the field writes the slider and fires its own 'input' / 'change' events, so every owner keeps its handlers;
//   • the slider (dragged, or set by code) writes the field back;
//   • a value label next to the slider (<b> / <span> showing the same number, e.g. „4.5 px”) is replaced by the field and
//     its unit kept; a label showing a converted value (brush size, shutter °, ×strength) is left alone – no field;
//   • sliders that already have their own number field in the same row (MergeMatte, the C1 band) or carry
//     data-num="off" are skipped.
(function () {
  'use strict';
  // a value label: an optional prefix (×), the number, the rest kept as the unit text („px”, „° (manifest)”)
  const NUM_RE = /^\s*([x×]?)\s*(-?\d+(?:[.,]\d+)?)\s*(.*)$/i;
  const restOf = (txt) => { const m = String(txt || '').trim().match(NUM_RE); return m ? { pre: m[1] || '', post: (m[3] || '').trim() } : null; };
  const isRange = (el) => el instanceof HTMLInputElement && el.type === 'range';
  const decimals = (step) => { const s = String(step || '1'); return s.includes('.') ? s.split('.')[1].length : 0; };
  const fmt = (v, step) => { const n = parseFloat(v); return Number.isFinite(n) ? n.toFixed(Math.min(4, decimals(step))) : ''; };
  const typedStep = (txt, base) => { const m = String(txt).match(/[.,](\d+)/); const d = m ? m[1].length : 0; return d >= 2 ? 10 ** -d : base; };
  const pending = new WeakSet();

  function hasOwnField(r) {
    const row = r.parentElement;
    return !!(row && [...row.querySelectorAll('input[type=number]')].some((n) => !n.classList.contains('md-num')));
  }
  function valueLabel(r) {
    const nx = r.nextElementSibling;
    return nx && (nx.tagName === 'B' || nx.tagName === 'SPAN') && !nx.classList.contains('md-num-unit') ? nx : null;
  }

  function enhance(r) {
    if (!isRange(r) || r.dataset.num === 'off' || r.dataset.numDone) return;
    if (hasOwnField(r)) { r.dataset.numDone = 'own'; return; }
    const lbl = valueLabel(r);
    let unit = '', pre = '';
    if (lbl) {
      const txt = (lbl.textContent || '').trim();
      if (!txt) {                                   // the owner fills it later: decide then
        if (!pending.has(lbl)) {
          pending.add(lbl);
          new MutationObserver((ms, ob) => { if ((lbl.textContent || '').trim()) { ob.disconnect(); pending.delete(lbl); enhance(r); } })
            .observe(lbl, { childList: true, characterData: true, subtree: true });
        }
        return;
      }
      const m = txt.match(NUM_RE);
      if (!m || m[3].length > 24 || Math.abs(parseFloat(m[2].replace(',', '.')) - parseFloat(r.value)) > Math.max(1e-6, parseFloat(r.step || '1') / 2 + 1e-9)) {
        r.dataset.numDone = 'label';                // a converted value (or text): the label stays, no field
        return;
      }
      unit = (m[3] || '').trim();
      pre = m[1] || '';
      lbl.style.display = 'none';
    }
    r.dataset.numDone = '1';
    const n = document.createElement('input');
    n.type = 'number';
    n.className = 'form-input edge-num md-num';
    n.min = r.min; n.max = r.max; n.step = r.step || '1';
    n.title = `${r.title ? r.title + ' – ' : ''}wpisz i Enter; strzałki ↑ / ↓ = krok suwaka, po wpisaniu setnych ±0,01`;
    n.value = fmt(r.value, r.step);
    r.after(n);
    if (pre) {
      const pr = document.createElement('span');
      pr.className = 'md-num-unit muted';
      pr.textContent = pre;
      r.after(pr);
    }
    if (lbl) {   // the unit / note of the hidden label follows it (e.g. „° (manifest)” ↔ „° (własny)”)
      const u = document.createElement('span');
      u.className = 'md-num-unit muted';
      u.textContent = unit;
      n.after(u);
      new MutationObserver(() => { const x = restOf(lbl.textContent); if (x) u.textContent = x.post; })
        .observe(lbl, { childList: true, characterData: true, subtree: true });
    }
    const toRange = (live) => {
      const v = parseFloat(String(n.value).replace(',', '.'));
      if (!Number.isFinite(v)) { if (!live) n.value = fmt(r.value, r.step); return; }
      const lo = parseFloat(r.min), hi = parseFloat(r.max);
      const c = Math.min(Number.isFinite(hi) ? hi : v, Math.max(Number.isFinite(lo) ? lo : v, v));
      if (String(c) !== String(parseFloat(r.value))) {
        r.value = c;
        r.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (!live) { r.dispatchEvent(new Event('change', { bubbles: true })); n.value = fmt(r.value, r.step); }
    };
    n.addEventListener('input', (e) => {
      if (e.inputType && e.inputType !== 'insertReplacementText') n.step = String(typedStep(n.value, parseFloat(r.step || '1')));
      toRange(true);
    });
    n.addEventListener('change', () => toRange(false));
    n.addEventListener('blur', () => { n.step = r.step || '1'; });
    n.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); n.blur(); } e.stopPropagation(); });
    for (const ev of ['mousedown', 'pointerdown', 'wheel', 'dblclick']) n.addEventListener(ev, (e) => e.stopPropagation());
    const sync = () => { if (document.activeElement !== n) n.value = fmt(r.value, r.step); n.disabled = r.disabled; };
    r.addEventListener('input', sync);
    r.addEventListener('change', sync);
    r._mdNumSync = sync;
    sync();
  }

  // code sets slider.value directly (no event): keep the field in step (range_fill.js wraps the same setter)
  const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
  if (desc && desc.set) {
    Object.defineProperty(HTMLInputElement.prototype, 'value', {
      configurable: true, enumerable: desc.enumerable, get: desc.get,
      set(v) { desc.set.call(this, v); if (this.type === 'range' && this._mdNumSync) this._mdNumSync(); },
    });
  }
  function scan(root) {
    if (isRange(root)) enhance(root);
    if (root.querySelectorAll) root.querySelectorAll('input[type=range]').forEach(enhance);
  }
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === 'attributes') { const t = m.target; if (isRange(t) && t._mdNumSync) { const n = t.nextElementSibling; if (n && n.classList.contains('md-num')) { n.min = t.min; n.max = t.max; n.step = t.step || '1'; } t._mdNumSync(); } continue; }
      m.addedNodes.forEach(scan);
    }
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['min', 'max', 'step', 'disabled'] });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => scan(document)); else scan(document);
  window.MD_SLIDER_NUM = { enhance, scan };
})();
