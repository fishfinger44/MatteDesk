// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Sliders in the app style (user request 2026-10-23): the amber filled part of the track is drawn by index.css from
// the `--fill` custom property of each <input type=range>. Chromium has no ::range-progress, so this file keeps
// `--fill` up to date: on user input, when sliders are added to the page and when code sets `.value` directly.
(function () {
  'use strict';

  function update(el) {
    const min = parseFloat(el.min), max = parseFloat(el.max), v = parseFloat(el.value);
    const lo = isNaN(min) ? 0 : min, hi = isNaN(max) ? 100 : max;
    const f = hi > lo && !isNaN(v) ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0;
    el.style.setProperty('--fill', (f * 100).toFixed(2) + '%');
  }
  const isRange = (el) => el instanceof HTMLInputElement && el.type === 'range';

  // programmatic `slider.value = x` fires no event: wrap the value setter once
  const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
  if (desc && desc.set) {
    Object.defineProperty(HTMLInputElement.prototype, 'value', {
      configurable: true, enumerable: desc.enumerable, get: desc.get,
      set(v) { desc.set.call(this, v); if (this.type === 'range') update(this); },
    });
  }

  document.addEventListener('input', (e) => { if (isRange(e.target)) update(e.target); }, true);
  document.addEventListener('change', (e) => { if (isRange(e.target)) update(e.target); }, true);

  function scan(root) {
    if (isRange(root)) update(root);
    if (root.querySelectorAll) root.querySelectorAll('input[type=range]').forEach(update);
  }
  // panels are rebuilt with innerHTML; min/max/value attributes can change after insertion too
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === 'attributes') { if (isRange(m.target)) update(m.target); continue; }
      m.addedNodes.forEach(scan);
    }
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['value', 'min', 'max'] });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => scan(document));
  else scan(document);
})();
