// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Top bar that always fits the window (user report 2026-10-15: the Doctor button fell off the right edge on smaller
// screens). When the header overflows, the least important groups move into the "⋯" menu, one by one, until it fits;
// "📁 Projekt", the view tabs and "🩺 Doktor" never move. Nodes are moved, not cloned, so their listeners keep working;
// a comment placeholder marks where each one goes back.
(function (root) {
  'use strict';

  // first = folded first
  const FOLD_ORDER = ['overlayStyle', 'shotInfoPill', 'plateSelectorPill', 'comfyStatusIndicator'];

  // Pure part (unit-tested in node): fold items in order while `overflows()` says the bar is too wide.
  // `fold(item)` moves one item into the menu, `unfoldAll()` puts everything back. Returns the folded items.
  function fitItems(items, overflows, fold, unfoldAll) {
    unfoldAll();
    const folded = [];
    for (const it of items) {
      if (!overflows()) break;
      fold(it);
      folded.push(it);
    }
    return folded;
  }

  function init(doc) {
    const header = doc.querySelector('.app-header');
    const wrap = doc.getElementById('headerMoreWrap');
    const btn = doc.getElementById('headerMoreBtn');
    const menu = doc.getElementById('headerMoreMenu');
    if (!header || !wrap || !btn || !menu) return null;

    const items = [];
    for (const id of FOLD_ORDER) {
      const el = doc.getElementById(id);
      if (!el) continue;
      const ph = doc.createComment('header-fit:' + id);
      el.parentNode.insertBefore(ph, el);
      items.push({ id, el, ph });
    }

    const overflows = () => header.scrollWidth > header.clientWidth + 1;
    const fold = (it) => { wrap.hidden = false; menu.appendChild(it.el); it.el.classList.add('in-more-menu'); };
    const unfoldAll = () => {
      for (const it of items) {
        if (it.el.parentNode === menu) { it.ph.parentNode.insertBefore(it.el, it.ph.nextSibling); it.el.classList.remove('in-more-menu'); }
      }
      wrap.hidden = true;
    };

    let fitting = false, queued = false, mo = null;
    function fit() {
      if (fitting) return;
      fitting = true;
      try {
        const open = !menu.hidden;
        const folded = fitItems(items, overflows, fold, unfoldAll);
        btn.title = folded.length
          ? 'Schowane, bo okno jest za wąskie: ' + folded.map(it => it.el.dataset.foldName || it.id).join(', ')
          : '';
        if (!folded.length) menu.hidden = true; else menu.hidden = !open;
        btn.setAttribute('aria-expanded', String(!menu.hidden));
      } finally {
        if (mo) mo.takeRecords();   // our own moves are not content changes
        fitting = false;
      }
    }
    function schedule() {
      if (queued) return;
      queued = true;
      // not requestAnimationFrame: it is paused in hidden / embedded views, and then the bar would never refit
      setTimeout(() => { queued = false; fit(); }, 0);
    }

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      menu.hidden = !menu.hidden;
      btn.setAttribute('aria-expanded', String(!menu.hidden));
    });
    doc.addEventListener('mousedown', (e) => {
      if (!menu.hidden && !wrap.contains(e.target)) { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); }
    });
    doc.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !menu.hidden) { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); }
    });

    root.addEventListener('resize', schedule);
    // content of the groups changes (shot name, plate list, overlay layer) -> their width changes
    if (root.ResizeObserver) {
      const ro = new root.ResizeObserver(() => { if (!fitting) schedule(); });
      for (const g of header.children) ro.observe(g);
      for (const it of items) ro.observe(it.el);
    }
    // text / options filled in later (shot name, plate list) - also when rendering is paused and ResizeObserver is silent
    if (root.MutationObserver) {
      mo = new root.MutationObserver(() => { if (!fitting) schedule(); });
      mo.observe(header, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['style', 'class', 'hidden'] });
    }
    // web fonts and late content change the widths after the first fit
    root.addEventListener('load', schedule);
    if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(schedule);
    fit();
    return { fit, items };
  }

  const api = { FOLD_ORDER, fitItems, init };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root && root.document) {
    root.HeaderFit = api;
    const start = () => { api.instance = init(root.document); };
    if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', start);
    else start();
  }
})(typeof window !== 'undefined' ? window : globalThis);
