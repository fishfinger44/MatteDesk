// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/i18n.js - UI language. Polish is the source language of the code; English (default) is applied at run time
// from the catalog static/i18n/en.json: text nodes, tooltips / placeholders / data-ask, alert / confirm / prompt,
// canvas labels and server messages shown in the page. Exact matches first, then patterns with placeholders {0} {1}
// (captured parts are translated too), then a substring pass for texts glued together from several messages.
// New UI strings: `python tools/i18n_extract.py` lists the ones missing from the catalog.
// Load FIRST (before app.js). window.mdT(s) = translate one string (identity in Polish).
(function () {
  const KEY = 'md_lang';
  let lang = 'en';
  try { lang = localStorage.getItem(KEY) || 'en'; } catch (e) { /* storage blocked: default */ }
  if (lang !== 'pl') lang = 'en';
  window.mdLang = lang;
  window.mdT = (s) => s;
  window.mdSetLang = (l) => {
    try { localStorage.setItem(KEY, l); } catch (e) { /* storage blocked */ }
    location.reload();
  };
  document.documentElement.lang = lang;
  // header switch: shows the current language, click = the other one (reload)
  const wire = () => {
    const b = document.getElementById('langToggle');
    if (!b) return;
    b.textContent = lang === 'pl' ? '🌐 PL' : '🌐 EN';
    b.addEventListener('click', () => window.mdSetLang(lang === 'pl' ? 'en' : 'pl'));
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire); else wire();
  if (lang === 'pl') return;

  const ATTRS = ['title', 'placeholder', 'aria-label', 'alt', 'data-fold-name', 'data-ask', 'data-tip', 'label'];   // label: <optgroup>
  // looks Polish: diacritics, common words, typical letter groups / endings (same idea as tools/i18n_extract.py)
  const PL = /[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ„”]|\b(Zapisz|Anuluj|Zamknij|Projekt|Obiekt|Klatka|klatka|klatki|Etap|etap|Brak|brak|Wybierz|Dodaj|Policz|Przelicz|Kliknij|Trwa|nie|jest|albo|tylko|przez|dla|bez|potem|teraz|maska|strefa|klucz|Tak|Nie|w|z|na|lub|gdy|czy|jak|razem)\b|[a-z]*(sz|cz|rz)[a-z]*|\b\w+(anie|enie|ania|enia|acja|acji|cja|cji|ować|ują|uje|ych|ymi|owy|owa|owe|ego|iej|ami|ach)\b/
  let exact = new Map();
  let patterns = [];        // {re (anchored), reG (unanchored, global), en}
  let subRe = null;          // alternation of exact sources for the substring pass
  const memo = new Map();
  const MAX_TR = 1000;       // pattern / substring passes only up to this length (longest pattern source ~500 chars)
  let ready = false;

  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const fill = (en, groups) => en.replace(/\{(\w+)\}/g, (m, i) => (groups[i] !== undefined ? tr(groups[i]) : m));   // {0} or named {zone_id}

  function build(cat) {
    const strings = cat.strings || {};
    const subs = [];
    for (let [pl, en] of Object.entries(strings)) {
      if (!en || en === pl) continue;
      // adjacent placeholders ("{0}{1}") cannot be split reliably: when the translation keeps them together in the same
      // order, match them as ONE part, translated as a whole (e.g. install text + licence note)
      (pl.match(/(?:\{\w+\}){2,}/g) || []).forEach((run, i) => {
        if (en.includes(run)) { const tok = `{${900 + i}}`; pl = pl.split(run).join(tok); en = en.split(run).join(tok); }
      });
      if (/\{\w+\}/.test(pl)) {
        const raw = pl.split(/\{\w+\}/);
        const order = [...pl.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
        if (!raw.join('').replace(/\W/g, '')) continue;
        // literal parts: spaces next to a placeholder are optional (text nodes are trimmed), other spaces flexible
        const lit = raw.map((p, i) => {
          let s = p;
          if (i > 0) s = s.replace(/^\s+/, '');                 // after a placeholder
          if (i < raw.length - 1) s = s.replace(/\s+$/, '');    // before a placeholder
          return esc(s).replace(/\s+/g, '\\s+');
        });
        // inside a longer text (unanchored pass) the source spacing is kept as is, so "klucz {0}" never eats "klucza"
        const kept = raw.map((p) => esc(p).replace(/\s+/g, '\\s+'));
        const body = (edge) => (edge ? kept : lit).reduce((acc, p, i, arr) => {
          if (i === order.length) return acc + p;
          const atEdge = (i === 0 && !arr[0]) || (i === order.length - 1 && !arr[arr.length - 1]);
          const sp = edge ? '' : '\\s*';
          return acc + p + sp + (atEdge ? (edge ? '(\\S+?)' : '(.*?)') : '(.+?)') + sp;
        }, '');
        patterns.push({ re: new RegExp('^' + body(false) + '$', 's'), reG: new RegExp('(?<![\\p{L}\\p{N}])' + body(true) + '(?![\\p{L}\\p{N}])', 'gsu'), en, order,
          len: raw.join('').length });
      } else {
        exact.set(pl, en);
        // substring pass only for phrases: single short words glued into other sentences would give mixed language
        if (pl.length >= 6 && /\s/.test(pl.trim()) || pl.length >= 10) subs.push(pl);
      }
    }
    patterns.sort((a, b) => b.len - a.len);
    subs.sort((a, b) => b.length - a.length);
    if (subs.length) subRe = new RegExp('(?<![\\p{L}\\p{N}])(?:' + subs.map(esc).join('|') + ')(?![\\p{L}\\p{N}])', 'gu');
  }

  function byOrder(m, order) {   // regex groups come in text order; map them back to {n}
    const g = {};
    order.forEach((n, i) => { g[n] = m[i + 1]; });
    return g;
  }

  function tr(s) {
    if (!ready || typeof s !== 'string' || s.length < 2 || !/[A-Za-zÀ-ž]{2}/.test(s)) return s;
    const hit = memo.get(s);
    if (hit !== undefined) return hit;
    const lead = s.match(/^\s*/)[0];
    const trail = s.length > lead.length ? s.match(/\s*$/)[0] : '';
    const core = s.trim().replace(/\s+/g, ' ');
    let out = exact.get(core);
    // MAX_TR: a long text (report preview with the log) through thousands of catalog regexes froze the page for minutes
    // (user report 2026-10-23, ~7 s already for 4000 chars) -> long texts get the exact lookup only
    if (out === undefined && core.length > MAX_TR) return s;
    if (out === undefined) {
      for (const p of patterns) {
        const m = core.match(p.re);
        if (m) { out = fill(p.en, byOrder(m, p.order)).trim(); break; }
      }
    }
    if (out === undefined) {
      // whole known sentences first (exact catalog entries), then patterns for what is left; no "looks Polish" gate:
      // only known Polish sources match, and Polish without diacritics ("zapis klatek") slipped through it
      out = core;
      if (subRe) out = out.replace(subRe, (m) => exact.get(m) || m);
      for (const p of patterns) if (p.reG.test(out)) { p.reG.lastIndex = 0; out = out.replace(p.reG, (...m) => fill(p.en, byOrder(m, p.order))); } else p.reG.lastIndex = 0;
      if (out === core) out = undefined;
    }
    let res = out === undefined ? s : lead + out + trail;
    res = res.replace(/(\d)(\s*)kl\./g, '$1$2fr.').replace(/(\d)(\s*)odc\./g, '$1$2seg.').replace(/s\/kl\./g, 's/fr.').replace(/\(rdzeń\)/g, '(core)');   // "27 kl." = frames, "1 odc." = segments
    if (memo.size > 20000) memo.clear();
    memo.set(s, res);
    return res;
  }

  function trNode(node) {
    if (node.nodeType === 3) {
      const p = node.parentNode;
      if (p && (p.nodeName === 'SCRIPT' || p.nodeName === 'STYLE' || p.nodeName === 'TEXTAREA' || (p.closest && p.closest('[translate="no"]')))) return;
      const t = tr(node.data);
      if (t !== node.data) node.data = t;
      return;
    }
    if (node.nodeType !== 1) return;
    for (const a of ATTRS) {
      const v = node.getAttribute(a);
      if (v) { const t = tr(v); if (t !== v) node.setAttribute(a, t); }
    }
    if (node.nodeName === 'INPUT' && (node.type === 'button' || node.type === 'submit') && node.value) {
      const t = tr(node.value); if (t !== node.value) node.value = t;
    }
  }

  function trTree(root) {
    trNode(root);
    if (root.nodeType !== 1 && root.nodeType !== 9 && root.nodeType !== 11) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) trNode(n);
  }

  // dialogs and canvas labels
  for (const f of ['alert', 'confirm', 'prompt']) {
    const orig = window[f].bind(window);
    window[f] = (msg, ...rest) => orig(tr(msg), ...rest);
  }
  const fillText = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (text, ...rest) { return fillText.call(this, tr(String(text)), ...rest); };

  // hide the page until the catalog is in (no Polish flash); never longer than 3 s
  document.documentElement.style.visibility = 'hidden';
  const show = () => { document.documentElement.style.visibility = ''; };
  setTimeout(show, 3000);

  fetch('/static/i18n/en.json', { cache: 'no-cache' }).then((r) => r.json()).then((cat) => {
    build(cat);
    ready = true;
    window.mdT = tr;
    const start = () => {
      trTree(document.head); trTree(document.body); show();
      new MutationObserver((recs) => {
        for (const r of recs) {
          if (r.type === 'characterData') trNode(r.target);
          else if (r.type === 'attributes') trNode(r.target);
          else for (const n of r.addedNodes) trTree(n);
        }
      }).observe(document.documentElement, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS.concat(['value']) });
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
  }).catch(show);
})();
