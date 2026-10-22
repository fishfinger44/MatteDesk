// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Object life range (zakres życia obiektu, user request 2026-10-15): an object can leave the frame and come back.
// Visible intervals of the selected object on its keys lane: outside them a dark band (alpha 0, nothing computed – SAM3 /
// MatAnyone never see those frames, the export writes black frames). Handles [ ] drag the ends of an interval
// (dwuklik = całe ujęcie); buttons in stage 1 SAM3: „⟦ Widoczny od tej klatki”, „⟧ Widoczny do tej klatki”, „↺ całe ujęcie”.
// A frame in a gap + „od tej klatki” = the object comes back (a new interval). Hint from the last core: the mask gets
// empty right after touching the frame border -> „obiekt znika od F…, ustawić koniec zakresu?”.
// Stored in objects/<name>/object.json -> life (PUT /api/object/{name}/life); with the work area the intersection counts.
(function () {
  'use strict';
  const V = () => window.viewer;
  const L = { obj: null, life: null, shot: null, hints: [], hidden: [], drag: null, layer: null, asked: {} };
  const eff = () => L.life || (L.shot ? [L.shot.slice()] : []);
  window.MD_LIFE = {
    get life() { return L.life; },
    visible(f) { return !L.life || L.life.some(([a, b]) => a <= f && f <= b); },
    label() { return L.life ? L.life.map(([a, b]) => `F${a}–${b}`).join(', ') : 'całe ujęcie'; },
    set: (life) => save(life),
  };
  const fmt = (ivs) => ivs.map(([a, b]) => (a === b ? `F${a}` : `F${a}–${b}`)).join(', ');

  function norm(ivs) {   // sorted, merged, clipped to the shot
    const s = L.shot; if (!s) return ivs;
    const xs = ivs.map(([a, b]) => [Math.max(s[0], Math.min(a, b)), Math.min(s[1], Math.max(a, b))]).filter(([a, b]) => a <= b)
      .sort((x, y) => x[0] - y[0]);
    const out = [];
    for (const [a, b] of xs) { if (out.length && a <= out[out.length - 1][1] + 1) out[out.length - 1][1] = Math.max(out[out.length - 1][1], b); else out.push([a, b]); }
    return out;
  }
  // „⟦ od tej klatki”: inside an interval -> its start = f; in a gap -> the object comes back: [f, the next start - 1 /
  // the shot end] (merges with the next interval)
  function startAt(f) {
    const iv = eff().map((x) => x.slice());
    const cur = iv.find(([a, b]) => a <= f && f <= b);
    if (cur) cur[0] = f;
    else { const nx = iv.find(([a]) => a > f); iv.push([f, nx ? nx[0] - 1 : L.shot[1]]); }
    return norm(iv);
  }
  // „⟧ do tej klatki”: inside an interval -> its end = f; in a gap -> the previous interval ends at f
  function endAt(f) {
    const iv = eff().map((x) => x.slice());
    const cur = iv.find(([a, b]) => a <= f && f <= b);
    if (cur) cur[1] = f;
    else { const pv = iv.filter(([, b]) => b < f).pop(); if (pv) pv[1] = f; else iv.push([L.shot[0], f]); }
    return norm(iv);
  }

  function ensureLayer() {
    const keys = document.getElementById('timelineKeys');
    if (!keys) return null;
    if (!L.layer) {
      L.layer = document.createElement('div');
      L.layer.className = 'timeline-life'; L.layer.id = 'timelineLife';
      keys.parentNode.insertBefore(L.layer, keys);
      window.addEventListener('mousemove', (e) => {
        if (!L.drag) return;
        const v = V(); const rect = L.layer.getBoundingClientRect();
        const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
        const f = Math.max(v.firstFrame, Math.min(v.lastFrame, v.firstFrame + Math.floor(pos * n + (L.drag.side ? -0.5 : 0.5))));
        const d = L.drag; const iv = d.iv;
        const lo = d.i > 0 ? iv[d.i - 1][1] + 2 : L.shot[0];          // never merge into the neighbour while dragging
        const hi = d.i < iv.length - 1 ? iv[d.i + 1][0] - 2 : L.shot[1];
        if (d.side === 0) iv[d.i][0] = Math.max(lo, Math.min(f, iv[d.i][1]));
        else iv[d.i][1] = Math.min(hi, Math.max(f, iv[d.i][0]));
        d.moved = true; render(iv);
      });
      window.addEventListener('mouseup', () => {
        if (!L.drag) return;
        const d = L.drag; L.drag = null;
        if (d.moved) save(norm(d.iv)); else render();
      });
    }
    return L.layer;
  }

  function render(iv) {
    const layer = ensureLayer(); const v = V();
    if (!layer || !v) return;
    layer.innerHTML = '';
    layer.style.display = L.obj && L.shot ? '' : 'none';
    renderRows();
    if (!L.obj || !L.shot) return;
    iv = iv || eff();
    const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
    const pc = (f) => 100 * (f - v.firstFrame) / n;
    const whole = !L.drag && !L.life;
    const tip = 'Zakres życia obiektu: poza nim maska = pusta (alpha 0), SAM 3.1 / MatAnyone2 tych klatek nie liczą, eksport zapisuje czarne klatki. ' +
      'Przeciągnij [ / ] (dwuklik = całe ujęcie); przyciski „⟦ od tej klatki” / „⟧ do tej klatki” w panelu etapu';
    const mk = (cls, style, title) => { const d = document.createElement('div'); d.className = cls; d.style.cssText = style; if (title) d.title = title; layer.appendChild(d); return d; };
    let prev = v.firstFrame;
    iv.forEach(([a, b]) => { if (a > prev) mk('life-dim', `left:${pc(prev)}%;width:${pc(a) - pc(prev)}%`, `Obiekt niewidoczny F${prev}–${a - 1} (poza zakresem życia)`); prev = b + 1; });
    if (prev <= v.lastFrame) mk('life-dim', `left:${pc(prev)}%;right:0`, `Obiekt niewidoczny F${prev}–${v.lastFrame} (poza zakresem życia)`);
    iv.forEach(([a, b], i) => {
      for (const side of [0, 1]) {
        const h = mk(`life-handle ${side ? 'life-right' : 'life-left'}${whole ? ' whole' : ''}`, `left:${side ? pc(b + 1) : pc(a)}%`,
          `${side ? 'Koniec' : 'Początek'} widoczności obiektu (F${side ? b : a}) – ${tip}`);
        h.addEventListener('mousedown', (e) => {
          if (e.button !== 0) return;
          e.stopPropagation(); e.preventDefault();
          L.drag = { i, side, iv: iv.map((x) => x.slice()), moved: false };
        });
        h.addEventListener('dblclick', (e) => { e.stopPropagation(); e.preventDefault(); save(null); });
      }
    });
  }

  function rowHtml() {
    if (!L.obj || !L.shot) return '';
    const v = V(); const f = v ? v.currentFrame : null;
    const vis = window.MD_LIFE.visible(f);
    const life = L.life ? `<b class="life-on">${fmt(L.life)}</b>` : 'całe ujęcie';
    const hid = L.hidden.length ? ` · <span class="warn" title="Klucze poza zakresem życia nie są używane w liczeniu">klucze poza: ${L.hidden.join(', ')}</span>` : '';
    const hints = L.hints.map((h, i) => `<div class="life-hint">💡 ${esc(hintText(h))} <button class="btn btn-secondary btn-sm" data-hint="${i}">Ustaw</button></div>`).join('');
    return `<div class="sec-title">Zakres życia obiektu</div>
      <div class="muted">Widoczny: ${life}${f != null && !vis ? ` · <span class="warn">F${f}: poza zakresem (alpha 0)</span>` : ''}${hid}</div>
      <div class="row"><button class="btn btn-ghost btn-sm" data-life="start" title="Obiekt widoczny od bieżącej klatki (w przerwie = obiekt wraca: nowy przedział)">⟦ od tej klatki</button><button class="btn btn-ghost btn-sm" data-life="end" title="Obiekt widoczny do bieżącej klatki (dalej wychodzi z kadru / znika)">⟧ do tej klatki</button><button class="btn btn-ghost btn-sm" data-life="all" title="Obiekt w całym ujęciu" ${L.life ? '' : 'disabled'}>↺ całe ujęcie</button></div>${hints}`;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
  function hintText(h) {
    const e = h.empty[0] === h.empty[1] ? `F${h.empty[0]}` : `F${h.empty[0]}–${h.empty[1]}`;
    return h.kind === 'end' ? `Obiekt znika od F${h.frame} (ostatnia maska przy krawędzi kadru w F${h.last_visible}) – ustawić koniec zakresu na F${h.last_visible} (ukryć ${e})?`
      : `Obiekt pojawia się przy krawędzi kadru w F${h.frame} (wcześniej pusto: ${e}) – ustawić początek zakresu na F${h.frame}?`;
  }
  // a hint hides exactly the empty frames (exit: the end of the interval; entry: its start; both when it comes back)
  function applyHint(h) {
    const [x, y] = h.empty;
    return norm(eff().flatMap(([a, b]) => (b < x || a > y ? [[a, b]] : [[a, x - 1], [y + 1, b]].filter(([p, q]) => p <= q))));
  }
  function renderRows() {
    // in the floating „Punkty SAM3” palette on the left (user request 2026-10-16: next to where the object is found);
    // without the palette (older layout) after the stage-1 keys list as before
    const pal = document.getElementById('pointsPalette');
    for (const anchor of [pal ? 'pointsPalette' : 'st1Keys']) {
      const a = document.getElementById(anchor); if (!a) continue;
      const host = pal || a.parentNode;
      let box = host.querySelector(`.life-row[data-for="${anchor}"]`);
      if (!box) {
        box = document.createElement('div'); box.className = pal ? 'life-row pp-life' : 'life-row pane-section'; box.dataset.for = anchor;
        if (pal) pal.appendChild(box); else a.parentNode.insertBefore(box, a.nextSibling);
      }
      box.innerHTML = rowHtml();
      box.querySelectorAll('[data-life]').forEach((b) => b.addEventListener('click', () => {
        const f = V().currentFrame;
        save(b.dataset.life === 'all' ? null : b.dataset.life === 'start' ? startAt(f) : endAt(f));
      }));
      box.querySelectorAll('[data-hint]').forEach((b) => b.addEventListener('click', () => save(applyHint(L.hints[+b.dataset.hint]))));
    }
  }

  async function load(name) {
    L.obj = name || null;
    if (!L.obj) { L.life = null; L.hints = []; L.hidden = []; render(); return; }
    try {
      const r = await fetch(`/api/object/${encodeURIComponent(L.obj)}/life`);
      if (!r.ok) { L.life = null; L.hints = []; render(); return; }
      const d = await r.json();
      if (name !== L.obj) return;
      L.life = d.life; L.shot = d.shot; L.hints = d.hints || []; L.hidden = d.hidden_keys || [];
    } catch (e) { return; }
    render();
    ask();
  }
  // one dialog per hint of the core (object + kind + frame; remembered in this browser), afterwards only in the panel
  const ASKED = 'md_life_asked';
  function wasAsked(sig) {
    if (L.asked[sig]) return true;
    try { return (JSON.parse(localStorage.getItem(ASKED) || '[]')).includes(sig); } catch (e) { return false; }
  }
  function markAsked(sig) {
    L.asked[sig] = true;
    try { const a = JSON.parse(localStorage.getItem(ASKED) || '[]'); a.push(sig); localStorage.setItem(ASKED, JSON.stringify(a.slice(-200))); } catch (e) { /* no storage */ }
  }
  async function ask() {
    const v = V(); if (!v || !v.uiConfirm || !L.hints.length || document.querySelector('.md-dialog')) return;
    const h = L.hints[0]; const sig = `${(v.projectData && v.projectData.manifest_path) || ''}:${L.obj}:${h.kind}:${h.frame}`;
    if (wasAsked(sig)) return;
    markAsked(sig);
    const ok = await v.uiConfirm(`${hintText(h)}\n\nPoza zakresem życia maska jest pusta (alpha 0), SAM 3.1 / MatAnyone2 tych klatek nie liczą, eksport zapisuje czarne klatki.`,
      { title: `„${L.obj}”: obiekt wychodzi z kadru`, ok: 'Ustaw zakres', cancel: 'Nie teraz' }).catch(() => false);
    if (ok) save(applyHint(h));
  }

  async function save(life) {
    if (!L.obj) return;
    const name = L.obj;
    try {
      const res = await fetch(`/api/object/${encodeURIComponent(name)}/life`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ life }) });
      const d = await res.json();
      if (!res.ok) { V() && V().showToast(`Zakres życia: ${d.detail || res.statusText}`, 4000); render(); return; }
      L.life = d.life; L.hints = d.hints || []; L.hidden = d.hidden_keys || [];
    } catch (e) { render(); return; }
    render();
    V() && V().showToast(L.life ? `„${name}” widoczny: ${fmt(L.life)} (poza – alpha 0, bez liczenia)` : `„${name}”: całe ujęcie`, 3500);
    if (window.MD_OBJ && window.MD_OBJ.name === name) window.MD_OBJ.reload();
  }

  window.addEventListener('md-object-stages', (e) => load(e.detail && e.detail.name));
  window.addEventListener('md-frame', () => renderRows());
  window.addEventListener('md-project', () => { L.obj = null; L.shot = null; L.asked = {}; render(); });
  window.addEventListener('resize', () => render());
})();
