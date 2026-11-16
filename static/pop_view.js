// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// ONE big image preview popup (user 2026-11-11: „the frame mask with a preview like the anchors”): the look and the
// handling of the anchor popup (static/cleanup_anchors.js, the same CSS classes cu-anc-pop / cu-pop-*) – dragged by its
// head, ✕ / Esc close, wheel = zoom at the cursor, drag = pan, double click = the whole image. Any panel opens it with
//   MD_POPVIEW.open({ title, src })   // src = an image URL or a data: URL (a canvas: canvas.toDataURL())
(function () {
  'use strict';
  const P = { z: 1, x: 0, y: 0, pos: null, open: false };
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function el() {
    let p = $('mdPopView'); if (p) return p;
    document.body.insertAdjacentHTML('beforeend', `<div class="cu-anc-pop md-popview" id="mdPopView" style="display:none" role="dialog" aria-label="Podgląd">
      <div class="cu-pop-head"><span class="ol-grip">⠿</span><span id="mdPopTitle"></span><button type="button" class="ol-fold cu-pop-x" id="mdPopClose" title="Zamknij (Esc)">✕</button></div>
      <div class="cu-pop-view" id="mdPopViewArea" title="Kółko = zoom, przeciągnij = przesuń, dwuklik = cały obraz"><img id="mdPopImg" alt="" draggable="false"></div>
      <div class="muted pp-hint" id="mdPopNote"></div></div>`);
    p = $('mdPopView');
    $('mdPopClose').addEventListener('click', close);
    p.querySelector('.cu-pop-head').addEventListener('mousedown', (e) => {
      if (e.button !== 0 || e.target.closest('button')) return;
      e.preventDefault();
      const r = p.getBoundingClientRect(); const dx = e.clientX - r.left, dy = e.clientY - r.top;
      const move = (ev) => {
        P.pos = { x: Math.max(0, Math.min(window.innerWidth - 80, ev.clientX - dx)), y: Math.max(0, Math.min(window.innerHeight - 40, ev.clientY - dy)) };
        p.style.left = `${P.pos.x}px`; p.style.top = `${P.pos.y}px`;
      };
      const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });
    const view = $('mdPopViewArea');
    view.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = view.getBoundingClientRect(); const cx = e.clientX - r.left, cy = e.clientY - r.top;
      const z1 = Math.max(0.25, Math.min(16, P.z * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
      P.x = cx - (cx - P.x) * (z1 / P.z); P.y = cy - (cy - P.y) * (z1 / P.z); P.z = z1; xf();
    }, { passive: false });
    view.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const pan = { x: e.clientX - P.x, y: e.clientY - P.y };
      const move = (ev) => { P.x = ev.clientX - pan.x; P.y = ev.clientY - pan.y; xf(); };
      const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });
    view.addEventListener('dblclick', fit);
    $('mdPopImg').addEventListener('load', fit);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && P.open) { e.stopPropagation(); close(); } }, true);
    for (const ev of ['wheel', 'mousedown', 'dblclick']) p.addEventListener(ev, (e) => e.stopPropagation());
    return p;
  }
  function xf() { const im = $('mdPopImg'); if (im) im.style.transform = `translate(${P.x}px, ${P.y}px) scale(${P.z})`; }
  function fit() {
    const im = $('mdPopImg'); const view = $('mdPopViewArea'); if (!im || !view || !im.naturalWidth) return;
    const vw = view.clientWidth || 600, vh = view.clientHeight || 400;
    P.z = Math.min(vw / im.naturalWidth, vh / im.naturalHeight);
    P.x = (vw - im.naturalWidth * P.z) / 2; P.y = (vh - im.naturalHeight * P.z) / 2; xf();
  }
  function open(o) {
    const p = el();
    $('mdPopTitle').textContent = (o && o.title) || '';
    $('mdPopNote').innerHTML = o && o.note ? esc(o.note) : '';
    p.style.display = ''; P.open = true;
    if (P.pos) { p.style.left = `${P.pos.x}px`; p.style.top = `${P.pos.y}px`; }
    $('mdPopImg').src = (o && o.src) || '';
  }
  function close() { P.open = false; const p = $('mdPopView'); if (p) p.style.display = 'none'; }
  window.MD_POPVIEW = { open, close, get isOpen() { return P.open; } };
})();
