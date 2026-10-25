// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// Work area on the timeline (zakres pracy, user request 2026-10-15) – like the AE / Nuke work area.
// Two brackets [ ] on the timeline track, dragged with the mouse; Shift+B / Shift+N = start / end at the current frame,
// Alt+X = the whole shot. Stored per shot in project.json -> work_area (PUT /api/project/work_area).
// „▶▶ Przelicz nieaktualne”, „Przelicz finalnie” and the export "work area" compute only these frames.
(function () {
  'use strict';
  const V = () => window.viewer;
  const track = () => document.getElementById('timelineTrack');
  const W = { range: null, shot: null, drag: null, els: null };
  window.MD_WA = {
    get range() { return W.range; },                      // [a, b] or null (= whole shot)
    get effective() { return W.range || W.shot; },
    label() { const e = W.range || W.shot; return e ? `F${e[0]}–${e[1]}` : ''; },
    set: (r) => save(r),
  };

  function build() {
    const track = document.getElementById('timelineTrack');
    if (!track || W.els) return;
    const mk = (cls, title) => { const d = document.createElement('div'); d.className = cls; if (title) d.title = title; track.appendChild(d); return d; };
    const tip = 'Zakres pracy (work area): przeciągnij [ / ] – „▶▶ Przelicz nieaktualne”, „Przelicz finalnie” i eksport „work area” liczą tylko te klatki. ' +
      'Shift+B = początek w bieżącej klatce, Shift+N = koniec, Alt+X = całe ujęcie; dwuklik na znaczniku = całe ujęcie';
    W.els = {
      dimL: mk('wa-dim'), dimR: mk('wa-dim'),
      band: mk('wa-band', tip),
      hL: mk('wa-handle wa-left', `Początek zakresu pracy – ${tip}`),
      hR: mk('wa-handle wa-right', `Koniec zakresu pracy – ${tip}`),
    };
    // frame number next to each bracket (user request 2026-10-16), inside the range so the track does not clip it
    for (const h of [W.els.hL, W.els.hR]) { const s = document.createElement('span'); s.className = 'wa-num'; h.appendChild(s); }
    const start = (side) => (e) => {
      if (e.button !== 0) return;
      e.stopPropagation(); e.preventDefault();
      W.drag = { side, r: (W.range || W.shot).slice() };
    };
    W.els.hL.addEventListener('mousedown', start(0));
    W.els.hR.addEventListener('mousedown', start(1));
    for (const h of [W.els.hL, W.els.hR]) h.addEventListener('dblclick', (e) => { e.stopPropagation(); save(null); });
    window.addEventListener('mousemove', (e) => {
      if (!W.drag) return;
      const v = V(); const rect = track.getBoundingClientRect();
      const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const f = Math.round(v.firstFrame + pos * (v.lastFrame - v.firstFrame));
      const r = W.drag.r;
      if (W.drag.side === 0) r[0] = Math.min(f, r[1]); else r[1] = Math.max(f, r[0]);
      render(r);
    });
    window.addEventListener('mouseup', () => {
      if (!W.drag) return;
      const r = W.drag.r; W.drag = null;
      save(r);
    });
  }

  function render(r) {
    const v = V(); if (!W.els || !v || !W.shot) return;
    r = r || W.range || W.shot;
    const n = Math.max(1, v.lastFrame - v.firstFrame + 1);
    const l = 100 * (r[0] - v.firstFrame) / n, rr = 100 * (r[1] - v.firstFrame + 1) / n;
    const whole = !W.drag && !W.range;
    W.els.dimL.style.cssText = `left:0;width:${l}%`;
    W.els.dimR.style.cssText = `left:${rr}%;right:0`;
    W.els.band.style.cssText = `left:${l}%;width:${rr - l}%`;
    W.els.band.classList.toggle('whole', whole);
    W.els.hL.style.left = `${l}%`; W.els.hR.style.left = `${rr}%`;
    for (const h of [W.els.hL, W.els.hR]) h.classList.toggle('whole', whole);
    W.els.hL.firstChild.textContent = r[0]; W.els.hR.firstChild.textContent = r[1];
    // narrow range: the two numbers would overlap -> keep them, but outside the brackets
    const narrow = (rr - l) / 100 * track().clientWidth < 64;
    W.els.hL.classList.toggle('wa-narrow', narrow); W.els.hR.classList.toggle('wa-narrow', narrow);
    const lbl = document.getElementById('waLabel');
    if (lbl) {
      lbl.textContent = whole ? 'Zakres: całe ujęcie' : `Zakres: F${r[0]}–${r[1]} (${r[1] - r[0] + 1} kl.)`;
      lbl.classList.toggle('active', !whole);
    }
  }

  async function load() {
    try {
      const r = await fetch('/api/project/work_area');
      if (!r.ok) return;
      const d = await r.json();
      W.range = d.range; W.shot = d.shot;
    } catch (e) { return; }
    build(); render();
    window.dispatchEvent(new CustomEvent('md-work-area', { detail: { range: W.range, shot: W.shot, loaded: true } }));
  }

  async function save(r) {
    if (!W.shot) return;
    try {
      const res = await fetch('/api/project/work_area', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ range: r }) });
      const d = await res.json();
      if (!res.ok) { V() && V().showToast(`Zakres pracy: ${d.detail || res.statusText}`, 4000); render(); return; }
      W.range = d.range; W.shot = d.shot;
    } catch (e) { render(); return; }
    render();
    V() && V().showToast(W.range ? `Zakres pracy F${W.range[0]}–${W.range[1]} (liczenie i eksport „work area” tylko w nim)` : 'Zakres pracy: całe ujęcie', 3000);
    window.dispatchEvent(new CustomEvent('md-work-area', { detail: { range: W.range, shot: W.shot } }));
  }

  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || !W.shot) return;
    const v = V(); if (!v || !v.projectLoaded) return;
    const cur = (W.range || W.shot).slice(), f = v.currentFrame;
    if (e.code === 'KeyB' && e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) { e.preventDefault(); save([f, Math.max(f, cur[1])]); }
    else if (e.code === 'KeyN' && e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) { e.preventDefault(); save([Math.min(f, cur[0]), f]); }
    else if (e.code === 'KeyX' && e.altKey && !e.ctrlKey && !e.metaKey) { e.preventDefault(); save(null); }
  });
  window.addEventListener('md-project', load);
  window.addEventListener('resize', () => render());
})();
