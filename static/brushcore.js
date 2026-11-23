// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/brushcore.js – the brush pieces shared by stage 2 (static/brush.js, „Poprawki maski”) and the Cleanup 3D area
// editor (static/cleanup3d_areas.js). Knows nothing about stages, objects or the server:
//   strokePath(ctx, st, k, drawVar)                  one stroke {p: [[x, y(, r)]…], radius_px} drawn with the caller's styles;
//                                                    drawVar = objects.js drawVarStroke for pen strokes (radius per point)
//   paintRegion(ctx, k, polys, strokes, drawVar)     a region as ONE flat shape: polygons + strokes − erase strokes
//   regionCanvas(W, H, k, polys, strokes, drawVar)   the same on a new W × H canvas (white on transparent)
//   StrokeInput                                      pointer capture of strokes on an element: size with [ ] and
//                                                    Shift + drag, the same rules as the stage brushes of objects.js
// Stage 2 takes its pointer events from objects.js (S.drawing / S.strokes, unchanged); StrokeInput serves the editors
// that live outside the object stages.
// Load before brush.js. window.MD_BRUSHCORE.
(function () {
  'use strict';
  const R_MIN = 1, R_MAX = 200;      // brush radius limits in image px (as objects.js)
  const clampR = (r) => Math.max(R_MIN, Math.min(R_MAX, Math.round(r || 1)));

  function strokePath(ctx, st, k, drawVar) {
    if (st.p.some((q) => q.length >= 3) && drawVar) { drawVar(ctx, st.p, k, st.radius_px); return; }
    ctx.lineWidth = 2 * (st.radius_px || 8) * k; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); st.p.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k)));
    if (st.p.length === 1) ctx.lineTo(st.p[0][0] * k + 0.01, st.p[0][1] * k);
    ctx.stroke();
  }

  // polygons + strokes − erase strokes in the caller's fillStyle / strokeStyle; leaves source-over behind
  function paintRegion(ctx, k, polys, strokes, drawVar) {
    for (const poly of polys || []) {
      if (!poly || poly.length < 3) continue;
      ctx.beginPath(); poly.forEach((q, i) => (i ? ctx.lineTo(q[0] * k, q[1] * k) : ctx.moveTo(q[0] * k, q[1] * k))); ctx.closePath(); ctx.fill();
    }
    for (const st of (strokes || []).filter((x) => !x.erase)) strokePath(ctx, st, k, drawVar);
    ctx.globalCompositeOperation = 'destination-out';
    for (const st of (strokes || []).filter((x) => x.erase)) strokePath(ctx, st, k, drawVar);
    ctx.globalCompositeOperation = 'source-over';
  }

  function regionCanvas(W, H, k, polys, strokes, drawVar) {
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d');
    g.fillStyle = g.strokeStyle = '#fff';
    paintRegion(g, k, polys, strokes, drawVar);
    return c;
  }

  // Strokes drawn with the pointer on `el`.
  //   o.el                element that receives the pointer events
  //   o.toImage(e)        event -> [x, y] in image px (null = outside / image not ready)
  //   o.enabled()         gate: false = the events are left alone
  //   o.canStart()        asked on the press: false = no stroke (the host says why)
  //   o.mode()            'add' | 'remove' | 'erase' – stored in the stroke
  //   o.onStroke(st)      a finished stroke {mode, radius_px, p}
  //   o.onChange()        redraw request (stroke in progress, hover, size)
  //   o.radius / o.onRadius(r)   start size in image px / told after every size change
  // .radius, .drawing (stroke in progress or null), .hover ([x, y] | null), .resizing
  class StrokeInput {
    constructor(o) {
      this.o = Object.assign({ enabled: () => true, canStart: () => true, mode: () => 'add', onStroke: () => {}, onChange: () => {}, onRadius: () => {} }, o);
      this.radius = clampR(this.o.radius || 24);
      this.drawing = null; this.hover = null; this.resizing = null;
      const el = this.o.el;
      this._down = (e) => this.down(e); this._move = (e) => this.move(e); this._up = (e) => this.up(e);
      this._leave = () => { if (this.hover) { this.hover = null; this.o.onChange(); } };
      this._key = (e) => this.key(e);
      el.addEventListener('pointerdown', this._down);
      el.addEventListener('pointermove', this._move);
      el.addEventListener('pointerup', this._up);
      el.addEventListener('pointercancel', this._up);
      el.addEventListener('pointerleave', this._leave);
      window.addEventListener('keydown', this._key, true);   // before the stage shortcuts ([ ] of the stage brushes)
    }
    destroy() {
      const el = this.o.el;
      el.removeEventListener('pointerdown', this._down);
      el.removeEventListener('pointermove', this._move);
      el.removeEventListener('pointerup', this._up);
      el.removeEventListener('pointercancel', this._up);
      el.removeEventListener('pointerleave', this._leave);
      window.removeEventListener('keydown', this._key, true);
    }
    setRadius(r) { this.radius = clampR(r); this.o.onRadius(this.radius); this.o.onChange(); }
    down(e) {
      if (!this.o.enabled() || (window.mdAltHeld ? window.mdAltHeld(e) : e.altKey) || e.button !== 0) return;   // Alt = hand (pan), as in the stages
      const p = this.o.toImage(e); if (!p) return;
      e.preventDefault(); e.stopPropagation();
      if (!e.shiftKey && !this.o.canStart()) return;
      try { this.o.el.setPointerCapture(e.pointerId); } catch (er) { /* not capturable */ }
      if (e.shiftKey) { this.resizing = { x0: e.clientX, r0: this.radius, p, pid: e.pointerId }; this.o.onChange(); return; }
      this.drawing = { mode: this.o.mode(), radius_px: this.radius, p: [p], pid: e.pointerId };
      this.o.onChange();
    }
    move(e) {
      if (this.resizing && e.pointerId === this.resizing.pid) { this.setRadius(this.resizing.r0 * Math.exp((e.clientX - this.resizing.x0) / 120)); return; }
      if (this.drawing && e.pointerId === this.drawing.pid) {
        const evs = (e.getCoalescedEvents && e.getCoalescedEvents().length) ? e.getCoalescedEvents() : [e];
        for (const ev of evs) { const p = this.o.toImage(ev); if (p) this.drawing.p.push(p); }
        this.hover = this.o.toImage(e);
        this.o.onChange(); return;
      }
      if (this.o.enabled()) { this.hover = this.o.toImage(e); this.o.onChange(); }
    }
    up(e) {
      if (this.resizing && e.pointerId === this.resizing.pid) { this.resizing = null; this.o.onChange(); return; }
      if (this.drawing && e.pointerId === this.drawing.pid) {
        const d = this.drawing; this.drawing = null;
        this.o.onStroke({ mode: d.mode, radius_px: d.radius_px, p: d.p });
        this.o.onChange();
      }
    }
    key(e) {
      if (!this.o.enabled()) return;
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.code === 'BracketLeft') { e.preventDefault(); e.stopPropagation(); this.setRadius(this.radius / 1.25); }
      else if (e.code === 'BracketRight') { e.preventDefault(); e.stopPropagation(); this.setRadius(this.radius * 1.25 + 1); }
    }
  }

  const api = { strokePath, paintRegion, regionCanvas, StrokeInput, clampR, R_MIN, R_MAX };
  if (typeof window !== 'undefined') window.MD_BRUSHCORE = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
