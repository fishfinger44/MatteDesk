/* MD_CURVES - ONE Bezier curve editor for every place that needs an editable curve (user decision 2026-11-09).
   Same JSON format as engine/curves.py:
     {type: "bezier", closed: false, pts: [{p: [x, y], in: [dx, dy], out: [dx, dy], cusp: false}, ...]}
   (image pixels; in / out = tangent handles relative to p).
   The editor knows nothing about where it is used: the host gives it an overlay canvas, the image <-> canvas mapping
   and a list of layers (curves) and gets onChange callbacks.

     const ed = new MD_CURVES.Editor({
       canvas,                                   // overlay <canvas> (receives the mouse events, drawn by draw())
       toScreen: p => [sx, sy],                  // image px -> canvas px
       toImage: (sx, sy) => [x, y],              // canvas px -> image px
       layers: () => [{id, spline, color, dash, editable, label}],   // live objects - the editor edits spline in place
       onChange: (id, spline, final) => {},      // final = mouse released / key action (save / re-render then)
       enabled: () => true,                      // optional gate
     });
     ed.draw(); ed.destroy(); ed.undo(); ed.redo(); ed.select(id, index);

   Mouse / keys (Nuke roto style):
     drag point                 move it (its handles go with it)
     Shift + drag point         soft move: neighbours follow with a smooth falloff (radius: Shift + wheel)
     drag handle                move the tangent; the opposite one mirrors its direction unless the point is a cusp
     Alt + drag handle          break the tangent (point becomes a cusp)
     Ctrl + click point         toggle cusp / smooth (smooth = handles made collinear)
     click on a curve           insert a point there (shape unchanged)
     Delete / Backspace         remove the selected point (a curve keeps >= 2)
     Ctrl+Z / Ctrl+Shift+Z      undo / redo (per editor)
*/
(function () {
  'use strict';
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const mul = (a, s) => [a[0] * s, a[1] * s];
  const len = a => Math.hypot(a[0], a[1]);
  const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const clone = o => JSON.parse(JSON.stringify(o));

  function point(p, tin, tout, cusp) {
    return { p: [p[0], p[1]], in: tin ? [tin[0], tin[1]] : [0, 0], out: tout ? [tout[0], tout[1]] : [0, 0], cusp: !!cusp };
  }
  function fromPoints(pts, closed) {          // Catmull-Rom through the points (same as engine/curves.from_points)
    const n = pts.length, out = [];
    for (let i = 0; i < n; i++) {
      let a, b;
      if (closed) { a = pts[(i - 1 + n) % n]; b = pts[(i + 1) % n]; }
      else {
        a = i > 0 ? pts[i - 1] : (n > 1 ? sub(mul(pts[0], 2), pts[1]) : pts[0]);
        b = i < n - 1 ? pts[i + 1] : (n > 1 ? sub(mul(pts[n - 1], 2), pts[n - 2]) : pts[n - 1]);
      }
      const t = mul(sub(b, a), 1 / 6);
      out.push(point(pts[i], mul(t, -1), t, false));
    }
    return { type: 'bezier', closed: !!closed, pts: out };
  }
  function normalize(o) { return (o && o.type === 'bezier') ? o : fromPoints(o || [], false); }
  function segments(sp) {
    const P = sp.pts, n = P.length, segs = [];
    const last = sp.closed && n > 2 ? n : n - 1;
    for (let i = 0; i < last; i++) {
      const a = P[i], b = P[(i + 1) % n];
      segs.push([a.p, add(a.p, a.out), add(b.p, b.in), b.p]);
    }
    return segs;
  }
  function bez(c, t) {
    const m = 1 - t;
    return [m * m * m * c[0][0] + 3 * m * m * t * c[1][0] + 3 * m * t * t * c[2][0] + t * t * t * c[3][0],
            m * m * m * c[0][1] + 3 * m * m * t * c[1][1] + 3 * m * t * t * c[2][1] + t * t * t * c[3][1]];
  }
  function sample(sp, step) {               // {xy, par} dense points (image px)
    sp = normalize(sp); step = step || 2;
    const segs = segments(sp), xy = [], par = [];
    segs.forEach((c, i) => {
      const L = len(sub(c[1], c[0])) + len(sub(c[2], c[1])) + len(sub(c[3], c[2]));
      const n = Math.max(2, Math.ceil(L / step));
      for (let k = 0; k < n; k++) { xy.push(bez(c, k / n)); par.push(i + k / n); }
    });
    if (segs.length) { xy.push(segs[segs.length - 1][3]); par.push(segs.length); }
    else sp.pts.forEach(q => { xy.push(q.p); par.push(0); });
    return { xy, par };
  }
  function split(sp, par) {                 // insert a point at segment parameter par, shape unchanged (de Casteljau)
    const segs = segments(sp), i = Math.min(Math.floor(par), segs.length - 1), t = par - i, c = segs[i];
    const a = lerp(c[0], c[1], t), b = lerp(c[1], c[2], t), cc = lerp(c[2], c[3], t);
    const d = lerp(a, b, t), e = lerp(b, cc, t), m = lerp(d, e, t);
    const n = sp.pts.length;
    sp.pts[i].out = sub(a, c[0]);
    sp.pts[(i + 1) % n].in = sub(cc, c[3]);
    sp.pts.splice(i + 1, 0, point(m, sub(d, m), sub(e, m), false));
    return i + 1;
  }
  function makeSmooth(q) {                  // collinear handles, keeping both lengths
    const li = len(q.in), lo = len(q.out);
    if (li < 1e-9 && lo < 1e-9) return;
    let dir = sub(q.out, q.in);
    const ld = len(dir);
    if (ld < 1e-9) return;
    dir = mul(dir, 1 / ld);
    q.in = mul(dir, -(li || lo)); q.out = mul(dir, lo || li);
  }

  class Editor {
    constructor(o) {
      this.o = Object.assign({ enabled: () => true, onChange: () => {}, tol: 9, handleR: 4.5 }, o);
      this.sel = null;                        // {id, i}
      this.drag = null;
      this.soft = 120;                        // soft-move radius, image px
      this.undoStack = []; this.redoStack = [];
      const c = this.o.canvas;
      this._md = e => this.mousedown(e); this._mm = e => this.mousemove(e); this._mu = e => this.mouseup(e);
      this._kd = e => this.keydown(e); this._wh = e => this.wheel(e);
      c.addEventListener('mousedown', this._md);
      window.addEventListener('mousemove', this._mm);
      window.addEventListener('mouseup', this._mu);
      window.addEventListener('keydown', this._kd);
      c.addEventListener('wheel', this._wh, { passive: false });
    }
    destroy() {
      const c = this.o.canvas;
      c.removeEventListener('mousedown', this._md);
      window.removeEventListener('mousemove', this._mm);
      window.removeEventListener('mouseup', this._mu);
      window.removeEventListener('keydown', this._kd);
      c.removeEventListener('wheel', this._wh);
    }
    layers() { return (this.o.layers() || []).filter(l => l && l.spline); }
    layer(id) { return this.layers().find(l => l.id === id); }
    select(id, i) { this.sel = id == null ? null : { id, i }; this.draw(); }
    evPos(e) { const r = this.o.canvas.getBoundingClientRect();
      const sx = (e.clientX - r.left) * this.o.canvas.width / r.width, sy = (e.clientY - r.top) * this.o.canvas.height / r.height;
      return { s: [sx, sy], img: this.o.toImage(sx, sy) }; }
    snapshot(id) { const l = this.layer(id); if (!l) return; this.undoStack.push({ id, sp: clone(l.spline) }); if (this.undoStack.length > 200) this.undoStack.shift(); this.redoStack = []; }
    restore(from, to) { const st = from.pop(); if (!st) return; const l = this.layer(st.id); if (!l) return;
      to.push({ id: st.id, sp: clone(l.spline) });
      l.spline.pts = st.sp.pts; l.spline.closed = st.sp.closed;
      if (this.sel && this.sel.id === st.id) this.sel.i = Math.min(this.sel.i, l.spline.pts.length - 1);
      this.o.onChange(st.id, l.spline, true); this.draw(); }
    undo() { this.restore(this.undoStack, this.redoStack); }
    redo() { this.restore(this.redoStack, this.undoStack); }
    changed(id, final) { const l = this.layer(id); if (l) this.o.onChange(id, l.spline, final); this.draw(); }

    hit(s) {
      // the NEAREST element wins (curves overlap when the view is zoomed out: src / dst a few px apart); handles and
      // points get a small head start (3 / 2 px), the selected curve 1 px
      const tol = this.o.tol, ts = this.o.toScreen;
      let best = null, bestD = tol;
      const take = (d, h) => { if (d < bestD) { bestD = d; best = h; } };
      if (this.sel) {
        const l = this.layer(this.sel.id), q = l && l.spline.pts[this.sel.i];
        if (q && l.editable !== false) for (const kind of ['in', 'out'])
          if (len(q[kind]) > 1e-9) take(len(sub(ts(add(q.p, q[kind])), s)) - 3, { type: 'handle', id: l.id, i: this.sel.i, kind });
      }
      for (const l of this.layers()) { if (l.editable === false) continue;
        const bias = this.sel && this.sel.id === l.id ? 1 : 0;
        l.spline.pts.forEach((q, i) => take(len(sub(ts(q.p), s)) - 2 - bias, { type: 'point', id: l.id, i }));
        const sm = sample(l.spline, 2);
        sm.xy.forEach((p, k) => take(len(sub(ts(p), s)) - bias, { type: 'curve', id: l.id, par: sm.par[k] }));
      }
      return best;
    }
    mousedown(e) {
      if (e.button !== 0 || !this.o.enabled()) return;
      const { s, img } = this.evPos(e), h = this.hit(s);
      if (!h) { if (this.sel) { this.sel = null; this.draw(); } return; }
      const l = this.layer(h.id);
      e.preventDefault(); e.stopPropagation();
      if (h.type === 'curve') {
        this.snapshot(h.id);
        const i = split(l.spline, h.par);
        this.sel = { id: h.id, i }; this.changed(h.id, true);
        this.drag = { type: 'point', id: h.id, i, start: img, orig: clone(l.spline), soft: false };
        return;
      }
      if (h.type === 'point' && (e.ctrlKey || e.metaKey)) {
        this.snapshot(h.id);
        const q = l.spline.pts[h.i];
        q.cusp = !q.cusp; if (!q.cusp) makeSmooth(q);
        this.sel = { id: h.id, i: h.i }; this.changed(h.id, true); return;
      }
      this.snapshot(h.id);
      this.sel = { id: h.id, i: h.i };
      this.drag = Object.assign({}, h, { start: img, orig: clone(l.spline), soft: h.type === 'point' && e.shiftKey, alt: e.altKey });
      this.draw();
    }
    mousemove(e) {
      if (!this.drag) return;
      const { img } = this.evPos(e), d = this.drag, l = this.layer(d.id);
      if (!l) return;
      const dv = sub(img, d.start), P = l.spline.pts, O = d.orig.pts;
      if (d.type === 'point') {
        if (d.soft) {                          // neighbours follow by distance along the original points
          const c = O[d.i].p;
          P.forEach((q, k) => { const r = len(sub(O[k].p, c)); let w = Math.max(0, 1 - r / this.soft); w = w * w * (3 - 2 * w);
            if (k === d.i) w = 1; q.p = add(O[k].p, mul(dv, w)); });
        } else P[d.i].p = add(O[d.i].p, dv);
      } else {                                 // handle
        const q = P[d.i], o = O[d.i], other = d.kind === 'in' ? 'out' : 'in';
        q[d.kind] = add(o[d.kind], dv);
        if (d.alt || e.altKey) q.cusp = true;
        if (!q.cusp) { const lo = len(o[other]), lk = len(q[d.kind]);
          if (lk > 1e-9) q[other] = mul(q[d.kind], -(lo || lk) / lk); }
      }
      this.changed(d.id, false);
    }
    mouseup() { if (!this.drag) return; const id = this.drag.id; this.drag = null; this.changed(id, true); }
    keydown(e) {
      if (!this.o.enabled()) return;
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        if (!this.undoStack.length && !this.redoStack.length) return;
        e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); return;
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel) {
        const l = this.layer(this.sel.id); if (!l || l.spline.pts.length <= 2) return;
        e.preventDefault(); this.snapshot(l.id);
        l.spline.pts.splice(this.sel.i, 1); this.sel.i = Math.min(this.sel.i, l.spline.pts.length - 1);
        this.changed(l.id, true);
      }
    }
    wheel(e) {
      if (!e.shiftKey || !this.o.enabled()) return;
      e.preventDefault();
      this.soft = Math.max(10, Math.min(2000, this.soft * (e.deltaY > 0 ? 0.85 : 1.18)));
      this.draw();
    }
    draw() {
      const c = this.o.canvas, ctx = c.getContext('2d'), ts = this.o.toScreen;
      ctx.clearRect(0, 0, c.width, c.height);
      if (this.o.beforeDraw) this.o.beforeDraw(ctx);
      for (const l of this.layers()) {
        const segs = segments(l.spline), sel = this.sel && this.sel.id === l.id;
        ctx.strokeStyle = l.color || '#ffd24a'; ctx.lineWidth = sel ? 2 : 1.25;
        ctx.setLineDash(l.dash || []);
        ctx.beginPath();
        if (segs.length) {
          const p0 = ts(segs[0][0]); ctx.moveTo(p0[0], p0[1]);
          for (const s of segs) { const a = ts(s[1]), b = ts(s[2]), d = ts(s[3]); ctx.bezierCurveTo(a[0], a[1], b[0], b[1], d[0], d[1]); }
        }
        ctx.stroke(); ctx.setLineDash([]);
        if (l.editable === false) continue;
        l.spline.pts.forEach((q, i) => {
          const p = ts(q.p), isSel = sel && this.sel.i === i, r = this.o.handleR;
          if (isSel) for (const kind of ['in', 'out']) { if (len(q[kind]) < 1e-9) continue;
            const h = ts(add(q.p, q[kind]));
            ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(h[0], h[1]); ctx.stroke();
            ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(h[0], h[1], r - 1, 0, 7); ctx.fill(); }
          ctx.fillStyle = isSel ? '#fff' : (l.color || '#ffd24a');
          ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
          if (q.cusp) { ctx.beginPath(); ctx.moveTo(p[0], p[1] - r - 1); ctx.lineTo(p[0] + r + 1, p[1]); ctx.lineTo(p[0], p[1] + r + 1); ctx.lineTo(p[0] - r - 1, p[1]); ctx.closePath(); }
          else { ctx.beginPath(); ctx.rect(p[0] - r, p[1] - r, 2 * r, 2 * r); }
          ctx.fill(); ctx.stroke();
        });
      }
      if (this.drag && this.drag.soft) {       // soft radius around the dragged point
        const l = this.layer(this.drag.id), q = l && l.spline.pts[this.drag.i];
        if (q) { const p = ts(q.p), e2 = ts(add(q.p, [this.soft, 0]));
          ctx.strokeStyle = 'rgba(255,255,255,.4)'; ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.arc(p[0], p[1], Math.abs(e2[0] - p[0]), 0, 7); ctx.stroke(); ctx.setLineDash([]); }
      }
      if (this.o.afterDraw) this.o.afterDraw(ctx);
    }
  }

  const api = { Editor, sample, split, segments, fromPoints, normalize, point, makeSmooth };
  if (typeof window !== 'undefined') window.MD_CURVES = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
