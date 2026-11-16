// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/cleanup3d_areas.js – the area editor of the „🧹 Czyszczenie” mode (docs/PLAN_CLEANUP_3D.md §2, package P7):
//   step C4 „Klucze”  areas of a key to regenerate – drawn in the KEY's pixels when the viewer shows a key frame of the
//                     selected card; on any other frame the drawing is a MARK that the 3D camera carries to the keys
//                     (POST /api/cleanup3d/mark/{frame}, sent with a button);
//   step C3 „Karty”   the card's EXTENT – the same editor, drawn once in one frame (= extent_frame), outline in the
//                     card's colour.
// Tools on the left tool bar (MD_CLEANUP3D.toolMount()): 🖌 Obszar (brush), ✒ Krzywa (closed Bézier), ⌫ Gumka, ▭ Prostokąt.
// Layers in the palette (areaMount() / extentMount()): add / remove per layer, visibility, opacity of the overlay,
// delete, Ctrl+Z. One layer = one Area of engine/cleanup3d/types.py: {id, kind: curve | brush, op: add | remove,
// curve | png, label}.
// Building blocks, not copies: curves = MD_CURVES.Editor (static/curves.js – the ONE Bézier editor; a new curve is
// clicked point by point and closed with Enter / double click, as removers_test/cards_tool does); brush strokes =
// MD_BRUSHCORE (static/brushcore.js: stroke geometry shared with stage 2, pointer capture with [ ] and Shift + drag).
// The editor draws on its own canvas over the viewer (#c3AreaCanvas) and follows the viewer's pan / zoom.
//
// Server (api/cleanup3d.py): GET / PUT /api/cleanup3d/keys/{card}/{frame}/areas, PUT /api/cleanup3d/cards/{card}/extent
// (body + `frame`), POST /api/cleanup3d/mark/{frame}. Saving is debounced; a brush layer goes as PNG data
// (`png_data`: data URL, 8-bit, white = area, black = outside, size of the key = `width` × `height`) only when its
// pixels changed – without `png_data` the server keeps the file `png` it has. While a write answers an error (501
// until package P8) the server's message is shown and the drawing stays here (per key / card / frame, until reload).
//
// window.MD_C3AREAS: .state (internals for tests), .setTool(t), .payload(), .save(), .sendMark(), .undo(), .sync()
(function () {
  const C3 = () => window.MD_CLEANUP3D;
  const V = () => window.viewer;
  const BC = () => window.MD_BRUSHCORE;
  const CV = () => window.MD_CURVES;
  const api = (m, u, b) => window.UI.api(m, u, b);
  const toast = (t, ms) => window.UI.toast(t, ms);
  const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clone = (o) => JSON.parse(JSON.stringify(o));

  const TOOLS = [
    { t: 'brush', icon: '🖌', label: 'Obszar', title: 'Pędzel: zamaluj obszar. Rozmiar: [ ] albo Shift + przeciągnij; Alt + przeciągnij = przesuwanie obrazu; Ctrl+Z cofa' },
    { t: 'curve', icon: '✒', label: 'Krzywa', title: 'Zamknięta krzywa Béziera: klikaj punkty wokół obszaru, Enter albo dwuklik zamyka, Esc anuluje. Potem punkty i styczne jak w roto: Shift + przeciągnij = miękko, Alt = załamanie, Ctrl + klik = ostry ↔ gładki, klik na krzywej = nowy punkt, Delete usuwa punkt' },
    { t: 'erase', icon: '⌫', label: 'Gumka', title: 'Gumka: wymazuje z aktywnej warstwy pędzla. Rozmiar: [ ] albo Shift + przeciągnij' },
    { t: 'rect', icon: '▭', label: 'Prostokąt', title: 'Prostokąt: przeciągnij po obrazie – powstaje zamknięta krzywa z czterech punktów (dalej edytowana narzędziem Krzywa)' },
  ];
  const COL = { add: '#50ff78', remove: '#ff3c5a' };
  const SAVE_MS = 600;

  const A = { ready: false, tool: 'none', cur: null, cache: {}, imgs: {}, show: true, op: 60, radius: 24, adding: null, rect: null,
    pendingNew: false, onlyCard: false, ed: null, input: null, canvas: null, off: null, raf: 0, toasted: {} };
  try { const o = localStorage.getItem('md_c3_areas_o'); if (o !== null && +o >= 0 && +o <= 100) A.op = +o; } catch (e) { /* storage blocked */ }
  try { const r = +localStorage.getItem('md_c3_brush_r'); if (r >= 1) A.radius = r; } catch (e) { /* storage blocked */ }

  // ------------------------------------------------------------------------------------------------ state of Cleanup 3D
  const on = () => document.body.classList.contains('c3-on') && !!(C3() && C3().ready && V());
  const curFrame = () => V().currentFrame;
  const cardsAll = () => ((C3() && C3().state && C3().state.cards) || []);
  const cardOf = (id) => cardsAll().find((c) => c.id === id) || null;
  const keysOf = (id) => { const c = cardOf(id); return (c && c.keys) || []; };
  const cardName = (id) => { const c = cardOf(id); return c ? c.name : id; };
  function plateSize() {
    const v = V(); const m = (v && (v.projectData || v.manifest)) || {};
    if (m.plate_w && m.plate_h) return [+m.plate_w, +m.plate_h];
    const img = v && v.getCurrentImage && v.getCurrentImage();
    return img && img.width ? [img.width, img.height] : null;
  }

  // what the editor works on now: a key's areas, a mark of a frame that is not a key, a card's extent
  function wanted() {
    if (!on()) return null;
    const c3 = C3(); const card = c3.card; const f = curFrame();
    if (c3.step === 'cards') return card ? { type: 'extent', card, key: `ext:${card}` } : null;
    if (c3.step !== 'keys') return null;
    if (card && keysOf(card).some((k) => k.frame === f)) return { type: 'key', card, frame: f, key: `key:${card}:${f}` };
    return { type: 'mark', frame: f, key: `mark:${f}` };
  }
  const urlOf = (T) => (T.type === 'key' ? `/api/cleanup3d/keys/${encodeURIComponent(T.card)}/${T.frame}/areas`
    : T.type === 'extent' ? `/api/cleanup3d/cards/${encodeURIComponent(T.card)}/extent` : `/api/cleanup3d/mark/${T.frame}`);
  // pixels of a saved brush layer: GET <areas url>/<area id>/png (key) · <extent url>/<area id>/png (extent)
  const pngUrl = (T, id) => `${urlOf(T)}/${encodeURIComponent(id)}/png`;

  // ------------------------------------------------------------------------------------------------ layers
  function fromServer(T, a) {
    const l = { id: String(a.id), kind: a.kind, op: a.op === 'remove' ? 'remove' : 'add', label: a.label || '', visible: true };
    if (a.kind === 'curve') l.curve = clone(a.curve);
    else { l.png = a.png; l.strokes = []; l.rev = 0; l.savedRev = 0; loadImg(T, l); }
    return l;
  }
  // a saved brush layer: its PNG (white = area) as a white-on-transparent mask for the overlay; the PNG itself is kept
  // for the next upload (new strokes are painted over it)
  function loadImg(T, l) {
    const k = `${T.key}/${l.id}`;
    if (A.imgs[k] || typeof Image === 'undefined') return;
    const img = new Image();
    const rec = A.imgs[k] = { img, mask: null, failed: false };
    img.onload = () => {
      try {
        const w = Math.min(img.naturalWidth || img.width, 2048), h = Math.round(w * (img.naturalHeight || img.height) / (img.naturalWidth || img.width));
        const c = document.createElement('canvas'); c.width = w; c.height = h;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(img, 0, 0, w, h);
        const d = g.getImageData(0, 0, w, h);
        for (let i = 0; i < d.data.length; i += 4) { d.data[i + 3] = d.data[i]; d.data[i] = d.data[i + 1] = d.data[i + 2] = 255; }
        g.putImageData(d, 0, 0);
        rec.mask = c;
        let x0 = w, y0 = h, x1 = -1, y1 = -1;                  // bounds of the saved pixels (the generate bar's place)
        for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
          if (d.data[(y * w + x) * 4 + 3] > 127) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
        }
        const ps = plateSize(); const k = ps ? ps[0] / w : 1;
        rec.box = x1 < 0 ? null : [x0 * k, y0 * k, (x1 + 2) * k, (y1 + 2) * k];
      } catch (e) { rec.failed = true; }
      redraw(); renderList();
    };
    img.onerror = () => { rec.failed = true; renderList(); };
    img.src = pngUrl(T, l.id);
  }
  const imgOf = (T, l) => A.imgs[`${T.key}/${l.id}`] || null;
  // new strokes may go to a brush layer only when its saved pixels are here (or it has none)
  const paintable = (T, l) => l && l.kind === 'brush' && (!l.png || !!(imgOf(T, l) && imgOf(T, l).mask) || l.fresh);
  const layerById = (T, id) => T.layers.find((l) => l.id === id) || null;
  function newId(T) { let n = T.layers.length + 1; while (layerById(T, `a${n}`)) n++; return `a${n}`; }
  const layerNo = (T, l) => T.layers.filter((x) => x.kind === l.kind).indexOf(l) + 1;
  const colorOf = (T, l) => (l.op === 'remove' ? COL.remove : (T.type === 'extent' ? ((cardOf(T.card) || {}).color || COL.add) : COL.add));
  // an extent is edited only in the frame it was drawn in (the camera carries it to the others)
  const editable = (T) => !!T && (T.type !== 'extent' || !T.layers.length || T.frame == null || T.frame === curFrame());
  const visibleNow = (T) => !!T && A.show && (T.type !== 'extent' || T.frame == null || T.frame === curFrame());

  function newTarget(w) {
    return { key: w.key, type: w.type, card: w.card || null, frame: w.frame == null ? null : w.frame, layers: [], active: null, undo: [],
      dirty: false, loaded: false, saving: false, again: false, saveT: null, error: null, rev: 0 };
  }
  async function load(T) {
    T.loaded = true;
    if (T.type === 'mark') return;
    if (T.type === 'extent') {
      const c = cardOf(T.card);
      T.layers = ((c && c.extent) || []).map((a) => fromServer(T, a));
      T.frame = c && c.extent_frame != null ? +c.extent_frame : null;
      T.active = T.layers.length ? T.layers[T.layers.length - 1].id : null;
      return;
    }
    try {
      const d = await api('GET', urlOf(T));
      if (T.dirty) return;                                   // drawn while the list was loading: the drawing wins
      T.layers = ((d && d.areas) || []).map((a) => fromServer(T, a));
      T.active = T.layers.length ? T.layers[T.layers.length - 1].id : null;
      T.error = null;
    } catch (e) { T.error = e.message || String(e); }
    if (A.cur === T) { render(); redraw(); }
  }

  // ------------------------------------------------------------------------------------------------ undo / change
  function pushUndo(T) {
    T.undo.push(JSON.stringify({ layers: T.layers, active: T.active, frame: T.frame }));
    if (T.undo.length > 100) T.undo.shift();
  }
  function undo() {
    const T = A.cur; if (!T) return false;
    if (A.adding) { A.adding.pop(); if (!A.adding.length) A.adding = null; redraw(); return true; }
    const s = T.undo.pop(); if (!s) return false;
    const st = JSON.parse(s);
    T.layers = st.layers; T.active = st.active; T.frame = st.frame;
    for (const l of T.layers) if (l.kind === 'brush') l.savedRev = -1;   // pixels on the server may be newer: send again
    if (A.ed) A.ed.select(null);
    changed(T);
    return true;
  }
  function changed(T) {
    T.dirty = true; T.rev++;
    if (T.type === 'extent') { if (!T.layers.length) T.frame = null; else if (T.frame == null) T.frame = curFrame(); }
    if (T.type === 'mark') A.cache[T.key] = T;               // a mark waits for „Przenieś” – kept when the frame changes
    render(); redraw();
    if (T.type !== 'mark') { clearTimeout(T.saveT); T.saveT = setTimeout(() => save(T), SAVE_MS); }
  }

  // ------------------------------------------------------------------------------------------------ payload / saving
  // a brush layer at the key's resolution: the saved PNG (when there is one) + strokes in order, eraser = black
  function pngData(T, l) {
    const ps = plateSize(); if (!ps) return null;
    const c = document.createElement('canvas'); c.width = ps[0]; c.height = ps[1];
    const g = c.getContext('2d');
    g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height);
    const rec = imgOf(T, l);
    if (rec && rec.img && !rec.failed && l.png) g.drawImage(rec.img, 0, 0, c.width, c.height);
    for (const st of l.strokes || []) { g.fillStyle = g.strokeStyle = st.erase ? '#000' : '#fff'; BC().strokePath(g, st, 1, drawVar()); }
    let url = null;
    try { url = c.toDataURL('image/png'); } catch (e) { url = null; }
    return url ? { png_data: url, width: c.width, height: c.height } : null;
  }
  const drawVar = () => (window.MD && window.MD.drawVarStroke) || null;
  function areaJson(T, l, withPixels) {
    const a = { id: l.id, kind: l.kind, op: l.op, label: l.label || '' };
    if (l.kind === 'curve') a.curve = clone(l.curve);
    else {
      a.png = l.png || `a_${l.id}.png`;
      if (withPixels && l.savedRev !== l.rev) Object.assign(a, pngData(T, l) || {});
    }
    return a;
  }
  function payload(T = A.cur) {
    if (!T) return null;
    const body = { areas: T.layers.map((l) => areaJson(T, l, true)) };
    if (T.type === 'extent') body.frame = T.layers.length ? T.frame : null;
    if (T.type === 'mark') body.cards = A.onlyCard && C3().card ? [C3().card] : null;
    return body;
  }
  function failed(T, e, always) {
    T.error = e.message || String(e);
    const k = `${T.key}|${T.error}`;
    if (always || !A.toasted[k]) { A.toasted[k] = true; toast(T.error, 8000); }   // the server's text (501 until P8), once per place
  }
  async function save(T = A.cur) {
    if (!T || T.type === 'mark') return false;
    clearTimeout(T.saveT); T.saveT = null;
    if (!T.dirty) return true;
    if (T.saving) { T.again = true; return false; }
    T.saving = true;
    const rev0 = T.rev; const revs = Object.fromEntries(T.layers.map((l) => [l.id, l.rev]));
    let ok = false;
    try {
      const body = payload(T);
      const r = await api('PUT', urlOf(T), body);
      for (const l of T.layers) if (l.kind === 'brush' && revs[l.id] === l.rev) { l.savedRev = l.rev; if (!l.png) { l.png = `a_${l.id}.png`; l.fresh = true; } }
      if (T.rev === rev0) T.dirty = false;
      T.error = null; ok = true;
      if (T.type === 'extent' && C3().setCardExtent) {
        const plain = (r && Array.isArray(r.areas) ? r.areas : T.layers.map((l) => areaJson(T, l, false)));
        C3().setCardExtent(T.card, plain, body.frame);
      }
    } catch (e) { failed(T, e, false); }
    T.saving = false;
    if (T.again) { T.again = false; if (ok) { clearTimeout(T.saveT); T.saveT = setTimeout(() => save(T), SAVE_MS); } }
    if (A.cur === T) renderStatus();
    return ok;
  }
  async function sendMark() {
    const T = A.cur; if (!T || T.type !== 'mark' || !T.layers.length || T.saving) return false;
    T.saving = true; renderStatus();
    let ok = false;
    try {
      const r = await api('POST', urlOf(T), payload(T));
      const ks = (r && r.keys) || [];
      toast(ks.length ? `Zaznaczenie przeniesione na klucze: ${ks.map((k) => `${cardName(k.card)} F${k.frame}`).join(', ')}` : 'Zaznaczenie przeniesione na klucze', 6000);
      T.layers = []; T.undo = []; T.active = null; T.dirty = false; T.error = null; delete A.cache[T.key];
      for (const [k, t] of Object.entries(A.cache)) if (t.type === 'key' && !t.dirty) delete A.cache[k];   // the keys got new areas
      ok = true;
    } catch (e) { failed(T, e, true); }
    T.saving = false;
    render(); redraw();
    if (ok && C3().refresh) C3().refresh();
    return ok;
  }

  // ------------------------------------------------------------------------------------------------ drawing actions
  function refuse(T) {
    if (!T) { toast(C3() && C3().step === 'cards' ? 'Najpierw wybierz kartę z listy' : 'Najpierw wybierz kartę', 4000); return true; }
    if (!editable(T)) { toast(`Zasięg tej karty jest narysowany w klatce F${T.frame} – edytuj go tam („Skocz”) albo wyczyść i narysuj od nowa`, 6000); return true; }
    return false;
  }
  function addLayer(T, l) { T.layers.push(l); T.active = l.id; A.pendingNew = false; }
  function onStroke(st) {
    const T = A.cur; if (!T || refuse(T)) return;
    const erase = st.mode === 'erase';
    let l = layerById(T, T.active);
    if (erase) {
      if (!paintable(T, l)) { toast('Gumka działa na aktywnej warstwie pędzla – wybierz ją na liście', 4000); return; }
      pushUndo(T);
    } else {
      pushUndo(T);
      if (A.pendingNew || !paintable(T, l)) { l = { id: newId(T), kind: 'brush', op: 'add', label: '', visible: true, strokes: [], rev: 0, savedRev: -1 }; addLayer(T, l); }
    }
    l.strokes.push({ p: st.p, radius_px: st.radius_px, ...(erase ? { erase: true } : {}) });
    l.rev++; l.visible = true;
    changed(T);
  }
  function addCurve(T, spline) {
    if (refuse(T)) return;
    pushUndo(T);
    const l = { id: newId(T), kind: 'curve', op: 'add', label: '', visible: true, curve: spline };
    addLayer(T, l);
    changed(T);
    if (A.ed) A.ed.select(l.id, 0);
  }
  function finishCurve() {
    const pts = A.adding; A.adding = null;
    if (A.cur && pts && pts.length >= 3) addCurve(A.cur, CV().fromPoints(pts, true));
    else { if (pts && pts.length) toast('Zamknięta krzywa potrzebuje co najmniej 3 punktów', 4000); redraw(); }
  }
  function rectSpline(a, b) {
    const P = CV().point;
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    return { type: 'bezier', closed: true, pts: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map((p) => P(p, null, null, true)) };
  }
  function setTool(t, quiet) {
    t = t === A.tool ? 'none' : t;
    if (t !== 'none' && !TOOLS.some((x) => x.t === t)) return;
    A.tool = t; A.adding = null; A.rect = null;
    if (A.ed && t !== 'curve') A.ed.select(null);
    renderTools(); if (!quiet) redraw();
  }

  // ------------------------------------------------------------------------------------------------ canvas over the viewer
  function geom() {
    const v = V(); const img = v && v.getCurrentImage && v.getCurrentImage(); const ps = plateSize();
    if (!img || !ps || !img.width) return null;
    return { kx: (img.width / ps[0]) * v.zoom, ky: (img.height / ps[1]) * v.zoom, px: v.panX || 0, py: v.panY || 0 };
  }
  const toScreen = (p) => { const g = geom(); return g ? [g.px + p[0] * g.kx, g.py + p[1] * g.ky] : [0, 0]; };
  const toImageXY = (sx, sy) => { const g = geom(); return g ? [(sx - g.px) / g.kx, (sy - g.py) / g.ky] : [0, 0]; };
  function evScreen(e) {
    const c = A.canvas; const r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * (r.width ? c.width / r.width : 1), (e.clientY - r.top) * (r.height ? c.height / r.height : 1)];
  }
  function evImage(e) {
    const ps = plateSize(); if (!geom() || !ps) return null;
    const s = evScreen(e); const p = toImageXY(s[0], s[1]);
    return [Math.min(Math.max(p[0], 0), ps[0] - 1), Math.min(Math.max(p[1], 0), ps[1] - 1)];
  }
  function fitCanvas() {
    const c = A.canvas; const vc = V() && V().canvas; if (!c || !vc) return;
    if (c.width !== vc.width || c.height !== vc.height) { c.width = vc.width; c.height = vc.height; }
    const st = c.style; const w = `${vc.offsetWidth || vc.width}px`, h = `${vc.offsetHeight || vc.height}px`;
    if (st.width !== w) st.width = w;
    if (st.height !== h) st.height = h;
    const l = `${vc.offsetLeft || 0}px`, t = `${vc.offsetTop || 0}px`;
    if (st.left !== l) st.left = l;
    if (st.top !== t) st.top = t;
  }
  // curve layers for MD_CURVES.Editor (live objects – it edits the spline in place)
  function edLayers() {
    const T = A.cur; const out = [];
    if (T && visibleNow(T)) {
      const can = A.tool === 'curve' && editable(T);
      for (const l of T.layers) if (l.kind === 'curve' && l.visible) out.push({ id: l.id, spline: l.curve, color: colorOf(T, l), dash: l.op === 'remove' ? [6, 4] : undefined, editable: can });
    }
    if (A.adding && A.adding.length) out.push({ id: '+adding', spline: CV().fromPoints(A.adding, A.adding.length >= 3), color: '#ffffff', dash: [4, 4], editable: false });
    if (A.rect) out.push({ id: '+rect', spline: rectSpline(A.rect.a, A.rect.b), color: '#ffffff', dash: [4, 4], editable: false });
    return out;
  }
  function tintLayer(ctx, T, l, g, live) {     // one brush layer as a flat tinted shape (eraser cuts, in stroke order)
    const off = A.off || (A.off = document.createElement('canvas'));
    const c = ctx.canvas;
    if (off.width !== c.width || off.height !== c.height) { off.width = c.width; off.height = c.height; }
    const o = off.getContext('2d');
    o.setTransform(1, 0, 0, 1, 0, 0); o.globalCompositeOperation = 'source-over'; o.clearRect(0, 0, off.width, off.height);
    const ps = plateSize(); const rec = imgOf(T, l);
    if (rec && rec.mask && ps) o.drawImage(rec.mask, g.px, g.py, ps[0] * g.kx, ps[1] * g.ky);
    o.setTransform(1, 0, 0, 1, g.px, g.py);
    o.fillStyle = o.strokeStyle = '#fff';
    for (const st of (l.strokes || []).concat(live || [])) {
      o.globalCompositeOperation = st.erase ? 'destination-out' : 'source-over';
      BC().strokePath(o, st, g.kx, drawVar());
    }
    o.setTransform(1, 0, 0, 1, 0, 0);
    o.globalCompositeOperation = 'source-in'; o.fillStyle = colorOf(T, l); o.fillRect(0, 0, off.width, off.height);
    o.globalCompositeOperation = 'source-over';
    ctx.save();
    try { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = A.op / 100; ctx.drawImage(off, 0, 0); } finally { ctx.restore(); }
  }
  function beforeDraw(ctx) {
    const T = A.cur; const g = geom(); if (!T || !g || !visibleNow(T)) return;
    const d = A.input && A.input.drawing;
    const liveOn = d ? (d.mode === 'erase' || (!A.pendingNew && paintable(T, layerById(T, T.active))) ? T.active : '+new') : null;
    for (const l of T.layers) {
      if (!l.visible) continue;
      if (l.kind === 'brush') tintLayer(ctx, T, l, g, liveOn === l.id ? [{ p: d.p, radius_px: d.radius_px, erase: d.mode === 'erase' }] : null);
      else {                                                     // light fill of a closed curve; the outline = the editor
        const segs = CV().segments(l.curve); if (!segs.length) continue;
        ctx.save();
        try {
          ctx.globalAlpha = 0.3 * A.op / 100; ctx.fillStyle = colorOf(T, l);
          ctx.beginPath(); const p0 = toScreen(segs[0][0]); ctx.moveTo(p0[0], p0[1]);
          for (const s of segs) { const a = toScreen(s[1]), b = toScreen(s[2]), e = toScreen(s[3]); ctx.bezierCurveTo(a[0], a[1], b[0], b[1], e[0], e[1]); }
          ctx.closePath(); ctx.fill();
        } finally { ctx.restore(); }
      }
    }
    if (liveOn === '+new') tintLayer(ctx, T, { id: '+new', kind: 'brush', op: 'add', strokes: [] }, g, [{ p: d.p, radius_px: d.radius_px }]);
  }
  function afterDraw(ctx) {
    const g = geom(); const inp = A.input;
    if (!g || !inp || !['brush', 'erase'].includes(A.tool)) return;
    const h = inp.resizing ? inp.resizing.p : inp.hover; if (!h) return;
    const p = toScreen(h);
    ctx.save();
    try {
      ctx.lineWidth = 1.5; ctx.strokeStyle = A.tool === 'erase' ? '#c4c4c8' : COL.add;
      ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(1, inp.radius * g.kx), 0, 2 * Math.PI); ctx.stroke();
    } finally { ctx.restore(); }
  }
  function drawNow() {
    A.raf = 0;
    if (!A.canvas) return;
    const act = on() && !!A.cur;
    A.canvas.style.display = act ? '' : 'none';
    A.canvas.classList.toggle('c3-area-on', act && A.tool !== 'none');
    if (act) { fitCanvas(); if (A.ed) A.ed.draw(); }
    const c3 = C3(); if (c3 && c3.onAreasDrawn) { try { c3.onAreasDrawn(); } catch (e) { console.error(e); } }
  }
  // the visible „add” layers of a key, in key pixels: [x0, y0, x1, y1] or null
  function imageBox(T) {
    let b = null;
    const grow = (x0, y0, x1, y1) => { b = b ? [Math.min(b[0], x0), Math.min(b[1], y0), Math.max(b[2], x1), Math.max(b[3], y1)] : [x0, y0, x1, y1]; };
    for (const l of T.layers) {
      if (!l.visible || l.op !== 'add') continue;
      if (l.kind === 'curve') { for (const s of CV().segments(l.curve)) for (const p of s) grow(p[0], p[1], p[0], p[1]); continue; }
      const rec = imgOf(T, l); if (rec && rec.box) grow(...rec.box);
      for (const st of l.strokes || []) { if (st.erase) continue; const r = st.radius_px || 8; for (const p of st.p) grow(p[0] - r, p[1] - r, p[0] + r, p[1] + r); }
    }
    return b;
  }
  // where the selection of the current KEY is on screen (CSS px in #canvasWrapper) – for the generate bar; null when
  // the editor shows no key or the key has no area; `drawing` while a stroke / curve / rectangle is under way
  function selectionBox() {
    const T = A.cur; const g = geom(); const c = A.canvas;
    if (!on() || !T || T.type !== 'key' || !g || !c || !visibleNow(T)) return null;
    const b = imageBox(T); if (!b) return null;
    const sx = c.width ? (c.offsetWidth || c.width) / c.width : 1; const sy = c.height ? (c.offsetHeight || c.height) / c.height : 1;
    const ox = c.offsetLeft || 0; const oy = c.offsetTop || 0;
    const p0 = toScreen([b[0], b[1]]); const p1 = toScreen([b[2], b[3]]);
    return { x0: ox + p0[0] * sx, y0: oy + p0[1] * sy, x1: ox + p1[0] * sx, y1: oy + p1[1] * sy,
             drawing: !!(A.adding || A.rect || (A.input && A.input.drawing)) };
  }
  function redraw() {
    if (A.raf) return;
    const raf = window.requestAnimationFrame || ((fn) => setTimeout(fn, 0));
    A.raf = raf(drawNow);
  }

  // ------------------------------------------------------------------------------------------------ left tool bar
  function renderTools() {
    const box = C3() && C3().toolMount(); if (!box) return;
    if (!box.querySelector('[data-c3tool]')) {
      box.innerHTML = TOOLS.map((t) => `<button type="button" class="tool-btn" data-c3tool="${t.t}" title="${esc(t.title)}"><span class="tool-icon">${t.icon}</span><span class="tool-label">${esc(t.label)}</span></button>`).join('');
      box.addEventListener('click', (e) => { const b = e.target.closest('[data-c3tool]'); if (b && !b.disabled) setTool(b.dataset.c3tool); });
    }
    box.querySelectorAll('[data-c3tool]').forEach((b) => { b.classList.toggle('active', b.dataset.c3tool === A.tool); b.disabled = !A.cur; });
    if (A.canvas) A.canvas.classList.toggle('c3-area-on', on() && !!A.cur && A.tool !== 'none');
  }

  // ------------------------------------------------------------------------------------------------ palette
  const q = (m, name) => m.querySelector(`[data-ar="${name}"]`);
  function mountOf(T) { const c3 = C3(); if (!c3 || !T) return null; return T.type === 'extent' ? (c3.extentMount && c3.extentMount()) : c3.areaMount(); }
  function build(m) {
    if (!m.dataset.arWired) { m.dataset.arWired = '1'; wireMount(m); }
    if (q(m, 'list')) return;
    m.innerHTML = `<div class="muted pp-hint" data-ar="info"></div>
      <div class="row c3-row" data-ar="jumprow" style="display:none"><button type="button" class="btn btn-secondary btn-xs" data-ar="jump" title="Skocz do klatki, w której narysowano zasięg tej karty"></button></div>
      <div class="pp-chk"><label class="chk" title="Obszary na obrazie (zielone = dodaje, czerwone = odejmuje; zasięg karty w kolorze karty)"><input type="checkbox" data-ar="show" checked title="Pokaż obszary na obrazie"> 👁 obszary</label>
        <input type="range" data-ar="op" min="0" max="100" step="5" value="${A.op}" style="width:70px" title="Krycie podglądu obszarów (%)"><span class="muted" data-ar="opv">${A.op} %</span></div>
      <div class="row c3-row"><button type="button" class="btn btn-ghost btn-xs" data-ar="new" title="Nowa warstwa pędzla: następna kreska zaczyna nową warstwę (osobno włączana, osobno dodaje / odejmuje)">＋ warstwa pędzla</button>
        <button type="button" class="btn btn-ghost btn-xs" data-ar="undo" title="Cofnij ostatnią zmianę obszarów (Ctrl+Z)">↶ Cofnij</button>
        <button type="button" class="btn btn-ghost btn-xs" data-ar="clear" title="Usuń wszystkie warstwy (Ctrl+Z przywraca)">🗑 Wyczyść</button></div>
      <div class="c3-list" data-ar="list"></div>
      <div class="c3-ar-mark" data-ar="mark" style="display:none">
        <div class="muted pp-hint" data-ar="markto"></div>
        <label class="chk" title="Przenieś zaznaczenie tylko na klucze karty wybranej na górze palety; wyłączone = na każdą kartę, której zaznaczenie dotyka"><input type="checkbox" data-ar="only" title="Tylko wybrana karta"> tylko wybrana karta</label>
        <button type="button" class="btn btn-secondary btn-xs" data-ar="send" title="Przenieś zaznaczenie z tej klatki przez kamerę 3D na klucze kart (staje się ich obszarem do dogenerowania)">⇢ Przenieś zaznaczenie na klucze</button></div>
      <div class="muted pp-hint c3-ar-status" data-ar="status"></div>
      <button type="button" class="btn btn-ghost btn-xs" data-ar="retry" style="display:none" title="Wyślij obszary na serwer jeszcze raz">↻ Zapisz ponownie</button>`;
  }
  function wireMount(m) {
    m.addEventListener('click', (e) => {
      const T = A.cur; if (!T || mountOf(T) !== m) return;
      const t = e.target;
      const b = t.closest('[data-ar]'); const name = b && b.dataset.ar;
      if (name === 'new') { A.pendingNew = true; if (A.tool !== 'brush') setTool('brush'); renderList(); return; }
      if (name === 'undo') { undo(); return; }
      if (name === 'clear') { if (T.layers.length) { pushUndo(T); T.layers = []; T.active = null; if (A.ed) A.ed.select(null); changed(T); } return; }
      if (name === 'jump') { if (T.frame != null) V().goToFrame(T.frame); return; }
      if (name === 'send') { sendMark(); return; }
      if (name === 'retry') { T.dirty = true; save(T); return; }
      const del = t.closest('[data-ar-del]');
      if (del) { pushUndo(T); T.layers = T.layers.filter((l) => l.id !== del.dataset.arDel); if (T.active === del.dataset.arDel) T.active = T.layers.length ? T.layers[T.layers.length - 1].id : null; if (A.ed) A.ed.select(null); changed(T); return; }
      const opb = t.closest('[data-ar-op]');
      if (opb) { const l = layerById(T, opb.dataset.arOp); if (l) { pushUndo(T); l.op = l.op === 'add' ? 'remove' : 'add'; changed(T); } return; }
      if (t.closest('input')) return;
      const row = t.closest('[data-ar-layer]');
      if (row) { T.active = row.dataset.arLayer; A.pendingNew = false; renderList(); redraw(); }
    });
    m.addEventListener('change', (e) => {
      const T = A.cur; if (!T || mountOf(T) !== m) return;
      const t = e.target;
      if (t.dataset.ar === 'show') { A.show = t.checked; redraw(); }
      else if (t.dataset.ar === 'only') { A.onlyCard = t.checked; renderInfo(); }
      else if (t.dataset.arVis !== undefined) { const l = layerById(T, t.dataset.arVis); if (l) { l.visible = t.checked; redraw(); } }   // view only: not saved
    });
    m.addEventListener('input', (e) => {
      if (e.target.dataset.ar !== 'op') return;
      A.op = +e.target.value; q(m, 'opv').textContent = `${A.op} %`;
      try { localStorage.setItem('md_c3_areas_o', String(A.op)); } catch (er) { /* storage blocked */ }
      redraw();
    });
  }
  function renderList() {
    const T = A.cur; const m = mountOf(T); if (!m || !q(m, 'list')) return;
    const rows = T.layers.map((l) => {
      const brush = l.kind === 'brush'; const rec = brush ? imgOf(T, l) : null;
      const lost = brush && l.png && rec && rec.failed && !l.fresh;
      return `<div class="c3-ar-row ${l.id === T.active && !A.pendingNew ? 'active' : ''}" data-ar-layer="${esc(l.id)}" title="Klik = warstwa aktywna (pędzel i gumka działają na niej)">
        <label class="chk" title="Pokaż / ukryj tę warstwę na obrazie (tylko podgląd – warstwa dalej działa)"><input type="checkbox" data-ar-vis="${esc(l.id)}"${l.visible ? ' checked' : ''}></label>
        <span class="c3-ar-ic">${brush ? '🖌' : '✒'}</span><span class="c3-ar-name"><span>${brush ? 'Pędzel' : 'Krzywa'}</span> <span>${layerNo(T, l)}</span>${lost ? ' <span class="err" title="Nie udało się wczytać pikseli tej warstwy z serwera – zostaje bez zmian, nowe kreski idą do nowej warstwy">⚠</span>' : ''}</span>
        <button type="button" class="btn btn-ghost btn-xs c3-ar-op ${l.op}" data-ar-op="${esc(l.id)}" title="Warstwa dodaje do obszaru albo odejmuje od niego (warstwy działają po kolei, od góry listy) – klik przełącza">${l.op === 'add' ? '＋ dodaje' : '− odejmuje'}</button>
        <button type="button" class="btn btn-ghost btn-xs" data-ar-del="${esc(l.id)}" title="Usuń tę warstwę (Ctrl+Z przywraca)">✕</button></div>`;
    }).join('');
    const pend = A.pendingNew ? '<div class="c3-ar-row active"><span class="c3-ar-ic">🖌</span><span class="c3-ar-name">nowa warstwa – następna kreska ją tworzy</span></div>' : '';
    q(m, 'list').innerHTML = rows + pend || '<div class="muted">Brak warstw – narysuj obszar narzędziem z lewego paska.</div>';
    q(m, 'undo').disabled = !T.undo.length && !A.adding;
    q(m, 'clear').disabled = !T.layers.length;
  }
  function markTargets() {       // keys the mark of this frame can reach: the selected card's, or every card's
    const cards = A.onlyCard && C3().card ? [cardOf(C3().card)].filter(Boolean) : cardsAll();
    return cards.filter((c) => (c.keys || []).length).map((c) => `${c.name}: ${c.keys.map((k) => `F${k.frame}`).join(', ')}`);
  }
  function renderInfo() {
    const T = A.cur; const m = mountOf(T); if (!m || !q(m, 'info')) return;
    const info = q(m, 'info'); const jr = q(m, 'jumprow'); const mk = q(m, 'mark');
    jr.style.display = 'none'; mk.style.display = 'none';
    if (T.type === 'key') info.textContent = `Klucz F${T.frame} karty „${cardName(T.card)}”: obszary rysujesz w pikselach tego klucza.`;
    else if (T.type === 'extent') {
      if (!T.layers.length) info.textContent = 'Zasięg: cała karta. Narysuj obrys w bieżącej klatce (pędzel albo zamknięta krzywa) – kamera 3D przeniesie go na pozostałe klatki.';
      else if (editable(T)) info.textContent = `Zasięg narysowany w klatce F${T.frame} (bieżąca) – obrys w kolorze karty.`;
      else {
        info.textContent = `Zasięg narysowany w klatce F${T.frame} – edycja tylko w tamtej klatce.`;
        jr.style.display = ''; q(m, 'jump').textContent = `Skocz do F${T.frame}`;
      }
    } else {
      const card = C3().card;
      info.textContent = card && keysOf(card).length ? `Klatka F${T.frame} nie jest kluczem karty „${cardName(card)}” – zaznaczenie z tej klatki kamera 3D przenosi na klucze.`
        : `Klatka F${T.frame} nie jest kluczem – zaznaczenie z tej klatki kamera 3D przenosi na klucze.`;
      mk.style.display = '';
      const to = markTargets();
      q(m, 'markto').textContent = to.length ? `Trafi na klucze kart – ${to.join(' · ')}` : 'Żadna karta nie ma jeszcze klucza – dodaj klucz („＋ klucz w tej klatce”).';
      q(m, 'only').checked = A.onlyCard;
      q(m, 'send').disabled = !T.layers.length || !to.length || T.saving;
    }
  }
  function renderStatus() {
    const T = A.cur; const m = mountOf(T); if (!m || !q(m, 'status')) return;
    const st = q(m, 'status'); const rt = q(m, 'retry');
    rt.style.display = 'none'; st.classList.toggle('err', !!T.error);
    if (T.error) { st.textContent = `⚠ Nie zapisano na serwerze: ${T.error} Narysowane obszary są trzymane tutaj do zamknięcia strony.`; if (T.type !== 'mark') rt.style.display = ''; }
    else if (T.type === 'mark') st.textContent = T.layers.length ? 'Zaznaczenie czeka na „Przenieś zaznaczenie na klucze”.' : '';
    else st.textContent = T.saving ? 'Zapisuję obszary…' : (T.dirty ? 'Zmiany do zapisania…' : (T.layers.length ? '✔ Obszary zapisane.' : ''));
    if (q(m, 'send')) q(m, 'send').disabled = T.type !== 'mark' || !T.layers.length || !markTargets().length || T.saving;
  }
  function render() {
    renderTools();
    const c3 = C3(); if (!c3) return;
    for (const m of [c3.areaMount(), c3.extentMount && c3.extentMount()]) {
      if (!m) continue;
      const mine = A.cur && mountOf(A.cur) === m;
      if (!mine) { if (!m.dataset.arEmpty) { m.innerHTML = '<div class="muted">Wybierz kartę z listy.</div>'; m.dataset.arEmpty = '1'; } continue; }
      if (m.dataset.arEmpty) { m.innerHTML = ''; delete m.dataset.arEmpty; }
      build(m);
      q(m, 'show').checked = A.show;
    }
    if (!A.cur) return;
    renderInfo(); renderList(); renderStatus();
  }

  // ------------------------------------------------------------------------------------------------ sync with Cleanup 3D
  function sync() {
    if (!A.ready && !init()) return;
    const w = wanted();
    if (!w) {
      if (A.cur && A.cur.saveT) save(A.cur);
      A.cur = null; A.adding = null; A.rect = null;
      if (A.tool !== 'none') setTool(A.tool, true);
      render(); redraw(); return;
    }
    if (!A.cur || A.cur.key !== w.key) {
      if (A.cur && A.cur.saveT) save(A.cur);                 // a debounced save of the place we leave goes now
      A.adding = null; A.rect = null; A.pendingNew = false;
      if (A.ed) A.ed.select(null);
      const T = A.cache[w.key] || newTarget(w);
      if (w.type !== 'mark') A.cache[w.key] = T;
      A.cur = T;
      if (!T.loaded) load(T);
    }
    render(); redraw();
  }

  // ------------------------------------------------------------------------------------------------ events
  const typing = (e) => { const t = e.target; const tag = (t && t.tagName) || ''; return tag === 'TEXTAREA' || tag === 'SELECT' || (tag === 'INPUT' && !['checkbox', 'range', 'button', 'radio', 'color'].includes(t.type)); };
  function onKey(e) {
    if (!on() || !A.cur || typing(e)) return;
    const stop = () => { e.preventDefault(); e.stopImmediatePropagation(); };
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z')) { if (A.adding || A.cur.undo.length) { stop(); undo(); } return; }
    if (e.key === 'Enter' && A.adding) { stop(); finishCurve(); return; }
    if (e.key === 'Escape') {
      if (A.adding) { stop(); A.adding = null; redraw(); renderList(); }
      else if (A.rect) { stop(); A.rect = null; redraw(); }
      else if (A.tool !== 'none') { stop(); setTool(A.tool); }
    }
  }
  function forwardPan(e) {       // Alt + drag / middle button = the viewer's hand, wheel = its zoom
    const vc = V() && V().canvas; if (!vc) return;
    vc.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, clientX: e.clientX, clientY: e.clientY, button: 1 }));
  }
  function onMouseDown(e) {
    if (A.tool === 'none' || !A.cur) return;
    // Alt + drag = hand, except on a curve handle in the curve tool (there Alt breaks the tangent – the editor's rule)
    const onCurve = A.tool === 'curve' && !A.adding && A.ed && !!A.ed.hit(evScreen(e));
    if (e.button === 1 || (e.button === 0 && e.altKey && !onCurve)) { e.preventDefault(); e.stopImmediatePropagation(); forwardPan(e); return; }
    if (e.button !== 0) return;
    const p = evImage(e); if (!p) return;
    if (A.tool === 'rect') {
      if (refuse(A.cur)) return;
      e.preventDefault(); A.rect = { a: p, b: p }; redraw(); return;
    }
    if (A.tool !== 'curve') return;
    if (A.adding) {
      const s = evScreen(e); const last = toScreen(A.adding[A.adding.length - 1]);
      if (Math.hypot(s[0] - last[0], s[1] - last[1]) >= 2) A.adding.push(p);       // the 2nd press of a double click adds nothing
      e.preventDefault(); redraw(); return;
    }
    if (A.ed && A.ed.hit(evScreen(e))) return;                 // on a curve / point: the editor takes it
    if (refuse(A.cur)) return;
    e.preventDefault(); A.adding = [p]; redraw(); renderList();
  }
  function onMouseMove(e) { if (A.rect) { const p = evImage(e); if (p) { A.rect.b = p; redraw(); } } }
  function onMouseUp() {
    if (!A.rect) return;
    const r = A.rect; A.rect = null;
    const a = toScreen(r.a), b = toScreen(r.b);
    if (A.cur && Math.abs(a[0] - b[0]) >= 3 && Math.abs(a[1] - b[1]) >= 3) addCurve(A.cur, rectSpline(r.a, r.b)); else redraw();
  }
  function onWheel(e) {
    if (e.shiftKey) return;                                    // Shift + wheel = the curve editor's soft radius
    const vc = V() && V().canvas; if (!vc) return;
    e.preventDefault();
    vc.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: e.clientX, clientY: e.clientY, deltaY: e.deltaY }));
  }

  function init() {
    const c3 = C3(); const v = V(); const cw = document.getElementById('canvasWrapper');
    if (A.ready) return true;
    if (!c3 || !c3.ready || !v || !cw || !BC() || !CV()) return false;
    const c = document.createElement('canvas');
    c.id = 'c3AreaCanvas'; c.className = 'c3-area-canvas'; c.style.display = 'none';
    if (v.canvas && v.canvas.parentNode === cw) cw.insertBefore(c, v.canvas.nextSibling); else cw.appendChild(c);
    A.canvas = c;
    c.addEventListener('mousedown', onMouseDown, true);        // before the curve editor's own handler
    c.addEventListener('dblclick', () => { if (A.adding) finishCurve(); });
    c.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKey, true);           // before the stage shortcuts of objects.js and the editor
    A.input = new (BC().StrokeInput)({
      el: c, radius: A.radius, toImage: evImage,
      enabled: () => on() && !!A.cur && (A.tool === 'brush' || A.tool === 'erase'),
      mode: () => (A.tool === 'erase' ? 'erase' : 'add'),
      canStart: () => !refuse(A.cur) && (A.tool !== 'erase' || paintable(A.cur, layerById(A.cur, A.cur.active)) || (toast('Gumka działa na aktywnej warstwie pędzla – wybierz ją na liście', 4000), false)),
      onStroke, onChange: redraw,
      onRadius: (r) => { A.radius = r; try { localStorage.setItem('md_c3_brush_r', String(r)); } catch (e) { /* storage blocked */ } },
    });
    A.ed = new (CV().Editor)({
      canvas: c, toScreen, toImage: toImageXY, layers: edLayers,
      enabled: () => on() && !!A.cur && A.tool === 'curve' && !A.adding && editable(A.cur),
      onChange: (id, sp, final) => {
        const T = A.cur; if (!T || !final) return;
        // a click that changed nothing (select a point) leaves no undo step and no save
        const top = T.undo[T.undo.length - 1];
        if (top && top === JSON.stringify({ layers: T.layers, active: T.active, frame: T.frame })) { T.undo.pop(); return; }
        T.active = id; changed(T);
      },
      beforeDraw, afterDraw,
    });
    // ONE undo list for strokes, layers and curve edits: the editor's own stack stays empty (its Ctrl+Z does nothing),
    // every edit it is about to make is recorded here instead
    A.ed.snapshot = () => { if (A.cur) pushUndo(A.cur); };
    if (Array.isArray(v.overlayHooks)) v.overlayHooks.push(() => { if (A.cur) redraw(); });   // follow the viewer's pan / zoom
    window.addEventListener('md-frame', () => sync());
    window.addEventListener('md-project', () => { A.cache = {}; A.imgs = {}; A.toasted = {}; A.cur = null; A.adding = null; A.rect = null; sync(); });
    window.addEventListener('resize', () => redraw());
    c3.onChange(() => sync());
    A.ready = true;
    return true;
  }

  window.MD_C3AREAS = {
    get state() { return A; }, get ready() { return A.ready; },
    setTool, payload, save, sendMark, undo, sync, finishCurve, selectionBox,
  };
  function start() {
    if (init()) { sync(); return; }
    setTimeout(() => { if (init()) sync(); }, 0);
    window.addEventListener('md-project', () => { if (!A.ready && init()) sync(); });   // the viewer came later
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
