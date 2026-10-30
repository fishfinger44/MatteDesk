// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/ft_gpu.js – Final Touch strokes on the GPU in the browser (NEXT_STEPS 7a, 2026-10-28): the result of a stroke
// is on screen in the same animation frame, also while drawing; the server stays the source of truth (finaltouch.js
// compares the two once the server's image of the new strokes arrives).
// The maths follows engine/final_touch.py at the proxy size of the preview:
//  - depth inside a stroke D = −SDF(uneven capsules of the path, as engine/keys.draw_variable_stroke) + GROW, at the
//    pixel centres of the FULL plate resolution (pass A: one instance per segment, blend MAX, R16F plate-size texture);
//  - opacity g(D) = D > 0 ? smoothstep(clamp(max(1, D + OFF) / feather)) : 0, feather = rmax · (1 − hardness), < 0.5 = hard –
//    the server's cv2 distance transform of the raster (GROW / OFF measured: the raster reaches ~0.5 px past the shape,
//    the distance counts to the nearest outside pixel centre);
//  - m of a proxy pixel = the INTER_AREA average of g over its plate footprint (pass B), blended into the correction
//    layer C (RGBA16F) as C' = s + C (1 − m): add s = m, remove s = −m, erase s = 0 (= final_touch.layer);
//  - composite: a = C ≥ 0 ? b + C (1 − b) : b (1 + C) (final_touch.compose) on the base proxy b, then black / white /
//    gamma (final_touch.finish_alpha), u8 round half up. The stroke being drawn goes to its own layer L every frame.
// `cpu(job)` is the same maths in plain JS: the contract with final_touch.py (tests/test_ft_gpu.py, node) and the
// reference of the GLSL (`selfTest()` in the browser). localStorage md_ft_gpu = '0' switches the GPU path off.
(function (root) {
  const GROW = 0.5, OFF = 0.3, HARD = 0.6;
  const MODE_SIGN = { add: 1, remove: -1, erase: 0 };

  // ------------------------------------------------------------------------------------------------ shared maths
  function prep(st) {   // -> { sign, F, pts [[x, y, r]], box [x0, y0, x1, y1] plate px } (final_touch.stroke_mask_roi)
    const rDef = Math.max(0.5, Math.min(2000, +st.radius_px || 8));
    const h = Math.max(0, Math.min(1, st.hardness == null ? HARD : +st.hardness));
    let rmax = rDef;
    for (const q of st.p) if (q.length >= 3) rmax = Math.max(rmax, +q[2]);
    const pts = st.p.map((q) => [+q[0], +q[1], Math.max(0.5, q.length >= 3 ? +q[2] : rDef)]);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y, r] of pts) { x0 = Math.min(x0, x - r); y0 = Math.min(y0, y - r); x1 = Math.max(x1, x + r); y1 = Math.max(y1, y + r); }
    return { sign: MODE_SIGN[st.mode] ?? 0, F: rmax * (1 - h), pts, box: [x0 - 2, y0 - 2, x1 + 2, y1 + 2] };
  }
  // signed distance of the hull of two discs (an "uneven capsule"; one disc inside the other = the bigger disc)
  function sdPiece(px, py, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], hh = Math.hypot(dx, dy);
    const da = Math.hypot(px - a[0], py - a[1]) - a[2], db = Math.hypot(px - b[0], py - b[1]) - b[2];
    if (hh <= Math.abs(a[2] - b[2]) + 1e-6) return Math.min(da, db);
    const ux = dx / hh, uy = dy / hh;
    const qy = (px - a[0]) * ux + (py - a[1]) * uy, qx = Math.abs((px - a[0]) * uy - (py - a[1]) * ux);
    const bb = (a[2] - b[2]) / hh, aa = Math.sqrt(1 - bb * bb);
    const k = -bb * qx + aa * qy;
    if (k < 0) return da;
    if (k > aa * hh) return db;
    return qx * aa + qy * bb - a[2];
  }
  function pieces(pts) { return pts.length === 1 ? [[pts[0], pts[0]]] : pts.slice(1).map((b, i) => [pts[i], b]); }
  function g(D, F) {
    if (!(D > 0)) return 0;
    if (F < 0.5) return 1;
    const t = Math.min(1, Math.max(0, Math.max(1, D + OFF) / F));
    return t * t * (3 - 2 * t);
  }
  // INTER_AREA footprint of proxy pixel j on an axis of n plate px (s = n / proxy n): [[i, weight], ...]
  function foot(j, s, n) {
    const a = j * s, b = (j + 1) * s, out = [];
    for (let i = Math.floor(a); i < Math.min(n, Math.ceil(b)); i++) out.push([i, (Math.min(i + 1, b) - Math.max(i, a)) / s]);
    return out;
  }
  // proxy box of a plate box (+ its plate footprint), clipped: [jx0, jy0, jx1, jy1, ix0, iy0, ix1, iy1] | null
  function proxyBox(box, W, H, PW, PH) {
    const sx = W / PW, sy = H / PH;
    const jx0 = Math.max(0, Math.floor(box[0] / sx)), jy0 = Math.max(0, Math.floor(box[1] / sy));
    const jx1 = Math.min(PW, Math.ceil((box[2] + 1) / sx)), jy1 = Math.min(PH, Math.ceil((box[3] + 1) / sy));
    if (jx1 <= jx0 || jy1 <= jy0) return null;
    return [jx0, jy0, jx1, jy1, Math.floor(jx0 * sx), Math.floor(jy0 * sy), Math.min(W, Math.ceil(jx1 * sx)), Math.min(H, Math.ceil(jy1 * sy))];
  }
  function levels(a, fin) {
    const b = +(fin && fin.black) || 0, w = fin && fin.white != null ? +fin.white : 1, gm = fin && fin.gamma != null ? +fin.gamma : 1;
    if (b > 0 || w < 1) a = Math.min(1, Math.max(0, (a - b) / (Math.max(w, b + 1e-3) - b)));
    if (Math.abs(gm - 1) > 1e-9) a = Math.pow(a, 1 / gm);
    return a;
  }
  const u8 = (a) => Math.floor(Math.min(1, Math.max(0, a)) * 255 + 0.5);

  // job = { W, H (plate), PW, PH (proxy), base: Uint8Array PW·PH (the base proxy), strokes: [...], live: stroke | null,
  //         finish: { black, white, gamma } } -> Uint8Array PW·PH (the preview's u8 alpha)
  function cpu(job) {
    const { W, H, PW, PH } = job, sx = W / PW, sy = H / PH;
    const C = new Float64Array(PW * PH);
    for (const st of (job.strokes || []).concat(job.live ? [job.live] : [])) {
      const P = prep(st), pb = proxyBox(P.box, W, H, PW, PH);
      if (!pb) continue;
      const [jx0, jy0, jx1, jy1, ix0, iy0, ix1, iy1] = pb, dw = ix1 - ix0;
      const D = new Float32Array(dw * (iy1 - iy0)), pcs = pieces(P.pts);
      for (let y = iy0; y < iy1; y++) for (let x = ix0; x < ix1; x++) {
        let sd = Infinity;
        for (const [a, b] of pcs) sd = Math.min(sd, sdPiece(x, y, a, b));
        D[(y - iy0) * dw + x - ix0] = -sd + GROW;
      }
      for (let j = jy0; j < jy1; j++) {
        const fy = foot(j, sy, H);
        for (let i = jx0; i < jx1; i++) {
          let m = 0;
          for (const [yy, wy] of fy) for (const [xx, wx] of foot(i, sx, W)) m += wx * wy * g(D[(yy - iy0) * dw + xx - ix0], P.F);
          const k = j * PW + i;
          C[k] = P.sign * m + C[k] * (1 - m);
        }
      }
    }
    const out = new Uint8Array(PW * PH);
    for (let k = 0; k < out.length; k++) {
      const b = job.base[k] / 255, c = C[k];
      out[k] = u8(levels(Math.min(1, Math.max(0, c >= 0 ? b + c * (1 - b) : b * (1 + c))), job.finish));
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------------ WebGL2
  const VS_SEG = `#version 300 es
    layout(location=0) in vec2 corner; layout(location=1) in vec3 ia; layout(location=2) in vec3 ib;
    uniform vec2 plate; flat out vec3 a; flat out vec3 b;
    void main(){ a = ia; b = ib; vec2 lo = min(ia.xy - ia.z, ib.xy - ib.z) - 2.0, hi = max(ia.xy + ia.z, ib.xy + ib.z) + 2.0;
      gl_Position = vec4((mix(lo, hi, corner) + 0.5) / plate * 2.0 - 1.0, 0.0, 1.0); }`;
  const FS_SEG = `#version 300 es
    precision highp float; flat in vec3 a; flat in vec3 b; uniform float grow; out vec4 o;
    float sdPiece(vec2 p){ vec2 d = b.xy - a.xy; float hh = length(d);
      float da = length(p - a.xy) - a.z, db = length(p - b.xy) - b.z;
      if (hh <= abs(a.z - b.z) + 1e-6) return min(da, db);
      vec2 u = d / hh; vec2 r = p - a.xy; float qy = dot(r, u), qx = abs(r.x * u.y - r.y * u.x);
      float bb = (a.z - b.z) / hh, aa = sqrt(1.0 - bb * bb), k = -bb * qx + aa * qy;
      if (k < 0.0) return da; if (k > aa * hh) return db; return qx * aa + qy * bb - a.z; }
    void main(){ o = vec4(-sdPiece(gl_FragCoord.xy - 0.5) + grow, 0.0, 0.0, 1.0); }`;
  const VS_BOX = `#version 300 es
    layout(location=0) in vec2 corner; uniform vec4 box; uniform vec2 size;
    void main(){ gl_Position = vec4(mix(box.xy, box.zw, corner) / size * 2.0 - 1.0, 0.0, 1.0); }`;
  const FS_MASK = `#version 300 es
    precision highp float; uniform highp sampler2D D; uniform vec2 sc; uniform ivec2 plate; uniform float F, off, sgn; out vec4 o;
    float g(float d){ if (!(d > 0.0)) return 0.0; if (F < 0.5) return 1.0; float t = clamp(max(1.0, d + off) / F, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
    void main(){ vec2 j = floor(gl_FragCoord.xy); vec2 a = j * sc, e = (j + 1.0) * sc; float m = 0.0;
      int y1 = min(plate.y, int(ceil(e.y))), x1 = min(plate.x, int(ceil(e.x)));
      for (int y = int(floor(a.y)); y < y1; y++) { float wy = (min(float(y) + 1.0, e.y) - max(float(y), a.y)) / sc.y;
        for (int x = int(floor(a.x)); x < x1; x++) m += (min(float(x) + 1.0, e.x) - max(float(x), a.x)) / sc.x * wy * g(texelFetch(D, ivec2(x, y), 0).r); }
      o = vec4(sgn * m, 0.0, 0.0, m); }`;
  const FS_COMP = `#version 300 es
    precision highp float; uniform highp sampler2D B, Cc, L; uniform int ph; uniform vec3 lv; out vec4 o;
    void main(){ ivec2 q = ivec2(int(gl_FragCoord.x), ph - 1 - int(gl_FragCoord.y));
      float b = texelFetch(B, q, 0).r; vec4 l = texelFetch(L, q, 0); float c = l.r + texelFetch(Cc, q, 0).r * (1.0 - l.a);
      float al = clamp(c >= 0.0 ? b + c * (1.0 - b) : b * (1.0 + c), 0.0, 1.0);
      if (lv.x > 0.0 || lv.y < 1.0) al = clamp((al - lv.x) / (max(lv.y, lv.x + 1e-3) - lv.x), 0.0, 1.0);
      if (abs(lv.z - 1.0) > 1e-9) al = pow(al, 1.0 / lv.z);
      float v = floor(al * 255.0 + 0.5) / 255.0; o = vec4(v, v, v, 1.0); }`;

  let off = false;
  try { off = root.localStorage && root.localStorage.getItem('md_ft_gpu') === '0'; } catch (e) { /* storage blocked: on */ }
  let G = null;   // context state | false (no WebGL2 / no float render targets / lost)
  function ctx() {
    if (off || typeof document === 'undefined') return null;
    if (G !== null) return G || null;
    G = false;
    try {
      const cv = document.createElement('canvas');
      const gl = cv.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false, alpha: false, depth: false, stencil: false });
      if (!gl || !gl.getExtension('EXT_color_buffer_float')) return null;
      const hp = gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT);
      if (!hp || hp.precision < 23) return null;
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      const prog = (vs, fs) => {
        const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < n; i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
        return { p, u };
      };
      const seg = prog(VS_SEG, FS_SEG), mask = prog(VS_BOX, FS_MASK), comp = prog(VS_BOX, FS_COMP);
      const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const inst = gl.createBuffer();
      const vaoQuad = gl.createVertexArray(); gl.bindVertexArray(vaoQuad);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      const vaoSeg = gl.createVertexArray(); gl.bindVertexArray(vaoSeg);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, inst);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 0); gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 24, 12); gl.vertexAttribDivisor(2, 1);
      gl.bindVertexArray(null);
      cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); G = false; });
      G = { cv, gl, seg, mask, comp, inst, vaoQuad, vaoSeg, max: gl.getParameter(gl.MAX_TEXTURE_SIZE), tex: {}, key: null, list: [], base: null };
    } catch (e) { G = false; if (root.console) console.warn('Final Touch GPU off:', e && e.message); }
    return G || null;
  }
  function target(g, name, w, h, fmt) {   // a float texture + framebuffer, (re)made at w × h
    const gl = g.gl; let t = g.tex[name];
    if (t && t.w === w && t.h === h) return t;
    if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
    const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer ${name}`);
    t = g.tex[name] = { tex, fb, w, h };
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    return t;
  }
  // one stroke into the layer `dst` (C or L) – pass A (depth, plate size) then pass B (proxy, blended)
  function strokeInto(g, job, st, dst) {
    const gl = g.gl, { W, H, PW, PH } = job;
    const P = prep(st), pb = proxyBox(P.box, W, H, PW, PH);
    if (!pb) return;
    const [jx0, jy0, jx1, jy1, ix0, iy0, ix1, iy1] = pb;
    const D = target(g, 'D', W, H, gl.R16F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, D.fb); gl.viewport(0, 0, W, H);
    gl.enable(gl.SCISSOR_TEST); gl.scissor(ix0, iy0, ix1 - ix0, iy1 - iy0);
    gl.clearBufferfv(gl.COLOR, 0, [-1000, 0, 0, 0]);
    const pcs = pieces(P.pts), data = new Float32Array(pcs.length * 6);
    pcs.forEach(([a, b], i) => data.set([a[0], a[1], a[2], b[0], b[1], b[2]], i * 6));
    gl.useProgram(g.seg.p); gl.uniform2f(g.seg.u.plate, W, H); gl.uniform1f(g.seg.u.grow, GROW);
    gl.bindVertexArray(g.vaoSeg); gl.bindBuffer(gl.ARRAY_BUFFER, g.inst); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
    gl.enable(gl.BLEND); gl.blendEquation(gl.MAX); gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, pcs.length);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, PW, PH); gl.scissor(jx0, jy0, jx1 - jx0, jy1 - jy0);
    gl.useProgram(g.mask.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, D.tex); gl.uniform1i(g.mask.u.D, 0);
    gl.uniform4f(g.mask.u.box, jx0, jy0, jx1, jy1); gl.uniform2f(g.mask.u.size, PW, PH);
    gl.uniform2f(g.mask.u.sc, W / PW, H / PH); gl.uniform2i(g.mask.u.plate, W, H);
    gl.uniform1f(g.mask.u.F, P.F); gl.uniform1f(g.mask.u.off, OFF); gl.uniform1f(g.mask.u.sgn, P.sign);
    gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(g.vaoQuad); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
  }
  // job as for cpu() plus `key` (frame + base identity: another key = the layer rebuilt) and `baseSrc` (image / canvas of
  // the base proxy) -> a fresh 2D canvas PW × PH with the gray alpha (R = G = B, opaque) | null (no GPU here)
  function render(job) {
    const g = ctx(); if (!g) return null;
    const { W, H, PW, PH } = job;
    if (W > g.max || H > g.max) return null;
    const gl = g.gl;
    try {
      const C = target(g, 'C', PW, PH, gl.RGBA16F), L = target(g, 'L', PW, PH, gl.RGBA16F);
      const list = (job.strokes || []).map((s) => JSON.stringify(s));
      const prefix = g.key === job.key && g.list.length <= list.length && g.list.every((s, i) => s === list[i]);
      if (!prefix) { gl.bindFramebuffer(gl.FRAMEBUFFER, C.fb); gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]); g.list = []; }
      for (let i = g.list.length; i < list.length; i++) strokeInto(g, job, job.strokes[i], C);
      g.list = list; g.key = job.key;
      gl.bindFramebuffer(gl.FRAMEBUFFER, L.fb); gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
      if (job.live && job.live.p && job.live.p.length) strokeInto(g, job, job.live, L);
      if (!g.base || g.base.src !== job.baseSrc || g.base.w !== PW || g.base.h !== PH) {
        if (g.base) gl.deleteTexture(g.base.tex);
        const c2 = document.createElement('canvas'); c2.width = PW; c2.height = PH; c2.getContext('2d').drawImage(job.baseSrc, 0, 0, PW, PH);
        const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, PW, PH, 0, gl.RGBA, gl.UNSIGNED_BYTE, c2);
        for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
        g.base = { src: job.baseSrc, tex, w: PW, h: PH };
      }
      if (g.cv.width !== PW || g.cv.height !== PH) { g.cv.width = PW; g.cv.height = PH; }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, PW, PH);
      gl.useProgram(g.comp.p);
      [g.base.tex, C.tex, L.tex].forEach((t, i) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t); });
      gl.uniform1i(g.comp.u.B, 0); gl.uniform1i(g.comp.u.Cc, 1); gl.uniform1i(g.comp.u.L, 2);
      gl.uniform4f(g.comp.u.box, 0, 0, PW, PH); gl.uniform2f(g.comp.u.size, PW, PH); gl.uniform1i(g.comp.u.ph, PH);
      const fin = job.finish || {};
      gl.uniform3f(g.comp.u.lv, +fin.black || 0, fin.white != null ? +fin.white : 1, fin.gamma != null ? +fin.gamma : 1);
      gl.bindVertexArray(g.vaoQuad); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (gl.isContextLost()) { G = false; return null; }
      const out = document.createElement('canvas'); out.width = PW; out.height = PH;
      out.getContext('2d').drawImage(g.cv, 0, 0);
      return out;
    } catch (e) {
      if (root.console) console.warn('Final Touch GPU failed, back to the server preview:', e && e.message);
      G = false; return null;
    }
  }
  // the GLSL against the JS reference on a synthetic frame (LOCAL_CHECKS): {max, over1, n} of |GPU − CPU| in u8
  function selfTest(W = 4096, H = 2731) {
    const s = Math.min(1, 1920 / Math.max(W, H)), PW = Math.max(1, Math.round(W * s)), PH = Math.max(1, Math.round(H * s));
    const base = new Uint8Array(PW * PH);
    for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) base[y * PW + x] = Math.max(0, Math.min(255, Math.round(128 + 127 * Math.sin(x / 37) * Math.cos(y / 23))));
    const c = document.createElement('canvas'); c.width = PW; c.height = PH;
    const g2 = c.getContext('2d'), im = g2.createImageData(PW, PH);
    for (let k = 0; k < base.length; k++) { im.data[4 * k] = im.data[4 * k + 1] = im.data[4 * k + 2] = base[k]; im.data[4 * k + 3] = 255; }
    g2.putImageData(im, 0, 0);
    const strokes = [
      { mode: 'add', radius_px: 40, hardness: 0.6, p: [[W * 0.3, H * 0.3], [W * 0.5, H * 0.4], [W * 0.55, H * 0.6]] },
      { mode: 'remove', radius_px: 25, hardness: 1, p: [[W * 0.35, H * 0.5, 10], [W * 0.5, H * 0.35, 45]] },
      { mode: 'erase', radius_px: 30, hardness: 0, p: [[W * 0.45, H * 0.45]] },
    ];
    const job = { W, H, PW, PH, base, baseSrc: c, strokes: strokes.slice(0, 2), live: strokes[2], key: `selftest_${Math.random()}`, finish: { gamma: 1.3 } };
    const t0 = performance.now(); const gc = render(job); const tg = performance.now() - t0;
    if (!gc) return null;
    const gd = gc.getContext('2d').getImageData(0, 0, PW, PH).data;
    const ref = cpu(job);
    let max = 0, over1 = 0;
    for (let k = 0; k < ref.length; k++) { const d = Math.abs(gd[4 * k] - ref[k]); max = Math.max(max, d); if (d > 1) over1++; }
    return { max, over1, n: ref.length, gpu_ms: Math.round(tg * 10) / 10 };
  }

  const api = { cpu, render, selfTest, prep, GROW, OFF, get on() { return !!ctx(); }, setOff(v) { off = !!v; if (v) G = null; } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.mdFtGpu = api;
})(typeof window !== 'undefined' ? window : globalThis);
