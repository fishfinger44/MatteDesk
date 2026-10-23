// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/gpu_overlay.js – the viewport overlays built on the GPU instead of a JS loop over the pixels (getImageData ->
// JS -> putImageData, 20–80 ms per proxy frame; user request 2026-10-23). Every builder gives the SAME pixels as the old
// CPU code – also the same premultiplied values inside the canvas, so the view is identical at every zoom / opacity
// (measured on sh060: masks core + alpha, 2 frames, 4 colours, zoom 0.31–2, opacity 45 / 90 %: 0 differing values);
// null = not available here -> the caller keeps its CPU path. localStorage md_gpu_overlay = '0' switches it off.
//  - tint (mask colour, fill), line (outline, 4-neighbour edge of mask > 127), comp (plate over a background through the
//    mask): WebGL shaders in integers. tint writes premultiplied round(rgb·a / 255) like putImageData does – an SVG
//    colour-matrix filter read back the same but stored other premultiplied values (±1 in the view when zoomed);
//    comp = round((P·A + G·(255 − A)) / 255) – the float form was off by 1 on ~0.005 % of the values
//  - bw (Alpha view, opaque): Canvas 2D with an SVG feColorMatrix filter
//  Sources go to WebGL through a 2D canvas: WebGL decodes a JPEG differently from Canvas 2D (off by 1 at alpha 1).
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  let off = false;
  try { off = localStorage.getItem('md_gpu_overlay') === '0'; } catch (e) { /* storage blocked: on */ }
  const can2dFilter = (() => { try { return 'filter' in document.createElement('canvas').getContext('2d'); } catch (e) { return false; } })();
  const size = (s) => [s.naturalWidth || s.width, s.naturalHeight || s.height];
  const canvas = (W, H) => { const c = document.createElement('canvas'); c.width = W; c.height = H; return c; };

  // ---- SVG colour-matrix filter, in a hidden <svg> of the page
  let svg = null;
  function filter(id, values) {
    if (!svg) {
      svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('width', '0'); svg.setAttribute('height', '0'); svg.setAttribute('aria-hidden', 'true');
      svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;';
      document.body.appendChild(svg);
    }
    if (!document.getElementById(id)) {
      const f = document.createElementNS(NS, 'filter');
      f.id = id; f.setAttribute('color-interpolation-filters', 'sRGB');   // linearRGB (the default) would change the values
      f.setAttribute('x', '0'); f.setAttribute('y', '0'); f.setAttribute('width', '1'); f.setAttribute('height', '1');
      const m = document.createElementNS(NS, 'feColorMatrix');
      m.setAttribute('type', 'matrix'); m.setAttribute('values', values);
      f.appendChild(m); svg.appendChild(f);
    }
    return `url(#${id})`;
  }
  // mask -> opaque black / white R·A (= bwCanvas)
  function bw(ov) {
    if (off || !can2dFilter || !document.body) return null;
    const [W, H] = size(ov); const c = canvas(W, H); const g = c.getContext('2d');
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.filter = filter('mdBw', '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0');
    g.drawImage(ov, 0, 0); g.filter = 'none';
    return c;
  }

  // ---- WebGL: one shared context (premultiplied output, copied into the 2D canvas as it is), three programs
  const VS = 'attribute vec2 p; varying vec2 uv; void main(){ uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }';
  const B8 = 'float b8(float v){ return floor(v * 255. + .5); }\n';   // texture value -> its 0..255 integer
  const FS_TINT = 'precision highp float; varying vec2 uv; uniform sampler2D A; uniform vec3 rgb;\n' + B8
    + 'void main(){ float a = b8(texture2D(A, uv).r); gl_FragColor = vec4(floor(rgb * a / 255. + .5) / 255., a / 255.); }';
  const FS_LINE = 'precision highp float; varying vec2 uv; uniform sampler2D A; uniform vec2 px; uniform vec3 rgb;\n' + B8
    + 'float m(vec2 o){ return step(127.5, b8(texture2D(A, uv + o * px).r)); }\n'   // CLAMP_TO_EDGE: off the border = itself
    + 'void main(){ float v = m(vec2(0.));\n'
    + ' float e = (m(vec2(-1., 0.)) != v || m(vec2(1., 0.)) != v || m(vec2(0., -1.)) != v || m(vec2(0., 1.)) != v) ? 1. : 0.;\n'
    + ' gl_FragColor = vec4(rgb / 255. * e, e); }';
  const FS_COMP = 'precision highp float; varying vec2 uv; uniform sampler2D P, A, G;\n' + B8
    + 'void main(){ vec3 p = vec3(b8(texture2D(P, uv).r), b8(texture2D(P, uv).g), b8(texture2D(P, uv).b));\n'
    + ' vec3 g = vec3(b8(texture2D(G, uv).r), b8(texture2D(G, uv).g), b8(texture2D(G, uv).b)); float a = b8(texture2D(A, uv).r);\n'
    + ' gl_FragColor = vec4(floor((p * a + g * (255. - a)) / 255. + .5) / 255., 1.); }';
  let G = null;   // {cv, gl, tint, line, comp, tex[3], max} | false (no WebGL / lost)
  function gl() {
    if (off) return null;
    if (G !== null) return G || null;
    G = false;
    try {
      const cv = document.createElement('canvas');
      const ctx = cv.getContext('webgl', { premultipliedAlpha: true, preserveDrawingBuffer: true, antialias: false, alpha: true, depth: false, stencil: false });
      if (!ctx) return null;
      const hp = ctx.getShaderPrecisionFormat(ctx.FRAGMENT_SHADER, ctx.HIGH_FLOAT);
      if (!hp || hp.precision < 23) return null;   // the integer maths needs float32 (exact up to 2^24)
      const sh = (type, src) => { const s = ctx.createShader(type); ctx.shaderSource(s, src); ctx.compileShader(s); if (!ctx.getShaderParameter(s, ctx.COMPILE_STATUS)) throw new Error(ctx.getShaderInfoLog(s)); return s; };
      const prog = (fs, names) => {
        const pr = ctx.createProgram();
        ctx.attachShader(pr, sh(ctx.VERTEX_SHADER, VS)); ctx.attachShader(pr, sh(ctx.FRAGMENT_SHADER, fs));
        ctx.bindAttribLocation(pr, 0, 'p'); ctx.linkProgram(pr);
        if (!ctx.getProgramParameter(pr, ctx.LINK_STATUS)) throw new Error(ctx.getProgramInfoLog(pr));
        ctx.useProgram(pr); names.forEach((n, i) => ctx.uniform1i(ctx.getUniformLocation(pr, n), i));
        return pr;
      };
      const tint = prog(FS_TINT, ['A']), line = prog(FS_LINE, ['A']), comp = prog(FS_COMP, ['P', 'A', 'G']);
      const buf = ctx.createBuffer(); ctx.bindBuffer(ctx.ARRAY_BUFFER, buf);
      ctx.bufferData(ctx.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), ctx.STATIC_DRAW);
      ctx.enableVertexAttribArray(0); ctx.vertexAttribPointer(0, 2, ctx.FLOAT, false, 0, 0);
      const tex = [0, 1, 2].map((i) => {
        const t = ctx.createTexture(); ctx.activeTexture(ctx.TEXTURE0 + i); ctx.bindTexture(ctx.TEXTURE_2D, t);
        ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_MIN_FILTER, ctx.NEAREST); ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_MAG_FILTER, ctx.NEAREST);
        ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_WRAP_S, ctx.CLAMP_TO_EDGE); ctx.texParameteri(ctx.TEXTURE_2D, ctx.TEXTURE_WRAP_T, ctx.CLAMP_TO_EDGE);
        return t;
      });
      ctx.pixelStorei(ctx.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); G = false; });   // back to the CPU path for good
      G = { cv, gl: ctx, tint, line, comp, tex, max: ctx.getParameter(ctx.MAX_TEXTURE_SIZE) };
    } catch (e) { G = false; }
    return G || null;
  }
  // a source at W×H in a 2D canvas (the same decode / scaling as the CPU code, which drew it into a 2D canvas too)
  function at(src, W, H) { const c = canvas(W, H); c.getContext('2d').drawImage(src, 0, 0, W, H); return c; }
  function draw(prog, W, H, srcs, uniforms) {
    const g = gl(); if (!g || W > g.max || H > g.max) return null;
    const ctx = g.gl;
    if (g.cv.width !== W || g.cv.height !== H) { g.cv.width = W; g.cv.height = H; }
    ctx.viewport(0, 0, W, H); ctx.useProgram(g[prog]);
    srcs.forEach((s, i) => { ctx.activeTexture(ctx.TEXTURE0 + i); ctx.bindTexture(ctx.TEXTURE_2D, g.tex[i]); ctx.texImage2D(ctx.TEXTURE_2D, 0, ctx.RGBA, ctx.RGBA, ctx.UNSIGNED_BYTE, s); });
    for (const [n, v] of Object.entries(uniforms || {})) ctx[`uniform${v.length}f`](ctx.getUniformLocation(g[prog], n), ...v);
    ctx.drawArrays(ctx.TRIANGLE_STRIP, 0, 4);
    if (ctx.isContextLost()) { G = false; return null; }
    const c = canvas(W, H); c.getContext('2d').drawImage(g.cv, 0, 0);
    return c;
  }
  // mask -> colour rgb with alpha = R of the mask (= tintCanvas fill)
  function tint(ov, rgb) {
    if (off) return null;
    const [W, H] = size(ov);
    return draw('tint', W, H, [at(ov, W, H)], { rgb });
  }
  // outline 2 px of mask > 127 (inner + outer edge pixel) in colour rgb (= tintCanvas line)
  function line(ov, rgb) {
    if (off) return null;
    const [W, H] = size(ov);
    return draw('line', W, H, [at(ov, W, H)], { rgb, px: [1 / W, 1 / H] });
  }
  // plate over the background through the mask (R), opaque W×H; bg = a W×H canvas already holding the background
  function comp(plate, mask, bg, W, H) {
    if (off) return null;
    return draw('comp', W, H, [at(plate, W, H), at(mask, W, H), bg]);
  }

  // setOff(true) = the CPU path from now on (A/B comparison of the two paths in one page; the caller clears its caches)
  window.mdGpu = { tint, bw, line, comp, get on() { return !off && !!gl(); }, setOff(v) { off = !!v; } };
})();
