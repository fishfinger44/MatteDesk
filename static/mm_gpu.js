// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// static/mm_gpu.js – the MergeMatte group on the GPU in the browser (WebGL2, float textures), so the Final Touch palette
// „MergeMatte” previews in real time (user request 2026-10-30: the server's numpy port took ~4 s a frame). The same node
// chain as engine/merge_matte.py, pass by pass and formula by formula: Blur = cv2 gaussian (sigma = size · 0.4236, kernel
// radius 4 sigma, normalised, black outside the frame), FilterErode = cv2 erode / dilate with the elliptic element
// (approximate in both – the Nuke FilterErode open item), Keymix / Merge over / xor / min / divide, Saturation (Rec 709),
// luminance Keyer, ACEScct log, sRGB → ACEScg (Utility - sRGB - Texture). The server stays the truth (export / Nuke) and
// the fallback: null = no WebGL2 / EXT_color_buffer_float here. Parity with the port: tools/mm_gpu_parity.py (max 1 / 255).
// `finish` = the Final Touch finishing (levels, gamma, erode, blur; no motion blur) – parity with final_touch.finish_alpha
// on 5 synthetic sets (2026-10-30): max 1 / 255.
//   mdMmGpu.render({plate, alpha, bg: {mode: 'color' | 'checker' | 'image', color: [r, g, b] 0..1, cell, img}, knobs, W, H})
//   -> a W×H 2D canvas (display sRGB, opaque) | null
(function () {
  const SIGMA = 0.4236;   // engine/merge_matte.py BLUR_SIGMA_PER_SIZE
  let G = null;           // {cv, gl, prog: {}, pool: {}, W, H} | false (not available)

  const VS = '#version 300 es\nvoid main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2. - 1., 0., 1.); }';
  const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform ivec2 sz;
out vec4 o;
ivec2 px(){ return ivec2(gl_FragCoord.xy); }
vec4 tf(sampler2D t, ivec2 p){ return (p.x < 0 || p.y < 0 || p.x >= sz.x || p.y >= sz.y) ? vec4(0.) : texelFetch(t, p, 0); }
vec4 unp(vec4 c){ return c.a != 0. ? vec4(c.rgb / c.a, c.a) : c; }
vec4 over(vec4 a, vec4 b){ return a + b * (1. - a.a); }
float lum(vec3 c){ return dot(c, vec3(.2126, .7152, .0722)); }
vec3 sat(vec3 c, float s){ float l = lum(c); return l + (c - l) * s; }
vec3 s2l(vec3 c){ return mix(c / 12.92, pow((max(c, 0.) + .055) / 1.055, vec3(2.4)), vec3(greaterThan(c, vec3(.04045)))); }
vec3 l2s(vec3 c){ return mix(12.92 * c, 1.055 * pow(max(c, 0.), vec3(1. / 2.4)) - .055, vec3(greaterThan(c, vec3(.0031308)))); }
const mat3 S2AP1 = mat3(.6131, .0702, .0206, .3395, .9164, .1096, .0474, .0134, .8698);   // column-major = rows of the numpy matrix
vec3 cct(vec3 x){ vec3 hi = (log2(max(x, 1e-10)) + 9.72) / 17.52; return mix(hi, 10.5402377416545 * x + .0729055341958355, vec3(lessThanEqual(x, vec3(.0078125)))); }
vec3 icct(vec3 y){ vec3 hi = exp2(min(y, 10.) * 17.52 - 9.72); return mix(hi, (y - .0729055341958355) / 10.5402377416545, vec3(lessThanEqual(y, vec3(.155251141552511)))); }
`;
  const FS = {
    // straight FG: ACEScg plate colour where the 8-bit alpha > 0 (unpremult of a premultiplied 0 is 0), alpha
    prepFg: `uniform sampler2D P, A;
void main(){ ivec2 p = px(); vec2 uv = (vec2(p) + .5) / vec2(sz);
  float a = texture(A, uv).r; vec3 lin = S2AP1 * s2l(texture(P, uv).rgb);
  o = vec4(a > 0. ? lin : vec3(0.), a); }`,
    // BG: colour / checker (cells in view px) / image, ACEScg, alpha 0; log = ACEScct
    prepBg: `uniform sampler2D B; uniform int mode, logOn; uniform vec3 col; uniform float cell;
void main(){ ivec2 p = px(); vec2 uv = (vec2(p) + .5) / vec2(sz); vec3 c;
  if (mode == 2) c = texture(B, uv).rgb;
  else if (mode == 1) c = (int(floor(float(p.x) / cell) + floor(float(p.y) / cell)) % 2 == 1) ? vec3(153. / 255.) : vec3(102. / 255.);
  else c = col;
  vec3 l = S2AP1 * s2l(c); o = vec4(logOn == 1 ? cct(l) : l, 0.); }`,
    // FilterErode: grayscale erode (choke > 0) / dilate of the alpha with the cv2 elliptic element of radius r
    erode: `uniform sampler2D S; uniform int r, dil;
void main(){ ivec2 p = px(); float m = dil == 1 ? -1e9 : 1e9;
  for (int dy = -r; dy <= r; dy++) { int hw = int(round(sqrt(float(r * r - dy * dy))));
    for (int dx = -hw; dx <= hw; dx++) { ivec2 q = p + ivec2(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;   // cv2: the border never wins
      float v = texelFetch(S, q, 0).a; m = dil == 1 ? max(m, v) : min(m, v); } }
  o = vec4(0., 0., 0., m); }`,
    // Final Touch finishing (engine/final_touch.finish_alpha): levels (<= black -> 0, >= white -> 1) and gamma of the
    // 8-bit alpha (R), into .a
    fin0: `uniform sampler2D A; uniform float black, white, gam;
void main(){ ivec2 p = px(); vec2 uv = (vec2(p) + .5) / vec2(sz); float a = clamp(texture(A, uv).r, 0., 1.);
  if (black > 0. || white < 1.) a = clamp((a - black) / (max(white, black + 1e-3) - black), 0., 1.);
  if (abs(gam - 1.) > 1e-9) a = pow(a, 1. / gam);
  o = vec4(0., 0., 0., a); }`,
    // cv2.GaussianBlur(ksize = 2·round(3 sigma) + 1, sigma) of .a with cv2's default border (reflect 101)
    blurR: `uniform sampler2D S; uniform int r; uniform ivec2 dir; uniform float inv2s2, norm;
int refl(int i, int n){ if (n == 1) return 0; for (int k = 0; k < 4; k++) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; } return clamp(i, 0, n - 1); }
void main(){ ivec2 p = px(); float acc = 0.;
  for (int i = -r; i <= r; i++) { ivec2 q = p + dir * i; q = ivec2(refl(q.x, sz.x), refl(q.y, sz.y)); acc += texelFetch(S, q, 0).a * exp(-float(i * i) * inv2s2); }
  o = vec4(0., 0., 0., acc * norm); }`,
    // the finished alpha as a grey image (R = alpha, like the server's frame PNG), rows flipped for the canvas
    finOut: `uniform sampler2D S;
void main(){ ivec2 p = ivec2(int(gl_FragCoord.x), sz.y - 1 - int(gl_FragCoord.y)); float a = clamp(texelFetch(S, p, 0).a, 0., 1.);
  o = vec4(a, a, a, 1.); }`,
    // separable gaussian, all four channels, black outside
    blur: `uniform sampler2D S; uniform int r; uniform ivec2 dir; uniform float inv2s2, norm;
void main(){ ivec2 p = px(); vec4 acc = vec4(0.);
  for (int i = -r; i <= r; i++) acc += tf(S, p + dir * i) * exp(-float(i * i) * inv2s2);
  o = acc * norm; }`,
    // straight FG + its new alpha (choke / soften) -> premultiplied FG; log: ACEScct of the straight colour
    fg: `uniform sampler2D S, A; uniform int useA, logOn;
void main(){ ivec2 p = px(); vec4 s = texelFetch(S, p, 0); float a = useA == 1 ? texelFetch(A, p, 0).a : s.a;
  vec3 c = s.rgb;   // straight (0 where the plate alpha was 0)
  if (logOn == 1) { vec4 f = vec4(c * a, a); f = unp(f); c = cct(f.rgb); }
  o = vec4(c * a, a); }`,
    // Decontaminate: interior colour (premult -> Blur -> unpremult) keymixed into the edge by 1 - alpha; -> straight FGD
    decon: `uniform sampler2D F, SP; uniform float amt;
void main(){ ivec2 p = px(); vec4 u = unp(texelFetch(F, p, 0));
  if (amt != 0.) { vec4 s = unp(texelFetch(SP, p, 0)); u.rgb = mix(u.rgb, s.rgb, clamp(1. - u.a, 0., 1.) * amt); }
  o = u; }`,
    // BG outside the FG (bg_out) or the flare source: bright BG (luminance key) outside the FG
    bgout: `uniform sampler2D F, B; uniform int flare; uniform float ka, kb;
void main(){ ivec2 p = px(); float ia = 1. - texelFetch(F, p, 0).a; vec3 b = texelFetch(B, p, 0).rgb;
  if (flare == 0) { o = vec4(b * ia, ia); return; }
  float x = lum(b), k = 0.;
  if (kb != ka && x >= ka && x < kb) k = (x - ka) / (kb - ka);
  if (x >= kb && x <= 1.) k = 1.;
  o = vec4(b * k * ia, k * ia); }`,
    // wrap / flare / bleed blur stack: blur2 over blur1, blur3 over that, blur4 over that
    chain: `uniform sampler2D B1, B2, B3, B4;
void main(){ ivec2 p = px(); vec4 r = over(texelFetch(B2, p, 0), texelFetch(B1, p, 0));
  r = over(texelFetch(B3, p, 0), r); o = over(texelFetch(B4, p, 0), r); }`,
    // bleed colour, light wrap, flare on the straight FG, then premult with the FG alpha and over the BG
    assemble: `uniform sampler2D D, B, CB, CW, CF; uniform float bleed, wrap, flare;
void main(){ ivec2 p = px(); vec4 d = texelFetch(D, p, 0); vec4 f2 = d;
  if (bleed != 0.) { vec4 bl = clamp(texelFetch(CB, p, 0), .001, 100.);
    vec4 luma = vec4(vec3(lum(bl.rgb)), bl.a);
    vec4 chroma = vec4(bl.r != 0. ? luma.r / bl.r : 0., bl.g != 0. ? luma.g / bl.g : 0., bl.b != 0. ? luma.b / bl.b : 0., bl.a != 0. ? luma.a / bl.a : 0.);
    vec4 de = vec4(sat(d.rgb, 1. - bleed), d.a);
    vec4 ap = de + (chroma * de - de) * bleed;
    f2 = mix(d, ap, clamp(bl.a, 0., 1.)); }
  vec4 f4 = f2;
  if (wrap != 0.) f4 = over(texelFetch(CW, p, 0) * wrap, f4);
  if (flare != 0.) f4 = over(texelFetch(CF, p, 0) * flare, f4);
  o = over(vec4(f4.rgb * d.a, d.a), texelFetch(B, p, 0)); }`,
    // Darken edge (min with the BG, rgba) / Match edge (exposure + saturation, rgb) inside 2·bl·(1 − bl) of the blurred alpha
    band: `uniform sampler2D C, BL, B; uniform int kind; uniform float amt, gain, satv;
void main(){ ivec2 p = px(); vec4 c = texelFetch(C, p, 0); float bl = texelFetch(BL, p, 0).a;
  float m = clamp(2. * bl * (1. - bl), 0., 1.) * amt;
  if (kind == 0) { o = mix(c, min(c, texelFetch(B, p, 0)), m); return; }
  vec3 g = sat(c.rgb * gain, satv); o = vec4(mix(c.rgb, g, m), c.a); }`,
    // Edge blur where xor(Blur 3, Blur 3) / 0.9, bypass, mix, back from log, ACEScg -> display sRGB (rows flipped for the canvas)
    final: `uniform sampler2D C, BL3, EB, F, B; uniform int bypass, logOn, eb; uniform float mixv;
void main(){ ivec2 p = ivec2(int(gl_FragCoord.x), sz.y - 1 - int(gl_FragCoord.y));
  vec4 c = texelFetch(C, p, 0), b = texelFetch(B, p, 0), out_ = c;
  if (eb == 1) { float bl = texelFetch(BL3, p, 0).a; float m = clamp(2. * bl * (1. - bl) / .9, 0., 1.);
    out_ = texelFetch(EB, p, 0) * m + c * (1. - m); }
  if (bypass == 1) out_ = over(texelFetch(F, p, 0), b);
  out_ = b + (out_ - b) * mixv;
  vec3 l = logOn == 1 ? icct(out_.rgb) : out_.rgb;
  o = vec4(clamp(l2s(inverse(S2AP1) * l), 0., 1.), 1.); }`,
  };

  function init() {
    if (G !== null) return G;
    G = false;
    try {
      const cv = document.createElement('canvas');
      const gl = cv.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false, depth: false });
      if (!gl || !gl.getExtension('EXT_color_buffer_float')) return G;
      const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
      const vs = sh(gl.VERTEX_SHADER, VS);
      const prog = {};
      for (const [k, src] of Object.entries(FS)) {
        const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, HEAD + src)); gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        prog[k] = { p, u: {} };
      }
      cv.addEventListener('webglcontextlost', () => { G = false; });
      G = { cv, gl, prog, pool: {}, img: {}, W: 0, H: 0, fb: gl.createFramebuffer() };
    } catch (e) { console.warn('mm_gpu: no GPU MergeMatte', e); G = false; }
    return G;
  }
  function tex(name) {   // a W×H RGBA32F render target of the pool
    const g = G, gl = g.gl;
    let t = g.pool[name];
    if (t && t.W === g.W && t.H === g.H) return t.t;
    if (t) gl.deleteTexture(t.t);
    const x = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, x);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, g.W, g.H);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    g.pool[name] = { t: x, W: g.W, H: g.H };
    return x;
  }
  function upload(name, src) {   // an 8-bit source image / canvas, sampled by uv (linear: any size against the view)
    const gl = G.gl;
    let t = G.img[name];
    if (!t) { t = G.img[name] = gl.createTexture(); }
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    return t;
  }
  // one pass: program k, sampler inputs {name: texture}, uniforms {name: number | [..] | {i: int}}, into `out` (null = canvas)
  function pass(k, inputs, uni, out) {
    const g = G, gl = g.gl, pr = g.prog[k];
    gl.useProgram(pr.p);
    const loc = (n) => (n in pr.u ? pr.u[n] : (pr.u[n] = gl.getUniformLocation(pr.p, n)));
    gl.uniform2i(loc('sz'), g.W, g.H);
    let unit = 0;
    for (const [n, t] of Object.entries(inputs)) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(loc(n), unit); unit++; }
    for (const [n, v] of Object.entries(uni || {})) {
      const l = loc(n); if (l === null) continue;
      if (v && v.i !== undefined) { Array.isArray(v.i) ? gl[`uniform${v.i.length}i`](l, ...v.i) : gl.uniform1i(l, v.i); } else if (Array.isArray(v)) gl[`uniform${v.length}f`](l, ...v); else gl.uniform1f(l, v);
    }
    if (out) { gl.bindFramebuffer(gl.FRAMEBUFFER, g.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out, 0); } else gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, g.W, g.H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return out;
  }
  // Blur(size) of texture src (all channels) -> pool target `name` (src itself when size <= 0, like the port's copy)
  function blur(src, size, name) {
    if (!(size > 0)) return src;
    const s = size * SIGMA, ks = (Math.round(s * 8 + 1) | 1), r = (ks - 1) / 2;
    let sum = 0; for (let i = -r; i <= r; i++) sum += Math.exp(-(i * i) / (2 * s * s));
    const u = { r: { i: r }, inv2s2: 1 / (2 * s * s), norm: 1 / sum };
    pass('blur', { S: src }, { ...u, dir: { i: [1, 0] } }, tex(name + '_h'));
    return pass('blur', { S: tex(name + '_h') }, { ...u, dir: { i: [0, 1] } }, tex(name));
  }
  function chain(src, size, name) {
    const b = [1, 2, 4, 8].map((m) => blur(src, size * m, `${name}${m}`));
    return pass('chain', { B1: b[0], B2: b[1], B3: b[2], B4: b[3] }, {}, tex(name));
  }

  // k = MergeMatte knobs with the px knobs already in view px (the caller scales them like preview_region does)
  function render(o) {
    if (!init()) return null;
    const g = G, gl = g.gl, k = o.knobs;
    if (!o.plate || !o.alpha || !(o.W > 0 && o.H > 0)) return null;
    if (o.W > gl.getParameter(gl.MAX_TEXTURE_SIZE) || o.H > gl.getParameter(gl.MAX_TEXTURE_SIZE)) return null;
    g.W = o.W; g.H = o.H;
    if (g.cv.width !== o.W || g.cv.height !== o.H) { g.cv.width = o.W; g.cv.height = o.H; }
    const logOn = k.log ? 1 : 0;
    const P = upload('P', o.plate), A = upload('A', o.alpha);
    const bgm = o.bg.mode === 'image' && o.bg.img ? 2 : o.bg.mode === 'checker' ? 1 : 0;
    const Bi = bgm === 2 ? upload('B', o.bg.img) : P;
    const S0 = pass('prepFg', { P, A }, {}, tex('S0'));
    const BG = pass('prepBg', { B: Bi }, { mode: { i: bgm }, logOn: { i: logOn }, col: o.bg.color || [0, 0, 0], cell: o.bg.cell || 16 }, tex('BG'));
    // Alpha: choke (FilterErode) + soften (Blur) of the alpha
    let Aeff = null;
    if (k.choke !== 0 || k.soften !== 0) {
      const r = Math.round(Math.abs(k.choke));
      Aeff = r > 0 ? pass('erode', { S: S0 }, { r: { i: r }, dil: { i: k.choke < 0 ? 1 : 0 } }, tex('ER')) : S0;
      Aeff = blur(Aeff, k.soften, 'SO');
    }
    const F = pass('fg', { S: S0, A: Aeff || S0 }, { useA: { i: Aeff ? 1 : 0 }, logOn: { i: logOn } }, tex('F'));
    const SP = k.decon !== 0 ? blur(F, k.decon_spread, 'SP') : F;
    const D = pass('decon', { F, SP }, { amt: k.decon }, tex('D'));
    const need = k.bleed !== 0 || k.wrap !== 0;
    const BO = need ? pass('bgout', { F, B: BG }, { flare: { i: 0 }, ka: 0, kb: 0 }, tex('BO')) : null;
    const CB = k.bleed !== 0 ? chain(BO, k.bleed_size, 'CB') : D;
    const CW = k.wrap !== 0 ? chain(BO, k.wrap_size, 'CW') : D;
    let CF = D;
    if (k.flare !== 0) {
      const F0 = pass('bgout', { F, B: BG }, { flare: { i: 1 }, ka: k.flare_threshold, kb: k.flare_threshold + k.flare_softness }, tex('F0'));
      CF = chain(F0, k.wrap_size, 'CF');
    }
    let C = pass('assemble', { D, B: BG, CB, CW, CF }, { bleed: k.bleed, wrap: k.wrap, flare: k.flare }, tex('C1'));
    if (k.darken !== 0) {
      const BL = blur(C, k.darken_width, 'BLD');
      C = pass('band', { C, BL, B: BG }, { kind: { i: 0 }, amt: k.darken, gain: 1, satv: 1 }, tex('C2'));
    }
    if (k.match !== 0) {
      const BL = blur(C, k.match_width, 'BLM');
      C = pass('band', { C, BL, B: BG }, { kind: { i: 1 }, amt: k.match, gain: 2 ** k.match_exposure, satv: k.match_saturation }, tex(C === tex('C2') ? 'C3' : 'C2'));
    }
    const eb = k.edge_blur > 0 ? 1 : 0;
    const BL3 = eb ? blur(C, 3.0, 'BL3') : C;
    const EB = eb ? blur(C, k.edge_blur, 'EB') : C;
    pass('final', { C, BL3, EB, F, B: BG }, { bypass: { i: k.bypass ? 1 : 0 }, logOn: { i: logOn }, eb: { i: eb }, mixv: k.mix }, null);
    if (gl.isContextLost()) { G = false; return null; }
    const out = o.into && o.into.width === o.W && o.into.height === o.H ? o.into : document.createElement('canvas');
    out.width = o.W; out.height = o.H;
    out.getContext('2d').drawImage(g.cv, 0, 0);
    return out;
  }

  // Final Touch finishing of an alpha image (R) on the GPU: levels, gamma, erode / dilate, Gaussian blur – the order of
  // engine/final_touch.finish_alpha WITHOUT the motion blur (it sits between erode and blur and needs the server's flow:
  // the caller keeps the server image whenever mb_strength > 0). fin = {black, white, gamma, erode_px, blur_px} in the
  // image's px. -> W×H canvas | null
  function finish(alpha, fin, W, H, into) {
    if (!init() || !alpha || !(W > 0 && H > 0)) return null;
    const g = G, gl = g.gl;
    if (W > gl.getParameter(gl.MAX_TEXTURE_SIZE) || H > gl.getParameter(gl.MAX_TEXTURE_SIZE)) return null;
    g.W = W; g.H = H;
    if (g.cv.width !== W || g.cv.height !== H) { g.cv.width = W; g.cv.height = H; }
    const A = upload('FA', alpha);
    let T = pass('fin0', { A }, { black: +fin.black || 0, white: fin.white != null ? +fin.white : 1, gam: fin.gamma != null ? +fin.gamma : 1 }, tex('FN0'));
    const e = +fin.erode_px || 0;
    if (Math.abs(e) >= 0.5) T = pass('erode', { S: T }, { r: { i: Math.round(Math.abs(e)) }, dil: { i: e < 0 ? 1 : 0 } }, tex('FN1'));
    const bl = +fin.blur_px || 0;
    if (bl > 0) {
      const r = Math.round(bl * 3);
      let sum = 0; for (let i = -r; i <= r; i++) sum += Math.exp(-(i * i) / (2 * bl * bl));
      const u = { r: { i: r }, inv2s2: 1 / (2 * bl * bl), norm: 1 / sum };
      pass('blurR', { S: T }, { ...u, dir: { i: [1, 0] } }, tex('FN2'));
      T = pass('blurR', { S: tex('FN2') }, { ...u, dir: { i: [0, 1] } }, tex('FN3'));
    }
    pass('finOut', { S: T }, {}, null);
    if (gl.isContextLost()) { G = false; return null; }
    const out = into && into.width === W && into.height === H ? into : document.createElement('canvas');
    out.width = W; out.height = H;
    out.getContext('2d').drawImage(g.cv, 0, 0);
    return out;
  }

  window.mdMmGpu = { render, finish, get on() { return !!init(); } };
})();
