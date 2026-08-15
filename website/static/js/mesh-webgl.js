// VisionFlow — WebGL2 hero stars + scroll-driven ontology figures.
//
// Two jobs, one canvas:
//   1. A hero star-field that bounces in on load (spring overshoot) and fades
//      out as you scroll past the hero, so the background video takes over.
//   2. Labelled "spiral" figures — the knowledge-stack rings and the substrate
//      helix — parked in the right-hand dead space beside specific text blocks,
//      spun and revealed by scroll as their section passes.
//
// Progressive enhancement: returns null without WebGL2; under
// prefers-reduced-motion it renders a single settled frame (no rAF, no bounce).

export function initMesh(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: false });
  if (!gl) return null;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- palette (site tokens, linear rgb) ----
  const GOLD = [0.831, 0.647, 0.455], BRIGHT = [1.0, 0.843, 0.10], BRONZE = [0.804, 0.498, 0.196];
  const CYAN = [0.0, 0.831, 1.0], PURPLE = [0.545, 0.361, 0.965], EMERALD = [0.063, 0.725, 0.506];
  const AMBER = [0.961, 0.620, 0.043], CRIMSON = [0.914, 0.271, 0.376], SNOW = [0.86, 0.84, 0.90];
  const HUBCOL = [CYAN, PURPLE, EMERALD, AMBER, CRIMSON, GOLD];

  // ---- mat4 ----
  const persp = (f, a, n, fa) => { const t = 1 / Math.tan(f / 2), d = 1 / (n - fa); return [t / a, 0, 0, 0, 0, t, 0, 0, 0, 0, (n + fa) * d, -1, 0, 0, 2 * n * fa * d, 0]; };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function lookAt(e, c, u) {
    const z = nrm(sub(e, c)), x = nrm(cross(u, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]), -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]), -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]), 1];
  }
  function mul(a, b) { const o = new Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3]; return o; }
  function transRotY(cx, cy, cz, a, dy) { // rotate about vertical axis through (cx,cz), translate y by dy
    const c = Math.cos(a), s = Math.sin(a);
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, cx - c * cx - s * cz, dy, cz + s * cx - c * cz, 1];
  }
  const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const lerp = (a, b, t) => a + (b - a) * t;
  const l3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const smooth = (t) => t * t * (3 - 2 * t);
  const backOut = (t) => { const c = 1.70158; const p = t - 1; return 1 + (c + 1) * p * p * p + c * p * p; }; // overshoot
  const R = (s) => { const x = Math.sin(s * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  // ---- shaders ----
  const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = (v, f) => { const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, v)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, f)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; };
  const PT = prog(
    `#version 300 es
    layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol; layout(location=2) in float aSize;
    uniform mat4 uVP, uModel; uniform float uScale, uAlpha; out vec4 vCol;
    void main(){ gl_Position = uVP*uModel*vec4(aPos,1.0); float w = max(gl_Position.w,.1);
    gl_PointSize = clamp(aSize*uScale/w, 1.0, 180.0); vCol = vec4(aCol.rgb, aCol.a*uAlpha); }`,
    `#version 300 es
    precision mediump float; in vec4 vCol; out vec4 o;
    void main(){ vec2 d = gl_PointCoord-.5; float r = length(d)*2.0;
    float a = exp(-r*r*4.5)*smoothstep(1.0,.55,r); o = vec4(vCol.rgb*a*vCol.a, a*vCol.a); }`);
  const LN = prog(
    `#version 300 es
    layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol;
    uniform mat4 uVP, uModel; uniform float uAlpha; out vec4 vCol;
    void main(){ gl_Position = uVP*uModel*vec4(aPos,1.0); vCol = vec4(aCol.rgb, aCol.a*uAlpha); }`,
    `#version 300 es
    precision mediump float; in vec4 vCol; out vec4 o; void main(){ o = vec4(vCol.rgb*vCol.a, vCol.a); }`);
  const uPT = { vp: gl.getUniformLocation(PT, 'uVP'), m: gl.getUniformLocation(PT, 'uModel'), s: gl.getUniformLocation(PT, 'uScale'), a: gl.getUniformLocation(PT, 'uAlpha') };
  const uLN = { vp: gl.getUniformLocation(LN, 'uVP'), m: gl.getUniformLocation(LN, 'uModel'), a: gl.getUniformLocation(LN, 'uAlpha') };

  const vaoP = (data, dyn) => { const v = gl.createVertexArray(); gl.bindVertexArray(v); const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, dyn ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 12); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 28); return { vao: v, buf: b, n: data.length / 8 }; };
  const vaoL = (data) => { const v = gl.createVertexArray(); gl.bindVertexArray(v); const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12); return { vao: v, buf: b, n: data.length / 7 }; };

  // ---- hero stars (bounce-in) ----
  const NSTAR = 460;
  const starHome = [], starStart = [];
  const starArr = new Float32Array(NSTAR * 8);
  for (let i = 0; i < NSTAR; i++) {
    const th = R(i) * 6.283, ph = Math.acos(2 * R(i + 900) - 1), rr = 5 + Math.pow(R(i + 1800), 0.6) * 22;
    const home = [rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph) * 0.72, rr * Math.sin(ph) * Math.sin(th)];
    starHome.push(home);
    starStart.push([home[0] * 0.08, home[1] * 0.08, home[2] * 0.08]); // collapsed toward centre
    const c = R(i + 40) < 0.24 ? l3(GOLD, BRIGHT, R(i + 5)) : (R(i + 41) < 0.5 ? BRONZE : (R(i + 42) < 0.62 ? SNOW : GOLD));
    const o = i * 8;
    starArr[o + 3] = c[0]; starArr[o + 4] = c[1]; starArr[o + 5] = c[2];
    starArr[o + 6] = 0.7 + R(i + 9) * 0.5;
    starArr[o + 7] = 0.06 + R(i + 13) * 0.10;
  }
  const stars = vaoP(starArr, true);
  function writeStars(t) { // t: 0..1 bounce progress
    const e = reduced ? 1 : backOut(clamp(t, 0, 1));
    for (let i = 0; i < NSTAR; i++) {
      const h = starHome[i], s = starStart[i], o = i * 8;
      starArr[o] = lerp(s[0], h[0], e); starArr[o + 1] = lerp(s[1], h[1], e); starArr[o + 2] = lerp(s[2], h[2], e);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, stars.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, starArr);
  }

  // ---- identity spine (hero) ----
  const spineArr = [];
  for (let i = 0; i <= 60; i++) { const t = i / 60, y = -8 + t * 16, x = Math.sin(t * 3.1) * 0.7, z = Math.cos(t * 2.4) * 0.7; const c = l3(BRIGHT, GOLD, Math.abs(t - 0.5) * 2); spineArr.push(x, y, z, c[0], c[1], c[2], 0.5); }
  const spine = vaoL(new Float32Array(spineArr));

  // ---- hero hubs (six substrate colours, gentle cluster) ----
  const hubArr = [];
  for (let i = 0; i < 6; i++) { const a = i / 6 * 6.283, r = 4.4; const c = HUBCOL[i]; const p = [r * Math.cos(a), Math.sin(a * 1.3) * 2.4, r * Math.sin(a)]; hubArr.push(p[0], p[1], p[2], c[0] * 1.5, c[1] * 1.5, c[2] * 1.5, 1.0, 0.6); hubArr.push(p[0], p[1], p[2], c[0], c[1], c[2], 0.3, 1.9); }
  const heroHubs = vaoP(new Float32Array(hubArr));

  // ---- FIGURE: knowledge-stack rings (labelled spiral), parked screen-right ----
  const FIG = [5.0, -0.3, -1];           // world origin, projects to the right
  const RINGS = [
    { nm: 'Corpus', rl: '8,100+ pages', col: GOLD },
    { nm: 'Ontology', rl: 'OWL 2 EL', col: l3(GOLD, PURPLE, 0.4) },
    { nm: 'Knowledge graph', rl: 'runtime instances', col: PURPLE },
    { nm: 'Reasoning', rl: 'EL + Whelk', col: l3(PURPLE, CYAN, 0.5) },
    { nm: 'Grounding', rl: 'the Loom', col: CYAN },
  ];
  const ringArr = [], ringNodeArr = [];
  RINGS.forEach((rd, ri) => {
    const y = FIG[1] - 4.8 + ri * 2.4, rad = 4.0 - ri * 0.5, twist = ri * 0.5;
    for (let i = 0; i < 60; i++) {
      const a1 = i / 60 * 6.283 + twist, a2 = (i + 1) / 60 * 6.283 + twist;
      ringArr.push(FIG[0] + rad * Math.cos(a1), y, FIG[2] + rad * Math.sin(a1), rd.col[0] * 1.3, rd.col[1] * 1.3, rd.col[2] * 1.3, 1.0,
        FIG[0] + rad * Math.cos(a2), y, FIG[2] + rad * Math.sin(a2), rd.col[0] * 1.3, rd.col[1] * 1.3, rd.col[2] * 1.3, 1.0);
    }
    // a marker node on each ring (label anchor at the near-front)
    ringNodeArr.push(FIG[0] + rad, y, FIG[2], rd.col[0] * 1.7, rd.col[1] * 1.7, rd.col[2] * 1.7, 1.0, 0.75);
    ringNodeArr.push(FIG[0] + rad, y, FIG[2], rd.col[0], rd.col[1], rd.col[2], 0.34, 2.2);
  });
  // vertical thread through the ring centres
  for (let i = 0; i < 4; i++) { const y = FIG[1] - 4.8 + i * 2.4; ringArr.push(FIG[0], y, FIG[2], BRIGHT[0], BRIGHT[1], BRIGHT[2], 0.85, FIG[0], y + 2.4, FIG[2], GOLD[0], GOLD[1], GOLD[2], 0.85); }
  const ringLines = vaoL(new Float32Array(ringArr));
  const ringNodes = vaoP(new Float32Array(ringNodeArr));
  const ringAnchor = RINGS.map((rd, ri) => { const rad = 4.0 - ri * 0.5; return [FIG[0] + rad, FIG[1] - 4.8 + ri * 2.4, FIG[2]]; });

  // ---- FIGURE: substrate helix (labelled), parked screen-right ----
  const SUBS = [
    { nm: 'VisionClaw', col: CYAN }, { nm: 'agentbox', col: PURPLE }, { nm: 'solid-pod-rs', col: EMERALD },
    { nm: 'nostr-rust-forum', col: AMBER }, { nm: 'dreamlab-ai-website', col: CRIMSON }, { nm: 'Ontology Loom', col: GOLD },
  ];
  const helixArr = [], helixPos = [];
  SUBS.forEach((s, i) => {
    const a = i * 1.02, y = FIG[1] - 5.0 + i * 2.1, r = 3.4;
    const p = [FIG[0] + r * Math.cos(a), y, FIG[2] + r * Math.sin(a)];
    helixPos.push(p);
    helixArr.push(p[0], p[1], p[2], s.col[0] * 1.7, s.col[1] * 1.7, s.col[2] * 1.7, 1.0, 0.82);
    helixArr.push(p[0], p[1], p[2], s.col[0], s.col[1], s.col[2], 0.34, 2.5);
  });
  const helixNodes = vaoP(new Float32Array(helixArr));
  const helixSpineArr = [];
  for (let i = 0; i <= 44; i++) { const t = i / 44, y = FIG[1] - 5.0 + t * 11.0; helixSpineArr.push(FIG[0], y, FIG[2], GOLD[0] * 1.2, GOLD[1] * 1.2, GOLD[2] * 1.2, 0.9); }
  const helixSpine = vaoL(new Float32Array(helixSpineArr));
  const helixSpokesArr = [];
  helixPos.forEach((p, i) => { const c = SUBS[i].col; helixSpokesArr.push(FIG[0], p[1], FIG[2], GOLD[0], GOLD[1], GOLD[2], 0.5, p[0], p[1], p[2], c[0] * 1.4, c[1] * 1.4, c[2] * 1.4, 0.85); });
  const helixSpokes = vaoL(new Float32Array(helixSpokesArr));

  // ---- HTML labels ----
  const labelHost = document.createElement('div');
  labelHost.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2;overflow:hidden';
  labelHost.setAttribute('aria-hidden', 'true');
  document.body.appendChild(labelHost);
  function mkLabel(nm, rl) {
    const d = document.createElement('div');
    d.className = 'mesh-label';
    d.innerHTML = `<span class="ml-nm"></span>${rl ? '<span class="ml-rl"></span>' : ''}`;
    d.querySelector('.ml-nm').textContent = nm;
    if (rl) d.querySelector('.ml-rl').textContent = rl;
    labelHost.appendChild(d); return d;
  }
  const ringLabels = RINGS.map((r) => mkLabel(r.nm, r.rl));
  const helixLabels = SUBS.map((s) => mkLabel(s.nm, ''));

  // ---- figures wired to sections ----
  const FIGURES = [
    { sel: '#evolution', lines: ringLines, nodes: ringNodes, anchors: ringAnchor, labels: ringLabels },
    { sel: '#loom', lines: helixSpokes, nodes: helixNodes, anchors: helixPos, labels: helixLabels, extra: helixSpine },
  ];
  let heroTop = 0, heroBot = 1;
  function measure() {
    const hero = document.getElementById('hero');
    if (hero) { heroTop = hero.offsetTop; heroBot = hero.offsetTop + hero.offsetHeight; }
    FIGURES.forEach((f) => { const el = document.querySelector(f.sel); f.top = el ? el.offsetTop : null; f.h = el ? el.offsetHeight : 0; });
  }

  // ---- state ----
  let W = 0, H = 0, VP = IDENT, P = IDENT, mx = 0, my = 0;
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
    P = persp(50 * Math.PI / 180, W / H, 0.1, 200);
    measure();
  }
  resize();
  addEventListener('resize', () => { clearTimeout(resize._t); resize._t = setTimeout(() => { resize(); if (reduced) requestAnimationFrame(frame); }, 150); });
  if (!reduced) addEventListener('pointermove', (e) => { mx = e.clientX / W - 0.5; my = e.clientY / H - 0.5; }, { passive: true });

  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.clearColor(0, 0, 0, 0);
  const dpt = (o, model, alpha, scale) => { gl.useProgram(PT); gl.uniformMatrix4fv(uPT.vp, false, VP); gl.uniformMatrix4fv(uPT.m, false, model); gl.uniform1f(uPT.s, scale); gl.uniform1f(uPT.a, alpha); gl.bindVertexArray(o.vao); gl.drawArrays(gl.POINTS, 0, o.n); };
  const dln = (o, model, alpha, mode) => { gl.useProgram(LN); gl.uniformMatrix4fv(uLN.vp, false, VP); gl.uniformMatrix4fv(uLN.m, false, model); gl.uniform1f(uLN.a, alpha); gl.bindVertexArray(o.vao); gl.drawArrays(mode, 0, o.n); };
  function placeLabel(el, wp, op, rot) {
    if (op < 0.03) { el.style.opacity = '0'; return; }
    // apply the same Y-rotation-about-FIG as the figure model
    const c = Math.cos(rot), s = Math.sin(rot);
    const x = FIG[0] + c * (wp[0] - FIG[0]) + s * (wp[2] - FIG[2]);
    const z = FIG[2] - s * (wp[0] - FIG[0]) + c * (wp[2] - FIG[2]);
    const cl = [VP[0] * x + VP[4] * wp[1] + VP[8] * z + VP[12], VP[1] * x + VP[5] * wp[1] + VP[9] * z + VP[13], VP[2] * x + VP[6] * wp[1] + VP[10] * z + VP[14], VP[3] * x + VP[7] * wp[1] + VP[11] * z + VP[15]];
    if (cl[3] <= 0.1) { el.style.opacity = '0'; return; }
    const sx = (cl[0] / cl[3] * 0.5 + 0.5) * W, sy = (1 - (cl[1] / cl[3] * 0.5 + 0.5)) * H;
    el.style.opacity = String(op);
    el.style.transform = `translate(-6px,-50%) translate(${sx}px,${sy}px)`;
  }

  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000;
    const bounce = reduced ? 1 : clamp(t / 1.5, 0, 1);
    writeStars(bounce);

    // fixed-ish hero camera; content scrolls, figures live in world space to the right
    let eye = [0.6, 1.2, 23], tgt = [0.6, 0.4, 0];
    if (!reduced) { eye = [eye[0] + mx * 1.2, eye[1] - my * 0.8, eye[2]]; eye[1] += Math.sin(t * 0.35) * 0.15; }
    VP = mul(P, lookAt(eye, tgt, [0, 1, 0]));
    const ptScale = 0.5 * canvas.height * P[5];
    const yrot = reduced ? 0 : t * 0.05;

    // hero presence fades out once scrolled past the hero
    const heroFade = clamp(1 - Math.max(0, scrollY - heroTop * 0.2) / (heroBot * 0.75), 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (heroFade > 0.01) {
      dpt(stars, transRotY(0, 0, 0, yrot * 0.4, 0), heroFade, ptScale);
      dln(spine, IDENT, heroFade * 0.5, gl.LINE_STRIP);
      dpt(heroHubs, transRotY(0, 0, 0, yrot, 0), heroFade * 0.9, ptScale);
    } else {
      dpt(stars, transRotY(0, 0, 0, yrot * 0.4, 0), 0.06, ptScale); // faint ambient dust over the video
    }

    // figures: reveal + spin as their section passes the viewport centre
    const mid = scrollY + H * 0.5;
    FIGURES.forEach((f) => {
      f.labels.forEach((l) => (l.style.opacity = '0'));
      if (f.top == null || W < 900) return; // desktop only; mobile hides the figure
      const prog = (mid - f.top) / f.h;         // 0 entering .. 1 leaving
      if (prog < -0.15 || prog > 1.15) return;
      const env = smooth(clamp((prog + 0.15) / 0.3, 0, 1)) * smooth(clamp((1.15 - prog) / 0.3, 0, 1)); // fade in/out
      const spin = reduced ? 0.4 : prog * 2.2;   // scroll drives the spin
      const model = transRotY(FIG[0], 0, FIG[2], spin, 0);
      if (f.extra) dln(f.extra, model, env * 1.6, gl.LINE_STRIP);
      dln(f.lines, model, env * 1.9, gl.LINES);
      dpt(f.nodes, model, env * 1.2, ptScale);
      f.anchors.forEach((a, i) => placeLabel(f.labels[i], a, env, spin));
    });

    if (!reduced) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  setTimeout(measure, 400);
  addEventListener('load', () => setTimeout(measure, 200));
  return { measure };
}
