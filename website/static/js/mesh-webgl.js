// VisionFlow — living-mesh WebGL2 backdrop.
//
// One fixed, full-viewport canvas that renders the Dynamic Agentic Mesh as an
// ambient, textless field of light: an identity-spine, six substrate hubs on a
// helix, a lattice of the knowledge graph, a governance signing-pulse, and the
// Loom's grounding convergence. A scroll-driven camera flies the narrative —
// its keyframes anchored to the real content sections, so the visual tracks the
// copy that scrolls over it. All colour comes from the site's own palette.
//
// Progressive enhancement: returns silently without WebGL2, and renders a
// single calm frame (no rAF loop) under prefers-reduced-motion. The page's
// content is complete without it.

export function initMesh(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: false });
  if (!gl) return null;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- palette (site tokens, linear rgb 0..1) ----
  const GOLD = [0.831, 0.647, 0.455];
  const BRIGHT = [1.0, 0.843, 0.10];
  const BRONZE = [0.804, 0.498, 0.196];
  const CYAN = [0.0, 0.831, 1.0];
  const PURPLE = [0.545, 0.361, 0.965];
  const EMERALD = [0.063, 0.725, 0.506];
  const AMBER = [0.961, 0.620, 0.043];
  const CRIMSON = [0.914, 0.271, 0.376];
  const SNOW = [0.86, 0.84, 0.90];
  // Six substrate hub colours, matching the substrate-card --accent order.
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
  const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]; };
  const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const lerp = (a, b, t) => a + (b - a) * t;
  const l3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const smooth = (t) => t * t * (3 - 2 * t);
  const near = (x, c, w) => clamp(1 - Math.abs(x - c) / (w || 0.75), 0, 1);
  const R = (s) => { const x = Math.sin(s * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

  // ---- shaders ----
  const compile = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const program = (vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; };

  const PT = program(
    `#version 300 es
    layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol; layout(location=2) in float aSize;
    uniform mat4 uVP, uModel; uniform float uScale, uAlpha; out vec4 vCol;
    void main(){ vec4 wp = uModel*vec4(aPos,1.0); gl_Position = uVP*wp; float w = max(gl_Position.w,.1);
    gl_PointSize = clamp(aSize*uScale/w, 1.0, 170.0); vCol = vec4(aCol.rgb, aCol.a*uAlpha); }`,
    `#version 300 es
    precision mediump float; in vec4 vCol; out vec4 o;
    void main(){ vec2 d = gl_PointCoord-.5; float r = length(d)*2.0;
    float a = exp(-r*r*4.5)*smoothstep(1.0,.55,r); o = vec4(vCol.rgb*a*vCol.a, a*vCol.a); }`);
  const LN = program(
    `#version 300 es
    layout(location=0) in vec3 aPos; layout(location=1) in vec4 aCol;
    uniform mat4 uVP, uModel; uniform float uAlpha; out vec4 vCol;
    void main(){ gl_Position = uVP*uModel*vec4(aPos,1.0); vCol = vec4(aCol.rgb, aCol.a*uAlpha); }`,
    `#version 300 es
    precision mediump float; in vec4 vCol; out vec4 o; void main(){ o = vec4(vCol.rgb*vCol.a, vCol.a); }`);
  const uPT = { vp: gl.getUniformLocation(PT, 'uVP'), m: gl.getUniformLocation(PT, 'uModel'), s: gl.getUniformLocation(PT, 'uScale'), a: gl.getUniformLocation(PT, 'uAlpha') };
  const uLN = { vp: gl.getUniformLocation(LN, 'uVP'), m: gl.getUniformLocation(LN, 'uModel'), a: gl.getUniformLocation(LN, 'uAlpha') };

  // ---- buffers ----
  function vaoPoints(data, dyn) {
    const v = gl.createVertexArray(); gl.bindVertexArray(v);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, dyn ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 32, 28);
    return { vao: v, buf: b, n: data.length / 8 };
  }
  function vaoLines(data, dyn) {
    const v = gl.createVertexArray(); gl.bindVertexArray(v);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, dyn ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    return { vao: v, buf: b, n: data.length / 7 };
  }

  // ---- geometry: ambient dust ----
  const dustArr = [];
  for (let i = 0; i < 620; i++) {
    const th = R(i) * 6.283, ph = Math.acos(2 * R(i + 900) - 1), rr = 20 + R(i + 1800) * 30;
    const c = R(i + 40) < 0.28 ? l3(GOLD, BRIGHT, R(i + 5)) : (R(i + 41) < 0.5 ? BRONZE : SNOW);
    dustArr.push(rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph) * 0.7, rr * Math.sin(ph) * Math.sin(th), c[0], c[1], c[2], 0.30, 0.05 + R(i + 7) * 0.05);
  }
  const dust = vaoPoints(new Float32Array(dustArr));

  // ---- geometry: knowledge lattice ----
  const NODES = 250, npos = [], nodeArr = [];
  for (let i = 0; i < NODES; i++) {
    const th = R(i * 3 + 1) * 6.283, ph = Math.acos(2 * R(i * 3 + 2) - 1), rr = 6 + Math.pow(R(i * 3 + 3), 0.7) * 8.5;
    const p = [rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph) * 0.85, rr * Math.sin(ph) * Math.sin(th)];
    npos.push(p);
    const c = R(i + 55) < 0.14 ? l3(GOLD, BRIGHT, R(i + 8)) : (R(i + 56) < 0.28 ? BRONZE : GOLD);
    const b = 0.75 + R(i + 9) * 0.5;
    nodeArr.push(p[0], p[1], p[2], c[0] * b, c[1] * b, c[2] * b, 0.85, 0.10 + R(i + 13) * 0.08);
  }
  const latNodes = vaoPoints(new Float32Array(nodeArr));
  const edgeArr = [];
  for (let i = 0; i < NODES; i++) {
    let d1 = 1e9, d2 = 1e9, j1 = -1, j2 = -1;
    for (let j = 0; j < NODES; j++) {
      if (j === i) continue;
      const dx = npos[i][0] - npos[j][0], dy = npos[i][1] - npos[j][1], dz = npos[i][2] - npos[j][2], dd = dx * dx + dy * dy + dz * dz;
      if (dd < d1) { d2 = d1; j2 = j1; d1 = dd; j1 = j; } else if (dd < d2) { d2 = dd; j2 = j; }
    }
    [j1, j2].forEach((j) => { if (j < 0) return; edgeArr.push(npos[i][0], npos[i][1], npos[i][2], GOLD[0], GOLD[1], GOLD[2], 0.10, npos[j][0], npos[j][1], npos[j][2], BRONZE[0], BRONZE[1], BRONZE[2], 0.02); });
  }
  const latEdges = vaoLines(new Float32Array(edgeArr));

  // ---- geometry: identity spine ----
  const spineArr = [];
  for (let i = 0; i <= 64; i++) { const t = i / 64, y = -9 + t * 18, x = Math.sin(t * 3.1) * 0.8, z = Math.cos(t * 2.4) * 0.8; const c = l3(BRIGHT, GOLD, Math.abs(t - 0.5) * 2); spineArr.push(x, y, z, c[0], c[1], c[2], 0.55); }
  const spine = vaoLines(new Float32Array(spineArr));

  // ---- geometry: six substrate hubs on a helix ----
  const hubPos = [];
  for (let i = 0; i < 6; i++) { const a = i * 1.05 + 0.5; hubPos.push([7 * Math.cos(a), -6 + i * 2.4, 7 * Math.sin(a)]); }
  const hubArr = [];
  hubPos.forEach((p, i) => { const c = HUBCOL[i]; hubArr.push(p[0], p[1], p[2], c[0] * 1.6, c[1] * 1.6, c[2] * 1.6, 1.0, 0.66); hubArr.push(p[0], p[1], p[2], c[0], c[1], c[2], 0.32, 2.1); });
  const hubs = vaoPoints(new Float32Array(hubArr));
  const tethArr = [];
  hubPos.forEach((p, i) => {
    const c = HUBCOL[i];
    const t = (p[1] + 9) / 18, spx = [Math.sin(t * 3.1) * 0.8, p[1], Math.cos(t * 2.4) * 0.8];
    tethArr.push(p[0], p[1], p[2], c[0], c[1], c[2], 0.40, spx[0], spx[1], spx[2], c[0], c[1], c[2], 0.06);
    const best = [], bd = [];
    for (let j = 0; j < NODES; j++) { const dx = p[0] - npos[j][0], dy = p[1] - npos[j][1], dz = p[2] - npos[j][2], dd = dx * dx + dy * dy + dz * dz; if (best.length < 4) { best.push(j); bd.push(dd); } else { let mi = 0; for (let k = 1; k < 4; k++) if (bd[k] > bd[mi]) mi = k; if (dd < bd[mi]) { best[mi] = j; bd[mi] = dd; } } }
    best.forEach((j) => { tethArr.push(p[0], p[1], p[2], c[0], c[1], c[2], 0.16, npos[j][0], npos[j][1], npos[j][2], c[0], c[1], c[2], 0.02); });
  });
  const tethers = vaoLines(new Float32Array(tethArr));

  // ---- geometry: governance signing loop ----
  const agentP = hubPos[1], forumP = hubPos[3], clawP = hubPos[0];
  const humanP = [forumP[0] * 1.18, forumP[1] + 1.0, forumP[2] * 1.18];
  const goldPt = vaoPoints(new Float32Array([
    humanP[0], humanP[1], humanP[2], BRIGHT[0] * 1.5, BRIGHT[1] * 1.5, BRIGHT[2] * 1.5, 1.0, 0.44,
    humanP[0], humanP[1], humanP[2], BRIGHT[0], BRIGHT[1], BRIGHT[2], 0.3, 1.4,
  ]));
  const govArr = [];
  [[agentP, forumP], [forumP, humanP], [humanP, forumP], [forumP, clawP]].forEach((s) => { govArr.push(s[0][0], s[0][1], s[0][2], BRIGHT[0], BRIGHT[1], BRIGHT[2], 0.14, s[1][0], s[1][1], s[1][2], BRIGHT[0], BRIGHT[1], BRIGHT[2], 0.14); });
  const govLines = vaoLines(new Float32Array(govArr));
  const pulseData = new Float32Array([0, 0, 0, 1, 1, 1, 1.0, 0.5]);
  const pulse = vaoPoints(pulseData, true);

  // ---- geometry: Loom grounding core ----
  const loomCore = vaoPoints(new Float32Array([
    0, 0.5, 0, BRIGHT[0] * 1.6, BRIGHT[1] * 1.6, BRIGHT[2] * 1.6, 1.0, 0.7,
    0, 0.5, 0, GOLD[0], GOLD[1], GOLD[2], 0.32, 2.4,
  ]));

  // ---- scroll-anchored camera ----
  // Each anchor pairs a section selector with a camera pose. The active scene is
  // whichever pair the viewport centre sits between; the camera interpolates.
  const ANCHORS = [
    { sel: '#hero', eye: [1.2, 1.6, 23], tgt: [0, 0.4, 0], scene: 'hero' },
    { sel: '#substrates', eye: [13, 3.5, 15], tgt: [0, 1, 0], scene: 'substrates' },
    { sel: '#broker', eye: [-8.5, 1.5, 14], tgt: [-0.5, -1, 1], scene: 'governance' },
    { sel: '#loom', eye: [0, 3, 13], tgt: [0, 0.5, 0], scene: 'loom' },
    { sel: '#cases', eye: [0, 5, 34], tgt: [0, 0, 0], scene: 'calm' },
    { sel: '#repos', eye: [2, 4, 40], tgt: [0, 1, 0], scene: 'calm' },
  ];
  let anchors = [];
  function measure() {
    anchors = ANCHORS.map((a) => { const el = document.querySelector(a.sel); return el ? { ...a, top: el.offsetTop + el.offsetHeight * 0.5 } : null; }).filter(Boolean);
  }

  // scene intensity weights, blended from the two bracketing anchors
  function sceneWeights(i, f) {
    const w = { hero: 0, substrates: 0, governance: 0, loom: 0, calm: 0 };
    if (anchors[i]) w[anchors[i].scene] += 1 - f;
    if (anchors[i + 1]) w[anchors[i + 1].scene] += f;
    return w;
  }

  // ---- state ----
  let W = 0, H = 0, VP = IDENT, P = IDENT, mx = 0, my = 0;
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
    P = persp(52 * Math.PI / 180, W / H, 0.1, 200);
    measure();
  }
  resize();
  addEventListener('resize', () => {
    clearTimeout(resize._t);
    resize._t = setTimeout(() => { resize(); if (reduced) requestAnimationFrame(frame); }, 150);
  });
  if (!reduced) addEventListener('pointermove', (e) => { mx = e.clientX / W - 0.5; my = e.clientY / H - 0.5; }, { passive: true });

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE); // additive — light on the dark page ground
  gl.clearColor(0, 0, 0, 0);

  function drawPts(o, model, alpha, scale) { gl.useProgram(PT); gl.uniformMatrix4fv(uPT.vp, false, VP); gl.uniformMatrix4fv(uPT.m, false, model); gl.uniform1f(uPT.s, scale); gl.uniform1f(uPT.a, alpha); gl.bindVertexArray(o.vao); gl.drawArrays(gl.POINTS, 0, o.n); }
  function drawLin(o, model, alpha, mode) { gl.useProgram(LN); gl.uniformMatrix4fv(uLN.vp, false, VP); gl.uniformMatrix4fv(uLN.m, false, model); gl.uniform1f(uLN.a, alpha); gl.bindVertexArray(o.vao); gl.drawArrays(mode, 0, o.n); }

  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000;
    // Guard: if the anchor sections are all absent, don't dereference them.
    if (!anchors.length) { if (!reduced) requestAnimationFrame(frame); return; }
    // camera from scroll: viewport centre in document space
    const mid = scrollY + H * 0.5;
    let i = 0;
    for (let k = 0; k < anchors.length - 1; k++) { if (mid >= anchors[k].top) i = k; }
    const a = anchors[i], b = anchors[Math.min(i + 1, anchors.length - 1)];
    const span = Math.max(b.top - a.top, 1);
    const f = smooth(clamp((mid - a.top) / span, 0, 1));
    let eye = l3(a.eye, b.eye, f), tgt = l3(a.tgt, b.tgt, f);
    if (!reduced) { eye = [eye[0] + mx * 1.5, eye[1] - my * 1.0, eye[2]]; eye[1] += Math.sin(t * 0.35) * 0.18; }
    VP = mul(P, lookAt(eye, tgt, [0, 1, 0]));
    const ptScale = 0.5 * canvas.height * P[5];
    const rot = reduced ? IDENT : rotY(t * 0.018);
    const w = sceneWeights(i, f);
    const bright = w.hero + w.substrates + w.governance + w.loom; // 1 in narrative, →0 in calm
    const dim = 0.20 + bright * 0.80;

    gl.clear(gl.COLOR_BUFFER_BIT);
    drawPts(dust, rot, 0.45 + w.calm * 0.15, ptScale);
    drawLin(latEdges, rot, (0.42 + w.hero * 0.68 + w.calm * 0.25) * dim, gl.LINES);
    drawPts(latNodes, rot, (0.5 + w.hero * 0.6 + w.calm * 0.3) * dim, ptScale);
    drawLin(spine, IDENT, 0.16 + w.hero * 0.5 + (w.substrates + w.governance) * 0.55 + w.loom * 0.3, gl.LINE_STRIP);
    drawLin(tethers, IDENT, 0.12 + w.hero * 0.35 + w.substrates * 0.9 + w.governance * 0.4, gl.LINES);
    drawPts(hubs, IDENT, 0.28 + w.substrates * 0.72 + w.governance * 0.55 + w.loom * 0.4 + w.hero * 0.9, ptScale);

    // governance pulse loop
    drawLin(govLines, IDENT, w.governance * 0.8, gl.LINES);
    let goldFlash = 0.3;
    if (!reduced && w.governance > 0.02) {
      const cyc = t % 6; let pp, pa = 1;
      if (cyc < 2) pp = l3(agentP, forumP, smooth(cyc / 2));
      else if (cyc < 3) { pp = forumP; goldFlash = 0.5 + Math.sin((cyc - 2) * Math.PI) * 1.5; pa = 0.4; }
      else if (cyc < 5) pp = l3(forumP, clawP, smooth((cyc - 3) / 2));
      else { pp = clawP; pa = 1 + Math.sin((cyc - 5) * Math.PI) * 1.6; }
      pulseData[0] = pp[0]; pulseData[1] = pp[1]; pulseData[2] = pp[2];
      gl.bindBuffer(gl.ARRAY_BUFFER, pulse.buf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pulseData);
      drawPts(pulse, IDENT, w.governance, ptScale);
    }
    drawPts(goldPt, IDENT, (0.08 + w.governance * 0.92) * goldFlash, ptScale);

    // Loom grounding core — pulses as the mesh converges to one served truth
    if (w.loom > 0.02) {
      const puls = reduced ? 1 : 0.8 + Math.sin(t * 1.6) * 0.25;
      drawPts(loomCore, IDENT, w.loom * puls, ptScale);
    }

    if (!reduced) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // one extra measure after fonts/layout settle
  setTimeout(measure, 400);
  addEventListener('load', () => setTimeout(measure, 200));
  return { measure };
}
