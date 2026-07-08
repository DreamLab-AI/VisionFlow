// Render presentation/report/diagrams/*.mmd to light-theme SVG (RES-b gate).
//
// Renders each mermaid source in a real headless Chrome using a locally vendored
// mermaid bundle (vendor/mermaid.min.js — no CDN dependency), applying the LIGHT
// `default` theme and baking the browser-computed text fill + a white background
// into every SVG. Output lands in presentation/report/diagrams/rendered/.
//
// Browser selection (first that works):
//   DIAGRAM_CDP_URL   explicit http(s) DevTools endpoint
//   sidecar           browsercontainer:9223 (this environment's GPU sidecar)
//   DIAGRAM_CHROME_BIN launch a local Chrome/Chromium (CI runners)

import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserEndpointFrom, launchLocalChrome, CdpConnection } from './lib/cdp.mjs';
import { preprocessSource } from './lib/preprocess.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');
const SRC_DIR = join(REPO, 'presentation', 'report', 'diagrams');
const OUT_DIR = join(SRC_DIR, 'rendered');
const MERMAID_BUNDLE = join(HERE, 'vendor', 'mermaid.min.js');

// The in-page renderer, kept as a source string injected via Runtime.evaluate.
const IN_PAGE_RENDERER = `
window.__renderDiagram = async function (definition, id) {
  const mermaid = globalThis.mermaid;
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    deterministicIds: true,
    // Top-level htmlLabels:false is what actually forces SVG <text> node labels;
    // the flowchart-scoped key alone is ignored by mermaid. Real <text> is what
    // the text-fill assertion in check-diagram-text.js measures.
    htmlLabels: false,
    flowchart: { htmlLabels: false, useMaxWidth: false },
    themeVariables: { background: '#ffffff', fontFamily: 'sans-serif' },
  });
  const { svg } = await mermaid.render(id, definition);
  const holder = document.createElement('div');
  holder.style.position = 'absolute';
  holder.style.left = '-99999px';
  holder.style.top = '0';
  holder.innerHTML = svg;
  document.body.appendChild(holder);
  const svgEl = holder.querySelector('svg');
  const NS = 'http://www.w3.org/2000/svg';
  // Bake a white background so PDF embedding never depends on container CSS —
  // the transparent-background trap that let dark-theme text vanish on a page.
  const bg = document.createElementNS(NS, 'rect');
  bg.setAttribute('x', '0'); bg.setAttribute('y', '0');
  bg.setAttribute('width', '100%'); bg.setAttribute('height', '100%');
  bg.setAttribute('fill', '#ffffff');
  svgEl.insertBefore(bg, svgEl.firstChild);
  // Bake the true rendered fill onto every text node so the standalone checker
  // reads the visible colour without re-running a browser.
  let total = 0, visible = 0;
  for (const t of svgEl.querySelectorAll('text, tspan')) {
    if (!(t.textContent || '').trim()) continue;
    total++;
    const cs = getComputedStyle(t);
    const fill = cs.fill && cs.fill !== 'none' ? cs.fill : (t.getAttribute('fill') || '#000000');
    t.setAttribute('fill', fill);
    const op = cs.fillOpacity;
    if (op !== '' && Number(op) < 1) t.setAttribute('fill-opacity', op);
    const m = /rgba?\\(([^)]+)\\)/.exec(fill);
    if (m) {
      const [r, g, b] = m[1].split(',').map((n) => parseFloat(n));
      if (!(r >= 240 && g >= 240 && b >= 240)) visible++;
    } else if (fill !== 'none' && fill !== 'transparent' && fill.toLowerCase() !== '#ffffff') {
      visible++;
    }
  }
  const out = new XMLSerializer().serializeToString(svgEl);
  holder.remove();
  return JSON.stringify({ svg: out, total, visible });
};
`;

async function evaluate(conn, sessionId, expression, awaitPromise = false) {
  const result = await conn.send('Runtime.evaluate', {
    expression,
    awaitPromise,
    returnByValue: true,
    userGesture: true,
  }, sessionId);
  if (result.exceptionDetails) {
    const ex = result.exceptionDetails;
    const text = ex.exception?.description || ex.exception?.value || ex.text || 'evaluation failed';
    throw new Error(text);
  }
  return result.result?.value;
}

async function resolveConnection() {
  if (process.env.DIAGRAM_CHROME_BIN) {
    const launched = await launchLocalChrome(process.env.DIAGRAM_CHROME_BIN);
    const conn = await CdpConnection.open(launched.endpoint);
    return { conn, endpoint: launched.endpoint, cleanup: launched.kill, mode: 'launched' };
  }
  const candidates = [];
  if (process.env.DIAGRAM_CDP_URL) candidates.push(process.env.DIAGRAM_CDP_URL);
  candidates.push(`http://${process.env.BROWSER_CDP_HOST || 'browsercontainer'}:${process.env.BROWSER_CDP_PORT || '9223'}`);
  let lastError;
  for (const httpEndpoint of candidates) {
    try {
      const endpoint = await browserEndpointFrom(httpEndpoint);
      const conn = await CdpConnection.open(endpoint);
      return { conn, endpoint, cleanup: () => {}, mode: 'connected' };
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`No reachable browser (tried ${candidates.join(', ')}): ${lastError?.message}`);
}

async function main() {
  if (!existsSync(MERMAID_BUNDLE)) {
    throw new Error(`Vendored mermaid bundle missing: ${MERMAID_BUNDLE}`);
  }
  const sources = readdirSync(SRC_DIR).filter((f) => f.endsWith('.mmd')).sort();
  if (sources.length === 0) throw new Error(`No .mmd sources under ${SRC_DIR}`);
  mkdirSync(OUT_DIR, { recursive: true });

  const mermaidSource = readFileSync(MERMAID_BUNDLE, 'utf8');
  const { conn, endpoint, cleanup, mode } = await resolveConnection();
  console.log(`browser: ${mode} ${endpoint}`);

  const { sessionId, targetId } = await conn.newPage();
  await evaluate(conn, sessionId, mermaidSource);
  const kind = await evaluate(conn, sessionId, 'typeof globalThis.mermaid');
  if (kind !== 'object' && kind !== 'function') {
    throw new Error(`vendored mermaid did not load (typeof mermaid === '${kind}')`);
  }
  await evaluate(conn, sessionId, IN_PAGE_RENDERER);

  let failures = 0;
  for (const file of sources) {
    const name = file.replace(/\.mmd$/, '');
    const raw = readFileSync(join(SRC_DIR, file), 'utf8');
    const definition = preprocessSource(raw);
    const id = `diagram-${name.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const call = `window.__renderDiagram(${JSON.stringify(definition)}, ${JSON.stringify(id)})`;
    try {
      const payload = await evaluate(conn, sessionId, call, true);
      const { svg, total, visible } = JSON.parse(payload);
      const outPath = join(OUT_DIR, `${name}.svg`);
      writeFileSync(outPath, `<?xml version="1.0" encoding="UTF-8"?>\n${svg}\n`);
      console.log(`  rendered ${file} -> rendered/${name}.svg  (text nodes: ${total}, visible: ${visible})`);
    } catch (error) {
      failures++;
      console.error(`  FAILED  ${file}: ${error.message.split('\n')[0]}`);
    }
  }

  await conn.closePage(targetId);
  conn.close();
  cleanup();

  if (failures > 0) {
    console.error(`\n${failures} diagram(s) failed to render.`);
    process.exit(1);
  }
  console.log(`\nRendered ${sources.length} diagram(s) to ${OUT_DIR}`);
}

main().catch((error) => {
  console.error(`render-diagrams: ${error.message}`);
  process.exit(1);
});
