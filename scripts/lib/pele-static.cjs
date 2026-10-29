'use strict';

// Static Pele rendering for files that leave the browser: committed SVG
// baselines, PDF embedding, the diagram corpus' rendered/ tree and its
// --report consumers. Pele colours everything through CSS custom properties
// (`var(--pele-fg,#1f1f1f)` and its own `--_fg` aliases); a browser resolves
// them, but a PDF embedder or image converter may not, which is the
// invisible-text defect the RES-b gate exists to stop. flatten() replaces every
// variable with the concrete value Pele itself declares as the default, so the
// SVG renders identically with no stylesheet at all.

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');

const VENDOR = path.join(__dirname, '..', 'vendor', 'pele');

// Options for a gate render: fixed pixel size and the authored direction, so a
// width is a property of the diagram, not of the viewport it happened to meet.
const GATE_OPTIONS = Object.freeze({ responsive: false, autoDirection: false });

let peleModule = null;

/** Import the vendored Pele build after checking it against VERSION's sha256. */
async function loadPele() {
  if (peleModule) return peleModule;
  const file = path.join(VENDOR, 'pele.js');
  const pinned = /^sha256\s+([0-9a-f]{64})$/m.exec(fs.readFileSync(path.join(VENDOR, 'VERSION'), 'utf8'));
  if (!pinned) throw new Error(`${VENDOR}/VERSION carries no sha256 line`);
  const actual = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  if (actual !== pinned[1]) throw new Error(`vendored pele.js sha256 ${actual} does not match VERSION ${pinned[1]}`);
  peleModule = await import(pathToFileURL(file).href);
  return peleModule;
}

/** Replace every Pele CSS variable with its declared default. */
function flatten(svg, { background = null } = {}) {
  let out = svg.replace(/var\(--pele-[\w-]+,([^()]*?)\)/g, (_, fallback) => fallback.trim());
  const root = /^<svg\b[^>]*\bstyle="([^"]*)"/.exec(out);
  const local = new Map();
  if (root) for (const m of root[1].matchAll(/(--_[\w-]+):([^;]*)/g)) local.set(m[1], m[2].trim());
  for (let pass = 0; pass < 4 && out.includes('var(--_'); pass++) {
    out = out.replace(/var\((--_[\w-]+)\)/g, (whole, name) => (local.has(name) ? local.get(name) : whole));
  }
  if (out.includes('var(')) throw new Error(`unresolved CSS variable left in SVG: ${/var\([^)]*\)/.exec(out)[0]}`);
  if (background) out = out.replace(/^(<svg\b[^>]*>)/, `$1<rect x="0" y="0" width="100%" height="100%" fill="${background}"/>`);
  return out;
}

/** Render one Mermaid source to a static SVG. Throws PeleError on bad source. */
function renderStatic(pele, source, { background = null, options = {} } = {}) {
  const r = pele.render(source, { ...GATE_OPTIONS, ...options });
  return { svg: flatten(r.svg, { background }), width: Math.round(r.width), height: Math.round(r.height), type: r.mermaidType };
}

/** One line for a render failure: the Mermaid line and its caret snippet. */
function describeError(e) {
  if (e && e.code) return `pele ${e.code}${e.line ? ` at mermaid line ${e.line}` : ''}: ${String(e.message).split('\n')[0]}${e.snippet ? ` | ${e.snippet.split('\n').join(' | ')}` : ''}`;
  return String(e && e.message || e);
}

module.exports = { GATE_OPTIONS, loadPele, flatten, renderStatic, describeError };
