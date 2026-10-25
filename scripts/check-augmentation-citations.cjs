#!/usr/bin/env node
// EXP-AC-001 stabilising check: the compatibility matrix's "Augmentation conditions"
// table has exactly the six condition rows C1–C6, every substrate column maps to a
// known sibling checkout, and every non-`absent` cell cites `path:line` where the
// path is a regular file in that checkout and every cited line exists. A `measured`
// cell must cite code, config or a receipt — never a document.
// Hardened 2026-09-14 after the EDD auditor's five counter-examples (see
// .claude/evidence/EXP-AC-001.evidence.md §Auditor adversarial probes).
// Usage: node scripts/check-augmentation-citations.cjs [--root <dir with sibling checkouts>]
'use strict';
const fs = require('node:fs');
const path = require('node:path');
// Opt-in Jev option (VF-01-S1-01), advisory only: `--jev` or VISIONFLOW_JEV=claim.
// It never changes a FAIL, the summary line or the exit code. See docs/jev-option.md.
const jev = require('./lib/jev.cjs');
const jevQ = require('./lib/jev-questions.cjs');

const root = (() => { const i = process.argv.indexOf('--root'); return i > -1 ? process.argv[i + 1] : path.resolve(__dirname, '..', '..'); })();
const SUBSTRATE_DIRS = { 'nostr-rust-forum': 'nostr-rust-forum', agentbox: 'agentbox', VisionClaw: 'project' };
const CONDITIONS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'];
// What a `measured` grade may cite: source, config, schema, migration, or a receipt.
const EVIDENCE_EXT = new Set(['.rs', '.js', '.cjs', '.mjs', '.ts', '.tsx', '.sql', '.toml', '.json', '.sh', '.yml', '.yaml', '.nix']);
const CITE_RE = /`([^`\s]+?):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)`/g;

const matrix = fs.readFileSync(path.resolve(__dirname, '..', 'docs/architecture/compatibility-matrix.md'), 'utf8');
let failures = 0, checked = 0;
const wantJev = jev.enabled('claim', { env: process.env, cli: jev.cliFlag(process.argv.slice(2)) });
const jevCells = []; // non-absent cells with their cited excerpts, for the advisory pass
const fail = (msg) => { console.error(`FAIL ${msg}`); failures++; };

const start = matrix.indexOf('## Augmentation conditions');
if (start < 0) { fail('no "Augmentation conditions" section'); process.exit(1); }
const nextHeading = matrix.indexOf('\n## ', start + 5);
const section = matrix.slice(start, nextHeading < 0 ? undefined : nextHeading);
const lines = section.split('\n');
const headerIdx = lines.findIndex(l => /^\| Condition \|/.test(l));
if (headerIdx < 0) { fail('table header missing'); process.exit(1); }

// Columns: every header must be a known substrate and appear once.
const cols = lines[headerIdx].split('|').map(s => s.trim()).filter(Boolean).slice(1);
const seen = new Set();
for (const c of cols) {
  if (!SUBSTRATE_DIRS[c]) fail(`unknown substrate column "${c}" (known: ${Object.keys(SUBSTRATE_DIRS).join(', ')})`);
  if (seen.has(c)) fail(`duplicate substrate column "${c}"`);
  seen.add(c);
}

// Body: contiguous table rows after the separator line; every row label must be C1–C6, each once.
const body = [];
for (let i = headerIdx + 2; i < lines.length && /^\|/.test(lines[i]); i++) body.push(lines[i]);
const labels = body.map(r => (r.split('|')[1] || '').trim().split(/\s+/)[0]);
for (const l of labels) if (!CONDITIONS.includes(l)) fail(`row label "${l}" is not one of ${CONDITIONS.join(', ')}`);
for (const c of CONDITIONS) { const n = labels.filter(l => l === c).length; if (n !== 1) fail(`expected exactly one ${c} row, found ${n}`); }
if (body.length !== 6) fail(`expected 6 condition rows, found ${body.length}`);

const lineCount = (file) => fs.readFileSync(file, 'utf8').split('\n').length;
for (const row of body) {
  const cells = row.split('|').map(s => s.trim()).filter(Boolean);
  const cond = cells[0];
  if (cells.length - 1 !== cols.length) fail(`${cond}: ${cells.length - 1} cells for ${cols.length} substrate columns`);
  cells.slice(1).forEach((cell, i) => {
    const substrate = cols[i];
    const dir = SUBSTRATE_DIRS[substrate];
    const status = (cell.match(/`(absent|partial|measured)`/) || [])[1];
    if (!status) { fail(`${cond}/${substrate}: no status token`); return; }
    const cites = [...cell.matchAll(CITE_RE)].map(m => ({ file: m[1], lines: m[2] }));
    if (status !== 'absent' && cites.length === 0) fail(`${cond}/${substrate}: ${status} without a citation`);
    const excerpts = [];
    for (const { file, lines: ln } of cites) {
      checked++;
      if (file.includes('...')) { fail(`${cond}/${substrate}: elided path ${file}`); continue; }
      const ext = path.extname(file).toLowerCase();
      if (status === 'measured' && !EVIDENCE_EXT.has(ext)) { fail(`${cond}/${substrate}: measured must cite code/config/receipt, not "${file}"`); }
      if (!dir) continue;
      const full = path.join(root, dir, file);
      let st; try { st = fs.statSync(full); } catch { fail(`${cond}/${substrate}: missing ${full}`); continue; }
      if (!st.isFile()) { fail(`${cond}/${substrate}: not a regular file ${full}`); continue; }
      const total = lineCount(full);
      const maxLine = Math.max(...ln.split(',').flatMap(r => r.split('-').map(Number)));
      if (maxLine > total) fail(`${cond}/${substrate}: ${file}:${ln} cites line ${maxLine} but the file has ${total} lines`);
      else if (wantJev) {
        const fl = fs.readFileSync(full, 'utf8').split('\n');
        for (const part of ln.split(',')) {
          const [a, b] = part.split('-').map(Number);
          const text = jevQ.excerpt(fl, a, b);
          if (text) excerpts.push({ ref: `${file}:${part}`, text });
        }
      }
    }
    if (status !== 'absent' && excerpts.length) jevCells.push({ key: `${cond}/${substrate}`, text: cell.replace(/\s+/g, ' ').slice(0, 900), status, excerpts: excerpts.slice(0, 6) });
  });
}
console.log(`${checked} citations checked across ${body.length} rows × ${cols.length} substrates; ${failures} failure(s)`);

/**
 * The Jev pass: runs only when asked for, after the deterministic gate has
 * printed its verdict, and can only append `JEV` lines. Excerpts of sibling
 * checkouts leave the process, so this consumer is local-only: it needs a
 * declared-local endpoint (AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL=1 or an explicit
 * VISIONFLOW_JEV_ENDPOINT). Any fallback prints nothing more.
 */
async function jevPass(deps = {}) {
  if (!wantJev || !jevCells.length) return;
  const q = jevQ.claimCheck(jevCells);
  const r = await jev.decide({ ...q, dataClass: 'local-only', env: process.env, ...deps });
  if (r.outcome !== 'accepted') return;
  for (const { id, key } of q.ids) {
    const a = r.answers[id];
    if (a.choice !== 'supported') {
      console.log(`JEV claim ${key}: ${a.choice} conf=${a.confidence.toFixed(2)} (advisory; reviewer disposition and exact citations required; no status is changed)`);
    }
  }
}

jevPass().catch(() => {}).finally(() => process.exit(failures ? 1 : 0));
