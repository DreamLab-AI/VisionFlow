#!/usr/bin/env node
// @ts-check
'use strict';

/**
 * drift-counter — the canon's self-description drift gate (RES-d).
 *
 * PRD-gap-close-canon.md §"Automated Self-Description Counter (RES-d)";
 * ADR-005 §Decision 2 ("one canon counter, substrate-exposed sources, four
 * axes"); DDD-gap-close-canon-context §9 (`DriftCounter` service).
 *
 * The canon owns the counter and the CI gate; the substrates expose the count
 * sources. This script reads each counted axis from its substrate-exposed
 * source of truth, then checks every self-description figure the canon asserts
 * against that truth. It fails (exit 1) when a canon figure disagrees with its
 * queried source, or when a second distinct figure for an axis appears at a
 * policed site.
 *
 * Four axes (ADR-005 §Decision 2):
 *   - skills                → agentbox scripts/skill-count-check.js (invoked)
 *   - mcp-ontology-tools    → agentbox mcp/servers/ontology-bridge.js TOOLS length
 *   - ontology-classes      → VisionClaw ClassCountSource (published count file / env)
 *   - roster                → forum agent_registry (not yet single-sourced → planned)
 *
 * Partial-source failure mode (ADR-005 §Decision 2 tradeoff; DDD-gap-close-
 * visionclaw-context §"RES-d source-of-truth split"): an axis whose source is
 * not exposed this wave is reported UNAVAILABLE and is not enforced — a source
 * being down blocks *that* axis, it does not turn the whole gate red. A figure
 * that *disagrees* with an *available* source is a hard failure. `--strict`
 * flips unavailability into a failure (for a wave where every source must be
 * live).
 *
 * The gate is allowlist-anchored, not a blind tree grep, because the tree
 * legitimately carries distinct "N skills" / "N MCP tools" figures about other
 * subjects (a Ramp/Glass case study's "350 skills"; VisionClaw's own native
 * "7 MCP tools"; agentbox's "180+ MCP tools" total). Only the sites in
 * allowlist.json describe a tracked axis and are policed.
 *
 * Usage:
 *   node scripts/drift-counter/drift-counter.mjs           # report + exit code
 *   node scripts/drift-counter/drift-counter.mjs --json     # JSON only
 *   node scripts/drift-counter/drift-counter.mjs --quiet    # exit code only
 *   node scripts/drift-counter/drift-counter.mjs --strict   # unavailable axis fails
 *
 * Env:
 *   DRIFT_AGENTBOX_DIR                agentbox checkout (skills + mcp sources)
 *   DRIFT_VISIONCLAW_CLASS_COUNT      integer ontology class count (source override)
 *   DRIFT_VISIONCLAW_CLASS_COUNT_FILE path to a committed count file (single integer)
 */

import { readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');
const ALLOWLIST = JSON.parse(readFileSync(join(__dirname, 'allowlist.json'), 'utf8'));

const argv = new Set(process.argv.slice(2));
const OUT_JSON = argv.has('--json');
const QUIET = argv.has('--quiet');
const STRICT = argv.has('--strict');

/** Resolve the agentbox checkout that exposes the skill + mcp count sources. */
function resolveAgentbox() {
  const candidates = [
    process.env.DRIFT_AGENTBOX_DIR,
    join(REPO_ROOT, '_agentbox'),
    join(REPO_ROOT, 'agentbox'),
    join(REPO_ROOT, '..', 'agentbox'),
    '/home/devuser/workspace/project/agentbox',
  ].filter(Boolean);
  for (const c of candidates) {
    if (c && existsSync(join(c, 'mcp', 'servers', 'ontology-bridge.js'))) return c;
  }
  return null;
}

/** Skills axis — invoke agentbox's own single-source-of-truth counter. */
function querySkills(agentbox) {
  if (!agentbox) return { status: 'unavailable', reason: 'agentbox checkout not found (set DRIFT_AGENTBOX_DIR)' };
  const script = join(agentbox, 'scripts', 'skill-count-check.js');
  if (!existsSync(script)) return { status: 'unavailable', reason: `skill-count-check.js not found in ${agentbox}` };
  try {
    const out = execFileSync('node', [script], { cwd: agentbox, encoding: 'utf8' });
    const parsed = JSON.parse(out);
    if (typeof parsed.count !== 'number') throw new Error('no numeric .count in output');
    return { status: 'available', truth: parsed.count, source: `${script} (.count)` };
  } catch (err) {
    // exit 1 from the agentbox script means *it* found a divergence in agentbox's
    // own tree; the JSON is still on stdout, recover the count from it.
    const stdout = /** @type {any} */ (err).stdout;
    if (stdout) {
      try {
        const parsed = JSON.parse(String(stdout));
        if (typeof parsed.count === 'number') {
          return { status: 'available', truth: parsed.count, source: `${script} (.count; agentbox self-check reported drift)` };
        }
      } catch { /* fall through */ }
    }
    return { status: 'unavailable', reason: `skill-count-check.js failed: ${err.message}` };
  }
}

/**
 * MCP-ontology-bridge axis — count the elements of the `const TOOLS = [ … ];`
 * registry in ontology-bridge.js. Top-level elements are either object
 * literals (`  {`) or bare imported references (`  propose.ONTOLOGY_PROPOSE_TOOL,`);
 * nested inputSchema objects are indented deeper than two spaces and are not
 * counted.
 */
function queryMcpTools(agentbox) {
  if (!agentbox) return { status: 'unavailable', reason: 'agentbox checkout not found (set DRIFT_AGENTBOX_DIR)' };
  const file = join(agentbox, 'mcp', 'servers', 'ontology-bridge.js');
  if (!existsSync(file)) return { status: 'unavailable', reason: `ontology-bridge.js not found in ${agentbox}` };
  const lines = readFileSync(file, 'utf8').split(/\r?\n/);
  let inBlock = false;
  let count = 0;
  for (const line of lines) {
    if (!inBlock) {
      if (/^const TOOLS = \[/.test(line)) inBlock = true;
      continue;
    }
    if (/^\];/.test(line)) break;
    if (/^ {2}\{/.test(line)) count++;                                // object-literal element
    else if (/^ {2}[A-Za-z_$][A-Za-z0-9_$.]*,?\s*$/.test(line)) count++; // bare reference element
  }
  if (count === 0) return { status: 'unavailable', reason: 'TOOLS array not found or empty in ontology-bridge.js' };
  return { status: 'available', truth: count, source: `${file} (TOOLS.length)` };
}

/** Ontology-classes axis — VisionClaw ClassCountSource (env or committed count file). */
function queryOntologyClasses() {
  const env = process.env.DRIFT_VISIONCLAW_CLASS_COUNT;
  if (env && /^\d+$/.test(env.trim())) {
    return { status: 'available', truth: Number(env.trim()), source: 'DRIFT_VISIONCLAW_CLASS_COUNT (env)' };
  }
  const f = process.env.DRIFT_VISIONCLAW_CLASS_COUNT_FILE;
  if (f && existsSync(f)) {
    const m = readFileSync(f, 'utf8').match(/\d[\d,]*/);
    if (m) return { status: 'available', truth: Number(m[0].replace(/,/g, '')), source: `${f} (committed count file)` };
  }
  return {
    status: 'unavailable',
    reason: 'VisionClaw ClassCountSource is `planned`, not published as a script-queryable committed count this wave '
      + '(ddd-gap-close-visionclaw-context.md:155). Partial-source mode: axis not enforced (ADR-005 §Decision 2).',
  };
}

/** Scan mode — every occurrence of the axis pattern in each file must equal truth. */
function checkScan(axis, truth, spec) {
  const findings = [];
  const re = new RegExp(spec.pattern, 'g');
  for (const rel of spec.files) {
    const abs = join(REPO_ROOT, rel);
    if (!existsSync(abs)) { findings.push({ axis, file: rel, ok: false, kind: 'file-missing' }); continue; }
    const raw = readFileSync(abs, 'utf8');
    const lines = raw.split(/\r?\n/);
    let any = false;
    lines.forEach((line, i) => {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(line)) !== null) {
        any = true;
        const stated = Number(m[1]);
        findings.push({ axis, file: rel, line: i + 1, stated, truth, ok: stated === truth, text: line.trim().slice(0, 120) });
      }
    });
    if (!any) findings.push({ axis, file: rel, ok: false, kind: 'no-match', note: 'policed file carries no tracked figure — pattern moved?' });
  }
  return findings;
}

/** Sites mode — each pinned site's pattern must match once and equal truth. */
function checkSites(axis, truth, spec) {
  const findings = [];
  const byFile = new Map();
  for (const site of spec.sites) {
    if (!byFile.has(site.file)) {
      const abs = join(REPO_ROOT, site.file);
      byFile.set(site.file, existsSync(abs) ? readFileSync(abs, 'utf8') : null);
    }
    const raw = byFile.get(site.file);
    if (raw == null) { findings.push({ axis, file: site.file, ok: false, kind: 'file-missing' }); continue; }
    const re = new RegExp(site.pattern);
    const m = re.exec(raw);
    if (!m) { findings.push({ axis, file: site.file, ok: false, kind: 'site-missing', note: `pinned site pattern did not match: /${site.pattern}/` }); continue; }
    const stated = Number(m[1]);
    const line = raw.slice(0, m.index).split(/\r?\n/).length;
    findings.push({ axis, file: site.file, line, stated, truth, ok: stated === truth, text: m[0].slice(0, 120) });
  }
  return findings;
}

function run() {
  const agentbox = resolveAgentbox();
  const sources = {
    'skills': querySkills(agentbox),
    'mcp-ontology-tools': queryMcpTools(agentbox),
    'ontology-classes': queryOntologyClasses(),
  };

  const axisReports = [];
  let hardFail = false;

  for (const [axis, spec] of Object.entries(ALLOWLIST.axes)) {
    // Roster: declared planned in the allowlist, no source, no sites.
    if (spec.status === 'planned') {
      axisReports.push({ axis, state: 'planned', reason: spec.reason, findings: [] });
      continue;
    }
    const src = sources[axis] || { status: 'unavailable', reason: 'no source query registered' };
    if (src.status !== 'available') {
      if (STRICT) hardFail = true;
      axisReports.push({ axis, state: 'unavailable', reason: src.reason, enforced: false, findings: [] });
      continue;
    }
    const findings = spec.match === 'scan'
      ? checkScan(axis, src.truth, spec)
      : checkSites(axis, src.truth, spec);
    const drift = findings.filter((f) => !f.ok);
    if (drift.length) hardFail = true;
    axisReports.push({ axis, state: 'enforced', truth: src.truth, source: src.source, enforced: true, findings });
  }

  const result = {
    ok: !hardFail,
    truth_as_of: ALLOWLIST.truth_as_of,
    agentbox: agentbox || null,
    strict: STRICT,
    axes: axisReports,
  };

  if (OUT_JSON) { process.stdout.write(JSON.stringify(result, null, 2) + '\n'); }
  else if (!QUIET) { printHuman(result); }

  process.exit(hardFail ? 1 : 0);
}

function printHuman(result) {
  const out = [];
  out.push(`drift-counter (RES-d) — self-description drift gate`);
  out.push(`  truth as of : ${result.truth_as_of}`);
  out.push(`  agentbox    : ${result.agentbox || '(not found)'}`);
  out.push(`  strict      : ${result.strict}`);
  out.push('');
  for (const a of result.axes) {
    if (a.state === 'planned') { out.push(`  [PLANNED]     ${a.axis} — ${a.reason}`); continue; }
    if (a.state === 'unavailable') { out.push(`  [UNAVAILABLE] ${a.axis} — ${a.reason}`); continue; }
    out.push(`  [ENFORCED]    ${a.axis} = ${a.truth}   (source: ${a.source})`);
    for (const f of a.findings) {
      if (f.ok) { out.push(`      ok    ${f.file}:${f.line ?? '?'}  states ${f.stated}`); continue; }
      if (f.kind) { out.push(`      FAIL  ${f.file}  ${f.kind}${f.note ? ' — ' + f.note : ''}`); continue; }
      out.push(`      DRIFT ${f.file}:${f.line}  states ${f.stated}, truth ${f.truth}  «${f.text}»`);
    }
  }
  out.push('');
  out.push(result.ok ? 'RESULT: PASS — no drift on any enforced axis.' : 'RESULT: FAIL — drift detected (see DRIFT/FAIL lines above).');
  process.stdout.write(out.join('\n') + '\n');
}

run();
