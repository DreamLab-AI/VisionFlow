#!/usr/bin/env node
'use strict';
/*
 * check-pele-pin.cjs — ADR-2020: assert one exact Pele pin (version, npm
 * integrity, dist/pele.js sha256) across every estate surface that carries it.
 *
 * Usage: node scripts/check-pele-pin.cjs [--workspace <dir>] [--strict]
 *
 *   --workspace  directory holding the sibling checkouts (default: parent of
 *                this repo, or $ESTATE_WORKSPACE)
 *   --strict     a missing sibling checkout is an error (default: skipped, so
 *                the check still runs in CI where only this repo exists)
 *
 * Skipped surfaces are listed; when run from a worktree pass --workspace, or use
 * --strict, otherwise most siblings are absent and the result covers fewer
 * surfaces than the final line suggests.
 *
 * Reference pin: scripts/vendor/pele/VERSION in this repo. Also verifies that
 * the vendored pele.js hashes to the declared sha256. Exit 0 agree, 1 drift.
 * Read-only: nothing is written anywhere.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const args = process.argv.slice(2);
const strict = args.includes('--strict');
const wi = args.indexOf('--workspace');
const repo = path.resolve(__dirname, '..');
const ws = path.resolve(wi >= 0 ? args[wi + 1] : (process.env.ESTATE_WORKSPACE || path.join(repo, '..')));

const read = p => fs.readFileSync(p, 'utf8');
const fromVersionFile = file => {
  const t = read(file);
  return {
    version: (/^pele\s+(\S+)/m.exec(t) || [])[1],
    integrity: (/^integrity\s+(\S+)/m.exec(t) || [])[1],
    sha256: (/^sha256\s+([0-9a-f]{64})/m.exec(t) || [])[1],
  };
};

// Each surface: where it lives, and how to read the pin from it.
const surfaces = [
  { name: 'VisionFlow scripts/vendor/pele/VERSION', file: path.join(repo, 'scripts/vendor/pele/VERSION'), local: true, read: fromVersionFile, vendored: path.join(repo, 'scripts/vendor/pele/pele.js') },
  { name: 'agentbox lib/pele.nix', file: path.join(ws, 'project/agentbox/lib/pele.nix'), read: f => {
      const t = read(f);
      return { version: (/^\s*version\s*=\s*"([^"]+)"/m.exec(t) || [])[1],
               integrity: (/^\s*hash\s*=\s*"(sha512-[^"]+)"/m.exec(t) || [])[1],
               sha256: (/^\s*fileSha256\s*=\s*"([0-9a-f]{64})"/m.exec(t) || [])[1] };
    } },
  { name: 'website package.json + lockfile', file: path.join(ws, 'dreamlab-ai-website/package.json'), read: f => {
      const pj = JSON.parse(read(f));
      const lockPath = path.join(path.dirname(f), 'package-lock.json');
      const lock = fs.existsSync(lockPath) ? JSON.parse(read(lockPath)).packages?.['node_modules/pele'] : undefined;
      const dep = (pj.dependencies || {}).pele || (pj.devDependencies || {}).pele;
      return { version: dep, integrity: lock && lock.integrity, lockVersion: lock && lock.version };
    } },
  { name: 'sealmap docs/diagrams/tools/vendor/pele/VERSION', file: path.join(ws, 'sealmap/docs/diagrams/tools/vendor/pele/VERSION'), read: fromVersionFile, vendored: path.join(ws, 'sealmap/docs/diagrams/tools/vendor/pele/pele.js') },
  { name: 'dream-engine docs/diagrams/tools/vendor/pele/VERSION', file: path.join(ws, 'dream-engine/docs/diagrams/tools/vendor/pele/VERSION'), read: fromVersionFile, vendored: path.join(ws, 'dream-engine/docs/diagrams/tools/vendor/pele/pele.js') },
  { name: 'agentbox skills/diagrams-as-code vendored Pele', file: path.join(ws, 'project/agentbox/skills/diagrams-as-code/scripts/vendor/pele/VERSION'), read: fromVersionFile, vendored: path.join(ws, 'project/agentbox/skills/diagrams-as-code/scripts/vendor/pele/pele.js') },
  // Quartz serves the dist/esm build, so there is no single pele.js to hash:
  // version and integrity must agree, and SHA256SUMS must match every served file.
  { name: 'visionGraph quartz static/pele (dist/esm)', file: path.join(ws, 'visionGraph/quartz/quartz/static/pele', 'VERSION_DIR'), resolveDir: true, read: fromVersionFile },
];

let errors = 0;
const err = m => { console.error(`error: ${m}`); errors++; };
const ref = fromVersionFile(surfaces[0].file);
if (!ref.version || !ref.integrity || !ref.sha256) { err('reference VERSION file lacks version/integrity/sha256'); process.exit(1); }
console.log(`reference pin: pele ${ref.version} (${surfaces[0].name})`);

for (const s of surfaces) {
  if (s.resolveDir) s.file = path.join(path.dirname(s.file), ref.version, 'VERSION');
  if (!fs.existsSync(s.file)) {
    if (strict) err(`${s.name}: not found at ${s.file}`); else console.log(`skip  ${s.name}: ${s.file} absent`);
    continue;
  }
  const got = s.read(s.file);
  const bad = [];
  if (got.version !== ref.version) bad.push(`version ${got.version} != ${ref.version}`);
  if (got.lockVersion !== undefined && got.lockVersion !== ref.version) bad.push(`lockfile version ${got.lockVersion} != ${ref.version}`);
  if (got.integrity !== undefined && got.integrity !== ref.integrity) bad.push('integrity differs');
  else if (got.integrity === undefined) bad.push('integrity missing');
  if (got.sha256 !== undefined && got.sha256 !== ref.sha256) bad.push('sha256 differs');
  if (s.vendored) {
    const actual = crypto.createHash('sha256').update(fs.readFileSync(s.vendored)).digest('hex');
    if (actual !== ref.sha256) bad.push(`vendored pele.js hashes to ${actual.slice(0, 16)}…, not the pinned sha256`);
  }
  if (s.resolveDir) {
    const dir = path.dirname(s.file);
    const sums = read(path.join(dir, 'SHA256SUMS')).split(/\r?\n/).filter(Boolean);
    if (!sums.length) bad.push('SHA256SUMS is empty');
    for (const line of sums) {
      const m = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line);
      if (!m) { bad.push(`unparseable SHA256SUMS line: ${line}`); continue; }
      const f = path.join(dir, m[2]);
      if (!fs.existsSync(f)) bad.push(`${m[2]} listed but missing`);
      else if (crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex') !== m[1]) bad.push(`${m[2]} does not match SHA256SUMS`);
    }
  }
  if (bad.length) err(`${s.name}: ${bad.join('; ')}`); else console.log(`ok    ${s.name}`);
}
if (errors) { console.error(`${errors} drift(s): bump every surface together (ADR-2020)`); process.exit(1); }
const skipped = surfaces.filter(x => !fs.existsSync(x.file)).length;
console.log(`ok: Pele pin agrees on ${surfaces.length - skipped} of ${surfaces.length} surfaces${skipped ? ` (${skipped} skipped: not checked out)` : ''}`);
