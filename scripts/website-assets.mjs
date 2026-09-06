#!/usr/bin/env node
// @ts-check
'use strict';

/**
 * website-assets — asset-inventory enforcement and build receipt for the
 * copy-only static site (ADR-2002 closeout; ADR-2003 publication gate).
 *
 * The defect this closes: `website/build.sh` staged the repo image directories
 * with `cp -r ... 2>/dev/null || true`. Every optional copy could fail and the
 * build still exited 0 and printed BUILD-COMPLETE, so "the build ran" certified
 * nothing about the media actually present in `dist/`. There was no declaration
 * anywhere of which assets the page genuinely requires.
 *
 * `website/assets.manifest.json` is now that declaration:
 *   - `required` — must exist in dist/ after the build, or the build FAILS;
 *   - `derived`  — required assets copied in from outside `static/` (fail if the
 *                  source is missing, because the destination is required);
 *   - `optional` — staged best-effort; presence or absence is recorded
 *                  EXPLICITLY in the receipt rather than silently swallowed.
 *
 * Subcommands
 *   stage    Copy `derived` + `optional` entries into dist/. Missing derived
 *            source => exit 1. Missing optional source => recorded, exit 0.
 *   verify   Assert every `required` dest exists and is non-empty; emit the
 *            build receipt (file count, byte total, per-asset SHA-256).
 *            Missing/empty required asset => exit 1.
 *
 * Usage
 *   node scripts/website-assets.mjs stage  [--root website]
 *   node scripts/website-assets.mjs verify [--root website]
 *                                          [--receipt website/build-receipt.json]
 *                                          [--published-revision <sha>]
 *
 * The receipt is written OUTSIDE dist/ so it never becomes part of the artefact
 * it describes (which would make its own hash unstable).
 */

import {
  readFileSync, writeFileSync, existsSync, statSync, readdirSync,
  mkdirSync, copyFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, relative } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..');

// ── argument parsing ─────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const CMD = argv[0];

/** Read `--flag value`, or a default. */
function opt(name, fallback) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
}

const WEBSITE_DIR = resolve(REPO_ROOT, opt('root', 'website'));
const MANIFEST_PATH = join(WEBSITE_DIR, 'assets.manifest.json');

if (!['stage', 'verify'].includes(CMD)) {
  process.stderr.write(
    'usage: website-assets.mjs <stage|verify> [--root website] '
    + '[--receipt PATH] [--published-revision SHA]\n');
  process.exit(2);
}

if (!existsSync(MANIFEST_PATH)) {
  process.stderr.write(`ERROR: asset manifest not found: ${MANIFEST_PATH}\n`);
  process.exit(1);
}

const MANIFEST_RAW = readFileSync(MANIFEST_PATH);
const MANIFEST = JSON.parse(MANIFEST_RAW.toString('utf8'));
const DIST = join(WEBSITE_DIR, MANIFEST.output || 'dist');

// ── helpers ──────────────────────────────────────────────────────────────
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const sha256File = (p) => sha256(readFileSync(p));

/** Every regular file under `dir`, as repo-relative-to-dir POSIX paths. */
function walk(dir, base = dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const abs = join(dir, entry.name);
    if (entry.isDirectory()) walk(abs, base, out);
    else if (entry.isFile()) out.push(relative(base, abs).split('\\').join('/'));
  }
  return out;
}

/** Glob a single `*`-style include pattern (no path separators). */
function matchesInclude(name, pattern) {
  if (!pattern || pattern === '*') return true;
  const re = new RegExp('^' + pattern.split('*').map(
    (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$');
  return re.test(name);
}

function ensureDir(p) { mkdirSync(p, { recursive: true }); }

/** Current git revision of the canon, or null outside a checkout. */
function gitRevision() {
  try {
    return execFileSync('git', ['-C', REPO_ROOT, 'rev-parse', 'HEAD'],
      { encoding: 'utf8' }).trim();
  } catch { return null; }
}

// ── stage ────────────────────────────────────────────────────────────────
/**
 * Copy `derived` (required, hard-fail) and `optional` (best-effort, recorded)
 * entries into dist/. Returns a staging report for the receipt.
 */
function stage() {
  const report = { derived: [], optional: [] };
  let failed = false;

  for (const d of MANIFEST.derived || []) {
    const src = resolve(WEBSITE_DIR, d.from);
    const dst = join(DIST, d.dest);
    if (!existsSync(src) || !statSync(src).isFile()) {
      // A derived entry backs a REQUIRED dest, so a missing source is fatal —
      // this is precisely the failure the old `|| true` swallowed.
      process.stderr.write(
        `ERROR: derived asset source missing: ${d.from} -> ${d.dest}\n`);
      report.derived.push({ dest: d.dest, from: d.from, status: 'source-missing' });
      failed = true;
      continue;
    }
    ensureDir(dirname(dst));
    copyFileSync(src, dst);
    report.derived.push({
      dest: d.dest, from: d.from, status: 'copied',
      bytes: statSync(dst).size, sha256: sha256File(dst),
    });
    process.stdout.write(`    derived  ${d.dest}  <- ${d.from}\n`);
  }

  for (const g of MANIFEST.optional || []) {
    const src = resolve(WEBSITE_DIR, g.from);
    if (!existsSync(src)) {
      // Explicitly recorded, not silently swallowed.
      report.optional.push({ id: g.id, from: g.from, status: 'source-absent', files: 0, bytes: 0 });
      process.stdout.write(`    optional ${g.id}: SOURCE ABSENT (${g.from}) — recorded, not fatal\n`);
      continue;
    }
    const destDir = join(DIST, g.dest || '');
    ensureDir(destDir);
    let files = 0; let bytes = 0;
    const names = statSync(src).isDirectory()
      ? readdirSync(src, { withFileTypes: true })
        .filter((e) => e.isFile() && matchesInclude(e.name, g.include))
        .map((e) => e.name)
      : [];
    for (const name of names) {
      const to = join(destDir, name);
      copyFileSync(join(src, name), to);
      files += 1; bytes += statSync(to).size;
    }
    report.optional.push({ id: g.id, from: g.from, dest: g.dest, status: 'staged', files, bytes });
    process.stdout.write(`    optional ${g.id}: ${files} file(s), ${bytes} bytes\n`);
  }

  if (failed) process.exit(1);
  return report;
}

// ── verify + receipt ─────────────────────────────────────────────────────
function verify() {
  const missing = [];
  const required = [];

  for (const r of MANIFEST.required || []) {
    const abs = join(DIST, r.dest);
    if (!existsSync(abs) || !statSync(abs).isFile()) {
      missing.push({ dest: r.dest, why: r.why, reason: 'absent' });
      continue;
    }
    const bytes = statSync(abs).size;
    if (bytes === 0) {
      missing.push({ dest: r.dest, why: r.why, reason: 'empty' });
      continue;
    }
    required.push({ dest: r.dest, bytes, sha256: sha256File(abs) });
  }

  const all = walk(DIST);
  const totalBytes = all.reduce((n, rel) => n + statSync(join(DIST, rel)).size, 0);

  // A single stable fingerprint for the whole artefact: SHA-256 over the
  // sorted "sha256␠path" lines of every file in dist/. Two builds agree iff
  // they published byte-identical trees.
  const treeLines = all.slice().sort()
    .map((rel) => `${sha256File(join(DIST, rel))}  ${rel}`);
  const treeDigest = sha256(treeLines.join('\n'));

  const localRev = gitRevision();
  const receipt = {
    receipt_version: 1,
    generated_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    manifest: {
      path: relative(REPO_ROOT, MANIFEST_PATH).split('\\').join('/'),
      sha256: sha256(MANIFEST_RAW),
      required_count: (MANIFEST.required || []).length,
      optional_count: (MANIFEST.optional || []).length,
    },
    revision: {
      // The revision the artefact was BUILT from, and the revision it is
      // PUBLISHED as. In GitHub Actions these are the same commit; locally
      // `published` is null because nothing was published (ADR-2003).
      source: localRev,
      published: opt('published-revision', process.env.GITHUB_SHA || null),
      workflow_run: process.env.GITHUB_RUN_ID || null,
      dirty: localRev
        ? (() => {
          try {
            return execFileSync('git', ['-C', REPO_ROOT, 'status', '--short'],
              { encoding: 'utf8' }).trim().length > 0;
          } catch { return null; }
        })()
        : null,
    },
    dist: {
      path: relative(REPO_ROOT, DIST).split('\\').join('/'),
      file_count: all.length,
      bytes: totalBytes,
      tree_sha256: treeDigest,
    },
    required_assets: required,
    missing_required: missing,
    staging: readStagingSidecar(),
    ok: missing.length === 0,
  };

  const receiptPath = resolve(REPO_ROOT,
    opt('receipt', relative(REPO_ROOT, join(WEBSITE_DIR, 'build-receipt.json'))));
  ensureDir(dirname(receiptPath));
  writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');

  process.stdout.write(
    `    required : ${required.length}/${(MANIFEST.required || []).length} present\n`
    + `    dist     : ${all.length} files, ${totalBytes} bytes\n`
    + `    tree     : sha256:${treeDigest}\n`
    + `    receipt  : ${relative(REPO_ROOT, receiptPath)}\n`);

  if (missing.length) {
    process.stderr.write('\nERROR: required assets missing from dist/:\n');
    for (const m of missing) {
      process.stderr.write(`  - ${m.dest} (${m.reason}) — ${m.why}\n`);
    }
    process.stderr.write(
      '\nThe asset inventory (website/assets.manifest.json) declares these as required.\n'
      + 'A build that cannot produce them is not publishable.\n');
    process.exit(1);
  }
  process.stdout.write('ASSET-INVENTORY-OK\n');
}

/**
 * `stage` writes its report beside dist/ so `verify` (a separate process, and
 * a separate CI step) can fold it into the receipt without re-copying.
 */
const SIDECAR = () => join(WEBSITE_DIR, '.staging-report.json');
function writeStagingSidecar(report) {
  writeFileSync(SIDECAR(), JSON.stringify(report, null, 2) + '\n');
}
function readStagingSidecar() {
  try { return JSON.parse(readFileSync(SIDECAR(), 'utf8')); }
  catch { return null; }
}

// ── main ─────────────────────────────────────────────────────────────────
if (CMD === 'stage') {
  if (!existsSync(DIST)) {
    process.stderr.write(`ERROR: dist not found: ${DIST} (run the copy step first)\n`);
    process.exit(1);
  }
  writeStagingSidecar(stage());
} else {
  verify();
}
