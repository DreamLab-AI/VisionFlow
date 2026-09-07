#!/usr/bin/env bash
# dream-asset-refs-scan.sh — provably-live asset-inventory and orphan scan of
# the built site (content-integrity deep, surface: asset-refs). Checked-in
# script because the annexe ssh dispatch strips nested double quotes from
# inline entrypoints. Sentinel ASSET-SCAN-OK / ASSET-SCAN-FAIL.
#
# Why this exists: dream.config.json has always listed `asset-refs` as a scan
# surface for the content-integrity deep, but no evaluator covered it, so
# 2026-09-04 honestly recorded FALLBACK rather than inventing a measurement.
# Two questions went unanswered every night:
#
#   1. Is every asset the page REQUIRES actually in dist/? Answered by reusing
#      the build's own gate, `website-assets.mjs verify`, rather than by a
#      second, drifting implementation of the same rule.
#   2. What is the other 99.9% of the payload? The site is ~60 MB of which
#      index.html is ~0.16%; until now that was an unexplained number. This
#      reports orphans — files staged into dist/ that no shipped HTML, CSS or
#      JS refers to — and decomposes dist/ by directory with du.
#
# Orphans are a MEASUREMENT, not a failure: assets.manifest.json stages
# `optional` groups deliberately for deep links, and the manifest says so. As
# with the zero-blocks case in dream-structured-data-scan.sh, an honest number
# is OK. ASSET-SCAN-FAIL is reserved for a genuine defect — dist/ absent, or a
# required asset missing or empty, which is the build's own failure condition.
set -uo pipefail
cd "$(dirname "$0")/.." || { echo "NO-REPO"; echo ASSET-SCAN-FAIL; exit 1; }
test -d website/dist || { echo "NO-DIST (run build first)"; echo ASSET-SCAN-FAIL; exit 1; }

# ── 1. required-asset inventory, via the build's own gate ────────────────
inv_out=$(node scripts/website-assets.mjs verify 2>&1); inv_rc=$?
echo "$inv_out" | grep -E '^\s+(required|dist|tree)\s+:' | sed -E 's/^\s+/inventory-/'
if [ "$inv_rc" -ne 0 ]; then
  echo "REQUIRED-ASSETS-MISSING"
  echo "$inv_out" | grep -E '^\s+- ' || true
fi

# ── 2. orphan detection ─────────────────────────────────────────────────
node - <<'NODE'
const fs = require('fs');
const path = require('path');
const DIST = 'website/dist';
const manifest = JSON.parse(fs.readFileSync('website/assets.manifest.json', 'utf8'));

const walk = (dir, out = []) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, e.name);
    if (e.isDirectory()) walk(abs, out);
    else if (e.isFile()) out.push(path.relative(DIST, abs).split(path.sep).join('/'));
  }
  return out;
};

const all = walk(DIST);
const isText = (p) => /\.(html|css|js|json)$/i.test(p);

// The corpus of things that can REFER to an asset: shipped markup, styles and
// modules. A ref found here is a ref the browser can follow.
const corpus = all.filter((p) => /\.(html|css|js)$/i.test(p))
  .map((p) => fs.readFileSync(path.join(DIST, p), 'utf8')).join('\n');

// Required assets are needed by declaration, so they are never orphans.
const required = new Set((manifest.required || []).map((r) => r.dest));

// Deliberately conservative: a file counts as referenced if either its full
// dist-relative path or its bare basename appears anywhere in the corpus.
// That over-counts references rather than over-reporting orphans, so an
// orphan in this output is a real one, not a parser artefact.
const referenced = (rel) =>
  corpus.includes(rel) || corpus.includes(path.basename(rel));

const bytes = (rel) => fs.statSync(path.join(DIST, rel)).size;
const total = all.reduce((n, r) => n + bytes(r), 0);

const orphans = all
  .filter((r) => !required.has(r) && !isText(r) && !referenced(r))
  .map((r) => ({ rel: r, bytes: bytes(r) }))
  .sort((a, b) => b.bytes - a.bytes);

const orphanBytes = orphans.reduce((n, o) => n + o.bytes, 0);
const pct = (n) => total ? ((n / total) * 100).toFixed(2) : '0.00';

console.log(`dist-files: ${all.length}  dist-bytes: ${total}`);
console.log(`required-declared: ${required.size}  optional-groups: ${(manifest.optional || []).length}`);
console.log(`orphans: ${orphans.length}  orphan-bytes: ${orphanBytes}  orphan-share: ${pct(orphanBytes)}%`);
for (const o of orphans.slice(0, 10)) console.log(`  orphan ${o.bytes}  ${o.rel}`);
if (orphans.length > 10) console.log(`  … ${orphans.length - 10} more`);

const html = all.filter((r) => /\.html$/i.test(r)).reduce((n, r) => n + bytes(r), 0);
console.log(`html-bytes: ${html}  html-share: ${pct(html)}%`);
NODE
node_rc=$?

# ── 3. payload decomposition ────────────────────────────────────────────
echo "dist-breakdown (bytes, by top-level entry):"
du -sb website/dist/* 2>/dev/null | sort -rn | sed -E 's#\twebsite/dist/#\t#' | sed 's/^/  /'

if [ "$inv_rc" -eq 0 ] && [ "$node_rc" -eq 0 ]; then
  echo ASSET-SCAN-OK
else
  echo ASSET-SCAN-FAIL
  exit 1
fi
