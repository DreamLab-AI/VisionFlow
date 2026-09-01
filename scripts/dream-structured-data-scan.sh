#!/usr/bin/env bash
# dream-structured-data-scan.sh — provably-live JSON-LD structured-data scan
# of the built site (seo-and-meta deep, surface: structured-data). Checked-in
# script because the annexe ssh dispatch strips nested double quotes from
# inline entrypoints. Verbatim (non-tail) receipt; sentinel SD-SCAN-OK /
# SD-SCAN-FAIL. Zero blocks is an honest measurement (OK), not a failure;
# unparseable JSON-LD is a failure.
set -uo pipefail
cd "$(dirname "$0")/../website/dist" 2>/dev/null || { echo "NO-DIST (run build first)"; echo SD-SCAN-FAIL; exit 0; }
test -f index.html || { echo "NO-INDEX"; echo SD-SCAN-FAIL; exit 0; }

node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const re = /<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
let m, blocks = 0, bad = 0;
const types = [];
while ((m = re.exec(html)) !== null) {
  blocks++;
  try {
    const doc = JSON.parse(m[1]);
    for (const node of Array.isArray(doc) ? doc : [doc]) {
      if (node && node['@type']) types.push(String(node['@type']));
    }
  } catch (e) {
    bad++;
    console.log(`PARSE-ERROR: block ${blocks}: ${e.message}`);
  }
}
console.log(`json-ld-blocks: ${blocks}  parse-errors: ${bad}`);
if (types.length) console.log(`types: ${types.join(', ')}`);
console.log(bad === 0 ? 'SD-SCAN-OK' : 'SD-SCAN-FAIL');
NODE
