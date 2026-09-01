#!/usr/bin/env bash
# dream-meta-tags-scan.sh — provably-live meta-tag scan of the built site
# (seo-and-meta deep, surface: meta-tags). Checked-in script because the
# annexe ssh dispatch strips nested double quotes from inline entrypoints.
# Verbatim (non-tail) receipt: every line is part of the evidence; sentinel
# META-SCAN-OK/META-SCAN-FAIL terminates the run (Issue: seo-and-meta deeps
# produced no receipts — 2026-08-31 night).
set -uo pipefail
cd "$(dirname "$0")/../website/dist" 2>/dev/null || { echo "NO-DIST (run build first)"; echo META-SCAN-FAIL; exit 0; }
test -f index.html || { echo "NO-INDEX"; echo META-SCAN-FAIL; exit 0; }

missing=0
have() { # have <label> <grep-pattern> [required]
  local label=$1 pattern=$2 required=${3:-}
  local n
  n=$(grep -c -iE "$pattern" index.html 2>/dev/null || true)
  echo "$label: $n"
  if [ "$n" -eq 0 ] && [ "$required" = required ]; then
    echo "MISSING-REQUIRED: $label"
    missing=$((missing+1))
  fi
}

have "title"            '<title[ >]'                                        required
have "meta-description" '<meta[^>]+name="description"'                      required
have "canonical"        '<link[^>]+rel="canonical"'                         required
have "viewport"         '<meta[^>]+name="viewport"'                         required
have "og-tags"          '<meta[^>]+property="og:'
have "twitter-tags"     '<meta[^>]+name="twitter:'
have "robots"           '<meta[^>]+name="robots"'

echo "required-missing: $missing"
[ "$missing" -eq 0 ] && echo META-SCAN-OK || echo META-SCAN-FAIL
