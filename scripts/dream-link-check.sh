#!/bin/bash
# dream-link-check.sh — provably-live internal-link/asset integrity check for
# the built static site. Emits a surface-dependent count; non-zero missing
# refs prints LINK-INTEGRITY-FAIL. Used as a dream-cycle evaluator entrypoint.
set -uo pipefail
cd "$(dirname "$0")/../website/dist" 2>/dev/null || { echo "NO-DIST (run build first)"; exit 0; }
miss=0; checked=0
while IFS= read -r f; do
  d=$(dirname "$f")
  for ref in $(grep -oE '(src|href)="[^"#:]+"' "$f" 2>/dev/null | sed -E 's/.*="([^"]+)".*/\1/' | grep -vE '^(https?:|//|mailto:|data:)'); do
    t="${ref%%\?*}"; checked=$((checked+1))
    [ -e "$d/$t" ] || [ -e "./$t" ] || { echo "MISSING: $f -> $ref"; miss=$((miss+1)); }
  done
done < <(find . -name '*.html')
echo "internal-refs-checked: $checked  missing: $miss"
[ "$miss" -eq 0 ] && echo LINK-INTEGRITY-OK || echo LINK-INTEGRITY-FAIL
