#!/usr/bin/env bash
# ADR-2020: check-pele-pin.cjs passes on agreement, fails on drift, and --strict fails on a missing sibling.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
chk="$root/scripts/check-pele-pin.cjs"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fails=0
expect() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (exit $2, want $3)"; fails=$((fails+1)); fi; }
d="$tmp/dream-engine/docs/diagrams/tools/vendor/pele"; mkdir -p "$d"
cp "$root/scripts/vendor/pele/VERSION" "$root/scripts/vendor/pele/pele.js" "$d/"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "agreeing sibling passes (others skipped)" $? 0
node "$chk" --workspace "$tmp" --strict >/dev/null 2>&1; expect "--strict fails on absent siblings" $? 1
sed -i 's/^pele 1\.[0-9.]*/pele 9.9.9/' "$d/VERSION"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "version drift fails" $? 1
cp "$root/scripts/vendor/pele/VERSION" "$d/VERSION"; echo "// tampered" >> "$d/pele.js"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "tampered pele.js fails" $? 1
exit $fails
