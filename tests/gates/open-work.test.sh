#!/usr/bin/env bash
# open-work.test.sh — the open-work register gate must be able to fail.
#
# The register (docs/open-work/register.jsonl) is the one record of open work
# across the estate, and its views are generated. This suite checks the live
# register passes, then drives four deliberate defects through the gate and
# asserts each turns it red: a reused id, an undefined seam, a state with no
# date, and a view that no longer matches the register.
#
#   bash tests/gates/open-work.test.sh

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/scripts/open-work/Cargo.toml"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0

cargo build --quiet --locked --offline --manifest-path "$MANIFEST" || { echo "FAIL: open-work does not build"; exit 1; }
BIN="$ROOT/scripts/open-work/target/debug/open-work"

ok() { echo "  ok   $1"; pass=$((pass + 1)); }
no() { echo "  FAIL $1"; fail=$((fail + 1)); }

expect_green() {
  if "$BIN" "$2" --check >/dev/null 2>&1; then ok "$1"; else no "$1"; fi
}
expect_red() {
  local out
  if out="$("$BIN" "$2" --check 2>&1)"; then no "$1 (gate stayed green)"; return; fi
  if grep -q "$3" <<<"$out"; then ok "$1"; else no "$1 (red, but not for: $3)"; fi
}
fresh() {
  rm -rf "$TMP/case"
  cp -r "$ROOT/docs/open-work" "$TMP/case"
}

expect_green "the committed register and its views are in step" "$ROOT/docs/open-work"

fresh
head -n 1 "$TMP/case/register.jsonl" >>"$TMP/case/register.jsonl"
expect_red "a reused id is refused" "$TMP/case" "already an id or alias"

fresh
sed -i '1s/"seam": *"[^"]*"/"seam": "not-a-seam"/' "$TMP/case/register.jsonl"
expect_red "an undefined seam is refused" "$TMP/case" "is not defined in seams.json"

fresh
sed -i '1s/"date": *"[0-9-]*"/"date": ""/' "$TMP/case/register.jsonl"
expect_red "a state with no date is refused" "$TMP/case" "is not YYYY-MM-DD"

fresh
# A closed item's title appears in no view, so change the first item that is
# still open: line 1 stops being one as soon as its item is fixed.
live_line="$(grep -n -m1 '"status": *"open"' "$TMP/case/register.jsonl" | cut -d: -f1)"
sed -i "${live_line}s/\"title\": *\"/\"title\": \"changed /" "$TMP/case/register.jsonl"
expect_red "a view that no longer matches the register is refused" "$TMP/case" "is stale"

echo "open-work gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
