#!/usr/bin/env bash
# dream-evaluator-exit.test.sh — dream-build-check.sh and dream-link-check.sh
# print a verdict marker AND exit with it (TODO-unified S-7d). Each script
# resolves website/dist relative to itself, so the test copies them into a
# scratch tree.
#
#   bash tests/gates/dream-evaluator-exit.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
eq()  { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "expected '$3', got '$2'"; fi; }

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

fresh_tree() {
  local t="$SCRATCH/$1"
  mkdir -p "$t/scripts" "$t/website/dist"
  printf "example.test\n" > "$t/website/dist/CNAME"
  cp "$REPO_ROOT/scripts/dream-build-check.sh" "$REPO_ROOT/scripts/dream-link-check.sh" "$t/scripts/"
  printf '%s' "$t"
}

echo "dream evaluator exit-code tests"
echo "==============================="

echo
echo "[1] dream-build-check.sh"
T="$(fresh_tree build-missing)"
OUT="$(cd "$T" && bash scripts/dream-build-check.sh)"; RC=$?
eq "no index.html prints BUILD-FAIL" "$(tail -1 <<< "$OUT")" "BUILD-FAIL"
eq "no index.html exits 1" "$RC" "1"

T="$(fresh_tree build-ok)"
printf '<!doctype html><title>x</title>' > "$T/website/dist/index.html"
OUT="$(cd "$T" && bash scripts/dream-build-check.sh)"; RC=$?
eq "index.html prints BUILD-OK" "$(tail -1 <<< "$OUT")" "BUILD-OK"
eq "index.html exits 0" "$RC" "0"

echo
echo "[2] dream-link-check.sh"
T="$(fresh_tree links-ok)"
printf '<!doctype html><a href="about.html">a</a><a href="#top"></a><h1 id="top"></h1>' > "$T/website/dist/index.html"
printf '<!doctype html><p>about</p>' > "$T/website/dist/about.html"
OUT="$(cd "$T" && bash scripts/dream-link-check.sh)"; RC=$?
eq "resolving links print LINK-INTEGRITY-OK" "$(tail -1 <<< "$OUT")" "LINK-INTEGRITY-OK"
eq "resolving links exit 0" "$RC" "0"

T="$(fresh_tree links-missing)"
printf '<!doctype html><a href="gone.html">a</a>' > "$T/website/dist/index.html"
OUT="$(cd "$T" && bash scripts/dream-link-check.sh)"; RC=$?
eq "a missing file prints LINK-INTEGRITY-FAIL" "$(tail -1 <<< "$OUT")" "LINK-INTEGRITY-FAIL"
eq "a missing file exits 1" "$RC" "1"

T="$(fresh_tree links-dangling)"
printf '<!doctype html><a href="#nowhere">a</a>' > "$T/website/dist/index.html"
OUT="$(cd "$T" && bash scripts/dream-link-check.sh)"; RC=$?
eq "a dangling anchor prints LINK-INTEGRITY-FAIL" "$(tail -1 <<< "$OUT")" "LINK-INTEGRITY-FAIL"
eq "a dangling anchor exits 1" "$RC" "1"

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
