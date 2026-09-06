#!/usr/bin/env bash
# drift-counter.test.sh — regression tests for scripts/drift-counter/drift-counter.mjs
#
# The counter's whole value is that an intentional mismatch turns the build red.
# Until 2026-09-05 nothing proved it still could: two policed sites pointed at a
# path that had been archived, so they reported `file-missing` on every run — a
# hole in the gate that looked like coverage — and the sibling count source was
# unpinned, so the same canon commit could pass or fail depending on which
# agentbox HEAD the runner happened to fetch.
#
# These tests drive the counter with a synthetic allowlist (DRIFT_ALLOWLIST) and
# a settable truth (DRIFT_VISIONCLAW_CLASS_COUNT), so they assert the mechanism
# rather than today's figures.
#
#   bash tests/gates/drift-counter.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
COUNTER="$REPO_ROOT/scripts/drift-counter/drift-counter.mjs"

# Policed sites are resolved relative to the repo root, so fixtures must live
# inside the tree. Everything under this directory is removed on exit.
SCRATCH="$REPO_ROOT/tests/gates/.tmp-drift"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
check() { if [[ "$3" == *"$2"* ]]; then ok "$1"; else bad "$1" "expected to find: $2"; fi; }
nocheck() { if [[ "$3" != *"$2"* ]]; then ok "$1"; else bad "$1" "did not expect: $2"; fi; }

# Write a synthetic allowlist policing one site. The axis is `ontology-classes`
# because its source is settable from the environment, which lets the test fix
# the truth without depending on a sibling checkout.
# usage: mk_allowlist <file> <policed-relative-path> [pin-revision]
mk_allowlist() {
  local out="$1" policed="$2" pin="${3:-}"
  local pinblock=""
  if [[ -n "$pin" ]]; then
    pinblock="\"source_pin\": { \"repository\": \"DreamLab-AI/agentbox\", \"revision\": \"$pin\" },"
  fi
  cat > "$out" <<JSON
{
  "truth_as_of": "test",
  $pinblock
  "axes": {
    "ontology-classes": {
      "label": "test axis",
      "denominator": { "counts": "test units" },
      "source": { "kind": "visionclaw-class-count", "env": "DRIFT_VISIONCLAW_CLASS_COUNT" },
      "match": "sites",
      "sites": [ { "file": "$policed", "pattern": "(\\\\d+) test units" } ]
    }
  }
}
JSON
}

REL="tests/gates/.tmp-drift/site.md"
ABS="$REPO_ROOT/$REL"
AL="$SCRATCH/allowlist.json"

echo "drift-counter.mjs regression tests"
echo "=================================="

# ── 1. Agreement passes ──────────────────────────────────────────────────
echo
echo "[1] a policed figure that agrees with the source passes"
echo "The system exposes 42 test units today." > "$ABS"
mk_allowlist "$AL" "$REL"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" 2>&1)"; RC=$?
check "reports PASS" "RESULT: PASS" "$OUT"
check "site reported ok" "states 42" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0" || bad "exit 0" "got $RC"

# ── 2. AN INTENTIONAL MISMATCH IS DETECTED ───────────────────────────────
# This is the canary the gate exists for: change the figure at a policed site
# so it disagrees with its source, and the build must go red.
echo
echo "[2] an intentional mismatch at a policed site is detected"
echo "The system exposes 41 test units today." > "$ABS"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" 2>&1)"; RC=$?
check "reports FAIL"          "RESULT: FAIL" "$OUT"
check "names the drift"       "DRIFT" "$OUT"
check "shows stated vs truth" "states 41, truth 42" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 on drift" || bad "exit 1 on drift" "got $RC"

echo
echo "[2b] the mismatch is machine-readable for CI"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" --json 2>&1)"
check "ok:false in JSON"  '"ok": false' "$OUT"
check "finding is ok:false" '"ok": false' "$OUT"
check "denominator is published" '"counts": "test units"' "$OUT"

# ── 3. A policed site that cannot be read is a FAILURE, not a pass ───────
# The archived-ADR-002 defect: a moved target reported file-missing forever
# while the gate still went green overall on that axis' other sites.
echo
echo "[3] a policed site whose file has moved fails the gate"
mk_allowlist "$AL" "tests/gates/.tmp-drift/moved-away.md"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" 2>&1)"; RC=$?
check "reports file-missing" "file-missing" "$OUT"
check "reports FAIL"         "RESULT: FAIL" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 on unreadable policed site" || bad "exit 1" "got $RC"

# ── 4. A policed site that no longer carries the figure fails ────────────
echo
echo "[4] a policed site whose pattern no longer matches fails"
echo "This file no longer states the figure at all." > "$ABS"
mk_allowlist "$AL" "$REL"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" 2>&1)"; RC=$?
check "reports site-missing" "site-missing" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 on vanished figure" || bad "exit 1" "got $RC"

# ── 5. The sibling source pin is enforced ────────────────────────────────
echo
echo "[5] a sibling checkout off the pinned revision fails the gate"
echo "The system exposes 42 test units today." > "$ABS"
mk_allowlist "$AL" "$REL" "0000000000000000000000000000000000000000"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" 2>&1)"; RC=$?
check "pin failure reported" "source pin  : FAIL" "$OUT"
check "names the pinned revision" "0000000000000000000000000000000000000000" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 on pin mismatch" || bad "exit 1 on pin mismatch" "got $RC"

echo
echo "[5b] --allow-pin-drift downgrades the pin mismatch to a warning"
OUT="$(DRIFT_ALLOWLIST="$AL" DRIFT_VISIONCLAW_CLASS_COUNT=42 node "$COUNTER" --allow-pin-drift 2>&1)"; RC=$?
check "warned, not failed" "source pin  : WARN" "$OUT"
check "gate still passes"  "RESULT: PASS" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0 with explicit override" || bad "exit 0 with override" "got $RC"

# ── 6. The committed allowlist is green and reads every policed site ─────
echo
echo "[6] the committed allowlist passes with no unreadable policed site"
OUT="$(node "$COUNTER" 2>&1)"; RC=$?
check "reports PASS"        "RESULT: PASS" "$OUT"
nocheck "no file-missing"   "file-missing" "$OUT"
nocheck "no site-missing"   "site-missing" "$OUT"
check "pin verified"        "source pin  : OK" "$OUT"
check "archived ADR-002 site is read" "docs/archive/adr/ADR-002-ecosystem-alignment-governance.md" "$OUT"
check "unavailable axis states its exit condition" "resolves when:" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0" || bad "exit 0" "got $RC"

# ── summary ──────────────────────────────────────────────────────────────
echo
echo "=================================="
echo "passed: $PASS   failed: $FAIL"
[[ $FAIL -eq 0 ]] && { echo "DRIFT-COUNTER-TESTS-OK"; exit 0; }
echo "DRIFT-COUNTER-TESTS-FAIL"; exit 1
