#!/usr/bin/env bash
# harness-audit.test.sh — regression tests for scripts/harness-audit.sh
#
# Each test pins one of the three defects closed on 2026-09-05 (engineering
# ADR-004 closeout). The originals are reproduced in
# docs/estate-review/evidence/harness-audit-probe.json, where the old script
# scored a fabricated-source template 100% source-backed and a
# duplicate-pairing template 200% coverage — both PASS.
#
#   bash tests/gates/harness-audit.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
AUDIT="$REPO_ROOT/scripts/harness-audit.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
check() { # check <name> <expected-substring> <actual>
  if [[ "$3" == *"$2"* ]]; then ok "$1"; else bad "$1" "expected to find: $2"; fi
}

# Build a one-guide/one-sensor template in a fresh directory.
# usage: mk <dir> <guide-source> <sensor-source> <pairings-json> [guide-status] [sensor-status]
mk() {
  local dir="$1" gsrc="$2" ssrc="$3" pairings="$4"
  local gstat="${5:-present}" sstat="${6:-present}"
  mkdir -p "$dir"
  cat > "$dir/fixture.json" <<JSON
{
  "version": "1.0.0",
  "topology": "fixture",
  "structure": { "substrates": ["VisionFlow"] },
  "guides":  [ { "id": "g1", "type": "instruction", "source": "$gsrc",
                 "source_status": "$gstat", "applies_to": ["VisionFlow"],
                 "description": "fixture guide" } ],
  "sensors": [ { "id": "s1", "type": "computational", "source": "$ssrc",
                 "source_status": "$sstat", "applies_to": ["VisionFlow"],
                 "frequency": "per_commit", "description": "fixture sensor" } ],
  "pairings": $pairings
}
JSON
}

ONE_PAIR='[{"guide_id":"g1","sensor_id":"s1","validation_mode":"blocking"}]'
DUP_PAIR='[{"guide_id":"g1","sensor_id":"s1","validation_mode":"blocking"},
           {"guide_id":"g1","sensor_id":"s1","validation_mode":"advisory"}]'

echo "harness-audit.sh regression tests"
echo "================================="

# ── 1. Duplicate pairing edges must not inflate coverage past 100% ───────
# Old behaviour: 1 guide, 1 sensor, 2 identical pairings => "200.0%" and PASS.
echo
echo "[1] duplicate pairing edges are de-duplicated"
mk "$TMP/dup" "VisionFlow:scripts/harness-audit.sh" "VisionFlow:scripts/harness-audit.sh" "$DUP_PAIR"
OUT="$(bash "$AUDIT" --dir "$TMP/dup" 2>&1)"; RC=$?
check "coverage is 100%, not 200%" "| 100.0% |" "$OUT"
if [[ "$OUT" == *"200.0%"* ]]; then bad "no 200% anywhere" "output still reports 200.0%"; else ok "no 200% anywhere"; fi
check "duplicate edge is reported"  "Duplicate pairing edges" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0 (coverage genuinely complete)" || bad "exit 0" "got $RC"

# ── 2. A nonexistent source declared "present" is NOT source-backed ──────
# Old behaviour: source_status defaulted to present and was never checked, so
# a fabricated path scored "2/2 controls source-backed (100.0% present)".
echo
echo "[2] nonexistent source path is not source-backed"
mk "$TMP/ghost" "VisionFlow:does/not/exist.rs" "VisionFlow:also/missing.rs" "$ONE_PAIR"
OUT="$(bash "$AUDIT" --dir "$TMP/ghost" 2>&1)"; RC=$?
check "unresolved count is 2"          "unresolved (declared present, path absent) : 2" "$OUT"
check "resolved count is 0"            "resolved (source-backed)  : 0" "$OUT"
check "backing is 0%"                  "backing = 0/2 = 0.0%" "$OUT"
check "unresolved sources are listed"  "UNRESOLVED — declared present but the path does not exist" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0 without --strict-sources (reported, not enforced)" || bad "exit 0" "got $RC"

echo
echo "[2b] --strict-sources turns unresolved sources into a failure"
OUT="$(bash "$AUDIT" --dir "$TMP/ghost" --strict-sources 2>&1)"; RC=$?
check "strict verdict present" "FAIL (--strict-sources)" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 under --strict-sources" || bad "exit 1 under --strict-sources" "got $RC"

# ── 3. A real source path resolves ───────────────────────────────────────
echo
echo "[3] an existing source path resolves and is source-backed"
mk "$TMP/real" "VisionFlow:scripts/harness-audit.sh" "VisionFlow:scripts/website-assets.mjs" "$ONE_PAIR"
OUT="$(bash "$AUDIT" --dir "$TMP/real" --strict-sources 2>&1)"; RC=$?
check "both sources resolved" "resolved (source-backed)  : 2" "$OUT"
check "backing is 100%"       "backing = 2/2 = 100.0%" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0 under --strict-sources" || bad "exit 0 under --strict-sources" "got $RC"

# ── 4. Backing is capped at DISTINCT sources ─────────────────────────────
# Two controls naming one file are one source, not two.
echo
echo "[4] two controls sharing one source count as one distinct source"
mk "$TMP/shared" "VisionFlow:scripts/harness-audit.sh" "VisionFlow:scripts/harness-audit.sh" "$ONE_PAIR"
OUT="$(bash "$AUDIT" --dir "$TMP/shared" 2>&1)"
check "one distinct source across two controls" "distinct sources declared : 1  (across 2 controls)" "$OUT"
check "per-control figure still reported"       "per-control: 2 resolved" "$OUT"

# ── 5. An unknown substrate is unverifiable, not a silent pass ───────────
echo
echo "[5] a substrate with no checkout is unverifiable, not source-backed"
mk "$TMP/nosub" "nosuchrepo:src/lib.rs" "nosuchrepo:tests/t.rs" "$ONE_PAIR"
OUT="$(bash "$AUDIT" --dir "$TMP/nosub" 2>&1)"; RC=$?
check "counted unverifiable" "unverifiable (substrate not checked out)   : 2" "$OUT"
check "not counted as backed" "resolved (source-backed)  : 0" "$OUT"
check "listed explicitly"     "Unverifiable (substrate checkout absent)" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0 (partial-source failure mode: reported, not enforced)" || bad "exit 0" "got $RC"

# ── 6. A dangling pairing edge cannot evidence coverage ──────────────────
echo
echo "[6] a pairing referencing a non-existent control is not counted"
mk "$TMP/dangle" "VisionFlow:scripts/harness-audit.sh" "VisionFlow:scripts/harness-audit.sh" \
  '[{"guide_id":"g1","sensor_id":"ghost","validation_mode":"blocking"}]'
OUT="$(bash "$AUDIT" --dir "$TMP/dangle" --target 80 2>&1)"; RC=$?
check "dangling edge reported" "Dangling pairing edges" "$OUT"
check "coverage is 0%"         "|     0 |   0.0% |" "$OUT"
[[ $RC -eq 1 ]] && ok "exit 1 (coverage below target)" || bad "exit 1" "got $RC"

# ── 7. The real templates still pass and carry no unresolved source ──────
echo
echo "[7] the committed templates pass with zero unresolved sources"
OUT="$(bash "$AUDIT" 2>&1)"; RC=$?
check "zero unresolved" "unresolved (declared present, path absent) : 0" "$OUT"
check "coverage 100%"   "TOTAL                          |     20 |      20 |    20 | 100.0% |" "$OUT"
[[ $RC -eq 0 ]] && ok "exit 0" || bad "exit 0" "got $RC"

# ── summary ──────────────────────────────────────────────────────────────
echo
echo "================================="
echo "passed: $PASS   failed: $FAIL"
[[ $FAIL -eq 0 ]] && { echo "HARNESS-AUDIT-TESTS-OK"; exit 0; }
echo "HARNESS-AUDIT-TESTS-FAIL"; exit 1
