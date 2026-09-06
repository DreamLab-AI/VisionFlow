#!/usr/bin/env bash
# run-all.sh — every governance-gate regression suite.
#
# These suites exist because the estate review found three gates that could not
# fail: the harness audit counted unchecked source declarations and duplicate
# pairing edges (200% coverage, PASS), the drift counter policed two sites whose
# file had been archived away, and the release manifest could be promoted to
# candidate while asserting fixture parity nothing had compared.
#
# A gate nobody tests is a gate nobody knows is broken, so each suite drives a
# deliberate defect through its gate and asserts the gate goes red.
#
#   bash tests/gates/run-all.sh

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUITES=(
  "harness-audit.test.sh"
  "drift-counter.test.sh"
  "release-manifest.test.sh"
  "website-assets.test.sh"
)

FAILED=()
for suite in "${SUITES[@]}"; do
  printf '\n\n########## %s ##########\n' "$suite"
  if ! bash "$HERE/$suite"; then FAILED+=("$suite"); fi
done

printf '\n\n==================================================\n'
if [[ ${#FAILED[@]} -eq 0 ]]; then
  printf 'ALL GATE SUITES PASSED (%d)\n' "${#SUITES[@]}"
  echo "GATE-TESTS-OK"
  exit 0
fi
printf 'FAILED SUITES (%d of %d):\n' "${#FAILED[@]}" "${#SUITES[@]}"
printf '  %s\n' "${FAILED[@]}"
echo "GATE-TESTS-FAIL"
exit 1
