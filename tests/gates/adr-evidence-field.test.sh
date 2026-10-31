#!/usr/bin/env bash
# ADR-2007: adr-index-gen --check gates evidence_class / acceptance_boundary from the cut-over.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
gen="$root/scripts/adr-index-gen.cjs"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
rec() { # date extra-frontmatter
cat > "$tmp/ADR-9001-x.md" <<EOT
---
id: ADR-9001
title: Fixture
date: $1
decision_status: accepted
implementation_status: complete
activation_status: staged
supersedes: []
superseded_by: []
verified_commit: abc1234
owner: test
review_trigger: never
repo: visionflow
$2
---
# ADR-9001
EOT
}
fails=0
expect() { # name want-exit
  node "$gen" "$tmp" --check >/dev/null 2>&1; got=$?
  if [ "$got" = "$2" ]; then echo "ok   $1"; else echo "FAIL $1 (exit $got, want $2)"; fails=$((fails+1)); fi
}
rec 2026-10-30 "";                                                    expect "pre-cutover record without fields passes" 0
rec 2026-10-31 "";                                                    expect "post-cutover record without fields fails" 1
rec 2026-10-31 "evidence_class: test"$'\n'"acceptance_boundary: unit only"; expect "post-cutover with both passes" 0
rec 2026-10-31 "evidence_class: test+ci";                             expect "missing boundary fails" 1
rec 2026-10-31 "evidence_class: vibes"$'\n'"acceptance_boundary: x"; expect "unknown class fails" 1
rec 2026-10-31 "evidence_class: test+runtime"$'\n'"acceptance_boundary: x"; expect "joined classes pass" 0
rec 2026-01-01 "evidence_class: vibes";                               expect "pre-cutover invalid class still fails" 1
exit $fails
