#!/usr/bin/env bash
# Commit audit wave 1 on audit/wave1, then merge into VisionFlow main.
# Usage: commit-wave1.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave1
P=/home/devuser/workspace/VisionFlow
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave1
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
cd "$W"
# Final ledger check on the bytes being committed.
CARGO_HOME=/home/devuser/workspace/.cargo CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/ledger-target \
  cargo run -q --locked --offline --manifest-path scripts/estate-review-ledger/Cargo.toml -- \
  /home/devuser/workspace "$W" docs/estate-review/completion/scope.json \
  docs/estate-review/completion/claims.json "$D/snapshot-wave1-end.json" > /dev/null
python3 -I -c "
import json;c=json.load(open('$D/snapshot-wave1-end.json'))['counts']
assert c['stale_claim_records']==0, c; print('final counts', json.dumps(c, sort_keys=True))"
cp "$SP"/wave1-end.sh "$SP"/codex-audit-wave1.sh "$SP"/codex-audit-wave1b.sh "$SP"/remaining.py "$D"/
git add -- docs/diagrams/solid-pod-rs docs/estate-review/2026-11-02-*.md \
  docs/estate-review/completion/claims.json docs/estate-review/evidence/2026-11-02/audit-wave1 \
  docs/estate-review/remediation/audit-wave1-2026-11-02.json
echo "--- staged"; git diff --cached --stat | tail -3
test -z "$(git status --porcelain | grep -v '^[AM] ')" || { echo "unstaged leftovers:"; git status --porcelain | grep -v '^[AM] '; exit 1; }
git commit -q -m "docs(audit): wave 1 completes Solid, loom and dream-machine; rebinds stale claims

Solid: 11/11 numbered ADRs and 142/142 panels assessed at solid-pod-rs
9b6143c, every Solid topic restamped and passing the strict gate (1,758
citations, 142/142 rendered). loom and dream-machine: 16/16 ADRs. Twelve
stale claims rebound or replaced. Ledger: 88 -> 130 ADRs, 375 -> 480
panels, 12 -> 0 stale claims.

Source tracing only. Cross-vendor Codex spot-check is partial, not an
independent review. New findings AW1-01..AW1-11 queued in
remediation/audit-wave1-2026-11-02.json.

Co-Authored-By: jjohare <github@thedreamlab.uk>"
git log -1 --format='%h %s'
