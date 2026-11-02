#!/usr/bin/env bash
# End of audit wave 1: rebuild the snapshot from the integration worktree, merge
# the topic claims against it, then rebuild again and report the counts.
# Usage: wave1-end.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave1
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave1
export CARGO_HOME=/home/devuser/workspace/.cargo
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/ledger-target
cd "$W"
collect() {
  cargo run -q --locked --offline --manifest-path scripts/estate-review-ledger/Cargo.toml -- \
    /home/devuser/workspace "$W" \
    docs/estate-review/completion/scope.json \
    docs/estate-review/completion/claims.json \
    "$1"
}
collect "$D/snapshot-wave1-premerge.json"
cd "$D"
python3 -I merge_claims.py snapshot-wave1-premerge.json \
  claims-solid-composition-and-boot.json \
  claims-solid-ldp-and-wac.json \
  claims-solid-identity-and-storage.json \
  claims-solid-payments-and-release.json \
  claims-solid-federation-forge-mcp.json \
  claims-solid-failure-modes.json \
  claims-lead-sp14.json
cd "$W"
collect "$D/snapshot-wave1-end.json"
python3 -I - "$D/snapshot-wave1-start.json" "$D/snapshot-wave1-end.json" <<'EOF'
import json, sys
for p in sys.argv[1:]:
    c = json.load(open(p))["counts"]
    print(p.rsplit("/", 1)[1], json.dumps(c, sort_keys=True))
EOF
