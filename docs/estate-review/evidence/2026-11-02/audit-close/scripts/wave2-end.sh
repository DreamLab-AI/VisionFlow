#!/usr/bin/env bash
# End of audit wave 2: rebuild the snapshot from the integration worktree, merge
# every wave 2 claim file against it (re-merging is idempotent: an existing
# unit is replaced), rebuild again, then report counts, stale claims and the
# whole-corpus structure and verification checks.
# Usage: wave2-end.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave2
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave2
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
collect "$D/snapshot-wave2-premerge.json"
cd "$D"
python3 -I merge_claims.py snapshot-wave2-premerge.json claims-*.json
cd "$W"
collect "$D/snapshot-wave2-end.json"
python3 -I - "$D/snapshot-wave2-start.json" "$D/snapshot-wave2-end.json" <<'EOF'
import json, sys
for p in sys.argv[1:]:
    c = json.load(open(p))["counts"]
    print(p.rsplit("/", 1)[1], json.dumps(c, sort_keys=True))
EOF
echo "== whole-corpus structure check"
node scripts/diagram-index-gen.cjs docs/diagrams --check 2>&1 | tail -2
echo "== regenerate indexes"
node scripts/diagram-index-gen.cjs docs/diagrams 2>&1 | tail -2
