#!/usr/bin/env bash
# End of audit wave 3: rebuild the snapshot from the integration worktree,
# merge every wave 3 claim file against it (re-merging replaces an existing
# unit, so ADR claims merged earlier are idempotent), rebuild again, report.
# Usage: wave3-end.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave3
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave3
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
export CARGO_HOME=/home/devuser/workspace/.cargo CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/estate-ledger-target CARGO_BUILD_JOBS=4
collect() {
  cargo run --quiet --locked --offline --manifest-path "$W/scripts/estate-review-ledger/Cargo.toml" -- \
    /home/devuser/workspace "$W" \
    "$W/docs/estate-review/completion/scope.json" \
    "$W/docs/estate-review/completion/claims.json" "$1"
}
collect "$D/snapshot-wave3-mid.json"
cd "$D"
python3 -I merge_claims.py snapshot-wave3-mid.json claims-*.json
collect "$D/snapshot-wave3-end.json"
python3 -I - "$D/snapshot-wave3-end.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))["counts"]
print({k: c[k] for k in sorted(c) if "stale" in k or "panel" in k or "adr" in k})
EOF
python3 -I "$SP/remaining.py" "$D/snapshot-wave3-end.json"
