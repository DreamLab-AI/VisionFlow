#!/usr/bin/env bash
# Commit audit wave 2 on audit/wave2, then merge into VisionFlow main.
# Usage: commit-wave2.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave2
P=/home/devuser/workspace/VisionFlow
cd "$W"
git diff --name-only -- docs/diagrams | grep -v -E '^docs/diagrams/(estate|nostr-rust-forum|ontomark|visionflow|dreamlab-ai-website)/|^docs/diagrams/(README|COVERAGE|VERIFICATION)\.md$|^docs/diagrams/citation-allowlist\.json$' && { echo "unexpected diagram change"; exit 1; } || true
git add -- \
  docs/diagrams/estate docs/diagrams/nostr-rust-forum docs/diagrams/ontomark \
  docs/diagrams/visionflow docs/diagrams/dreamlab-ai-website \
  docs/diagrams/README.md docs/diagrams/COVERAGE.md docs/diagrams/VERIFICATION.md \
  docs/diagrams/citation-allowlist.json \
  docs/estate-review/completion/claims.json \
  docs/estate-review/remediation/audit-wave2-2026-11-02.json \
  docs/estate-review/evidence/2026-11-02/audit-wave2 \
  docs/estate-review/2026-11-02-*.md
git status --porcelain | grep -v '^[AM] ' | grep -v '^!!' || true
git commit -q -F - <<'EOF'
docs(audit): estate audit wave 2, every repository except agentbox and VisionClaw

Panel credit 480 to 907 of 1,750 and numbered-decision credit 130 to 181,
no stale claims. estate, nostr-rust-forum, visionflow, dreamlab-ai-website
and ontomark are complete; agentbox and VisionClaw remain.

Fourteen Opus reviewers, source tracing at pinned revisions. Every edited
topic restamped and passing the strict gate with full panel coverage,
checked by the lead per worker. Whole-corpus citation check: 26,400
verified, 0 unverified, 0 unresolvable. Codex spot-check of 12 panels:
1 holds, 2 wrong (both checked and corrected by the lead), 9 cannot-judge
from a weak sampler; recorded as such.

Lead-confirmed findings in remediation/audit-wave2-2026-11-02.json, chief
among them W2-01: agentbox forwards agent events without the JSON-RPC
envelope, so VisionClaw rejects every frame.

Co-Authored-By: jjohare <github@thedreamlab.uk>
EOF
git log -1 --format='%h %s'
cd "$P"
test "$(git branch --show-current)" = main
git merge-tree --write-tree main audit/wave2 >/dev/null
git merge --no-ff -q audit/wave2 -m "merge: estate audit wave 2

Co-Authored-By: jjohare <github@thedreamlab.uk>"
git log -1 --format='%h %s'
node scripts/diagram-index-gen.cjs docs/diagrams --check >/dev/null 2>&1 && echo "main corpus check ok"
