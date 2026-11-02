#!/usr/bin/env bash
# Commit audit wave 3 on audit/wave3, then merge into VisionFlow main.
# Usage: commit-wave3.sh
set -eu
W=/home/devuser/workspace/wt/VisionFlow-audit-wave3
P=/home/devuser/workspace/VisionFlow
cd "$W"
node scripts/diagram-index-gen.cjs docs/diagrams > /dev/null
node scripts/diagram-index-gen.cjs docs/diagrams --check > /dev/null
paths=(
  docs/diagrams/agentbox docs/diagrams/visionclaw
  docs/diagrams/VERIFICATION.md docs/diagrams/README.md docs/diagrams/COVERAGE.md
  docs/estate-review/completion/claims.json
  docs/estate-review/2026-11-02-audit-wave3.md
  docs/estate-review/remediation/audit-wave3-2026-11-02.json
  docs/estate-review/evidence/2026-11-02/audit-wave3
  scripts/estate-review-ledger/component_coverage.py
)
git add -- "${paths[@]}"
git add -- docs/estate-review/2026-11-02-w3-*.md
left=$(git status --porcelain | grep -v '^[AMR] ' | grep -v 'docs/diagrams/rendered/' || true)
if [ -n "$left" ]; then echo "UNSTAGED:"; echo "$left"; exit 1; fi
git commit -q -F - <<'MSG'
docs(audit): estate audit wave 3, agentbox and VisionClaw; ledger complete

Assesses every agentbox and VisionClaw diagram panel (859) and numbered
decision ADR (294 claims, 10 imported copies classified as not adopted),
restamping 77 topics to agentbox c57d9c0 and VisionClaw 86cdd76. Every
topic passes the strict generator gate and each claim covers every
rendered panel.

With this wave the completion ledger has a recorded assessment for all
1,750 panels and all 448 numbered decision ADRs, with no stale claim. The
evidence class is source tracing at pinned revisions; nothing was run.

The lead corrected four panels after cross-topic and cross-vendor checks
(AB-17.1, VC-34.9 twice, AB-13.4). The Codex spot-check of 12 panels found
2 wrong, both confirmed and fixed, and could not judge 10 from the line
windows supplied, so it is recorded as partial.

Findings are queued in remediation/audit-wave3-2026-11-02.json; four are
lead-verified, including a relay consumer that admits every signed author
on an empty allowlist and saved API keys that can never be read back.

Co-Authored-By: jjohare <github@thedreamlab.uk>
MSG
git log -1 --format='%h %s'
cd "$P"
git merge --no-ff -q audit/wave3 -m "merge: estate audit wave 3, ledger complete

Co-Authored-By: jjohare <github@thedreamlab.uk>"
git log -1 --format='%h %s'
node scripts/diagram-index-gen.cjs docs/diagrams --check > /dev/null && echo "main corpus check ok"
