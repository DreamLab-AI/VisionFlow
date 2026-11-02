#!/usr/bin/env bash
# Cross-vendor audit of round 6 (the final Commit A round). Usage: codex-audit-a6.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
M=$W/crates/solid-pod-rs-forge/src/mirror.rs
B=$SP/audit-a6-bundle.txt
T=$(grep -n '^mod tests' "$M" | head -1 | cut -d: -f1)
{
  echo "=== ROUND-6 BRIEF ==="; cat "$EV/fix-a6-brief.md"
  echo "=== ROUND-6 REPORT ==="; cat "$EV/fix-a6/fix-a6-report.md"
  echo "=== ROUND-6 INTERDIFF ==="; cat "$EV/fix-a6/fix-a6-interdiff.patch"
  echo "=== CURRENT mirror.rs PRODUCTION CODE (lines 1..$((T-1))) ==="; head -n $((T-1)) "$M"
  echo "=== YOUR ROUND-5 ANSWER ==="; tail -n +$(grep -n "^codex$" "$EV/audit-a5-codex.txt" | tail -1 | cut -d: -f1) "$EV/audit-a5-codex.txt"
  echo "=== CLAUDE ROUND-5 SECURITY REVIEW ==="; cat "$EV/review-a5-security.md"
} > "$B"
wc -c "$B"
cd "$SP" || exit 2
timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - <<EOF > "$EV/audit-a6-codex.txt" 2>&1
You are the cross-vendor evidence auditor for a Rust security fix. Use no tools; everything you need is below. Answer in UK English, outcome first.

Round 6 is the final narrow round for Commit A. It answers your round-5 findings and the Claude review's S2-S7 and I4-I5. For each brief item, say FIXED, PARTLY FIXED or NOT FIXED, with file:line from the current mirror.rs. Then say whether round 6 introduced any regression, and above all whether any path now FAILS OPEN on confidentiality, meaning a store holding legacy held objects gets marked or served. List anything else ranked by severity, and say plainly if nothing material remains for Commit A.

$(cat "$B")
EOF
echo "exit=$?" >> "$EV/audit-a6-codex.txt"
