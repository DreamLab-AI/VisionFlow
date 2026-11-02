#!/usr/bin/env bash
# Cross-vendor audit of the round-2 production code (never independently reviewed). Usage: codex-audit-r2.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
B=$SP/audit-r2-bundle.txt
M=$W/crates/solid-pod-rs-forge/src/mirror.rs
T=$(grep -n '^mod tests' "$M" | head -1 | cut -d: -f1)
{
  echo "=== BRIEF ==="; cat "$EV/review-a2-brief.md"
  echo "=== ROUND-2 REPORT ==="; cat "$EV/fix-a2/fix-a2-report.md"
  echo "=== ROUND-2 INTERDIFF ==="; cat "$EV/fix-a2-interdiff.patch"
  echo "=== CURRENT mirror.rs PRODUCTION CODE (lines 1..$((T-1)), after rounds 3-4) ==="; head -n $((T-1)) "$M"
  echo "=== CHANGELOG.md (Unreleased section) ==="; awk '/^## /{n++} n<=2' "$W/CHANGELOG.md"
} > "$B"
wc -c "$B"
cd "$SP" || exit 2
timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - <<EOF > "$EV/audit-r2-codex.txt" 2>&1
You are the cross-vendor evidence auditor for a Rust security fix. Use no tools; everything you need is below. Answer in UK English, outcome first.

The round-2 production code below has never been independently reviewed. Answer every question in the BRIEF section with file:line taken from the current mirror.rs. Then list anything else you find, ranked by severity. Say plainly if nothing material remains.

$(cat "$B")
EOF
echo "exit=$?" >> "$EV/audit-r2-codex.txt"
