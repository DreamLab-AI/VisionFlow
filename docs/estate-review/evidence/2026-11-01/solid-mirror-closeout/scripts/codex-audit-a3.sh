#!/usr/bin/env bash
# Cross-vendor audit of the round-3 interdiff. Usage: codex-audit-a3.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
B=$SP/audit-a3-bundle.txt
{
  echo "=== ROUND-3 BRIEF ==="; cat "$EV/fix-a3-brief.md"
  echo "=== ROUND-3 REPORT ==="; cat "$EV/fix-a3/fix-a3-report.md"
  echo "=== YOUR ROUND-2 AUDIT (for reference) ==="; cat "$EV/audit-a2-codex.txt" 2>/dev/null
  echo "=== ROUND-3 INTERDIFF ==="; cat "$EV/fix-a3/fix-a3-interdiff.patch"
} > "$B"
wc -c "$B"
cd "$SP" || exit 2
timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - <<EOF > "$EV/audit-a3-codex.txt" 2>&1
You are the cross-vendor evidence auditor for a Rust security fix. Use no tools; everything you need is below. Answer in UK English, outcome first.

Round 3 claims to fix your round-2 High 1 (a compare/PR fetch injecting a caller's objects into a mirror's served store) and High 2 (mark_clean_store stamping a store holding unreachable objects), plus a sweep-age fix and two test tightenings.

For each of High 1 and High 2 give a verdict: FIXED, PARTLY FIXED, or NOT FIXED, with file:line from the interdiff. Specifically check:
1. Is there any remaining path, in any file, that writes objects or refs into a repository that has a mirror record?
2. Is the reachability check sound: can an object listed by --batch-all-objects escape it (packed or loose, promisor, alternates, or a check that fails open)?
3. Does the new 409 refusal fire before any lock, fetch or write?
4. Is any new regression introduced?
Then list anything else you find, ranked by severity.

$(cat "$B")
EOF
echo "exit=$?" >> "$EV/audit-a3-codex.txt"
